//! AVX2 and NEON pair tiles.
//!
//! Each kernel screens with the expanded distance and the same margin as
//! the AVX-512 tile. A lane inside the margin is kept. A lane in the
//! margin band is decided by the direct formula. The stored squared
//! distance is always that direct formula, with a separate multiply and
//! add. FMA stays inside the screen.
//!
//! An 8-wide target block is two AVX2 vectors or four NEON vectors, low
//! lanes first, so the hit order matches the scalar walk and the AVX-512 tile.

use super::{Block, Scratch, TileCols, CULL_RUNS};

/// `(dx * dx + dy * dy) + dz * dz`, the scalar tile's association.
#[inline(always)]
fn direct_scalar(dx: f64, dy: f64, dz: f64) -> f64 {
    (dx * dx + dy * dy) + dz * dz
}

/// # Safety
/// `block` indexes every column of `c`. `scratch` has room for every
/// pair in the block plus [`super::STORE_SLACK`] slots.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
pub(super) unsafe fn avx2_scan(
    c: TileCols,
    cut2: f64,
    margin: f64,
    block: &Block,
    scratch: &mut Scratch,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_cmp_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_max_pd,
        _mm256_movemask_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
        _mm256_sub_pd, _CMP_LT_OQ,
    };
    debug_assert!(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"));
    let atom_p = scratch.atom.as_mut_ptr();
    let js_p = scratch.js.as_mut_ptr();
    let d2_p = scratch.d2.as_mut_ptr();
    let mut n = scratch.js.len();
    let [sx, sy, sz] = block.shift;
    let [ox, oy, oz] = block.delta;
    let cut_lo = cut2 - margin;
    let cut_hi = cut2 + margin;
    let j_hi = block.j_hi;
    let nvec = (j_hi - block.j_lo + 7) / 8;
    let do_cull = !block.tri && c.cull && nvec >= CULL_RUNS;

    macro_rules! push_scalar {
        ($iu:expr, $j:expr, $px:expr, $py:expr, $pz:expr) => {{
            let j = $j;
            let dx = *c.x.add(j) - $px;
            let dy = *c.y.add(j) - $py;
            let dz = *c.z.add(j) - $pz;
            let d2 = direct_scalar(dx, dy, dz);
            if d2 < cut2 {
                atom_p.add(n).write($iu);
                js_p.add(n).write(*c.id.add(j));
                d2_p.add(n).write(d2);
                n += 1;
            }
        }};
    }
    // Four targets at `slot`. The screen is three FMAs. A block that is
    // entirely inside the margin stores the direct squared distance.
    macro_rules! push4 {
        ($s:expr, $slot:expr, $iu:expr, $ax:expr, $ay:expr, $az:expr, $thr:expr, $pp:expr, $px:expr, $py:expr, $pz:expr) => {{
            let slot = $slot;
            let jx = _mm256_loadu_pd(c.rx.add(slot));
            let jy = _mm256_loadu_pd(c.ry.add(slot));
            let jz = _mm256_loadu_pd(c.rz.add(slot));
            let jr = _mm256_loadu_pd(c.r2.add(slot));
            let t = _mm256_fmadd_pd(
                jz,
                $az,
                _mm256_fmadd_pd(jy, $ay, _mm256_fmadd_pd(jx, $ax, jr)),
            );
            let hi = _mm256_movemask_pd(_mm256_cmp_pd(t, _mm256_set1_pd($thr), _CMP_LT_OQ));
            if hi != 0 {
                let exp = _mm256_add_pd(t, _mm256_set1_pd($pp));
                let lo =
                    _mm256_movemask_pd(_mm256_cmp_pd(exp, _mm256_set1_pd(cut_lo), _CMP_LT_OQ)) & hi;
                if lo == hi {
                    let dx = _mm256_sub_pd(_mm256_loadu_pd(c.x.add(slot)), _mm256_set1_pd($px));
                    let dy = _mm256_sub_pd(_mm256_loadu_pd(c.y.add(slot)), _mm256_set1_pd($py));
                    let dz = _mm256_sub_pd(_mm256_loadu_pd(c.z.add(slot)), _mm256_set1_pd($pz));
                    let d2v = _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                        _mm256_mul_pd(dz, dz),
                    );
                    let mut lane = [0.0f64; 4];
                    _mm256_storeu_pd(lane.as_mut_ptr(), d2v);
                    let mut bits = lo as u32;
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(*c.id.add(slot + k));
                        d2_p.add(n).write(lane[k]);
                        n += 1;
                    }
                } else {
                    let mut bits = hi as u32;
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        push_scalar!($iu, slot + k, $px, $py, $pz);
                    }
                }
            }
        }};
    }
    macro_rules! one_block {
        ($slot:expr, $iu:expr, $ax:expr, $ay:expr, $az:expr, $thr:expr, $pp:expr, $px:expr, $py:expr, $pz:expr) => {{
            let slot = $slot;
            let left = (j_hi - slot).min(8);
            let mut off = 0usize;
            while off + 4 <= left {
                push4!(0, slot + off, $iu, $ax, $ay, $az, $thr, $pp, $px, $py, $pz);
                off += 4;
            }
            while off < left {
                push_scalar!($iu, slot + off, $px, $py, $pz);
                off += 1;
            }
        }};
    }

    for s in block.i_lo..block.i_hi {
        let px_rel = *c.rx.add(s) + ox;
        let py_rel = *c.ry.add(s) + oy;
        let pz_rel = *c.rz.add(s) + oz;
        let pp = px_rel * px_rel + py_rel * py_rel + pz_rel * pz_rel;
        let ax = _mm256_set1_pd(-2.0 * px_rel);
        let ay = _mm256_set1_pd(-2.0 * py_rel);
        let az = _mm256_set1_pd(-2.0 * pz_rel);
        let thr = cut_hi - pp;
        let px = *c.x.add(s) - sx;
        let py = *c.y.add(s) - sy;
        let pz = *c.z.add(s) - sz;
        let iu = *c.id.add(s);
        if do_cull {
            // Four box-runs at a time. A run that can hold a pair is
            // visited at once, in slot order. Nothing is buffered.
            let g = block.ig + (s - block.i_lo) / 4;
            let blo = [
                _mm256_set1_pd(*c.gb[0].add(g) + ox),
                _mm256_set1_pd(*c.gb[1].add(g) + oy),
                _mm256_set1_pd(*c.gb[2].add(g) + oz),
            ];
            let bhi = [
                _mm256_set1_pd(*c.gb[3].add(g) + ox),
                _mm256_set1_pd(*c.gb[4].add(g) + oy),
                _mm256_set1_pd(*c.gb[5].add(g) + oz),
            ];
            let zero = _mm256_setzero_pd();
            let cut = _mm256_set1_pd(cut2 + 2.0 * margin);
            let mut m0 = 0usize;
            while m0 < nvec {
                let v = block.jv + m0;
                let lanes = (nvec - m0).min(4);
                let mut box_d2 = zero;
                for axis in 0..3 {
                    let gap = _mm256_max_pd(
                        _mm256_max_pd(
                            _mm256_sub_pd(_mm256_loadu_pd(c.vb[axis].add(v)), bhi[axis]),
                            _mm256_sub_pd(blo[axis], _mm256_loadu_pd(c.vb[3 + axis].add(v))),
                        ),
                        zero,
                    );
                    box_d2 = _mm256_fmadd_pd(gap, gap, box_d2);
                }
                let mut near = _mm256_movemask_pd(_mm256_cmp_pd(box_d2, cut, _CMP_LT_OQ)) as u32;
                let valid = (1u32 << lanes) - 1;
                near &= valid;
                #[cfg(test)]
                super::CULL_SKIPS.fetch_add(
                    (valid & !near).count_ones() as usize,
                    std::sync::atomic::Ordering::Relaxed,
                );
                while near != 0 {
                    let lane = near.trailing_zeros() as usize;
                    near &= near - 1;
                    let slot = block.j_lo + 8 * (m0 + lane);
                    one_block!(slot, iu, ax, ay, az, thr, pp, px, py, pz);
                }
                m0 += 4;
            }
        } else {
            let start = if block.tri { s + 1 } else { block.j_lo };
            let mut slot = start;
            while slot < j_hi {
                one_block!(slot, iu, ax, ay, az, thr, pp, px, py, pz);
                slot += 8;
            }
        }
    }
    scratch.finish(n);
}

/// # Safety
/// `block` indexes every column of `c`. `scratch` has room for every
/// pair in the block plus [`super::STORE_SLACK`] slots.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
pub(super) unsafe fn neon_scan(
    c: TileCols,
    cut2: f64,
    margin: f64,
    block: &Block,
    scratch: &mut Scratch,
) {
    use std::arch::aarch64::{
        vaddq_f64, vcltq_f64, vdupq_n_f64, vfmaq_f64, vgetq_lane_f64, vgetq_lane_u64, vld1q_f64,
        vmulq_f64, vsubq_f64,
    };
    let atom_p = scratch.atom.as_mut_ptr();
    let js_p = scratch.js.as_mut_ptr();
    let d2_p = scratch.d2.as_mut_ptr();
    let mut n = scratch.js.len();
    let [sx, sy, sz] = block.shift;
    let [ox, oy, oz] = block.delta;
    let cut_lo = cut2 - margin;
    let cut_hi = cut2 + margin;
    let j_hi = block.j_hi;
    let nvec = (j_hi - block.j_lo + 7) / 8;
    let do_cull = !block.tri && c.cull && nvec >= CULL_RUNS;

    macro_rules! push_scalar {
        ($iu:expr, $j:expr, $px:expr, $py:expr, $pz:expr) => {{
            let j = $j;
            let dx = *c.x.add(j) - $px;
            let dy = *c.y.add(j) - $py;
            let dz = *c.z.add(j) - $pz;
            let d2 = direct_scalar(dx, dy, dz);
            if d2 < cut2 {
                atom_p.add(n).write($iu);
                js_p.add(n).write(*c.id.add(j));
                d2_p.add(n).write(d2);
                n += 1;
            }
        }};
    }
    macro_rules! push2 {
        ($slot:expr, $iu:expr, $ax:expr, $ay:expr, $az:expr, $thr:expr, $ppv:expr, $bx:expr, $by:expr, $bz:expr, $px:expr, $py:expr, $pz:expr) => {{
            let slot = $slot;
            let jx = vld1q_f64(c.rx.add(slot));
            let jy = vld1q_f64(c.ry.add(slot));
            let jz = vld1q_f64(c.rz.add(slot));
            let jr = vld1q_f64(c.r2.add(slot));
            let t = vfmaq_f64(vfmaq_f64(vfmaq_f64(jr, jx, $ax), jy, $ay), jz, $az);
            let hi = vcltq_f64(t, $thr);
            let h0 = vgetq_lane_u64(hi, 0) != 0;
            let h1 = vgetq_lane_u64(hi, 1) != 0;
            if h0 || h1 {
                let exp = vaddq_f64(t, $ppv);
                let lo = vcltq_f64(exp, vdupq_n_f64(cut_lo));
                let l0 = h0 && vgetq_lane_u64(lo, 0) != 0;
                let l1 = h1 && vgetq_lane_u64(lo, 1) != 0;
                if h0 == l0 && h1 == l1 {
                    let dx = vsubq_f64(vld1q_f64(c.x.add(slot)), $bx);
                    let dy = vsubq_f64(vld1q_f64(c.y.add(slot)), $by);
                    let dz = vsubq_f64(vld1q_f64(c.z.add(slot)), $bz);
                    let d2v = vaddq_f64(
                        vaddq_f64(vmulq_f64(dx, dx), vmulq_f64(dy, dy)),
                        vmulq_f64(dz, dz),
                    );
                    if h0 {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(*c.id.add(slot));
                        d2_p.add(n).write(vgetq_lane_f64(d2v, 0));
                        n += 1;
                    }
                    if h1 {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(*c.id.add(slot + 1));
                        d2_p.add(n).write(vgetq_lane_f64(d2v, 1));
                        n += 1;
                    }
                } else {
                    if h0 {
                        push_scalar!($iu, slot, $px, $py, $pz);
                    }
                    if h1 {
                        push_scalar!($iu, slot + 1, $px, $py, $pz);
                    }
                }
            }
        }};
    }

    for s in block.i_lo..block.i_hi {
        let px_rel = *c.rx.add(s) + ox;
        let py_rel = *c.ry.add(s) + oy;
        let pz_rel = *c.rz.add(s) + oz;
        let pp = px_rel * px_rel + py_rel * py_rel + pz_rel * pz_rel;
        let ax = vdupq_n_f64(-2.0 * px_rel);
        let ay = vdupq_n_f64(-2.0 * py_rel);
        let az = vdupq_n_f64(-2.0 * pz_rel);
        let thr = vdupq_n_f64(cut_hi - pp);
        let ppv = vdupq_n_f64(pp);
        let px = *c.x.add(s) - sx;
        let py = *c.y.add(s) - sy;
        let pz = *c.z.add(s) - sz;
        let iu = *c.id.add(s);
        let bx = vdupq_n_f64(px);
        let by = vdupq_n_f64(py);
        let bz = vdupq_n_f64(pz);
        macro_rules! one_block {
            ($slot:expr) => {{
                let slot = $slot;
                let left = (j_hi - slot).min(8);
                let mut off = 0usize;
                while off + 2 <= left {
                    push2!(slot + off, iu, ax, ay, az, thr, ppv, bx, by, bz, px, py, pz);
                    off += 2;
                }
                if off < left {
                    push_scalar!(iu, slot + off, px, py, pz);
                }
            }};
        }
        if do_cull {
            use std::arch::aarch64::{
                vcltq_f64, vdupq_n_f64, vfmaq_f64, vgetq_lane_u64, vld1q_f64, vmaxq_f64, vsubq_f64,
            };
            let g = block.ig + (s - block.i_lo) / 4;
            let blo = [
                vdupq_n_f64(*c.gb[0].add(g) + ox),
                vdupq_n_f64(*c.gb[1].add(g) + oy),
                vdupq_n_f64(*c.gb[2].add(g) + oz),
            ];
            let bhi = [
                vdupq_n_f64(*c.gb[3].add(g) + ox),
                vdupq_n_f64(*c.gb[4].add(g) + oy),
                vdupq_n_f64(*c.gb[5].add(g) + oz),
            ];
            let zero = vdupq_n_f64(0.0);
            let cut = vdupq_n_f64(cut2 + 2.0 * margin);
            let mut m0 = 0usize;
            while m0 < nvec {
                let v = block.jv + m0;
                let lanes = (nvec - m0).min(2);
                let mut box_d2 = zero;
                for axis in 0..3 {
                    let gap = vmaxq_f64(
                        vmaxq_f64(
                            vsubq_f64(vld1q_f64(c.vb[axis].add(v)), bhi[axis]),
                            vsubq_f64(blo[axis], vld1q_f64(c.vb[3 + axis].add(v))),
                        ),
                        zero,
                    );
                    box_d2 = vfmaq_f64(box_d2, gap, gap);
                }
                let near = vcltq_f64(box_d2, cut);
                let n0 = lanes >= 1 && vgetq_lane_u64(near, 0) != 0;
                let n1 = lanes >= 2 && vgetq_lane_u64(near, 1) != 0;
                #[cfg(test)]
                {
                    let mut skipped = 0usize;
                    if lanes >= 1 && !n0 {
                        skipped += 1;
                    }
                    if lanes >= 2 && !n1 {
                        skipped += 1;
                    }
                    super::CULL_SKIPS.fetch_add(skipped, std::sync::atomic::Ordering::Relaxed);
                }
                if n0 {
                    one_block!(block.j_lo + 8 * m0);
                }
                if n1 {
                    one_block!(block.j_lo + 8 * (m0 + 1));
                }
                m0 += 2;
            }
        } else {
            let start = if block.tri { s + 1 } else { block.j_lo };
            let mut slot = start;
            while slot < j_hi {
                one_block!(slot);
                slot += 8;
            }
        }
    }
    scratch.finish(n);
}
