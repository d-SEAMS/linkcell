//! Cutoff pair list with integer cell shifts.
//!
//! [`knearest`](crate::knearest) unique-indexes the neighbour and
//! drops the image. This walk keeps every atom-image pair whose
//! squared distance is strictly below `cutoff²`, including periodic
//! self-images. Each unordered pair is tested once. A full list
//! writes both `(i, j, S)` and `(j, i, -S)`. Displacement is
//! `q - p + lattice_shift(S)`.

use crate::bins::{self, axis_gap, Mesh};
use crate::cell::Cell;
use crate::Error;

const MAX_IMAGES: i64 = 1 << 24;

/// One atom-image pair inside the cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    /// Source index.
    pub i: usize,
    /// Target index.
    pub j: usize,
    /// Integer cell shift applied to the target (`S` in vesin/tonari).
    pub shift: [i32; 3],
    /// Squared distance after the shift.
    pub dist2: f64,
}

fn keep_half(i: usize, j: usize, shift: [i32; 3]) -> bool {
    if i != j {
        return i < j;
    }
    for s in shift {
        if s != 0 {
            return s < 0;
        }
    }
    true
}

/// Pairs with `dist2 < cutoff²`, one row per atom-image.
///
/// `half` keeps the canonical side of `(i, j, S)` vs `(j, i, -S)`.
/// Zero-shift self pairs are omitted. Periodic self-images stay.
/// Image count above 2^24 is [`Error::TooManyImages`].
pub fn pairs_within(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    half: bool,
) -> Result<Vec<Pair>, Error> {
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(Error::BadCutoff);
    }
    if xyz.is_empty() {
        return Err(Error::Empty);
    }
    let n = xyz.len();
    if n > u32::MAX as usize {
        return Err(Error::Overflow);
    }
    if let Some(m) = mask {
        if m.len() != n {
            return Err(Error::MaskLen);
        }
    }
    let active: Vec<usize> = (0..n)
        .filter(|&i| mask.map(|m| m[i]).unwrap_or(true))
        .collect();
    if active.is_empty() {
        return Ok(Vec::new());
    }

    let w = simbox.widths();
    let mut image_count: i64 = 1;
    let mut repeats = [0i32; 3];
    for a in 0..3 {
        let r = (cutoff / w[a]).ceil();
        if !r.is_finite() || r > i32::MAX as f64 {
            return Err(Error::TooManyImages);
        }
        repeats[a] = r as i32;
        let factor = 2 * i64::from(repeats[a]) + 1;
        if image_count > MAX_IMAGES / factor {
            return Err(Error::TooManyImages);
        }
        image_count *= factor;
    }

    let edge = bins::target_edge(simbox, cell_hint, cutoff);
    let mesh = Mesh::build(xyz, simbox, Some(&active), edge)?;
    let cell_min = (mesh.widths[0] / f64::from(mesh.nx))
        .min(mesh.widths[1] / f64::from(mesh.ny))
        .min(mesh.widths[2] / f64::from(mesh.nz));
    let reach_cut = (cutoff / cell_min).ceil();
    if !reach_cut.is_finite() || reach_cut > i32::MAX as f64 {
        return Err(Error::TooManyImages);
    }
    let max_reach = (reach_cut as i32)
        .max(repeats[0] * mesh.nx)
        .max(repeats[1] * mesh.ny)
        .max(repeats[2] * mesh.nz)
        .max(1);
    let cut2 = cutoff * cutoff;
    // One reach for every atom: the shorter gap at either edge of a bin.
    // The box is symmetric, so each unordered pair is visited from one
    // side and written out in both shift directions when `half` is off.
    let reach = uniform_reach([mesh.nx, mesh.ny, mesh.nz], mesh.widths, cut2, max_reach);
    let nslot = mesh.occupants.len();
    let mut coords = Coords {
        x: vec![0.0; nslot],
        y: vec![0.0; nslot],
        z: vec![0.0; nslot],
    };
    for (slot, &i) in mesh.occupants.iter().enumerate() {
        let p = mesh.folded[i];
        coords.x[slot] = p[0];
        coords.y[slot] = p[1];
        coords.z[slot] = p[2];
    }
    let partners = build_partners(&mesh, simbox, reach, cut2);
    let walk = Walk {
        mesh: &mesh,
        coords: &coords,
        cut2,
        half,
        partners: &partners,
    };
    Ok(walk.collect())
}

struct Partner {
    lo: usize,
    hi: usize,
    shift_s: [i32; 3],
    shift: [f64; 3],
}

struct PartnerList {
    off: Vec<usize>,
    items: Vec<Partner>,
}

fn bin_gap(src: i32, dst: i32, width: f64) -> f64 {
    let lo_s = f64::from(src) * width;
    let hi_s = f64::from(src + 1) * width;
    let lo_d = f64::from(dst) * width;
    let hi_d = f64::from(dst + 1) * width;
    if hi_s <= lo_d {
        lo_d - hi_s
    } else if hi_d <= lo_s {
        lo_s - hi_d
    } else {
        0.0
    }
}

fn bins_too_far(ortho: bool, src: [i32; 3], dst: [i32; 3], width: [f64; 3], cut2: f64) -> bool {
    let gap = [
        bin_gap(src[0], dst[0], width[0]),
        bin_gap(src[1], dst[1], width[1]),
        bin_gap(src[2], dst[2], width[2]),
    ];
    let bound = if ortho {
        let d = (gap[0] * gap[0] + gap[1] * gap[1] + gap[2] * gap[2]).sqrt();
        let d = bins::certify(d);
        d * d
    } else {
        let d = bins::certify(gap[0].max(gap[1]).max(gap[2]));
        d * d
    };
    bound >= cut2
}

fn build_partners(mesh: &Mesh, simbox: &Cell, reach: [i32; 3], cut2: f64) -> PartnerList {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    let ncell = (nx as usize) * (ny as usize) * (nz as usize);
    let width = [
        mesh.widths[0] / f64::from(nx),
        mesh.widths[1] / f64::from(ny),
        mesh.widths[2] / f64::from(nz),
    ];
    let ortho = simbox.is_ortho();
    let [rx, ry, rz] = reach;
    let mut off = vec![0usize; ncell + 1];
    let mut items = Vec::new();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let cell = ((iz * ny + iy) * nx + ix) as usize;
                let src = [ix, iy, iz];
                for dz in -rz..=rz {
                    for dy in -ry..=ry {
                        for dx in -rx..=rx {
                            if !keep_dir(dx, dy, dz) {
                                continue;
                            }
                            let dst = [ix + dx, iy + dy, iz + dz];
                            if bins_too_far(ortho, src, dst, width, cut2) {
                                continue;
                            }
                            let jc = mesh.cell_of(dst[0], dst[1], dst[2]);
                            let lo = mesh.offsets[jc];
                            let hi = mesh.offsets[jc + 1];
                            if lo == hi {
                                continue;
                            }
                            let shift_s = [
                                dst[0].div_euclid(nx),
                                dst[1].div_euclid(ny),
                                dst[2].div_euclid(nz),
                            ];
                            items.push(Partner {
                                lo,
                                hi,
                                shift_s,
                                shift: simbox.lattice_shift(shift_s[0], shift_s[1], shift_s[2]),
                            });
                        }
                    }
                }
                off[cell + 1] = items.len();
            }
        }
    }
    PartnerList { off, items }
}

struct Coords {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

#[cfg(target_arch = "x86_64")]
fn simd_avx() -> bool {
    std::is_x86_feature_detected!("avx")
}

/// `true` for exactly one of `(d)` and `(-d)`. The zero offset is the home cell.
fn keep_dir(dx: i32, dy: i32, dz: i32) -> bool {
    if dx != 0 {
        return dx > 0;
    }
    if dy != 0 {
        return dy > 0;
    }
    if dz != 0 {
        return dz > 0;
    }
    false
}

fn uniform_reach(nbin: [i32; 3], widths: [f64; 3], cut2: f64, max_reach: i32) -> [i32; 3] {
    let mut reach = [1i32; 3];
    loop {
        let mut grew = false;
        for a in 0..3 {
            let n = nbin[a];
            let w = widths[a];
            let nf = f64::from(n);
            let s_hi = (1.0 - 1.0e-12) / nf;
            let gap = bins::certify(
                axis_gap(0.0, 0, reach[a], n, w).min(axis_gap(s_hi, 0, reach[a], n, w)),
            );
            let bound = if gap > 0.0 && gap.is_finite() {
                gap * gap
            } else {
                0.0
            };
            if bound < cut2 && reach[a] < max_reach {
                reach[a] += 1;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    reach
}

struct Walk<'a> {
    mesh: &'a Mesh,
    coords: &'a Coords,
    cut2: f64,
    half: bool,
    partners: &'a PartnerList,
}

struct Block {
    i_lo: usize,
    i_hi: usize,
    j_lo: usize,
    j_hi: usize,
    shift_s: [i32; 3],
    shift: [f64; 3],
}

#[derive(Clone, Copy)]
struct Compact {
    i: u32,
    j: u32,
    shift: [i32; 3],
    d2: f64,
}

impl Walk<'_> {
    fn record(&self, found: &mut Vec<Pair>, i: usize, j: usize, shift: [i32; 3], dist2: f64) {
        let neg = [-shift[0], -shift[1], -shift[2]];
        if !self.half {
            found.push(Pair { i, j, shift, dist2 });
            found.push(Pair {
                i: j,
                j: i,
                shift: neg,
                dist2,
            });
            return;
        }
        if keep_half(i, j, shift) {
            found.push(Pair { i, j, shift, dist2 });
        } else {
            found.push(Pair {
                i: j,
                j: i,
                shift: neg,
                dist2,
            });
        }
    }

    fn scan_block(&self, found: &mut Vec<Pair>, block: &Block) {
        if block.i_lo >= block.i_hi || block.j_lo >= block.j_hi {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if block.j_hi - block.j_lo >= 4 && simd_avx() {
            unsafe {
                self.scan_block_avx(found, block);
            }
            return;
        }
        let sx = block.shift[0];
        let sy = block.shift[1];
        let sz = block.shift[2];
        for s in block.i_lo..block.i_hi {
            let i = self.mesh.occupants[s];
            let px = self.coords.x[s] - sx;
            let py = self.coords.y[s] - sy;
            let pz = self.coords.z[s] - sz;
            for slot in block.j_lo..block.j_hi {
                let dx = self.coords.x[slot] - px;
                let dy = self.coords.y[slot] - py;
                let dz = self.coords.z[slot] - pz;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < self.cut2 {
                    let ju = self.mesh.occupants[slot];
                    if ju == i && block.shift_s == [0, 0, 0] {
                        continue;
                    }
                    self.write_staged(
                        found,
                        &[Compact {
                            i: i as u32,
                            j: ju as u32,
                            shift: block.shift_s,
                            d2,
                        }],
                    );
                }
            }
        }
    }

    /// # Safety
    /// `block` ranges index `coords` and `occupants`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn scan_block_avx(&self, found: &mut Vec<Pair>, block: &Block) {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_cmp_pd, _mm256_loadu_pd, _mm256_movemask_pd, _mm256_mul_pd,
            _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd, _CMP_LT_OQ,
        };
        let sx = block.shift[0];
        let sy = block.shift[1];
        let sz = block.shift[2];
        let cut = _mm256_set1_pd(self.cut2);
        let xs = self.coords.x.as_ptr();
        let ys = self.coords.y.as_ptr();
        let zs = self.coords.z.as_ptr();
        let ids = self.mesh.occupants.as_ptr();
        let end = block.j_lo + ((block.j_hi - block.j_lo) & !3);
        let shift_s = block.shift_s;
        let skip_self = shift_s == [0, 0, 0];
        let mut staged = [Compact {
            i: 0,
            j: 0,
            shift: [0, 0, 0],
            d2: 0.0,
        }; 64];
        for s in block.i_lo..block.i_hi {
            let i = *ids.add(s);
            let iu = i as u32;
            let bx = _mm256_set1_pd(*xs.add(s) - sx);
            let by = _mm256_set1_pd(*ys.add(s) - sy);
            let bz = _mm256_set1_pd(*zs.add(s) - sz);
            let mut n = 0usize;
            let mut slot = block.j_lo;
            while slot < end {
                let dx = _mm256_sub_pd(_mm256_loadu_pd(xs.add(slot)), bx);
                let dy = _mm256_sub_pd(_mm256_loadu_pd(ys.add(slot)), by);
                let dz = _mm256_sub_pd(_mm256_loadu_pd(zs.add(slot)), bz);
                let d2v = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                    _mm256_mul_pd(dz, dz),
                );
                let bits = _mm256_movemask_pd(_mm256_cmp_pd(d2v, cut, _CMP_LT_OQ));
                if bits != 0 {
                    let mut lane = [0.0f64; 4];
                    _mm256_storeu_pd(lane.as_mut_ptr(), d2v);
                    if bits & 1 != 0 {
                        let j = *ids.add(slot);
                        if !(skip_self && j == i) {
                            staged[n] = Compact {
                                i: iu,
                                j: j as u32,
                                shift: shift_s,
                                d2: lane[0],
                            };
                            n += 1;
                        }
                    }
                    if bits & 2 != 0 {
                        let j = *ids.add(slot + 1);
                        if !(skip_self && j == i) {
                            staged[n] = Compact {
                                i: iu,
                                j: j as u32,
                                shift: shift_s,
                                d2: lane[1],
                            };
                            n += 1;
                        }
                    }
                    if bits & 4 != 0 {
                        let j = *ids.add(slot + 2);
                        if !(skip_self && j == i) {
                            staged[n] = Compact {
                                i: iu,
                                j: j as u32,
                                shift: shift_s,
                                d2: lane[2],
                            };
                            n += 1;
                        }
                    }
                    if bits & 8 != 0 {
                        let j = *ids.add(slot + 3);
                        if !(skip_self && j == i) {
                            staged[n] = Compact {
                                i: iu,
                                j: j as u32,
                                shift: shift_s,
                                d2: lane[3],
                            };
                            n += 1;
                        }
                    }
                    if n > staged.len() - 4 {
                        self.write_staged(found, &staged[..n]);
                        n = 0;
                    }
                }
                slot += 4;
            }
            for slot in end..block.j_hi {
                let dx = *xs.add(slot) - (*xs.add(s) - sx);
                let dy = *ys.add(slot) - (*ys.add(s) - sy);
                let dz = *zs.add(slot) - (*zs.add(s) - sz);
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < self.cut2 {
                    let j = *ids.add(slot);
                    if !(skip_self && j == i) && n < staged.len() {
                        staged[n] = Compact {
                            i: iu,
                            j: j as u32,
                            shift: shift_s,
                            d2,
                        };
                        n += 1;
                    }
                }
            }
            if n > 0 {
                self.write_staged(found, &staged[..n]);
            }
        }
    }

    fn append_cell(&self, found: &mut Vec<Pair>, cell: usize) {
        let lo = self.mesh.offsets[cell];
        let hi = self.mesh.offsets[cell + 1];
        if lo == hi {
            return;
        }
        for s in lo..hi {
            if s + 1 >= hi {
                break;
            }
            self.scan_block(
                found,
                &Block {
                    i_lo: s,
                    i_hi: s + 1,
                    j_lo: s + 1,
                    j_hi: hi,
                    shift_s: [0, 0, 0],
                    shift: [0.0; 3],
                },
            );
        }
        let p0 = self.partners.off[cell];
        let p1 = self.partners.off[cell + 1];
        for partner in &self.partners.items[p0..p1] {
            self.scan_block(
                found,
                &Block {
                    i_lo: lo,
                    i_hi: hi,
                    j_lo: partner.lo,
                    j_hi: partner.hi,
                    shift_s: partner.shift_s,
                    shift: partner.shift,
                },
            );
        }
    }

    fn guess_rows(&self, n_src: usize) -> usize {
        let w = self.mesh.widths;
        let volume = (w[0] * w[1] * w[2]).max(1.0e-30);
        let radius = self.cut2.sqrt();
        let shell = 4.1887902047863905 * radius * radius * radius;
        let n = self.mesh.folded.len().max(1) as f64;
        let neighbors = (n * shell / volume).ceil().max(1.0);
        let rows = neighbors * if self.half { 0.5 } else { 1.0 };
        ((n_src as f64) * rows * 1.25) as usize
    }

    fn write_staged(&self, found: &mut Vec<Pair>, hits: &[Compact]) {
        if hits.is_empty() {
            return;
        }
        if !self.half {
            found.reserve(hits.len() * 2);
            let mut len = found.len();
            let ptr: *mut Pair = found.as_mut_ptr();
            for hit in hits {
                let i = hit.i as usize;
                let j = hit.j as usize;
                let neg = [-hit.shift[0], -hit.shift[1], -hit.shift[2]];
                unsafe {
                    ptr.add(len).write(Pair {
                        i,
                        j,
                        shift: hit.shift,
                        dist2: hit.d2,
                    });
                    ptr.add(len + 1).write(Pair {
                        i: j,
                        j: i,
                        shift: neg,
                        dist2: hit.d2,
                    });
                }
                len += 2;
            }
            unsafe {
                found.set_len(len);
            }
            return;
        }
        for hit in hits {
            self.record(found, hit.i as usize, hit.j as usize, hit.shift, hit.d2);
        }
    }

    fn collect(&self) -> Vec<Pair> {
        let ncell = self.mesh.offsets.len() - 1;
        #[cfg(feature = "parallel")]
        let offsets = &self.mesh.offsets;
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let threads = rayon::current_num_threads().max(1);
            let chunk = (ncell / threads).max(1);
            let parts: Vec<Vec<Pair>> = (0..ncell)
                .step_by(chunk)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|start| {
                    let _timer = crate::pop::JobTimer::new();
                    let end = (start + chunk).min(ncell);
                    let n_src = offsets[end] - offsets[start];
                    let mut found = Vec::with_capacity(self.guess_rows(n_src));
                    for cell in start..end {
                        self.append_cell(&mut found, cell);
                    }
                    found
                })
                .collect();
            let mut pairs = Vec::with_capacity(parts.iter().map(Vec::len).sum());
            for part in parts {
                pairs.extend(part);
            }
            pairs
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _timer = crate::pop::JobTimer::new();
            let mut found = Vec::with_capacity(self.guess_rows(self.mesh.occupants.len()));
            for cell in 0..ncell {
                self.append_cell(&mut found, cell);
            }
            found
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cell;

    #[test]
    fn wrap_pair_keeps_shift() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
        let pairs = pairs_within(&xyz, &sim, 1.0, None, None, false).unwrap();
        assert_eq!(pairs.len(), 2);
        let a = pairs.iter().find(|p| p.i == 0 && p.j == 1).unwrap();
        assert_eq!(a.shift, [-1, 0, 0]);
        assert!((a.dist2 - 0.64).abs() < 1e-12);
    }

    #[test]
    fn half_list_is_canonical() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
        let pairs = pairs_within(&xyz, &sim, 1.0, None, None, true).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].i, 0);
        assert_eq!(pairs[0].j, 1);
    }

    #[test]
    fn sheared_cutoff_matches_shift_scan() {
        let sim = Cell::from_vectors(
            [4.0, 0.0, 0.0],
            [1.5, 3.0, 0.0],
            [0.2, -0.4, 3.5],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        let xyz = [
            sim.cartesian([0.05, 0.10, 0.20]),
            sim.cartesian([0.90, 0.15, 0.80]),
            sim.cartesian([0.40, 0.70, 0.30]),
            sim.cartesian([0.12, 0.85, 0.05]),
        ];
        let cutoff = 2.5;
        let got = pairs_within(&xyz, &sim, cutoff, None, Some(1.0), false).unwrap();
        let mut want = Vec::new();
        let cut2 = cutoff * cutoff;
        for i in 0..xyz.len() {
            for j in 0..xyz.len() {
                for na in -2..=2 {
                    for nb in -2..=2 {
                        for nc in -2..=2 {
                            if i == j && na == 0 && nb == 0 && nc == 0 {
                                continue;
                            }
                            let shift = sim.lattice_shift(na, nb, nc);
                            let pi = sim.cartesian(sim.fractional(xyz[i]));
                            let pj = sim.cartesian(sim.fractional(xyz[j]));
                            let d2 = sim.dist2_shifted(pi, pj, shift);
                            if d2 < cut2 {
                                want.push((i, j, [na, nb, nc]));
                            }
                        }
                    }
                }
            }
        }
        let mut got_key: Vec<_> = got.iter().map(|p| (p.i, p.j, p.shift)).collect();
        got_key.sort();
        want.sort();
        assert_eq!(got_key, want);
    }

    #[test]
    fn dense_cube_matches_shift_scan() {
        let sim = Cell::ortho(6.0, 6.0, 6.0).unwrap();
        let mut xyz = Vec::new();
        for iz in 0..5 {
            for iy in 0..5 {
                for ix in 0..5 {
                    xyz.push([
                        (ix as f64 + 0.5) * 1.2,
                        (iy as f64 + 0.5) * 1.2,
                        (iz as f64 + 0.5) * 1.2,
                    ]);
                }
            }
        }
        let cutoff = 2.0;
        let got = pairs_within(&xyz, &sim, cutoff, None, None, false).unwrap();
        let mut keys: Vec<_> = got.iter().map(|p| (p.i, p.j, p.shift)).collect();
        keys.sort();
        let cut2 = cutoff * cutoff;
        let mut want = Vec::new();
        for i in 0..xyz.len() {
            for j in 0..xyz.len() {
                for na in -1..=1 {
                    for nb in -1..=1 {
                        for nc in -1..=1 {
                            if i == j && na == 0 && nb == 0 && nc == 0 {
                                continue;
                            }
                            let shift = [na, nb, nc];
                            let pi = sim.cartesian(sim.fractional(xyz[i]));
                            let pj = sim.cartesian(sim.fractional(xyz[j]));
                            let d2 = sim.dist2_shifted(pi, pj, sim.lattice_shift(na, nb, nc));
                            if d2 < cut2 {
                                want.push((i, j, shift));
                            }
                        }
                    }
                }
            }
        }
        want.sort();
        assert_eq!(keys, want);
    }

    #[test]
    fn bad_cutoff() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.0, 0.0, 0.0]];
        assert_eq!(
            pairs_within(&xyz, &sim, 0.0, None, None, false).unwrap_err(),
            Error::BadCutoff
        );
    }
}
