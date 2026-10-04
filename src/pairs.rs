//! Cutoff pair list with integer cell shifts.
//!
//! [`knearest`](crate::knearest) unique-indexes the neighbour and
//! drops the image. This walk keeps every atom-image pair whose
//! squared distance is strictly below `cutoff²`, including periodic
//! self-images. Each unordered pair is tested once. A full list
//! writes both `(i, j, S)` and `(j, i, -S)`. Displacement is
//! `q - p + lattice_shift(S)`.

use crate::bins::{self, axis_gap, slab_dist2, Mesh};
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
    let walk = Walk {
        mesh: &mesh,
        simbox,
        coords: &coords,
        cut2,
        reach,
        half,
    };
    Ok(walk.collect())
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
    simbox: &'a Cell,
    coords: &'a Coords,
    cut2: f64,
    reach: [i32; 3],
    half: bool,
}

struct Src {
    i: usize,
    pi: [f64; 3],
    shift_s: [i32; 3],
    shift: [f64; 3],
    home: bool,
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

    #[inline(always)]
    fn consider(
        &self,
        found: &mut Vec<Pair>,
        i: usize,
        ju: usize,
        shift_s: [i32; 3],
        dist2: f64,
        home: bool,
    ) {
        if home && ju <= i {
            return;
        }
        if !home && ju == i && shift_s == [0, 0, 0] {
            return;
        }
        if dist2 < self.cut2 {
            self.record(found, i, ju, shift_s, dist2);
        }
    }

    fn scan_cell(&self, found: &mut Vec<Pair>, cell: usize, src: &Src) {
        let lo = self.mesh.offsets[cell];
        let hi = self.mesh.offsets[cell + 1];
        if lo == hi {
            return;
        }
        debug_assert_eq!(hi - lo, self.mesh.slots(cell).len());
        #[cfg(target_arch = "x86_64")]
        if !src.home && simd_avx() {
            unsafe {
                self.scan_avx(found, lo, hi, src);
            }
            return;
        }
        let sx = src.shift[0];
        let sy = src.shift[1];
        let sz = src.shift[2];
        for (k, &ju) in self.mesh.slots(cell).iter().enumerate() {
            let slot = lo + k;
            let dx = self.coords.x[slot] + sx - src.pi[0];
            let dy = self.coords.y[slot] + sy - src.pi[1];
            let dz = self.coords.z[slot] + sz - src.pi[2];
            self.consider(
                found,
                src.i,
                ju,
                src.shift_s,
                dx * dx + dy * dy + dz * dz,
                src.home,
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn scan_avx(&self, found: &mut Vec<Pair>, lo: usize, hi: usize, src: &Src) {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_cmp_pd, _mm256_loadu_pd, _mm256_movemask_pd, _mm256_mul_pd,
            _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd, _CMP_LT_OQ,
        };
        let bx = _mm256_set1_pd(src.pi[0] - src.shift[0]);
        let by = _mm256_set1_pd(src.pi[1] - src.shift[1]);
        let bz = _mm256_set1_pd(src.pi[2] - src.shift[2]);
        let cut = _mm256_set1_pd(self.cut2);
        let xs = self.coords.x.as_ptr();
        let ys = self.coords.y.as_ptr();
        let zs = self.coords.z.as_ptr();
        let end = lo + ((hi - lo) & !3);
        let mut slot = lo;
        while slot < end {
            let dx = _mm256_sub_pd(_mm256_loadu_pd(xs.add(slot)), bx);
            let dy = _mm256_sub_pd(_mm256_loadu_pd(ys.add(slot)), by);
            let dz = _mm256_sub_pd(_mm256_loadu_pd(zs.add(slot)), bz);
            let d2 = _mm256_add_pd(
                _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                _mm256_mul_pd(dz, dz),
            );
            let bits = _mm256_movemask_pd(_mm256_cmp_pd(d2, cut, _CMP_LT_OQ));
            if bits != 0 {
                let mut lane = [0.0f64; 4];
                _mm256_storeu_pd(lane.as_mut_ptr(), d2);
                if bits & 1 != 0 {
                    self.consider(
                        found,
                        src.i,
                        self.mesh.occupants[slot],
                        src.shift_s,
                        lane[0],
                        false,
                    );
                }
                if bits & 2 != 0 {
                    self.consider(
                        found,
                        src.i,
                        self.mesh.occupants[slot + 1],
                        src.shift_s,
                        lane[1],
                        false,
                    );
                }
                if bits & 4 != 0 {
                    self.consider(
                        found,
                        src.i,
                        self.mesh.occupants[slot + 2],
                        src.shift_s,
                        lane[2],
                        false,
                    );
                }
                if bits & 8 != 0 {
                    self.consider(
                        found,
                        src.i,
                        self.mesh.occupants[slot + 3],
                        src.shift_s,
                        lane[3],
                        false,
                    );
                }
            }
            slot += 4;
        }
        let sx = src.shift[0];
        let sy = src.shift[1];
        let sz = src.shift[2];
        for slot in end..hi {
            let dx = self.coords.x[slot] + sx - src.pi[0];
            let dy = self.coords.y[slot] + sy - src.pi[1];
            let dz = self.coords.z[slot] + sz - src.pi[2];
            self.consider(
                found,
                src.i,
                self.mesh.occupants[slot],
                src.shift_s,
                dx * dx + dy * dy + dz * dz,
                false,
            );
        }
    }

    fn append(&self, found: &mut Vec<Pair>, i: usize) {
        let mesh = self.mesh;
        let [ix, iy, iz] = mesh.bin[i];
        let origin = mesh.frac[i];
        let pi = mesh.folded[i];
        let nbin = [mesh.nx, mesh.ny, mesh.nz];
        let home = Src {
            i,
            pi,
            shift_s: [0, 0, 0],
            shift: [0.0; 3],
            home: true,
        };
        self.scan_cell(found, mesh.cell_of(ix, iy, iz), &home);
        let [rx, ry, rz] = self.reach;
        for dz in -rz..=rz {
            for dy in -ry..=ry {
                for dx in -rx..=rx {
                    if !keep_dir(dx, dy, dz) {
                        continue;
                    }
                    let jx = ix + dx;
                    let jy = iy + dy;
                    let jz = iz + dz;
                    if slab_dist2(origin, [jx, jy, jz], nbin, mesh.widths) >= self.cut2 {
                        continue;
                    }
                    let shift_s = [
                        jx.div_euclid(mesh.nx),
                        jy.div_euclid(mesh.ny),
                        jz.div_euclid(mesh.nz),
                    ];
                    let shift = self
                        .simbox
                        .lattice_shift(shift_s[0], shift_s[1], shift_s[2]);
                    let src = Src {
                        i,
                        pi,
                        shift_s,
                        shift,
                        home: false,
                    };
                    self.scan_cell(found, mesh.cell_of(jx, jy, jz), &src);
                }
            }
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

    fn collect(&self) -> Vec<Pair> {
        let ncell = self.mesh.offsets.len() - 1;
        let occupants = &self.mesh.occupants;
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
                        let rows = &occupants[offsets[cell]..offsets[cell + 1]];
                        for &src in rows {
                            self.append(&mut found, src);
                        }
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
            let mut pairs = Vec::with_capacity(self.guess_rows(occupants.len()));
            for cell in 0..ncell {
                let rows = &occupants[offsets[cell]..offsets[cell + 1]];
                for &src in rows {
                    self.append(&mut pairs, src);
                }
            }
            pairs
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
    fn bad_cutoff() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.0, 0.0, 0.0]];
        assert_eq!(
            pairs_within(&xyz, &sim, 0.0, None, None, false).unwrap_err(),
            Error::BadCutoff
        );
    }
}
