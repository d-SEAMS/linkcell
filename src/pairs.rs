//! Cutoff pair list with integer cell shifts.
//!
//! [`knearest`](crate::knearest) unique-indexes the neighbour and
//! drops the image. This walk keeps every atom-image pair whose
//! squared distance is strictly below `cutoff²`, including periodic
//! self-images. Displacement is
//! `q - p + lattice_shift(S)`.

use crate::bins::{self, for_shell, frontier_dist2, slab_dist2, Mesh};
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
    let mesh = Mesh::build(xyz, simbox, &active, edge)?;
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
    let nbin = [mesh.nx, mesh.ny, mesh.nz];
    let widths = mesh.widths;

    let one = |i: usize| -> Vec<Pair> {
        let [ix, iy, iz] = mesh.bin[i];
        let origin = mesh.frac[i];
        let pi = mesh.folded[i];
        let mut found = Vec::new();
        let mut reach = 1i32;
        while reach <= max_reach {
            for_shell(reach, |dx, dy, dz| {
                let jx = ix + dx;
                let jy = iy + dy;
                let jz = iz + dz;
                let lb = slab_dist2(origin, [jx, jy, jz], nbin, widths);
                if lb >= cut2 {
                    return;
                }
                let na = jx.div_euclid(mesh.nx);
                let nb = jy.div_euclid(mesh.ny);
                let nc = jz.div_euclid(mesh.nz);
                let shift_s = [na, nb, nc];
                let shift = simbox.lattice_shift(na, nb, nc);
                for &ju in mesh.slots(mesh.cell_of(jx, jy, jz)) {
                    let zero_self = ju == i && na == 0 && nb == 0 && nc == 0;
                    if zero_self {
                        continue;
                    }
                    let d2 = simbox.dist2_shifted(pi, mesh.folded[ju], shift);
                    if d2 < cut2 && (!half || keep_half(i, ju, shift_s)) {
                        found.push(Pair {
                            i,
                            j: ju,
                            shift: shift_s,
                            dist2: d2,
                        });
                    }
                }
            });
            let bound = frontier_dist2(origin, [ix, iy, iz], reach, nbin, widths);
            if bound >= cut2 {
                break;
            }
            reach += 1;
        }
        found
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let chunks: Vec<Vec<Pair>> = active.par_iter().copied().map(one).collect();
        let mut pairs = Vec::new();
        for chunk in chunks {
            pairs.extend(chunk);
        }
        Ok(pairs)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut pairs = Vec::new();
        for &i in &active {
            pairs.extend(one(i));
        }
        Ok(pairs)
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
