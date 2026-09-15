//! Cutoff pair list with integer cell shifts.
//!
//! [`knearest`](crate::knearest) unique-indexes the neighbour and
//! drops the image. This walk keeps every atom-image pair whose
//! squared distance is strictly below `cutoff²`, including periodic
//! self-images. Displacement is
//! `q - p + lattice_shift(S)`.

use crate::cell::Cell;
use crate::Error;

const MAX_CELLS: i64 = 16_777_216;
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

fn bins_1d(width: f64, edge: f64) -> Result<i32, Error> {
    let n = (width / edge).floor().max(1.0);
    if !n.is_finite() || n > 1_000_000.0 {
        return Err(Error::TooManyCells);
    }
    Ok(n as i32)
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

    let mut edge = cell_hint.unwrap_or(cutoff);
    if !edge.is_finite() || edge <= 0.0 {
        edge = cutoff;
    }
    edge = edge.min(w[0]).min(w[1]).min(w[2]);

    let nx = bins_1d(w[0], edge)?;
    let ny = bins_1d(w[1], edge)?;
    let nz = bins_1d(w[2], edge)?;
    let ncell = (i64::from(nx))
        .checked_mul(i64::from(ny))
        .and_then(|v| v.checked_mul(i64::from(nz)))
        .filter(|&v| v > 0 && v <= MAX_CELLS)
        .ok_or(Error::TooManyCells)? as usize;
    let invx = f64::from(nx);
    let invy = f64::from(ny);
    let invz = f64::from(nz);
    let cell_min = (w[0] / f64::from(nx))
        .min(w[1] / f64::from(ny))
        .min(w[2] / f64::from(nz));

    let mut folded = vec![[0.0; 3]; n];
    let mut bin = vec![(0i32, 0i32, 0i32); n];
    for &i in &active {
        let s = simbox.fractional(xyz[i]);
        folded[i] = simbox.cartesian(s);
        bin[i] = (
            ((s[0] * invx) as i32).clamp(0, nx - 1),
            ((s[1] * invy) as i32).clamp(0, ny - 1),
            ((s[2] * invz) as i32).clamp(0, nz - 1),
        );
    }

    let mut head = vec![-1isize; ncell];
    let mut next = vec![-1isize; n];
    let cell_index = |ix: i32, iy: i32, iz: i32| -> usize {
        let cx = ix.rem_euclid(nx);
        let cy = iy.rem_euclid(ny);
        let cz = iz.rem_euclid(nz);
        ((cz * ny + cy) * nx + cx) as usize
    };
    for &i in &active {
        let (ix, iy, iz) = bin[i];
        let c = cell_index(ix, iy, iz);
        next[i] = head[c];
        head[c] = i as isize;
    }

    let reach_cut = (cutoff / cell_min).ceil();
    if !reach_cut.is_finite() || reach_cut > i32::MAX as f64 {
        return Err(Error::TooManyImages);
    }
    let max_reach = (reach_cut as i32)
        .max(repeats[0] * nx)
        .max(repeats[1] * ny)
        .max(repeats[2] * nz)
        .max(1);
    let cut2 = cutoff * cutoff;
    let mut pairs = Vec::new();

    for &i in &active {
        let (ix, iy, iz) = bin[i];
        let mut reach = 1i32;
        while reach <= max_reach {
            for dx in -reach..=reach {
                for dy in -reach..=reach {
                    for dz in -reach..=reach {
                        let shell = reach == 1
                            || dx.abs() == reach
                            || dy.abs() == reach
                            || dz.abs() == reach;
                        if !shell && reach > 1 {
                            continue;
                        }
                        let jx = ix + dx;
                        let jy = iy + dy;
                        let jz = iz + dz;
                        let c = cell_index(jx, jy, jz);
                        let na = jx.div_euclid(nx);
                        let nb = jy.div_euclid(ny);
                        let nc = jz.div_euclid(nz);
                        let shift_s = [na, nb, nc];
                        let shift = simbox.lattice_shift(na, nb, nc);
                        let mut j = head[c];
                        while j >= 0 {
                            let ju = j as usize;
                            let zero_self = ju == i && na == 0 && nb == 0 && nc == 0;
                            if !zero_self {
                                let d2 = simbox.dist2_shifted(folded[i], folded[ju], shift);
                                if d2 < cut2 && (!half || keep_half(i, ju, shift_s)) {
                                    pairs.push(Pair {
                                        i,
                                        j: ju,
                                        shift: shift_s,
                                        dist2: d2,
                                    });
                                }
                            }
                            j = next[ju];
                        }
                    }
                }
            }
            let bound = f64::from(reach) * cell_min;
            if bound * bound >= cut2 {
                break;
            }
            reach += 1;
        }
    }
    Ok(pairs)
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
    fn bad_cutoff() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.0, 0.0, 0.0]];
        assert_eq!(
            pairs_within(&xyz, &sim, 0.0, None, None, false).unwrap_err(),
            Error::BadCutoff
        );
    }
}
