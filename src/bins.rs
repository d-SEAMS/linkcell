//! Cell-major bins and the certified Chebyshev frontier.
//!
//! Occupants of one bin sit in a contiguous slice (HOOMD / vesin), so the
//! stencil walk is a sequential read. After shell `reach`, every unvisited
//! image lies beyond one of six lattice planes. The perpendicular distance
//! to the nearest of those planes is a lower bound on the unvisited
//! distance, and it is safe to stop once the k-th neighbour is inside it.

use crate::cell::Cell;
use crate::Error;

const MAX_CELLS: i64 = 16_777_216;
/// Shrink a geometric lower bound so a rounded-up plane cannot certify early.
const CERT_REL: f64 = 1.0e-8;
const CERT_ABS: f64 = 1.0e-12;

/// Fractional bins with cell-major occupants.
pub(crate) struct Mesh {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub widths: [f64; 3],
    pub frac: Vec<[f64; 3]>,
    pub folded: Vec<[f64; 3]>,
    pub bin: Vec<[i32; 3]>,
    offsets: Vec<usize>,
    occupants: Vec<usize>,
}

impl Mesh {
    pub(crate) fn build(
        xyz: &[[f64; 3]],
        simbox: &Cell,
        active: &[usize],
        edge: f64,
    ) -> Result<Self, Error> {
        let widths = simbox.widths();
        let nx = bins_1d(widths[0], edge)?;
        let ny = bins_1d(widths[1], edge)?;
        let nz = bins_1d(widths[2], edge)?;
        let ncell = (i64::from(nx))
            .checked_mul(i64::from(ny))
            .and_then(|v| v.checked_mul(i64::from(nz)))
            .filter(|&v| v > 0 && v <= MAX_CELLS)
            .ok_or(Error::TooManyCells)? as usize;

        let n = xyz.len();
        let mut frac = vec![[0.0; 3]; n];
        let mut folded = vec![[0.0; 3]; n];
        let mut bin = vec![[0; 3]; n];
        let mut counts = vec![0usize; ncell];
        for &i in active {
            let s = simbox.fractional(xyz[i]);
            let ix = bin_coord(s[0], nx);
            let iy = bin_coord(s[1], ny);
            let iz = bin_coord(s[2], nz);
            frac[i] = s;
            folded[i] = simbox.cartesian(s);
            bin[i] = [ix, iy, iz];
            counts[cell_index(ix, iy, iz, nx, ny, nz)] += 1;
        }

        let mut offsets = vec![0usize; ncell + 1];
        for c in 0..ncell {
            offsets[c + 1] = offsets[c] + counts[c];
        }
        let mut cursor = offsets.clone();
        let mut occupants = vec![0usize; active.len()];
        for &i in active {
            let [ix, iy, iz] = bin[i];
            let c = cell_index(ix, iy, iz, nx, ny, nz);
            let slot = cursor[c];
            occupants[slot] = i;
            cursor[c] = slot + 1;
        }

        Ok(Self {
            nx,
            ny,
            nz,
            widths,
            frac,
            folded,
            bin,
            offsets,
            occupants,
        })
    }

    pub(crate) fn cell_of(&self, ix: i32, iy: i32, iz: i32) -> usize {
        cell_index(ix, iy, iz, self.nx, self.ny, self.nz)
    }

    pub(crate) fn slots(&self, cell: usize) -> &[usize] {
        let lo = self.offsets[cell];
        let hi = self.offsets[cell + 1];
        &self.occupants[lo..hi]
    }

    /// Half-box of bins. The search cap is at least this, and at least
    /// the space diagonal in cell heights.
    pub(crate) fn image_reach(&self) -> i32 {
        self.nx.max(self.ny).max(self.nz) / 2 + 2
    }
}

pub(crate) fn target_edge(simbox: &Cell, hint: Option<f64>, fallback: f64) -> f64 {
    let mut edge = hint.unwrap_or(fallback);
    if !edge.is_finite() || edge <= 0.0 {
        edge = fallback;
    }
    let w = simbox.widths();
    edge.min(w[0]).min(w[1]).min(w[2])
}

/// Visit the Chebyshev shell `max(|dx|,|dy|,|dz|) == reach`.
///
/// Reach 1 also visits the home cell. Later shells are the surface only.
pub(crate) fn for_shell(reach: i32, mut visit: impl FnMut(i32, i32, i32)) {
    if reach == 1 {
        visit(0, 0, 0);
    }
    let r = reach;
    for dz in [-r, r] {
        for dx in -r..=r {
            for dy in -r..=r {
                visit(dx, dy, dz);
            }
        }
    }
    for dy in [-r, r] {
        for dx in -r..=r {
            for dz in (1 - r)..r {
                visit(dx, dy, dz);
            }
        }
    }
    for dx in [-r, r] {
        for dy in (1 - r)..r {
            for dz in (1 - r)..r {
                visit(dx, dy, dz);
            }
        }
    }
}

/// Squared lower bound on any point outside the visited Chebyshev cube.
pub(crate) fn frontier_dist2(
    s: [f64; 3],
    bin: [i32; 3],
    reach: i32,
    n: [i32; 3],
    w: [f64; 3],
) -> f64 {
    let mut gap = f64::INFINITY;
    for a in 0..3 {
        let nf = f64::from(n[a]);
        let plus = (f64::from(bin[a] + reach + 1) / nf - s[a]) * w[a];
        let minus = (s[a] - f64::from(bin[a] - reach) / nf) * w[a];
        gap = gap.min(plus).min(minus);
    }
    let d = certify(gap);
    d * d
}

/// Squared lower bound on the distance from `s` to an image bin.
pub(crate) fn slab_dist2(s: [f64; 3], unfolded: [i32; 3], n: [i32; 3], w: [f64; 3]) -> f64 {
    let mut gap = 0.0_f64;
    for a in 0..3 {
        let nf = f64::from(n[a]);
        let lo = f64::from(unfolded[a]) / nf;
        let hi = f64::from(unfolded[a] + 1) / nf;
        let axis = if s[a] < lo {
            (lo - s[a]) * w[a]
        } else if s[a] > hi {
            (s[a] - hi) * w[a]
        } else {
            0.0
        };
        gap = gap.max(axis);
    }
    let d = certify(gap);
    d * d
}

fn bins_1d(width: f64, edge: f64) -> Result<i32, Error> {
    let n = (width / edge).floor().max(1.0);
    if !n.is_finite() || n > 1_000_000.0 {
        return Err(Error::TooManyCells);
    }
    Ok(n as i32)
}

fn bin_coord(s: f64, n: i32) -> i32 {
    let mut c = (s * f64::from(n)) as i32;
    if c < 0 {
        c = 0;
    }
    if c >= n {
        c = n - 1;
    }
    c
}

fn cell_index(ix: i32, iy: i32, iz: i32, nx: i32, ny: i32, nz: i32) -> usize {
    let cx = ix.rem_euclid(nx);
    let cy = iy.rem_euclid(ny);
    let cz = iz.rem_euclid(nz);
    ((cz * ny + cy) * nx + cx) as usize
}

fn certify(dist: f64) -> f64 {
    if dist <= 0.0 || !dist.is_finite() {
        return 0.0;
    }
    let d = dist - dist * CERT_REL - CERT_ABS;
    if d > 0.0 {
        d
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_covers_the_cube_surface_once() {
        for reach in 1..=4 {
            let mut seen = Vec::new();
            for_shell(reach, |dx, dy, dz| seen.push((dx, dy, dz)));
            seen.sort_unstable();
            let mut uniq = seen.clone();
            uniq.dedup();
            assert_eq!(seen.len(), uniq.len(), "reach {reach} repeats");
            let side = 2 * reach + 1;
            let volume = (side * side * side) as usize;
            let inner = if reach == 1 {
                0
            } else {
                let s = 2 * (reach - 1) + 1;
                (s * s * s) as usize
            };
            assert_eq!(seen.len(), volume - inner);
            assert_eq!(seen.contains(&(0, 0, 0)), reach == 1);
        }
    }

    #[test]
    fn frontier_is_at_least_the_min_cell_bound() {
        // Bin 2 of 8 covers fractional [0.25, 0.375). The source sits inside it.
        let s = [0.30, 0.30, 0.30];
        let bin = [2, 2, 2];
        let n = [8, 8, 8];
        let w = [10.0, 10.0, 10.0];
        let h = w[0] / f64::from(n[0]);
        for reach in 1..=3 {
            let bound = frontier_dist2(s, bin, reach, n, w);
            let loose = (f64::from(reach) * h) * (f64::from(reach) * h);
            assert!(
                bound + 1e-9 >= loose * (1.0 - 2.0 * CERT_REL),
                "reach {reach}: {bound} vs {loose}"
            );
        }
    }
}
