//! Cell-major bins and the certified Chebyshev frontier.
//!
//! Occupants of one bin sit in a contiguous slice (HOOMD / vesin), so the
//! stencil walk is a sequential read. After shell `reach`, every unvisited
//! image lies beyond one of six lattice planes. The perpendicular distance
//! to the nearest of those planes is a lower bound on the unvisited
//! distance, and it is safe to stop once the k-th neighbour is inside it.

use std::cell::RefCell;

use crate::cell::Cell;
use crate::Error;

const MAX_CELLS: i64 = 16_777_216;
/// Shrink a cutoff slab so a rounded-up plane cannot drop a pair.
const CERT_REL: f64 = 1.0e-8;
const CERT_ABS: f64 = 1.0e-12;

#[derive(Default)]
struct Recycled {
    frac: Vec<[f64; 3]>,
    folded: Vec<[f64; 3]>,
    bin: Vec<[i32; 3]>,
    offsets: Vec<usize>,
    occupants: Vec<usize>,
    counts: Vec<usize>,
    cursor: Vec<usize>,
}

thread_local! {
    // `const { ... }` is newer than this crate's 1.70 floor.
    #[allow(clippy::missing_const_for_thread_local)]
    static MESH_POOL: RefCell<Option<Recycled>> = RefCell::new(None);
}

fn take_recycled() -> Recycled {
    MESH_POOL.with(|slot| slot.borrow_mut().take().unwrap_or_default())
}

fn recycle(buf: Recycled) {
    const MAX_KEEP: usize = 1 << 20;
    if buf.frac.capacity() > MAX_KEEP || buf.offsets.capacity() > MAX_KEEP {
        return;
    }
    MESH_POOL.with(|slot| {
        let mut guard = slot.borrow_mut();
        if guard.is_none() {
            *guard = Some(buf);
        }
    });
}

/// Fractional bins with cell-major occupants.
pub(crate) struct Mesh {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub widths: [f64; 3],
    pub frac: Vec<[f64; 3]>,
    pub folded: Vec<[f64; 3]>,
    pub bin: Vec<[i32; 3]>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) occupants: Vec<usize>,
    counts: Vec<usize>,
    cursor: Vec<usize>,
}

impl Drop for Mesh {
    fn drop(&mut self) {
        recycle(Recycled {
            frac: std::mem::take(&mut self.frac),
            folded: std::mem::take(&mut self.folded),
            bin: std::mem::take(&mut self.bin),
            offsets: std::mem::take(&mut self.offsets),
            occupants: std::mem::take(&mut self.occupants),
            counts: std::mem::take(&mut self.counts),
            cursor: std::mem::take(&mut self.cursor),
        });
    }
}

impl Mesh {
    pub(crate) fn build(
        xyz: &[[f64; 3]],
        simbox: &Cell,
        active: Option<&[usize]>,
        edge: f64,
    ) -> Result<Self, Error> {
        let n_active = active.map_or(xyz.len(), |list| list.len());
        if let Some(list) = active {
            Self::assemble(xyz, simbox, edge, n_active, list.iter().copied())
        } else {
            Self::assemble(xyz, simbox, edge, n_active, 0..xyz.len())
        }
    }

    fn assemble<I>(
        xyz: &[[f64; 3]],
        simbox: &Cell,
        edge: f64,
        n_active: usize,
        indices: I,
    ) -> Result<Self, Error>
    where
        I: Iterator<Item = usize> + Clone,
    {
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
        let mut buf = take_recycled();
        if buf.frac.len() != n {
            buf.frac.resize(n, [0.0; 3]);
            buf.folded.resize(n, [0.0; 3]);
            buf.bin.resize(n, [0; 3]);
        }
        buf.counts.clear();
        buf.counts.resize(ncell, 0);
        for i in indices.clone() {
            let s = simbox.fractional(xyz[i]);
            let ix = bin_coord(s[0], nx);
            let iy = bin_coord(s[1], ny);
            let iz = bin_coord(s[2], nz);
            buf.frac[i] = s;
            buf.folded[i] = simbox.cartesian(s);
            buf.bin[i] = [ix, iy, iz];
            buf.counts[cell_index(ix, iy, iz, nx, ny, nz)] += 1;
        }

        buf.offsets.clear();
        buf.offsets.resize(ncell + 1, 0);
        for c in 0..ncell {
            buf.offsets[c + 1] = buf.offsets[c] + buf.counts[c];
        }
        buf.cursor.clear();
        buf.cursor.resize(ncell + 1, 0);
        buf.cursor.copy_from_slice(&buf.offsets);
        buf.occupants.resize(n_active, 0);
        for i in indices {
            let [ix, iy, iz] = buf.bin[i];
            let c = cell_index(ix, iy, iz, nx, ny, nz);
            let slot = buf.cursor[c];
            buf.occupants[slot] = i;
            buf.cursor[c] = slot + 1;
        }

        Ok(Self {
            nx,
            ny,
            nz,
            widths,
            frac: buf.frac,
            folded: buf.folded,
            bin: buf.bin,
            offsets: buf.offsets,
            occupants: buf.occupants,
            counts: buf.counts,
            cursor: buf.cursor,
        })
    }

    /// Bin index plus the integer image `(na, nb, nc)` of an unfolded cell.
    #[inline(always)]
    pub(crate) fn locate(&self, ix: i32, iy: i32, iz: i32) -> (usize, i32, i32, i32) {
        let (cx, na) = split_axis(ix, self.nx);
        let (cy, nb) = split_axis(iy, self.ny);
        let (cz, nc) = split_axis(iz, self.nz);
        let cell = ((cz * self.ny + cy) * self.nx + cx) as usize;
        (cell, na, nb, nc)
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
///
/// Bins are half-open, so an unvisited point lies strictly past this
/// plane. `conservative` shrinks the plane for a cutoff test, where a
/// rounded-up gap would drop a pair that is still inside the cutoff.
/// k-nearest passes `false`: a neighbour sitting on the plane is still
/// the nearest, and shrinking it forces another shell.
pub(crate) fn frontier_dist2(
    s: [f64; 3],
    bin: [i32; 3],
    reach: i32,
    n: [i32; 3],
    w: [f64; 3],
    conservative: bool,
) -> f64 {
    let mut gap = f64::INFINITY;
    for a in 0..3 {
        let inv = 1.0 / f64::from(n[a]);
        let plus = (f64::from(bin[a] + reach + 1) * inv - s[a]) * w[a];
        let minus = (s[a] - f64::from(bin[a] - reach) * inv) * w[a];
        gap = gap.min(plus).min(minus);
    }
    if conservative {
        gap = certify(gap);
    }
    if gap > 0.0 && gap.is_finite() {
        gap * gap
    } else {
        0.0
    }
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

#[inline(always)]
fn split_axis(i: i32, n: i32) -> (i32, i32) {
    if i >= 0 && i < n {
        (i, 0)
    } else {
        (i.rem_euclid(n), i.div_euclid(n))
    }
}

fn cell_index(ix: i32, iy: i32, iz: i32, nx: i32, ny: i32, nz: i32) -> usize {
    let (cx, _) = split_axis(ix, nx);
    let (cy, _) = split_axis(iy, ny);
    let (cz, _) = split_axis(iz, nz);
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
            let bound = frontier_dist2(s, bin, reach, n, w, false);
            let loose = (f64::from(reach) * h) * (f64::from(reach) * h);
            assert!(
                bound + 1e-9 >= loose * (1.0 - 2.0 * CERT_REL),
                "reach {reach}: {bound} vs {loose}"
            );
        }
    }
}
