//! Linked-cell k-nearest search (Allen and Tildesley).
//!
//! Fold into the primary cell, bin on the fractional mesh, then expand
//! Chebyshev shells until the k-th neighbour cannot sit outside the
//! visited cube. Distances are [`crate::Cell::dist2_shifted`] plus
//! [`crate::Cell::lattice_shift`]. The walk keys on the integer stencil,
//! not a unique-cell stamp: occupants of one bin can need different
//! lattice images of the same source.
//!
//! [`knearest`] returns one [`Neighbors`] row per point. [`knearest_into`]
//! writes packed `n * k` indices (`-1` unused). [`knearest_into_d2`]
//! also writes the matching squared distances (`NaN` unused). Sources run under
//! rayon when the `parallel` feature is on (the default); build with
//! `--no-default-features` to serialize. The per-source `KHeap` stays on
//! the stack for `k <= 16`.

use crate::cell::Cell;
use crate::Error;

const MAX_CELLS: i64 = 16_777_216;

type SearchHits = Vec<(usize, Vec<(f64, usize)>)>;

fn pair_dist2(simbox: &Cell, p: [f64; 3], q: [f64; 3]) -> f64 {
    simbox.dist2_euclidean(p, q)
}

fn bins_1d(width: f64, edge: f64) -> Result<i32, Error> {
    let n = (width / edge).floor().max(1.0);
    if !n.is_finite() || n > 1_000_000.0 {
        return Err(Error::TooManyCells);
    }
    Ok(n as i32)
}

/// Bounded max-heap of `(dist2, index)`. `k <= 16` stays in
/// `[f64; 16]` / `[usize; 16]` so the pair loop does not allocate;
/// larger `k` uses `extra_*` vectors.
struct KHeap {
    d2: [f64; 16],
    idx: [usize; 16],
    extra_d2: Vec<f64>,
    extra_idx: Vec<usize>,
    n: usize,
    k: usize,
}

impl KHeap {
    fn new(k: usize) -> Self {
        let mut extra_d2 = Vec::new();
        let mut extra_idx = Vec::new();
        if k > 16 {
            extra_d2.resize(k, 0.0);
            extra_idx.resize(k, 0);
        }
        Self {
            d2: [0.0; 16],
            idx: [0; 16],
            extra_d2,
            extra_idx,
            n: 0,
            k,
        }
    }

    fn d2_at(&self, t: usize) -> f64 {
        if self.k <= 16 {
            self.d2[t]
        } else {
            self.extra_d2[t]
        }
    }

    fn set(&mut self, t: usize, d2: f64, j: usize) {
        if self.k <= 16 {
            self.d2[t] = d2;
            self.idx[t] = j;
        } else {
            self.extra_d2[t] = d2;
            self.extra_idx[t] = j;
        }
    }

    fn idx_at(&self, t: usize) -> usize {
        if self.k <= 16 {
            self.idx[t]
        } else {
            self.extra_idx[t]
        }
    }

    fn push(&mut self, d2: f64, j: usize) {
        for t in 0..self.n {
            if self.idx_at(t) == j {
                if d2 < self.d2_at(t) {
                    self.set(t, d2, j);
                }
                return;
            }
        }
        if self.n < self.k {
            self.set(self.n, d2, j);
            self.n += 1;
            return;
        }
        let mut worst = 0;
        for t in 1..self.n {
            if self.d2_at(t) > self.d2_at(worst) {
                worst = t;
            }
        }
        if d2 < self.d2_at(worst) {
            self.set(worst, d2, j);
        }
    }

    fn full(&self) -> bool {
        self.n >= self.k
    }

    fn worst(&self) -> f64 {
        let mut w = self.d2_at(0);
        for t in 1..self.n {
            let v = self.d2_at(t);
            if v > w {
                w = v;
            }
        }
        w
    }

    fn finish(self) -> Vec<(f64, usize)> {
        let mut pairs = Vec::with_capacity(self.n);
        for t in 0..self.n {
            let j = if self.k <= 16 {
                self.idx[t]
            } else {
                self.extra_idx[t]
            };
            pairs.push((self.d2_at(t), j));
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        pairs
    }
}

/// One source's k nearest neighbours, nearest first.
///
/// Empty `indices` / `dist2` when the point is masked or isolated.
/// Length is `min(k, n_active - 1)` for an active source.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Neighbors {
    /// Candidate indices into the input point list.
    pub indices: Vec<usize>,
    /// Squared minimum-image distances, parallel to [`Self::indices`].
    pub dist2: Vec<f64>,
}

/// Write k-nearest indices, nearest first, into caller storage.
///
/// `out` has length `n * k` ([`Error::BufferSize`] otherwise). Unused
/// slots are `-1`. Neighbours of source `i` occupy `out[i * k ..]`.
/// `mask` is `None` or length `n` ([`Error::MaskLen`] otherwise).
///
/// ```
/// use linkcell::{knearest_into, Cell};
///
/// # fn main() -> Result<(), linkcell::Error> {
/// let sim = Cell::ortho(10.0, 10.0, 10.0)?;
/// let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
/// let mut out = [0; 4];
/// knearest_into(&xyz, &sim, 2, None, None, &mut out)?;
/// assert_eq!(out, [1, -1, 0, -1]);
/// # Ok(())
/// # }
/// ```
pub fn knearest_into(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    out: &mut [i32],
) -> Result<(), Error> {
    let n = xyz.len();
    if n.checked_mul(k) != Some(out.len()) {
        return Err(Error::BufferSize);
    }
    knearest_into_d2(xyz, simbox, k, mask, cell_hint, out, None)
}

/// Write k-nearest indices and squared distances, nearest first.
///
/// `out_nn` has length `n * k`. Unused index slots are `-1`.
/// `out_d2`, when `Some`, has the same length; unused slots are `NaN`.
pub fn knearest_into_d2(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    out_nn: &mut [i32],
    out_d2: Option<&mut [f64]>,
) -> Result<(), Error> {
    write_one(xyz, simbox, k, mask, cell_hint, out_nn, out_d2)
}

/// Frame-major batch: `xyz` is `n_frames * n` points, `out_*` are
/// `n_frames * n * k`. One shared cell. `mask` is length `n` or `None`.
#[allow(clippy::too_many_arguments)]
pub fn knearest_into_many(
    xyz: &[[f64; 3]],
    n: usize,
    n_frames: usize,
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    out_nn: &mut [i32],
    mut out_d2: Option<&mut [f64]>,
) -> Result<(), Error> {
    if n_frames == 0 {
        return Err(Error::Empty);
    }
    let Some(n_pts) = n.checked_mul(n_frames) else {
        return Err(Error::Overflow);
    };
    if xyz.len() != n_pts {
        return Err(Error::BufferSize);
    }
    let Some(need) = n_pts.checked_mul(k) else {
        return Err(Error::Overflow);
    };
    if out_nn.len() != need {
        return Err(Error::BufferSize);
    }
    if let Some(d2) = out_d2.as_ref() {
        if d2.len() != need {
            return Err(Error::BufferSize);
        }
    }
    for f in 0..n_frames {
        let lo = f * n;
        let hi = lo + n;
        let nn_lo = f * n * k;
        let nn_hi = nn_lo + n * k;
        let frame_d2 = out_d2.as_mut().map(|d| &mut d[nn_lo..nn_hi]);
        write_one(
            &xyz[lo..hi],
            simbox,
            k,
            mask,
            cell_hint,
            &mut out_nn[nn_lo..nn_hi],
            frame_d2,
        )?;
    }
    Ok(())
}

fn write_one(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    out_nn: &mut [i32],
    mut out_d2: Option<&mut [f64]>,
) -> Result<(), Error> {
    let n = xyz.len();
    if n.checked_mul(k) != Some(out_nn.len()) {
        return Err(Error::BufferSize);
    }
    if let Some(d2) = out_d2.as_ref() {
        if d2.len() != out_nn.len() {
            return Err(Error::BufferSize);
        }
    }
    out_nn.fill(-1);
    if let Some(d2) = out_d2.as_mut() {
        d2.fill(f64::NAN);
    }
    let rows = search(xyz, simbox, k, mask, cell_hint)?;
    for (i, pairs) in rows {
        for (t, &(d2, j)) in pairs.iter().enumerate() {
            out_nn[i * k + t] = j as i32;
            if let Some(buf) = out_d2.as_mut() {
                buf[i * k + t] = d2;
            }
        }
    }
    Ok(())
}

/// k-nearest neighbours of every point (or of the masked subset).
///
/// `mask[i] == false` drops point `i` from both sources and candidates.
/// `mask` is `None` or length `n` ([`Error::MaskLen`] otherwise).
/// `cell_hint` is the target cell edge; `None` uses 3.0 in the same units
/// as the box. Each row has `min(k, n_active - 1)` entries.
///
/// Fold, bin, then expand Chebyshev shells. Distances are
/// [`Cell::dist2_shifted`] plus [`Cell::lattice_shift`]. The walk does
/// not stamp unique cells: occupants of one bin can need different
/// images.
///
/// ```
/// use linkcell::{knearest, Cell};
///
/// # fn main() -> Result<(), linkcell::Error> {
/// let sim = Cell::ortho(10.0, 10.0, 10.0)?;
/// let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let mask = [true, false, true];
/// let rows = knearest(&xyz, &sim, 1, Some(&mask), None)?;
/// assert!(rows[1].indices.is_empty());
/// assert_eq!(rows[0].indices, vec![2]);
/// assert_eq!(rows[2].indices, vec![0]);
/// # Ok(())
/// # }
/// ```
pub fn knearest(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
) -> Result<Vec<Neighbors>, Error> {
    let n = xyz.len();
    let mut out = vec![Neighbors::default(); n];
    for (i, pairs) in search(xyz, simbox, k, mask, cell_hint)? {
        out[i].dist2 = pairs.iter().map(|p| p.0).collect();
        out[i].indices = pairs.iter().map(|p| p.1).collect();
    }
    Ok(out)
}

fn search(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
) -> Result<SearchHits, Error> {
    if k == 0 {
        return Err(Error::ZeroK);
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

    let mut edge = cell_hint.unwrap_or(3.0);
    if !edge.is_finite() || edge <= 0.0 {
        edge = 3.0;
    }
    let w = simbox.widths();
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

    // Fold into the primary cell once. Pair distances are then a
    // Cartesian subtract plus a lattice shift (vesin / LAMMPS ghosts).
    // The walk keys on the integer stencil, not a unique rem_euclid
    // cell: occupants of one bin can need different images.
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

    let max_reach = nx.max(ny).max(nz) / 2 + 1;
    let one = |i: usize| -> (usize, Vec<(f64, usize)>) {
        let mut heap = KHeap::new(k);
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
                        let shift = simbox.lattice_shift(
                            jx.div_euclid(nx),
                            jy.div_euclid(ny),
                            jz.div_euclid(nz),
                        );
                        let mut j = head[c];
                        while j >= 0 {
                            let ju = j as usize;
                            if ju != i {
                                heap.push(simbox.dist2_shifted(folded[i], folded[ju], shift), ju);
                            }
                            j = next[ju];
                        }
                    }
                }
            }
            if heap.full() {
                let bound = f64::from(reach) * cell_min;
                if heap.worst() <= bound * bound {
                    break;
                }
            }
            reach += 1;
        }
        (i, heap.finish())
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        Ok(active.par_iter().copied().map(one).collect())
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok(active.iter().copied().map(one).collect())
    }
}

/// Brute-force k-nearest. Tests and small systems only.
///
/// Orthorhombic boxes use [`Cell::dist2`]. Sheared boxes take the
/// minimum over the 27 nearest lattice images: the single
/// parallelepiped wrap is not the Wigner-Seitz cell of a 60-degree
/// hex prism.
///
/// ```
/// use linkcell::{knearest, knearest_brute, Cell};
///
/// # fn main() -> Result<(), linkcell::Error> {
/// let sim = Cell::ortho(10.0, 10.0, 10.0)?;
/// let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
/// let cell = knearest(&xyz, &sim, 1, None, None)?;
/// let brute = knearest_brute(&xyz, &sim, 1, None)?;
/// assert_eq!(cell[0].indices, brute[0].indices);
/// assert!((cell[0].dist2[0] - brute[0].dist2[0]).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
pub fn knearest_brute(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
) -> Result<Vec<Neighbors>, Error> {
    if k == 0 {
        return Err(Error::ZeroK);
    }
    if xyz.is_empty() {
        return Err(Error::Empty);
    }
    let n = xyz.len();
    let active: Vec<usize> = (0..n)
        .filter(|&i| {
            mask.map(|m| m.get(i).copied().unwrap_or(false))
                .unwrap_or(true)
        })
        .collect();
    let mut out = vec![Neighbors::default(); n];
    for &i in &active {
        let mut pairs: Vec<(f64, usize)> = active
            .iter()
            .copied()
            .filter(|&j| j != i)
            .map(|j| (pair_dist2(simbox, xyz[i], xyz[j]), j))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        pairs.truncate(k);
        out[i].dist2 = pairs.iter().map(|p| p.0).collect();
        out[i].indices = pairs.iter().map(|p| p.1).collect();
    }
    Ok(out)
}
