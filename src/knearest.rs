//! Linked-cell k-nearest search (Allen and Tildesley).
//!
//! Fold into the primary cell, bin on the fractional mesh in cell-major
//! order, then grow the index box until the k-th neighbour lies inside
//! the perpendicular distance to the unvisited frontier. An orthorhombic
//! box is walked as stored. A restricted triclinic box is tilt-reduced
//! first. A general orientation is Minkowski-reduced. Distances are
//! [`crate::Cell::dist2_shifted`]
//! plus [`crate::Cell::lattice_shift`]. The walk keys on the integer
//! stencil, not a unique-cell stamp: occupants of one bin can need
//! different lattice images of the same source.
//!
//! [`knearest`] returns one [`Neighbors`] row per point. [`knearest_into`]
//! writes packed `n * k` indices (`-1` unused). [`knearest_into_d2`]
//! also writes the matching squared distances (`NaN` unused). Sources run under
//! rayon when the `parallel` feature is on (the default); build with
//! `--no-default-features` to serialize. At 8192 active points and above,
//! that feature also builds the mesh in parallel. The per-source `KHeap` stays
//! on the stack for `k <= 16`.

use crate::bins::{self, axis_gap, for_new_layer, slab_dist2, Mesh};
use crate::cell::Cell;
use crate::Error;

fn pair_dist2(simbox: &Cell, p: [f64; 3], q: [f64; 3]) -> f64 {
    simbox.dist2_euclidean(p, q)
}

/// Orthorhombic MIC for every candidate, via minimage's Highway kernel.
fn ortho_highway_dist2(
    p: [f64; 3],
    others: &[usize],
    xyz: &[[f64; 3]],
    widths: [f64; 3],
) -> Result<Vec<f64>, Error> {
    let n = others.len();
    let mut dx = Vec::with_capacity(n);
    let mut dy = Vec::with_capacity(n);
    let mut dz = Vec::with_capacity(n);
    for &j in others {
        let q = xyz[j];
        dx.push(q[0] - p[0]);
        dy.push(q[1] - p[1]);
        dz.push(q[2] - p[2]);
    }
    let mut out = vec![0.0; n];
    crate::dist2_ortho_diffs(&dx, &dy, &dz, widths[0], widths[1], widths[2], &mut out)?;
    Ok(out)
}

fn box_diameter(cell: &Cell) -> f64 {
    let a = cell.a();
    let b = cell.b();
    let c = cell.c();
    let mut best = 0.0_f64;
    for &sa in &[-1.0, 1.0] {
        for &sb in &[-1.0, 1.0] {
            for &sc in &[-1.0, 1.0] {
                let x = sa * a[0] + sb * b[0] + sc * c[0];
                let y = sa * a[1] + sb * b[1] + sc * c[1];
                let z = sa * a[2] + sb * b[2] + sc * c[2];
                best = best.max(x * x + y * y + z * z);
            }
        }
    }
    best.sqrt()
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
    /// Slot of the lexicographic maximum `(dist2, index)`.
    worst_at: usize,
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
            worst_at: 0,
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

    fn worse_than(&self, d2: f64, j: usize, other: usize) -> bool {
        let od = self.d2_at(other);
        let oj = self.idx_at(other);
        d2 > od || (d2 == od && j > oj)
    }

    fn recompute_worst(&mut self) {
        let mut w = 0;
        for t in 1..self.n {
            if self.worse_than(self.d2_at(t), self.idx_at(t), w) {
                w = t;
            }
        }
        self.worst_at = w;
    }

    /// Insert the nearest image of `j`. Equal distances keep the smaller index.
    fn push(&mut self, d2: f64, j: usize) {
        if self.n == self.k {
            let wd = self.d2_at(self.worst_at);
            let wj = self.idx_at(self.worst_at);
            // A candidate that does not beat the worst cannot improve any
            // stored image either: those distances are at most the worst.
            if d2 > wd || (d2 == wd && j >= wj) {
                return;
            }
        }
        for t in 0..self.n {
            if self.idx_at(t) == j {
                if d2 < self.d2_at(t) {
                    self.set(t, d2, j);
                    if t == self.worst_at {
                        self.recompute_worst();
                    }
                }
                return;
            }
        }
        if self.n < self.k {
            let slot = self.n;
            self.set(slot, d2, j);
            if slot == 0 || self.worse_than(d2, j, self.worst_at) {
                self.worst_at = slot;
            }
            self.n += 1;
            return;
        }
        self.set(self.worst_at, d2, j);
        self.recompute_worst();
    }

    fn full(&self) -> bool {
        self.n >= self.k
    }

    fn worst(&self) -> f64 {
        self.d2_at(self.worst_at)
    }

    /// Write nearest-first into caller slots. Unused tail stays as the caller left it.
    fn write_sorted(&self, nn: &mut [i32], mut d2: Option<&mut [f64]>) {
        if self.k <= 16 {
            let n = self.n;
            let mut order = [0u8; 16];
            for (t, slot) in order.iter_mut().enumerate().take(n) {
                *slot = t as u8;
            }
            order[..n].sort_by(|&a, &b| {
                let (a, b) = (a as usize, b as usize);
                self.d2[a]
                    .total_cmp(&self.d2[b])
                    .then(self.idx[a].cmp(&self.idx[b]))
            });
            for (t, &slot) in order[..n].iter().enumerate() {
                let s = slot as usize;
                nn[t] = self.idx[s] as i32;
                if let Some(buf) = d2.as_mut() {
                    buf[t] = self.d2[s];
                }
            }
        } else {
            let mut order: Vec<usize> = (0..self.n).collect();
            order.sort_by(|&a, &b| {
                self.extra_d2[a]
                    .total_cmp(&self.extra_d2[b])
                    .then(self.extra_idx[a].cmp(&self.extra_idx[b]))
            });
            for (t, &s) in order.iter().enumerate() {
                nn[t] = self.extra_idx[s] as i32;
                if let Some(buf) = d2.as_mut() {
                    buf[t] = self.extra_d2[s];
                }
            }
        }
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
    execute(xyz, simbox, k, mask, cell_hint, out_nn, out_d2)
}

/// k-nearest neighbours of every point (or of the masked subset).
///
/// `mask[i] == false` drops point `i` from both sources and candidates.
/// `mask` is `None` or length `n` ([`Error::MaskLen`] otherwise).
/// `cell_hint` is the target cell edge; `None` uses 3.0 in the same units
/// as the box. Each row has `min(k, n_active - 1)` entries.
///
/// Fold, bin, then grow a rectangular index box. An orthorhombic cell
/// is used as stored. A restricted triclinic cell is tilt-reduced
/// first. A general orientation is Minkowski-reduced. Distances are
/// [`Cell::dist2_shifted`] plus [`Cell::lattice_shift`]. The walk does
/// not stamp unique cells: occupants of one bin can need different
/// images. A full heap stops when its worst distance is at most the
/// nearest unvisited face.
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
    let Some(nk) = n.checked_mul(k) else {
        return Err(Error::Overflow);
    };
    let mut nn = vec![-1i32; nk];
    let mut d2 = vec![f64::NAN; nk];
    write_one(xyz, simbox, k, mask, cell_hint, &mut nn, Some(&mut d2))?;
    let mut out = vec![Neighbors::default(); n];
    for i in 0..n {
        let row_nn = &nn[i * k..(i + 1) * k];
        let row_d2 = &d2[i * k..(i + 1) * k];
        let m = row_nn.iter().take_while(|j| **j >= 0).count();
        out[i].indices = row_nn[..m].iter().map(|&j| j as usize).collect();
        out[i].dist2 = row_d2[..m].to_vec();
    }
    Ok(out)
}

struct Geom {
    cols: [[f64; 3]; 3],
    lengths: [f64; 3],
    nbin: [i32; 3],
    widths: [f64; 3],
}

fn execute(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    out_nn: &mut [i32],
    out_d2: Option<&mut [f64]>,
) -> Result<(), Error> {
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
    let n_active = match mask {
        Some(m) => m.iter().filter(|&&on| on).count(),
        None => n,
    };
    if n_active <= 1 {
        return Ok(());
    }
    crate::pop::prepare();

    // Orthorhombic MIC is the per-axis wrap. A LAMMPS / GROMACS
    // restricted cell is tilt-reduced in place (same Cartesian frame).
    // Anything else is Minkowski-reduced so the short vectors are the edges.
    let walk = if simbox.is_ortho() {
        *simbox
    } else if simbox.is_restricted() {
        simbox.reduce_tilts().unwrap_or(*simbox)
    } else {
        match minimage::minkowski_reduce(simbox) {
            Ok(cell) => cell,
            Err(_) => *simbox,
        }
    };
    let edge = bins::target_edge(&walk, cell_hint, 3.0);
    let active_owned = mask.map(|m| (0..n).filter(|&i| m[i]).collect::<Vec<_>>());
    let mesh = Mesh::build(xyz, &walk, active_owned.as_deref(), edge)?;
    let nbin = [mesh.nx, mesh.ny, mesh.nz];
    let widths = mesh.widths;
    let cell_min = (widths[0] / f64::from(mesh.nx))
        .min(widths[1] / f64::from(mesh.ny))
        .min(widths[2] / f64::from(mesh.nz));
    let need = (box_diameter(&walk) / cell_min).ceil();
    let max_reach = if need.is_finite() && need < 1.0e7 {
        (need as i32).saturating_add(3).max(mesh.image_reach())
    } else {
        mesh.image_reach()
    };
    let geom = Geom {
        cols: [walk.a(), walk.b(), walk.c()],
        lengths: walk.widths(),
        nbin,
        widths,
    };
    if walk.is_ortho() {
        dispatch::<0>(&mesh, &geom, k, max_reach, mask, out_nn, out_d2);
    } else if walk.is_restricted() {
        dispatch::<1>(&mesh, &geom, k, max_reach, mask, out_nn, out_d2);
    } else {
        dispatch::<2>(&mesh, &geom, k, max_reach, mask, out_nn, out_d2);
    }
    Ok(())
}

fn dispatch<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    mask: Option<&[bool]>,
    out_nn: &mut [i32],
    out_d2: Option<&mut [f64]>,
) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        match (mask, out_d2) {
            (None, None) => {
                out_nn.par_chunks_mut(k).enumerate().for_each_init(
                    crate::pop::JobTimer::new,
                    |_timer, (i, nn)| {
                        walk_source::<MODE>(mesh, geom, k, max_reach, i, nn, None);
                    },
                );
            }
            (None, Some(d2)) => {
                out_nn
                    .par_chunks_mut(k)
                    .zip(d2.par_chunks_mut(k))
                    .enumerate()
                    .for_each_init(crate::pop::JobTimer::new, |_timer, (i, (nn, dd))| {
                        walk_source::<MODE>(mesh, geom, k, max_reach, i, nn, Some(dd));
                    });
            }
            (Some(mask), None) => {
                out_nn.par_chunks_mut(k).enumerate().for_each_init(
                    crate::pop::JobTimer::new,
                    |_timer, (i, nn)| {
                        if mask[i] {
                            walk_source::<MODE>(mesh, geom, k, max_reach, i, nn, None);
                        }
                    },
                );
            }
            (Some(mask), Some(d2)) => {
                out_nn
                    .par_chunks_mut(k)
                    .zip(d2.par_chunks_mut(k))
                    .enumerate()
                    .for_each_init(crate::pop::JobTimer::new, |_timer, (i, (nn, dd))| {
                        if mask[i] {
                            walk_source::<MODE>(mesh, geom, k, max_reach, i, nn, Some(dd));
                        }
                    });
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _timer = crate::pop::JobTimer::new();
        match (mask, out_d2) {
            (None, None) => {
                for (i, nn) in out_nn.chunks_mut(k).enumerate() {
                    walk_source::<MODE>(mesh, geom, k, max_reach, i, nn, None);
                }
            }
            (None, Some(d2)) => {
                for (i, (nn, dd)) in out_nn.chunks_mut(k).zip(d2.chunks_mut(k)).enumerate() {
                    walk_source::<MODE>(mesh, geom, k, max_reach, i, nn, Some(dd));
                }
            }
            (Some(mask), None) => {
                for (i, nn) in out_nn.chunks_mut(k).enumerate() {
                    if mask[i] {
                        walk_source::<MODE>(mesh, geom, k, max_reach, i, nn, None);
                    }
                }
            }
            (Some(mask), Some(d2)) => {
                for (i, (nn, dd)) in out_nn.chunks_mut(k).zip(d2.chunks_mut(k)).enumerate() {
                    if mask[i] {
                        walk_source::<MODE>(mesh, geom, k, max_reach, i, nn, Some(dd));
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn walk_source<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    i: usize,
    nn: &mut [i32],
    d2: Option<&mut [f64]>,
) {
    let mut heap = KHeap::new(k);
    let [ix, iy, iz] = mesh.bin[i];
    let origin = mesh.frac[i];
    let pi = mesh.folded[i];
    // Nothing visited yet. The first layer is the 3x3x3 around the source.
    let mut prev = [-1i32; 3];
    let mut reach = [1i32; 3];
    loop {
        let mut query = CellQuery {
            heap: &mut heap,
            mesh,
            geom,
            i,
            pi,
            origin,
        };
        let allow_slab = prev[0] >= 0;
        for_new_layer(prev, reach, |dx, dy, dz| {
            visit_cell::<MODE>(&mut query, ix + dx, iy + dy, iz + dz, allow_slab);
        });
        let gaps = [
            axis_gap(origin[0], ix, reach[0], geom.nbin[0], geom.widths[0]),
            axis_gap(origin[1], iy, reach[1], geom.nbin[1], geom.widths[1]),
            axis_gap(origin[2], iz, reach[2], geom.nbin[2], geom.widths[2]),
        ];
        if heap.full() {
            let bound = gaps
                .into_iter()
                .map(|gap| {
                    if gap > 0.0 && gap.is_finite() {
                        gap * gap
                    } else {
                        0.0
                    }
                })
                .fold(f64::INFINITY, f64::min);
            if heap.worst() <= bound {
                break;
            }
        }
        prev = reach;
        let mut grew = false;
        if heap.full() {
            let worst = heap.worst();
            for a in 0..3 {
                let gap2 = if gaps[a] > 0.0 && gaps[a].is_finite() {
                    gaps[a] * gaps[a]
                } else {
                    0.0
                };
                if worst > gap2 && reach[a] < max_reach {
                    reach[a] += 1;
                    grew = true;
                }
            }
        } else {
            for slot in &mut reach {
                if *slot < max_reach {
                    *slot += 1;
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    heap.write_sorted(nn, d2);
}

struct CellQuery<'a> {
    heap: &'a mut KHeap,
    mesh: &'a Mesh,
    geom: &'a Geom,
    i: usize,
    pi: [f64; 3],
    origin: [f64; 3],
}

#[inline(always)]
fn visit_cell<const MODE: u8>(q: &mut CellQuery<'_>, jx: i32, jy: i32, jz: i32, allow_slab: bool) {
    let (cell, na, nb, nc) = q.mesh.locate(jx, jy, jz);
    let lo = q.mesh.offsets[cell];
    let hi = q.mesh.offsets[cell + 1];
    let count = hi - lo;
    if count == 0 {
        return;
    }
    // One orthorhombic occupant is a subtract and three multiplies.
    // The slab test is three divisions, so it only pays for a crowded bin.
    // Mode 0 is orthorhombic: one occupant is a subtract, and the slab
    // test costs more than the distance. Restricted and general cells
    // keep the slab, because the shift is a matrix product.
    if allow_slab && !(MODE == 0 && count == 1) && q.heap.full() {
        let lb = slab_dist2(q.origin, [jx, jy, jz], q.geom.nbin, q.geom.widths);
        if q.heap.worst() <= lb {
            return;
        }
    }
    let shift = if MODE == 0 {
        if (na | nb | nc) == 0 {
            [0.0; 3]
        } else {
            [
                f64::from(na) * q.geom.lengths[0],
                f64::from(nb) * q.geom.lengths[1],
                f64::from(nc) * q.geom.lengths[2],
            ]
        }
    } else if MODE == 1 {
        // Restricted triclinic: a along x, b in the xy plane.
        let fa = f64::from(na);
        let fb = f64::from(nb);
        let fc = f64::from(nc);
        let a = q.geom.cols[0];
        let b = q.geom.cols[1];
        let c = q.geom.cols[2];
        [
            fa * a[0] + fb * b[0] + fc * c[0],
            fb * b[1] + fc * c[1],
            fc * c[2],
        ]
    } else {
        let fa = f64::from(na);
        let fb = f64::from(nb);
        let fc = f64::from(nc);
        let a = q.geom.cols[0];
        let b = q.geom.cols[1];
        let c = q.geom.cols[2];
        [
            fa * a[0] + fb * b[0] + fc * c[0],
            fa * a[1] + fb * b[1] + fc * c[1],
            fa * a[2] + fb * b[2] + fc * c[2],
        ]
    };
    let pi = q.pi;
    for &ju in &q.mesh.occupants[lo..hi] {
        if ju != q.i {
            let p = q.mesh.folded[ju];
            let dx = p[0] + shift[0] - pi[0];
            let dy = p[1] + shift[1] - pi[1];
            let dz = p[2] + shift[2] - pi[2];
            q.heap.push(dx * dx + dy * dy + dz * dz, ju);
        }
    }
}

/// Brute-force k-nearest. Tests and small systems only.
///
/// An orthorhombic box calls [`crate::dist2_ortho_diffs`]: SoA
/// differences, one reciprocal per axis, then the Highway wrap
/// `dr -= box * round(dr / box)`. On a rectangular box that wrap is
/// the Euclidean nearest image. Any other box stays on
/// [`Cell::dist2_euclidean`]: Smith half-edge test, then a
/// Minkowski-reduced 27-image. A hex-prism body diagonal is a
/// fractional wrap that is not the nearest image.
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
    let ortho = simbox.is_ortho();
    let widths = simbox.widths();
    for &i in &active {
        let others: Vec<usize> = active.iter().copied().filter(|&j| j != i).collect();
        let d2 = if ortho {
            ortho_highway_dist2(xyz[i], &others, xyz, widths)?
        } else {
            others
                .iter()
                .map(|&j| pair_dist2(simbox, xyz[i], xyz[j]))
                .collect()
        };
        let mut pairs: Vec<(f64, usize)> = d2.into_iter().zip(others).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        pairs.truncate(k);
        out[i].dist2 = pairs.iter().map(|p| p.0).collect();
        out[i].indices = pairs.iter().map(|p| p.1).collect();
    }
    Ok(out)
}

#[cfg(test)]
mod scale_mesh_tests {
    use super::*;

    #[test]
    fn brute_ortho_matches_euclidean_mic() {
        let (xyz, cell) = ortho_lattice(4);
        let k = 4;
        let brute = knearest_brute(&xyz, &cell, k, None).unwrap();
        let mut best: Vec<(f64, usize)> = (1..xyz.len())
            .map(|j| (cell.dist2_euclidean(xyz[0], xyz[j]), j))
            .collect();
        best.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        best.truncate(k);
        assert_eq!(
            brute[0].indices,
            best.iter().map(|p| p.1).collect::<Vec<_>>()
        );
        for (t, (d, _)) in best.iter().enumerate() {
            assert!((brute[0].dist2[t] - d).abs() < 1e-12);
        }
    }

    fn ortho_lattice(nside: usize) -> (Vec<[f64; 3]>, Cell) {
        let a = 3.125;
        let boxl = nside as f64 * a;
        let cell = Cell::ortho(boxl, boxl, boxl).unwrap();
        let mut xyz = Vec::with_capacity(nside * nside * nside);
        for iz in 0..nside {
            for iy in 0..nside {
                for ix in 0..nside {
                    xyz.push([ix as f64 * a, iy as f64 * a, iz as f64 * a]);
                }
            }
        }
        (xyz, cell)
    }

    fn assert_certified(
        cell: &Cell,
        xyz: &[[f64; 3]],
        active: &[usize],
        i: usize,
        row: &Neighbors,
    ) {
        let k = 4;
        let mut best: Vec<(f64, usize)> = active
            .iter()
            .copied()
            .filter(|&j| j != i)
            .map(|j| (cell.dist2_euclidean(xyz[i], xyz[j]), j))
            .collect();
        best.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
        let kth = best[k - 1].0;
        assert_eq!(row.indices.len(), k);
        for (&d, &j) in row.dist2.iter().zip(&row.indices) {
            let true_d = cell.dist2_euclidean(xyz[i], xyz[j]);
            assert!((d - true_d).abs() <= 1e-8 * true_d.max(1.0));
            assert!(
                d <= kth * (1.0 + 1e-8) + 1e-9,
                "i={i} j={j} d={d} kth={kth}"
            );
        }
        for (d, j) in &best {
            if *d + 1e-8 < kth {
                assert!(
                    row.indices.contains(j),
                    "i={i} missed closer {j} d={d} kth={kth}"
                );
            }
        }
    }

    #[test]
    fn parallel_mesh_matches_euclidean_on_a_large_lattice() {
        // 22^3 = 10648 active points, above the parallel-mesh threshold.
        let (xyz, cell) = ortho_lattice(22);
        let rows = knearest(&xyz, &cell, 4, None, Some(3.0)).unwrap();
        let active: Vec<usize> = (0..xyz.len()).collect();
        for i in (0..xyz.len()).step_by(700) {
            assert_certified(&cell, &xyz, &active, i, &rows[i]);
        }
    }

    #[test]
    fn parallel_mesh_respects_a_mask() {
        let (xyz, cell) = ortho_lattice(26);
        let mask: Vec<bool> = (0..xyz.len()).map(|i| i % 2 == 0).collect();
        let rows = knearest(&xyz, &cell, 4, Some(&mask), Some(3.0)).unwrap();
        let active: Vec<usize> = (0..xyz.len()).filter(|&i| mask[i]).collect();
        assert!(active.len() >= 8_192);
        for &i in active.iter().step_by(900) {
            assert_certified(&cell, &xyz, &active, i, &rows[i]);
            assert!(rows[i + 1].indices.is_empty());
        }
    }
}
