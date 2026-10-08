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
#[derive(Clone)]
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

    /// Empty again, for the next source.
    fn clear(&mut self) {
        self.n = 0;
        self.worst_at = 0;
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
            // Insertion sort: a handful of entries.
            let before = |a: u8, b: u8| {
                let (a, b) = (a as usize, b as usize);
                self.d2[a]
                    .total_cmp(&self.d2[b])
                    .then(self.idx[a].cmp(&self.idx[b]))
                    .is_lt()
            };
            for t in 1..n {
                let x = order[t];
                let mut u = t;
                while u > 0 && before(x, order[u - 1]) {
                    order[u] = order[u - 1];
                    u -= 1;
                }
                order[u] = x;
            }
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
    match (
        block_side(&mesh, nbin),
        walk.is_ortho(),
        walk.is_restricted(),
    ) {
        (Some(side), true, _) => {
            dispatch_blocks::<0>(&mesh, &geom, k, max_reach, side, out_nn, out_d2)
        }
        (Some(side), false, true) => {
            dispatch_blocks::<1>(&mesh, &geom, k, max_reach, side, out_nn, out_d2)
        }
        (Some(side), false, false) => {
            dispatch_blocks::<2>(&mesh, &geom, k, max_reach, side, out_nn, out_d2)
        }
        (None, true, _) => dispatch::<0>(&mesh, &geom, k, max_reach, mask, out_nn, out_d2),
        (None, false, true) => dispatch::<1>(&mesh, &geom, k, max_reach, mask, out_nn, out_d2),
        (None, false, false) => dispatch::<2>(&mesh, &geom, k, max_reach, mask, out_nn, out_d2),
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
    // Masked points are not in the mesh, so every slot is an active source.
    let _ = mask;
    // Sources go in bin order: consecutive sources are neighbours, so their
    // shells share cache lines whatever order the caller's points came in.
    // Source `i` writes its own row, `out[i * k ..]`.
    let nn = RowsOut(out_nn.as_mut_ptr());
    let dd = out_d2.map(|d| RowsOut(d.as_mut_ptr()));
    let [nx, ny, _] = geom.nbin;
    let cell = |c: usize| {
        let c = c as i32;
        [c % nx, (c / nx) % ny, c / (nx * ny)]
    };
    let job = |c: usize| {
        let bin = cell(c);
        for slot in mesh.offsets[c]..mesh.offsets[c + 1] {
            let i = mesh.occupants[slot];
            // Safety: each source owns row `i` of both outputs, and the
            // caller sized them to `n * k`.
            let (row, row_d2) = unsafe { (nn.row(i, k), dd.map(|d| d.row(i, k))) };
            walk_source::<MODE>(mesh, geom, k, max_reach, bin, slot, row, row_d2, None);
        }
    };
    let ncell = mesh.offsets.len() - 1;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..ncell)
            .into_par_iter()
            .with_min_len(64)
            .for_each_init(crate::pop::JobTimer::new, |_timer, c| job(c));
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _timer = crate::pop::JobTimer::new();
        for c in 0..ncell {
            job(c);
        }
    }
}

/// Bins a side of the source blocks of [`dispatch_blocks`] for this mesh:
/// two when bins hold few points, so eight bins share one gather, and one
/// otherwise. `None` when a candidate box would wrap onto itself.
fn block_side(mesh: &Mesh, nbin: [i32; 3]) -> Option<i32> {
    #[cfg(test)]
    if NO_BLOCKS.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let ncell = nbin.iter().map(|&n| n as usize).product::<usize>().max(1);
    let side = if mesh.occupants.len() < 4 * ncell {
        2
    } else {
        1
    };
    nbin.iter().all(|&n| n >= side + 2).then_some(side)
}

/// Tests turn the blocks off to compare against the shell walk.
#[cfg(test)]
static NO_BLOCKS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The points around one block of bins, each moved by its image shift as
/// [`visit_cell`] moves it, and their indices.
#[derive(Default)]
struct Candidates {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    id: Vec<u64>,
    /// One source's distances, read only through a raw pointer.
    #[cfg_attr(not(all(target_arch = "x86_64", linkcell_avx512)), allow(dead_code))]
    d2: Vec<f64>,
}

/// [`dispatch`] by blocks of `side` bins a side: a block's sources share
/// one candidate list, the occupants of the `side + 2` bins a side around
/// the block with their images, gathered once. When a source's k-th
/// neighbour is within the plane bound of the gathered bins, the answer
/// is among the candidates: every other point lies strictly past one of
/// those planes. Distances are formed as [`visit_cell`] forms them and
/// the heap keeps the same k, so every row is the shell walk's; a source
/// the bound does not settle resumes the shell walk from its heap, which
/// holds its 3 x 3 x 3 bins already.
#[allow(clippy::too_many_arguments)]
fn dispatch_blocks<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    side: i32,
    out_nn: &mut [i32],
    out_d2: Option<&mut [f64]>,
) {
    let nn = RowsOut(out_nn.as_mut_ptr());
    let dd = out_d2.map(|d| RowsOut(d.as_mut_ptr()));
    let [nx, ny, nz] = geom.nbin;
    let blocks = [
        (nx + side - 1) / side,
        (ny + side - 1) / side,
        (nz + side - 1) / side,
    ];
    let nblock = (blocks[0] * blocks[1] * blocks[2]) as usize;
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    let wide = std::is_x86_feature_detected!("avx512f");
    let job = |cand: &mut Candidates, heap: &mut KHeap, blk: usize| {
        let blk = blk as i32;
        let x0 = side * (blk % blocks[0]);
        let y0 = side * ((blk / blocks[0]) % blocks[1]);
        let z0 = side * (blk / (blocks[0] * blocks[1]));
        cand.x.clear();
        cand.y.clear();
        cand.z.clear();
        cand.id.clear();
        let interior =
            x0 >= 1 && y0 >= 1 && z0 >= 1 && x0 + side < nx && y0 + side < ny && z0 + side < nz;
        for dz in -1..=side {
            for dy in -1..=side {
                // Inside the box a row of bins is one run of slots with no
                // shift: `p + 0` is `p`.
                if interior {
                    let row = ((z0 + dz) * ny + (y0 + dy)) * nx + x0;
                    let (lo, hi) = (
                        mesh.offsets[(row - 1) as usize],
                        mesh.offsets[(row + side + 1) as usize],
                    );
                    for slot in lo..hi {
                        let p = mesh.slot_folded[slot];
                        cand.x.push(p[0]);
                        cand.y.push(p[1]);
                        cand.z.push(p[2]);
                        cand.id.push(mesh.occupants[slot] as u64);
                    }
                    continue;
                }
                for dx in -1..=side {
                    let (cell, na, nb, nc) = mesh.locate(x0 + dx, y0 + dy, z0 + dz);
                    let shift = image_shift::<MODE>(geom, na, nb, nc);
                    for slot in mesh.offsets[cell]..mesh.offsets[cell + 1] {
                        let p = mesh.slot_folded[slot];
                        cand.x.push(p[0] + shift[0]);
                        cand.y.push(p[1] + shift[1]);
                        cand.z.push(p[2] + shift[2]);
                        cand.id.push(mesh.occupants[slot] as u64);
                    }
                }
            }
        }
        let planes = box_planes([x0, y0, z0], side, geom);
        for iz in z0..(z0 + side).min(nz) {
            for iy in y0..(y0 + side).min(ny) {
                for ix in x0..(x0 + side).min(nx) {
                    let c = ((iz * ny + iy) * nx + ix) as usize;
                    for slot in mesh.offsets[c]..mesh.offsets[c + 1] {
                        let i = mesh.occupants[slot];
                        // Safety: each source owns row `i` of both outputs,
                        // and the caller sized them to `n * k`.
                        let (row, row_d2) = unsafe { (nn.row(i, k), dd.map(|d| d.row(i, k))) };
                        heap.clear();
                        let pi = mesh.slot_folded[slot];
                        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
                        if wide {
                            // Safety: AVX-512F was detected.
                            unsafe { scan_avx512(cand, pi, i, heap) };
                        } else {
                            scan(cand, pi, i, heap);
                        }
                        #[cfg(not(all(target_arch = "x86_64", linkcell_avx512)))]
                        scan(cand, pi, i, heap);
                        let bound = box_bound(mesh.slot_frac[slot], &planes, geom);
                        if heap.full() && heap.worst() <= bound {
                            heap.write_sorted(row, row_d2);
                        } else {
                            walk_source::<MODE>(
                                mesh,
                                geom,
                                k,
                                max_reach,
                                [ix, iy, iz],
                                slot,
                                row,
                                row_d2,
                                Some(heap.clone()),
                            );
                        }
                    }
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..nblock).into_par_iter().with_min_len(8).for_each_init(
            || {
                (
                    crate::pop::JobTimer::new(),
                    Candidates::default(),
                    KHeap::new(k),
                )
            },
            |(_timer, cand, heap), blk| job(cand, heap, blk),
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _timer = crate::pop::JobTimer::new();
        let mut cand = Candidates::default();
        let mut heap = KHeap::new(k);
        for blk in 0..nblock {
            job(&mut cand, &mut heap, blk);
        }
    }
}

/// Fractional planes of the gathered bins, from `corner - 1` to `corner +
/// side` on each axis: `[lower, upper]` per axis.
fn box_planes(corner: [i32; 3], side: i32, geom: &Geom) -> [[f64; 2]; 3] {
    std::array::from_fn(|a| {
        let nf = f64::from(geom.nbin[a]);
        [
            f64::from(corner[a] - 1) / nf,
            f64::from(corner[a] + side + 1) / nf,
        ]
    })
}

/// The squared plane bound of the gathered bins, formed as [`axis_gap`]
/// forms the shell walk's: bins are half-open, so every point not
/// gathered lies strictly past one of these planes.
fn box_bound(origin: [f64; 3], planes: &[[f64; 2]; 3], geom: &Geom) -> f64 {
    (0..3)
        .map(|a| {
            let plus = (planes[a][1] - origin[a]) * geom.widths[a];
            let minus = (origin[a] - planes[a][0]) * geom.widths[a];
            let gap = plus.min(minus);
            if gap > 0.0 && gap.is_finite() {
                gap * gap
            } else {
                0.0
            }
        })
        .fold(f64::INFINITY, f64::min)
}

/// Push every candidate but the source itself.
fn scan(cand: &Candidates, pi: [f64; 3], i: usize, heap: &mut KHeap) {
    for t in 0..cand.x.len() {
        let j = cand.id[t] as usize;
        if j != i {
            let dx = cand.x[t] - pi[0];
            let dy = cand.y[t] - pi[1];
            let dz = cand.z[t] - pi[2];
            heap.push(dx * dx + dy * dy + dz * dz, j);
        }
    }
}

/// [`scan`] eight candidates at a time, in two passes. The first forms
/// every distance and the lane-wise minimum over all of them: eight
/// distances of eight different candidates, so the k-th smallest of them
/// bounds the k-th nearest from above. The second pushes only candidates
/// at or under that bound (and under the heap's worst once it is full),
/// which are all [`KHeap::push`] could keep.
///
/// # Safety
/// AVX-512F is available.
#[allow(clippy::incompatible_msrv)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f")]
unsafe fn scan_avx512(cand: &mut Candidates, pi: [f64; 3], i: usize, heap: &mut KHeap) {
    use std::arch::x86_64::{
        _mm512_add_pd, _mm512_cmp_pd_mask, _mm512_cmpneq_epi64_mask, _mm512_loadu_pd,
        _mm512_mask_blend_pd, _mm512_maskz_loadu_epi64, _mm512_maskz_loadu_pd, _mm512_min_pd,
        _mm512_mul_pd, _mm512_set1_epi64, _mm512_set1_pd, _mm512_storeu_pd, _mm512_sub_pd,
        _CMP_LE_OQ, _CMP_LT_OQ,
    };
    let m = cand.x.len();
    let padded = (m + 7) & !7;
    cand.d2.clear();
    cand.d2.reserve(padded);
    let dp = cand.d2.as_mut_ptr();
    let (px, py, pz) = (
        _mm512_set1_pd(pi[0]),
        _mm512_set1_pd(pi[1]),
        _mm512_set1_pd(pi[2]),
    );
    let me = _mm512_set1_epi64(i as i64);
    let inf = _mm512_set1_pd(f64::INFINITY);
    let mut low = inf;
    let mut t = 0usize;
    while t < m {
        let tm: u8 = if m - t >= 8 {
            0xff
        } else {
            ((1u32 << (m - t)) - 1) as u8
        };
        let dx = _mm512_sub_pd(_mm512_maskz_loadu_pd(tm, cand.x.as_ptr().add(t)), px);
        let dy = _mm512_sub_pd(_mm512_maskz_loadu_pd(tm, cand.y.as_ptr().add(t)), py);
        let dz = _mm512_sub_pd(_mm512_maskz_loadu_pd(tm, cand.z.as_ptr().add(t)), pz);
        let d2 = _mm512_add_pd(
            _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)),
            _mm512_mul_pd(dz, dz),
        );
        let ids = _mm512_maskz_loadu_epi64(tm, cand.id.as_ptr().add(t) as *const i64);
        let keep = _mm512_cmpneq_epi64_mask(ids, me) & tm;
        // The source itself and lanes past the end never count.
        let d2 = _mm512_mask_blend_pd(keep, inf, d2);
        low = _mm512_min_pd(low, d2);
        _mm512_storeu_pd(dp.add(t), d2);
        t += 8;
    }
    let mut bound = f64::INFINITY;
    if heap.k <= 8 {
        let mut lanes = [0.0f64; 8];
        _mm512_storeu_pd(lanes.as_mut_ptr(), low);
        lanes.sort_unstable_by(f64::total_cmp);
        bound = lanes[heap.k - 1];
    }
    let mut t = 0usize;
    while t < padded {
        let limit = if heap.full() {
            heap.worst().min(bound)
        } else {
            bound
        };
        let d2 = _mm512_loadu_pd(dp.add(t));
        let mut take: u8 = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(d2, _mm512_set1_pd(limit))
            & _mm512_cmp_pd_mask::<_CMP_LT_OQ>(d2, inf);
        while take != 0 {
            let l = take.trailing_zeros() as usize;
            take &= take - 1;
            heap.push(*dp.add(t + l), cand.id[t + l] as usize);
        }
        t += 8;
    }
}

/// An output many sources write, one row each.
#[derive(Clone, Copy)]
struct RowsOut<T>(*mut T);
unsafe impl<T: Send> Send for RowsOut<T> {}
unsafe impl<T: Send> Sync for RowsOut<T> {}

impl<T> RowsOut<T> {
    /// Row `i` of width `k`.
    ///
    /// # Safety
    /// The output holds row `i`, and no other job touches it while the
    /// slice lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row<'a>(self, i: usize, k: usize) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(i * k), k)
    }
}

/// The shell walk of one source. `start` is a heap that already holds
/// every point of the source's 3 x 3 x 3 bins (and maybe more): the walk
/// then begins at the next layer, and a point it meets again changes
/// nothing, since the heap keeps one image per point.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn walk_source<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    bin: [i32; 3],
    slot: usize,
    nn: &mut [i32],
    d2: Option<&mut [f64]>,
    start: Option<KHeap>,
) {
    let resumed = start.is_some();
    let mut heap = start.unwrap_or_else(|| KHeap::new(k));
    let [ix, iy, iz] = bin;
    let i = mesh.occupants[slot];
    let origin = mesh.slot_frac[slot];
    let pi = mesh.slot_folded[slot];
    // Nothing visited yet. The first layer is the 3x3x3 around the source.
    let mut prev = [-1i32; 3];
    let mut reach = [1i32; 3];
    let mut first = true;
    loop {
        if !(first && resumed) {
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
        }
        first = false;
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

/// Cartesian shift of image `(na, nb, nc)` of a bin.
#[inline(always)]
fn image_shift<const MODE: u8>(geom: &Geom, na: i32, nb: i32, nc: i32) -> [f64; 3] {
    if MODE == 0 {
        if (na | nb | nc) == 0 {
            [0.0; 3]
        } else {
            [
                f64::from(na) * geom.lengths[0],
                f64::from(nb) * geom.lengths[1],
                f64::from(nc) * geom.lengths[2],
            ]
        }
    } else if MODE == 1 {
        // Restricted triclinic: a along x, b in the xy plane.
        let fa = f64::from(na);
        let fb = f64::from(nb);
        let fc = f64::from(nc);
        let a = geom.cols[0];
        let b = geom.cols[1];
        let c = geom.cols[2];
        [
            fa * a[0] + fb * b[0] + fc * c[0],
            fb * b[1] + fc * c[1],
            fc * c[2],
        ]
    } else {
        let fa = f64::from(na);
        let fb = f64::from(nb);
        let fc = f64::from(nc);
        let a = geom.cols[0];
        let b = geom.cols[1];
        let c = geom.cols[2];
        [
            fa * a[0] + fb * b[0] + fc * c[0],
            fa * a[1] + fb * b[1] + fc * c[1],
            fa * a[2] + fb * b[2] + fc * c[2],
        ]
    }
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
    let shift = image_shift::<MODE>(q.geom, na, nb, nc);
    let pi = q.pi;
    for (&ju, &p) in q.mesh.occupants[lo..hi]
        .iter()
        .zip(&q.mesh.slot_folded[lo..hi])
    {
        if ju != q.i {
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

    #[test]
    fn blocks_write_the_shell_walk_rows() {
        // Lattices a few ulps off (many equal distances, so index order
        // decides), random points, and a sparse box, in an orthorhombic,
        // a restricted, and a general cell; bins of one point and of
        // several, so both block sides run; k up to 20, past the bound
        // of the two-pass scan.
        let mut state = 0x853c_49e6_748f_ea9bu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let cells = [
            Cell::ortho(25.0, 25.0, 25.0).unwrap(),
            Cell::from_vectors(
                [25.0, 0.0, 0.0],
                [4.0, 24.0, 0.0],
                [-3.0, 2.0, 24.5],
                [0.0; 3],
            )
            .unwrap(),
            Cell::from_vectors(
                [24.0, 3.0, -2.0],
                [2.5, 23.5, 3.0],
                [-1.5, 2.0, 24.0],
                [0.5, 0.0, -0.5],
            )
            .unwrap(),
        ];
        for sim in &cells {
            let mut sets: Vec<Vec<[f64; 3]>> = Vec::new();
            let mut lattice = Vec::new();
            for iz in 0..10 {
                for iy in 0..10 {
                    for ix in 0..10 {
                        let mut s = [
                            (ix as f64 + 0.5) / 10.0,
                            (iy as f64 + 0.5) / 10.0,
                            (iz as f64 + 0.5) / 10.0,
                        ];
                        for v in s.iter_mut() {
                            let ulps = (next() % 7) as i64 - 3;
                            *v = f64::from_bits((v.to_bits() as i64 + ulps) as u64);
                        }
                        lattice.push(sim.cartesian(s));
                    }
                }
            }
            sets.push(lattice);
            let unit = |v: u64| (v >> 11) as f64 / (1u64 << 53) as f64;
            sets.push(
                (0..1500)
                    .map(|_| sim.cartesian([unit(next()), unit(next()), unit(next())]))
                    .collect(),
            );
            sets.push(
                (0..60)
                    .map(|_| sim.cartesian([unit(next()), unit(next()), unit(next())]))
                    .collect(),
            );
            for xyz in &sets {
                let mask: Vec<bool> = (0..xyz.len()).map(|t| t % 7 != 3).collect();
                for hint in [Some(2.4), Some(5.0)] {
                    for k in [1, 4, 8, 12, 20] {
                        for mask in [None, Some(mask.as_slice())] {
                            let blocked = knearest(xyz, sim, k, mask, hint).unwrap();
                            NO_BLOCKS.store(true, std::sync::atomic::Ordering::Relaxed);
                            let walked = knearest(xyz, sim, k, mask, hint).unwrap();
                            NO_BLOCKS.store(false, std::sync::atomic::Ordering::Relaxed);
                            for (i, (b, w)) in blocked.iter().zip(&walked).enumerate() {
                                assert_eq!(
                                    b.indices,
                                    w.indices,
                                    "n={} k={k} hint={hint:?} i={i}",
                                    xyz.len()
                                );
                                let bits = |r: &Neighbors| -> Vec<u64> {
                                    r.dist2.iter().map(|d| d.to_bits()).collect()
                                };
                                assert_eq!(
                                    bits(b),
                                    bits(w),
                                    "n={} k={k} hint={hint:?} i={i}",
                                    xyz.len()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
