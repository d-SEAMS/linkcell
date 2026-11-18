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
//! also writes the matching squared distances (`NaN` unused). With the
//! `parallel` feature (the default) an unmasked query of at least 8192
//! points runs on one linkcell thread team on Linux: bin, count, scan, fill,
//! and select, each thread with its own histogram and scratch. Masked queries,
//! smaller queries, and builds without that region still walk under rayon.
//! `--no-default-features` serializes the walk. The per-source `KHeap` stays
//! on the stack for `k <= 16`.

use crate::bins::{self, axis_gap, for_new_layer, slab_dist2, Mesh};
use crate::cell::Cell;
use crate::team::Stage;
use crate::Error;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};
use std::sync::Mutex;

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
    #[cfg(feature = "timing")]
    let started = std::time::Instant::now();
    let found = execute(
        xyz,
        simbox,
        k,
        mask,
        cell_hint,
        out_nn,
        out_d2.as_deref_mut(),
        None,
        &[],
    );
    #[cfg(feature = "timing")]
    if crate::simd::coarse_timing_enabled() {
        crate::simd::add_knn_ns(started.elapsed().as_nanos());
    }
    if found.is_err() {
        out_nn.fill(-1);
        if let Some(d2) = out_d2.as_mut() {
            d2.fill(f64::NAN);
        }
    }
    found
}

/// A pointer the chunks of one region share; each chunk writes slots no
/// other chunk of its stage writes.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// Rows of the graph stages per chunk.
const GRAPH_ROWS: usize = 2048;

/// Buffers of [`knearest_graph`], kept for the next call.
#[derive(Default)]
struct GraphBufs {
    nom: Vec<i32>,
    accepted: Vec<u8>,
    lo: Vec<AtomicU32>,
    hi: Vec<AtomicU32>,
    left: Vec<AtomicU32>,
    start: Vec<u32>,
}

static GRAPH_POOL: Mutex<Option<GraphBufs>> = Mutex::new(None);

/// `true` when row `j` of the packed nominations lists `i`.
#[inline(always)]
fn lists(nom: &[i32], k: usize, j: usize, i: i32) -> bool {
    nom[j * k..(j + 1) * k].contains(&i)
}

/// The k-nearest graph as CSR, built in the search's region.
///
/// Row `i` of the nominations is its `k` nearest, nearest first. With
/// `mutual`, row `i` holds every `j` that row `i` lists and that lists
/// `i`: the lower indices ascending, then the higher ones nearest first.
/// Otherwise an edge comes from the nomination slot `s = i * k + t` of
/// the lower index when both rows list each other, and from the only
/// slot that names it when one does; row `i` holds its edges in slot
/// order: the edges from lower rows ascending, then its own nearest
/// first, then the edges from higher rows ascending.
///
/// `offsets` has `n + 1` entries. `alloc(total)` is called once, from
/// any worker, and returns `total` writable ints for the columns, or
/// null ([`Error::BufferSize`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn knearest_graph(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    mutual: bool,
    offsets: &mut [i32],
    alloc: &(dyn Fn(usize) -> *mut i32 + Sync),
) -> Result<(), Error> {
    let n = xyz.len();
    if offsets.len() != n + 1 {
        return Err(Error::BufferSize);
    }
    let Some(nk) = n.checked_mul(k) else {
        return Err(Error::Overflow);
    };
    if n > i32::MAX as usize {
        return Err(Error::Overflow);
    }
    let mut bufs = GRAPH_POOL
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .take()
        .unwrap_or_default();
    bufs.nom.resize(nk, -1);
    bufs.accepted.resize(nk, 0);
    bufs.start.resize(n, 0);
    for v in [&mut bufs.lo, &mut bufs.hi, &mut bufs.left] {
        v.truncate(n);
        v.resize_with(n, || AtomicU32::new(0));
    }
    let failed = AtomicBool::new(false);
    let overflow = AtomicBool::new(false);
    let cols = AtomicPtr::new(std::ptr::null_mut::<i32>());
    let result = {
        let GraphBufs {
            nom,
            accepted,
            lo,
            hi,
            left,
            start,
        } = &mut bufs;
        let nom_p = Shared(nom.as_mut_ptr());
        let acc_p = Shared(accepted.as_mut_ptr());
        let start_p = Shared(start.as_mut_ptr());
        let off_p = Shared(offsets.as_mut_ptr());
        let (lo, hi, left) = (&lo[..], &hi[..], &left[..]);
        // SAFETY (every stage below): the scan has written every row of
        // `nom` before the first graph stage starts, and chunk `c` owns
        // rows `c * GRAPH_ROWS ..` of `accepted`, `start`, and `offsets`.
        let nom_ro = move || unsafe { std::slice::from_raw_parts(nom_p.get(), nk) };
        let rows = |c: usize| c * GRAPH_ROWS..((c + 1) * GRAPH_ROWS).min(n);
        let chunks = n.div_ceil(GRAPH_ROWS);
        let prep = |r: std::ops::Range<usize>| {
            for i in r {
                lo[i].store(0, Ordering::Relaxed);
                hi[i].store(0, Ordering::Relaxed);
            }
        };
        let degree = |_: &mut Scratch, c: usize| {
            let nom = nom_ro();
            for i in rows(c) {
                let row = &nom[i * k..(i + 1) * k];
                let mut own = 0u32;
                for (t, &j) in row.iter().enumerate() {
                    let take = j >= 0
                        && (j as usize) < n
                        && j as usize != i
                        && if mutual {
                            lists(nom, k, j as usize, i as i32)
                        } else {
                            (i as i32) < j || !lists(nom, k, j as usize, i as i32)
                        };
                    if !mutual {
                        unsafe { acc_p.get().add(i * k + t).write(u8::from(take)) };
                    }
                    if !take {
                        continue;
                    }
                    own += 1;
                    if !mutual {
                        if (i as i32) < j {
                            lo[j as usize].fetch_add(1, Ordering::Relaxed);
                        } else {
                            hi[j as usize].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                unsafe { start_p.get().add(i).write(own) };
            }
        };
        let prefix = |_: &mut Scratch, _: usize| {
            // SAFETY: one chunk; the degree stage has finished.
            let (off, start) = unsafe {
                (
                    std::slice::from_raw_parts_mut(off_p.get(), n + 1),
                    std::slice::from_raw_parts_mut(start_p.get(), n),
                )
            };
            let mut acc = 0u64;
            for i in 0..n {
                off[i] = acc as i32;
                let own = u64::from(start[i]);
                if mutual {
                    acc += own;
                } else {
                    let l = lo[i].load(Ordering::Relaxed);
                    let h = hi[i].load(Ordering::Relaxed);
                    start[i] = (acc + u64::from(l)) as u32;
                    left[i].store(l + h, Ordering::Relaxed);
                    acc += u64::from(l) + own + u64::from(h);
                }
                if acc > i32::MAX as u64 {
                    overflow.store(true, Ordering::Relaxed);
                    return;
                }
            }
            off[n] = acc as i32;
            let p = alloc(acc as usize);
            if p.is_null() && acc > 0 {
                failed.store(true, Ordering::Relaxed);
                return;
            }
            cols.store(p, Ordering::Relaxed);
        };
        let fill = |s: &mut Scratch, c: usize| {
            let out = cols.load(Ordering::Relaxed);
            if out.is_null() {
                return;
            }
            let nom = nom_ro();
            // SAFETY: the prefix stage has finished; `out` holds
            // `offsets[n]` ints and each write below lands in a slot of
            // one row that no other write fills.
            let off = unsafe { std::slice::from_raw_parts(off_p.get(), n + 1) };
            let start = unsafe { std::slice::from_raw_parts(start_p.get(), n) };
            let keep = |i: usize, j: i32| {
                j >= 0 && (j as usize) < n && j as usize != i && lists(nom, k, j as usize, i as i32)
            };
            if mutual {
                for i in rows(c) {
                    let row = &nom[i * k..(i + 1) * k];
                    let low = &mut s.rows;
                    low.clear();
                    low.extend(
                        row.iter()
                            .copied()
                            .filter(|&j| (j as usize) < i && keep(i, j)),
                    );
                    low.sort_unstable();
                    let mut put = off[i] as usize;
                    for &j in low.iter() {
                        unsafe { out.add(put).write(j) };
                        put += 1;
                    }
                    for &j in row {
                        if j as usize > i && keep(i, j) {
                            unsafe { out.add(put).write(j) };
                            put += 1;
                        }
                    }
                }
                return;
            }
            // SAFETY: the degree stage has finished.
            let acc = unsafe { std::slice::from_raw_parts(acc_p.get(), nk) };
            let finish = |j: usize| {
                let own = acc[j * k..(j + 1) * k].iter().filter(|&&a| a != 0).count();
                let (b, m, e) = (off[j] as usize, start[j] as usize, off[j + 1] as usize);
                // SAFETY: every edge from another row into row `j` is
                // written (`left` reached zero), and row `j`'s own edges
                // lie between these two runs.
                unsafe {
                    std::slice::from_raw_parts_mut(out.add(b), m - b).sort_unstable();
                    std::slice::from_raw_parts_mut(out.add(m + own), e - m - own).sort_unstable();
                }
            };
            for i in rows(c) {
                let mut put = start[i] as usize;
                for t in 0..k {
                    if acc[i * k + t] == 0 {
                        continue;
                    }
                    let j = nom[i * k + t] as usize;
                    unsafe { out.add(put).write(j as i32) };
                    put += 1;
                    let at = if i < j {
                        off[j] as usize + lo[j].fetch_sub(1, Ordering::Relaxed) as usize - 1
                    } else {
                        off[j + 1] as usize - hi[j].fetch_sub(1, Ordering::Relaxed) as usize
                    };
                    unsafe { out.add(at).write(i as i32) };
                    if left[j].fetch_sub(1, Ordering::AcqRel) == 1 {
                        finish(j);
                    }
                }
            }
        };
        let tail = [
            Stage {
                chunks,
                run: &degree,
            },
            Stage {
                chunks: 1,
                run: &prefix,
            },
            Stage { chunks, run: &fill },
        ];
        let nn = unsafe { std::slice::from_raw_parts_mut(nom_p.get(), nk) };
        execute(
            xyz,
            simbox,
            k,
            mask,
            cell_hint,
            nn,
            None,
            Some(&prep),
            &tail,
        )
    };
    *GRAPH_POOL.lock().unwrap_or_else(|err| err.into_inner()) = Some(bufs);
    result?;
    if overflow.load(Ordering::Relaxed) {
        return Err(Error::Overflow);
    }
    if failed.load(Ordering::Relaxed) {
        return Err(Error::BufferSize);
    }
    Ok(())
}

/// The k-nearest graph as CSR `(offsets, columns)`; see the C entry
/// `lc_knearest_csr` for the row order.
pub fn knearest_csr(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    mutual: bool,
) -> Result<(Vec<i32>, Vec<i32>), Error> {
    let mut offsets = vec![0i32; xyz.len() + 1];
    let cols = Mutex::new(Vec::<i32>::new());
    let alloc = |total: usize| {
        let mut guard = cols.lock().unwrap_or_else(|err| err.into_inner());
        guard.resize(total, 0);
        guard.as_mut_ptr()
    };
    knearest_graph(
        xyz,
        simbox,
        k,
        mask,
        cell_hint,
        mutual,
        &mut offsets,
        &alloc,
    )?;
    let cols = cols.into_inner().unwrap_or_else(|err| err.into_inner());
    Ok((offsets, cols))
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

#[derive(Clone, Copy)]
struct Geom {
    cols: [[f64; 3]; 3],
    lengths: [f64; 3],
    nbin: [i32; 3],
    widths: [f64; 3],
}

/// Threads in the last k-nearest team. Zero before the first query.
static TEAM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Opt-in placement of the k-nearest team. Off unless a caller asks.
static PIN_LEAST_LOADED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Place the k-nearest team workers on the least-loaded CPUs.
///
/// Off by default. A query does not read CPU counters or change affinity
/// until this is set, and it does not consult an environment variable.
/// Turning it off restores each worker to the mask it had before the pin.
/// A team of one is the calling thread, pinned for the query only.
/// Builds without the team query store the flag and leave placement alone.
pub fn pin_knearest_to_least_loaded(enabled: bool) {
    PIN_LEAST_LOADED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Threads that ran the last k-nearest query.
///
/// Zero before the first query. This is the k-nearest team when it ran,
/// and the rayon pool size otherwise. The team is sized as libgomp sizes
/// a parallel region the caller opens: `omp_get_max_threads()`, trimmed
/// by the load average under `OMP_DYNAMIC` and capped by
/// `OMP_THREAD_LIMIT`. Its stacks follow `OMP_STACKSIZE`. It is resized
/// on the next query when that size changes. A team of one is the calling
/// thread, with no worker.
pub fn knearest_team_threads() -> usize {
    TEAM.load(std::sync::atomic::Ordering::Relaxed)
}

/// Join the standing k-nearest team.
///
/// The image destructor does this on unload and on process exit. Calling
/// it directly joins the workers before a `dlclose` and leaves no thread
/// behind. A later query starts a new team. Safe to call more than once.
pub fn shutdown_knearest_team() {
    #[cfg(all(feature = "parallel", linkcell_openmp))]
    unsafe {
        lc_knn_shutdown_impl();
    }
}

/// C ABI for the same join, so a `dlclose` of this image can stop the team
/// first. The image destructor calls the implementation directly.
#[cfg(all(feature = "parallel", linkcell_openmp))]
#[no_mangle]
pub extern "C" fn lc_knn_shutdown() {
    unsafe { lc_knn_shutdown_impl() }
}

#[cfg(all(feature = "parallel", linkcell_openmp))]
extern "C" {
    fn lc_knn_shutdown_impl();
}

#[cfg(all(test, feature = "parallel", linkcell_openmp))]
thread_local! {
    static FORCE_FUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FORCE_LEGACY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(feature = "parallel", linkcell_openmp))]
fn force_fused() -> bool {
    #[cfg(test)]
    {
        FORCE_FUSED.with(|cell| cell.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

#[cfg(all(feature = "parallel", linkcell_openmp))]
fn force_legacy() -> bool {
    #[cfg(test)]
    {
        FORCE_LEGACY.with(|cell| cell.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Per-worker buffers of one search region.
#[derive(Default)]
pub(crate) struct Scratch {
    cand: Candidates,
    heap: Option<KHeap>,
    /// Row scratch for stages appended to the search.
    pub(crate) rows: Vec<i32>,
}

/// Units of the scan (blocks or bins) per chunk: about sixteen chunks a
/// worker, so a worker the scheduler parks holds a small share.
fn scan_chunk(units: usize, width: usize) -> usize {
    (units / (16 * width.max(1))).max(1)
}

#[allow(clippy::too_many_arguments)]
fn execute(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    k: usize,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    out_nn: &mut [i32],
    mut out_d2: Option<&mut [f64]>,
    prep: Option<&(dyn Fn(std::ops::Range<usize>) + Sync)>,
    tail: &[Stage<'_, Scratch>],
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
        out_nn.fill(-1);
        if let Some(d2) = out_d2.as_mut() {
            d2.fill(f64::NAN);
        }
        if let Some(prep) = prep {
            prep(0..n);
        }
        crate::team::run(1, None, tail);
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
    #[cfg(all(feature = "parallel", linkcell_openmp))]
    if active_owned.is_none()
        && prep.is_none()
        && tail.is_empty()
        && !crate::team::installed()
        && !crate::on_caller()
        && !force_legacy()
        && (xyz.len() >= bins::PARALLEL_MESH || force_fused())
    {
        // The region writes only the slots it finds.
        out_nn.fill(-1);
        if let Some(d2) = out_d2.as_mut() {
            d2.fill(f64::NAN);
        }
        return fused::run(xyz, &walk, k, edge, out_nn, out_d2);
    }
    let nn = RowsOut(out_nn.as_mut_ptr());
    let dd = out_d2.map(|d| RowsOut(d.as_mut_ptr()));
    let mode = match (walk.is_ortho(), walk.is_restricted()) {
        (true, _) => 0u8,
        (false, true) => 1,
        (false, false) => 2,
    };
    crate::team::with_current(|width, spawn| {
        if width <= 1 || spawn.is_none() {
            if let Some(m) = mask {
                for (i, &on) in m.iter().enumerate() {
                    if !on {
                        // SAFETY: row `i` of an `n * k` output.
                        unsafe {
                            nn.row(i, k).fill(-1);
                            if let Some(d) = dd {
                                d.row(i, k).fill(f64::NAN);
                            }
                        }
                    }
                }
            }
            if let Some(prep) = prep {
                prep(0..n);
            }
            let mesh = timed_mesh(|| Mesh::build(xyz, &walk, active_owned.as_deref(), edge))?;
            let nbin = [mesh.nx, mesh.ny, mesh.nz];
            let geom = geom_of(&walk, nbin, mesh.widths);
            let max_reach = reach_of(&walk, &geom, mesh.image_reach());
            let side = block_side(mesh.occupants.len(), nbin);
            let (units, _) = scan_units(nbin, side);
            let mut scratch = Scratch::default();
            match mode {
                0 => serial_scan::<0>(
                    &mesh,
                    &geom,
                    k,
                    max_reach,
                    side,
                    nn,
                    dd,
                    units,
                    &mut scratch,
                ),
                1 => serial_scan::<1>(
                    &mesh,
                    &geom,
                    k,
                    max_reach,
                    side,
                    nn,
                    dd,
                    units,
                    &mut scratch,
                ),
                _ => serial_scan::<2>(
                    &mesh,
                    &geom,
                    k,
                    max_reach,
                    side,
                    nn,
                    dd,
                    units,
                    &mut scratch,
                ),
            }
            drop(mesh);
            crate::team::run(1, None, tail);
            return Ok(());
        }
        let stages = bins::MeshStages::new(xyz, &walk, active_owned.as_deref(), edge, width)?;
        let nbin = stages.nbin();
        let geom = geom_of(&walk, nbin, stages.widths());
        let max_reach = reach_of(
            &walk,
            &geom,
            nbin.iter().copied().max().unwrap_or(1) / 2 + 2,
        );
        let side = block_side(stages.n_active(), nbin);
        let (units, _) = scan_units(nbin, side);
        let per = scan_chunk(units, width);
        let fold_chunks = stages.fold_chunks();
        let fold = |_: &mut Scratch, c: usize| {
            let _timer = crate::pop::JobTimer::new();
            stages.fold(c);
            let lo = n * c / fold_chunks;
            let hi = n * (c + 1) / fold_chunks;
            if let Some(prep) = prep {
                prep(lo..hi);
            }
            if let Some(m) = mask {
                for i in (lo..hi).filter(|&i| !m[i]) {
                    // SAFETY: chunk `c` alone fills masked rows `lo..hi`.
                    unsafe {
                        nn.row(i, k).fill(-1);
                        if let Some(d) = dd {
                            d.row(i, k).fill(f64::NAN);
                        }
                    }
                }
            }
        };
        let prefix = |_: &mut Scratch, _: usize| stages.prefix();
        let scatter = |_: &mut Scratch, t: usize| {
            let _timer = crate::pop::JobTimer::new();
            stages.scatter(t);
        };
        let mesh = stages.mesh();
        let geom = &geom;
        let path = crate::simd::select();
        let scan = move |s: &mut Scratch, c: usize| {
            let _timer = crate::pop::JobTimer::new();
            let lo = c * per;
            let hi = (lo + per).min(units);
            match mode {
                0 => scan_range::<0>(mesh, geom, k, max_reach, side, nn, dd, lo..hi, path, s),
                1 => scan_range::<1>(mesh, geom, k, max_reach, side, nn, dd, lo..hi, path, s),
                _ => scan_range::<2>(mesh, geom, k, max_reach, side, nn, dd, lo..hi, path, s),
            }
        };
        let mut list: Vec<Stage<'_, Scratch>> = vec![
            Stage {
                chunks: fold_chunks,
                run: &fold,
            },
            Stage {
                chunks: 1,
                run: &prefix,
            },
            Stage {
                chunks: stages.scatter_chunks(),
                run: &scatter,
            },
            Stage {
                chunks: units.div_ceil(per),
                run: &scan,
            },
        ];
        list.extend(tail.iter().map(|t| Stage {
            chunks: t.chunks,
            run: t.run,
        }));
        crate::team::run(width, spawn, &list);
        Ok(())
    })
}

fn timed_mesh(build: impl FnOnce() -> Result<Mesh, Error>) -> Result<Mesh, Error> {
    #[cfg(feature = "timing")]
    if crate::simd::coarse_timing_enabled() {
        let started = std::time::Instant::now();
        let mesh = build()?;
        crate::simd::add_mesh_ns(started.elapsed().as_nanos());
        return Ok(mesh);
    }
    build()
}

fn geom_of(walk: &Cell, nbin: [i32; 3], widths: [f64; 3]) -> Geom {
    Geom {
        cols: [walk.a(), walk.b(), walk.c()],
        lengths: walk.widths(),
        nbin,
        widths,
    }
}

/// Shell walk of every source in bin `c`.
#[cfg(all(feature = "parallel", linkcell_openmp))]
fn walk_cell<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    bin: [i32; 3],
    c: usize,
    out: (RowsOut<i32>, Option<RowsOut<f64>>),
) {
    let (nn, dd) = out;
    for slot in mesh.offsets[c]..mesh.offsets[c + 1] {
        let i = mesh.occupants[slot];
        // Safety: each source owns row `i` of both outputs, and the caller
        // sized them to `n * k`. No other slot writes that row.
        let (row, row_d2) = unsafe { (nn.row(i, k), dd.map(|d| d.row(i, k))) };
        walk_source::<MODE>(mesh, geom, k, max_reach, bin, slot, row, row_d2, None);
    }
}

/// Shell walk of the cells in `begin..end`.
#[cfg(all(feature = "parallel", linkcell_openmp))]
fn walk_cells<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    out: (RowsOut<i32>, Option<RowsOut<f64>>),
    begin: usize,
    end: usize,
) {
    let [nx, ny, _] = geom.nbin;
    for c in begin..end {
        let c32 = c as i32;
        let bin = [c32 % nx, (c32 / nx) % ny, c32 / (nx * ny)];
        walk_cell::<MODE>(mesh, geom, k, max_reach, bin, c, out);
    }
}

fn reach_of(walk: &Cell, geom: &Geom, image_reach: i32) -> i32 {
    let [nx, ny, nz] = geom.nbin;
    let widths = geom.widths;
    let cell_min = (widths[0] / f64::from(nx))
        .min(widths[1] / f64::from(ny))
        .min(widths[2] / f64::from(nz));
    let need = (box_diameter(walk) / cell_min).ceil();
    if need.is_finite() && need < 1.0e7 {
        (need as i32).saturating_add(3).max(image_reach)
    } else {
        image_reach
    }
}

/// Scan units, blocks of `side` bins or single bins, and the blocks per
/// axis.
fn scan_units(nbin: [i32; 3], side: Option<i32>) -> (usize, [i32; 3]) {
    let [nx, ny, nz] = nbin;
    match side {
        Some(side) => {
            let blocks = [
                (nx + side - 1) / side,
                (ny + side - 1) / side,
                (nz + side - 1) / side,
            ];
            ((blocks[0] * blocks[1] * blocks[2]) as usize, blocks)
        }
        None => ((nx as usize) * (ny as usize) * (nz as usize), nbin),
    }
}

#[allow(clippy::too_many_arguments)]
fn serial_scan<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    side: Option<i32>,
    nn: RowsOut<i32>,
    dd: Option<RowsOut<f64>>,
    units: usize,
    scratch: &mut Scratch,
) {
    let _timer = crate::pop::JobTimer::new();
    let path = crate::simd::select();
    scan_range::<MODE>(
        mesh,
        geom,
        k,
        max_reach,
        side,
        nn,
        dd,
        0..units,
        path,
        scratch,
    );
}

/// Scan units `range`: blocks through [`scan_block`] when `side` is set,
/// bins through the shell walk otherwise.
#[allow(clippy::too_many_arguments)]
fn scan_range<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    side: Option<i32>,
    nn: RowsOut<i32>,
    dd: Option<RowsOut<f64>>,
    range: std::ops::Range<usize>,
    path: crate::simd::Path,
    scratch: &mut Scratch,
) {
    match side {
        Some(side) => {
            let (_, blocks) = scan_units(geom.nbin, Some(side));
            let Scratch { cand, heap, .. } = scratch;
            let heap = heap.get_or_insert_with(|| KHeap::new(k));
            for blk in range {
                scan_block::<MODE>(
                    mesh, geom, k, max_reach, side, blocks, path, nn, dd, cand, heap, blk,
                );
            }
        }
        None => {
            let [nx, ny, _] = geom.nbin;
            for c in range {
                let ci = c as i32;
                let bin = [ci % nx, (ci / nx) % ny, ci / (nx * ny)];
                for slot in mesh.offsets[c]..mesh.offsets[c + 1] {
                    let i = mesh.occupants[slot];
                    // Safety: each source owns row `i` of both outputs, and the
                    // caller sized them to `n * k`.
                    let (row, mut row_d2) = unsafe { (nn.row(i, k), dd.map(|d| d.row(i, k))) };
                    clear_row(row, row_d2.as_deref_mut());
                    walk_source::<MODE>(mesh, geom, k, max_reach, bin, slot, row, row_d2, None);
                }
            }
        }
    }
}

/// Every slot of a row unused: the search writes the found prefix.
#[inline(always)]
fn clear_row(row: &mut [i32], d2: Option<&mut [f64]>) {
    row.fill(-1);
    if let Some(d2) = d2 {
        d2.fill(f64::NAN);
    }
}

/// Bins a side of the source blocks of [`scan_block`] for this mesh:
/// two when bins hold few points, so eight bins share one gather, and one
/// otherwise. `None` when a candidate box would wrap onto itself.
fn block_side(n_active: usize, nbin: [i32; 3]) -> Option<i32> {
    #[cfg(test)]
    if NO_BLOCKS.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let ncell = nbin.iter().map(|&n| n as usize).product::<usize>().max(1);
    let side = if n_active < 4 * ncell { 2 } else { 1 };
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
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code)
    )]
    d2: Vec<f64>,
}

/// Shared inputs for one block scan. The candidate list and heap stay
/// per thread.
#[cfg(all(feature = "parallel", linkcell_openmp))]
struct BlockScan<'a> {
    mesh: &'a Mesh,
    geom: &'a Geom,
    k: usize,
    max_reach: i32,
    side: i32,
    blocks: [i32; 3],
    path: crate::simd::Path,
    time_scan: bool,
    nn: RowsOut<i32>,
    dd: Option<RowsOut<f64>>,
}

/// One block of the shared-candidate scan. `blk` is the block index.
#[cfg(all(feature = "parallel", linkcell_openmp))]
fn select_one_block<const MODE: u8>(
    scan: &BlockScan<'_>,
    cand: &mut Candidates,
    heap: &mut KHeap,
    blk: usize,
) {
    let mesh = scan.mesh;
    let geom = scan.geom;
    let k = scan.k;
    let max_reach = scan.max_reach;
    let side = scan.side;
    let blocks = scan.blocks;
    let path = scan.path;
    let time_scan = scan.time_scan;
    let nn = scan.nn;
    let dd = scan.dd;
    let [nx, ny, nz] = geom.nbin;
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
                    #[cfg(feature = "timing")]
                    if time_scan {
                        let started = std::time::Instant::now();
                        scan_at(path, cand, pi, i, heap);
                        crate::simd::add_scan_ns(started.elapsed().as_nanos());
                    } else {
                        scan_at(path, cand, pi, i, heap);
                    }
                    #[cfg(not(feature = "timing"))]
                    {
                        let _ = time_scan;
                        scan_at(path, cand, pi, i, heap);
                    }
                    let bound = box_bound(mesh.slot_frac[slot], &planes, geom);
                    if heap.full() && heap.worst() < bound {
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
}

/// One block of `side` bins a side: a block's sources share one
/// candidate list, the occupants of the `side + 2` bins a side around
/// the block with their images, gathered once. When a source's k-th
/// neighbour is closer than the plane bound of the gathered bins, the answer
/// is among the candidates: every other point lies strictly past one of
/// those planes. Distances are formed as [`visit_cell`] forms them and
/// the heap keeps the same k, so every row is the shell walk's; a source
/// the bound does not settle resumes the shell walk from its heap, which
/// holds its 3 x 3 x 3 bins already.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scan_block<const MODE: u8>(
    mesh: &Mesh,
    geom: &Geom,
    k: usize,
    max_reach: i32,
    side: i32,
    blocks: [i32; 3],
    path: crate::simd::Path,
    nn: RowsOut<i32>,
    dd: Option<RowsOut<f64>>,
    cand: &mut Candidates,
    heap: &mut KHeap,
    blk: usize,
) {
    let [nx, ny, nz] = geom.nbin;
    #[cfg(feature = "timing")]
    let time_scan = crate::simd::scan_timing_enabled();
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
                    let (row, mut row_d2) = unsafe { (nn.row(i, k), dd.map(|d| d.row(i, k))) };
                    clear_row(row, row_d2.as_deref_mut());
                    heap.clear();
                    let pi = mesh.slot_folded[slot];
                    #[cfg(feature = "timing")]
                    if time_scan {
                        let started = std::time::Instant::now();
                        scan_at(path, cand, pi, i, heap);
                        crate::simd::add_scan_ns(started.elapsed().as_nanos());
                    } else {
                        scan_at(path, cand, pi, i, heap);
                    }
                    #[cfg(not(feature = "timing"))]
                    scan_at(path, cand, pi, i, heap);
                    let bound = box_bound(mesh.slot_frac[slot], &planes, geom);
                    if heap.full() && heap.worst() < bound {
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

/// [`scan`] on `path`. [`crate::simd::select`] already rejected a path
/// this CPU cannot run. A stale path falls through to [`scan`].
fn scan_at(
    path: crate::simd::Path,
    cand: &mut Candidates,
    pi: [f64; 3],
    i: usize,
    heap: &mut KHeap,
) {
    match path {
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        crate::simd::Path::Avx512 => {
            debug_assert!(std::is_x86_feature_detected!("avx512f"));
            // Safety: `select` only returns this path when AVX-512F is available.
            unsafe { scan_avx512(cand, pi, i, heap) };
        }
        #[cfg(target_arch = "x86_64")]
        crate::simd::Path::Avx2 => {
            debug_assert!(
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            );
            // Safety: `select` only returns this path when AVX2 and FMA are available.
            // FMA is not enabled on the kernel.
            unsafe { scan_avx2(cand, pi, i, heap) };
        }
        #[cfg(target_arch = "aarch64")]
        crate::simd::Path::Neon => {
            // Safety: NEON is baseline on aarch64, and `select` only
            // returns this path when that target feature is on.
            unsafe { scan_neon(cand, pi, i, heap) };
        }
        _ => scan(cand, pi, i, heap),
    }
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

/// [`scan`] four candidates at a time. Sums are separate multiplies and
/// adds, the same order as [`scan`], so each distance is the same bits.
/// The lane-wise minimums bound the push when `k` fits in four lanes.
///
/// # Safety
/// AVX2 is available. FMA is a selection gate and is not enabled here.
#[allow(clippy::incompatible_msrv)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn scan_avx2(cand: &mut Candidates, pi: [f64; 3], i: usize, heap: &mut KHeap) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_and_pd, _mm256_blendv_pd, _mm256_castsi256_pd, _mm256_cmp_pd,
        _mm256_cmpeq_epi64, _mm256_loadu_pd, _mm256_loadu_si256, _mm256_min_pd, _mm256_movemask_pd,
        _mm256_mul_pd, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
        _CMP_LE_OQ, _CMP_LT_OQ,
    };
    let m = cand.x.len();
    let padded = (m + 3) & !3;
    cand.d2.clear();
    cand.d2.reserve(padded);
    let dp = cand.d2.as_mut_ptr();
    let (px, py, pz) = (
        _mm256_set1_pd(pi[0]),
        _mm256_set1_pd(pi[1]),
        _mm256_set1_pd(pi[2]),
    );
    let me = _mm256_set1_epi64x(i as i64);
    let inf = _mm256_set1_pd(f64::INFINITY);
    let mut low = inf;
    let mut t = 0usize;
    while t < m {
        let (xv, yv, zv, ids) = if m - t >= 4 {
            (
                _mm256_loadu_pd(cand.x.as_ptr().add(t)),
                _mm256_loadu_pd(cand.y.as_ptr().add(t)),
                _mm256_loadu_pd(cand.z.as_ptr().add(t)),
                _mm256_loadu_si256(cand.id.as_ptr().add(t).cast()),
            )
        } else {
            let n = m - t;
            let mut x = [0.0f64; 4];
            let mut y = [0.0f64; 4];
            let mut z = [0.0f64; 4];
            let mut id = [i as u64; 4];
            x[..n].copy_from_slice(&cand.x[t..m]);
            y[..n].copy_from_slice(&cand.y[t..m]);
            z[..n].copy_from_slice(&cand.z[t..m]);
            id[..n].copy_from_slice(&cand.id[t..m]);
            (
                _mm256_loadu_pd(x.as_ptr()),
                _mm256_loadu_pd(y.as_ptr()),
                _mm256_loadu_pd(z.as_ptr()),
                _mm256_loadu_si256(id.as_ptr().cast()),
            )
        };
        let dx = _mm256_sub_pd(xv, px);
        let dy = _mm256_sub_pd(yv, py);
        let dz = _mm256_sub_pd(zv, pz);
        let d2 = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
            _mm256_mul_pd(dz, dz),
        );
        let eq = _mm256_cmpeq_epi64(ids, me);
        let d2 = _mm256_blendv_pd(d2, inf, _mm256_castsi256_pd(eq));
        low = _mm256_min_pd(low, d2);
        _mm256_storeu_pd(dp.add(t), d2);
        t += 4;
    }
    let mut bound = f64::INFINITY;
    if (1..=4).contains(&heap.k) && m > 0 {
        let mut lanes = [0.0f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), low);
        lanes.sort_unstable_by(f64::total_cmp);
        bound = lanes[heap.k - 1];
    }
    let mut t = 0usize;
    while t < m {
        let limit = if heap.full() {
            heap.worst().min(bound)
        } else {
            bound
        };
        let d2 = _mm256_loadu_pd(dp.add(t));
        let le = _mm256_cmp_pd::<_CMP_LE_OQ>(d2, _mm256_set1_pd(limit));
        let lt = _mm256_cmp_pd::<_CMP_LT_OQ>(d2, inf);
        let mut take = _mm256_movemask_pd(_mm256_and_pd(le, lt));
        while take != 0 {
            let l = take.trailing_zeros() as usize;
            take &= take - 1;
            if t + l < m {
                heap.push(*dp.add(t + l), cand.id[t + l] as usize);
            }
        }
        t += 4;
    }
}

/// [`scan`] two candidates at a time. The arithmetic matches [`scan`].
///
/// # Safety
/// NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scan_neon(cand: &mut Candidates, pi: [f64; 3], i: usize, heap: &mut KHeap) {
    use std::arch::aarch64::{
        vaddq_f64, vandq_u64, vbslq_f64, vceqq_u64, vcleq_f64, vcltq_f64, vdupq_n_f64, vdupq_n_u64,
        vgetq_lane_f64, vgetq_lane_u64, vld1q_f64, vld1q_u64, vminq_f64, vmulq_f64, vst1q_f64,
        vsubq_f64,
    };
    let m = cand.x.len();
    let padded = (m + 1) & !1;
    cand.d2.clear();
    cand.d2.reserve(padded);
    let dp = cand.d2.as_mut_ptr();
    let (px, py, pz) = (vdupq_n_f64(pi[0]), vdupq_n_f64(pi[1]), vdupq_n_f64(pi[2]));
    let me = vdupq_n_u64(i as u64);
    let inf = vdupq_n_f64(f64::INFINITY);
    let mut low = inf;
    let mut t = 0usize;
    while t < m {
        let (xv, yv, zv, ids) = if m - t >= 2 {
            (
                vld1q_f64(cand.x.as_ptr().add(t)),
                vld1q_f64(cand.y.as_ptr().add(t)),
                vld1q_f64(cand.z.as_ptr().add(t)),
                vld1q_u64(cand.id.as_ptr().add(t)),
            )
        } else {
            let mut x = [0.0f64; 2];
            let mut y = [0.0f64; 2];
            let mut z = [0.0f64; 2];
            let mut id = [i as u64; 2];
            x[0] = cand.x[t];
            y[0] = cand.y[t];
            z[0] = cand.z[t];
            id[0] = cand.id[t];
            (
                vld1q_f64(x.as_ptr()),
                vld1q_f64(y.as_ptr()),
                vld1q_f64(z.as_ptr()),
                vld1q_u64(id.as_ptr()),
            )
        };
        let dx = vsubq_f64(xv, px);
        let dy = vsubq_f64(yv, py);
        let dz = vsubq_f64(zv, pz);
        let d2 = vaddq_f64(
            vaddq_f64(vmulq_f64(dx, dx), vmulq_f64(dy, dy)),
            vmulq_f64(dz, dz),
        );
        let eq = vceqq_u64(ids, me);
        let d2 = vbslq_f64(eq, inf, d2);
        low = vminq_f64(low, d2);
        vst1q_f64(dp.add(t), d2);
        t += 2;
    }
    let mut bound = f64::INFINITY;
    if (1..=2).contains(&heap.k) && m > 0 {
        let mut lanes = [0.0f64; 2];
        vst1q_f64(lanes.as_mut_ptr(), low);
        lanes.sort_unstable_by(f64::total_cmp);
        bound = lanes[heap.k - 1];
    }
    let mut t = 0usize;
    while t < m {
        let limit = if heap.full() {
            heap.worst().min(bound)
        } else {
            bound
        };
        let d2 = vld1q_f64(dp.add(t));
        let take = vandq_u64(vcleq_f64(d2, vdupq_n_f64(limit)), vcltq_f64(d2, inf));
        if vgetq_lane_u64::<0>(take) != 0 {
            heap.push(vgetq_lane_f64::<0>(d2), cand.id[t] as usize);
        }
        if t + 1 < m && vgetq_lane_u64::<1>(take) != 0 {
            heap.push(vgetq_lane_f64::<1>(d2), cand.id[t + 1] as usize);
        }
        t += 2;
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

#[cfg(all(feature = "parallel", linkcell_openmp))]
#[path = "knn_fused.rs"]
mod fused;

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

    fn row_bits(row: &Neighbors) -> (Vec<usize>, Vec<u64>) {
        (
            row.indices.clone(),
            row.dist2.iter().map(|d| d.to_bits()).collect(),
        )
    }

    #[test]
    fn every_simd_path_matches_scalar_neighbours() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let unit = |v: u64| (v >> 11) as f64 / (1u64 << 53) as f64;
        let cells = [
            Cell::ortho(22.0, 19.0, 25.0).unwrap(),
            Cell::from_vectors(
                [22.0, 0.0, 0.0],
                [3.5, 20.0, 0.0],
                [-2.0, 1.5, 21.0],
                [0.0; 3],
            )
            .unwrap(),
            Cell::from_vectors(
                [20.0, 2.0, -1.0],
                [1.5, 19.0, 2.5],
                [-1.0, 1.5, 20.0],
                [0.25, 0.0, -0.25],
            )
            .unwrap(),
        ];
        for sim in &cells {
            let xyz: Vec<[f64; 3]> = (0..96)
                .map(|_| sim.cartesian([unit(next()), unit(next()), unit(next())]))
                .collect();
            let mask: Vec<bool> = (0..xyz.len()).map(|t| t % 5 != 1).collect();
            for k in [1usize, 4, 7, 8, 20] {
                for mask in [None, Some(mask.as_slice())] {
                    let scalar = {
                        let _guard = crate::simd::force(crate::simd::Path::Scalar);
                        knearest(&xyz, sim, k, mask, Some(3.0)).unwrap()
                    };
                    for path in crate::simd::paths_to_compare() {
                        let got = {
                            let _guard = crate::simd::force(path);
                            knearest(&xyz, sim, k, mask, Some(3.0)).unwrap()
                        };
                        for i in 0..xyz.len() {
                            assert_eq!(
                                row_bits(&got[i]),
                                row_bits(&scalar[i]),
                                "{path:?} k={k} i={i}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scan_paths_match_scalar_bits() {
        let mut state = 0xc2b2_ae3d_27d4_eb4fu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let unit = |v: u64| (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        for &m in &[1usize, 2, 3, 4, 5, 8, 15, 32] {
            let mut cand = Candidates::default();
            for t in 0..m {
                cand.x.push(unit(next()) * 20.0);
                cand.y.push(unit(next()) * 20.0);
                cand.z.push(unit(next()) * 20.0);
                cand.id.push(t as u64);
            }
            let pi = [
                unit(next()) * 20.0,
                unit(next()) * 20.0,
                unit(next()) * 20.0,
            ];
            let source = m / 2;
            for k in [1usize, 2, 4, 8, 20] {
                let scalar = scan_row(crate::simd::Path::Scalar, &mut cand, pi, source, k);
                for path in crate::simd::paths_to_compare() {
                    let got = scan_row(path, &mut cand, pi, source, k);
                    assert_eq!(got, scalar, "{path:?} m={m} k={k}");
                }
            }
        }
    }

    fn scan_row(
        path: crate::simd::Path,
        cand: &mut Candidates,
        pi: [f64; 3],
        i: usize,
        k: usize,
    ) -> (Vec<i32>, Vec<u64>) {
        let mut heap = KHeap::new(k);
        scan_at(path, cand, pi, i, &mut heap);
        let mut nn = vec![-1i32; k];
        let mut d2 = vec![f64::NAN; k];
        heap.write_sorted(&mut nn, Some(&mut d2));
        (nn, d2.iter().map(|v| v.to_bits()).collect())
    }

    /// Fold columns and neighbour rows stay bitwise and ordering identical
    /// at 1 thread, 2 threads, and every core, on every SIMD path.
    #[cfg(feature = "parallel")]
    #[test]
    fn rank_invariant_across_threads_and_simd_paths() {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let mut counts = vec![1usize, 2, 3, 8, n_threads];
        counts.sort_unstable();
        counts.dedup();
        let (mut xyz, cell) = ortho_lattice(21);
        xyz[0][0] = -0.0;
        xyz[1][1] = f64::from_bits(xyz[1][1].to_bits() + 1);
        xyz[2][2] = 1.0e-320;
        let reference = rank_snapshot(1, crate::simd::Path::Scalar, &xyz, &cell);
        for &threads in &counts {
            for path in crate::simd::paths_to_compare() {
                let got = rank_snapshot(threads, path, &xyz, &cell);
                let tag = format!("threads={threads} path={path:?}");
                assert_eq!(got.frac.len(), reference.frac.len(), "{tag}");
                for i in 0..got.frac.len() {
                    assert_eq!(got.frac[i], reference.frac[i], "{tag} frac {i}");
                    assert_eq!(got.folded[i], reference.folded[i], "{tag} folded {i}");
                }
                assert_eq!(got.offsets, reference.offsets, "{tag} offsets");
                assert_eq!(got.occupants, reference.occupants, "{tag} occupants");
                assert_eq!(got.nn.len(), reference.nn.len(), "{tag}");
                for i in 0..got.nn.len() {
                    assert_eq!(got.nn[i], reference.nn[i], "{tag} nn {i}");
                    assert_eq!(got.d2[i], reference.d2[i], "{tag} d2 {i}");
                }
                assert_eq!(got.pairs, reference.pairs, "{tag} pairs");
            }
        }
    }

    #[cfg(feature = "parallel")]
    struct RankSnap {
        frac: Vec<[u64; 3]>,
        folded: Vec<[u64; 3]>,
        offsets: Vec<usize>,
        occupants: Vec<usize>,
        nn: Vec<Vec<usize>>,
        d2: Vec<Vec<u64>>,
        pairs: Vec<(usize, usize, [i32; 3], u64)>,
    }

    #[cfg(feature = "parallel")]
    fn rank_snapshot(
        threads: usize,
        path: crate::simd::Path,
        xyz: &[[f64; 3]],
        cell: &Cell,
    ) -> RankSnap {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(move || {
            let _guard = crate::simd::force(path);
            let mesh = Mesh::build(xyz, cell, None, 3.0).unwrap();
            let rows = knearest(xyz, cell, 4, None, Some(3.0)).unwrap();
            let pairs = crate::pairs_within(xyz, cell, 4.5, None, None, false).unwrap();
            let mut cols = crate::PairColumns::default();
            crate::pairs_within_columns(xyz, cell, 4.5, None, None, false, &mut cols).unwrap();
            let pair_rows: Vec<_> = pairs
                .iter()
                .map(|p| (p.i, p.j, p.shift, p.dist2.to_bits()))
                .collect();
            let col_rows: Vec<_> = (0..cols.len())
                .map(|t| {
                    (
                        cols.i[t] as usize,
                        cols.j[t] as usize,
                        cols.shift[t],
                        cols.dist2[t].to_bits(),
                    )
                })
                .collect();
            assert_eq!(pair_rows, col_rows, "columns differ from rows");
            let bits = |v: [f64; 3]| [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
            RankSnap {
                frac: mesh.frac.iter().copied().map(bits).collect(),
                folded: mesh.folded.iter().copied().map(bits).collect(),
                offsets: mesh.offsets.clone(),
                occupants: mesh.occupants.clone(),
                nn: rows.iter().map(|r| r.indices.clone()).collect(),
                d2: rows
                    .iter()
                    .map(|r| r.dist2.iter().map(|d| d.to_bits()).collect())
                    .collect(),
                pairs: pair_rows,
            }
        })
    }
}

/// The fused region against brute force and the rayon walk.
#[cfg(all(test, feature = "parallel", linkcell_openmp))]
mod fused_agree {
    use super::{knearest_brute, knearest_into_d2, FORCE_FUSED, FORCE_LEGACY};
    use crate::Cell;

    fn lcg(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn unit(state: &mut u64) -> f64 {
        (lcg(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn periodic(cell: &Cell, n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut state = seed;
        (0..n)
            .map(|_| cell.cartesian([unit(&mut state), unit(&mut state), unit(&mut state)]))
            .collect()
    }

    /// A cloud in the middle of the box. Images are a full edge away.
    fn open_cloud(cell: &Cell, n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                cell.cartesian([
                    0.40 + 0.20 * unit(&mut state),
                    0.40 + 0.20 * unit(&mut state),
                    0.40 + 0.20 * unit(&mut state),
                ])
            })
            .collect()
    }

    fn buffers(xyz: &[[f64; 3]], cell: &Cell, k: usize, hint: Option<f64>) -> (Vec<i32>, Vec<u64>) {
        let mut nn = vec![7i32; xyz.len() * k];
        let mut d2 = vec![1.0; xyz.len() * k];
        knearest_into_d2(xyz, cell, k, None, hint, &mut nn, Some(&mut d2)).unwrap();
        (nn, d2.into_iter().map(|v| v.to_bits()).collect())
    }

    fn assert_same(label: &str, got: &(Vec<i32>, Vec<u64>), want: &(Vec<i32>, Vec<u64>)) {
        assert_eq!(got.0, want.0, "{label} indices");
        assert_eq!(got.1, want.1, "{label} dist2 bits");
    }

    fn with_fused<T>(body: impl FnOnce() -> T) -> T {
        FORCE_FUSED.with(|cell| cell.set(true));
        let out = body();
        FORCE_FUSED.with(|cell| cell.set(false));
        out
    }

    fn with_legacy<T>(body: impl FnOnce() -> T) -> T {
        FORCE_LEGACY.with(|cell| cell.set(true));
        let out = body();
        FORCE_LEGACY.with(|cell| cell.set(false));
        out
    }

    fn cells(span: f64) -> Vec<Cell> {
        vec![
            Cell::ortho(span, span * 0.9, span * 1.15).unwrap(),
            Cell::from_vectors(
                [span, 0.0, 0.0],
                [span * 0.15, span * 0.95, 0.0],
                [span * -0.05, span * 0.08, span * 1.05],
                [0.0; 3],
            )
            .unwrap(),
        ]
    }

    #[test]
    fn fused_matches_brute_and_legacy_on_periodic_and_open_boxes() {
        let shapes = [
            (
                "periodic",
                periodic as fn(&Cell, usize, u64) -> Vec<[f64; 3]>,
            ),
            ("open", open_cloud),
        ];
        for (shape, build) in shapes {
            for cell in cells(36.0) {
                for &(n, k, seed, hint) in &[
                    (48usize, 1usize, 0x11u64, None),
                    (96, 4, 0x22, Some(2.5)),
                    (160, 17, 0x33, Some(4.0)),
                    (320, 4, 0x44, Some(3.0)),
                    (7, 10, 0x55, None),
                ] {
                    let mut xyz = build(&cell, n, seed);
                    // A point outside the primary cell still folds in.
                    if n > 3 {
                        xyz[1][0] += cell.a()[0];
                        xyz[2][1] -= cell.b()[1];
                    }
                    // Two coincident sites.
                    if n > 8 {
                        xyz[5] = xyz[4];
                    }
                    let label = format!("{shape} n={n} k={k} seed={seed} hint={hint:?}");
                    let fused = with_fused(|| buffers(&xyz, &cell, k, hint));
                    let legacy = with_legacy(|| buffers(&xyz, &cell, k, hint));
                    let brute = knearest_brute(&xyz, &cell, k, None).unwrap();
                    assert_same(&format!("{label} fused vs legacy"), &fused, &legacy);
                    let mut brute_nn = vec![-1i32; n * k];
                    let mut brute_d2 = vec![f64::NAN; n * k];
                    for (i, row) in brute.iter().enumerate() {
                        for (t, (&j, &d)) in row.indices.iter().zip(&row.dist2).enumerate() {
                            brute_nn[i * k + t] = j as i32;
                            brute_d2[i * k + t] = d;
                        }
                    }
                    assert_eq!(fused.0, brute_nn, "{label} fused vs brute indices");
                    for (i, (&bits, brute_d)) in fused.1.iter().zip(&brute_d2).enumerate() {
                        let got = f64::from_bits(bits);
                        let scale = 1.0 + got.abs().max(brute_d.abs());
                        assert!(
                            (got.is_nan() && brute_d.is_nan())
                                || (got - brute_d).abs() <= 1e-12 * scale,
                            "{label} fused vs brute dist2 [{i}]: {got} vs {brute_d}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn large_unmasked_query_uses_the_region_and_matches_legacy() {
        let cell = Cell::ortho(48.0, 47.0, 49.5).unwrap();
        let mut state = 0xabcdu64;
        let xyz: Vec<[f64; 3]> = (0..9_000)
            .map(|_| cell.cartesian([unit(&mut state), unit(&mut state), unit(&mut state)]))
            .collect();
        let fused = buffers(&xyz, &cell, 4, Some(3.0));
        let legacy = with_legacy(|| buffers(&xyz, &cell, 4, Some(3.0)));
        assert_same("n=9000", &fused, &legacy);
        assert!(
            super::knearest_team_threads() >= 1,
            "team {}",
            super::knearest_team_threads()
        );
    }

    /// Every thread of this process as `(id, name)`. Linux reads
    /// `/proc/self/task`. macOS asks Mach for the thread ports and
    /// pthread for each name.
    #[cfg(target_os = "linux")]
    fn threads() -> Vec<(String, String)> {
        let Ok(dir) = std::fs::read_dir("/proc/self/task") else {
            return Vec::new();
        };
        dir.flatten()
            .map(|entry| {
                let comm = std::fs::read_to_string(entry.path().join("comm")).unwrap_or_default();
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    comm.trim_end().to_string(),
                )
            })
            .collect()
    }

    #[cfg(target_os = "macos")]
    fn threads() -> Vec<(String, String)> {
        use std::os::raw::{c_char, c_int, c_void};
        extern "C" {
            static mach_task_self_: u32;
            fn task_threads(task: u32, list: *mut *mut u32, count: *mut u32) -> c_int;
            fn mach_port_deallocate(task: u32, name: u32) -> c_int;
            fn vm_deallocate(task: u32, address: usize, size: usize) -> c_int;
            fn pthread_from_mach_thread_np(port: u32) -> *mut c_void;
            fn pthread_getname_np(thread: *mut c_void, name: *mut c_char, len: usize) -> c_int;
        }
        let task = unsafe { mach_task_self_ };
        let mut list: *mut u32 = std::ptr::null_mut();
        let mut count = 0u32;
        if unsafe { task_threads(task, &mut list, &mut count) } != 0 {
            return Vec::new();
        }
        let ports = unsafe { std::slice::from_raw_parts(list, count as usize) };
        let found = ports
            .iter()
            .map(|&port| {
                let thread = unsafe { pthread_from_mach_thread_np(port) };
                let mut buf = [0 as c_char; 64];
                let named = !thread.is_null()
                    && unsafe { pthread_getname_np(thread, buf.as_mut_ptr(), buf.len()) } == 0;
                let name = if named {
                    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                        .to_string_lossy()
                        .into_owned()
                } else {
                    String::new()
                };
                (port.to_string(), name)
            })
            .collect();
        for &port in ports {
            unsafe { mach_port_deallocate(task, port) };
        }
        unsafe {
            vm_deallocate(
                task,
                list as usize,
                count as usize * std::mem::size_of::<u32>(),
            )
        };
        found
    }

    fn task_count() -> usize {
        threads().len()
    }

    fn wait_tasks(want: usize) -> usize {
        let mut last = task_count();
        for _ in 0..100 {
            if last <= want {
                return last;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            last = task_count();
        }
        last
    }

    /// The name `src/knn_region.c` gives every worker of the team.
    const TEAM_NAME: &str = "linkcell-knn";

    /// Workers of linkcell's k-nearest team. The caller, rayon's pool and
    /// any thread an OpenMP runtime keeps do not carry the name.
    fn team_workers() -> usize {
        threads()
            .iter()
            .filter(|(_, name)| name == TEAM_NAME)
            .count()
    }

    fn wait_team_workers(want: usize) -> usize {
        let mut last = team_workers();
        for _ in 0..100 {
            if last == want {
                return last;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            last = team_workers();
        }
        last
    }

    /// The team is joined, so no worker is left and the process is back to
    /// `before` threads under libgomp and libomp alike.
    fn assert_threads_back(before: usize, label: &str) {
        let workers = wait_team_workers(0);
        assert_eq!(
            workers,
            0,
            "{label} left linkcell workers running\n{}",
            task_names()
        );
        let after = wait_tasks(before);
        assert_eq!(
            after,
            before,
            "{label} left threads: before {before} after {after}\n{}",
            task_names()
        );
    }

    fn task_names() -> String {
        let mut names = String::new();
        for (id, name) in threads() {
            names.push_str(&id);
            names.push(':');
            names.push_str(&name);
            names.push('\n');
        }
        names
    }

    /// Child process: the team follows `omp_set_num_threads`, and shutdown
    /// joins it. When `liblinkcell.so` is beside the build, also query
    /// through that image and `dlclose` it.
    #[test]
    fn team_follows_omp_set_num_threads_and_joins_on_shutdown() {
        if std::env::var_os("LC_TEAM_PROBE").is_none() {
            let exe = std::env::current_exe().unwrap();
            let out = std::process::Command::new(&exe)
                .arg(
                    "knearest::fused_agree::team_follows_omp_set_num_threads_and_joins_on_shutdown",
                )
                .args(["--exact", "--test-threads", "1", "--nocapture"])
                .env("LC_TEAM_PROBE", "1")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "probe status {:?}\n{}{}",
                out.status,
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        extern "C" {
            fn omp_set_num_threads(n: i32);
            fn omp_get_max_threads() -> i32;
        }
        let before = task_count();
        assert_eq!(team_workers(), 0, "linkcell workers before the first query");
        let saved = unsafe { omp_get_max_threads() }.max(1);
        unsafe { omp_set_num_threads(2) };
        let cell = Cell::ortho(30.0, 30.0, 30.0).unwrap();
        let xyz = periodic(&cell, 64, 9);
        let _ = with_fused(|| buffers(&xyz, &cell, 4, Some(3.0)));
        assert_eq!(super::knearest_team_threads(), 2);
        let mid = task_count();
        assert!(
            mid >= before + 2,
            "team not running: before {before} mid {mid}"
        );
        assert_eq!(team_workers(), 2, "team of two not named\n{}", task_names());
        super::shutdown_knearest_team();
        assert_threads_back(before, "shutdown");
        unsafe { omp_set_num_threads(saved) };
        let _ = with_fused(|| buffers(&xyz, &cell, 4, Some(3.0)));
        assert_eq!(super::knearest_team_threads(), saved as usize);
        assert_eq!(
            team_workers(),
            saved as usize,
            "team of {saved} not named\n{}",
            task_names()
        );
        super::shutdown_knearest_team();
        assert_threads_back(before, "second shutdown");

        #[cfg(feature = "capi")]
        {
            let so = std::env::current_exe().ok().and_then(|mut exe| {
                for _ in 0..6 {
                    exe.pop();
                    let candidate = exe.join("liblinkcell.so");
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
                None
            });
            if let Some(so) = so {
                unload_cdylib(&so, before, saved as usize);
            }
        }
    }

    /// Child process: after `lc_set_threads(1)` a C API search adds no
    /// thread. Asked from outside a pool, rayon would build its global
    /// pool at the cpu count, and the fused region would start its team.
    #[cfg(feature = "capi")]
    #[test]
    fn one_thread_capi_search_starts_no_thread() {
        if std::env::var_os("LC_ONE_THREAD_PROBE").is_none() {
            let exe = std::env::current_exe().unwrap();
            let out = std::process::Command::new(&exe)
                .arg("knearest::fused_agree::one_thread_capi_search_starts_no_thread")
                .args(["--exact", "--test-threads", "1", "--nocapture"])
                .env("LC_ONE_THREAD_PROBE", "1")
                .env_remove("RAYON_NUM_THREADS")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "probe status {:?}\n{}{}",
                out.status,
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        let before = task_count();
        crate::capi::lc_set_threads(1);
        let n = 8192usize;
        let xyz: Vec<f64> = (0..n)
            .flat_map(|i| [(i % 20) as f64, ((i / 20) % 20) as f64, (i / 400) as f64])
            .collect();
        let cell = [
            40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0,
        ];
        let mut out = vec![-1i32; n * 2];
        let rc = unsafe {
            crate::capi::lc_knearest(
                xyz.as_ptr(),
                n,
                cell.as_ptr().cast(),
                2,
                std::ptr::null(),
                3.0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "one-thread query");
        assert!(
            out.iter().all(|&j| j >= 0),
            "one-thread query left a slot empty"
        );
        let same = |label: &str| {
            assert_eq!(
                task_count(),
                before,
                "a one-thread {label} started threads\n{}",
                task_names()
            );
        };
        same("k-nearest search");

        let simbox: *const crate::capi::lc_cell = cell.as_ptr().cast();
        let mut rows = 0usize;
        let rc = unsafe {
            crate::capi::lc_pairs_within(
                xyz.as_ptr(),
                n,
                simbox,
                1.5,
                std::ptr::null(),
                0.0,
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                &mut rows,
            )
        };
        assert_eq!(rc, 0, "one-thread pair count");
        assert!(rows > 0, "one-thread pair count found no pair");
        same("pair count");
        let mut pi = vec![0i32; rows];
        let mut pj = vec![0i32; rows];
        let mut shift = vec![0i32; rows * 3];
        let mut d2 = vec![0.0f64; rows];
        let rc = unsafe {
            crate::capi::lc_pairs_within(
                xyz.as_ptr(),
                n,
                simbox,
                1.5,
                std::ptr::null(),
                0.0,
                1,
                pi.as_mut_ptr(),
                pj.as_mut_ptr(),
                shift.as_mut_ptr(),
                d2.as_mut_ptr(),
                rows,
                &mut rows,
            )
        };
        assert_eq!(rc, 0, "one-thread pair fill");
        same("pair fill");

        let mut found = vec![crate::capi::lc_pair::default(); rows];
        let rc = unsafe {
            crate::capi::lc_pairs_within_rows(
                xyz.as_ptr(),
                n,
                simbox,
                1.5,
                std::ptr::null(),
                0.0,
                1,
                found.as_mut_ptr(),
                rows,
                &mut rows,
            )
        };
        assert_eq!(rc, 0, "one-thread pair rows");
        same("pair row fill");

        let pbc = [1, 1, 0];
        let mut count = 0usize;
        let rc = unsafe {
            crate::capi::lc_pairs_within_pbc(
                xyz.as_ptr(),
                n,
                simbox,
                pbc.as_ptr(),
                1.5,
                std::ptr::null(),
                0.0,
                1,
                0,
                0,
                std::ptr::null_mut(),
                0,
                &mut count,
            )
        };
        assert_eq!(rc, 0, "one-thread mixed pair count");
        same("mixed-boundary pair count");
        let rc = unsafe {
            crate::capi::lc_pairs_open(
                xyz.as_ptr(),
                n,
                1.5,
                0.0,
                std::ptr::null_mut(),
                0,
                &mut count,
            )
        };
        assert_eq!(rc, 0, "one-thread open pair count");
        same("open pair count");

        let mut ncell = 0usize;
        let rc = unsafe {
            crate::capi::lc_cell_grid(
                xyz.as_ptr(),
                n,
                simbox,
                1.5,
                std::ptr::null(),
                0.0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                &mut ncell,
            )
        };
        assert_eq!(rc, 0, "one-thread cell grid");
        same("cell grid");
    }

    #[cfg(feature = "capi")]
    fn unload_cdylib(so: &std::path::Path, before: usize, team: usize) {
        use std::os::raw::{c_char, c_int, c_void};
        extern "C" {
            fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            fn dlclose(handle: *mut c_void) -> c_int;
        }
        const RTLD_NOW: c_int = 2;
        let path = std::ffi::CString::new(so.to_str().unwrap()).unwrap();
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        assert!(!handle.is_null(), "dlopen {}", so.display());
        let name = std::ffi::CString::new("lc_knearest").unwrap();
        let sym = unsafe { dlsym(handle, name.as_ptr()) };
        assert!(!sym.is_null(), "dlsym lc_knearest");
        type Knn = unsafe extern "C" fn(
            *const f64,
            usize,
            *const f64,
            usize,
            *const c_int,
            f64,
            *mut c_int,
        ) -> c_int;
        let knearest: Knn = unsafe { std::mem::transmute(sym) };
        let n = 8192usize;
        let mut xyz = vec![0.0f64; n * 3];
        for i in 0..n {
            xyz[i * 3] = (i % 20) as f64;
            xyz[i * 3 + 1] = ((i / 20) % 20) as f64;
            xyz[i * 3 + 2] = (i / 400) as f64;
        }
        let cell = [
            40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0,
        ];
        let mut out = vec![-1i32; n * 2];
        let rc = unsafe {
            knearest(
                xyz.as_ptr(),
                n,
                cell.as_ptr(),
                2,
                std::ptr::null(),
                3.0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "cdylib query");
        assert_eq!(
            team_workers(),
            team,
            "cdylib team of {team} did not start\n{}",
            task_names()
        );
        let stop_name = std::ffi::CString::new("lc_knn_shutdown").unwrap();
        let stop_sym = unsafe { dlsym(handle, stop_name.as_ptr()) };
        assert!(!stop_sym.is_null(), "dlsym lc_knn_shutdown");
        let stop: unsafe extern "C" fn() = unsafe { std::mem::transmute(stop_sym) };
        unsafe { stop() };
        assert_threads_back(before, "cdylib shutdown");
        let closed = unsafe { dlclose(handle) };
        assert_eq!(closed, 0, "dlclose");
        assert_threads_back(before, "dlclose");
    }

    /// Guard size of worker `slot`, or 0 when that worker is not up.
    fn team_guard(slot: i32) -> usize {
        extern "C" {
            fn lc_knn_team_guard(slot: i32) -> usize;
        }
        unsafe { lc_knn_team_guard(slot) }
    }

    /// The stack size of each worker of the standing team, in slot order,
    /// as the C library reports it for that thread.
    fn team_stacks() -> Vec<usize> {
        extern "C" {
            fn lc_knn_team_stack(slot: i32) -> usize;
        }
        (0..)
            .map(|slot| unsafe { lc_knn_team_stack(slot) })
            .take_while(|&size| size != 0)
            .collect()
    }

    /// Child processes: `OMP_THREAD_LIMIT` caps the team below what
    /// `omp_set_num_threads` asks for, `OMP_STACKSIZE` sizes the worker
    /// stacks, a team of one is the caller with no worker, and
    /// `OMP_DYNAMIC` trims the team by the load average as libgomp does.
    #[test]
    fn team_honours_thread_limit_stacksize_and_dynamic() {
        let Some(mode) = std::env::var_os("LC_LIMIT_PROBE") else {
            let exe = std::env::current_exe().unwrap();
            for (mode, var, value) in [
                ("2", "OMP_THREAD_LIMIT", "2"),
                ("1", "OMP_THREAD_LIMIT", "1"),
                ("dyn", "OMP_DYNAMIC", "true"),
            ] {
                let out = std::process::Command::new(&exe)
                    .arg("knearest::fused_agree::team_honours_thread_limit_stacksize_and_dynamic")
                    .args(["--exact", "--test-threads", "1", "--nocapture"])
                    .env_remove("OMP_THREAD_LIMIT")
                    .env_remove("OMP_DYNAMIC")
                    .env("OMP_STACKSIZE", " 12m ")
                    .env(var, value)
                    .env("LC_LIMIT_PROBE", mode)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "probe {mode} status {:?}\n{}{}",
                    out.status,
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            return;
        };
        extern "C" {
            fn omp_set_num_threads(n: i32);
            fn omp_get_thread_limit() -> i32;
            fn omp_get_num_procs() -> i32;
            fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
        }
        let dynamic = || {
            let mut load = [0.0f64; 3];
            let busy = if unsafe { getloadavg(load.as_mut_ptr(), 3) } == 3 {
                (load[2] + 0.1) as i32
            } else {
                0
            };
            let procs = unsafe { omp_get_num_procs() }.min(4);
            if busy >= procs {
                1
            } else {
                (procs - busy) as usize
            }
        };
        let stack = 12usize << 20;
        let cell = Cell::ortho(30.0, 30.0, 30.0).unwrap();
        let xyz = periodic(&cell, 64, 9);
        // Another thread holds a 12 MiB stack through the query. No thread
        // exits before it, so no cached stack can stand in for a worker's.
        let (release, held) = std::sync::mpsc::channel::<()>();
        let holder = std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || {
                let _ = held.recv();
            })
            .unwrap();
        // A team of one warms everything else up and leaves no worker
        // stack in glibc's cache of joined threads.
        unsafe { omp_set_num_threads(1) };
        let _ = with_fused(|| buffers(&xyz, &cell, 4, Some(3.0)));
        assert_eq!(super::knearest_team_threads(), 1);
        unsafe { omp_set_num_threads(4) };
        let before = task_count();
        let expect = match mode.to_str() {
            Some("2") => vec![2],
            Some("1") => vec![1],
            _ => vec![dynamic()],
        };
        assert!(
            mode != "dyn" || expect[0] <= 4,
            "dynamic size above nthreads-var"
        );
        let _ = with_fused(|| buffers(&xyz, &cell, 4, Some(3.0)));
        let team = super::knearest_team_threads();
        if mode == "dyn" {
            assert!(
                team == expect[0] || team == dynamic(),
                "OMP_DYNAMIC team {team}, load-average size {expect:?}"
            );
        } else {
            assert_eq!(unsafe { omp_get_thread_limit() }, expect[0] as i32);
            assert_eq!(team, expect[0], "OMP_THREAD_LIMIT={}", expect[0]);
        }
        let workers = if team > 1 { team } else { 0 };
        let mid = task_count();
        assert_eq!(
            mid,
            before + workers,
            "team {team}: before {before} mid {mid}\n{}",
            task_names()
        );
        assert_eq!(
            team_stacks(),
            vec![stack; workers],
            "OMP_STACKSIZE=12m stacks for a team of {team}"
        );
        for slot in 0..workers {
            let guard = team_guard(slot as i32);
            assert!(guard > 0, "worker {slot} guard size {guard}");
        }
        super::shutdown_knearest_team();
        let after = wait_tasks(before);
        assert_eq!(after, before, "shutdown left threads\n{}", task_names());
        assert_eq!(team_stacks(), Vec::<usize>::new(), "stacks after shutdown");
        assert_eq!(team_guard(0), 0, "guard after shutdown");
        drop(release);
        holder.join().unwrap();
    }

    /// `OMP_STACKSIZE=1k` is below `PTHREAD_STACK_MIN`, and the mid size is
    /// one page above it, still below the stack floor. Both run the workers
    /// on the floor and warn once. `2^63` bytes is larger than the user
    /// address space. `pthread_attr_setstacksize` accepts it and
    /// `pthread_create` fails, so the query stays on the caller, the
    /// process warns once, and the next query tries again.
    #[test]
    fn small_stack_is_raised_and_a_refused_stack_stays_on_the_caller() {
        extern "C" {
            fn omp_set_num_threads(n: i32);
            fn omp_set_dynamic(dynamic_threads: i32);
            fn lc_knn_pthread_stack_min() -> usize;
            fn lc_knn_stack_floor() -> usize;
            fn lc_knn_stack_starts() -> usize;
        }
        let Some(mode) = std::env::var_os("LC_STACK_PROBE") else {
            let exe = std::env::current_exe().unwrap();
            let min = unsafe { lc_knn_pthread_stack_min() };
            let floor = unsafe { lc_knn_stack_floor() };
            let mut probes = vec![("small", "1k".to_string())];
            // One page above the minimum: pthread accepts it, and it sits in
            // the range where the fold frame can reach the guard.
            let mid = min + 4096;
            if mid < floor {
                probes.push(("mid", format!("{mid}b")));
            }
            // 2^63 bytes. Larger than a 57-bit virtual address space, so
            // the mapping cannot succeed under any overcommit setting.
            // glibc's pthread_attr_setstacksize accepts it.
            probes.push(("huge", "9223372036854775808b".to_string()));
            for (mode, stack) in probes {
                let out = std::process::Command::new(&exe)
                    .arg(
                        "knearest::fused_agree::small_stack_is_raised_and_a_refused_stack_stays_on_the_caller",
                    )
                    .args(["--exact", "--test-threads", "1", "--nocapture"])
                    .env_remove("OMP_THREAD_LIMIT")
                    .env_remove("OMP_DYNAMIC")
                    .env_remove("GOMP_STACKSIZE")
                    .env("OMP_DYNAMIC", "false")
                    .env("OMP_STACKSIZE", &stack)
                    .env("LC_STACK_PROBE", mode)
                    .output()
                    .unwrap();
                let err = String::from_utf8_lossy(&out.stderr);
                assert!(
                    out.status.success(),
                    "probe {mode} status {:?}\n{}{err}",
                    out.status,
                    String::from_utf8_lossy(&out.stdout)
                );
                let raised = err.matches("below the worker stack floor").count();
                let refused = err.matches("was refused").count();
                let create_failed = err.matches("could not start a worker").count();
                if mode != "huge" {
                    assert_eq!(raised, 1, "{err}");
                    assert_eq!(refused, 0, "{err}");
                    assert_eq!(create_failed, 0, "{err}");
                } else {
                    assert_eq!(create_failed, 1, "{err}");
                    assert_eq!(refused, 0, "{err}");
                    assert_eq!(raised, 0, "{err}");
                }
            }
            return;
        };
        let cell = Cell::ortho(30.0, 30.0, 30.0).unwrap();
        let xyz = periodic(&cell, 64, 9);
        unsafe {
            omp_set_dynamic(0);
            omp_set_num_threads(1);
        }
        let reference = with_fused(|| buffers(&xyz, &cell, 4, Some(3.0)));
        assert_eq!(super::knearest_team_threads(), 1);
        unsafe { omp_set_num_threads(4) };
        let first = with_fused(|| buffers(&xyz, &cell, 4, Some(3.0)));
        let second = with_fused(|| buffers(&xyz, &cell, 4, Some(3.0)));
        assert_same("first query", &first, &reference);
        assert_same("second query", &second, &reference);
        let team = super::knearest_team_threads();
        let starts = unsafe { lc_knn_stack_starts() };
        if mode != "huge" {
            assert!(team > 1, "raised stack started no team");
            let min = unsafe { lc_knn_pthread_stack_min() };
            let floor = unsafe { lc_knn_stack_floor() };
            assert!(
                floor >= min,
                "floor {floor} is below PTHREAD_STACK_MIN {min}"
            );
            let stacks = team_stacks();
            assert_eq!(stacks.len(), team, "raised stack workers");
            for (slot, size) in stacks.iter().copied().enumerate() {
                assert!(
                    size >= min,
                    "worker {slot} stack {size} is below PTHREAD_STACK_MIN {min}"
                );
                assert_eq!(size, floor, "worker {slot} stack {size}");
            }
            assert_eq!(starts, 1, "raised stack was started twice");
        } else {
            assert_eq!(team, 1, "huge stack started a team");
            assert!(team_stacks().is_empty(), "huge stack left workers");
            assert_eq!(starts, 2, "pthread_create failure was remembered");
        }
        super::shutdown_knearest_team();
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    /// The graph from the nominations one slot at a time: an accepted slot
    /// `(i, t) -> j` appends `j` to row `i` and `i` to row `j`. Mutual rows
    /// are the lower partners ascending, then the higher ones in row order.
    #[cfg(feature = "parallel")]
    fn reference(nom: &[i32], n: usize, k: usize, mutual: bool) -> (Vec<i32>, Vec<i32>) {
        let listed = |j: usize, i: usize| nom[j * k..(j + 1) * k].contains(&(i as i32));
        let mut rows = vec![Vec::new(); n];
        if mutual {
            for (i, row) in rows.iter_mut().enumerate() {
                let keep: Vec<i32> = nom[i * k..(i + 1) * k]
                    .iter()
                    .copied()
                    .filter(|&j| j >= 0 && (j as usize) != i && listed(j as usize, i))
                    .collect();
                let mut low: Vec<i32> =
                    keep.iter().copied().filter(|&j| (j as usize) < i).collect();
                low.sort_unstable();
                row.extend(low);
                row.extend(keep.iter().copied().filter(|&j| (j as usize) > i));
            }
        } else {
            for i in 0..n {
                for t in 0..k {
                    let j = nom[i * k + t];
                    if j < 0 || j as usize == i {
                        continue;
                    }
                    let j = j as usize;
                    if i < j || !listed(j, i) {
                        rows[i].push(j as i32);
                        rows[j].push(i as i32);
                    }
                }
            }
        }
        let mut offsets = vec![0i32];
        let mut cols = Vec::new();
        for row in rows {
            cols.extend(row);
            offsets.push(cols.len() as i32);
        }
        (offsets, cols)
    }

    #[cfg(feature = "parallel")]
    fn points(n: usize, cell: &Cell, state: &mut u64) -> Vec<[f64; 3]> {
        let mut next = || {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let s = [next(), next(), next()];
                let (a, b, c) = (cell.a(), cell.b(), cell.c());
                std::array::from_fn(|x| s[0] * a[x] + s[1] * b[x] + s[2] * c[x])
            })
            .collect()
    }

    #[cfg(feature = "parallel")]
    fn in_pool<R: Send>(width: usize, f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .build()
            .unwrap()
            .install(f)
    }

    /// Every width, cell shape, mask, and k gives the graph the
    /// nominations define, row for row.
    #[cfg(feature = "parallel")]
    #[test]
    fn csr_rows_follow_the_nominations_at_every_width() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let cells = [
            Cell::ortho(41.0, 37.0, 45.0).unwrap(),
            Cell::from_vectors(
                [40.0, 0.0, 0.0],
                [6.0, 38.0, 0.0],
                [-5.0, 4.0, 42.0],
                [0.0; 3],
            )
            .unwrap(),
            Cell::from_vectors(
                [30.0, 9.0, 2.0],
                [-4.0, 33.0, 7.0],
                [3.0, -6.0, 36.0],
                [1.0, 2.0, 3.0],
            )
            .unwrap(),
        ];
        for cell in &cells {
            let xyz = points(9_000, cell, &mut state);
            let mask: Vec<bool> = (0..xyz.len()).map(|i| i % 7 != 3).collect();
            for k in [1usize, 4, 12] {
                for masked in [false, true] {
                    let m = masked.then_some(&mask[..]);
                    let mut nom = vec![0i32; xyz.len() * k];
                    knearest_into(&xyz, cell, k, m, Some(3.0), &mut nom).unwrap();
                    for mutual in [false, true] {
                        let want = reference(&nom, xyz.len(), k, mutual);
                        for width in [1usize, 2, 3, 8] {
                            let got = in_pool(width, || {
                                knearest_csr(&xyz, cell, k, m, Some(3.0), mutual).unwrap()
                            });
                            assert!(
                                got == want,
                                "k={k} masked={masked} mutual={mutual} width={width}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// One or no active point, and a box of two, leave every row empty
    /// or give the single pair.
    #[test]
    fn csr_of_tiny_inputs() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let one = [[1.0, 1.0, 1.0]];
        assert_eq!(
            knearest_csr(&one, &cell, 3, None, None, false).unwrap(),
            (vec![0, 0], vec![])
        );
        let two = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]];
        let off = [false, true];
        assert_eq!(
            knearest_csr(&two, &cell, 3, Some(&off), None, true).unwrap(),
            (vec![0, 0, 0], vec![])
        );
        assert_eq!(
            knearest_csr(&two, &cell, 3, None, None, false).unwrap(),
            (vec![0, 1, 2], vec![1, 0])
        );
    }
}
