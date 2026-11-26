//! Cutoff neighbours as CSR rows, built from one search.
//!
//! A periodic cell gathers each row whole, source by source, and lays
//! the rows out in atom order (see `pair_gather.rs`). A cell with an
//! open axis is searched as half rows. Those land in lanes, one or more
//! hit buffers each, which count their entries per row. A prefix sum
//! over the rows gives every lane its own slots in every row, the lanes
//! write their columns there, and each row is then sorted. The rows
//! hold no repeated column and every distance is the same bits at
//! every thread count, so the rows do not depend on how the work was
//! split.
//!
//! An atom meets another through two images only when the cutoff is at
//! least half the narrowest periodic width. Those rows keep the nearest
//! image of each column.

use super::gather::{gather, Gathered};
use super::{bin_atoms, hit_estimate, in_pool, prepare_pairs, row_guess, try_hold, Pair};
#[cfg(test)]
use super::{plan_unshifted, Found, Scratch};
use crate::cell::Cell;
use crate::open_cell::{pairs_within_pbc, PbcQuery};
use crate::Error;

/// Cutoff neighbours of every point as CSR rows.
///
/// Row `i` is `cols[offsets[i] as usize..offsets[i + 1] as usize]`. It
/// holds every `j != i` with an image strictly within the cutoff, once,
/// in ascending order. A half list keeps only `j > i`. `dist2` holds the
/// squared distance of the nearest such image, entry for entry with
/// `cols`, when it was asked for, and is empty otherwise.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PairCsr {
    /// Row starts, `n + 1` values, starting at zero.
    pub offsets: Vec<i32>,
    /// Neighbour indices, row after row.
    pub cols: Vec<i32>,
    /// Squared distance of the nearest image, or empty.
    pub dist2: Vec<f64>,
}

impl PairCsr {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Neighbours of point `i`.
    pub fn row(&self, i: usize) -> &[i32] {
        &self.cols[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }
}

/// [`pairs_within_pbc`](crate::pairs_within_pbc) as CSR rows: each
/// point's neighbours within `query.cutoff`, deduplicated over images.
///
/// Row `i` lists every `j != i` that has some image with squared distance
/// strictly below `cutoff²` (or at most `cutoff²` with `query.closed` on
/// an open or mixed cell), once, ascending. On a periodic axis that is
/// the minimum-image distance, including a cutoff past half the box,
/// where one atom can meet another through several images. A periodic
/// image of `i` itself is never listed, whatever
/// `query.exclude_self_images` says. `query.half` keeps `j > i` only.
/// A masked-out point has an empty row and appears in no row.
///
/// With `dist2` set, `out.dist2[k]` is the squared distance of the
/// nearest image of `out.cols[k]`; otherwise `out.dist2` is left empty.
/// The rows, and every stored distance, are the same bits at every
/// thread count and on every SIMD path.
///
/// An empty `xyz` gives one offset and no columns. Returns
/// [`Error::BadCutoff`] for a cutoff that is not finite and strictly
/// positive, [`Error::MaskLen`] for a mask of the wrong length,
/// [`Error::NonFinite`] for a coordinate that is not finite, and
/// [`Error::Overflow`] when an index or a column count does not fit
/// `i32`. The search's own errors are those of
/// [`pairs_within_pbc`](crate::pairs_within_pbc).
///
/// ```
/// use linkcell::{pairs_within_csr, Cell, PairCsr, PbcQuery};
///
/// # fn main() -> Result<(), linkcell::Error> {
/// let sim = Cell::ortho(10.0, 10.0, 10.0)?;
/// let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0], [5.0, 5.0, 5.0]];
/// let mut out = PairCsr::default();
/// pairs_within_csr(&xyz, &sim, &PbcQuery::periodic(1.0), true, &mut out)?;
/// assert_eq!(out.offsets, vec![0, 1, 2, 2]);
/// assert_eq!(out.cols, vec![1, 0]);
/// assert!((out.dist2[0] - 0.64).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
pub fn pairs_within_csr(
    xyz: &[[f64; 3]],
    cell: &Cell,
    query: &PbcQuery<'_>,
    dist2: bool,
    out: &mut PairCsr,
) -> Result<(), Error> {
    check(xyz, query)?;
    out.offsets.clear();
    out.cols.clear();
    out.dist2.clear();
    let n = xyz.len();
    if n == 0 {
        out.offsets.push(0);
        return Ok(());
    }
    if query.periodic == [true, true, true] {
        let n_act = prepare_pairs(xyz, cell, query.cutoff, query.mask, query.cell_hint)?;
        if n_act == 0 {
            out.offsets.resize(n + 1, 0);
            return Ok(());
        }
    }
    let est = estimate(n, cell, query);
    // Reserved on the caller's thread, as in `pairs_within`, and only
    // after the image and bin caps. A guess that does not fit is
    // TooManyImages and does not search.
    let guess = row_guess(est, query.half);
    try_hold(&mut out.offsets, n + 1)?;
    try_hold(&mut out.cols, guess)?;
    if dist2 {
        try_hold(&mut out.dist2, guess)?;
    }
    in_pool(est, || {
        if query.periodic != [true, true, true] {
            let source = Source::search(xyz, cell, query)?;
            let rows = Rows::new(&source, n, query.half, repeats(cell, query), dist2)?;
            rows.write_vec(out, dist2)?;
            return Ok(());
        }
        let Some(mut rows) = gathered(xyz, cell, query, dist2)? else {
            out.offsets.resize(n + 1, 0);
            return Ok(());
        };
        let need = rows.scan()?;
        try_hold(&mut out.cols, need)?;
        if dist2 {
            try_hold(&mut out.dist2, need)?;
        }
        let d2 = if dist2 {
            out.dist2.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        // Safety: the buffers hold `n + 1` offsets and `need` columns and
        // distances, and `write` sets every one.
        unsafe {
            rows.write(out.offsets.as_mut_ptr(), out.cols.as_mut_ptr(), d2);
            out.offsets.set_len(n + 1);
            out.cols.set_len(need);
            if dist2 {
                out.dist2.set_len(need);
            }
        }
        Ok(())
    })
}

/// The staged rows of a fully periodic search, or `None` when no point
/// is active.
fn gathered(
    xyz: &[[f64; 3]],
    cell: &Cell,
    query: &PbcQuery<'_>,
    dist2: bool,
) -> Result<Option<Gathered>, Error> {
    let Some(binned) = bin_atoms(xyz, cell, query.cutoff, query.mask, query.cell_hint, true)?
    else {
        return Ok(None);
    };
    let repeat = repeats(cell, query);
    Ok(Some(gather(
        &binned,
        cell,
        xyz.len(),
        query.half,
        repeat,
        dist2,
    )?))
}

/// Destination of [`Rows::write`]: `n + 1` offsets, and the columns and
/// (when non-null) distances for [`Rows::need`] entries.
#[derive(Clone, Copy)]
pub(crate) struct CsrDest {
    pub(crate) offsets: *mut i32,
    pub(crate) cols: *mut i32,
    pub(crate) dist2: *mut f64,
}
// Safety: the caller owns the buffers until `csr_rows` returns, and the
// writers touch disjoint slots.
unsafe impl Send for CsrDest {}
unsafe impl Sync for CsrDest {}

/// Where [`csr_rows`] put the rows.
#[cfg(feature = "capi")]
pub(crate) enum CsrWrite {
    /// Into the [`CsrDest`] `room` returned; this many columns.
    Direct(usize),
    /// `room` had no buffer for them: owned rows, with distances when
    /// asked for.
    Owned(PairCsr),
}

/// The C ABI's build: the rows go straight into `room(columns)` when it
/// returns a destination. Otherwise they are built into an owned
/// [`PairCsr`]. Distances are computed only with `dist2`, and then a
/// destination's non-null distance pointer receives them. `xyz` is not
/// empty and has passed [`check`].
#[cfg(feature = "capi")]
pub(crate) fn csr_rows(
    xyz: &[[f64; 3]],
    cell: &Cell,
    query: &PbcQuery<'_>,
    dist2: bool,
    room: impl FnOnce(usize) -> Option<CsrDest> + Send,
) -> Result<CsrWrite, Error> {
    let n = xyz.len();
    let est = estimate(n, cell, query);
    in_pool(est, || {
        if query.periodic == [true, true, true] {
            let Some(mut rows) = gathered(xyz, cell, query, dist2)? else {
                let empty = PairCsr {
                    offsets: vec![0; n + 1],
                    ..PairCsr::default()
                };
                return Ok(match room(0) {
                    Some(dest) => {
                        // Safety: `room` gave `n + 1` offsets.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                empty.offsets.as_ptr(),
                                dest.offsets,
                                n + 1,
                            )
                        };
                        CsrWrite::Direct(0)
                    }
                    None => CsrWrite::Owned(empty),
                });
            };
            let need = rows.scan()?;
            return Ok(match room(need) {
                Some(dest) => {
                    // Safety: `room` gave `n + 1` offsets and `need` columns
                    // (and distances when that pointer is non-null).
                    let d2 = if dist2 {
                        dest.dist2
                    } else {
                        std::ptr::null_mut()
                    };
                    unsafe { rows.write(dest.offsets, dest.cols, d2) };
                    CsrWrite::Direct(need)
                }
                None => {
                    let mut owned = PairCsr::default();
                    try_hold(&mut owned.offsets, n + 1)?;
                    try_hold(&mut owned.cols, need)?;
                    if dist2 {
                        try_hold(&mut owned.dist2, need)?;
                    }
                    // Safety: as in `pairs_within_csr`.
                    unsafe {
                        let d2 = if dist2 {
                            owned.dist2.as_mut_ptr()
                        } else {
                            std::ptr::null_mut()
                        };
                        rows.write(owned.offsets.as_mut_ptr(), owned.cols.as_mut_ptr(), d2);
                        owned.offsets.set_len(n + 1);
                        owned.cols.set_len(need);
                        if dist2 {
                            owned.dist2.set_len(need);
                        }
                    }
                    CsrWrite::Owned(owned)
                }
            });
        }
        let source = Source::search(xyz, cell, query)?;
        let rows = Rows::new(&source, n, query.half, repeats(cell, query), dist2)?;
        let need = rows.need();
        match room(need) {
            Some(mut dest) => {
                if !dist2 {
                    dest.dist2 = std::ptr::null_mut();
                }
                // Safety: `room` gave `n + 1` offsets and `need` columns
                // (and distances when that pointer is non-null).
                unsafe { rows.write(dest) };
                Ok(CsrWrite::Direct(need))
            }
            None => {
                let mut owned = PairCsr::default();
                rows.write_vec(&mut owned, dist2)?;
                Ok(CsrWrite::Owned(owned))
            }
        }
    })
}

/// The argument checks of [`pairs_within_csr`], before any search.
pub(crate) fn check(xyz: &[[f64; 3]], query: &PbcQuery<'_>) -> Result<(), Error> {
    if !query.cutoff.is_finite() || query.cutoff <= 0.0 {
        return Err(Error::BadCutoff);
    }
    if let Some(mask) = query.mask {
        if mask.len() != xyz.len() {
            return Err(Error::MaskLen);
        }
    }
    if xyz.len() > i32::MAX as usize {
        return Err(Error::Overflow);
    }
    // A fold rather than `any`: no early exit, so it vectorizes.
    if !xyz
        .as_flattened()
        .iter()
        .fold(true, |ok, v| ok & v.is_finite())
    {
        return Err(Error::NonFinite);
    }
    Ok(())
}

/// Expected pairs, for the pool decision. An open axis has no width to
/// divide by, so a cell with one takes a few neighbours a point.
fn estimate(n: usize, cell: &Cell, query: &PbcQuery<'_>) -> usize {
    if query.periodic == [true, true, true] {
        hit_estimate(n, cell, query.cutoff)
    } else {
        n.saturating_mul(OPEN_PAIRS_PER_POINT)
    }
}

const OPEN_PAIRS_PER_POINT: usize = 8;

/// Whether a row can meet one atom through two images. Two images of
/// one atom inside the cutoff differ by a lattice vector shorter than
/// twice the cutoff, and a lattice vector that moves along periodic axis
/// `a` is at least that axis's perpendicular width. The slack covers the
/// rounding in the widths and in the stored distances.
fn repeats(cell: &Cell, query: &PbcQuery<'_>) -> bool {
    let w = cell.widths();
    (0..3).any(|a| query.periodic[a] && 2.0 * query.cutoff >= w[a] * (1.0 - 1.0e-6))
}

/// Most lane counters kept at once (`lanes * n`). Past this, several
/// search buffers share a lane.
const LANE_COUNTS: usize = 1 << 25;

/// Rows below which the row passes run on one thread.
const PARALLEL_ROWS: usize = 4096;

/// Row blocks per thread in the row passes.
const BLOCKS_PER_THREAD: usize = 8;

/// The hits one search found: the periodic walk's buffers, or an open or
/// mixed cell's half rows.
enum Source {
    /// The pair walk's hit buffers: the v0.6.0 rows of a periodic cell,
    /// kept as the reference the gathered rows are tested against.
    #[cfg(test)]
    Found(Found),
    Pairs(Vec<Pair>),
}

impl Source {
    fn search(xyz: &[[f64; 3]], cell: &Cell, query: &PbcQuery<'_>) -> Result<Self, Error> {
        #[cfg(test)]
        if query.periodic == [true, true, true] {
            let plan = plan_unshifted(
                xyz,
                cell,
                query.cutoff,
                query.mask,
                query.cell_hint,
                query.half,
            )?;
            return Ok(Source::Found(plan.search()));
        }
        let half = PbcQuery::new(query.periodic, query.cutoff)
            .mask(query.mask)
            .cell_hint(query.cell_hint)
            .half(true)
            .closed(query.closed)
            .exclude_self_images(true);
        Ok(Source::Pairs(pairs_within_pbc(xyz, cell, &half)?))
    }

    /// The hits in at most `most` lanes, in search order.
    fn lanes(&self, most: usize) -> Vec<Lane<'_>> {
        match self {
            #[cfg(test)]
            Source::Found(found) => {
                let chunks = &found.chunks[..];
                let count = chunks.len().clamp(1, most);
                (0..count)
                    .map(|t| {
                        Lane::Hits(
                            &chunks[chunks.len() * t / count..chunks.len() * (t + 1) / count],
                        )
                    })
                    .collect()
            }
            Source::Pairs(rows) => {
                let count = width().clamp(1, most);
                (0..count)
                    .map(|t| {
                        Lane::Pairs(&rows[rows.len() * t / count..rows.len() * (t + 1) / count])
                    })
                    .collect()
            }
        }
    }
}

/// One lane's hits. Each is an atom pair and the squared distance of one
/// of its images, seen once.
enum Lane<'a> {
    #[cfg(test)]
    Hits(&'a [Scratch]),
    Pairs(&'a [Pair]),
}

impl Lane<'_> {
    fn len(&self) -> usize {
        match self {
            #[cfg(test)]
            Lane::Hits(chunks) => chunks.iter().map(|c| c.js.len()).sum(),
            Lane::Pairs(rows) => rows.len(),
        }
    }

    #[inline(always)]
    fn each(&self, mut visit: impl FnMut(u32, u32, f64)) {
        match self {
            #[cfg(test)]
            Lane::Hits(chunks) => {
                for chunk in chunks.iter() {
                    for ((&a, &b), &d) in chunk.atom.iter().zip(&chunk.js).zip(&chunk.d2) {
                        visit(a, b, d);
                    }
                }
            }
            Lane::Pairs(rows) => {
                for pair in rows.iter() {
                    visit(pair.i as u32, pair.j as u32, pair.dist2);
                }
            }
        }
    }
}

/// A buffer several jobs write at disjoint slots.
struct Shared<T>(*mut T);
impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Shared<T> {}
// Safety: jobs write distinct slots and read only slots they wrote, or
// slots no job writes in that pass.
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}
impl<T> Shared<T> {
    fn null() -> Self {
        Shared(std::ptr::null_mut())
    }

    fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// # Safety
    /// `index` is inside the buffer.
    unsafe fn at(self, index: usize) -> *mut T {
        unsafe { self.0.add(index) }
    }
}

/// Threads the row passes may use.
fn width() -> usize {
    #[cfg(feature = "parallel")]
    {
        crate::pool_width().max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// `job(t)` for `t` in `0..count`, on the pool when it has more than one
/// thread.
fn each_index(count: usize, job: impl Fn(usize) + Sync + Send) {
    #[cfg(feature = "parallel")]
    if count > 1 && width() > 1 {
        use rayon::prelude::*;
        (0..count).into_par_iter().for_each(job);
        return;
    }
    (0..count).for_each(job);
}

/// `job(t)` for `t` in `0..count`, in order.
fn map_index<R: Send>(count: usize, job: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "parallel")]
    if count > 1 && width() > 1 {
        use rayon::prelude::*;
        return (0..count).into_par_iter().map(job).collect();
    }
    (0..count).map(job).collect()
}

/// Rows `lo..hi` of block `b` of `blocks` over `n` rows.
fn span(n: usize, blocks: usize, b: usize) -> (usize, usize) {
    (n * b / blocks, n * (b + 1) / blocks)
}

/// Exclusive prefix of `sums`, and their total.
fn bases(sums: &[usize]) -> (Vec<usize>, usize) {
    let mut run = 0usize;
    let base = sums
        .iter()
        .map(|&s| {
            let at = run;
            run += s;
            at
        })
        .collect();
    (base, run)
}

/// Rows counted and placed. Each lane has its slots in every row; the
/// columns are written by [`Rows::write`].
struct Rows<'a> {
    lanes: Vec<Lane<'a>>,
    n: usize,
    half: bool,
    blocks: usize,
    /// `next[l * n + i]`: lane `l`'s next slot in row `i`.
    next: Vec<u32>,
    /// Row starts before repeated columns are dropped, `n + 1` values.
    start: Vec<usize>,
    /// The rows of a search that can repeat a column, already sorted
    /// and deduplicated.
    staged: Option<Staged>,
}

struct Staged {
    /// Sorted unique columns at the front of each `Rows::start` slot.
    cols: Vec<i32>,
    /// Their distances, or empty.
    dist2: Vec<f64>,
    /// Row starts after deduplication, `n + 1` values.
    start: Vec<usize>,
}

impl<'a> Rows<'a> {
    fn new(
        source: &'a Source,
        n: usize,
        half: bool,
        repeat: bool,
        dist2: bool,
    ) -> Result<Self, Error> {
        Self::with_lanes(source, n, half, repeat, dist2, (LANE_COUNTS / n).max(1))
    }

    /// [`Rows::new`] with at most `most` lanes.
    fn with_lanes(
        source: &'a Source,
        n: usize,
        half: bool,
        repeat: bool,
        dist2: bool,
        most: usize,
    ) -> Result<Self, Error> {
        let lanes = source.lanes(most);
        let hits: usize = lanes.iter().map(Lane::len).sum();
        let per_hit = if half { 1 } else { 2 };
        if hits
            .checked_mul(per_hit)
            .map_or(true, |e| e > i32::MAX as usize)
        {
            return Err(Error::Overflow);
        }
        let blocks = if n >= PARALLEL_ROWS && width() > 1 {
            (width() * BLOCKS_PER_THREAD).min(n)
        } else {
            1
        };
        let mut rows = Rows {
            lanes,
            n,
            half,
            blocks,
            next: Vec::new(),
            start: vec![0; n + 1],
            staged: None,
        };
        rows.count();
        rows.place();
        if repeat {
            rows.stage(dist2)?;
        }
        Ok(rows)
    }

    /// Columns the rows hold.
    fn need(&self) -> usize {
        match &self.staged {
            Some(staged) => staged.start[self.n],
            None => self.start[self.n],
        }
    }

    /// Each lane's entries per row. A periodic image of the source is
    /// not a row entry.
    fn count(&mut self) {
        let n = self.n;
        let half = self.half;
        let lanes = &self.lanes;
        self.next.clear();
        self.next.resize(lanes.len() * n, 0);
        let next = Shared(self.next.as_mut_ptr());
        each_index(lanes.len(), |l| {
            // Safety: lane `l` counts into its own `n` entries alone.
            let count = unsafe { std::slice::from_raw_parts_mut(next.at(l * n), n) };
            lanes[l].each(|a, b, _| {
                if a == b {
                    return;
                }
                if half {
                    count[a.min(b) as usize] += 1;
                } else {
                    count[a as usize] += 1;
                    count[b as usize] += 1;
                }
            });
        });
    }

    /// Row starts, and each lane's first slot in each row: lane `l`
    /// writes after lanes `0..l` in every row.
    fn place(&mut self) {
        let (n, blocks) = (self.n, self.blocks);
        let next = &self.next;
        let sums = map_index(blocks, |b| {
            let (lo, hi) = span(n, blocks, b);
            next.chunks(n)
                .map(|count| count[lo..hi].iter().map(|&c| c as usize).sum::<usize>())
                .sum()
        });
        let (base, total) = bases(&sums);
        let lanes: Vec<Shared<u32>> = self
            .next
            .chunks_mut(n)
            .map(|c| Shared(c.as_mut_ptr()))
            .collect();
        let start = Shared(self.start.as_mut_ptr());
        // Block `b` reads and writes rows `lo..hi` of every lane alone.
        each_index(blocks, |b| {
            let (lo, hi) = span(n, blocks, b);
            let mut run = base[b];
            for i in lo..hi {
                unsafe {
                    *start.at(i) = run;
                    for lane in &lanes {
                        let slot = lane.at(i);
                        let count = *slot as usize;
                        // `run` is below the column total, which fits i32.
                        *slot = run as u32;
                        run += count;
                    }
                }
            }
        });
        self.start[n] = total;
    }

    /// Every lane writes its columns into its slots of `cols`, and
    /// distances into `dist2` unless it is null.
    ///
    /// # Safety
    /// `cols`, and `dist2` when non-null, are writable for
    /// `self.start[n]` values.
    unsafe fn fill(&mut self, cols: Shared<i32>, dist2: Shared<f64>) {
        let half = self.half;
        let lanes = &self.lanes;
        let next: Vec<Shared<u32>> = self
            .next
            .chunks_mut(self.n)
            .map(|c| Shared(c.as_mut_ptr()))
            .collect();
        // Lane `l` owns its row of `next`, and its slots in `cols` are its own.
        each_index(lanes.len(), |l| {
            let next = next[l];
            let put = |row: u32, col: u32, d: f64| unsafe {
                let slot = next.at(row as usize);
                let at = *slot as usize;
                *slot += 1;
                *cols.at(at) = col as i32;
                if !dist2.is_null() {
                    *dist2.at(at) = d;
                }
            };
            lanes[l].each(|a, b, d| {
                if a == b {
                    return;
                }
                if half {
                    put(a.min(b), a.max(b), d);
                } else {
                    put(a, b, d);
                    put(b, a, d);
                }
            });
        });
    }

    /// Sort each row of `cols` (by column, then distance) at the
    /// [`Rows::start`] slots. With `unique` set, also drop repeated
    /// columns, keeping the nearest image, and return each row's length.
    ///
    /// # Safety
    /// `cols`, and `dist2` when non-null, hold the rows [`Rows::fill`]
    /// wrote.
    unsafe fn sort(&self, cols: Shared<i32>, dist2: Shared<f64>, unique: bool) -> Vec<u32> {
        let (n, blocks) = (self.n, self.blocks);
        let start = &self.start;
        let lens = map_index(blocks, |b| {
            let (lo, hi) = span(n, blocks, b);
            let mut lens = Vec::with_capacity(if unique { hi - lo } else { 0 });
            let mut scratch = Vec::new();
            for i in lo..hi {
                let (s, e) = (start[i], start[i + 1]);
                // Row `i` is `s..e`, which only this block touches.
                let row = unsafe { std::slice::from_raw_parts_mut(cols.at(s), e - s) };
                let d2 = (!dist2.is_null())
                    .then(|| unsafe { std::slice::from_raw_parts_mut(dist2.at(s), e - s) });
                sort_row(row, d2, &mut scratch);
                if unique {
                    let d2 = (!dist2.is_null())
                        .then(|| unsafe { std::slice::from_raw_parts_mut(dist2.at(s), e - s) });
                    lens.push(dedup_row(row, d2) as u32);
                }
            }
            lens
        });
        lens.concat()
    }

    /// Fill a staging buffer, sort and deduplicate each row, and set the
    /// packed row starts.
    fn stage(&mut self, dist2: bool) -> Result<(), Error> {
        let (n, blocks) = (self.n, self.blocks);
        let total = self.start[n];
        let mut cols: Vec<i32> = Vec::new();
        let mut d2: Vec<f64> = Vec::new();
        try_hold(&mut cols, total)?;
        if dist2 {
            try_hold(&mut d2, total)?;
        }
        let cp = Shared(cols.as_mut_ptr());
        let dp = if dist2 {
            Shared(d2.as_mut_ptr())
        } else {
            Shared::null()
        };
        // Safety: both buffers hold `total` values, every one of which
        // the fill writes; the sort reads only those.
        let lens = unsafe {
            self.fill(cp, dp);
            cols.set_len(total);
            if dist2 {
                d2.set_len(total);
            }
            self.sort(cp, dp, true)
        };
        let sums = map_index(blocks, |b| {
            let (lo, hi) = span(n, blocks, b);
            lens[lo..hi].iter().map(|&c| c as usize).sum()
        });
        let (base, packed) = bases(&sums);
        let mut start = vec![0usize; n + 1];
        let sp = Shared(start.as_mut_ptr());
        each_index(blocks, |b| {
            let (lo, hi) = span(n, blocks, b);
            let mut run = base[b];
            for (i, &len) in (lo..hi).zip(&lens[lo..hi]) {
                // Block `b` writes starts `lo..hi` alone.
                unsafe { *sp.at(i) = run };
                run += len as usize;
            }
        });
        start[n] = packed;
        self.staged = Some(Staged {
            cols,
            dist2: d2,
            start,
        });
        Ok(())
    }

    /// Write the offsets, the columns, and (when `dest.dist2` is non-null)
    /// the distances.
    ///
    /// # Safety
    /// `dest.offsets` is writable for `n + 1` values, and `dest.cols`,
    /// and `dest.dist2` when non-null, for [`Rows::need`] values. A
    /// non-null `dest.dist2` needs rows built with distances.
    unsafe fn write(mut self, dest: CsrDest) {
        let (n, blocks) = (self.n, self.blocks);
        let cols = Shared(dest.cols);
        let dist2 = Shared(dest.dist2);
        let offsets = Shared(dest.offsets);
        let Some(staged) = self.staged.take() else {
            unsafe {
                self.fill(cols, dist2);
                self.sort(cols, dist2, false);
            }
            write_offsets(offsets, &self.start, blocks);
            return;
        };
        let raw = &self.start;
        let with_d2 = !dist2.is_null() && !staged.dist2.is_empty();
        debug_assert!(dist2.is_null() || with_d2 || staged.cols.is_empty());
        each_index(blocks, |b| {
            let (lo, hi) = span(n, blocks, b);
            for i in lo..hi {
                let at = staged.start[i];
                let len = staged.start[i + 1] - at;
                let from = raw[i];
                // Block `b` writes the packed rows `lo..hi` alone.
                unsafe {
                    std::ptr::copy_nonoverlapping(staged.cols.as_ptr().add(from), cols.at(at), len);
                    if with_d2 {
                        std::ptr::copy_nonoverlapping(
                            staged.dist2.as_ptr().add(from),
                            dist2.at(at),
                            len,
                        );
                    }
                }
            }
        });
        write_offsets(offsets, &staged.start, blocks);
    }

    /// [`Rows::write`] into `out`, which this resizes.
    fn write_vec(self, out: &mut PairCsr, dist2: bool) -> Result<(), Error> {
        let n = self.n;
        let need = self.need();
        out.offsets.clear();
        out.cols.clear();
        out.dist2.clear();
        try_hold(&mut out.offsets, n + 1)?;
        try_hold(&mut out.cols, need)?;
        if dist2 {
            try_hold(&mut out.dist2, need)?;
        }
        let dest = CsrDest {
            offsets: out.offsets.as_mut_ptr(),
            cols: out.cols.as_mut_ptr(),
            dist2: if dist2 {
                out.dist2.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
        };
        // Safety: the buffers were reserved for `n + 1` offsets and
        // `need` columns and distances, and `write` sets every one.
        unsafe {
            self.write(dest);
            out.offsets.set_len(n + 1);
            out.cols.set_len(need);
            if dist2 {
                out.dist2.set_len(need);
            }
        }
        Ok(())
    }
}

/// `offsets[i] = start[i]` for every start.
fn write_offsets(offsets: Shared<i32>, start: &[usize], blocks: usize) {
    let n = start.len() - 1;
    each_index(blocks, |b| {
        let (lo, hi) = span(n, blocks, b);
        for (i, &s) in start.iter().enumerate().take(hi).skip(lo) {
            // Block `b` writes offsets `lo..hi` alone; the total fits i32.
            unsafe { *offsets.at(i) = s as i32 };
        }
    });
    unsafe { *offsets.at(n) = start[n] as i32 };
}

/// Rows up to this length sort by insertion.
const SHORT_ROW: usize = 24;

/// Sort `cols` ascending, carrying `dist2` along and breaking a tie on
/// the distance (a repeated column is one atom through two images).
fn sort_row(cols: &mut [i32], dist2: Option<&mut [f64]>, scratch: &mut Vec<(i32, u64)>) {
    let Some(d2) = dist2 else {
        cols.sort_unstable();
        return;
    };
    // Distances are not negative, so their bits sort as the values do.
    if cols.len() <= SHORT_ROW {
        for k in 1..cols.len() {
            let key = (cols[k], d2[k].to_bits());
            let d = d2[k];
            let mut at = k;
            while at > 0 && (cols[at - 1], d2[at - 1].to_bits()) > key {
                cols[at] = cols[at - 1];
                d2[at] = d2[at - 1];
                at -= 1;
            }
            cols[at] = key.0;
            d2[at] = d;
        }
        return;
    }
    scratch.clear();
    scratch.extend(cols.iter().zip(d2.iter()).map(|(&c, &d)| (c, d.to_bits())));
    scratch.sort_unstable();
    for (k, &(c, d)) in scratch.iter().enumerate() {
        cols[k] = c;
        d2[k] = f64::from_bits(d);
    }
}

/// Keep the first entry of each column of a sorted row at its front, and
/// return how many there are.
fn dedup_row(cols: &mut [i32], mut dist2: Option<&mut [f64]>) -> usize {
    let mut kept = 0usize;
    for k in 0..cols.len() {
        if kept > 0 && cols[kept - 1] == cols[k] {
            continue;
        }
        cols[kept] = cols[k];
        if let Some(d2) = dist2.as_deref_mut() {
            d2[kept] = d2[k];
        }
        kept += 1;
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
        #[cfg(feature = "parallel")]
        {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(f)
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = threads;
            f()
        }
    }

    /// `n` uniform points in an `edge` cube, and a sheared cell of the
    /// same volume when `tilt` is nonzero.
    fn frame(n: usize, edge: f64, tilt: f64, seed: u64) -> (Vec<[f64; 3]>, Cell) {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let xyz = (0..n)
            .map(|_| [edge * next(), edge * next(), edge * next()])
            .collect();
        let cell = Cell::from_vectors(
            [edge, 0.0, 0.0],
            [tilt * edge, edge, 0.0],
            [0.5 * tilt * edge, -0.5 * tilt * edge, edge],
            [0.0; 3],
        )
        .unwrap();
        (xyz, cell)
    }

    /// The rows from `pairs_within_pbc`: drop self images, keep the
    /// nearest image of each column, then sort.
    fn reference(xyz: &[[f64; 3]], cell: &Cell, query: &PbcQuery<'_>) -> PairCsr {
        let full = PbcQuery::new(query.periodic, query.cutoff)
            .mask(query.mask)
            .cell_hint(query.cell_hint)
            .closed(query.closed);
        let mut rows = vec![std::collections::BTreeMap::<i32, f64>::new(); xyz.len()];
        for pair in pairs_within_pbc(xyz, cell, &full).unwrap() {
            if pair.i == pair.j || (query.half && pair.j < pair.i) {
                continue;
            }
            let at = rows[pair.i].entry(pair.j as i32).or_insert(pair.dist2);
            *at = at.min(pair.dist2);
        }
        let mut out = PairCsr {
            offsets: vec![0],
            ..PairCsr::default()
        };
        for row in rows {
            for (j, d) in row {
                out.cols.push(j);
                out.dist2.push(d);
            }
            out.offsets.push(out.cols.len() as i32);
        }
        out
    }

    fn bits(csr: &PairCsr) -> (Vec<i32>, Vec<i32>, Vec<u64>) {
        (
            csr.offsets.clone(),
            csr.cols.clone(),
            csr.dist2.iter().map(|d| d.to_bits()).collect(),
        )
    }

    #[test]
    fn rows_match_the_pair_list_on_every_thread_count() {
        for (tilt, edge, cutoff) in [(0.0, 50.0, 3.5), (0.2, 50.0, 3.5), (0.0, 6.0, 3.5)] {
            let n = (0.0334 * edge * edge * edge) as usize;
            let (xyz, cell) = frame(n, edge, tilt, 0x9e37_79b9_7f4a_7c15);
            for half in [false, true] {
                let query = PbcQuery::periodic(cutoff).half(half);
                let want = reference(&xyz, &cell, &query);
                for threads in [1, 2, 8] {
                    let mut got = PairCsr::default();
                    on_threads(threads, || {
                        pairs_within_csr(&xyz, &cell, &query, true, &mut got)
                    })
                    .unwrap();
                    assert_eq!(
                        bits(&got),
                        bits(&want),
                        "tilt {tilt} edge {edge} half {half} threads {threads}"
                    );
                    let mut bare = PairCsr::default();
                    on_threads(threads, || {
                        pairs_within_csr(&xyz, &cell, &query, false, &mut bare)
                    })
                    .unwrap();
                    assert_eq!(bare.offsets, want.offsets);
                    assert_eq!(bare.cols, want.cols);
                    assert!(bare.dist2.is_empty());
                }
            }
        }
    }

    #[test]
    fn shared_lanes_write_the_same_rows() {
        for edge in [24.0, 5.0] {
            let n = (0.0334 * edge * edge * edge) as usize;
            let (xyz, cell) = frame(n, edge, 0.1, 0x2545_f491_4f6c_dd1d);
            let query = PbcQuery::periodic(3.5);
            let repeat = repeats(&cell, &query);
            assert_eq!(repeat, edge < 7.0);
            let rows = |most: usize, half: bool| {
                on_threads(8, || {
                    let source = Source::search(&xyz, &cell, &query).unwrap();
                    let rows = Rows::with_lanes(&source, n, half, repeat, true, most).unwrap();
                    let mut out = PairCsr::default();
                    rows.write_vec(&mut out, true).unwrap();
                    bits(&out)
                })
            };
            for half in [false, true] {
                let one = rows(1, half);
                assert_eq!(rows(3, half), one);
                assert_eq!(rows(64, half), one);
            }
        }
    }

    /// A box shorter than twice the cutoff stages its rows; with no pair
    /// they hold no distance, and asking for distances is still fine,
    /// gathered on a periodic cell or written from the walk's hits on a
    /// cell with an open axis.
    #[test]
    fn empty_staged_rows_with_distances() {
        let cell = Cell::ortho(6.0, 6.0, 6.0).unwrap();
        for periodic in [[true; 3], [true, true, false]] {
            let mut out = PairCsr::default();
            pairs_within_csr(
                &[[1.0, 2.0, 3.0]],
                &cell,
                &PbcQuery::new(periodic, 3.5),
                true,
                &mut out,
            )
            .unwrap();
            assert_eq!(out.offsets, [0, 0], "{periodic:?}");
            assert!(out.cols.is_empty() && out.dist2.is_empty(), "{periodic:?}");
        }
    }

    #[test]
    fn sort_row_orders_by_column_then_distance() {
        let mut scratch = Vec::new();
        for len in [0usize, 1, 5, SHORT_ROW, SHORT_ROW + 1, 200] {
            let mut cols: Vec<i32> = (0..len).map(|k| ((k * 7919) % 13) as i32).collect();
            let mut d2: Vec<f64> = (0..len)
                .map(|k| ((k * 104_729) % 17) as f64 * 0.25)
                .collect();
            let mut want: Vec<(i32, u64)> = cols
                .iter()
                .zip(&d2)
                .map(|(&c, d)| (c, d.to_bits()))
                .collect();
            want.sort_unstable();
            sort_row(&mut cols, Some(&mut d2), &mut scratch);
            let got: Vec<(i32, u64)> = cols
                .iter()
                .zip(&d2)
                .map(|(&c, d)| (c, d.to_bits()))
                .collect();
            assert_eq!(got, want, "length {len}");
            let kept = dedup_row(&mut cols, Some(&mut d2));
            want.dedup_by_key(|e| e.0);
            let got: Vec<(i32, u64)> = cols[..kept]
                .iter()
                .zip(&d2[..kept])
                .map(|(&c, d)| (c, d.to_bits()))
                .collect();
            assert_eq!(got, want, "length {len} after dedup");
        }
    }

    /// The rows of v0.6.0: the pair walk's hit buffers, counted, placed,
    /// filled, and sorted per row.
    fn v060(xyz: &[[f64; 3]], cell: &Cell, query: &PbcQuery<'_>) -> PairCsr {
        let source = Source::search(xyz, cell, query).unwrap();
        let rows = Rows::new(&source, xyz.len(), query.half, repeats(cell, query), true).unwrap();
        let mut out = PairCsr::default();
        rows.write_vec(&mut out, true).unwrap();
        out
    }

    /// Every row against all pairs and images: the columns, but those
    /// within rounding of the cutoff, and the distances to rounding.
    fn assert_brute(
        got: &PairCsr,
        xyz: &[[f64; 3]],
        cell: &Cell,
        query: &PbcQuery<'_>,
        label: &str,
    ) {
        let reach = repeats_of(cell, query.cutoff);
        let cut2 = query.cutoff * query.cutoff;
        let tol = 1e-9 * (1.0 + cut2);
        let active = |i: usize| query.mask.map_or(true, |m| m[i]);
        for i in 0..xyz.len() {
            let row = got.offsets[i] as usize..got.offsets[i + 1] as usize;
            let mut have: std::collections::BTreeMap<i32, f64> = got.cols[row.clone()]
                .iter()
                .zip(&got.dist2[row])
                .map(|(&j, &d)| (j, d))
                .collect();
            for j in 0..xyz.len() {
                if j == i || !active(i) || !active(j) || (query.half && j < i) {
                    continue;
                }
                let mut best = f64::INFINITY;
                for a in -reach[0]..=reach[0] {
                    for b in -reach[1]..=reach[1] {
                        for c in -reach[2]..=reach[2] {
                            let t = cell.lattice_shift(a, b, c);
                            let mut d2 = 0.0;
                            for k in 0..3 {
                                let d = xyz[j][k] - xyz[i][k] + t[k];
                                d2 += d * d;
                            }
                            best = best.min(d2);
                        }
                    }
                }
                match have.remove(&(j as i32)) {
                    Some(d) => {
                        assert!(best < cut2 + tol, "{label}: {i}-{j} at {best} listed");
                        assert!(
                            (d - best).abs() <= tol,
                            "{label}: {i}-{j} at {d}, brute {best}"
                        );
                    }
                    None => assert!(best >= cut2 - tol, "{label}: {i}-{j} at {best} missing"),
                }
            }
            assert!(
                have.is_empty(),
                "{label}: row {i} lists {have:?} beyond the pairs"
            );
        }
    }

    /// Lattice images each way that can hold a pair within `cutoff`, and
    /// one more for the fold.
    fn repeats_of(cell: &Cell, cutoff: f64) -> [i32; 3] {
        cell.widths().map(|w| (cutoff / w).ceil() as i32 + 1)
    }

    /// CSR rows against v0.6.0's, bit for bit, on every SIMD path at
    /// one and eight threads, and against brute force: orthorhombic and
    /// triclinic cells, bins below, at, and above the cutoff, boxes
    /// shorter than twice the cutoff, atom counts on both sides of the
    /// parallel thresholds, half and full rows, with and without a mask.
    #[test]
    fn csr_rows_match_v060_and_brute_force() {
        let cutoff = 3.5;
        let at_threshold = |pairs: usize| {
            // `hit_estimate` at the bench density: 0.5 n rho 4/3 pi r^3.
            (pairs as f64 / (0.5 * 0.0334 * 4.1887902047863905 * cutoff * cutoff * cutoff)) as usize
        };
        // `PARALLEL_PAIRS`, which builds only with `parallel`.
        let split = at_threshold(2_000);
        let grid = super::super::PARALLEL_GRID;
        let mut sizes: Vec<(usize, Option<f64>)> = vec![(1, None), (2, None), (40, None)];
        for n in [split - 2, split, split + 2, grid - 1, grid, grid + 1] {
            sizes.push((n, None));
        }
        // Boxes past half the box: one atom meets another through
        // several images.
        sizes.push((7, Some(6.0)));
        sizes.push((24, Some(9.0)));
        sizes.push((4096, None));
        let paths = crate::simd::paths_to_compare();
        for (case, &(n, edge)) in sizes.iter().enumerate() {
            let edge = edge.unwrap_or((n as f64 / 0.0334).cbrt());
            // The n=40 system keeps every hint and every half/mask pair.
            // Every other size keeps both tilts, both half lists, both
            // thread counts, and every SIMD path, at the default hint and
            // with no mask. n <= 64 still checks brute force.
            let full = n == 40;
            for tilt in [0.0, 0.3] {
                let (xyz, cell) = frame(n, edge, tilt, 0x51ed_270b_27a4_3c11 ^ case as u64);
                let mask: Vec<bool> = (0..n).map(|i| (i * 7919) % 5 != 0).collect();
                let hints: &[Option<f64>] = if full {
                    &[None, Some(0.6 * cutoff), Some(1.7 * cutoff)]
                } else {
                    &[None]
                };
                let halves: &[(bool, bool)] = if full {
                    &[(false, false), (true, false), (false, true), (true, true)]
                } else {
                    &[(false, false), (true, false)]
                };
                for &hint in hints {
                    for &(half, masked) in halves {
                        let query = PbcQuery::periodic(cutoff)
                            .cell_hint(hint)
                            .half(half)
                            .mask(masked.then_some(&mask[..]));
                        let label = format!(
                            "n {n} edge {edge:.2} tilt {tilt} hint {hint:?} half {half} mask {masked}"
                        );
                        let want = on_threads(1, || v060(&xyz, &cell, &query));
                        // The columns do not depend on the bins; brute force
                        // is quadratic, so it checks the small systems.
                        if n <= 64 && hint.is_none() {
                            assert_brute(&want, &xyz, &cell, &query, &label);
                        }
                        for threads in [1, 8] {
                            for &path in &paths {
                                let (got, bare) = on_threads(threads, || {
                                    let _path = crate::simd::force(path);
                                    let mut got = PairCsr::default();
                                    pairs_within_csr(&xyz, &cell, &query, true, &mut got).unwrap();
                                    let mut bare = PairCsr::default();
                                    pairs_within_csr(&xyz, &cell, &query, false, &mut bare)
                                        .unwrap();
                                    (got, bare)
                                });
                                let label = format!("{label} threads {threads} {}", path.name());
                                assert_eq!(bits(&got), bits(&want), "{label}");
                                assert_eq!(
                                    (&bare.offsets, &bare.cols),
                                    (&want.offsets, &want.cols),
                                    "{label}"
                                );
                                assert!(bare.dist2.is_empty(), "{label}");
                            }
                        }
                    }
                }
            }
        }
    }
}
