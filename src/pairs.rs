//! Cutoff pair list with integer cell shifts.
//!
//! [`knearest`](crate::knearest) unique-indexes the neighbour and
//! drops the image. This walk keeps every atom-image pair whose
//! squared distance is strictly below `cutoff²`, including periodic
//! self-images. Each unordered pair is tested once. Hits from a bin
//! are buffered, then the rows are written. A full list writes both
//! `(i, j, S)` and `(j, i, -S)`. Displacement is `q - p + lattice_shift(S)`,
//! which is [`Cell::dist2_shifted`](crate::Cell::dist2_shifted). This is
//! not [`dist2_ortho_diffs`](crate::dist2_ortho_diffs): that Highway
//! kernel wraps a raw difference into the central cell, and a second
//! wrap on an already shifted delta changes which rows survive.

use crate::bins::{self, axis_gap};
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
    let est = hit_estimate(xyz.len(), simbox, cutoff);
    if walk_threads(est) == 1 {
        let plan = plan(xyz, simbox, cutoff, mask, cell_hint, half)?;
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        if let Some(words) = plan.fused().filter(|_| knobs::fused()) {
            return Ok(plan.fused_rows(&words));
        }
        return Ok(plan.search().into_pairs());
    }
    // The buffer is reserved here, on the caller's thread, from the
    // ideal-gas estimate; search and write then run in one pool entry.
    let rows_guess = row_guess(est, half);
    let mut out: Vec<Pair> = Vec::with_capacity(rows_guess);
    advise_huge(&mut out);
    let base = RowPtr(out.as_mut_ptr() as *mut std::mem::MaybeUninit<Pair>);
    let (rows, rest) = in_pool(est, || -> Result<(usize, Option<Found>), Error> {
        let found = plan(xyz, simbox, cutoff, mask, cell_hint, half)?.search();
        let rows = found.rows();
        if rows > rows_guess {
            return Ok((rows, Some(found)));
        }
        // Safety: `out` holds `rows_guess >= rows` rows.
        unsafe { found.write_pairs(base.ptr()) };
        Ok((rows, None))
    })?;
    if let Some(found) = rest {
        out.reserve(rows);
        // Safety: `out` now holds `rows` rows.
        unsafe { found.write_pairs(out.as_mut_ptr()) };
    }
    // Safety: every row below `rows` is written.
    unsafe { out.set_len(rows) };
    Ok(out)
}

/// Ask Linux to back the 2 MB-aligned interior of a large buffer this
/// library allocated with transparent huge pages, so the row stores take
/// one page walk per 2 MB rather than per 4 KB. It is advice: the system's
/// policy decides, and nothing outside the buffer is touched.
fn advise_huge<T>(buf: &mut Vec<T>) {
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        const HUGE: usize = 1 << 21;
        const MADV_HUGEPAGE: std::ffi::c_int = 14;
        extern "C" {
            fn madvise(
                addr: *mut std::ffi::c_void,
                len: usize,
                advice: std::ffi::c_int,
            ) -> std::ffi::c_int;
        }
        let bytes = buf.capacity().saturating_mul(std::mem::size_of::<T>());
        if bytes < 2 * HUGE {
            return;
        }
        let start = buf.as_mut_ptr() as usize;
        let lo = (start + HUGE - 1) & !(HUGE - 1);
        let hi = (start + bytes) & !(HUGE - 1);
        if hi > lo {
            // Safety: `lo..hi` lies inside the buffer's allocation, and the
            // advice changes neither its contents nor its mapping.
            unsafe { madvise(lo as *mut std::ffi::c_void, hi - lo, MADV_HUGEPAGE) };
        }
    }
    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    let _ = buf;
}

/// Rows to reserve for `hits` expected pairs: a quarter more than the
/// ideal gas, two rows per pair on a full list.
fn row_guess(hits: usize, half: bool) -> usize {
    let rows = if half { hits } else { hits.saturating_mul(2) };
    rows.saturating_add(rows / 4).saturating_add(64)
}

/// The same rows as [`pairs_within`], as four columns.
///
/// `i`, `j`, and `shift` are the vesin / tonari `ijS` layout and
/// `dist2` is the squared distance. Row `t` of each column is one
/// atom-image. The vectors are cleared, then filled once; their
/// capacity is kept for the next call. An index above `i32::MAX` is
/// [`Error::Overflow`].
pub fn pairs_within_columns(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    half: bool,
    out: &mut PairColumns,
) -> Result<(), Error> {
    if xyz.len() > i32::MAX as usize {
        return Err(Error::Overflow);
    }
    let est = hit_estimate(xyz.len(), simbox, cutoff);
    if walk_threads(est) == 1 {
        let plan = plan(xyz, simbox, cutoff, mask, cell_hint, half)?;
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        if plan.fused().is_some() && knobs::fused() {
            plan.fused_columns(out);
            return Ok(());
        }
        plan.search().fill_columns(out);
        return Ok(());
    }
    // Columns are reserved on the caller's thread; search and write run
    // in one pool entry.
    let rows_guess = row_guess(est, half);
    out.i.clear();
    out.j.clear();
    out.shift.clear();
    out.dist2.clear();
    out.i.reserve(rows_guess);
    out.j.reserve(rows_guess);
    out.shift.reserve(rows_guess);
    out.dist2.reserve(rows_guess);
    out.advise_huge();
    let ptrs = ColumnPtrs {
        i: RowPtr(out.i.as_mut_ptr() as *mut std::mem::MaybeUninit<i32>),
        j: RowPtr(out.j.as_mut_ptr() as *mut std::mem::MaybeUninit<i32>),
        shift: RowPtr(out.shift.as_mut_ptr() as *mut std::mem::MaybeUninit<i32>),
        d2: RowPtr(out.dist2.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>),
    };
    let (rows, rest) = in_pool(est, || -> Result<(usize, Option<Found>), Error> {
        let found = plan(xyz, simbox, cutoff, mask, cell_hint, half)?.search();
        let rows = found.rows();
        if rows > rows_guess {
            return Ok((rows, Some(found)));
        }
        // Safety: every column holds `rows_guess >= rows` rows.
        unsafe { found.write_columns(ptrs.i.ptr(), ptrs.j.ptr(), ptrs.shift.ptr(), ptrs.d2.ptr()) };
        Ok((rows, None))
    })?;
    match rest {
        Some(found) => found.fill_columns(out),
        // Safety: every row below `rows` is written in every column.
        None => unsafe {
            out.i.set_len(rows);
            out.j.set_len(rows);
            out.shift.set_len(rows);
            out.dist2.set_len(rows);
        },
    }
    Ok(())
}

/// Run `work` on a pool thread when the call will use several threads.
///
/// A caller outside the pool would otherwise run the serial steps while
/// every worker spins, one thread more than the pool has cores. Inside
/// the pool the caller sleeps on the join instead. Output buffers are
/// still allocated on the caller's thread: glibc gives a large block on
/// a worker its own heap and unmaps it when the block is freed, so the
/// next call would fault every page again.
pub(crate) fn in_pool<R: Send>(hits: usize, work: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    if walk_threads(hits) > 1 && rayon::current_thread_index().is_none() {
        return rayon::scope(|_| work());
    }
    let _ = hits;
    work()
}

/// Ideal-gas count of unordered pairs within `cutoff` for `n` points.
pub(crate) fn hit_estimate(n: usize, simbox: &Cell, cutoff: f64) -> usize {
    let w = simbox.widths();
    let volume = (w[0] * w[1] * w[2]).max(1.0e-30);
    let shell = 4.1887902047863905 * cutoff * cutoff * cutoff;
    let pairs = 0.5 * (n as f64) * (n as f64) * shell / volume;
    if pairs.is_finite() {
        pairs.min(usize::MAX as f64 / 2.0) as usize
    } else {
        0
    }
}

/// Cutoff rows as columns: `i[t]`, `j[t]`, `shift[t]`, `dist2[t]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PairColumns {
    /// Source index.
    pub i: Vec<i32>,
    /// Target index.
    pub j: Vec<i32>,
    /// Integer cell shift applied to the target (`S`).
    pub shift: Vec<[i32; 3]>,
    /// Squared distance after the shift.
    pub dist2: Vec<f64>,
}

impl PairColumns {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.i.len()
    }

    /// `true` when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.i.is_empty()
    }

    /// [`advise_huge`] for every column.
    fn advise_huge(&mut self) {
        advise_huge(&mut self.i);
        advise_huge(&mut self.j);
        advise_huge(&mut self.shift);
        advise_huge(&mut self.dist2);
    }
}

/// The grid, stencil, and bounds of one cutoff search.
pub(crate) struct Plan {
    grid: Grid,
    partners: std::sync::Arc<PartnerList>,
    cut2: f64,
    margin: f64,
    simd: u8,
    half: bool,
    threads: usize,
}

impl Plan {
    fn walk(&self) -> Walk<'_> {
        Walk {
            grid: &self.grid,
            cut2: self.cut2,
            margin: self.margin,
            partners: &self.partners,
            simd: self.simd,
        }
    }

    /// Every hit, by cell range.
    pub(crate) fn search(&self) -> Found {
        Found {
            chunks: self.walk().collect(self.threads),
            half: self.half,
            simd: self.simd,
        }
    }

    /// One thread, a full list, and AVX-512: the rows can be written
    /// from the tile kernel in one pass.
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    fn fused(&self) -> Option<RowWords> {
        if self.threads == 1 && !self.half && self.simd == 2 {
            RowWords::of_pair()
        } else {
            None
        }
    }
}

/// [`Plan::search`] after [`plan`].
#[cfg_attr(not(feature = "capi"), allow(dead_code))]
pub(crate) fn search(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    half: bool,
) -> Result<Found, Error> {
    Ok(plan(xyz, simbox, cutoff, mask, cell_hint, half)?.search())
}

pub(crate) fn plan(
    xyz: &[[f64; 3]],
    simbox: &Cell,
    cutoff: f64,
    mask: Option<&[bool]>,
    cell_hint: Option<f64>,
    half: bool,
) -> Result<Plan, Error> {
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(Error::BadCutoff);
    }
    if xyz.is_empty() {
        return Err(Error::Empty);
    }
    let n = xyz.len();
    if n > u32::MAX as usize {
        return Err(Error::Overflow);
    }
    if let Some(m) = mask {
        if m.len() != n {
            return Err(Error::MaskLen);
        }
    }
    let active: Option<Vec<usize>> = mask.map(|m| (0..n).filter(|&i| m[i]).collect());
    let n_act = active.as_ref().map_or(n, |a| a.len());
    if n_act == 0 {
        return Ok(Plan {
            grid: Grid::empty(),
            partners: std::sync::Arc::new(PartnerList {
                key: ([0; 12], [1, 1, 1], [0; 3], 0),
                off: vec![0],
                items: Vec::new(),
                s_max: 0.0,
                d_max: 0.0,
            }),
            cut2: cutoff * cutoff,
            margin: 0.0,
            simd: 0,
            half,
            threads: 1,
        });
    }
    retain_pair_pages();

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
    let dims = bins::grid_dims(simbox, edge)?;
    let cell_min = (w[0] / f64::from(dims[0]))
        .min(w[1] / f64::from(dims[1]))
        .min(w[2] / f64::from(dims[2]));
    let reach_cut = (cutoff / cell_min).ceil();
    if !reach_cut.is_finite() || reach_cut > i32::MAX as f64 {
        return Err(Error::TooManyImages);
    }
    let max_reach = (reach_cut as i32)
        .max(repeats[0] * dims[0])
        .max(repeats[1] * dims[1])
        .max(repeats[2] * dims[2])
        .max(1);
    let cut2 = cutoff * cutoff;
    // One reach for every atom: the shorter gap at either edge of a bin.
    // The box is symmetric, so each unordered pair is visited from one
    // side and written out in both shift directions when `half` is off.
    let reach = uniform_reach(dims, w, cut2, max_reach);
    let threads = walk_threads(hit_estimate(n_act, simbox, cutoff));
    let grid_threads = if n_act >= knobs::grid_atoms() {
        threads
    } else {
        1
    };
    let grid = Grid::build(xyz, simbox, active.as_deref(), dims, grid_threads);
    let partners = partners_for(&grid, simbox, reach, cut2);
    let margin = expanded_margin(&grid, &partners, cutoff);
    Ok(Plan {
        grid,
        partners,
        cut2,
        margin,
        simd: simd_mode(),
        half,
        threads,
    })
}

/// Threads for one search with `hits` expected pairs: one without
/// `parallel`, or when the search is too short to pay for waking the pool.
fn walk_threads(hits: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        let threads = rayon::current_num_threads().max(1);
        if threads > 1 && hits >= knobs::split_pairs() {
            return threads;
        }
    }
    let _ = hits;
    1
}

/// glibc drops a large free buffer (`MADV_DONTNEED`) and the next cutoff
/// list faults every page. Keep those pages on the heap. The knobs are
/// `M_TRIM_THRESHOLD` and `M_MMAP_THRESHOLD`.
fn retain_pair_pages() {
    #[cfg(target_os = "linux")]
    {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            // Safety: mallopt is process-wide and these two knobs are integers.
            unsafe {
                extern "C" {
                    fn mallopt(param: i32, value: i32) -> i32;
                }
                let _ = mallopt(-1, -1);
                let _ = mallopt(-3, 256 << 20);
            }
        });
    }
}

struct Partner {
    /// Target bin.
    jc: usize,
    shift_s: [i32; 3],
    shift: [f64; 3],
    /// Source corner minus target corner minus `shift`: a source slot's
    /// relative position plus this is its image relative to the target
    /// corner.
    delta: [f64; 3],
}

/// The bins each bin looks into, with their shifts. It depends on the
/// box, the bin counts, the reach, and the cutoff, not on the atoms, so a
/// search of the same box reuses it; the walk skips an empty target bin.
struct PartnerList {
    key: PartnerKey,
    off: Vec<usize>,
    items: Vec<Partner>,
    /// Largest |component| of any shift and of any `delta`.
    s_max: f64,
    d_max: f64,
}

/// The box (edges and origin, as bits), the bin counts, the reach, and
/// the squared cutoff.
type PartnerKey = ([u64; 12], [i32; 3], [i32; 3], u64);

static PARTNERS: std::sync::Mutex<Option<std::sync::Arc<PartnerList>>> =
    std::sync::Mutex::new(None);

/// [`build_partners`], or the last list when its key matches.
fn partners_for(
    grid: &Grid,
    simbox: &Cell,
    reach: [i32; 3],
    cut2: f64,
) -> std::sync::Arc<PartnerList> {
    let h = simbox.h();
    let o = simbox.origin();
    let mut boxbits = [0u64; 12];
    for a in 0..3 {
        for b in 0..3 {
            boxbits[3 * a + b] = h[a][b].to_bits();
        }
        boxbits[9 + a] = o[a].to_bits();
    }
    let key: PartnerKey = (boxbits, grid.n, reach, cut2.to_bits());
    if let Ok(slot) = PARTNERS.lock() {
        if let Some(list) = slot.as_ref() {
            if list.key == key {
                return std::sync::Arc::clone(list);
            }
        }
    }
    let built = std::sync::Arc::new(build_partners(grid, simbox, reach, cut2, key));
    if let Ok(mut slot) = PARTNERS.lock() {
        *slot = Some(std::sync::Arc::clone(&built));
    }
    built
}

fn bin_gap(src: i32, dst: i32, width: f64) -> f64 {
    let lo_s = f64::from(src) * width;
    let hi_s = f64::from(src + 1) * width;
    let lo_d = f64::from(dst) * width;
    let hi_d = f64::from(dst + 1) * width;
    if hi_s <= lo_d {
        lo_d - hi_s
    } else if hi_d <= lo_s {
        lo_s - hi_d
    } else {
        0.0
    }
}

fn bins_too_far(ortho: bool, src: [i32; 3], dst: [i32; 3], width: [f64; 3], cut2: f64) -> bool {
    let gap = [
        bin_gap(src[0], dst[0], width[0]),
        bin_gap(src[1], dst[1], width[1]),
        bin_gap(src[2], dst[2], width[2]),
    ];
    let bound = if ortho {
        let d = (gap[0] * gap[0] + gap[1] * gap[1] + gap[2] * gap[2]).sqrt();
        let d = bins::certify(d);
        d * d
    } else {
        let d = bins::certify(gap[0].max(gap[1]).max(gap[2]));
        d * d
    };
    bound >= cut2
}

fn build_partners(
    grid: &Grid,
    simbox: &Cell,
    reach: [i32; 3],
    cut2: f64,
    key: PartnerKey,
) -> PartnerList {
    let [nx, ny, nz] = grid.n;
    let ncell = (nx as usize) * (ny as usize) * (nz as usize);
    let width = [
        grid.widths[0] / f64::from(nx),
        grid.widths[1] / f64::from(ny),
        grid.widths[2] / f64::from(nz),
    ];
    let ortho = simbox.is_ortho();
    let [rx, ry, rz] = reach;
    let mut off = vec![0usize; ncell + 1];
    let mut items = Vec::new();
    let (mut s_max, mut d_max) = (0.0f64, 0.0f64);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let cell = ((iz * ny + iy) * nx + ix) as usize;
                let src = [ix, iy, iz];
                for dz in -rz..=rz {
                    for dy in -ry..=ry {
                        for dx in -rx..=rx {
                            if !keep_dir(dx, dy, dz) {
                                continue;
                            }
                            let dst = [ix + dx, iy + dy, iz + dz];
                            if bins_too_far(ortho, src, dst, width, cut2) {
                                continue;
                            }
                            let jc = bins::wrap_cell(dst, grid.n);
                            let shift_s = [
                                dst[0].div_euclid(nx),
                                dst[1].div_euclid(ny),
                                dst[2].div_euclid(nz),
                            ];
                            let shift = simbox.lattice_shift(shift_s[0], shift_s[1], shift_s[2]);
                            let (a, b) = (grid.corners[cell], grid.corners[jc]);
                            let delta = [
                                a[0] - b[0] - shift[0],
                                a[1] - b[1] - shift[1],
                                a[2] - b[2] - shift[2],
                            ];
                            for k in 0..3 {
                                s_max = s_max.max(shift[k].abs());
                                d_max = d_max.max(delta[k].abs());
                            }
                            items.push(Partner {
                                jc,
                                shift_s,
                                shift,
                                delta,
                            });
                        }
                    }
                }
                off[cell + 1] = items.len();
            }
        }
    }
    PartnerList {
        key,
        off,
        items,
        s_max,
        d_max,
    }
}

#[derive(Default)]
struct Coords {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    /// Position relative to the corner of the slot's own bin.
    rx: Vec<f64>,
    ry: Vec<f64>,
    rz: Vec<f64>,
    /// `rx^2 + ry^2 + rz^2`.
    r2: Vec<f64>,
    /// Atom index of each slot.
    id: Vec<u32>,
}

impl Coords {
    /// Empty columns with room for `nslot` slots each.
    fn clear_for(&mut self, nslot: usize) {
        for v in [
            &mut self.x,
            &mut self.y,
            &mut self.z,
            &mut self.rx,
            &mut self.ry,
            &mut self.rz,
            &mut self.r2,
        ] {
            v.clear();
            v.reserve(nslot);
        }
        self.id.clear();
        self.id.reserve(nslot);
    }

    /// # Safety
    /// Every column holds `nslot` written slots.
    unsafe fn set_len(&mut self, nslot: usize) {
        for v in [
            &mut self.x,
            &mut self.y,
            &mut self.z,
            &mut self.rx,
            &mut self.ry,
            &mut self.rz,
            &mut self.r2,
        ] {
            v.set_len(nslot);
        }
        self.id.set_len(nslot);
    }
}

/// Buffers a grid lends back when it drops, so the next search of a
/// similar size neither allocates nor faults them in.
#[derive(Default)]
struct GridBufs {
    offsets: Vec<usize>,
    corners: Vec<[f64; 3]>,
    coords: Coords,
    fold: FoldCols,
    tally: Vec<u32>,
    order: Vec<u32>,
    boxes: Boxes,
}

/// Bounding boxes, in each bin's relative coordinates, of the slot runs
/// the tile kernel reads together: runs of eight from a bin's first slot
/// (targets) and runs of four (sources). A source run and a target run
/// whose boxes are a cutoff apart hold no pair, so the kernel skips them
/// (the cluster pairs of GROMACS, decided per search).
#[derive(Default)]
struct Boxes {
    /// Whether the boxes are built: only for bins of [`CULL_RUNS`] runs of
    /// eight on average. Without them every run start is zero.
    on: bool,
    /// First run of eight, and of four, of each bin; one more entry past
    /// the last bin.
    vstart: Vec<usize>,
    gstart: Vec<usize>,
    /// `lo x y z` then `hi x y z` per run of eight, with eight runs of
    /// padding so a kernel can read eight runs from any bin's first.
    v: [Vec<f64>; 6],
    /// The same per run of four.
    g: [Vec<f64>; 6],
}

impl Boxes {
    /// Run starts of every bin, and room for every box, when `on`.
    fn starts(&mut self, offsets: &[usize], on: bool) {
        self.on = on;
        self.vstart.clear();
        self.gstart.clear();
        if !on {
            self.vstart.resize(offsets.len(), 0);
            self.gstart.resize(offsets.len(), 0);
            return;
        }
        let (mut v, mut g) = (0usize, 0usize);
        for w in offsets.windows(2) {
            self.vstart.push(v);
            self.gstart.push(g);
            let n = w[1] - w[0];
            v += (n + 7) / 8;
            g += (n + 3) / 4;
        }
        self.vstart.push(v);
        self.gstart.push(g);
        // The columns are read and written only through raw pointers, so
        // they stay empty: `fill` writes every run, and the padding is
        // written here.
        for a in self.v.iter_mut() {
            a.clear();
            a.reserve(v + 8);
            // Safety: the capacity covers the padding.
            unsafe { std::ptr::write_bytes(a.as_mut_ptr().add(v), 0, 8) };
        }
        for a in self.g.iter_mut() {
            a.clear();
            a.reserve(g);
        }
    }

    fn ptrs(&mut self) -> BoxPtrs {
        let raw = |a: &mut Vec<f64>| RowPtr(a.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>);
        let [v0, v1, v2, v3, v4, v5] = &mut self.v;
        let [g0, g1, g2, g3, g4, g5] = &mut self.g;
        BoxPtrs {
            v: [raw(v0), raw(v1), raw(v2), raw(v3), raw(v4), raw(v5)],
            g: [raw(g0), raw(g1), raw(g2), raw(g3), raw(g4), raw(g5)],
        }
    }
}

/// Raw [`Boxes`] columns for writes at disjoint bins from several threads.
#[derive(Clone, Copy)]
struct BoxPtrs {
    v: [RowPtr<f64>; 6],
    g: [RowPtr<f64>; 6],
}

impl BoxPtrs {
    /// Boxes of the runs of slots `lo..hi` (one bin), from run `v0` of
    /// eight and `g0` of four.
    ///
    /// # Safety
    /// This thread wrote the slots, the runs are in range, and no other
    /// thread writes them.
    unsafe fn fill(self, slots: SlotPtrs, lo: usize, hi: usize, v0: usize, g0: usize) {
        let (rx, ry, rz) = (slots.rx.ptr(), slots.ry.ptr(), slots.rz.ptr());
        // Runs of four, each run of eight the union of two of them.
        let mut last = ([0.0f64; 3], [0.0f64; 3]);
        for (m, a) in (lo..hi).step_by(4).enumerate() {
            let (mut l, mut h) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for s in a..(a + 4).min(hi) {
                let r = [*rx.add(s), *ry.add(s), *rz.add(s)];
                for ax in 0..3 {
                    l[ax] = l[ax].min(r[ax]);
                    h[ax] = h[ax].max(r[ax]);
                }
            }
            for ax in 0..3 {
                (*self.g[ax].at(g0 + m)).write(l[ax]);
                (*self.g[3 + ax].at(g0 + m)).write(h[ax]);
            }
            let odd = m % 2 == 1;
            if odd {
                for ax in 0..3 {
                    l[ax] = l[ax].min(last.0[ax]);
                    h[ax] = h[ax].max(last.1[ax]);
                }
            }
            if odd || a + 4 >= hi {
                for ax in 0..3 {
                    (*self.v[ax].at(v0 + m / 2)).write(l[ax]);
                    (*self.v[3 + ax].at(v0 + m / 2)).write(h[ax]);
                }
            }
            last = (l, h);
        }
    }
}

static GRID_POOL: std::sync::Mutex<Option<GridBufs>> = std::sync::Mutex::new(None);

fn take_grid_bufs() -> GridBufs {
    GRID_POOL
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
        .unwrap_or_default()
}

fn give_grid_bufs(bufs: GridBufs) {
    const MAX_KEEP: usize = 1 << 20;
    if bufs.coords.x.capacity() > MAX_KEEP || bufs.offsets.capacity() > MAX_KEEP {
        return;
    }
    if let Ok(mut slot) = GRID_POOL.lock() {
        if slot.is_none() {
            *slot = Some(bufs);
        }
    }
}

/// Active atoms in bin order. The occupants of bin `c` are the slots
/// `offsets[c]..offsets[c + 1]`, ordered inside the bin by [`Keys`] and
/// then by atom index, so eight slots in a row are a compact block
/// whatever the input order.
struct Grid {
    n: [i32; 3],
    widths: [f64; 3],
    offsets: Vec<usize>,
    /// Cartesian corner of every bin, `H (ix/nx, iy/ny, iz/nz) + origin`.
    corners: Vec<[f64; 3]>,
    coords: Coords,
    /// Largest |component| of the folded and of the relative positions.
    max_abs: f64,
    max_rel: f64,
    /// The fold, the counts, and the slot order, kept for the next grid.
    fold: FoldCols,
    tally: Vec<u32>,
    order: Vec<u32>,
    boxes: Boxes,
}

impl Drop for Grid {
    fn drop(&mut self) {
        give_grid_bufs(GridBufs {
            offsets: std::mem::take(&mut self.offsets),
            corners: std::mem::take(&mut self.corners),
            coords: std::mem::take(&mut self.coords),
            fold: std::mem::take(&mut self.fold),
            tally: std::mem::take(&mut self.tally),
            order: std::mem::take(&mut self.order),
            boxes: std::mem::take(&mut self.boxes),
        });
    }
}

/// One atom in a thread's staging area: folded position, sub-cell inside
/// its bin, and atom index.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy, Default)]
struct Staged {
    p: [f64; 3],
    sub: u32,
    atom: u32,
}

/// Every active atom's folded position and key, between the passes.
#[derive(Default)]
struct FoldCols {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    key: Vec<u32>,
}

impl FoldCols {
    fn clear_for(&mut self, n: usize) {
        for v in [&mut self.x, &mut self.y, &mut self.z] {
            v.clear();
            v.reserve(n);
        }
        self.key.clear();
        self.key.reserve(n);
    }

    fn ptrs(&mut self) -> FoldPtrs {
        let raw = |v: &mut Vec<f64>| RowPtr(v.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>);
        FoldPtrs {
            x: raw(&mut self.x),
            y: raw(&mut self.y),
            z: raw(&mut self.z),
            key: RowPtr(self.key.as_mut_ptr() as *mut std::mem::MaybeUninit<u32>),
        }
    }
}

/// Raw [`FoldCols`] for writes at disjoint atoms from several threads.
#[derive(Clone, Copy)]
struct FoldPtrs {
    x: RowPtr<f64>,
    y: RowPtr<f64>,
    z: RowPtr<f64>,
    key: RowPtr<u32>,
}

impl FoldPtrs {
    /// # Safety
    /// `k` is in range and no other thread touches it.
    #[inline(always)]
    unsafe fn put(self, k: usize, p: [f64; 3], key: usize) {
        (*self.x.at(k)).write(p[0]);
        (*self.y.at(k)).write(p[1]);
        (*self.z.at(k)).write(p[2]);
        (*self.key.at(k)).write(key as u32);
    }

    /// # Safety
    /// Atom `k` is written.
    #[inline(always)]
    unsafe fn get(self, k: usize) -> ([f64; 3], usize) {
        (
            [
                (*self.x.at(k)).assume_init(),
                (*self.y.at(k)).assume_init(),
                (*self.z.at(k)).assume_init(),
            ],
            (*self.key.at(k)).assume_init() as usize,
        )
    }
}

/// Sort keys of the slots: the bin, then a Morton code of the atom's
/// sub-cell, `sub` per axis (a power of two), so a run of slots inside a
/// bin covers a compact part of it.
#[derive(Clone, Copy)]
struct Keys {
    n: [i32; 3],
    sub: i32,
}

impl Keys {
    /// Sub-cells per axis for `n_act` atoms in `ncell` bins: about one
    /// atom per sub-cell, so eight slots in a row are a compact block, and
    /// never more keys than atoms.
    fn sub_for(n_act: usize, ncell: usize) -> i32 {
        let per_bin = n_act / ncell.max(1);
        if per_bin >= 48 {
            4
        } else if per_bin >= 6 {
            2
        } else {
            1
        }
    }

    /// Sub-cells per axis over the whole box.
    fn fine(self) -> [i32; 3] {
        [
            self.n[0] * self.sub,
            self.n[1] * self.sub,
            self.n[2] * self.sub,
        ]
    }

    /// Keys per bin.
    fn per_bin(self) -> usize {
        (self.sub * self.sub * self.sub) as usize
    }

    /// Key of the sub-cell `b`: the bin's flat index times
    /// [`Keys::per_bin`], plus the Morton code inside the bin.
    #[inline(always)]
    fn key(self, b: [i32; 3]) -> usize {
        let shift = self.sub.trailing_zeros();
        let low = self.sub - 1;
        let coarse = bins::flat_cell([b[0] >> shift, b[1] >> shift, b[2] >> shift], self.n);
        // Interleave the low bits of the three axes, x lowest.
        let spread = |v: i32| {
            let v = (v & low) as usize;
            (v & 1) | ((v & 2) << 2) | ((v & 4) << 4)
        };
        let m = spread(b[0]) | (spread(b[1]) << 1) | (spread(b[2]) << 2);
        coarse * self.per_bin() + m
    }
}

/// Fold atoms `lo..hi` into `fold` and count each key (or each bin,
/// with `by_bin`), exactly as [`bins::fold_point`]: unmasked atoms go
/// eight at a time on AVX-512, with the same operations in the same order.
#[allow(clippy::too_many_arguments)]
fn fold_range(
    simbox: &Cell,
    xyz: &[[f64; 3]],
    active: Option<&[usize]>,
    keys: Keys,
    by_bin: bool,
    lo: usize,
    hi: usize,
    fold: FoldPtrs,
    count: &mut [u32],
) {
    let shift = if by_bin {
        keys.per_bin().trailing_zeros()
    } else {
        0
    };
    #[allow(unused_mut)]
    let mut slot = lo;
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    if active.is_none() && std::is_x86_feature_detected!("avx512f") {
        // Safety: AVX-512F was detected, `lo..hi` indexes `xyz`, and this
        // caller owns `fold` over that range.
        slot = unsafe { fold_avx512(simbox, xyz, keys, shift, lo, hi, fold, count) };
    }
    let fine = keys.fine();
    for k in slot..hi {
        let i = active.map_or(k, |a| a[k]);
        let (q, b) = bins::fold_point(simbox, xyz[i], fine);
        let key = keys.key(b);
        count[key >> shift] += 1;
        // Safety: the caller owns `fold` over `lo..hi`.
        unsafe { fold.put(k, q, key) };
    }
}

/// [`fold_range`] for eight unmasked atoms at a time, with [`Keys::key`]
/// in registers; returns the first slot left for the scalar fold.
///
/// # Safety
/// AVX-512F is available, `lo..hi` indexes `xyz`, and the caller owns
/// `fold` over that range.
#[allow(clippy::incompatible_msrv, clippy::too_many_arguments)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f")]
unsafe fn fold_avx512(
    simbox: &Cell,
    xyz: &[[f64; 3]],
    keys: Keys,
    shift: u32,
    lo: usize,
    hi: usize,
    fold: FoldPtrs,
    count: &mut [u32],
) -> usize {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_mullo_epi32, _mm256_or_si256,
        _mm256_set1_epi32, _mm256_sll_epi32, _mm256_slli_epi32, _mm256_srl_epi32,
        _mm256_storeu_si256, _mm512_storeu_pd, _mm_cvtsi32_si128,
    };
    let f = bins::Fold8::new(simbox, keys.fine());
    let flat = xyz.as_ptr() as *const f64;
    let bits = keys.sub.trailing_zeros() as i32;
    let down = _mm_cvtsi32_si128(bits);
    let up = _mm_cvtsi32_si128(3 * bits);
    let (nx, ny) = (_mm256_set1_epi32(keys.n[0]), _mm256_set1_epi32(keys.n[1]));
    let low = _mm256_set1_epi32(keys.sub - 1);
    let (one, two, four) = (
        _mm256_set1_epi32(1),
        _mm256_set1_epi32(2),
        _mm256_set1_epi32(4),
    );
    let spread = |v: __m256i| {
        let v = _mm256_and_si256(v, low);
        _mm256_or_si256(
            _mm256_and_si256(v, one),
            _mm256_or_si256(
                _mm256_slli_epi32::<2>(_mm256_and_si256(v, two)),
                _mm256_slli_epi32::<4>(_mm256_and_si256(v, four)),
            ),
        )
    };
    let mut lane = [0u32; 8];
    let mut k = lo;
    while k + 8 <= hi {
        let (_, p, b) = bins::fold8_regs(&f, flat.add(3 * k));
        let coarse = _mm256_add_epi32(
            _mm256_mullo_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_srl_epi32(b[2], down), ny),
                    _mm256_srl_epi32(b[1], down),
                ),
                nx,
            ),
            _mm256_srl_epi32(b[0], down),
        );
        let m = _mm256_or_si256(
            spread(b[0]),
            _mm256_or_si256(
                _mm256_slli_epi32::<1>(spread(b[1])),
                _mm256_slli_epi32::<2>(spread(b[2])),
            ),
        );
        let key = _mm256_or_si256(_mm256_sll_epi32(coarse, up), m);
        _mm512_storeu_pd(fold.x.ptr().add(k), p[0]);
        _mm512_storeu_pd(fold.y.ptr().add(k), p[1]);
        _mm512_storeu_pd(fold.z.ptr().add(k), p[2]);
        _mm256_storeu_si256(fold.key.ptr().add(k) as *mut _, key);
        _mm256_storeu_si256(lane.as_mut_ptr() as *mut _, key);
        for &c in &lane {
            count[(c >> shift) as usize] += 1;
        }
        k += 8;
    }
    k
}

/// Raw slot columns for writes at disjoint slots from several threads.
#[derive(Clone, Copy)]
struct SlotPtrs {
    x: RowPtr<f64>,
    y: RowPtr<f64>,
    z: RowPtr<f64>,
    rx: RowPtr<f64>,
    ry: RowPtr<f64>,
    rz: RowPtr<f64>,
    r2: RowPtr<f64>,
    id: RowPtr<u32>,
}

impl SlotPtrs {
    fn of(c: &mut Coords) -> Self {
        let raw = |v: &mut Vec<f64>| RowPtr(v.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>);
        SlotPtrs {
            x: raw(&mut c.x),
            y: raw(&mut c.y),
            z: raw(&mut c.z),
            rx: raw(&mut c.rx),
            ry: raw(&mut c.ry),
            rz: raw(&mut c.rz),
            r2: raw(&mut c.r2),
            id: RowPtr(c.id.as_mut_ptr() as *mut std::mem::MaybeUninit<u32>),
        }
    }

    /// Write atom `i` at `slot`; returns the largest |component| of the
    /// folded and of the relative position.
    ///
    /// # Safety
    /// `slot` is in range and no other thread writes it.
    #[inline(always)]
    unsafe fn put(self, slot: usize, i: usize, p: [f64; 3], o: [f64; 3]) -> (f64, f64) {
        let (rx, ry, rz) = (p[0] - o[0], p[1] - o[1], p[2] - o[2]);
        (*self.x.at(slot)).write(p[0]);
        (*self.y.at(slot)).write(p[1]);
        (*self.z.at(slot)).write(p[2]);
        (*self.rx.at(slot)).write(rx);
        (*self.ry.at(slot)).write(ry);
        (*self.rz.at(slot)).write(rz);
        (*self.r2.at(slot)).write(rx * rx + ry * ry + rz * rz);
        (*self.id.at(slot)).write(i as u32);
        (
            p[0].abs().max(p[1].abs()).max(p[2].abs()),
            rx.abs().max(ry.abs()).max(rz.abs()),
        )
    }
}

impl Grid {
    /// Fold every active atom, count each key, scan, then scatter into
    /// the slot columns. With several threads each thread folds one block
    /// of atoms (a block distribution) and groups it by bin in its own
    /// staging area; then each thread owns a run of bins, reads them from
    /// every staging area in thread order, sorts them by sub-cell, and
    /// writes its run of slots alone. The order is one thread's.
    fn build(
        xyz: &[[f64; 3]],
        simbox: &Cell,
        active: Option<&[usize]>,
        n: [i32; 3],
        threads: usize,
    ) -> Grid {
        let ncell = (n[0] as usize) * (n[1] as usize) * (n[2] as usize);
        let n_act = active.map_or(xyz.len(), |a| a.len());
        let atom = |k: usize| active.map_or(k, |a| a[k]);
        // Buffers come from the last grid and are not zeroed: the fold
        // writes every atom, the corner pass every bin, and the scatter
        // every slot, before any of them is read.
        let GridBufs {
            mut offsets,
            mut corners,
            mut coords,
            mut fold,
            mut tally,
            mut order,
            mut boxes,
        } = take_grid_bufs();
        offsets.clear();
        offsets.reserve(ncell + 1);
        corners.clear();
        corners.reserve(ncell);
        coords.clear_for(n_act);
        // The fold is read only through `fp`; its vectors stay empty.
        fold.clear_for(n_act);
        let fp = fold.ptrs();
        let cp = RowPtr(corners.as_mut_ptr() as *mut std::mem::MaybeUninit<[f64; 3]>);
        let slots = SlotPtrs::of(&mut coords);
        // Corners of bins `lo..hi`, stepping the bin indices rather than
        // dividing the flat index.
        let corners_of = |lo: usize, hi: usize| {
            let (nx, ny) = (n[0] as usize, n[1] as usize);
            let (mut ix, mut iy, mut iz) = (lo % nx, (lo / nx) % ny, lo / (nx * ny));
            for c in lo..hi {
                let corner = simbox.cartesian([
                    f64::from(ix as i32) / f64::from(n[0]),
                    f64::from(iy as i32) / f64::from(n[1]),
                    f64::from(iz as i32) / f64::from(n[2]),
                ]);
                // Safety: the caller owns `corners` over `lo..hi`.
                unsafe { (*cp.at(c)).write(corner) };
                ix += 1;
                if ix == nx {
                    ix = 0;
                    iy += 1;
                    if iy == ny {
                        iy = 0;
                        iz += 1;
                    }
                }
            }
        };
        let (mut max_abs, mut max_rel) = (0.0f64, 0.0f64);
        let keys = Keys {
            n,
            sub: Keys::sub_for(n_act, ncell),
        };
        let per = keys.per_bin();
        let nkey = ncell * per;
        let cull = cfg!(all(target_arch = "x86_64", linkcell_avx512))
            && n_act >= 8 * CULL_RUNS * ncell
            && !no_cull();
        #[cfg(feature = "parallel")]
        let parallel = threads > 1;
        #[cfg(not(feature = "parallel"))]
        let parallel = {
            let _ = threads;
            false
        };
        if parallel {
            #[cfg(feature = "parallel")]
            {
                let p = rayon::current_num_threads().max(1);
                let block = |k: usize, len: usize| (k * len / p, (k + 1) * len / p);
                let shift = per.trailing_zeros();
                // Pass 1: each thread folds a block of atoms, groups it by bin
                // in its own staging area (atom order inside a bin), and folds
                // a block of bin corners.
                let staged: Vec<(Vec<Staged>, Vec<usize>)> = rayon::broadcast(|ctx| {
                    let _timer = crate::pop::JobTimer::new();
                    let k = ctx.index();
                    let (lo, hi) = block(k, n_act);
                    let mut count = vec![0u32; ncell];
                    // Atom blocks are disjoint, so each thread owns its range.
                    fold_range(simbox, xyz, active, keys, true, lo, hi, fp, &mut count);
                    let mut first = Vec::with_capacity(ncell + 1);
                    let mut at = 0usize;
                    for &c in &count {
                        first.push(at);
                        at += c as usize;
                    }
                    first.push(at);
                    let mut next = first.clone();
                    let mut stage = vec![Staged::default(); hi - lo];
                    for slot in lo..hi {
                        // Safety: the fold above wrote this thread's atoms.
                        let (p, key) = unsafe { fp.get(slot) };
                        let c = key >> shift;
                        stage[next[c]] = Staged {
                            p,
                            sub: (key & (per - 1)) as u32,
                            atom: atom(slot) as u32,
                        };
                        next[c] += 1;
                    }
                    let (clo, chi) = block(k, ncell);
                    // Corner blocks are disjoint.
                    corners_of(clo, chi);
                    (stage, first)
                });
                // Safety: pass 1 wrote every corner; the grid keeps no fold.
                unsafe { corners.set_len(ncell) };
                // Scan: bin `c` starts at `offsets[c]`.
                let mut at = 0usize;
                for c in 0..ncell {
                    offsets.push(at);
                    for (_, first) in &staged {
                        at += first[c + 1] - first[c];
                    }
                }
                offsets.push(at);
                boxes.starts(&offsets, cull);
                let bp = boxes.ptrs();
                let (vstart, gstart) = (&boxes.vstart, &boxes.gstart);
                // Pass 2: each thread owns a run of bins, balanced by atoms,
                // reads their atoms from every staging area in thread order
                // (so in atom order), sorts each bin by sub-cell with a stable
                // count, and writes that run of slots, and their boxes,
                // alone: the order is one thread's, and no cache line is
                // written by two threads.
                let ranges = cell_ranges(&offsets, p);
                let (staged, corners, offsets, ranges) = (&staged, &corners, &offsets, &ranges);
                let maxima: Vec<(f64, f64)> = rayon::broadcast(|ctx| {
                    let _timer = crate::pop::JobTimer::new();
                    let (mut ma, mut mr) = (0.0f64, 0.0f64);
                    let Some(&(clo, chi)) = ranges.get(ctx.index()) else {
                        return (ma, mr);
                    };
                    let mut bucket = vec![0usize; per + 1];
                    for c in clo..chi {
                        bucket.iter_mut().for_each(|b| *b = 0);
                        for (stage, first) in staged {
                            for e in &stage[first[c]..first[c + 1]] {
                                bucket[e.sub as usize + 1] += 1;
                            }
                        }
                        let mut acc = offsets[c];
                        for b in bucket.iter_mut() {
                            acc += *b;
                            *b = acc;
                        }
                        for (stage, first) in staged {
                            for e in &stage[first[c]..first[c + 1]] {
                                let dest = bucket[e.sub as usize];
                                bucket[e.sub as usize] += 1;
                                // Safety: bin `c`, and so its slots, belongs to
                                // this thread alone.
                                let (a, r) =
                                    unsafe { slots.put(dest, e.atom as usize, e.p, corners[c]) };
                                ma = ma.max(a);
                                mr = mr.max(r);
                            }
                        }
                        if cull {
                            // Safety: this thread wrote bin `c` and owns its runs.
                            unsafe {
                                bp.fill(slots, offsets[c], offsets[c + 1], vstart[c], gstart[c])
                            };
                        }
                    }
                    (ma, mr)
                });
                for (a, r) in maxima {
                    max_abs = max_abs.max(a);
                    max_rel = max_rel.max(r);
                }
            }
        } else {
            let _timer = crate::pop::JobTimer::new();
            tally.clear();
            tally.resize(nkey, 0);
            fold_range(simbox, xyz, active, keys, false, 0, n_act, fp, &mut tally);
            corners_of(0, ncell);
            // Safety: `corners_of` wrote every corner.
            unsafe { corners.set_len(ncell) };
            // `tally` becomes the next free slot of each key.
            let mut at = 0usize;
            for bin in tally.chunks_exact_mut(per) {
                offsets.push(at);
                for t in bin {
                    let count = *t as usize;
                    *t = at as u32;
                    at += count;
                }
            }
            offsets.push(at);
            // Sort atom numbers by key, then fill the slot columns in slot
            // order: the random accesses are reads of the fold, and every
            // column is written front to back.
            order.clear();
            order.reserve(n_act);
            let op = order.as_mut_ptr();
            for k in 0..n_act {
                // Safety: the fold wrote atom `k`; `tally` keeps each
                // destination inside `0..n_act`.
                unsafe {
                    let key = (*fp.key.at(k)).assume_init() as usize;
                    let dest = tally[key] as usize;
                    tally[key] += 1;
                    op.add(dest).write(k as u32);
                }
            }
            boxes.starts(&offsets, cull);
            let bp = boxes.ptrs();
            for c in 0..ncell {
                let corner = corners[c];
                for dest in offsets[c]..offsets[c + 1] {
                    // Safety: every slot below `n_act` holds an atom number,
                    // the fold wrote that atom, and one thread writes.
                    let (a, r) = unsafe {
                        let k = *op.add(dest) as usize;
                        let (p, _) = fp.get(k);
                        slots.put(dest, atom(k), p, corner)
                    };
                    max_abs = max_abs.max(a);
                    max_rel = max_rel.max(r);
                }
                if cull {
                    // Safety: one thread, and bin `c` is written.
                    unsafe {
                        bp.fill(
                            slots,
                            offsets[c],
                            offsets[c + 1],
                            boxes.vstart[c],
                            boxes.gstart[c],
                        )
                    };
                }
            }
        }
        // Safety: the scatter wrote every slot of every column once.
        unsafe { coords.set_len(n_act) };
        Grid {
            n,
            widths: simbox.widths(),
            offsets,
            corners,
            coords,
            max_abs,
            max_rel,
            fold,
            tally,
            order,
            boxes,
        }
    }

    fn atoms(&self) -> usize {
        self.coords.id.len()
    }

    /// No bins and no atoms: every search on it is empty.
    fn empty() -> Grid {
        Grid {
            n: [1, 1, 1],
            widths: [0.0; 3],
            offsets: vec![0],
            corners: Vec::new(),
            coords: Coords::default(),
            max_abs: 0.0,
            max_rel: 0.0,
            fold: FoldCols::default(),
            tally: Vec::new(),
            order: Vec::new(),
            boxes: Boxes {
                vstart: vec![0],
                gstart: vec![0],
                ..Boxes::default()
            },
        }
    }
}

/// Bound on the difference between the expanded squared distance
/// `|r_q|^2 - 2 r_q . r_p' + |r_p'|^2` and the direct one
/// `|q - (p - S)|^2`, both in double precision.
///
/// Each value is a short sum of terms no larger than the square of the
/// largest magnitude in it: absolute coordinates `A`, shifts `S`,
/// relative coordinates `R`, and block offsets `D`, per component. Every
/// rounding is at most `2^-53` of such a term, and there are fewer than
/// 32 of them per value; `192 = 64 * 3` covers both values and the
/// three components. A lane within this of `cutoff^2` is decided by the
/// direct formula, so the expanded form never changes a row.
fn expanded_margin(grid: &Grid, partners: &PartnerList, cutoff: f64) -> f64 {
    let (a, r) = (grid.max_abs, grid.max_rel);
    let (s, d) = (partners.s_max, partners.d_max);
    let span = 2.0 * a + s + 2.0 * r + d + cutoff;
    192.0 * f64::EPSILON * 0.5 * span * span
}

fn simd_mode() -> u8 {
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("popcnt")
            && std::is_x86_feature_detected!("fma")
        {
            return 2;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            return 1;
        }
    }
    0
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
    grid: &'a Grid,
    cut2: f64,
    /// [`expanded_margin`] of this search.
    #[cfg_attr(not(all(target_arch = "x86_64", linkcell_avx512)), allow(dead_code))]
    margin: f64,
    partners: &'a PartnerList,
    /// 2 = AVX-512, 1 = AVX, 0 = scalar.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    simd: u8,
}

#[derive(Clone, Copy)]
struct Block {
    i_lo: usize,
    i_hi: usize,
    j_lo: usize,
    j_hi: usize,
    shift_s: [i32; 3],
    shift: [f64; 3],
    /// [`Partner::delta`]; zero for the home cell.
    #[cfg_attr(not(all(target_arch = "x86_64", linkcell_avx512)), allow(dead_code))]
    delta: [f64; 3],
    /// Home cell: source `s` only sees occupants `s + 1 ..`.
    tri: bool,
    /// [`Boxes`] of the target bin's first run of eight and the source
    /// bin's first run of four.
    #[cfg_attr(not(all(target_arch = "x86_64", linkcell_avx512)), allow(dead_code))]
    jv: usize,
    #[cfg_attr(not(all(target_arch = "x86_64", linkcell_avx512)), allow(dead_code))]
    ig: usize,
}

/// Expected pairs where the walk splits across threads. On this 8-core
/// host one thread is faster at 512 atoms in an 18 Å cube at a 4 Å
/// cutoff (6 thousand pairs) and slower from 768 (14 thousand).
#[cfg(feature = "parallel")]
const PARALLEL_PAIRS: usize = 10_000;

/// Active atoms from which a split walk also builds its bins on several
/// threads. The fold then wakes the workers the search uses next; at 1024
/// atoms in an 18 Å cube on this 8-core host the two builds tie, and at
/// 4096 the threaded one is faster.
const PARALLEL_GRID: usize = 1_024;

/// The walk's thresholds and paths as values an autotuner can set (the
/// `tune` feature, through `lc_tune_set`); otherwise each is its
/// constant. Every one is read once per call, outside the tiles.
pub(crate) mod knobs {
    // `lc_tune_set` keys.
    #[cfg(feature = "tune")]
    pub(crate) const SPLIT_PAIRS: usize = 0;
    #[cfg(feature = "tune")]
    pub(crate) const GRID_ATOMS: usize = 1;
    #[cfg(feature = "tune")]
    pub(crate) const FUSED: usize = 2;
    #[cfg(feature = "tune")]
    pub(crate) const CHUNKS: usize = 3;

    #[cfg(feature = "tune")]
    pub(crate) static VALUES: [std::sync::atomic::AtomicUsize; 4] = [
        std::sync::atomic::AtomicUsize::new(super::DEFAULT_SPLIT_PAIRS),
        std::sync::atomic::AtomicUsize::new(super::PARALLEL_GRID),
        std::sync::atomic::AtomicUsize::new(1),
        std::sync::atomic::AtomicUsize::new(1),
    ];

    #[cfg(feature = "tune")]
    fn get(key: usize) -> usize {
        VALUES[key].load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Expected pairs where the walk splits across threads.
    #[cfg(feature = "parallel")]
    pub(crate) fn split_pairs() -> usize {
        #[cfg(feature = "tune")]
        return get(SPLIT_PAIRS);
        #[cfg(not(feature = "tune"))]
        super::PARALLEL_PAIRS
    }

    /// Active atoms from which a split walk builds its bins on several threads.
    pub(crate) fn grid_atoms() -> usize {
        #[cfg(feature = "tune")]
        return get(GRID_ATOMS);
        #[cfg(not(feature = "tune"))]
        super::PARALLEL_GRID
    }

    /// One thread writes a full list straight from the tile.
    #[cfg_attr(not(all(target_arch = "x86_64", linkcell_avx512)), allow(dead_code))]
    pub(crate) fn fused() -> bool {
        #[cfg(feature = "tune")]
        return get(FUSED) != 0;
        #[cfg(not(feature = "tune"))]
        true
    }

    /// Bin ranges per thread in a split search.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) fn chunks() -> usize {
        #[cfg(feature = "tune")]
        return get(CHUNKS).max(1);
        #[cfg(not(feature = "tune"))]
        1
    }
}

/// The split knob's default: [`PARALLEL_PAIRS`], or never without `parallel`.
#[cfg(feature = "tune")]
#[cfg(feature = "parallel")]
const DEFAULT_SPLIT_PAIRS: usize = PARALLEL_PAIRS;
#[cfg(feature = "tune")]
#[cfg(not(feature = "parallel"))]
const DEFAULT_SPLIT_PAIRS: usize = usize::MAX;

/// Hits for the whole chunk. The distance loop appends here, then one
/// pass writes the rows. Runs share a shift so the inner loop does not.
struct Scratch {
    atom: Vec<u32>,
    js: Vec<u32>,
    d2: Vec<f64>,
    run_shift: Vec<[i32; 3]>,
    run_end: Vec<usize>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            atom: Vec::new(),
            js: Vec::new(),
            d2: Vec::new(),
            run_shift: Vec::new(),
            run_end: Vec::new(),
        }
    }

    fn reserve_more(&mut self, extra: usize) {
        let need = self.js.len().saturating_add(extra);
        if self.js.capacity() < need {
            self.atom.reserve(extra);
            self.js.reserve(extra);
            self.d2.reserve(extra);
        }
    }

    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    fn finish(&mut self, n: usize) {
        debug_assert!(n <= self.js.capacity());
        unsafe {
            self.atom.set_len(n);
            self.js.set_len(n);
            self.d2.set_len(n);
        }
    }

    fn push(&mut self, atom: u32, j: u32, d2: f64) {
        self.atom.push(atom);
        self.js.push(j);
        self.d2.push(d2);
    }

    fn note(&mut self, before: usize, shift: [i32; 3]) {
        let after = self.js.len();
        if after > before {
            self.run_shift.push(shift);
            self.run_end.push(after);
        }
    }
}

/// Every hit of one search, by cell range. Nothing is a row yet: each
/// caller expands these once into its own layout.
pub(crate) struct Found {
    chunks: Vec<Scratch>,
    half: bool,
    /// [`simd_mode`] of the search; the writers use the same level.
    #[cfg_attr(not(all(target_arch = "x86_64", linkcell_avx512)), allow(dead_code))]
    simd: u8,
}

impl Found {
    fn rows_of(&self, chunk: &Scratch) -> usize {
        if self.half {
            chunk.js.len()
        } else {
            chunk.js.len() * 2
        }
    }

    /// Output rows over every chunk.
    pub(crate) fn rows(&self) -> usize {
        self.chunks.iter().map(|c| self.rows_of(c)).sum()
    }

    fn offsets(&self) -> Vec<usize> {
        let mut off = Vec::with_capacity(self.chunks.len() + 1);
        off.push(0usize);
        for chunk in &self.chunks {
            off.push(off.last().copied().unwrap_or(0) + self.rows_of(chunk));
        }
        off
    }

    fn into_pairs(self) -> Vec<Pair> {
        let total = self.rows();
        let mut found: Vec<Pair> = Vec::with_capacity(total);
        // Safety: `total` slots are allocated.
        unsafe {
            self.write_pairs(found.as_mut_ptr());
            found.set_len(total);
        }
        found
    }

    /// Every row, written at `out`.
    ///
    /// # Safety
    /// `out` has room for [`Found::rows`] rows.
    unsafe fn write_pairs(&self, out: *mut Pair) {
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        let words = if self.simd == 2 {
            RowWords::of_pair()
        } else {
            None
        };
        let base = RowPtr(out as *mut std::mem::MaybeUninit<Pair>);
        // Chunk `t` writes the rows `off[t]..off[t + 1]`, which partition
        // `0..rows`, and nothing reads them before every chunk is done.
        self.each_chunk(|t, off, chunk| {
            let rows = self.rows_of(chunk);
            if rows == 0 {
                return;
            }
            let dst = unsafe { std::slice::from_raw_parts_mut(base.at(off[t]), rows) };
            if self.half {
                write_half(
                    dst,
                    &chunk.atom,
                    &chunk.js,
                    &chunk.d2,
                    &chunk.run_shift,
                    &chunk.run_end,
                );
                return;
            }
            #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
            if let Some(words) = words {
                // Safety: AVX-512 was detected for this search, and `dst`
                // holds two rows per hit of `chunk`.
                unsafe { write_full_avx512(dst.as_mut_ptr(), chunk, &words) };
                return;
            }
            write_full(
                dst,
                &chunk.atom,
                &chunk.js,
                &chunk.d2,
                &chunk.run_shift,
                &chunk.run_end,
            );
        });
    }

    /// Clear `out`, then write every row into its four columns.
    pub(crate) fn fill_columns(&self, out: &mut PairColumns) {
        let total = self.rows();
        out.i.clear();
        out.j.clear();
        out.shift.clear();
        out.dist2.clear();
        out.i.reserve(total);
        out.j.reserve(total);
        out.shift.reserve(total);
        out.dist2.reserve(total);
        // Safety: each column has room for `total` rows.
        unsafe {
            self.write_columns(
                out.i.as_mut_ptr(),
                out.j.as_mut_ptr(),
                out.shift.as_mut_ptr() as *mut i32,
                out.dist2.as_mut_ptr(),
            );
            out.i.set_len(total);
            out.j.set_len(total);
            out.shift.set_len(total);
            out.dist2.set_len(total);
        }
    }

    /// # Safety
    /// `i`, `j`, and `d2` are writable for [`Found::rows`] values and
    /// `shift` for three times that.
    pub(crate) unsafe fn write_columns(
        &self,
        i: *mut i32,
        j: *mut i32,
        shift: *mut i32,
        d2: *mut f64,
    ) {
        let cols = ColumnPtrs {
            i: RowPtr(i as *mut std::mem::MaybeUninit<i32>),
            j: RowPtr(j as *mut std::mem::MaybeUninit<i32>),
            shift: RowPtr(shift as *mut std::mem::MaybeUninit<i32>),
            d2: RowPtr(d2 as *mut std::mem::MaybeUninit<f64>),
        };
        self.each_chunk(|t, off, chunk| {
            let rows = self.rows_of(chunk);
            if rows == 0 {
                return;
            }
            let at = off[t];
            let (ci, cj, cs, cd) = unsafe {
                (
                    std::slice::from_raw_parts_mut(cols.i.at(at), rows),
                    std::slice::from_raw_parts_mut(cols.j.at(at), rows),
                    std::slice::from_raw_parts_mut(cols.shift.at(3 * at), 3 * rows),
                    std::slice::from_raw_parts_mut(cols.d2.at(at), rows),
                )
            };
            #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
            if self.simd == 2 && !self.half {
                // Safety: AVX-512 was detected, and each column holds two
                // rows per hit of `chunk` (three values per row of shift).
                unsafe { write_columns_avx512(ci, cj, cs, cd, chunk) };
                return;
            }
            write_columns(ci, cj, cs, cd, chunk, self.half);
        });
    }

    /// `out[t] = row(i, j, shift, dist2)` for every row `t`.
    ///
    /// # Safety
    /// `out` is writable for [`Found::rows`] values of `R`.
    #[cfg_attr(not(feature = "capi"), allow(dead_code))]
    pub(crate) unsafe fn write_rows<R, F>(&self, out: *mut R, row: F)
    where
        R: Send,
        F: Fn(i32, i32, [i32; 3], f64) -> R + Sync,
    {
        let base = RowPtr(out as *mut std::mem::MaybeUninit<R>);
        self.each_chunk(|t, off, chunk| {
            let rows = self.rows_of(chunk);
            if rows == 0 {
                return;
            }
            let dst = unsafe { std::slice::from_raw_parts_mut(base.at(off[t]), rows) };
            let mut at = 0usize;
            let mut lo = 0usize;
            for (r, &hi) in chunk.run_end.iter().enumerate() {
                let shift = chunk.run_shift[r];
                let neg = [-shift[0], -shift[1], -shift[2]];
                for k in lo..hi {
                    let a = chunk.atom[k] as i32;
                    let b = chunk.js[k] as i32;
                    let d = chunk.d2[k];
                    if self.half {
                        if keep_half(a as usize, b as usize, shift) {
                            dst[at].write(row(a, b, shift, d));
                        } else {
                            dst[at].write(row(b, a, neg, d));
                        }
                        at += 1;
                    } else {
                        dst[at].write(row(a, b, shift, d));
                        dst[at + 1].write(row(b, a, neg, d));
                        at += 2;
                    }
                }
                lo = hi;
            }
        });
    }

    /// `job(t, offsets, chunk)` for every chunk, on several threads when
    /// there is more than one chunk.
    fn each_chunk<F>(&self, job: F)
    where
        F: Fn(usize, &[usize], &Scratch) + Sync,
    {
        let off = self.offsets();
        #[cfg(feature = "parallel")]
        if self.chunks.len() > 1 {
            // Chunk `t` was searched on thread `t`; writing it there reads
            // that core's cache.
            if self.chunks.len() <= rayon::current_num_threads() {
                rayon::broadcast(|ctx| {
                    if let Some(chunk) = self.chunks.get(ctx.index()) {
                        let _timer = crate::pop::JobTimer::new();
                        job(ctx.index(), &off, chunk);
                    }
                });
            } else {
                use rayon::prelude::*;
                self.chunks.par_iter().enumerate().for_each(|(t, chunk)| {
                    let _timer = crate::pop::JobTimer::new();
                    job(t, &off, chunk);
                });
            }
            return;
        }
        for (t, chunk) in self.chunks.iter().enumerate() {
            let _timer = crate::pop::JobTimer::new();
            job(t, &off, chunk);
        }
    }
}

/// A destination several jobs write at disjoint offsets.
///
/// Safety: jobs write distinct slots and do not read a slot another job writes.
struct RowPtr<T>(*mut std::mem::MaybeUninit<T>);
impl<T> Clone for RowPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for RowPtr<T> {}
unsafe impl<T: Send> Send for RowPtr<T> {}
unsafe impl<T: Send> Sync for RowPtr<T> {}
impl<T> RowPtr<T> {
    unsafe fn at(self, index: usize) -> *mut std::mem::MaybeUninit<T> {
        self.0.add(index)
    }

    /// The pointer itself; a closure that calls this captures the whole
    /// `RowPtr`, which is `Send`, not the raw field.
    fn ptr(self) -> *mut T {
        self.0 as *mut T
    }
}

#[derive(Clone, Copy)]
struct ColumnPtrs {
    i: RowPtr<i32>,
    j: RowPtr<i32>,
    shift: RowPtr<i32>,
    d2: RowPtr<f64>,
}

#[inline(never)]
fn write_columns(
    ci: &mut [std::mem::MaybeUninit<i32>],
    cj: &mut [std::mem::MaybeUninit<i32>],
    cs: &mut [std::mem::MaybeUninit<i32>],
    cd: &mut [std::mem::MaybeUninit<f64>],
    chunk: &Scratch,
    half: bool,
) {
    let mut out = 0usize;
    let mut lo = 0usize;
    for (r, &hi) in chunk.run_end.iter().enumerate() {
        let shift = chunk.run_shift[r];
        let neg = [-shift[0], -shift[1], -shift[2]];
        for k in lo..hi {
            let a = chunk.atom[k] as i32;
            let b = chunk.js[k] as i32;
            let d = chunk.d2[k];
            if half {
                let (i, j, s) = if keep_half(a as usize, b as usize, shift) {
                    (a, b, shift)
                } else {
                    (b, a, neg)
                };
                ci[out].write(i);
                cj[out].write(j);
                cs[3 * out].write(s[0]);
                cs[3 * out + 1].write(s[1]);
                cs[3 * out + 2].write(s[2]);
                cd[out].write(d);
                out += 1;
            } else {
                ci[out].write(a);
                cj[out].write(b);
                cs[3 * out].write(shift[0]);
                cs[3 * out + 1].write(shift[1]);
                cs[3 * out + 2].write(shift[2]);
                cd[out].write(d);
                ci[out + 1].write(b);
                cj[out + 1].write(a);
                cs[3 * out + 3].write(neg[0]);
                cs[3 * out + 4].write(neg[1]);
                cs[3 * out + 5].write(neg[2]);
                cd[out + 1].write(d);
                out += 2;
            }
        }
        lo = hi;
    }
}

/// Full-width stores past the last hit stay inside this many extra slots.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const STORE_SLACK: usize = 16;

/// Worst-case hits kept in the scratch buffer. Larger blocks use the scalar walk.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const MAX_SCRATCH: usize = 1 << 20;

#[inline(never)]
fn write_full(
    dst: &mut [std::mem::MaybeUninit<Pair>],
    atom: &[u32],
    js: &[u32],
    d2: &[f64],
    shifts: &[[i32; 3]],
    ends: &[usize],
) {
    let dp = dst.as_mut_ptr();
    let ap = atom.as_ptr();
    let jp = js.as_ptr();
    let yp = d2.as_ptr();
    let mut out = 0usize;
    let mut lo = 0usize;
    for (r, &hi) in ends.iter().enumerate() {
        let shift = shifts[r];
        let neg = [-shift[0], -shift[1], -shift[2]];
        for k in lo..hi {
            unsafe {
                let i = (*ap.add(k)) as usize;
                let j = (*jp.add(k)) as usize;
                let dist2 = *yp.add(k);
                (*dp.add(out)).write(Pair { i, j, shift, dist2 });
                (*dp.add(out + 1)).write(Pair {
                    i: j,
                    j: i,
                    shift: neg,
                    dist2,
                });
            }
            out += 2;
        }
        lo = hi;
    }
}

#[inline(never)]
fn write_half(
    dst: &mut [std::mem::MaybeUninit<Pair>],
    atom: &[u32],
    js: &[u32],
    d2: &[f64],
    shifts: &[[i32; 3]],
    ends: &[usize],
) {
    let dp = dst.as_mut_ptr();
    let ap = atom.as_ptr();
    let jp = js.as_ptr();
    let yp = d2.as_ptr();
    let mut out = 0usize;
    let mut lo = 0usize;
    for (r, &hi) in ends.iter().enumerate() {
        let shift = shifts[r];
        let neg = [-shift[0], -shift[1], -shift[2]];
        for k in lo..hi {
            unsafe {
                let i = (*ap.add(k)) as usize;
                let j = (*jp.add(k)) as usize;
                let dist2 = *yp.add(k);
                let pair = if keep_half(i, j, shift) {
                    Pair { i, j, shift, dist2 }
                } else {
                    Pair {
                        i: j,
                        j: i,
                        shift: neg,
                        dist2,
                    }
                };
                (*dp.add(out)).write(pair);
            }
            out += 1;
        }
        lo = hi;
    }
}

/// Word slots of a 40-byte `Pair`: `i`, `j`, `dist2`, and the first of
/// the two words that hold `shift` and its padding.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[derive(Clone, Copy)]
struct RowWords {
    /// Permute indices for four hits: eight rows, five registers.
    index: [[u64; 8]; 5],
    /// Dword permute indices for up to eight hits from two registers, one
    /// table of five registers per four hits.
    index32: [[[u32; 16]; 5]; 2],
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
impl RowWords {
    /// `None` unless `Pair` is five whole words. The layout of a Rust
    /// struct is the compiler's choice, so it is read, not assumed.
    fn of_pair() -> Option<Self> {
        let p = Pair {
            i: 0,
            j: 0,
            shift: [0; 3],
            dist2: 0.0,
        };
        let base = &p as *const Pair as usize;
        let at = |a: usize| a - base;
        let off = [
            at(&p.i as *const usize as usize),
            at(&p.j as *const usize as usize),
            at(&p.dist2 as *const f64 as usize),
            at(&p.shift as *const [i32; 3] as usize),
        ];
        if std::mem::size_of::<Pair>() != 40 || off.iter().any(|o| o % 8 != 0) {
            return None;
        }
        let (wi, wj, wd, ws) = (off[0] / 8, off[1] / 8, off[2] / 8, off[3] / 8);
        let mut seen = [false; 5];
        for w in [wi, wj, wd, ws, ws + 1] {
            if w >= 5 || seen[w] {
                return None;
            }
            seen[w] = true;
        }
        // Sources: [a0 a1 a2 a3 b0 b1 b2 b3] then [d0 d1 d2 d3 S01 S2 N01 N2].
        // Row 2k is (a_k, b_k, S, d_k); row 2k + 1 is (b_k, a_k, -S, d_k).
        let mut index = [[0u64; 8]; 5];
        // Dword sources: [j0 .. j7, i, 0, s0, s1, s2, n0, n1, n2] then the
        // eight distances, two dwords each. Group `g` holds hits 4g..4g + 3.
        let mut index32 = [[[0u32; 16]; 5]; 2];
        for r in 0..8 {
            let (k, mirror) = ((r / 2) as u64, r % 2 == 1);
            let mut row = [0u64; 5];
            row[wi] = if mirror { 4 + k } else { k };
            row[wj] = if mirror { k } else { 4 + k };
            row[wd] = 8 + k;
            row[ws] = if mirror { 14 } else { 12 };
            row[ws + 1] = if mirror { 15 } else { 13 };
            for (w, &v) in row.iter().enumerate() {
                let t = 5 * r + w;
                index[t / 8][t % 8] = v;
            }
            for (g, table) in index32.iter_mut().enumerate() {
                let h = (4 * g + r / 2) as u32;
                let mut row = [[0u32; 2]; 5];
                row[wi] = if mirror { [h, 9] } else { [8, 9] };
                row[wj] = if mirror { [8, 9] } else { [h, 9] };
                row[wd] = [16 + 2 * h, 17 + 2 * h];
                row[ws] = if mirror { [13, 14] } else { [10, 11] };
                row[ws + 1] = if mirror { [15, 9] } else { [12, 9] };
                for (w, pair) in row.iter().enumerate() {
                    for (half, &v) in pair.iter().enumerate() {
                        let q = 10 * r + 2 * w + half;
                        table[q / 16][q % 16] = v;
                    }
                }
            }
        }
        Some(Self { index, index32 })
    }
}

/// `shift` as the two little-endian words of a row: `(s0, s1)` and `(s2, 0)`.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
fn shift_words(s: [i32; 3]) -> (u64, u64) {
    (
        u64::from(s[0] as u32) | (u64::from(s[1] as u32) << 32),
        u64::from(s[2] as u32),
    )
}

/// [`write_full`] four hits at a time: eight rows are five permutes of two
/// registers and five stores.
///
/// # Safety
/// AVX-512F is available, and `dst` holds two rows per hit of `chunk`.
#[allow(clippy::incompatible_msrv)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn write_full_avx512(
    dst: *mut std::mem::MaybeUninit<Pair>,
    chunk: &Scratch,
    words: &RowWords,
) {
    use std::arch::x86_64::{
        __m128i, __m512i, _mm256_cvtepu32_epi64, _mm256_set_epi64x, _mm512_castsi256_si512,
        _mm512_inserti64x4, _mm512_loadu_si512, _mm512_permutex2var_epi64, _mm512_storeu_si512,
        _mm_loadu_si128,
    };
    let idx: [__m512i; 5] = [
        _mm512_loadu_si512(words.index[0].as_ptr() as *const _),
        _mm512_loadu_si512(words.index[1].as_ptr() as *const _),
        _mm512_loadu_si512(words.index[2].as_ptr() as *const _),
        _mm512_loadu_si512(words.index[3].as_ptr() as *const _),
        _mm512_loadu_si512(words.index[4].as_ptr() as *const _),
    ];
    let ap = chunk.atom.as_ptr();
    let jp = chunk.js.as_ptr();
    let dp = chunk.d2.as_ptr();
    let mut row = dst;
    let mut lo = 0usize;
    for (r, &hi) in chunk.run_end.iter().enumerate() {
        let shift = chunk.run_shift[r];
        let neg = [-shift[0], -shift[1], -shift[2]];
        let (s01, s2) = shift_words(shift);
        let (n01, n2) = shift_words(neg);
        let sh = _mm256_set_epi64x(n2 as i64, n01 as i64, s2 as i64, s01 as i64);
        let mut k = lo;
        while k + 4 <= hi {
            let a = _mm256_cvtepu32_epi64(_mm_loadu_si128(ap.add(k) as *const __m128i));
            let b = _mm256_cvtepu32_epi64(_mm_loadu_si128(jp.add(k) as *const __m128i));
            let ab = _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
            let d = _mm512_loadu_si512(dp.add(k) as *const _);
            let ds = _mm512_inserti64x4(d, sh, 1);
            let out = row as *mut u64;
            for (m, ix) in idx.iter().enumerate() {
                _mm512_storeu_si512(
                    out.add(8 * m) as *mut _,
                    _mm512_permutex2var_epi64(ab, *ix, ds),
                );
            }
            row = row.add(8);
            k += 4;
        }
        while k < hi {
            let i = *ap.add(k) as usize;
            let j = *jp.add(k) as usize;
            let dist2 = *dp.add(k);
            (*row).write(Pair { i, j, shift, dist2 });
            (*row.add(1)).write(Pair {
                i: j,
                j: i,
                shift: neg,
                dist2,
            });
            row = row.add(2);
            k += 1;
        }
        lo = hi;
    }
}

/// [`write_columns`] for a full list, eight hits (sixteen rows) at a time.
///
/// # Safety
/// AVX-512F is available, and the columns hold two rows per hit of
/// `chunk`; `cs` holds three values per row.
#[allow(clippy::incompatible_msrv)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn write_columns_avx512(
    ci: &mut [std::mem::MaybeUninit<i32>],
    cj: &mut [std::mem::MaybeUninit<i32>],
    cs: &mut [std::mem::MaybeUninit<i32>],
    cd: &mut [std::mem::MaybeUninit<f64>],
    chunk: &Scratch,
) {
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm512_castsi256_si512, _mm512_loadu_pd, _mm512_loadu_si512,
        _mm512_permutex2var_epi32, _mm512_permutexvar_epi32, _mm512_permutexvar_pd,
        _mm512_setr_epi32, _mm512_setr_epi64, _mm512_storeu_pd, _mm512_storeu_si512,
    };
    // Row 2k is (a_k, b_k, S, d_k); row 2k + 1 is (b_k, a_k, -S, d_k).
    let ab = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let ba = _mm512_setr_epi32(16, 0, 17, 1, 18, 2, 19, 3, 20, 4, 21, 5, 22, 6, 23, 7);
    let dup_lo = _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3);
    let dup_hi = _mm512_setr_epi64(4, 4, 5, 5, 6, 6, 7, 7);
    // Shift column of sixteen rows: (S, -S) eight times, 48 values.
    let pat = [
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3),
        _mm512_setr_epi32(4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1),
        _mm512_setr_epi32(2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5),
    ];
    let ap = chunk.atom.as_ptr();
    let jp = chunk.js.as_ptr();
    let dp = chunk.d2.as_ptr();
    let ip = ci.as_mut_ptr() as *mut i32;
    let jo = cj.as_mut_ptr() as *mut i32;
    let sp = cs.as_mut_ptr() as *mut i32;
    let dq = cd.as_mut_ptr() as *mut f64;
    let mut out = 0usize;
    let mut lo = 0usize;
    for (r, &hi) in chunk.run_end.iter().enumerate() {
        let s = chunk.run_shift[r];
        let six = [
            s[0], s[1], s[2], -s[0], -s[1], -s[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let src = _mm512_loadu_si512(six.as_ptr() as *const _);
        let sv = [
            _mm512_permutexvar_epi32(pat[0], src),
            _mm512_permutexvar_epi32(pat[1], src),
            _mm512_permutexvar_epi32(pat[2], src),
        ];
        let mut k = lo;
        while k + 8 <= hi {
            let a = _mm512_castsi256_si512(_mm256_loadu_si256(ap.add(k) as *const __m256i));
            let b = _mm512_castsi256_si512(_mm256_loadu_si256(jp.add(k) as *const __m256i));
            _mm512_storeu_si512(ip.add(out) as *mut _, _mm512_permutex2var_epi32(a, ab, b));
            _mm512_storeu_si512(jo.add(out) as *mut _, _mm512_permutex2var_epi32(a, ba, b));
            let d = _mm512_loadu_pd(dp.add(k));
            _mm512_storeu_pd(dq.add(out), _mm512_permutexvar_pd(dup_lo, d));
            _mm512_storeu_pd(dq.add(out + 8), _mm512_permutexvar_pd(dup_hi, d));
            let so = sp.add(3 * out);
            _mm512_storeu_si512(so as *mut _, sv[0]);
            _mm512_storeu_si512(so.add(16) as *mut _, sv[1]);
            _mm512_storeu_si512(so.add(32) as *mut _, sv[2]);
            out += 16;
            k += 8;
        }
        while k < hi {
            let a = *ap.add(k) as i32;
            let b = *jp.add(k) as i32;
            let d = *dp.add(k);
            *ip.add(out) = a;
            *jo.add(out) = b;
            *dq.add(out) = d;
            *ip.add(out + 1) = b;
            *jo.add(out + 1) = a;
            *dq.add(out + 1) = d;
            let so = sp.add(3 * out);
            *so = s[0];
            *so.add(1) = s[1];
            *so.add(2) = s[2];
            *so.add(3) = -s[0];
            *so.add(4) = -s[1];
            *so.add(5) = -s[2];
            out += 2;
            k += 1;
        }
        lo = hi;
    }
}

/// Target runs of eight a bin needs before the tile kernel tests boxes:
/// with fewer, the bin stencil has already done most of the culling and
/// the box test costs more than it skips.
const CULL_RUNS: usize = 4;

/// Tests turn the box test off to compare against it, and count the
/// target runs it skips.
#[cfg(test)]
static NO_CULL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
static CULL_SKIPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn no_cull() -> bool {
    #[cfg(test)]
    return NO_CULL.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(test))]
    false
}

/// Run `$body` with `$slot` at the first slot of every target run of
/// eight in `$block` whose [`Boxes`] entry comes within `$cut` (squared)
/// of the box of the source run of four holding slot `$s`, moved by
/// `($ox, $oy, $oz)`; in slot order. Eight target boxes are tested at a
/// time.
///
/// A source in the run, moved, lies inside the moved box (rounding is
/// monotonic), and each target lies inside its own box, so a pair's
/// relative distance is at least the box gap. `$cut` is the cutoff plus
/// twice the expansion margin, so a skipped run holds no lane the tile
/// kernel would keep, and the rows and their order are unchanged.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
macro_rules! near_runs {
    ($c:expr, $block:expr, $nvec:expr, $ox:expr, $oy:expr, $oz:expr, $cut:expr, $zero:expr,
     $s:expr, $slot:ident, $body:block) => {{
        use std::arch::x86_64::{
            _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_mask_cmp_pd_mask, _mm512_max_pd,
            _mm512_set1_pd, _mm512_sub_pd, _CMP_LT_OQ,
        };
        let g = $block.ig + ($s - $block.i_lo) / 4;
        let lo = [
            _mm512_set1_pd(*$c.gb[0].add(g) + $ox),
            _mm512_set1_pd(*$c.gb[1].add(g) + $oy),
            _mm512_set1_pd(*$c.gb[2].add(g) + $oz),
        ];
        let hi = [
            _mm512_set1_pd(*$c.gb[3].add(g) + $ox),
            _mm512_set1_pd(*$c.gb[4].add(g) + $oy),
            _mm512_set1_pd(*$c.gb[5].add(g) + $oz),
        ];
        let mut m0 = 0usize;
        while m0 < $nvec {
            let v = $block.jv + m0;
            let mut d2 = $zero;
            for ax in 0..3 {
                let gap = _mm512_max_pd(
                    _mm512_max_pd(
                        _mm512_sub_pd(_mm512_loadu_pd($c.vb[ax].add(v)), hi[ax]),
                        _mm512_sub_pd(lo[ax], _mm512_loadu_pd($c.vb[3 + ax].add(v))),
                    ),
                    $zero,
                );
                d2 = _mm512_fmadd_pd(gap, gap, d2);
            }
            let valid: u8 = if $nvec - m0 >= 8 {
                0xff
            } else {
                ((1u32 << ($nvec - m0)) - 1) as u8
            };
            let mut near: u8 = _mm512_mask_cmp_pd_mask(valid, d2, $cut, _CMP_LT_OQ);
            #[cfg(test)]
            CULL_SKIPS.fetch_add(
                (valid & !near).count_ones() as usize,
                std::sync::atomic::Ordering::Relaxed,
            );
            while near != 0 {
                let $slot = $block.j_lo + 8 * (m0 + near.trailing_zeros() as usize);
                near &= near - 1;
                $body
            }
            m0 += 8;
        }
    }};
}

/// Where the fused kernel writes: `Pair` rows as words, or four columns.
/// `n` is the row count so far. Each hit vector writes whole registers
/// past its last kept row; the next vector overwrites them, so every
/// buffer keeps [`FUSED_SLACK`] rows of room past `n`.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
struct FusedOut {
    rows: *mut u64,
    ci: *mut i32,
    cj: *mut i32,
    cs: *mut i32,
    cd: *mut f64,
    n: usize,
}

/// Mean atoms per bin under which one thread runs the tile inlined into
/// its block loop: with a few atoms a block, the call into the tile is
/// most of the block (256 atoms in the 18 Å cube, four a bin, 0.020 ->
/// 0.017 ms), while with full bins the tile's own register allocation is
/// a little better out of line.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
const INLINE_BLOCKS: usize = 8;

/// Where [`Plan::fused_into`] writes: room for `room` more rows past the
/// first `n`, on demand.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
trait FusedSink {
    fn room(&mut self, n: usize, room: usize) -> FusedOut;
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
impl FusedSink for Vec<Pair> {
    fn room(&mut self, n: usize, room: usize) -> FusedOut {
        if self.capacity() - n < room {
            // Safety: the first `n` rows are written.
            unsafe { self.set_len(n) };
            self.reserve(room);
        }
        FusedOut {
            rows: self.as_mut_ptr() as *mut u64,
            ci: std::ptr::null_mut(),
            cj: std::ptr::null_mut(),
            cs: std::ptr::null_mut(),
            cd: std::ptr::null_mut(),
            n,
        }
    }
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
impl FusedSink for PairColumns {
    fn room(&mut self, n: usize, room: usize) -> FusedOut {
        let cap = self
            .i
            .capacity()
            .min(self.j.capacity())
            .min(self.shift.capacity())
            .min(self.dist2.capacity());
        if cap - n < room {
            // Safety: the first `n` rows of every column are written.
            unsafe {
                self.i.set_len(n);
                self.j.set_len(n);
                self.shift.set_len(n);
                self.dist2.set_len(n);
            }
            self.i.reserve(room);
            self.j.reserve(room);
            self.shift.reserve(room);
            self.dist2.reserve(room);
        }
        FusedOut {
            rows: std::ptr::null_mut(),
            ci: self.i.as_mut_ptr(),
            cj: self.j.as_mut_ptr(),
            cs: self.shift.as_mut_ptr() as *mut i32,
            cd: self.dist2.as_mut_ptr(),
            n,
        }
    }
}

/// Rows a hit vector may write past the last row it keeps.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
const FUSED_SLACK: usize = 16;

/// The AVX-512 tile kernel of [`avx512_scan`] with a full list written
/// in place: `Pair` rows when `COLS` is false, the four columns when it
/// is true. Hit lanes are compressed, then four hits become eight rows
/// (or sixteen column entries) in whole registers.
///
/// # Safety
/// `block` indexes every column of `c`. Every buffer of `out` has room
/// for `out.n` plus every row of the block plus [`FUSED_SLACK`].
#[allow(clippy::incompatible_msrv)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f,avx512vl,popcnt,fma")]
#[inline]
unsafe fn avx512_fused<const COLS: bool, const INLINED: bool>(
    c: TileCols,
    cut2: f64,
    margin: f64,
    block: &Block,
    words: &RowWords,
    out: &mut FusedOut,
) {
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_maskz_compress_epi32, _mm256_maskz_loadu_epi32,
        _mm512_add_pd, _mm512_castpd_si512, _mm512_castsi256_si512, _mm512_fmadd_pd,
        _mm512_loadu_pd, _mm512_loadu_si512, _mm512_mask_blend_epi32, _mm512_mask_cmp_pd_mask,
        _mm512_mask_set1_epi32, _mm512_maskz_compress_pd, _mm512_maskz_loadu_pd,
        _mm512_permutex2var_epi32, _mm512_permutexvar_epi32, _mm512_permutexvar_pd,
        _mm512_set1_epi32, _mm512_set1_pd, _mm512_setr_epi32, _mm512_setr_epi64, _mm512_storeu_pd,
        _mm512_storeu_si512, _CMP_LT_OQ,
    };
    let [sx, sy, sz] = block.shift;
    let [ox, oy, oz] = block.delta;
    let shift_s = block.shift_s;
    let neg = [-shift_s[0], -shift_s[1], -shift_s[2]];
    // Dwords 8..16 of the row source: the source id, a zero, S, and -S.
    let consts = _mm512_setr_epi32(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, shift_s[0], shift_s[1], shift_s[2], neg[0], neg[1], neg[2],
    );
    // Columns: row 2k is (i, j_k, S, d_k) and row 2k + 1 is (j_k, i, -S, d_k).
    let ij = _mm512_setr_epi32(0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23);
    let ji = _mm512_setr_epi32(16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0);
    let dup_lo = _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3);
    let dup_hi = _mm512_setr_epi64(4, 4, 5, 5, 6, 6, 7, 7);
    let six = [
        shift_s[0], shift_s[1], shift_s[2], neg[0], neg[1], neg[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let six = _mm512_loadu_si512(six.as_ptr() as *const _);
    let sv = [
        _mm512_permutexvar_epi32(
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3),
            six,
        ),
        _mm512_permutexvar_epi32(
            _mm512_setr_epi32(4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1),
            six,
        ),
        _mm512_permutexvar_epi32(
            _mm512_setr_epi32(2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5),
            six,
        ),
    ];
    let cut_lo = _mm512_set1_pd(cut2 - margin);
    let cut_hi = cut2 + margin;
    let mut n = out.n;

    // One hit decided by the direct formula: both rows, exactly.
    macro_rules! put_one {
        ($iu:expr, $j:expr, $d2:expr) => {{
            let (a, b, d) = ($iu, $j, $d2);
            if COLS {
                *out.ci.add(n) = a as i32;
                *out.cj.add(n) = b as i32;
                *out.cd.add(n) = d;
                *out.ci.add(n + 1) = b as i32;
                *out.cj.add(n + 1) = a as i32;
                *out.cd.add(n + 1) = d;
                let so = out.cs.add(3 * n);
                *so = shift_s[0];
                *so.add(1) = shift_s[1];
                *so.add(2) = shift_s[2];
                *so.add(3) = neg[0];
                *so.add(4) = neg[1];
                *so.add(5) = neg[2];
            } else {
                let row = (out.rows as *mut std::mem::MaybeUninit<Pair>).add(n);
                let (i, j) = (a as usize, b as usize);
                (*row).write(Pair {
                    i,
                    j,
                    shift: shift_s,
                    dist2: d,
                });
                (*row.add(1)).write(Pair {
                    i: j,
                    j: i,
                    shift: neg,
                    dist2: d,
                });
            }
            n += 2;
        }};
    }
    macro_rules! direct {
        ($s:expr, $slot:expr, $mask:expr) => {{
            let s = $s;
            let iu = *c.id.add(s);
            let px = *c.x.add(s) - sx;
            let py = *c.y.add(s) - sy;
            let pz = *c.z.add(s) - sz;
            let mut bits: u32 = u32::from($mask);
            while bits != 0 {
                let slot = $slot + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let dx = *c.x.add(slot) - px;
                let dy = *c.y.add(slot) - py;
                let dz = *c.z.add(slot) - pz;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < cut2 {
                    put_one!(iu, *c.id.add(slot), d2);
                }
            }
        }};
    }
    macro_rules! source {
        ($s:expr) => {{
            let s = $s;
            let px = *c.rx.add(s) + ox;
            let py = *c.ry.add(s) + oy;
            let pz = *c.rz.add(s) + oz;
            let pp = px * px + py * py + pz * pz;
            (
                *c.id.add(s),
                _mm512_set1_pd(-2.0 * px),
                _mm512_set1_pd(-2.0 * py),
                _mm512_set1_pd(-2.0 * pz),
                _mm512_set1_pd(pp),
                _mm512_set1_pd(cut_hi - pp),
            )
        }};
    }
    // The kept lanes `lo` of one source: compress, then whole registers.
    macro_rules! emit {
        ($lo:expr, $iu:expr, $d2v:expr, $jids:expr) => {{
            let lo: u8 = $lo;
            let k = lo.count_ones() as usize;
            let dc = _mm512_maskz_compress_pd(lo, $d2v);
            let jc = _mm256_maskz_compress_epi32(lo, $jids);
            if COLS {
                let iv = _mm512_set1_epi32($iu as i32);
                let jz = _mm512_castsi256_si512(jc);
                _mm512_storeu_si512(
                    out.ci.add(n) as *mut _,
                    _mm512_permutex2var_epi32(iv, ij, jz),
                );
                _mm512_storeu_si512(
                    out.cj.add(n) as *mut _,
                    _mm512_permutex2var_epi32(iv, ji, jz),
                );
                _mm512_storeu_pd(out.cd.add(n), _mm512_permutexvar_pd(dup_lo, dc));
                let so = out.cs.add(3 * n);
                _mm512_storeu_si512(so as *mut _, sv[0]);
                if k > 2 {
                    _mm512_storeu_si512(so.add(16) as *mut _, sv[1]);
                }
                if k > 4 {
                    _mm512_storeu_pd(out.cd.add(n + 8), _mm512_permutexvar_pd(dup_hi, dc));
                }
                if k > 5 {
                    _mm512_storeu_si512(so.add(32) as *mut _, sv[2]);
                }
            } else {
                // Dword sources, as in [`RowWords::index32`]: the eight ids
                // and this source's constants, then the eight distances, so
                // both groups of four hits permute from the same pair.
                let at = out.rows.add(5 * n);
                let a = _mm512_mask_blend_epi32(
                    0x00ff,
                    _mm512_mask_set1_epi32(consts, 0x0100, $iu as i32),
                    _mm512_castsi256_si512(jc),
                );
                let b = _mm512_castpd_si512(dc);
                // Two rows per hit are ten words: only the registers that
                // hold them, so a lone hit takes two stores, not five.
                let regs = if k >= 4 { 5 } else { (10 * k + 7) / 8 };
                for (m, ix) in words.index32[0].iter().enumerate().take(regs) {
                    _mm512_storeu_si512(
                        at.add(8 * m) as *mut _,
                        _mm512_permutex2var_epi32(
                            a,
                            _mm512_loadu_si512(ix.as_ptr() as *const _),
                            b,
                        ),
                    );
                }
                if k > 4 {
                    let at = at.add(40);
                    let regs = if k >= 8 { 5 } else { (10 * (k - 4) + 7) / 8 };
                    for (m, ix) in words.index32[1].iter().enumerate().take(regs) {
                        _mm512_storeu_si512(
                            at.add(8 * m) as *mut _,
                            _mm512_permutex2var_epi32(
                                a,
                                _mm512_loadu_si512(ix.as_ptr() as *const _),
                                b,
                            ),
                        );
                    }
                }
            }
            n += 2 * k;
        }};
    }
    macro_rules! lanes {
        ($s:expr, $slot:expr, $tm:expr, $src:expr, $jx:expr, $jy:expr, $jz:expr, $jr:expr, $jids:expr) => {{
            let (iu, ax, ay, az, pp, thr) = $src;
            let t = _mm512_fmadd_pd(
                $jz,
                az,
                _mm512_fmadd_pd($jy, ay, _mm512_fmadd_pd($jx, ax, $jr)),
            );
            let hi: u8 = _mm512_mask_cmp_pd_mask($tm, t, thr, _CMP_LT_OQ);
            if hi != 0 {
                let d2v = _mm512_add_pd(t, pp);
                let lo: u8 = _mm512_mask_cmp_pd_mask(hi, d2v, cut_lo, _CMP_LT_OQ);
                if lo == hi {
                    emit!(lo, iu, d2v, $jids);
                } else {
                    direct!($s, $slot, hi);
                }
            }
        }};
    }
    macro_rules! load {
        ($slot:expr, $j_hi:expr) => {{
            let slot = $slot;
            if slot + 8 <= $j_hi {
                (
                    0xffu8,
                    _mm512_loadu_pd(c.rx.add(slot)),
                    _mm512_loadu_pd(c.ry.add(slot)),
                    _mm512_loadu_pd(c.rz.add(slot)),
                    _mm512_loadu_pd(c.r2.add(slot)),
                    _mm256_loadu_si256(c.id.add(slot) as *const __m256i),
                )
            } else {
                let tm: u8 = ((1u32 << ($j_hi - slot)) - 1) as u8;
                (
                    tm,
                    _mm512_maskz_loadu_pd(tm, c.rx.add(slot)),
                    _mm512_maskz_loadu_pd(tm, c.ry.add(slot)),
                    _mm512_maskz_loadu_pd(tm, c.rz.add(slot)),
                    _mm512_maskz_loadu_pd(tm, c.r2.add(slot)),
                    _mm256_maskz_loadu_epi32(tm, c.id.add(slot) as *const i32),
                )
            }
        }};
    }
    let j_hi = block.j_hi;
    let nvec = (j_hi - block.j_lo + 7) / 8;
    if !block.tri && c.cull && nvec >= CULL_RUNS {
        let cut_box = _mm512_set1_pd(cut2 + 2.0 * margin);
        let zero = std::arch::x86_64::_mm512_setzero_pd();
        let mut s = block.i_lo;
        while s + 4 <= block.i_hi {
            let s0 = source!(s);
            let s1 = source!(s + 1);
            let s2 = source!(s + 2);
            let s3 = source!(s + 3);
            near_runs!(c, block, nvec, ox, oy, oz, cut_box, zero, s, slot, {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
                lanes!(s + 1, slot, tm, s1, jx, jy, jz, jr, jids);
                lanes!(s + 2, slot, tm, s2, jx, jy, jz, jr, jids);
                lanes!(s + 3, slot, tm, s3, jx, jy, jz, jr, jids);
            });
            s += 4;
        }
        while s < block.i_hi {
            let s0 = source!(s);
            near_runs!(c, block, nvec, ox, oy, oz, cut_box, zero, s, slot, {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
            });
            s += 1;
        }
    } else if !block.tri {
        let mut s = block.i_lo;
        while s + 4 <= block.i_hi {
            let s0 = source!(s);
            let s1 = source!(s + 1);
            let s2 = source!(s + 2);
            let s3 = source!(s + 3);
            let mut slot = block.j_lo;
            while slot < j_hi {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
                lanes!(s + 1, slot, tm, s1, jx, jy, jz, jr, jids);
                lanes!(s + 2, slot, tm, s2, jx, jy, jz, jr, jids);
                lanes!(s + 3, slot, tm, s3, jx, jy, jz, jr, jids);
                slot += 8;
            }
            s += 4;
        }
        while s < block.i_hi {
            let s0 = source!(s);
            let mut slot = block.j_lo;
            while slot < j_hi {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
                slot += 8;
            }
            s += 1;
        }
    } else {
        for s in block.i_lo..block.i_hi {
            let s0 = source!(s);
            let mut slot = s + 1;
            while slot < j_hi {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
                slot += 8;
            }
        }
    }
    out.n = n;
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
impl Plan {
    fn tile_cols(&self) -> TileCols {
        TileCols::of(&self.grid)
    }

    /// `visit(block)` for the home block and every partner of each bin,
    /// in bin order.
    fn each_block(&self, mut visit: impl FnMut(&Block)) {
        let ncell = self.grid.offsets.len() - 1;
        for cell in 0..ncell {
            let lo = self.grid.offsets[cell];
            let hi = self.grid.offsets[cell + 1];
            if lo == hi {
                continue;
            }
            if hi - lo >= 2 {
                visit(&Block {
                    i_lo: lo,
                    i_hi: hi,
                    j_lo: lo,
                    j_hi: hi,
                    shift_s: [0, 0, 0],
                    shift: [0.0; 3],
                    delta: [0.0; 3],
                    tri: true,
                    jv: self.grid.boxes.vstart[cell],
                    ig: self.grid.boxes.gstart[cell],
                });
            }
            let (p0, p1) = (self.partners.off[cell], self.partners.off[cell + 1]);
            for p in &self.partners.items[p0..p1] {
                let (j_lo, j_hi) = (self.grid.offsets[p.jc], self.grid.offsets[p.jc + 1]);
                if j_lo == j_hi {
                    continue;
                }
                visit(&Block {
                    i_lo: lo,
                    i_hi: hi,
                    j_lo,
                    j_hi,
                    shift_s: p.shift_s,
                    shift: p.shift,
                    delta: p.delta,
                    tri: false,
                    jv: self.grid.boxes.vstart[p.jc],
                    ig: self.grid.boxes.gstart[cell],
                });
            }
        }
    }

    /// Rows a block can add, plus the unkept registers past them.
    fn block_room(block: &Block) -> usize {
        2 * (block.i_hi - block.i_lo) * (block.j_hi - block.j_lo) + FUSED_SLACK
    }

    /// [`Plan::fused`]: every row appended from the tile kernel.
    fn fused_rows(&self, words: &RowWords) -> Vec<Pair> {
        let _timer = crate::pop::JobTimer::new();
        let cols = self.tile_cols();
        let guess = 2 * self.walk().guess_hits(self.grid.atoms()) + FUSED_SLACK;
        let mut out: Vec<Pair> = Vec::with_capacity(guess);
        advise_huge(&mut out);
        let ncell = self.grid.offsets.len() - 1;
        if self.grid.atoms() < INLINE_BLOCKS * ncell {
            // Safety: AVX-512 was detected for this plan.
            let n = unsafe { self.fused_into::<false, _>(cols, words, &mut out) };
            // Safety: every row below `n` is written.
            unsafe { out.set_len(n) };
            return out;
        }
        let mut n = 0usize;
        self.each_block(|block| {
            let room = Self::block_room(block);
            if out.capacity() - n < room {
                // Safety: the first `n` rows are written.
                unsafe { out.set_len(n) };
                out.reserve(room);
            }
            let mut fo = FusedOut {
                rows: out.as_mut_ptr() as *mut u64,
                ci: std::ptr::null_mut(),
                cj: std::ptr::null_mut(),
                cs: std::ptr::null_mut(),
                cd: std::ptr::null_mut(),
                n,
            };
            // Safety: AVX-512 was detected for this plan, and `out` has
            // room for `n` plus the block plus the slack.
            unsafe {
                avx512_fused::<false, false>(cols, self.cut2, self.margin, block, words, &mut fo)
            };
            n = fo.n;
        });
        // Safety: every row below `n` is written.
        unsafe { out.set_len(n) };
        out
    }

    /// The block loop of [`Plan::fused_rows`] and [`Plan::fused_columns`],
    /// compiled with the tile's target features so the tile inlines into
    /// it: [`Plan::each_block`]'s order, the home block of each bin and
    /// then its partners.
    ///
    /// # Safety
    /// AVX-512F, AVX-512VL, POPCNT, and FMA are available.
    #[allow(clippy::incompatible_msrv)]
    #[target_feature(enable = "avx512f,avx512vl,popcnt,fma")]
    unsafe fn fused_into<const COLS: bool, S: FusedSink>(
        &self,
        cols: TileCols,
        words: &RowWords,
        sink: &mut S,
    ) -> usize {
        let mut n = 0usize;
        let ncell = self.grid.offsets.len() - 1;
        for cell in 0..ncell {
            let (lo, hi) = (self.grid.offsets[cell], self.grid.offsets[cell + 1]);
            if lo == hi {
                continue;
            }
            let (p0, p1) = (self.partners.off[cell], self.partners.off[cell + 1]);
            for q in p0..=p1 {
                let block = if q == p0 {
                    if hi - lo < 2 {
                        continue;
                    }
                    Block {
                        i_lo: lo,
                        i_hi: hi,
                        j_lo: lo,
                        j_hi: hi,
                        shift_s: [0, 0, 0],
                        shift: [0.0; 3],
                        delta: [0.0; 3],
                        tri: true,
                        jv: self.grid.boxes.vstart[cell],
                        ig: self.grid.boxes.gstart[cell],
                    }
                } else {
                    let p = &self.partners.items[q - 1];
                    let (j_lo, j_hi) = (self.grid.offsets[p.jc], self.grid.offsets[p.jc + 1]);
                    if j_lo == j_hi {
                        continue;
                    }
                    Block {
                        i_lo: lo,
                        i_hi: hi,
                        j_lo,
                        j_hi,
                        shift_s: p.shift_s,
                        shift: p.shift,
                        delta: p.delta,
                        tri: false,
                        jv: self.grid.boxes.vstart[p.jc],
                        ig: self.grid.boxes.gstart[cell],
                    }
                };
                let mut fo = sink.room(n, Self::block_room(&block));
                // Safety: the sink has room for `n` plus the block plus the
                // slack, and the features are the caller's.
                avx512_fused::<COLS, true>(cols, self.cut2, self.margin, &block, words, &mut fo);
                n = fo.n;
            }
        }
        n
    }

    /// [`Plan::fused`] into four columns.
    fn fused_columns(&self, out: &mut PairColumns) {
        let _timer = crate::pop::JobTimer::new();
        let words = match RowWords::of_pair() {
            Some(w) => w,
            None => return self.search().fill_columns(out),
        };
        let cols = self.tile_cols();
        out.i.clear();
        out.j.clear();
        out.shift.clear();
        out.dist2.clear();
        let guess = 2 * self.walk().guess_hits(self.grid.atoms()) + FUSED_SLACK;
        let grow = |out: &mut PairColumns, n: usize, room: usize| {
            // Safety: the first `n` rows of every column are written.
            unsafe {
                out.i.set_len(n);
                out.j.set_len(n);
                out.shift.set_len(n);
                out.dist2.set_len(n);
            }
            out.i.reserve(room);
            out.j.reserve(room);
            out.shift.reserve(room);
            out.dist2.reserve(room);
        };
        grow(out, 0, guess);
        out.advise_huge();
        let ncell = self.grid.offsets.len() - 1;
        if self.grid.atoms() < INLINE_BLOCKS * ncell {
            // Safety: AVX-512 was detected for this plan.
            let n = unsafe { self.fused_into::<true, _>(cols, &words, out) };
            // Safety: every row below `n` is written in every column.
            unsafe {
                out.i.set_len(n);
                out.j.set_len(n);
                out.shift.set_len(n);
                out.dist2.set_len(n);
            }
            return;
        }
        let mut n = 0usize;
        self.each_block(|block| {
            let room = Self::block_room(block);
            let cap = out
                .i
                .capacity()
                .min(out.j.capacity())
                .min(out.shift.capacity())
                .min(out.dist2.capacity());
            if cap - n < room {
                grow(out, n, room);
            }
            let mut fo = FusedOut {
                rows: std::ptr::null_mut(),
                ci: out.i.as_mut_ptr(),
                cj: out.j.as_mut_ptr(),
                cs: out.shift.as_mut_ptr() as *mut i32,
                cd: out.dist2.as_mut_ptr(),
                n,
            };
            // Safety: AVX-512 was detected for this plan, and every column
            // has room for `n` plus the block plus the slack.
            unsafe {
                avx512_fused::<true, false>(cols, self.cut2, self.margin, block, &words, &mut fo)
            };
            n = fo.n;
        });
        // Safety: every row below `n` is written in every column.
        unsafe {
            out.i.set_len(n);
            out.j.set_len(n);
            out.shift.set_len(n);
            out.dist2.set_len(n);
        }
    }
}

impl Walk<'_> {
    fn scan_block(&self, scratch: &mut Scratch, block: &Block) {
        if block.i_lo >= block.i_hi || block.j_lo >= block.j_hi {
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.scan_scalar(scratch, block);
        }
        #[cfg(target_arch = "x86_64")]
        {
            let ns = block.i_hi - block.i_lo;
            let span = block.j_hi - block.j_lo;
            if span == 0 || ns > MAX_SCRATCH / span {
                self.scan_scalar(scratch, block);
                return;
            }
            let before = scratch.js.len();
            scratch.reserve_more(ns * span + STORE_SLACK);
            #[cfg(linkcell_avx512)]
            if self.simd == 2 {
                let cols = TileCols::of(self.grid);
                unsafe {
                    avx512_scan(cols, self.cut2, self.margin, block, scratch);
                }
                scratch.note(before, block.shift_s);
                return;
            }
            if self.simd >= 1 {
                unsafe {
                    avx_scan(
                        &self.grid.coords.x,
                        &self.grid.coords.y,
                        &self.grid.coords.z,
                        &self.grid.coords.id,
                        self.cut2,
                        block,
                        scratch,
                    );
                }
                scratch.note(before, block.shift_s);
                return;
            }
            self.scan_scalar(scratch, block);
        }
    }

    fn scan_scalar(&self, scratch: &mut Scratch, block: &Block) {
        let before = scratch.js.len();
        let sx = block.shift[0];
        let sy = block.shift[1];
        let sz = block.shift[2];
        for s in block.i_lo..block.i_hi {
            let j_lo = if block.tri { s + 1 } else { block.j_lo };
            if j_lo >= block.j_hi {
                continue;
            }
            let c = &self.grid.coords;
            let i = c.id[s];
            let px = c.x[s] - sx;
            let py = c.y[s] - sy;
            let pz = c.z[s] - sz;
            for slot in j_lo..block.j_hi {
                let dx = c.x[slot] - px;
                let dy = c.y[slot] - py;
                let dz = c.z[slot] - pz;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < self.cut2 {
                    let j = c.id[slot];
                    if j == i && block.shift_s == [0, 0, 0] {
                        continue;
                    }
                    scratch.push(i, j, d2);
                }
            }
        }
        scratch.note(before, block.shift_s);
    }

    fn append_cell(&self, scratch: &mut Scratch, cell: usize) {
        let lo = self.grid.offsets[cell];
        let hi = self.grid.offsets[cell + 1];
        if lo == hi {
            return;
        }
        if hi - lo >= 2 {
            self.scan_block(
                scratch,
                &Block {
                    i_lo: lo,
                    i_hi: hi,
                    j_lo: lo,
                    j_hi: hi,
                    shift_s: [0, 0, 0],
                    shift: [0.0; 3],
                    delta: [0.0; 3],
                    tri: true,
                    jv: self.grid.boxes.vstart[cell],
                    ig: self.grid.boxes.gstart[cell],
                },
            );
        }
        let p0 = self.partners.off[cell];
        let p1 = self.partners.off[cell + 1];
        for partner in &self.partners.items[p0..p1] {
            let (j_lo, j_hi) = (
                self.grid.offsets[partner.jc],
                self.grid.offsets[partner.jc + 1],
            );
            if j_lo == j_hi {
                continue;
            }
            self.scan_block(
                scratch,
                &Block {
                    i_lo: lo,
                    i_hi: hi,
                    j_lo,
                    j_hi,
                    shift_s: partner.shift_s,
                    shift: partner.shift,
                    delta: partner.delta,
                    tri: false,
                    jv: self.grid.boxes.vstart[partner.jc],
                    ig: self.grid.boxes.gstart[cell],
                },
            );
        }
    }

    /// Ideal-gas hit count for `n_src` sources, with a quarter more.
    /// A denser shell grows the buffer.
    fn guess_hits(&self, n_src: usize) -> usize {
        let w = self.grid.widths;
        let volume = (w[0] * w[1] * w[2]).max(1.0e-30);
        let radius = self.cut2.sqrt();
        let shell = 4.1887902047863905 * radius * radius * radius;
        let n = self.grid.atoms().max(1) as f64;
        let neighbors = (n * shell / volume).ceil().max(1.0);
        ((n_src as f64) * neighbors * 0.5 * 1.25) as usize
    }

    fn fill_range(&self, start: usize, end: usize) -> Scratch {
        let _timer = crate::pop::JobTimer::new();
        let mut scratch = Scratch::new();
        let n_src = self.grid.offsets[end] - self.grid.offsets[start];
        scratch.reserve_more(self.guess_hits(n_src));
        for cell in start..end {
            self.append_cell(&mut scratch, cell);
        }
        scratch
    }

    /// Hits by cell range: with several threads, thread `k` searches
    /// range `k`, so the writer on that thread reads its own cache.
    fn collect(&self, threads: usize) -> Vec<Scratch> {
        let ncell = self.grid.offsets.len() - 1;
        #[cfg(feature = "parallel")]
        if threads > 1 && ncell > 1 {
            let ranges = cell_ranges(&self.grid.offsets, threads * knobs::chunks());
            if ranges.len() > 1 && ranges.len() <= rayon::current_num_threads() {
                let found: Vec<Option<Scratch>> = rayon::broadcast(|ctx| {
                    ranges
                        .get(ctx.index())
                        .map(|&(start, end)| self.fill_range(start, end))
                });
                return found.into_iter().flatten().collect();
            }
            if ranges.len() > 1 {
                use rayon::prelude::*;
                return ranges
                    .par_iter()
                    .map(|&(start, end)| self.fill_range(start, end))
                    .collect();
            }
        }
        let _ = threads;
        vec![self.fill_range(0, ncell)]
    }
}

#[cfg(feature = "parallel")]
fn cell_ranges(offsets: &[usize], threads: usize) -> Vec<(usize, usize)> {
    let ncell = offsets.len() - 1;
    if ncell == 0 {
        return Vec::new();
    }
    let threads = threads.max(1).min(ncell);
    if threads == 1 {
        return vec![(0, ncell)];
    }
    let total = offsets[ncell];
    if total == 0 {
        return vec![(0, ncell)];
    }
    let mut ranges = Vec::with_capacity(threads);
    let mut start = 0usize;
    for t in 1..threads {
        let target = total * t / threads;
        let mut end = start;
        while end < ncell && offsets[end] < target {
            end += 1;
        }
        if end == start {
            end = (start + 1).min(ncell);
        }
        if end <= start {
            break;
        }
        ranges.push((start, end));
        start = end;
    }
    if start < ncell {
        ranges.push((start, ncell));
    }
    ranges
}

/// Slot columns one AVX-512 tile reads, and the [`Boxes`] of its runs.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[derive(Clone, Copy)]
struct TileCols {
    x: *const f64,
    y: *const f64,
    z: *const f64,
    rx: *const f64,
    ry: *const f64,
    rz: *const f64,
    r2: *const f64,
    id: *const u32,
    cull: bool,
    vb: [*const f64; 6],
    gb: [*const f64; 6],
}

#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
impl TileCols {
    fn of(grid: &Grid) -> Self {
        let c = &grid.coords;
        let b = &grid.boxes;
        TileCols {
            x: c.x.as_ptr(),
            y: c.y.as_ptr(),
            z: c.z.as_ptr(),
            rx: c.rx.as_ptr(),
            ry: c.ry.as_ptr(),
            rz: c.rz.as_ptr(),
            r2: c.r2.as_ptr(),
            id: c.id.as_ptr(),
            cull: b.on,
            vb: std::array::from_fn(|k| b.v[k].as_ptr()),
            gb: std::array::from_fn(|k| b.g[k].as_ptr()),
        }
    }
}

/// # Safety
/// `block` indexes every column of `c`. `scratch` has room for every
/// pair in the block plus [`STORE_SLACK`] slots.
///
/// A source `p` in the block, moved by `block.delta`, is `p'` relative
/// to the target bin's corner. Each lane forms
/// `t = |r_q|^2 - 2 r_q . p'` with three fused multiply-adds, and
/// `t + |p'|^2` is its squared distance. A lane within `margin` of
/// `cut2` is decided by the direct `|q - (p - S)|^2` of [`Walk::scan_scalar`].
#[allow(clippy::incompatible_msrv)]
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[target_feature(enable = "avx512f,avx512vl,popcnt,fma")]
#[inline(never)]
unsafe fn avx512_scan(c: TileCols, cut2: f64, margin: f64, block: &Block, scratch: &mut Scratch) {
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_maskz_compress_epi32, _mm256_maskz_loadu_epi32,
        _mm256_set1_epi32, _mm256_storeu_si256, _mm512_add_pd, _mm512_fmadd_pd, _mm512_loadu_pd,
        _mm512_mask_cmp_pd_mask, _mm512_maskz_compress_pd, _mm512_maskz_loadu_pd, _mm512_set1_pd,
        _mm512_storeu_pd, _CMP_LT_OQ,
    };
    let atom_p = scratch.atom.as_mut_ptr();
    let js_p = scratch.js.as_mut_ptr();
    let d2_p = scratch.d2.as_mut_ptr();
    let mut n = scratch.js.len();
    let [sx, sy, sz] = block.shift;
    let [ox, oy, oz] = block.delta;
    let cut_lo = _mm512_set1_pd(cut2 - margin);
    let cut_hi = cut2 + margin;

    // Lanes of one source the expanded form cannot decide: the direct
    // formula, one lane at a time.
    macro_rules! direct {
        ($s:expr, $slot:expr, $mask:expr) => {{
            let s = $s;
            let iu = *c.id.add(s);
            let px = *c.x.add(s) - sx;
            let py = *c.y.add(s) - sy;
            let pz = *c.z.add(s) - sz;
            let mut bits: u32 = u32::from($mask);
            while bits != 0 {
                let slot = $slot + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let dx = *c.x.add(slot) - px;
                let dy = *c.y.add(slot) - py;
                let dz = *c.z.add(slot) - pz;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < cut2 {
                    atom_p.add(n).write(iu);
                    js_p.add(n).write(*c.id.add(slot));
                    d2_p.add(n).write(d2);
                    n += 1;
                }
            }
        }};
    }
    // A source: its id lane, -2 p', |p'|^2, and the threshold on `t`.
    macro_rules! source {
        ($s:expr) => {{
            let s = $s;
            let px = *c.rx.add(s) + ox;
            let py = *c.ry.add(s) + oy;
            let pz = *c.rz.add(s) + oz;
            let pp = px * px + py * py + pz * pz;
            (
                _mm256_set1_epi32(*c.id.add(s) as i32),
                _mm512_set1_pd(-2.0 * px),
                _mm512_set1_pd(-2.0 * py),
                _mm512_set1_pd(-2.0 * pz),
                _mm512_set1_pd(pp),
                _mm512_set1_pd(cut_hi - pp),
            )
        }};
    }
    // Hit lanes are compressed in a register and stored at full width.
    // Lanes past the hit count are scratch that the next store or
    // `finish` overwrites; `STORE_SLACK` keeps them inside capacity.
    // The occupants of a mesh are distinct atoms, so a zero-shift block
    // never pairs an atom with itself: the home cell starts at `s + 1`
    // and every partner is another bin.
    macro_rules! lanes {
        ($s:expr, $slot:expr, $tm:expr, $src:expr, $jx:expr, $jy:expr, $jz:expr, $jr:expr, $jids:expr) => {{
            let (iv, ax, ay, az, pp, thr) = $src;
            let t = _mm512_fmadd_pd(
                $jz,
                az,
                _mm512_fmadd_pd($jy, ay, _mm512_fmadd_pd($jx, ax, $jr)),
            );
            let hi: u8 = _mm512_mask_cmp_pd_mask($tm, t, thr, _CMP_LT_OQ);
            if hi != 0 {
                let d2v = _mm512_add_pd(t, pp);
                let lo: u8 = _mm512_mask_cmp_pd_mask(hi, d2v, cut_lo, _CMP_LT_OQ);
                if lo == hi {
                    _mm512_storeu_pd(d2_p.add(n), _mm512_maskz_compress_pd(lo, d2v));
                    _mm256_storeu_si256(
                        js_p.add(n) as *mut __m256i,
                        _mm256_maskz_compress_epi32(lo, $jids),
                    );
                    _mm256_storeu_si256(atom_p.add(n) as *mut __m256i, iv);
                    n += lo.count_ones() as usize;
                } else {
                    direct!($s, $slot, hi);
                }
            }
        }};
    }
    macro_rules! load {
        ($slot:expr, $j_hi:expr) => {{
            let slot = $slot;
            if slot + 8 <= $j_hi {
                (
                    0xffu8,
                    _mm512_loadu_pd(c.rx.add(slot)),
                    _mm512_loadu_pd(c.ry.add(slot)),
                    _mm512_loadu_pd(c.rz.add(slot)),
                    _mm512_loadu_pd(c.r2.add(slot)),
                    _mm256_loadu_si256(c.id.add(slot) as *const __m256i),
                )
            } else {
                let tm: u8 = ((1u32 << ($j_hi - slot)) - 1) as u8;
                (
                    tm,
                    _mm512_maskz_loadu_pd(tm, c.rx.add(slot)),
                    _mm512_maskz_loadu_pd(tm, c.ry.add(slot)),
                    _mm512_maskz_loadu_pd(tm, c.rz.add(slot)),
                    _mm512_maskz_loadu_pd(tm, c.r2.add(slot)),
                    _mm256_maskz_loadu_epi32(tm, c.id.add(slot) as *const i32),
                )
            }
        }};
    }

    let j_hi = block.j_hi;
    let nvec = (j_hi - block.j_lo + 7) / 8;
    if !block.tri && c.cull && nvec >= CULL_RUNS {
        let cut_box = _mm512_set1_pd(cut2 + 2.0 * margin);
        let zero = std::arch::x86_64::_mm512_setzero_pd();
        let mut s = block.i_lo;
        while s + 4 <= block.i_hi {
            let s0 = source!(s);
            let s1 = source!(s + 1);
            let s2 = source!(s + 2);
            let s3 = source!(s + 3);
            near_runs!(c, block, nvec, ox, oy, oz, cut_box, zero, s, slot, {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
                lanes!(s + 1, slot, tm, s1, jx, jy, jz, jr, jids);
                lanes!(s + 2, slot, tm, s2, jx, jy, jz, jr, jids);
                lanes!(s + 3, slot, tm, s3, jx, jy, jz, jr, jids);
            });
            s += 4;
        }
        while s < block.i_hi {
            let s0 = source!(s);
            near_runs!(c, block, nvec, ox, oy, oz, cut_box, zero, s, slot, {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
            });
            s += 1;
        }
    } else if !block.tri {
        let mut s = block.i_lo;
        while s + 4 <= block.i_hi {
            let s0 = source!(s);
            let s1 = source!(s + 1);
            let s2 = source!(s + 2);
            let s3 = source!(s + 3);
            let mut slot = block.j_lo;
            while slot < j_hi {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
                lanes!(s + 1, slot, tm, s1, jx, jy, jz, jr, jids);
                lanes!(s + 2, slot, tm, s2, jx, jy, jz, jr, jids);
                lanes!(s + 3, slot, tm, s3, jx, jy, jz, jr, jids);
                slot += 8;
            }
            s += 4;
        }
        while s < block.i_hi {
            let s0 = source!(s);
            let mut slot = block.j_lo;
            while slot < j_hi {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
                slot += 8;
            }
            s += 1;
        }
    } else {
        for s in block.i_lo..block.i_hi {
            let s0 = source!(s);
            let mut slot = s + 1;
            while slot < j_hi {
                let (tm, jx, jy, jz, jr, jids) = load!(slot, j_hi);
                lanes!(s, slot, tm, s0, jx, jy, jz, jr, jids);
                slot += 8;
            }
        }
    }
    scratch.finish(n);
}

/// # Safety
/// `block` indexes `xs`, `ys`, `zs`, and `ids`. `scratch` holds every pair in the block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline(never)]
unsafe fn avx_scan(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    ids: &[u32],
    cut2: f64,
    block: &Block,
    scratch: &mut Scratch,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_cmp_pd, _mm256_loadu_pd, _mm256_movemask_pd, _mm256_mul_pd,
        _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd, _CMP_LT_OQ,
    };
    let xp = xs.as_ptr();
    let yp = ys.as_ptr();
    let zp = zs.as_ptr();
    let idp = ids.as_ptr();
    let atom_p = scratch.atom.as_mut_ptr();
    let js_p = scratch.js.as_mut_ptr();
    let d2_p = scratch.d2.as_mut_ptr();
    let mut n = scratch.js.len();
    let sx = block.shift[0];
    let sy = block.shift[1];
    let sz = block.shift[2];
    let skip_self = block.shift_s == [0, 0, 0];
    let cutv = _mm256_set1_pd(cut2);

    macro_rules! take4 {
        ($bits:expr, $d2v:expr, $slot:expr, $iu:expr, $iatom:expr) => {{
            if $bits != 0 {
                let mut lane = [0.0f64; 4];
                _mm256_storeu_pd(lane.as_mut_ptr(), $d2v);
                let iatom = $iatom;
                if $bits & 1 != 0 {
                    let j = *idp.add($slot);
                    if !(skip_self && j == iatom) {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(j);
                        d2_p.add(n).write(lane[0]);
                        n += 1;
                    }
                }
                if $bits & 2 != 0 {
                    let j = *idp.add($slot + 1);
                    if !(skip_self && j == iatom) {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(j);
                        d2_p.add(n).write(lane[1]);
                        n += 1;
                    }
                }
                if $bits & 4 != 0 {
                    let j = *idp.add($slot + 2);
                    if !(skip_self && j == iatom) {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(j);
                        d2_p.add(n).write(lane[2]);
                        n += 1;
                    }
                }
                if $bits & 8 != 0 {
                    let j = *idp.add($slot + 3);
                    if !(skip_self && j == iatom) {
                        atom_p.add(n).write($iu);
                        js_p.add(n).write(j);
                        d2_p.add(n).write(lane[3]);
                        n += 1;
                    }
                }
            }
        }};
    }
    macro_rules! dist4 {
        ($jx:expr, $jy:expr, $jz:expr, $bx:expr, $by:expr, $bz:expr) => {{
            let dx = _mm256_sub_pd($jx, $bx);
            let dy = _mm256_sub_pd($jy, $by);
            let dz = _mm256_sub_pd($jz, $bz);
            _mm256_add_pd(
                _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                _mm256_mul_pd(dz, dz),
            )
        }};
    }
    macro_rules! one_src {
        ($s:expr, $j_lo:expr) => {{
            let s = $s;
            let j_lo = $j_lo;
            if j_lo < block.j_hi {
                let i = *idp.add(s);
                let iu = i;
                let px = *xp.add(s) - sx;
                let py = *yp.add(s) - sy;
                let pz = *zp.add(s) - sz;
                let bx = _mm256_set1_pd(px);
                let by = _mm256_set1_pd(py);
                let bz = _mm256_set1_pd(pz);
                let span = block.j_hi - j_lo;
                let end = j_lo + (span & !3);
                let mut slot = j_lo;
                while slot < end {
                    let jx = _mm256_loadu_pd(xp.add(slot));
                    let jy = _mm256_loadu_pd(yp.add(slot));
                    let jz = _mm256_loadu_pd(zp.add(slot));
                    let d2v = dist4!(jx, jy, jz, bx, by, bz);
                    let bits = _mm256_movemask_pd(_mm256_cmp_pd(d2v, cutv, _CMP_LT_OQ));
                    take4!(bits, d2v, slot, iu, i);
                    slot += 4;
                }
                for slot in end..block.j_hi {
                    let j = *idp.add(slot);
                    if skip_self && j == i {
                        continue;
                    }
                    let dx = *xp.add(slot) - px;
                    let dy = *yp.add(slot) - py;
                    let dz = *zp.add(slot) - pz;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < cut2 {
                        atom_p.add(n).write(iu);
                        js_p.add(n).write(j);
                        d2_p.add(n).write(d2);
                        n += 1;
                    }
                }
            }
        }};
    }

    if !block.tri {
        let j_lo = block.j_lo;
        let j_hi = block.j_hi;
        let end = j_lo + ((j_hi - j_lo) & !3);
        let mut s = block.i_lo;
        while s + 2 <= block.i_hi {
            let i0 = *idp.add(s);
            let i1 = *idp.add(s + 1);
            let iu0 = i0;
            let iu1 = i1;
            let p0x = *xp.add(s) - sx;
            let p0y = *yp.add(s) - sy;
            let p0z = *zp.add(s) - sz;
            let p1x = *xp.add(s + 1) - sx;
            let p1y = *yp.add(s + 1) - sy;
            let p1z = *zp.add(s + 1) - sz;
            let b0x = _mm256_set1_pd(p0x);
            let b0y = _mm256_set1_pd(p0y);
            let b0z = _mm256_set1_pd(p0z);
            let b1x = _mm256_set1_pd(p1x);
            let b1y = _mm256_set1_pd(p1y);
            let b1z = _mm256_set1_pd(p1z);
            let mut slot = j_lo;
            while slot < end {
                let jx = _mm256_loadu_pd(xp.add(slot));
                let jy = _mm256_loadu_pd(yp.add(slot));
                let jz = _mm256_loadu_pd(zp.add(slot));
                let d0 = dist4!(jx, jy, jz, b0x, b0y, b0z);
                let d1 = dist4!(jx, jy, jz, b1x, b1y, b1z);
                take4!(
                    _mm256_movemask_pd(_mm256_cmp_pd(d0, cutv, _CMP_LT_OQ)),
                    d0,
                    slot,
                    iu0,
                    i0
                );
                take4!(
                    _mm256_movemask_pd(_mm256_cmp_pd(d1, cutv, _CMP_LT_OQ)),
                    d1,
                    slot,
                    iu1,
                    i1
                );
                slot += 4;
            }
            for slot in end..j_hi {
                let j = *idp.add(slot);
                let jx = *xp.add(slot);
                let jy = *yp.add(slot);
                let jz = *zp.add(slot);
                if !(skip_self && j == i0) {
                    let dx = jx - p0x;
                    let dy = jy - p0y;
                    let dz = jz - p0z;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < cut2 {
                        atom_p.add(n).write(iu0);
                        js_p.add(n).write(j);
                        d2_p.add(n).write(d2);
                        n += 1;
                    }
                }
                if !(skip_self && j == i1) {
                    let dx = jx - p1x;
                    let dy = jy - p1y;
                    let dz = jz - p1z;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < cut2 {
                        atom_p.add(n).write(iu1);
                        js_p.add(n).write(j);
                        d2_p.add(n).write(d2);
                        n += 1;
                    }
                }
            }
            s += 2;
        }
        if s < block.i_hi {
            one_src!(s, j_lo);
        }
    } else {
        for s in block.i_lo..block.i_hi {
            one_src!(s, s + 1);
        }
    }
    scratch.finish(n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cell;

    #[test]
    fn tight_cluster_matches_brute_force() {
        // Every pair sits inside the cutoff, and the box is large, so the
        // ideal-gas row estimate is a handful. The walk still has to
        // return the full list. 520 is above the parallel split.
        let n = 520usize;
        let mut xyz = vec![[0.0f64; 3]; n];
        let g = 9i32;
        let mut k = 0usize;
        'fill: for z in 0..g {
            for y in 0..g {
                for x in 0..g {
                    if k >= n {
                        break 'fill;
                    }
                    xyz[k] = [x as f64 * 0.05, y as f64 * 0.05, z as f64 * 0.05];
                    k += 1;
                }
            }
        }
        let sim = Cell::ortho(80.0, 80.0, 80.0).unwrap();
        let cutoff = 2.0;
        let cut2 = cutoff * cutoff;
        let mut brute: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = xyz[j][0] - xyz[i][0];
                let dy = xyz[j][1] - xyz[i][1];
                let dz = xyz[j][2] - xyz[i][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < cut2 {
                    brute.push((i, j, d2));
                    brute.push((j, i, d2));
                }
            }
        }
        brute.sort_by_key(|a| (a.0, a.1));
        let run = || pairs_within(&xyz, &sim, cutoff, None, None, false).unwrap();
        #[cfg(feature = "parallel")]
        let got = {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap();
            pool.install(run)
        };
        #[cfg(not(feature = "parallel"))]
        let got = run();
        let mut rows: Vec<(usize, usize, f64)> = got.iter().map(|p| (p.i, p.j, p.dist2)).collect();
        rows.sort_by_key(|a| (a.0, a.1));
        assert_eq!(rows.len(), n * (n - 1));
        assert_eq!(rows.len(), brute.len());
        for (got_row, brute_row) in rows.iter().zip(brute.iter()) {
            assert_eq!(got_row.0, brute_row.0);
            assert_eq!(got_row.1, brute_row.1);
            assert!((got_row.2 - brute_row.2).abs() < 1e-9);
        }
    }

    /// Run `f` on a pool of `threads` workers.
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

    fn row_keys(rows: &[Pair]) -> Vec<(usize, usize, [i32; 3], u64)> {
        let mut k: Vec<_> = rows
            .iter()
            .map(|p| (p.i, p.j, p.shift, p.dist2.to_bits()))
            .collect();
        k.sort();
        k
    }

    #[test]
    fn one_thread_and_eight_write_the_same_rows() {
        // 1000 jittered points: the one-thread call writes rows straight
        // from the tile kernel, the eight-thread call buffers hits first.
        let sim = Cell::ortho(18.0, 17.0, 19.0).unwrap();
        let mut xyz = Vec::new();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for iz in 0..10 {
            for iy in 0..10 {
                for ix in 0..10 {
                    xyz.push([
                        (ix as f64 + 0.3 * next()) * 1.8,
                        (iy as f64 + 0.3 * next()) * 1.7,
                        (iz as f64 + 0.3 * next()) * 1.9,
                    ]);
                }
            }
        }
        for half in [false, true] {
            let one = on_threads(1, || {
                pairs_within(&xyz, &sim, 4.0, None, None, half).unwrap()
            });
            let eight = on_threads(8, || {
                pairs_within(&xyz, &sim, 4.0, None, None, half).unwrap()
            });
            assert_eq!(row_keys(&one), row_keys(&eight), "half={half}");
            let mut c1 = PairColumns::default();
            let mut c8 = PairColumns::default();
            on_threads(1, || {
                pairs_within_columns(&xyz, &sim, 4.0, None, None, half, &mut c1).unwrap()
            });
            on_threads(8, || {
                pairs_within_columns(&xyz, &sim, 4.0, None, None, half, &mut c8).unwrap()
            });
            let cols = |c: &PairColumns| {
                let mut k: Vec<_> = (0..c.len())
                    .map(|t| {
                        (
                            c.i[t] as usize,
                            c.j[t] as usize,
                            c.shift[t],
                            c.dist2[t].to_bits(),
                        )
                    })
                    .collect();
                k.sort();
                k
            };
            assert_eq!(cols(&c1), row_keys(&one), "columns half={half}");
            assert_eq!(cols(&c8), row_keys(&one), "columns eight half={half}");
        }
    }

    #[test]
    fn sub_cells_keep_the_same_rows_in_the_same_order_on_eight_threads() {
        // 64 atoms per bin, so each bin is sorted by 4x4x4 sub-cells, and
        // above the threaded-bins threshold, so eight threads sort them in
        // a third pass. Atom order is shuffled, so it does not follow position.
        let sim = Cell::ortho(18.0, 18.0, 18.0).unwrap();
        let mut state = 0x5851_f42d_4c95_7f2du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut xyz = Vec::new();
        for iz in 0..16 {
            for iy in 0..16 {
                for ix in 0..16 {
                    let j = |v: u64| 0.3 * ((v >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
                    xyz.push([
                        (ix as f64 + 0.5 + j(next())) * 1.125,
                        (iy as f64 + 0.5 + j(next())) * 1.125,
                        (iz as f64 + 0.5 + j(next())) * 1.125,
                    ]);
                }
            }
        }
        for i in (1..xyz.len()).rev() {
            let k = (next() % (i as u64 + 1)) as usize;
            xyz.swap(i, k);
        }
        assert!(xyz.len() >= PARALLEL_GRID);
        for half in [false, true] {
            let one = on_threads(1, || {
                pairs_within(&xyz, &sim, 4.0, None, None, half).unwrap()
            });
            let eight = on_threads(8, || {
                pairs_within(&xyz, &sim, 4.0, None, None, half).unwrap()
            });
            assert_eq!(one.len(), eight.len(), "half={half}");
            assert!(one == eight, "half={half}: rows differ in content or order");
        }
    }

    #[test]
    fn box_culling_keeps_every_row_in_order() {
        // 1100 random points in boxes three bins a side at a 3 Å cutoff:
        // about 40 atoms per bin, so the tile kernel tests run boxes, on
        // the threaded bins too. Then a 1 Å cubic lattice with the cutoff
        // on its sqrt(11) shell and every coordinate a few ulps off, so
        // thousands of pairs sit within the margin of the cutoff.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut unit = || (next() >> 11) as f64 / (1u64 << 53) as f64;
        let mut systems = Vec::new();
        for sim in [
            Cell::ortho(9.3, 9.6, 9.9).unwrap(),
            Cell::from_vectors(
                [9.6, 0.0, 0.0],
                [1.2, 9.5, 0.0],
                [-0.8, 0.9, 9.7],
                [0.5, -0.3, 0.2],
            )
            .unwrap(),
        ] {
            let xyz: Vec<[f64; 3]> = (0..1100)
                .map(|_| sim.cartesian([1.2 * unit() - 0.1, unit(), 1.1 * unit() - 0.05]))
                .collect();
            systems.push((sim, xyz, 3.0));
        }
        let mut lattice = Vec::new();
        for iz in 0..10 {
            for iy in 0..10 {
                for ix in 0..10 {
                    let mut p = [ix as f64 + 0.5, iy as f64 + 0.5, iz as f64 + 0.5];
                    for v in p.iter_mut() {
                        let ulps = (next() % 13) as i64 - 6;
                        *v = f64::from_bits((v.to_bits() as i64 + ulps) as u64);
                    }
                    lattice.push(p);
                }
            }
        }
        let shell = 11.0f64.sqrt();
        systems.push((Cell::ortho(10.0, 10.0, 10.0).unwrap(), lattice, shell));
        let before = CULL_SKIPS.load(std::sync::atomic::Ordering::Relaxed);
        for (sim, xyz, cutoff) in &systems {
            let (sim, cutoff) = (sim, *cutoff);
            let mask: Vec<bool> = (0..xyz.len()).map(|k| k % 9 != 4).collect();
            if cutoff == shell {
                let rows = pairs_within(xyz, sim, cutoff, None, None, false).unwrap();
                let near = rows
                    .iter()
                    .filter(|p| (p.dist2 - cutoff * cutoff).abs() < 1e-12)
                    .count();
                assert!(near > 1000, "only {near} rows near the cutoff");
            }
            for threads in [1, 8] {
                for half in [false, true] {
                    for mask in [None, Some(mask.as_slice())] {
                        let run = |cull: bool| {
                            NO_CULL.store(!cull, std::sync::atomic::Ordering::Relaxed);
                            let (rows, cols) = on_threads(threads, || {
                                let rows =
                                    pairs_within(xyz, sim, cutoff, mask, None, half).unwrap();
                                let mut cols = PairColumns::default();
                                pairs_within_columns(xyz, sim, cutoff, mask, None, half, &mut cols)
                                    .unwrap();
                                (rows, cols)
                            });
                            NO_CULL.store(false, std::sync::atomic::Ordering::Relaxed);
                            (rows, cols)
                        };
                        let (culled, culled_cols) = run(true);
                        let (plain, plain_cols) = run(false);
                        let what = format!("threads={threads} half={half} mask={}", mask.is_some());
                        assert!(culled.len() > 1000, "{what}");
                        assert!(culled == plain, "{what}: rows differ in content or order");
                        assert_eq!(culled_cols.i, plain_cols.i, "{what}");
                        assert_eq!(culled_cols.j, plain_cols.j, "{what}");
                        assert_eq!(culled_cols.shift, plain_cols.shift, "{what}");
                        let bits = |c: &PairColumns| -> Vec<u64> {
                            c.dist2.iter().map(|d| d.to_bits()).collect()
                        };
                        assert_eq!(bits(&culled_cols), bits(&plain_cols), "{what}");
                    }
                }
            }
        }
        if cfg!(all(target_arch = "x86_64", linkcell_avx512)) && simd_mode() == 2 {
            let skipped = CULL_SKIPS.load(std::sync::atomic::Ordering::Relaxed) - before;
            assert!(skipped > 0, "no target run was skipped");
        }
    }

    #[test]
    fn vector_fold_matches_fold_point_bit_for_bit() {
        let cells = [
            Cell::ortho(18.0, 17.0, 19.0).unwrap(),
            Cell::from_vectors(
                [10.0, 0.0, 0.0],
                [5.0, 8.660254037844386, 0.0],
                [1.0, -2.0, 9.5],
                [0.3, -1.0, 2.0],
            )
            .unwrap(),
            Cell::from_vectors(
                [9.0, 3.0, 2.0],
                [1.0, 9.0, -2.0],
                [-1.5, 2.0, 9.5],
                [0.0; 3],
            )
            .unwrap(),
        ];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for sim in &cells {
            let mut xyz: Vec<[f64; 3]> = (0..203)
                .map(|_| {
                    [
                        (next() - 0.3) * 60.0,
                        (next() - 0.7) * 45.0,
                        (next() - 0.5) * 80.0,
                    ]
                })
                .collect();
            let w = sim.widths();
            xyz.extend_from_slice(&[
                [0.0; 3],
                [w[0], w[1], w[2]],
                [-0.0, -1e-300, 1e-300],
                [w[0] * 0.5, -w[1], 3.0 * w[2]],
            ]);
            let n = [5, 4, 7];
            for sub in [1, 2, 4] {
                let keys = Keys { n, sub };
                let nkey = 140 * keys.per_bin();
                let mut fold = FoldCols::default();
                fold.clear_for(xyz.len());
                let fp = fold.ptrs();
                let mut count = vec![0u32; nkey];
                fold_range(sim, &xyz, None, keys, false, 0, xyz.len(), fp, &mut count);
                let mut want_count = vec![0u32; nkey];
                for (k, r) in xyz.iter().enumerate() {
                    let (q, b) = bins::fold_point(sim, *r, keys.fine());
                    let key = keys.key(b);
                    want_count[key] += 1;
                    // Safety: the fold wrote every atom.
                    let (p, got) = unsafe { fp.get(k) };
                    assert_eq!(got, key, "atom {k} sub {sub}");
                    for (a, (got, want)) in p.iter().zip(q).enumerate() {
                        assert_eq!(got.to_bits(), want.to_bits(), "atom {k} axis {a}");
                    }
                }
                assert_eq!(count, want_count);
            }
        }
    }

    #[test]
    fn dense_cluster_on_several_threads_matches_one() {
        // The ideal-gas estimate sees 4520 atoms in a 20 Å box and splits
        // the walk; 520 of them sit inside 0.4 Å, so their bin holds far
        // more pairs than the estimate.
        let sim = Cell::ortho(20.0, 20.0, 20.0).unwrap();
        let mut xyz = Vec::new();
        'fill: for z in 0..9 {
            for y in 0..9 {
                for x in 0..9 {
                    if xyz.len() == 520 {
                        break 'fill;
                    }
                    xyz.push([
                        10.0 + x as f64 * 0.05,
                        10.0 + y as f64 * 0.05,
                        10.0 + z as f64 * 0.05,
                    ]);
                }
            }
        }
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    if xyz.len() == 4520 {
                        break;
                    }
                    xyz.push([
                        x as f64 * 1.25 + 0.31,
                        y as f64 * 1.25 + 0.17,
                        z as f64 * 1.25 + 0.53,
                    ]);
                }
            }
        }
        #[cfg(feature = "parallel")]
        assert!(hit_estimate(xyz.len(), &sim, 2.0) >= PARALLEL_PAIRS);
        let one = on_threads(1, || {
            pairs_within(&xyz, &sim, 2.0, None, None, false).unwrap()
        });
        let eight = on_threads(8, || {
            pairs_within(&xyz, &sim, 2.0, None, None, false).unwrap()
        });
        assert_eq!(row_keys(&one), row_keys(&eight));
        let cluster = eight.iter().filter(|p| p.i < 520 && p.j < 520).count();
        assert_eq!(cluster, 520 * 519);
    }

    #[test]
    fn one_thread_matches_the_shift_scan() {
        let sim = Cell::ortho(18.0, 18.0, 18.0).unwrap();
        let mut xyz = Vec::new();
        for iz in 0..9 {
            for iy in 0..9 {
                for ix in 0..9 {
                    xyz.push([
                        (ix as f64 + 0.5) * 2.0,
                        (iy as f64 + 0.5) * 2.0,
                        (iz as f64 + 0.5) * 2.0,
                    ]);
                }
            }
        }
        let cutoff = 4.0;
        let got = on_threads(1, || {
            pairs_within(&xyz, &sim, cutoff, None, None, false).unwrap()
        });
        let mut keys: Vec<_> = got.iter().map(|p| (p.i, p.j, p.shift)).collect();
        keys.sort();
        let mut want = Vec::new();
        for i in 0..xyz.len() {
            for j in 0..xyz.len() {
                for na in -1..=1 {
                    for nb in -1..=1 {
                        for nc in -1..=1 {
                            if i == j && na == 0 && nb == 0 && nc == 0 {
                                continue;
                            }
                            let d2 =
                                sim.dist2_shifted(xyz[i], xyz[j], sim.lattice_shift(na, nb, nc));
                            if d2 < cutoff * cutoff {
                                want.push((i, j, [na, nb, nc]));
                            }
                        }
                    }
                }
            }
        }
        want.sort();
        assert_eq!(keys, want);
        for p in &got {
            let d2 = sim.dist2_shifted(
                xyz[p.i],
                xyz[p.j],
                sim.lattice_shift(p.shift[0], p.shift[1], p.shift[2]),
            );
            assert!((p.dist2 - d2).abs() < 1e-9);
        }
    }

    #[test]
    fn columns_hold_the_same_rows() {
        let sim = Cell::ortho(18.0, 18.0, 18.0).unwrap();
        let mut xyz = Vec::new();
        for iz in 0..9 {
            for iy in 0..9 {
                for ix in 0..9 {
                    xyz.push([
                        ix as f64 * 2.0 + 0.3,
                        iy as f64 * 2.0 + 0.1,
                        iz as f64 * 2.0 + 0.7,
                    ]);
                }
            }
        }
        let mut cols = PairColumns::default();
        for half in [false, true] {
            let rows = pairs_within(&xyz, &sim, 4.0, None, None, half).unwrap();
            pairs_within_columns(&xyz, &sim, 4.0, None, None, half, &mut cols).unwrap();
            assert_eq!(cols.len(), rows.len());
            let mut a: Vec<_> = rows
                .iter()
                .map(|p| (p.i as i32, p.j as i32, p.shift, p.dist2.to_bits()))
                .collect();
            let mut b: Vec<_> = (0..cols.len())
                .map(|t| (cols.i[t], cols.j[t], cols.shift[t], cols.dist2[t].to_bits()))
                .collect();
            a.sort();
            b.sort();
            assert_eq!(a, b, "half={half}");
        }
    }

    #[test]
    fn sparse_bins_write_the_same_rows_on_one_thread_and_eight() {
        // Under eight atoms a bin, one thread runs the tile inlined into its
        // block loop, for rows and for columns; eight threads buffer hits.
        // 3000 atoms in a 40 A cell: about three a bin, and enough pairs
        // that eight threads split the walk.
        let cells = [
            Cell::ortho(40.0, 39.0, 41.0).unwrap(),
            Cell::from_vectors(
                [40.0, 0.0, 0.0],
                [6.0, 39.0, 0.0],
                [-4.0, 3.0, 40.5],
                [0.3, 0.0, -0.2],
            )
            .unwrap(),
        ];
        let mut state = 0x6a09_e667_f3bc_c908u64;
        let mut unit = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for sim in &cells {
            let xyz: Vec<[f64; 3]> = (0..3000)
                .map(|_| sim.cartesian([unit(), unit(), unit()]))
                .collect();
            assert!(
                walk_threads(hit_estimate(xyz.len(), sim, 4.0)) > 1 || !cfg!(feature = "parallel")
            );
            for half in [false, true] {
                let (one, one_cols) = on_threads(1, || {
                    let rows = pairs_within(&xyz, sim, 4.0, None, None, half).unwrap();
                    let mut cols = PairColumns::default();
                    pairs_within_columns(&xyz, sim, 4.0, None, None, half, &mut cols).unwrap();
                    (rows, cols)
                });
                let eight = on_threads(8, || {
                    pairs_within(&xyz, sim, 4.0, None, None, half).unwrap()
                });
                assert!(one.len() > 10_000, "half={half}");
                assert!(one == eight, "half={half}: rows differ in content or order");
                assert_eq!(one_cols.len(), one.len());
                for (t, p) in one.iter().enumerate() {
                    assert_eq!(
                        (
                            one_cols.i[t],
                            one_cols.j[t],
                            one_cols.shift[t],
                            one_cols.dist2[t].to_bits()
                        ),
                        (p.i as i32, p.j as i32, p.shift, p.dist2.to_bits()),
                        "half={half} row {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn far_image_is_not_rewrapped() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]];
        let pairs = pairs_within(&xyz, &sim, 3.0, None, Some(10.0), false).unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().all(|p| p.shift == [0, 0, 0]));
        assert!(pairs.iter().all(|p| (p.dist2 - 0.01).abs() < 1e-12));
    }

    #[test]
    fn highway_mic_matches_wrap_pair_distance() {
        let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
        let pairs = pairs_within(&xyz, &sim, 1.0, None, None, true).unwrap();
        let mut mic = [0.0];
        crate::dist2_ortho_diffs(
            &[xyz[1][0] - xyz[0][0]],
            &[xyz[1][1] - xyz[0][1]],
            &[xyz[1][2] - xyz[0][2]],
            10.0,
            10.0,
            10.0,
            &mut mic,
        )
        .unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].shift, [-1, 0, 0]);
        assert!((pairs[0].dist2 - mic[0]).abs() < 1e-12);
        assert!((mic[0] - 0.64).abs() < 1e-12);
    }

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
    fn wide_cube_matches_shift_scan() {
        let sim = Cell::ortho(18.0, 18.0, 18.0).unwrap();
        let mut xyz = Vec::new();
        for iz in 0..8 {
            for iy in 0..8 {
                for ix in 0..8 {
                    xyz.push([
                        (ix as f64 + 0.5) * 18.0 / 8.0,
                        (iy as f64 + 0.5) * 18.0 / 8.0,
                        (iz as f64 + 0.5) * 18.0 / 8.0,
                    ]);
                }
            }
        }
        let cutoff = 4.0;
        let cut2 = cutoff * cutoff;
        for half in [false, true] {
            let got = pairs_within(&xyz, &sim, cutoff, None, None, half).unwrap();
            let mut want = Vec::new();
            for i in 0..xyz.len() {
                for j in 0..xyz.len() {
                    for na in -1..=1 {
                        for nb in -1..=1 {
                            for nc in -1..=1 {
                                if i == j && na == 0 && nb == 0 && nc == 0 {
                                    continue;
                                }
                                let shift = [na, nb, nc];
                                let d2 = sim.dist2_shifted(
                                    xyz[i],
                                    xyz[j],
                                    sim.lattice_shift(na, nb, nc),
                                );
                                if d2 < cut2 && (!half || keep_half(i, j, shift)) {
                                    want.push((i, j, shift));
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
    }

    #[test]
    fn dense_cube_matches_shift_scan() {
        let sim = Cell::ortho(6.0, 6.0, 6.0).unwrap();
        let mut xyz = Vec::new();
        for iz in 0..5 {
            for iy in 0..5 {
                for ix in 0..5 {
                    xyz.push([
                        (ix as f64 + 0.5) * 1.2,
                        (iy as f64 + 0.5) * 1.2,
                        (iz as f64 + 0.5) * 1.2,
                    ]);
                }
            }
        }
        let cutoff = 2.0;
        let got = pairs_within(&xyz, &sim, cutoff, None, None, false).unwrap();
        let mut keys: Vec<_> = got.iter().map(|p| (p.i, p.j, p.shift)).collect();
        keys.sort();
        let cut2 = cutoff * cutoff;
        let mut want = Vec::new();
        for i in 0..xyz.len() {
            for j in 0..xyz.len() {
                for na in -1..=1 {
                    for nb in -1..=1 {
                        for nc in -1..=1 {
                            if i == j && na == 0 && nb == 0 && nc == 0 {
                                continue;
                            }
                            let shift = [na, nb, nc];
                            let pi = sim.cartesian(sim.fractional(xyz[i]));
                            let pj = sim.cartesian(sim.fractional(xyz[j]));
                            let d2 = sim.dist2_shifted(pi, pj, sim.lattice_shift(na, nb, nc));
                            if d2 < cut2 {
                                want.push((i, j, shift));
                            }
                        }
                    }
                }
            }
        }
        want.sort();
        assert_eq!(keys, want);
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
