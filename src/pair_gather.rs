//! Cutoff rows of a periodic cell, gathered source by source.
//!
//! The pair walk tests each unordered pair once and leaves the hits for
//! a writer; a CSR row then has to collect its entries from every hit
//! buffer. Here every active atom instead gathers its whole row from
//! the bins around its own, so a row is complete the moment it is
//! staged; [`finish_row`] sorts it when its atoms are not numbered in
//! bin order and keeps the nearest image when a column can repeat. One
//! copy then lays the staged rows out in atom order. No hit is buffered and no row is
//! counted, placed, or filled from another thread's hits.
//!
//! The bins, the folded positions, and the reach are the walk's
//! ([`bin_atoms`](super::bin_atoms)), and so is every stored distance:
//! the walk sees a pair from the bin `keep_dir` picks, as
//! `q - (p - S)` with `p` the source. A row whose atom is the walk's
//! target computes that same expression, with the roles the walk gave
//! them. With no shift the two roles square to the same bits, so the
//! bins around an inner bin are nine runs of three bins, read with one
//! expression.

use super::{bins_too_far, keep_dir, Binned, Coords, Grid};
use crate::bins;
use crate::cell::Cell;
#[cfg(target_arch = "x86_64")]
use crate::simd::Path;
use crate::Error;

/// One thread's staged rows, in slot order.
#[derive(Default)]
struct Stage {
    cols: Vec<i32>,
    d2: Vec<f64>,
    /// Runs of the bin being walked that its rows read in place.
    runs: Vec<Run>,
    /// The same, as offsets from that bin.
    steps: Vec<Step>,
    /// The bin's other candidates, packed.
    pack: Pack,
    /// Per axis, the fold of each offset of that bin.
    axes: [Vec<(i32, i32)>; 3],
}

/// Buffers one gather keeps for the next.
#[derive(Default)]
pub(super) struct RowBufs {
    stages: Vec<Stage>,
    /// Per atom, its stage and its row's start there: `stage << 40 | start`.
    at: Vec<u64>,
    /// Per atom, its row length; after [`Gathered::scan`], the CSR row
    /// starts, `n + 1` values.
    first: Vec<u32>,
}

static ROW_BUFS: std::sync::Mutex<Option<RowBufs>> = std::sync::Mutex::new(None);

/// Values a buffer may hold and still be kept for the next call.
const MAX_KEEP: usize = 1 << 26;

impl RowBufs {
    fn take() -> Self {
        ROW_BUFS
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .unwrap_or_default()
    }

    fn give(self) {
        let big = self.at.capacity() > MAX_KEEP
            || self.stages.iter().any(|s| s.cols.capacity() > MAX_KEEP);
        if big {
            return;
        }
        if let Ok(mut slot) = ROW_BUFS.lock() {
            if slot.is_none() {
                *slot = Some(self);
            }
        }
    }
}

const STAGE_BITS: u32 = 40;

/// Slots `lo..hi` of the grid, and how a row reads them.
#[derive(Clone, Copy)]
struct Run {
    lo: usize,
    hi: usize,
    kind: Kind,
    shift: [f64; 3],
}

/// How a row reads a run.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    /// No shift: `q - p`, which squares to the same bits in either role.
    Plain,
    /// The row's atom is the walk's source: `q - (p - shift)`.
    Source,
    /// The row's atom is the walk's target: `p - (q - shift)`.
    Target,
}

/// A run of bins at a fixed flat offset from the bin being walked.
#[derive(Clone, Copy)]
pub(super) struct Step {
    pub(super) rel: isize,
    pub(super) bins: usize,
    pub(super) kind: Kind,
    /// The shift, in cells.
    pub(super) s: [i32; 3],
    pub(super) shift: [f64; 3],
}

/// A buffer several jobs write at disjoint slots.
struct Shared<T>(*mut T);
impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Shared<T> {}
// Safety: jobs write distinct slots.
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}
impl<T> Shared<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// Lattice shifts of the bin offsets a walk can take, by shift index.
pub(super) struct Shifts {
    lo: [i32; 3],
    len: [i32; 3],
    table: Vec<[f64; 3]>,
}

/// Table entries past which a shift is computed where it is used.
const SHIFT_TABLE: i64 = 1 << 12;

impl Shifts {
    fn new(cell: &Cell, n: [i32; 3], reach: [i32; 3]) -> Self {
        let mut lo = [0i32; 3];
        let mut len = [0i32; 3];
        let mut size = 1i64;
        for a in 0..3 {
            lo[a] = (-reach[a]).div_euclid(n[a]);
            let hi = (n[a] - 1 + reach[a]).div_euclid(n[a]);
            len[a] = hi - lo[a] + 1;
            size = size.saturating_mul(i64::from(len[a]));
        }
        let mut table = Vec::new();
        if size <= SHIFT_TABLE {
            for c in 0..len[2] {
                for b in 0..len[1] {
                    for a in 0..len[0] {
                        table.push(cell.lattice_shift(lo[0] + a, lo[1] + b, lo[2] + c));
                    }
                }
            }
        }
        Shifts { lo, len, table }
    }

    #[inline]
    fn get(&self, cell: &Cell, s: [i32; 3]) -> [f64; 3] {
        if self.table.is_empty() {
            return cell.lattice_shift(s[0], s[1], s[2]);
        }
        let k = (((s[2] - self.lo[2]) * self.len[1] + (s[1] - self.lo[1])) * self.len[0]
            + (s[0] - self.lo[0])) as usize;
        self.table[k]
    }
}

/// What the runs of a bin depend on besides the bin: the bin counts, the
/// reach, and the cell.
pub(super) struct Reach<'a> {
    pub(super) n: [i32; 3],
    pub(super) reach: [i32; 3],
    pub(super) ortho: bool,
    /// Bin widths, as the stencil takes them.
    pub(super) width: [f64; 3],
    pub(super) cut2: f64,
    pub(super) shifts: &'a Shifts,
    pub(super) cell: &'a Cell,
}

/// Runs of bin `c` at `b`, as flat offsets: every bin offset in reach,
/// the walk's roles and shifts, less the bins the stencil drops. Runs of
/// neighbouring bins read the same way merge. `axes` is scratch.
pub(super) fn bin_steps(
    at: &Reach<'_>,
    c: usize,
    b: [i32; 3],
    axes: &mut [Vec<(i32, i32)>; 3],
    steps: &mut Vec<Step>,
) -> Result<(), Error> {
    let n = at.n;
    let r = at.reach;
    // Per axis and offset `d`: the bin `b + d` folds into and its
    // shift. The walk sees an offset `keep_dir` drops from that bin,
    // at `-d`, with the opposite shift.
    for a in 0..3 {
        let axis = &mut axes[a];
        axis.clear();
        axis.extend((-r[a]..=r[a]).map(|d| fold(b[a] + d, n[a])));
    }
    steps.clear();
    for dz in -r[2]..=r[2] {
        let (wz, sz) = axes[2][(dz + r[2]) as usize];
        for dy in -r[1]..=r[1] {
            let (wy, sy) = axes[1][(dy + r[1]) as usize];
            for dx in -r[0]..=r[0] {
                let (wx, sx) = axes[0][(dx + r[0]) as usize];
                let d = [dx, dy, dz];
                let other = [wx, wy, wz];
                let source = d == [0, 0, 0] || keep_dir(dx, dy, dz);
                // Neighbouring bins touch, so only a farther one can be
                // out of reach.
                if dx.abs() > 1 || dy.abs() > 1 || dz.abs() > 1 {
                    let (src, dst) = if source {
                        (b, [b[0] + dx, b[1] + dy, b[2] + dz])
                    } else {
                        (other, [wx - dx, wy - dy, wz - dz])
                    };
                    if bins_too_far(at.ortho, src, dst, at.width, at.cut2) {
                        continue;
                    }
                }
                let rel = bins::flat_cell(other, n) as isize - c as isize;
                let s = if source {
                    [sx, sy, sz]
                } else {
                    [-sx, -sy, -sz]
                };
                let kind = if s == [0, 0, 0] {
                    Kind::Plain
                } else if source {
                    Kind::Source
                } else {
                    Kind::Target
                };
                if let Some(last) = steps.last_mut() {
                    if last.rel + last.bins as isize == rel && last.kind == kind && last.s == s {
                        last.bins += 1;
                        continue;
                    }
                }
                if steps.len() == steps.capacity() {
                    let extra = steps.capacity().max(64);
                    super::try_hold(steps, extra)?;
                }
                steps.push(Step {
                    rel,
                    bins: 1,
                    kind,
                    s,
                    shift: if kind == Kind::Plain {
                        [0.0; 3]
                    } else {
                        at.shifts.get(at.cell, s)
                    },
                });
            }
        }
    }
    Ok(())
}

/// The rows of one gather, staged, before they are laid out.
pub(super) struct Gathered {
    bufs: RowBufs,
    n: usize,
    blocks: usize,
    dist2: bool,
}

/// Gather every row of `binned` for `n` points. `half` keeps `j > i`;
/// `repeat` (an atom can meet another through two images) keeps the
/// nearest image of each column; `dist2` stages the distances.
pub(super) fn gather(
    binned: &Binned,
    cell: &Cell,
    n: usize,
    half: bool,
    repeat: bool,
    dist2: bool,
) -> Result<Gathered, Error> {
    let grid = &binned.grid;
    let threads = binned.threads;
    let ncell = grid.offsets.len() - 1;
    let ranges: Vec<(usize, usize)> = {
        #[cfg(feature = "parallel")]
        {
            if threads > 1 {
                super::cell_ranges(&grid.offsets, threads)
            } else {
                vec![(0, ncell)]
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = threads;
            vec![(0, ncell)]
        }
    };
    let mut bufs = RowBufs::take();
    bufs.stages.resize_with(ranges.len(), Stage::default);
    bufs.at.clear();
    bufs.at.resize(n, 0);
    bufs.first.clear();
    bufs.first.resize(n + 1, 0);
    let walk = Walker::new(binned, cell)?;
    let failed = std::sync::atomic::AtomicBool::new(false);
    let stages = Shared(bufs.stages.as_mut_ptr());
    let at = Shared(bufs.at.as_mut_ptr());
    let first = Shared(bufs.first.as_mut_ptr());
    let staged_d2 = dist2 || repeat;
    each_index(ranges.len(), |t| {
        let (lo, hi) = ranges[t];
        // Safety: range `t` owns stage `t`, and its atoms' `at` and
        // `first` entries, alone.
        let stage = unsafe { &mut *stages.get().add(t) };
        stage.cols.clear();
        stage.d2.clear();
        let rows = RowOut {
            at,
            first,
            stage: (t as u64) << STAGE_BITS,
        };
        let result = match (staged_d2, half) {
            (false, false) => walk.range_on::<false, false>(stage, lo, hi, repeat, rows),
            (false, true) => walk.range_on::<false, true>(stage, lo, hi, repeat, rows),
            (true, false) => walk.range_on::<true, false>(stage, lo, hi, repeat, rows),
            (true, true) => walk.range_on::<true, true>(stage, lo, hi, repeat, rows),
        };
        if result.is_err() {
            failed.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    if failed.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(Error::TooManyImages);
    }
    Ok(Gathered {
        bufs,
        n,
        blocks: if threads > 1 {
            (threads * BLOCKS_PER_THREAD).min(n)
        } else {
            1
        },
        dist2,
    })
}

/// Row blocks per thread in the layout passes.
const BLOCKS_PER_THREAD: usize = 4;

/// Where [`Walker::range`] records each row.
#[derive(Clone, Copy)]
struct RowOut {
    at: Shared<u64>,
    first: Shared<u32>,
    stage: u64,
}

/// Where bin `b` of an `n` mesh sits on each axis (first, inner, last),
/// as an index into the 27 tabulated runs.
#[inline(always)]
pub(super) fn bin_class(b: [i32; 3], n: [i32; 3]) -> usize {
    let at = |v: i32, n: i32| {
        if v == 0 {
            0
        } else if v == n - 1 {
            2
        } else {
            1
        }
    };
    (at(b[0], n[0]) + 3 * at(b[1], n[1]) + 9 * at(b[2], n[2])) as usize
}

pub(super) struct Walker<'a> {
    grid: &'a Grid,
    cell: &'a Cell,
    cut2: f64,
    reach: [i32; 3],
    ortho: bool,
    /// Bin widths, as the stencil takes them.
    width: [f64; 3],
    shifts: Shifts,
    /// With a reach of one bin and three bins or more on every axis,
    /// the runs of a bin by where it sits on each axis (first, inner,
    /// last): flat offsets, roles, and shifts. Otherwise empty.
    steps: Vec<Vec<Step>>,
    /// [`Path`] the search chose on the calling thread.
    #[cfg(target_arch = "x86_64")]
    path: Path,
}

impl<'a> Walker<'a> {
    pub(super) fn new(binned: &'a Binned, cell: &'a Cell) -> Result<Self, Error> {
        let grid = &binned.grid;
        let [nx, ny, nz] = grid.n;
        let width = [
            grid.widths[0] / f64::from(nx),
            grid.widths[1] / f64::from(ny),
            grid.widths[2] / f64::from(nz),
        ];
        let mut walk = Walker {
            grid,
            cell,
            cut2: binned.cut2,
            reach: binned.reach,
            ortho: cell.is_ortho(),
            width,
            shifts: Shifts::new(cell, grid.n, binned.reach),
            steps: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            path: binned.path,
        };
        if binned.reach == [1, 1, 1] && nx >= 3 && ny >= 3 && nz >= 3 {
            let mut stage = Stage::default();
            for class in 0..27 {
                let at = |k: i32, n: i32| [0, 1, n - 1][k as usize];
                let b = [at(class % 3, nx), at(class / 3 % 3, ny), at(class / 9, nz)];
                let c = bins::flat_cell(b, grid.n);
                walk.steps_of(c, b, &mut stage)?;
                walk.steps.push(stage.steps.clone());
            }
        }
        Ok(walk)
    }

    /// Where bin `b` sits on each axis, as an index into `steps`.
    #[inline(always)]
    fn class(&self, b: [i32; 3]) -> usize {
        bin_class(b, self.grid.n)
    }

    /// The bin widths, the shift table, and the tabulated runs, for a
    /// caller that keeps them beside the grid.
    pub(super) fn into_tables(self) -> ([f64; 3], Shifts, Vec<Vec<Step>>) {
        (self.width, self.shifts, self.steps)
    }

    /// [`bin_steps`] of bin `c` at `b`, into `stage.steps`.
    fn steps_of(&self, c: usize, b: [i32; 3], stage: &mut Stage) -> Result<(), Error> {
        let reach = Reach {
            n: self.grid.n,
            reach: self.reach,
            ortho: self.ortho,
            width: self.width,
            cut2: self.cut2,
            shifts: &self.shifts,
            cell: self.cell,
        };
        bin_steps(&reach, c, b, &mut stage.axes, &mut stage.steps)
    }

    /// Gather the rows of the atoms in bins `lo..hi` into `stage`, `W`
    /// packed candidates at a time (`0`: every run read in place).
    #[inline(always)]
    fn range<const D2: bool, const HALF: bool, const W: usize>(
        &self,
        stage: &mut Stage,
        lo: usize,
        hi: usize,
        repeat: bool,
        rows: RowOut,
    ) -> Result<(), Error> {
        if lo >= hi {
            return Ok(());
        }
        let grid = self.grid;
        let off = &grid.offsets;
        let [nx, ny, _] = grid.n;
        let (mut bx, mut by, mut bz) = (
            (lo % nx as usize) as i32,
            ((lo / nx as usize) % ny as usize) as i32,
            (lo / (nx as usize * ny as usize)) as i32,
        );
        for c in lo..hi {
            let (s_lo, s_hi) = (off[c], off[c + 1]);
            if s_lo < s_hi {
                let steps = if self.steps.is_empty() {
                    self.steps_of(c, [bx, by, bz], stage)?;
                    &stage.steps[..]
                } else {
                    &self.steps[self.class([bx, by, bz])][..]
                };
                // A wide row reads every run but a shifted source one from
                // one packed list, a target run as `q - S`: the walk's own
                // `p - S` of that pair, read here as `q`. A run of fewer
                // slots than lanes then costs no more than a full one.
                if W != 0 {
                    let packed: usize = steps
                        .iter()
                        .filter(|step| step.kind != Kind::Source)
                        .map(|step| {
                            let first = (c as isize + step.rel) as usize;
                            off[first + step.bins] - off[first]
                        })
                        .sum();
                    stage.pack.reserve(packed + SLACK);
                }
                stage.runs.clear();
                let mut room = SLACK;
                let mut k = 0usize;
                for step in steps {
                    let first = (c as isize + step.rel) as usize;
                    let (a, b) = (off[first], off[first + step.bins]);
                    if a >= b {
                        continue;
                    }
                    room += b - a;
                    if W == 0 || step.kind == Kind::Source {
                        stage.runs.push(Run {
                            lo: a,
                            hi: b,
                            kind: step.kind,
                            shift: step.shift,
                        });
                        continue;
                    }
                    // Safety: the pack has room for the bin's packed slots
                    // and the wide copy's last store.
                    unsafe {
                        pack_w::<W>(grid.coords.cols(), a, b, step.shift, stage.pack.cols(), k)
                    };
                    k += b - a;
                }
                let packed = stage.pack.cols();
                let Stage { cols, d2, runs, .. } = stage;
                let bin = Bin {
                    runs: &runs[..],
                    packed,
                    k,
                    room,
                };
                for s in s_lo..s_hi {
                    let out = (&mut *cols, &mut *d2);
                    self.row::<D2, HALF, W>(out, bin, s, repeat, rows);
                }
            }
            bx += 1;
            if bx == nx {
                bx = 0;
                by += 1;
                if by == ny {
                    by = 0;
                    bz += 1;
                }
            }
        }
        Ok(())
    }

    /// [`Walker::range`] on the AVX-512 path.
    ///
    /// # Safety
    /// The CPU has AVX-512F, AVX-512VL, and POPCNT.
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    #[target_feature(enable = "avx512f,avx512vl,popcnt")]
    unsafe fn range_avx512<const D2: bool, const HALF: bool>(
        &self,
        stage: &mut Stage,
        lo: usize,
        hi: usize,
        repeat: bool,
        rows: RowOut,
    ) -> Result<(), Error> {
        self.range::<D2, HALF, 8>(stage, lo, hi, repeat, rows)
    }

    /// [`Walker::range`] on the AVX2 path.
    ///
    /// # Safety
    /// The CPU has AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn range_avx2<const D2: bool, const HALF: bool>(
        &self,
        stage: &mut Stage,
        lo: usize,
        hi: usize,
        repeat: bool,
        rows: RowOut,
    ) -> Result<(), Error> {
        self.range::<D2, HALF, 4>(stage, lo, hi, repeat, rows)
    }

    /// [`Walker::range`] on the path the search chose.
    fn range_on<const D2: bool, const HALF: bool>(
        &self,
        stage: &mut Stage,
        lo: usize,
        hi: usize,
        repeat: bool,
        rows: RowOut,
    ) -> Result<(), Error> {
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        if self.path == Path::Avx512 {
            // Safety: `select` returned AVX-512, so F, VL, and POPCNT are on.
            return unsafe { self.range_avx512::<D2, HALF>(stage, lo, hi, repeat, rows) };
        }
        #[cfg(target_arch = "x86_64")]
        if self.path == Path::Avx2 {
            // Safety: `select` returned AVX2, so the CPU has it.
            return unsafe { self.range_avx2::<D2, HALF>(stage, lo, hi, repeat, rows) };
        }
        self.range::<D2, HALF, 0>(stage, lo, hi, repeat, rows)
    }

    /// Stage the row of slot `s` from `bin`.
    #[inline(always)]
    fn row<const D2: bool, const HALF: bool, const W: usize>(
        &self,
        (out_cols, out_d2): (&mut Vec<i32>, &mut Vec<f64>),
        bin: Bin,
        s: usize,
        repeat: bool,
        rows: RowOut,
    ) {
        let c: &Coords = &self.grid.coords;
        let start = out_cols.len();
        out_cols.reserve(bin.room);
        if D2 {
            out_d2.reserve(bin.room);
        }
        let cols = unsafe { out_cols.as_mut_ptr().add(start) };
        let d2s = if D2 {
            unsafe { out_d2.as_mut_ptr().add(start) }
        } else {
            std::ptr::null_mut()
        };
        let i = c.id[s];
        let p = [c.x[s], c.y[s], c.z[s]];
        let probe = Probe {
            p,
            i,
            cut2: self.cut2,
        };
        // Safety: `bin.room` covers every candidate and the wide kernel's
        // last store; the packed list holds `bin.k` and the runs are
        // slots of the grid.
        let mut len =
            unsafe { scan_w::<D2, HALF, W>(bin.packed, bin.k, probe, Out { cols, d2s, len: 0 }) };
        let grid = c.cols();
        for run in bin.runs {
            let at = Out { cols, d2s, len };
            len = unsafe {
                match run.kind {
                    Kind::Plain => {
                        scan::<D2, HALF, false>(grid, run.lo, run.hi, probe, [0.0; 3], at)
                    }
                    Kind::Source => {
                        let q = [
                            p[0] - run.shift[0],
                            p[1] - run.shift[1],
                            p[2] - run.shift[2],
                        ];
                        let probe = Probe { p: q, ..probe };
                        scan::<D2, HALF, false>(grid, run.lo, run.hi, probe, [0.0; 3], at)
                    }
                    Kind::Target => {
                        scan::<D2, HALF, true>(grid, run.lo, run.hi, probe, run.shift, at)
                    }
                }
            };
        }
        // Safety: the scans wrote `len < room` entries.
        let row = unsafe { std::slice::from_raw_parts_mut(cols, len) };
        let d2 = D2.then(|| unsafe { std::slice::from_raw_parts_mut(d2s, len) });
        let len = finish_row(row, d2, repeat);
        unsafe {
            out_cols.set_len(start + len);
            if D2 {
                out_d2.set_len(start + len);
            }
            *rows.at.get().add(i as usize) = rows.stage | start as u64;
            *rows.first.get().add(i as usize) = len as u32;
        }
    }
}

/// Entries past a row's candidates that the wide kernel may store.
const SLACK: usize = 8;

/// `v` folded into `0..n`, and the shift that takes it there.
#[inline]
fn fold(v: i32, n: i32) -> (i32, i32) {
    if (0..n).contains(&v) {
        (v, 0)
    } else if (-n..0).contains(&v) {
        (v + n, -1)
    } else if (n..2 * n).contains(&v) {
        (v - n, 1)
    } else {
        (v.rem_euclid(n), v.div_euclid(n))
    }
}

/// What every row of a bin reads: the runs read in place, the `k`
/// packed candidates, and `room`, more slots than a row can stage.
#[derive(Clone, Copy)]
struct Bin<'a> {
    runs: &'a [Run],
    packed: Cols,
    k: usize,
    room: usize,
}

/// The atom a row belongs to: its number `i`, its position `p` (less
/// the run's shift, for a shifted source run), and the squared cutoff.
#[derive(Clone, Copy)]
struct Probe {
    p: [f64; 3],
    i: u32,
    cut2: f64,
}

/// A row's staged entries and how many it holds.
#[derive(Clone, Copy)]
struct Out {
    cols: *mut i32,
    d2s: *mut f64,
    len: usize,
}

/// Slot columns: positions and atom numbers.
#[derive(Clone, Copy)]
struct Cols {
    x: *const f64,
    y: *const f64,
    z: *const f64,
    id: *const u32,
}

impl Coords {
    fn cols(&self) -> Cols {
        Cols {
            x: self.x.as_ptr(),
            y: self.y.as_ptr(),
            z: self.z.as_ptr(),
            id: self.id.as_ptr(),
        }
    }
}

/// One bin's candidates, packed.
#[derive(Default)]
struct Pack {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    id: Vec<u32>,
}

impl Pack {
    /// Room for `len` candidates in every column. The vectors stay
    /// empty; their spare capacity is written through [`Pack::cols`],
    /// which a reallocation need not keep, so a bin reserves before it
    /// packs anything.
    fn reserve(&mut self, len: usize) {
        self.x.reserve(len);
        self.y.reserve(len);
        self.z.reserve(len);
        self.id.reserve(len);
    }

    fn cols(&mut self) -> Cols {
        Cols {
            x: self.x.as_mut_ptr(),
            y: self.y.as_mut_ptr(),
            z: self.z.as_mut_ptr(),
            id: self.id.as_mut_ptr(),
        }
    }
}

/// Copy slots `lo..hi` of `from` to `to[k..]`, less `shift`.
///
/// # Safety
/// `lo..hi` are slots of `from`, and `to` has room for `k + hi - lo`.
#[inline(always)]
unsafe fn pack(from: Cols, lo: usize, hi: usize, shift: [f64; 3], to: Cols, k: usize) {
    for (t, at) in (lo..hi).zip(k..) {
        unsafe {
            *(to.x as *mut f64).add(at) = *from.x.add(t) - shift[0];
            *(to.y as *mut f64).add(at) = *from.y.add(t) - shift[1];
            *(to.z as *mut f64).add(at) = *from.z.add(t) - shift[2];
            *(to.id as *mut u32).add(at) = *from.id.add(t);
        }
    }
}

/// [`pack`] eight slots at a time, the same subtraction lane by lane.
///
/// # Safety
/// As [`pack`], with [`SLACK`] more room in `to`, on a CPU with
/// AVX-512F and AVX-512VL.
// `linkcell_avx512` builds this only from rustc 1.89, which stabilised
// these intrinsics.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[clippy::msrv = "1.89"]
#[inline(always)]
unsafe fn pack8(from: Cols, lo: usize, hi: usize, shift: [f64; 3], to: Cols, k: usize) {
    use std::arch::x86_64::{
        __m256i, _mm256_maskz_loadu_epi32, _mm256_storeu_si256, _mm512_maskz_loadu_pd,
        _mm512_set1_pd, _mm512_storeu_pd, _mm512_sub_pd,
    };
    let (sx, sy, sz) = (
        _mm512_set1_pd(shift[0]),
        _mm512_set1_pd(shift[1]),
        _mm512_set1_pd(shift[2]),
    );
    let (mut t, mut at) = (lo, k);
    while t < hi {
        let m = lanes(hi - t);
        unsafe {
            let x = _mm512_sub_pd(_mm512_maskz_loadu_pd(m, from.x.add(t)), sx);
            let y = _mm512_sub_pd(_mm512_maskz_loadu_pd(m, from.y.add(t)), sy);
            let z = _mm512_sub_pd(_mm512_maskz_loadu_pd(m, from.z.add(t)), sz);
            let id = _mm256_maskz_loadu_epi32(m, from.id.add(t) as *const i32);
            _mm512_storeu_pd((to.x as *mut f64).add(at), x);
            _mm512_storeu_pd((to.y as *mut f64).add(at), y);
            _mm512_storeu_pd((to.z as *mut f64).add(at), z);
            _mm256_storeu_si256((to.id as *mut u32).add(at) as *mut __m256i, id);
        }
        t += 8;
        at += 8;
    }
}

/// Lanes `0..left` of eight.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[inline(always)]
fn lanes(left: usize) -> u8 {
    if left >= 8 {
        0xff
    } else {
        (1u8 << left) - 1
    }
}

/// Lanes `0..left` of four, as 64-bit and as 32-bit lanes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn lanes4(left: usize) -> (std::arch::x86_64::__m256i, std::arch::x86_64::__m128i) {
    use std::arch::x86_64::{
        _mm256_cvtepi32_epi64, _mm_cmpgt_epi32, _mm_set1_epi32, _mm_setr_epi32,
    };
    unsafe {
        let m = _mm_cmpgt_epi32(
            _mm_set1_epi32(left.min(4) as i32),
            _mm_setr_epi32(0, 1, 2, 3),
        );
        (_mm256_cvtepi32_epi64(m), m)
    }
}

/// [`pack`] four slots at a time, the same subtraction lane by lane.
///
/// # Safety
/// As [`pack`], with [`SLACK`] more room in `to`, on a CPU with AVX2.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pack4(from: Cols, lo: usize, hi: usize, shift: [f64; 3], to: Cols, k: usize) {
    use std::arch::x86_64::{
        __m128i, _mm256_maskload_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
        _mm_maskload_epi32, _mm_storeu_si128,
    };
    unsafe {
        let (sx, sy, sz) = (
            _mm256_set1_pd(shift[0]),
            _mm256_set1_pd(shift[1]),
            _mm256_set1_pd(shift[2]),
        );
        let (mut t, mut at) = (lo, k);
        while t < hi {
            let (m, m32) = lanes4(hi - t);
            let x = _mm256_sub_pd(_mm256_maskload_pd(from.x.add(t), m), sx);
            let y = _mm256_sub_pd(_mm256_maskload_pd(from.y.add(t), m), sy);
            let z = _mm256_sub_pd(_mm256_maskload_pd(from.z.add(t), m), sz);
            let id = _mm_maskload_epi32(from.id.add(t) as *const i32, m32);
            _mm256_storeu_pd((to.x as *mut f64).add(at), x);
            _mm256_storeu_pd((to.y as *mut f64).add(at), y);
            _mm256_storeu_pd((to.z as *mut f64).add(at), z);
            _mm_storeu_si128((to.id as *mut u32).add(at) as *mut __m128i, id);
            t += 4;
            at += 4;
        }
    }
}

/// Per mask of four kept lanes: the bytes that bring those 32-bit
/// lanes, in order, to the front.
#[cfg(target_arch = "x86_64")]
const PICK4: [[u8; 16]; 16] = {
    let mut table = [[0x80u8; 16]; 16];
    let mut keep = 0;
    while keep < 16 {
        let (mut to, mut lane) = (0, 0);
        while lane < 4 {
            if keep & (1 << lane) != 0 {
                let mut b = 0;
                while b < 4 {
                    table[keep][4 * to + b] = (4 * lane + b) as u8;
                    b += 1;
                }
                to += 1;
            }
            lane += 1;
        }
        keep += 1;
    }
    table
};

/// The same for 64-bit lanes, as pairs of 32-bit lane numbers.
#[cfg(target_arch = "x86_64")]
const PICK4_WIDE: [[i32; 8]; 16] = {
    let mut table = [[0i32; 8]; 16];
    let mut keep = 0;
    while keep < 16 {
        let (mut to, mut lane) = (0, 0);
        while lane < 4 {
            if keep & (1 << lane) != 0 {
                table[keep][2 * to] = 2 * lane;
                table[keep][2 * to + 1] = 2 * lane + 1;
                to += 1;
            }
            lane += 1;
        }
        keep += 1;
    }
    table
};

/// [`scan`] four candidates at a time: the same products and sums, lane
/// by lane, so the same distances. Each step stores four columns and
/// distances at the row's end; [`SLACK`] covers the lanes past it.
///
/// # Safety
/// As [`scan`], with [`SLACK`] more room, on a CPU with AVX2.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn scan4<const D2: bool, const HALF: bool>(
    c: Cols,
    lo: usize,
    hi: usize,
    probe: Probe,
    out: Out,
) -> usize {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_pd, _mm256_and_pd, _mm256_castpd_ps, _mm256_castps_pd,
        _mm256_castsi256_pd, _mm256_cmp_pd, _mm256_loadu_si256, _mm256_maskload_pd,
        _mm256_movemask_pd, _mm256_mul_pd, _mm256_permutevar8x32_ps, _mm256_set1_pd,
        _mm256_storeu_pd, _mm256_sub_pd, _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_cmpgt_epi32,
        _mm_loadu_si128, _mm_maskload_epi32, _mm_movemask_ps, _mm_set1_epi32, _mm_shuffle_epi8,
        _mm_storeu_si128, _CMP_LT_OQ,
    };
    let Out { cols, d2s, mut len } = out;
    let Probe { p, i, cut2 } = probe;
    unsafe {
        let (px, py, pz) = (
            _mm256_set1_pd(p[0]),
            _mm256_set1_pd(p[1]),
            _mm256_set1_pd(p[2]),
        );
        let cut = _mm256_set1_pd(cut2);
        let me = _mm_set1_epi32(i as i32);
        let mut t = lo;
        while t < hi {
            let (m, m32) = lanes4(hi - t);
            let dx = _mm256_sub_pd(_mm256_maskload_pd(c.x.add(t), m), px);
            let dy = _mm256_sub_pd(_mm256_maskload_pd(c.y.add(t), m), py);
            let dz = _mm256_sub_pd(_mm256_maskload_pd(c.z.add(t), m), pz);
            let j = _mm_maskload_epi32(c.id.add(t) as *const i32, m32);
            let d2 = _mm256_add_pd(
                _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                _mm256_mul_pd(dz, dz),
            );
            let near = _mm256_and_pd(_mm256_cmp_pd(d2, cut, _CMP_LT_OQ), _mm256_castsi256_pd(m));
            // Atom numbers fit i32, so the signed compare orders them.
            let (bits, flip) = if HALF {
                (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(j, me))), 0)
            } else {
                (
                    _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(j, me))),
                    0xf,
                )
            };
            let keep = (_mm256_movemask_pd(near) & (bits ^ flip)) as usize;
            let pick = _mm_loadu_si128(PICK4[keep].as_ptr() as *const __m128i);
            _mm_storeu_si128(cols.add(len) as *mut __m128i, _mm_shuffle_epi8(j, pick));
            if D2 {
                let pick = _mm256_loadu_si256(PICK4_WIDE[keep].as_ptr() as *const __m256i);
                let moved = _mm256_permutevar8x32_ps(_mm256_castpd_ps(d2), pick);
                _mm256_storeu_pd(d2s.add(len), _mm256_castps_pd(moved));
            }
            len += keep.count_ones() as usize;
            t += 4;
        }
    }
    len
}

/// Test candidates `lo..hi` against the row of `probe` and append its
/// hits at `out.cols[out.len..]` (and `out.d2s`) as `q - p`, with `q`
/// less `shift` under `SHIFTED` (a target run). Returns the new length.
///
/// # Safety
/// `lo..hi` are candidates of `c`, and `out.cols` (and `out.d2s` under
/// `D2`) have room for `hi - lo` entries past `out.len`.
#[inline(always)]
unsafe fn scan<const D2: bool, const HALF: bool, const SHIFTED: bool>(
    c: Cols,
    lo: usize,
    hi: usize,
    probe: Probe,
    shift: [f64; 3],
    out: Out,
) -> usize {
    let Out { cols, d2s, mut len } = out;
    let Probe { p, i, cut2 } = probe;
    for t in lo..hi {
        let (mut q, j) = unsafe { ([*c.x.add(t), *c.y.add(t), *c.z.add(t)], *c.id.add(t)) };
        if SHIFTED {
            q = [q[0] - shift[0], q[1] - shift[1], q[2] - shift[2]];
        }
        let (dx, dy, dz) = (q[0] - p[0], q[1] - p[1], q[2] - p[2]);
        let d2 = dx * dx + dy * dy + dz * dz;
        unsafe {
            *cols.add(len) = j as i32;
            if D2 {
                *d2s.add(len) = d2;
            }
        }
        let other = if HALF { j > i } else { j != i };
        len += usize::from((d2 < cut2) & other);
    }
    len
}

/// [`pack`] with the kernel of `W` lanes.
///
/// # Safety
/// As [`pack`], with [`SLACK`] more room in `to`, on a CPU with that
/// kernel's features.
#[inline(always)]
unsafe fn pack_w<const W: usize>(
    from: Cols,
    lo: usize,
    hi: usize,
    shift: [f64; 3],
    to: Cols,
    k: usize,
) {
    unsafe {
        #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
        if W == 8 {
            return pack8(from, lo, hi, shift, to, k);
        }
        #[cfg(target_arch = "x86_64")]
        if W == 4 {
            return pack4(from, lo, hi, shift, to, k);
        }
        pack(from, lo, hi, shift, to, k)
    }
}

/// [`scan`] of the first `k` packed candidates with the kernel of `W`
/// lanes, or one at a time for a `W` with no kernel; nothing for
/// `W == 0`, whose rows pack nothing.
///
/// # Safety
/// As [`scan`], with [`SLACK`] more room, on a CPU with that kernel's
/// features.
#[inline(always)]
unsafe fn scan_w<const D2: bool, const HALF: bool, const W: usize>(
    packed: Cols,
    k: usize,
    probe: Probe,
    out: Out,
) -> usize {
    #[cfg(all(target_arch = "x86_64", linkcell_avx512))]
    if W == 8 {
        return unsafe { scan8::<D2, HALF>(packed, 0, k, probe, out) };
    }
    #[cfg(target_arch = "x86_64")]
    if W == 4 {
        return unsafe { scan4::<D2, HALF>(packed, 0, k, probe, out) };
    }
    if W != 0 {
        return unsafe { scan::<D2, HALF, false>(packed, 0, k, probe, [0.0; 3], out) };
    }
    out.len
}

/// [`scan`] eight candidates at a time: the same products and sums,
/// lane by lane, so the same distances. Each step stores a full vector
/// of columns and distances at the row's end; [`SLACK`] covers the
/// lanes past it.
///
/// # Safety
/// As [`scan`], with [`SLACK`] more room, on a CPU with AVX-512F,
/// AVX-512VL, and POPCNT.
// `linkcell_avx512` builds this only from rustc 1.89, which stabilised
// these intrinsics.
#[cfg(all(target_arch = "x86_64", linkcell_avx512))]
#[clippy::msrv = "1.89"]
#[inline(always)]
unsafe fn scan8<const D2: bool, const HALF: bool>(
    c: Cols,
    lo: usize,
    hi: usize,
    probe: Probe,
    out: Out,
) -> usize {
    use std::arch::x86_64::{
        __m256i, _mm256_mask_cmpgt_epi32_mask, _mm256_mask_cmpneq_epi32_mask,
        _mm256_maskz_compress_epi32, _mm256_maskz_loadu_epi32, _mm256_set1_epi32,
        _mm256_storeu_si256, _mm512_add_pd, _mm512_mask_cmp_pd_mask, _mm512_maskz_compress_pd,
        _mm512_maskz_loadu_pd, _mm512_mul_pd, _mm512_set1_pd, _mm512_storeu_pd, _mm512_sub_pd,
        _CMP_LT_OQ,
    };
    let Out { cols, d2s, mut len } = out;
    let Probe { p, i, cut2 } = probe;
    let (px, py, pz) = (
        _mm512_set1_pd(p[0]),
        _mm512_set1_pd(p[1]),
        _mm512_set1_pd(p[2]),
    );
    let cut = _mm512_set1_pd(cut2);
    let me = _mm256_set1_epi32(i as i32);
    let mut t = lo;
    while t < hi {
        let m = lanes(hi - t);
        unsafe {
            let dx = _mm512_sub_pd(_mm512_maskz_loadu_pd(m, c.x.add(t)), px);
            let dy = _mm512_sub_pd(_mm512_maskz_loadu_pd(m, c.y.add(t)), py);
            let dz = _mm512_sub_pd(_mm512_maskz_loadu_pd(m, c.z.add(t)), pz);
            let j = _mm256_maskz_loadu_epi32(m, c.id.add(t) as *const i32);
            let d2 = _mm512_add_pd(
                _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)),
                _mm512_mul_pd(dz, dz),
            );
            let near = _mm512_mask_cmp_pd_mask(m, d2, cut, _CMP_LT_OQ);
            // Atom numbers fit i32, so the signed compare orders them.
            let keep = if HALF {
                _mm256_mask_cmpgt_epi32_mask(near, j, me)
            } else {
                _mm256_mask_cmpneq_epi32_mask(near, j, me)
            };
            _mm256_storeu_si256(
                cols.add(len) as *mut __m256i,
                _mm256_maskz_compress_epi32(keep, j),
            );
            if D2 {
                _mm512_storeu_pd(d2s.add(len), _mm512_maskz_compress_pd(keep, d2));
            }
            len += keep.count_ones() as usize;
        }
        t += 8;
    }
    len
}

/// Sort a staged row by column and, when `repeat`, keep the nearest
/// image of each column. Returns its length.
///
/// A row is sorted already when its atoms are numbered in bin order;
/// otherwise it is short, and sorts by insertion. Without `repeat` no
/// column comes twice: one atom sits in one bin, and the bins around a
/// bin are distinct bin-image pairs, of which at most one image is in
/// reach.
#[inline]
fn finish_row(cols: &mut [i32], d2: Option<&mut [f64]>, repeat: bool) -> usize {
    let len = cols.len();
    if len < 2 {
        return len;
    }
    let sorted = cols.windows(2).all(|w| w[0] < w[1]);
    if sorted {
        return len;
    }
    match d2 {
        None => {
            for k in 1..len {
                let col = cols[k];
                let mut at = k;
                while at > 0 && cols[at - 1] > col {
                    cols[at] = cols[at - 1];
                    at -= 1;
                }
                cols[at] = col;
            }
            if repeat {
                return dedup(cols, None);
            }
            len
        }
        Some(d2) => {
            if repeat {
                // Ties on the column go to the nearer image; distances are
                // not negative, so their bits sort as they do.
                for k in 1..len {
                    let key = (cols[k], d2[k].to_bits());
                    let mut at = k;
                    while at > 0 && (cols[at - 1], d2[at - 1].to_bits()) > key {
                        cols[at] = cols[at - 1];
                        d2[at] = d2[at - 1];
                        at -= 1;
                    }
                    cols[at] = key.0;
                    d2[at] = f64::from_bits(key.1);
                }
                return dedup(cols, Some(d2));
            }
            for k in 1..len {
                let (col, d) = (cols[k], d2[k]);
                let mut at = k;
                while at > 0 && cols[at - 1] > col {
                    cols[at] = cols[at - 1];
                    d2[at] = d2[at - 1];
                    at -= 1;
                }
                cols[at] = col;
                d2[at] = d;
            }
            len
        }
    }
}

/// Keep the first entry of each column of a sorted row; return how many.
fn dedup(cols: &mut [i32], mut d2: Option<&mut [f64]>) -> usize {
    let mut kept = 0usize;
    for k in 0..cols.len() {
        if kept > 0 && cols[kept - 1] == cols[k] {
            continue;
        }
        cols[kept] = cols[k];
        if let Some(d2) = d2.as_deref_mut() {
            d2[kept] = d2[k];
        }
        kept += 1;
    }
    kept
}

/// `job(t)` for `t` in `0..count`, on the pool when it has more than one
/// thread.
fn each_index(count: usize, job: impl Fn(usize) + Sync + Send) {
    #[cfg(feature = "parallel")]
    if count > 1 && crate::pool_width() > 1 {
        use rayon::prelude::*;
        (0..count).into_par_iter().for_each(job);
        return;
    }
    (0..count).for_each(job);
}

impl Gathered {
    /// Turn the row lengths into CSR row starts, and return the column
    /// count. [`Error::Overflow`] when it does not fit `i32`.
    pub(super) fn scan(&mut self) -> Result<usize, Error> {
        let (n, blocks) = (self.n, self.blocks);
        let first = &mut self.bufs.first;
        let span = |b: usize| (n * b / blocks, n * (b + 1) / blocks);
        let sums: Vec<u64> = if blocks > 1 {
            let fp = Shared(first.as_mut_ptr());
            let mut sums = vec![0u64; blocks];
            let sp = Shared(sums.as_mut_ptr());
            each_index(blocks, |b| {
                let (lo, hi) = span(b);
                // Safety: block `b` reads its own rows and writes its sum.
                let total = (lo..hi)
                    .map(|i| u64::from(unsafe { *fp.get().add(i) }))
                    .sum();
                unsafe { *sp.get().add(b) = total };
            });
            sums
        } else {
            vec![first[..n].iter().map(|&l| u64::from(l)).sum()]
        };
        let total: u64 = sums.iter().sum();
        if total > i32::MAX as u64 {
            return Err(Error::Overflow);
        }
        let mut base = Vec::with_capacity(blocks);
        let mut run = 0u64;
        for s in &sums {
            base.push(run as u32);
            run += s;
        }
        let fp = Shared(first.as_mut_ptr());
        each_index(blocks, |b| {
            let (lo, hi) = span(b);
            let mut run = base[b];
            for i in lo..hi {
                // Safety: block `b` owns rows `lo..hi`; the total fits i32.
                unsafe {
                    let len = *fp.get().add(i);
                    *fp.get().add(i) = run;
                    run += len;
                }
            }
        });
        first[n] = total as u32;
        Ok(total as usize)
    }

    /// Lay the rows out at `offsets` (`n + 1` values), `cols`, and, when
    /// non-null, `dist2`, then keep the buffers for the next call.
    ///
    /// # Safety
    /// [`Gathered::scan`] has run. `offsets` is writable for `n + 1`
    /// values, and `cols` (and a non-null `dist2`) for the column count.
    /// A non-null `dist2` needs rows gathered with distances.
    pub(super) unsafe fn write(self, offsets: *mut i32, cols: *mut i32, dist2: *mut f64) {
        debug_assert!(dist2.is_null() || self.dist2);
        let (n, blocks) = (self.n, self.blocks);
        let bufs = &self.bufs;
        let (op, cp, dp) = (Shared(offsets), Shared(cols), Shared(dist2));
        let stages: Vec<(Shared<i32>, Shared<f64>)> = bufs
            .stages
            .iter()
            .map(|s| {
                (
                    Shared(s.cols.as_ptr() as *mut i32),
                    Shared(s.d2.as_ptr() as *mut f64),
                )
            })
            .collect();
        let mask = (1u64 << STAGE_BITS) - 1;
        each_index(blocks, |b| {
            let (lo, hi) = (n * b / blocks, n * (b + 1) / blocks);
            for i in lo..hi {
                let (from, to) = (bufs.first[i] as usize, bufs.first[i + 1] as usize);
                // Safety: block `b` writes rows `lo..hi` alone; each staged
                // row holds `to - from` entries.
                unsafe {
                    *op.get().add(i) = from as i32;
                    if to == from {
                        continue;
                    }
                    let at = bufs.at[i];
                    let (sc, sd) = stages[(at >> STAGE_BITS) as usize];
                    let start = (at & mask) as usize;
                    for k in 0..to - from {
                        *cp.get().add(from + k) = *sc.get().add(start + k);
                    }
                    if !dp.get().is_null() {
                        for k in 0..to - from {
                            *dp.get().add(from + k) = *sd.get().add(start + k);
                        }
                    }
                }
            }
        });
        unsafe { *op.get().add(n) = bufs.first[n] as i32 };
        drop(stages);
        self.bufs.give();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` points in a cell of edge `edge` sheared by `tilt`, the first
    /// `dense` of them in the middle third of every axis.
    fn frame(n: usize, dense: usize, edge: f64, tilt: f64) -> (Vec<[f64; 3]>, Cell) {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let a = [edge, 0.0, 0.0];
        let b = [tilt * edge, edge, 0.0];
        let c = [0.5 * tilt * edge, -0.5 * tilt * edge, edge];
        let xyz = (0..n)
            .map(|k| {
                let mut f = [next(), next(), next()];
                if k < dense {
                    f = f.map(|v| (1.0 + v) / 3.0);
                }
                [0, 1, 2].map(|x| f[0] * a[x] + f[1] * b[x] + f[2] * c[x])
            })
            .collect();
        (xyz, Cell::from_vectors(a, b, c, [0.0; 3]).unwrap())
    }

    /// The rows `walk` stages into one fresh stage: per point, its
    /// columns and distance bits.
    fn staged(n: usize, walk: impl FnOnce(&mut Stage, RowOut)) -> Vec<(Vec<i32>, Vec<u64>)> {
        let mut stage = Stage::default();
        let mut at = vec![0u64; n];
        let mut first = vec![0u32; n];
        walk(
            &mut stage,
            RowOut {
                at: Shared(at.as_mut_ptr()),
                first: Shared(first.as_mut_ptr()),
                stage: 0,
            },
        );
        (0..n)
            .map(|i| {
                let row = at[i] as usize..at[i] as usize + first[i] as usize;
                let d2 = stage.d2.get(row.clone()).unwrap_or_default();
                (
                    stage.cols[row].to_vec(),
                    d2.iter().map(|d| d.to_bits()).collect(),
                )
            })
            .collect()
    }

    /// The packed walks, one candidate at a time and on every path,
    /// against the walk that packs nothing.
    fn compare<const D2: bool, const HALF: bool>(
        walk: &mut Walker,
        n: usize,
        repeat: bool,
        label: &str,
    ) {
        let hi = walk.grid.offsets.len() - 1;
        let want = staged(n, |s, out| {
            walk.range::<D2, HALF, 0>(s, 0, hi, repeat, out).unwrap()
        });
        assert!(want.iter().any(|(cols, _)| !cols.is_empty()), "{label}");
        let got = staged(n, |s, out| {
            walk.range::<D2, HALF, 1>(s, 0, hi, repeat, out).unwrap()
        });
        assert_eq!(got, want, "{label} packed one at a time");
        for path in crate::simd::paths_to_compare() {
            #[cfg(target_arch = "x86_64")]
            {
                walk.path = path;
            }
            let got = staged(n, |s, out| {
                walk.range_on::<D2, HALF>(s, 0, hi, repeat, out).unwrap()
            });
            assert_eq!(got, want, "{label} {}", path.name());
        }
    }

    /// Bins whose rows pack several runs, one of them the dense bin, so
    /// a fresh pack needs more than the first run: every packed walk
    /// stages the rows of the walk that packs nothing, bit for bit.
    #[test]
    fn packed_rows_grow_across_steps() {
        let (cutoff, edge) = (3.5, 10.5);
        for (tilt, hint) in [(0.0, 3.5), (0.3, 3.5), (0.0, 1.75)] {
            let label = format!("tilt {tilt} hint {hint}");
            let (xyz, cell) = frame(160, 100, edge, tilt);
            let n = xyz.len();
            let binned = super::super::bin_atoms(&xyz, &cell, cutoff, None, Some(hint), true)
                .unwrap()
                .unwrap();
            let mut walk = Walker::new(&binned, &cell).unwrap();
            let off = &walk.grid.offsets;
            let [nx, ny, _] = walk.grid.n.map(|v| v as usize);
            let mut stage = Stage::default();
            let grows = (0..off.len() - 1).any(|c| {
                let b = [c % nx, c / nx % ny, c / (nx * ny)].map(|v| v as i32);
                let steps = if walk.steps.is_empty() {
                    walk.steps_of(c, b, &mut stage).unwrap();
                    stage.steps.clone()
                } else {
                    walk.steps[walk.class(b)].clone()
                };
                let runs: Vec<usize> = steps
                    .iter()
                    .filter(|step| step.kind != Kind::Source)
                    .map(|step| {
                        let first = (c as isize + step.rel) as usize;
                        off[first + step.bins] - off[first]
                    })
                    .filter(|&len| len > 0)
                    .collect();
                let packed: usize = runs.iter().sum();
                off[c] < off[c + 1] && runs.len() > 1 && packed >= 100 && runs[0] + SLACK < packed
            });
            assert!(grows, "{label}: no bin packs past its first run");
            let w = cell.widths();
            let repeat = (0..3).any(|a| 2.0 * cutoff >= w[a] * (1.0 - 1.0e-6));
            assert!(!repeat, "{label}");
            compare::<true, false>(&mut walk, n, repeat, &label);
            compare::<true, true>(&mut walk, n, repeat, &label);
            compare::<false, false>(&mut walk, n, repeat, &label);
        }
    }
}
