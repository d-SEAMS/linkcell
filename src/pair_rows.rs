//! Cutoff rows read from a kept grid, with no stored pair list.
//!
//! [`pairs_within`](crate::pairs_within) and
//! [`pairs_within_pbc`](crate::pairs_within_pbc) bin the points, search,
//! and return every row. A [`RowGrid`] keeps the bins and hands the rows
//! of the points a caller names to a closure, one row at a time. Nothing
//! but the bins is held between calls, so a list too large to store can
//! still be walked, and several threads can walk disjoint point ranges
//! of one grid at once.
//!
//! A periodic cell reads each row from the bins around the point's own,
//! with the bin runs, roles, and shifts of the CSR gather (see
//! `pair_gather.rs`). An open or mixed cell reads it from the occupied
//! bins and the partner stencil of `open_cell.rs`. Either way a row holds
//! the entries the pair list has for that source: the same targets, the
//! same shifts, and the same bits in every distance.

use std::ops::{ControlFlow, Range};

use super::gather::{bin_class, bin_steps, Kind, Reach, Shifts, Step, Walker};
use super::{bin_atoms, hit_estimate, in_pool, keep_half, try_hold, Binned, ImageOffsets};
use crate::cell::{periodic_face_collapsed, Cell};
use crate::open_cell::{next_up, MixedRows, PbcQuery};
use crate::Error;

/// Marks a point the mask dropped.
const MASKED: u32 = u32::MAX;

/// One entry of a row: an image of point `j` within the cutoff of point `i`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    /// Source index: the point the row belongs to.
    pub i: usize,
    /// Target index.
    pub j: usize,
    /// Integer cell shift applied to the target, as in
    /// [`Pair`](crate::Pair).
    pub shift: [i32; 3],
    /// `r_j - r_i` with the shift applied. Its squared length is `dist2`.
    /// On a fully periodic cell it is taken from the positions folded
    /// into the cell, so it equals `r_j + shift H - r_i` of the positions
    /// as given up to the rounding of that fold. With an open axis it is
    /// that expression itself.
    pub vector: [f64; 3],
    /// Squared distance after the shift, the bits
    /// [`pairs_within_pbc`](crate::pairs_within_pbc) stores.
    pub dist2: f64,
}

/// Where a buffered read of rows stands: the next entry is entry
/// `entry` of the row of point `point`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowCursor {
    /// Point whose row is read next.
    pub point: usize,
    /// Entries of that row already read.
    pub entry: usize,
}

impl RowCursor {
    /// The first entry of the row of `point`.
    pub fn at(point: usize) -> Self {
        RowCursor { point, entry: 0 }
    }
}

/// The bins of one cutoff search, kept so its rows can be read on demand.
///
/// [`RowGrid::build`] bins the points once. [`RowGrid::for_rows`] and
/// [`RowGrid::for_point`] then visit the rows of the points they are
/// given and store no pair. The grid holds the folded positions, the
/// bins, and two indices per point, so its size grows with the number of
/// points and bins and not with the number of pairs.
///
/// Both visits take `&self` and write nothing in the grid. Any number of
/// threads may visit one grid at the same time; disjoint point ranges
/// then cover each row once.
///
/// ```
/// use linkcell::{Cell, PbcQuery, RowGrid};
///
/// # fn main() -> Result<(), linkcell::Error> {
/// let sim = Cell::ortho(10.0, 10.0, 10.0)?;
/// let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0], [5.0, 5.0, 5.0]];
/// let grid = RowGrid::build(&xyz, &sim, &PbcQuery::periodic(1.0))?;
/// let mut rows = Vec::new();
/// grid.for_rows(0..3, false, |row| rows.push(*row))?;
/// assert_eq!(rows.len(), 2);
/// assert_eq!((rows[0].i, rows[0].j, rows[0].shift), (0, 1, [-1, 0, 0]));
/// assert_eq!((rows[1].i, rows[1].j, rows[1].shift), (1, 0, [1, 0, 0]));
/// assert!((rows[0].vector[0] + 0.8).abs() < 1e-12);
/// assert!((rows[0].dist2 - 0.64).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
pub struct RowGrid {
    points: usize,
    cutoff: f64,
    periodic: [bool; 3],
    kind: Rows,
}

enum Rows {
    /// No point is active.
    Empty,
    Periodic(Box<PeriodicRows>),
    Mixed(Box<MixedRows>),
}

impl RowGrid {
    /// Bin `xyz` for the rows of `query`.
    ///
    /// `query.periodic`, `query.cutoff`, `query.mask`, `query.cell_hint`,
    /// `query.closed`, and `query.exclude_self_images` mean what they do
    /// in [`pairs_within_pbc`](crate::pairs_within_pbc). `query.half` is
    /// not read: each visit says which side it wants. `cell` is not read
    /// when every axis is open.
    ///
    /// The errors are those of
    /// [`pairs_within_pbc`](crate::pairs_within_pbc) for the same
    /// arguments, less the ones that only a row buffer can raise.
    pub fn build(xyz: &[[f64; 3]], cell: &Cell, query: &PbcQuery<'_>) -> Result<Self, Error> {
        let mut grid = RowGrid {
            points: 0,
            cutoff: query.cutoff,
            periodic: query.periodic,
            kind: Rows::Empty,
        };
        grid.update(xyz, cell, query)?;
        Ok(grid)
    }

    /// Bin new points, a new cell, or a new query in place.
    ///
    /// The previous bins are released first, so the new ones reuse their
    /// buffers. After an error the grid holds no point and every row is
    /// out of range.
    pub fn update(
        &mut self,
        xyz: &[[f64; 3]],
        cell: &Cell,
        query: &PbcQuery<'_>,
    ) -> Result<(), Error> {
        // A periodic grid lends its columns to the next one when it drops.
        let (slot_of, bin_of) = match std::mem::replace(&mut self.kind, Rows::Empty) {
            Rows::Periodic(old) => {
                let old = *old;
                (old.slot_of, old.bin_of)
            }
            _ => (Vec::new(), Vec::new()),
        };
        self.points = 0;
        self.cutoff = query.cutoff;
        self.periodic = query.periodic;
        check(xyz, query)?;
        self.kind = if query.periodic == [true, true, true] {
            if periodic_face_collapsed(cell, query.periodic) {
                return Err(Error::Degenerate);
            }
            match PeriodicRows::build(xyz, cell, query, slot_of, bin_of)? {
                Some(rows) => Rows::Periodic(Box::new(rows)),
                None => Rows::Empty,
            }
        } else {
            match MixedRows::build(xyz, cell, query)? {
                Some(rows) => Rows::Mixed(Box::new(rows)),
                None => Rows::Empty,
            }
        };
        self.points = xyz.len();
        Ok(())
    }

    /// Release the bins. The grid then holds no point and every row is
    /// out of range, until the next [`RowGrid::update`].
    pub fn clear(&mut self) {
        self.kind = Rows::Empty;
        self.points = 0;
    }

    /// Number of points binned, masked ones included. Rows are numbered
    /// `0..points()`.
    pub fn points(&self) -> usize {
        self.points
    }

    /// Number of bins held.
    pub fn bins(&self) -> usize {
        match &self.kind {
            Rows::Empty => 0,
            Rows::Periodic(rows) => rows.binned.grid.offsets.len() - 1,
            Rows::Mixed(rows) => rows.bins(),
        }
    }

    /// An upper bound on the entries of every full row together, from the
    /// bin occupancy alone: for each point, the points in the bins its
    /// row reads, less itself. No distance is computed. A caller that
    /// must decide whether a stored list fits in memory reads this
    /// instead of counting the pairs. A half list holds at most half of
    /// it, rounded up. The sum saturates at `u64::MAX`.
    ///
    /// Returns [`Error::TooManyImages`] when a bin's partner runs cannot
    /// be held.
    pub fn candidate_bound(&self) -> Result<u64, Error> {
        match &self.kind {
            Rows::Empty => Ok(0),
            Rows::Periodic(rows) => rows.candidates(),
            Rows::Mixed(rows) => Ok(rows.candidates()),
        }
    }

    /// Cutoff the rows are tested against.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Axes that take images.
    pub fn periodic(&self) -> [bool; 3] {
        self.periodic
    }

    /// Visit the rows of the points in `rows`, in ascending point order.
    ///
    /// `visit` is called once for every image of `j` within the cutoff of
    /// `i`, with that entry as a [`Row`]. With `half` unset a row holds
    /// every such entry, so a pair is visited from both of its points,
    /// with opposite shifts and vectors and the same `dist2`. With `half`
    /// set a row keeps the entries the half list of
    /// [`pairs_within_pbc`](crate::pairs_within_pbc) lists under `i`:
    /// `j > i`, and the self images whose first non-zero shift is
    /// negative.
    ///
    /// The entries visited over `0..points()` are the rows of
    /// [`pairs_within_pbc`](crate::pairs_within_pbc) for the query the
    /// grid was built with, with the same shifts and the same bits in
    /// every `dist2`. Inside one row the entries come in bin order: the
    /// order of the bins around the point and of the points inside each
    /// bin. It is not sorted by `j`. It depends on the grid alone, so
    /// splitting a range does not change what any row holds or the order
    /// inside it. A masked point has an empty row.
    ///
    /// Returns [`Error::IndexOutOfRange`] before any visit when `rows`
    /// runs past [`RowGrid::points`], [`Error::Overflow`] when a shift
    /// does not fit `i32`, and [`Error::TooManyImages`] when a bin's
    /// partner runs cannot be held. A range whose start is not below its
    /// end is empty.
    pub fn for_rows(
        &self,
        rows: Range<usize>,
        half: bool,
        mut visit: impl FnMut(&Row),
    ) -> Result<(), Error> {
        let walked = self.try_for_rows(rows, half, |row| {
            visit(row);
            ControlFlow::<()>::Continue(())
        })?;
        debug_assert!(walked.is_continue());
        Ok(())
    }

    /// [`RowGrid::for_rows`] with a visit that can stop the walk.
    ///
    /// `visit` returns [`ControlFlow::Break`] to stop. No entry is
    /// visited after that, and the value comes back as
    /// `Ok(ControlFlow::Break(value))`. A walk that visits every entry
    /// returns `Ok(ControlFlow::Continue(()))`.
    pub fn try_for_rows<B>(
        &self,
        rows: Range<usize>,
        half: bool,
        mut visit: impl FnMut(&Row) -> ControlFlow<B>,
    ) -> Result<ControlFlow<B>, Error> {
        if rows.start >= rows.end {
            return Ok(ControlFlow::Continue(()));
        }
        if rows.end > self.points {
            return Err(Error::IndexOutOfRange);
        }
        let mut scratch = Scratch::default();
        for i in rows {
            if let ControlFlow::Break(stop) = self.row(i, half, &mut scratch, &mut visit)? {
                return Ok(ControlFlow::Break(stop));
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    /// Visit the row of point `i`: every entry [`RowGrid::for_rows`]
    /// gives that point, in the same order.
    ///
    /// Returns [`Error::IndexOutOfRange`] when `i` is not below
    /// [`RowGrid::points`], and otherwise the errors of
    /// [`RowGrid::for_rows`].
    pub fn for_point(
        &self,
        i: usize,
        half: bool,
        mut visit: impl FnMut(&Row),
    ) -> Result<(), Error> {
        let walked = self.try_for_point(i, half, |row| {
            visit(row);
            ControlFlow::<()>::Continue(())
        })?;
        debug_assert!(walked.is_continue());
        Ok(())
    }

    /// [`RowGrid::for_point`] with a visit that can stop the walk, as in
    /// [`RowGrid::try_for_rows`].
    pub fn try_for_point<B>(
        &self,
        i: usize,
        half: bool,
        mut visit: impl FnMut(&Row) -> ControlFlow<B>,
    ) -> Result<ControlFlow<B>, Error> {
        if i >= self.points {
            return Err(Error::IndexOutOfRange);
        }
        self.row(i, half, &mut Scratch::default(), &mut visit)
    }

    /// Copy row entries into `out`, from `cursor` up to the row of point
    /// `end`, and move `cursor` past them. Returns how many were written.
    ///
    /// The entries are those of [`RowGrid::for_rows`] over
    /// `cursor.point..end`, less the first `cursor.entry` of the first
    /// row, in the same order. The read stops when `out` is full or the
    /// rows are exhausted; it is done when `cursor.point == end`. A
    /// caller that walks many rows calls this in a loop and handles a
    /// whole buffer at a time, with one call per buffer instead of one
    /// per entry. A read that resumes inside a row walks that row again
    /// to its place, so a buffer shorter than a row costs that row once
    /// per call.
    ///
    /// Returns [`Error::IndexOutOfRange`] when `end` is past
    /// [`RowGrid::points`] or `cursor.point` is past `end`, and otherwise
    /// the errors of [`RowGrid::for_rows`]. An empty `out` reads nothing.
    pub fn fill_rows(
        &self,
        cursor: &mut RowCursor,
        end: usize,
        half: bool,
        out: &mut [Row],
    ) -> Result<usize, Error> {
        if end > self.points || cursor.point > end {
            return Err(Error::IndexOutOfRange);
        }
        let mut written = 0;
        let mut scratch = Scratch::default();
        while cursor.point < end && written < out.len() {
            let skip = cursor.entry;
            let mut seen = 0usize;
            let room = &mut out[written..];
            let mut taken = 0usize;
            let walked = self.row(cursor.point, half, &mut scratch, &mut |row: &Row| {
                if seen < skip {
                    seen += 1;
                    return ControlFlow::Continue(());
                }
                if taken == room.len() {
                    return ControlFlow::Break(());
                }
                room[taken] = *row;
                taken += 1;
                seen += 1;
                ControlFlow::Continue(())
            })?;
            written += taken;
            if walked.is_break() {
                cursor.entry = seen;
                break;
            }
            cursor.point += 1;
            cursor.entry = 0;
        }
        Ok(written)
    }

    /// The row of point `i`, which is in range.
    fn row<B>(
        &self,
        i: usize,
        half: bool,
        scratch: &mut Scratch,
        visit: &mut impl FnMut(&Row) -> ControlFlow<B>,
    ) -> Result<ControlFlow<B>, Error> {
        let mut stop = None;
        let mut entry = |j: usize, shift: [i32; 3], vector: [f64; 3], dist2: f64| {
            let row = Row {
                i,
                j,
                shift,
                vector,
                dist2,
            };
            match visit(&row) {
                ControlFlow::Continue(()) => true,
                ControlFlow::Break(value) => {
                    stop = Some(value);
                    false
                }
            }
        };
        match &self.kind {
            Rows::Empty => {}
            Rows::Periodic(grid) => {
                grid.row(i, half, scratch, &mut entry)?;
            }
            Rows::Mixed(grid) => {
                grid.row(i, half, &mut entry)?;
            }
        }
        Ok(match stop {
            Some(value) => ControlFlow::Break(value),
            None => ControlFlow::Continue(()),
        })
    }
}

/// The argument checks of [`pairs_within_pbc`](crate::pairs_within_pbc).
fn check(xyz: &[[f64; 3]], query: &PbcQuery<'_>) -> Result<(), Error> {
    if !query.cutoff.is_finite() || query.cutoff <= 0.0 {
        return Err(Error::BadCutoff);
    }
    if xyz.is_empty() {
        return Err(Error::Empty);
    }
    if let Some(mask) = query.mask {
        if mask.len() != xyz.len() {
            return Err(Error::MaskLen);
        }
    }
    for point in xyz {
        if !point[0].is_finite() || !point[1].is_finite() || !point[2].is_finite() {
            return Err(Error::NonFinite);
        }
    }
    Ok(())
}

/// The runs of the bin a visit read last, for a mesh with no tabulated
/// runs. Each visit owns one, so the grid stays shared.
#[derive(Default)]
struct Scratch {
    bin: Option<usize>,
    axes: [Vec<(i32, i32)>; 3],
    steps: Vec<Step>,
}

/// A fully periodic grid: the bins of the pair walk, the runs of the CSR
/// gather, and where each point sits.
struct PeriodicRows {
    binned: Binned,
    cell: Cell,
    images: ImageOffsets,
    /// Bin widths, as the stencil takes them.
    width: [f64; 3],
    shifts: Shifts,
    /// The runs of a bin by where it sits on each axis, or empty when
    /// they depend on the bin (see `Walker::new`).
    steps: Vec<Vec<Step>>,
    /// Slot of each point in the grid columns, or [`MASKED`].
    slot_of: Vec<u32>,
    /// Bin of each point.
    bin_of: Vec<u32>,
    exclude_self: bool,
}

impl PeriodicRows {
    /// `None` when no point is active. `slot_of` and `bin_of` are
    /// buffers to reuse.
    fn build(
        xyz: &[[f64; 3]],
        cell: &Cell,
        query: &PbcQuery<'_>,
        mut slot_of: Vec<u32>,
        mut bin_of: Vec<u32>,
    ) -> Result<Option<Self>, Error> {
        let n = xyz.len();
        // A closed cutoff is binned one double wider, so the reach and
        // the runs cover the boundary, and its rows are tested against
        // the next double above `cutoff²`; see `next_up`.
        let cutoff = if query.closed {
            next_up(query.cutoff)
        } else {
            query.cutoff
        };
        let est = hit_estimate(n, cell, cutoff);
        let binned = in_pool(est, || {
            bin_atoms(xyz, cell, cutoff, query.mask, query.cell_hint, true)
        })?;
        let Some(mut binned) = binned else {
            return Ok(None);
        };
        if query.closed {
            binned.cut2 = next_up(query.cutoff * query.cutoff);
        }
        let images = ImageOffsets::new(xyz, cell, binned.active.as_deref(), binned.repeats)?;
        let (width, shifts, steps) = Walker::new(&binned, cell)?.into_tables();
        slot_of.clear();
        bin_of.clear();
        try_hold(&mut slot_of, n)?;
        try_hold(&mut bin_of, n)?;
        slot_of.resize(n, MASKED);
        bin_of.resize(n, 0);
        let grid = &binned.grid;
        for c in 0..grid.offsets.len() - 1 {
            for s in grid.offsets[c]..grid.offsets[c + 1] {
                let i = grid.coords.id[s] as usize;
                slot_of[i] = s as u32;
                bin_of[i] = c as u32;
            }
        }
        Ok(Some(PeriodicRows {
            binned,
            cell: *cell,
            images,
            width,
            shifts,
            steps,
            slot_of,
            bin_of,
            exclude_self: query.exclude_self_images,
        }))
    }

    /// Candidates every row tests, summed over the points: for each
    /// point, the slots of the runs of its bin less the point itself.
    fn candidates(&self) -> Result<u64, Error> {
        let grid = &self.binned.grid;
        let off = &grid.offsets;
        let [nx, ny, _] = grid.n;
        let mut scratch = Scratch::default();
        let mut total = 0u64;
        for c in 0..off.len() - 1 {
            let here = (off[c + 1] - off[c]) as u64;
            if here == 0 {
                continue;
            }
            let b = [
                (c % nx as usize) as i32,
                ((c / nx as usize) % ny as usize) as i32,
                (c / (nx as usize * ny as usize)) as i32,
            ];
            let mut around = 0u64;
            for step in self.steps_of(c, b, &mut scratch)? {
                let first = (c as isize + step.rel) as usize;
                around += (off[first + step.bins] - off[first]) as u64;
            }
            total = total.saturating_add(here.saturating_mul(around.saturating_sub(1)));
        }
        Ok(total)
    }

    /// The runs of bin `c` at `b`: tabulated, or built into `scratch`.
    fn steps_of<'s>(
        &'s self,
        c: usize,
        b: [i32; 3],
        scratch: &'s mut Scratch,
    ) -> Result<&'s [Step], Error> {
        let grid = &self.binned.grid;
        if !self.steps.is_empty() {
            return Ok(&self.steps[bin_class(b, grid.n)]);
        }
        if scratch.bin != Some(c) {
            scratch.bin = None;
            let reach = Reach {
                n: grid.n,
                reach: self.binned.reach,
                ortho: self.cell.is_ortho(),
                width: self.width,
                cut2: self.binned.cut2,
                shifts: &self.shifts,
                cell: &self.cell,
            };
            bin_steps(&reach, c, b, &mut scratch.axes, &mut scratch.steps)?;
            scratch.bin = Some(c);
        }
        Ok(&scratch.steps)
    }

    /// The row of point `i`, which is in range.
    ///
    /// Each run is read with the expression the CSR gather reads it
    /// with, which is the pair walk's for that pair or its negation, so
    /// `dist2` is the walk's bits. A source run reports its own shift. A
    /// target run holds the walk's shift for the reversed pair, so the
    /// row reports the opposite one. In both the difference the row
    /// forms is `r_j - r_i` with the reported shift applied. `visit`
    /// returns `false` to stop; the result is `false` when it did.
    fn row(
        &self,
        i: usize,
        half: bool,
        scratch: &mut Scratch,
        visit: &mut impl FnMut(usize, [i32; 3], [f64; 3], f64) -> bool,
    ) -> Result<bool, Error> {
        let s = self.slot_of[i];
        if s == MASKED {
            return Ok(true);
        }
        let s = s as usize;
        let grid = &self.binned.grid;
        let off = &grid.offsets;
        let c = self.bin_of[i] as usize;
        let [nx, ny, _] = grid.n;
        let b = [
            (c % nx as usize) as i32,
            ((c / nx as usize) % ny as usize) as i32,
            (c / (nx as usize * ny as usize)) as i32,
        ];
        let steps = self.steps_of(c, b, scratch)?;
        let cut2 = self.binned.cut2;
        let co = &grid.coords;
        let p = [co.x[s], co.y[s], co.z[s]];
        for step in steps {
            let first = (c as isize + step.rel) as usize;
            let (lo, hi) = (off[first], off[first + step.bins]);
            if lo >= hi {
                continue;
            }
            let (probe, wrapped) = match step.kind {
                Kind::Plain => (p, [0; 3]),
                Kind::Source => (
                    [
                        p[0] - step.shift[0],
                        p[1] - step.shift[1],
                        p[2] - step.shift[2],
                    ],
                    step.s,
                ),
                Kind::Target => (p, [-step.s[0], -step.s[1], -step.s[2]]),
            };
            for t in lo..hi {
                let mut q = [co.x[t], co.y[t], co.z[t]];
                if step.kind == Kind::Target {
                    q = [
                        q[0] - step.shift[0],
                        q[1] - step.shift[1],
                        q[2] - step.shift[2],
                    ];
                }
                let (dx, dy, dz) = (q[0] - probe[0], q[1] - probe[1], q[2] - probe[2]);
                let dist2 = dx * dx + dy * dy + dz * dz;
                if dist2 >= cut2 {
                    continue;
                }
                let j = co.id[t] as usize;
                if j == i && (step.kind == Kind::Plain || self.exclude_self) {
                    continue;
                }
                if half && j < i {
                    continue;
                }
                let shift = self.images.shift(i, j, wrapped);
                if half && j == i && !keep_half(i, j, shift) {
                    continue;
                }
                if !visit(j, shift, [dx, dy, dz], dist2) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_shared<T: Send + Sync>() {}

    #[test]
    fn a_grid_is_shared_between_threads() {
        assert_shared::<RowGrid>();
    }
}
