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

use std::ops::Range;

use super::gather::{bin_class, bin_steps, Kind, Reach, Shifts, Step, Walker};
use super::{bin_atoms, hit_estimate, in_pool, keep_half, try_hold, Binned, ImageOffsets};
use crate::cell::{periodic_face_collapsed, Cell};
use crate::open_cell::{MixedRows, PbcQuery};
use crate::Error;

/// Marks a point the mask dropped.
const MASKED: u32 = u32::MAX;

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
/// grid.for_rows(0..3, false, |i, j, shift, dist2| rows.push((i, j, shift, dist2)))?;
/// assert_eq!(rows.len(), 2);
/// assert_eq!((rows[0].0, rows[0].1, rows[0].2), (0, 1, [-1, 0, 0]));
/// assert_eq!((rows[1].0, rows[1].1, rows[1].2), (1, 0, [1, 0, 0]));
/// assert!((rows[0].3 - 0.64).abs() < 1e-12);
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
    /// `visit(i, j, shift, dist2)` is called once for every image of `j`
    /// within the cutoff of `i`: `shift` is the integer cell shift applied
    /// to `j` and `dist2` the squared distance after it, as in
    /// [`Pair`](crate::Pair). With `half` unset a row holds every such
    /// entry, so a pair is visited from both of its points, with opposite
    /// shifts and the same `dist2`. With `half` set a row keeps the
    /// entries the half list of
    /// [`pairs_within_pbc`](crate::pairs_within_pbc) lists under `i`:
    /// `j > i`, and the self images whose first non-zero shift is
    /// negative.
    ///
    /// The entries visited over `0..points()` are the rows of
    /// [`pairs_within_pbc`](crate::pairs_within_pbc) for the query the
    /// grid was built with, with the same shifts and the same bits in
    /// every `dist2`. Inside one row the entries come in bin order, which
    /// depends on the grid alone, so splitting a range does not change
    /// what any row holds or the order inside it. A masked point has an
    /// empty row.
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
        mut visit: impl FnMut(usize, usize, [i32; 3], f64),
    ) -> Result<(), Error> {
        if rows.start >= rows.end {
            return Ok(());
        }
        if rows.end > self.points {
            return Err(Error::IndexOutOfRange);
        }
        match &self.kind {
            Rows::Empty => Ok(()),
            Rows::Periodic(grid) => {
                let mut scratch = Scratch::default();
                for i in rows {
                    grid.row(i, half, &mut scratch, &mut |j, shift, dist2| {
                        visit(i, j, shift, dist2)
                    })?;
                }
                Ok(())
            }
            Rows::Mixed(grid) => {
                for i in rows {
                    grid.row(i, half, &mut |j, shift, dist2| visit(i, j, shift, dist2))?;
                }
                Ok(())
            }
        }
    }

    /// Visit the row of point `i`: `visit(j, shift, dist2)` for every
    /// entry [`RowGrid::for_rows`] gives that point, in the same order.
    ///
    /// Returns [`Error::IndexOutOfRange`] when `i` is not below
    /// [`RowGrid::points`], and otherwise the errors of
    /// [`RowGrid::for_rows`].
    pub fn for_point(
        &self,
        i: usize,
        half: bool,
        mut visit: impl FnMut(usize, [i32; 3], f64),
    ) -> Result<(), Error> {
        if i >= self.points {
            return Err(Error::IndexOutOfRange);
        }
        match &self.kind {
            Rows::Empty => Ok(()),
            Rows::Periodic(grid) => grid.row(i, half, &mut Scratch::default(), &mut visit),
            Rows::Mixed(grid) => grid.row(i, half, &mut visit),
        }
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
        let est = hit_estimate(n, cell, query.cutoff);
        let binned = in_pool(est, || {
            bin_atoms(xyz, cell, query.cutoff, query.mask, query.cell_hint, true)
        })?;
        let Some(binned) = binned else {
            return Ok(None);
        };
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

    /// The row of point `i`, which is in range.
    ///
    /// Each run is read with the expression the CSR gather reads it
    /// with, which is the pair walk's for that pair or its negation, so
    /// `dist2` is the walk's bits. A source run reports its own shift. A
    /// target run holds the walk's shift for the reversed pair, so the
    /// row reports the opposite one.
    fn row(
        &self,
        i: usize,
        half: bool,
        scratch: &mut Scratch,
        visit: &mut impl FnMut(usize, [i32; 3], f64),
    ) -> Result<(), Error> {
        let s = self.slot_of[i];
        if s == MASKED {
            return Ok(());
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
        let steps: &[Step] = if self.steps.is_empty() {
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
            &scratch.steps
        } else {
            &self.steps[bin_class(b, grid.n)]
        };
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
                visit(j, shift, dist2);
            }
        }
        Ok(())
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
