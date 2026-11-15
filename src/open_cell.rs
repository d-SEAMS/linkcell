//! Cutoff pairs on open axes.
//!
//! [`crate::pairs_within`](crate::pairs_within) is a periodic list. An
//! open axis has no images. Padding that axis out to a periodic box
//! makes the partner stencil depend on the bounding span, so moving one
//! atom rebuilds it. This walk bins the occupied Cartesian cells and
//! keeps that stencil until the bounding bins change.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::bins::certify;
use crate::pairs::{keep_half, Pair};
use crate::{Cell, Error};

const MAX_CELLS: i64 = 16_777_216;
const MAX_IMAGES: i64 = 1 << 24;

static STENCIL_BUILDS: AtomicU64 = AtomicU64::new(0);
static CACHE: Mutex<Option<Arc<Stencil>>> = Mutex::new(None);

/// How many partner stencils have been built since the process started.
///
/// A reused stencil does not increment the count. The periodic list and
/// the open-cell list share the counter.
pub fn stencil_builds() -> u64 {
    STENCIL_BUILDS.load(Ordering::Relaxed)
}

pub(crate) fn note_stencil_build() {
    STENCIL_BUILDS.fetch_add(1, Ordering::Relaxed);
}

/// Options for [`pairs_within_pbc`].
pub struct PbcQuery<'a> {
    /// `periodic[k]` is false when lattice axis `k` is open.
    pub periodic: [bool; 3],
    /// Pairs with squared distance below `cutoff` squared are kept.
    /// [`Self::closed`] also keeps the boundary.
    pub cutoff: f64,
    /// `None` keeps every point. A `false` entry drops that point as
    /// both source and candidate.
    pub mask: Option<&'a [bool]>,
    /// Bin edge. `None` uses the cutoff, which does not move when the
    /// bounding span jitters inside a bin.
    pub cell_hint: Option<f64>,
    /// Keep one side of `(i, j, S)`.
    pub half: bool,
    /// Also keep `dist2 == cutoff²` on an open or mixed cell.
    pub closed: bool,
    /// Drop rows with `i == j`, including a periodic image of the source.
    pub exclude_self_images: bool,
}

impl<'a> PbcQuery<'a> {
    /// Fully periodic search. The other flags start unset.
    pub fn periodic(cutoff: f64) -> Self {
        Self::new([true, true, true], cutoff)
    }

    /// Every axis open.
    pub fn open(cutoff: f64) -> Self {
        Self::new([false, false, false], cutoff)
    }

    /// `periodic` selects the axes that take images.
    pub fn new(periodic: [bool; 3], cutoff: f64) -> Self {
        Self {
            periodic,
            cutoff,
            mask: None,
            cell_hint: None,
            half: false,
            closed: false,
            exclude_self_images: false,
        }
    }

    /// Set the point mask.
    pub fn mask(mut self, mask: Option<&'a [bool]>) -> Self {
        self.mask = mask;
        self
    }

    /// Set the bin edge.
    pub fn cell_hint(mut self, cell_hint: Option<f64>) -> Self {
        self.cell_hint = cell_hint;
        self
    }

    /// Keep one side of each pair.
    pub fn half(mut self, half: bool) -> Self {
        self.half = half;
        self
    }

    /// Keep the cutoff boundary.
    pub fn closed(mut self, closed: bool) -> Self {
        self.closed = closed;
        self
    }

    /// Drop `i == j`.
    pub fn exclude_self_images(mut self, exclude: bool) -> Self {
        self.exclude_self_images = exclude;
        self
    }
}

/// Cartesian half pair list of an open cluster.
///
/// Every axis is open. Row `(i, j, [0, 0, 0], dist2)` has `i < j` and
/// `dist2 < cutoff²`. The atoms are binned on the occupied Cartesian
/// cells. [`stencil_builds`] does not move while later calls stay inside
/// those bins.
///
/// ```
/// use linkcell::pairs_open;
///
/// let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
/// let rows = pairs_open(&xyz, 1.5, None).unwrap();
/// assert_eq!(rows.len(), 1);
/// assert_eq!((rows[0].i, rows[0].j, rows[0].shift), (0, 1, [0, 0, 0]));
/// ```
pub fn pairs_open(
    xyz: &[[f64; 3]],
    cutoff: f64,
    cell_hint: Option<f64>,
) -> Result<Vec<Pair>, Error> {
    let cell = Cell::ortho(1.0, 1.0, 1.0).map_err(|_| Error::BadBox)?;
    let query = PbcQuery::open(cutoff).half(true).cell_hint(cell_hint);
    pairs_within_pbc(xyz, &cell, &query)
}

/// Cutoff pairs with a periodicity mask.
///
/// `query.periodic[k]` false keeps axis `k` out of the image sum. A
/// fully periodic mask is [`crate::pairs_within`](crate::pairs_within).
/// Any open axis uses occupied Cartesian bins and no periodic padding.
/// `cell` supplies the periodic lattice. It is not read when every axis
/// is open.
pub fn pairs_within_pbc(
    xyz: &[[f64; 3]],
    cell: &Cell,
    query: &PbcQuery<'_>,
) -> Result<Vec<Pair>, Error> {
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
    if query.periodic == [true, true, true] {
        let mut rows = crate::pairs::pairs_within(
            xyz,
            cell,
            query.cutoff,
            query.mask,
            query.cell_hint,
            query.half,
        )?;
        if query.exclude_self_images {
            rows.retain(|pair| pair.i != pair.j);
        }
        return Ok(rows);
    }
    let frame = place(xyz, cell, query)?;
    if frame.n_active == 0 {
        return Ok(Vec::new());
    }
    let stencil = stencil_for(&frame, cell, query)?;
    walk(&frame, cell, &stencil, query)
}

/// Lattice for `periodic`. A zero row on an open axis is replaced by a
/// unit axis so the periodic rows still define a cell. A fully open
/// mask does not read `rows`.
pub fn cell_for_pbc(
    mut rows: [[f64; 3]; 3],
    origin: [f64; 3],
    periodic: [bool; 3],
) -> Result<Cell, Error> {
    if periodic == [false, false, false] {
        return Cell::ortho(1.0, 1.0, 1.0).map_err(|_| Error::BadBox);
    }
    for axis in 0..3 {
        if periodic[axis] {
            continue;
        }
        let len2 = rows[axis][0] * rows[axis][0]
            + rows[axis][1] * rows[axis][1]
            + rows[axis][2] * rows[axis][2];
        if !len2.is_finite() || len2 <= 1.0e-24 {
            rows[axis] = [0.0, 0.0, 0.0];
            rows[axis][axis] = 1.0;
        }
    }
    Cell::from_vectors(rows[0], rows[1], rows[2], origin).map_err(|_| Error::BadBox)
}

struct Frame {
    pos: Vec<[f64; 3]>,
    image: Vec<[i32; 3]>,
    off: Vec<u32>,
    occ: Vec<u32>,
    origin: [i32; 3],
    dims: [i32; 3],
    edge: f64,
    n_active: usize,
    any_image: bool,
}

struct Link {
    dst: u32,
    shift: [i32; 3],
}

struct Stencil {
    key: Key,
    off: Vec<u32>,
    links: Vec<Link>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    edge: u64,
    cut: u64,
    periodic: [u8; 3],
    origin: [i32; 3],
    dims: [i32; 3],
    repeats: [i32; 3],
    box_bits: [u64; 12],
}

fn bin_edge(hint: Option<f64>, cutoff: f64) -> f64 {
    match hint {
        Some(edge) if edge.is_finite() && edge > 0.0 => edge,
        _ => cutoff,
    }
}

fn floor_i32(value: f64) -> Result<i32, Error> {
    if !value.is_finite() {
        return Err(Error::NonFinite);
    }
    let floored = value.floor();
    if floored < i32::MIN as f64 || floored > i32::MAX as f64 {
        return Err(Error::Overflow);
    }
    Ok(floored as i32)
}

fn mul_columns(h: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        h[0][0] * v[0] + h[1][0] * v[1] + h[2][0] * v[2],
        h[0][1] * v[0] + h[1][1] * v[1] + h[2][1] * v[2],
        h[0][2] * v[0] + h[1][2] * v[1] + h[2][2] * v[2],
    ]
}

fn unwrapped_frac(cell: &Cell, point: [f64; 3]) -> [f64; 3] {
    let origin = cell.origin();
    if cell.is_ortho() {
        let widths = cell.widths();
        return [
            (point[0] - origin[0]) / widths[0],
            (point[1] - origin[1]) / widths[1],
            (point[2] - origin[2]) / widths[2],
        ];
    }
    let delta = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    mul_columns(cell.hinv(), delta)
}

fn repeats_of(cell: &Cell, periodic: [bool; 3], cutoff: f64) -> Result<[i32; 3], Error> {
    let widths = cell.widths();
    let mut repeats = [0i32; 3];
    let mut images = 1i64;
    for axis in 0..3 {
        if !periodic[axis] {
            continue;
        }
        if !widths[axis].is_finite() || widths[axis] <= 0.0 {
            return Err(Error::BadBox);
        }
        let count = (cutoff / widths[axis]).ceil();
        if !count.is_finite() || count > i32::MAX as f64 {
            return Err(Error::TooManyImages);
        }
        repeats[axis] = count as i32;
        let factor = 2 * i64::from(repeats[axis]) + 1;
        if images > MAX_IMAGES / factor {
            return Err(Error::TooManyImages);
        }
        images *= factor;
    }
    Ok(repeats)
}

fn place(xyz: &[[f64; 3]], cell: &Cell, query: &PbcQuery<'_>) -> Result<Frame, Error> {
    let edge = bin_edge(query.cell_hint, query.cutoff);
    let periodic = query.periodic;
    let open = periodic == [false, false, false];
    let n = xyz.len();
    let mut pos = vec![[0.0; 3]; n];
    let mut image = vec![[0i32; 3]; n];
    let mut bins = vec![[0i32; 3]; n];
    let mut active = Vec::with_capacity(n);
    let mut any_image = false;
    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    for (index, point) in xyz.iter().enumerate() {
        if query.mask.is_some_and(|mask| !mask[index]) {
            continue;
        }
        let located = if open {
            *point
        } else {
            let frac = unwrapped_frac(cell, *point);
            let mut folded = frac;
            for axis in 0..3 {
                if !periodic[axis] {
                    continue;
                }
                let whole = floor_i32(frac[axis])?;
                image[index][axis] = whole;
                if whole != 0 {
                    any_image = true;
                }
                folded[axis] = frac[axis] - f64::from(whole);
            }
            cell.cartesian(folded)
        };
        if !located[0].is_finite() || !located[1].is_finite() || !located[2].is_finite() {
            return Err(Error::NonFinite);
        }
        pos[index] = *point;
        for axis in 0..3 {
            let bin = floor_i32(located[axis] / edge)?;
            bins[index][axis] = bin;
            lo[axis] = lo[axis].min(bin);
            hi[axis] = hi[axis].max(bin);
        }
        active.push(index);
    }
    if active.is_empty() {
        return Ok(Frame {
            pos,
            image,
            off: vec![0],
            occ: Vec::new(),
            origin: [0; 3],
            dims: [1, 1, 1],
            edge,
            n_active: 0,
            any_image: false,
        });
    }
    let mut dims = [0i32; 3];
    let mut ncell: i64 = 1;
    for axis in 0..3 {
        let span = i64::from(hi[axis]) - i64::from(lo[axis]) + 1;
        if span <= 0 || span > i32::MAX as i64 {
            return Err(Error::TooManyCells);
        }
        dims[axis] = span as i32;
        ncell = ncell.saturating_mul(span);
        if ncell > MAX_CELLS {
            return Err(Error::TooManyCells);
        }
    }
    let ncell = ncell as usize;
    let mut count = vec![0u32; ncell];
    let mut which = vec![0u32; active.len()];
    for (slot, &index) in active.iter().enumerate() {
        let local = [
            bins[index][0] - lo[0],
            bins[index][1] - lo[1],
            bins[index][2] - lo[2],
        ];
        let flat = ((local[2] as usize * dims[1] as usize) + local[1] as usize) * dims[0] as usize
            + local[0] as usize;
        which[slot] = flat as u32;
        count[flat] += 1;
    }
    let mut off = vec![0u32; ncell + 1];
    for cell_index in 0..ncell {
        off[cell_index + 1] = off[cell_index] + count[cell_index];
    }
    let mut cursor = off.clone();
    let mut occ = vec![0u32; active.len()];
    for (slot, &index) in active.iter().enumerate() {
        let at = cursor[which[slot] as usize];
        occ[at as usize] = index as u32;
        cursor[which[slot] as usize] = at + 1;
    }
    Ok(Frame {
        pos,
        image,
        off,
        occ,
        origin: lo,
        dims,
        edge,
        n_active: active.len(),
        any_image,
    })
}

fn box_bits(cell: &Cell, periodic: [bool; 3]) -> [u64; 12] {
    if periodic == [false, false, false] {
        return [0; 12];
    }
    let h = cell.h();
    let origin = cell.origin();
    let mut bits = [0u64; 12];
    for axis in 0..3 {
        if !periodic[axis] {
            continue;
        }
        for component in 0..3 {
            bits[3 * axis + component] = h[axis][component].to_bits();
        }
        bits[9 + axis] = origin[axis].to_bits();
    }
    bits
}

fn axis_gap(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    if a1 <= b0 {
        b0 - a1
    } else if b1 <= a0 {
        a0 - b1
    } else {
        0.0
    }
}

fn cells_too_far(
    src: [i32; 3],
    dst: [i32; 3],
    origin: [i32; 3],
    edge: f64,
    delta: [f64; 3],
    cutoff: f64,
) -> bool {
    let mut gap2 = 0.0;
    for axis in 0..3 {
        let s0 = f64::from(origin[axis] + src[axis]) * edge;
        let d0 = f64::from(origin[axis] + dst[axis]) * edge + delta[axis];
        let gap = axis_gap(s0, s0 + edge, d0, d0 + edge);
        gap2 += gap * gap;
    }
    let gap = certify(gap2.sqrt());
    gap > cutoff
}

fn local_of(flat: usize, dims: [i32; 3]) -> [i32; 3] {
    let nx = dims[0] as usize;
    let ny = dims[1] as usize;
    let x = (flat % nx) as i32;
    let y = ((flat / nx) % ny) as i32;
    let z = (flat / (nx * ny)) as i32;
    [x, y, z]
}

fn flat_of(local: [i32; 3], dims: [i32; 3]) -> Option<usize> {
    if local[0] < 0 || local[1] < 0 || local[2] < 0 {
        return None;
    }
    if local[0] >= dims[0] || local[1] >= dims[1] || local[2] >= dims[2] {
        return None;
    }
    Some(
        ((local[2] as usize * dims[1] as usize) + local[1] as usize) * dims[0] as usize
            + local[0] as usize,
    )
}

fn build_stencil(
    frame: &Frame,
    cell: &Cell,
    query: &PbcQuery<'_>,
    key: Key,
) -> Result<Stencil, Error> {
    let repeats = key.repeats;
    let ncell = frame.off.len() - 1;
    let edge = frame.edge;
    let reach = (query.cutoff / edge).ceil();
    // `reach as i32 + 2` is the stencil pad. A value past `i32::MAX - 2`
    // overflows that add in debug and wraps the image window in release.
    if !reach.is_finite() || reach > (i32::MAX - 2) as f64 {
        return Err(Error::TooManyImages);
    }
    let reach = reach as i32 + 2;
    let mut shifts = Vec::new();
    for sa in -repeats[0]..=repeats[0] {
        for sb in -repeats[1]..=repeats[1] {
            for sc in -repeats[2]..=repeats[2] {
                let shift = [sa, sb, sc];
                let delta = if shift == [0, 0, 0] {
                    [0.0; 3]
                } else {
                    cell.lattice_shift(sa, sb, sc)
                };
                shifts.push((shift, delta));
            }
        }
    }
    let scan_all = (ncell as u64).saturating_mul(shifts.len() as u64) <= 250_000;
    let mut off = vec![0u32; ncell + 1];
    let mut links = Vec::new();
    for src in 0..ncell {
        let src_local = local_of(src, frame.dims);
        for &(shift, delta) in &shifts {
            if scan_all {
                for dst in 0..ncell {
                    let dst_local = local_of(dst, frame.dims);
                    if cells_too_far(
                        src_local,
                        dst_local,
                        frame.origin,
                        edge,
                        delta,
                        query.cutoff,
                    ) {
                        continue;
                    }
                    links.push(Link {
                        dst: dst as u32,
                        shift,
                    });
                }
            } else {
                let center = [
                    src_local[0] - (delta[0] / edge).round() as i32,
                    src_local[1] - (delta[1] / edge).round() as i32,
                    src_local[2] - (delta[2] / edge).round() as i32,
                ];
                for dz in -reach..=reach {
                    for dy in -reach..=reach {
                        for dx in -reach..=reach {
                            let dst_local = [center[0] + dx, center[1] + dy, center[2] + dz];
                            let Some(dst) = flat_of(dst_local, frame.dims) else {
                                continue;
                            };
                            if cells_too_far(
                                src_local,
                                dst_local,
                                frame.origin,
                                edge,
                                delta,
                                query.cutoff,
                            ) {
                                continue;
                            }
                            links.push(Link {
                                dst: dst as u32,
                                shift,
                            });
                        }
                    }
                }
            }
        }
        off[src + 1] = links.len() as u32;
    }
    Ok(Stencil { key, off, links })
}

fn stencil_for(frame: &Frame, cell: &Cell, query: &PbcQuery<'_>) -> Result<Arc<Stencil>, Error> {
    let repeats = if query.periodic == [false, false, false] {
        [0; 3]
    } else {
        repeats_of(cell, query.periodic, query.cutoff)?
    };
    let key = Key {
        edge: frame.edge.to_bits(),
        cut: query.cutoff.to_bits(),
        periodic: [
            u8::from(query.periodic[0]),
            u8::from(query.periodic[1]),
            u8::from(query.periodic[2]),
        ],
        origin: frame.origin,
        dims: frame.dims,
        repeats,
        box_bits: box_bits(cell, query.periodic),
    };
    {
        let guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(stencil) = guard.as_ref() {
            if stencil.key == key {
                return Ok(Arc::clone(stencil));
            }
        }
    }
    let built = Arc::new(build_stencil(frame, cell, query, key)?);
    note_stencil_build();
    let mut guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
    *guard = Some(Arc::clone(&built));
    Ok(built)
}

fn reported_shift(frame: &Frame, i: usize, j: usize, shift: [i32; 3]) -> Result<[i32; 3], Error> {
    if !frame.any_image {
        return Ok(shift);
    }
    let mut out = [0i32; 3];
    for axis in 0..3 {
        let value = i64::from(shift[axis]) + i64::from(frame.image[i][axis])
            - i64::from(frame.image[j][axis]);
        if value < i64::from(i32::MIN) || value > i64::from(i32::MAX) {
            return Err(Error::Overflow);
        }
        out[axis] = value as i32;
    }
    Ok(out)
}

fn neg_shift(shift: [i32; 3]) -> Option<[i32; 3]> {
    Some([
        shift[0].checked_neg()?,
        shift[1].checked_neg()?,
        shift[2].checked_neg()?,
    ])
}

fn accept(dist2: f64, cut2: f64, closed: bool) -> bool {
    if closed {
        dist2 <= cut2
    } else {
        dist2 < cut2
    }
}

fn walk(
    frame: &Frame,
    cell: &Cell,
    stencil: &Stencil,
    query: &PbcQuery<'_>,
) -> Result<Vec<Pair>, Error> {
    let cut2 = query.cutoff * query.cutoff;
    let mut rows = Vec::new();
    let ncell = frame.off.len() - 1;
    for src in 0..ncell {
        let src_atoms = &frame.occ[frame.off[src] as usize..frame.off[src + 1] as usize];
        if src_atoms.is_empty() {
            continue;
        }
        for link in &stencil.links[stencil.off[src] as usize..stencil.off[src + 1] as usize] {
            let dst_atoms = &frame.occ
                [frame.off[link.dst as usize] as usize..frame.off[link.dst as usize + 1] as usize];
            if dst_atoms.is_empty() {
                continue;
            }
            for &ia in src_atoms {
                let i = ia as usize;
                let pi = frame.pos[i];
                for &ja in dst_atoms {
                    let j = ja as usize;
                    if i == j && link.shift == [0, 0, 0] {
                        continue;
                    }
                    let reported = reported_shift(frame, i, j, link.shift)?;
                    if i > j {
                        continue;
                    }
                    if i == j && !keep_half(i, j, reported) {
                        continue;
                    }
                    if query.exclude_self_images && i == j {
                        continue;
                    }
                    let pj = frame.pos[j];
                    let delta = if reported == [0, 0, 0] {
                        [0.0; 3]
                    } else {
                        cell.lattice_shift(reported[0], reported[1], reported[2])
                    };
                    let dx = pj[0] + delta[0] - pi[0];
                    let dy = pj[1] + delta[1] - pi[1];
                    let dz = pj[2] + delta[2] - pi[2];
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    if !accept(dist2, cut2, query.closed) {
                        continue;
                    }
                    rows.push(Pair {
                        i,
                        j,
                        shift: reported,
                        dist2,
                    });
                    if !query.half {
                        let flipped = neg_shift(reported).ok_or(Error::Overflow)?;
                        if i == j {
                            rows.push(Pair {
                                i,
                                j,
                                shift: flipped,
                                dist2,
                            });
                        } else {
                            rows.push(Pair {
                                i: j,
                                j: i,
                                shift: flipped,
                                dist2,
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(rows)
}
