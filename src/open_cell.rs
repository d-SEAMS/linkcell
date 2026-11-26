//! Cutoff pairs on open axes.
//!
//! [`crate::pairs_within`](crate::pairs_within) is a periodic list. An
//! open axis has no images. Padding that axis out to a periodic box
//! makes the partner stencil depend on the bounding span, so moving one
//! atom rebuilds it. This walk bins the occupied Cartesian cells. With
//! every axis open the stencil is the reach in x of each row of bin
//! offsets, which depends on the bin edge and the cutoff alone, so
//! frames that occupy other bins reuse it. A periodic axis keeps the
//! stencil until the bounding bins change.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::bins::certify;
use crate::cell::periodic_face_collapsed;
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
/// cells. One partner stencil serves every call with the same cutoff and
/// bin edge, so [`stencil_builds`] does not move when later frames occupy
/// other bins. A reach past 16 bins can widen it when a frame spans more
/// bins on y or z than it covers.
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
/// `cell` supplies the periodic rows and the origin. An open row is not
/// read: [`cell_for_pbc`] replaces it with a unit normal of the periodic
/// rows before the bins are built. The cell is not read when every axis
/// is open. A fully periodic cell whose face width is zero or
/// non-finite returns [`Error::Degenerate`]. A positive finite width
/// uses the cutoff-aware image and bin caps.
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
        if periodic_face_collapsed(cell, query.periodic) {
            return Err(Error::Degenerate);
        }
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
    let binning = cell_for_pbc(cell.h(), cell.origin(), query.periodic)?;
    let cell = &binning;
    let frame = place(xyz, cell, query)?;
    if frame.n_active == 0 {
        return Ok(Vec::new());
    }
    let stencil = stencil_for(&frame, cell, query)?;
    walk(&frame, cell, &stencil, query)
}

/// Lattice used to bin `periodic`.
///
/// Each open row is replaced by a unit vector normal to the span of the
/// periodic rows, whatever length the caller stored. One periodic row
/// gets the orthonormal complement of that line. Two periodic rows get
/// the unit normal of their plane. The periodic rows and `origin` are
/// kept. A fully open mask does not read `rows` and returns a unit cube.
///
/// A fully periodic cell is [`Cell::from_vectors`]. A periodic face
/// width that is zero or non-finite returns [`Error::Degenerate`].
/// A positive finite width is returned as stored. The pair entries
/// then apply the image count, which depends on the cutoff, and the
/// bin cap. The mesh edge is clamped to the shortest width, so a face
/// much shorter than the other edges can exceed the bin cap at every
/// cutoff. When both caps hold, a short face still builds a list.
///
/// A periodic row that is non-finite, zero, or collinear with the other
/// periodic rows returns [`Error::BadBox`]. The same error is returned
/// when the resulting box is singular.
///
/// ```
/// use linkcell::cell_for_pbc;
///
/// # fn main() -> Result<(), linkcell::Error> {
/// let rows = [[4.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 1.0, 1.0e-8]];
/// let cell = cell_for_pbc(rows, [0.0, 0.0, 0.0], [true, true, false])?;
/// let c = cell.c();
/// assert!(c[0].abs() + c[1].abs() < 1e-12);
/// assert!((c[2] - 1.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
pub fn cell_for_pbc(
    mut rows: [[f64; 3]; 3],
    origin: [f64; 3],
    periodic: [bool; 3],
) -> Result<Cell, Error> {
    if periodic == [false, false, false] {
        return Cell::ortho(1.0, 1.0, 1.0).map_err(|_| Error::BadBox);
    }
    if periodic != [true, true, true] {
        replace_open_rows(&mut rows, periodic)?;
    } else if rows.iter().any(|row| !finite3(*row)) {
        return Err(Error::BadBox);
    }
    let cell = Cell::from_vectors(rows[0], rows[1], rows[2], origin).map_err(|_| Error::BadBox)?;
    if periodic_face_collapsed(&cell, periodic) {
        return Err(Error::Degenerate);
    }
    Ok(cell)
}

fn finite3(v: [f64; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(v: [f64; 3]) -> Result<[f64; 3], Error> {
    let n2 = dot(v, v);
    if !n2.is_finite() || n2 <= 0.0 {
        return Err(Error::BadBox);
    }
    let n = n2.sqrt();
    Ok([v[0] / n, v[1] / n, v[2] / n])
}

/// A unit vector perpendicular to `a`. The seed is the coordinate axis
/// most orthogonal to `a`, so the cross product is not a cancellation.
fn unit_perp(a: [f64; 3]) -> Result<[f64; 3], Error> {
    let ax = a[0].abs();
    let ay = a[1].abs();
    let az = a[2].abs();
    let seed = if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    unit(cross(a, seed))
}

/// Write a unit normal of the periodic rows into every open slot.
///
/// Two periodic rows use their plane's unit normal. One periodic row
/// uses an orthonormal frame of the perpendicular plane, assigned to
/// the open slots in axis order. The periodic rows are left unchanged.
fn replace_open_rows(rows: &mut [[f64; 3]; 3], periodic: [bool; 3]) -> Result<(), Error> {
    let count = periodic.iter().filter(|&&flag| flag).count();
    if count == 2 {
        let mut kept = [[0.0; 3]; 2];
        let mut n = 0;
        let mut open = 0;
        for axis in 0..3 {
            if periodic[axis] {
                kept[n] = rows[axis];
                n += 1;
            } else {
                open = axis;
            }
        }
        let (u, v) = (kept[0], kept[1]);
        if !finite3(u) || !finite3(v) {
            return Err(Error::BadBox);
        }
        let uu = dot(u, u);
        let vv = dot(v, v);
        let normal = cross(u, v);
        let nn = dot(normal, normal);
        // The same relative gram determinant as minimage `PartialCell`.
        if !uu.is_finite()
            || !vv.is_finite()
            || uu <= 0.0
            || vv <= 0.0
            || !nn.is_finite()
            || nn <= 1.0e-24 * uu * vv
        {
            return Err(Error::BadBox);
        }
        rows[open] = unit(normal)?;
        return Ok(());
    }
    if count == 1 {
        let mut axis = 0;
        for (slot, &is_periodic) in periodic.iter().enumerate() {
            if is_periodic {
                axis = slot;
            }
        }
        let a = rows[axis];
        if !finite3(a) {
            return Err(Error::BadBox);
        }
        let n1 = unit_perp(a)?;
        let n2 = unit(cross(a, n1))?;
        let normals = [n1, n2];
        let mut slot = 0;
        for open in 0..3 {
            if periodic[open] {
                continue;
            }
            rows[open] = normals[slot];
            slot += 1;
        }
    }
    Ok(())
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
    /// The partner rows of a fully open frame. `None` when `links` holds
    /// the stencil.
    open: Option<OpenRows>,
}

/// Partner bins of a fully open frame, one row of `dx` per `(dy, dz)`.
///
/// `width[(dz + ext[1]) * (2 * ext[0] + 1) + dy + ext[0]]` is the largest
/// `|dx|` within reach on that row, or -1 when no bin of the row is. Rows
/// past `ext` are not stored. A frame that spans more bins on y or z
/// needs a wider table, unless `ext` already covers `reach` on that axis.
struct OpenRows {
    reach: i32,
    ext: [i32; 2],
    width: Vec<i32>,
}

/// Rows stored on y and z before a frame asks for more.
const OPEN_EXTENT: i32 = 16;

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
    let reach = stencil_reach(query.cutoff, edge)?;
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
    Ok(Stencil {
        key,
        off,
        links,
        open: None,
    })
}

fn stencil_reach(cutoff: f64, edge: f64) -> Result<i32, Error> {
    let reach = (cutoff / edge).ceil();
    // `reach as i32 + 2` is the stencil pad. A value past `i32::MAX - 2`
    // overflows that add in debug and wraps the image window in release.
    if !reach.is_finite() || reach > (i32::MAX - 2) as f64 {
        return Err(Error::TooManyImages);
    }
    Ok(reach as i32 + 2)
}

/// Bin offsets a fully open frame can pair across. The gap between two
/// bins depends on their offset and not on where they sit, so one table
/// serves every frame with this bin edge and cutoff that spans at most
/// `need[0] + 1` bins on y and `need[1] + 1` on z.
fn build_open_stencil(edge: f64, cutoff: f64, key: Key, need: [i32; 2]) -> Result<Stencil, Error> {
    let reach = stencil_reach(cutoff, edge)?;
    let ext = [
        need[0].max(OPEN_EXTENT).min(reach),
        need[1].max(OPEN_EXTENT).min(reach),
    ];
    let mut width = Vec::with_capacity((2 * ext[0] + 1) as usize * (2 * ext[1] + 1) as usize);
    for dz in -ext[1]..=ext[1] {
        for dy in -ext[0]..=ext[0] {
            width.push(row_width(dy, dz, reach, edge, cutoff));
        }
    }
    Ok(Stencil {
        key,
        off: Vec::new(),
        links: Vec::new(),
        open: Some(OpenRows { reach, ext, width }),
    })
}

/// Largest `|dx|` up to `reach` whose bin is within reach of the origin
/// bin on row `(dy, dz)`, or -1. The true gap grows with `|dx|`, so once
/// the certified gap passes the cutoff no farther bin on the row holds a
/// pair.
fn row_width(dy: i32, dz: i32, reach: i32, edge: f64, cutoff: f64) -> i32 {
    let near = |dx: i32| !cells_too_far([0; 3], [dx, dy, dz], [0; 3], edge, [0.0; 3], cutoff);
    if !near(0) {
        return -1;
    }
    let (mut lo, mut hi) = (0, reach);
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if near(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

impl Stencil {
    fn covers(&self, frame: &Frame) -> bool {
        let Some(open) = &self.open else {
            return true;
        };
        (0..2)
            .all(|axis| frame.dims[axis + 1] - 1 <= open.ext[axis] || open.ext[axis] == open.reach)
    }
}

fn stencil_for(frame: &Frame, cell: &Cell, query: &PbcQuery<'_>) -> Result<Arc<Stencil>, Error> {
    let open = query.periodic == [false, false, false];
    let repeats = if open {
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
        origin: if open { [0; 3] } else { frame.origin },
        dims: if open { [0; 3] } else { frame.dims },
        repeats,
        box_bits: box_bits(cell, query.periodic),
    };
    let mut need = [frame.dims[1] - 1, frame.dims[2] - 1];
    {
        let guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(stencil) = guard.as_ref() {
            if stencil.key == key {
                if stencil.covers(frame) {
                    return Ok(Arc::clone(stencil));
                }
                if let Some(open) = &stencil.open {
                    for (want, &ext) in need.iter_mut().zip(open.ext.iter()) {
                        *want = if *want > ext {
                            (*want).max(ext.saturating_mul(2))
                        } else {
                            ext
                        };
                    }
                }
            }
        }
    }
    let built = Arc::new(if open {
        build_open_stencil(frame.edge, query.cutoff, key, need)?
    } else {
        build_stencil(frame, cell, query, key)?
    });
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
    if let Some(open) = &stencil.open {
        walk_open(frame, open, query, cut2, &mut rows);
        return Ok(rows);
    }
    let ncell = frame.off.len() - 1;
    for src in 0..ncell {
        let src_atoms = &frame.occ[frame.off[src] as usize..frame.off[src + 1] as usize];
        if src_atoms.is_empty() {
            continue;
        }
        for link in &stencil.links[stencil.off[src] as usize..stencil.off[src + 1] as usize] {
            pair_bins(
                frame,
                cell,
                query,
                cut2,
                src_atoms,
                link.dst as usize,
                link.shift,
                &mut rows,
            )?;
        }
    }
    Ok(rows)
}

/// Each source bin visits its partner rows clipped to the frame, in
/// ascending flat order, which is the order of a per-bin link list.
fn walk_open(
    frame: &Frame,
    open: &OpenRows,
    query: &PbcQuery<'_>,
    cut2: f64,
    rows: &mut Vec<Pair>,
) {
    let dims = frame.dims;
    let [ey, ez] = open.ext;
    let stride = (2 * ey + 1) as usize;
    let mut src = 0usize;
    for z in 0..dims[2] {
        let dz_lo = (-z).max(-ez);
        let dz_hi = (dims[2] - 1 - z).min(ez);
        for y in 0..dims[1] {
            let dy_lo = (-y).max(-ey);
            let dy_hi = (dims[1] - 1 - y).min(ey);
            for x in 0..dims[0] {
                let src_atoms = &frame.occ[frame.off[src] as usize..frame.off[src + 1] as usize];
                src += 1;
                if src_atoms.is_empty() {
                    continue;
                }
                for dz in dz_lo..=dz_hi {
                    let plane = (dz + ez) as usize * stride;
                    let row_z = (z + dz) as usize * dims[1] as usize;
                    for dy in dy_lo..=dy_hi {
                        let width = open.width[plane + (dy + ey) as usize];
                        if width < 0 {
                            continue;
                        }
                        let row = (row_z + (y + dy) as usize) * dims[0] as usize;
                        let first = row + (x - width.min(x)) as usize;
                        let last = row + (x + width.min(dims[0] - 1 - x)) as usize;
                        for dst in first..=last {
                            let dst_atoms =
                                &frame.occ[frame.off[dst] as usize..frame.off[dst + 1] as usize];
                            pair_open_bins(frame, query, cut2, src_atoms, dst_atoms, rows);
                        }
                    }
                }
            }
        }
    }
}

/// [`pair_bins`] with no shift and no images. A bin lists its atoms in
/// ascending index, so the partners `j > i` are a suffix of `dst_atoms`.
#[inline(always)]
fn pair_open_bins(
    frame: &Frame,
    query: &PbcQuery<'_>,
    cut2: f64,
    src_atoms: &[u32],
    dst_atoms: &[u32],
    rows: &mut Vec<Pair>,
) {
    for &ia in src_atoms {
        let i = ia as usize;
        let start = match (dst_atoms.first(), dst_atoms.last()) {
            (Some(&first), _) if first > ia => 0,
            (_, Some(&last)) if last > ia => dst_atoms.partition_point(|&ja| ja <= ia),
            _ => continue,
        };
        let pi = frame.pos[i];
        for &ja in &dst_atoms[start..] {
            let j = ja as usize;
            let pj = frame.pos[j];
            // The zero shift [`pair_bins`] adds can only flip the sign of
            // a zero difference, which squares to the same `dist2` bits.
            let dx = pj[0] - pi[0];
            let dy = pj[1] - pi[1];
            let dz = pj[2] - pi[2];
            let dist2 = dx * dx + dy * dy + dz * dz;
            if !accept(dist2, cut2, query.closed) {
                continue;
            }
            rows.push(Pair {
                i,
                j,
                shift: [0; 3],
                dist2,
            });
            if !query.half {
                rows.push(Pair {
                    i: j,
                    j: i,
                    shift: [0; 3],
                    dist2,
                });
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pair_bins(
    frame: &Frame,
    cell: &Cell,
    query: &PbcQuery<'_>,
    cut2: f64,
    src_atoms: &[u32],
    dst: usize,
    shift: [i32; 3],
    rows: &mut Vec<Pair>,
) -> Result<(), Error> {
    let dst_atoms = &frame.occ[frame.off[dst] as usize..frame.off[dst + 1] as usize];
    if dst_atoms.is_empty() {
        return Ok(());
    }
    for &ia in src_atoms {
        let i = ia as usize;
        let pi = frame.pos[i];
        for &ja in dst_atoms {
            let j = ja as usize;
            if i == j && shift == [0, 0, 0] {
                continue;
            }
            let reported = reported_shift(frame, i, j, shift)?;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use minimage::PartialCell;

    use super::{cell_for_pbc, pairs_within_pbc, PbcQuery};
    use crate::{
        pairs_within, pairs_within_columns, pairs_within_csr, Cell, Error, Pair, PairColumns,
        PairCsr,
    };

    type Key = (usize, usize, [i32; 3]);

    /// Lengths 13, 13, 23.3 Å and angles 86.892, 16.892, 70 degrees.
    /// The dimensionless residual keeps a positive c height of about 1.2e-7 Å.
    /// `lz² - cx² - cy²` cancels through zero and is not this height.
    fn near_flat_rows() -> [[f64; 3]; 3] {
        let (lx, ly, lz) = (13.0, 13.0, 23.3);
        let alpha = 86.892_f64.to_radians();
        let beta = 16.892_f64.to_radians();
        let gamma = 70.0_f64.to_radians();
        let (cos_a, cos_b, cos_g) = (alpha.cos(), beta.cos(), gamma.cos());
        let sin_g = gamma.sin();
        let factor = (cos_a - cos_b * cos_g) / sin_g;
        let inside = 1.0 - cos_b * cos_b - factor * factor;
        assert!(inside > 0.0, "c height {inside}");
        let cz = lz * inside.sqrt();
        assert!(cz > 1.0e-7 && cz < 1.5e-7, "c_z {cz}");
        [
            [lx, 0.0, 0.0],
            [ly * cos_g, ly * sin_g, 0.0],
            [lz * cos_b, lz * factor, cz],
        ]
    }

    fn near_flat_cu() -> Vec<[f64; 3]> {
        vec![
            [2.7431309316168475, 6.108002035108404, 0.0],
            [13.232056168881522, 6.352322116512741, 0.2],
            [4.03338785589701, 0.3664801221065042, 0.0],
            [7.720397346483967, 10.13928337827995, -0.2],
            [8.19626621664133, 0.45462981824422477, 0.0412786853694912],
            [13.762071462898417, 10.626955433611426, -0.03622779513074964],
            [11.205363013255461, 7.1146751150038385, 0.027634492938765798],
            [4.492545207753816, 4.035049921555236, -0.19606254358135186],
            [16.50539159880175, 12.057883571293173, 0.15819319967883666],
            [4.199134240190456, 10.099861087740562, 0.2068132352186905],
            [6.126746340992574, 8.464007919000748, -0.11518997428508809],
            [9.814929593400677, 8.903620103137243, -0.1682723449043315],
            [2.2857588312509485, 2.9564446590077442, 0.12103716729274078],
            [
                5.9510448718450855,
                0.20132540631560192,
                -0.08736864655602095,
            ],
            [11.368678075155005, 10.008767396697653, -0.05551644303227138],
            [15.870150122394206, 8.195507208935341, 0.1519522101214148],
            [10.42169020181862, 1.7934744461922287, 0.10864204430766089],
            [8.351747146366385, 3.7093332382537727, 0.02911048113793241],
            [4.732924885010005, 6.6345398008188905, 0.1202865161255377],
            [11.924591995381464, 4.883493475804255, -0.06332713818033944],
            [12.308118627317418, 2.485340130933891, -0.24174491728692993],
            [0.9990718536434593, 0.9081763430908019, 0.09809706259195394],
            [8.401410062080734, 6.873783879816667, 0.223667971540852],
            [5.947793909899321, 2.4918353256604986, -0.27449681317456937],
        ]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn norm(a: [f64; 3]) -> f64 {
        dot(a, a).sqrt()
    }

    fn keep(i: usize, j: usize, shift: [i32; 3]) -> bool {
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

    fn raw_cell(rows: [[f64; 3]; 3], origin: [f64; 3]) -> Cell {
        Cell::from_vectors(rows[0], rows[1], rows[2], origin).unwrap()
    }

    fn unwrapped(cell: &Cell, point: [f64; 3]) -> [f64; 3] {
        let origin = cell.origin();
        let delta = [
            point[0] - origin[0],
            point[1] - origin[1],
            point[2] - origin[2],
        ];
        if cell.is_ortho() {
            let widths = cell.widths();
            return [
                delta[0] / widths[0],
                delta[1] / widths[1],
                delta[2] / widths[2],
            ];
        }
        let h = cell.hinv();
        [
            h[0][0] * delta[0] + h[1][0] * delta[1] + h[2][0] * delta[2],
            h[0][1] * delta[0] + h[1][1] * delta[1] + h[2][1] * delta[2],
            h[0][2] * delta[0] + h[1][2] * delta[1] + h[2][2] * delta[2],
        ]
    }

    fn shift_window(cell: &Cell, xyz: &[[f64; 3]], periodic: [bool; 3], cutoff: f64) -> i32 {
        let widths = cell.widths();
        let mut reach = 1i32;
        for axis in 0..3 {
            if !periodic[axis] {
                continue;
            }
            let reps = (cutoff / widths[axis]).ceil();
            assert!(
                reps.is_finite() && (0.0..64.0).contains(&reps),
                "repeats {reps}"
            );
            let reps = reps as i32;
            let mut lo = 0i32;
            let mut hi = 0i32;
            for point in xyz {
                let whole = unwrapped(cell, *point)[axis].floor();
                assert!(whole.is_finite(), "fraction {whole}");
                let whole = whole as i32;
                lo = lo.min(whole);
                hi = hi.max(whole);
            }
            reach = reach.max((hi - lo).abs() + reps + 1);
        }
        reach
    }

    fn brute(
        xyz: &[[f64; 3]],
        cell: &Cell,
        periodic: [bool; 3],
        cutoff: f64,
        half: bool,
        exclude_self: bool,
        window: i32,
    ) -> BTreeMap<Key, f64> {
        let cut2 = cutoff * cutoff;
        let limit = |axis| {
            if periodic[axis] {
                window
            } else {
                0
            }
        };
        let mut out = BTreeMap::new();
        for na in -limit(0)..=limit(0) {
            for nb in -limit(1)..=limit(1) {
                for nc in -limit(2)..=limit(2) {
                    let shift = [na, nb, nc];
                    for (i, pi) in xyz.iter().enumerate() {
                        for (j, pj) in xyz.iter().enumerate() {
                            if exclude_self && i == j {
                                continue;
                            }
                            if i == j && shift == [0, 0, 0] {
                                continue;
                            }
                            if half && !keep(i, j, shift) {
                                continue;
                            }
                            let delta = if shift == [0, 0, 0] {
                                [0.0; 3]
                            } else {
                                cell.lattice_shift(shift[0], shift[1], shift[2])
                            };
                            let dx = (pj[0] + delta[0]) - pi[0];
                            let dy = (pj[1] + delta[1]) - pi[1];
                            let dz = (pj[2] + delta[2]) - pi[2];
                            let dist2 = dx * dx + dy * dy + dz * dz;
                            if dist2 < cut2 {
                                out.insert((i, j, shift), dist2);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn assert_minimum_image(
        xyz: &[[f64; 3]],
        rows: [[f64; 3]; 3],
        pairs: &[Pair],
        periodic: [bool; 3],
        cutoff: f64,
    ) {
        let partial = PartialCell::from_ase(rows, periodic).unwrap();
        let cut2 = cutoff * cutoff;
        let mut mins: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for pair in pairs {
            if pair.i == pair.j {
                continue;
            }
            let key = if pair.i < pair.j {
                (pair.i, pair.j)
            } else {
                (pair.j, pair.i)
            };
            mins.entry(key)
                .and_modify(|best| {
                    if pair.dist2 < *best {
                        *best = pair.dist2;
                    }
                })
                .or_insert(pair.dist2);
        }
        for i in 0..xyz.len() {
            for j in (i + 1)..xyz.len() {
                let dist2 = partial.dist2_euclidean(xyz[i], xyz[j]);
                match mins.get(&(i, j)) {
                    Some(&got) => {
                        let scale = dist2.abs().max(got.abs()).max(1.0);
                        assert!(
                            (got - dist2).abs() <= 1.0e-8 * scale,
                            "pair {i},{j}: list {got} minimage {dist2}"
                        );
                    }
                    None => assert!(
                        dist2 >= cut2,
                        "missing pair {i},{j} at minimage distance {dist2}"
                    ),
                }
            }
        }
    }

    fn assert_basis(rows: [[f64; 3]; 3], origin: [f64; 3], periodic: [bool; 3]) -> Cell {
        let cell = cell_for_pbc(rows, origin, periodic).unwrap();
        assert_eq!(cell.origin(), origin);
        if periodic == [false, false, false] {
            let cube = Cell::ortho(1.0, 1.0, 1.0).unwrap();
            assert_eq!(cell.h(), cube.h());
            return cell;
        }
        let columns = cell.h();
        for axis in 0..3 {
            if periodic[axis] {
                assert_eq!(columns[axis], rows[axis]);
                continue;
            }
            let row = columns[axis];
            assert!((norm(row) - 1.0).abs() < 1.0e-12);
            for other in 0..3 {
                if periodic[other] {
                    let scale = norm(rows[other]).max(1.0);
                    assert!(
                        dot(row, rows[other]).abs() <= 1.0e-9 * scale,
                        "open axis {axis} not normal to {other}"
                    );
                }
            }
        }
        let open: Vec<[f64; 3]> = (0..3)
            .filter(|&a| !periodic[a])
            .map(|a| columns[a])
            .collect();
        if open.len() == 2 {
            assert!(dot(open[0], open[1]).abs() < 1.0e-9);
        }
        let again = cell_for_pbc(columns, origin, periodic).unwrap();
        assert_eq!(again.h(), columns);
        cell
    }

    fn expect_pairs(
        rows: [[f64; 3]; 3],
        origin: [f64; 3],
        xyz: &[[f64; 3]],
        periodic: [bool; 3],
        cutoff: f64,
        half: bool,
    ) -> Vec<Pair> {
        let binning = assert_basis(rows, origin, periodic);
        let query = PbcQuery::new(periodic, cutoff)
            .half(half)
            .exclude_self_images(true);
        let found = pairs_within_pbc(xyz, &binning, &query).unwrap();
        let window = shift_window(&binning, xyz, periodic, cutoff);
        for pair in &found {
            for (axis, &component) in pair.shift.iter().enumerate() {
                if periodic[axis] {
                    assert!(
                        component.abs() <= window,
                        "shift {component} window {window}"
                    );
                } else {
                    assert_eq!(component, 0);
                }
            }
        }
        let scanned = brute(xyz, &binning, periodic, cutoff, half, true, window);
        let got: BTreeMap<Key, f64> = found
            .iter()
            .map(|pair| ((pair.i, pair.j, pair.shift), pair.dist2))
            .collect();
        if periodic == [true, true, true] {
            let mut direct = pairs_within(xyz, &binning, cutoff, None, None, half).unwrap();
            direct.retain(|pair| pair.i != pair.j);
            assert_eq!(found, direct);
            assert_eq!(
                got.keys().collect::<Vec<_>>(),
                scanned.keys().collect::<Vec<_>>()
            );
            for (key, dist2) in &scanned {
                let scale = dist2.abs().max(got[key].abs()).max(1.0);
                assert!(
                    (got[key] - dist2).abs() <= 1.0e-9 * scale,
                    "periodic distance {key:?}"
                );
            }
        } else {
            assert_eq!(
                got, scanned,
                "periodic {periodic:?} half {half} window {window}"
            );
        }
        if periodic != [false, false, false] {
            let raw = raw_cell(rows, origin);
            let from_raw = pairs_within_pbc(xyz, &raw, &query).unwrap();
            assert_eq!(from_raw, found);
        }
        assert_minimum_image(xyz, rows, &found, periodic, cutoff);
        found
    }

    #[test]
    fn near_flat_cu_open_patterns_match_minimum_image() {
        let rows = near_flat_rows();
        let xyz = near_flat_cu();
        let patterns = [
            [true, true, false],
            [true, false, true],
            [false, true, true],
            [true, false, false],
            [false, true, false],
            [false, false, true],
            [false, false, false],
        ];
        let mut saw_wrapped = false;
        for periodic in patterns {
            let found = expect_pairs(rows, [0.0; 3], &xyz, periodic, 2.6, true);
            if periodic == [true, true, false] {
                assert_eq!(found.len(), 32);
                saw_wrapped = found.iter().any(|pair| pair.shift != [0, 0, 0]);
                let mut csr = PairCsr::default();
                let query = PbcQuery::new(periodic, 2.6)
                    .half(true)
                    .exclude_self_images(true);
                pairs_within_csr(&xyz, &raw_cell(rows, [0.0; 3]), &query, true, &mut csr).unwrap();
                let mut from_csr = BTreeMap::new();
                for i in 0..xyz.len() {
                    let start = usize::try_from(csr.offsets[i]).unwrap();
                    let end = usize::try_from(csr.offsets[i + 1]).unwrap();
                    for slot in start..end {
                        let j = usize::try_from(csr.cols[slot]).unwrap();
                        from_csr.insert((i, j), csr.dist2[slot]);
                    }
                }
                let mut from_pairs = BTreeMap::new();
                for pair in &found {
                    from_pairs
                        .entry((pair.i, pair.j))
                        .and_modify(|best: &mut f64| {
                            if pair.dist2 < *best {
                                *best = pair.dist2;
                            }
                        })
                        .or_insert(pair.dist2);
                }
                assert_eq!(from_csr, from_pairs);
            }
        }
        assert!(
            saw_wrapped,
            "the slab has a periodic image inside the cutoff"
        );
    }

    #[test]
    fn near_flat_cu_fully_periodic_is_too_many_images() {
        let rows = near_flat_rows();
        let xyz = near_flat_cu();
        let raw = raw_cell(rows, [0.0; 3]);
        let built = cell_for_pbc(rows, [0.0; 3], [true, true, true]).unwrap();
        assert!(built.widths().iter().all(|w| w.is_finite() && *w > 0.0));
        assert_eq!(
            pairs_within(&xyz, &raw, 2.6, None, None, true).unwrap_err(),
            Error::TooManyImages
        );
        assert_eq!(
            pairs_within_pbc(&xyz, &raw, &PbcQuery::periodic(2.6)).unwrap_err(),
            Error::TooManyImages
        );
        let mut csr = PairCsr::default();
        assert_eq!(
            pairs_within_csr(&xyz, &raw, &PbcQuery::periodic(2.6), true, &mut csr).unwrap_err(),
            Error::TooManyImages
        );
        // Tilt reduction leaves perpendicular widths near 1e-7 and a mesh
        // of 10 x 2 x 1 bins, inside the bin cap. The box is far more
        // than 1e7 of those bin edges across, so the walk cannot certify
        // a neighbour and returns TooManyCells. Brute force has no mesh.
        assert_eq!(
            crate::knearest(&xyz, &raw, 1, None, None).unwrap_err(),
            Error::TooManyCells
        );
        let brute = crate::knearest_brute(&xyz, &raw, 1, None).unwrap();
        assert!(brute
            .iter()
            .all(|row| row.dist2.iter().all(|d| d.is_finite())));
        let partial = PartialCell::from_ase(rows, [true, true, true]).unwrap();
        let mut farthest = 0.0f64;
        for i in 0..xyz.len() {
            for j in (i + 1)..xyz.len() {
                farthest = farthest.max(partial.dist2_euclidean(xyz[i], xyz[j]).sqrt());
            }
        }
        assert!(
            farthest.is_finite() && farthest < 0.1,
            "fully periodic minimum image on this flat cell is {farthest}"
        );
    }

    #[test]
    fn skewed_open_row_is_replaced_by_the_plane_normal() {
        let rows = [[4.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 1.0, 1.0e-8]];
        let cell = assert_basis(rows, [1.0, 2.0, 3.0], [true, true, false]);
        let c = cell.c();
        assert!(c[0].abs() + c[1].abs() < 1.0e-12);
        assert!((c[2] - 1.0).abs() < 1.0e-12);
        let wire = [[0.0, 4.0, 0.0], [9.0, 9.0, 9.0], [-3.0, 1.0, 2.0]];
        let cell = assert_basis(wire, [0.0; 3], [true, false, false]);
        assert!((cell.widths()[0] - 4.0).abs() < 1.0e-12);
    }

    fn rnd(state: &mut u64) -> f64 {
        *state ^= state.wrapping_shl(13);
        *state ^= state.wrapping_shr(7);
        *state ^= state.wrapping_shl(17);
        (*state >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn healthy_rows(state: &mut u64) -> [[f64; 3]; 3] {
        let length = |state: &mut u64| 8.0 + 5.0 * rnd(state);
        let la = length(state);
        let lb = length(state);
        let gamma = std::f64::consts::FRAC_PI_3 + rnd(state) * std::f64::consts::FRAC_PI_3;
        let a = [la, 0.0, 0.0];
        let b = [lb * gamma.cos(), lb * gamma.sin(), 0.0];
        let c = [
            (rnd(state) - 0.5) * la,
            (rnd(state) - 0.5) * lb,
            8.0 + 4.0 * rnd(state),
        ];
        [a, b, c]
    }

    fn points(rows: [[f64; 3]; 3], n: usize, state: &mut u64) -> Vec<[f64; 3]> {
        (0..n)
            .map(|_| {
                let s = [rnd(state), rnd(state), rnd(state)];
                let mut r = [0.0; 3];
                for axis in 0..3 {
                    for k in 0..3 {
                        r[k] += s[axis] * rows[axis][k];
                    }
                }
                r
            })
            .collect()
    }

    fn flatten(rows: &mut [[f64; 3]; 3], axis: usize, eps: f64) {
        let (u, v) = match axis {
            0 => (rows[1], rows[2]),
            1 => (rows[0], rows[2]),
            _ => (rows[0], rows[1]),
        };
        let n = super::cross(u, v);
        let n = super::unit(n).unwrap();
        let height = dot(rows[axis], n);
        for k in 0..3 {
            rows[axis][k] += (eps - height) * n[k];
        }
    }

    #[test]
    fn random_near_flat_cells_match_minimum_image() {
        let mut state = 0xC0FFEE_u64;
        for trial in 0..6 {
            let rows = healthy_rows(&mut state);
            let xyz = points(rows, 8, &mut state);
            for mask in 0..8 {
                let periodic = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
                expect_pairs(rows, [0.0; 3], &xyz, periodic, 1.8, trial % 2 == 0);
            }
            let mut flat = rows;
            let axis = trial % 3;
            flatten(&mut flat, axis, 1.0e-8);
            let built = cell_for_pbc(flat, [0.0; 3], [true, true, true]).unwrap();
            assert!(built.widths().iter().all(|w| w.is_finite() && *w > 0.0));
            let raw = raw_cell(flat, [0.0; 3]);
            assert_eq!(
                pairs_within_pbc(&xyz, &raw, &PbcQuery::periodic(1.8)).unwrap_err(),
                Error::TooManyImages
            );
            let mut periodic = [true, true, true];
            periodic[axis] = false;
            expect_pairs(flat, [0.0; 3], &xyz, periodic, 1.8, true);
        }
    }

    #[cfg(feature = "capi")]
    #[test]
    fn near_flat_cu_c_api_matches_rust() {
        use std::ffi::CStr;

        let rows = near_flat_rows();
        let xyz = near_flat_cu();
        let periodic = [true, true, false];
        let rust = expect_pairs(rows, [0.0; 3], &xyz, periodic, 2.6, true);
        let raw = crate::lc_cell {
            ax: rows[0][0],
            ay: rows[0][1],
            az: rows[0][2],
            bx: rows[1][0],
            by: rows[1][1],
            bz: rows[1][2],
            cx: rows[2][0],
            cy: rows[2][1],
            cz: rows[2][2],
            ox: 0.0,
            oy: 0.0,
            oz: 0.0,
        };
        let flat: Vec<f64> = xyz.iter().copied().flatten().collect();
        let pbc = [1, 1, 0];
        let mut count = 0usize;
        let status = unsafe {
            crate::lc_pairs_within_pbc(
                flat.as_ptr(),
                xyz.len(),
                &raw,
                pbc.as_ptr(),
                2.6,
                std::ptr::null(),
                0.0,
                1,
                0,
                1,
                std::ptr::null_mut(),
                0,
                &mut count,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(count, rust.len());
        let mut out = vec![crate::lc_pair::default(); count];
        let status = unsafe {
            crate::lc_pairs_within_pbc(
                flat.as_ptr(),
                xyz.len(),
                &raw,
                pbc.as_ptr(),
                2.6,
                std::ptr::null(),
                0.0,
                1,
                0,
                1,
                out.as_mut_ptr(),
                count,
                &mut count,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(out.len(), rust.len());
        for (row, pair) in out.iter().zip(&rust) {
            assert_eq!(row.i, i32::try_from(pair.i).unwrap());
            assert_eq!(row.j, i32::try_from(pair.j).unwrap());
            assert_eq!(row.shift, pair.shift);
            assert_eq!(row.dist2, pair.dist2);
        }
        let mut ignored = 0usize;
        let status = unsafe {
            crate::lc_pairs_within_pbc(
                flat.as_ptr(),
                xyz.len(),
                &raw,
                std::ptr::null(),
                2.6,
                std::ptr::null(),
                0.0,
                1,
                0,
                1,
                std::ptr::null_mut(),
                0,
                &mut ignored,
            )
        };
        assert_ne!(status, 0);
        let message = unsafe { CStr::from_ptr(crate::lc_last_error()) };
        assert_eq!(
            message.to_str().unwrap(),
            "periodic image count exceeds 2^24"
        );
    }

    fn close_dist(left: f64, right: f64) -> bool {
        let scale = left.abs().max(right.abs()).max(1.0);
        (left - right).abs() <= 1.0e-9 * scale
    }

    fn slab(ratio: f64) -> (Cell, Vec<[f64; 3]>) {
        // One short face. The mesh edge clamps to that face, so the two
        // long axes exceed the bin cap (or the image cap, when the
        // cutoff asks for more than 2^24 images). Either is a typed
        // error and neither reserves the row buffer.
        let long = 10.0;
        let thin = long * ratio;
        let cell = Cell::ortho(long, long, thin).unwrap();
        let xyz = vec![[1.0, 1.0, thin * 0.30], [1.4, 1.2, thin * 0.70]];
        (cell, xyz)
    }

    /// Face ratio `1e-4`, short face equal to `cutoff`. The mesh edge
    /// clamps to that face, so the long axis has 10_000 bins and the
    /// reach stays 1: a stencil the bin cap accepts, at a ratio the old
    /// face-ratio cutoff would also have accepted. Ratios below that
    /// cutoff are the slab sweep, which the bin cap refuses.
    fn accepted_rod(cutoff: f64) -> (Cell, Vec<[f64; 3]>) {
        let ratio = 1.0e-4;
        let thin = cutoff;
        let long = cutoff / ratio;
        let cell = Cell::ortho(long, thin, thin).unwrap();
        let xyz = vec![
            [long * 0.20, thin * 0.25, thin * 0.30],
            [long * 0.20 + 0.40 * cutoff, thin * 0.55, thin * 0.40],
        ];
        (cell, xyz)
    }

    fn brute_axes(
        xyz: &[[f64; 3]],
        cell: &Cell,
        cutoff: f64,
        reps: [i32; 3],
    ) -> BTreeMap<Key, f64> {
        let cut2 = cutoff * cutoff;
        let mut out = BTreeMap::new();
        for na in -reps[0]..=reps[0] {
            for nb in -reps[1]..=reps[1] {
                for nc in -reps[2]..=reps[2] {
                    let shift = [na, nb, nc];
                    for (i, pi) in xyz.iter().enumerate() {
                        for (j, pj) in xyz.iter().enumerate() {
                            if i == j && shift == [0, 0, 0] {
                                continue;
                            }
                            if !keep(i, j, shift) {
                                continue;
                            }
                            let delta = if shift == [0, 0, 0] {
                                [0.0; 3]
                            } else {
                                cell.lattice_shift(shift[0], shift[1], shift[2])
                            };
                            let dx = (pj[0] + delta[0]) - pi[0];
                            let dy = (pj[1] + delta[1]) - pi[1];
                            let dz = (pj[2] + delta[2]) - pi[2];
                            let dist2 = dx * dx + dy * dy + dz * dz;
                            if dist2 < cut2 {
                                out.insert((i, j, shift), dist2);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn assert_typed(err: Error) {
        assert!(
            matches!(
                err,
                Error::TooManyImages | Error::TooManyCells | Error::Degenerate
            ),
            "unexpected {err}"
        );
    }

    fn expect_list(cell: &Cell, xyz: &[[f64; 3]], cutoff: f64) {
        let found = pairs_within(xyz, cell, cutoff, None, None, true).unwrap();
        let via_pbc = pairs_within_pbc(xyz, cell, &PbcQuery::periodic(cutoff).half(true)).unwrap();
        assert_eq!(found.len(), via_pbc.len());
        for (row, other) in found.iter().zip(&via_pbc) {
            assert_eq!((row.i, row.j, row.shift), (other.i, other.j, other.shift));
            assert!(close_dist(row.dist2, other.dist2));
        }
        let mut columns = PairColumns::default();
        pairs_within_columns(xyz, cell, cutoff, None, None, true, &mut columns).unwrap();
        assert_eq!(columns.len(), found.len());
        let mut csr = PairCsr::default();
        pairs_within_csr(
            xyz,
            cell,
            &PbcQuery::periodic(cutoff).half(true),
            true,
            &mut csr,
        )
        .unwrap();
        assert_eq!(csr.offsets.len(), xyz.len() + 1);
        let reps = cell.widths().map(|width| (cutoff / width).ceil() as i32);
        let want = brute_axes(xyz, cell, cutoff, reps);
        let mut got = BTreeMap::new();
        for pair in &found {
            got.insert((pair.i, pair.j, pair.shift), pair.dist2);
        }
        assert_eq!(got.len(), want.len());
        for (key, dist2) in &want {
            let stored = got[key];
            assert!(close_dist(stored, *dist2), "{key:?} {stored} {dist2}");
        }
    }

    fn expect_capped(cell: &Cell, xyz: &[[f64; 3]], cutoff: f64) {
        let err = pairs_within(xyz, cell, cutoff, None, None, true).unwrap_err();
        assert_typed(err);
        assert_ne!(err, Error::Degenerate);
        assert_typed(
            pairs_within_pbc(xyz, cell, &PbcQuery::periodic(cutoff).half(true)).unwrap_err(),
        );
        let mut columns = PairColumns::default();
        assert_typed(
            pairs_within_columns(xyz, cell, cutoff, None, None, true, &mut columns).unwrap_err(),
        );
        let mut csr = PairCsr::default();
        assert_typed(
            pairs_within_csr(
                xyz,
                cell,
                &PbcQuery::periodic(cutoff).half(true),
                true,
                &mut csr,
            )
            .unwrap_err(),
        );
    }

    /// Ratios around the old face-ratio cutoff, at three cutoffs. A cell
    /// the image and bin caps accept matches the minimum image. A cell
    /// they refuse is a typed error and is not `Degenerate`. Neither aborts.
    #[test]
    fn thin_periodic_ratios_match_or_error() {
        let ratios = [5.0e-7, 9.9e-7, 1.0e-6, 1.1e-6, 2.0e-6, 1.0e-5, 1.0e-4];
        let cutoffs = [0.1, 2.6, 10.0];
        for ratio in ratios {
            let (cell, xyz) = slab(ratio);
            for cutoff in cutoffs {
                expect_capped(&cell, &xyz, cutoff);
            }
        }
        // Face ratio 1e-4 at each cutoff. The short face equals the
        // cutoff, so the image and bin caps accept the cell and the list
        // matches the minimum image. A positive finite width is not
        // Degenerate, including a ratio below the old face-ratio cutoff
        // (those slabs error above with TooManyCells or TooManyImages).
        for cutoff in cutoffs {
            let (cell, xyz) = accepted_rod(cutoff);
            let widths = cell.widths();
            assert!((widths[1] / widths[0] - 1.0e-4).abs() < 1.0e-12);
            expect_list(&cell, &xyz, cutoff);
        }

        // Enough points that reserving the ideal-gas estimate would abort.
        // The image cap is exceeded, so the call returns before that reserve.
        let long = 100.0;
        let thin = long * 1.1e-6;
        let cell = Cell::ortho(long, thin, thin).unwrap();
        let xyz = vec![[long * 0.2, thin * 0.5, thin * 0.5]; 512];
        assert_eq!(
            pairs_within(&xyz, &cell, 2.6, None, None, true).unwrap_err(),
            Error::TooManyImages
        );
        assert_eq!(
            pairs_within_pbc(&xyz, &cell, &PbcQuery::periodic(2.6)).unwrap_err(),
            Error::TooManyImages
        );
        let mut columns = PairColumns::default();
        assert_eq!(
            pairs_within_columns(&xyz, &cell, 2.6, None, None, true, &mut columns).unwrap_err(),
            Error::TooManyImages
        );
        let mut csr = PairCsr::default();
        assert_eq!(
            pairs_within_csr(&xyz, &cell, &PbcQuery::periodic(2.6), true, &mut csr).unwrap_err(),
            Error::TooManyImages
        );
    }

    #[cfg(feature = "capi")]
    #[test]
    fn thin_periodic_ratios_c_api_matches_rust() {
        use std::ffi::CStr;

        let ratios = [5.0e-7, 9.9e-7, 1.0e-6, 1.1e-6, 2.0e-6, 1.0e-5, 1.0e-4];
        let cutoffs = [0.1, 2.6, 10.0];
        for ratio in ratios {
            let (cell, xyz) = slab(ratio);
            let h = cell.h();
            let raw = crate::lc_cell {
                ax: h[0][0],
                ay: h[0][1],
                az: h[0][2],
                bx: h[1][0],
                by: h[1][1],
                bz: h[1][2],
                cx: h[2][0],
                cy: h[2][1],
                cz: h[2][2],
                ox: 0.0,
                oy: 0.0,
                oz: 0.0,
            };
            let flat: Vec<f64> = xyz.iter().copied().flatten().collect();
            for cutoff in cutoffs {
                let rust = pairs_within(&xyz, &cell, cutoff, None, None, true);
                let mut count = 0usize;
                let status = unsafe {
                    crate::lc_pairs_within_pbc(
                        flat.as_ptr(),
                        xyz.len(),
                        &raw,
                        std::ptr::null(),
                        cutoff,
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
                match rust {
                    Ok(found) => {
                        assert_eq!(status, 0);
                        assert_eq!(count, found.len());
                        let mut out = vec![crate::lc_pair::default(); count];
                        let mut filled = 0usize;
                        let status = unsafe {
                            crate::lc_pairs_within_pbc(
                                flat.as_ptr(),
                                xyz.len(),
                                &raw,
                                std::ptr::null(),
                                cutoff,
                                std::ptr::null(),
                                0.0,
                                1,
                                0,
                                0,
                                out.as_mut_ptr(),
                                count,
                                &mut filled,
                            )
                        };
                        assert_eq!(status, 0);
                        assert_eq!(filled, found.len());
                        for (row, pair) in out.iter().zip(&found) {
                            assert_eq!(row.i, i32::try_from(pair.i).unwrap());
                            assert_eq!(row.j, i32::try_from(pair.j).unwrap());
                            assert_eq!(row.shift, pair.shift);
                            assert!(close_dist(row.dist2, pair.dist2));
                        }
                    }
                    Err(err) => {
                        assert_ne!(status, 0);
                        let message = unsafe { CStr::from_ptr(crate::lc_last_error()) };
                        assert_eq!(message.to_str().unwrap(), err.to_string());
                    }
                }
            }
        }
        for cutoff in cutoffs {
            let (cell, xyz) = accepted_rod(cutoff);
            let h = cell.h();
            let raw = crate::lc_cell {
                ax: h[0][0],
                ay: h[0][1],
                az: h[0][2],
                bx: h[1][0],
                by: h[1][1],
                bz: h[1][2],
                cx: h[2][0],
                cy: h[2][1],
                cz: h[2][2],
                ox: 0.0,
                oy: 0.0,
                oz: 0.0,
            };
            let flat: Vec<f64> = xyz.iter().copied().flatten().collect();
            let found = pairs_within(&xyz, &cell, cutoff, None, None, true).unwrap();
            let mut count = 0usize;
            let status = unsafe {
                crate::lc_pairs_within_pbc(
                    flat.as_ptr(),
                    xyz.len(),
                    &raw,
                    std::ptr::null(),
                    cutoff,
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
            assert_eq!(status, 0);
            assert_eq!(count, found.len());
            let mut out = vec![crate::lc_pair::default(); count];
            let mut filled = 0usize;
            let status = unsafe {
                crate::lc_pairs_within_pbc(
                    flat.as_ptr(),
                    xyz.len(),
                    &raw,
                    std::ptr::null(),
                    cutoff,
                    std::ptr::null(),
                    0.0,
                    1,
                    0,
                    0,
                    out.as_mut_ptr(),
                    count,
                    &mut filled,
                )
            };
            assert_eq!(status, 0);
            assert_eq!(filled, found.len());
            for (row, pair) in out.iter().zip(&found) {
                assert_eq!(row.i, i32::try_from(pair.i).unwrap());
                assert_eq!(row.j, i32::try_from(pair.j).unwrap());
                assert_eq!(row.shift, pair.shift);
                assert!(close_dist(row.dist2, pair.dist2));
            }
        }
        let long = 100.0;
        let thin = long * 1.1e-6;
        let cell = Cell::ortho(long, thin, thin).unwrap();
        let h = cell.h();
        let raw = crate::lc_cell {
            ax: h[0][0],
            ay: h[0][1],
            az: h[0][2],
            bx: h[1][0],
            by: h[1][1],
            bz: h[1][2],
            cx: h[2][0],
            cy: h[2][1],
            cz: h[2][2],
            ox: 0.0,
            oy: 0.0,
            oz: 0.0,
        };
        let xyz = vec![[long * 0.2, thin * 0.5, thin * 0.5]; 512];
        let flat: Vec<f64> = xyz.iter().copied().flatten().collect();
        let mut count = 0usize;
        let status = unsafe {
            crate::lc_pairs_within_pbc(
                flat.as_ptr(),
                xyz.len(),
                &raw,
                std::ptr::null(),
                2.6,
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
        assert_ne!(status, 0);
        let message = unsafe { CStr::from_ptr(crate::lc_last_error()) };
        assert_eq!(
            message.to_str().unwrap(),
            "periodic image count exceeds 2^24"
        );
    }

    /// Widths 10, 10, 1.1e-5 at cutoff 10 stay under the image cap
    /// (16,363,647 images) and miss the bin cap. Eight points make the
    /// ideal-gas estimate about 12 GB on a full list. The bin cap is
    /// reported before that buffer is reserved.
    #[test]
    fn flat_face_under_the_image_cap_is_too_many_cells() {
        let thin = 1.1e-5;
        let cell = Cell::ortho(10.0, 10.0, thin).unwrap();
        let xyz = vec![[1.0, 2.0, thin * 0.3]; 8];
        for half in [false, true] {
            assert_eq!(
                pairs_within(&xyz, &cell, 10.0, None, None, half).unwrap_err(),
                Error::TooManyCells
            );
            let mut columns = PairColumns::default();
            assert_eq!(
                pairs_within_columns(&xyz, &cell, 10.0, None, None, half, &mut columns)
                    .unwrap_err(),
                Error::TooManyCells
            );
        }
        for half in [false, true] {
            let query = PbcQuery::periodic(10.0).half(half);
            assert_eq!(
                pairs_within_pbc(&xyz, &cell, &query).unwrap_err(),
                Error::TooManyCells
            );
            let mut csr = PairCsr::default();
            assert_eq!(
                pairs_within_csr(&xyz, &cell, &query, true, &mut csr).unwrap_err(),
                Error::TooManyCells
            );
        }
        // Ratio 1e-6, same long edges. Every cutoff is a typed error.
        let cell = Cell::ortho(10.0, 10.0, 1.0e-5).unwrap();
        let xyz = vec![[1.0, 2.0, 1.0e-5 * 0.3]; 4];
        for cutoff in [0.1, 2.6, 10.0] {
            let err = pairs_within(&xyz, &cell, cutoff, None, None, false).unwrap_err();
            assert!(
                matches!(err, Error::TooManyImages | Error::TooManyCells),
                "{cutoff} {err}"
            );
            assert_ne!(err, Error::Degenerate);
        }
        // A cube the baseline accepts still matches the minimum image.
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let xyz = vec![[0.2, 0.0, 0.0], [9.4, 0.0, 0.0], [5.0, 5.0, 5.0]];
        expect_list(&cell, &xyz, 1.5);
    }

    /// Widths 0.001, 0.001, 10 at cutoff 0.1 pass the image cap and the
    /// bin cap. The reach box would store far more than 2^24 partners.
    #[test]
    fn needle_partner_stencil_is_capped() {
        let cell = Cell::ortho(0.001, 0.001, 10.0).unwrap();
        let xyz = vec![[0.0002, 0.0003, 1.0], [0.0005, 0.0007, 1.04]];
        for half in [false, true] {
            assert_eq!(
                pairs_within(&xyz, &cell, 0.1, None, None, half).unwrap_err(),
                Error::TooManyImages
            );
            let mut columns = PairColumns::default();
            assert_eq!(
                pairs_within_columns(&xyz, &cell, 0.1, None, None, half, &mut columns).unwrap_err(),
                Error::TooManyImages
            );
            let query = PbcQuery::periodic(0.1).half(half);
            assert_eq!(
                pairs_within_pbc(&xyz, &cell, &query).unwrap_err(),
                Error::TooManyImages
            );
        }
        // The CSR gather does not build that stencil. It keeps the
        // nearest image of the other point and drops self-images.
        let mut csr = PairCsr::default();
        pairs_within_csr(
            &xyz,
            &cell,
            &PbcQuery::periodic(0.1).half(true),
            true,
            &mut csr,
        )
        .unwrap();
        assert_eq!(csr.offsets, vec![0, 1, 1]);
        assert_eq!(csr.cols, vec![1]);
        let dx = 0.0003_f64;
        let dy = 0.0004_f64;
        let dz = 0.04_f64;
        let dist2 = dx * dx + dy * dy + dz * dz;
        assert!(close_dist(csr.dist2[0], dist2), "{}", csr.dist2[0]);
    }

    /// Cells whose partner box is past 2^24 slots and that the previous
    /// revision searched successfully. The stencil cap has to admit them.
    /// Side 52 at cutoff 1 is reach two: 52^3 * 5^3 = 17_576_000 slots.
    /// Widths 0.01, 0.01, 2.12 at cutoff 0.2 keep reach 21 on every axis:
    /// 212 * 43^3 = 16_855_484 slots. Both sit over 2^24 and under the
    /// image and bin caps. Side 51 and length 2.11 fall under 2^24.
    #[test]
    fn wide_partner_stencils_match_brute_force() {
        let cell = Cell::ortho(52.0, 52.0, 52.0).unwrap();
        let xyz = vec![[0.2, 0.2, 0.2], [0.7, 0.2, 0.2]];
        expect_list(&cell, &xyz, 1.0);

        let cell = Cell::ortho(0.01, 0.01, 2.12).unwrap();
        let xyz = vec![[0.002, 0.003, 1.0], [0.005, 0.007, 1.05]];
        expect_list(&cell, &xyz, 0.2);
    }

    #[cfg(feature = "capi")]
    #[test]
    fn flat_and_needle_c_api_are_typed_errors() {
        use std::ffi::CStr;

        fn raw(cell: &Cell) -> crate::lc_cell {
            let h = cell.h();
            crate::lc_cell {
                ax: h[0][0],
                ay: h[0][1],
                az: h[0][2],
                bx: h[1][0],
                by: h[1][1],
                bz: h[1][2],
                cx: h[2][0],
                cy: h[2][1],
                cz: h[2][2],
                ox: 0.0,
                oy: 0.0,
                oz: 0.0,
            }
        }

        fn expect_message(cell: &Cell, xyz: &[[f64; 3]], cutoff: f64, message: &str) {
            let raw = raw(cell);
            let flat: Vec<f64> = xyz.iter().copied().flatten().collect();
            let mut count = 0usize;
            let status = unsafe {
                crate::lc_pairs_within_pbc(
                    flat.as_ptr(),
                    xyz.len(),
                    &raw,
                    std::ptr::null(),
                    cutoff,
                    std::ptr::null(),
                    0.0,
                    0,
                    0,
                    0,
                    std::ptr::null_mut(),
                    0,
                    &mut count,
                )
            };
            assert_ne!(status, 0);
            let got = unsafe { CStr::from_ptr(crate::lc_last_error()) };
            assert_eq!(got.to_str().unwrap(), message);
            let status = unsafe {
                crate::lc_pairs_within(
                    flat.as_ptr(),
                    xyz.len(),
                    &raw,
                    cutoff,
                    std::ptr::null(),
                    0.0,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    &mut count,
                )
            };
            assert_ne!(status, 0);
            let got = unsafe { CStr::from_ptr(crate::lc_last_error()) };
            assert_eq!(got.to_str().unwrap(), message);
        }

        let thin = 1.1e-5;
        let cell = Cell::ortho(10.0, 10.0, thin).unwrap();
        let xyz = vec![[1.0, 2.0, thin * 0.3]; 8];
        expect_message(&cell, &xyz, 10.0, "linked-cell mesh is too fine");

        let cell = Cell::ortho(10.0, 10.0, 1.0e-5).unwrap();
        let xyz = vec![[1.0, 2.0, 1.0e-5 * 0.3]; 4];
        for cutoff in [0.1, 2.6, 10.0] {
            let err = pairs_within(&xyz, &cell, cutoff, None, None, false).unwrap_err();
            expect_message(&cell, &xyz, cutoff, &err.to_string());
        }

        let cell = Cell::ortho(0.001, 0.001, 10.0).unwrap();
        let xyz = vec![[0.0002, 0.0003, 1.0], [0.0005, 0.0007, 1.04]];
        expect_message(&cell, &xyz, 0.1, "periodic image count exceeds 2^24");
    }
}
