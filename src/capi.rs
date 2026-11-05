//! C ABI. Prefix `lc_`. Caller owns every buffer.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CString};
use std::ptr;

use crate::{Cell, Error};

/// Periodic parallelepiped. Lattice vectors are a, b, c (same as vesin rows).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct lc_cell {
    /// Lattice vector a, x.
    pub ax: f64,
    /// Lattice vector a, y.
    pub ay: f64,
    /// Lattice vector a, z.
    pub az: f64,
    /// Lattice vector b, x.
    pub bx: f64,
    /// Lattice vector b, y.
    pub by: f64,
    /// Lattice vector b, z.
    pub bz: f64,
    /// Lattice vector c, x.
    pub cx: f64,
    /// Lattice vector c, y.
    pub cy: f64,
    /// Lattice vector c, z.
    pub cz: f64,
    /// Origin x.
    pub ox: f64,
    /// Origin y.
    pub oy: f64,
    /// Origin z.
    pub oz: f64,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: &str) {
    LAST_ERROR.with(|slot| {
        let cstr = CString::new(msg).unwrap_or_else(|_| {
            CString::new("error message contained NUL").expect("fallback has no NUL")
        });
        *slot.borrow_mut() = Some(cstr);
    });
}

fn clear_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn fail(err: Error) -> c_int {
    set_error(&err.to_string());
    1
}

fn fail_msg(msg: &str) -> c_int {
    set_error(msg);
    1
}

/// Thread-local last-error string from this thread's most recent
/// [`lc_knearest`].
///
/// Returns a pointer to a NUL-terminated UTF-8 C string, or `NULL` if
/// the last `lc_knearest` on this thread succeeded, or if none has
/// failed yet. [`lc_version`] does not read or write the slot.
/// Distinct threads have independent slots. The pointer is valid until
/// the next `lc_knearest` on this thread. Do not free it.
#[no_mangle]
pub extern "C" fn lc_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

/// Library version string. Process-static, NUL-terminated. Do not free.
#[no_mangle]
pub extern "C" fn lc_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// k-nearest neighbours for `n` points.
///
/// `xyz` is packed row-major `n` triples `(x, y, z)` (`n * 3` doubles).
/// `n` is the point count (`size_t`); `0` is an error.
/// `simbox` is the periodic cell (lattice vectors a, b, c and origin).
/// `k` is neighbours per source (`size_t`); `0` is an error.
/// `mask` of `NULL` includes every point; otherwise `n` ints, nonzero to
/// include that point as both source and candidate.
/// `cell_hint` is the target cell edge. Values `<= 0` select the default
/// (3.0 in the box units).
/// `out_nn` is caller-owned output. Neighbours of source `i` occupy
/// `out_nn[i*k + t]`, nearest first. Missing slots are `-1`. Length is
/// `n * k` ints.
///
/// `xyz`, `simbox`, and `out_nn` must be non-null when `n > 0` and
/// `k > 0`. `mask` may be `NULL`.
///
/// Returns `0` on success. Nonzero on failure. Read `lc_last_error()` on
/// the same thread. The last-error string is thread-local: it is not
/// shared across threads, is valid until the next `lc_*` call on this
/// thread, and must not be freed.
///
/// # Safety
///
/// `xyz` is aligned for `double` and readable for `n * 3` doubles.
/// `simbox` is aligned and points at one valid `lc_cell`.
/// `out_nn` is aligned for `int` and writable for `n * k` ints.
/// `mask`, if non-null, is aligned for `int` and readable for `n` ints.
/// `n * 3` and `n * k` fit in `size_t`.
#[no_mangle]
pub unsafe extern "C" fn lc_knearest(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    k: usize,
    mask: *const c_int,
    cell_hint: f64,
    out_nn: *mut c_int,
) -> c_int {
    // SAFETY: same contract as lc_knearest_d2 with a single frame and
    // no distance buffer.
    unsafe {
        lc_knearest_many(
            xyz,
            n,
            1,
            simbox,
            k,
            mask,
            cell_hint,
            out_nn,
            ptr::null_mut(),
        )
    }
}

/// Like [`lc_knearest`], and write squared distances into `out_d2`.
///
/// `out_d2` is caller-owned, length `n * k`. Unused slots are `NaN`.
/// `out_d2` may be `NULL` to skip distances (same as [`lc_knearest`]).
///
/// # Safety
///
/// Same pointer contract as [`lc_knearest`]. `out_d2`, if non-null,
/// is aligned for `double` and writable for `n * k` doubles.
#[no_mangle]
pub unsafe extern "C" fn lc_knearest_d2(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    k: usize,
    mask: *const c_int,
    cell_hint: f64,
    out_nn: *mut c_int,
    out_d2: *mut f64,
) -> c_int {
    unsafe { lc_knearest_many(xyz, n, 1, simbox, k, mask, cell_hint, out_nn, out_d2) }
}

/// Frame-major batch. `xyz` is `n_frames * n` packed triples.
/// `out_nn` / `out_d2` are `n_frames * n * k`. One shared cell.
/// `mask` is length `n` or `NULL`. `out_d2` may be `NULL`.
///
/// # Safety
///
/// `xyz` is readable for `n_frames * n * 3` doubles. `out_nn` is
/// writable for `n_frames * n * k` ints. `out_d2`, if non-null, is
/// writable for `n_frames * n * k` doubles. `simbox` and `mask`
/// follow [`lc_knearest`].
#[no_mangle]
pub unsafe extern "C" fn lc_knearest_many(
    xyz: *const f64,
    n: usize,
    n_frames: usize,
    simbox: *const lc_cell,
    k: usize,
    mask: *const c_int,
    cell_hint: f64,
    out_nn: *mut c_int,
    out_d2: *mut f64,
) -> c_int {
    if n == 0 || n_frames == 0 {
        return fail(Error::Empty);
    }
    if k == 0 {
        return fail(Error::ZeroK);
    }
    if xyz.is_null() || simbox.is_null() || out_nn.is_null() {
        return fail_msg("null pointer");
    }
    let Some(n_pts) = n.checked_mul(n_frames) else {
        return fail(Error::Overflow);
    };
    let Some(need) = n_pts.checked_mul(k) else {
        return fail(Error::Overflow);
    };
    let max_xyz = (isize::MAX as usize) / 3;
    let max_out = (isize::MAX as usize) / std::mem::size_of::<c_int>();
    if n_pts > max_xyz || need > max_out {
        return fail(Error::Overflow);
    }
    let box_c = unsafe { *simbox };
    let sim = match Cell::from_vectors(
        [box_c.ax, box_c.ay, box_c.az],
        [box_c.bx, box_c.by, box_c.bz],
        [box_c.cx, box_c.cy, box_c.cz],
        [box_c.ox, box_c.oy, box_c.oz],
    ) {
        Ok(b) => b,
        Err(e) => {
            set_error(&e.to_string());
            return 1;
        }
    };
    let pts: &[[f64; 3]] = unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n_pts) };
    let mask_vec: Option<Vec<bool>> = if mask.is_null() {
        None
    } else {
        let raw = unsafe { std::slice::from_raw_parts(mask, n) };
        Some(raw.iter().map(|&v| v != 0).collect())
    };
    let hint = if cell_hint > 0.0 {
        Some(cell_hint)
    } else {
        None
    };
    let nn = unsafe { std::slice::from_raw_parts_mut(out_nn, need) };
    let d2_store;
    let d2 = if out_d2.is_null() {
        None
    } else {
        d2_store = unsafe { std::slice::from_raw_parts_mut(out_d2, need) };
        Some(&mut d2_store[..])
    };
    let err = if n_frames == 1 {
        crate::knearest_into_d2(pts, &sim, k, mask_vec.as_deref(), hint, nn, d2)
    } else {
        crate::knearest_into_many(pts, n, n_frames, &sim, k, mask_vec.as_deref(), hint, nn, d2)
    };
    if let Err(e) = err {
        set_error(&e.to_string());
        return 1;
    }
    clear_error();
    0
}

/// Cutoff pairs with the vesin / tonari shift `S`.
///
/// Each row is one atom-image: `out_i[t]`, `out_j[t]`,
/// `out_shift[3*t + 0..3]` = `(na, nb, nc)`, and `out_d2[t]`.
/// Displacement is `r_j - r_i + S H` in the caller's basis.
/// `dist2` is strictly below `cutoff` squared. `half` nonzero keeps
/// the canonical side of `(i, j, S)` versus `(j, i, -S)`.
///
/// Pass null `out_i`, `out_j`, `out_shift`, and `out_d2` to query.
/// `*out_count` receives the row count and the return is 0.
/// Otherwise all four buffers are required and `cap` is their row
/// capacity. A short buffer sets `*out_count` to the needed row count
/// and returns nonzero without writing.
///
/// A query, or a short buffer, keeps its search on the calling thread.
/// The next call on that thread takes it when every input is the same
/// bit for bit (`xyz`, the box, `cutoff`, `mask`, `cell_hint`, `half`)
/// and writes the rows without searching again. Any other call drops
/// it. A fill that writes does not keep anything.
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `simbox` points at one
/// `lc_cell`. `mask`, if non-null, is readable for `n` ints.
/// `out_count` is writable. On a fill, `out_i` and `out_j` are
/// writable for `cap` ints, `out_shift` for `cap * 3` ints, and
/// `out_d2` for `cap` doubles.
#[no_mangle]
pub unsafe extern "C" fn lc_pairs_within(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    half: c_int,
    out_i: *mut c_int,
    out_j: *mut c_int,
    out_shift: *mut c_int,
    out_d2: *mut f64,
    cap: usize,
    out_count: *mut usize,
) -> c_int {
    if out_count.is_null() {
        return fail_msg("null pointer");
    }
    let (key, found) = match unsafe { pairs_call(xyz, n, simbox, cutoff, mask, cell_hint, half) } {
        Ok(v) => v,
        Err(code) => return code,
    };
    let rows = found.rows();
    unsafe {
        *out_count = rows;
    }
    let query = out_i.is_null() && out_j.is_null() && out_shift.is_null() && out_d2.is_null();
    if query {
        park(&key, found);
        clear_error();
        return 0;
    }
    if out_i.is_null() || out_j.is_null() || out_shift.is_null() || out_d2.is_null() {
        return fail_msg("null pointer");
    }
    if cap < rows {
        park(&key, found);
        return fail_msg("pair buffer is shorter than the pair count");
    }
    match rows.checked_mul(3) {
        Some(v) if v <= isize::MAX as usize => {}
        _ => return fail(Error::Overflow),
    }
    // Safety: the caller gave `cap >= rows` slots in each buffer and
    // three times that in `out_shift`.
    let cols = ColumnOut(out_i, out_j, out_shift, out_d2);
    crate::pairs::in_pool(found.rows(), move || {
        let (i, j, s, d) = cols.get();
        unsafe { found.write_columns(i, j, s, d) }
    });
    clear_error();
    0
}

/// One cutoff row: source `i`, target `j`, shift `S`, squared distance.
/// The C++ `linkcell::ShiftedPair` has this layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct lc_pair {
    /// Source index.
    pub i: c_int,
    /// Target index.
    pub j: c_int,
    /// Integer cell shift `(na, nb, nc)` applied to the target.
    pub shift: [c_int; 3],
    /// Squared distance after the shift.
    pub dist2: f64,
}

/// [`lc_pairs_within`] with one array of [`lc_pair`] rows.
///
/// Null `out` is a query: `*out_count` receives the row count and the
/// return is 0. Otherwise `out` holds `cap` rows. A short buffer sets
/// `*out_count` to the needed count and returns nonzero without
/// writing. A query or a short buffer keeps its search on the calling
/// thread for the next call with the same inputs, as in
/// [`lc_pairs_within`].
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `simbox` points at one
/// `lc_cell`. `mask`, if non-null, is readable for `n` ints.
/// `out_count` is writable. On a fill, `out` is writable for `cap`
/// rows.
#[no_mangle]
pub unsafe extern "C" fn lc_pairs_within_rows(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    half: c_int,
    out: *mut lc_pair,
    cap: usize,
    out_count: *mut usize,
) -> c_int {
    if out_count.is_null() {
        return fail_msg("null pointer");
    }
    let (key, found) = match unsafe { pairs_call(xyz, n, simbox, cutoff, mask, cell_hint, half) } {
        Ok(v) => v,
        Err(code) => return code,
    };
    let rows = found.rows();
    unsafe {
        *out_count = rows;
    }
    if out.is_null() {
        park(&key, found);
        clear_error();
        return 0;
    }
    if cap < rows {
        park(&key, found);
        return fail_msg("pair buffer is shorter than the pair count");
    }
    // Safety: the caller gave `cap >= rows` rows.
    let rows_out = RowOut(out);
    crate::pairs::in_pool(found.rows(), move || {
        let out = rows_out.get();
        unsafe { found.write_rows(out, |i, j, shift, dist2| lc_pair { i, j, shift, dist2 }) }
    });
    clear_error();
    0
}

/// Check the inputs, then take the parked search or run a new one.
unsafe fn pairs_call<'a>(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    half: c_int,
) -> Result<(PairKey<'a>, crate::pairs::Found), c_int> {
    if n == 0 {
        return Err(fail(Error::Empty));
    }
    if xyz.is_null() || simbox.is_null() {
        return Err(fail_msg("null pointer"));
    }
    if n > c_int::MAX as usize || n > (isize::MAX as usize) / 3 {
        return Err(fail(Error::Overflow));
    }
    let box_c = unsafe { *simbox };
    let sim = Cell::from_vectors(
        [box_c.ax, box_c.ay, box_c.az],
        [box_c.bx, box_c.by, box_c.bz],
        [box_c.cx, box_c.cy, box_c.cz],
        [box_c.ox, box_c.oy, box_c.oz],
    )
    .map_err(|e| {
        set_error(&e.to_string());
        1
    })?;
    let pts: &'a [[f64; 3]] = unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n) };
    let mask_raw: Option<&'a [c_int]> = if mask.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(mask, n) })
    };
    let hint = if cell_hint > 0.0 {
        Some(cell_hint)
    } else {
        None
    };
    let key = PairKey {
        xyz: pts,
        simbox: box_c,
        cutoff,
        mask: mask_raw,
        cell_hint: hint,
        half: half != 0,
    };
    if let Some(found) = take_parked(&key) {
        return Ok((key, found));
    }
    let mask_vec: Option<Vec<bool>> = mask_raw.map(|m| m.iter().map(|&v| v != 0).collect());
    let found = crate::pairs::in_pool(crate::pairs::hit_estimate(n, &sim, cutoff), || {
        crate::pairs::search(pts, &sim, cutoff, mask_vec.as_deref(), hint, half != 0)
    });
    match found {
        Ok(found) => Ok((key, found)),
        Err(e) => Err(fail(e)),
    }
}

/// Caller buffers moved onto a pool thread for the fill. The caller
/// owns them, so the fill can run where the search ran.
#[derive(Clone, Copy)]
struct ColumnOut(*mut c_int, *mut c_int, *mut c_int, *mut f64);
unsafe impl Send for ColumnOut {}
impl ColumnOut {
    fn get(self) -> (*mut c_int, *mut c_int, *mut c_int, *mut f64) {
        (self.0, self.1, self.2, self.3)
    }
}

#[derive(Clone, Copy)]
struct RowOut(*mut lc_pair);
unsafe impl Send for RowOut {}
impl RowOut {
    fn get(self) -> *mut lc_pair {
        self.0
    }
}

/// Inputs of one [`lc_pairs_within`] call, compared bit for bit.
struct PairKey<'a> {
    xyz: &'a [[f64; 3]],
    simbox: lc_cell,
    cutoff: f64,
    mask: Option<&'a [c_int]>,
    cell_hint: Option<f64>,
    half: bool,
}

/// A query's search, kept until the matching fill on the same thread.
struct ParkedPairs {
    xyz: Vec<[f64; 3]>,
    simbox: [u64; 12],
    cutoff: u64,
    mask: Option<Vec<bool>>,
    cell_hint: Option<u64>,
    half: bool,
    found: crate::pairs::Found,
}

thread_local! {
    // `const { ... }` is newer than this crate's 1.70 floor.
    #[allow(clippy::missing_const_for_thread_local)]
    static PARKED_PAIRS: RefCell<Option<ParkedPairs>> = RefCell::new(None);
}

fn box_bits(c: &lc_cell) -> [u64; 12] {
    [
        c.ax.to_bits(),
        c.ay.to_bits(),
        c.az.to_bits(),
        c.bx.to_bits(),
        c.by.to_bits(),
        c.bz.to_bits(),
        c.cx.to_bits(),
        c.cy.to_bits(),
        c.cz.to_bits(),
        c.ox.to_bits(),
        c.oy.to_bits(),
        c.oz.to_bits(),
    ]
}

fn same_bits(a: &[[f64; 3]], b: &[[f64; 3]]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(p, q)| {
            p[0].to_bits() == q[0].to_bits()
                && p[1].to_bits() == q[1].to_bits()
                && p[2].to_bits() == q[2].to_bits()
        })
}

/// The parked search when every input matches. Any other call drops it,
/// so a parked search never outlives the next call on this thread.
fn take_parked(key: &PairKey<'_>) -> Option<crate::pairs::Found> {
    let parked = PARKED_PAIRS.with(|slot| slot.borrow_mut().take())?;
    let mask_same = match (&parked.mask, key.mask) {
        (None, None) => true,
        (Some(a), Some(b)) => a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| x == (y != 0)),
        _ => false,
    };
    let same = parked.half == key.half
        && parked.cutoff == key.cutoff.to_bits()
        && parked.cell_hint == key.cell_hint.map(f64::to_bits)
        && parked.simbox == box_bits(&key.simbox)
        && mask_same
        && same_bits(&parked.xyz, key.xyz);
    if same {
        Some(parked.found)
    } else {
        None
    }
}

fn park(key: &PairKey<'_>, found: crate::pairs::Found) {
    let parked = ParkedPairs {
        xyz: key.xyz.to_vec(),
        simbox: box_bits(&key.simbox),
        cutoff: key.cutoff.to_bits(),
        mask: key.mask.map(|m| m.iter().map(|&v| v != 0).collect()),
        cell_hint: key.cell_hint.map(f64::to_bits),
        half: key.half,
        found,
    };
    PARKED_PAIRS.with(|slot| *slot.borrow_mut() = Some(parked));
}

unsafe fn load_frame<'a>(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    mask: *const c_int,
    cell_hint: f64,
) -> Result<(&'a [[f64; 3]], Cell, Option<Vec<bool>>, Option<f64>), c_int> {
    if n == 0 {
        return Err(fail(Error::Empty));
    }
    if xyz.is_null() || simbox.is_null() {
        return Err(fail_msg("null pointer"));
    }
    if n > c_int::MAX as usize || n > (isize::MAX as usize) / 3 {
        return Err(fail(Error::Overflow));
    }
    let box_c = unsafe { *simbox };
    let sim = Cell::from_vectors(
        [box_c.ax, box_c.ay, box_c.az],
        [box_c.bx, box_c.by, box_c.bz],
        [box_c.cx, box_c.cy, box_c.cz],
        [box_c.ox, box_c.oy, box_c.oz],
    )
    .map_err(|err| fail(err.into()))?;
    let pts: &'a [[f64; 3]] = unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n) };
    let mask_vec = if mask.is_null() {
        None
    } else {
        let raw = unsafe { std::slice::from_raw_parts(mask, n) };
        Some(raw.iter().map(|&v| v != 0).collect())
    };
    let hint = if cell_hint > 0.0 { Some(cell_hint) } else { None };
    Ok((pts, sim, mask_vec, hint))
}

fn write_grid_head(grid: &crate::CellGrid, out_nxyz: *mut c_int, out_width: *mut f64) {
    if !out_nxyz.is_null() {
        unsafe {
            *out_nxyz = grid.nx;
            *out_nxyz.add(1) = grid.ny;
            *out_nxyz.add(2) = grid.nz;
        }
    }
    if !out_width.is_null() {
        unsafe {
            *out_width = grid.bin_width[0];
            *out_width.add(1) = grid.bin_width[1];
            *out_width.add(2) = grid.bin_width[2];
        }
    }
}

/// Point counts on the linked-cell mesh.
///
/// `out_nxyz` receives `nx, ny, nz`. `out_width` receives the three bin
/// widths. `out_weight` receives one count per cell, in the order
/// `((iz * ny) + iy) * nx + ix`. A null `out_weight` is a query:
/// `*out_ncell` is the cell count and the return is 0.
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `simbox` points at one
/// `lc_cell`. `mask`, if non-null, is readable for `n` ints.
/// `out_ncell` is writable. On a fill, `out_nxyz` is writable for 3
/// ints, `out_width` for 3 doubles, and `out_weight` for `*out_ncell`
/// counts.
#[no_mangle]
pub unsafe extern "C" fn lc_cell_grid(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    out_nxyz: *mut c_int,
    out_width: *mut f64,
    out_weight: *mut u64,
    cap: usize,
    out_ncell: *mut usize,
) -> c_int {
    if out_ncell.is_null() {
        return fail_msg("null pointer");
    }
    let (pts, sim, mask_vec, hint) =
        match unsafe { load_frame(xyz, n, simbox, mask, cell_hint) } {
            Ok(v) => v,
            Err(rc) => return rc,
        };
    let grid = match crate::cell_grid(pts, &sim, cutoff, mask_vec.as_deref(), hint) {
        Ok(grid) => grid,
        Err(err) => return fail(err),
    };
    let ncell = grid.ncell();
    unsafe { *out_ncell = ncell };
    if out_weight.is_null() {
        clear_error();
        return 0;
    }
    if cap < ncell || out_nxyz.is_null() || out_width.is_null() {
        return fail_msg("cell grid buffer is shorter than the cell count");
    }
    write_grid_head(&grid, out_nxyz, out_width);
    unsafe {
        std::ptr::copy_nonoverlapping(grid.weight.as_ptr(), out_weight, ncell);
    }
    clear_error();
    0
}

/// Periodic cell adjacency. Edge weights are pair counts from
/// `lc_pairs_within` with `half` set.
///
/// A null `out_weight` queries `*out_ncell` and `*out_arcs` and
/// returns 0. Otherwise every output buffer is required. `cap_weight`
/// is the cell capacity and `cap_arcs` is the arc capacity.
/// `out_offsets` has length `ncell + 1`.
///
/// # Safety
///
/// Same point contract as [`lc_cell_grid`]. On a fill, `out_offsets`
/// is writable for `ncell + 1` counts, `out_neighbor` and `out_edge`
/// for `out_arcs` entries.
#[no_mangle]
pub unsafe extern "C" fn lc_cell_graph(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    out_nxyz: *mut c_int,
    out_width: *mut f64,
    out_weight: *mut u64,
    out_offsets: *mut u32,
    out_neighbor: *mut u32,
    out_edge: *mut u64,
    cap_weight: usize,
    cap_arcs: usize,
    out_ncell: *mut usize,
    out_arcs: *mut usize,
) -> c_int {
    if out_ncell.is_null() || out_arcs.is_null() {
        return fail_msg("null pointer");
    }
    let (pts, sim, mask_vec, hint) =
        match unsafe { load_frame(xyz, n, simbox, mask, cell_hint) } {
            Ok(v) => v,
            Err(rc) => return rc,
        };
    let graph = match crate::cell_graph(pts, &sim, cutoff, mask_vec.as_deref(), hint) {
        Ok(graph) => graph,
        Err(err) => return fail(err),
    };
    let ncell = graph.grid.ncell();
    let arcs = graph.neighbor.len();
    unsafe {
        *out_ncell = ncell;
        *out_arcs = arcs;
    }
    if out_weight.is_null() {
        clear_error();
        return 0;
    }
    if cap_weight < ncell
        || cap_arcs < arcs
        || out_nxyz.is_null()
        || out_width.is_null()
        || out_offsets.is_null()
        || out_neighbor.is_null()
        || out_edge.is_null()
    {
        return fail_msg("cell graph buffer is shorter than the graph");
    }
    write_grid_head(&graph.grid, out_nxyz, out_width);
    unsafe {
        std::ptr::copy_nonoverlapping(graph.grid.weight.as_ptr(), out_weight, ncell);
        std::ptr::copy_nonoverlapping(graph.offsets.as_ptr(), out_offsets, ncell + 1);
        if arcs > 0 {
            std::ptr::copy_nonoverlapping(graph.neighbor.as_ptr(), out_neighbor, arcs);
            std::ptr::copy_nonoverlapping(graph.edge_weight.as_ptr(), out_edge, arcs);
        }
    }
    clear_error();
    0
}

/// Domain id of each cell.
///
/// `method` is 0 Morton, 1 Hilbert, 2 recursive coordinate bisection,
/// 3 Hilbert plus Fiduccia-Mattheyses, 4 bisection plus that pass,
/// 5 Scotch, 6 multilevel, 7 PT-Scotch. `balance <= 0` selects the
/// derived bound. `device` nonzero runs the
/// curve split on the CUDA device and returns nonzero when that device
/// is absent. A null `out_domain` queries `*out_ncell`.
///
/// # Safety
///
/// Same point contract as [`lc_cell_grid`]. On a fill, `out_domain` is
/// writable for `ncell` ints.
#[no_mangle]
pub unsafe extern "C" fn lc_partition(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    parts: usize,
    method: u8,
    balance: f64,
    device: c_int,
    out_domain: *mut c_int,
    cap: usize,
    out_ncell: *mut usize,
) -> c_int {
    if out_ncell.is_null() {
        return fail_msg("null pointer");
    }
    if parts == 0 || parts > u32::MAX as usize {
        return fail(Error::BadPartition);
    }
    let (pts, sim, mask_vec, hint) =
        match unsafe { load_frame(xyz, n, simbox, mask, cell_hint) } {
            Ok(v) => v,
            Err(rc) => return rc,
        };
    let method = match crate::partition::method_from_c(method) {
        Ok(method) => method,
        Err(err) => return fail(err),
    };
    let balance = if balance <= 0.0 { None } else { Some(balance) };
    let request = crate::PartRequest {
        parts: parts as u32,
        method,
        balance,
        device: device != 0,
    };
    let assign = match crate::partition(pts, &sim, cutoff, mask_vec.as_deref(), hint, request) {
        Ok(assign) => assign,
        Err(err) => return fail(err),
    };
    unsafe { *out_ncell = assign.len() };
    if out_domain.is_null() {
        clear_error();
        return 0;
    }
    if cap < assign.len() {
        return fail_msg("partition buffer is shorter than the cell count");
    }
    for (i, &domain) in assign.iter().enumerate() {
        if domain > c_int::MAX as u32 {
            return fail(Error::Overflow);
        }
        unsafe { *out_domain.add(i) = domain as c_int };
    }
    clear_error();
    0
}

/// Bin radii for `cutoff * depth`. `out_width` receives three counts.
///
/// # Safety
///
/// `bin_width` is readable for 3 doubles. `out_width` is writable for
/// 3 counts.
#[no_mangle]
pub unsafe extern "C" fn lc_halo_width(
    cutoff: f64,
    depth: u32,
    bin_width: *const f64,
    out_width: *mut u32,
) -> c_int {
    if bin_width.is_null() || out_width.is_null() {
        return fail_msg("null pointer");
    }
    let width = unsafe { [*bin_width, *bin_width.add(1), *bin_width.add(2)] };
    match crate::halo_width(cutoff, depth, width) {
        Ok(radius) => {
            unsafe {
                *out_width = radius[0];
                *out_width.add(1) = radius[1];
                *out_width.add(2) = radius[2];
            }
            clear_error();
            0
        }
        Err(err) => fail(err),
    }
}

/// Ghost cells of `domain`. A null `out_ghost` queries `*out_count`.
///
/// # Safety
///
/// `nxyz` is readable for 3 ints. `assign` is readable for `ncell`
/// ints. `width` is readable for 3 counts. On a fill, `out_ghost` is
/// writable for `*out_count` ints.
#[no_mangle]
pub unsafe extern "C" fn lc_halo(
    nxyz: *const c_int,
    assign: *const c_int,
    ncell: usize,
    domain: c_int,
    width: *const u32,
    out_ghost: *mut c_int,
    cap: usize,
    out_count: *mut usize,
) -> c_int {
    if nxyz.is_null() || assign.is_null() || width.is_null() || out_count.is_null() {
        return fail_msg("null pointer");
    }
    if domain < 0 {
        return fail(Error::BadPartition);
    }
    let nxyz = unsafe { [*nxyz, *nxyz.add(1), *nxyz.add(2)] };
    if nxyz[0] <= 0 || nxyz[1] <= 0 || nxyz[2] <= 0 {
        return fail(Error::BadGraph);
    }
    let expect = (nxyz[0] as usize)
        .saturating_mul(nxyz[1] as usize)
        .saturating_mul(nxyz[2] as usize);
    if expect != ncell {
        return fail(Error::BadGraph);
    }
    let raw = unsafe { std::slice::from_raw_parts(assign, ncell) };
    let mut owned = Vec::with_capacity(ncell);
    for &v in raw {
        if v < 0 {
            return fail(Error::BadPartition);
        }
        owned.push(v as u32);
    }
    let grid = crate::CellGrid {
        nx: nxyz[0],
        ny: nxyz[1],
        nz: nxyz[2],
        bin_width: [0.0; 3],
        weight: vec![0; ncell],
    };
    let radius = unsafe { [*width, *width.add(1), *width.add(2)] };
    let ghosts = match crate::halo(&grid, &owned, domain as u32, radius) {
        Ok(ghosts) => ghosts,
        Err(err) => return fail(err),
    };
    unsafe { *out_count = ghosts.len() };
    if out_ghost.is_null() {
        clear_error();
        return 0;
    }
    if cap < ghosts.len() {
        return fail_msg("halo buffer is shorter than the ghost count");
    }
    for (i, &cell) in ghosts.iter().enumerate() {
        unsafe { *out_ghost.add(i) = cell as c_int };
    }
    clear_error();
    0
}
