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
    if n == 0 {
        return fail(Error::Empty);
    }
    if xyz.is_null() || simbox.is_null() {
        return fail_msg("null pointer");
    }
    if n > c_int::MAX as usize {
        return fail(Error::Overflow);
    }
    let max_xyz = (isize::MAX as usize) / 3;
    if n > max_xyz {
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
    let pts: &[[f64; 3]] = unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n) };
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
    let pairs = match crate::pairs_within(pts, &sim, cutoff, mask_vec.as_deref(), hint, half != 0) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    unsafe {
        *out_count = pairs.len();
    }
    let query = out_i.is_null() && out_j.is_null() && out_shift.is_null() && out_d2.is_null();
    if query {
        clear_error();
        return 0;
    }
    if out_i.is_null() || out_j.is_null() || out_shift.is_null() || out_d2.is_null() {
        return fail_msg("null pointer");
    }
    if cap < pairs.len() {
        return fail_msg("pair buffer is shorter than the pair count");
    }
    let shift_len = match pairs.len().checked_mul(3) {
        Some(v) if v <= isize::MAX as usize => v,
        _ => return fail(Error::Overflow),
    };
    let ii = unsafe { std::slice::from_raw_parts_mut(out_i, pairs.len()) };
    let jj = unsafe { std::slice::from_raw_parts_mut(out_j, pairs.len()) };
    let ss = unsafe { std::slice::from_raw_parts_mut(out_shift, shift_len) };
    let dd = unsafe { std::slice::from_raw_parts_mut(out_d2, pairs.len()) };
    for (t, pair) in pairs.iter().enumerate() {
        ii[t] = pair.i as c_int;
        jj[t] = pair.j as c_int;
        ss[3 * t] = pair.shift[0];
        ss[3 * t + 1] = pair.shift[1];
        ss[3 * t + 2] = pair.shift[2];
        dd[t] = pair.dist2;
    }
    clear_error();
    0
}
