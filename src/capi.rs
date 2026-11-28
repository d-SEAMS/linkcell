//! C ABI. Prefix `lc_`. Caller owns every buffer.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{Cell, Error, NeighbourIndex};

/// How many Rayon threads the next search should use. Zero is unset: the
/// search runs from the calling thread with rayon's and OpenMP's own
/// defaults. One runs on the calling thread alone.
static WANTED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// One pool for the process. A different width drops the previous pool
/// before the new one starts, so two widths never run together. The
/// workers sleep on a condvar between searches (rayon's idle path),
/// which is the passive wait. A one-thread search starts no pool, builds
/// no global pool and starts no OpenMP team.
#[cfg(not(feature = "parallel"))]
fn with_threads<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    work()
}

#[cfg(feature = "parallel")]
struct Slot {
    n: usize,
    pool: rayon::ThreadPool,
}

#[cfg(feature = "parallel")]
static CACHE: std::sync::Mutex<Option<Slot>> = std::sync::Mutex::new(None);

#[cfg(feature = "parallel")]
fn with_threads<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    let n = WANTED_THREADS.load(Ordering::Relaxed);
    if n == 0 {
        return work();
    }
    if n == 1 {
        return crate::run_on_caller(work);
    }
    if rayon::current_thread_index().is_some() && rayon::current_num_threads() == n {
        return work();
    }
    let mut guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
    if guard.as_ref().map(|slot| slot.n) != Some(n) {
        *guard = None;
        let pool = crate::place::build_pool(n);
        *guard = Some(Slot { n, pool });
    }
    guard.as_ref().unwrap().pool.install(work)
}

/// Run a k-nearest search on the calling thread, with the workers of the
/// pool [`lc_set_threads`] chose joining it (see `team`). The caller does
/// not hand the search to a pool thread and sleep: it is one of the
/// workers, and the call returns when the last stage closes.
#[cfg(not(feature = "parallel"))]
fn with_team<R>(work: impl FnOnce() -> R) -> R {
    work()
}

#[cfg(feature = "parallel")]
fn with_team<R>(work: impl FnOnce() -> R) -> R {
    let n = WANTED_THREADS.load(Ordering::Relaxed);
    if n == 0 || (rayon::current_thread_index().is_some() && rayon::current_num_threads() == n) {
        return work();
    }
    if n == 1 {
        return crate::run_on_caller(|| crate::team::install(1, &|job| job(), work));
    }
    let mut guard = CACHE.lock().unwrap_or_else(|err| err.into_inner());
    if guard.as_ref().map(|slot| slot.n) != Some(n) {
        *guard = None;
        let pool = crate::place::build_pool(n);
        *guard = Some(Slot { n, pool });
    }
    let pool = &guard.as_ref().unwrap().pool;
    crate::team::install(n, &|job| pool.spawn(job), work)
}

/// Choose the Rayon width for later searches. `n == 0` leaves it unset:
/// the search starts on the calling thread with Rayon's and OpenMP's own
/// widths. `n == 1` runs the whole search on the calling thread and
/// starts no thread. `n < 0` selects one thread.
#[no_mangle]
pub extern "C" fn lc_set_threads(n: c_int) {
    let n = if n < 0 { 1 } else { n as usize };
    WANTED_THREADS.store(n, Ordering::Relaxed);
}

/// Width selected by [`lc_set_threads`], or the global pool when unset.
#[no_mangle]
pub extern "C" fn lc_num_threads() -> c_int {
    let n = WANTED_THREADS.load(Ordering::Relaxed);
    if n == 0 {
        default_width() as c_int
    } else {
        n as c_int
    }
}

/// Width of the global Rayon pool, or one thread without `parallel`.
fn default_width() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

#[derive(Clone, Copy)]
struct Continuation {
    ctx: usize,
    cb: usize,
}

thread_local! {
    /// A callback the calling thread's next search runs before its pool
    /// goes idle, so the CSR merge does not pay a second wake. Each thread
    /// sets and clears its own hook, so concurrent searches on different
    /// threads never run each other's callback.
    static CONTINUATION: std::cell::Cell<Option<Continuation>> = const { std::cell::Cell::new(None) };
}

/// The hook the calling thread set, read before the search enters the pool.
fn take_continuation() -> Option<Continuation> {
    CONTINUATION.with(|slot| slot.get())
}

fn run_continuation(cont: Option<Continuation>) {
    let Some(cont) = cont else {
        return;
    };
    let cb: extern "C" fn(*mut std::ffi::c_void) = unsafe { std::mem::transmute(cont.cb) };
    cb(cont.ctx as *mut std::ffi::c_void);
}

/// Run `cb` at the end of this thread's next `lc_knearest`, still inside
/// its pool. A null `cb` clears it. The hook belongs to the calling
/// thread; a search started on another thread does not run it. The search
/// clears nothing by itself: the caller drops the hook when the call
/// returns.
#[no_mangle]
pub extern "C" fn lc_set_continuation(
    ctx: *mut std::ffi::c_void,
    cb: Option<extern "C" fn(*mut std::ffi::c_void)>,
) {
    CONTINUATION.with(|slot| {
        slot.set(cb.map(|cb| Continuation {
            ctx: ctx as usize,
            cb: cb as usize,
        }))
    });
}

/// Install the device curve split used by the `device` partition. The
/// GPU library calls this from a static initializer when it is loaded. A
/// null `f` removes it, and the device partition then reports no device.
#[no_mangle]
pub extern "C" fn lc_set_gpu_curve_split(
    f: Option<
        unsafe extern "C" fn(
            atom_bin: *const u32,
            natom: usize,
            coords: *const u32,
            ncell: usize,
            parts: u32,
            assign: *mut u32,
            out_weight: *mut u64,
            method: u8,
        ) -> i32,
    >,
) {
    crate::partition::register_device_curve(f);
}

/// Run `cb(ctx, begin, end)` over disjoint spans of `0..n` on the pool
/// selected by [`lc_set_threads`]. One thread runs the whole span on the
/// caller. The spans cover `0..n` exactly once.
#[no_mangle]
pub extern "C" fn lc_for_slabs(
    n: c_int,
    ctx: *mut std::ffi::c_void,
    cb: Option<extern "C" fn(*mut std::ffi::c_void, c_int, c_int)>,
) {
    let Some(cb) = cb else {
        return;
    };
    if n <= 0 {
        return;
    }
    let ctx_bits = ctx as usize;
    let cb_bits = cb as usize;
    with_threads(move || {
        let call = |begin: c_int, end: c_int| {
            let ctx = ctx_bits as *mut std::ffi::c_void;
            let cb: extern "C" fn(*mut std::ffi::c_void, c_int, c_int) =
                unsafe { std::mem::transmute(cb_bits) };
            cb(ctx, begin, end);
        };
        #[cfg(not(feature = "parallel"))]
        call(0, n);
        #[cfg(feature = "parallel")]
        if WANTED_THREADS.load(Ordering::Relaxed) <= 1 {
            call(0, n);
        } else {
            rayon::broadcast(move |worker| {
                let workers = worker.num_threads().max(1);
                let t = worker.index();
                let begin = (n as usize) * t / workers;
                let end = (n as usize) * (t + 1) / workers;
                call(begin as c_int, end as c_int);
            });
        }
    });
}

/// Workers an [`lc_team_run`] started now may use: the `worker` index it
/// passes is below this.
#[no_mangle]
pub extern "C" fn lc_team_width() -> usize {
    with_team(|| {
        crate::team::with_current(|width, spawn| if spawn.is_some() { width.max(1) } else { 1 })
    })
}

/// Run `nstages` stages in one parallel region on the pool selected by
/// [`lc_set_threads`], the caller one of the workers (see `team`). Stage
/// `s` is `chunks[s]` calls of `fn_(ctx, s, chunk, worker)`, in any order
/// and on any worker; every chunk of a stage returns before any chunk of
/// the next starts. `worker` is fixed for one worker thread during the
/// call and below [`lc_team_width`], so it can index per-worker scratch.
/// No worker that has not started holds up a stage or the return.
///
/// # Safety
///
/// `chunks` is readable for `nstages` sizes. `fn_` does not unwind and
/// may be called from any thread of the pool until the call returns.
#[no_mangle]
pub unsafe extern "C" fn lc_team_run(
    nstages: usize,
    chunks: *const usize,
    fn_: Option<extern "C" fn(*mut std::ffi::c_void, usize, usize, usize)>,
    ctx: *mut std::ffi::c_void,
) -> c_int {
    let Some(fn_) = fn_ else {
        return fail_msg("null pointer");
    };
    if nstages == 0 {
        clear_error();
        return 0;
    }
    if chunks.is_null() {
        return fail_msg("null pointer");
    }
    let sizes = unsafe { std::slice::from_raw_parts(chunks, nstages) }.to_vec();
    let ctx_bits = ctx as usize;
    let workers = AtomicUsize::new(0);
    let runs: Vec<_> = (0..nstages)
        .map(|stage| {
            let workers = &workers;
            move |worker: &mut Option<usize>, chunk: usize| {
                let w = *worker.get_or_insert_with(|| workers.fetch_add(1, Ordering::Relaxed));
                fn_(ctx_bits as *mut std::ffi::c_void, stage, chunk, w);
            }
        })
        .collect();
    let stages: Vec<crate::team::Stage<'_, Option<usize>>> = sizes
        .iter()
        .zip(&runs)
        .map(|(&chunks, run)| crate::team::Stage { chunks, run })
        .collect();
    with_team(|| crate::team::with_current(|width, spawn| crate::team::run(width, spawn, &stages)));
    clear_error();
    0
}

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

/// Thread-local last-error string shared by the status-returning `lc_*`
/// calls in `linkcell.h`.
///
/// Returns a pointer to a NUL-terminated UTF-8 C string, or `NULL` if
/// the last such call on this thread succeeded, or if none has failed
/// yet. [`lc_version`] does not read or write the slot.
/// Distinct threads have independent slots. The pointer is valid until
/// the next such `lc_*` call on this thread. Do not free it.
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

struct NdHandle {
    tree: crate::VpTreeIndex<crate::OwnedContiguousSpace, crate::PeriodicSpace>,
}

/// Opaque N-dimensional neighbour index handle.
///
/// Callers only ever hold a pointer returned by `lc_nd_build`. The byte
/// makes this a complete object type in C99 and C++17. A zero-length array
/// is not one.
#[repr(C)]
pub struct lc_nd_index {
    _private: u8,
}

unsafe fn nd_ref<'a>(ptr: *const lc_nd_index) -> Result<&'a NdHandle, c_int> {
    if ptr.is_null() {
        return Err(fail_msg("null pointer"));
    }
    Ok(unsafe { &*(ptr as *const NdHandle) })
}

/// Build an N-dimensional exact neighbour index from row-major `n * d` data.
///
/// `periods` is null for a fully open space, or length `d` with
/// `periods[k] > 0` for periodic axis `k`, `periods[k] == 0` for open axis,
/// and `periods[k] < 0` rejected.
/// The index owns a copy of `data` and is released by [`lc_nd_free`].
/// On failure, `*out` is set to null.
///
/// The period table is one optional period per axis, 16 bytes. `d` greater
/// than what fits in `isize::MAX` bytes fails with `the period table does
/// not fit in isize::MAX bytes` before any allocation. A `d` that passes
/// that check still fails with `the period table could not be allocated`
/// when the allocator refuses the table. `n * d` that does not fit in
/// `usize` fails with `n * d overflows` before the coordinates are read.
///
/// # Safety
///
/// When `n * d` fits in `usize`, `data` is readable for that many doubles.
/// `periods`, if non-null, is readable for `d` doubles. `out` is writable
/// for one pointer.
#[no_mangle]
pub unsafe extern "C" fn lc_nd_build(
    data: *const f64,
    n: usize,
    d: usize,
    periods: *const f64,
    out: *mut *mut lc_nd_index,
) -> c_int {
    if out.is_null() {
        return fail_msg("null pointer");
    }
    unsafe {
        *out = ptr::null_mut();
    }
    if data.is_null() || d == 0 {
        return fail_msg("null pointer");
    }
    let max_d = (isize::MAX as usize) / std::mem::size_of::<Option<f64>>();
    if d > max_d {
        return fail_msg("the period table does not fit in isize::MAX bytes");
    }
    let Some(len) = n.checked_mul(d) else {
        return fail(Error::NdProduct);
    };
    let raw = unsafe { std::slice::from_raw_parts(data, len) };
    if !raw.iter().all(|v| v.is_finite()) {
        return fail(Error::NonFinite);
    }
    let mut period_vec = Vec::new();
    if period_vec.try_reserve_exact(d).is_err() {
        return fail_msg("the period table could not be allocated");
    }
    if periods.is_null() {
        period_vec.resize(d, None);
    } else {
        let p = unsafe { std::slice::from_raw_parts(periods, d) };
        if p.iter().any(|&v| v.is_nan() || v < 0.0) {
            return fail(Error::BadPeriod);
        }
        for &v in p {
            period_vec.push(if v > 0.0 { Some(v) } else { None });
        }
    }
    let space = match crate::OwnedContiguousSpace::from_row_major(raw, n, d) {
        Ok(space) => space,
        Err(err) => return fail(err),
    };
    let metric = match crate::PeriodicSpace::new(period_vec) {
        Ok(metric) => metric,
        Err(err) => return fail(err),
    };
    let tree = match crate::VpTreeIndex::new(space, metric) {
        Ok(tree) => tree,
        Err(err) => return fail(err),
    };
    let boxed = Box::new(NdHandle { tree });
    unsafe {
        *out = Box::into_raw(boxed) as *mut lc_nd_index;
    }
    clear_error();
    0
}

/// Number of indexed points in an N-dimensional index.
///
/// # Safety
///
/// `index` is null or a valid pointer returned by [`lc_nd_build`].
#[no_mangle]
pub unsafe extern "C" fn lc_nd_len(index: *const lc_nd_index) -> usize {
    match unsafe { nd_ref(index) } {
        Ok(h) => h.tree.len(),
        Err(_) => 0,
    }
}

/// Point dimensionality of an N-dimensional index.
///
/// # Safety
///
/// `index` is null or a valid pointer returned by [`lc_nd_build`].
#[no_mangle]
pub unsafe extern "C" fn lc_nd_dims(index: *const lc_nd_index) -> usize {
    match unsafe { nd_ref(index) } {
        Ok(h) => h.tree.dims(),
        Err(_) => 0,
    }
}

/// Free an index built by [`lc_nd_build`]. Null is accepted.
///
/// # Safety
///
/// `index` is null or a pointer returned by [`lc_nd_build`] and not already
/// freed.
#[no_mangle]
pub unsafe extern "C" fn lc_nd_free(index: *mut lc_nd_index) {
    if index.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(index as *mut NdHandle));
    }
}

/// k-nearest neighbours for one query point.
///
/// Writes exactly `k` slots: found neighbours first, then `-1` / `NaN`.
///
/// # Safety
///
/// `query` is readable for `lc_nd_dims(index)` doubles. `out_idx` is writable
/// for `k` ints. `out_d2`, if non-null, is writable for `k` doubles.
#[no_mangle]
pub unsafe extern "C" fn lc_nd_knn(
    index: *const lc_nd_index,
    query: *const f64,
    k: usize,
    out_idx: *mut c_int,
    out_d2: *mut f64,
) -> c_int {
    if query.is_null() || out_idx.is_null() {
        return fail_msg("null pointer");
    }
    let h = match unsafe { nd_ref(index) } {
        Ok(h) => h,
        Err(code) => return code,
    };
    let d = h.tree.dims();
    let q = unsafe { std::slice::from_raw_parts(query, d) };
    let hits = match with_threads(|| h.tree.knn(q, k)) {
        Ok(hits) => hits,
        Err(err) => return fail(err),
    };
    let out = unsafe { std::slice::from_raw_parts_mut(out_idx, k) };
    out.fill(-1);
    let mut d2_out = if out_d2.is_null() {
        None
    } else {
        let s = unsafe { std::slice::from_raw_parts_mut(out_d2, k) };
        s.fill(f64::NAN);
        Some(s)
    };
    for (slot, hit) in hits.iter().take(k).enumerate() {
        if hit.index > c_int::MAX as usize {
            return fail(Error::Overflow);
        }
        out[slot] = hit.index as c_int;
        if let Some(buf) = d2_out.as_deref_mut() {
            buf[slot] = hit.dist2;
        }
    }
    clear_error();
    0
}

/// Radius neighbours for one query point.
///
/// Null `out_idx` (and `out_d2`) queries `*out_count`.
///
/// # Safety
///
/// `query` is readable for `lc_nd_dims(index)` doubles. `out_count` is writable.
/// On a fill, `out_idx` is writable for `cap` ints and `out_d2`, if non-null,
/// for `cap` doubles.
#[no_mangle]
pub unsafe extern "C" fn lc_nd_within(
    index: *const lc_nd_index,
    query: *const f64,
    radius: f64,
    out_idx: *mut c_int,
    out_d2: *mut f64,
    cap: usize,
    out_count: *mut usize,
) -> c_int {
    if query.is_null() || out_count.is_null() {
        return fail_msg("null pointer");
    }
    let h = match unsafe { nd_ref(index) } {
        Ok(h) => h,
        Err(code) => return code,
    };
    let d = h.tree.dims();
    let q = unsafe { std::slice::from_raw_parts(query, d) };
    let hits = match with_threads(|| h.tree.within(q, radius)) {
        Ok(hits) => hits,
        Err(err) => return fail(err),
    };
    unsafe {
        *out_count = hits.len();
    }
    if out_idx.is_null() {
        clear_error();
        return 0;
    }
    if cap < hits.len() {
        return fail_msg("radius buffer is shorter than the neighbour count");
    }
    let out = unsafe { std::slice::from_raw_parts_mut(out_idx, cap) };
    for (slot, hit) in hits.iter().enumerate() {
        if hit.index > c_int::MAX as usize {
            return fail(Error::Overflow);
        }
        out[slot] = hit.index as c_int;
    }
    if !out_d2.is_null() {
        let d2 = unsafe { std::slice::from_raw_parts_mut(out_d2, cap) };
        for (slot, hit) in hits.iter().enumerate() {
            d2[slot] = hit.dist2;
        }
    }
    clear_error();
    0
}

/// All index pairs `i < j` within radius.
///
/// Null `out_i`, `out_j`, and `out_d2` queries `*out_count`.
///
/// # Safety
///
/// `out_count` is writable. On a fill, `out_i`/`out_j` are writable for `cap`
/// ints and `out_d2`, if non-null, for `cap` doubles.
#[no_mangle]
pub unsafe extern "C" fn lc_nd_pairs_within(
    index: *const lc_nd_index,
    radius: f64,
    out_i: *mut c_int,
    out_j: *mut c_int,
    out_d2: *mut f64,
    cap: usize,
    out_count: *mut usize,
) -> c_int {
    if out_count.is_null() {
        return fail_msg("null pointer");
    }
    let h = match unsafe { nd_ref(index) } {
        Ok(h) => h,
        Err(code) => return code,
    };
    let pairs = match with_threads(|| h.tree.pairs_within(radius)) {
        Ok(rows) => rows,
        Err(err) => return fail(err),
    };
    unsafe {
        *out_count = pairs.len();
    }
    if out_i.is_null() && out_j.is_null() && out_d2.is_null() {
        clear_error();
        return 0;
    }
    if out_i.is_null() || out_j.is_null() {
        return fail_msg("null pointer");
    }
    if cap < pairs.len() {
        return fail_msg("pair buffer is shorter than the pair count");
    }
    let i_out = unsafe { std::slice::from_raw_parts_mut(out_i, cap) };
    let j_out = unsafe { std::slice::from_raw_parts_mut(out_j, cap) };
    let mut d2_out = if out_d2.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts_mut(out_d2, cap) })
    };
    for (slot, row) in pairs.iter().enumerate() {
        if row.i > c_int::MAX as usize || row.j > c_int::MAX as usize {
            return fail(Error::Overflow);
        }
        i_out[slot] = row.i as c_int;
        j_out[slot] = row.j as c_int;
        if let Some(buf) = d2_out.as_deref_mut() {
            buf[slot] = row.dist2;
        }
    }
    clear_error();
    0
}

/// Deterministic DBSCAN over the indexed points.
///
/// `min_pts` includes the point itself.
///
/// # Safety
///
/// `out_labels` and `out_core` are writable for `lc_nd_len(index)` entries.
/// `out_clusters`, if non-null, is writable for one count.
#[no_mangle]
pub unsafe extern "C" fn lc_nd_dbscan(
    index: *const lc_nd_index,
    eps: f64,
    min_pts: usize,
    out_labels: *mut c_int,
    out_core: *mut c_int,
    out_clusters: *mut usize,
) -> c_int {
    if out_labels.is_null() || out_core.is_null() {
        return fail_msg("null pointer");
    }
    let h = match unsafe { nd_ref(index) } {
        Ok(h) => h,
        Err(code) => return code,
    };
    let res = match with_threads(|| h.tree.dbscan(eps, min_pts)) {
        Ok(res) => res,
        Err(err) => return fail(err),
    };
    let labels = unsafe { std::slice::from_raw_parts_mut(out_labels, res.labels.len()) };
    let core = unsafe { std::slice::from_raw_parts_mut(out_core, res.core.len()) };
    for (i, &v) in res.labels.iter().enumerate() {
        labels[i] = v;
    }
    for (i, &v) in res.core.iter().enumerate() {
        core[i] = if v { 1 } else { 0 };
    }
    if !out_clusters.is_null() {
        unsafe {
            *out_clusters = res.clusters;
        }
    }
    clear_error();
    0
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
    let cont = take_continuation();
    let err = with_team(|| {
        let found = if n_frames == 1 {
            crate::knearest_into_d2(pts, &sim, k, mask_vec.as_deref(), hint, nn, d2)
        } else {
            crate::knearest_into_many(pts, n, n_frames, &sim, k, mask_vec.as_deref(), hint, nn, d2)
        };
        if found.is_ok() {
            run_continuation(cont);
        }
        found
    });
    if let Err(e) = err {
        set_error(&e.to_string());
        return 1;
    }
    clear_error();
    0
}

/// The k-nearest graph of `n` points as CSR.
///
/// Inputs are those of [`lc_knearest`]. Row `i` of the nominations is
/// its `k` nearest, nearest first. With `mutual` nonzero, row `i` holds
/// every `j` that row `i` lists and that lists `i`: the lower indices
/// ascending, then the higher ones nearest first. Otherwise each edge
/// comes from one nomination slot `s = i * k + t`: the lower index's
/// when both rows list each other, the only one that names it when one
/// does. A row holds its edges in slot order: edges from lower rows
/// ascending, then its own nearest first, then edges from higher rows
/// ascending.
///
/// `out_offsets` is caller-owned, `n + 1` ints. `alloc(ctx, total)` is
/// called once, from any thread of the search, and returns `total`
/// writable ints for the columns; null fails the call. The search, the
/// degrees, and the fill run in one parallel region.
///
/// # Safety
///
/// Same pointer contract as [`lc_knearest`]. `out_offsets` is writable
/// for `n + 1` ints. `alloc` returns null or `total` writable ints and
/// does not unwind.
#[no_mangle]
pub unsafe extern "C" fn lc_knearest_csr(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    k: usize,
    mask: *const c_int,
    cell_hint: f64,
    mutual: c_int,
    out_offsets: *mut c_int,
    alloc: Option<extern "C" fn(*mut std::ffi::c_void, usize) -> *mut c_int>,
    ctx: *mut std::ffi::c_void,
) -> c_int {
    if n == 0 {
        return fail(Error::Empty);
    }
    if k == 0 {
        return fail(Error::ZeroK);
    }
    let Some(alloc) = alloc else {
        return fail_msg("null pointer");
    };
    if xyz.is_null() || simbox.is_null() || out_offsets.is_null() {
        return fail_msg("null pointer");
    }
    let Some(need) = n.checked_mul(k) else {
        return fail(Error::Overflow);
    };
    if n > (isize::MAX as usize) / 3 || need > (isize::MAX as usize) / std::mem::size_of::<c_int>()
    {
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
        Err(e) => return fail_msg(&e.to_string()),
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
    let offsets = unsafe { std::slice::from_raw_parts_mut(out_offsets, n + 1) };
    let ctx_bits = ctx as usize;
    let get = move |total: usize| alloc(ctx_bits as *mut std::ffi::c_void, total);
    let found = with_team(|| {
        crate::knearest::knearest_graph(
            pts,
            &sim,
            k,
            mask_vec.as_deref(),
            hint,
            mutual != 0,
            offsets,
            &get,
        )
    });
    if let Err(e) = found {
        return fail(e);
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
    // three times that in `out_shift`. The fill runs on the same pool as
    // the search, as in `lc_pairs_within_rows`.
    let cols = ColumnOut(out_i, out_j, out_shift, out_d2);
    with_threads(move || {
        crate::pairs::in_pool(found.rows(), move || {
            let (i, j, s, d) = cols.get();
            unsafe { found.write_columns(i, j, s, d) }
        })
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
    // Safety: the caller gave `cap >= rows` rows. The fill runs on the
    // same pool as the search. `in_pool` would otherwise open the global
    // Rayon pool beside it.
    let rows_out = RowOut(out);
    with_threads(move || {
        crate::pairs::in_pool(found.rows(), move || {
            let out = rows_out.get();
            unsafe { found.write_rows(out, |i, j, shift, dist2| lc_pair { i, j, shift, dist2 }) }
        })
    });
    clear_error();
    0
}

/// Major version of the C ABI. A library serves a host compiled against
/// the same major.
pub const LC_ABI_MAJOR: u16 = 1;
/// Minor version of the C ABI. It grows when an entry point or a
/// capability is added. A library serves a host compiled against the
/// same or a lower minor.
pub const LC_ABI_MINOR: u16 = 0;
/// Revision of the struct layouts the row-grid calls pass (`lc_row`,
/// `lc_caps`, `lc_abi`). A library serves a host compiled against the
/// same revision.
pub const LC_ABI_LAYOUT_REVISION: u16 = 1;

/// The C ABI a header or a library was built for.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct lc_abi {
    /// `LC_ABI_MAJOR`.
    pub major: u16,
    /// `LC_ABI_MINOR`.
    pub minor: u16,
    /// `LC_ABI_LAYOUT_REVISION`.
    pub layout_revision: u16,
}

/// The ABI this library was built for. A host compares it with the
/// `LC_ABI_*` constants of the header it compiled against, or passes
/// those to [`lc_abi_compatible`]. Does not touch `lc_last_error`.
#[no_mangle]
pub extern "C" fn lc_abi_stamp() -> lc_abi {
    lc_abi {
        major: LC_ABI_MAJOR,
        minor: LC_ABI_MINOR,
        layout_revision: LC_ABI_LAYOUT_REVISION,
    }
}

/// 1 when this library serves a host compiled against `host`: the same
/// major and layout revision, and a host minor no higher than the
/// library's. 0 otherwise, and for a null `host`. Does not touch
/// `lc_last_error`.
///
/// # Safety
///
/// `host` is null or points at one `lc_abi`.
#[no_mangle]
pub unsafe extern "C" fn lc_abi_compatible(host: *const lc_abi) -> c_int {
    if host.is_null() {
        return 0;
    }
    let host = unsafe { *host };
    // The minor is 0 today, which makes the last comparison an equality.
    #[allow(clippy::absurd_extreme_comparisons)]
    let serves = host.major == LC_ABI_MAJOR
        && host.layout_revision == LC_ABI_LAYOUT_REVISION
        && host.minor <= LC_ABI_MINOR;
    c_int::from(serves)
}

/// What this library can do, for a host to read before it calls.
///
/// The host sets `size` to `sizeof(lc_caps)` as it compiled it. The
/// library writes the members that fit in that size and sets `size` to
/// the bytes it wrote, so a member a host knows and the library does not
/// stays as the host initialised it (zero it first). Each flag is 1 when
/// supported and 0 when not.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct lc_caps {
    /// Bytes of this struct: in, the host's; out, those written.
    pub size: u32,
    /// The row grid: `lc_row_grid_build`, `lc_row_grid_update`,
    /// `lc_row_grid_len`, `lc_row_grid_free`, `lc_row_grid_rows`, and
    /// `lc_row_grid_point`.
    pub row_grid: c_int,
    /// `lc_row.vector` is filled.
    pub row_vector: c_int,
    /// A nonzero return of the visitor stops the walk.
    pub row_stop: c_int,
    /// `lc_row_grid_build_request` decodes a `NeighborRequest`.
    pub row_request: c_int,
    /// Open and mixed-periodic cells are accepted.
    pub open_axes: c_int,
    /// Binning may run on the pool `lc_set_threads` sets.
    pub threads: c_int,
    /// `lc_row_grid_fill` reads rows into a buffer.
    pub row_fill: c_int,
    /// `closed` keeps `dist2 == cutoff²` on a fully periodic cell too.
    pub closed_periodic: c_int,
    /// `lc_row_grid_bound` and `lc_row_grid_bins` exist.
    pub row_bound: c_int,
}

/// Fill `out` with the capabilities of this library. Returns 0, or
/// nonzero with `lc_last_error` set when `out` is null or `out->size`
/// is smaller than the `size` member itself.
///
/// # Safety
///
/// `out` is null or writable for `out->size` bytes, which the host set.
#[no_mangle]
pub unsafe extern "C" fn lc_capabilities(out: *mut lc_caps) -> c_int {
    if out.is_null() {
        return fail_msg("null pointer");
    }
    let host = unsafe { std::ptr::addr_of!((*out).size).read_unaligned() } as usize;
    if host < std::mem::size_of::<u32>() {
        return fail_msg("capability struct is smaller than its size member");
    }
    let mut caps = lc_caps {
        size: 0,
        row_grid: 1,
        row_vector: 1,
        row_stop: 1,
        row_request: c_int::from(cfg!(feature = "request")),
        open_axes: 1,
        threads: c_int::from(cfg!(feature = "parallel")),
        row_fill: 1,
        closed_periodic: 1,
        row_bound: 1,
    };
    let wrote = host.min(std::mem::size_of::<lc_caps>());
    caps.size = wrote as u32;
    // Safety: `out` is writable for the `host >= wrote` bytes it declared.
    unsafe {
        std::ptr::copy_nonoverlapping(
            std::ptr::addr_of!(caps).cast::<u8>(),
            out.cast::<u8>(),
            wrote,
        );
    }
    clear_error();
    0
}

/// Runs an entry point's body. A panic does not cross into C: it becomes
/// a nonzero return with `lc_last_error` set.
fn no_unwind(body: impl FnOnce() -> c_int) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(_) => fail_msg("internal error: a panic was caught at the C boundary"),
    }
}

struct RowHandle {
    grid: crate::RowGrid,
}

/// Opaque handle of the bins of one cutoff search.
///
/// Callers only ever hold a pointer returned by `lc_row_grid_build`. The
/// byte makes this a complete object type in C99 and C++17.
#[repr(C)]
pub struct lc_row_grid {
    _private: u8,
}

/// One entry of a row: an image of point `j` within the cutoff of point
/// `i`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct lc_row {
    /// Source index: the point the row belongs to.
    pub i: c_int,
    /// Target index.
    pub j: c_int,
    /// Integer cell shift `(na, nb, nc)` applied to the target.
    pub shift: [c_int; 3],
    /// `r_j - r_i` with the shift applied. Its squared length is `dist2`.
    pub vector: [f64; 3],
    /// Squared distance after the shift.
    pub dist2: f64,
}

/// Receives one row entry from [`lc_row_grid_rows`] or
/// [`lc_row_grid_point`]. `user` is the pointer given to that call.
/// `row` is valid until the callback returns. Return 0 to go on. Any
/// other value stops the walk, and the call returns that value.
#[allow(non_camel_case_types)]
pub type lc_row_visit =
    Option<unsafe extern "C" fn(user: *mut c_void, row: *const lc_row) -> c_int>;

unsafe fn row_ref<'a>(ptr: *const lc_row_grid) -> Result<&'a RowHandle, c_int> {
    if ptr.is_null() {
        return Err(fail_msg("null pointer"));
    }
    Ok(unsafe { &*(ptr as *const RowHandle) })
}

/// Checks the points both builds share and views them as triples.
unsafe fn row_points<'a>(xyz: *const f64, n: usize) -> Result<&'a [[f64; 3]], c_int> {
    if n == 0 {
        return Err(fail(Error::Empty));
    }
    if xyz.is_null() {
        return Err(fail_msg("null pointer"));
    }
    if n > c_int::MAX as usize || n > (isize::MAX as usize) / 3 {
        return Err(fail(Error::Overflow));
    }
    Ok(unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n) })
}

/// The lattice of `simbox` for `periodic`, or a unit cube when every
/// axis is open.
unsafe fn row_cell(simbox: *const lc_cell, periodic: [bool; 3]) -> Result<Cell, c_int> {
    if periodic == [false, false, false] {
        return Cell::ortho(1.0, 1.0, 1.0).map_err(|err| fail(err.into()));
    }
    if simbox.is_null() {
        return Err(fail_msg("null pointer"));
    }
    let raw = unsafe { *simbox };
    crate::cell_for_pbc(
        [
            [raw.ax, raw.ay, raw.az],
            [raw.bx, raw.by, raw.bz],
            [raw.cx, raw.cy, raw.cz],
        ],
        [raw.ox, raw.oy, raw.oz],
        periodic,
    )
    .map_err(fail)
}

/// Checks the arguments [`lc_row_grid_build`] and [`lc_row_grid_update`]
/// share, then runs `bin` with the points, the cell, and the query.
#[allow(clippy::too_many_arguments)]
unsafe fn row_grid_bin(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    pbc: *const c_int,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    closed: c_int,
    exclude_self_images: c_int,
    bin: impl FnOnce(&[[f64; 3]], &Cell, &crate::PbcQuery<'_>) -> Result<(), Error> + Send,
) -> c_int {
    let periodic = if pbc.is_null() {
        [true, true, true]
    } else {
        let flags = unsafe { [*pbc, *pbc.add(1), *pbc.add(2)] };
        [flags[0] != 0, flags[1] != 0, flags[2] != 0]
    };
    let pts = match unsafe { row_points(xyz, n) } {
        Ok(pts) => pts,
        Err(code) => return code,
    };
    let cell = match unsafe { row_cell(simbox, periodic) } {
        Ok(cell) => cell,
        Err(code) => return code,
    };
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
    let mut query = crate::PbcQuery::new(periodic, cutoff)
        .cell_hint(hint)
        .closed(closed != 0)
        .exclude_self_images(exclude_self_images != 0);
    query.mask = mask_vec.as_deref();
    match with_threads(|| bin(pts, &cell, &query)) {
        Ok(()) => {
            clear_error();
            0
        }
        Err(err) => fail(err),
    }
}

/// Bin `n` points for the rows of one cutoff search and keep the bins.
///
/// The arguments are those of [`lc_pairs_within_pbc`], less `half` and
/// the row buffer: `pbc` is null for a fully periodic cell or three
/// ints, nonzero for a periodic axis; `mask` is null or `n` ints;
/// `cell_hint <= 0` selects the default bin edge; `closed` nonzero keeps
/// `dist2 <= cutoff²`, on a periodic cell as on an open one;
/// `exclude_self_images` nonzero drops `i == j`. The grid copies what it needs from `xyz` and
/// is released by [`lc_row_grid_free`]. It holds the bins and no pair.
///
/// Returns 0 and writes the handle to `*out`. On failure `*out` is null,
/// the return is nonzero, and `lc_last_error` holds the message.
/// `n == 0` fails with "no points".
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `simbox`, if any axis is
/// periodic, points at one `lc_cell`. `pbc`, if non-null, is three
/// ints. `mask`, if non-null, is `n` ints. `out` is writable for one
/// pointer.
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_build(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    pbc: *const c_int,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    closed: c_int,
    exclude_self_images: c_int,
    out: *mut *mut lc_row_grid,
) -> c_int {
    if out.is_null() {
        return fail_msg("null pointer");
    }
    unsafe {
        *out = ptr::null_mut();
    }
    no_unwind(|| {
        let mut built = None;
        let slot = &mut built;
        let code = unsafe {
            row_grid_bin(
                xyz,
                n,
                simbox,
                pbc,
                cutoff,
                mask,
                cell_hint,
                closed,
                exclude_self_images,
                move |pts, cell, query| {
                    *slot = Some(crate::RowGrid::build(pts, cell, query)?);
                    Ok(())
                },
            )
        };
        if let Some(grid) = built {
            unsafe {
                *out = Box::into_raw(Box::new(RowHandle { grid })) as *mut lc_row_grid;
            }
        }
        code
    })
}

/// [`lc_row_grid_build`] from a Cap'n Proto `NeighborRequest`.
///
/// `request` holds one message in the standard stream framing (segment
/// table, then segments, not packed) whose root is the `NeighborRequest`
/// of the shared potential schema. The request is read as follows.
///
/// - `cutoff` and `skin`: with `cut == 0` the rows have `dist2 <
///   cutoff²` and `skin` must be 0. With `cut == 1` they have `dist2 <
///   (cutoff + skin)²`; `cut == 1` with a zero `skin` asks for `dist2 <=
///   cutoff²`, the closed cutoff of [`lc_row_grid_build`].
/// - `periodic`: empty for a fully periodic cell, or three flags.
/// - `systems`: exactly one, with `count == n` and `box` the nine
///   lattice entries, rows a, b, c. The box is not read when every axis
///   is open and may then be empty.
/// - `format` must be 0 (pairs) and `mode` must not be 0.
/// - `shifts`, `vectors`, `distances`, `cursor`, and `limit` describe a
///   stored reply and are not read: a visit always passes all three and
///   takes its range from the call.
///
/// - `shell`: written to `*out_half` when `out_half` is not null, 1
///   for a half list and 0 for a full one, for the caller to pass to
///   each visit. The grid itself serves both.
/// - `binEdge`: the bin edge; 0 or less selects the default.
/// - `origin`: empty for a cell at the origin, or three coordinates.
/// - `excludeSelfImages`: drops `i == j`.
///
/// `mask` is null or `n` ints, nonzero to keep the point, as in
/// [`lc_row_grid_build`]. It is a buffer beside the positions and not a
/// field of the request.
///
/// Returns nonzero with `lc_last_error` set for a message that does not
/// decode, a request this entry does not serve, and every failure of
/// [`lc_row_grid_build`]. A library built without request support
/// returns nonzero with "this library was built without NeighborRequest
/// support"; `lc_capabilities` reports that as `row_request == 0`.
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `mask`, if non-null, is `n`
/// ints. `request` is readable for `request_len` bytes. `out_half`, if
/// non-null, is writable for one int. `out` is writable for one pointer.
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_build_request(
    xyz: *const f64,
    n: usize,
    mask: *const c_int,
    request: *const u8,
    request_len: usize,
    out_half: *mut c_int,
    out: *mut *mut lc_row_grid,
) -> c_int {
    if out.is_null() {
        return fail_msg("null pointer");
    }
    unsafe {
        *out = ptr::null_mut();
    }
    no_unwind(|| unsafe {
        row_grid_from_request(xyz, n, mask, request, request_len, out_half, out)
    })
}

#[cfg(feature = "request")]
unsafe fn row_grid_from_request(
    xyz: *const f64,
    n: usize,
    mask: *const c_int,
    request: *const u8,
    request_len: usize,
    out_half: *mut c_int,
    out: *mut *mut lc_row_grid,
) -> c_int {
    let pts = match unsafe { row_points(xyz, n) } {
        Ok(pts) => pts,
        Err(code) => return code,
    };
    if request.is_null() {
        return fail_msg("null pointer");
    }
    let bytes = unsafe { std::slice::from_raw_parts(request, request_len) };
    let decoded = match crate::request::decode(bytes, n) {
        Ok(decoded) => decoded,
        Err(msg) => return fail_msg(&msg),
    };
    let cell = if decoded.periodic == [false, false, false] {
        Cell::ortho(1.0, 1.0, 1.0).map_err(Error::from)
    } else {
        let h = decoded.lattice;
        crate::cell_for_pbc(
            [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]],
            decoded.origin,
            decoded.periodic,
        )
    };
    let cell = match cell {
        Ok(cell) => cell,
        Err(err) => return fail(err),
    };
    let mask_vec: Option<Vec<bool>> = if mask.is_null() {
        None
    } else {
        let raw = unsafe { std::slice::from_raw_parts(mask, n) };
        Some(raw.iter().map(|&v| v != 0).collect())
    };
    let mut query = crate::PbcQuery::new(decoded.periodic, decoded.cutoff)
        .cell_hint(decoded.cell_hint)
        .closed(decoded.closed)
        .exclude_self_images(decoded.exclude_self);
    query.mask = mask_vec.as_deref();
    match with_threads(|| crate::RowGrid::build(pts, &cell, &query)) {
        Ok(grid) => {
            unsafe {
                *out = Box::into_raw(Box::new(RowHandle { grid })) as *mut lc_row_grid;
                if !out_half.is_null() {
                    *out_half = c_int::from(decoded.half);
                }
            }
            clear_error();
            0
        }
        Err(err) => fail(err),
    }
}

#[cfg(not(feature = "request"))]
unsafe fn row_grid_from_request(
    _xyz: *const f64,
    _n: usize,
    _mask: *const c_int,
    _request: *const u8,
    _request_len: usize,
    _out_half: *mut c_int,
    _out: *mut *mut lc_row_grid,
) -> c_int {
    fail_msg("this library was built without NeighborRequest support")
}

/// Bin new points, a new cell, or a new query into a grid from
/// [`lc_row_grid_build`], reusing its buffers.
///
/// The arguments after `grid` are those of [`lc_row_grid_build`]. On
/// failure the grid holds no point, every row is out of range, and the
/// handle is still valid for another update or for
/// [`lc_row_grid_free`].
///
/// # Safety
///
/// `grid` is a pointer returned by [`lc_row_grid_build`], not freed, and
/// no other thread is reading it. The other pointers are as in
/// [`lc_row_grid_build`].
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_update(
    grid: *mut lc_row_grid,
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    pbc: *const c_int,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    closed: c_int,
    exclude_self_images: c_int,
) -> c_int {
    if grid.is_null() {
        return fail_msg("null pointer");
    }
    no_unwind(|| {
        let handle = unsafe { &mut *(grid as *mut RowHandle) };
        // A call that fails on its arguments never reaches the update, and
        // still leaves the grid without rows.
        handle.grid.clear();
        let target = &mut handle.grid;
        unsafe {
            row_grid_bin(
                xyz,
                n,
                simbox,
                pbc,
                cutoff,
                mask,
                cell_hint,
                closed,
                exclude_self_images,
                move |pts, cell, query| target.update(pts, cell, query),
            )
        }
    })
}

/// Number of points a grid holds, masked ones included. Rows are
/// numbered from zero up to this. Null gives zero.
///
/// # Safety
///
/// `grid` is null or a valid pointer returned by [`lc_row_grid_build`].
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_len(grid: *const lc_row_grid) -> usize {
    if grid.is_null() {
        return 0;
    }
    unsafe { &*(grid as *const RowHandle) }.grid.points()
}

/// Number of bins a grid holds. Null gives zero.
///
/// # Safety
///
/// `grid` is null or a valid pointer returned by [`lc_row_grid_build`].
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_bins(grid: *const lc_row_grid) -> usize {
    if grid.is_null() {
        return 0;
    }
    unsafe { &*(grid as *const RowHandle) }.grid.bins()
}

/// An upper bound on the entries of every full row of a grid together,
/// from the bin occupancy alone: for each point, the points in the bins
/// its row reads, less itself. No distance is computed, so a host that
/// must decide whether a stored list fits in memory reads this and does
/// not count the pairs. A half list holds at most half of it, rounded
/// up. The sum saturates at `UINT64_MAX`.
///
/// Returns 0 and writes `*out`, or 1 with `lc_last_error` set.
///
/// # Safety
///
/// `grid` is a valid pointer returned by [`lc_row_grid_build`], and no
/// thread updates or frees it during the call. `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_bound(grid: *const lc_row_grid, out: *mut u64) -> c_int {
    let handle = match unsafe { row_ref(grid) } {
        Ok(handle) => handle,
        Err(code) => return code,
    };
    if out.is_null() {
        return fail_msg("null pointer");
    }
    no_unwind(|| match handle.grid.candidate_bound() {
        Ok(bound) => {
            unsafe { *out = bound };
            clear_error();
            0
        }
        Err(err) => fail(err),
    })
}

/// Free a grid built by [`lc_row_grid_build`]. Null is accepted.
///
/// # Safety
///
/// `grid` is null or a pointer returned by [`lc_row_grid_build`] and not
/// already freed.
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_free(grid: *mut lc_row_grid) {
    if grid.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(|| unsafe {
        drop(Box::from_raw(grid as *mut RowHandle));
    });
}

/// The entry a visit passes to C, and what its callback said.
fn row_call(
    visit: unsafe extern "C" fn(*mut c_void, *const lc_row) -> c_int,
    user: *mut c_void,
    row: &crate::Row,
) -> std::ops::ControlFlow<c_int> {
    let entry = lc_row {
        i: row.i as c_int,
        j: row.j as c_int,
        shift: row.shift,
        vector: row.vector,
        dist2: row.dist2,
    };
    match unsafe { visit(user, &entry) } {
        0 => std::ops::ControlFlow::Continue(()),
        stop => std::ops::ControlFlow::Break(stop),
    }
}

/// The return of a visit: 0 after a whole walk, the callback's value
/// after a stop, both with the last error cleared, or a failure.
fn row_done(walked: Result<std::ops::ControlFlow<c_int>, Error>) -> c_int {
    match walked {
        Ok(std::ops::ControlFlow::Continue(())) => {
            clear_error();
            0
        }
        Ok(std::ops::ControlFlow::Break(stop)) => {
            clear_error();
            stop
        }
        Err(err) => fail(err),
    }
}

/// Visit the rows of points `begin .. end` of a grid, in ascending point
/// order, without storing them.
///
/// `visit(user, row)` is called on the calling thread once for every
/// image of `row->j` within the cutoff of `row->i`, with the shift and
/// squared distance [`lc_pairs_within_pbc`] writes for that row and the
/// vector `r_j - r_i` with that shift applied. `half` nonzero keeps the
/// entries the half list has under `i`. Over every point the entries are
/// the rows of [`lc_pairs_within_pbc`] for the grid's arguments. Entries
/// of one row come in bin order (the bins around the point, then the
/// points in each bin), not sorted by `j`; that order does not depend on
/// how the points are split into ranges.
///
/// The call reads the grid and writes nothing in it, so several threads
/// may visit one grid at once, on disjoint or overlapping ranges.
///
/// Returns 0 after every entry was visited. When `visit` returns a
/// nonzero value the walk stops and the call returns that value with
/// `lc_last_error` cleared. A failure returns 1 with `lc_last_error`
/// set, before any visit when `visit` is null or `end` is past
/// [`lc_row_grid_len`] ("point index is out of range"). A callback that
/// returns 1 to stop is told apart from a failure by a null
/// `lc_last_error`. `begin >= end` visits nothing and returns 0.
///
/// # Safety
///
/// `grid` is a valid pointer returned by [`lc_row_grid_build`], and no
/// thread updates or frees it during the call. `visit` may be called
/// with `user` any number of times and must not unwind.
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_rows(
    grid: *const lc_row_grid,
    begin: usize,
    end: usize,
    half: c_int,
    visit: lc_row_visit,
    user: *mut c_void,
) -> c_int {
    let handle = match unsafe { row_ref(grid) } {
        Ok(handle) => handle,
        Err(code) => return code,
    };
    let Some(visit) = visit else {
        return fail_msg("null pointer");
    };
    no_unwind(|| {
        row_done(
            handle
                .grid
                .try_for_rows(begin..end, half != 0, |row| row_call(visit, user, row)),
        )
    })
}

/// Visit the row of point `i` of a grid: [`lc_row_grid_rows`] on
/// `i .. i + 1`, for a caller that reads one point at a time.
///
/// # Safety
///
/// As [`lc_row_grid_rows`].
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_point(
    grid: *const lc_row_grid,
    i: usize,
    half: c_int,
    visit: lc_row_visit,
    user: *mut c_void,
) -> c_int {
    let handle = match unsafe { row_ref(grid) } {
        Ok(handle) => handle,
        Err(code) => return code,
    };
    let Some(visit) = visit else {
        return fail_msg("null pointer");
    };
    no_unwind(|| {
        row_done(
            handle
                .grid
                .try_for_point(i, half != 0, |row| row_call(visit, user, row)),
        )
    })
}

/// Where a buffered read of rows stands: the next entry is entry
/// `entry` of the row of point `point`. Start a read with `{begin, 0}`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct lc_row_cursor {
    /// Point whose row is read next.
    pub point: usize,
    /// Entries of that row already read.
    pub entry: usize,
}

/// Copy row entries into `out`, from `*cursor` up to the row of point
/// `end`, and move `*cursor` past them.
///
/// The entries are those [`lc_row_grid_rows`] visits over
/// `cursor->point .. end`, less the first `cursor->entry` of the first
/// row, in the same order. `*count` receives how many were written, at
/// most `cap`. The read is done when `cursor->point == end`. A host
/// calls this in a loop and handles a buffer of rows at a time, with one
/// call per buffer and none per entry.
///
/// Returns 0, or 1 with `lc_last_error` set: a null pointer, or `end`
/// past [`lc_row_grid_len`] or `cursor->point` past `end` ("point index
/// is out of range"). `cap == 0` writes nothing and returns 0.
///
/// # Safety
///
/// `grid` is a valid pointer returned by [`lc_row_grid_build`], and no
/// thread updates or frees it during the call. `cursor` and `count` are
/// writable. `out` is writable for `cap` rows.
#[no_mangle]
pub unsafe extern "C" fn lc_row_grid_fill(
    grid: *const lc_row_grid,
    cursor: *mut lc_row_cursor,
    end: usize,
    half: c_int,
    out: *mut lc_row,
    cap: usize,
    count: *mut usize,
) -> c_int {
    let handle = match unsafe { row_ref(grid) } {
        Ok(handle) => handle,
        Err(code) => return code,
    };
    if cursor.is_null() || count.is_null() || (cap > 0 && out.is_null()) {
        return fail_msg("null pointer");
    }
    no_unwind(|| {
        unsafe { *count = 0 };
        let mut at = unsafe {
            crate::RowCursor {
                point: (*cursor).point,
                entry: (*cursor).entry,
            }
        };
        // The rows are read in blocks on the stack and copied out, so
        // the caller's buffer needs no initial value.
        let mut block = [crate::Row {
            i: 0,
            j: 0,
            shift: [0; 3],
            vector: [0.0; 3],
            dist2: 0.0,
        }; 64];
        let mut written = 0usize;
        while written < cap && at.point < end {
            let want = block.len().min(cap - written);
            let got = match handle
                .grid
                .fill_rows(&mut at, end, half != 0, &mut block[..want])
            {
                Ok(got) => got,
                Err(err) => return fail(err),
            };
            for (k, row) in block[..got].iter().enumerate() {
                let entry = lc_row {
                    i: row.i as c_int,
                    j: row.j as c_int,
                    shift: row.shift,
                    vector: row.vector,
                    dist2: row.dist2,
                };
                unsafe { out.add(written + k).write(entry) };
            }
            written += got;
            unsafe {
                *cursor = lc_row_cursor {
                    point: at.point,
                    entry: at.entry,
                };
                *count = written;
            }
        }
        if cap == 0 || at.point >= end {
            // Nothing was read: the range is still checked.
            if let Err(err) = handle.grid.fill_rows(&mut at, end, half != 0, &mut []) {
                return fail(err);
            }
        }
        clear_error();
        0
    })
}

/// The row-grid operations of this library, as function pointers.
///
/// A host that loads the library at run time resolves `lc_abi_stamp`,
/// `lc_abi_compatible`, and [`lc_row_ops_get`] by name, and takes every
/// row-grid entry from this table. The host sets `size` to
/// `sizeof(lc_row_ops)` as it compiled it and zeroes the rest; the
/// library writes the members that fit and sets `size` to the bytes it
/// wrote. A null member is an operation this library does not have: one
/// the host's header knows and an older library lacks, or
/// `build_request` in a library built without request support. Each
/// member is the entry point of the same name with the `lc_row_grid_`
/// prefix, and `last_error` is `lc_last_error`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct lc_row_ops {
    /// Bytes of this struct: in, the host's; out, those written.
    pub size: u32,
    /// [`lc_row_grid_build`].
    pub build: Option<
        unsafe extern "C" fn(
            *const f64,
            usize,
            *const lc_cell,
            *const c_int,
            f64,
            *const c_int,
            f64,
            c_int,
            c_int,
            *mut *mut lc_row_grid,
        ) -> c_int,
    >,
    /// [`lc_row_grid_build_request`], or null without request support.
    pub build_request: Option<
        unsafe extern "C" fn(
            *const f64,
            usize,
            *const c_int,
            *const u8,
            usize,
            *mut c_int,
            *mut *mut lc_row_grid,
        ) -> c_int,
    >,
    /// [`lc_row_grid_update`].
    pub update: Option<
        unsafe extern "C" fn(
            *mut lc_row_grid,
            *const f64,
            usize,
            *const lc_cell,
            *const c_int,
            f64,
            *const c_int,
            f64,
            c_int,
            c_int,
        ) -> c_int,
    >,
    /// [`lc_row_grid_len`].
    pub len: Option<unsafe extern "C" fn(*const lc_row_grid) -> usize>,
    /// [`lc_row_grid_free`].
    pub free: Option<unsafe extern "C" fn(*mut lc_row_grid)>,
    /// [`lc_row_grid_rows`].
    pub rows: Option<
        unsafe extern "C" fn(
            *const lc_row_grid,
            usize,
            usize,
            c_int,
            lc_row_visit,
            *mut c_void,
        ) -> c_int,
    >,
    /// [`lc_row_grid_point`].
    pub point: Option<
        unsafe extern "C" fn(*const lc_row_grid, usize, c_int, lc_row_visit, *mut c_void) -> c_int,
    >,
    /// [`lc_row_grid_fill`].
    pub fill: Option<
        unsafe extern "C" fn(
            *const lc_row_grid,
            *mut lc_row_cursor,
            usize,
            c_int,
            *mut lc_row,
            usize,
            *mut usize,
        ) -> c_int,
    >,
    /// [`lc_last_error`].
    pub last_error: Option<extern "C" fn() -> *const c_char>,
    /// [`lc_capabilities`].
    pub capabilities: Option<unsafe extern "C" fn(*mut lc_caps) -> c_int>,
    /// [`lc_row_grid_bins`].
    pub bins: Option<unsafe extern "C" fn(*const lc_row_grid) -> usize>,
    /// [`lc_row_grid_bound`].
    pub bound: Option<unsafe extern "C" fn(*const lc_row_grid, *mut u64) -> c_int>,
}

/// Fill `out` with the row-grid operations of this library; see
/// [`lc_row_ops`]. Returns 0, or nonzero with `lc_last_error` set when
/// `out` is null or `out->size` is smaller than the `size` member.
///
/// # Safety
///
/// `out` is null or writable for `out->size` bytes, which the host set.
#[no_mangle]
pub unsafe extern "C" fn lc_row_ops_get(out: *mut lc_row_ops) -> c_int {
    if out.is_null() {
        return fail_msg("null pointer");
    }
    let host = unsafe { std::ptr::addr_of!((*out).size).read_unaligned() } as usize;
    if host < std::mem::size_of::<u32>() {
        return fail_msg("operations struct is smaller than its size member");
    }
    let wrote = host.min(std::mem::size_of::<lc_row_ops>());
    let ops = lc_row_ops {
        size: wrote as u32,
        build: Some(lc_row_grid_build),
        build_request: if cfg!(feature = "request") {
            Some(lc_row_grid_build_request)
        } else {
            None
        },
        update: Some(lc_row_grid_update),
        len: Some(lc_row_grid_len),
        free: Some(lc_row_grid_free),
        rows: Some(lc_row_grid_rows),
        point: Some(lc_row_grid_point),
        fill: Some(lc_row_grid_fill),
        last_error: Some(lc_last_error),
        capabilities: Some(lc_capabilities),
        bins: Some(lc_row_grid_bins),
        bound: Some(lc_row_grid_bound),
    };
    // Safety: `out` is writable for the `host >= wrote` bytes it declared.
    unsafe {
        std::ptr::copy_nonoverlapping(
            std::ptr::addr_of!(ops).cast::<u8>(),
            out.cast::<u8>(),
            wrote,
        );
    }
    clear_error();
    0
}

/// How many partner stencils have been built in this process.
///
/// A reused stencil does not increment the count. The periodic list and
/// the open-cell list share the counter.
#[no_mangle]
pub extern "C" fn lc_stencil_builds() -> u64 {
    crate::stencil_builds()
}

/// Cartesian half pair list of an open cluster.
///
/// Every axis is open. Each row has `i < j`, shift `(0, 0, 0)`, and
/// `dist2` strictly below `cutoff` squared. Null `out` queries
/// `*out_count`. A short buffer sets `*out_count` and returns nonzero
/// without writing. The bins are the occupied Cartesian cells. Their
/// partner stencil depends on the cutoff and `cell_hint` alone, so it is
/// reused by frames that occupy other bins.
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `out_count` is writable.
/// On a fill, `out` is writable for `cap` rows.
#[no_mangle]
pub unsafe extern "C" fn lc_pairs_open(
    xyz: *const f64,
    n: usize,
    cutoff: f64,
    cell_hint: f64,
    out: *mut lc_pair,
    cap: usize,
    out_count: *mut usize,
) -> c_int {
    let periodic = [0, 0, 0];
    unsafe {
        lc_pairs_within_pbc(
            xyz,
            n,
            std::ptr::null(),
            periodic.as_ptr(),
            cutoff,
            std::ptr::null(),
            cell_hint,
            1,
            0,
            0,
            out,
            cap,
            out_count,
        )
    }
}

/// [`lc_pairs_within_rows`] with a periodicity mask.
///
/// `pbc` is three ints. Nonzero means that axis is periodic. A null
/// `pbc` is fully periodic. An open axis is replaced by a unit normal
/// of the periodic rows, however long the stored row is, and is binned
/// on the occupied Cartesian cells of that lattice. A fully periodic
/// cell whose face width is zero or non-finite returns nonzero;
/// `lc_last_error` is
/// "periodic cell is degenerate: a face width is zero or non-finite".
/// A positive finite width that needs more than 2^24 images, a partner
/// stencil of more than 2^31 slots, or a row buffer that does not
/// fit, returns nonzero with
/// "periodic image count exceeds 2^24". A mesh past the bin cap returns
/// "linked-cell mesh is too fine". `closed`
/// nonzero keeps `dist2 <= cutoff²`, on a periodic cell too.
/// `exclude_self_images` nonzero drops `i == j`.
///
/// Null `out` is a count query. Otherwise `out` holds `cap` rows.
/// A query or a short buffer keeps its rows on the calling thread for
/// the next call with the same inputs bit for bit.
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `simbox`, if any axis is
/// periodic, points at one `lc_cell`. `pbc`, if non-null, is three
/// ints. `mask`, if non-null, is `n` ints. `out_count` is writable.
/// On a fill, `out` is writable for `cap` rows.
#[no_mangle]
pub unsafe extern "C" fn lc_pairs_within_pbc(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    pbc: *const c_int,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    half: c_int,
    closed: c_int,
    exclude_self_images: c_int,
    out: *mut lc_pair,
    cap: usize,
    out_count: *mut usize,
) -> c_int {
    if out_count.is_null() {
        return fail_msg("null pointer");
    }
    let periodic = if pbc.is_null() {
        [true, true, true]
    } else {
        let flags = unsafe { [*pbc, *pbc.add(1), *pbc.add(2)] };
        [flags[0] != 0, flags[1] != 0, flags[2] != 0]
    };
    if n == 0 {
        unsafe { *out_count = 0 };
        clear_error();
        return 0;
    }
    if xyz.is_null() || (periodic != [false, false, false] && simbox.is_null()) {
        return fail_msg("null pointer");
    }
    if n > c_int::MAX as usize || n > (isize::MAX as usize) / 3 {
        return fail(Error::Overflow);
    }
    let hint = if cell_hint > 0.0 {
        Some(cell_hint)
    } else {
        None
    };
    let pts = unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n) };
    let mask_raw: Option<&[c_int]> = if mask.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(mask, n) })
    };
    let key = OpenKey {
        xyz: xyz_bits(pts),
        periodic,
        cutoff: cutoff.to_bits(),
        mask: mask_raw.map(|m| m.iter().map(|&v| v != 0).collect()),
        cell_hint: hint.map(f64::to_bits),
        half: half != 0,
        closed: closed != 0,
        exclude_self: exclude_self_images != 0,
        box_bits: if periodic == [false, false, false] || simbox.is_null() {
            [0; 12]
        } else {
            box_bits(unsafe { &*simbox })
        },
    };
    let rows = if let Some(parked) = take_open(&key) {
        parked
    } else {
        match open_rows(pts, simbox, periodic, cutoff, mask_raw, hint, &key) {
            Ok(rows) => rows,
            Err(code) => return code,
        }
    };
    unsafe { *out_count = rows.len() };
    if out.is_null() {
        park_open(key, rows);
        clear_error();
        return 0;
    }
    if cap < rows.len() {
        let len = rows.len();
        park_open(key, rows);
        let _ = len;
        return fail_msg("pair buffer is shorter than the pair count");
    }
    let dest = unsafe { std::slice::from_raw_parts_mut(out, rows.len()) };
    dest.copy_from_slice(&rows);
    clear_error();
    0
}

#[derive(PartialEq)]
struct OpenKey {
    xyz: Vec<u64>,
    periodic: [bool; 3],
    cutoff: u64,
    mask: Option<Vec<bool>>,
    cell_hint: Option<u64>,
    half: bool,
    closed: bool,
    exclude_self: bool,
    box_bits: [u64; 12],
}

struct ParkedOpen {
    key: OpenKey,
    rows: Vec<lc_pair>,
}

thread_local! {
    static PARKED_OPEN: RefCell<Option<ParkedOpen>> = const { RefCell::new(None) };
}

fn xyz_bits(xyz: &[[f64; 3]]) -> Vec<u64> {
    let mut bits = Vec::with_capacity(xyz.len() * 3);
    for point in xyz {
        bits.push(point[0].to_bits());
        bits.push(point[1].to_bits());
        bits.push(point[2].to_bits());
    }
    bits
}

fn take_open(key: &OpenKey) -> Option<Vec<lc_pair>> {
    let parked = PARKED_OPEN.with(|slot| slot.borrow_mut().take())?;
    let same = parked.key.periodic == key.periodic
        && parked.key.cutoff == key.cutoff
        && parked.key.cell_hint == key.cell_hint
        && parked.key.half == key.half
        && parked.key.closed == key.closed
        && parked.key.exclude_self == key.exclude_self
        && parked.key.box_bits == key.box_bits
        && parked.key.mask == key.mask
        && parked.key.xyz == key.xyz;
    if same {
        Some(parked.rows)
    } else {
        None
    }
}

fn park_open(key: OpenKey, rows: Vec<lc_pair>) {
    PARKED_OPEN.with(|slot| {
        *slot.borrow_mut() = Some(ParkedOpen { key, rows });
    });
}

fn open_rows(
    pts: &[[f64; 3]],
    simbox: *const lc_cell,
    periodic: [bool; 3],
    cutoff: f64,
    mask: Option<&[c_int]>,
    hint: Option<f64>,
    key: &OpenKey,
) -> Result<Vec<lc_pair>, c_int> {
    let cell = if periodic == [false, false, false] {
        Cell::ortho(1.0, 1.0, 1.0).map_err(|err| fail(err.into()))?
    } else {
        let raw = unsafe { *simbox };
        crate::cell_for_pbc(
            [
                [raw.ax, raw.ay, raw.az],
                [raw.bx, raw.by, raw.bz],
                [raw.cx, raw.cy, raw.cz],
            ],
            [raw.ox, raw.oy, raw.oz],
            periodic,
        )
        .map_err(fail)?
    };
    let mask_vec: Option<Vec<bool>> = mask.map(|m| m.iter().map(|&v| v != 0).collect());
    let mut query = crate::PbcQuery::new(periodic, cutoff)
        .cell_hint(hint)
        .half(key.half)
        .closed(key.closed)
        .exclude_self_images(key.exclude_self);
    query.mask = mask_vec.as_deref();
    let found = with_threads(|| crate::pairs_within_pbc(pts, &cell, &query)).map_err(fail)?;
    let mut rows = Vec::new();
    rows.try_reserve(found.len())
        .map_err(|_| fail(Error::TooManyImages))?;
    for pair in found {
        if pair.i > c_int::MAX as usize || pair.j > c_int::MAX as usize {
            return Err(fail(Error::Overflow));
        }
        rows.push(lc_pair {
            i: pair.i as c_int,
            j: pair.j as c_int,
            shift: pair.shift,
            dist2: pair.dist2,
        });
    }
    Ok(rows)
}

/// Cutoff neighbours of every point as CSR rows.
///
/// Row `i` is `out_cols[out_offsets[i] .. out_offsets[i + 1]]`: every
/// `j != i` with an image whose squared distance is strictly below
/// `cutoff` squared, once, ascending. On a periodic axis that is the
/// minimum-image distance, also past half the box, where one atom meets
/// another through several images. A periodic image of `i` is not
/// listed. `half` nonzero keeps `j > i` only. `out_d2`, if non-null,
/// receives the squared distance of the nearest image of each column.
///
/// `pbc` is three ints, nonzero for a periodic axis, or null for a fully
/// periodic cell; `simbox` may be null when every axis is open. `mask`
/// and `cell_hint` are those of [`lc_pairs_within_pbc`]. A masked-out
/// point has an empty row and appears in no row. The rows and distances
/// are the same bits at every [`lc_set_threads`] width and SIMD path.
///
/// Sizing: every call sets `*need_offsets` (`n + 1`) and `*need_cols`.
/// Null `out_offsets`, `out_cols`, and `out_d2` is a query and returns
/// 0. Otherwise `out_offsets` holds `cap_offsets` ints and `out_cols`
/// (and `out_d2` if non-null) `cap_cols` values. A short buffer returns
/// nonzero without writing. A query or a short buffer keeps its rows on
/// the calling thread, and the next call there with the same inputs bit
/// for bit copies them out without searching again; a short buffer with
/// null `out_d2` keeps no distances, so a next call that asks for them
/// searches again. A buffer large enough on the first call takes the
/// rows straight from the search, and computes distances only when
/// `out_d2` is non-null.
///
/// `n == 0` succeeds with one offset, `0`. A cutoff that is not finite
/// and positive, a coordinate that is not finite, or more columns than
/// `INT_MAX` returns nonzero with [`lc_last_error`] set.
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `simbox`, if any axis is
/// periodic, points at one `lc_cell`. `pbc`, if non-null, is three ints.
/// `mask`, if non-null, is `n` ints. `need_offsets` and `need_cols` are
/// writable. On a fill, `out_offsets` is writable for `cap_offsets`
/// ints, `out_cols` for `cap_cols` ints, and `out_d2`, if non-null, for
/// `cap_cols` doubles.
#[no_mangle]
pub unsafe extern "C" fn lc_pairs_within_csr(
    xyz: *const f64,
    n: usize,
    simbox: *const lc_cell,
    pbc: *const c_int,
    cutoff: f64,
    mask: *const c_int,
    cell_hint: f64,
    half: c_int,
    out_offsets: *mut c_int,
    cap_offsets: usize,
    out_cols: *mut c_int,
    out_d2: *mut f64,
    cap_cols: usize,
    need_offsets: *mut usize,
    need_cols: *mut usize,
) -> c_int {
    if need_offsets.is_null() || need_cols.is_null() {
        return fail_msg("null pointer");
    }
    let query_only = out_offsets.is_null() && out_cols.is_null() && out_d2.is_null();
    if !query_only && (out_offsets.is_null() || out_cols.is_null()) {
        return fail_msg("null pointer");
    }
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return fail(Error::BadCutoff);
    }
    let periodic = if pbc.is_null() {
        [true, true, true]
    } else {
        let flags = unsafe { [*pbc, *pbc.add(1), *pbc.add(2)] };
        [flags[0] != 0, flags[1] != 0, flags[2] != 0]
    };
    if n == 0 {
        unsafe {
            *need_offsets = 1;
            *need_cols = 0;
        }
        if query_only {
            clear_error();
            return 0;
        }
        if cap_offsets < 1 {
            return fail_msg(CSR_SHORT);
        }
        unsafe { *out_offsets = 0 };
        clear_error();
        return 0;
    }
    if xyz.is_null() || (periodic != [false, false, false] && simbox.is_null()) {
        return fail_msg("null pointer");
    }
    if n > c_int::MAX as usize || n > (isize::MAX as usize) / 3 {
        return fail(Error::Overflow);
    }
    let hint = if cell_hint > 0.0 {
        Some(cell_hint)
    } else {
        None
    };
    let pts = unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n) };
    let mask_raw: Option<&[c_int]> = if mask.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(mask, n) })
    };
    let mask_vec: Option<Vec<bool>> = mask_raw.map(|m| m.iter().map(|&v| v != 0).collect());
    let mut query = crate::PbcQuery::new(periodic, cutoff)
        .cell_hint(hint)
        .half(half != 0);
    query.mask = mask_vec.as_deref();
    if let Err(err) = crate::pairs::csr_check(pts, &query) {
        return fail(err);
    }
    let key = || OpenKey {
        xyz: xyz_bits(pts),
        periodic,
        cutoff: cutoff.to_bits(),
        mask: mask_vec.clone(),
        cell_hint: hint.map(f64::to_bits),
        half: half != 0,
        closed: false,
        exclude_self: true,
        box_bits: if periodic == [false, false, false] {
            [0; 12]
        } else {
            box_bits(unsafe { &*simbox })
        },
    };
    let out = CsrOut {
        offsets: out_offsets,
        cap_offsets,
        cols: out_cols,
        d2: out_d2,
        cap_cols,
        query_only,
    };
    if let Some(parked) = take_csr(key, out) {
        return unsafe { deliver_csr(key, parked, out, need_offsets, need_cols) };
    }
    let cell = if periodic == [false, false, false] {
        match Cell::ortho(1.0, 1.0, 1.0) {
            Ok(cell) => cell,
            Err(err) => return fail(err.into()),
        }
    } else {
        let raw = unsafe { *simbox };
        match crate::cell_for_pbc(
            [
                [raw.ax, raw.ay, raw.az],
                [raw.bx, raw.by, raw.bz],
                [raw.cx, raw.cy, raw.cz],
            ],
            [raw.ox, raw.oy, raw.oz],
            periodic,
        ) {
            Ok(cell) => cell,
            Err(err) => return fail(err),
        }
    };
    let dest = crate::pairs::CsrDest {
        offsets: out_offsets,
        cols: out_cols,
        dist2: out_d2,
    };
    let room = move |need: usize| {
        (!out.query_only && out.cap_offsets > n && out.cap_cols >= need).then_some(dest)
    };
    // A query's parked rows carry distances: the next call may ask for them.
    let dist2 = query_only || !out_d2.is_null();
    let made = with_threads(|| crate::pairs::csr_rows(pts, &cell, &query, dist2, room));
    match made {
        Ok(crate::pairs::CsrWrite::Direct(need)) => {
            unsafe {
                *need_offsets = n + 1;
                *need_cols = need;
            }
            clear_error();
            0
        }
        Ok(crate::pairs::CsrWrite::Owned(rows)) => unsafe {
            deliver_csr(key, rows, out, need_offsets, need_cols)
        },
        Err(err) => fail(err),
    }
}

const CSR_SHORT: &str = "CSR buffer is shorter than the row count";

/// Caller buffers of one [`lc_pairs_within_csr`] call.
#[derive(Clone, Copy)]
struct CsrOut {
    offsets: *mut c_int,
    cap_offsets: usize,
    cols: *mut c_int,
    d2: *mut f64,
    cap_cols: usize,
    query_only: bool,
}
// Safety: only the sizes and the query flag are read off the calling
// thread; the buffers are written through `CsrDest`.
unsafe impl Send for CsrOut {}
unsafe impl Sync for CsrOut {}

/// Rows a query or a short buffer left for the next matching call.
struct ParkedCsr {
    key: OpenKey,
    rows: crate::PairCsr,
}

thread_local! {
    static PARKED_CSR: RefCell<Option<ParkedCsr>> = const { RefCell::new(None) };
}

/// The parked rows when every input matches and they hold the
/// distances `out` asks for. Any other call drops them.
fn take_csr(key: impl FnOnce() -> OpenKey, out: CsrOut) -> Option<crate::PairCsr> {
    let parked = PARKED_CSR.with(|slot| slot.borrow_mut().take())?;
    let d2 = out.d2.is_null() || parked.rows.dist2.len() == parked.rows.cols.len();
    (d2 && parked.key == key()).then_some(parked.rows)
}

/// Report the sizes of `rows`, then park them (a query or a short
/// buffer) or copy them out.
///
/// # Safety
///
/// `need_offsets` and `need_cols` are writable, and `out` follows the
/// contract of [`lc_pairs_within_csr`].
unsafe fn deliver_csr(
    key: impl FnOnce() -> OpenKey,
    rows: crate::PairCsr,
    out: CsrOut,
    need_offsets: *mut usize,
    need_cols: *mut usize,
) -> c_int {
    unsafe {
        *need_offsets = rows.offsets.len();
        *need_cols = rows.cols.len();
    }
    if out.query_only {
        PARKED_CSR.with(|slot| *slot.borrow_mut() = Some(ParkedCsr { key: key(), rows }));
        clear_error();
        return 0;
    }
    if out.cap_offsets < rows.offsets.len() || out.cap_cols < rows.cols.len() {
        PARKED_CSR.with(|slot| *slot.borrow_mut() = Some(ParkedCsr { key: key(), rows }));
        return fail_msg(CSR_SHORT);
    }
    unsafe {
        ptr::copy_nonoverlapping(rows.offsets.as_ptr(), out.offsets, rows.offsets.len());
        ptr::copy_nonoverlapping(rows.cols.as_ptr(), out.cols, rows.cols.len());
        if !out.d2.is_null() {
            ptr::copy_nonoverlapping(rows.dist2.as_ptr(), out.d2, rows.dist2.len());
        }
    }
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
    let found = with_threads(|| {
        crate::pairs::in_pool(crate::pairs::hit_estimate(n, &sim, cutoff), || {
            crate::pairs::search(pts, &sim, cutoff, mask_vec.as_deref(), hint, half != 0)
        })
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
    static PARKED_PAIRS: RefCell<Option<ParkedPairs>> = const { RefCell::new(None) };
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

#[allow(clippy::type_complexity)]
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
    let hint = if cell_hint > 0.0 {
        Some(cell_hint)
    } else {
        None
    };
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
    let (pts, sim, mask_vec, hint) = match unsafe { load_frame(xyz, n, simbox, mask, cell_hint) } {
        Ok(v) => v,
        Err(rc) => return rc,
    };
    let grid = match with_threads(|| crate::cell_grid(pts, &sim, cutoff, mask_vec.as_deref(), hint))
    {
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
    let (pts, sim, mask_vec, hint) = match unsafe { load_frame(xyz, n, simbox, mask, cell_hint) } {
        Ok(v) => v,
        Err(rc) => return rc,
    };
    let graph =
        match with_threads(|| crate::cell_graph(pts, &sim, cutoff, mask_vec.as_deref(), hint)) {
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
/// 5 Scotch, 6 multilevel, 7 PT-Scotch. `balance <= 0` selects
/// `1 + parts * w_max / total`, where `w_max` is the heaviest cell
/// weight and `total` is the sum of the cell weights. `device` nonzero runs the
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
    let (pts, sim, mask_vec, hint) = match unsafe { load_frame(xyz, n, simbox, mask, cell_hint) } {
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
    let assign = match with_threads(|| {
        crate::partition(pts, &sim, cutoff, mask_vec.as_deref(), hint, request)
    }) {
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
    let ghosts = match with_threads(|| crate::halo(&grid, &owned, domain as u32, radius)) {
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

/// Domain of each cell and the bin of each point, from one mesh cut.
///
/// `method` and `balance` follow [`lc_partition`]. A null `out_domain`
/// queries `*out_ncell` and does not write. On a fill, `out_nxyz` is
/// three ints, `out_width` is three bin widths, `out_domain` is one
/// domain per cell, and `out_point_cell` is one bin per point.
///
/// # Safety
///
/// Same point contract as [`lc_cell_grid`]. On a fill, `out_domain` is
/// writable for `*out_ncell` ints and `out_point_cell` for `n` ints.
#[no_mangle]
pub unsafe extern "C" fn lc_mesh_cut(
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
    out_nxyz: *mut c_int,
    out_width: *mut f64,
    out_domain: *mut c_int,
    out_point_cell: *mut c_int,
    cap_cell: usize,
    cap_point: usize,
    out_ncell: *mut usize,
) -> c_int {
    if out_ncell.is_null() {
        return fail_msg("null pointer");
    }
    if parts == 0 || parts > u32::MAX as usize {
        return fail(Error::BadPartition);
    }
    let (pts, sim, mask_vec, hint) = match unsafe { load_frame(xyz, n, simbox, mask, cell_hint) } {
        Ok(v) => v,
        Err(rc) => return rc,
    };
    let method = match crate::partition::method_from_c(method) {
        Ok(method) => method,
        Err(err) => return fail(err),
    };
    let request = crate::PartRequest {
        parts: parts as u32,
        method,
        balance: if balance <= 0.0 { None } else { Some(balance) },
        device: device != 0,
    };
    let cut = match with_threads(|| {
        crate::mesh_cut(pts, &sim, cutoff, mask_vec.as_deref(), hint, request)
    }) {
        Ok(cut) => cut,
        Err(err) => return fail(err),
    };
    let ncell = cut.assign.len();
    unsafe { *out_ncell = ncell };
    if out_domain.is_null() {
        clear_error();
        return 0;
    }
    if out_nxyz.is_null()
        || out_width.is_null()
        || out_point_cell.is_null()
        || cap_cell < ncell
        || cap_point < n
    {
        return fail_msg("mesh cut buffer is shorter than the cell count");
    }
    write_grid_head(&cut.grid, out_nxyz, out_width);
    for (i, &domain) in cut.assign.iter().enumerate() {
        if domain > c_int::MAX as u32 {
            return fail(Error::Overflow);
        }
        unsafe { *out_domain.add(i) = domain as c_int };
    }
    for (i, &cell) in cut.point_cell.iter().enumerate() {
        if cell > c_int::MAX as u32 {
            return fail(Error::Overflow);
        }
        unsafe { *out_point_cell.add(i) = cell as c_int };
    }
    clear_error();
    0
}

/// Domain of each cell from caller-supplied weights.
///
/// `nxyz` is `nx, ny, nz`. `weight` has one count per cell, in the
/// order `((iz * ny) + iy) * nx + ix`. `method` and `balance` follow
/// [`lc_partition`]. A plain bisection reads the weights and not the
/// pair arcs. `out_domain` receives one domain per cell.
///
/// # Safety
///
/// `nxyz` is readable for 3 ints. `weight` is readable for `ncell`
/// counts. `out_domain` is writable for `ncell` ints.
#[no_mangle]
pub unsafe extern "C" fn lc_partition_weights(
    nxyz: *const c_int,
    weight: *const u64,
    ncell: usize,
    parts: usize,
    method: u8,
    balance: f64,
    out_domain: *mut c_int,
) -> c_int {
    if nxyz.is_null() || weight.is_null() || out_domain.is_null() {
        return fail_msg("null pointer");
    }
    if parts == 0 || parts > u32::MAX as usize {
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
    let method = match crate::partition::method_from_c(method) {
        Ok(method) => method,
        Err(err) => return fail(err),
    };
    let raw = unsafe { std::slice::from_raw_parts(weight, ncell) };
    let graph = crate::CellGraph {
        grid: crate::CellGrid {
            nx: nxyz[0],
            ny: nxyz[1],
            nz: nxyz[2],
            bin_width: [1.0; 3],
            weight: raw.to_vec(),
        },
        offsets: vec![0u32; ncell + 1],
        neighbor: Vec::new(),
        edge_weight: Vec::new(),
    };
    let request = crate::PartRequest {
        parts: parts as u32,
        method,
        balance: if balance <= 0.0 { None } else { Some(balance) },
        device: false,
    };
    let assign = match with_threads(|| crate::partition_graph(&graph, request)) {
        Ok(assign) => assign,
        Err(err) => return fail(err),
    };
    for (i, &domain) in assign.iter().enumerate() {
        if domain > c_int::MAX as u32 {
            return fail(Error::Overflow);
        }
        unsafe { *out_domain.add(i) = domain as c_int };
    }
    clear_error();
    0
}

/// Owned points are 2. Ghosts within `reach` of an owned atom are 1.
///
/// # Safety
///
/// `xyz` is readable for `n * 3` doubles. `lengths` and `nxyz` are
/// readable for 3 entries. `assign` is readable for `ncell` ints.
/// `point_cell` is readable for `n` ints. `hwidth` is readable for 3
/// counts. `out_hit` is writable for `n` ints.
#[no_mangle]
pub unsafe extern "C" fn lc_within_reach(
    xyz: *const f64,
    n: usize,
    lengths: *const f64,
    nxyz: *const c_int,
    assign: *const c_int,
    ncell: usize,
    point_cell: *const c_int,
    hwidth: *const u32,
    domain: c_int,
    reach: f64,
    out_hit: *mut c_int,
) -> c_int {
    if xyz.is_null()
        || lengths.is_null()
        || nxyz.is_null()
        || assign.is_null()
        || point_cell.is_null()
        || hwidth.is_null()
        || out_hit.is_null()
        || domain < 0
    {
        return fail_msg("null pointer");
    }
    let pts: Vec<[f64; 3]> =
        unsafe { std::slice::from_raw_parts(xyz.cast::<[f64; 3]>(), n) }.to_vec();
    let lengths = unsafe { [*lengths, *lengths.add(1), *lengths.add(2)] };
    let nxyz = unsafe { [*nxyz, *nxyz.add(1), *nxyz.add(2)] };
    let hwidth = unsafe { [*hwidth, *hwidth.add(1), *hwidth.add(2)] };
    let assign_raw = unsafe { std::slice::from_raw_parts(assign, ncell) };
    let mut owned = Vec::with_capacity(ncell);
    for &v in assign_raw {
        if v < 0 {
            return fail(Error::BadPartition);
        }
        owned.push(v as u32);
    }
    let cells_raw = unsafe { std::slice::from_raw_parts(point_cell, n) };
    let mut cells = Vec::with_capacity(n);
    for &v in cells_raw {
        if v < 0 {
            return fail(Error::BadPartition);
        }
        cells.push(v as u32);
    }
    let hit = match with_threads(|| {
        crate::mark_reached(
            &pts,
            lengths,
            nxyz,
            &owned,
            &cells,
            hwidth,
            domain as u32,
            reach,
        )
    }) {
        Ok(hit) => hit,
        Err(err) => return fail(err),
    };
    for (i, &flag) in hit.iter().enumerate() {
        unsafe { *out_hit.add(i) = flag as c_int };
    }
    clear_error();
    0
}

/// Owned points are 2. Points in a cell painted from the owned walk are 1.
///
/// The stencil is the one [`crate::rank_mask`] paints while it visits the
/// cells of `domain`. `width` is three bin radii.
///
/// # Safety
///
/// `nxyz` is readable for 3 ints. `assign` is readable for `ncell` ints.
/// `point_cell` is readable for `n` ints. `width` is readable for 3
/// counts. `out_hit` is writable for `n` ints.
#[no_mangle]
pub unsafe extern "C" fn lc_owned_mask(
    nxyz: *const c_int,
    assign: *const c_int,
    ncell: usize,
    point_cell: *const c_int,
    n: usize,
    domain: c_int,
    width: *const u32,
    out_hit: *mut c_int,
) -> c_int {
    if nxyz.is_null()
        || assign.is_null()
        || point_cell.is_null()
        || width.is_null()
        || out_hit.is_null()
        || domain < 0
    {
        return fail_msg("null pointer");
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
    let assign_raw = unsafe { std::slice::from_raw_parts(assign, ncell) };
    let mut owned_assign = Vec::with_capacity(ncell);
    let mut owned_cells = Vec::new();
    for (cell, &v) in assign_raw.iter().enumerate() {
        if v < 0 {
            return fail(Error::BadPartition);
        }
        let domain_id = v as u32;
        if domain_id == domain as u32 {
            owned_cells.push(cell as u32);
        }
        owned_assign.push(domain_id);
    }
    let cells_raw = unsafe { std::slice::from_raw_parts(point_cell, n) };
    let mut point_bins = Vec::with_capacity(n);
    for &v in cells_raw {
        if v < 0 || v as usize >= ncell {
            return fail(Error::BadPartition);
        }
        point_bins.push(v as u32);
    }
    let width = unsafe { [*width, *width.add(1), *width.add(2)] };
    let grid = crate::CellGrid {
        nx: nxyz[0],
        ny: nxyz[1],
        nz: nxyz[2],
        bin_width: [1.0; 3],
        weight: vec![0; ncell],
    };
    let view = with_threads(|| {
        crate::rank_mask(
            &grid,
            &owned_assign,
            &point_bins,
            &owned_cells,
            domain as u32,
            width,
        )
    });
    for (i, &painted) in view.mask.iter().enumerate() {
        let flag = if !painted {
            0
        } else if owned_assign[point_bins[i] as usize] == domain as u32 {
            2
        } else {
            1
        };
        unsafe { *out_hit.add(i) = flag };
    }
    clear_error();
    0
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    struct Tally {
        sizes: [usize; 3],
        hits: Vec<Vec<AtomicUsize>>,
        closed: [AtomicUsize; 3],
        early: AtomicUsize,
        widest: AtomicUsize,
    }

    extern "C" fn tally(ctx: *mut std::ffi::c_void, stage: usize, chunk: usize, worker: usize) {
        let t = unsafe { &*ctx.cast::<Tally>() };
        if stage > 0 && t.closed[stage - 1].load(Ordering::SeqCst) != t.sizes[stage - 1] {
            t.early.fetch_add(1, Ordering::SeqCst);
        }
        t.widest.fetch_max(worker + 1, Ordering::SeqCst);
        t.hits[stage][chunk].fetch_add(1, Ordering::SeqCst);
        t.closed[stage].fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn a_team_runs_every_chunk_once_and_the_stages_in_order() {
        for threads in [1, 4] {
            lc_set_threads(threads);
            let sizes = [97, 0, 13];
            let t = Tally {
                sizes,
                hits: sizes
                    .iter()
                    .map(|&n| (0..n).map(|_| AtomicUsize::new(0)).collect())
                    .collect(),
                closed: Default::default(),
                early: AtomicUsize::new(0),
                widest: AtomicUsize::new(0),
            };
            let code = unsafe {
                lc_team_run(
                    3,
                    sizes.as_ptr(),
                    Some(tally),
                    (&t as *const Tally).cast_mut().cast(),
                )
            };
            lc_set_threads(0);
            assert_eq!(code, 0);
            assert_eq!(t.early.load(Ordering::SeqCst), 0);
            assert!(t
                .hits
                .iter()
                .flatten()
                .all(|h| h.load(Ordering::SeqCst) == 1));
            let bound = (threads as usize).max(rayon::current_num_threads());
            assert!(t.widest.load(Ordering::SeqCst) <= bound);
        }
        assert_ne!(
            unsafe { lc_team_run(1, ptr::null(), Some(tally), ptr::null_mut()) },
            0
        );
        assert_ne!(
            unsafe { lc_team_run(1, [1].as_ptr(), None, ptr::null_mut()) },
            0
        );
    }
}
