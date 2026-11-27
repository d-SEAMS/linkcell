#![cfg(all(feature = "capi", target_os = "linux", target_env = "gnu"))]
//! After `lc_set_threads(1)` a C API call creates no thread, not even one
//! it joins before returning. This executable defines `pthread_create`
//! and forwards it to libc, so every thread linkcell's Rust code starts,
//! however briefly, is counted.

use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use linkcell::lc_cell;

static CREATED: AtomicUsize = AtomicUsize::new(0);

type Start = extern "C" fn(*mut c_void) -> *mut c_void;
type Create = unsafe extern "C" fn(*mut c_void, *const c_void, Start, *mut c_void) -> c_int;

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn abort() -> !;
}

const RTLD_NEXT: *mut c_void = -1isize as *mut c_void;

/// Counts the thread and creates it with libc's `pthread_create`.
///
/// # Safety
///
/// The arguments are those of `pthread_create(3)`.
#[no_mangle]
pub unsafe extern "C" fn pthread_create(
    thread: *mut c_void,
    attr: *const c_void,
    start: Start,
    arg: *mut c_void,
) -> c_int {
    CREATED.fetch_add(1, Ordering::SeqCst);
    let real = dlsym(RTLD_NEXT, c"pthread_create".as_ptr());
    if real.is_null() {
        abort();
    }
    let real: Create = std::mem::transmute(real);
    real(thread, attr, start, arg)
}

fn task_count() -> usize {
    std::fs::read_dir("/proc/self/task")
        .map(|dir| dir.count())
        .unwrap_or(0)
}

/// 12000 points on a jittered lattice, above the size at which binning
/// splits over threads.
fn frame() -> (Vec<f64>, lc_cell) {
    let n = 12000usize;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut jitter = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let xyz = (0..n)
        .flat_map(|i| {
            [
                (i % 24) as f64 * 1.5 + 0.3 * jitter(),
                ((i / 24) % 25) as f64 * 1.5 + 0.3 * jitter(),
                (i / 600) as f64 * 1.5 + 0.3 * jitter(),
            ]
        })
        .collect();
    let cell = lc_cell {
        ax: 36.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 37.5,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 30.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    (xyz, cell)
}

#[test]
fn one_thread_calls_create_no_thread() {
    let (xyz, cell) = frame();
    let n = xyz.len() / 3;
    let cutoff = 2.0;
    let null = std::ptr::null::<c_int>();
    linkcell::lc_set_threads(1);
    let tasks = task_count();
    let created = CREATED.load(Ordering::SeqCst);
    let same = |label: &str| {
        assert_eq!(
            CREATED.load(Ordering::SeqCst),
            created,
            "a one-thread {label} created a thread"
        );
        assert_eq!(task_count(), tasks, "a one-thread {label} left a thread");
    };

    let mut nn = vec![-1 as c_int; n * 4];
    let rc =
        unsafe { linkcell::lc_knearest(xyz.as_ptr(), n, &cell, 4, null, 0.0, nn.as_mut_ptr()) };
    assert_eq!(rc, 0, "k-nearest");
    same("k-nearest search");

    let mut rows = 0usize;
    let rc = unsafe {
        linkcell::lc_pairs_within(
            xyz.as_ptr(),
            n,
            &cell,
            cutoff,
            null,
            0.0,
            1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut rows,
        )
    };
    assert_eq!(rc, 0, "pair count");
    assert!(rows > 0, "the frame has pairs within the cutoff");
    same("pair count");

    // A query at another cutoff sizes the buffers and drops the kept
    // search, so the fill below searches and fills its columns itself.
    let mut cap = 0usize;
    let rc = unsafe {
        linkcell::lc_pairs_within(
            xyz.as_ptr(),
            n,
            &cell,
            cutoff + 0.25,
            null,
            0.0,
            1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut cap,
        )
    };
    assert_eq!(rc, 0, "pair count at the wider cutoff");
    assert!(cap >= rows, "a wider cutoff keeps every pair");
    let mut pi = vec![0 as c_int; cap];
    let mut pj = vec![0 as c_int; cap];
    let mut shift = vec![0 as c_int; cap * 3];
    let mut d2 = vec![0.0f64; cap];
    let mut filled = 0usize;
    let rc = unsafe {
        linkcell::lc_pairs_within(
            xyz.as_ptr(),
            n,
            &cell,
            cutoff,
            null,
            0.0,
            1,
            pi.as_mut_ptr(),
            pj.as_mut_ptr(),
            shift.as_mut_ptr(),
            d2.as_mut_ptr(),
            cap,
            &mut filled,
        )
    };
    assert_eq!(rc, 0, "pair fill");
    assert_eq!(filled, rows, "pair fill row count");
    same("pair fill");

    let mut need = (0usize, 0usize);
    let csr = |offsets: *mut c_int, cols: *mut c_int, cap: usize, need: &mut (usize, usize)| unsafe {
        linkcell::lc_pairs_within_csr(
            xyz.as_ptr(),
            n,
            &cell,
            std::ptr::null(),
            cutoff,
            null,
            0.0,
            0,
            offsets,
            n + 1,
            cols,
            std::ptr::null_mut(),
            cap,
            &mut need.0,
            &mut need.1,
        )
    };
    let rc = csr(std::ptr::null_mut(), std::ptr::null_mut(), 0, &mut need);
    assert_eq!(rc, 0, "CSR query");
    assert!(need.1 > 0, "the frame has CSR columns");
    same("CSR query");
    let mut offsets = vec![0 as c_int; n + 1];
    let mut cols = vec![0 as c_int; need.1];
    let rc = csr(offsets.as_mut_ptr(), cols.as_mut_ptr(), need.1, &mut need);
    assert_eq!(rc, 0, "CSR fill");
    same("CSR fill");
    // A different cutoff drops the kept rows, so this one searches.
    let rc = unsafe {
        linkcell::lc_pairs_within_csr(
            xyz.as_ptr(),
            n,
            &cell,
            std::ptr::null(),
            cutoff - 0.25,
            null,
            0.0,
            1,
            offsets.as_mut_ptr(),
            n + 1,
            cols.as_mut_ptr(),
            std::ptr::null_mut(),
            need.1,
            &mut need.0,
            &mut need.1,
        )
    };
    assert_eq!(rc, 0, "CSR fill straight from the search");
    same("CSR search and fill");

    let mut ncell = 0usize;
    let rc = unsafe {
        linkcell::lc_cell_grid(
            xyz.as_ptr(),
            n,
            &cell,
            cutoff,
            null,
            0.0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut ncell,
        )
    };
    assert_eq!(rc, 0, "cell grid");
    assert!(ncell > 1, "the grid has more than one cell");
    same("cell grid");

    let mut arcs = 0usize;
    let rc = unsafe {
        linkcell::lc_cell_graph(
            xyz.as_ptr(),
            n,
            &cell,
            cutoff,
            null,
            0.0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            0,
            &mut ncell,
            &mut arcs,
        )
    };
    assert_eq!(rc, 0, "cell graph");
    same("cell graph");

    let one_grid = grid_fill(&xyz, &cell, cutoff, ncell);
    same("cell grid fill");
    let one_parts: Vec<Vec<c_int>> = [1u8, 4]
        .into_iter()
        .map(|method| {
            let domain = partition_fill(&xyz, &cell, cutoff, ncell, method);
            same(&format!("partition (method {method})"));
            domain
        })
        .collect();

    nd_calls(&xyz[..600 * 3], &same);

    linkcell::lc_set_threads(0);
    assert_eq!(
        grid_fill(&xyz, &cell, cutoff, ncell),
        one_grid,
        "the cell grid differs between one thread and the default width"
    );
    for (method, one) in [1u8, 4].into_iter().zip(&one_parts) {
        assert_eq!(
            &partition_fill(&xyz, &cell, cutoff, ncell, method),
            one,
            "partition method {method} differs between one thread and the default width"
        );
    }
}

/// Every N-dimensional call, each checked by `same`. Without the caller
/// route, `lc_nd_pairs_within` would split its rows over rayon's global pool
/// at any point count.
fn nd_calls(xyz: &[f64], same: &dyn Fn(&str)) {
    let n = xyz.len() / 3;
    let mut nd = std::ptr::null_mut();
    let rc = unsafe { linkcell::lc_nd_build(xyz.as_ptr(), n, 3, std::ptr::null(), &mut nd) };
    assert_eq!(rc, 0, "ND build");
    same("ND build");

    let q = &xyz[..3];
    let mut nn = vec![-1 as c_int; 8];
    let rc =
        unsafe { linkcell::lc_nd_knn(nd, q.as_ptr(), 8, nn.as_mut_ptr(), std::ptr::null_mut()) };
    assert_eq!(rc, 0, "ND k-nearest");
    assert!(nn.iter().all(|&j| j >= 0), "ND k-nearest left a slot empty");
    same("ND k-nearest");

    let mut count = 0usize;
    let rc = unsafe {
        linkcell::lc_nd_within(
            nd,
            q.as_ptr(),
            2.0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(rc, 0, "ND radius count");
    assert!(count > 1, "the query has radius neighbours");
    same("ND radius count");

    let rc = unsafe {
        linkcell::lc_nd_pairs_within(
            nd,
            2.0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(rc, 0, "ND pair count");
    assert!(count > 0, "the slice has pairs within the radius");
    same("ND pair count");

    let mut labels = vec![-2 as c_int; n];
    let mut core = vec![0 as c_int; n];
    let mut clusters = 0usize;
    let rc = unsafe {
        linkcell::lc_nd_dbscan(
            nd,
            2.0,
            4,
            labels.as_mut_ptr(),
            core.as_mut_ptr(),
            &mut clusters,
        )
    };
    assert_eq!(rc, 0, "ND DBSCAN");
    assert!(clusters > 0, "DBSCAN found a cluster");
    same("ND DBSCAN");

    unsafe { linkcell::lc_nd_free(nd) };
    same("ND free");
}

fn grid_fill(xyz: &[f64], cell: &lc_cell, cutoff: f64, ncell: usize) -> (Vec<c_int>, Vec<u64>) {
    let mut nxyz = vec![0 as c_int; 3];
    let mut width = vec![0.0f64; 3];
    let mut weight = vec![0u64; ncell];
    let mut filled = 0usize;
    let rc = unsafe {
        linkcell::lc_cell_grid(
            xyz.as_ptr(),
            xyz.len() / 3,
            cell,
            cutoff,
            std::ptr::null(),
            0.0,
            nxyz.as_mut_ptr(),
            width.as_mut_ptr(),
            weight.as_mut_ptr(),
            ncell,
            &mut filled,
        )
    };
    assert_eq!(rc, 0, "cell grid fill");
    assert_eq!(filled, ncell, "cell grid fill cell count");
    (nxyz, weight)
}

fn partition_fill(
    xyz: &[f64],
    cell: &lc_cell,
    cutoff: f64,
    ncell: usize,
    method: u8,
) -> Vec<c_int> {
    let mut domain = vec![-1 as c_int; ncell];
    let mut filled = 0usize;
    let rc = unsafe {
        linkcell::lc_partition(
            xyz.as_ptr(),
            xyz.len() / 3,
            cell,
            cutoff,
            std::ptr::null(),
            0.0,
            4,
            method,
            0.0,
            0,
            domain.as_mut_ptr(),
            ncell,
            &mut filled,
        )
    };
    assert_eq!(rc, 0, "partition method {method}");
    assert_eq!(filled, ncell, "partition method {method} cell count");
    assert!(
        domain.iter().all(|&d| (0..4).contains(&d)),
        "partition method {method} left a cell without a domain"
    );
    domain
}
