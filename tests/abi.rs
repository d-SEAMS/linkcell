#![cfg(feature = "capi")]
//! C ABI buffer contract. Feature `capi` is on by default.

use std::os::raw::c_int;

use linkcell::{knearest_into, Cell, Error};

#[repr(C)]
struct LcCell {
    ax: f64,
    ay: f64,
    az: f64,
    bx: f64,
    by: f64,
    bz: f64,
    cx: f64,
    cy: f64,
    cz: f64,
    ox: f64,
    oy: f64,
    oz: f64,
}

#[repr(C)]
struct LcNdIndex {
    _private: [u8; 0],
}

extern "C" {
    fn lc_knearest(
        xyz: *const f64,
        n: usize,
        simbox: *const LcCell,
        k: usize,
        mask: *const c_int,
        cell_hint: f64,
        out_nn: *mut c_int,
    ) -> c_int;
    fn lc_knearest_d2(
        xyz: *const f64,
        n: usize,
        simbox: *const LcCell,
        k: usize,
        mask: *const c_int,
        cell_hint: f64,
        out_nn: *mut c_int,
        out_d2: *mut f64,
    ) -> c_int;
    fn lc_knearest_many(
        xyz: *const f64,
        n: usize,
        n_frames: usize,
        simbox: *const LcCell,
        k: usize,
        mask: *const c_int,
        cell_hint: f64,
        out_nn: *mut c_int,
        out_d2: *mut f64,
    ) -> c_int;
    fn lc_pairs_within(
        xyz: *const f64,
        n: usize,
        simbox: *const LcCell,
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
    ) -> c_int;
    fn lc_pairs_within_rows(
        xyz: *const f64,
        n: usize,
        simbox: *const LcCell,
        cutoff: f64,
        mask: *const c_int,
        cell_hint: f64,
        half: c_int,
        out: *mut linkcell::lc_pair,
        cap: usize,
        out_count: *mut usize,
    ) -> c_int;
    fn lc_nd_build(
        data: *const f64,
        n: usize,
        d: usize,
        periods: *const f64,
        out: *mut *mut LcNdIndex,
    ) -> c_int;
    fn lc_nd_free(index: *mut LcNdIndex);
    fn lc_nd_len(index: *const LcNdIndex) -> usize;
    fn lc_nd_dims(index: *const LcNdIndex) -> usize;
    fn lc_nd_knn(
        index: *const LcNdIndex,
        query: *const f64,
        k: usize,
        out_idx: *mut c_int,
        out_d2: *mut f64,
    ) -> c_int;
    fn lc_nd_within(
        index: *const LcNdIndex,
        query: *const f64,
        radius: f64,
        out_idx: *mut c_int,
        out_d2: *mut f64,
        cap: usize,
        out_count: *mut usize,
    ) -> c_int;
    fn lc_nd_pairs_within(
        index: *const LcNdIndex,
        radius: f64,
        out_i: *mut c_int,
        out_j: *mut c_int,
        out_d2: *mut f64,
        cap: usize,
        out_count: *mut usize,
    ) -> c_int;
    fn lc_nd_dbscan(
        index: *const LcNdIndex,
        eps: f64,
        min_pts: usize,
        out_labels: *mut c_int,
        out_core: *mut c_int,
        out_clusters: *mut usize,
    ) -> c_int;
}

fn ortho_c(lx: f64, ly: f64, lz: f64) -> LcCell {
    LcCell {
        ax: lx,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: ly,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: lz,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    }
}

fn pack_xyz(xyz: &[[f64; 3]]) -> Vec<f64> {
    let mut packed = Vec::with_capacity(xyz.len() * 3);
    for p in xyz {
        packed.extend_from_slice(p);
    }
    packed
}

#[test]
fn safe_and_c_abi_write_the_same_packed_row() {
    let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mut rust_out = [-1; 2];
    knearest_into(&xyz, &sim, 1, None, None, &mut rust_out).unwrap();

    let box_c = linkcell::lc_cell {
        ax: 10.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 10.0,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 10.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let packed = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut c_out = [-1; 2];
    let rc = unsafe {
        linkcell::lc_knearest(
            packed.as_ptr(),
            2,
            &box_c,
            1,
            std::ptr::null(),
            0.0,
            c_out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(c_out, rust_out);
    assert!(linkcell::lc_last_error().is_null());
}

#[test]
fn c_abi_empty_n_sets_thread_local_error() {
    let box_c = linkcell::lc_cell {
        ax: 10.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 10.0,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 10.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let dummy = 0.0;
    let mut out = -1;
    let rc =
        unsafe { linkcell::lc_knearest(&dummy, 0, &box_c, 1, std::ptr::null(), 0.0, &mut out) };
    assert_ne!(rc, 0);
    let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
    assert_eq!(msg.to_str().unwrap(), "no points");
}

#[test]
fn buffer_size_is_not_empty() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0]];
    let mut out = [];
    assert_eq!(
        linkcell::knearest_into(&xyz, &b, 1, None, None, &mut out).unwrap_err(),
        Error::BufferSize
    );
}

#[test]
fn lc_knearest_matches_knearest_into_packed() {
    let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 2.0],
        [9.7, 0.2, 0.1],
    ];
    let k = 3;
    let n = xyz.len();
    let mut rust_out = vec![-2i32; n * k];
    knearest_into(&xyz, &sim, k, None, None, &mut rust_out).unwrap();

    let packed = pack_xyz(&xyz);
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let mut c_out = vec![-7i32; n * k];
    let rc = unsafe {
        lc_knearest(
            packed.as_ptr(),
            n,
            &box_c,
            k,
            std::ptr::null(),
            0.0,
            c_out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(c_out, rust_out);
    for i in 0..n {
        for j in 0..k {
            assert_eq!(c_out[i * k + j], rust_out[i * k + j], "out[{i}*{k}+{j}]");
        }
    }
}

#[test]
fn lc_knearest_null_mask_matches_all_ones() {
    let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.4, 0.0],
        [0.0, 0.0, 1.6],
    ];
    let k = 2;
    let n = xyz.len();
    let packed = pack_xyz(&xyz);
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let ones = vec![1 as c_int; n];
    let mut out_null = vec![-1i32; n * k];
    let mut out_ones = vec![-1i32; n * k];
    let rc_null = unsafe {
        lc_knearest(
            packed.as_ptr(),
            n,
            &box_c,
            k,
            std::ptr::null(),
            0.0,
            out_null.as_mut_ptr(),
        )
    };
    let rc_ones = unsafe {
        lc_knearest(
            packed.as_ptr(),
            n,
            &box_c,
            k,
            ones.as_ptr(),
            0.0,
            out_ones.as_mut_ptr(),
        )
    };
    assert_eq!(rc_null, 0);
    assert_eq!(rc_ones, 0);
    assert_eq!(out_null, out_ones);

    let mut rust_out = vec![-1i32; n * k];
    knearest_into(&xyz, &sim, k, None, None, &mut rust_out).unwrap();
    assert_eq!(out_null, rust_out);
}

#[test]
fn lc_knearest_zero_k_message_is_not_empty() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let packed = [0.0, 0.0, 0.0];
    let mut out = -1;
    let rc = unsafe {
        lc_knearest(
            packed.as_ptr(),
            1,
            &box_c,
            0,
            std::ptr::null(),
            0.0,
            &mut out,
        )
    };
    assert_ne!(rc, 0);
    let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
    assert_eq!(msg.to_str().unwrap(), "k must be at least 1");
    assert_ne!(msg.to_str().unwrap(), "no points");
}

#[test]
fn last_error_slots_are_independent_across_threads() {
    use std::thread;
    let a = thread::spawn(|| {
        let box_c = ortho_c(10.0, 10.0, 10.0);
        let dummy = 0.0;
        let mut out = -1;
        let rc = unsafe { lc_knearest(&dummy, 0, &box_c, 1, std::ptr::null(), 0.0, &mut out) };
        assert_ne!(rc, 0);
        let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "no points");
    });
    let b = thread::spawn(|| {
        let box_c = ortho_c(10.0, 10.0, 10.0);
        let dummy = 0.0;
        let mut out = -1;
        let rc = unsafe { lc_knearest(&dummy, 1, &box_c, 0, std::ptr::null(), 0.0, &mut out) };
        assert_ne!(rc, 0);
        let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "k must be at least 1");
    });
    a.join().unwrap();
    b.join().unwrap();
}

#[test]
fn lc_version_does_not_clear_last_error() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let dummy = 0.0;
    let mut out = -1;
    let rc = unsafe { lc_knearest(&dummy, 0, &box_c, 1, std::ptr::null(), 0.0, &mut out) };
    assert_ne!(rc, 0);
    let before = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) }
        .to_str()
        .unwrap()
        .to_string();
    let _v = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_version()) };
    let after = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) }
        .to_str()
        .unwrap();
    assert_eq!(before, "no points");
    assert_eq!(after, before);
}

#[test]
fn c_abi_overflow_is_not_empty() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let dummy = 0.0;
    let mut out = -1;
    let n = (isize::MAX as usize) / 3 + 1;
    let rc = unsafe { lc_knearest(&dummy, n, &box_c, 1, std::ptr::null(), 0.0, &mut out) };
    assert_ne!(rc, 0);
    let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
    assert_eq!(msg.to_str().unwrap(), "n * k overflows");
    assert_ne!(msg.to_str().unwrap(), "no points");
    assert_ne!(msg.to_str().unwrap(), "out buffer length must be n * k");
}

#[test]
fn lc_knearest_d2_writes_periodic_image() {
    let packed = [0.2, 0.0, 0.0, 9.4, 0.0, 0.0];
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let mut nn = [-1i32; 2];
    let mut d2 = [0.0f64; 2];
    let rc = unsafe {
        lc_knearest_d2(
            packed.as_ptr(),
            2,
            &box_c,
            1,
            std::ptr::null(),
            0.0,
            nn.as_mut_ptr(),
            d2.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(nn, [1, 0]);
    assert!((d2[0] - 0.64).abs() < 1e-12);
}

#[test]
fn lc_knearest_many_two_frames() {
    let packed = [0.2, 0.0, 0.0, 9.4, 0.0, 0.0, 0.2, 0.0, 0.0, 9.4, 0.0, 0.0];
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let mut nn = [-1i32; 4];
    let mut d2 = [0.0f64; 4];
    let rc = unsafe {
        lc_knearest_many(
            packed.as_ptr(),
            2,
            2,
            &box_c,
            1,
            std::ptr::null(),
            0.0,
            nn.as_mut_ptr(),
            d2.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(nn, [1, 0, 1, 0]);
    assert!((d2[3] - 0.64).abs() < 1e-12);
}

#[test]
fn lc_pairs_within_queries_then_writes_the_shift() {
    let packed = [0.2, 0.0, 0.0, 9.4, 0.0, 0.0];
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let mut count = 0usize;
    let rc = unsafe {
        lc_pairs_within(
            packed.as_ptr(),
            2,
            &box_c,
            1.0,
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
    assert_eq!(rc, 0);
    assert_eq!(count, 2);
    let mut ii = [0i32; 1];
    let mut jj = [0i32; 1];
    let mut shift = [0i32; 3];
    let mut d2 = [0.0f64; 1];
    let short = unsafe {
        lc_pairs_within(
            packed.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            shift.as_mut_ptr(),
            d2.as_mut_ptr(),
            1,
            &mut count,
        )
    };
    assert_ne!(short, 0);
    assert_eq!(count, 2);
    let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
    assert_eq!(
        msg.to_str().unwrap(),
        "pair buffer is shorter than the pair count"
    );
    let mut ii = [0i32; 2];
    let mut jj = [0i32; 2];
    let mut shift = [0i32; 6];
    let mut d2 = [0.0f64; 2];
    let rc = unsafe {
        lc_pairs_within(
            packed.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            shift.as_mut_ptr(),
            d2.as_mut_ptr(),
            2,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 2);
    let row = ii.iter().position(|&i| i == 0).unwrap();
    assert_eq!(jj[row], 1);
    assert_eq!(&shift[3 * row..3 * row + 3], &[-1, 0, 0]);
    assert!((d2[row] - 0.64).abs() < 1e-12);
}

fn lattice_xyz(side: usize, boxl: f64) -> Vec<f64> {
    let mut xyz = Vec::with_capacity(side * side * side * 3);
    for iz in 0..side {
        for iy in 0..side {
            for ix in 0..side {
                xyz.push((ix as f64 + 0.5) * boxl / side as f64);
                xyz.push((iy as f64 + 0.5) * boxl / side as f64);
                xyz.push((iz as f64 + 0.5) * boxl / side as f64);
            }
        }
    }
    xyz
}

fn c_rows(xyz: &[f64], box_c: &LcCell, cutoff: f64, half: c_int) -> Vec<(i32, i32, [i32; 3], u64)> {
    let n = xyz.len() / 3;
    let mut count = 0usize;
    let null = std::ptr::null_mut();
    let rc = unsafe {
        lc_pairs_within(
            xyz.as_ptr(),
            n,
            box_c,
            cutoff,
            std::ptr::null(),
            0.0,
            half,
            null,
            null,
            null,
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    let mut ii = vec![0i32; count];
    let mut jj = vec![0i32; count];
    let mut ss = vec![0i32; 3 * count];
    let mut dd = vec![0.0f64; count];
    let rc = unsafe {
        lc_pairs_within(
            xyz.as_ptr(),
            n,
            box_c,
            cutoff,
            std::ptr::null(),
            0.0,
            half,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            ss.as_mut_ptr(),
            dd.as_mut_ptr(),
            count,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    let mut rows: Vec<_> = (0..count)
        .map(|t| {
            (
                ii[t],
                jj[t],
                [ss[3 * t], ss[3 * t + 1], ss[3 * t + 2]],
                dd[t].to_bits(),
            )
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn c_rows_match_the_rust_list() {
    let xyz = lattice_xyz(9, 18.0);
    let pts: Vec<[f64; 3]> = xyz.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let cell = linkcell::Cell::ortho(18.0, 18.0, 18.0).unwrap();
    let box_c = ortho_c(18.0, 18.0, 18.0);
    for half in [false, true] {
        let mut want: Vec<_> = linkcell::pairs_within(&pts, &cell, 4.0, None, None, half)
            .unwrap()
            .iter()
            .map(|p| (p.i as i32, p.j as i32, p.shift, p.dist2.to_bits()))
            .collect();
        want.sort();
        assert_eq!(
            c_rows(&xyz, &box_c, 4.0, half as c_int),
            want,
            "half={half}"
        );

        let mut count = 0usize;
        let rc = unsafe {
            lc_pairs_within_rows(
                xyz.as_ptr(),
                pts.len(),
                &box_c,
                4.0,
                std::ptr::null(),
                0.0,
                half as c_int,
                std::ptr::null_mut(),
                0,
                &mut count,
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(count, want.len());
        let mut rows = vec![linkcell::lc_pair::default(); count];
        let rc = unsafe {
            lc_pairs_within_rows(
                xyz.as_ptr(),
                pts.len(),
                &box_c,
                4.0,
                std::ptr::null(),
                0.0,
                half as c_int,
                rows.as_mut_ptr(),
                count,
                &mut count,
            )
        };
        assert_eq!(rc, 0);
        let mut got: Vec<_> = rows
            .iter()
            .map(|r| (r.i, r.j, r.shift, r.dist2.to_bits()))
            .collect();
        got.sort();
        assert_eq!(got, want, "rows half={half}");
    }
}

#[test]
fn a_parked_query_is_not_reused_for_other_points() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let near = [0.2, 0.0, 0.0, 9.4, 0.0, 0.0];
    let far = [0.2, 0.0, 0.0, 5.0, 0.0, 0.0];
    let mut count = 0usize;
    let null = std::ptr::null_mut();
    let rc = unsafe {
        lc_pairs_within(
            near.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            null,
            null,
            null,
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 2);
    // Same buffers, different points: the fill must search again.
    let mut ii = [7i32; 2];
    let mut jj = [7i32; 2];
    let mut ss = [7i32; 6];
    let mut dd = [7.0f64; 2];
    let rc = unsafe {
        lc_pairs_within(
            far.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            ss.as_mut_ptr(),
            dd.as_mut_ptr(),
            2,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 0);
    assert_eq!(ii, [7, 7]);
    // A fill with the queried points still writes both rows.
    let rc = unsafe {
        lc_pairs_within(
            near.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            ss.as_mut_ptr(),
            dd.as_mut_ptr(),
            2,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 2);
    let mut got = [(ii[0], jj[0]), (ii[1], jj[1])];
    got.sort();
    assert_eq!(got, [(0, 1), (1, 0)]);
}

#[test]
fn c_pair_layouts_keep_the_input_image_shift() {
    let cell = ortho_c(10.0, 10.0, 10.0);
    let xyz = [0.2, 0.0, 0.0, 29.4, 0.0, 0.0];
    for half in [0, 1] {
        let columns = c_rows(&xyz, &cell, 1.0, half);
        let mut count = 0;
        let mut rows = [linkcell::lc_pair::default(); 2];
        let rc = unsafe {
            lc_pairs_within_rows(
                xyz.as_ptr(),
                2,
                &cell,
                1.0,
                std::ptr::null(),
                0.0,
                half,
                rows.as_mut_ptr(),
                rows.len(),
                &mut count,
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(count, if half == 0 { 2 } else { 1 });
        let mut got: Vec<_> = rows[..count]
            .iter()
            .map(|r| (r.i, r.j, r.shift, r.dist2.to_bits()))
            .collect();
        got.sort();
        assert_eq!(got, columns);
        assert_eq!((got[0].0, got[0].1, got[0].2), (0, 1, [-3, 0, 0]));
        if half == 0 {
            assert_eq!((got[1].0, got[1].1, got[1].2), (1, 0, [3, 0, 0]));
        }
        for r in &got {
            assert!((f64::from_bits(r.3) - 0.64).abs() < 1e-12);
        }
    }
}

extern "C" fn count_hook(ctx: *mut std::ffi::c_void) {
    // SAFETY: the test passes a pointer to an AtomicUsize that outlives every search.
    let hits = unsafe { &*(ctx as *const std::sync::atomic::AtomicUsize) };
    hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// A continuation set on one thread runs only at the end of that thread's
/// own searches, never inside a search started on another thread.
#[test]
fn continuation_runs_only_on_the_thread_that_set_it() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    const SEARCHES: usize = 200;
    let xyz: Vec<[f64; 3]> = (0..512)
        .map(|i| {
            let i = i as f64;
            [(i * 0.37) % 8.0, (i * 0.61) % 8.0, (i * 0.83) % 8.0]
        })
        .collect();
    let packed = pack_xyz(&xyz);
    let hits: Vec<AtomicUsize> = (0..2).map(|_| AtomicUsize::new(0)).collect();
    std::thread::scope(|s| {
        for (t, slot) in hits.iter().enumerate() {
            let packed = &packed;
            s.spawn(move || {
                let box_c = ortho_c(8.0, 8.0, 8.0);
                let ctx = slot as *const AtomicUsize as *mut std::ffi::c_void;
                linkcell::lc_set_continuation(ctx, Some(count_hook));
                let mut out = vec![-1; 512 * 4];
                for _ in 0..SEARCHES {
                    let rc = unsafe {
                        lc_knearest(
                            packed.as_ptr(),
                            512,
                            &box_c,
                            4,
                            std::ptr::null(),
                            0.0,
                            out.as_mut_ptr(),
                        )
                    };
                    assert_eq!(rc, 0, "thread {t}");
                }
                linkcell::lc_set_continuation(std::ptr::null_mut(), None);
            });
        }
        // A third thread searches with no hook of its own.
        let packed = &packed;
        s.spawn(move || {
            let box_c = ortho_c(8.0, 8.0, 8.0);
            let mut out = vec![-1; 512 * 4];
            for _ in 0..SEARCHES {
                let rc = unsafe {
                    lc_knearest(
                        packed.as_ptr(),
                        512,
                        &box_c,
                        4,
                        std::ptr::null(),
                        0.0,
                        out.as_mut_ptr(),
                    )
                };
                assert_eq!(rc, 0, "hookless thread");
            }
        });
    });
    for (t, slot) in hits.iter().enumerate() {
        assert_eq!(
            slot.load(Ordering::SeqCst),
            SEARCHES,
            "thread {t} hook count"
        );
    }
}

fn last_error() -> String {
    let msg = linkcell::lc_last_error();
    assert!(!msg.is_null());
    unsafe { std::ffi::CStr::from_ptr(msg) }
        .to_str()
        .unwrap()
        .to_owned()
}

/// Three points with `bad` at axis `axis` of point 1, packed.
fn poisoned_c(bad: f64, axis: usize) -> [f64; 9] {
    let mut packed = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
    packed[3 + axis] = bad;
    packed
}

const NON_FINITE: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

#[test]
fn lc_knearest_entries_reject_non_finite_coordinates() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let want = Error::NonFinite.to_string();
    for bad in NON_FINITE {
        for axis in 0..3 {
            let packed = poisoned_c(bad, axis);
            let mut nn = [7i32; 6];
            let rc = unsafe {
                lc_knearest(
                    packed.as_ptr(),
                    3,
                    &box_c,
                    2,
                    std::ptr::null(),
                    0.0,
                    nn.as_mut_ptr(),
                )
            };
            assert_ne!(rc, 0);
            assert_eq!(last_error(), want);
            assert_eq!(nn, [-1; 6]);

            let mut nn = [7i32; 6];
            let mut d2 = [1.0f64; 6];
            let rc = unsafe {
                lc_knearest_d2(
                    packed.as_ptr(),
                    3,
                    &box_c,
                    2,
                    std::ptr::null(),
                    0.0,
                    nn.as_mut_ptr(),
                    d2.as_mut_ptr(),
                )
            };
            assert_ne!(rc, 0);
            assert_eq!(last_error(), want);
            assert_eq!(nn, [-1; 6]);
            assert!(d2.iter().all(|v| v.is_nan()));
        }
    }
}

#[test]
fn lc_knearest_many_rejects_a_non_finite_later_frame() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    for bad in NON_FINITE {
        let mut packed = [0.0f64; 18];
        packed[..9].copy_from_slice(&poisoned_c(0.0, 0));
        packed[9..].copy_from_slice(&poisoned_c(bad, 2));
        let mut nn = [7i32; 12];
        let mut d2 = [1.0f64; 12];
        let rc = unsafe {
            lc_knearest_many(
                packed.as_ptr(),
                3,
                2,
                &box_c,
                2,
                std::ptr::null(),
                0.0,
                nn.as_mut_ptr(),
                d2.as_mut_ptr(),
            )
        };
        assert_ne!(rc, 0);
        assert_eq!(last_error(), Error::NonFinite.to_string());
        assert_eq!(nn, [-1; 12]);
        assert!(d2.iter().all(|v| v.is_nan()));
    }
}

extern "C" fn csr_alloc(ctx: *mut std::ffi::c_void, total: usize) -> *mut c_int {
    let cols = unsafe { &mut *(ctx as *mut Vec<c_int>) };
    cols.resize(total, 0);
    cols.as_mut_ptr()
}

fn c_csr(packed: &[f64], n: usize, box_c: &linkcell::lc_cell, mutual: c_int) -> c_int {
    let mut offsets = vec![0 as c_int; n + 1];
    let mut cols: Vec<c_int> = Vec::new();
    unsafe {
        linkcell::lc_knearest_csr(
            packed.as_ptr(),
            n,
            box_c,
            2,
            std::ptr::null(),
            0.0,
            mutual,
            offsets.as_mut_ptr(),
            Some(csr_alloc),
            (&mut cols as *mut Vec<c_int>).cast(),
        )
    }
}

fn ortho_lc(l: f64) -> linkcell::lc_cell {
    linkcell::lc_cell {
        ax: l,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: l,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: l,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    }
}

#[test]
fn lc_knearest_csr_rejects_non_finite_coordinates() {
    let box_c = ortho_lc(10.0);
    assert_eq!(c_csr(&poisoned_c(0.0, 0), 3, &box_c, 0), 0);
    for bad in NON_FINITE {
        for axis in 0..3 {
            for mutual in [0, 1] {
                assert_ne!(c_csr(&poisoned_c(bad, axis), 3, &box_c, mutual), 0);
                assert_eq!(last_error(), Error::NonFinite.to_string());
            }
        }
    }
}

#[test]
fn lc_knearest_entries_reject_a_non_finite_origin() {
    let packed = poisoned_c(0.0, 0);
    for bad in NON_FINITE {
        let mut box_c = ortho_lc(10.0);
        box_c.oy = bad;
        let mut nn = [7i32; 6];
        let rc = unsafe {
            linkcell::lc_knearest(
                packed.as_ptr(),
                3,
                &box_c,
                2,
                std::ptr::null(),
                0.0,
                nn.as_mut_ptr(),
            )
        };
        assert_ne!(rc, 0);
        assert_eq!(last_error(), Error::NonFinite.to_string());
        assert_ne!(c_csr(&packed, 3, &box_c, 0), 0);
        assert_eq!(last_error(), Error::NonFinite.to_string());
    }
}

#[test]
fn lc_knearest_entries_reject_non_finite_box_lengths() {
    let packed = poisoned_c(0.0, 0);
    let want = Error::BadBox.to_string();
    for bad in NON_FINITE {
        for slot in 0..9 {
            let mut box_c = ortho_lc(10.0);
            match slot {
                0 => box_c.ax = bad,
                1 => box_c.ay = bad,
                2 => box_c.az = bad,
                3 => box_c.bx = bad,
                4 => box_c.by = bad,
                5 => box_c.bz = bad,
                6 => box_c.cx = bad,
                7 => box_c.cy = bad,
                _ => box_c.cz = bad,
            }
            let mut nn = [7i32; 6];
            let rc = unsafe {
                linkcell::lc_knearest(
                    packed.as_ptr(),
                    3,
                    &box_c,
                    2,
                    std::ptr::null(),
                    0.0,
                    nn.as_mut_ptr(),
                )
            };
            assert_ne!(rc, 0);
            assert_eq!(last_error(), want);
            assert_ne!(c_csr(&packed, 3, &box_c, 0), 0);
            assert_eq!(last_error(), want);
        }
    }
}

fn build_nd_index(data: &[f64], n: usize, d: usize, periods: Option<&[f64]>) -> *mut LcNdIndex {
    let mut out = std::ptr::null_mut();
    let rc = unsafe {
        lc_nd_build(
            data.as_ptr(),
            n,
            d,
            periods.map_or(std::ptr::null(), |p| p.as_ptr()),
            &mut out,
        )
    };
    assert_eq!(rc, 0, "build failed: {}", last_error());
    out
}

#[test]
fn lc_nd_build_and_queries_work() {
    let data = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let idx = build_nd_index(&data, 3, 2, None);
    assert_eq!(unsafe { lc_nd_len(idx) }, 3);
    assert_eq!(unsafe { lc_nd_dims(idx) }, 2);

    let mut nn = [-1; 5];
    let mut d2 = [f64::NAN; 5];
    let q = [0.0, 0.0];
    let rc = unsafe { lc_nd_knn(idx, q.as_ptr(), 5, nn.as_mut_ptr(), d2.as_mut_ptr()) };
    assert_eq!(rc, 0);
    assert_eq!(nn[0], 0);
    assert!(d2[0].abs() <= 1.0e-14);

    let mut need = 0usize;
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            1.01,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut need,
        )
    };
    assert_eq!(rc, 0);
    let mut out_i = vec![-1; need];
    let mut out_d2 = vec![0.0; need];
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            1.01,
            out_i.as_mut_ptr(),
            out_d2.as_mut_ptr(),
            need,
            &mut need,
        )
    };
    assert_eq!(rc, 0);
    assert!(out_i.contains(&0));

    let mut labels = vec![-9; 3];
    let mut core = vec![0; 3];
    let mut clusters = 0usize;
    let rc = unsafe {
        lc_nd_dbscan(
            idx,
            1.1,
            2,
            labels.as_mut_ptr(),
            core.as_mut_ptr(),
            &mut clusters,
        )
    };
    assert_eq!(rc, 0);
    assert!(clusters >= 1);

    unsafe { lc_nd_free(idx) };
}

#[test]
fn lc_nd_rejects_bad_dimensions_nulls_and_bad_periods() {
    let data = [0.0, 1.0];
    let periods = [1.0];
    let mut out = std::ptr::null_mut();
    assert_ne!(
        unsafe { lc_nd_build(std::ptr::null(), 1, 1, periods.as_ptr(), &mut out) },
        0
    );
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), 1, 0, periods.as_ptr(), &mut out) },
        0
    );
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), 1, 1, periods.as_ptr(), std::ptr::null_mut()) },
        0
    );
    let bad_neg = [-1.0f64];
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), 1, 1, bad_neg.as_ptr(), &mut out) },
        0
    );
    let bad_nan = [f64::NAN];
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), 1, 1, bad_nan.as_ptr(), &mut out) },
        0
    );
}

#[test]
fn lc_nd_count_queries_report_short_buffers() {
    let data = [
        0.0, 0.0, //
        0.1, 0.0, //
        0.0, 0.1, //
        0.09, 0.09,
    ];
    let idx = build_nd_index(&data, 4, 2, None);
    let q = [0.0, 0.0];
    let mut need = 0usize;
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            0.2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut need,
        )
    };
    assert_eq!(rc, 0);
    let mut got = need;
    let mut short_i = vec![-1; need.saturating_sub(1)];
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            0.2,
            short_i.as_mut_ptr(),
            std::ptr::null_mut(),
            short_i.len(),
            &mut got,
        )
    };
    assert_ne!(rc, 0);
    assert_eq!(got, need);

    let mut need_pairs = 0usize;
    let rc = unsafe {
        lc_nd_pairs_within(
            idx,
            0.2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut need_pairs,
        )
    };
    assert_eq!(rc, 0);
    let mut out_i = vec![-1; need_pairs.saturating_sub(1)];
    let mut out_j = vec![-1; need_pairs.saturating_sub(1)];
    let mut got_pairs = need_pairs;
    let rc = unsafe {
        lc_nd_pairs_within(
            idx,
            0.2,
            out_i.as_mut_ptr(),
            out_j.as_mut_ptr(),
            std::ptr::null_mut(),
            out_i.len(),
            &mut got_pairs,
        )
    };
    assert_ne!(rc, 0);
    assert_eq!(got_pairs, need_pairs);
    unsafe { lc_nd_free(idx) };
}

#[test]
fn lc_nd_rejects_non_finite_coordinates_and_null_query_pointers() {
    let nan_data = [f64::NAN, 0.0];
    let mut out = std::ptr::null_mut();
    assert_ne!(
        unsafe { lc_nd_build(nan_data.as_ptr(), 1, 2, std::ptr::null(), &mut out) },
        0
    );
    assert_eq!(last_error(), Error::NonFinite.to_string());

    let good_data = [0.0, 0.0, 1.0, 0.0];
    let idx = build_nd_index(&good_data, 2, 2, None);
    let mut nn = [0; 2];
    let q = [0.0, 0.0];
    assert_ne!(
        unsafe {
            lc_nd_knn(
                std::ptr::null(),
                q.as_ptr(),
                2,
                nn.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lc_nd_knn(
                idx,
                std::ptr::null(),
                2,
                nn.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    let q_nan = [f64::NAN, 0.0];
    assert_ne!(
        unsafe {
            lc_nd_knn(
                idx,
                q_nan.as_ptr(),
                2,
                nn.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    let mut count = 0usize;
    assert_ne!(
        unsafe {
            lc_nd_within(
                idx,
                std::ptr::null(),
                1.0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                &mut count,
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lc_nd_pairs_within(
                idx,
                1.0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lc_nd_dbscan(
                idx,
                1.0,
                2,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_eq!(unsafe { lc_nd_len(std::ptr::null()) }, 0);
    assert_eq!(unsafe { lc_nd_dims(std::ptr::null()) }, 0);
    unsafe { lc_nd_free(idx) };
}
