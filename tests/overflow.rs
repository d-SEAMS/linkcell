//! Overflow conditions named on the public entry points.
//!
//! A point count above `i32::MAX` or `u32::MAX`, a CSR edge count above
//! `i32::MAX`, and a fused-team `i64` conversion of that point count are
//! not allocated here.

use linkcell::{
    cell_graph, halo_width, knearest, knearest_csr, knearest_into, knearest_into_d2,
    knearest_into_many, pairs_open, pairs_within, pairs_within_columns, pairs_within_count,
    pairs_within_into, pairs_within_pbc, partition, Cell, Error, PairColumns, PartRequest,
    PbcQuery, SkinList,
};

fn unit() -> Cell {
    Cell::ortho(1.0, 1.0, 1.0).unwrap()
}

fn assert_overflow<T>(result: Result<T, Error>) {
    assert_eq!(result.err(), Some(Error::Overflow));
}

fn span_points() -> [[f64; 3]; 2] {
    [[0.0; 3], [i32::MAX as f64, 0.0, 0.0]]
}

fn past_i32() -> [[f64; 3]; 2] {
    [[0.0; 3], [i32::MAX as f64 + 1.0, 0.0, 0.0]]
}

#[test]
fn relative_image_past_i32_is_overflow() {
    let cell = unit();
    let xyz = past_i32();
    assert_overflow(pairs_within(&xyz, &cell, 0.4, None, None, false));
    assert_overflow(pairs_within_count(&xyz, &cell, 0.4, None, None, false));
    let mut rows = Vec::new();
    assert_overflow(pairs_within_into(
        &xyz, &cell, 0.4, None, None, false, &mut rows,
    ));
    let mut cols = PairColumns::default();
    assert_overflow(pairs_within_columns(
        &xyz, &cell, 0.4, None, None, false, &mut cols,
    ));
    let query = PbcQuery::periodic(0.4);
    assert_overflow(pairs_within_pbc(&xyz, &cell, &query));
}

#[test]
fn non_finite_relative_image_is_overflow() {
    let cell = unit();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let xyz = [[0.0; 3], [bad, 0.0, 0.0]];
        assert_overflow(pairs_within(&xyz, &cell, 0.4, None, None, true));
        let mut cols = PairColumns::default();
        assert_overflow(pairs_within_columns(
            &xyz, &cell, 0.4, None, None, true, &mut cols,
        ));
    }
}

#[test]
fn image_span_plus_repeats_exceeds_i32() {
    let cell = unit();
    let xyz = span_points();
    assert_overflow(pairs_within(&xyz, &cell, 0.5, None, None, false));
    assert_overflow(pairs_within_count(&xyz, &cell, 0.5, None, None, true));
    let mut rows = Vec::new();
    assert_overflow(pairs_within_into(
        &xyz, &cell, 0.5, None, None, false, &mut rows,
    ));
    let mut cols = PairColumns::default();
    assert_overflow(pairs_within_columns(
        &xyz, &cell, 0.5, None, None, false, &mut cols,
    ));
    assert_overflow(pairs_within_pbc(&xyz, &cell, &PbcQuery::periodic(0.5)));
    assert_overflow(cell_graph(&xyz, &cell, 0.5, None, None));
    assert_overflow(partition(&xyz, &cell, 0.5, None, None, PartRequest::new(2)));
    let mut skin = SkinList::new(cell, 0.4, 0.0, true).unwrap();
    assert_overflow(skin.update(&xyz, false));
    let kept = pairs_within(&xyz, &cell, 0.5, Some(&[true, false]), None, false).unwrap();
    assert!(kept.is_empty());
}

#[test]
fn knearest_product_overflows() {
    let cell = unit();
    let xyz = [[0.0; 3], [0.2, 0.0, 0.0]];
    assert_overflow(knearest(&xyz, &cell, usize::MAX, None, None));
    assert_overflow(knearest_csr(&xyz, &cell, usize::MAX, None, None, false));
    assert_eq!(
        knearest_into(&xyz, &cell, usize::MAX, None, None, &mut []),
        Err(Error::BufferSize)
    );
    assert_eq!(
        knearest_into_d2(&xyz, &cell, usize::MAX, None, None, &mut [], None),
        Err(Error::BufferSize)
    );
}

#[test]
fn knearest_many_products_overflow() {
    let cell = unit();
    let mut nn = [0i32; 1];
    assert_overflow(knearest_into_many(
        &[],
        usize::MAX,
        2,
        &cell,
        1,
        None,
        None,
        &mut nn,
        None,
    ));
    let xyz = [[0.0; 3]; 4];
    assert_overflow(knearest_into_many(
        &xyz,
        2,
        2,
        &cell,
        usize::MAX / 2 + 1,
        None,
        None,
        &mut nn,
        None,
    ));
}

#[test]
fn open_bin_floor_outside_i32_is_overflow() {
    for x in [1.0e20, -1.0e20] {
        let xyz = [[x, 0.0, 0.0]];
        assert_overflow(pairs_open(&xyz, 1.0, None));
        let query = PbcQuery::open(1.0);
        let cell = unit();
        assert_overflow(pairs_within_pbc(&xyz, &cell, &query));
    }
}

#[test]
fn mixed_image_shift_outside_i32_is_overflow() {
    let cell = unit();
    let xyz = span_points();
    let query = PbcQuery::new([true, false, false], 1.5);
    assert_overflow(pairs_within_pbc(&xyz, &cell, &query));
}

#[test]
fn halo_width_ratio_overflows() {
    assert_overflow(halo_width(1.0, 1, [1.0e-10, 1.0, 1.0]));
    assert_overflow(halo_width(1.0, 1, [1.0e-300, 1.0, 1.0]));
}

#[cfg(feature = "capi")]
fn ortho_c() -> linkcell::lc_cell {
    linkcell::lc_cell {
        ax: 1.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 1.0,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 1.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    }
}

#[cfg(feature = "capi")]
fn c_overflow(msg: &std::ffi::CStr) {
    assert_eq!(msg.to_str().unwrap(), "n * k overflows");
}

#[cfg(feature = "capi")]
#[test]
fn c_knearest_many_products_overflow() {
    let xyz = [0.0f64; 3];
    let cell = ortho_c();
    let mut out = 0i32;
    let rc = unsafe {
        linkcell::lc_knearest_many(
            xyz.as_ptr(),
            usize::MAX,
            2,
            &cell,
            1,
            std::ptr::null(),
            0.0,
            &mut out,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(rc, 0);
    c_overflow(unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) });
    let rc = unsafe {
        linkcell::lc_knearest_many(
            xyz.as_ptr(),
            2,
            1,
            &cell,
            usize::MAX,
            std::ptr::null(),
            0.0,
            &mut out,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(rc, 0);
    c_overflow(unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) });
}

#[cfg(feature = "capi")]
#[test]
fn c_pairs_index_and_image_overflow() {
    let cell = ortho_c();
    let dummy = [0.0f64; 3];
    let mut count = 0usize;
    let rc = unsafe {
        linkcell::lc_pairs_within(
            dummy.as_ptr(),
            i32::MAX as usize + 1,
            &cell,
            0.5,
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
    assert_ne!(rc, 0);
    c_overflow(unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) });

    let xyz = span_points();
    let flat = [
        xyz[0][0], xyz[0][1], xyz[0][2], xyz[1][0], xyz[1][1], xyz[1][2],
    ];
    let rc = unsafe {
        linkcell::lc_pairs_within(
            flat.as_ptr(),
            2,
            &cell,
            0.5,
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
    assert_ne!(rc, 0);
    c_overflow(unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) });
}

#[cfg(feature = "capi")]
#[test]
fn c_halo_width_overflows() {
    let width = [1.0e-10, 1.0, 1.0];
    let mut out = [0u32; 3];
    let rc = unsafe { linkcell::lc_halo_width(1.0, 1, width.as_ptr(), out.as_mut_ptr()) };
    assert_ne!(rc, 0);
    c_overflow(unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) });
}

#[cfg(feature = "capi")]
#[test]
fn c_csr_product_overflows() {
    let xyz = [0.0f64; 3];
    let cell = ortho_c();
    let mut offsets = [0i32; 3];
    extern "C" fn alloc(_: *mut std::ffi::c_void, _: usize) -> *mut i32 {
        std::ptr::null_mut()
    }
    let rc = unsafe {
        linkcell::lc_knearest_csr(
            xyz.as_ptr(),
            2,
            &cell,
            usize::MAX,
            std::ptr::null(),
            0.0,
            0,
            offsets.as_mut_ptr(),
            Some(alloc),
            std::ptr::null_mut(),
        )
    };
    assert_ne!(rc, 0);
    c_overflow(unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) });
}

#[cfg(feature = "capi")]
#[test]
fn vesin_image_span_overflows() {
    use linkcell::{NeighborCalc, VesinOptions};
    let cell = unit();
    let xyz = span_points();
    let mut calc = NeighborCalc::default();
    let opt = VesinOptions::cpu(0.5, true, 0.0, 1, false, false, false);
    assert_overflow(calc.compute(&xyz, &cell, [true, true, true], opt));
}

#[cfg(feature = "capi")]
#[test]
fn vesin_null_error_message_returns_one() {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct VesinDevice {
        kind: i32,
        device_id: i32,
    }
    #[repr(C)]
    struct VesinNeighborList {
        length: usize,
        device: VesinDevice,
        pairs: *mut [usize; 2],
        shifts: *mut [i32; 3],
        distances: *mut f64,
        vectors: *mut [f64; 3],
        opaque: *mut std::ffi::c_void,
    }
    extern "C" {
        fn lc_vesin_neighbors(
            points: *const [f64; 3],
            n_points: usize,
            box_: *const [[f64; 3]; 3],
            periodic: *const [bool; 3],
            device: VesinDevice,
            options: linkcell::VesinOptions,
            neighbors: *mut VesinNeighborList,
            error_message: *mut *const std::os::raw::c_char,
        ) -> i32;
    }

    let points = [[0.0_f64, 0.0, 0.0]];
    let box_ = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
    let periodic = [true, true, true];
    let device = VesinDevice {
        kind: 1,
        device_id: 0,
    };
    let options = linkcell::VesinOptions::cpu(1.0, false, 0.0, 1, false, false, false);
    let mut list = VesinNeighborList {
        length: 0,
        device: VesinDevice {
            kind: 0,
            device_id: 0,
        },
        pairs: std::ptr::null_mut(),
        shifts: std::ptr::null_mut(),
        distances: std::ptr::null_mut(),
        vectors: std::ptr::null_mut(),
        opaque: std::ptr::null_mut(),
    };
    let rc = unsafe {
        lc_vesin_neighbors(
            points.as_ptr(),
            points.len(),
            &box_,
            &periodic,
            device,
            options,
            &mut list,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, 1);
}

#[test]
fn replica_list_non_finite_centroid_overflows() {
    let cell = unit();
    let xyz = [[0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]];
    assert_overflow(linkcell::ReplicaList::build(&xyz, 1, cell, 0.4, 0.1, true));
}

#[test]
fn open_cell_tiny_edge_quotient_overflows() {
    let xyz = [[1.0e308, 0.0, 0.0]];
    assert_overflow(pairs_open(&xyz, 1.0, Some(1.0e-10)));
    let tiny = Cell::ortho(1.0e-10, 1.0, 1.0).unwrap();
    let query = PbcQuery::new([true, false, false], 1.0e-11);
    assert_overflow(pairs_within_pbc(&xyz, &tiny, &query));
}

#[cfg(any(feature = "scotch", feature = "ptscotch"))]
fn assert_library_i32_overflow(method: linkcell::Method) {
    use linkcell::{partition_graph, CellGraph, CellGrid};
    let heavy = CellGraph {
        grid: CellGrid {
            nx: 1,
            ny: 1,
            nz: 1,
            bin_width: [1.0; 3],
            weight: vec![i32::MAX as u64 + 1],
        },
        offsets: vec![0, 0],
        neighbor: Vec::new(),
        edge_weight: Vec::new(),
    };
    let mut request = PartRequest {
        parts: 1,
        method,
        balance: None,
        device: false,
    };
    assert_overflow(partition_graph(&heavy, request));
    request.parts = i32::MAX as u32 + 1;
    let light = CellGraph {
        grid: CellGrid {
            weight: vec![1],
            ..heavy.grid.clone()
        },
        ..heavy.clone()
    };
    assert_overflow(partition_graph(&light, request));
    let heavy_edge = CellGraph {
        grid: CellGrid {
            weight: vec![1],
            ..heavy.grid
        },
        offsets: vec![0, 1],
        neighbor: vec![0],
        edge_weight: vec![i32::MAX as u64 + 1],
    };
    request.parts = 1;
    assert_overflow(partition_graph(&heavy_edge, request));
}

#[cfg(feature = "scotch")]
#[test]
fn scotch_i32_weights_overflow() {
    assert_library_i32_overflow(linkcell::Method::Scotch);
}

#[cfg(feature = "ptscotch")]
#[test]
fn ptscotch_i32_weights_overflow() {
    assert_library_i32_overflow(linkcell::Method::PtScotch);
}
