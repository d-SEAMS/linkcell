//! Open-cell pairs: occupied Cartesian bins, no periodic padding.

use std::collections::BTreeMap;
use std::sync::Mutex;

use linkcell::{
    cell_for_pbc, lc_cell, lc_pair, pairs_open, pairs_within, pairs_within_pbc, stencil_builds,
    Cell, Error, PbcQuery,
};

static LOCK: Mutex<()> = Mutex::new(());

type Key = (usize, usize, [i32; 3]);

fn xorshift(state: &mut u64) -> u64 {
    *state ^= state.wrapping_shl(13);
    *state ^= state.wrapping_shr(7);
    *state ^= state.wrapping_shl(17);
    *state
}

fn cluster() -> Vec<[f64; 3]> {
    let mut xyz = Vec::with_capacity(27);
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                xyz.push([(ix + 2) as f64, (iy + 2) as f64, (iz + 2) as f64]);
            }
        }
    }
    xyz
}

/// One step of the corner atom. Seed 42 and step 0.25 change the
/// bounding span on 267 of 400 moves.
fn nudge(xyz: &mut [[f64; 3]], state: &mut u64) {
    let step = 0.25;
    for coord in &mut xyz[26] {
        let draw = xorshift(state);
        let unit = (draw % 10001) as f64 / 10000.0;
        *coord += unit * 2.0 * step - step;
    }
}

fn padded_box(xyz: &[[f64; 3]], cutoff: f64) -> Cell {
    let mut length = [0.0; 3];
    for axis in 0..3 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for point in xyz {
            lo = lo.min(point[axis]);
            hi = hi.max(point[axis]);
        }
        let span = if lo.is_finite() {
            (hi - lo).max(0.0)
        } else {
            0.0
        };
        length[axis] = 2.0 * span + 4.0 * cutoff + 10.0;
    }
    Cell::ortho(length[0], length[1], length[2]).unwrap()
}

fn map_of(rows: &[linkcell::Pair]) -> BTreeMap<Key, f64> {
    rows.iter()
        .map(|pair| ((pair.i, pair.j, pair.shift), pair.dist2))
        .collect()
}

fn brute_open(xyz: &[[f64; 3]], cutoff: f64, half: bool) -> BTreeMap<Key, f64> {
    let mut found = BTreeMap::new();
    let limit = cutoff * cutoff;
    for i in 0..xyz.len() {
        for j in 0..xyz.len() {
            if i == j || (half && i > j) {
                continue;
            }
            let dx = xyz[j][0] - xyz[i][0];
            let dy = xyz[j][1] - xyz[i][1];
            let dz = xyz[j][2] - xyz[i][2];
            let dist2 = dx * dx + dy * dy + dz * dz;
            if dist2 < limit {
                found.insert((i, j, [0, 0, 0]), dist2);
            }
        }
    }
    found
}

fn assert_same(rows: &[linkcell::Pair], want: &BTreeMap<Key, f64>) {
    let got = map_of(rows);
    assert_eq!(got.len(), rows.len(), "duplicate rows");
    assert_eq!(got, *want);
}

#[test]
fn four_hundred_moves_keep_the_open_stencil() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let cutoff = 2.5;
    let hint = Some(8.0);
    let mut xyz = cluster();
    let box0 = padded_box(&xyz, cutoff);
    pairs_within(&xyz, &box0, cutoff, None, hint, true).unwrap();
    let before = stencil_builds();
    let mut state = 42u64;
    for _ in 0..400 {
        nudge(&mut xyz, &mut state);
        let widened = padded_box(&xyz, cutoff);
        pairs_within(&xyz, &widened, cutoff, None, hint, true).unwrap();
    }
    let padded = stencil_builds() - before;
    assert_eq!(
        padded, 267,
        "padding each open axis rebuilds the stencil when the span changes"
    );

    let mut xyz = cluster();
    let first = pairs_open(&xyz, cutoff, hint).unwrap();
    assert_same(&first, &brute_open(&xyz, cutoff, true));
    let before = stencil_builds();
    let mut state = 42u64;
    for _ in 0..400 {
        nudge(&mut xyz, &mut state);
        let rows = pairs_open(&xyz, cutoff, hint).unwrap();
        assert_same(&rows, &brute_open(&xyz, cutoff, true));
    }
    assert_eq!(
        stencil_builds() - before,
        0,
        "the walk stays inside the Cartesian bin of edge 8, so the bounding bins do not change"
    );
}

#[test]
fn open_reach_past_the_image_cast_is_too_many_images() {
    let err = pairs_open(
        &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        2_147_483_646.0,
        Some(1.0),
    )
    .unwrap_err();
    assert_eq!(err, Error::TooManyImages);
}

#[test]
fn crossing_a_bin_face_keeps_the_open_stencil() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let mut xyz = cluster();
    pairs_open(&xyz, 2.5, Some(8.0)).unwrap();
    let before = stencil_builds();
    xyz[26][0] += 20.0;
    let rows = pairs_open(&xyz, 2.5, Some(8.0)).unwrap();
    assert_same(&rows, &brute_open(&xyz, 2.5, true));
    assert_eq!(stencil_builds() - before, 0);
}

/// Band images of one molecule sit in different bins and span different
/// bin counts. Visiting them in turn is the contact check of a band step.
#[test]
fn alternating_open_frames_share_one_stencil() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let cutoff = 1.5;
    let near = cluster();
    let mut far: Vec<[f64; 3]> = near
        .iter()
        .map(|p| [1.3 * p[0] + 7.1, 0.9 * p[1] - 3.4, 1.1 * p[2] + 11.6])
        .collect();
    far[5][1] -= 2.2;
    let frames = [near, far];
    pairs_open(&frames[0], cutoff, None).unwrap();
    let before = stencil_builds();
    for round in 0..20 {
        let xyz = &frames[round % 2];
        let rows = pairs_open(xyz, cutoff, None).unwrap();
        assert_same(&rows, &brute_open(xyz, cutoff, true));
    }
    assert_eq!(
        stencil_builds() - before,
        0,
        "an open stencil depends on the bin edge and the cutoff, not on the occupied bins"
    );
}

#[test]
fn randomized_clusters_match_brute_force_and_a_large_box() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let mut state = 0x1234_5678_9abc_def0u64;
    for trial in 0..12 {
        let n = 5 + (xorshift(&mut state) % 20) as usize;
        let mut xyz = Vec::with_capacity(n);
        for _ in 0..n {
            xyz.push([
                20.0 + (xorshift(&mut state) % 10_000) as f64 / 1000.0,
                20.0 + (xorshift(&mut state) % 10_000) as f64 / 1000.0,
                20.0 + (xorshift(&mut state) % 10_000) as f64 / 1000.0,
            ]);
        }
        let cutoff = 1.2 + (trial % 4) as f64 * 0.4;
        let hint = if trial % 2 == 0 { None } else { Some(1.5) };
        for half in [true, false] {
            let query = PbcQuery::open(cutoff).half(half).cell_hint(hint);
            let dummy = Cell::ortho(1.0, 1.0, 1.0).unwrap();
            let rows = pairs_within_pbc(&xyz, &dummy, &query).unwrap();
            assert_same(&rows, &brute_open(&xyz, cutoff, half));
            if half {
                let big = Cell::ortho(80.0, 80.0, 80.0).unwrap();
                let periodic = pairs_within(&xyz, &big, cutoff, None, hint, true).unwrap();
                let open_pairs: BTreeMap<_, _> = map_of(&rows);
                let periodic_pairs = map_of(&periodic);
                assert!(periodic_pairs.keys().all(|key| key.2 == [0, 0, 0]));
                assert_eq!(
                    open_pairs.keys().collect::<Vec<_>>(),
                    periodic_pairs.keys().collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn mixed_slab_matches_an_image_scan() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let cell = Cell::ortho(5.0, 6.0, 8.0).unwrap();
    let periodic = [true, true, false];
    let mut state = 7u64;
    let mut xyz = Vec::new();
    for _ in 0..10 {
        xyz.push([
            (xorshift(&mut state) % 5000) as f64 / 1000.0,
            (xorshift(&mut state) % 6000) as f64 / 1000.0,
            1.0 + (xorshift(&mut state) % 4000) as f64 / 1000.0,
        ]);
    }
    let cutoff = 2.2;
    let query = PbcQuery::new(periodic, cutoff).half(true);
    let rows = pairs_within_pbc(&xyz, &cell, &query).unwrap();
    let mut want = BTreeMap::new();
    let limit = cutoff * cutoff;
    let span = [1i32, 1, 0];
    for i in 0..xyz.len() {
        for j in 0..xyz.len() {
            if i > j {
                continue;
            }
            for sa in -span[0]..=span[0] {
                for sb in -span[1]..=span[1] {
                    let shift = [sa, sb, 0];
                    if i == j && shift == [0, 0, 0] {
                        continue;
                    }
                    if i == j && shift[0] > 0 {
                        continue;
                    }
                    if i == j && shift[0] == 0 && shift[1] > 0 {
                        continue;
                    }
                    let delta = cell.lattice_shift(sa, sb, 0);
                    let dx = xyz[j][0] + delta[0] - xyz[i][0];
                    let dy = xyz[j][1] + delta[1] - xyz[i][1];
                    let dz = xyz[j][2] + delta[2] - xyz[i][2];
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    if dist2 < limit {
                        want.insert((i, j, shift), dist2);
                    }
                }
            }
        }
    }
    assert_same(&rows, &want);
    assert!(!want.is_empty());
}

#[test]
fn closed_cutoff_keeps_the_boundary() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let xyz = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]];
    let cell = Cell::ortho(1.0, 1.0, 1.0).unwrap();
    let strict = pairs_within_pbc(&xyz, &cell, &PbcQuery::open(2.5).half(true)).unwrap();
    assert!(strict.is_empty());
    let closed =
        pairs_within_pbc(&xyz, &cell, &PbcQuery::open(2.5).half(true).closed(true)).unwrap();
    assert_eq!(closed.len(), 1);
    assert_eq!(closed[0].dist2, 6.25);
}

#[test]
fn mask_drops_a_point() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.0, 0.0]];
    let cell = Cell::ortho(1.0, 1.0, 1.0).unwrap();
    let query = PbcQuery::open(2.0)
        .half(true)
        .mask(Some(&[true, false, true]));
    let rows = pairs_within_pbc(&xyz, &cell, &query).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!((rows[0].i, rows[0].j), (0, 2));
}

#[test]
fn zero_open_vector_still_bins() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let rows = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]];
    let cell = cell_for_pbc(rows, [0.0, 0.0, 0.0], [true, true, false]).unwrap();
    let xyz = [[0.2, 0.2, 1.0], [3.9, 0.3, 1.2], [0.4, 0.2, 3.0]];
    let query = PbcQuery::new([true, true, false], 2.0).half(true);
    let found = pairs_within_pbc(&xyz, &cell, &query).unwrap();
    assert!(!found.is_empty());
    assert!(found.iter().all(|pair| pair.shift[2] == 0));
}

extern "C" {
    fn lc_pairs_open(
        xyz: *const f64,
        n: usize,
        cutoff: f64,
        cell_hint: f64,
        out: *mut lc_pair,
        cap: usize,
        out_count: *mut usize,
    ) -> i32;
    fn lc_stencil_builds() -> u64;
    fn lc_pairs_within_pbc(
        xyz: *const f64,
        n: usize,
        simbox: *const lc_cell,
        pbc: *const i32,
        cutoff: f64,
        mask: *const i32,
        cell_hint: f64,
        half: i32,
        closed: i32,
        exclude_self_images: i32,
        out: *mut lc_pair,
        cap: usize,
        out_count: *mut usize,
    ) -> i32;
}

#[test]
fn c_api_open_list_matches_the_rust_rows() {
    let _guard = LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let xyz = cluster();
    let flat: Vec<f64> = xyz.iter().flat_map(|p| *p).collect();
    let want = pairs_open(&xyz, 2.5, Some(8.0)).unwrap();
    let mut count = 0usize;
    let query = unsafe {
        lc_pairs_open(
            flat.as_ptr(),
            xyz.len(),
            2.5,
            8.0,
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(query, 0);
    assert_eq!(count, want.len());
    let mut out = vec![lc_pair::default(); count];
    let mut wrote = 0usize;
    let fill = unsafe {
        lc_pairs_open(
            flat.as_ptr(),
            xyz.len(),
            2.5,
            8.0,
            out.as_mut_ptr(),
            count,
            &mut wrote,
        )
    };
    assert_eq!(fill, 0);
    assert_eq!(wrote, want.len());
    for (got, row) in out.iter().zip(&want) {
        assert_eq!(got.i as usize, row.i);
        assert_eq!(got.j as usize, row.j);
        assert_eq!(got.shift, row.shift);
        assert_eq!(got.dist2, row.dist2);
    }
    let builds = unsafe { lc_stencil_builds() };
    assert!(builds >= stencil_builds());
    let _ = builds;

    let cell = Cell::ortho(9.0, 9.0, 9.0).unwrap();
    let raw = lc_cell {
        ax: 9.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 9.0,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 9.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let pbc = [1, 1, 0];
    let mut count = 0usize;
    let status = unsafe {
        lc_pairs_within_pbc(
            flat.as_ptr(),
            xyz.len(),
            &raw,
            pbc.as_ptr(),
            2.5,
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
    let rust = pairs_within_pbc(
        &xyz,
        &cell,
        &PbcQuery::new([true, true, false], 2.5).half(true),
    )
    .unwrap();
    assert_eq!(count, rust.len());
}
