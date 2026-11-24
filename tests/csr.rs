//! Cutoff CSR rows: the pair list grouped by source, against brute force,
//! and through the C ABI at several thread widths.

use std::collections::BTreeMap;

use linkcell::{pairs_within_csr, pairs_within_pbc, Cell, Error, PairCsr, PbcQuery};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Water density, about one oxygen per 30 cubic angstroms.
const WATER: f64 = 0.0334;

/// Uniform points at water density in a cell with edge `edge` and the
/// given tilts, in fractional coordinates drawn from `[-0.1, 1.1)` so
/// that some lie outside the primary cell. A cell narrower than twice
/// the cutoff is filled ten times as dense, so it still has rows.
fn frame(edge: f64, tilt: [f64; 3], seed: u64) -> (Vec<[f64; 3]>, [[f64; 3]; 3]) {
    let density = if edge < 7.0 { 10.0 * WATER } else { WATER };
    let n = (density * edge * edge * edge) as usize;
    let h = [
        [edge, 0.0, 0.0],
        [tilt[0] * edge, edge, 0.0],
        [tilt[1] * edge, tilt[2] * edge, edge],
    ];
    let mut next = rng(seed);
    let xyz = (0..n)
        .map(|_| {
            let s = [1.2 * next() - 0.1, 1.2 * next() - 0.1, 1.2 * next() - 0.1];
            std::array::from_fn(|a| s[0] * h[0][a] + s[1] * h[1][a] + s[2] * h[2][a])
        })
        .collect();
    (xyz, h)
}

fn cell_of(h: [[f64; 3]; 3]) -> Cell {
    Cell::from_vectors(h[0], h[1], h[2], [0.0; 3]).unwrap()
}

/// Rows from `pairs_within_pbc`'s full list: no self image, the nearest
/// image of each column, ascending; a half list keeps `j > i`.
fn from_pairs(xyz: &[[f64; 3]], cell: &Cell, query: &PbcQuery<'_>) -> PairCsr {
    let full = PbcQuery::new(query.periodic, query.cutoff)
        .mask(query.mask)
        .cell_hint(query.cell_hint);
    let mut rows = vec![BTreeMap::<i32, f64>::new(); xyz.len()];
    for pair in pairs_within_pbc(xyz, cell, &full).unwrap() {
        if pair.i == pair.j || (query.half && pair.j < pair.i) {
            continue;
        }
        let at = rows[pair.i].entry(pair.j as i32).or_insert(pair.dist2);
        *at = at.min(pair.dist2);
    }
    pack(rows)
}

fn pack(rows: Vec<BTreeMap<i32, f64>>) -> PairCsr {
    let mut out = PairCsr {
        offsets: vec![0],
        ..PairCsr::default()
    };
    for row in rows {
        for (j, d) in row {
            out.cols.push(j);
            out.dist2.push(d);
        }
        out.offsets.push(out.cols.len() as i32);
    }
    out
}

/// Every image within reach on the periodic axes, nearest kept.
fn brute(
    xyz: &[[f64; 3]],
    h: [[f64; 3]; 3],
    periodic: [bool; 3],
    cutoff: f64,
    half: bool,
) -> PairCsr {
    let reach: Vec<i32> = (0..3).map(|a| if periodic[a] { 3 } else { 0 }).collect();
    let cut2 = cutoff * cutoff;
    let mut rows = vec![BTreeMap::<i32, f64>::new(); xyz.len()];
    for i in 0..xyz.len() {
        for j in 0..xyz.len() {
            if i == j || (half && j < i) {
                continue;
            }
            for na in -reach[0]..=reach[0] {
                for nb in -reach[1]..=reach[1] {
                    for nc in -reach[2]..=reach[2] {
                        let d: [f64; 3] = std::array::from_fn(|a| {
                            xyz[j][a] - xyz[i][a]
                                + f64::from(na) * h[0][a]
                                + f64::from(nb) * h[1][a]
                                + f64::from(nc) * h[2][a]
                        });
                        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if d2 < cut2 {
                            let at = rows[i].entry(j as i32).or_insert(d2);
                            *at = at.min(d2);
                        }
                    }
                }
            }
        }
    }
    pack(rows)
}

fn bits(csr: &PairCsr) -> (Vec<i32>, Vec<i32>, Vec<u64>) {
    (
        csr.offsets.clone(),
        csr.cols.clone(),
        csr.dist2.iter().map(|d| d.to_bits()).collect(),
    )
}

fn csr(xyz: &[[f64; 3]], cell: &Cell, query: &PbcQuery<'_>) -> PairCsr {
    let mut out = PairCsr::default();
    pairs_within_csr(xyz, cell, query, true, &mut out).unwrap();
    out
}

fn assert_close(got: &PairCsr, want: &PairCsr, label: &str) {
    assert_eq!(got.offsets, want.offsets, "{label}: offsets");
    assert_eq!(got.cols, want.cols, "{label}: columns");
    for (k, (a, b)) in got.dist2.iter().zip(&want.dist2).enumerate() {
        assert!(
            (a - b).abs() <= 1e-9 * b.max(1.0),
            "{label}: dist2[{k}] {a} vs {b}"
        );
    }
}

/// (label, edge, tilts, periodic axes) of the random systems.
fn systems() -> Vec<(&'static str, f64, [f64; 3], [bool; 3])> {
    vec![
        ("orthorhombic", 40.0, [0.0; 3], [true; 3]),
        ("triclinic", 40.0, [0.2, 0.1, -0.15], [true; 3]),
        ("open", 30.0, [0.0; 3], [false; 3]),
        ("slab", 32.0, [0.2, 0.0, 0.0], [true, true, false]),
        ("narrow", 6.5, [0.2, 0.0, 0.0], [true; 3]),
    ]
}

#[test]
fn rows_are_the_pair_list_grouped_by_source() {
    for (label, edge, tilt, periodic) in systems() {
        let (xyz, h) = frame(edge, tilt, 0x9e37_79b9_7f4a_7c15);
        let cell = cell_of(h);
        for half in [false, true] {
            let query = PbcQuery::new(periodic, 3.5).half(half);
            let got = csr(&xyz, &cell, &query);
            let want = from_pairs(&xyz, &cell, &query);
            assert_eq!(bits(&got), bits(&want), "{label} half {half}");
            assert_eq!(got.rows(), xyz.len());
            for i in 0..got.rows() {
                let row = got.row(i);
                assert!(
                    row.windows(2).all(|w| w[0] < w[1]),
                    "{label}: row {i} sorted"
                );
                assert!(!row.contains(&(i as i32)), "{label}: row {i} lists itself");
                assert!(!half || row.iter().all(|&j| j > i as i32));
            }
        }
    }
}

#[test]
fn rows_match_brute_force_past_half_the_box() {
    // 2 * 3.5 is wider than every cell here, so one atom can meet
    // another through two images, and an atom can meet its own image.
    let cases = [
        ("cube", 6.0, [0.0; 3], [true; 3]),
        ("sheared", 6.5, [0.3, -0.2, 0.25], [true; 3]),
        ("film", 5.0, [0.1, 0.0, 0.0], [true, false, true]),
        ("wire", 4.0, [0.0; 3], [false, false, true]),
    ];
    for (label, edge, tilt, periodic) in cases {
        let (xyz, h) = frame(edge, tilt, 0x2545_f491_4f6c_dd1d);
        let cell = cell_of(h);
        for half in [false, true] {
            let query = PbcQuery::new(periodic, 3.5).half(half);
            let got = csr(&xyz, &cell, &query);
            let want = brute(&xyz, h, periodic, 3.5, half);
            assert_close(&got, &want, &format!("{label} half {half}"));
            assert_eq!(
                bits(&got),
                bits(&from_pairs(&xyz, &cell, &query)),
                "{label}"
            );
            let images = pairs_within_pbc(&xyz, &cell, &PbcQuery::new(periodic, 3.5))
                .unwrap()
                .iter()
                .filter(|p| p.i != p.j && (!half || p.j > p.i))
                .count();
            assert!(
                !got.cols.is_empty() && images > got.cols.len(),
                "{label}: some atom meets another through two images ({images} images, {} columns)",
                got.cols.len()
            );
        }
    }
}

#[test]
fn masked_points_have_empty_rows() {
    let (xyz, h) = frame(30.0, [0.1, 0.0, 0.0], 0x1234_5678_9abc_def1);
    let cell = cell_of(h);
    let mask: Vec<bool> = (0..xyz.len()).map(|i| i % 3 != 0).collect();
    for periodic in [[true; 3], [false; 3]] {
        let query = PbcQuery::new(periodic, 3.5).mask(Some(&mask));
        let got = csr(&xyz, &cell, &query);
        assert_eq!(bits(&got), bits(&from_pairs(&xyz, &cell, &query)));
        for i in (0..xyz.len()).step_by(3) {
            assert!(got.row(i).is_empty());
        }
        assert!(got.cols.iter().all(|&j| j % 3 != 0));
    }
}

#[test]
fn edge_cases_return_errors_or_empty_rows() {
    let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let mut out = PairCsr {
        offsets: vec![7; 4],
        cols: vec![7; 4],
        dist2: vec![7.0; 4],
    };
    pairs_within_csr(&[], &cell, &PbcQuery::periodic(3.5), true, &mut out).unwrap();
    assert_eq!(out.offsets, vec![0]);
    assert!(out.cols.is_empty() && out.dist2.is_empty());

    let lone = [[1.0, 2.0, 3.0]];
    pairs_within_csr(&lone, &cell, &PbcQuery::periodic(3.5), true, &mut out).unwrap();
    assert_eq!(out.offsets, vec![0, 0]);

    let xyz = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]];
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let err = pairs_within_csr(&xyz, &cell, &PbcQuery::periodic(bad), true, &mut out);
        assert_eq!(err, Err(Error::BadCutoff), "cutoff {bad}");
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let xyz = [[1.0, 1.0, 1.0], [2.0, bad, 1.0]];
        for periodic in [[true; 3], [false; 3]] {
            let err = pairs_within_csr(&xyz, &cell, &PbcQuery::new(periodic, 3.5), true, &mut out);
            assert_eq!(err, Err(Error::NonFinite), "coordinate {bad}");
        }
    }
    let short = [true];
    let query = PbcQuery::periodic(3.5).mask(Some(&short));
    assert_eq!(
        pairs_within_csr(&xyz, &cell, &query, true, &mut out),
        Err(Error::MaskLen)
    );
}

#[cfg(feature = "capi")]
mod capi {
    use super::*;
    use linkcell::{lc_cell, lc_last_error, lc_pairs_within_csr, lc_set_threads};
    use std::ffi::CStr;
    use std::os::raw::c_int;
    use std::ptr::{null, null_mut};

    fn raw(h: [[f64; 3]; 3]) -> lc_cell {
        lc_cell {
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

    struct Call<'a> {
        xyz: &'a [[f64; 3]],
        cell: Option<lc_cell>,
        pbc: Option<[c_int; 3]>,
        cutoff: f64,
        mask: Option<&'a [c_int]>,
        half: bool,
    }

    impl Call<'_> {
        /// # Safety
        /// The buffers hold their capacities.
        unsafe fn run(
            &self,
            offsets: *mut c_int,
            cap_offsets: usize,
            cols: *mut c_int,
            d2: *mut f64,
            cap_cols: usize,
            need: &mut (usize, usize),
        ) -> c_int {
            let cell = self.cell.as_ref().map_or(null(), |c| c as *const lc_cell);
            let pbc = self.pbc.as_ref().map_or(null(), |p| p.as_ptr());
            let mask = self.mask.map_or(null(), |m| m.as_ptr());
            unsafe {
                lc_pairs_within_csr(
                    self.xyz.as_ptr().cast(),
                    self.xyz.len(),
                    cell,
                    pbc,
                    self.cutoff,
                    mask,
                    0.0,
                    c_int::from(self.half),
                    offsets,
                    cap_offsets,
                    cols,
                    d2,
                    cap_cols,
                    &mut need.0,
                    &mut need.1,
                )
            }
        }

        /// Query, then fill exactly.
        fn rows(&self) -> PairCsr {
            let mut need = (0, 0);
            let rc = unsafe { self.run(null_mut(), 0, null_mut(), null_mut(), 0, &mut need) };
            assert_eq!(rc, 0, "query: {}", last_error());
            assert_eq!(need.0, self.xyz.len() + 1);
            let mut out = PairCsr {
                offsets: vec![-1; need.0],
                cols: vec![-1; need.1],
                dist2: vec![-1.0; need.1],
            };
            let mut again = (0, 0);
            let rc = unsafe {
                self.run(
                    out.offsets.as_mut_ptr(),
                    need.0,
                    out.cols.as_mut_ptr(),
                    out.dist2.as_mut_ptr(),
                    need.1,
                    &mut again,
                )
            };
            assert_eq!(rc, 0, "fill: {}", last_error());
            assert_eq!(again, need);
            out
        }

        /// One call into buffers larger than the rows.
        fn rows_at_once(&self, spare: usize) -> PairCsr {
            let n = self.xyz.len();
            let cap = n * n.min(256) + spare;
            let mut out = PairCsr {
                offsets: vec![-1; n + 1],
                cols: vec![-1; cap],
                dist2: vec![-1.0; cap],
            };
            let mut need = (0, 0);
            let rc = unsafe {
                self.run(
                    out.offsets.as_mut_ptr(),
                    n + 1,
                    out.cols.as_mut_ptr(),
                    out.dist2.as_mut_ptr(),
                    cap,
                    &mut need,
                )
            };
            assert_eq!(rc, 0, "fill: {}", last_error());
            assert!(need.1 <= cap);
            out.cols.truncate(need.1);
            out.dist2.truncate(need.1);
            out
        }
    }

    fn last_error() -> String {
        let msg = lc_last_error();
        if msg.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(msg) }
                .to_string_lossy()
                .into_owned()
        }
    }

    fn call<'a>(
        xyz: &'a [[f64; 3]],
        h: [[f64; 3]; 3],
        periodic: [bool; 3],
        cutoff: f64,
        half: bool,
    ) -> Call<'a> {
        let open = periodic == [false; 3];
        Call {
            xyz,
            cell: (!open).then(|| raw(h)),
            pbc: (periodic != [true; 3]).then(|| periodic.map(c_int::from)),
            cutoff,
            mask: None,
            half,
        }
    }

    #[test]
    fn c_rows_match_the_rust_rows() {
        for (label, edge, tilt, periodic) in systems() {
            let (xyz, h) = frame(edge, tilt, 0x0ddb_1a5e_5bad_5eed);
            let cell = cell_of(h);
            for half in [false, true] {
                let c = call(&xyz, h, periodic, 3.5, half);
                let want = csr(&xyz, &cell, &PbcQuery::new(periodic, 3.5).half(half));
                assert_eq!(
                    bits(&c.rows()),
                    bits(&want),
                    "{label} half {half} two calls"
                );
                assert_eq!(
                    bits(&c.rows_at_once(0)),
                    bits(&want),
                    "{label} half {half} one call"
                );
            }
        }
    }

    #[test]
    fn short_buffers_report_the_size_and_keep_the_rows() {
        let (xyz, h) = frame(20.0, [0.1, 0.0, 0.0], 7);
        let c = call(&xyz, h, [true; 3], 3.5, false);
        let want = c.rows();
        let n = xyz.len();
        let total = want.cols.len();
        let mut offsets = vec![-1 as c_int; n + 1];
        let mut cols = vec![-1 as c_int; total];
        let mut need = (0, 0);
        let rc = unsafe {
            c.run(
                offsets.as_mut_ptr(),
                n + 1,
                cols.as_mut_ptr(),
                null_mut(),
                total - 1,
                &mut need,
            )
        };
        assert_ne!(rc, 0);
        assert_eq!(need, (n + 1, total));
        assert!(last_error().contains("shorter"), "{}", last_error());
        assert!(cols.iter().all(|&c| c == -1), "a short buffer was written");
        let rc = unsafe {
            c.run(
                offsets.as_mut_ptr(),
                n,
                cols.as_mut_ptr(),
                null_mut(),
                total,
                &mut need,
            )
        };
        assert_ne!(rc, 0, "short offsets");
        // The rows kept above, without distances this time.
        let rc = unsafe {
            c.run(
                offsets.as_mut_ptr(),
                n + 1,
                cols.as_mut_ptr(),
                null_mut(),
                total,
                &mut need,
            )
        };
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(offsets, want.offsets);
        assert_eq!(cols, want.cols);
        assert!(last_error().is_empty());
    }

    #[test]
    fn c_edge_cases_fail_with_a_message() {
        let mut need = (9, 9);
        let empty = Call {
            xyz: &[],
            cell: None,
            pbc: None,
            cutoff: 3.5,
            mask: None,
            half: false,
        };
        let rc = unsafe { empty.run(null_mut(), 0, null_mut(), null_mut(), 0, &mut need) };
        assert_eq!((rc, need), (0, (1, 0)));
        let mut offset = [-1 as c_int];
        let rc = unsafe { empty.run(offset.as_mut_ptr(), 1, null_mut(), null_mut(), 0, &mut need) };
        assert_ne!(rc, 0, "a fill needs a column buffer");
        let mut col = [0 as c_int];
        let rc = unsafe {
            empty.run(
                offset.as_mut_ptr(),
                1,
                col.as_mut_ptr(),
                null_mut(),
                0,
                &mut need,
            )
        };
        assert_eq!((rc, offset[0]), (0, 0));

        let h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        for bad in [f64::NAN, f64::INFINITY, -f64::INFINITY] {
            let xyz = [[1.0, 1.0, 1.0], [bad, 1.0, 1.0]];
            for periodic in [[true; 3], [false; 3], [true, true, false]] {
                let c = call(&xyz, h, periodic, 3.5, false);
                let rc = unsafe { c.run(null_mut(), 0, null_mut(), null_mut(), 0, &mut need) };
                assert_ne!(rc, 0, "coordinate {bad} on {periodic:?}");
                assert!(last_error().contains("not finite"), "{}", last_error());
            }
        }
        let xyz = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]];
        for bad in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            let c = call(&xyz, h, [true; 3], bad, false);
            let rc = unsafe { c.run(null_mut(), 0, null_mut(), null_mut(), 0, &mut need) };
            assert_ne!(rc, 0, "cutoff {bad}");
            assert!(last_error().contains("cutoff"), "{}", last_error());
        }
        let c = call(&xyz, h, [true; 3], 3.5, false);
        let rc = unsafe {
            lc_pairs_within_csr(
                c.xyz.as_ptr().cast(),
                2,
                &raw(h),
                null(),
                3.5,
                null(),
                0.0,
                0,
                null_mut(),
                0,
                null_mut(),
                null_mut(),
                0,
                null_mut(),
                &mut need.1,
            )
        };
        assert_ne!(rc, 0, "null need pointer");
        let rc = unsafe {
            lc_pairs_within_csr(
                c.xyz.as_ptr().cast(),
                2,
                null(),
                null(),
                3.5,
                null(),
                0.0,
                0,
                null_mut(),
                0,
                null_mut(),
                null_mut(),
                0,
                &mut need.0,
                &mut need.1,
            )
        };
        assert_ne!(rc, 0, "a periodic cell needs a box");
        let singular = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let c = call(&xyz, singular, [true; 3], 0.5, false);
        let rc = unsafe { c.run(null_mut(), 0, null_mut(), null_mut(), 0, &mut need) };
        assert_ne!(rc, 0, "singular box");
    }

    #[test]
    fn c_mask_drops_points() {
        let (xyz, h) = frame(24.0, [0.0; 3], 11);
        let keep: Vec<c_int> = (0..xyz.len()).map(|i| c_int::from(i % 4 != 1)).collect();
        let flags: Vec<bool> = keep.iter().map(|&k| k != 0).collect();
        let mut c = call(&xyz, h, [true; 3], 3.5, true);
        c.mask = Some(&keep);
        let query = PbcQuery::periodic(3.5).half(true).mask(Some(&flags));
        assert_eq!(bits(&c.rows()), bits(&csr(&xyz, &cell_of(h), &query)));
    }

    /// Every C test that changes the width is this one, so no other test
    /// of this binary moves it in between.
    #[test]
    fn rows_are_the_same_bits_at_every_width() {
        let mut cases: Vec<(&str, f64, [f64; 3], [bool; 3])> = vec![
            ("orthorhombic 20k", 84.0, [0.0; 3], [true; 3]),
            ("triclinic 20k", 84.0, [0.2, 0.0, 0.0], [true; 3]),
        ];
        cases.extend(systems());
        for (label, edge, tilt, periodic) in cases {
            let (xyz, h) = frame(edge, tilt, 0x5eed_1234_dead_beef);
            for half in [false, true] {
                let c = call(&xyz, h, periodic, 3.5, half);
                lc_set_threads(1);
                let one = (bits(&c.rows()), bits(&c.rows_at_once(0)));
                assert_eq!(one.0, one.1, "{label}: one thread, one call and two");
                for width in [2, 8] {
                    lc_set_threads(width);
                    assert_eq!(bits(&c.rows()), one.0, "{label} half {half} width {width}");
                    assert_eq!(
                        bits(&c.rows_at_once(width as usize)),
                        one.0,
                        "{label} half {half} width {width}, one call"
                    );
                }
            }
        }
        lc_set_threads(0);
    }
}
