use linkcell::{
    knearest, knearest_brute, knearest_into, knearest_into_d2, knearest_into_many, Cell, Error,
    Neighbors,
};

fn almost(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn rejects_zero_k() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0]];
    assert!(knearest(&xyz, &b, 0, None, None).is_err());
}

#[test]
fn rejects_bad_box() {
    assert!(Cell::ortho(0.0, 1.0, 1.0).is_err());
    assert!(Cell::from_vectors(
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    )
    .is_err());
}

#[test]
fn two_points_k1() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let rows = knearest(&xyz, &b, 1, None, Some(2.0)).unwrap();
    assert_eq!(rows[0].indices, vec![1]);
    assert_eq!(rows[1].indices, vec![0]);
    assert!(almost(rows[0].dist2[0], 1.0));
}

#[test]
fn periodic_image_is_nearer() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
    let rows = knearest(&xyz, &b, 1, None, Some(2.0)).unwrap();
    assert_eq!(rows[0].indices, vec![1]);
    // 0.2 + (10-9.4) = 0.8, not 9.2
    assert!(almost(rows[0].dist2[0], 0.8 * 0.8));
}

#[test]
fn mask_drops_sources_and_candidates() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mask = [true, false, true];
    let rows = knearest(&xyz, &b, 1, Some(&mask), None).unwrap();
    assert!(rows[1].indices.is_empty());
    assert_eq!(rows[0].indices, vec![2]);
    assert_eq!(rows[2].indices, vec![0]);
}

#[test]
fn agrees_with_brute_force_on_a_random_cell() {
    let b = Cell::ortho(12.0, 11.0, 13.0).unwrap();
    let mut xyz = Vec::new();
    let mut s: u64 = 1;
    for _ in 0..64 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((s >> 11) as f64 / (1u64 << 53) as f64) * 12.0;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let y = ((s >> 11) as f64 / (1u64 << 53) as f64) * 11.0;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let z = ((s >> 11) as f64 / (1u64 << 53) as f64) * 13.0;
        xyz.push([x, y, z]);
    }
    let cell = knearest(&xyz, &b, 4, None, Some(2.5)).unwrap();
    let brute = knearest_brute(&xyz, &b, 4, None).unwrap();
    for i in 0..xyz.len() {
        assert_eq!(cell[i].indices, brute[i].indices, "source {i}");
        for (a, c) in cell[i].dist2.iter().zip(brute[i].dist2.iter()) {
            assert!(almost(*a, *c), "{a} vs {c}");
        }
    }
}

fn hex_prism() -> Cell {
    // a = (10,0,0), b = (5, 5*sqrt(3), 0), c = (0,0,10)
    Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [5.0, 8.660254037844386, 0.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
    )
    .unwrap()
}

#[test]
fn triclinic_image_beats_the_cartesian_far_point() {
    let b = hex_prism();
    // Near a and near a+b, which are 10 apart along a, but the
    // minimum image across -b + a is shorter than the raw vector.
    let xyz = [[0.2, 0.1, 1.0], [9.7, 0.1, 1.0]];
    let rows = knearest(&xyz, &b, 1, None, Some(2.0)).unwrap();
    assert_eq!(rows[0].indices, vec![1]);
    let brute = knearest_brute(&xyz, &b, 1, None).unwrap();
    assert_eq!(rows[0].indices, brute[0].indices);
    assert!(almost(rows[0].dist2[0], brute[0].dist2[0]));
    assert!(rows[0].dist2[0] < 1.0);
}

#[test]
fn hex_body_diagonal_knn_matches_euclidean() {
    let b = hex_prism();
    let origin = [0.0, 0.0, 0.0];
    let body = b.cartesian([0.49, 0.49, 0.49]);
    let xyz = [origin, body];
    let rows = knearest(&xyz, &b, 1, None, Some(2.0)).unwrap();
    let brute = knearest_brute(&xyz, &b, 1, None).unwrap();
    let euc = b.dist2_euclidean(origin, body);
    let frac = b.dist2(origin, body);
    assert_eq!(rows[0].indices, vec![1]);
    assert_eq!(rows[0].indices, brute[0].indices);
    assert!(almost(rows[0].dist2[0], brute[0].dist2[0]));
    assert!(almost(rows[0].dist2[0], euc));
    assert!(euc + 1e-8 < frac);
}

#[test]
fn agrees_with_brute_force_on_a_sheared_cell() {
    let b = hex_prism();
    let mut xyz = Vec::new();
    let mut s: u64 = 7;
    for _ in 0..48 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = (s >> 11) as f64 / (1u64 << 53) as f64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let w = (s >> 11) as f64 / (1u64 << 53) as f64;
        let r = [10.0 * u + 5.0 * v, 8.660254037844386 * v, 10.0 * w];
        xyz.push(r);
    }
    let cell = knearest(&xyz, &b, 4, None, Some(2.0)).unwrap();
    let brute = knearest_brute(&xyz, &b, 4, None).unwrap();
    for i in 0..xyz.len() {
        assert_eq!(cell[i].indices, brute[i].indices, "source {i}");
        for (a, c) in cell[i].dist2.iter().zip(brute[i].dist2.iter()) {
            assert!(almost(*a, *c), "{a} vs {c} at {i}");
        }
    }
}

fn triclinic() -> Cell {
    Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [3.0, 9.0, 0.0],
        [1.0, 2.0, 8.0],
        [0.0, 0.0, 0.0],
    )
    .unwrap()
}

fn lcg_frac(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn points_in_cell(cell: &Cell, n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut s = seed;
    let mut xyz = Vec::with_capacity(n);
    for _ in 0..n {
        let u = lcg_frac(&mut s);
        let v = lcg_frac(&mut s);
        let w = lcg_frac(&mut s);
        xyz.push(cell.cartesian([u, v, w]));
    }
    xyz
}

/// Independent 27-image MIC. Oracle for small parallelepipeds.
fn mic_dist2_27(cell: &Cell, p: [f64; 3], q: [f64; 3]) -> f64 {
    let a = cell.a();
    let b = cell.b();
    let c = cell.c();
    let mut best = f64::INFINITY;
    for na in -1..=1 {
        for nb in -1..=1 {
            for nc in -1..=1 {
                let fa = f64::from(na);
                let fb = f64::from(nb);
                let fc = f64::from(nc);
                let dx = q[0] + fa * a[0] + fb * b[0] + fc * c[0] - p[0];
                let dy = q[1] + fa * a[1] + fb * b[1] + fc * c[1] - p[1];
                let dz = q[2] + fa * a[2] + fb * b[2] + fc * c[2] - p[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < best {
                    best = r2;
                }
            }
        }
    }
    best
}

fn oracle_knearest(
    xyz: &[[f64; 3]],
    cell: &Cell,
    k: usize,
    mask: Option<&[bool]>,
) -> Vec<Neighbors> {
    let n = xyz.len();
    let active: Vec<usize> = (0..n)
        .filter(|&i| mask.map(|m| *m.get(i).unwrap_or(&false)).unwrap_or(true))
        .collect();
    let mut out = vec![Neighbors::default(); n];
    for &i in &active {
        let mut pairs: Vec<(f64, usize)> = active
            .iter()
            .copied()
            .filter(|&j| j != i)
            .map(|j| (mic_dist2_27(cell, xyz[i], xyz[j]), j))
            .collect();
        pairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        pairs.truncate(k);
        out[i].dist2 = pairs.iter().map(|p| p.0).collect();
        out[i].indices = pairs.iter().map(|p| p.1).collect();
    }
    out
}

fn assert_rows_match(label: &str, got: &[Neighbors], want: &[Neighbors]) {
    assert_eq!(got.len(), want.len(), "{label} row count");
    for i in 0..got.len() {
        assert_eq!(
            got[i].indices, want[i].indices,
            "{label} source {i} indices"
        );
        assert_eq!(
            got[i].dist2.len(),
            want[i].dist2.len(),
            "{label} source {i} dist2 len"
        );
        for (a, c) in got[i].dist2.iter().zip(want[i].dist2.iter()) {
            assert!(almost(*a, *c), "{label} source {i}: {a} vs {c}");
        }
    }
}

#[test]
fn sheared_brute_agrees_for_small_n_k1_and_k4() {
    let hex = hex_prism();
    let tri = triclinic();
    let cells: &[(&str, &Cell)] = &[("hex", &hex), ("triclinic", &tri)];
    let cases: &[(&str, usize, usize, u64, Option<f64>)] = &[
        ("n=5 k=1", 5, 1, 1, Some(2.0)),
        ("n=5 k=4", 5, 4, 1, Some(2.0)),
        ("n=8 k=1", 8, 1, 11, None),
        ("n=8 k=4", 8, 4, 11, Some(2.0)),
        ("n=12 k=1", 12, 1, 3, Some(1.5)),
        ("n=12 k=4", 12, 4, 3, None),
    ];
    for &(cell_name, cell) in cells {
        for &(nk_name, n, k, seed, hint) in cases {
            let label = format!("{cell_name} {nk_name} seed={seed}");
            let xyz = points_in_cell(cell, n, seed);
            let linked = knearest(&xyz, cell, k, None, hint).unwrap();
            let brute = knearest_brute(&xyz, cell, k, None).unwrap();
            let ora = oracle_knearest(&xyz, cell, k, None);
            assert_rows_match(&format!("{label} vs brute"), &linked, &brute);
            assert_rows_match(&format!("{label} vs 27-image"), &linked, &ora);
        }
    }
}

#[test]
fn wrap_around_sources_near_faces_and_corners() {
    let ortho = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let hex = hex_prism();
    let tri = triclinic();
    let cases: &[(&str, &Cell, &[[f64; 3]], usize)] = &[
        (
            "ortho-face-x k=1",
            &ortho,
            &[[0.12, 5.0, 5.0], [9.79, 5.0, 5.0], [5.0, 5.0, 5.0]],
            1,
        ),
        (
            "ortho-face-y k=1",
            &ortho,
            &[[5.0, 0.12, 5.0], [5.0, 9.79, 5.0], [5.0, 5.0, 5.0]],
            1,
        ),
        (
            "ortho-face-z k=1",
            &ortho,
            &[[5.0, 5.0, 0.12], [5.0, 5.0, 9.79], [5.0, 5.0, 5.0]],
            1,
        ),
        (
            "ortho-edge-xy k=1",
            &ortho,
            &[[0.12, 0.11, 5.0], [9.80, 9.81, 5.0], [5.0, 5.0, 5.0]],
            1,
        ),
        (
            "ortho-corner k=1",
            &ortho,
            &[[0.11, 0.12, 0.13], [9.82, 9.81, 9.80], [5.0, 5.0, 5.0]],
            1,
        ),
        (
            "ortho-faces k=4",
            &ortho,
            &[
                [0.10, 5.0, 5.0],
                [9.85, 5.0, 5.0],
                [5.0, 0.11, 5.0],
                [5.0, 9.84, 5.0],
                [5.0, 5.0, 0.12],
                [5.0, 5.0, 9.83],
                [5.0, 5.0, 5.0],
            ],
            4,
        ),
        (
            "ortho-same-bin two images k=2",
            &ortho,
            &[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [9.8, 0.0, 0.0]],
            2,
        ),
        (
            "hex-face-a k=1",
            &hex,
            &[[0.20, 0.10, 5.0], [9.70, 0.10, 5.0], [5.0, 4.0, 5.0]],
            1,
        ),
        (
            "hex-corner k=1",
            &hex,
            &[[0.25, 0.20, 0.25], [14.75, 8.46, 9.75], [5.0, 4.3, 5.0]],
            1,
        ),
        (
            "hex-faces k=4",
            &hex,
            &[
                [0.20, 0.15, 5.0],
                [9.70, 0.15, 5.0],
                [5.2, 0.20, 5.0],
                [7.7, 8.40, 5.0],
                [5.0, 4.3, 0.20],
                [5.0, 4.3, 9.75],
                [5.0, 4.3, 5.0],
            ],
            4,
        ),
        (
            "triclinic-face-a k=1",
            &tri,
            &[[0.20, 0.20, 4.0], [9.75, 0.20, 4.0], [6.0, 5.0, 4.0]],
            1,
        ),
        (
            "triclinic-corner k=1",
            &tri,
            &[[0.20, 0.20, 0.20], [13.80, 10.80, 7.80], [7.0, 5.5, 4.0]],
            1,
        ),
        (
            "triclinic-wrap k=4",
            &tri,
            &[
                [0.18, 0.15, 0.20],
                [13.82, 10.78, 7.79],
                [9.80, 0.25, 0.30],
                [0.25, 8.80, 0.22],
                [0.22, 0.30, 7.75],
                [7.0, 5.5, 4.0],
            ],
            4,
        ),
    ];
    for &(name, cell, xyz, k) in cases {
        let linked = knearest(xyz, cell, k, None, Some(2.0)).unwrap();
        let brute = knearest_brute(xyz, cell, k, None).unwrap();
        let ora = oracle_knearest(xyz, cell, k, None);
        assert_rows_match(&format!("{name} vs brute"), &linked, &brute);
        assert_rows_match(&format!("{name} vs 27-image"), &linked, &ora);
        assert!(
            linked[0].dist2[0] < 1.0,
            "{name} wrap dist2 {}",
            linked[0].dist2[0]
        );
        if xyz.len() == 3 && k == 1 {
            assert_eq!(linked[0].indices[0], 1, "{name} wrap neighbour");
        }
    }
}

#[test]
fn empty_xyz_is_error_empty() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz: [[f64; 3]; 0] = [];
    let err = knearest(&xyz, &b, 1, None, None).unwrap_err();
    assert_eq!(err, Error::Empty);
    assert_eq!(err.to_string(), "no points");
    assert_eq!(knearest_brute(&xyz, &b, 1, None).unwrap_err(), Error::Empty);
    let mut out: [i32; 0] = [];
    assert_eq!(
        knearest_into(&xyz, &b, 1, None, None, &mut out).unwrap_err(),
        Error::Empty
    );
}

#[test]
fn k_zero_is_error_zero_k() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0]];
    let err = knearest(&xyz, &b, 0, None, None).unwrap_err();
    assert_eq!(err, Error::ZeroK);
    assert_eq!(err.to_string(), "k must be at least 1");
    assert_eq!(knearest_brute(&xyz, &b, 0, None).unwrap_err(), Error::ZeroK);
    let mut out: [i32; 0] = [];
    assert_eq!(
        knearest_into(&xyz, &b, 0, None, None, &mut out).unwrap_err(),
        Error::ZeroK
    );
}

#[test]
fn short_out_buffer_is_not_empty() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let two = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let empty: [[f64; 3]; 0] = [];
    let cases: &[(&str, &[[f64; 3]], usize, usize)] = &[
        ("n=2 k=1 out=1", &two, 1, 1),
        ("n=2 k=2 out=2", &two, 2, 2),
        ("n=2 k=1 out=8", &two, 1, 8),
        ("n=0 k=1 out=1", &empty, 1, 1),
    ];
    for &(name, xyz, k, out_len) in cases {
        let mut out = vec![-1i32; out_len];
        let err = knearest_into(xyz, &b, k, None, None, &mut out).unwrap_err();
        assert_ne!(err, Error::Empty, "{name}");
        assert_eq!(err, Error::BufferSize, "{name}");
        assert_eq!(err.to_string(), "out buffer length must be n * k", "{name}");
    }
}

#[test]
fn packed_layout_is_i_times_k_plus_j() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ];
    let cases: &[usize] = &[1, 2, 3, 5];
    for &k in cases {
        let rows = knearest(&xyz, &b, k, None, None).unwrap();
        let mut out = vec![-2i32; xyz.len() * k];
        knearest_into(&xyz, &b, k, None, None, &mut out).unwrap();
        for i in 0..xyz.len() {
            let filled = rows[i].indices.len();
            for j in 0..filled {
                assert_eq!(
                    out[i * k + j],
                    rows[i].indices[j] as i32,
                    "out[{i}*{k}+{j}]"
                );
            }
            for j in filled..k {
                assert_eq!(out[i * k + j], -1, "unused out[{i}*{k}+{j}]");
            }
        }
    }
}

#[test]
fn mask_none_matches_all_ones() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.4, 0.0],
        [0.0, 0.0, 1.6],
        [2.0, 2.0, 2.0],
    ];
    let ones = [true; 5];
    let k = 3;
    let none_rows = knearest(&xyz, &b, k, None, None).unwrap();
    let ones_rows = knearest(&xyz, &b, k, Some(&ones), None).unwrap();
    assert_eq!(none_rows, ones_rows);
    let mut out_none = vec![-1i32; xyz.len() * k];
    let mut out_ones = vec![-1i32; xyz.len() * k];
    knearest_into(&xyz, &b, k, None, None, &mut out_none).unwrap();
    knearest_into(&xyz, &b, k, Some(&ones), None, &mut out_ones).unwrap();
    assert_eq!(out_none, out_ones);
}

#[test]
fn short_mask_is_mask_len() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mask = [true];
    assert_eq!(
        knearest(&xyz, &b, 1, Some(&mask), None).unwrap_err(),
        Error::MaskLen
    );
}

#[test]
fn empty_xyz_is_empty_not_buffer() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    assert_eq!(knearest(&[], &b, 1, None, None).unwrap_err(), Error::Empty);
}

#[test]
fn knearest_into_rejects_short_buffer() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mut out = [-1; 1];
    assert_eq!(
        knearest_into(&xyz, &b, 1, None, None, &mut out).unwrap_err(),
        Error::BufferSize
    );
}

#[test]
fn packed_layout_is_row_major() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mut out = [0; 4];
    knearest_into(&xyz, &b, 2, None, None, &mut out).unwrap();
    assert_eq!(out, [1, -1, 0, -1]);
}

#[test]
fn into_d2_writes_periodic_image() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.2, 0.0, 0.0], [9.4, 0.0, 0.0]];
    let mut nn = [-1; 2];
    let mut d2 = [0.0; 2];
    knearest_into_d2(&xyz, &b, 1, None, None, &mut nn, Some(&mut d2)).unwrap();
    assert_eq!(nn, [1, 0]);
    assert!(almost(d2[0], 0.64));
    assert!(almost(d2[1], 0.64));
}

#[test]
fn into_many_shares_one_cell() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [
        [0.2, 0.0, 0.0],
        [9.4, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [9.4, 0.0, 0.0],
    ];
    let mut nn = [-1; 4];
    let mut d2 = [0.0; 4];
    knearest_into_many(&xyz, 2, 2, &b, 1, None, None, &mut nn, Some(&mut d2)).unwrap();
    assert_eq!(nn, [1, 0, 1, 0]);
    assert!(almost(d2[0], 0.64));
    assert!(almost(d2[3], 0.64));
}

#[test]
fn corner_wrap_agrees_with_brute() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.1, 0.1, 0.1], [9.8, 9.8, 9.8], [5.0, 5.0, 5.0]];
    let cell = knearest(&xyz, &b, 1, None, Some(2.0)).unwrap();
    let brute = knearest_brute(&xyz, &b, 1, None).unwrap();
    assert_eq!(cell[0].indices, brute[0].indices);
    assert_eq!(cell[1].indices, brute[1].indices);
    assert_eq!(cell[0].indices, vec![1]);
}
