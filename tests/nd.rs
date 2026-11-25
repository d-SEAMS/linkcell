use linkcell::{ContiguousSpace, Error, Metric, NeighbourIndex, PeriodicSpace, VpTreeIndex};

#[path = "common/rational.rs"]
mod rational;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn build_index<'a>(
    data: &'a [f64],
    n: usize,
    d: usize,
    periods: &[Option<f64>],
) -> VpTreeIndex<ContiguousSpace<'a>, PeriodicSpace> {
    let space = ContiguousSpace::from_row_major(data, n, d).unwrap();
    let metric = PeriodicSpace::new(periods.to_vec()).unwrap();
    VpTreeIndex::new(space, metric).unwrap()
}

fn brute_knn(
    metric: &PeriodicSpace,
    data: &[f64],
    n: usize,
    d: usize,
    q: &[f64],
    k: usize,
) -> Vec<(usize, f64)> {
    let mut hits: Vec<_> = (0..n)
        .map(|i| {
            let p = &data[i * d..(i + 1) * d];
            (i, metric.dist2(p, q).unwrap())
        })
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    hits.truncate(k.min(n));
    hits
}

fn brute_within(
    metric: &PeriodicSpace,
    data: &[f64],
    n: usize,
    d: usize,
    q: &[f64],
    r: f64,
) -> Vec<(usize, f64)> {
    let r2 = r * r;
    let mut hits: Vec<_> = (0..n)
        .filter_map(|i| {
            let p = &data[i * d..(i + 1) * d];
            let d2 = metric.dist2(p, q).unwrap();
            (d2 <= r2).then_some((i, d2))
        })
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    hits
}

fn brute_dbscan(
    metric: &PeriodicSpace,
    data: &[f64],
    n: usize,
    d: usize,
    eps: f64,
    min_pts: usize,
) -> Vec<i32> {
    const UNASSIGNED: i32 = -2;
    const NOISE: i32 = -1;
    let mut labels = vec![UNASSIGNED; n];
    let mut cluster = 0i32;
    for i in 0..n {
        if labels[i] != UNASSIGNED {
            continue;
        }
        let qi = &data[i * d..(i + 1) * d];
        let mut neigh: Vec<_> = brute_within(metric, data, n, d, qi, eps)
            .into_iter()
            .map(|(idx, _)| idx)
            .collect();
        neigh.sort_unstable();
        if neigh.len() < min_pts {
            labels[i] = NOISE;
            continue;
        }
        labels[i] = cluster;
        let mut queued = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        for j in neigh {
            if j != i && !queued[j] {
                queued[j] = true;
                queue.push_back(j);
            }
        }
        while let Some(p) = queue.pop_front() {
            if labels[p] == NOISE {
                labels[p] = cluster;
            }
            if labels[p] != UNASSIGNED {
                continue;
            }
            labels[p] = cluster;
            let qp = &data[p * d..(p + 1) * d];
            let mut pn: Vec<_> = brute_within(metric, data, n, d, qp, eps)
                .into_iter()
                .map(|(idx, _)| idx)
                .collect();
            pn.sort_unstable();
            if pn.len() >= min_pts {
                for j in pn {
                    if !queued[j] {
                        queued[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        cluster += 1;
    }
    labels
        .into_iter()
        .map(|v| if v == UNASSIGNED { -1 } else { v })
        .collect()
}

fn brute_pairs_within(
    metric: &PeriodicSpace,
    data: &[f64],
    n: usize,
    d: usize,
    radius: f64,
) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    let r2 = radius * radius;
    for i in 0..n {
        let pi = &data[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let pj = &data[j * d..(j + 1) * d];
            let d2 = metric.dist2(pi, pj).unwrap();
            if d2 <= r2 {
                out.push((i, j, d2));
            }
        }
    }
    out
}

#[test]
fn random_knn_and_within_match_bruteforce() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..24 {
        let d = rng.random_range(1..=40);
        let n = rng.random_range(2..=96);
        let mut periods = vec![None; d];
        for p in &mut periods {
            if rng.random::<f64>() < 0.35 {
                *p = Some(rng.random_range(2.0..12.0));
            }
        }

        let mut data = vec![0.0; n * d];
        for i in 0..n {
            for k in 0..d {
                let mut v = rng.random_range(-8.0..8.0);
                if let Some(l) = periods[k] {
                    // Keep points away from [0, L) to exercise wrap handling.
                    v += rng.random_range(-2.0..2.0) * l;
                }
                data[i * d + k] = v;
            }
        }
        if n >= 2 {
            for k in 0..d {
                data[(n - 1) * d + k] = data[(n - 2) * d + k];
            }
        }

        let idx = build_index(&data, n, d, &periods);
        let metric = PeriodicSpace::new(periods.clone()).unwrap();
        let q = &data[..d];
        let k = rng.random_range(1..=n + 7);
        let got = idx.knn(q, k).unwrap();
        let want = brute_knn(&metric, &data, n, d, q, k);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.index, w.0);
            assert_eq!(g.dist2.to_bits(), w.1.to_bits());
        }

        let r = rng.random_range(0.1..4.0);
        let got_w = idx.within(q, r).unwrap();
        let want_w = brute_within(&metric, &data, n, d, q, r);
        assert_eq!(got_w.len(), want_w.len());
        for (g, w) in got_w.iter().zip(want_w.iter()) {
            assert_eq!(g.index, w.0);
            assert_eq!(g.dist2.to_bits(), w.1.to_bits());
        }
    }
}

#[test]
fn many_seeds_cover_edge_cases_and_ties() {
    let dims = [1usize, 2, 3, 8, 20, 40];
    for seed in 0..18u64 {
        let mut rng = StdRng::seed_from_u64(100 + seed);
        for &d in &dims {
            let n = rng.random_range(2..=2_400usize);
            let mut periods = vec![None; d];
            for p in &mut periods {
                if rng.random::<f64>() < 0.4 {
                    *p = Some(rng.random_range(1.5..15.0));
                }
            }
            let mut data = vec![0.0; n * d];
            for i in 0..n {
                for k in 0..d {
                    let mut v = rng.random_range(-20.0..20.0);
                    if let Some(l) = periods[k] {
                        v += rng.random_range(-4.0..4.0) * l;
                    }
                    data[i * d + k] = v;
                }
            }
            // Duplicates and exact ties.
            if n > 3 {
                for k in 0..d {
                    data[d + k] = data[k];
                    data[2 * d + k] = data[k];
                }
            }
            let idx = build_index(&data, n, d, &periods);
            let metric = PeriodicSpace::new(periods.clone()).unwrap();
            let query_ids = [0usize, n / 3, n - 1];
            for &qi in &query_ids {
                let q = &data[qi * d..(qi + 1) * d];
                for &k in &[0usize, 1, 4, n / 2, n + 13] {
                    let got = idx.knn(q, k).unwrap();
                    let want = brute_knn(&metric, &data, n, d, q, k);
                    assert_eq!(got.len(), want.len());
                    for (g, w) in got.iter().zip(&want) {
                        assert_eq!(g.index, w.0);
                        assert_eq!(g.dist2.to_bits(), w.1.to_bits());
                    }
                }
                let zero = idx.within(q, 0.0).unwrap();
                for hit in &zero {
                    assert_eq!(hit.dist2, 0.0);
                }
                let large = idx.within(q, 1000.0).unwrap();
                assert_eq!(large.len(), n);
            }
        }
    }
}

#[test]
fn edge_cases_work() {
    let d = 3usize;
    let periods = vec![None, Some(6.0), None];
    let empty = build_index(&[], 0, d, &periods);
    assert!(empty.knn(&[0.0, 0.0, 0.0], 4).unwrap().is_empty());
    assert!(empty.within(&[0.0, 0.0, 0.0], 1.0).unwrap().is_empty());

    let one_data = vec![0.0, 0.0, 0.0];
    let one = build_index(&one_data, 1, d, &periods);
    assert_eq!(one.knn(&[0.0, 0.0, 0.0], 3).unwrap().len(), 1);
    let self_hits = one.within_index(0, 0.01).unwrap();
    assert_eq!(self_hits.len(), 1);
    assert_eq!(self_hits[0].index, 0);
    assert_eq!(
        one.within_index(1, 0.1).unwrap_err(),
        linkcell::Error::IndexOutOfRange
    );
}

#[test]
fn n_zero_and_n_one_and_periodic_half_box_radius() {
    let d = 1usize;
    let periods = vec![Some(10.0)];
    let empty = build_index(&[], 0, d, &periods);
    assert!(empty.knn(&[0.0], 3).unwrap().is_empty());
    assert!(empty.within(&[0.0], 5.0).unwrap().is_empty());

    let data = vec![-11.0, -5.0, 0.0, 4.0, 19.0];
    let idx = build_index(&data, 5, d, &periods);
    // On a 1D circle every point is within L/2 of every point.
    let hits = idx.within(&[0.0], 5.0).unwrap();
    assert_eq!(hits.len(), 5);

    let one = build_index(&[3.0], 1, d, &periods);
    assert_eq!(one.knn(&[3.0], 0).unwrap().len(), 0);
    assert_eq!(one.knn(&[3.0], 5).unwrap().len(), 1);
}

#[test]
fn rust_index_build_rejects_nan_points() {
    let data = [f64::NAN, 0.0];
    let space = ContiguousSpace::from_row_major(&data, 1, 2).unwrap();
    let metric = PeriodicSpace::new(vec![None, None]).unwrap();
    match VpTreeIndex::new(space, metric) {
        Ok(_) => panic!("building index with NaN should fail"),
        Err(err) => assert_eq!(err, linkcell::Error::NonFinite),
    }
}

#[test]
fn dbscan_matches_oracle_and_sklearn_toy() {
    let d = 2usize;
    let periods = vec![None, None];
    let x = [
        1.0, 2.0, //
        2.0, 2.0, //
        2.0, 3.0, //
        8.0, 7.0, //
        8.0, 8.0, //
        25.0, 80.0,
    ];
    let idx = build_index(&x, 6, d, &periods);
    let out = idx.dbscan(3.0, 2).unwrap();
    assert_eq!(out.labels, vec![0, 0, 0, 1, 1, -1]);

    let mut rng = StdRng::seed_from_u64(17);
    for _ in 0..8 {
        let n = rng.random_range(12..=64);
        let d = rng.random_range(1..=10);
        let mut periods = vec![None; d];
        for p in &mut periods {
            if rng.random::<f64>() < 0.3 {
                *p = Some(rng.random_range(2.0..8.0));
            }
        }
        let mut data = vec![0.0; n * d];
        for v in &mut data {
            *v = rng.random_range(-5.0..5.0);
        }
        let idx = build_index(&data, n, d, &periods);
        let metric = PeriodicSpace::new(periods.clone()).unwrap();
        let eps = rng.random_range(0.3..2.0);
        let min_pts = rng.random_range(2..=6);
        let got = idx.dbscan(eps, min_pts).unwrap();
        let want = brute_dbscan(&metric, &data, n, d, eps, min_pts);
        assert_eq!(got.labels, want);
    }
}

#[test]
fn dbscan_border_assignment_is_deterministic() {
    // Two dense cores with one border point touching both.
    let d = 2usize;
    let periods = vec![None, None];
    let points = vec![
        0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // cluster A core
        1.0, 0.0, 1.1, 0.0, 1.0, 0.1, // cluster B core
        0.55, 0.02, // border near both
    ];
    let idx = build_index(&points, 7, d, &periods);
    let a = idx.dbscan(0.5, 3).unwrap();
    let b = idx.dbscan(0.5, 3).unwrap();
    assert_eq!(a.labels, b.labels);
}

#[test]
fn metric_distance_is_symmetric_and_self_zero() {
    let metric = PeriodicSpace::new(vec![Some(0.5), Some(1.0), None]).unwrap();
    let a = [0.0, 0.3, -1.0];
    let b = [0.1, -0.2, 4.0];
    assert_eq!(metric.dist2(&a, &a).unwrap(), 0.0);
    assert_eq!(metric.dist2(&b, &b).unwrap(), 0.0);
    assert_eq!(metric.dist2(&a, &b).unwrap(), metric.dist2(&b, &a).unwrap());
}

#[test]
fn exact_boundary_queries_match_oracle_on_periodic_lattice() {
    let d = 1usize;
    let periods = vec![Some(0.5)];
    let metric = PeriodicSpace::new(periods.clone()).unwrap();
    let mut data = Vec::new();
    for i in 0..24usize {
        data.push(i as f64 * 0.1);
    }
    let idx = build_index(&data, data.len(), d, &periods);
    for i in 0..data.len() {
        let q = [data[i]];
        for &radius in &[0.0, 0.1, 0.2, 0.25] {
            let got = idx.within(&q, radius).unwrap();
            let want = brute_within(&metric, &data, data.len(), d, &q, radius);
            let got_pairs: Vec<_> = got.into_iter().map(|h| (h.index, h.dist2)).collect();
            assert_eq!(got_pairs, want);
        }
        let got_knn = idx.knn(&q, 7).unwrap();
        let want_knn = brute_knn(&metric, &data, data.len(), d, &q, 7);
        let got_pairs: Vec<_> = got_knn.into_iter().map(|h| (h.index, h.dist2)).collect();
        assert_eq!(got_pairs, want_knn);
    }
    let got_pairs = idx.pairs_within(0.1).unwrap();
    let got: Vec<_> = got_pairs.into_iter().map(|p| (p.i, p.j, p.dist2)).collect();
    let want = brute_pairs_within(&metric, &data, data.len(), d, 0.1);
    assert_eq!(got, want);
    let db = idx.dbscan(0.1, 2).unwrap();
    let want_labels = brute_dbscan(&metric, &data, data.len(), d, 0.1, 2);
    assert_eq!(db.labels, want_labels);
}

#[test]
fn exact_boundary_queries_match_oracle_on_open_integer_lattice() {
    let d = 1usize;
    let periods = vec![None];
    let metric = PeriodicSpace::new(periods.clone()).unwrap();
    let data: Vec<f64> = (0..40usize).map(|i| i as f64).collect();
    let idx = build_index(&data, data.len(), d, &periods);
    let q = [17.0];
    for &radius in &[0.0, 1.0, 2.0, 5.0] {
        let got = idx.within(&q, radius).unwrap();
        let want = brute_within(&metric, &data, data.len(), d, &q, radius);
        let got_pairs: Vec<_> = got.into_iter().map(|h| (h.index, h.dist2)).collect();
        assert_eq!(got_pairs, want);
    }
}

#[test]
fn axis_and_aabb_bounds_do_not_exceed_the_rational_distance() {
    let mut rng = StdRng::seed_from_u64(999);
    for _ in 0..400 {
        let l = rng.random_range(2.0..13.0);
        let metric = PeriodicSpace::new(vec![Some(l), None]).unwrap();
        let x = rng.random_range(-90.0..90.0);
        let lo = rng.random_range(-60.0..20.0);
        let width = rng.random_range(0.0..(2.5 * l));
        let hi = lo + width;
        let got = metric.axis_min_dist2(x, lo, hi, 0).unwrap();
        let exact = rational::periodic_interval_sq(x, lo, hi, l);
        assert!(
            rational::rat_from_f64(got) <= exact,
            "axis bound {got} above the rational distance"
        );
        if width >= l {
            assert_eq!(got, 0.0);
        }

        let y = rng.random_range(-8.0..8.0);
        let y_lo = -2.0;
        let y_hi = 4.0;
        let lb = metric
            .aabb_min_dist2(&[x, y], &[lo, y_lo], &[hi, y_hi])
            .unwrap();
        let exact_box = rational::periodic_interval_sq(x, lo, hi, l)
            + rational::open_interval_sq(y, y_lo, y_hi);
        assert!(
            rational::rat_from_f64(lb) <= exact_box,
            "box bound {lb} above the rational distance"
        );
    }

    let witnesses = [
        (10.0, 1.0e13, 3.0, 5.0),
        (10.0, 2.0f64.powi(62), 0.0, 2.0),
        (7.0, 1.0e13, 0.0, 1.0),
        (1.0e16, -1.0, 0.0, 0.0),
        (2.0f64.powi(53), -0.5, 0.0, 0.0),
    ];
    for (l, x, lo, hi) in witnesses {
        let metric = PeriodicSpace::new(vec![Some(l)]).unwrap();
        let got = metric.axis_min_dist2(x, lo, hi, 0).unwrap();
        let exact = rational::periodic_interval_sq(x, lo, hi, l);
        assert!(rational::rat_from_f64(got) <= exact, "L={l:e} bound {got}");
        if lo == hi {
            assert!(got > 0.0, "degenerate bound collapsed");
        }
    }
}

fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == -0.0 {
        return f64::from_bits(1);
    }
    if x >= 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

#[test]
fn high_dimensional_boundary_radii_match_oracle_on_collinear_rays() {
    let dims = [2usize, 8, 64, 72, 96, 128, 1024, 4096];
    let n = 6usize;
    for &d in &dims {
        let periods = vec![None; d];
        let metric = PeriodicSpace::new(periods.clone()).unwrap();
        let mut data = vec![0.0f64; n * d];
        for i in 0..n {
            let t = i as f64 * 0.1;
            for k in 0..d {
                data[i * d + k] = t;
            }
        }
        let idx = build_index(&data, n, d, &periods);
        let q = &data[2 * d..3 * d];
        let p3 = &data[3 * d..4 * d];
        let r0 = metric.distance(q, p3).unwrap();
        let r1 = next_up(r0);
        for &radius in &[r0, r1] {
            let got = idx.within(q, radius).unwrap();
            let want = brute_within(&metric, &data, n, d, q, radius);
            let got_pairs: Vec<_> = got.into_iter().map(|h| (h.index, h.dist2)).collect();
            assert_eq!(got_pairs, want, "within mismatch at d={d}, radius={radius}");

            let got_pairs = idx.pairs_within(radius).unwrap();
            let got: Vec<_> = got_pairs.into_iter().map(|p| (p.i, p.j, p.dist2)).collect();
            let want = brute_pairs_within(&metric, &data, n, d, radius);
            assert_eq!(got, want, "pairs mismatch at d={d}, radius={radius}");

            let got_db = idx.dbscan(radius, 2).unwrap();
            let want_db = brute_dbscan(&metric, &data, n, d, radius, 2);
            assert_eq!(
                got_db.labels, want_db,
                "dbscan mismatch at d={d}, radius={radius}"
            );
        }

        let got_knn = idx.knn(q, n).unwrap();
        let want_knn = brute_knn(&metric, &data, n, d, q, n);
        let got_pairs: Vec<_> = got_knn.into_iter().map(|h| (h.index, h.dist2)).collect();
        assert_eq!(got_pairs, want_knn, "knn mismatch at d={d}");
    }
}

fn assert_hit_pairs(got: &[linkcell::NeighborHit], want: &[(usize, f64)], what: &str) {
    let got_pairs: Vec<(usize, f64)> = got.iter().map(|h| (h.index, h.dist2)).collect();
    assert_eq!(got_pairs, want, "{what}");
}

/// Reviewer repro: L = 2π, coordinates across 20 periods, and the boundary
/// radius that dropped point 1. Every listed radius is checked against brute
/// force, including the exact distance to each point and its neighbours.
#[test]
fn periodic_span_of_many_periods_matches_bruteforce_at_boundary_radii() {
    let l = 2.0 * std::f64::consts::PI;
    let periods = vec![Some(l)];
    let metric = PeriodicSpace::new(periods.clone()).unwrap();
    let mut data = Vec::new();
    for i in 0..96usize {
        let periods_out = -10.0 + (i as f64) * 20.0 / 95.0;
        let jitter = (i as f64) * 0.37 + (i * i) as f64 * 1.0e-4;
        data.push(jitter + periods_out * l);
    }
    let q_review = -5.675156211725825;
    let r_review = 5.4225756419539195e-3;
    data.push(q_review);
    data.push(q_review + l);
    data.push(q_review - 3.0 * l);
    data.push(q_review + 20.0 * l);
    data.push(q_review + r_review);
    data.push(q_review + 20.0 * l + r_review);
    data.push(q_review - r_review * 0.5 + 7.0 * l);
    let n = data.len();
    let idx = build_index(&data, n, 1, &periods);
    let queries = [
        q_review,
        0.0,
        l,
        -l,
        data[1],
        data[50.min(n - 1)],
        data[82.min(n - 1)],
        data[0],
        data[n - 1],
    ];
    for &q in &queries {
        let qv = [q];
        let mut radii = vec![
            0.0,
            r_review,
            next_down(r_review),
            next_up(r_review),
            0.5 * l,
        ];
        for &p in &data {
            let dist = metric.distance(&[p], &qv).unwrap();
            radii.push(dist);
            if dist > 0.0 {
                radii.push(next_down(dist));
            }
            radii.push(next_up(dist));
        }
        for &r in &radii {
            if !r.is_finite() || r < 0.0 {
                continue;
            }
            let got = idx.within(&qv, r).unwrap();
            let want = brute_within(&metric, &data, n, 1, &qv, r);
            assert_hit_pairs(&got, &want, &format!("within q={q} r={r}"));
        }
        let got = idx.knn(&qv, n).unwrap();
        let want = brute_knn(&metric, &data, n, 1, &qv, n);
        assert_hit_pairs(&got, &want, &format!("knn q={q}"));
    }
    for &r in &[0.0, r_review, next_down(r_review), next_up(r_review)] {
        let got: Vec<_> = idx
            .pairs_within(r)
            .unwrap()
            .into_iter()
            .map(|p| (p.i, p.j, p.dist2))
            .collect();
        let want = brute_pairs_within(&metric, &data, n, 1, r);
        assert_eq!(got, want, "pairs r={r}");
    }

    // Mixed open and periodic axes, still spanning many periods.
    let d = 3usize;
    let periods3 = vec![Some(l), None, Some(1.7)];
    let metric3 = PeriodicSpace::new(periods3.clone()).unwrap();
    let n3 = 36usize;
    let mut data3 = vec![0.0; n3 * d];
    for i in 0..n3 {
        data3[i * d] = -8.0 * l + (i as f64) * (16.0 * l) / (n3 as f64);
        data3[i * d + 1] = (i as f64) * 0.2 - 3.0;
        data3[i * d + 2] = (i as f64) * 0.15 + (i / 3) as f64 * 1.7;
    }
    let idx3 = build_index(&data3, n3, d, &periods3);
    for i in [0usize, 1, 17, n3 - 1] {
        let q = &data3[i * d..(i + 1) * d];
        let mut radii3 = vec![0.0, r_review, 1e-15];
        for j in 0..n3 {
            let p = &data3[j * d..(j + 1) * d];
            let dist = metric3.distance(q, p).unwrap();
            radii3.push(dist);
            if dist > 0.0 {
                radii3.push(next_down(dist));
            }
            radii3.push(next_up(dist));
        }
        for &r in &radii3 {
            let got = idx3.within(q, r).unwrap();
            let want = brute_within(&metric3, &data3, n3, d, q, r);
            assert_hit_pairs(&got, &want, &format!("3d within i={i} r={r}"));
        }
        let got = idx3.knn(q, n3).unwrap();
        let want = brute_knn(&metric3, &data3, n3, d, q, n3);
        assert_hit_pairs(&got, &want, &format!("3d knn i={i}"));
    }
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

#[test]
fn tiny_radii_on_periodic_duplicates_match_bruteforce() {
    let l = 2.0 * std::f64::consts::PI;
    let x = 1.0 / 3.0;
    let mut data = Vec::new();
    for k in -6..=16 {
        data.push(x + (k as f64) * l);
    }
    data.push(x + 20.0 * l);
    data.push(x - 20.0 * l);
    data.push(x + 40.0 * l);
    data.push(x + 1.0e-18);
    data.push(x + l + 5.0e-324);
    let periods = vec![Some(l), Some(4.0)];
    // Second axis is a periodic duplicate too: (x, 0.5) and (x+kL, 0.5+m*4).
    let d = 2usize;
    let n1 = data.len();
    let mut pts = Vec::with_capacity(n1 * 2 * d);
    for (i, &a) in data.iter().enumerate() {
        pts.push(a);
        pts.push(0.5);
        pts.push(a);
        pts.push(0.5 + (i as f64) * 4.0);
    }
    let n = pts.len() / d;
    let metric = PeriodicSpace::new(periods.clone()).unwrap();
    let idx = build_index(&pts, n, d, &periods);
    let radii = [
        0.0,
        f64::from_bits(1),
        f64::from_bits(2),
        1.0e-320,
        1.0e-300,
        1.0e-200,
        1.0e-154,
        1.0e-40,
        1.0e-20,
        1.0e-16,
        1.0e-15,
        1.0e-14,
        1.0e-12,
        1.0e-9,
    ];
    for &r in &radii {
        for i in 0..n {
            let q = &pts[i * d..(i + 1) * d];
            let got = idx.within(q, r).unwrap();
            let want = brute_within(&metric, &pts, n, d, q, r);
            assert_hit_pairs(&got, &want, &format!("within i={i} r={r}"));
            for &k in &[1usize, 3, n] {
                let got = idx.knn(q, k).unwrap();
                let want = brute_knn(&metric, &pts, n, d, q, k);
                assert_hit_pairs(&got, &want, &format!("knn i={i} k={k} r={r}"));
            }
        }
        let got: Vec<_> = idx
            .pairs_within(r)
            .unwrap()
            .into_iter()
            .map(|p| (p.i, p.j, p.dist2))
            .collect();
        let want = brute_pairs_within(&metric, &pts, n, d, r);
        assert_eq!(got, want, "pairs r={r}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_and_serial_paths_match_exactly() {
    let mut rng = StdRng::seed_from_u64(2026);
    let n = 1400usize;
    let d = 8usize;
    let periods = vec![
        Some(9.0),
        None,
        Some(5.0),
        None,
        None,
        Some(11.0),
        None,
        None,
    ];
    let mut data = vec![0.0; n * d];
    for i in 0..n {
        for k in 0..d {
            let mut v = rng.random_range(-15.0..15.0);
            if let Some(l) = periods[k] {
                v += rng.random_range(-3.0..3.0) * l;
            }
            data[i * d + k] = v;
        }
    }
    let idx = build_index(&data, n, d, &periods);
    let query_n = 200usize;
    let mut queries = Vec::with_capacity(query_n * d);
    for q in 0..query_n {
        let p = &data[(q * d)..((q + 1) * d)];
        queries.extend_from_slice(p);
    }

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let k1 = pool1.install(|| idx.knn_many(&queries, 9).unwrap());
    let k4 = pool4.install(|| idx.knn_many(&queries, 9).unwrap());
    assert_eq!(k1, k4);

    let w1 = pool1.install(|| idx.within_many(&queries, 1.5).unwrap());
    let w4 = pool4.install(|| idx.within_many(&queries, 1.5).unwrap());
    assert_eq!(w1, w4);

    let d1 = pool1.install(|| idx.dbscan(1.8, 5).unwrap());
    let d4 = pool4.install(|| idx.dbscan(1.8, 5).unwrap());
    assert_eq!(d1.labels, d4.labels);
}

#[test]
fn periodic_reduction_matches_bruteforce_across_magnitudes() {
    let l = 2.0 * std::f64::consts::PI;
    let q = -5.675156211725825;
    let periods = vec![Some(l)];
    let metric = PeriodicSpace::new(periods.clone()).unwrap();
    let mut data = vec![q, 0.0, l, -l];
    for &k in &[
        1.0,
        -1.0,
        20.0,
        -20.0,
        1.0e6,
        -1.0e6,
        2.0f64.powi(40),
        -2.0f64.powi(50),
        1.0e16,
        -1.0e16,
    ] {
        let shifted = q + k * l;
        let scaled = k * l;
        assert!(shifted.is_finite() && scaled.is_finite());
        data.push(shifted);
        data.push(scaled);
    }
    for x in [1.0e300, -1.0e300, f64::MAX / 4.0, -f64::MAX / 4.0] {
        assert!(x.is_finite());
        data.push(x);
    }
    let n = data.len();
    let idx = build_index(&data, n, 1, &periods);
    let radii = [
        0.0,
        f64::from_bits(1),
        1.0e-15,
        5.4225756419539195e-3,
        0.1,
        1.0,
        0.5 * l,
    ];
    for i in 0..n {
        let qv = [data[i]];
        for &r in &radii {
            let got = idx.within(&qv, r).unwrap();
            let want = brute_within(&metric, &data, n, 1, &qv, r);
            assert_hit_pairs(&got, &want, &format!("within i={i} r={r}"));
        }
        let got = idx.knn(&qv, n).unwrap();
        let want = brute_knn(&metric, &data, n, 1, &qv, n);
        assert_hit_pairs(&got, &want, &format!("knn i={i}"));
    }
    for &r in &radii {
        let got: Vec<_> = idx
            .pairs_within(r)
            .unwrap()
            .into_iter()
            .map(|p| (p.i, p.j, p.dist2))
            .collect();
        let want = brute_pairs_within(&metric, &data, n, 1, r);
        assert_eq!(got, want, "pairs r={r}");
        if r > 0.0 {
            let got_db = idx.dbscan(r, 2).unwrap();
            let want_db = brute_dbscan(&metric, &data, n, 1, r, 2);
            assert_eq!(got_db.labels, want_db, "dbscan r={r}");
        }
    }
}

/// 20_000 identical points used to recurse one level per point and overflow
/// a 2 MiB stack. The build and the queries run on a 32 KiB stack.
#[test]
fn large_duplicate_set_builds_on_a_small_stack() {
    const N: usize = 20_000;
    let built = std::thread::Builder::new()
        .name("nd-small-stack".into())
        .stack_size(32 * 1024)
        .spawn(|| {
            let data = vec![0.125f64; N];
            let periods = vec![Some(1.0)];
            let idx = build_index(&data, N, 1, &periods);
            let metric = PeriodicSpace::new(periods).unwrap();
            let q = [0.125];
            let got = idx.knn(&q, 8).unwrap();
            let want = brute_knn(&metric, &data, N, 1, &q, 8);
            assert_hit_pairs(&got, &want, "identical knn");
            let got_w = idx.within(&q, 0.0).unwrap();
            let want_w = brute_within(&metric, &data, N, 1, &q, 0.0);
            assert_hit_pairs(&got_w, &want_w, "identical within 0");
            assert_eq!(want_w.len(), N);

            let l = 2.0 * std::f64::consts::PI;
            let base = -5.675156211725825;
            let mut images = Vec::with_capacity(N);
            for i in 0..N {
                let k = (i as i32) - 10_000;
                images.push(base + f64::from(k) * l);
            }
            let periods = vec![Some(l)];
            let idx = build_index(&images, N, 1, &periods);
            let metric = PeriodicSpace::new(periods).unwrap();
            let q = [base];
            let got = idx.knn(&q, 5).unwrap();
            let want = brute_knn(&metric, &images, N, 1, &q, 5);
            assert_hit_pairs(&got, &want, "image knn");
            for &r in &[0.0, f64::from_bits(1), 1.0e-15, 5.4225756419539195e-3] {
                let got_w = idx.within(&q, r).unwrap();
                let want_w = brute_within(&metric, &images, N, 1, &q, r);
                assert_hit_pairs(&got_w, &want_w, &format!("image within r={r}"));
            }
        })
        .expect("spawn small stack");
    built.join().expect("small-stack build panicked");
}

fn from_f64(x: f64) -> (i128, i32) {
    if x == 0.0 {
        return (0, 0);
    }
    let bits = x.to_bits();
    let sign: i128 = if bits >> 63 == 1 { -1 } else { 1 };
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & 0x000f_ffff_ffff_ffff) as i128;
    if exp_bits == 0 {
        (sign * frac, -1074)
    } else {
        (sign * ((1_i128 << 52) + frac), exp_bits - 1023 - 52)
    }
}

fn round_shift(m: u128, shift: i32) -> u128 {
    if shift >= 0 {
        return m << shift;
    }
    let s = (-shift) as u32;
    if s >= 128 {
        return 0;
    }
    let half = 1_u128 << (s - 1);
    let mask = (half << 1) - 1;
    let rem = m & mask;
    let quot = m >> s;
    let up = if rem > half {
        true
    } else if rem < half {
        false
    } else {
        (quot & 1) == 1
    };
    if up {
        quot + 1
    } else {
        quot
    }
}

/// Round `mant * 2^exp` to binary64, ties to even.
fn to_f64(mant: i128, exp: i32) -> f64 {
    if mant == 0 {
        return 0.0;
    }
    let sign = mant < 0;
    let mut m = mant.unsigned_abs();
    let mut e = exp;
    while m % 2 == 0 {
        m /= 2;
        e += 1;
    }
    let bits = 128 - m.leading_zeros();
    let e_unb = e + bits as i32 - 1;
    if e_unb > 1023 {
        return if sign {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if e_unb < -1074 {
        if e_unb < -1075 || (e_unb == -1075 && m == 1) {
            return 0.0;
        }
        return if sign {
            -f64::from_bits(1)
        } else {
            f64::from_bits(1)
        };
    }
    if e_unb < -1022 {
        let sh = e + 1074;
        let k = if sh >= 0 { m << sh } else { round_shift(m, sh) };
        if k == 0 {
            return 0.0;
        }
        if k >= (1_u128 << 52) {
            let mut bits_out = 1_u64 << 52;
            if sign {
                bits_out |= 1_u64 << 63;
            }
            return f64::from_bits(bits_out);
        }
        let mut bits_out = k as u64;
        if sign {
            bits_out |= 1_u64 << 63;
        }
        return f64::from_bits(bits_out);
    }
    let shift = 53 - bits as i32;
    let mut sig = round_shift(m, shift);
    let mut eu = e_unb;
    if sig == (1_u128 << 53) {
        sig = 1_u128 << 52;
        eu += 1;
        if eu > 1023 {
            return if sign {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
    }
    assert!(
        ((1_u128 << 52)..(1_u128 << 53)).contains(&sig),
        "sig {sig} bits {bits} shift {shift}"
    );
    let frac = (sig - (1_u128 << 52)) as u64;
    let mut bits_out = ((eu + 1023) as u64) << 52 | frac;
    if sign {
        bits_out |= 1_u64 << 63;
    }
    f64::from_bits(bits_out)
}

fn sub_exact(a: (i128, i32), b: (i128, i32)) -> (i128, i32) {
    if a.0 == 0 {
        return (-b.0, b.1);
    }
    if b.0 == 0 {
        return a;
    }
    let exp = a.1.min(b.1);
    let sa = a.1 - exp;
    let sb = b.1 - exp;
    assert!((0..71).contains(&sa) && (0..71).contains(&sb));
    let av = a.0.checked_shl(sa as u32).expect("shift");
    let bv = b.0.checked_shl(sb as u32).expect("shift");
    (av - bv, exp)
}

fn rounded_square(mant: i128, exp: i32) -> f64 {
    if mant == 0 {
        return 0.0;
    }
    let mut m = mant.unsigned_abs();
    let mut e = exp;
    while m % 2 == 0 {
        m /= 2;
        e += 1;
    }
    let sq = m.checked_mul(m).expect("square fits in u128");
    assert!(sq < (1_u128 << 127));
    to_f64(sq as i128, e.checked_mul(2).unwrap())
}

fn open_dist2(a: f64, b: f64) -> f64 {
    let (m, e) = sub_exact(from_f64(a), from_f64(b));
    rounded_square(m, e)
}

fn residue_f64(x: f64, l: f64) -> f64 {
    let (mx, ex) = from_f64(x);
    let (ml, el) = from_f64(l);
    assert!(ml > 0);
    let exp = ex.min(el);
    let sx = ex - exp;
    let sl = el - exp;
    assert!((0..71).contains(&sx) && (0..71).contains(&sl));
    let ix = mx.checked_shl(sx as u32).expect("residue shift");
    let il = ml.checked_shl(sl as u32).expect("period shift");
    let y = to_f64(ix.rem_euclid(il), exp);
    if y > 0.0 && y < l {
        y
    } else {
        0.0
    }
}

/// Circular square of two binary64 values, as a binary64 when that square is one.
///
/// The square is [`rational::periodic_sq`]. It does not call [`Metric::dist2`].
fn periodic_dist2(a: f64, b: f64, l: f64) -> f64 {
    rational::approx_f64(&rational::periodic_sq(a, b, l))
}

fn circ_i128(a: i128, b: i128, period: i128) -> i128 {
    let d = (a - b).rem_euclid(period);
    let other = period - d;
    if other < d {
        other
    } else {
        d
    }
}

fn assert_same_bits(got: f64, want: f64, what: &str) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{what}: got {got:e} want {want:e}"
    );
}

fn assert_search_matches<F>(
    data: &[f64],
    d: usize,
    periods: &[Option<f64>],
    queries: &[&[f64]],
    radii: &[f64],
    exact_dist2: bool,
    dist: F,
) where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let n = data.len() / d;
    let idx = build_index(data, n, d, periods);
    let metric = PeriodicSpace::new(periods.to_vec()).unwrap();
    if exact_dist2 {
        for i in 0..n {
            for j in 0..n {
                let pi = &data[i * d..(i + 1) * d];
                let pj = &data[j * d..(j + 1) * d];
                assert_same_bits(
                    metric.dist2(pi, pj).unwrap(),
                    dist(pi, pj),
                    &format!("dist2 {i},{j}"),
                );
            }
        }
    }
    for q in queries {
        if exact_dist2 {
            for i in 0..n {
                let pi = &data[i * d..(i + 1) * d];
                assert_same_bits(
                    metric.dist2(pi, q).unwrap(),
                    dist(pi, q),
                    &format!("query {i}"),
                );
            }
        }
        for &r in radii {
            let exact_r2 = rounded_square(from_f64(r).0, from_f64(r).1);
            assert_same_bits(r * r, exact_r2, "radius square");
            let mut want = Vec::new();
            for i in 0..n {
                let d2 = dist(&data[i * d..(i + 1) * d], q);
                if d2 <= exact_r2 {
                    want.push((i, d2));
                }
            }
            want.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let got: Vec<_> = idx
                .within(q, r)
                .unwrap()
                .into_iter()
                .map(|h| (h.index, h.dist2))
                .collect();
            if exact_dist2 {
                assert_eq!(got, want, "within r={r} q={q:?}");
            } else {
                let got_i: Vec<_> = got.iter().map(|h| h.0).collect::<Vec<_>>();
                let want_i: Vec<_> = want.iter().map(|h| h.0).collect::<Vec<_>>();
                assert_eq!(got_i, want_i, "within r={r} q={q:?}");
            }
            for &k in &[1usize, n] {
                let mut all: Vec<_> = (0..n)
                    .map(|i| (i, dist(&data[i * d..(i + 1) * d], q)))
                    .collect();
                all.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                all.truncate(k.min(n));
                let got: Vec<_> = idx
                    .knn(q, k)
                    .unwrap()
                    .into_iter()
                    .map(|h| (h.index, h.dist2))
                    .collect();
                if exact_dist2 {
                    assert_eq!(got, all, "knn k={k}");
                } else {
                    let got_i: Vec<_> = got.iter().map(|h| h.0).collect::<Vec<_>>();
                    let want_i: Vec<_> = all.iter().map(|h| h.0).collect::<Vec<_>>();
                    assert_eq!(got_i, want_i, "knn k={k}");
                }
            }
        }
        for &r in radii {
            let exact_r2 = rounded_square(from_f64(r).0, from_f64(r).1);
            let mut want = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let d2 = dist(&data[i * d..(i + 1) * d], &data[j * d..(j + 1) * d]);
                    if d2 <= exact_r2 {
                        want.push((i, j, d2));
                    }
                }
            }
            let got: Vec<_> = idx
                .pairs_within(r)
                .unwrap()
                .into_iter()
                .map(|p| (p.i, p.j, p.dist2))
                .collect();
            if exact_dist2 {
                assert_eq!(got, want, "pairs r={r}");
            } else {
                let got_ij: Vec<_> = got.iter().map(|p| (p.0, p.1)).collect();
                let want_ij: Vec<_> = want.iter().map(|p| (p.0, p.1)).collect();
                assert_eq!(got_ij, want_ij, "pairs r={r}");
            }
        }
    }
}

/// Exact residue arithmetic, independent of `Metric::dist2`.
#[test]
fn exact_residue_oracle_matches_neighbour_sets() {
    let two55 = 2.0f64.powi(55);
    let two62 = 2.0f64.powi(62);
    assert_eq!(residue_f64(two55, 10.0), 8.0);
    assert_eq!(residue_f64(two62, 10.0), 4.0);
    assert_eq!(residue_f64(11.0, 10.0), 1.0);
    assert_eq!(residue_f64(-11.0, 10.0), 9.0);
    assert_eq!(residue_f64(8.25, 7.0), 1.25);
    assert_eq!(residue_f64(1.25, 7.0), 1.25);
    assert_eq!(residue_f64(-1.0e-20, 1.0), 0.0);
    assert_eq!((1_i128 << 55) % 10, 8);
    assert_eq!((1_i128 << 62) % 10, 4);
    assert_eq!(circ_i128(1_i128 << 55, 0, 10), 2);
    assert_eq!(circ_i128(1_i128 << 55, 8, 10), 0);
    assert_eq!(circ_i128(1, 11, 10), 0);
    assert_eq!(circ_i128(0, 11, 10), 1);
    assert_eq!(circ_i128(5, 33, 28), 0);

    let open_periods = [None];
    let f1 = [2.0e-162, 2.2e-162, 0.0];
    let f1q = [3.5e-162];
    assert_eq!(open_dist2(f1q[0], f1[0]), 0.0);
    assert_eq!(open_dist2(f1q[0], f1[1]), 0.0);
    assert_ne!(open_dist2(f1q[0], f1[2]), 0.0);
    assert_eq!(open_dist2(f1[0], f1[1]), 0.0);
    assert_search_matches(&f1, 1, &open_periods, &[&f1q], &[0.0], true, |a, b| {
        open_dist2(a[0], b[0])
    });
    let f1_idx = build_index(&f1, 3, 1, &open_periods);
    let f1_hits: Vec<_> = f1_idx
        .within(&f1q, 0.0)
        .unwrap()
        .into_iter()
        .map(|h| (h.index, h.dist2))
        .collect();
    assert_eq!(f1_hits, vec![(0, 0.0), (1, 0.0)]);

    let l_tiny = 1.0e-150;
    let f2 = [
        -1.0540256054825181e-153_f64,
        2.9184493613603147e-153,
        -6.4388324845298795e-155,
    ];
    let f2q = [1.290897940786428e-154_f64];
    let f2r = 2.789359567281672e-153_f64;
    let f2_periods = [Some(l_tiny)];
    // exact_dist2 is false: `periodic_dist2` falls back to `BigRational::to_f64`
    // when the square is not a binary64 number, and that conversion is not a
    // bit-identical promise. The check is the neighbour index set.
    assert_search_matches(&f2, 1, &f2_periods, &[&f2q], &[0.0, f2r], false, |a, b| {
        periodic_dist2(a[0], b[0], l_tiny)
    });
    let f2_metric = PeriodicSpace::new(f2_periods.to_vec()).unwrap();
    let boundary = f2_metric.dist2(&[f2[1]], &f2q).unwrap();
    assert_same_bits(
        boundary,
        periodic_dist2(f2[1], f2q[0], l_tiny),
        "f2 boundary",
    );
    assert_eq!(boundary.to_bits(), (f2r * f2r).to_bits());
    let f2_idx = build_index(&f2, 3, 1, &f2_periods);
    let f2_hits: Vec<_> = f2_idx
        .within(&f2q, f2r)
        .unwrap()
        .into_iter()
        .map(|h| h.index)
        .collect();
    assert_eq!(f2_hits, vec![2, 0, 1]);

    let coords = [0_i128, 4, 8, 1, 11, -11, 1 << 55, 1 << 62];
    let lattice10: Vec<f64> = coords.iter().map(|c| *c as f64).collect();
    let q0 = [0.0_f64];
    let q4 = [4.0_f64];
    assert_search_matches(
        &lattice10,
        1,
        &[Some(10.0)],
        &[&q0, &q4],
        &[0.0, 2.0, 4.0],
        true,
        |a, b| {
            let d = circ_i128(a[0] as i128, b[0] as i128, 10);
            (d * d) as f64
        },
    );

    let seventh = [1.25_f64, 8.25, 4.25];
    let q125 = [1.25_f64];
    assert_search_matches(
        &seventh,
        1,
        &[Some(7.0)],
        &[&q125],
        &[0.0, 3.0],
        true,
        |a, b| {
            let aq = (a[0] * 4.0) as i128;
            let bq = (b[0] * 4.0) as i128;
            let d = circ_i128(aq, bq, 28);
            (d * d) as f64 / 16.0
        },
    );

    let mut pts = Vec::new();
    for i in -4..=4 {
        for j in 0..6 {
            pts.push((i as i128, j as i128));
        }
    }
    pts.push((360 * 5 + 7, 10 * 3 + 4));
    pts.push((-(360 * 3), 0));
    pts.push((1_i128 << 40, 8));
    pts.push((997 * 20, 360));
    let lattice360: Vec<f64> = pts
        .iter()
        .flat_map(|(x, y)| [*x as f64, *y as f64])
        .collect();
    let origin = [0.0_f64, 0.0];
    let spot = [7.0_f64, 4.0];
    assert_search_matches(
        &lattice360,
        2,
        &[Some(360.0), Some(10.0)],
        &[&origin, &spot],
        &[0.0, 1.0, 5.0, 25.0],
        true,
        |a, b| {
            let dx = circ_i128(a[0] as i128, b[0] as i128, 360);
            let dy = circ_i128(a[1] as i128, b[1] as i128, 10);
            (dx * dx + dy * dy) as f64
        },
    );

    let below = f64::from_bits(1.0f64.to_bits() - 1);
    let seam = [0.0_f64, 0.2, 0.8, 1.0, -0.2, -1.0, -1.0e-20, below];
    let seam_q = [0.0_f64];
    assert_eq!(periodic_dist2(0.0, 1.0, 1.0), 0.0);
    assert_eq!(periodic_dist2(0.0, -1.0, 1.0), 0.0);
    let tiny_sq = periodic_dist2(0.0, -1.0e-20, 1.0);
    assert_eq!(tiny_sq, open_dist2(0.0, -1.0e-20));
    assert_ne!(tiny_sq, 0.0);
    assert_eq!(periodic_dist2(0.0, 0.5, 1.0), 0.25);
    assert_eq!(periodic_dist2(0.0, below, 1.0), 2.0f64.powi(-106));
    // exact_dist2 is false for the same reason as the tiny-period set above:
    // the rational fallback is not required to match `dist2` bit for bit, so
    // only the neighbour indices are compared.
    assert_search_matches(
        &seam,
        1,
        &[Some(1.0)],
        &[&seam_q],
        &[0.0, 0.25, 2.0f64.powi(-53)],
        false,
        |a, b| periodic_dist2(a[0], b[0], 1.0),
    );
}

#[test]
fn huge_period_negative_delta_repros() {
    let specs = [
        (2.0f64.powi(53), [-0.5, 0.0], 0.25),
        (1.0e16, [-1.0, 0.0], 1.0),
        (1.0e300, [-1.0, 0.0], 1.0),
        (0.1 * 2.0f64.powi(60), [-0.125, 0.0], 0.015625),
    ];
    for (l, pair, want) in specs {
        let metric = PeriodicSpace::new(vec![Some(l)]).unwrap();
        assert_eq!(metric.dist2(&pair[..1], &pair[1..]).unwrap(), want);
        let bound = metric.axis_min_dist2(pair[0], pair[1], pair[1], 0).unwrap();
        assert!(bound > 0.0 && bound <= want, "L={l:e} bound {bound}");
    }

    let period = 1.0e16;
    let points = [-0.5_f64, 0.25];
    let idx = build_index(&points, 2, 1, &[Some(period)]);
    let nearest = idx.knn(&[0.0], 1).unwrap();
    assert_eq!(nearest.len(), 1);
    assert_eq!(nearest[0].index, 1);
    assert_eq!(nearest[0].dist2, 0.0625);
    let both = idx.knn(&[0.0], 2).unwrap();
    assert_eq!(both[0].index, 1);
    assert_eq!(both[0].dist2, 0.0625);
    assert!(idx.within(&[0.0], 0.1).unwrap().is_empty());

    let cluster = [-0.5_f64, 0.0, 0.5];
    let db = build_index(&cluster, 3, 1, &[Some(period)])
        .dbscan(0.4, 2)
        .unwrap();
    assert_eq!(db.labels, vec![-1, -1, -1]);
    assert!(db.core.iter().all(|flag| !flag));
    assert_eq!(db.clusters, 0);

    let metric = PeriodicSpace::new(vec![Some(period)]).unwrap();
    let axis = metric.axis_min_dist2(-1.0, 0.0, 0.0, 0).unwrap();
    assert!(axis > 0.0 && axis <= 1.0, "{axis}");
}

#[test]
fn huge_period_matches_exact_rational_reference() {
    let coords = [
        -4.0, -2.0, -1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0,
    ];
    let periods = [
        2.0f64.powi(53),
        2.0f64.powi(60),
        2.0f64.powi(100),
        2.0f64.powi(500),
        2.0f64.powi(1023),
        1.0e16,
        1.0e300,
        0.1 * 2.0f64.powi(60),
        f64::MAX,
    ];
    for &l in &periods {
        assert!(l >= 64.0 && l.is_finite());
        let metric = PeriodicSpace::new(vec![Some(l)]).unwrap();
        let boxed = PeriodicSpace::new(vec![Some(l), Some(l)]).unwrap();
        for &a in &coords {
            for &b in &coords {
                // |a - b| <= 8 and L/2 >= 32, so the circular distance is the
                // open difference. The reference is that rational, not `dist2`.
                let want = rational::periodic_sq(a, b, l);
                let got = metric.dist2(&[a], &[b]).unwrap();
                assert!(
                    rational::eq_f64(&want, got),
                    "L={l:e} a={a} b={b}: got {got:e}"
                );
                let bound = metric.axis_min_dist2(a, b, b, 0).unwrap();
                assert!(
                    rational::rat_from_f64(bound) <= want,
                    "axis {bound} above the rational distance"
                );
                if !rational::eq_f64(&want, 0.0) {
                    assert!(bound > 0.0, "collapsed axis L={l:e} a={a} b={b}");
                }
                let want_box = rational::periodic_sq(a, 0.0, l) + rational::periodic_sq(b, 0.0, l);
                let lb = boxed
                    .aabb_min_dist2(&[a, b], &[0.0, 0.0], &[0.0, 0.0])
                    .unwrap();
                assert!(
                    rational::rat_from_f64(lb) <= want_box,
                    "box {lb} above the rational distance"
                );
                if !rational::eq_f64(&want_box, 0.0) {
                    assert!(lb > 0.0, "collapsed box L={l:e} a={a} b={b}");
                }
            }
        }
        assert_eq!(metric.dist2(&[l], &[0.0]).unwrap(), 0.0);
        assert_eq!(metric.dist2(&[-l], &[0.0]).unwrap(), 0.0);
    }

    // One ulp below a power-of-two period. The rational reference is the
    // integer circular distance, not `Metric::dist2`.
    for exp in [53_u32, 60, 100] {
        let l_i = 1_i128 << exp;
        let ulp = 1_i128 << (exp - 52);
        let x_i = l_i - ulp;
        let gap = circ_i128(x_i, 0, l_i);
        assert_eq!(gap, ulp);
        let want = (gap * gap) as f64;
        let metric = PeriodicSpace::new(vec![Some(l_i as f64)]).unwrap();
        assert_eq!(metric.dist2(&[x_i as f64], &[0.0]).unwrap(), want);
        let bound = metric.axis_min_dist2(x_i as f64, 0.0, 0.0, 0).unwrap();
        assert!(bound > 0.0 && bound <= want, "{bound} {want}");
        let exact = rational::periodic_interval_sq(x_i as f64, 0.0, 0.0, l_i as f64);
        assert!(rational::rat_from_f64(bound) <= exact);
    }
}

#[test]
fn fmod_before_difference_keeps_seam_and_huge_magnitudes() {
    let ten = PeriodicSpace::new(vec![Some(10.0)]).unwrap();
    let huge = ten.dist2(&[1.7e308], &[-1.7e308]).unwrap();
    assert_eq!(huge, 16.0);
    assert!(rational::eq_f64(
        &rational::periodic_sq(1.7e308, -1.7e308, 10.0),
        huge
    ));

    let data = [-1.0e-20_f64, 0.0];
    let idx = build_index(&data, 2, 1, &[Some(10.0)]);
    let hit = idx.knn(&[0.0], 1).unwrap();
    assert_eq!(hit.len(), 1);
    assert_eq!(hit[0].index, 1);
    assert_eq!(hit[0].dist2, 0.0);

    let r: f64 = 2.000000000000001e-3;
    let near = PeriodicSpace::new(vec![Some(1.0)]).unwrap();
    let d = near.dist2(&[0.999], &[0.001]).unwrap();
    // The circular square is not a binary64 number. The computed square is
    // the binary64 rounding of that rational, and it is the squared radius.
    let exact = rational::periodic_sq(0.999, 0.001, 1.0);
    assert_eq!(d.to_bits(), rational::approx_f64(&exact).to_bits());
    assert_eq!(d.to_bits(), (r * r).to_bits());
    let idx = build_index(&[0.999], 1, 1, &[Some(1.0)]);
    let hits = idx.within(&[0.001], r).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].index, 0);
    assert_eq!(hits[0].dist2.to_bits(), d.to_bits());
}

#[test]
fn wrap_helpers_match_huge_multiples_modulo_the_period() {
    let multiples = [
        (2.0f64.powi(62), 10.0),
        (-2.0f64.powi(62), 10.0),
        (1.0e16, 10.0),
        (-1.0e16, 10.0),
        (1.7e308, 10.0),
        (-1.7e308, 10.0),
        (30.0, 10.0),
        (-30.0, 10.0),
        (1.0e16, 1.0e16),
        (-1.0e16, 1.0e16),
        (2.0f64.powi(53), 2.0f64.powi(53)),
        (-2.0f64.powi(53), 2.0f64.powi(53)),
        (2.0f64.powi(50) * 10.0, 10.0),
        (-(2.0f64.powi(50) * 10.0), 10.0),
    ];
    for (x, l) in multiples {
        let metric = PeriodicSpace::new(vec![Some(l)]).unwrap();
        let wrapped = metric.wrap_point_axis(x, 0).unwrap();
        assert!(
            rational::congruent(x, wrapped, l),
            "wrap_point {x:e} -> {wrapped:e} mod {l:e}"
        );
        let delta = metric.wrap_delta_axis(x, 0).unwrap();
        assert!(
            rational::congruent(x, delta, l),
            "wrap_delta {x:e} -> {delta:e} mod {l:e}"
        );
    }
}

#[test]
fn min_pts_zero_and_distance_overflow_have_their_own_messages() {
    let idx = build_index(&[0.0, 1.0], 2, 1, &[Some(1.0)]);
    let min_pts = idx.dbscan(0.5, 0).unwrap_err();
    assert_eq!(min_pts.to_string(), "min_pts must be at least 1");
    let eps = idx.dbscan(0.0, 2).unwrap_err();
    assert_eq!(eps.to_string(), "eps must be finite and strictly positive");
    assert_ne!(min_pts.to_string(), eps.to_string());
    assert_eq!(Error::Overflow.to_string(), "n * k overflows");
    assert_eq!(Error::DistanceOverflow.to_string(), "distance overflowed");
    assert_ne!(
        Error::Overflow.to_string(),
        Error::DistanceOverflow.to_string()
    );
    let wide = PeriodicSpace::new(vec![Some(1.0e200)]).unwrap();
    assert_eq!(
        wide.dist2(&[0.0], &[6.0e199]).unwrap_err(),
        Error::DistanceOverflow
    );
}

#[test]
fn gaps_that_straddle_the_cut_keep_their_square() {
    let l53 = 2.0f64.powi(53);
    let half = 2.0f64.powi(52);
    let m = PeriodicSpace::new(vec![Some(l53)]).unwrap();
    assert_eq!(m.dist2(&[half - 1.0], &[half]).unwrap(), 1.0);
    assert_eq!(m.dist2(&[half + 1.0], &[half]).unwrap(), 1.0);
    let pair = [half - 1.0, half];
    let shifted = build_index(&pair, 2, 1, &[Some(l53)]);
    let hits = shifted.within(&[half], 1.0).unwrap();
    assert_eq!(hits.len(), 2);
    let triple = [half - 1.0, half, half + 1.0];
    let db = build_index(&triple, 3, 1, &[Some(l53)])
        .dbscan(1.5, 2)
        .unwrap();
    assert!(db.clusters >= 1);
    let axis = m.axis_min_dist2(half - 1.0, half, half, 0).unwrap();
    assert!(axis > 0.0 && axis <= 1.0, "{axis}");

    let l16 = 1.0e16_f64;
    let wide = PeriodicSpace::new(vec![Some(l16)]).unwrap();
    let a = -5.0e15_f64;
    let b = 4999999999999999.0_f64;
    assert_eq!(wide.dist2(&[a], &[b]).unwrap(), 1.0);
    let points = [a, b];
    let idx = build_index(&points, 2, 1, &[Some(l16)]);
    assert_eq!(idx.within(&[a], 0.5).unwrap().len(), 1);
    assert_eq!(idx.knn(&[a], 2).unwrap()[1].dist2, 1.0);
    assert!(idx.pairs_within(0.5).unwrap().is_empty());
    assert_eq!(idx.dbscan(0.5, 2).unwrap().clusters, 0);
    let bound = wide
        .axis_min_dist2(
            5.000000000000002e15,
            4.999999999999999e15,
            4.999999999999999e15,
            0,
        )
        .unwrap();
    assert!(bound <= 9.0, "{bound}");
    assert_eq!(wide.wrap_delta_axis(-1.0, 0).unwrap(), -1.0);

    let tiny_period = PeriodicSpace::new(vec![Some(1.0e-10)]).unwrap();
    assert!(tiny_period.wrap_delta_axis(1.0e300, 0).unwrap().is_finite());
    let sub = PeriodicSpace::new(vec![Some(5.0e-324)]).unwrap();
    assert_ne!(sub.wrap_delta_axis(0.0, 0).unwrap(), -5.0e-324);

    let huge = PeriodicSpace::new(vec![Some(1.0e300)]).unwrap();
    assert_eq!(
        huge.dist2(&[-5.0e299], &[4.9999999999999995e299])
            .unwrap_err(),
        Error::DistanceOverflow
    );
    let biggest = PeriodicSpace::new(vec![Some(1.0e308)]).unwrap();
    assert_eq!(biggest.dist2(&[1.0], &[0.0]).unwrap(), 1.0);
    assert!(biggest.dist2(&[0.0], &[1.0e308 * 0.5]).is_err());

    let unit = PeriodicSpace::new(vec![Some(1.0)]).unwrap();
    let near_half = unit
        .dist2(&[4.999999999999974e-1], &[5.000000000000074e-1])
        .unwrap();
    let scale = 1.0095262545291237e-28_f64;
    assert!((near_half / scale - 1.0).abs() < 1e-12, "{near_half:e}");

    let near_cut = [-0.5_f64, 0.49999999999999994];
    let dup = build_index(&near_cut, 2, 1, &[Some(1.0)]);
    assert_eq!(dup.within(&[-0.5], 0.0).unwrap().len(), 1);

    let open = linkcell::Euclidean::new(4).unwrap();
    let u = 2.0f64.powi(-27);
    let exact = 1.0 + 2.0f64.powi(-26) + 2.0f64.powi(-52);
    assert_eq!(
        open.dist2(&[1.0 + u, u, u, u], &[0.0; 4])
            .unwrap()
            .to_bits(),
        exact.to_bits()
    );
}

#[test]
fn straddling_fold_matches_the_rational_square() {
    let periods = [
        1.0,
        7.0,
        0.1,
        3.0,
        1.0e16,
        2.0f64.powi(53),
        0.1 * 2.0f64.powi(60),
        1.0e300,
        1.0e308,
    ];
    let mut straddling = 0_u64;
    let mut outside = 0_u64;
    let mut false_zeros = 0_u64;
    let mut worst = 0.0_f64;
    let max_sq = rational::rat_from_f64(f64::MAX);
    for &l in &periods {
        let metric = PeriodicSpace::new(vec![Some(l)]).unwrap();
        let samples = fold_samples(l);
        for &a in &samples {
            for &b in &samples {
                if !rational::straddles_fold(a, b, l) {
                    continue;
                }
                straddling += 1;
                let exact = rational::periodic_sq(a, b, l);
                let got = metric.dist2(&[a], &[b]);
                if exact > max_sq {
                    assert!(
                        got.is_err(),
                        "L={l:e} a={a:e} b={b:e} fitted but overflowed"
                    );
                    continue;
                }
                let got =
                    got.unwrap_or_else(|_| panic!("spurious overflow L={l:e} a={a:e} b={b:e}"));
                if got == 0.0
                    && exact > rational::rat_from_f64(0.0)
                    && !rational::is_nearest(0.0, &exact)
                {
                    false_zeros += 1;
                }
                if !rational::is_nearest(got, &exact) {
                    outside += 1;
                }
                worst = worst.max(rational::error_ulps(got, &exact));
            }
        }
    }
    assert!(straddling > 0, "the sweep produced no straddling pairs");
    assert_eq!(
        outside, 0,
        "straddling {straddling} outside {outside} worst {worst}"
    );
    assert_eq!(
        false_zeros, 0,
        "false zeros in {straddling} straddling pairs"
    );
    assert!(
        worst <= 0.5,
        "worst ulp {worst} over {straddling} straddling pairs"
    );
    eprintln!(
        "straddling {straddling} outside {outside} false_zeros {false_zeros} worst_ulp {worst}"
    );
}

fn fold_samples(l: f64) -> Vec<f64> {
    let mut v = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 8.0, -8.0];
    for k in -2..=2 {
        for n in [0.0, 1.0, 7.0, 8.0, 15.0] {
            let x = (k as f64) * l + n * l / 16.0;
            if x.is_finite() {
                v.push(x);
            }
        }
    }
    let half = l * 0.5;
    if half.is_finite() && half != 0.0 {
        v.push(half);
        v.push(-half);
        let up = f64::from_bits(half.to_bits() + 1);
        let down = f64::from_bits(half.to_bits() - 1);
        v.push(up);
        v.push(down);
        for d in [1.0, 0.5, 8.0] {
            let a = half - d;
            let b = -half + d;
            if a.is_finite() {
                v.push(a);
            }
            if b.is_finite() {
                v.push(b);
            }
        }
    }
    let base = v.clone();
    for x in base {
        for shift in [l + l, -(l + l)] {
            let y = x + shift;
            if y.is_finite() {
                v.push(y);
            }
        }
    }
    v.retain(|x| x.is_finite());
    v.sort_by(f64::total_cmp);
    v.dedup();
    v
}
