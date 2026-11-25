use linkcell::{ContiguousSpace, Metric, NeighbourIndex, PeriodicSpace, VpTreeIndex};
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

fn brute_axis_min_dist2(x: f64, lo: f64, hi: f64, period: Option<f64>) -> f64 {
    if let Some(l) = period {
        if hi - lo >= l {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        let steps = 8_000usize;
        for t in 0..=steps {
            let y = lo + (hi - lo) * (t as f64 / steps as f64);
            let mut d = x - y;
            d = d - l * (d / l).trunc();
            if d < 0.0 {
                d += l;
            }
            d = d.min(l - d);
            best = best.min(d * d);
        }
        best
    } else if x < lo {
        (lo - x) * (lo - x)
    } else if x > hi {
        (x - hi) * (x - hi)
    } else {
        0.0
    }
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
            assert!((g.dist2 - w.1).abs() < 1e-12);
        }

        let r = rng.random_range(0.1..4.0);
        let got_w = idx.within(q, r).unwrap();
        let want_w = brute_within(&metric, &data, n, d, q, r);
        assert_eq!(got_w.len(), want_w.len());
        for (g, w) in got_w.iter().zip(want_w.iter()) {
            assert_eq!(g.index, w.0);
            assert!((g.dist2 - w.1).abs() < 1e-12);
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
                        assert!((g.dist2 - w.1).abs() < 1e-12);
                    }
                }
                let zero = idx.within(q, 0.0).unwrap();
                for hit in &zero {
                    assert!(hit.dist2 <= 1e-14);
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
fn metric_bounds_are_sound_for_unwrapped_and_wide_intervals() {
    let mut rng = StdRng::seed_from_u64(999);
    for _ in 0..600 {
        let l = rng.random_range(2.0..13.0);
        let metric = PeriodicSpace::new(vec![Some(l), None]).unwrap();
        let x = rng.random_range(-90.0..90.0);
        let lo = rng.random_range(-60.0..20.0);
        let width = rng.random_range(0.0..(2.5 * l));
        let hi = lo + width;
        let got = metric.axis_min_dist2(x, lo, hi, 0).unwrap();
        let want = brute_axis_min_dist2(x, lo, hi, Some(l));
        assert!(got <= want + 1.0e-9);
        if width >= l {
            assert_eq!(got, 0.0);
        }

        let p = [x, rng.random_range(-8.0..8.0)];
        let lo2 = [lo, -2.0];
        let hi2 = [hi, 4.0];
        let lb = metric.aabb_min_dist2(&p, &lo2, &hi2).unwrap();
        let mut min_seen = f64::INFINITY;
        for a in 0..80 {
            for b in 0..40 {
                let q = [
                    lo2[0] + (hi2[0] - lo2[0]) * a as f64 / 79.0,
                    lo2[1] + (hi2[1] - lo2[1]) * b as f64 / 39.0,
                ];
                min_seen = min_seen.min(metric.dist2(&p, &q).unwrap());
            }
        }
        assert!(lb <= min_seen + 1.0e-9);
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
/// a 2 MiB stack. The build and the queries run on a 256 KiB stack.
#[test]
fn large_duplicate_set_builds_on_a_small_stack() {
    const N: usize = 20_000;
    let built = std::thread::Builder::new()
        .name("nd-small-stack".into())
        .stack_size(256 * 1024)
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
