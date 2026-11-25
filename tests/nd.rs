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

fn periodic_dist2(a: &[f64], b: &[f64], periods: &[Option<f64>]) -> f64 {
    let mut sum = 0.0;
    for k in 0..a.len() {
        let mut d = (a[k] - b[k]).abs();
        if let Some(l) = periods[k] {
            d = d - l * (d / l).trunc();
            if d < 0.0 {
                d += l;
            }
            d = d.min(l - d);
        }
        sum += d * d;
    }
    sum
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
    data: &[f64],
    n: usize,
    d: usize,
    periods: &[Option<f64>],
    q: &[f64],
    k: usize,
) -> Vec<(usize, f64)> {
    let mut hits: Vec<_> = (0..n)
        .map(|i| {
            let p = &data[i * d..(i + 1) * d];
            (i, periodic_dist2(p, q, periods))
        })
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    hits.truncate(k.min(n));
    hits
}

fn brute_within(
    data: &[f64],
    n: usize,
    d: usize,
    periods: &[Option<f64>],
    q: &[f64],
    r: f64,
) -> Vec<(usize, f64)> {
    let r2 = r * r;
    let mut hits: Vec<_> = (0..n)
        .filter_map(|i| {
            let p = &data[i * d..(i + 1) * d];
            let d2 = periodic_dist2(p, q, periods);
            (d2 <= r2).then_some((i, d2))
        })
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    hits
}

fn brute_dbscan(
    data: &[f64],
    n: usize,
    d: usize,
    periods: &[Option<f64>],
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
        let mut neigh: Vec<_> = brute_within(data, n, d, periods, qi, eps)
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
            let mut pn: Vec<_> = brute_within(data, n, d, periods, qp, eps)
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
        let q = &data[..d];
        let k = rng.random_range(1..=n + 7);
        let got = idx.knn(q, k).unwrap();
        let want = brute_knn(&data, n, d, &periods, q, k);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.index, w.0);
            assert!((g.dist2 - w.1).abs() < 1e-12);
        }

        let r = rng.random_range(0.1..4.0);
        let got_w = idx.within(q, r).unwrap();
        let want_w = brute_within(&data, n, d, &periods, q, r);
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
            let query_ids = [0usize, n / 3, n - 1];
            for &qi in &query_ids {
                let q = &data[qi * d..(qi + 1) * d];
                for &k in &[0usize, 1, 4, n / 2, n + 13] {
                    let got = idx.knn(q, k).unwrap();
                    let want = brute_knn(&data, n, d, &periods, q, k);
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
        let eps = rng.random_range(0.3..2.0);
        let min_pts = rng.random_range(2..=6);
        let got = idx.dbscan(eps, min_pts).unwrap();
        let want = brute_dbscan(&data, n, d, &periods, eps, min_pts);
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
