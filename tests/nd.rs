use linkcell::{ContiguousSpace, NeighbourIndex, PeriodicEuclidean, VpTreeIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn build_index<'a>(
    data: &'a [f64],
    n: usize,
    d: usize,
    periods: &[Option<f64>],
) -> VpTreeIndex<ContiguousSpace<'a>, PeriodicEuclidean> {
    let space = ContiguousSpace::from_row_major(data, n, d).unwrap();
    let metric = PeriodicEuclidean::new(periods.to_vec()).unwrap();
    VpTreeIndex::new(space, metric).unwrap()
}

fn periodic_dist2(a: &[f64], b: &[f64], periods: &[Option<f64>]) -> f64 {
    let mut sum = 0.0;
    for k in 0..a.len() {
        let mut d = (a[k] - b[k]).abs();
        if let Some(l) = periods[k] {
            d = d.rem_euclid(l);
            d = d.min(l - d);
        }
        sum += d * d;
    }
    sum
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
