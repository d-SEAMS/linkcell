use std::ops::Deref;

use linkcell::{ContiguousSpace, Error, Metric, NeighbourIndex, PeriodicSpace, Space, VpTreeIndex};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;

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

/// `PeriodicSpace` and its periods. The periods give exact rational squares
/// that do not go through the crate.
struct Oracle {
    metric: PeriodicSpace,
    periods: Vec<Option<f64>>,
    /// Pairs whose overflow was checked against the rational square.
    overflowed: std::cell::RefCell<std::collections::HashSet<Vec<u64>>>,
}

impl Oracle {
    fn new(periods: &[Option<f64>]) -> Self {
        Self {
            metric: PeriodicSpace::new(periods.to_vec()).unwrap(),
            periods: periods.to_vec(),
            overflowed: Default::default(),
        }
    }

    /// Exact squared distance from rationals.
    fn exact(&self, a: &[f64], b: &[f64]) -> BigRational {
        let mut sum = BigRational::from_integer(0.into());
        for (k, period) in self.periods.iter().enumerate() {
            sum += match period {
                Some(l) => rational::periodic_sq(a[k], b[k], *l),
                None => {
                    let gap = rational::rat_from_f64(a[k]) - rational::rat_from_f64(b[k]);
                    &gap * &gap
                }
            };
        }
        sum
    }

    /// `Metric::dist2`. A failure must be an exact square at or above
    /// `2^1024 - 2^970`, the least value that rounds to infinity.
    fn dist2(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        let got = self.metric.dist2(a, b);
        let key = || a.iter().chain(b).map(|v| v.to_bits()).collect();
        if let Err(err) = &got {
            assert_eq!(*err, Error::DistanceOverflow);
            if !self.overflowed.borrow_mut().insert(key()) {
                return got;
            }
            let floor = BigRational::from_integer((BigInt::one() << 1024) - (BigInt::one() << 970));
            assert!(
                self.exact(a, b) >= floor,
                "{a:?} {b:?} failed below the overflow floor"
            );
        }
        got
    }

    /// `d2 <= r2`, except that a `d2` of 0 from a point that does not
    /// coincide with `q` is outside when `r2` is 0. The exact square comes
    /// from `Metric::dist2_exact`, which [`exact_squares_order_like_rationals`]
    /// checks against rationals.
    fn inside(&self, p: &[f64], q: &[f64], d2: f64, r2: f64) -> bool {
        d2 < r2 || (d2 == r2 && (r2 > 0.0 || self.metric.dist2_exact(p, q).unwrap().is_zero()))
    }

    /// Sort by exact square, then index. Only a run of equal rounded values
    /// needs the exact square.
    fn rank(&self, data: &[f64], d: usize, q: &[f64], hits: &mut [(usize, f64)]) {
        hits.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let mut start = 0;
        while start < hits.len() {
            let tied = hits[start..]
                .iter()
                .take_while(|h| h.1 == hits[start].1)
                .count();
            if tied > 1 {
                let run = &mut hits[start..start + tied];
                let mut keyed: Vec<_> = run
                    .iter()
                    .map(|&(i, d2)| {
                        let exact = self.metric.dist2_exact(&data[i * d..(i + 1) * d], q);
                        (exact.unwrap(), i, d2)
                    })
                    .collect();
                keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                for (slot, (_, i, d2)) in run.iter_mut().zip(keyed) {
                    *slot = (i, d2);
                }
            }
            start += tied;
        }
    }
}

impl Deref for Oracle {
    type Target = PeriodicSpace;

    fn deref(&self) -> &PeriodicSpace {
        &self.metric
    }
}

/// Full scan: every distance from `q` is read, and any failure fails the
/// query. `k = 0` reads none.
fn scan_knn(
    metric: &Oracle,
    data: &[f64],
    n: usize,
    d: usize,
    q: &[f64],
    k: usize,
) -> Result<Vec<(usize, f64)>, Error> {
    if k == 0 {
        return Ok(Vec::new());
    }
    let mut hits = Vec::with_capacity(n);
    for i in 0..n {
        hits.push((i, metric.dist2(&data[i * d..(i + 1) * d], q)?));
    }
    metric.rank(data, d, q, &mut hits);
    hits.truncate(k);
    Ok(hits)
}

fn scan_within(
    metric: &Oracle,
    data: &[f64],
    n: usize,
    d: usize,
    q: &[f64],
    r: f64,
) -> Result<Vec<(usize, f64)>, Error> {
    let r2 = r * r;
    let mut hits = Vec::new();
    for i in 0..n {
        let p = &data[i * d..(i + 1) * d];
        let d2 = metric.dist2(p, q)?;
        if metric.inside(p, q, d2, r2) {
            hits.push((i, d2));
        }
    }
    metric.rank(data, d, q, &mut hits);
    Ok(hits)
}

fn scan_pairs(
    metric: &Oracle,
    data: &[f64],
    n: usize,
    d: usize,
    radius: f64,
) -> Result<Vec<(usize, usize, f64)>, Error> {
    let r2 = radius * radius;
    let mut out = Vec::new();
    for i in 0..n {
        let pi = &data[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let pj = &data[j * d..(j + 1) * d];
            let d2 = metric.dist2(pi, pj)?;
            if metric.inside(pi, pj, d2, r2) {
                out.push((i, j, d2));
            }
        }
    }
    Ok(out)
}

/// DBSCAN asks for the neighbours of every point, so it reads every pair.
fn scan_dbscan(
    metric: &Oracle,
    data: &[f64],
    n: usize,
    d: usize,
    eps: f64,
    min_pts: usize,
) -> Result<Vec<i32>, Error> {
    let r2 = eps * eps;
    let mut neighbours: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    for i in 0..n {
        let pi = &data[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let pj = &data[j * d..(j + 1) * d];
            let d2 = metric.dist2(pi, pj)?;
            if metric.inside(pi, pj, d2, r2) {
                neighbours[i].push(j);
                neighbours[j].push(i);
            }
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
    }
    const UNASSIGNED: i32 = -2;
    const NOISE: i32 = -1;
    let mut labels = vec![UNASSIGNED; n];
    let mut cluster = 0i32;
    for i in 0..n {
        if labels[i] != UNASSIGNED {
            continue;
        }
        if neighbours[i].len() < min_pts {
            labels[i] = NOISE;
            continue;
        }
        labels[i] = cluster;
        let mut queued = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        for &j in &neighbours[i] {
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
            if neighbours[p].len() >= min_pts {
                for &j in &neighbours[p] {
                    if !queued[j] {
                        queued[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        cluster += 1;
    }
    Ok(labels
        .into_iter()
        .map(|v| if v == UNASSIGNED { -1 } else { v })
        .collect())
}

fn brute_knn(
    metric: &Oracle,
    data: &[f64],
    n: usize,
    d: usize,
    q: &[f64],
    k: usize,
) -> Vec<(usize, f64)> {
    scan_knn(metric, data, n, d, q, k).unwrap()
}

fn brute_within(
    metric: &Oracle,
    data: &[f64],
    n: usize,
    d: usize,
    q: &[f64],
    r: f64,
) -> Vec<(usize, f64)> {
    scan_within(metric, data, n, d, q, r).unwrap()
}

fn brute_dbscan(
    metric: &Oracle,
    data: &[f64],
    n: usize,
    d: usize,
    eps: f64,
    min_pts: usize,
) -> Vec<i32> {
    scan_dbscan(metric, data, n, d, eps, min_pts).unwrap()
}

fn brute_pairs_within(
    metric: &Oracle,
    data: &[f64],
    n: usize,
    d: usize,
    radius: f64,
) -> Vec<(usize, usize, f64)> {
    scan_pairs(metric, data, n, d, radius).unwrap()
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
        let metric = Oracle::new(&periods);
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
            let metric = Oracle::new(&periods);
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
        let metric = Oracle::new(&periods);
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
    let metric = Oracle::new(&periods);
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
    let metric = Oracle::new(&periods);
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
        let metric = Oracle::new(&periods);
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
    let metric = Oracle::new(&periods);
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
    let metric3 = Oracle::new(&periods3);
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
    let metric = Oracle::new(&periods);
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
    let metric = Oracle::new(&periods);
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

/// Results of every query in [`large_duplicate_set_builds_on_a_small_stack`].
struct SmallStackRun {
    dup_knn: Vec<linkcell::NeighborHit>,
    dup_within: Vec<linkcell::NeighborHit>,
    image_knn: Vec<linkcell::NeighborHit>,
    image_within: Vec<Vec<linkcell::NeighborHit>>,
    knn: Vec<Result<Vec<linkcell::NeighborHit>, Error>>,
    within: Vec<Result<Vec<linkcell::NeighborHit>, Error>>,
    within_index: Vec<linkcell::NeighborHit>,
    pairs: [Result<Vec<linkcell::PairHit>, Error>; 2],
    dbscan: [Result<Vec<i32>, Error>; 2],
}

/// 20_000 identical points used to recurse one level per point and overflow
/// a 2 MiB stack. Every build and query runs on a 32 KiB stack, in the debug
/// profile too. The full scans they are checked against run on the test
/// thread.
#[test]
fn large_duplicate_set_builds_on_a_small_stack() {
    const N: usize = 20_000;
    let dup = vec![0.125f64; N];
    let l = 2.0 * std::f64::consts::PI;
    let base = -5.675156211725825;
    let images: Vec<f64> = (0..N)
        .map(|i| base + f64::from((i as i32) - 10_000) * l)
        .collect();
    let radii = [0.0, f64::from_bits(1), 1.0e-15, 5.4225756419539195e-3];
    // A cluster with duplicates, then two points whose gap overflows.
    let mut rng = StdRng::seed_from_u64(0x32_1024);
    let mut mixed: Vec<f64> = (0..300 * 2).map(|_| rng.random_range(0.0..1.0)).collect();
    for _ in 0..50 {
        mixed.extend_from_slice(&[mixed[0], mixed[1]]);
    }
    let finite = mixed.len() / 2;
    mixed.extend_from_slice(&[huge(), 0.5, -0.5 * huge(), 0.5]);
    let mixed_periods = [None, Some(1.0)];
    let queries = [vec![mixed[0], mixed[1]], vec![0.5, 0.5], vec![huge(), 0.5]];

    let run = {
        let (dup, images, mixed, queries) =
            (dup.clone(), images.clone(), mixed.clone(), queries.clone());
        std::thread::Builder::new()
            .name("nd-small-stack".into())
            .stack_size(32 * 1024)
            .spawn(move || {
                let idx = build_index(&dup, N, 1, &[Some(1.0)]);
                let dup_knn = idx.knn(&[0.125], 8).unwrap();
                let dup_within = idx.within(&[0.125], 0.0).unwrap();
                let idx = build_index(&images, N, 1, &[Some(l)]);
                let image_knn = idx.knn(&[base], 5).unwrap();
                let image_within = radii
                    .iter()
                    .map(|&r| idx.within(&[base], r).unwrap())
                    .collect();
                let all = build_index(&mixed, mixed.len() / 2, 2, &mixed_periods);
                let some = build_index(&mixed[..finite * 2], finite, 2, &mixed_periods);
                let mut knn = Vec::new();
                let mut within = Vec::new();
                for q in &queries {
                    for k in [0, 1, 60] {
                        knn.push(all.knn(q, k));
                    }
                    for r in [0.0, 0.05] {
                        within.push(all.within(q, r));
                    }
                }
                SmallStackRun {
                    dup_knn,
                    dup_within,
                    image_knn,
                    image_within,
                    knn,
                    within,
                    within_index: some.within_index(0, 0.05).unwrap(),
                    pairs: [all.pairs_within(0.05), some.pairs_within(0.05)],
                    dbscan: [
                        all.dbscan(0.05, 4).map(|res| res.labels),
                        some.dbscan(0.05, 4).map(|res| res.labels),
                    ],
                }
            })
            .expect("spawn small stack")
            .join()
            .expect("small-stack query panicked")
    };

    let metric = Oracle::new(&[Some(1.0)]);
    let want = brute_knn(&metric, &dup, N, 1, &[0.125], 8);
    assert_hit_pairs(&run.dup_knn, &want, "identical knn");
    let want = brute_within(&metric, &dup, N, 1, &[0.125], 0.0);
    assert_eq!(want.len(), N);
    assert_hit_pairs(&run.dup_within, &want, "identical within 0");

    let metric = Oracle::new(&[Some(l)]);
    let want = brute_knn(&metric, &images, N, 1, &[base], 5);
    assert_hit_pairs(&run.image_knn, &want, "image knn");
    for (got, &r) in run.image_within.iter().zip(&radii) {
        let want = brute_within(&metric, &images, N, 1, &[base], r);
        assert_hit_pairs(got, &want, &format!("image within r={r}"));
    }

    let metric = Oracle::new(&mixed_periods);
    let n = mixed.len() / 2;
    let mut knn = run.knn.into_iter();
    let mut within = run.within.into_iter();
    for q in &queries {
        for k in [0, 1, 60] {
            let want = scan_bits(scan_knn(&metric, &mixed, n, 2, q, k));
            assert_eq!(hit_bits(knn.next().unwrap()), want, "knn({q:?}, {k})");
        }
        for r in [0.0, 0.05] {
            let want = scan_bits(scan_within(&metric, &mixed, n, 2, q, r));
            assert_eq!(hit_bits(within.next().unwrap()), want, "within({q:?}, {r})");
        }
    }
    let want = brute_within(&metric, &mixed[..finite * 2], finite, 2, &mixed[..2], 0.05);
    assert_hit_pairs(&run.within_index, &want, "within_index");
    assert_eq!(run.pairs[0].as_ref().unwrap_err(), &Error::DistanceOverflow);
    let want = brute_pairs_within(&metric, &mixed[..finite * 2], finite, 2, 0.05);
    let got: Vec<_> = run.pairs[1]
        .as_ref()
        .unwrap()
        .iter()
        .map(|h| (h.i, h.j, h.dist2))
        .collect();
    assert_eq!(got, want);
    assert_eq!(run.dbscan[0], Err(Error::DistanceOverflow));
    let want = brute_dbscan(&metric, &mixed[..finite * 2], finite, 2, 0.05, 4);
    assert_eq!(run.dbscan[1].as_ref().unwrap(), &want);
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
    let oracle = Oracle::new(periods);
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
        let ranked = |hits: &mut Vec<(usize, f64)>| {
            hits.sort_by(|a, b| {
                let pa = &data[a.0 * d..(a.0 + 1) * d];
                let pb = &data[b.0 * d..(b.0 + 1) * d];
                oracle
                    .exact(pa, q)
                    .cmp(&oracle.exact(pb, q))
                    .then_with(|| a.0.cmp(&b.0))
            });
        };
        for &r in radii {
            let exact_r2 = rounded_square(from_f64(r).0, from_f64(r).1);
            assert_same_bits(r * r, exact_r2, "radius square");
            let mut want = Vec::new();
            for i in 0..n {
                let p = &data[i * d..(i + 1) * d];
                if oracle.inside(p, q, dist(p, q), exact_r2) {
                    want.push((i, dist(p, q)));
                }
            }
            ranked(&mut want);
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
                ranked(&mut all);
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
                    let pi = &data[i * d..(i + 1) * d];
                    let pj = &data[j * d..(j + 1) * d];
                    if oracle.inside(pi, pj, dist(pi, pj), exact_r2) {
                        want.push((i, j, dist(pi, pj)));
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
    assert!(f1_idx.within(&f1q, 0.0).unwrap().is_empty());
    // Both squares round to 0. Point 1 is 1.3e-162 from the query and point 0
    // is 1.5e-162, so point 1 comes first despite its larger index.
    let f1_knn: Vec<_> = f1_idx
        .knn(&f1q, 2)
        .unwrap()
        .into_iter()
        .map(|h| (h.index, h.dist2))
        .collect();
    assert_eq!(f1_knn, vec![(1, 0.0), (0, 0.0)]);

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
    // The product does not fit, so `new` must not ask for a point.
    struct ProductOverflow {
        n: usize,
        d: usize,
    }
    impl Space for ProductOverflow {
        fn len(&self) -> usize {
            self.n
        }
        fn dims(&self) -> usize {
            self.d
        }
        fn point(&self, _index: usize) -> &[f64] {
            panic!("n * d overflow must not read a point");
        }
    }
    let built = VpTreeIndex::new(
        ProductOverflow {
            n: usize::MAX / 2 + 1,
            d: 2,
        },
        PeriodicSpace::new(vec![None, None]).unwrap(),
    );
    let Err(err) = built else {
        panic!("n * d overflow built an index");
    };
    assert_eq!(err, Error::NdProduct);
    assert_eq!(err.to_string(), "n * d overflows");
    assert_ne!(err.to_string(), Error::Overflow.to_string());
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
    let (below, above) = (4.999999999999974e-1, 5.000000000000074e-1);
    let near_half = unit.dist2(&[below], &[above]).unwrap();
    assert_eq!(
        near_half.to_bits(),
        1.0095262545291237e-28_f64.to_bits(),
        "{near_half:e}"
    );
    assert!(rational::is_nearest(
        near_half,
        &rational::periodic_sq(below, above, 1.0)
    ));

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

/// Largest value whose square is finite: `x^2` rounds to `f64::MAX`, and any
/// gap of `1.01 * x` or more overflows.
fn huge() -> f64 {
    f64::from_bits(0x5fef_ffff_ffff_ffff)
}

/// Every ordering of `0..n`.
fn orderings(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in orderings(n - 1) {
        for slot in 0..n {
            let mut order = rest.clone();
            order.insert(slot, n - 1);
            out.push(order);
        }
    }
    out
}

fn hit_bits(hits: Result<Vec<linkcell::NeighborHit>, Error>) -> Result<Vec<(usize, u64)>, Error> {
    hits.map(|hits| hits.iter().map(|h| (h.index, h.dist2.to_bits())).collect())
}

fn scan_bits(hits: Result<Vec<(usize, f64)>, Error>) -> Result<Vec<(usize, u64)>, Error> {
    hits.map(|hits| hits.iter().map(|&(i, d2)| (i, d2.to_bits())).collect())
}

/// Builds `data` and compares every query with a full scan, errors included.
/// Returns how many queries failed and how many succeeded.
fn assert_full_scan(
    oracle: &Oracle,
    data: &[f64],
    d: usize,
    queries: &[Vec<f64>],
    radii: &[f64],
    pair_radii: &[f64],
    what: &str,
) -> [usize; 2] {
    let n = data.len() / d;
    let idx = build_index(data, n, d, &oracle.periods);
    let mut seen = [0usize; 2];
    let mut tally = |ok: bool| seen[usize::from(ok)] += 1;
    for q in queries {
        for k in [0, 1, 2, n + 1] {
            let want = scan_knn(oracle, data, n, d, q, k);
            tally(want.is_ok());
            assert_eq!(
                hit_bits(idx.knn(q, k)),
                scan_bits(want),
                "{what}: knn({q:?}, {k})"
            );
        }
        for &r in radii {
            let want = scan_within(oracle, data, n, d, q, r);
            tally(want.is_ok());
            assert_eq!(
                hit_bits(idx.within(q, r)),
                scan_bits(want),
                "{what}: within({q:?}, {r:e})"
            );
        }
    }
    for &r in pair_radii {
        let want = scan_pairs(oracle, data, n, d, r).map(|rows| {
            rows.iter()
                .map(|&(i, j, d2)| (i, j, d2.to_bits()))
                .collect::<Vec<_>>()
        });
        tally(want.is_ok());
        let got = idx.pairs_within(r).map(|rows| {
            rows.iter()
                .map(|h| (h.i, h.j, h.dist2.to_bits()))
                .collect::<Vec<_>>()
        });
        assert_eq!(got, want, "{what}: pairs_within({r:e})");
    }
    for &eps in pair_radii.iter().filter(|&&r| r > 0.0) {
        for min_pts in [1, 2] {
            let want = scan_dbscan(oracle, data, n, d, eps, min_pts);
            tally(want.is_ok());
            assert_eq!(
                idx.dbscan(eps, min_pts).map(|res| res.labels),
                want,
                "{what}: dbscan({eps:e}, {min_pts})"
            );
        }
    }
    seen
}

/// The gate's repros. Pruning skipped the overflowing pair, so these returned
/// `Ok` in some build orders and failed to build in others. Overflow sits on
/// the last axis; the other axes hold small offsets.
#[test]
fn overflow_repros_fail_in_every_order() {
    let x = huge();
    let open = Oracle::new(&[None]);
    assert_eq!(open.dist2(&[-0.5 * x], &[x]), Err(Error::DistanceOverflow));
    assert!(open.dist2(&[-0.5 * x], &[0.0]).unwrap().is_finite());
    assert!(open.dist2(&[x], &[0.0]).unwrap().is_finite());
    for axis in [None, Some(8.0 * x), Some(4.0 * x), Some(3.5 * x)] {
        for d in 1..=3 {
            let mut periods = vec![None; d];
            periods[d - 1] = axis;
            let lift = |v: f64, i: usize| -> Vec<f64> {
                let mut row: Vec<f64> = (0..d).map(|k| (i + k) as f64 * 0.125).collect();
                row[d - 1] = v;
                row
            };
            let three = [x, -0.5 * x, 0.0];
            let q = lift(-0.5 * x, 1);
            for order in orderings(3) {
                let data: Vec<f64> = order.iter().flat_map(|&i| lift(three[i], i)).collect();
                let what = format!("{periods:?} {order:?}");
                let idx = build_index(&data, 3, d, &periods);
                for k in [1, 2] {
                    assert_eq!(
                        idx.knn(&q, k).unwrap_err(),
                        Error::DistanceOverflow,
                        "{what}"
                    );
                }
                for r in [0.0, 1.0e-300, 1.0, 1.0e100, 1.0e153] {
                    assert_eq!(
                        idx.within(&q, r).unwrap_err(),
                        Error::DistanceOverflow,
                        "{what} {r:e}"
                    );
                }
                assert!(idx.within(&lift(0.0, 2), 1.0).is_ok(), "{what}");
                assert_eq!(idx.pairs_within(1.0).unwrap_err(), Error::DistanceOverflow);
                assert_eq!(idx.dbscan(1.0, 1).unwrap_err(), Error::DistanceOverflow);
            }
            let four = [x, -0.1 * x, 0.3 * x, -1.0];
            for order in orderings(4) {
                let data: Vec<f64> = order.iter().flat_map(|&i| lift(four[i], i)).collect();
                let idx = build_index(&data, 4, d, &periods);
                let what = format!("{periods:?} {order:?}");
                assert_eq!(
                    idx.pairs_within(1.0).unwrap_err(),
                    Error::DistanceOverflow,
                    "{what}"
                );
                assert_eq!(
                    idx.dbscan(1.0, 1).unwrap_err(),
                    Error::DistanceOverflow,
                    "{what}"
                );
            }
        }
    }
}

/// Rounded squares that tie are ordered by the exact square. A radius whose
/// square is 0 admits only coincident points, and any other radius admits
/// every point with `dist2 <= fl(r * r)`.
#[test]
fn rounded_ties_order_by_the_exact_square() {
    let tiny = build_index(&[1.0e-170, 0.0], 2, 1, &[None]);
    let pairs = |hits: Vec<linkcell::NeighborHit>| -> Vec<(usize, f64)> {
        hits.iter().map(|h| (h.index, h.dist2)).collect()
    };
    assert_eq!(pairs(tiny.knn(&[0.0], 1).unwrap()), vec![(1, 0.0)]);
    assert_eq!(
        pairs(tiny.knn(&[0.0], 2).unwrap()),
        vec![(1, 0.0), (0, 0.0)]
    );
    // `fl(1e-170 * 1e-170)` is 0, so a radius of the gap itself admits only
    // the coincident point.
    for r in [0.0, 1.0e-300, 1.0e-170] {
        assert_eq!(
            pairs(tiny.within(&[0.0], r).unwrap()),
            vec![(1, 0.0)],
            "{r:e}"
        );
    }
    assert!(tiny.pairs_within(0.0).unwrap().is_empty());
    assert_eq!(tiny.dbscan(1.0e-170, 2).unwrap().clusters, 0);

    let images = [0.0, 1.0, 2.0f64.powi(-600), -0.0, 3.0];
    let ring = build_index(&images, 5, 1, &[Some(1.0)]);
    assert_eq!(
        pairs(ring.within(&[0.0], 0.0).unwrap()),
        vec![(0, 0.0), (1, 0.0), (3, 0.0), (4, 0.0)]
    );
    assert_eq!(pairs(ring.knn(&[0.0], 5).unwrap())[4], (2, 0.0));

    // `(1, 2^-30)` has the exact square `1 + 2^-60`, which rounds to 1.
    let plane = [1.0, 2.0f64.powi(-30), 1.0, 0.0, 0.0, 0.0];
    let idx = build_index(&plane, 3, 2, &[None, None]);
    assert_eq!(
        pairs(idx.within(&[0.0, 0.0], 1.0).unwrap()),
        vec![(2, 0.0), (1, 1.0), (0, 1.0)]
    );
    assert_eq!(
        pairs(idx.within(&[0.0, 0.0], 1.0 - f64::EPSILON / 2.0).unwrap()),
        vec![(2, 0.0)]
    );
    assert_eq!(
        pairs(idx.knn(&[0.0, 0.0], 3).unwrap()),
        vec![(2, 0.0), (1, 1.0), (0, 1.0)]
    );
    let rows: Vec<_> = idx
        .pairs_within(1.0)
        .unwrap()
        .iter()
        .map(|h| (h.i, h.j, h.dist2))
        .collect();
    assert_eq!(
        rows,
        vec![(0, 1, 2.0f64.powi(-60)), (0, 2, 1.0), (1, 2, 1.0)]
    );
}

/// `dist2_exact` orders squares as the rationals do, also where the rounded
/// squares tie, and places each rounded square on the right side of its
/// exact value.
#[test]
fn exact_squares_order_like_rationals() {
    let values = [
        0.0,
        1.0,
        -1.0,
        3.0,
        1.0 + f64::EPSILON,
        2.0f64.powi(-30),
        -(2.0f64.powi(-30)),
        1.0e-170,
        2.0f64.powi(-600),
        f64::from_bits(1),
    ];
    let mut rng = StdRng::seed_from_u64(0xe4ac7);
    let mut ties = 0usize;
    for case in 0..600 {
        let d = 1 + case % 3;
        let periods: Vec<Option<f64>> = (0..d)
            .map(|axis| {
                if (case / 3 + axis) % 2 == 0 {
                    None
                } else {
                    Some(4.0)
                }
            })
            .collect();
        let oracle = Oracle::new(&periods);
        let mut point = || -> Vec<f64> {
            (0..d)
                .map(|_| values[rng.random_range(0..values.len())])
                .collect()
        };
        let (a, b, q) = (point(), point(), point());
        let (ea, eb) = (
            oracle.dist2_exact(&a, &q).unwrap(),
            oracle.dist2_exact(&b, &q).unwrap(),
        );
        let (ra, rb) = (oracle.exact(&a, &q), oracle.exact(&b, &q));
        assert_eq!(ea.cmp(&eb), ra.cmp(&rb), "{periods:?} {a:?} {b:?} {q:?}");
        let rounded = oracle.dist2(&a, &q).unwrap();
        let back = linkcell::ExactDist2::from_f64(rounded).unwrap();
        assert_eq!(
            back.cmp(&ea),
            rational::rat_from_f64(rounded).cmp(&ra),
            "{periods:?} {a:?} {q:?}"
        );
        if rounded == oracle.dist2(&b, &q).unwrap() && ra != rb {
            ties += 1;
        }
    }
    assert!(ties > 0);
    assert_eq!(linkcell::ExactDist2::from_f64(-1.0), None);
    assert_eq!(linkcell::ExactDist2::from_f64(f64::INFINITY), None);
}

/// Every query on every ordering of small sets matches a full scan. The
/// values put squares at the top of the binary64 range beside tiny ones, so
/// some pairs overflow and some rounded squares tie.
#[test]
fn every_query_matches_a_full_scan_in_every_order() {
    let x = huge();
    let palette = [
        x,
        -x,
        0.5 * x,
        -0.5 * x,
        -0.1 * x,
        0.3 * x,
        0.75 * x,
        1.0e153,
        -1.0e153,
        1.0,
        -1.0,
        0.0,
        1.0e-170,
        -1.0e-170,
        2.0f64.powi(-600),
    ];
    let axes = [None, Some(8.0 * x), Some(4.0 * x), Some(3.5 * x), Some(4.0)];
    let radii = [0.0, 1.0e-300, 1.0e-170, 1.0, 1.0e100, 1.0e153, 1.0e300];
    let mut rng = StdRng::seed_from_u64(0x0f10_0a11);
    let mut seen = [0usize; 2];
    for case in 0..150 {
        let d = 1 + case % 3;
        let n = if case % 25 == 24 {
            5
        } else {
            2 + (case / 3) % 3
        };
        let periods: Vec<Option<f64>> = (0..d)
            .map(|_| axes[rng.random_range(0..axes.len())])
            .collect();
        let oracle = Oracle::new(&periods);
        let mut row = || -> Vec<f64> {
            (0..d)
                .map(|_| palette[rng.random_range(0..palette.len())])
                .collect()
        };
        let rows: Vec<Vec<f64>> = (0..n).map(|_| row()).collect();
        let mut queries = rows.clone();
        queries.push(row());
        for order in orderings(n) {
            let data: Vec<f64> = order
                .iter()
                .flat_map(|&i| rows[i].iter().copied())
                .collect();
            let got = assert_full_scan(
                &oracle,
                &data,
                d,
                &queries,
                &radii,
                &radii,
                &format!("{periods:?} {data:?}"),
            );
            seen[0] += got[0];
            seen[1] += got[1];
        }
    }
    assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
}

/// Overflow and results match a full scan in trees deep enough to prune: a
/// cluster near the origin and a few points with one huge coordinate, in
/// sorted, reversed and shuffled build orders.
#[test]
fn overflow_is_reported_however_the_tree_prunes() {
    use rand::seq::SliceRandom;
    let x = huge();
    let far = [x, -x, -0.5 * x, 0.3 * x, -0.1 * x, 0.75 * x];
    let axes = [None, Some(8.0 * x), Some(4.0 * x), Some(3.5 * x)];
    let mut rng = StdRng::seed_from_u64(0xd33e);
    let mut seen = [0usize; 2];
    for case in 0..12 {
        let d = 1 + case % 3;
        let periods: Vec<Option<f64>> = (0..d)
            .map(|axis| axes[(case / 3 + axis) % axes.len()])
            .collect();
        let oracle = Oracle::new(&periods);
        let mut rows: Vec<Vec<f64>> = (0..80)
            .map(|_| (0..d).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        for _ in 0..1 + case % 3 {
            let mut row: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
            row[rng.random_range(0..d)] = far[rng.random_range(0..far.len())];
            rows.push(row);
        }
        let mut queries = rows[76..].to_vec();
        queries.push(vec![0.0; d]);
        let mut toward = vec![0.25; d];
        toward[0] = -0.5 * x;
        queries.push(toward);
        for round in 0..3 {
            match round {
                0 => rows.sort_by(|a, b| a[0].total_cmp(&b[0])),
                1 => rows.reverse(),
                _ => rows.shuffle(&mut rng),
            }
            let data = rows.concat();
            let got = assert_full_scan(
                &oracle,
                &data,
                d,
                &queries,
                &[0.0, 0.25, 1.0e153],
                &[0.0, 0.25, 1.0e153],
                &format!("case {case} round {round}"),
            );
            seen[0] += got[0];
            seen[1] += got[1];
        }
    }
    assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
}

/// `PeriodicSpace` that counts the distances a query reads.
struct Counted {
    inner: PeriodicSpace,
    reads: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Counted {
    fn read(&self) {
        self.reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Metric for Counted {
    fn dims(&self) -> usize {
        self.inner.dims()
    }
    fn dist2(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        self.read();
        self.inner.dist2(a, b)
    }
    fn dist2_exact(&self, a: &[f64], b: &[f64]) -> Result<linkcell::ExactDist2, Error> {
        self.read();
        self.inner.dist2_exact(a, b)
    }
    fn distance(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        self.inner.distance(a, b)
    }
    fn axis_min_dist2(&self, x: f64, lo: f64, hi: f64, axis: usize) -> Result<f64, Error> {
        self.inner.axis_min_dist2(x, lo, hi, axis)
    }
    fn aabb_min_dist2(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        self.inner.aabb_min_dist2(p, lo, hi)
    }
    fn box_dist2(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        self.inner.box_dist2(p, lo, hi)
    }
    fn box_distance(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        self.inner.box_distance(p, lo, hi)
    }
    fn dist2_batch(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.inner.dist2_batch(a, b, out)
    }
    fn distance_batch(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.inner.distance_batch(a, b, out)
    }
    fn write_canonical(&self, src: &[f64], dst: &mut [f64]) -> Result<(), Error> {
        self.inner.write_canonical(src, dst)
    }
    fn vp_absolute_slack(&self) -> f64 {
        self.inner.vp_absolute_slack()
    }
}

/// `knn(q, 0)` reads no distance. `within(q, 0)` reads about one
/// root-to-leaf path; the bound is four distances per level of a balanced
/// tree. The set holds duplicates and points whose squares underflow.
#[test]
fn zero_k_and_zero_radius_do_not_walk_the_tree() {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let mut rng = StdRng::seed_from_u64(0x0e0);
    for (d, periods) in [
        (1, vec![None]),
        (2, vec![None, Some(1.0)]),
        (3, vec![Some(1.0); 3]),
    ] {
        let n = 4096usize;
        let mut data: Vec<f64> = (0..n * d).map(|_| rng.random_range(0.0..1.0)).collect();
        for i in 1..9 {
            data.copy_within(0..d, i * d);
        }
        for (i, v) in [0.0, 1.0e-170, 2.0f64.powi(-600)].into_iter().enumerate() {
            let row = &mut data[(9 + i) * d..(10 + i) * d];
            row.fill(0.0);
            row[0] = v;
        }
        let reads = std::sync::Arc::new(AtomicUsize::new(0));
        let metric = Counted {
            inner: PeriodicSpace::new(periods.clone()).unwrap(),
            reads: reads.clone(),
        };
        let idx = VpTreeIndex::new(
            ContiguousSpace::from_row_major(&data, n, d).unwrap(),
            metric,
        )
        .unwrap();
        let oracle = Oracle::new(&periods);
        let levels = (usize::BITS - n.leading_zeros()) as usize;
        let mut queries: Vec<Vec<f64>> =
            (0..16).map(|i| data[i * d..(i + 1) * d].to_vec()).collect();
        queries.extend((0..16).map(|_| (0..d).map(|_| rng.random_range(0.0..1.0)).collect()));
        for q in &queries {
            reads.store(0, Relaxed);
            assert!(idx.knn(q, 0).unwrap().is_empty());
            assert_eq!(reads.swap(0, Relaxed), 0, "knn({q:?}, 0) read a distance");
            let got = idx.within(q, 0.0).unwrap();
            let read = reads.swap(0, Relaxed);
            assert_eq!(
                hit_bits(Ok(got)),
                scan_bits(scan_within(&oracle, &data, n, d, q, 0.0)),
                "within({q:?}, 0)"
            );
            assert!(read <= 4 * levels, "within({q:?}, 0) read {read} of {n}");
        }
    }
}

/// Collinear points in up to 256 dimensions, with radii at a point's own
/// distance and one ulp on either side. Each triangle test is tight on a
/// line. A prune margin of a fixed few ulps misses points here from
/// `d = 48`. Every query matches a brute-force scan over the crate's
/// `dist2`.
#[test]
fn collinear_points_in_high_dimension_match_a_brute_force_scan() {
    let mut rng = StdRng::seed_from_u64(0xc011_17ea);
    for (case, d) in [1usize, 3, 16, 48, 64, 256].into_iter().enumerate() {
        let n = if d > 16 { 96 } else { 200 };
        let dir: Vec<f64> = (0..d).map(|_| rng.random_range(0.5..2.0)).collect();
        let data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let t = i as f64 * 0.25 - 4.0;
                dir.iter().map(move |&w| t * w)
            })
            .collect();
        let periods: Vec<Option<f64>> = (0..d)
            .map(|axis| (case % 2 == 1 && axis % 2 == 0).then_some(64.0))
            .collect();
        let oracle = Oracle::new(&periods);
        let point = |i: usize| data[i * d..(i + 1) * d].to_vec();
        let queries = vec![point(0), point(n / 3), point(n - 1)];
        let mut radii = vec![0.0];
        for j in [1, 5, n / 2] {
            let r = oracle.dist2(&queries[0], &point(j)).unwrap().sqrt();
            radii.extend([next_down(r), r, next_up(r)]);
        }
        let pair_radii = [radii[2], radii[4]];
        assert_full_scan(
            &oracle,
            &data,
            d,
            &queries,
            &radii,
            &pair_radii,
            &format!("collinear d={d}"),
        );
    }
}

/// Points `i * 1e-163` have squares in the subnormal range, where one unit
/// in the last place is `2^-1074`. With `r = 27e-163`, `fl(r * r)` is that
/// unit, and the squares of the first 28 points round to at most it. The
/// sets match a brute-force scan over the crate's `dist2` on every axis
/// count.
#[test]
fn subnormal_squares_match_a_brute_force_scan() {
    let n = 64;
    let r = 27.0e-163;
    assert_eq!(r * r, f64::from_bits(1));
    let radii = [
        0.0, 1.0e-163, 15.0e-163, 16.0e-163, 27.0e-163, 28.0e-163, 40.0e-163, 1.0e-161,
    ];
    for d in [1usize, 2, 48] {
        let data: Vec<f64> = (0..n)
            .flat_map(|i| std::iter::repeat(i as f64 * 1.0e-163).take(d))
            .collect();
        let spaces = if d > 2 {
            vec![vec![None; d]]
        } else {
            vec![vec![None; d], vec![Some(1.0); d]]
        };
        for periods in spaces {
            let oracle = Oracle::new(&periods);
            let idx = build_index(&data, n, d, &periods);
            if d == 1 {
                let got: Vec<usize> = idx
                    .within(&[0.0], r)
                    .unwrap()
                    .iter()
                    .map(|h| h.index)
                    .collect();
                assert_eq!(got, (0..28).collect::<Vec<_>>());
            }
            let origin = vec![0.0; d];
            let want: Vec<usize> = (0..n)
                .filter(|&i| oracle.dist2(&data[i * d..(i + 1) * d], &origin).unwrap() <= r * r)
                .collect();
            let got: Vec<usize> = idx
                .within(&origin, r)
                .unwrap()
                .iter()
                .map(|h| h.index)
                .collect();
            assert_eq!(got, want, "d={d} {periods:?}");
            let queries: Vec<Vec<f64>> = [0usize, 5, 27, 63]
                .iter()
                .map(|&i| data[i * d..(i + 1) * d].to_vec())
                .collect();
            let pair_radii = if d > 2 { &radii[3..5] } else { &radii[..] };
            assert_full_scan(
                &oracle,
                &data,
                d,
                &queries,
                &radii,
                pair_radii,
                &format!("subnormal d={d} {periods:?}"),
            );
        }
    }
}

/// Wraps on subnormal periods: each image differs from its input by a whole
/// number of periods, a delta lands in `[-L/2, L/2)` and a point in
/// `[0, L)`. `0.5 * L` underflows for the smallest periods, so a bound
/// formed from it would misplace the tie.
#[test]
fn wraps_stay_exact_on_subnormal_periods() {
    let unit = f64::from_bits(1);
    let mut periods: Vec<f64> = (1..=40u64).map(f64::from_bits).collect();
    periods.extend([
        f64::from_bits((1 << 52) - 1),
        f64::MIN_POSITIVE,
        3.0 * f64::MIN_POSITIVE,
    ]);
    let mut xs: Vec<f64> = (-120i32..=120).map(|j| f64::from(j) * unit).collect();
    xs.extend([
        1.0,
        -1.0,
        1.0e300,
        -1.0e300,
        f64::MAX,
        -f64::MAX,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        2.5e-308,
        -2.5e-308,
    ]);
    let two = BigRational::from_integer(2.into());
    let zero = BigRational::from_integer(0.into());
    for &l in &periods {
        let m = PeriodicSpace::new(vec![Some(l)]).unwrap();
        let period = rational::rat_from_f64(l);
        let half = &period / &two;
        for &x in &xs {
            let what = format!("x = {x:e}, period {l:e}");
            let exact_x = rational::rat_from_f64(x);
            let w = m.wrap_delta_axis(x, 0).unwrap();
            let image = rational::rat_from_f64(w);
            assert!(
                ((&exact_x - &image) / &period).is_integer(),
                "wrap_delta {w:e}: {what}"
            );
            assert!(
                -half.clone() <= image && image < half,
                "wrap_delta {w:e}: {what}"
            );
            assert_eq!(
                m.dist2(&[x], &[0.0]).unwrap().to_bits(),
                (w * w).to_bits(),
                "{what}"
            );
            // On a subnormal period every value is a multiple of `2^-1074`, so
            // the representative is exact. On a normal period it is the
            // nearest binary64, or 0 when that is `L` itself.
            let p = m.wrap_point_axis(x, 0).unwrap();
            let point = rational::rat_from_f64(p);
            let shifts = (&exact_x / &period).floor();
            let exact_point = &exact_x - shifts * &period;
            if l < f64::MIN_POSITIVE {
                assert_eq!(point, exact_point, "wrap_point {p:e}: {what}");
            } else {
                assert!(
                    point == exact_point
                        || rational::is_nearest(p, &exact_point)
                        || (p == 0.0 && rational::is_nearest(l, &exact_point)),
                    "wrap_point {p:e}: {what}"
                );
            }
            assert!(zero <= point && point < period, "wrap_point {p:e}: {what}");
        }
    }
    let two_units = PeriodicSpace::new(vec![Some(2.0 * unit)]).unwrap();
    assert_eq!(two_units.wrap_delta_axis(unit, 0).unwrap(), -unit);
    assert_eq!(two_units.wrap_delta_axis(-unit, 0).unwrap(), -unit);
}
