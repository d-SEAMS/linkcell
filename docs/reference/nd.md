# N-dimensional neighbour index and DBSCAN

`linkcell::VpTreeIndex` provides exact N-dimensional neighbour search over
row-major `f64` feature buffers (`n x d`) with optional per-axis periodicity.
The API is generic over a metric trait (`linkcell::Metric`) implemented
currently by `linkcell::PeriodicSpace` (with `PeriodicEuclidean` as a
compatibility alias).

The implementation is intentionally split so that metric and bound logic sits
behind a single shim module (`src/nd/metric.rs`), which is planned to be
replaced by the upcoming upstream minimage N-dimensional metric implementation.

## Why this tree

This index targets feature-space clustering workloads (`d ~= 2..50`,
`n` up to millions) where periodic and non-periodic axes are mixed.
Axis-aligned kd-tree pruning degrades as dimension rises, and its bounds are
awkward on toroidal axes.

A metric tree keeps pruning in metric space and directly supports periodic
minimum-image distance through one metric implementation.
The current choice is a vantage-point tree (Yianilos, SODA 1993): exact search
with branch-and-bound from the triangle inequality.

For `d` in the tens, distance concentration weakens all exact tree pruning,
including VP-trees. The exact worst case remains linear scan per query.
In this crate, VP-trees are still preferred over kd-trees for mixed periodic
and open axes because VP pruning uses only the metric, while kd pruning relies
on axis-aligned bounds that become loose in high `d` and awkward on toroidal
dimensions.

Related work considered:

- Friedman, Bentley, Finkel (ACM TOMS 1977): kd-tree nearest neighbour.
- Omohundro (ICSI TR-89-063): ball trees and variants.
- Beygelzimer, Kakade, Langford (ICML 2006): cover trees.
- Arya, Mount et al. (SODA 1993; JACM 1998): ANN via balanced box-decomposition.
- scikit-learn / FLANN / nanoflann design notes (practical high-dimensional
  trade-offs between exact and approximate trees).

The present API is exact by default; approximate search with explicit epsilon
is not enabled yet.

## Space and metric

- `OwnedContiguousSpace` / `ContiguousSpace` read row-major buffers.
- `PeriodicSpace` supports per-axis `Option<f64>` periods:
  - `Some(L)`: periodic axis with period `L`.
  - `None`: open axis.
- Wrapping follows minimage/linkcell floor-fast conventions (`trunc`-style
  fold semantics), including coordinates outside `[0, L)`.
- Periodic deltas wrap to `[-L/2, L/2)`, with the tie at `+L/2` mapped to
  `-L/2`.
- The torus minimum-image distance on periodic axes is a metric, and the
  Cartesian product of periodic and open axes with Euclidean aggregation is
  therefore also a metric; triangle-inequality pruning in the VP-tree is sound.

Metric trait functions:

- `dist2(a, b)`
- `axis_min_dist2(x, lo, hi, axis)`
- `aabb_min_dist2(point, lo, hi)`

The axis and box lower bounds are sound for periodic intervals including
intervals wider than `L/2` and full-period coverage (`width >= L`), where the
bound is exactly zero.

## Public Rust API

```rust
pub trait NeighbourIndex {
    fn len(&self) -> usize;
    fn dims(&self) -> usize;
    fn knn(&self, query: &[f64], k: usize) -> Result<Vec<NeighborHit>, Error>;
    fn within(&self, query: &[f64], radius: f64) -> Result<Vec<NeighborHit>, Error>;
    fn within_index(&self, index: usize, radius: f64) -> Result<Vec<NeighborHit>, Error>;
    fn pairs_within(&self, radius: f64) -> Result<Vec<PairHit>, Error>;
}
```

Results are deterministic:

- `knn` and `within` sort by `(dist2, index)`.
- Ties resolve by index.
- Pair output uses `i < j`.

## DBSCAN

`VpTreeIndex::dbscan(eps, min_pts)` implements deterministic DBSCAN
(Ester et al., KDD 1996) on top of radius queries.

- `min_pts` includes the point itself.
- Noise points get label `-1`.
- `core` flags mark core points.
- Border assignment rule: a border point keeps the first cluster that reaches
  it while seeds are scanned in ascending index order and each expansion
  visits neighbours in ascending index order.

HDBSCAN is out of scope in this crate.

## C ABI index width

The current `lc_nd_*` neighbour and pair outputs use `int` indices, matching
the existing `lc_knearest` and pair-list ABI conventions in linkcell.
Inputs whose indices would exceed `INT_MAX` fail with an error code rather than
silently truncating.
`lc_nd_build` uses `periods[i] == 0.0` for open axes and `periods[i] > 0.0` for
periodic axes; negative or NaN period entries are rejected.
