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
- Arya, Mount et al. (SODA 1994; JACM 1998): ANN via balanced box-decomposition.

The present API is exact by default; approximate search with explicit epsilon
is not enabled yet.

## Space and metric

- `OwnedContiguousSpace` / `ContiguousSpace` read row-major buffers.
- `PeriodicSpace` supports per-axis `Option<f64>` periods:
  - `Some(L)`: periodic axis with period `L`.
  - `None`: open axis.
- Each periodic coordinate is reduced into `[0, L)` once: at index build
  for stored points, and at the start of a query. `dist2` reduces its
  arguments the same way, and a value already in `[0, L)` is left unchanged.
- The reduction is the IEEE remainder `x.rem_euclid(L)`. It is the exact
  mathematical residue whenever that residue is a binary64 number. That
  includes `2^55 ≡ 8` and `2^62 ≡ 4` (mod 10), `1` and `11` (mod 10),
  `8.25 ≡ 1.25` (mod 7), and an integer lattice modulo 360 when the
  coordinate and the period are binary64 integers. It is not claimed for a
  residue that is not a binary64 number. A remainder of `L` is the seam,
  where the mathematical residue is closer to `L` than to any smaller
  binary64 value; the representative in `[0, L)` is `0`, at most half an
  ulp of `L` from the true residue on the circle. `±L` are that seam, so
  `dist2(L, -L)` is `0` even when `L` is `1e308` and the raw difference
  overflows. Overflow is returned only when a reduced axis distance's square,
  or the sum of those squares, is non-finite.
- Periodic deltas wrap to `[-L/2, L/2)`, with the tie at `+L/2` mapped to
  `-L/2`.
- On those representatives the circular distance of one axis is within
  `4 * ulp(L)` of the real circular distance, including the half-ulp seam.
  The Euclidean norm of the per-axis circular distances and open absolute
  differences is a metric. The floating-point norm stays within the slack
  below of that metric, and pruning widens every triangle test by that slack.

Metric trait functions:

- `dist2(a, b)`
- `axis_min_dist2(x, lo, hi, axis)`
- `aabb_min_dist2(point, lo, hi)`

The point metric is exactly symmetric (`dist2(a, b) == dist2(b, a)`) and
`dist2(a, a) == 0`.

For finite inputs whose squared axis distances do not overflow (otherwise the
API returns `Overflow`), acceptance is exactly `dist2 <= r^2` using this
metric. Branch pruning widens each triangle test by a relative Higham factor
on the norm, the square root, and the branch-test addition, plus
`Metric::vp_absolute_slack`.

That slack is the sum of two absolute terms, each an upper bound.

- Square underflow. `(2^-537)^2` is the subnormal unit in the last place
  `2^-1074`. A positive binary64 whose square underflows is strictly below
  `2^-537`, so `sqrt(fl(c*c))` can hide a gap of that size on one axis.
  Over `d` axes the Euclidean norm of those gaps is at most
  `sqrt(d) * 2^-537`. The floor stored on the index is at least
  `8 * sqrt(d) * 2^-537`, with the square root and the products rounded
  upward: three distances in the triangle test, and room for rounding the
  floor. It is present for an open axis. It does not underflow to zero.
- Periodic residue. `η_i = 4 * ulp(L_i)` on each periodic axis. The norm is
  bounded by `sqrt(n) * max η_i`, not by a sum of squares, so a period below
  about `1e-146` cannot underflow the slack to zero. Eight times that norm
  covers a defect of `3 * ||η||_2`.

The relative factor is still added, including sixteen ulps of each of the
three distances in the test. At the origin that relative piece is forty-eight
units in the last place of zero. It is not the only absolute guard.

The neighbour set, the k-NN order `(dist2, index)`, the pair list, and the
DBSCAN labels therefore match brute force under this `dist2`, including
squared distances that underflow and periodic pairs whose raw coordinates
differ by many periods. When every periodic residue is a binary64 number and
every axis minimum-image distance is a binary64 number, `dist2` is the sum of
the squares of those distances whenever that sum is a binary64 number. In
particular, with period 10, `dist2(2^55, 0) = 4`, `dist2(2^55, 8) = 0`,
`dist2(2^62, 4) = 0`, `dist2(1, 11) = 0` and `dist2(0, 11) = 1`; with period
7, `dist2(1.25, 8.25) = 0`; an integer lattice modulo 360 has the integer
minimum-image square. A squared distance that underflows is `0`. A residue
that is not a binary64 number is not claimed to be correctly rounded.

`axis_min_dist2` and `aabb_min_dist2` are lower bounds on the squared
distance to the interval or box. They are zero when the interval may cover a
full period (`width >= L`, including a width that rounding could push across
`L`) and when the gap is inside the rounding budget. They are not claimed to
be tight.

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
