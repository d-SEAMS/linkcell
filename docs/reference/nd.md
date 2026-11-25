# N-dimensional neighbour index and DBSCAN

`linkcell::VpTreeIndex` provides N-dimensional neighbour search over
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
The current choice is a vantage-point tree (Yianilos, SODA 1993):
branch-and-bound from the triangle inequality.

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

Approximate search with an explicit epsilon is not enabled. Queries are
checked against a full scan under this `dist2`.

## Space and metric

- `OwnedContiguousSpace` / `ContiguousSpace` read row-major buffers.
- `PeriodicSpace` supports per-axis `Option<f64>` periods:
  - `Some(L)`: periodic axis with period `L`.
  - `None`: open axis.
- The index stores the original finite coordinates. A query is copied the
  same way. On a periodic axis, `dist2` takes the IEEE remainder of each
  coordinate, subtracts those remainders, and folds that difference into
  `[-L/2, L/2]` (the tie `+L/2` has the same square as `-L/2`). A coordinate
  is not reflected by subtracting `L` before that difference exists. A stored
  `[0, L)` representative cannot hold a negative offset smaller than `ulp(L)`.
  With period `2^53`, `dist2(-0.5, 0)` is `0.25` and `dist2(2^52 - 1, 2^52)`
  is `1`. With period `1e16`, `dist2(-1, 0)` is `1` and
  `dist2(-5e15, 4999999999999999)` is `1`. With period `1e300`,
  `dist2(-1, 0)` is `1`. With period `0.1 * 2^60`, `dist2(-0.125, 0)` is
  `0.015625`.
- `dist2` is the binary64 nearest to the exact sum of the squares of those
  per-axis distances, with ties rounded to even. An open axis uses the exact
  difference of its two coordinates. When that exact sum is itself a binary64
  number, `dist2` is that number. The point
  `(1 + 2^-27, 2^-27, 2^-27, 2^-27)` against the origin has exact sum of
  squares `1 + 2^-26 + 2^-52`, and `dist2` returns that value. The same holds
  for the integer residues with period 10: `dist2(2^55, 0) = 4`,
  `dist2(2^55, 8) = 0`, `dist2(2^62, 4) = 0`, `dist2(1, 11) = 0`,
  `dist2(0, 11) = 1`, and `dist2(1.7e308, -1.7e308) = 16`; with period 7,
  `dist2(1.25, 8.25) = 0`. `±L` are the same point, so `dist2(L, -L)` is `0`
  even when `L` is `1e308`.
- `DistanceOverflow` (`distance overflowed`) is returned when that nearest
  value is not finite. The message is not `Overflow` (`n * k overflows`).
  A half-period gap whose square is outside the finite range (`0` and
  `6e199` with period `1e200`, and the same class at `1e300` and `1e308`)
  is `DistanceOverflow`, not `Ok(0)`. A gap whose exact square is finite is
  that finite value: `dist2(1, 0)` at period `1e308` is `1`.
- The nearest binary64 is `0` when the exact sum is closer to zero than to
  the smallest positive binary64, and on the tie at exactly half of that
  value. A nonzero exact sum that is a binary64 number is not replaced by
  `0`. `2^-106`, the square of `2^-53`, is returned unchanged.
- `wrap_point` returns a representative in `[0, L)`. For a negative `x` it
  is `fl(L - (|x| % L))`, and `0` when that is not in `(0, L)`. `dist2`
  does not use it. `wrap_delta` is a minimum image in `[-L/2, L/2)`, with
  `+L/2` returned as `-L/2`; a result of `±L` is returned as `0`. It
  forms `L - r` only when Sterbenz' lemma makes it exact, so it differs
  from its input by a whole number of periods. With period `5e-324`,
  `0.5 * L` underflows and `wrap_delta(0)` is `0`, not `-L`; the tie is
  found from `r + r`, not from `0.5 * L`. On a subnormal period every value
  is a multiple of `2^-1074`, so `wrap_point` is exact too.
  `wrap_delta(-1)` at period `1e16` is `-1`. `wrap_delta(1e300)` at period
  `1e-10` is finite.

Metric trait functions:

- `dist2(a, b)`
- `dist2_exact(a, b)`: the unrounded sum as an `ExactDist2`, which orders
  squares that `dist2` rounds to the same value
- `axis_min_dist2(x, lo, hi, axis)`
- `aabb_min_dist2(point, lo, hi)`

The point metric is symmetric (`dist2(a, b) == dist2(b, a)`) and
`dist2(a, a) == 0`.

Acceptance is `dist2 <= fl(r * r)`, with `r * r` rounded to binary64. When
`fl(r * r)` is `0`, only points that coincide with the query are accepted,
not points whose square underflowed to `0`. With points `i * 1e-163` and the
query `0`, `within(0, 27e-163)` returns the 28 points `i <= 27`:
`fl(r * r)` is `2^-1074`, and each of their squares rounds to at most that.

A query returns `DistanceOverflow` exactly when a full scan of the distances
it reads would. `knn` with `k > 0` and `within` read the distance from the
query to every point, and `pairs_within` and `dbscan` read every pair. A
subtree that may hold a point `2^510` or farther from the query is never
pruned, so the result does not depend on the build order. The build stores
such a distance as infinite instead of failing.

Branch pruning widens each triangle test by a relative Higham factor
on the norm, the square root, and the branch-test addition, plus
`Metric::vp_absolute_slack`.

That slack is the sum of two absolute terms.

- Square underflow. `(2^-537)^2` is the subnormal unit in the last place
  `2^-1074`. A positive binary64 whose square underflows is strictly below
  `2^-537`, so `sqrt(fl(c*c))` can hide a gap of that size on one axis.
  Over `d` axes the Euclidean norm of those gaps is at most
  `sqrt(d) * 2^-537`. The floor stored on the index is at least
  `8 * sqrt(d) * 2^-537`, with the square root and the products rounded
  upward: three distances in the triangle test, and room for rounding the
  floor. It is present for an open axis. It does not underflow to zero.
- Periodic term. `η_i = 4 * ulp(L_i)` on each periodic axis. `p` is the
  number of periodic axes, not the number of points. The norm is
  `sqrt(p) * max η_i`, not a sum of squares, so a period below about
  `1e-146` cannot underflow the slack to zero. Eight times that norm is
  added. It is a prune guard. It is not an error bound on `dist2`.

The relative factor is still added, including sixteen ulps of each of the
three distances in the test. At the origin that relative piece is forty-eight
units in the last place of zero. It is not the only absolute guard.

`dist2` is the nearest binary64 to the exact sum at every dimension, so each
distance in a triangle test is within a relative `2^-52` and an absolute
`2^-537` of its exact value. The relative factor counts `6d + 16` roundings
and the floor grows with `sqrt(d)`, so neither is a fixed number of ulps.
Collinear sets in up to 256 dimensions, where every triangle test is tight,
and subnormal squares are checked against a full scan.

The neighbour set, the k-NN order, the pair list, and the DBSCAN labels are
checked against a full scan under this `dist2`. `knn` with `k = 0` returns
an empty list without reading a distance. `within` with radius `0` reads a
few distances per tree level, not one per point: at most `4 * levels` in the
tests, where `levels` is the bit length of the point count.

`axis_min_dist2` and `aabb_min_dist2` are lower bounds on the squared
distance to the interval or box. A degenerate interval rounds the same
circular distance downward before squaring, so the bound does not exceed the
exact square. With period `1e16`, `axis_min_dist2(-1, 0, 0)` is positive and
at most `1`. The bounds are zero when the width may reach half a period or a
full period, and when the point may lie on the short arc between the
endpoints. They are not claimed to be tight. `square_down` never exceeds the
exact square, including a square in the subnormal range.

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

- `knn` and `within` sort by exact squared distance, then index. Two
  `dist2` values that tie keep the order of their unrounded squares:
  `knn(0, 1)` on `[1e-170, 0]` returns index 1.
- Equal exact squares resolve by index.
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
The period table is one optional period per axis, 16 bytes. A dimensionality
whose table would not fit in `isize::MAX` bytes fails with `the period table
does not fit in isize::MAX bytes` before allocation. A dimensionality the
size check accepts can still fail with `the period table could not be
allocated`. `n * d` that does not fit in `usize` fails with `n * d overflows`
before the coordinates are read. `VpTreeIndex::new` returns `Overflow` for
that product, and that error's message remains `n * k overflows`.
