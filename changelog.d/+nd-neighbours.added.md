- Added an N-dimensional neighbour index API with mixed periodic/open
  axes: Rust `VpTreeIndex` over row-major `f64` buffers, radius and k-nearest
  queries, all-pairs-within-radius, and deterministic DBSCAN labels/core flags.
- Added a C ABI for the N-dimensional index lifecycle and queries
  (`lc_nd_build`, `lc_nd_free`, `lc_nd_knn`, `lc_nd_within`,
  `lc_nd_pairs_within`, `lc_nd_dbscan`).
- Added the N-dimensional metric shim module (`src/nd/metric.rs`) that centralizes
  distance and lower-bound logic behind one trait for future replacement by
  upstream minimage N-dimensional metrics.
- Notable API surface change: public `linkcell::Error` gained
  `BadDimensions`, `BadPeriod`, `IndexOutOfRange`, `BadNdRadius`,
  `BadNdEps`, `BadNdMinPts`, and `DistanceOverflow` for N-dimensional
  validation, a zero `min_pts`, and a distance that overflows binary64.
  `BadNdMinPts` is `min_pts must be at least 1`. `DistanceOverflow` is
  `distance overflowed`, which is not `Overflow` (`n * k overflows`).
  `BadCutoff` now reads `cutoff must be finite and strictly positive`
  (previously `cutoff must be finite and positive`).
  This is recorded as a semver-significant API expansion because `Error`
  is not marked `#[non_exhaustive]` in this release line.
- N-dimensional `dist2` is the binary64 nearest to the exact sum of the
  squares of the per-axis distances, ties to even. When that sum is a
  binary64 number, the result is that number. Periodic distance folds the
  difference of the IEEE remainders; it does not reflect a coordinate
  before subtracting. Queries are checked against a full scan under this
  `dist2`. Pruning adds a square-underflow floor and a periodic term
  `8 * sqrt(p) * max(4 * ulp(L_i))`, where `p` is the periodic-axis count,
  which stays positive when `4 * ulp(L)` squares to zero, plus a Higham
  relative factor.

