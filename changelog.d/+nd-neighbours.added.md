- Added an N-dimensional exact neighbour index API with mixed periodic/open
  axes: Rust `VpTreeIndex` over row-major `f64` buffers, radius and k-nearest
  queries, all-pairs-within-radius, and deterministic DBSCAN labels/core flags.
- Added a C ABI for the N-dimensional index lifecycle and queries
  (`lc_nd_build`, `lc_nd_free`, `lc_nd_knn`, `lc_nd_within`,
  `lc_nd_pairs_within`, `lc_nd_dbscan`).
- Added the N-dimensional metric shim module (`src/nd/metric.rs`) that centralizes
  distance and lower-bound logic behind one trait for future replacement by
  upstream minimage N-dimensional metrics.
- Notable API surface change: public `linkcell::Error` gained
  `BadDimensions`, `BadPeriod`, and `IndexOutOfRange` variants for
  N-dimensional shape/period/index validation failures. This is recorded as a
  semver-significant API expansion because `Error` is not marked
  `#[non_exhaustive]` in this release line.
- N-dimensional queries compare equal to brute force under the same `dist2`
  on finite inputs that do not overflow. Periodic coordinates use the IEEE
  remainder into `[0, L)`, and pruning adds a square-underflow floor plus the
  proved periodic slack to a Higham relative factor.
