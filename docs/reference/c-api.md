# C application binary interface (ABI)

Header: `include/linkcell.h`.
Prefix `lc_`.
Caller owns every buffer.
C++ (`include/linkcell.hpp`) wraps these entry points. `linkcell::Error` has an out-of-line destructor in the shared library `linkcell_eh`, so every shared library sees one typeinfo.
Device searches are `lc_gpu_*` in `include/linkcell_gpu.h`; link the package's `linkcell-gpu` target.

## Cell

```c
typedef struct lc_cell {
  double ax, ay, az;
  double bx, by, bz;
  double cx, cy, cz;
  double ox, oy, oz;
} lc_cell;

lc_cell lc_cell_ortho(double lx, double ly, double lz);
```

Lattice vectors are rows `a`, `b`, `c` (the same order vesin uses) plus an origin.
`lc_cell_ortho` builds a diagonal box at the origin.

## Search

```c
int lc_knearest(const double *xyz, size_t n, const lc_cell *simbox,
                size_t k, const int *mask, double cell_hint,
                int *out_nn);

int lc_knearest_d2(const double *xyz, size_t n, const lc_cell *simbox,
                   size_t k, const int *mask, double cell_hint,
                   int *out_nn, double *out_d2);

int lc_knearest_many(const double *xyz, size_t n, size_t n_frames,
                     const lc_cell *simbox, size_t k, const int *mask,
                     double cell_hint, int *out_nn, double *out_d2);
```

| Argument | Contract |
| --- | --- |
| `xyz` | `n` packed `x y z` triples. Not null. |
| `n` | Point count. `size_t`. Zero is an error. |
| `simbox` | Periodic parallelepiped. Not null. |
| `k` | Neighbours per source. `size_t`. Zero is an error. |
| `mask` | `NULL` (keep every point) or `n` ints, nonzero to include. A zero drops the point as both source and candidate. |
| `cell_hint` | Target bin edge. `<= 0` selects the default (3.0 in box units). |
| `out_nn` | Caller-owned, length `n * k`. Unused slots are `-1`. Neighbours of source `i` are `out_nn[i*k + t]`, nearest first. |
| `out_d2` | Optional. Caller-owned, length `n * k` (`n_frames * n * k` for `lc_knearest_many`). Unused slots are `NaN`. `NULL` skips distances. |
| `n_frames` | Frame count for `lc_knearest_many`. `xyz` is frame-major. |

`lc_knearest` is the one-frame, indices-only entry.
Returns 0 on success, nonzero on failure.
A NaN or infinite coordinate, masked or not and in any frame, or a non-finite `simbox` origin returns nonzero with `lc_last_error()` set to `a coordinate is not finite`, before any search; `out_nn` is filled with `-1` and `out_d2` with `NaN`.
A non-finite lattice entry returns nonzero as a singular or non-positive cell.
The device entries `lc_gpu_knearest*` in `linkcell_gpu.h` do not scan their device coordinates for non-finite values.

## N-dimensional index

```c
typedef struct lc_nd_index {
  uint8_t _private;
} lc_nd_index;

int lc_nd_build(const double *data, size_t n, size_t d,
                const double *periods, lc_nd_index **out);
void lc_nd_free(lc_nd_index *index);
size_t lc_nd_len(const lc_nd_index *index);
size_t lc_nd_dims(const lc_nd_index *index);

int lc_nd_knn(const lc_nd_index *index, const double *query, size_t k,
              int *out_idx, double *out_d2);

int lc_nd_within(const lc_nd_index *index, const double *query,
                 double radius, int *out_idx, double *out_d2,
                 size_t cap, size_t *out_count);

int lc_nd_pairs_within(const lc_nd_index *index, double radius,
                       int *out_i, int *out_j, double *out_d2,
                       size_t cap, size_t *out_count);

int lc_nd_dbscan(const lc_nd_index *index, double eps, size_t min_pts,
                 int *out_labels, int *out_core, size_t *out_clusters);
```

The handle is opaque. Callers only hold a pointer returned by `lc_nd_build`.
The one byte makes it a complete object type in C99 and C++17.

`lc_nd_build` takes row-major `data` of shape `n x d`.
`periods` is optional (`NULL` means every axis open); with a non-null pointer,
length is `d`; `periods[k] > 0` marks axis `k` periodic, `periods[k] == 0`
marks it open, and negative or NaN values are rejected.
On any `lc_nd_build` failure, `*out` is set to `NULL`.

The period table is one optional period per axis, 16 bytes. `d` such that
this table would not fit in `isize::MAX` bytes fails with `the period table
does not fit in isize::MAX bytes` before any allocation. A `d` that passes
that check can still fail with `the period table could not be allocated`
when the allocator refuses the table. `n * d` that does not fit in `usize`
fails with `n * d overflows` before the coordinates are read. Those three
strings are the C messages for these refusals. `VpTreeIndex::new` returns
`NdProduct` (`n * d overflows`) for the same product before it reads a
point. `Error::Overflow` still displays `n * k overflows`.

`lc_nd_knn` writes exactly `k` slots (`-1` and optional `NaN` for unused
slots when `k > n`).
`radius` for `lc_nd_within` / `lc_nd_pairs_within` must be finite and
non-negative. `eps` for `lc_nd_dbscan` must be finite and strictly positive.

`lc_nd_within` and `lc_nd_pairs_within` accept count-query mode:

- for `lc_nd_within`, set `out_idx = NULL` (and `out_d2 = NULL`) to query
  `*out_count`;
- for `lc_nd_pairs_within`, set `out_i = out_j = out_d2 = NULL` to query
  `*out_count`.

For fill calls, if `cap < *out_count`, the call returns nonzero and keeps
`*out_count` at the required size without writing partial output.

`lc_nd_dbscan` writes one label and one core flag per indexed point.
Noise has label `-1`.
`min_pts` includes the point itself.
Index outputs are `int` to match existing `lc_*` neighbour and pair ABI
conventions; if an index would not fit in `int`, the call returns nonzero.
For finite input coordinates, `dist2` is the binary64 nearest to the exact
sum of the squares of the per-axis distances, ties to even. When that exact
sum is a binary64 number, the result is that number, including
`(1 + 2^-27, 2^-27, 2^-27, 2^-27)` against the origin
(`1 + 2^-26 + 2^-52`) and, for period 10, `dist2(2^55, 0) = 4`,
`dist2(2^55, 8) = 0`, and `dist2(2^62, 4) = 0`. The call fails with
`distance overflowed` when that nearest value is not finite. That string is
not `n * k overflows`. A half-period gap whose square is outside the finite
range is that error, not a zero distance. `dist2(1, 0)` at period `1e308`
is `1`. The nearest value is `0` only when zero is a nearest binary64 to the
exact sum. `min_pts == 0` fails with `min_pts must be at least 1`.

A query fails with `distance overflowed` exactly when a full scan would.
`lc_nd_knn` with `k > 0` and `lc_nd_within` read the distance from the
query to every point, and `lc_nd_pairs_within` and `lc_nd_dbscan` read every
pair. The tree never prunes a subtree that may hold a point `2^510` or
farther from the query, so the result does not depend on the point order.
`lc_nd_build` does not fail on such a distance.

`lc_nd_within`, `lc_nd_pairs_within` and `lc_nd_dbscan` accept a point when
`dist2 <= fl(radius * radius)`, the product rounded to binary64. When that
product is `0`, only a point that coincides with the query is accepted, not
one whose square underflowed to `0`. With points `i * 1e-163` and the query
`0`, radius `27e-163` returns the 28 points `i <= 27`: `fl(radius * radius)`
is `2^-1074`, and each of their squares rounds to at most that. Hits are
ordered by exact squared distance, then index. Two `dist2` values that tie
keep the order of their unrounded squares: for points `1e-170` and `0`,
`lc_nd_knn` from `0` with `k = 1` returns index 1.

Returned ND results are checked against a full scan under the same `dist2`.
Each periodic coordinate is reduced with the IEEE remainder, the remainders
are subtracted, and that difference is folded into `[-L/2, L/2]`. A
coordinate is not reflected before the subtraction. A signed `rem_euclid` is
not used, so a small negative delta on a huge period is kept: with period
`2^53`, `dist2(-0.5, 0)` is `0.25` and `dist2(2^52 - 1, 2^52)` is `1`; with
period `1e16` or `1e300`, `dist2(-1, 0)` is `1`; with period `1e16`,
`dist2(-5e15, 4999999999999999)` is `1`. With period 10,
`dist2(1.7e308, -1.7e308)` is `16`. Pruning widens every triangle test by a
Higham relative factor, by at least `8 * sqrt(dims) * 2^-537`, and by
`8 * sqrt(p) * max(4 * ulp(L_i))` where `p` is the number of periodic axes,
not the point count. The periodic term is not a sum of squares, so a tiny
period cannot zero it. It is a prune guard, not an error bound on `dist2`.
`dist2` is rounded once from the exact sum, so each distance in a triangle
test is within a relative `2^-52` and an absolute `2^-537` of its exact
value at every dimension. The relative factor grows with the dimension and
the absolute floor with its square root; neither is a fixed number of ulps.
`±L` are the same point, including `L = 1e308`. `knn` with `k = 0` does not
walk the tree. `lc_nd_within` with radius `0` reads a few distances per tree
level, not one per point.

```c
int lc_pairs_within(const double *xyz, size_t n, const lc_cell *simbox,
                    double cutoff, const int *mask, double cell_hint,
                    int half, int *out_i, int *out_j, int *out_shift,
                    double *out_d2, size_t cap, size_t *out_count);
```

A cutoff list is `lc_pairs_within`.
Each row is one atom-image.
`out_shift[3 * t + 0..3]` is `(na, nb, nc)`, the vesin / tonari shift `S` in the caller's basis.
Squared distance is strictly below `cutoff * cutoff`.
`half` nonzero keeps one side of `(i, j, S)`.
Null `out_i`, `out_j`, `out_shift`, and `out_d2` query `*out_count`.
A short `cap` writes that count and returns nonzero.

```c
typedef struct lc_pair { int i; int j; int shift[3]; double dist2; } lc_pair;

int lc_pairs_within_rows(const double *xyz, size_t n, const lc_cell *simbox,
                         double cutoff, const int *mask, double cell_hint,
                         int half, lc_pair *out, size_t cap,
                         size_t *out_count);
```

`lc_pairs_within_rows` is the same list as one array of `lc_pair` rows.
Null `out` queries `*out_count`.

A query, or a short buffer, keeps its search on the calling thread.
That thread writes the rows from the saved search when the next call repeats `xyz`, the box, `cutoff`, `mask`, `cell_hint`, and `half` bit for bit.
Any other call drops the saved search.
A fill that writes keeps nothing, so a count-then-fill pair is one search.

```c
uint64_t lc_stencil_builds(void);

int lc_pairs_open(const double *xyz, size_t n, double cutoff,
                  double cell_hint, lc_pair *out, size_t cap,
                  size_t *out_count);

int lc_pairs_within_pbc(const double *xyz, size_t n, const lc_cell *simbox,
                        const int *pbc, double cutoff, const int *mask,
                        double cell_hint, int half, int closed,
                        int exclude_self_images, lc_pair *out, size_t cap,
                        size_t *out_count);
```

`lc_pairs_open` is the Cartesian half list of an open cluster: `i < j`,
shift zero, squared distance strictly below `cutoff * cutoff`.
`lc_pairs_within_pbc` is that list with a periodicity mask.
`pbc[k]` nonzero keeps axis `k` periodic. A null `pbc` is fully
periodic and is `lc_pairs_within_rows`. An open axis is replaced by a
unit normal of the periodic rows, however long the stored row is, and
is binned on the occupied Cartesian cells of that lattice. It is not
padded into a periodic box. A fully periodic cell whose face width
is zero or non-finite returns nonzero. A positive finite width is not
rejected by its ratio to the longest edge. A width that needs more
than 2^24 images, or whose estimated row buffer does not fit, returns
nonzero with the image-count error. The mesh edge is clamped to the
shortest width, so a face much shorter than the other edges can exceed
the bin cap at every cutoff and returns the mesh error. When both caps
hold, including a short face no shorter than the cutoff, the call
returns the pairs. The partner stencil is capped at 2^31 slots,
enough for a reach of two bins on every axis of a mesh at the bin cap.
A larger reach box returns the image-count error and is not stored.
`lc_stencil_builds` counts partner stencils that were built. A stencil
that is reused does not increment it. A fully open stencil is reused
by every call with the same cutoff and bin edge, wherever the atoms
sit. With a periodic axis it is reused while the atoms stay inside the
same bounding bins.

```c
int lc_pairs_within_csr(const double *xyz, size_t n, const lc_cell *simbox,
                        const int *pbc, double cutoff, const int *mask,
                        double cell_hint, int half,
                        int *out_offsets, size_t cap_offsets,
                        int *out_cols, double *out_d2, size_t cap_cols,
                        size_t *need_offsets, size_t *need_cols);
```

`lc_pairs_within_csr` writes the cutoff neighbours of each point as compressed sparse row (CSR) rows.
Row `i` is `out_cols[out_offsets[i] .. out_offsets[i + 1]]`.
It holds every `j != i` with an image strictly within `cutoff`, once, in ascending order.
On a periodic axis that is the minimum-image distance, also when `cutoff` is past half the box, where one atom meets another through several images.
A periodic image of `i` is not listed.
`half` nonzero keeps `j > i` only.
`out_d2`, if non-null, receives the squared distance of the nearest image of each column.
A masked-out point has an empty row and appears in no row.
`pbc` and `mask` are those of `lc_pairs_within_pbc`.
`simbox` may be null when every axis is open.

| Call | Result |
| --- | --- |
| `out_offsets`, `out_cols`, `out_d2` all null | Query: sets `*need_offsets = n + 1` and `*need_cols`, returns 0 |
| `cap_offsets >= n + 1` and `cap_cols >= *need_cols` | Writes the rows, sets both counts, returns 0 |
| Either buffer short | Sets both counts, writes nothing, returns nonzero |
| `n == 0` | Returns 0 with one offset, `0` |
| Bad cutoff, non-finite coordinate, null count pointer, more than `INT_MAX` columns | Returns nonzero with `lc_last_error` set |

A query or a short buffer keeps its rows on the calling thread, like the other count queries.
The next call there with the same inputs bit for bit copies them out without searching again.
A short buffer with a null `out_d2` keeps no distances, so a next call that asks for them searches again.
A buffer large enough on the first call takes the rows straight from the search.
The distances are computed only when `out_d2` is non-null.
The rows and distances are the same bits at every `lc_set_threads` width and SIMD path.
On a periodic cell each point gathers its row from the bins around it, a bin at a time on the pool, and one copy lays the rows out in point order.
No hit is buffered and no step sorts the whole list.
A cell with an open axis is searched on the calling thread, and its rows are then built on the pool.
`pairs_within_csr` is the Rust entry and fills a `PairCsr`.

## Row grid

```c
typedef struct lc_row_grid {
  uint8_t _private;
} lc_row_grid;

typedef void (*lc_row_visit)(void *user, const lc_pair *row);

int lc_row_grid_build(const double *xyz, size_t n, const lc_cell *simbox,
                      const int *pbc, double cutoff, const int *mask,
                      double cell_hint, int closed, int exclude_self_images,
                      lc_row_grid **out);
int lc_row_grid_update(lc_row_grid *grid, const double *xyz, size_t n,
                       const lc_cell *simbox, const int *pbc, double cutoff,
                       const int *mask, double cell_hint, int closed,
                       int exclude_self_images);
size_t lc_row_grid_len(const lc_row_grid *grid);
void lc_row_grid_free(lc_row_grid *grid);

int lc_row_grid_rows(const lc_row_grid *grid, size_t begin, size_t end,
                     int half, lc_row_visit visit, void *user);
int lc_row_grid_point(const lc_row_grid *grid, size_t i, int half,
                      lc_row_visit visit, void *user);
```

`lc_row_grid_build` bins the points of one cutoff search and keeps the bins.
It stores no pair.
`xyz`, `simbox`, `pbc`, `cutoff`, `mask`, `cell_hint`, `closed`, and `exclude_self_images` are those of `lc_pairs_within_pbc`.
`simbox` may be null when every axis is open.
The grid copies the positions it needs, so `xyz` may be released after the call.
`lc_row_grid_free` releases the grid and accepts null.

`lc_row_grid_rows` visits the rows of points `begin .. end` in ascending point order.
`visit(user, row)` runs on the calling thread once for every image of `row->j` within the cutoff of `row->i`.
`row->shift` and `row->dist2` are the shift and squared distance `lc_pairs_within_pbc` writes for that row, with the same bits.
`row` is valid until the callback returns.
`half` nonzero keeps the entries the half list has under `i`: `j > i`, and the self images whose first non-zero shift is negative.
Over `0 .. lc_row_grid_len(grid)` the entries are the rows of `lc_pairs_within_pbc` for the same arguments.
Entries of one row come in bin order.
That order depends on the grid alone, so a split into ranges changes neither a row nor the order inside it.
A masked-out point has an empty row.
`lc_row_grid_point` visits the row of one point, with the entries and order `lc_row_grid_rows` gives it.

A visit reads the grid and writes nothing in it.
Several threads may visit one grid at once, and disjoint ranges cover each row once.
`lc_row_grid_update` and `lc_row_grid_free` need the grid to themselves.

| Call | Result |
| --- | --- |
| `lc_row_grid_build` with `n == 0` | Returns nonzero with `no points`; `*out` is null |
| `lc_row_grid_build` with a bad cutoff, cell, mask, or coordinate | Returns nonzero with the message `lc_pairs_within_pbc` sets; `*out` is null |
| `lc_row_grid_update` that fails | Returns nonzero; the grid holds no point and stays valid for another update or the free |
| `lc_row_grid_rows` with `begin >= end` | Visits nothing, returns 0 |
| `lc_row_grid_rows` with `end` past the length, or `lc_row_grid_point` with `i` at or past it | Returns nonzero with `point index is out of range` before any visit |
| Null `grid` or null `visit` | Returns nonzero with `null pointer` |
| `lc_row_grid_len(NULL)` | 0 |

`lc_row_grid_update` bins new points, a new cell, or new arguments into the same handle.
It releases the previous bins first, so the new ones reuse their buffers.
The callback must not unwind and must not update or free the grid it is reading.
`RowGrid` is the Rust entry: `RowGrid::build`, `update`, `clear`, `for_rows`, and `for_point`, which take a closure.

## Vesin compatibility

`include/linkcell_vesin.h` declares `lc_vesin_neighbors` and
`lc_vesin_free`. Their device, options, and output structures have the
same layout as vesin's C API. `lc_vesin_neighbors` computes fully
periodic CPU lists in linkcell. It forwards a non-periodic list or a
CUDA list to the platform vesin library when that library is visible to
the loader.

```c
int32_t lc_vesin_neighbors(
    const double (*points)[3], size_t n_points,
    const double box[3][3], const bool periodic[3],
    struct lc_vesin_device device, struct lc_vesin_options options,
    struct lc_vesin_list *neighbors, const char **error_message);
void lc_vesin_free(struct lc_vesin_list *neighbors);
```

The caller zero-initializes the output list before its first call and
may pass it back to reuse its buffers. A zero return is success. On
failure, `error_message` receives a borrowed string. The output pointers
remain owned by the list and are valid until its next call or
`lc_vesin_free`. `lc_vesin_free` accepts a null pointer, releases a list
filled by linkcell, and zeros it. It must not receive a list allocated
by another vesin build.

`include/vesin_alias.c` supplies the unprefixed `vesin_neighbors` and
`vesin_free` names for drop-in use.

## Errors and version

```c
const char *lc_last_error(void);
const char *lc_version(void);
```

Both return pointers the caller leaves alone.
`lc_last_error` is the one thread-local error slot shared by the
status-returning `lc_*` calls in `linkcell.h`. It is `NULL` after a
successful call on that thread, or if none has failed yet.
Its pointer is valid until the next such `lc_*` call on that thread.
`lc_version` returns the process-static version string and does not read or write the error slot.
Distinct searches may run concurrently; each thread reads its own error slot.

## C++ wrap

```cpp
namespace linkcell {
class Error : public std::runtime_error { /* status() is the C return */ };

struct Cell {
  std::array<double, 3> a, b, c, origin;
  static Cell ortho(double lx, double ly, double lz);
  static Cell from_vectors(std::array<double, 3> a, std::array<double, 3> b,
                           std::array<double, 3> c,
                           std::array<double, 3> origin = {});
  lc_cell raw() const;
};

class Neighbours {
  int neighbour(std::size_t i, std::size_t j) const; /* or -1 */
  double dist2(std::size_t i, std::size_t j) const;  /* unused NaN */
  std::size_t n() const;
  std::size_t k() const;
  const int *data() const;
  const double *distances() const;
};

void knearest_into(const double *xyz, std::size_t n, const Cell &cell,
                   std::size_t k, int *out, std::size_t out_len,
                   const int *mask = nullptr, double cell_hint = 0.0);

Neighbours knearest(const double *xyz, std::size_t n, const Cell &cell,
                    std::size_t k, const int *mask = nullptr,
                    double cell_hint = 0.0);

const char *version();

struct ShiftedPair {
  int i, j;
  std::array<int, 3> shift;
  double dist2;
};

std::vector<ShiftedPair> pairs_within(const double *xyz, std::size_t n,
                                      const Cell &cell, double cutoff,
                                      const int *mask = nullptr,
                                      double cell_hint = 0.0,
                                      bool half = false);

struct PairCsr {
  std::vector<int> offsets;
  std::vector<int> cols;
  std::vector<double> dist2;
};

void pairs_within_csr(const double *xyz, std::size_t n, const Cell &cell,
                      double cutoff, PairCsr &out, bool half = false,
                      const std::array<bool, 3> &periodic = {true, true, true},
                      const int *mask = nullptr, double cell_hint = 0.0,
                      bool with_dist2 = false);
}
```

`knearest` returns an owning packed `Neighbours` (`n * k` ints).
`knearest_into` writes a caller buffer and requires `out_len == n * k`.
Neither returns `std::vector<std::vector<int>>`.
Failure throws `linkcell::Error`.
Requires C++17.

`ShiftedPair` has the layout of `lc_pair`.
`pairs_within` queries the count, then `lc_pairs_within_rows` writes the returned vector from that same search.
`pairs_within_csr` calls `lc_pairs_within_csr` into the capacity `out` already holds.
When that is short it resizes `out` and copies the kept rows, so a reused `PairCsr` costs one search per frame.

## Partition

`lc_cell_grid` writes `nx, ny, nz`, three bin widths, and one point count per cell.
`lc_cell_graph` adds the compressed sparse row (CSR) of the periodic adjacency.
Edge weights are pair counts from `lc_pairs_within` with `half` set.
`lc_partition` writes one domain id per cell.
`method` is 0 Morton, 1 Hilbert, 2 recursive coordinate bisection (RCB), 3 Hilbert plus Fiduccia-Mattheyses, 4 bisection plus that pass, 5 Scotch, 6 multilevel, 7 parallel Scotch.
`balance <= 0` selects `1 + parts * w_max / total`, where `w_max` is the heaviest cell weight and `total` is the sum of the cell weights.
That is the tightest bound a cell-aligned cut can promise, and it is the ratio the d-SEAMS domain driver derives per frame.
`lc_halo_width` writes three bin radii for `cutoff * depth`.
`lc_halo` writes the ghost cell ids.

A null weight or domain pointer queries the count.
The C++ wrappers are `linkcell::partition`, `linkcell::halo_width`, and `linkcell::halo`.

`lc_partition_weights` assigns domains from caller-supplied weights.
`nxyz` is `nx, ny, nz`.
`weight` has one count per cell, in the order `((iz * ny) + iy) * nx + ix`.
`method` and `balance` follow `lc_partition`.
A plain bisection reads the weights and not the pair arcs.
`out_domain` receives one domain per cell.

`lc_mesh_cut` writes the same domain ids and also the bin of each point.
A null `out_domain` queries `*out_ncell` and leaves the other buffers alone.
On a fill, `out_nxyz` is three ints, `out_width` is three bin widths, `out_domain` is one domain per cell, and `out_point_cell` is one bin per point.

`lc_owned_mask` writes 2 for an owned point and 1 for a point in a cell the owned walk paints.
`width` is three bin radii.
`lc_within_reach` writes 2 for an owned point and 1 for a ghost within `reach` of an owned atom.

`lc_scotch_part` and `lc_ptscotch_part` are the serial and parallel Scotch entry points.
They are compiled with the `scotch` and `ptscotch` features.
Both features are off by default.
See [features](features.md).

## k-nearest graph

```c
int lc_knearest_csr(const double *xyz, size_t n, const lc_cell *simbox,
                    size_t k, const int *mask, double cell_hint,
                    int mutual, int *out_offsets,
                    int *(*alloc)(void *, size_t), void *ctx);
```

Inputs and argument errors match `lc_knearest`, including the non-finite coordinate and origin check.
`out_offsets` has length `n + 1`.
`alloc(ctx, total)` is called once, from any thread of the search, and returns `total` writable ints for the columns.
A null return fails the call.
With `mutual` nonzero, row `i` holds every `j` that row `i` lists and that lists `i`: the lower indices ascending, then the higher ones nearest first.
Otherwise an edge that both rows list is stored from the lower index, and an edge that one row lists is stored from that row.
A row holds edges from lower rows ascending, then its own nearest first, then edges from higher rows ascending.
The search, the degrees, and the fill run in one parallel region.
`knearest_csr` is the Rust entry and returns `(offsets, columns)`.

## Thread width

`lc_set_threads(n)` chooses the width of later searches.
`n == 0` leaves the width unset: the search starts on the calling thread, and Rayon and OpenMP keep their own widths.
`n == 1` runs the whole search on the calling thread and starts no thread.
`n < 0` selects one thread.
`lc_num_threads` returns the selected width, or the global pool width when the selection is unset.

`lc_set_places(cpus, n)` records CPUs that Rayon workers may use.
Every worker may run on any CPU in the list.
A pool that has already started keeps the CPUs it was built with.
`n <= 0` is ignored.

`lc_for_slabs(n, ctx, cb)` calls `cb(ctx, begin, end)` over disjoint spans of `0..n` on that pool.
One thread runs the whole span on the caller.
The spans cover `0..n` exactly once.

`lc_team_run` runs `nstages` stages in one parallel region on the same pool.
Stage `s` is `chunks[s]` calls of `fn_(ctx, s, chunk, worker)`, in any order and on any worker.
Every chunk of a stage returns before any chunk of the next stage starts.
`worker` is fixed for one worker during the call and is below `lc_team_width`.
A null `fn_`, or a null `chunks` when `nstages > 0`, returns nonzero.
`nstages == 0` returns 0.
The callback must not unwind.

`lc_set_continuation(ctx, cb)` runs `cb` at the end of this thread's next `lc_knearest`, still inside its pool.
A null `cb` clears the hook.
The hook belongs to the calling thread, and a search runs the hook set on the thread that started it.
The search leaves the hook in place. The caller clears it.

`void lc_knn_shutdown(void)` joins the standing k-nearest worker team before
library unload. It is safe to call repeatedly, and a later query starts
a new team. The symbol is present when the `parallel` build uses the
OpenMP-compatible pthread team.

## Device hook

`lc_set_gpu_curve_split` installs the device curve split used when `lc_partition` is called with `device` nonzero.
The GPU library calls it from a static initializer when that library is loaded.
A null function removes the hook, and the device partition then reports that no device is loaded.

## Rust errors (same crate)

`linkcell::Error`:

| Variant | Meaning |
| --- | --- |
| `ZeroK` | `k == 0` |
| `BadBox` | A box length is not strictly positive, or H is singular |
| `Empty` | The point list is empty |
| `BufferSize` | `knearest_into` / `out` length is not `n * k` |
| `MaskLen` | `mask` is `Some` and `mask.len() != n` |
| `TooManyCells` | linked-cell mesh overflows the bin cap |
| `BadCutoff` | `cutoff` is not finite and strictly positive |
| `BadNdRadius` | ND query radius is not finite or is negative |
| `BadNdEps` | ND DBSCAN epsilon is not finite and strictly positive |
| `BadNdMinPts` | ND DBSCAN `min_pts` is zero |
| `Overflow` | A size, index, graph weight, open-cell coordinate, or periodic image offset does not fit its representation (`n * k overflows`) |
| `NdProduct` | `n * d` does not fit in `usize` (`n * d overflows`) |
| `DistanceOverflow` | An N-dimensional distance overflowed binary64 (`distance overflowed`) |
| `BadPartition` | The part count is zero, or the balance bound is below 1 |
| `BadGraph` | The cell graph must be a square compressed sparse row (CSR) of that grid |
| `NoDevice` | A device curve split was requested and no device is loaded |
| `ScotchDisabled` | Scotch support was not compiled in |
| `TooManyImages` | The periodic image count exceeds 2^24 |
| `Degenerate` | A periodic face width is zero or non-finite (`periodic cell is degenerate: a face width is zero or non-finite`). A positive finite width is cutoff-aware and uses the image and bin caps. An open axis is replaced before this test |
| `Partial` | An axis is not periodic and `libvesin.so` is not loaded |
| `Device` | The device is not the CPU and `libvesin.so` is not loaded |
| `Padded` | A point has more neighbours than the requested pad |
| `NonFinite` | A coordinate is not finite. `knearest`, `knearest_into`, `knearest_into_d2`, `knearest_into_many`, `knearest_csr`, `knearest_brute` (and through them `lc_knearest`, `lc_knearest_d2`, `lc_knearest_many`, `lc_knearest_csr`), `pairs_within_pbc`, `pairs_within_csr`, `pairs_open`, and the partition entry points return this variant. The k-nearest entries also return it for a non-finite cell origin |
| `DeviceMismatch` | The device curve split disagreed with the host weights |
| `ScotchFailed` | The Scotch library rejected the graph |
| `BadDimensions` | N-dimensional row-major shape is inconsistent |
| `BadPeriod` | N-dimensional period value is not finite and strictly positive |
| `IndexOutOfRange` | N-dimensional point index is outside the indexed range |

All status-returning C ABI calls currently return `0` on success and `1` on
failure; error detail is carried by `lc_last_error()`.

`Error::Empty` is not a wrong-length buffer.
`knearest` returns `ZeroK`, `Empty`, `MaskLen`, `Overflow`, `NonFinite`, and `TooManyCells`. `TooManyCells` is also a box at least 10^7 bin edges across: the walk would not certify the neighbours.
`knearest_csr` returns the same variants.
`knearest_into`, `knearest_into_d2`, and `knearest_into_many` return them and `BufferSize`.
`knearest_brute` returns `ZeroK`, `Empty`, `NonFinite`, and `Degenerate`.
A non-finite lattice vector is `BadBox` from the `Cell` constructor, before any search.
`pairs_within` returns `BadCutoff`, `Empty`, `MaskLen`, `Overflow`,
`TooManyImages`, `TooManyCells`, and `Degenerate`. Its `Overflow` cases include a
point count above `u32::MAX` and a relative periodic image offset that
does not fit the stored shift.
Positions and cell vectors share one length unit.
`dist2` is that unit squared.
`cutoff` and `cell_hint` are lengths in that unit.

Walk, stop rule, and cost: [algorithm](algorithm.md).
Features and instruction-set builds: [features](features.md).
