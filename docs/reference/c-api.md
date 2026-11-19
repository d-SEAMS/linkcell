# C application binary interface (ABI)

Header: `include/linkcell.h`.
Prefix `lc_`.
Caller owns every buffer.
C++ (`include/linkcell.hpp`) is a header-only wrap of these entry points.
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
periodic and is `lc_pairs_within_rows`. An open axis is binned on the
occupied Cartesian cells. It is not padded into a periodic box.
`lc_stencil_builds` counts partner stencils that were built. A stencil
that is reused does not increment it. A fully open stencil is reused
by every call with the same cutoff and bin edge, wherever the atoms
sit. With a periodic axis it is reused while the atoms stay inside the
same bounding bins.

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
may pass it back to reuse its buffers. A zero return is success (and
sets `*error_message` to `NULL`). If `error_message` itself is `NULL`,
the call returns `1` immediately without writing it. Otherwise, on
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
}
```

`knearest` returns an owning packed `Neighbours` (`n * k` ints).
`knearest_into` writes a caller buffer and requires `out_len == n * k`.
Neither returns `std::vector<std::vector<int>>`.
Failure throws `linkcell::Error`, and `Neighbours::neighbour` and
`Neighbours::dist2` throw `std::out_of_range` when `i >= n()` or `j >= k()`.
Requires C++17.

`ShiftedPair` has the layout of `lc_pair`.
`pairs_within` queries the count, then `lc_pairs_within_rows` writes the returned vector from that same search.

## Partition

`lc_cell_grid` writes `nx, ny, nz`, three bin widths, and one point count per cell.
`lc_cell_graph` adds the compressed sparse row (CSR) of the periodic adjacency.
Edge weights are pair counts from `lc_pairs_within` with `half` set.
`lc_partition` writes one domain id per cell.
`method` is 0 Morton, 1 Hilbert, 2 recursive coordinate bisection (RCB), 3 Hilbert plus Fiduccia-Mattheyses, 4 bisection plus that pass, 5 Scotch, 6 multilevel, 7 parallel Scotch.
`balance <= 0` (including `-INFINITY`) selects `1 + parts * w_max / total`, where `w_max` is the heaviest cell weight and `total` is the sum of the cell weights; `NaN`, `+INFINITY`, a value in `(0, 1)`, or a value that scales past `UINT64_MAX` fails.
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

Inputs match `lc_knearest`.
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
| `Overflow` | A size, index, graph weight, open-cell coordinate, or periodic image offset does not fit its representation |
| `BadPartition` | The part count is zero, or the balance bound is not usable |
| `BadGraph` | The cell graph must be a square compressed sparse row (CSR) of that grid |
| `NoDevice` | A device curve split was requested and no device is loaded, the device call failed, or the method is not a curve |
| `ScotchDisabled` | Scotch support was not compiled in |
| `TooManyImages` | The periodic image count exceeds 2^24 |
| `Partial` | An axis is not periodic and `libvesin.so` is not loaded |
| `Device` | The device is not the CPU and `libvesin.so` is not loaded |
| `Padded` | A point has more neighbours than the requested pad |
| `NonFinite` | A coordinate is not finite. `pairs_within_pbc`, `pairs_open`, and the partition entry points return this variant |
| `DeviceMismatch` | The device curve split disagreed with the host weights |
| `ScotchFailed` | The Scotch library rejected the graph |

`Error::Empty` is not a wrong-length buffer.
`knearest` returns `ZeroK`, `Empty`, `MaskLen`, `Overflow`, and `TooManyCells`.
Its `Overflow` is `n * k` not fitting in a slice, and, on a fused-team
build, a point count that does not fit in `i64`.
`knearest_into` reports a product that does not fit as `BufferSize`.
`pairs_within` returns `BadCutoff`, `Empty`, `MaskLen`, `Overflow`,
`TooManyImages`, and `TooManyCells`. Its `Overflow` cases are a point
count above `u32::MAX` (`i32::MAX` for the column entry points), a
relative periodic image that is not finite or whose absolute value
exceeds `i32::MAX`, and an image span on an axis plus
`ceil(cutoff / width)` on that axis that exceeds `i32::MAX`.
`pairs_within_pbc` adds `NonFinite`, and on an open or mixed cell an
`Overflow` when a bin floor or `shift + image_i - image_j` (or its
negation on a full list) does not fit in `i32`. `TooManyImages` is also
`ceil(cutoff / width)` on a periodic axis (or `ceil(cutoff / cell_min)`
on a fully periodic cell) not finite or above `i32::MAX`, and on an
open or mixed cell the stencil reach `ceil(cutoff / edge)` not finite or
above `i32::MAX - 2`.
`halo_width` returns `Overflow` when a positive length over a
positive width is not finite or its ceiling does not fit in `u32`.
`partition` returns `Overflow` for the `pairs_within` image cases when
the method reads arcs, and, with Scotch or PT-Scotch compiled in, when
a cell count, part count, CSR offset, cell weight, or edge weight does
not fit in `i32`.
Positions and cell vectors share one length unit.
`dist2` is that unit squared.
`cutoff` and `cell_hint` are lengths in that unit.

Walk, stop rule, and cost: [algorithm](algorithm.md).
Features and instruction-set builds: [features](features.md).
