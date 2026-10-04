# C ABI

Header: `include/linkcell.h`. Prefix `lc_`. Caller owns every
buffer. C++ (`include/linkcell.hpp`) is a header-only wrap of these
entry points. Device searches are `lc_gpu_*` in
`include/linkcell_gpu.h`; link the package's `linkcell-gpu` target.

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

Lattice vectors are rows `a`, `b`, `c` (the same order vesin uses)
plus an origin. `lc_cell_ortho` builds a diagonal box at the origin.

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

`lc_knearest` is the one-frame, indices-only entry. Returns 0 on success, nonzero on failure.

```c
int lc_pairs_within(const double *xyz, size_t n, const lc_cell *simbox,
                    double cutoff, const int *mask, double cell_hint,
                    int half, int *out_i, int *out_j, int *out_shift,
                    double *out_d2, size_t cap, size_t *out_count);
```

A cutoff list is `lc_pairs_within`. Each row is one atom-image.
`out_shift[3 * t + 0..3]` is `(na, nb, nc)`, the vesin / tonari shift
`S` in the caller's basis. Squared distance is strictly below
`cutoff * cutoff`. `half` nonzero keeps one side of `(i, j, S)`.
Null `out_i`, `out_j`, `out_shift`, and `out_d2` query `*out_count`.
A short `cap` writes that count and returns nonzero.

## Errors and version

```c
const char *lc_last_error(void);
const char *lc_version(void);
```

Both return pointers the caller must not free. `lc_last_error` is
thread-local and `NULL` after a successful `lc_knearest` on that
thread, or if none has failed yet. `lc_version` does not read or write
the slot. The pointer is invalid after the next `lc_knearest` on the
same thread. Distinct searches may run concurrently; each thread reads
its own slot. `lc_version` is process-static.

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
Neither returns `std::vector<std::vector<int>>`. Failure throws
`linkcell::Error`. Requires C++17.

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
| `Overflow` | `n * k` does not fit a slice |

`Error::Empty` is not a wrong-length buffer.

Walk and stop rule: [algorithm](algorithm.md).
