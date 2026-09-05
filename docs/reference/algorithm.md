# Algorithm

Linked-cell k-nearest search in `src/knearest.rs`. Allen and
Tildesley linked cells, a k-heap per source, Chebyshev shells until
the k-th neighbour cannot lie outside the visited cube. The search
takes no cutoff.

## Inputs

- `xyz`: Cartesian points.
- `simbox`: periodic parallelepiped (`Cell` / `lc_cell`).
- `k`: neighbours per source. Must be at least 1.
- `mask`: optional. `false` / zero drops a point as source and as
  candidate.
- `cell_hint`: target bin edge. `None` or `<= 0` uses 3.0, then the
  value is clamped to the smallest perpendicular box width.

Each source keeps `min(k, n_active - 1)` neighbours. Packed outputs
fill unused slots with `-1`.

## Fold and bin

1. Take fractional coordinates in `[0, 1)` (`Cell::fractional`).
2. Store the Cartesian image in the primary cell (`Cell::cartesian`).
3. Bin in fractional space: `floor(s * n*)`, clamped to
   `[0, n* - 1]`.
4. Chain each active point onto a linked list per bin.

`nx, ny, nz` are `floor(width / edge)`, at least 1. `cell_min` is the
smallest of the three actual cell edges (perpendicular width over
count). Orthorhombic boxes wrap each lattice direction independently;
a sheared box uses `Hinv` only for this fold.

## Shells

For each source, walk integer cell offsets `(dx, dy, dz)` in
Chebyshev shells. Shell `reach` is the surface
`max(|dx|, |dy|, |dz|) == reach` (`reach == 1` also visits the home
cell). `max_reach` is `max(nx, ny, nz) / 2 + 1`.

Each offset maps to:

- a primary bin, `rem_euclid` on the cell indices
- a lattice translation, `div_euclid` counts through
  `Cell::lattice_shift`

The pair distance is `Cell::dist2_shifted`: Cartesian subtract of the
folded points plus that shift. The inner loop does not wrap with
`Hinv`.

The same primary bin can appear under more than one wrap. Each wrap
is a separate visit. Skipping those repeats (one shift per unique
bin) misses images. That construction, and the ortho cheap path, is
in [MIC and cells](../explanation/mic-and-cells.md).

## Heap and stop

A max-heap of size `k` stores `(dist2, index)`. For `k <= 16` it
lives on the stack. After each shell, if the heap is full and the
worst `dist2` is at most `(reach * cell_min)^2`, no unvisited point
can beat the k-th neighbour, and the walk stops.

`knearest` returns `Neighbors` rows (`indices`, `dist2`), nearest
first. `knearest_into` / `lc_knearest` write packed indices.
`knearest_into_d2` / `lc_knearest_d2` also write squared distances.
`knearest_into_many` / `lc_knearest_many` cover a frame-major batch.

`knearest_brute` is the all-pairs check used by tests and small
systems. It calls `Cell::dist2_euclidean` per pair: Smith half-edge
test, then a Minkowski-reduced 27-image. The 27-image of an
unreduced H misses lattice points such as `2(a-b)`. The fractional
wrap of a hex-prism body diagonal is not nearest. It is not the
production walk.

## Parallel

The `parallel` Cargo feature (on by default) maps sources with
rayon. Each source owns its heap.

## Device

`linkcell::gpu::Workspace` is the same walk on a CUDA device. Fold,
bin, then a tiled Hillis-Steele exclusive scan (CUB DeviceScan /
HOOMD cell offsets). Occupants are stored in cell-major order with
an O(1) home slot. The stencil is a precomputed Chebyshev shell
table (LAMMPS `NStencil`, HOOMD `d_cell_adj`), not a nested 3-D
loop. Eight threads share each source and stride occupants; after
each shell they merge heaps and apply the host stop. Output is
Cabana's 2-D packed `n * k` list. `knearest_into_many` covers every
frame that shares a cell. Fold uses `Hinv`; the pair shift is
`na a + nb b + nc c`. `k <= 16`.
