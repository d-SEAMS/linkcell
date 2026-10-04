# Algorithm

Linked-cell k-nearest search in `src/knearest.rs`. Allen and
Tildesley linked cells, a k-heap per source, and a rectangular index
box that grows until the k-th neighbour cannot lie past an unvisited
face. The search takes no cutoff.

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
4. Count occupants per bin, exclusive-scan the counts, and store the
   active indices in cell order.

`nx, ny, nz` are `floor(width / edge)`, at least 1. `cell_min` is the
smallest of the three actual cell edges (perpendicular width over
count). Orthorhombic boxes wrap each lattice direction independently;
a sheared box uses `Hinv` only for this fold.

## Shells

For each source, walk integer cell offsets `(dx, dy, dz)`. The first
visit is the 3×3×3 around the home bin. After that the index box
grows one axis at a time: an axis whose unvisited plane is already
farther than the k-th neighbour stays put, and a short face can take
another layer while the long faces do not. A cube shell is the case
where all three planes fail together. `max_reach` is the space
diagonal of the walk cell, in units of the shortest bin edge, and at
least half the longest bin count.

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

## Basis

An orthorhombic box is already the basis the shells walk.
A restricted triclinic box (LAMMPS dump bounds, a CON file with a
non-right angle, the cells seams and rgsaddle pass) is tilt-reduced
in place, GROMACS `correct_box`, and the shift is the triangular
product. A general orientation is Minkowski-reduced (Nguyen–Stehlé).
The basis spans the same Cartesian lattice. A 27-image check on that
basis can still miss a closer shift; the shell cap is the space
diagonal divided by the minimum cell height, and the frontier test
stops the walk once the k-th neighbour is certified.
`pairs_within` keeps the caller's basis, because the returned shift
`S` is an integer combination of that H. Each unordered pair is
tested once. Hits from that walk are buffered, then the rows are
written in one pass. A full list writes both `(i, j, S)` and
`(j, i, -S)`. With more than one thread and at least 512 atoms, each
thread searches a slice of cells and writes a disjoint range of the
same pair buffer. On Linux that buffer stays on the heap, so a
repeated call does not fault those pages in again.
The C, C++, and Python entries are that list (`lc_pairs_within`,
`linkcell::pairs_within`, `linkcell.pairs_within`). A cutoff
neighbour list is this call.

## Heap and stop

A max-heap of size `k` stores `(dist2, index)`, ordered
lexicographically so an equal distance keeps the smaller index. For
`k <= 16` it lives on the stack. Occupants of a bin are a contiguous
slice. A cell is skipped when the perpendicular distance from the
source to that image's slab is already at least the worst heap entry.

After each shell, if the heap is full and the worst `dist2` is at
most the squared perpendicular distance to the nearest unvisited
lattice plane, the walk stops. Bins are half-open, so every unvisited
point lies strictly past that plane: a neighbour that sits on the
plane is still the nearest, and shrinking the plane would walk another
shell. The older `reach * cell_min` bound is that distance when the
source sits on the outer face of its cell; a source in the interior
stops sooner. The device walk still uses `reach * cell_min`, which is
a lower bound on the same plane, so it does not stop earlier.

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

The citations are in [MIC and cells](../explanation/mic-and-cells.md).
The pair kernel is Rapaport's cell shift. The expanding shells are
Bentley, Weide, and Yao's cell technique. Schnorr–Euchner enumeration
and McKilliam, Grant, and Clarkson's obtuse-superbase search certify
the closest lattice vector; they do not list the k nearest sites.
A periodic Delaunay triangulation certifies the nearest site, not the
fourth. A kd-tree on a product of circles (Yershova and LaValle) is
exact for one neighbour on an orthorhombic torus. Crystal-graph
builders take a cutoff, or k neighbours inside a ball.

## Parallel

linkcell is a sub-library. An MPI caller keeps the primary
communicator and passes a sub-communicator. MPI setup stays in that
caller. The POP figures below are the threads inside one search: the
share that sub-communicator already assigned to this call.

The `parallel` Cargo feature (on by default) maps sources with
rayon. Each source owns its heap. From 8192 active points upward the
same feature builds the mesh in parallel: a thread writes each point's
fractional coordinate once, histograms with atomics, then scatters
occupants with atomics. Bins are sorted by index afterwards, so the
occupant order matches the serial build. Below that count the mesh
stays serial. The cubic hot path is 4096 points and does not take the
parallel build. `--no-default-features` keeps the whole search serial.

`examples/knn_scale.rs` times one fixed problem. The box grows with
`n` so the spacing stays 3.125. `RAYON_NUM_THREADS` has to be set
before the process starts. `LINKCELL_POP=1` records useful time per
worker: the monotonic clock runs only inside that worker's mesh and
walk loops, not while it waits for another worker. Load balance is
the average of that time over the maximum, communication efficiency
is the maximum over the wall time, and parallel efficiency is their
product. Strong-scaling computation scaling is the 1-thread useful
total over the useful total at the larger thread count, per
repetition. Global efficiency is parallel efficiency times that
computation scaling. `scripts/bench-strong.sh` prints the hierarchy.
Instruction scaling, IPC scaling, and frequency scaling need a PMU.
This host's `perf_event_open` has no PMU, so those three are not
reported.

## Device

`linkcell::gpu::Workspace` is the gpulite CUDA walk. It still stops on
`reach * cell_min`, which is a lower bound on the host plane, so it
does not stop earlier than the host. `k <= 16`. That path is separate
from the Kokkos walk.

`src/kokkos/` is the certified walk on `Kokkos::DefaultExecutionSpace`.
The caller starts Kokkos. The standalone timer calls
`Kokkos::initialize` because that program is the whole process. A
caller that already started Kokkos keeps doing so, and an MPI caller
still passes a sub-communicator. Fold, bin, exclusive scan, and the
per-axis index box run as Kokkos parallel loops. The stop is the host
plane test, not `reach * cell_min`.
`k <= 16`. The sources are not in the Cargo or CMake graph, so a
build without Kokkos stays the same. A Cuda execution space uses this
source; the timings in the changelog were taken with the OpenMP
backend, because the machine that measured them has no CUDA device.
`LC_KOKKOS_POP=1` records thread CPU time for each OpenMP static
slice of the fill, the scatter, and the walk, plus the host copies
into and out of the views. One sample covers the slice. The exclusive
scan and the view allocation stay outside that clock, so they widen
the communication-efficiency gap. The hierarchy is the same one as
the host script, and the same missing PMU means no instruction, IPC,
or frequency scaling. `scripts/bench-kokkos.sh` builds that bench
against a Kokkos install and prints the hierarchy.
