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
tested once. A full list writes both `(i, j, S)` and `(j, i, -S)`.
On Linux the pair buffer stays on the heap, so a repeated call does
not fault those pages in again.
The C, C++, and Python entries are that list (`lc_pairs_within`,
`linkcell::pairs_within`, `linkcell.pairs_within`). A cutoff
neighbour list is this call.

The bins are one fold, one count per key, a scan, and a fill of the
slot columns in bin order: positions, positions relative to the bin
corner, their squared length, and the atom index. The fold writes
position and key columns, with the key formed in registers from the
eight-wide fold; the scan steps bin indices rather than dividing; then
the atom numbers are sorted by key and every slot column is filled
front to back, so the random accesses are reads of the fold. For 4096
atoms on one thread the bins take 0.038 ms (0.044 ms shuffled), against
0.061 and 0.082 ms with the key formed per lane and a scatter into the
slot columns. Inside a bin the
slots follow a Morton code of the atom's sub-cell, two or four per axis
so that a sub-cell holds about one atom, then atom index. Eight slots
in a row, one vector of targets, are then a compact block of the bin
whatever order the caller's atoms come in; with the atoms of a lattice
shuffled, the one-thread 4096-atom call took 1.67 ms with that order
and 2.21 ms in atom order when the order was added. The fold takes eight points at a time on
AVX-512 with minimage's own operations in its order (the division or
the `Hinv` product, `wrap01`, the bin truncation, and `H`), so every
position and bin is the same bits as `Cell::fractional` and
`Cell::cartesian`; the k-nearest mesh uses the same fold. When the walk
splits across threads, from 1024 atoms each thread folds one block of
atoms and groups it by bin in its own staging area. Each thread then
owns a run of bins, balanced by atoms, reads their atoms from every
staging area in thread order, sorts each bin by sub-cell with a stable
count, and writes its run of slots alone, so the order is the same as
one thread's and no cache line is written by two threads. The first
pass also wakes the workers the search uses next. A grid lends its slot columns, offsets, corners, fold, and
counts back when it drops, and the next grid of a similar size takes
them without zeroing them, since every value is written before it is
read. Which bins each bin looks into, with their shifts and corner
offsets, depends only on the box, the bin counts, the reach, and the
cutoff; the last such stencil is kept under that key, and the walk
reads each target bin's occupancy from the offsets.

The walk splits across threads from an ideal-gas estimate of ten
thousand pairs (on this 8-core host one thread is faster at 512 atoms
in an 18 Å cube at 4 Å, eight from 768). A split call reserves the row
buffer on the caller's thread from that estimate, then builds the bins,
searches, and writes in one pool entry: the caller does not run the
serial steps while every worker spins, and does not wake between
search and write. A short estimate grows the buffer on the caller's
thread. glibc gives a large block allocated on a worker its own heap
and unmaps it when the block is freed, so the buffer the caller keeps
is allocated where the caller runs.

The distance tile is four sources against eight targets. A source
`p`, moved into the target bin's frame, is `p'`; each lane forms
`|r_q|^2 - 2 r_q . p'` with three fused multiply-adds, and adding
`|p'|^2` gives the squared distance. A lane within a rounding bound of
`cutoff^2` is decided by the direct `|q - (p - S)|^2`, so the rows are
exactly the rows of the direct formula; the bound is a short sum of
`2^-53` times the squares of the largest coordinates, shifts, and bin
offsets, about `2e-10` Å² on an 18 Å cube. `dist2` of a row agrees
with the direct value to within that bound.

On one thread, a full list on AVX-512 is written straight from the
tile: hit lanes are compressed, and four hits become eight 40-byte
rows (or sixteen column entries) in five registers. The rows are dword
permutes of two registers, one holding the eight compressed targets,
the source, and both shifts, the other the eight compressed distances,
so the second four hits of a vector permute from the same pair. A half
list is written from the tile too, one row per hit on the side
`keep_half` keeps: one compare of the source against the compressed
targets picks each hit's side (a target with the source's own index,
which only an image of the source's own bin holds, takes the shift's
side), and each register of rows is one permute through an index
blended between the forward and the mirrored table, so eight hits are
five registers; the columns blend `i`, `j`, the shift, and `dist2` the
same way. Before, a half list searched into the hit buffer and wrote
each row with a scalar branch on its side, and took longer than the
full list (4096 atoms: 1.33 ms, now 0.82 ms; 0.3.8 1.88 ms). When
bins hold under eight atoms on average the block loop runs as a twin
compiled with the tile's target features and the tile inlines into it,
since with a few atoms a block the call into the tile is most of the
block (256 atoms in the 18 Å cube: 0.020 to 0.017 ms).
Otherwise each thread buffers the hits of its range of bins, then
writes them into the caller's layout: `Pair` rows, four columns, or
`lc_pair` rows. Thread `k` searches and writes range `k`, so the writer
reads its own cache.

On one core the 29 MB of 40-byte rows for 4096 atoms are most of a
one-thread call. A row store whose line is out in L3 waits at the head
of the store buffer for the line, and once the buffer fills, the tile
behind it stalls too, so the walk and the row stream took turns: the
same walk with its rows folded into a ring that stays in L2 took
0.93 ms, the call 1.47 ms. Every writer now prefetches the line
`AHEAD_ROWS` (52, about 2 KB of rows) past each store once its output
passes 3 MB a thread, so the store finds its line in L1: the tile for
rows and for columns, the buffered AVX-512 writers, and the scalar row,
half-list, column, and C writers. The one-thread call drops to 1.15 to
1.21 ms. Smaller lists stay in a 2 MB L2 and only pay the instructions,
so 256 and 1024 atoms do not prefetch; from 1536 atoms the gain is 10
to 16%, on two and four threads 2 to 4%, and on eight threads none,
where the cores share the L3's bandwidth rather than each waiting on its
own fill buffers. A store-only probe bounds the rest: straight 64-byte
stores of those 29 MB with the same prefetch take 0.95 ms (a memset
0.85 ms), and the same stores in the tile's own bursts, two to ten
registers a hit vector, 1.03 to 1.16 ms, depending on how predictable
the burst lengths are. The walk is within that range, and dropping both
compresses, or broadcasting each group's sources from memory, does not
change it: the tile's own work now runs behind the stores. Staging the
rows in L1 and copying them out in 1 to 4 KB bursts, or streaming every
few lines past the cache, was slower. On Linux the `Pair` vector and the
columns of at least 4 MB that a call allocates are advised to use
transparent huge pages on their 2 MB-aligned interior, so those stores
walk the page tables once per 2 MB (without the advice the call takes
1.32 ms); buffers a caller passes in are left alone.

The bindings write their own layout from the same hits: Python and
`pairs_within_columns` get the `ijS` columns without a `Pair` vector,
`lc_pairs_within` writes the caller's columns, and a count query keeps
its search for the matching fill on the same thread.
`lc_pairs_within_rows` writes `lc_pair` rows, which the C++
`ShiftedPair` vector uses directly.

With the bin edge equal to the cutoff the search radius is 1, and the
gap between any two cells inside that stencil is 0, so the bin test
does not drop a cell. vesin's cell list then evaluates
`27 ρ − 1` ordered distances per atom (`ρ` atoms in the cell). This
walk evaluates `(ρ − 1) / 2 + 13 ρ`, which is exactly half:
`scripts/cutoff_work.py` simplifies the ratio to 2. vesin also forms
`S H` with a 3×3 product on every candidate. This walk forms that
shift once per cell pair. The extra traffic is the hit buffer and the
40-byte row.

GROMACS nbnxn stores a cluster-pair list and an interaction bitmask,
and the force kernel reads that list. This call returns one atom-image
row, so its clusters are decided per search. With each bin in Morton
order, a run of eight targets from a bin's first slot and a run of four
sources are compact clusters, and the bins record each run's bounding
box in the bin's relative coordinates. For a source run and a partner
bin, the tile moves the source box into the target frame and tests it
against eight target boxes in one vector; a target run whose box is at
least `cutoff^2` plus twice the rounding bound away is not loaded.
Every source lies inside its moved box (rounding is monotonic) and
every target inside its own, so a skipped run holds no lane the tile
would keep: the rows, their order, and their bits do not change, and a
test compares them against the tile with the boxes off. The boxes are
built when the bins hold four runs of eight on average and tested for
partner bins of four runs or more; with fewer, the bin stencil has
already culled at that scale and the box test costs more than it skips
(it made the 256- and 1024-atom calls 20 to 40% slower). For the
4096-atom cube at 4 Å, 64 atoms per bin, 82% of the 444 thousand
source-vectors held no pair; with the boxes 124 thousand are tested, and
the one-thread call drops from 1.49 to 1.42 ms. Those vectors were
cheap, three fused multiply-adds and a compare mostly hidden behind the
row stores, so the gain is the compute that did not overlap. An earlier
box test per 4×8 tile, one box pair at a time, was slower than the tile
alone. A cluster bitmask would still expand into the same row.

Columns half a cutoff across along `b` and `c`, each sorted along `a`
in eighth-cutoff bins, with a window into every neighbour column for
each four sources, test about half as many vectors as the 27-bin
stencil (200 thousand against 442 thousand source-vectors for that
cube). They were slower on this host: about 1.9 ms on one thread and
0.48 ms on eight, against 1.6 ms and 0.38 ms. Almost every vector in a
window holds a hit, so each one pays the compress and the row permutes,
and those share the one 512-bit shuffle port; the 27-bin tile skips
most vectors after three fused multiply-adds and a compare.

The walk's thresholds and paths can be autotuned. The `tune` feature
builds `lc_tune_set` (expected pairs where the walk splits, atoms from
which a split walk builds its bins on several threads, the one-thread
full list written from the tile or buffered first, bin ranges per
thread, and the output bytes per thread from which the writers
prefetch) and `lc_tune_pairs`, a timed call that returns a checksum of its
rows; the shipped header declares neither. `scripts/tune-pairs.py`
drives them with Kernel Tuner, one compiled C function per
configuration, and checks every configuration's rows against the
default one. On this host, for the 4096-atom cube, the 1080
configurations on eight threads (bin edge times both thresholds times
one to four ranges per thread) and the 18 on one thread all put the
defaults within timing noise of the best; a jittered lattice and 1024
atoms on eight threads agree. Another host can rerun the script.

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
systems. An orthorhombic box calls `dist2_ortho_diffs` (the Highway
batch: SoA differences, one reciprocal per axis, then the per-axis
round). Any other box calls `Cell::dist2_euclidean` per pair: Smith
half-edge test, then a Minkowski-reduced 27-image. The 27-image of an
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
caller. The POP3 figures below are the threads inside one search:
the share that sub-communicator already assigned to this call.

The `parallel` Cargo feature (on by default) maps sources with
rayon. Each source owns its heap. Sources go in bin order, reading
their own fraction and position and every candidate's position from
slot-ordered copies, and each writes its own row of the output: a
neighbour's shell is then still in cache, whatever order the caller's
points come in. Sources go in blocks of two bins a side (one when bins
hold four points or more), and a block gathers the points of the bins
one beyond it once, each moved by its image shift as the shell walk
moves it. A source forms all its distances eight at a time; the
lane-wise minimum over them is eight distances of eight different
points, so its k-th smallest bounds the k-th nearest, and only points
at or under it reach the heap. When the k-th neighbour is within the
plane bound of the gathered bins the answer is the shell walk's, since
every other point lies past one of those planes; otherwise the shell
walk resumes from the heap at its second layer. A test checks indices
and `dist2` bits against the shell walk. For the 262144-point cube and
k = 4 one thread takes 37 ms and eight 6.3 ms (64 and 15.6 ms with the
points shuffled), against 66 and 11 ms (251 and 40 ms) for 0.3.8. From
8192 active points upward the same feature builds the mesh in parallel:
a thread writes each point's fractional coordinate and flat bin once
and counts it into an atomic counter; the offsets are a parallel scan
over runs of bins (a mesh can have more bins than points), the same
counters become scatter cursors, and crowded bins are sorted by index
in parallel, so the occupant order matches the serial build. Below that
count the mesh stays serial. The cubic hot path is 4096 points and does
not take the parallel build. `--no-default-features` keeps the whole
search serial.

`examples/knn_scale.rs` and `examples/pairs_scale.rs` time one fixed
problem and always record the POP3 hierarchy. `RAYON_NUM_THREADS` has
to be set before the process starts. The box in the k-nearest probe
grows with `n` so the spacing stays 3.125. Useful time is the
monotonic clock inside that worker's mesh and walk loops, not while
it waits for another worker. Load balance is the average of that time
over the maximum, communication efficiency is the maximum over the
wall time, and parallel efficiency is their product. Strong-scaling
computation scaling is the 1-thread useful total over the useful
total at the larger thread count, per repetition. Global efficiency
is parallel efficiency times that computation scaling.
`scripts/bench-strong.sh` and `scripts/bench-pairs.sh` print the
hierarchy. Instruction scaling, IPC scaling, and frequency scaling
need a PMU. This host's `perf_event_open` has no PMU, so those three
are not reported. Serialization and transfer are not split: there is
no ideal-network model, so communication efficiency stays one number.

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
The Kokkos bench always records thread CPU time for each OpenMP static
slice of the fill, the scatter, and the walk, plus the host copies
into and out of the views. One sample covers the slice. The exclusive
scan and the view allocation stay outside that clock, so they widen
the communication-efficiency gap. The hierarchy is the same one as
the host script, and the same missing PMU means no instruction, IPC,
or frequency scaling. `scripts/bench-kokkos.sh` builds that bench
against a Kokkos install and prints the hierarchy.
