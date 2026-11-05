# Where the partitioner lives

The cell graph, the partition, and the geometric halo belong in
linkcell. The rank share, the ring search, and ice clustering stay in
seams-core. This is a split, not a move of the whole of
`seams::domain`.

## What each repository owns today

linkcell owns the periodic cell, the fractional bin mesh, the
linked-cell walk, lattice shifts, `pairs_within`, and the gpulite
workspace. Bin counts use the perpendicular widths, so a triclinic
cell is binned in fractional space. The C application binary interface
(`lc_*`) is the Library of Data Exchange (LODE) hourglass. C++ is a
Resource Acquisition Is Initialization (RAII) header over that
interface. Python crosses it with DLPack.

seams-core (cloned from `https://github.com/d-SEAMS/seams-core`,
default branch) is the analysis engine. `src/neighbours.cpp` calls
`linkcell::pairs_within` and `linkcell::knearest` when
`SEAMS_HAS_LINKCELL` is set. Meson asks for `linkcell >= 0.3.11`.
`src/ring.cpp` and `src/include/internal/ring.hpp` identify primitive
rings (Franzblau) and tag them with the bulk and confined ice
criteria. `src/cluster.cpp` and `clump::Domain` are Stoddard's
connected components of an ice mask. That `Domain` is a cluster, not
a spatial part.

`src/domain.cpp` and `src/include/internal/domain.hpp` already
decompose one frame across ranks. Cells are ordered on a Hilbert
curve and cut into contiguous runs of near equal atom count
(Springel, Monthly Notices of the Royal Astronomical Society (MNRAS)
364, 1105, 2005). A rank's halo is every cell within
a caller distance of a cell it owns. `rings`, `gatherRings`, and
`reduceRings` then keep the rings whose lowest-index member the rank
owns, and fold them with the Message Passing Interface (MPI). The hop length is analysis-specific:
`maxLvl = maxDepth / 2` and `hops = maxLvl + (maxLvl - 1) / 2`, times
the cutoff. The grid itself is not the linkcell mesh. It is a second
fractional binning with fixed caps (`kCellsPerRank`, `kMaxBits`,
`kHaloCells`, `kMaxHaloCells`).

## Other consumers

eOn (`TheochemUI/eOn`, `docs/source/user_guide/neighbor_lists.md`)
calls `linkcell.pairs_within` for a fully periodic cutoff list and
keeps vesin when an axis is open. `neighbor_list_linkcell` is that
list and nothing else. eOn does not import seams-core.

rgpot (`OmniPotentRPC/rgpot`) does not call linkcell. Its docs and
sources link to the linkcell documentation from the site nav. Pair
potentials use `rgpot::nlist::PairListCache` over vesin, with an
internal linked-cell search once a box holds enough cells. A
partitioner inside seams-core would be unreachable from rgpot without
a dependency on the analysis engine.

readcon-core (`lode-org/readcon-core`) reads and writes configuration (CON) frames.
It shares the hourglass application binary interface shape. It does not call linkcell. The
only mentions are documentation cross-links.

The dependency already runs from seams-core to linkcell. Reversing
it would put Eigen, Franzblau, and `PointCloud` under every neighbour
caller.

## Geometry versus analysis

A part is a set of linkcell bins. A cell weight is the number of
points in that bin. An edge weight is the number of `pairs_within`
rows whose endpoints fall in those two bins, periodic images
included. Those three objects are properties of the mesh and the
cutoff list.

The halo of a part is the set of bins that can hold a point within a
caller length of an owned point under the minimum image. The length
is `cutoff * depth`. The bin radius on axis `i` is the smallest
integer `r` with `r * bin_width_i >= length`. That statement does not
mention rings, cages, or ranks.

What is analysis, and stays in seams-core:

- `Share` (`local`, `owned`, `owner`) on a `PointCloud`
- the ring hop count that chooses `depth`
- `rings`, `gatherRings`, `reduceRings`, `RingSummary`
- Message Passing Interface collectives
- `subCloud`
- `clump::Domain` and the ice-mask component

## Interface

linkcell, in Rust, C (`lc_*`), C++ Resource Acquisition Is Initialization, and Python (DLPack):

- `cell_grid` returns `nx, ny, nz`, the bin widths, and a point count
  per cell. The edge is the one `pairs_within` uses for the same
  cutoff and hint.
- `cell_graph` adds the periodic adjacency. Each undirected edge's
  weight is the pair count from `pairs_within` (`half` set).
- `partition` returns one domain id per cell. Methods are a Morton
  curve, a Hilbert curve, weighted recursive coordinate bisection,
  and a Fiduccia-Mattheyses pass under a balance bound. The bound is
  an argument. The default is `1 + parts * heaviest_cell / total_weight`
  for every system.
- `halo_width(cutoff, depth, bin_width)` returns the three bin radii.
- `halo` returns the ghost cell ids of one domain.
- Parallel Scotch (PT-Scotch) is the `ptscotch` feature, off by default, and is not
  vendored. Serial Scotch is the `scotch` feature, also off by default.
  The default build does not link either library.
- The device path, when a Compute Unified Device Architecture (CUDA) driver is present, computes cell
  weights, curve keys, and the prefix split through gpulite. Copies
  both ways sit inside that time.

seams-core, later, not in this tree:

- build `xyz` from the cloud and call `partition`
- set `depth` from the ring hop count, then call `halo_width` and
  `halo`
- map cell ids to atom indices and fill `Share`
- leave `rings` and the Message Passing Interface reduction as they are

eOn and any other periodic caller use the same entry points on
packed coordinates and a cell. They do not take a `PointCloud`.
