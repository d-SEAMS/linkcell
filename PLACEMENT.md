# Where the partitioner lives

The cell graph, the partition, and the geometric halo belong in
linkcell. The rank share, the ring search, and ice clustering stay in
seams-core. The placement keeps the linkcell graph here and leaves
`seams::domain` in seams-core.

## What each repository owns today

linkcell owns the periodic cell, the fractional bin mesh, the
linked-cell walk, lattice shifts, `pairs_within`, and the gpulite
workspace. Bin counts use the perpendicular widths, so a triclinic
cell is binned in fractional space. The C application binary interface
(`lc_*`) is the Library of Data Exchange (LODE) hourglass. C++ is a
Resource Acquisition Is Initialization (RAII) header over that
interface. Python crosses it with DLPack.

seams-core calls the linkcell mesh and the linked-cell walk from the analysis engine
(cloned from `https://github.com/d-SEAMS/seams-core`, default branch).
`src/neighbours.cpp` calls
`linkcell::pairs_within` and `linkcell::knearest` when
`SEAMS_HAS_LINKCELL` is set. Meson asks for `linkcell >= 0.3.11`.
`src/ring.cpp` and `src/include/internal/ring.hpp` identify primitive
rings (Franzblau) and tag them with the bulk and confined ice
criteria. `src/cluster.cpp` and `clump::Domain` are Stoddard's
connected components of an ice mask. That `Domain` is a cluster, not
a spatial part of the frame.

The linkcell mesh in `src/domain.cpp` already decomposes one analysis frame across ranks.
The Hilbert curve orders the cells.
The cut makes contiguous runs of near equal atom count
(Springel, Monthly Notices of the Royal Astronomical Society (MNRAS)
364, 1105, 2005).
A rank's halo collects every cell within a caller distance of a cell it owns.
`reduceRings` folds the owned rings with the Message Passing Interface (MPI).
The hop length follows the cutoff.
`maxLvl` equals `maxDepth / 2`.
`hops` equals `maxLvl + (maxLvl - 1) / 2`.
The domain grid is a second fractional binning.
`kCellsPerRank`, `kMaxBits`, `kHaloCells`, and `kMaxHaloCells` cap that grid.

## Other consumers

eOn calls `linkcell.pairs_within` for a fully periodic cutoff list.
Vesin covers an open direction.
`neighbor_list_linkcell` is that pair list.
eOn stays on the pair list and leaves seams-core unloaded.

rgpot (`OmniPotentRPC/rgpot`) builds pairs through vesin.
Its docs and sources link to the linkcell documentation from the site nav.
Pair potentials use `rgpot::nlist::PairListCache` over vesin, with an
internal linked-cell search once a box contains enough cells.
A partitioner inside seams-core would sit behind a dependency on the analysis engine, so rgpot would still miss it.

readcon-core (`lode-org/readcon-core`) reads and writes configuration (CON) frames.
It shares the hourglass application binary interface shape.
readcon-core stops at configuration frames.
The only mentions of linkcell are documentation cross-links.

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
is `cutoff * depth`. The bin radius along direction `i` is the smallest
integer `r` with `r * bin_width_i >= length`. The halo statement covers bins, the cutoff, and the minimum image.

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
- `partition` and `lc_partition` return one domain id per cell.
  Methods are a Morton curve, a Hilbert curve, weighted recursive
  coordinate bisection, and a Fiduccia-Mattheyses pass under a balance
  bound. With `balance <= 0`, both use `1 + parts * w_max / total`.
  `w_max` is the heaviest cell weight. `total` is the sum of the cell
  weights. A weighted median assigns that indivisible cell to one side,
  so the bound matches `w_max`. The d-SEAMS
  domain driver derives this ratio for each frame. A caller may
  pass a looser bound. Every system uses the ratio by default.
- `halo_width(cutoff, depth, bin_width)` returns the three bin radii.
- `halo` returns the ghost cell ids of one domain.
- Serial Scotch and parallel Scotch (PT-Scotch) stay optional.
  The default build leaves `libscotch` and `libptscotch` out.
- The device path computes cell weights, curve keys, and the prefix split through gpulite when a Compute Unified Device Architecture (CUDA) driver is present.
  Copies both ways sit inside that time.

seams-core will call this partition interface later:

- build `xyz` from the cloud and call `partition`
- set `depth` from the ring hop count, then call `halo_width` and
  `halo`
- map cell ids to atom indices and fill `Share`
- leave `rings` and the rank reduction as they are

eOn passes cell weights through this partition and halo, and through the ring depth, on a packed frame.
Callers hand over that packed frame.
