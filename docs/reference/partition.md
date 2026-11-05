# Cell partition

The mesh is the one `pairs_within` builds for the same cutoff and cell hint.
Triclinic cells are binned in fractional space, using the perpendicular widths.
A cell weight is the number of points in that bin.
An edge joins two bins when a cutoff pair has one end in each, and the weight is the number of such pairs.
Periodic images are already folded, so an edge that crosses a face of the box is an edge of this graph.

`partition` returns one domain id per cell. The methods are:

| Code | Name | Cut |
| --- | --- | --- |
| 0 | `morton` | Morton order, weighted prefix |
| 1 | `hilbert` | Hilbert order, weighted prefix |
| 2 | `rcb` | Weighted recursive coordinate bisection (RCB) |
| 3 | `hilbert-fm` | Hilbert, then Fiduccia-Mattheyses |
| 4 | `rcb-fm` | Recursive coordinate bisection, then Fiduccia-Mattheyses |
| 5 | `scotch` | `SCOTCH_graphPart`, feature `scotch` only |
| 6 | `multi` | Heavy-edge coarsening after the `rcb-fm` cut |
| 7 | `ptscotch` | `SCOTCH_dgraphPart`, feature `ptscotch` only |

The default method is `rcb-fm`.
On the five-system panel it has the smallest sum of edge cuts among the native methods that stay within the balance bound on every system.
A tie breaks toward the smaller sum of median wall times.

With `balance: None`, and with `lc_partition` when `balance <= 0`, the bound is `1 + parts * w_max / total`.
`w_max` is the heaviest cell weight and `total` is the sum of the cell weights.
A weighted median assigns that indivisible cell to one side, so a part may exceed the ideal share `total / parts` by at most `w_max`.
That is the tightest bound a cell-aligned cut can promise.
The d-SEAMS domain driver derives the same ratio for each frame, and this entry point uses that rule so the two stay consistent.
The same formula is used for every system.
The caller can pass another bound of at least 1.
A cell is never split.

`halo_width(cutoff, depth, bin_width)` returns three bin radii.
The length is `cutoff * depth`.
Axis `i` gets the smallest integer `r` with `r * bin_width[i]` at least that length.
`depth` is the number of cutoff shells the caller must cover.
`halo` lists the cells inside that radius of a domain, periodically, and omits the domain's own cells.
The halo fraction is the largest ghost-cell count divided by the owned-cell count.
A value above 1 means the ghost set is larger than the owned set.

The device path computes the histogram of bin ids, the curve keys, and the exclusive prefix of the ordered weights.
Host-to-device and device-to-host copies sit inside that call.
Without a CUDA driver the call returns `NoDevice`.
The host process (HOST) does not report a device time in that case.
The log line is `gpu_curve HOST NOT-RUN`.

Serial Scotch is not vendored.
`--features scotch` links the system `libscotch` and calls `SCOTCH_graphPart`.
The strategy is `SCOTCH_STRATQUALITY` together with `SCOTCH_STRATBALANCE`.
The imbalance argument is `balance - 1`.
Before the strategy is built, the code calls `SCOTCH_randomReset` and then `SCOTCH_randomSeed(1)`.

Parallel Scotch (PT-Scotch) uses the Message Passing Interface (MPI).
It is the `ptscotch` feature, also off by default and not vendored.
The panel times `SCOTCH_dgraphPart` under `mpirun` on one node.
A second node is not available here, so that case is `multi_node HOST NOT-RUN`.

## Placement

Ring membership, the Message Passing Interface (MPI) share, and ice clustering stay in seams-core.
See `PLACEMENT.md`.
