# Cell partition

Our method computes a domain id for every cell.
`pairs_within` builds the mesh for that cutoff and cell hint.
Triclinic cells use fractional bins, and the bin counts follow the perpendicular widths.
A cell weight counts the points in that bin.
An edge joins two bins.
The edge weight counts the cutoff pairs with one end in each bin.
Folded periodic images stay in the cell graph when a pair crosses a face.

`partition` returns the domain id.
Morton order is code 0 and cuts the cells with a weighted prefix.
Hilbert order is code 1 and uses that prefix on the cells.
Weighted recursive coordinate bisection (RCB) is code 2.
Code 3 adds Fiduccia-Mattheyses after the Hilbert order.
Code 4, `rcb-fm`, refines the bisection.
`SCOTCH_graphPart` is code 5 when the `scotch` feature is on.
Heavy-edge coarsening after the `rcb-fm` cut is code 6, `multi`.
`SCOTCH_dgraphPart` is code 7 when the `ptscotch` feature is on.

Our method computes the default bisection on those cells with code 4, `rcb-fm`.
On the five-system panel that cut has the smallest sum of edge cuts among the native cuts that stay inside the balance bound on every system.
A tie breaks toward the smaller sum of median wall times.
We report that choice from the panel log.

With `balance: None`, and with `lc_partition` when `balance <= 0`, the bound is `1 + parts * w_max / total`.
`w_max` is the heaviest cell weight and `total` is the sum of the cell weights.
A weighted median assigns that indivisible cell to one side.
A part may therefore exceed the ideal share `total / parts` by at most `w_max`.
The bound matches that heaviest cell.
The d-SEAMS domain driver derives this ratio for each frame, and `lc_partition` uses the ratio so the two stay consistent.
Every system uses this formula.
The caller can pass another bound of at least 1.
A cell stays whole.

`halo_width(cutoff, depth, bin_width)` returns three bin radii for that cell graph.
The length is `cutoff * depth`.
Direction `i` gets the smallest integer `r` with `r * bin_width[i]` at least that length.
`depth` counts the cutoff shells the caller must cover.
`halo` lists the cells inside that radius of a domain, periodically, and omits the domain's own cells.
The halo fraction is the largest ghost-cell count divided by the owned-cell count.
A value above 1 means the ghost set is larger than the owned set.

The device path computes the histogram of bin ids, the curve keys, and the exclusive prefix of the ordered weights.
Host-to-device and device-to-host copies sit inside that call.
Without a CUDA driver the call returns `NoDevice`.
The host process (HOST) leaves the device time out of the log in that case.
The log line is `gpu_curve HOST NOT-RUN`.

The same device log also records serial Scotch, which uses the system `libscotch` behind `--features scotch`.
The feature calls `SCOTCH_graphPart`.
The strategy is `SCOTCH_STRATQUALITY` together with `SCOTCH_STRATBALANCE`.
The imbalance argument is `balance - 1`.
Before the strategy is built, the code calls `SCOTCH_randomReset` and then `SCOTCH_randomSeed(1)`.

Parallel Scotch (PT-Scotch) uses the Message Passing Interface (MPI).
It is the `ptscotch` feature, off by default, and the build leaves the library out.
The panel times `SCOTCH_dgraphPart` under `mpirun` on one node.
This machine has one node, so the log line for a second node is `multi_node HOST NOT-RUN`.

## Placement

Ring membership, the Message Passing Interface (MPI) share, and ice clustering stay in seams-core.
See `PLACEMENT.md`.
