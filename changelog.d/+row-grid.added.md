Add `RowGrid`: the bins of one cutoff search, kept so its rows can be read without a stored pair list.
`RowGrid::build` bins the points for a `PbcQuery` on an orthorhombic, triclinic, open, or mixed cell, and `update` rebins in place, reusing the buffers.
`for_rows(begin..end, half, visit)` calls `visit(&Row)` for every row entry of those points, and `for_point(i, half, visit)` for one point; `try_for_rows` and `try_for_point` take a visit that returns `ControlFlow` to stop the walk.
A `Row` holds `i`, `j`, the shift, the vector `r_j - r_i` with the shift applied, and the squared distance.
The entries are the rows of `pairs_within_pbc` for the same query, with the same shifts and the same bits in every distance, in bin order, and the memory of the grid grows with the points and bins, not the pairs.
A visit takes `&self` and writes nothing in the grid, so several threads can visit disjoint ranges at once; the order inside a row does not depend on the split.
`RowGrid::candidate_bound` (`lc_row_grid_bound`) is an upper bound on the full-row entries from the bin occupancy alone, for a caller that must decide whether a stored list fits in memory, and `bins` (`lc_row_grid_bins`) is the bin count.
The C ABI adds `lc_row_grid`, `lc_row`, `lc_row_visit`, `lc_row_grid_build`, `lc_row_grid_build_request`, `lc_row_grid_update`, `lc_row_grid_len`, `lc_row_grid_free`, `lc_row_grid_rows`, and `lc_row_grid_point`; a nonzero return of the visitor stops the walk and is returned.
`lc_row_grid_fill` copies the same entries into a caller buffer from an `lc_row_cursor`, one call per buffer; `RowGrid::fill_rows` is the Rust form.
`lc_row_grid_build_request` takes the declaration as a Cap'n Proto `NeighborRequest` (cutoff, skin, cut, periodicity, box, shell, bin edge, origin, self-image exclusion) and a mask buffer beside it; it is the Cargo feature `request`, on by default, which adds the `capnp` crate and raises the minimum Rust version to 1.81.
The C++ header adds `linkcell::RowGrid`, `Row`, and `RowQuery`, with `for_rows` and `for_point` taking a closure and `fill_rows` reading into a vector.
