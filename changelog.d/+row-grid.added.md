Add `RowGrid`: the bins of one cutoff search, kept so its rows can be read without a stored pair list.
`RowGrid::build` bins the points for a `PbcQuery` on an orthorhombic, triclinic, open, or mixed cell, and `update` rebins in place, reusing the buffers.
`for_rows(begin..end, half, visit)` calls `visit(i, j, shift, dist2)` for every row entry of those points, and `for_point(i, half, visit)` for one point.
The entries are the rows of `pairs_within_pbc` for the same query, with the same shifts and the same bits in every distance, and the memory of the grid grows with the points and bins, not the pairs.
A visit takes `&self` and writes nothing in the grid, so several threads can visit disjoint ranges at once; the order inside a row does not depend on the split.
The C ABI adds `lc_row_grid`, `lc_row_visit`, `lc_row_grid_build`, `lc_row_grid_update`, `lc_row_grid_len`, `lc_row_grid_free`, `lc_row_grid_rows`, and `lc_row_grid_point`.
