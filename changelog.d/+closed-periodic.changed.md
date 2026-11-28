`PbcQuery::closed` now keeps `dist2 == cutoff²` on a fully periodic cell too, in `pairs_within_pbc`, `lc_pairs_within_pbc`, and the row grid. It was read only with an open axis and ignored on a periodic cell.
The search runs at the next double above the cutoff and its rows are held to `dist2 <= cutoff²`, so a strict query is unchanged.
`pairs_within_csr` still reads `closed` only with an open axis.
