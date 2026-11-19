`NeighborCalc::compute` updates its stored `SkinList` cell on every call so periodic box changes rebuild the candidate list instead of reusing shifts from the initial cell.
