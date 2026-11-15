The partitioner cuts the cell grid from measured per-cell weights and keeps a ghost only when it reaches an owned atom.
`lc_owned_mask` marks owned points 2 and points in a walked cell 1, and `lc_within_reach` marks ghosts within a reach of an owned atom.
