The k-nearest mesh is scattered by bin owner.
Each worker counts and places the points of its own run of bins, so the build has no atomic counters.
Each bin still lists its points in ascending order, so the neighbour order is unchanged at every pool width.
One thread builds the mesh serially.
The mesh buffers are kept once for the process, so a search on any worker reuses them.
The output rows are filled by the pool before a large search.
