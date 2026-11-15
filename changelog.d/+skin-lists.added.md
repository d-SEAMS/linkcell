`SkinList` keeps candidates out to cutoff plus skin and filters them while every atom stays within skin/2.
`ReplicaList` builds one candidate list on the ring centroid and filters every bead from it.
`lc_vesin_neighbors` follows the vesin C layout for a fully periodic CPU list and loads the platform vesin library otherwise.
A half list with many neighbours per atom is counted per thread and written from the tile kernel.
