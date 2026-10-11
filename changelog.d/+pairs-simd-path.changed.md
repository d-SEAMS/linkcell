Cutoff pairs use the same SIMD path as the mesh fold and the k-nearest scan.
AVX2 and NEON run the pair cull and the pair tile. AVX-512 is selected only when VL, POPCNT, and FMA are present as well.
Every stored squared distance is the direct sum of squares, so the bits match the scalar path.
`LINKCELL_SIMD` still forces the path. `simd_path` reports it for the fold, the scan, and the pairs.
