Run the mesh fold and the k-nearest distance scan on AVX2 or NEON when AVX-512 is absent.
`simd_path` reports the path in use.
`LINKCELL_SIMD=scalar|avx2|avx512|neon` forces a path.
AVX2 is selected only when FMA is present. The kernels do not enable FMA.
The fold floor follows minimage `floor_fast` for this architecture.
On x86_64 the AVX-512 wrap now matches scalar for an input of -0. The previous round toward -inf mapped that input to +0.
