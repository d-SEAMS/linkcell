The mesh fold stores `s + 0.0` after the wrap, so a fractional coordinate of -0 is +0 on every SIMD path.
`Cell::fractional` is still minimage's and can return -0 on x86_64.
