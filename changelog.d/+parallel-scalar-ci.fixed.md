Parallel-only builds no longer compile `place` internals that are reachable only through the C ABI, and fused-team tests no longer probe `lc_knearest` from the cdylib when `capi` is off.

Scalar-forced SIMD test runs now assert the selected backend correctly, and CI now exercises `--no-default-features --features parallel` in the cargo and MSRV matrices plus `LINKCELL_SIMD=scalar cargo test --lib`.
