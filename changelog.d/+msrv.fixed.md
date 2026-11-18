`rust-version` in `Cargo.toml` is now 1.80, the oldest Rust that builds and
tests the default `parallel` feature: the locked rayon 1.12 and rayon-core 1.13
need it, and 1.79 fails to compile rayon. It was 1.70, which no build with the
default features could meet. The `scotch` and `ptscotch` externs no longer use
`unsafe extern` blocks (Rust 1.82), `bw_roof` only uses AVX-512 intrinsics when
`build.rs` sets `linkcell_avx512`, and its runtime gate falls back unless the
CPU has AVX-512F, AVX-512VL, POPCNT, and FMA. The `open_cell` test requires
`capi`, so `cargo test --no-default-features` builds. CI builds and tests on
exactly that toolchain, and `scripts/check-versions.py` checks the README
against it.
