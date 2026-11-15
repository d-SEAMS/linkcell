# Contributing to linkcell

linkcell builds periodic linked-cell neighbour lists. The Rust crate in
`src/` is the implementation. Every other language reaches it through the
C ABI in `include/linkcell.h`, which cbindgen writes from `src/capi.rs`.

## Layout

| Path | What it holds |
|---|---|
| `src/` | The crate: binning (`bins.rs`), cutoff pairs (`pairs.rs`, `open_cell.rs`), k-nearest (`knearest.rs`, `knn_region.c`), skin lists, partition, SIMD dispatch (`simd.rs`), the C ABI (`capi.rs`) |
| `include/` | `linkcell.h` (generated, never edited by hand) and the C++ header `linkcell.hpp` |
| `python/`, `linkcell/` | The DLPack Python module and its package |
| `tests/` | Rust integration tests, the CMake consumer, the static-minimage link test, Python tests |
| `docs/` | Diataxis pages: tutorials, how-to, reference, explanation |
| `changelog.d/` | towncrier fragments, one per change |
| `isa-allow.toml` | Functions allowed to carry wide instructions, all runtime-dispatched |

## Commands

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --all-targets --no-default-features --features capi -- -D warnings
cargo test --locked
cargo test --locked --no-default-features --features capi
scripts/regen-capi-headers.sh          # after any change to src/capi.rs
scripts/regen-capi-headers.sh --check
python3 scripts/check-llms-txt.py
meson setup build && meson compile -C build && meson test -C build
```

CI runs the cargo matrix on Linux x86_64, Linux aarch64, macOS and
Windows, the Meson and CMake builds, the Python tests, the header check,
and an instruction-set audit of the release library against `isa-allow.toml`.

## Rules

- Every stored distance and fraction is the same bits on the scalar,
  AVX2, AVX-512 and NEON paths and at every thread count. A new kernel
  comes with a test that compares it to the scalar path.
- A new wide kernel is reached only through `simd::select()` and gets an
  `isa-allow.toml` entry with its reason.
- `include/linkcell.h` changes only through `scripts/regen-capi-headers.sh`.
- A behaviour change carries a fragment in `changelog.d/`
  (`+slug.added.md`, `.changed.md`, `.fixed.md`).
- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`,
  `docs:`, `build:`, `ci:`, `chore:`).
- minimage is a crates.io dependency at a released version, never a git pin.

## Releases

A release commit sets the version in
`Cargo.toml`, `python/Cargo.toml`, `pyproject.toml`, `meson.build`,
`CITATION.cff`, the wrap snippets in `README.md` and `docs/how-to/embed-c.md`,
and `tests/python/test_knearest.py`, then runs
`towncrier build --version X.Y.Z`. Pushing the `vX.Y.Z` tag publishes the
crate to crates.io and the wheels to PyPI. seams-core then pins that tag
in `subprojects/linkcell.wrap`.

## Related repositories

- [lode-org/minimage](https://github.com/lode-org/minimage): cell geometry and the floor the fold follows.
- [d-SEAMS/seams-core](https://github.com/d-SEAMS/seams-core): the analysis engine that calls this library.
