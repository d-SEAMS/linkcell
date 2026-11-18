# Features and instruction-set builds

The crate in `Cargo.toml` is the implementation.
The C ABI, the thread pool, and the wide kernels are selected by Cargo features and by the CPU, not by a `-C target-cpu` flag.

## Cargo features

| Feature | Default | What it builds |
| --- | --- | --- |
| `capi` | yes | The `lc_*` entry points in `include/linkcell.h`. |
| `parallel` | yes | The rayon pool, and on Linux and macOS the OpenMP k-nearest team in `src/knn_region.c`. |
| `minimage-capi` | no | Also exports minimage's C ABI (`minimage_*`) from this library. |
| `tune` | no | `lc_tune_set` and `lc_tune_pairs` for `scripts/tune-pairs.py`. The public header does not declare them. Requires `capi`. |
| `timing` | no | `SimdTiming` (`mesh_ns`, `scan_ns`, `knn_ns`). |
| `scotch` | no | Serial Scotch (`SCOTCH_graphPart`). The library is not vendored. |
| `ptscotch` | no | PT-Scotch (`SCOTCH_dgraphPart`). The library is not vendored. The caller initializes MPI. |

`default` is `capi` and `parallel`.
`--no-default-features` builds the Rust API on one thread and omits the C ABI.
Add `--features capi` when that build still has to export `lc_*`.

minimage is a dependency with `default-features = false`.
The default link of `liblinkcell` omits `minimage_*`.
`minimage-capi` turns those symbols on for a caller that wants one archive.
A static consumer that links minimage itself leaves the feature off, so the two archives do not define the same C symbols.

`scotch` compiles `src/scotch_part.c` and exports `lc_scotch_part`.
`ptscotch` compiles `src/ptscotch_part.c` and exports `lc_ptscotch_part`.
Without the feature, `Method::Scotch` and `Method::PtScotch` return `ScotchDisabled`.

The crate `rust-version` is 1.80.
That is the oldest toolchain that builds the default features: the locked rayon 1.12 requires it.
The Python module in `python/Cargo.toml` has `rust-version` 1.88.
Its features are `scotch` (passes through to the crate), `abi3` (CPython 3.12+ stable ABI), and `abi3t` (CPython 3.15+ free-threaded ABI).

## Instruction-set builds

`build.rs` passes no `-C target-cpu` and no crate-wide `target-feature`.
Each wide kernel carries its own `target_feature` and is called only after `simd::select()`.
`isa-allow.toml` names the symbols allowed to contain those instructions.
The release-library audit in CI checks the object code against that list.

`select` reads a test override on the calling thread, then `LINKCELL_SIMD`, then the widest path this CPU can run.
The environment variable is read once and cached.
Unknown names are ignored.
A name the CPU cannot run selects `scalar`.
Workers use that cached path.
Accepted names are `scalar`, `avx2`, `avx512`, and `neon`.
`avx2fma` is `avx2`, `avx512f` is `avx512`, and `0` is `scalar`.

AVX-512 is selected only when AVX-512F, AVX-512VL, POPCNT, and fused multiply-add (FMA) are all present.
The build compiles those kernels only when rustc is 1.89 or newer (`linkcell_avx512` in `build.rs`).
On 1.80 through 1.88 the same source builds the scalar and AVX2 paths, and NEON on aarch64.
A CPU with AVX-512F and without AVX-512VL uses AVX2.
AVX2 is selected only when FMA is present, so Sandy Bridge stays on `scalar`.
NEON is the aarch64 baseline.
SVE intrinsics are nightly-only, so this crate ships the NEON kernel on aarch64 and no SVE kernel.

Stored distances are separate multiplies and adds, in the scalar order.
FMA is a selection gate and, inside the pair tile, a screen.
The value written for a kept lane is the direct sum of squares.
The same input gives the same stored bits on every path and every thread count.
`simd_path` returns `scalar`, `avx2`, `avx512`, or `neon`.
The [SIMD paths](../explanation/simd-paths.md) page is the dispatch explanation.

## OpenMP team

With `parallel`, Linux and macOS compile `src/knn_region.c`.
A failed probe is a build error on glibc Linux and on macOS.
musl and any other target keep the rayon walk when that region does not build.
macOS needs libomp.
`LIBOMP_PREFIX` names a prefix that contains `include/omp.h` and `lib/libomp.dylib`, ahead of Homebrew.

An unmasked k-nearest query of at least 8192 points uses that team when the region was built.
`lc_set_threads(0)` leaves the width unset.
`lc_set_threads(1)` runs the search on the calling thread and starts no worker.
`OMP_STACKSIZE` (or `GOMP_STACKSIZE`) sets the worker stack.
A size below the floor, 64 KiB or `PTHREAD_STACK_MIN` when that is larger, is raised.
`OMP_THREAD_LIMIT` caps the team.
Under `OMP_DYNAMIC` the load average trims it.
`pin_knearest_to_least_loaded` stays off until the caller sets it in code.
The function ignores the environment.
