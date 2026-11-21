# Changelog

All notable changes to this project are documented in this file.

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Unreleased notes live in [`changelog.d/`](changelog.d/) and are assembled by [towncrier](https://towncrier.readthedocs.io/).

#skip_lint_start

<!-- towncrier release notes start -->
#skip_lint_end

## [0.5.0] - 2026-11-20

### Added

- Add `ReplicaList::from_frames`, `update`, `update_in`, `update_frames`, `needs_rebuild`, `needs_rebuild_in`, `needs_rebuild_frames`, `vectors_of`, `centroid`, cell and query accessors, and `builds`, `reuses` and `last_reused` counters matching `SkinList`. Add `PairColumns::sparsity`, `PairColumns::sort_rows` (returns `Error::BufferSize` for columns of unequal length), `SkinList::sparsity`, `half`, `sorted`, `set_sorted`, `needs_rebuild_in` and `update_in`.
- The Python package has a `numpy` extra: `pip install linkcell[numpy]` installs `numpy>=1.26` for ndarray output. NumPy stays optional, and without it results are DLPack arrays.
- `scripts/seo.py` is the MkDocs hook named in `mkdocs.yml`. After the site is built it gives every HTML page one title, a description, a canonical link, Open Graph and Twitter card tags, and JSON-LD for `SoftwareApplication` and `SoftwareSourceCode`. It writes `sitemap.xml` and a `robots.txt` that allows crawling and names that sitemap. `python3 scripts/seo.py --check` rejects a page that is missing any of those, a duplicate title or description, or a sitemap that does not match the canonical URLs.
- `LIBOMP_PREFIX` names the libomp prefix (`include/omp.h` and `lib/libomp.dylib`) for macOS source builds, ahead of Homebrew's. The macOS wheels build libomp for their deployment target with `scripts/build-libomp-macos.sh` and bundle it.
- `SkinList` now tracks build and reuse counts (`builds`, `reuses`, `last_reused`), reports `max_displacement`, accepts cell updates (`set_cell`) and per-axis periodicity (`with_periodic`), and canonicalizes half-list candidate rows so filtered pair order is independent of skin width.

### Changed

- Fully periodic orthorhombic `SkinList` reuse filters at once when every raw displacement stays inside the primary image and the skin, instead of scanning minimum-image motion and lattice wraps first. Repeated `ReplicaList::filter_all` calls keep that classification for the stored coordinates, cell, and sort mode, and drop it when any of those change.
- The algorithm reference states the cost of the cutoff list and the k-nearest walk, including the triclinic basis and the caps on images and bins. The C reference lists the remaining entry points. A features page records the Cargo flags and the instruction-set builds.
- The documentation site moves to https://linkcell.dseams.info/; the package metadata, README, mkdocs `site_url` and `llms.txt` name the new address.
- The documentation site and the rustdoc pages use the shared d-SEAMS theme. `mkdocs.yml` sets `theme` to Material with `custom_dir: overrides`, and the site loads `docs/stylesheets/dseams-tokens.css`, `dseams-family.css` and `dseams-mkdocs.css`. The family bar is `overrides/partials/dseams-family.html`. rustdoc applies the same tokens and header with `--extend-css assets/rustdoc/dseams-rustdoc.css`, `--html-before-content assets/rustdoc/dseams-rustdoc-before.html` and `--html-in-header assets/rustdoc/dseams-rustdoc-head.html`. The header, palette, type scale and logo follow dseams.info.
- The k-nearest mesh is scattered by bin owner.
  Each worker counts and places the points of its own run of bins, so the build has no atomic counters.
  Each bin still lists its points in ascending order, so the neighbour order is unchanged at every pool width.
  One thread builds the mesh serially.
  The mesh buffers are kept once for the process, so a search on any worker reuses them.
  The output rows are filled by the pool before a large search.
- `SkinList` and `ReplicaList` on a fully periodic cell measure rebuild motion with minimum-image displacement (`Cell::dist2`) and unwrap boundary-crossing coordinates onto the reference or centroid image. Wrap counts are computed in `i64` and only the final row shift must fit `i32` (`Error::Overflow` otherwise). Non-finite coordinates or recomputed distances return `Error::NonFinite`, and any failed update leaves rows, references and counters unchanged. A changed cell no longer forces a rebuild: the candidates are kept while twice the largest displacement from the fractionally carried reference plus `|H' H0^-1 - I| (cutoff + skin)` stays within `skin`, which is valid for any triclinic cell. `ReplicaList::filter_all` and `update_bead` rebuild or refilter every bead after `set_cell` or `set_sorted`. `SkinList` with an open axis keeps raw displacement and the box-corner correction. Public API break: `ReplicaList::sparsity` returns `Vec<(usize, usize)>` instead of `Vec<(u32, u32)>`.

### Fixed

- A clang `parallel` build finds libomp when clang answers `-print-file-name=libomp.so` with the bare name, as Ubuntu's clang does for the libomp beside its resource directory (`<llvm>/lib/clang/<v>`), instead of stopping on "finding libomp failed".
- A k-nearest worker that cannot take its CPU place is started again with the same stack size and the caller's mask. If that start fails, linkcell creates no worker on the default stack. The query runs on the calling thread.
- An `OMP_STACKSIZE` below the worker stack floor is raised to the floor. linkcell warns once per process. The workers use that explicit size. The floor is 64 KiB, or `PTHREAD_STACK_MIN` when that is larger. The wide fold frame does not fit in 16 KiB, so a size that pthread accepts is raised too when it is below the floor.
  `pthread_attr_setstacksize` returns `EINVAL` for a size. linkcell warns once per process and remembers the refusal. Later queries stay on the calling thread.
  `pthread_create` may fail for an `OMP_STACKSIZE`, including with `EAGAIN`. linkcell warns once per process. That query stays on the calling thread. The next query tries the size again.
- Meson and CMake find `libomp` beside the Clang resource directory when `-print-file-name` prints the bare name.
- Meson links libomp or libgomp by name, with `-L` for the directory it found the runtime in, so `linkcell.pc` keeps `-lomp` where it had started to name the file's full path on Linux.
- Meson, CMake, and Cargo now require and link Homebrew libomp for threaded macOS builds, while Meson discovers external Clang headers before its Linux OpenMP probe.
- Parallel-only builds no longer compile `place` internals that are reachable only through the C ABI, and fused-team tests no longer probe `lc_knearest` from the cdylib when `capi` is off.

  Scalar-forced SIMD test runs now assert the selected backend correctly, and CI now exercises `--no-default-features --features parallel` in the cargo and MSRV matrices plus `LINKCELL_SIMD=scalar cargo test --lib`.
- Rayon workers started after `lc_set_places` may each run on any cpu of the list instead of one cpu each. Tied to a cpu that another process was using, a worker held up the search while work stealing could not move it: the 110592-molecule k-nearest search beside an 8-thread spinner took 78 ms instead of 36. Idle timings are unchanged.
- The Rust API section of the site has an index page; the rustdoc help and settings pages linked one that did not exist.
- The `scotch` and `ptscotch` sources include `stdio.h` before the Scotch headers. Scotch 7.0.11 uses `FILE` in those headers and does not include `stdio.h` itself.
- The k-nearest team is now linkcell's own pthreads. `shutdown_knearest_team`, `lc_knn_shutdown`, and the unload destructor join every worker under LLVM libomp. They also join every worker when `libgomp.so.1` resolves to libomp. No thread is left behind. Each worker is named `linkcell-knn` for as long as it lives.
  OpenMP is only read. The team size follows libgomp for a region the caller opens. The size starts from `omp_get_max_threads()`, the load average trims it under `OMP_DYNAMIC`, and `OMP_THREAD_LIMIT` caps it. Worker stacks follow `OMP_STACKSIZE`, or `GOMP_STACKSIZE`, as libgomp parses it. Workers bind to `OMP_PLACES` under `OMP_PROC_BIND`. Neighbour rows are unchanged.
  A team of one runs on the calling thread. For that query it takes the pinned CPU, or the place of slot 0, and then gets its own mask back. A standing team is joined when the size drops to one.
  Under libomp, clang uses libomp's bind kind and place list. A `libgomp.so.1` that resolves to libomp does the same. With `OMP_PLACES` set and `OMP_PROC_BIND` unset, libomp reports `spread` where libgomp reports `true`. The workers take that placement.
- The k-nearest team now compiles on macOS with libomp, and its searches complete there. Its barrier sleeps on a condition variable where Linux uses a futex, and `pin_knearest_to_least_loaded` and the OpenMP place list leave workers where the scheduler puts them, since macOS has no CPU affinity.
- `NeighborCalc::compute` updates its stored `SkinList` cell on every call. A changed cell no longer forces a rebuild: the candidates are kept while twice the largest displacement from the fractionally carried reference plus `|H' H0^-1 - I| (cutoff + skin)` stays within `skin`, which is valid for any triclinic cell. The stored list no longer reuses shifts from the initial cell.
- `SkinList` and `ReplicaList` store packed periodic shift codes in `[-3, 3]^3` as `u16` instead of `u8`, so codes `256..=342` (`sx >= 2`) no longer wrap in release builds or panic on overflow in debug builds.
- `rust-version` in `Cargo.toml` is now 1.80, the oldest Rust that builds and
  tests the default `parallel` feature: the locked rayon 1.12 and rayon-core 1.13
  need it, and 1.79 fails to compile rayon. It was 1.70, which no build with the
  default features could meet. The `scotch` and `ptscotch` externs no longer use
  `unsafe extern` blocks (Rust 1.82), `bw_roof` only uses AVX-512 intrinsics when
  `build.rs` sets `linkcell_avx512`, and its runtime gate falls back unless the
  CPU has AVX-512F, AVX-512VL, POPCNT, and FMA. The `open_cell` test requires
  `capi`, so `cargo test --no-default-features` builds. CI builds and tests on
  exactly that toolchain, and `scripts/check-versions.py` checks the README
  against it.
- linkcell depends on minimage with `default-features = false` and exposes
  optional `minimage-capi` when callers need minimage's C ABI; the default
  build no longer pulls minimage's C ABI (`minimage_*`) into `liblinkcell`.

## [0.4.2] - 2026-11-16

### Changed

- The threads of the k-nearest OpenMP team are named `linkcell-knn` while they run linkcell's region, and take their own names back when it ends, so `top -H` and `/proc/<pid>/task/*/comm` tell them apart from threads the OpenMP runtime keeps.

### Fixed

- After `lc_set_threads(1)`, `lc_cell_grid`, `lc_cell_graph` and `lc_partition` bin the points on the calling thread, and a build without `parallel` bins them on one thread. Binning a frame of 8192 points or more started a thread per CPU for the length of the call, and a smaller frame started one thread.
- Meson links the OpenMP runtime by name, `omp` under Clang and `gomp` otherwise, as `build.rs` and CMake do, so its `linkcell.pc` says `-lomp` or `-lgomp` instead of `-fopenmp` and a wrap consumer's own sources are no longer compiled with `-fopenmp`. Configure stops when the C compiler has no `omp.h`.
- The `lc_set_threads` documentation in `linkcell.h` says that a width of one runs the whole search on the calling thread and starts no thread, and that zero leaves the width unset. A build with `parallel` and without `capi` compiles its tests again.
- The column fill of `lc_pairs_within` runs on the width chosen with `lc_set_threads`, like its search and the row fill of `lc_pairs_within_rows`. At width one it no longer builds Rayon's global pool at the CPU count.
- `build.rs` finds the OpenMP runtime in `LIBRARY_PATH` when the compiler does not report it. Clang's `-print-file-name` does not search `LIBRARY_PATH`, so a Clang build against a libomp reached only through it, such as an Archer runtime for ThreadSanitizer, stopped on "finding libomp failed" although the region compiled and `-lomp` would link.

## [0.4.1] - 2026-11-16

### Fixed

- A Clang build links libomp instead of libgomp. The Meson and CMake builds hand their C compiler to the cc crate, and `build.rs`, the link line and `linkcell.pc` name the OpenMP runtime of that compiler, so one process no longer loads both runtimes.
- A fully open frame reuses one stencil for every frame with the same bin edge and cutoff, so band images that occupy other bins no longer rebuild it. The stencil is the farthest x offset within reach on each row of bins, and the walk clips the rows to the bins the frame spans, so a small molecule in a few bins is no slower than with the per-bin link list, and a reach of many bins builds rows only out to the span the frames need.
- A native Linux build with no C compiler chosen compiles the OpenMP k-nearest region with gcc, so the `parallel` build no longer stops on a missing `omp.h` when `cc` is clang.
  `CC`, `TARGET_CC` and `CC_<target>` still choose the compiler.
- A search asked to run on one thread (`lc_set_threads(1)`) stays on the calling thread. The grid, pair, partition and skin walks read that width instead of rayon's global pool, the fused k-nearest region runs every phase on the caller, and the cell grid and cell graph entry points use the same width. An unset width (zero) keeps rayon's and OpenMP's defaults.
- Meson and CMake rebuild the library when any crate source changes. Meson reads the sources from the Cargo dep-info, and CMake also tracks `build.rs` and the C sources.

## [0.4.0] - 2026-11-15

### Added

- Add a native partitioner for the periodic cell grid, with Morton and Hilbert prefix splits, weighted recursive coordinate bisection, a Fiduccia-Mattheyses pass, optional Scotch, and a gpulite curve split.
- Open clusters use occupied Cartesian bins. An open axis is not padded into a periodic box, and the partner stencil is rebuilt only when the bounding bins change. `lc_stencil_builds` counts builds. `pairs_open` is the Cartesian half list.
- The partitioner cuts the cell grid from measured per-cell weights and keeps a ghost only when it reaches an owned atom.
  `lc_owned_mask` marks owned points 2 and points in a walked cell 1, and `lc_within_reach` marks ghosts within a reach of an owned atom.
- `LINKCELL_SANITIZE=thread` makes the Meson build rebuild the crate and std with nightly `-Zsanitizer=thread`, so a C++ ThreadSanitizer link sees the Rayon joins.
- `SkinList` keeps candidates out to cutoff plus skin and filters them while every atom stays within skin/2.
  `ReplicaList` builds one candidate list on the ring centroid and filters every bead from it.
  `lc_vesin_neighbors` follows the vesin C layout for a fully periodic CPU list and loads the platform vesin library otherwise.
  A half list with many neighbours per atom is counted per thread and written from the tile kernel.
- `simd_path` reports the Single Instruction, Multiple Data (SIMD) path selected for the mesh fold and the k-nearest scan.
  The `timing` feature exports `SimdTiming` with `mesh_ns`, `scan_ns`, and `knn_ns`.
  `LINKCELL_SCAN_TIME` counts the scan.
  `LINKCELL_COARSE_TIME` counts the search and the mesh build.

### Changed

- An unmasked k-nearest query of at least 8192 points bins, counts, scans, fills, and selects on one OpenMP team. Each thread keeps its own histogram and scratch. Phase edges wait on linkcell's own futex. The team is joined from a library destructor when the image is unloaded or the process exits, and the next query rebuilds it when `omp_get_max_threads()` has changed. The library does not change the process OpenMP wait policy.
  Packed rows match the previous walk bit for bit.
  `pin_knearest_to_least_loaded` optionally places those workers on the least-loaded CPUs. It is off unless the caller asks, and it does not read an environment variable.
- Cutoff pairs use the same Single Instruction, Multiple Data (SIMD) path as the mesh fold and the k-nearest scan.
  AVX2 and the Neon extension (NEON) run the pair cull and the pair tile.
  AVX-512 is selected only when VL, POPCNT, and fused multiply-add (FMA) are present as well.
  Every stored squared distance is the direct sum of squares, so the bits match the scalar path.
  `LINKCELL_SIMD` still forces the path.
  `simd_path` reports it for the fold, the scan, and the pairs.
- The k-nearest scan and the CSR merge run over z-slabs of the cell list on one Rayon pool.
- The mesh fold is on AVX2 or the Neon extension (NEON), and so is the k-nearest distance scan, when AVX-512 is absent.
  `simd_path` reports the path in use.
  `LINKCELL_SIMD=scalar|avx2|avx512|neon` forces a path.
  AVX2 is selected only when fused multiply-add (FMA) is present.
  The kernels run without enabling FMA.
  The fold floor follows minimage `floor_fast`, truncation on every architecture.
  On x86_64 the AVX-512 wrap now matches scalar for an input of -0.
  The previous round toward -inf mapped that input to +0.
- `lc_set_threads(0)` runs the next search on the calling thread instead of selecting one Rayon thread.
  A negative width still selects one thread.
- linkcell depends on minimage 0.2, whose `floor_fast` truncates on every architecture.

### Fixed

- The NEON fold truncates like minimage `floor_fast`, so an input of -0 keeps a
  fraction of -0 on aarch64 and matches the scalar path.
- The `lc_set_continuation` hook belongs to the thread that set it.
  A k-nearest search on another thread no longer runs a callback it did not set.
- linkcell links against musl: the glibc `mallopt` page knobs are set only on
  glibc targets.
- A k-nearest block search keeps the smaller index when a neighbour on the gathered block face ties the k-th distance, as the shell walk does.
- An open-axis pair query whose bin reach does not fit in `i32` returns `TooManyImages` instead of overflowing.
- The device partition finds the GPU curve split through `lc_set_gpu_curve_split`, which the GPU library calls when it loads, so the crate links on macOS and with MSVC without a weak stub.
- `include/linkcell.h` no longer carries the vesin entry points (they are in `include/linkcell_vesin.h`) or the internal OpenMP schedule.

## [0.3.11] - 2026-10-09

### Fixed

- Preserve integer cell shifts relative to unwrapped input positions in Rust, C and Python cutoff pair lists.
- Verify the registry package during publication.

## [0.3.10] - 2026-10-09

### Fixed

- Use minimage 0.1.4 without its optional C exports so applications can link both static libraries.
  Test both APIs together across translated periodic images.


## [0.3.9] - 2026-10-09

### Changed

- The mesh build and the cutoff list run on more than one thread.
  One thread and eight threads write the same rows, in the same order, after each bin is sorted.
- `src/kokkos` builds the bench executable `lc_kokkos_bench` and is not linked into the library.

## [0.3.8] - 2026-10-04

### Fixed

- The dense-shell regression test sorts pair rows by index.

## [0.3.7] - 2026-10-04

### Fixed

- The cutoff list sizes its row buffer from the pairs it found.
  A shell denser than the ideal-gas estimate no longer writes past that buffer.

## [0.3.6] - 2026-10-04

### Changed

- Orthorhombic brute-force neighbours call minimage's Highway kernel (`dist2_ortho_diffs`).
  `pairs_within` still records the stencil shift and leaves that difference unwrapped.
  The batch helpers are re-exported.
- Scale probes always record the parallel overhead profile (POP) as the POP3 hierarchy: load balance, communication efficiency, parallel efficiency, then computation scaling and global efficiency against the 1-thread run.
  `scripts/bench-pairs.sh` prints that table for the cutoff list.
- The crate depends on minimage 0.1.3.
- The mesh build is parallel from 8192 active points, and `src/kokkos` runs the certified walk on a Kokkos execution space.
  Scaling runs report the POP hierarchy: load balance, communication efficiency, parallel efficiency, computation scaling, and global efficiency.
  On this 8-core host a cubic 262144-point, k=4 search has global efficiency 0.74 at 8 threads (10.7 ms); the Kokkos OpenMP backend has global efficiency 0.56 at 8 threads (14.5 ms).
  This host has no performance monitoring unit (PMU) and no CUDA device.
- With more than one thread and at least 512 atoms, `pairs_within` writes disjoint ranges of one pair buffer for the buffered cutoff hits.
  A full list still has both shifts.
  On Linux that buffer stays on the heap so a repeated call reuses those pages.
  On this host a periodic 18 Å cube, 4 Å cutoff, and 4096 atoms is 0.8 ms once the buffer is warm on 8 threads, and the first call is about 4 ms. vesin's cell list tests twice as many distances on that grid.
  `scripts/cutoff_work.py` reduces the ratio to 2.

### Fixed

- The Meson project version matches the crate version.


## [0.3.5] - 2026-10-04

### Changed

- The cutoff list is on the C, C++, and Python APIs as `lc_pairs_within`, `linkcell::pairs_within`, and `linkcell.pairs_within`.
  Each row is an atom-image with the caller's shift `S`.
- The minimum-image note now covers closest-vector search, periodic Delaunay triangulations, the robotics kd-tree on a product of circles, tensor sketches, and crystal-graph construction.
  None of those replaces the cell walk on a filled cubic lattice.
- The minimum-image note now places the walk against Quentrec-Brot, Rapaport's cell shift, the Bentley-Weide-Yao cell technique, and the cutoff lists in LAMMPS, GROMACS, Highly Optimized Object Oriented Molecular Dynamics (HOOMD), vesin, and freud.
  The host stop described there is the frontier plane.
- `knearest_into` stops when the k-th neighbour reaches the unvisited plane, including a neighbour that sits on that plane.
  Each packed row is written from the stack heap.
- `knearest` bins in a Minkowski-reduced basis and stores occupants in cell order.
  Shells stop on the perpendicular distance to the unvisited frontier.
  A cell is skipped when its slab exceeds the distance to the k-th neighbour.
  `pairs_within` uses the same bins and frontier, and still reports shifts in the caller's basis.
- k-nearest and cutoff walks grow one coordinate of the index box at a time.
  A restricted triclinic cell is tilt-reduced before the walk.
  Orthorhombic films and LAMMPS tilts no longer take the cubic shell.


## [0.3.4] - 2026-09-27

### Added

- `pairs_within` returns every atom-image pair inside a cutoff with integer cell shift `S`.
  `knearest` still unique-indexes.
  Image stencils above 2^24 fail closed.

### Changed

- `knearest_brute` ranks with `Cell::dist2_euclidean` (Smith then Minkowski 27-image).
- The crate depends on minimage 0.1.2 from crates.io.
- CHANGELOG.md follows Keep a Changelog.
  Unreleased notes are changelog.d fragments assembled by towncrier.

### Fixed

- The MSVC build defines `NOMINMAX` before `windows.h` and calls `(std::min)` / `(std::max)`, so those names are not macros.
  The header includes `<cassert>` and `<optional>` itself.
  MSVC leaves those names out of the other standard headers.

## [0.3.3] - 2026-08-17

### Changed

- Two wheels: `abi3-py312` (GIL, CPython 3.12+) and `abi3t-py315` (Python Enhancement Proposal (PEP) 803, CPython 3.15 GIL and free-threaded). cibuildwheel 4.x builds 3.15.0rc1 by default. PyO3 0.29 `abi3t-py315`; maturin 1.14+ tags the `abi3t` ABI. The 0.3.2 `cp314t` set is gone. cibuildwheel's abi3audit step is off for the abi3t job: current abi3audit flags PEP 793 `PyModExport` and `PyType_FromSlots`.

## [0.3.2] - 2026-08-17

### Changed

- Python `linkcell` module (maturin / PyO3).
  Arrays are DLPack via dlpk: any `__dlpack__()` object in, `(indices, dist2)` out.
  `xyz` may be `(n, 3)` or `(n_frames, n, 3)`.
  `cell` is a DLPack tensor on any device.
  A CUDA cell is inverted on device; the host reads only the four launch ints.
  Two wheels: CPython 3.12 limited ABI, and a free-threaded `cp314t` set.
  Both compile the gpulite device walk.
  CUDA `__dlpack__` tensors (`torch`) go to `lc_gpu_*` by device pointer; the result is a pair of CUDA DLPack capsules.
  `lc_knearest_d2` and `lc_knearest_many` write squared distances and a frame-major batch.
  `lc_gpu_*` is the C waist for the device `Workspace`.
  A `v*` tag on d-SEAMS/linkcell publishes the two wheel kinds and the sdist to PyPI (trusted publisher, no Actions environment).

## [0.3.1] - 2026-08-16

### Changed

- Device `Workspace` folds with `Hinv` and shifts with `na a + nb b + nc c`.
  Sheared cells use the host walk.
  `k <= 16`.

## [0.3.0] - 2026-08-16

### Changed

- Optional gpulite device path: `linkcell::gpu::Workspace::knearest_into` writes the same packed `n * k` indices as the host walk (fold, bin, Chebyshev shells).
  The device walk uses a tiled exclusive scan, a precomputed Chebyshev stencil, cell-major coordinates with an O(1) home slot, and `knearest_into_many` for a frame batch.
  Occupancy defaults to threads-per-particle 8 (HOOMD/vesin), overridable by `LINKCELL_TPP` and `LINKCELL_BLOCK`.
  Setup copies ride the workspace stream.
  Orthorhombic cells, `k <= 16`.
  Meson feature `with_gpulite`.

## [0.2.4] - 2026-08-15

### Changed

- A wrap consumer always links the static archive.
  The parent `default_library` leaves this option unchanged, so a shared default left `pydseams.yoda` needing `liblinkcell.so` at import time.

## [0.2.3] - 2026-08-15

### Changed

- A static `libyodaLib.a` no longer passes `liblinkcell.so` to `ar`.
  The Meson dependency attaches a generated header as the build-order edge, not both cargo outputs.

## [0.2.2] - 2026-08-15

### Changed

- The crate is `d-SEAMS/linkcell`.
  `HaoZeke/linkcell` redirects.

## [0.2.1] - 2026-08-15

### Changed

- `Error::MaskLen` when `mask` is `Some` and `mask.len() != n`.
- `Error::Overflow` when `n * k` overflows a slice.
  `BufferSize`
- remains a caller `out` whose length is not `n * k`.
- `lc_last_error` is written only by `lc_knearest`.
  `lc_version` leaves the slot unchanged.
  An interior null byte (NUL) in a message no longer drops the string to `NULL`.
- C++ `Neighbours` owns the packed `n * k` buffer.
  `knearest_into`
- takes `out_len`.
  The API omits a 5-argument `knearest(..., int *out)`.
- CI checks `include/linkcell.h` against cbindgen 0.29.4.
- Branding: sheared linked cells, k=4 neighbours, periodic wrap
- (`assets/branding/`).

## [0.2.0] - 2026-08-15

### Changed

- ABI break.
  Existing 0.1.x C and C++ callers must rebuild.
- `lc_knearest` takes `size_t n` and `size_t k`.
  The 0.1.x `int` counts
- overflowed on large frames.
- `lc_last_error` is thread-local.
  Concurrent searches no longer share
- one process-wide slot.
  `lc_version` stays process-static.
- C++ `linkcell::knearest` writes a packed `n * k` index buffer (unused
- slots `-1`) and returns a `Neighbours` view.
  It no longer returns `std::vector<std::vector<int>>`.
  Failure throws `linkcell::Error`.
- Rust `Error::Empty` is an empty point list only.
  A wrong-length
- `knearest_into` buffer is `Error::BufferSize`.
  An overflowing linked-cell mesh is `Error::TooManyCells`.
- The per-source k-set keeps one image of each neighbour (the
- nearest).
  The same particle visited through two wraps occupies one slot.
- `knearest_brute` on a sheared cell takes the 27-image minimum.
- The single parallelepiped wrap is not the Wigner-Seitz cell of a 60-degree hex prism.

## [0.1.2] - 2026-08-15

### Changed

- Fold once, then Cartesian pair distances plus a lattice shift (the vesin / LAMMPS ghost trick).
  Sources run in parallel.
  The C ABI writes indices into the caller buffer.

## [0.1.1] - 2026-08-15

### Changed

- Orthorhombic boxes use the three-wrap minimum image, in place of two three-by-three matvecs.
  The C ABI reads packed xyz in place.
  The k-heap for k <= 16 stays on the stack.

## [0.1.0] - 2026-08-15

### Changed

- Periodic linked-cell k-nearest neighbour search.
  Rust crate, C ABI (`lc_*`), C++ header.
  The cell is a general parallelepiped; orthorhombic boxes are `Cell::ortho` / `lc_cell_ortho`.
- Installable from Meson (`linkcell_dep`, `pkg.generate`), CMake (`find_package(linkcell)`, `linkcell::linkcell`), and pkg-config (`linkcell.pc`).
