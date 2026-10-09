# Changelog

All notable changes to this project are documented in this file.

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Unreleased notes live in [`changelog.d/`](changelog.d/) and are assembled
by [towncrier](https://towncrier.readthedocs.io/).

<!-- towncrier release notes start -->

## [0.3.10] - 2026-10-09

### Fixed

- Use minimage 0.1.4 without its optional C exports so applications can link both static libraries. Test both APIs together across translated periodic images.


## [0.3.9] - 2026-10-09

### Changed

- The mesh build and the cutoff list run on more than one thread. One thread and eight threads write the same rows, in the same order, after each bin is sorted.
- `src/kokkos` builds the bench executable `lc_kokkos_bench` and is not linked into the library.

## [0.3.8] - 2026-10-04

### Fixed

- The dense-shell regression test sorts pair rows by index.

## [0.3.7] - 2026-10-04

### Fixed

- The cutoff list sizes its row buffer from the pairs it found. A shell denser than the ideal-gas estimate no longer writes past that buffer.

## [0.3.6] - 2026-10-04

### Changed

- Orthorhombic brute-force neighbours call minimage's Highway kernel (`dist2_ortho_diffs`). `pairs_within` still records the stencil shift and does not wrap that difference a second time. The batch helpers are re-exported.
- Scale probes always record the POP3 hierarchy: load balance, communication efficiency, parallel efficiency, then computation scaling and global efficiency against the 1-thread run. `scripts/bench-pairs.sh` prints that table for the cutoff list.
- The crate depends on minimage 0.1.3.
- The mesh build is parallel from 8192 active points, and `src/kokkos` runs the certified walk on a Kokkos execution space. Scaling runs report the POP hierarchy: load balance, communication efficiency, parallel efficiency, computation scaling, and global efficiency. On this 8-core host a cubic 262144-point, k=4 search has global efficiency 0.74 at 8 threads (10.7 ms); the Kokkos OpenMP backend has global efficiency 0.56 at 8 threads (14.5 ms). This host has no PMU and no CUDA device.
- `pairs_within` buffers cutoff hits and, with more than one thread and at least 512 atoms, writes disjoint ranges of one pair buffer. A full list still has both shifts. On Linux that buffer stays on the heap so a repeated call does not fault those pages again. On this host a periodic 18 Å cube, 4 Å cutoff, and 4096 atoms is 0.8 ms once the buffer is warm on 8 threads, and the first call is about 4 ms. vesin's cell list tests twice as many distances on that grid; `scripts/cutoff_work.py` reduces the ratio to 2.

### Fixed

- The Meson project version matches the crate version.


## [0.3.5] - 2026-10-04

### Changed

- The cutoff list is on the C, C++, and Python APIs as `lc_pairs_within`, `linkcell::pairs_within`, and `linkcell.pairs_within`. Each row is an atom-image with the caller's shift `S`.
- The minimum-image note now covers closest-vector search, periodic Delaunay triangulations, the robotics kd-tree on a product of circles, tensor sketches, and crystal-graph construction. None of those replaces the cell walk on a filled cubic lattice.
- The minimum-image note now places the walk against Quentrec-Brot, Rapaport's cell shift, the Bentley-Weide-Yao cell technique, and the cutoff lists in LAMMPS, GROMACS, HOOMD, vesin, and freud. The host stop described there is the frontier plane.
- `knearest_into` stops when the k-th neighbour reaches the unvisited plane, including a neighbour that sits on that plane, and writes each packed row from the stack heap.
- `knearest` bins in a Minkowski-reduced basis and stores occupants in cell order. Shells stop on the perpendicular distance to the unvisited frontier, and a cell is skipped when its slab cannot beat the k-th neighbour. `pairs_within` uses the same bins and frontier, and still reports shifts in the caller's basis.
- k-nearest and cutoff walks grow one axis of the index box at a time, and a restricted triclinic cell is tilt-reduced before the walk. Orthorhombic films and LAMMPS tilts no longer take the cubic shell.


## [0.3.4] - 2026-09-27

### Added

- `pairs_within` returns every atom-image pair inside a cutoff with integer cell shift `S`. `knearest` still unique-indexes. Image stencils above 2^24 fail closed.

### Changed

- `knearest_brute` ranks with `Cell::dist2_euclidean` (Smith then Minkowski 27-image).
- The crate depends on minimage 0.1.2 from crates.io.
- CHANGELOG.md follows Keep a Changelog. Unreleased notes are changelog.d fragments assembled by towncrier.

### Fixed

- The MSVC build defines `NOMINMAX` before `windows.h` and calls `(std::min)` / `(std::max)`, so those names are not macros. The header includes `<cassert>` and `<optional>` itself. MSVC does not provide those through the other standard headers.

## [0.3.3] - 2026-08-17

### Changed

- Two wheels: `abi3-py312` (GIL, CPython 3.12+) and `abi3t-py315` (PEP 803, CPython 3.15 GIL and free-threaded). cibuildwheel 4.x builds 3.15.0rc1 by default. PyO3 0.29 `abi3t-py315`; maturin 1.14+ tags the `abi3t` ABI. The 0.3.2 `cp314t` set is gone. cibuildwheel's abi3audit step is off for the abi3t job: current abi3audit flags PEP 793 `PyModExport` and `PyType_FromSlots`.

## [0.3.2] - 2026-08-17

### Changed

- Python `linkcell` module (maturin / PyO3). Arrays are DLPack via dlpk: any `__dlpack__()` object in, `(indices, dist2)` out. `xyz` may be `(n, 3)` or `(n_frames, n, 3)`. `cell` is a DLPack tensor on any device. A CUDA cell is inverted on device; the host reads only the four launch ints. Two wheels: CPython 3.12 limited ABI, and a free-threaded `cp314t` set. Both compile the gpulite device walk. CUDA `__dlpack__` tensors (`torch`) go to `lc_gpu_*` by device pointer; the result is a pair of CUDA DLPack capsules. `lc_knearest_d2` and `lc_knearest_many` write squared distances and a frame-major batch. `lc_gpu_*` is the C waist for the device `Workspace`. A `v*` tag on d-SEAMS/linkcell publishes the two wheel kinds and the sdist to PyPI (trusted publisher, no Actions environment).

## [0.3.1] - 2026-08-16

### Changed

- Device `Workspace` folds with `Hinv` and shifts with `na a + nb b + nc c`. Sheared cells use the host walk. `k <= 16`.

## [0.3.0] - 2026-08-16

### Changed

- Optional gpulite device path: `linkcell::gpu::Workspace::knearest_into` writes the same packed `n * k` indices as the host walk (fold, bin, Chebyshev shells). The device walk uses a tiled exclusive scan, a precomputed Chebyshev stencil, cell-major coordinates with an O(1) home slot, and `knearest_into_many` for a frame batch. Occupancy defaults to threads-per-particle 8 (HOOMD/vesin), overridable by `LINKCELL_TPP` and `LINKCELL_BLOCK`. Setup copies ride the workspace stream. Orthorhombic cells, `k <= 16`. Meson feature `with_gpulite`.

## [0.2.4] - 2026-08-15

### Changed

- A wrap consumer always links the static archive. The parent `default_library` does not change this option, so a shared default left `pydseams.yoda` needing `liblinkcell.so` at import time.

## [0.2.3] - 2026-08-15

### Changed

- A static `libyodaLib.a` no longer passes `liblinkcell.so` to `ar`. The Meson dependency attaches a generated header as the build-order edge, not both cargo outputs.

## [0.2.2] - 2026-08-15

### Changed

- The crate is `d-SEAMS/linkcell`. `HaoZeke/linkcell` redirects.

## [0.2.1] - 2026-08-15

### Changed

- `Error::MaskLen` when `mask` is `Some` and `mask.len() != n`.
- `Error::Overflow` when `n * k` does not fit a slice. `BufferSize`
- remains a caller `out` whose length is not `n * k`.
- `lc_last_error` is written only by `lc_knearest`. `lc_version` does
- not clear the slot. An interior NUL in a message no longer drops the string to `NULL`.
- C++ `Neighbours` owns the packed `n * k` buffer. `knearest_into`
- takes `out_len`. There is no 5-argument `knearest(..., int *out)`.
- CI checks `include/linkcell.h` against cbindgen 0.29.4.
- Branding: sheared linked cells, k=4 neighbours, periodic wrap
- (`assets/branding/`).

## [0.2.0] - 2026-08-15

### Changed

- ABI break. Existing 0.1.x C and C++ callers must rebuild.
- `lc_knearest` takes `size_t n` and `size_t k`. The 0.1.x `int` counts
- overflowed on large frames.
- `lc_last_error` is thread-local. Concurrent searches no longer share
- one process-wide slot. `lc_version` stays process-static.
- C++ `linkcell::knearest` writes a packed `n * k` index buffer (unused
- slots `-1`) and returns a `Neighbours` view. It no longer returns `std::vector<std::vector<int>>`. Failure throws `linkcell::Error`.
- Rust `Error::Empty` is an empty point list only. A wrong-length
- `knearest_into` buffer is `Error::BufferSize`. An overflowing linked-cell mesh is `Error::TooManyCells`.
- The per-source k-set keeps one image of each neighbour (the
- nearest). The same particle visited through two wraps does not occupy two slots.
- `knearest_brute` on a sheared cell takes the 27-image minimum.
- The single parallelepiped wrap is not the Wigner-Seitz cell of a 60-degree hex prism.

## [0.1.2] - 2026-08-15

### Changed

- Fold once, then Cartesian pair distances plus a lattice shift (the vesin / LAMMPS ghost trick). Sources run in parallel. The C ABI writes indices into the caller buffer.

## [0.1.1] - 2026-08-15

### Changed

- Orthorhombic boxes use the three-wrap minimum image, not two 3x3 matvecs. The C ABI reads packed xyz in place. The k-heap for k <= 16 stays on the stack.

## [0.1.0] - 2026-08-15

### Changed

- Periodic linked-cell k-nearest neighbour search. Rust crate, C ABI (`lc_*`), C++ header. The cell is a general parallelepiped; orthorhombic boxes are `Cell::ortho` / `lc_cell_ortho`.
- Installable from Meson (`linkcell_dep`, `pkg.generate`), CMake (`find_package(linkcell)`, `linkcell::linkcell`), and pkg-config (`linkcell.pc`).
