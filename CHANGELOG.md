# Changelog

## Unreleased

- `knearest_brute` / `pair_dist2` use `Cell::dist2_euclidean` (Smith
  half-edge test, then a Minkowski-reduced 27-image). A 27-image of
  an unreduced H misses lattice points outside `{-1,0,1}^3`. A
  hex-prism body diagonal is a fractional wrap that is not nearest.

## 0.3.3

Two wheels: `abi3-py312` (GIL, CPython 3.12+) and `abi3t-py315`
(PEP 803, CPython 3.15 GIL and free-threaded). cibuildwheel 4.x
builds 3.15.0rc1 by default. PyO3 0.29 `abi3t-py315`; maturin
1.14+ tags the `abi3t` ABI. The 0.3.2 `cp314t` set is gone.
cibuildwheel's abi3audit step is off for the abi3t job: current
abi3audit flags PEP 793 `PyModExport` and `PyType_FromSlots`.

## 0.3.2

Python `linkcell` module (maturin / PyO3). Arrays are DLPack via
dlpk: any `__dlpack__()` object in, `(indices, dist2)` out.
`xyz` may be `(n, 3)` or `(n_frames, n, 3)`. `cell` is a DLPack
tensor on any device. A CUDA cell is inverted on device; the host
reads only the four launch ints. Two wheels: CPython 3.12 limited
ABI, and a free-threaded `cp314t` set. Both compile the gpulite device
walk. CUDA `__dlpack__` tensors (`torch`) go to `lc_gpu_*` by
device pointer; the result is a pair of CUDA DLPack capsules.
`lc_knearest_d2` and `lc_knearest_many` write squared distances
and a frame-major batch. `lc_gpu_*` is the C waist for the device
`Workspace`. A `v*` tag on d-SEAMS/linkcell publishes the two
wheel kinds and the sdist to PyPI (trusted publisher, no Actions
environment).

## 0.3.1

Device `Workspace` folds with `Hinv` and shifts with
`na a + nb b + nc c`. Sheared cells use the host walk. `k <= 16`.

## 0.3.0

Optional gpulite device path: `linkcell::gpu::Workspace::knearest_into`
writes the same packed `n * k` indices as the host walk (fold, bin,
Chebyshev shells). The device walk uses a tiled exclusive scan, a precomputed
Chebyshev stencil, cell-major coordinates with an O(1) home slot,
and `knearest_into_many` for a frame batch. Occupancy defaults to
threads-per-particle 8 (HOOMD/vesin), overridable by `LINKCELL_TPP`
and `LINKCELL_BLOCK`. Setup copies ride the workspace stream.
Orthorhombic cells, `k <= 16`. Meson feature `with_gpulite`.

## 0.2.4

A wrap consumer always links the static archive. The parent
`default_library` does not change this option, so a shared default
left `pydseams.yoda` needing `liblinkcell.so` at import time.

## 0.2.3

A static `libyodaLib.a` no longer passes `liblinkcell.so` to `ar`.
The Meson dependency attaches a generated header as the build-order
edge, not both cargo outputs.

## 0.2.2

The crate is `d-SEAMS/linkcell`. `HaoZeke/linkcell` redirects.

## 0.2.1

- `Error::MaskLen` when `mask` is `Some` and `mask.len() != n`.
- `Error::Overflow` when `n * k` does not fit a slice. `BufferSize`
  remains a caller `out` whose length is not `n * k`.
- `lc_last_error` is written only by `lc_knearest`. `lc_version` does
  not clear the slot. An interior NUL in a message no longer drops the
  string to `NULL`.
- C++ `Neighbours` owns the packed `n * k` buffer. `knearest_into`
  takes `out_len`. There is no 5-argument `knearest(..., int *out)`.
- CI checks `include/linkcell.h` against cbindgen 0.29.4.
- Branding: sheared linked cells, k=4 neighbours, periodic wrap
  (`assets/branding/`).

## 0.2.0

ABI break. Existing 0.1.x C and C++ callers must rebuild.

- `lc_knearest` takes `size_t n` and `size_t k`. The 0.1.x `int` counts
  overflowed on large frames.
- `lc_last_error` is thread-local. Concurrent searches no longer share
  one process-wide slot. `lc_version` stays process-static.
- C++ `linkcell::knearest` writes a packed `n * k` index buffer (unused
  slots `-1`) and returns a `Neighbours` view. It no longer returns
  `std::vector<std::vector<int>>`. Failure throws `linkcell::Error`.
- Rust `Error::Empty` is an empty point list only. A wrong-length
  `knearest_into` buffer is `Error::BufferSize`. An overflowing
  linked-cell mesh is `Error::TooManyCells`.
- The per-source k-set keeps one image of each neighbour (the
  nearest). The same particle visited through two wraps does not
  occupy two slots.
- `knearest_brute` on a sheared cell takes the 27-image minimum.
  The single parallelepiped wrap is not the Wigner-Seitz cell of a
  60-degree hex prism.

## 0.1.2

Fold once, then Cartesian pair distances plus a lattice shift
(the vesin / LAMMPS ghost trick). Sources run in parallel. The C
ABI writes indices into the caller buffer.

## 0.1.1

Orthorhombic boxes use the three-wrap minimum image, not two
3x3 matvecs. The C ABI reads packed xyz in place. The k-heap for
k <= 16 stays on the stack.

## 0.1.0

Periodic linked-cell k-nearest neighbour search. Rust crate, C ABI
(`lc_*`), C++ header. The cell is a general parallelepiped;
orthorhombic boxes are `Cell::ortho` / `lc_cell_ortho`.

Installable from Meson (`linkcell_dep`, `pkg.generate`), CMake
(`find_package(linkcell)`, `linkcell::linkcell`), and pkg-config
(`linkcell.pc`).
