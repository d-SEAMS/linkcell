# linkcell documentation

Periodic linked-cell k-nearest neighbour search. The Rust crate is the
implementation. The C ABI (`lc_*`) is the hourglass waist. C++ is a
typed header over that ABI. Python is DLPack (dlpk) over that waist.

## Tutorials

- [Two points](tutorials/two-points.md)

## How-to

- [Embed from C](how-to/embed-c.md)
- [Embed from C++](how-to/embed-cpp.md)
- [Embed from Python](how-to/embed-python.md)
- [pkg-config / Meson / CMake](how-to/pkg-config.md)

## Reference

- [C API](reference/c-api.md)
- [Algorithm](reference/algorithm.md)

## Explanation

- [Minimum image and linked cells](explanation/mic-and-cells.md)
- [SIMD paths and dispatch](explanation/simd-paths.md)
