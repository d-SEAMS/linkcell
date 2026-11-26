# Embed linkcell from C++

`include/linkcell.hpp` is a C++17 Resource Acquisition Is Initialization (RAII) header over `linkcell.h`.
The header wraps that one CPU library.
Link the same package target a C consumer links.

The CPU library header call `linkcell::knearest` returns an owning packed `Neighbours` for that package target.
`knearest_into` writes a caller-owned `int *` and takes `out_len`, which must be `n * k`.
Neither returns `std::vector<std::vector<int>>`.
Unused slots are `-1`.
Failure throws `linkcell::Error`.

## Meson wrap

Same wrap as C. Ask Meson for C++:

```meson
project('app', 'cpp', default_options: ['cpp_std=c++17'])
linkcell_dep = dependency('linkcell', fallback: ['linkcell', 'linkcell_dep'])
executable('app', 'app.cpp', dependencies: linkcell_dep)
```

## CMake

```cmake
cmake_minimum_required(VERSION 3.22)
project(app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(linkcell 0.6 REQUIRED)
add_executable(app app.cpp)
target_link_libraries(app PRIVATE linkcell::cpp)
```

`linkcell::cpp` links the C ABI and the shared library `linkcell_eh`, and requires C++17. `linkcell::linkcell` has no C++ language requirement and remains usable by C-only projects.
`linkcell_eh` owns `linkcell::Error`'s typeinfo. It stays shared when linkcell is a Meson subproject or when `default_library` is static, so a hidden-visibility DSO and its caller catch the exception by type. Meson `eh_shared=false` and CMake `LINKCELL_EH_STATIC=ON` link that object into each final image instead. That copy is unique only inside one final link; two DSOs that each link the static archive each get a typeinfo, and libc++ then misses the handler.
`tests/cmake-consumer/` exercises both patterns against an installed prefix, including a catch of `linkcell::Error` thrown from a second shared library.

## Call

```cpp
#include "linkcell.hpp"

const linkcell::Cell box = linkcell::Cell::ortho(10.0, 10.0, 10.0);
const double xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
const linkcell::Neighbours nn = linkcell::knearest(xyz, 2, box, 1);
int j = nn.neighbour(0, 0);
double r2 = nn.dist2(0, 0);
```

Device k-nearest (`linkcell_gpu.hpp`) takes device pointers and writes the same packed buffer.
Link `linkcell_gpu_dep` in Meson, `linkcell::gpu` in CMake, or `linkcell-gpu` with pkg-config.
A device-packed cell (3 / 9 / 12 doubles) is inverted on device:

```cpp
#include "linkcell_gpu.hpp"

linkcell::gpu::Workspace ws;
ws.knearest_into(xyz_dev, n, cell, k, out_dev, n * k, nullptr, 5.5);
ws.knearest_into_many_dcell(xyz_dev, n, 1, cell_dev, 9, k, out_dev,
                            n * k);
```
The device k-nearest workspace writes that same packed buffer and reports `available()` as false when the CUDA driver or NVIDIA Runtime Compilation (NVRTC) fails to load.
gpulite loads both at runtime.
The build skips the CUDA SDK.
The device workspace does not scan its device coordinates for non-finite values; check them before the call.
The host calls `linkcell::knearest`, `knearest_into`, `knearest_into_many`, and `knearest_csr` throw `linkcell::Error` (`a coordinate is not finite`) for a NaN or infinite coordinate or cell origin.

A general parallelepiped cell is `linkcell::Cell::from_vectors(a, b, c, origin)`.
`Cell::raw()` is the `lc_cell` the ABI consumes.

`mask` is an optional `const int *` of length `n` (nonzero keeps the point).
`cell_hint <= 0` selects the default bin edge.

An open cluster is `linkcell::pairs_open`.
The rows are the Cartesian half list.
`linkcell::pairs_within_pbc` takes a periodicity mask, and `linkcell::stencil_builds()` counts partner stencils that were built.

```cpp
const auto rows = linkcell::pairs_open(xyz, n, 2.5);
const auto builds = linkcell::stencil_builds();
```

Signatures: [C ABI](../reference/c-api.md).
First program: [two points](../tutorials/two-points.md).
