# Embed linkcell from C++

`include/linkcell.hpp` is a C++17 RAII header over `linkcell.h`. It
does not add a second CPU library. Link the same package target a C
consumer links.

`linkcell::knearest` returns an owning packed `Neighbours`.
`knearest_into` writes a caller-owned `int *` and takes `out_len`,
which must be `n * k`. Neither returns `std::vector<std::vector<int>>`.
Unused slots are `-1`. Failure throws `linkcell::Error`.

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
find_package(linkcell 0.3 REQUIRED)
add_executable(app app.cpp)
target_link_libraries(app PRIVATE linkcell::cpp)
```

`linkcell::cpp` links the C ABI and requires C++17. `linkcell::linkcell`
has no C++ language requirement and remains usable by C-only projects.
`tests/cmake-consumer/` exercises both patterns against an installed prefix.

## Call

```cpp
#include "linkcell.hpp"

const linkcell::Cell box = linkcell::Cell::ortho(10.0, 10.0, 10.0);
const double xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
const linkcell::Neighbours nn = linkcell::knearest(xyz, 2, box, 1);
int j = nn.neighbour(0, 0);
double r2 = nn.dist2(0, 0);
```

Device k-nearest (`linkcell_gpu.hpp`) takes device pointers and writes
the same packed buffer. Link `linkcell_gpu_dep` in Meson,
`linkcell::gpu` in CMake, or `linkcell-gpu` with pkg-config. A
device-packed cell (3 / 9 / 12 doubles) is inverted on device:

```cpp
#include "linkcell_gpu.hpp"

linkcell::gpu::Workspace ws;
ws.knearest_into(xyz_dev, n, cell, k, out_dev, n * k, nullptr, 5.5);
ws.knearest_into_many_dcell(xyz_dev, n, 1, cell_dev, 9, k, out_dev,
                            n * k);
```

`linkcell::gpu::available()` is false when a CUDA driver or NVRTC
cannot be loaded. gpulite loads both at runtime; no CUDA SDK is needed
to build the target.

A general parallelepiped is `linkcell::Cell::from_vectors(a, b, c,
origin)`. `Cell::raw()` is the `lc_cell` the ABI consumes.

`mask` is an optional `const int *` of length `n` (nonzero keeps the
point). `cell_hint <= 0` selects the default bin edge.

Signatures: [C ABI](../reference/c-api.md). First program:
[two points](../tutorials/two-points.md).
