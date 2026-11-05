# Two points

Find the nearest neighbour of each of two atoms in a 10-unit cubic box.

## C

Save as `two_points.c`:

```c
#include "linkcell.h"
#include <stdio.h>

int main(void) {
  double xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  lc_cell box = lc_cell_ortho(10.0, 10.0, 10.0);
  int out[2];
  if (lc_knearest(xyz, 2, &box, 1, NULL, 0.0, out) != 0) {
    fprintf(stderr, "%s\n", lc_last_error());
    return 1;
  }
  printf("0 -> %d\n1 -> %d\n", out[0], out[1]);
  return 0;
}
```

`xyz` is packed `x y z` triples. `out` has length `n * k` (here 2).
`n` and `k` are `size_t`; the integer literals convert.
`out[i * k + j]` is the `j`-th neighbour of source `i`, nearest first.
Unused slots are `-1`.

With an installed prefix:

```
cc two_points.c $(pkg-config --cflags --libs linkcell) -o two_points
./two_points
```

Prints:

```
0 -> 1
1 -> 0
```

`examples/two_points.c` is a longer form (cubic and sheared). A
Meson or CMake build of the crate runs it as the `two_points` test.

## Periodic image

Replace the coordinates with a pair that wraps:

```c
  double xyz[] = {0.2, 0.0, 0.0, 9.4, 0.0, 0.0};
```

The printed pairing is still `0 -> 1` and `1 -> 0`. The image across
the a-face (distance 0.8) is nearer than the raw 9.2 vector.

## C++

Same packed `n * k` contract. `Neighbours` owns the buffer. Failure
throws `linkcell::Error`.

```cpp
#include "linkcell.hpp"
#include <iostream>

int main() {
  const linkcell::Cell box = linkcell::Cell::ortho(10.0, 10.0, 10.0);
  const double xyz[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const linkcell::Neighbours nn = linkcell::knearest(xyz, 2, box, 1);
  std::cout << "0 -> " << nn.neighbour(0, 0) << "\n";
  std::cout << "1 -> " << nn.neighbour(1, 0) << "\n";
}
```

```
c++ -std=c++17 two_points.cpp $(pkg-config --cflags --libs linkcell) -o two_points_cpp
```

Next: [embed from C](../how-to/embed-c.md) or
[embed from C++](../how-to/embed-cpp.md). Why the wrap is 0.8, not 9.2:
[MIC and cells](../explanation/mic-and-cells.md).
