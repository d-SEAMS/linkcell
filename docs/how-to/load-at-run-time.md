# Load linkcell at run time

A host can compile against the linkcell headers alone and take the shared library when it runs.
It then needs a C compiler and no Rust toolchain, and the same host binary runs with any library that serves its header.

## Headers only

`linkcell-headers` is the include directory and nothing to link.
A build of linkcell provides it next to `linkcell`.
A host that builds no Rust takes it from a wrap that lays `packaging/linkcell-headers/meson.build` over the linkcell sources, so configuring the subproject never looks for cargo.

`subprojects/linkcell-headers.wrap`, with `packaging/linkcell-headers/meson.build` copied to `subprojects/packagefiles/linkcell-headers/meson.build`:

```
[wrap-git]
directory = linkcell-headers
url = https://github.com/d-SEAMS/linkcell.git
revision = v0.6.0
depth = 1
patch_directory = linkcell-headers

[provide]
dependency_names = linkcell-headers
```

```meson
project('host', 'c', default_options: ['c_std=c11'])
linkcell_headers = dependency('linkcell-headers')
cc = meson.get_compiler('c')
executable('host', 'host.c',
  dependencies: [linkcell_headers, cc.find_library('dl', required: false)])
```

An installed linkcell gives CMake the same thing as `linkcell::headers`.

`scripts/check-headers-only.sh <liblinkcell>` builds `tests/row_grid.c` this way with no cargo and no rustc on `PATH`, and runs it against the library.

## Check the ABI, then resolve

`linkcell.h` names the ABI it declares in `LC_ABI_MAJOR`, `LC_ABI_MINOR`, and `LC_ABI_LAYOUT_REVISION`.
The library reports its own through `lc_abi_stamp`, and `lc_abi_compatible` says whether it serves a host compiled against a given stamp.
Resolve those two symbols first.

```c
#include <dlfcn.h>
#include <stdio.h>
#include "linkcell.h"

void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
lc_abi (*stamp)(void) = (lc_abi (*)(void))dlsym(lib, "lc_abi_stamp");
int (*compatible)(const lc_abi *) =
    (int (*)(const lc_abi *))dlsym(lib, "lc_abi_compatible");
const lc_abi header = {LC_ABI_MAJOR, LC_ABI_MINOR, LC_ABI_LAYOUT_REVISION};
const lc_abi loaded = stamp();
printf("abi library=linkcell header=%u.%u.%u loaded=%u.%u.%u status=%s\n",
       header.major, header.minor, header.layout_revision,
       loaded.major, loaded.minor, loaded.layout_revision,
       compatible(&header) ? "verified" : "mismatch");
```

A library that lacks either symbol, or returns 0, is not used.

## Take the operations from the table

`lc_row_ops_get` is the third and last symbol such a host resolves by name.
It fills an `lc_row_ops` whose first member is the size the host compiled, with one function pointer per row-grid entry point.

```c
int (*ops_get)(lc_row_ops *) = (int (*)(lc_row_ops *))dlsym(lib, "lc_row_ops_get");
lc_row_ops ops = {0};
ops.size = sizeof ops;
if (ops_get(&ops) != 0 || ops.build == NULL || ops.rows == NULL) {
    /* this library cannot serve the host: report it and stop */
}
```

A null member is an operation the library does not have.
A host checks the members it needs when it is set up, and reports a missing one then, before any search.
`ops.build_request` is null in a library built without the `request` feature.

`ops.capabilities` fills an `lc_caps`, also size-prefixed, for the properties that are not an entry point: whether `lc_row.vector` is filled, whether a visitor can stop the walk, whether open axes are accepted.

`tests/row_grid.c` is a complete host of this kind.
It takes the library path from its first argument or from `LINKCELL_LIBRARY`.
