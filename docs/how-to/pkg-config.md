# Use pkg-config

Meson and CMake both write `linkcell.pc` at install.
Device-enabled installs also write `linkcell-gpu.pc`.

## Prefix

After `meson install` or `cmake --install` into `$PREFIX`:

```
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig
pkg-config --exists --print-errors linkcell
pkg-config --modversion linkcell
pkg-config --cflags --libs linkcell
pkg-config --cflags --libs linkcell-gpu
```

On Debian multiarch the `.pc` may sit under `$PREFIX/lib/x86_64-linux-gnu/pkgconfig`.
Put that directory on `PKG_CONFIG_PATH` instead.

CPU `--cflags` contains `-I$PREFIX/include`; GPU cflags also define `LINKCELL_HAS_GPULITE`.
The GPU query adds `-llinkcell_gpu` and the CPU package.
`pkg-config --static --libs linkcell` includes the private Rust system libraries (`dl`, threads, and `m` where required).

## Compile

```
cc app.c $(pkg-config --cflags --libs linkcell) -o app
c++ -std=c++17 app.cpp $(pkg-config --cflags --libs linkcell) -o app
c++ -std=c++17 device.cpp $(pkg-config --cflags --libs linkcell-gpu) -o device
```

The GPU implementation is a static archive.
Keep the object before `--libs`; the `.pc` files carry the package and system link order.

## Check

`scripts/check-install.sh` in this repository checks CPU and GPU metadata from Meson and CMake prefixes.
