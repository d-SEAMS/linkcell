#!/usr/bin/env bash
# Verify a Meson or CMake install of linkcell: CPU/GPU headers, libraries,
# pkg-config, and find_package(linkcell) / tiny consumer configure.
#
# Does not compile the consumer. Set LINKCELL_SKIP_BUILD=1 when the prefixes
# already exist and only the layout / configure checks should run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MESON_BUILD="${LINKCELL_MESON_BUILD:-$ROOT/build-meson}"
CMAKE_BUILD="${LINKCELL_CMAKE_BUILD:-$ROOT/build-cmake}"
MESON_DEST="${LINKCELL_MESON_DEST:-$ROOT/destdir-meson}"
CMAKE_PREFIX="${LINKCELL_CMAKE_PREFIX:-$ROOT/prefix-cmake}"

cargo_ver="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
if [[ -z "$cargo_ver" ]]; then
  echo "could not read version from Cargo.toml"
  exit 1
fi

have_lib() {
  local prefix="$1" stem="$2"
  local f
  for f in \
    "$prefix/lib/${stem}" \
    "$prefix/lib64/${stem}" \
    "$prefix/lib/x86_64-linux-gnu/${stem}" \
    "$prefix/bin/${stem}"; do
    if [[ -f "$f" || -L "$f" ]]; then
      return 0
    fi
  done
  return 1
}

find_pc_dir() {
  local prefix="$1" cand
  for cand in \
    "$prefix/lib/pkgconfig" \
    "$prefix/lib64/pkgconfig" \
    "$prefix/lib/x86_64-linux-gnu/pkgconfig" \
    "$prefix/share/pkgconfig"; do
    if [[ -f "$cand/linkcell.pc" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

check_pkgconfig() {
  local pcdir="$1" label="$2"
  export PKG_CONFIG_PATH="$pcdir${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  pkg-config --exists --print-errors linkcell
  pkg-config --exists --print-errors linkcell-gpu
  local ver cflags libs gpu_ver gpu_cflags gpu_libs
  ver="$(pkg-config --modversion linkcell)"
  cflags="$(pkg-config --cflags linkcell)"
  libs="$(pkg-config --libs linkcell)"
  gpu_ver="$(pkg-config --modversion linkcell-gpu)"
  gpu_cflags="$(pkg-config --cflags linkcell-gpu)"
  gpu_libs="$(pkg-config --libs linkcell-gpu)"
  echo "$label pkg-config: $ver"
  echo "$label cflags: $cflags"
  echo "$label libs: $libs"
  if [[ "$ver" != "$cargo_ver" ]]; then
    echo "$label pkg-config version $ver != Cargo.toml $cargo_ver"
    exit 1
  fi
  if [[ "$gpu_ver" != "$cargo_ver" ]]; then
    echo "$label GPU pkg-config version $gpu_ver != Cargo.toml $cargo_ver"
    exit 1
  fi
  case "$cflags" in
    *-I*) ;;
    *)
      echo "$label Cflags missing -I: $cflags"
      exit 1
      ;;
  esac
  case "$libs" in
    *-llinkcell*) ;;
    *)
      echo "$label Libs must contain -llinkcell: $libs"
      exit 1
      ;;
  esac
  case "$libs" in
    *-llinkcell_eh*) ;;
    *)
      echo "$label Libs must contain -llinkcell_eh: $libs"
      exit 1
      ;;
  esac
  case "$gpu_cflags" in
    *-DLINKCELL_HAS_GPULITE=1*) ;;
    *)
      echo "$label GPU Cflags missing LINKCELL_HAS_GPULITE: $gpu_cflags"
      exit 1
      ;;
  esac
  case "$gpu_libs" in
    *-llinkcell_gpu*) ;;
    *)
      echo "$label GPU Libs must contain -llinkcell_gpu: $gpu_libs"
      exit 1
      ;;
  esac
  case "$libs" in
    *-L*) ;;
    *)
      echo "$label Libs missing -L: $libs"
      exit 1
      ;;
  esac
  unset PKG_CONFIG_PATH
}

check_headers() {
  local prefix="$1"
  if [[ ! -f "$prefix/include/linkcell.h" ]]; then
    echo "missing $prefix/include/linkcell.h"
    exit 1
  fi
  if [[ ! -f "$prefix/include/linkcell.hpp" ]]; then
    echo "missing $prefix/include/linkcell.hpp"
    exit 1
  fi
  if [[ ! -f "$prefix/include/linkcell_gpu.h" \
     || ! -f "$prefix/include/linkcell_gpu.hpp" ]]; then
    echo "missing GPU headers under $prefix/include"
    exit 1
  fi
}

check_libs() {
  local prefix="$1"
  if ! have_lib "$prefix" "liblinkcell.so" \
    && ! have_lib "$prefix" "liblinkcell.dylib" \
    && ! have_lib "$prefix" "linkcell.dll"; then
    echo "missing installed cdylib under $prefix"
    find "$prefix" -name '*linkcell*' | head
    exit 1
  fi
  if ! have_lib "$prefix" "liblinkcell.a" \
    && ! have_lib "$prefix" "linkcell.lib"; then
    echo "missing installed staticlib under $prefix"
    find "$prefix" -name '*linkcell*' | head
    exit 1
  fi
  if ! have_lib "$prefix" "liblinkcell_gpu.a" \
    && ! have_lib "$prefix" "linkcell_gpu.lib"; then
    echo "missing installed GPU static library under $prefix"
    find "$prefix" -name '*linkcell*' | head
    exit 1
  fi
  if ! have_lib "$prefix" "liblinkcell_eh.so" \
    && ! have_lib "$prefix" "liblinkcell_eh.dylib" \
    && ! have_lib "$prefix" "linkcell_eh.dll"; then
    echo "missing installed linkcell_eh shared library under $prefix"
    find "$prefix" -name '*linkcell*' | head
    exit 1
  fi
}

if [[ "${LINKCELL_SKIP_BUILD:-0}" != "1" ]]; then
  echo "=== meson ==="
  meson setup "$MESON_BUILD" --wipe
  meson compile -C "$MESON_BUILD"
  meson test -C "$MESON_BUILD" --print-errorlogs
  rm -rf "$MESON_DEST"
  meson install -C "$MESON_BUILD" --destdir "$MESON_DEST"
fi

prefix_meson=""
for cand in \
  "$MESON_DEST/usr/local" \
  "$MESON_DEST/usr" \
  "$MESON_DEST"; do
  if [[ -f "$cand/include/linkcell.h" ]]; then
    prefix_meson="$cand"
    break
  fi
done
if [[ -z "$prefix_meson" ]]; then
  echo "meson install did not write include/linkcell.h under $MESON_DEST"
  find "$MESON_DEST" -type f | head
  exit 1
fi
check_headers "$prefix_meson"
check_libs "$prefix_meson"

pc_meson="$(find_pc_dir "$prefix_meson" || true)"
if [[ -z "$pc_meson" ]]; then
  echo "meson install did not write linkcell.pc"
  find "$MESON_DEST" -name 'linkcell.pc'
  exit 1
fi
check_pkgconfig "$pc_meson" "meson"

if [[ "${LINKCELL_SKIP_BUILD:-0}" != "1" ]]; then
  echo "=== cmake ==="
  cmake -S "$ROOT" -B "$CMAKE_BUILD" -DCMAKE_INSTALL_PREFIX="$CMAKE_PREFIX"
  cmake --build "$CMAKE_BUILD"
  ctest --test-dir "$CMAKE_BUILD" --output-on-failure
  rm -rf "$CMAKE_PREFIX"
  cmake --install "$CMAKE_BUILD"
fi

check_headers "$CMAKE_PREFIX"
check_libs "$CMAKE_PREFIX"
if [[ ! -f "$CMAKE_PREFIX/lib/cmake/linkcell/linkcellConfig.cmake" \
   && ! -f "$CMAKE_PREFIX/lib64/cmake/linkcell/linkcellConfig.cmake" ]]; then
  echo "cmake install did not write linkcellConfig.cmake"
  exit 1
fi

pc_cmake="$(find_pc_dir "$CMAKE_PREFIX" || true)"
if [[ -z "$pc_cmake" ]]; then
  echo "cmake install did not write linkcell.pc"
  exit 1
fi
check_pkgconfig "$pc_cmake" "cmake"

echo "=== cmake --find-package ==="
if ! CMAKE_PREFIX_PATH="$CMAKE_PREFIX" cmake --find-package \
    -DNAME=linkcell -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST; then
  echo "cmake --find-package MODE=EXIST did not find linkcell (CONFIG package; consumer configure is authoritative)"
fi

echo "=== cmake consumer configure ==="
cmake -S "$ROOT/tests/cmake-consumer" -B "$CMAKE_BUILD/find-consumer" \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" \
  -DLINKCELL_C_EXAMPLE="$ROOT/examples/two_points.c" \
  -DLINKCELL_CPP_EXAMPLE="$ROOT/examples/two_points.cpp"

echo "OK"
