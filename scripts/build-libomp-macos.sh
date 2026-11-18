#!/usr/bin/env bash
# Build LLVM's OpenMP runtime into PREFIX for the macOS wheels.
#
# The wheels bundle libomp with delocate, and delocate refuses a library
# whose minimum macOS is newer than the wheel's MACOSX_DEPLOYMENT_TARGET.
# Homebrew's libomp targets the runner's own macOS release, so the wheels
# build their own with the wheel's target and name it with LIBOMP_PREFIX,
# which build.rs reads before it asks Homebrew.
#
# Usage: MACOSX_DEPLOYMENT_TARGET=11.0 scripts/build-libomp-macos.sh PREFIX
#
# Needs cmake, curl and a C++ compiler. LIBOMP_LLVM_VERSION picks another
# LLVM release; only the pinned releases below have checksums.
set -euo pipefail

prefix=${1:?usage: build-libomp-macos.sh PREFIX}
version=${LIBOMP_LLVM_VERSION:-20.1.8}
target=${MACOSX_DEPLOYMENT_TARGET:-11.0}
arch=${LIBOMP_ARCH:-$(uname -m)}
case "$arch" in
  arm64 | aarch64) arch=arm64 ;;
  x86_64) ;;
  *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
esac

case "$version" in
  20.1.8)
    openmp_sha256=b21c04ee9cbe56e200c5d83823765a443ee6389bbc3f64154c96e94016e6cee9
    cmake_sha256=3319203cfd1172bbac50f06fa68e318af84dcb5d65353310c0586354069d6634
    ;;
  *)
    echo "no checksums for LLVM $version" >&2
    exit 1
    ;;
esac

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

base=https://github.com/llvm/llvm-project/releases/download/llvmorg-$version
for part in openmp cmake; do
  case "$part" in
    openmp) want=$openmp_sha256 ;;
    cmake) want=$cmake_sha256 ;;
  esac
  curl -fsSL -o "$work/$part.tar.xz" "$base/$part-$version.src.tar.xz"
  got=$(sha256_of "$work/$part.tar.xz")
  if [ "$got" != "$want" ]; then
    echo "$part-$version.src.tar.xz: sha256 $got, expected $want" >&2
    exit 1
  fi
  mkdir -p "$work/src/$part"
  tar -xf "$work/$part.tar.xz" -C "$work/src/$part" --strip-components=1
done

mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

cmake -S "$work/src/openmp" -B "$work/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$target" \
  -DCMAKE_OSX_ARCHITECTURES="$arch" \
  -DCMAKE_INSTALL_NAME_DIR="$prefix/lib" \
  -DLIBOMP_ENABLE_SHARED=ON \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DLIBOMP_OMPT_SUPPORT=OFF \
  -DLIBOMP_OMPD_SUPPORT=OFF \
  -DLIBOMP_USE_HWLOC=OFF
cmake --build "$work/build" --parallel
cmake --install "$work/build"

if [ "$(uname -s)" = Darwin ]; then
  lib="$prefix/lib/libomp.dylib"
  # delocate follows the install name, so it must be the file itself.
  install_name_tool -id "$lib" "$lib"
  minos=$(otool -l "$lib" | awk '/LC_BUILD_VERSION/ { found = 1 } found && $1 == "minos" { print $2; exit }')
  echo "libomp.dylib: install name $lib, minos $minos, target $target"
  newest=$(printf '%s\n%s\n' "$minos" "$target" | sort -t. -k1,1n -k2,2n | tail -n 1)
  if [ -z "$minos" ] || [ "$newest" != "$target" ]; then
    echo "libomp.dylib needs macOS $minos, newer than the target $target" >&2
    exit 1
  fi
fi

test -f "$prefix/include/omp.h"
ls -l "$prefix/lib"
