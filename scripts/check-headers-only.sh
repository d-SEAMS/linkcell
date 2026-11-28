#!/usr/bin/env bash
# Configure and build a host against the linkcell headers with no cargo
# and no rustc on PATH, then run it against a built shared library.
#
# The subproject is assembled the way packaging/linkcell-headers.wrap
# does it: the linkcell sources with packaging/linkcell-headers/meson.build
# laid over their meson.build.
#
# Usage: scripts/check-headers-only.sh <path to liblinkcell> [work dir]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBRARY="$(realpath "${1:?usage: check-headers-only.sh <liblinkcell> [work dir]}")"
WORK="${2:-$(mktemp -d)}"
mkdir -p "${WORK}"
WORK="$(realpath "${WORK}")"

HOST="${WORK}/host"
SUB="${HOST}/subprojects/linkcell-headers"
mkdir -p "${SUB}/include"
cp "${ROOT_DIR}/tests/headers-consumer/meson.build" "${HOST}/meson.build"
cp "${ROOT_DIR}/tests/row_grid.c" "${HOST}/row_grid.c"
cp "${ROOT_DIR}/include/linkcell.h" "${ROOT_DIR}/include/linkcell.hpp" "${SUB}/include/"
cp "${ROOT_DIR}/packaging/linkcell-headers/meson.build" "${SUB}/meson.build"

# A PATH with meson, ninja, and the C compiler under their plain names
# and nothing else beside the system directories. The tools are reached
# through one-line wrappers, so a directory that also holds a Rust
# toolchain does not put cargo on PATH.
BARE="${WORK}/bin"
mkdir -p "${BARE}"
wrap() {
    local name="$1" real="$2"
    [[ -x "${real}" ]] || { echo "${name} not found" >&2; exit 1; }
    printf '#!/bin/sh\nexec "%s" "$@"\n' "${real}" >"${BARE}/${name}"
    chmod +x "${BARE}/${name}"
}
wrap meson "$(command -v meson)"
wrap ninja "$(command -v ninja)"
wrap cc "$(command -v "${CC:-cc}")"

env -i PATH="${BARE}:/usr/bin:/bin" HOME="${WORK}" \
    LDFLAGS="${LDFLAGS:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" bash -c '
    set -euo pipefail
    if command -v cargo >/dev/null 2>&1 || command -v rustc >/dev/null 2>&1; then
        echo "a Rust toolchain is still on PATH: $(command -v cargo rustc)" >&2
        exit 1
    fi
    echo "cargo on PATH: none; rustc on PATH: none"
    meson setup "$1/build" "$1" >"$1/setup.log" 2>&1 || { cat "$1/setup.log" >&2; exit 1; }
    meson compile -C "$1/build" >"$1/compile.log" 2>&1 || { cat "$1/compile.log" >&2; exit 1; }
' _ "${HOST}"
"${HOST}/build/row_grid_host" "${LIBRARY}"
echo "headers-only host: ok"
