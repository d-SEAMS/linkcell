#!/usr/bin/env bash
# Regenerate or check src/neighbor_capnp.rs from schema/neighbor.capnp.
#
# The generated file is shipped, so a build needs neither the capnp
# compiler nor capnpc-rust. Run this when schema/neighbor.capnp changes,
# then commit src/neighbor_capnp.rs.
#
# Usage:
#   scripts/regen-neighbor-capnp.sh           Regenerate the file in place.
#   scripts/regen-neighbor-capnp.sh --check   Diff the shipped file against
#                                             fresh output.
#
# Requires capnp and capnpc-rust on PATH. capnpc-rust comes from
# `cargo install capnpc`, at the capnp crate version in Cargo.toml.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${ROOT_DIR}/src/neighbor_capnp.rs"

for tool in capnp capnpc-rust; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "${tool} not found on PATH" >&2
        exit 1
    fi
done

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
(cd "${ROOT_DIR}/schema" && capnp compile -orust:"${tmp}" neighbor.capnp)

if [[ "${1:-}" == "--check" ]]; then
    if ! diff -u "${DST}" "${tmp}/neighbor_capnp.rs"; then
        echo "drift detected: src/neighbor_capnp.rs is out of sync with schema/neighbor.capnp" >&2
        echo "run: scripts/regen-neighbor-capnp.sh" >&2
        exit 1
    fi
    echo "src/neighbor_capnp.rs is in sync with schema/neighbor.capnp"
else
    cp "${tmp}/neighbor_capnp.rs" "${DST}"
    echo "Regenerated ${DST}"
fi
