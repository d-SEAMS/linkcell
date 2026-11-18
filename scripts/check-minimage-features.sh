#!/usr/bin/env bash
# Default linkcell must not enable minimage/capi (mi_* collides with mimalloc).
# Optional minimage-capi forwards that feature when a caller needs it.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
locked=()
if [[ -f Cargo.lock ]]; then
  locked=(--locked)
fi
tree_default=$(cargo tree -e features -i minimage "${locked[@]}")
if printf '%s\n' "$tree_default" | grep -q 'minimage feature "capi"'; then
  echo "default linkcell build enables minimage/capi:" >&2
  printf '%s\n' "$tree_default" >&2
  exit 1
fi
tree_on=$(cargo tree -e features -i minimage --features minimage-capi "${locked[@]}")
if ! printf '%s\n' "$tree_on" | grep -q 'minimage feature "capi"'; then
  echo "minimage-capi did not enable minimage/capi:" >&2
  printf '%s\n' "$tree_on" >&2
  exit 1
fi
echo "minimage features ok (default: no capi; minimage-capi: capi on)"