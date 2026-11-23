#!/usr/bin/env bash
# The source route takes both the tag version and the sha256 from the
# pinned recipe. A fake recipe carries decoy values outside context and
# source; the parser must ignore them.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
source "$root/scripts/prove-published-install.sh"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/linkcell-recipe-source.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

real=$(python3 -c 'print("a" * 64)')
decoy=$(python3 -c 'print("1" * 64)')

cat > "$tmp/recipe.yaml" << EOF
package:
  name: decoy
  version: "1.2.3"
note:
  sha256: $decoy
context:
  version: "9.8.7"
source:
  url: https://github.com/d-SEAMS/linkcell/archive/refs/tags/v\${{ version }}.tar.gz
  sha256: $real
about:
  version: "0.0.1"
EOF

got=$(read_recipe_source "$tmp/recipe.yaml")
expected=$(printf '9.8.7\n%s' "$real")
if [[ "$got" != "$expected" ]]; then
  printf 'recipe source parser returned:\n%s\nexpected:\n%s\n' "$got" "$expected" >&2
  exit 1
fi

cat > "$tmp/nosource.yaml" << EOF
context:
  version: "9.8.7"
package:
  sha256: $decoy
EOF
if read_recipe_source "$tmp/nosource.yaml" >/dev/null 2>"$tmp/err"; then
  echo "parser accepted a recipe with no source sha256" >&2
  exit 1
fi
if ! grep -q 'recipe has no source section' "$tmp/err"; then
  echo "missing source section did not report that" >&2
  cat "$tmp/err" >&2
  exit 1
fi

# Nested keys inside context and source are not the recipe's own fields.
cat > "$tmp/nested.yaml" << EOF
context:
  extra:
    version: "6.6.6"
  version: "9.8.7"
source:
  mirror:
    sha256: $decoy
  # sha256: $decoy
  url: https://github.com/d-SEAMS/linkcell/archive/refs/tags/v\${{ version }}.tar.gz
  sha256: $real
EOF
got=$(read_recipe_source "$tmp/nested.yaml")
if [[ "$got" != "$expected" ]]; then
  printf 'nested decoys: parser returned:\n%s\nexpected:\n%s\n' "$got" "$expected" >&2
  exit 1
fi

# A form with more than one candidate sha256 has no single answer.
reject() {
  local name=$1 message=$2
  if read_recipe_source "$tmp/$name.yaml" >/dev/null 2>"$tmp/err"; then
    echo "parser accepted the $name recipe" >&2
    exit 1
  fi
  if ! grep -q -- "$message" "$tmp/err"; then
    echo "$name recipe did not report: $message" >&2
    cat "$tmp/err" >&2
    exit 1
  fi
}

cat > "$tmp/repeated-key.yaml" << EOF
context:
  version: "9.8.7"
source:
  url: u
  sha256: $decoy
  sha256: $real
EOF
reject repeated-key 'recipe source repeats sha256'

cat > "$tmp/repeated-section.yaml" << EOF
context:
  version: "9.8.7"
source:
  url: u
  sha256: $decoy
source:
  url: u
  sha256: $real
EOF
reject repeated-section 'recipe has 2 source sections'

cat > "$tmp/list.yaml" << EOF
context:
  version: "9.8.7"
source:
  - url: a
    sha256: $decoy
  - url: b
    sha256: $real
EOF
reject list 'recipe source is a list, not one mapping'

echo "recipe source fields ok"
