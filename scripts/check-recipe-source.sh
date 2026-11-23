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

accept() {
  local name=$1
  local got
  got=$(read_recipe_source "$tmp/$name.yaml")
  if [[ "$got" != "$expected" ]]; then
    printf '%s: parser returned:\n%s\nexpected:\n%s\n' "$name" "$got" "$expected" >&2
    exit 1
  fi
}

# Valid rattler-build YAML forms that must still be accepted.
cat > "$tmp/blank-and-comment.yaml" << EOF
%YAML 1.2
--- # document start
context: # recipe variables
  version: '9.8.7'  # single-quoted with inline comment

source: # upstream tarball
  url: https://github.com/d-SEAMS/linkcell/archive/refs/tags/v\${{ version }}.tar.gz

# column-0 comment inside source
  sha256: "$(printf '%s' "$real" | tr 'a-f' 'A-F')"  # quoted uppercase hex
... # document end
EOF
accept blank-and-comment

cat > "$tmp/unquoted-and-indentless-seq.yaml" << EOF
context:
  version: 9.8.7
source:
  url:
  - https://github.com/d-SEAMS/linkcell/archive/refs/tags/v\${{ version }}.tar.gz
  - https://mirror.example.org/linkcell-v\${{ version }}.tar.gz
  patches:
  - fix-build.patch
  mirror:
  - url: https://other.example.org
    sha256: $decoy
  sha256: '$real'
EOF
accept unquoted-and-indentless-seq

printf 'context:\n  version: "9.8.7"\nsource:\n  url: u\n  sha256: %s' "$real" > "$tmp/no-eof-newline.yaml"
accept no-eof-newline

cat > "$tmp/flow-mapping.yaml" << EOF
context: { version: "9.8.7" }
source: { mirror: { sha256: $decoy }, url: u, sha256: $real }
EOF
accept flow-mapping

cat > "$tmp/multiline-quoted-decoys.yaml" << EOF
about:
  summary: "text
source:
  sha256: $decoy
"
context:
  version: "9.8.7"
source:
  note: "hello
  sha256: $decoy
  end: ok"

  url: https://github.com/d-SEAMS/linkcell/archive/refs/tags/v\${{ version }}.tar.gz
  sha256: $real
EOF
accept multiline-quoted-decoys

# Adversarial decoys that must be rejected rather than returning the decoy.
cat > "$tmp/blank-split-dup-sha.yaml" << EOF
context:
  version: "9.8.7"
source:
  url: u
  sha256: $decoy

  sha256: $real
EOF
reject blank-split-dup-sha 'recipe source repeats sha256'

cat > "$tmp/comment-split-dup-sha.yaml" << EOF
context:
  version: "9.8.7"
source:
  url: u
  sha256: $decoy
# real hash below
  sha256: $real
EOF
reject comment-split-dup-sha 'recipe source repeats sha256'

cat > "$tmp/blank-split-dup-ver.yaml" << EOF
context:
  version: "6.6.6"

  version: "9.8.7"
source:
  sha256: $real
EOF
reject blank-split-dup-ver 'recipe context repeats version'

cat > "$tmp/second-section-comment.yaml" << EOF
context:
  version: "9.8.7"
source:
  sha256: $decoy
source: # real
  sha256: $real
EOF
reject second-section-comment 'recipe has 2 source sections'

cat > "$tmp/second-section-flow.yaml" << EOF
context:
  version: "9.8.7"
source:
  sha256: $decoy
source: { url: u, sha256: $real }
EOF
reject second-section-flow 'recipe has 2 source sections'

cat > "$tmp/second-section-quoted.yaml" << EOF
context:
  version: "9.8.7"
source:
  sha256: $decoy
"source":
  sha256: $real
EOF
reject second-section-quoted 'recipe has 2 source sections'

cat > "$tmp/second-section-space-colon.yaml" << EOF
context:
  version: "9.8.7"
source:
  sha256: $decoy
source :
  sha256: $real
EOF
reject second-section-space-colon 'recipe has 2 source sections'

printf 'context:\n  version: "9.8.7"\nsource:\n  sha256: %s\n  sha256: %s' "$decoy" "$real" > "$tmp/no-eof-newline-dup.yaml"
reject no-eof-newline-dup 'recipe source repeats sha256'

cat > "$tmp/escaped-dup-key.yaml" << EOF
context:
  version: "9.8.7"
source:
  sha256: $decoy
  "sha\x32\x35\x36": $real
EOF
reject escaped-dup-key 'recipe source repeats sha256'

cat > "$tmp/col0-list.yaml" << EOF
context:
  version: "9.8.7"
source:
- url: a
  sha256: $decoy
- url: b
  sha256: $real
EOF
reject col0-list 'recipe source is a list, not one mapping'

cat > "$tmp/flow-list.yaml" << EOF
context:
  version: "9.8.7"
source: [{ url: a, sha256: $decoy }, { url: b, sha256: $real }]
EOF
reject flow-list 'recipe source is a list, not one mapping'

cat > "$tmp/flow-dup-key.yaml" << EOF
context:
  version: "9.8.7"
source: { sha256: $decoy, sha256: $real }
EOF
reject flow-dup-key 'recipe source repeats sha256'

cat > "$tmp/multi-doc.yaml" << EOF
---
context:
  version: "6.6.6"
source:
  sha256: $decoy
---
context:
  version: "9.8.7"
source:
  sha256: $real
EOF
reject multi-doc 'recipe has multiple YAML documents'

printf 'context:\n  version: "9.8.7"\nsource:' > "$tmp/eof-empty-section.yaml"
reject eof-empty-section 'recipe source section is empty'

cat > "$tmp/uneven-indent.yaml" << EOF
context:
  version: "9.8.7"
source:
    url: u
  sha256: $real
EOF
reject uneven-indent 'recipe source indentation is inconsistent'

cat > "$tmp/merge-key.yaml" << EOF
context:
  version: "9.8.7"
source:
  <<: *base
  sha256: $real
EOF
reject merge-key 'recipe source uses unsupported key syntax'

cat > "$tmp/plain-continuation.yaml" << EOF
context:
  version: "9.8.7"
source:
  sha256: $decoy
    extra
EOF
reject plain-continuation 'recipe source has no sha256'

echo "recipe source fields ok"
