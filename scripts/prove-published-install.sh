#!/usr/bin/env bash
# Install each published linkcell route in a clean prefix and run the
# README quickstart plus an independent periodic k-nearest check.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECKS=${CHECKS:-$SCRIPT_DIR/prove-published}
# main fills these in. They stay empty when the file is sourced so the
# recipe parser can be checked without creating a work directory.
ROOT=${PROVE_ROOT:-}
LOG_DIR=${PROVE_LOG_DIR:-}
RECIPE_COMMIT=922ae1481b3d983d7814c18bc03085ba67a91704
RECIPE_URL=https://raw.githubusercontent.com/HaoZeke/staged-recipes/${RECIPE_COMMIT}/recipes/linkcell

log() { printf '%s\n' "$*"; }

crate_version() {
  curl -fsSL -A 'linkcell-install-check (https://github.com/d-SEAMS/linkcell)' \
    https://crates.io/api/v1/crates/linkcell \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["crate"]["max_version"])'
}

pypi_version() {
  curl -fsSL https://pypi.org/pypi/linkcell/json \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])'
}

require_cmd() {
  local c
  for c in "$@"; do
    command -v "$c" >/dev/null || { log "missing command: $c"; exit 1; }
  done
}

run_c_consumers() {
  local prefix=$1
  local version=$2
  local pc libdir work
  pc=$(find "$prefix" -name linkcell.pc -printf '%h\n' | head -1)
  libdir=$(find "$prefix" -name 'liblinkcell.so' -printf '%h\n' | head -1)
  if [[ -z "$pc" || -z "$libdir" ]]; then
    log "installed prefix $prefix has no linkcell.pc or liblinkcell.so"
    find "$prefix" -maxdepth 4 -type f | head -200
    return 1
  fi
  work=$(mktemp -d "$ROOT/consumer.XXXXXX")
  cp "$CHECKS/two_points.c" "$CHECKS/two_points.cpp" \
    "$CHECKS/periodic.c" "$CHECKS/periodic.cpp" "$CHECKS/periodic_ref.h" \
    "$work/"
  local cc_bin=${CC:-cc}
  local cxx_bin=${CXX:-c++}
  (
    cd "$work"
    export PKG_CONFIG_PATH="$pc"
    export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "prefix=$prefix"
    log "CC=$cc_bin"
    log "CXX=$cxx_bin"
    log "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
    log "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    "$cc_bin" --version | head -1
    pkg-config --exists --print-errors linkcell
    pkg-config --modversion linkcell
    test "$(pkg-config --modversion linkcell)" = "$version"
    pkg-config --cflags --libs linkcell
    # shellcheck disable=SC2046
    "$cc_bin" two_points.c $(pkg-config --cflags --libs linkcell) -o two_points
    ./two_points > two_points.out
    printf '0 -> 1\n1 -> 0\n' > expected.out
    diff -u expected.out two_points.out
    # shellcheck disable=SC2046
    "$cxx_bin" -std=c++17 two_points.cpp $(pkg-config --cflags --libs linkcell) -o two_points_cpp
    ./two_points_cpp > two_points_cpp.out
    diff -u expected.out two_points_cpp.out
    # shellcheck disable=SC2046
    "$cc_bin" periodic.c $(pkg-config --cflags --libs linkcell) -lm -o periodic
    ./periodic "$version"
    # shellcheck disable=SC2046
    "$cxx_bin" -std=c++17 periodic.cpp $(pkg-config --cflags --libs linkcell) -o periodic_cpp
    ./periodic_cpp "$version"
    log "ldd liblinkcell.so"
    ldd "$libdir/liblinkcell.so"
    log "ldd two_points"
    ldd two_points
  )
}

route_crates() {
  local ver home proj
  require_cmd cargo rustc curl python3
  ver=$(crate_version)
  log "crates.io max version $ver"
  log "rustc $(rustc --version)"
  home=$ROOT/crates/cargo-home
  proj=$ROOT/crates/quickstart
  rm -rf "$ROOT/crates"
  mkdir -p "$home"
  export CARGO_HOME=$home
  export CARGO_TARGET_DIR=$ROOT/crates/target
  export EXPECT=$ver
  cargo new --bin "$proj"
  (
    cd "$proj"
    cargo add linkcell
    cp "$CHECKS/quickstart.rs" src/main.rs
    python3 - << 'PY'
import json, os, subprocess, sys
meta = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--locked"]))
hits = [p for p in meta["packages"] if p["name"] == "linkcell"]
if len(hits) != 1:
    sys.exit(f"expected one linkcell package, found {len(hits)}")
src = hits[0].get("source") or ""
print("resolved", hits[0]["version"], src)
if not src.startswith("registry+"):
    sys.exit(f"linkcell did not come from a registry: {src}")
if os.environ["EXPECT"] != hits[0]["version"]:
    sys.exit(f"resolved {hits[0]['version']} != crates.io max {os.environ['EXPECT']}")
PY
    cargo run --locked
  )
}

route_pypi() {
  local ver venv
  require_cmd python3 curl
  ver=$(pypi_version)
  log "PyPI version $ver"
  venv=$ROOT/pypi/venv
  rm -rf "$ROOT/pypi"
  mkdir -p "$ROOT/pypi"
  python3 -m venv "$venv"
  "$venv/bin/python" -m pip install --upgrade pip
  "$venv/bin/python" -m pip install --report "$ROOT/pypi/report.json" linkcell numpy
  export EXPECT=$ver
  export REPORT=$ROOT/pypi/report.json
  "$venv/bin/python" - << 'PY'
import json, os, sys
report = json.load(open(os.environ["REPORT"]))
found = False
for item in report["install"]:
    name = item["metadata"]["name"].lower()
    if name != "linkcell":
        continue
    found = True
    url = item["download_info"]["url"]
    version = item["metadata"]["version"]
    print("downloaded", version, url)
    if version != os.environ["EXPECT"]:
        sys.exit(f"installed {version} != PyPI latest {os.environ['EXPECT']}")
    if not url.endswith(".whl"):
        sys.exit(f"linkcell was not a wheel: {url}")
    if "abi3" not in url:
        sys.exit(f"wheel is not the limited-ABI artifact: {url}")
if not found:
    sys.exit("pip report has no linkcell")
PY
  (cd "$ROOT/pypi" && "$venv/bin/python" -I "$CHECKS/periodic.py" "$ver")
}

# Print the recipe's tag version and source sha256, one per line.
# context.version is what the recipe's ${{ version }} expands to, including
# the tag in source.url. source.sha256 is that tarball. Only direct keys of
# the single context and source mappings count; a nested, repeated or list
# form is rejected rather than guessed at.
read_recipe_source() {
  local recipe=$1
  python3 - "$recipe" << 'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    text = path.read_text(encoding="utf-8")
except OSError as exc:
    raise SystemExit(f"cannot read recipe: {exc}") from exc

if text.startswith("\ufeff"):
    text = text[1:]
raw_lines = text.splitlines()

ESCAPES = {
    "0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t",
    "n": "\n", "v": "\v", "f": "\f", "r": "\r", "e": "\x1b",
    " ": " ", '"': '"', "/": "/", "\\": "\\",
    "N": "\u0085", "_": "\u00a0", "L": "\u2028", "P": "\u2029",
}
HEX_ESCAPES = {"x": 2, "u": 4, "U": 8}
BLOCK_HEADER = re.compile(r"^[|>](?:[+-][1-9]?|[1-9][+-]?)?$")
NODE_PROP = re.compile(r"^(?:&[^\s,\[\]{}]+|!(?:<[^>]*>|[^\s,\[\]{}]*))(?=$|[ \t])")
DOC_MARKER = re.compile(r"^(?:---|\.\.\.)(?:$|[ \t])")
STRICT_KEY = re.compile(r"^([A-Za-z0-9_-]+)[ \t]*:(?:[ \t]+(.*)|[ \t]*)$")
LOOSE_KEY = re.compile(r"""^([^#"'{\[\]}:,][^:"']*?)[ \t]*:(?:[ \t]+(.*)|[ \t]*)$""")
MERGE_KEY = re.compile(r"^<<[ \t]*:(?:[ \t]+(.*)|[ \t]*)$")


def strip_plain_comment(s):
    for i, ch in enumerate(s):
        if ch == "#" and (i == 0 or s[i - 1] in (" ", "\t")):
            return s[:i].rstrip(" \t")
    return s.rstrip(" \t")


def strip_props(s):
    s = s.lstrip(" \t")
    while s:
        m = NODE_PROP.match(s)
        if not m:
            break
        s = s[m.end():].lstrip(" \t")
    return s


def scan_quoted_inline(s, start):
    quote = s[start]
    out = []
    i = start + 1
    n = len(s)
    while i < n:
        ch = s[i]
        if quote == '"':
            if ch == "\\":
                if i + 1 >= n:
                    return False, n, ""
                esc = s[i + 1]
                if esc in ESCAPES:
                    out.append(ESCAPES[esc])
                    i += 2
                elif esc in HEX_ESCAPES:
                    width = HEX_ESCAPES[esc]
                    hex_part = s[i + 2:i + 2 + width]
                    if len(hex_part) != width or not re.fullmatch(r"[0-9a-fA-F]+", hex_part):
                        raise SystemExit(f"recipe has invalid escape sequence: \\{esc}{hex_part}")
                    out.append(chr(int(hex_part, 16)))
                    i += 2 + width
                else:
                    raise SystemExit(f"recipe has unsupported escape sequence: \\{esc}")
                continue
            if ch == '"':
                return True, i + 1, "".join(out)
        else:
            if ch == "'":
                if i + 1 < n and s[i + 1] == "'":
                    out.append("'")
                    i += 2
                    continue
                return True, i + 1, "".join(out)
        out.append(ch)
        i += 1
    return False, n, ""


def advance_quoted(line, pos, quote):
    n = len(line)
    i = pos
    while i < n:
        ch = line[i]
        if quote == '"':
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                return True, i + 1
        else:
            if ch == "'":
                if i + 1 < n and line[i + 1] == "'":
                    i += 2
                    continue
                return True, i + 1
        i += 1
    return False, n


def check_trailing_comment(tail, where):
    t = tail.lstrip(" \t")
    if t and not t.startswith("#"):
        raise SystemExit(f"recipe {where} has unexpected trailing tokens: {t}")


def scan_flow_line(line, pos, stack, quote, where):
    n = len(line)
    i = pos
    while i < n:
        if quote is not None:
            closed, i = advance_quoted(line, i, quote)
            if not closed:
                return stack, quote, n
            quote = None
            continue
        ch = line[i]
        if ch == "#" and (i == 0 or line[i - 1] in (" ", "\t")):
            return stack, None, n
        if ch in ('"', "'") and (i == 0 or line[i - 1] in (" ", "\t", "{", "[", ",", ":")):
            quote = ch
            i += 1
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if not stack or stack[-1] != ch:
                raise SystemExit(f"recipe {where} has mismatched flow delimiter: {ch}")
            stack.pop()
            if not stack:
                return stack, None, i + 1
        i += 1
    return stack, quote, n


def parse_key(s, strict_key):
    if s[0] in ('"', "'"):
        closed, end, key_name = scan_quoted_inline(s, 0)
        if not closed:
            return "multiline_quoted_key", s[0], ""
        after = s[end:].lstrip(" \t")
        if after.startswith(":") and (len(after) == 1 or after[1] in (" ", "\t")):
            return "key", key_name, after[1:].lstrip(" \t")
        return "not_a_key", s, ""
    if s == "?" or s.startswith(("? ", "?\t")):
        return "explicit_key", "?", s[1:].lstrip(" \t")
    m = MERGE_KEY.match(s)
    if m:
        return "merge_key", "<<", (m.group(1) or "").lstrip(" \t")
    pat = STRICT_KEY if strict_key else LOOSE_KEY
    m = pat.match(s)
    if m:
        return "key", m.group(1), (m.group(2) or "").lstrip(" \t")
    return "not_a_key", s, ""


def consume_value(lines, val_rest, line_idx, parent_col, where):
    val = strip_props(val_rest)
    if not val or val.startswith("#"):
        return "empty", "", line_idx + 1
    plain_head = strip_plain_comment(val)
    if BLOCK_HEADER.fullmatch(plain_head):
        j = line_idx + 1
        while j < len(lines):
            ln = lines[j]
            if not ln.strip():
                j += 1
                continue
            lead = len(ln) - len(ln.lstrip(" \t"))
            if lead > parent_col:
                j += 1
                continue
            break
        return "block", "", j
    if val[0] in ('"', "'"):
        closed, end, decoded = scan_quoted_inline(val, 0)
        if closed:
            check_trailing_comment(val[end:], where)
            return "quoted", decoded, line_idx + 1
        quote = val[0]
        j = line_idx + 1
        while j < len(lines):
            closed, end = advance_quoted(lines[j], 0, quote)
            if closed:
                check_trailing_comment(lines[j][end:], where)
                return "multiline_quoted", "", j + 1
            j += 1
        raise SystemExit(f"recipe {where} has an unterminated quoted string")
    if val[0] in ("{", "["):
        stack, quote, end = scan_flow_line(val, 0, [], None, where)
        if not stack and quote is None:
            check_trailing_comment(val[end:], where)
            return "flow", val[:end], line_idx + 1
        chunks = [val]
        j = line_idx + 1
        while j < len(lines):
            stack, quote, end = scan_flow_line(lines[j], 0, stack, quote, where)
            if not stack and quote is None:
                chunks.append(lines[j][:end])
                check_trailing_comment(lines[j][end:], where)
                return "flow", "\n".join(chunks), j + 1
            chunks.append(lines[j])
            j += 1
        raise SystemExit(f"recipe {where} has an unterminated flow collection")
    return "plain", plain_head, line_idx + 1


def advance_node(lines, idx, col, where):
    s = lines[idx][col:]
    while s == "-" or s.startswith(("- ", "-\t")):
        s = s[1:]
        step = len(s) - len(s.lstrip(" \t"))
        col += 1 + step
        s = s[step:]
    if not s or s.startswith("#"):
        return idx + 1
    kind, _, rest = parse_key(s, strict_key=False)
    if kind == "key":
        _, _, next_idx = consume_value(lines, rest, idx, col, where)
        return next_idx
    _, _, next_idx = consume_value(lines, s, idx, col, where)
    return next_idx


def parse_flow_mapping(flow_text, name):
    if flow_text[0] == "[":
        raise SystemExit(f"recipe {name} is a list, not one mapping")
    if flow_text[0] != "{" or flow_text[-1] != "}":
        raise SystemExit(f"recipe {name} is not a mapping")
    inner = flow_text[1:-1]
    n = len(inner)
    i = 0

    def skip_ws_comments(pos):
        while pos < n:
            if inner[pos] in (" ", "\t", "\n"):
                pos += 1
            elif inner[pos] == "#":
                nl = inner.find("\n", pos)
                pos = n if nl == -1 else nl + 1
            else:
                break
        return pos

    keys = {}
    while True:
        i = skip_ws_comments(i)
        if i >= n:
            break
        if inner[i] in ('"', "'"):
            closed, i, key = scan_quoted_inline(inner, i)
            if not closed:
                raise SystemExit(f"recipe {name} has an unterminated quoted key")
        else:
            start = i
            while i < n and inner[i] not in (":", ",", "{", "}", "[", "]", "\n", "#"):
                i += 1
            key = inner[start:i].rstrip(" \t")
            if key == "<<" or key.startswith("?"):
                raise SystemExit(f"recipe {name} uses unsupported key syntax: {key}")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
                raise SystemExit(f"recipe {name} line is not a key: {key}")
        i = skip_ws_comments(i)
        if i >= n or inner[i] != ":":
            raise SystemExit(f"recipe {name} mapping is missing ':' after {key}")
        i += 1
        i = skip_ws_comments(i)
        while i < n:
            m = NODE_PROP.match(inner[i:])
            if not m:
                break
            i = skip_ws_comments(i + m.end())
        if i >= n or inner[i] == ",":
            entry = ("empty", "")
        elif inner[i] in ("{", "["):
            stack = []
            quote = None
            while i < n:
                ch = inner[i]
                if quote is not None:
                    if quote == '"':
                        if ch == "\\":
                            i += 2
                            continue
                        if ch == '"':
                            quote = None
                    else:
                        if ch == "'":
                            if i + 1 < n and inner[i + 1] == "'":
                                i += 2
                                continue
                            quote = None
                    i += 1
                    continue
                if ch == "#" and (i == 0 or inner[i - 1] in (" ", "\t", "\n")):
                    nl = inner.find("\n", i)
                    i = n if nl == -1 else nl + 1
                    continue
                if ch in ('"', "'") and (i == 0 or inner[i - 1] in (" ", "\t", "\n", "{", "[", ",", ":")):
                    quote = ch
                    i += 1
                    continue
                if ch in ("{", "["):
                    stack.append("}" if ch == "{" else "]")
                elif ch in ("}", "]"):
                    if not stack or stack[-1] != ch:
                        raise SystemExit(f"recipe {name} has mismatched flow delimiter: {ch}")
                    stack.pop()
                    if not stack:
                        i += 1
                        break
                i += 1
            if stack or quote is not None:
                raise SystemExit(f"recipe {name} has an unterminated flow collection")
            entry = ("flow", "")
        elif inner[i] in ('"', "'"):
            closed, i, val = scan_quoted_inline(inner, i)
            if not closed:
                raise SystemExit(f"recipe {name} has an unterminated quoted string")
            entry = ("quoted", val)
        else:
            start = i
            while i < n and inner[i] != ",":
                if inner[i] == "#" and (i == start or inner[i - 1] in (" ", "\t", "\n")):
                    break
                i += 1
            val = inner[start:i].strip()
            entry = ("plain", val)
        if key in keys:
            raise SystemExit(f"recipe {name} repeats {key}")
        keys[key] = entry
        i = skip_ws_comments(i)
        if i < n:
            if inner[i] != ",":
                raise SystemExit(f"recipe {name} flow mapping expected ','")
            i += 1
    if not keys:
        raise SystemExit(f"recipe {name} section is empty")
    return keys


def parse_block_section(section_lines, name):
    i = 0
    while i < len(section_lines):
        ln = section_lines[i]
        if ln.strip() and not ln.lstrip(" \t").startswith("#"):
            break
        i += 1
    if i >= len(section_lines):
        raise SystemExit(f"recipe {name} section is empty")
    first = section_lines[i]
    ws = first[:len(first) - len(first.lstrip(" \t"))]
    if "\t" in ws:
        raise SystemExit(f"recipe {name} indentation is inconsistent")
    rest0 = first[len(ws):]
    if rest0 == "-" or rest0.startswith(("- ", "-\t")):
        raise SystemExit(f"recipe {name} is a list, not one mapping")
    if not ws:
        raise SystemExit(f"recipe {name} indentation is inconsistent")
    indent = ws
    indent_len = len(indent)

    keys = {}
    seq_state = "none"
    last_plain_key = None
    while i < len(section_lines):
        ln = section_lines[i]
        if not ln.strip() or ln.lstrip(" \t").startswith("#"):
            i += 1
            continue
        if not ln.startswith(indent):
            raise SystemExit(f"recipe {name} indentation is inconsistent")
        rest = ln[indent_len:]
        if rest[0] in (" ", "\t"):
            if seq_state == "maybe":
                seq_state = "none"
            if last_plain_key is not None:
                keys[last_plain_key] = ("multiline_plain", "")
                last_plain_key = None
            lead = len(ln) - len(ln.lstrip(" \t"))
            i = advance_node(section_lines, i, lead, name)
            continue
        if rest == "-" or rest.startswith(("- ", "-\t")):
            if seq_state == "none":
                raise SystemExit(f"recipe {name} is a list, not one mapping")
            seq_state = "in_seq"
            i = advance_node(section_lines, i, indent_len, name)
            continue
        last_plain_key = None
        seq_state = "none"
        kind, key_name, val_rest = parse_key(rest, strict_key=True)
        if kind in ("merge_key", "explicit_key"):
            raise SystemExit(f"recipe {name} uses unsupported key syntax: {rest}")
        if kind != "key":
            raise SystemExit(f"recipe {name} line is not a key: {rest}")
        if key_name in keys:
            raise SystemExit(f"recipe {name} repeats {key_name}")
        vkind, vval, next_i = consume_value(section_lines, val_rest, i, indent_len, name)
        keys[key_name] = (vkind, vval)
        if vkind == "empty":
            seq_state = "maybe"
        elif vkind == "plain":
            last_plain_key = key_name
        i = next_i
    return keys


sections = {}
doc_started = False
doc_ended = False
current_top = None
idx = 0
while idx < len(raw_lines):
    line = raw_lines[idx]
    if not line.strip() or line.lstrip(" \t").startswith("#"):
        idx += 1
        continue
    if line.startswith("%") and not doc_started and not doc_ended:
        idx += 1
        continue
    if DOC_MARKER.match(line):
        check_trailing_comment(line[3:], "document marker")
        if line.startswith("---"):
            if doc_started or doc_ended:
                raise SystemExit("recipe has multiple YAML documents")
            doc_started = True
        else:
            if current_top is not None:
                top_name, start_idx = current_top
                sections.setdefault(top_name, []).append(("block", raw_lines[start_idx:idx]))
                current_top = None
            doc_ended = True
        idx += 1
        continue
    if doc_ended:
        raise SystemExit("recipe has multiple YAML documents")
    lead = len(line) - len(line.lstrip(" \t"))
    if lead == 0:
        if line == "-" or line.startswith(("- ", "-\t")):
            if current_top is None:
                raise SystemExit("recipe root is a list, not a mapping")
            idx = advance_node(raw_lines, idx, 0, current_top[0])
            continue
        if current_top is not None:
            top_name, start_idx = current_top
            sections.setdefault(top_name, []).append(("block", raw_lines[start_idx:idx]))
            current_top = None
        doc_started = True
        kind, key_name, val_rest = parse_key(line, strict_key=False)
        if kind in ("merge_key", "explicit_key"):
            raise SystemExit(f"recipe uses unsupported top-level key syntax: {line}")
        if kind == "multiline_quoted_key":
            raise SystemExit("recipe has an unsupported multiline top-level key")
        if kind != "key":
            raise SystemExit(f"recipe top-level line is not a key: {line}")
        vkind, vval, next_idx = consume_value(raw_lines, val_rest, idx, 0, key_name)
        if vkind == "empty":
            current_top = (key_name, next_idx)
        else:
            sections.setdefault(key_name, []).append((vkind, vval))
        idx = next_idx
        continue
    if not doc_started:
        raise SystemExit(f"recipe has unexpected indented line before any section: {line}")
    idx = advance_node(raw_lines, idx, lead, current_top[0] if current_top is not None else "top-level")

if current_top is not None:
    top_name, start_idx = current_top
    sections.setdefault(top_name, []).append(("block", raw_lines[start_idx:]))


def section(name):
    entries = sections.get(name, [])
    if not entries:
        raise SystemExit(f"recipe has no {name} section")
    if len(entries) > 1:
        raise SystemExit(f"recipe has {len(entries)} {name} sections")
    kind, payload = entries[0]
    if kind == "block":
        return parse_block_section(payload, name)
    if kind == "flow":
        return parse_flow_mapping(payload, name)
    raise SystemExit(f"recipe {name} is not a mapping")


context = section("context")
source = section("source")
v_entry = context.get("version")
if not v_entry or v_entry[0] not in ("quoted", "plain") or not v_entry[1]:
    raise SystemExit("recipe context has no version")
version = v_entry[1]
if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
    raise SystemExit(f"recipe context version is not X.Y.Z: {version}")
s_entry = source.get("sha256")
if not s_entry or s_entry[0] not in ("quoted", "plain") or not s_entry[1]:
    raise SystemExit("recipe source has no sha256")
sha = re.fullmatch(r"[0-9a-fA-F]{64}", s_entry[1])
if not sha:
    raise SystemExit("recipe source has no sha256")
print(version)
print(sha.group(0).lower())
PY
}

prepare_source() {
  local ver sha tar fields
  mkdir -p "$ROOT/source/recipe"
  if [[ ! -f "$ROOT/source/recipe/recipe.yaml" ]]; then
    curl -fsSL -o "$ROOT/source/recipe/recipe.yaml" "$RECIPE_URL/recipe.yaml"
  fi
  fields=$(read_recipe_source "$ROOT/source/recipe/recipe.yaml")
  ver=${fields%%$'\n'*}
  sha=${fields#*$'\n'}
  [[ "$ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]]
  log "recipe version $ver"
  log "recipe sha256 $sha"
  tar=$ROOT/source/linkcell-$ver.tar.gz
  curl -fsSL -L -o "$tar" "https://github.com/d-SEAMS/linkcell/archive/refs/tags/v${ver}.tar.gz"
  echo "$sha  $tar" | sha256sum -c -
  tar -C "$ROOT/source" -xzf "$tar"
  test -f "$ROOT/source/linkcell-$ver/Cargo.lock"
  test -f "$ROOT/source/linkcell-$ver/meson.build"
  test -f "$ROOT/source/linkcell-$ver/CMakeLists.txt"
  SOURCE_VERSION=$ver
}

route_source_meson() {
  local ver=$1
  local src=$ROOT/source/linkcell-$ver
  local prefix=$ROOT/source/meson-prefix
  log "meson $(meson --version) CC=${CC:-cc} $(${CC:-cc} --version | head -1)"
  rm -rf "$ROOT/source/meson-tree" "$prefix"
  cp -a "$src" "$ROOT/source/meson-tree"
  export CARGO_HOME=$ROOT/source/meson-cargo
  export CARGO_TARGET_DIR=$ROOT/source/meson-target
  rm -rf "$CARGO_HOME" "$CARGO_TARGET_DIR"
  meson setup "$ROOT/source/meson-tree/build" "$ROOT/source/meson-tree" --prefix "$prefix"
  meson compile -C "$ROOT/source/meson-tree/build"
  meson install -C "$ROOT/source/meson-tree/build"
  run_c_consumers "$prefix" "$ver"
}

route_source_cmake() {
  local ver=$1
  local src=$ROOT/source/linkcell-$ver
  local prefix=$ROOT/source/cmake-prefix
  log "cmake $(cmake --version | head -1)"
  rm -rf "$ROOT/source/cmake-tree" "$prefix"
  cp -a "$src" "$ROOT/source/cmake-tree"
  export CARGO_HOME=$ROOT/source/cmake-cargo
  export CARGO_TARGET_DIR=$ROOT/source/cmake-target
  rm -rf "$CARGO_HOME" "$CARGO_TARGET_DIR"
  cmake -S "$ROOT/source/cmake-tree" -B "$ROOT/source/cmake-tree/build" -DCMAKE_INSTALL_PREFIX="$prefix"
  cmake --build "$ROOT/source/cmake-tree/build"
  cmake --install "$ROOT/source/cmake-tree/build"
  run_c_consumers "$prefix" "$ver"
}

route_source_wrap() {
  local ver=$1
  local dir=$ROOT/source/wrap
  rm -rf "$dir"
  mkdir -p "$dir/subprojects"
  cp "$CHECKS/two_points.c" "$dir/app.c"
  cat > "$dir/subprojects/linkcell.wrap" << EOF
[wrap-git]
url = https://github.com/d-SEAMS/linkcell.git
revision = v${ver}
depth = 1

[provide]
linkcell = linkcell_dep
EOF
  cat > "$dir/meson.build" << 'EOF'
project('app', 'c', default_options: ['c_std=c11'])
linkcell_dep = dependency('linkcell', fallback: ['linkcell', 'linkcell_dep'])
executable('app', 'app.c', dependencies: linkcell_dep)
EOF
  export CARGO_HOME=$ROOT/source/wrap-cargo
  export CARGO_TARGET_DIR=$ROOT/source/wrap-target
  rm -rf "$CARGO_HOME" "$CARGO_TARGET_DIR"
  meson setup "$dir/build" "$dir"
  meson compile -C "$dir/build"
  local app
  app=$(find "$dir/build" -type f -name app -perm -111 | head -1)
  test -n "$app"
  log "wrap app $app"
  "$app" > "$dir/app.out"
  printf '0 -> 1\n1 -> 0\n' > "$dir/expected.out"
  diff -u "$dir/expected.out" "$dir/app.out"
  log "wrap quickstart ok"
}

# Prove the pinned recipe's source. context.version selects the tag
# tarball and source.sha256 is checked with sha256sum -c. A recipe version
# that differs from the crates.io max is not a failure here. The conda
# route requires that match, because it installs this recipe and checks the
# built package against the published crate. This workflow has no schedule;
# published-install runs on push and pull request, and `all` runs this
# route before conda and stops at the first failure. The PyPI route
# likewise follows its own version, which may lag crates.io.
route_source() {
  local ver
  require_cmd curl python3 meson cmake cc c++ pkg-config sha256sum
  prepare_source
  ver=$SOURCE_VERSION
  [[ "$ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
  log "source tag v$ver"
  route_source_meson "$ver"
  route_source_cmake "$ver"
  route_source_wrap "$ver"
}

repo_version() {
  python3 - "$1" << 'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r'(?m)^version = "([^"]+)"', text)
if not match:
    raise SystemExit(f"no version in {sys.argv[1]}")
print(match.group(1))
PY
}

copy_tree() {
  local src=$1
  local dest=$2
  rm -rf "$dest"
  mkdir -p "$dest"
  tar -C "$src" --exclude target --exclude .git --exclude build -cf - . | tar -C "$dest" -xf -
}

# Meson and CMake of this tree, not the published tag. This is the
# regression check for a compiler the published tarball does not link.
route_checkout() {
  local repo ver src prefix csrc cprefix
  require_cmd meson cmake pkg-config python3 tar cc c++
  repo=$(cd "$SCRIPT_DIR/.." && pwd)
  ver=$(repo_version "$repo/Cargo.toml")
  log "checkout $repo version $ver CC=${CC:-cc}"
  "${CC:-cc}" --version | head -1
  src=$ROOT/checkout/tree
  prefix=$ROOT/checkout/meson-prefix
  copy_tree "$repo" "$src"
  export CARGO_HOME=$ROOT/checkout/meson-cargo
  export CARGO_TARGET_DIR=$ROOT/checkout/meson-target
  rm -rf "$CARGO_HOME" "$CARGO_TARGET_DIR" "$prefix"
  meson setup "$src/build" "$src" --prefix "$prefix"
  meson compile -C "$src/build"
  meson install -C "$src/build"
  run_c_consumers "$prefix" "$ver"
  csrc=$ROOT/checkout/cmake-tree
  cprefix=$ROOT/checkout/cmake-prefix
  copy_tree "$repo" "$csrc"
  export CARGO_HOME=$ROOT/checkout/cmake-cargo
  export CARGO_TARGET_DIR=$ROOT/checkout/cmake-target
  rm -rf "$CARGO_HOME" "$CARGO_TARGET_DIR" "$cprefix"
  cmake -S "$csrc" -B "$csrc/build" -DCMAKE_INSTALL_PREFIX="$cprefix"
  cmake --build "$csrc/build"
  cmake --install "$csrc/build"
  run_c_consumers "$cprefix" "$ver"
}

route_conda() {
  local ver recipe_ver prefix pkg env micromamba
  require_cmd curl python3
  micromamba=${MICROMAMBA:-}
  if [[ -z "$micromamba" ]]; then
    mkdir -p "$ROOT/conda/bin"
    if [[ ! -x "$ROOT/conda/bin/micromamba" ]]; then
      curl -fsSL -o "$ROOT/conda/micromamba.tar.bz2" https://micro.mamba.pm/api/micromamba/linux-64/latest
      tar -C "$ROOT/conda" -xjf "$ROOT/conda/micromamba.tar.bz2"
    fi
    micromamba=$ROOT/conda/bin/micromamba
  fi
  ver=$(crate_version)
  [[ "$ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
  log "conda recipe for published $ver from $RECIPE_COMMIT"
  rm -rf "$ROOT/conda/recipe" "$ROOT/conda/output" "$ROOT/conda/env" "$ROOT/conda/rattler"
  mkdir -p "$ROOT/conda/recipe"
  local f
  for f in recipe.yaml build.sh build.bat test_knearest.c test_knearest_csr.cpp; do
    curl -fsSL -o "$ROOT/conda/recipe/$f" "$RECIPE_URL/$f"
  done
  chmod +x "$ROOT/conda/recipe/build.sh"
  sha256sum "$ROOT/conda/recipe/"*
  recipe_ver=$(python3 - << PY
import re
text = open("$ROOT/conda/recipe/recipe.yaml").read()
m = re.search(r'version:\s*"([^"]+)"', text)
print(m.group(1) if m else "")
PY
)
  log "recipe version $recipe_ver"
  test "$recipe_ver" = "$ver"
  # Same compiler and stdlib pins as staged-recipes linux-64 CI.
  # channel_sources is omitted: it conflicts with the explicit conda-forge channel.
  curl -fsSL -o "$ROOT/conda/linux64.yaml" \
    "https://raw.githubusercontent.com/HaoZeke/staged-recipes/${RECIPE_COMMIT}/.ci_support/linux64.yaml"
  python3 - << PY
from pathlib import Path
src = Path("$ROOT/conda/linux64.yaml").read_text().splitlines()
keys = {"c_compiler", "cxx_compiler", "c_stdlib", "c_stdlib_version", "target_platform"}
out = []
keep = False
seen = set()
for line in src:
    if line and not line.startswith((" ", "-", "#")) and line.endswith(":"):
        key = line[:-1]
        keep = key in keys
        if keep:
            seen.add(key)
    if keep:
        out.append(line)
missing = keys - seen
if missing:
    raise SystemExit(f"linux64.yaml missing {sorted(missing)}")
Path("$ROOT/conda/variants.yaml").write_text("\n".join(out) + "\n")
print("variant pins:")
print("\n".join(out))
PY
  export MAMBA_ROOT_PREFIX=$ROOT/conda/root
  "$micromamba" create -y -p "$ROOT/conda/rattler" -c conda-forge "rattler-build=0.80.2"
  env -u CARGO_HOME -u CARGO_TARGET_DIR -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS \
    RATTLER_BUILD_LOG_STYLE=plain RATTLER_BUILD_COLOR=never \
    "$ROOT/conda/rattler/bin/rattler-build" build \
      --recipe "$ROOT/conda/recipe" \
      --channel conda-forge \
      --variant-config "$ROOT/conda/variants.yaml" \
      --output-dir "$ROOT/conda/output" \
      --test native \
      --color never \
      --log-style plain \
      --no-config
  pkg=$(find "$ROOT/conda/output" -type f \( -name 'linkcell-*.conda' -o -name 'linkcell-*.tar.bz2' \) | head -1)
  test -n "$pkg"
  log "conda artifact $pkg"
  "$micromamba" create -y -p "$ROOT/conda/env" -c conda-forge gcc_linux-64 gxx_linux-64 pkg-config
  "$micromamba" install -y -p "$ROOT/conda/env" -c conda-forge "$pkg"
  prefix=$ROOT/conda/env
  # Compile with the prefix compilers so the consumer matches the package.
  (
    export MAMBA_ROOT_PREFIX=$ROOT/conda/root
    # shellcheck disable=SC1091
    eval "$("$micromamba" shell hook -s bash)"
    micromamba activate "$prefix"
    log "conda CC=${CC:-cc} $(${CC:-cc} --version | head -1)"
    run_c_consumers "$prefix" "$ver"
  )
}

dispatch() {
  local name=$1
  local status=0
  shift
  log "===== $name ====="
  # A subshell keeps set -e in force. An if-test would disable it.
  set +e
  (
    set -euo pipefail
    "$@"
  ) > >(tee "$LOG_DIR/$name.log") 2>&1
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    printf 'PASS\n' > "$LOG_DIR/$name.result"
    log "$name PASS"
  else
    printf 'FAIL %s\n' "$status" > "$LOG_DIR/$name.result"
    log "$name FAIL ($status)"
    return "$status"
  fi
}

main() {
  local which=${1:-all}
  if [[ -z "$ROOT" ]]; then
    ROOT=$(mktemp -d "${TMPDIR:-/tmp}/linkcell-prove.XXXXXX")
  fi
  if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR=$ROOT/logs
  fi
  mkdir -p "$ROOT" "$LOG_DIR"
  EXPECT=${EXPECT:-}
  REPORT=${REPORT:-$ROOT/pypi/report.json}
  export EXPECT REPORT
  log "root $ROOT"
  log "logs $LOG_DIR"
  case "$which" in
    crates) dispatch crates-io route_crates ;;
    pypi) dispatch pypi-wheel route_pypi ;;
    source) dispatch source route_source ;;
    conda) dispatch conda-recipe route_conda ;;
    checkout) dispatch checkout route_checkout ;;
    all)
      dispatch crates-io route_crates
      dispatch pypi-wheel route_pypi
      dispatch source route_source
      dispatch conda-recipe route_conda
      ;;
    *)
      log "usage: $0 [crates|pypi|source|conda|checkout|all]"
      exit 2
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
