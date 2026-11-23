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
RECIPE_COMMIT=8c81464f1dac30fd6d9967c71e5257c82375e0ab
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
# the tag in source.url. source.sha256 is that tarball. Other version or
# sha256 fields are ignored.
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


def section(name):
    match = re.search(rf"(?m)^{re.escape(name)}:\n((?:[ \t].*\n)*)", text)
    if not match:
        raise SystemExit(f"recipe has no {name} section")
    return match.group(1)


context = section("context")
source = section("source")
version = re.search(r'(?m)^[ \t]+version:\s*"([^"]+)"\s*$', context)
if not version:
    raise SystemExit("recipe context has no version")
version = version.group(1)
if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
    raise SystemExit(f"recipe context version is not X.Y.Z: {version}")
sha = re.search(r"(?m)^[ \t]+sha256:\s*([0-9a-f]{64})\s*$", source)
if not sha:
    raise SystemExit("recipe source has no sha256")
print(version)
print(sha.group(1))
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
  for f in recipe.yaml build.sh build.bat test_knearest.c; do
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
