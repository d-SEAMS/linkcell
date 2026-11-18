#!/usr/bin/env python3
"""Fail when a copy of the version disagrees with Cargo.toml [package].version.

The copies are the Python crate, pyproject.toml, Cargo.lock, meson.build,
CITATION.cff, the `revision = vX.Y.Z` wrap snippets and the
`find_package(linkcell X.Y` lines in the docs. `--tag NAME` also requires
NAME to be v<version>, so a tag pushed before the release commit fails.

It also fails when the Rust floor drifts. `rust-version` in Cargo.toml is
the crate's minimum, `rust-version` in python/Cargo.toml is the Python
module's, and the README states both. `--msrv linkcell|python` prints one
for the CI jobs that install exactly that toolchain, as X.Y.0.
"""

import argparse
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SNIPPETS = [
    "README.md",
    "docs/how-to/embed-c.md",
    "docs/how-to/embed-cpp.md",
    "tests/cmake-consumer/CMakeLists.txt",
]


def toml(rel):
    return tomllib.loads((ROOT / rel).read_text(encoding="utf-8"))


def copies():
    yield "python/Cargo.toml", toml("python/Cargo.toml")["package"]["version"]
    yield "pyproject.toml", toml("pyproject.toml")["project"]["version"]
    for package in toml("Cargo.lock")["package"]:
        if package["name"] in ("linkcell", "linkcell-python"):
            yield f"Cargo.lock {package['name']}", package["version"]
    meson = (ROOT / "meson.build").read_text(encoding="utf-8")
    project = re.search(r"^project\((.*?)^\)", meson, re.M | re.S)
    found = re.findall(r"^\s*version:\s*'([^']*)'", project.group(1), re.M) if project else []
    yield "meson.build project()", found[0] if len(found) == 1 else found
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    found = re.findall(r"^version:\s*['\"]?([^'\"\s]+)", cff, re.M)
    yield "CITATION.cff", found[0] if len(found) == 1 else found


def rust_versions():
    return {
        "linkcell": toml("Cargo.toml")["package"]["rust-version"],
        "python": toml("python/Cargo.toml")["package"]["rust-version"],
    }


def toolchain(version):
    """`1.80` is rustup's newest 1.80.x; the floor is the first release, 1.80.0."""
    return version if version.count(".") >= 2 else f"{version}.0"


def as_tuple(version):
    return tuple(int(part) for part in version.split("."))


def msrv_errors():
    floors = rust_versions()
    errors = []
    if as_tuple(floors["python"]) < as_tuple(floors["linkcell"]):
        errors.append(
            f"python/Cargo.toml rust-version {floors['python']} is below "
            f"Cargo.toml rust-version {floors['linkcell']}"
        )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for key, pattern in (
        ("linkcell", r"The crate needs Rust (\d+\.\d+) or newer"),
        ("python", r"Python module from source needs Rust (\d+\.\d+) or newer"),
    ):
        found = re.findall(pattern, readme)
        if found != [floors[key]]:
            errors.append(f"README.md states Rust {found}, but the {key} rust-version is {floors[key]}")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for got in re.findall(r"toolchain:\s*['\"]?(\d+\.\d+)", ci):
        if got not in floors.values():
            errors.append(f"ci.yml pins toolchain {got}, which is not a rust-version")
    for key in floors:
        if f"check-versions.py --msrv {key}" not in ci:
            errors.append(f"ci.yml has no job that reads the {key} rust-version")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="release tag, which must be v<version>")
    parser.add_argument(
        "--msrv",
        choices=("linkcell", "python"),
        help="print that crate's rust-version and exit",
    )
    args = parser.parse_args()
    if args.msrv:
        print(toolchain(rust_versions()[args.msrv]))
        return

    version = toml("Cargo.toml")["package"]["version"]
    minor = ".".join(version.split(".")[:2])
    errors = [f"{where} has {got!r}" for where, got in copies() if got != version]
    for rel in SNIPPETS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for got in re.findall(r"^revision = (\S+)", text, re.M):
            if got != f"v{version}":
                errors.append(f"{rel} wrap revision is {got}")
        for got in re.findall(r"find_package\(linkcell (\S+)", text):
            if got != minor:
                errors.append(f"{rel} asks find_package(linkcell {got})")
    if args.tag is not None and args.tag != f"v{version}":
        errors.append(f"tag {args.tag} is not v{version}")
    for error in errors:
        print(f"Cargo.toml is {version}, but {error}", file=sys.stderr)
    floor_errors = msrv_errors()
    for error in floor_errors:
        print(error, file=sys.stderr)
    if errors or floor_errors:
        sys.exit(1)
    floors = rust_versions()
    print(
        f"every copy of the version is {version}; "
        f"Rust floor {floors['linkcell']} (python {floors['python']})"
    )


if __name__ == "__main__":
    main()
