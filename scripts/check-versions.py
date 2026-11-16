#!/usr/bin/env python3
"""Fail when a copy of the version disagrees with Cargo.toml [package].version.

The copies are the Python crate, pyproject.toml, Cargo.lock, meson.build,
CITATION.cff, the `revision = vX.Y.Z` wrap snippets and the
`find_package(linkcell X.Y` lines in the docs. `--tag NAME` also requires
NAME to be v<version>, so a tag pushed before the release commit fails.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="release tag, which must be v<version>")
    args = parser.parse_args()

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
    if errors:
        sys.exit(1)
    print(f"every copy of the version is {version}")


if __name__ == "__main__":
    main()
