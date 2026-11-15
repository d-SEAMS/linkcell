#!/usr/bin/env python3
"""Fail when llms.txt misses a docs page or links a file that does not exist."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "https://github.com/d-SEAMS/linkcell/blob/main/"


def main():
    text = (ROOT / "llms.txt").read_text()
    linked = {m.group(1) for m in re.finditer(re.escape(PREFIX) + r"([^)\s]+)", text)}
    errors = [f"llms.txt links missing file {p}" for p in sorted(linked) if not (ROOT / p).is_file()]
    for page in sorted((ROOT / "docs").rglob("*.md")):
        rel = page.relative_to(ROOT).as_posix()
        if rel not in linked:
            errors.append(f"llms.txt does not list {rel}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"llms.txt lists {len(linked)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
