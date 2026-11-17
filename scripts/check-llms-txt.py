#!/usr/bin/env python3
"""Fail when llms.txt misses a docs page or links a page or file that does not exist."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOB = "https://github.com/d-SEAMS/linkcell/blob/main/"
SITE = "https://linkcell.dseams.info/"


def main():
    text = (ROOT / "llms.txt").read_text()
    errors = []
    pages = set()
    for m in re.finditer(re.escape(SITE) + r"([^)\s]*)", text):
        rel = m.group(1).rstrip("/")
        if rel.startswith("api/"):
            continue
        page = "docs/" + (rel or "index") + ".md"
        if not (ROOT / page).is_file():
            errors.append(f"llms.txt links {m.group(0)} but {page} does not exist")
        pages.add(page)
    for m in re.finditer(re.escape(BLOB) + r"([^)\s]+)", text):
        if not (ROOT / m.group(1)).is_file():
            errors.append(f"llms.txt links missing file {m.group(1)}")
    for page in sorted((ROOT / "docs").rglob("*.md")):
        rel = page.relative_to(ROOT).as_posix()
        if rel.startswith("docs/assets/"):
            continue
        if rel not in pages:
            errors.append(f"llms.txt does not list {rel}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"llms.txt lists {len(pages)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
