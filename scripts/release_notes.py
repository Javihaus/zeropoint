"""Print the CHANGELOG.md section of a version, to use as the notes of its GitHub release.

    python scripts/release_notes.py            # the version in src/zeropoint/_version.py
    python scripts/release_notes.py 0.2.0

Exits with an error when the changelog has no dated section for that version, so a release cannot be made
without its entry.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def current_version() -> str:
    text = (ROOT / "src" / "zeropoint" / "_version.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "(.+?)"', text).group(1)


def section(version: str) -> str:
    """The body of '## [version] - YYYY-MM-DD', without the heading."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)",
                      text, re.M | re.S)
    if not match:
        raise SystemExit(f"CHANGELOG.md has no dated section '## [{version}] - YYYY-MM-DD'.")
    return match.group(1).strip()


if __name__ == "__main__":
    print(section(sys.argv[1] if len(sys.argv) > 1 else current_version()))
