"""The version number and the changelog stay in step: every version has its entry."""

import re
from pathlib import Path

import zeropoint
from zeropoint._version import FORMAT

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_version_is_semantic():
    assert re.fullmatch(r"\d+\.\d+\.\d+", zeropoint.__version__)


def test_changelog_has_a_dated_section_for_this_version():
    heading = rf"^## \[{re.escape(zeropoint.__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}$"
    assert re.search(heading, CHANGELOG, re.M), f"add '## [{zeropoint.__version__}] - YYYY-MM-DD' to CHANGELOG.md"


def test_changelog_keeps_an_unreleased_section_on_top():
    sections = re.findall(r"^## \[([^\]]+)\]", CHANGELOG, re.M)
    assert sections[0] == "Unreleased"
    assert sections[1] == zeropoint.__version__, "the newest released section must be the current version"


def test_file_format_is_a_positive_integer():
    assert isinstance(FORMAT, int) and FORMAT >= 1
