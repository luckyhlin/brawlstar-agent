#!/usr/bin/env python3
"""Sync .cursor/rules/*.mdc content into CLAUDE.md below a marker line.

Keeps Claude-Code-specific instructions (above the marker) intact,
and replaces everything between the markers with current cursor rules.

Usage:
    python scripts/sync-cursor-rules.py          # from project root
    uv run python scripts/sync-cursor-rules.py   # with uv
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
CURSOR_RULES_DIR = PROJECT_ROOT / ".cursor" / "rules"

MARKER_START = "<!-- BEGIN CURSOR RULES (auto-synced — do not edit below) -->"
MARKER_END = "<!-- END CURSOR RULES -->"


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter delimited by ---."""
    return re.sub(r"^---\n.*?\n---\n*", "", text, count=1, flags=re.DOTALL).strip()


def sync() -> int:
    if not CURSOR_RULES_DIR.is_dir():
        print(f"No cursor rules directory at {CURSOR_RULES_DIR}", file=sys.stderr)
        return 1

    mdc_files = sorted(CURSOR_RULES_DIR.glob("*.mdc"))
    if not mdc_files:
        print("No .cursor/rules/*.mdc files found — nothing to sync.")
        return 0

    # Read current CLAUDE.md
    content = CLAUDE_MD.read_text() if CLAUDE_MD.exists() else ""

    # Extract base (everything above the start marker)
    if MARKER_START in content:
        base = content[: content.index(MARKER_START)].rstrip()
    elif MARKER_END in content:
        base = content[: content.index(MARKER_END)].rstrip()
    else:
        base = content.rstrip()

    # Collect cursor rules
    rules = []
    for mdc_path in mdc_files:
        raw = mdc_path.read_text()
        body = strip_frontmatter(raw)
        if body:
            rel = mdc_path.relative_to(PROJECT_ROOT)
            rules.append(f"<!-- source: {rel} -->\n\n{body}")

    synced = "\n\n".join(rules)
    new_content = f"{base}\n\n{MARKER_START}\n\n{synced}\n\n{MARKER_END}\n"

    CLAUDE_MD.write_text(new_content)
    print(f"Synced {len(rules)} cursor rule(s) into CLAUDE.md")
    for f in mdc_files:
        print(f"  - {f.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(sync())
