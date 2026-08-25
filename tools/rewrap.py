"""Rewrap over-long prose lines in a Python file's comments and module docstring.

Written because hand-wrapping prose at 88 columns has cost this run four separate E501
round-trips: write the paragraph, run ruff, find one line at 89, edit, run again. The
formatter will not do it -- `ruff format` reflows code and leaves prose alone, by
design, because it cannot tell a sentence from a doctest.

Scope is deliberately narrow, and it got narrower after this tool broke a file. It
rewraps **comment lines only** -- a run of consecutive `#` lines at one indent. It does
not touch docstrings, and that is not an omission: the first version did, and it
reflowed a line inside an f-string in an assertion message, producing
`f-string: unterminated string` and an unparseable file. Telling prose from code by
regex fails on exactly the cases that matter, and a tool that silently corrupts source
is worse than the E501 it was written to save. Docstrings get wrapped by hand.

Usage: uv run python tools/rewrap.py <file> [<file> ...]
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

WIDTH = 88


def _is_comment(line: str) -> bool:
    """A whole-line comment, which is the only thing this tool will reflow."""
    return bool(re.match(r"^\s*#", line)) and bool(line.strip().lstrip("#").strip())


def rewrap(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if len(line) <= WIDTH or not _is_comment(line):
            out.append(line)
            i += 1
            continue
        # Gather the paragraph: this line plus following prose lines at the same indent.
        indent_match = re.match(r"^(\s*#?\s*)", line)
        indent = indent_match.group(1) if indent_match else ""
        para = [line[len(indent) :]]
        j = i + 1
        while j < len(lines) and _is_comment(lines[j]) and lines[j].startswith(indent):
            para.append(lines[j][len(indent) :])
            j += 1
        wrapped = textwrap.wrap(" ".join(para), width=WIDTH - len(indent))
        out.extend(indent + w for w in (wrapped or [""]))
        i = j
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for name in argv:
        path = Path(name)
        before = path.read_text()
        after = rewrap(before)
        if after != before:
            path.write_text(after)
            print(f"rewrapped {name}")
        else:
            print(f"unchanged {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
