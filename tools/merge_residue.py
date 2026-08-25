"""Fail when a tracked file carries text a conflict resolution should have removed.

A merge conflict region is a line range, not a syntactic unit. Git will happily split a
function call, a YAML block or a table row across the `=======` line, so "keep both
sides" can leave a file that is textually plausible and structurally broken -- and a
merge commit runs no gate on its own, so nothing looks at it before it lands.

Two residues are checked, because the two have different reach:

- **Conflict markers.** A leftover `<<<<<<<` in Python is a syntax error `ruff` reports,
  but in YAML, Markdown, JSON or Terraform it is just text, and every one of those is a
  deployable artefact here. This is the half `ruff` cannot see.
- **Unparseable Python.** Redundant with `ruff check .` and kept anyway, because this is
  the tool a person reaches for straight after resolving a conflict, and a guard that
  answers half the question invites the other half to go unasked.

Exits 0 when clean, 1 with the offending paths when not. Reads the git index rather than
walking the tree, so untracked scratch files and anything gitignored are out of scope by
construction.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

# Only the opening and closing markers. `=======` alone is a legal Markdown setext
# underline and appears in this repository's own documents, so matching it would be a
# false positive generator rather than a guard.
_MARKERS = ("<<<<<<< ", ">>>>>>> ")


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=True, text=True
    ).stdout
    return [Path(name) for name in out.split("\0") if name]


def main() -> int:
    marked: list[str] = []
    unparseable: list[str] = []

    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: no residue is detectable, so not judged

        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.startswith(_MARKERS):
                marked.append(f"{path}:{line_no}: {line[:40]}")

        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as broken:
                unparseable.append(f"{path}:{broken.lineno}: {broken.msg}")

    for label, found in (
        ("MERGE RESIDUE - conflict marker left in a tracked file", marked),
        ("MERGE RESIDUE - tracked Python does not parse", unparseable),
    ):
        if found:
            print(label)
            for entry in found:
                print(f"  {entry}")

    return 1 if marked or unparseable else 0


if __name__ == "__main__":
    sys.exit(main())
