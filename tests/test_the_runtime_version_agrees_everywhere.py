"""Every declaration of the codex runtime version in this tree says the same thing.

The version is a contract between the image the Session pod runs and every measurement
taken against it, and `docs/lessons.md:1199` says such a constant is imported and never
restated. It cannot be imported everywhere here: pytest puts each test directory on
`sys.path` separately with no `tests/__init__.py`, so `tests/deploy/` cannot import from
`tests/pod/`, and the repo's convention is that no test imports another across
directories. So there are several declarations by necessity.

This makes that duplication *checked* rather than merely accepted. Declarations that
agree are safe; declarations that can drift silently are the defect the lesson
describes, and the way they drift is that somebody re-measures against a new runtime and
updates the file they happened to be looking at. No count is written here on purpose --
the first run of this scan found four where two were expected, one of them in
`tests/spike/`, so a number in this docstring would have been wrong the day it was
typed.

Found by walking the syntax tree for the **assignment**, not by grepping for a version
string. A grep for `0.149.0` would pass the day somebody changes every copy but one to
`0.150.0`, because the one it cannot see is the one that still matches.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[1]
_NAME: Final = "CODEX_VERSION"


def _declarations() -> dict[str, str]:
    """Each file declaring the version, mapped to the literal it assigns.

    Only module-level assignments of a plain string are collected. An assignment built
    from an expression is not a declaration of a version; it is a derivation, and a
    derivation cannot disagree with its source.
    """
    found: dict[str, str] = {}
    for path in sorted(_ROOT.joinpath("tests").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            named = any(
                isinstance(one, ast.Name) and one.id == _NAME for one in targets
            )
            if not named:
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found[str(path.relative_to(_ROOT))] = value.value
    return found


def test_the_scan_finds_the_declarations_it_exists_to_compare() -> None:
    """At least two, or the comparison below is vacuous.

    One declaration agrees with itself and zero agree even harder. This is the assertion
    that turns "they all match" into a statement about something.
    """
    found = _declarations()
    assert len(found) >= 2, (
        f"found {len(found)} module-level {_NAME} declarations under tests/: "
        f"{found}. With fewer than two there is nothing to compare, and this file "
        "passes while checking nothing -- if the constant really has been consolidated "
        "to one place, delete this file rather than letting it pass vacuously."
    )


def test_every_declaration_of_the_runtime_version_agrees() -> None:
    """A disagreement means some measurement in this repo is about a different runtime.

    The failure it prevents: one file is re-measured against a new codex-cli and
    updated, the others keep asserting the old version, both sets of tests pass, so the
    repository certifies two runtimes at once and says nothing about which one the
    compiler's docstrings were reasoning about.
    """
    found = _declarations()
    versions = set(found.values())
    assert len(versions) == 1, (
        f"the codex runtime version is declared {len(versions)} different ways: "
        f"{found}. Every empirical claim in tests/pod/ and tests/deploy/ is measured "
        "against one runtime, and a split here means at least one of them is about a "
        "version nothing runs. Update them together or consolidate the declaration."
    )
