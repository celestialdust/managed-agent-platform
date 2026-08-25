"""Enumerate every environment variable that can skip a test in this suite.

A gate is an environment variable read inside a `pytest.mark.skipif(...)` condition:
set it and a tier of tests runs, leave it and that tier reports as skipped. They are
the most dangerous kind of configuration in a test suite, because an unreachable guard
and a passing guard produce the same green summary -- and this project has paid for
that once already, with six verification rounds spent on a question a test in the tree
answered in two seconds, gated behind a name that appeared nowhere else in the
repository. See `docs/lessons.md`.

Prints one gate per line, sorted, with the files that define it. `tests/test_gates.py`
compares this set against `docs/test-gates.md`, so a new gate fails the suite until it
is written down somewhere a person looking for it would look.

Resolves module-level string constants, because gates are conventionally named once at
the top of a file (`_READBACK_GATE = "MAP_IAM_READBACK"`) and referenced by that name
inside the decorator -- a scanner that only saw string literals would miss every gate
written the tidy way, which is most of them.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = _ROOT / "tests"
_NAME_SHAPE = re.compile(r"\AMAP_[A-Z0-9_]+\Z")


def _module_constants(tree: ast.Module) -> dict[str, list[str]]:
    """Module-level `NAME = "STR"` and `NAME = ("STR", ...)` bindings.

    Only module level: a name bound inside a function cannot be referenced from a
    decorator, so including those would invent gates that do not exist.
    """
    found: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found[target.id] = [value.value]
        elif isinstance(value, (ast.Tuple, ast.List)):
            strings = [
                element.value
                for element in value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if strings:
                found[target.id] = strings
    return found


def _is_skipif(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "skipif"


def _names_within(node: ast.AST, constants: dict[str, list[str]]) -> set[str]:
    """Every gate-shaped environment name reachable from one skipif condition."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if _NAME_SHAPE.fullmatch(child.value):
                names.add(child.value)
        elif isinstance(child, ast.Name):
            for resolved in constants.get(child.id, []):
                if _NAME_SHAPE.fullmatch(resolved):
                    names.add(resolved)
    return names


def gates() -> dict[str, set[str]]:
    """Gate name -> the test files whose skipif conditions read it."""
    found: dict[str, set[str]] = defaultdict(set)
    for path in sorted(_TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        constants = _module_constants(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_skipif(node):
                for name in _names_within(node, constants):
                    found[name].add(str(path.relative_to(_ROOT)))
    return dict(found)


def main() -> int:
    for name, files in sorted(gates().items()):
        print(f"{name}\t{' '.join(sorted(files))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
