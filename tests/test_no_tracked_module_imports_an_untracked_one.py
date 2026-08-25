"""Nothing this repository ships imports a file git does not have.

The defect this exists for is invisible to every other test in the suite, because every
other test runs against the **working tree** and this is a property of the **committed**
tree. A new module sitting on disk, imported by a file that is committed, passes the
entire suite and fails the moment anybody checks the commit out -- a fresh clone, a CI
runner, or a git worktree, which is how this project's parallel agents read each other's
work.

It happened. `control/api/app.py` was committed with `include_router` calls for two
modules that were still untracked: the suite was green, the routes worked, and the ref
could not import.

**The check is "exists on disk but is not tracked", and the precision is the whole
design.** A dotted name that resolves to no file at all is not a defect -- `from
managed_agent.core import ids` imports a *name* out of `core/__init__.py`, and there is
no `core/ids.py` to track. An earlier version of this guard asked whether the parent
package resolved instead, which accepted everything: the real defect's parent
(`managed_agent.control.api`) has an `__init__.py`, so that version would have passed
over the exact commit that motivated it. Its own can-fail test caught that.

Reads imports with `ast` rather than by importing anything, so a module with a side
effect at import time is not run in order to be checked.

Skipped when git is unavailable or this is not a work tree, because a guard that fails
for the absence of its own instrument reports the wrong thing. A skip is visible in the
suite's output; a silent pass would not be.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_PACKAGE = "managed_agent"


def _tracked_paths() -> frozenset[str]:
    """Every path git has, repo-relative, with forward slashes."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        pytest.skip("git is unavailable or this is not a work tree")
    return frozenset(p for p in listed.stdout.split("\0") if p)


def _imported_modules(source: str) -> set[str]:
    """Every `managed_agent.*` module name this source could be importing.

    Both statement forms, and `from x import y` contributes `x.y` as well as `x` --
    `y` may be a submodule or a name inside `x`, and which one it is is decided later
    by whether a file for it exists.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(
                alias.name for alias in node.names if alias.name.startswith(_PACKAGE)
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or not node.module.startswith(_PACKAGE):
                continue
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _candidate_paths(module: str) -> tuple[str, ...]:
    """The two paths that could hold this module, either of which satisfies it."""
    stem = "src/" + module.replace(".", "/")
    return (f"{stem}.py", f"{stem}/__init__.py")


def _untracked_file_behind(module: str, tracked: frozenset[str]) -> str | None:
    """The on-disk path for this module that git has no record of, if there is one.

    Returns None both when the module is properly tracked and when it names no file at
    all -- the second is not a defect, it is a name imported out of a package's
    `__init__.py`, and the import machinery is the thing that decides which.
    """
    candidates = _candidate_paths(module)
    if any(path in tracked for path in candidates):
        return None
    for path in candidates:
        if (_REPO / path).exists():
            return path
    return None


def test_no_tracked_module_imports_a_file_git_does_not_have() -> None:
    """Every import in every committed source file resolves in the committed tree."""
    tracked = _tracked_paths()
    ours = sorted(p for p in tracked if p.startswith("src/") and p.endswith(".py"))
    assert ours, "no tracked source files found, so this guard proved nothing"

    dangling: list[str] = []
    for path in ours:
        for module in sorted(_imported_modules((_REPO / path).read_text())):
            missing = _untracked_file_behind(module, tracked)
            if missing is not None:
                dangling.append(f"{path} imports {module} -> {missing} is untracked")

    assert not dangling, (
        "these imports resolve on disk but not in the committed tree, so this ref "
        "cannot be checked out and imported -- which is how a fresh clone, a CI "
        f"runner and a git worktree all read it: {dangling}"
    )


def test_the_scan_can_fail() -> None:
    """The three answers this guard has to keep apart, asserted one by one.

    Without this the assertion above passes over an empty scan, and a resolver that
    accepted everything would look exactly like a clean tree. The middle case is the
    one an earlier version of this file got wrong.
    """
    tracked = _tracked_paths()

    # Tracked module: fine.
    assert _untracked_file_behind(f"{_PACKAGE}.composition", tracked) is None
    # A name inside a package, no file of its own: also fine, and not the same fine.
    assert _untracked_file_behind(f"{_PACKAGE}.core.ids.SessionId", tracked) is None
    # A file on disk that git does not have: reported, with the path.
    scratch = _REPO / "src" / _PACKAGE / "_guard_probe_module.py"
    assert not scratch.exists(), "the probe path is already taken"
    scratch.write_text("# written and removed by test_the_scan_can_fail\n")
    try:
        found = _untracked_file_behind(f"{_PACKAGE}._guard_probe_module", tracked)
        assert found == f"src/{_PACKAGE}/_guard_probe_module.py"
    finally:
        scratch.unlink()

    assert _imported_modules(
        f"from {_PACKAGE}.core import ids\nimport {_PACKAGE}.composition\n"
    ) == {
        f"{_PACKAGE}.core",
        f"{_PACKAGE}.core.ids",
        f"{_PACKAGE}.composition",
    }
