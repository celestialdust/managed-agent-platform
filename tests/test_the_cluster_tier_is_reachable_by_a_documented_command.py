"""The command documented for running the cluster tier must actually select it.

The defect: `docs/session-state.md` disclosed for several sessions that the cluster
tests "run only when a human sets `MAP_CLUSTER_TESTS=1`". Setting it and running
`pytest -q tests/pod` gives `30 passed` and places nothing, because a second and
independent gate is still shut -- `addopts = "-m 'not network and not image'"`
deselects every file marked `network`, and a *deselected* test leaves no trace in the
summary at all. The tier that grades the whole objective was reported as thirty
passing tests.

A docstring cannot fix that, which is the point of this file. The disclosure was
accurate about the gate its author added; the gate that hid the tier lives in another
file and was added earlier. So this asserts the property directly: for every file
that opts into the cluster, the documented invocation selects at least one test.

Asserted on the selected count, never on an exit code. `N passed` with none of the
interesting tests selected is indistinguishable from success by every signal except
that number, and nobody reads it. This reads it.

`--collect-only` is used rather than a real run: whether the tier *passes* depends on
the cluster and on secrets that may be absent, and this is not about that. It is about
whether the tests are reachable, which is a property of the configuration alone and is
therefore checkable offline, in the default run, on any machine.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

_ROOT: Final = Path(__file__).resolve().parents[1]
_OPT_IN: Final = "requires_the_cluster"
_EVERY_MARK: Final = "image or not image"
"""The `-m` the disclosure tells a reader to pass, selecting every mark.

One expression rather than `-m network -m image`: `pyproject.toml` records the
measurement that a second `-m` replaces the first rather than adding to it, so the
two-flag form silently drops one opt-in suite. A tautology is the shortest expression
that selects everything, and it stays correct when a third mark is registered.
"""

_COLLECTED: Final = re.compile(
    r"(?P<selected>\d+)/(?P<of>\d+) tests collected"
    r"|(?P<only>\d+) tests? collected"
    r"|(?P<none>no tests collected)"
)
"""pytest spells the collected count three ways, and one of them is the word "no".

`no tests collected (4 deselected)` is the exact line the defect this file exists for
produces, so a pattern that cannot read it fails on its most important input.
"""


def _cluster_files() -> list[Path]:
    """Every test file gated on the cluster, found by the gate's own name.

    Searched for rather than listed. A list would be a second place to update when a
    cluster test is added, and the failure mode of forgetting is this file passing
    while the new tier file is unreachable -- the defect it exists to prevent.
    """
    return sorted(
        path
        for path in _ROOT.joinpath("tests").rglob("test_*.py")
        if _OPT_IN in path.read_text()
    )


def _selected(path: Path, *, marks: str | None) -> int:
    """How many tests the given invocation selects from one file.

    Run in a subprocess because the question is about pytest's own configuration --
    `addopts`, marks, `testpaths` -- and an in-process check would have to
    re-implement the mark expression evaluator. A re-implementation would agree with
    itself and not necessarily with the tool a human runs.
    """
    argv = [sys.executable, "-m", "pytest", str(path), "--collect-only", "-q"]
    if marks is not None:
        argv += ["-m", marks]
    done = subprocess.run(argv, cwd=_ROOT, capture_output=True, text=True)
    found = _COLLECTED.search(done.stdout)
    assert found, (
        f"no collected count in pytest's output for {path.name}; "
        f"tail: {done.stdout[-300:]!r}"
    )
    if found.group("none") is not None:
        return 0
    if found.group("selected") is not None:
        return int(found.group("selected"))
    return int(found.group("only"))


def test_some_file_opts_into_the_cluster() -> None:
    """At least one, or every check below sweeps an empty list and says nothing.

    This is the assertion that fails if `requires_the_cluster` is renamed. Without it
    the rename turns this file into a pass over zero files, which is the state it
    exists to make impossible elsewhere.
    """
    found = _cluster_files()
    assert found, (
        f"no test file under tests/ mentions {_OPT_IN!r}. Either the opt-in gate "
        "was renamed -- update the constant here -- or the cluster tier was "
        "removed, in which case delete this file rather than letting it pass over "
        "nothing."
    )


@pytest.mark.parametrize("path", _cluster_files(), ids=lambda p: p.name)
def test_the_documented_command_selects_the_cluster_tier(path: Path) -> None:
    """A cluster file the documented command cannot reach is a missing tier.

    The failure this prevents: somebody adds a third gate -- a new mark, a
    `collect_ignore` entry, a `conftest` hook -- and the documented command starts
    reporting a smaller tier with no error anywhere. The count going to zero is the
    only observable, so the count is what is asserted.
    """
    count = _selected(path, marks=_EVERY_MARK)
    relative = path.relative_to(_ROOT)
    assert count > 0, (
        f"{relative} opts into the cluster but the documented invocation selects "
        f"{count} of its tests. Something gates this file beyond the mark "
        "expression and the MAP_CLUSTER_TESTS skip, and a reader following the "
        "documented command will be told the tier passed without it having run."
    )


def test_the_default_run_really_does_hide_some_of_it() -> None:
    """Two sides: without the override, something must be deselected.

    Without this, the check above would keep passing on a day somebody deleted
    `addopts` entirely -- and would then be asserting that an unrestricted command
    selects tests, which is true of every command and says nothing. This is what
    makes that assertion a statement about the override rather than about pytest.
    """
    hidden = [p for p in _cluster_files() if _selected(p, marks=None) == 0]
    assert hidden, (
        "every cluster-tier file is already selected by a bare `pytest`, so the "
        "override is not what makes them reachable and the check above is vacuous. "
        "If the default run now includes the cluster tier, that is a real change "
        "and this file should be deleted rather than kept passing on a comparison "
        "with one side."
    )
