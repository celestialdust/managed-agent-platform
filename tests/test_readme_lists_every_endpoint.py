"""README.md's endpoint tables and the surface the app serves are the same set.

The README is the only description of this API that somebody outside the repository
reads, and a table of endpoints is the part of it most likely to rot: a route is added
in a router module, the app publishes it, and nothing anywhere disagrees with a README
that has never heard of it. The two directions fail for different reasons and a reader
of the failure needs to know which. An endpoint the README omits is the dangerous one --
callers cannot use what nobody told them about, and the omission is invisible from
inside the code. A documented endpoint that no longer exists is milder, but it still
sends somebody to write a client against a path that answers 404.

The comparison is against `tools/api_surface.py` run as a subprocess rather than
against its `operations()` imported directly, because the command line is the interface
that tool actually has -- `make api-surface` is what a person types, and importing the
function would leave the thing anyone uses unexercised.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"
_SURFACE = _ROOT / "tools" / "api_surface.py"

# A table row whose first two cells are a backticked verb and a backticked path. Prose
# mentioning an endpoint is not a row and is deliberately not matched: the tables are
# the reference, and a sentence in the philosophy section naming one path is not a claim
# that the table is complete.
_A_TABLE_ROW = re.compile(
    r"^\|\s*`(GET|POST|PUT|PATCH|DELETE)`\s*\|\s*`(/v1/[^`]*)`\s*\|", re.MULTILINE
)


@cache
def _served() -> frozenset[tuple[str, str]]:
    finished = subprocess.run(
        [sys.executable, str(_SURFACE)],
        capture_output=True,
        text=True,
        check=True,
        cwd=_ROOT,
    )
    rows = (
        line.partition("\t") for line in finished.stdout.splitlines() if line.strip()
    )
    return frozenset((method, path) for method, _, path in rows)


def _documented() -> frozenset[tuple[str, str]]:
    return frozenset(_A_TABLE_ROW.findall(_README.read_text(encoding="utf-8")))


def test_every_served_endpoint_appears_in_the_readme() -> None:
    missing = sorted(_served() - _documented())
    assert not missing, (
        f"{len(missing)} operation(s) the app serves are in no README table: "
        f"{missing}. Add a row for each under the matching heading in "
        "'The full surface'."
    )


def test_the_readme_documents_no_endpoint_the_app_does_not_serve() -> None:
    stale = sorted(_documented() - _served())
    assert not stale, (
        f"README.md documents {stale}, which the app does not serve. Run "
        "`make api-surface` for the list it does."
    )


def test_the_surface_is_not_accidentally_empty() -> None:
    """A scanner finding nothing would pass both checks above against an empty README.

    The two comparisons are set differences, so `frozenset() - frozenset()` is empty and
    both assertions hold while nothing at all has been graded. This is the case that
    turns a silently broken tool into a failure instead of a green run.
    """
    assert len(_served()) > 50, f"the app published {len(_served())} operations"
