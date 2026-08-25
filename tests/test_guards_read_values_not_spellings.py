"""No test in this suite asserts on a boolean CLI flag by matching its spelling.

The defect this exists to prevent has now happened twice, and `docs/lessons.md` carries
both entries. A guard wrote `assert "--skip-nodes-with-local-storage=false" not in args`
and believed it had covered the flag. Go's `flag` package parses a bool with
`strconv.ParseBool`, which reads false from six spellings, so the guard covered one of
six and five edits could disable a live protection with the suite green.

The rule is one sentence: **ask the parser, not the string.** Read the flag's value and
reject every value that is not provably safe -- values no parser accepts included,
since a program handed one exits at startup. Then a spelling nobody thought of
fails closed instead of passing.

This is deliberately narrow. It catches the shape that recurred -- a string literal
carrying a flag and a boolean spelling, inside an assertion -- and does not judge
assertions in general. A narrow check that fires is worth more than a broad one
that has to be suppressed everywhere, and being suppressed everywhere is how a
guard stops being read.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = _ROOT / "tests"

# `--some-flag=false` and the other five spellings Go's strconv.ParseBool accepts, plus
# the true side: asserting a flag is present as `=true` is the same mistake wearing the
# safe value, because it fails when somebody writes the equally-valid `=1`.
_FLAG_WITH_A_BOOL = re.compile(
    r"--[a-z0-9][a-z0-9-]*=(?:true|True|TRUE|t|T|1|false|False|FALSE|f|F|0)\Z"
)

# This file quotes the shape in its own docstring and in the exemption list below, and
# `docs/`-facing text is not an assertion. Only assertions are examined, so prose is
# out of scope by construction -- but a test file that must carry such a literal for a
# reason (rendering one, say, rather than asserting on one) names itself here with why.
_EXEMPT: dict[str, str] = {}


def _offences() -> list[str]:
    found: list[str] = []
    for path in sorted(_TESTS.rglob("*.py")):
        relative = str(path.relative_to(_ROOT))
        if relative in _EXEMPT or path.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            for child in ast.walk(node.test):
                if (
                    isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and _FLAG_WITH_A_BOOL.search(child.value)
                ):
                    found.append(f"{relative}:{child.lineno}: {child.value!r}")
    return found


def test_no_assertion_matches_a_boolean_flag_by_its_spelling() -> None:
    offences = _offences()
    assert not offences, (
        "these assertions match a boolean CLI flag as text:\n  "
        + "\n  ".join(offences)
        + "\n\nA bool flag has six false spellings and six true ones (Go's "
        "strconv.ParseBool: false/False/FALSE/f/F/0). Matching one covers one. Read "
        "the "
        "flag's value instead and reject everything that is not provably safe -- an "
        "unparseable value included, because the program would exit at startup. See "
        "test_a_node_running_a_session_is_never_drained_to_save_money in "
        "tests/deploy/test_cluster_autoscaler.py for the shape, and docs/lessons.md "
        "for "
        "the two times this cost us."
    )


def test_the_detector_finds_the_shape_it_exists_to_find() -> None:
    """The detector is falsifiable, which a detector asserting an empty set must prove.

    A scan that returns nothing passes whether it is working or broken -- a bug that
    stops it parsing, a regex that matches nothing, a directory it never walks all look
    identical to a clean repository. So the pattern is exercised against text known to
    contain it, rather than trusting the empty result above to mean anything.
    """
    for spelling in ("false", "False", "FALSE", "f", "F", "0", "true", "1"):
        assert _FLAG_WITH_A_BOOL.search(
            f"--skip-nodes-with-local-storage={spelling}"
        ), f"the detector does not recognise ={spelling}, which Go parses as a bool"
    for safe in (
        "--namespace=map-dev",
        "--max-nodes-total=8",
        "--cloud-provider=aws",
        "--node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled",
    ):
        assert not _FLAG_WITH_A_BOOL.search(safe), f"false positive on {safe}"
