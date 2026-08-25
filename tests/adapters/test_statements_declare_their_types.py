"""Every textual SQL statement in an adapter declares the types of its parameters.

A `sa.text(...)` statement carries no column metadata, so SQLAlchemy has nothing to
infer from and asyncpg receives whatever Python object the caller handed it. A mapping
bound to a `jsonb` column is refused outright — `'dict' object has no attribute
'encode'`, on the first insert.

**What an undeclared uuid actually does, measured — because the first version of this
docstring got it wrong.** It said a uuid bound as a `str` "is compared against whatever
punctuation and case that particular spelling used, so two spellings of one id stop
matching and nothing raises." That is false, and the measurement is easy: against real
PostgreSQL 17 over asyncpg, inserting `str(uuid4())` into a `uuid` column with no
declared type and then selecting on it matched for the canonical spelling, for the
UPPERCASE spelling, and for the unhyphenated 32-character spelling. asyncpg parses the
text into a uuid before it goes near the wire, and PostgreSQL normalises what it
accepts, so parseable spellings converge rather than diverge. The braced form `{...}` is
38 characters and asyncpg *refuses* it — `DataError: invalid UUID ... length must be
between 32..36 characters` — which is a loud failure, not a silent mismatch.

So the honest reason to declare a uuid's type is not a silent-mismatch bug. It is that
the guard below requires it, and that the declaration is what keeps the adapter correct
under a driver that does not parse for you: the psycopg path, or a future asyncpg that
stops. That is a real reason and it is smaller than the one first written here. Claiming
the larger one made this file an instance of the defect it exists to prevent — a
declaration documented as load-bearing that measurement shows is not. See the entry in
`docs/lessons.md` about `.columns(...)` being measured inert in three adapters; this is
the same mistake, in the guard for it.

Checked here rather than left to review because the class keeps coming back. The comment
at the top of `event_log_append.py` argues it in almost these words, and a later adapter
was still written without it — the knowledge existed, in a file sitting beside the new
one, and being written down was not enough. A sweep of the plan's step files then found
22 statements with the same shape waiting in ten slices that have not run yet.

An AST walk rather than a grep, so a parameter inside a comment or a docstring cannot
trip it and a statement split across concatenated string literals cannot hide from it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ADAPTERS = Path(__file__).resolve().parents[2] / "src" / "managed_agent" / "adapters"

# Parameters whose Python value is not a scalar the driver can guess at. Uuids and
# mappings are the two that actually bite: everything else here is one of those two
# wearing a domain name.
#
# And that is the standing weakness of this check: it recognises a mapping only by a
# name somebody has already added here, so a new adapter binding a `dict` under a word
# this list has not heard of passes and then fails on its first insert. `scores` and
# `regressions` were exactly that -- a mapping and a list of mappings under two domain
# words, in a statement this regex read as needing nothing. Every name added below is a
# name that got past it once; adding one is cheap and the alternative -- inferring the
# Python type of a bound value statically -- is not decidable from the source.
_NEEDS_A_TYPE = re.compile(
    r":(\w*(?:id|sid|uuid)|payload|body|template|spec|config|meta\w*"
    r"|scores|regressions|denied_paths)\b",
    re.I,
)


def _statements(module: Path) -> list[tuple[str, str]]:
    """Every assignment in `module` that builds a `sa.text(...)`, as (name, source).

    Returns the whole assignment's source rather than the `sa.text` call alone, because
    the thing under test is what is *chained onto* it — `.bindparams(...)` sits outside
    the call this walk finds.
    """
    source = module.read_text()
    tree = ast.parse(source)
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        builds_text = any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "text"
            for inner in ast.walk(node.value)
        )
        if not builds_text:
            continue
        segment = ast.get_source_segment(source, node) or ""
        name = next(
            (t.id for t in node.targets if isinstance(t, ast.Name)),
            "<unnamed>",
        )
        found.append((name, segment))
    return found


def _modules() -> list[Path]:
    return sorted(p for p in _ADAPTERS.rglob("*.py") if p.name != "__init__.py")


def test_there_are_adapter_modules_to_check() -> None:
    """The discovery found something.

    Without this the assertions below are vacuous the moment the glob stops matching —
    a rename of `adapters/`, a move, a typo in the path — and a suite that checks
    nothing is indistinguishable from a suite that finds nothing wrong. This repository
    has now shipped that mistake three times; see `docs/lessons.md`.
    """
    assert _modules(), f"no adapter modules found under {_ADAPTERS}"


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_every_statement_taking_a_uuid_or_a_mapping_declares_its_bind_types(
    module: Path,
) -> None:
    """A statement with a uuid or json parameter names that parameter's type.

    Scoped to the parameters that actually need it. A statement binding only integers
    and text needs nothing, and demanding a declaration there would be noise that gets
    the whole check switched off.
    """
    for name, segment in _statements(module):
        risky = sorted({m.lower() for m in _NEEDS_A_TYPE.findall(segment)})
        if not risky:
            continue
        assert ".bindparams(" in segment, (
            f"{module.name}: {name} binds {', '.join(risky)} and declares no types. "
            "A textual statement carries no column metadata, so asyncpg gets the bare "
            "Python object: a mapping is refused on the first insert. A uuid passed as "
            "a string is parsed by this driver and does still match — the declaration "
            "is what keeps that true under a driver that does not parse for you. Add "
            "`.bindparams(sa.bindparam('<name>', type_=sa.Uuid()))` — or `sa.JSON()` "
            "for a mapping."
        )


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_every_select_declares_the_types_of_the_columns_it_returns(
    module: Path,
) -> None:
    """A SELECT says what comes back, not just what goes in.

    The read side fails more quietly than the write side, which is why it needs its own
    assertion. Without `.columns(...)` a `jsonb` column arrives as JSON *text*, so
    `model_validate(row.body)` is handed a string where it expects a mapping — and a
    `bigint` arrives however the driver decoded it. Nothing raises at the boundary; the
    failure surfaces later, somewhere that looks unrelated.

    Only SELECTs, and only those returning a named column: `SELECT coalesce(...)` read
    through `scalar_one()` has one value of an obvious type and declaring it buys
    nothing.
    """
    for name, segment in _statements(module):
        body = segment.upper()
        if "SELECT" not in body or "INSERT" in body or "UPDATE" in body:
            continue
        # A single aggregate or expression read via scalar_one() has nothing to name.
        if not re.search(r"SELECT\s+[\w.]+\s*,", segment, re.I):
            continue
        assert ".columns(" in segment, (
            f"{module.name}: {name} selects several columns and declares none of their "
            "types. A jsonb column comes back as JSON text without this, so the caller "
            "parses a string where it expects a mapping, and the error surfaces far "
            "from here. Add `.columns(col=sa.JSON(), ...)`."
        )
