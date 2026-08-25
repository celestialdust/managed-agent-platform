"""Every coded refusal a route emits comes from the published closed set.

Tier 1 (local, no infrastructure). Realizes MAP-A51 structurally: the route tests grade
the refusals they provoke, and this grades the ones nobody thought to provoke.

ADR-013 makes the published set the API version, so a route inventing a code string is
not a typo — it is an unversioned addition to the contract that no consumer can branch
on and no release note mentions. A literal is how that happens: `core/errors.py` did not
exist until this slice, so the routes written before it each named their code in place,
and a route written after it can still do the same by hand.

Read from the source text rather than from responses, because what is being asserted is
a property of every refusal a route *can* emit. Provoking them all would mean knowing
them all, and the ones worth catching are the ones nobody listed.

Two shapes are read, because on 2026-08-24 the routes changed to the second one. A code
written as a `{"code": ...}` string is the shape a route can invent in, and one written
as `ErrorCode.MEMBER` is the shape that cannot -- mypy and the enum refuse an
unpublished member before this file would ever see it. Both are collected anyway: a
walk that only knew the first would find almost nothing today and would report that as
"every code is published" rather than as "I no longer recognise how routes refuse".

The other three checks here are structural rather than about the code strings, and
each closes a door a code has actually walked through. Ten codes were folded into the
enum on 2026-08-24, and nine of them had been reaching callers as
`HTTPException(status_code=..., detail={"code": ...})` — a status named at the call
site instead of looked up, and an exception the framework renders itself. So: a status
a route writes by hand must be one some member of the set carries, and the only
exception a route may raise at a caller is `Refusal`.

The tenth is why the third check exists, and it is the one worth reading twice. Its
route built its own `JSONResponse`, so it was never a member the enum could be missing
and no membership check could fail on it — it sat outside the published set for months
with every code test passing. What was always visible is the construction: a response
assembled at a call site around a literal `"code"` key. That is what the third check
looks for, and there are none left.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest

from managed_agent.core.errors import STATUS_FOR, ErrorCode

_API = Path(__file__).resolve().parents[2] / "src" / "managed_agent" / "control" / "api"
_PUBLISHED = {code.value for code in ErrorCode}

# There is no inventory of unfolded codes any more, and the empty space is the point.
# Six codes were recorded here as legitimately outside the enum, each with a reason, and
# every reason was a version of "minting a member belongs to whoever owns
# core/errors.py". That reason expired on 2026-08-24, when one change minted ten. What
# replaced the list is `test_no_route_builds_a_refusal_body_by_hand` below: the last
# entry, `session.turn_in_flight`, was not catchable by any check on enum membership,
# because it was never a member to be missing. It was catchable by noticing that one
# route still assembled a response body itself.


def _modules() -> list[Path]:
    return sorted(path for path in _API.rglob("*.py") if path.name != "__init__.py")


_SRC = _API.parents[2]


def _imported(path: Path) -> ModuleType:
    """Import the module living at `path`, naming it from the path itself.

    Derived rather than pasted together from a fixed prefix and `path.stem`. The route
    modules sit one directory down now, and a fixed prefix names a module that does not
    exist for every one of them -- as an ImportError, which is at least loud, but the
    same assembly would quietly import the wrong module if a name collided.
    """
    parts = path.resolve().relative_to(_SRC).with_suffix("").parts
    return importlib.import_module(".".join(parts))


def _resolve(node: ast.expr, module: ModuleType, where: str) -> str:
    """The string a `"code"` value evaluates to, or an assertion naming why it cannot.

    Names are resolved through the imported module rather than by re-reading the source,
    which is what lets a code defined in one module and used in another — `sessions.py`
    imports both of the ones it emits — be graded where it is emitted.

    Anything that cannot be resolved statically fails rather than being skipped. A code
    assembled at run time is exactly the shape that escapes a check like this one, so
    "I could not tell" has to be an answer that fails.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        value = getattr(module, node.id, None)
        assert isinstance(value, str), (
            f"{where}: the code is the name {node.id!r}, which is not a module-level "
            "string in that module, so its value cannot be checked against the set"
        )
        return value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _resolve(node.func.value, module, where)
    raise AssertionError(
        f"{where}: a refusal code is built by an expression this check cannot "
        f"evaluate ({ast.dump(node)[:120]}). A code assembled at run time cannot be "
        "held to the published set — name it as a module-level constant instead."
    )


def _coded_refusals(path: Path) -> dict[str, str]:
    """Every `{"code": ...}` in one route module, as code -> where it was written."""
    module = _imported(path)
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "code":
                where = f"{path.name}:{value.lineno}"
                found[_resolve(value, module, where)] = where
    return found


def _member_refusals(path: Path) -> dict[str, str]:
    """Every `ErrorCode.MEMBER` one module names, as its published string -> where.

    The shape routes refuse in since 2026-08-24, and the shape that cannot carry an
    invented code: `ErrorCode.NOPE` is an `AttributeError` at import and a mypy error
    before that. Collected all the same, so the walk keeps finding refusals once the
    last hand-written string is folded in — a walk that found none would report the
    published set as clean by having nothing to look at.

    Read as "names" rather than "emits", which is the honest reading of a static walk:
    a member mentioned in a module is one that module can answer with.
    """
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Attribute):
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id == "ErrorCode"):
            continue
        member = getattr(ErrorCode, node.attr, None)
        assert isinstance(member, ErrorCode), (
            f"{path.name}:{node.lineno}: ErrorCode.{node.attr} is not a member of the "
            "published set, so this module names a code no consumer can branch on"
        )
        found[member.value] = f"{path.name}:{node.lineno}"
    return found


def _refusals_in(path: Path) -> dict[str, str]:
    """Both shapes in one module: the hand-written strings and the named members."""
    return _coded_refusals(path) | _member_refusals(path)


def _every_coded_refusal() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in _modules():
        found.update(_refusals_in(path))
    return found


def _status_int(node: ast.expr, module: ModuleType, where: str) -> int | None:
    """The status one `status_code=` names, or `None` when it is the contract's lookup.

    `None` is returned for `STATUS_FOR[code]`, which is not a status a route chose --
    it is the route declining to choose, which is the shape every refusal should have.
    Anything else that cannot be reduced to an integer fails rather than being skipped,
    for the reason `_resolve` fails: a status computed at run time is exactly what a
    check like this one would otherwise wave through.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        holder = getattr(module, node.value.id, None)
        found = getattr(holder, node.attr, None)
        assert isinstance(found, int), (
            f"{where}: the status is {node.value.id}.{node.attr}, which does not "
            "resolve to an integer in that module"
        )
        return found
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        looked_up = getattr(module, node.value.id, None)
        assert looked_up is STATUS_FOR, (
            f"{where}: the status is subscripted out of {node.value.id!r}, which is "
            "not STATUS_FOR, so it is a second status table this check cannot read"
        )
        return None
    raise AssertionError(
        f"{where}: a status is built by an expression this check cannot evaluate "
        f"({ast.dump(node)[:120]}). Answer with `refuse(code, ...)` so the status "
        "comes from the published set instead of from the call site."
    )


def _hand_written_statuses(path: Path) -> dict[int, str]:
    """Every status one module writes at a call site, as status -> where.

    Both shapes: `status_code=...` passed to a decorator or a response, and
    `something.status_code = ...` assigned onto one.
    """
    module = _imported(path)
    tree = ast.parse(path.read_text())
    found: dict[int, str] = {}

    def record(value: ast.expr) -> None:
        where = f"{path.name}:{value.lineno}"
        status = _status_int(value, module, where)
        if status is not None:
            found[status] = where

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "status_code":
                    record(keyword.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "status_code":
                    record(node.value)
    return found


_RESPONSE_CONSTRUCTORS = frozenset(
    {"JSONResponse", "Response", "PlainTextResponse", "HTMLResponse", "HTTPException"}
)
"""What counts as building a response. Named rather than inferred, so the check fails
loudly on a constructor nobody listed instead of quietly ignoring it."""


def _carries_a_code_key(node: ast.expr) -> bool:
    """Whether an argument is, or contains, a dict with a literal `"code"` key.

    Nested, because the shape being caught was `detail={"code": ...}` one level in and
    `content={"error": {"code": ...}}` two levels in. A check that only looked at the
    top level would have passed the exact body it exists to fail.
    """
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "code":
                return True
            if _carries_a_code_key(value):
                return True
    return False


def _refusal_factories(path: Path) -> set[str]:
    """Functions in one module that declare they return a `Refusal`.

    `beta.py` raises `_refuse_a_shape_we_do_not_answer_in(value)` from two branches,
    which is the same refusal built once and thrown twice. Allowed by the declaration
    rather than by name: the annotation is what mypy --strict holds every `return` in
    that function to, so a factory that quietly grew a second return type stops
    qualifying here at the same moment it stops type-checking.
    """
    return {
        node.name
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef)
        and node.returns is not None
        and ast.unparse(node.returns) == "Refusal"
    }


def _raised_exceptions(path: Path) -> dict[str, str]:
    """Every exception class name one module raises, as name -> where.

    A bare `raise` is skipped: it re-raises whatever is already in flight, and whatever
    that is was graded where it was first raised.
    """
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        where = f"{path.name}:{node.lineno}"
        if isinstance(raised, ast.Name):
            found[raised.id] = where
        elif isinstance(raised, ast.Attribute):
            found[raised.attr] = where
        else:
            raise AssertionError(
                f"{where}: a refusal is raised from an expression this check cannot "
                f"name ({ast.dump(raised)[:120]})."
            )
    return found


def _classes_defined(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ClassDef)
    }


def test_there_are_route_modules_and_coded_refusals_to_check() -> None:
    """Guard the guard, twice.

    A glob that matched nothing, or an AST walk that found no refusals, would satisfy
    every assertion below by having nothing to assert about — and this file has no other
    way to tell "all the codes are published" from "there are no codes".
    """
    assert _modules(), f"no route modules found under {_API}"
    assert _every_coded_refusal(), (
        "no coded refusal was found in any route module, so the membership test below "
        "passes vacuously. Either the routes stopped refusing anything, or the shape "
        "they refuse with changed and this walk no longer recognises it."
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_route_invents_a_code_outside_the_published_set(path: Path) -> None:
    """A route may emit a member of `ErrorCode`, and nothing else.

    There is no recorded-exception clause any more. It read `code not in
    _NOT_YET_FOLDED`, and once that list emptied the clause was inert text that still
    described a way to pass — which is the shape of an allowance somebody reinstates by
    adding one line and a comment. Folding the member in is the same amount of work and
    leaves the set closed.
    """
    unpublished = {
        code: where
        for code, where in _refusals_in(path).items()
        if code not in _PUBLISHED
    }

    assert unpublished == {}, (
        f"{path.name} emits a code that is not in the published closed set: "
        f"{unpublished}. ADR-013 makes the set the API version, so a code invented in "
        "a route is an unversioned addition to the contract no consumer can branch on "
        "and no release note mentions. Add a member to ErrorCode in core/errors.py — "
        "one member, one `_status` arm and one `_public_type` arm."
    )


def test_no_route_builds_a_refusal_body_by_hand() -> None:
    """No response in this package is constructed around a literal `"code"` key.

    This is the check the inventory turned into, and it catches a class of defect no
    check on enum membership can. `session.turn_in_flight` sat outside the published set
    for as long as it did because the question "is every emitted code a member?" cannot
    fail on a code that was never a member: the route assembled its own body, named its
    own status, and the enum had nothing to say about either. What was visible the whole
    time is the thing being asserted here — one route still writing an envelope itself.

    So the property is about the shape of the construction rather than the value in it.
    A body built at a call site is a second envelope on the same API, and it goes wrong
    in the ways a duplicate always does: it kept `{code, message, detail}` for months
    after the published envelope became `{type, error, request_id}`, and it carried no
    request id at all, so the one refusal a caller could not report was the one they
    were most likely to hit twice.

    `refuse()` and the two enveloped handlers pass, because their `content=` is a
    `public_envelope(...)` dumped — the code goes in through the contract, and the
    status and the published class are looked up from it rather than typed in beside it.
    """
    responses, dicts, hand_built = 0, 0, {}
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Dict):
                dicts += 1
            if not isinstance(node, ast.Call):
                continue
            built = node.func.attr if isinstance(node.func, ast.Attribute) else None
            built = built or getattr(node.func, "id", None)
            if built not in _RESPONSE_CONSTRUCTORS:
                continue
            responses += 1
            for keyword in node.keywords:
                if _carries_a_code_key(keyword.value):
                    hand_built[f"{path.name}:{node.lineno}"] = built

    assert responses and dicts, (
        f"the walk found {responses} response constructions and {dicts} dict literals "
        "in this package, so it is not looking at the shape it is written to catch. "
        "Either responses are built some other way now, or this walk is broken."
    )

    assert hand_built == {}, (
        f"these construct a response around a literal `code` key: {hand_built}. That "
        "is a second envelope on the same API — it will not carry a request id, and it "
        "drifts from the published shape the moment the published shape changes, which "
        "it has. Return `refuse(code, ...)`, or raise `Refusal(code, ...)`."
    )


def test_the_codes_that_are_published_are_used_by_their_published_spelling() -> None:
    """A route emitting a published code spells it exactly as the enum does.

    It did that by agreeing with the enum's string until 2026-08-24, when the routes
    moved to naming the member. So the check is now the stronger one the move made
    available: a published code appears at a call site as `ErrorCode.MEMBER` and never
    as its string, because a string is a second spelling of a committed contract and
    nothing keeps the two in step. Reword the enum and the route keeps answering the
    old code, silently, and every consumer branching on it is right about a value the
    set no longer publishes.

    Every code in this package is published now, so there is no code this may not
    catch. `session.turn_in_flight` was the last one written as a string, and it reads
    `ErrorCode.SESSION_TURN_IN_FLIGHT.value` — a member named once and spelled nowhere.
    """
    emitted = _every_coded_refusal()
    published = {code: where for code, where in emitted.items() if code in _PUBLISHED}

    assert published, (
        "no route emits a code from the published set, so this test asserts nothing. "
        f"Codes found: {sorted(emitted)}"
    )
    for code in published:
        assert ErrorCode(code).value == code

    as_string = {
        code: where
        for path in _modules()
        for code, where in _coded_refusals(path).items()
        if code in _PUBLISHED
    }

    assert as_string == {}, (
        f"these published codes are hand-written as strings rather than named as "
        f"members of the set: {as_string}. The two spellings agree today and nothing "
        "keeps them agreeing — reword the enum and the route goes on answering the old "
        "string. Write `ErrorCode.MEMBER`, which cannot drift and cannot be invented."
    )


def test_no_route_answers_with_a_status_no_published_code_carries() -> None:
    """A refusal status a route writes by hand is one some member of the set carries.

    The eight codes folded in on 2026-08-24 travelled as `HTTPException(status_code=...,
    detail={"code": ...})`, and the status half of that is the half no code check would
    have caught: the code string was invented *and* the status was chosen at the call
    site, so the published table could say 400 for a family while a route answered 422
    for a member of it. A consumer reading the documented status table would be wrong
    about a live route, and nothing in this repository disagreed with either of them.

    Success statuses are ignored, because they are not this set's business. A 201 is an
    answer, and the closed set names refusals.

    What passes is a route that does not choose at all: `refuse(code, ...)` looks the
    status up out of `STATUS_FOR`, and `_status_int` returns `None` for that lookup
    rather than a number, so the ideal shape contributes nothing to check.
    """
    written: dict[int, str] = {}
    for path in _modules():
        written.update(_hand_written_statuses(path))

    assert written, (
        "no status literal was found in any route module, so this test asserts "
        "nothing. The shape routes declare a status with has changed and this walk no "
        "longer recognises it."
    )

    published = set(STATUS_FOR.values())
    unpublished = {
        status: where
        for status, where in written.items()
        if status >= 400 and status not in published
    }

    assert unpublished == {}, (
        f"these refusal statuses are written at a call site and no code in the "
        f"published set carries them: {unpublished}. A caller reading the published "
        "status table would be wrong about this route. Answer with `refuse(code, ...)` "
        "so the status is looked up, or give the code a member whose `_status` arm "
        "returns it."
    )


def test_a_route_refuses_by_raising_only_the_enveloped_refusal() -> None:
    """`Refusal` is the one exception a route may raise at a caller.

    This is the other door the eight codes escaped through, and the one no check on
    code strings can close. `HTTPException` is rendered by FastAPI itself, into
    `{"detail": ...}` — so a dependency raising one produced a body no handler here
    wrote, at a status no table here assigned, and the refusal a new integrator meets
    first (a missing tenant header) was the one shaped least like every documented one.

    `Refusal` carries an `ErrorCode` and nothing else, which is what makes it
    equivalent to `refuse()`: the status and the published class are looked up from the
    code on both paths, so a dependency and a route refusing for the same reason cannot
    answer differently.

    An exception a module both raises and defines is allowed, and is not an exception to
    the rule. `InvalidCursor` never leaves the module that raises it — it is caught
    there and turned into a `Refusal` — so it is a signal between two functions rather
    than something a caller can receive.
    """
    checked = 0
    for path in _modules():
        allowed = {"Refusal"} | _classes_defined(path) | _refusal_factories(path)
        for name, where in _raised_exceptions(path).items():
            checked += 1
            assert name in allowed, (
                f"{where}: raises {name}, which is not `Refusal`, not a function "
                f"declared to return one, and not a class {path.name} defines and "
                "catches itself. An exception this module does not handle is rendered "
                "by the framework, at a status and in a body the published set never "
                "agreed to — which is how nine codes reached callers from outside the "
                "enum. Raise `Refusal(code, ...)`."
            )

    assert checked, (
        "no route module raises anything, so this test asserts nothing. Either the "
        "dependencies stopped refusing, or `raise` is no longer how they do it."
    )
