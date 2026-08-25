"""The one way a route in this package learns who the tenant is.

The function under test is a placeholder that trusts an unauthenticated header, so the
valuable test here is not "does it parse a uuid" -- it is the structural one: **no route
obtains a tenant any other way.** That is what makes replacing it a deletion. Remove the
function and every call site becomes an import error; leave a second path in place and
one route keeps trusting a header after the authenticated claim arrives, silently, and
nothing in the diff of that later slice would show it.

Read from the source text rather than by importing, because what is being asserted is a
property of what is *written* in this package -- a route that built a `TenantId` from a
query parameter would import and run perfectly.
"""

from __future__ import annotations

import ast
import uuid
from pathlib import Path

import pytest
from starlette.datastructures import Headers
from starlette.requests import Request

from managed_agent.control.api.refusals import Refusal
from managed_agent.control.api.request.tenancy import (
    TENANT_HEADER,
    unauthenticated_tenant_from_header,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode

_API = Path(__file__).resolve().parents[2] / "src" / "managed_agent" / "control" / "api"
_TENANCY = "tenancy.py"
_DEPENDENCY = "unauthenticated_tenant_from_header"
_HTTP_METHODS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
)
"""Every HTTP verb FastAPI gives a router a decorator for.

All eight, not the five anybody writes by hand. An earlier version of this set held
`{get, post, put, patch, delete}`, and a `@router.head(...)` route appended to
`sessions.py` was measured answering **200 with no tenant header** while
`tests/control/` reported 607 passed. A verb missing from this set is a verb the scan
below cannot see, and there is no cost to naming all of them.
"""

_REVIEWER_GATE = "platform_reviewer_of"
_REVIEWER_GATED_MODULES = frozenset({"audit.py", "capacity.py"})
"""The modules allowed to gate on the reviewer instead of on a tenant, by name.

`platform_reviewer_of` is a real gate, but it is the *wrong* gate for a tenant surface:
it authorises a platform reader who holds no tenant, so a route serving tenant data
behind it is authenticated and unscoped. Nothing establishes the reviewer claim today —
`audit.py` answers 401 to everything — so this is inert now and would not be the day an
authenticator lands. Pinned by name rather than inferred, so adding a second module here
is a deliberate edit somebody has to justify, and `grep` finds every one.

`capacity.py`, the justification. `GET /v1/capacity` publishes how much work is waiting
for a pod and how much room the cluster has left. Every number on it is a **fleet**
aggregate: no field is scoped to a tenant, none is derived from one tenant's Sessions,
and there is no tenant this surface could be scoped BY without inventing one -- an
invented tenant is a filter that always agrees with itself. So the tenant gate is not a
stricter option here, it is a meaningless one.

The direction of the risk also runs the other way round from a tenant surface's. What a
tenant must not learn from these numbers is the platform's shape -- how many other
Sessions hold pods, how close the cluster is to refusing, whether now is a good moment
to submit -- so this route is dangerous when it is UNDER-gated, and the reviewer gate is
the strictest one this codebase has. The tenant-facing half of the same capacity work is
carried on the Session's own event stream instead, which is gated on the tenant exactly
as every other Session read is.
"""

_GATES = frozenset({_DEPENDENCY, _REVIEWER_GATE})

_UNGATED_PROBE_MODULES = frozenset({"health.py"})
"""The modules whose routes may take no principal at all, by name.

A kubelet presents no tenant and no credential. Gating the liveness and readiness probe
would therefore mean a long-lived credential in the cluster for the sake of a health
check, so `health.py` answers without one -- and the manifest's two `httpGet` probes are
the callers that need it to.

Pinned by name for `_REVIEWER_GATED_MODULES`'s reason: a second entry is a deliberate
edit somebody has to justify and `grep` finds every one. And narrowed rather than merely
allowed -- `test_a_module_exempt_from_the_gate_reaches_nothing_it_could_leak` below
asserts an exempt module never names `platform`, so the exemption cannot be taken by a
route that reads or writes anything. That is the structural half; the behavioural half
is `tests/control/test_health_is_reachable_without_a_tenant.py`, which drives the route
over a Platform whose every port raises on access.
"""


_LOG_READ = "platform.event_log_range"
_CROSS_TENANT_READ = "read_span_of_any_session"
_OWNERSHIP_CHECK = "session_registry.fetch"


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/agents",
            "headers": Headers(headers).raw,
        }
    )


def _route_files() -> list[Path]:
    """Every module in the package that could hold a route, at any depth.

    `rglob`, not `glob`. With the non-recursive form a module one directory down —
    `control/api/admin/routes.py` — was invisible to **all six** structural checks in
    this file, the two vacuity guards included, and a route in it answered 200 with no
    tenant header while `tests/control/` reported 607 passed. That was the widest hole
    of the seven found, because it defeats the checks that were here before the route
    scan was added as well as the scan itself.
    """
    return sorted(
        path
        for path in _API.rglob("*.py")
        if path.name not in {"__init__.py", _TENANCY}
        and "__pycache__" not in path.parts
    )


def test_there_are_route_files_to_check() -> None:
    """Guard the guard: a glob that matched nothing would pass every test below."""
    assert _route_files(), f"no route modules found under {_API}"


def test_at_least_one_route_actually_takes_its_tenant_from_the_dependency() -> None:
    """The positive half, and without it every test below can pass vacuously.

    The three structural tests are all negative -- no route builds its own `TenantId`,
    no route names the header. The cheapest way to satisfy a negative assertion is for
    nothing to do the thing at all, so a package where **no** route uses tenancy passes
    all three while providing no tenant scoping whatever. Verified: removing the
    dependency from both routes left every other test in this file green.

    It looks for the name inside `Depends(...)` rather than anywhere in the file, which
    matters for the same reason. A leftover import satisfies "the name appears here"
    while the route it was imported for no longer asks for a tenant -- and an unused
    import is exactly what a half-finished edit leaves behind.
    """
    wired = [
        path.name
        for path in _route_files()
        if f"Depends({_DEPENDENCY})" in path.read_text()
    ]

    assert wired, (
        f"no route module wires {_DEPENDENCY} as a dependency. Either tenant scoping "
        "was removed from every route, or a route acquired a tenant some other way -- "
        "and the negative assertions in this file cannot tell either apart from a "
        f"package that never needed a tenant. Modules checked: "
        f"{[p.name for p in _route_files()]}"
    )


@pytest.mark.parametrize("path", _route_files(), ids=lambda p: p.name)
def test_no_route_file_constructs_a_tenant_id_of_its_own(path: Path) -> None:
    """`TenantId(...)` appears in `tenancy.py` and nowhere else in this package.

    Constructing one elsewhere means a second place decides who the caller is, and the
    two are then free to disagree -- which is the shape of a cross-tenant read that no
    test is looking for.
    """
    source = path.read_text()

    assert "TenantId(" not in source, (
        f"{path.name} builds a TenantId itself; every route must take its tenant from "
        f"unauthenticated_tenant_from_header in {_TENANCY}, so that replacing the "
        "placeholder is one deletion rather than a search"
    )


@pytest.mark.parametrize("path", _route_files(), ids=lambda p: p.name)
def test_no_route_file_reads_the_tenant_header_itself(path: Path) -> None:
    """The header name appears in one module, so renaming or retiring it is one edit."""
    source = path.read_text()

    assert TENANT_HEADER not in source, (
        f"{path.name} names the {TENANT_HEADER} header directly; it should depend on "
        f"{_TENANCY} instead"
    )


@pytest.mark.parametrize("path", _route_files(), ids=lambda p: p.name)
def test_a_route_that_uses_a_tenant_takes_it_from_the_one_dependency(
    path: Path,
) -> None:
    """Mentioning a tenant at all obliges a route to import the shared source."""
    source = path.read_text()
    if "TenantId" not in source:
        return

    assert _DEPENDENCY in source, (
        f"{path.name} refers to TenantId without importing "
        "unauthenticated_tenant_from_header, so it has some other way of getting one"
    )


def _called_name(call: ast.Call) -> str | None:
    """The bare name of whatever a call expression calls.

    `f()` gives `f`, `mod.f()` gives `f`. Used wherever a check cares which function is
    being called and not how the caller happened to import it.
    """
    called = call.func
    if isinstance(called, ast.Name):
        return called.id
    if isinstance(called, ast.Attribute):
        return called.attr
    return None


def _depends_names(node: ast.AST) -> set[str]:
    """Every name passed to a `Depends(...)` anywhere under `node`.

    Read from the syntax tree and not from the source text, because the text is where
    this check would go wrong in the direction that matters. `_DEPENDENCY` appears in
    every one of these modules -- in the import, and in the sibling routes that do take
    it -- so "the name occurs in this file" is true of a file containing an ungated
    route. The tree is what distinguishes *which function* asks for it.
    """
    found: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        called = sub.func
        if not isinstance(called, ast.Name) or called.id != "Depends":
            continue
        # Positional and keyword, `name` and `module.name`. All four forms gate
        # identically at runtime, and an earlier version read only positional
        # `ast.Name` -- so `Depends(dependency=...)` and
        # `Depends(tenancy.unauthenticated_tenant_from_header)` were both reported as
        # ungated. A guard that fails correct code is a guard somebody deletes, which
        # costs more than the two lines it takes to accept them.
        passed = list(sub.args) + [kw.value for kw in sub.keywords]
        for arg in passed:
            if isinstance(arg, ast.Name):
                found.add(arg.id)
            elif isinstance(arg, ast.Attribute):
                found.add(arg.attr)
    return found


def _signature_depends(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """The dependencies a route function asks for **in its own signature**.

    Only the signature, because that is the only place FastAPI looks: it resolves
    dependencies from the parameter list, whether written as a default
    (`t: TenantId = Depends(f)`) or inside `Annotated[TenantId, Depends(f)]`. Both live
    under `function.args`, so walking that node covers the two forms and nothing else.

    Walking the whole function body instead was measured passing a route whose only
    `Depends(...)` sat in a nested function that is never called -- FastAPI resolves
    nothing there, so the route answered **200 with no tenant header** while the check
    read the name and called it gated. A gate that any mention of the right identifier
    satisfies is a gate satisfied by a comment.
    """
    return _depends_names(function.args)


def _router_level_gates(tree: ast.Module) -> set[str]:
    """The gates an `APIRouter(dependencies=[...])` applies to every route on it.

    `audit.py` does it this way and its docstring says why: the principal sits on the
    router, so a route added later is gated by construction rather than by whoever adds
    it remembering. A module gated here needs nothing on its individual routes.
    """
    routers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) == "APIRouter"
    ]
    if not routers:
        return set()
    # **Every** router, intersected -- not the first one carrying a `dependencies=`.
    # Returning on the first match exempted the whole module, so a module holding a
    # gated `APIRouter` beside a bare one had every route skipped unchecked, and a
    # route on the bare router answered 200 with no tenant header. One gated router is
    # not a statement about a second.
    per_router = []
    for node in routers:
        gates: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg == "dependencies":
                gates = _depends_names(keyword.value)
        per_router.append(gates)
    common = set(per_router[0])
    for gates in per_router[1:]:
        common &= gates
    return common


def _route_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every function carrying an `@router.<http method>(...)` decorator."""
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            call = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(call, ast.Attribute) and call.attr in _HTTP_METHODS:
                out.append(node)
                break
    return out


def _modules_declaring_a_router() -> list[Path]:
    return [path for path in _route_files() if "APIRouter(" in path.read_text()]


def test_every_module_with_a_router_has_at_least_one_route_the_scan_can_see() -> None:
    """Guard the guard, and this one has a specific failure it is guarding against.

    The check below is a negative: it passes when it finds no ungated route. A decorator
    scan that silently stops matching -- because a module starts mounting routes some
    other way, or the decorator is renamed -- finds no routes at all and therefore
    reports nothing ungated. That reads exactly like a clean package.
    """
    blind = [
        path.name
        for path in _modules_declaring_a_router()
        if not _route_functions(ast.parse(path.read_text()))
    ]

    assert not blind, (
        f"{blind} declare a router and the decorator scan sees no route on it, so "
        "every assertion about their routes below is vacuous"
    )


def test_every_route_is_gated_at_the_route_or_at_its_router() -> None:
    """No operation on this app is reachable without a principal.

    This is the check that makes the next route safe rather than the current ones. Every
    other structural test in this file reads a *file*: whether the module builds its own
    `TenantId`, names the header, or mentions a tenant without importing the dependency.
    All three pass for a module holding one gated route and one ungated one, because the
    gated route satisfies them on the ungated route's behalf.

    Measured, on this tree, before the check existed: appending a route to `sessions.py`
    that took no tenant dependency at all left **605 of 605** tests under
    `tests/control/` passing, this file's sixty included. The app has no
    application-wide authentication -- scoping is declared per route -- so an operation
    that forgets to declare it is not refused, it is open, and the failure is silent in
    both directions: nothing at import time, nothing at request time, and no reviewer
    sees an absence.
    """
    ungated: list[str] = []
    for path in _route_files():
        if path.name in _UNGATED_PROBE_MODULES:
            continue
        tree = ast.parse(path.read_text())
        router_gates = _router_level_gates(tree)
        if router_gates & _GATES:
            continue
        for function in _route_functions(tree):
            if not _signature_depends(function) & _GATES:
                ungated.append(f"{path.name}::{function.name}")

    assert not ungated, (
        f"these routes are reachable with no principal: {ungated}. Declare "
        f"{_DEPENDENCY} on the route, or -- better, and the way `audit.py` does it -- "
        "put the gate on the APIRouter so every route added later inherits it."
    )


def _names_the_platform(path: Path) -> bool:
    """Whether this module's *code* reaches the wired Platform.

    Read off identifiers rather than out of the source text, because the source text of
    a module that touches nothing still says the word: `health.py`'s docstring explains
    what the platform's three services probe at. A substring search reads that as a
    leak.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "platform":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "platform":
            return True
        if isinstance(node, ast.arg) and node.arg == "platform":
            return True
        if isinstance(node, ast.ImportFrom) and "composition" in (node.module or ""):
            return True
    return False


def test_a_module_exempt_from_the_gate_reaches_nothing_it_could_leak() -> None:
    """An unauthenticated route may exist; one that reads may not.

    This is what keeps `_UNGATED_PROBE_MODULES` from being an exemption list in the
    sense the file warns about elsewhere. A named module is not trusted -- it is held to
    a narrower property than the gated ones: it may not name `platform` at all, so it
    cannot reach a registry, a store or the Event Log whether or not it holds a tenant.
    A probe that grew a database read would fail here rather than serve one tenant's
    data to an unauthenticated caller.

    The set is checked against the tree as well, so a module deleted or renamed leaves a
    stale name behind loudly instead of silently exempting nothing.
    """
    present = {path.name for path in _route_files()}
    assert present >= _UNGATED_PROBE_MODULES, (
        f"{sorted(_UNGATED_PROBE_MODULES - present)} is exempted from the gate and is "
        "not a route module in this package any more"
    )

    reaching = [
        path.name
        for path in _route_files()
        if path.name in _UNGATED_PROBE_MODULES and _names_the_platform(path)
    ]
    assert not reaching, (
        f"{reaching} take no principal and reach `platform`, so an unauthenticated "
        "caller reaches something. Either gate the route or stop reading."
    )


_UNSCANNABLE_REGISTRATION = ("api_route", "add_api_route", "add_route", "websocket")


def test_no_route_is_registered_by_a_form_the_scan_cannot_read() -> None:
    """Forbid the spellings, rather than chase them.

    The gate check below reads `@router.<verb>(...)` decorators. FastAPI offers three
    other ways to mount an operation — `@router.api_route(..., methods=[...])`,
    `router.add_api_route(path, fn, methods=[...])`, and `@router.websocket(...)` — and
    all three were measured mounting a route that answered **200 with no tenant header**
    while `tests/control/` reported 607 passed. So was a decorator behind an alias:
    `_alias = router.get` then `@_alias("/x")`.

    Enumerating spellings is a losing game — the scan can only ever see the forms
    somebody thought of, and each new one is silent. Forbidding them inverts it: the
    package may use exactly the form the scan reads, and anything else fails **here**,
    loudly, with the reason. That is the difference between a check that is complete
    today and one that stays complete.

    If a future route genuinely needs `api_route`, this test is the place to notice, and
    the fix is to teach the scan that form and then narrow this list — not to delete the
    line.
    """
    offenders: list[str] = []
    for path in _route_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) in (
                _UNSCANNABLE_REGISTRATION
            ):
                offenders.append(f"{path.name}::{_called_name(node)}")
            # `_alias = router.get` — an assignment whose value is an attribute of
            # something router-shaped. The decorator then reads `@_alias(...)`, which
            # carries none of the verb the scan matches on.
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in _HTTP_METHODS
            ):
                offenders.append(f"{path.name}::alias of .{node.value.attr}")

    assert not offenders, (
        f"these register or alias a route in a form the gate scan cannot read: "
        f"{offenders}. Every operation in this package must be mounted as "
        "`@router.<verb>(...)`, because that is the only form the scan sees -- one "
        "mounted another way is unauthenticated and no test says so."
    )


def test_a_module_gates_on_the_reviewer_only_if_it_is_the_audit_surface() -> None:
    """`platform_reviewer_of` authorises a reader who holds **no tenant**.

    That is right for the audit surface, whose whole purpose is reading across tenants,
    and wrong for anything serving one tenant's data: such a route would be
    authenticated and unscoped, which is the more dangerous of the two failures because
    it looks gated. Inert today — nothing establishes the reviewer claim, so `audit.py`
    401s everything — and not inert the day an authenticator lands.
    """
    reviewer_gated = [
        path.name
        for path in _route_files()
        if _REVIEWER_GATE in _router_level_gates(ast.parse(path.read_text()))
    ]

    assert set(reviewer_gated) <= _REVIEWER_GATED_MODULES, (
        f"{sorted(set(reviewer_gated) - _REVIEWER_GATED_MODULES)} gate on "
        f"{_REVIEWER_GATE}, which authorises a platform reader holding no tenant. If "
        "the module really reads across tenants, add it to _REVIEWER_GATED_MODULES "
        "with the reason; if it serves one tenant's data, it needs the tenant gate."
    )


def _routes_reading_the_event_log() -> list[str]:
    """Route functions that read the Event Log without asking who owns the Session."""
    unscoped: list[str] = []
    for path in _route_files():
        tree = ast.parse(path.read_text())
        # A reviewer-gated module reads across tenants by design, and there is no owner
        # to check against: the caller holds no tenant, so scoping the read to "their"
        # Sessions is not a weaker check, it is a meaningless one. Which modules may
        # claim this is pinned in `_REVIEWER_GATED_MODULES` and asserted by its own test
        # above, so this exemption cannot be taken by adding a dependency -- a module
        # has to be named, in a commit somebody signs.
        if _REVIEWER_GATE in _router_level_gates(tree):
            continue
        for function in _route_functions(tree):
            body = ast.unparse(function)
            reads = _LOG_READ in body or _CROSS_TENANT_READ in body
            if reads and _OWNERSHIP_CHECK not in body:
                unscoped.append(f"{path.name}::{function.name}")
    return unscoped


def test_a_route_reading_the_event_log_asks_who_owns_the_session_route_by_route() -> (
    None
):
    """The same granularity defect as the gate check had, for scoping instead.

    A per-**file** version of this already existed below and still does, and it is the
    one this replaces in strength: appending to `events.py` a route that takes the
    tenant dependency, binds it, and then calls `read_span_of_any_session` with no
    `session_registry.fetch` left `607 passed` — because the sibling `read_events`
    satisfied the file-level assertion on the new route's behalf. The result is an
    unauthenticated-in-effect cross-tenant Event Log read for anybody holding a Session
    id.

    Taking the tenant and *ignoring* it is the failure this catches, which is why it is
    separate from the gate check: the gate proves a route asks who the caller is, and
    nothing there proves it uses the answer. The Event Log carries no tenant, so there
    is nothing an absent ownership check can fail against — the read simply succeeds.

    `read_span_of_any_session` is matched as well as the port, because it is the
    deliberately tenant-blind seam and a route reaching it directly is doing the same
    thing by another name.
    """
    unscoped = _routes_reading_the_event_log()

    assert not unscoped, (
        f"these routes read the Event Log and never call {_OWNERSHIP_CHECK}: "
        f"{unscoped}. The log is keyed by Session and holds no tenant, so the read "
        "succeeds and hands back another tenant's events."
    )


def test_a_well_formed_header_parses_into_a_tenant_id() -> None:
    tenant = uuid.uuid4()

    assert (
        unauthenticated_tenant_from_header(_request({TENANT_HEADER: str(tenant)}))
        == tenant
    )


def test_an_absent_header_is_refused_rather_than_defaulted() -> None:
    """No fallback tenant, by design.

    A default is invisible when real multi-tenancy arrives: every call site keeps
    working and serves one tenant's data as another's. A refusal fails loudly while it
    is still cheap to fix.
    """
    with pytest.raises(Refusal) as raised:
        unauthenticated_tenant_from_header(_request({}))

    assert STATUS_FOR[raised.value.code] == 400
    assert raised.value.code == ErrorCode.REQUEST_TENANT_MISSING


@pytest.mark.parametrize("value", ["", "not-a-uuid", "12345", "0" * 32 + "extra", "  "])
def test_a_header_that_is_not_a_uuid_is_refused(value: str) -> None:
    with pytest.raises(Refusal) as raised:
        unauthenticated_tenant_from_header(_request({TENANT_HEADER: value}))

    assert STATUS_FOR[raised.value.code] == 400
    assert raised.value.code == ErrorCode.REQUEST_TENANT_MALFORMED


def test_the_placeholder_says_in_its_own_docstring_that_it_is_one() -> None:
    """The name and the docstring are the only warning a reader gets.

    Asserted rather than trusted, because this function returns a value that looks
    exactly like an authenticated one and the whole risk is a reader assuming it is.
    """
    doc = unauthenticated_tenant_from_header.__doc__ or ""
    module_doc = (
        __import__("managed_agent.control.api.request.tenancy", fromlist=["x"]).__doc__
        or ""
    )

    assert "refuse" in doc.lower()
    assert "placeholder" in module_doc.lower(), (
        "the module does not say it is a placeholder, so nothing warns a reader that "
        "the tenant is unauthenticated"
    )
    assert "trusts the caller" in module_doc.lower()


def _modules_that_read_the_event_log() -> list[Path]:
    return [path for path in _route_files() if _LOG_READ in path.read_text()]


def test_at_least_one_route_reads_the_event_log_directly() -> None:
    """Guard the guard: with no module matching, the check below asserts nothing."""
    assert _modules_that_read_the_event_log(), (
        f"no route module names {_LOG_READ}, so the scoping check below is vacuous. "
        "Either every log read moved behind a helper, or the attribute was renamed."
    )


@pytest.mark.parametrize(
    "path", _modules_that_read_the_event_log(), ids=lambda p: p.name
)
def test_a_route_that_reads_the_event_log_also_asks_who_owns_the_session(
    path: Path,
) -> None:
    """The Event Log carries no tenant, so a read beside no lookup is unscoped.

    This is the half the tests above cannot reach. They prove a route *obtains* a
    tenant; nothing proved it *used* one, and the difference is a route that takes the
    dependency, ignores it, and serves any tenant's Session to whoever knows its uuid.
    That hole has been written into a plan three times -- `events.py` (MAP-7),
    `stream.py` (MAP-9) and `turns.py` (MAP-12) -- and each time every behavioural test
    in the slice's own tactic passed against the unscoped version, because they all
    address a Session the caller had just created.

    A route that must read across tenants goes through
    `events.read_span_of_any_session`, which is deliberately tenant-blind and says so;
    it therefore does not name the port itself and is not matched here. That is the
    seam, and moving a genuinely cross-tenant read behind it is the intended way past
    this check -- not an exemption list, which would grow.
    """
    assert _OWNERSHIP_CHECK in path.read_text(), (
        f"{path.name} reads {_LOG_READ} and never calls {_OWNERSHIP_CHECK}. The Event "
        "Log is keyed by Session and holds no tenant, so there is nothing for an "
        "absent ownership check to fail against -- the read simply succeeds and hands "
        "back another tenant's events."
    )
