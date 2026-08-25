"""What a platform reviewer reads, and what neither principal buys on the other side.

Tier 1. The structural half runs against in-memory ports; the behavioural half runs
against the real wired platform on PostgreSQL 17, because the claim under test is that a
reviewer reads *another tenant's* Session and only a real Session registry can make the
tenant boundary it crosses a real one. Realizes MAP-A42 (every Session's record is
available in the same shape, without asking any tenant for anything) and MAP-A43 (the
record is available to a reviewer holding no tenant credential, and nothing they read
lets them act as a tenant).

Three claims, and each needs a test that fails if it is violated:

**Authorized differently.** Both crossings are exercised on one app over one log. A
tenant credential answers 401 on the audit path while it answers 200 on the tenant path
for the same Session; a reviewer principal answers 400 for a missing tenant on the
tenant path while it answers 200 on the audit path. Neither principal is the other with
a setting changed, and no single request satisfies both surfaces.

**Holding no tenant credential.** Structural, because a runtime assertion cannot prove
an absence: the audit modules name no tenant type, no tenant dependency and no Session
registry, so there is nothing on that path to obtain a credential *with*. The runtime
half is the corroboration — a valid tenant header changes neither the authorization nor
a single byte of what comes back — and the successful read is served by a platform whose
Session registry raises on contact.

**The same page.** For one Session and one range the two surfaces must return the same
status and the same bytes across all three answers a range can have. The table below
reaches a span, an empty page above the head, a refusal below the retained floor, a
backwards range and an over-wide one.

The reviewer principal is established by the shipped authenticator, from a token minted
here with the key the platform is wired with — so every case below travels the path a
deployment travels, and the eleven-row refusal table grades the real door rather than a
stand-in for it. What is still faked is the *tenant* claim, because nothing
authenticates a tenant yet; that half comes from a header that exists only here.
"""

from __future__ import annotations

import ast
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import sqlalchemy as sa
from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute, APIRouter
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request import reviewer_auth
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import audit, events
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.reviewers.audit_reader import (
    AUDIT_PRINCIPAL_UNRESOLVED,
    REVIEWER_CLAIM,
    TENANT_CLAIM,
    AuditPrincipalRefused,
    PlatformReviewer,
    resolve_reviewer,
)
from managed_agent.control.reviewers.token import (
    REVIEWER_AUDIENCE,
    HmacReviewerTokens,
    mint_reviewer_token,
)
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import (
    FIRST_SEQ,
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    new_session_id,
)
from managed_agent.core.ports import EventRecord, Resolution, SessionListing
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord, SessionState
from managed_agent.core.session.session_token import mint_session_token

_SRC = Path(__file__).resolve().parents[2] / "src" / "managed_agent"
_API = _SRC / "control" / "api"
_AUDIT_PATH = "/v1/audit/sessions/{session_id}/events"
_TENANT_PATH = "/v1/sessions/{session_id}/events"

_TENANT_CLAIM_TEST_HEADER = "X-Test-Tenant-Claim"
"""A header no production code reads.

It stands in for the authenticator that will one day establish a *tenant* claim; nothing
does today, and the tenant surface parses an unauthenticated header instead. Named with
`Test` in the middle so that grepping the tree for it finds only this file, and so a
reader who meets it in a trace knows it came from here and not from a deployment.

There is no longer a sibling for the reviewer claim. A reviewer principal is established
by the real authenticator from a real signed token, so the cases below exercise the
credential a deployment would actually receive rather than a stand-in for it.
"""

_REVIEWER_KEY = b"the control plane's own signing key"
_REVIEWER_NOW = 1788664689

_REVIEWER_EXPIRY = 4102444800
"""2100-01-01, and distant on purpose.

Some cases here run against a platform wired by the composition root, which reads the
real clock, so a credential meant to be valid has to outlive the calendar rather than
this file's idea of now. An expiry a plausible hour ahead is a test that starts failing
on a date nobody chose, for a reason nobody looking at it would guess.
"""

_REVIEWER_LONG_EXPIRED = 1
"""1970, so a credential meant to be expired is expired under any clock at all."""


class _FixedClock:
    """A clock that does not advance, so nothing here turns on how long a test took."""

    def now_epoch_ms(self) -> int:
        return _REVIEWER_NOW * 1000


_SKILLS_SHA = "0" * 39 + "b"

_CREDENTIAL_WORDS = ("credential", "token", "secret", "authorization", "password")


# --------------------------------------------------------------------------------------
# The principal, parsed away from any HTTP request
# --------------------------------------------------------------------------------------


def test_a_reviewer_claim_on_its_own_resolves() -> None:
    reviewer_id = uuid.uuid4()

    assert resolve_reviewer(reviewer_id, None) == PlatformReviewer(reviewer_id)


@pytest.mark.parametrize(
    ("reviewer_claim", "tenant_claim"),
    [
        (None, None),
        (uuid.uuid4(), uuid.uuid4()),
        (None, uuid.uuid4()),
        (str(uuid.uuid4()), None),
        ("not-a-uuid", None),
        (1, None),
    ],
    ids=[
        "neither",
        "both",
        "a-tenant-alone",
        "a-reviewer-as-a-string",
        "a-reviewer-that-is-not-a-uuid",
        "a-reviewer-that-is-an-int",
    ],
)
def test_every_other_claim_shape_is_refused(
    reviewer_claim: object, tenant_claim: object
) -> None:
    """Including the string that looks like a uuid.

    Whatever eventually sets this claim is an authenticator, and an authenticator that
    hands over an unparsed string has not finished its job. Accepting one here would
    mean the audit surface doing the parse instead, on a value it has no way to judge.
    """
    with pytest.raises(AuditPrincipalRefused):
        resolve_reviewer(reviewer_claim, tenant_claim)


def test_a_tenant_alongside_a_reviewer_is_refused_rather_than_resolved_either_way() -> (
    None
):
    """The both-claims case, called out on its own because it is the dangerous one.

    Resolving it in the reviewer's favour would let an audit read happen on a request
    that also carries a tenant credential — the one thing this read is defined by not
    doing — and the page it returned would be indistinguishable from a legitimate one.
    """
    with pytest.raises(AuditPrincipalRefused):
        resolve_reviewer(uuid.uuid4(), uuid.uuid4())


def test_the_reviewer_carries_nothing_it_could_act_as_a_tenant_with() -> None:
    """One field, and it is an identity.

    Asserted over the declared fields rather than by reading the class, because "holds
    no tenant credential" stops being structural the moment something else is added
    here and starts depending on every caller unpacking it carefully.
    """
    reviewer = PlatformReviewer(uuid.uuid4())
    names = [field.name for field in fields(reviewer)]

    assert names == ["reviewer_id"]
    for word in (*_CREDENTIAL_WORDS, "tenant"):
        assert not [name for name in names if word in name.lower()], (
            f"PlatformReviewer carries a field naming a {word}; holding one would then "
            "confer some ability to act as a tenant"
        )


def test_a_reviewer_cannot_be_widened_after_it_is_handed_over() -> None:
    reviewer = PlatformReviewer(uuid.uuid4())

    with pytest.raises(FrozenInstanceError):
        reviewer.reviewer_id = uuid.uuid4()  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# What the audit modules are allowed to name
# --------------------------------------------------------------------------------------


_AUDIT_PATH_MODULES = [
    _API / "routes" / "audit.py",
    _API / "request" / "reviewer_auth.py",
    _SRC / "control" / "reviewers" / "audit_reader.py",
    _SRC / "control" / "reviewers" / "token.py",
]
"""Every module a request to the audit surface passes through.

The authenticator and the token format are on this path too, so they are held to the
same property as the two that were here first. The authenticator in particular reaches
the wired platform -- which is what a tenant credential would be obtained *from* -- so
it is the module where the omission would matter most and the one easiest to leave out.
"""


@pytest.mark.parametrize(
    "module",
    _AUDIT_PATH_MODULES,
    ids=lambda path: str(path.name),
)
@pytest.mark.parametrize(
    "forbidden",
    [
        "TenantId",
        "unauthenticated_tenant_from_header",
        TENANT_HEADER,
        "session_registry",
    ],
)
def test_the_audit_path_names_nothing_that_yields_a_tenant(
    module: Path, forbidden: str
) -> None:
    """The mechanical form of "holds no tenant credential".

    An absence cannot be proved by a request, because a request only shows what happened
    on the path it took. What can be proved is that there is nothing on this path to
    obtain a credential *with*: no tenant type is constructed, no tenant dependency is
    declared, and the Session registry — the one component that maps a Session to its
    owner — is never named. A future edit that reaches for any of them fails here before
    it can reach a reviewer.
    """
    assert forbidden not in module.read_text(), (
        f"{module.name} names {forbidden!r}. The audit surface is authorized as a "
        "reader of all tenants and must hold no tenant credential; anything here that "
        "obtains one makes that a claim about care rather than about structure"
    )


def _claim_writers() -> set[str]:
    """Every module in the package that establishes the reviewer claim on a request.

    Both spellings, because they are equivalent at runtime and a guard that read only
    one would be satisfied by the other: `setattr(state, REVIEWER_CLAIM, ...)` and a
    direct `state.platform_reviewer_id = ...`. Read from the syntax tree so that a
    docstring naming either cannot trip it.
    """
    writers: set[str] = set()
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            by_setattr = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) == 3
                and isinstance(node.args[1], ast.Name)
                and node.args[1].id == "REVIEWER_CLAIM"
            )
            by_assignment = isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == REVIEWER_CLAIM
                for target in node.targets
            )
            if by_setattr or by_assignment:
                writers.add(path.relative_to(_SRC).as_posix())
    return writers


def test_exactly_one_module_can_make_a_request_a_reviewers() -> None:
    """A second writer of the claim would be a second way to become a reviewer.

    And it would not be the one that checks a signature. The gate downstream trusts the
    claim completely -- that is what lets it stay a function of two values rather than
    of a request -- so the whole weight of "a reviewer proved who they are" rests on
    there being one place that can put the claim there.

    The positive half is asserted too: a guard that found *no* writer would pass while
    the surface was dead, which is the state this endpoint actually shipped in.
    """
    writers = _claim_writers()

    assert writers == {"control/api/request/reviewer_auth.py"}, (
        f"the reviewer claim is established in {sorted(writers)}. Exactly one module "
        "may do it, and it is the one that verifies a signature first; any other is an "
        "unsigned path to reading every tenant's history"
    )


def test_only_the_audit_surface_reaches_the_unscoped_reader_unchecked() -> None:
    """`read_span_of_any_session` applies no tenant predicate, so callers are graded.

    The Event Log is keyed by Session and carries no tenant, so that function hands back
    any Session's events to whoever names the id. Exactly one caller is allowed to reach
    it with nothing in front of it, and that caller is the surface authorized across
    tenants. Any other route module that names it must also name the registry lookup
    that establishes its caller may address the Session — which is the check MAP-7's
    plan omitted, letting anyone holding a uuid read another tenant's whole log.
    """
    reachers = {
        path
        for path in _API.rglob("*.py")
        if "read_span_of_any_session" in path.read_text()
    }

    assert reachers, (
        "no module under control/api/ names read_span_of_any_session, so this guard "
        "passes vacuously. Either it was renamed, or the shared span rule was inlined "
        "back into its callers and each of them now has a copy to keep in step."
    )
    unchecked = sorted(
        path.name
        for path in reachers
        if path.name != "audit.py" and "session_registry.fetch(" not in path.read_text()
    )
    assert unchecked == [], (
        f"{unchecked} reach the tenant-blind span reader without an ownership check. "
        "A tenant-facing route must call session_registry.fetch before it, or it "
        "returns any tenant's events to any caller who knows a Session id."
    )


def test_the_audit_router_is_read_only_and_gated_by_construction() -> None:
    """The principal is on the router, so a route added later is gated by default.

    Both halves, in order, and the order is asserted rather than the set: the first
    dependency establishes the principal and the second resolves it, so resolving first
    would refuse every request no matter what credential it carried. Either one missing
    is a hole -- without the authenticator nothing is ever established and the surface
    is dead, without the gate nothing is ever checked and it is open.
    """
    assert [dependency.dependency for dependency in audit.router.dependencies] == [
        reviewer_auth.establish_reviewer_principal,
        audit.platform_reviewer_of,
    ]
    assert audit.router.routes, "the audit router declares no routes"
    for route in audit.router.routes:
        assert isinstance(route, APIRoute)
        assert route.methods == {"GET"}, (
            f"{route.path} declares {route.methods}; a surface that reads "
            "across every tenant must open no path that changes anything"
        )


def _endpoints_of(router: APIRouter) -> list[Callable[..., object]]:
    return [route.endpoint for route in router.routes if isinstance(route, APIRoute)]


def test_the_two_surfaces_are_two_endpoints_rather_than_one_with_a_setting() -> None:
    """Two paths on one app, two handlers, and the gate on exactly one of them.

    The paths are read from the published document rather than from `app.routes`, which
    on this FastAPI holds one opaque wrapper per `include_router` call and no leaf
    routes at all — a walk over it finds nothing and every assertion about it passes
    vacuously.
    """
    paths = create_app(_platform_whose_registries_raise()).openapi()["paths"]

    assert {_AUDIT_PATH, _TENANT_PATH} <= set(paths), sorted(paths)
    assert set(paths[_AUDIT_PATH]) == {"get"}
    assert _endpoints_of(audit.router) == [audit.read_audit_events]
    assert audit.read_audit_events not in _endpoints_of(events.router)
    assert events.read_events in _endpoints_of(events.router)
    assert events.router.dependencies == [], (
        "the tenant events router has grown a router-level dependency; the audit gate "
        "is meant to be the one thing the two surfaces do not share"
    )


# --------------------------------------------------------------------------------------
# In-memory ports: what a refused request touches, and what a served one does not
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FakeRow:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


class CountingLog:
    """One contiguous in-memory Event Log per Session, which counts what reached it."""

    def __init__(self) -> None:
        self._rows: dict[SessionId, list[FakeRow]] = {}
        self.reads = 0

    def seed(self, session_id: SessionId, count: int) -> None:
        self._rows[session_id] = [
            FakeRow(session_id, Seq(n), f"event.{n}", {"n": n})
            for n in range(1, count + 1)
        ]

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        raise AssertionError("the audit surface appended to the log")

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]:
        self.reads += 1
        rows = [
            row for row in self._rows.get(session_id, []) if start <= row.seq <= end
        ]
        return rows[:limit]

    def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[EventRecord]:
        raise AssertionError("the audit surface followed the log")

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ


class RegistryThatRefusesContact:
    """Satisfies the definition-registry port; a call is the failure being tested."""

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        raise AssertionError("the audit surface registered a definition")

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        raise AssertionError("the audit surface resolved a definition")

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        raise AssertionError("the audit surface listed a definition's versions")

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        raise AssertionError("the audit surface read one definition revision")

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("the audit surface archived a definition revision")


class ToolRegistryThatRefusesContact:
    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("the audit surface registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("the audit surface looked up a tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("the audit surface listed a tenant's tools")


class SessionRegistryThatRefusesContact:
    """Every method raises, which is the assertion.

    The Session registry is the only component that maps a Session to the tenant that
    owns it. A read served while this is wired in is a read that consulted no owner and
    therefore had no owner to be scoped to — the cross-tenant property, asserted from
    the inside rather than inferred from a status code.
    """

    async def create(self, record: SessionRecord) -> None:
        raise AssertionError("the audit surface wrote a Session registry row")

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        raise AssertionError(
            "the audit surface asked the Session registry who owns this Session, which "
            "it can only do by holding a tenant to ask about"
        )

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("the audit surface paged the Session registry")


def _platform_whose_registries_raise(log: CountingLog | None = None) -> Platform:
    unused = log if log is not None else CountingLog()
    return Platform(
        event_log_append=unused,
        event_log_range=unused,
        definition_registry=RegistryThatRefusesContact(),
        tool_registry=ToolRegistryThatRefusesContact(),
        session_registry=SessionRegistryThatRefusesContact(),
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
        reviewer_authenticator=HmacReviewerTokens(
            key=_REVIEWER_KEY, clock=_FixedClock()
        ),
    )


def _establish_test_principals(app: FastAPI) -> FastAPI:
    """Stand in for the *tenant* authenticator nobody has written, ahead of the router.

    Sets the claim only when its header is present, so the absent-principal cases stay
    reachable through the same app: this middleware cannot make a request authorized
    that did not ask to be.

    It no longer touches the reviewer claim. That one is established by the real
    authenticator on the audit router, from a token this file mints with the same key
    the platform is wired with, so the reviewer half of every case below travels the
    path a deployment travels.
    """

    @app.middleware("http")
    async def establish(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        tenant = request.headers.get(_TENANT_CLAIM_TEST_HEADER)
        if tenant is not None:
            setattr(request.state, TENANT_CLAIM, UUID(tenant))
        return await call_next(request)

    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://control")


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _reviewer() -> dict[str, str]:
    """A real credential: signed, unexpired, and carrying the audit audience."""
    return _bearer(
        mint_reviewer_token(
            reviewer_id=uuid.uuid4(),
            expiry_epoch_s=_REVIEWER_EXPIRY,
            key=_REVIEWER_KEY,
        )
    )


def _tenant() -> dict[str, str]:
    return {TENANT_HEADER: str(uuid.uuid4())}


async def test_the_mounted_surface_refuses_a_caller_that_proves_nothing() -> None:
    """The shipped app mounts this route and answers 401 to anyone without a credential.

    Deny by default, and asserted rather than assumed. A caller who presents no token is
    nobody, and a caller who presents a tenant credential is somebody the surface does
    not serve — and a placeholder that trusted a header, the way the tenant surface's
    does, would instead have made every tenant's history readable by anyone who set one.
    """
    log = CountingLog()
    session_id = new_session_id()
    log.seed(session_id, 3)

    async with _client(create_app(_platform_whose_registries_raise(log))) as caller:
        anonymous = await caller.get(f"/v1/audit/sessions/{session_id}/events")
        with_a_tenant = await caller.get(
            f"/v1/audit/sessions/{session_id}/events", headers=_tenant()
        )

    assert anonymous.status_code == 401, anonymous.text
    assert with_a_tenant.status_code == 401, (
        "a tenant credential opened the cross-tenant audit surface: "
        f"{with_a_tenant.status_code} {with_a_tenant.text}"
    )
    assert log.reads == 0, "a refused audit request still read the Event Log"


def _with_last_hex_digit_changed(token: str) -> str:
    """The same token with its signature altered, for certain.

    Replacing the last character with a fixed digit is what this did, and one run in
    sixteen the digit already there WAS that digit -- so the case presented a valid
    credential and read a 200 as a failure to refuse. Measured at 2 failures in 20 runs
    of this case alone, because the reviewer id is a fresh uuid4 each time.

    A mutation whose green state can be "the credential was never altered" grades
    whatever fails first rather than what it names. Rotating within the hex alphabet
    cannot land on the original, so this case now always presents a bad signature.
    """
    return token[:-1] + ("1" if token[-1] == "0" else "0")


_UNAUTHORIZED: Mapping[str, Callable[[], dict[str, str]]] = {
    "no credential at all": dict,
    "a tenant credential, which this surface does not serve": _tenant,
    "a tenant principal established alongside a reviewer one": lambda: {
        **_reviewer(),
        _TENANT_CLAIM_TEST_HEADER: str(uuid.uuid4()),
    },
    "a tenant principal on its own": lambda: {
        _TENANT_CLAIM_TEST_HEADER: str(uuid.uuid4())
    },
    "a Session's own token, replayed as a reviewer credential": lambda: _bearer(
        mint_session_token(
            session_id=new_session_id(),
            tenant_id=TenantId(uuid.uuid4()),
            expiry_epoch_s=_REVIEWER_EXPIRY,
            key=_REVIEWER_KEY,
        )
    ),
    "a reviewer token signed with the wrong key": lambda: _bearer(
        mint_reviewer_token(
            reviewer_id=uuid.uuid4(),
            expiry_epoch_s=_REVIEWER_EXPIRY,
            key=b"a key this platform does not hold",
        )
    ),
    "a reviewer token that has expired": lambda: _bearer(
        mint_reviewer_token(
            reviewer_id=uuid.uuid4(),
            expiry_epoch_s=_REVIEWER_LONG_EXPIRED,
            key=_REVIEWER_KEY,
        )
    ),
    "a reviewer token whose signature was altered": lambda: _bearer(
        _with_last_hex_digit_changed(
            mint_reviewer_token(
                reviewer_id=uuid.uuid4(),
                expiry_epoch_s=_REVIEWER_EXPIRY,
                key=_REVIEWER_KEY,
            )
        )
    ),
    "an unsigned assertion of the audience and a reviewer id": lambda: _bearer(
        f"{REVIEWER_AUDIENCE}.{uuid.uuid4()}.{_REVIEWER_EXPIRY}.{'0' * 64}"
    ),
    "a bearer scheme with no token behind it": lambda: {"Authorization": "Bearer"},
    "a credential presented under the wrong scheme": lambda: {
        "Authorization": "Basic "
        + mint_reviewer_token(
            reviewer_id=uuid.uuid4(),
            expiry_epoch_s=_REVIEWER_EXPIRY,
            key=_REVIEWER_KEY,
        )
    },
}
"""Every way of arriving at this surface without becoming a reviewer.

Parametrized over rather than sampled. Each entry is a separate decision about what must
not authorize a cross-tenant read, so a case written against the first would leave the
other ten graded by nothing -- and the entry that matters most is not the first one: a
Session token is a file inside a tenant's own pod, readable by the tenant's own agent
code, so if that one were ever believed here every tenant would hold a credential for
every other tenant's history.
"""


@pytest.mark.parametrize("presented", sorted(_UNAUTHORIZED), ids=lambda name: name)
async def test_a_caller_who_is_not_a_reviewer_reads_nothing(presented: str) -> None:
    """One status, one code, one message, and no read of the log, for all eleven.

    The message is asserted equal across the whole table and not merely present. A
    caller who could tell "your token expired" from "you are not a reviewer" from "that
    audience is wrong" would be handed a map of the door they are trying, one request at
    a time -- and the eleven reasons arrive from three different layers, so their
    collapsing into one answer is a property worth pinning rather than assuming.

    `log.reads` is what distinguishes a refusal from a read that happened to 401 on the
    way out.
    """
    log = CountingLog()
    session_id = new_session_id()
    log.seed(session_id, 3)
    app = _establish_test_principals(create_app(_platform_whose_registries_raise(log)))

    async with _client(app) as caller:
        refused = await caller.get(
            f"/v1/audit/sessions/{session_id}/events",
            headers=_UNAUTHORIZED[presented](),
        )

    assert refused.status_code == 401, refused.text
    body = refused.json()["error"]
    assert body["code"] == AUDIT_PRINCIPAL_UNRESOLVED, refused.text
    assert body["message"] == audit.UNAUTHORIZED_MESSAGE, refused.text
    assert log.reads == 0, "a refused audit request still read the Event Log"


async def test_the_one_credential_that_does_authorize_reads_the_log() -> None:
    """The other arm of the table above.

    Without this, every assertion in it could pass on a surface that refuses
    *everything* -- which is exactly the state this endpoint shipped in, and a refusal
    table alone cannot tell that apart from a working gate.
    """
    log = CountingLog()
    session_id = new_session_id()
    log.seed(session_id, 3)
    app = _establish_test_principals(create_app(_platform_whose_registries_raise(log)))

    async with _client(app) as caller:
        served = await caller.get(
            f"/v1/audit/sessions/{session_id}/events", headers=_reviewer()
        )

    assert served.status_code == 200, served.text
    assert [event["seq"] for event in served.json()["events"]] == [1, 2, 3]
    assert log.reads == 1


async def test_both_refusals_carry_the_one_code() -> None:
    """A caller cannot tell which half of the check it reached.

    Two codes would tell whoever is probing this surface whether their principal was
    missing or merely disqualified, which is a map of the door they are trying.
    """
    session_id = new_session_id()
    app = _establish_test_principals(create_app(_platform_whose_registries_raise()))

    async with _client(app) as caller:
        no_principal = await caller.get(f"/v1/audit/sessions/{session_id}/events")
        both_principals = await caller.get(
            f"/v1/audit/sessions/{session_id}/events",
            headers={**_reviewer(), _TENANT_CLAIM_TEST_HEADER: str(uuid.uuid4())},
        )
        a_tenant_alone = await caller.get(
            f"/v1/audit/sessions/{session_id}/events",
            headers={_TENANT_CLAIM_TEST_HEADER: str(uuid.uuid4())},
        )

    for refused in (no_principal, both_principals, a_tenant_alone):
        assert refused.status_code == 401, refused.text
        assert refused.json()["error"]["code"] == AUDIT_PRINCIPAL_UNRESOLVED
    assert AUDIT_PRINCIPAL_UNRESOLVED == "auth.audit_principal_unresolved"


async def test_a_served_audit_read_never_asks_who_owns_the_session() -> None:
    """The cross-tenant property from the inside: no owner is consulted.

    Every method on the Session registry raises here, so a 200 is only possible if the
    audit path reached none of them. A status code alone could not tell this apart from
    a lookup that happened to succeed.
    """
    log = CountingLog()
    session_id = new_session_id()
    log.seed(session_id, 4)
    app = _establish_test_principals(create_app(_platform_whose_registries_raise(log)))

    async with _client(app) as reviewer:
        page = await reviewer.get(
            f"/v1/audit/sessions/{session_id}/events",
            params={"from_seq": 1, "to_seq": 4},
            headers=_reviewer(),
        )

    assert page.status_code == 200, page.text
    assert [event["seq"] for event in page.json()["events"]] == [1, 2, 3, 4]
    assert log.reads == 1


# --------------------------------------------------------------------------------------
# The real platform: the two crossings, three tenants, and page-for-page equality
# --------------------------------------------------------------------------------------


@pytest.fixture
async def platform_client(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Platform, AsyncClient]]:
    """Both surfaces on one wired platform, authenticating reviewers for real.

    One app rather than two, because every claim here is a relationship between the two
    surfaces over one Event Log and two apps could not show that the pages are equal.

    The signing key is put in the environment the composition root reads, so the
    reviewer credential these cases present is verified by the object a deployment
    builds rather than by one assembled here. Without it `build` hands back the refusing
    default and every reviewer case below would 401 -- which is a real risk and not a
    hypothetical, because that default is deliberately indistinguishable from a wrong
    credential.
    """
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", _REVIEWER_KEY.decode())
    platform, engine = build(database_url)
    try:
        app = _establish_test_principals(create_app(platform))
        async with _client(app) as client:
            yield platform, client
    finally:
        await engine.dispose()


def _a_definition() -> dict[str, object]:
    return {
        "name": "audit-fixture",
        "instructions": "irrelevant to these tests",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SKILLS_SHA,
    }


_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
"""A digest-pinned image, because a registered shape refuses anything else."""


def _create_body(definition_id: str, environment_id: str) -> dict[str, object]:
    return {
        "definition_id": definition_id,
        "environment_id": environment_id,
        "grant": ["fs.read"],
        "scope": {"repository": "acme/widgets"},
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 7,
    }


async def _a_session(client: AsyncClient, owner: dict[str, str]) -> SessionId:
    """A Session owned by that tenant. Its creation event is seq 1."""
    registered = await client.post("/v1/agents", json=_a_definition(), headers=owner)
    assert registered.status_code == 201, registered.text
    # The real store is wired here, so the sandbox shape has to really exist and has to
    # belong to this owner -- the create path resolves it under the caller's tenant.
    shape = await client.post(
        "/v1/environments",
        json={"name": "audit-fixture", "runtime_image": _FIXTURE_IMAGE},
        headers=owner,
    )
    assert shape.status_code == 201, shape.text
    created = await client.post(
        "/v1/sessions",
        json=_create_body(registered.json()["id"], shape.json()["id"]),
        headers=owner,
    )
    assert created.status_code == 201, created.text
    return SessionId(uuid.UUID(created.json()["id"]))


async def _append(platform: Platform, session_id: SessionId, upto: int) -> None:
    for n in range(2, upto + 1):
        await platform.event_log_append.append(session_id, f"event.{n}", {"n": n})


async def _sweep(engine: AsyncEngine, session_id: SessionId, through: int) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "DELETE FROM event_log WHERE session_id = :sid AND seq <= :through"
            ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
            {"sid": session_id, "through": through},
        )


async def test_a_reviewer_reads_a_session_the_tenant_asking_would_be_refused(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """MAP-A42's substance: the read genuinely crosses the boundary MAP-8 holds.

    The same Session, in three requests. Its owner reads it. A stranger tenant is
    refused with the closed-set code, which is the isolation guard still working. The
    reviewer reads it in full while holding no tenant at all — so this is a third
    authorization rather than the stranger's with something switched off.
    """
    platform, client = platform_client
    owner = _tenant()
    stranger = _tenant()
    reviewer = _reviewer()
    session_id = await _a_session(client, owner)
    await _append(platform, session_id, 5)
    span = {"from_seq": 1, "to_seq": 5}

    theirs = await client.get(
        f"/v1/sessions/{session_id}/events", params=span, headers=owner
    )
    refused = await client.get(
        f"/v1/sessions/{session_id}/events", params=span, headers=stranger
    )
    ours = await client.get(
        f"/v1/audit/sessions/{session_id}/events", params=span, headers=reviewer
    )

    assert theirs.status_code == 200, theirs.text
    assert refused.status_code == 404, (
        "another tenant read a Session it does not own; MAP-8's isolation has been "
        f"weakened: {refused.status_code} {refused.text}"
    )
    assert refused.json()["error"]["code"] == "session.not_found"
    assert ours.status_code == 200, ours.text
    assert [event["seq"] for event in ours.json()["events"]] == [1, 2, 3, 4, 5]


async def test_a_reviewer_reads_three_tenants_sessions_in_one_shape(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """MAP-A42. Three owners, one principal, one page shape, and nobody was asked."""
    platform, client = platform_client
    reviewer = _reviewer()
    sessions = []
    for _ in range(3):
        session_id = await _a_session(client, _tenant())
        await _append(platform, session_id, 4)
        sessions.append(session_id)

    pages = []
    for session_id in sessions:
        page = await client.get(
            f"/v1/audit/sessions/{session_id}/events", headers=reviewer
        )
        assert page.status_code == 200, page.text
        pages.append(page.json())

    assert [page["session_id"] for page in pages] == [str(s) for s in sessions]
    assert {tuple(sorted(page)) for page in pages} == {
        ("events", "from_seq", "retained_floor", "session_id", "to_seq")
    }
    assert all(len(page["events"]) == 4 for page in pages)


async def test_a_tenant_credential_buys_nothing_on_the_audit_surface(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """The first crossing. The credential that works one route over is refused here.

    Asserted against a Session the tenant genuinely owns and can genuinely read, so the
    401 is about the surface's authorization and not about the caller having asked for
    something it was never entitled to.
    """
    platform, client = platform_client
    owner = _tenant()
    session_id = await _a_session(client, owner)
    await _append(platform, session_id, 3)

    theirs = await client.get(f"/v1/sessions/{session_id}/events", headers=owner)
    refused = await client.get(f"/v1/audit/sessions/{session_id}/events", headers=owner)

    assert theirs.status_code == 200, theirs.text
    assert refused.status_code == 401, (
        "a tenant credential authorized a cross-tenant read: "
        f"{refused.status_code} {refused.text}"
    )
    assert refused.json()["error"]["code"] == AUDIT_PRINCIPAL_UNRESOLVED
    assert "event." not in refused.text, "the refusal carried the events it refused"


async def test_a_reviewer_principal_buys_nothing_on_the_tenant_surface(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """The other crossing. Without it, "authorized differently" is only half shown.

    A reviewer claim satisfies the audit surface and is invisible to the tenant one,
    which refuses for a missing tenant exactly as it would for an anonymous caller.
    """
    platform, client = platform_client
    session_id = await _a_session(client, _tenant())
    await _append(platform, session_id, 3)
    reviewer = _reviewer()

    ours = await client.get(f"/v1/audit/sessions/{session_id}/events", headers=reviewer)
    refused = await client.get(f"/v1/sessions/{session_id}/events", headers=reviewer)

    assert ours.status_code == 200, ours.text
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "request.tenant_missing"


async def test_a_tenant_header_changes_nothing_the_audit_surface_returns(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """The audit answer is a function of the reviewer and the range, of nothing else.

    A stranger's tenant header is sent alongside the reviewer claim. If any part of this
    path consulted a tenant, the two bodies would differ — most likely by the second one
    becoming a 404, which is what the tenant surface answers that stranger.
    """
    platform, client = platform_client
    session_id = await _a_session(client, _tenant())
    await _append(platform, session_id, 4)
    reviewer = _reviewer()

    alone = await client.get(
        f"/v1/audit/sessions/{session_id}/events", headers=reviewer
    )
    alongside = await client.get(
        f"/v1/audit/sessions/{session_id}/events", headers={**reviewer, **_tenant()}
    )

    assert alone.status_code == alongside.status_code == 200, alongside.text
    assert alone.json() == alongside.json()


async def test_nothing_the_audit_surface_returns_carries_a_credential(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """MAP-A43's second half, over the page and over a refusal.

    Nothing read here may grant the ability to act as any tenant, and the cheapest way
    that promise breaks is a body echoing the principal that fetched it.
    """
    platform, client = platform_client
    session_id = await _a_session(client, _tenant())
    await _append(platform, session_id, 3)
    reviewer = _reviewer()

    page = await client.get(f"/v1/audit/sessions/{session_id}/events", headers=reviewer)
    refused = await client.get(
        f"/v1/audit/sessions/{session_id}/events",
        params={"from_seq": 5, "to_seq": 3},
        headers=reviewer,
    )

    assert page.status_code == 200, page.text
    assert sorted(page.json()) == [
        "events",
        "from_seq",
        "retained_floor",
        "session_id",
        "to_seq",
    ]
    assert refused.status_code == 400, refused.text
    for body in (page.text.lower(), refused.text.lower()):
        assert [word for word in _CREDENTIAL_WORDS if word in body] == [], body
        assert reviewer["Authorization"].removeprefix("Bearer ").lower() not in body, (
            "the audit surface echoed back the credential that fetched the page"
        )


def _without_request_id(body: Any) -> Any:
    """One response body with `request_id` dropped, so two calls can be compared.

    Every refusal carries an id minted per call, so two responses that refuse
    identically still differ in that one field, and comparing whole bodies across two
    requests would fail on nothing but the id. Dropping it leaves the comparison over
    everything that is *not* defined to differ — the class, the sentence, the code and
    the detail. A success page carries no such field and comes back untouched.
    """
    if not isinstance(body, dict):
        return body
    return {key: value for key, value in body.items() if key != "request_id"}


@pytest.mark.parametrize(
    ("from_seq", "to_seq"),
    [(3, 6), (3, 3), (9, 12), (1, 6), (1, 2), (5, 3), (1, 1001)],
    ids=[
        "a-span",
        "one-event",
        "above-the-head",
        "spanning-the-floor",
        "below-the-floor",
        "backwards",
        "wider-than-the-cap",
    ],
)
async def test_the_audit_page_equals_the_tenant_page_over_the_same_range(
    platform_client: tuple[Platform, AsyncClient],
    engine: AsyncEngine,
    from_seq: int,
    to_seq: int,
) -> None:
    """One Session, one range, two surfaces, and the bytes must match.

    Compared as whole bodies rather than as statuses, because the divergence worth
    catching is a reviewer being shown a differently-shaped page of the same events —
    a retained floor omitted, a range echoed back narrowed, an envelope with a different
    code. The Session is swept so that all three answers a range can have are reachable
    from one fixture.

    Every field but `request_id`, which is minted per call and so differs between any
    two requests. Its presence is asserted separately rather than left unchecked, so
    dropping it from the comparison cannot hide a refusal that carries none.
    """
    platform, client = platform_client
    owner = _tenant()
    reviewer = _reviewer()
    session_id = await _a_session(client, owner)
    await _append(platform, session_id, 6)
    await _sweep(engine, session_id, through=2)
    span = {"from_seq": from_seq, "to_seq": to_seq}

    theirs = await client.get(
        f"/v1/sessions/{session_id}/events", params=span, headers=owner
    )
    ours = await client.get(
        f"/v1/audit/sessions/{session_id}/events", params=span, headers=reviewer
    )

    assert (ours.status_code, _without_request_id(ours.json())) == (
        theirs.status_code,
        _without_request_id(theirs.json()),
    ), (
        f"range {from_seq}..{to_seq} differs between the two surfaces: audit "
        f"{ours.status_code} {ours.text} vs tenant {theirs.status_code} {theirs.text}"
    )
    if not ours.is_success:
        assert ours.json()["request_id"] and theirs.json()["request_id"], (
            "a refusal carried no request id, so the field dropped from the "
            "comparison above was never there to compare"
        )


async def test_the_comparison_above_reaches_all_three_answers(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """Guard the guard: equality over one answer would prove almost nothing.

    The parametrized cases are compared one at a time and each of them passes on its
    own, so nothing there notices if the fixture stops producing an expired range or an
    empty page. This asserts the statuses those seven spans actually reach.
    """
    platform, client = platform_client
    owner = _tenant()
    session_id = await _a_session(client, owner)
    await _append(platform, session_id, 6)
    await _sweep(engine, session_id, through=2)

    statuses = []
    for from_seq, to_seq in [
        (3, 6),
        (3, 3),
        (9, 12),
        (1, 6),
        (1, 2),
        (5, 3),
        (1, 1001),
    ]:
        answer = await client.get(
            f"/v1/sessions/{session_id}/events",
            params={"from_seq": from_seq, "to_seq": to_seq},
            headers=owner,
        )
        has_events = bool(answer.json().get("events")) if answer.is_success else False
        statuses.append((answer.status_code, has_events))

    assert statuses == [
        (200, True),
        (200, True),
        (200, False),
        (410, False),
        (410, False),
        (400, False),
        (400, False),
    ], statuses


async def test_the_audit_path_publishes_the_refusals_the_tenant_path_does(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """A caller generating a client from the schema sees the same closed set.

    Compared against the tenant path's own entries rather than against a copy written
    here, so the two cannot drift into agreeing with this test and not with each other.
    The ownership refusal is absent because the audit surface cannot emit one -- there
    is no owner to check -- and its absence is asserted rather than left unstated.

    The set no longer holds `422`, and the reason is worth keeping. FastAPI attaches one
    to every operation taking a typed parameter, and this app answers `400` instead; no
    published code carries `422` at all. Both paths carried the framework's entry
    identically, so comparing them to each other passed while both were wrong -- which
    is the blind spot in any test that only checks two surfaces agree.
    `test_the_schema_matches_the_answers.py` is what checks they agree with the app.
    """
    _, client = platform_client
    paths = (await client.get("/openapi.json")).json()["paths"]

    ours = paths[_AUDIT_PATH]["get"]["responses"]
    theirs = paths[_TENANT_PATH]["get"]["responses"]

    assert set(ours) == {"200", "400", "410"}
    assert "404" not in ours
    for status_code in ("200", "400", "410"):
        assert _schema_of(ours[status_code]) == _schema_of(theirs[status_code]), (
            f"the audit path publishes a different body for {status_code} than the "
            "tenant path does"
        )


def _schema_of(response: dict[str, Any]) -> Any:
    return response["content"]["application/json"]["schema"]


class UnusedWebhooks:
    """Satisfies the webhook store port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    webhook store would be grading something this file does not grade, and a quiet stub
    would let it pass while doing so.
    """

    async def register(
        self,
        tenant_id: TenantId,
        url: CallbackUrl,
        states: frozenset[SessionState],
        secret_ref: str,
    ) -> WebhookRecord:
        raise AssertionError("a test in this file registered a webhook")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file listed a tenant's webhooks")

    async def delete(self, webhook_id: UUID, tenant_id: TenantId) -> bool:
        raise AssertionError("a test in this file deleted a webhook")

    async def watching(
        self, tenant_id: TenantId, state: SessionState
    ) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file asked what watches a state")


class UnusedEnvironmentStore:
    """Satisfies the environment-store port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    environment store would be grading something this file does not grade, and a quiet
    stub would let it pass while doing so.
    """

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        raise AssertionError("a test in this file resolved an environment")
