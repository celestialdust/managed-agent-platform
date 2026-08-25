"""POST /v1/agents, and the Session that resolves what it registered.

The skills-publication check the plan's step 4 described is deliberately **not** here.
No component in this system can answer whether a commit is published -- there is no
owner for one anywhere in the plan and no git source provisioned -- so the clause was
dropped from this slice rather than faked. A test asserting a refusal that a stub
produced would be a test of the stub.

Two tiers in one file, deliberately. Most of it is tier 1 -- real routes over real HTTP
against in-memory ports -- because what is being graded is the route: which status comes
back, which field a refusal names, and whether a refused registration writes anything.
The store's own guarantees are graded against real PostgreSQL in
`tests/adapters/test_definition_schema.py` and `test_definition_registry.py`, and
repeating them here would assert one property in two places, free to disagree.

The last test is the exception and it is the slice's checkpoint: register a definition
and create a Session against it, through the real routes, against a real PostgreSQL and
the real adapters, and check the Session's own event records the revision that was
resolved. Nothing in-memory can prove that, because the thing being proved is that two
adapters and two routes agree about one number.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Starlette's TestClient is built on `httpx2`, while the async client used for the
# real-database tests below is `httpx`. Two distinct Response types, so the sync
# helper is annotated with the one its own client actually returns rather than with
# whichever happens to be imported.
from httpx2 import Response as SyncResponse
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.definition_registry import (
    PostgresDefinitionRegistry,
)
from managed_agent.adapters.postgres.event_log_append import PostgresEventLogAppend
from managed_agent.adapters.postgres.event_log_range import PostgresEventLogRange
from managed_agent.adapters.postgres.session_registry import PostgresSessionRegistry
from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.skills.evaluation import (
    Baseline,
    EvalFacts,
    Grade,
    RunRecord,
)
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import (
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    new_definition_id,
)
from managed_agent.core.ports import Resolution, SessionListing, UnknownDefinition
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord, SessionState
from managed_agent.core.vocabulary import lifecycle

_SHA = "0" * 39 + "a"


def _definition_body(**overrides: object) -> dict[str, object]:
    return {
        "name": "slr-reviewer",
        "instructions": "Extract findings and name the source document for each.",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SHA,
    } | overrides


@dataclass(frozen=True, slots=True)
class _Resolved:
    definition: AgentDefinition
    revision: int


class InMemoryRegistry:
    """Both registry methods over one dict, counting what it was asked to write.

    The write count is what lets a test assert that a *refused* registration wrote
    nothing. A fake that only answered reads could not tell us that, and "nothing was
    written" is the half of a refusal that actually matters -- a 400 that stored the row
    anyway is worse than no check at all, because the tenant believes it was rejected.
    """

    def __init__(self) -> None:
        self._rows: dict[DefinitionId, list[AgentDefinition]] = {}
        self._owner: dict[DefinitionId, TenantId] = {}
        self.writes = 0

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        self.writes += 1
        self._owner.setdefault(definition_id, tenant_id)
        revisions = self._rows.setdefault(definition_id, [])
        revisions.append(definition)
        return len(revisions)

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        revisions = self._rows.get(definition_id)
        if not revisions or self._owner.get(definition_id) != tenant_id:
            raise UnknownDefinition(str(definition_id))
        return _Resolved(revisions[-1], len(revisions))

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        if self._owner.get(definition_id) != tenant_id:
            return ()
        return tuple(
            VersionFact(revision=n, archived=False)
            for n, _ in enumerate(self._rows.get(definition_id, ()), start=1)
        )

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        if self._owner.get(definition_id) != tenant_id:
            return None
        revisions = self._rows.get(definition_id, [])
        return revisions[revision - 1] if 1 <= revision <= len(revisions) else None

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a test in this file retired a definition version")

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        raise AssertionError("a definition-registration test read an eval baseline")

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        raise AssertionError("a definition-registration test recorded an eval run")

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        return EvalFacts(repository_enrolled=False, revision_accepted=False)


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class InMemoryLog:
    def __init__(self) -> None:
        self._events: dict[SessionId, list[Event]] = {}

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        log = self._events.setdefault(session_id, [])
        seq = Seq(len(log) + 1)
        log.append(Event(session_id, seq, type_, dict(payload)))
        return seq

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Event]:
        span = [
            event
            for event in self._events.get(session_id, [])
            if start <= event.seq <= end
        ]
        return span[:limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Event]:
        for event in self._events.get(session_id, []):
            if event.seq > after:
                yield event

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return Seq(1)

    def events(self, session_id: SessionId) -> list[Event]:
        return list(self._events.get(session_id, []))


@dataclass(frozen=True, slots=True)
class Harness:
    client: TestClient
    registry: InMemoryRegistry
    log: InMemoryLog
    tenant: TenantId

    def register(self, **overrides: object) -> SyncResponse:
        return self.client.post(
            "/v1/agents",
            json=_definition_body(**overrides),
            headers={TENANT_HEADER: str(self.tenant)},
        )


@pytest.fixture
def harness() -> Harness:
    registry, log = InMemoryRegistry(), InMemoryLog()
    platform = Platform(
        event_log_append=log,
        event_log_range=log,
        definition_registry=registry,
        tool_registry=UnusedToolRegistry(),
        session_registry=WriteOnlySessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=AnyEnvironmentResolves(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    return Harness(
        client=TestClient(create_app(platform)),
        registry=registry,
        log=log,
        tenant=TenantId(uuid.uuid4()),
    )


def test_a_valid_definition_registers_as_revision_one(harness: Harness) -> None:
    """201, an addressable id, and the revision a Session would pin."""
    response = harness.register()

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["revision"] == 1
    assert uuid.UUID(body["id"])


def test_the_registered_id_is_not_one_the_tenant_chose(harness: Harness) -> None:
    """The platform mints the id, so a tenant cannot aim a registration at another
    tenant's definition by naming its id."""
    first = harness.register().json()["id"]
    second = harness.register().json()["id"]

    assert first != second


def test_a_malformed_definition_is_refused_and_the_body_names_the_field(
    harness: Harness,
) -> None:
    """400, and the response says which field was wrong.

    A refusal that does not name the field leaves a tenant guessing across seven of
    them, and the commonest mistake -- a branch name where a commit id belongs -- looks
    identical to a missing field.
    """
    response = harness.register(skills_revision="main")

    assert response.status_code == 400, response.text
    assert "skills_revision" in response.text
    assert harness.registry.writes == 0, (
        "a malformed definition was written to the registry; the tenant was told "
        "it was refused while the row exists"
    )


def test_an_unknown_field_is_refused_and_named(harness: Harness) -> None:
    response = harness.register(multiagnet={"enabled": True})

    assert response.status_code == 400, response.text
    assert "multiagnet" in response.text
    assert harness.registry.writes == 0


def test_registration_requires_a_tenant(harness: Harness) -> None:
    """No tenant, no registration -- rather than a registration nobody owns.

    400 rather than a default tenant. A default is the failure that survives real
    multi-tenancy arriving: every call site keeps working and quietly serves one
    tenant's data as another's.
    """
    response = harness.client.post("/v1/agents", json=_definition_body())

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "request.tenant_missing"
    assert harness.registry.writes == 0


def test_a_malformed_tenant_is_refused_rather_than_coerced(harness: Harness) -> None:
    response = harness.client.post(
        "/v1/agents",
        json=_definition_body(),
        headers={TENANT_HEADER: "not-a-uuid"},
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "request.tenant_malformed"
    assert harness.registry.writes == 0


def test_creating_a_session_records_the_resolved_definition_revision(
    harness: Harness,
) -> None:
    """The revision the Session pinned is in the event that created it.

    Pinned at creation and written into the log, so a definition registered afterwards
    cannot change what this Session is. Read out of the event rather than out of the
    response, because the event is what every later reader folds.
    """
    definition_id = harness.register().json()["id"]

    created = harness.client.post(
        "/v1/sessions",
        json={
            "definition_id": definition_id,
            # Any id resolves through this file's environment store; which sandbox
            # shape a Session runs in is graded in test_environment_reference.py.
            "environment_id": str(uuid.uuid4()),
            "budget_minor_units": 500,
            "budget_currency": "USD",
            "retention_days": 30,
        },
        headers={TENANT_HEADER: str(harness.tenant)},
    )

    assert created.status_code == 201, created.text
    session_id = SessionId(uuid.UUID(created.json()["id"]))
    events = harness.log.events(session_id)
    assert len(events) == 1
    assert events[0].type == lifecycle.SESSION_CREATED
    assert events[0].payload["definition_revision"] == 1, (
        f"the created event does not record the resolved revision: {events[0].payload}"
    )


def test_a_session_against_an_unknown_definition_appends_nothing(
    harness: Harness,
) -> None:
    """The refusal comes before the append, so no Session half-exists.

    A Session whose creation event was written and whose definition does not resolve is
    the worst of both: it is addressable, it folds to RUNNING, and nothing can run it.
    """
    unknown = str(new_definition_id())

    response = harness.client.post(
        "/v1/sessions",
        json={
            "definition_id": unknown,
            # Any id resolves through this file's environment store; which sandbox
            # shape a Session runs in is graded in test_environment_reference.py.
            "environment_id": str(uuid.uuid4()),
            "budget_minor_units": 500,
            "budget_currency": "USD",
            "retention_days": 30,
        },
        headers={TENANT_HEADER: str(harness.tenant)},
    )

    # 404 from the table, not a status chosen at the call site. Creating a Session
    # now resolves through the published closed error set, whose table gives
    # `definition.not_found` a 404 -- MAP-4 wrote the literal code before that set
    # existed and chose a status by hand. The code is unchanged; where it sits in
    # the body and which status carries it are the set's answer now.
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "definition.not_found"
    assert harness.log.events(SessionId(uuid.UUID(unknown))) == []


def test_another_tenants_definition_cannot_start_a_session(harness: Harness) -> None:
    """The tenant filter is applied on the resolve, not assumed from the register."""
    definition_id = harness.register().json()["id"]

    response = harness.client.post(
        "/v1/sessions",
        json={
            "definition_id": definition_id,
            # Any id resolves through this file's environment store; which sandbox
            # shape a Session runs in is graded in test_environment_reference.py.
            "environment_id": str(uuid.uuid4()),
            "budget_minor_units": 500,
            "budget_currency": "USD",
            "retention_days": 30,
        },
        headers={TENANT_HEADER: str(uuid.uuid4())},
    )

    # 404 from the table, not a status chosen at the call site. Creating a Session
    # now resolves through the published closed error set, whose table gives
    # `definition.not_found` a 404 -- MAP-4 wrote the literal code before that set
    # existed and chose a status by hand. The code is unchanged; where it sits in
    # the body and which status carries it are the set's answer now.
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "definition.not_found"


async def test_register_then_create_resolves_the_pinned_revision_end_to_end(
    engine: AsyncEngine,
) -> None:
    """The slice's checkpoint, through both real routes against real PostgreSQL.

    Registered, then a Session created against it, then the Session's own event read
    back out of the real Event Log -- and the revision in it is the one the real
    registry wrote. Every in-memory test above grades one route in isolation; this is
    the only one that can fail if the two adapters disagree about the number, which is
    the way this slice is most likely to be wrong.

    Driven with `AsyncClient` over `ASGITransport` rather than with `TestClient`.
    `TestClient` runs the app in an event loop of its own, and an async engine's pooled
    connections belong to the loop that opened them -- sharing one across the two
    produces `got Future attached to a different loop` on the second request, which
    reads like a database fault and is not one.
    """
    platform = Platform(
        event_log_append=PostgresEventLogAppend(engine),
        event_log_range=PostgresEventLogRange(engine),
        definition_registry=PostgresDefinitionRegistry(engine),
        tool_registry=UnusedToolRegistry(),
        session_registry=PostgresSessionRegistry(engine),
        webhooks=UnusedWebhooks(),
        environment_store=AnyEnvironmentResolves(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    tenant = {TENANT_HEADER: str(uuid.uuid4())}

    async with AsyncClient(
        transport=ASGITransport(app=create_app(platform)), base_url="http://tenant"
    ) as client:
        registered = await client.post(
            "/v1/agents", json=_definition_body(), headers=tenant
        )
        assert registered.status_code == 201, registered.text
        assert registered.json()["revision"] == 1

        created = await client.post(
            "/v1/sessions",
            json={
                "definition_id": registered.json()["id"],
                # Any id resolves through this file's environment store; which sandbox
                # shape a Session runs in is graded in test_environment_reference.py.
                "environment_id": str(uuid.uuid4()),
                "budget_minor_units": 500,
                "budget_currency": "USD",
                "retention_days": 30,
            },
            headers=tenant,
        )
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]

        # The tenant header is sent because reading a Session is now tenant-scoped:
        # the registry settles who owns it before the log is folded, and a read with no
        # tenant is refused rather than served.
        read = await client.get(f"/v1/sessions/{session_id}", headers=tenant)
        assert read.status_code == 200, read.text
        assert read.json()["state"] == "running"

        # A second registration of a *different* definition must not move what this
        # Session pinned -- the whole reason the revision is written down.
        again = await client.post(
            "/v1/agents", json=_definition_body(name="other"), headers=tenant
        )
        assert again.status_code == 201, again.text

    events = await PostgresEventLogRange(engine).read(
        SessionId(uuid.UUID(session_id)), Seq(1), Seq(10)
    )
    assert [event.type for event in events] == [lifecycle.SESSION_CREATED]
    assert events[0].payload["definition_revision"] == 1, (
        f"the pinned revision did not reach the log: {events[0].payload}"
    )


async def test_a_session_cannot_be_created_against_another_tenants_definition_for_real(
    engine: AsyncEngine,
) -> None:
    """The tenant filter holds through the real adapter, not only the in-memory one.

    The in-memory test above proves the route calls `resolve` with a tenant. This proves
    the SQL actually filters on it -- which is a property of the statement and of the
    column, and a fake keyed by id could satisfy the first while the second was missing.
    """
    platform = Platform(
        event_log_append=PostgresEventLogAppend(engine),
        event_log_range=PostgresEventLogRange(engine),
        definition_registry=PostgresDefinitionRegistry(engine),
        tool_registry=UnusedToolRegistry(),
        session_registry=PostgresSessionRegistry(engine),
        webhooks=UnusedWebhooks(),
        environment_store=AnyEnvironmentResolves(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    owner = {TENANT_HEADER: str(uuid.uuid4())}
    stranger = {TENANT_HEADER: str(uuid.uuid4())}

    async with AsyncClient(
        transport=ASGITransport(app=create_app(platform)), base_url="http://tenant"
    ) as client:
        registered = await client.post(
            "/v1/agents", json=_definition_body(), headers=owner
        )
        refused = await client.post(
            "/v1/sessions",
            json={
                "definition_id": registered.json()["id"],
                # Any id resolves through this file's environment store; which sandbox
                # shape a Session runs in is graded in test_environment_reference.py.
                "environment_id": str(uuid.uuid4()),
                "budget_minor_units": 500,
                "budget_currency": "USD",
                "retention_days": 30,
            },
            headers=stranger,
        )

    # 404 from the table, not a status chosen at the call site. Creating a Session
    # now resolves through the published closed error set, whose table gives
    # `definition.not_found` a 404 -- MAP-4 wrote the literal code before that set
    # existed and chose a status by hand. The code is unchanged; where it sits in
    # the body and which status carries it are the set's answer now.
    assert refused.status_code == 404, refused.text
    assert refused.json()["error"]["code"] == "definition.not_found"


class FixedRevisionRegistry:
    """A registry whose `register` reports a revision that is deliberately not 1.

    Every definition registered through the route lands on revision 1, because the route
    mints a fresh id and a fresh id has no earlier revision. That makes 1 indistinguish-
    able from a hardcoded 1 in every test that uses the real path -- so this returns 7,
    and the only way the response can say 7 is by reading what the registry returned.
    """

    def __init__(self, revision: int) -> None:
        self.revision = revision

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        return self.revision

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        return _Resolved(
            AgentDefinition.model_validate(_definition_body()), self.revision
        )

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        return (VersionFact(revision=self.revision, archived=False),)

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        return AgentDefinition.model_validate(_definition_body())

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a test in this file retired a definition version")

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        raise AssertionError("a revision-number test read a skill eval baseline")

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        raise AssertionError("a revision-number test recorded a skill eval run")

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        return EvalFacts(repository_enrolled=False, revision_accepted=False)


def test_the_response_reports_the_revision_the_registry_wrote() -> None:
    """The number comes from the store, not from an assumption in the handler.

    Today every registration is revision 1, so a handler that returned a literal 1 would
    pass every other test in this file. It would then be wrong the moment the endpoint
    that registers a new revision of an existing id exists, and wrong in the worst way:
    a tenant told it created revision 1 while the store holds revision 4, pinning
    Sessions to a revision nobody named.
    """
    log = InMemoryLog()
    platform = Platform(
        event_log_append=log,
        event_log_range=log,
        definition_registry=FixedRevisionRegistry(7),
        tool_registry=UnusedToolRegistry(),
        session_registry=WriteOnlySessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=AnyEnvironmentResolves(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    client = TestClient(create_app(platform))

    response = client.post(
        "/v1/agents",
        json=_definition_body(),
        headers={TENANT_HEADER: str(uuid.uuid4())},
    )

    assert response.status_code == 201, response.text
    assert response.json()["revision"] == 7, (
        "the response did not report the revision the registry returned; a handler "
        "assuming 1 is indistinguishable from a correct one until a second revision "
        "exists"
    )


def test_a_session_records_the_revision_the_registry_resolved_not_a_literal_one() -> (
    None
):
    """A Session created against revision 7 records 7.

    The same blind spot as the test above, on the other route. Every Session in every
    other test pins revision 1, so `"definition_revision": 1` written as a literal would
    pass all of them -- and would silently pin every Session in production to a revision
    that is only sometimes the right one.
    """
    log = InMemoryLog()
    platform = Platform(
        event_log_append=log,
        event_log_range=log,
        definition_registry=FixedRevisionRegistry(7),
        tool_registry=UnusedToolRegistry(),
        session_registry=WriteOnlySessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=AnyEnvironmentResolves(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    client = TestClient(create_app(platform))

    created = client.post(
        "/v1/sessions",
        json={
            "definition_id": str(new_definition_id()),
            # Any id resolves through this file's environment store; which sandbox
            # shape a Session runs in is graded in test_environment_reference.py.
            "environment_id": str(uuid.uuid4()),
            "budget_minor_units": 500,
            "budget_currency": "USD",
            "retention_days": 30,
        },
        headers={TENANT_HEADER: str(uuid.uuid4())},
    )

    assert created.status_code == 201, created.text
    events = log.events(SessionId(uuid.UUID(created.json()["id"])))
    assert events[0].payload["definition_revision"] == 7, (
        f"the created event records {events[0].payload.get('definition_revision')!r} "
        "rather than the resolved revision 7"
    )


async def test_a_session_pins_the_revision_current_when_it_was_created(
    engine: AsyncEngine,
) -> None:
    """Against real PostgreSQL: two revisions exist, and the Session pins the later one.

    Registered twice through the real adapter so a second revision genuinely exists in
    the table, then a Session created through the route. This is the property the whole
    slice is for, and it cannot be stated at all while only revision 1 exists.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id, tenant = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    first = await registry.register(
        definition_id, tenant, AgentDefinition.model_validate(_definition_body())
    )
    second = await registry.register(
        definition_id,
        tenant,
        AgentDefinition.model_validate(_definition_body(name="second")),
    )
    assert (first, second) == (1, 2)

    platform = Platform(
        event_log_append=PostgresEventLogAppend(engine),
        event_log_range=PostgresEventLogRange(engine),
        definition_registry=registry,
        tool_registry=UnusedToolRegistry(),
        session_registry=PostgresSessionRegistry(engine),
        webhooks=UnusedWebhooks(),
        environment_store=AnyEnvironmentResolves(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    async with AsyncClient(
        transport=ASGITransport(app=create_app(platform)), base_url="http://tenant"
    ) as client:
        created = await client.post(
            "/v1/sessions",
            json={
                "definition_id": str(definition_id),
                # Any id resolves through this file's environment store; which sandbox
                # shape a Session runs in is graded in test_environment_reference.py.
                "environment_id": str(uuid.uuid4()),
                "budget_minor_units": 500,
                "budget_currency": "USD",
                "retention_days": 30,
            },
            headers={TENANT_HEADER: str(tenant)},
        )

    assert created.status_code == 201, created.text
    events = await PostgresEventLogRange(engine).read(
        SessionId(uuid.UUID(created.json()["id"])), Seq(1), Seq(10)
    )
    assert events[0].payload["definition_revision"] == 2, (
        "the Session pinned "
        f"{events[0].payload.get('definition_revision')!r} rather than the revision "
        "that was current when it was created"
    )


class UnusedToolRegistry:
    """Satisfies the tool-registry port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    tool registry would be grading something this file does not grade, and a quiet stub
    would let it pass while doing so.
    """

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("a test in this file registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("a test in this file looked up a registered tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("a test in this file listed a tenant's tools")


class WriteOnlySessionRegistry:
    """Accepts the row a create writes and refuses every read of it.

    Creating a Session now records its owner, so a fake here has to accept that write --
    but nothing in this file reads a Session back through the in-memory platform, and a
    fake that answered reads would let a test start doing so without saying it had. The
    real registry is wired in the tests below that need one.
    """

    def __init__(self) -> None:
        self.written: list[SessionRecord] = []

    async def create(self, record: SessionRecord) -> None:
        self.written.append(record)

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        raise AssertionError("a test in this file read a Session through the fake")

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("a test in this file listed Sessions through the fake")


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


FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
"""A digest-pinned image, because `Environment` refuses anything else."""


class AnyEnvironmentResolves:
    """An environment store that resolves any id to one fixed shape.

    Deliberately unable to refuse. Whether an unknown environment is refused, and what a
    refusal leaves behind, are graded in `test_environment_reference.py` where the store
    can say no -- asserting it here as well would put one behaviour in two places, free
    to disagree the first time the refusal changes shape.
    """

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        return {
            "id": str(environment_id),
            "tenant_id": str(tenant_id),
            "name": "session-fixture",
            "runtime_image": FIXTURE_IMAGE,
            "denied_paths": [],
        }
