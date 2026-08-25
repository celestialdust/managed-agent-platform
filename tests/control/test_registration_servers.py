"""POST /v1/mcp_servers, and the one registration two agent definitions both reach.

Two tiers in one file, deliberately. Most of it is tier 1 -- real routes over real HTTP
against in-memory ports -- because what is being graded is the route: which status comes
back, which tool a refusal names, and whether a refused registration writes anything.
The store's own guarantees are graded against real PostgreSQL in
`tests/adapters/test_tool_registry_schema.py` and `test_tool_registry.py`, and repeating
them here would assert one property in two places, free to disagree.

The last two tests are the exception and they are the slice's checkpoint: one server
registered and two agent definitions naming it, through the real routes against real
PostgreSQL and the real adapters, with the endpoint read back once per definition and
the server row counted. Nothing in-memory can prove that, because the thing being proved
is that two adapters and three routes agree that the configuration was stated once.

The server registered throughout is this project's own MCP server, at its real URL and
with the three tool names it really offers. Nothing here dials it -- the suite is
hermetic and a registration is a write to two tables -- but the fixture is not a fiction
either, so the shapes asserted are shapes a real registration takes.
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
# real-database tests below is `httpx`. Two distinct Response types, so the sync helper
# is annotated with the one its own client actually returns.
from httpx2 import Response as SyncResponse
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.definition_registry import (
    PostgresDefinitionRegistry,
)
from managed_agent.adapters.postgres.event_log_append import PostgresEventLogAppend
from managed_agent.adapters.postgres.event_log_range import PostgresEventLogRange
from managed_agent.adapters.postgres.tool_registry import PostgresToolRegistry
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
from managed_agent.core.ids import DefinitionId, Seq, SessionId, TenantId
from managed_agent.core.ports import Resolution, SessionListing, UnknownDefinition
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    NameAlreadyRegistered,
    RegisteredTool,
    ServerRegistration,
    StreamableHttpServer,
    UnknownTool,
)
from managed_agent.core.session.session import SessionRecord, SessionState

_SHA = "0" * 39 + "a"

DEEPWIKI_ENDPOINT: dict[str, object] = {
    "transport": "streamable_http",
    "url": "https://mcp.deepwiki.com",
    "credential_ref": "vault/acme/deepwiki",
}


def _tool_body(name: str = "ask_question", **overrides: object) -> dict[str, object]:
    return {
        "name": name,
        "remote_name": name,
        "parameters": {"repoName": "string", "question": "string"},
        "scope_bindings": [{"dimension": "repository", "argument": "repoName"}],
    } | overrides


def _registration_body(**overrides: object) -> dict[str, object]:
    return {
        "server_name": "deepwiki",
        "endpoint": DEEPWIKI_ENDPOINT,
        "tools": [_tool_body()],
    } | overrides


def _definition_body(**overrides: object) -> dict[str, object]:
    return {
        "name": "slr-reviewer",
        "instructions": "Extract findings and name the source document for each.",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SHA,
        "tool_servers": ["deepwiki"],
    } | overrides


class InMemoryToolRegistry:
    """The registry port over two dicts, counting what it was asked to write.

    The write count is what lets a test assert that a *refused* registration wrote
    nothing. A fake that only answered reads could not tell us that, and "nothing was
    written" is the half of a refusal that actually matters -- a 400 that stored the
    catalog anyway is worse than no check at all, because the tenant believes it was
    rejected.
    """

    def __init__(self) -> None:
        self._servers: dict[tuple[TenantId, str], ServerRegistration] = {}
        self._tools: dict[tuple[TenantId, str], RegisteredTool] = {}
        self.writes = 0

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        if (tenant_id, registration.server_name) in self._servers:
            raise NameAlreadyRegistered("server", (registration.server_name,))
        taken = tuple(
            sorted(
                tool.name
                for tool in registration.tools
                if (tenant_id, tool.name) in self._tools
            )
        )
        if taken:
            raise NameAlreadyRegistered("tool", taken)
        self.writes += 1
        self._servers[(tenant_id, registration.server_name)] = registration
        for tool in registration.tools:
            self._tools[(tenant_id, tool.name)] = RegisteredTool(
                name=tool.name,
                remote_name=tool.remote_name,
                parameters=tool.parameters,
                scope_bindings=tool.scope_bindings,
                server_name=registration.server_name,
                endpoint=registration.endpoint,
            )

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        found = self._tools.get((tenant_id, tool_name))
        if found is None:
            raise UnknownTool(tool_name)
        return found

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        return [
            tool
            for (owner, _), tool in sorted(self._tools.items())
            if owner == tenant_id
        ]

    def server_count(self, tenant_id: TenantId) -> int:
        return sum(1 for owner, _ in self._servers if owner == tenant_id)


@dataclass(frozen=True, slots=True)
class _Resolved:
    definition: AgentDefinition
    revision: int


class InMemoryDefinitionRegistry:
    def __init__(self) -> None:
        self._rows: dict[DefinitionId, list[AgentDefinition]] = {}

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        revisions = self._rows.setdefault(definition_id, [])
        revisions.append(definition)
        return len(revisions)

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        revisions = self._rows.get(definition_id)
        if not revisions:
            raise UnknownDefinition(str(definition_id))
        return _Resolved(revisions[-1], len(revisions))

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        return tuple(
            VersionFact(revision=n, archived=False)
            for n, _ in enumerate(self._rows.get(definition_id, ()), start=1)
        )

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        revisions = self._rows.get(definition_id, [])
        return revisions[revision - 1] if 1 <= revision <= len(revisions) else None

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a test in this file retired an agent definition version")

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        raise AssertionError("a tool-registration test read a skill eval baseline")

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        raise AssertionError("a tool-registration test recorded a skill eval run")

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


@dataclass(frozen=True, slots=True)
class Harness:
    client: TestClient
    registry: InMemoryToolRegistry
    tenant: TenantId

    def register(self, **overrides: object) -> SyncResponse:
        return self.client.post(
            "/v1/mcp_servers",
            json=_registration_body(**overrides),
            headers={TENANT_HEADER: str(self.tenant)},
        )


@pytest.fixture
def harness() -> Harness:
    registry, log = InMemoryToolRegistry(), InMemoryLog()
    platform = Platform(
        event_log_append=log,
        event_log_range=log,
        definition_registry=InMemoryDefinitionRegistry(),
        tool_registry=registry,
        session_registry=UnusedSessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    return Harness(
        client=TestClient(create_app(platform)),
        registry=registry,
        tenant=TenantId(uuid.uuid4()),
    )


def test_a_valid_registration_echoes_the_names_the_caller_may_now_address(
    harness: Harness,
) -> None:
    """201, the server name a definition will use, and the tool names a Grant will use.

    Three tools rather than one: a handler echoing only the first would satisfy a
    single-tool registration, and a caller checking what it can now name would be told
    two thirds of the truth.
    """
    response = harness.register(
        tools=[
            _tool_body("read_wiki_structure", parameters={"repoName": "string"}),
            _tool_body("read_wiki_contents", parameters={"repoName": "string"}),
            _tool_body("ask_question"),
        ]
    )

    assert response.status_code == 201, response.text
    assert response.json() == {
        "server_name": "deepwiki",
        "tools": ["read_wiki_structure", "read_wiki_contents", "ask_question"],
    }
    assert harness.registry.writes == 1


def test_a_tool_binding_an_argument_it_does_not_declare_is_refused_and_named(
    harness: Harness,
) -> None:
    """400 naming the tool and the argument, and nothing is written.

    The refusal has to name both. A tenant reading only "inexpressible" is comparing two
    identical-looking strings across a catalog, and the commonest cause is a typo in the
    argument name.
    """
    response = harness.register(
        tools=[
            _tool_body(
                scope_bindings=[{"dimension": "repository", "argument": "repoNmae"}]
            )
        ]
    )

    assert response.status_code == 400, response.text
    assert "ask_question" in response.text
    assert "repoNmae" in response.text
    assert harness.registry.writes == 0, (
        "an inexpressible Scope Binding was written; the tenant was told it was "
        "refused while the tool is reachable"
    )


def test_a_tool_with_no_scope_binding_is_refused_and_nothing_is_written(
    harness: Harness,
) -> None:
    """The security-observable half: refused, and reachable by no Session afterwards.

    Both halves are asserted. A 400 that stored the row anyway would leave a tool with
    no enforceable narrowing in the catalog, reachable at the full breadth of the
    tenant's data by every Session whose Grant names it.
    """
    response = harness.register(tools=[_tool_body(scope_bindings=[])])

    assert response.status_code == 400, response.text
    assert "ask_question" in response.text
    assert "no Scope Binding" in response.text
    assert harness.registry.writes == 0
    assert harness.registry.server_count(harness.tenant) == 0


def test_one_bad_tool_refuses_the_whole_registration(harness: Harness) -> None:
    """Not "the good ones registered and the bad one did not".

    A partial write leaves the tenant believing it registered a catalog it did not, and
    the tools that did land are the ones nobody was told about.
    """
    response = harness.register(
        tools=[_tool_body("read_wiki_contents"), _tool_body(scope_bindings=[])]
    )

    assert response.status_code == 400, response.text
    assert harness.registry.writes == 0


def test_a_tool_name_already_registered_is_refused_with_a_conflict(
    harness: Harness,
) -> None:
    """409 rather than 400, and the code says it was the *tool* name that collided.

    The two codes are what a caller branches on: a taken server name means this
    registration was already made, while a taken tool name means two servers are
    offering one name to the same Grant, which the caller has to resolve.
    """
    assert harness.register().status_code == 201

    response = harness.register(server_name="acme-wiki")

    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "tool_registration.tool_name_already_registered"
    assert error["detail"]["names"] == "ask_question"
    assert harness.registry.writes == 1, "the refused registration wrote a second time"
    assert harness.registry.server_count(harness.tenant) == 1


def test_re_registering_a_server_name_is_refused_with_the_server_code(
    harness: Harness,
) -> None:
    response = harness.register()
    assert response.status_code == 201

    again = harness.register(tools=[_tool_body("ask_question_v2")])

    assert again.status_code == 409, again.text
    error = again.json()["error"]
    assert error["code"] == "tool_registration.server_name_already_registered"
    assert error["detail"]["names"] == "deepwiki"
    assert harness.registry.writes == 1


def test_another_tenant_may_register_the_same_names(harness: Harness) -> None:
    """The collision is per tenant, so the first tenant to take a name does not take it
    from everyone."""
    assert harness.register().status_code == 201

    other = harness.client.post(
        "/v1/mcp_servers",
        json=_registration_body(),
        headers={TENANT_HEADER: str(uuid.uuid4())},
    )

    assert other.status_code == 201, other.text
    assert harness.registry.writes == 2


def test_registration_requires_a_tenant(harness: Harness) -> None:
    """No tenant, no registration -- rather than a catalog nobody owns."""
    response = harness.client.post("/v1/mcp_servers", json=_registration_body())

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "request.tenant_missing"
    assert harness.registry.writes == 0


def test_the_route_is_reachable_on_the_real_app_under_the_version_prefix(
    harness: Harness,
) -> None:
    """Attaching the router really happened, and it happened under `/v1`.

    A router defined correctly and never included 404s in production while every test
    that drives it through its own `APIRouter` keeps passing. The empty body proves the
    route was *reached* -- a 404 would not -- and the unversioned path proves the prefix
    is what carries it.
    """
    reached = harness.client.post(
        "/v1/mcp_servers", json={}, headers={TENANT_HEADER: str(harness.tenant)}
    )

    assert reached.status_code == 400, reached.text
    assert harness.client.post("/mcp_servers", json={}).status_code == 404
    assert "/v1/mcp_servers" in create_app(_unused_platform()).openapi()["paths"]


def _unused_platform() -> Platform:
    """Ports that would raise if a route reached them; this test reaches none."""
    log, registry = InMemoryLog(), InMemoryToolRegistry()
    return Platform(
        event_log_append=log,
        event_log_range=log,
        definition_registry=InMemoryDefinitionRegistry(),
        tool_registry=registry,
        session_registry=UnusedSessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )


async def test_one_registration_serves_two_definitions_naming_it(
    engine: AsyncEngine,
) -> None:
    """The slice's checkpoint, through the real routes against real PostgreSQL.

    One server registered, two agent definitions naming it, and the endpoint read back
    once for each definition -- the same endpoint both times, from one server row. That
    last clause is what "its configuration was stated once" means as something a test
    can fail on: a design that copied the endpoint per definition would answer both
    reads correctly and leave two rows free to drift.

    Driven with `AsyncClient` over `ASGITransport` rather than with `TestClient`.
    `TestClient` runs the app in an event loop of its own, and an async engine's pooled
    connections belong to the loop that opened them -- sharing one across the two
    produces `got Future attached to a different loop` on the second request, which
    reads like a database fault and is not one.
    """
    registry = PostgresToolRegistry(engine)
    platform = Platform(
        event_log_append=PostgresEventLogAppend(engine),
        event_log_range=PostgresEventLogRange(engine),
        definition_registry=PostgresDefinitionRegistry(engine),
        tool_registry=registry,
        session_registry=UnusedSessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    tenant_id = TenantId(uuid.uuid4())
    tenant = {TENANT_HEADER: str(tenant_id)}

    async with AsyncClient(
        transport=ASGITransport(app=create_app(platform)), base_url="http://tenant"
    ) as client:
        registered = await client.post(
            "/v1/mcp_servers",
            json=_registration_body(
                tools=[
                    _tool_body(
                        "read_wiki_structure", parameters={"repoName": "string"}
                    ),
                    _tool_body("ask_question"),
                ]
            ),
            headers=tenant,
        )
        assert registered.status_code == 201, registered.text

        reviewer = await client.post(
            "/v1/agents", json=_definition_body(name="slr-reviewer"), headers=tenant
        )
        triager = await client.post(
            "/v1/agents", json=_definition_body(name="issue-triager"), headers=tenant
        )
        assert (reviewer.status_code, triager.status_code) == (201, 201)
        assert reviewer.json()["id"] != triager.json()["id"]

    tools = await registry.list_for_tenant(tenant_id)

    assert [tool.name for tool in tools] == ["ask_question", "read_wiki_structure"]
    endpoints = {tool.endpoint for tool in tools}
    assert len(endpoints) == 1, (
        f"the two tools resolve to {len(endpoints)} endpoints; a definition naming "
        "this server would reach a different one depending on which tool it called"
    )
    only = endpoints.pop()
    assert isinstance(only, StreamableHttpServer)
    assert only.url == "https://mcp.deepwiki.com"
    assert only.credential_ref == "vault/acme/deepwiki"


async def test_a_refused_registration_leaves_nothing_a_definition_could_reach(
    engine: AsyncEngine,
) -> None:
    """Through the real store: 409, and no server row and no tool row behind it.

    The in-memory tests above prove the route maps the refusal. This proves the store
    was not written first -- a registration that inserted its server row before
    discovering the tool collision would leave a server nothing can reach and a name
    the tenant cannot re-use, and the 409 would look identical.
    """
    registry = PostgresToolRegistry(engine)
    platform = Platform(
        event_log_append=PostgresEventLogAppend(engine),
        event_log_range=PostgresEventLogRange(engine),
        definition_registry=PostgresDefinitionRegistry(engine),
        tool_registry=registry,
        session_registry=UnusedSessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )
    tenant_id = TenantId(uuid.uuid4())
    tenant = {TENANT_HEADER: str(tenant_id)}

    async with AsyncClient(
        transport=ASGITransport(app=create_app(platform)), base_url="http://tenant"
    ) as client:
        assert (
            await client.post(
                "/v1/mcp_servers", json=_registration_body(), headers=tenant
            )
        ).status_code == 201

        refused = await client.post(
            "/v1/mcp_servers",
            json=_registration_body(
                server_name="acme-wiki",
                tools=[_tool_body(), _tool_body("read_wiki_contents")],
            ),
            headers=tenant,
        )

    assert refused.status_code == 409, refused.text
    assert (
        refused.json()["error"]["code"]
        == "tool_registration.tool_name_already_registered"
    )
    assert [tool.name for tool in await registry.list_for_tenant(tenant_id)] == [
        "ask_question"
    ]
    with pytest.raises(UnknownTool):
        await registry.lookup(tenant_id, "read_wiki_contents")


class UnusedSessionRegistry:
    """Satisfies the Session-registry port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    Session registry would be grading something this file does not grade, and a quiet
    stub would let it pass while doing so.
    """

    async def create(self, record: SessionRecord) -> None:
        raise AssertionError("a test in this file wrote a Session registry row")

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        raise AssertionError("a test in this file fetched a Session registry row")

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("a test in this file paged the Session registry")


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
