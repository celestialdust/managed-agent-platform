"""The app factory takes its ports and builds no infrastructure of its own.

Tier 1 (local, no infrastructure). Two properties. The app serves the Session
routes, so attaching a router really happened rather than being asserted about.
And create_app reaches no database: it is handed a Platform, so a test can drive
the real routes against in-memory ports and the composition root stays the only
place a concrete adapter is chosen. The second is checked twice — once by making
the engine constructor explode, and once over the source of every module under
control/, because "no second composition root" is a claim about the tree and not
about one function.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Never
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from managed_agent import composition
from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import FIRST_SEQ, DefinitionId, Seq, SessionId, TenantId
from managed_agent.core.ports import EventRecord, Resolution, SessionListing
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord, SessionState

_CONTROL = Path(__file__).parents[2] / "src" / "managed_agent" / "control"


class UnusedLog:
    """Satisfies both log ports and is never called: these test wiring only."""

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        raise AssertionError("a wiring test appended to the log")

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]:
        raise AssertionError("a wiring test read the log")

    async def follow(
        self, session_id: SessionId, after: Seq
    ) -> AsyncIterator[EventRecord]:
        raise AssertionError("a wiring test followed the log")
        yield  # pragma: no cover - unreachable, and what makes this a generator

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ


class UnusedRegistry:
    """Satisfies the definition-registry port and is never called.

    Registering or resolving from a wiring test would mean a route ran, which is what
    these tests are asserting does *not* happen — so both raise rather than returning a
    harmless value that would let such a test pass quietly.
    """

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        raise AssertionError("a wiring test registered a definition")

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        raise AssertionError("a wiring test resolved a definition")

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        raise AssertionError("a wiring test listed a definition's versions")

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        raise AssertionError("a wiring test read a definition revision")

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a wiring test retired a definition version")


_HEADERS = {TENANT_HEADER: str(uuid4())}


def _platform() -> Platform:
    unused = UnusedLog()
    return Platform(
        event_log_append=unused,
        event_log_range=unused,
        definition_registry=UnusedRegistry(),
        tool_registry=UnusedToolRegistry(),
        session_registry=UnusedSessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )


def test_the_app_holds_the_platform_it_was_handed() -> None:
    platform = _platform()
    app = create_app(platform)
    assert isinstance(app, FastAPI)
    assert app.state.platform is platform


def test_the_session_routes_are_mounted_under_the_version_prefix() -> None:
    """A malformed body proves the route is reached, where a 404 would not.

    The tenant header is sent even though the body is empty. Creating a Session now
    depends on a tenant, and that dependency resolves before the body is validated — so
    without the header the answer is 400 for a missing tenant, which proves the route
    was reached but no longer proves the *body* was the thing refused.
    """
    client = TestClient(create_app(_platform()), headers=_HEADERS)

    assert client.post("/v1/sessions", json={}).status_code == 400
    assert client.post("/sessions", json={}).status_code == 404


def test_the_registration_route_is_mounted_under_the_version_prefix() -> None:
    """The second router is attached, and under the same prefix as the first.

    Asserted here rather than only in the registration tests, because this file is what
    grades the app factory: a router defined correctly and never included would pass
    every test in `test_registration_definitions.py` that builds its own app, and 404
    in production.
    """
    client = TestClient(create_app(_platform()), headers=_HEADERS)

    assert client.post("/v1/agents", json={}).status_code == 400
    assert client.post("/agents", json={}).status_code == 404


def test_the_openapi_surface_publishes_both_session_routes() -> None:
    paths = create_app(_platform()).openapi()["paths"]
    assert "/v1/sessions" in paths
    assert "/v1/sessions/{session_id}" in paths


def test_create_app_builds_no_database_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handed a Platform, it must not reach for one.

    The engine constructor is booby-trapped at the composition root, which is the
    only place in the tree that names it.
    """

    def explode(*args: object, **kwargs: object) -> Never:
        raise AssertionError("create_app constructed a database engine")

    monkeypatch.setattr(composition, "create_async_engine", explode)

    assert create_app(_platform()).state.platform is not None


def test_nothing_under_control_names_the_engine_constructor() -> None:
    offenders = [
        str(module.relative_to(_CONTROL))
        for module in _CONTROL.rglob("*.py")
        if "create_async_engine" in module.read_text()
    ]
    assert offenders == [], f"control/ reaches for infrastructure in {offenders}"


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
