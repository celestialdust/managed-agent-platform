"""What naming a registered shape buys, and what naming an absent one costs.

Tier 1 (local, in-memory ports). Two claims.

The first is that one shape reaches two Sessions unchanged, so the proof compares two
compiled configurations built from one Environment and requires them equal in every part
the Environment decides, while the Sessions themselves differ. Both configurations are
built from shapes that came back *out of a store*, not from the value that went in, so
the parse on the read path is exercised rather than bypassed.

The second is that an unregistered shape costs a Session nothing: the refusal is proved
against what the collaborators hold afterwards, which is empty, rather than against the
status code alone.

The store below is a real in-memory store rather than a mock, and it hands rows back the
way the Postgres adapter does -- the same column names, uuids already rendered to
strings -- so a rename or a type change on either side surfaces here.
"""

import re
import tomllib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from pydantic import ValidationError

from managed_agent.composition import Platform
from managed_agent.control.api import refusals
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import environments, sessions
from managed_agent.control.catalog.environments import (
    ALREADY_DENIED,
    ENVIRONMENT_COLUMNS,
    UnknownEnvironment,
    parse_environment,
    resolve_environment,
    resolve_environment_at,
    resolve_environment_revision,
)
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.pod_config.compiler import (
    PROFILE_NAME,
    WORKSPACE_ROOT,
    CompiledConfig,
    compile_session_config,
    session_profile,
)
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import (
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    new_definition_id,
    new_session_id,
)
from managed_agent.core.ports import EventRecord, Resolution, SessionListing
from managed_agent.core.registration.definition import (
    AgentDefinition,
    SkillsRevision,
    VersionFact,
)
from managed_agent.core.registration.environment import (
    MAX_DENIED_PATHS,
    CreateEnvironment,
    Environment,
    EnvironmentId,
    new_environment_id,
)
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import (
    CreateSession,
    SessionRecord,
    SessionState,
)

GATEWAY_URL = "https://tool-gateway.map.internal/mcp"

# Where a Session pod reaches the Model Gateway. The `/v1` is load-bearing at both ends:
# the Agent Runtime POSTs `{base_url}/responses`, and the Gateway's router mounts
# `POST /v1/responses`.
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"

# The Gateway's signing key and the token's deadline, which the compiler takes from
# its caller and never defaults. Literals, so no case here can expire mid-run.
SESSION_TOKEN_KEY = b"a signing key that is thirty-two"
SESSION_TOKEN_EXPIRY = 4102444800
IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
OTHER_IMAGE = "registry.map.internal/session@sha256:" + "b" * 64
SECRETS = f"{WORKSPACE_ROOT}/secrets"
_ROOT = Path(__file__).resolve().parents[2]
MIGRATION = _ROOT / "migrations" / "versions" / "0014_environments.py"
SESSIONS_ROUTE = (
    _ROOT / "src" / "managed_agent" / "control" / "api" / "routes" / "sessions.py"
)
_SKILLS_SHA = "0" * 39 + "a"


# --------------------------------------------------------------------------------------
# Fixtures: a store that behaves like the adapter, and the collaborators a create needs
# --------------------------------------------------------------------------------------


class FakeEnvironmentStore:
    """The whole store, in memory, keyed the way the Postgres adapter keys it.

    `fetch` takes the tenant as part of the key rather than filtering afterwards, the
    property the real adapter holds with a WHERE clause: another tenant's shape is
    absent from the answer instead of read and then discarded.

    Revisions are a list per `(id, tenant)` and a read takes the LAST one, which is the
    adapter's `ORDER BY revision DESC LIMIT 1` written the only way an in-memory store
    can write it. Rows carry `revision` and `archived_at` because the adapter's rows do
    -- a stand-in that omitted either would let the parse above fall back to its default
    and every case here would pass while the real read was broken.

    `created_at_ms` is a counter rather than a clock, so the order a page comes back in
    is the order these tests registered things and not the resolution of the machine's
    clock. Two Environments registered in one millisecond is a real case for the adapter
    and an untestable one here; the keyset's second half is what covers it, and it is
    covered against a real database instead.
    """

    def __init__(self) -> None:
        self._revisions: dict[tuple[UUID, UUID], list[dict[str, object]]] = {}
        self._archived: dict[UUID, datetime] = {}
        self._registered_at: dict[UUID, int] = {}
        self.inserted: list[EnvironmentId] = []
        self.fetched: list[EnvironmentId] = []
        self.sessions_holding: dict[UUID, int] = {}
        """How many unstopped Sessions name each id. Set by a test; nothing here can
        derive it, because the pairing lives in an Event Log this store never reads."""

    def _row(self, environment: Environment, revision: int) -> dict[str, object]:
        return {
            "id": str(environment.id),
            "tenant_id": str(environment.tenant_id),
            "name": environment.name,
            "runtime_image": environment.runtime_image,
            "denied_paths": list(environment.denied_paths),
            "allowed_domains": list(environment.allowed_domains),
            "revision": revision,
            "archived_at": self._archived.get(environment.id),
        }

    async def insert(self, environment: Environment, /) -> None:
        key = (environment.id, environment.tenant_id)
        assert key not in self._revisions, (
            "a create wrote an id that already holds a revision; the real store "
            "answers that with a primary-key violation, and an edit is insert_revision"
        )
        self.inserted.append(environment.id)
        self._registered_at[environment.id] = len(self._registered_at) + 1
        self._revisions[key] = [self._row(environment, 1)]

    async def insert_revision(self, environment: Environment, /) -> int:
        key = (environment.id, environment.tenant_id)
        assert key in self._revisions, "an edit was written for an id nobody registered"
        revision = len(self._revisions[key]) + 1
        self._revisions[key].append(self._row(environment, revision))
        return revision

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        self.fetched.append(environment_id)
        held = self._revisions.get((environment_id, tenant_id))
        if not held:
            return None
        return {**held[-1], "archived_at": self._archived.get(environment_id)}

    async def fetch_revision(
        self,
        environment_id: EnvironmentId,
        tenant_id: TenantId,
        revision: int,
        /,
    ) -> Mapping[str, object] | None:
        self.fetched.append(environment_id)
        held = self._revisions.get((environment_id, tenant_id)) or []
        for row in held:
            if row["revision"] == revision:
                return {**row, "archived_at": self._archived.get(environment_id)}
        return None

    async def page(
        self,
        tenant_id: TenantId,
        after: tuple[int, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Mapping[str, object]]:
        rows = [
            {
                **held[-1],
                "archived_at": self._archived.get(environment_id),
                "created_at_ms": self._registered_at[environment_id],
            }
            for (environment_id, owner), held in self._revisions.items()
            if owner == tenant_id
            and held
            and (include_archived or environment_id not in self._archived)
        ]
        rows.sort(
            key=lambda row: (int(str(row["created_at_ms"])), str(row["id"])),
            reverse=True,
        )
        if after is not None:
            rows = [
                row
                for row in rows
                if (int(str(row["created_at_ms"])), str(row["id"]))
                < (after[0], str(after[1]))
            ]
        return rows[:limit]

    async def archive(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> datetime | None:
        if not self._revisions.get((environment_id, tenant_id)):
            return None
        # setdefault, so a repeat answers with the FIRST retirement's moment. A fresh
        # one would let a retried call claim the Environment stopped being
        # referenceable later than it did.
        return self._archived.setdefault(
            environment_id, datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
        )

    async def delete(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> bool:
        if self._revisions.pop((environment_id, tenant_id), None) is None:
            return False
        self._archived.pop(environment_id, None)
        return True

    async def sessions_referencing(self, environment_id: EnvironmentId, /) -> int:
        return self.sessions_holding.get(environment_id, 0)

    def touched(self) -> list[EnvironmentId]:
        """Every id this store was asked about, whichever way round."""
        return self.inserted + self.fetched


# The definition a Session pins, for the one field the compiler reads off it: the model.
# The provider is not here because it is not the definition's to name -- every model
# call leaves a Session pod through the Model Gateway.
A_DEFINITION = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)


def an_environment(
    tenant_id: TenantId,
    *,
    image: str = IMAGE,
    denied_paths: tuple[str, ...] = (SECRETS,),
) -> Environment:
    return parse_environment(
        environment_id=new_environment_id(),
        tenant_id=tenant_id,
        name="analysis",
        runtime_image=image,
        denied_paths=denied_paths,
    )


def a_record() -> SessionRecord:
    return SessionRecord(
        id=new_session_id(),
        tenant_id=TenantId(uuid4()),
        definition_id=new_definition_id(),
        definition_revision="rev-1",
        grant=frozenset(),
        scope=(),
        budget_minor_units=10_000,
        budget_currency="USD",
        retention_days=30,
    )


def _a_definition() -> dict[str, object]:
    return {
        "name": "environment-fixture",
        "instructions": "irrelevant to these tests",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SKILLS_SHA,
    }


@dataclass(frozen=True, slots=True)
class _Resolved:
    definition: AgentDefinition
    revision: int


@dataclass(frozen=True, slots=True)
class _Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


@dataclass
class RecordingLog:
    """Records appends and refuses every read: these cases never fold a log."""

    appended: list[_Event] = field(default_factory=list)

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        seq = len(self.appended) + 1
        self.appended.append(_Event(session_id, seq, type_, payload))
        return seq

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]:
        raise AssertionError("a test in this file read the log")

    def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[EventRecord]:
        raise AssertionError("a test in this file followed the log")

    async def retained_floor(self, session_id: SessionId) -> Seq:
        raise AssertionError("a test in this file asked for a retained floor")


@dataclass
class RecordingRegistry:
    """The Session registry, recording what a create actually wrote."""

    created: list[SessionRecord] = field(default_factory=list)

    async def create(self, record: SessionRecord) -> None:
        self.created.append(record)

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        raise AssertionError("a test in this file fetched a Session registry row")

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("a test in this file paged the Session registry")


class AlwaysResolves:
    """A definition registry that resolves any id to revision 1.

    Whether an unknown definition is refused is graded where the registry can say no.
    Here it must never be the reason a create fails, or a case about environments would
    pass for the wrong reason.
    """

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        return 1

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        return _Resolved(AgentDefinition.model_validate(_a_definition()), 1)

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        return (VersionFact(revision=1, archived=False),)

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        return AgentDefinition.model_validate(_a_definition())

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a test in this file retired a definition version")


class UnusedToolRegistry:
    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("a test in this file registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("a test in this file looked up a registered tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("a test in this file listed a tenant's tools")


class UnusedWebhooks:
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


@dataclass(frozen=True, slots=True)
class Harness:
    """The real `Platform`, plus the two recorders the negative claims are read from.

    A duck-typed stand-in cannot serve a route here: `platform_from_request` narrows
    with `isinstance(platform, Platform)`, and `Platform` is a frozen dataclass rather
    than a Protocol, so a four-field look-alike raises before any assertion is reached.
    """

    platform: Platform
    store: FakeEnvironmentStore
    log: RecordingLog
    registry: RecordingRegistry


def a_harness(store: FakeEnvironmentStore | None = None) -> Harness:
    environment_store = store if store is not None else FakeEnvironmentStore()
    log = RecordingLog()
    registry = RecordingRegistry()
    return Harness(
        platform=Platform(
            event_log_append=log,
            event_log_range=log,
            definition_registry=AlwaysResolves(),
            tool_registry=UnusedToolRegistry(),
            session_registry=registry,
            webhooks=UnusedWebhooks(),
            environment_store=environment_store,
            turn_dispatch=NoPodTransport(),
            file_store=unconfigured_file_store(),
        ),
        store=environment_store,
        log=log,
        registry=registry,
    )


def build_app(platform: Platform) -> FastAPI:
    """The two routers over one platform, refused the way the real app refuses.

    Nothing plants a tenant on the app: both routers read theirs from the placeholder in
    `tenancy.py`, so a caller names one in a header the way any client does.

    The envelope install is load-bearing rather than decoration, and it goes before the
    routers because that is the order it covers them in. Two refusals exercised below
    are *raised*, not returned: a missing tenant header refuses inside a dependency, and
    a body FastAPI rejects never enters a route at all. Without these handlers the first
    escapes as an unhandled exception and the second answers in Starlette's shape, so a
    hand-mounted router would pass every route-level test here while disagreeing with
    the deployed app on both.
    """
    app = FastAPI()
    app.state.platform = platform
    refusals.install_request_envelope(app)
    app.include_router(sessions.router, prefix="/v1")
    app.include_router(environments.router, prefix="/v1")
    return app


def caller(app: FastAPI, tenant: TenantId) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://platform",
        headers={TENANT_HEADER: str(tenant)},
    )


def _without_request_id(response: httpx.Response) -> str:
    """One response's body text with its own per-request id masked out.

    The id is minted per request, before any handler decides anything, so no two
    responses carry the same one and it reports nothing about either. It has to come out
    before two refusals can be compared byte for byte: left in, every pair differs on it
    and the comparison loses the power to detect the differences it exists to detect.
    """
    body: dict[str, object] = response.json()
    return response.text.replace(str(body["request_id"]), "<request>")


def a_create_body(environment_id: EnvironmentId) -> dict[str, object]:
    return {
        "definition_id": str(new_definition_id()),
        "environment_id": str(environment_id),
        "budget_minor_units": 10_000,
        "budget_currency": "USD",
        "retention_days": 30,
    }


# --------------------------------------------------------------------------------------
# The shape itself: what cannot be constructed
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image",
    [
        "registry.map.internal/session:latest",
        "registry.map.internal/session",
        "registry.map.internal/session@sha256:" + "a" * 63,
        "registry.map.internal/session@sha256:" + "g" * 64,
        "registry.map.internal/session@" + "a" * 64,
        "registry.map.internal/session@md5:" + "a" * 64,
        "   @sha256:" + "a" * 64,
        "@sha256:" + "a" * 64,
    ],
)
def test_an_image_that_is_not_pinned_to_bytes_is_refused(image: str) -> None:
    """A tag names whatever was pushed last, so it cannot make two Sessions one
    shape."""
    with pytest.raises(ValueError, match="runtime_image"):
        an_environment(TenantId(uuid4()), image=image)


def test_a_registry_host_with_a_port_is_accepted() -> None:
    """The refusal is about the digest, not about punctuation in the reference."""
    pinned = "registry.map.internal:5000/team/session@sha256:" + "c" * 64
    assert an_environment(TenantId(uuid4()), image=pinned).runtime_image == pinned


@pytest.mark.parametrize(
    "path",
    [
        "session/workspace/x",
        f"{WORKSPACE_ROOT}/*",
        f"{WORKSPACE_ROOT}/logs?",
        f"{WORKSPACE_ROOT}/[abc]",
        f"{WORKSPACE_ROOT}/x/",
        f"{WORKSPACE_ROOT}//x",
        f"{WORKSPACE_ROOT}/./x",
        f"{WORKSPACE_ROOT}/../etc",
    ],
)
def test_a_denied_path_the_sandbox_would_not_recognise_is_refused(path: str) -> None:
    """Every refusal here is a path the runtime would look for somewhere else.

    A glob makes the runtime scan the tree while it compiles the argv; an unnormalised
    spelling names a directory the pod created under a different string.
    """
    with pytest.raises(ValueError, match="denied path"):
        an_environment(TenantId(uuid4()), denied_paths=(path,))


def test_the_same_path_denied_twice_is_refused() -> None:
    """The Permission Profile refuses two rules over one path, so this must never reach
    it: a shape that registered and then failed every Session created against it would
    show the tenant the fault far from the mistake."""
    with pytest.raises(ValueError, match="denied twice"):
        an_environment(TenantId(uuid4()), denied_paths=(SECRETS, SECRETS))


def test_more_denied_paths_than_the_bound_are_refused() -> None:
    """Every entry becomes a rule in a sandbox argv compiled on every launch."""
    too_many = tuple(f"{WORKSPACE_ROOT}/p{n}" for n in range(MAX_DENIED_PATHS + 1))
    with pytest.raises(ValueError, match=str(MAX_DENIED_PATHS)):
        an_environment(TenantId(uuid4()), denied_paths=too_many)


def test_a_name_that_is_only_whitespace_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="   ",
            runtime_image=IMAGE,
            denied_paths=(),
        )


@pytest.mark.parametrize(
    "attribute", ["id", "tenant_id", "name", "runtime_image", "denied_paths"]
)
def test_no_field_of_a_registered_shape_can_be_rewritten(attribute: str) -> None:
    """Every field, not a sample: the guarantee is that the *shape* cannot change, and a
    single writable field is enough to break it on the second Session."""
    environment = an_environment(TenantId(uuid4()))
    with pytest.raises(FrozenInstanceError):
        setattr(environment, attribute, "anything")


def test_a_registration_body_refuses_a_field_the_platform_does_not_offer() -> None:
    """An ignored unknown field would let a tenant believe it had configured
    something."""
    with pytest.raises(ValidationError):
        CreateEnvironment.model_validate(
            {"name": "x", "runtime_image": IMAGE, "egress": ["example.com"]}
        )


def test_a_registration_body_refuses_an_empty_name() -> None:
    with pytest.raises(ValidationError):
        CreateEnvironment.model_validate({"name": "", "runtime_image": IMAGE})


# --------------------------------------------------------------------------------------
# The registry: the one direction a shape may narrow
#
# Two of the four clauses live here -- outside the writable root, and restating a path
# the platform already denies. The other two are about one path's relation to another
# and to the names the runtime protects, and they live with the shapes they refuse in
# `test_environment_shapes_the_sandbox_can_build.py`. A reader who stops here would
# conclude the two below are all of them.
# --------------------------------------------------------------------------------------


def test_a_path_under_the_writable_root_is_accepted() -> None:
    """The positive half. Without it every refusal below is satisfied by a parse that
    refuses everything, and nothing here would tell the two apart."""
    assert an_environment(TenantId(uuid4())).denied_paths == (SECRETS,)


def test_the_platforms_own_denies_under_the_workspace_are_not_an_empty_set() -> None:
    """Guard the guard for the clause below.

    `ALREADY_DENIED` is derived from the compiled profile rather than restated, so if
    the derivation ever yields nothing the "already denied" refusal becomes unreachable
    and every case that exercises it passes by never entering the branch.
    """
    assert ALREADY_DENIED, (
        "no path under the writable root is denied by the platform profile, so the "
        "clause refusing a shape that redeclares one can never fire"
    )
    assert set(ALREADY_DENIED) <= set(session_profile().denied())


@pytest.mark.parametrize(
    "path",
    [
        WORKSPACE_ROOT,
        "/run/codex/ctl/app-server-control.sock",
        "/etc/codex",
        "/var/lib/map/codex",
        "/data/reference",
        f"{WORKSPACE_ROOT}x/escape",
    ],
)
def test_a_shape_reaching_outside_what_it_may_narrow_is_refused(path: str) -> None:
    """An environment narrows what the agent may write and reaches nothing else.

    Written as an allowlist -- strictly inside the writable root -- rather than as a
    list of platform paths to keep out, so a platform path added later is refused by a
    rule nobody has to remember to extend. `{root}x/escape` is here because a prefix
    test written without the trailing separator would admit it.
    """
    with pytest.raises(ValueError, match=r"is outside"):
        an_environment(TenantId(uuid4()), denied_paths=(path,))


def test_the_filesystem_root_cannot_be_denied() -> None:
    """Separate from the case above because it is refused one clause earlier.

    `"/"` ends in a separator, so it is rejected as an unnormalised path before the
    writable-root clause is reached. `FsRule` exempts the root from that rule; nothing
    here needs to, because a shape may not name the root under either reading. What is
    asserted is that it is refused, not which sentence says so.
    """
    with pytest.raises(ValueError):
        an_environment(TenantId(uuid4()), denied_paths=("/",))


@pytest.mark.parametrize("path", ALREADY_DENIED)
def test_a_shape_redeclaring_a_platform_deny_is_refused_at_registration(
    path: str,
) -> None:
    """Refused here rather than at the first Session, which is where it would otherwise
    surface: the Permission Profile refuses two rules over one path, so such a shape
    would register cleanly and then fail every Session created against it."""
    with pytest.raises(ValueError, match="already denied"):
        an_environment(TenantId(uuid4()), denied_paths=(path,))


async def test_the_fake_store_round_trips_one_row_per_tenant() -> None:
    """Held honest before anything below leans on it."""
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    environment = an_environment(tenant)
    await store.insert(environment)

    row = await store.fetch(environment.id, tenant)
    assert row is not None
    assert tuple(sorted(row)) == tuple(
        sorted((*ENVIRONMENT_COLUMNS, "allowed_domains", "revision", "archived_at"))
    ), (
        "the stand-in's row and the adapter's row have to carry the same keys. The "
        "three beyond ENVIRONMENT_COLUMNS are the ones that tuple names as absent from "
        "0014 and present in a read; drop one here and the parse falls back to a "
        "default, so every case in this file passes over a read that lost a column"
    )
    assert await store.fetch(environment.id, TenantId(uuid4())) is None
    assert await store.fetch(new_environment_id(), tenant) is None


async def test_a_stored_row_resolves_to_an_equal_shape() -> None:
    """Equality of the whole value, not of a field or two: the read path re-parses, and
    a parse that dropped `denied_paths` would pass a spot check on the image."""
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    registered = an_environment(tenant)
    await store.insert(registered)

    assert await resolve_environment(store, registered.id, tenant) == registered


async def test_a_read_resolves_the_latest_revision_and_says_which() -> None:
    """The one claim every route in the lifecycle rests on.

    Before revisions, "the row with this id" had one answer and the store could select
    on the id alone. It cannot now: an edit appends, so a read that did not order by
    revision would return whichever row the plan reached first -- right until the first
    tenant edits anything, and then silently wrong in the worst possible place, since
    the value decides what a sandbox may reach.

    Both halves are asserted. The shape is the second one's, and the number is 2 -- a
    read that returned the newest shape while reporting revision 1 would pin every later
    Session to a revision it is not running.
    """
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    first = an_environment(tenant)
    await store.insert(first)
    edited = parse_environment(
        environment_id=first.id,
        tenant_id=tenant,
        name="analysis-with-egress",
        runtime_image=OTHER_IMAGE,
        denied_paths=(SECRETS,),
        allowed_domains=("api.example.com",),
    )
    assert await store.insert_revision(edited) == 2

    resolved = await resolve_environment_revision(store, first.id, tenant)

    assert resolved.revision == 2
    assert resolved.environment == edited
    assert resolved.archived_at is None
    assert not resolved.archived
    # The narrow function agrees with the wide one rather than reading separately.
    assert await resolve_environment(store, first.id, tenant) == edited


async def test_one_revision_resolves_as_revision_one() -> None:
    """The floor of the claim above, and it is not decoration: `revision` is read with a
    default of 1, so a read that lost the column entirely would satisfy the case above
    only by accident and this one always."""
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    registered = an_environment(tenant)
    await store.insert(registered)

    resolved = await resolve_environment_revision(store, registered.id, tenant)

    assert resolved.revision == 1


async def test_a_pinned_revision_resolves_to_the_shape_it_named() -> None:
    """The other half of pinning, and the half that makes the number worth writing down.

    Recording the revision buys nothing on its own; what it buys is this read. Resolved
    at the pin, an edit reaches the next Session and no earlier one; resolved at the
    newest revision, the edit is retroactive and a Session runs in a sandbox its creator
    never agreed to, with nothing in the log saying when the reach changed.
    """
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    first = an_environment(tenant)
    await store.insert(first)
    widened = parse_environment(
        environment_id=first.id,
        tenant_id=tenant,
        name="analysis-with-egress",
        runtime_image=OTHER_IMAGE,
        denied_paths=(SECRETS,),
        allowed_domains=("api.example.com",),
    )
    await store.insert_revision(widened)

    assert await resolve_environment_at(store, first.id, tenant, 1) == first
    assert await resolve_environment_at(store, first.id, tenant, 2) == widened
    # And the unpinned read is the one that moved, which is what makes the pair a real
    # distinction rather than two names for one answer.
    assert await resolve_environment(store, first.id, tenant) == widened


async def test_a_pinned_revision_of_a_retired_shape_still_resolves() -> None:
    """Archiving refuses a NEW Session and does not stop one already created.

    A pod for a Session created before the retirement is still built, or archiving
    becomes a way to stop live work instead of a way to stop new work. So this read is
    deliberately not gated on the retirement -- the refusal lives at the create route,
    where the Session that does not exist yet is being asked for.
    """
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    registered = an_environment(tenant)
    await store.insert(registered)
    await store.archive(registered.id, tenant)

    assert await resolve_environment_at(store, registered.id, tenant, 1) == registered


@pytest.mark.parametrize("revision", [2, 99])
async def test_a_pinned_revision_that_names_no_row_does_not_resolve(
    revision: int,
) -> None:
    """One refusal for a revision nobody wrote and for one another tenant wrote.

    A caller able to tell "no such revision" from "no such id" could count another
    tenant's edits, which is the same oracle a distinguishable not-found would be.
    """
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    registered = an_environment(tenant)
    await store.insert(registered)

    with pytest.raises(UnknownEnvironment):
        await resolve_environment_at(store, registered.id, tenant, revision)
    with pytest.raises(UnknownEnvironment):
        await resolve_environment_at(store, registered.id, TenantId(uuid4()), 1)


async def test_a_retired_shape_still_resolves_and_says_it_is_retired() -> None:
    """Retirement is a fact on the resource and not a deletion.

    Refusing the read instead would make an Environment archived by mistake unreadable,
    which is exactly the moment somebody needs to see what they had. What retirement
    refuses is a new Session and an edit, and both of those are decided by the routes
    that do them -- so this function reports and does not judge.
    """
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    registered = an_environment(tenant)
    await store.insert(registered)
    retired_at = await store.archive(registered.id, tenant)

    resolved = await resolve_environment_revision(store, registered.id, tenant)

    assert resolved.archived
    assert resolved.archived_at == retired_at
    assert resolved.environment == registered


@pytest.mark.parametrize(
    ("column", "value", "complaint"),
    [
        ("revision", 0, "not a revision"),
        ("revision", "2", "not a revision"),
        ("archived_at", "2026-08-24T12:00:00Z", "not a time"),
    ],
)
async def test_a_row_whose_revision_or_retirement_is_not_one_is_a_fault(
    column: str, value: object, complaint: str
) -> None:
    """A row and the rules disagreeing is a fault, the same way a bad `denied_paths` is.

    A revision of 0 cannot exist -- the table has a check constraint refusing it -- and
    a revision of `"2"` is a driver that stopped parsing integers. Either one reaching a
    caller would be a number written into a Session's pin that no row can be found by
    later. An `archived_at` arriving as text is the same failure on the column that
    decides whether this shape may be used at all.
    """

    class OneWrongColumn:
        async def insert(self, environment: Environment, /) -> None:
            raise AssertionError("this store only reads")

        async def fetch(
            self, environment_id: EnvironmentId, tenant_id: TenantId, /
        ) -> Mapping[str, object] | None:
            return {
                "id": str(environment_id),
                "tenant_id": str(tenant_id),
                "name": "analysis",
                "runtime_image": IMAGE,
                "denied_paths": [SECRETS],
                "allowed_domains": [],
                "revision": 1,
                "archived_at": None,
                column: value,
            }

    with pytest.raises(ValueError, match=complaint):
        await resolve_environment_revision(
            OneWrongColumn(), new_environment_id(), TenantId(uuid4())
        )


async def test_an_id_nothing_registered_does_not_resolve() -> None:
    with pytest.raises(UnknownEnvironment):
        await resolve_environment(
            FakeEnvironmentStore(), new_environment_id(), TenantId(uuid4())
        )


async def test_another_tenants_id_refuses_with_the_same_message_shape() -> None:
    """One refusal for "no such id" and "not yours". Two would let anybody holding an id
    learn from the refusal whether it names somebody else's shape."""
    store = FakeEnvironmentStore()
    theirs = an_environment(TenantId(uuid4()))
    await store.insert(theirs)
    mine = TenantId(uuid4())

    with pytest.raises(UnknownEnvironment) as hidden:
        await resolve_environment(store, theirs.id, mine)
    with pytest.raises(UnknownEnvironment) as absent:
        await resolve_environment(store, new_environment_id(), mine)

    assert str(hidden.value) == str(theirs.id)
    assert str(absent.value) != str(theirs.id)
    assert type(hidden.value) is type(absent.value)


async def test_a_stored_row_whose_paths_are_not_a_list_is_a_fault_not_a_refusal() -> (
    None
):
    """A row and the rules disagreeing is a fault: handing it back would put a pod in a
    shape the platform would not register today."""

    class WrongShapeStore:
        async def insert(self, environment: Environment, /) -> None:
            raise AssertionError("this store only reads")

        async def fetch(
            self, environment_id: EnvironmentId, tenant_id: TenantId, /
        ) -> Mapping[str, object] | None:
            return {
                "id": str(environment_id),
                "tenant_id": str(tenant_id),
                "name": "analysis",
                "runtime_image": IMAGE,
                "denied_paths": SECRETS,
            }

    with pytest.raises(ValueError, match="not a list"):
        await resolve_environment(
            WrongShapeStore(), new_environment_id(), TenantId(uuid4())
        )


def test_the_stored_columns_are_the_migration_columns() -> None:
    """A rename on either side is a KeyError at the first resolve; this is where it
    lands. Both directions, so a column the migration grows and the port never learned
    about fails here too."""
    declared = set(
        re.findall(r"""sa\.Column\(\s*["']([^"']+)["']""", MIGRATION.read_text())
    )
    assert set(ENVIRONMENT_COLUMNS) <= declared
    assert declared - set(ENVIRONMENT_COLUMNS) == {"created_at_ms"}


# --------------------------------------------------------------------------------------
# MAP-A123: one shape, two Sessions, and a create call that cannot restate it
# --------------------------------------------------------------------------------------


def _without_the_token(compiled: CompiledConfig) -> str:
    """Every line of the user document except the one naming this Session's token.

    Two Sessions of one Environment run in one *shape*, and their identities differ --
    which the case below already asserts on `session_id` one line earlier. Comparing
    the documents line by line except for that one line is the stronger claim: it pins
    that the token is the only thing that differs, where plain equality could only ever
    have pinned that nothing does.
    """
    return "\n".join(
        line
        for line in compiled.config_toml.splitlines()
        if not line.startswith("http_headers =")
    )


async def test_two_sessions_naming_one_environment_run_in_one_shape() -> None:
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    registered = an_environment(tenant)
    await store.insert(registered)

    first_read = await resolve_environment(store, registered.id, tenant)
    second_read = await resolve_environment(store, registered.id, tenant)
    first = compile_session_config(
        a_record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        environment=first_read,
        definition=A_DEFINITION,
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )
    second = compile_session_config(
        a_record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        environment=second_read,
        definition=A_DEFINITION,
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )

    assert first.session_id != second.session_id
    assert first.runtime_image == second.runtime_image == IMAGE
    assert first.requirements_toml == second.requirements_toml
    assert _without_the_token(first) == _without_the_token(second)
    assert first.config_toml != second.config_toml, (
        "the two documents are identical, so the token is not in them"
    )
    assert first.launch_argv == second.launch_argv
    rules = _filesystem_rules(first)
    assert rules[SECRETS] == "deny"
    for already_denied in ALREADY_DENIED:
        assert rules[already_denied] == "deny", "the compiler stopped denying this"


async def test_a_second_shape_with_another_image_compiles_to_another_image() -> None:
    """So the equality above is a property of the shape and not of the compiler."""
    store = FakeEnvironmentStore()
    tenant = TenantId(uuid4())
    one = an_environment(tenant)
    other = an_environment(tenant, image=OTHER_IMAGE, denied_paths=())
    await store.insert(one)
    await store.insert(other)

    compiled_one = compile_session_config(
        a_record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=await resolve_environment(store, one.id, tenant),
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )
    compiled_other = compile_session_config(
        a_record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=await resolve_environment(store, other.id, tenant),
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )

    assert compiled_one.runtime_image != compiled_other.runtime_image
    assert compiled_other.runtime_image == OTHER_IMAGE


def test_a_shape_that_denies_nothing_adds_no_rule_to_the_platforms_own() -> None:
    """The narrowing is the only thing an environment contributes to the profile, so a
    shape that narrows nothing must leave the compiled rules exactly as they were."""
    compiled = compile_session_config(
        a_record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=an_environment(TenantId(uuid4()), denied_paths=()),
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )

    assert _filesystem_rules(compiled) == {
        rule.path: rule.access.value for rule in session_profile().rules
    }


def test_a_create_call_cannot_restate_the_sandbox_configuration() -> None:
    fields = set(CreateSession.model_fields)
    assert "environment_id" in fields
    assert not fields & {
        "runtime_image",
        "image",
        "denied_paths",
        "permission_profile",
        "sandbox",
    }
    with pytest.raises(ValidationError):
        CreateSession.model_validate(
            {
                "definition_id": str(new_definition_id()),
                "environment_id": str(new_environment_id()),
                "budget_minor_units": 10_000,
                "budget_currency": "USD",
                "retention_days": 30,
                "runtime_image": IMAGE,
            }
        )


def _filesystem_rules(compiled: CompiledConfig) -> dict[str, Any]:
    """The profile's filesystem table, read back out of the rendered document.

    Read from the document rather than from the profile object, because what the sandbox
    enforces is what the runtime loads and not what this process assembled.
    """
    parsed = tomllib.loads(compiled.requirements_toml)
    rules: dict[str, Any] = parsed["permissions"][PROFILE_NAME]["filesystem"]
    return rules


# --------------------------------------------------------------------------------------
# The two routes
# --------------------------------------------------------------------------------------


async def test_a_registered_shape_reads_back_as_it_was_registered() -> None:
    tenant = TenantId(uuid4())
    app = build_app(a_harness().platform)

    async with caller(app, tenant) as client:
        created = await client.post(
            "/v1/environments",
            json={
                "name": "analysis",
                "runtime_image": IMAGE,
                "denied_paths": [SECRETS],
            },
        )
        assert created.status_code == 201, created.text
        read = await client.get(f"/v1/environments/{created.json()['id']}")

    assert read.status_code == 200, read.text
    assert read.json() == {
        "id": created.json()["id"],
        "name": "analysis",
        "runtime_image": IMAGE,
        "denied_paths": [SECRETS],
        # Empty because this shape asked for no domain, and empty is no network at all
        # rather than an unrestricted default. Asserted as part of the whole body, so a
        # field that started defaulting to something permissive fails here.
        "allowed_domains": [],
        # A create writes the first revision and nothing else can: this route mints the
        # id, so there is no earlier row for it to land after.
        "revision": 1,
        # Null and not absent. Absent would leave a consumer's own default to say
        # whether this shape is retired; null says it is not.
        "archived_at": None,
    }


async def test_a_read_back_shape_names_no_tenant() -> None:
    """The caller is the tenant, so echoing one buys nothing and leaks an id."""
    tenant = TenantId(uuid4())
    harness = a_harness()
    app = build_app(harness.platform)

    async with caller(app, tenant) as client:
        created = await client.post(
            "/v1/environments", json={"name": "analysis", "runtime_image": IMAGE}
        )
        read = await client.get(f"/v1/environments/{created.json()['id']}")

    assert "tenant" not in read.text.lower()
    assert str(tenant) not in read.text


async def test_an_unpinned_image_is_refused_and_writes_no_row() -> None:
    harness = a_harness()
    app = build_app(harness.platform)

    async with caller(app, TenantId(uuid4())) as client:
        refused = await client.post(
            "/v1/environments",
            json={"name": "analysis", "runtime_image": "registry/session:latest"},
        )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert refused.json()["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert harness.store.inserted == [], "a refused registration reached the store"


async def test_a_path_outside_the_writable_root_is_refused_and_names_the_path() -> None:
    """The message is the deliverable here: a tenant told only "invalid" would have to
    guess which of its paths the platform would not let it narrow."""
    harness = a_harness()
    app = build_app(harness.platform)

    async with caller(app, TenantId(uuid4())) as client:
        refused = await client.post(
            "/v1/environments",
            json={
                "name": "analysis",
                "runtime_image": IMAGE,
                "denied_paths": ["/etc/codex"],
            },
        )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert refused.json()["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert "/etc/codex" in refused.json()["error"]["message"]
    assert harness.store.inserted == []


async def test_reading_an_id_nobody_registered_is_the_published_not_found() -> None:
    app = build_app(a_harness().platform)

    async with caller(app, TenantId(uuid4())) as client:
        missing = await client.get(f"/v1/environments/{new_environment_id()}")

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "environment.not_found"


async def test_reading_another_tenants_shape_is_the_identical_answer() -> None:
    """Byte-for-byte, not merely the same status: a message naming the shape would make
    the refusal an existence oracle just as surely as a different status would."""
    store = FakeEnvironmentStore()
    theirs = an_environment(TenantId(uuid4()))
    await store.insert(theirs)
    app = build_app(a_harness(store).platform)

    async with caller(app, TenantId(uuid4())) as client:
        hidden = await client.get(f"/v1/environments/{theirs.id}")
        absent = await client.get(f"/v1/environments/{new_environment_id()}")

    assert (hidden.status_code, _without_request_id(hidden)) == (
        absent.status_code,
        _without_request_id(absent),
    )


async def test_a_request_with_no_tenant_reaches_no_store() -> None:
    harness = a_harness()
    app = build_app(harness.platform)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://platform"
    ) as client:
        registered = await client.post(
            "/v1/environments", json={"name": "analysis", "runtime_image": IMAGE}
        )
        read = await client.get(f"/v1/environments/{new_environment_id()}")

    assert registered.status_code == 400
    assert registered.json()["error"]["code"] == "request.tenant_missing"
    assert read.status_code == 400
    assert harness.store.touched() == [], (
        "a request that never said who it was still reached the store"
    )


def test_the_environment_surface_is_exactly_the_six_operations_it_claims() -> None:
    """The inventory, pinned as an equality rather than as a ceiling.

    This test used to assert the opposite -- that no verb here could edit or remove a
    shape -- on the argument that a shape is immutable by construction. That argument
    survived the mechanism it rested on: an edit now appends a revision instead of
    rewriting one, so an id still means one shape for every Session already naming it
    and an edit has somewhere to go.

    Equality and not `<=`, because the failure this replaces was a ceiling passing a
    surface that had grown. A seventh operation appearing here is either a deliberate
    addition somebody records in this set, or an `/unarchive` route somebody added
    without noticing that retirement is terminal -- and a subset assertion cannot tell
    those two apart.
    """
    offered = {
        (method, route.path)
        for route in environments.router.routes
        if isinstance(route, APIRoute)
        for method in route.methods or ()
        # HEAD and OPTIONS are the framework's, not this module's.
        if method in {"GET", "POST", "DELETE", "PUT", "PATCH"}
    }

    assert offered == {
        ("POST", "/environments"),
        ("GET", "/environments"),
        ("GET", "/environments/{environment_id}"),
        ("POST", "/environments/{environment_id}"),
        ("POST", "/environments/{environment_id}/archive"),
        ("DELETE", "/environments/{environment_id}"),
    }


# --------------------------------------------------------------------------------------
# MAP-A124: an unknown id, and the collaborators that were never touched
# --------------------------------------------------------------------------------------


async def test_an_unknown_environment_writes_no_session_and_appends_nothing() -> None:
    harness = a_harness()
    app = build_app(harness.platform)

    async with caller(app, TenantId(uuid4())) as client:
        refused = await client.post(
            "/v1/sessions", json=a_create_body(new_environment_id())
        )

    assert refused.status_code == 404
    assert refused.json()["error"]["code"] == "environment.not_found"
    assert harness.log.appended == [], "a refused create left an event behind"
    assert harness.registry.created == [], "a refused create left a Session record"


async def test_another_tenants_environment_refuses_a_create_as_an_absent_one() -> None:
    """The two answers must be one answer, or the refusal enumerates other tenants'
    ids."""
    store = FakeEnvironmentStore()
    not_mine = an_environment(TenantId(uuid4()))
    await store.insert(not_mine)
    harness = a_harness(store)
    app = build_app(harness.platform)

    async with caller(app, TenantId(uuid4())) as client:
        absent = await client.post(
            "/v1/sessions", json=a_create_body(new_environment_id())
        )
        hidden = await client.post("/v1/sessions", json=a_create_body(not_mine.id))

    assert (hidden.status_code, _without_request_id(hidden)) == (
        absent.status_code,
        _without_request_id(absent),
    )
    assert harness.log.appended == []
    assert harness.registry.created == []


async def test_a_create_naming_a_registered_shape_records_that_shape_once() -> None:
    """The happy path the two refusals above are measured against.

    Without it every "nothing was written" assertion is satisfied by a create path that
    never writes anything at all.
    """
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    mine = an_environment(tenant)
    await store.insert(mine)
    harness = a_harness(store)
    app = build_app(harness.platform)

    async with caller(app, tenant) as client:
        created = await client.post("/v1/sessions", json=a_create_body(mine.id))

    assert created.status_code == 201, created.text
    assert len(harness.log.appended) == 1
    (event,) = harness.log.appended
    assert event.type == "session.created"
    assert event.payload["environment_id"] == str(mine.id)
    assert event.payload["environment_revision"] == 1
    assert len(harness.registry.created) == 1


async def test_a_create_pins_the_environment_revision_it_resolved() -> None:
    """The id alone stopped being enough the moment an Environment could be edited.

    Two Sessions naming one id across an edit would otherwise run in different sandboxes
    with nothing anywhere recording that they differed -- which is the invariant this
    table's append-only trigger was installed to protect, lost by the feature that made
    the table appendable. The number in the payload is what keeps it: 2 here, because
    the edit landed before the create, and it stays 2 for this Session forever.
    """
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    mine = an_environment(tenant)
    await store.insert(mine)
    await store.insert_revision(
        parse_environment(
            environment_id=mine.id,
            tenant_id=tenant,
            name="analysis",
            runtime_image=OTHER_IMAGE,
            denied_paths=(SECRETS,),
        )
    )
    harness = a_harness(store)
    app = build_app(harness.platform)

    async with caller(app, tenant) as client:
        created = await client.post("/v1/sessions", json=a_create_body(mine.id))

    assert created.status_code == 201, created.text
    (event,) = harness.log.appended
    assert event.payload["environment_revision"] == 2, (
        "the create resolved the latest revision and wrote down the wrong number, so "
        "an audit of what this Session was allowed to do reads the wrong shape"
    )


async def test_a_create_in_a_retired_environment_writes_nothing() -> None:
    """Retirement's whole content is "no new Session in this shape", so the refusal has
    to reach the one route that starts one -- and before either write, or a Session
    exists that folds to RUNNING and may never be placed."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    retired = an_environment(tenant)
    await store.insert(retired)
    await store.archive(retired.id, tenant)
    harness = a_harness(store)
    app = build_app(harness.platform)

    async with caller(app, tenant) as client:
        refused = await client.post("/v1/sessions", json=a_create_body(retired.id))

    assert refused.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_ARCHIVED]
    assert refused.json()["error"]["code"] == ErrorCode.ENVIRONMENT_ARCHIVED.value
    assert harness.log.appended == [], "a refused create left an event behind"
    assert harness.registry.created == [], "a refused create left a Session record"


async def test_a_create_body_with_no_environment_is_refused_before_any_store() -> None:
    """Refused at the boundary as a 400 naming the field, not as a domain refusal: a
    create that could omit the shape would be a Session running in an unnamed one."""
    harness = a_harness()
    app = build_app(harness.platform)
    body = a_create_body(new_environment_id())
    del body["environment_id"]

    async with caller(app, TenantId(uuid4())) as client:
        refused = await client.post("/v1/sessions", json=body)

    assert refused.status_code == 400
    assert "environment_id" in refused.text
    assert harness.store.touched() == []
    assert harness.log.appended == []
    assert harness.registry.created == []


def test_nothing_on_the_create_path_places_a_pod_yet() -> None:
    """The other half of MAP-A124's claim, and the honest form of it.

    "No pod is placed" cannot be measured from a recorder today, because this create
    path has no placement call at all and `Platform` carries no placement port -- a
    recorder asserted empty would pass for the whole life of the slice without ever
    being able to
    fail. So the claim is held structurally instead: the day a placement call is added
    to this handler, this fails, and whoever adds it has to make the recorder real and
    assert the refusal comes first.
    """
    source = SESSIONS_ROUTE.read_text()

    assert "placement" not in source.lower() and ".place(" not in source, (
        "the Session create path now reaches placement. Add a recording placement port "
        "to the harness in this file and assert it stays empty on the two refusals "
        "above -- until then nothing proves an unknown environment places no pod."
    )
