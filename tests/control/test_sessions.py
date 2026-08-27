"""Create a Session with one call, then read its state back out of its own log.

Tier 1 (local, in-memory ports). The routes are driven over real HTTP against the real
app, with the log ports replaced by an in-memory pair — the point being that nothing on
the read path consults anything but the log.

Four properties are graded. One call creates a Session and hands back an addressable id.
A read folds the log rather than reading a stored value, shown by appending a lifecycle
event behind the API's back and watching the next read change. A read writes nothing, so
there is no repair path that could quietly become a second source. And the response
carries none of the Agent Runtime's own identifiers.
"""

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.sessions import SessionCreated, SessionView
from managed_agent.control.files.store import unconfigured_file_store
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
from managed_agent.core.ports import (
    Resolution,
    SessionListing,
    SessionNotVisible,
)
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord, SessionState
from managed_agent.core.vocabulary import lifecycle, turn

_MIGRATIONS = Path(__file__).parents[2] / "migrations" / "versions"

# Creating a Session resolves its definition, so every create here needs a tenant and a
# definition that resolves. Both are supplied once, at the client and the platform,
# rather than at each call site: these tests are about the Event Log fold, and a
# definition id threaded through twenty assertions would read as though it mattered to
# them.
_TENANT = TenantId(uuid4())
_HEADERS = {TENANT_HEADER: str(_TENANT)}
_SKILLS_SHA = "0" * 39 + "a"


def _a_definition() -> dict[str, object]:
    return {
        "name": "fold-fixture",
        "instructions": "irrelevant to these tests",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SKILLS_SHA,
    }


@dataclass(frozen=True, slots=True)
class _Resolved:
    definition: AgentDefinition
    revision: int


class AlwaysResolves:
    """A registry that resolves any id to revision 1, for tests about the log.

    Deliberately unable to refuse. Whether an unknown definition is refused, and whether
    a refusal appends anything, are graded in `test_registration_definitions.py` where
    the registry can say no — asserting it here as well would put one behaviour in two
    places, free to disagree the first time the refusal changes shape.
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


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class InMemoryLog:
    """Both log ports over one dict, and it counts its own appends.

    The count is what lets a test assert that reading a Session writes nothing. A read
    that repaired or cached a stored state would append or mutate here, and a fake that
    only answered reads could not tell us that.
    """

    def __init__(self) -> None:
        self._events: dict[SessionId, list[Event]] = {}
        self.appends = 0

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.appends += 1
        return self.add(session_id, type_, payload)

    def add(
        self,
        session_id: SessionId,
        type_: str,
        payload: dict[str, object] | None = None,
    ) -> Seq:
        """Append without going through the port, to stand in for another writer.

        Seven components append to one Session's log, so a state read has to be correct
        about events this process never wrote. This is how a test produces one.
        """
        log = self._events.setdefault(session_id, [])
        seq = Seq(len(log) + 1)
        log.append(Event(session_id, seq, type_, dict(payload or {})))
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
        return FIRST_SEQ


class CappedLog(InMemoryLog):
    """An in-memory log that returns at most `cap` rows per read, as the real one does.

    The Postgres adapter caps a range read and documents a short result as "the range
    is exhausted, so page". The port now says so too — it did not when this class was
    written, and a caller reading the port alone therefore could not know to page, which
    is exactly how a fold over a whole log silently became a fold over its first page.
    This class stays because a stated contract is not an honoured one: it is what makes
    a caller that ignores the cap fail here rather than in production.
    """

    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap
        self.reads = 0

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Event]:
        self.reads += 1
        return (await super().read(session_id, start, end, limit))[: self.cap]


@pytest.fixture
def log() -> InMemoryLog:
    return InMemoryLog()


@pytest.fixture
def client(log: InMemoryLog) -> TestClient:
    return TestClient(
        create_app(
            Platform(
                event_log_append=log,
                event_log_range=log,
                definition_registry=AlwaysResolves(),
                tool_registry=UnusedToolRegistry(),
                session_registry=InMemorySessionRegistry(),
                webhooks=UnusedWebhooks(),
                environment_store=AnyEnvironmentResolves(),
                turn_dispatch=NoPodTransport(),
                file_store=unconfigured_file_store(),
            )
        ),
        headers=_HEADERS,
    )


def _create_body() -> dict[str, object]:
    return {
        "definition_id": str(uuid4()),
        # Any id resolves through this file's environment store. Which sandbox shape a
        # Session runs in is graded in `test_environment_reference.py`; these cases are
        # about the fold, and a real shape threaded through them would read as though it
        # mattered to it. The one case below that wires the real adapters registers one.
        "environment_id": str(uuid4()),
        "grant": [],
        "scope": {"repo": "acme/widgets"},
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 30,
    }


def test_one_call_creates_a_session_and_hands_back_its_id(
    client: TestClient, log: InMemoryLog
) -> None:
    response = client.post("/v1/sessions", json=_create_body())

    assert response.status_code == 201
    created = response.json()
    assert UUID(created["id"])
    assert created["state"] == "idle"
    assert created["seq"] == 1
    assert log.appends == 1


def test_creation_writes_exactly_one_published_lifecycle_event(
    client: TestClient, log: InMemoryLog
) -> None:
    """The Session exists because the log says so — by one event, not a record too."""
    created = client.post("/v1/sessions", json=_create_body()).json()
    session_id = SessionId(UUID(created["id"]))

    written = log._events[session_id]
    assert [event.type for event in written] == [lifecycle.SESSION_CREATED]
    assert written[0].payload["budget_currency"] == "USD"
    # Empty, and this file cannot honestly assert otherwise: every name in a Grant is
    # resolved against the tenant's registrations at creation now, and this file's
    # registry double raises on sight. That the Grant travels from the request into the
    # event with real names in it is asserted where a real registry exists, in
    # `test_a_grant_names_a_registered_tool.py`.
    assert written[0].payload["grant"] == []


def test_a_created_session_reads_back_as_idle(client: TestClient) -> None:
    session_id = client.post("/v1/sessions", json=_create_body()).json()["id"]

    response = client.get(f"/v1/sessions/{session_id}")

    assert response.status_code == 200
    assert response.json() == {"id": session_id, "state": "idle", "seq": 1}


def test_a_read_reports_what_the_log_says_and_not_what_creation_returned(
    client: TestClient, log: InMemoryLog
) -> None:
    """An event appended behind the API's back changes the next read.

    This is the whole claim: had the state been stored at creation, the read would still
    say idle, because nothing went back and updated a column.
    """
    session_id = client.post("/v1/sessions", json=_create_body()).json()["id"]
    log.add(SessionId(UUID(session_id)), turn.TURN_SUBMITTED)

    body = client.get(f"/v1/sessions/{session_id}").json()

    assert body["state"] == "running"
    assert body["seq"] == 2


def test_reading_a_session_writes_nothing(client: TestClient, log: InMemoryLog) -> None:
    """No repair, no cache write-back: a read has no path that could become a source."""
    session_id = client.post("/v1/sessions", json=_create_body()).json()["id"]
    appends_after_creation = log.appends

    for _ in range(3):
        assert client.get(f"/v1/sessions/{session_id}").status_code == 200

    assert log.appends == appends_after_creation


def test_neither_response_carries_a_runtime_identifier(client: TestClient) -> None:
    """The runtime names its own units `thread_id` and `parent_thread_id`; a tenant sees
    neither, and the id it does see is this platform's own uuid rather than a runtime
    handle (ADR-007)."""
    created = client.post("/v1/sessions", json=_create_body()).json()
    read = client.get(f"/v1/sessions/{created['id']}").json()

    for body in (created, read):
        rendered = repr(body)
        assert "thread" not in rendered.lower()
        assert not re.search(r"\b(thread|sess|msg|run|asst|conv)_", rendered)
        assert UUID(body["id"])


_NOT_A_SESSIONS_STATE = {
    # MAP-51's webhook tables, as `0016_webhooks.py` first declared them. Neither column
    # held what this check exists to forbid -- a Session's *current* state, cached where
    # the fold's answer could disagree with it. `webhook.states` was a registration's
    # filter: the set of states a tenant asked to hear about, which belonged to the
    # registration and to no Session. `state` on `webhook_delivery` recorded which
    # change had been called back.
    #
    # Both are gone: `0030_webhooks_by_event_type.py` renames them to `event_types` and
    # `event_type`, so a registration names what happened rather than what it meant and
    # nothing on this side of the platform mentions a state at all. The entries stay
    # because this check reads every migration source including the one that created
    # them, and the second assertion below fails on an entry whose column no revision
    # declares.
    "0016_webhooks.py:states",
    "0016_webhooks.py:state",
}
"""Columns whose name trips the check below and which are not a Session's state.

An inventory, not an allowance. The check asserts this is *exactly* what is outstanding,
so a new state column still fails until somebody writes down why it is not one, and an
entry whose column is gone fails until it is removed. Either direction fails, which is
what keeps the list from becoming a place to hide a cache.
"""


def test_no_migration_declares_a_state_column() -> None:
    """The schema has nowhere to keep a state, which is what leaves the log alone.

    Asserted over the migration sources rather than a live database, so a column a later
    revision adds is caught before anyone runs it.

    Both spellings are read, because this project uses both: `sa.Column(...)` in a
    `create_table`, and bare SQL in an `op.execute` where alembic has no helper for what
    is being done. A check that knew only the first would pass a `state` column added by
    the second.

    The "did the pattern match anything" guard is over all the migrations together, not
    each one. A revision that only converts a type or adds an index declares no column
    at
    all and is perfectly normal; a pattern that matches nothing anywhere is a broken
    pattern reporting a clean bill of health.
    """
    sources = list(_MIGRATIONS.glob("*.py"))
    assert sources, "found no migrations to check"

    corpus: list[tuple[str, str]] = []
    for source in sources:
        text = source.read_text()
        for name in re.findall(r"""Column\(\s*["']([^"']+)["']""", text):
            corpus.append((source.name, name))
        for name in re.findall(
            r"""ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?["']?(\w+)""",
            text,
            re.IGNORECASE,
        ):
            corpus.append((source.name, name))

    assert corpus, "matched no column in any migration — the patterns are broken"
    named = [
        f"{where}:{name}"
        for where, name in corpus
        if "state" in name.lower() or "status" in name.lower()
    ]
    offenders = sorted(set(named) - _NOT_A_SESSIONS_STATE)
    assert offenders == [], (
        f"a migration declares {offenders}. A Session's state is folded from its Event "
        "Log on every read, so a column holding one is a second source free to "
        "disagree with the log. If the column is genuinely not a Session's current "
        "state, argue it into _NOT_A_SESSIONS_STATE by name."
    )
    assert set(named) >= _NOT_A_SESSIONS_STATE, (
        "these columns are recorded as exceptions and no migration declares them any "
        f"more: {sorted(_NOT_A_SESSIONS_STATE - set(named))}. Remove them, so the list "
        "cannot outlive what it excuses."
    )


def test_the_fold_reads_past_a_capped_page_rather_than_stopping_at_it() -> None:
    """A port that returns one page per call must not truncate the state.

    The state and the sequence a tenant reads have to describe the whole log. A read
    that
    stopped at the first page would report a Session as running long after it stopped,
    and hand back a resume position hundreds of events behind the head — both silently,
    which is what makes this worth a test rather than a comment.
    """
    capped = CappedLog(cap=7)
    client = TestClient(
        create_app(
            Platform(
                event_log_append=capped,
                event_log_range=capped,
                definition_registry=AlwaysResolves(),
                tool_registry=UnusedToolRegistry(),
                session_registry=InMemorySessionRegistry(),
                webhooks=UnusedWebhooks(),
                environment_store=AnyEnvironmentResolves(),
                turn_dispatch=NoPodTransport(),
                file_store=unconfigured_file_store(),
            )
        ),
        headers=_HEADERS,
    )
    session_id = client.post("/v1/sessions", json=_create_body()).json()["id"]
    for _ in range(20):
        capped.add(SessionId(UUID(session_id)), "turn.completed")
    capped.add(SessionId(UUID(session_id)), lifecycle.SESSION_STOPPED)

    body = client.get(f"/v1/sessions/{session_id}").json()

    assert body["seq"] == 22
    assert body["state"] == "stopped"
    assert capped.reads > 1, "one read cannot have covered 22 events at a cap of 7"


def test_a_single_page_log_costs_one_read_beyond_the_page() -> None:
    """Paging stops as soon as a page comes back empty, so a short log stays cheap."""
    capped = CappedLog(cap=500)
    client = TestClient(
        create_app(
            Platform(
                event_log_append=capped,
                event_log_range=capped,
                definition_registry=AlwaysResolves(),
                tool_registry=UnusedToolRegistry(),
                session_registry=InMemorySessionRegistry(),
                webhooks=UnusedWebhooks(),
                environment_store=AnyEnvironmentResolves(),
                turn_dispatch=NoPodTransport(),
                file_store=unconfigured_file_store(),
            )
        ),
        headers=_HEADERS,
    )
    session_id = client.post("/v1/sessions", json=_create_body()).json()["id"]
    capped.reads = 0

    assert client.get(f"/v1/sessions/{session_id}").json()["seq"] == 1
    assert capped.reads == 2


def test_a_response_refuses_a_sequence_below_the_first() -> None:
    """The sequence constraint is enforced here, at the model, and nowhere else.

    `Seq` is a pydantic annotation, so it bites where a model validates a field of that
    type and not where the name is written as a call — `Seq(0)` is `int(0)`. The read
    path
    casts on its way out and does not check, which is sound only because the store
    refuses
    a sequence below 1 and an empty fold raises. This test is the other half of that: if
    a zero ever did reach the boundary, it stops here rather than reaching a tenant as a
    resume position that no event has.
    """
    with pytest.raises(ValidationError):
        SessionView(id=new_session_id(), state=SessionState.RUNNING, seq=0)
    with pytest.raises(ValidationError):
        SessionCreated(id=new_session_id(), state=SessionState.RUNNING, seq=0)


async def test_a_log_longer_than_one_adapter_page_folds_whole(
    database_url: str,
) -> None:
    """The real adapter caps a read; the fold must still see the last event.

    Tier 1 (testcontainers, real PostgreSQL 17). The capped-fake test above proves the
    paging logic; this proves it against the cap that actually exists, which is the one
    that matters — the defect this guards was invisible precisely because an in-memory
    fake returns everything it holds and so can never truncate. The Session is stopped
    by
    its final event, so a fold that stopped at the first page would report it running.
    """
    platform, engine = build(database_url)
    try:
        app = create_app(platform)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://tenant",
            headers=_HEADERS,
        ) as client:
            # The real registry is wired here, so the definition has to really exist —
            # registered through its own route rather than inserted behind it, so this
            # test breaks if registration and resolution stop agreeing.
            registered = await client.post("/v1/agents", json=_a_definition())
            assert registered.status_code == 201, registered.text
            # Same reasoning for the sandbox shape: the real store is wired here, so the
            # id the create names has to be one the platform issued.
            shape = await client.post(
                "/v1/environments",
                json={"name": "fold-fixture", "runtime_image": FIXTURE_IMAGE},
            )
            assert shape.status_code == 201, shape.text
            created = await client.post(
                "/v1/sessions",
                json=_create_body()
                | {
                    "definition_id": registered.json()["id"],
                    "environment_id": shape.json()["id"],
                },
            )
            session_id = created.json()["id"]
            typed_id = SessionId(UUID(session_id))

            for _ in range(600):
                await platform.event_log_append.append(typed_id, "turn.completed", {})
            last = await platform.event_log_append.append(
                typed_id, lifecycle.SESSION_STOPPED, {}
            )

            body = (await client.get(f"/v1/sessions/{session_id}")).json()

        assert last == 602
        assert body["seq"] == 602, "the fold stopped short of the head of the log"
        assert body["state"] == "stopped"
    finally:
        await engine.dispose()


class HeldPods:
    """A stand-in cluster holding one pod per Session, so a handback is an absence.

    Not a mock and not a spy: `held` is the measurement, and every case below asks it
    what is left rather than asking what was called. An assertion that `release` ran
    would stay green with a no-op behind it, and green if the route handed back some
    other Session's pod.
    """

    def __init__(self) -> None:
        self.held: set[SessionId] = set()

    async def release(self, session_id: SessionId) -> None:
        self.held.discard(session_id)


@pytest.fixture
def pods() -> HeldPods:
    return HeldPods()


@pytest.fixture
def archiving_client(log: InMemoryLog, pods: HeldPods) -> TestClient:
    """The same app as `client`, wired with a cluster that really holds pods.

    A second fixture rather than a field on the first, so the twenty cases above keep
    building a `Platform` with no release at all -- which is what a control plane that
    places no pods is wired with, and is the shape whose default must not raise.
    """
    return TestClient(
        create_app(
            Platform(
                event_log_append=log,
                event_log_range=log,
                definition_registry=AlwaysResolves(),
                tool_registry=UnusedToolRegistry(),
                session_registry=InMemorySessionRegistry(),
                webhooks=UnusedWebhooks(),
                environment_store=AnyEnvironmentResolves(),
                turn_dispatch=NoPodTransport(),
                file_store=unconfigured_file_store(),
                session_pod_release=pods,
            )
        ),
        headers=_HEADERS,
    )


def _a_placed_session(client: TestClient, pods: HeldPods) -> str:
    """A created Session whose pod the cluster is holding."""
    session_id = client.post("/v1/sessions", json=_create_body()).json()["id"]
    pods.held.add(SessionId(UUID(session_id)))
    return str(session_id)


def test_archiving_stops_the_session_and_gives_its_pod_back(
    archiving_client: TestClient, log: InMemoryLog, pods: HeldPods
) -> None:
    session_id = _a_placed_session(archiving_client, pods)

    response = archiving_client.post(f"/v1/sessions/{session_id}/archive")

    assert response.status_code == 200, response.text
    assert response.json() == {"id": session_id, "state": "stopped", "seq": 2}
    assert pods.held == set(), "the pod was not handed back"
    written = log._events[SessionId(UUID(session_id))]
    assert [event.type for event in written] == [
        lifecycle.SESSION_CREATED,
        lifecycle.SESSION_STOPPED,
    ]
    assert written[-1].payload == {"stop_reason": "archived"}


def test_an_archived_session_reads_back_as_stopped(
    archiving_client: TestClient, pods: HeldPods
) -> None:
    """The read route folds the log, so it sees the archive without being told."""
    session_id = _a_placed_session(archiving_client, pods)

    archiving_client.post(f"/v1/sessions/{session_id}/archive")

    assert archiving_client.get(f"/v1/sessions/{session_id}").json()["state"] == (
        "stopped"
    )


def test_archiving_twice_answers_the_same_and_appends_once(
    archiving_client: TestClient, log: InMemoryLog, pods: HeldPods
) -> None:
    session_id = _a_placed_session(archiving_client, pods)

    first = archiving_client.post(f"/v1/sessions/{session_id}/archive")
    appends = log.appends
    second = archiving_client.post(f"/v1/sessions/{session_id}/archive")

    assert first.json() == second.json()
    assert log.appends == appends, "the retry appended a second stop"


def test_a_running_turn_refuses_the_archive_with_a_coded_conflict(
    archiving_client: TestClient, log: InMemoryLog, pods: HeldPods
) -> None:
    """The refusal names the Turn, so a caller knows what to interrupt.

    The code is not a member of the published closed set and this asserts the value the
    route actually sends rather than an enum member; `test_closed_error_set.py` is what
    keeps that value inventoried with its reason.
    """
    session_id = _a_placed_session(archiving_client, pods)
    turn_id = uuid4()
    log.add(
        SessionId(UUID(session_id)),
        "turn.submitted",
        {"turn_id": str(turn_id), "idempotency_key": "k" * 8, "prompt": "hello"},
    )

    response = archiving_client.post(f"/v1/sessions/{session_id}/archive")

    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "session.turn_in_flight"
    assert body["error"]["detail"]["turn_id"] == str(turn_id)
    assert pods.held == {SessionId(UUID(session_id))}, "a live Turn lost its pod"


def test_archiving_a_session_this_tenant_cannot_see_is_a_404(
    archiving_client: TestClient, log: InMemoryLog, pods: HeldPods
) -> None:
    """Ownership is settled before the fold, so no stranger's Session is stopped.

    The log is keyed by Session and carries no tenant, so a route that folded first
    would archive somebody else's Session and hand back its pod. One 404 covers absent
    and not-yours, so an id cannot be probed for which it is.
    """
    stranger = new_session_id()
    pods.held.add(stranger)
    appends = log.appends

    response = archiving_client.post(f"/v1/sessions/{stranger}/archive")

    assert response.status_code == 404
    assert log.appends == appends
    assert pods.held == {stranger}


def test_a_control_plane_that_places_no_pods_still_archives(
    client: TestClient, log: InMemoryLog
) -> None:
    """The default release does nothing, and doing nothing is right where there is
    nothing.

    This `Platform` is built with no `session_pod_release` at all, which is every
    process wired without a pod runner. A refusing default would make archiving fail
    there, so the field's default has to be the no-op, and this is the case saying so.
    """
    session_id = client.post("/v1/sessions", json=_create_body()).json()["id"]

    response = client.post(f"/v1/sessions/{session_id}/archive")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "stopped"


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


class InMemorySessionRegistry:
    """Records who owns a Session and hands it back to that tenant only.

    Enough of the port for this file, which creates Sessions and reads them: `page` is
    not exercised here and raises rather than answering, so a test that started listing
    would say so instead of quietly passing against a fake nobody had checked.
    """

    def __init__(self) -> None:
        self.records: dict[SessionId, SessionRecord] = {}

    async def create(self, record: SessionRecord) -> None:
        self.records[record.id] = record

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        found = self.records.get(session_id)
        if found is None or found.tenant_id != tenant_id:
            raise SessionNotVisible(str(session_id))
        return found

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("a test in this file listed Sessions")


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
        event_types: frozenset[str],
        secret_ref: str,
    ) -> WebhookRecord:
        raise AssertionError("a test in this file registered a webhook")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file listed a tenant's webhooks")

    async def delete(self, webhook_id: UUID, tenant_id: TenantId) -> bool:
        raise AssertionError("a test in this file deleted a webhook")

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file asked what watches a type")


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
