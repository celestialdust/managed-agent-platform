"""Deleting a Session closes it and keeps its history; updating one changes nothing.

Tier 1 (local, in-memory ports). Both routes are driven over real HTTP against the
real app, because what is graded is what a tenant observes: the status, the body,
what the log holds afterwards, and whether the pod is gone.

Three claims, and each is graded by an outcome rather than by a call record.

**A delete closes and does not erase.** `DELETE /v1/sessions/{id}` appends the one
stop this platform has and hands the pod back, and afterwards the Session's whole
Event Log is still readable at the same sequence numbers, the read route still
answers, and the list route still shows it. The route's docstring makes those
survivals an explicit promise, so they are asserted here rather than left to a
reader's trust -- a delete that started removing something fails these cases, which
is the point of writing them down.

**Nothing about a Session is updatable.** `POST /v1/sessions/{id}` refuses the Grant
and the Budget with a published code and a named reason, refuses by name a field this
platform has no store for, and answers an empty ask with the current state. Every
refusal is checked against the registry record and the log: a route that refused and
wrote anyway would pass a status assertion on its own.

**A closed Session has one decided answer to each of the next four calls.** The
matrix at the bottom is parametrized over both closing verbs crossed with all four
follow-up calls, because each cell is a decision and a cell nobody wrote down is
whatever the code happens to do. Two of the eight cells are a delete on an archived
Session and an archive on a deleted one; the other six are their neighbours, and
leaving those unstated is how the two that were stated drift.

**The pod is a real thing here.** `HeldPods` holds Session ids and a handback is
asserted as an absence from it, so a test passes only if the pod is actually gone. An
assertion that `release` was called would stay green with a no-op behind it.

**The log pages at two rows.** That is what the Postgres adapter does, and three
shipped defects in this repository came from a caller that folded the first page and
believed it had the whole log. Every fold behind these routes has to page to be
right, and at a cap of two it does not get the benefit of the doubt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.sessions import (
    REASON_BUDGET_NOT_REVISABLE,
    REASON_GRANT_NOT_REVISABLE,
    SESSION_TURN_IN_FLIGHT,
)
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import (
    FIRST_SEQ,
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    new_session_id,
    new_turn_id,
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

_TENANT = TenantId(uuid4())
_HEADERS = {TENANT_HEADER: str(_TENANT)}
_SKILLS_SHA = "0" * 39 + "a"
_LOG_PAGE = 2
"""Rows one range read returns, matching what the real adapter caps at."""


# --------------------------------------------------------------------------------------
# The ports, in memory. Each one either records what a route did to it or refuses.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class PagingLog:
    """Both log ports over one dict, capped per read as the Postgres adapter is.

    The append count is what lets a test say a refusal wrote nothing, and the cap is
    what makes a fold that ignores paging fail here instead of in production.
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
        """Append without going through the port, standing in for another writer.

        Seven components append to one Session's log, so a route has to be correct about
        events this process never wrote. This is how a test produces one.
        """
        log = self._events.setdefault(session_id, [])
        seq = Seq(len(log) + 1)
        log.append(Event(session_id, seq, type_, dict(payload or {})))
        return seq

    def rows(self, session_id: SessionId) -> list[tuple[int, str]]:
        """Every event of one Session as (sequence, type), to compare before/after."""
        return [(event.seq, event.type) for event in self._events.get(session_id, [])]

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int | None = None
    ) -> Sequence[Event]:
        """At most `_LOG_PAGE` rows when the caller took the default, else `limit` rows.

        `limit: int | None` rather than the port's `int = 500`, so that "took the
        default" is distinguishable here. A fake that capped every read regardless would
        break the settled re-read inside the archive transition, which names `limit=seq`
        precisely because that span is provably covered -- and the failure would look
        like the transition being wrong rather than the double.
        """
        span = [
            event
            for event in self._events.get(session_id, [])
            if start <= event.seq <= end
        ]
        return span[: _LOG_PAGE if limit is None else limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Event]:
        for event in self._events.get(session_id, []):
            if event.seq > after:
                yield event

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ


class HeldPods:
    """A cluster holding Session pods, so a handback is observable as an absence.

    Asserting on what is left rather than on what was called: a check that `release`
    ran stays green with a no-op behind it, and green if the route released some other
    Session's pod.
    """

    def __init__(self) -> None:
        self.held: set[SessionId] = set()

    async def release(self, session_id: SessionId) -> None:
        self.held.discard(session_id)


class InMemorySessionRegistry:
    """Who owns a Session, and one tenant's Sessions newest first.

    `page` is implemented rather than raising, because one claim in this file is that a
    closed Session is still listed -- and a fake that refused to list would make that
    claim untestable.
    """

    def __init__(self) -> None:
        self.records: dict[SessionId, SessionRecord] = {}
        self._created: list[SessionId] = []

    async def create(self, record: SessionRecord) -> None:
        self.records[record.id] = record
        self._created.append(record.id)

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        found = self.records.get(session_id)
        if found is None or found.tenant_id != tenant_id:
            raise SessionNotVisible(str(session_id))
        return found

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        rows = [
            _Listed(
                id=session_id,
                definition_id=self.records[session_id].definition_id,
                definition_revision=self.records[session_id].definition_revision,
                created_at_ms=position,
            )
            for position, session_id in enumerate(self._created)
            if self.records[session_id].tenant_id == tenant_id
        ]
        return list(reversed(rows))[:limit]


@dataclass(frozen=True, slots=True)
class _Listed:
    id: SessionId
    definition_id: DefinitionId
    definition_revision: str
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class _Resolved:
    definition: AgentDefinition
    revision: int


def _a_definition() -> dict[str, object]:
    return {
        "name": "close-fixture",
        "instructions": "irrelevant to these tests",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SKILLS_SHA,
    }


class AlwaysResolves:
    """Resolves any definition id to revision 1. Whether a refusal is right is graded
    where the registry can refuse, in `test_registration_definitions.py`."""

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


_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64


class AnyEnvironmentResolves:
    """Resolves any environment id to one fixed shape, and cannot refuse."""

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        return {
            "id": str(environment_id),
            "tenant_id": str(tenant_id),
            "name": "close-fixture",
            "runtime_image": _FIXTURE_IMAGE,
            "denied_paths": [],
        }


class UnusedToolRegistry:
    """Satisfies the port and raises, so reaching it fails rather than passing."""

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("a test in this file registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("a test in this file looked up a registered tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("a test in this file listed a tenant's tools")


class UnusedWebhooks:
    """Satisfies the port and raises, for the reason above."""

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


# --------------------------------------------------------------------------------------
# The app, and the one Session most cases start from.
# --------------------------------------------------------------------------------------


@pytest.fixture
def log() -> PagingLog:
    return PagingLog()


@pytest.fixture
def pods() -> HeldPods:
    return HeldPods()


@pytest.fixture
def registry() -> InMemorySessionRegistry:
    return InMemorySessionRegistry()


@pytest.fixture
def client(
    log: PagingLog, pods: HeldPods, registry: InMemorySessionRegistry
) -> TestClient:
    return TestClient(
        create_app(
            Platform(
                event_log_append=log,
                event_log_range=log,
                definition_registry=AlwaysResolves(),
                tool_registry=UnusedToolRegistry(),
                session_registry=registry,
                webhooks=UnusedWebhooks(),
                environment_store=AnyEnvironmentResolves(),
                turn_dispatch=NoPodTransport(),
                file_store=unconfigured_file_store(),
                session_pod_release=pods,
            )
        ),
        headers=_HEADERS,
    )


def _create_body() -> dict[str, object]:
    return {
        "definition_id": str(uuid4()),
        "environment_id": str(uuid4()),
        "grant": ["fs.read"],
        "scope": {"repo": "acme/widgets"},
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 30,
    }


def _a_placed_session(client: TestClient, log: PagingLog, pods: HeldPods) -> SessionId:
    """A created Session, holding a pod, with enough log to force a fold to page.

    Four events, at a page cap of two: a route that read one page would see the creation
    and one Turn event and miss everything after them, so every state read below has to
    page to be right.
    """
    created = client.post("/v1/sessions", json=_create_body()).json()
    session_id = SessionId(UUID(created["id"]))
    pods.held.add(session_id)
    turn_id = str(new_turn_id())
    log.add(session_id, turn.TURN_SUBMITTED, {"turn_id": turn_id})
    log.add(session_id, turn.TURN_STARTED, {"turn_id": turn_id})
    log.add(session_id, turn.TURN_COMPLETED, {"turn_id": turn_id})
    return session_id


def _open_a_turn(log: PagingLog, session_id: SessionId) -> str:
    turn_id = str(new_turn_id())
    log.add(
        session_id,
        turn.TURN_SUBMITTED,
        {"turn_id": turn_id, "idempotency_key": "k" * 8, "prompt": "hello"},
    )
    return turn_id


# --------------------------------------------------------------------------------------
# A delete closes the Session and erases nothing.
# --------------------------------------------------------------------------------------


def test_deleting_a_session_stops_it_and_hands_its_pod_back(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    session_id = _a_placed_session(client, log, pods)

    response = client.delete(f"/v1/sessions/{session_id}")

    assert response.status_code == 200, response.text
    assert response.json() == {"id": str(session_id), "state": "stopped", "seq": 5}
    assert pods.held == set(), "the pod outlived the delete"
    assert log.rows(session_id)[-1] == (5, lifecycle.SESSION_STOPPED)


def test_a_delete_appends_the_one_stop_this_platform_has(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """The event a delete writes is a `session.stopped` carrying the archived reason.

    Asserted rather than assumed, because it is the claim the delete route's docstring
    makes about being indistinguishable from an archive in the log. If a later slice
    publishes a `session.deleted` type and this route starts writing it, this is the
    case that fails and asks for the docstring to be rewritten.
    """
    session_id = _a_placed_session(client, log, pods)

    client.delete(f"/v1/sessions/{session_id}")

    stops = [
        event
        for event in log._events[session_id]
        if event.type == lifecycle.SESSION_STOPPED
    ]
    assert len(stops) == 1
    assert stops[0].payload == {"stop_reason": lifecycle.StopReason.ARCHIVED.value}


def test_a_deleted_sessions_whole_history_survives_at_the_same_sequences(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """Nothing before the stop moved, and nothing before it went away.

    This is the route's central promise stated as an assertion: a delete that began
    removing rows, or renumbering them, fails here. Sequence numbers are compared and
    not only the count, because a log that lost its first event and gained a stop would
    keep the same length.
    """
    session_id = _a_placed_session(client, log, pods)
    before = log.rows(session_id)

    client.delete(f"/v1/sessions/{session_id}")

    after = log.rows(session_id)
    assert after[: len(before)] == before
    assert after[len(before) :] == [(5, lifecycle.SESSION_STOPPED)]


def test_a_deleted_session_is_still_readable_and_still_listed(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """A tenant can still find and read what they asked to delete.

    Stated as a passing test rather than left implicit, because it is the part of this
    route a caller is most likely to be surprised by, and because the day a real erasure
    capability arrives this is the case that has to be deliberately inverted rather than
    quietly stopping being true.
    """
    session_id = _a_placed_session(client, log, pods)

    client.delete(f"/v1/sessions/{session_id}")

    read = client.get(f"/v1/sessions/{session_id}")
    assert read.status_code == 200, read.text
    assert read.json()["state"] == "stopped"
    listed = client.get("/v1/sessions").json()["sessions"]
    assert [row["id"] for row in listed] == [str(session_id)]


def test_a_deleted_session_still_serves_every_event_it_ever_had(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """The event range read answers after a delete, over the whole span."""
    session_id = _a_placed_session(client, log, pods)

    client.delete(f"/v1/sessions/{session_id}")

    page = client.get(f"/v1/sessions/{session_id}/events?from_seq=1&to_seq=1")
    assert page.status_code == 200, page.text
    assert [event["type"] for event in page.json()["events"]] == [
        lifecycle.SESSION_CREATED
    ]


def test_a_deleted_session_refuses_a_further_turn(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """The one thing a delete does take away, graded through the Turn route."""
    session_id = _a_placed_session(client, log, pods)
    client.delete(f"/v1/sessions/{session_id}")
    appends = log.appends

    refused = client.post(
        f"/v1/sessions/{session_id}/events",
        json={"prompt": "one more thing"},
        headers={"Idempotency-Key": "k" * 8},
    )

    assert refused.status_code == 409, refused.text
    body = refused.json()
    assert body["error"]["code"] == ErrorCode.SESSION_NOT_ACCEPTING_TURNS.value
    assert body["error"]["detail"]["state"] == "stopped"
    assert log.appends == appends, "the refused Turn was recorded anyway"


def test_deleting_twice_appends_once_and_answers_the_same(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    session_id = _a_placed_session(client, log, pods)

    first = client.delete(f"/v1/sessions/{session_id}")
    appends = log.appends
    second = client.delete(f"/v1/sessions/{session_id}")

    assert first.json() == second.json()
    assert log.appends == appends, "the retry put a second stop in the log"


def test_a_retried_delete_still_hands_back_a_pod_the_first_one_left(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """A call that died between its append and its handback is repaired by a retry.

    That shape is reachable: the stop is appended first on purpose, so a crash between
    the two leaves a stopped Session holding a pod. A retry that answered "already
    stopped" without looking at the pod would leave the slot for a later sweep.
    """
    session_id = _a_placed_session(client, log, pods)
    client.delete(f"/v1/sessions/{session_id}")
    pods.held.add(session_id)

    again = client.delete(f"/v1/sessions/{session_id}")

    assert again.status_code == 200, again.text
    assert pods.held == set()


def test_a_running_turn_refuses_the_delete_and_keeps_the_pod(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """A Turn a tenant is waiting on does not lose its pod to a delete."""
    session_id = _a_placed_session(client, log, pods)
    turn_id = _open_a_turn(log, session_id)
    appends = log.appends

    response = client.delete(f"/v1/sessions/{session_id}")

    assert response.status_code == 409, response.text
    body = response.json()
    assert body["error"]["code"] == SESSION_TURN_IN_FLIGHT
    assert body["error"]["detail"]["turn_id"] == turn_id
    assert pods.held == {session_id}, "a live Turn lost its pod"
    assert log.appends == appends


def test_deleting_a_session_belonging_to_another_tenant_is_a_404(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """Ownership is settled before anything is folded or appended.

    The other tenant's Session is **real** here -- created, logged, holding a pod -- so
    a route that folded first would find a log, reach RUNNING, append a stop and give
    the pod away. A test aimed at an id nobody created would only prove that a fold
    over an empty log fails, which is a different and much weaker claim.

    One 404 covers absent and not-yours, so an id cannot be probed for which it is; the
    case below aims at an uncreated id to hold that half.
    """
    neighbour = TestClient(client.app, headers={TENANT_HEADER: str(uuid4())})
    theirs = _a_placed_session(neighbour, log, pods)
    appends = log.appends
    before = log.rows(theirs)

    response = client.delete(f"/v1/sessions/{theirs}")

    assert response.status_code == 404, response.text
    assert log.appends == appends, "another tenant's Session was stopped"
    assert log.rows(theirs) == before
    assert pods.held == {theirs}, "another tenant's pod was handed back"


def test_deleting_a_session_that_does_not_exist_is_the_same_404(
    client: TestClient, log: PagingLog
) -> None:
    """The other half of the one refusal: absent and not-yours are indistinguishable."""
    appends = log.appends

    response = client.delete(f"/v1/sessions/{new_session_id()}")

    assert response.status_code == 404
    assert log.appends == appends


def test_a_control_plane_that_places_no_pods_still_deletes(log: PagingLog) -> None:
    """The default release does nothing, and doing nothing is right where there is
    nothing.

    This `Platform` is wired with no `session_pod_release` at all, which is every
    process built without a pod runner. A refusing default would fail a delete there.
    """
    bare = TestClient(
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
    session_id = bare.post("/v1/sessions", json=_create_body()).json()["id"]

    response = bare.delete(f"/v1/sessions/{session_id}")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "stopped"


# --------------------------------------------------------------------------------------
# Nothing about a Session is updatable.
# --------------------------------------------------------------------------------------


def test_an_empty_update_reports_the_current_state_and_writes_nothing(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """The identity ask: the only body this route accepts, and it changes nothing."""
    session_id = _a_placed_session(client, log, pods)
    appends = log.appends

    response = client.post(f"/v1/sessions/{session_id}", json={})

    assert response.status_code == 200, response.text
    assert response.json() == {"id": str(session_id), "state": "running", "seq": 4}
    assert log.appends == appends
    assert pods.held == {session_id}, "an update took a pod away"


_REFUSED_FIELD = {
    "grant": ({"grant": ["fs.write"]}, REASON_GRANT_NOT_REVISABLE),
    "budget_minor_units": ({"budget_minor_units": 9999}, REASON_BUDGET_NOT_REVISABLE),
}
"""The two fields this route parses in order to refuse, and the reason each carries.

A collection whose members each encode a decision: which fields the mirrored surface's
update maps onto here, and what a caller is told about each. A member with no case would
be a field that is accepted, or refused with the wrong explanation.
"""


@pytest.mark.parametrize(
    ("body", "reason"),
    [case for _, case in sorted(_REFUSED_FIELD.items())],
    ids=sorted(_REFUSED_FIELD),
)
def test_naming_a_creation_fact_is_refused_with_its_own_reason(
    body: dict[str, object],
    reason: str,
    client: TestClient,
    log: PagingLog,
    registry: InMemorySessionRegistry,
    pods: HeldPods,
) -> None:
    session_id = _a_placed_session(client, log, pods)
    before = registry.records[session_id]
    appends = log.appends

    response = client.post(f"/v1/sessions/{session_id}", json=body)

    assert response.status_code == 400, response.text
    envelope = response.json()
    assert envelope["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert envelope["error"]["detail"]["reason"] == reason
    assert envelope["error"]["detail"]["session_id"] == str(session_id)
    # Nothing is asserted about the sentence. The code and the named reason are the
    # contract; `message` is free text a reader of a log sees and may be reworded at any
    # time (ADR-013), so a test that pinned its wording would fail on an edit that broke
    # nothing.
    assert registry.records[session_id] == before, "a refused update rewrote the record"
    assert log.appends == appends, "a refused update appended an event"


def test_a_body_naming_both_fields_reports_the_grant_first(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """A fixed check order, so a caller that named two fields gets a stable answer."""
    session_id = _a_placed_session(client, log, pods)

    response = client.post(
        f"/v1/sessions/{session_id}",
        json={"grant": ["fs.write"], "budget_minor_units": 9999},
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["detail"]["reason"] == REASON_GRANT_NOT_REVISABLE


@pytest.mark.parametrize("absent", ["title", "metadata", "vault_ids", "retention_days"])
def test_a_field_this_platform_has_no_store_for_is_refused_by_name(
    absent: str, client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """The rest of the mirrored update surface, plus one creation fact with no case.

    These get the generic `request.invalid` naming the field rather than a refusal
    carrying a reason of its own, and that is the honest answer: there is no title, no
    metadata map and no vault anywhere in this platform, so there is no concept to
    explain the refusal of. `retention_days` is in the list because it *is* a creation
    fact and still gets this answer -- the route declares only the two fields the
    mirrored surface can change, so anything else is unknown rather than refused, and a
    reader should see that decided rather than discover it.
    """
    session_id = _a_placed_session(client, log, pods)

    response = client.post(f"/v1/sessions/{session_id}", json={absent: "anything"})

    assert response.status_code == 400, response.text
    # Split rather than searched for as a substring: `fields` is a `", ".join` of whole
    # field paths, so splitting on the separator recovers them exactly and the match
    # below is on a whole name. `absent in fields` would also pass on a field this
    # platform does happen to store whose name merely contains one of these -- `title`
    # inside a `subtitle` -- and would report the wrong field as refused.
    named = response.json()["error"]["detail"]["fields"].split(", ")
    assert absent in named, response.text


def test_updating_another_tenants_session_is_a_404_even_with_a_refused_field(
    client: TestClient, log: PagingLog, pods: HeldPods
) -> None:
    """Ownership is settled before the field refusal, so a stranger learns nothing.

    The field refusals are permanent and identical for every Session, so answering one
    for an id this tenant does not own would tell the caller the request reached a real
    Session. The 404 comes first for that reason, and this is the case that says so --
    against a Session that really exists, so a route that checked the field first would
    answer 400 and be caught rather than crash on an empty log.
    """
    neighbour = TestClient(client.app, headers={TENANT_HEADER: str(uuid4())})
    theirs = _a_placed_session(neighbour, log, pods)
    appends = log.appends

    response = client.post(f"/v1/sessions/{theirs}", json={"grant": ["fs.write"]})

    assert response.status_code == 404, response.text
    assert log.appends == appends


# --------------------------------------------------------------------------------------
# What a closed Session answers to each of the next four calls.
# --------------------------------------------------------------------------------------


def _archive(client: TestClient, session_id: SessionId) -> Response:
    return client.post(f"/v1/sessions/{session_id}/archive")


def _delete(client: TestClient, session_id: SessionId) -> Response:
    return client.delete(f"/v1/sessions/{session_id}")


def _update_empty(client: TestClient, session_id: SessionId) -> Response:
    return client.post(f"/v1/sessions/{session_id}", json={})


def _update_grant(client: TestClient, session_id: SessionId) -> Response:
    return client.post(f"/v1/sessions/{session_id}", json={"grant": ["fs.write"]})


_CLOSERS: dict[str, Callable[[TestClient, SessionId], Response]] = {
    "archive": _archive,
    "delete": _delete,
}
"""The two verbs that close a Session. Both reach the one terminal transition."""

_THEN: dict[str, Callable[[TestClient, SessionId], Response]] = {
    "archive": _archive,
    "delete": _delete,
    "update-empty": _update_empty,
    "update-grant": _update_grant,
}

_AFTER_A_CLOSE: dict[str, tuple[int, str | None]] = {
    # A second close by either verb is idempotent: the same view at the sequence of the
    # stop already in the log, and no second stop appended.
    "archive": (200, None),
    "delete": (200, None),
    # An update that asks for nothing is answered in every state, stopped included --
    # there is nothing about being closed that makes reporting the state wrong.
    "update-empty": (200, None),
    # The field refusal wins over the state. "This field is not revisable" holds in
    # every state, so reporting the state instead would send a caller to un-archive and
    # retry a call that fails for the permanent reason anyway.
    "update-grant": (400, ErrorCode.REQUEST_INVALID.value),
}
"""What each follow-up call answers once a Session is closed, whichever verb closed it.

Eight cells, and each is a decision rather than whatever the code happens to do. Two of
them are the pair worth naming -- a delete on an archived Session, and an archive on a
deleted one -- and the other six are their neighbours: leaving those unstated is how the
two that were stated drift apart from them.
"""


@pytest.mark.parametrize("closed_by", sorted(_CLOSERS))
@pytest.mark.parametrize("then", sorted(_THEN))
def test_a_closed_session_has_one_decided_answer_to_the_next_call(
    closed_by: str,
    then: str,
    client: TestClient,
    log: PagingLog,
    pods: HeldPods,
) -> None:
    session_id = _a_placed_session(client, log, pods)
    closing = _CLOSERS[closed_by](client, session_id)
    assert closing.status_code == 200, closing.text
    stopped_at = log.rows(session_id)[-1][0]
    appends = log.appends

    answer = _THEN[then](client, session_id)

    expected_status, expected_code = _AFTER_A_CLOSE[then]
    assert answer.status_code == expected_status, answer.text
    body = answer.json()
    if expected_code is None:
        assert body == {
            "id": str(session_id),
            "state": "stopped",
            "seq": stopped_at,
        }
    else:
        assert body["error"]["code"] == expected_code
    assert log.appends == appends, f"{then} after {closed_by} wrote to the log"
    assert pods.held == set()
