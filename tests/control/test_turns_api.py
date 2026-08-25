"""POST /v1/sessions/{id}/events: the three answers, and who is allowed to ask.

Tier 1 (local, no infrastructure). Realizes the surface half of MAP-A9 -- a Turn sent to
a stopped Session is refused, says so with a published code, and starts no work -- and
the dispatch half of MAP-A110, which is what carries an admitted Turn to the pod.

The tenant case is not decoration. The Event Log is keyed by Session and carries no
tenant, so with the registry lookup removed this route reads *and appends into* any
Session whose uuid a caller knows, and every other test in this file still passes,
because they all address a Session the caller owns. That hole has been root-caused twice
in this repository already, on two other routes over the same store.

The dispatch cases at the end grade the two implementations of the port. One of them
refuses every Turn, and that is the one this process is wired with: nothing here can
reach the Agent Runtime inside another pod, so a refusal is the honest answer and a
success would be a lie recorded in the Event Log.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import Placement, PodBinding, PodPhase
from managed_agent.control.session.turn_dispatch import (
    NoPodTransport,
    PodRuntime,
    PodTurnDispatch,
    TurnUndeliverable,
)
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import (
    FIRST_SEQ,
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    TurnId,
    new_session_id,
)
from managed_agent.core.pod.repertoire import TurnStartRequest
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
from managed_agent.core.vocabulary import turn

_KEY = "retry-key-0001"
_PROMPT = "summarise the findings"


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class InMemoryLog:
    """Both log ports over one list, counting its own appends."""

    def __init__(self) -> None:
        self._events: list[Event] = []
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
        seq = Seq(len(self._events) + 1)
        self._events.append(Event(session_id, seq, type_, dict(payload or {})))
        return seq

    def of(self, session_id: SessionId) -> list[Event]:
        return [event for event in self._events if event.session_id == session_id]

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Event]:
        return [event for event in self.of(session_id) if start <= event.seq <= end][
            :limit
        ]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Event]:
        for event in self.of(session_id):
            if event.seq > after:
                yield event

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ


class InMemorySessionRegistry:
    """Records who owns a Session and hands it back to that tenant only."""

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


class RecordingDispatch:
    """Accepts every Turn and remembers what it was handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[SessionId, TurnId, str]] = []

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        self.calls.append((session_id, turn_id, prompt))


class UnusedRegistry:
    """Satisfies the definition-registry port; a test that reached it is off topic."""

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        raise AssertionError("a test in this file registered a definition")

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        raise AssertionError("a test in this file resolved a definition")

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        raise AssertionError("a test in this file listed a definition's versions")

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        raise AssertionError("a test in this file read a definition revision")

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a test in this file retired a definition version")


class UnusedToolRegistry:
    """Satisfies the tool-registry port and is never called."""

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("a test in this file registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("a test in this file looked up a registered tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("a test in this file listed a tenant's tools")


class UnusedWebhooks:
    """Satisfies the webhook store port and is never called."""

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
    """Satisfies the environment-store port and is never called."""

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        raise AssertionError("a test in this file resolved an environment")


@dataclass(frozen=True, slots=True)
class Surface:
    """One app, with the two collaborators a test in this file inspects."""

    client: TestClient
    log: InMemoryLog
    dispatch: RecordingDispatch
    tenant: TenantId


def _a_record(session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
    return SessionRecord(
        id=session_id,
        tenant_id=tenant_id,
        definition_id=DefinitionId(uuid4()),
        definition_revision="1",
        grant=frozenset(),
        scope=(),
        budget_minor_units=500,
        budget_currency="USD",
        retention_days=30,
    )


def _surface(
    dispatch: RecordingDispatch | NoPodTransport | None = None,
) -> tuple[Surface, InMemorySessionRegistry]:
    log = InMemoryLog()
    recording = RecordingDispatch()
    registry = InMemorySessionRegistry()
    tenant = TenantId(uuid4())
    platform = Platform(
        event_log_append=log,
        event_log_range=log,
        definition_registry=UnusedRegistry(),
        tool_registry=UnusedToolRegistry(),
        session_registry=registry,
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=recording if dispatch is None else dispatch,
        file_store=unconfigured_file_store(),
    )
    client = TestClient(create_app(platform), headers={TENANT_HEADER: str(tenant)})
    return Surface(client, log, recording, tenant), registry


async def _a_running_session(
    surface: Surface, registry: InMemorySessionRegistry
) -> SessionId:
    session_id = new_session_id()
    await registry.create(_a_record(session_id, surface.tenant))
    surface.log.add(session_id, "session.created")
    return session_id


def _submit(surface: Surface, session_id: SessionId, key: str = _KEY) -> Any:
    return surface.client.post(
        f"/v1/sessions/{session_id}/events",
        json={"prompt": _PROMPT},
        headers={"Idempotency-Key": key},
    )


async def test_a_turn_is_accepted_and_carried_to_the_pod_exactly_once() -> None:
    surface, registry = _surface()
    session_id = await _a_running_session(surface, registry)

    answered = _submit(surface, session_id)

    assert answered.status_code == 202, answered.text
    body = answered.json()
    assert body["seq"] == 2
    assert surface.dispatch.calls == [(session_id, UUID(body["turn_id"]), _PROMPT)], (
        "the Turn the caller was told about is not the Turn that was dispatched"
    )


async def test_the_same_key_answers_200_with_the_same_turn_and_dispatches_nothing() -> (
    None
):
    """200 rather than 202, so a client that retried can tell which answer it got."""
    surface, registry = _surface()
    session_id = await _a_running_session(surface, registry)
    first = _submit(surface, session_id)

    again = _submit(surface, session_id)

    assert again.status_code == 200, again.text
    assert again.json() == first.json()
    assert len(surface.dispatch.calls) == 1, (
        "a retried submission was dispatched a second time, so one Turn ran twice"
    )


async def test_a_submission_with_no_key_is_refused_and_nothing_is_written() -> None:
    """The key is required, because a retry with no key silently spends twice."""
    surface, registry = _surface()
    session_id = await _a_running_session(surface, registry)

    answered = surface.client.post(
        f"/v1/sessions/{session_id}/events", json={"prompt": _PROMPT}
    )

    assert answered.status_code == 400
    assert surface.log.appends == 0
    assert surface.dispatch.calls == []


@pytest.mark.parametrize("key", ["short12", "has spaces", "x" * 256])
async def test_a_key_outside_the_one_rule_is_refused(key: str) -> None:
    surface, registry = _surface()
    session_id = await _a_running_session(surface, registry)

    assert _submit(surface, session_id, key=key).status_code == 400
    assert surface.log.appends == 0


@pytest.mark.parametrize("body", [{}, {"prompt": ""}, {"prompt": "x", "priority": 1}])
async def test_a_body_the_route_does_not_declare_is_refused(
    body: dict[str, object],
) -> None:
    """An empty prompt would be answered at full cost; an extra field is a
    misunderstanding worth telling the caller about rather than ignoring."""
    surface, registry = _surface()
    session_id = await _a_running_session(surface, registry)

    answered = surface.client.post(
        f"/v1/sessions/{session_id}/events",
        json=body,
        headers={"Idempotency-Key": _KEY},
    )

    assert answered.status_code == 400
    assert surface.log.appends == 0


@pytest.mark.parametrize(
    ("event", "state"),
    [("session.stopped", "stopped"), ("session.suspended", "suspended")],
)
async def test_a_session_that_takes_no_turn_is_refused_and_no_work_begins(
    event: str, state: str
) -> None:
    """MAP-A9. One published code, with the state named in `detail`.

    A code per state would grow the published set every time the state machine did,
    and each addition is a version event -- so the code says "this session accepts no
    Turn" and the fact a consumer branches on travels beside it under a name.
    """
    surface, registry = _surface()
    session_id = await _a_running_session(surface, registry)
    surface.log.add(session_id, event)

    answered = _submit(surface, session_id)

    assert answered.status_code == 409, answered.text
    assert answered.json()["error"]["code"] == "session.not_accepting_turns"
    assert answered.json()["error"]["detail"]["state"] == state
    assert surface.log.appends == 0, "a refused submission was recorded as one"
    assert surface.dispatch.calls == []


async def test_a_turn_that_cannot_be_carried_to_the_pod_is_closed_as_failed() -> None:
    """502, and the log says the Turn was asked for and did not run.

    The Turn is recorded before it is dispatched, so a dispatch that fails cannot
    simply be un-asked. What it can do is close it: the submission is followed by a
    `turn.failed` naming `pod_unreachable`, which is the honest reading of what
    happened and is what a caller paging the log later will find.
    """
    surface, registry = _surface(dispatch=NoPodTransport())
    session_id = await _a_running_session(surface, registry)

    answered = _submit(surface, session_id)

    assert answered.status_code == 502, answered.text
    assert answered.json()["error"]["code"] == "turn.undeliverable"
    written = surface.log.of(session_id)
    assert [event.type for event in written] == [
        "session.created",
        turn.TURN_SUBMITTED,
        turn.TURN_FAILED,
    ]
    assert written[-1].payload["cause"] == "pod_unreachable"
    assert (
        written[-1].payload["turn_id"] == answered.json()["error"]["detail"]["turn_id"]
    )


async def test_the_platforms_own_log_gets_the_words_the_refusal_withholds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The other half of the test above, and the half that did not exist.

    `test_the_refusal_carries_no_word_of_a_pod_or_a_runtime` proves the tenant is told
    nothing about the platform's topology. On its own that is only half a rule, and the
    missing half cost a real diagnosis on 2026-08-23: a Turn against the deployed
    control plane answered 502, no pod was created, and **nothing anywhere** said why.
    The words were built -- `control/session/pods.py` composes them deliberately, to
    tell "this Session may not be resumed" from "that environment is not registered"
    from "the image will not pull" -- and then dropped at the `except`.

    So withheld from the tenant is not the same as discarded. This asserts the words
    reach the platform's own log, which is stderr and which no tenant reads, and it
    asserts the Turn and the Session are named beside them -- a warning that says a Turn
    failed without saying which one is a line nobody can act on in a process serving
    many Sessions at once.
    """
    surface, registry = _surface(dispatch=NoPodTransport())
    session_id = await _a_running_session(surface, registry)

    with caplog.at_level(logging.WARNING):
        answered = _submit(surface, session_id)

    assert answered.status_code == 502, answered.text
    logged = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    carrying = [m for m in logged if str(session_id) in m]
    assert carrying, f"no warning named the session; got {logged}"
    assert answered.json()["error"]["detail"]["turn_id"] in carrying[0], carrying[0]
    assert "undeliverable" in carrying[0]


async def test_the_refusal_carries_no_word_of_a_pod_or_a_runtime() -> None:
    """A tenant sees a platform cause and never the platform's own topology.

    `NoPodTransport` raises an exception naming a session and the absence of a pod
    transport; nothing of that text may reach the response body (ADR-013).
    """
    surface, registry = _surface(dispatch=NoPodTransport())
    session_id = await _a_running_session(surface, registry)

    answered = _submit(surface, session_id)

    assert "pod" not in answered.text.lower()
    assert "transport" not in answered.text.lower()


async def test_another_tenants_session_and_an_absent_one_are_refused_alike() -> None:
    """The registry lookup is the only thing between a caller and another tenant's log.

    Both halves are asserted. The stranger is refused *and* the owner still succeeds,
    so the guard cannot be satisfied by breaking the route for everybody. And nothing
    is appended under the stranger's call, which is the part a read-only version of
    this hole would not have: this route writes.

    Falsifiable two ways: deleting the registry call, and deleting the tenant
    dependency, each turn the stranger's 404 into a 202.
    """
    surface, registry = _surface()
    mine = await _a_running_session(surface, registry)
    absent = new_session_id()
    appends_before = surface.log.appends

    stranger = TestClient(
        surface.client.app, headers={TENANT_HEADER: str(TenantId(uuid4()))}
    )
    refused_mine = stranger.post(
        f"/v1/sessions/{mine}/events",
        json={"prompt": _PROMPT},
        headers={"Idempotency-Key": _KEY},
    )
    refused_absent = stranger.post(
        f"/v1/sessions/{absent}/events",
        json={"prompt": _PROMPT},
        headers={"Idempotency-Key": _KEY},
    )

    assert refused_mine.status_code == 404, refused_mine.text
    assert refused_absent.status_code == 404
    assert (
        refused_mine.json()["error"]["code"] == refused_absent.json()["error"]["code"]
    )
    assert (
        refused_mine.json()["error"]["message"]
        == refused_absent.json()["error"]["message"]
    ), (
        "the two refusals differ, so the shape of the answer tells a caller holding an "
        "id whether it names another tenant's Session"
    )
    assert surface.log.appends == appends_before
    assert surface.dispatch.calls == []
    assert _submit(surface, mine).status_code == 202


class FixedPhase:
    """A cluster that reports one phase for every pod and starts nothing."""

    def __init__(self, phase: PodPhase) -> None:
        self.phase = phase

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        raise AssertionError("dispatching a Turn tried to start a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return self.phase

    async def remove(self, pod_name: str) -> None:
        raise AssertionError("dispatching a Turn tried to remove a pod")


class ScriptedRuntime:
    """A runtime that answers one Turn with one delta and a completion.

    The completion names the thread the Turn was **started on**, read back out of the
    request rather than written as a literal beside it. The two were once separate
    strings that did not match, which said nothing while nothing compared them and
    became a Turn that never completed the day the runner began to: a subagent's
    completion no longer ends the root Turn, so a frame naming an unrelated thread is
    now correctly ignored. Deriving it keeps this runtime honest about the one thing it
    is standing in for.
    """

    def __init__(self) -> None:
        self.started: list[TurnStartRequest] = []

    async def start_turn(self, request: TurnStartRequest) -> str:
        self.started.append(request)
        return "runtime-turn-1"

    async def notifications(self) -> AsyncIterator[dict[str, Any]]:
        yield {"method": "item/agentMessage/delta", "params": {"delta": "the answer"}}
        yield {
            "method": "turn/completed",
            "params": {
                "threadId": self.started[0].thread_id,
                "turn": {"id": "t1", "status": "completed"},
            },
        }


class OneChannel:
    """Hands out one prepared runtime, and counts how often it was asked."""

    def __init__(self, runtime: ScriptedRuntime) -> None:
        self.runtime = runtime
        self.opens = 0

    async def open(self, binding: PodBinding) -> PodRuntime:
        self.opens += 1
        return PodRuntime(connection=self.runtime, thread_id="thread-in-the-pod")


class Notified:
    def __init__(self) -> None:
        self.told: list[tuple[SessionId, TurnId]] = []

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        self.told.append((session_id, turn_id))


async def test_the_dispatch_this_process_is_wired_with_refuses_every_turn() -> None:
    """Fail-safe, and it is what `composition.build` hands the app.

    Nothing in this tree can open the Agent Runtime's socket inside another pod, so a
    dispatch that reported success would put a Turn in the Event Log that nothing ever
    ran. The refusal is the honest answer, and the route turns it into a recorded
    failure rather than into silence.
    """
    with pytest.raises(TurnUndeliverable):
        await NoPodTransport().dispatch(new_session_id(), TurnId(uuid4()), _PROMPT)


async def test_a_turn_runs_against_the_runtime_in_the_session_s_pod() -> None:
    """The half of the dispatch that is real, given something that opens a channel."""
    log = InMemoryLog()
    runtime = ScriptedRuntime()
    channel = OneChannel(runtime)
    notified = Notified()
    session_id, turn_id = new_session_id(), TurnId(uuid4())

    await PodTurnDispatch(
        Placement(FixedPhase(PodPhase.RUNNING)), channel, log, notified
    ).dispatch(session_id, turn_id, _PROMPT)

    assert channel.opens == 1
    assert runtime.started[0].thread_id == "thread-in-the-pod"
    assert [event.type for event in log.of(session_id)] == [
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert notified.told == [(session_id, turn_id)]


@pytest.mark.parametrize("phase", [PodPhase.ABSENT, PodPhase.STARTING, PodPhase.GONE])
async def test_a_pod_that_is_not_running_is_refused_before_a_channel_is_opened(
    phase: PodPhase,
) -> None:
    """The cluster's own answer, rather than a connection timing out.

    A tenant waiting out a socket timeout for a pod the cluster already knows is gone
    is a refusal delivered slowly, and STARTING is the case that makes the check more
    than an optimisation: the pod exists and its socket does not answer yet.
    """
    log = InMemoryLog()
    channel = OneChannel(ScriptedRuntime())

    with pytest.raises(TurnUndeliverable):
        await PodTurnDispatch(
            Placement(FixedPhase(phase)), channel, log, Notified()
        ).dispatch(new_session_id(), TurnId(uuid4()), _PROMPT)

    assert channel.opens == 0
    assert log.appends == 0
