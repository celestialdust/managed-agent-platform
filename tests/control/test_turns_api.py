"""POST /v1/sessions/{id}/events: the three answers, and who is allowed to ask.

Tier 1 (local, no infrastructure). Realizes the surface half of MAP-A9 -- a Turn sent to
a stopped Session is refused, says so with a published code, and starts no work -- and
the dispatch half of MAP-A110, which is what carries an admitted Turn to the pod.

The tenant case is not decoration. The Event Log is keyed by Session and carries no
tenant, so with the registry lookup removed this route reads *and appends into* any
Session whose uuid a caller knows, and every other test in this file still passes,
because they all address a Session the caller owns. That hole has been root-caused twice
in this repository already, on two other routes over the same store.

The dispatch case at the end grades `NoPodTransport`, the port's refusing
implementation, which is what `composition.build` wires when it is handed no
`PodRunner`. The real implementation is `HttpPodDispatch` and it is graded where it
lives, in `tests/session_shim/test_shim_serves_a_turn.py` against the shim it speaks to
and in `tests/control/test_a_first_turn_places_the_session_pod.py` against the placement
it drives; a case here that stood one up would be grading the wire twice and the route
not at all.
"""

from __future__ import annotations

import logging
import time
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
from managed_agent.control.session.turn_dispatch import (
    NoPodTransport,
    TurnDispatch,
    TurnOutputNotRevisable,
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
from managed_agent.core.session.session import SessionRecord
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


class ArtifactRewritten:
    """A dispatch whose Turn ran and whose ship-out met an occupied artifact path.

    The Turn itself succeeded here -- the model answered and its events were appended
    -- and what failed is the step after it, which is why this cannot be told apart
    from a pod that never answered by anything except the type raised.
    """

    def __init__(self, path: str = "report.md") -> None:
        self.path = path

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        raise TurnOutputNotRevisable(self.path)


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
    dispatch: TurnDispatch | None = None,
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


def _closed(surface: Surface, session_id: SessionId, within_s: float = 2.0) -> Any:
    """Wait for the Turn's terminal event and hand back the event that closed it.

    A submission is answered 202 before its Turn runs, so the outcome is no longer in
    the response -- it arrives in the log some time after it. Polled rather than
    slept-on so the case takes as long as the Turn does, and bounded so a Turn that
    never closes fails as an assertion instead of hanging the suite.
    """
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        written = surface.log.of(session_id)
        if written and written[-1].type in (turn.TURN_COMPLETED, turn.TURN_FAILED):
            return written[-1]
        time.sleep(0.005)
    raise AssertionError(
        f"no terminal event for session {session_id} within {within_s}s; the log "
        f"holds {[event.type for event in surface.log.of(session_id)]}"
    )


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
    [("session.stopped", "stopped"), ("turn.submitted", "running")],
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


@pytest.mark.parametrize(
    ("event", "remedy"),
    [("session.stopped", "accepts no Turn"), ("turn.submitted", "interrupt it")],
)
async def test_the_refusal_says_which_of_the_two_remedies_applies(
    event: str, remedy: str
) -> None:
    """One code, and the message has to separate what the code deliberately fuses.

    A stopped Session needs a new Session; one already running a Turn needs the caller
    to wait or interrupt. Both answer `session.not_accepting_turns`, so if the sentence
    were the same too, a caller would read a transient refusal as a permanent one and go
    off to recreate a Session that was seconds from being free.
    """
    surface, registry = _surface()
    session_id = await _a_running_session(surface, registry)
    surface.log.add(session_id, event)

    answered = _submit(surface, session_id)

    assert answered.status_code == 409, answered.text
    assert remedy in answered.json()["error"]["message"], answered.text


async def test_a_turn_that_cannot_be_carried_to_the_pod_is_closed_as_failed() -> None:
    """202, and the log says the Turn was asked for and did not run.

    The Turn is recorded before it is dispatched, so a dispatch that fails cannot
    simply be un-asked. What it can do is close it: the submission is followed by a
    `turn.failed` naming `pod_unreachable`, which is the honest reading of what
    happened and is what a caller paging the log later will find.

    The status was 502 until the dispatch moved off the request path. It cannot be one
    now -- the Turn is answered before it runs, so there is no failure to report on
    that response -- and the assertion moved to the log rather than being dropped,
    because *closing the Turn* was always the load-bearing half of this case. A Turn
    left open refuses the Session's next Turn and its archive for ever.
    """
    surface, registry = _surface(dispatch=NoPodTransport())
    session_id = await _a_running_session(surface, registry)

    answered = _submit(surface, session_id)

    assert answered.status_code == 202, answered.text
    closed = _closed(surface, session_id)
    assert [event.type for event in surface.log.of(session_id)] == [
        "session.created",
        turn.TURN_SUBMITTED,
        turn.TURN_FAILED,
    ]
    assert closed.payload["cause"] == "no_runtime_configured"
    assert closed.payload["turn_id"] == answered.json()["turn_id"]


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
        _closed(surface, session_id)

    assert answered.status_code == 202, answered.text
    logged = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    carrying = [m for m in logged if str(session_id) in m]
    assert carrying, f"no warning named the session; got {logged}"
    assert answered.json()["turn_id"] in carrying[0], carrying[0]
    assert "undeliverable" in carrying[0]


async def test_the_refusal_carries_no_word_of_a_pod_or_a_runtime() -> None:
    """A tenant sees a platform cause and never the platform's own topology.

    `NoPodTransport` raises an exception naming a session and the absence of a pod
    transport; nothing of that text may reach the tenant (ADR-013).

    Read off the closing event as well as the response, and the event is the half that
    now does the work. The response is a 202 carrying a turn id, so asserting the words
    are absent from it passes whatever the platform does -- a guard that cannot fail is
    not a guard, and this one exists to catch an exception's text being copied onto a
    tenant surface. The event log is that surface now.

    `cause` is excluded from the scan and asserted separately, because the published
    vocabulary is not a leak: `pod_unreachable` contains the word this looks for and is
    a member of a closed set a consumer branches on. What must not appear is anything
    the exception said -- the session it named, the absence of a transport -- in any
    field beside it.
    """
    surface, registry = _surface(dispatch=NoPodTransport())
    session_id = await _a_running_session(surface, registry)

    answered = _submit(surface, session_id)
    closed = _closed(surface, session_id)
    assert closed.payload["cause"] == turn.TurnFailureCause.NO_RUNTIME_CONFIGURED.value
    assert (
        closed.payload["remedy"]
        == turn.REMEDY_FOR[turn.TurnFailureCause.NO_RUNTIME_CONFIGURED]
    ), (
        "the remedy was not the published sentence for this cause, which is how an "
        "exception's own text would reach a tenant -- the mapping is the only source"
    )
    beside = {
        key: value
        for key, value in closed.payload.items()
        if key not in ("cause", "remedy")
    }
    told = f"{answered.text} {beside}".lower()

    assert "pod" not in told
    assert "transport" not in told


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


async def test_a_deploy_configured_with_no_pod_runner_refuses_every_turn() -> None:
    """What an unconfigured deploy does, and it is a loud failure rather than a quiet
    success.

    `composition.build` binds `NoPodTransport` on the branch where it was handed no
    `PodRunner`, and binds `HttpPodDispatch` on the branch where it was -- a process
    with a runner never reaches this class at all. So this is not the platform's
    transport; it is what stands in for one in a process that has no cluster to place a
    Session's pod in, and therefore nothing for a Turn to run on.

    Refusing is what makes such a deploy fail where an operator can see it. A dispatch
    that reported success would put a Turn in the Event Log that nothing ever ran, and
    the log would then say a Turn was asked for and is running while no pod exists to
    run it. The route turns the refusal into a recorded `turn.failed` instead, which
    `test_a_turn_that_cannot_be_carried_to_the_pod_is_closed_as_failed` grades.
    """
    with pytest.raises(TurnUndeliverable):
        await NoPodTransport().dispatch(new_session_id(), TurnId(uuid4()), _PROMPT)


async def test_a_turn_that_rewrote_a_delivered_artifact_is_refused_409_naming_it() -> (
    None
):
    """The refusal a tenant can act on, told apart from the one they cannot.

    Ship-out runs after the Turn's events are streamed, so a Turn whose agent rewrote
    an artifact it had already produced fails at a step the tenant *caused* and can
    fix. Reported as `pod_unreachable` at 502 -- which is what it was until 2026-08-25
    -- it read as a platform fault: the two moves that follow a 502 are retry and open
    a ticket, and both are wrong here. Retrying re-runs an agent that will write the
    same path again, and the platform has nothing to fix.

    The cause and the path are both in the closing event now, and the 409 that used to
    carry them is gone -- a Turn answered 202 before it runs cannot report on its own
    ship-out. The path is what moved rather than what was dropped: it is a Session's
    only way to tell which of several produced files collided, and the whole value of
    the precise cause is that the next move is "write it under a different name".
    """
    surface, registry = _surface(dispatch=ArtifactRewritten("survey/v1/report.md"))
    session_id = await _a_running_session(surface, registry)

    answered = _submit(surface, session_id)

    assert answered.status_code == 202, answered.text
    closed = _closed(surface, session_id)
    assert [event.type for event in surface.log.of(session_id)] == [
        "session.created",
        turn.TURN_SUBMITTED,
        turn.TURN_FAILED,
    ]
    assert closed.payload["cause"] == "output_not_revisable"
    assert closed.payload["path"] == "survey/v1/report.md"
    assert closed.payload["turn_id"] == answered.json()["turn_id"]


async def test_the_two_ways_a_turn_fails_are_told_apart_by_status_and_by_cause() -> (
    None
):
    """Both halves: a cause that never differs from its neighbour is not precise.

    A guard that only asserted the new cause would pass equally if every Turn failure
    recorded `output_not_revisable` -- the same defect facing the other way. This runs
    the two failures against the same route and asserts they disagree.

    The status pair was `(502, 409)` and is now `(202, 202)`, which is the point rather
    than a weakening: both submissions are accepted, and the whole distinction lives in
    the log now. So the case asserts the two responses are *indistinguishable* and the
    two causes are not -- which is a stronger statement about where the answer is than
    the old pair was.
    """
    unreachable, unreachable_registry = _surface(dispatch=NoPodTransport())
    unreachable_session = await _a_running_session(unreachable, unreachable_registry)
    rewritten, rewritten_registry = _surface(dispatch=ArtifactRewritten())
    rewritten_session = await _a_running_session(rewritten, rewritten_registry)

    first = _submit(unreachable, unreachable_session)
    second = _submit(rewritten, rewritten_session)

    assert (first.status_code, second.status_code) == (202, 202)
    causes = (
        _closed(unreachable, unreachable_session).payload["cause"],
        _closed(rewritten, rewritten_session).payload["cause"],
    )
    assert causes == ("no_runtime_configured", "output_not_revisable")


async def test_the_rewritten_artifact_refusal_names_no_pod_no_bucket_and_no_store() -> (
    None
):
    """A tenant sees their own path and the platform's cause, and nothing else.

    The path is the tenant's own string and is the point of the event, so it is not
    what ADR-013 withholds. What it withholds is the topology underneath -- that the
    artifact went to an object store, that a lane holds it, that a pod served it -- and
    a message assembled from the exception's own text would carry all three, because
    `output_shipout.py` builds its text for an operator reading stderr.

    Graded on the closing event rather than on the response body, because that is where
    the tenant reads this now. Left on the body it would assert nothing at all: a 202
    carries a turn id and a sequence number, and every word below is absent from it
    whether the redaction works or not.
    """
    surface, registry = _surface(dispatch=ArtifactRewritten("report.md"))
    session_id = await _a_running_session(surface, registry)

    _submit(surface, session_id)
    told = str(_closed(surface, session_id).payload).lower()

    assert "report.md" in told
    for topology in ("pod", "bucket", "s3", "lane", "object store"):
        assert topology not in told, f"the closing event named {topology!r}"
