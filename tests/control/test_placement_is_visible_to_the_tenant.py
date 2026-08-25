"""A tenant can tell "waiting for a node" from "the model is thinking" on its own
stream.

Tier 1: no cluster, no network. What is under test is the half of the capacity work with
no operator credential in front of it -- everything on `GET /v1/capacity` is a fleet
aggregate, and this is the part that answers "why was *my* Turn slow" for the person who
actually paid the latency.

**The failure this file exists to prevent is a silent one.** An event type appended to
the log and missing from the published set does not error anywhere: the stream drops it
on the way out and the range read omits it, so the log and the tenant's view disagree
with nothing raising. That exact defect is live in this repository for the runtime's
`thread/started` notification. So the cases below do not assert that the type exists --
they carry a `session.placing` row through the two real routes a tenant reads and assert
it arrives.

**The second claim is that the number is a number.** A `placement_waited_ms` that were
always present and always zero would be worse than no field, because it would read as a
measurement. So a Turn that waited and a Turn that did not are compared, and they must
differ in the payload rather than merely in a docstring.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.placement import (
    Placement,
    PlacementWait,
    PlacementWaits,
    PodPhase,
)
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.dispatcher import LIFECYCLE_TYPES
from managed_agent.core.ids import (
    FIRST_SEQ,
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    TurnId,
    new_session_id,
    new_turn_id,
)
from managed_agent.core.ports import SessionNotVisible
from managed_agent.core.session.projection import _TRANSITIONS
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vocabulary import is_published, placement, turn
from managed_agent.session_shim.pod_channel import HttpPodDispatch
from managed_agent.session_shim.serve import SHIM_EVENT_TYPES

_NOW_MS = 1_700_000_000_000


class MovableClock:
    """A clock a test advances by hand, so a wait is a stated number and not a range."""

    def __init__(self, now_ms: int = _NOW_MS) -> None:
        self.now_ms = now_ms

    def advance(self, by_ms: int) -> None:
        self.now_ms += by_ms

    def now_epoch_ms(self) -> int:
        return self.now_ms


@dataclass(frozen=True, slots=True)
class Row:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


class FiniteLog:
    """A real in-memory log whose tail ends, so a streaming case terminates.

    A real log rather than a mock, because every assertion here is about the bytes a
    tenant receives -- the event name and the payload -- and a mock would let both be
    wrong while the cases passed.

    `follow` stops once it has yielded everything it holds, which a production log never
    does. That is what makes an SSE case finite: the stream route returns on
    `StopAsyncIteration`, so the response completes and its body can be read whole
    instead of being held open and cancelled.
    """

    def __init__(self) -> None:
        self._rows: list[Row] = []

    def append(self, session_id: SessionId, type_: str, **payload: object) -> Seq:
        seq = Seq(len(self._rows) + FIRST_SEQ)
        self._rows.append(Row(session_id, seq, type_, dict(payload)))
        return seq

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return Seq(FIRST_SEQ)

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Row]:
        span = [r for r in self._rows if r.session_id == session_id]
        return [r for r in span if start <= r.seq <= end][:limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Row]:
        for row in self._rows:
            if row.session_id == session_id and row.seq > after:
                yield row


class UnusedAppend:
    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        raise AssertionError("a read surface appended an event")


class OneOwnedSession:
    """A registry that shows one Session to one tenant and refuses everything else.

    The ownership check is the only thing standing between a caller and another tenant's
    events on both routes below, so it is real here rather than waved through.
    """

    def __init__(self, owner: TenantId, session_id: SessionId) -> None:
        self._owner = owner
        self._session_id = session_id

    async def create(self, record: SessionRecord) -> None:
        raise AssertionError("a read surface created a Session")

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        if session_id != self._session_id or tenant_id != self._owner:
            raise SessionNotVisible(str(session_id))
        return SessionRecord(
            id=session_id,
            tenant_id=tenant_id,
            definition_id=DefinitionId(UUID(int=1)),
            definition_revision="1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=1_000,
            budget_currency="USD",
            retention_days=30,
        )

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[Any]:
        raise AssertionError("a read surface paged the Session registry")


class RefusesContact:
    """Every port these two read routes must not touch, as one refusing object."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"a read surface reached a store: .{name}()")


def _app(log: FiniteLog, owner: TenantId, session_id: SessionId) -> FastAPI:
    """The shipped app factory over a real `Platform`, with fakes behind the ports.

    Built through `create_app` rather than by mounting the two routers by hand, so these
    cases exercise the stream and range routes exactly as a deployment serves them --
    including the published-set filter the stream applies on the way out, which is the
    thing most likely to swallow a new event type.
    """
    return create_app(
        Platform(
            event_log_append=UnusedAppend(),
            event_log_range=log,
            definition_registry=RefusesContact(),
            tool_registry=RefusesContact(),
            session_registry=OneOwnedSession(owner, session_id),
            webhooks=RefusesContact(),
            environment_store=RefusesContact(),
            turn_dispatch=NoPodTransport(),
            file_store=unconfigured_file_store(),
        )
    )


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://control")


def _frames(body: str) -> list[dict[str, str]]:
    """The SSE frames in a completed response body, as name/data pairs.

    Parsed rather than substring-matched. A case that asserted `"session.placing" in
    body` would pass on a payload that merely mentioned the string, and would not notice
    the type arriving without its data line.
    """
    parsed: list[dict[str, str]] = []
    for block in body.split("\n\n"):
        fields = {}
        for line in block.splitlines():
            name, _, value = line.partition(": ")
            if name in ("event", "data", "id"):
                fields[name] = value
        if "event" in fields:
            parsed.append(fields)
    return parsed


# --------------------------------------------------------------------------------------
# It reaches a tenant. Both routes, carried end to end rather than asserted structurally
# --------------------------------------------------------------------------------------


async def test_a_placing_event_reaches_the_tenant_on_the_live_stream() -> None:
    """The whole point: the wait is visible to the person waiting.

    Carried through the real route so the published-set filter is exercised. A type
    absent from that set is dropped on the way out with nothing reported -- the route's
    own docstring says a drop means a defect upstream and that the route is not where it
    would be noticed -- so this case is the only place such a drop becomes visible.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    turn_id = new_turn_id()
    log = FiniteLog()
    log.append(session_id, turn.TURN_SUBMITTED, turn_id=str(turn_id))
    log.append(session_id, placement.SESSION_PLACING, turn_id=str(turn_id))
    log.append(session_id, turn.TURN_STARTED, turn_id=str(turn_id))

    async with _client(_app(log, owner, session_id)) as caller:
        answer = await caller.get(
            f"/v1/sessions/{session_id}/events/stream",
            headers={TENANT_HEADER: str(owner)},
        )

    frames = _frames(answer.text)
    assert [frame["event"] for frame in frames] == [
        turn.TURN_SUBMITTED,
        placement.SESSION_PLACING,
        turn.TURN_STARTED,
    ]
    placing = next(f for f in frames if f["event"] == placement.SESSION_PLACING)
    assert json.loads(placing["data"]) == {"turn_id": str(turn_id)}


async def test_a_placing_event_reaches_the_tenant_on_the_range_read() -> None:
    """The other route a tenant reads, because a caller that polls must see it too.

    A tenant reconstructing why a Turn was slow after the fact reads the range rather
    than the tail, and an event visible only on one of the two surfaces would make the
    answer depend on which one they happened to use.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    turn_id = new_turn_id()
    log = FiniteLog()
    log.append(session_id, placement.SESSION_PLACING, turn_id=str(turn_id))

    async with _client(_app(log, owner, session_id)) as caller:
        answer = await caller.get(
            f"/v1/sessions/{session_id}/events",
            headers={TENANT_HEADER: str(owner)},
        )

    assert answer.status_code == 200
    assert [event["type"] for event in answer.json()["events"]] == [
        placement.SESSION_PLACING
    ]
    assert answer.json()["events"][0]["payload"] == {"turn_id": str(turn_id)}


def test_the_type_is_published_so_neither_read_surface_can_swallow_it() -> None:
    """The one-line invariant behind both cases above, stated where it can be found.

    Kept alongside them rather than instead of them: this says the registry knows the
    name, and the two cases say the routes act on it. The registry check on its own has
    been the shape of the defect before -- a type that was published and still never
    emitted -- so neither replaces the other.
    """
    assert is_published(placement.SESSION_PLACING)


# --------------------------------------------------------------------------------------
# What it must NOT do, which is where the family choice becomes checkable
# --------------------------------------------------------------------------------------


def test_placing_moves_no_session_state() -> None:
    """A Session waiting for a pod is still RUNNING, and the fold has to agree.

    A transition here would move a Session out of the one state that accepts a Turn
    while it was merely waiting for capacity -- so its next submission would be refused
    for a reason that had nothing to do with the Session.
    """
    assert placement.SESSION_PLACING not in _TRANSITIONS


def test_placing_is_not_posted_to_a_tenant_s_registered_callback() -> None:
    """It is a stream signal, and the webhook tail is derived from one family string.

    `LIFECYCLE_TYPES` is every published name whose family is `lifecycle`, so declaring
    this there would have started posting a callback per placement to every registered
    endpoint -- a delivery nobody asked for, on the platform's retry budget, announcing
    that a pod was being started.
    """
    assert placement.FAMILY != "lifecycle"
    assert placement.SESSION_PLACING not in LIFECYCLE_TYPES


def test_a_pod_may_not_claim_it_was_waiting_for_a_pod() -> None:
    """The control plane appends this; the pod is never allowed to.

    The shim's closed set is what a compromised pod may write into an untenanted table.
    A pod able to write this could claim a wait that never happened -- fabricating the
    evidence for a slow Turn -- and the event's whole value is that the process which
    did the waiting is the one that reported it.
    """
    assert placement.SESSION_PLACING not in SHIM_EVENT_TYPES


# --------------------------------------------------------------------------------------
# The number: a Turn that waited and a Turn that did not do not look the same
# --------------------------------------------------------------------------------------


def test_a_turn_that_waited_and_one_that_did_not_carry_different_numbers() -> None:
    """The deliverable: the field is always there, and it says different things.

    The point is not that the key exists -- it is that a queued Turn and a prompt one
    are told apart *by the number*. A field that were always present and always zero
    would be worse than no field, because it would read as a measurement of no wait on
    every Turn, so this asserts a readable 1500 against a 0 rather than the key alone.

    Zero here is a true measurement and not a stand-in: that Turn found a pod running
    and waited no time for one. Whether placement happened at all stays separately
    answerable -- a `session.placing` is in the log for that Turn, or it is not -- so
    making this field unconditional loses nothing and spares every consumer that
    subtracts it from total latency a special case at every call site.
    """
    clock = MovableClock()
    wait = PlacementWait(new_session_id(), clock.now_epoch_ms(), clock)
    clock.advance(1_500)
    wait._finish()

    queued_turn, prompt_turn = new_turn_id(), new_turn_id()
    queued = placement.with_placement_wait(
        {"turn_id": str(queued_turn)}, wait.elapsed_ms
    )
    straight_through = placement.with_placement_wait({"turn_id": str(prompt_turn)}, 0)

    assert queued == {
        "turn_id": str(queued_turn),
        placement.PLACEMENT_WAITED_MS: 1_500,
    }
    assert straight_through == {
        "turn_id": str(prompt_turn),
        placement.PLACEMENT_WAITED_MS: 0,
    }
    assert (
        queued[placement.PLACEMENT_WAITED_MS]
        != straight_through[placement.PLACEMENT_WAITED_MS]
    )


def test_the_wait_keeps_running_until_it_is_finished_and_then_stops() -> None:
    """An open wait reports what it has cost so far; a closed one stops moving.

    Both halves matter and for different readers. The running total is what lets the
    fleet aggregate name how long the unluckiest request has been waiting while it is
    still waiting; the frozen total is what makes the number stamped on `turn.started`
    the wait that Turn actually had, rather than however long the process lived after.
    """
    clock = MovableClock()
    wait = PlacementWait(new_session_id(), clock.now_epoch_ms(), clock)

    clock.advance(400)
    while_open = wait.elapsed_ms
    clock.advance(600)
    still_open = wait.elapsed_ms
    wait._finish()
    clock.advance(10_000)
    after_finishing = wait.elapsed_ms

    assert (while_open, still_open) == (400, 1_000)
    assert after_finishing == 1_000


def test_a_clock_that_stepped_backwards_does_not_publish_a_negative_wait() -> None:
    """Two reads of a wall clock, and nothing here controls the wall clock.

    An NTP correction or a resumed node can make the second read earlier than the
    first. Floored rather than admitted, because a negative duration on a tenant's
    stream reads as a clock this platform vouches for.
    """
    clock = MovableClock()
    wait = PlacementWait(new_session_id(), clock.now_epoch_ms(), clock)
    clock.advance(-5_000)

    assert wait.elapsed_ms == 0
    wait._finish()
    assert wait.elapsed_ms == 0


def test_a_negative_wait_is_refused_rather_than_stamped() -> None:
    """The model checks what the caller passed, so the floor is not the only guard.

    Belt and braces on purpose: the floor above protects the value this platform
    computes, and this protects the field from any other caller that arrives at it with
    a number of its own.
    """
    with pytest.raises(ValidationError):
        placement.with_placement_wait({"turn_id": "x"}, -1)


# --------------------------------------------------------------------------------------
# The payload's own discipline
# --------------------------------------------------------------------------------------


def test_the_placing_payload_is_frozen_and_admits_no_extra_field() -> None:
    """Written once, read back by anything reconstructing why a Turn was slow.

    A field that could be added later is a field an older reader would silently ignore,
    and a payload that could be mutated after the append is a record that stops matching
    the row it was written from.
    """
    payload = placement.SessionPlacing(turn_id=new_turn_id())

    with pytest.raises(ValidationError):
        placement.SessionPlacing(  # type: ignore[call-arg]
            turn_id=new_turn_id(), node="ip-10-0-1-7"
        )
    with pytest.raises(ValidationError):
        payload.turn_id = new_turn_id()
    with pytest.raises(ValidationError):
        placement.SessionPlacing()  # type: ignore[call-arg]


def test_the_placing_payload_carries_the_turn_and_nothing_about_the_cluster() -> None:
    """A tenant learns which of its Turns is waiting, and nothing about the fleet.

    A node name or a pod name here would put a cluster detail on a tenant's stream, and
    the pod name is derivable from the Session anyway -- so it would be a disclosure
    bought for nothing.
    """
    assert set(placement.SessionPlacing.model_fields) == {"turn_id"}


# --------------------------------------------------------------------------------------
# End to end, through the dispatch a deployment actually runs
# --------------------------------------------------------------------------------------


class ShimThatStreams(httpx.AsyncBaseTransport):
    """A stand-in for the shim inside a pod: one status line, then scripted events.

    The dispatch under test is `HttpPodDispatch`, the one `composition.build` wires, and
    it reaches its pod over HTTP -- so the pod is faked at the transport rather than the
    dispatch being faked at the seam. That is the difference between grading the code a
    deployment runs and grading a class shaped like it.
    """

    def __init__(self, lines: Sequence[str]) -> None:
        self._lines = tuple(lines)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        lines = self._lines

        async def body() -> AsyncIterator[bytes]:
            for line in lines:
                yield line.encode() + b"\n"

        return httpx.Response(
            200, headers={"content-type": "application/x-ndjson"}, content=body()
        )


class RecordingLog:
    """Every append in the order it was made, which is the order under test."""

    def __init__(self) -> None:
        self.appended: list[tuple[str, dict[str, object]]] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.appended.append((type_, payload))
        return Seq(len(self.appended))

    def types(self) -> list[str]:
        return [type_ for type_, _ in self.appended]

    def payload_of(self, type_: str) -> dict[str, object]:
        return next(payload for name, payload in self.appended if name == type_)


class ClusterThatPlacesOnDemand:
    """ABSENT until a pod is asked for, RUNNING afterwards.

    The two answers are what drive the branch under test: the dispatch places only on
    ABSENT, and re-reads the phase afterwards rather than trusting what placement
    returned. A runner fixed at one phase could not reach both halves.
    """

    def __init__(self) -> None:
        self.placed = False

    async def ensure(self, pod_name: str, compiled: object) -> PodPhase:
        raise AssertionError("the dispatch bypassed SessionPods and placed directly")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.RUNNING if self.placed else PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        raise AssertionError("a test in this file released a pod")


class ClusterAlreadyRunning:
    """A pod that is up before the Turn arrives, so no placement happens at all."""

    async def ensure(self, pod_name: str, compiled: object) -> PodPhase:
        raise AssertionError("a Session with a running pod was placed again")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.RUNNING

    async def remove(self, pod_name: str) -> None:
        raise AssertionError("a test in this file released a pod")


class PodsThatTakeTime:
    """The placement seam, advancing the clock by however long the pod took.

    Standing in for a real placement's wait rather than sleeping, so the number the
    stamp carries is a stated one. A test that slept would assert a range instead of a
    value, and a range is what would pass against a broken measurement.
    """

    def __init__(
        self, cluster: ClusterThatPlacesOnDemand, clock: MovableClock, took_ms: int
    ) -> None:
        self._cluster = cluster
        self._clock = clock
        self._took_ms = took_ms

    async def ensure_for(self, session_id: SessionId) -> None:
        self._clock.advance(self._took_ms)
        self._cluster.placed = True


class NeverPlaces:
    """The placement seam for the case where a pod is already running."""

    async def ensure_for(self, session_id: SessionId) -> None:
        raise AssertionError("a Session with a running pod was placed again")


class Unnotified:
    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        return None


def _a_whole_turn(turn_id: TurnId) -> list[str]:
    """The lines a shim streams for a Turn that starts, answers and completes."""
    started = json.dumps(
        {
            "kind": "event",
            "type": turn.TURN_STARTED,
            "payload": {"turn_id": str(turn_id)},
        }
    )
    completed = json.dumps(
        {
            "kind": "event",
            "type": turn.TURN_COMPLETED,
            "payload": {"turn_id": str(turn_id), "text": "the answer"},
        }
    )
    return [started, completed, json.dumps({"kind": "completed"})]


async def test_a_queued_turn_is_announced_then_timed_when_it_starts() -> None:
    """Both halves of the tenant surface, on the dispatch a deployment runs.

    The ordered list settles that the announcement precedes the Turn's own events. It
    does not settle that it precedes the *wait* -- an append moved to just after the
    placement would produce this same list -- so that stronger claim is graded by
    `test_the_announcement_is_in_the_log_while_the_turn_is_still_queued`, which reads
    the log from inside the placement window.

    The stamp is then the same wait as a number, on the event that marks the model being
    given the work -- so the two together bracket exactly what the tenant paid for.
    """
    clock = MovableClock()
    cluster = ClusterThatPlacesOnDemand()
    placer = Placement(cluster, PlacementWaits(clock))
    log = RecordingLog()
    session_id, turn_id = new_session_id(), new_turn_id()

    await HttpPodDispatch(
        placement=placer,
        pods=PodsThatTakeTime(cluster, clock, took_ms=61_000),
        log=log,
        on_completed=Unnotified(),
        namespace="map-test",
        token_key=b"a shim signing key",
        transport=ShimThatStreams(_a_whole_turn(turn_id)),
    ).dispatch(session_id, turn_id, "summarise it")

    assert log.types() == [
        placement.SESSION_PLACING,
        turn.TURN_STARTED,
        turn.TURN_COMPLETED,
    ]
    assert log.payload_of(placement.SESSION_PLACING) == {"turn_id": str(turn_id)}
    assert log.payload_of(turn.TURN_STARTED)[placement.PLACEMENT_WAITED_MS] == 61_000


async def test_a_turn_that_found_a_pod_is_timed_at_zero_and_not_announced() -> None:
    """The other arm, and the pair is the deliverable.

    No `session.placing`, because nothing was placed -- so the event's presence stays a
    true answer to "did this Turn queue". And `placement_waited_ms` is still there,
    reading 0, so a consumer subtracting it from total latency needs no special case for
    the ordinary Turn.

    The clock advances by a minute across the dispatch anyway, which is the trap this
    guards: a measurement taken from the wall clock rather than from the placement
    window would report 60000 here, and the field would silently become "how long the
    Turn took" on every Turn that never queued.
    """
    clock = MovableClock()
    placer = Placement(ClusterAlreadyRunning(), PlacementWaits(clock))
    log = RecordingLog()
    session_id, turn_id = new_session_id(), new_turn_id()
    clock.advance(60_000)

    await HttpPodDispatch(
        placement=placer,
        pods=NeverPlaces(),
        log=log,
        on_completed=Unnotified(),
        namespace="map-test",
        token_key=b"a shim signing key",
        transport=ShimThatStreams(_a_whole_turn(turn_id)),
    ).dispatch(session_id, turn_id, "summarise it")

    assert log.types() == [turn.TURN_STARTED, turn.TURN_COMPLETED]
    assert placement.SESSION_PLACING not in log.types()
    assert log.payload_of(turn.TURN_STARTED)[placement.PLACEMENT_WAITED_MS] == 0


async def test_only_turn_started_is_stamped_and_the_other_events_pass_through() -> None:
    """The stamp goes on one event, not on everything the pod streamed.

    `turn.completed` carrying a placement wait would be a second copy of one number,
    free to disagree with the first, on an event whose subject is the answer rather than
    the queue. Asserted rather than assumed because the append loop handles every line
    and a condition dropped there would stamp all of them.
    """
    clock = MovableClock()
    cluster = ClusterThatPlacesOnDemand()
    placer = Placement(cluster, PlacementWaits(clock))
    log = RecordingLog()
    session_id, turn_id = new_session_id(), new_turn_id()

    await HttpPodDispatch(
        placement=placer,
        pods=PodsThatTakeTime(cluster, clock, took_ms=250),
        log=log,
        on_completed=Unnotified(),
        namespace="map-test",
        token_key=b"a shim signing key",
        transport=ShimThatStreams(_a_whole_turn(turn_id)),
    ).dispatch(session_id, turn_id, "summarise it")

    carrying = [
        type_
        for type_, payload in log.appended
        if placement.PLACEMENT_WAITED_MS in payload
    ]
    assert carrying == [turn.TURN_STARTED]


class PodsThatReadTheLogWhileWaiting:
    """The placement seam, reading the log from inside its own placement.

    **This is the only vantage point from which "announced while queued" is checkable.**
    Every other case here reads the log after the dispatch returns, and by then an
    append made before the wait and one made after it have produced the same list. Taken
    from in here, the observation is of the log as a tenant would see it at the moment
    the Turn is actually waiting.
    """

    def __init__(
        self, cluster: ClusterThatPlacesOnDemand, log: RecordingLog, clock: MovableClock
    ) -> None:
        self._cluster = cluster
        self._log = log
        self._clock = clock
        self.seen_while_waiting: list[str] | None = None

    async def ensure_for(self, session_id: SessionId) -> None:
        self.seen_while_waiting = self._log.types()
        self._clock.advance(3_000)
        self._cluster.placed = True


async def test_the_announcement_is_in_the_log_while_the_turn_is_still_queued() -> None:
    """A tenant learns its Turn is queued while that is still true.

    This is the whole point of the event. Under load, "waiting for a node" and "the
    model is thinking" are indistinguishable from outside, and an announcement that
    lands after the wait resolves it for nobody: it arrives beside `turn.started`, at
    the moment the tenant no longer needs it, describing a state that has ended.

    A stream is read forwards and once, so lateness here is not a small imprecision --
    it is the difference between an answer and a record.
    """
    clock = MovableClock()
    cluster = ClusterThatPlacesOnDemand()
    placer = Placement(cluster, PlacementWaits(clock))
    log = RecordingLog()
    pods = PodsThatReadTheLogWhileWaiting(cluster, log, clock)
    session_id, turn_id = new_session_id(), new_turn_id()

    await HttpPodDispatch(
        placement=placer,
        pods=pods,
        log=log,
        on_completed=Unnotified(),
        namespace="map-test",
        token_key=b"a shim signing key",
        transport=ShimThatStreams(_a_whole_turn(turn_id)),
    ).dispatch(session_id, turn_id, "summarise it")

    assert pods.seen_while_waiting == [placement.SESSION_PLACING], (
        "the tenant had not been told it was queued at the moment it was queued"
    )
