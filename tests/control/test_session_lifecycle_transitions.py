"""Ending a Session's working life, and giving its pod back when it does.

Tier 1 (local, no infrastructure). The two transitions `control/session/lifecycle.py`
grows are graded here: archiving a Session, and parking one nothing is using.

**The pod is a real thing in this file, not a call record.** `TrackedCluster` below is a
`PodRunner` that holds pods in a dict, and every claim that a pod was handed back is
asserted by asking `Placement.locate` what the cluster now says -- so a test passes only
if the pod is actually gone. An assertion that `release` was called would grade the
wiring of these tests instead: it stays green if `release` is a no-op, and green if the
runner deletes the wrong pod's name.

The two collections here each encode a decision and each is parametrized over. Archive
has an answer for all four `SessionState` members, and getting one of them wrong is a
Session that either cannot be archived or is stopped twice; the four turn-event shapes a
log can end on decide whether a Turn is open, and reading only the completion is how
every failed Turn stays open for ever.

The log fake **pages** at two rows by default, which is what the real adapter does and
what three shipped defects in this repository came from. A transition that folded one
page here reads a stopped Session as running and a live Session as idle.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from managed_agent.control.session.lifecycle import (
    ArchiveRefused,
    NoSessionPods,
    SessionAlreadyArchived,
    SessionArchived,
    archive_session,
    open_turn,
    suspend_session,
)
from managed_agent.control.session.placement import (
    Placement,
    PodPhase,
    pod_name_for,
)
from managed_agent.core.ids import (
    FIRST_SEQ,
    Seq,
    SessionId,
    TurnId,
    new_session_id,
    new_turn_id,
)
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import lifecycle, turn
from managed_agent.core.vocabulary.lifecycle import StopReason

_PROMPT = "summarise the findings"


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class PagingLog:
    """Both log ports over one list, capped at two rows a read as the adapter is.

    Two rows rather than five hundred, because the cap is the whole hazard: a caller
    that names no limit sees the first page of a longer log and believes it has the
    whole thing. Every operation yields to the loop first, so the concurrency case below
    genuinely interleaves rather than running each coroutine to completion in turn.
    """

    default_page = 2

    def __init__(self) -> None:
        self._events: list[Event] = []
        self.appends = 0

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        await asyncio.sleep(0)
        self.appends += 1
        return self.add(session_id, type_, payload)

    def add(
        self,
        session_id: SessionId,
        type_: str,
        payload: dict[str, object] | None = None,
    ) -> Seq:
        """Append without going through the port, to stand in for another writer."""
        seq = Seq(len(self._events) + 1)
        self._events.append(Event(session_id, seq, type_, dict(payload or {})))
        return seq

    def types(self) -> list[str]:
        return [event.type for event in self._events]

    def payload_of(self, type_: str) -> dict[str, object]:
        matching = [event for event in self._events if event.type == type_]
        assert len(matching) == 1, f"{len(matching)} events of type {type_}"
        return matching[0].payload

    def events(self) -> list[Event]:
        return list(self._events)

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int | None = None
    ) -> Sequence[Event]:
        await asyncio.sleep(0)
        span = [
            event
            for event in self._events
            if event.session_id == session_id and start <= event.seq <= end
        ]
        return span[: self.default_page if limit is None else limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Event]:
        for event in self._events:
            if event.session_id == session_id and event.seq > after:
                yield event

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ


class TrackedCluster:
    """A cluster that really holds pods, so a handback is observable as an absence.

    Keyed by pod name and not by Session, deliberately: `Placement` derives the name and
    a runner that took a Session would hide a transition releasing the wrong one. A pod
    that was never placed is absent, and deleting an absent pod is success -- which is
    what `PodRunner.remove` promises and what makes a repeated end idempotent.
    """

    def __init__(self) -> None:
        self._pods: set[str] = set()
        self.deletions = 0

    def place(self, session_id: SessionId) -> None:
        self._pods.add(pod_name_for(session_id))

    async def ensure(self, pod_name: str, compiled: object) -> PodPhase:
        raise AssertionError("no transition in this file may start a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.RUNNING if pod_name in self._pods else PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        self.deletions += 1
        self._pods.discard(pod_name)


@dataclass(frozen=True, slots=True)
class _Fixture:
    session_id: SessionId
    log: PagingLog
    placement: Placement
    cluster: TrackedCluster

    async def pod_phase(self) -> PodPhase:
        return (await self.placement.locate(self.session_id)).phase

    def state(self) -> SessionState:
        state, _ = project(self.log.events())
        return state


def _created(state: SessionState = SessionState.RUNNING) -> _Fixture:
    """A Session whose log lands on `state`, with a pod placed and running.

    The log is built out of the events that actually cause each state rather than out of
    a stored field, because that is the only way a Session reaches one: `TAKEN_OVER` has
    no published event yet, so it is reached by appending the type the projection table
    would read for it, which is the honest way to exercise a state the vocabulary cannot
    yet produce.
    """
    session_id = new_session_id()
    log = PagingLog()
    log.add(session_id, lifecycle.SESSION_CREATED, {"environment_id": str(uuid4())})
    # Padding, so every fold below has to page past the two-row cap to be right.
    log.add(session_id, turn.TURN_SUBMITTED, {"turn_id": str(new_turn_id())})
    log.add(session_id, turn.TURN_COMPLETED, {"turn_id": _last_turn_id(log)})
    match state:
        case SessionState.RUNNING:
            pass
        case SessionState.SUSPENDED:
            log.add(
                session_id,
                lifecycle.SESSION_SUSPENDED,
                {"stop_reason": StopReason.IDLE_TIMEOUT.value},
            )
        case SessionState.STOPPED:
            log.add(
                session_id,
                lifecycle.SESSION_STOPPED,
                {"stop_reason": StopReason.ARCHIVED.value},
            )
        case SessionState.TAKEN_OVER:
            log.add(session_id, "takeover.held", {})
    cluster = TrackedCluster()
    cluster.place(session_id)
    return _Fixture(session_id, log, Placement(cluster), cluster)


def _last_turn_id(log: PagingLog) -> str:
    for event in reversed(log.events()):
        if event.type == turn.TURN_SUBMITTED:
            return str(event.payload["turn_id"])
    raise AssertionError("no submission in the log")


def _open_a_turn(fixture: _Fixture) -> TurnId:
    turn_id = new_turn_id()
    fixture.log.add(
        fixture.session_id,
        turn.TURN_SUBMITTED,
        {"turn_id": str(turn_id), "idempotency_key": "k" * 8, "prompt": _PROMPT},
    )
    return turn_id


# `TAKEN_OVER` is reachable in this file only because `_created` appends the type the
# projection table would read for it, and that type is not published. The projection has
# no row for it either -- so the fold reports RUNNING, and the archive answer for it is
# the RUNNING answer. Listed as its own row anyway: the day the takeover family lands,
# this row is where the decision has to be made rather than discovered.
_ARCHIVE_APPENDS = {
    SessionState.RUNNING: True,
    SessionState.SUSPENDED: True,
    SessionState.STOPPED: False,
    SessionState.TAKEN_OVER: True,
}


@pytest.mark.parametrize("state", list(SessionState), ids=lambda s: s.value)
async def test_archive_has_a_decided_answer_for_every_state(
    state: SessionState,
) -> None:
    """One assertion per member, because each one is a separate decision.

    A single case over "some state" would be satisfied by whichever one it picked, and
    the member most likely to be wrong is the one nobody thought about -- which is how a
    Session in that state ends up unarchivable, holding its pod for good.
    """
    fixture = _created(state)
    before = fixture.log.appends

    outcome = await archive_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    appended = fixture.log.appends > before
    assert appended is _ARCHIVE_APPENDS[state], (
        f"archiving a {state.value} session appended={appended}, expected "
        f"{_ARCHIVE_APPENDS[state]}"
    )
    assert await fixture.pod_phase() is PodPhase.ABSENT, (
        f"a {state.value} session kept its pod after being archived"
    )
    assert fixture.state() is SessionState.STOPPED
    assert isinstance(outcome, SessionArchived | SessionAlreadyArchived)


async def test_archiving_appends_one_stop_carrying_the_archived_reason() -> None:
    fixture = _created()

    outcome = await archive_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    assert isinstance(outcome, SessionArchived)
    assert outcome.pod_released is True
    assert fixture.log.types()[-1] == lifecycle.SESSION_STOPPED
    assert fixture.log.payload_of(lifecycle.SESSION_STOPPED) == {
        "stop_reason": StopReason.ARCHIVED.value
    }


async def test_the_pod_is_gone_after_an_archive_and_the_session_still_reads() -> None:
    """The pod is the disposable half. Its Session's history survives it intact."""
    fixture = _created()
    history_before = len(fixture.log.events())

    await archive_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    assert await fixture.pod_phase() is PodPhase.ABSENT
    assert len(fixture.log.events()) == history_before + 1, "nothing was removed"
    assert fixture.log.types()[0] == lifecycle.SESSION_CREATED


async def test_archiving_twice_stops_the_session_once() -> None:
    """A retried archive is not a second stop, and it is not an error either."""
    fixture = _created()

    first = await archive_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )
    second = await archive_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    assert isinstance(first, SessionArchived)
    assert isinstance(second, SessionAlreadyArchived)
    assert fixture.log.types().count(lifecycle.SESSION_STOPPED) == 1
    assert second.seq == first.seq
    assert await fixture.pod_phase() is PodPhase.ABSENT


async def test_a_running_turn_refuses_the_archive_and_keeps_the_pod() -> None:
    """The refusal is the point: the pod a Turn is using is not taken from under it."""
    fixture = _created()
    running = _open_a_turn(fixture)
    before = fixture.log.appends

    outcome = await archive_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    assert outcome == ArchiveRefused(turn_id=running)
    assert fixture.log.appends == before, "a refused archive appends nothing"
    assert await fixture.pod_phase() is PodPhase.RUNNING
    assert fixture.state() is SessionState.RUNNING


async def test_a_turn_admitted_across_the_append_keeps_its_pod() -> None:
    """The race the settled re-read exists for, driven from the log rather than argued.

    A submission that lands after the fold and before the append is invisible to the
    fold, so the stop goes in -- which is correct, the caller asked for it -- but the
    pod must not go with it, because a Turn is running on it. The settled read of
    `1..seq` is what sees that submission, and without it this test finds the pod gone.
    """
    fixture = _created()
    log = fixture.log
    session_id = fixture.session_id
    real_append = log.append

    async def append_after_a_turn_arrives(
        sid: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        if type_ == lifecycle.SESSION_STOPPED:
            _open_a_turn(fixture)
        return await real_append(sid, type_, payload)

    # Patched on the instance rather than by monkeypatching the module, so what is being
    # simulated is another writer reaching the log between this caller's fold and its
    # own append -- which is exactly what a second replica does.
    log.append = append_after_a_turn_arrives  # type: ignore[assignment]

    outcome = await archive_session(session_id, log, log, fixture.placement)

    assert isinstance(outcome, SessionArchived)
    assert outcome.pod_released is False
    # The stop is the last event, so the Session is archived -- the caller asked for
    # that and got it. What the settled read changes is only the pod: the submission
    # sits one sequence below the stop, so the fold this caller did could not see it and
    # the re-read of `1..seq` could not miss it.
    assert fixture.state() is SessionState.STOPPED
    assert open_turn(log.events()) is not None, "the submission really did land"
    assert await fixture.pod_phase() is PodPhase.RUNNING, (
        "the pod of a Turn that arrived across the append was deleted"
    )


async def test_suspending_parks_the_session_and_gives_the_pod_back() -> None:
    fixture = _created()

    seq = await suspend_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    assert fixture.state() is SessionState.SUSPENDED
    assert fixture.state().accepts_a_turn() is False
    assert await fixture.pod_phase() is PodPhase.ABSENT
    assert fixture.log.payload_of(lifecycle.SESSION_SUSPENDED) == {
        "stop_reason": StopReason.IDLE_TIMEOUT.value
    }
    assert seq == len(fixture.log.events())


async def test_a_suspended_session_keeps_every_event_it_had() -> None:
    """Parking is not deletion. What a resume would need is still in the log."""
    fixture = _created()
    before = fixture.log.types()

    await suspend_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    assert fixture.log.types() == [*before, lifecycle.SESSION_SUSPENDED]


async def test_the_stop_is_appended_before_the_pod_goes() -> None:
    """A crash between the two must leave a stopped Session with a pod, never the
    reverse.

    Observed rather than argued: the cluster raises on the deletion, so the append has
    either happened already or it has not. A Session left reading as live with no pod
    behind it answers 502 on every later Turn with nothing naming the cause, which is
    why the order is this way round.
    """
    fixture = _created()

    async def refuse_to_delete(pod_name: str) -> None:
        raise RuntimeError("the cluster refused the deletion")

    fixture.cluster.remove = refuse_to_delete  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await archive_session(
            fixture.session_id, fixture.log, fixture.log, fixture.placement
        )

    assert fixture.state() is SessionState.STOPPED, (
        "the stop was appended after the handback, so a failed handback loses it"
    )


async def test_a_process_with_no_pods_archives_without_raising() -> None:
    """`NoSessionPods` is what a control plane that places nothing is wired with."""
    fixture = _created()

    outcome = await archive_session(
        fixture.session_id, fixture.log, fixture.log, NoSessionPods()
    )

    assert isinstance(outcome, SessionArchived)
    assert fixture.state() is SessionState.STOPPED


# The four shapes a Turn's tail can take, and whether the Turn is still running. Each
# row is a decision: reading only `turn.completed` would leave the failed row open for
# ever, and a Session that had ever failed a Turn could then never be archived at all.
_TURN_TAILS: dict[str, tuple[list[str], bool]] = {
    "submitted_only": ([turn.TURN_SUBMITTED], True),
    "completed": ([turn.TURN_SUBMITTED, turn.TURN_COMPLETED], False),
    "failed": ([turn.TURN_SUBMITTED, turn.TURN_FAILED], False),
    "deltas_then_completed": (
        [turn.TURN_SUBMITTED, turn.TURN_MESSAGE_DELTA, turn.TURN_COMPLETED],
        False,
    ),
}


@pytest.mark.parametrize(
    ("tail", "still_open"), list(_TURN_TAILS.values()), ids=list(_TURN_TAILS)
)
def test_whether_a_turn_is_open_is_decided_per_terminal_type(
    tail: list[str], still_open: bool
) -> None:
    session_id = new_session_id()
    turn_id = new_turn_id()
    events = [
        Event(session_id, Seq(index + 1), type_, {"turn_id": str(turn_id)})
        for index, type_ in enumerate(tail)
    ]

    assert (open_turn(events) is not None) is still_open, (
        f"a log ending on {tail[-1]} reported the Turn "
        f"{'open' if not still_open else 'closed'}"
    )


def test_two_interleaved_turns_are_matched_by_identifier_and_not_by_position() -> None:
    """A delta of one Turn between another's submission and completion closes nothing.

    A rule reading "the last turn event wins" would call the earlier Turn closed here,
    and archiving would then delete a pod a Turn was running on.
    """
    session_id = new_session_id()
    first, second = new_turn_id(), new_turn_id()
    events = [
        Event(session_id, Seq(1), turn.TURN_SUBMITTED, {"turn_id": str(first)}),
        Event(session_id, Seq(2), turn.TURN_MESSAGE_DELTA, {"turn_id": str(second)}),
        Event(session_id, Seq(3), turn.TURN_COMPLETED, {"turn_id": str(second)}),
    ]

    assert open_turn(events) == first


def test_a_close_naming_a_turn_that_was_never_submitted_opens_nothing() -> None:
    session_id = new_session_id()
    stray = new_turn_id()
    events = [
        Event(session_id, Seq(1), lifecycle.SESSION_CREATED, {}),
        Event(session_id, Seq(2), turn.TURN_COMPLETED, {"turn_id": str(stray)}),
    ]

    assert open_turn(events) is None


def test_an_event_with_no_turn_id_is_not_read_as_a_turn() -> None:
    """Lifecycle events share the log and carry no `turn_id`; none of them opens a
    Turn."""
    session_id = new_session_id()
    events = [
        Event(session_id, Seq(1), lifecycle.SESSION_CREATED, {"grant": []}),
        Event(session_id, Seq(2), turn.TURN_SUBMITTED, {"turn_id": 17}),
    ]

    assert open_turn(events) is None


async def test_the_fold_pages_past_the_adapter_cap() -> None:
    """A transition that read one page would archive a Session it thought was running.

    The log here answers two rows a read. This Session's stop sits at row six, so a
    caller taking the default reads `created` and a submission, folds `RUNNING`, and
    appends a second stop -- which is the defect this repository has shipped three
    times in other callers.
    """
    fixture = _created(SessionState.STOPPED)
    for _ in range(3):
        fixture.log.add(fixture.session_id, turn.TURN_MESSAGE_DELTA, {})
    assert len(fixture.log.events()) > PagingLog.default_page

    outcome = await archive_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    assert isinstance(outcome, SessionAlreadyArchived)
    assert fixture.log.types().count(lifecycle.SESSION_STOPPED) == 1


async def test_sixteen_concurrent_archives_settle_and_then_append_nothing() -> None:
    """Sixteen clients retrying at once converge, and the convergence is the assertion.

    **A redundant stop is possible here and this does not assert it away.** Callers that
    all fold before any of them appends all decide to archive, and the extra stops land
    behind the first -- the same trade `admit_turn` already takes, where a submission
    that loses its race stays in the log while its caller is told who won. The fold
    reads the last event, so every reader sees one state.

    What must hold is that it settles, and the clause with teeth is the last one: once
    the dust is down, a further archive appends **nothing**. That fails if the
    already-stopped check before the append is removed, which is what would turn a
    bounded duplicate into one stop per retry for the life of the Session.
    """
    fixture = _created()

    outcomes = await asyncio.gather(
        *(
            archive_session(
                fixture.session_id, fixture.log, fixture.log, fixture.placement
            )
            for _ in range(16)
        )
    )
    settled = fixture.log.appends

    assert fixture.state() is SessionState.STOPPED
    assert await fixture.pod_phase() is PodPhase.ABSENT
    assert all(
        isinstance(outcome, SessionArchived | SessionAlreadyArchived)
        for outcome in outcomes
    )

    again = await archive_session(
        fixture.session_id, fixture.log, fixture.log, fixture.placement
    )

    assert isinstance(again, SessionAlreadyArchived)
    assert fixture.log.appends == settled, "a later archive appended a further stop"


def test_the_turn_ids_this_file_builds_are_the_ids_the_fold_returns() -> None:
    """Guard the guard: a `turn_id` these fakes write must parse back to the same id.

    Without this, every case above that compares an id could be comparing two values
    that were both wrong in the same way.
    """
    turn_id = new_turn_id()
    session_id = new_session_id()
    events = [Event(session_id, Seq(1), turn.TURN_SUBMITTED, {"turn_id": str(turn_id)})]

    found = open_turn(events)
    assert found == turn_id
    assert found == TurnId(UUID(str(turn_id)))
