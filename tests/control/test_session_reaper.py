"""The sweep that gives back Session pods nothing owns, and everything it refuses to.

Tier 1 (local, no infrastructure). Realizes the reclamation half of the leak: the
cluster holds roughly forty-five Session pods, nothing in this tree ever handed one
back, so the platform served forty-five Sessions and then wedged.

**Pods are real objects here.** `TrackedCluster` holds them in a dict, lists them, and
deletes them, standing in for `KubernetesPodRunner`, which does all three. Every claim
that a pod was handed back is asserted by asking the cluster what it still holds -- so a
test goes green only if the pod is genuinely gone, and stays red if the handback is
removed, if it deletes the wrong name, or if it is replaced by a no-op. No case below
asserts that a method was called.

`ReapVerdict` is a collection whose members each encode a decision about deleting a
running process, so the first case parametrizes over every one of them and requires that
the sweep can actually reach it. The four `SessionState` members are the other such
collection, and `verdict_for` is graded per member -- including `TAKEN_OVER`, which no
fold can currently produce, because a decision graded by nothing is how a guard gets
removed without a test noticing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from managed_agent.control.session.placement import (
    PlacedPod,
    Placement,
    PodPhase,
    pod_name_for,
)
from managed_agent.control.session.reaper import (
    IDLE_GRACE_MS,
    ReapOutcome,
    ReapVerdict,
    SessionPodReaper,
    verdict_for,
)
from managed_agent.core.ids import (
    FIRST_SEQ,
    Seq,
    SessionId,
    new_session_id,
    new_turn_id,
)
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import lifecycle, turn
from managed_agent.core.vocabulary.lifecycle import StopReason

_NOW_MS = 1_800_000_000_000
_A_MINUTE_MS = 60_000


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class PagingLog:
    """Both log ports over one list, capped at two rows a read as the adapter is.

    Also answers the sweep's cross-Session activity question, because the one adapter
    behind both is `PostgresEventLogRange`. The window is honoured against a per-event
    timestamp this fake keeps beside each row -- the real column is `appended_at` and
    the port hands back no timestamp at all, which is why the sweep asks "which Sessions
    appear in this window" rather than "how old is this Session's last event".
    """

    default_page = 2

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._at_ms: list[int] = []
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
        at_ms: int = _NOW_MS,
    ) -> Seq:
        seq = Seq(len(self._events) + 1)
        self._events.append(Event(session_id, seq, type_, dict(payload or {})))
        self._at_ms.append(at_ms)
        return seq

    def types_of(self, session_id: SessionId) -> list[str]:
        return [e.type for e in self._events if e.session_id == session_id]

    def state_of(self, session_id: SessionId) -> SessionState:
        state, _ = project(
            [event for event in self._events if event.session_id == session_id]
        )
        return state

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

    async def lifecycle_events_between(
        self, types: Sequence[str], from_ms: int, to_ms: int
    ) -> Sequence[Event]:
        """Half-open below and closed above, exactly as the adapter's SQL is."""
        await asyncio.sleep(0)
        return [
            event
            for event, at_ms in zip(self._events, self._at_ms, strict=True)
            if event.type in set(types) and from_ms < at_ms <= to_ms
        ]


class TrackedCluster:
    """A cluster that holds, lists and deletes pods, as the real adapter does.

    One object for both ports on purpose: `KubernetesPodRunner` satisfies `PodRunner`
    and `PlacedPods` together, so splitting them here would let a sweep list from one
    cluster and delete in another and never be caught.

    `phase_of` reports what is held, so a handback is observable as an absence rather
    than as a recorded call. A pod that was never placed is absent, and deleting an
    absent one is success -- which is what `remove` promises and what makes a repeated
    sweep idempotent.
    """

    def __init__(self) -> None:
        self._pods: dict[str, PlacedPod] = {}
        self.deleted: list[str] = []

    def place(
        self,
        session_id: SessionId,
        *,
        phase: PodPhase = PodPhase.RUNNING,
        age_ms: int = IDLE_GRACE_MS * 2,
    ) -> None:
        self._pods[pod_name_for(session_id)] = PlacedPod(
            session_id=session_id, phase=phase, created_at_ms=_NOW_MS - age_ms
        )

    def holds(self, session_id: SessionId) -> bool:
        return pod_name_for(session_id) in self._pods

    async def placed_pods(self) -> Sequence[PlacedPod]:
        return list(self._pods.values())

    async def ensure(self, pod_name: str, compiled: object) -> PodPhase:
        raise AssertionError("the sweep may never start a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        held = self._pods.get(pod_name)
        return held.phase if held is not None else PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        self.deleted.append(pod_name)
        self._pods.pop(pod_name, None)


class FrozenClock:
    def now_epoch_ms(self) -> int:
        return _NOW_MS


@dataclass(frozen=True, slots=True)
class _Bench:
    log: PagingLog
    cluster: TrackedCluster
    reaper: SessionPodReaper

    def verdict_for_session(
        self, outcomes: Sequence[ReapOutcome], session_id: SessionId
    ) -> ReapVerdict:
        matching = [outcome for outcome in outcomes if outcome.session_id == session_id]
        assert len(matching) == 1, f"{len(matching)} outcomes for {session_id}"
        return matching[0].verdict


@pytest.fixture
def bench() -> _Bench:
    log = PagingLog()
    cluster = TrackedCluster()
    return _Bench(
        log=log,
        cluster=cluster,
        reaper=SessionPodReaper(
            pods=cluster,
            release=Placement(cluster),
            log=log,
            events=log,
            activity=log,
            clock=FrozenClock(),
        ),
    )


def _a_session(
    bench: _Bench,
    *,
    state: SessionState = SessionState.IDLE,
    last_event_ms: int = _NOW_MS - IDLE_GRACE_MS * 2,
    open_turn: bool = False,
    phase: PodPhase = PodPhase.RUNNING,
    pod_age_ms: int = IDLE_GRACE_MS * 2,
    with_a_log: bool = True,
) -> SessionId:
    """One Session with a pod, dialled to whichever branch a case is about.

    The log is built from the events that actually cause each state rather than from a
    stored field, because a fold is the only way a Session has a state at all. Padding
    puts every fold past the two-row page cap.
    """
    session_id = new_session_id()
    if with_a_log:
        log = bench.log
        log.add(
            session_id,
            lifecycle.SESSION_CREATED,
            {"environment_id": str(uuid4())},
            at_ms=last_event_ms,
        )
        first = new_turn_id()
        log.add(
            session_id,
            turn.TURN_SUBMITTED,
            {"turn_id": str(first)},
            at_ms=last_event_ms,
        )
        log.add(
            session_id,
            turn.TURN_COMPLETED,
            {"turn_id": str(first)},
            at_ms=last_event_ms,
        )
        if state is SessionState.STOPPED:
            log.add(
                session_id,
                lifecycle.SESSION_STOPPED,
                {"stop_reason": StopReason.ARCHIVED.value},
                at_ms=last_event_ms,
            )
        if open_turn or state is SessionState.RUNNING:
            # An unclosed submission is what `RUNNING` *is* now -- there is no event
            # that says "running" on its own -- so the two dials build the same log.
            log.add(
                session_id,
                turn.TURN_SUBMITTED,
                {"turn_id": str(new_turn_id())},
                at_ms=last_event_ms,
            )
    bench.cluster.place(session_id, phase=phase, age_ms=pod_age_ms)
    return session_id


# One dial per verdict, so the parametrized case below reaches every member. A member
# no set of dials can reach is a branch nothing exercises, which is how four of five
# refusal reasons in `pod_runner.py` came to be graded by nothing.
_REACHES: dict[ReapVerdict, dict[str, object]] = {
    ReapVerdict.A_POD_IS_STILL_COMING_UP: {"phase": PodPhase.STARTING},
    ReapVerdict.THE_POD_IS_TOO_YOUNG_TO_JUDGE: {"pod_age_ms": _A_MINUTE_MS},
    ReapVerdict.A_TURN_IS_STILL_OPEN: {"open_turn": True},
    ReapVerdict.THE_SESSION_WAS_USED_RECENTLY: {
        "last_event_ms": _NOW_MS - _A_MINUTE_MS
    },
    ReapVerdict.NO_SESSION_OWNS_IT: {"with_a_log": False},
    ReapVerdict.THE_SESSION_HAS_ENDED: {"state": SessionState.STOPPED},
    ReapVerdict.THE_SESSION_WENT_IDLE: {},
}


@pytest.mark.parametrize("verdict", list(_REACHES), ids=lambda v: v.value)
async def test_the_sweep_reaches_every_verdict_and_acts_on_it(
    bench: _Bench, verdict: ReapVerdict
) -> None:
    """One case per member, and the pod is checked against the member's own claim.

    `gave_the_pod_back` is what says which way each member goes, so this reads the
    answer off the member rather than restating a list here -- a second list would agree
    with itself the first time somebody moved a member from one side to the other.
    """
    session_id = _a_session(bench, **_REACHES[verdict])  # type: ignore[arg-type]

    outcomes = await bench.reaper.sweep()

    assert bench.verdict_for_session(outcomes, session_id) is verdict
    assert bench.cluster.holds(session_id) is not verdict.gave_the_pod_back(), (
        f"{verdict.value} says gave_the_pod_back={verdict.gave_the_pod_back()} and the "
        f"cluster {'holds' if bench.cluster.holds(session_id) else 'lost'} the pod"
    )


_REACHED_WITHOUT_A_DIAL = {
    # Reaching it means making a collaborator fail rather than putting a Session in some
    # state, so it has its own case below.
    ReapVerdict.THE_SWEEP_COULD_NOT_DECIDE,
    # **Not reachable through a sweep at all today, and that is a fact about the
    # projection rather than a gap in these dials.** Nothing publishes a takeover event
    # and `projection.py` has no row that produces `TAKEN_OVER`, so no log can fold to
    # it -- there is no sequence of events a test could write that would take this
    # branch. Graded instead by `test_every_session_state_has_a_decided_verdict` over
    # the pure `verdict_for`, which is why that function was split out of the sweep: a
    # decision graded by nothing is a guard that gets deleted without a failure, and
    # this one costs somebody their work the day takeovers start arriving.
    ReapVerdict.A_HUMAN_HAS_TAKEN_OVER,
}


def test_every_verdict_has_a_case_that_reaches_it() -> None:
    """Guard the guard. A member with no dial above would be graded by nothing.

    The exemptions are enumerated with their reasons rather than left as a slack
    inequality, and the assertion is an equality in both directions -- so a dial added
    for an exempt verdict, or an exemption left behind after its verdict became
    reachable, fails here instead of quietly widening the allowance.
    """
    covered = set(_REACHES) | _REACHED_WITHOUT_A_DIAL
    assert covered == set(ReapVerdict), (
        "verdicts nothing reaches: "
        f"{sorted(v.value for v in set(ReapVerdict) - covered)}; verdicts named twice: "
        f"{sorted(v.value for v in set(_REACHES) & _REACHED_WITHOUT_A_DIAL)}"
    )


# What each Session state settles on its own, before any clock or Turn is consulted.
# `TAKEN_OVER` cannot be produced by a fold today -- nothing publishes a takeover event
# and `projection.py` has no row for one -- so this row is the only thing standing
# between the guard and its silent removal.
_STATE_SETTLES: dict[SessionState, ReapVerdict | None] = {
    SessionState.IDLE: None,
    SessionState.RUNNING: None,
    SessionState.STOPPED: ReapVerdict.THE_SESSION_HAS_ENDED,
    SessionState.TAKEN_OVER: ReapVerdict.A_HUMAN_HAS_TAKEN_OVER,
}


@pytest.mark.parametrize("state", list(SessionState), ids=lambda s: s.value)
def test_every_session_state_has_a_decided_verdict(state: SessionState) -> None:
    assert verdict_for(state) is _STATE_SETTLES[state]


def test_a_session_a_human_is_driving_keeps_its_pod() -> None:
    """Stated as its own case because it is the one that costs somebody their work.

    A taken-over Session is idle by design -- a person is reading it -- so every timing
    guard the sweep has would let it through, and the state is the only thing that says
    no. The tenant can still archive it, which is a deliberate call by whoever owns it.
    """
    assert verdict_for(SessionState.TAKEN_OVER) is ReapVerdict.A_HUMAN_HAS_TAKEN_OVER
    assert ReapVerdict.A_HUMAN_HAS_TAKEN_OVER.gave_the_pod_back() is False


async def test_going_idle_gives_the_pod_back_and_says_nothing_in_the_log(
    bench: _Bench,
) -> None:
    """The reclamation is silent now, and the silence is the decision.

    This used to append `session.suspended`, which was worth saying while a pod outlived
    the Turn that made it: a pod going was a thing that happened *to* a Session between
    Turns, and a tenant could reasonably want to hear about it. A pod is now leased for
    one Turn, so the only pod this sweep ever finds is one whose control plane died
    holding it -- and an event announcing that would be the platform reporting its own
    crash into a tenant's Session history, once per replica that noticed.

    The Session does not move either. It was `IDLE` before the sweep and it is `IDLE`
    after, because nothing about a pod is a claim about whether a Session will work
    again.
    """
    session_id = _a_session(bench)
    before = bench.log.types_of(session_id)

    await bench.reaper.sweep()

    assert bench.log.types_of(session_id) == before
    assert bench.log.state_of(session_id) is SessionState.IDLE, (
        "the pod went; the Session did not move, and is still takeable"
    )
    assert not bench.cluster.holds(session_id)


async def test_a_turn_that_opens_across_the_decision_keeps_its_pod(
    bench: _Bench,
) -> None:
    """The race the re-read in front of the deletion exists for.

    Every guard runs against one fold, and a submission that lands after that fold is
    invisible to it: the sweep would go on to delete the pod of a Turn a tenant is
    waiting on, on the strength of a log that was already stale when it was read. The
    reclamation used to append, and the append settled this half of the race by
    sequence; nothing is appended now, so what is left is to look again as late as
    possible.

    Driven from the log rather than argued. The read is patched on the instance, so what
    is simulated is another writer reaching the store at that instant -- and it fires on
    the empty page that ends a fold, which is exactly the gap between the fold the
    guards used and the read below them.
    """
    session_id = _a_session(bench)
    real_read = bench.log.read
    arrived = False

    async def read_and_then_a_turn_arrives(
        sid: SessionId, start: Seq, end: Seq, limit: int | None = None
    ) -> Sequence[Event]:
        nonlocal arrived
        page = await real_read(sid, start, end, limit)
        if not page and not arrived and sid == session_id:
            arrived = True
            bench.log.add(
                session_id, turn.TURN_SUBMITTED, {"turn_id": str(new_turn_id())}
            )
        return page

    bench.log.read = read_and_then_a_turn_arrives  # type: ignore[assignment]

    outcomes = await bench.reaper.sweep()

    assert arrived, "the submission never landed, so this case proved nothing"
    assert (
        bench.verdict_for_session(outcomes, session_id)
        is ReapVerdict.A_TURN_IS_STILL_OPEN
    )
    assert bench.cluster.holds(session_id), (
        "the pod of a Turn that arrived across the decision was deleted"
    )


async def test_an_idle_session_keeps_its_history_when_its_pod_goes(
    bench: _Bench,
) -> None:
    """The pod is the disposable half. Everything a resume would read is still there."""
    session_id = _a_session(bench)
    before = bench.log.types_of(session_id)

    await bench.reaper.sweep()

    assert bench.log.types_of(session_id) == before


async def test_an_ended_session_is_swept_without_appending_anything(
    bench: _Bench,
) -> None:
    """Reclaiming an archived Session's pod is not a second ending in its log."""
    session_id = _a_session(bench, state=SessionState.STOPPED)
    appends_before = bench.log.appends

    await bench.reaper.sweep()

    assert bench.log.appends == appends_before
    assert bench.log.types_of(session_id).count(lifecycle.SESSION_STOPPED) == 1
    assert not bench.cluster.holds(session_id)


async def test_sweeping_twice_is_the_same_as_sweeping_once(bench: _Bench) -> None:
    """Idempotent, which is what lets two replicas and a retry all run it."""
    idle = _a_session(bench)
    ended = _a_session(bench, state=SessionState.STOPPED)
    types_before_the_first_sweep = bench.log.types_of(idle)

    first = await bench.reaper.sweep()
    deletions_after_one = len(bench.cluster.deleted)
    second = await bench.reaper.sweep()

    assert len(first) == 2
    assert second == [], "the second sweep found nothing left to judge"
    assert len(bench.cluster.deleted) == deletions_after_one
    assert bench.log.types_of(idle) == types_before_the_first_sweep
    assert bench.log.types_of(ended).count(lifecycle.SESSION_STOPPED) == 1


async def test_two_replicas_sweeping_at_once_settle_on_one_state(
    bench: _Bench,
) -> None:
    """Concurrent sweeps converge, and the convergence is what is asserted.

    Idempotence here is now a property of the deletion alone, which is why the case is
    stated as an append count that never moves. Two sweeps that both fold before either
    acts both decide to reclaim, and the second handback meets a pod that is already
    gone -- which the release treats as success. There is nothing else either sweep
    does, so "both ran" and "one ran" are indistinguishable afterwards by construction
    rather than by a lock.

    **The bound used to need an argument and does not any more.** While this appended,
    the reclamation was a write whose only limit was that the sweep walks pods: two
    replicas could each append, a second pod placed later could be reclaimed and
    appended for again, and the log grew by one honest event per reclamation. Nothing is
    written now, so the last stanza below places a second pod, has it reclaimed, and
    still expects the same count -- the property is no longer "bounded growth" but "no
    growth".
    """
    session_id = _a_session(bench)
    before = bench.log.types_of(session_id)

    await asyncio.gather(bench.reaper.sweep(), bench.reaper.sweep())
    settled = bench.log.appends

    assert bench.log.state_of(session_id) is SessionState.IDLE
    assert not bench.cluster.holds(session_id)
    assert bench.log.types_of(session_id) == before

    await bench.reaper.sweep()

    assert bench.log.appends == settled, (
        "a sweep with no pod to walk still appended an event"
    )

    bench.cluster.place(session_id)
    await bench.reaper.sweep()

    assert bench.log.appends == settled, (
        "a second pod was placed and reclaimed, and the reclamation wrote to the log"
    )
    assert not bench.cluster.holds(session_id)
    assert bench.log.state_of(session_id) is SessionState.IDLE
    assert not bench.cluster.holds(session_id), "and it still reclaimed the pod"


async def test_a_pod_at_one_millisecond_inside_the_grace_is_kept(
    bench: _Bench,
) -> None:
    """The boundary, both sides, because an off-by-one here deletes a live Session.

    A pod exactly at the grace is old enough; one millisecond younger is not. Stated as
    a pair rather than a single case, so a comparison flipped from `<` to `<=` fails
    here instead of shifting the whole policy by a millisecond nobody notices.
    """
    just_inside = _a_session(bench, pod_age_ms=IDLE_GRACE_MS - 1)
    exactly_at = _a_session(bench, pod_age_ms=IDLE_GRACE_MS)

    outcomes = await bench.reaper.sweep()

    assert (
        bench.verdict_for_session(outcomes, just_inside)
        is ReapVerdict.THE_POD_IS_TOO_YOUNG_TO_JUDGE
    )
    assert bench.cluster.holds(just_inside)
    assert (
        bench.verdict_for_session(outcomes, exactly_at)
        is ReapVerdict.THE_SESSION_WENT_IDLE
    )
    assert not bench.cluster.holds(exactly_at)


async def test_a_turn_silent_past_the_grace_still_keeps_its_pod(bench: _Bench) -> None:
    """The guard that does not depend on a timer, which is the point of having it.

    A first Turn's placement was measured holding for minutes with nothing appended, so
    a Turn can be older than the grace and still be live. The activity window cannot see
    that -- its newest row for this Session is well outside the window -- and the fold
    over the Session's own log is what does.
    """
    session_id = _a_session(
        bench, open_turn=True, last_event_ms=_NOW_MS - IDLE_GRACE_MS * 10
    )

    outcomes = await bench.reaper.sweep()

    assert (
        bench.verdict_for_session(outcomes, session_id)
        is ReapVerdict.A_TURN_IS_STILL_OPEN
    )
    assert bench.cluster.holds(session_id)
    assert bench.log.state_of(session_id) is SessionState.RUNNING


async def test_a_pod_whose_session_cannot_be_read_is_kept_and_reported(
    bench: _Bench,
) -> None:
    """One unreadable Session must not stop the sweep, and must not be a silent keep.

    The failure mode this closes is the wedge arriving by another route: a sweep that
    raised on the first bad Session would reclaim nothing for the rest of the cluster.
    """
    unreadable = _a_session(bench)
    healthy = _a_session(bench, state=SessionState.STOPPED)
    real_read = bench.log.read

    async def fail_for_one(
        session_id: SessionId, start: Seq, end: Seq, limit: int | None = None
    ) -> Sequence[Event]:
        if session_id == unreadable:
            raise RuntimeError("the store refused this read")
        return await real_read(session_id, start, end, limit)

    bench.log.read = fail_for_one  # type: ignore[method-assign]

    outcomes = await bench.reaper.sweep()

    assert (
        bench.verdict_for_session(outcomes, unreadable)
        is ReapVerdict.THE_SWEEP_COULD_NOT_DECIDE
    )
    assert bench.cluster.holds(unreadable), "an undecided pod is kept, not deleted"
    assert (
        bench.verdict_for_session(outcomes, healthy)
        is ReapVerdict.THE_SESSION_HAS_ENDED
    )
    assert not bench.cluster.holds(healthy), "one bad Session stopped the whole sweep"


async def test_a_pod_the_cluster_says_is_gone_is_judged_on_its_session(
    bench: _Bench,
) -> None:
    """GONE is not a licence to delete: the Session behind it may still be working."""
    working = _a_session(bench, phase=PodPhase.GONE, open_turn=True)
    finished = _a_session(bench, phase=PodPhase.GONE, state=SessionState.STOPPED)

    outcomes = await bench.reaper.sweep()

    assert (
        bench.verdict_for_session(outcomes, working) is ReapVerdict.A_TURN_IS_STILL_OPEN
    )
    assert (
        bench.verdict_for_session(outcomes, finished)
        is ReapVerdict.THE_SESSION_HAS_ENDED
    )


async def test_the_sweep_reports_the_pods_it_left_alone(bench: _Bench) -> None:
    """A sweep reporting only its deletions cannot be told from one that refused all."""
    kept = _a_session(bench, phase=PodPhase.STARTING)
    taken = _a_session(bench)

    outcomes = await bench.reaper.sweep()

    assert {outcome.session_id for outcome in outcomes} == {kept, taken}
    assert sum(1 for o in outcomes if o.verdict.gave_the_pod_back()) == 1


async def test_an_empty_cluster_sweeps_to_an_empty_answer(bench: _Bench) -> None:
    assert await bench.reaper.sweep() == []
    assert bench.cluster.deleted == []


async def test_the_fold_pages_past_the_adapter_cap(bench: _Bench) -> None:
    """A sweep that read one page would suspend a Session with a Turn running.

    The log answers two rows a read, and this Session's open submission sits at row six.
    A `_decide` taking the default sees `created` and one submission, folds RUNNING, and
    finds the submission closed -- so it suspends and deletes the pod of a live Turn.
    """
    session_id = _a_session(bench, open_turn=True)
    assert len(bench.log.types_of(session_id)) > PagingLog.default_page

    outcomes = await bench.reaper.sweep()

    assert (
        bench.verdict_for_session(outcomes, session_id)
        is ReapVerdict.A_TURN_IS_STILL_OPEN
    )
    assert bench.cluster.holds(session_id)


async def test_the_activity_window_is_read_once_for_the_whole_sweep(
    bench: _Bench,
) -> None:
    """A window read per pod slides across the sweep, judging two pods differently."""
    for _ in range(4):
        _a_session(bench)
    scans = 0
    real_scan = bench.log.lifecycle_events_between

    async def counted(
        types: Sequence[str], from_ms: int, to_ms: int
    ) -> Sequence[Event]:
        nonlocal scans
        scans += 1
        return await real_scan(types, from_ms, to_ms)

    bench.log.lifecycle_events_between = counted  # type: ignore[method-assign]

    await bench.reaper.sweep()

    assert scans == 1


def test_the_grace_period_is_longer_than_the_worst_measured_placement() -> None:
    """The floor the number was chosen against, asserted rather than left in prose.

    A placement can take 660 seconds when an autoscaled node has to arrive -- measured
    back when a first Turn's HTTP response was held for the whole of it, which is how
    the number came to be a test client's timeout. The Turn no longer waits inside that
    response, but the placement still takes as long, so a grace below this can still
    expire inside one Turn. The open-Turn guard would refuse that Session anyway, and
    the whole guarantee would then rest on one check instead of three.
    """
    worst_measured_placement_ms = 660_000
    assert worst_measured_placement_ms < IDLE_GRACE_MS
