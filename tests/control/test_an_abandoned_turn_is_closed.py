"""The sweep that closes the Turns a dead control plane left open, and what it refuses.

Tier 1 (local, no infrastructure). Realizes the wedge: nothing appends `turn.failed`
outside the request path, so a control plane killed mid-Turn leaves a Session that
refuses its next Turn, refuses its archive, and pins the pod sweep -- permanently, with
no call a tenant can make that recovers it.

**The Session's own operations are the oracle, not the sweep's return value.** Two cases
below close a Turn and then call `admit_turn` and `archive_session` for real, because
those two refusals are what "wedged" actually means to a tenant. A sweep that appended
something a fold did not read would pass a test about verdicts and change nothing.

**Pods are real objects here**, held in a dict by `TrackedCluster` and listed back, so a
claim about whether a Turn's pod is present is answered by the cluster rather than by a
recorded call.

The case that matters most is the cold-placement one. A first Turn's placement was
measured holding for 660 seconds while an autoscaled node arrived, and for every second
of it the Turn is open and no pod exists -- so a sweep reading "no pod" as "abandoned"
kills a live Turn, and the two-minute grace does not save it because 120 is less than
660. What saves it is `turn.started`, and the case asserts that at a Turn age far past
both the grace and the measured worst wait.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Collection, Sequence
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from managed_agent.control.session.abandoned_turns import (
    PLACEMENT_DEADLINE_MS,
    POD_GONE_GRACE_MS,
    REPORTS_CEASED_MS,
    STUCK_IDLE_MS,
    AbandonedTurnSweeper,
    TurnOutcome,
    TurnVerdict,
)
from managed_agent.control.session.lifecycle import (
    ArchiveRefused,
    SessionArchived,
    TurnAdmitted,
    TurnRefused,
    admit_turn,
    archive_session,
    whole_log,
)
from managed_agent.control.session.placement import PlacedPod, PodPhase, pod_name_for
from managed_agent.control.session.turn_execution import run_turn
from managed_agent.core.ids import (
    FIRST_SEQ,
    Seq,
    SessionId,
    TurnId,
    new_session_id,
    new_turn_id,
)
from managed_agent.core.session.session import SessionState
from managed_agent.core.session.turns import open_turn
from managed_agent.core.vocabulary import lifecycle, turn

# The reporting cadence the silence deadline has to clear, read off the emitter rather
# than restated. A second copy of this number could fall behind the shim's and would
# then assert the relationship the deadline is sized for against a cadence nobody runs.
from managed_agent.session_shim.turn_runner import _PROGRESS_INTERVAL_S

_NOW_MS = 1_800_000_000_000
_A_MINUTE_MS = 60_000

_A_LONG_TIME_MS = 3 * 60 * 60 * 1000
"""An age well past what any ceiling would once have allowed.

Used wherever a case needs a Turn that is simply old. Since the age ceiling was removed
this closes nothing on its own, which is exactly why it is the right value to set up a
case whose subject is some *other* signal -- if age ever starts closing Turns again,
every one of those cases changes verdict and says so.
"""

WORST_MEASURED_PLACEMENT_MS = 660_000
"""The longest a first Turn was measured waiting for an autoscaled node, in ms.

Written out here rather than referred to in prose, because it is the number the
cold-placement case is about: a Turn this old with no pod is an ordinary event on this
platform, not an abandoned one.
"""


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class PagingLog:
    """Both log ports over one list, capped at two rows a read as the adapter is.

    Also answers the cross-Session window question, because the one adapter behind both
    is `PostgresEventLogRange`. The window is honoured against a per-event timestamp
    this fake keeps beside each row -- the real column is `appended_at` and the port
    hands back no timestamp at all, which is why the sweep asks "which Sessions appear
    in this window" rather than "how old is this Turn".

    The two-row page cap is load-bearing rather than incidental: every fold under test
    here is longer than two rows, so a sweep that read one page and called it the log
    would see a Turn that is not open and do nothing at all.
    """

    default_page = 2

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._at_ms: list[int] = []
        self.rows_read = 0

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        await asyncio.sleep(0)
        return self.add(session_id, type_, payload, at_ms=_NOW_MS)

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

    def payloads_of(self, session_id: SessionId, type_: str) -> list[dict[str, object]]:
        return [
            e.payload
            for e in self._events
            if e.session_id == session_id and e.type == type_
        ]

    def open_turn_of(self, session_id: SessionId) -> TurnId | None:
        return open_turn(
            [event for event in self._events if event.session_id == session_id]
        )

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int | None = None
    ) -> Sequence[Event]:
        await asyncio.sleep(0)
        span = [
            event
            for event in self._events
            if event.session_id == session_id and start <= event.seq <= end
        ]
        page = span[: self.default_page if limit is None else limit]
        self.rows_read += len(page)
        return page

    async def turn_boundaries_of(
        self, session_id: SessionId, types: Collection[str]
    ) -> Sequence[Event]:
        await asyncio.sleep(0)
        wanted = set(types)
        page = [
            event
            for event in self._events
            if event.session_id == session_id and event.type in wanted
        ]
        self.rows_read += len(page)
        return page

    async def latest_progress_of(
        self, session_id: SessionId, type_: str, turn_id: str
    ) -> Sequence[Event]:
        """At most one row, as the adapter's `ORDER BY seq DESC LIMIT 1` returns."""
        await asyncio.sleep(0)
        page = [
            event
            for event in self._events
            if event.session_id == session_id
            and event.type == type_
            and event.payload.get("turn_id") == turn_id
        ][-1:]
        self.rows_read += len(page)
        return page

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
    """A cluster that holds and lists pods, as the real adapter does.

    `placed_pods` is the only method this sweep uses, and the absence of a `remove` here
    is deliberate: closing a Turn must not delete a pod, because reclaiming pods belongs
    to `reaper.py`. A sweep that grew a handback would fail to construct against this.
    """

    def __init__(self) -> None:
        self._pods: dict[str, PlacedPod] = {}

    def place(
        self,
        session_id: SessionId,
        *,
        phase: PodPhase = PodPhase.RUNNING,
        age_ms: int = _A_MINUTE_MS,
    ) -> None:
        self._pods[pod_name_for(session_id)] = PlacedPod(
            session_id=session_id, phase=phase, created_at_ms=_NOW_MS - age_ms
        )

    def evict(self, session_id: SessionId) -> None:
        self._pods.pop(pod_name_for(session_id), None)

    async def placed_pods(self) -> Sequence[PlacedPod]:
        await asyncio.sleep(0)
        return list(self._pods.values())


class MovableClock:
    """A clock a case advances by hand, because two of the signals are about elapsed
    time and a frozen clock can express neither."""

    def __init__(self) -> None:
        self.at_ms = _NOW_MS

    def now_epoch_ms(self) -> int:
        return self.at_ms

    def advance(self, by_ms: int) -> None:
        self.at_ms += by_ms


@dataclass(frozen=True, slots=True)
class _Bench:
    log: PagingLog
    cluster: TrackedCluster
    clock: MovableClock
    sweeper: AbandonedTurnSweeper

    def verdict_for(
        self, outcomes: Sequence[TurnOutcome], session_id: SessionId
    ) -> TurnVerdict:
        matching = [o for o in outcomes if o.session_id == session_id]
        assert len(matching) == 1, f"{len(matching)} outcomes for {session_id}"
        return matching[0].verdict

    def a_new_sweeper(self) -> AbandonedTurnSweeper:
        """A second replica: the same log and the same cluster, its own memory."""
        return AbandonedTurnSweeper(
            pods=self.cluster,
            scan=self.log,
            events=self.log,
            log=self.log,
            clock=self.clock,
        )


@pytest.fixture
def bench() -> _Bench:
    log = PagingLog()
    cluster = TrackedCluster()
    clock = MovableClock()
    return _Bench(
        log=log,
        cluster=cluster,
        clock=clock,
        sweeper=AbandonedTurnSweeper(
            pods=cluster, scan=log, events=log, log=log, clock=clock
        ),
    )


def _a_session_mid_turn(
    bench: _Bench,
    *,
    submitted_ms_ago: int,
    the_pod_answered: bool,
    the_pod_is_there: bool,
    it_reported_idling_for: Sequence[int] = (),
) -> tuple[SessionId, TurnId]:
    """One Session with an open Turn, dialled to whichever branch a case is about.

    The log is built from the events that actually cause the state rather than from a
    stored field, because a fold is the only way a Session has one. The first Turn is
    complete and the second is not, which is also what puts every fold past the two-row
    page cap.
    """
    session_id = new_session_id()
    log = bench.log
    at = bench.clock.at_ms - submitted_ms_ago
    log.add(
        session_id,
        lifecycle.SESSION_CREATED,
        {"environment_id": str(uuid4())},
        at_ms=at,
    )
    first = new_turn_id()
    log.add(session_id, turn.TURN_SUBMITTED, {"turn_id": str(first)}, at_ms=at)
    log.add(session_id, turn.TURN_COMPLETED, {"turn_id": str(first)}, at_ms=at)
    open_id = new_turn_id()
    log.add(session_id, turn.TURN_SUBMITTED, {"turn_id": str(open_id)}, at_ms=at)
    if the_pod_answered:
        log.add(
            session_id,
            turn.TURN_STARTED,
            {"turn_id": str(open_id), "placement_waited_ms": 0},
            at_ms=at,
        )
    for idle_ms in it_reported_idling_for:
        log.add(
            session_id,
            turn.TURN_PROGRESS,
            {"turn_id": str(open_id), "frames": 1, "idle_ms": idle_ms},
            at_ms=at,
        )
    if the_pod_is_there:
        bench.cluster.place(session_id)
    return session_id, open_id


# One dial per verdict, so the parametrized case below reaches every member. A member
# no set of dials can reach is a branch nothing exercises, which is how four of five
# refusal reasons in `pod_runner.py` came to be graded by nothing.
_REACHES: dict[TurnVerdict, dict[str, object]] = {
    TurnVerdict.THE_POD_WAS_NEVER_PLACED: {
        "submitted_ms_ago": WORST_MEASURED_PLACEMENT_MS,
        "the_pod_answered": False,
        "the_pod_is_there": False,
    },
    TurnVerdict.THE_POD_IS_STILL_THERE: {
        "submitted_ms_ago": WORST_MEASURED_PLACEMENT_MS,
        "the_pod_answered": True,
        "the_pod_is_there": True,
    },
    TurnVerdict.THE_POD_HAS_ONLY_JUST_GONE: {
        "submitted_ms_ago": WORST_MEASURED_PLACEMENT_MS,
        "the_pod_answered": True,
        "the_pod_is_there": False,
    },
    TurnVerdict.ITS_RUNTIME_STOPPED_TALKING: {
        "submitted_ms_ago": WORST_MEASURED_PLACEMENT_MS,
        "the_pod_answered": True,
        "the_pod_is_there": True,
        "it_reported_idling_for": (STUCK_IDLE_MS,),
    },
    # Reached over two sweeps rather than one, because this verdict is about a
    # *continuously* observed absence and a single sweep can only ever start the count.
    # The dial below is the setup; the second sweep is added by the case itself.
    TurnVerdict.IT_NEVER_GOT_A_POD: {
        "submitted_ms_ago": 0,
        "the_pod_answered": False,
        "the_pod_is_there": False,
        "then_wait_ms": PLACEMENT_DEADLINE_MS,
    },
    # Also two sweeps: the first can only record where the report sequence had got to,
    # and the silence is the distance between that observation and the next one.
    TurnVerdict.ITS_REPORTS_STOPPED_ARRIVING: {
        "submitted_ms_ago": WORST_MEASURED_PLACEMENT_MS,
        "the_pod_answered": True,
        "the_pod_is_there": True,
        "it_reported_idling_for": (0,),
        "then_wait_ms": REPORTS_CEASED_MS,
    },
}


@pytest.mark.parametrize("verdict", list(_REACHES))
async def test_the_sweep_can_reach_this_verdict(
    verdict: TurnVerdict, bench: _Bench
) -> None:
    """Every member of the closed set is reachable from some real Session.

    Parametrized over the members rather than asserted in one case per branch, because
    the set is what a caller counts and a member nothing can produce is a decision no
    test grades.
    """
    dials = dict(_REACHES[verdict])
    # Two of the three closing verdicts are about a *continuously* observed absence, so
    # one sweep can only start their count. A dial carrying a wait spends it between two
    # sweeps rather than before the first, which is the only order that builds the
    # continuity those verdicts are asserting on.
    waits_ms = dials.pop("then_wait_ms", 0)
    assert isinstance(waits_ms, int)
    session_id, _ = _a_session_mid_turn(bench, **dials)  # type: ignore[arg-type]
    if waits_ms:
        await bench.sweeper.sweep()
        bench.clock.advance(waits_ms)
    assert bench.verdict_for(await bench.sweeper.sweep(), session_id) is verdict


async def test_a_turn_waiting_out_a_cold_placement_is_left_alone(
    bench: _Bench,
) -> None:
    """The case a naive implementation kills: open, no pod, and legitimately so.

    Held at the worst wait ever measured on this platform -- 660 seconds for an
    autoscaled node -- which is five and a half times the pod-gone grace. So this is not
    passing because the grace has not expired; the grace expired long ago. It passes
    because no `turn.started` was ever appended for this Turn, which is the only thing
    that distinguishes a pod that has not been placed yet from one that has died.
    """
    assert POD_GONE_GRACE_MS < WORST_MEASURED_PLACEMENT_MS, (
        "the grace is shorter than the platform's own worst placement wait, which is "
        "why it cannot be what protects a Turn in cold placement"
    )
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=False,
        the_pod_is_there=False,
    )

    outcomes = await bench.sweeper.sweep()

    assert (
        bench.verdict_for(outcomes, session_id) is TurnVerdict.THE_POD_WAS_NEVER_PLACED
    )
    assert bench.log.open_turn_of(session_id) == turn_id
    assert turn.TURN_FAILED not in bench.log.types_of(session_id)


async def test_a_turn_that_never_got_a_pod_is_closed_once_placement_has_run_out(
    bench: _Bench,
) -> None:
    """The hole the removed ceiling used to cover: a Turn that never reached a pod.

    Signal 1 cannot see this Turn -- it rests on `turn.started`, which is exactly what a
    Turn still in placement has never appended -- and signal 2 reads a report the pod
    would have had to be alive to write. So before this bound the Turn stayed open for
    the life of the Session, refusing its own successor and refusing the archive.

    The wait is asserted from both sides in one case, because a bound is two claims and
    a test of only the closing half would pass with a deadline of zero.

    **The deadline is spent between two sweeps, not before the first**, and the case is
    written that way because the count is of *observed* absence rather than of age since
    submission. `EventRecord` carries no timestamp, so the sweep cannot read how long a
    Turn has been waiting -- it can only count the sweeps that saw it waiting. A control
    plane that restarts therefore starts the count again, which delays a close and can
    never bring one forward. Advancing the clock before the first sweep would assert the
    opposite property.
    """
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=0,
        the_pod_answered=False,
        the_pod_is_there=False,
    )

    opening = await bench.sweeper.sweep()
    assert (
        bench.verdict_for(opening, session_id) is TurnVerdict.THE_POD_WAS_NEVER_PLACED
    )

    bench.clock.advance(PLACEMENT_DEADLINE_MS - _A_MINUTE_MS)
    early = await bench.sweeper.sweep()
    assert bench.verdict_for(early, session_id) is TurnVerdict.THE_POD_WAS_NEVER_PLACED
    assert bench.log.open_turn_of(session_id) == turn_id

    bench.clock.advance(_A_MINUTE_MS)
    late = await bench.sweeper.sweep()

    assert bench.verdict_for(late, session_id) is TurnVerdict.IT_NEVER_GOT_A_POD
    assert bench.log.open_turn_of(session_id) is None


async def test_the_placement_bound_says_the_runtime_did_not_start(
    bench: _Bench,
) -> None:
    """The cause a tenant reads, and why it is not the one the other two signals use.

    `RUNTIME_LOST` means the work was lost with the process carrying it. Nothing was
    carrying this one: no pod was ever running, so nothing ran and nothing was lost.
    `RUNTIME_DID_NOT_START` is the closed set's member for exactly that, and its remedy
    differs -- which is the entire reason the set was widened on 2026-08-26 rather than
    left answering four situations with one name.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=0,
        the_pod_answered=False,
        the_pod_is_there=False,
    )

    await bench.sweeper.sweep()
    bench.clock.advance(PLACEMENT_DEADLINE_MS)
    await bench.sweeper.sweep()

    failures = bench.log.payloads_of(session_id, turn.TURN_FAILED)
    assert [f["cause"] for f in failures] == [
        turn.TurnFailureCause.RUNTIME_DID_NOT_START.value
    ]


async def test_a_pod_arriving_late_clears_the_placement_count(
    bench: _Bench,
) -> None:
    """Absence has to be continuous, so a placement that succeeds late is not punished.

    A Turn that waits out most of the deadline and *then* gets its pod must start from
    zero if that pod later goes missing, rather than inheriting the waiting it already
    did. Without the clear, a slow placement followed by a pod dying would close on the
    placement bound the moment the deadline arrived -- reporting `runtime_did_not_start`
    for a runtime that demonstrably started.

    The final wait is a whole deadline rather than the remainder of one, which is what
    makes this fail if the clear is removed: without it the count still runs from the
    first sweep, and a full deadline past that is a close. A remainder would pass either
    way and grade nothing.
    """
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=0,
        the_pod_answered=False,
        the_pod_is_there=False,
    )
    await bench.sweeper.sweep()

    bench.cluster.place(session_id)
    await bench.sweeper.sweep()
    bench.cluster.evict(session_id)

    bench.clock.advance(PLACEMENT_DEADLINE_MS)
    after = await bench.sweeper.sweep()

    assert bench.verdict_for(after, session_id) is TurnVerdict.THE_POD_WAS_NEVER_PLACED
    assert bench.log.open_turn_of(session_id) == turn_id


def test_the_placement_deadline_clears_the_worst_placement_ever_measured() -> None:
    """The one number this bound must not get wrong, asserted rather than commented.

    A cold placement was measured holding 660 seconds while an autoscaled node arrived,
    and for every second of it the Turn is open with no pod -- indistinguishable from
    the Turn this bound exists to close. A deadline at or under that measurement closes
    live Turns waiting on capacity, which is the precise failure the removed ceiling was
    retired for. The margin is deliberately large: 660 seconds is the worst *observed*
    wait, not a proven maximum, and the cost of waiting too long is a Session idle a few
    extra minutes against a Turn killed mid-flight.
    """
    assert PLACEMENT_DEADLINE_MS >= 2 * WORST_MEASURED_PLACEMENT_MS, (
        f"a placement deadline of {PLACEMENT_DEADLINE_MS}ms leaves too little margin "
        f"over the worst placement wait this platform has measured "
        f"({WORST_MEASURED_PLACEMENT_MS}ms), which is the only thing standing between "
        "this bound and a Turn killed while it waits for a node"
    )


async def test_a_turn_whose_pod_answered_and_then_vanished_is_closed_after_the_grace(
    bench: _Bench,
) -> None:
    """The fast path, and both halves of it: not on sight, and not never.

    One sweep with the pod gone must keep the Turn -- an instantaneous absence is one
    read of a cluster, not evidence. Two minutes later the same absence is evidence, and
    the Turn is closed with a cause a tenant can act on.
    """
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
    )
    bench.cluster.evict(session_id)

    first = await bench.sweeper.sweep()
    assert (
        bench.verdict_for(first, session_id) is TurnVerdict.THE_POD_HAS_ONLY_JUST_GONE
    )
    assert bench.log.open_turn_of(session_id) == turn_id

    bench.clock.advance(POD_GONE_GRACE_MS)
    second = await bench.sweeper.sweep()

    assert bench.verdict_for(second, session_id) is TurnVerdict.ITS_POD_IS_GONE
    assert bench.log.open_turn_of(session_id) is None
    assert bench.log.payloads_of(session_id, turn.TURN_FAILED) == [
        {
            "turn_id": str(turn_id),
            "cause": turn.TurnFailureCause.RUNTIME_LOST.value,
        }
    ]


async def test_a_pod_that_ran_to_completion_is_gone_even_while_still_listed(
    bench: _Bench,
) -> None:
    """The commonest shape of this defect, and the one a phase-blind read misses.

    A control plane that dies mid-Turn leaves a pod that finishes on its own, and
    nothing collects it -- `_make_way_for_the_next_pod` calls that "the residue of a
    Turn whose control plane died". The object therefore stays in the listing for ever
    with `_phase_of` answering GONE. A sweep that counted any listed pod as present
    would reset the grace on every pass and never reach the fast path at all, leaving
    the hour-long ceiling as the only thing that ever closed the commonest case -- the
    case the two-minute signal was written for.
    """
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=False,
    )
    bench.cluster.place(session_id, phase=PodPhase.GONE)

    first = await bench.sweeper.sweep()
    assert (
        bench.verdict_for(first, session_id) is TurnVerdict.THE_POD_HAS_ONLY_JUST_GONE
    )

    bench.clock.advance(POD_GONE_GRACE_MS)
    second = await bench.sweeper.sweep()

    assert bench.verdict_for(second, session_id) is TurnVerdict.ITS_POD_IS_GONE
    assert bench.log.open_turn_of(session_id) is None


async def test_a_pod_that_comes_back_between_sweeps_restarts_the_grace(
    bench: _Bench,
) -> None:
    """The absence has to be continuous, and this is what "continuous" is worth.

    A Turn seen without a pod, then with one, then without again must wait the whole
    grace from the second absence rather than inheriting the first one's clock. Without
    that, two unrelated unlucky reads a day apart would add up to a close.
    """
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=False,
    )
    await bench.sweeper.sweep()

    bench.cluster.place(session_id)
    bench.clock.advance(POD_GONE_GRACE_MS)
    assert (
        bench.verdict_for(await bench.sweeper.sweep(), session_id)
        is TurnVerdict.THE_POD_IS_STILL_THERE
    )

    bench.cluster.evict(session_id)
    assert (
        bench.verdict_for(await bench.sweeper.sweep(), session_id)
        is TurnVerdict.THE_POD_HAS_ONLY_JUST_GONE
    ), "the grace restarted from the pod's return, not from the first absence"
    assert bench.log.open_turn_of(session_id) == turn_id


async def test_an_old_turn_with_a_live_pod_is_left_alone_however_old_it_gets(
    bench: _Bench,
) -> None:
    """Age is not evidence of death, and this is the case that says so.

    The inverse of what stood here until 2026-08-26, which asserted that a Turn past
    sixty minutes was closed whatever its pod was doing. That rule came within
    twenty-two minutes of ending a real delegating run that was visibly healthy, and the
    reason it could is that a clock cannot tell a long agent run from a dead one. Three
    hours old, pod present, reporting a healthy idle: kept.

    The `it_reported_idling_for` value is deliberately a healthy one rather than absent.
    Absent would also pass, but for the wrong reason -- signal 2 never acts on a missing
    report, so the case would hold even if age *and* idle both closed Turns.
    """
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS // 10,),
    )

    outcomes = await bench.sweeper.sweep()

    assert bench.verdict_for(outcomes, session_id) is TurnVerdict.THE_POD_IS_STILL_THERE
    assert bench.log.open_turn_of(session_id) == turn_id, (
        "a Turn was closed for its age, which is the rule that was removed -- a long "
        "agent run is now indistinguishable to this sweep from a short one, and only "
        "what the pod reports about itself may close it"
    )
    assert bench.log.types_of(session_id).count(turn.TURN_FAILED) == 0


async def test_a_turn_stuck_in_placement_stays_open_and_that_is_a_known_hole(
    bench: _Bench,
) -> None:
    """A Turn that never got a pod is now never closed, and this records the cost.

    Not an endorsement. Removing the age ceiling removed the only thing that ended a
    Turn with no pod and no `turn.started`: signal 1 needs a pod to have existed, and
    signal 2 needs a report only a pod can send. So this Session is wedged -- it refuses
    every later Turn and cannot be archived.

    Asserted rather than left undiscovered, because a hole nothing tests is a hole the
    next reader finds in production. When a placement bound is built, this case fails
    and is replaced by one asserting *that* bound -- which is the correct way for it to
    end, and the reason it is written as an assertion rather than a comment.
    """
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=False,
        the_pod_is_there=False,
    )

    outcomes = await bench.sweeper.sweep()

    assert (
        bench.verdict_for(outcomes, session_id) is TurnVerdict.THE_POD_WAS_NEVER_PLACED
    )
    assert bench.log.open_turn_of(session_id) == turn_id


async def test_a_session_with_nothing_open_is_not_swept_at_all(bench: _Bench) -> None:
    """The prescreen's other direction: a closed Turn is not a candidate.

    Asserted as an absence from the outcomes rather than as a keep verdict, because the
    two mean different things to a caller counting them -- and because folding the whole
    log of every Session that ran today is the cost this screen exists to avoid.
    """
    session_id = new_session_id()
    bench.log.add(
        session_id, lifecycle.SESSION_CREATED, {"environment_id": str(uuid4())}
    )
    done = new_turn_id()
    bench.log.add(session_id, turn.TURN_SUBMITTED, {"turn_id": str(done)})
    bench.log.add(session_id, turn.TURN_COMPLETED, {"turn_id": str(done)})
    bench.cluster.place(session_id)

    assert [o.session_id for o in await bench.sweeper.sweep()] == []


async def test_two_replicas_sweeping_the_same_turn_close_it_once(
    bench: _Bench,
) -> None:
    """Trap 3: the sweep is idempotent by construction rather than by a lock.

    Two sweepers over one log and one cluster, each with its own memory, exactly as two
    control-plane replicas are. The second must find the Turn closed and append nothing:
    two `turn.failed` events for one Turn would be the platform reporting one crash
    twice into a tenant's history.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS,),
    )
    other = bench.a_new_sweeper()

    await bench.sweeper.sweep()
    await other.sweep()
    await bench.sweeper.sweep()

    assert bench.log.types_of(session_id).count(turn.TURN_FAILED) == 1


async def test_closing_the_turn_lets_the_session_take_its_next_turn(
    bench: _Bench,
) -> None:
    """Half of what "wedged" means, asserted against the real admission path.

    `admit_turn` requires `state.accepts_a_turn()`, which is `IDLE` and nothing else, so
    a Session with an open Turn refuses every later submission. The refusal before the
    sweep is asserted too -- without it this case would pass against a platform that was
    never wedged.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS,),
    )
    refused = await admit_turn(
        session_id, "key-before-0001", "hello", bench.log, bench.log
    )
    assert refused == TurnRefused(state=SessionState.RUNNING)

    await bench.sweeper.sweep()

    admitted = await admit_turn(
        session_id, "key-after-00001", "hello", bench.log, bench.log
    )
    assert isinstance(admitted, TurnAdmitted)


async def test_closing_the_turn_lets_the_session_be_archived(bench: _Bench) -> None:
    """The other half, asserted against the real archive path.

    `archive_session` answers `ArchiveRefused` while `open_turn` names a Turn, so a
    tenant cannot even end a Session a dead control plane wedged. Both sides are
    asserted for the same reason as above.
    """
    session_id, turn_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS,),
    )
    release = _CountingRelease()
    before = await archive_session(session_id, bench.log, bench.log, release)
    assert before == ArchiveRefused(turn_id=turn_id)

    await bench.sweeper.sweep()

    after = await archive_session(session_id, bench.log, bench.log, release)
    assert isinstance(after, SessionArchived)


class _CountingRelease:
    """A handback that records the Sessions it was asked about.

    Present so `archive_session` can be called for real rather than against a null
    object that would let a refusal and a success look the same from outside.
    """

    def __init__(self) -> None:
        self.released: list[SessionId] = []

    async def release(self, session_id: SessionId) -> None:
        await asyncio.sleep(0)
        self.released.append(session_id)


async def test_the_sweep_never_deletes_a_pod(bench: _Bench) -> None:
    """The boundary between this sweep and the one that owns pods.

    Reclaiming a pod is `reaper.py`'s single job, and it was already willing to reclaim
    this one -- its third guard was the only thing stopping it. Two sweeps deleting pods
    would be two answers to one question, and the pod outliving this close by one reaper
    pass is the intended shape rather than a leak.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS,),
    )

    await bench.sweeper.sweep()

    assert [pod.session_id for pod in await bench.cluster.placed_pods()] == [session_id]


async def test_one_unreadable_session_does_not_stop_the_rest(bench: _Bench) -> None:
    """A single bad Session must not restore the wedge for the whole platform.

    That failure mode is this file's own defect arriving by another route: a sweep that
    stops at the first raise leaves every Session behind it open for ever, and the
    symptom is indistinguishable from the sweep never having been wired.
    """
    broken, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS,),
    )
    healthy, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS,),
    )
    real_read = bench.log.read

    async def refuse_the_broken_one(
        session_id: SessionId, start: Seq, end: Seq, limit: int | None = None
    ) -> Sequence[Event]:
        if session_id == broken:
            raise RuntimeError("this Session's log cannot be read")
        return await real_read(session_id, start, end, limit)

    bench.log.read = refuse_the_broken_one  # type: ignore[method-assign]
    outcomes = await bench.sweeper.sweep()

    assert bench.verdict_for(outcomes, broken) is TurnVerdict.THE_SWEEP_COULD_NOT_DECIDE
    assert (
        bench.verdict_for(outcomes, healthy) is TurnVerdict.ITS_RUNTIME_STOPPED_TALKING
    )


def test_the_only_closing_threshold_is_shorter_than_a_cold_placement() -> None:
    """`STUCK_IDLE_MS` is 600 000ms and the worst measured placement is 660 000ms.

    Recorded rather than guarded, and the direction is the surprising one: the only
    threshold that can close a live pod's Turn is *shorter* than a placement this
    platform has legitimately taken. The case that stood here asserted the removed
    ceiling cleared that wait, and the obvious replacement -- assert this threshold
    clears it too -- is simply false.

    It is safe anyway, and the reason is a precondition rather than a number. A Turn in
    placement has no pod, a `turn.progress` report can only come from a pod, and this
    signal never acts on a report's absence. So the threshold is never consulted during
    a placement at all, however long the placement runs.

    That is a narrow protection resting on one fact, which is why it is written down
    here: if this signal ever learns to act on silence, or a report ever reaches the log
    from anywhere but a running pod, a cold placement starts being closed as a wedge and
    nothing in the arithmetic would warn anyone.
    """
    assert STUCK_IDLE_MS < WORST_MEASURED_PLACEMENT_MS, (
        "the numbers moved; re-read this docstring, because the argument above is why "
        "the ordering was safe and it may no longer describe the platform"
    )


def test_every_verdict_that_closes_a_turn_says_so() -> None:
    """The four closing members are the four the sweep acts on, and no others.

    Asserted against the whole set rather than member by member, so a member added later
    is caught here as well as by `mypy --strict`.
    """
    closing = {v for v in TurnVerdict if v.closed_the_turn()}
    assert closing == {
        TurnVerdict.ITS_POD_IS_GONE,
        TurnVerdict.ITS_RUNTIME_STOPPED_TALKING,
        TurnVerdict.IT_NEVER_GOT_A_POD,
        TurnVerdict.ITS_REPORTS_STOPPED_ARRIVING,
    }


async def test_a_turn_whose_runtime_stopped_talking_is_closed_on_its_own_report(
    bench: _Bench,
) -> None:
    """The second signal: a live pod whose own report says the runtime went quiet.

    This is the wedge the pod signal cannot see. The pod is there and `Ready`, so signal
    1 never fires, and since the ceiling was removed on 2026-08-26 there is no clock
    behind it either -- this signal is now the only thing that closes a Turn whose pod
    is alive. What has actually happened is that the process inside the pod stopped
    speaking to the shim, and the only thing in the platform that can see that is the
    pod's own progress report.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS,),
    )

    outcomes = await bench.sweeper.sweep()

    assert (
        bench.verdict_for(outcomes, session_id)
        is TurnVerdict.ITS_RUNTIME_STOPPED_TALKING
    )
    assert open_turn(await whole_log(bench.log, session_id)) is None, (
        "the verdict said the Turn was closed and the log still has it open"
    )


async def test_a_turn_that_never_reported_progress_is_never_closed_for_it(
    bench: _Bench,
) -> None:
    """**The invariant.** Absence of reports is never evidence of a stuck runtime.

    `runtime_image` is tenant-supplied and digest-pinned, and nothing ever forces a
    tenant off an old digest -- so a pod that emits no progress at all is not a
    transitional population that drains after a rollout, it is one any tenant can create
    at any time, permanently. If this sweep ever treated silence as a signal it would
    close every Turn on every environment pinned to an image built before the emitter,
    and it would present to that tenant as the platform killing working agents.

    Dialled to a Turn far older than any report cadence, with a live pod, so the only
    thing separating it from the case above is that it has said nothing.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
    )

    outcomes = await bench.sweeper.sweep()

    assert bench.verdict_for(outcomes, session_id) is TurnVerdict.THE_POD_IS_STILL_THERE
    assert open_turn(await whole_log(bench.log, session_id)) is not None, (
        "a Turn that reported nothing was closed anyway, which is the one outcome this "
        "signal must never produce -- every pod running a pre-emitter image reports "
        "nothing, and there is no mechanism that ever migrates them"
    )


async def test_a_turn_reporting_a_healthy_idle_is_left_alone(bench: _Bench) -> None:
    """A runtime that is talking is not stuck, however long the Turn has run.

    The measured figure from a live pod told to sleep 150 seconds was an `idle_ms`
    between 16 and 24 seconds throughout -- the runtime talks to the shim continuously
    even while the published event stream is silent. So a healthy report is a small
    number, and this case is dialled just under the threshold rather than to zero,
    because zero would also pass a comparison that had been written backwards.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS - 1,),
    )

    assert (
        bench.verdict_for(await bench.sweeper.sweep(), session_id)
        is TurnVerdict.THE_POD_IS_STILL_THERE
    )


async def test_only_the_most_recent_report_decides(bench: _Bench) -> None:
    """A runtime that went quiet and then came back is working, not stuck.

    Without this, the rule would be "some report once said stuck", which is a latch: a
    single long model call early in a Turn would condemn the whole Turn however much
    work followed it. The reports are a running commentary and only the last one
    describes the present.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(STUCK_IDLE_MS * 2, 1_000),
    )

    assert (
        bench.verdict_for(await bench.sweeper.sweep(), session_id)
        is TurnVerdict.THE_POD_IS_STILL_THERE
    )


def test_the_stuck_threshold_clears_the_longest_measured_healthy_silence() -> None:
    """The only threshold that can close a live pod's Turn, against real measurements.

    Replaces a case comparing this threshold to the removed ceiling. That comparison
    answered "is the signal worth having", which the ceiling's removal settles outright:
    it is now the only thing that closes a Turn whose pod is alive.

    The question that outlives it is the dangerous one. The largest `idle_ms` ever seen
    on healthy work is 72 053 ms, measured on a real delegating review with `frames`
    frozen across three consecutive reports -- and that figure has risen every time the
    workload got more realistic (24s synthetic, then 31s, 42s, 72s). A threshold that
    drifts down toward it kills working agents, and it would do so silently, because
    every case above would still pass.
    """
    largest_healthy_idle_ms = 72_053
    assert largest_healthy_idle_ms * 5 < STUCK_IDLE_MS, (
        f"a threshold of {STUCK_IDLE_MS}ms leaves too little margin over the largest "
        f"measured healthy silence of {largest_healthy_idle_ms}ms; healthy work has "
        "reached a higher number every time it was measured against a more realistic "
        "workload, so this margin is the whole protection"
    )


def test_the_placement_deadline_outlasts_the_longest_a_placement_can_run() -> None:
    """The bound's real safety argument, asserted against the numbers it rests on.

    A margin over the worst *measured* placement is not what makes this deadline safe --
    a measurement is not a maximum. What makes it safe is that a placement cannot run
    forever by construction: the pod runner waits at most `_SCHEDULING_TIMEOUT_SECONDS`
    for a node and then at most `_READY_TIMEOUT_SECONDS` for the pod, and gives up. Past
    that sum, any process that was placing this Turn has already failed it or died, so
    the sweep is never closing a Turn out from under a live placement.

    Imported from the adapter rather than restated, because a restated copy is free to
    fall behind the numbers it claims to track -- and the failure mode when it does is
    silent: the deadline keeps passing this test while live placements start outrunning
    it, and tenants get `runtime_did_not_start` for pods that went on to answer.
    """
    from managed_agent.adapters.kubernetes.pod_runner import (
        _READY_TIMEOUT_SECONDS,
        _SCHEDULING_TIMEOUT_SECONDS,
    )

    longest_live_placement_ms = int(
        (_SCHEDULING_TIMEOUT_SECONDS + _READY_TIMEOUT_SECONDS) * 1000
    )
    assert longest_live_placement_ms < PLACEMENT_DEADLINE_MS, (
        f"a placement can run for {longest_live_placement_ms}ms and this deadline "
        f"fires at {PLACEMENT_DEADLINE_MS}ms, so the sweep would close Turns whose "
        "placement is still running -- raise the deadline or lower the runner's waits"
    )


async def test_a_turn_whose_reports_stopped_arriving_is_closed(bench: _Bench) -> None:
    """**The third signal**: the pod is up, it reported, and then it went silent.

    The hole the other two leave. Signal 1 asks the cluster and the pod is there.
    Signal 2 asks the pod's own report and the last one it managed said the runtime was
    perfectly live -- `idle_ms` is a number a *report* carries, so a shim that stops
    reporting freezes it at whatever it last said rather than letting it grow. Between
    them a Turn whose shim died while its pod stayed Running is invisible for ever,
    which is what wedges the Session.

    What this reads instead is the control plane's own observation that the report
    sequence has not moved. The pod appends its reports itself, on its own timer, with
    no control plane involved -- which is exactly what makes their absence mean
    something after a control plane restart, where every other stream has stopped by
    construction.
    """
    session_id, _ = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(0,),
    )

    await bench.sweeper.sweep()
    bench.clock.advance(REPORTS_CEASED_MS)
    outcomes = await bench.sweeper.sweep()

    assert (
        bench.verdict_for(outcomes, session_id)
        is TurnVerdict.ITS_REPORTS_STOPPED_ARRIVING
    )
    assert open_turn(await whole_log(bench.log, session_id)) is None, (
        "the verdict said the Turn was closed and the log still has it open"
    )


async def test_a_turn_that_keeps_reporting_is_left_alone_however_long_it_runs(
    bench: _Bench,
) -> None:
    """The case a naive implementation kills: a long, healthy, talkative Turn.

    Held past the silence deadline twice over, with one further report arriving in
    between. A signal that counted from the *first* report rather than from the newest
    one would close this, and it would close every long agent run on the platform --
    which is the same mistake the retired age ceiling made, arriving through a
    different door.
    """
    session_id, open_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(0,),
    )

    await bench.sweeper.sweep()
    bench.clock.advance(REPORTS_CEASED_MS)
    bench.log.add(
        session_id,
        turn.TURN_PROGRESS,
        {"turn_id": str(open_id), "frames": 2, "idle_ms": 0},
        at_ms=bench.clock.at_ms,
    )
    await bench.sweeper.sweep()
    bench.clock.advance(REPORTS_CEASED_MS - _A_MINUTE_MS)
    outcomes = await bench.sweeper.sweep()

    assert bench.verdict_for(outcomes, session_id) is TurnVerdict.THE_POD_IS_STILL_THERE
    assert bench.log.open_turn_of(session_id) == open_id


async def test_a_turn_that_never_reported_is_not_closed_for_reports_ceasing(
    bench: _Bench,
) -> None:
    """**The invariant that makes this signal safe.** Ceasing is not never having begun.

    Every pod running an image built before the progress emitter reports nothing, for
    ever, while working perfectly. A signal that read "no recent report" would close
    every one of them the moment it shipped. What is read here is a report sequence
    that *moved and then stopped*, which a Turn that never reported can never satisfy
    -- so those pods are outside this signal's reach by construction rather than by a
    version check somebody has to remember to remove.
    """
    session_id, open_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
    )

    await bench.sweeper.sweep()
    bench.clock.advance(REPORTS_CEASED_MS * 3)
    outcomes = await bench.sweeper.sweep()

    assert bench.verdict_for(outcomes, session_id) is TurnVerdict.THE_POD_IS_STILL_THERE
    assert bench.log.open_turn_of(session_id) == open_id
    assert turn.TURN_FAILED not in bench.log.types_of(session_id)


async def test_one_sweep_can_never_close_a_turn_for_silence(bench: _Bench) -> None:
    """A replica that has just started knows nothing about the past, and must not guess.

    The silence is measured between two observations this process made, not from any
    timestamp on the last report -- the log port hands back none. So a control plane
    that comes up next to a Turn silent for hours starts its count at zero and waits
    the full deadline before acting. That costs one deadline after every restart, and
    it is the direction to be wrong in: the alternative is a fresh replica closing
    live Turns during the very restart this module exists to recover from.
    """
    session_id, open_id = _a_session_mid_turn(
        bench,
        submitted_ms_ago=_A_LONG_TIME_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(0,),
    )
    bench.clock.advance(REPORTS_CEASED_MS * 2)

    outcomes = await bench.a_new_sweeper().sweep()

    assert bench.verdict_for(outcomes, session_id) is TurnVerdict.THE_POD_IS_STILL_THERE
    assert bench.log.open_turn_of(session_id) == open_id


async def test_the_next_turn_does_not_inherit_the_last_turns_silence(
    bench: _Bench,
) -> None:
    """A Session runs one Turn after another, and each is silent on its own account.

    The count is keyed on the Turn for the same reason the other two are: a Session is
    a long-lived name and a Turn is the short-lived thing the count is actually about.
    Inherited across the boundary, a previous Turn's silence would be spent closing the
    next Turn seconds after it was submitted.
    """
    session_id, first_open = _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=(0,),
    )
    await bench.sweeper.sweep()
    bench.clock.advance(REPORTS_CEASED_MS - _A_MINUTE_MS)

    bench.log.add(
        session_id,
        turn.TURN_COMPLETED,
        {"turn_id": str(first_open)},
        at_ms=bench.clock.at_ms,
    )
    second_open = new_turn_id()
    fresh: tuple[tuple[str, dict[str, object]], ...] = (
        (turn.TURN_SUBMITTED, {"turn_id": str(second_open)}),
        (turn.TURN_STARTED, {"turn_id": str(second_open), "placement_waited_ms": 0}),
        (turn.TURN_PROGRESS, {"turn_id": str(second_open), "frames": 1, "idle_ms": 0}),
    )
    for type_, payload in fresh:
        bench.log.add(session_id, type_, payload, at_ms=bench.clock.at_ms)

    await bench.sweeper.sweep()
    bench.clock.advance(_A_MINUTE_MS * 2)
    outcomes = await bench.sweeper.sweep()

    assert bench.verdict_for(outcomes, session_id) is TurnVerdict.THE_POD_IS_STILL_THERE
    assert bench.log.open_turn_of(session_id) == second_open


def test_the_silence_deadline_clears_a_long_run_of_dropped_reports() -> None:
    """The deadline is many reporting intervals, not one or two.

    A report is attempted every `_PROGRESS_INTERVAL_S` and a failed append is dropped
    rather than retried, on purpose -- the next one thirty seconds later carries
    everything the lost one did. So a database blip that eats several reports in a row
    is an ordinary event, and a deadline of one or two intervals would read it as a
    dead shim. Twenty intervals is the floor asserted here; the constant is above it.
    """
    assert 20 * int(_PROGRESS_INTERVAL_S * 1000) <= REPORTS_CEASED_MS, (
        "the silence deadline is close enough to the reporting interval that a short "
        "run of dropped reports reads as a dead shim"
    )
    assert REPORTS_CEASED_MS > STUCK_IDLE_MS, (
        "a Turn should be given longer to be silent about itself than to report that "
        "it is idle: the idle report is the pod's own measurement and this is only an "
        "inference from an absence"
    )


def test_the_turn_runner_names_the_bound_that_closes_a_stuck_placement() -> None:
    """`run_turn`'s docstring may not deny a bound this module actually holds.

    That docstring explains why nothing bounds a Turn's length, and it used to close by
    naming what the removal gave up: a Turn stuck in placement, with no pod and no
    `turn.started`, "has nothing ending it ... it wants a placement bound, which is not
    this change". True when it was written, in `195c03f`. False one commit later, when
    `c41b5b4` built exactly that bound -- and nobody went back for the sentence.

    The cost was not untidiness. A prose hole is read as a real one: this claim was
    carried into `docs/session-state.md` as the platform's top open issue and reported
    to the person who owns the work as the one remaining way a Session can wedge, while
    `PLACEMENT_DEADLINE_MS` had been closing that case for a day.

    So the docstring has to name the constant. Asserted on the imported constant's own
    name rather than on a phrase, because a phrase is a second spelling free to drift:
    if the bound is ever deleted this test stops importing, and if the bound is renamed
    the docstring is what fails.
    """
    doc = run_turn.__doc__
    assert doc is not None, "run_turn lost its docstring; it carries this argument"
    assert "PLACEMENT_DEADLINE_MS" in doc, (
        "run_turn's docstring describes a Turn stuck in placement without naming "
        f"{PLACEMENT_DEADLINE_MS}ms `PLACEMENT_DEADLINE_MS`, the bound that closes it. "
        "A docstring that says a case is unhandled while it is handled is read as a "
        "gap, and gets reported as one."
    )


async def _rows_to_sweep_a_turn_reporting(bench: _Bench, reports: int) -> int:
    """How many log rows one sweep pulls for a Session reporting this many times."""
    _a_session_mid_turn(
        bench,
        submitted_ms_ago=WORST_MEASURED_PLACEMENT_MS,
        the_pod_answered=True,
        the_pod_is_there=True,
        it_reported_idling_for=[1_000] * reports,
    )
    bench.log.rows_read = 0
    await bench.sweeper.sweep()
    return bench.log.rows_read


@pytest.mark.anyio
async def test_the_sweep_does_not_reread_a_turns_whole_commentary(
    bench: _Bench,
) -> None:
    """A sweep's cost must not grow with how much the Turn it is judging has said.

    The sweep asks three questions of an open Turn -- did its pod ever answer, what did
    its newest progress report say, and where in the log that report sits -- and every
    one of them is answered by a handful of rows near the two ends of the log. Reading
    the whole log to reach them makes the cost of judging a Turn proportional to how
    long that Turn has been running, which is backwards: the Turns this sweep exists to
    catch are precisely the long ones.

    Measured against real Postgres before this guard: 40 Sessions holding 3,000 events
    each cost one sweep 782.6 ms and 120,000 rows over 280 queries, against 233.6 ms
    and 24,000 rows for the same 40 Sessions holding 600 -- linear in total log size,
    on a pass that runs every 30 seconds while every running Turn appends a report
    every 30 seconds. Nothing was slow yet at that size; the shape is what this holds.

    Asserted as a comparison between two report counts rather than against a constant,
    because the constant would be a second copy of whatever the read strategy happens
    to fetch today and would fail on any harmless change to it. Growth is the defect.
    """
    few = await _rows_to_sweep_a_turn_reporting(bench, 5)
    many = await _rows_to_sweep_a_turn_reporting(bench, 200)
    assert many <= few * 2, (
        f"a Turn with 200 progress reports cost the sweep {many} rows against {few} "
        f"for one with 5. The sweep is reading the commentary it does not need: its "
        f"cost per Turn should be flat in the number of reports, since only the "
        f"newest one is ever read."
    )
