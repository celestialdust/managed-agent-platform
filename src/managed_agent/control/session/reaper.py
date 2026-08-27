"""Giving back the Session pods nothing owns any more.

A pod is a scarce, countable thing here in a way it is not on a hosted API. The limit
that binds is the per-node ENI address count -- measured at seventeen on a `t3.medium`,
not CPU and not memory -- against a Cluster Autoscaler passing `--max-nodes-total=4`,
so the cluster holds somewhere around forty-five Session pods at once. Nothing in this
tree ever gave one back, and the arithmetic of that is worse than it sounds: the
platform did not serve forty-five *concurrent* Sessions, it served forty-five Sessions
and then wedged, because every pod a completed Session left behind held its slot for
good.

This sweep is the backstop and not the primary path. `archive_session` hands a pod back
the moment a caller archives, which is what makes an explicit end fast; this catches
what no call is coming for -- an archive that crashed between its append and its
handback, a pod whose Session was never recorded, and the Session nobody ever comes back
to. It reads the cluster rather than any record of what was placed, for the reason
`placement.py` keeps no binding: a table of placed pods is a second source free to name
one that is gone and to miss one that is not, and the pod this deletes has to be a pod
that exists.

**Every branch below is a decision about deleting a running process, so the safe answer
is always "leave it".** Three guards stand in front of every deletion and each is
sufficient on its own, which is deliberate -- the hazard `pod_runner.ensure` names in
its own words is exactly this one: "a pod that was already here and is merely slow
belongs to a Session that may still be coming up, and deleting it on a timeout would
destroy a live Session to tidy up after a caller."

  1. A pod that is still starting is left alone, whatever its Session's log says. That
     is the hazard's own case, refused on the cluster's own answer.
  2. A pod younger than the grace period is left alone, whatever its Session's log
     says. This is the guard that does not depend on a store: a pod's creation
     timestamp is set by the API server at admission, so no read of any log -- stale,
     lagging, or mid-transaction -- can make a young pod look abandoned.
  3. A Session with an open Turn is left alone, whatever the clock says. This is the
     guard that does not depend on a timer: a Turn can be silent for longer than the
     grace period and still be live, because a first Turn's placement was measured
     holding for minutes while an autoscaled node arrived, and nothing is appended
     during that wait.

Idempotent by construction rather than by a lock, which is what lets two control-plane
replicas run this at once. Every verdict is a function of state both replicas can read,
and handing a pod back treats an absent one as success, so a double delete is not an
error.

**This sweep writes nothing to any Session's log, and that is what makes the idempotence
above the whole story.** It used to append `session.suspended` where it reclaimed, and
that append needed an argument of its own about how far the log could grow -- two
replicas could each write one, and a pod placed again later was a second reclamation
that honestly deserved a second event. Under a per-Turn lease there is nothing to
announce: every pod this sweep finds is one a control plane died holding, and reporting
that into a tenant's Session history would be the platform describing its own crash
(ADR-041). What is left is a deletion, and a deletion nobody records is the same
deletion however many times it runs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, assert_never

from managed_agent.control.session.lifecycle import SessionPodRelease, whole_log
from managed_agent.control.session.placement import PlacedPods, PodPhase
from managed_agent.core.ids import SessionId
from managed_agent.core.ports import Clock, EventLogAppend, EventLogRange
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.session.turns import open_turn
from managed_agent.core.vocabulary import lifecycle, turn

_LOG = logging.getLogger(__name__)

IDLE_GRACE_MS: Final = 15 * 60 * 1000
"""How long a Session may sit untouched before its pod is taken back: fifteen minutes.

One number, fixed here rather than read from the environment, because a deployment that
could choose it would be a deployment where nobody had to decide -- and this decision
has a cost either way that somebody has to have accepted.

**What it trades.** A pure cost dial, and nothing more. Handing a pod back leaves the
Session `IDLE`, which is takeable: the next Turn asks for a pod, the placement path
compiles one for a Session that has already run, and the stored Rollout is seeded so the
thread continues rather than restarting (ADR-031). So what this number buys is an
address freed for another Session, and what it pays is the cold start the next Turn on
this one waits through. Lower frees addresses sooner and pays more cold starts; higher
does the reverse.

That is the whole trade now, and it was not before. This same knob used to decide
whether a Session could ever work again, because a reclaimed Session was parked in a
state with no exit -- which is why it could not honestly be tuned: one side of the
scale was a cache eviction and the other was a Session's life. Both sides are now
measured in seconds of latency (ADR-032).

**Why fifteen minutes and not one, and not sixty.** The floor is the platform's own
slowest operation: a first Turn's HTTP response was measured held for the whole
placement, and the test client's timeout had to go to 660 seconds to cover an
autoscaled node arriving. A grace shorter than that is a grace that can expire inside
one Turn, and while guard 3 above already refuses a Session with an open Turn, a bound
below the platform's own worst wait would rest the whole guarantee on that one check.
Fifteen minutes clears it with room and sits far outside interactive think-time, so an
ordinary multi-Turn conversation never reaches it. The ceiling is throughput: at
roughly forty-five slots, fifteen minutes of grace costs at most that many slots for a
quarter hour, where sixty minutes would cost them for an hour and put the wedge back
under any real burst.

**When this number should change.** That day has arrived, and this number has
deliberately not moved with it -- lowering it here would have mixed a cost decision into
a correctness one. It is now tunable against measurements rather than against a
Session's survival: how long a cold start actually takes, and how often the platform
runs out of addresses. Both are observable, and neither was worth measuring while the
number also decided whether a Session lived.
"""

_ACTIVITY: Final = (
    lifecycle.SESSION_CREATED,
    turn.TURN_SUBMITTED,
    turn.TURN_STARTED,
    turn.TURN_COMPLETED,
    turn.TURN_FAILED,
)
"""The event types that count as a Session having been used recently.

Boundaries only. `turn.message_delta` and the tool and stream families are deliberately
absent, and leaving them out costs nothing: a Session producing deltas has a Turn that
was submitted and not closed, which guard 3 catches by folding that Session's own log
rather than by finding a recent row. Including them would make this scan return every
token of every Turn in the window to answer a question about which Sessions exist.

`session.stopped` is absent for the opposite reason. It is not use, it is the end of it
-- and counting it as activity would hold an archived Session's pod for a whole extra
window, which is the leak this file exists to close.

`session.resumed` was here and is gone with its producer. A Session no longer resumes:
a pod is created for one Turn and destroyed with it (ADR-041), so the type is one only
stored history holds and a scan of a recent window can only ever miss it. Leaving it
would have cost nothing and said something false -- that a row of it could still land
inside the window this reads.
"""


class ReapVerdict(StrEnum):
    """What the sweep decided about one pod. Every member is reached by some branch.

    A closed set rather than a log sentence, because it is what a caller counts: "nine
    kept because a Turn was open" and "nine kept because the sweep could not decide"
    are the difference between a busy cluster and a broken one, and a sentence makes
    them the same line.

    Six of the nine keep the pod. That ratio is the design and not an accident -- the
    sweep deletes only where it has a positive reason, and every guard that fires is
    named, so a pod nobody expected to survive can be explained without a cluster read.
    """

    A_POD_IS_STILL_COMING_UP = "a_pod_is_still_coming_up"
    THE_POD_IS_TOO_YOUNG_TO_JUDGE = "the_pod_is_too_young_to_judge"
    A_HUMAN_HAS_TAKEN_OVER = "a_human_has_taken_over"
    A_TURN_IS_STILL_OPEN = "a_turn_is_still_open"
    THE_SESSION_WAS_USED_RECENTLY = "the_session_was_used_recently"
    THE_SWEEP_COULD_NOT_DECIDE = "the_sweep_could_not_decide"

    NO_SESSION_OWNS_IT = "no_session_owns_it"
    THE_SESSION_HAS_ENDED = "the_session_has_ended"
    THE_SESSION_WENT_IDLE = "the_session_went_idle"

    def gave_the_pod_back(self) -> bool:
        """Whether this verdict handed the pod back to the cluster.

        The verdict answers this rather than each caller re-deriving it from a set of
        members, so a member added later cannot be counted as a keep by one reader and
        a handback by another. Written as a match with an `assert_never` tail and no
        default arm, which is what makes a new member fail `mypy --strict` until it has
        chosen a side -- a member falling through to a default would silently read as a
        keep.
        """
        match self:
            case (
                ReapVerdict.NO_SESSION_OWNS_IT
                | ReapVerdict.THE_SESSION_HAS_ENDED
                | ReapVerdict.THE_SESSION_WENT_IDLE
            ):
                return True
            case (
                ReapVerdict.A_POD_IS_STILL_COMING_UP
                | ReapVerdict.THE_POD_IS_TOO_YOUNG_TO_JUDGE
                | ReapVerdict.A_HUMAN_HAS_TAKEN_OVER
                | ReapVerdict.A_TURN_IS_STILL_OPEN
                | ReapVerdict.THE_SESSION_WAS_USED_RECENTLY
                | ReapVerdict.THE_SWEEP_COULD_NOT_DECIDE
            ):
                return False
            case _ as unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class ReapOutcome:
    """One pod, and what the sweep did about it."""

    session_id: SessionId
    verdict: ReapVerdict


def verdict_for(state: SessionState) -> ReapVerdict | None:
    """The verdict a Session's state settles on its own, or None to keep looking.

    Split out of the sweep and made pure, for two reasons that point the same way. It is
    the one place a `SessionState` member is turned into a decision about deleting a
    pod, and every member has to have an answer here -- so it is written as a match with
    an `assert_never` tail and no default arm, which makes a fifth state fail
    `mypy --strict` rather than fall through to whatever the last branch happened to be.
    And it is reachable from a test without a log that folds to each state, which
    matters because one member cannot be produced by a fold at all: nothing publishes a
    takeover event and `projection.py` has no row for one, so `TAKEN_OVER` is a decision
    that would otherwise be graded by nothing until the day it starts arriving.

    `TAKEN_OVER` keeps its pod. A human has taken control of that Session, so its pod is
    the thing they are using and idleness is expected -- the tenant can still archive
    it, which is a deliberate call, but a sweep must not decide for them.

    `IDLE` and `RUNNING` both return None rather than a verdict, because neither settles
    it and for opposite reasons. A `RUNNING` Session has a Turn executing, which the
    open-turn guard below reads directly from the log rather than trusting a state to
    stand in for it. An `IDLE` one is alive and merely at rest, and whether *this* rest
    has lasted long enough to be worth a pod is a question about a clock -- which is the
    caller's next one. Only a stop is a fact about the Session that a sweep may act on
    alone: it is the single state a Session cannot move out of.

    This function no longer decides that a Session has ended merely because its pod was
    taken back, which is what made a fifteen-minute cost dial permanent (ADR-032).
    """
    match state:
        case SessionState.STOPPED:
            return ReapVerdict.THE_SESSION_HAS_ENDED
        case SessionState.TAKEN_OVER:
            return ReapVerdict.A_HUMAN_HAS_TAKEN_OVER
        case SessionState.IDLE | SessionState.RUNNING:
            return None
        case _ as unreachable:
            assert_never(unreachable)


class RecentActivity(Protocol):
    """The one cross-Session question the sweep asks of the Event Log.

    Narrower than the store that answers it, and narrower than the `LifecycleScan` the
    webhook sweep declares for itself over the same method -- neither imports the
    other, because what they share is one adapter's capability rather than a rule
    either could change. Declared here so a double for this sweep implements one method
    instead of a whole log.
    """

    async def lifecycle_events_between(
        self, types: Sequence[str], from_ms: int, to_ms: int
    ) -> Sequence[object]:
        """Events of these types appended after `from_ms` and at or before `to_ms`.

        The rows come back carrying a `session_id`, which is the only field read here:
        the sweep wants the set of Sessions that appear, not the events themselves.
        Typed as `object` for that reason -- naming a row type would make every double
        reproduce fields this never touches -- and the field is read off the row by
        `_sessions_in`, which refuses a row without one rather than guessing.
        """
        ...


class SessionPodReaper:
    """Reclaims Session pods nothing owns, once per call.

    Holds no state between sweeps. Everything a verdict turns on is read fresh from the
    cluster and the log, so a sweep interrupted halfway leaves nothing to reconcile and
    the next one starts from what is actually true. That is also what makes it safe to
    run from a second replica, or twice in a row, or after a restart.

    The six collaborators arrive as constructor arguments rather than being reached
    through a `Platform`: a collaborator that read fields of the object it is a field
    of would make construction order load-bearing, which is why `FirstTurnPlacement` is
    written the same way.
    """

    def __init__(
        self,
        *,
        pods: PlacedPods,
        release: SessionPodRelease,
        log: EventLogAppend,
        events: EventLogRange,
        activity: RecentActivity,
        clock: Clock,
    ) -> None:
        self._pods = pods
        self._release = release
        self._log = log
        self._events = events
        self._activity = activity
        self._clock = clock

    async def sweep(self) -> Sequence[ReapOutcome]:
        """Decide about every Session pod the cluster is holding, and act on each.

        Returns one outcome per pod, including the ones left alone, because a sweep
        that reported only its deletions could not be told apart from one that found
        nothing and one that refused everything -- and those are three different
        operational situations.

        The recent-activity window is read once, before the loop, rather than per pod.
        One query answers it for every Session, and reading it per pod would also make
        the window slide across the sweep, so that two pods judged seconds apart were
        judged against different windows.

        A pod whose own verdict raises is recorded as undecided and the sweep goes on. A
        single unreadable Session must not stop reclamation for the rest of the cluster:
        that failure mode is the wedge this file exists to prevent, arriving by a
        different route.
        """
        now = self._clock.now_epoch_ms()
        placed = await self._pods.placed_pods()
        used_recently = await self._used_since(now - IDLE_GRACE_MS, now)
        outcomes: list[ReapOutcome] = []
        for pod in placed:
            try:
                verdict = await self._decide(
                    pod.session_id, pod.phase, pod.created_at_ms, now, used_recently
                )
            except Exception:
                # Logged with the traceback rather than swallowed, and recorded as a
                # verdict rather than dropped from the result: a pod this sweep could
                # not judge is a pod still holding its slot, and a caller counting
                # outcomes has to be able to see that.
                _LOG.exception(
                    "the sweep could not decide about session %s", pod.session_id
                )
                verdict = ReapVerdict.THE_SWEEP_COULD_NOT_DECIDE
            outcomes.append(ReapOutcome(session_id=pod.session_id, verdict=verdict))
        return outcomes

    async def _decide(
        self,
        session_id: SessionId,
        phase: PodPhase,
        pod_created_at_ms: int,
        now_ms: int,
        used_recently: frozenset[SessionId],
    ) -> ReapVerdict:
        """The whole decision table for one pod, in the order the guards must run.

        Ordered cheapest-and-safest first, and the order is load-bearing rather than an
        optimisation. Both cluster-side guards are asked before any log is read, so a
        pod that is starting or newly created is kept even if reading its Session's log
        would have failed -- which is what keeps an unreachable store from turning into
        a deletion.

        `GONE` and `ABSENT` fall through to the log-driven branches rather than being
        handed back on sight. A pod the cluster reports gone is one Kubernetes is
        already tearing down, so there is nothing to reclaim from it and a handback on
        it is success -- but its Session may still be `RUNNING` with a Turn open, and
        the honest verdict for that is the one the fold gives.
        """
        if phase is PodPhase.STARTING:
            return ReapVerdict.A_POD_IS_STILL_COMING_UP
        if now_ms - pod_created_at_ms < IDLE_GRACE_MS:
            return ReapVerdict.THE_POD_IS_TOO_YOUNG_TO_JUDGE

        events = await whole_log(self._events, session_id)
        try:
            state, _ = project(events)
        except ValueError:
            # No `session.created` anywhere in the log, so no Session owns this pod.
            # `project` raising is the whole signal: a Session's creation event is
            # appended before its registry row and long before any pod, so a pod
            # standing here with an empty log is residue rather than a Session about
            # to appear -- and guard 2 has already established it is not new.
            await self._release.release(session_id)
            return ReapVerdict.NO_SESSION_OWNS_IT
        if (settled := verdict_for(state)) is not None:
            if settled.gave_the_pod_back():
                await self._release.release(session_id)
            return settled
        if open_turn(events) is not None:
            return ReapVerdict.A_TURN_IS_STILL_OPEN
        if session_id in used_recently:
            return ReapVerdict.THE_SESSION_WAS_USED_RECENTLY
        # Read once more, immediately in front of the deletion, and keep the pod if a
        # Turn opened across the decision above. Every guard so far ran against a fold
        # taken before the recency check, and a submission landing after that fold is
        # invisible to it -- so without this read, a sweep that decided a Session was at
        # rest could delete the pod of a Turn a tenant is waiting on.
        #
        # **A narrowing and not a guarantee, which is a change from how this used to
        # hold.** The reclamation appended an event, and the append gave a total order:
        # a submission below its sequence was seen by the settled re-read of `1..seq`,
        # so the store decided that half of the race outright. Nothing is appended now,
        # so there is no sequence to compare against and this only shrinks the window to
        # the gap between the read and the delete. That is what a silent reclamation
        # costs, and it is named here rather than left for a reader to discover.
        if open_turn(await whole_log(self._events, session_id)) is not None:
            return ReapVerdict.A_TURN_IS_STILL_OPEN
        # Handed back and not announced. The reclamation used to append
        # `session.suspended`, which was worth saying while a pod outlived the Turn that
        # made it; a pod leased for one Turn makes every pod this branch reaches the
        # residue of a control plane that died holding one, and an event for that would
        # be the platform reporting its own crash into a tenant's history once per
        # replica that noticed (ADR-041).
        await self._release.release(session_id)
        return ReapVerdict.THE_SESSION_WENT_IDLE

    async def _used_since(self, from_ms: int, to_ms: int) -> frozenset[SessionId]:
        """Every Session that saw a boundary event inside the window.

        A set of identifiers and not the rows, because that is the whole question: was
        this Session touched. Keeping the rows would invite a later reader to draw a
        second conclusion from them -- how recently, in what order -- and the window's
        own edges are the only resolution this scan has.
        """
        rows = await self._activity.lifecycle_events_between(_ACTIVITY, from_ms, to_ms)
        return frozenset(_sessions_in(rows))


def _sessions_in(rows: Sequence[object]) -> list[SessionId]:
    """The `session_id` of each row, refusing a row that does not carry one.

    Read by attribute rather than by unpacking a declared row type, because the scan's
    port names none -- and refused rather than skipped, because a row shape this cannot
    read means the sweep's idea of "recently used" is silently empty. An empty answer
    there is not a safe default: it reads as "no Session has been touched", which is
    the input that makes every idle branch fire at once.
    """
    out: list[SessionId] = []
    for row in rows:
        session_id = getattr(row, "session_id", None)
        if session_id is None:
            raise TypeError(
                f"a recent-activity row carries no session_id: {type(row).__name__}"
            )
        out.append(SessionId(session_id))
    return out
