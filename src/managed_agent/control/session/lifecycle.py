"""Whether a Session may take a Turn right now, and what a repeat submission gets back.

One question, one answer: the state is folded out of the Event Log and the state itself
says whether a Turn may start, so no component here keeps a copy that could disagree
with the log (ADR-008).

Deduplication is settled by sequence instead of by a lock. The submission is appended
first; then the log up to and including that append is re-read, and the lowest-sequence
`turn.submitted` bearing the key is the Turn. Every submitter reads a prefix that ends
at its own sequence, so a later racer always sees an earlier one and an earlier racer
never sees a later one -- the two reach the same verdict without coordinating, and only
one Turn is ever dispatched. The bounded end of that re-read is the whole trick;
reading to the head instead would let two racers each see the other and both stand
down.

That same re-read catches a Session that stopped between the fold and the append: the
settled prefix contains the stop, so the admission refuses rather than dispatching onto
a Session that is no longer running.

Both reads name how much they need. `EventLogRange.read` returns at most `limit`
records and a short result means "page for the rest", so a read that takes the default
answers a long Session with its first page: the fold would report a stale state and the
key lookup would miss a submission past the cap, selling a second Turn to a retry. This
repository has shipped that defect three times, so neither read here takes the default
-- the whole-log read pages until a page comes back empty, and the settled read names a
limit that provably covers its span.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, TurnId, new_turn_id
from managed_agent.core.ports import EventLogAppend, EventLogRange, EventRecord
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.session.turns import open_turn
from managed_agent.core.vocabulary import lifecycle, turn

_UNBOUNDED_END: Seq = 2**62
"""An end above any sequence a Session will reach, for a read that wants the head.

The port's range is inclusive of both ends and the head has no number until it is
written, so asking for "everything from here" means naming a number nothing can pass.
Paired with paging below, never used alone: a wide end does not lift the port's row
cap, and the two are only correct together.

A fixed narrow window would be wrong rather than merely slower. Retention raises a
Session's lowest surviving sequence, so a first window of `1..500` over a swept log
falls entirely below the survivors and reads as an empty log.
"""


@dataclass(frozen=True, slots=True)
class TurnAdmitted:
    """A new Turn. It is recorded and has not been dispatched yet."""

    turn_id: TurnId
    seq: Seq


@dataclass(frozen=True, slots=True)
class TurnReplayed:
    """This key already submitted a Turn. This is that Turn, not a second one."""

    turn_id: TurnId
    seq: Seq


@dataclass(frozen=True, slots=True)
class TurnRefused:
    """The Session cannot take a Turn, and this is the state that says so."""

    state: SessionState


TurnAdmission = TurnAdmitted | TurnReplayed | TurnRefused


@dataclass(frozen=True, slots=True)
class _Submission:
    seq: Seq
    turn_id: TurnId


class _RangeRead(Protocol):
    """The one method of `EventLogRange` the paging read below calls.

    Narrower than the port so the helper says what it uses; a caller passes the port.
    """

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Iterable[EventRecord]: ...


async def whole_log(ranged: _RangeRead, session_id: SessionId) -> list[EventRecord]:
    """Every event of one Session, across as many reads as the port's cap takes.

    Reads from just past the highest sequence seen so far until a page comes back
    empty. That terminates for any page size, one included, because each non-empty page
    raises the cursor above its own highest sequence and a log is finite. Taking the
    highest rather than the last element keeps termination independent of whether the
    port happens to return a page in order.
    """
    events: list[EventRecord] = []
    cursor = 0
    while True:
        page = list(await ranged.read(session_id, Seq(cursor + 1), _UNBOUNDED_END))
        if not page:
            return events
        events.extend(page)
        cursor = max(event.seq for event in page)


def _first_submission(events: Iterable[EventRecord], key: str) -> _Submission | None:
    """The earliest submission bearing this key, or None.

    Events arrive in sequence order, so the first match is the lowest sequence and no
    sorting is needed.
    """
    for event in events:
        if event.type != turn.TURN_SUBMITTED:
            continue
        if event.payload.get("idempotency_key") == key:
            return _Submission(event.seq, TurnId(UUID(str(event.payload["turn_id"]))))
    return None


async def admit_turn(
    session_id: SessionId,
    idempotency_key: str,
    prompt: str,
    log: EventLogAppend,
    ranged: EventLogRange,
) -> TurnAdmission:
    """Decide what one submission gets, and record it if it is a new Turn.

    The idempotency check runs before the state check on purpose. A retry that crosses
    a stop must still get back the Turn it originally started; refusing it instead would
    turn a submission that succeeded into an error the client cannot tell apart from one
    that never happened.

    The two state checks ask different questions, and they have to. The first asks
    whether a Turn may start, which is `IDLE` and refuses a Session already working. The
    second cannot ask that, because by then this call's own `turn.submitted` is in the
    prefix it re-reads -- so the Session is `RUNNING` *because of this submission*, and
    re-asking would have every admission refuse itself. What the second read is for is
    narrower: something else may have ended the Session in the window between the fold
    and the append, and a stop is the only ending there is. So it asks about a stop.
    """
    before = await whole_log(ranged, session_id)
    if (earlier := _first_submission(before, idempotency_key)) is not None:
        return TurnReplayed(turn_id=earlier.turn_id, seq=earlier.seq)
    state, _ = project(before)
    if not state.accepts_a_turn():
        return TurnRefused(state=state)

    submission = turn.TurnSubmitted(
        turn_id=new_turn_id(), idempotency_key=idempotency_key, prompt=prompt
    )
    seq = await log.append(
        session_id, turn.TURN_SUBMITTED, submission.model_dump(mode="json")
    )
    # `limit=seq` provably covers the span: the range is `1..seq` and `(session_id,
    # seq)` is the log's primary key, so at most `seq` events lie inside it.
    settled = await ranged.read(session_id, FIRST_SEQ, seq, limit=seq)
    winner = _first_submission(settled, idempotency_key)
    assert winner is not None, "the append above is inside the prefix just read"
    if winner.turn_id != submission.turn_id:
        return TurnReplayed(turn_id=winner.turn_id, seq=winner.seq)
    settled_state, _ = project(settled)
    if settled_state is SessionState.STOPPED:
        return TurnRefused(state=settled_state)
    return TurnAdmitted(turn_id=submission.turn_id, seq=seq)


class SessionPodRelease(Protocol):
    """Whatever can give a Session's pod back to the cluster.

    One method, and narrower than `Placement` on purpose: a transition here decides
    *when* a pod stops being needed and must not acquire the ability to place one, or
    every end-of-life path becomes a place a Session can be revived by accident.
    `Placement` satisfies this by having `release`, so the composition root wires the
    object it already builds.
    """

    async def release(self, session_id: SessionId) -> None:
        """Give this Session's pod back. Absent is success, so a repeat is not an
        error."""
        ...


class NoSessionPods:
    """The release wired into a process that places no Session pods: it does nothing.

    A `Platform` is built without one in two dozen places, so the field it fills is
    defaulted rather than required -- and unlike the other defaulted fields on it, the
    safe default here is a no-op rather than a refusal. The reason is a wiring invariant
    and not an opinion: the only thing in this tree that creates a Session pod is
    `FirstTurnPlacement`, which `composition.build` constructs inside the same
    `pod_runner is not None` branch that wires the real release, so a process holding
    this object is one in which no Session pod can exist. Refusing instead would make
    archiving fail in every process that has nothing to give back.

    What that invariant does not cover is a later wiring that places pods and forgets
    this field, and the honest answer is that the type cannot catch it. The backstop is
    `session_reaper.py`, which reads the cluster rather than this object and reclaims a
    pod whose Session has ended however it ended -- so the cost of that mistake is
    reclamation delayed by one sweep instead of a pod held for ever.
    """

    async def release(self, session_id: SessionId) -> None:
        return None


@dataclass(frozen=True, slots=True)
class SessionArchived:
    """The Session now accepts no further event. `pod_released` says whether one went.

    False does not mean the archive was partial: the stop is appended either way and is
    what makes the Session archived. It means a Turn was still open across the append,
    so the pod was left for whatever is sweeping to reclaim once that Turn ends, rather
    than deleted out from under a Turn a tenant is waiting on.
    """

    seq: Seq
    pod_released: bool


@dataclass(frozen=True, slots=True)
class SessionAlreadyArchived:
    """Nothing was appended because this Session had already stopped."""

    seq: Seq


@dataclass(frozen=True, slots=True)
class ArchiveRefused:
    """A Turn is running. The caller interrupts it, then archives."""

    turn_id: TurnId


ArchiveOutcome = SessionArchived | SessionAlreadyArchived | ArchiveRefused


async def archive_session(
    session_id: SessionId,
    log: EventLogAppend,
    ranged: EventLogRange,
    release: SessionPodRelease,
) -> ArchiveOutcome:
    """Stop this Session accepting events, and give its pod back.

    Refuses while a Turn is open, which is the one refusal this operation has. There is
    now a state that says exactly that -- `RUNNING` means an agent is executing right
    now -- and this still folds the Turn events instead of reading it. The two are not
    the same question. `open_turn` names *which* Turn is unfinished, and the refusal has
    to carry that identifier so the caller knows what to interrupt; a state check could
    only say that something was running. Reading the state and then folding anyway to
    get the id would be one question asked twice, with a race between the answers.

    An earlier version of this paragraph argued the opposite -- that no member stood for
    "an agent is executing right now", so the refusal *had* to be folded out of the Turn
    events. That member exists now (ADR-032). The conclusion survives its premise, for
    the reason above.

    Archives from any state but `STOPPED`, which needs no append and is reported as
    already archived so a retried call is not a second stop in the log. An `IDLE`
    Session archives into a stop, which is what deliberately ending a resting Session
    means. A `TAKEN_OVER` one archives too: archiving is a deliberate call by the tenant
    that owns the Session, and refusing it would leave the one state a human is watching
    as the state whose pod can never be handed back.

    The stop is appended before the pod is released, and a crash between them is why
    that order and not the other. This way the log says the Session is archived and a
    pod outlives it, which whatever sweeps the cluster next reclaims because the state
    it reads is `STOPPED`. The other order leaves a Session reading as live with no pod
    behind it -- every later Turn answering 502 with nothing naming the cause.
    """
    before = await whole_log(ranged, session_id)
    state, seq = project(before)
    if state is SessionState.STOPPED:
        # Appends nothing and still hands the pod back, which is the whole value of
        # retrying this call. The archive that crashed between its append and its
        # handback leaves exactly this shape -- a stopped Session holding a pod -- and a
        # retry that returned "already archived" without looking at the pod would leave
        # the slot for a sweep that is fifteen minutes away. Guarded on the open Turn
        # for the same reason every other branch is: the pod of a Turn somebody is
        # waiting on is not deleted, however the Session got to `STOPPED`.
        if open_turn(before) is None:
            await release.release(session_id)
        return SessionAlreadyArchived(seq=seq)
    if (running := open_turn(before)) is not None:
        return ArchiveRefused(turn_id=running)
    stopped_at, released = await _end_and_release(session_id, log, ranged, release)
    return SessionArchived(seq=stopped_at, pod_released=released)


async def _end_and_release(
    session_id: SessionId,
    log: EventLogAppend,
    ranged: EventLogRange,
    release: SessionPodRelease,
) -> tuple[Seq, bool]:
    """Append the stop that ends this Session, then give its pod back.

    One caller, and it stays a function of its own because what it holds is an ordering
    argument rather than a step: the lines below are correct only in this order, and the
    reasons are longer than the code. It took the event and its payload as arguments
    while reclaiming a resting Session's pod was a second kind of ending; that caller is
    gone and the parameters went with it, rather than staying as a widening nothing
    asks for and nothing grades.

    Append first: a failure after it leaves the event correctly recorded with a pod
    still up, which a sweep reclaims on its next pass. Releasing first would leave a
    Session whose log says nothing happened and whose pod is gone, which reads as live
    with nothing behind it -- every later Turn answering 502 with nothing naming the
    cause.

    Between the append and the release, the prefix ending at the append is re-read and
    the pod is kept if a Turn is open in it -- the same append-then-settle move
    `admit_turn` makes, and for the mirror-image race. A Turn admitted after the caller
    folded and before this appended is invisible to that fold, and deleting its pod
    would kill a Turn a tenant is waiting on. The settled read cannot miss it: the
    submission's own sequence is below the append's, so it lies inside `1..seq`.

    `limit=seq` provably covers that span, because the range is `1..seq` and
    `(session_id, seq)` is the log's primary key -- so at most `seq` events lie in it.

    **A redundant ending event is reachable and is not prevented here.** Two callers
    that both fold before either appends both decide to end the Session, and the second
    event lands behind the first. That is the same trade `admit_turn` takes one screen
    up, where a `turn.submitted` that loses its race stays in the log while its caller
    is told who won, and the reason is the same: this port offers no conditional append,
    so the only thing that could make it impossible is a store-level constraint that
    does not exist. It is bounded rather than unbounded, which is what makes it
    acceptable -- the fold reads the last event, so every reader sees one state, and
    once the state reads as ended the caller's own check refuses before appending
    again. One extra event per retry for the life of the Session would not be
    acceptable, and that is the property the tests assert.
    """
    seq = await log.append(
        session_id,
        lifecycle.SESSION_STOPPED,
        lifecycle.SessionStopped(stop_reason=lifecycle.StopReason.ARCHIVED).model_dump(
            mode="json"
        ),
    )
    settled = await ranged.read(session_id, FIRST_SEQ, seq, limit=seq)
    if open_turn(settled) is not None:
        return seq, False
    await release.release(session_id)
    return seq, True


async def close_abandoned_turn(
    session_id: SessionId,
    turn_id: TurnId,
    log: EventLogAppend,
    ranged: EventLogRange,
    cause: turn.TurnFailureCause,
) -> bool:
    """Record that this Turn produced no answer, so its Session can work again.

    Here beside `admit_turn` and `archive_session` rather than in the sweep that calls
    it, because it is the inverse of the refusal those two make: they read `open_turn`
    and stand down, and this is the only thing in the tree that makes `open_turn` stop
    naming a Turn from outside the request path. A transition and its two refusals that
    lived in different modules would be free to disagree about what closing means.

    Returns whether this call is the one that appended. False means the Turn was already
    closed -- by the request path finishing after all, or by another replica's sweep --
    and a caller counting closes must not count it.

    **Folded in front rather than appended blind, and that fold is what bounds the
    redundancy.** It refuses a Turn that is not open, so calling this a second time on a
    closed Turn appends nothing and a sweep may run every tick for the life of the
    process at no cost to the log.

    What it does not do is make two replicas racing impossible. Both fold, both see the
    Turn open, and both append -- and no re-read afterwards could prevent that, because
    by then the event is written and this port offers no conditional append. The reason
    that is acceptable rather than merely tolerated is the fold every reader takes:
    `open_turn` pops a Turn on whichever terminal event names it first, so the second
    ending changes no state anybody reads, and the guard above means the excess is one
    event per racing sweep on a Turn that was ending anyway rather than one per tick.
    That is the same bound `_end_and_release` records for the stop event, in the same
    words, for the same reason.

    **The cause is the caller's to name, and has no default.** Naming a cause at all is
    what makes this event worth more to the tenant than the state change it buys -- a
    Turn that simply stopped is indistinguishable from one the platform never received.
    But the callers here do not all mean the same thing. A Turn whose pod died means
    `RUNTIME_LOST`: the work was lost with the process carrying it, and resubmitting is
    the remedy. A Turn that was never given a pod means `RUNTIME_DID_NOT_START`: nothing
    was carrying it, so nothing was lost, and the tenant's next move differs.

    A default would collapse that distinction silently, which is the exact failure
    `TurnFailureCause`'s own docstring records: `POD_UNREACHABLE` answered for four
    unrelated situations with disagreeing remedies, and one confident wrong name is
    worse than no name. Requiring the argument makes a new caller choose, and makes a
    wrong choice visible in its own call rather than inherited from here.
    """
    before = await whole_log(ranged, session_id)
    if open_turn(before) != turn_id:
        return False
    await log.append(
        session_id,
        turn.TURN_FAILED,
        {
            "turn_id": str(turn_id),
            "cause": cause.value,
        },
    )
    return True
