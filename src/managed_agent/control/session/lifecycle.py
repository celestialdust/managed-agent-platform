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
    if not settled_state.accepts_a_turn():
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


def open_turn(events: Iterable[EventRecord]) -> TurnId | None:
    """The Turn this Session has submitted and not yet closed, or None.

    A Turn is open from its `turn.submitted` until whichever of `turn.completed` or
    `turn.failed` names it. Both terminal types are read, because reading only the
    completion would leave every failed Turn open for ever -- and a Session that had
    ever failed a Turn could then never be archived or reclaimed.

    Matched by the `turn_id` in the payload rather than by "the last turn event wins",
    because the events of two Turns can interleave in the log -- a `turn.message_delta`
    of one arriving between another's submission and completion -- and a positional rule
    would then close the wrong Turn. A terminal event naming a Turn that was never
    submitted is ignored rather than treated as an error: this answers what is still
    running, and a close with no open is not something still running.

    Returns the earliest still-open Turn, so a caller telling somebody to interrupt one
    names the Turn that has been waiting longest. In practice at most one is ever open,
    because `admit_turn` is the only thing that appends a submission and it refuses a
    Session whose state does not accept a Turn -- but nothing here depends on that, and
    a rule reading "the only one" would be a claim about a component this does not own.
    """
    open_turns: dict[str, TurnId] = {}
    for event in events:
        identifier = event.payload.get("turn_id")
        if not isinstance(identifier, str):
            continue
        if event.type == turn.TURN_SUBMITTED:
            open_turns[identifier] = TurnId(UUID(identifier))
        elif event.type in (turn.TURN_COMPLETED, turn.TURN_FAILED):
            open_turns.pop(identifier, None)
    return next(iter(open_turns.values()), None)


async def archive_session(
    session_id: SessionId,
    log: EventLogAppend,
    ranged: EventLogRange,
    release: SessionPodRelease,
) -> ArchiveOutcome:
    """Stop this Session accepting events, and give its pod back.

    Refuses while a Turn is open, which is the one refusal this operation has -- and it
    is about a Turn rather than about the Session's state. That distinction is the whole
    reading of this platform's state machine against the API it mirrors: `RUNNING` here
    means "would accept a Turn", which is the *idle* end of that API, and there is no
    member standing for "an agent is executing right now" -- a Turn in flight is visible
    only as a submission nothing has closed. So the refusal has to be folded out of the
    Turn events, and refusing on `SessionState.RUNNING` would refuse every Session that
    was archivable.

    Archives from any state but `STOPPED`, which needs no append and is reported as
    already archived so a retried call is not a second stop in the log. A `SUSPENDED`
    Session archives into a stop, which is what finalising a parked Session means. A
    `TAKEN_OVER` one archives too: archiving is a deliberate call by the tenant that
    owns the Session, and refusing it would leave the one state a human is watching as
    the state whose pod can never be handed back.

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
    stopped_at, released = await _end_and_release(
        session_id,
        lifecycle.SESSION_STOPPED,
        lifecycle.SessionStopped(stop_reason=lifecycle.StopReason.ARCHIVED),
        log,
        ranged,
        release,
    )
    return SessionArchived(seq=stopped_at, pod_released=released)


async def suspend_session(
    session_id: SessionId,
    log: EventLogAppend,
    ranged: EventLogRange,
    release: SessionPodRelease,
) -> Seq:
    """Park this Session and give its pod back, because nothing is using it.

    **A Session this reaches cannot currently be resumed, and a caller has to have
    decided that is acceptable.** `SUSPENDED` accepts no Turn, so what this appends is,
    today, the end of that Session's working life. It is not the end of its history:
    every event stays readable, and its Rollout is preserved by whatever
    `ShipOutAtTurnCompletion` was wired with, so the state a resume needs is off the pod
    rather than dying with it (ADR-004).

    What is missing is now this transition and nothing below it. A pod that continues a
    Session's thread from its stored Rollout exists -- the placement path compiles one
    and the shim resumes from the seeded record (ADR-031) -- and a Session whose pod
    simply vanished is re-placed by the next Turn that finds none. Only a Session parked
    HERE stays parked, because no Turn is admitted to go and ask for that pod.

    So this is deliberately not called at `turn.completed`. A Session that finishes its
    work stays at `RUNNING` and waits, and suspending there would make every Session
    single-Turn -- reclaiming the slot by destroying the thing occupying it.

    Whether nothing is using it is the caller's judgement and not checked here: this
    appends and releases, and `session_reaper.py` owns the rules about what counts as
    idle. Splitting it that way keeps one place that knows the ordering below and one
    that knows the policy, rather than a policy that can be right while the ordering is
    wrong.

    Returns the sequence of the suspension. The pod is not released when a Turn opened
    across the append, for the reason `archive_session` gives: the Session is parked
    either way, and a sweep reclaims the pod once that Turn is closed.
    """
    seq, _ = await _end_and_release(
        session_id,
        lifecycle.SESSION_SUSPENDED,
        lifecycle.SessionSuspended(stop_reason=lifecycle.StopReason.IDLE_TIMEOUT),
        log,
        ranged,
        release,
    )
    return seq


async def _end_and_release(
    session_id: SessionId,
    type_: str,
    payload: lifecycle.SessionSuspended | lifecycle.SessionStopped,
    log: EventLogAppend,
    ranged: EventLogRange,
    release: SessionPodRelease,
) -> tuple[Seq, bool]:
    """Append the event that ends a Session's working life, then give its pod back.

    The order is the invariant both callers need, and the reason this is one function
    rather than six lines written twice. Append first: a failure after it leaves a
    Session correctly recorded as ended with a pod still up, which a sweep reclaims from
    the state it reads. Releasing first would leave a Session recorded as live with
    nothing behind it.

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
    seq = await log.append(session_id, type_, payload.model_dump(mode="json"))
    settled = await ranged.read(session_id, FIRST_SEQ, seq, limit=seq)
    if open_turn(settled) is not None:
        return seq, False
    await release.release(session_id)
    return seq, True
