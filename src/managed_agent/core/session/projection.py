"""Computing a Session's current view by reading its Event Log forward.

The fold applies each event as it meets it and lets a later one override an
earlier one, which is what allows superseded work to stay in the log without
being mistaken for current work: the superseded events are read, then what
follows them tells the fold what to make of them.

**This fold is no longer total over the event types it does not know, and the loss is
deliberate.** It used to be: a type with no row advanced the sequence and changed
nothing, so a family added later could not break a state read and this function never
had to be edited. That property cannot survive a state meaning "a Turn is executing
right now", because the only events that know when one starts and stops are the turn
family's -- a state read folded from lifecycle events alone would have to guess. What
is bought is a `RUNNING` that is true when it is read; what is paid is that a family
which opens or closes a unit of work now has to add its rows here, and one that forgets
leaves a Session reading as idle while it works. Every family that says nothing about
work starting or stopping still passes through untouched, so the cost falls only on the
narrow case (ADR-032).

A stop is the end of the fold rather than one more move in it. Both writers here append
before they re-read -- the admission path to settle a race by sequence, the sweep to
avoid deleting a pod out from under a Turn -- so an event can legitimately land behind a
stop, and under a plain last-row-wins rule a `turn.submitted` that lost such a race
would read an archived Session back to `RUNNING`. Absorbing at `STOPPED` is what keeps
terminal meaning terminal without any writer having to coordinate.

The table is narrower than SessionState: a state exists as soon as something can be in
it, but a transition into it exists only once some family module publishes the event
that causes it. TAKEN_OVER has no row here because the `takeover` family is not built
yet, and a row reading an unpublished type would be a transition that can never fire --
nothing may emit a type the vocabulary has not published.
"""

from collections.abc import Iterable

from managed_agent.core.ids import Seq
from managed_agent.core.ports import EventRecord
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import turn

_TRANSITIONS: dict[str, SessionState | None] = {
    "session.created": SessionState.IDLE,
    "session.suspended": SessionState.IDLE,
    "session.resumed": None,
    "session.stopped": SessionState.STOPPED,
    turn.TURN_SUBMITTED: SessionState.RUNNING,
    turn.TURN_COMPLETED: SessionState.IDLE,
    turn.TURN_FAILED: SessionState.IDLE,
}
"""What each event type says the Session is, once it has been read.

A `None` is a type this table has read and deliberately moves nothing for, which is a
different answer from a type with no row at all: the row is what says somebody decided,
and it is what keeps a lifecycle type from being added with no thought about the state.

The two pod events are a pair and still land differently, because the two moments are
not alike. A reclaim is only ever appended with no Turn open, so `IDLE` is a true
statement about the Session at the moment it lands. A placement is only ever appended
*inside* an open Turn -- a pod is placed lazily, by the Turn that finds none -- so any
state at all would be this event overruling the Turn events about work it knows nothing
about, and `IDLE` in particular would read a cold-started Session as at rest while its
agent ran. Neither event is a claim about whether the Session will work again.

`turn.started` and `turn.message_delta` are absent rather than mapped to `RUNNING`.
They arrive while a Turn is already open, so a row for either would restate what the
submission already said -- and the absence is what keeps the rule readable as "a
submission opens the work, a closure ends it".
"""


def project(events: Iterable[EventRecord]) -> tuple[SessionState, Seq]:
    """Fold the log forward. Returns the state and the last sequence read.

    Raises ValueError on an empty log: a Session with no created event does not
    exist, and returning a default here would invent one.

    Every event advances the sequence, including the ones read after a stop. The two
    returned values answer different questions -- what the Session is, and how far this
    read got -- and a caller resuming a stream from the second would miss events if the
    stop froze it too.

    `Seq(last)` on the way out is a cast and not a check — the annotation is erased at
    runtime, so it asserts nothing here. What makes the value sound is that `last` is
    only ever assigned from an event that was read, every stored event carries a
    sequence of at least 1 by the store's own check constraint, and a log that yielded
    no event raises above instead of returning the initial 0.
    """
    state: SessionState | None = None
    last: int = 0
    for event in events:
        last = event.seq
        if state is SessionState.STOPPED:
            continue
        if (moved := _TRANSITIONS.get(event.type)) is not None:
            state = moved
    if state is None:
        raise ValueError("no session.created event in the log")
    return state, Seq(last)
