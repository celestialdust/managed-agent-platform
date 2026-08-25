"""Computing a Session's current view by reading its Event Log forward.

The fold applies each event as it meets it and lets a later one override an
earlier one, which is what allows superseded work to stay in the log without
being mistaken for current work: the superseded events are read, then what
follows them tells the fold what to make of them.

This function is deliberately total over the event types it does not know. A
type it has no case for advances the sequence and changes nothing — so adding an
event type in a later slice cannot break a state read, and the projection never
has to be edited to stay correct.

The table below is narrower than SessionState: a state exists as soon as
something can be in it, but a transition into it exists only once some family
module publishes the event that causes it. TAKEN_OVER has no row here because
the `takeover` family is not built yet, and a row reading an unpublished type
would be a transition that can never fire — nothing may emit a type the
vocabulary has not published.
"""

from collections.abc import Iterable

from managed_agent.core.ids import Seq
from managed_agent.core.ports import EventRecord
from managed_agent.core.session.session import SessionState

_TRANSITIONS: dict[str, SessionState] = {
    "session.created": SessionState.RUNNING,
    "session.suspended": SessionState.SUSPENDED,
    "session.resumed": SessionState.RUNNING,
    "session.stopped": SessionState.STOPPED,
}


def project(events: Iterable[EventRecord]) -> tuple[SessionState, Seq]:
    """Fold the log forward. Returns the state and the last sequence read.

    Raises ValueError on an empty log: a Session with no created event does not
    exist, and returning a default here would invent one.

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
        if (moved := _TRANSITIONS.get(event.type)) is not None:
            state = moved
    if state is None:
        raise ValueError("no session.created event in the log")
    return state, Seq(last)
