"""Which Turn a Session has open, folded from its own Turn events.

Here rather than beside the transitions that append these events, because the answer
is a fold over the Event Log and nothing else -- no store, no clock, no pod. Two
callers that live in different layers need it: the route that refuses to attach a file
while a Turn is running, and the sweep that refuses to take a pod back out from under
one. Neither may import the other, and `core` may not import `control`, so the fold
belongs to the layer both can reach.

It is deliberately not part of `project`. The state fold answers *what the Session is*
from the last event that moved it; this answers *which Turn is unfinished*, which needs
every submission matched against its own terminal event by identifier. Folding both in
one pass would make one function that returns two unrelated answers, and the callers
here want one or the other, never both (ADR-032).
"""

from collections.abc import Iterable
from uuid import UUID

from managed_agent.core.ids import TurnId
from managed_agent.core.ports import EventRecord
from managed_agent.core.vocabulary import turn


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
    because the admission path is the only thing that appends a submission and it
    refuses a Session whose state does not accept a Turn -- but nothing here depends on
    that, and a rule reading "the only one" would be a claim about a component this does
    not own.
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
