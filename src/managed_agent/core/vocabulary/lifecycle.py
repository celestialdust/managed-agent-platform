"""Lifecycle event types. The projection's transition table reads these names."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from managed_agent.core.vocabulary import declare

FAMILY = "lifecycle"

SESSION_CREATED = declare("session.created", FAMILY)
SESSION_SUSPENDED = declare("session.suspended", FAMILY)
SESSION_RESUMED = declare("session.resumed", FAMILY)
SESSION_STOPPED = declare("session.stopped", FAMILY)


class StopReason(StrEnum):
    """Why a Session stopped doing work. Machine-readable, and closed.

    Sits beside the two events that end a Session the way `TurnFailureCause` sits beside
    `turn.failed`: the event says *what* happened to the Session and this says *why*, so
    a consumer that reads only the type still gets a correct answer and one that needs
    the cause does not have to parse a sentence.

    Every member is emitted by something in this tree, which is the same rule
    `projection.py` states for its transition table: a member nothing appends is a
    branch a consumer would write and never reach. Two upstream reasons are therefore
    deliberately absent. `budget_reached` is the obvious third member and nothing here
    measures a Session's spend against `budget_minor_units` -- the field is stored, read
    by no comparison, and `ErrorCode.BUDGET_EXHAUSTED` is emitted by no route -- so a
    member for it would describe a control this platform does not have. `end_turn` is
    absent because finishing a Turn does not end a Session here: the Session stays at
    `RUNNING` and waits, which is the same shape as a session that "finishes its work
    goes idle, not terminated".

    Not the `stop_reason` the Model Gateway classifies. That one is a field of one
    upstream model response (`gateway/model/anthropic_table.py`) and answers why one
    completion stopped generating; this answers why a Session stopped taking work. The
    two share a name upstream and never share a value.
    """

    IDLE_TIMEOUT = "idle_timeout"
    """The pod was reclaimed because nothing had happened on this Session for a while.

    No upstream reason means this, and the absence is informative rather than an
    oversight: upstream keeps a session's sandbox for thirty days from its creation and
    reclaims it on a clock nobody watches, so it never has to tell a caller that idling
    cost them anything. This platform reclaims in minutes because a pod is a node's
    ENI-limited address rather than a row in a bucket, so the reason has to be nameable.
    """

    ARCHIVED = "archived"
    """A caller archived the Session: no further event is accepted and history stays."""


class SessionSuspended(BaseModel):
    """The payload of a `session.suspended` event.

    The reason is a one-member `Literal` rather than the whole enum, so the illegal
    pairing is a type error at the call rather than a value a reader has to know is
    impossible: `ARCHIVED` stops a Session and cannot suspend one, and mypy refuses it
    here without a runtime check existing at all. Widening this is a deliberate edit
    that has to name the transition appending the new reason.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stop_reason: Literal[StopReason.IDLE_TIMEOUT]


class SessionStopped(BaseModel):
    """The payload of a `session.stopped` event. Narrowed the same way as above."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stop_reason: Literal[StopReason.ARCHIVED]
