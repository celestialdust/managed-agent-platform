"""Lifecycle event types. The projection's transition table reads these names.

**Two of the four are declared here and appended by nothing.** `session.suspended` said
a pod had been reclaimed from a resting Session and `session.resumed` said one had been
placed for a Session that had none. A pod is now leased for a single Turn, so the first
moment never arrives and the second is every Turn there is (ADR-041).

The producers are gone and the declarations are not, because declaring is what makes a
stored row readable at all: every surface a tenant reads history through emits a row
only if its type is published and drops it in silence otherwise. Undeclaring would
therefore not retire a type -- it would delete every row of it a tenant already holds,
out of a log that is supposed to be append-only, with nothing anywhere reporting the
loss.

What ended with the producers is eligibility. A callback registration is a thing a
tenant maintains -- an endpoint, a certificate, a secret in the vault -- and one pointed
at a type nothing will ever append is maintenance spent on a delivery that is never
coming. `webhook=False` moves that refusal to the registration, where it is answered,
out of a silence the tenant would otherwise have to notice on their own.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from managed_agent.core.vocabulary import declare

FAMILY = "lifecycle"

SESSION_CREATED = declare("session.created", FAMILY, webhook=True)
SESSION_SUSPENDED = declare("session.suspended", FAMILY, webhook=False)
SESSION_RESUMED = declare("session.resumed", FAMILY, webhook=False)
SESSION_STOPPED = declare("session.stopped", FAMILY, webhook=True)


class StopReason(StrEnum):
    """Why a Session stopped taking work. Machine-readable, and closed.

    Sits beside `session.stopped` the way `TurnFailureCause` sits beside `turn.failed`:
    the event says *what* happened and this says *why*, so a consumer that reads only
    the type still gets a correct answer and one that needs the cause does not have to
    parse a sentence. An enum holding one member rather than a bare constant, because
    the value is on the wire and a consumer branching on it needs a closed set -- so a
    second ending arriving later costs a member rather than a change of shape.

    Every member is emitted by something in this tree, which is the same rule
    `projection.py` states for its transition table: a member nothing appends is a
    branch a consumer would write and never reach. `idle_timeout` was one until a pod
    became a thing leased for one Turn, and it is gone by that rule -- no pod is
    reclaimed from a resting Session any more, so nothing measures an idle clock and
    nothing has that answer to give (ADR-041).

    Two upstream reasons are absent under the same rule. `budget_reached` is the obvious
    next member and nothing here measures a Session's spend against `budget_minor_units`
    -- the field is stored, read by no comparison, and `ErrorCode.BUDGET_EXHAUSTED` is
    emitted by no route -- so a member for it would describe a control this platform
    does not have. `end_turn` is absent because finishing a Turn does not end a Session
    here: the Session goes `IDLE` and waits, which is the same shape as a session that
    "finishes its work goes idle, not terminated".

    Not the `stop_reason` the Model Gateway classifies. That one is a field of one
    upstream model response (`gateway/model/anthropic_table.py`) and answers why one
    completion stopped generating; this answers why a Session stopped taking work. The
    two share a name upstream and never share a value.
    """

    ARCHIVED = "archived"
    """A caller archived the Session: no further event is accepted and history stays."""


class SessionStopped(BaseModel):
    """The payload of a `session.stopped` event.

    The reason is a one-member `Literal` rather than the whole enum, so a pairing the
    event cannot carry is a type error at the call rather than a value a reader has to
    know is impossible, and mypy refuses it without a runtime check existing at all.
    That reads as a restatement of the enum while `ARCHIVED` is the only member and is
    not one: widening it is a deliberate edit that has to name the transition appending
    the new reason.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stop_reason: Literal[StopReason.ARCHIVED]
