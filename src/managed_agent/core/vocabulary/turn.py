"""The turn family of the published event vocabulary, and what a submission records.

A failed Turn gets its own published type rather than a status field on the completed
one. The Agent Runtime does the opposite -- it publishes no failure notification at
all, and a failed turn arrives as `turn/completed` whose `turn` object carries a
`failed` status plus the runtime's own error (`TurnCompletedNotification` and `Turn` in
the reference protocol under `.reference/codex`). Splitting it here means a consumer
that handles `turn.completed` cannot accidentally treat a failure as a success by
forgetting to read one field, which is the one mistake worth making impossible.

The submission payload carries the idempotency key because detecting a repeat
submission is a question about what already happened, and what already happened is the
Event Log (ADR-008). The key is scoped to one Session: the same string under two
Sessions names two unrelated submissions and is never looked up across them.

TurnFailureCause covers the three ways a Turn stops short of an answer, and
deliberately carries no runtime error name or error text -- a tenant sees a platform
cause and never the runtime's own (ADR-013). It is not the marker cause set: a marker
answers "why was this work abandoned", and this answers "why did this Turn not produce
an answer".
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.ids import TurnId
from managed_agent.core.vocabulary import declare

FAMILY = "turn"

TURN_SUBMITTED = declare("turn.submitted", FAMILY)
TURN_STARTED = declare("turn.started", FAMILY)
TURN_MESSAGE_DELTA = declare("turn.message_delta", FAMILY)
TURN_COMPLETED = declare("turn.completed", FAMILY)
TURN_FAILED = declare("turn.failed", FAMILY)

IdempotencyKey = Annotated[str, Field(pattern=r"^[A-Za-z0-9._:-]{8,255}$")]
"""A well-formed submission key: 8 to 255 characters of an unambiguous alphabet.

One rule, in one place. The request header and the recorded payload are two spellings
of the same value, so both annotate this alias rather than each restating a length and
a character class of their own -- two statements of one rule are free to disagree, and
the disagreement would show up as a key a route accepted and the log could not store.

Written as a type rather than as an exported pattern string on purpose: a module-level
constant here would look like an event type to anything reading this package's
constants, and the alias also carries the rule to every field that names it.
"""


class TurnFailureCause(StrEnum):
    """Why a Turn produced no answer. Machine-readable, and closed."""

    RUNTIME_REPORTED_FAILURE = "runtime_reported_failure"
    RUNTIME_LOST = "runtime_lost"
    POD_UNREACHABLE = "pod_unreachable"


class TurnSubmitted(BaseModel):
    """The payload of a `turn.submitted` event.

    Frozen and extra-forbidding because it is written once and read back by every later
    admission decision for this Session; a field that could be added later is a field
    an older reader would silently ignore.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: TurnId
    idempotency_key: IdempotencyKey
    prompt: str = Field(min_length=1)
