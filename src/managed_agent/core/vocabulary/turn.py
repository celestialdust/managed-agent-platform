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

TurnFailureCause covers the ways a Turn stops short of a delivered answer, and
deliberately carries no runtime error name or error text -- a tenant sees a platform
cause and never the runtime's own (ADR-013). It is not the marker cause set: a marker
answers "why was this work abandoned", and this answers "why did this Turn not produce
an answer".
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.ids import TurnId
from managed_agent.core.vocabulary import declare

FAMILY = "turn"

TURN_SUBMITTED = declare("turn.submitted", FAMILY)
TURN_STARTED = declare("turn.started", FAMILY)
TURN_MESSAGE_DELTA = declare("turn.message_delta", FAMILY)
TURN_COMPLETED = declare("turn.completed", FAMILY)
TURN_FAILED = declare("turn.failed", FAMILY)
TURN_PROGRESS = declare("turn.progress", FAMILY)
"""What the pod is doing, said on a timer rather than when something happens.

Every other type in this family is caused by the runtime saying something. This one is
caused by the clock, and that difference is the whole reason it exists: a Turn that
emits nothing for two minutes and a pod that died two minutes ago produced identical
evidence until this type did, because the only signal either offered was an absence.
The shim appends one of these every `turn_runner._PROGRESS_INTERVAL_S` for as long as
it is running a Turn, so a Turn saying nothing at all is now a broken one.

It moves no Session state -- see the fold's table in `core/session/projection.py`,
which gives it no row -- and it is not webhook-eligible, for the reason
`turn.message_delta` is not: a per-tick type would put one row per tick through the
delivery ledger to say something a reader can get by reading the log.
"""

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
    """Why a Turn produced no answer. Machine-readable, and closed.

    **One name per situation the platform can actually tell apart**, which is the rule
    this set was failing. `POD_UNREACHABLE` was answered for at least four unrelated
    things, and a tenant counted them from the outside before anybody here did: a
    missing internal CA, an MCP server answering 421 from a loopback default, a
    genuinely dead pod, and a deadline killing an agent that was writing a file. Their
    remedies disagree -- resubmitting is right for one and wrong for the rest -- so one
    name for four is worse than no name, because it is a confident wrong answer.

    Members are appended and never inserted: the published table is ordered and a
    member in the middle re-orders a document somebody reads. Every member owes a
    sentence in `REMEDY_FOR` below; a cause a tenant cannot act on has not helped them.
    """

    RUNTIME_REPORTED_FAILURE = "runtime_reported_failure"
    RUNTIME_LOST = "runtime_lost"
    POD_UNREACHABLE = "pod_unreachable"
    # Added 2026-08-25. Until then a Turn whose agent rewrote a delivered artifact was
    # recorded under POD_UNREACHABLE, which the seam that raised it knew was wrong and
    # said so in a comment: the pod was reachable, the store refused the write, and
    # nothing in the closed set named that. A consumer reading the log to decide
    # whether to resubmit got the one answer that is wrong here -- resubmitting re-runs
    # an agent that writes the same path a second time.
    OUTPUT_NOT_REVISABLE = "output_not_revisable"
    # The five below were added 2026-08-26, and each one is a situation that used to
    # arrive as POD_UNREACHABLE. They are exactly the distinctions the code can already
    # make at the point of failure -- no cause here is a guess about which of several
    # things went wrong, and none collapses two the raising seam had told apart.
    #
    # This Session may not be given a pod at all: its environment is not registered,
    # its agent version is archived, its skills do not resolve, its files cannot be
    # placed, or its compiled configuration violates a floor. The pod was never asked
    # for, and no amount of retrying changes any of it.
    SESSION_NOT_PLACEABLE = "session_not_placeable"
    # A pod was asked for and is not running: it never started, or the cluster still
    # reports it starting or gone when the Turn needs it. Nothing about the Session is
    # wrong. This is the one in the group that is worth resubmitting.
    RUNTIME_DID_NOT_START = "runtime_did_not_start"
    # Something is listening on the pod's address and it declined the Turn. Told apart
    # from POD_UNREACHABLE by the fact that there was an answer at all -- which is a
    # real distinction to a tenant, because "nothing is there" and "it said no" have
    # different people fixing them.
    RUNTIME_REFUSED_THE_TURN = "runtime_refused_the_turn"
    # The Turn ran past the total time a Turn is allowed. The pod was reachable and the
    # agent may well have been working -- this is the cause the retired inter-byte
    # deadline used to report as POD_UNREACHABLE, which sent people to look at a
    # network that was fine.
    TURN_DEADLINE_EXCEEDED = "turn_deadline_exceeded"
    # This control plane was built with no transport to a pod, so it can run no Turn at
    # all. Nothing about this Session or this cluster is at fault and no tenant action
    # helps; it is a deployment that is not finished.
    NO_RUNTIME_CONFIGURED = "no_runtime_configured"


REMEDY_FOR: Final[Mapping[TurnFailureCause, str]] = {
    TurnFailureCause.RUNTIME_REPORTED_FAILURE: (
        "the agent runtime reported the Turn as failed -- read the Turn's events for "
        "what it was doing, and submit a new Turn once the prompt or the tools it "
        "needs have been corrected"
    ),
    TurnFailureCause.RUNTIME_LOST: (
        "the Turn was lost while it was running and nothing it produced was kept -- "
        "submit it again"
    ),
    TurnFailureCause.POD_UNREACHABLE: (
        "this session's runtime could not be reached -- submit the Turn again, and "
        "report it if a second attempt fails the same way"
    ),
    TurnFailureCause.OUTPUT_NOT_REVISABLE: (
        "the agent wrote a file it had already produced under that path, and a "
        "delivered file cannot be revised -- have it write the new version under a "
        "different path; submitting the same Turn again will collide again"
    ),
    TurnFailureCause.SESSION_NOT_PLACEABLE: (
        "this session cannot be given a runtime as it is configured -- check that its "
        "environment is registered, that its agent version is not archived, and that "
        "the skills and files it names still exist; submitting again will not help "
        "until one of those changes"
    ),
    TurnFailureCause.RUNTIME_DID_NOT_START: (
        "a runtime was requested for this session and did not come up in time -- "
        "submit the Turn again; nothing about the session needs changing"
    ),
    TurnFailureCause.RUNTIME_REFUSED_THE_TURN: (
        "this session's runtime answered and declined the Turn -- report it rather "
        "than resubmitting, because a second identical Turn is declined the same way"
    ),
    TurnFailureCause.TURN_DEADLINE_EXCEEDED: (
        "this Turn ran past the time a single Turn is allowed and was ended -- split "
        "the work across several Turns, or narrow the prompt, rather than submitting "
        "the same one again"
    ),
    TurnFailureCause.NO_RUNTIME_CONFIGURED: (
        "this deployment has no runtime configured and can run no Turn -- report it; "
        "there is no change to this session that makes a Turn succeed"
    ),
}
"""What a tenant should do next, one sentence per cause and never two the same.

A cause is only worth telling apart from its neighbour if the next move differs, so
these sentences are the test of whether the set above earned its members: two causes
sharing a sentence would be two names for one situation. `tests/control/` asserts both
that the mapping is total and that the sentences are distinct.

Written into each `turn.failed` event rather than looked up when the event is read.
That duplicates a derivable fact, which is normally the wrong trade, and here it is the
right one: the Event Log is append-only and read long after the fact, so a stored row
carries what the platform meant at the time, while a live lookup would answer for
today's vocabulary about a failure from months ago.

Names no pod, no cluster, no bucket and no lane (ADR-013). Every sentence is about the
tenant's session and the tenant's next move, which is what makes it safe to put on a
surface a tenant reads.
"""


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
