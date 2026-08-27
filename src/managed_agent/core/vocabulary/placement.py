"""The placement family: a Turn is waiting for a pod, and how long it waited for one.

**This is the only thing that separates "waiting for a node" from "the model is
thinking".** From outside, both are a Turn that takes a long time, and before this
family nothing anywhere reported which one a tenant was paying for. The fleet
aggregate at `GET /v1/capacity` answers it for an operator; these two answer it for
the one Session whose latency somebody is actually experiencing, on the stream that
Session's tenant already reads, with no operator credential involved.

**Its own family rather than `lifecycle`, and the reason is mechanical.** Every
lifecycle-family type is required to have a row in `projection._TRANSITIONS`, which
placing must not have: a Session waiting for a pod is still `RUNNING`, so a transition
for it would move a state that did not move. Its own family also keeps it away from the
mistake next door -- the lifecycle types are the four a tenant may point a callback at,
and a placement type declared among them invites being marked deliverable, which would
post one callback per pod start to every registered endpoint.

That marking, not the family, is what the webhook tail now reads: `declare(...,
webhook=True)` is the gate, and this family declares nothing with it.

Published, which is what makes either fact reach a tenant at all. The package's
registry discovers the modules in this directory at import and keys the published set
on the type *name*, so the stream's visibility check and the range read both admit
this with no edit to either -- adding a family here is a new file and never an edit to
a switch statement.

**That closes one of the two gates between a runtime and a tenant, and only one.** This
one is publication: which types a tenant may SEE, and a type appended to the log but
absent from it is dropped on the way out silently, with the log and the stream
disagreeing. The other is translation -- `shim/turn_runner.py::_MAPPED`, which decides
which runtime notifications become log events AT ALL, and a notification with no row
there never reaches the log to be published from. Declaring a family here does nothing
about that second gate, and the `thread/started` defect this repository carries is in it
rather than here. What keeps this event clear of it is not the declaration but the
appender: `session.placing` is written by the control plane in `shim/pod_channel.py`,
which does not consult `_MAPPED`, so translation never applies to it.
"""

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.ids import TurnId
from managed_agent.core.vocabulary import declare

FAMILY = "placement"

SESSION_PLACING = declare("session.placing", FAMILY)
"""This Turn found no pod for its Session, and one is being brought up for it.

Appended once per *Turn* rather than once per Session, and that is not a redundancy.
Two Turns can be admitted on one Session before either has a pod -- admission refuses
a Session that will not take a Turn, not a Session with a Turn already open -- so both
wait, and a per-Session event would leave the second one with no record of its own
wait. The `turn_id` is what lets a tenant reading the stream attribute the wait to the
Turn it asked about.

Nothing terminates this event. What ends the wait is the Turn's own next event:
`turn.started` if the pod came up, `turn.failed` if it did not. A closing event of its
own would be a second record of one fact, free to disagree with the Turn's -- and a
reader that has to join two events to learn whether a wait ended is a reader who can
be told "still waiting" by a log that says otherwise.
"""

PLACEMENT_WAITED_MS = "placement_waited_ms"
"""The field name a `turn.started` payload carries its Turn's placement wait under.

A constant rather than a literal at each end, because the field is written by the
dispatch that measured the wait and read by whatever a tenant points at the stream: two
spellings of one name are free to diverge, and the divergence shows up as a field that
is always absent rather than as anything that fails.

**Always present, and `0` for a Turn that found a pod already running.** The field's job
is arithmetic: a tenant subtracts it from the Turn's total latency to learn how much of
that was capacity rather than the model. A consumer doing that subtraction against a
field that is sometimes missing has to special-case the absence at every call site, and
the special case is the whole cost of leaving it out -- while zero is a true measurement
here rather than a stand-in, because a Turn that did not queue did in fact wait no time
for a pod.

Whether placement happened at all is a separate question and it has its own answer:
`session.placing` is in the log for that Turn, or it is not. So nothing is lost by
making this field unconditional -- the two questions stay separately answerable, and the
one that needs a number always has one.
"""


class SessionPlacing(BaseModel):
    """The payload of a `session.placing` event.

    Frozen and extra-forbidding like every other payload in this package: it is written
    once and read back by anything reconstructing why a Turn was slow, and a field that
    could be added later is a field an older reader would silently ignore.

    Carries the Turn and nothing else. Not the pod name, which is a pure function of
    the Session and so already derivable; not the node, which is the scheduler's
    business and would put a cluster detail on a tenant's stream; and not a timestamp,
    because the log row this sits in has a server-set `created_at` and a payload copy
    would be a second spelling of one fact, written by the one clock nobody checks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: TurnId


class PlacementWaited(BaseModel):
    """The one field placement adds to a `turn.started` payload, when it adds any.

    A model rather than a bare int so the field's name and its bound live together and
    are validated at the append rather than trusted. Non-negative because it is an
    elapsed duration: a negative value here would mean the clock went backwards
    between the two reads, and admitting it would publish that as a wait.

    Dumped and merged into the Turn's payload rather than nested under a key, so a
    consumer reads `placement_waited_ms` beside `turn_id` instead of walking into an
    object that exists only for Turns that queued.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    placement_waited_ms: int = Field(ge=0)


def with_placement_wait(
    payload: dict[str, object], waited_ms: int
) -> dict[str, object]:
    """A `turn.started` payload carrying how long that Turn waited for its pod.

    Returns a new mapping and never mutates the one passed in: the caller is inside a
    loop appending one streamed line after another, and a stamp applied in place would
    outlive the line it was meant for.

    **The field is unconditional, and `0` is the honest value for a Turn that found a
    pod already running.** It exists to be subtracted from the Turn's total latency, and
    a consumer doing that against a field that is sometimes missing has to special-case
    the absence everywhere it reads it. Zero is not a stand-in for "we did not measure"
    here -- a Turn that never entered placement genuinely waited no time for a pod, and
    "did placement happen at all" is answered by whether a `session.placing` is in the
    log for that Turn rather than by this number's presence.

    Takes a plain `int` rather than an optional one so a caller cannot express "no
    measurement". Every caller is on the dispatch path and every one of them knows the
    answer: it either wrapped a placement and has an elapsed time, or it did not place
    and the elapsed time is zero.

    The value is validated rather than trusted, so a negative elapsed time -- two reads
    of a wall clock that stepped backwards -- is refused here instead of being published
    as a wait.
    """
    return {**payload, **PlacementWaited(placement_waited_ms=waited_ms).model_dump()}
