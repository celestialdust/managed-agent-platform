"""Thread event types: which agent said a thing, and whose child that agent is.

ADR-007 accepted that subagent activity appears on the Session's event stream "tagged
with the thread that produced it and that thread's parent", and that the event schema
therefore needs a thread identifier and a parent pointer. Neither existed. A multiagent
Session's subagent text arrived under the same method the root agent uses, so it was
published unattributed -- one undifferentiated voice, which is worse than publishing
nothing because it reads as coherent.

Its own family rather than a member of `turn`, because a thread is not a phase of a
Turn. Several threads live inside one Turn and a thread is addressed by grouping the
events attributed to it, so a type sitting in the turn family would suggest an ordering
between threads that does not exist.

**The parent pointer is on this event and on no other.** A thread's parent is decided
once, when the thread begins, and it cannot change. Carried on every event of the thread
it would be N copies of one fact, free to disagree; carried here it is stated once by
the event that declares the thread exists.

**The root thread gets one too, and its parent is null.** This was written the other
    way round first -- that the runtime announces a thread only when it spawns one, so
    an absent `thread.started` is how a reader identifies the root. Measured against
    codex-cli 0.149.0 on 2026-08-24, that is false: the root thread is announced before
    the Turn even starts, with no parent, and `parent_thread_id` being null is what
    identifies it. The correction is here rather than in a note because the claim it
    replaces was load-bearing -- a consumer that identified the root by absence would
    find no root at all.

    So `parent_thread_id` is optional for two reasons that are worth keeping apart: the
    root has no parent and never will, and a spawned thread's frame may carry no parent
    the platform can see. Both are real states, and dropping the event for want of a
    parent would lose the only record that a thread began.
"""

from pydantic import BaseModel, ConfigDict

from managed_agent.core.vocabulary import declare

FAMILY = "thread"

THREAD_STARTED = declare("thread.started", FAMILY)


class ThreadStarted(BaseModel):
    """The payload of a `thread.started` event: a thread and the thread that spawned it.

    `parent_thread_id` is optional because the runtime's frame may carry no parent, and
    a thread whose parent is unknown is a real state rather than an error: dropping the
    event for want of a parent would lose the only record that the thread began, which
    is the one thing this event exists to say.

    Both are strings and not typed identifiers, and they are **not** the runtime's own
    values. The runtime mints a thread id per process, ADR-007 (MAP-A10) forbids
    one reaching a caller, so the appending side derives a v5 UUID from the Session and
    the runtime's string and publishes that. A string rather than `UUID` because the
    derivation is the appender's rule and not this payload's: typing the field would
    make every consumer's parse depend on a choice that could change without the
    contract changing, and a thread identifier is opaque to everyone downstream anyway.

    No name, no model, no timestamp. The thread's position in time is its event's own
    sequence, and everything else about the agent that runs in it belongs to the
    Session's definition, which is already recorded once at creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    parent_thread_id: str | None = None


THREAD_ARCHIVED = declare("thread.archived", FAMILY)


class ThreadArchived(BaseModel):
    """The payload of a `thread.archived` event: this thread will publish nothing more.

    **Appended by the control plane, never by a pod.** It is not in `ShimEventType` and
    must not be: a Session pod reports what its runtime did, and retirement is a
    decision a tenant made through the API. That is also why the archive carries no
    runtime call -- see `control/session/threads.py` for what this platform can and
    cannot retire.

    One field, and no timestamp. The event's own `appended_at` is when the archive
    happened, so a time in the payload would be a second copy free to disagree with it.

    Terminal and idempotent at the route, which is what keeps this one event per thread:
    a second archive answers with the first one's sequence rather than appending again,
    so a client whose call timed out cannot put two retirements of one thread in a log
    that has no way to express which of them counted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
