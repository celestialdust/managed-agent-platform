"""What a Session's threads are, read out of the log that already records them.

A thread is not a row anywhere. It is the set of events carrying one thread identifier,
which is what ADR-007's attribution made addressable and what ADR-007's revision
now permits a tenant to name -- read and archive only, so no tenant-facing
verb spawns one and fan-out stays the model's decision.

**Nothing new is written for a thread to exist.** `thread.started` already carries the
identifier and the parent pointer, every event the shim appends already carries the
thread it came from, and both were platform-issued at the append (wave 3). So this
module reads; the only event it adds is the archive, which is a fact about the
platform's record rather than about the runtime.

**Why a query of its own rather than folding the whole log in Python.** `whole_log` plus
a fold is the pattern two other routes use and it was the first design here. It cannot
answer this one: a thread's `created_at` is the wall-clock of its `thread.started` row,
`EventRecord` does not carry `appended_at`, and adding it there is the same bill wave 3
already refused -- upwards of twenty hand-written `Event` classes across `tests/`, every
one of them failing `mypy --strict` for a field only this surface reads. A new port
costs one fake in new tests instead, so the cheaper change is the narrower one.

**Why the timestamps are milliseconds.** `SessionListed.created_at_ms` is the same
quantity on the same wire and it is milliseconds; a second unit here would make two
creation times on one API incomparable without a conversion nobody documents.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from managed_agent.core.ids import Seq, SessionId

DEFAULT_PAGE_SIZE: Final = 25
MAX_PAGE_SIZE: Final = 100
"""How many threads one page may hold.

The default is the runtime's own live-thread ceiling, so a Session whose threads are all
live fits in one page and a caller that never pages is not silently reading a prefix.
The maximum is above it because archived threads stay listed -- `archived_at` is a
published field, so retirement is visible rather than a disappearance -- and a
long-lived Session therefore accumulates more threads than it can ever run at once.
"""


class ThreadStatus(StrEnum):
    """What a thread is doing, in the three states this platform can tell apart.

    Upstream publishes a fourth, `rescheduling`. It is deliberately absent: a thread
    here is rescheduled only by its Session's pod being replaced, and that replacement
    is reported on the Session, not per thread. Publishing a member this platform can
    never emit would put a state in a client's switch that no answer ever reaches.

    `terminated` covers both retirement and the end of the Session that held the thread.
    They are one state because a thread cannot outlive its Session: once the Session is
    stopped, no further event can carry the thread's identifier, so "archived" and "the
    Session is over" are indistinguishable to every consumer downstream of this.
    """

    RUNNING = "running"
    IDLE = "idle"
    TERMINATED = "terminated"


@dataclass(frozen=True, slots=True)
class ThreadActivity:
    """One thread as the log holds it: where it began, when it was last touched, whether
    the Turn that opened it has closed.

    `turn_ended` rather than a status, because the status also depends on whether the
    Session is still open and this record is about one thread's own events. Keeping the
    two apart is what lets the rule be tested without a store: `status_of` is a function
    of this plus one boolean.

    `updated_at_ms` is the wall-clock of the thread's most recent event and not of its
    Turn's. A thread that produced nothing after it began therefore reports its creation
    time, which is accurate -- nothing has happened on it since.

    **`was_announced` is why `parent_thread_id` being None is not enough.** A thread
    exists here because it produced an event, not because the runtime announced it, and
    those are different sets: measured against codex-cli 0.149.0 on 2026-08-24, one Turn
    produced six threads and exactly one `thread.started` -- the root's. So five threads
    had events, a `turn_id` and a position in the log, and no parent pointer anywhere.
    Without this flag their `parent_thread_id` of None would be indistinguishable from
    the root's, and a consumer looking for the root by that null would find six.

    Derived from whether the announcement exists rather than from the parent being
    absent, because those genuinely differ: an announced thread may legitimately carry
    no parent, which is exactly what the root is.
    """

    thread_id: str
    parent_thread_id: str | None
    was_announced: bool
    started_seq: Seq
    created_at_ms: int
    updated_at_ms: int
    archived_at_ms: int | None
    turn_ended: bool


@runtime_checkable
class SessionThreadIndex(Protocol):
    """The threads of one Session, keyed by Session and by nothing else.

    **This applies no tenant predicate and cannot.** The log is keyed by Session and
    carries no tenant, exactly as `EventLogRange` does, so every caller is responsible
    for having established that its caller may address this Session before it gets here.

    `runtime_checkable` for the reason the other capability ports are: the composition
    root wires one object and a process assembled without a store narrows to the
    refusing stand-in rather than failing at import.
    """

    async def threads_of(
        self,
        session_id: SessionId,
        *,
        after_seq: Seq | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> Sequence[ThreadActivity]:
        """A page of this Session's threads in the order they began, oldest first.

        Ordered by the sequence of the `thread.started` event, which is why the cursor
        is that sequence: it is already strictly ordered within a Session, it is unique
        -- `(session_id, seq)` is the log's primary key -- and a thread's opening event
        cannot be rewritten, so a position in this order is stable across re-reads in a
        way a creation timestamp shared by two threads in one millisecond would not be.

        `after_seq` is exclusive and names a position already read. **At most `limit`
        records come back**, so a short page means the end and a full one says nothing
        about whether more exist; the caller reads again from the last sequence it saw.
        """
        ...

    async def thread_at(
        self, session_id: SessionId, thread_id: str
    ) -> ThreadActivity | None:
        """One named thread of this Session, or None if the Session has no such thread.

        None rather than an exception, because the caller turns it into the same refusal
        an unreadable Session gets and has nothing else to do with the distinction.
        """
        ...


class NoSessionThreads:
    """Answers every read as though the Session had no threads. What a `Platform` built
    without a thread index holds.

    Empty rather than raising, and the asymmetry with `UnconfiguredAttachments` is
    deliberate: refusing a placement protects a Session from running without a file it
    was promised, while refusing a read protects nothing -- it turns a deployment that
    cannot answer into a 500 on a route whose honest answer is that it knows of no
    threads. A Session whose events predate wave 3 genuinely has none, so a caller must
    already handle an empty listing.
    """

    async def threads_of(
        self,
        session_id: SessionId,
        *,
        after_seq: Seq | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> Sequence[ThreadActivity]:
        return ()

    async def thread_at(
        self, session_id: SessionId, thread_id: str
    ) -> ThreadActivity | None:
        return None


def status_of(activity: ThreadActivity, *, session_open: bool) -> ThreadStatus:
    """Which of the three states this thread is in.

    Archived wins over everything, including a Turn still in flight. That ordering is
    not reachable through the archive route -- it refuses a thread whose Turn is open --
    but it is reachable in the log, because a Session can be resumed and take a further
    Turn after a thread was archived. Reporting such a thread as running would say the
    platform is about to publish events for it, and it will not: the archive is what
    tells a consumer to stop waiting.

    A closed Session terminates every thread it held, for the reason `ThreadStatus`
    gives: no further event can carry the identifier, so nothing distinguishes those
    threads from retired ones.

    Otherwise the thread is running exactly while the Turn that opened it is open. A
    thread cannot outlive its Turn -- the runtime's subagents are spawned inside one and
    the Turn does not close until they are done -- so the Turn's own closure is the
    thread's, and no per-thread terminal event is needed to know it.
    """
    if activity.archived_at_ms is not None or not session_open:
        return ThreadStatus.TERMINATED
    return ThreadStatus.IDLE if activity.turn_ended else ThreadStatus.RUNNING
