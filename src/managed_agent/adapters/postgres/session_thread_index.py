"""One Session's threads, folded out of the events that already carry them.

**A thread here is one that produced an event, not one that was announced.** Those are
different sets and the difference is most of this file: measured against codex-cli
0.149.0 on 2026-08-24, one multiagent Turn emitted six distinct `thread_id` values and
exactly one `thread.started` -- the root's. Deriving the set from announcements listed
one thread of six and answered 404 for five whose events were sitting in the log. So the
set comes from `payload ->> 'thread_id'`, over every type, and the announcement is
demoted to what it actually is: the only place a parent pointer is recorded, present for
some threads and absent for most.

Nothing about a thread is stored as a row, so every fact is derived and no write pays
for any of it. Read-only and lock-free, for the reason the range reader beside it is: it
reads rows already committed and has nothing to serialize against, so a replica can back
it.

**Which index does what, re-measured after the thread set stopped coming from
announcements.** PostgreSQL 17, `event_log` at 89,762 rows, one Session holding 11,125
of them across 201 threads, first page of 25, median of seven runs. All four of
migration 0026's indexes still earn their place, but `event_log_thread_activity` has
changed jobs: the listing's plan no longer touches it at all -- dropping it leaves the
page read at 3.87 ms against 3.97 ms with the buffer count identical at 3,421 -- while
`thread_at` goes from 0.066 ms over 91 buffers to 1.427 ms over 2,340, because without
it the whole Session's thread set is built before one row of it is selected.
`event_log_turn_events` is now the load-bearing one: without it the page read is 121 ms
rather than 4. Dropping `event_log_thread_starts` costs 0.86 ms and
`event_log_thread_archives` 0.78 ms. With none of the four the page read is 130 ms.

**Two statements built from one derivation.** Listing a page and answering for one named
thread differ only in which rows of the thread set they select; the per-thread facts are
identical. They are composed from shared fragments rather than written out twice,
because a second copy of the derivation is free to drift from the first and the drift
would surface as `threads_of` and `thread_at` disagreeing about one thread -- which is
the one thing a caller cannot check for itself.

The cost of composing them is that the guard in
`tests/adapters/test_statements_declare_their_types.py` cannot see it. That guard reads
the source of each assignment that builds a `sa.text(...)`, and here the parameters and
the SELECT list live in fragment variables rather than in the assignment. So the bind
and column types below are declared by hand and are not held in place by that guard; a
reader changing this file has to keep them in step.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.control.session.threads import DEFAULT_PAGE_SIZE, ThreadActivity
from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId
from managed_agent.core.vocabulary import thread, turn

_TERMINAL = (turn.TURN_COMPLETED, turn.TURN_FAILED)
"""The two events that end a Turn, and both are needed.

The placement backlog beside this one deliberately names `turn.started` and
`turn.failed` instead, and the difference is not an inconsistency: that query asks when
a *wait* ended, and a Turn that began has stopped waiting. This one asks whether the
Turn is over, and a Turn that began is exactly the one still running.

`session.archived` is not among them. Stopping a Session does terminate its threads, but
that is `status_of`'s rule and it takes the Session's own state as a separate
argument -- folding it in here would put one fact in two places and make this record
untestable without a Session.
"""

_THREADS = """
    SELECT
        claim.thread_id,
        min(claim.first_seq)                         AS started_seq,
        max(claim.last_at)                           AS updated_at,
        array_remove(array_agg(claim.turn_id), NULL) AS turn_ids
    FROM (
        SELECT
            payload ->> 'thread_id' AS thread_id,
            payload ->> 'turn_id'   AS turn_id,
            min(seq)                AS first_seq,
            max(appended_at)        AS last_at
        FROM event_log
        WHERE session_id = :sid AND payload ->> 'thread_id' IS NOT NULL
        GROUP BY payload ->> 'thread_id', payload ->> 'turn_id'
    ) claim
    GROUP BY claim.thread_id
"""
"""Every thread this Session ever produced an event on, with where it began, when it was
last touched, and which Turns have claimed it.

One pass over the Session's thread-carrying events answers all four, which is the
point: the alternative computes the thread set and then goes back to the log once per
thread for its activity time, and the second visit costs more than the first.

**Grouped twice rather than once, and the inner grouping is the whole speed of this.**
The outer aggregate wants each thread's distinct Turn identifiers, and
`array_agg(DISTINCT ...)` makes PostgreSQL sort every one of the Session's events by
`(thread_id, turn_id)` to feed it -- 7.1 ms of an 8.9 ms read on an 11,000-event
Session, measured. Grouping by `(thread_id, turn_id)` first makes them distinct by
construction, so both levels are hash aggregates over unsorted input and the same read
is 3.9 ms. The inner grouping collapses
11,086 rows to 241 before the outer one sees them.

`started_seq` is the earliest sequence of any event carrying the identifier, not of the
announcement, because most threads have no announcement. It remains a sound cursor for
the same two reasons it was one before: `(session_id, seq)` is the log's primary key so
it is unique within the Session, and `event_log` refuses an UPDATE so the row it names
cannot be rewritten underneath a caller paging through.

`turn_ids` carries no NULL. An event with no `turn_id` contributes nothing to whether a
Turn is open, and leaving the NULL in would make the emptiness test below have to say so
twice.
"""

_ANNOUNCED = """
    SELECT DISTINCT ON (payload ->> 'thread_id')
        payload ->> 'thread_id'        AS thread_id,
        payload ->> 'parent_thread_id' AS parent_thread_id
    FROM event_log
    WHERE session_id = :sid
      AND type = :started_type
      AND payload ->> 'thread_id' IS NOT NULL
    ORDER BY payload ->> 'thread_id', seq
"""
"""The threads the runtime actually announced, and the parent each one named.

This is the only place a parent pointer is written down, and it is a minority of
threads -- so it is joined to the thread set rather than being the thread set. A thread
missing from here is not an error and not a root; it is a thread whose parent nobody
recorded, which is why the record carries `was_announced` rather than leaving a reader
to infer the root from a null parent.

`DISTINCT ON` collapses repeat announcements to the earliest. A thread is announced once
per Turn rather than once per Session -- the shim maps the runtime's `thread/started`
frame inside the Turn loop and the platform's identifier is a v5 UUID of the Session and
the runtime's own string, so the root is re-announced under the same identifier on every
Turn. Without the collapse the join below would multiply a thread's row by the number of
Turns it has run. Earliest rather than latest because a thread's parent is decided when
it begins and cannot change, so the first statement of it is the one that counted.
"""

_BY_SEQ = """
    SELECT * FROM threads
    WHERE started_seq > :after_seq
    ORDER BY started_seq
    LIMIT :limit
"""
"""One page of threads, oldest first, after the cursor and no larger than asked.

`:after_seq` is exclusive, so it names a position already read. A caller that passed no
cursor gets `FIRST_SEQ - 1` rather than a NULL and a second branch in the SQL: sequences
are contiguous from 1, so "above zero" is every sequence there can be.
"""

_BY_ID = """
    SELECT * FROM threads WHERE thread_id = :thread_id
"""
"""The one named thread, wherever it sits in the order.

No LIMIT and no cursor, which is the whole point of a second statement: a thread the
caller can name must answer even when it is not on the first page. At most one row can
match, because the set above is already one row per thread.
"""

_FACTS = """
    SELECT
        page.thread_id,
        announced.parent_thread_id,
        announced.thread_id IS NOT NULL AS was_announced,
        page.started_seq,
        page.updated_at,
        (
            SELECT born.appended_at
            FROM event_log born
            WHERE born.session_id = :sid AND born.seq = page.started_seq
        ) AS created_at,
        (
            SELECT min(retired.appended_at)
            FROM event_log retired
            WHERE retired.session_id = :sid
              AND retired.type = :archived_type
              AND retired.payload ->> 'thread_id' = page.thread_id
        ) AS archived_at,
        NOT EXISTS (
            SELECT 1
            FROM unnest(page.turn_ids) AS claimed(turn_id)
            WHERE NOT EXISTS (
                SELECT 1
                FROM event_log ended
                WHERE ended.session_id = :sid
                  AND ended.type = ANY(:terminal)
                  AND ended.payload ->> 'turn_id' = claimed.turn_id
            )
        ) AS turn_ended
    FROM page
    LEFT JOIN announced ON announced.thread_id = page.thread_id
    ORDER BY page.started_seq
"""
"""What the page's rows still need: their parent, their birth instant, their retirement,
and whether anything is still running on them.

Computed above the page rather than beside the thread set, so each lookup runs at most
`limit` times instead of once per thread the Session ever had. The ORDER BY is repeated
here because a CTE's ordering is not a promise the outer query inherits.

`was_announced` is read off the join rather than from the parent being non-null, because
an announced root legitimately carries no parent and would otherwise be reported as
unannounced -- which is the one thread a consumer most needs to identify.

`created_at` is a primary-key lookup and not a `min(appended_at)`, and the two are not
the same instant. `appended_at` defaults to `now()`, which is the transaction's start
time, so a writer that began early and waited on the append lock lands a later sequence
with an earlier timestamp. The sequence is the ordering this record claims, so the birth
instant has to be the one belonging to that row rather than the smallest in the group.

`archived_at` takes the earliest of however many archives exist. The route is idempotent
and answers a repeat with the first retirement's sequence, so a second `thread.archived`
in the log means a client's call was retried past a response it never saw -- and the
retirement that counted is the one that happened.

`turn_ended` is written as an absence: none of the Turns that claimed this thread is
still open. Read the other way round -- take one `turn_id` and look for its terminal
event -- it needs a branch for a thread that carries none, and it has to choose between
a thread's several Turns. Both fall out of the absence. A thread with no `turn_id` at
all has an empty `turn_ids`, so the search finds nothing to be open and it reads as
ended, which is deliberate: nothing in the log could ever end such a thread, and
reporting it as running would leave a consumer waiting for as long as the log survives.
And a thread claimed by two Turns is running while *either* is open, which is what keeps
the root honest -- its earliest Turn has completed on every Session past its first, and
keying off that one alone would report the root as idle while it was mid-answer.
"""


def _statement(page: str) -> sa.TextualSelect:
    """One of the two reads: the shared thread set and announcements, the given page
    selection, and the shared per-thread facts.

    The types are declared rather than left to the driver for the reason every textual
    statement here declares them: `sa.text` carries no metadata, so asyncpg is handed
    the bare Python object going in and the raw decode coming back. The three timestamps
    are the ones that would bite -- one arriving as text has no `.timestamp()` to call,
    and the failure would surface in the millisecond conversion rather than here.
    """
    return (
        sa.text(
            f"WITH threads AS ({_THREADS}), announced AS ({_ANNOUNCED}),"
            f" page AS ({page}) {_FACTS}"
        )
        .bindparams(
            sa.bindparam("sid", type_=sa.Uuid()),
            sa.bindparam("terminal", type_=sa.ARRAY(sa.Text())),
        )
        .columns(
            thread_id=sa.Text(),
            parent_thread_id=sa.Text(),
            was_announced=sa.Boolean(),
            started_seq=sa.BigInteger(),
            created_at=sa.TIMESTAMP(timezone=True),
            updated_at=sa.TIMESTAMP(timezone=True),
            archived_at=sa.TIMESTAMP(timezone=True),
            turn_ended=sa.Boolean(),
        )
    )


_PAGE = _statement(_BY_SEQ)
_ONE = _statement(_BY_ID)


def _binds(session_id: SessionId) -> dict[str, object]:
    """The parameters both statements share: the Session, and the event types they read.

    The type names come from the vocabulary rather than being spelled into the SQL, so
    renaming a published type is one edit and cannot leave this query matching a string
    nothing appends any more.

    Built per call rather than held as a module constant, because `terminal` is a list:
    one shared mutable value handed to every read is a thing a caller could alter for
    every other caller, and the cost of not having it is one dict per query.
    """
    return {
        "sid": session_id,
        "started_type": thread.THREAD_STARTED,
        "archived_type": thread.THREAD_ARCHIVED,
        "terminal": list(_TERMINAL),
    }


def _epoch_ms(when: datetime) -> int:
    """A server clock read as epoch milliseconds.

    Milliseconds, and by the same arithmetic the placement backlog uses, so that two
    instants this platform publishes are comparable without a conversion nobody wrote
    down. Truncating rather than rounding is monotonic in the instant, which is what
    keeps `updated_at_ms` from landing below `created_at_ms` when the two share a
    millisecond.
    """
    return int(when.timestamp() * 1000)


def _activity(row: sa.Row[Any]) -> ThreadActivity:
    """One result row as the record the port publishes.

    Neither assertion is defensive. `updated_at` aggregates over the rows carrying this
    thread's identifier and the thread exists precisely because such rows do, so an
    empty aggregate would mean two halves of one statement disagreed about which rows
    they saw. `created_at` reads the row at the sequence the same aggregate reported, in
    the same snapshot, so a miss would mean the sequence named a row that was never
    there. Either would put a thread with no clock in front of a caller, and stopping is
    louder than inventing one.
    """
    updated, created, archived = row.updated_at, row.created_at, row.archived_at
    assert updated is not None, (
        f"thread {row.thread_id} aggregated to no most-recent event although it exists "
        "only because its events do, so two halves of one statement saw different rows"
    )
    assert created is not None, (
        f"thread {row.thread_id} reports earliest sequence {row.started_seq}, which "
        "holds no row in this Session -- the aggregate and the lookup read one "
        "snapshot and cannot disagree"
    )
    parent = row.parent_thread_id
    return ThreadActivity(
        thread_id=str(row.thread_id),
        parent_thread_id=None if parent is None else str(parent),
        was_announced=bool(row.was_announced),
        started_seq=Seq(int(row.started_seq)),
        created_at_ms=_epoch_ms(created),
        updated_at_ms=_epoch_ms(updated),
        archived_at_ms=None if archived is None else _epoch_ms(archived),
        turn_ended=bool(row.turn_ended),
    )


class PostgresSessionThreadIndex:
    """Reads one Session's threads out of `event_log`.

    Holds the engine and nothing else, because there is nothing to keep: a thread's
    facts are a property of the log at the moment it is asked, so two replicas answering
    the same question answer it the same way.

    Keyed on the Session and on nothing else, exactly as the port says. There is no
    tenant term because `event_log` has no tenant column, so a caller reaching here has
    already established that its own caller may address this Session -- and a listing
    that leaked another Session's threads would do so through the Session predicate
    being wrong, which is why that is what the tests hold down hardest.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def threads_of(
        self,
        session_id: SessionId,
        *,
        after_seq: Seq | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> Sequence[ThreadActivity]:
        """A page of this Session's threads, oldest first, after `after_seq`.

        `limit` is passed through rather than clamped. The port publishes a maximum and
        the surface that parses a caller's query string is where it belongs: clamping
        here as well would put the same bound in two places, and the one that silently
        shrank a page would be indistinguishable from the end of the listing.
        """
        async with self._engine.connect() as connection:
            rows = (
                await connection.execute(
                    _PAGE,
                    _binds(session_id)
                    | {
                        "after_seq": FIRST_SEQ - 1 if after_seq is None else after_seq,
                        "limit": limit,
                    },
                )
            ).all()
        return [_activity(row) for row in rows]

    async def thread_at(
        self, session_id: SessionId, thread_id: str
    ) -> ThreadActivity | None:
        """The named thread of this Session, or None if this Session has no such thread.

        A thread that produced events but was never announced answers here like any
        other, which is the half of this that was missing: its events are addressable
        through the same identifier the listing hands out, so a caller that reads a
        thread out of a listing can always read it back.
        """
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    _ONE, _binds(session_id) | {"thread_id": thread_id}
                )
            ).one_or_none()
        return None if row is None else _activity(row)
