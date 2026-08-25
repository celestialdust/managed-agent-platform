"""Reading one Session's threads back out of the log, against real PostgreSQL.

Tier 1 (testcontainers, real PostgreSQL 17), and the tier is the whole point rather
than a habit. Every fact this adapter reports is produced by SQL a fake cannot stand in
for: two levels of grouping over a JSONB extraction, an outer join to a set most threads
are missing from, and a nested anti join over an array. Those are exactly where it would
be wrong, so a double that answered from Python would prove nothing about the thing that
ships.

**Most of these cases seed threads that were never announced**, because that is what a
multiagent Turn actually produces: measured against the cluster, one Turn emitted six
`thread_id` values and one `thread.started`. A suite that announced every thread it
tested would have passed against the implementation that listed one thread of six.

**The events are written with an append time the test chooses**, one second apart,
rather than through `PostgresEventLogAppend` with its `now()` default. Two appends
microseconds apart share a millisecond, and half of what is under test here is which
instant each field takes -- "the archive that counted is the earliest" and "activity
moves with the last event" are both unfalsifiable when every row reports the same
millisecond. A test that passes because two values collided is worse than no test, so
the clock is an input.

No shared-state fixture, and the absence is deliberate. This read is scoped to one
Session, so a case that gives itself a fresh Session id cannot see another case's
rows -- which is the same isolation the table itself provides, the sequence being per
Session. The listing-wide gauges next door need their table emptied; this one does not.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.session_thread_index import (
    PostgresSessionThreadIndex,
)
from managed_agent.control.session.threads import SessionThreadIndex, ThreadActivity
from managed_agent.core.ids import Seq, SessionId, new_session_id
from managed_agent.core.vocabulary import thread, tool_call, turn

_START = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
_STEP = timedelta(seconds=1)

_APPEND = sa.text(
    "INSERT INTO event_log (session_id, seq, type, payload, appended_at)"
    " VALUES (:sid, :seq, :type, :payload, :when)"
).bindparams(
    sa.bindparam("sid", type_=sa.Uuid()),
    sa.bindparam("payload", type_=sa.JSON()),
    sa.bindparam("when", type_=sa.TIMESTAMP(timezone=True)),
)


def _ms(when: datetime) -> int:
    """The instant as this platform publishes it.

    Here to say *which* instant a field carries, which is what these cases assert. It is
    not an independent check of the millisecond arithmetic -- it is the same arithmetic,
    written out so a case can name the row it expects rather than a number.
    """
    return int(when.timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class _Written:
    """Where one seeded event landed: its sequence and the instant it was written."""

    seq: Seq
    at: datetime


class _Log:
    """One Session's log, written a second at a time from a clock the test owns.

    Sequences are handed out from 1 upwards, as the append adapter would, because they
    are the paging cursor and a case reads its own expectations off them. `session_seq`
    is deliberately not maintained: nothing read here consults it, and a test that keeps
    a table it does not exercise in step is a test that will drift from the real writer
    without anything noticing.
    """

    def __init__(self, engine: AsyncEngine, session_id: SessionId) -> None:
        self.session_id = session_id
        self._engine = engine
        self._seq = 0
        self._at = _START

    async def add(
        self, type_: str, payload: dict[str, object], *, at: datetime | None = None
    ) -> _Written:
        """Append one event, at the next second unless the case names an instant.

        `at` exists for the one property that needs the two clocks to disagree:
        `appended_at` defaults to the transaction's start time, so a writer that began
        early and waited on the append lock lands a later sequence carrying an earlier
        timestamp. Passing an instant leaves the running clock where it was, so the
        events after it continue in order and only the one row is out of step.
        """
        self._seq += 1
        if at is None:
            self._at += _STEP
        async with self._engine.begin() as connection:
            await connection.execute(
                _APPEND,
                {
                    "sid": self.session_id,
                    "seq": self._seq,
                    "type": type_,
                    "payload": payload,
                    "when": self._at if at is None else at,
                },
            )
        return _Written(Seq(self._seq), self._at if at is None else at)


def _log(engine: AsyncEngine) -> _Log:
    return _Log(engine, new_session_id())


def _id() -> str:
    """A thread or Turn identifier shaped like the ones the platform issues.

    A v4 here where production derives a v5 from the Session and the runtime's own
    string: nothing under test parses one, so what matters is only that two calls
    differ.
    """
    return str(uuid.uuid4())


async def _announced(
    log: _Log, thread_id: str, *, turn_id: str | None = None, parent: str | None = None
) -> _Written:
    """A `thread.started` shaped the way the shim appends one.

    `turn_id` and `parent_thread_id` are both omitted from the payload when absent
    rather than written as JSON nulls, because that is what the shim produces -- it
    spreads the fields a frame carried, and a frame carrying no parent contributes no
    key. The adapter has to read an absent key and a null key alike, and only one of the
    two is what it will actually meet.
    """
    payload: dict[str, object] = {"thread_id": thread_id}
    if turn_id is not None:
        payload["turn_id"] = turn_id
    if parent is not None:
        payload["parent_thread_id"] = parent
    return await log.add(thread.THREAD_STARTED, payload)


async def _turn_opened(log: _Log, thread_id: str, turn_id: str) -> _Written:
    """A `turn.started` attributed to a thread -- how an unannounced thread first shows.

    This is the event that revealed the defect this file was rewritten for. The runtime
    announces the root and nothing else, but it opens a turn on every thread it runs, so
    a spawned subagent's first appearance in the log is one of these and never a
    `thread.started`.
    """
    return await log.add(
        turn.TURN_STARTED, {"turn_id": turn_id, "thread_id": thread_id}
    )


async def _spoke(log: _Log, thread_id: str, turn_id: str) -> _Written:
    """An ordinary event attributed to a thread -- what makes its activity time move."""
    return await log.add(
        turn.TURN_MESSAGE_DELTA,
        {"turn_id": turn_id, "thread_id": thread_id, "text": "hello"},
    )


async def _completed(log: _Log, turn_id: str) -> _Written:
    """A `turn.completed`, which the shim appends with no thread attribution at all."""
    return await log.add(turn.TURN_COMPLETED, {"turn_id": turn_id, "text": "done"})


async def _failed(log: _Log, turn_id: str) -> _Written:
    return await log.add(
        turn.TURN_FAILED,
        {"turn_id": turn_id, "cause": turn.TurnFailureCause.RUNTIME_LOST.value},
    )


async def _archived(log: _Log, thread_id: str) -> _Written:
    return await log.add(thread.THREAD_ARCHIVED, {"thread_id": thread_id})


def _index(engine: AsyncEngine) -> PostgresSessionThreadIndex:
    return PostgresSessionThreadIndex(engine)


def test_the_adapter_satisfies_the_thread_index_port() -> None:
    """Graded at run time, because the composition root narrows to this protocol.

    `issubclass` rather than `isinstance`, so a question about the class needs no engine
    to answer it -- and `runtime_checkable` decides it on method names, which is exactly
    what the narrowing asks.
    """
    assert issubclass(PostgresSessionThreadIndex, SessionThreadIndex)


async def test_an_announced_root_reports_the_announcement_as_its_beginning(
    engine: AsyncEngine,
) -> None:
    """The root is announced, so its parent is a real null and its birth is that row.

    A `turn.started` follows on the same thread deliberately. Both events carry the
    identifier, so both are candidates for "where this thread began", and the earliest
    is the announcement -- which is why the root's numbers do not move when the thread
    set stops being derived from announcements.
    """
    log = _log(engine)
    root, turn_id = _id(), _id()
    opening = await _announced(log, root, turn_id=turn_id)
    await _turn_opened(log, root, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert listed == [
        ThreadActivity(
            thread_id=root,
            parent_thread_id=None,
            was_announced=True,
            started_seq=opening.seq,
            created_at_ms=_ms(opening.at),
            updated_at_ms=_ms(opening.at + _STEP),
            archived_at_ms=None,
            turn_ended=False,
        )
    ]


async def test_a_thread_with_events_and_no_announcement_is_listed(
    engine: AsyncEngine,
) -> None:
    """A thread exists because it produced an event, not because anyone announced it.

    This is the whole defect. The runtime opens a turn on a spawned subagent's thread
    and never says the thread began, so before this the thread had events, an
    identifier and a `turn_id` in the log, and no listing would admit it existed.

    `was_announced` False is what keeps its null parent honest: it has no parent
    recorded, which is a different fact from the root's, which has none because there
    is none.
    """
    log = _log(engine)
    child, turn_id = _id(), _id()
    first = await _turn_opened(log, child, turn_id)
    await _spoke(log, child, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.thread_id for one in listed] == [child]
    assert listed[0].was_announced is False
    assert listed[0].parent_thread_id is None
    assert listed[0].started_seq == first.seq
    assert listed[0].created_at_ms == _ms(first.at)


async def test_a_multiagent_turn_lists_every_thread_it_produced(
    engine: AsyncEngine,
) -> None:
    """Six threads, one announcement: the shape the cluster produced.

    The measured Turn emitted one `thread.started` for the root and a `turn.started` on
    each of six thread identifiers. Derived from announcements this listed one thread of
    six and answered 404 for the other five, whose events were in the log the whole
    time.

    Ordered by earliest event, so the root leads -- its announcement precedes every
    turn -- and the five children follow in the order the runtime opened them. Only the
    root reports `was_announced`.
    """
    log = _log(engine)
    root, turn_id = _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    await _turn_opened(log, root, turn_id)
    children = [_id() for _ in range(5)]
    for child in children:
        await _turn_opened(log, child, turn_id)
    await _spoke(log, root, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.thread_id for one in listed] == [root, *children]
    assert [one.was_announced for one in listed] == [True, *([False] * 5)]
    seqs = [one.started_seq for one in listed]
    assert seqs == sorted(seqs), f"threads came back out of log order: {seqs}"


async def test_an_unannounced_thread_ends_when_its_turn_does(
    engine: AsyncEngine,
) -> None:
    """A child shares the Turn it was spawned inside, so the Turn's end is the child's.

    There is no per-thread terminal event and there does not need to be one: the
    runtime's subagents run inside a Turn and the Turn does not close until they are
    done, so a completion for the Turn settles every thread that ran in it.
    """
    log = _log(engine)
    child, turn_id = _id(), _id()
    await _turn_opened(log, child, turn_id)
    await _completed(log, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.turn_ended for one in listed] == [True]


async def test_an_unannounced_thread_whose_turn_is_open_is_still_running(
    engine: AsyncEngine,
) -> None:
    """Nothing has ended the Turn, so the thread it runs in has not ended either.

    The pair with the case above is the point: an unannounced thread has no
    `thread.started` to read a `turn_id` off, so an implementation that looked there
    would answer the same for both -- and answering "ended" for both is the failure that
    hides, because it looks like an idle Session rather than a broken one.
    """
    log = _log(engine)
    child, turn_id = _id(), _id()
    await _turn_opened(log, child, turn_id)
    await _spoke(log, child, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.turn_ended for one in listed] == [False]


async def test_a_child_is_listed_after_its_parent_and_names_it(
    engine: AsyncEngine,
) -> None:
    """When the runtime does announce a child, the parent pointer survives the join.

    Oldest first, and the child carries the id of the thread that spawned it -- the pair
    being what lets a consumer rebuild the tree from a flat listing.
    """
    log = _log(engine)
    root, child, turn_id = _id(), _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    await _announced(log, child, turn_id=turn_id, parent=root)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.thread_id for one in listed] == [root, child]
    assert [one.parent_thread_id for one in listed] == [None, root]
    assert [one.was_announced for one in listed] == [True, True]
    assert listed[0].started_seq < listed[1].started_seq


async def test_the_cursor_pages_forward_and_then_runs_out(
    engine: AsyncEngine,
) -> None:
    """`after_seq` is exclusive, so a page starts past the last thread already read.

    Read one at a time deliberately: with a limit above the number of threads a caller
    never exercises the cursor at all, and the bug this pins -- an inclusive cursor
    returning the last thread of the previous page forever -- only shows up when the
    cursor is the thing that advances.
    """
    log = _log(engine)
    first, second, turn_id = _id(), _id(), _id()
    await _turn_opened(log, first, turn_id)
    await _turn_opened(log, second, turn_id)
    index = _index(engine)

    page_one = await index.threads_of(log.session_id, limit=1)
    assert [one.thread_id for one in page_one] == [first]

    page_two = await index.threads_of(
        log.session_id, after_seq=page_one[0].started_seq, limit=1
    )
    assert [one.thread_id for one in page_two] == [second]

    beyond = await index.threads_of(
        log.session_id, after_seq=page_two[0].started_seq, limit=1
    )
    assert beyond == [], "a cursor past the last thread must read as the end, not wrap"


async def test_a_limit_caps_the_page(engine: AsyncEngine) -> None:
    """Three threads, a limit of two, and the two returned are the oldest two."""
    log = _log(engine)
    turn_id = _id()
    ids = [_id() for _ in range(3)]
    for one in ids:
        await _turn_opened(log, one, turn_id)

    listed = await _index(engine).threads_of(log.session_id, limit=2)

    assert [one.thread_id for one in listed] == ids[:2]


async def test_a_completed_turn_ends_its_thread(engine: AsyncEngine) -> None:
    log = _log(engine)
    root, turn_id = _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    await _completed(log, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.turn_ended for one in listed] == [True]


async def test_a_failed_turn_ends_its_thread(engine: AsyncEngine) -> None:
    """A failure ends a Turn as surely as a completion.

    Its own case rather than a parameter on the one above, because the two are separate
    published types precisely so a consumer cannot treat one as the other by forgetting
    a field -- and a query naming only `turn.completed` would report a thread whose Turn
    failed as running for as long as the log survives.
    """
    log = _log(engine)
    root, turn_id = _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    await _failed(log, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.turn_ended for one in listed] == [True]


async def test_a_turn_still_in_flight_leaves_its_thread_running(
    engine: AsyncEngine,
) -> None:
    """No terminal event for the Turn, so the thread it opened has not ended.

    The Turn is deliberately given a `turn.started` and some activity: an implementation
    that answered from "has anything happened since" rather than from the Turn's own
    terminal event would pass with an empty log and fail here.
    """
    log = _log(engine)
    root, turn_id = _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    await _turn_opened(log, root, turn_id)
    await _spoke(log, root, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.turn_ended for one in listed] == [False]


async def test_a_completion_of_another_turn_does_not_end_this_thread(
    engine: AsyncEngine,
) -> None:
    """Two Turns of one Session, and only one of them has finished.

    A query matched on the Session rather than on the Turn identifier would let the
    first Turn's completion close the second Turn's thread -- and a Session with two
    Turns is the ordinary case, not an exotic one.
    """
    log = _log(engine)
    done, running = _id(), _id()
    finished_thread, live_thread = _id(), _id()
    await _turn_opened(log, finished_thread, done)
    await _completed(log, done)
    await _turn_opened(log, live_thread, running)

    listed = await _index(engine).threads_of(log.session_id)

    assert {one.thread_id: one.turn_ended for one in listed} == {
        finished_thread: True,
        live_thread: False,
    }


async def test_a_thread_naming_no_turn_anywhere_reads_as_ended(
    engine: AsyncEngine,
) -> None:
    """A thread whose Turn cannot be identified is not reported as running.

    There is nothing in the log that could ever end it, so the alternative is a thread
    that stays running for as long as the Session's events survive -- a consumer waiting
    on it would wait forever.
    """
    log = _log(engine)
    orphan = _id()
    await _announced(log, orphan)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.turn_ended for one in listed] == [True]


async def test_a_thread_touched_by_two_turns_is_listed_once(
    engine: AsyncEngine,
) -> None:
    """The root runs again on the Session's second Turn, under the same identifier.

    That is the ordinary shape of any Session past its first Turn: the identifier is
    derived from the Session and the runtime's own string, and neither changes between
    Turns. Listed twice, one thread would occupy two rows of a page and a consumer
    keying on the identifier would see it overwrite itself. The surviving row keeps the
    earliest sequence, because that is the paging cursor and the port promises a
    position in that order is stable across re-reads.
    """
    log = _log(engine)
    root, first, second = _id(), _id(), _id()
    opening = await _announced(log, root, turn_id=first)
    await _completed(log, first)
    await _announced(log, root, turn_id=second)
    await _completed(log, second)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.thread_id for one in listed] == [root]
    assert listed[0].started_seq == opening.seq
    assert listed[0].created_at_ms == _ms(opening.at)


async def test_a_thread_is_running_while_either_of_its_turns_is_open(
    engine: AsyncEngine,
) -> None:
    """The same thread on two Turns, the second still in flight.

    Answered from the thread's earliest Turn alone it reads as idle, because that Turn
    did finish -- while the agent in it is mid-answer. This is not a corner: the root
    thread of every Session past its first Turn is in exactly this state, so keying off
    one Turn would report the most visible thread on the platform as idle whenever it
    was working.
    """
    log = _log(engine)
    root, finished, running = _id(), _id(), _id()
    await _announced(log, root, turn_id=finished)
    await _completed(log, finished)
    await _turn_opened(log, root, running)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.turn_ended for one in listed] == [False]


async def test_an_archived_thread_carries_the_instant_it_was_retired(
    engine: AsyncEngine,
) -> None:
    log = _log(engine)
    root, turn_id = _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    await _completed(log, turn_id)
    retired = await _archived(log, root)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.archived_at_ms for one in listed] == [_ms(retired.at)]


async def test_a_second_archive_does_not_displace_the_first(
    engine: AsyncEngine,
) -> None:
    """Two retirements of one thread, and the earlier one is the answer.

    The route is idempotent, so a second `thread.archived` in the log means a client
    retried past a response it never received. Nothing about the thread changed at the
    second row, and reporting it would move a published instant for a call that was only
    ever repeated.
    """
    log = _log(engine)
    root, turn_id = _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    await _completed(log, turn_id)
    first = await _archived(log, root)
    second = await _archived(log, root)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.archived_at_ms for one in listed] == [_ms(first.at)]
    assert listed[0].archived_at_ms != _ms(second.at), (
        "the later retirement was published; a repeated call moved a recorded instant"
    )


async def test_activity_moves_with_the_last_event_naming_the_thread(
    engine: AsyncEngine,
) -> None:
    """Any event carrying the identifier counts, whatever family it belongs to.

    A tool call is used rather than a message delta on purpose: a thread's activity is
    everything the agent in it did, and a query narrowed to the thread family or to the
    turn family would report a busy thread as untouched since it began.
    """
    log = _log(engine)
    child, turn_id = _id(), _id()
    first = await _turn_opened(log, child, turn_id)
    last = await log.add(
        tool_call.TOOL_CALLED,
        {"turn_id": turn_id, "thread_id": child, "name": "bash"},
    )

    listed = await _index(engine).threads_of(log.session_id)

    assert listed[0].created_at_ms == _ms(first.at)
    assert listed[0].updated_at_ms == _ms(last.at)
    assert listed[0].updated_at_ms > listed[0].created_at_ms


async def test_a_thread_that_did_nothing_reports_its_own_creation_time(
    engine: AsyncEngine,
) -> None:
    """Nothing has happened on it, so its last activity is the event that revealed it.

    Asserted rather than assumed because the alternative is a NULL: the aggregate runs
    over the events carrying this identifier, and if the first of them were excluded
    from it a thread with exactly one event would have no activity time at all.

    A second thread speaks afterwards, so the aggregate has other rows to be wrong
    about -- keyed on the Session rather than on the thread it would report the later
    thread's instant for both.
    """
    log = _log(engine)
    quiet, busy, turn_id = _id(), _id(), _id()
    only = await _turn_opened(log, quiet, turn_id)
    await _spoke(log, busy, turn_id)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.thread_id for one in listed] == [quiet, busy]
    assert listed[0].updated_at_ms == listed[0].created_at_ms == _ms(only.at)


async def test_a_thread_begins_at_its_earliest_row_not_its_smallest_instant(
    engine: AsyncEngine,
) -> None:
    """When the two clocks disagree, the sequence decides where the thread began.

    They can disagree. `appended_at` defaults to `now()`, which is the *transaction's*
    start time, and appends to one Session serialize on an advisory lock -- so a writer
    that opened its transaction first and then waited for the lock commits a later
    sequence carrying an earlier timestamp. The second event here is written that way.

    The sequence is the order this record publishes and pages on, so the birth instant
    has to belong to the row that sequence names. Taking the smallest timestamp in the
    group instead reads a thread as older than the event it actually began with, and
    nothing else in this file can tell the two apart -- every other case writes its
    clock and its sequence in step, which is what makes them agree by accident.
    """
    log = _log(engine)
    child, turn_id = _id(), _id()
    first = await _turn_opened(log, child, turn_id)
    skewed = await log.add(
        turn.TURN_MESSAGE_DELTA,
        {"turn_id": turn_id, "thread_id": child, "text": "late lock, early clock"},
        at=first.at - timedelta(seconds=30),
    )
    assert skewed.seq > first.seq and skewed.at < first.at

    listed = await _index(engine).threads_of(log.session_id)

    assert listed[0].started_seq == first.seq
    assert listed[0].created_at_ms == _ms(first.at), (
        "the thread was dated from the smallest timestamp rather than from the row its "
        "own sequence names"
    )
    assert listed[0].updated_at_ms == _ms(first.at)


async def test_a_named_thread_answers_beyond_the_first_page(
    engine: AsyncEngine,
) -> None:
    """A thread the caller can name is readable wherever it sits in the order.

    The whole reason `thread_at` is a second statement rather than a filter over a page:
    the thread asked for here is deliberately past the first page's limit, so an
    implementation that paged and then searched the page would answer None for a thread
    that exists.
    """
    log = _log(engine)
    turn_id = _id()
    ids = [_id() for _ in range(5)]
    written = {}
    for one in ids:
        written[one] = await _turn_opened(log, one, turn_id)
    index = _index(engine)
    assert len(await index.threads_of(log.session_id, limit=2)) == 2

    found = await index.thread_at(log.session_id, ids[-1])

    assert found is not None, "a thread off the first page was reported as absent"
    assert found.thread_id == ids[-1]
    assert found.was_announced is False
    assert found.started_seq == written[ids[-1]].seq
    assert found.created_at_ms == _ms(written[ids[-1]].at)


async def test_a_named_unannounced_thread_answers_like_any_other(
    engine: AsyncEngine,
) -> None:
    """Naming a thread nobody announced reads it back, beside one that was announced.

    Its own case because the two reads are separate statements: a listing that admits
    unannounced threads while the single read still requires an announcement hands a
    caller an identifier it cannot then use, which is a 404 on a thread the same API
    just published.
    """
    log = _log(engine)
    root, child, turn_id = _id(), _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    opened = await _turn_opened(log, child, turn_id)

    found = await _index(engine).thread_at(log.session_id, child)

    assert found is not None, "an unannounced thread was unreadable by name"
    assert found.thread_id == child
    assert found.was_announced is False
    assert found.parent_thread_id is None
    assert found.created_at_ms == _ms(opened.at)


async def test_an_unknown_thread_is_none(engine: AsyncEngine) -> None:
    """None rather than an exception: the caller turns it into the same refusal an
    unreadable Session gets and has nothing else to do with the distinction."""
    log = _log(engine)
    await _turn_opened(log, _id(), _id())

    assert await _index(engine).thread_at(log.session_id, _id()) is None


async def test_one_sessions_listing_does_not_see_anothers_threads(
    engine: AsyncEngine,
) -> None:
    """The Session predicate is the only thing scoping this read, so it is the one to
    hold down hardest.

    The port applies no tenant predicate and says so: the log carries no tenant column,
    so every caller has already established that its own caller may address this Session
    before it arrives here. That makes a leak across Sessions a leak across tenants, and
    it would not be visible as one -- both listings would simply hold one thread too
    many.

    Both Sessions are seeded identically, from the same clock, so the two threads differ
    in nothing except which Session they belong to. Anything that answers from the
    events rather than from the Session gets both.
    """
    mine, theirs = _log(engine), _log(engine)
    my_thread, their_thread = _id(), _id()
    await _turn_opened(mine, my_thread, _id())
    await _turn_opened(theirs, their_thread, _id())
    index = _index(engine)

    assert [one.thread_id for one in await index.threads_of(mine.session_id)] == [
        my_thread
    ]
    assert [one.thread_id for one in await index.threads_of(theirs.session_id)] == [
        their_thread
    ]


async def test_another_sessions_thread_is_not_readable_by_name(
    engine: AsyncEngine,
) -> None:
    """Naming a thread does not reach across Sessions either.

    The listing and the single read are separate statements, so the predicate has to be
    right in both -- and this is the one an attacker would reach for, because it needs
    no listing to succeed first.
    """
    mine, theirs = _log(engine), _log(engine)
    their_thread = _id()
    await _turn_opened(mine, _id(), _id())
    await _turn_opened(theirs, their_thread, _id())

    assert await _index(engine).thread_at(mine.session_id, their_thread) is None


async def test_a_thread_id_reused_in_another_session_stays_on_its_own_side(
    engine: AsyncEngine,
) -> None:
    """Two Sessions, one identifier, and none of the derived facts cross over.

    Production cannot produce this collision -- a thread id is a v5 UUID of the Session
    and the runtime's own string -- and that is exactly why it is worth arranging. Four
    of the five reads behind a thread's facts are correlated on the identifier and not
    on the sequence, so a Session predicate missing from any one of them is invisible
    for as long as every case invents a fresh id: the wrong rows simply do not exist.
    Here they do, and the predicate is the only thing keeping them apart.

    The other Session is given the busier log on purpose. Its thread is announced with a
    parent, speaks, finishes its Turn and is retired, so each fact has a wrong answer
    available: a parent pointer where there is none, an announcement that never
    happened, an activity time later than this thread's, an archive instant, and a
    completion that would close a Turn still in flight.
    """
    mine, theirs = _log(engine), _log(engine)
    shared_thread, shared_turn = _id(), _id()
    opening = await _turn_opened(mine, shared_thread, shared_turn)
    await _announced(theirs, shared_thread, turn_id=shared_turn, parent=_id())
    await _spoke(theirs, shared_thread, shared_turn)
    await _completed(theirs, shared_turn)
    await _archived(theirs, shared_thread)

    listed = await _index(engine).threads_of(mine.session_id)

    assert [one.thread_id for one in listed] == [shared_thread]
    assert listed[0].was_announced is False, (
        "another Session's announcement of the same identifier was read as this one's"
    )
    assert listed[0].parent_thread_id is None, (
        "another Session's parent pointer was published on this thread"
    )
    assert listed[0].archived_at_ms is None, (
        "another Session's retirement of the same identifier was published here"
    )
    assert listed[0].turn_ended is False, (
        "another Session's completion closed a Turn that is still in flight here"
    )
    assert listed[0].updated_at_ms == _ms(opening.at), (
        "another Session's activity moved this thread's last-activity time"
    )


async def test_a_session_with_no_events_lists_nothing(engine: AsyncEngine) -> None:
    """Empty rather than an error: a Session whose events predate thread attribution
    genuinely has no threads, so a caller already has to handle an empty listing."""
    assert await _index(engine).threads_of(new_session_id()) == []


async def test_events_carrying_no_thread_do_not_invent_one(
    engine: AsyncEngine,
) -> None:
    """A Session that only ever ran unattributed Turns has no threads.

    `turn.completed` carries no `thread_id` and neither do the lifecycle events, so
    grouping on the extraction without excluding its nulls would collect all of them
    into one group and publish a thread whose identifier was NULL.
    """
    log = _log(engine)
    turn_id = _id()
    await log.add(turn.TURN_SUBMITTED, {"turn_id": turn_id, "idempotency_key": "k"})
    await _completed(log, turn_id)

    assert await _index(engine).threads_of(log.session_id) == []


@pytest.mark.parametrize("archived_first", [True, False])
async def test_an_archived_thread_is_still_listed(
    engine: AsyncEngine, archived_first: bool
) -> None:
    """Retirement is a published field, not a disappearance.

    Both orders are checked because the archive can arrive either side of the Turn's
    completion: the route refuses a thread whose Turn is open, but a Session that is
    resumed can take a further Turn after a thread was archived.
    """
    log = _log(engine)
    root, turn_id = _id(), _id()
    await _announced(log, root, turn_id=turn_id)
    if archived_first:
        await _archived(log, root)
        await _completed(log, turn_id)
    else:
        await _completed(log, turn_id)
        await _archived(log, root)

    listed = await _index(engine).threads_of(log.session_id)

    assert [one.thread_id for one in listed] == [root]
    assert listed[0].archived_at_ms is not None
