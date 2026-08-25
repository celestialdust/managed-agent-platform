"""The placement backlog read against real PostgreSQL, including its time bound.

Tier 1 (testcontainers, real PostgreSQL 17), and the tier is the point rather than a
habit. The query is one statement with a correlated `NOT EXISTS`, a JSONB extraction and
`make_interval` -- none of which a fake can represent, and all three of which are where
it would be wrong. The property under test is the reason it exists: the answer is the
same from any process, because it is a property of the log.

`appended_at` is a server clock read with a `now()` default, so a test that wants an OLD
wait cannot get one by waiting. It writes the row and then moves its `appended_at`
backwards -- which `event_log`'s append-only trigger refuses on an UPDATE, so the row is
deleted and re-inserted with the time it needs. That is the one operation migration 0001
deliberately leaves open, and it is how a retention sweep works too.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.event_log_append import PostgresEventLogAppend
from managed_agent.adapters.postgres.placement_backlog import (
    PostgresPlacementBacklog,
)
from managed_agent.control.session.placement import PlacementBacklogReader
from managed_agent.core.ids import SessionId, new_session_id, new_turn_id
from managed_agent.core.vocabulary import placement, turn

_WINDOW_S = 15 * 60


@pytest.fixture(autouse=True)
async def an_empty_log(engine: AsyncEngine) -> AsyncIterator[None]:
    """Start every case here from an empty `event_log`, and leave it empty.

    **Not tidiness -- correctness.** This gauge is deliberately fleet-wide: it has no
    tenant term and no Session term, because the question it answers is "is the platform
    behind", which is not a question about anybody's Session. That makes every other
    test's rows part of this one's subject. Without this, the first case here leaves a
    waiting Turn behind and every `== 0` assertion after it reads that row instead of
    its own arrangement.

    A DELETE rather than a TRUNCATE, because `event_log` has an append-only trigger on
    UPDATE and nothing against DELETE -- which is also how a retention sweep clears it.

    Cleared on the way out as well as in, so a case here cannot leave a waiting Turn for
    a test in another file that reads this gauge.
    """
    async with engine.begin() as connection:
        await connection.execute(sa.text("DELETE FROM event_log"))
        await connection.execute(sa.text("DELETE FROM session_seq"))
    yield
    async with engine.begin() as connection:
        await connection.execute(sa.text("DELETE FROM event_log"))
        await connection.execute(sa.text("DELETE FROM session_seq"))


def test_the_adapter_satisfies_the_backlog_port() -> None:
    """Graded at run time, because `capacity()` narrows to this protocol.

    `issubclass` and not `isinstance`, so no instance has to be built for a question
    about the class -- and `runtime_checkable` answers it on method names, which is
    exactly what the narrowing in `capacity()` asks.
    """
    assert issubclass(PostgresPlacementBacklog, PlacementBacklogReader)


async def _placing(engine: AsyncEngine, session_id: SessionId, turn_id: str) -> None:
    await PostgresEventLogAppend(engine).append(
        session_id, placement.SESSION_PLACING, {"turn_id": turn_id}
    )


async def _started(engine: AsyncEngine, session_id: SessionId, turn_id: str) -> None:
    await PostgresEventLogAppend(engine).append(
        session_id, turn.TURN_STARTED, {"turn_id": turn_id}
    )


_READ_ROW = (
    sa.text(
        "SELECT type, payload FROM event_log WHERE session_id = :sid AND seq = :seq"
    )
    .bindparams(sa.bindparam("sid", type_=sa.Uuid()))
    .columns(type=sa.Text(), payload=sa.JSON())
)

_DROP_ROW = sa.text(
    "DELETE FROM event_log WHERE session_id = :sid AND seq = :seq"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

_WRITE_ROW = (
    sa.text(
        "INSERT INTO event_log (session_id, seq, type, payload, appended_at)"
        " VALUES (:sid, :seq, :type, :payload, :when)"
    )
    .bindparams(sa.bindparam("sid", type_=sa.Uuid()))
    .bindparams(sa.bindparam("payload", type_=sa.JSON()))
    .bindparams(sa.bindparam("when", type_=sa.TIMESTAMP(timezone=True)))
)


async def _backdate(
    engine: AsyncEngine, session_id: SessionId, seq: int, seconds: int
) -> None:
    """Move one row's `appended_at` back: read it, delete it, write it again.

    Three statements and not one UPDATE, because migration 0001's trigger refuses an
    UPDATE on `event_log` outright. DELETE is the operation it deliberately leaves open,
    which is also how a retention sweep clears the table.

    Read before delete and all three in one transaction, so there is no window where the
    row is gone and its replacement is not yet written -- and the first version of this
    helper got the order wrong the other way, inserting the copy while the original was
    still there, which is a primary-key collision on `(session_id, seq)`.
    """
    when = datetime.now(UTC) - timedelta(seconds=seconds)
    key = {"sid": uuid.UUID(str(session_id)), "seq": seq}
    async with engine.begin() as connection:
        row = (await connection.execute(_READ_ROW, key)).one()
        await connection.execute(_DROP_ROW, key)
        await connection.execute(
            _WRITE_ROW,
            {**key, "type": row.type, "payload": row.payload, "when": when},
        )


async def test_a_turn_with_no_terminal_event_is_counted(engine: AsyncEngine) -> None:
    """One placing event and nothing after it: one Turn waiting, on one Session."""
    session_id = new_session_id()
    await _placing(engine, session_id, str(new_turn_id()))

    read = await PostgresPlacementBacklog(engine).placement_backlog(_WINDOW_S)
    assert read.turns_awaiting == 1
    assert read.sessions_placing == 1
    assert read.oldest_awaiting_at_ms is not None


async def test_a_started_turn_stops_being_counted(engine: AsyncEngine) -> None:
    """`turn.started` ends the wait, and nothing else has to be written for it.

    The arms-disagree half of the case above. Without this, an implementation that
    counted every `session.placing` ever written would pass that one.
    """
    session_id = new_session_id()
    turn_id = str(new_turn_id())
    await _placing(engine, session_id, turn_id)
    await _started(engine, session_id, turn_id)

    read = await PostgresPlacementBacklog(engine).placement_backlog(_WINDOW_S)
    assert read.turns_awaiting == 0
    assert read.sessions_placing == 0
    assert read.oldest_awaiting_at_ms is None


async def test_a_failed_turn_also_stops_being_counted(engine: AsyncEngine) -> None:
    """The other terminal event, asserted because there are exactly two.

    A Turn whose placement failed is not waiting -- it was answered. Counting it would
    make the gauge climb on exactly the failures an operator is looking at it to find.
    """
    session_id = new_session_id()
    turn_id = str(new_turn_id())
    await _placing(engine, session_id, turn_id)
    await PostgresEventLogAppend(engine).append(
        session_id, turn.TURN_FAILED, {"turn_id": turn_id}
    )

    read = await PostgresPlacementBacklog(engine).placement_backlog(_WINDOW_S)
    assert read.turns_awaiting == 0


async def test_one_turns_start_does_not_clear_another_turns_wait(
    engine: AsyncEngine,
) -> None:
    """Two Turns on ONE Session, one started: the other is still waiting.

    This is the case that decides the query is keyed on `turn_id` rather than on the
    Session. Two Turns of one Session can wait at once -- admission refuses a Session
    that will not take a Turn, not one that already has a Turn open -- so a
    Session-keyed match would let the first Turn's `turn.started` clear the second
    Turn's wait, and the depth would under-report exactly when it mattered.
    """
    session_id = new_session_id()
    first, second = str(new_turn_id()), str(new_turn_id())
    await _placing(engine, session_id, first)
    await _placing(engine, session_id, second)
    await _started(engine, session_id, first)

    read = await PostgresPlacementBacklog(engine).placement_backlog(_WINDOW_S)
    assert read.turns_awaiting == 1
    assert read.sessions_placing == 1, "both waits are on the same Session"


async def test_two_sessions_waiting_count_as_two_sessions(engine: AsyncEngine) -> None:
    """Turns and Sessions are counted separately, so both numbers mean something."""
    one, other = new_session_id(), new_session_id()
    await _placing(engine, one, str(new_turn_id()))
    await _placing(engine, one, str(new_turn_id()))
    await _placing(engine, other, str(new_turn_id()))

    read = await PostgresPlacementBacklog(engine).placement_backlog(_WINDOW_S)
    assert read.turns_awaiting == 3
    assert read.sessions_placing == 2


async def test_a_wait_older_than_the_window_is_not_counted(
    engine: AsyncEngine,
) -> None:
    """The time bound, which is what keeps a dead process out of the gauge forever.

    A `session.placing` with no terminal event and an age past the placement timeout is
    not a Turn still queued -- it is a Turn whose process died mid-placement, and its
    connection died with it. Without this bound the depth would climb monotonically and
    never come back down, which is the one failure that makes a gauge worth less than
    no gauge.
    """
    session_id = new_session_id()
    await _placing(engine, session_id, str(new_turn_id()))
    await _backdate(engine, session_id, seq=1, seconds=_WINDOW_S + 60)

    read = await PostgresPlacementBacklog(engine).placement_backlog(_WINDOW_S)
    assert read.turns_awaiting == 0
    assert read.oldest_awaiting_at_ms is None


async def test_the_backdating_helper_really_moves_the_row(
    engine: AsyncEngine,
) -> None:
    """The vacuity control for the case above.

    A helper that silently failed would leave the row at `now()`, the assertion would
    read zero for the wrong reason, and the time bound would be untested while looking
    tested. Asserted against the window's own edge: the row is old enough to be excluded
    by a 15-minute bound and still present under an hour's.
    """
    session_id = new_session_id()
    await _placing(engine, session_id, str(new_turn_id()))
    await _backdate(engine, session_id, seq=1, seconds=_WINDOW_S + 60)

    reader = PostgresPlacementBacklog(engine)
    assert (await reader.placement_backlog(_WINDOW_S)).turns_awaiting == 0
    assert (await reader.placement_backlog(3600)).turns_awaiting == 1


async def test_the_oldest_instant_is_the_oldest_live_wait(engine: AsyncEngine) -> None:
    """Two waits, one backdated: the reported instant is the older one's.

    An operator reads this to tell a queue that is draining from one stuck at the same
    depth, so it has to be the oldest LIVE wait rather than the oldest event -- and a
    minimum taken before the `NOT EXISTS` filter would report a wait that had already
    ended.
    """
    session_id = new_session_id()
    settled, waiting = str(new_turn_id()), str(new_turn_id())
    await _placing(engine, session_id, settled)
    await _backdate(engine, session_id, seq=1, seconds=300)
    await _placing(engine, session_id, waiting)
    await _started(engine, session_id, settled)

    read = await PostgresPlacementBacklog(engine).placement_backlog(_WINDOW_S)
    assert read.turns_awaiting == 1
    assert read.oldest_awaiting_at_ms is not None
    # The settled wait is five minutes old and excluded; the live one is seconds old.
    age_ms = int(datetime.now(UTC).timestamp() * 1000) - read.oldest_awaiting_at_ms
    assert age_ms < 120_000, (
        f"the reported oldest instant is {age_ms}ms old, which is the SETTLED wait's "
        "age -- the minimum was taken before the terminal-event filter"
    )
