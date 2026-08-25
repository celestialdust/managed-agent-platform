"""The append path hands out a contiguous sequence, and says so when it cannot.

Tier 1 (testcontainers, real PostgreSQL 17). Realizes MAP-A44. The concurrency case is
the reason this tier is a real database rather than a fake: contiguity under a single
writer is arithmetic, and contiguity under sixteen simultaneous writers is a property of
the lock and the primary key. A single-threaded loop would pass on an implementation
that has neither.
"""

import asyncio

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.event_log_append import PostgresEventLogAppend
from managed_agent.core.ids import SessionId, new_session_id
from managed_agent.core.ports import EventLogAppend, SequenceRace

_SEQS = sa.text(
    "SELECT seq FROM event_log WHERE session_id = :sid ORDER BY seq"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

_RAW_INSERT = sa.text(
    "INSERT INTO event_log (session_id, seq, type, payload)"
    " VALUES (:sid, :seq, :type, :payload)"
).bindparams(
    sa.bindparam("sid", type_=sa.Uuid()),
    sa.bindparam("payload", type_=sa.JSON()),
)


async def _stored_seqs(engine: AsyncEngine, session_id: SessionId) -> list[int]:
    async with engine.connect() as conn:
        result = await conn.execute(_SEQS, {"sid": session_id})
        return [row.seq for row in result]


def test_the_adapter_satisfies_the_append_port() -> None:
    assert issubclass(PostgresEventLogAppend, EventLogAppend)


async def test_two_appends_to_one_session_are_one_then_two(
    engine: AsyncEngine,
) -> None:
    log = PostgresEventLogAppend(engine)
    session_id = new_session_id()

    first = await log.append(session_id, "turn.started", {"turn": 1})
    second = await log.append(session_id, "turn.completed", {"turn": 1})

    assert (first, second) == (1, 2)
    assert await _stored_seqs(engine, session_id) == [1, 2]


async def test_sixteen_concurrent_appends_are_gapless_and_unique(
    engine: AsyncEngine,
) -> None:
    """Sixteen writers, one Session, one contiguous run from 1 with nothing missing."""
    log = PostgresEventLogAppend(engine)
    session_id = new_session_id()

    returned = await asyncio.gather(
        *(log.append(session_id, "tool.called", {"n": n}) for n in range(16))
    )

    assert sorted(returned) == list(range(1, 17))
    assert await _stored_seqs(engine, session_id) == list(range(1, 17))


async def test_two_sessions_number_independently(engine: AsyncEngine) -> None:
    """Interleaved appends: each Session starts at 1 and neither counts the other's."""
    log = PostgresEventLogAppend(engine)
    first, second = new_session_id(), new_session_id()

    first_one = await log.append(first, "turn.started", {})
    second_one = await log.append(second, "turn.started", {})
    first_two = await log.append(first, "turn.completed", {})
    second_two = await log.append(second, "turn.completed", {})

    assert (first_one, first_two) == (1, 2)
    assert (second_one, second_two) == (1, 2)
    assert await _stored_seqs(engine, first) == [1, 2]
    assert await _stored_seqs(engine, second) == [1, 2]


async def test_two_sessions_appending_at_once_do_not_share_a_run(
    engine: AsyncEngine,
) -> None:
    """Concurrently, too: a lock held for one Session must not number the other."""
    log = PostgresEventLogAppend(engine)
    first, second = new_session_id(), new_session_id()

    await asyncio.gather(
        *(log.append(first, "tool.called", {"n": n}) for n in range(8)),
        *(log.append(second, "tool.called", {"n": n}) for n in range(8)),
    )

    assert await _stored_seqs(engine, first) == list(range(1, 9))
    assert await _stored_seqs(engine, second) == list(range(1, 9))


async def test_a_writer_that_took_the_sequence_first_makes_the_append_raise(
    engine: AsyncEngine,
) -> None:
    """A writer that ignores the lock is the one case the primary key has to catch.

    The competing INSERT is held open in an uncommitted transaction, so the adapter's
    `max(seq)` cannot see it and picks the same sequence. Its INSERT then blocks on the
    primary key until the competitor commits, and the violation surfaces as SequenceRace
    rather than as a second row at the same sequence.
    """
    log = PostgresEventLogAppend(engine)
    session_id = new_session_id()

    competitor = await engine.connect()
    try:
        transaction = await competitor.begin()
        await competitor.execute(
            _RAW_INSERT,
            {"sid": session_id, "seq": 1, "type": "raw.write", "payload": {}},
        )

        racing = asyncio.create_task(log.append(session_id, "turn.started", {}))
        await asyncio.sleep(0.3)
        await transaction.commit()

        with pytest.raises(SequenceRace):
            await racing
    finally:
        await competitor.close()

    assert await _stored_seqs(engine, session_id) == [1]


async def test_a_retry_after_a_race_lands_on_the_next_sequence(
    engine: AsyncEngine,
) -> None:
    """The caller retries and gets 2 — the contract the port's docstring states."""
    log = PostgresEventLogAppend(engine)
    session_id = new_session_id()

    async with engine.begin() as conn:
        await conn.execute(
            _RAW_INSERT,
            {"sid": session_id, "seq": 1, "type": "raw.write", "payload": {}},
        )

    assert await log.append(session_id, "turn.started", {}) == 2


@pytest.mark.asyncio
async def test_an_integrity_error_that_is_not_a_race_is_not_renamed_into_one(
    engine: AsyncEngine,
) -> None:
    """A non-uniqueness integrity failure propagates unchanged.

    The port tells the caller to retry a SequenceRace. Reporting every IntegrityError as
    one turns a permanent failure into an infinite loop: nothing about a foreign key to
    a Session that does not exist, or a violated check, becomes true on a second
    attempt, and no sequence was ever contested. Provoked with a real check constraint
    rather than a patched driver, because what is under test is which SQLSTATE the
    adapter treats as retryable, and only the database assigns those.
    """
    append = PostgresEventLogAppend(engine)
    session_id = new_session_id()
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "ALTER TABLE event_log ADD CONSTRAINT tmp_no_forbidden_type"
                " CHECK (type <> 'forbidden')"
            )
        )
    try:
        with pytest.raises(IntegrityError) as caught:
            await append.append(session_id, "forbidden", {})
        assert not isinstance(caught.value, SequenceRace), (
            "a check-constraint violation was reported as a retryable SequenceRace; "
            "the caller the port instructs to retry would loop forever"
        )
        assert getattr(caught.value.orig, "sqlstate", None) == "23514"
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text("ALTER TABLE event_log DROP CONSTRAINT tmp_no_forbidden_type")
            )


@pytest.mark.asyncio
async def test_the_lock_is_per_session_and_not_one_mutex_for_the_platform(
    engine: AsyncEngine,
) -> None:
    """Holding Session A's lock does not delay an append to Session B.

    `test_two_sessions_number_independently` reads as though it graded this and does
    not: a single global lock also keeps two Sessions' numbering separate, so replacing
    `hashtextextended(session_id)` with a constant passes it, and passes the sixteen-way
    concurrency test too. What that collapse would actually cost is the whole platform's
    append path serialising onto one mutex — a change that would ship green.

    Asserted by contention rather than by outcome: the lock for A is taken and held in
    an open transaction, and an append to B has to finish anyway. The bound is generous
    because the point is the difference between "proceeds" and "blocks until the holder
    commits", not a latency budget.
    """
    append = PostgresEventLogAppend(engine)
    session_a = new_session_id()
    session_b = new_session_id()

    holder = sa.text(
        "SELECT pg_advisory_xact_lock(hashtextextended(cast(:sid AS text), 0))"
    ).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

    async with engine.begin() as held:
        await held.execute(holder, {"sid": session_a})
        # A's lock is now held by a transaction that has not committed.
        seq_b = await asyncio.wait_for(
            append.append(session_b, "marker", {}), timeout=5.0
        )
        assert seq_b == 1

        with pytest.raises(TimeoutError):
            # And the control: an append to A itself must block, or the lock is not
            # doing anything at all and the test above proves nothing.
            await asyncio.wait_for(append.append(session_a, "marker", {}), timeout=1.0)
