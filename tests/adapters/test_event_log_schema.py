"""The store, not the application, is what refuses an illegal Event Log row.

Tier 1 (testcontainers, real PostgreSQL 17). Realizes MAP-A45: an earlier event cannot
be changed, and an attempt is refused rather than silently ignored. Each assertion here
is made with raw SQL rather than through the adapter, because the point is that the
guarantee survives a writer that never loads our code.
"""

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

_INSERT = sa.text(
    "INSERT INTO event_log (session_id, seq, type, payload)"
    " VALUES (:sid, :seq, :type, :payload)"
).bindparams(
    sa.bindparam("sid", type_=sa.Uuid()),
    sa.bindparam("payload", type_=sa.JSON()),
)


async def _write(engine: AsyncEngine, sid: uuid.UUID, seq: int, type_: str) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            _INSERT, {"sid": sid, "seq": seq, "type": type_, "payload": {"n": seq}}
        )


async def test_an_update_against_a_written_row_is_refused_and_changes_nothing(
    engine: AsyncEngine,
) -> None:
    """Both halves: the earlier event is unchanged, and the writer is told so."""
    sid = uuid.uuid4()
    await _write(engine, sid, 1, "turn.started")

    with pytest.raises(IntegrityError) as caught:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE event_log SET type = 'tampered' WHERE session_id = :sid"
                ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
                {"sid": sid},
            )
    assert "append-only" in str(caught.value)

    async with engine.connect() as conn:
        after = (
            await conn.execute(
                sa.text(
                    "SELECT type FROM event_log WHERE session_id = :sid"
                ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
                {"sid": sid},
            )
        ).scalar_one()
    assert after == "turn.started"


async def test_sequence_zero_violates_the_check_constraint(engine: AsyncEngine) -> None:
    with pytest.raises(IntegrityError) as caught:
        await _write(engine, uuid.uuid4(), 0, "turn.started")
    assert "event_log_seq_from_one" in str(caught.value)


async def test_a_duplicate_sequence_violates_the_primary_key(
    engine: AsyncEngine,
) -> None:
    sid = uuid.uuid4()
    await _write(engine, sid, 1, "turn.started")
    with pytest.raises(IntegrityError):
        await _write(engine, sid, 1, "turn.started")


async def test_the_same_sequence_under_two_sessions_is_two_rows(
    engine: AsyncEngine,
) -> None:
    """The key is the pair, so two Sessions both hold seq 1 without colliding."""
    first, second = uuid.uuid4(), uuid.uuid4()
    await _write(engine, first, 1, "turn.started")
    await _write(engine, second, 1, "turn.started")

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                sa.text(
                    "SELECT count(*) FROM event_log WHERE session_id IN (:a, :b)"
                ).bindparams(
                    sa.bindparam("a", type_=sa.Uuid()),
                    sa.bindparam("b", type_=sa.Uuid()),
                ),
                {"a": first, "b": second},
            )
        ).scalar_one()
    assert rows == 2


@pytest.mark.xfail(
    strict=True,
    reason="MAP-A45's 'remove' half is not enforced -- the docstring below says what "
    "it asks for and what is built instead. Strict so "
    "that whoever closes the gap is forced to delete this marker.",
)
async def test_a_written_event_cannot_be_deleted(engine: AsyncEngine) -> None:
    """The half of MAP-A45 that is not built, asserted as the behaviour it asks for.

    MAP-A45 forbids three things -- changing, reordering and removing an earlier event
    -- and requires the attempt to be refused rather than silently ignored. The first
    two hold: `seq` is half the primary key, and the `event_log_no_update` trigger
    raises. Removal does not. Migration 0001 leaves DELETE open on purpose, because the
    retention sweep is a real operation and a table that cannot forget cannot honour a
    retention policy -- but nothing then separates the sweep from any other caller, and
    there is no GRANT, REVOKE or second role anywhere in this repository. So every
    request handler holds DELETE on every event, and the attempt succeeds silently.

    Written as a strict xfail rather than as a comment or as a passing test that asserts
    the present behaviour. A comment is invisible to the suite; a test asserting that
    DELETE succeeds would lock the gap in and read, forever, as a guarantee somebody
    intended. Strict means the day the guard lands this test fails as XPASS, and closing
    it requires deleting the marker -- so the gap cannot be fixed and left undocumented,
    and cannot be forgotten while it is open.

    `IntegrityError` and not a bare `Exception`: the refusal this asks for is the one
    the UPDATE trigger already gives, raised in the integrity-violation class. A blind
    `except` would also be satisfied by a connection error or a typo in the SQL, which
    would let this test pass for reasons that have nothing to do with MAP-A45.
    """
    sid = uuid.uuid4()
    await _write(engine, sid, 1, "turn.started")

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                sa.text("DELETE FROM event_log WHERE session_id = :sid").bindparams(
                    sa.bindparam("sid", type_=sa.Uuid())
                ),
                {"sid": sid},
            )
