"""Reading a span of one Session's log, and following it as it grows.

Tier 1 (testcontainers, real PostgreSQL 17). Realizes MAP-A47 (a sequence range returns
exactly that span, in order, with nothing added or omitted) and MAP-A48 (a range past
the end is an empty result, not an error).
"""

import asyncio
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.event_log_append import PostgresEventLogAppend
from managed_agent.adapters.postgres.event_log_range import (
    _FLOOR,
    PostgresEventLogRange,
    Row,
)
from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, new_session_id
from managed_agent.core.ports import EventLogRange, EventRecord


async def _seed(engine: AsyncEngine, session_id: SessionId, count: int) -> None:
    append = PostgresEventLogAppend(engine)
    for n in range(1, count + 1):
        await append.append(session_id, f"event.{n}", {"n": n})


def test_the_adapter_satisfies_the_range_port() -> None:
    assert issubclass(PostgresEventLogRange, EventLogRange)


def test_a_row_is_an_event_record() -> None:
    row = Row(SessionId(uuid.uuid4()), 1, "event.1", {"n": 1})
    assert isinstance(row, EventRecord)


async def test_a_range_returns_exactly_that_span_in_order(engine: AsyncEngine) -> None:
    session_id = new_session_id()
    await _seed(engine, session_id, 5)

    span = await PostgresEventLogRange(engine).read(session_id, 1, 3)

    assert [row.seq for row in span] == [1, 2, 3]
    assert [row.type for row in span] == ["event.1", "event.2", "event.3"]
    assert [row.payload for row in span] == [{"n": 1}, {"n": 2}, {"n": 3}]
    assert {row.session_id for row in span} == {session_id}


async def test_a_range_in_the_middle_omits_both_ends(engine: AsyncEngine) -> None:
    session_id = new_session_id()
    await _seed(engine, session_id, 5)

    span = await PostgresEventLogRange(engine).read(session_id, 2, 4)

    assert [row.seq for row in span] == [2, 3, 4]


async def test_a_range_past_the_end_is_empty_rather_than_an_error(
    engine: AsyncEngine,
) -> None:
    session_id = new_session_id()
    await _seed(engine, session_id, 3)

    assert await PostgresEventLogRange(engine).read(session_id, 10, 20) == []


async def test_a_range_on_a_session_with_no_events_is_empty(
    engine: AsyncEngine,
) -> None:
    assert await PostgresEventLogRange(engine).read(new_session_id(), 1, 5) == []


async def test_a_range_does_not_see_another_sessions_events(
    engine: AsyncEngine,
) -> None:
    mine, theirs = new_session_id(), new_session_id()
    await _seed(engine, mine, 2)
    await _seed(engine, theirs, 2)

    span = await PostgresEventLogRange(engine).read(mine, 1, 5)

    assert {row.session_id for row in span} == {mine}


async def test_follow_yields_what_is_appended_after_the_cursor(
    engine: AsyncEngine,
) -> None:
    """Events at or below the cursor are never yielded; later ones arrive as written."""
    session_id = new_session_id()
    await _seed(engine, session_id, 2)
    log = PostgresEventLogRange(engine)
    append = PostgresEventLogAppend(engine)
    seen: list[int] = []

    async def drain() -> None:
        async for row in log.follow(session_id, 2):
            seen.append(row.seq)

    follower = asyncio.create_task(drain())
    try:
        await asyncio.sleep(0.4)
        assert seen == [], "events at or below the cursor must not be replayed"

        await append.append(session_id, "event.3", {"n": 3})
        await append.append(session_id, "event.4", {"n": 4})
        for _ in range(40):
            if len(seen) >= 2:
                break
            await asyncio.sleep(0.1)
    finally:
        follower.cancel()
        await asyncio.gather(follower, return_exceptions=True)

    assert seen == [3, 4]


async def test_the_retained_floor_of_an_untouched_log_is_one(
    engine: AsyncEngine,
) -> None:
    """Nothing has expired, so the floor is the first sequence a Session can hold."""
    log = PostgresEventLogRange(engine)
    assert await log.retained_floor(new_session_id()) == 1

    session_id = new_session_id()
    await _seed(engine, session_id, 3)
    assert await log.retained_floor(session_id) == 1


@pytest.mark.asyncio
async def test_a_swept_log_is_distinguishable_from_one_never_written(
    engine: AsyncEngine,
) -> None:
    """`retained_floor` tells expiry from emptiness — which is the only thing it is for.

    The port states the property: a read below the floor is a distinct refusal rather
    than an empty range, so a caller can tell "your position expired" from "nothing is
    here". Deriving the floor from surviving rows alone cannot do that: sweep every row
    and `min(seq)` is null, so the answer collapses onto the never-written answer and
    the two cases become indistinguishable at exactly the moment the distinction
    matters.
    """
    append = PostgresEventLogAppend(engine)
    never_written = new_session_id()
    swept = new_session_id()
    for _ in range(5):
        await append.append(swept, "marker", {})

    read = PostgresEventLogRange(engine)
    assert await read.retained_floor(never_written) == 1

    async with engine.begin() as conn:
        await conn.execute(
            sa.text("DELETE FROM event_log WHERE session_id = :sid").bindparams(
                sa.bindparam("sid", type_=sa.Uuid())
            ),
            {"sid": swept},
        )

    floor_after_full_sweep = await read.retained_floor(swept)
    assert floor_after_full_sweep != 1, (
        "a Session whose whole log was swept reports the same floor as one that was "
        "never written, so a caller cannot tell an expired position from an empty log"
    )
    assert floor_after_full_sweep == 6, (
        f"floor after sweeping seq 1-5 should be 6, the position the log reached; "
        f"got {floor_after_full_sweep}"
    )


@pytest.mark.asyncio
async def test_a_swept_log_does_not_reissue_sequences_it_already_handed_out(
    engine: AsyncEngine,
) -> None:
    """The sequence never rewinds, because consumers have already been given it.

    §6 q29 makes `seq` the SSE event id. Deriving the next one from `max(seq)` over
    surviving rows means a full retention sweep restarts the log at 1 — so a consumer
    resuming at "last event id 3" is handed a *different* event 3. Not a gap, which the
    primary key catches loudly, but a duplicate identity across time, which nothing
    catches at all.
    """
    append = PostgresEventLogAppend(engine)
    session_id = new_session_id()
    first_pass = [await append.append(session_id, "marker", {}) for _ in range(5)]
    assert first_pass == [1, 2, 3, 4, 5]

    async with engine.begin() as conn:
        await conn.execute(
            sa.text("DELETE FROM event_log WHERE session_id = :sid").bindparams(
                sa.bindparam("sid", type_=sa.Uuid())
            ),
            {"sid": session_id},
        )

    after_sweep = await append.append(session_id, "marker", {})
    assert after_sweep == 6, (
        f"the sequence restarted at {after_sweep} after a full sweep, reusing an "
        "event id already delivered to a consumer as an SSE id"
    )


@pytest.mark.asyncio
async def test_an_inverted_range_is_refused_rather_than_answered(
    engine: AsyncEngine,
) -> None:
    """`start > end` raises instead of returning an empty sequence.

    Returning `[]` makes a caller-error indistinguishable from "you have read to the end
    of the log", so a paging loop stops early and silently. Both ends arrive from
    outside once MAP-7 puts them on a route.
    """
    read = PostgresEventLogRange(engine)
    session_id = new_session_id()
    with pytest.raises(ValueError, match="inverted range"):
        await read.read(session_id, Seq(3), Seq(1))


@pytest.mark.asyncio
async def test_a_start_below_the_first_sequence_is_refused(engine: AsyncEngine) -> None:
    """`start < 1` raises rather than being silently clamped.

    `Seq` is a pydantic annotation, so `Seq(0)` is just `int(0)` and constrains nothing
    at runtime — the alias documents an intent that only takes effect where a pydantic
    model validates it. Until MAP-7's route does, this is the boundary, and 0 or a
    negative start reaching the SQL as-is returned rows as though the range were valid.
    """
    read = PostgresEventLogRange(engine)
    session_id = new_session_id()
    for bad in (0, -9):
        with pytest.raises(ValueError, match="below the first sequence"):
            await read.read(session_id, Seq(bad), Seq(3))


@pytest.mark.asyncio
async def test_a_read_returns_at_most_one_batch_and_follow_still_drains(
    engine: AsyncEngine,
) -> None:
    """`read` caps its result, and `follow` keeps going until the backlog is gone.

    The cap is what stops a reconnect with `after=0` on a long-lived Session from
    materialising the whole log into one list before yielding anything. `follow`
    therefore has to loop on full batches rather than sleep between them, or a follower
    catching up would crawl through a backlog one poll interval per batch.
    """
    append = PostgresEventLogAppend(engine)
    read = PostgresEventLogRange(engine)
    session_id = new_session_id()
    total = 12
    for _ in range(total):
        await append.append(session_id, "marker", {})

    capped = await read.read(session_id, Seq(1), Seq(total), limit=5)
    assert [row.seq for row in capped] == [1, 2, 3, 4, 5], (
        "read ignored its limit, so a caller cannot page and one request can "
        "return the whole log"
    )

    drained: list[int] = []
    async for row in read.follow(session_id, Seq(0)):
        drained.append(row.seq)
        if len(drained) == total:
            break
    assert drained == list(range(1, total + 1))


async def test_the_floor_read_is_an_index_lookup_rather_than_a_scan(
    engine: AsyncEngine,
) -> None:
    """The retention sweep depends on this read being cheap, not just correct.

    A sweep decides what to expire by asking where the floor is, so this query runs on
    every sweep of every Session. A sequential scan here would make the sweep's cost
    track the whole table rather than the one Session it is trimming — getting slower
    exactly as the thing it exists to trim grows. Asserted on the plan rather than on a
    timing, because a timing passes on a small fixture whatever the plan says.

    **Both** tables are seeded and analysed first, because on a table small enough a
    sequential scan really is the cheaper plan and PostgreSQL is right to choose it.
    Asserting against an un-analysed near-empty table would grade the fixture's size
    rather than the query's shape.

    `session_seq` is seeded here rather than left to whatever earlier tests in this
    container happened to write. The high-water-mark term reads it on every floor read,
    so an empty `session_seq` makes the plan for that term depend on test ordering --
    which is the difference between a test that grades the query and a test that passes
    or fails depending on who ran before it.

    Twenty thousand rows and not a token few. Measured: at four hundred rows PostgreSQL
    picks a sequential scan on `session_seq` and is right to, because the whole table is
    three pages. A seed too small to beat therefore asserts nothing about the index and
    would fail here for a reason that is not a defect.

    `EXPLAIN` runs over `_FLOOR.text` rather than over a copy of the SQL, so what is
    graded is the statement the adapter executes and the two cannot drift apart.
    """
    session_id = new_session_id()
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO event_log (session_id, seq, type, payload)"
                " SELECT :sid, g, 'marker', '{}'"
                " FROM generate_series(1, 400) g"
            ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
            {"sid": session_id},
        )
        await conn.execute(
            sa.text(
                "INSERT INTO session_seq (session_id, next_seq)"
                " SELECT gen_random_uuid(), 2 FROM generate_series(1, 20000)"
                " ON CONFLICT DO NOTHING"
            )
        )
        await conn.execute(
            sa.text(
                "INSERT INTO session_seq (session_id, next_seq) VALUES (:sid, 401)"
                " ON CONFLICT (session_id) DO UPDATE SET next_seq = 401"
            ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
            {"sid": session_id},
        )
        await conn.execute(sa.text("ANALYZE event_log, session_seq"))
        plan = "\n".join(
            line
            for (line,) in await conn.execute(
                sa.text(f"EXPLAIN {_FLOOR.text}").bindparams(
                    sa.bindparam("sid", type_=sa.Uuid())
                ),
                {"sid": session_id, "first": FIRST_SEQ},
            )
        )

    assert plan.strip(), "EXPLAIN returned nothing, so the assertions below say nothing"
    assert "Seq Scan" not in plan, (
        "the floor read scans a table sequentially, so its cost grows with the whole "
        f"log rather than with the one Session being swept:\n{plan}"
    )
    assert "event_log_pkey" in plan, (
        "the oldest-survivor term does not reach event_log through the "
        f"(session_id, seq) primary key:\n{plan}"
    )
    assert "session_seq_pkey" in plan, (
        "the high-water-mark term does not reach session_seq through its primary key, "
        f"and it runs on every floor read:\n{plan}"
    )
    assert plan.count("Index Cond: (session_id = ") == 2, (
        "both terms of the coalesce should be keyed on the Session; a term with no "
        f"index condition is reading rows belonging to other Sessions:\n{plan}"
    )
