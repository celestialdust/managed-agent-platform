"""Reading a span of one Session's Event Log, and following it as it grows.

Deliberately lock-free: appending serializes per Session to keep the sequence
contiguous, but reading a span that is already written has nothing to serialize against.
Keeping the two in separate adapters is what lets a replica back this one and not the
other.

`follow` polls rather than listening on a channel. A poll cannot deliver an event the
log has not committed, which is the property resume-from-sequence rests on;
LISTEN/NOTIFY can, because a notification fires on commit of the notifying transaction
and not of the row.
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, TenantId

# The column types are declared, and under this driver they are measured to change
# nothing: deleting them breaks no test and alters no value. SQLAlchemy's asyncpg
# dialect registers its own json codec at connection setup, so `payload` is already a
# mapping before any result-type declaration is consulted. They stay as insurance
# against a driver that does not do that, and because this is a textual statement with
# no column metadata of its own -- but a reader should not believe they are what makes
# this read work today. That belief was written here first and measured second, which
# is the wrong order; MAP-5 caught it by deleting the line and watching nothing
# happen.
_READ = (
    sa.text(
        "SELECT seq, type, payload FROM event_log"
        " WHERE session_id = :sid AND seq >= :start AND seq <= :end"
        " ORDER BY seq LIMIT :limit"
    )
    .bindparams(sa.bindparam("sid", type_=sa.Uuid()))
    .columns(seq=sa.BigInteger(), type=sa.Text(), payload=sa.JSON())
)

# Three answers in priority order, and the middle one is the whole point. The oldest
# surviving row, if any rows survive. Otherwise the high-water mark, which says how far
# the numbering went before a sweep removed everything -- so an expired position reads
# as expired rather than as an empty log. Only a Session that has never been written
# reaches the third and gets FIRST_SEQ. Without the middle term the first and third
# collapse together, and telling those two apart is the only reason this method exists.
_FLOOR = sa.text(
    "SELECT coalesce("
    " (SELECT min(seq) FROM event_log WHERE session_id = :sid),"
    " (SELECT next_seq FROM session_seq WHERE session_id = :sid),"
    " :first)"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

# The same read without an upper bound, for `follow`. A separate statement rather than a
# sentinel passed into the bounded one: `read` documents both ends as inclusive, and the
# only way to satisfy that with an open end is to pass a value that is not an end.
_READ_FROM = (
    sa.text(
        "SELECT seq, type, payload FROM event_log"
        " WHERE session_id = :sid AND seq >= :start"
        " ORDER BY seq LIMIT :limit"
    )
    .bindparams(sa.bindparam("sid", type_=sa.Uuid()))
    .columns(seq=sa.BigInteger(), type=sa.Text(), payload=sa.JSON())
)

_POLL_INTERVAL_S = 0.25

# How many rows one read returns at most. `follow` reads to the end of the log each pass
# and the end has no number until it is written, so this used to be an upper *bound* of
# 2**62 -- a value that is not an end, satisfying a method documented as inclusive of
# both ends. Worse, it was unbounded in fact as well as in name: a reconnect with
# `after=0` on a 50,000-event Session materialised every row into one list before
# yielding the first, measured at 203.8 ms and ~17 MB of heap. With a cap it is 3.5 ms.
# So the cap is not a micro-optimisation, it is the difference between a paging read and
# a whole-log read that a caller can ask for by supplying one small number.
_BATCH = 500

# The cross-Session tail. A range read over the Event Log like the two above, with the
# range expressed in time and the Session left open, which is why it belongs on this
# class rather than in an adapter of its own.
#
# The join is where the tenant comes from: `event_log` carries no tenant column and must
# not grow one, because the Session record is where a Session's owner is fixed and a
# second copy could disagree with it.
#
# The window is half-open below and closed above, and that asymmetry is the whole reason
# consecutive passes cover the line between them exactly once. A repeat costs one
# refused insert against the delivery claim; nothing recovers a skip.
#
# No LIMIT on purpose. The caller bounds the pass by time -- a row cap would leave the
# tail's watermark unable to say how far it had actually read, since the rows beyond the
# cap share timestamps with the rows inside it.
_LIFECYCLE_BETWEEN = (
    sa.text(
        "SELECT s.tenant_id, e.session_id, e.seq"
        " FROM event_log e JOIN session s ON s.id = e.session_id"
        " WHERE e.type = ANY(:types)"
        " AND e.appended_at >  to_timestamp(:from_ms / 1000.0)"
        " AND e.appended_at <= to_timestamp(:to_ms / 1000.0)"
        " ORDER BY e.appended_at, e.session_id, e.seq"
    )
    .bindparams(sa.bindparam("types", type_=sa.ARRAY(sa.Text())))
    .columns(tenant_id=sa.Uuid(), session_id=sa.Uuid(), seq=sa.BigInteger())
)


@dataclass(frozen=True, slots=True)
class Row:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class LifecycleRow:
    """One lifecycle event the cross-Session tail found, with the Session's owner.

    Deliberately not a `Row`: it carries no type and no payload, because the caller does
    not read either -- it folds the Session's own log to learn what state the event
    arrived at, and a payload here would be a second thing to keep in step. The
    tenant is present and is the reason this row type exists at all, since `Row`
    has nowhere to put one.
    """

    tenant_id: TenantId
    session_id: SessionId
    seq: Seq


class PostgresEventLogRange:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = _BATCH
    ) -> Sequence[Row]:
        """Return this Session's events with `start <= seq <= end`, in sequence order.

        Both ends are inclusive, and at most `limit` rows come back — a short result
        means the range is exhausted, which is how a caller pages. A range with nothing
        in it — past the head of the log, or over a Session that has written nothing —
        is an empty sequence and not an error, because a caller paging forward reaches
        the end by reading it.

        An inverted range is refused rather than answered. `start > end` describes no
        events, so an empty sequence for it is indistinguishable from "you have read to
        the end", and a caller looping on that answer stops silently at the wrong place.
        Both ends arrive from outside — MAP-7 puts them on a route — so this is a
        boundary, and a boundary that turns bad input into a plausible answer is worse
        than one with no check at all.
        """
        if start > end:
            raise ValueError(
                f"inverted range for session {session_id}: start {start} is above "
                f"end {end}. No events lie in it, and an empty result would read as "
                "the end of the log."
            )
        if start < FIRST_SEQ:
            raise ValueError(
                f"start {start} is below the first sequence {FIRST_SEQ} for session "
                f"{session_id}. Sequences begin at {FIRST_SEQ}, so a lower start is a "
                "caller error rather than an empty range."
            )
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _READ,
                {"sid": session_id, "start": start, "end": end, "limit": limit},
            )
            return [
                Row(session_id, Seq(row.seq), row.type, dict(row.payload))
                for row in result
            ]

    async def _read_from(self, session_id: SessionId, start: Seq) -> Sequence[Row]:
        """At most `_BATCH` events at or after `start`, in sequence order.

        `follow`'s reader. It has no upper bound because the end of a live log has no
        number until it is written, which is exactly the shape `read` cannot express.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _READ_FROM, {"sid": session_id, "start": start, "limit": _BATCH}
            )
            return [
                Row(session_id, Seq(row.seq), row.type, dict(row.payload))
                for row in result
            ]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Row]:
        """Yield this Session's events as they are appended, starting after `after`.

        The cursor only ever moves forward, so an event at or below `after` is never
        yielded — that is what makes a reconnect continue rather than repeat. The
        iterator does not end on its own: a Session with no events in flight is
        indistinguishable from one that has finished, so ending is the caller's decision
        and it makes it by ceasing to iterate.
        """
        cursor = after
        while True:
            batch = await self._read_from(session_id, Seq(cursor + 1))
            for row in batch:
                cursor = row.seq
                yield row
            if len(batch) == _BATCH:
                # A full batch means there is more already written. Go straight back for
                # it rather than sleeping: the poll interval is for waiting on events
                # that do not exist yet, and a follower catching up on a backlog would
                # otherwise crawl through it one interval per batch.
                continue
            if not batch:
                await asyncio.sleep(_POLL_INTERVAL_S)

    async def retained_floor(self, session_id: SessionId) -> Seq:
        """The lowest sequence this Session still holds.

        With nothing yet expired this is 1, and it stays 1 for a Session that has never
        written — the answer being "nothing has been dropped", not "the log is empty".
        Once a retention sweep removes a prefix, this rises to the oldest survivor,
        which is what lets a caller be told its position expired rather than handed a
        later range as though it were contiguous.

        A sweep that removes *everything* is the case worth stating, because the obvious
        implementation gets it wrong: with no surviving rows there is no oldest
        survivor, and deriving the floor from the rows alone answers 1 — identical to
        the never-written answer, at exactly the moment a caller most needs the two
        distinguished. So the fallback is the Session's high-water mark rather than 1,
        and only a Session with no high-water mark at all answers 1.
        """
        async with self._engine.connect() as conn:
            floor = (
                await conn.execute(_FLOOR, {"sid": session_id, "first": FIRST_SEQ})
            ).scalar_one()
            return Seq(floor)

    async def lifecycle_events_between(
        self, types: Sequence[str], from_ms: int, to_ms: int
    ) -> Sequence[LifecycleRow]:
        """Events of these types appended after `from_ms` and at or before `to_ms`.

        Half-open below, closed above, in epoch milliseconds. Two consecutive
        calls whose windows meet at one instant return the event at that instant
        exactly once -- neither twice nor never -- which is the property a
        watermark that only moves forward depends on.

        Every event across every Session comes back, each paired with the tenant that
        owns its Session. There is no cap: the caller narrows by choosing the
        window, and a row cap here would return a prefix of one instant's events
        with no way to say where it stopped.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _LIFECYCLE_BETWEEN,
                {"types": list(types), "from_ms": from_ms, "to_ms": to_ms},
            )
            return [
                LifecycleRow(
                    tenant_id=TenantId(row.tenant_id),
                    session_id=SessionId(row.session_id),
                    seq=Seq(int(row.seq)),
                )
                for row in result
            ]
