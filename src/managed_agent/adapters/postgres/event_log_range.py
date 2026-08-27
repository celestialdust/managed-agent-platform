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
from collections.abc import AsyncIterator, Collection, Sequence
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
        "SELECT s.tenant_id, e.session_id, e.seq, e.type"
        " FROM event_log e JOIN session s ON s.id = e.session_id"
        " WHERE e.type = ANY(:types)"
        " AND e.appended_at >  to_timestamp(:from_ms / 1000.0)"
        " AND e.appended_at <= to_timestamp(:to_ms / 1000.0)"
        " ORDER BY e.appended_at, e.session_id, e.seq"
    )
    .bindparams(sa.bindparam("types", type_=sa.ARRAY(sa.Text())))
    .columns(
        tenant_id=sa.Uuid(), session_id=sa.Uuid(), seq=sa.BigInteger(), type=sa.Text()
    )
)


# What the abandoned-Turn sweep needs of one Session, in two reads that do not grow
# with how long its Turns have run.
#
# The sweep asks three questions of an open Turn -- did its pod ever answer, what its
# newest progress report said, and where in the log that report sits. It used to reach
# them by paging the Session's whole log, which made judging a Turn cost a row for every
# token-level delta that Turn had ever emitted. Measured on Postgres 17: 40 Sessions of
# 3,000 events cost one sweep 782.6 ms over 120,000 rows, against 233.6 ms over 24,000
# for the same Sessions at 600 events -- linear in log size, on a pass that runs every
# 30 seconds while every live Turn appends a report every 30 seconds.
#
# Both statements filter on `session_id` with an equality and order by `seq`, so each
# walks the (session_id, seq) primary key rather than sorting anything.
_BOUNDARIES_OF = (
    sa.text(
        "SELECT seq, type, payload FROM event_log"
        " WHERE session_id = :sid AND type = ANY(:types)"
        " ORDER BY seq"
    )
    .bindparams(
        sa.bindparam("sid", type_=sa.Uuid()),
        sa.bindparam("types", type_=sa.ARRAY(sa.Text())),
    )
    .columns(seq=sa.BigInteger(), type=sa.Text(), payload=sa.JSON())
)

# Deliberately without a LIMIT, and the asymmetry with the read below is the point. This
# one is bounded by how many Turns a Session has ever run, which is small and grows only
# when a tenant submits; the other would be bounded by how much those Turns said, which
# is what the sweep must stop paying for. Capping this one would instead risk truncating
# the boundary set that `open_turn` folds, and a fold missing its own submission answers
# "nothing is open" -- the one wrong answer that leaves a wedged Session unswept.
_LATEST_PROGRESS_OF = (
    sa.text(
        "SELECT seq, type, payload FROM event_log"
        " WHERE session_id = :sid AND type = :type"
        " AND payload ->> 'turn_id' = :turn"
        " ORDER BY seq DESC LIMIT 1"
    )
    .bindparams(sa.bindparam("sid", type_=sa.Uuid()))
    .columns(seq=sa.BigInteger(), type=sa.Text(), payload=sa.JSON())
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

    Deliberately not a `Row`: it carries no payload, because the caller does not read
    one -- a callback carries a sequence rather than content, so a payload here would be
    a second thing to keep in step with the log a receiver reads anyway. The tenant is
    present and is the reason this row type exists at all, since `Row` has nowhere to
    put one.

    The type is present because it is what the caller matches registrations against, and
    it is the event's own rather than anything derived: reconstructing it -- from the
    Session's state, say -- would be a second answer to a question the row already
    holds, free to disagree with it on the one surface where disagreeing means a
    callback naming something that did not happen.
    """

    tenant_id: TenantId
    session_id: SessionId
    seq: Seq
    type: str


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

    async def turn_boundaries_of(
        self, session_id: SessionId, types: Collection[str]
    ) -> Sequence[Row]:
        """This Session's events of these types, in sequence order.

        The sweep's substitute for reading the whole log. It hands back only the events
        that open, start and close a Turn, which is exactly what `open_turn` and
        `started` fold -- both ignore every other type, so restricting the read changes
        neither answer while removing the running commentary between them.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _BOUNDARIES_OF, {"sid": session_id, "types": list(types)}
            )
            return [
                Row(session_id, Seq(row.seq), row.type, dict(row.payload))
                for row in result
            ]

    async def latest_progress_of(
        self, session_id: SessionId, type_: str, turn_id: str
    ) -> Sequence[Row]:
        """This Turn's newest event of this type, as a sequence of nought or one.

        A sequence rather than a row-or-None so that the two pure readers above it --
        `latest_idle_ms` and `latest_report_seq` -- keep taking the log slice they
        already take. Handed one row they return that row's reading; handed none they
        return None, which is what "this Turn has never reported" has always meant to
        them. Neither needs to learn a second calling convention to become cheap.

        Matched on the payload's `turn_id` in SQL for the same reason those two match on
        it in Python: a Session's log holds every Turn it has ever run, and a previous
        Turn's last report can sit arbitrarily close to the end of the log.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _LATEST_PROGRESS_OF,
                {"sid": session_id, "type": type_, "turn": turn_id},
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
        self, types: Collection[str], from_ms: int, to_ms: int
    ) -> Sequence[LifecycleRow]:
        """Events of these types appended after `from_ms` and at or before `to_ms`.

        Half-open below, closed above, in epoch milliseconds. Two consecutive
        calls whose windows meet at one instant return the event at that instant
        exactly once -- neither twice nor never -- which is the property a
        watermark that only moves forward depends on.

        Every event across every Session comes back, each paired with the tenant that
        owns its Session and carrying its own type. There is no cap: the caller narrows
        by choosing the window, and a row cap here would return a prefix of one
        instant's events with no way to say where it stopped.

        `types` is a `Collection` rather than a `Sequence` because order is not read --
        it becomes an array bound to `= ANY(...)`. A caller holding a set can pass it
        without inventing one.
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
                    type=str(row.type),
                )
                for row in result
            ]
