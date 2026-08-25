"""Appending to the Event Log, serialized per Session.

Contiguity is the expensive half of the guarantee: it forbids gaps, so a sequence cannot
be reserved in a block and a failed write cannot silently burn one. That forces the next
sequence to be read and written inside one transaction, under a lock scoped to the one
Session — across Sessions there is no ordering relationship at all, so nothing contends.

The advisory lock is keyed on the Session's uuid rather than on a table row, so it costs
no write and disappears with the transaction even if the process dies holding it. It is
also not the whole guarantee: a writer that never took the lock is still refused,
because the primary key on `(session_id, seq)` catches it and this adapter turns that
refusal into `SequenceRace` for the caller to retry. Lock for the common path,
constraint for the truth.
"""

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.core.ids import Seq, SessionId
from managed_agent.core.ports import SequenceRace

# Bind parameter types are declared rather than inferred: these are textual statements,
# so without a type the driver receives a bare Python object and either guesses or
# refuses. `payload` in particular has to reach PostgreSQL as JSON and not as a dict
# repr.
# PostgreSQL's SQLSTATE for unique_violation. Named rather than inlined because it is
# what distinguishes a contested sequence from every other integrity failure.
_UNIQUE_VIOLATION = "23505"

# The next sequence is the higher of two facts, and it needs both. `max(seq)` over
# surviving rows is what keeps the numbering gap-free: a transaction that claims a
# sequence and then rolls back un-does its row, and the next claim recomputes from what
# is actually there rather than from a counter that already moved.
# `session_seq.next_seq`
# is what keeps it from rewinding: once a retention sweep removes the rows, `max(seq)`
# is
# null and, alone, it would restart at 1 and reissue event ids already delivered.
# Neither alone is right. Taken together under the advisory lock, the numbering is
# contiguous from 1 and never repeats.
_NEXT_SEQ = sa.text(
    "SELECT greatest("
    " coalesce((SELECT max(seq) FROM event_log WHERE session_id = :sid), 0),"
    " coalesce((SELECT next_seq FROM session_seq WHERE session_id = :sid), 1) - 1"
    ") + 1"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

# Raise the high-water mark in the same transaction as the row it describes, so the two
# roll back together. `greatest` rather than plain assignment: a concurrent path that
# somehow got further must not be walked backwards by a slower one.
_BUMP_HIGH_WATER = sa.text(
    "INSERT INTO session_seq (session_id, next_seq) VALUES (:sid, :next)"
    " ON CONFLICT (session_id) DO UPDATE"
    " SET next_seq = greatest(session_seq.next_seq, :next)"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

# hashtextextended maps the Session's uuid onto the bigint keyspace advisory locks use.
# A collision between two Sessions costs a little serialization and never correctness,
# because the sequence itself is still read and written under the primary key.
#
# The cast happens in PostgreSQL, from a uuid-typed bind, rather than in Python
# from `str(session_id)`. SessionId is a NewType and erased at runtime, so a caller
# holding a differently-cased or differently-punctuated id string would have hashed to a
# different lock key for the same row -- the serialization control silently degrading to
# the primary-key race path, which is the one failure this lock exists to prevent.
# Binding it as a uuid makes one canonical form the only form.
_LOCK = sa.text(
    "SELECT pg_advisory_xact_lock(hashtextextended(cast(:sid AS text), 0))"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

_INSERT = sa.text(
    "INSERT INTO event_log (session_id, seq, type, payload)"
    " VALUES (:sid, :seq, :type, :payload)"
).bindparams(
    sa.bindparam("sid", type_=sa.Uuid()),
    sa.bindparam("payload", type_=sa.JSON()),
)


class PostgresEventLogAppend:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        """Append one event to this Session's log and return the sequence it was given.

        The read of the next sequence and the write that claims it are one transaction,
        so no other writer can land between them. Raises SequenceRace when the sequence
        was taken anyway — by something that did not take the lock — and the caller
        retries.
        """
        async with self._engine.begin() as conn:
            await conn.execute(_LOCK, {"sid": session_id})
            seq = (await conn.execute(_NEXT_SEQ, {"sid": session_id})).scalar_one()
            try:
                await conn.execute(
                    _INSERT,
                    {
                        "sid": session_id,
                        "seq": seq,
                        "type": type_,
                        "payload": payload,
                    },
                )
            except IntegrityError as exc:
                # Only a unique violation is a lost sequence race. Every other integrity
                # error means something the caller cannot fix by trying again: a foreign
                # key to a Session that does not exist, a check constraint, a not-null.
                # The port tells the caller to retry a SequenceRace, so renaming those
                # turns a permanent failure into an infinite loop -- and the retry can
                # never succeed, because no sequence was ever contested.
                if getattr(exc.orig, "sqlstate", None) != _UNIQUE_VIOLATION:
                    raise
                raise SequenceRace(
                    f"sequence {seq} for session {session_id} was taken"
                ) from exc
            await conn.execute(_BUMP_HIGH_WATER, {"sid": session_id, "next": seq + 1})
            return Seq(seq)
