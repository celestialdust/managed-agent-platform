"""One uploaded file's two halves: its bytes in the object store, its row in Postgres.

A missing object reads as None rather than raising, because "the key holds nothing" is a
fact the caller has to act on — it means the row and the bucket disagree — while a
botocore error type reaching control-plane code would put an infrastructure vocabulary
in a decision that is about the platform.

Nothing here updates. An uploaded file's row is written once and read many times, and a
delete does not change that: `erase` removes the object and `record_deletion` appends a
tombstone row, while the metadata row it describes is never touched. The sweep that
removes a tenant's uploads wholesale still works by prefix and is still not this
module's.

Two statements read tables this adapter does not own. The listing anti-joins
`uploaded_file_deletion` so a deleted file is absent from a page, and the in-use count
reads `event_log` and `session`, because a Session's attached file ids live only in its
`session.created` payload -- there is no join table to count instead. Both are reads,
and the second is the only place in this file that knows a Session exists.

The media type is not written onto the object. It is a column, so a download reads the
type and the bytes from one place and the two cannot come apart.

The bind and column types are declared because a textual statement carries no column
metadata for SQLAlchemy to infer from, so the driver receives whatever Python object the
caller handed it. Under asyncpg a uuid passed as a string is parsed anyway and does
match — that was measured, and the honest reason to declare it is that the declaration
is what keeps this adapter correct under a driver that does not parse for you, and that
`tests/adapters/test_statements_declare_their_types.py` requires it.
"""

from typing import Any

import aioboto3  # type: ignore[import-untyped]
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.control.files.store import FileId, FileWindow, UploadedFile
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vocabulary import lifecycle

_ROW_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "filename": sa.Text(),
    "media_type": sa.Text(),
    "byte_length": sa.BigInteger(),
    "content_sha256": sa.Text(),
    "produced_in_session_id": sa.Uuid(),
}

_INSERT = sa.text(
    "INSERT INTO uploaded_file"
    " (id, tenant_id, filename, media_type, byte_length, content_sha256,"
    " produced_in_session_id)"
    " VALUES (:id, :tenant, :filename, :media_type, :len, :hex,"
    " :produced_in_session_id)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    # Declared even though the value is often None, because the type is what tells the
    # driver how to send the value when it is NOT: a uuid bound with nothing declared
    # goes as whatever Python object the caller held.
    sa.bindparam("produced_in_session_id", type_=sa.Uuid()),
)

_LOOKUP = (
    sa.text(
        "SELECT id, filename, media_type, byte_length, content_sha256,"
        " produced_in_session_id"
        " FROM uploaded_file WHERE id = :id AND tenant_id = :tenant"
    )
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant", type_=sa.Uuid()),
    )
    .columns(**_ROW_TYPES)
)

_TOMBSTONE = sa.text(
    "INSERT INTO uploaded_file_deletion (file_id) VALUES (:file_id)"
    " ON CONFLICT (file_id) DO NOTHING"
).bindparams(sa.bindparam("file_id", type_=sa.Uuid()))

_DELETED = sa.text(
    "SELECT 1 FROM uploaded_file_deletion WHERE file_id = :file_id"
).bindparams(sa.bindparam("file_id", type_=sa.Uuid()))

_LIVE_SESSIONS_HOLDING = sa.text(
    "SELECT count(*) FROM event_log creation"
    " JOIN session s ON s.id = creation.session_id"
    " WHERE creation.type = :created_type"
    " AND s.tenant_id = :tenant"
    " AND jsonb_exists(creation.payload -> 'file_ids', :file_id)"
    " AND NOT EXISTS ("
    "  SELECT 1 FROM event_log ending"
    "  WHERE ending.session_id = creation.session_id"
    "  AND ending.type = :stopped_type)"
).bindparams(
    sa.bindparam("tenant", type_=sa.Uuid()),
    # Text, not Uuid, and this one is load-bearing rather than a declaration for the
    # guard's sake. The id is compared against an element of the creation payload's
    # `file_ids` array, which holds them as JSON strings, and `jsonb_exists` takes text
    # on its right: a uuid sent here matches no overload and the statement fails
    # outright rather than quietly returning nothing.
    sa.bindparam("file_id", type_=sa.Text()),
)
"""How many of a tenant's Sessions hold one file and have not stopped.

`session.stopped` is the only terminal transition the projection knows, so "not stopped"
is what non-terminal means here -- a suspended Session still holds its files and its pod
can still be placed again. The absence is asserted with NOT EXISTS over the log rather
than by folding each Session's state in Python, because the question is a count and
folding would mean reading every event of every candidate to answer it.

The join onto `session` is only for the tenant: `event_log` carries no tenant column.
A Session naming another tenant's file cannot be created, so this is a second lock on a
door already shut -- and it is what keeps the count from depending on that.
"""


def _page_statement(comparison: str, direction: str) -> sa.TextualSelect:
    """The listing, walking one way from an optional anchor.

    Two statements built from one template rather than one statement that can order
    either way. The comparison and the ORDER BY always flip together, and a single
    statement doing it from a bound flag needs CASE expressions in the ORDER BY whose
    NULL ordering is a trap nobody reading the query would see.

    Both nullable parameters are cast in the SQL text instead of being declared as bind
    types. A parameter appearing beside `IS NULL` gives PostgreSQL nothing to infer a
    type from, and the cast is what keeps that from depending on how the driver renders
    a bind that is used twice.

    The anchor's position is read as a row comparison against the pair the index is
    ordered by, so an anchor sharing a timestamp with another file still names one
    position. Selecting the pair in a subquery, under the same tenant predicate, is what
    makes a foreign id name no position at all rather than a position in another
    tenant's collection.
    """
    statement = (
        sa.text(
            "SELECT f.id, f.filename, f.media_type, f.byte_length,"
            " f.content_sha256, f.produced_in_session_id"
            " FROM uploaded_file f"
            " WHERE f.tenant_id = :tenant"
            " AND NOT EXISTS ("
            "  SELECT 1 FROM uploaded_file_deletion d WHERE d.file_id = f.id)"
            " AND (CAST(:scope_id AS uuid) IS NULL"
            "  OR f.produced_in_session_id = CAST(:scope_id AS uuid))"
            " AND (CAST(:anchor_id AS uuid) IS NULL"
            f"  OR (f.uploaded_at, f.id) {comparison} ("
            "   SELECT a.uploaded_at, a.id FROM uploaded_file a"
            "   WHERE a.id = CAST(:anchor_id AS uuid) AND a.tenant_id = :tenant))"
            f" ORDER BY f.uploaded_at {direction}, f.id {direction}"
            " LIMIT :limit"
        )
        .bindparams(sa.bindparam("tenant", type_=sa.Uuid()))
        .columns(**_ROW_TYPES)
    )
    return statement


_PAGE_OLDER = _page_statement("<", "DESC")
"""Newest first, walking towards older files. The unanchored page, and `after_id`."""

_PAGE_NEWER = _page_statement(">", "ASC")
"""Oldest first, walking towards newer files. `before_id` only, and the caller reverses
the rows into the newest-first order every page is presented in."""


class S3UploadedFiles:
    """Uploaded files in one bucket prefix and one table."""

    def __init__(
        self, session: aioboto3.Session, bucket: str, engine: AsyncEngine
    ) -> None:
        self._session = session
        self._bucket = bucket
        self._engine = engine

    async def write(self, key: str, body: bytes) -> None:
        async with self._session.client("s3") as s3:
            await s3.put_object(Bucket=self._bucket, Key=key, Body=body)

    async def read_bytes(self, key: str) -> bytes | None:
        async with self._session.client("s3") as s3:
            try:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
            except s3.exceptions.NoSuchKey:
                return None
            body: bytes = await response["Body"].read()
        return body

    async def record(self, file: UploadedFile) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                _INSERT,
                {
                    "id": str(file.id),
                    "tenant": str(file.tenant_id),
                    "filename": file.filename,
                    "media_type": file.media_type,
                    "len": file.byte_length,
                    "hex": file.content_sha256,
                    # str() for the same reason every other id here is stringified, and
                    # None passed through as None: a tenant's upload names no Session
                    # and the column is nullable, so there is nothing to spell.
                    "produced_in_session_id": (
                        None
                        if file.produced_in_session_id is None
                        else str(file.produced_in_session_id)
                    ),
                },
            )

    async def lookup(self, tenant_id: TenantId, file_id: FileId) -> UploadedFile | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _LOOKUP, {"id": str(file_id), "tenant": str(tenant_id)}
                )
            ).one_or_none()
        return None if row is None else _record(tenant_id, row)

    async def page(
        self, tenant_id: TenantId, window: FileWindow, limit: int
    ) -> tuple[UploadedFile, ...]:
        """One page, walked away from the anchor rather than presented.

        The tenant is passed to `_record` rather than read from the rows, because it is
        already a term in the query: selecting it back would be asking the database to
        confirm the predicate it was just given.
        """
        backward = window.walks_backward
        anchor = window.before_id if backward else window.after_id
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    _PAGE_NEWER if backward else _PAGE_OLDER,
                    {
                        "tenant": str(tenant_id),
                        "scope_id": (
                            None if window.scope_id is None else str(window.scope_id)
                        ),
                        "anchor_id": None if anchor is None else str(anchor),
                        "limit": limit,
                    },
                )
            ).all()
        return tuple(_record(tenant_id, row) for row in rows)

    async def erase(self, key: str) -> None:
        """Remove one object.

        S3 answers a delete of a key that holds nothing with a success, and that is the
        behaviour this port wants rather than one worked around: a retried delete has
        nothing left to remove and no caller can act on the difference.
        """
        async with self._session.client("s3") as s3:
            await s3.delete_object(Bucket=self._bucket, Key=key)

    async def record_deletion_unless_held(
        self, tenant_id: TenantId, file_id: FileId
    ) -> int:
        """Write the tombstone, count the Sessions holding the file, roll back if any.

        Two statements in one transaction, in this order, and the transaction is what
        makes the pair a guard rather than two facts read a moment apart.

        `ON CONFLICT DO NOTHING` on the insert rather than a check followed by a write:
        the check would be a read whose answer can change before the write lands, and
        the primary key already decides the question. It also preserves the first
        deletion's `deleted_at`, so a retried delete does not move the recorded moment
        to the retry.

        The rollback is the refusal. Migration 0024's trigger refuses an UPDATE, not an
        aborted transaction, so a rolled-back insert leaves no row and no trace -- there
        is nothing to compensate for and nothing for a later reader to mistake for a
        deletion that happened.

        The two event types are bound rather than written into the SQL, so the query
        names the published vocabulary instead of two string literals that would keep
        compiling after a type was renamed.

        `connect()` with an explicit transaction rather than `begin()`, because
        `begin()`'s context manager commits on the way out and would refuse to commit a
        transaction this method had already rolled back. An exception escaping still
        rolls back, on the connection's own close.
        """
        async with self._engine.connect() as conn:
            transaction = await conn.begin()
            await conn.execute(_TOMBSTONE, {"file_id": str(file_id)})
            counted = await conn.scalar(
                _LIVE_SESSIONS_HOLDING,
                {
                    "tenant": str(tenant_id),
                    "file_id": str(file_id),
                    "created_type": lifecycle.SESSION_CREATED,
                    "stopped_type": lifecycle.SESSION_STOPPED,
                },
            )
            holding = int(counted or 0)
            if holding:
                await transaction.rollback()
            else:
                await transaction.commit()
        return holding

    async def deletion_recorded(self, file_id: FileId) -> bool:
        async with self._engine.connect() as conn:
            found = await conn.scalar(_DELETED, {"file_id": str(file_id)})
        return found is not None


def _record(tenant_id: TenantId, row: Any) -> UploadedFile:
    """One selected row as the record every read of this table hands back.

    A function rather than four copies of the same six assignments: the lookup and the
    listing select the same columns, and a field added to one of them by hand is a field
    the other silently stops carrying.
    """
    return UploadedFile(
        id=FileId(row.id),
        tenant_id=tenant_id,
        filename=str(row.filename),
        media_type=str(row.media_type),
        byte_length=int(row.byte_length),
        content_sha256=str(row.content_sha256),
        produced_in_session_id=(
            None
            if row.produced_in_session_id is None
            else SessionId(row.produced_in_session_id)
        ),
    )
