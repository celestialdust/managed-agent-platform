"""The store, not the application, is what makes a deletion a fact written once.

Tier 1 (testcontainers, real PostgreSQL 17). Asserted with raw SQL rather than through
the adapter, because the point of each of these is that the guarantee survives a writer
that never loads our code -- a psql session, a later slice's adapter, a migration
somebody writes in a hurry.

A deletion lives in a table of its own precisely because `uploaded_file` refuses an
UPDATE by raising, so it could never have been a column there. This table then owes the
guarantees the one it describes owes: a key that names a file somebody uploaded, one row
per deletion, and no rewriting.

The single-row rule is the one with a consequence at the surface. It is what lets the
route absorb a repeated delete and answer about the moment the file first stopped being
usable: a second row would date the deletion to whichever retry wrote it, so a caller
retrying a timeout would move the answer to "when did this data go" every time they
asked.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

_SHA = "0" * 64

_UPLOAD = sa.text(
    "INSERT INTO uploaded_file"
    " (id, tenant_id, filename, media_type, byte_length, content_sha256)"
    " VALUES (:id, :tenant, 'notes.txt', 'text/plain', 3, :hex)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
)

_DELETE = sa.text(
    "INSERT INTO uploaded_file_deletion (file_id) VALUES (:file_id)"
).bindparams(sa.bindparam("file_id", type_=sa.Uuid()))

_ABSORBED = sa.text(
    "INSERT INTO uploaded_file_deletion (file_id) VALUES (:file_id)"
    " ON CONFLICT (file_id) DO NOTHING"
).bindparams(sa.bindparam("file_id", type_=sa.Uuid()))

_MOMENT = sa.text(
    "SELECT deleted_at FROM uploaded_file_deletion WHERE file_id = :file_id"
).bindparams(sa.bindparam("file_id", type_=sa.Uuid()))

_COUNT = sa.text(
    "SELECT count(*) FROM uploaded_file_deletion WHERE file_id = :file_id"
).bindparams(sa.bindparam("file_id", type_=sa.Uuid()))


async def _uploaded(engine: AsyncEngine) -> uuid.UUID:
    file_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            _UPLOAD, {"id": file_id, "tenant": uuid.uuid4(), "hex": _SHA}
        )
    return file_id


async def _deleted_rows(engine: AsyncEngine, file_id: uuid.UUID) -> int:
    async with engine.connect() as conn:
        return int(await conn.scalar(_COUNT, {"file_id": file_id}) or 0)


async def test_deleting_a_file_twice_is_one_row_and_a_constraint_violation(
    engine: AsyncEngine,
) -> None:
    """The key is the file, so a repeat is refused rather than counted twice.

    That is what lets the adapter's `ON CONFLICT DO NOTHING` mean "already deleted"
    instead of hiding a second deletion that was never distinguishable anyway.
    """
    file_id = await _uploaded(engine)
    async with engine.begin() as conn:
        await conn.execute(_DELETE, {"file_id": file_id})

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_DELETE, {"file_id": file_id})

    assert await _deleted_rows(engine, file_id) == 1


async def test_an_absorbed_repeat_keeps_the_first_moment(engine: AsyncEngine) -> None:
    """The statement the adapter actually sends, and the property it is sent for.

    `ON CONFLICT DO NOTHING` has to leave `deleted_at` alone, not refresh it. A caller
    retrying a timed-out delete would otherwise be told a later moment each time they
    asked, and a tenant answering "when was this data removed" would have no stable
    answer to give.
    """
    file_id = await _uploaded(engine)
    async with engine.begin() as conn:
        await conn.execute(_ABSORBED, {"file_id": file_id})
    async with engine.connect() as conn:
        first = await conn.scalar(_MOMENT, {"file_id": file_id})

    async with engine.begin() as conn:
        await conn.execute(_ABSORBED, {"file_id": file_id})

    async with engine.connect() as conn:
        assert await conn.scalar(_MOMENT, {"file_id": file_id}) == first
    assert await _deleted_rows(engine, file_id) == 1


async def test_deleting_a_file_nobody_uploaded_is_refused_by_the_store(
    engine: AsyncEngine,
) -> None:
    """The foreign key is what makes this a refusal rather than a row about nothing.

    Without it the table would hold a deletion of an id that was never issued, and every
    later read anti-joining against it would silently skip a row that does not exist --
    a listing that quietly hides files nobody deleted.
    """
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_DELETE, {"file_id": uuid.uuid4()})


async def test_an_update_to_a_deletion_raises_rather_than_being_ignored(
    engine: AsyncEngine,
) -> None:
    """Refused loudly, which is the mechanism migration 0001 settled for this tree.

    A rewrite rule doing nothing would leave the stored row correct while reporting
    success to whoever tried to change it -- the failure discovered a month later by
    somebody wondering why their edit did not take. Both halves are asserted: the raise,
    and that the moment did not move.
    """
    file_id = await _uploaded(engine)
    async with engine.begin() as conn:
        await conn.execute(_DELETE, {"file_id": file_id})
    async with engine.connect() as conn:
        recorded = await conn.scalar(_MOMENT, {"file_id": file_id})

    with pytest.raises(DBAPIError) as raised:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE uploaded_file_deletion SET deleted_at = now()"
                    " WHERE file_id = :file_id"
                ).bindparams(sa.bindparam("file_id", type_=sa.Uuid())),
                {"file_id": file_id},
            )

    assert "append-only" in str(raised.value)
    async with engine.connect() as conn:
        assert await conn.scalar(_MOMENT, {"file_id": file_id}) == recorded


async def test_the_metadata_row_survives_its_own_deletion(engine: AsyncEngine) -> None:
    """The whole point of the table, asserted at the level that decides it.

    The id outlives the object: a Session's creation event names the files it was
    created with, so the row has to still be there for that history to resolve. Nothing
    in the schema deletes it, and this is what would fail if a later migration added a
    cascade.
    """
    file_id = await _uploaded(engine)
    async with engine.begin() as conn:
        await conn.execute(_DELETE, {"file_id": file_id})

    async with engine.connect() as conn:
        surviving = await conn.scalar(
            sa.text("SELECT count(*) FROM uploaded_file WHERE id = :id").bindparams(
                sa.bindparam("id", type_=sa.Uuid())
            ),
            {"id": file_id},
        )

    assert surviving == 1


async def test_the_deletion_carries_no_tenant_column(engine: AsyncEngine) -> None:
    """Which tenant owns a file is `uploaded_file`'s fact and is not copied here.

    A copy would be free to disagree with the row it describes, and the read that
    matters -- "may this tenant delete this file" -- is answered by selecting the file
    out of `uploaded_file` under a tenant predicate first.
    """
    async with engine.connect() as conn:
        columns = set(
            (
                await conn.execute(
                    sa.text(
                        "SELECT column_name FROM information_schema.columns"
                        " WHERE table_name = 'uploaded_file_deletion'"
                    )
                )
            )
            .scalars()
            .all()
        )

    assert columns == {"file_id", "deleted_at"}
