"""The store, not the application, is what makes a retirement a fact written once.

Tier 1 (testcontainers, real PostgreSQL 17). Asserted with raw SQL rather than through
the registry adapter, because the point of each of these is that the guarantee survives
a writer that never loads our code -- a psql session, a later slice's adapter, a
migration somebody writes in a hurry.

Retirement lives in a table of its own precisely because `agent_definition` refuses an
UPDATE by raising, so it could not have been recorded as a column there at all. This
table then owes the same guarantees the one it describes owes: a pair that names a
registered version, one row per retirement, and no rewriting.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

_SHA = "0" * 39 + "a"

_REGISTER = sa.text(
    "INSERT INTO agent_definition (id, tenant_id, revision, body, skills_revision)"
    " VALUES (:id, :tenant, :revision, :body, :skills)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("body", type_=sa.JSON()),
)

_ARCHIVE = sa.text(
    "INSERT INTO agent_version_archive (definition_id, revision)"
    " VALUES (:id, :revision)"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_COUNT = sa.text(
    "SELECT count(*) FROM agent_version_archive"
    " WHERE definition_id = :id AND revision = :revision"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))


async def _register(
    engine: AsyncEngine, definition_id: uuid.UUID, revision: int
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            _REGISTER,
            {
                "id": definition_id,
                "tenant": uuid.uuid4(),
                "revision": revision,
                "body": {"name": f"r{revision}"},
                "skills": _SHA,
            },
        )


async def _archive(
    engine: AsyncEngine, definition_id: uuid.UUID, revision: int
) -> None:
    async with engine.begin() as conn:
        await conn.execute(_ARCHIVE, {"id": definition_id, "revision": revision})


async def _archived_rows(
    engine: AsyncEngine, definition_id: uuid.UUID, revision: int
) -> int:
    async with engine.connect() as conn:
        return int(
            await conn.scalar(_COUNT, {"id": definition_id, "revision": revision}) or 0
        )


async def test_retiring_a_version_twice_is_one_row_and_a_constraint_violation(
    engine: AsyncEngine,
) -> None:
    """The key is the pair, so a repeat is refused rather than counted twice.

    That is what lets the adapter's `ON CONFLICT DO NOTHING` mean "already retired"
    instead of hiding a second retirement that was never distinguishable anyway.
    """
    definition_id = uuid.uuid4()
    await _register(engine, definition_id, 1)
    await _archive(engine, definition_id, 1)

    with pytest.raises(IntegrityError):
        await _archive(engine, definition_id, 1)

    assert await _archived_rows(engine, definition_id, 1) == 1


async def test_retiring_a_version_nobody_registered_is_refused_by_the_store(
    engine: AsyncEngine,
) -> None:
    """The composite foreign key is what makes this a refusal, not a row about nothing.

    Without it the table would happily hold a retirement of revision 4 of an agent that
    has three revisions, and every later read joining against it would silently skip it.
    """
    definition_id = uuid.uuid4()
    await _register(engine, definition_id, 1)

    with pytest.raises(IntegrityError):
        await _archive(engine, definition_id, 2)

    with pytest.raises(IntegrityError):
        await _archive(engine, uuid.uuid4(), 1)


async def test_an_update_to_a_retirement_raises_rather_than_being_ignored(
    engine: AsyncEngine,
) -> None:
    """Refused loudly, which is the mechanism migration 0001 settled for this tree.

    A rewrite rule doing nothing would leave the stored row correct while reporting
    success to whoever tried to change it -- the failure discovered a month later by
    somebody wondering why their edit did not take. Both halves are asserted: the raise,
    and that nothing changed.
    """
    definition_id = uuid.uuid4()
    await _register(engine, definition_id, 1)
    await _archive(engine, definition_id, 1)

    with pytest.raises(DBAPIError) as raised:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE agent_version_archive SET archived_at = now()"
                    " WHERE definition_id = :id"
                ).bindparams(sa.bindparam("id", type_=sa.Uuid())),
                {"id": definition_id},
            )

    assert "append-only" in str(raised.value)
    assert await _archived_rows(engine, definition_id, 1) == 1


async def test_the_archive_carries_no_tenant_column(engine: AsyncEngine) -> None:
    """Which tenant owns a revision is `agent_definition`'s fact and is not copied.

    A copy here would be free to disagree with the row it describes, and the read that
    matters -- "may this tenant retire this version" -- is answered by selecting the key
    out of `agent_definition` under a tenant predicate instead.
    """
    async with engine.connect() as conn:
        columns = set(
            (
                await conn.execute(
                    sa.text(
                        "SELECT column_name FROM information_schema.columns"
                        " WHERE table_name = 'agent_version_archive'"
                    )
                )
            )
            .scalars()
            .all()
        )

    assert columns == {"definition_id", "revision", "archived_at"}
