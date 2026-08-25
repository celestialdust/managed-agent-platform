"""The store, not the application, is what refuses an illegal agent_definition row.

Tier 1 (testcontainers, real PostgreSQL 17). Asserted with raw SQL rather than through
the registry adapter, because the point of each of these is that the guarantee survives
a writer that never loads our code -- a psql session, a later slice's adapter, a
migration somebody writes in a hurry.

A definition is versioned rather than edited, and this table is where that stops being
an intention. `(id, revision)` is the key, so registering again writes a new row instead
of replacing one a running Session may already have pinned; an UPDATE raises rather than
succeeding quietly.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

_SHA = "0" * 39 + "a"

_INSERT = sa.text(
    "INSERT INTO agent_definition (id, tenant_id, revision, body, skills_revision)"
    " VALUES (:id, :tenant, :revision, :body, :skills)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("body", type_=sa.JSON()),
)


async def _write(
    engine: AsyncEngine,
    definition_id: uuid.UUID,
    tenant_id: uuid.UUID,
    revision: int,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            _INSERT,
            {
                "id": definition_id,
                "tenant": tenant_id,
                "revision": revision,
                "body": {"name": f"r{revision}"},
                "skills": _SHA,
            },
        )


async def test_the_same_id_holds_more_than_one_revision(engine: AsyncEngine) -> None:
    """Two revisions of one definition coexist, rather than the second replacing the
    first.

    This is the whole reason the key is a pair. A Session that resolved revision 1 keeps
    resolving revision 1 after revision 2 is registered, and it can only do that if
    revision 1's row is still there.
    """
    definition_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, definition_id, tenant_id, 1)
    await _write(engine, definition_id, tenant_id, 2)

    async with engine.connect() as conn:
        names = (
            await conn.execute(
                sa.text(
                    "SELECT revision, body->>'name' AS name FROM agent_definition"
                    " WHERE id = :id ORDER BY revision"
                ).bindparams(sa.bindparam("id", type_=sa.Uuid())),
                {"id": definition_id},
            )
        ).all()

    assert [(row.revision, row.name) for row in names] == [(1, "r1"), (2, "r2")], (
        "registering a second revision did not leave the first intact; a Session "
        "pinned to revision 1 would silently start running revision 2"
    )


async def test_a_duplicate_revision_of_one_definition_is_refused(
    engine: AsyncEngine,
) -> None:
    """The key catches it, so two writers cannot both believe they made revision 2."""
    definition_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, definition_id, tenant_id, 1)

    with pytest.raises(IntegrityError):
        await _write(engine, definition_id, tenant_id, 1)


async def test_an_update_against_a_registered_revision_is_refused_and_changes_nothing(
    engine: AsyncEngine,
) -> None:
    """Both halves: the row is unchanged, and the writer is told so.

    Refused rather than absorbed. A rewrite rule doing nothing would leave the row
    correct while reporting success to whoever tried to change it, which is the shape of
    failure that gets discovered a month later -- so this raises and the writer sees it.
    Migration 0001 made the same call for `event_log` and said so in the same words.
    """
    definition_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, definition_id, tenant_id, 1)

    with pytest.raises(DBAPIError) as caught:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE agent_definition SET skills_revision = :other"
                    " WHERE id = :id"
                ).bindparams(sa.bindparam("id", type_=sa.Uuid())),
                {"id": definition_id, "other": "f" * 40},
            )

    assert "append-only" in str(caught.value) or "may not be updated" in str(
        caught.value
    ), f"the refusal does not say why the write was rejected: {caught.value}"

    async with engine.connect() as conn:
        stored = await conn.scalar(
            sa.text(
                "SELECT skills_revision FROM agent_definition WHERE id = :id"
            ).bindparams(sa.bindparam("id", type_=sa.Uuid())),
            {"id": definition_id},
        )
    assert stored == _SHA, (
        f"the pinned skills revision changed to {stored!r}; a definition that can be "
        "edited in place changes what an already-running Session is"
    )


async def test_revision_zero_is_refused_by_the_store(engine: AsyncEngine) -> None:
    """Revisions count from 1, so 0 is not an earlier revision -- it is a bug.

    Usually an index/count confusion. Refusing it here means the confusion cannot reach
    a Session's pin, where it would resolve to nothing and be read as a missing
    definition.
    """
    with pytest.raises(IntegrityError):
        await _write(engine, uuid.uuid4(), uuid.uuid4(), 0)


async def test_the_body_is_stored_as_jsonb(engine: AsyncEngine) -> None:
    """`jsonb`, not `json`, and the difference is a table rewrite once rows exist.

    `json` keeps the original text and re-parses it on every access; it cannot be
    indexed or queried inside. Later slices read fields out of this column -- picking a
    capability out of a definition, listing versions by name -- and converting the
    column afterwards is a full rewrite holding an ACCESS EXCLUSIVE lock. Migration 0003
    paid exactly that lesson on `event_log.payload`; this is it not being paid twice.
    """
    async with engine.connect() as conn:
        column_type = await conn.scalar(
            sa.text(
                "SELECT data_type FROM information_schema.columns"
                " WHERE table_name = 'agent_definition' AND column_name = 'body'"
            )
        )

    assert column_type == "jsonb", (
        f"body is {column_type!r}; json cannot be indexed or queried inside, and "
        "converting it once the table holds rows is a full rewrite under an ACCESS "
        "EXCLUSIVE lock"
    )
