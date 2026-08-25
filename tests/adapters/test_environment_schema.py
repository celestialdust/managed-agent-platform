"""The `environment` table's own guarantees, and the two statements that use it.

Tier 1 (testcontainers, real PostgreSQL 17). Two halves, and both are here for reasons
`docs/lessons.md` records.

The schema half is asserted with raw SQL rather than through the adapter, because the
point of each guarantee is that it survives a writer that never loads our code -- a psql
session, a later slice's adapter, a migration somebody writes in a hurry. An id that
could be rewritten stops meaning one shape for the Sessions already naming it, and no
amount of care in the parse fixes that.

The adapter half exists because a fake satisfies a call signature and cannot fail on a
type the driver will refuse. `denied_paths` is bound as a Python `list` into a json
column, which is exactly the shape that is rejected on the first insert when a textual
statement declares no bind type -- and it is a domain word rather than `payload` or
`body`, so the AST guard over `adapters/` had to be taught the name in the same change.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.environment_store import PostgresEnvironmentStore
from managed_agent.core.ids import TenantId
from managed_agent.core.registration.environment import (
    Environment,
    EnvironmentId,
    new_environment_id,
)

_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64

_WRITE = sa.text(
    "INSERT INTO environment (id, tenant_id, name, runtime_image, denied_paths)"
    " VALUES (:id, :tenant_id, :name, :runtime_image, :denied_paths)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant_id", type_=sa.Uuid()),
    sa.bindparam("denied_paths", type_=sa.JSON()),
)

_RENAME = sa.text("UPDATE environment SET name = :name WHERE id = :id").bindparams(
    sa.bindparam("id", type_=sa.Uuid())
)

_READ = (
    sa.text("SELECT name, created_at_ms FROM environment WHERE id = :id")
    .bindparams(sa.bindparam("id", type_=sa.Uuid()))
    .columns(name=sa.Text(), created_at_ms=sa.BigInteger())
)


async def _write(
    engine: AsyncEngine,
    *,
    environment_id: uuid.UUID | None = None,
    name: str = "analysis",
    runtime_image: str = _IMAGE,
    denied_paths: list[str] | None = None,
) -> uuid.UUID:
    written = environment_id or uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            _WRITE,
            {
                "id": written,
                "tenant_id": uuid.uuid4(),
                "name": name,
                "runtime_image": runtime_image,
                "denied_paths": denied_paths if denied_paths is not None else [],
            },
        )
    return written


async def test_an_update_against_a_written_shape_is_refused(
    engine: AsyncEngine,
) -> None:
    """Refused, not ignored.

    The distinction is the whole reason this is a raising trigger and not a rewrite rule
    with DO INSTEAD NOTHING: a rule leaves the stored row correct while telling the
    writer it succeeded, and a writer that believes it edited a shape will act as though
    the Sessions naming that id are running the new one.
    """
    written = await _write(engine, name="analysis")

    with pytest.raises(DBAPIError, match="append-only"):
        async with engine.begin() as conn:
            await conn.execute(_RENAME, {"id": written, "name": "renamed"})

    async with engine.connect() as conn:
        assert await conn.scalar(_READ, {"id": written}) == "analysis"


async def test_a_shape_that_is_not_pinned_to_bytes_cannot_be_stored(
    engine: AsyncEngine,
) -> None:
    """The pin holds in the store as well as in the parse.

    A row written by anything but the parse still cannot carry a floating tag, and a tag
    is what would let one id mean two different sets of bytes on two different days.
    """
    with pytest.raises(IntegrityError, match="environment_image_digest_pinned"):
        await _write(engine, runtime_image="registry.map.internal/session:latest")


async def test_a_shape_with_no_name_cannot_be_stored(engine: AsyncEngine) -> None:
    with pytest.raises(IntegrityError, match="environment_name_present"):
        await _write(engine, name="")


async def test_denied_paths_cannot_be_left_unwritten(engine: AsyncEngine) -> None:
    """A shape that denies nothing is an empty array, so NULL is never "not set yet"."""
    async with engine.begin() as conn:
        with pytest.raises(IntegrityError):
            await conn.execute(
                sa.text(
                    "INSERT INTO environment (id, tenant_id, name, runtime_image)"
                    " VALUES (:id, :tenant_id, :name, :runtime_image)"
                ).bindparams(
                    sa.bindparam("id", type_=sa.Uuid()),
                    sa.bindparam("tenant_id", type_=sa.Uuid()),
                ),
                {
                    "id": uuid.uuid4(),
                    "tenant_id": uuid.uuid4(),
                    "name": "analysis",
                    "runtime_image": _IMAGE,
                },
            )


async def test_the_creation_instant_is_written_without_the_writer_supplying_it(
    engine: AsyncEngine,
) -> None:
    """Taken from the server clock, so two writers cannot disagree about when."""
    written = await _write(engine)

    async with engine.connect() as conn:
        row = (await conn.execute(_READ, {"id": written})).one()

    assert int(row.created_at_ms) > 0


async def test_one_id_can_only_be_written_once(engine: AsyncEngine) -> None:
    """The registration route mints a fresh id per call, so a repeat is a store fault.

    Asserted here because it is the constraint, not the route, that makes it impossible:
    a second INSERT under one id is the only way an id could come to name two shapes,
    and the update trigger above cannot see it.
    """
    written = await _write(engine)

    with pytest.raises(IntegrityError, match="environment_pkey"):
        await _write(engine, environment_id=written, name="second")


async def test_the_adapter_round_trips_a_shape_including_its_denied_paths(
    engine: AsyncEngine,
) -> None:
    """Both statements against the real database, over the real column types.

    The list is the part that matters. Bound into a json column with no declared type it
    is refused on the first insert; read back without a declared column type it can
    arrive as JSON *text*, and the parse above this would then be handed a string where
    it expects a list -- a failure that surfaces far from here.
    """
    store = PostgresEnvironmentStore(engine)
    tenant = TenantId(uuid.uuid4())
    environment = Environment(
        id=new_environment_id(),
        tenant_id=tenant,
        name="analysis",
        runtime_image=_IMAGE,
        denied_paths=("/session/workspace/secrets", "/session/workspace/keys"),
    )

    await store.insert(environment)
    row = await store.fetch(environment.id, tenant)

    assert row is not None
    assert row["denied_paths"] == [
        "/session/workspace/secrets",
        "/session/workspace/keys",
    ], "the denied paths did not come back as a list of strings"
    assert str(row["id"]) == str(environment.id)
    assert str(row["tenant_id"]) == str(tenant)
    assert row["name"] == "analysis"
    assert row["runtime_image"] == _IMAGE


async def test_the_adapter_hides_another_tenants_shape_rather_than_filtering_it(
    engine: AsyncEngine,
) -> None:
    """The tenant is a WHERE clause, so the row is never fetched in the first place.

    Checked against the real statement because that is where the predicate lives; a fake
    keyed on a tuple would satisfy this while the SQL selected on the id alone.
    """
    store = PostgresEnvironmentStore(engine)
    owner = TenantId(uuid.uuid4())
    environment = Environment(
        id=new_environment_id(),
        tenant_id=owner,
        name="analysis",
        runtime_image=_IMAGE,
        denied_paths=(),
    )
    await store.insert(environment)

    assert await store.fetch(environment.id, TenantId(uuid.uuid4())) is None
    assert await store.fetch(new_environment_id(), owner) is None
    assert await store.fetch(environment.id, owner) is not None


async def test_the_adapter_refuses_to_write_one_id_twice(engine: AsyncEngine) -> None:
    """A collision is a store fault surfaced to the caller, never a silent edit."""
    store = PostgresEnvironmentStore(engine)
    tenant = TenantId(uuid.uuid4())
    environment_id: EnvironmentId = new_environment_id()
    first = Environment(
        id=environment_id,
        tenant_id=tenant,
        name="first",
        runtime_image=_IMAGE,
        denied_paths=(),
    )
    await store.insert(first)

    with pytest.raises(IntegrityError):
        await store.insert(
            Environment(
                id=environment_id,
                tenant_id=tenant,
                name="second",
                runtime_image=_IMAGE,
                denied_paths=(),
            )
        )
