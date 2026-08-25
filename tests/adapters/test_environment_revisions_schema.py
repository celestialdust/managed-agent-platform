"""What makes an Environment editable, retirable and removable is the store.

Tier 1 (testcontainers, real PostgreSQL 17). Two halves, and both are here for reasons
`docs/lessons.md` records.

The schema half is asserted with raw SQL rather than through the adapter, because the
point of each guarantee is that it survives a writer that never loads our code -- a psql
session, a later slice's adapter, a migration somebody writes in a hurry. An id that
can be rewritten stops meaning one shape for the Sessions already naming it, and no
amount of care in the parse fixes that.

The adapter half is what no fake can stand in for, and this file is where three specific
defaults stop being taken on trust. `revision` and `archived_at` are read out of a row
with a `.get` and a default, because five in-memory stand-ins in this repository return
rows without them. Those defaults are safe *for a store with one revision and no
archive table*, and quietly wrong for a read that lost a column -- so the only place the
join and the ordering can be pinned is against a database that has both. That is here.
The third is the SQL itself: a bound boolean in a WHERE clause and a row-constructor
keyset are both statements no fake executes.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.environment_store import PostgresEnvironmentStore
from managed_agent.control.catalog.environments import (
    EnvironmentLifecycle,
    parse_environment,
)
from managed_agent.core.ids import TenantId
from managed_agent.core.registration.environment import (
    Environment,
    EnvironmentId,
    new_environment_id,
)
from managed_agent.core.vocabulary import lifecycle

_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
_OTHER_IMAGE = "registry.map.internal/session@sha256:" + "b" * 64

_WRITE = sa.text(
    "INSERT INTO environment"
    " (id, tenant_id, name, runtime_image, denied_paths, revision)"
    " VALUES (:id, :tenant, :name, :image, :denied_paths, :revision)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("denied_paths", type_=sa.JSON()),
)

_WRITE_WITHOUT_A_REVISION = sa.text(
    "INSERT INTO environment (id, tenant_id, name, runtime_image, denied_paths)"
    " VALUES (:id, :tenant, :name, :image, :denied_paths)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("denied_paths", type_=sa.JSON()),
)

_RENAME_A_REVISION = sa.text(
    "UPDATE environment SET name = :name WHERE id = :id AND revision = :revision"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_REVISIONS = (
    sa.text("SELECT revision, name FROM environment WHERE id = :id ORDER BY revision")
    .bindparams(sa.bindparam("id", type_=sa.Uuid()))
    .columns(revision=sa.Integer(), name=sa.Text())
)

_ARCHIVE = sa.text(
    "INSERT INTO environment_archive (environment_id) VALUES (:id)"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_MOVE_AN_ARCHIVE = sa.text(
    "UPDATE environment_archive SET archived_at = now() WHERE environment_id = :id"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_ARCHIVE_ROWS = sa.text(
    "SELECT count(*) FROM environment_archive WHERE environment_id = :id"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_APPEND_EVENT = sa.text(
    "INSERT INTO event_log (session_id, seq, type, payload)"
    " VALUES (:session_id, :seq, :type, :payload)"
).bindparams(
    sa.bindparam("session_id", type_=sa.Uuid()),
    sa.bindparam("payload", type_=sa.JSON()),
)


def _a_shape(
    environment_id: EnvironmentId,
    tenant_id: TenantId,
    *,
    name: str = "analysis",
    image: str = _IMAGE,
    domains: tuple[str, ...] = (),
) -> Environment:
    return parse_environment(
        environment_id=environment_id,
        tenant_id=tenant_id,
        name=name,
        runtime_image=image,
        denied_paths=("/session/workspace/secrets",),
        allowed_domains=domains,
    )


async def _write(
    engine: AsyncEngine,
    environment_id: uuid.UUID,
    tenant_id: uuid.UUID,
    revision: int | None,
    *,
    name: str = "analysis",
) -> None:
    statement = _WRITE if revision is not None else _WRITE_WITHOUT_A_REVISION
    values: dict[str, object] = {
        "id": environment_id,
        "tenant": tenant_id,
        "name": name,
        "image": _IMAGE,
        "denied_paths": [],
    }
    if revision is not None:
        values["revision"] = revision
    async with engine.begin() as conn:
        await conn.execute(statement, values)


async def _created_in(
    engine: AsyncEngine, environment_id: EnvironmentId, *, stopped: bool
) -> None:
    """One Session's log: created in this Environment, and optionally stopped.

    Written as raw events rather than through the create route, because what is under
    test is a query over the log and the route is not what put the pairing there -- the
    payload key is.
    """
    session_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            _APPEND_EVENT,
            {
                "session_id": session_id,
                "seq": 1,
                "type": lifecycle.SESSION_CREATED,
                "payload": {"environment_id": str(environment_id)},
            },
        )
        if stopped:
            await conn.execute(
                _APPEND_EVENT,
                {
                    "session_id": session_id,
                    "seq": 2,
                    "type": lifecycle.SESSION_STOPPED,
                    "payload": {"stop_reason": "archived"},
                },
            )


# --------------------------------------------------------------------------------------
# The schema, asserted against a writer that never loads our code
# --------------------------------------------------------------------------------------


async def test_two_revisions_of_one_id_coexist(engine: AsyncEngine) -> None:
    """The primary key is the pair now, which is what makes an edit expressible at all.

    Before 0022 this second row was a primary-key violation, so `POST
    /v1/environments/{id}` had nowhere to write: the table refuses an UPDATE by raising,
    and the key refused an append.
    """
    environment_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, environment_id, tenant_id, 1, name="first")
    await _write(engine, environment_id, tenant_id, 2, name="second")

    async with engine.connect() as conn:
        rows = (await conn.execute(_REVISIONS, {"id": environment_id})).all()

    assert [(row.revision, row.name) for row in rows] == [(1, "first"), (2, "second")]


async def test_one_revision_of_one_id_twice_is_refused(engine: AsyncEngine) -> None:
    """The pair is still a key, so an append that reuses a number is a violation rather
    than a second row -- which is what makes the adapter's self-numbering insert safe
    under a race instead of merely usually right."""
    environment_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, environment_id, tenant_id, 1)

    with pytest.raises(IntegrityError):
        await _write(engine, environment_id, tenant_id, 1)


async def test_a_write_that_names_no_revision_lands_on_the_first(
    engine: AsyncEngine,
) -> None:
    """The server default of 1 survives 0022, and this is what rests on it.

    An earlier draft of that migration dropped the default on the argument that a writer
    omitting the column would silently keep landing on 1. Dropped, it instead breaks
    every insert that exists: the create statement does not name the column, so a NOT
    NULL with nothing behind it refuses the first Environment anybody registers.
    """
    environment_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, environment_id, tenant_id, None)

    async with engine.connect() as conn:
        rows = (await conn.execute(_REVISIONS, {"id": environment_id})).all()

    assert [row.revision for row in rows] == [1]


async def test_a_revision_below_one_is_refused(engine: AsyncEngine) -> None:
    """`revision >= 1` is a check constraint, so 0 cannot be stored by anything.

    It matters because the read parses the number and pins it into a Session: a 0 in a
    payload names a revision no row can ever be found by.
    """
    with pytest.raises(IntegrityError):
        await _write(engine, uuid.uuid4(), uuid.uuid4(), 0)


async def test_an_update_to_any_revision_raises_naming_that_revision(
    engine: AsyncEngine,
) -> None:
    """The append-only trigger survives 0022 and its message was rewritten to be true.

    The old wording said the shape registered under an id may not be changed, which now
    misleads: a reader hitting it would conclude an Environment cannot be edited, when
    what cannot happen is a revision being rewritten after something pinned it. Both
    revisions are tried, because a trigger reinstalled for the newest row only would
    leave every earlier one -- the ones Sessions actually pinned -- rewritable.
    """
    environment_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, environment_id, tenant_id, 1)
    await _write(engine, environment_id, tenant_id, 2)

    for revision in (1, 2):
        with pytest.raises(DBAPIError) as refused:
            async with engine.begin() as conn:
                await conn.execute(
                    _RENAME_A_REVISION,
                    {"id": environment_id, "revision": revision, "name": "renamed"},
                )
        assert "append-only" in str(refused.value)
        assert "an edit is a new revision" in str(refused.value)

    async with engine.connect() as conn:
        rows = (await conn.execute(_REVISIONS, {"id": environment_id})).all()
    assert [row.name for row in rows] == ["analysis", "analysis"]


async def test_retiring_one_environment_twice_is_one_row(engine: AsyncEngine) -> None:
    """One row per Environment and not per revision, so the timestamp answers "when did
    this stop being referenceable" once -- and the adapter's `ON CONFLICT DO NOTHING`
    means "already retired" rather than hiding a second retirement."""
    environment_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, environment_id, tenant_id, 1)
    async with engine.begin() as conn:
        await conn.execute(_ARCHIVE, {"id": environment_id})

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_ARCHIVE, {"id": environment_id})

    async with engine.connect() as conn:
        assert await conn.scalar(_ARCHIVE_ROWS, {"id": environment_id}) == 1


async def test_moving_a_retirement_raises(engine: AsyncEngine) -> None:
    """An archive is terminal, so its own row is append-only too.

    Without this the timestamp is advisory: anything could move it forward and the
    Environment would report a retirement later than the one new Sessions were actually
    refused from.
    """
    environment_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    await _write(engine, environment_id, tenant_id, 1)
    async with engine.begin() as conn:
        await conn.execute(_ARCHIVE, {"id": environment_id})

    with pytest.raises(DBAPIError) as refused:
        async with engine.begin() as conn:
            await conn.execute(_MOVE_AN_ARCHIVE, {"id": environment_id})

    assert "append-only" in str(refused.value)


# --------------------------------------------------------------------------------------
# The adapter, against the database the fakes stand in for
# --------------------------------------------------------------------------------------


def test_the_store_is_the_whole_lifecycle_and_not_only_the_reader() -> None:
    """The widening the four lifecycle routes assert at runtime, checked at type time.

    `Platform.environment_store` is typed at the two-method port, so nothing in `src`
    ever asks mypy whether this class satisfies the wider one. This return does, and the
    `isinstance` is the same question at runtime -- which is the form the routes ask it
    in, since the composition root hands them an object and not a type.
    """

    def widens(store: PostgresEnvironmentStore) -> EnvironmentLifecycle:
        return store

    store = PostgresEnvironmentStore(engine=None)  # type: ignore[arg-type]
    assert isinstance(widens(store), EnvironmentLifecycle)


async def test_a_fetch_returns_the_latest_revision_and_says_which(
    engine: AsyncEngine,
) -> None:
    """The claim every route in the lifecycle rests on, against a real ORDER BY.

    `revision` is read out of the row with a default of 1, so a store that never
    selected the column would satisfy every in-memory case in this repository. This is
    the only place that default can be told apart from a real read.
    """
    store = PostgresEnvironmentStore(engine)
    tenant_id = TenantId(uuid.uuid4())
    first = _a_shape(new_environment_id(), tenant_id)
    await store.insert(first)
    edited = _a_shape(
        first.id, tenant_id, name="revised", image=_OTHER_IMAGE, domains=("a.example",)
    )

    assert await store.insert_revision(edited) == 2

    row = await store.fetch(first.id, tenant_id)
    assert row is not None
    assert row["revision"] == 2
    assert row["name"] == "revised"
    assert row["runtime_image"] == _OTHER_IMAGE
    assert row["allowed_domains"] == ["a.example"]
    assert row["archived_at"] is None, (
        "archived_at has to be present and null rather than absent: absent is what a "
        "read that lost the join looks like, and the parse reads absent as not retired"
    )
    # The revision a Session may already have pinned is still there, unchanged.
    async with engine.connect() as conn:
        rows = (await conn.execute(_REVISIONS, {"id": first.id})).all()
    assert [(row.revision, row.name) for row in rows] == [
        (1, "analysis"),
        (2, "revised"),
    ]


async def test_a_pinned_fetch_answers_the_revision_it_names(
    engine: AsyncEngine,
) -> None:
    """The read that honours a Session's pin, against a real WHERE on the pair.

    Both revisions, because a statement that ignored the `revision` parameter and kept
    the `ORDER BY revision DESC LIMIT 1` would satisfy a case that only asked for the
    newest -- and that is exactly the shape this read is a departure from.
    """
    store = PostgresEnvironmentStore(engine)
    tenant_id = TenantId(uuid.uuid4())
    first = _a_shape(new_environment_id(), tenant_id, name="first")
    await store.insert(first)
    await store.insert_revision(
        _a_shape(first.id, tenant_id, name="second", image=_OTHER_IMAGE)
    )

    pinned = await store.fetch_revision(first.id, tenant_id, 1)
    newest = await store.fetch_revision(first.id, tenant_id, 2)

    assert pinned is not None and newest is not None
    assert (pinned["revision"], pinned["name"]) == (1, "first")
    assert (newest["revision"], newest["name"]) == (2, "second")
    assert pinned["runtime_image"] == _IMAGE


async def test_a_pinned_fetch_is_absent_for_a_revision_or_tenant_that_has_none(
    engine: AsyncEngine,
) -> None:
    """One absence for a revision nobody wrote and for another tenant's id, because a
    caller able to tell them apart could count somebody else's edits."""
    store = PostgresEnvironmentStore(engine)
    owner = TenantId(uuid.uuid4())
    registered = _a_shape(new_environment_id(), owner)
    await store.insert(registered)

    assert await store.fetch_revision(registered.id, owner, 2) is None
    assert await store.fetch_revision(registered.id, TenantId(uuid.uuid4()), 1) is None
    assert await store.fetch_revision(new_environment_id(), owner, 1) is None


async def test_a_pinned_fetch_still_answers_for_a_retired_environment(
    engine: AsyncEngine,
) -> None:
    """No archive predicate on this read, and that is the decision rather than an
    omission: a pod for a Session created before the retirement is still placed, or
    archiving would stop live work instead of stopping new work."""
    store = PostgresEnvironmentStore(engine)
    tenant_id = TenantId(uuid.uuid4())
    registered = _a_shape(new_environment_id(), tenant_id)
    await store.insert(registered)
    retired_at = await store.archive(registered.id, tenant_id)

    row = await store.fetch_revision(registered.id, tenant_id, 1)

    assert row is not None
    assert row["archived_at"] == retired_at


async def test_a_fetch_of_another_tenants_id_is_absent(engine: AsyncEngine) -> None:
    """The tenant is a term in the query, so the row is not fetched and then dropped."""
    store = PostgresEnvironmentStore(engine)
    owner = TenantId(uuid.uuid4())
    theirs = _a_shape(new_environment_id(), owner)
    await store.insert(theirs)

    assert await store.fetch(theirs.id, TenantId(uuid.uuid4())) is None
    assert await store.fetch(new_environment_id(), owner) is None


async def test_archiving_through_the_adapter_answers_the_first_time_twice(
    engine: AsyncEngine,
) -> None:
    """Idempotent, and observably so: the second call reports the first retirement.

    A fresh timestamp on a retry would claim the Environment stopped being referenceable
    at the moment of the retry, which is a false fact about when new Sessions began
    being refused.
    """
    store = PostgresEnvironmentStore(engine)
    tenant_id = TenantId(uuid.uuid4())
    registered = _a_shape(new_environment_id(), tenant_id)
    await store.insert(registered)

    first = await store.archive(registered.id, tenant_id)
    again = await store.archive(registered.id, tenant_id)

    assert isinstance(first, datetime)
    assert first == again
    row = await store.fetch(registered.id, tenant_id)
    assert row is not None
    assert row["archived_at"] == first


async def test_archiving_another_tenants_environment_writes_nothing(
    engine: AsyncEngine,
) -> None:
    """The id is selected out of `environment` under the tenant predicate, so the insert
    has no row to write and the answer is the same absence a fetch gives."""
    store = PostgresEnvironmentStore(engine)
    owner = TenantId(uuid.uuid4())
    theirs = _a_shape(new_environment_id(), owner)
    await store.insert(theirs)

    assert await store.archive(theirs.id, TenantId(uuid.uuid4())) is None

    async with engine.connect() as conn:
        assert await conn.scalar(_ARCHIVE_ROWS, {"id": theirs.id}) == 0


async def test_a_delete_removes_every_revision_and_the_retirement(
    engine: AsyncEngine,
) -> None:
    """Every revision, because a delete of an Environment is not a delete of a revision:
    leaving the earlier rows would leave the id resolvable at a shape older than the one
    the caller asked to remove."""
    store = PostgresEnvironmentStore(engine)
    tenant_id = TenantId(uuid.uuid4())
    registered = _a_shape(new_environment_id(), tenant_id)
    await store.insert(registered)
    await store.insert_revision(_a_shape(registered.id, tenant_id, name="revised"))
    await store.archive(registered.id, tenant_id)

    assert await store.delete(registered.id, tenant_id) is True

    assert await store.fetch(registered.id, tenant_id) is None
    async with engine.connect() as conn:
        assert (await conn.execute(_REVISIONS, {"id": registered.id})).all() == []
        assert await conn.scalar(_ARCHIVE_ROWS, {"id": registered.id}) == 0


async def test_a_delete_of_an_absent_or_foreign_id_removes_nothing(
    engine: AsyncEngine,
) -> None:
    """False for both, and the second half is the one worth the round trip: a delete
    that ignored the tenant would let anybody holding an id remove somebody else's
    Environment."""
    store = PostgresEnvironmentStore(engine)
    owner = TenantId(uuid.uuid4())
    theirs = _a_shape(new_environment_id(), owner)
    await store.insert(theirs)

    assert await store.delete(theirs.id, TenantId(uuid.uuid4())) is False
    assert await store.delete(new_environment_id(), owner) is False
    assert await store.fetch(theirs.id, owner) is not None


async def test_a_page_answers_one_row_per_id_newest_first(engine: AsyncEngine) -> None:
    """The listing statement, executed. Three things here run nowhere else.

    The grouped subquery that picks the latest revision per id, the bound boolean in the
    WHERE clause, and the row-constructor keyset are all SQL: a fake reproduces their
    intent and cannot fail on their syntax. The keyset in particular reaches the driver
    as a comparison of two composite values, which is the shape SQLAlchemy's text()
    parameter rules make easiest to get wrong.
    """
    store = PostgresEnvironmentStore(engine)
    tenant_id = TenantId(uuid.uuid4())
    made = []
    for index in range(3):
        registered = _a_shape(new_environment_id(), tenant_id, name=f"shape-{index}")
        await store.insert(registered)
        made.append(registered)
    await store.insert_revision(_a_shape(made[0].id, tenant_id, name="edited"))

    whole = await store.page(tenant_id, None, 10, False)

    assert len(whole) == 3, "an edit added a row to the list instead of changing one"
    assert [str(row["id"]) for row in whole] == [str(one.id) for one in reversed(made)]
    edited = next(row for row in whole if str(row["id"]) == str(made[0].id))
    assert (edited["revision"], edited["name"]) == (2, "edited")

    # The walk resumes after the first row and covers the rest exactly once.
    first = await store.page(tenant_id, None, 1, False)
    position = (int(str(first[0]["created_at_ms"])), uuid.UUID(str(first[0]["id"])))
    rest = await store.page(tenant_id, position, 10, False)
    assert [str(row["id"]) for row in first] + [str(row["id"]) for row in rest] == [
        str(row["id"]) for row in whole
    ]


async def test_a_page_hides_a_retired_environment_unless_asked(
    engine: AsyncEngine,
) -> None:
    """The bound boolean, both ways round. One direction alone would pass for a filter
    that ignored the parameter and always did the same thing."""
    store = PostgresEnvironmentStore(engine)
    tenant_id = TenantId(uuid.uuid4())
    live = _a_shape(new_environment_id(), tenant_id, name="live")
    retired = _a_shape(new_environment_id(), tenant_id, name="retired")
    await store.insert(live)
    await store.insert(retired)
    await store.archive(retired.id, tenant_id)

    hidden = await store.page(tenant_id, None, 10, False)
    shown = await store.page(tenant_id, None, 10, True)

    assert [str(row["id"]) for row in hidden] == [str(live.id)]
    assert {str(row["id"]) for row in shown} == {str(live.id), str(retired.id)}
    was_retired = next(row for row in shown if str(row["id"]) == str(retired.id))
    assert isinstance(was_retired["archived_at"], datetime)


async def test_a_page_does_not_carry_another_tenants_environments(
    engine: AsyncEngine,
) -> None:
    store = PostgresEnvironmentStore(engine)
    mine, theirs = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())
    yours = _a_shape(new_environment_id(), mine)
    await store.insert(yours)
    await store.insert(_a_shape(new_environment_id(), theirs))

    assert [str(row["id"]) for row in await store.page(mine, None, 10, False)] == [
        str(yours.id)
    ]


@pytest.mark.parametrize("limit", [0, 501])
async def test_a_page_outside_the_bound_is_refused_not_clamped(
    engine: AsyncEngine, limit: int
) -> None:
    """A clamped page is a short page, and a short page is how this port says the walk
    is over -- so a caller reading one would stop early with no sign that it had."""
    store = PostgresEnvironmentStore(engine)

    with pytest.raises(ValueError, match="outside 1..500"):
        await store.page(TenantId(uuid.uuid4()), None, limit, False)


async def test_the_count_of_holders_is_the_sessions_that_have_not_stopped(
    engine: AsyncEngine,
) -> None:
    """The number a delete is refused on, read out of the Event Log.

    Three Sessions on one Environment: two running, one stopped. The stopped one is
    excluded because what the guard protects is a Session that can still be resumed into
    this shape, and a Session that has stopped cannot be.
    """
    store = PostgresEnvironmentStore(engine)
    tenant_id = TenantId(uuid.uuid4())
    held = _a_shape(new_environment_id(), tenant_id)
    untouched = _a_shape(new_environment_id(), tenant_id)
    await store.insert(held)
    await store.insert(untouched)
    await _created_in(engine, held.id, stopped=False)
    await _created_in(engine, held.id, stopped=False)
    await _created_in(engine, held.id, stopped=True)

    assert await store.sessions_referencing(held.id) == 2
    assert await store.sessions_referencing(untouched.id) == 0
