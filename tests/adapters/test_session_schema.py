"""The store, not the application, is what refuses an illegal Session row.

Tier 1 (testcontainers, real PostgreSQL 17). Asserted with raw SQL rather than through
the registry adapter, because the point of each of these is that the guarantee survives
a writer that never loads our code -- a psql session, a later slice's adapter, a
migration somebody writes in a hurry.

Every column here is a fact fixed when the Session was created, which is what makes the
update rule cost nothing and what makes it necessary: a row that could be rewritten
would let the record of what a Session *is* drift from the log of what it did.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

# Nothing here is a default. A budget of 500 or a retention of 30 would be
# indistinguishable from the values the rest of the suite happens to use, so a read that
# returned somebody else's row, or a constant, would still look right.
_BUDGET = 731
_RETENTION = 17
_REVISION = "4"

_INSERT = sa.text(
    "INSERT INTO session"
    " (id, tenant_id, definition_id, definition_revision, grant_tools, scope,"
    "  budget_minor_units, budget_currency, retention_days)"
    " VALUES (:id, :tenant, :definition, :revision, :grant, :scope,"
    "  :budget, :currency, :retention)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("definition", type_=sa.Uuid()),
    sa.bindparam("grant", type_=sa.JSON()),
    sa.bindparam("scope", type_=sa.JSON()),
)

_INSERT_AT = sa.text(
    "INSERT INTO session"
    " (id, tenant_id, definition_id, definition_revision, grant_tools, scope,"
    "  budget_minor_units, budget_currency, retention_days, created_at_ms)"
    " VALUES (:id, :tenant, :definition, :revision, :grant, :scope,"
    "  :budget, :currency, :retention, :created_at_ms)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("definition", type_=sa.Uuid()),
    sa.bindparam("grant", type_=sa.JSON()),
    sa.bindparam("scope", type_=sa.JSON()),
)


async def _write(
    engine: AsyncEngine, tenant_id: uuid.UUID, **overrides: Any
) -> uuid.UUID:
    """Insert one row, overriding any column by keyword, and return its id.

    `created_at_ms` selects the second statement rather than being one more override,
    because the column's whole point is that the store fills it in -- naming it is the
    exception a test makes on purpose when it needs a chosen position in the ordering.
    """
    written = uuid.uuid4()
    row: dict[str, Any] = {
        "id": written,
        "tenant": tenant_id,
        "definition": uuid.uuid4(),
        "revision": _REVISION,
        "grant": ["fs.read", "web.fetch"],
        "scope": {"repository": "acme/widgets"},
        "budget": _BUDGET,
        "currency": "EUR",
        "retention": _RETENTION,
    }
    row.update(overrides)
    statement = _INSERT_AT if "created_at_ms" in row else _INSERT
    async with engine.begin() as conn:
        await conn.execute(statement, row)
    return written


async def test_a_row_is_written_without_the_writer_supplying_a_creation_time(
    engine: AsyncEngine,
) -> None:
    """The ordering key is the store's answer, not the caller's.

    A caller-supplied timestamp is a caller-chosen position in the list, and two callers
    disagreeing about the clock would interleave their Sessions wrongly for everyone. It
    is also what lets the insert statement stay one column shorter than the table.
    """
    tenant_id = uuid.uuid4()
    session_id = await _write(engine, tenant_id)

    async with engine.connect() as conn:
        created_at_ms = await conn.scalar(
            sa.text("SELECT created_at_ms FROM session WHERE id = :id").bindparams(
                sa.bindparam("id", type_=sa.Uuid())
            ),
            {"id": session_id},
        )

    assert isinstance(created_at_ms, int)
    # Epoch milliseconds, not seconds and not microseconds. 1e12 ms is 2001 and 1e13 ms
    # is 2286, so any clock this decade lands between them in exactly one of the three
    # units -- which is the mistake worth catching, because a unit error only shows up
    # as a list in a plausible but wrong order.
    assert 10**12 < created_at_ms < 10**13, (
        f"created_at_ms is {created_at_ms}, which is not epoch milliseconds; a "
        "seconds-or-microseconds value still sorts, so the list would look fine and be "
        "ordered against a different clock than the cursor's"
    )


@pytest.mark.parametrize(
    ("column", "value", "constraint"),
    [
        ("budget", 0, "session_budget_positive"),
        ("budget", -1, "session_budget_positive"),
        ("retention", 0, "session_retention_positive"),
        ("retention", -1, "session_retention_positive"),
    ],
)
async def test_a_non_positive_budget_or_retention_is_refused_by_the_store(
    engine: AsyncEngine, column: str, value: int, constraint: str
) -> None:
    """Neither is a Session that does less; both are a Session nobody meant to create.

    A zero budget buys no Turn, and a zero retention expires the log as it is written --
    so each would produce a Session that exists, accepts a create call, and can never do
    the thing it was created for.
    """
    with pytest.raises(IntegrityError) as refused:
        await _write(engine, uuid.uuid4(), **{column: value})

    assert constraint in str(refused.value), str(refused.value)


@pytest.mark.parametrize(
    ("column", "value", "constraint"),
    [
        # A JSON `null`, which is what binding Python `None` through a `sa.JSON()`
        # parameter actually writes. This is the case `NOT NULL` does not cover, and the
        # one that motivated the two type checks: the insert succeeded before they
        # existed, and the row read back with `grant_tools` as Python `None`.
        ("grant", None, "session_grant_is_an_array"),
        ("grant", {"fs.read": True}, "session_grant_is_an_array"),
        ("grant", "fs.read", "session_grant_is_an_array"),
        ("scope", None, "session_scope_is_an_object"),
        ("scope", ["repository", "acme/widgets"], "session_scope_is_an_object"),
    ],
)
async def test_a_grant_or_scope_of_the_wrong_json_shape_is_refused(
    engine: AsyncEngine, column: str, value: object, constraint: str
) -> None:
    """A Grant is a list of names and a Scope is a mapping; anything else is refused.

    An absent Grant and an empty Grant are different, and only one is writable. An empty
    Grant is a Session that may call no tool, which is a legitimate thing to create. A
    null one is a Session whose Grant nobody decided, and reading it later forces every
    consumer to guess -- with "no tools" and "all tools" both being plausible guesses.

    `NOT NULL` does not draw that line on a jsonb column, which is the whole reason
    these constraints exist rather than being left to the nullability the column already
    declares.
    """
    with pytest.raises(IntegrityError) as refused:
        await _write(engine, uuid.uuid4(), **{column: value})

    assert constraint in str(refused.value), str(refused.value)


async def test_a_row_with_a_sql_null_grant_is_refused(engine: AsyncEngine) -> None:
    """The nullability half, which the type checks above do not cover.

    A check constraint is not evaluated against SQL NULL -- it evaluates to unknown and
    the row is admitted -- so `jsonb_typeof(grant_tools) = 'array'` would let a NULL
    through on its own. The two constraints and the `NOT NULL` each close a case the
    others do not.
    """
    with pytest.raises(IntegrityError) as refused:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO session"
                    " (id, tenant_id, definition_id, definition_revision, grant_tools,"
                    "  scope, budget_minor_units, budget_currency, retention_days)"
                    " VALUES (:id, :tenant, :definition, :revision, NULL, :scope,"
                    "  :budget, :currency, :retention)"
                ).bindparams(
                    sa.bindparam("id", type_=sa.Uuid()),
                    sa.bindparam("tenant", type_=sa.Uuid()),
                    sa.bindparam("definition", type_=sa.Uuid()),
                    sa.bindparam("scope", type_=sa.JSON()),
                ),
                {
                    "id": uuid.uuid4(),
                    "tenant": uuid.uuid4(),
                    "definition": uuid.uuid4(),
                    "revision": _REVISION,
                    "scope": {},
                    "budget": _BUDGET,
                    "currency": "EUR",
                    "retention": _RETENTION,
                },
            )

    assert "grant_tools" in str(refused.value), str(refused.value)


async def test_an_empty_grant_is_writable_and_reads_back_as_empty(
    engine: AsyncEngine,
) -> None:
    """The other half of the test above, and without it that one proves nothing.

    A schema that refused *every* Grant would satisfy the null case perfectly.
    """
    tenant_id = uuid.uuid4()
    session_id = await _write(engine, tenant_id, grant=[])

    async with engine.connect() as conn:
        stored = await conn.scalar(
            sa.text("SELECT grant_tools FROM session WHERE id = :id").bindparams(
                sa.bindparam("id", type_=sa.Uuid())
            ),
            {"id": session_id},
        )

    assert stored == []


async def test_an_update_raises_rather_than_quietly_changing_nothing(
    engine: AsyncEngine,
) -> None:
    """Refused, not absorbed -- and the row is unchanged afterwards.

    A rewrite rule doing nothing would satisfy the second half and fail the first,
    telling a writer its edit succeeded while the stored row stayed put. Both halves are
    asserted because either alone admits the wrong implementation, and the two
    mechanisms differ in nothing else.
    """
    tenant_id = uuid.uuid4()
    session_id = await _write(engine, tenant_id)

    with pytest.raises(DBAPIError) as refused:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE session SET budget_minor_units = 1 WHERE id = :id"
                ).bindparams(sa.bindparam("id", type_=sa.Uuid())),
                {"id": session_id},
            )

    assert "append-only" in str(refused.value), str(refused.value)

    async with engine.connect() as conn:
        unchanged = await conn.scalar(
            sa.text("SELECT budget_minor_units FROM session WHERE id = :id").bindparams(
                sa.bindparam("id", type_=sa.Uuid())
            ),
            {"id": session_id},
        )

    assert unchanged == _BUDGET


async def test_two_sessions_may_share_a_creation_millisecond(
    engine: AsyncEngine,
) -> None:
    """The ordering key is not unique, which is why the id is in it.

    A unique constraint on `(tenant_id, created_at_ms)` would make this a store that
    refuses two Sessions created in the same millisecond -- and 50 concurrent creates
    is the capacity plan, so it would refuse them routinely.
    """
    tenant_id = uuid.uuid4()

    first = await _write(engine, tenant_id, created_at_ms=1_700_000_000_007)
    second = await _write(engine, tenant_id, created_at_ms=1_700_000_000_007)

    async with engine.connect() as conn:
        stored = await conn.scalar(
            sa.text("SELECT count(*) FROM session WHERE id = ANY(:ids)").bindparams(
                sa.bindparam("ids", type_=sa.ARRAY(sa.Uuid()))
            ),
            {"ids": [first, second]},
        )

    assert stored == 2


async def test_the_live_table_has_no_state_pod_or_status_column(
    engine: AsyncEngine,
) -> None:
    """Checked against the running database, not against the migration source.

    `test_sessions.py` reads the migration files for the same property, and this is the
    other side of it: that check would pass a column added by a hand-run `ALTER TABLE`
    in an environment, and this one would not. State is a fold over the Event Log, so a
    column here is a second source free to disagree with it, and the only way to keep
    them from disagreeing is for the second one to have nowhere to live.
    """
    async with engine.connect() as conn:
        columns = (
            (
                await conn.execute(
                    sa.text(
                        "SELECT column_name FROM information_schema.columns"
                        " WHERE table_name = 'session'"
                    )
                )
            )
            .scalars()
            .all()
        )

    assert columns, "the session table reported no columns at all"
    offenders = [
        name
        for name in columns
        if any(word in name.lower() for word in ("state", "status", "pod"))
    ]
    assert offenders == [], f"the session table declares {offenders}"
