"""Retiring an agent: what the store guarantees, and what the adapter reads back.

Tier 1 (testcontainers, real PostgreSQL 17). Two halves, asserted through two different
doors on purpose.

The first half is raw SQL against `agent_archive`. Those claims are about the *table* --
one row per agent, no rewriting, no tenant column -- and the point of each is that the
guarantee survives a writer that never loads our code: a psql session, a later slice's
adapter, a migration somebody writes in a hurry. Asserting them through the adapter
would only ever prove the adapter agrees with itself.

The second half drives `PostgresDefinitionRegistry` against the same database. Those
claims are about the *statements* -- that the fold really takes the newest revision and
the earliest timestamp, that the keyset boundary neither repeats a row nor drops one,
that a repeat archive comes back with the original moment, that the optimistic-
concurrency `HAVING` refuses a stale number without writing. None of those is a property
of PostgreSQL a fake could stand in for, and none is visible from the route tests, which
run over an in-memory store.

`agent_archive` is deliberately distinct from `agent_version_archive` beside it. That
one retires a single revision so no new Session resolves to it while its siblings stay
live; this one retires the agent, terminally. The sibling file
`test_agent_version_archive_schema.py` grades the first table, this one grades the
second, and neither substitutes for the other.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.definition_registry import (
    PostgresDefinitionRegistry,
)
from managed_agent.control.catalog.definitions import agent_lifecycle_of
from managed_agent.core.ids import DefinitionId, TenantId
from managed_agent.core.registration.definition import AgentDefinition

_SHA = "0" * 39 + "a"
_REPOSITORY = "git@github.com:acme/skills.git"

_REGISTER = sa.text(
    "INSERT INTO agent_definition (id, tenant_id, revision, body, skills_revision)"
    " VALUES (:id, :tenant, :revision, :body, :skills)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("body", type_=sa.JSON()),
)

_ARCHIVE = sa.text("INSERT INTO agent_archive (definition_id) VALUES (:id)").bindparams(
    sa.bindparam("id", type_=sa.Uuid())
)

_COUNT = sa.text(
    "SELECT count(*) FROM agent_archive WHERE definition_id = :id"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_MOMENT = sa.text(
    "SELECT archived_at FROM agent_archive WHERE definition_id = :id"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_EARLIEST = sa.text(
    "SELECT min(registered_at) FROM agent_definition WHERE id = :id"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))


def a_definition(name: str) -> AgentDefinition:
    return AgentDefinition.model_validate(
        {
            "name": name,
            "instructions": "read the diff before the plan",
            "model": "gpt-5-codex",
            "skills_repository": _REPOSITORY,
            "skills_revision": _SHA,
        }
    )


async def _register_raw(
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


async def _archive_raw(engine: AsyncEngine, definition_id: uuid.UUID) -> None:
    async with engine.begin() as conn:
        await conn.execute(_ARCHIVE, {"id": definition_id})


async def _archived_rows(engine: AsyncEngine, definition_id: uuid.UUID) -> int:
    async with engine.connect() as conn:
        return int(await conn.scalar(_COUNT, {"id": definition_id}) or 0)


async def _archived_moment(
    engine: AsyncEngine, definition_id: uuid.UUID
) -> datetime | None:
    async with engine.connect() as conn:
        found = await conn.scalar(_MOMENT, {"id": definition_id})
    assert found is None or isinstance(found, datetime)
    return found


# -- the table's own guarantees, asserted without our code -------------------


async def test_retiring_an_agent_twice_is_one_row_and_a_constraint_violation(
    engine: AsyncEngine,
) -> None:
    """The key is the agent, so a repeat is refused rather than counted twice.

    That is what lets the adapter's `ON CONFLICT DO NOTHING` mean "already retired"
    instead of hiding a second retirement -- and it is why a retry can be handed the
    original timestamp: there is only ever one to hand back.
    """
    definition_id = uuid.uuid4()
    await _register_raw(engine, definition_id, 1)
    await _archive_raw(engine, definition_id)

    with pytest.raises(IntegrityError):
        await _archive_raw(engine, definition_id)

    assert await _archived_rows(engine, definition_id) == 1


async def test_the_key_is_the_agent_and_not_one_of_its_revisions(
    engine: AsyncEngine,
) -> None:
    """One row retires every revision at once, which is what makes it terminal.

    A caller could approximate this by retiring each revision in
    `agent_version_archive` one at a time, and the result would not be the same thing:
    a later `POST /v1/agents/{id}/versions` would succeed and the agent would be live
    again. Retirement has to be a fact about the agent, so the key is the agent -- and
    a second row keyed on a revision is not expressible here at all.
    """
    definition_id = uuid.uuid4()
    for revision in (1, 2, 3):
        await _register_raw(engine, definition_id, revision)
    await _archive_raw(engine, definition_id)

    assert await _archived_rows(engine, definition_id) == 1


async def test_an_update_to_a_retirement_raises_rather_than_being_ignored(
    engine: AsyncEngine,
) -> None:
    """Refused loudly, which is the mechanism migration 0001 settled for this tree.

    Here it is also what makes the archive terminal: with no unarchive to build, the
    only way back is to rewrite or delete this row, and the first of those now raises.
    Both halves are asserted -- the raise, and that the moment did not move.
    """
    definition_id = uuid.uuid4()
    await _register_raw(engine, definition_id, 1)
    await _archive_raw(engine, definition_id)
    before = await _archived_moment(engine, definition_id)

    with pytest.raises(DBAPIError) as raised:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE agent_archive SET archived_at = now()"
                    " WHERE definition_id = :id"
                ).bindparams(sa.bindparam("id", type_=sa.Uuid())),
                {"id": definition_id},
            )

    assert "append-only" in str(raised.value)
    assert await _archived_moment(engine, definition_id) == before


async def test_the_archive_carries_no_tenant_column(engine: AsyncEngine) -> None:
    """Which tenant owns an agent is `agent_definition`'s fact and is not copied.

    A copy here would be free to disagree with the rows it describes, and the read that
    matters -- "may this tenant retire this agent" -- is answered by selecting the key
    out of `agent_definition` under a tenant predicate instead.
    """
    async with engine.connect() as conn:
        columns = set(
            (
                await conn.execute(
                    sa.text(
                        "SELECT column_name FROM information_schema.columns"
                        " WHERE table_name = 'agent_archive'"
                    )
                )
            )
            .scalars()
            .all()
        )

    assert columns == {"definition_id", "archived_at"}


async def test_the_table_holds_no_foreign_key_so_existence_is_the_routes_job(
    engine: AsyncEngine,
) -> None:
    """Asserted because a reader will look for the constraint and needs to know why not.

    `agent_version_archive` references `agent_definition(id, revision)`, which works
    because that pair is that table's primary key. This table's key is `id` alone, and
    `id` alone is not unique there -- holding several revisions of one agent is the
    whole point of it -- so there is no unique target a reference could name. The
    consequence is real and is what the routes carry: this store accepts a retirement
    of an id nobody registered, so the check that it exists has to run before the
    insert.
    """
    async with engine.connect() as conn:
        references = (
            (
                await conn.execute(
                    sa.text(
                        "SELECT count(*) FROM information_schema.table_constraints"
                        " WHERE table_name = 'agent_archive'"
                        " AND constraint_type = 'FOREIGN KEY'"
                    )
                )
            )
            .scalars()
            .one()
        )

    assert references == 0

    unregistered = uuid.uuid4()
    await _archive_raw(engine, unregistered)
    assert await _archived_rows(engine, unregistered) == 1


# -- the adapter's statements, against the real database ---------------------


async def test_the_wired_registry_answers_the_whole_agent_reads(
    engine: AsyncEngine,
) -> None:
    """The narrowing the routes perform, against the object composition wires.

    `agent_lifecycle_of` checks method names on a registry typed as the revision port.
    It is the only thing standing between a registry that never grew these methods and
    a 500 on the first request, and it is checked here because nothing else compares
    the concrete adapter against that protocol.
    """
    assert agent_lifecycle_of(PostgresDefinitionRegistry(engine)) is not None


async def test_an_agent_folds_to_its_newest_shape_and_its_earliest_moment(
    engine: AsyncEngine,
) -> None:
    """The two aggregates that turn a stack of revisions back into one agent.

    `version` is the newest revision and `created_at` is the earliest timestamp, and
    getting either from the wrong row is invisible at a glance: a `created_at` taken
    from the newest revision reads as a perfectly plausible date and means "last
    edited".
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id = DefinitionId(uuid.uuid4())
    tenant_id = TenantId(uuid.uuid4())
    await registry.register(definition_id, tenant_id, a_definition("first"))
    await registry.register(definition_id, tenant_id, a_definition("second"))
    await registry.register(definition_id, tenant_id, a_definition("third"))

    record = await registry.read_agent(definition_id, tenant_id)

    assert record is not None
    assert record.version == 3
    assert record.definition.name == "third"
    assert record.archived_at is None
    first_body = await registry.read_version(definition_id, tenant_id, 1)
    assert first_body is not None
    assert first_body.name == "first"
    async with engine.connect() as conn:
        earliest = await conn.scalar(_EARLIEST, {"id": definition_id})
    assert record.created_at == earliest


async def test_another_tenants_agent_is_absent_from_the_read(
    engine: AsyncEngine,
) -> None:
    """The tenant is a term in the query, so the row is never fetched to be dropped."""
    registry = PostgresDefinitionRegistry(engine)
    definition_id = DefinitionId(uuid.uuid4())
    await registry.register(definition_id, TenantId(uuid.uuid4()), a_definition("x"))

    assert await registry.read_agent(definition_id, TenantId(uuid.uuid4())) is None


async def test_a_page_walks_the_tenants_agents_exactly_once(
    engine: AsyncEngine,
) -> None:
    """The keyset boundary, against a real index rather than a sorted list.

    Walked one row at a time so the boundary is exercised at every position: written
    inclusively it repeats a row, written past the end it drops one, and both look like
    a working listing until the pages are laid end to end.
    """
    registry = PostgresDefinitionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    minted = [DefinitionId(uuid.uuid4()) for _ in range(5)]
    for definition_id in minted:
        await registry.register(definition_id, tenant_id, a_definition("agent"))

    walked: list[DefinitionId] = []
    after: tuple[datetime, DefinitionId] | None = None
    for _ in range(len(minted) + 2):
        page = await registry.page_agents(
            tenant_id,
            include_archived=False,
            created_from=None,
            created_to=None,
            after=after,
            limit=1,
        )
        if not page:
            break
        walked.append(page[-1].definition_id)
        after = (page[-1].created_at, page[-1].definition_id)

    assert sorted(walked) == sorted(minted)
    assert len(walked) == len(minted)


async def test_a_page_holds_no_other_tenants_agents(engine: AsyncEngine) -> None:
    registry = PostgresDefinitionRegistry(engine)
    mine, theirs = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())
    ours = DefinitionId(uuid.uuid4())
    await registry.register(ours, mine, a_definition("mine"))
    await registry.register(DefinitionId(uuid.uuid4()), theirs, a_definition("theirs"))

    page = await registry.page_agents(
        mine,
        include_archived=False,
        created_from=None,
        created_to=None,
        after=None,
        limit=10,
    )

    assert [record.definition_id for record in page] == [ours]


async def test_a_retired_agent_leaves_the_page_and_returns_when_asked(
    engine: AsyncEngine,
) -> None:
    """Both halves, because either alone is satisfied by a filter that does nothing."""
    registry = PostgresDefinitionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    live = DefinitionId(uuid.uuid4())
    retired = DefinitionId(uuid.uuid4())
    await registry.register(live, tenant_id, a_definition("live"))
    await registry.register(retired, tenant_id, a_definition("retired"))
    await registry.archive_agent(retired, tenant_id)

    hidden = await registry.page_agents(
        tenant_id,
        include_archived=False,
        created_from=None,
        created_to=None,
        after=None,
        limit=10,
    )
    shown = await registry.page_agents(
        tenant_id,
        include_archived=True,
        created_from=None,
        created_to=None,
        after=None,
        limit=10,
    )

    assert [record.definition_id for record in hidden] == [live]
    assert {record.definition_id for record in shown} == {live, retired}
    retirement = next(
        record.archived_at for record in shown if record.definition_id == retired
    )
    assert retirement is not None


async def test_the_created_at_bounds_are_inclusive_at_both_ends(
    engine: AsyncEngine,
) -> None:
    """Inclusive, and asserted on the boundary values themselves.

    A bound written with `>` instead of `>=` drops exactly the row whose timestamp the
    caller quoted -- which is the row a caller paging by time is most likely to have
    just read and asked to start from.
    """
    registry = PostgresDefinitionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    minted = [DefinitionId(uuid.uuid4()) for _ in range(3)]
    for definition_id in minted:
        await registry.register(definition_id, tenant_id, a_definition("agent"))
    everything = await registry.page_agents(
        tenant_id,
        include_archived=False,
        created_from=None,
        created_to=None,
        after=None,
        limit=10,
    )
    moments = sorted(record.created_at for record in everything)

    bounded = await registry.page_agents(
        tenant_id,
        include_archived=False,
        created_from=moments[0],
        created_to=moments[-1],
        after=None,
        limit=10,
    )
    narrowed = await registry.page_agents(
        tenant_id,
        include_archived=False,
        created_from=moments[0] + timedelta(microseconds=1),
        created_to=moments[-1] - timedelta(microseconds=1),
        after=None,
        limit=10,
    )

    assert len(bounded) == 3
    assert len(narrowed) == 1


async def test_retiring_an_agent_reports_the_moment_and_a_repeat_reports_the_first(
    engine: AsyncEngine,
) -> None:
    """One statement writes and reads, and a retry gets the original moment back.

    A fresh timestamp would say the agent was retired at the moment of the retry, which
    is a different and false fact about when anything referencing it stopped resolving.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id = DefinitionId(uuid.uuid4())
    tenant_id = TenantId(uuid.uuid4())
    await registry.register(definition_id, tenant_id, a_definition("retiring"))

    first = await registry.archive_agent(definition_id, tenant_id)
    second = await registry.archive_agent(definition_id, tenant_id)

    assert first is not None
    assert second == first
    assert await _archived_rows(engine, definition_id) == 1


async def test_retiring_another_tenants_agent_writes_nothing_and_reports_nothing(
    engine: AsyncEngine,
) -> None:
    """Scoped by the select that feeds the insert, so there is no branch to forget."""
    registry = PostgresDefinitionRegistry(engine)
    definition_id = DefinitionId(uuid.uuid4())
    await registry.register(
        definition_id, TenantId(uuid.uuid4()), a_definition("theirs")
    )

    assert await registry.archive_agent(definition_id, TenantId(uuid.uuid4())) is None
    assert await _archived_rows(engine, definition_id) == 0


async def test_retiring_an_agent_nobody_registered_reports_nothing(
    engine: AsyncEngine,
) -> None:
    """No foreign key protects this table, so the statement's own select has to.

    Without the tenant-scoped select feeding the insert, this call would write a row
    retiring an agent that does not exist -- and every later read joining against it
    would silently skip it.
    """
    registry = PostgresDefinitionRegistry(engine)
    unregistered = DefinitionId(uuid.uuid4())

    assert await registry.archive_agent(unregistered, TenantId(uuid.uuid4())) is None
    assert await _archived_rows(engine, unregistered) == 0


async def test_a_conditional_write_appends_only_while_the_version_matches(
    engine: AsyncEngine,
) -> None:
    """The optimistic-concurrency check and the write, in one statement.

    The stale attempt is asserted to have written nothing as well as to have reported
    nothing: a `HAVING` that let the insert through would answer with a revision number
    while the caller had been told its version was stale, and the two answers would
    disagree about whether the edit happened.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id = DefinitionId(uuid.uuid4())
    tenant_id = TenantId(uuid.uuid4())
    await registry.register(definition_id, tenant_id, a_definition("first"))

    applied = await registry.register_at_revision(
        definition_id, tenant_id, a_definition("second"), 1
    )
    stale = await registry.register_at_revision(
        definition_id, tenant_id, a_definition("third"), 1
    )

    assert applied == 2
    assert stale is None
    facts = await registry.list_versions(definition_id, tenant_id)
    assert [fact.revision for fact in facts] == [1, 2]


async def test_a_conditional_write_against_another_tenants_agent_writes_nothing(
    engine: AsyncEngine,
) -> None:
    """The same `HAVING` refuses both a stale number and a foreign id.

    Over no rows `max(revision)` is NULL and `NULL = 1` is NULL, so the clause is not
    satisfied. Both refusals fall out of one expression rather than out of a check
    somebody has to remember to write.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id = DefinitionId(uuid.uuid4())
    owner = TenantId(uuid.uuid4())
    await registry.register(definition_id, owner, a_definition("theirs"))

    written = await registry.register_at_revision(
        definition_id, TenantId(uuid.uuid4()), a_definition("hijacked"), 1
    )

    assert written is None
    facts = await registry.list_versions(definition_id, owner)
    assert [fact.revision for fact in facts] == [1]


async def test_a_conditional_write_against_an_unregistered_id_writes_nothing(
    engine: AsyncEngine,
) -> None:
    """Otherwise a call that reads as an edit would create an agent at revision 1."""
    registry = PostgresDefinitionRegistry(engine)
    unregistered = DefinitionId(uuid.uuid4())
    tenant_id = TenantId(uuid.uuid4())

    written = await registry.register_at_revision(
        unregistered, tenant_id, a_definition("invented"), 1
    )

    assert written is None
    assert await registry.list_versions(unregistered, tenant_id) == ()


async def test_the_page_reports_the_retirement_moment_the_archive_wrote(
    engine: AsyncEngine,
) -> None:
    """The two statements read one table and must agree about what it says.

    A page that derived `archived_at` from anything but the archive row -- a boolean
    widened to a timestamp, the definition's own `registered_at` -- would still look
    like a date to every caller.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id = DefinitionId(uuid.uuid4())
    tenant_id = TenantId(uuid.uuid4())
    await registry.register(definition_id, tenant_id, a_definition("retiring"))
    written = await registry.archive_agent(definition_id, tenant_id)

    page = await registry.page_agents(
        tenant_id,
        include_archived=True,
        created_from=None,
        created_to=None,
        after=None,
        limit=10,
    )
    read = await registry.read_agent(definition_id, tenant_id)

    assert read is not None
    assert written is not None
    assert page[0].archived_at == written
    assert read.archived_at == written
    assert written.tzinfo is not None
    assert written > datetime(2020, 1, 1, tzinfo=UTC)
