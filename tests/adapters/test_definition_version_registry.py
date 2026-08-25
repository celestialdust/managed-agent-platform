"""Listing, reading and retiring one agent definition's revisions, against real SQL.

Tier 1 (testcontainers, real PostgreSQL 17), against the adapter rather than raw SQL --
the properties here are the adapter's: whether the LEFT JOIN reports retirement, whether
the tenant predicate actually filters, and whether `ON CONFLICT DO NOTHING` makes a
repeat idempotent instead of a raised constraint the route would have to interpret.

Real Postgres and not a fake. Every one of those is a property of a statement and of the
tables it names -- a dict keyed by id would satisfy each call signature and could not
fail on the join, the predicate, or the conflict. A separate file from
`test_definition_registry.py` so the two slices' assertions do not interleave.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.definition_registry import (
    PostgresDefinitionRegistry,
)
from managed_agent.core.ids import DefinitionId, TenantId
from managed_agent.core.ports import DefinitionRegistry
from managed_agent.core.registration.definition import AgentDefinition, VersionFact

_SHA = "0" * 39 + "a"


def _definition(instructions: str) -> AgentDefinition:
    return AgentDefinition(
        name="slr-reviewer",
        instructions=instructions,
        model="gpt-5-codex",
        skills_repository="git@github.com:acme/skills.git",
        skills_revision=_SHA,
    )


def test_the_postgres_registry_satisfies_the_definition_registry_port() -> None:
    """Checked where the binding happens rather than at the first missing call.

    Shallow by design -- the runtime protocol check sees names, not signatures -- so it
    catches an adapter that never grew a method, and the tests below catch one that
    grew it wrong.
    """
    assert issubclass(PostgresDefinitionRegistry, DefinitionRegistry)


async def test_versions_come_back_ascending_with_only_the_retired_one_marked(
    engine: AsyncEngine,
) -> None:
    registry = PostgresDefinitionRegistry(engine)
    agent, tenant = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    await registry.register(agent, tenant, _definition("the first"))
    await registry.register(agent, tenant, _definition("the second"))

    before = await registry.list_versions(agent, tenant)
    await registry.archive_version(agent, tenant, 2)
    after = await registry.list_versions(agent, tenant)

    assert before == (VersionFact(1, archived=False), VersionFact(2, archived=False))
    assert after == (VersionFact(1, archived=False), VersionFact(2, archived=True))


async def test_another_tenants_versions_are_absent_rather_than_returned(
    engine: AsyncEngine,
) -> None:
    """The tenant is a term in the query, so the rows never leave the database."""
    registry = PostgresDefinitionRegistry(engine)
    agent, owner = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    await registry.register(agent, owner, _definition("mine"))

    assert await registry.list_versions(agent, TenantId(uuid.uuid4())) == ()
    assert await registry.list_versions(DefinitionId(uuid.uuid4()), owner) == ()


async def test_an_earlier_revisions_body_is_unchanged_after_a_later_one_is_written(
    engine: AsyncEngine,
) -> None:
    """The property the whole slice exists for, at the level where it is actually true.

    Read back through the model rather than by comparing a column, so a body that came
    back as JSON text instead of a mapping fails here rather than somewhere downstream.
    """
    registry = PostgresDefinitionRegistry(engine)
    agent, tenant = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    await registry.register(agent, tenant, _definition("the first"))
    await registry.register(agent, tenant, _definition("the second"))

    first = await registry.read_version(agent, tenant, 1)
    second = await registry.read_version(agent, tenant, 2)

    assert first == _definition("the first")
    assert second == _definition("the second")


async def test_reading_a_revision_that_is_not_there_or_not_yours_is_none(
    engine: AsyncEngine,
) -> None:
    registry = PostgresDefinitionRegistry(engine)
    agent, owner = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    await registry.register(agent, owner, _definition("mine"))

    assert await registry.read_version(agent, owner, 2) is None
    assert await registry.read_version(agent, TenantId(uuid.uuid4()), 1) is None


async def test_retiring_the_same_version_twice_reports_it_once_and_writes_one_row(
    engine: AsyncEngine,
) -> None:
    """`ON CONFLICT DO NOTHING` is what makes the second call an answer, not an error.

    Without it a caller retrying a retirement would get a raised unique violation and
    would have to read before writing to avoid it -- a race, in exchange for nothing.
    """
    registry = PostgresDefinitionRegistry(engine)
    agent, tenant = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    await registry.register(agent, tenant, _definition("mine"))

    assert await registry.archive_version(agent, tenant, 1) is True
    assert await registry.archive_version(agent, tenant, 1) is False
    assert await registry.list_versions(agent, tenant) == (
        VersionFact(1, archived=True),
    )


async def test_retiring_another_tenants_version_writes_nothing(
    engine: AsyncEngine,
) -> None:
    """The key is selected out of `agent_definition` under the tenant predicate.

    So a cross-tenant retirement inserts no row without a second round trip -- and the
    owner still sees a live version afterwards, which is the half that would be missing
    if the statement wrote first and checked later.
    """
    registry = PostgresDefinitionRegistry(engine)
    agent, owner = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    await registry.register(agent, owner, _definition("mine"))

    assert await registry.archive_version(agent, TenantId(uuid.uuid4()), 1) is False
    assert await registry.list_versions(agent, owner) == (
        VersionFact(1, archived=False),
    )
