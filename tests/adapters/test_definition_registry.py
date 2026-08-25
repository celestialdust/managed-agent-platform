"""Registering a definition, and resolving the revision a Session will pin.

Tier 1 (testcontainers, real PostgreSQL 17), against the adapter rather than raw SQL --
the properties here are the adapter's: which revision a register lands on, which one a
resolve comes back with, and whether another tenant's definition is visible.

Real Postgres and not a fake, because two of these are properties of the database and
not of any code we could stand in for it: the primary key that makes the second
registration a new revision instead of an overwrite, and the round trip of a JSON
document through a `jsonb` column and back into a model. A fake dict keyed by id would
pass a version of every test here while telling us nothing about either.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.definition_registry import (
    PostgresDefinitionRegistry,
)
from managed_agent.core.ids import DefinitionId, TenantId
from managed_agent.core.ports import UnknownDefinition
from managed_agent.core.registration.definition import (
    AgentDefinition,
    MultiAgentPosture,
)

_SHA = "0" * 39 + "a"
_OTHER_SHA = "f" * 39 + "e"


def _definition(name: str = "slr-reviewer", revision: str = _SHA) -> AgentDefinition:
    return AgentDefinition(
        name=name,
        instructions="Extract findings and name the source document for each.",
        model="gpt-5-codex",
        skills_repository="git@github.com:acme/skills.git",
        skills_revision=revision,
        tool_servers=frozenset({"crossref", "openalex"}),
        multiagent=MultiAgentPosture(enabled=True, max_depth=2),
    )


async def test_the_first_registration_is_revision_one(engine: AsyncEngine) -> None:
    registry = PostgresDefinitionRegistry(engine)

    revision = await registry.register(
        DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4()), _definition()
    )

    assert revision == 1, (
        f"the first registration landed on revision {revision}; a Session pinning it "
        "would record a revision that never existed"
    )


async def test_registering_the_same_id_again_makes_the_next_revision(
    engine: AsyncEngine,
) -> None:
    """The number the adapter returns is the one it actually wrote, not a guess.

    Returned rather than derived by the caller, because a caller that counted its own
    registrations would be right until anything else registered -- and the number is
    what a Session pins.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id, tenant_id = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())

    first = await registry.register(definition_id, tenant_id, _definition("v1"))
    second = await registry.register(definition_id, tenant_id, _definition("v2"))
    third = await registry.register(definition_id, tenant_id, _definition("v3"))

    assert (first, second, third) == (1, 2, 3)


async def test_resolve_returns_the_latest_revision_and_its_own_body(
    engine: AsyncEngine,
) -> None:
    """The latest body, paired with the number that identifies it.

    Both together, because either alone is a half-answer: a body with no number cannot
    be pinned, and a number with no body cannot be run.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id, tenant_id = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    await registry.register(definition_id, tenant_id, _definition("v1"))
    await registry.register(definition_id, tenant_id, _definition("v2", _OTHER_SHA))

    resolved = await registry.resolve(definition_id, tenant_id)

    assert resolved.revision == 2
    assert resolved.definition.name == "v2"
    assert resolved.definition.skills_revision == _OTHER_SHA, (
        "the resolved body is not the one the latest revision stored; a Session would "
        "pin revision 2 and run revision 1's skills"
    )


async def test_the_resolved_body_round_trips_to_an_equal_definition(
    engine: AsyncEngine,
) -> None:
    """Everything the tenant submitted survives the store, in the same types.

    The types are the half that quietly breaks. `tool_servers` is a frozenset and JSON
    has only arrays; `multiagent` is a nested model and JSON has only objects. A round
    trip that returned a list and a dict would compare unequal here, and would compare
    *equal* in any test that only checked a couple of scalar fields.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id, tenant_id = DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4())
    submitted = _definition()
    await registry.register(definition_id, tenant_id, submitted)

    resolved = await registry.resolve(definition_id, tenant_id)

    assert resolved.definition == submitted, (
        f"the stored definition came back different: {resolved.definition!r} vs "
        f"{submitted!r}"
    )
    assert isinstance(resolved.definition.tool_servers, frozenset)
    assert resolved.definition.multiagent == MultiAgentPosture(
        enabled=True, max_depth=2
    )


async def test_another_tenants_definition_is_not_resolvable(
    engine: AsyncEngine,
) -> None:
    """A definition another tenant registered is refused, not returned.

    The id is a uuid and so unguessable, but unguessable is not a boundary -- an id
    leaks through a log line, a support ticket, a screenshot. The tenant filter is the
    boundary, and this is the test that it is applied on the read and not only assumed
    from the write.
    """
    registry = PostgresDefinitionRegistry(engine)
    definition_id = DefinitionId(uuid.uuid4())
    await registry.register(definition_id, TenantId(uuid.uuid4()), _definition())

    with pytest.raises(UnknownDefinition):
        await registry.resolve(definition_id, TenantId(uuid.uuid4()))


async def test_an_id_that_was_never_registered_is_refused(
    engine: AsyncEngine,
) -> None:
    """Refused rather than answered with an empty definition.

    A default-shaped definition returned here would start a Session with no
    instructions and no skills pin, and the tenant would see an agent that does nothing
    rather than a refusal naming what is missing.
    """
    registry = PostgresDefinitionRegistry(engine)

    with pytest.raises(UnknownDefinition):
        await registry.resolve(DefinitionId(uuid.uuid4()), TenantId(uuid.uuid4()))


async def test_two_tenants_may_hold_definitions_with_the_same_name(
    engine: AsyncEngine,
) -> None:
    """Names are a tenant's own vocabulary, so two tenants naming an agent the same
    thing is normal and neither registration disturbs the other."""
    registry = PostgresDefinitionRegistry(engine)
    one, two = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())
    one_id, two_id = DefinitionId(uuid.uuid4()), DefinitionId(uuid.uuid4())

    await registry.register(one_id, one, _definition("shared-name"))
    await registry.register(two_id, two, _definition("shared-name", _OTHER_SHA))

    assert (await registry.resolve(one_id, one)).definition.skills_revision == _SHA
    assert (
        await registry.resolve(two_id, two)
    ).definition.skills_revision == _OTHER_SHA
