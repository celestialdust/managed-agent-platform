"""Adding, listing and retiring the versions of one registered agent definition.

Three routes under `/v1/agents/{agent_id}`: a version is appended, the versions are
listed with their retirement state, and one is retired.

`POST /v1/agents` mints the id itself, so it can only ever write revision 1 and can
never be the way an edit lands. This router is the only path to revision N+1, and it
writes rather than rewrites: a Session that already resolved revision N goes on reading
exactly the bytes it resolved to.

Retirement is a refusal at the door of a *new* Session and nothing else. A Session
already running the retired revision is untouched by these routes — it resolved its
revision when it was created and reads the same immutable row it always did — which is
what lets a bad edit be withdrawn without stopping work already in flight.

Retiring twice is not an error. The second call reports that it retired nothing and
returns the same state as the first, because a caller retrying a retirement wants it
retired, and a refusal there would push every caller into a read-then-write race to
avoid it.

Retirement is addressed by version rather than by agent. What has to be expressible is
one revision being withdrawn while another stays live, so the number is in the path;
retiring a whole agent is retiring each of its versions and needs no second route to
mean that.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.catalog.definitions import agent_lifecycle_of
from managed_agent.control.skills.evaluation import (
    Standing,
    skill_evals_of,
    standing_of,
)
from managed_agent.control.skills.registry import SkillsUnresolvable, read_attached
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import DefinitionId, TenantId
from managed_agent.core.registration.definition import AgentDefinition

router = APIRouter(tags=["agent versions"])


class VersionCreated(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: DefinitionId
    version: int


class VersionListed(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    archived: bool


class VersionPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: DefinitionId
    versions: tuple[VersionListed, ...]


class VersionRetired(BaseModel):
    """`newly_archived` is false when the version was already retired before this
    call, which is a success rather than a refusal."""

    model_config = ConfigDict(frozen=True)

    id: DefinitionId
    version: int
    newly_archived: bool


_NOT_YOURS: dict[int | str, dict[str, Any]] = {
    STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]: {"model": PublicErrorEnvelope}
}
"""The one refusal all three routes share, declared once for the OpenAPI schema.

One code answers "no such agent", "no such version" and "not yours". Distinguishing
them would let anyone holding an id learn from the refusal whether it names another
tenant's agent.

Annotated because FastAPI's `responses` parameter is `dict[int | str, dict[str, Any]]`
and a bare literal assigned to a name infers a narrower type than that.
"""


_VERSION_REFUSALS: dict[int | str, dict[str, Any]] = {
    **_NOT_YOURS,
    STATUS_FOR[ErrorCode.DEFINITION_SKILLS_REVISION_NOT_ACCEPTED]: {
        "model": PublicErrorEnvelope
    },
    STATUS_FOR[ErrorCode.DEFINITION_INVALID]: {"model": PublicErrorEnvelope},
    STATUS_FOR[ErrorCode.AGENT_ARCHIVED]: {"model": PublicErrorEnvelope},
}
"""The four refusals the append route can give. Only this route consults the eval gate
or the skill store -- listing and retiring name neither a skills revision nor a skill,
so neither can be blocked by one."""


@router.post(
    "/agents/{agent_id}/versions",
    status_code=status.HTTP_201_CREATED,
    response_model=VersionCreated,
    responses=_VERSION_REFUSALS,
)
async def create_version(
    agent_id: DefinitionId,
    body: AgentDefinition,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> VersionCreated | JSONResponse:
    """Append the next revision of an agent this tenant owns.

    The existence check is not a nicety. The underlying insert numbers the new revision
    from whatever rows already carry this id, so without it an id nobody registered
    would be written as a brand-new agent at revision 1 — by a call that reads as an
    edit to an existing one, and answers 201 either way.

    Whether the body is well-formed is decided entirely by the `AgentDefinition`
    annotation: FastAPI answers 400 naming the offending field before this runs. A
    second check here would be a second answer to that question, free to disagree.

    The CI eval gate is consulted here as well as on `POST /v1/agents`, and it has to
    be: this route accepts a whole `AgentDefinition`, so it accepts a `skills_revision`,
    and a gate held on the registration door alone would be walked around by editing an
    agent instead of creating one. Two doors, one check -- and the check runs before the
    write, so a refusal leaves no revision behind.

    The attached skill ids are checked here for exactly that reason too. This route
    accepts a `skills` array, so a check held only on the registration door would be
    walked around by appending a version instead of creating an agent -- and the version
    that got through would name a skill that never reaches the agent.
    """
    platform = platform_from_request(request)
    registry = platform.definition_registry
    # One read answers both questions, and it replaced a `list_versions` call that
    # answered only the first. Two reads would have been two chances to disagree about
    # whether this agent exists, and the second question cannot be asked of
    # `list_versions` at all: what it reports is whether each individual REVISION was
    # retired, which is a different fact from the agent having been retired -- an agent
    # with no retired revisions can itself be archived, and every revision of a live
    # agent can be retired without the agent being.
    whole = await agent_lifecycle_of(registry).read_agent(agent_id, tenant_id)
    if whole is None:
        return refuse(
            ErrorCode.DEFINITION_NOT_FOUND,
            "no agent definition with that id is registered to this tenant",
            agent_id=str(agent_id),
        )
    if whole.archived_at is not None:
        return refuse(
            ErrorCode.AGENT_ARCHIVED,
            "this agent was retired, so no further version may be appended to it",
            agent_id=str(agent_id),
        )
    facts = await skill_evals_of(registry).eval_facts(
        tenant_id, body.skills_repository, body.skills_revision
    )
    if standing_of(facts) is Standing.BLOCKED:
        return refuse(
            ErrorCode.DEFINITION_SKILLS_REVISION_NOT_ACCEPTED,
            "this skills revision has not cleared its repository's CI eval "
            "baseline, so no agent may be pinned to it",
            field="skills_revision",
            skills_repository=body.skills_repository,
            skills_revision=body.skills_revision,
        )
    try:
        await read_attached(platform.skill_store, tenant_id, body)
    except SkillsUnresolvable as unresolvable:
        return refuse(
            ErrorCode.DEFINITION_INVALID,
            unresolvable.message,
            field="skills",
            **unresolvable.detail,
        )
    revision = await registry.register(agent_id, tenant_id, body)
    return VersionCreated(id=agent_id, version=revision)


@router.get(
    "/agents/{agent_id}/versions",
    response_model=VersionPage,
    responses=_NOT_YOURS,
)
async def list_versions(
    agent_id: DefinitionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> VersionPage | JSONResponse:
    """Every revision of one agent, oldest first, each saying whether it is retired.

    The read half of a versioned resource. Retirement is addressed by number, and a
    surface that could only be written blind would leave a tenant guessing which number
    to send.
    """
    registry = platform_from_request(request).definition_registry
    facts = await registry.list_versions(agent_id, tenant_id)
    if not facts:
        return refuse(
            ErrorCode.DEFINITION_NOT_FOUND,
            "no agent definition with that id is registered to this tenant",
            agent_id=str(agent_id),
        )
    return VersionPage(
        id=agent_id,
        versions=tuple(
            VersionListed(version=fact.revision, archived=fact.archived)
            for fact in facts
        ),
    )


@router.post(
    "/agents/{agent_id}/versions/{version}/archive",
    response_model=VersionRetired,
    responses=_NOT_YOURS,
)
async def archive_version(
    agent_id: DefinitionId,
    version: Annotated[int, Path(ge=1)],
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> VersionRetired | JSONResponse:
    """Retire one revision, so no new Session resolves to it.

    Nothing about a running Session changes here. This writes one row saying the
    revision is withdrawn; the check that reads it runs when a Session is created, and
    a Session that has already resolved never asks again.

    Existence is settled before the write rather than inferred from it, because the
    write is idempotent: it reports "nothing retired" both for a version that was
    already retired and for one that does not exist, and those are a 200 and a 404.
    """
    registry = platform_from_request(request).definition_registry
    facts = await registry.list_versions(agent_id, tenant_id)
    if not any(fact.revision == version for fact in facts):
        return refuse(
            ErrorCode.DEFINITION_NOT_FOUND,
            "that agent definition has no such version for this tenant",
            agent_id=str(agent_id),
            version=version,
        )
    newly = await registry.archive_version(agent_id, tenant_id, version)
    return VersionRetired(id=agent_id, version=version, newly_archived=newly)
