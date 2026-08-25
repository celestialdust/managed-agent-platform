"""POST /v1/agents — the only place a definition is accepted or refused.

The whole of the *well-formedness* decision is the `AgentDefinition` model in the body
annotation: FastAPI parses the request against it and answers 400 naming the offending
field before this handler runs at all. That is why there is no validation code here, and
it is the point rather than an omission — a second check in the handler would be a
second answer to "is this definition well-formed", free to disagree with the first.

The handler makes two other decisions, and neither is about the body's shape. A skills
revision its repository's CI eval gate refused may not be pinned. And a skill attached
by an id this tenant does not hold may not be attached at all. Both are facts held in a
store rather than in the submission, so nothing in the annotation can express either.

The id is minted here rather than taken from the tenant. A tenant-chosen id would let
a caller aim a registration at an id another tenant already holds, and since
registering an existing id writes the *next revision* of it, that is not a collision --
it is an edit to somebody else's agent.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.skills.evaluation import (
    Standing,
    skill_evals_of,
    standing_of,
)
from managed_agent.control.skills.registry import SkillsUnresolvable, read_attached
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import DefinitionId, TenantId, new_definition_id
from managed_agent.core.registration.definition import AgentDefinition

router = APIRouter(tags=["registration"])


class DefinitionRegistered(BaseModel):
    """What a tenant gets back: the id to address the definition by, and its revision.

    The revision is returned because it is what a Session pins, and a tenant that
    registers twice needs to know which number the second call landed on to reason about
    what its running Sessions are on.
    """

    id: DefinitionId
    revision: int


@router.post(
    "/agents",
    status_code=status.HTTP_201_CREATED,
    response_model=DefinitionRegistered,
    responses={
        STATUS_FOR[ErrorCode.DEFINITION_SKILLS_REVISION_NOT_ACCEPTED]: {
            "model": PublicErrorEnvelope
        },
        STATUS_FOR[ErrorCode.DEFINITION_INVALID]: {"model": PublicErrorEnvelope},
    },
)
async def register(
    body: AgentDefinition,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> DefinitionRegistered | JSONResponse:
    """Register a definition and report the revision it was written as.

    Always revision 1 in practice today, because the id is minted per call and a fresh
    id has no earlier revision. The number is read back from the registry rather than
    assumed to be 1, so the route stays correct when a later slice adds the endpoint
    that registers a *new revision of an existing* id.

    A revision its repository's CI eval gate did not accept is refused here, before the
    id is minted and before anything is written -- which is what makes "a degraded skill
    change never reaches an agent" a property of the door rather than of CI discipline.
    A repository nobody has ever submitted an eval run for is not under the gate at all
    and registers freely; the gate turns on for a repository with its first submission,
    pass or fail.

    The refusal is deliberately not `definition.skills_revision_unreachable`, which sits
    beside it at the same status. The commit is perfectly reachable, and the fix -- get
    the revision through the gate -- is nothing like the fix for a commit that was never
    pushed. The two are told apart by the only party who can act on either.

    The attached skill ids are checked here for the same reason and with the same
    timing: before the id is minted and before anything is written, so a refusal leaves
    no definition behind. An id that resolves now resolves forever, because a stored
    skill is immutable and never deleted -- which is what makes this the right door for
    that check. The two refusals that cannot be settled this early, a name two skills
    share and a total over the per-Session limit, both depend on what a repository
    submits later and are refused where a Session resolves.
    """
    platform = platform_from_request(request)
    facts = await skill_evals_of(platform.definition_registry).eval_facts(
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
    definition_id = new_definition_id()
    revision = await platform.definition_registry.register(
        definition_id, tenant_id, body
    )
    return DefinitionRegistered(id=definition_id, revision=revision)
