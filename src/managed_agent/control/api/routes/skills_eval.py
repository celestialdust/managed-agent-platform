"""POST /v1/skills/evals and GET /v1/skills/baselines — the CI eval gate's own surface.

Submitting is the only way a skills revision becomes pinnable, and it is graded once.
The refused runs are recorded before the refusal is returned, deliberately: a refusal
that wrote nothing would let the same revision be submitted again later, against a
baseline that had moved, and the second answer would silently overwrite the first.

Neither route stores, edits or lists a skill. What is stored is the score a revision
got, and what is done with that score is refuse.

Neither route reads or writes a Session, so neither appends to the Event Log. The gate
runs before any agent definition can name the revision, which is before any Session
exists.

Both are scoped the way every other route in this package is scoped, through the
unauthenticated tenant placeholder in `tenancy.py`: a skills repository belongs to a
domain team, its CI runs as that team, and inventing a separate principal here would
decide an authorization question no decision record has opened.

The refusal detail names one skill rather than all of them, because `ErrorEnvelope`'s
detail is flat and a list would have to be flattened into the message, where nothing may
branch on it (ADR-013). `regressed_skills` says how many there were, and `scored` is
absent exactly when the run did not report that skill at all — which is a different
mistake from scoring low and needs a different fix.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.skills.evaluation import (
    Regression,
    SkillEvalRun,
    grade,
    skill_evals_of,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import TenantId

router = APIRouter(tags=["skills"])


def _regression_detail(regressions: tuple[Regression, ...]) -> dict[str, str | int]:
    """The flat, branchable facts about the first regression, plus how many there were.

    `scored` is omitted rather than sent as null when the run did not report the skill,
    so a caller can tell "you scored below the bar" from "you stopped running this eval"
    by the presence of the key alone.
    """
    first = regressions[0]
    detail: dict[str, str | int] = {
        "skill": first.skill,
        "baseline": first.baseline,
        "baseline_set_by_revision": first.set_by_revision,
        "regressed_skills": len(regressions),
    }
    if first.scored is not None:
        detail["scored"] = first.scored
    return detail


class EvalRecorded(BaseModel):
    """What CI gets back when a run cleared the bar.

    `newly_graded` is False when this submission found a grade already recorded and
    wrote nothing, which is how a retried job tells its own retry from a fresh grading.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str
    revision: str
    newly_graded: bool


class BaselineView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill: str
    score: int
    set_by_revision: str


class BaselinePage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str
    baselines: list[BaselineView]


@router.post(
    "/skills/evals",
    response_model=EvalRecorded,
    responses={
        STATUS_FOR[ErrorCode.SKILL_EVAL_REGRESSED]: {"model": PublicErrorEnvelope}
    },
)
async def submit_eval_run(
    body: SkillEvalRun,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> EvalRecorded | JSONResponse:
    """Grade one revision against its repository's baseline and record the verdict.

    The verdict returned is the one the store holds, not the one just computed. On a
    first submission those are the same object; on a retry the computed grade is
    discarded and the original stands, which is what makes the gate un-re-rollable.

    A repeat submission answers 200 rather than a conflict. A CI job is retried for
    reasons that have nothing to do with the code under grading — a runner died, a
    network blipped — and a retry that failed differently from the first attempt would
    make the pipeline's own flakiness look like a skill regression.
    """
    store = skill_evals_of(platform_from_request(request).definition_registry)
    baselines = await store.eval_baselines(tenant_id, body.repository)
    recorded = await store.record_eval_run(
        tenant_id,
        body.repository,
        body.revision,
        body.scores,
        grade(baselines, body.scores),
    )
    if not recorded.accepted:
        return refuse(
            ErrorCode.SKILL_EVAL_REGRESSED,
            f"{len(recorded.regressions)} skill(s) scored below the recorded baseline; "
            "this revision cannot be pinned by an agent definition",
            repository=body.repository,
            revision=body.revision,
            **_regression_detail(recorded.regressions),
        )
    return EvalRecorded(
        repository=body.repository,
        revision=body.revision,
        newly_graded=recorded.first_grading,
    )


@router.get("/skills/baselines", response_model=BaselinePage)
async def read_baselines(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    repository: Annotated[str, Query(min_length=1)],
) -> BaselinePage:
    """The bar every skill in one repository currently has to clear.

    A repository nobody has submitted a run for reads as an empty list rather than a
    404: "no bar yet" is the true answer and it is the same answer for a repository that
    does not exist, which this surface has no way to distinguish and no reason to guess
    at.
    """
    store = skill_evals_of(platform_from_request(request).definition_registry)
    baselines = await store.eval_baselines(tenant_id, repository)
    return BaselinePage(
        repository=repository,
        baselines=[
            BaselineView(
                skill=baseline.skill,
                score=baseline.score,
                set_by_revision=baseline.set_by_revision,
            )
            for baseline in baselines
        ],
    )
