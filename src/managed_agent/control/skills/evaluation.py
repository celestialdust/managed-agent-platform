"""Grading one CI eval run against the baseline its repository already reached.

A repository's baseline per skill is the highest score any accepted run ever recorded
for that skill. It ratchets and never falls: that is what makes "below the baseline"
mean "worse than the best this repository has been", rather than "worse than last
time", which a run could satisfy while sliding downward one point at a time.

A skill named by a baseline and not by the run under grading is a regression with no
score rather than a skill nobody measured. Treating omission as neutral would make
deleting the failing eval the cheapest way through the gate.

Scores are integer basis points (0..10000) and never floats, because the only
comparison made here is against a boundary and float equality at a boundary is where a
gate reports a pass it does not have.

Nothing in this module reads or writes anything. The store is named as a Protocol at
the bottom and satisfied elsewhere, so the policy above is exercised without a database.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.ids import TenantId
from managed_agent.core.registration.definition import SkillsRevision
from managed_agent.core.registration.skill import SKILL_NAME_PATTERN

Score = Annotated[int, Field(ge=0, le=10_000, strict=True)]
"""An eval score in basis points. 10000 is a perfect run; strict, so True is not 1."""

SkillName = Annotated[
    str, Field(min_length=1, max_length=128, pattern=SKILL_NAME_PATTERN)
]
"""The name a repository's eval harness reports a skill under. A key, not prose.

The pattern is imported rather than written here, so a name the gate grades and a name
the source scanner accepts cannot become two different grammars. They already have to
match on one point that is easy to miss: a name that clears the gate and is then
refused as a directory would be a skill nothing could deliver and nothing could
explain.
"""


class Standing(StrEnum):
    """Where one skills revision stands with the gate, from the pinning side."""

    NOT_ENROLLED = "not_enrolled"
    """No eval run has ever been submitted for this repository, so no gate applies."""

    ACCEPTED = "accepted"
    """This exact revision was graded and cleared its baseline."""

    BLOCKED = "blocked"
    """This repository is under the gate and this revision is not one it accepted."""


@dataclass(frozen=True, slots=True)
class Baseline:
    """The bar one skill has to clear, and the revision that last raised it."""

    skill: str
    score: Score
    set_by_revision: str


@dataclass(frozen=True, slots=True)
class Regression:
    """One skill that did not clear its bar.

    `scored` is None exactly when the run omitted the skill rather than scoring it low.
    """

    skill: str
    baseline: Score
    scored: Score | None
    set_by_revision: str


@dataclass(frozen=True, slots=True)
class Grade:
    """The whole answer for one run. Accepted exactly when nothing regressed."""

    accepted: bool
    regressions: tuple[Regression, ...]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """What the store holds for one already-graded revision.

    `first_grading` is False when this call found a grade already recorded and wrote
    nothing. A revision is graded once and keeps that grade: an eval gate whose verdict
    a caller can re-roll by submitting again is not a gate, it is a delay.
    """

    accepted: bool
    regressions: tuple[Regression, ...]
    first_grading: bool


@dataclass(frozen=True, slots=True)
class EvalFacts:
    """The two facts a pin needs, read together so they cannot be seen out of step."""

    repository_enrolled: bool
    revision_accepted: bool


class SkillEvalRun(BaseModel):
    """One CI submission, parsed once at the boundary into a typed value.

    `repository` carries the same constraint as `AgentDefinition.skills_repository` and
    `revision` reuses that model's `SkillsRevision`, so the string a definition pins and
    the string CI grades cannot drift into two shapes that never match.

    An empty `scores` is refused here rather than graded. Against an enrolled repository
    it would regress every baselined skill by omission, which is a correct answer to a
    submission nobody meant to make; against a fresh one it would enroll the repository
    while measuring nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1)
    revision: SkillsRevision
    scores: dict[SkillName, Score] = Field(min_length=1)


def grade(baselines: Sequence[Baseline], scores: Mapping[str, int]) -> Grade:
    """Grade one run's scores against the bars its repository already reached.

    Iterates the baselines rather than the scores: a skill with a bar and no score is
    the case that matters, and iterating the scores would never visit it. A score for a
    skill with no baseline is therefore not examined at all -- it clears nothing and
    fails nothing, and it becomes a bar of its own once this run is recorded as
    accepted.

    Regressions come out sorted by skill so the refusal a caller sees for one run is the
    same refusal every time it is graded.
    """
    regressions = tuple(
        Regression(
            skill=baseline.skill,
            baseline=baseline.score,
            scored=scored,
            set_by_revision=baseline.set_by_revision,
        )
        for baseline in sorted(baselines, key=lambda b: b.skill)
        if (scored := scores.get(baseline.skill)) is None or scored < baseline.score
    )
    return Grade(accepted=not regressions, regressions=regressions)


def standing_of(facts: EvalFacts) -> Standing:
    """Where a revision stands, from two facts about its repository.

    Acceptance is checked first, so a revision the gate accepted is never reported
    blocked by a later failing run on the same repository. That ordering also makes the
    function total over the pair the store cannot produce -- accepted but not enrolled
    -- rather than leaving it to a branch nobody wrote.

    A repository nobody has ever submitted a run for is NOT_ENROLLED and its revisions
    pin freely. That is the one permissive edge in this file and it is deliberate: the
    gate is turned on for a repository by its first submission, pass or fail, and from
    then on the default within it is deny. Refusing every repository on a platform where
    no CI has ever run would refuse every agent definition, which is not a gate either.
    """
    if facts.revision_accepted:
        return Standing.ACCEPTED
    if facts.repository_enrolled:
        return Standing.BLOCKED
    return Standing.NOT_ENROLLED


@runtime_checkable
class SkillEvalStore(Protocol):
    """What the gate needs held for it.

    Satisfied structurally, and deliberately not a second store: the definition registry
    already answers "what does this definition resolve to at the moment a Session is
    created", and whether the pinned revision cleared its gate is that same question
    asked at the same door. A store of its own would be a second thing `POST /v1/agents`
    has to consult and keep in step.

    `runtime_checkable` so `skill_evals_of` below can narrow a registry that is typed as
    the definition-registry port. That check sees method names and not signatures, which
    is the same shallowness `core.ports` documents: it catches the object that never
    grew the methods, not the one that grew them wrong.
    """

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        """Every skill's current bar for one repository, ascending by skill name."""
        ...

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        """Record this grading, or report the one already recorded for that revision.

        Writes the refused runs too. A refusal that left no row would let the same
        revision be graded again against a baseline that had since moved.
        """
        ...

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        """Both facts `standing_of` needs, from one read."""
        ...


def skill_evals_of(registry: object) -> SkillEvalStore:
    """Narrow the wired definition registry to the eval store the gate reads.

    A type narrowing rather than a validation: the registry comes from the composition
    root and no request can influence which object is there, so a registry without these
    methods is a wiring mistake and not something a caller did.

    It raises rather than degrading to "no gate applies". A registry that cannot answer
    whether a repository is enrolled has not told us it is unenrolled -- it has told us
    nothing -- and turning that into a pin is the one outcome this whole module exists
    to prevent. A `TypeError` here surfaces as a 500 on the very first registration,
    which is loud at the only moment it is still cheap.
    """
    if not isinstance(registry, SkillEvalStore):
        raise TypeError(
            f"{type(registry).__name__} is wired as the definition registry but does "
            "not hold the CI eval gate's reads, so a skills revision could be pinned "
            "without its baseline ever being consulted"
        )
    return registry
