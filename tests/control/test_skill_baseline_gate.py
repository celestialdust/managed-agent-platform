"""The CI eval gate: what it accepts, what it refuses, and what a pin resolves to after.

Three tiers in one file, and the split is deliberate.

The grading policy is two total functions over frozen values, so every case in it is
reachable with no app, no HTTP request and no database.

The route cases run the real routers over one in-memory store, because the claim under
test is a relationship *between* two surfaces: a revision refused at the CI surface must
be unpinnable at the registration surface, and the definition already registered against
the passing revision must still be what a Session resolves. Two stores could not show
that, and a mock could let both the response and the recorded verdict be wrong while the
assertions passed.

The store's own guarantees — the ratchet, the single grading per revision, the check
constraints — are graded against real PostgreSQL at the bottom, because every one of
them is a property of the database and not of anything we could stand in for it.

**What is not proved here.** The eval *cases* a real run would score do not exist in
this repository: `environment.md` records the `slr-case-corpus` fixture as absent, and
no eval harness is provisioned. So every score in this file is a fixture, and what is
graded is the gate's arithmetic and its refusals — never that a particular skill really
scores what CI would say it scores.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.definition_registry import (
    PostgresDefinitionRegistry,
    Resolved,
)
from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.catalog.definitions import AgentRecord
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.skills.evaluation import (
    Baseline,
    EvalFacts,
    Grade,
    Regression,
    RunRecord,
    SkillEvalRun,
    SkillEvalStore,
    Standing,
    grade,
    skill_evals_of,
    standing_of,
)
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import (
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    new_definition_id,
)
from managed_agent.core.ports import Resolution, SessionListing, UnknownDefinition
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord

REPOSITORY = "git@internal.example:domain/slr-skills.git"
PASSING = "a" * 40
REGRESSED = "b" * 40
UNSUBMITTED = "c" * 40
IMPROVED = "d" * 40


# ---------------------------------------------------------------------------
# The grading policy, with no store anywhere near it.
# ---------------------------------------------------------------------------


def _bar(skill: str, score: int, set_by: str = PASSING) -> Baseline:
    return Baseline(skill=skill, score=score, set_by_revision=set_by)


def test_a_score_at_the_bar_clears_it_and_a_point_below_does_not() -> None:
    """The boundary itself, which is the only comparison this module makes.

    Equality passing is the decision, not an accident: a baseline is the best a
    repository has been, and matching it is not a regression. One basis point below is,
    and it is refused with the score it actually got so the reader knows how far.
    """
    bars = [_bar("extract", 8700)]

    assert grade(bars, {"extract": 8700}) == Grade(accepted=True, regressions=())
    assert grade(bars, {"extract": 9000}) == Grade(accepted=True, regressions=())
    assert grade(bars, {"extract": 8699}) == Grade(
        accepted=False,
        regressions=(Regression("extract", 8700, 8699, PASSING),),
    )


def test_a_baselined_skill_the_run_never_reported_is_a_regression() -> None:
    """Omission is not neutral, and the refusal says so by carrying no score.

    If a missing eval were silent, the cheapest way past this gate would be to delete
    the eval that fails -- the one failure mode a baseline gate exists to prevent, and
    it would leave no trace anywhere.
    """
    graded = grade([_bar("extract", 8700)], {"screen": 10_000})

    assert graded.accepted is False
    assert graded.regressions == (Regression("extract", 8700, None, PASSING),)


def test_a_repository_with_no_bars_accepts_anything_it_reports() -> None:
    """The first run enrolls a repository; it cannot fail against bars not yet set.

    A skill the run scores and no baseline covers is not examined at all -- it clears
    nothing and fails nothing, and becomes a bar of its own once this run is recorded.
    """
    assert grade([], {"extract": 1, "brand-new": 0}) == Grade(True, ())


def test_regressions_come_back_sorted_by_skill_whatever_order_the_bars_arrive_in() -> (
    None
):
    """One run graded twice produces the same refusal, so the detail is stable.

    The route names the *first* regression in its detail, so an unordered answer would
    make the same submission blame a different skill on different runs.
    """
    bars = [_bar("screen", 9000), _bar("extract", 9000), _bar("cite", 9000)]

    graded = grade(bars, {"screen": 1, "extract": 1, "cite": 1})

    assert [r.skill for r in graded.regressions] == ["cite", "extract", "screen"]


def test_a_run_that_regresses_two_of_three_skills_names_exactly_those_two() -> None:
    bars = [_bar("cite", 9000), _bar("extract", 8700), _bar("screen", 9200)]

    graded = grade(bars, {"cite": 9000, "extract": 100, "screen": 9199})

    assert graded.accepted is False
    assert [(r.skill, r.scored) for r in graded.regressions] == [
        ("extract", 100),
        ("screen", 9199),
    ]


@pytest.mark.parametrize("score", [-1, 10_001, True, 1.0, "8700"])
def test_a_score_outside_the_basis_point_range_is_refused(score: object) -> None:
    """Strict integers, so `True` is not 1 and `1.0` is not 1.

    The comparison this module exists to make is against a boundary, and a float at a
    boundary is exactly where a gate reports a pass it did not have.
    """
    with pytest.raises(ValidationError):
        SkillEvalRun(repository=REPOSITORY, revision=PASSING, scores={"extract": score})  # type: ignore[dict-item]


@pytest.mark.parametrize("name", ["", "Upper", "-leading", "x" * 129])
def test_a_skill_name_that_is_not_a_key_is_refused(name: str) -> None:
    """A skill name is a key CI and the baseline both spell the same way, not prose."""
    with pytest.raises(ValidationError):
        SkillEvalRun(repository=REPOSITORY, revision=PASSING, scores={name: 1})


def test_a_submission_is_frozen_and_refuses_junk_fields_and_an_empty_scores_map() -> (
    None
):
    """An empty run would enroll a repository while measuring nothing.

    Against an already-enrolled repository it would regress every baselined skill by
    omission, which is a correct answer to a submission nobody meant to make.
    """
    with pytest.raises(ValidationError):
        SkillEvalRun(
            repository=REPOSITORY,
            revision=PASSING,
            scores={"extract": 1},
            harness="v2",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        SkillEvalRun(repository=REPOSITORY, revision=PASSING, scores={})
    with pytest.raises(ValidationError):
        SkillEvalRun(repository=REPOSITORY, revision="a" * 39, scores={"extract": 1})

    parsed = SkillEvalRun(
        repository=REPOSITORY, revision=PASSING, scores={"extract": 1}
    )
    with pytest.raises(ValidationError):
        parsed.repository = "elsewhere"


@pytest.mark.parametrize(
    ("enrolled", "accepted", "expected"),
    [
        (True, True, Standing.ACCEPTED),
        (False, True, Standing.ACCEPTED),
        (True, False, Standing.BLOCKED),
        (False, False, Standing.NOT_ENROLLED),
    ],
)
def test_standing_is_total_over_both_facts(
    enrolled: bool, accepted: bool, expected: Standing
) -> None:
    """Four pairs, four answers, including the one the store cannot produce.

    Acceptance is read first, so a revision the gate accepted is never reported blocked
    by a later failing run on the same repository.
    """
    facts = EvalFacts(repository_enrolled=enrolled, revision_accepted=accepted)

    assert standing_of(facts) is expected


# ---------------------------------------------------------------------------
# The fake store both surfaces read, held honest before anything reads through it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredRun:
    scores: Mapping[str, int]
    accepted: bool
    regressions: tuple[Regression, ...]
    ordinal: int


class FakeSkillEvals:
    """One in-memory `skill_eval_run` table, keyed as the real one is.

    `ordinal` stands in for `submitted_at`: the baseline read breaks a tie on score in
    favour of the earliest accepted run, and an in-memory dict has no clock.
    """

    def __init__(self) -> None:
        self._runs: dict[tuple[str, str, str], StoredRun] = {}

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        best: dict[str, tuple[int, int, str]] = {}
        for (tenant, repo, revision), run in self._runs.items():
            if (tenant, repo) != (str(tenant_id), repository) or not run.accepted:
                continue
            for skill, score in run.scores.items():
                held = best.get(skill)
                if held is None or (score, -run.ordinal) > (held[0], -held[1]):
                    best[skill] = (score, run.ordinal, revision)
        return tuple(
            Baseline(skill=skill, score=score, set_by_revision=revision)
            for skill, (score, _, revision) in sorted(best.items())
        )

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        key = (str(tenant_id), repository, revision)
        first = key not in self._runs
        if first:
            self._runs[key] = StoredRun(
                scores=dict(scores),
                accepted=graded.accepted,
                regressions=graded.regressions,
                ordinal=len(self._runs),
            )
        held = self._runs[key]
        return RunRecord(
            accepted=held.accepted,
            regressions=held.regressions,
            first_grading=first,
        )

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        rows = [
            (key, run)
            for key, run in self._runs.items()
            if key[0] == str(tenant_id) and key[1] == repository
        ]
        return EvalFacts(
            repository_enrolled=bool(rows),
            revision_accepted=any(
                key[2] == revision and run.accepted for key, run in rows
            ),
        )

    def rows(self) -> int:
        return len(self._runs)


async def test_the_fake_ratchets_and_grades_once() -> None:
    """Pinned before any route case reads through it.

    Its own ratchet and its own idempotence are asserted here so that a later failure is
    a failure of the code under test rather than of the stand-in it runs against.
    """
    store = FakeSkillEvals()
    tenant = TenantId(uuid.uuid4())

    assert await store.eval_baselines(tenant, REPOSITORY) == ()
    assert await store.eval_facts(tenant, REPOSITORY, PASSING) == EvalFacts(
        repository_enrolled=False, revision_accepted=False
    )

    first = await store.record_eval_run(
        tenant, REPOSITORY, PASSING, {"extract": 8700}, Grade(True, ())
    )
    assert (first.accepted, first.first_grading) == (True, True)

    again = await store.record_eval_run(
        tenant,
        REPOSITORY,
        PASSING,
        {"extract": 10_000},
        Grade(False, (Regression("extract", 9000, 10, PASSING),)),
    )
    assert (again.accepted, again.first_grading) == (True, False)

    await store.record_eval_run(
        tenant, REPOSITORY, UNSUBMITTED, {"extract": 9100}, Grade(True, ())
    )
    assert await store.eval_baselines(tenant, REPOSITORY) == (
        Baseline(skill="extract", score=9100, set_by_revision=UNSUBMITTED),
    )
    assert await store.eval_baselines(TenantId(uuid.uuid4()), REPOSITORY) == ()


async def test_the_fake_keeps_the_earlier_revision_when_a_later_run_ties() -> None:
    """`set_by_revision` names the run that *set* the bar, not the last one to match it.

    Asserted on the stand-in as well as on the adapter, because the walk at the bottom
    reads the attribution back through this object.
    """
    store = FakeSkillEvals()
    tenant = TenantId(uuid.uuid4())

    await store.record_eval_run(
        tenant, REPOSITORY, PASSING, {"extract": 8700}, Grade(True, ())
    )
    await store.record_eval_run(
        tenant, REPOSITORY, IMPROVED, {"extract": 8700}, Grade(True, ())
    )

    assert await store.eval_baselines(tenant, REPOSITORY) == (
        Baseline(skill="extract", score=8700, set_by_revision=PASSING),
    )


# ---------------------------------------------------------------------------
# The routes, over one store, through the real app factory.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class InMemoryLog:
    def __init__(self) -> None:
        self._events: dict[SessionId, list[Event]] = {}

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        log = self._events.setdefault(session_id, [])
        seq = Seq(len(log) + 1)
        log.append(Event(session_id, seq, type_, dict(payload)))
        return seq

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Event]:
        span = [
            event
            for event in self._events.get(session_id, [])
            if start <= event.seq <= end
        ]
        return span[:limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Event]:
        for event in self._events.get(session_id, []):
            if event.seq > after:
                yield event

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return Seq(1)

    def events(self, session_id: SessionId) -> list[Event]:
        return list(self._events.get(session_id, []))


class InMemorySessionRegistry:
    def __init__(self) -> None:
        self._rows: dict[SessionId, SessionRecord] = {}

    async def create(self, record: SessionRecord) -> None:
        self._rows[record.id] = record

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        raise AssertionError(
            "this file never reads a Session back through the registry"
        )

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("this file never pages Sessions")


class UnusedToolRegistry:
    """Raises rather than answering: nothing here registers or resolves a tool."""

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("a skill-gate test registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("a skill-gate test looked up a tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("a skill-gate test listed a tenant's tools")


_REGISTERED_AT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
"""When every agent in this file came into being.

Fixed rather than `now()`, because `AgentRecord.created_at` is published and a moving
value would make a response body differ between runs. Nothing here asserts on it -- what
these cases are about is which gate the append route consults -- so one moment for all
of them is the honest shape.
"""


class GatedRegistry:
    """The definition registry and this slice's gate on one object, as the adapter is.

    One object rather than two, because that is what the real store is: the registration
    route reads the gate through the very same port it writes the definition to, and a
    test that split them could not catch a route consulting a gate nobody wired.

    `writes` is what lets a case assert that a *refused* registration wrote nothing --
    the half of a refusal that actually matters, since a 400 that stored the row anyway
    is worse than no check at all.
    """

    def __init__(self, evals: FakeSkillEvals) -> None:
        self._evals = evals
        self._rows: dict[DefinitionId, list[AgentDefinition]] = {}
        self._owner: dict[DefinitionId, TenantId] = {}
        self.writes = 0

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        self.writes += 1
        self._owner.setdefault(definition_id, tenant_id)
        revisions = self._rows.setdefault(definition_id, [])
        revisions.append(definition)
        return len(revisions)

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        revisions = self._rows.get(definition_id)
        if not revisions or self._owner.get(definition_id) != tenant_id:
            raise UnknownDefinition(str(definition_id))
        return Resolved(revisions[-1], len(revisions))

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        if self._owner.get(definition_id) != tenant_id:
            return ()
        return tuple(
            VersionFact(revision=n, archived=False)
            for n, _ in enumerate(self._rows.get(definition_id, ()), start=1)
        )

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        if self._owner.get(definition_id) != tenant_id:
            return None
        revisions = self._rows.get(definition_id, [])
        return revisions[revision - 1] if 1 <= revision <= len(revisions) else None

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a gate test archived a definition revision")

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        return await self._evals.eval_baselines(tenant_id, repository)

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        return await self._evals.record_eval_run(
            tenant_id, repository, revision, scores, graded
        )

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        return await self._evals.eval_facts(tenant_id, repository, revision)

    async def read_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> AgentRecord | None:
        """The agent as a whole, folded from the revisions this fake holds.

        Answers rather than raises, because the append route now reads it on every call
        in order to refuse a retired agent -- a raising stub would turn every case in
        this file that appends a revision into a 500.

        `archived_at` is always None, and that is not a convenience: `archive_agent`
        below raises, so no case here can retire an agent and a live agent is the only
        reachable state. A case that ever needs a retired one has to teach that method
        to record the retirement first, or this would report it as live.

        The tenant is checked the way `read_version` checks it, so another tenant's id
        reads as absent rather than as somebody else's agent.
        """
        if self._owner.get(definition_id) != tenant_id:
            return None
        revisions = self._rows.get(definition_id, [])
        if not revisions:
            return None
        return AgentRecord(
            definition_id=definition_id,
            version=len(revisions),
            created_at=_REGISTERED_AT,
            archived_at=None,
            definition=revisions[-1],
        )

    async def page_agents(
        self,
        tenant_id: TenantId,
        *,
        include_archived: bool,
        created_from: datetime | None,
        created_to: datetime | None,
        after: tuple[datetime, DefinitionId] | None,
        limit: int,
    ) -> tuple[AgentRecord, ...]:
        raise AssertionError("a gate test listed a tenant's agents")

    async def archive_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> datetime | None:
        raise AssertionError("a gate test retired a whole agent")

    async def register_at_revision(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
        expected: int,
    ) -> int | None:
        raise AssertionError(
            "a gate test appended a revision against an expected number"
        )


@dataclass(frozen=True, slots=True)
class Harness:
    client: AsyncClient
    registry: GatedRegistry
    evals: FakeSkillEvals
    log: InMemoryLog
    tenant: TenantId

    async def submit(self, revision: str, **scores: int) -> Any:
        return await self.client.post(
            "/v1/skills/evals",
            json={
                "repository": REPOSITORY,
                "revision": revision,
                "scores": scores,
            },
        )

    async def pin(self, revision: str) -> Any:
        return await self.client.post("/v1/agents", json=_definition_pinning(revision))

    async def edit(self, agent_id: str, revision: str) -> Any:
        """Append a revision to an existing agent -- the gate's second door.

        `POST /v1/agents` mints its own id and can only write revision 1, so an
        edit is the only way a *live* agent's skills revision changes. A gate on
        the first door alone is a gate with a corridor around it.
        """
        return await self.client.post(
            f"/v1/agents/{agent_id}/versions", json=_definition_pinning(revision)
        )

    async def baselines(self, repository: str = REPOSITORY) -> Any:
        return await self.client.get(
            "/v1/skills/baselines", params={"repository": repository}
        )


def _definition_pinning(revision: str) -> dict[str, object]:
    return {
        "name": "slr-reviewer",
        "instructions": "Review the systematic literature.",
        "model": "gpt-5-codex",
        "skills_repository": REPOSITORY,
        "skills_revision": revision,
    }


def _platform(registry: GatedRegistry, log: InMemoryLog) -> Platform:
    return Platform(
        event_log_append=log,
        event_log_range=log,
        definition_registry=registry,
        tool_registry=UnusedToolRegistry(),
        session_registry=InMemorySessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=AnyEnvironmentResolves(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    """Every route in the walk, over one store, behind one tenant's client.

    The tenant travels as a default header rather than on `request.state`, because
    `unauthenticated_tenant_from_header` is the one place a route in this package learns
    who is calling: a case supplying the tenant any other way would pass against a route
    that never asks for one.
    """
    evals = FakeSkillEvals()
    registry, log = GatedRegistry(evals), InMemoryLog()
    tenant = TenantId(uuid.uuid4())
    async with AsyncClient(
        transport=ASGITransport(app=create_app(_platform(registry, log))),
        base_url="http://tenant",
        headers={TENANT_HEADER: str(tenant)},
    ) as client:
        yield Harness(
            client=client, registry=registry, evals=evals, log=log, tenant=tenant
        )


async def test_a_first_run_is_graded_and_a_retry_of_it_is_not(harness: Harness) -> None:
    """A retried CI job gets the original verdict, not a second grading.

    A runner dies, a network blips, the job runs again. A retry that failed differently
    from the first attempt would make the pipeline's own flakiness look like a skill
    regression -- so the answer comes out of the recorded row, and `newly_graded` is how
    the caller tells the retry from the grading.
    """
    first = await harness.submit(PASSING, extract=8700)
    assert first.status_code == 200, first.text
    assert first.json() == {
        "repository": REPOSITORY,
        "revision": PASSING,
        "newly_graded": True,
    }

    again = await harness.submit(PASSING, extract=8700)
    assert again.status_code == 200, again.text
    assert again.json()["newly_graded"] is False
    assert harness.evals.rows() == 1


async def test_a_run_below_the_bar_is_refused_and_recorded_as_refused(
    harness: Harness,
) -> None:
    """The refusal carries branchable facts, and the refused run is written down.

    Written down deliberately: a refusal that left no row would let the same revision be
    submitted again later, against a baseline that had since moved, and the second
    answer would silently replace the first. The re-submission below is what proves it.
    """
    assert (await harness.submit(PASSING, extract=8700, screen=9200)).status_code == 200

    refused = await harness.submit(REGRESSED, extract=8100, screen=9200)

    assert refused.status_code == 400, refused.text
    body = refused.json()
    assert body["error"]["code"] == "skill_eval.regressed"
    assert body["error"]["detail"] == {
        "repository": REPOSITORY,
        "revision": REGRESSED,
        "skill": "extract",
        "baseline": 8700,
        "scored": 8100,
        "baseline_set_by_revision": PASSING,
        "regressed_skills": 1,
    }
    assert harness.evals.rows() == 2

    perfect = await harness.submit(REGRESSED, extract=10_000, screen=10_000)
    assert perfect.status_code == 400, perfect.text
    assert perfect.json()["error"]["detail"] == body["error"]["detail"], (
        "a refused revision was re-graded on a second submission; the verdict must "
        "come out of the recorded row or the gate is a delay rather than a gate"
    )
    assert harness.evals.rows() == 2


async def test_a_run_omitting_a_baselined_skill_is_refused_with_no_scored_key(
    harness: Harness,
) -> None:
    """Deleting the failing eval is no way through, and the detail says which mistake.

    `scored` absent rather than null: "you scored below the bar" and "you stopped
    running this eval" need different fixes, and a caller must be able to tell them
    apart from the refusal alone.
    """
    assert (await harness.submit(PASSING, extract=8700, screen=9200)).status_code == 200

    refused = await harness.submit(REGRESSED, screen=9200)

    assert refused.status_code == 400, refused.text
    detail = refused.json()["error"]["detail"]
    assert detail["skill"] == "extract"
    assert "scored" not in detail
    assert detail["regressed_skills"] == 1


async def test_the_bar_ratchets_so_yesterdays_passing_scores_stop_passing(
    harness: Harness,
) -> None:
    """A baseline is the best a repository has been, not the last thing it did.

    Without the ratchet a run could satisfy "no worse than last time" while sliding
    downward one point at a time, and nothing would ever refuse it.
    """
    assert (await harness.submit(PASSING, extract=8700)).status_code == 200
    assert (await harness.submit(IMPROVED, extract=9400)).status_code == 200

    fell_back = await harness.submit(REGRESSED, extract=8700)

    assert fell_back.status_code == 400, fell_back.text
    assert fell_back.json()["error"]["detail"]["baseline"] == 9400
    assert fell_back.json()["error"]["detail"]["baseline_set_by_revision"] == IMPROVED


async def test_baselines_read_empty_for_a_repository_nobody_has_submitted(
    harness: Harness,
) -> None:
    """ "No bar yet" is the true answer, and it is the same answer for a repository that
    does not exist -- which this surface cannot distinguish and has no reason to guess
    at.
    """
    page = await harness.baselines("git@internal.example:domain/nothing.git")

    assert page.status_code == 200, page.text
    assert page.json() == {
        "repository": "git@internal.example:domain/nothing.git",
        "baselines": [],
    }


async def test_baselines_report_the_highest_score_and_the_revision_that_set_it(
    harness: Harness,
) -> None:
    assert (await harness.submit(PASSING, extract=8700, screen=9200)).status_code == 200
    assert (
        await harness.submit(IMPROVED, extract=9400, screen=9200)
    ).status_code == 200

    page = await harness.baselines()

    assert page.json()["baselines"] == [
        {"skill": "extract", "score": 9400, "set_by_revision": IMPROVED},
        {"skill": "screen", "score": 9200, "set_by_revision": PASSING},
    ]


async def test_a_submission_with_an_unknown_field_is_refused_as_a_bad_request(
    harness: Harness,
) -> None:
    """FastAPI answers before the handler runs: nothing graded, nothing stored."""
    refused = await harness.client.post(
        "/v1/skills/evals",
        json={
            "repository": REPOSITORY,
            "revision": PASSING,
            "scores": {"extract": 8700},
            "harness_version": "v2",
        },
    )

    assert refused.status_code == 400, refused.text
    assert harness.evals.rows() == 0


async def test_both_ci_routes_refuse_a_caller_that_has_not_said_which_tenant(
    harness: Harness,
) -> None:
    """The tenant is read one way in this package, and these two are no exception."""
    submitted = await harness.client.post(
        "/v1/skills/evals",
        json={"repository": REPOSITORY, "revision": PASSING, "scores": {"extract": 1}},
        headers={TENANT_HEADER: ""},
    )
    read = await harness.client.get(
        "/v1/skills/baselines",
        params={"repository": REPOSITORY},
        headers={TENANT_HEADER: ""},
    )

    assert submitted.status_code == 400, submitted.text
    assert read.status_code == 400, read.text


async def test_one_tenants_bar_does_not_constrain_another(harness: Harness) -> None:
    """The identical repository string under two tenants is two independent ratchets."""
    assert (await harness.submit(PASSING, extract=8700)).status_code == 200

    other = TenantId(uuid.uuid4())
    elsewhere = await harness.client.post(
        "/v1/skills/evals",
        json={
            "repository": REPOSITORY,
            "revision": REGRESSED,
            "scores": {"extract": 10},
        },
        headers={TENANT_HEADER: str(other)},
    )

    assert elsewhere.status_code == 200, elsewhere.text
    assert elsewhere.json()["newly_graded"] is True


# ---------------------------------------------------------------------------
# The gate in front of the pin, which is the half that is easy to forget.
# ---------------------------------------------------------------------------


async def test_an_unenrolled_repository_pins_freely(harness: Harness) -> None:
    """The permissive edge, asserted rather than assumed.

    A platform where no CI has ever run would otherwise refuse every agent definition,
    which is not a gate either. The gate turns on for a repository with its first
    submission, pass or fail -- the case below is what proves the "or fail" half.
    """
    registered = await harness.pin(UNSUBMITTED)

    assert registered.status_code == 201, registered.text
    assert harness.registry.writes == 1


async def test_a_refused_revision_cannot_be_pinned_and_the_refusal_writes_nothing(
    harness: Harness,
) -> None:
    """The whole checkpoint's first half: refused *before* anything is written.

    A 400 that stored the definition anyway would be worse than no check at all, because
    the tenant believes it was rejected while a Session can still resolve it. The write
    count is what says nothing was stored; the status alone cannot.
    """
    assert (await harness.submit(PASSING, extract=8700)).status_code == 200
    assert (await harness.submit(REGRESSED, extract=8100)).status_code == 400

    refused = await harness.pin(REGRESSED)

    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "definition.skills_revision_not_accepted"
    assert refused.json()["error"]["detail"] == {
        "field": "skills_revision",
        "skills_repository": REPOSITORY,
        "skills_revision": REGRESSED,
    }
    assert harness.registry.writes == 0, (
        "a refused registration reached the store; nothing may be written before the "
        "gate has cleared the revision"
    )


async def test_a_never_submitted_revision_of_an_enrolled_repository_is_refused_too(
    harness: Harness,
) -> None:
    """Enrollment is per repository, so an ungraded sibling commit is blocked as well.

    Otherwise the way past the gate would be to never submit the revision you intend to
    pin, which costs an attacker nothing.
    """
    assert (await harness.submit(PASSING, extract=8700)).status_code == 200

    refused = await harness.pin(UNSUBMITTED)

    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "definition.skills_revision_not_accepted"
    assert harness.registry.writes == 0


async def test_a_repository_whose_only_run_failed_is_still_under_the_gate(
    harness: Harness,
) -> None:
    """Failing once and then not calling again must not be a way out.

    Enrollment counts every run and acceptance only the accepted ones, which is the
    asymmetry that makes this hold. The refused revision here is the *only* thing that
    ever came back from the gate for it, and it stays unpinnable however long CI stays
    quiet afterwards.
    """
    assert (await harness.submit(PASSING, extract=8700)).status_code == 200
    assert (await harness.submit(REGRESSED, extract=10)).status_code == 400

    assert (await harness.pin(REGRESSED)).status_code == 400
    assert (await harness.pin(UNSUBMITTED)).status_code == 400
    assert harness.registry.writes == 0


async def test_an_accepted_revision_pins(harness: Harness) -> None:
    assert (await harness.submit(PASSING, extract=8700)).status_code == 200

    registered = await harness.pin(PASSING)

    assert registered.status_code == 201, registered.text
    assert registered.json()["revision"] == 1


async def test_a_degraded_skill_change_never_reaches_an_agent(harness: Harness) -> None:
    """The whole walk MAP-A144 describes, across three surfaces over one store.

    The second half of the checkpoint is the one that is easy to lose: after the
    refusal, the definition registered against the passing revision must still be what a
    Session resolves. That is a positive claim about the *old* revision, so it is
    resolved and read rather than inferred from the refusal.
    """
    assert (await harness.submit(PASSING, extract=8700, screen=9200)).status_code == 200

    registered = await harness.pin(PASSING)
    assert registered.status_code == 201, registered.text
    definition_id = DefinitionId(UUID(registered.json()["id"]))

    degraded = await harness.submit(REGRESSED, extract=8100, screen=9200)
    assert degraded.status_code == 400, degraded.text
    assert degraded.json()["error"]["detail"]["scored"] == 8100

    refused_pin = await harness.pin(REGRESSED)
    assert refused_pin.status_code == 400, refused_pin.text

    assert (await harness.baselines()).json()["baselines"] == [
        {"skill": "extract", "score": 8700, "set_by_revision": PASSING},
        {"skill": "screen", "score": 9200, "set_by_revision": PASSING},
    ]

    started = await harness.client.post(
        "/v1/sessions",
        json={
            "definition_id": str(definition_id),
            # Any id resolves through this file's environment store; which shape a
            # Session runs in is graded in `test_environment_reference.py`.
            "environment_id": str(uuid.uuid4()),
            "budget_minor_units": 5_000,
            "budget_currency": "USD",
            "retention_days": 30,
        },
    )
    assert started.status_code == 201, started.text

    assert harness.registry.writes == 1
    resolved = await harness.registry.resolve(definition_id, harness.tenant)
    assert resolved.revision == 1
    assert resolved.definition.skills_revision == PASSING, (
        "the Session resolved something other than the revision that passed; a refused "
        "change must not disturb work already registered"
    )
    created = harness.log.events(SessionId(UUID(started.json()["id"])))
    assert [event.payload["definition_revision"] for event in created] == [1]


# ---------------------------------------------------------------------------
# The wiring that makes the gate unavoidable rather than merely available.
# ---------------------------------------------------------------------------


def test_the_wired_definition_registry_holds_the_gates_reads() -> None:
    """A positive check, because the refusals above are all negative ones.

    Every case in this file would still pass if `PostgresDefinitionRegistry` had never
    grown these methods -- they run against a fake. This is what says the object the
    composition root actually wires can answer the gate, so the refusal is a property of
    the deployed system and not of the stand-in.
    """
    assert issubclass(PostgresDefinitionRegistry, SkillEvalStore), (
        "the wired definition registry cannot answer the CI eval gate's reads, so "
        "POST /v1/agents would refuse every registration at run time"
    )


def test_a_registry_without_the_gates_reads_is_refused_rather_than_waved_through() -> (
    None
):
    """A wiring mistake must not read as "no repository is enrolled".

    A registry that cannot answer whether a repository is enrolled has not told us it is
    unenrolled -- it has told us nothing -- and turning that into a pin is the one
    outcome this module exists to prevent.
    """

    class RegistryWithoutTheGate:
        async def register(
            self,
            definition_id: DefinitionId,
            tenant_id: TenantId,
            definition: AgentDefinition,
        ) -> int:
            return 1

    with pytest.raises(TypeError, match="CI eval gate"):
        skill_evals_of(RegistryWithoutTheGate())


def test_both_ci_routes_are_mounted_under_the_version_prefix() -> None:
    """The router is attached by the app factory, not just by a test with its own app.

    A router defined correctly and never included passes every route case above and 404s
    in production.
    """
    paths = create_app(
        _platform(GatedRegistry(FakeSkillEvals()), InMemoryLog())
    ).openapi()["paths"]

    assert "/v1/skills/evals" in paths
    assert "/v1/skills/baselines" in paths


# ---------------------------------------------------------------------------
# The store itself, against real PostgreSQL. Tier 1, testcontainers.
# ---------------------------------------------------------------------------


def _row(**overrides: object) -> dict[str, object]:
    return {
        "tenant": uuid.uuid4(),
        "repository": REPOSITORY,
        "revision": PASSING,
        "scores": {"extract": 8700},
        "accepted": True,
        "regressions": [],
    } | overrides


_RAW_INSERT = sa.text(
    "INSERT INTO skill_eval_run"
    " (tenant_id, repository, revision, scores, accepted, regressions)"
    " VALUES (:tenant, :repository, :revision, :scores, :accepted, :regressions)"
).bindparams(
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("scores", type_=sa.JSON()),
    sa.bindparam("regressions", type_=sa.JSON()),
)


async def test_one_grading_per_revision_is_the_primary_key(engine: AsyncEngine) -> None:
    """A second row for one revision is impossible, not merely avoided by the writer."""
    row = _row()
    async with engine.begin() as conn:
        await conn.execute(_RAW_INSERT, row)

    with pytest.raises(sa.exc.IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_RAW_INSERT, row)

    async with engine.connect() as conn:
        held = await conn.scalar(
            sa.text(
                "SELECT count(*) FROM skill_eval_run WHERE tenant_id = :tenant"
            ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
            {"tenant": row["tenant"]},
        )
    assert held == 1


@pytest.mark.parametrize(
    ("accepted", "regressions"),
    [
        (True, [{"skill": "extract", "baseline": 1, "scored": 0}]),
        (False, []),
    ],
)
async def test_accepted_and_regressions_cannot_contradict_each_other(
    engine: AsyncEngine, accepted: bool, regressions: list[dict[str, object]]
) -> None:
    """`Grade`'s one invariant, held in the store rather than in what writes to it.

    A bug in the writer then produces an error instead of a row that reads as an
    acceptance nobody granted.
    """
    with pytest.raises(sa.exc.IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                _RAW_INSERT, _row(accepted=accepted, regressions=regressions)
            )


@pytest.mark.parametrize("scores", [{}, [], "extract", None])
async def test_scores_must_be_a_non_empty_object(
    engine: AsyncEngine, scores: object
) -> None:
    """`NOT NULL` on a jsonb column does not say this.

    A JSON null, a string and an array are all perfectly good jsonb *values*, and every
    one of them would make `jsonb_each_text` in the baseline read fail far from here.
    """
    with pytest.raises(sa.exc.IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_RAW_INSERT, _row(scores=scores))


async def test_an_update_is_refused_rather_than_silently_ignored(
    engine: AsyncEngine,
) -> None:
    """A verdict a caller can rewrite is not a gate.

    The trigger raises, which is the mechanism every append-only table in this tree
    uses: a rewrite rule with DO INSTEAD NOTHING would leave the row correct while
    telling the writer it succeeded.
    """
    row = _row(accepted=False, regressions=[{"skill": "x", "baseline": 1, "scored": 0}])
    async with engine.begin() as conn:
        await conn.execute(_RAW_INSERT, row)

    with pytest.raises(sa.exc.DBAPIError):
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE skill_eval_run SET accepted = true"
                    " WHERE tenant_id = :tenant"
                ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
                {"tenant": row["tenant"]},
            )

    async with engine.connect() as conn:
        still = await conn.scalar(
            sa.text(
                "SELECT accepted FROM skill_eval_run WHERE tenant_id = :tenant"
            ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
            {"tenant": row["tenant"]},
        )
    assert still is False


async def test_two_tenants_may_hold_the_same_repository_and_revision(
    engine: AsyncEngine,
) -> None:
    """The key is (tenant, repository, revision), so one tenant cannot block another."""
    async with engine.begin() as conn:
        await conn.execute(_RAW_INSERT, _row())
        await conn.execute(_RAW_INSERT, _row())


async def test_the_baseline_is_the_highest_accepted_score_and_names_who_set_it(
    engine: AsyncEngine,
) -> None:
    """Derived from the runs, so there is no stored summary to drift from them.

    The tie case is the one worth stating: a later accepted run matching the bar leaves
    the attribution on the run that *reached* it, which is the revision MAP-A144 means
    by "the previously passing revision".
    """
    registry = PostgresDefinitionRegistry(engine)
    tenant = TenantId(uuid.uuid4())

    await registry.record_eval_run(
        tenant, REPOSITORY, PASSING, {"extract": 8700, "screen": 9200}, Grade(True, ())
    )
    await registry.record_eval_run(
        tenant, REPOSITORY, IMPROVED, {"extract": 9400, "screen": 9200}, Grade(True, ())
    )
    await registry.record_eval_run(
        tenant,
        REPOSITORY,
        REGRESSED,
        {"extract": 10_000},
        Grade(False, (Regression("extract", 9400, 10_000, IMPROVED),)),
    )

    assert await registry.eval_baselines(tenant, REPOSITORY) == (
        Baseline(skill="extract", score=9400, set_by_revision=IMPROVED),
        Baseline(skill="screen", score=9200, set_by_revision=PASSING),
    ), "a refused run set a bar, or a tie moved the attribution off the run that set it"

    assert await registry.eval_baselines(TenantId(uuid.uuid4()), REPOSITORY) == ()


async def test_a_revision_is_graded_once_and_the_second_call_reads_the_stored_verdict(
    engine: AsyncEngine,
) -> None:
    """Handed a contradicting `Grade`, the store still reports the one it holds.

    That is what makes the gate un-re-rollable: submitting again cannot change the
    answer, even when the caller computed a different one.
    """
    registry = PostgresDefinitionRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    refusal = Grade(False, (Regression("extract", 8700, None, PASSING),))

    first = await registry.record_eval_run(
        tenant, REPOSITORY, REGRESSED, {"screen": 9200}, refusal
    )
    again = await registry.record_eval_run(
        tenant, REPOSITORY, REGRESSED, {"screen": 10_000}, Grade(True, ())
    )

    assert (first.accepted, first.first_grading) == (False, True)
    assert (again.accepted, again.first_grading) == (False, False)
    assert again.regressions == refusal.regressions, (
        "a resubmission re-graded a revision; the verdict must come out of the row"
    )
    assert again.regressions[0].scored is None, (
        "an omitted skill's absent score did not survive the jsonb round trip"
    )

    async with engine.connect() as conn:
        rows = await conn.scalar(
            sa.text(
                "SELECT count(*) FROM skill_eval_run WHERE tenant_id = :tenant"
            ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
            {"tenant": tenant},
        )
    assert rows == 1


async def test_eval_facts_reports_enrollment_and_acceptance_from_one_read(
    engine: AsyncEngine,
) -> None:
    """Four states over the life of one repository, in the order they really occur."""
    registry = PostgresDefinitionRegistry(engine)
    tenant = TenantId(uuid.uuid4())

    assert await registry.eval_facts(tenant, REPOSITORY, PASSING) == EvalFacts(
        repository_enrolled=False, revision_accepted=False
    )

    await registry.record_eval_run(
        tenant,
        REPOSITORY,
        REGRESSED,
        {"extract": 10},
        Grade(False, (Regression("extract", 8700, 10, PASSING),)),
    )
    assert await registry.eval_facts(tenant, REPOSITORY, REGRESSED) == EvalFacts(
        repository_enrolled=True, revision_accepted=False
    ), "a repository whose only run failed must still be under the gate"

    await registry.record_eval_run(
        tenant, REPOSITORY, PASSING, {"extract": 8700}, Grade(True, ())
    )
    assert await registry.eval_facts(tenant, REPOSITORY, PASSING) == EvalFacts(
        repository_enrolled=True, revision_accepted=True
    )
    assert await registry.eval_facts(tenant, REPOSITORY, UNSUBMITTED) == EvalFacts(
        repository_enrolled=True, revision_accepted=False
    )


async def test_registering_and_resolving_still_work_beside_the_gate(
    engine: AsyncEngine,
) -> None:
    """The methods this slice added did not disturb the two that were already here."""
    registry = PostgresDefinitionRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    definition_id = new_definition_id()
    definition = AgentDefinition.model_validate(_definition_pinning(PASSING))

    assert await registry.register(definition_id, tenant, definition) == 1
    resolved = await registry.resolve(definition_id, tenant)

    assert resolved.revision == 1
    assert resolved.definition.skills_revision == PASSING


class UnusedWebhooks:
    """Satisfies the webhook store port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    webhook store would be grading something this file does not grade, and a quiet stub
    would let it pass while doing so.
    """

    async def register(
        self,
        tenant_id: TenantId,
        url: CallbackUrl,
        event_types: frozenset[str],
        secret_ref: str,
    ) -> WebhookRecord:
        raise AssertionError("a test in this file registered a webhook")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file listed a tenant's webhooks")

    async def delete(self, webhook_id: UUID, tenant_id: TenantId) -> bool:
        raise AssertionError("a test in this file deleted a webhook")

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file asked what watches a type")


FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
"""A digest-pinned image, because `Environment` refuses anything else."""


class AnyEnvironmentResolves:
    """An environment store that resolves any id to one fixed shape.

    Deliberately unable to refuse. Whether an unknown environment is refused, and what a
    refusal leaves behind, are graded in `test_environment_reference.py` where the store
    can say no -- asserting it here as well would put one behaviour in two places, free
    to disagree the first time the refusal changes shape.
    """

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        return {
            "id": str(environment_id),
            "tenant_id": str(tenant_id),
            "name": "session-fixture",
            "runtime_image": FIXTURE_IMAGE,
            "denied_paths": [],
        }


async def test_editing_an_agent_cannot_pin_a_revision_registering_one_could_not(
    harness: Harness,
) -> None:
    """The gate's second door: the version route takes a skills revision too.

    `POST /v1/agents` mints its own id, so it only ever writes revision 1 and is
    not how a live agent's skills change -- an edit is. That makes the two routes
    one surface for this purpose, and a gate on only the first has a corridor
    around it: register against the passing revision, then edit to the regressed
    one, and the agent every new Session resolves is running degraded skills.

    Both halves are asserted, because the refusal is the cheaper half.
    `writes == 2` (the registration plus one accepted edit) is what says the
    refused edit stored nothing, and re-resolving is what says the live agent is
    still on the revision that passed. A route that refuses and writes anyway is
    worse than no gate: the tenant believes the change did not land.
    """
    assert (await harness.submit(PASSING, extract=8700, screen=9200)).status_code == 200
    registered = await harness.pin(PASSING)
    assert registered.status_code == 201, registered.text
    agent_id = registered.json()["id"]

    # An edit that does not move the skills revision is not what is gated, and it
    # has to keep working -- otherwise the refusal below could be green because
    # editing is broken rather than because the gate is closed.
    permitted = await harness.edit(agent_id, PASSING)
    assert permitted.status_code == 201, permitted.text
    assert permitted.json()["version"] == 2

    degraded = await harness.submit(REGRESSED, extract=8100, screen=9200)
    assert degraded.status_code == 400, degraded.text

    refused = await harness.edit(agent_id, REGRESSED)
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "definition.skills_revision_not_accepted"
    assert refused.json()["error"]["detail"]["field"] == "skills_revision"

    assert harness.registry.writes == 2, (
        f"the registry was written {harness.registry.writes} times, not 2 (one "
        "registration, one permitted edit). A third write means the refused edit "
        "stored a revision anyway, which is worse than no gate at all: the tenant "
        "was told the change was refused."
    )
    resolved = await harness.registry.resolve(
        DefinitionId(UUID(agent_id)), harness.tenant
    )
    assert resolved.definition.skills_revision == PASSING, (
        "the live agent resolves the regressed revision. The edit door let "
        "through what the registration door refuses, so every new Session now "
        "runs degraded skills."
    )
