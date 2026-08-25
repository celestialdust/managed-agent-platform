"""A skill registered here reaches a definition, and one that cannot is refused.

Tier 1 (local, no infrastructure). Two altitudes in one file, and the split is what the
claim needs.

The resolution cases are a total function over frozen values and a store held in a dict,
so every refusal is reachable with no app and no database.

The route cases run the real routers over one in-memory store, because the property
under test is a relationship *between* two surfaces: a skill uploaded at one door has to
be attachable at another, and an id no upload ever minted has to be refused there. Two
stores could not show that, and asserting on a mock's calls would let both the response
and the stored row be wrong while the assertions passed.

**What is not proved here.** Nothing in this file puts a file inside a pod. The delivery
tuple is graded as a value -- the paths, the order, the bytes -- and what mounts it is
the pod manifest and the pod adapter, neither of which this slice touches. So "the agent
can see the skill" is not a claim this file makes, and the run record has to say so.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.catalog.definitions import AgentRecord
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.skills.evaluation import Baseline, EvalFacts, RunRecord
from managed_agent.control.skills.inventory import (
    RepositorySkillHeld,
    SkillRow,
)
from managed_agent.control.skills.registry import (
    SkillHeld,
    SkillListing,
    SkillRecord,
    SkillsUnresolvable,
    SkillVersionBundle,
    SkillVersionFile,
    SkillVersionRecord,
    UnconfiguredSkills,
    read_attached,
    resolve,
)
from managed_agent.core.ids import DefinitionId, SkillId, TenantId, new_skill_id
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.skill import (
    MAX_SKILLS_PER_AGENT,
    SKILL_DELIVERY_MAX_BYTES,
    SKILL_ROOTS,
    ValidatedSkill,
    parse_skill_md,
    repository_skill_id,
)

_SHA = "0" * 39 + "a"
_REPOSITORY = "git@github.com:acme/skills.git"


def _skill_md(name: str, description: str = "Build a PDF report.") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\nDo the thing.\n"


def _skill(name: str) -> ValidatedSkill:
    return parse_skill_md(_skill_md(name), source=f"{name}/SKILL.md")


def _definition(**overrides: object) -> AgentDefinition:
    body: dict[str, object] = {
        "name": "slr-reviewer",
        "instructions": "Extract findings and name the source document for each.",
        "model": "gpt-5-codex",
        "skills_repository": _REPOSITORY,
        "skills_revision": _SHA,
    } | overrides
    return AgentDefinition.model_validate(body)


def _attaching(*skill_ids: SkillId) -> AgentDefinition:
    return _definition(
        skills=[{"type": "custom", "skill_id": str(one)} for one in skill_ids]
    )


class FakeSkillStore:
    """The skill store in two dicts, satisfying the port structurally.

    A real store's own guarantees -- the tenant term in the WHERE clause, the
    append-only trigger, the conflicting insert that writes nothing -- are properties
    of PostgreSQL and are graded against it in
    `tests/adapters/test_skill_registry.py`. What this stands in for is only the
    holding, so the resolution policy above it can be exercised without a container.
    """

    def __init__(self) -> None:
        self.skills: dict[tuple[TenantId, SkillId], ValidatedSkill] = {}
        self.labels: dict[tuple[TenantId, SkillId], str | None] = {}
        self.repositories: dict[tuple[TenantId, str, str], list[ValidatedSkill]] = {}
        self.versions: dict[tuple[TenantId, SkillId, int], SkillVersionBundle] = {}

    async def add_skill(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        skill: ValidatedSkill,
        *,
        display_name: str | None = None,
    ) -> None:
        """Hold the skill and its label.

        Defaulted where the port requires the argument, which an implementation is
        allowed to be: it keeps the cases here that predate labels reading as they did,
        and none of them is about a label.
        """
        self.skills[(tenant_id, skill_id)] = skill
        self.labels[(tenant_id, skill_id)] = display_name

    async def read_skills(
        self, tenant_id: TenantId, skill_ids: Sequence[SkillId]
    ) -> tuple[SkillRecord, ...]:
        held = [
            SkillRecord(skill_id=one, skill=self.skills[(tenant_id, one)])
            for one in skill_ids
            if (tenant_id, one) in self.skills
        ]
        return tuple(sorted(held, key=lambda record: record.skill.name))

    async def set_repository_skills(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        skills: Sequence[ValidatedSkill],
    ) -> int:
        key = (tenant_id, repository, revision)
        if key in self.repositories:
            return 0
        self.repositories[key] = list(skills)
        return len(skills)

    async def read_repository_skills(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> tuple[ValidatedSkill, ...]:
        held = self.repositories.get((tenant_id, repository, revision), [])
        return tuple(sorted(held, key=lambda skill: skill.name))

    async def page_uploaded_skills(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillListing, ...]:
        raise AssertionError("a skill-delivery test listed a tenant's skills")

    async def read_skill(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> SkillHeld | None:
        """The skill, with `latest_version` derived the way the real store derives it.

        Greatest version with no retirement against it, or None when every version has
        been retired or none was ever written. Stated here rather than borrowed because
        the real one is a correlated subquery in SQL; if the two ever disagree, the
        adapter's own tests are what say which is wrong.
        """
        held = self.skills.get((tenant_id, skill_id))
        if held is None:
            return None
        live = [
            version
            for (tenant, skill, version), bundle in self.versions.items()
            if tenant == tenant_id and skill == skill_id and not bundle.record.retired
        ]
        return SkillHeld(
            skill_id=skill_id,
            name=held.name,
            description=held.description,
            display_name=self.labels.get((tenant_id, skill_id)),
            latest_version=max(live) if live else None,
            deleted=False,
        )

    async def delete_skill(self, tenant_id: TenantId, skill_id: SkillId) -> None:
        raise AssertionError("a skill-delivery test deleted a skill")

    async def add_skill_version(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        version: int,
        skill: ValidatedSkill,
        directory: str,
        files: Sequence[SkillVersionFile],
    ) -> None:
        self.versions[(tenant_id, skill_id, version)] = SkillVersionBundle(
            record=SkillVersionRecord(
                version=version,
                name=skill.name,
                description=skill.description,
                directory=directory,
                retired=False,
            ),
            skill_md=skill.text,
            files=tuple(files),
        )

    async def page_skill_versions(
        self, tenant_id: TenantId, skill_id: SkillId, after: int | None, limit: int
    ) -> tuple[SkillVersionRecord, ...]:
        raise AssertionError("a skill-delivery test paged a skill's versions")

    async def read_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionRecord | None:
        raise AssertionError("a skill-delivery test read one version")

    async def read_skill_version_bundle(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionBundle | None:
        return self.versions.get((tenant_id, skill_id, version))

    async def retire_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> None:
        """Tombstone the version, keeping it readable. A new value, never a mutation."""
        key = (tenant_id, skill_id, version)
        held = self.versions[key]
        self.versions[key] = replace(held, record=replace(held.record, retired=True))


_REGISTERED_AT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
"""When every agent in this file came into being.

Fixed rather than `now()`, because `AgentRecord.created_at` is published and a moving
value would make a response body differ between runs. Nothing here asserts on it -- what
these cases are about is which gate the append route consults -- so one moment for all
of them is the honest shape.
"""


class DefinitionsWithNoGate:
    """A definition registry that stores rows and enrolls nobody in the eval gate.

    Both halves are needed: the registration route writes definitions through this port
    and reads the CI eval gate through the same one, because that is what the real
    adapter is.
    """

    def __init__(self) -> None:
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

    async def resolve(self, definition_id: DefinitionId, tenant_id: TenantId) -> Any:
        raise AssertionError("a skill-delivery test resolved a definition")

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
        revisions = self._rows.get(definition_id, [])
        return revisions[revision - 1] if 1 <= revision <= len(revisions) else None

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a skill-delivery test archived a revision")

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        return ()

    async def record_eval_run(
        self, *args: object, **kwargs: object
    ) -> RunRecord:  # pragma: no cover - never submitted here
        raise AssertionError("a skill-delivery test submitted an eval run")

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        return EvalFacts(repository_enrolled=False, revision_accepted=False)

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
        raise AssertionError("a skill-delivery test listed a tenant's agents")

    async def archive_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> datetime | None:
        raise AssertionError("a skill-delivery test retired a whole agent")

    async def register_at_revision(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
        expected: int,
    ) -> int | None:
        raise AssertionError(
            "a skill-delivery test appended a revision against an expected number"
        )


class Unused:
    """One raising stand-in for every port a skill never touches.

    Raising rather than answering, so a route that reached one of these fails loudly
    here instead of passing on an answer nothing meant to give it.
    """

    def __getattr__(self, name: str) -> Any:
        async def refuse(*args: object, **kwargs: object) -> Any:
            raise AssertionError(f"a skill-delivery test called {name}")

        return refuse


@dataclass(frozen=True, slots=True)
class Harness:
    client: AsyncClient
    skills: FakeSkillStore
    definitions: DefinitionsWithNoGate
    tenant: TenantId

    async def upload(self, name: str) -> Any:
        return await self.client.post("/v1/skills", json={"skill_md": _skill_md(name)})

    async def submit_repository(
        self, files: dict[str, str], revision: str = _SHA
    ) -> Any:
        return await self.client.post(
            "/v1/skills/repository",
            json={
                "repository": _REPOSITORY,
                "revision": revision,
                "files": files,
            },
        )

    async def register(self, definition: dict[str, object]) -> Any:
        return await self.client.post("/v1/agents", json=definition)

    async def append_version(self, agent_id: str, body: dict[str, object]) -> Any:
        return await self.client.post(f"/v1/agents/{agent_id}/versions", json=body)


class InventoryInMemory:
    """Assigns repository-skill ids the way the real store does, and reads none back.

    Here because the submission route now asks for ids, and `NoSkillInventory` -- the
    default on a `Platform` built without a store -- refuses rather than answering
    empty. That refusal is deliberate: an empty inventory is indistinguishable from a
    tenant who holds no skills. It does mean a test exercising the submission door has
    to supply one, which is this.

    Ids come from `repository_skill_id` rather than from a counter, so this fake has the
    property the tests below actually rely on: resubmitting a commit computes the ids it
    computed the first time. A counter would make a resubmission look like new work and
    the no-op test would pass for the wrong reason.

    The two reads raise. Nothing in this file lists or reads a skill back by id, and a
    fake that answered them would be a fake nobody checked -- if a route starts calling
    them, this should fail loudly rather than return something invented.
    """

    def __init__(self) -> None:
        self.assigned: list[tuple[str, str, str]] = []

    async def assign_repository_ids(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        names: Sequence[str],
    ) -> tuple[tuple[str, SkillId], ...]:
        self.assigned.extend((repository, revision, name) for name in names)
        return tuple(
            (name, repository_skill_id(tenant_id, repository, revision, name))
            for name in names
        )

    async def repository_skill_at(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> RepositorySkillHeld | None:
        raise AssertionError("a test in this file read a repository skill by id")

    async def page(
        self,
        tenant_id: TenantId,
        after: tuple[str, SkillId] | None,
        limit: int,
    ) -> tuple[SkillRow, ...]:
        raise AssertionError("a test in this file listed the skill inventory")


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    skills = FakeSkillStore()
    definitions = DefinitionsWithNoGate()
    unused = Unused()
    platform = Platform(
        event_log_append=unused,
        event_log_range=unused,
        definition_registry=definitions,
        tool_registry=unused,
        session_registry=unused,
        webhooks=unused,
        environment_store=unused,
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
        skill_store=skills,
        skill_inventory=InventoryInMemory(),
    )
    tenant = TenantId(uuid.uuid4())
    async with AsyncClient(
        transport=ASGITransport(app=create_app(platform)),
        base_url="http://control-plane",
        headers={TENANT_HEADER: str(tenant)},
    ) as client:
        yield Harness(
            client=client, skills=skills, definitions=definitions, tenant=tenant
        )


async def test_an_uploaded_skill_comes_back_with_an_id_a_definition_can_attach(
    harness: Harness,
) -> None:
    """The upload door and the attach door, joined -- which is the whole slice.

    Before this existed, a definition's skill fields validated and nothing consumed
    them. What makes that fixed is not that an upload succeeds: it is that the id an
    upload returns is one a definition is allowed to name and a Session can resolve.
    """
    uploaded = await harness.upload("pdf-report")
    assert uploaded.status_code == 201, uploaded.text
    skill_id = uploaded.json()["id"]
    assert uploaded.json()["name"] == "pdf-report", (
        "the response does not report the name read out of the document, so an "
        "uploader who mistyped one would find out from an agent that never invokes it"
    )

    registered = await harness.register(
        _definition(skills=[{"type": "custom", "skill_id": skill_id}]).model_dump(
            mode="json"
        )
    )

    assert registered.status_code == 201, registered.text
    stored = await harness.definitions.read_version(
        DefinitionId(UUID(registered.json()["id"])), harness.tenant, 1
    )
    assert stored is not None
    assert [str(one.skill_id) for one in stored.skills] == [skill_id]


async def test_a_definition_naming_a_skill_id_nobody_uploaded_is_refused(
    harness: Harness,
) -> None:
    """The accept-and-deliver-nothing defect, one level up, refused at the door.

    An id that resolves to nothing is exactly what the old `skills_repository` field
    was: a value that validated and reached no runtime. Attaching one has to be a
    refusal rather than an agent that quietly has no skill -- and the refusal has to
    leave no definition behind, because a 201 that stored the row anyway is worse than
    no check at all.
    """
    refused = await harness.register(_attaching(new_skill_id()).model_dump(mode="json"))

    assert refused.status_code == 400, refused.text
    body = refused.json()
    assert body["error"]["code"] == "definition.invalid"
    assert body["error"]["detail"]["field"] == "skills"
    assert body["error"]["detail"]["unresolved_skills"] == 1
    assert harness.definitions.writes == 0, (
        "the refused registration wrote a definition anyway, so a definition naming an "
        "unresolvable skill is stored and will be resolved by the next Session"
    )


async def test_the_refusal_for_an_unattachable_id_says_what_an_id_attaches(
    harness: Harness,
) -> None:
    """An id that names nothing here is refused, and the refusal cannot claim too much.

    A repository skill has an id of its own now, and a tenant handed one by a listing
    is entitled to try attaching it. What they must not be told is that the platform
    holds no such skill -- that is the platform contradicting its own output. So the
    refusal says what an id attaches and how the other route is attached instead,
    rather than asserting anything about what the tenant does or does not hold.
    """
    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching(new_skill_id()))

    message = raised.value.message
    assert "POST /v1/skills" in message, (
        f"the refusal does not say what kind of id attaches a skill: {message}"
    )
    assert "skills_revision" in message, (
        "the refusal does not say how a repository skill is attached, so a tenant "
        f"holding one of its ids has nowhere to go: {message}"
    )
    assert "not registered to this tenant" not in message, (
        "the refusal still claims the tenant holds no such skill, which is false for "
        f"the id of a skill their own CI submitted: {message}"
    )


async def test_appending_a_version_cannot_attach_a_skill_registering_one_could_not(
    harness: Harness,
) -> None:
    """The second door. A check on the first alone is a check with a corridor round it.

    `POST /v1/agents` mints its own id and can only write revision 1, so an edit is the
    only way a live agent's skills change. This route accepts a whole `AgentDefinition`
    and therefore a `skills` array, so it has to answer the same way.
    """
    created = await harness.register(_definition().model_dump(mode="json"))
    agent_id = created.json()["id"]
    before = harness.definitions.writes

    refused = await harness.append_version(
        agent_id, _attaching(new_skill_id()).model_dump(mode="json")
    )

    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "definition.invalid"
    assert harness.definitions.writes == before, (
        "the refused edit appended a revision anyway, so the gate on the registration "
        "door can be walked around by editing an agent instead of creating one"
    )


async def test_another_tenants_skill_id_reads_as_absent_rather_than_forbidden(
    harness: Harness,
) -> None:
    """A caller holding an id learns nothing about whether it exists elsewhere.

    The tenant is a term in the read, so a foreign id is indistinguishable from one
    nobody ever minted -- which is the same posture `definition.not_found` takes.
    """
    other_tenant = TenantId(uuid.uuid4())
    theirs = new_skill_id()
    await harness.skills.add_skill(other_tenant, theirs, _skill("pdf-report"))

    refused = await harness.register(_attaching(theirs).model_dump(mode="json"))

    assert refused.status_code == 400
    assert refused.json()["error"]["detail"]["skill_id"] == str(theirs)


async def test_the_anthropic_type_is_refused_through_the_real_http_surface(
    harness: Harness,
) -> None:
    """The documented payload, sent as a caller would send it, and the answer it gets.

    Graded here as well as at the parse because the value of the decision is what a
    caller sees: a 400 whose body names the catalogue and says what to do instead. A
    caller told only "input should be 'custom'" would not learn that the four pre-built
    ids mean something on another platform and nothing here.
    """
    refused = await harness.register(
        {
            "name": "slr-reviewer",
            "instructions": "Extract findings.",
            "model": "gpt-5-codex",
            "skills_repository": _REPOSITORY,
            "skills_revision": _SHA,
            "skills": [{"type": "anthropic", "skill_id": "pdf"}],
        }
    )

    assert refused.status_code == 400, refused.text
    assert "POST /v1/skills" in refused.text, (
        "the refusal does not tell the caller how to attach the skill instead: "
        f"{refused.text}"
    )
    assert harness.definitions.writes == 0


async def test_a_malformed_upload_is_refused_at_the_door_naming_the_reason(
    harness: Harness,
) -> None:
    """Refused where it was submitted, not discovered as missing inside a pod.

    This is the whole reason validation lives at the boundary: the submitter is still on
    the other end of the connection and can read why, where a skill validated any later
    is validated somewhere a malformed one and a missing one look identical.
    """
    refused = await harness.client.post(
        "/v1/skills", json={"skill_md": "no frontmatter at all"}
    )

    assert refused.status_code == 400, refused.text
    assert "frontmatter" in refused.text
    assert harness.skills.skills == {}


async def test_a_repository_submission_gives_the_pinned_revision_a_meaning(
    harness: Harness,
) -> None:
    """`skills_repository` and `skills_revision` resolve to what was submitted for them.

    The platform does not go and read the repository -- the API this one is shaped after
    says its own repository discovery does not run for self-hosted sandboxes, which is
    what we are. So the checkout's holder submits, and the pin then means exactly what
    it always claimed to: the skills of that commit.
    """
    submitted = await harness.submit_repository(
        {
            f"{SKILL_ROOTS[0]}/pdf-report/SKILL.md": _skill_md("pdf-report"),
            f"{SKILL_ROOTS[0]}/citation-check/SKILL.md": _skill_md("citation-check"),
            "README.md": "# not a skill",
        }
    )

    assert submitted.status_code == 201, submitted.text
    assert submitted.json()["skills"] == ["citation-check", "pdf-report"]
    assert submitted.json()["newly_recorded"] == 2

    delivered = await resolve(harness.skills, harness.tenant, _definition())

    assert [one.relative_path for one in delivered] == [
        "skills/citation-check/SKILL.md",
        "skills/pdf-report/SKILL.md",
    ]


async def test_resubmitting_the_same_commit_writes_nothing_and_says_so(
    harness: Harness,
) -> None:
    """A commit's skills do not change, so the second submission is a retry.

    A CI job is retried for reasons that have nothing to do with the code under it, and
    a retry that answered differently would make the pipeline's own flakiness look like
    a change to the skills.
    """
    files = {f"{SKILL_ROOTS[0]}/pdf-report/SKILL.md": _skill_md("pdf-report")}
    await harness.submit_repository(files)

    again = await harness.submit_repository(files)

    assert again.status_code == 201
    assert again.json()["newly_recorded"] == 0
    assert again.json()["skills"] == ["pdf-report"]


async def test_a_repository_submission_holding_a_near_miss_is_refused_whole(
    harness: Harness,
) -> None:
    """One unreachable path refuses the submission rather than being passed over.

    Registering the good ones and dropping this one is the failure: the submitter would
    be told 201 and the agent would be missing exactly the skill nobody mentioned.
    """
    refused = await harness.submit_repository(
        {
            f"{SKILL_ROOTS[0]}/pdf-report/SKILL.md": _skill_md("pdf-report"),
            "skills/citation-check/SKILL.md": _skill_md("citation-check"),
        }
    )

    assert refused.status_code == 400, refused.text
    assert "skills/citation-check/SKILL.md" in refused.text
    assert harness.skills.repositories == {}


async def test_a_definition_pinning_a_revision_nobody_submitted_resolves_to_nothing(
    harness: Harness,
) -> None:
    """Empty is the true answer, and it is not a refusal.

    Every definition registered before this table existed pins a revision with no
    submission behind it, and an agent with no repository skills is a real thing to
    want. This mirrors the eval gate exactly: the mechanism turns on for a pair with its
    first submission.
    """
    assert await resolve(harness.skills, harness.tenant, _definition()) == ()


async def test_two_skills_resolving_to_one_name_are_refused_before_a_session_starts(
    harness: Harness,
) -> None:
    """One flat delivery directory means two files at one path, so one would vanish.

    A deliberate difference from the API this is shaped after, where a repository skill
    and an attached skill may share a name and both stay available under their own
    paths. Here they would not, so the collision is refused where it can still be
    explained rather than discovered as a skill that stopped working.
    """
    attached = new_skill_id()
    await harness.skills.add_skill(harness.tenant, attached, _skill("pdf-report"))
    await harness.skills.set_repository_skills(
        harness.tenant, _REPOSITORY, _SHA, [_skill("pdf-report")]
    )

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching(attached))

    assert raised.value.detail["skill"] == "pdf-report"
    assert raised.value.detail["first_origin"] == "attached"
    assert raised.value.detail["second_origin"] == "repository"
    assert "skills/pdf-report/SKILL.md" in raised.value.message, (
        "the refusal does not name the path the two would collide at, so the reason "
        f"one name is a problem is not visible: {raised.value.message}"
    )


_V1 = 1774000000000000
"""One version, spelled as the platform mints them: microseconds since the epoch."""

_V2 = _V1 + 1_000_000
_V3 = _V1 + 2_000_000


async def _write_version(
    harness: Harness,
    skill_id: SkillId,
    version: int,
    *,
    name: str = "pdf-report",
    body: str | None = None,
    files: Sequence[SkillVersionFile] = (),
) -> None:
    """Put one version of a skill in the store, the way the create route would.

    Through the port rather than into the dict, so a test cannot write a shape the
    real store would not hold.
    """
    document = parse_skill_md(
        body if body is not None else _skill_md(name, f"Version {version}."),
        source=f"{name}/SKILL.md",
    )
    await harness.skills.add_skill_version(
        harness.tenant, skill_id, version, document, name, files
    )


def _attaching_at(skill_id: SkillId, version: int) -> AgentDefinition:
    return _definition(
        skills=[{"type": "custom", "skill_id": str(skill_id), "version": str(version)}]
    )


async def test_a_skills_label_is_held_beside_it_and_never_delivered(
    harness: Harness,
) -> None:
    """A display name is for people reading a list, and the model is never shown it.

    Two properties, and the second is the one worth a test. The label is stored and read
    back, so a tenant who set one is not told the platform kept nothing -- and it is
    absent from every delivered byte, because the document is what the model reads and a
    label injected into it would be this platform putting words in a tenant's skill.

    Graded through the port rather than against Postgres: what this pins is the shape
    the store is addressed by, which is the thing every implementation has to agree on.
    The column and its append-only trigger are the adapter's own tests' subject.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(
        harness.tenant,
        skill_id,
        _skill("pdf-report"),
        display_name="PDF Report Builder",
    )

    held = await harness.skills.read_skill(harness.tenant, skill_id)
    delivered = await resolve(harness.skills, harness.tenant, _attaching(skill_id))

    assert held is not None and held.display_name == "PDF Report Builder"
    assert held.name == "pdf-report", (
        "the label overwrote the name, and the name is what the runtime announces the "
        "skill's directory as"
    )
    assert not any("PDF Report Builder" in one.text for one in delivered), (
        "the label reached the delivered document, so this platform is putting words "
        "into what the model reads that the tenant did not write there"
    )


async def test_a_pinned_version_delivers_its_own_files_and_not_the_latest_ones(
    harness: Harness,
) -> None:
    """The whole point of the pin, and of the field that used to refuse every number.

    Two properties in one case because they are one claim: what arrives is the pinned
    version's document *and* the files stored beside it. A pin that delivered the right
    SKILL.md and the newest `forms.md` would be worse than no pin at all -- the model
    would be reading one version's instructions against another's reference.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(
        harness,
        skill_id,
        _V1,
        body=_skill_md("pdf-report", "Fill a form."),
        files=(SkillVersionFile(path="forms.md", text="the old forms"),),
    )
    await _write_version(
        harness,
        skill_id,
        _V2,
        body=_skill_md("pdf-report", "Fill a form better."),
        files=(SkillVersionFile(path="forms.md", text="the new forms"),),
    )

    delivered = await resolve(
        harness.skills, harness.tenant, _attaching_at(skill_id, _V1)
    )

    assert {one.relative_path: one.text for one in delivered} == {
        "skills/pdf-report/SKILL.md": _skill_md("pdf-report", "Fill a form."),
        "skills/pdf-report/forms.md": "the old forms",
    }


async def test_a_pinned_version_delivers_every_file_stored_beside_the_document(
    harness: Harness,
) -> None:
    """A nested sibling arrives nested, because that is where the document points.

    Anthropic's published `pdf` skill tells the model to read `reference.md`, and a
    skill whose references resolve to nothing is the defect versions exist to remove.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(
        harness,
        skill_id,
        _V1,
        files=(
            SkillVersionFile(path="reference/fonts.md", text="the fonts"),
            SkillVersionFile(path="forms.md", text="the forms"),
        ),
    )

    delivered = await resolve(
        harness.skills, harness.tenant, _attaching_at(skill_id, _V1)
    )

    assert [one.relative_path for one in delivered] == [
        "skills/pdf-report/SKILL.md",
        "skills/pdf-report/forms.md",
        "skills/pdf-report/reference/fonts.md",
    ]


async def test_latest_delivers_the_greatest_version_nobody_has_retired(
    harness: Harness,
) -> None:
    """One rule for what `latest` means, shared with the read that publishes it.

    Retiring the newest version moves `latest` to the newest survivor, which is exactly
    the case retirement exists for: a regression shipped, and every Session started
    afterwards has to stop getting it. A `latest` that still resolved to the retired
    version would make the retire route a button that does nothing.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(harness, skill_id, _V1)
    await _write_version(harness, skill_id, _V2)
    await _write_version(harness, skill_id, _V3)
    await harness.skills.retire_skill_version(harness.tenant, skill_id, _V3)

    delivered = await resolve(harness.skills, harness.tenant, _attaching(skill_id))

    assert [one.relative_path for one in delivered] == ["skills/pdf-report/SKILL.md"]
    assert "Version " + str(_V2) in delivered[0].text, (
        "`latest` did not resolve to the greatest version nobody has retired, so it "
        f"means something different here than it does on a skill read: {delivered[0]}"
    )


async def test_a_skill_with_no_versions_still_delivers_its_stored_body(
    harness: Harness,
) -> None:
    """Every skill uploaded before versions existed has none, and none may be stranded.

    Refusing them would turn this change into an outage for every agent already
    running: the fallback is not a convenience, it is the reason the change can ship.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))

    delivered = await resolve(harness.skills, harness.tenant, _attaching(skill_id))

    assert [one.relative_path for one in delivered] == ["skills/pdf-report/SKILL.md"]
    assert delivered[0].text == _skill_md("pdf-report")


async def test_a_pinned_version_nobody_wrote_is_refused_rather_than_served_latest(
    harness: Harness,
) -> None:
    """Falling back to latest is the defect this whole change exists to remove.

    A caller who pinned a version and was quietly served a different one has exactly
    the failure the old blanket refusal was protecting against, only now with a number
    in the definition to make it look like it worked.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(harness, skill_id, _V1)

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching_at(skill_id, _V2))

    assert raised.value.detail["version"] == str(_V2)
    assert raised.value.detail["skill_id"] == str(skill_id)
    assert "latest" in raised.value.message, (
        "the refusal does not say that the pin was not quietly answered with the "
        f"latest version, which is the thing a caller would assume: {raised.value}"
    )


async def test_a_pinned_version_that_was_retired_is_refused_at_the_door(
    harness: Harness,
) -> None:
    """Retirement is the statement that this version may not resolve any more.

    Serving it would make the retire route a no-op in its motivating case -- the
    definition pinning the regressed version is precisely the one a retirement is
    aimed at. The version stays readable and downloadable; what stops is placement
    into a new Session.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(harness, skill_id, _V1)
    await harness.skills.retire_skill_version(harness.tenant, skill_id, _V1)

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching_at(skill_id, _V1))

    assert raised.value.detail["version"] == str(_V1)
    assert "retired" in raised.value.message, (
        f"the refusal does not say the version was retired: {raised.value}"
    )


async def test_two_sibling_paths_that_flatten_to_one_secret_key_are_refused(
    harness: Harness,
) -> None:
    """A nested path can reintroduce the collision one flat directory was refusing.

    Both of these are paths the upload door accepts, and both become one entry in the
    Secret that carries them, so one file would arrive and the other would not. The
    same fold is refused again where the Secret is written; two checks, because the
    one that can name both paths to a tenant is this one.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(
        harness,
        skill_id,
        _V1,
        files=(
            SkillVersionFile(path="a/b.md", text="nested"),
            SkillVersionFile(path="a_b.md", text="flat"),
        ),
    )

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching_at(skill_id, _V1))

    assert "skills/pdf-report/a/b.md" in raised.value.message
    assert "skills/pdf-report/a_b.md" in raised.value.message


async def test_a_stored_sibling_path_that_climbs_out_of_the_directory_is_refused(
    harness: Harness,
) -> None:
    """The third check on a path, and the one the filesystem cannot do for us.

    The upload door refuses this and so does the store, and it is refused here too
    because a delivered path is joined onto a directory inside a Session's workspace:
    the writer that produced the row is not knowable from the row, so the set is
    checked where it is built rather than trusted for having been checked twice.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(
        harness,
        skill_id,
        _V1,
        files=(SkillVersionFile(path="../../etc/passwd", text="nope"),),
    )

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching_at(skill_id, _V1))

    assert raised.value.detail["path"] == "../../etc/passwd"


async def test_a_version_that_renames_a_skill_onto_another_name_is_refused(
    harness: Harness,
) -> None:
    """A version may rename its skill, which is why the collision is decided here.

    The create route deliberately does not guard the rename -- it says so -- because
    the two names only collide once both have been resolved. What that promise costs
    is this check being over the name each version's own document declares, rather
    than over the name the skill was uploaded under.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(harness, skill_id, _V1, name="citation-check")
    await harness.skills.set_repository_skills(
        harness.tenant, _REPOSITORY, _SHA, [_skill("citation-check")]
    )

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching_at(skill_id, _V1))

    assert raised.value.detail["skill"] == "citation-check"


async def test_the_same_skill_pinned_at_two_versions_is_refused_naming_both(
    harness: Harness,
) -> None:
    """Two versions of one skill are two file sets for one directory.

    Not a duplicate attachment -- the two are genuinely different grants, which is why
    the parse keeps both -- and not deliverable either, because they land on one path.
    Refused here with both versions in the message, where a tenant can see which two.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(harness, skill_id, _V1)
    await _write_version(harness, skill_id, _V2)
    definition = _definition(
        skills=[
            {"type": "custom", "skill_id": str(skill_id), "version": str(_V1)},
            {"type": "custom", "skill_id": str(skill_id), "version": str(_V2)},
        ]
    )

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, definition)

    assert raised.value.detail["skill_id"] == str(skill_id)
    assert str(_V1) in raised.value.message and str(_V2) in raised.value.message


async def test_a_delivery_too_heavy_for_the_secret_is_refused_naming_its_size(
    harness: Harness,
) -> None:
    """The count of skills stopped bounding the Secret when one skill became many files.

    Sixteen skills was half a megabyte when a skill was one 32 KiB document. A version
    carries up to thirty-two of those, so the count now bounds sixteen megabytes and
    the thing that has to be bounded is the total. The refusal names the actual weight,
    because a tenant told only "too big" cannot tell what to remove.
    """
    skill_id = new_skill_id()
    await harness.skills.add_skill(harness.tenant, skill_id, _skill("pdf-report"))
    await _write_version(
        harness,
        skill_id,
        _V1,
        files=tuple(
            SkillVersionFile(path=f"bulk-{n:02d}.md", text="x" * (32 * 1024))
            for n in range(20)
        ),
    )

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching_at(skill_id, _V1))

    assert raised.value.detail["resolved_bytes"] == pytest.approx(
        20 * 32 * 1024, abs=1024
    )
    assert raised.value.detail["limit"] == SKILL_DELIVERY_MAX_BYTES
    assert str(raised.value.detail["resolved_bytes"]) in raised.value.message


async def test_the_two_routes_together_cannot_exceed_what_a_session_carries(
    harness: Harness,
) -> None:
    """The parse-time bound sees only the attached half; the sum is what has to fit.

    A repository can bring skills after the definition was registered, so this is the
    one count that cannot be settled at the door -- and it has to be settled somewhere,
    because a Secret over 1 MiB is a pod that never starts.
    """
    attached = []
    for n in range(MAX_SKILLS_PER_AGENT):
        one = new_skill_id()
        await harness.skills.add_skill(harness.tenant, one, _skill(f"attached-{n:02d}"))
        attached.append(one)
    await harness.skills.set_repository_skills(
        harness.tenant, _REPOSITORY, _SHA, [_skill("from-repository")]
    )

    with pytest.raises(SkillsUnresolvable) as raised:
        await resolve(harness.skills, harness.tenant, _attaching(*attached))

    assert raised.value.detail["resolved_skills"] == MAX_SKILLS_PER_AGENT + 1
    assert raised.value.detail["limit"] == MAX_SKILLS_PER_AGENT


async def test_a_definition_attaching_nothing_never_asks_the_store_anything() -> None:
    """No question to ask, so the store is not consulted -- and this is what it buys.

    It keeps every definition written before this existed working on a platform with no
    skill store wired, while one that *does* attach a skill still fails loudly there.
    The alternative -- a store that answered emptily -- would make an unwired platform
    indistinguishable from a tenant who attached none, and every agent would quietly
    have no skills.
    """
    assert (
        await read_attached(UnconfiguredSkills(), TenantId(uuid.uuid4()), _definition())
        == ()
    )


async def test_an_unwired_skill_store_refuses_loudly_rather_than_answering_emptily(
    harness: Harness,
) -> None:
    """The default has to be a refusing object and not an empty one.

    An empty store is a perfectly plausible answer, so a platform assembled without one
    would serve every route, start every Session, and give every agent none of the
    skills its definition names -- which is the exact silence this whole slice removes.
    """
    with pytest.raises(RuntimeError, match="no skill store is wired"):
        await read_attached(
            UnconfiguredSkills(), harness.tenant, _attaching(new_skill_id())
        )
