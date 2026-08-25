"""Versions of a skill: writing one, reading one back, retiring one, deleting the skill.

Tier 1 (local, no infrastructure). Every claim here is a property of the routes and of
the policy above the store -- which version a listing shows, what `latest_version` is
after a retirement, what a deleted skill answers, whether a collision is retried -- and
each is reachable without a database.

The store's own guarantees are not claimed here. That the tenant is a term in the SQL,
that the primary key refuses a second version at one microsecond, and that the path and
directory checks hold against a writer that never loads our code are properties of
PostgreSQL, graded in `tests/adapters/test_skill_version_schema.py`.

So the fake below is not a convenient one. It refuses a duplicate `(skill, version)` the
way the primary key does, so the retry the route performs is exercised rather than
assumed; it refuses the same page window the adapter refuses; it excludes retired
versions from the listing and keeps them readable one at a time; and it records the
moment of a delete once, so "a retry does not move the recorded moment" is a claim this
file can actually fail on.

**Two tenants appear in the isolation tests on purpose.** A route that dropped its
tenant term reads perfectly normally with one tenant in the store, so a single-tenant
test proves nothing at all about isolation.
"""

from __future__ import annotations

import time
import uuid
import zipfile
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import pytest
from fastapi import UploadFile
from fixtures.anthropic_pdf_skill import REFERENCED_FILES, SKILL_MD
from httpx import ASGITransport, AsyncClient

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import skill_versions
from managed_agent.control.api.routes.skill_bundles import (
    MAX_BUNDLE_FILES,
    REASON_BUNDLE_INVALID,
    parse_bundle,
)
from managed_agent.control.api.routes.skill_versions import (
    MAX_VERSION_PAGE_SIZE,
    REASON_CURSOR_INVALID,
    version_cursor,
)
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.skills.inventory import (
    RepositorySkillHeld,
    SkillRow,
)
from managed_agent.control.skills.registry import (
    SkillHeld,
    SkillListing,
    SkillRecord,
    SkillVersionBundle,
    SkillVersionCollision,
    SkillVersionFile,
    SkillVersionRecord,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import SkillId, TenantId
from managed_agent.core.registration.skill import SKILL_MD_MAX_BYTES, ValidatedSkill

_ADAPTER_MAX_VERSION_PAGE = 1024
"""The window the adapter serves, mirrored so this fake refuses what it refuses."""

_PAGE_BUDGET = 20
_NEVER_ENDED = (
    f"the walk read {_PAGE_BUDGET} pages without a null next_page; it is either "
    "repeating rows at the page boundary or never reporting the end"
)


def _skill_md(name: str, body: str = "Do the thing.") -> str:
    return f"---\nname: {name}\ndescription: Build a report.\n---\n\n{body}\n"


@dataclass(frozen=True, slots=True)
class _Stored:
    """One version as the fake holds it, keyed outside by `(tenant, skill, version)`."""

    name: str
    description: str
    body: str
    directory: str
    files: tuple[SkillVersionFile, ...]


class SkillsHeldInMemory:
    """The skill store in dicts, with the version rules the real schema enforces.

    `moments` stands in for `now()`. It is what lets a test assert that a retried delete
    or a retried retirement keeps the moment the first one recorded -- neither response
    carries a timestamp, so the store's own record is the only place that claim is
    checkable.
    """

    def __init__(self) -> None:
        self.skills: dict[tuple[TenantId, SkillId], ValidatedSkill] = {}
        self.labels: dict[tuple[TenantId, SkillId], str | None] = {}
        self.versions: dict[tuple[TenantId, SkillId, int], _Stored] = {}
        self.deleted: dict[tuple[TenantId, SkillId], int] = {}
        self.retired: dict[tuple[TenantId, SkillId, int], int] = {}
        self.moments = 0

    def _moment(self) -> int:
        self.moments += 1
        return self.moments

    def _live(self, tenant_id: TenantId, skill_id: SkillId) -> list[int]:
        return sorted(
            version
            for (owner, held, version) in self.versions
            if (owner, held) == (tenant_id, skill_id)
            and (tenant_id, skill_id, version) not in self.retired
        )

    async def add_skill(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        skill: ValidatedSkill,
        *,
        display_name: str | None,
    ) -> None:
        self.skills[(tenant_id, skill_id)] = skill
        self.labels[(tenant_id, skill_id)] = display_name

    async def read_skill(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> SkillHeld | None:
        skill = self.skills.get((tenant_id, skill_id))
        if skill is None:
            return None
        return SkillHeld(
            skill_id=skill_id,
            name=skill.name,
            description=skill.description,
            display_name=self.labels.get((tenant_id, skill_id)),
            latest_version=max(self._live(tenant_id, skill_id), default=None),
            deleted=(tenant_id, skill_id) in self.deleted,
        )

    async def delete_skill(self, tenant_id: TenantId, skill_id: SkillId) -> None:
        if (tenant_id, skill_id) in self.skills:
            self.deleted.setdefault((tenant_id, skill_id), self._moment())

    async def add_skill_version(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        version: int,
        skill: ValidatedSkill,
        directory: str,
        files: Sequence[SkillVersionFile],
    ) -> None:
        key = (tenant_id, skill_id, version)
        if key in self.versions:
            raise SkillVersionCollision(f"version {version} is taken")
        self.versions[key] = _Stored(
            name=skill.name,
            description=skill.description,
            body=skill.text,
            directory=directory,
            files=tuple(files),
        )

    async def page_skill_versions(
        self, tenant_id: TenantId, skill_id: SkillId, after: int | None, limit: int
    ) -> tuple[SkillVersionRecord, ...]:
        if limit < 1 or limit > _ADAPTER_MAX_VERSION_PAGE:
            raise ValueError(f"version page limit {limit} is outside the window")
        found = sorted(self._live(tenant_id, skill_id), reverse=True)
        if after is not None:
            found = [version for version in found if version < after]
        return tuple(
            self._record(tenant_id, skill_id, version) for version in found[:limit]
        )

    def _record(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionRecord:
        held = self.versions[(tenant_id, skill_id, version)]
        return SkillVersionRecord(
            version=version,
            name=held.name,
            description=held.description,
            directory=held.directory,
            retired=(tenant_id, skill_id, version) in self.retired,
        )

    async def read_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionRecord | None:
        if (tenant_id, skill_id, version) not in self.versions:
            return None
        return self._record(tenant_id, skill_id, version)

    async def read_skill_version_bundle(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionBundle | None:
        held = self.versions.get((tenant_id, skill_id, version))
        if held is None:
            return None
        return SkillVersionBundle(
            record=self._record(tenant_id, skill_id, version),
            skill_md=held.body,
            files=held.files,
        )

    async def retire_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> None:
        if (tenant_id, skill_id, version) in self.versions:
            self.retired.setdefault((tenant_id, skill_id, version), self._moment())

    async def read_skills(
        self, tenant_id: TenantId, skill_ids: Sequence[SkillId]
    ) -> tuple[SkillRecord, ...]:
        raise AssertionError("a version test resolved a definition's skills")

    async def page_uploaded_skills(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillListing, ...]:
        raise AssertionError("a version test paged a tenant's skills")

    async def set_repository_skills(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        skills: Sequence[ValidatedSkill],
    ) -> int:
        raise AssertionError("a version test submitted a repository's skills")

    async def read_repository_skills(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> tuple[ValidatedSkill, ...]:
        raise AssertionError("a version test read a repository's skills")


class InventoryInMemory:
    """The combined inventory for a harness whose skills all arrive by upload.

    Every method is answered from what the store actually holds rather than stubbed, and
    the repository half is genuinely empty here -- no test in this file submits a
    checkout. That is the difference that matters: `repository_skill_at` returning None
    is this fake telling the truth about an empty half, not a stand-in shrugging. A fake
    that raised would make the version routes' 409 arm unreachable; one that returned a
    row would make the 404 arm unreachable.
    """

    def __init__(self, skills: SkillsHeldInMemory) -> None:
        self._skills = skills

    async def assign_repository_ids(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        names: Sequence[str],
    ) -> tuple[tuple[str, SkillId], ...]:
        raise AssertionError("a version test submitted a repository's skills")

    async def repository_skill_at(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> RepositorySkillHeld | None:
        return None

    async def page(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillRow, ...]:
        raise AssertionError("a version test paged the inventory")


class Unused:
    """One raising stand-in for every port a version route never touches."""

    def __getattr__(self, name: str) -> Any:
        async def refuse(*args: object, **kwargs: object) -> Any:
            raise AssertionError(f"a version test called {name}")

        return refuse


@dataclass(frozen=True, slots=True)
class Harness:
    """One app, one store, and two tenants that both talk to it."""

    store: SkillsHeldInMemory
    owner: AsyncClient
    stranger: AsyncClient

    async def upload(self, name: str = "pdf", *, stranger: bool = False) -> str:
        client = self.stranger if stranger else self.owner
        answer = await client.post("/v1/skills", json={"skill_md": _skill_md(name)})
        assert answer.status_code == 201, answer.text
        return str(answer.json()["id"])

    async def add_version(
        self,
        skill_id: str,
        parts: Sequence[tuple[str, str]],
        *,
        stranger: bool = False,
    ) -> Any:
        client = self.stranger if stranger else self.owner
        return await client.post(
            f"/v1/skills/{skill_id}/versions",
            files=[
                ("files", (path, text.encode(), "text/markdown"))
                for path, text in parts
            ],
        )

    async def version_of(self, skill_id: str, name: str = "pdf") -> str:
        """One version written through the real route, answering with its version."""
        answer = await self.add_version(
            skill_id, [(f"{name}/SKILL.md", _skill_md(name))]
        )
        assert answer.status_code == 201, answer.text
        return str(answer.json()["version"])


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    store = SkillsHeldInMemory()
    platform = Platform(
        event_log_append=Unused(),
        event_log_range=Unused(),
        definition_registry=Unused(),
        tool_registry=Unused(),
        session_registry=Unused(),
        webhooks=Unused(),
        environment_store=Unused(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
        skill_store=store,
        skill_inventory=InventoryInMemory(store),
    )
    app = create_app(platform)
    async with (
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://control-plane",
            headers={TENANT_HEADER: str(TenantId(uuid.uuid4()))},
        ) as owner,
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://control-plane",
            headers={TENANT_HEADER: str(TenantId(uuid.uuid4()))},
        ) as stranger,
    ):
        yield Harness(store=store, owner=owner, stranger=stranger)


async def test_a_version_delivers_the_files_the_skill_md_told_the_model_to_read(
    harness: Harness,
) -> None:
    """The whole reason versions exist, checked against the real published skill.

    Anthropic's `pdf` skill sends the model to `forms.md` and `reference.md`. Through
    `POST /v1/skills` those references point at files that do not exist and the model is
    instructed to consult them anyway. Here the same document arrives with its siblings
    and comes back out of the archive with them, at paths under the skill's directory.
    """
    skill_id = await harness.upload("pdf")
    written = await harness.add_version(
        skill_id,
        [
            ("pdf/SKILL.md", SKILL_MD),
            *(
                (f"pdf/{named}", f"# {named}\n\nHow to do it.\n")
                for named in REFERENCED_FILES
            ),
        ],
    )
    assert written.status_code == 201, written.text
    version = written.json()["version"]

    got = await harness.owner.get(f"/v1/skills/{skill_id}/versions/{version}/content")

    assert got.status_code == 200
    assert got.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(BytesIO(got.content))
    assert sorted(archive.namelist()) == [
        "pdf/SKILL.md",
        "pdf/forms.md",
        "pdf/reference.md",
    ]
    assert archive.read("pdf/SKILL.md").decode() == SKILL_MD


async def test_the_version_is_a_microsecond_timestamp_nobody_asked_for(
    harness: Harness,
) -> None:
    """Sixteen digits, published as a string, and close to the moment of the write.

    The window is generous on purpose: what is being asserted is the unit, not the
    clock. A millisecond timestamp would be three digits shorter and land in 1970, and
    a sequence would start near 1 -- both fail this without needing a tight bound.
    """
    skill_id = await harness.upload("pdf")
    before = time.time_ns() // 1_000

    version = await harness.version_of(skill_id)

    after = time.time_ns() // 1_000
    assert len(version) == 16
    assert before <= int(version) <= after


async def test_two_versions_minted_in_one_microsecond_take_consecutive_values(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collision the primary key refuses, retried at the next microsecond.

    The clock is held still, which is the only way to force the case: two real requests
    inside one microsecond is not something a test can arrange by being quick. The store
    refuses the duplicate exactly as the key does, so what is exercised is the route's
    retry and not a fake that shrugged.
    """
    fixed = 1_759_178_010_641_129
    monkeypatch.setattr(skill_versions, "mint_version", lambda: fixed)
    # The create door mints too, so with the clock held still it takes `fixed` itself
    # and the two writes below are the second and third collision rather than the first
    # two.
    skill_id = await harness.upload("pdf")

    first = await harness.version_of(skill_id)
    second = await harness.version_of(skill_id)

    assert (first, second) == (str(fixed + 1), str(fixed + 2))
    page = await harness.owner.get(f"/v1/skills/{skill_id}/versions")
    assert [row["version"] for row in page.json()["data"]] == [
        str(fixed + 2),
        str(fixed + 1),
        str(fixed),
    ]


async def test_a_clock_that_never_advances_is_reported_as_the_platforms_fault(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end of the retry walk, which is a fault the caller cannot do anything about.

    The clock is held still and every microsecond the route is willing to try is filled,
    so the next write has nowhere to go. Answered as `platform.internal` rather than as
    a refusal of the request: the bundle was fine, and what has actually happened is
    that the clock is not advancing. The alternative -- looping until one frees up -- is
    a handler that never answers.
    """
    fixed = 1_759_178_010_641_129
    monkeypatch.setattr(skill_versions, "mint_version", lambda: fixed)
    # One of the attempts is spent by the create door itself, so the window is filled by
    # one fewer explicit write than there are attempts.
    skill_id = await harness.upload("pdf")
    for _ in range(skill_versions.MINT_ATTEMPTS - 1):
        await harness.version_of(skill_id)

    refused = await harness.add_version(skill_id, [("pdf/SKILL.md", _skill_md("pdf"))])

    assert refused.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    assert refused.json()["error"]["code"] == ErrorCode.INTERNAL.value
    assert (
        refused.json()["error"]["detail"]["reason"]
        == skill_versions.REASON_VERSION_UNMINTABLE
    )
    # The message names which of the two causes it was, rather than falling back to the
    # sentence an unhandled exception gets. A caller reading "it broke" cannot tell this
    # from a database outage; a caller reading this one can hand it to us as a symptom.
    assert "clock is not advancing" in refused.json()["error"]["message"]
    assert len(harness.store.versions) == skill_versions.MINT_ATTEMPTS


async def test_a_skill_read_back_names_its_greatest_version(harness: Harness) -> None:
    """`latest_version` is derived from the set and moves when the set does.

    The payload is asserted whole rather than field by field, so a field appearing,
    vanishing or changing shape fails here. `source` is the nested object a generated
    client reads as `skill.source.type`, and `origin` says which write door made the row
    -- a different axis from `source`, which says whose catalogue the body belongs to.

    A freshly created skill already names a version, which is the whole of the create
    door's fix: it used to read null, and a client doing create then read then
    use-the-latest had nowhere to go.
    """
    skill_id = await harness.upload("pdf")
    fresh = await harness.owner.get(f"/v1/skills/{skill_id}")
    assert fresh.status_code == 200
    minted = fresh.json()["latest_version"]
    assert fresh.json() == {
        "id": skill_id,
        "name": "pdf",
        "description": "Build a report.",
        "display_name": None,
        "source": {"type": "custom"},
        "origin": "upload",
        "latest_version": minted,
    }
    assert minted is not None

    first = await harness.version_of(skill_id)
    second = await harness.version_of(skill_id)

    read = await harness.owner.get(f"/v1/skills/{skill_id}")
    assert read.json()["latest_version"] == max(minted, first, second)


async def test_retiring_the_latest_version_moves_latest_to_the_survivor(
    harness: Harness,
) -> None:
    """The motivating case: the newest version regressed, so it is the one to retire.

    Refusing to retire the newest would disable the feature exactly where it is needed,
    which is why `latest_version` is computed over the versions that are still live
    rather than over all of them.
    """
    skill_id = await harness.upload("pdf")
    first = await harness.version_of(skill_id)
    second = await harness.version_of(skill_id)

    retired = await harness.owner.delete(f"/v1/skills/{skill_id}/versions/{second}")

    assert retired.status_code == 200
    assert retired.json() == {"id": second, "type": "skill_version_deleted"}
    read = await harness.owner.get(f"/v1/skills/{skill_id}")
    assert read.json()["latest_version"] == first


async def test_retiring_every_version_leaves_a_readable_skill_with_no_latest(
    harness: Harness,
) -> None:
    """Readable and unusable, a state this surface reports rather than hides.

    Every version means the one the create door minted as well as the one added here, so
    the retirements are driven off the listing rather than off a remembered value -- a
    test that retired only what it wrote itself would leave the created version live and
    pass for the wrong reason.
    """
    skill_id = await harness.upload("pdf")
    await harness.version_of(skill_id)
    live = (await harness.owner.get(f"/v1/skills/{skill_id}/versions")).json()["data"]
    assert len(live) == 2, live
    for row in live:
        await harness.owner.delete(f"/v1/skills/{skill_id}/versions/{row['version']}")

    read = await harness.owner.get(f"/v1/skills/{skill_id}")

    assert read.status_code == 200
    assert read.json()["latest_version"] is None
    page = await harness.owner.get(f"/v1/skills/{skill_id}/versions")
    assert page.json() == {"data": [], "next_page": None, "has_more": False}


async def test_a_retired_version_stays_readable_and_says_it_is_retired(
    harness: Harness,
) -> None:
    """The tombstone exists because a definition can have pinned this version.

    So the read answers 200 with `retired` set rather than 410: removing the row would
    leave a Session's history naming something that resolves to nothing, which reads as
    lost data rather than as a retirement.
    """
    skill_id = await harness.upload("pdf")
    version = await harness.version_of(skill_id)
    await harness.owner.delete(f"/v1/skills/{skill_id}/versions/{version}")

    read = await harness.owner.get(f"/v1/skills/{skill_id}/versions/{version}")

    assert read.status_code == 200
    assert read.json()["retired"] is True
    assert read.json()["version"] == version
    got = await harness.owner.get(f"/v1/skills/{skill_id}/versions/{version}/content")
    assert got.status_code == 200


async def test_retiring_a_version_twice_answers_the_same_and_keeps_the_first_moment(
    harness: Harness,
) -> None:
    """A retried request that timed out must not become an error or move the record.

    The response carries no timestamp, so the second half of that claim is checked
    against what the store recorded -- which is the only place it exists.
    """
    skill_id = await harness.upload("pdf")
    version = await harness.version_of(skill_id)

    first = await harness.owner.delete(f"/v1/skills/{skill_id}/versions/{version}")
    recorded = dict(harness.store.retired)
    second = await harness.owner.delete(f"/v1/skills/{skill_id}/versions/{version}")

    assert (first.status_code, second.status_code) == (200, 200)
    assert first.json() == second.json()
    assert harness.store.retired == recorded


async def test_a_deleted_skill_is_gone_for_reads_and_for_its_versions(
    harness: Harness,
) -> None:
    """410 and not 404, on the skill and on every route beneath it.

    The id was real and the platform deliberately stopped serving it. A 404 would send a
    tenant who has just honoured a deletion request looking for a mistake in the id.
    """
    skill_id = await harness.upload("pdf")
    version = await harness.version_of(skill_id)
    assert (await harness.owner.delete(f"/v1/skills/{skill_id}")).status_code == 200

    for path in (
        f"/v1/skills/{skill_id}",
        f"/v1/skills/{skill_id}/versions",
        f"/v1/skills/{skill_id}/versions/{version}",
        f"/v1/skills/{skill_id}/versions/{version}/content",
    ):
        refused = await harness.owner.get(path)
        assert refused.status_code == STATUS_FOR[ErrorCode.SKILL_DELETED], path
        assert refused.json()["error"]["code"] == ErrorCode.SKILL_DELETED.value
    written = await harness.add_version(skill_id, [("pdf/SKILL.md", _skill_md("pdf"))])
    assert written.status_code == STATUS_FOR[ErrorCode.SKILL_DELETED]


async def test_deleting_a_skill_twice_answers_the_same_and_keeps_the_first_moment(
    harness: Harness,
) -> None:
    """Idempotent for the reason retirement is: a retry is not a second deletion."""
    skill_id = await harness.upload("pdf")

    first = await harness.owner.delete(f"/v1/skills/{skill_id}")
    recorded = dict(harness.store.deleted)
    second = await harness.owner.delete(f"/v1/skills/{skill_id}")

    assert (first.status_code, second.status_code) == (200, 200)
    assert first.json() == second.json() == {"id": skill_id, "type": "skill_deleted"}
    assert harness.store.deleted == recorded


async def test_a_skill_with_live_versions_can_be_deleted_without_retiring_them(
    harness: Harness,
) -> None:
    """Requiring it would make a deletion a walk of unknown length for no gain.

    The versions of a deleted skill are already unreachable: every route that reads one
    goes through the same lookup, so nothing new resolves through them either way.
    """
    skill_id = await harness.upload("pdf")
    await harness.version_of(skill_id)

    deleted = await harness.owner.delete(f"/v1/skills/{skill_id}")

    assert deleted.status_code == 200
    assert not harness.store.retired


async def test_another_tenants_skill_reads_as_absent_rather_than_forbidden(
    harness: Harness,
) -> None:
    """Two tenants, because a route with no tenant term looks correct with one."""
    mine = await harness.upload("pdf")
    theirs = await harness.upload("pdf", stranger=True)

    for path in (f"/v1/skills/{theirs}", f"/v1/skills/{theirs}/versions"):
        refused = await harness.owner.get(path)
        assert refused.status_code == STATUS_FOR[ErrorCode.SKILL_NOT_FOUND], path
        assert refused.json()["error"]["code"] == ErrorCode.SKILL_NOT_FOUND.value
    assert (await harness.owner.get(f"/v1/skills/{mine}")).status_code == 200
    written = await harness.add_version(theirs, [("pdf/SKILL.md", _skill_md("pdf"))])
    assert written.status_code == STATUS_FOR[ErrorCode.SKILL_NOT_FOUND]


async def test_a_version_nobody_wrote_is_refused_rather_than_answered_emptily(
    harness: Harness,
) -> None:
    """A well-formed version that names no row, and one that names nothing at all."""
    skill_id = await harness.upload("pdf")
    absent = "1759178010641129"

    for path in (
        f"/v1/skills/{skill_id}/versions/{absent}",
        f"/v1/skills/{skill_id}/versions/{absent}/content",
    ):
        refused = await harness.owner.get(path)
        assert refused.status_code == STATUS_FOR[ErrorCode.SKILL_VERSION_NOT_FOUND]
        assert (
            refused.json()["error"]["code"] == ErrorCode.SKILL_VERSION_NOT_FOUND.value
        )
    unusable = await harness.owner.get(f"/v1/skills/{skill_id}/versions/latest")
    assert unusable.status_code == STATUS_FOR[ErrorCode.SKILL_VERSION_NOT_FOUND]
    assert unusable.json()["error"]["code"] == ErrorCode.SKILL_VERSION_NOT_FOUND.value


async def test_the_version_walk_reads_newest_first_and_ends(harness: Harness) -> None:
    """A bounded walk, because a cursor written inclusively reads pages forever."""
    skill_id = await harness.upload("pdf")
    created = (await harness.owner.get(f"/v1/skills/{skill_id}")).json()[
        "latest_version"
    ]
    written = [created]
    for _ in range(5):
        written.append(await harness.version_of(skill_id))

    seen: list[str] = []
    cursor: str | None = None
    for _ in range(_PAGE_BUDGET):
        query: dict[str, str | int] = {"limit": 2}
        if cursor is not None:
            query["page"] = cursor
        page = await harness.owner.get(f"/v1/skills/{skill_id}/versions", params=query)
        assert page.status_code == 200, page.text
        body = page.json()
        seen.extend(row["version"] for row in body["data"])
        cursor = body["next_page"]
        assert body["has_more"] is (cursor is not None)
        if cursor is None:
            break
    else:
        raise AssertionError(_NEVER_ENDED)

    assert seen == sorted(written, reverse=True)


async def test_a_cursor_this_surface_did_not_issue_is_refused(
    harness: Harness,
) -> None:
    """Refused rather than read as the start, which would look like a looping walk."""
    skill_id = await harness.upload("pdf")
    await harness.version_of(skill_id)

    refused = await harness.owner.get(
        f"/v1/skills/{skill_id}/versions", params={"page": "not-a-cursor"}
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]
    assert refused.json()["error"]["detail"]["reason"] == REASON_CURSOR_INVALID
    issued = await harness.owner.get(
        f"/v1/skills/{skill_id}/versions",
        params={"page": version_cursor(1_759_178_010_641_129)},
    )
    assert issued.status_code == 200


async def test_a_page_larger_than_the_published_bound_is_refused_by_the_field(
    harness: Harness,
) -> None:
    """The bound is published, so a caller learns it from a 400 naming the field."""
    skill_id = await harness.upload("pdf")

    refused = await harness.owner.get(
        f"/v1/skills/{skill_id}/versions", params={"limit": MAX_VERSION_PAGE_SIZE + 1}
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert "limit" in refused.json()["error"]["detail"]["fields"]


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../escaped.md",
        "pdf/../../escaped.md",
        "pdf//forms.md",
        "pdf/./forms.md",
        "pdf\\forms.md",
    ],
)
async def test_a_path_that_could_leave_the_skill_directory_is_refused(
    harness: Harness, path: str
) -> None:
    """Each of these is joined onto a directory inside a Session's workspace.

    Refused here as well as by the store's own check, because the rule protects a
    filesystem and the filesystem cannot tell which writer produced the row. The two
    spelling cases -- `//` and `/./` -- are refused because they are second spellings of
    a path already in the bundle, which the primary key would treat as a different file.
    """
    skill_id = await harness.upload("pdf")
    # A count and not an emptiness check: the create door has already written version
    # one, so what this asserts is that the *refused* bundle added nothing.
    before = len(harness.store.versions)

    refused = await harness.add_version(
        skill_id, [("pdf/SKILL.md", _skill_md("pdf")), (path, "# notes\n")]
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID], path
    assert refused.json()["error"]["detail"]["reason"] == REASON_BUNDLE_INVALID
    assert len(harness.store.versions) == before


async def test_a_bundle_with_no_skill_md_says_what_it_was_sent_instead(
    harness: Harness,
) -> None:
    """Nothing in it says what the skill is called, so nothing can be written."""
    skill_id = await harness.upload("pdf")

    refused = await harness.add_version(skill_id, [("pdf/forms.md", "# forms\n")])

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert "forms.md" in refused.json()["error"]["message"]


async def test_a_bundle_with_no_single_root_is_refused_rather_than_guessed_at(
    harness: Harness,
) -> None:
    """Two directories, or a directory beside a loose file: both read two ways.

    Either reading -- flatten everything, or invent a root -- changes where the model
    looks for the file, so neither is chosen for the caller. `directory` is "the
    top-level directory name that was extracted from the uploaded files", and that
    phrase has one answer only when the bundle has one root.
    """
    skill_id = await harness.upload("pdf")
    document = ("pdf/SKILL.md", _skill_md("pdf"))

    two_roots = await harness.add_version(
        skill_id, [document, ("docx/forms.md", "# forms\n")]
    )
    mixed = await harness.add_version(skill_id, [document, ("forms.md", "# forms\n")])

    assert two_roots.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert two_roots.json()["error"]["detail"]["directory_count"] == 2
    assert mixed.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert mixed.json()["error"]["detail"]["top_level_file"] == "forms.md"


async def test_a_flat_bundle_is_filed_under_the_skills_own_name(
    harness: Harness,
) -> None:
    """No leading segment to extract, so the directory is the name the model would see.

    The same rule the migration backfilled the pre-version rows with, which is what
    keeps a flat upload and a backfilled one describing the same shape.
    """
    skill_id = await harness.upload("pdf")

    written = await harness.add_version(
        skill_id, [("SKILL.md", _skill_md("pdf")), ("forms.md", "# forms\n")]
    )

    assert written.status_code == 201, written.text
    assert written.json()["directory"] == "pdf"
    got = await harness.owner.get(
        f"/v1/skills/{skill_id}/versions/{written.json()['version']}/content"
    )
    assert sorted(zipfile.ZipFile(BytesIO(got.content)).namelist()) == [
        "pdf/SKILL.md",
        "pdf/forms.md",
    ]


async def test_a_file_that_is_not_text_is_refused_rather_than_stored(
    harness: Harness,
) -> None:
    """A skill's files are read by the model out of a Secret, so they are text."""
    skill_id = await harness.upload("pdf")

    refused = await harness.owner.post(
        f"/v1/skills/{skill_id}/versions",
        files=[
            ("files", ("pdf/SKILL.md", _skill_md("pdf").encode(), "text/markdown")),
            ("files", ("pdf/logo.png", b"\x89PNG\r\n\x1a\n\xff\xfe", "image/png")),
        ],
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    # The path as the caller sent it, not as it is stored. The read happens
    # before the bundle's root is worked out, and the refusal has to name the
    # part the caller can go and look at.
    assert refused.json()["error"]["detail"]["path"] == "pdf/logo.png"


async def test_the_documented_multipart_example_is_refused_and_says_why(
    harness: Harness,
) -> None:
    """The reference's own example is a form field with no filename, so it cannot land.

    Kept as a test rather than a comment because it is the request a caller writing
    against that documentation will actually send: the refusal is the platform's answer
    to "where does this file go", and answering it silently would be the worse outcome.
    """
    skill_id = await harness.upload("pdf")

    refused = await harness.owner.post(
        f"/v1/skills/{skill_id}/versions", data={"files": '["Example data"]'}
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert "files" in refused.json()["error"]["detail"]["fields"]


async def test_a_bundle_larger_than_a_session_can_carry_is_refused(
    harness: Harness,
) -> None:
    """Two bounds, one ceiling: the file count and the size of any one file."""
    skill_id = await harness.upload("pdf")
    parts = [("pdf/SKILL.md", _skill_md("pdf"))]

    too_many = await harness.add_version(
        skill_id,
        parts + [(f"pdf/note{n}.md", "# note\n") for n in range(MAX_BUNDLE_FILES)],
    )
    too_big = await harness.add_version(
        skill_id, parts + [("pdf/forms.md", "x" * (SKILL_MD_MAX_BYTES + 1))]
    )

    assert too_many.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert too_many.json()["error"]["detail"]["file_count"] == MAX_BUNDLE_FILES + 1
    assert too_big.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert too_big.json()["error"]["detail"]["byte_length"] == SKILL_MD_MAX_BYTES + 1


async def test_an_empty_file_is_refused_because_the_model_is_told_to_read_it(
    harness: Harness,
) -> None:
    skill_id = await harness.upload("pdf")

    refused = await harness.add_version(
        skill_id, [("pdf/SKILL.md", _skill_md("pdf")), ("pdf/forms.md", "   \n")]
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert refused.json()["error"]["detail"]["path"] == "pdf/forms.md"


async def test_the_same_path_twice_in_one_bundle_is_refused(
    harness: Harness,
) -> None:
    """Which of the two the model would read is not a question worth answering."""
    skill_id = await harness.upload("pdf")

    refused = await harness.add_version(
        skill_id,
        [
            ("pdf/SKILL.md", _skill_md("pdf")),
            ("pdf/forms.md", "# first\n"),
            ("pdf/forms.md", "# second\n"),
        ],
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert refused.json()["error"]["detail"]["path"] == "pdf/forms.md"


async def test_every_archive_entry_is_stamped_from_the_version_not_the_download(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two downloads of one version are the same bytes, and this is why.

    The clock is held at the reference's own sample version, which is months away from
    now, so an entry stamped at the moment of the *download* fails this outright. Two
    identical downloads alone would not: a zip entry's time has two-second granularity,
    so a pair of requests in one test would agree even with the wrong stamp.

    It matters because a stable archive is what lets a caller compare a digest of one
    download against a digest of another instead of unpacking both to find out whether
    anything changed.
    """
    minted = 1_759_178_010_641_129
    monkeypatch.setattr(skill_versions, "mint_version", lambda: minted)
    skill_id = await harness.upload("pdf")
    written = await harness.add_version(
        skill_id, [("pdf/SKILL.md", _skill_md("pdf")), ("pdf/forms.md", "# forms\n")]
    )
    where = f"/v1/skills/{skill_id}/versions/{written.json()['version']}/content"

    first = await harness.owner.get(where)
    second = await harness.owner.get(where)

    stamped = datetime.fromtimestamp(minted // 1_000_000, tz=UTC)
    archive = zipfile.ZipFile(BytesIO(first.content))
    assert [archive.getinfo(name).date_time for name in archive.namelist()] == [
        (
            stamped.year,
            stamped.month,
            stamped.day,
            stamped.hour,
            stamped.minute,
            # A zip entry holds seconds in two-second units. The sample version's
            # second is even, so nothing is being rounded away here.
            stamped.second,
        )
    ] * 2
    assert first.content == second.content
    assert 'filename="pdf-' in first.headers["content-disposition"]


async def test_the_parse_answers_for_the_two_shapes_http_cannot_deliver() -> None:
    """A nameless part and an empty bundle, called directly: HTTP cannot send them.

    A multipart part with no filename arrives as a form field and FastAPI refuses it a
    step earlier; a missing `files` field is refused there too. Both branches stay
    because Starlette types a filename as optional and the sequence as a sequence -- the
    null and the empty case are reachable through the type even where the wire cannot
    produce them, and without these the fallback would be a path of `''` reaching the
    store's own check as a 500.
    """
    with pytest.raises(Exception) as nameless:
        await parse_bundle([UploadFile(file=BytesIO(b"# notes\n"), filename=None)])
    with pytest.raises(Exception) as empty:
        await parse_bundle([])

    assert "carries no filename" in str(nameless.value)
    assert "at least one file" in str(empty.value)


# --- The create door: two shapes in, one skill and one first version out ----------


async def _create_multipart(
    harness: Harness,
    parts: Sequence[tuple[str, str]],
    *,
    fields: Sequence[tuple[str, str]] = (),
) -> Any:
    """POST /v1/skills as multipart, the way an SDK's create call sends it."""
    return await harness.owner.post(
        "/v1/skills",
        files=[
            ("files", (path, text.encode(), "text/markdown")) for path, text in parts
        ],
        data=dict(fields),
    )


async def test_the_create_door_accepts_the_multipart_form_an_sdk_sends(
    harness: Harness,
) -> None:
    """The gap that made a generated client's create call fail outright.

    An SDK built against the reference sends `-F files=@pdf/SKILL.md`. Before this the
    route took JSON with one `skill_md` field and nothing else, so that request was
    refused as malformed and no skill could be created by a client that had not been
    written against this platform specifically.
    """
    created = await _create_multipart(harness, [("pdf/SKILL.md", _skill_md("pdf"))])

    assert created.status_code == 201, created.text
    assert created.json()["name"] == "pdf"
    assert created.json()["description"] == "Build a report."


async def test_a_multipart_create_keeps_the_files_beside_the_document(
    harness: Harness,
) -> None:
    """The reason the create door takes a bundle rather than one document.

    Anthropic's published `pdf` skill tells the model to read `forms.md`. A create that
    dropped everything but the `SKILL.md` would register a skill whose own instructions
    point at a file that is not there -- and the model is still told to consult it. So
    the archive is checked rather than the 201: what matters is that the sibling comes
    back out, not that the request was accepted.
    """
    created = await _create_multipart(
        harness,
        [("pdf/SKILL.md", _skill_md("pdf")), ("pdf/forms.md", "Fill the form.\n")],
    )
    assert created.status_code == 201, created.text
    skill_id = created.json()["id"]

    page = await harness.owner.get(f"/v1/skills/{skill_id}/versions")
    version = page.json()["data"][0]["version"]
    archive = await harness.owner.get(
        f"/v1/skills/{skill_id}/versions/{version}/content"
    )

    assert archive.status_code == 200, archive.text
    with zipfile.ZipFile(BytesIO(archive.content)) as held:
        assert sorted(held.namelist()) == ["pdf/SKILL.md", "pdf/forms.md"]


async def test_the_json_door_still_creates_a_skill(harness: Harness) -> None:
    """Kept working, because removing a door that works is not what parity asks for.

    The repository submission route and every test in this package send JSON. A create
    surface that accepted only multipart would be a second break dressed as a fix.
    """
    created = await harness.owner.post(
        "/v1/skills", json={"skill_md": _skill_md("pdf")}
    )

    assert created.status_code == 201, created.text
    assert created.json()["name"] == "pdf"


async def test_both_create_doors_refuse_a_document_that_is_not_a_skill_alike(
    harness: Harness,
) -> None:
    """One refusal set, reached by two parses.

    What makes a document unregisterable is a property of the document -- no
    frontmatter, no description, a name that is not a legal directory -- so it cannot
    depend on which content type carried it. Two doors that disagreed here would mean a
    caller could get a skill stored by re-sending it the other way.
    """
    not_a_skill = "no frontmatter here at all\n"

    as_json = await harness.owner.post("/v1/skills", json={"skill_md": not_a_skill})
    as_form = await _create_multipart(harness, [("pdf/SKILL.md", not_a_skill)])

    assert as_json.status_code == as_form.status_code == 400
    assert (
        as_json.json()["error"]["code"]
        == as_form.json()["error"]["code"]
        == ErrorCode.REQUEST_INVALID.value
    )
    # The branchable half too, and this is the assertion with teeth: both doors reach
    # the *same* conversion, so a caller cannot tell which one decided the document was
    # unregisterable. Two doors that merely both answered 400 would pass the lines above
    # while disagreeing about everything a client actually switches on.
    assert (
        as_json.json()["error"]["detail"]["reason"]
        == as_form.json()["error"]["detail"]["reason"]
        == REASON_BUNDLE_INVALID
    )


async def test_a_created_skill_names_a_first_version_rather_than_null(
    harness: Harness,
) -> None:
    """A create that minted nothing left `latest_version` null on a brand-new skill.

    The reference says a Skill always holds at least one version, so a client doing
    create then read then use-the-latest got null and had nowhere to go. Both doors mint
    it, so which content type was used cannot change whether the skill is usable.
    """
    from_json = await harness.owner.post(
        "/v1/skills", json={"skill_md": _skill_md("pdf")}
    )
    from_form = await _create_multipart(harness, [("doc/SKILL.md", _skill_md("doc"))])

    for created in (from_json, from_form):
        skill_id = created.json()["id"]
        read = await harness.owner.get(f"/v1/skills/{skill_id}")
        assert read.json()["latest_version"] is not None, read.text
        page = await harness.owner.get(f"/v1/skills/{skill_id}/versions")
        assert [row["version"] for row in page.json()["data"]] == [
            read.json()["latest_version"]
        ]


async def test_a_json_created_first_version_holds_the_document_and_nothing_else(
    harness: Harness,
) -> None:
    """The JSON door carries one document, so its first version carries one file.

    Not an empty version and not an invented sibling: the archive holds exactly the
    `SKILL.md`, filed under the skill's own name because a flat submission names no
    directory of its own.
    """
    created = await harness.owner.post(
        "/v1/skills", json={"skill_md": _skill_md("pdf")}
    )
    skill_id = created.json()["id"]
    version = (await harness.owner.get(f"/v1/skills/{skill_id}")).json()[
        "latest_version"
    ]

    archive = await harness.owner.get(
        f"/v1/skills/{skill_id}/versions/{version}/content"
    )

    with zipfile.ZipFile(BytesIO(archive.content)) as held:
        assert held.namelist() == ["pdf/SKILL.md"]


async def test_a_display_name_is_stored_and_read_back(harness: Harness) -> None:
    """The reference's optional label, accepted on the create door and published again.

    Asserted through the read rather than off the 201, because the 201 could echo a
    value that was never written. What makes the field real is that a later request,
    which holds only the id, gets it back -- and that is exactly the property an
    accept-and-drop implementation fails while every other test still passes.
    """
    created = await _create_multipart(
        harness,
        [("pdf/SKILL.md", _skill_md("pdf"))],
        fields=[("display_name", "PDF Report Builder")],
    )
    assert created.status_code == 201, created.text

    read = await harness.owner.get(f"/v1/skills/{created.json()['id']}")

    assert read.json()["display_name"] == "PDF Report Builder"


async def test_a_skill_created_without_a_label_reads_back_null_rather_than_its_name(
    harness: Harness,
) -> None:
    """Null is the honest value for a skill nobody labelled, and it is not the name.

    Defaulting the label to `name` would make "this skill has no label" unrepresentable:
    every row would carry one and no reader could tell an author's choice from a
    fallback. A label somebody chose and a directory name parsed out of frontmatter are
    different facts.
    """
    created = await harness.owner.post(
        "/v1/skills", json={"skill_md": _skill_md("pdf")}
    )

    read = await harness.owner.get(f"/v1/skills/{created.json()['id']}")

    assert read.json()["display_name"] is None
    assert read.json()["name"] == "pdf"


async def test_the_json_door_takes_the_label_too(harness: Harness) -> None:
    """Both create doors carry it, so which content type was used cannot change what is
    stored.

    A field one door accepted and the other refused would make the two doors two
    surfaces, and a caller would have to know which one to use in order to set a label.
    """
    created = await harness.owner.post(
        "/v1/skills",
        json={"skill_md": _skill_md("pdf"), "display_name": "PDF Report Builder"},
    )
    assert created.status_code == 201, created.text

    read = await harness.owner.get(f"/v1/skills/{created.json()['id']}")

    assert read.json()["display_name"] == "PDF Report Builder"


async def test_a_label_that_is_not_a_string_is_refused_naming_the_field(
    harness: Harness,
) -> None:
    """The label is text or absent, and a number is neither.

    Refused rather than coerced: `str(3)` is a label the tenant did not write, and it
    would be read back by whoever wonders why their skill is called "3".
    """
    refused = await harness.owner.post(
        "/v1/skills", json={"skill_md": _skill_md("pdf"), "display_name": 3}
    )

    assert refused.status_code == 400, refused.text
    assert "display_name" in refused.json()["error"]["message"]


async def test_two_paths_that_collide_once_delivered_are_refused_naming_both(
    harness: Harness,
) -> None:
    """A Kubernetes Secret key holds no `/`, so delivery folds one into `_`.

    That fold is not injective: `a/b.md` and `a_b.md` are two distinct paths in the
    bundle and one key in the Secret, so the second silently replaces the first and the
    model reads whichever won. Refused at the parse, naming both sides, because at
    delivery time the only available answer is which key was overwritten.
    """
    skill_id = await harness.upload("pdf")

    refused = await harness.add_version(
        skill_id,
        [
            ("pdf/SKILL.md", _skill_md("pdf")),
            ("pdf/a/b.md", "one\n"),
            ("pdf/a_b.md", "two\n"),
        ],
    )

    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["detail"]["reason"] == REASON_BUNDLE_INVALID
    message = refused.json()["error"]["message"]
    assert "a/b.md" in message and "a_b.md" in message
