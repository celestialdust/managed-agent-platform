"""GET /v1/skills: what a tenant holds, whose it is, and how the walk ends.

Tier 1 (local, no infrastructure). The route runs inside the real app over an in-memory
store, because every claim here is a property of *this module* -- which rows a page
carries, which fields a row carries, what a bad cursor gets, whose skills come back --
and each one is reachable without a database.

The store's own guarantees are not claimed here. That the tenant is a term in the SQL
rather than a filter over fetched rows, and that the keyset boundary neither repeats a
row nor drops one against a real index, are properties of PostgreSQL and are graded
against it in `tests/adapters/test_skill_registry.py`.

The in-memory store below therefore does two things a convenient fake would not. It
sorts and pages by the same `(name, id)` keyset the adapter uses, so a route that read
the boundary backwards fails here rather than only in the container run; and it refuses
the same `limit` window the adapter refuses, so this file cannot certify a request the
real store would reject.

**Two tenants appear in every isolation test on purpose.** A listing that dropped its
tenant term reads perfectly normally with one tenant in the database -- it returns
exactly the rows the single tenant uploaded -- so a single-tenant test proves nothing
at all about isolation.
"""

from __future__ import annotations

import uuid
from base64 import urlsafe_b64encode
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from managed_agent.adapters.postgres.skill_registry import _MAX_PAGE
from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.skills import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    REASON_CURSOR_INVALID,
    SkillCursor,
    UploadedSkillListed,
)
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.skills.inventory import (
    RepositorySkillHeld,
    RepositorySkillRow,
    SkillRow,
    UploadedSkillRow,
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
from managed_agent.core.registration.skill import (
    ValidatedSkill,
    parse_skill_md,
    repository_skill_id,
)

_SHA = "0" * 39 + "a"
_DESCRIPTION = "Build a PDF report."
_REPOSITORY = "git@github.com:acme/skills.git"

# Every walk in this file is bounded. A keyset boundary written inclusively, or a route
# that never stops issuing a cursor, makes a walk that reads pages forever -- and an
# unbounded loop turns that into a test that hangs rather than one that fails.
_PAGE_BUDGET = 20
_NEVER_ENDED = (
    f"the walk read {_PAGE_BUDGET} pages without a null next_page; it is either "
    "repeating rows at the page boundary or never reporting the end"
)

_ADAPTER_MAX_PAGE = _MAX_PAGE
"""The window the adapter serves, imported rather than copied.

It was a literal `500`, and that is exactly what broke when the adapter's bound moved to
1024: a fake that mirrors a number by hand certifies requests the real store refuses, or
refuses ones it serves, and either way the container run is the first thing to notice.
Bound to the adapter's own constant so this fake cannot drift from it at all.
"""


def _skill_md(name: str, description: str = "Build a PDF report.") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\nDo the thing.\n"


def _skill(name: str, description: str = "Build a PDF report.") -> ValidatedSkill:
    return parse_skill_md(_skill_md(name, description), source=f"{name}/SKILL.md")


class SkillsHeldInMemory:
    """The skill store in two dicts, paged by the keyset the real adapter uses.

    `page_calls` is counted because it is the only way to assert that a refused request
    reached no store at all: a route that refused *after* querying would be identical
    from the response alone.
    """

    def __init__(self) -> None:
        self.skills: dict[tuple[TenantId, SkillId], ValidatedSkill] = {}
        self.labels: dict[tuple[TenantId, SkillId], str | None] = {}
        self.repositories: dict[tuple[TenantId, str, str], list[ValidatedSkill]] = {}
        self.versions: dict[tuple[TenantId, SkillId, int], str] = {}
        self.page_calls = 0

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

    async def read_skills(
        self, tenant_id: TenantId, skill_ids: Sequence[SkillId]
    ) -> tuple[SkillRecord, ...]:
        raise AssertionError("a listing test resolved a definition's skills")

    async def page_uploaded_skills(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillListing, ...]:
        if limit < 1 or limit > _ADAPTER_MAX_PAGE:
            raise ValueError(f"page limit {limit} is outside 1..{_ADAPTER_MAX_PAGE}")
        self.page_calls += 1
        mine = sorted(
            (
                SkillListing(
                    skill_id=skill_id,
                    name=skill.name,
                    description=skill.description,
                    display_name=self.labels.get((owner, skill_id)),
                )
                for (owner, skill_id), skill in self.skills.items()
                if owner == tenant_id
            ),
            key=lambda row: (row.name, row.skill_id),
        )
        if after is not None:
            mine = [row for row in mine if (row.name, row.skill_id) > after]
        return tuple(mine[:limit])

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
            latest_version=max(
                (
                    version
                    for (owner, held, version) in self.versions
                    if (owner, held) == (tenant_id, skill_id)
                ),
                default=None,
            ),
            deleted=False,
        )

    async def delete_skill(self, tenant_id: TenantId, skill_id: SkillId) -> None:
        raise AssertionError("a listing test deleted a skill")

    async def add_skill_version(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        version: int,
        skill: ValidatedSkill,
        directory: str,
        files: Sequence[SkillVersionFile],
    ) -> None:
        """Held rather than refused, because creating a skill now writes version one.

        Refusing here used to be the guard that a listing test never wrote a version.
        That guard has to go: every create through the real route writes one, so keeping
        it would make this fake refuse the very requests the file is built on. The key
        collision the primary key enforces is kept, so a route that stopped minting
        against a collision still fails here.
        """
        key = (tenant_id, skill_id, version)
        if key in self.versions:
            raise SkillVersionCollision(f"version {version} is taken")
        self.versions[key] = skill.name

    async def page_skill_versions(
        self, tenant_id: TenantId, skill_id: SkillId, after: int | None, limit: int
    ) -> tuple[SkillVersionRecord, ...]:
        raise AssertionError("a listing test paged a skill's versions")

    async def read_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionRecord | None:
        raise AssertionError("a listing test read one version")

    async def read_skill_version_bundle(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionBundle | None:
        raise AssertionError("a listing test downloaded a version")

    async def retire_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> None:
        raise AssertionError("a listing test retired a version")


class InventoryInMemory:
    """Both doors' skills in one keyset-ordered collection, as the real inventory is.

    Not a convenient fake in the one way that matters: it sorts and pages by the same
    `(name, id)` keyset over the *union* that the adapter's `UNION ALL` does, so a route
    that merged two origins by concatenating them, or read the boundary backwards, fails
    here rather than only in the container run. That is the property the whole two-door
    listing rests on -- a walk that drops one origin's tail looks exactly like a walk
    that finished.

    `page_calls` is counted because it is the only way to assert that a refused request
    reached no store at all: a route that refused after querying is identical from the
    response alone.
    """

    def __init__(self, skills: SkillsHeldInMemory) -> None:
        self._skills = skills
        self.repository_rows: dict[SkillId, RepositorySkillRow] = {}
        self.bodies: dict[SkillId, str] = {}
        self.owners: dict[SkillId, TenantId] = {}
        self.page_calls = 0
        self.assign_calls = 0

    async def assign_repository_ids(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        names: Sequence[str],
    ) -> tuple[tuple[str, SkillId], ...]:
        """Idempotent per the four columns, exactly as its uuid5 key is."""
        self.assign_calls += 1
        assigned: list[tuple[str, SkillId]] = []
        for name in names:
            skill_id = repository_skill_id(tenant_id, repository, revision, name)
            self.owners[skill_id] = tenant_id
            self.repository_rows[skill_id] = RepositorySkillRow(
                skill_id=skill_id,
                name=name,
                description=_DESCRIPTION,
                repository=repository,
                revision=revision,
            )
            self.bodies[skill_id] = _skill_md(name)
            assigned.append((name, skill_id))
        return tuple(assigned)

    async def repository_skill_at(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> RepositorySkillHeld | None:
        row = self.repository_rows.get(skill_id)
        if row is None or self.owners.get(skill_id) != tenant_id:
            return None
        return RepositorySkillHeld(
            skill_id=row.skill_id,
            name=row.name,
            description=row.description,
            body=self.bodies[skill_id],
            repository=row.repository,
            revision=row.revision,
        )

    async def page(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillRow, ...]:
        if limit < 1 or limit > _ADAPTER_MAX_PAGE:
            raise ValueError(f"page limit {limit} is outside 1..{_ADAPTER_MAX_PAGE}")
        self.page_calls += 1
        rows: list[SkillRow] = [
            UploadedSkillRow(
                skill_id=skill_id,
                name=skill.name,
                description=skill.description,
                display_name=self._skills.labels.get((owner, skill_id)),
            )
            for (owner, skill_id), skill in self._skills.skills.items()
            if owner == tenant_id
        ]
        rows.extend(
            row
            for skill_id, row in self.repository_rows.items()
            if self.owners.get(skill_id) == tenant_id
        )
        rows.sort(key=lambda row: (row.name, row.skill_id))
        if after is not None:
            rows = [row for row in rows if (row.name, row.skill_id) > after]
        return tuple(rows[:limit])


class Unused:
    """One raising stand-in for every port a listing never touches."""

    def __getattr__(self, name: str) -> Any:
        async def refuse(*args: object, **kwargs: object) -> Any:
            raise AssertionError(f"a listing test called {name}")

        return refuse


@dataclass(frozen=True, slots=True)
class Harness:
    """One app, one store, and two tenants that both talk to it.

    Two clients rather than one plus a header override, so no test can accidentally
    read as the wrong tenant by forgetting to pass one.
    """

    store: SkillsHeldInMemory
    inventory: InventoryInMemory
    owner: AsyncClient
    stranger: AsyncClient

    async def upload(self, name: str, *, as_stranger: bool = False) -> Any:
        client = self.stranger if as_stranger else self.owner
        return await client.post("/v1/skills", json={"skill_md": _skill_md(name)})

    async def list_skills(
        self, *, as_stranger: bool = False, **query: str | int
    ) -> Any:
        client = self.stranger if as_stranger else self.owner
        return await client.get("/v1/skills", params=query)


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    store = SkillsHeldInMemory()
    inventory = InventoryInMemory(store)
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
        skill_inventory=inventory,
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
        yield Harness(store=store, inventory=inventory, owner=owner, stranger=stranger)


async def _walk(
    harness: Harness, *, as_stranger: bool = False, **query: str | int
) -> Any:
    """Every row of the listing, read page by page until the cursor is null.

    Returns the rows in the order they arrived, duplicates included, because a walk that
    repeated a row has to be able to fail on the count rather than on a set comparison
    that would silently absorb it.
    """
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(_PAGE_BUDGET):
        page = await harness.list_skills(
            as_stranger=as_stranger,
            **({} if cursor is None else {"page": cursor}),
            **query,
        )
        assert page.status_code == 200, page.text
        body = page.json()
        rows.extend(body["data"])
        cursor = body["next_page"]
        if cursor is None:
            return rows
    raise AssertionError(_NEVER_ENDED)


async def test_an_upload_can_be_found_again_after_its_response_is_lost(
    harness: Harness,
) -> None:
    """The whole reason this route exists.

    An uploaded skill is immutable and is never deleted, so before the listing the id in
    the 201 was the only handle a tenant could ever have on it: lose the response and
    the body sat in the platform forever, attachable by nobody.
    """
    uploaded = await harness.upload("pdf-report")
    assert uploaded.status_code == 201, uploaded.text
    minted = uploaded.json()["id"]

    listed = await harness.list_skills()

    assert listed.status_code == 200, listed.text
    assert [row["id"] for row in listed.json()["data"]] == [minted]


async def test_a_tenant_never_sees_another_tenants_skill(harness: Harness) -> None:
    """The tenant term, graded with two tenants because one proves nothing.

    A listing that forgot to scope by tenant returns exactly the right rows for a
    database holding one tenant, and every assertion about ordering, paging and payload
    shape keeps passing. It is only a second tenant that can tell the two apart.
    """
    mine = await harness.upload("pdf-report")
    theirs = await harness.upload("their-skill", as_stranger=True)
    assert (mine.status_code, theirs.status_code) == (201, 201)

    ours = await harness.list_skills()
    others = await harness.list_skills(as_stranger=True)

    assert [row["name"] for row in ours.json()["data"]] == ["pdf-report"]
    assert [row["name"] for row in others.json()["data"]] == ["their-skill"]
    assert ours.json()["data"][0]["id"] != others.json()["data"][0]["id"]


async def test_a_tenant_holding_nothing_reads_an_empty_page_rather_than_a_refusal(
    harness: Harness,
) -> None:
    """Empty is the true answer for a tenant who has uploaded no skill.

    A 404 would say the collection does not exist, which is a different claim and one
    this surface cannot make: every tenant has a skill collection and most of them are
    empty.
    """
    listed = await harness.list_skills()

    assert listed.status_code == 200, listed.text
    assert listed.json() == {"data": [], "next_page": None, "has_more": False}


async def test_a_listing_row_carries_no_skill_body(harness: Harness) -> None:
    """A page carries what identifies a skill, never the document itself.

    A `SKILL.md` runs to 32 KiB and a tenant's collection has no ceiling, so a page of
    bodies is a page nobody can walk. Asserted on the row's exact key set rather than on
    the absence of one name, so a body arriving under `text`, `body` or `skill_md` all
    fail the same way.
    """
    await harness.upload("pdf-report")

    listed = await harness.list_skills()

    row = listed.json()["data"][0]
    assert set(row) == set(UploadedSkillListed.model_fields)
    assert set(row) == {
        "id",
        "name",
        "description",
        "display_name",
        "source",
        "origin",
    }


async def test_the_walk_reads_every_skill_exactly_once_across_pages(
    harness: Harness,
) -> None:
    """Ordered by name, and a page boundary neither repeats a row nor drops one.

    Walked one row at a time so every boundary in the collection is exercised, which is
    where an inclusive comparison shows itself: with a page big enough to hold
    everything there is no boundary to get wrong.
    """
    names = ["mike", "alpha", "zulu", "bravo", "yankee"]
    for name in names:
        assert (await harness.upload(name)).status_code == 201

    rows = await _walk(harness, limit=1)

    assert [row["name"] for row in rows] == sorted(names)


async def test_the_boundary_between_two_skills_of_one_name_holds(
    harness: Harness,
) -> None:
    """Two uploads may share a name on purpose, so the name alone is not a position.

    A cursor naming only the name cannot say which of the two the caller already has, so
    the page boundary between them would repeat one row or drop the other. Both ids come
    back exactly once, which is what the id in the keyset buys.
    """
    first = (await harness.upload("pdf-report")).json()["id"]
    second = (await harness.upload("pdf-report")).json()["id"]
    assert first != second

    rows = await _walk(harness, limit=1)

    assert sorted(row["id"] for row in rows) == sorted([first, second])


async def test_the_last_page_reports_the_end_instead_of_a_cursor_leading_nowhere(
    harness: Harness,
) -> None:
    """The store is asked for one row more than is returned, and that row is the answer.

    A `next_page` on a page that happens to be exactly full would send a caller for
    one more round trip to learn what this page could already have told it.

    Both fields, at both boundaries. They are two spellings of one fact, computed from
    the one extra row the store was asked for -- so a caller that branches on `has_more`
    and one that branches on `next_page` must never disagree, and only asserting the
    pair at the end and at the middle can catch it if they do.
    """
    for name in ("alpha", "bravo"):
        await harness.upload(name)

    exactly_full = await harness.list_skills(limit=2)

    assert len(exactly_full.json()["data"]) == 2
    assert exactly_full.json()["next_page"] is None
    assert exactly_full.json()["has_more"] is False


async def test_a_page_holds_no_more_than_the_limit_asked_for(harness: Harness) -> None:
    """The page is the caller's size, and the extra probe row is not handed out."""
    for name in ("alpha", "bravo", "charlie"):
        await harness.upload(name)

    page = await harness.list_skills(limit=2)

    assert [row["name"] for row in page.json()["data"]] == ["alpha", "bravo"]
    assert page.json()["next_page"] is not None
    assert page.json()["has_more"] is True


async def test_the_default_page_size_is_the_published_one(harness: Harness) -> None:
    """A caller that names no limit gets the documented page rather than everything.

    Asserted through the store's own window rather than by uploading twenty-six skills:
    what matters is that the route asks for the default plus the probe row, and a route
    that asked for everything would ask for a limit the store never saw.
    """
    for index in range(3):
        await harness.upload(f"skill-{index}")

    page = await harness.list_skills()

    assert page.status_code == 200, page.text
    assert len(page.json()["data"]) == 3
    assert DEFAULT_PAGE_SIZE < MAX_PAGE_SIZE, (
        "the default page has to leave room under the maximum, or naming a larger "
        "limit could never do anything"
    )


_CURSORS_THIS_SURFACE_NEVER_ISSUED = {
    "not base64 at all": "not-a-cursor!!",
    "base64 of something with no separator": urlsafe_b64encode(b"nonsense")
    .decode()
    .rstrip("="),
    "base64 of a non-uuid position": urlsafe_b64encode(b"not-a-uuid.pdf-report")
    .decode()
    .rstrip("="),
    "base64 of a uuid with no name": urlsafe_b64encode(
        f"{uuid.uuid4()}.".encode()
    ).decode(),
    "base64 of bytes that are not utf-8": urlsafe_b64encode(b"\xff\xfe.x")
    .decode()
    .rstrip("="),
    "an empty string": "",
}
"""Every shape a cursor can arrive in that this surface did not issue.

Each entry is a separate decision in `SkillCursor.decode`: the base64 that will not
decode, the text with no separator, the half that is not a uuid, the half that is a
uuid with nothing after it, the bytes that are not text. One assertion over the set
would pass on whichever member happened to be checked first.
"""


@pytest.mark.parametrize(
    "cursor",
    list(_CURSORS_THIS_SURFACE_NEVER_ISSUED.values()),
    ids=list(_CURSORS_THIS_SURFACE_NEVER_ISSUED),
)
async def test_a_cursor_this_surface_did_not_issue_is_refused(
    harness: Harness, cursor: str
) -> None:
    """Refused in the published closed set, and refused before the store is asked.

    Starting the walk over on an unreadable cursor would hand a caller the first page
    again, which reads as the walk having looped rather than as the request having
    failed. Refusing after querying would be indistinguishable in the response and would
    still have read a page nobody receives, so the store's call count is asserted too.
    """
    await harness.upload("pdf-report")
    before = harness.store.page_calls

    refused = await harness.list_skills(page=cursor)

    assert refused.status_code == STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID], (
        refused.text
    )
    body = refused.json()
    assert body["error"]["code"] == ErrorCode.PAGINATION_CURSOR_INVALID.value
    assert body["error"]["detail"]["reason"] == REASON_CURSOR_INVALID
    assert harness.store.page_calls == before, (
        "the refused request read a page anyway, so a malformed cursor costs a query "
        "whose rows nothing receives"
    )


_LIMITS_OUTSIDE_THE_WINDOW = {
    "zero": 0,
    "negative": -1,
    "one past the published maximum": MAX_PAGE_SIZE + 1,
    "far past it": _ADAPTER_MAX_PAGE + 1,
}
"""Each end of the published window, and each way past it.

A route that bounded only the top would serve `limit=0` as an empty page that reports no
cursor, which reads exactly like the end of the collection.
"""


@pytest.mark.parametrize(
    "limit",
    list(_LIMITS_OUTSIDE_THE_WINDOW.values()),
    ids=list(_LIMITS_OUTSIDE_THE_WINDOW),
)
async def test_a_limit_outside_the_published_window_is_refused(
    harness: Harness, limit: int
) -> None:
    """Refused at the surface rather than clamped, and before the store is reached.

    A clamped limit produces a short page, and a short page is how this surface says the
    walk is over -- so clamping would tell a caller it had seen everything it holds. The
    adapter refuses a wider window than this; publishing the tighter one is what turns a
    500 into a 422 naming the field.
    """
    before = harness.store.page_calls

    refused = await harness.list_skills(limit=limit)

    assert refused.status_code == 400, refused.text
    assert harness.store.page_calls == before


async def test_a_cursor_round_trips_through_the_token_it_is_issued_as() -> None:
    """The token is opaque to a caller and exact for this surface.

    A name may contain a dot, which is why the id leads the encoded form: a position
    whose name did not survive the round trip would restart the walk somewhere the
    caller has already been.
    """
    position = SkillCursor(name="pdf.report-v2", skill_id=SkillId(uuid.uuid4()))

    assert SkillCursor.decode(position.encode()) == position
    assert "=" not in position.encode(), (
        "a token carrying padding is percent-encoded in a query string and comes back "
        "looking different from what was issued"
    )


# --- The published envelope, the row's shape, and the filter it accepts -----------


async def test_the_page_carries_its_rows_under_data(harness: Harness) -> None:
    """`data` is the key every collection on this API publishes its rows under.

    A generated client reads `page.data`, so a collection answering under its own name
    hands that client nothing and the walk stops before it starts. Asserted on the
    envelope's exact key set, so a row list arriving under both names -- the compatible-
    looking change -- fails here rather than shipping two spellings of one list.
    """
    await harness.upload("pdf-report")

    listed = await harness.list_skills()

    assert listed.status_code == 200, listed.text
    assert set(listed.json()) == {"data", "next_page", "has_more"}
    assert [row["name"] for row in listed.json()["data"]] == ["pdf-report"]


async def test_the_row_publishes_source_as_the_object_a_client_reads(
    harness: Harness,
) -> None:
    """`source` is an object, and it is the same field the read publishes.

    A client generated from the reference evaluates `skill.source.type`. A bare string
    there raises rather than returning the wrong answer, so this is a break and not a
    cosmetic difference -- and the listing row spelled the same fact a third way, under
    `type`, which meant one hardcoded value in two places free to diverge.
    """
    await harness.upload("pdf-report")

    row = (await harness.list_skills()).json()["data"][0]

    assert row["source"] == {"type": "custom"}
    assert "type" not in row, (
        "the row still publishes the source under a second name, so one value has two "
        "places to be changed and one chance to be changed in only one of them"
    )
    assert set(row) == set(UploadedSkillListed.model_fields)


async def test_the_row_says_which_door_it_arrived_by(harness: Harness) -> None:
    """`origin` is ours, not the reference's, and it is a different axis from `source`.

    `source` says whose catalogue a skill belongs to -- Anthropic's pre-built set or the
    tenant's own -- and everything this platform holds is `custom`. `origin` says which
    of this platform's two write doors wrote it, which decides what a caller may then do
    to it: an upload has an id, versions and a delete, and a repository skill is owned
    by the commit that carries it. Two facts, two fields, and a reader who merged them
    would conclude that a repository skill is not the tenant's own.
    """
    await harness.upload("pdf-report")

    row = (await harness.list_skills()).json()["data"][0]

    assert row["origin"] == "upload"
    assert row["source"] == {"type": "custom"}


async def test_a_source_filter_naming_a_legal_value_we_hold_none_of_answers_empty(
    harness: Harness,
) -> None:
    """`?source=anthropic` is a legal value of its own enum, so it cannot be a refusal.

    Nothing here is `anthropic` -- those are bodies Anthropic ships and this platform
    does not hold -- so the true answer is an empty page. A 400 would say the filter is
    unreadable, which is a different claim and a false one; a full page would say the
    filter was ignored, which is worse than either.
    """
    await harness.upload("pdf-report")

    filtered = await harness.list_skills(source="anthropic")

    assert filtered.status_code == 200, filtered.text
    assert filtered.json() == {"data": [], "next_page": None, "has_more": False}


async def test_a_source_filter_naming_what_we_do_hold_answers_the_rows(
    harness: Harness,
) -> None:
    """The other arm, without which an always-empty filter would pass the test above."""
    await harness.upload("pdf-report")

    filtered = await harness.list_skills(source="custom")

    assert filtered.status_code == 200, filtered.text
    assert [row["name"] for row in filtered.json()["data"]] == ["pdf-report"]


async def test_a_source_outside_the_published_enum_is_refused(
    harness: Harness,
) -> None:
    """A value the enum does not hold is a malformed request, unlike one it does."""
    refused = await harness.list_skills(source="whatever")

    assert refused.status_code == 400, refused.text
    assert "source" in refused.json()["error"]["detail"]["fields"]


async def test_the_published_limit_is_one_the_store_will_actually_serve(
    harness: Harness,
) -> None:
    """The largest published limit must not produce a 500, which is the whole point.

    This route asks the store for one row more than the page, so a published cap equal
    to the store's own window would make the biggest legal request the one request that
    fails. Exercised through the route rather than reasoned about: the fake refuses the
    window the adapter refuses, so a page the real store would reject cannot pass here.

    `test_the_published_cap_leaves_room_for_the_probe_row` pins the same relationship
    arithmetically. Both are worth having and neither replaces the other -- that one
    fails on the constants alone, before any request is built, and names the fix; this
    one proves the route actually asks for what those constants allow, which no
    arithmetic about two integers can show.
    """
    await harness.upload("pdf-report")

    served = await harness.list_skills(limit=MAX_PAGE_SIZE)

    assert served.status_code == 200, served.text
    assert [row["name"] for row in served.json()["data"]] == ["pdf-report"]


def test_the_published_cap_leaves_room_for_the_probe_row() -> None:
    """The published maximum plus the probe row must fit inside the store's own window.

    The one relationship between these two constants, and until now nothing failed when
    it broke. `list_skills` asks the store for `limit + 1` -- one row past the page,
    which is the whole answer to "is there another page" -- so the adapter's bound has
    to exceed the published cap by at least one. Set `MAX_PAGE_SIZE` to the adapter's
    bound, or past it, and every other test in this file still passes: the failure
    appears only when a caller actually sends the published maximum, and it appears as a
    500, the surface reporting that its own documented limit is a server fault.

    Arithmetic and no database, because the claim is about two integers. The over-cap
    cases below exercise the refusal; this pins the number the refusal is drawn at.

    Both sides are imported rather than restated. A copy of either would make this a
    test of two literals agreeing with each other, which is the failure it exists to
    catch -- the adapter's bound was mirrored as `500` in this file and went stale the
    moment the real one moved.
    """
    assert MAX_PAGE_SIZE + 1 <= _MAX_PAGE, (
        f"the listing publishes a maximum page of {MAX_PAGE_SIZE} and asks the store "
        f"for one probe row past it, which is {MAX_PAGE_SIZE + 1} rows; the adapter "
        f"refuses anything above {_MAX_PAGE}. A caller sending the published maximum "
        "would get a 500 for a request this surface documents as legal. Raise "
        "`_MAX_PAGE` in adapters/postgres/skill_registry.py, or lower MAX_PAGE_SIZE."
    )


def test_the_published_cap_is_the_one_the_reference_publishes() -> None:
    """A bare literal, because 1000 is a fact about somebody else's surface.

    Normally an assertion that a constant equals its own value tests nothing. This one
    is different: the number is not ours to choose. A client generated against the
    reference API sends `limit` up to 1000, and a platform capping at anything lower
    refuses a request that client considers legal -- which is how this was found in the
    first place, when the cap here was 100.

    The guard above is one-sided and cannot catch that. It refuses a cap raised past the
    adapter's window, and every test in this file is written relative to
    `MAX_PAGE_SIZE`, so lowering the number back to 100 leaves the whole file green and
    silently restores the incompatibility. This is the assertion that fails instead.

    Raise it only when the reference raises it, and change this number in the same
    commit -- the point is that the two move together, not that either is fixed.
    """
    assert MAX_PAGE_SIZE == 1000, (
        f"this surface publishes a maximum page of {MAX_PAGE_SIZE}; the reference API "
        "publishes 1000, and a client generated against it will send that value. A "
        "lower cap refuses a request the client believes is legal."
    )


async def test_the_row_carries_the_label_the_create_door_was_given(
    harness: Harness,
) -> None:
    """A label exists to be read in a list, so the listing is where it has to appear.

    Published on the row and not only on the read: a caller that had to fetch every
    skill individually to find out what to call it would be paying a request per row for
    a field that exists precisely so a page can be rendered. Both a labelled and an
    unlabelled skill are in the page, because a listing that emitted the label
    unconditionally -- or dropped it unconditionally -- would pass a test holding only
    one of the two.
    """
    labelled = await harness.owner.post(
        "/v1/skills",
        json={"skill_md": _skill_md("alpha"), "display_name": "Alpha Reports"},
    )
    assert labelled.status_code == 201, labelled.text
    assert (await harness.upload("bravo")).status_code == 201

    rows = (await harness.list_skills()).json()["data"]

    assert [(row["name"], row["display_name"]) for row in rows] == [
        ("alpha", "Alpha Reports"),
        ("bravo", None),
    ]


# --- Both write doors in one collection ------------------------------------------


async def _submit(harness: Harness, *names: str, revision: str = _SHA) -> Any:
    """One checkout's skill directory through the real repository door."""
    return await harness.owner.post(
        "/v1/skills/repository",
        json={
            "repository": _REPOSITORY,
            "revision": revision,
            "files": {
                f".claude/skills/{name}/SKILL.md": _skill_md(name) for name in names
            },
        },
    )


async def test_a_repository_submission_hands_back_an_id_for_every_skill(
    harness: Harness,
) -> None:
    """The hole this closes: CI could register skills and then never name one.

    Before the id space covered both doors, a submission answered with bare names and
    every read, delete and version route took a uuid a repository skill did not have --
    so a team whose CI submitted its skills could not enumerate, read or retire any of
    them. The names still come back in order; the ids come back beside them.
    """
    submitted = await _submit(harness, "pdf-report", "citation-check")

    assert submitted.status_code == 201, submitted.text
    body = submitted.json()
    assert body["skills"] == ["citation-check", "pdf-report"]
    assert set(body["skill_ids"]) == {"citation-check", "pdf-report"}
    assert all(uuid.UUID(one) for one in body["skill_ids"].values())


async def test_resubmitting_one_commit_returns_the_same_ids_and_writes_nothing(
    harness: Harness,
) -> None:
    """A retried CI job is a retry, not a second identity for the same skill.

    The id is a `uuid5` over `(tenant, repository, revision, name)` precisely so this
    holds: a fresh uuid would hand the same unchanged skill a second identity, and a
    definition pinning the first would read as retired. `newly_recorded` going to 0 is
    how a retry tells itself from a first submission, and the ids not moving is what
    makes that claim mean anything.
    """
    first = await _submit(harness, "pdf-report")
    again = await _submit(harness, "pdf-report")

    assert (first.status_code, again.status_code) == (201, 201)
    assert first.json()["newly_recorded"] == 1
    assert again.json()["newly_recorded"] == 0
    assert again.json()["skill_ids"] == first.json()["skill_ids"]


async def test_a_different_commit_of_one_repository_gets_different_ids(
    harness: Harness,
) -> None:
    """The revision is part of the identity, so two commits are two sets of skills.

    Without this the id would collapse two checkouts into one row and a definition
    pinning the older commit would resolve to the newer body -- which is the whole thing
    a pinned revision exists to prevent.
    """
    old = await _submit(harness, "pdf-report", revision="a" * 40)
    new = await _submit(harness, "pdf-report", revision="b" * 40)

    assert old.json()["skill_ids"] != new.json()["skill_ids"]


async def test_the_listing_carries_both_origins_with_the_fields_each_one_has(
    harness: Harness,
) -> None:
    """One collection, two row shapes, and the origin says which is which.

    An uploaded row carries a `display_name` and no checkout; a repository row carries
    the `(repository, revision)` pair a definition pins and no label -- a checkout
    submits a directory rather than a labelled skill, so the label is a field that does
    not apply rather than a null waiting to be filled.

    Asserted on each row's exact key set, so a field leaking across the boundary as a
    null fails here. That is the failure the union exists to prevent: one row shape with
    every column nullable cannot tell "not applicable" from "not set", and a caller
    reading a null `revision` would conclude an uploaded skill came from a commit
    nobody recorded.
    """
    assert (await harness.upload("alpha")).status_code == 201
    assert (await _submit(harness, "bravo")).status_code == 201

    rows = (await harness.list_skills()).json()["data"]

    assert [(row["name"], row["origin"]) for row in rows] == [
        ("alpha", "upload"),
        ("bravo", "repository"),
    ]
    assert set(rows[0]) == {
        "id",
        "name",
        "description",
        "display_name",
        "source",
        "origin",
    }
    assert set(rows[1]) == {
        "id",
        "name",
        "description",
        "repository",
        "revision",
        "source",
        "origin",
    }
    assert rows[1]["revision"] == _SHA
    assert rows[0]["source"] == rows[1]["source"] == {"type": "custom"}


async def test_the_walk_crosses_the_origin_boundary_without_repeating_or_dropping(
    harness: Harness,
) -> None:
    """The claim the whole two-door listing rests on, walked one row at a time.

    A page boundary that falls *between* the two origins is the case a merged pair of
    per-origin pages gets wrong, and it gets it wrong silently: the walk reaches what
    looks like the end having served part of what the tenant holds. Interleaved by name
    on purpose -- `alpha` and `charlie` uploaded, `bravo` and `delta` submitted -- so
    the keyset has to be total across the union rather than across either half, and
    every boundary in the collection is a boundary between origins.

    One row per page so every boundary is exercised: with a page big enough to hold
    everything there is no boundary left to get wrong.
    """
    for name in ("alpha", "charlie"):
        assert (await harness.upload(name)).status_code == 201
    assert (await _submit(harness, "bravo", "delta")).status_code == 201

    rows = await _walk(harness, limit=1)

    assert [row["name"] for row in rows] == ["alpha", "bravo", "charlie", "delta"]
    assert [row["origin"] for row in rows] == [
        "upload",
        "repository",
        "upload",
        "repository",
    ]


async def test_a_repository_skill_reads_back_by_the_id_it_was_assigned(
    harness: Harness,
) -> None:
    """The read that made the id worth assigning, body included.

    A body is served here and on no listing row: the caller asked for exactly this skill
    and there is one of it, where a page of bodies is a page nobody can walk.

    `latest_version` is null and the `(repository, revision)` pair beside it is what
    actually pins the body. Reporting the revision there was the alternative and it is
    worse: `latest_version` is a version string this platform minted, every version
    route parses one as sixteen-to-nineteen digits, and a caller handed a 40-character
    commit id would take it straight to a route that refuses it. Null plus the commit
    says the same thing without publishing a value nothing accepts."""
    submitted = await _submit(harness, "pdf-report")
    skill_id = submitted.json()["skill_ids"]["pdf-report"]

    read = await harness.owner.get(f"/v1/skills/{skill_id}")

    assert read.status_code == 200, read.text
    body = read.json()
    assert body["name"] == "pdf-report"
    assert body["origin"] == "repository"
    assert body["repository"] == _REPOSITORY
    assert body["revision"] == _SHA
    assert body["latest_version"] is None
    assert body["body"] == _skill_md("pdf-report")


async def test_another_tenants_repository_skill_reads_as_absent(
    harness: Harness,
) -> None:
    """The id is a pure function of the key, so a stranger can compute one and try it.

    That is exactly why the tenant has to be a term in the lookup rather than a filter
    over what came back: `repository_skill_id` is `uuid5`, so anyone who knows the
    repository, the revision and the name can derive another tenant's id without ever
    having seen it. Absent rather than forbidden, so the answer reveals nothing about
    whether the row exists.
    """
    submitted = await _submit(harness, "pdf-report")
    skill_id = submitted.json()["skill_ids"]["pdf-report"]

    refused = await harness.stranger.get(f"/v1/skills/{skill_id}")

    assert refused.status_code == STATUS_FOR[ErrorCode.SKILL_NOT_FOUND], refused.text


async def test_deleting_a_repository_skill_is_refused_naming_the_commit(
    harness: Harness,
) -> None:
    """A repository skill is owned by its commit, so this surface cannot remove it.

    A definition pins `(repository, revision)`, and deleting the row would make an
    already-registered definition unresolvable while the commit it names still exists --
    the platform reporting lost data for a skill that is still exactly where CI put it.
    The way to stop using one is to stop pinning that revision, and the refusal says so.
    """
    submitted = await _submit(harness, "pdf-report")
    skill_id = submitted.json()["skill_ids"]["pdf-report"]

    refused = await harness.owner.delete(f"/v1/skills/{skill_id}")

    assert refused.status_code == STATUS_FOR[ErrorCode.SKILL_OWNED_BY_COMMIT], (
        refused.text
    )
    assert refused.json()["error"]["code"] == ErrorCode.SKILL_OWNED_BY_COMMIT.value
    assert "revision" in refused.json()["error"]["message"]


async def test_writing_a_version_of_a_repository_skill_is_refused(
    harness: Harness,
) -> None:
    """The commit is that skill's version, so a version row here would shadow it.

    Refused for the same reason the delete is, through the same code: a version written
    against a repository skill would be a body this platform minted sitting under an id
    whose body is fixed by a commit, and resolution would have two answers to what the
    skill says.
    """
    submitted = await _submit(harness, "pdf-report")
    skill_id = submitted.json()["skill_ids"]["pdf-report"]

    refused = await harness.owner.post(
        f"/v1/skills/{skill_id}/versions",
        files=[("files", ("pdf-report/SKILL.md", b"x", "text/markdown"))],
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.SKILL_OWNED_BY_COMMIT], (
        refused.text
    )
    assert refused.json()["error"]["code"] == ErrorCode.SKILL_OWNED_BY_COMMIT.value


async def test_an_id_neither_door_minted_is_one_not_found(harness: Harness) -> None:
    """Both halves are tried before a skill is called absent.

    A caller holding an id has no way to know which door minted it, so a 404 that only
    consulted the uploaded half would be wrong about every repository skill -- and a
    single not-found is the right answer for an id neither holds, rather than two
    refusals a caller would have to tell apart.
    """
    refused = await harness.owner.get(f"/v1/skills/{uuid.uuid4()}")

    assert refused.status_code == STATUS_FOR[ErrorCode.SKILL_NOT_FOUND], refused.text
