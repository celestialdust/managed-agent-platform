"""Reading an agent back, editing it under a version, and retiring it for good.

Tier 1 (local, no infrastructure). The routes run inside the real `create_app` over an
in-memory store, because every claim here is a property of *this surface* -- which rows
a page carries, which refusal a stale version gets, what a second archive answers -- and
each one is reachable without a database.

The store's own guarantees are not claimed here. That a revision cannot be rewritten,
that a retirement is one row whose timestamp never moves, and that the keyset boundary
neither repeats a row nor drops one against a real index, are properties of PostgreSQL
and are graded against it in `tests/adapters/test_agent_archive_schema.py`.

The in-memory store below therefore holds the rules the schema holds rather than a
convenient subset: revisions are numbered per id and never replaced, an id belongs to
one tenant, an agent is retired at most once, and a page is sorted and cut by the same
`(created_at, id)` keyset the adapter uses. A fake that let any of them go would certify
a route the real store refuses.

**Two tenants appear in every isolation test on purpose.** A read that dropped its
tenant term behaves perfectly normally with one tenant in the store -- it returns
exactly the rows that tenant registered -- so a single-tenant test proves nothing about
isolation.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.agents_lifecycle import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    Cursor,
)
from managed_agent.control.catalog.definitions import AgentRecord
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.skills.evaluation import (
    Baseline,
    EvalFacts,
    Grade,
    RunRecord,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import DefinitionId, TenantId
from managed_agent.core.ports import UnknownDefinition
from managed_agent.core.registration.definition import AgentDefinition, VersionFact

_SHA = "0" * 39 + "a"
_BLOCKED_SHA = "b" * 40
_REPOSITORY = "git@github.com:acme/skills.git"

# Every walk in this file is bounded. A keyset boundary written inclusively, or a route
# that never stops issuing a cursor, makes a walk that reads pages forever -- and an
# unbounded loop turns that into a test that hangs rather than one that fails.
_PAGE_BUDGET = 20


def a_definition(name: str = "reviewer", *, revision: str = _SHA) -> dict[str, object]:
    return {
        "name": name,
        "instructions": "read the diff before the plan",
        "model": "gpt-5-codex",
        "skills_repository": _REPOSITORY,
        "skills_revision": revision,
    }


@dataclass(frozen=True, slots=True)
class _Row:
    """One row of `agent_definition`, as that table holds it."""

    tenant_id: TenantId
    revision: int
    registered_at: datetime
    body: AgentDefinition


@dataclass(frozen=True, slots=True)
class _Resolved:
    definition: AgentDefinition
    revision: int


class AgentsHeldInMemory:
    """`agent_definition`, `agent_version_archive` and `agent_archive` in memory.

    Holds the tables' own rules, not a convenient subset. A revision is never rewritten,
    an id belongs to one tenant, a version and an agent are each retired at most once,
    and the whole-agent reads fold the revisions exactly as the adapter's CTE does --
    newest revision for the shape, earliest timestamp for the creation.

    The clock is a counter rather than a real one, so `created_at` is distinct and
    increasing per registration. Two agents sharing a timestamp is the case the keyset's
    second half exists for, and it is reachable here through `register_at` rather than
    by hoping two calls land in the same microsecond.
    """

    def __init__(self) -> None:
        self._rows: dict[DefinitionId, list[_Row]] = {}
        self._retired_versions: set[tuple[DefinitionId, int]] = set()
        self._retired_agents: dict[DefinitionId, datetime] = {}
        self._blocked: set[str] = set()
        self._ticks = 0
        self._forced: dict[DefinitionId, datetime] = {}

    # -- the fake's own controls, not part of any port -----------------------

    def block(self, revision: str) -> None:
        """Put this skills revision behind the CI eval gate."""
        self._blocked.add(revision)

    def register_at(self, definition_id: DefinitionId, moment: datetime) -> None:
        """Pin the creation timestamp the next registration of this id will take."""
        self._forced[definition_id] = moment

    def revisions(self, definition_id: DefinitionId) -> list[int]:
        return [row.revision for row in self._rows.get(definition_id, ())]

    def _next_moment(self, definition_id: DefinitionId) -> datetime:
        forced = self._forced.pop(definition_id, None)
        if forced is not None:
            return forced
        self._ticks += 1
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self._ticks)

    def _owned(self, definition_id: DefinitionId, tenant_id: TenantId) -> list[_Row]:
        return [
            row
            for row in self._rows.get(definition_id, ())
            if row.tenant_id == tenant_id
        ]

    def _fold(self, definition_id: DefinitionId, rows: Sequence[_Row]) -> AgentRecord:
        newest = max(rows, key=lambda row: row.revision)
        return AgentRecord(
            definition_id=definition_id,
            version=newest.revision,
            created_at=min(row.registered_at for row in rows),
            archived_at=self._retired_agents.get(definition_id),
            definition=newest.body,
        )

    # -- DefinitionRegistry -------------------------------------------------

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        rows = self._rows.setdefault(definition_id, [])
        assert all(row.tenant_id == tenant_id for row in rows), (
            "an id belongs to one tenant; the real table numbers revisions across the "
            "id so a second tenant cannot hold its own revision 1"
        )
        revision = max((row.revision for row in rows), default=0) + 1
        rows.append(
            _Row(
                tenant_id=tenant_id,
                revision=revision,
                registered_at=self._next_moment(definition_id),
                body=definition,
            )
        )
        return revision

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> _Resolved:
        rows = self._owned(definition_id, tenant_id)
        if not rows:
            raise UnknownDefinition(str(definition_id))
        newest = max(rows, key=lambda row: row.revision)
        return _Resolved(definition=newest.body, revision=newest.revision)

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> tuple[VersionFact, ...]:
        return tuple(
            VersionFact(
                revision=row.revision,
                archived=(definition_id, row.revision) in self._retired_versions,
            )
            for row in sorted(
                self._owned(definition_id, tenant_id), key=lambda r: r.revision
            )
        )

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        for row in self._owned(definition_id, tenant_id):
            if row.revision == revision:
                return row.body
        return None

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        key = (definition_id, revision)
        if key in self._retired_versions:
            return False
        owned = self._owned(definition_id, tenant_id)
        if not any(row.revision == revision for row in owned):
            return False
        self._retired_versions.add(key)
        return True

    # -- AgentLifecycle -----------------------------------------------------

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
        folded = [
            self._fold(definition_id, rows)
            for definition_id, rows in self._rows.items()
            if any(row.tenant_id == tenant_id for row in rows)
        ]
        kept = [
            record
            for record in folded
            if (include_archived or record.archived_at is None)
            and (created_from is None or record.created_at >= created_from)
            and (created_to is None or record.created_at <= created_to)
            and (after is None or (record.created_at, record.definition_id) < after)
        ]
        kept.sort(key=lambda record: (record.created_at, record.definition_id))
        kept.reverse()
        return tuple(kept[:limit])

    async def read_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> AgentRecord | None:
        rows = self._owned(definition_id, tenant_id)
        return None if not rows else self._fold(definition_id, rows)

    async def archive_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> datetime | None:
        if not self._owned(definition_id, tenant_id):
            return None
        held = self._retired_agents.get(definition_id)
        if held is not None:
            return held
        self._ticks += 1
        written = datetime(2026, 6, 1, tzinfo=UTC) + timedelta(seconds=self._ticks)
        self._retired_agents[definition_id] = written
        return written

    async def register_at_revision(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
        expected: int,
    ) -> int | None:
        rows = self._owned(definition_id, tenant_id)
        if not rows or max(row.revision for row in rows) != expected:
            return None
        return await self.register(definition_id, tenant_id, definition)

    # -- SkillEvalStore -----------------------------------------------------

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        blocked = revision in self._blocked
        return EvalFacts(repository_enrolled=blocked, revision_accepted=False)

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        raise AssertionError("an agent-lifecycle test read the eval baselines")

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        raise AssertionError("an agent-lifecycle test recorded an eval run")


class Unused:
    """One raising stand-in for every port these routes never touch."""

    def __getattr__(self, name: str) -> Any:
        async def refuse(*args: object, **kwargs: object) -> Any:
            raise AssertionError(f"an agent-lifecycle test called {name}")

        return refuse


@dataclass(frozen=True, slots=True)
class Harness:
    """One app, one store, and two tenants that both talk to it.

    Two clients rather than one plus a header override, so no test can accidentally read
    as the wrong tenant by forgetting to pass one.
    """

    store: AgentsHeldInMemory
    owner: AsyncClient
    stranger: AsyncClient
    created: list[str] = field(default_factory=list)

    async def create(self, name: str = "reviewer", *, as_stranger: bool = False) -> str:
        """Register an agent through the real create route and return its id."""
        client = self.stranger if as_stranger else self.owner
        answered = await client.post("/v1/agents", json=a_definition(name))
        assert answered.status_code == 201, answered.text
        minted = str(answered.json()["id"])
        self.created.append(minted)
        return minted


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    store = AgentsHeldInMemory()
    platform = Platform(
        event_log_append=Unused(),
        event_log_range=Unused(),
        definition_registry=store,
        tool_registry=Unused(),
        session_registry=Unused(),
        webhooks=Unused(),
        environment_store=Unused(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
        skill_store=Unused(),
    )
    app = create_app(platform)
    async with (
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://control-plane",
            headers={TENANT_HEADER: str(uuid.uuid4())},
        ) as owner,
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://control-plane",
            headers={TENANT_HEADER: str(uuid.uuid4())},
        ) as stranger,
    ):
        yield Harness(store=store, owner=owner, stranger=stranger)


def _code(answered: Any) -> str:
    """The published code out of a refusal envelope, as its wire string."""
    return str(answered.json()["error"]["code"])


async def _walk(harness: Harness, **query: str | int) -> list[dict[str, Any]]:
    """Every row of a listing, following `next_page` to the end.

    Written as a walk rather than one request because the claims that matter about
    pagination -- no row repeated, no row dropped -- are claims about the whole sequence
    and are invisible in any single page.
    """
    collected: list[dict[str, Any]] = []
    page: str | None = None
    for _ in range(_PAGE_BUDGET):
        sent = dict(query)
        if page is not None:
            sent["page"] = page
        answered = await harness.owner.get("/v1/agents", params=sent)
        assert answered.status_code == 200, answered.text
        body = answered.json()
        collected.extend(body["data"])
        page = body["next_page"]
        if page is None:
            return collected
    raise AssertionError("the listing never stopped issuing a next page")


# -- reading one agent back --------------------------------------------------


async def test_an_agent_reads_back_as_the_definition_that_was_registered(
    harness: Harness,
) -> None:
    """The whole point of the read: what went in comes out, addressable by its id.

    The configuration fields are asserted at the top level rather than under a nested
    key, because that is the shape a generated client parses and moving them would be a
    breaking change no status code would report.
    """
    agent_id = await harness.create("reviewer")

    answered = await harness.owner.get(f"/v1/agents/{agent_id}")

    assert answered.status_code == 200, answered.text
    body = answered.json()
    assert body["id"] == agent_id
    assert body["name"] == "reviewer"
    assert body["instructions"] == "read the diff before the plan"
    assert body["model"] == "gpt-5-codex"
    assert body["skills_revision"] == _SHA
    assert body["version"] == 1
    assert body["archived_at"] is None
    assert body["created_at"]


async def test_an_id_nobody_registered_is_refused_as_not_found(
    harness: Harness,
) -> None:
    answered = await harness.owner.get(f"/v1/agents/{uuid.uuid4()}")

    assert answered.status_code == STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]
    assert _code(answered) == ErrorCode.DEFINITION_NOT_FOUND.value


async def test_another_tenants_agent_reads_as_absent_rather_than_forbidden(
    harness: Harness,
) -> None:
    """The same refusal an unregistered id gets, and that is the point.

    A distinct refusal would tell anybody holding an id whether it names an agent
    somebody else owns, which is a fact about another tenant leaking out of a 404.
    """
    theirs = await harness.create("theirs", as_stranger=True)

    answered = await harness.owner.get(f"/v1/agents/{theirs}")

    assert answered.status_code == STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]
    assert _code(answered) == ErrorCode.DEFINITION_NOT_FOUND.value


async def test_the_version_a_read_reports_is_the_newest_revision(
    harness: Harness,
) -> None:
    """An edit moves the number a read reports, which is what makes it a token.

    A `version` that stayed at 1 while revisions accumulated would be returned by every
    read, sent back by every update, and match forever -- an optimistic-concurrency
    check that can never fail.
    """
    agent_id = await harness.create("reviewer")
    edited = await harness.owner.post(
        f"/v1/agents/{agent_id}", json=a_definition("renamed")
    )
    assert edited.status_code == 200, edited.text

    answered = await harness.owner.get(f"/v1/agents/{agent_id}")

    assert answered.json()["version"] == 2
    assert answered.json()["name"] == "renamed"


# -- listing -----------------------------------------------------------------


async def test_a_listing_carries_the_tenants_agents_newest_first(
    harness: Harness,
) -> None:
    first = await harness.create("first")
    second = await harness.create("second")
    third = await harness.create("third")

    answered = await harness.owner.get("/v1/agents")

    assert answered.status_code == 200, answered.text
    body = answered.json()
    assert [row["id"] for row in body["data"]] == [third, second, first]
    assert body["next_page"] is None


async def test_a_listing_holds_no_other_tenants_agents(harness: Harness) -> None:
    mine = await harness.create("mine")
    await harness.create("theirs", as_stranger=True)

    listed = await _walk(harness)

    assert [row["id"] for row in listed] == [mine]


async def test_a_retired_agent_leaves_the_listing_and_comes_back_when_asked(
    harness: Harness,
) -> None:
    """Both halves, because either alone is satisfiable by a filter that does nothing.

    A listing that always hid retired agents would pass the first assertion and a
    listing that never hid them would pass the second.
    """
    live = await harness.create("live")
    retired = await harness.create("retired")
    archived = await harness.owner.post(f"/v1/agents/{retired}/archive")
    assert archived.status_code == 200, archived.text

    by_default = await _walk(harness)
    with_retired = await _walk(harness, include_archived="true")

    assert [row["id"] for row in by_default] == [live]
    assert {row["id"] for row in with_retired} == {live, retired}
    assert (
        next(row["archived_at"] for row in with_retired if row["id"] == retired)
        is not None
    )


async def test_a_walk_a_page_at_a_time_reads_every_agent_exactly_once(
    harness: Harness,
) -> None:
    """The property that matters about a cursor, and it is invisible in one page.

    A boundary written inclusively repeats a row and one written past the end drops one;
    both look like a working listing until the pages are laid end to end.
    """
    minted = [await harness.create(f"agent-{n}") for n in range(5)]

    walked = await _walk(harness, limit=1)

    assert [row["id"] for row in walked] == list(reversed(minted))


async def test_two_agents_created_in_the_same_moment_are_still_walked_once(
    harness: Harness,
) -> None:
    """Why the cursor carries the id as well as the timestamp.

    A position naming only the timestamp cannot say which of two agents sharing it the
    caller already holds, so the page boundary either repeats one or drops the other.
    """
    together = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    first = DefinitionId(uuid.uuid4())
    second = DefinitionId(uuid.uuid4())
    tenant = TenantId(uuid.UUID(harness.owner.headers[TENANT_HEADER]))
    for minted in (first, second):
        harness.store.register_at(minted, together)
        await harness.store.register(
            minted, tenant, AgentDefinition.model_validate(a_definition("twin"))
        )

    walked = await _walk(harness, limit=1)

    assert sorted(row["id"] for row in walked) == sorted([str(first), str(second)])


async def test_a_cursor_this_surface_never_issued_is_refused(
    harness: Harness,
) -> None:
    """Refused rather than treated as the start of the collection.

    Starting over on a bad cursor hands the caller the newest page again, which reads as
    the walk having looped rather than failed.
    """
    answered = await harness.owner.get("/v1/agents", params={"page": "not-a-cursor"})

    assert answered.status_code == STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]
    assert _code(answered) == ErrorCode.PAGINATION_CURSOR_INVALID.value


async def test_a_cursor_round_trips_through_its_own_encoding(harness: Harness) -> None:
    """The token is opaque to a caller and exact to this surface.

    Asserted directly because the walk above would still pass if the timestamp were
    rounded -- it only fails once two rows land close enough for the rounding to matter.
    """
    moment = datetime(2026, 3, 1, 12, 0, 0, 123_456, tzinfo=UTC)
    position = Cursor(moment, DefinitionId(uuid.uuid4()))

    assert Cursor.decode(position.encode()) == position


async def test_a_page_larger_than_the_published_maximum_is_refused(
    harness: Harness,
) -> None:
    """An unbounded page is a whole-collection read wearing a limit parameter."""
    answered = await harness.owner.get(
        "/v1/agents", params={"limit": MAX_PAGE_SIZE + 1}
    )

    assert answered.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert _code(answered) == ErrorCode.REQUEST_INVALID.value


async def test_the_default_page_size_is_the_one_the_surface_publishes(
    harness: Harness,
) -> None:
    """A caller naming no limit gets the documented page, not the collection."""
    for n in range(DEFAULT_PAGE_SIZE + 3):
        await harness.create(f"agent-{n}")

    answered = await harness.owner.get("/v1/agents")

    body = answered.json()
    assert len(body["data"]) == DEFAULT_PAGE_SIZE
    assert body["next_page"] is not None


async def test_the_created_at_bounds_are_inclusive_and_cut_both_ends(
    harness: Harness,
) -> None:
    """Both bounds at once: each alone passes for a filter that ignores it."""
    tenant = TenantId(uuid.UUID(harness.owner.headers[TENANT_HEADER]))
    moments = {
        "early": datetime(2026, 2, 1, tzinfo=UTC),
        "middle": datetime(2026, 3, 1, tzinfo=UTC),
        "late": datetime(2026, 4, 1, tzinfo=UTC),
    }
    minted: dict[str, DefinitionId] = {}
    for label, moment in moments.items():
        agent_id = DefinitionId(uuid.uuid4())
        minted[label] = agent_id
        harness.store.register_at(agent_id, moment)
        await harness.store.register(
            agent_id, tenant, AgentDefinition.model_validate(a_definition(label))
        )

    bounded = await _walk(
        harness,
        **{
            "created_at[gte]": moments["middle"].isoformat(),
            "created_at[lte]": moments["late"].isoformat(),
        },
    )

    assert {row["id"] for row in bounded} == {
        str(minted["middle"]),
        str(minted["late"]),
    }


async def test_a_created_at_bound_with_no_offset_is_refused_naming_the_field(
    harness: Harness,
) -> None:
    """A bound with no offset selects different rows depending on where you stand.

    Refused at the boundary rather than interpreted, because either interpretation --
    the server's zone or UTC -- is a guess about the caller's intent that silently
    shifts which agents come back by hours.
    """
    answered = await harness.owner.get(
        "/v1/agents", params={"created_at[gte]": "2026-03-01T00:00:00"}
    )

    assert answered.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert "created_at[gte]" in answered.json()["error"]["detail"]["fields"]


# -- updating ----------------------------------------------------------------


async def test_an_update_appends_a_revision_and_reports_the_number_it_landed_on(
    harness: Harness,
) -> None:
    """Additive, because `agent_definition` refuses an UPDATE by trigger.

    The earlier revision is asserted still present: a Session that resolved it goes on
    reading exactly the bytes it resolved to, and an edit that replaced the row would
    change what a running Session is.
    """
    agent_id = await harness.create("before")

    answered = await harness.owner.post(
        f"/v1/agents/{agent_id}", json=a_definition("after")
    )

    assert answered.status_code == 200, answered.text
    assert answered.json()["version"] == 2
    assert answered.json()["name"] == "after"
    assert harness.store.revisions(DefinitionId(uuid.UUID(agent_id))) == [1, 2]
    kept = await harness.store.read_version(
        DefinitionId(uuid.UUID(agent_id)),
        TenantId(uuid.UUID(harness.owner.headers[TENANT_HEADER])),
        1,
    )
    assert kept is not None and kept.name == "before"


async def test_an_update_carrying_the_current_version_is_applied(
    harness: Harness,
) -> None:
    agent_id = await harness.create("before")

    answered = await harness.owner.post(
        f"/v1/agents/{agent_id}", json={**a_definition("after"), "version": 1}
    )

    assert answered.status_code == 200, answered.text
    assert answered.json()["version"] == 2


async def test_an_update_carrying_a_stale_version_is_refused_and_writes_nothing(
    harness: Harness,
) -> None:
    """The refusal names the version the platform holds, so the retry is a re-read.

    Nothing written is asserted separately from the status: a route that refused and
    appended anyway would answer 409 while leaving the edit in place, which is the worse
    of the two failures because the caller believes it did not happen.
    """
    agent_id = await harness.create("before")
    moved = await harness.owner.post(
        f"/v1/agents/{agent_id}", json=a_definition("second")
    )
    assert moved.status_code == 200, moved.text

    answered = await harness.owner.post(
        f"/v1/agents/{agent_id}", json={**a_definition("third"), "version": 1}
    )

    assert answered.status_code == STATUS_FOR[ErrorCode.AGENT_VERSION_CONFLICT]
    assert _code(answered) == ErrorCode.AGENT_VERSION_CONFLICT.value
    assert answered.json()["error"]["detail"]["version"] == 2
    assert harness.store.revisions(DefinitionId(uuid.UUID(agent_id))) == [1, 2]


async def test_a_stale_version_is_refused_even_when_the_body_is_already_stored(
    harness: Harness,
) -> None:
    """The check compares versions, not content, and this is the case that shows it.

    A caller re-sending exactly what is stored is still refused, because what it is
    telling us is which state it read -- and it has not read the state its write would
    land on, so it cannot know whether re-sending its own body undoes an edit.
    """
    agent_id = await harness.create("before")
    settled = a_definition("settled")
    first = await harness.owner.post(f"/v1/agents/{agent_id}", json=settled)
    assert first.status_code == 200, first.text

    answered = await harness.owner.post(
        f"/v1/agents/{agent_id}", json={**settled, "version": 1}
    )

    assert answered.status_code == STATUS_FOR[ErrorCode.AGENT_VERSION_CONFLICT]
    assert _code(answered) == ErrorCode.AGENT_VERSION_CONFLICT.value


async def test_an_update_with_no_version_applies_unconditionally(
    harness: Harness,
) -> None:
    """Last write wins, and that is the caller's choice rather than this surface's.

    Omitting the field is a different request from sending a stale number: the first
    says "I have no opinion about what was there", and only the second is a claim that
    can be wrong.
    """
    agent_id = await harness.create("before")
    await harness.owner.post(f"/v1/agents/{agent_id}", json=a_definition("second"))

    answered = await harness.owner.post(
        f"/v1/agents/{agent_id}", json=a_definition("third")
    )

    assert answered.status_code == 200, answered.text
    assert answered.json()["version"] == 3
    assert answered.json()["name"] == "third"


async def test_a_version_below_one_is_refused_at_the_boundary(
    harness: Harness,
) -> None:
    """Zero is not a version any agent has ever had, so it can only be a mistake."""
    agent_id = await harness.create("before")

    answered = await harness.owner.post(
        f"/v1/agents/{agent_id}", json={**a_definition("after"), "version": 0}
    )

    assert answered.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert "version" in answered.json()["error"]["detail"]["fields"]


async def test_updating_an_id_nobody_registered_is_refused_as_not_found(
    harness: Harness,
) -> None:
    """Refused rather than written as a brand-new agent at revision 1.

    The store numbers a revision from whatever rows carry the id, so an unchecked update
    of an unknown id would register an agent -- by a call that reads as an edit.
    """
    answered = await harness.owner.post(
        f"/v1/agents/{uuid.uuid4()}", json=a_definition("invented")
    )

    assert answered.status_code == STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]
    assert _code(answered) == ErrorCode.DEFINITION_NOT_FOUND.value


async def test_updating_another_tenants_agent_is_refused_and_writes_nothing(
    harness: Harness,
) -> None:
    theirs = await harness.create("theirs", as_stranger=True)

    answered = await harness.owner.post(
        f"/v1/agents/{theirs}", json=a_definition("hijacked")
    )

    assert answered.status_code == STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]
    assert harness.store.revisions(DefinitionId(uuid.UUID(theirs))) == [1]


async def test_an_update_pinning_a_blocked_skills_revision_is_refused(
    harness: Harness,
) -> None:
    """The third door onto the same gate, and it has to be checked here too.

    `POST /v1/agents` and `POST /v1/agents/{id}/versions` both consult the CI eval gate.
    This route accepts a whole definition, so it accepts a `skills_revision` -- and a
    gate held on the other two doors alone would be walked around by updating an agent
    instead of creating or versioning one.
    """
    agent_id = await harness.create("before")
    harness.store.block(_BLOCKED_SHA)

    answered = await harness.owner.post(
        f"/v1/agents/{agent_id}",
        json=a_definition("after", revision=_BLOCKED_SHA),
    )

    assert (
        answered.status_code
        == STATUS_FOR[ErrorCode.DEFINITION_SKILLS_REVISION_NOT_ACCEPTED]
    )
    assert _code(answered) == ErrorCode.DEFINITION_SKILLS_REVISION_NOT_ACCEPTED.value
    assert harness.store.revisions(DefinitionId(uuid.UUID(agent_id))) == [1]


# -- archiving ---------------------------------------------------------------


async def test_archiving_returns_the_whole_agent_with_the_moment_it_was_retired(
    harness: Harness,
) -> None:
    """The whole agent rather than an acknowledgement.

    There is no unarchive and no delete, so this response is the last full description
    of the agent that will ever change.
    """
    agent_id = await harness.create("retiring")

    answered = await harness.owner.post(f"/v1/agents/{agent_id}/archive")

    assert answered.status_code == 200, answered.text
    body = answered.json()
    assert body["id"] == agent_id
    assert body["name"] == "retiring"
    assert body["version"] == 1
    assert body["archived_at"] is not None


async def test_a_retired_agent_still_reads_back_with_its_retirement_showing(
    harness: Harness,
) -> None:
    """Unusable, not invisible.

    A caller holding a Session that resolved this agent needs to find out what it was
    and why it can no longer be started; a 404 would send them looking for a mistake in
    an id that is correct.
    """
    agent_id = await harness.create("retiring")
    archived = await harness.owner.post(f"/v1/agents/{agent_id}/archive")

    answered = await harness.owner.get(f"/v1/agents/{agent_id}")

    assert answered.status_code == 200, answered.text
    assert answered.json()["archived_at"] == archived.json()["archived_at"]


async def test_archiving_twice_succeeds_and_keeps_the_first_moment(
    harness: Harness,
) -> None:
    """A retry is not a second request, and the timestamp is a fact about the first.

    A fresh timestamp would say the agent was retired at the moment of the retry, which
    is wrong by however long the caller's first call took to time out -- and anything
    reasoning about when references to it stopped resolving would inherit the error.
    """
    agent_id = await harness.create("retiring")
    first = await harness.owner.post(f"/v1/agents/{agent_id}/archive")

    second = await harness.owner.post(f"/v1/agents/{agent_id}/archive")

    assert second.status_code == 200, second.text
    assert second.json()["archived_at"] == first.json()["archived_at"]


async def test_a_retired_agent_refuses_an_update(harness: Harness) -> None:
    """Archive is terminal, so an edit after it would make it reversible in effect.

    A newer revision than the one that was current when the agent was retired would make
    "what was this agent when it was retired" unanswerable from the store.
    """
    agent_id = await harness.create("retiring")
    await harness.owner.post(f"/v1/agents/{agent_id}/archive")

    answered = await harness.owner.post(
        f"/v1/agents/{agent_id}", json=a_definition("after")
    )

    assert answered.status_code == STATUS_FOR[ErrorCode.AGENT_ARCHIVED]
    assert _code(answered) == ErrorCode.AGENT_ARCHIVED.value
    assert harness.store.revisions(DefinitionId(uuid.UUID(agent_id))) == [1]


async def test_archiving_an_id_nobody_registered_is_refused_as_not_found(
    harness: Harness,
) -> None:
    answered = await harness.owner.post(f"/v1/agents/{uuid.uuid4()}/archive")

    assert answered.status_code == STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]
    assert _code(answered) == ErrorCode.DEFINITION_NOT_FOUND.value


async def test_archiving_another_tenants_agent_refuses_and_retires_nothing(
    harness: Harness,
) -> None:
    """The refusal is not the whole claim: the agent must still be startable.

    A route that refused the caller and wrote the retirement anyway would let anybody
    holding an id retire somebody else's agent, and the 404 would read as though nothing
    had happened.
    """
    theirs = await harness.create("theirs", as_stranger=True)

    answered = await harness.owner.post(f"/v1/agents/{theirs}/archive")

    assert answered.status_code == STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]
    still_live = await harness.stranger.get(f"/v1/agents/{theirs}")
    assert still_live.json()["archived_at"] is None
