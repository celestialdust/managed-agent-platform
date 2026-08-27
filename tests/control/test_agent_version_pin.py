"""What a version pin buys, and what retiring a version does and does not stop.

Tier 1 (local, in-memory ports). Both claims are relationships between three surfaces --
registering an agent, adding a version to it, and creating a Session against it -- so
all three run on **one** app over **one** store, built by the real `create_app`. Two
apps could not show that a Session created before an edit still resolves what it
resolved, and a hand-assembled app could not show that the versions router is attached.

The store is a fake rather than a mock, because every assertion here is about a response
body or a resolved revision and a mock would let those be wrong while the assertions
passed. Its own behaviour is pinned by the first cases in this file, so a later failure
is a failure of the code under test.

`choose_revision` is graded directly as well. The interesting cases -- a pin on a
retired revision, an unpinned reference whose newest revision was withdrawn, an agent
with nothing live left -- are a pure function of the version facts, and stating them
through HTTP would hide which of the two was wrong when one broke.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.catalog.definitions import (
    AgentRecord,
    AgentReference,
    AgentVersionArchived,
    UnknownAgentVersion,
    choose_revision,
    resolve_reference,
)
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.skills.evaluation import (
    Baseline,
    EvalFacts,
    Grade,
    RunRecord,
)
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import FIRST_SEQ, DefinitionId, Seq, SessionId, TenantId
from managed_agent.core.ports import (
    Resolution,
    SessionListing,
    SessionNotVisible,
    UnknownDefinition,
)
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord

SKILLS_REPO = "git@github.com:acme/skills.git"
PUBLISHED = "a" * 40


def a_definition(instructions: str) -> dict[str, object]:
    return {
        "name": "slr-reviewer",
        "instructions": instructions,
        "model": "gpt-5-codex",
        "skills_repository": SKILLS_REPO,
        "skills_revision": PUBLISHED,
    }


def a_session(definition_id: str, version: int | None = None) -> dict[str, object]:
    """A create body, pinned or not. `version=None` omits the field entirely.

    Omitted rather than sent as `null`, because those are two different requests and the
    one a caller with no opinion actually sends is the one that leaves the key out.
    """
    body: dict[str, object] = {
        "definition_id": definition_id,
        # Any id resolves through this file's environment store: which sandbox shape a
        # Session runs in is graded in `test_environment_reference.py`, and pinning one
        # here would read as though it mattered to the version claims below.
        "environment_id": str(uuid4()),
        "budget_minor_units": 5_000,
        "budget_currency": "USD",
        "retention_days": 30,
    }
    if version is not None:
        body["definition_version"] = version
    return body


# Fixed rather than `now()`, so a test comparing two reads of one agent compares
# values instead of instants. Which moment it is does not matter to any case here;
# that it is the same moment every time is what lets `created_at` be asserted at all.
_REGISTERED_AT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
_RETIRED_AT = datetime(2026, 8, 24, 9, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _Row:
    tenant_id: TenantId
    revision: int
    body: AgentDefinition


@dataclass(frozen=True, slots=True)
class _Resolved:
    definition: AgentDefinition
    revision: int


class FakeDefinitions:
    """`agent_definition` and its archive in memory, holding the tables' own rules.

    Four rules, and they are the ones the schema enforces rather than a convenient
    subset: a revision is never rewritten, a revision belongs to one tenant, a version
    is retired at most once, and an agent is retired at most once with the first
    moment standing. A fake that let any of them go would certify a route the real
    store refuses.
    """

    def __init__(self) -> None:
        self._rows: dict[DefinitionId, list[_Row]] = {}
        self._archived: set[tuple[DefinitionId, int]] = set()
        # Whole-agent retirement, which is a different table from `_archived` above for
        # the same reason the schema separates them: that one retires a revision so no
        # new Session resolves to it while others stay live, and this one retires the
        # agent so no revision is resolvable and none may be added.
        self._retired: dict[DefinitionId, datetime] = {}

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        rows = self._rows.setdefault(definition_id, [])
        revision = len(rows) + 1
        rows.append(_Row(tenant_id, revision, definition))
        return revision

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        mine = [
            row
            for row in self._rows.get(definition_id, [])
            if row.tenant_id == tenant_id
        ]
        if not mine:
            raise UnknownDefinition(str(definition_id))
        newest = max(mine, key=lambda row: row.revision)
        return _Resolved(newest.body, newest.revision)

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        return tuple(
            VersionFact(
                revision=row.revision,
                archived=(definition_id, row.revision) in self._archived,
            )
            for row in self._rows.get(definition_id, [])
            if row.tenant_id == tenant_id
        )

    async def register_at_revision(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
        expected: int,
    ) -> int | None:
        """Append only while `expected` is still the newest revision number.

        The conditional half of `register` above, holding the rule the real store holds
        in one SQL statement: compare and append with no gap between them. Here the gap
        is closed by there being no await between the read and the write, which is the
        same guarantee for the same reason -- a caller working from a stale number must
        lose rather than overwrite.

        Compares NUMBERS and not bodies, so a caller re-sending exactly what is stored
        is still refused when its number is stale. A fake that compared bodies would
        pass the route's happy path and quietly certify the wrong contract.
        """
        mine = [
            row
            for row in self._rows.get(definition_id, [])
            if row.tenant_id == tenant_id
        ]
        newest = max((row.revision for row in mine), default=None)
        if newest != expected:
            return None
        return await self.register(definition_id, tenant_id, definition)

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
        """Raises, because no test in this file lists agents.

        Present so `agent_lifecycle_of` can narrow this fake at all -- its check sees
        method names -- and raising rather than returning `()` so a test that started
        listing says so instead of quietly asserting against an empty page.
        """
        raise AssertionError("a test in this file paged agents; it should not")

    async def read_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> AgentRecord | None:
        """The agent as a whole, folded from the revisions this fake holds.

        Answers for a retired agent as well as a live one, with `archived_at` set --
        which is the behaviour under test in `test_a_retired_agent_takes_no_further
        _version`. A fake that returned None for a retired agent would turn the
        retirement refusal into a not-found and the case would pass for the wrong
        reason.
        """
        mine = [
            row
            for row in self._rows.get(definition_id, [])
            if row.tenant_id == tenant_id
        ]
        if not mine:
            return None
        newest = max(mine, key=lambda row: row.revision)
        return AgentRecord(
            definition_id=definition_id,
            version=newest.revision,
            created_at=_REGISTERED_AT,
            archived_at=self._retired.get(definition_id),
            definition=newest.body,
        )

    async def archive_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> datetime | None:
        """Retire the whole agent, idempotently, returning the ORIGINAL moment.

        `setdefault` rather than an assignment, so a second archive answers the first
        one's timestamp -- the same thing the real table's primary key does by making
        the repeat insert a conflict the route absorbs. An assignment here would let a
        retry move the moment the agent stopped being usable.
        """
        mine = [
            row
            for row in self._rows.get(definition_id, [])
            if row.tenant_id == tenant_id
        ]
        if not mine:
            return None
        return self._retired.setdefault(definition_id, _RETIRED_AT)

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        for row in self._rows.get(definition_id, []):
            if row.revision == revision and row.tenant_id == tenant_id:
                return row.body
        return None

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        owned = any(
            row.revision == revision and row.tenant_id == tenant_id
            for row in self._rows.get(definition_id, [])
        )
        if not owned or (definition_id, revision) in self._archived:
            return False
        self._archived.add((definition_id, revision))
        return True

    # `POST /v1/agents` consults the CI skill-eval gate through this same port before it
    # writes, so a registry that cannot answer is refused outright rather than allowed
    # to pin an ungraded revision. No test here submits an eval run, so every repository
    # is genuinely unenrolled; the two reads that would mean something raise instead of
    # returning a quiet default.

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        raise AssertionError("a version-pin test read a skill eval baseline")

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        raise AssertionError("a version-pin test recorded a skill eval run")

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        return EvalFacts(repository_enrolled=False, revision_accepted=False)


@dataclass(frozen=True, slots=True)
class _Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class FakeLog:
    def __init__(self) -> None:
        self._rows: dict[SessionId, list[_Event]] = {}

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        return self.add(session_id, type_, payload)

    def add(self, session_id: SessionId, type_: str, payload: dict[str, object]) -> Seq:
        rows = self._rows.setdefault(session_id, [])
        seq = Seq(len(rows) + 1)
        rows.append(_Event(session_id, seq, type_, dict(payload)))
        return seq

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[_Event]:
        span = [
            row for row in self._rows.get(session_id, []) if start <= row.seq <= end
        ]
        return span[:limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[_Event]:
        for row in self._rows.get(session_id, []):
            if row.seq > after:
                yield row

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ

    def events(self, session_id: SessionId) -> list[_Event]:
        return list(self._rows.get(session_id, []))

    def sessions_written(self) -> int:
        """How many Sessions have any event at all.

        A refused create must leave this unchanged. Checking the refused id's own log
        would pass trivially -- the caller never learns an id for a Session that was
        not created -- so the count over the whole log is what actually asserts it.
        """
        return len(self._rows)


class InMemorySessionRegistry:
    def __init__(self) -> None:
        self.records: dict[SessionId, SessionRecord] = {}

    async def create(self, record: SessionRecord) -> None:
        self.records[record.id] = record

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        found = self.records.get(session_id)
        if found is None or found.tenant_id != tenant_id:
            raise SessionNotVisible(str(session_id))
        return found

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("a test in this file listed Sessions")


class UnusedToolRegistry:
    """Satisfies the tool-registry port and is never called.

    Raising rather than returning a harmless value: a test here that reached the tool
    registry would be grading something this file does not grade, and a quiet stub
    would let it pass while doing so.
    """

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("a test in this file registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("a test in this file looked up a registered tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("a test in this file listed a tenant's tools")


@dataclass(frozen=True, slots=True)
class Harness:
    definitions: FakeDefinitions
    log: FakeLog
    tenant: TenantId

    def platform(self) -> Platform:
        return Platform(
            event_log_append=self.log,
            event_log_range=self.log,
            definition_registry=self.definitions,
            tool_registry=UnusedToolRegistry(),
            session_registry=InMemorySessionRegistry(),
            webhooks=UnusedWebhooks(),
            environment_store=AnyEnvironmentResolves(),
            turn_dispatch=NoPodTransport(),
            file_store=unconfigured_file_store(),
        )

    def caller(self, tenant: TenantId | None = None) -> AsyncClient:
        """One caller, one tenant, sent in the header the routes actually read.

        The tenant travels as a default header rather than on `request.state`, because
        `unauthenticated_tenant_from_header` is the one place a route in this package
        learns who is calling: a test that supplied the tenant any other way would pass
        against a route that never asks for one.
        """
        return AsyncClient(
            transport=ASGITransport(app=create_app(self.platform())),
            base_url="http://tenant",
            headers={TENANT_HEADER: str(tenant or self.tenant)},
        )


@pytest.fixture
def harness() -> Harness:
    return Harness(FakeDefinitions(), FakeLog(), TenantId(uuid4()))


# --- the fake, held honest before anything is graded through it -----------------


async def test_the_fake_numbers_revisions_per_id_and_keeps_bodies(
    harness: Harness,
) -> None:
    """Register twice on one id and once on another; nothing is rewritten."""
    store, tenant = harness.definitions, harness.tenant
    one, two = DefinitionId(uuid4()), DefinitionId(uuid4())
    first = AgentDefinition.model_validate(a_definition("review the papers"))
    second = AgentDefinition.model_validate(a_definition("review, and rank them"))

    assert await store.register(one, tenant, first) == 1
    assert await store.register(one, tenant, second) == 2
    assert await store.register(two, tenant, first) == 1

    assert [fact.revision for fact in await store.list_versions(one, tenant)] == [1, 2]
    kept = await store.read_version(one, tenant, 1)
    assert kept is not None and kept.instructions == "review the papers"


async def test_the_fake_hides_and_refuses_another_tenants_rows(
    harness: Harness,
) -> None:
    store, tenant = harness.definitions, harness.tenant
    stranger = TenantId(uuid4())
    agent = DefinitionId(uuid4())
    await store.register(
        agent, tenant, AgentDefinition.model_validate(a_definition("mine"))
    )

    assert await store.list_versions(agent, stranger) == ()
    assert await store.read_version(agent, stranger, 1) is None
    assert await store.archive_version(agent, stranger, 1) is False
    assert await store.list_versions(agent, tenant) != ()


async def test_the_fake_retires_a_version_at_most_once(harness: Harness) -> None:
    store, tenant = harness.definitions, harness.tenant
    agent = DefinitionId(uuid4())
    await store.register(
        agent, tenant, AgentDefinition.model_validate(a_definition("mine"))
    )

    assert await store.archive_version(agent, tenant, 1) is True
    assert await store.archive_version(agent, tenant, 1) is False
    assert await store.list_versions(agent, tenant) == (
        VersionFact(revision=1, archived=True),
    )


# --- choosing a revision, as a pure function ------------------------------------


def _reference(version: int | None) -> AgentReference:
    return AgentReference(DefinitionId(uuid4()), version)


def test_an_unpinned_reference_takes_the_newest_and_a_pin_takes_its_own() -> None:
    facts = (VersionFact(1, archived=False), VersionFact(2, archived=False))

    assert choose_revision(facts, _reference(None)) == 2
    assert choose_revision(facts, _reference(1)) == 1


def test_a_pin_on_a_retired_revision_refuses_rather_than_falling_back() -> None:
    """The whole point of a pin: it is never quietly moved to another revision.

    A fallback to revision 1 here would be invisible -- the Session would be created,
    would run, and would run different instructions than the caller named.
    """
    facts = (VersionFact(1, archived=False), VersionFact(2, archived=True))

    assert choose_revision(facts, _reference(None)) == 1
    with pytest.raises(AgentVersionArchived) as refused:
        choose_revision(facts, _reference(2))
    assert refused.value.revision == 2


def test_an_agent_with_every_revision_retired_starts_nothing_even_unpinned() -> None:
    """Retiring the last live revision closes the agent, rather than being ignored."""
    facts = (VersionFact(1, archived=True), VersionFact(2, archived=True))

    with pytest.raises(AgentVersionArchived) as refused:
        choose_revision(facts, _reference(None))
    assert refused.value.revision == 2


def test_a_pin_on_a_revision_that_was_never_registered_is_unknown() -> None:
    facts = (VersionFact(1, archived=False), VersionFact(2, archived=False))

    with pytest.raises(UnknownAgentVersion):
        choose_revision(facts, _reference(3))


def test_an_agent_with_no_registered_version_is_unknown() -> None:
    with pytest.raises(UnknownAgentVersion):
        choose_revision((), _reference(None))


async def test_resolve_reference_returns_the_body_of_the_revision_it_chose(
    harness: Harness,
) -> None:
    store, tenant = harness.definitions, harness.tenant
    agent = DefinitionId(uuid4())
    await store.register(
        agent, tenant, AgentDefinition.model_validate(a_definition("the first"))
    )
    await store.register(
        agent, tenant, AgentDefinition.model_validate(a_definition("the second"))
    )

    newest = await resolve_reference(store, tenant, AgentReference(agent, None))
    pinned = await resolve_reference(store, tenant, AgentReference(agent, 1))

    assert (newest.revision, newest.definition.instructions) == (2, "the second")
    assert (pinned.revision, pinned.definition.instructions) == (1, "the first")


async def test_a_body_that_vanished_between_the_two_reads_is_unknown_not_invented(
    harness: Harness,
) -> None:
    """`agent_definition` has no delete path, so this combination means a lost row.

    Defaulting here would hide it behind a Session that runs the wrong instructions.
    """

    class LosesTheBody(FakeDefinitions):
        async def read_version(
            self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
        ) -> AgentDefinition | None:
            return None

    store = LosesTheBody()
    agent, tenant = DefinitionId(uuid4()), harness.tenant
    await store.register(
        agent, tenant, AgentDefinition.model_validate(a_definition("gone"))
    )

    with pytest.raises(UnknownAgentVersion):
        await resolve_reference(store, tenant, AgentReference(agent, 1))


# --- the routes -----------------------------------------------------------------


async def test_adding_a_version_leaves_the_earlier_one_byte_identical(
    harness: Harness,
) -> None:
    async with harness.caller() as caller:
        registered = await caller.post("/v1/agents", json=a_definition("review them"))
        agent_id = registered.json()["id"]
        added = await caller.post(
            f"/v1/agents/{agent_id}/versions",
            json=a_definition("review them, and rank them"),
        )
        listed = await caller.get(f"/v1/agents/{agent_id}/versions")

    assert added.status_code == 201, added.text
    assert added.json() == {"id": agent_id, "version": 2}
    assert listed.json()["versions"] == [
        {"version": 1, "archived": False},
        {"version": 2, "archived": False},
    ]
    kept = await harness.definitions.read_version(
        DefinitionId(UUID(agent_id)), harness.tenant, 1
    )
    assert kept is not None and kept.instructions == "review them"


async def test_adding_a_version_to_an_id_nobody_registered_writes_nothing(
    harness: Harness,
) -> None:
    """404 rather than a brand-new agent at revision 1.

    The insert numbers a revision from whatever rows carry the id, so without the
    existence check this call would silently *create* an agent under a path that reads
    as an edit -- and answer 201 while doing it.
    """
    unknown = DefinitionId(uuid4())

    async with harness.caller() as caller:
        refused = await caller.post(
            f"/v1/agents/{unknown}/versions", json=a_definition("out of nowhere")
        )

    assert refused.status_code == 404, refused.text
    assert refused.json()["error"]["code"] == "definition.not_found"
    assert await harness.definitions.list_versions(unknown, harness.tenant) == ()


async def test_a_retired_agent_takes_no_further_version(harness: Harness) -> None:
    """409 `agent.archived`, and the revision count does not move.

    This is what makes archive terminal rather than cosmetic. Retiring an agent and
    then appending a version to it would leave the agent live in every way that
    matters -- a new Session resolves to the newest revision, and the newest revision
    would be one written after the retirement.

    The refusal is a 409 and not a 404 because the caller's id is correct and the agent
    is deliberately unusable; a 404 would send them to check an id they got right.
    """
    async with harness.caller() as caller:
        agent_id = (await caller.post("/v1/agents", json=a_definition("mine"))).json()[
            "id"
        ]
        retired = await harness.definitions.archive_agent(
            DefinitionId(UUID(agent_id)), harness.tenant
        )
        refused = await caller.post(
            f"/v1/agents/{agent_id}/versions", json=a_definition("one more")
        )

    assert retired is not None
    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == "agent.archived"
    assert await harness.definitions.list_versions(
        DefinitionId(UUID(agent_id)), harness.tenant
    ) == (VersionFact(revision=1, archived=False),)


async def test_a_live_agent_still_takes_a_version_after_the_retirement_check(
    harness: Harness,
) -> None:
    """The other arm, without which the case above passes for a refusal of everything.

    A check that refused every append would satisfy the retirement test and break the
    feature, and nothing in that test could tell. This is the pair that makes it
    evidence.
    """
    async with harness.caller() as caller:
        agent_id = (await caller.post("/v1/agents", json=a_definition("mine"))).json()[
            "id"
        ]
        appended = await caller.post(
            f"/v1/agents/{agent_id}/versions", json=a_definition("one more")
        )

    assert appended.status_code == 201, appended.text
    assert await harness.definitions.list_versions(
        DefinitionId(UUID(agent_id)), harness.tenant
    ) == (
        VersionFact(revision=1, archived=False),
        VersionFact(revision=2, archived=False),
    )


async def test_another_tenants_agent_cannot_be_edited_or_listed(
    harness: Harness,
) -> None:
    stranger = TenantId(uuid4())

    async with harness.caller() as owner:
        agent_id = (await owner.post("/v1/agents", json=a_definition("mine"))).json()[
            "id"
        ]
    async with harness.caller(stranger) as intruder:
        edited = await intruder.post(
            f"/v1/agents/{agent_id}/versions", json=a_definition("yours now")
        )
        listed = await intruder.get(f"/v1/agents/{agent_id}/versions")
        retired = await intruder.post(f"/v1/agents/{agent_id}/versions/1/archive")

    assert [edited.status_code, listed.status_code, retired.status_code] == [
        404,
        404,
        404,
    ]
    assert await harness.definitions.list_versions(
        DefinitionId(UUID(agent_id)), harness.tenant
    ) == (VersionFact(revision=1, archived=False),)


async def test_retiring_is_idempotent_and_shows_up_in_the_listing(
    harness: Harness,
) -> None:
    async with harness.caller() as caller:
        agent_id = (await caller.post("/v1/agents", json=a_definition("first"))).json()[
            "id"
        ]
        await caller.post(
            f"/v1/agents/{agent_id}/versions", json=a_definition("second")
        )
        first = await caller.post(f"/v1/agents/{agent_id}/versions/2/archive")
        again = await caller.post(f"/v1/agents/{agent_id}/versions/2/archive")
        listed = await caller.get(f"/v1/agents/{agent_id}/versions")

    assert first.json() == {"id": agent_id, "version": 2, "newly_archived": True}
    assert again.json() == {"id": agent_id, "version": 2, "newly_archived": False}
    assert listed.json()["versions"] == [
        {"version": 1, "archived": False},
        {"version": 2, "archived": True},
    ]


async def test_retiring_a_version_that_does_not_exist_or_cannot_exist(
    harness: Harness,
) -> None:
    """404 for a number no revision has; 422 for a number no revision could have.

    The second is the path constraint, so it never reaches the handler -- which is why
    it is a different status from the first rather than the same refusal twice.
    """
    async with harness.caller() as caller:
        agent_id = (await caller.post("/v1/agents", json=a_definition("first"))).json()[
            "id"
        ]
        missing = await caller.post(f"/v1/agents/{agent_id}/versions/9/archive")
        impossible = await caller.post(f"/v1/agents/{agent_id}/versions/0/archive")

    assert missing.status_code == 404, missing.text
    assert missing.json()["error"]["code"] == "definition.not_found"
    assert impossible.status_code == 400, impossible.text


async def test_no_version_route_response_carries_a_runtime_identifier(
    harness: Harness,
) -> None:
    """Nothing the Agent Runtime names itself with reaches a tenant (ADR-007).

    Nor does the skills repository, which is a credentialed remote in production and
    has no business in a version listing.
    """
    async with harness.caller() as caller:
        agent_id = (await caller.post("/v1/agents", json=a_definition("first"))).json()[
            "id"
        ]
        bodies = [
            (
                await caller.post(
                    f"/v1/agents/{agent_id}/versions", json=a_definition("second")
                )
            ).text,
            (await caller.get(f"/v1/agents/{agent_id}/versions")).text,
            (await caller.post(f"/v1/agents/{agent_id}/versions/2/archive")).text,
        ]

    for rendered in bodies:
        assert "thread" not in rendered.lower()
        assert SKILLS_REPO not in rendered
        assert PUBLISHED not in rendered


# --- the two scenarios, end to end ----------------------------------------------


async def test_an_edit_makes_version_two_and_a_pin_keeps_reading_version_one(
    harness: Harness,
) -> None:
    """A Session pinned to version 1 is untouched by the edit that made version 2.

    The pinned Session is created *before* the edit and the unpinned one *after*, which
    is the ordering the claim is about: one revision was current at each moment, and
    each Session recorded the one that was current for it.
    """
    async with harness.caller() as caller:
        registered = await caller.post("/v1/agents", json=a_definition("review them"))
        agent_id = registered.json()["id"]
        pinned = await caller.post("/v1/sessions", json=a_session(agent_id, version=1))
        added = await caller.post(
            f"/v1/agents/{agent_id}/versions",
            json=a_definition("review them, and rank them"),
        )
        unpinned = await caller.post("/v1/sessions", json=a_session(agent_id))

    assert added.json()["version"] == 2
    assert [pinned.status_code, unpinned.status_code] == [201, 201]

    recorded = {
        which: harness.log.events(SessionId(UUID(response.json()["id"])))[0].payload[
            "definition_revision"
        ]
        for which, response in (("pinned", pinned), ("unpinned", unpinned))
    }
    assert recorded == {"pinned": 1, "unpinned": 2}

    still = await resolve_reference(
        harness.definitions,
        harness.tenant,
        AgentReference(DefinitionId(UUID(agent_id)), 1),
    )
    assert still.definition.instructions == "review them"


async def test_an_archived_version_starts_no_session_and_stops_no_running_one(
    harness: Harness,
) -> None:
    """The retirement bites at creation and nowhere else.

    A Session that already resolved revision 1 never asks again, so its log is byte
    identical across the retirement and it still reads back as running -- while a new
    Session naming that revision is refused with the code that says *withdrawn* rather
    than the one that says *never here*.
    """
    async with harness.caller() as caller:
        registered = await caller.post("/v1/agents", json=a_definition("review them"))
        agent_id = registered.json()["id"]
        running = await caller.post("/v1/sessions", json=a_session(agent_id))
        running_id = SessionId(UUID(running.json()["id"]))
        before = harness.log.events(running_id)

        retired = await caller.post(f"/v1/agents/{agent_id}/versions/1/archive")
        refused_pinned = await caller.post(
            "/v1/sessions", json=a_session(agent_id, version=1)
        )
        refused_unpinned = await caller.post("/v1/sessions", json=a_session(agent_id))
        readable = await caller.get(f"/v1/sessions/{running_id}")

    assert retired.json()["newly_archived"] is True
    for refused in (refused_pinned, refused_unpinned):
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["code"] == "definition.version_archived"

    assert harness.log.events(running_id) == before
    assert readable.status_code == 200
    assert readable.json()["state"] == "idle"
    assert (
        await harness.definitions.read_version(
            DefinitionId(UUID(agent_id)), harness.tenant, 1
        )
    ) is not None


async def test_retiring_the_newest_version_sends_an_unpinned_create_to_the_one_below(
    harness: Harness,
) -> None:
    """Withdrawing a bad edit is the reason to retire, so the withdrawal has to take."""
    async with harness.caller() as caller:
        agent_id = (
            await caller.post("/v1/agents", json=a_definition("the good one"))
        ).json()["id"]
        await caller.post(
            f"/v1/agents/{agent_id}/versions", json=a_definition("the bad edit")
        )
        await caller.post(f"/v1/agents/{agent_id}/versions/2/archive")
        created = await caller.post("/v1/sessions", json=a_session(agent_id))

    assert created.status_code == 201, created.text
    payload = harness.log.events(SessionId(UUID(created.json()["id"])))[0].payload
    assert payload["definition_revision"] == 1


async def test_a_session_pinned_to_a_version_that_was_never_registered_is_refused(
    harness: Harness,
) -> None:
    async with harness.caller() as caller:
        agent_id = (
            await caller.post("/v1/agents", json=a_definition("only one"))
        ).json()["id"]
        refused = await caller.post("/v1/sessions", json=a_session(agent_id, version=9))

    assert refused.status_code == 404, refused.text
    assert refused.json()["error"]["code"] == "definition.not_found"
    assert harness.log.sessions_written() == 0, (
        "a refused create appended to the log; the refusal has to come before the "
        "append or a Session half-exists"
    )


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
