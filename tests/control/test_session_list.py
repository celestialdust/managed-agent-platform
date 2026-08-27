"""GET /v1/sessions: the cursor, the page, and the tenant the page belongs to.

Two tiers in one file, deliberately. The cursor and the route's own decisions -- how big
a page is, when `next_page` is null, what a bad cursor gets -- are graded against an
in-memory registry, because they are properties of this module and a database would only
slow them down. The walk at the end is tier 1 against real PostgreSQL through the real
adapter, because "seven Sessions come back exactly once, newest first" is a property of
the index and the keyset comparison as much as of the route, and an in-memory fake sorts
a list in Python and can never fail the way the store fails.

The in-memory registry refuses the same `limit` window the real adapter refuses. A fake
that served any request would certify a route the store would reject, which is the
failure mode a fake exists to avoid rather than to introduce.
"""

from __future__ import annotations

import uuid
from base64 import urlsafe_b64encode
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.session_list import (
    CURSOR_INVALID,
    MAX_PAGE_SIZE,
    Cursor,
    InvalidCursor,
)
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import (
    FIRST_SEQ,
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
)
from managed_agent.core.ports import EventRecord, Resolution, SessionListing
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord

_SKILLS_SHA = "0" * 39 + "a"

# Not 1, not 30, not 500. Every value a listing carries is asserted below, and a
# hardcoded default in the route would be indistinguishable from a read of the store if
# the fixtures used the values everything else uses.
_REVISION = "4"
_BUDGET = 731
_RETENTION = 17
_BASE_MS = 1_700_000_000_000

# Every walk in this file is bounded, and the bound is not defensive padding. A keyset
# boundary written inclusively, or a route that never stops handing out a cursor, makes
# a walk that reads pages forever -- and an unbounded loop turns that into a test that
# *hangs* rather than one that fails. Measured: an inclusive comparison in the adapter
# ran past ten minutes before this bound existed anywhere. No test here needs more than
# a handful of pages, so twenty is far above any honest walk and far below forever.
_PAGE_BUDGET = 20
_NEVER_ENDED = (
    f"the walk read {_PAGE_BUDGET} pages without a null next_page; it is either "
    "repeating rows at the page boundary or never reporting the end"
)


@dataclass(frozen=True, slots=True)
class Listing:
    id: SessionId
    definition_id: DefinitionId
    definition_revision: str
    created_at_ms: int


class FakeSessionRegistry:
    """Rows with chosen creation times, paged by the keyset the real registry uses.

    Times are chosen rather than clocked so a test can put two Sessions in one
    millisecond on purpose. It also counts its `page` calls, which is how a test asserts
    that a refused request reached no store at all -- a route that refused *after*
    querying would look identical from the response alone.
    """

    def __init__(self) -> None:
        self.rows: list[Listing] = []
        self.owner: dict[SessionId, TenantId] = {}
        self.page_calls = 0

    def add(self, tenant_id: TenantId, created_at_ms: int) -> Listing:
        listing = Listing(
            id=SessionId(uuid.uuid4()),
            definition_id=DefinitionId(uuid.uuid4()),
            definition_revision=_REVISION,
            created_at_ms=created_at_ms,
        )
        self.rows.append(listing)
        self.owner[listing.id] = tenant_id
        return listing

    async def create(self, record: SessionRecord) -> None:
        raise AssertionError("a test in this file created a Session through the route")

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        raise AssertionError("a test in this file read a single Session")

    async def page(
        self, tenant_id: TenantId, after: tuple[int, uuid.UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        self.page_calls += 1
        if limit < 1 or limit > 500:
            raise ValueError(f"page limit {limit} is outside 1..500")
        mine = sorted(
            (row for row in self.rows if self.owner[row.id] == tenant_id),
            key=lambda row: (row.created_at_ms, row.id),
            reverse=True,
        )
        if after is not None:
            mine = [row for row in mine if (row.created_at_ms, row.id) < after]
        return mine[:limit]


class UnusedLog:
    """Satisfies both log ports and is never called: listing folds no log."""

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        raise AssertionError("listing a tenant's Sessions appended to a log")

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]:
        raise AssertionError("listing a tenant's Sessions read a log")

    async def follow(
        self, session_id: SessionId, after: Seq
    ) -> AsyncIterator[EventRecord]:
        raise AssertionError("listing a tenant's Sessions followed a log")
        yield  # pragma: no cover - unreachable, and what makes this a generator

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ


class UnusedDefinitionRegistry:
    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        raise AssertionError("listing a tenant's Sessions registered a definition")

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        raise AssertionError("listing a tenant's Sessions resolved a definition")

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        raise AssertionError("listing a tenant's Sessions read a definition's versions")

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        raise AssertionError("listing a tenant's Sessions read a definition revision")

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("listing a tenant's Sessions retired a definition version")


class UnusedToolRegistry:
    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("listing a tenant's Sessions registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("listing a tenant's Sessions looked up a tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("listing a tenant's Sessions listed a tenant's tools")


@pytest.fixture
def registry() -> FakeSessionRegistry:
    return FakeSessionRegistry()


@pytest.fixture
def tenant() -> TenantId:
    return TenantId(uuid.uuid4())


@pytest.fixture
def client(registry: FakeSessionRegistry, tenant: TenantId) -> TestClient:
    unused = UnusedLog()
    return TestClient(
        create_app(
            Platform(
                event_log_append=unused,
                event_log_range=unused,
                definition_registry=UnusedDefinitionRegistry(),
                tool_registry=UnusedToolRegistry(),
                session_registry=registry,
                webhooks=UnusedWebhooks(),
                environment_store=UnusedEnvironmentStore(),
                turn_dispatch=NoPodTransport(),
                file_store=unconfigured_file_store(),
            )
        ),
        headers={TENANT_HEADER: str(tenant)},
    )


# --- the cursor ------------------------------------------------------------------


def test_a_cursor_round_trips_to_an_equal_position() -> None:
    position = Cursor(_BASE_MS + 7, SessionId(uuid.uuid4()))

    assert Cursor.decode(position.encode()) == position


def test_two_sessions_in_one_millisecond_get_two_different_cursors() -> None:
    """The id is in the token, which is the whole reason the token is not a timestamp.

    A cursor naming only the millisecond cannot say which of two Sessions the caller
    already holds, so the next page would either repeat one or skip both.
    """
    first = Cursor(_BASE_MS, SessionId(uuid.uuid4()))
    second = Cursor(_BASE_MS, SessionId(uuid.uuid4()))

    assert first.encode() != second.encode()
    assert Cursor.decode(first.encode()) != Cursor.decode(second.encode())


def test_a_cursor_does_not_read_as_its_contents() -> None:
    """Opaque in fact, not only by intention.

    What the token encodes is the store's ordering key, and a caller that could read it
    would start depending on that ordering -- which is the coupling an opaque token
    exists to prevent. Asserted rather than asserted-about: a token that happened to
    contain the plain timestamp would satisfy every other test in this file.
    """
    session_id = SessionId(uuid.uuid4())
    token = Cursor(_BASE_MS, session_id).encode()

    assert str(_BASE_MS) not in token
    assert str(session_id) not in token
    assert "=" not in token, "a padded token is percent-encoded in a query string"


@pytest.mark.parametrize(
    ("token", "why"),
    [
        ("", "empty"),
        ("!!!!", "not base64"),
        (urlsafe_b64encode(b"\xff\xfe").decode().rstrip("="), "not utf-8"),
        (urlsafe_b64encode(b"1700000000000").decode().rstrip("="), "no separator"),
        (urlsafe_b64encode(b"1700000000000.not-a-uuid").decode().rstrip("="), "bad id"),
        (urlsafe_b64encode(b"later.0" * 4).decode().rstrip("="), "bad millisecond"),
    ],
)
def test_anything_this_surface_did_not_issue_is_refused(token: str, why: str) -> None:
    with pytest.raises(InvalidCursor):
        Cursor.decode(token)


# --- the route -------------------------------------------------------------------


def test_a_tenant_with_no_sessions_gets_an_empty_page_rather_than_a_refusal(
    client: TestClient,
) -> None:
    """Having no Sessions is a normal thing to have, and it is not an error."""
    response = client.get("/v1/sessions")

    assert response.status_code == 200
    assert response.json() == {"sessions": [], "next_page": None}


def test_a_page_carries_the_facts_a_caller_lists_by(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """Every field is read from the store, and none of them is a default.

    The revision is "4" and the creation time is not now, so a route that filled either
    in from a constant would be visible here.
    """
    written = registry.add(tenant, _BASE_MS + 3)

    (listed,) = client.get("/v1/sessions").json()["sessions"]

    assert listed == {
        "id": str(written.id),
        "definition_id": str(written.definition_id),
        "definition_revision": _REVISION,
        "created_at_ms": _BASE_MS + 3,
    }


def test_seven_sessions_at_a_limit_of_three_page_three_three_one(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """The walk the checkpoint describes, and the null cursor that ends it.

    Seven and three so the final page is short. `next_page` is null on that page and
    on no earlier one, which is what stops a caller on the page that has the last row
    rather than on a wasted round trip after it.
    """
    written = [registry.add(tenant, _BASE_MS + step) for step in range(7)]

    sizes: list[int] = []
    seen: list[str] = []
    cursors: list[str | None] = []
    cursor: str | None = None
    for _ in range(_PAGE_BUDGET):
        query = f"/v1/sessions?limit=3{'' if cursor is None else f'&page={cursor}'}"
        body = client.get(query).json()
        sizes.append(len(body["sessions"]))
        seen.extend(row["id"] for row in body["sessions"])
        cursors.append(body["next_page"])
        cursor = body["next_page"]
        if cursor is None:
            break
    else:
        raise AssertionError(_NEVER_ENDED)

    assert sizes == [3, 3, 1]
    assert cursors[-1] is None
    assert all(token is not None for token in cursors[:-1])
    assert seen == [str(row.id) for row in reversed(written)], (
        "the walk did not return every Session exactly once, newest first"
    )


def test_a_full_final_page_still_ends_without_a_cursor(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """Six rows at a limit of three: the second page is full and is still the last.

    This is what the extra row the route asks for buys. Without it the route would have
    to guess from the page being full, hand back a cursor, and send the caller after an
    empty page -- and a caller that trusted the cursor would report one more page than
    exists.
    """
    for step in range(6):
        registry.add(tenant, _BASE_MS + step)

    first = client.get("/v1/sessions?limit=3").json()
    second = client.get(f"/v1/sessions?limit=3&page={first['next_page']}").json()

    assert len(second["sessions"]) == 3
    assert second["next_page"] is None


def test_two_sessions_sharing_a_millisecond_cross_a_page_boundary_intact(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """The boundary falls between two equal timestamps, on purpose.

    Ordering on the millisecond alone leaves their relative order undefined, so the
    second page would either repeat the one already handed out or skip past both. The id
    in the cursor is what decides it.
    """
    oldest = registry.add(tenant, _BASE_MS)
    tied_low = registry.add(tenant, _BASE_MS + 1)
    tied_high = registry.add(tenant, _BASE_MS + 1)
    newest = registry.add(tenant, _BASE_MS + 2)

    walked: list[str] = []
    cursor: str | None = None
    for _ in range(_PAGE_BUDGET):
        query = f"/v1/sessions?limit=2{'' if cursor is None else f'&page={cursor}'}"
        body = client.get(query).json()
        walked.extend(row["id"] for row in body["sessions"])
        cursor = body["next_page"]
        if cursor is None:
            break
    else:
        raise AssertionError(_NEVER_ENDED)

    assert len(walked) == len(set(walked)) == 4, walked
    assert walked[0] == str(newest.id)
    assert walked[3] == str(oldest.id)
    assert set(walked[1:3]) == {str(tied_low.id), str(tied_high.id)}


def test_one_tenants_page_holds_none_of_anothers(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """MAP-A11: absent rather than present-and-redacted.

    Both tenants hold Sessions and their creation times interleave, so a page that
    filtered late -- or not at all -- would show the other tenant's rows here rather
    than merely mis-ordering its own.
    """
    stranger = TenantId(uuid.uuid4())
    mine = [registry.add(tenant, _BASE_MS + step) for step in (0, 2, 4)]
    theirs = [registry.add(stranger, _BASE_MS + step) for step in (1, 3, 5)]

    listed = [row["id"] for row in client.get("/v1/sessions").json()["sessions"]]

    assert sorted(listed) == sorted(str(row.id) for row in mine)
    assert not set(listed) & {str(row.id) for row in theirs}


def test_a_cursor_issued_for_one_tenant_does_not_read_across_to_another(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """A cursor is a position, not an authorization.

    It names a place in an ordering, so replaying another tenant's cursor is legitimate
    and must simply return this tenant's rows below that position. What must not happen
    is the other tenant's rows coming back because their cursor was presented.
    """
    stranger = TenantId(uuid.uuid4())
    theirs = [registry.add(stranger, _BASE_MS + step) for step in (10, 11, 12)]
    mine = [registry.add(tenant, _BASE_MS + step) for step in (0, 1, 2)]
    borrowed = Cursor(theirs[-1].created_at_ms, theirs[-1].id).encode()

    body = client.get(f"/v1/sessions?page={borrowed}").json()

    assert [row["id"] for row in body["sessions"]] == [
        str(row.id) for row in reversed(mine)
    ]


def test_a_cursor_this_surface_did_not_issue_is_refused(client: TestClient) -> None:
    response = client.get("/v1/sessions?page=not-a-cursor-we-issued")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == CURSOR_INVALID


def test_a_bad_cursor_is_refused_rather_than_read_as_the_start(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """Starting over on an unreadable cursor would hand back the newest page again.

    That reads to a caller as the walk having looped rather than failed, and a walk that
    loops does not terminate. Refusing is the only answer that tells the caller
    anything.
    """
    registry.add(tenant, _BASE_MS)

    assert client.get("/v1/sessions?page=%21%21%21%21").status_code == 400
    assert registry.page_calls == 0, (
        "the route queried the store before deciding the cursor was unreadable"
    )


@pytest.mark.parametrize("limit", [0, -1, MAX_PAGE_SIZE + 1, 1000])
def test_a_page_size_outside_the_published_window_is_refused_at_the_boundary(
    client: TestClient, registry: FakeSessionRegistry, limit: int
) -> None:
    """422 from the annotation, and the store is never asked.

    An unbounded page is a whole-collection read wearing a limit parameter, and the
    adapter refuses one too -- but a refusal that came from the adapter would reach the
    caller as a 500 rather than as a message naming the field.
    """
    response = client.get(f"/v1/sessions?limit={limit}")

    assert response.status_code == 400
    assert registry.page_calls == 0


def test_a_caller_that_names_no_limit_still_gets_a_bounded_page(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """The default bound sits strictly below the ceiling, and a cursor comes with it.

    The exact number is deliberately not pinned. `DEFAULT_PAGE_SIZE` is 25 because
    twenty-five is a reasonable page and nothing signed names a figure, so an assertion
    that it equals 25 would only restate the constant -- and one written as
    `== DEFAULT_PAGE_SIZE` restates it while looking like a measurement: that form
    survived a mutation from 25 to 100 with the fixture sized off the same constant.

    The bound is asserted as *strictly* below the ceiling, and that word is the whole
    test. Written as `<= MAX_PAGE_SIZE` it had a hole exactly one value wide, and the
    value in the hole is the one a careless edit reaches for: a default of 100 --
    quadruple the intended page, equal to the published ceiling -- passed this
    assertion and all 101 control tests. Measured, twice, independently. A default
    *above* the ceiling is caught, but not here: Pydantic validates the default against
    `le=MAX_PAGE_SIZE` and every request 422s, so this test fails for a reason that has
    nothing to do with paging. The mutation this test exists for is the one that stays
    inside the annotation, and until the comparison was tightened it saw none of them.

    So: a default of zero, a default at or above the ceiling, and a route that dropped
    the default and read the whole collection each fail here. The fixture holds more
    rows than the ceiling so that a full read is distinguishable from a bounded one.
    """
    for step in range(MAX_PAGE_SIZE + 20):
        registry.add(tenant, _BASE_MS + step)

    response = client.get("/v1/sessions")
    body = response.json()

    assert response.status_code == 200, response.text
    assert 1 <= len(body["sessions"]) < MAX_PAGE_SIZE, (
        f"an unasked caller got {len(body['sessions'])} rows against a published "
        f"ceiling of {MAX_PAGE_SIZE}. The default has to sit strictly below the "
        "ceiling: a default equal to it means an unasked caller is served the largest "
        "page the surface will ever produce, and `<=` here cannot tell that from the "
        "intended page."
    )
    assert body["next_page"] is not None


def test_a_request_with_no_tenant_is_refused_and_reaches_no_store(
    registry: FakeSessionRegistry,
) -> None:
    """An unscoped read of this collection is every tenant's Sessions at once.

    Refused before anything is read, which the call count is what proves: a route that
    queried first and refused afterwards would return the same status and would have
    already asked the store for a page with no tenant to key on.
    """
    unused = UnusedLog()
    client = TestClient(
        create_app(
            Platform(
                event_log_append=unused,
                event_log_range=unused,
                definition_registry=UnusedDefinitionRegistry(),
                tool_registry=UnusedToolRegistry(),
                session_registry=registry,
                webhooks=UnusedWebhooks(),
                environment_store=UnusedEnvironmentStore(),
                turn_dispatch=NoPodTransport(),
                file_store=unconfigured_file_store(),
            )
        )
    )

    response = client.get("/v1/sessions")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request.tenant_missing"
    assert registry.page_calls == 0


def test_listing_folds_no_event_log(
    client: TestClient, registry: FakeSessionRegistry, tenant: TenantId
) -> None:
    """A listing row carries no state, and this is that claim made falsifiable.

    The log ports raise on every method, so a route that folded a Session's log to put a
    state on the row would fail here rather than merely being slow. Twenty-five rows
    would be twenty-five folds on a read whose job is to help a caller find a Session.
    """
    for step in range(5):
        registry.add(tenant, _BASE_MS + step)

    body = client.get("/v1/sessions").json()

    assert len(body["sessions"]) == 5
    assert all("state" not in row for row in body["sessions"])


# --- tier 1: the same walk against the real store ---------------------------------


def _a_definition() -> dict[str, object]:
    return {
        "name": "list-fixture",
        "instructions": "irrelevant to these tests",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SKILLS_SHA,
    }


_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
"""A digest-pinned image, because a registered shape refuses anything else."""


async def _an_environment(client: AsyncClient, headers: dict[str, str]) -> str:
    """A registered sandbox shape, because creating a Session names one.

    Registered through the route rather than written into the table, so the id the
    create call names is one the platform actually issued.
    """
    registered = await client.post(
        "/v1/environments",
        json={"name": "listing-fixture", "runtime_image": _FIXTURE_IMAGE},
        headers=headers,
    )
    assert registered.status_code == 201, registered.text
    return str(registered.json()["id"])


def _create_body(definition_id: str, environment_id: str) -> dict[str, object]:
    return {
        "definition_id": definition_id,
        "environment_id": environment_id,
        "grant": [],
        "scope": {"repository": "acme/widgets"},
        "budget_minor_units": _BUDGET,
        "budget_currency": "EUR",
        "retention_days": _RETENTION,
    }


async def test_seven_real_sessions_walk_once_each_newest_first(
    database_url: str,
) -> None:
    """The checkpoint, end to end: create seven, walk them three at a time.

    Tier 1 (testcontainers, real PostgreSQL 17). The in-memory tests above grade the
    route's decisions; this grades the keyset comparison against the real index, which
    is where a page boundary actually repeats or skips a row. A Python `sorted()` over a
    list cannot fail that way, so a fake could never have caught it.

    Driven with `AsyncClient` over `ASGITransport` rather than `TestClient`, which
    runs the app in an event loop of its own -- an async engine's pooled connections
    belong to the loop that opened them, so sharing one across the two fails on the
    *second* request with `got Future attached to a different loop`, which reads as a
    database fault and is not one.

    The order is asserted as strictly decreasing `(created_at_ms, id)` rather than as
    "the reverse of the order they were created in", and the difference matters: these
    seven take their creation times from the server clock, so two of them landing in one
    millisecond is a real possibility on a fast machine and their relative order is then
    decided by id rather than by which call came first. Asserting creation order here
    would be a test that passes or fails on how quickly the machine ran. That order is
    graded where it can be graded honestly --
    `tests/adapters/test_session_registry.py`, against the same real database with the
    creation times written explicitly.
    """
    platform, engine = build(database_url)
    tenant = {TENANT_HEADER: str(uuid.uuid4())}
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app(platform)),
            base_url="http://tenant",
            headers=tenant,
        ) as client:
            registered = await client.post("/v1/agents", json=_a_definition())
            assert registered.status_code == 201, registered.text
            definition_id = registered.json()["id"]
            environment_id = await _an_environment(client, tenant)

            created: list[str] = []
            for _ in range(7):
                response = await client.post(
                    "/v1/sessions", json=_create_body(definition_id, environment_id)
                )
                assert response.status_code == 201, response.text
                created.append(response.json()["id"])

            sizes: list[int] = []
            walked: list[dict[str, object]] = []
            cursor: str | None = None
            for _ in range(_PAGE_BUDGET):
                query = "/v1/sessions?limit=3"
                if cursor is not None:
                    query = f"{query}&page={cursor}"
                body = (await client.get(query)).json()
                sizes.append(len(body["sessions"]))
                walked.extend(body["sessions"])
                cursor = body["next_page"]
                if cursor is None:
                    break
            else:
                raise AssertionError(_NEVER_ENDED)
    finally:
        await engine.dispose()

    assert sizes == [3, 3, 1]
    ids = [str(row["id"]) for row in walked]
    assert len(ids) == len(set(ids)) == 7, ids
    assert set(ids) == set(created), (
        "the walk did not return the seven Sessions that were created"
    )
    keys = [(int(str(row["created_at_ms"])), str(row["id"])) for row in walked]
    assert keys == sorted(keys, reverse=True), (
        f"the walk came back out of order: {keys}"
    )


async def test_a_real_tenants_list_holds_none_of_anothers(database_url: str) -> None:
    """MAP-A11 against the real store: the SQL filters, not only the fake.

    Tier 1 (testcontainers, real PostgreSQL 17). The in-memory test above proves the
    route passes a tenant down; this proves the statement and the column actually
    exclude the other tenant's rows, which is a property of the WHERE clause and would
    be satisfied by a fake keyed on nothing at all.

    Both tenants create Sessions and the creates interleave, so a missing tenant term
    shows up as the other tenant's Sessions in the list rather than as an ordering
    difference.
    """
    platform, engine = build(database_url)
    ours = {TENANT_HEADER: str(uuid.uuid4())}
    theirs = {TENANT_HEADER: str(uuid.uuid4())}
    try:
        app = create_app(platform)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://tenant"
        ) as client:
            mine: list[str] = []
            yours: list[str] = []
            for headers, into in ((ours, mine), (theirs, yours), (ours, mine)):
                registered = await client.post(
                    "/v1/agents", json=_a_definition(), headers=headers
                )
                assert registered.status_code == 201, registered.text
                created = await client.post(
                    "/v1/sessions",
                    json=_create_body(
                        registered.json()["id"],
                        await _an_environment(client, headers),
                    ),
                    headers=headers,
                )
                assert created.status_code == 201, created.text
                into.append(created.json()["id"])

            listed = (await client.get("/v1/sessions", headers=ours)).json()
    finally:
        await engine.dispose()

    ids = [row["id"] for row in listed["sessions"]]
    assert sorted(ids) == sorted(mine)
    assert not set(ids) & set(yours), (
        "another tenant's Session was returned in this tenant's list"
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

    async def delete(self, webhook_id: uuid.UUID, tenant_id: TenantId) -> bool:
        raise AssertionError("a test in this file deleted a webhook")

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file asked what watches a type")


class UnusedEnvironmentStore:
    """Satisfies the environment-store port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    environment store would be grading something this file does not grade, and a quiet
    stub would let it pass while doing so.
    """

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        raise AssertionError("a test in this file resolved an environment")
