"""GET /v1/sessions walked BACKWARD, with the direction inside the token.

Tier 0 against in-memory registries, because everything graded here is a property of
this module. Two registries, and the second one is the point: one implements
`page_ending_at` and one does not, and the difference between them is the difference
between a deployment that offers `prev_page` and a deployment that does not mention it.
A single fake could not tell those apart, and the field being absent rather than null is
the whole contract for the second.

The backward fake implements the comparison from its definition -- rows at or newer than
the key, oldest-first, cut to the limit -- rather than by transcribing the adapter's
SQL. A fake written from the SQL agrees with the SQL by construction and grades nothing;
the one below can disagree with it, and the walk that would notice is the last test in
this file.

What no fake can grade is whether PostgreSQL's `>=` on a row constructor walks the index
the way this expects, which is why `tests/adapters/test_session_registry.py` walks the
same pages through the real adapter.
"""

from __future__ import annotations

import uuid
from base64 import urlsafe_b64encode
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.session_list import (
    CURSOR_INVALID,
    Cursor,
    InvalidCursor,
    Walk,
)
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import FIRST_SEQ, DefinitionId, Seq, SessionId, TenantId
from managed_agent.core.ports import (
    EventRecord,
    Resolution,
    SessionListing,
    SessionsWalkedBackward,
)
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord, SessionState

_REVISION = "9"
_BASE_MS = 1_700_000_000_000

_PAGE_BUDGET = 20
_NEVER_ENDED = (
    f"the walk read {_PAGE_BUDGET} pages without running out of cursor; it is either "
    "repeating rows at a page boundary or never reporting an end"
)


@dataclass(frozen=True, slots=True)
class Listing:
    id: SessionId
    definition_id: DefinitionId
    definition_revision: str
    created_at_ms: int


class ForwardOnlyRegistry:
    """A registry with `page` and nothing else -- what most of this tree's fakes are.

    Here on purpose rather than by omission. A deployment whose store cannot walk
    backward must not put `prev_page` on a response at all, and this is the store that
    makes that path run."""

    def __init__(self) -> None:
        self.rows: list[Listing] = []
        self.owner: dict[SessionId, TenantId] = {}

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

    def _newest_first(self, tenant_id: TenantId) -> list[Listing]:
        return sorted(
            (row for row in self.rows if self.owner[row.id] == tenant_id),
            key=lambda row: (row.created_at_ms, row.id),
            reverse=True,
        )

    async def create(self, record: SessionRecord) -> None:
        raise AssertionError("a test in this file created a Session through the route")

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        raise AssertionError("a test in this file read a single Session")

    async def page(
        self, tenant_id: TenantId, after: tuple[int, uuid.UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        if limit < 1 or limit > 500:
            raise ValueError(f"page limit {limit} is outside 1..500")
        mine = self._newest_first(tenant_id)
        if after is not None:
            mine = [row for row in mine if (row.created_at_ms, row.id) < after]
        return mine[:limit]


class BackwardRegistry(ForwardOnlyRegistry):
    """The same registry, plus the one method that makes it able to page backward.

    Subclassed rather than written out again so the two differ in exactly the method
    under test: a second hand-written `page` could disagree with the first, and then a
    backward walk failing to retrace a forward one would say nothing about the route."""

    async def page_ending_at(
        self, tenant_id: TenantId, oldest: tuple[int, uuid.UUID], limit: int
    ) -> Sequence[SessionListing]:
        if limit < 1 or limit > 500:
            raise ValueError(f"page limit {limit} is outside 1..500")
        # At or newer than the key -- inclusive, which is what lets a forward cursor
        # name the page it closed -- and handed back oldest-first, walked order, so the
        # `limit` cut keeps the rows next to the key rather than the newest of them all.
        walked = sorted(
            (
                row
                for row in self._newest_first(tenant_id)
                if (row.created_at_ms, row.id) >= oldest
            ),
            key=lambda row: (row.created_at_ms, row.id),
        )
        return walked[:limit]


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


class UnusedWebhooks:
    async def register(
        self,
        tenant_id: TenantId,
        url: CallbackUrl,
        states: frozenset[SessionState],
        secret_ref: str,
    ) -> WebhookRecord:
        raise AssertionError("a test in this file registered a webhook")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file listed a tenant's webhooks")

    async def delete(self, webhook_id: uuid.UUID, tenant_id: TenantId) -> bool:
        raise AssertionError("a test in this file deleted a webhook")

    async def watching(
        self, tenant_id: TenantId, state: SessionState
    ) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file asked what watches a state")


class UnusedEnvironmentStore:
    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        raise AssertionError("a test in this file resolved an environment")


def _client(registry: ForwardOnlyRegistry, tenant: TenantId) -> TestClient:
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


@pytest.fixture
def tenant() -> TenantId:
    return TenantId(uuid.uuid4())


@pytest.fixture
def registry() -> BackwardRegistry:
    return BackwardRegistry()


@pytest.fixture
def client(registry: BackwardRegistry, tenant: TenantId) -> TestClient:
    return _client(registry, tenant)


def _legacy_token(created_at_ms: int, session_id: SessionId) -> str:
    """A two-field token, the shape this surface issued before directions existed.

    Built here from the encoding rather than taken from `Cursor.encode`, so it stays the
    old shape even if the encoder changes. That is the whole point of it: it is the
    fixed thing the current encoder is compared against."""
    raw = f"{created_at_ms}.{session_id}".encode()
    return urlsafe_b64encode(raw).decode().rstrip("=")


# --- the direction in the token ---------------------------------------------------


def test_a_backward_cursor_round_trips_to_an_equal_position() -> None:
    position = Cursor(_BASE_MS + 4, SessionId(uuid.uuid4()), Walk.BACKWARD)

    assert Cursor.decode(position.encode()) == position


def test_the_two_directions_on_one_row_are_two_different_tokens() -> None:
    """One row, two positions, and they must not collide.

    A token that dropped the direction would round-trip to a forward cursor, and a
    `prev_page` handed back to the one page parameter would walk forward -- landing on
    the page the caller was already on, which reads as paging and loops."""
    session_id = SessionId(uuid.uuid4())
    forward = Cursor(_BASE_MS, session_id, Walk.FORWARD)
    backward = Cursor(_BASE_MS, session_id, Walk.BACKWARD)

    assert forward.encode() != backward.encode()
    assert Cursor.decode(backward.encode()).walk is Walk.BACKWARD
    assert Cursor.decode(forward.encode()).walk is Walk.FORWARD


def test_a_forward_token_is_byte_for_byte_what_it_was_before_directions() -> None:
    """A forward token gained no bytes when the direction was added.

    Asserted against a token built from the old encoding, not against itself, which is
    what makes it a compatibility claim: every cursor a caller is holding right now was
    issued by the encoder this compares to."""
    session_id = SessionId(uuid.uuid4())

    issued = Cursor(_BASE_MS + 11, session_id).encode()

    assert issued == _legacy_token(_BASE_MS + 11, session_id)


def test_a_token_from_before_directions_reads_as_forward() -> None:
    """And it still decodes, to forward, with no branch spent on saying so.

    `Walk("")` is forward, so a token with two fields reads as forward out of the enum
    lookup itself. The pair of this test and the one above is the no-migration claim in
    both directions: old tokens in, identical tokens out."""
    session_id = SessionId(uuid.uuid4())

    decoded = Cursor.decode(_legacy_token(_BASE_MS + 2, session_id))

    assert decoded == Cursor(_BASE_MS + 2, session_id, Walk.FORWARD)


def test_a_token_naming_a_direction_that_does_not_exist_is_refused() -> None:
    """A third field this surface has no meaning for is a refusal, not a default.

    Reading an unknown mark as forward would take a token nobody issued and answer it
    with a page, and the caller would have no way to tell that the direction it asked
    for was discarded."""
    raw = f"{_BASE_MS}.{uuid.uuid4()}.sideways"
    token = urlsafe_b64encode(raw.encode()).decode().rstrip("=")

    with pytest.raises(InvalidCursor):
        Cursor.decode(token)


# --- what the two registries are ------------------------------------------------


def test_only_one_of_the_two_registries_can_walk_backward() -> None:
    """The control for every test below, and it is not decoration.

    Both fakes answer `page`, so a route that offered `prev_page` unconditionally would
    satisfy the tests that walk backward and the ones that check the field is absent
    would be the only failures -- unless `isinstance` were answering the same for both,
    in which case those would pass too and nothing here would be graded. This is the
    assertion that says the two fakes really are two cases."""
    assert isinstance(BackwardRegistry(), SessionsWalkedBackward)
    assert not isinstance(ForwardOnlyRegistry(), SessionsWalkedBackward)


# --- the field, and the three things it says -------------------------------------


def test_a_store_that_cannot_walk_backward_never_mentions_the_field(
    tenant: TenantId,
) -> None:
    """Absent, not null. A null would tell this caller it is on the first page.

    It is on the first page, as it happens, and that is why the assertion is on the key
    rather than the value: the same response shape has to mean "cannot answer" on page
    four, where null would be a lie."""
    registry = ForwardOnlyRegistry()
    registry.add(tenant, _BASE_MS)

    body = _client(registry, tenant).get("/v1/sessions").json()

    assert "prev_page" not in body, body
    assert body["next_page"] is None


def test_a_backward_cursor_is_refused_where_no_backward_cursor_was_issued(
    tenant: TenantId,
) -> None:
    """A backward cursor at a store that cannot walk one is 400, not 500.

    This request cannot arise from following links -- such a deployment issues no
    backward token -- so it is a caller replaying a token from somewhere else. The
    refusal is the same one every unissued cursor gets, because that is what it is."""
    registry = ForwardOnlyRegistry()
    row = registry.add(tenant, _BASE_MS)
    token = Cursor(row.created_at_ms, row.id, Walk.BACKWARD).encode()

    response = _client(registry, tenant).get(f"/v1/sessions?page={token}")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == CURSOR_INVALID


def test_the_first_page_says_so_with_a_null_rather_than_a_token(
    client: TestClient, registry: BackwardRegistry, tenant: TenantId
) -> None:
    """Null and present: the one place the route can say "first" without a read.

    There is no cursor, so the page starts at the newest row, so there is nothing before
    it. Five rows at a limit of two so `next_page` is a token -- otherwise a route that
    emitted null for both fields on every page would pass this."""
    for step in range(5):
        registry.add(tenant, _BASE_MS + step)

    body = client.get("/v1/sessions?limit=2").json()

    assert body["prev_page"] is None
    assert body["next_page"] is not None


# --- the walk ---------------------------------------------------------------------


def test_walking_back_one_page_lands_on_exactly_the_page_before(
    client: TestClient, registry: BackwardRegistry, tenant: TenantId
) -> None:
    """The off-by-one test, and the reason `page_ending_at` is inclusive.

    A forward cursor is the LAST row of the page before the one it opens. Walking back
    from it exclusively would return that page minus its final row, plus one row from
    further up -- a window shifted by one, which still looks like a page of three and
    drifts further every step. Compared against the first page's ids in order, so a
    shift by one row fails here."""
    for step in range(7):
        registry.add(tenant, _BASE_MS + step)

    first = client.get("/v1/sessions?limit=3").json()
    second = client.get(f"/v1/sessions?limit=3&page={first['next_page']}").json()
    back = client.get(f"/v1/sessions?limit=3&page={second['prev_page']}").json()

    assert [row["id"] for row in back["sessions"]] == [
        row["id"] for row in first["sessions"]
    ]
    assert back["prev_page"] is None, "the first page reached backward is still first"


def test_a_page_reached_backward_offers_the_page_it_came_from(
    client: TestClient, registry: BackwardRegistry, tenant: TenantId
) -> None:
    """A page reached backward can always be left the way the caller entered it.

    `next_page` on a backward page needs no extra row to prove itself: the caller
    arrived from the page after this one, so that page exists by construction. What this
    asserts is that the token points at it and not somewhere adjacent."""
    for step in range(7):
        registry.add(tenant, _BASE_MS + step)

    first = client.get("/v1/sessions?limit=3").json()
    second = client.get(f"/v1/sessions?limit=3&page={first['next_page']}").json()
    back = client.get(f"/v1/sessions?limit=3&page={second['prev_page']}").json()
    forward_again = client.get(f"/v1/sessions?limit=3&page={back['next_page']}").json()

    assert back["next_page"] is not None
    assert [row["id"] for row in forward_again["sessions"]] == [
        row["id"] for row in second["sessions"]
    ]


def test_the_whole_forward_walk_replays_in_reverse(
    client: TestClient, registry: BackwardRegistry, tenant: TenantId
) -> None:
    """Seven rows at a limit of two, walked out and walked back.

    The strongest statement this file makes, because it compares the two walks page for
    page rather than checking one hop. A boundary off by a row, a page that drops its
    oldest member, or a `prev_page` that skips a page all break the equality without
    breaking any single-hop assertion.

    The last forward page has no page after it, so the backward walk starts from it and
    retraces the ones before -- which is `forward[:-1]`, reversed."""
    for step in range(7):
        registry.add(tenant, _BASE_MS + step)

    forward: list[list[str]] = []
    cursor: str | None = None
    for _ in range(_PAGE_BUDGET):
        query = f"/v1/sessions?limit=2{'' if cursor is None else f'&page={cursor}'}"
        body = client.get(query).json()
        forward.append([row["id"] for row in body["sessions"]])
        cursor = body["next_page"]
        if cursor is None:
            break
    else:
        raise AssertionError(_NEVER_ENDED)

    backward: list[list[str]] = []
    cursor = body["prev_page"]
    for _ in range(_PAGE_BUDGET):
        if cursor is None:
            break
        body = client.get(f"/v1/sessions?limit=2&page={cursor}").json()
        backward.append([row["id"] for row in body["sessions"]])
        cursor = body["prev_page"]
    else:
        raise AssertionError(_NEVER_ENDED)

    assert [len(page) for page in forward] == [2, 2, 2, 1], forward
    assert backward == list(reversed(forward[:-1])), (
        "the backward walk did not retrace the forward one page for page"
    )


def test_a_backward_page_is_newest_first_like_every_other_page(
    client: TestClient, registry: BackwardRegistry, tenant: TenantId
) -> None:
    """Newest-first, the same as every other page, though the store walked it up.

    The store returns a backward page oldest-first, because that is the order the limit
    has to cut in. Flipping it back into presentation order is the route's job, and a
    route that skipped it would hand back one page sorted against every other."""
    for step in range(6):
        registry.add(tenant, _BASE_MS + step)

    first = client.get("/v1/sessions?limit=3").json()
    second = client.get(f"/v1/sessions?limit=3&page={first['next_page']}").json()
    back = client.get(f"/v1/sessions?limit=3&page={second['prev_page']}").json()

    times = [row["created_at_ms"] for row in back["sessions"]]
    assert times == sorted(times, reverse=True), times


def test_two_sessions_in_one_millisecond_survive_a_backward_boundary(
    client: TestClient, registry: BackwardRegistry, tenant: TenantId
) -> None:
    """Two Sessions in one millisecond, with the page boundary between them.

    Ordering on the millisecond alone leaves their order undefined, so a backward page
    whose comparison dropped the id could return the tied pair in the other order -- or
    return one of them twice. Compared to the forward page's ids in order, which is the
    only thing that pins it."""
    registry.add(tenant, _BASE_MS)
    registry.add(tenant, _BASE_MS + 1)
    registry.add(tenant, _BASE_MS + 1)
    registry.add(tenant, _BASE_MS + 2)

    first = client.get("/v1/sessions?limit=2").json()
    second = client.get(f"/v1/sessions?limit=2&page={first['next_page']}").json()
    back = client.get(f"/v1/sessions?limit=2&page={second['prev_page']}").json()

    assert [row["id"] for row in back["sessions"]] == [
        row["id"] for row in first["sessions"]
    ]
