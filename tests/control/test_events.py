"""Paging one Session's Event Log by sequence range, and the three answers it gives.

Tier 1 (testcontainers, real PostgreSQL 17). Realizes MAP-A47 (a sequence range returns
exactly that span, in order, with nothing added or omitted), MAP-A48 (a range past the
end is empty rather than an error) and MAP-A51 (every refusal carries a code from the
published closed set).

**The whole slice turns on three answers staying three.** A span of events, an empty
page above the head, and a refusal below the retained floor are easy to collapse into
two: the bug that loses expiry returns an empty page for a swept range, and an empty
page is exactly what a correct read above the head returns. A test asserting "empty"
for a below-floor read therefore passes while the feature is broken. So each answer is
asserted against *both* of the others rather than only against itself, and
`test_the_three_answers_are_three` puts all three side by side in one fixture where the
collapse would be visible as two of them being equal.

Driven with `AsyncClient` over `ASGITransport` rather than `TestClient`, which runs the
app in an event loop of its own — an async engine's pooled connections belong to the
loop that opened them, so sharing one across the two fails on the *second* request with
`got Future attached to a different loop`, which reads as a database fault and is not
one.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient, Response
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.events import MAX_RANGE
from managed_agent.core.errors import ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import SessionId

_SKILLS_SHA = "0" * 39 + "b"

# Substrings of the Agent Runtime's own error vocabulary. ADR-013 closes the tenant-
# facing set precisely so none of these can reach a caller; a refusal body is searched
# for them rather than only checked for a valid code, because a leak would arrive in the
# message or the detail, which no schema check looks at.
_RUNTIME_TAXONOMY = (
    "codex",
    "jsonrpc",
    "json-rpc",
    "app-server",
    "turnabort",
    "threadsource",
    "multiagentmode",
    "traceback",
)


def _a_definition() -> dict[str, object]:
    return {
        "name": "events-fixture",
        "instructions": "irrelevant to these tests",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SKILLS_SHA,
    }


_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
"""A digest-pinned image, because a registered shape refuses anything else."""


async def _an_environment(client: AsyncClient, headers: dict[str, str]) -> str:
    """A registered sandbox shape, because creating a Session names one.

    Registered through the route rather than written into the table, so the id
    the create call names is one the platform actually issued.
    """
    registered = await client.post(
        "/v1/environments",
        json={"name": "events-fixture", "runtime_image": _FIXTURE_IMAGE},
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
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 7,
    }


@pytest.fixture
async def platform_client(
    database_url: str,
) -> AsyncIterator[tuple[Platform, AsyncClient]]:
    """The whole wired platform behind a client, disposed at the end of the test."""
    platform, engine = build(database_url)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app(platform)),
            base_url="http://tenant",
        ) as client:
            yield platform, client
    finally:
        await engine.dispose()


def _headers() -> dict[str, str]:
    return {TENANT_HEADER: str(uuid.uuid4())}


async def _a_session(client: AsyncClient, headers: dict[str, str]) -> SessionId:
    """A created Session. Its creation event is seq 1, so appended events start at 2."""
    registered = await client.post("/v1/agents", json=_a_definition(), headers=headers)
    assert registered.status_code == 201, registered.text
    created = await client.post(
        "/v1/sessions",
        json=_create_body(
            registered.json()["id"], await _an_environment(client, headers)
        ),
        headers=headers,
    )
    assert created.status_code == 201, created.text
    return SessionId(uuid.UUID(created.json()["id"]))


async def _append(platform: Platform, session_id: SessionId, upto: int) -> None:
    """Append events until the log's highest sequence is `upto`."""
    for n in range(2, upto + 1):
        await platform.event_log_append.append(session_id, f"event.{n}", {"n": n})


async def _sweep(engine: AsyncEngine, session_id: SessionId, through: int) -> None:
    """Expire every event at or below `through`, the way a retention sweep does."""
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "DELETE FROM event_log WHERE session_id = :sid AND seq <= :through"
            ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
            {"sid": session_id, "through": through},
        )


def _seqs(body: dict[str, object]) -> list[int]:
    events = body["events"]
    assert isinstance(events, list)
    return [event["seq"] for event in events]


def _is_a_published_refusal(body: object) -> PublicErrorEnvelope:
    """Parse a refusal body back through the published envelope, or fail saying so.

    Parsed rather than spot-checked: `PublicErrorEnvelope` forbids extra fields and
    types its code against the closed enum, so a body that round-trips through it
    cannot carry an unpublished code or an undeclared top-level field. A couple of `in`
    assertions would pass on both.
    """
    assert isinstance(body, dict)
    try:
        return PublicErrorEnvelope.model_validate(body)
    except ValidationError as invalid:
        raise AssertionError(
            f"a refusal body is not a published PublicErrorEnvelope: {body!r}"
            f"\n{invalid}"
        ) from invalid


def _leaks_the_runtime(text: str) -> list[str]:
    lowered = text.lower()
    return [name for name in _RUNTIME_TAXONOMY if name in lowered]


async def test_a_range_returns_exactly_that_span_in_order(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """MAP-A47. Nothing added, nothing omitted, and the ends are inclusive."""
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)

    page = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 2, "to_seq": 4},
        headers=headers,
    )

    assert page.status_code == 200, page.text
    body = page.json()
    assert _seqs(body) == [2, 3, 4]
    assert [event["type"] for event in body["events"]] == [
        "event.2",
        "event.3",
        "event.4",
    ]
    assert body["from_seq"] == 2
    assert body["to_seq"] == 4
    assert body["session_id"] == str(session_id)


async def test_a_span_is_not_answered_as_an_empty_page(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """The first off-diagonal pair: events that exist must not read as none.

    Separate from the test above because that one would also pass if the route returned
    the right *shape* with the wrong contents only sometimes. This asserts the one thing
    a collapse toward emptiness breaks.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)

    page = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 1, "to_seq": 5},
        headers=headers,
    )

    assert page.status_code == 200, page.text
    assert _seqs(page.json()) == [1, 2, 3, 4, 5], (
        "a range wholly inside a retained log came back short or empty, which a caller "
        "cannot tell from having read to the end"
    )


async def test_a_range_past_the_head_is_an_empty_page_and_not_a_refusal(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """MAP-A48, and the second off-diagonal pair: empty must not read as expired.

    Those events do not exist *yet*. A caller polling forward reaches the head by
    reading past it, so a refusal here would turn ordinary paging into an error path.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)

    page = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 9, "to_seq": 12},
        headers=headers,
    )

    assert page.status_code == 200, page.text
    assert page.json()["events"] == []
    assert page.json()["from_seq"] == 9
    assert page.json()["to_seq"] == 12


async def test_reading_past_the_head_leaves_the_log_alone(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """MAP-A48's second half: the Session is unaffected.

    A read is a query and must change nothing. Asserted by reading the whole log before
    and after, because the way this would go wrong — a write that advances a cursor or
    a sequence on behalf of the reader — leaves the response identical.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)
    before = await platform.event_log_range.read(session_id, 1, 50)

    await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 9, "to_seq": 12},
        headers=headers,
    )

    after = await platform.event_log_range.read(session_id, 1, 50)
    assert [(row.seq, row.type) for row in after] == [
        (row.seq, row.type) for row in before
    ]
    assert await platform.event_log_range.retained_floor(session_id) == 1


async def test_a_range_below_the_floor_is_refused_rather_than_emptied(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """The pair the whole slice exists for: expiry must not read as emptiness.

    This is the assertion a naive implementation passes for the wrong reason. A route
    that simply read the range would return `[]` here, and `[]` is a legitimate answer
    one sequence number away — so the status is asserted, not the emptiness.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)
    await _sweep(engine, session_id, through=2)

    refused = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 1, "to_seq": 2},
        headers=headers,
    )

    assert refused.status_code == 410, (
        f"a range whose events were swept answered {refused.status_code}; an expired "
        f"position must be distinguishable from an empty one: {refused.text}"
    )
    envelope = _is_a_published_refusal(refused.json())
    assert envelope.error.code is ErrorCode.EVENT_RANGE_EXPIRED
    assert envelope.error.detail["retained_floor"] == 3
    assert "events" not in refused.json()


async def test_a_range_spanning_the_floor_is_refused_rather_than_answered_with_the_tail(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """The third off-diagonal pair: a refusal must not decay into a shorter span.

    Handing back seq 3-5 for a request for 1-5 is the most dangerous of the wrong
    answers, because it looks complete. A caller diffing sequence numbers would see a
    contiguous run starting where it asked for something earlier, and conclude the
    earlier events never existed.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)
    await _sweep(engine, session_id, through=2)

    refused = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 1, "to_seq": 5},
        headers=headers,
    )

    assert refused.status_code == 410, refused.text
    assert (
        _is_a_published_refusal(refused.json()).error.code
        is ErrorCode.EVENT_RANGE_EXPIRED
    )
    assert "event.3" not in refused.text, (
        "the refusal carried the surviving tail, so a caller could mistake a partial "
        "answer for the range it asked for"
    )


async def test_the_surviving_range_still_reads_after_a_sweep(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """The refusal is about the expired position, not about the Session becoming
    unreadable.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)
    await _sweep(engine, session_id, through=2)

    page = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 3, "to_seq": 5},
        headers=headers,
    )

    assert page.status_code == 200, page.text
    assert _seqs(page.json()) == [3, 4, 5]
    assert page.json()["retained_floor"] == 3


async def test_the_three_answers_are_three(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """One fixture, three reads, three outcomes that must stay pairwise distinct.

    The individual tests above each assert one answer. This one is what fails if any
    two of them become the same thing — the failure mode that survives a suite of
    single-answer tests, because each of those keeps passing while the pair collapses.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)
    await _sweep(engine, session_id, through=2)

    async def ask(from_seq: int, to_seq: int) -> tuple[int, bool]:
        response = await client.get(
            f"/v1/sessions/{session_id}/events",
            params={"from_seq": from_seq, "to_seq": to_seq},
            headers=headers,
        )
        body = response.json()
        has_events = bool(body.get("events")) if response.status_code == 200 else False
        return response.status_code, has_events

    span = await ask(3, 5)
    above_the_head = await ask(40, 45)
    below_the_floor = await ask(1, 2)

    assert span == (200, True)
    assert above_the_head == (200, False)
    assert below_the_floor == (410, False)
    assert len({span, above_the_head, below_the_floor}) == 3, (
        f"two of the three answers are indistinguishable: span={span}, "
        f"above_the_head={above_the_head}, below_the_floor={below_the_floor}"
    )


async def test_every_page_carries_the_retained_floor(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """So a caller that just lost a range can move to one that still exists.

    Without it the only way back is to bisect: ask for a range, be refused, ask higher.
    The floor makes the next request the right one.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)

    fresh = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 1, "to_seq": 5},
        headers=headers,
    )
    assert fresh.json()["retained_floor"] == 1

    await _sweep(engine, session_id, through=3)
    swept = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 4, "to_seq": 5},
        headers=headers,
    )
    assert swept.json()["retained_floor"] == 4


async def test_the_floor_in_a_refusal_names_a_range_that_then_succeeds(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """The refusal's `retained_floor` is actionable, not decorative.

    Asserted by using it: the caller re-asks starting exactly where the refusal said the
    log still begins, and that read must succeed. A floor that were off by one would
    leave a caller in a refusal loop, and no assertion about its presence would notice.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)
    await _sweep(engine, session_id, through=2)

    refused = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 1, "to_seq": 5},
        headers=headers,
    )
    floor = _is_a_published_refusal(refused.json()).error.detail["retained_floor"]

    retried = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": floor, "to_seq": 5},
        headers=headers,
    )

    assert retried.status_code == 200, retried.text
    assert _seqs(retried.json())[0] == floor


@pytest.mark.parametrize(
    ("from_seq", "to_seq"),
    [(5, 3), (1, MAX_RANGE + 1)],
    ids=["inverted", "wider-than-the-cap"],
)
async def test_a_malformed_range_is_refused_with_the_published_request_code(
    platform_client: tuple[Platform, AsyncClient], from_seq: int, to_seq: int
) -> None:
    """An inverted range and an over-wide one are caller errors, not empty answers.

    An inverted range describes no events, so answering `[]` would be indistinguishable
    from "you have read to the end" and a paging loop would stop silently at the wrong
    place. An over-wide one is refused rather than narrowed for the same reason a short
    page is dangerous: a truncated answer a caller cannot identify as truncated.
    """
    _, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)

    refused = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": from_seq, "to_seq": to_seq},
        headers=headers,
    )

    assert refused.status_code == 400, refused.text
    assert (
        _is_a_published_refusal(refused.json()).error.code is ErrorCode.REQUEST_INVALID
    )


async def test_a_span_wider_than_the_adapters_page_still_returns_all_of_it(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """The range is the limit, so no page cap can silently shorten the answer.

    The port caps a read and documents a short result as "page for the rest", with a
    default well below this route's widest span. A route that took the default would
    return that many events for a wider range and label them with the range that was
    asked for — a truncation the caller cannot see, and exactly the defect that once
    made a state fold report a stale Session state with every test passing.

    Seeded by insert rather than through `append`, which serializes per Session and
    would spend six hundred round trips proving something about the read path.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    highest = 600
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO event_log (session_id, seq, type, payload)"
                " SELECT :sid, g, 'event.bulk', '{}'"
                " FROM generate_series(2, :highest) g"
            ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
            {"sid": session_id, "highest": highest},
        )

    page = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 1, "to_seq": highest},
        headers=headers,
    )

    assert page.status_code == 200, page.text
    assert _seqs(page.json()) == list(range(1, highest + 1)), (
        f"asked for 1..{highest} and got "
        f"{len(_seqs(page.json()))} events; the read was capped below the range it "
        "was told to return, so the page omits events without saying so"
    )


async def test_another_tenants_events_are_refused_and_not_returned(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """The Event Log carries no tenant, so this route is the only thing scoping it.

    A range read keyed on the Session id alone succeeds for anybody who knows the id and
    hands back the raw events — payloads included, which is strictly more than the state
    fold on the Session route would leak. The registry is asked first for that reason.
    """
    platform, client = platform_client
    owner = _headers()
    stranger = _headers()
    session_id = await _a_session(client, owner)
    await _append(platform, session_id, 5)

    refused = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 1, "to_seq": 5},
        headers=stranger,
    )

    assert refused.status_code == 404, (
        f"another tenant read the Event Log of a Session it does not own: "
        f"{refused.status_code} {refused.text}"
    )
    assert (
        _is_a_published_refusal(refused.json()).error.code
        is ErrorCode.SESSION_NOT_FOUND
    )
    assert "event." not in refused.text, "the refusal carried the events it refused"
    # And the owner still reads it, so the refusal is about the caller and not about
    # the Session having become unreadable.
    theirs = await client.get(
        f"/v1/sessions/{session_id}/events",
        params={"from_seq": 1, "to_seq": 5},
        headers=owner,
    )
    assert theirs.status_code == 200, theirs.text
    assert _seqs(theirs.json()) == [1, 2, 3, 4, 5]


async def test_a_hidden_session_and_an_absent_one_refuse_identically(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """One refusal for both, so the difference cannot be used to enumerate ids.

    Compared as whole bodies with the ids substituted out: a message or a detail key
    that differed by a word would answer "does this id exist somewhere else" just as
    well as a different status would.

    Two ids are substituted, not one. The Session id goes because it is the thing being
    probed for. The request id goes because it is minted per call and is therefore
    different on every request by construction -- it says nothing about whether a
    Session exists, and comparing it would fail every run while proving nothing.
    """
    platform, client = platform_client
    owner = _headers()
    stranger = _headers()
    hidden = await _a_session(client, owner)
    await _append(platform, hidden, 3)
    absent = uuid.uuid4()

    hidden_refusal = await client.get(
        f"/v1/sessions/{hidden}/events", params={"from_seq": 1}, headers=stranger
    )
    absent_refusal = await client.get(
        f"/v1/sessions/{absent}/events", params={"from_seq": 1}, headers=stranger
    )

    def without_the_ids(refusal: Response, session_id: object) -> str:
        return refusal.text.replace(str(session_id), "<id>").replace(
            str(refusal.json()["request_id"]), "<request>"
        )

    assert hidden_refusal.status_code == absent_refusal.status_code == 404
    assert without_the_ids(hidden_refusal, hidden) == without_the_ids(
        absent_refusal, absent
    )


async def test_a_read_with_no_tenant_is_refused(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """No tenant, no read, and no default tenant to fall back to.

    Asserted at runtime as well as structurally in `test_tenancy.py`: that file reads
    the source text, so it can tell the dependency is written down but not that FastAPI
    resolves it before the handler body runs.
    """
    _, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)

    response = await client.get(f"/v1/sessions/{session_id}/events")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request.tenant_missing"


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"from_seq": 1, "to_seq": 2}, 410),
        ({"from_seq": 5, "to_seq": 3}, 400),
    ],
    ids=["expired", "inverted"],
)
async def test_no_refusal_carries_the_runtime_vocabulary(
    platform_client: tuple[Platform, AsyncClient],
    engine: AsyncEngine,
    params: dict[str, int],
    expected: int,
) -> None:
    """MAP-A51/ADR-013: a refusal names this platform's vocabulary and no other.

    Searched over the whole body rather than only the code, because a leak arrives in
    the free-text message or in a detail value — neither of which the closed enum
    constrains.
    """
    platform, client = platform_client
    headers = _headers()
    session_id = await _a_session(client, headers)
    await _append(platform, session_id, 5)
    await _sweep(engine, session_id, through=2)

    refused = await client.get(
        f"/v1/sessions/{session_id}/events", params=params, headers=headers
    )

    assert refused.status_code == expected, refused.text
    _is_a_published_refusal(refused.json())
    assert _leaks_the_runtime(refused.text) == [], (
        f"a refusal body carries the Agent Runtime's own vocabulary: "
        f"{_leaks_the_runtime(refused.text)} in {refused.text}"
    )


async def test_the_route_is_published_with_both_refusals_in_its_schema(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """The OpenAPI document names the refusal statuses and the envelope they use.

    A caller generating a client from the schema otherwise sees only the 200 and treats
    a 410 as an unexpected fault — which is the opposite of a closed set being useful.
    """
    _, client = platform_client
    schema = (await client.get("/openapi.json")).json()

    responses = schema["paths"]["/v1/sessions/{session_id}/events"]["get"]["responses"]
    assert {"200", "400", "404", "410"} <= set(responses)
    for status in ("400", "404", "410"):
        body = responses[status]["content"]["application/json"]["schema"]
        assert "PublicErrorEnvelope" in str(body), (
            f"the {status} response does not publish PublicErrorEnvelope as its "
            f"body: {body}"
        )
