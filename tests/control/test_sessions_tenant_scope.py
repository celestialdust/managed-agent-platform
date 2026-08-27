"""Creating a Session records who owns it, and reading one is refused to anybody else.

Tier 1 (testcontainers, real PostgreSQL 17) throughout, and the tier is the point. What
is being graded is that a cross-tenant read is **refused rather than answered**, and the
way that goes wrong is a missing tenant term in a WHERE clause -- a defect a fake keyed
on the Session id alone cannot have and therefore cannot reveal.

The reason it has to be refused rather than emptied is MAP-A12's own: a Session's state
is folded from its Event Log, the log is keyed by Session and carries no tenant, so a
fold over somebody else's Session succeeds and hands back their state with nothing
raising. There is exactly one thing standing between that and a tenant boundary, and it
is the registry read this file exercises.

Driven with `AsyncClient` over `ASGITransport` rather than `TestClient`, which runs the
app in an event loop of its own -- an async engine's pooled connections belong to the
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
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.session_list import SESSION_NOT_FOUND
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.ports import SessionNotVisible
from managed_agent.core.registration.advertised_name import advertised_name_for
from managed_agent.core.vocabulary import lifecycle

_SKILLS_SHA = "0" * 39 + "a"

# None of these is the value any other fixture uses, so a registry row filled from a
# constant rather than from the request would be visible in the comparison below.
_BUDGET = 731
_RETENTION = 17
_CURRENCY = "EUR"
_SERVER = "scope-fixture-tools"
_TOOLS = ("read_file", "fetch_page")
_GRANT = [advertised_name_for(_SERVER, one) for one in _TOOLS]
"""Real advertised names, because the create route now resolves every one of them.

They were `fs.read` and `web.fetch` -- names nothing had ever registered, which the
route stored unread. It refuses them now, so the fixture registers the server that
makes them real rather than granting nothing: an empty Grant would be the same value
every other fixture uses, and this file's whole method is that no constant here is
shared, so a row filled from a constant rather than from the request shows up in the
comparison below.
"""
_SCOPE = {"repository": "acme/widgets", "workspace": "research"}


def _without_request_id(response: Response) -> str:
    """One response's body text with its own per-request id masked out.

    The id is minted per request, before any handler decides anything, so no two
    responses carry the same one and it reports nothing about either. It has to come out
    before two refusals can be compared byte for byte: left in, every pair differs on it
    and the comparison loses the power to detect the differences it exists to detect.
    """
    body: dict[str, object] = response.json()
    return response.text.replace(str(body["request_id"]), "<request>")


def _a_definition(name: str = "scope-fixture") -> dict[str, object]:
    return {
        "name": name,
        "instructions": "irrelevant to these tests",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SKILLS_SHA,
    }


_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
"""A digest-pinned image, because a registered shape refuses anything else."""


def _create_body(definition_id: str, environment_id: str) -> dict[str, object]:
    return {
        "definition_id": definition_id,
        "environment_id": environment_id,
        "grant": _GRANT,
        "scope": _SCOPE,
        "budget_minor_units": _BUDGET,
        "budget_currency": _CURRENCY,
        "retention_days": _RETENTION,
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


async def _a_session(client: AsyncClient, headers: dict[str, str]) -> str:
    server = await client.post(
        "/v1/mcp_servers",
        json={
            "server_name": _SERVER,
            "endpoint": {
                "transport": "streamable_http",
                "url": "https://mcp.example.invalid",
                "credential_ref": "vault/acme/scope-fixture",
            },
            "tools": [
                {
                    "name": one,
                    "remote_name": one,
                    "parameters": {"repoName": "string"},
                    "scope_bindings": [
                        {"dimension": "repository", "argument": "repoName"}
                    ],
                }
                for one in _TOOLS
            ],
        },
        headers=headers,
    )
    assert server.status_code == 201, server.text
    registered = await client.post("/v1/agents", json=_a_definition(), headers=headers)
    assert registered.status_code == 201, registered.text
    # The sandbox shape is registered through its own route rather than written into the
    # table, so the id the create call names is one the platform actually issued -- and
    # one owned by this tenant, which is what the create path re-checks.
    shape = await client.post(
        "/v1/environments",
        json={"name": "scope-fixture", "runtime_image": _FIXTURE_IMAGE},
        headers=headers,
    )
    assert shape.status_code == 201, shape.text
    created = await client.post(
        "/v1/sessions",
        json=_create_body(registered.json()["id"], shape.json()["id"]),
        headers=headers,
    )
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


_ROW = (
    sa.text(
        "SELECT tenant_id, grant_tools, scope, budget_minor_units,"
        " budget_currency, retention_days, definition_revision"
        " FROM session WHERE id = :id"
    )
    .bindparams(sa.bindparam("id", type_=sa.Uuid()))
    .columns(
        tenant_id=sa.Uuid(),
        grant_tools=sa.JSON(),
        scope=sa.JSON(),
        budget_minor_units=sa.BigInteger(),
        budget_currency=sa.Text(),
        retention_days=sa.Integer(),
        definition_revision=sa.Text(),
    )
)


async def test_creating_a_session_records_the_caller_as_its_owner(
    platform_client: tuple[Platform, AsyncClient], engine: AsyncEngine
) -> None:
    """The row exists, it belongs to the caller, and its facts are the ones submitted.

    Read straight out of the table rather than back through the API, because the API is
    the thing under test: a read path that answered from the request rather than from
    the store would agree with itself perfectly.
    """
    _, client = platform_client
    tenant = TenantId(uuid.uuid4())
    headers = {TENANT_HEADER: str(tenant)}

    session_id = await _a_session(client, headers)

    async with engine.connect() as conn:
        row = (await conn.execute(_ROW, {"id": uuid.UUID(session_id)})).one()

    assert row.tenant_id == tenant
    assert sorted(row.grant_tools) == sorted(_GRANT)
    assert row.scope == _SCOPE
    assert row.budget_minor_units == _BUDGET
    assert row.budget_currency == _CURRENCY
    assert row.retention_days == _RETENTION
    assert row.definition_revision == "1"


async def test_the_registry_row_and_the_created_event_agree(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """Two records of one creation, written from one parsed body, and they match.

    They are two stores and neither can be updated, so the only moment they could
    disagree is the moment they are written -- which is what this reads. A Session whose
    row said one budget and whose log said another would leave every later reader with a
    choice nobody had made.
    """
    platform, client = platform_client
    tenant = TenantId(uuid.uuid4())
    session_id = await _a_session(client, {TENANT_HEADER: str(tenant)})
    typed_id = SessionId(uuid.UUID(session_id))

    record = await platform.session_registry.fetch(typed_id, tenant)
    events = await platform.event_log_range.read(typed_id, 1, 10)

    (created,) = [e for e in events if e.type == lifecycle.SESSION_CREATED]
    assert sorted(record.grant) == created.payload["grant"]
    assert dict(record.scope) == created.payload["scope"]
    assert record.budget_minor_units == created.payload["budget_minor_units"]
    assert record.budget_currency == created.payload["budget_currency"]
    assert record.retention_days == created.payload["retention_days"]
    assert record.definition_revision == str(created.payload["definition_revision"])
    assert str(record.definition_id) == created.payload["definition_id"]


async def test_a_tenant_reads_back_its_own_session(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """MAP-A12's happy half: the state is readable on demand and it says running.

    Which Turn it is on and what it has spent are the other two halves of that scenario;
    they arrive with the slices that own the Turn and the spend ledger, and neither
    exists yet.
    """
    _, client = platform_client
    headers = {TENANT_HEADER: str(uuid.uuid4())}
    session_id = await _a_session(client, headers)

    response = await client.get(f"/v1/sessions/{session_id}", headers=headers)

    assert response.status_code == 200, response.text
    assert response.json() == {"id": session_id, "state": "idle", "seq": 1}


async def test_another_tenants_session_is_refused_and_not_answered_emptily(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """MAP-A12: refused, not empty, and not the other tenant's state either.

    Three answers are wrong here and only one is right. Returning the Session is the
    obvious failure. Returning 200 with a default-shaped or empty body is the quiet one
    -- a caller cannot tell it from a Session that has not started, so it would report
    somebody else's Session as idle rather than as none of its business. Refusing is the
    only answer that says what happened.
    """
    _, client = platform_client
    owner = {TENANT_HEADER: str(uuid.uuid4())}
    stranger = {TENANT_HEADER: str(uuid.uuid4())}
    session_id = await _a_session(client, owner)

    refused = await client.get(f"/v1/sessions/{session_id}", headers=stranger)

    assert refused.status_code == 404, refused.text
    assert refused.json()["error"]["code"] == SESSION_NOT_FOUND
    assert "running" not in refused.text
    # And the owner still reads it, so the refusal is about the tenant and not about the
    # Session having become unreadable.
    theirs = await client.get(f"/v1/sessions/{session_id}", headers=owner)
    assert theirs.status_code == 200, theirs.text


async def test_a_hidden_session_and_an_absent_one_refuse_identically(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """One refusal for both, so the difference cannot be used to enumerate ids.

    Compared as whole response bodies with the id substituted out, because the code
    alone is not the leak -- a message, a header or a status that differed by a word
    would answer "does this id exist somewhere else" just as well.
    """
    _, client = platform_client
    owner = {TENANT_HEADER: str(uuid.uuid4())}
    stranger = {TENANT_HEADER: str(uuid.uuid4())}
    hidden = await _a_session(client, owner)
    absent = str(uuid.uuid4())

    hidden_refusal = await client.get(f"/v1/sessions/{hidden}", headers=stranger)
    absent_refusal = await client.get(f"/v1/sessions/{absent}", headers=stranger)

    assert hidden_refusal.status_code == absent_refusal.status_code == 404
    hidden_body = _without_request_id(hidden_refusal).replace(hidden, "<id>")
    absent_body = _without_request_id(absent_refusal).replace(absent, "<id>")
    assert hidden_body == absent_body


async def test_a_read_with_no_tenant_is_refused(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """No tenant, no read. There is no default tenant to fall back to.

    A default is the failure mode that survives the arrival of real authentication:
    every call site keeps working and quietly serves one tenant's Session to another.
    """
    _, client = platform_client
    headers = {TENANT_HEADER: str(uuid.uuid4())}
    session_id = await _a_session(client, headers)

    response = await client.get(f"/v1/sessions/{session_id}")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request.tenant_missing"


async def test_the_registry_refuses_a_cross_tenant_fetch_beneath_the_route(
    platform_client: tuple[Platform, AsyncClient],
) -> None:
    """The refusal originates in the store's own query, not in the route.

    Worth asserting separately from the 404 above: a route that fetched without a tenant
    and compared afterwards would produce the same 404 and would have already read
    another tenant's row into this process. This is the assertion that the row is never
    fetched at all.
    """
    platform, client = platform_client
    owner = TenantId(uuid.uuid4())
    stranger = TenantId(uuid.uuid4())
    session_id = SessionId(
        uuid.UUID(await _a_session(client, {TENANT_HEADER: str(owner)}))
    )

    with pytest.raises(SessionNotVisible):
        await platform.session_registry.fetch(session_id, stranger)
