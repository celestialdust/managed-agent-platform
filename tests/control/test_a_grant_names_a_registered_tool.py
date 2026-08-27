"""A Grant naming a tool this tenant never registered starts no Session.

Tier 1 (testcontainers, real PostgreSQL 17). The tier is load-bearing: the question is
whether a name in `grant` resolves against the tenant's own `registered_tool` rows, and
a fake registry keyed on a dict would let the route agree with a store nobody wrote to.

**What this closes.** `POST /v1/sessions` stored whatever `grant` it was given
without resolving it. The Session was created 201, started with no usable tools, and
the first sign of trouble arrived mid-Turn as the model reporting a tool missing --
after a pod had been placed and paid for. `docs/lessons.md` holds two entries for what
that costs when it happens for a different reason: nineteen consecutive live Turns in
which the model hunted for a granted tool, worked around it or stopped to ask the
tenant for it, with every service reporting itself healthy throughout.

The refusal is also the tenant's only remedy. A Session's Grant is fixed for its
whole life -- the revision route refuses one outright -- so a Grant that is wrong at
creation is wrong until the Session is replaced. Checking it at the one moment it can
still be corrected is the difference between a 400 and a dead Session.

The route already treats `file_ids` exactly this way, refusing an id the tenant cannot
read before anything is appended. This is the same check one field over.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from httpx import ASGITransport, AsyncClient, Response

from managed_agent.composition import build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.sessions import REASON_GRANT_NOT_REGISTERED
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.registration.advertised_name import advertised_name_for

_SKILLS_SHA = "0" * 39 + "d"
_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "d" * 64
_SERVER = "deepwiki"
_TOOL = "ask_question"


@dataclass(frozen=True)
class Wired:
    client: AsyncClient
    tenant: uuid.UUID


@pytest.fixture
async def wired(database_url: str) -> AsyncIterator[Wired]:
    platform, engine = build(database_url)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app(platform)),
            base_url="http://tenant",
        ) as client:
            yield Wired(client=client, tenant=uuid.uuid4())
    finally:
        await engine.dispose()


def _headers(tenant: uuid.UUID) -> dict[str, str]:
    return {TENANT_HEADER: str(tenant)}


async def _register_a_tool(wired: Wired) -> str:
    """One MCP server with one tool, and the joined name a Grant must use."""
    registered = await wired.client.post(
        "/v1/mcp_servers",
        json={
            "server_name": _SERVER,
            "endpoint": {
                "transport": "streamable_http",
                "url": "https://mcp.deepwiki.com",
                "credential_ref": "vault/acme/deepwiki",
            },
            "tools": [
                {
                    "name": _TOOL,
                    "remote_name": _TOOL,
                    "parameters": {"repoName": "string", "question": "string"},
                    "scope_bindings": [
                        {"dimension": "repository", "argument": "repoName"}
                    ],
                }
            ],
        },
        headers=_headers(wired.tenant),
    )
    assert registered.status_code == 201, registered.text
    return advertised_name_for(_SERVER, _TOOL)


async def _an_agent_and_an_environment(wired: Wired) -> tuple[str, str]:
    headers = _headers(wired.tenant)
    definition = await wired.client.post(
        "/v1/agents",
        json={
            "name": f"grant-fixture-{uuid.uuid4()}",
            "instructions": "irrelevant to these tests",
            "model": "gpt-5-codex",
            "skills_repository": "git@github.com:acme/skills.git",
            "skills_revision": _SKILLS_SHA,
        },
        headers=headers,
    )
    assert definition.status_code == 201, definition.text
    environment = await wired.client.post(
        "/v1/environments",
        json={"name": "grant-fixture", "runtime_image": _FIXTURE_IMAGE},
        headers=headers,
    )
    assert environment.status_code == 201, environment.text
    return str(definition.json()["id"]), str(environment.json()["id"])


async def _create(wired: Wired, grant: list[str]) -> Response:
    definition_id, environment_id = await _an_agent_and_an_environment(wired)
    return await wired.client.post(
        "/v1/sessions",
        json={
            "definition_id": definition_id,
            "environment_id": environment_id,
            "file_ids": [],
            "grant": grant,
            "scope": {},
            "budget_minor_units": 500,
            "budget_currency": "USD",
            "retention_days": 7,
        },
        headers=_headers(wired.tenant),
    )


async def test_a_grant_naming_an_unregistered_tool_is_refused(wired: Wired) -> None:
    """400, and the refusal names the tool rather than only the field.

    Naming it is the whole value of the refusal. A tenant reading "one of your granted
    tools is unknown" has to diff their own list against a registry they cannot see;
    reading the name back tells them at once whether they mistyped it or never
    registered it.
    """
    await _register_a_tool(wired)

    refused = await _create(wired, ["ask_question"])

    expected = STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert refused.status_code == expected == 400, refused.text
    body = refused.json()
    assert body["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert body["error"]["detail"]["reason"] == REASON_GRANT_NOT_REGISTERED
    assert body["error"]["detail"]["unregistered"] == "ask_question", body


async def test_the_bare_name_is_the_case_this_catches(wired: Wired) -> None:
    """The bare tool name is refused and the joined one is accepted, same tool.

    This is the shape the defect actually takes. Per-server scoping made a tool's
    identity the pair `(server, tool)` joined before the model sees it, so every Grant
    written against the old spelling silently stopped matching -- and a bare name is
    what a caller reading their own registration call will reach for first.
    """
    advertised = await _register_a_tool(wired)
    assert advertised == f"{_SERVER}__{_TOOL}", advertised

    refused = await _create(wired, [_TOOL])
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["detail"]["unregistered"] == _TOOL

    accepted = await _create(wired, [advertised])
    assert accepted.status_code == 201, accepted.text


async def test_every_unregistered_name_is_named_not_just_the_first(
    wired: Wired,
) -> None:
    """A caller fixing them one round-trip at a time is the failure this avoids."""
    advertised = await _register_a_tool(wired)

    refused = await _create(wired, [advertised, "web.fetch", "fs.read"])

    assert refused.status_code == 400, refused.text
    detail = refused.json()["error"]["detail"]
    assert detail["unregistered"] == "fs.read, web.fetch", detail


async def test_an_empty_grant_still_creates_a_session(wired: Wired) -> None:
    """Empty means "no tools" and is a legitimate request, not an unresolved name.

    Guarding the refusal's floor. A check written as "the Grant must resolve" rather
    than "each name in it must resolve" would refuse every Session that asks for
    nothing, which is the default the create route ships with.
    """
    created = await _create(wired, [])

    assert created.status_code == 201, created.text


async def test_the_revision_refusal_does_not_claim_the_grant_is_unread(
    wired: Wired,
) -> None:
    """The two Grant refusals must not contradict each other.

    `PATCH` refused a revision partly on the grounds that "nothing here reads the
    Grant, so the revision would take effect nowhere". That stopped being true when the
    Tool Gateway began narrowing its offer and its calls by the field, and the paragraph
    above the refusal had already been corrected while the message a caller actually
    receives had not. A tenant reading it would conclude their Grant was decorative, at
    the same moment the create route above started refusing Sessions over it.

    Guarded as a claim the message may not make, rather than as wording it must have,
    so it constrains only the thing that was wrong.
    """
    advertised = await _register_a_tool(wired)
    created = await _create(wired, [advertised])
    assert created.status_code == 201, created.text

    refused = await wired.client.post(
        f"/v1/sessions/{created.json()['id']}",
        json={"grant": [advertised]},
        headers=_headers(wired.tenant),
    )

    assert refused.status_code == 400, refused.text
    message = refused.json()["error"]["message"]
    assert "nothing here reads the Grant" not in message, message
    assert "fixed at creation" in message, message


async def test_the_resolved_grant_reaches_the_creation_event(wired: Wired) -> None:
    """A Grant that resolves is carried into `session.created`, sorted, unchanged.

    The refusal above is only half the contract. A check that rejected the bad names
    and then dropped the good ones would pass every test in this file up to here, and
    the Session would start able to call nothing -- which is the failure the refusal
    exists to prevent, reached by the other side.
    """
    advertised = await _register_a_tool(wired)

    created = await _create(wired, [advertised])
    assert created.status_code == 201, created.text

    events = await wired.client.get(
        f"/v1/sessions/{created.json()['id']}/events",
        headers=_headers(wired.tenant),
    )
    assert events.status_code == 200, events.text
    creation = [
        one for one in events.json()["events"] if one["type"] == "session.created"
    ]
    assert len(creation) == 1, creation
    assert creation[0]["payload"]["grant"] == [advertised], creation[0]["payload"]
