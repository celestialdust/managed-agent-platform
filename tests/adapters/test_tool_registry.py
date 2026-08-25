"""Registering an MCP server, and reading its tools back as parsed values.

Tier 1 (testcontainers, real PostgreSQL 17). The schema test beside this one grades the
constraints; this grades the adapter that has to live within them -- that a refused
registration leaves nothing behind, that a name collision is reported as which kind of
name and which one, and that what comes back out of the JSON columns is the same value
that went in rather than the text of it.

Every fixture here registers **two** of whatever the assertion is about, and that is
deliberate. A test where one tool, one server and one endpoint exist cannot tell a read
from a constant: `lookup` returning the only row it could possibly return proves nothing
about the WHERE clause. So the two servers reach different transports, the two tools sit
behind different servers, and the two tenants hold the same names.

The coordinates are the project's real MCP server rather than an invented one. Nothing
here dials them -- registering a server is a write to two tables and reaches no network
-- but a fixture whose URL and tool names are real is a fixture that stays honest when a
later slice does dial it.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.tool_registry import PostgresToolRegistry
from managed_agent.composition import build
from managed_agent.core.ids import TenantId
from managed_agent.core.ports import ToolRegistry
from managed_agent.core.registration.scope_binding import (
    NameAlreadyRegistered,
    ParameterType,
    ScopeBinding,
    ServerRegistration,
    StdioServer,
    StreamableHttpServer,
    UnknownTool,
)

_OVER_HTTP: dict[str, object] = {
    "transport": "streamable_http",
    "url": "https://mcp.deepwiki.com",
    "credential_ref": "vault/acme/deepwiki",
}

_OVER_STDIO: dict[str, object] = {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "mcp-remote", "https://mcp.deepwiki.com/sse"],
    "credential_ref": "vault/acme/deepwiki-local",
    "credential_env_var": "DEEPWIKI_TOKEN",
}


def _tool(name: str, **overrides: object) -> dict[str, object]:
    return {
        "name": name,
        "remote_name": name,
        "parameters": {"repoName": "string", "question": "string"},
        "scope_bindings": [{"dimension": "repository", "argument": "repoName"}],
    } | overrides


def _registration(
    server_name: str,
    tools: list[str],
    endpoint: dict[str, object] | None = None,
) -> ServerRegistration:
    return ServerRegistration.model_validate(
        {
            "server_name": server_name,
            "endpoint": endpoint if endpoint is not None else _OVER_HTTP,
            "tools": [_tool(name) for name in tools],
        }
    )


def _tenant() -> TenantId:
    return TenantId(uuid.uuid4())


@pytest.fixture
def registry(engine: AsyncEngine) -> PostgresToolRegistry:
    return PostgresToolRegistry(engine)


async def _server_rows(engine: AsyncEngine, tenant_id: TenantId) -> int:
    async with engine.connect() as conn:
        return int(
            await conn.scalar(
                sa.text(
                    "SELECT count(*) FROM tool_server WHERE tenant_id = :tenant"
                ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
                {"tenant": tenant_id},
            )
            or 0
        )


def test_the_postgres_registry_satisfies_the_port(
    registry: PostgresToolRegistry,
) -> None:
    """Conformance asserted where the binding happens, not at the first call.

    Shallow by design -- the check sees names, not signatures -- so it catches the
    adapter that never grew a method and leaves the one that grew it wrong to the tests
    below.
    """
    assert isinstance(registry, ToolRegistry)


async def test_the_composition_root_wires_a_tool_registry_that_really_writes(
    database_url: str,
) -> None:
    """`build()`'s `tool_registry` field, exercised rather than asserted about.

    Every other test here constructs the adapter itself, so nothing they do can fail if
    the composition root hands out a different one -- and the field is new, so it had no
    coverage on the path production actually takes. mypy refuses an adapter of the wrong
    *shape*; only a round trip refuses one of the wrong *table*.
    """
    platform, engine = build(database_url)
    tenant = _tenant()
    try:
        await platform.tool_registry.register(
            tenant, _registration("deepwiki", ["ask_question"])
        )
        found = await platform.tool_registry.lookup(tenant, "ask_question")
    finally:
        await engine.dispose()

    assert found.server_name == "deepwiki"
    assert isinstance(found.endpoint, StreamableHttpServer)


async def test_register_writes_one_server_row_and_one_row_per_tool(
    registry: PostgresToolRegistry, engine: AsyncEngine
) -> None:
    """Three tools behind one server, which is the shape MAP-A19 turns on.

    Three rather than one: a `for` loop that inserted only the first tool would satisfy
    a single-tool registration and lose the rest of a real catalog silently.
    """
    tenant = _tenant()
    await registry.register(
        tenant,
        _registration(
            "deepwiki", ["read_wiki_structure", "read_wiki_contents", "ask_question"]
        ),
    )

    assert await _server_rows(engine, tenant) == 1
    assert [tool.name for tool in await registry.list_for_tenant(tenant)] == [
        "ask_question",
        "read_wiki_contents",
        "read_wiki_structure",
    ]


async def test_lookup_returns_the_endpoint_and_bindings_that_were_registered(
    registry: PostgresToolRegistry,
) -> None:
    """The JSON columns round-trip as values, not as their own text.

    Two servers on two transports, so a `lookup` that read the wrong row or ignored the
    stored transport would produce the other one. `parameters` is asserted as enum
    members rather than strings, because that is the difference between a parsed value
    and the text of one.
    """
    tenant = _tenant()
    await registry.register(tenant, _registration("deepwiki", ["ask_question"]))
    await registry.register(
        tenant,
        _registration("deepwiki-local", ["read_wiki_contents"], endpoint=_OVER_STDIO),
    )

    over_http = await registry.lookup(tenant, "ask_question")
    over_stdio = await registry.lookup(tenant, "read_wiki_contents")

    assert over_http.server_name == "deepwiki"
    assert isinstance(over_http.endpoint, StreamableHttpServer)
    assert over_http.endpoint.url == "https://mcp.deepwiki.com"
    assert over_http.endpoint.credential_ref == "vault/acme/deepwiki"
    assert over_http.scope_bindings == (
        ScopeBinding(dimension="repository", argument="repoName"),
    )
    assert over_http.parameters == {
        "repoName": ParameterType.STRING,
        "question": ParameterType.STRING,
    }

    assert over_stdio.server_name == "deepwiki-local"
    assert isinstance(over_stdio.endpoint, StdioServer)
    assert over_stdio.endpoint.args == (
        "-y",
        "mcp-remote",
        "https://mcp.deepwiki.com/sse",
    )
    assert over_stdio.endpoint.credential_env_var == "DEEPWIKI_TOKEN"


async def test_a_tool_keeps_the_remote_name_the_server_behind_it_uses(
    registry: PostgresToolRegistry,
) -> None:
    """The two names are stored apart, so renaming upstream leaves the Grant intact.

    The fixture makes them differ; with `remote_name == name` everywhere, an adapter
    that wrote the tenant-facing name into both columns would be indistinguishable from
    a correct one.
    """
    tenant = _tenant()
    await registry.register(
        tenant,
        ServerRegistration.model_validate(
            {
                "server_name": "deepwiki",
                "endpoint": _OVER_HTTP,
                "tools": [_tool("ask_question", remote_name="askQuestionV2")],
            }
        ),
    )

    found = await registry.lookup(tenant, "ask_question")

    assert found.name == "ask_question"
    assert found.remote_name == "askQuestionV2"


async def test_a_second_server_claiming_a_registered_tool_name_is_refused_whole(
    registry: PostgresToolRegistry, engine: AsyncEngine
) -> None:
    """Refused rather than disambiguated, and its server row is not left behind.

    A Grant naming `ask_question` cannot say which server it meant, so a second one
    offering that name is a collision and not a variant. The count is the important
    half: a registration that wrote its server row before discovering the collision
    would leave a server nothing can reach and a name the tenant cannot re-use.
    """
    tenant = _tenant()
    await registry.register(tenant, _registration("deepwiki", ["ask_question"]))

    with pytest.raises(NameAlreadyRegistered) as refused:
        await registry.register(
            tenant,
            _registration("acme-wiki", ["ask_question", "read_wiki_contents"]),
        )

    assert refused.value.kind == "tool"
    assert refused.value.names == ("ask_question",)
    assert await _server_rows(engine, tenant) == 1, (
        "the refused registration left its server row behind"
    )
    with pytest.raises(UnknownTool):
        await registry.lookup(tenant, "read_wiki_contents")


async def test_every_colliding_tool_name_is_named_and_not_only_the_first(
    registry: PostgresToolRegistry,
) -> None:
    """Two names taken, both reported -- one round trip rather than three.

    The store refuses each colliding insert on its own, so an adapter that let the
    constraint do the reporting would name whichever tool it happened to insert first
    and stop there. A tenant re-submitting a fixed catalog would then discover the
    second collision only on the next attempt, and the third on the one after.
    """
    tenant = _tenant()
    await registry.register(
        tenant, _registration("deepwiki", ["ask_question", "read_wiki_contents"])
    )

    with pytest.raises(NameAlreadyRegistered) as refused:
        await registry.register(
            tenant,
            _registration(
                "acme-wiki",
                ["read_wiki_contents", "read_wiki_structure", "ask_question"],
            ),
        )

    assert refused.value.kind == "tool"
    assert refused.value.names == ("ask_question", "read_wiki_contents")


async def test_a_repeat_of_an_identical_registration_names_the_server_not_a_tool(
    registry: PostgresToolRegistry,
) -> None:
    """Both kinds of name are taken here, and the answer has to be the useful one.

    Re-submitting a registration that already succeeded -- a retry after a dropped
    response -- collides on the server name *and* on every tool name. Reported as a tool
    collision it sends the caller hunting for a second server offering `ask_question`,
    which does not exist; reported as a server collision it says the thing that is
    actually true, that this registration was already made.
    """
    tenant = _tenant()
    registration = _registration("deepwiki", ["ask_question", "read_wiki_contents"])
    await registry.register(tenant, registration)

    with pytest.raises(NameAlreadyRegistered) as refused:
        await registry.register(tenant, registration)

    assert refused.value.kind == "server", (
        "a retry of an identical registration was reported as a tool collision; the "
        "caller is sent looking for a conflicting server that does not exist"
    )
    assert refused.value.names == ("deepwiki",)


async def test_re_registering_a_server_name_is_refused_as_a_server_collision(
    registry: PostgresToolRegistry, engine: AsyncEngine
) -> None:
    """The kind distinguishes "you already did this" from "two servers, one tool name".

    Asserted as `server` and not merely as a refusal, because what a caller does next
    differs: one is idempotency, the other is a catalog conflict to resolve.
    """
    tenant = _tenant()
    await registry.register(tenant, _registration("deepwiki", ["ask_question"]))

    with pytest.raises(NameAlreadyRegistered) as refused:
        await registry.register(tenant, _registration("deepwiki", ["ask_question_v2"]))

    assert refused.value.kind == "server"
    assert refused.value.names == ("deepwiki",)
    assert await _server_rows(engine, tenant) == 1
    with pytest.raises(UnknownTool):
        await registry.lookup(tenant, "ask_question_v2")


async def test_one_tenants_registration_is_invisible_to_another(
    registry: PostgresToolRegistry,
) -> None:
    """Both tenants register the same server and tool names, and neither sees the other.

    Same names on purpose: a tenant filter that was missing from the WHERE clause would
    return the other tenant's row and every field would look right.
    """
    owner, stranger = _tenant(), _tenant()
    await registry.register(owner, _registration("deepwiki", ["ask_question"]))
    await registry.register(
        stranger, _registration("deepwiki", ["ask_question"], endpoint=_OVER_STDIO)
    )

    assert [tool.name for tool in await registry.list_for_tenant(owner)] == [
        "ask_question"
    ]
    assert isinstance(
        (await registry.lookup(owner, "ask_question")).endpoint, StreamableHttpServer
    )
    assert isinstance(
        (await registry.lookup(stranger, "ask_question")).endpoint, StdioServer
    )


async def test_looking_up_a_name_the_tenant_never_registered_is_refused(
    registry: PostgresToolRegistry,
) -> None:
    tenant = _tenant()
    await registry.register(tenant, _registration("deepwiki", ["ask_question"]))

    with pytest.raises(UnknownTool):
        await registry.lookup(tenant, "read_wiki_contents")


async def test_every_tool_of_one_server_resolves_to_that_one_registration(
    registry: PostgresToolRegistry, engine: AsyncEngine
) -> None:
    """One server row, three tools, one endpoint -- stated once and read three times.

    This is the store-level half of "one registration serves many definitions": nothing
    a definition does can produce a second copy of the endpoint, because the endpoint
    lives on the server row and the tools point at it.
    """
    tenant = _tenant()
    await registry.register(
        tenant,
        _registration(
            "deepwiki", ["read_wiki_structure", "read_wiki_contents", "ask_question"]
        ),
    )

    tools = await registry.list_for_tenant(tenant)

    assert len(tools) == 3
    assert {tool.endpoint for tool in tools} == {tools[0].endpoint}
    assert await _server_rows(engine, tenant) == 1

    async with engine.connect() as conn:
        distinct_servers = await conn.scalar(
            sa.text(
                "SELECT count(DISTINCT server_id) FROM registered_tool"
                " WHERE tenant_id = :tenant"
            ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
            {"tenant": tenant},
        )

    assert distinct_servers == 1
