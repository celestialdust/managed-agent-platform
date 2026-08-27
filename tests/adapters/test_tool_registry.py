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
from managed_agent.core.registration.advertised_name import advertised_name_for
from managed_agent.core.registration.scope_binding import (
    NameAlreadyRegistered,
    ParameterType,
    ScopeBinding,
    ServerRegistration,
    StdioServer,
    StreamableHttpServer,
    UnknownTool,
)
from managed_agent.core.registration.tool_names import ServerName, ToolName

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


def _advertised(server_name: str, tool_name: str) -> str:
    """What `lookup` is keyed on: the pair, joined.

    Every call below goes through this rather than through a literal, so a case reads
    as "the `ask_question` behind `deepwiki`" and not as a string somebody has to parse
    back into a pair. It is also the one place a change to the join would land.
    """
    return advertised_name_for(ServerName(server_name), ToolName(tool_name))


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
        found = await platform.tool_registry.lookup(
            tenant, _advertised("deepwiki", "ask_question")
        )
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

    over_http = await registry.lookup(tenant, _advertised("deepwiki", "ask_question"))
    over_stdio = await registry.lookup(
        tenant, _advertised("deepwiki-local", "read_wiki_contents")
    )

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

    found = await registry.lookup(tenant, _advertised("deepwiki", "ask_question"))

    assert found.name == "ask_question"
    assert found.remote_name == "askQuestionV2"


async def test_a_second_server_may_offer_a_tool_name_the_first_already_uses(
    registry: PostgresToolRegistry, engine: AsyncEngine
) -> None:
    """The rule this revision changed, asserted from the tenant's side.

    Until `0032` the tenant-facing name was the key, so one tenant could hold exactly
    one `ask_question` however many servers it registered. That is not how tool names
    are distributed in the world -- `search`, `list_issues` and `get_file` are named the
    same by everybody -- and with no route that renames or removes a registered tool,
    the second server's only way forward was a permanent `ask_question_2`.

    A tool is the pair now, so both rows exist and each resolves to its own server. The
    endpoints are asserted rather than the names, because equal names proving nothing is
    exactly the failure this replaces: two rows that both resolved to the first server
    would satisfy any assertion about names.
    """
    tenant = _tenant()
    await registry.register(tenant, _registration("deepwiki", ["ask_question"]))

    await registry.register(
        tenant,
        _registration("acme-wiki", ["ask_question"], endpoint=_OVER_STDIO),
    )

    assert await _server_rows(engine, tenant) == 2
    theirs = await registry.lookup(tenant, _advertised("deepwiki", "ask_question"))
    ours = await registry.lookup(tenant, _advertised("acme-wiki", "ask_question"))
    assert theirs.name == ours.name == "ask_question"
    assert isinstance(theirs.endpoint, StreamableHttpServer)
    assert isinstance(ours.endpoint, StdioServer)


async def test_a_second_server_of_the_same_name_still_collides_on_every_tool(
    registry: PostgresToolRegistry, engine: AsyncEngine
) -> None:
    """What per-server scoping did *not* loosen, and why it must not.

    The guarantee the Agent Runtime depends on is over the *advertised* name, not the
    bare one: it appends a SHA1-derived suffix when two names it is handed would
    sanitize to one, and a Grant written against the original then resolves to nothing.
    So `deepwiki__ask_question` must still be unique within the tenant. Here the server
    name is what repeats, which makes every joined name repeat with it.

    Reported as a *server* collision rather than a tool one, and the order of the two
    checks in `register` is what decides that. Both kinds of name are taken, and "this
    server is already registered" is the true reading; "this tool name is taken" would
    send the caller hunting for a second server offering it.
    """
    tenant = _tenant()
    await registry.register(tenant, _registration("deepwiki", ["ask_question"]))

    with pytest.raises(NameAlreadyRegistered) as refused:
        await registry.register(
            tenant,
            _registration("deepwiki", ["ask_question", "read_wiki_contents"]),
        )

    assert refused.value.kind == "server"
    assert refused.value.names == ("deepwiki",)
    assert await _server_rows(engine, tenant) == 1, (
        "the refused registration left its server row behind"
    )
    with pytest.raises(UnknownTool):
        await registry.lookup(tenant, _advertised("deepwiki", "read_wiki_contents"))


async def test_two_pairs_that_advertise_one_name_are_refused_by_the_joined_name(
    registry: PostgresToolRegistry, engine: AsyncEngine
) -> None:
    """The guarantee per-server scoping had to keep, and the refusal that keeps it.

    `ServerName` admits `_`, so the join is not injective by shape: server `a` offering
    `b__c` and server `a__b` offering `c` both advertise `a__b__c`. Nothing about the
    patterns prevents that and nothing should -- forbidding `_` in a server name would
    refuse names tenants legitimately use. What must not happen is both rows existing,
    because the Agent Runtime would then be handed one name for two tools and its
    collision suffix would start rewriting names Grants were written against.

    So the second registration is refused, and refused by the *advertised* name. Told
    `c is already registered`, a tenant that holds no other `c` has been sent looking
    for something that does not exist; told `a__b__c`, they can see which pair took it.

    The server row is asserted absent as well, because the refusal happens before the
    server insert and a partial write here would leave a server nothing can reach.
    """
    tenant = _tenant()
    await registry.register(tenant, _registration("a", ["b__c"]))

    with pytest.raises(NameAlreadyRegistered) as refused:
        await registry.register(tenant, _registration("a__b", ["c"]))

    assert refused.value.kind == "tool"
    assert refused.value.names == ("a__b__c",)
    assert await _server_rows(engine, tenant) == 1, (
        "the refused registration left its server row behind"
    )


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
        await registry.lookup(tenant, _advertised("deepwiki", "ask_question_v2"))


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
        (
            await registry.lookup(owner, _advertised("deepwiki", "ask_question"))
        ).endpoint,
        StreamableHttpServer,
    )
    assert isinstance(
        (
            await registry.lookup(stranger, _advertised("deepwiki", "ask_question"))
        ).endpoint,
        StdioServer,
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
