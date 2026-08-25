"""The one line a tool listing leaves behind, and what each version of it says.

Tier 1 (local, no infrastructure). The upstream servers are fakes here because what is
under test is the *log*, and the three ways a catalogue comes back short -- nothing
registered, every tool dropped, some tools dropped -- have to be produced on demand
rather than waited for.

Every case asserts the same two things: exactly one census line was emitted, and it is
the one describing what happened. The count is half the property. A census that
sometimes says nothing is one a reader cannot tell from a healthy listing, and that
indistinguishability is the whole failure these cases exist to close.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final, cast
from uuid import uuid4

import pytest
from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.types import (
    ElicitRequestParams,
    ElicitResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
)
from tool_gateway_harness import (
    CREDENTIAL_REF,
    STDIO_SERVER,
    CountingVault,
    FixedScope,
    broker,
    capture,
    stdio_endpoint,
)

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.registration.scope_binding import (
    ParameterType,
    RegisteredTool,
    ScopeBinding,
    ServerEndpoint,
    ServerName,
    UnknownTool,
)
from managed_agent.gateway.tool.mcp_proxy import (
    McpProxy,
    SessionChannel,
    SessionUpstreams,
)

PROXY_LOGGER: Final = "managed_agent.gateway.tool.mcp_proxy"
TENANT: Final = TenantId(uuid4())


class OnePageSession:
    """Answers one page of tools, which is all a census case needs to ask for."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        self.tools = list(tools)

    async def list_tools(
        self, *, params: PaginatedRequestParams | None = None
    ) -> ListToolsResult:
        return ListToolsResult(tools=list(self.tools))


class ScriptedUpstreams(SessionUpstreams):
    """Hands back a scripted session per server, or raises the scripted failure."""

    def __init__(self, scripted: dict[str, OnePageSession | Exception]) -> None:
        super().__init__(
            tenant_id=TENANT, broker=broker(CountingVault()), elicitation=_decline
        )
        self.scripted = scripted

    async def session_for(
        self, server_name: ServerName, endpoint: ServerEndpoint
    ) -> ClientSession:
        answer = self.scripted[server_name]
        if isinstance(answer, Exception):
            raise answer
        return cast(ClientSession, answer)


class FixedRegistry:
    """A registry over a fixed tool set, answering both reads the proxy makes."""

    def __init__(self, tools: Sequence[RegisteredTool]) -> None:
        self.tools = list(tools)

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise UnknownTool(tool_name)

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        return self.tools


async def _decline(params: ElicitRequestParams) -> ElicitResult:
    return ElicitResult(action="decline")


def _tool(name: str, server: str, remote: str | None = None) -> RegisteredTool:
    return RegisteredTool(
        name=name,
        remote_name=remote or name,
        parameters={"query": ParameterType.STRING},
        scope_bindings=(ScopeBinding(dimension="account", argument="query"),),
        server_name=server,
        endpoint=stdio_endpoint(),
    )


def _remote(name: str) -> Tool:
    return Tool(
        name=name,
        description="the server's own words",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )


def _proxy(registry: FixedRegistry, upstreams: ScriptedUpstreams) -> McpProxy:
    return McpProxy(
        scopes=FixedScope(),
        tenant_id=TENANT,
        session_id=SessionId(uuid4()),
        registry=registry,
        upstreams=upstreams,
        channel=cast(SessionChannel, None),
        evidence=capture(),
    )


def _census(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    """The one census record, or a failure quoting however many there were instead."""
    lines = [r for r in caplog.records if r.getMessage().startswith("offer census")]
    assert len(lines) == 1, [r.getMessage() for r in lines]
    return lines[0]


async def test_a_tenant_with_nothing_registered_is_named_rather_than_read_as_healthy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The empty that logged nothing at all, which is the case this census is for.

    The loop that carries every other line in this listing does not run when the
    registry comes back empty, so before this the call was silent -- and silence is
    also what a healthy listing sounds like.
    """
    proxy = _proxy(FixedRegistry([]), ScriptedUpstreams({}))

    with caplog.at_level(logging.INFO, logger=PROXY_LOGGER):
        assert await proxy.list_tools() == []

    line = _census(caplog)
    assert line.levelno == logging.WARNING
    assert str(TENANT) in line.getMessage()
    assert "registered=0" in line.getMessage()


async def test_every_registered_tool_being_dropped_is_the_loudest_line_and_names_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tenant registered two tools and the Session was handed none of them."""
    registry = FixedRegistry(
        [_tool("invoices", "up", remote="gone"), _tool("ledger", "up", remote="also")]
    )
    upstreams = ScriptedUpstreams({"up": OnePageSession([_remote("something_else")])})

    with caplog.at_level(logging.INFO, logger=PROXY_LOGGER):
        assert await _proxy(registry, upstreams).list_tools() == []

    line = _census(caplog)
    assert line.levelno == logging.ERROR
    assert "registered=2" in line.getMessage()
    assert "invoices" in line.getMessage()
    assert "ledger" in line.getMessage()


async def test_a_partial_drop_names_the_tools_that_were_lost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two registered, one offered -- and the census says which one went missing."""
    registry = FixedRegistry(
        [_tool("kept", "up"), _tool("lost", "up", remote="renamed_away")]
    )
    upstreams = ScriptedUpstreams({"up": OnePageSession([_remote("kept")])})

    with caplog.at_level(logging.INFO, logger=PROXY_LOGGER):
        offered = await _proxy(registry, upstreams).list_tools()

    assert [tool.name for tool in offered] == ["kept"]
    line = _census(caplog)
    assert line.levelno == logging.WARNING
    assert "registered=2" in line.getMessage()
    assert "offered=1" in line.getMessage()
    assert "lost" in line.getMessage()


async def test_a_healthy_listing_still_leaves_one_line_and_leaves_it_cheaply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The counts a reader confirms the catalogue against, at a level worth keeping."""
    registry = FixedRegistry([_tool("kept", "up"), _tool("also_kept", "up")])
    upstreams = ScriptedUpstreams(
        {"up": OnePageSession([_remote("kept"), _remote("also_kept")])}
    )

    with caplog.at_level(logging.INFO, logger=PROXY_LOGGER):
        offered = await _proxy(registry, upstreams).list_tools()

    assert sorted(tool.name for tool in offered) == ["also_kept", "kept"]
    line = _census(caplog)
    assert line.levelno == logging.INFO
    assert "registered=2" in line.getMessage()
    assert "offered=2" in line.getMessage()


async def test_a_listing_that_reached_no_server_is_counted_before_it_refuses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The refusal is not the census, and the census must survive it.

    A listing that raises has still read the registry, and how much it was carrying
    when every server refused is the difference between one lost tool and forty. If
    the count came after the guard, this call would be the one call that says nothing.
    """
    registry = FixedRegistry([_tool("a", "down"), _tool("b", "also_down")])
    upstreams = ScriptedUpstreams(
        {
            "down": MCPError(code=0, message="unreachable"),
            "also_down": MCPError(code=0, message="unreachable"),
        }
    )

    with (
        caplog.at_level(logging.INFO, logger=PROXY_LOGGER),
        pytest.raises(MCPError),
    ):
        await _proxy(registry, upstreams).list_tools()

    line = _census(caplog)
    assert line.levelno == logging.ERROR
    assert "registered=2" in line.getMessage()


async def test_the_census_names_tools_and_servers_and_never_an_endpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool name is the tenant's vocabulary; an endpoint is how to reach it.

    The registration this census reads carries the command a stdio server is spawned
    with and the vault entry its credential is read from. Neither belongs in a line
    that exists to be grepped out of a pod's log and pasted into a ticket.
    """
    registry = FixedRegistry([_tool("invoices", "up", remote="gone")])
    upstreams = ScriptedUpstreams({"up": OnePageSession([_remote("something_else")])})

    with caplog.at_level(logging.INFO, logger=PROXY_LOGGER):
        await _proxy(registry, upstreams).list_tools()

    message = _census(caplog).getMessage()
    assert "invoices" in message
    assert CREDENTIAL_REF not in message
    assert str(STDIO_SERVER) not in message
