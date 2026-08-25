"""Listing across several servers: one that is down must not speak, or silence others.

Tier 1 (local, no infrastructure). The servers are fakes here on purpose, and it is the
one place in this slice where that is the right call: the property under test is what
happens when a registered server fails at a *specific* moment — while its connection is
being opened, which is where an unreachable or mis-credentialed one actually fails — and
a real server cannot be asked to fail there on demand.

Two things are asserted together every time, because either alone is satisfiable by
breaking the other: the healthy server's list still comes back, **and** the failing
server's own words appear nowhere in it. The SDK forwards a handler exception's `str()`
verbatim as the JSON-RPC error message, so an unguarded listing hands the model whatever
the upstream said — which has been observed carrying an internal hostname and a database
username.
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
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    TextResourceContents,
    Tool,
)
from tool_gateway_harness import (
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
    _MAX_LIST_PAGES,
    McpProxy,
    SessionUpstreams,
)

LEAKY = "psycopg2: FATAL password auth failed for 'acme_ro' at internal-host-7.corp"
TENANT = TenantId(uuid4())


class FakeSession:
    """Only the five methods the proxy calls, each answering from a script."""

    def __init__(
        self,
        tool_pages: Sequence[tuple[Sequence[Tool], str | None]] = (),
        resources: Sequence[Resource] = (),
        templates: Sequence[ResourceTemplate] = (),
        contents: str = "",
    ) -> None:
        self.tool_pages = list(tool_pages)
        self.resources = list(resources)
        self.templates = list(templates)
        self.contents = contents
        self.pages_drawn = 0
        self.read: list[str] = []

    async def list_tools(
        self, *, params: PaginatedRequestParams | None = None
    ) -> ListToolsResult:
        index = self.pages_drawn
        self.pages_drawn += 1
        if index >= len(self.tool_pages):
            return ListToolsResult(tools=[])
        tools, cursor = self.tool_pages[index]
        return ListToolsResult(tools=list(tools), next_cursor=cursor)

    async def list_resources(
        self, *, params: PaginatedRequestParams | None = None
    ) -> ListResourcesResult:
        return ListResourcesResult(resources=self.resources)

    async def list_resource_templates(
        self, *, params: PaginatedRequestParams | None = None
    ) -> ListResourceTemplatesResult:
        return ListResourceTemplatesResult(resource_templates=self.templates)

    async def read_resource(self, uri: str) -> ReadResourceResult:
        self.read.append(uri)
        return ReadResourceResult(
            contents=[TextResourceContents(uri=uri, text=self.contents)]
        )


class ScriptedUpstreams(SessionUpstreams):
    """Hands back a scripted session per server, or raises the scripted failure."""

    def __init__(self, scripted: dict[str, FakeSession | Exception]) -> None:
        super().__init__(
            tenant_id=TENANT, broker=broker(CountingVault()), elicitation=_decline
        )
        self.scripted = scripted
        self.asked: list[str] = []

    async def session_for(
        self, server_name: ServerName, endpoint: ServerEndpoint
    ) -> ClientSession:
        self.asked.append(server_name)
        answer = self.scripted[server_name]
        if isinstance(answer, Exception):
            raise answer
        return cast(ClientSession, answer)


class ListingRegistry:
    """A registry over a fixed tool set, counting every read of it."""

    def __init__(self, tools: Sequence[RegisteredTool]) -> None:
        self.tools = list(tools)
        self.reads = 0

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise UnknownTool(tool_name)

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        self.reads += 1
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


def _remote(
    name: str,
    description: str = "the server's own words",
    output_schema: dict[str, object] | None = None,
) -> Tool:
    return Tool(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        output_schema=output_schema,
    )


def _proxy(registry: ListingRegistry, upstreams: ScriptedUpstreams) -> McpProxy:
    return McpProxy(
        scopes=FixedScope(),
        tenant_id=TENANT,
        session_id=SessionId(uuid4()),
        registry=registry,
        upstreams=upstreams,
        channel=cast("McpProxy", None),  # type: ignore[arg-type]
        evidence=capture(),
    )


async def test_a_server_that_cannot_be_opened_neither_empties_nor_speaks_in_tools() -> (
    None
):
    """The failure is at `session_for`, which is where an unreachable server fails."""
    registry = ListingRegistry(
        [_tool("healthy_tool", "up"), _tool("dead_tool", "down")]
    )
    upstreams = ScriptedUpstreams(
        {
            "up": FakeSession(tool_pages=[([_remote("healthy_tool")], None)]),
            "down": MCPError(code=0, message=LEAKY),
        }
    )

    offered = await _proxy(registry, upstreams).list_tools()

    assert [tool.name for tool in offered] == ["healthy_tool"]
    assert LEAKY not in repr(offered)
    assert "internal-host-7.corp" not in repr(offered)


async def test_a_server_that_cannot_be_opened_does_not_empty_the_resource_list() -> (
    None
):
    registry = ListingRegistry([_tool("a", "up"), _tool("b", "down")])
    upstreams = ScriptedUpstreams(
        {
            "up": FakeSession(resources=[Resource(uri="db://up/x", name="x")]),
            "down": MCPError(code=0, message=LEAKY),
        }
    )

    found = await _proxy(registry, upstreams).list_resources()

    assert [r.uri for r in found] == ["db://up/x"]
    assert "internal-host-7.corp" not in repr(found)


async def test_a_server_that_cannot_be_opened_does_not_empty_the_template_list() -> (
    None
):
    registry = ListingRegistry([_tool("a", "up"), _tool("b", "down")])
    upstreams = ScriptedUpstreams(
        {
            "up": FakeSession(
                templates=[ResourceTemplate(uri_template="db://up/{id}", name="x")]
            ),
            "down": MCPError(code=0, message=LEAKY),
        }
    )

    found = await _proxy(registry, upstreams).list_resource_templates()

    assert [t.uri_template for t in found] == ["db://up/{id}"]
    assert "internal-host-7.corp" not in repr(found)


async def test_a_listing_follows_the_cursor_to_the_end() -> None:
    """A registered tool in a lost tail reads to the agent as a tool that never was."""
    registry = ListingRegistry(
        [_tool("one", "up"), _tool("two", "up"), _tool("three", "up")]
    )
    upstreams = ScriptedUpstreams(
        {
            "up": FakeSession(
                tool_pages=[
                    ([_remote("one")], "c1"),
                    ([_remote("two")], "c2"),
                    ([_remote("three")], None),
                ]
            )
        }
    )

    offered = await _proxy(registry, upstreams).list_tools()

    assert sorted(tool.name for tool in offered) == ["one", "three", "two"]


async def test_a_server_cursoring_forever_is_stopped_and_said_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bounded because a listing is not inside a tool call's deadline."""
    registry = ListingRegistry([_tool("one", "up")])
    endless = FakeSession(tool_pages=[([_remote("one")], "always")] * 10_000)
    upstreams = ScriptedUpstreams({"up": endless})

    with caplog.at_level(
        logging.WARNING, logger="managed_agent.gateway.tool.mcp_proxy"
    ):
        offered = await _proxy(registry, upstreams).list_tools()

    assert endless.pages_drawn == _MAX_LIST_PAGES
    assert [tool.name for tool in offered] == ["one"]
    assert "page cap" in caplog.text


async def test_a_registered_tool_the_server_no_longer_offers_is_not_advertised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Advertising it produces a call that fails at the server, which reads as us."""
    registry = ListingRegistry([_tool("gone", "up", remote="renamed_away")])
    upstreams = ScriptedUpstreams(
        {"up": FakeSession(tool_pages=[([_remote("something_else")], None)])}
    )

    with caplog.at_level(
        logging.WARNING, logger="managed_agent.gateway.tool.mcp_proxy"
    ):
        offered = await _proxy(registry, upstreams).list_tools()

    assert offered == []
    assert "renamed_away" in caplog.text


async def test_a_tool_is_offered_under_its_registered_name_and_the_servers_schema() -> (
    None
):
    """The registry decides the name; the server decides the rest of what is offered.

    Bar one exception, which the case below pins: the upstream's output schema is not
    forwarded.
    """
    registry = ListingRegistry([_tool("invoice_lookup", "up", remote="lookup")])
    upstreams = ScriptedUpstreams(
        {"up": FakeSession(tool_pages=[([_remote("lookup", "find an invoice")], None)])}
    )

    offered = await _proxy(registry, upstreams).list_tools()

    assert [tool.name for tool in offered] == ["invoice_lookup"]
    assert offered[0].description == "find an invoice"
    assert offered[0].input_schema == _remote("lookup").input_schema


async def test_the_upstreams_output_schema_is_not_offered_to_the_runtime() -> None:
    """The one thing the server does not get to decide, and the reason is the capture.

    A declared output schema obliges every successful call to carry structured content
    conforming to it, and the caller's MCP client enforces that unasked. A capture
    replaces the result, so the obligation is one this Gateway cannot meet for any tool
    whose output may be large -- which is every tool. Offering the schema anyway makes a
    large result from this server a raised exception in the Agent Runtime instead of an
    Evidence reference, and lets the upstream opt out of capture by declaring one.
    `tests/gateway/tool/test_gateway_front_door.py` measures that end to end; this pins
    where the schema is dropped.
    """
    schema: dict[str, object] = {"type": "object", "properties": {}}
    registry = ListingRegistry([_tool("invoice_lookup", "up", remote="lookup")])
    upstreams = ScriptedUpstreams(
        {
            "up": FakeSession(
                tool_pages=[([_remote("lookup", output_schema=schema)], None)]
            )
        }
    )

    offered = await _proxy(registry, upstreams).list_tools()

    assert [tool.output_schema for tool in offered] == [None]
    assert offered[0].input_schema == _remote("lookup").input_schema


async def test_a_uri_matching_a_template_routes_to_that_templates_server() -> None:
    """Longest prefix wins, so a broad publisher cannot capture a narrow one's URI."""
    registry = ListingRegistry([_tool("a", "broad"), _tool("b", "narrow")])
    broad = FakeSession(
        templates=[ResourceTemplate(uri_template="db://acme/{rest}", name="broad")],
        contents="the broad server answered",
    )
    narrow = FakeSession(
        templates=[
            ResourceTemplate(uri_template="db://acme/invoices/{id}", name="narrow")
        ],
        contents="the narrow server answered",
    )
    proxy = _proxy(registry, ScriptedUpstreams({"broad": broad, "narrow": narrow}))

    read = await proxy.read_resource("db://acme/invoices/9")

    assert [getattr(c, "text", None) for c in read.contents] == [
        "the narrow server answered"
    ]
    assert broad.read == []


async def test_an_unindexed_uri_is_refused_after_exactly_one_refresh() -> None:
    """Trying each server in turn would answer "does this exist anywhere" for free."""
    registry = ListingRegistry([_tool("a", "up")])
    upstreams = ScriptedUpstreams({"up": FakeSession()})
    proxy = _proxy(registry, upstreams)

    with pytest.raises(MCPError) as caught:
        await proxy.read_resource("db://acme/never-listed")

    assert caught.value.message == "the platform refused this read"
    assert isinstance(caught.value.data, dict)
    assert caught.value.data["code"] == "tool.not_granted"
    # One refresh is `list_resources` plus `list_resource_templates`, each reading the
    # registry once. A second miss must not refresh again.
    assert registry.reads == 2


_LISTINGS: Final = ("list_tools", "list_resources", "list_resource_templates")
"""Every listing method, so the case below grades each rather than the first.

Parametrized rather than written once against `list_tools` because each method carries
its own copy of the count-and-skip loop, so a guard added to one and left out of another
is a difference no single-method case can see.
"""


@pytest.mark.parametrize("listing", _LISTINGS)
async def test_a_listing_that_reached_no_server_refuses_instead_of_reading_empty(
    listing: str,
) -> None:
    """A total outage must not arrive as `[]`, which is what "nothing registered" is.

    The three cases above prove a listing survives losing one server. This proves the
    other edge: when every server fails there is nothing honest to return, because an
    empty list is indistinguishable from a tenant that registered nothing, and an Agent
    Runtime that cannot tell those apart carries on silently toolless.
    """
    registry = ListingRegistry([_tool("a", "down"), _tool("b", "also_down")])
    upstreams = ScriptedUpstreams(
        {
            "down": MCPError(code=0, message=LEAKY),
            "also_down": MCPError(code=0, message=LEAKY),
        }
    )

    with pytest.raises(MCPError) as raised:
        await getattr(_proxy(registry, upstreams), listing)()

    assert LEAKY not in repr(raised.value)
    assert "internal-host-7.corp" not in repr(raised.value)


@pytest.mark.parametrize("listing", _LISTINGS)
async def test_a_tenant_that_registered_nothing_is_told_so_and_not_refused(
    listing: str,
) -> None:
    """The guard's other arm: an empty list is the truth here, so it must be returned.

    Without this, a guard written as "empty means something broke" would refuse every
    tenant during onboarding -- before anything is registered there is no server to
    fail, and no failure is exactly what makes this empty list honest.
    """
    proxy = _proxy(ListingRegistry([]), ScriptedUpstreams({}))

    assert await getattr(proxy, listing)() == []
