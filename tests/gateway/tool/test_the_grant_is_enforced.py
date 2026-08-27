"""A Session reaches the tools its Grant names, and no others.

`SessionRecord.grant` has been written at Session creation since that route existed
and read by nothing, so every Session could reach every tool its tenant had ever
registered. A tenant with one safe server and one dangerous one had no way to give a
Session only the first.

Two halves, graded independently. What a Session is *offered* is not a control on its
own: a model that has seen a name once can call it without listing again, so a listing
filter alone would be a suggestion. The call path is checked here without going through
the listing at all.

The empty Grant is the case worth being deliberate about. It means *no tools*, not *all
tools*. A Session created without naming any is one nobody decided to give tools to, and
reading that as full access would make the safest-looking request the widest one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast
from uuid import uuid4

from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from tool_gateway_harness import CountingVault as _Vault
from tool_gateway_harness import FixedScope, broker, capture, stdio_endpoint

from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.registration.advertised_name import advertised_name_for
from managed_agent.core.registration.scope_binding import (
    ParameterType,
    RegisteredTool,
    ScopeBinding,
    ServerEndpoint,
    ServerName,
    UnknownTool,
)
from managed_agent.gateway.tool.mcp_proxy import McpProxy, SessionUpstreams

TENANT = TenantId(uuid4())
SERVER = "papers"
SAFE = "search"
DANGEROUS = "delete_everything"


def _tool(name: str) -> RegisteredTool:
    return RegisteredTool(
        name=name,
        advertised_name=advertised_name_for(SERVER, name),
        remote_name=name,
        parameters={"query": ParameterType.STRING},
        scope_bindings=(ScopeBinding(dimension="account", argument="query"),),
        server_name=SERVER,
        endpoint=stdio_endpoint(),
    )


class _Registry:
    """Every tool the tenant registered, which is what the Grant narrows."""

    def __init__(self, *names: str) -> None:
        self.tools = [_tool(name) for name in names]

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        for tool in self.tools:
            if tool.advertised_name == tool_name:
                return tool
        raise UnknownTool(tool_name)

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        return self.tools


class _Session:
    """An upstream that answers anything, so a refusal here is the proxy's own."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(
        self, cursor: str | None = None, **_: object
    ) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name=name,
                    description="the server's own words",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )
                for name in (SAFE, DANGEROUS)
            ]
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object] | None = None,
        **_: object,
    ) -> CallToolResult:
        self.calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="ok")])


class _Upstreams(SessionUpstreams):
    def __init__(self, session: _Session) -> None:
        super().__init__(
            tenant_id=TENANT,
            broker=broker(_Vault()),
            elicitation=None,  # type: ignore[arg-type]
        )
        self.session = session

    async def session_for(
        self, server_name: ServerName, endpoint: ServerEndpoint
    ) -> ClientSession:
        return cast(ClientSession, self.session)


def _proxy(registry: _Registry, upstreams: _Upstreams, *grant: str) -> McpProxy:
    return McpProxy(
        scopes=FixedScope(*grant),
        tenant_id=TENANT,
        session_id=SessionId(uuid4()),
        registry=registry,
        upstreams=upstreams,
        channel=cast("McpProxy", None),  # type: ignore[arg-type]
        evidence=capture(),
    )


async def test_a_session_is_offered_only_the_tools_its_grant_names() -> None:
    """The dangerous tool is registered to the tenant and absent from this listing."""
    registry = _Registry(SAFE, DANGEROUS)
    upstreams = _Upstreams(_Session())
    proxy = _proxy(registry, upstreams, advertised_name_for(SERVER, SAFE))

    offered = {tool.name for tool in await proxy.list_tools()}

    assert offered == {advertised_name_for(SERVER, SAFE)}


async def test_an_ungranted_tool_cannot_be_called_though_it_was_never_offered() -> None:
    """The control is on the call, not on the listing.

    A model that saw this name in an earlier Session, or guessed it, calls it directly.
    If the only filter were on the listing that call would succeed -- so this case never
    lists at all, and asserts the upstream was never reached.
    """
    registry = _Registry(SAFE, DANGEROUS)
    session = _Session()
    proxy = _proxy(registry, _Upstreams(session), advertised_name_for(SERVER, SAFE))

    result = await proxy.call_tool(advertised_name_for(SERVER, DANGEROUS), {})

    assert result.is_error
    assert session.calls == []


async def test_an_ungranted_name_and_an_unregistered_one_answer_the_same() -> None:
    """Two answers would let a model map the tenant's whole inventory by probing.

    Calling names and reading which refusal came back is a working enumeration oracle,
    and the tenant's tool inventory is not this Session's to learn.
    """
    registry = _Registry(SAFE, DANGEROUS)
    proxy = _proxy(registry, _Upstreams(_Session()), advertised_name_for(SERVER, SAFE))

    ungranted = await proxy.call_tool(advertised_name_for(SERVER, DANGEROUS), {})
    unregistered = await proxy.call_tool(
        advertised_name_for(SERVER, "never_registered"), {}
    )

    assert ungranted.is_error and unregistered.is_error
    assert _code(ungranted) == _code(unregistered) == ErrorCode.TOOL_NOT_GRANTED.value


async def test_an_empty_grant_reaches_nothing_rather_than_everything() -> None:
    """The safest-looking request must not be the widest one."""
    registry = _Registry(SAFE, DANGEROUS)
    session = _Session()
    proxy = _proxy(registry, _Upstreams(session))

    assert await proxy.list_tools() == []
    result = await proxy.call_tool(advertised_name_for(SERVER, SAFE), {})
    assert result.is_error
    assert session.calls == []


async def test_a_granted_tool_still_reaches_its_server() -> None:
    """The negative cases above prove nothing without this one."""
    registry = _Registry(SAFE, DANGEROUS)
    session = _Session()
    proxy = _proxy(registry, _Upstreams(session), advertised_name_for(SERVER, SAFE))

    result = await proxy.call_tool(advertised_name_for(SERVER, SAFE), {})

    assert not result.is_error
    assert session.calls == [SAFE]


def _code(result: CallToolResult) -> str:
    """The error code out of a refusal, without the correlation id that differs.

    Compared as a code rather than as a whole message because every refusal carries a
    fresh correlation id -- two identical refusals never render identically, so a string
    comparison would fail on the one field that is supposed to differ.
    """
    body = "".join(
        part.text for part in result.content if isinstance(part, TextContent)
    )
    return cast(str, json.loads(body)["code"])
