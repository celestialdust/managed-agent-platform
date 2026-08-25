"""What a narrowed call actually carries, proved against a server on the other end.

Tier 1 (local, no infrastructure, one real subprocess). The clamp's own unit tests
state the property about a return value; these state it about the bytes that left. The
server is the conformance stdio server speaking the real protocol down a pipe, and its
`echo_arguments` tool hands back exactly what the call carried — so what is asserted
here is the outbound request, not a fake's recollection of one.

The refusal case asserts something the result alone cannot show: that **the upstream
was never opened**. A refusal returned after the credential was fetched and the child
spawned would look identical to the agent, and it is not the same thing at all — the
whole reason the clamp sits ahead of `session_for` is that a call which will not be
made should not read a secret on its way to being refused. The counting vault is what
makes that visible, because a transport this proxy never opens is a fetch that never
happened.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Final
from uuid import uuid4

import pytest
from mcp.types import CallToolResult, ElicitRequestParams, ElicitResult
from tool_gateway_harness import (
    CREDENTIAL_ENV_VAR,
    CREDENTIAL_REF,
    STDIO_SERVER,
    TENANT,
    CountingVault,
    broker,
    capture,
)

from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.registration.scope_binding import (
    ParameterType,
    RegisteredTool,
    ScopeBinding,
    StdioServer,
    UnknownTool,
)
from managed_agent.core.session.session import SessionRecord
from managed_agent.gateway.tool.mcp_proxy import McpProxy, SessionUpstreams

SESSION: Final = SessionId(uuid4())
IN_SCOPE: Final = "acme/widgets"
"""What the Session is bounded to. The model never gets to name anything else."""


class OneScope:
    """A Session registry that answers one Scope and counts the reads.

    The count is the whole reason this is not a lambda: the proxy is supposed to read
    a Session's Scope once and hold it, and a proxy that re-read it per call would be
    a database round trip on the hot path with nothing going red.
    """

    def __init__(self, scope: dict[str, str]) -> None:
        self.scope = scope
        self.reads = 0

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        self.reads += 1
        return SessionRecord(
            id=session_id,
            tenant_id=tenant_id,
            definition_id=DefinitionId(uuid4()),
            definition_revision="0" * 40,
            grant=frozenset({"echo_arguments"}),
            scope=tuple(self.scope.items()),
            budget_minor_units=1_000,
            budget_currency="USD",
            retention_days=1,
        )


class FixedRegistry:
    def __init__(self, tools: Sequence[RegisteredTool]) -> None:
        self.tools = list(tools)

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise UnknownTool(tool_name)

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        return self.tools


class Silent:
    """A channel nothing here listens on. No test in this file elicits or reports."""

    async def progress(
        self, call_id: str, progress: float, total: float | None, message: str | None
    ) -> None: ...

    async def ask(
        self, params: ElicitRequestParams
    ) -> ElicitResult:  # pragma: no cover - never asked
        raise AssertionError("no test here elicits")


def _echo_tool(*bindings: ScopeBinding) -> RegisteredTool:
    """`echo_arguments`, bound however the case needs, over the real stdio server."""
    return RegisteredTool(
        name="echo_arguments",
        remote_name="echo_arguments",
        parameters={
            "repo_name": ParameterType.STRING,
            "branch": ParameterType.STRING,
            "question": ParameterType.STRING,
        },
        scope_bindings=bindings
        or (ScopeBinding(dimension="repository", argument="repo_name"),),
        server_name="conformance_stdio",
        endpoint=StdioServer(
            transport="stdio",
            command="python",
            args=(str(STDIO_SERVER),),
            credential_ref=CREDENTIAL_REF,
            credential_env_var=CREDENTIAL_ENV_VAR,
        ),
    )


async def _proxy_over(
    tool: RegisteredTool, scope: dict[str, str], vault: CountingVault
) -> AsyncIterator[McpProxy]:
    upstreams = SessionUpstreams(
        tenant_id=TENANT, broker=broker(vault), elicitation=Silent().ask
    )
    owner = asyncio.create_task(upstreams.run())
    try:
        yield McpProxy(
            tenant_id=TENANT,
            session_id=SESSION,
            registry=FixedRegistry([tool]),
            upstreams=upstreams,
            channel=Silent(),
            evidence=capture(),
            scopes=OneScope(scope),
        )
    finally:
        await upstreams.aclose()
        await owner


def _echoed(result: CallToolResult) -> dict[str, object]:
    text = "".join(getattr(part, "text", "") for part in result.content)
    parsed: dict[str, object] = json.loads(text)
    return parsed


@pytest.mark.anyio
async def test_the_outbound_call_carries_the_scope_value_not_the_models() -> None:
    """The model asked about somebody else's repository and did not get it."""
    vault = CountingVault()
    async for proxy in _proxy_over(_echo_tool(), {"repository": IN_SCOPE}, vault):
        result = await proxy.call_tool(
            "echo_arguments",
            {"repo_name": "someone-else/private", "question": "what is this"},
        )

    assert result.is_error is not True, result.content
    carried = _echoed(result)
    assert carried["repo_name"] == IN_SCOPE
    assert carried["question"] == "what is this"


@pytest.mark.anyio
async def test_a_bound_argument_the_model_left_out_still_reaches_the_server() -> None:
    """Omitting the bound argument is not a way to be answered without it."""
    vault = CountingVault()
    async for proxy in _proxy_over(_echo_tool(), {"repository": IN_SCOPE}, vault):
        result = await proxy.call_tool("echo_arguments", {"question": "what is this"})

    assert result.is_error is not True, result.content
    assert _echoed(result)["repo_name"] == IN_SCOPE


@pytest.mark.anyio
async def test_a_session_scope_missing_the_dimension_is_refused() -> None:
    """The refusal names the tool and the dimension, and carries no Scope value."""
    vault = CountingVault()
    async for proxy in _proxy_over(_echo_tool(), {"unrelated": "value"}, vault):
        result = await proxy.call_tool(
            "echo_arguments", {"repo_name": "anything", "question": "what"}
        )

    assert result.is_error is True
    said = "".join(getattr(part, "text", "") for part in result.content)
    assert ErrorCode.TOOL_OUT_OF_SCOPE.value in said
    assert "repository" in said
    assert "echo_arguments" in said


@pytest.mark.anyio
async def test_a_refused_call_never_opens_the_upstream_or_reads_its_credential() -> (
    None
):
    """No secret is read on the way to a refusal, because no transport is opened.

    This is the assertion the returned result cannot make. A clamp applied after
    `session_for` would refuse identically from the agent's side while having fetched
    the credential and spawned the child, and the fetch count is the only thing in
    reach that tells those two apart.
    """
    vault = CountingVault()
    async for proxy in _proxy_over(_echo_tool(), {}, vault):
        await proxy.call_tool("echo_arguments", {"question": "what"})

    assert vault.fetches == []


@pytest.mark.anyio
async def test_a_scope_that_narrows_only_one_of_two_bindings_refuses() -> None:
    """Half a narrowing is a widening, and the wide half is what would have gone out."""
    vault = CountingVault()
    two = _echo_tool(
        ScopeBinding(dimension="repository", argument="repo_name"),
        ScopeBinding(dimension="branch", argument="branch"),
    )
    async for proxy in _proxy_over(two, {"repository": IN_SCOPE}, vault):
        result = await proxy.call_tool("echo_arguments", {"question": "what"})

    assert result.is_error is True
    assert vault.fetches == []


@pytest.mark.anyio
async def test_the_session_scope_is_read_once_however_many_calls_are_made() -> None:
    """Held for the proxy's life, because no field of a `SessionRecord` is rewritten."""
    vault = CountingVault()
    scopes = OneScope({"repository": IN_SCOPE})
    upstreams = SessionUpstreams(
        tenant_id=TENANT, broker=broker(vault), elicitation=Silent().ask
    )
    owner = asyncio.create_task(upstreams.run())
    try:
        proxy = McpProxy(
            tenant_id=TENANT,
            session_id=SESSION,
            registry=FixedRegistry([_echo_tool()]),
            upstreams=upstreams,
            channel=Silent(),
            evidence=capture(),
            scopes=scopes,
        )
        for _ in range(3):
            await proxy.call_tool("echo_arguments", {"question": "what"})
    finally:
        await upstreams.aclose()
        await owner

    assert scopes.reads == 1


class UnreadableScope:
    """A Session registry that cannot answer, the way a database outage cannot."""

    def __init__(self) -> None:
        self.reads = 0

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        self.reads += 1
        raise RuntimeError("psycopg: connection to sessions at internal-host-7 refused")


@pytest.mark.anyio
async def test_a_scope_that_cannot_be_read_fails_the_call_closed_and_says_nothing() -> (
    None
):
    """Two properties, and the second is the one an outage would otherwise break.

    Fails **closed**: an unreadable Scope narrows nothing, so the call must not go out
    at the full breadth of the tenant's data because a database was down. The counting
    vault shows no upstream was opened.

    And the store's own words reach nobody. The low-level MCP server forwards a
    handler exception's `str()` verbatim, and this one carries an internal hostname --
    which is exactly the disclosure the rest of this package's error mapping exists to
    stop, now reachable through a read this slice added.
    """
    vault = CountingVault()
    scopes = UnreadableScope()
    upstreams = SessionUpstreams(
        tenant_id=TENANT, broker=broker(vault), elicitation=Silent().ask
    )
    owner = asyncio.create_task(upstreams.run())
    try:
        proxy = McpProxy(
            tenant_id=TENANT,
            session_id=SESSION,
            registry=FixedRegistry([_echo_tool()]),
            upstreams=upstreams,
            channel=Silent(),
            evidence=capture(),
            scopes=scopes,
        )
        result = await proxy.call_tool("echo_arguments", {"question": "what"})
    finally:
        await upstreams.aclose()
        await owner

    assert result.is_error is True
    assert vault.fetches == []
    said = "".join(getattr(part, "text", "") for part in result.content)
    assert "internal-host-7" not in said
    assert "psycopg" not in said
