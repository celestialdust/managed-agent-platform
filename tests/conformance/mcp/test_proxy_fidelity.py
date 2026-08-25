"""The proxy against real MCP servers, on both transports the platform allows.

Two servers, both real. The stdio half is a child process this repo spawns and speaks
the protocol to down a pipe; the Streamable HTTP half is `https://mcp.deepwiki.com/mcp`,
a third party this platform does not control.

**What the network-marked tests cost, and why they are still here.** They reach the
public internet, so they are excluded from the default run — `pytest -q` must be
deterministic and offline, and a gate that fails because somebody else's server is slow
teaches nothing. Run them with `pytest -m network tests/conformance/mcp/`. Skipping them
forever would hollow out this slice: proxying *a real third party* is the thing being
claimed, and a suite that only ever talks to a server this repo wrote proves the repo
agrees with itself.

**NOT PROVEN here, stated where it is felt.** The stdio server is written to the same
SDK the Gateway is, so a quirk of somebody else's stdio implementation is out of reach
until an stdio server is attested for this repo. And the exact bytes of the outbound
credential header are not directly observable: `StreamableHttpServer.url` requires
`https://`, so no local server can be registered to echo the request back, and deepwiki
does not return the headers it received. What *is* proven about the header is indirect
and real — deepwiki answers differently depending on the header name the registration
named, which it could not do if the registration's header were not reaching the wire.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

import pytest
from mcp.types import ElicitRequestFormParams, ElicitRequestParams, ElicitResult

from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.registration.scope_binding import (
    ParameterType,
    RegisteredTool,
    ScopeBinding,
    ServerEndpoint,
    StdioServer,
    StreamableHttpServer,
    UnknownTool,
)
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vfs.evidence import (
    CaptureAsEvidence,
    CaptureContext,
    CaptureThreshold,
    EvidenceRef,
    ReturnInline,
)
from managed_agent.gateway.tool import mcp_proxy
from managed_agent.gateway.tool.credential_broker import (
    ToolCredentialBroker,
    vault_name,
)
from managed_agent.gateway.tool.evidence_capture import EvidenceCapture
from managed_agent.gateway.tool.mcp_proxy import McpProxy, SessionUpstreams


class _NeverCaptures:
    """An `EvidenceRecorder` for a suite that is about proxy fidelity, not capture.

    The threshold beside it is the production default, well above anything the servers
    here return, so every result travels this file unsubstituted -- which is the whole
    point of a fidelity suite. `record_captured` therefore must never be reached, and
    says so rather than returning something plausible.

    Written out here rather than shared with
    `tests/gateway/tool/tool_gateway_harness.py`: pytest puts each test directory on
    `sys.path` separately and there is no import path from this package to that one.
    """

    async def record_captured(
        self,
        ctx: CaptureContext,
        payload: bytes,
        decision: CaptureAsEvidence,
        truncated_at_runtime_cap: bool = False,
    ) -> EvidenceRef:
        raise AssertionError(
            "a fidelity test returned more than the capture threshold; it is now "
            "asserting against a reference sentence rather than the server's result"
        )

    async def record_inline(self, ctx: CaptureContext, decision: ReturnInline) -> None:
        return None


STDIO_SERVER: Final[Path] = (
    Path(__file__).resolve().parent / "servers" / "stdio_server.py"
)
DEEPWIKI_URL: Final[str] = "https://mcp.deepwiki.com/mcp"
CREDENTIAL_VALUE: Final[str] = "conformance-placeholder-value"
LEAKED_HOST: Final[str] = "internal-host-7.corp"
TENANT: Final[TenantId] = TenantId(uuid4())


class PlaceholderVault:
    """One value for every reference.

    A credential-free public server still cannot be registered — `credential_ref` is
    `min_length=1` — so the conformance registration names a placeholder entry and this
    answers it. The value is fixed so a test can assert the child received exactly it.
    """

    def __init__(self, value: str = CREDENTIAL_VALUE) -> None:
        self.value = value
        self.fetches: list[str] = []

    async def fetch(self, name: str) -> str:
        self.fetches.append(name)
        return self.value


class RecordingChannel:
    """Stands in for the Session's Event Log, keeping what the proxy sent it.

    The answer to an elicitation is fixed at construction rather than computed, because
    what these tests measure is the round trip, not the decision: the Session's real
    answer comes from a human several hops away, and any logic here would only be this
    file agreeing with itself.
    """

    def __init__(self, answer: ElicitResult | None = None) -> None:
        self.reports: list[tuple[str, float, float | None, str | None]] = []
        self.asked: list[ElicitRequestParams] = []
        self.answer = answer or ElicitResult(action="decline")

    async def progress(
        self, call_id: str, progress: float, total: float | None, message: str | None
    ) -> None:
        self.reports.append((call_id, progress, total, message))

    async def ask(self, params: ElicitRequestParams) -> ElicitResult:
        self.asked.append(params)
        return self.answer


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


def stdio_server(*args: str) -> StdioServer:
    return StdioServer(
        transport="stdio",
        command=sys.executable,
        args=(str(STDIO_SERVER), *args),
        credential_ref="conformance/stdio",
        credential_env_var="MAP_CONFORMANCE_TOKEN",
    )


def deepwiki_server(
    credential_header: str = "X-Map-Conformance",
) -> StreamableHttpServer:
    """Deepwiki, registered under a header name of the test's choosing.

    The default is deliberately not `Authorization`: measured against the live endpoint,
    deepwiki accepts an `Authorization` header on `initialize` and `tools/list` and then
    answers every tool call with its "authentication is not allowed" message instead of
    the content. An unrelated header name is ignored on all three.
    """
    return StreamableHttpServer(
        transport="streamable_http",
        url=DEEPWIKI_URL,
        credential_ref="conformance/deepwiki",
        credential_header=credential_header,
    )


def tool(
    name: str, remote: str, server: str, endpoint: ServerEndpoint, argument: str
) -> RegisteredTool:
    return RegisteredTool(
        name=name,
        remote_name=remote,
        parameters={argument: ParameterType.STRING},
        scope_bindings=(ScopeBinding(dimension="account", argument=argument),),
        server_name=server,
        endpoint=endpoint,
    )


def stdio_tools(*args: str) -> list[RegisteredTool]:
    endpoint = stdio_server(*args)
    return [
        tool(name, name, "conformance_stdio", endpoint, "query")
        for name in (
            "echo_credential",
            "explode",
            "crawl",
            "sleep_forever",
            "ask_operator",
        )
    ]


DEEPWIKI_TOOL = tool(
    "repo_map", "read_wiki_structure", "deepwiki", deepwiki_server(), "repoName"
)


class FixedScope:
    """A `SessionScopeReader` answering the one Scope every tool here is bound to.

    Every registration in this file declares a Scope Binding on the `account`
    dimension, because a tool that declares none cannot be registered at all -- so a
    proxy in a fidelity suite needs a Scope for the same reason a real Session does,
    and not because anything here is about clamping.

    Written out rather than imported from `tests/gateway/tool/tool_gateway_harness.py`,
    for the reason `_NeverCaptures` above already gives: pytest puts each test
    directory on `sys.path` separately and there is no import path from this package to
    that one.
    """

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        return SessionRecord(
            id=session_id,
            tenant_id=tenant_id,
            definition_id=DefinitionId(UUID("33333333-3333-4333-8333-333333333333")),
            definition_revision="0" * 40,
            grant=frozenset(),
            scope=(("account", "the-tenants-own-account"),),
            budget_minor_units=1_000,
            budget_currency="USD",
            retention_days=1,
        )


@contextlib.asynccontextmanager
async def proxying(
    tools: Sequence[RegisteredTool],
    vault: PlaceholderVault | None = None,
    channel: RecordingChannel | None = None,
) -> AsyncIterator[McpProxy]:
    """A proxy over the given registrations, with its owning task running."""
    reporting = channel or RecordingChannel()
    upstreams = SessionUpstreams(
        tenant_id=TENANT,
        broker=ToolCredentialBroker(vault or PlaceholderVault()),
        elicitation=reporting.ask,
    )
    owner = asyncio.create_task(upstreams.run())
    try:
        yield McpProxy(
            scopes=FixedScope(),
            tenant_id=TENANT,
            session_id=SessionId(uuid4()),
            registry=FixedRegistry(tools),
            upstreams=upstreams,
            channel=reporting,
            evidence=EvidenceCapture(_NeverCaptures(), CaptureThreshold(64 * 1024)),
        )
    finally:
        await upstreams.aclose()
        await owner


def _text(content: Sequence[object]) -> str:
    return "".join(getattr(part, "text", "") for part in content)


def _envelope(content: Sequence[object]) -> dict[str, object]:
    parsed = json.loads(_text(content))
    assert isinstance(parsed, dict)
    return parsed


async def test_a_tool_is_offered_under_its_own_name_and_the_servers_shape() -> None:
    async with proxying(stdio_tools()) as proxy:
        offered = await proxy.list_tools()

    by_name = {t.name: t for t in offered}
    assert sorted(by_name) == [
        "ask_operator",
        "crawl",
        "echo_credential",
        "explode",
        "sleep_forever",
    ]
    assert by_name["echo_credential"].description == (
        "Return the credential this process was started with."
    )
    assert by_name["echo_credential"].input_schema == {
        "type": "object",
        "properties": {},
    }


async def test_only_the_registry_decides_which_tools_exist_here() -> None:
    """The server offers four; the registry names one, so one is offered."""
    async with proxying(stdio_tools()[:1]) as proxy:
        offered = await proxy.list_tools()

    assert [t.name for t in offered] == ["echo_credential"]


async def test_a_call_returns_the_servers_own_content_and_the_credential_it_got() -> (
    None
):
    vault = PlaceholderVault()

    async with proxying(stdio_tools(), vault=vault) as proxy:
        result = await proxy.call_tool("echo_credential", {})

    assert result.is_error is not True
    assert _text(result.content) == CREDENTIAL_VALUE
    assert vault.fetches == [vault_name(TENANT, "conformance/stdio")]


async def test_a_server_raising_out_of_its_handler_reaches_the_agent_as_a_code() -> (
    None
):
    """A real upstream failure over a real transport, not a constructed exception.

    The server raises a message naming an internal host. The SDK forwards that text
    verbatim as the JSON-RPC error message, so what this asserts is that the text stops
    here — and that the agent still learns the call failed.
    """
    async with proxying(stdio_tools()) as proxy:
        result = await proxy.call_tool("explode", {})

    assert result.is_error is True
    envelope = _envelope(result.content)
    assert envelope["code"] == ErrorCode.TOOL_UNAVAILABLE.value
    assert LEAKED_HOST not in _text(result.content)
    assert "conformance failure" not in _text(result.content)


async def test_a_server_that_never_answers_is_told_apart_from_one_that_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`tool.timed_out` and `tool.unavailable` are the two a tenant most needs apart.

    The real read deadline is 54 seconds, which is the right number for production and
    the wrong one for a test, so it is patched down. The constant is read at call time
    from the module rather than captured at import, which is what makes that possible
    without a fake transport.
    """
    monkeypatch.setattr(mcp_proxy, "UPSTREAM_READ_TIMEOUT_S", 0.75)

    async with proxying(stdio_tools()) as proxy:
        result = await proxy.call_tool("sleep_forever", {})

    assert result.is_error is True
    assert _envelope(result.content)["code"] == ErrorCode.TOOL_TIMED_OUT.value


async def test_a_registered_command_that_dies_on_startup_is_an_unavailable_server() -> (
    None
):
    dies = StdioServer(
        transport="stdio",
        command=sys.executable,
        args=("-c", "raise SystemExit(3)"),
        credential_ref="conformance/stdio",
        credential_env_var="MAP_CONFORMANCE_TOKEN",
    )
    registered = tool("doomed", "doomed", "dead_server", dies, "query")

    async with proxying([registered]) as proxy:
        result = await proxy.call_tool("doomed", {})

    assert result.is_error is True
    assert _envelope(result.content)["code"] == ErrorCode.TOOL_UNAVAILABLE.value


async def test_an_unregistered_name_is_refused_in_the_same_shape_as_a_failed_call() -> (
    None
):
    """One answer for "no such tool" and "not yours", so calling names maps nothing."""
    async with proxying(stdio_tools()) as proxy:
        unknown = await proxy.call_tool("a_tool_nobody_registered", {})
        failed = await proxy.call_tool("explode", {})

    assert unknown.is_error is True
    assert _envelope(unknown.content)["code"] == ErrorCode.TOOL_NOT_GRANTED.value
    assert sorted(_envelope(unknown.content)) == sorted(_envelope(failed.content))
    detail = _envelope(unknown.content)["detail"]
    assert isinstance(detail, dict)
    assert sorted(detail) == ["correlation_id", "subject"]


async def test_progress_a_server_reports_reaches_the_channel_before_the_result() -> (
    None
):
    """The Agent Runtime only logs `notifications/progress`; unappended, it is lost."""
    channel = RecordingChannel()

    async with proxying(stdio_tools(), channel=channel) as proxy:
        result = await proxy.call_tool("crawl", {})

    assert _text(result.content) == "crawled"
    assert [(p, t, m) for _, p, t, m in channel.reports] == [
        (1.0, 3.0, "step 1"),
        (2.0, 3.0, "step 2"),
        (3.0, 3.0, "step 3"),
    ]
    assert len({call_id for call_id, *_ in channel.reports}) == 1


async def test_a_servers_question_reaches_the_session_and_the_answer_gets_back() -> (
    None
):
    """Elicitation, in the direction it actually travels, over a real transport.

    The server asks while it still owes an answer to the call the Gateway made, so this
    only passes if the reply travelled back down the same open connection — a fake that
    resolved the question locally would return the answer without the child ever
    hearing it, and the child is what reports the outcome here.

    Both halves are asserted because either can fail alone: the question arriving with
    the server's own wording and schema, and the accepted content reaching the server
    unaltered.
    """
    channel = RecordingChannel(
        ElicitResult(action="accept", content={"account": "acct-42"})
    )

    async with proxying(stdio_tools(), channel=channel) as proxy:
        result = await proxy.call_tool("ask_operator", {})

    assert _text(result.content) == "accept:acct-42"
    assert len(channel.asked) == 1
    asked = channel.asked[0]
    assert isinstance(asked, ElicitRequestFormParams)
    assert asked.message == "which account should this run against?"
    assert asked.requested_schema["required"] == ["account"]


async def test_a_declined_question_is_carried_back_as_a_decline_not_a_failure() -> None:
    """A refusal is an answer, and the server is the one that decides what to do.

    Worth its own test because the tempting shortcut — treating anything that is not an
    accept as an upstream error — would turn a human saying no into `tool.unavailable`
    and lose the distinction the tenant needs.
    """
    channel = RecordingChannel()

    async with proxying(stdio_tools(), channel=channel) as proxy:
        result = await proxy.call_tool("ask_operator", {})

    assert result.is_error is not True
    assert _text(result.content) == "decline:None"


async def test_a_resource_is_listed_and_read_back_with_the_servers_own_contents() -> (
    None
):
    async with proxying(stdio_tools()) as proxy:
        listed = await proxy.list_resources()
        read = await proxy.read_resource("conformance://stdio/notes")

    assert [r.uri for r in listed] == ["conformance://stdio/notes"]
    assert _text(read.contents) == "a resource the stdio server serves"


async def test_a_uri_under_a_listed_template_is_read_without_a_prior_listing() -> None:
    """The index is built by the refresh a miss triggers, then the read is routed."""
    async with proxying(stdio_tools()) as proxy:
        read = await proxy.read_resource("conformance://stdio/pages/7")

    assert _text(read.contents) == "a resource the stdio server serves"


@pytest.mark.network
async def test_deepwiki_is_reached_over_streamable_http_and_answers_this_proxy() -> (
    None
):
    """The third-party half of the row's claim: a real registered server, live."""
    async with proxying([DEEPWIKI_TOOL]) as proxy:
        offered = await proxy.list_tools()
        result = await proxy.call_tool("repo_map", {"repoName": "pallets/click"})

    assert [t.name for t in offered] == ["repo_map"]
    assert offered[0].description is not None
    assert "repoName" in json.dumps(offered[0].input_schema)
    assert result.is_error is not True
    assert "pallets/click" in _text(result.content)


@pytest.mark.network
async def test_both_transports_are_proxied_through_one_tenants_tool_list() -> None:
    """One Session, two real servers, two transports, one list."""
    async with proxying([*stdio_tools(), DEEPWIKI_TOOL]) as proxy:
        offered = await proxy.list_tools()

    assert "repo_map" in {t.name for t in offered}
    assert "echo_credential" in {t.name for t in offered}


@pytest.mark.network
async def test_the_credential_goes_on_the_wire_as_the_vault_holds_it() -> None:
    """Indirect but real: deepwiki's answer changes with the header it is sent.

    Nothing here can read the outbound request directly. Deepwiki echoes no headers
    back, and `StreamableHttpServer.url` requires `https://`, so no local server can be
    registered to do it either. What is left is a far end whose behaviour depends on the
    header, and the pair below turns that into two facts rather than one:

    - Under `Authorization` with a value that is a well-formed bearer token, deepwiki
      answers "authentication is not allowed on the public endpoint" instead of the
      wiki structure. So the registration's header name and its value both reached it.
    - Under `Authorization` with a value that is *not* a scheme, deepwiki ignores it and
      answers normally. So the Gateway prefixed nothing of its own — had it composed a
      `Bearer ` in front of the vault's value, this arm would answer like the first.

    That second arm is the one worth having: the module docstring claims the vault holds
    the value in the form it goes on the wire, and this is what would fail if some later
    edit decided to be helpful about it.

    The cost is a coupling to a third party's undocumented behaviour, taken knowingly —
    the alternative is asserting nothing at all about the outbound credential. It fails
    in a `-m network` run rather than in the project's gate if deepwiki changes.
    """
    registered = tool(
        "repo_map",
        "read_wiki_structure",
        "deepwiki",
        deepwiki_server(credential_header="Authorization"),
        "repoName",
    )
    bearer = PlaceholderVault("Bearer map-conformance-not-a-real-token")
    bare = PlaceholderVault("map-conformance-not-a-scheme")

    async with proxying([registered], vault=bearer) as proxy:
        as_a_token = await proxy.call_tool("repo_map", {"repoName": "pallets/click"})
    async with proxying([registered], vault=bare) as proxy:
        as_a_string = await proxy.call_tool("repo_map", {"repoName": "pallets/click"})

    assert "Authentication is not allowed" in _text(as_a_token.content)
    assert "Authentication is not allowed" not in _text(as_a_string.content)
    assert "pallets/click" in _text(as_a_string.content)


@pytest.mark.network
async def test_one_unreachable_server_does_not_empty_the_reachable_ones_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both halves asserted together, because either alone is satisfiable alone."""
    dead = tool(
        "ghost",
        "ghost",
        "unreachable",
        StreamableHttpServer(
            transport="streamable_http",
            url="https://tool-gateway-conformance.invalid/mcp",
            credential_ref="conformance/dead",
            credential_header="X-Map-Conformance",
        ),
        "query",
    )

    with caplog.at_level(logging.ERROR, logger="managed_agent.gateway.tool.error_map"):
        async with proxying([DEEPWIKI_TOOL, dead]) as proxy:
            offered = await proxy.list_tools()

    assert [t.name for t in offered] == ["repo_map"]
    assert "upstream MCP failure subject=unreachable" in caplog.text
