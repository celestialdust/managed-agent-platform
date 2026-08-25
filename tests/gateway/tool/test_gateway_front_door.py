"""The edge: who is calling is decided here, and one Session's trouble stays its own.

Tier 1, driven through a real MCP client over a real ASGI stack, because the properties
worth asserting are about what a caller *receives* — a refusal that arrives as a result
rather than as a protocol fault reads to an agent as an unavailable tool, and the same
refusal arriving as a JSON-RPC error reads as this Gateway being broken.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import time
from collections.abc import AsyncIterator, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ElicitRequestParams, ElicitResult
from schema_stdio_server import TOOL_NAME as SCHEMA_TOOL_NAME
from starlette.types import Receive, Scope, Send
from tool_gateway_harness import (
    TENANT,
    CountingEvidence,
    CountingVault,
    FixedScope,
    broker,
    capture,
    schema_stdio_endpoint,
    stdio_endpoint,
)

from managed_agent.adapters.s3.session_vfs import UnconfiguredSessionVfs
from managed_agent.core.ids import Seq, SessionId, TenantId
from managed_agent.core.registration.scope_binding import (
    ParameterType,
    RegisteredTool,
    ScopeBinding,
    UnknownTool,
)
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER,
    InvalidSessionToken,
    SessionContext,
    verify_session_token,
)
from managed_agent.core.vfs.evidence import (
    CaptureAsEvidence,
    CaptureContext,
    EvidenceRef,
    ReturnInline,
    digest_of,
    evidence_object_key,
    evidence_vfs_path,
)
from managed_agent.gateway.tool import mcp_proxy
from managed_agent.gateway.tool.mcp_proxy import SessionUpstreams, ToolEventTypes
from managed_agent.gateway.tool.server import (
    MCP_PATH,
    GatewaySessions,
    SessionTokenMiddleware,
    create_gateway_app,
)

KEY = b"a signing key that is thirty-two"
TYPES = ToolEventTypes(
    progress="tool.progress",
    elicitation_requested="tool.elicitation_requested",
    elicitation_answered="tool.elicitation_answered",
)


def signed(session_id: UUID, tenant_id: UUID, expiry: int, key: bytes = KEY) -> str:
    body = f"{session_id}.{tenant_id}.{expiry}"
    return f"{body}.{hmac.new(key, body.encode(), hashlib.sha256).hexdigest()}"


def valid_for(session_id: UUID, tenant_id: UUID) -> str:
    return signed(session_id, tenant_id, int(time.time()) + 300)


class SilentLog:
    """An Event Log nothing in these tests reads back; it only has to accept."""

    def __init__(self) -> None:
        self.appended: list[tuple[str, dict[str, object]]] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.appended.append((type_, payload))
        return len(self.appended)

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[object]:
        while True:  # pragma: no cover - no test here answers an elicitation
            await asyncio.sleep(3600)
            yield None


class RecordingRegistry:
    """A tool registry that answers one stdio tool and remembers who asked."""

    def __init__(self, tools: Sequence[RegisteredTool] | None = None) -> None:
        self.tools = list(tools if tools is not None else [_stdio_tool()])
        self.asked_by: list[TenantId] = []

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        self.asked_by.append(tenant_id)
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise UnknownTool(tool_name)

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        self.asked_by.append(tenant_id)
        return self.tools


def _stdio_tool(name: str = "echo_credential") -> RegisteredTool:
    return RegisteredTool(
        name=name,
        remote_name="echo_credential",
        parameters={"query": ParameterType.STRING},
        scope_bindings=(ScopeBinding(dimension="account", argument="query"),),
        server_name="conformance_stdio",
        endpoint=stdio_endpoint(),
    )


def _schema_tool(name: str = "acme__big_report") -> RegisteredTool:
    """A registration for the upstream whose tool declares an output schema."""
    return RegisteredTool(
        name=name,
        remote_name=SCHEMA_TOOL_NAME,
        parameters={"query": ParameterType.STRING},
        scope_bindings=(ScopeBinding(dimension="account", argument="query"),),
        server_name="schema_stdio",
        endpoint=schema_stdio_endpoint(),
    )


@dataclass(slots=True)
class Harness:
    sessions: GatewaySessions
    registry: RecordingRegistry
    vault: CountingVault
    log: SilentLog
    recorder: CountingEvidence


def harness(
    credential: str | None = None,
    tool: RegisteredTool | None = None,
    recorder: CountingEvidence | None = None,
) -> Harness:
    """The whole Gateway over the conformance stdio server.

    `credential` is what that server echoes back, so it is also how a test chooses how
    large a result the Gateway has to classify -- the stub server takes no size
    argument and adding one would mean editing it. `tool` points the registration at a
    different real server; both stdio servers read the report out of the same variable,
    so the size lever works either way. `recorder` is how a test makes the Evidence
    store itself the thing under test rather than a bystander.
    """
    registry = RecordingRegistry(None if tool is None else [tool])
    vault = CountingVault() if credential is None else CountingVault(credential)
    log = SilentLog()
    recorder = recorder or CountingEvidence()
    sessions = GatewaySessions(
        scopes=FixedScope(),
        registry=registry,
        broker=broker(vault),
        append=log,
        events=log,  # type: ignore[arg-type]
        types_=TYPES,
        evidence=capture(recorder),
    )
    return Harness(
        sessions=sessions,
        registry=registry,
        vault=vault,
        log=log,
        recorder=recorder,
    )


@contextlib.asynccontextmanager
async def _client(
    built: Harness, token: str | None
) -> AsyncIterator[httpx2.AsyncClient]:
    """An HTTP client onto the served app, with the lifespan actually running."""
    app = create_gateway_app(
        built.sessions, KEY, UnconfiguredSessionVfs(), _NoRollouts()
    )
    headers = {} if token is None else {SESSION_TOKEN_HEADER.decode(): token}
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://gateway",
            headers=headers,
        ) as http,
    ):
        yield http


@contextlib.asynccontextmanager
async def _mcp(built: Harness, token: str) -> AsyncIterator[ClientSession]:
    """A real MCP client session speaking to the served app over Streamable HTTP."""
    async with (
        _client(built, token) as http,
        streamable_http_client(f"http://gateway{MCP_PATH}", http_client=http) as (
            read,
            write,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


def test_a_token_this_key_signed_yields_its_session_and_tenant() -> None:
    session_id, tenant_id = uuid4(), uuid4()

    context = verify_session_token(
        valid_for(session_id, tenant_id), KEY, int(time.time())
    )

    assert context == SessionContext(
        session_id=SessionId(session_id), tenant_id=TenantId(tenant_id)
    )


def test_every_wrong_token_is_refused_with_one_indistinguishable_message() -> None:
    """A caller who learns *which* part was wrong learns whether a Session id exists."""
    session_id, tenant_id = uuid4(), uuid4()
    good = valid_for(session_id, tenant_id)
    flipped = good[:-1] + ("0" if good[-1] != "0" else "1")

    wrong = [
        signed(session_id, tenant_id, int(time.time()) + 300, key=b"a different key"),
        flipped,
        ".".join(good.split(".")[:3]),
        signed(session_id, tenant_id, int(time.time()) - 1),
        "not-a-uuid.{}.{}.{}".format(
            tenant_id,
            int(time.time()) + 300,
            hmac.new(
                KEY,
                f"not-a-uuid.{tenant_id}.{int(time.time()) + 300}".encode(),
                hashlib.sha256,
            ).hexdigest(),
        ),
    ]

    messages = set()
    for token in wrong:
        with pytest.raises(InvalidSessionToken) as caught:
            verify_session_token(token, KEY, int(time.time()))
        messages.add(str(caught.value))

    assert messages == {"invalid session token"}


@pytest.mark.parametrize(
    "token",
    [None, "garbage", "a.b.c.d"],
    ids=["absent", "unsigned", "malformed"],
)
async def test_a_request_without_a_valid_token_is_refused_before_anything_is_read(
    token: str | None,
) -> None:
    built = harness()

    async with _client(built, token) as http:
        response = await http.post(
            MCP_PATH,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"accept": "application/json, text/event-stream"},
        )

    assert response.status_code == 401
    assert response.content == b'{"error":"invalid session token"}'
    assert built.registry.asked_by == []
    assert built.vault.fetches == []


async def test_an_expired_token_is_refused_even_though_this_key_signed_it() -> None:
    built = harness()
    expired = signed(uuid4(), uuid4(), int(time.time()) - 1)

    async with _client(built, expired) as http:
        response = await http.post(
            MCP_PATH,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"accept": "application/json, text/event-stream"},
        )

    assert response.status_code == 401
    assert built.registry.asked_by == []


async def test_the_tenant_a_handler_reads_is_the_one_out_of_that_callers_token() -> (
    None
):
    built = harness()
    tenant_id = uuid4()

    async with _mcp(built, valid_for(uuid4(), tenant_id)) as session:
        listed = await session.list_tools()

    assert [tool.name for tool in listed.tools] == ["echo_credential"]
    assert set(built.registry.asked_by) == {TenantId(tenant_id)}


async def test_two_concurrent_requests_each_see_their_own_session() -> None:
    """The context variable is per-request, and both requests are in flight at once.

    Asserted with a barrier rather than by racing and hoping: each inner call blocks
    until the other has arrived, so a context variable leaking between them would be
    read while both are live rather than after one has finished.
    """
    both_arrived = asyncio.Barrier(2)
    seen: list[SessionContext] = []

    from managed_agent.gateway.tool.server import _CURRENT

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        await both_arrived.wait()
        seen.append(_CURRENT.get())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guarded = SessionTokenMiddleware(inner, KEY)

    async def call(session_id: UUID, tenant_id: UUID) -> None:
        token = valid_for(session_id, tenant_id).encode()
        await guarded(
            {
                "type": "http",
                "headers": [(SESSION_TOKEN_HEADER, token)],
            },
            _no_messages,
            _discard,
        )

    first, second = (uuid4(), uuid4()), (uuid4(), uuid4())
    await asyncio.gather(call(*first), call(*second))

    assert {(c.session_id, c.tenant_id) for c in seen} == {
        (SessionId(first[0]), TenantId(first[1])),
        (SessionId(second[0]), TenantId(second[1])),
    }


async def _no_messages() -> MutableMapping[str, Any]:  # pragma: no cover
    raise AssertionError("the inner app read a body it was not given")


async def _discard(message: MutableMapping[str, Any]) -> None:
    return None


async def test_one_session_gets_one_proxy_and_one_owning_task() -> None:
    built = harness()
    context = SessionContext(SessionId(uuid4()), TenantId(uuid4()))
    other = SessionContext(SessionId(uuid4()), TenantId(uuid4()))

    first = await built.sessions.proxy_for(context)
    again = await built.sessions.proxy_for(context)
    different = await built.sessions.proxy_for(other)

    assert first is again
    assert first is not different
    assert await built.sessions.sweep(idle_for_s=0.0) == 2


async def test_a_second_tenant_presenting_one_session_id_gets_its_own_proxy() -> None:
    """The cache keeps the tenant the edge verified rather than dropping it.

    Both halves come out of one signed token today, so this is not currently reachable
    by a caller — which is exactly why it is worth pinning. A cache keyed by the Session
    alone would hand the second caller a proxy scoped to the *first* caller's tenant,
    and nothing below this line looks at the pair again.
    """
    built = harness()
    shared_session = SessionId(uuid4())
    mine = SessionContext(shared_session, TenantId(uuid4()))
    theirs = SessionContext(shared_session, TenantId(uuid4()))

    my_proxy = await built.sessions.proxy_for(mine)
    their_proxy = await built.sessions.proxy_for(theirs)

    assert my_proxy is not their_proxy
    await my_proxy.list_tools()
    await their_proxy.list_tools()
    assert built.registry.asked_by == [mine.tenant_id, theirs.tenant_id]

    await built.sessions.aclose()


async def test_a_refusal_arrives_as_a_failed_result_and_not_as_a_protocol_fault() -> (
    None
):
    """The low-level server has no exception-to-result conversion to borrow.

    Raising out of a handler reaches the Agent Runtime as a JSON-RPC error, which reads
    as the Gateway being broken rather than as this call being unavailable. A refusal is
    therefore returned. `call_tool` raising here is the failure this asserts against.
    """
    built = harness()

    async with _mcp(built, valid_for(uuid4(), uuid4())) as session:
        result = await session.call_tool("a_tool_nobody_registered", {})

    assert result.is_error is True
    text = "".join(c.text for c in result.content if c.type == "text")
    assert '"code":"tool.not_granted"' in text


async def test_a_real_tool_call_is_served_with_the_servers_own_content() -> None:
    built = harness()

    async with _mcp(built, valid_for(uuid4(), uuid4())) as session:
        result = await session.call_tool("echo_credential", {})

    assert result.is_error is not True
    assert [c.text for c in result.content if c.type == "text"] == [built.vault.value]


async def test_a_large_result_reaches_the_caller_as_a_reference_to_stored_bytes() -> (
    None
):
    """End to end, over a real MCP client and a real registered server: a result above
    the threshold never reaches the caller at all.

    The whole Gateway is assembled here rather than the capture point alone, because the
    property is about position: the substitution has to happen on the way back through
    `mcp_proxy`, before anything downstream of it can observe the payload. A capture
    wired one step later would satisfy every unit test in
    `tests/gateway/tool/test_evidence_capture.py` and fail this one.
    """
    body = "X" * 200_000
    session_id = uuid4()
    built = harness(credential=body)

    async with _mcp(built, valid_for(session_id, uuid4())) as session:
        result = await session.call_tool("echo_credential", {})

    assert result.is_error is not True
    text = "".join(c.text for c in result.content if c.type == "text")
    assert body not in text, "the payload the Gateway weighed was still handed over"
    assert str(len(body)) in text

    (ctx, decision) = built.recorder.captured[0]
    assert ctx.session_id == SessionId(session_id)
    key = evidence_object_key(ctx.session_id, decision.digest)
    stored = built.recorder.objects[key]
    assert stored == body.encode()
    assert digest_of(stored) == decision.digest
    assert digest_of(stored).hex in text
    assert evidence_vfs_path(decision.digest) in text


async def test_a_schema_declaring_tools_large_result_is_still_a_reference() -> None:
    """The capture that strips structured content must not be a promise the Gateway
    already broke by advertising a shape for that content.

    End to end, because the break is on the caller's side of the wire and nowhere else:
    `ClientSession` caches an output schema from the listing and revalidates every
    non-error result against it without being asked, so a Gateway that forwards the
    schema and then captures hands the Agent Runtime a raised `RuntimeError` instead of
    the Evidence reference -- every large result from every schema-declaring server, and
    the tenant's tool simply stops working above the threshold. A unit test on the
    capture point cannot see this, and neither can one on the listing.

    Restore the schema to what `list_tools` offers and this test fails where a runtime
    would: inside `call_tool`, with "has an output schema but did not return structured
    content".
    """
    body = "R" * 200_000
    session_id = uuid4()
    built = harness(credential=body, tool=_schema_tool())

    async with _mcp(built, valid_for(session_id, uuid4())) as session:
        listed = await session.list_tools()
        result = await session.call_tool("acme__big_report", {})

    assert [tool.output_schema for tool in listed.tools] == [None]
    assert result.is_error is not True
    assert result.structured_content is None
    text = "".join(c.text for c in result.content if c.type == "text")
    assert body not in text
    (_, decision) = built.recorder.captured[0]
    assert evidence_vfs_path(decision.digest) in text


async def test_a_small_result_from_that_tool_still_carries_its_structured_content() -> (
    None
):
    """Declining to advertise the schema costs the shape hint, not the data.

    The pair matters: a Gateway that simply stripped structured content everywhere would
    also pass the case above, and would have taken a first-class part of every result
    away from the model to protect the rare large one.
    """
    built = harness(credential="a short report", tool=_schema_tool())

    async with _mcp(built, valid_for(uuid4(), uuid4())) as session:
        result = await session.call_tool("acme__big_report", {})

    assert result.is_error is not True
    assert result.structured_content == {"report": "a short report"}


class _SlowEvidence(CountingEvidence):
    """A recorder whose writes outlive the deadline they are made inside.

    Both of them, because which one the capture point reaches depends on the size of
    the result and the property below is about the deadline rather than about the
    branch. A store that has gone slow is not hypothetical: the two writes are S3 and
    Postgres, and either can stall for longer than a Turn is willing to wait.
    """

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self._seconds = seconds

    async def record_captured(
        self,
        ctx: CaptureContext,
        payload: bytes,
        decision: CaptureAsEvidence,
        truncated_at_runtime_cap: bool = False,
    ) -> EvidenceRef:
        await asyncio.sleep(self._seconds)
        return await super().record_captured(
            ctx, payload, decision, truncated_at_runtime_cap
        )

    async def record_inline(self, ctx: CaptureContext, decision: ReturnInline) -> None:
        await asyncio.sleep(self._seconds)
        await super().record_inline(ctx, decision)


async def test_a_capture_that_outlives_the_tool_deadline_fails_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capture sits inside the tool deadline, not beside it.

    `call_tool` puts the capture inside its `asyncio.timeout` and argues in a comment
    that the position is load-bearing. Nothing held it there. Move the capture one
    dedent out of the `async with` -- still inside the `try`, so every failure still
    reads as a failed tool call -- and a slow Evidence store holds the call open past
    the moment the Agent Runtime stopped waiting for it: the runtime has already
    failed the Turn on its own deadline while the Gateway is still writing, and the
    result it eventually hands back is one nobody is reading.

    The deadline is patched down because the real one is tens of seconds; it is read
    from the module at call time, which is what makes that possible without a fake
    transport. What the assertion turns on is the elapsed time, so it fails on the
    dedent whichever way the late result is shaped.
    """
    monkeypatch.setattr(mcp_proxy, "GATEWAY_TOOL_TIMEOUT_S", 1.0)
    built = harness(recorder=_SlowEvidence(seconds=8.0))

    async with _mcp(built, valid_for(uuid4(), uuid4())) as session:
        started = time.monotonic()
        result = await session.call_tool("echo_credential", {})
        elapsed = time.monotonic() - started

    assert elapsed < 5.0, "the capture ran on past the deadline the runtime holds"
    assert result.is_error is True
    text = "".join(c.text for c in result.content if c.type == "text")
    assert '"code":"tool.timed_out"' in text
    assert built.recorder.inline == [], "the write the deadline cut short still landed"


async def test_a_resource_read_is_served_with_the_servers_contents_unchanged() -> None:
    built = harness()

    async with _mcp(built, valid_for(uuid4(), uuid4())) as session:
        listed = await session.list_resources()
        read = await session.read_resource("conformance://stdio/notes")

    assert [str(r.uri) for r in listed.resources] == ["conformance://stdio/notes"]
    assert [getattr(c, "text", None) for c in read.contents] == [
        "a resource the stdio server serves"
    ]


async def test_the_health_path_answers_without_a_token() -> None:
    built = harness()

    async with _client(built, None) as http:
        response = await http.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_the_app_serves_exactly_one_mcp_path() -> None:
    """A second MCP surface would be a second way in, past a different check."""
    app = create_gateway_app(
        harness().sessions, KEY, UnconfiguredSessionVfs(), _NoRollouts()
    )

    mcp_routes = [
        route for route in app.routes if getattr(route, "path", None) == MCP_PATH
    ]

    assert len(mcp_routes) == 1


async def test_sweeping_waits_for_each_owning_task_to_actually_end() -> None:
    """Awaiting the task is what proves the stdio children were reaped."""
    built = harness()
    context = SessionContext(SessionId(uuid4()), TenantId(uuid4()))
    proxy = await built.sessions.proxy_for(context)
    await proxy.list_tools()
    owner = built.sessions._live[(context.tenant_id, context.session_id)].owner

    closed = await built.sessions.sweep(idle_for_s=0.0)

    assert closed == 1
    assert owner.done()


async def test_a_session_that_will_not_close_does_not_stall_every_other_one() -> None:
    """The release happens outside the registry lock, and that is the whole test.

    Under the lock, one Session whose unwind blocks holds a process-wide lock that every
    inbound request for every other Session needs. Reaching into `_live` is the only way
    to build that state — no public call can produce a transport that refuses to close.
    """
    built = harness()
    wedged = SessionContext(SessionId(uuid4()), TenantId(uuid4()))
    healthy = SessionContext(SessionId(uuid4()), TenantId(uuid4()))
    await built.sessions.proxy_for(wedged)
    live = built.sessions._live[(wedged.tenant_id, wedged.session_id)]
    real_owner = live.owner
    live.upstreams = _NeverCloses(
        tenant_id=TENANT, broker=broker(built.vault), elicitation=_decline
    )

    sweeping = asyncio.create_task(built.sessions.sweep(idle_for_s=0.0))
    await asyncio.sleep(0)

    async with asyncio.timeout(2.0):
        assert await built.sessions.proxy_for(healthy) is not None

    sweeping.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sweeping
    real_owner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await real_owner
    await built.sessions.aclose()


async def test_one_session_that_fails_to_unwind_does_not_abort_the_sweep() -> None:
    """A raising close is logged and the loop goes on to the next Session."""
    built = harness()
    first = SessionContext(SessionId(uuid4()), TenantId(uuid4()))
    second = SessionContext(SessionId(uuid4()), TenantId(uuid4()))
    await built.sessions.proxy_for(first)
    await built.sessions.proxy_for(second)
    live = built.sessions._live[(first.tenant_id, first.session_id)]
    broken_owner = live.owner
    live.upstreams = _RaisesOnClose(
        tenant_id=TENANT, broker=broker(built.vault), elicitation=_decline
    )

    closed = await built.sessions.sweep(idle_for_s=0.0)

    assert closed == 2
    assert built.sessions._live == {}
    broken_owner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await broken_owner


async def _decline(params: ElicitRequestParams) -> ElicitResult:
    return ElicitResult(action="decline")


class _NeverCloses(SessionUpstreams):
    async def aclose(self) -> None:
        await asyncio.Event().wait()


class _RaisesOnClose(SessionUpstreams):
    async def aclose(self) -> None:
        raise RuntimeError("this transport will not come apart")


class _NoRollouts:
    """A rollout store holding nothing, for cases that are not about resuming.

    Answers None rather than raising, because that is the honest answer for the
    Sessions these cases drive: none of them has completed a Turn, so none has a
    stored Rollout. A raising stand-in would make every case here assert something
    about a store it is not testing.
    """

    async def restore_for_resume(self, session_id: SessionId) -> None:
        return None
