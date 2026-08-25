"""The Tool Gateway's front door: the one MCP server the Agent Runtime may reach.

Everything a Session's agent does with an enterprise tool arrives here over Streamable
HTTP from inside that Session's pod. There is exactly one of these in a Session's
compiled configuration, and that is load-bearing rather than tidy: the Agent Runtime
registers its resource built-ins whenever it has any MCP server at all, and what bounds
them is that the only server they can reach is this one. A second server configured
alongside would undo that silently (ADR-018, invariant I15).

This service is never told that a Session ended. That is the Control Plane's knowledge,
and reaching for it here would give this service a second reason to change, so what a
Session holds open — a stdio child, an HTTP connection — is released by disuse instead.

The low-level server registers no handlers by decorator: they are constructor arguments
taking `(context, params)` and returning the full result model, and `Server` takes one
type parameter. That shapes the refusal path — the low-level server has no
exception-to-result conversion, so a refusal is *returned* as a result with `is_error`
set rather than raised.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final

from fastapi import FastAPI
from mcp import types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.ports import EventLogAppend, EventLogRange
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER,
    InvalidSessionToken,
    SessionContext,
    verify_session_token,
)
from managed_agent.core.vfs.session_vfs import SessionFiles
from managed_agent.gateway.tool.credential_broker import ToolCredentialBroker
from managed_agent.gateway.tool.evidence_capture import EvidenceCapture
from managed_agent.gateway.tool.mcp_proxy import (
    EventLogSessionChannel,
    McpProxy,
    SessionScopeReader,
    SessionUpstreams,
    ToolEventTypes,
    ToolRegistryReader,
)
from managed_agent.gateway.tool.rollout_seed import (
    SessionRollouts,
    rollout_seed_endpoint,
)
from managed_agent.gateway.tool.working_lane import working_lane_endpoints

MCP_PATH: Final[str] = "/mcp"
"""Where the one MCP endpoint is served, exactly.

An exact route rather than a sub-application mounted here, and the difference is not
cosmetic: a mount matches `/mcp/<rest>` and answers a bare `/mcp` with a 307 to the
trailing-slash form. The Agent Runtime's MCP client does not follow redirects, so a
compiled configuration naming the natural spelling would fail to reach a service that
was running correctly. With a route it is the trailing-slash form that redirects, and
nothing in the compiled configuration ever writes that one.
"""
UPSTREAM_IDLE_TIMEOUT_S: Final[float] = 900.0
SWEEP_INTERVAL_S: Final[float] = 60.0

RELEASE_TIMEOUT_S: Final[float] = 30.0
"""How long one Session's connections get to unwind before the release gives up.

A backstop, not a tuning knob. Unwinding a transport can block on a child process that
will not die or a socket that will not close, and every caller of the release is either
the idle sweeper or process shutdown — both of which have other Sessions to get to.
"""

_log = logging.getLogger(__name__)


_CURRENT: ContextVar[SessionContext] = ContextVar("map_tool_gateway_session")


@dataclass(slots=True)
class _Live:
    """One Session's proxy, its connections, and the task that owns them.

    `owner` is the task running `SessionUpstreams.run`. It is held so that closing a
    Session can wait for the connections to actually unwind rather than only asking
    them to; a stdio child is reaped when that task leaves its exit stack, not when
    the request that last used it returns.
    """

    upstreams: SessionUpstreams
    proxy: McpProxy
    owner: asyncio.Task[None]
    touched_at: float


_LiveKey = tuple[TenantId, SessionId]
"""What a cached Session's connections are filed under.

Both halves, not just the Session. They arrive together in one signed token, so today
the pair is asserted by the signer — but a cache keyed by the Session alone silently
drops the only authorization term this edge verified, and then the second caller for
that Session id gets a proxy carrying the *first* caller's tenant. Nothing below this
line re-checks the pair, so the drop would be the whole of the check.
"""


class GatewaySessions:
    """The Sessions this service currently holds upstream connections for."""

    def __init__(
        self,
        registry: ToolRegistryReader,
        broker: ToolCredentialBroker,
        append: EventLogAppend,
        events: EventLogRange,
        types_: ToolEventTypes,
        evidence: EvidenceCapture,
        scopes: SessionScopeReader,
    ) -> None:
        self._registry = registry
        self._broker = broker
        # Handed to every proxy this service opens rather than read here. A Session's
        # Scope is read on its first tool call, not when its connections are opened,
        # so nothing on this path awaits a database while holding the lock every other
        # Session's first request is queued behind.
        self._scopes = scopes
        self._append = append
        self._events = events
        self._types = types_
        # One capture point for every Session this service holds, because the threshold
        # is one number read once at the composition root. A per-Session capture point
        # would be a second place the variable could be read, which is the one thing the
        # size rule exists to prevent.
        self._evidence = evidence
        self._live: dict[_LiveKey, _Live] = {}
        self._lock = asyncio.Lock()

    async def proxy_for(self, context: SessionContext) -> McpProxy:
        async with self._lock:
            key = (context.tenant_id, context.session_id)
            live = self._live.get(key)
            if live is None:
                live = self._open(context)
                self._live[key] = live
            live.touched_at = time.monotonic()
            return live.proxy

    def _open(self, context: SessionContext) -> _Live:
        channel = EventLogSessionChannel(
            session_id=context.session_id,
            append=self._append,
            events=self._events,
            types=self._types,
        )
        upstreams = SessionUpstreams(
            tenant_id=context.tenant_id,
            broker=self._broker,
            elicitation=channel.ask,
        )
        proxy = McpProxy(
            tenant_id=context.tenant_id,
            session_id=context.session_id,
            registry=self._registry,
            upstreams=upstreams,
            channel=channel,
            evidence=self._evidence,
            scopes=self._scopes,
        )
        # Created rather than awaited: this task must outlive the inbound request that
        # happened to be first through the door, because it is what holds every upstream
        # connection's cancel scope open for the rest of the Session.
        owner = asyncio.create_task(upstreams.run())
        return _Live(
            upstreams=upstreams,
            proxy=proxy,
            owner=owner,
            touched_at=time.monotonic(),
        )

    async def sweep(self, idle_for_s: float = UPSTREAM_IDLE_TIMEOUT_S) -> int:
        """Close what nothing has used lately. Returns how many Sessions were closed."""
        cutoff = time.monotonic() - idle_for_s
        async with self._lock:
            stale = [
                (key, live)
                for key, live in self._live.items()
                if live.touched_at < cutoff
            ]
            for key, _ in stale:
                del self._live[key]
        await _release_all(stale)
        return len(stale)

    async def aclose(self) -> None:
        async with self._lock:
            live = list(self._live.items())
            self._live.clear()
        await _release_all(live)


async def _release_all(entries: list[tuple[_LiveKey, _Live]]) -> None:
    """Unwind every detached Session's connections, outside the registry lock.

    The entries are removed from `_live` under the lock and released here without it,
    because unwinding can block: an exit stack that will not come apart, or a stdio
    child that will not die. Holding the process-wide lock across that would stall every
    inbound request for every other Session on the wedged one, which is a far larger
    failure than the leaked file descriptors it would be protecting against.
    """
    for (_, session_id), live in entries:
        await _release(session_id, live)


async def _release(session_id: SessionId, live: _Live) -> None:
    """Stop one Session's connections and wait for the task that holds them to end.

    Bounded and non-raising, both for the same reason: this is called in a loop over
    every idle Session, and one that cannot be closed must not take the rest with it.
    A timeout cancels the owning task, which is the strongest thing available from
    outside it. What is given up is certainty that the stdio child was reaped — so the
    give-up is logged rather than swallowed, since a repeated line here is the signal
    that this process is accumulating orphans.
    """
    try:
        async with asyncio.timeout(RELEASE_TIMEOUT_S):
            await live.upstreams.aclose()
            await live.owner
    except Exception:
        _log.exception(
            "upstream connections for session %s did not unwind cleanly", session_id
        )


def build_mcp_server(sessions: GatewaySessions) -> Server[object]:
    """The proxying MCP server, with one handler per method it forwards.

    Handlers are constructor arguments rather than decorated functions, and each returns
    the whole result model. That is what removes the refusal round-trip: a refusal is a
    `CallToolResult` with `is_error` set, returned like any other result, because the
    low-level server has no exception-to-result conversion of its own to borrow —
    raising here would reach the Agent Runtime as a protocol fault, which reads as this
    Gateway being broken rather than as the call being unavailable (ADR-014).

    Every listing is returned whole, with `next_cursor` unset: the proxy has already
    followed each registered server's pages to the end, and paginating outward again
    would make the Agent Runtime's client repeat work this service just finished.
    """

    async def proxy() -> McpProxy:
        return await sessions.proxy_for(_CURRENT.get())

    async def on_list_tools(
        context: ServerRequestContext[object],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=await (await proxy()).list_tools())

    async def on_call_tool(
        context: ServerRequestContext[object],
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        return await (await proxy()).call_tool(params.name, params.arguments or {})

    async def on_list_resources(
        context: ServerRequestContext[object],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=await (await proxy()).list_resources()
        )

    async def on_list_resource_templates(
        context: ServerRequestContext[object],
        params: types.PaginatedRequestParams | None,
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(
            resource_templates=await (await proxy()).list_resource_templates()
        )

    async def on_read_resource(
        context: ServerRequestContext[object],
        params: types.ReadResourceRequestParams,
    ) -> types.ReadResourceResult:
        return await (await proxy()).read_resource(params.uri)

    return Server(
        "managed-agent-tool-gateway",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_resources=on_list_resources,
        on_list_resource_templates=on_list_resource_templates,
        on_read_resource=on_read_resource,
    )


class SessionTokenMiddleware:
    """Put the calling Session in a context variable, or refuse before anything reads.

    Middleware rather than a per-handler concern because the MCP handlers above are
    called by the protocol layer and are given nowhere to receive a request. Refusing
    here also means an unsigned request never reaches a registry read or a vault fetch.
    """

    def __init__(self, app: ASGIApp, key: bytes) -> None:
        self._app = app
        self._key = key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        presented = dict(scope["headers"]).get(SESSION_TOKEN_HEADER)
        try:
            if presented is None:
                raise InvalidSessionToken("invalid session token")
            context = verify_session_token(
                presented.decode("ascii", "strict"), self._key, int(time.time())
            )
        except (InvalidSessionToken, UnicodeDecodeError):
            await _refuse(send)
            return
        token = _CURRENT.set(context)
        try:
            await self._app(scope, receive, send)
        finally:
            _CURRENT.reset(token)


async def _refuse(send: Send) -> None:
    """A fixed 401 with a fixed body.

    Deliberately not an ErrorEnvelope: that set is the tenant-facing vocabulary, and
    this response is seen by a pod rather than by a tenant. Putting a published code
    here would commit the platform to a code no tenant can ever observe.
    """
    body = b'{"error":"invalid session token"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _sweep_forever(sessions: GatewaySessions) -> None:
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_S)
        closed = await sessions.sweep()
        if closed:
            _log.info("released upstream connections for %d idle sessions", closed)


def create_gateway_app(
    sessions: GatewaySessions,
    token_key: bytes,
    files: SessionFiles,
    rollouts: SessionRollouts,
) -> FastAPI:
    """The whole service: two token-checked surfaces, plus a liveness path.

    `/healthz` sits outside the token check, because a probe has no Session and giving
    one a token would put a long-lived credential in the cluster for the sake of a
    health check.

    The MCP route is appended to the router rather than declared with a decorator
    because what it serves is a whole ASGI application — the token middleware wrapping
    the protocol manager — and there is no request-and-response function to decorate.

    `files` is the Session VFS the working-lane routes read. Required rather than
    defaulted, because the alternative to a wired store is a surface that refuses
    in production for as long as it takes somebody to notice — and a
    Gateway that starts happily while half of what it serves cannot work is the
    silent nothing this codebase has paid for more than once. A caller with no
    store to hand it has no business serving this app.

    The working-lane routes are appended the same way and wrapped in the same
    middleware, so what is behind the token check is decided in this one function.
    They are GET-only: the surface exists to hand a pod back its own earlier
    workspace, and a write door onto a lane the pod already writes through its own
    sync would be a second way in with a different check behind it.

    `rollouts` is where a Session's resume state is read back from, and it is required
    on the same terms and for a sharper version of the same reason. Left unwired, this
    service would answer "no Rollout" to every seeding pod, every resuming Session
    would then open a fresh thread over a record it should have continued, and the
    platform would replay folded history and report success -- the one failure the
    whole resume path exists to prevent (ADR-004). A caller with no store to hand it
    has no business serving this app.
    """
    mcp_server = build_mcp_server(sessions)
    manager = StreamableHTTPSessionManager(app=mcp_server, stateless=True)

    async def handle(scope: Scope, receive: Receive, send: Send) -> None:
        await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with manager.run():
            sweeper = asyncio.create_task(_sweep_forever(sessions))
            try:
                yield
            finally:
                sweeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sweeper
                await sessions.aclose()

    app = FastAPI(title="Managed Agent Tool Gateway", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.router.routes.append(
        Route(
            MCP_PATH,
            endpoint=SessionTokenMiddleware(handle, token_key),
            methods=["GET", "POST", "DELETE"],
        )
    )
    seed_path, seed_endpoint = rollout_seed_endpoint(rollouts, _CURRENT.get)
    for path, endpoint in (
        *working_lane_endpoints(files, _CURRENT.get),
        (seed_path, seed_endpoint),
    ):
        app.router.routes.append(
            Route(
                path,
                endpoint=SessionTokenMiddleware(endpoint, token_key),
                methods=["GET"],
            )
        )
    return app
