"""Reaching the MCP servers a tenant registered, over the two transports allowed.

Deadlines first. The Agent Runtime waits a fixed time for one MCP server to come up
and a fixed time for one tool call, and this platform adopts it rather than patching it,
so those two figures are received and every deadline here is derived strictly inside
them. An elicitation's deadline is inside a tool call's in turn, because the question is
asked while that call is still open — the Agent Runtime puts no bound on an elicitation
at all, which is exactly why one has to be put here.

Connections belong to one Session. The elicitation callback is bound at the moment a
client session is constructed, so a connection shared across Sessions could not tell
whose Event Log a question belongs in; and a stdio child shared across Sessions would be
a channel between them that nothing else in this design provides.

Credentials are not read here. `gateway/tool/credential_broker.py` composes the vault
key under the calling tenant, reads it, and returns an attachment whose one method
writes the value into a child's environment or into a request header — so this module
never holds a bare secret and cannot format one into a log line. The value is attached
in the form the vault holds it, `Bearer ` prefix included where there is one; nothing
here composes a credential it does not understand.

The Streamable HTTP entry point is `streamable_http_client` and yields two streams. It
takes neither `headers=` nor `timeout=`, so an outbound credential rides on an
`httpx2.AsyncClient` this module builds and owns; `read_timeout_seconds` is a float
rather than a `timedelta`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Final, Literal, Protocol, assert_never
from uuid import uuid4

import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.session import ClientRequestContext
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.dispatcher import ProgressFnT
from mcp.types import (
    CallToolResult,
    ElicitRequestFormParams,
    ElicitRequestParams,
    ElicitResult,
    PaginatedRequestParams,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    Tool,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from managed_agent.core.errors import ErrorCode, ErrorEnvelope
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.ports import (
    EventLogAppend,
    EventLogRange,
)
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerEndpoint,
    ServerName,
    StdioServer,
    StreamableHttpServer,
    UnknownTool,
)
from managed_agent.core.session.event_append import append_in_order
from managed_agent.core.session.session import SessionRecord
from managed_agent.gateway.tool import error_map
from managed_agent.gateway.tool.credential_broker import ToolCredentialBroker
from managed_agent.gateway.tool.evidence_capture import EvidenceCapture, advertisable
from managed_agent.gateway.tool.scope_clamp import OutOfScope, narrow

_log = logging.getLogger(__name__)

RUNTIME_MCP_STARTUP_TIMEOUT_S: Final[float] = 10.0
"""What the Agent Runtime waits for one MCP server to start. Received, not chosen."""

RUNTIME_MCP_TOOL_TIMEOUT_S: Final[float] = 60.0
"""What the Agent Runtime waits for one tool call. Received, not chosen."""

TIMEOUT_HEADROOM_S: Final[float] = 5.0
"""How far inside the enclosing deadline each of ours sits. One number, so widening the
margin is one edit and cannot be done to one deadline and not another."""

GATEWAY_STARTUP_TIMEOUT_S: Final[float] = (
    RUNTIME_MCP_STARTUP_TIMEOUT_S - TIMEOUT_HEADROOM_S
)
GATEWAY_TOOL_TIMEOUT_S: Final[float] = RUNTIME_MCP_TOOL_TIMEOUT_S - TIMEOUT_HEADROOM_S
GATEWAY_ELICITATION_TIMEOUT_S: Final[float] = (
    GATEWAY_TOOL_TIMEOUT_S - TIMEOUT_HEADROOM_S
)

UPSTREAM_READ_TIMEOUT_S: Final[float] = GATEWAY_TOOL_TIMEOUT_S - 1.0
"""The MCP client's own per-request deadline, set a second inside the tool backstop so a
slow server is classified by the request layer — which raises `MCPError` carrying
`REQUEST_TIMEOUT` and so records `tool.timed_out` — rather than by a cancelled task,
which reports nothing about why. It is deliberately *not* inside the startup deadline:
those two bound different things, a connection coming up versus a request on a
connection already up, and neither encloses the other."""

_MAX_LIST_PAGES: Final[int] = 100
"""How many pages one listing may draw before this module stops asking.

A registered server that returns the same cursor forever would otherwise hold a
listing open without bound, and a listing is not inside a tool call's deadline. Stopping
short is recorded rather than silent, because a truncated list and a server that dropped
a tool look identical to the agent and want different fixes."""

"""How many times a lost sequence race is re-attempted before it becomes a failure."""


def deadlines_nest(startup: float, tool: float, elicitation: float) -> bool:
    """Whether one set of Gateway deadlines sits strictly inside the Agent Runtime's.

    A function rather than a bare `if`, so the ordering can be checked against figures a
    test supplies. Re-importing this module cannot check it: a reload re-executes the
    module body, which reassigns `TIMEOUT_HEADROOM_S` to the source value before the
    guard below runs, so the guard is unreachable by any test that patches the constant.
    """
    return (
        0.0 < startup < RUNTIME_MCP_STARTUP_TIMEOUT_S
        and 0.0 < elicitation < tool < RUNTIME_MCP_TOOL_TIMEOUT_S
    )


if not deadlines_nest(
    GATEWAY_STARTUP_TIMEOUT_S, GATEWAY_TOOL_TIMEOUT_S, GATEWAY_ELICITATION_TIMEOUT_S
):
    raise RuntimeError(
        "every Tool Gateway deadline must sit strictly inside the Agent Runtime's; "
        "check TIMEOUT_HEADROOM_S against the two runtime figures above"
    )

ElicitationHandler = Callable[[ElicitRequestParams], Awaitable[ElicitResult]]


async def _drain[ItemT](
    page: Callable[[str | None], Awaitable[tuple[Sequence[ItemT], str | None]]],
    subject: str,
) -> list[ItemT]:
    """Follow `next_cursor` to the end of a paginated listing.

    Reading one page and stopping loses the tail, and a registered tool in that tail
    reads to the agent as a tool that does not exist — the same symptom as a server that
    dropped it, wanting a different fix. `subject` names what is being listed so a
    truncated listing is legible in the log.
    """
    collected: list[ItemT] = []
    cursor: str | None = None
    for _ in range(_MAX_LIST_PAGES):
        items, cursor = await page(cursor)
        collected.extend(items)
        if cursor is None:
            return collected
    _log.warning(
        "listing %s stopped at the %d page cap with a cursor still open",
        subject,
        _MAX_LIST_PAGES,
    )
    return collected


def _page_params(cursor: str | None) -> PaginatedRequestParams | None:
    """The params one page of a listing is asked for with.

    `None` rather than an empty params object for the first page: a server is entitled
    to treat a present-but-null cursor differently from an absent one, and the first
    page is the request that must work against every registered server.
    """
    return None if cursor is None else PaginatedRequestParams(cursor=cursor)


_OWNER_STOPPED: Final[str] = (
    "this Session's upstream connections are no longer being held open"
)
"""What a caller is told when the task that owns the connections has ended.

Raised as a `ConnectionError`, which classifies as `tool.unavailable`: from the agent's
side the server genuinely cannot be reached, and the reason — that this Session is being
torn down underneath the call — is not something a tenant can act on.
"""


@dataclass(frozen=True, slots=True, eq=False)
class _OpenRequest:
    """One "open me a connection to this server" handed to the owning task.

    Compared by identity rather than by value, because the owning task keeps a set of
    the ones still outstanding and two requests for the same endpoint are two waiters.
    """

    endpoint: ServerEndpoint
    answer: asyncio.Future[ClientSession]


def _answer_with(request: _OpenRequest, outcome: ClientSession | BaseException) -> None:
    """Settle one waiter, unless something already did."""
    if request.answer.done():
        return
    if isinstance(outcome, BaseException):
        request.answer.set_exception(outcome)
    else:
        request.answer.set_result(outcome)


async def _unwound(stack: AsyncExitStack, failure: BaseException) -> BaseException:
    """Close a half-open stack, and say which exception the caller should report.

    The unwind's own exception wins when there is one. A transport whose task group
    failed reports the real cause *there* — what reached the caller first was only the
    cancellation that the failure provoked, which says nothing about why.

    A cancellation of this task always wins over both, and `cancelling()` is what tells
    the two apart: anyio clears the cancellation it raised itself as its cancel scope
    exits, so a count still standing after the unwind is somebody cancelling this task
    from outside and must not be swallowed.
    """
    unwinding: BaseException | None = None
    try:
        await stack.aclose()
    except BaseException as exc:  # noqa: BLE001 - returned, not swallowed
        unwinding = exc
    task = asyncio.current_task()
    if task is not None and task.cancelling() > 0:
        return failure
    return unwinding or failure


class SessionUpstreams:
    """One live MCP client session per registered server, for the life of one Session.

    Opened lazily, so a Session whose agent never calls a tool spawns no stdio child
    and a registered-but-unused server costs nothing. Each server gets a stack of its
    own, and every one of those goes on a Session-wide stack that unwinds in reverse
    when the Session ends — a stdio child outliving the Session that spawned it is a
    process nobody owns.

    Per-server rather than one shared stack, because a failed open is otherwise not a
    local event: see `_open_one`, which is where the containment happens.

    The Session-wide stack is entered and left by `run`, in one task of its own, and
    every other
    method asks `run` to do the opening rather than opening anything itself. This is not
    a style choice. Both MCP transports start an anyio task group, and anyio binds
    a task group to the task that entered it: a connection opened inside one inbound
    request is torn down when that request's task ends, so the next request finds a
    `ClientSession` whose dispatcher is already closed and every call on it raises
    `MCPError(CONNECTION_CLOSED)`. Closing it from a third task — the idle sweeper —
    fails louder still, with `RuntimeError: Attempted to exit a cancel scope that isn't
    the current task's current cancel scope`. Measured against a real server, not
    reasoned about: an inbound request that opens its own connection succeeds, and the
    request after it fails.
    """

    def __init__(
        self,
        tenant_id: TenantId,
        broker: ToolCredentialBroker,
        elicitation: ElicitationHandler,
    ) -> None:
        self._tenant_id = tenant_id
        self._broker = broker
        self._elicitation = elicitation
        self._sessions: dict[ServerName, ClientSession] = {}
        self._requests: asyncio.Queue[_OpenRequest | None] = asyncio.Queue()
        self._pending: set[_OpenRequest] = set()
        self._opening = asyncio.Lock()
        self._finished = asyncio.Event()

    async def run(self) -> None:
        """Own every connection this Session opens, until `aclose` stops the loop.

        Must be scheduled as its own task — `asyncio.create_task`, never awaited inline
        — before any `session_for` call, and `aclose` waits on it.

        The `finally` is load-bearing twice over, and neither part is defensive tidying.
        Unwinding the stack is exactly where a half-entered transport raises, and
        `aclose` is a wait on `_finished`: set that only on the success path and one bad
        unwind leaves every `aclose` for this Session waiting forever — behind a
        process-wide lock, in the front door that calls it. And every caller sitting in
        `session_for` is waiting on a future only this task can settle, so this task
        ending for any reason at all has to settle them, or they wait for a reader of a
        queue nobody reads any more. Both were reproduced, not reasoned about.
        """
        try:
            async with AsyncExitStack() as session_stack:
                while True:
                    request = await self._requests.get()
                    if request is None:
                        break
                    try:
                        session, opened = await self._open_one(request.endpoint)
                    except Exception as exc:  # noqa: BLE001 - relayed to the asker
                        _answer_with(request, exc)
                    else:
                        session_stack.push_async_exit(opened)
                        _answer_with(request, session)
        finally:
            for waiting in self._pending:
                _answer_with(waiting, ConnectionError(_OWNER_STOPPED))
            self._finished.set()

    async def session_for(
        self, server_name: ServerName, endpoint: ServerEndpoint
    ) -> ClientSession:
        existing = self._sessions.get(server_name)
        if existing is not None:
            return existing
        async with self._opening:
            # Re-read under the lock: two concurrent first calls for one server would
            # otherwise both miss and spawn the registered command twice.
            already = self._sessions.get(server_name)
            if already is not None:
                return already
            if self._finished.is_set():
                raise ConnectionError(_OWNER_STOPPED)
            answer: asyncio.Future[ClientSession] = (
                asyncio.get_running_loop().create_future()
            )
            request = _OpenRequest(endpoint=endpoint, answer=answer)
            self._pending.add(request)
            # `put_nowait` rather than `await put`: the queue is unbounded so there is
            # nothing to wait for, and having no suspension point between the check
            # above and the hand-off is what pairs this with `run`'s `finally` — the
            # request is either already visible there, or the owner had already stopped
            # and the check refused.
            self._requests.put_nowait(request)
            try:
                session = await answer
            finally:
                self._pending.discard(request)
            self._sessions[server_name] = session
            return session

    async def _open_one(
        self, endpoint: ServerEndpoint
    ) -> tuple[ClientSession, AsyncExitStack]:
        """Open one server on a stack of its own, unwound here if the open fails.

        Both halves of that sentence are the fix for one measured failure, and neither
        is tidiness. Each transport runs on an anyio task group, and a child task that
        fails cancels the task that entered the group — this one. A half-open transport
        left on the Session-wide stack therefore delivers its cancellation *later*,
        while some other server is being opened or while the Session is being closed,
        and takes down every connection this Session holds along with the owning task;
        measured against an unresolvable host, `run` died 40ms in with an
        `ExceptionGroup` raised out of the shared stack's unwind, and the caller waiting
        on its answer waited forever.

        Unwound here instead, anyio's own cancel scope absorbs the cancellation as the
        transport's context manager exits and hands back an `ExceptionGroup` carrying
        the real cause. This task continues, and a live connection to another server
        opened beside it keeps answering — also measured.

        The returned stack is the caller's to keep: it holds this server's transport
        open and must be pushed onto something that outlives the request.
        """
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            async with asyncio.timeout(GATEWAY_STARTUP_TIMEOUT_S):
                session = await self._open(stack, endpoint)
                await session.initialize()
        except BaseException as failure:
            recorded = await _unwound(stack, failure)
            raise recorded from None
        return session, stack

    async def _open(
        self, stack: AsyncExitStack, endpoint: ServerEndpoint
    ) -> ClientSession:
        """Build the transport the registration declared, credential attached.

        Which attachment comes back is decided by the transport arm, and each
        attachment carries only the one method its transport can use — so a stdio
        credential cannot be written into a header by an edit that type-checks.

        The `httpx2.AsyncClient` goes on the same stack as everything else because the
        transport only closes a client it created itself; one passed in stays the
        caller's to close, and a client left open holds its connection pool.
        """
        match endpoint:
            case StdioServer():
                attached = await self._broker.for_stdio(self._tenant_id, endpoint)
                streams = await stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=endpoint.command,
                            args=list(endpoint.args),
                            # Merged rather than replaced: a bare environment
                            # holding one variable strips PATH, and the child then
                            # fails to start for a reason unlike the cause.
                            env=attached.into_env(get_default_environment()),
                        )
                    )
                )
            case StreamableHttpServer():
                over_http = await self._broker.for_http(self._tenant_id, endpoint)
                http_client = await stack.enter_async_context(
                    httpx2.AsyncClient(
                        headers=over_http.into_headers({}),
                        timeout=GATEWAY_TOOL_TIMEOUT_S,
                    )
                )
                streams = await stack.enter_async_context(
                    streamable_http_client(endpoint.url, http_client=http_client)
                )
            case _ as unreachable:
                assert_never(unreachable)
        read, write = streams
        return await stack.enter_async_context(
            ClientSession(
                read,
                write,
                read_timeout_seconds=UPSTREAM_READ_TIMEOUT_S,
                elicitation_callback=self._on_elicit,
            )
        )

    async def _on_elicit(
        self, context: ClientRequestContext, params: ElicitRequestParams
    ) -> ElicitResult:
        """Hand a server's question to whoever can reach this Session's caller.

        The MCP request context is unused: `elicitation/create` is its own request and
        names no tool call, so there is nothing in it to correlate on that the handler
        does not already hold. The parameter stays because the SDK's callback protocol
        matches on the name as well as the position.
        """
        return await self._elicitation(params)

    async def aclose(self) -> None:
        """Stop the owning task and wait for it to unwind every connection.

        Returns only once `run` has left its exit stack, so a caller that awaits this
        knows the stdio children are reaped rather than merely signalled. `run` sets the
        event in a `finally`, so an unwind that raises releases this wait too.
        """
        self._sessions.clear()
        await self._requests.put(None)
        await self._finished.wait()


@dataclass(frozen=True, slots=True)
class ToolEventTypes:
    """The three published event-type names the Tool Gateway writes.

    Handed in, never written here. The closed set lives in `core/vocabulary/` and a
    family module is what puts a name into it; holding a second copy of these strings
    would let this service emit a name the published set does not carry, which is the
    one thing a closed set exists to prevent (ADR-013).
    """

    progress: str
    elicitation_requested: str
    elicitation_answered: str


class SessionChannel(Protocol):
    """How the Tool Gateway reaches the Session whose agent is calling."""

    async def progress(
        self, call_id: str, progress: float, total: float | None, message: str | None
    ) -> None: ...

    async def ask(self, params: ElicitRequestParams) -> ElicitResult: ...


class _ElicitationAnswer(BaseModel):
    """One answer event, parsed out of the Event Log payload it was appended as."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    elicitation_id: str
    action: Literal["accept", "decline", "cancel"]
    content: dict[str, str | int | float | bool | list[str] | None] = Field(
        default_factory=dict
    )


class EventLogSessionChannel:
    """Progress out, and a question out and its answer back, over one Event Log.

    The question goes the same way the progress does, and the answer comes back the same
    way: the Gateway appends the question, then follows the Session's own log forward
    for the answer naming it. One channel in both directions means this service needs no
    inbound route of its own, and the tenant keeps talking to the one surface it already
    talks to (ADR-007). The Agent Runtime is deliberately not in this path — it would
    carry the question, but the answer belongs to the caller rather than to the agent,
    and the Agent Runtime puts no deadline on an elicitation at all.

    Both writes go through the retry above rather than calling `append` directly. The
    log this writes into is the same one the Session's Turn runner is writing to while
    the tool call is open, so losing a sequence race is the ordinary case; an unretried
    one escapes into the tool call and fails it.

    The wait is bounded because the tool call it happens inside is bounded. An
    unanswered question that outlived its call would hold a stdio child and a database
    follow open for a reader that stopped listening. An expired question becomes
    `cancel`, which is the MCP action meaning nobody answered.
    """

    def __init__(
        self,
        session_id: SessionId,
        append: EventLogAppend,
        events: EventLogRange,
        types: ToolEventTypes,
        timeout_s: float = GATEWAY_ELICITATION_TIMEOUT_S,
    ) -> None:
        self._session_id = session_id
        self._append = append
        self._events = events
        self._types = types
        self._timeout_s = timeout_s

    async def progress(
        self, call_id: str, progress: float, total: float | None, message: str | None
    ) -> None:
        await append_in_order(
            self._append,
            self._session_id,
            self._types.progress,
            {
                "call_id": call_id,
                "progress": progress,
                "total": total,
                "message": message,
            },
        )

    async def ask(self, params: ElicitRequestParams) -> ElicitResult:
        """Put one question on the log and wait there for its answer.

        Only a form-mode elicitation is carried. A URL-mode one asks a human to open a
        URL the registered server chose, and relaying that to a tenant would make this
        service a redirector for any third party a tenant ever registered; it also
        carries an `elicitation_id` of the server's own, which would compete with the
        one minted here for the answer to name. Declining is the fail-safe of the two:
        the server learns the question was not put, rather than waiting the deadline
        out.
        """
        if not isinstance(params, ElicitRequestFormParams):
            _log.warning(
                "declined a url-mode elicitation from a registered server: %s",
                self._session_id,
            )
            return ElicitResult(action="decline")
        elicitation_id = uuid4().hex
        asked_at = await append_in_order(
            self._append,
            self._session_id,
            self._types.elicitation_requested,
            {
                "elicitation_id": elicitation_id,
                "message": params.message,
                "requested_schema": params.requested_schema,
            },
        )
        try:
            async with asyncio.timeout(self._timeout_s):
                async for record in self._events.follow(self._session_id, asked_at):
                    if record.type != self._types.elicitation_answered:
                        continue
                    try:
                        answer = _ElicitationAnswer.model_validate(record.payload)
                    except ValidationError:
                        # Not an answer this Gateway can read. Skipping rather than
                        # declining is deliberate: it may be answering a different
                        # question, and declining here would answer ours on its behalf.
                        continue
                    if answer.elicitation_id != elicitation_id:
                        continue
                    return _result_for(answer)
        except TimeoutError:
            return ElicitResult(action="cancel")
        # `follow` does not end on its own; this is what makes the function total.
        return ElicitResult(action="cancel")


def _result_for(answer: _ElicitationAnswer) -> ElicitResult:
    """The MCP result the registered server is waiting on.

    Only `accept` carries content, so an accept with nothing in it cannot be constructed
    here — a tool proceeding on data nobody supplied is the failure worth designing out.
    """
    if answer.action == "accept":
        return ElicitResult(action="accept", content=answer.content)
    return ElicitResult(action=answer.action)


class ToolRegistryReader(Protocol):
    """What this service needs from the tool registry, and nothing more."""

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool: ...

    async def list_for_tenant(
        self, tenant_id: TenantId
    ) -> Sequence[RegisteredTool]: ...


class SessionScopeReader(Protocol):
    """Where one Session's Scope is read from, and nothing more.

    One method with the signature `SessionRegistry.fetch` already has, so the registry
    the control plane writes Sessions through satisfies this without an adapter, and
    this service gets no way to create or page one. Same narrowing as
    `ToolRegistryReader` above, for the same reason: the port a process is typed at is
    what decides which calls it is able to make.
    """

    async def fetch(
        self, session_id: SessionId, tenant_id: TenantId
    ) -> SessionRecord: ...


def _by_server(
    tools: Sequence[RegisteredTool],
) -> dict[ServerName, list[RegisteredTool]]:
    grouped: dict[ServerName, list[RegisteredTool]] = {}
    for tool in tools:
        grouped.setdefault(tool.server_name, []).append(tool)
    return grouped


def _raise_if_no_server_answered(
    reached: int, refusals: Sequence[ErrorEnvelope]
) -> None:
    """Refuse a listing that reached nothing, rather than reporting it as empty.

    Every listing method below tolerates a partial failure on purpose: one unreachable
    server must not empty the others' lists, so a server that fails is recorded and
    skipped. The tolerance has one blind spot -- when no server answers at all, the
    result is `[]`, and `[]` is also what a tenant with nothing registered gets. The
    caller cannot tell the two apart, so a Runtime facing a total outage decides it
    has no tools and carries on silently degraded.

    `reached` counts servers that answered, so zero-with-refusals is the only case that
    raises: a tenant with no servers registered still gets its honest empty list, and
    one server answering is a useful list even if three did not. The first refusal is
    the one surfaced -- it carries a correlation id, and the log holds every server's
    failure under it.
    """
    if reached or not refusals:
        return
    raise error_map.as_listing_error(refusals[0])


def _record_offer_census(
    tenant_id: TenantId,
    registered: Sequence[RegisteredTool],
    offered: Sequence[Tool],
) -> None:
    """Say what a tool listing was carrying, on every listing, without exception.

    A catalogue that comes back short has three causes and only two of them speak. A
    server that could not be reached is recorded by `error_map.record`, and a tool the
    server no longer offers is warned about at the point it is skipped -- but a registry
    read that comes back empty produces no line at all, because the loop that would
    carry one never runs. From outside the process that third case is indistinguishable
    from a healthy listing, which is how a Session whose tenant had been granted a tool
    was handed an empty catalogue, told the user that tool did not exist, and completed
    its Turn -- four times, with nothing in the log to separate it from a tenant who had
    registered nothing.

    So exactly one line leaves here per call, and it leaves before the refusal guard
    rather than after. A listing that goes on to raise has still read the registry, and
    what it was carrying when every server refused is the difference between one lost
    tool and forty.

    What absence means is therefore not symmetric, and reading it as though it were is
    the trap this paragraph exists to close. Every case that is worth acting on speaks
    at WARNING or above, so under the deployment's own filter no line means either a
    healthy listing or no listing at all -- never a short one. Separating those two is
    the Agent Runtime's job rather than this one's: its tool catalogue emits an
    `available_server_count` and a `tool_count` of its own, and a `tool_count` with no
    census here beside it says the listing never reached this process.

    Three of the four cases sit at WARNING or above deliberately. Nothing in this
    process configures logging -- uvicorn's default `dictConfig` names its own three
    loggers and leaves the root logger at WARNING holding no handler, so an INFO record
    from this package is dropped before it reaches one, and a census the deployment
    filters out is the silence it was written to end. The healthy case is the one that
    can afford INFO: it says the catalogue was the size the tenant registered, which is
    worth reading in a local run and worth nothing during an incident.

    Names are logged and endpoints are not. A tool or server name is the tenant's own
    vocabulary and the string an operator greps for, while the endpoint beside it in the
    same registration carries the command a stdio server is spawned with, a URL, and the
    vault entry its credential is read from.
    """
    if not registered:
        _log.warning(
            "offer census: tenant=%s registered=0 offered=0 -- the registry holds no "
            "tool for this tenant, so this Session was handed an empty catalogue",
            tenant_id,
        )
        return
    kept = {tool.name for tool in offered}
    dropped = sorted(tool.name for tool in registered if tool.name not in kept)
    if not offered:
        _log.error(
            "offer census: tenant=%s registered=%d offered=0 -- every registered tool "
            "was dropped on the way out, which no healthy listing does; dropped=%s",
            tenant_id,
            len(registered),
            dropped,
        )
        return
    if dropped:
        _log.warning(
            "offer census: tenant=%s registered=%d offered=%d -- dropped=%s",
            tenant_id,
            len(registered),
            len(offered),
            dropped,
        )
        return
    _log.info(
        "offer census: tenant=%s registered=%d offered=%d",
        tenant_id,
        len(registered),
        len(offered),
    )


class McpProxy:
    """One Session's view of every MCP server registered to its tenant.

    Resource reads are routed by an index this class builds while listing, because a
    resource URI names no server. A miss refreshes the index once and then refuses:
    trying each registered server in turn would answer the question "does this URI exist
    anywhere" for a caller that was told about none of them.

    Every listing method reaches several servers in one call, and one unreachable server
    must not empty the others' lists. So each server's turn — opening the connection as
    well as drawing the pages — sits inside its own `try`, and a failure is recorded and
    skipped. Opening is inside it rather than beside it because opening is where an
    unreachable or mis-credentialed server actually fails, and a failure that escapes a
    listing handler reaches the Agent Runtime as the upstream's own words: the low-level
    server forwards `str(exc)` as the JSON-RPC error message, which has been observed
    carrying an internal hostname and a database username.
    """

    def __init__(
        self,
        tenant_id: TenantId,
        session_id: SessionId,
        registry: ToolRegistryReader,
        upstreams: SessionUpstreams,
        channel: SessionChannel,
        evidence: EvidenceCapture,
        scopes: SessionScopeReader,
    ) -> None:
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._registry = registry
        self._upstreams = upstreams
        self._channel = channel
        self._evidence = evidence
        self._scopes = scopes
        self._scope: dict[str, str] | None = None
        self._resource_owner: dict[str, ServerName] = {}
        self._template_owner: dict[str, ServerName] = {}

    async def _session_scope(self) -> dict[str, str]:
        """This Session's Scope, read once and then held for as long as this proxy is.

        Cached without an expiry, which is safe here for a reason that is a property of
        the record rather than a judgement about how fast it changes: `SessionRecord`
        is written once at creation and no field of it is ever rewritten, so there is
        no later value for this one to be stale against. `UpdateSession` refuses every
        field a caller could send, and Scope is not among them either way.

        Read on the first tool call rather than when the proxy is built. A Session that
        only lists tools never needs it, and building the proxy is under a lock every
        other Session's first request queues behind.
        """
        if self._scope is None:
            record = await self._scopes.fetch(self._session_id, self._tenant_id)
            self._scope = dict(record.scope)
        return self._scope

    async def _servers(self) -> dict[ServerName, ServerEndpoint]:
        registered = await self._registry.list_for_tenant(self._tenant_id)
        return {tool.server_name: tool.endpoint for tool in registered}

    async def list_tools(self) -> list[Tool]:
        registered = await self._registry.list_for_tenant(self._tenant_id)
        offered: list[Tool] = []
        reached = 0
        refusals: list[ErrorEnvelope] = []
        for server_name, tools in _by_server(registered).items():
            try:
                session = await self._upstreams.session_for(
                    server_name, tools[0].endpoint
                )

                async def page(
                    cursor: str | None, session: ClientSession = session
                ) -> tuple[Sequence[Tool], str | None]:
                    result = await session.list_tools(params=_page_params(cursor))
                    return result.tools, result.next_cursor

                listed = await _drain(page, f"tools of {server_name}")
            except Exception as exc:
                # One server being down must not empty the other's list, and its own
                # words must not reach the agent either way.
                refusals.append(error_map.record(exc, server_name))
                continue
            reached += 1
            upstream = {tool.name: tool for tool in listed}
            for tool in tools:
                found = upstream.get(tool.remote_name)
                if found is None:
                    # The registration says this tool exists and the server no longer
                    # offers it. Advertising it anyway produces a call that fails at the
                    # server, which reads to the agent as the platform being unreliable.
                    _log.warning(
                        "registered tool %s has no remote %s on server %s",
                        tool.name,
                        tool.remote_name,
                        server_name,
                    )
                    continue
                # The registry decides the name; the server decides the rest, except
                # what this Gateway cannot honour -- see `advertisable`.
                offered.append(
                    advertisable(found).model_copy(update={"name": tool.name})
                )
        _record_offer_census(self._tenant_id, registered, offered)
        _raise_if_no_server_answered(reached, refusals)
        return offered

    async def call_tool(
        self, tool_name: str, arguments: dict[str, object]
    ) -> CallToolResult:
        try:
            tool = await self._registry.lookup(self._tenant_id, tool_name)
        except UnknownTool:
            # A name nobody registered and a name this Session may not reach are one
            # answer on purpose: two answers would let a model map the tenant's whole
            # tool inventory by calling names and reading which refusal came back.
            return error_map.as_tool_result(
                error_map.refusal(ErrorCode.TOOL_NOT_GRANTED, tool_name, uuid4().hex)
            )
        call_id = uuid4().hex
        try:
            # Before the upstream is opened, so a call that will not be made does not
            # read a credential on its way to being refused. The Scope read is inside
            # this `try` for the same reason every other failure here is: a Session
            # whose Scope cannot be read has to reach the agent as a failed tool call,
            # and failing closed is the only safe direction -- an unreadable Scope
            # narrows nothing.
            narrowed = narrow(tool, await self._session_scope(), arguments)
            if isinstance(narrowed, OutOfScope):
                return error_map.as_tool_result(
                    error_map.out_of_scope(narrowed.tool_name, narrowed.dimension)
                )
            session = await self._upstreams.session_for(tool.server_name, tool.endpoint)
            async with asyncio.timeout(GATEWAY_TOOL_TIMEOUT_S):
                result = await session.call_tool(
                    tool.remote_name,
                    narrowed,
                    read_timeout_seconds=UPSTREAM_READ_TIMEOUT_S,
                    progress_callback=_progress_into(self._channel, call_id),
                )
                # Here, and not after the return. This is the last point an enterprise
                # result passes that the pod cannot reach, so a large payload classified
                # here never enters the Agent Runtime at all -- and a capture moved
                # after the hand-back would keep every test green while losing exactly
                # that. Inside the tool deadline rather than beside it: the capture is
                # two writes to two stores, and a call that cannot be recorded has to
                # fail as a tool call rather than run past the deadline the Agent
                # Runtime is holding.
                return await self._evidence.apply(
                    self._session_id, call_id, tool_name, result
                )
        except Exception as exc:
            # Broad on purpose, and narrower than it looks: `BaseException` is not
            # caught, so cancellation still propagates. Every way a registered server
            # can fail has to reach the agent as a failed tool call rather than as a
            # fault of the Gateway, and `classify` keeps those two apart in the record.
            # Both transports wrap a failure in an `ExceptionGroup`, which is an
            # `Exception` and so is caught here; `classify` looks inside it.
            return error_map.as_tool_result(error_map.record(exc, tool_name))

    async def list_resources(self) -> list[Resource]:
        found: list[Resource] = []
        reached = 0
        refusals: list[ErrorEnvelope] = []
        for server_name, endpoint in (await self._servers()).items():
            try:
                session = await self._upstreams.session_for(server_name, endpoint)

                async def page(
                    cursor: str | None, session: ClientSession = session
                ) -> tuple[Sequence[Resource], str | None]:
                    result = await session.list_resources(params=_page_params(cursor))
                    return result.resources, result.next_cursor

                listed = await _drain(page, f"resources of {server_name}")
            except Exception as exc:
                # The failure is recorded, not returned: a listing that refuses whole is
                # worse for the agent than one short by the servers that could not
                # answer. Unless none of them could -- see the guard below.
                refusals.append(error_map.record(exc, server_name))
                continue
            reached += 1
            for resource in listed:
                self._resource_owner[resource.uri] = server_name
                found.append(resource)
        _raise_if_no_server_answered(reached, refusals)
        return found

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        found: list[ResourceTemplate] = []
        reached = 0
        refusals: list[ErrorEnvelope] = []
        for server_name, endpoint in (await self._servers()).items():
            try:
                session = await self._upstreams.session_for(server_name, endpoint)

                async def page(
                    cursor: str | None, session: ClientSession = session
                ) -> tuple[Sequence[ResourceTemplate], str | None]:
                    result = await session.list_resource_templates(
                        params=_page_params(cursor)
                    )
                    return result.resource_templates, result.next_cursor

                listed = await _drain(page, f"resource templates of {server_name}")
            except Exception as exc:
                refusals.append(error_map.record(exc, server_name))
                continue
            reached += 1
            for template in listed:
                prefix = template.uri_template.split("{", 1)[0]
                self._template_owner[prefix] = server_name
                found.append(template)
        _raise_if_no_server_answered(reached, refusals)
        return found

    async def read_resource(self, uri: str) -> ReadResourceResult:
        server_name = await self._resolve_owner(uri)
        if server_name is None:
            raise error_map.as_mcp_error(
                error_map.refusal(ErrorCode.TOOL_NOT_GRANTED, uri, uuid4().hex)
            )
        endpoint = (await self._servers())[server_name]
        try:
            session = await self._upstreams.session_for(server_name, endpoint)
            async with asyncio.timeout(GATEWAY_TOOL_TIMEOUT_S):
                return await session.read_resource(uri)
        except Exception as exc:
            raise error_map.as_mcp_error(error_map.record(exc, uri)) from exc

    async def _resolve_owner(self, uri: str) -> ServerName | None:
        owner = self._match(uri)
        if owner is not None:
            return owner
        await self.list_resources()
        await self.list_resource_templates()
        return self._match(uri)

    def _match(self, uri: str) -> ServerName | None:
        exact = self._resource_owner.get(uri)
        if exact is not None:
            return exact
        # Longest matching template prefix wins, so a server publishing `db://acme/`
        # does not capture a URI belonging to one publishing `db://acme/invoices/`.
        best: tuple[int, ServerName] | None = None
        for prefix, server_name in self._template_owner.items():
            if uri.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), server_name)
        return None if best is None else best[1]


def _progress_into(channel: SessionChannel, call_id: str) -> ProgressFnT:
    async def report(progress: float, total: float | None, message: str | None) -> None:
        await channel.progress(call_id, progress, total, message)

    return report
