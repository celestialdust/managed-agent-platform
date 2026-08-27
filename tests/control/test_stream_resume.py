"""What a caller actually receives on the stream, and where it resumes from.

Tier 1 (local, in-memory ports). The route is driven through the real FastAPI app, so
the frames under assertion are the bytes that would go on a socket rather than the
return value of a generator.

**Driven directly over ASGI, and it has to be.** Both of the usual clients -- httpx's
`ASGITransport` and Starlette's `TestClient` -- run the app to completion and hand back
a finished body. A live tail never completes, so with either of them the call that opens
the stream is itself the thing that blocks, and "arrived while the Turn was still
running" is not merely hard to assert, it is unreachable. `LiveResponse` below does what
an ASGI server does instead: feeds a scope, collects each `http.response.body` message
as the app sends it, and disconnects at the end.

Every read in this file is bounded by `_TIMEOUT_S`, for the same reason: a stream that
should have produced a frame and did not must fail saying what never arrived, not hang
the suite.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from starlette.types import Message, Scope

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import stream
from managed_agent.control.api.routes.stream import (
    BEFORE_FIRST,
    KEEPALIVE_FRAME,
    encode_event,
    encode_stream_error,
    resolve_resume_position,
)
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.errors import ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import (
    FIRST_SEQ,
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    new_definition_id,
    new_session_id,
)
from managed_agent.core.ports import (
    Resolution,
    SessionListing,
    SessionNotVisible,
)
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vocabulary import lifecycle
from managed_agent.core.vocabulary.stream import STREAM_ERROR, StreamError

_TIMEOUT_S = 2.0
"""How long any read in this file waits before failing.

An SSE response does not end on its own, so an unbounded read here would hang the whole
suite instead of failing one test. Every failure on this clock names what it expected
and never saw.
"""

_OWNER = TenantId(uuid4())
_HEADERS = {TENANT_HEADER: str(_OWNER)}

_PUBLISHED_TYPES = (
    lifecycle.SESSION_CREATED,
    lifecycle.SESSION_SUSPENDED,
    lifecycle.SESSION_RESUMED,
    lifecycle.SESSION_STOPPED,
)


@dataclass(frozen=True, slots=True)
class FakeRow:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


class FakeLog:
    """A contiguous in-memory Event Log whose retained floor can move under a reader.

    A real log rather than a mock: every assertion here is about the bytes a caller
    receives -- the id line, the event name, the order -- and a mock would let all three
    be wrong while the tests passed. It hands out contiguous sequences and can drop a
    prefix, which is what makes both retained-floor cases reachable without running a
    retention sweep.

    The next sequence is counted separately from the rows it holds, and that separation
    is the point. After a sweep the surviving rows no longer say how far the numbering
    got, so a fake deriving the next sequence from `len(rows)` would re-issue a
    number it had already used -- and the gap that a stream must refuse to read
    across could never occur here.
    """

    def __init__(self) -> None:
        self._rows: list[FakeRow] = []
        self._next_seq = FIRST_SEQ
        self._floor: Seq = FIRST_SEQ
        self._appended = asyncio.Event()

    def append(self, session_id: SessionId, type_: str) -> Seq:
        seq = Seq(self._next_seq)
        self._next_seq += 1
        self._rows.append(FakeRow(session_id, seq, type_, {"n": seq}))
        self._appended.set()
        return seq

    def expire_below(self, floor: Seq) -> None:
        """Drop everything under `floor`, as a retention sweep does."""
        self._floor = floor
        self._rows = [row for row in self._rows if row.seq >= floor]

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return self._floor

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[FakeRow]:
        span = [row for row in self._rows if start <= row.seq <= end]
        return span[:limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[FakeRow]:
        cursor = after
        while True:
            batch = [row for row in self._rows if row.seq > cursor]
            for row in batch:
                cursor = row.seq
                yield row
            if not batch:
                # Cleared before waiting and with no await in between, so an append
                # cannot slip through the gap and leave this asleep on an event that
                # has already fired.
                self._appended.clear()
                await self._appended.wait()


class UnusedAppend:
    """Satisfies the append port and is never called: nothing here writes through it."""

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        raise AssertionError("a test in this file appended through the port")


class UnusedDefinitionRegistry:
    """Satisfies the definition-registry port and is never called.

    Raising rather than answering: streaming a Session's events resolves no definition,
    so a call here means the route did something this file does not grade, and a quiet
    stub would let that pass unnoticed.
    """

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        raise AssertionError("a test in this file registered a definition")

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        raise AssertionError("a test in this file resolved a definition")

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        raise AssertionError("a test in this file listed a definition's versions")

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        raise AssertionError("a test in this file read one definition revision")

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a test in this file archived a definition revision")


class UnusedToolRegistry:
    """Satisfies the tool-registry port and is never called."""

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("a test in this file registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("a test in this file looked up a registered tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("a test in this file listed a tenant's tools")


class OneOwnedSession:
    """The Session registry, reduced to the one question this route asks it: whose?

    `fetch` is the only method the stream route reaches, and it is the only thing
    standing between a caller and another tenant's events -- the Event Log is keyed by
    Session and carries no tenant, so an unchecked follow succeeds and streams them in
    full. The other two raise, so a route that started creating or listing Sessions
    would say so here.
    """

    def __init__(self, owner: TenantId, session_id: SessionId) -> None:
        self._owner = owner
        self._session_id = session_id

    async def create(self, record: SessionRecord) -> None:
        raise AssertionError("a test in this file created a Session")

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        if session_id != self._session_id or tenant_id != self._owner:
            raise SessionNotVisible(str(session_id))
        return SessionRecord(
            id=session_id,
            tenant_id=tenant_id,
            definition_id=new_definition_id(),
            definition_revision="1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=500,
            budget_currency="USD",
            retention_days=7,
        )

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("a test in this file paged the Session registry")


def build_app(log: FakeLog, owner: TenantId, session_id: SessionId) -> FastAPI:
    """The real app factory over a real `Platform`, with a fake behind every port.

    Both "real"s are load-bearing. `platform_from_request` narrows `app.state.platform`
    with an isinstance check against the frozen dataclass, so a duck-typed stand-in
    fails the route before any assertion here is reached; and building the app through
    `create_app` is what makes this file notice if the router stops being attached.
    """
    return create_app(
        Platform(
            event_log_append=UnusedAppend(),
            event_log_range=log,
            definition_registry=UnusedDefinitionRegistry(),
            tool_registry=UnusedToolRegistry(),
            session_registry=OneOwnedSession(owner, session_id),
            webhooks=UnusedWebhooks(),
            environment_store=UnusedEnvironmentStore(),
            turn_dispatch=NoPodTransport(),
            file_store=unconfigured_file_store(),
        )
    )


def _scope(path: str, query: str, headers: dict[str, str]) -> Scope:
    """One GET request, as an ASGI server would present it.

    `spec_version` is deliberately below 2.4. At and above it Starlette expects the
    server to notice a dead socket and raise; below it, Starlette listens for the
    `http.disconnect` this harness sends, which is what lets a held-open stream be shut
    down cleanly at the end of a test instead of leaking a task.
    """
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [
            (name.lower().encode(), value.encode()) for name, value in headers.items()
        ],
        "client": ("tenant", 12345),
        "server": ("testserver", 80),
    }


class LiveResponse:
    """One in-flight ASGI response, read as the app emits it.

    Frames are parsed out of the raw byte stream rather than off message boundaries, so
    what is asserted is the SSE framing itself and not how the app happened to chunk it.
    """

    def __init__(self, app: FastAPI, scope: Scope) -> None:
        self._app = app
        self._scope = scope
        self._chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._started = asyncio.Event()
        self._disconnected = asyncio.Event()
        self._request_delivered = False
        self._task: asyncio.Task[None] | None = None
        self._buffer = ""
        self._ended = False
        self.status = 0
        self.headers: dict[str, str] = {}

    async def _receive(self) -> Message:
        if not self._request_delivered:
            self._request_delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await self._disconnected.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = {
                key.decode(): value.decode() for key, value in message["headers"]
            }
            self._started.set()
        elif message["type"] == "http.response.body":
            body = message.get("body", b"")
            if body:
                self._chunks.put_nowait(body)
            if not message.get("more_body", False):
                self._chunks.put_nowait(None)

    async def _run(self) -> None:
        """Run the app, so that a crash surfaces as an ended stream rather than silence.

        Without the `finally` a request that raised before sending anything would leave
        every reader below waiting out its full timeout and then reporting "nothing
        arrived", which is true and useless. Ending the queue makes the reader look at
        the task, and the task carries the real exception.
        """
        try:
            await self._app(self._scope, self._receive, self._send)
        finally:
            self._started.set()
            self._chunks.put_nowait(None)

    async def __aenter__(self) -> LiveResponse:
        task = asyncio.create_task(self._run())
        self._task = task
        try:
            await asyncio.wait_for(self._started.wait(), _TIMEOUT_S)
        except TimeoutError:
            task.cancel()
            raise AssertionError(
                f"no response started within {_TIMEOUT_S}s for {self._scope['path']}"
            ) from None
        if task.done():
            task.result()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._disconnected.set()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(task, _TIMEOUT_S)
        except (TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            # The app's own failure, raised only when it is not masking one of ours.
            if exc_type is None:
                raise

    def _reraise_app_failure(self) -> None:
        task = self._task
        if task is not None and task.done() and not task.cancelled():
            task.result()

    async def frames(self, count: int) -> list[str]:
        """The next `count` SSE frames, or an assertion naming what never arrived."""
        collected: list[str] = []
        try:
            await asyncio.wait_for(self._collect(collected, count), _TIMEOUT_S)
        except TimeoutError:
            raise AssertionError(
                f"expected {count} SSE frames within {_TIMEOUT_S}s; "
                f"{len(collected)} arrived: {collected}"
            ) from None
        return collected

    async def _collect(self, collected: list[str], count: int) -> None:
        while True:
            while "\n\n" in self._buffer and len(collected) < count:
                frame, _, self._buffer = self._buffer.partition("\n\n")
                collected.append(frame)
            if len(collected) >= count:
                return
            chunk = await self._chunks.get()
            if chunk is None:
                self._ended = True
                self._reraise_app_failure()
                raise AssertionError(
                    f"the response ended after {len(collected)} of {count} frames: "
                    f"{collected}"
                )
            self._buffer += chunk.decode()

    async def ended(self) -> None:
        """Assert the response is over, rather than there being more still to come."""
        if self._ended:
            return
        try:
            chunk = await asyncio.wait_for(self._chunks.get(), _TIMEOUT_S)
        except TimeoutError:
            raise AssertionError(
                f"the stream was still open {_TIMEOUT_S}s later; expected it to end"
            ) from None
        assert chunk is None, f"expected the stream to end; {chunk!r} arrived instead"
        self._ended = True

    async def body(self) -> str:
        """Everything a completing response sent -- a refusal, never a tail."""
        try:
            await asyncio.wait_for(self._drain(), _TIMEOUT_S)
        except TimeoutError:
            raise AssertionError(
                f"the response had not finished {_TIMEOUT_S}s later; "
                f"{self._buffer!r} so far"
            ) from None
        return self._buffer

    async def _drain(self) -> None:
        while not self._ended:
            chunk = await self._chunks.get()
            if chunk is None:
                self._ended = True
                self._reraise_app_failure()
                return
            self._buffer += chunk.decode()


def open_stream(
    app: FastAPI, path: str, *, headers: dict[str, str], query: str = ""
) -> LiveResponse:
    return LiveResponse(app, _scope(path, query, headers))


@pytest.fixture
def log() -> FakeLog:
    return FakeLog()


@pytest.fixture
def session_id() -> SessionId:
    return new_session_id()


@pytest.fixture
def app(log: FakeLog, session_id: SessionId) -> FastAPI:
    return build_app(log, _OWNER, session_id)


@pytest.fixture
def stream_path(session_id: SessionId) -> str:
    return f"/v1/sessions/{session_id}/events/stream"


def _fill(log: FakeLog, session_id: SessionId, count: int) -> None:
    """Append `count` events whose types are all in the published set."""
    for n in range(count):
        log.append(session_id, _PUBLISHED_TYPES[n % len(_PUBLISHED_TYPES)])


def _ids(frames: list[str]) -> list[str]:
    return [frame.splitlines()[0] for frame in frames]


def _stream_error(frame: str) -> StreamError:
    """The `stream.error` frame's data line, parsed back through the published model.

    Parsed rather than string-matched: `StreamError` forbids extra fields and types the
    code as the closed set, so a frame carrying an unpublished code, or a field nobody
    declared, fails here instead of being read straight past.
    """
    data = [line for line in frame.splitlines() if line.startswith("data: ")]
    assert len(data) == 1, f"expected exactly one data line in {frame!r}"
    return StreamError.model_validate_json(data[0].removeprefix("data: "))


async def _first_rows(rows: AsyncIterator[FakeRow], count: int) -> list[Seq]:
    seen: list[Seq] = []

    async def collect() -> None:
        async for row in rows:
            seen.append(row.seq)
            if len(seen) == count:
                return

    try:
        await asyncio.wait_for(collect(), _TIMEOUT_S)
    except TimeoutError:
        raise AssertionError(
            f"expected {count} rows from the fake within {_TIMEOUT_S}s; got {seen}"
        ) from None
    return seen


# --- the fake, held honest before anything is graded against it ----------------------


def test_the_fake_log_numbers_contiguously_from_one(
    log: FakeLog, session_id: SessionId
) -> None:
    assert [log.append(session_id, lifecycle.SESSION_CREATED) for _ in range(2)] == [
        1,
        2,
    ]


async def test_the_fake_log_follows_strictly_after_the_position_given(
    log: FakeLog, session_id: SessionId
) -> None:
    _fill(log, session_id, 4)

    assert await _first_rows(log.follow(session_id, Seq(2)), 2) == [3, 4]


async def test_the_fake_log_expires_a_prefix_and_moves_its_floor(
    log: FakeLog, session_id: SessionId
) -> None:
    _fill(log, session_id, 5)

    log.expire_below(Seq(4))

    assert await log.retained_floor(session_id) == 4
    assert [row.seq for row in await log.read(session_id, Seq(1), Seq(5))] == [4, 5]


async def test_the_fake_log_keeps_numbering_forward_across_a_sweep(
    log: FakeLog, session_id: SessionId
) -> None:
    """The gap this file needs is only reachable because of this.

    A fake that re-derived its next sequence from the rows it still holds would answer 1
    here, the log would silently renumber, and the contiguity break a stream must refuse
    to read across could not be produced at all.
    """
    _fill(log, session_id, 3)
    log.expire_below(Seq(4))

    assert log.append(session_id, lifecycle.SESSION_STOPPED) == 4


# --- the resume position, and the frames ---------------------------------------------


def test_a_caller_stating_no_position_starts_before_the_first_event() -> None:
    assert resolve_resume_position(None, None) == BEFORE_FIRST
    assert BEFORE_FIRST == FIRST_SEQ - 1


def test_the_query_parameter_is_used_when_it_is_the_only_position_given() -> None:
    assert resolve_resume_position(None, 7) == 7


def test_the_header_beats_the_query_parameter() -> None:
    """A reconnecting client re-issues the original URL, query string and all, and adds
    the header. The other order would rewind every reconnect to where the first request
    started and redeliver the whole span."""
    assert resolve_resume_position(9, 2) == 9


def test_an_event_frame_carries_the_row_sequence_as_the_sse_id() -> None:
    frame = encode_event(Seq(7), lifecycle.SESSION_CREATED, {"n": 7}).decode()

    assert frame.splitlines()[:2] == ["id: 7", f"event: {lifecycle.SESSION_CREATED}"]
    assert frame.endswith("\n\n")


def test_a_payload_containing_a_newline_does_not_end_the_frame_early() -> None:
    """A raw newline in the data line would terminate the frame at the wrong byte and
    leave the remainder to be parsed as the start of the next one."""
    frame = encode_event(Seq(1), lifecycle.SESSION_CREATED, {"note": "a\nb"}).decode()

    assert 'data: {"note":"a\\nb"}' in frame
    assert frame.count("\n\n") == 1


def test_the_error_frame_carries_no_id() -> None:
    """An id here would enter the caller's resume position, and its next reconnect would
    ask to continue after a number the log does not contain."""
    frame = encode_stream_error(
        StreamError(
            code=ErrorCode.EVENT_RANGE_EXPIRED, message="expired", retained_floor=4
        )
    ).decode()

    assert frame.startswith(f"event: {STREAM_ERROR}\n")
    assert not any(line.startswith("id:") for line in frame.splitlines())


def test_the_keepalive_is_a_comment_with_neither_a_name_nor_an_id() -> None:
    """It holds an idle connection open without widening the published vocabulary."""
    text = KEEPALIVE_FRAME.decode()

    assert text.startswith(":")
    assert "event:" not in text
    assert "id:" not in text
    assert text.endswith("\n\n")


# --- the route -----------------------------------------------------------------------


async def test_a_fresh_stream_starts_at_the_first_event(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    _fill(log, session_id, 2)

    async with open_stream(app, stream_path, headers=_HEADERS) as live:
        assert live.status == 200
        assert live.headers["content-type"].startswith("text/event-stream")
        frames = await live.frames(2)

    assert _ids(frames) == ["id: 1", "id: 2"]


async def test_the_response_forbids_caching_and_proxy_buffering(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """Both headers exist so the frames reach the caller as they are produced: a cache
    would replay a stale prefix on a reconnect, and the common reverse proxy holds a
    response until it ends unless told not to -- which is the all-at-the-end delivery
    this route exists to avoid."""
    _fill(log, session_id, 1)

    async with open_stream(app, stream_path, headers=_HEADERS) as live:
        await live.frames(1)

        assert live.headers["cache-control"] == "no-store"
        assert live.headers["x-accel-buffering"] == "no"


async def test_an_event_appended_after_the_stream_opens_arrives_before_the_next_one(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """The live-tail claim, asserted by the order of the statements below.

    The first frame is read before the second event is appended, so it cannot have been
    delivered as part of a completed response -- the response has not completed and,
    with nothing ending the tail, never will.
    """
    async with open_stream(app, stream_path, headers=_HEADERS) as live:
        log.append(session_id, lifecycle.SESSION_CREATED)
        first = await live.frames(1)
        log.append(session_id, lifecycle.SESSION_SUSPENDED)
        second = await live.frames(1)

    assert first[0].splitlines()[:2] == [
        "id: 1",
        f"event: {lifecycle.SESSION_CREATED}",
    ]
    assert second[0].splitlines()[:2] == [
        "id: 2",
        f"event: {lifecycle.SESSION_SUSPENDED}",
    ]


async def test_reconnecting_continues_exclusive_of_the_id_supplied(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """Exclusive because the header states what the caller already holds."""
    _fill(log, session_id, 4)

    async with open_stream(
        app, stream_path, headers={**_HEADERS, "Last-Event-ID": "2"}
    ) as live:
        assert live.status == 200
        frames = await live.frames(2)

    assert _ids(frames) == ["id: 3", "id: 4"]
    received = "\n\n".join(frames)
    assert "id: 1" not in received
    assert "id: 2" not in received


async def test_the_query_parameter_positions_a_first_connection(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """No client sends `Last-Event-ID` on a first connection, so a caller that paged the
    log by range needs some way to say where it stopped -- otherwise it chooses between
    replaying from 1 and losing whatever landed in between."""
    _fill(log, session_id, 4)

    async with open_stream(app, stream_path, headers=_HEADERS, query="after=2") as live:
        frames = await live.frames(2)

    assert _ids(frames) == ["id: 3", "id: 4"]


async def test_a_reconnect_does_not_rewind_to_where_the_first_request_started(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """The two positions really do arrive together, because an SSE client reconnects by
    re-issuing the original URL and adding the header."""
    _fill(log, session_id, 4)

    async with open_stream(
        app,
        stream_path,
        headers={**_HEADERS, "Last-Event-ID": "3"},
        query="after=0",
    ) as live:
        frames = await live.frames(1)

    assert _ids(frames) == ["id: 4"]


async def test_a_caller_under_the_retained_floor_is_told_and_sent_no_events(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """Told, rather than handed the surviving suffix as though it were the span."""
    _fill(log, session_id, 5)
    log.expire_below(Seq(4))

    async with open_stream(app, stream_path, headers=_HEADERS) as live:
        assert live.status == 200
        frames = await live.frames(1)
        await live.ended()

    assert frames[0].startswith(f"event: {STREAM_ERROR}\n")
    refusal = _stream_error(frames[0])
    assert refusal.code is ErrorCode.EVENT_RANGE_EXPIRED
    assert refusal.retained_floor == 4
    assert "id:" not in frames[0]


async def test_a_caller_below_a_floor_that_swept_everything_is_told_not_left_waiting(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """This is the case the floor read at open exists for, and the only one.

    Where rows survive above the caller, the contiguity check would reach the same
    refusal a moment later, off the first row it sees. Here there is no such row: the
    sweep took the whole log, so the floor sits above every sequence the log holds and
    nothing is arriving for a contiguity check to notice. Without a floor read at the
    moment the stream opens, a caller in this position waits on events it will never be
    sent instead of being told they are gone.
    """
    _fill(log, session_id, 3)
    log.expire_below(Seq(4))

    async with open_stream(app, stream_path, headers=_HEADERS) as live:
        frames = await live.frames(1)
        await live.ended()

    refusal = _stream_error(frames[0])
    assert refusal.code is ErrorCode.EVENT_RANGE_EXPIRED
    assert refusal.retained_floor == 4


async def test_a_caller_sitting_exactly_at_the_floor_has_lost_nothing(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """The boundary the guard must not refuse.

    A caller holding through 2 asks for 3 onward, and 3 is the oldest event retained --
    the sweep took only what it had already read. Refusing here would tell a caller its
    position expired at the exact moment nothing of its was lost, and it has no way to
    tell that refusal from a real one.
    """
    _fill(log, session_id, 4)
    log.expire_below(Seq(3))

    async with open_stream(
        app, stream_path, headers={**_HEADERS, "Last-Event-ID": "2"}
    ) as live:
        frames = await live.frames(2)

    assert _ids(frames) == ["id: 3", "id: 4"]


async def test_the_floor_moving_under_an_open_stream_ends_it_with_a_refusal(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """The second way the floor gets above a caller, and the only one a frame reports.

    Reading frame 3 first is what puts the stream past its opening floor check, so the
    refusal below can only have come from the contiguity check: the next row the log
    offers is 5 when 4 was expected, and 4 is gone.
    """
    _fill(log, session_id, 2)

    async with open_stream(
        app, stream_path, headers={**_HEADERS, "Last-Event-ID": "2"}
    ) as live:
        log.append(session_id, lifecycle.SESSION_RESUMED)
        assert _ids(await live.frames(1)) == ["id: 3"]

        log.append(session_id, lifecycle.SESSION_SUSPENDED)
        log.append(session_id, lifecycle.SESSION_STOPPED)
        log.expire_below(Seq(5))

        frames = await live.frames(1)
        await live.ended()

    assert frames[0].startswith(f"event: {STREAM_ERROR}\n")
    refusal = _stream_error(frames[0])
    assert refusal.code is ErrorCode.EVENT_RANGE_EXPIRED
    assert refusal.retained_floor == 5
    assert "id: 5" not in frames[0]


async def test_a_row_whose_type_is_unpublished_never_appears_as_itself(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """A runtime name with no mapping is dropped, not carried through (ADR-013)."""
    log.append(session_id, lifecycle.SESSION_CREATED)
    log.append(session_id, "codex.thread.started")
    log.append(session_id, lifecycle.SESSION_STOPPED)

    async with open_stream(app, stream_path, headers=_HEADERS) as live:
        frames = await live.frames(2)

    assert _ids(frames) == ["id: 1", "id: 3"]
    assert "codex" not in "\n\n".join(frames)


async def test_an_idle_stream_holds_the_connection_open_with_a_comment(
    app: FastAPI, stream_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stream, "KEEPALIVE_INTERVAL_S", 0.05)

    async with open_stream(app, stream_path, headers=_HEADERS) as live:
        frames = await live.frames(2)

    assert frames == [": keepalive", ": keepalive"]


@pytest.mark.parametrize("value", ["seven", "-1", "1.5", "", "3,4"])
async def test_a_malformed_resume_position_is_refused_and_nothing_is_streamed(
    app: FastAPI, stream_path: str, value: str
) -> None:
    """Refused rather than defaulted: a resume answered with a replay is the one outcome
    the caller asking to continue cannot detect."""
    async with open_stream(
        app, stream_path, headers={**_HEADERS, "Last-Event-ID": value}
    ) as live:
        assert live.status == 400
        body = await live.body()

    assert "text/event-stream" not in live.headers["content-type"]
    assert "data: " not in body


async def test_a_request_with_no_tenant_is_refused(
    app: FastAPI, stream_path: str
) -> None:
    """The tenant dependency is really wired, not merely imported."""
    async with open_stream(app, stream_path, headers={}) as live:
        assert live.status == 400
        body = await live.body()

    assert "data: " not in body


async def test_another_tenants_session_and_an_absent_one_are_refused_alike(
    app: FastAPI, log: FakeLog, session_id: SessionId, stream_path: str
) -> None:
    """Nothing under this route would otherwise have raised.

    The Event Log is keyed by Session and carries no tenant, so following somebody
    else's Session succeeds and streams their events in full; the registry is the only
    thing on this path that knows the owner. The two refusals are graded together
    because telling them apart would turn the route into an existence oracle for
    anyone holding an id.
    """
    _fill(log, session_id, 2)
    stranger = {TENANT_HEADER: str(uuid4())}
    absent = f"/v1/sessions/{new_session_id()}/events/stream"

    async with open_stream(app, stream_path, headers=stranger) as theirs:
        assert theirs.status == 404
        their_body = await theirs.body()
    async with open_stream(app, absent, headers=_HEADERS) as missing:
        assert missing.status == 404
        missing_body = await missing.body()

    refusals = [
        PublicErrorEnvelope.model_validate_json(their_body),
        PublicErrorEnvelope.model_validate_json(missing_body),
    ]
    assert [refusal.error.code for refusal in refusals] == (
        [ErrorCode.SESSION_NOT_FOUND] * 2
    )
    assert refusals[0].error.message == refusals[1].error.message
    assert "data: " not in their_body


class UnusedWebhooks:
    """Satisfies the webhook store port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    webhook store would be grading something this file does not grade, and a quiet stub
    would let it pass while doing so.
    """

    async def register(
        self,
        tenant_id: TenantId,
        url: CallbackUrl,
        event_types: frozenset[str],
        secret_ref: str,
    ) -> WebhookRecord:
        raise AssertionError("a test in this file registered a webhook")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file listed a tenant's webhooks")

    async def delete(self, webhook_id: UUID, tenant_id: TenantId) -> bool:
        raise AssertionError("a test in this file deleted a webhook")

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]:
        raise AssertionError("a test in this file asked what watches a type")


class UnusedEnvironmentStore:
    """Satisfies the environment-store port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    environment store would be grading something this file does not grade, and a quiet
    stub would let it pass while doing so.
    """

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        raise AssertionError("a test in this file resolved an environment")
