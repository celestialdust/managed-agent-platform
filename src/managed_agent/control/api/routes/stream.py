"""GET /v1/sessions/{session_id}/events/stream -- the live tail of one Session's log.

The sequence number is the SSE event id, so the resume mechanism every HTTP client
already ships is the one this surface has: a caller whose connection dropped mid-Turn
reconnects with `Last-Event-ID` and continues. Resume is exclusive of the id supplied --
the caller is stating what it already holds, not what it wants next -- so a clean
reconnect repeats nothing.

Delivery is still only at-least-once, and that is not a contradiction. A frame can be
handed to a socket that dies before the peer reads it, and this end cannot tell that
from one that arrived; the id is what makes a repeat cheap to detect rather than
something to prevent. A consumer that treats the sequence as a primary key is correct
under both sentences, which is why the id is committed and the absence of repeats is
not.

Ownership is checked against the Session registry before the response begins, and it has
to be: the Event Log is keyed by Session and carries no tenant, so following somebody
else's Session would succeed and stream their events with nothing raising. The registry
is the only thing on this path that knows the owner. A Session it will not show this
caller is refused with the code an absent one gets, so the refusal cannot be used to
learn whether an id names another tenant's Session.

Two refusals, two shapes, and the split is on when each becomes knowable. Ownership is
settled before a byte is written, so it is an HTTP status. A position below the retained
floor is a frame instead, because a retention sweep can move the floor while the stream
is held open, by which point the status line is long gone -- and one shape for that
condition is worth more than a status code available in only one of its two cases.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, TenantId
from managed_agent.core.ports import EventLogRange, EventRecord, SessionNotVisible
from managed_agent.core.vocabulary import is_published
from managed_agent.core.vocabulary.stream import STREAM_ERROR, StreamError

router = APIRouter(tags=["events"])

BEFORE_FIRST: Final[int] = FIRST_SEQ - 1
"""Where a caller that has read nothing starts.

Deliberately a plain int rather than a Seq: a Session's sequence is contiguous from 1,
so the position before the first event is not itself a sequence number, and giving it
that type would let it be handed to anything that expects a real one.
"""

KEEPALIVE_INTERVAL_S: Final[float] = 15.0
"""How long an idle stream waits before a comment goes out to hold the connection."""

KEEPALIVE_FRAME: Final[bytes] = b": keepalive\n\n"
"""An SSE comment: no event name and no id, so holding a connection open neither widens
the published vocabulary nor puts a number in the caller's resume position."""


def resolve_resume_position(last_event_id: int | None, after: int | None) -> int:
    """Reduce the two places a caller can state its position to one sequence number.

    `Last-Event-ID` beats the query parameter, and the order is load-bearing rather than
    a tie-break: a reconnecting client re-issues the original URL, query string
    included, and adds the header. If the parameter won, every reconnect would rewind to
    wherever the first request started and redeliver the whole span.

    `after` exists for the first connection, where no client sends the header: a caller
    that paged the log by range and now wants to go live has to be able to say where it
    stopped, or it chooses between replaying from 1 and losing whatever landed in
    between.

    Both name a position already read, so both are exclusive, and `BEFORE_FIRST` is what
    "I have read nothing" reduces to.
    """
    if last_event_id is not None:
        return last_event_id
    return BEFORE_FIRST if after is None else after


def encode_event(seq: Seq, type_: str, payload: dict[str, object]) -> bytes:
    """Frame one log row, with the row's own sequence as the SSE id.

    The payload is serialized rather than interpolated, which is what keeps a value
    containing a newline from ending the frame early: JSON escapes it, so the data line
    stays one line. The event name is not escaped and does not need to be -- the only
    caller passes a type that is in the published set, and every member of that set is a
    dotted identifier.
    """
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return f"id: {seq}\nevent: {type_}\ndata: {data}\n\n".encode()


def encode_stream_error(error: StreamError) -> bytes:
    """Frame the one event this surface originates, and the only one carrying no id.

    An id here would enter the caller's resume position, and its next reconnect would
    ask to continue after a number the log does not contain.
    """
    return f"event: {STREAM_ERROR}\ndata: {error.model_dump_json()}\n\n".encode()


def _below_floor(requested: int, floor: Seq) -> bytes:
    """The refusal frame for a caller asking below what this Session still retains."""
    return encode_stream_error(
        StreamError(
            code=ErrorCode.EVENT_RANGE_EXPIRED,
            message=(
                f"sequence {requested} has expired; this Session retains from {floor}"
            ),
            retained_floor=floor,
        )
    )


def _belongs(row: EventRecord, thread_id: str | None) -> bool:
    """Whether this row goes to a caller following `thread_id`, or every caller.

    `None` means the whole Session, so it admits every row including the ones that carry
    no thread at all -- a Session's own lifecycle events are nobody's thread and a
    Session stream that dropped them would lose the events it exists to deliver.

    A named thread admits only an exact match. An event with no `thread_id` is **not**
    admitted, and that is the choice worth stating: the Session's creation, its
    placement, its stop, and the Turn events appended before attribution existed all
    have none, and handing them to a thread follower would attribute the Session's
    history to whichever thread happened to be watched.
    """
    if thread_id is None:
        return True
    return row.payload.get("thread_id") == thread_id


async def frames(
    log: EventLogRange,
    session_id: SessionId,
    after: int,
    *,
    thread_id: str | None = None,
) -> AsyncIterator[bytes]:
    """Frame everything above `after`, then each row as it lands, until the caller goes.

    `thread_id` narrows what is *sent* and deliberately not what is *read*. The floor
    check below works by contiguity -- a Session's sequence has no gaps, so a row whose
    sequence is not the expected one means the rows between are gone -- and that only
    holds over the unfiltered sequence. Filtering the read instead would make every
    event of another thread look like an expiry, and the stream would refuse itself on
    its first frame from a sibling.

    The retained floor can be above the caller in two ways. It can already be there when
    the stream opens; or a retention sweep can move it while the stream is held open by
    a consumer that is not reading, by which point the response has begun and a frame is
    the only way left to say so. The second is caught by contiguity: a Session's
    sequence has no gaps, so a row whose sequence is not the expected one means the rows
    between them are gone, and carrying on would hand back a later range as though it
    were contiguous.

    A position above the head of the log is neither a refusal nor an error -- those
    events do not exist yet, so the stream waits for them, which is the whole point of a
    tail.

    Nothing here ends the stream at a Turn boundary. The terminating event a caller
    waits for is a log row like every other, so one connection spans as many Turns as
    the caller holds it open for.
    """
    floor = await log.retained_floor(session_id)
    if floor > after + 1:
        yield _below_floor(after + 1, floor)
        return

    expected = after + 1
    tail = log.follow(session_id, Seq(after))
    pending: asyncio.Task[EventRecord] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(tail))
            # `asyncio.wait` rather than a timeout around the await: a timeout cancels
            # what it waits on, and cancelling a half-run `__anext__` leaves the
            # iterator unusable. This leaves one read in flight across as many
            # keepalives as it takes.
            done, _ = await asyncio.wait({pending}, timeout=KEEPALIVE_INTERVAL_S)
            if not done:
                yield KEEPALIVE_FRAME
                continue
            try:
                row = pending.result()
            except StopAsyncIteration:
                return
            finally:
                pending = None
            if row.seq != expected:
                yield _below_floor(expected, await log.retained_floor(session_id))
                return
            expected = row.seq + 1
            # A type outside the published set never leaves as itself (ADR-013), and
            # dropping it is the fail-safe half of that rule. Nothing here reports the
            # drop: the log should hold only mapped types, so a drop means a defect
            # upstream of this route, and this route is not where it would be noticed.
            if is_published(row.type) and _belongs(row, thread_id):
                yield encode_event(row.seq, row.type, row.payload)
    finally:
        if pending is not None:
            pending.cancel()


@router.get(
    "/sessions/{session_id}/events/stream",
    response_model=None,
    responses={STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope}},
)
async def stream_events(
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    last_event_id: Annotated[
        int | None, Header(alias="Last-Event-ID", ge=BEFORE_FIRST)
    ] = None,
    after: Annotated[int | None, Query(ge=BEFORE_FIRST)] = None,
) -> StreamingResponse | JSONResponse:
    """The live tail of one Session's events, or the status that says why not.

    A malformed resume position is refused by the parameter types before this body runs
    rather than defaulted here. A resume request answered with a replay is the one
    outcome the caller asking to continue cannot detect, so guessing at what a
    non-numeric `Last-Event-ID` meant is the wrong direction to fail in.

    `response_model=None` is on the decorator because the return type is a union of two
    responses; without it FastAPI reads that annotation as a body schema and refuses the
    route at definition time.
    """
    platform = platform_from_request(request)
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return refuse(
            ErrorCode.SESSION_NOT_FOUND,
            "no such session is visible to this caller",
            session_id=str(session_id),
        )
    return StreamingResponse(
        frames(
            platform.event_log_range,
            session_id,
            resolve_resume_position(last_event_id, after),
        ),
        media_type="text/event-stream",
        # no-store stops an intermediary replaying a stale prefix on a reconnect, and
        # X-Accel-Buffering stops the common reverse proxy holding frames until the
        # response ends -- which is the all-at-the-end delivery this route exists to
        # avoid.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
