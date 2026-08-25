"""The five thread routes: list, read, archive, events, stream.

ADR-007 said a subagent's activity should reach a tenant tagged with the thread that
produced it, and ADR-007 originally said a thread was not addressable.
Wave 3 built the tagging; the invariant was then revised to permit addressing a thread
**read-and-archive only**. That scope is the whole design constraint here: no verb on
this surface spawns a thread, so fan-out stays the model's decision and this platform
does not grow a second mechanism for it.

**Nothing here is a new store.** A thread is the events that carry its identifier, so
the read routes are the Session's own event surface with a predicate, and the two
existing helpers do the work: `events.read_span_of_any_session` answers a span and
`stream.frames` answers a tail. Both were already written as functions rather than route
bodies so that a second caller could not arrive at a different contract for the same
question -- this is that second caller.

**Ownership is checked against the Session registry on every route, before anything is
read.** The event log is keyed by Session and carries no tenant, and the thread index is
keyed by Session for the same reason, so nothing below the registry lookup would stop a
caller naming somebody else's Session id. A Session the registry will not show this
caller is refused with the code an absent one gets.

**The lookup is written out in each of the five routes rather than factored into a
helper, and that is deliberate.** It was a helper first. A guard in
`tests/control/test_tenancy.py` reads each route function's own source for the call and
failed two of these, and it is right to: the defect it exists to catch is a route that
takes the tenant, binds it, and then reads the log without using it -- and a reviewer
reading one route cannot see a check that happens one call away any more than that guard
can. Four repeated lines are the price, and every other route on this surface pays it.

**A thread id that names no thread of this Session is a 404 and not a 403**, even when
it names a real thread of a Session belonging to somebody else. The refusal must not be
usable to learn that an identifier exists elsewhere, which is the same rule the Session
refusal above follows and the reason both answer the same way.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.composition import Platform
from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.api.routes.events import (
    EventPage,
    parse_span,
    read_span_of_any_session,
)
from managed_agent.control.api.routes.stream import (
    BEFORE_FIRST,
    frames,
    resolve_resume_position,
)
from managed_agent.control.session.lifecycle import whole_log
from managed_agent.control.session.threads import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    ThreadActivity,
    ThreadStatus,
    status_of,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import Seq, SessionId, TenantId
from managed_agent.core.ports import SessionNotVisible
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import thread as thread_events

router = APIRouter(tags=["threads"])

THREAD_TYPE: Final = "session_thread"
"""What the resource calls itself on the wire, matching the upstream surface."""


class SessionThreadView(BaseModel):
    """One thread as a tenant reads it.

    **No `agent`.** Upstream carries a resolved roster entry per thread -- the agent's
    model, skills, tools and system prompt as they stood when the thread began. This
    platform has no multiagent roster: a Session resolves one definition and its
    subagents inherit it, so there is nothing per thread to snapshot. The field is
    absent rather than null, which is the distinction wave 0 settled for `prev_page`:
    null would claim this thread has no agent, and absent says this platform does not
    publish the fact.

    **No `stats` and no `usage`.** Both are per-thread accounting -- active seconds,
    tokens, cache reads -- and nothing here meters a thread. A Session's budget is
    metered whole. Publishing zeros would read as a thread that cost nothing.

    **`parent_thread_id` is absent on a thread the runtime never announced, and null
    only on one it announced with no parent.** The two are different facts and one field
    has to carry both, so the routes answer with `response_model_exclude_unset=True` and
    the field is simply not set in the first case. Measured against codex-cli 0.149.0 on
    2026-08-24: one delegating Turn produced six threads and one `thread.started`, so
    five of the six had no parent pointer anywhere in the log. Publishing null for those
    would have told a caller looking for the root that it had found six roots.

    Times are milliseconds, matching `SessionListed.created_at_ms`, so two creation
    times on one API are comparable without a conversion nobody documents.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    session_id: SessionId
    type: Literal["session_thread"] = THREAD_TYPE
    parent_thread_id: str | None = None
    status: ThreadStatus
    created_at_ms: int
    updated_at_ms: int
    archived_at_ms: int | None


class SessionThreadPage(BaseModel):
    """A page of threads and where the next one starts, or null when this is the last.

    `next_page` is always present. A field that appears only when more pages exist reads
    as the end of the collection to an SDK auto-paginator that finds it missing, which
    is the defect wave 0 fixed on the Session listing and the reason it is stated here.
    """

    model_config = ConfigDict(frozen=True)

    data: tuple[SessionThreadView, ...]
    next_page: str | None


def _encode_page(after: Seq) -> str:
    """The position after this thread's opening event, as an opaque token.

    Base64url with the padding stripped, the way every other cursor on this API is
    encoded: an `=` would be percent-encoded in a query string and come back looking
    unlike what was issued. Opaque rather than the bare number because a caller that
    can construct a position can construct one this surface never issued, and a sequence
    is an internal fact about the log rather than part of the contract.
    """
    return urlsafe_b64encode(str(int(after)).encode()).decode().rstrip("=")


def _decode_page(token: str) -> Seq | None:
    """The position a token names, or None when it is not one this surface issued.

    None rather than an exception: the one caller turns it straight into a refusal and
    has nothing else to do with the distinction, so an exception would be plumbing for
    a single `except` clause.
    """
    try:
        padded = token + "=" * (-len(token) % 4)
        return Seq(int(urlsafe_b64decode(padded.encode()).decode()))
    except ValueError:
        # binascii.Error and UnicodeDecodeError are both ValueError, so one clause
        # covers bad base64, bad utf-8 and a non-numeric body.
        return None


def _view(
    session_id: SessionId, activity: ThreadActivity, *, session_open: bool
) -> SessionThreadView:
    """One thread's facts and the Session's openness, as the published resource.

    `type` is passed explicitly even though it has a default, and that is load-bearing
    rather than tidiness: the routes answer with `response_model_exclude_unset=True` so
    that an unannounced thread's `parent_thread_id` can be absent, and under that rule a
    field left to its default is a field that disappears. Every key this resource always
    carries has to be set here.
    """
    parent = (
        {"parent_thread_id": activity.parent_thread_id}
        if activity.was_announced
        else {}
    )
    return SessionThreadView(
        id=activity.thread_id,
        session_id=session_id,
        type=THREAD_TYPE,
        status=status_of(activity, session_open=session_open),
        created_at_ms=activity.created_at_ms,
        updated_at_ms=activity.updated_at_ms,
        archived_at_ms=activity.archived_at_ms,
        **parent,
    )


def _no_such_session(session_id: SessionId) -> JSONResponse:
    return refuse(
        ErrorCode.SESSION_NOT_FOUND,
        "no such session is visible to this caller",
        session_id=str(session_id),
    )


def _no_such_thread(session_id: SessionId, thread_id: str) -> JSONResponse:
    return refuse(
        ErrorCode.THREAD_NOT_FOUND,
        "no such thread belongs to this session",
        session_id=str(session_id),
        thread_id=thread_id,
    )


async def _session_is_open(platform: Platform, session_id: SessionId) -> bool:
    """Whether this Session can still produce events, folded from its own log.

    Through `project` rather than a second rule of this module's own. The state is a
    fold over the log and there is exactly one implementation of that fold; a local
    "has a stop event" test would be a second answer to one question, and the first
    time the two disagreed a thread would read as running on a Session that had ended.

    A log with no creation event raises inside `project`. That cannot be reached from
    here -- the registry lookup every route does first is what proves the Session was
    created -- so there is nothing to catch and no fallback to write.
    """
    state, _ = project(await whole_log(platform.event_log_range, session_id))
    return state is SessionState.RUNNING


@router.get(
    "/sessions/{session_id}/threads",
    response_model=SessionThreadPage,
    response_model_exclude_unset=True,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]: {"model": PublicErrorEnvelope},
    },
)
async def list_threads(
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    page: str | None = None,
) -> SessionThreadPage | JSONResponse:
    """This Session's threads, oldest first, with the position of the next page.

    One extra row is asked for and never shown. That is what makes `next_page` mean
    "there is more" rather than "the page was full": a page returned exactly full with
    nothing behind it would otherwise hand the caller a token that opens an empty page,
    and a caller cannot tell that from a page it has not read yet.

    A Session that has never run a multiagent Turn, or whose events predate attribution,
    has no threads and answers an empty page. That is not a refusal -- there is nothing
    wrong with a Session that never delegated.
    """
    platform = platform_from_request(request)
    after: Seq | None = None
    if page is not None:
        after = _decode_page(page)
        if after is None:
            return refuse(
                ErrorCode.PAGINATION_CURSOR_INVALID,
                "this is not a page token this surface issued",
            )
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return _no_such_session(session_id)
    walked = await platform.session_threads.threads_of(
        session_id, after_seq=after, limit=limit + 1
    )
    shown = tuple(walked[:limit])
    session_open = await _session_is_open(platform, session_id)
    return SessionThreadPage(
        data=tuple(_view(session_id, one, session_open=session_open) for one in shown),
        next_page=(
            _encode_page(shown[-1].started_seq) if len(walked) > limit else None
        ),
    )


@router.get(
    "/sessions/{session_id}/threads/{thread_id}",
    response_model=SessionThreadView,
    response_model_exclude_unset=True,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.THREAD_NOT_FOUND]: {"model": PublicErrorEnvelope},
    },
)
async def read_thread(
    session_id: SessionId,
    thread_id: str,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SessionThreadView | JSONResponse:
    """One named thread of this Session."""
    platform = platform_from_request(request)
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return _no_such_session(session_id)
    activity = await platform.session_threads.thread_at(session_id, thread_id)
    if activity is None:
        return _no_such_thread(session_id, thread_id)
    session_open = await _session_is_open(platform, session_id)
    return _view(session_id, activity, session_open=session_open)


@router.post(
    "/sessions/{session_id}/threads/{thread_id}/archive",
    status_code=status.HTTP_200_OK,
    response_model=SessionThreadView,
    response_model_exclude_unset=True,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.THREAD_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.THREAD_RUNNING]: {"model": PublicErrorEnvelope},
    },
)
async def archive_thread(
    session_id: SessionId,
    thread_id: str,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SessionThreadView | JSONResponse:
    """Record that this thread will publish nothing more.

    **This archives the platform's record and reaches no runtime.** Upstream's archive
    frees a slot against a live-thread ceiling; there is no slot to free here, because
    this platform never keeps the runtime's own thread identifier -- MAP-A10 forbids one
    reaching a caller and wave 3 chose to derive the published id rather than store the
    original beside it. Under this platform's model there is also nothing to reclaim: a
    subagent thread is spawned inside a Turn and cannot outlive it, so a thread whose
    Turn has closed is holding nothing.

    **Refused while the thread's Turn is open**, which is upstream's rule too: a thread
    is archivable only once it is idle. The reason is not politeness about ordering. The
    runtime is still producing frames for that thread, every one of them will be
    appended, and an archive already in the log would tell a consumer to stop reading
    before the events it was waiting for arrived. The caller closes the Turn first.

    **Terminal and idempotent.** A thread already archived answers with the first
    archive's timestamp rather than moving it, so a client whose call timed out and
    retried cannot put two retirements of one thread in a log that cannot express which
    of them counted.

    **A closed Session appends nothing and answers 200.** Its threads are already
    terminated -- no further event can carry their identifiers -- so the archive has
    nothing left to guarantee, and refusing would make a caller tidying up after a
    finished Session handle a failure that means "already done".
    """
    platform = platform_from_request(request)
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return _no_such_session(session_id)
    activity = await platform.session_threads.thread_at(session_id, thread_id)
    if activity is None:
        return _no_such_thread(session_id, thread_id)
    session_open = await _session_is_open(platform, session_id)
    if activity.archived_at_ms is not None or not session_open:
        return _view(session_id, activity, session_open=session_open)
    if not activity.turn_ended:
        return refuse(
            ErrorCode.THREAD_RUNNING,
            "this thread's Turn is still running; interrupt it before archiving",
            session_id=str(session_id),
            thread_id=thread_id,
        )
    await platform.event_log_append.append(
        session_id,
        thread_events.THREAD_ARCHIVED,
        thread_events.ThreadArchived(thread_id=thread_id).model_dump(),
    )
    # Re-read rather than stamping a clock here. The published `archived_at_ms` is the
    # appended row's own timestamp, so the value this answers with is the same one every
    # later read of the same thread returns -- two clocks for one fact could not agree.
    written = await platform.session_threads.thread_at(session_id, thread_id)
    if written is None:
        return _no_such_thread(session_id, thread_id)
    return _view(session_id, written, session_open=session_open)


@router.get(
    "/sessions/{session_id}/threads/{thread_id}/events",
    response_model=EventPage,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.THREAD_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.EVENT_RANGE_EXPIRED]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
    },
)
async def read_thread_events(
    session_id: SessionId,
    thread_id: str,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    from_seq: Annotated[int, Query(ge=1)] = 1,
    to_seq: Annotated[int | None, Query(ge=1)] = None,
) -> EventPage | JSONResponse:
    """One span of this Session's events, narrowed to the ones this thread produced.

    **A sequence range, not a cursor**, unlike upstream's `limit`/`page` on the same
    route. ADR-013 chose a range for the Session's event surface, and this is that
    surface with a predicate: two paging disciplines over one log would be the drift a
    single contract exists to prevent, and a caller that pages the Session's events and
    then narrows to a thread would otherwise have to hold two kinds of position.

    **The page is short and the span is not.** `from_seq` and `to_seq` come back as
    asked, and the events between them are only this thread's, so a caller reads the
    span it named and pages by the span rather than by the count. That is what keeps the
    filter honest: a page trimmed to a full `limit` of thread events would hide how far
    through the log it had reached, and a caller could not tell a quiet thread from the
    end of the log.

    The range is checked before the thread is looked up, because a malformed range gets
    the same 400 for a thread that exists, one that belongs elsewhere, and one that
    never did -- so it says nothing about which.
    """
    platform = platform_from_request(request)
    span = parse_span(from_seq, to_seq)
    if isinstance(span, JSONResponse):
        return span
    start, end = span
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return _no_such_session(session_id)
    if await platform.session_threads.thread_at(session_id, thread_id) is None:
        return _no_such_thread(session_id, thread_id)
    page = await read_span_of_any_session(platform, session_id, start, end)
    if isinstance(page, JSONResponse):
        return page
    return page.model_copy(
        update={
            "events": tuple(
                event
                for event in page.events
                if event.payload.get("thread_id") == thread_id
            )
        }
    )


@router.get(
    "/sessions/{session_id}/threads/{thread_id}/stream",
    response_model=None,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.THREAD_NOT_FOUND]: {"model": PublicErrorEnvelope},
    },
)
async def stream_thread_events(
    session_id: SessionId,
    thread_id: str,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    after: Annotated[int | None, Query(ge=BEFORE_FIRST)] = None,
) -> StreamingResponse | JSONResponse:
    """The live tail of one thread, as the Session's tail with a predicate.

    The SSE id stays the Session's sequence, not a per-thread counter. That is what
    makes `Last-Event-ID` work here at all: the resume position has to name a place in
    the log, and a thread-local ordinal would name a place only this route could
    resolve -- and could not resolve at all across a retention sweep.

    A caller therefore sees gaps in the ids, and must: the numbers it does not see are
    events of other threads. Documented rather than hidden, because a consumer that
    treats consecutive ids as a completeness check would read every gap as loss.
    """
    platform = platform_from_request(request)
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return _no_such_session(session_id)
    if await platform.session_threads.thread_at(session_id, thread_id) is None:
        return _no_such_thread(session_id, thread_id)
    return StreamingResponse(
        frames(
            platform.event_log_range,
            session_id,
            resolve_resume_position(None, after),
            thread_id=thread_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
