"""GET /v1/sessions/{session_id}/events — a span of one Session's Event Log.

Pagination is by sequence range rather than by cursor: the sequence is already strictly
ordered and contiguous per Session, so a range is stable across re-reads, is expressible
directly in a URL, and a caller that lost its place can name the place it lost
(ADR-013).

Three answers, kept distinct:

- A range above the head is an empty page. Those events do not exist yet and nothing is
  wrong, so this is a 200 with no events rather than a refusal.
- A range below the retained floor is a refusal. Those events existed and expired, and a
  caller told "empty" would read lost history as no history.
- A range that spans the floor is the same refusal, not a partial page, because a short
  span returned as though it were the requested one is indistinguishable from a complete
  answer.

The floor travels back with every page, so a caller re-paging after an expiry can move
to a place that still exists instead of bisecting for it.

Ownership is checked against the Session registry before any event is read, and it has
to be: the Event Log is keyed by Session and carries no tenant, so a range read over
somebody else's Session succeeds and hands back their events with nothing raising. The
registry is the only thing here that knows the owner. A Session it will not show this
caller is refused with the same code an absent one gets, so the refusal cannot be used
to learn whether an id names another tenant's Session.

The two query parameters are parsed as plain ints and converted, not annotated as `Seq`
directly: `Seq` is strict, and a strict int rejects the string a URL actually delivers.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.composition import Platform
from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import Seq, SessionId, TenantId
from managed_agent.core.ports import SessionNotVisible

router = APIRouter(tags=["events"])

MAX_RANGE = 1000
"""Widest span one page returns. A wider ask is refused rather than quietly narrowed,
because a truncated page a caller cannot tell from a complete one breaks the guarantee
that a range comes back with nothing added and nothing omitted."""


class EventView(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: Seq
    type: str
    payload: dict[str, object]


class EventPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: SessionId
    from_seq: Seq
    to_seq: Seq
    retained_floor: Seq
    events: tuple[EventView, ...]


def parse_span(from_seq: int, to_seq: int | None) -> tuple[Seq, Seq] | JSONResponse:
    """The two ends of the span a caller asked for, or the refusal that says why not.

    An absent `to_seq` means "as far as one page reaches", which is `MAX_RANGE`
    sequences from the start and not the head of the log — the head has no number until
    it is written, and a page whose width depended on it would differ between two reads
    of the same range.

    A refusal comes back as a value rather than an exception, the way every other
    refusal on this surface does, so a caller composes it into an answer instead of
    unwinding. It is returned separately from the read below because the two are
    ordered around a check this function knows nothing about: the tenant surface refuses
    a malformed range *before* it looks a Session up, so that the answer to a bad range
    is the same for an existing Session, another tenant's, and one that never existed —
    and therefore carries no signal about which of those an id names.
    """
    start = Seq(from_seq)
    end = Seq(to_seq) if to_seq is not None else Seq(start + MAX_RANGE - 1)
    if end < start or end - start + 1 > MAX_RANGE:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            f"a range must run forwards and span at most {MAX_RANGE} sequences",
            from_seq=start,
            to_seq=end,
        )
    return start, end


async def read_span_of_any_session(
    platform: Platform, session_id: SessionId, start: Seq, end: Seq
) -> EventPage | JSONResponse:
    """A retained span of one Session's log, keyed by Session id and by nothing else.

    **This applies no tenant predicate and cannot.** The Event Log is keyed by Session
    and carries no tenant, so this returns any Session's events to whoever names the id.
    Every caller is responsible for having already established that its caller may
    address this Session — the tenant surface does that with a Session-registry lookup
    before it gets here, and the Control Plane audit surface is the one caller for which
    there is deliberately nothing to establish, because it is authorized as a reader of
    all tenants and holds no tenant credential to check against.

    It exists as a function of its own so that the three answers a sequence range can
    have — a span, an empty page above the head, a refusal below the retained floor —
    are one rule with one implementation. A second copy would be a second contract,
    whose first divergence would show up as a reviewer reading a Session differently
    from the team that owns it.

    The read passes the span's own width as its limit. The port caps a read and treats a
    short result as "page for the rest", so a caller naming a limit smaller than its
    range gets a page it cannot tell from a complete answer. A range holds at most
    `end - start + 1` events, because `(session_id, seq)` is the log's primary key and a
    sequence therefore appears at most once — so a limit of exactly that width can never
    truncate, and nothing here pages internally.
    """
    floor = await platform.event_log_range.retained_floor(session_id)
    if start < floor:
        return refuse(
            ErrorCode.EVENT_RANGE_EXPIRED,
            "the requested range is no longer retained",
            from_seq=start,
            retained_floor=floor,
        )
    rows = await platform.event_log_range.read(
        session_id, start, end, limit=end - start + 1
    )
    return EventPage(
        session_id=session_id,
        from_seq=start,
        to_seq=end,
        retained_floor=floor,
        events=tuple(
            EventView(seq=row.seq, type=row.type, payload=row.payload) for row in rows
        ),
    )


@router.get(
    "/sessions/{session_id}/events",
    response_model=EventPage,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.EVENT_RANGE_EXPIRED]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
    },
)
async def read_events(
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    from_seq: Annotated[int, Query(ge=1)] = 1,
    to_seq: Annotated[int | None, Query(ge=1)] = None,
) -> EventPage | JSONResponse:
    """One span of a Session's events, or the one refusal that says why not.

    The range is checked before the Session is looked up, which is safe because the
    answer to a malformed range is the same for every id — an existing Session, another
    tenant's, and one that was never created all answer 400 — so it carries no signal
    about which of those an id names.

    What this route owns is that check and nothing else. The span rule and the retained
    floor live in `read_span_of_any_session`, which is deliberately tenant-blind, so the
    registry lookup above is the only thing standing between a caller and another
    tenant's events — remove it and this becomes an unscoped cross-tenant read that
    still passes every test about ranges.
    """
    platform = platform_from_request(request)
    span = parse_span(from_seq, to_seq)
    if isinstance(span, JSONResponse):
        return span
    start, end = span
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return refuse(
            ErrorCode.SESSION_NOT_FOUND,
            "no such session is visible to this caller",
            session_id=str(session_id),
        )
    return await read_span_of_any_session(platform, session_id, start, end)
