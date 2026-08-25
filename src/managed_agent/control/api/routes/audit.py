"""GET /v1/audit/sessions/{session_id}/events — any Session's log, read by the platform.

What comes back is the tenant surface's page, produced by the tenant surface's own span
rule. That is not a shortcut: the three answers a sequence range can have — a span, an
empty page above the head, a refusal below the retained floor — are one rule, and a
second copy of it here would be a second contract, whose first divergence would show up
as a reviewer reading a Session differently from the team that owns it.

What belongs to this module is the authorization, and that is why it is a separate
router rather than a wider setting on the tenant one. The principal sits on the router
itself, so every route added here later is gated by construction rather than by whoever
adds it remembering.

Nothing here writes, and the router declares nothing but GET. That is what makes "the
record is available to them" and "nothing they read grants them the ability to act as
any tenant" true at the same time: the surface hands back events and opens no path that
changes one.

This module names no tenant anywhere, and that is the mechanical form of "holds no
tenant credential" — there is no dependency here that yields one, no registry call that
takes one, and nothing to forward. A reviewer addresses a Session by the identifier the
platform issued, and there is deliberately no Session collection on this router: the
tenant collection is narrowed by a term in the store's own query, and a cross-tenant
copy of it here would widen exactly the boundary this module was separated out to keep
narrow.

**Two dependencies on the router, in order, and the order is the design.** The first
authenticates -- it establishes a reviewer principal when the request presents a token
that proves one, and establishes nothing otherwise. The second authorizes, from the
claims on the request and nothing else. Every way of failing the first arrives at the
second as a request carrying no principal, so a stranger, a Session's own token, a
forged signature and an expired credential are one refusal with one code, and none of
them tells the caller which it was. See `control/reviewers/audit_reader.py` and
`control/api/request/reviewer_auth.py`.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from managed_agent.control.api.refusals import Refusal
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.reviewer_auth import establish_reviewer_principal
from managed_agent.control.api.routes.events import (
    EventPage,
    parse_span,
    read_span_of_any_session,
)
from managed_agent.control.reviewers.audit_reader import (
    REVIEWER_CLAIM,
    TENANT_CLAIM,
    AuditPrincipalRefused,
    PlatformReviewer,
    resolve_reviewer,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import SessionId

_LOG = logging.getLogger(__name__)

UNAUTHORIZED_MESSAGE = "this request is not authorized as a platform reviewer"
"""The one sentence every refused caller is given, whichever check they tripped.

Accurate for all of them, and that is why it is worded this way rather than as "carries
no reviewer": a request that presents a reviewer credential *and* a tenant principal
does carry one, and is still refused. A message that fitted only the common case would
be a message that quietly identified the uncommon one.
"""


def platform_reviewer_of(request: Request) -> PlatformReviewer:
    """The platform reviewer this request is authorized as, or a 401.

    Reads the claims off the request rather than verifying a credential itself: the
    dependency ahead of it on the router does that, so this function stays a decision
    about which principals the surface accepts. It never falls back to a tenant and
    never invents a reviewer, because either fallback would turn an unauthorized
    request into a cross-tenant read.

    The refusal travels in the one envelope every refusal on this API uses, and its code
    is `auth.audit_principal_unresolved`, a member of the published closed set since
    2026-08-24. It was outside that set on the argument that the set says what a call
    was refused FOR while this says the caller was never established as anyone -- which
    is a real distinction and was not a reason to leave the code unpublished: it reached
    callers either way, and being outside the enum meant nothing could check that it was
    spelled the same twice.

    **Two audiences, and the split is the point.** The platform's own log gets the words
    — which check failed, and on a request carrying what — and the caller gets one fixed
    sentence. Echoing the reason back was measured doing real harm on this surface: it
    told an unauthenticated caller whether their token had merely expired, whether they
    had presented a credential for the wrong surface, and whether the principal they
    carried was disqualified rather than absent. Each answer is a probe of a door that
    opens onto every tenant's history, and there is no caller who both needs the
    distinction and is entitled to it.
    """
    try:
        return resolve_reviewer(
            getattr(request.state, REVIEWER_CLAIM, None),
            getattr(request.state, TENANT_CLAIM, None),
        )
    except AuditPrincipalRefused as refused:
        _LOG.warning("audit read refused: %s", refused.reason)
        raise Refusal(
            ErrorCode.AUDIT_PRINCIPAL_UNRESOLVED, UNAUTHORIZED_MESSAGE
        ) from refused


router = APIRouter(
    tags=["audit"],
    # Authenticate, then authorize, and both on the router so a route added here later
    # inherits the pair rather than needing whoever adds it to remember. Ordered: the
    # first establishes the principal the second resolves, and FastAPI resolves router
    # dependencies in the order they are listed.
    dependencies=[
        Depends(establish_reviewer_principal),
        Depends(platform_reviewer_of),
    ],
)


@router.get(
    "/audit/sessions/{session_id}/events",
    response_model=EventPage,
    responses={
        STATUS_FOR[ErrorCode.EVENT_RANGE_EXPIRED]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
    },
)
async def read_audit_events(
    session_id: SessionId,
    request: Request,
    from_seq: Annotated[int, Query(ge=1)] = 1,
    to_seq: Annotated[int | None, Query(ge=1)] = None,
) -> EventPage | JSONResponse:
    """One span of one Session's Event Log, for a reader who holds no tenant credential.

    The range is not narrowed, widened or re-checked on the way through: the two shared
    functions own the span rule and the retained floor, and this one owns who is allowed
    to ask.

    One refusal fewer than the tenant route publishes, and it is a real difference in
    behaviour rather than an omission. There is no ownership refusal because there is no
    owner to check against, so a Session id that names nothing reads as an empty log
    rather than as a 404 — the reviewer addresses ids the platform issued, and an
    existence oracle over them tells a Control Plane reader nothing it is not already
    entitled to know.
    """
    span = parse_span(from_seq, to_seq)
    if isinstance(span, JSONResponse):
        return span
    start, end = span
    return await read_span_of_any_session(
        platform_from_request(request), session_id, start, end
    )
