"""POST /v1/sessions/{session_id}/events — submit one Turn.

Three outcomes, three status codes, because a client that retried needs to learn which
one it got: 202 for a Turn that was just started, 200 for a key that already had one,
409 for a Session that cannot take one. The turn id is the same in the first two cases,
which is what makes the retry safe to treat as the original.

The key is a required header rather than an optional one. Without one a timed-out retry
silently spends twice, and that is the one place on this surface where a network failure
costs money -- so the permissive behaviour has no opt-in at all.

Ownership is checked against the Session registry before anything is read or written,
and it has to be: the Event Log is keyed by Session and carries no tenant, so an
admission decision folded over somebody else's Session succeeds and appends into their
log with nothing raising. The registry is the only thing here that knows the owner. A
Session it will not show this caller is refused with the same code an absent one gets,
so the refusal cannot be used to learn whether an id names another tenant's Session.

The Turn is recorded before it is dispatched. Dispatch is the step that can fail without
the platform knowing what the pod did, so it happens after the log already says a Turn
was asked for; a dispatch that cannot reach the pod closes the Turn as failed rather
than leaving a submission nobody will ever explain.
"""

import logging
from typing import Annotated, assert_never

from fastapi import APIRouter, Depends, Header, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.session.lifecycle import (
    TurnAdmitted,
    TurnRefused,
    TurnReplayed,
    admit_turn,
)
from managed_agent.control.session.turn_execution import run_turn, task_name
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import Seq, SessionId, TenantId, TurnId
from managed_agent.core.ports import SessionNotVisible
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import turn

_LOG = logging.getLogger(__name__)

router = APIRouter(tags=["turns"])


class SubmitTurn(BaseModel):
    """What a caller sends. The key is a header, not a field of this.

    Unknown fields are refused rather than ignored, so a caller that put its
    idempotency key in the body would be told rather than served a second Turn.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(min_length=1)


class TurnView(BaseModel):
    """The Turn a submission got, and where its first event sits in the log."""

    model_config = ConfigDict(frozen=True)

    turn_id: TurnId
    seq: Seq


@router.post(
    "/sessions/{session_id}/events",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TurnView,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.SESSION_NOT_ACCEPTING_TURNS]: {
            "model": PublicErrorEnvelope
        },
    },
)
async def submit(
    session_id: SessionId,
    body: SubmitTurn,
    request: Request,
    response: Response,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    idempotency_key: Annotated[turn.IdempotencyKey, Header(alias="Idempotency-Key")],
) -> TurnView | JSONResponse:
    """Submit one Turn, or the one refusal that says why not.

    The refusal for a Session that cannot take a Turn carries the state under a name in
    `detail` rather than a code of its own per state. One published code and a named
    fact is what a consumer can branch on exhaustively; a code per state would grow the
    published set every time the state machine did, and each addition is a version
    event under ADR-013.
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

    admission = await admit_turn(
        session_id,
        idempotency_key,
        body.prompt,
        platform.event_log_append,
        platform.event_log_range,
    )
    match admission:
        case TurnRefused(state=state):
            # One code, two sentences, because the two refusals need opposite remedies.
            # A stop is permanent and the caller needs a new Session; a Turn already
            # executing clears on its own, and the caller waits or interrupts. Saying
            # only "accepts no Turn" for the second reads as the first, and would send
            # somebody off to recreate a Session that was about to be free.
            explained = (
                "a Turn is already running on this session; wait for it to finish or "
                "interrupt it"
                if state is SessionState.RUNNING
                else f"this session is {state.value} and accepts no Turn"
            )
            return refuse(
                ErrorCode.SESSION_NOT_ACCEPTING_TURNS,
                explained,
                session_id=str(session_id),
                state=state.value,
            )
        case TurnReplayed(turn_id=replayed, seq=seq):
            response.status_code = status.HTTP_200_OK
            return TurnView(turn_id=replayed, seq=seq)
        case TurnAdmitted(turn_id=admitted, seq=seq):
            # Started, not awaited, and the 202 above stops being a promise this route
            # breaks. What the Turn does and how it ends belong to
            # `control/session/turn_execution.py` now -- including the `turn.failed`
            # that closes it, which cannot stay here: a Turn that fails after this
            # response has been sent would record nothing, and an open Turn refuses the
            # Session's next Turn and its archive for ever.
            platform.background_turns.start(
                run_turn(
                    platform.turn_dispatch,
                    platform.event_log_append,
                    session_id,
                    admitted,
                    body.prompt,
                ),
                name=task_name(admitted),
            )
            return TurnView(turn_id=admitted, seq=seq)
        case _:
            assert_never(admission)
