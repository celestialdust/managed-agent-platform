"""POST /v1/sessions/{session_id}/tool_calls/{call_id}/result — answer one tool call.

A tool call the platform cannot complete on its own puts its question on the Session's
Event Log and waits there for the answer. This is the inbound half of that: the answer
arrives on the surface the tenant already holds, is written into the same log, and the
waiting service picks it up by following the log forward. Nothing here talks to the Tool
Gateway, which is a different process and has no inbound route of its own.

**A generic tool-result envelope rather than an elicitation-shaped one.** What the body
carries -- content items and whether the call succeeded -- is the shape a dynamic tool
call is answered with, and a question put to the Session is normalised onto that same
envelope as a call of `request_user_input`. One inbound shape for both means a tenant
implements one thing, and a second kind of ask later is a new question shape rather than
a second route.

**The answer is accepted whether or not anybody is still waiting for it.** The wait
inside the tool call is bounded, and when it runs out the ask is cancelled and the Turn
ends so the pod can be reclaimed. The answer is still worth recording after that: it is
in the log, and the Session's next Turn reads it. Refusing a late answer would leave the
tenant holding a value with nowhere to put it (ADR-032).

**A question marked secret is refused outright, and no part of the answer is written.**
The Event Log is retained on the tenant's retention clock and the Rollout is copied out
of the pod at every Turn, so a secret answered here would come to rest in two places
that are not built to hold one. The refusal names the Vault surface, which is where a
value like that belongs and where a tool already reads its credentials from. What this
cannot do is stop the question being asked in the first place -- that belongs to
whatever normalises an ask onto the envelope, and refusing here is what keeps the value
out of the two stores regardless.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.session.lifecycle import whole_log
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import Seq, SessionId, TenantId
from managed_agent.core.ports import EventRecord, SessionNotVisible
from managed_agent.core.session.event_append import append_in_order
from managed_agent.core.vocabulary import tool_in_flight

router = APIRouter(tags=["tool results"])


class TextContentItem(BaseModel):
    """One answer, as text. The only content item this route serves.

    The generic envelope also carries image and audio items, and an answer to a question
    is words -- so the other two are refused at the boundary by having nowhere to parse
    into, rather than accepted and dropped somewhere below where the caller would never
    learn their answer went nowhere.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["input_text"] = "input_text"
    text: str


class ToolCallResult(BaseModel):
    """What a caller sends: the content items, and whether the call succeeded.

    `success` is required rather than defaulted. It decides whether the content is an
    answer or an account of why there is none, and a default would pick one of those for
    a caller who did not say.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_items: tuple[TextContentItem, ...]
    success: bool


class ResultRecorded(BaseModel):
    """Where the answer landed in the Session's log."""

    model_config = ConfigDict(frozen=True)

    seq: Seq


@dataclass(frozen=True, slots=True)
class Question:
    """One thing a tool call is waiting to be told.

    `id` is the key the answer is filed under, so the service that asked can hand the
    values back in the shape it promised its caller. `is_secret` is a claim by whatever
    put the question that the answer is a credential.
    """

    id: str
    is_secret: bool


@dataclass(frozen=True, slots=True)
class Ask:
    """One open question of a Session, parsed out of the event that recorded it."""

    elicitation_id: str
    questions: tuple[Question, ...]


def _questions_in(payload: Mapping[str, object]) -> tuple[Question, ...]:
    """The questions one ask puts, in the order their answers are expected.

    Two shapes reach here and they are read into one value, which is what lets
    everything below this line be written against questions rather than against
    payloads.

    The envelope shape states its questions outright, each with an id of its own and the
    secret flag this route turns on. The form shape states a JSON schema instead, whose
    property names are the keys the answer has to be filed under and whose order is the
    order they were declared in -- a schema has no way to say a field is a secret, so
    nothing read out of one is treated as one.

    An unreadable question is dropped rather than guessed at. A payload that yields no
    questions at all leaves the ask unanswerable, which the caller is told about; that
    is the safe end of this, because the other end files an answer under a key nobody
    asked for and hands it to a registered server as though somebody had.
    """
    arguments = payload.get("arguments")
    if isinstance(arguments, Mapping):
        stated = arguments.get("questions")
        if isinstance(stated, Sequence) and not isinstance(stated, str | bytes):
            return tuple(
                Question(id=str(item["id"]), is_secret=bool(item.get("is_secret")))
                for item in stated
                if isinstance(item, Mapping) and "id" in item
            )
    schema = payload.get("requested_schema")
    if isinstance(schema, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            return tuple(Question(id=str(name), is_secret=False) for name in properties)
    return ()


def ask_named(events: Iterable[EventRecord], call_id: str) -> Ask | None:
    """The open question this call id names, or None if the Session put no such one.

    The first match wins, and there can only be one: the id is minted per ask by the
    service that put the question, so two events carrying it would be one event written
    twice.
    """
    for event in events:
        if event.type != tool_in_flight.TOOL_ELICITATION_REQUESTED:
            continue
        if str(event.payload.get("elicitation_id", "")) != call_id:
            continue
        return Ask(elicitation_id=call_id, questions=_questions_in(event.payload))
    return None


def _answer_for(ask: Ask, result: ToolCallResult) -> dict[str, object] | JSONResponse:
    """The payload the waiting service reads, or the refusal saying there is none.

    Content only ever crosses into the log on a success. A result that did not succeed
    carries the caller's account of the failure rather than an answer, and the action it
    becomes carries nothing at all -- so filing that text as the answer would hand a
    registered server a value nobody supplied as one.

    The pairing is by position and the counts must agree. Both ends state their order --
    the questions as they were asked, the items as they were sent -- and a short list
    paired anyway would leave a question unanswered while the answer still read as
    complete.
    """
    if not result.success:
        return {
            "elicitation_id": ask.elicitation_id,
            "action": "decline",
            "content": {},
        }
    if len(result.content_items) != len(ask.questions):
        return refuse(
            ErrorCode.REQUEST_INVALID,
            "a result answers every question of the call it names, in order",
            call_id=ask.elicitation_id,
            questions=len(ask.questions),
            content_items=len(result.content_items),
        )
    return {
        "elicitation_id": ask.elicitation_id,
        "action": "accept",
        "content": {
            question.id: item.text
            for question, item in zip(ask.questions, result.content_items, strict=True)
        },
    }


@router.post(
    "/sessions/{session_id}/tool_calls/{call_id}/result",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ResultRecorded,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.ELICITATION_SECRET_REFUSED]: {
            "model": PublicErrorEnvelope
        },
    },
)
async def record_result(
    session_id: SessionId,
    call_id: str,
    body: ToolCallResult,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> ResultRecorded | JSONResponse:
    """Record one tool call's result on the Session that made the call.

    Ownership is checked before the log is read or written, and it has to be: the Event
    Log is keyed by Session and carries no tenant, so an answer written against somebody
    else's Session lands in their log with nothing raising. A Session this caller cannot
    see is refused with the code an absent one gets.

    The ask is looked up rather than trusted, so an id naming no open question is
    refused instead of leaving an answer nothing will ever match in the log -- which
    would expire minutes later as a cancelled tool call, far from the request that got
    it wrong.
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
    ask = ask_named(
        await whole_log(platform.event_log_range, session_id),
        call_id,
    )
    if ask is None or not ask.questions:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            "this session has no open tool call awaiting a result under that id",
            session_id=str(session_id),
            call_id=call_id,
        )
    if secret := tuple(question.id for question in ask.questions if question.is_secret):
        # Before anything is written, and before either arm of `success` is looked at.
        # A decline leaks nothing and is refused too, because accepting one would tell
        # the caller this is a question they may answer here -- and the next caller
        # would answer it with the value.
        return refuse(
            ErrorCode.ELICITATION_SECRET_REFUSED,
            "this call asks for a credential, and a credential is not answered here: "
            "an answer is kept in the session's event log for its retention period and "
            "in the rollout the pod ships out. Put the value in a vault and let the "
            "tool read it from there",
            session_id=str(session_id),
            call_id=call_id,
            questions=", ".join(secret),
        )
    answer = _answer_for(ask, body)
    if isinstance(answer, JSONResponse):
        return answer
    seq = await append_in_order(
        platform.event_log_append,
        session_id,
        tool_in_flight.TOOL_ELICITATION_ANSWERED,
        answer,
    )
    return ResultRecorded(seq=seq)
