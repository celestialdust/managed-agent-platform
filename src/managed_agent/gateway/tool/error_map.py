"""Turning a registered server's failure into a code from the platform's closed set.

A tool that runs and reports a domain failure — no such invoice, the record is locked —
is answering. MCP carries that as a result with `is_error` set, the agent is meant to
read it and act, and nothing here touches it. A server that will not answer at all —
refused the connection, rejected the credential, broke the protocol, ran past the
deadline — is failing, and what leaves this module for that is a published code and
never the server's own words.

The scrubbing is not cosmetic. An upstream message can carry an internal hostname, a
stack frame, a row id, or the text of another tenant's data, and whatever comes back is
read by the model. A registered server that raises out of its own handler is the sharp
case: the SDK forwards that exception's `str()` verbatim as the JSON-RPC error message,
so the words in `MCPError.message` are frequently the upstream's own traceback text. The
outgoing text is therefore assembled here out of a fixed sentence, the code, the name
the tenant registered the tool under, and a correlation id; the real exception goes to
this service's own log under that id, which is where an operator looks (ADR-013).

The exception class is `MCPError`, it takes its three fields directly rather than an
`ErrorData`, and the transport under a Streamable HTTP registration is `httpx2` — a
separate distribution from `httpx`, sharing no base class with it, so an arm matching
`httpx.HTTPError` here would never fire.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Final, assert_never
from uuid import uuid4

import httpx2
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CONNECTION_CLOSED,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    REQUEST_TIMEOUT,
    CallToolResult,
    TextContent,
)

from managed_agent.core.errors import ErrorCode, ErrorEnvelope
from managed_agent.core.ports import SessionNotVisible
from managed_agent.core.vfs.evidence import EvidenceStorageUnconfigured
from managed_agent.gateway.tool.credential_broker import CredentialUnavailable
from managed_agent.gateway.tool.scope_clamp import ScopeRefusal

_log = logging.getLogger(__name__)

_JSON_RPC_TO_CODE: Final[Mapping[int, ErrorCode]] = {
    REQUEST_TIMEOUT: ErrorCode.TOOL_TIMED_OUT,
    CONNECTION_CLOSED: ErrorCode.TOOL_UNAVAILABLE,
    INVALID_PARAMS: ErrorCode.REQUEST_INVALID,
    METHOD_NOT_FOUND: ErrorCode.TOOL_UNAVAILABLE,
}
"""How the SDK's own JSON-RPC codes land in the published set.

`REQUEST_TIMEOUT` is -32001 and `CONNECTION_CLOSED` is -32000: the SDK's own numbers,
imported rather than written, because a per-request read deadline and a dropped
connection are the two failures a tenant most needs told apart and the wrong constant
merges them. Anything unlisted defaults to `TOOL_UNAVAILABLE` rather than to `INTERNAL`,
which matters most for -32603: an upstream server reporting its own internal error is
the registering team's problem, and recording it as ours sends them to the wrong people.
"""


def _leaves(exc: BaseException) -> tuple[BaseException, ...]:
    """Every non-group exception inside `exc`, depth first.

    Both MCP transports run on anyio task groups, so a failure raised while a
    connection is being opened does not arrive bare — it arrives inside an
    `ExceptionGroup`, and one raised under a client session inside two nested ones. A
    `match` against the group itself falls through every arm below it, which is how a
    refused connection would otherwise be recorded as a fault of this platform rather
    than of the server.
    """
    if isinstance(exc, BaseExceptionGroup):
        return tuple(leaf for sub in exc.exceptions for leaf in _leaves(sub))
    return (exc,)


def _classify_one(exc: BaseException) -> ErrorCode:
    """Which published code one already-unwrapped exception becomes.

    `TimeoutError` covers `asyncio.TimeoutError` too; they have been the same class
    since Python 3.11, so a second arm for it would be unreachable. `OSError` is not
    dead weight next to the `httpx2` arms: a stdio registration naming a command that
    is not on PATH fails as `FileNotFoundError` before any protocol runs.

    `SessionNotVisible` is the one arm here that is not about an upstream at all. It
    reaches this module because a tool call reads the Session's Scope before it opens
    anything, and that read is deliberately inside the same `try` as the call -- so a
    Session the store will not hand back arrives beside the transport failures without
    being one. Without an arm it falls to `INTERNAL`, which is reserved for a fault of
    this platform's own; a Session swept at the end of its retention window is a
    routine tenant-side condition, and calling it ours would send an operator looking
    for a malfunction that is not there.
    """
    match exc:
        case SessionNotVisible():
            return ErrorCode.SESSION_NOT_FOUND
        case CredentialUnavailable():
            return ErrorCode.TOOL_UNAVAILABLE
        case MCPError():
            return _JSON_RPC_TO_CODE.get(exc.code, ErrorCode.TOOL_UNAVAILABLE)
        case TimeoutError() | httpx2.TimeoutException():
            return ErrorCode.TOOL_TIMED_OUT
        case httpx2.HTTPError() | OSError():
            return ErrorCode.TOOL_UNAVAILABLE
        case _:
            return ErrorCode.INTERNAL


def classify(exc: BaseException) -> ErrorCode:
    """Which published code one failure on the call path becomes.

    Almost always an upstream's failure, and not always: the Session's Scope is read on
    that path too, so `SessionNotVisible` is classified here as well -- see the arm in
    `_classify_one`.

    `INTERNAL` is reserved for a fault of this platform's own and is deliberately not
    the bucket everything unrecognized falls into: a server nobody can reach is the
    registering team's problem, and calling it ours sends them to the wrong people. A
    rejected credential is `TOOL_UNAVAILABLE` and pointedly not `TOOL_NOT_GRANTED` — a
    Grant is a platform decision, and a secret the server refused says nothing about
    one.

    A group carrying several failures is reported as the first one this module
    recognizes rather than as the first one raised, because anyio orders a group by when
    each task failed and the earliest failure is routinely a stream teardown provoked by
    the cause rather than the cause.
    """
    recognized = [
        code
        for code in map(_classify_one, _leaves(exc))
        if code is not ErrorCode.INTERNAL
    ]
    return recognized[0] if recognized else ErrorCode.INTERNAL


def refusal(code: ErrorCode, subject: str, correlation_id: str) -> ErrorEnvelope:
    """The only text that may leave this service about a failure it saw.

    `subject` is the tenant's own name for the thing that failed — a registered tool
    name or a resource URI it asked for. Both are names the caller supplied, so echoing
    one discloses nothing it did not already know.
    """
    return ErrorEnvelope(
        code=code,
        message=(
            f"the registered server behind {subject} did not complete the call; "
            f"the platform recorded the cause under {correlation_id}"
        ),
        detail={"subject": subject, "correlation_id": correlation_id},
    )


def _scope_mismatch_said(tool_name: str, dimension: str, cause: ScopeRefusal) -> str:
    """The sentence for one direction of a Scope-versus-bindings mismatch.

    Two sentences because the tenant's next move differs and only one of them is true
    of any given call. Told "this Session's Scope does not carry one" about a dimension
    their Scope plainly carries, a reader goes looking at the half of the pair that is
    already correct -- so a single sentence covering both would be actively misleading
    rather than merely vague.

    Matched exhaustively, so a third cause added to `ScopeRefusal` fails to type-check
    here rather than falling through to whichever sentence was written last.
    """
    match cause:
        case ScopeRefusal.DIMENSION_NOT_IN_SCOPE:
            return (
                f"the call to {tool_name} was not made: it is registered to be "
                f"narrowed by the Scope dimension {dimension}, and this Session's "
                f"Scope does not carry one"
            )
        case ScopeRefusal.DIMENSION_NOT_BOUND:
            return (
                f"the call to {tool_name} was not made: this Session's Scope narrows "
                f"{dimension}, and {tool_name} declares no Scope Binding that could "
                f"hold the call to it"
            )
        case _ as unreachable:
            assert_never(unreachable)


def out_of_scope(tool_name: str, dimension: str, cause: ScopeRefusal) -> ErrorEnvelope:
    """The refusal for a call that could not be held to this Session's Scope.

    Its own function rather than `refusal` above, because that one's text says the
    registered server did not complete the call and here no server was reached at all
    -- nothing was opened, no credential was read, and a message saying otherwise would
    send whoever reads it looking at an upstream that never saw the request.

    One code and one shape for both directions of the mismatch, and two sentences: the
    caller branches on `tool.out_of_scope` either way, and reads which half to fix.

    Names the tool and the dimension and no value. Both names came from the tenant --
    one out of their registration, one out of the Scope on their own create call -- so
    echoing them discloses nothing; a Scope *value* in here would let a model read one
    refusal and learn what the Session is bounded to.

    No correlation id, unlike every other envelope this module builds, and the
    omission is deliberate rather than an oversight. The others carry one because they
    stand in for a cause that was logged and withheld; this refusal withholds nothing.
    Which dimension, and which way it failed to meet the bindings, are both already in
    the message -- an id here would point at a log line holding strictly less than the
    reader is already looking at.

    The log line carries the cause as well, because the two want different fixes from
    different people: one is a Scope the Session was created with, the other a binding
    the registration never declared.
    """
    _log.warning(
        "tool call refused as out of scope tool=%s dimension=%s cause=%s",
        tool_name,
        dimension,
        cause.value,
    )
    return ErrorEnvelope(
        code=ErrorCode.TOOL_OUT_OF_SCOPE,
        message=_scope_mismatch_said(tool_name, dimension, cause),
        detail={"subject": tool_name, "dimension": dimension},
    )


def argument_not_offered(tool_name: str, argument: str) -> ErrorEnvelope:
    """The refusal for a call that supplied an argument the Session's Scope fills.

    Its own function rather than `out_of_scope` above, whose sentence says this
    Session's Scope does not carry the dimension -- here it carries it, and the call
    was refused for the opposite reason. A caller told the wrong one of those goes
    looking at the Session it was given instead of at the argument it sent.

    Its own sentence, but the same code, and that is deliberate. `REQUEST_INVALID`
    would read as "this argument was malformed", which invites the caller to try
    another value -- and trying another value is the probing loop the whole decision
    closes. `TOOL_OUT_OF_SCOPE` says the platform holds this field, which is both true
    and unrewarding to retry against.

    Names the tool and the argument and neither value. The argument's name came from
    the caller's own call, and it is the one thing it needs in order to stop sending
    it. The value it attempted and the value the Scope holds are the two strings that
    would let a model read the Session's bounds back out of a refusal, so neither is
    here (ADR-034).

    Nothing is logged here, unlike its two neighbours. The attempt is already recorded
    by `scope_clamp.narrow` at the point it was detected, at WARNING and carrying the
    attempted value this envelope may not repeat -- a second line here would be a
    strictly poorer copy of one an operator already has.
    """
    return ErrorEnvelope(
        code=ErrorCode.TOOL_OUT_OF_SCOPE,
        message=(
            f"the call to {tool_name} was not made: {argument} is written from this "
            f"Session's Scope and is not offered in this tool's schema"
        ),
        detail={"subject": tool_name, "argument": argument},
    )


def session_unreadable(tool_name: str) -> ErrorEnvelope:
    """The refusal for a call whose Session could not be read at all.

    Its own function rather than `refusal`, for the reason `out_of_scope` above gives:
    that sentence says the registered server did not complete the call, and here the
    Scope read that failed happens before anything is opened -- no connection, no
    credential, no request. A message blaming the server would send whoever reads it to
    an upstream that never heard of this call.

    **It says the Session could not be read and never which way.** The store answers a
    Session that is gone and a Session belonging to somebody else with one exception on
    purpose, so that a caller holding an id cannot learn from a refusal whether the id
    names another tenant's Session. Two sentences here would rebuild that oracle one
    layer out, so there is one -- and it carries no Session id either, since the only
    reader of this envelope is an agent inside the Session the id would name.

    No correlation id, like `out_of_scope` and unlike `refusal`: the withheld cause is
    an exception whose whole content is the id of a Session the caller is already
    running inside, so an id pointing at a log line holding that would be pointing at
    nothing the reader wants.
    """
    _log.warning("tool call refused, session not readable tool=%s", tool_name)
    return ErrorEnvelope(
        code=ErrorCode.SESSION_NOT_FOUND,
        message=(
            f"the call to {tool_name} was not made: this Session could not be read, "
            f"so nothing could narrow the call to its Scope"
        ),
        detail={"subject": tool_name},
    )


def evidence_unrecordable(tool_name: str, correlation_id: str) -> ErrorEnvelope:
    """The refusal for a call this platform could not record Evidence for.

    Its own function rather than `refusal`, for the reason `session_unreadable` above
    gives, and here the sentence is false in both halves: the registered server did
    complete the call, and what failed afterwards was this platform's own storage. A
    tenant handed "the registered server did not complete the call" goes and reads the
    logs of a service that did exactly what it was asked.

    **The call is still refused, and the message says so.** Evidence is not a side
    record that can be skipped when it is inconvenient -- a large tool result handed to
    the model with nothing recording it is the one outcome the capture design exists to
    make unreachable, so a result that cannot be recorded is not returned. The tenant's
    move is to tell whoever runs this deployment, which is why the text points at the
    platform rather than at anything they can change.

    A correlation id, unlike `session_unreadable`: the withheld cause names an
    environment variable and a deployment, which is exactly what an operator reading the
    log needs and exactly what must not go on the wire.
    """
    return ErrorEnvelope(
        code=ErrorCode.INTERNAL,
        message=(
            f"the call to {tool_name} completed but this platform could not record its "
            f"Evidence, so the result was not returned; the cause is recorded under "
            f"{correlation_id}"
        ),
        detail={"subject": tool_name, "correlation_id": correlation_id},
    )


def _cannot_record_evidence(exc: BaseException) -> bool:
    """Whether this failure is the Evidence store refusing for want of configuration.

    Over the unwrapped leaves rather than the exception itself, because capture runs
    inside the same task group as the call and arrives grouped with whatever else failed
    while it was unwinding -- the same reason `classify` walks them.
    """
    return any(isinstance(leaf, EvidenceStorageUnconfigured) for leaf in _leaves(exc))


def record(exc: BaseException, subject: str) -> ErrorEnvelope:
    """Log the real failure under a fresh correlation id, return only the envelope.

    One code leaves by a different door. `SESSION_NOT_FOUND` is not an upstream
    failure -- it is the Session's own row failing to come back, which happens before
    any server is reached -- so it takes `session_unreadable`'s envelope, and it takes
    that function's WARNING rather than the ERROR below. An ERROR with a traceback is
    what an operator is paged by, and a Session that has been swept is not something
    for anyone to fix.
    """
    code = classify(exc)
    if code is ErrorCode.SESSION_NOT_FOUND:
        return session_unreadable(subject)
    correlation_id = uuid4().hex
    if _cannot_record_evidence(exc):
        # Told apart here rather than by its code, because `INTERNAL` is honestly what
        # a caller should see -- it IS this platform's fault -- and the two things that
        # must change are the sentence and the log line, neither of which the code
        # decides. An ERROR with the traceback either way: an operator should be paged
        # for a deployment that cannot store Evidence, and the traceback is what tells
        # them which variable is unset.
        _log.error(
            "evidence could not be recorded, so the call was refused "
            "subject=%s correlation_id=%s",
            subject,
            correlation_id,
            exc_info=exc,
        )
        return evidence_unrecordable(subject, correlation_id)
    _log.error(
        "upstream MCP failure subject=%s code=%s correlation_id=%s",
        subject,
        code.value,
        correlation_id,
        exc_info=exc,
    )
    return refusal(code, subject, correlation_id)


def as_tool_result(envelope: ErrorEnvelope) -> CallToolResult:
    """A refusal in the shape of a failed tool call, which is the only shape it takes.

    Never a JSON-RPC error. The Agent Runtime treats a protocol fault and a tool that
    failed differently, and a platform refusal arriving as the former reads to the agent
    as the Tool Gateway being broken rather than as this call not being available to it
    (ADR-014).
    """
    return CallToolResult(
        is_error=True,
        content=[TextContent(type="text", text=envelope.model_dump_json())],
    )


def as_mcp_error(envelope: ErrorEnvelope) -> MCPError:
    """The same refusal on a method whose result shape cannot carry one: a read.

    A resource read has no result shape that can carry `is_error`, so this is the one
    path where a refusal leaves as a JSON-RPC error. The envelope rides in `data` rather
    than in `message`, so a client that renders only the message still shows a fixed
    sentence and a client that reads `data` gets the code it can branch on.
    """
    return MCPError(
        code=INTERNAL_ERROR,
        message="the platform refused this read",
        data=envelope.model_dump(mode="json"),
    )


def as_listing_error(envelope: ErrorEnvelope) -> MCPError:
    """The refusal for a listing that reached no server at all.

    A listing is normally allowed to come back short: one unreachable server must not
    empty the others' lists, so each server's failure is recorded and skipped. That
    tolerance has one blind spot, which this exists for. When *every* server fails the
    result is an empty list, and an empty list is exactly what a tenant with nothing
    registered gets -- so the caller cannot tell "you have no tools" from "nothing is
    reachable". Measured on the live cluster: `tools/list` answered
    `200 {"tools": []}` while its one registered server was failing a credential fetch.

    Separate from `as_mcp_error` only because that one's fixed sentence names a read,
    and a caller shown "refused this read" in response to a listing is being told
    something untrue about which call failed.
    """
    return MCPError(
        code=INTERNAL_ERROR,
        message="the platform reached none of this Session's servers",
        data=envelope.model_dump(mode="json"),
    )
