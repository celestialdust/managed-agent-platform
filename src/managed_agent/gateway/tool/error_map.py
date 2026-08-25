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
from typing import Final
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
from managed_agent.gateway.tool.credential_broker import CredentialUnavailable

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
    """
    match exc:
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
    """Which published code one upstream failure becomes.

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


def out_of_scope(tool_name: str, dimension: str) -> ErrorEnvelope:
    """The refusal for a call this Session's Scope could not narrow.

    Its own function rather than `refusal` above, because that one's text says the
    registered server did not complete the call and here no server was reached at all
    -- nothing was opened, no credential was read, and a message saying otherwise would
    send whoever reads it looking at an upstream that never saw the request.

    Names the tool and the dimension and no value. Both names came from the tenant's
    own registration, so echoing them discloses nothing; a Scope *value* in here would
    let a model read one refusal and learn what the Session is bounded to.

    No correlation id, unlike every other envelope this module builds, and the
    omission is deliberate rather than an oversight. The others carry one because they
    stand in for a cause that was logged and withheld; this refusal withholds nothing.
    The registration binds this dimension, this Session's Scope does not carry it, and
    both halves are already in the message -- an id here would point at a log line
    holding strictly less than the reader is already looking at.
    """
    _log.warning(
        "tool call refused as out of scope tool=%s dimension=%s", tool_name, dimension
    )
    return ErrorEnvelope(
        code=ErrorCode.TOOL_OUT_OF_SCOPE,
        message=(
            f"the call to {tool_name} was not made: it is registered to be narrowed "
            f"by the Scope dimension {dimension}, and this Session's Scope does not "
            f"carry one"
        ),
        detail={"subject": tool_name, "dimension": dimension},
    )


def record(exc: BaseException, subject: str) -> ErrorEnvelope:
    """Log the real failure under a fresh correlation id, return only the envelope."""
    code = classify(exc)
    correlation_id = uuid4().hex
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
