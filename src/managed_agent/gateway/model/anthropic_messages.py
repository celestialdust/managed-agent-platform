"""The Anthropic Messages wire: one HTTP hop, translated in both directions.

This is the `WireHandler` for `UpstreamWire.ANTHROPIC_MESSAGES`, and it is the only
module on this wire that holds a client. The two translators beside it are pure
functions over bodies and frames, which is what keeps a model's wire something its
Routing Entry declares rather than something this service could detect: a module with no
socket has no endpoint it could ask what it is.

The header posture is the pass-through's inverted, and the inversion is the point. That
handler forwards a pod's body, so it forwards a named list of the pod's headers with it.
This one *rewrites* the body, so no header the pod sent describes what leaves -- and the
whole outbound block is this service's own. There is no allowlist here because there is
nothing to allow: a pod's `openai-beta` names a protocol this hop does not speak, and a
pod's `anthropic-version` would put the wire version that decides how every construct is
read under the control of the thing this service trusts least.

What is *not* inverted is the bytes. Nothing here decodes a header value, in either
direction, for the reason `passthrough.py` records at length: a header block carries no
charset and every guess at one has been a bug. The outbound values are built here as
octets and the relayed ones are the octets that arrived.

Failure lands in two different places depending on when it happens, and the split is
forced by HTTP rather than chosen. A construct that cannot be carried *outward* is found
before the upstream is reached, so it becomes a refusal with a status. A construct that
cannot be carried *back* is found after the response headers have already gone to the
Agent Runtime, so there is no status left to send: the failure is that no
`response.completed` is ever written, which is exactly how the runtime is told an answer
is not whole. The operator gets the cause and the construct in a log line either way.
"""

import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Final
from urllib.parse import urlencode

import httpx
from pydantic import ValidationError

from managed_agent.core.ids import SessionId
from managed_agent.gateway.model.anthropic_request import (
    ResponsesTurn,
    to_messages_request,
)
from managed_agent.gateway.model.anthropic_stream import MessagesStream
from managed_agent.gateway.model.anthropic_table import ANTHROPIC_VERSION, WIRE
from managed_agent.gateway.model.classify import Untranslatable, classify
from managed_agent.gateway.model.credential_broker import (
    ProviderCredentialBroker,
    UpstreamCredential,
)
from managed_agent.gateway.model.router import (
    GatewayRefusal,
    InboundTurn,
    RoutingEntry,
    UpstreamResponse,
)

_LOG = logging.getLogger(__name__)

_MESSAGES_PATH = "v1/messages"
"""The tail this wire appends to the configured prefix.

Foundry's Anthropic route is `{host}.services.ai.azure.com/anthropic/v1/messages`, so
the prefix an operator configures stops before `v1` and this supplies the rest: the same
arrangement as the pass-through's `responses`, and for the same reason -- the entry's
`base_url` is a prefix each wire completes, not a URL.
"""

_HOP_BY_HOP: Final = frozenset(
    {
        b"connection",
        b"content-length",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    }
)
"""Headers that describe one TCP connection. This hop is a different one.

Spelled as the wire spells them, because the filter compares against the octets that
arrived and decoding a name in order to compare it would put a guessed codec on the one
path that must not guess. `passthrough.py` holds the same list; it is RFC 9110's set of
connection-specific fields rather than a decision either module made, so the two copies
have nothing they could come to disagree about.
"""

_STREAM_HEADERS: Final[tuple[tuple[bytes, bytes], ...]] = (
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache"),
)
"""The response headers for a translated stream, authored here rather than relayed.

The upstream's headers describe the upstream's body, and the body handed back on this
path is not it -- these are re-framed events with different lengths and a different
content coding. Relaying a header that described the original would describe this one
wrongly, which is how a stream ends up truncated or declared as a coding it is not in.
A refusal is the other case: that body *is* relayed, so its headers are too.
"""


def upstream_url(entry: RoutingEntry) -> str:
    """`{base_url}/v1/messages`, with the entry's query parameters appended once.

    Percent-encoded here: this service is the client on this hop, so the encoding is its
    responsibility. Foundry's Anthropic route takes no `api-version` -- that is an
    Azure-OpenAI convention on the same host -- but the parameters are appended anyway
    because what an entry declares is the entry's business, not this function's.
    """
    url = f"{entry.base_url.rstrip('/')}/{_MESSAGES_PATH}"
    return f"{url}?{urlencode(entry.query_params)}" if entry.query_params else url


def _request_headers(credential: UpstreamCredential) -> dict[bytes, bytes]:
    """The whole outbound header block, built rather than filtered.

    Default deny taken to its conclusion: the inbound headers are not consulted at all,
    so there is no list to keep in step with the Agent Runtime and no name that could be
    forgotten. Every value here is an ASCII literal or the credential's own octets, so
    nothing is encoded from text this service did not author.

    `accept-encoding: identity` is pinned because the response body is parsed here. The
    read below decodes a content coding if one arrives anyway, so this is the belt and
    that is the braces -- and a coding negotiated on the Agent Runtime's behalf would be
    meaningless, since the bytes it receives are re-framed here rather than relayed.
    """
    name, value = credential.header()
    return {
        b"anthropic-version": ANTHROPIC_VERSION.encode("ascii"),
        b"content-type": b"application/json",
        b"accept": b"text/event-stream",
        b"accept-encoding": b"identity",
        name: value,
    }


def _cannot_carry(model: str, exc: Untranslatable) -> GatewayRefusal:
    """Turn an outbound translation failure into the refusal the caller should see.

    The construct and the row's reason cross; the wire's name does not. Which shape a
    model is served over is this service's configuration, and a caller that read it out
    of an error message has learned something it did not send -- so the refusal is
    built from the classification rather than from `exc.detail`, which is prefixed with
    the wire for the operator's benefit.

    400 rather than a 5xx: the request is well-formed and this platform cannot carry it
    to this model, so the honest answer names the feature that could not be carried
    instead of reporting a generic upstream error.
    """
    return GatewayRefusal(
        400,
        "invalid_request",
        f"model {model} cannot be served a request carrying "
        f"{exc.classification.construct}: {exc.classification.why}",
    )


def census_of_tools_handed_to_the_model(
    body: bytes, model: str, session_id: SessionId
) -> None:
    """Say which tools the model was actually handed, once per Turn.

    This is the far side of the Tool Gateway's offer census, and the two exist for one
    reason between them. That census says what the Gateway handed out; this says what
    survived the crossing. A Session whose Gateway logged `registered=1 offered=1` ran
    nineteen consecutive Turns in which the model behaved as though the tool did not
    exist, and no line in this platform could separate a runtime that dropped it from a
    tenant who never granted it. Two censuses, one on each side of the pod, turn that
    into a subtraction anyone can do from the logs.

    It reads the TRANSLATED body and must keep doing so. Reading the inbound one instead
    would report the runtime's own offer a second time, which is the number that was
    already right during the outage this line was written for: the tool arrived, and it
    was this side's tool table that dropped it. A census taken upstream of the step that
    loses things cannot see anything being lost.

    Absence is asymmetric here in the opposite direction to the Gateway's. There, a
    healthy listing is the quiet case; here the empty one is, because a model given no
    tool on a platform whose whole purpose is tool use has already lost the Turn. So
    zero speaks at WARNING and a populated list at INFO.

    The Session id is on the line because a gateway serves every Session at once, and a
    census nobody can attribute is one an operator has to believe rather than check.

    Names and never schemas. A tool name is the tenant's own vocabulary and the string
    an operator greps for; the schema beside it is written by the tenant and can carry
    anything they chose to put there.

    It observes and never refuses. A body that will not parse is not this function's
    error to raise -- it was built one call earlier by code that would have refused
    already, so the only way to reach here with one is a defect somewhere else, and
    failing the Turn out of a diagnostic would bury it.
    """
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return
    offered = parsed.get("tools") if isinstance(parsed, dict) else None
    names = sorted(
        str(one["name"])
        for one in (offered if isinstance(offered, list) else ())
        if isinstance(one, Mapping) and isinstance(one.get("name"), str)
    )
    if not names:
        _LOG.warning(
            "tool census: session=%s model=%s tools=0 -- this model was handed no "
            "tool, so a granted tool cannot be called on this Turn",
            session_id,
            model,
        )
        return
    _LOG.info(
        "tool census: session=%s model=%s tools=%d names=%s",
        session_id,
        model,
        len(names),
        names,
    )


_REFUSAL_DETAIL_LIMIT: Final = 400
"""How much of an upstream refusal message is logged.

Long enough for the field path that names what was wrong -- upstream refusals read like
`messages.1.content.0.tool_use_id: ...` and the path is the whole diagnostic. Capped
because the same field can quote the offending value back, and on this wire a value is
tenant text.
"""


async def _one_chunk(body: bytes) -> AsyncIterator[bytes]:
    """One already-read body, as the stream the relay expects."""
    yield body


def _log_the_refusal(
    body: bytes, status: int, entry: RoutingEntry, turn: InboundTurn
) -> None:
    """Say why the upstream refused a Turn, in this service's own log.

    A refusal costs the tenant the whole Turn, and until this line existed the only
    account of it was a body relayed into a pod that is deleted when the Session ends.
    An operator asking "why did that Turn fail" had a 400 in an access log and nothing
    else -- which is how a translation defect stayed a mystery through nineteen live
    runs while every service involved reported itself healthy.

    Truncated, and the type logged separately from the message, because the type is
    always the provider's own vocabulary while the message can quote the request back
    at us -- and on this wire the request carries the tenant's document.
    """
    detail = body.decode("utf-8", "replace")
    kind = ""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(error := parsed.get("error"), dict):
        kind = str(error.get("type", ""))
        detail = str(error.get("message", detail))
    _LOG.warning(
        "upstream refused: session=%s model=%s status=%d type=%s detail=%.400s",
        turn.caller.session_id,
        entry.model,
        status,
        kind or "unknown",
        detail[:_REFUSAL_DETAIL_LIMIT],
    )


def _messages_body(turn: InboundTurn, entry: RoutingEntry, max_tokens: int) -> bytes:
    """The Messages request body for one Turn, or a refusal naming what stopped it.

    Nothing partial is ever returned. A body missing a construct asks a different
    question than the one the tenant asked, so there is no half-translated request to
    send and no reason to open a connection for one.
    """
    try:
        parsed = ResponsesTurn.model_validate_json(turn.body)
    except ValidationError as exc:
        _LOG.warning("model %s was sent a body this wire cannot read", entry.model)
        raise GatewayRefusal(
            400,
            "invalid_request",
            f"model {turn.model} was sent a request this service cannot read",
        ) from exc
    try:
        request = to_messages_request(
            parsed,
            deployment=entry.model,
            max_tokens=max_tokens,
            session_id=turn.caller.session_id,
        )
    except Untranslatable as exc:
        _LOG.warning("model %s: %s", entry.model, exc.detail)
        raise _cannot_carry(turn.model, exc) from exc
    return json.dumps(request, separators=(",", ":")).encode()


def _relayed_headers(response: httpx.Response) -> tuple[tuple[bytes, bytes], ...]:
    """An upstream refusal's own headers, minus the ones about its own connection."""
    return tuple(
        (name, value)
        for name, value in response.headers.raw
        if name.lower() not in _HOP_BY_HOP
    )


async def _frames(chunks: AsyncIterator[bytes]) -> AsyncIterator[Mapping[str, object]]:
    """Read an SSE byte stream out as the JSON objects its `data:` lines carry.

    A frame is a unit of the protocol and a chunk is a unit of the transport, so the two
    do not line up: an event can arrive in two reads, and two events can arrive in one.
    The buffer is what makes that irrelevant.

    Every line terminator SSE admits is handled, because an intermediary between here
    and the model may normalise them and a stream that parsed only one would fail as
    "no terminator" for a reason that has nothing to do with the answer. A block with no
    `data:` line -- a comment, or an `event:` name on its own -- carries no frame and is
    skipped; a `data:` line that is not JSON fails the Turn, because a frame this wire
    cannot read is not a frame it may assume was harmless.
    """
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while (split := _split_event(buffer)) is not None:
            block, buffer = split
            if (frame := _frame(block)) is not None:
                yield frame
    # A well-behaved producer ends the last event with a blank line, but a stream that
    # closes right after its final `data:` line has still delivered it, and dropping it
    # here would report a complete answer as truncated.
    if (frame := _frame(buffer)) is not None:
        yield frame


_EVENT_ENDS: Final = (b"\r\n\r\n", b"\n\n", b"\r\r")
"""The blank line that ends an SSE event, in each of the three line endings SSE admits.

Ordered longest first so a CRLF-terminated block is cut at its true end: the shorter
patterns do not occur inside the longer one, but reading the list as written keeps
that a property of the list rather than of the search order.
"""


def _split_event(buffer: bytes) -> tuple[bytes, bytes] | None:
    """The first whole event in the buffer and what follows it, or None if none yet."""
    found: tuple[int, int] | None = None
    for separator in _EVENT_ENDS:
        at = buffer.find(separator)
        if at != -1 and (found is None or at < found[0]):
            found = (at, len(separator))
    if found is None:
        return None
    at, width = found
    return buffer[:at], buffer[at + width :]


def _frame(block: bytes) -> Mapping[str, object] | None:
    """One SSE block's payload as a JSON object, or None when it carries none.

    Multiple `data:` lines in one block are joined with a newline, which is what the SSE
    specification says they mean. Anthropic sends single-line payloads, so this is here
    for the intermediary that folds a long one rather than for the upstream.

    A payload this wire cannot read fails the Turn rather than being skipped. `classify`
    has no row for either construct and should not: the fail-safe default is the answer,
    because a frame nobody could parse is not a frame anybody may assume was harmless.
    """
    payloads = [
        line[len(b"data:") :].lstrip(b" ")
        for line in block.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
        if line.startswith(b"data:")
    ]
    if not payloads:
        return None
    try:
        parsed = json.loads(b"\n".join(payloads))
    except json.JSONDecodeError as exc:
        raise Untranslatable(
            WIRE, classify(WIRE, "stream.data_that_is_not_json")
        ) from exc
    if not isinstance(parsed, dict):
        raise Untranslatable(WIRE, classify(WIRE, "stream.data_that_is_not_an_object"))
    return parsed


def _sse(event: Mapping[str, object]) -> bytes:
    """One Responses event as the SSE bytes the Agent Runtime reads.

    Both lines are written even though only one is read. The runtime dispatches on the
    `type` inside `data:` and ignores the `event:` name, but a real Responses endpoint
    sends both -- and this hop exists so what arrives is indistinguishable from one.
    """
    payload = json.dumps(event, separators=(",", ":"))
    return f"event: {event.get('type', '')}\ndata: {payload}\n\n".encode()


class AnthropicMessagesHandler:
    """Serves `UpstreamWire.ANTHROPIC_MESSAGES`: body rewritten, stream rewritten back.

    `max_tokens` has no counterpart in the Agent Runtime's request and is required on
    every Messages request, so it is a constructor argument with no default. The cap
    applies to every Turn this handler serves, which makes it a decision somebody has to
    make out loud at the composition root rather than one this module can make quietly.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        broker: ProviderCredentialBroker,
        *,
        max_tokens: int,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive; the upstream requires a cap")
        self._client = client
        self._broker = broker
        self._max_tokens = max_tokens

    @asynccontextmanager
    async def open(
        self, turn: InboundTurn, entry: RoutingEntry
    ) -> AsyncIterator[UpstreamResponse]:
        """Open the upstream exchange and hold it open while the body is read.

        A context manager rather than a coroutine because the body is a live stream: the
        connection has to outlive this call and has to close even when the Agent Runtime
        hangs up halfway through reading it.

        The body is translated before the credential is fetched, which is the one place
        the order is load-bearing. A credential is read on a named Session's behalf and
        appears in the vault's own audit trail, so fetching one for a request that was
        never going to leave writes a false entry into somebody's log -- and costs a
        round trip for nothing.
        """
        body = _messages_body(turn, entry, self._max_tokens)
        census_of_tools_handed_to_the_model(body, entry.model, turn.caller.session_id)
        credential = await self._broker.for_turn(turn.caller.session_id, entry)
        request = self._client.build_request(
            "POST",
            upstream_url(entry),
            headers=_request_headers(credential),
            content=body,
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.RequestError as exc:
            _LOG.warning(
                "upstream %s unreachable for model %s", entry.base_url, entry.model
            )
            raise GatewayRefusal(
                502, "server_error", f"model {turn.model} could not be reached"
            ) from exc
        try:
            if response.status_code != 200:
                # Relayed whole, status and body untouched. The Anthropic error envelope
                # carries `error.message`, the field the Agent Runtime reads out of a
                # failed request -- so relaying it is what makes an upstream refusal
                # land as a failed Turn rather than an unparsed body, and rewriting it
                # here would make this service the author of a message it did not write.
                # Read whole first so the reason can be logged; a refusal body is an
                # error envelope, not a stream, and it is the only account anyone gets
                # of why a Turn died.
                refused = await response.aread()
                _log_the_refusal(refused, response.status_code, entry, turn)
                yield UpstreamResponse(
                    status=response.status_code,
                    headers=_relayed_headers(response),
                    body=_one_chunk(refused),
                )
            else:
                yield UpstreamResponse(
                    status=200,
                    headers=_STREAM_HEADERS,
                    body=self._translated(response, turn, entry),
                )
        finally:
            await response.aclose()

    async def _translated(
        self, response: httpx.Response, turn: InboundTurn, entry: RoutingEntry
    ) -> AsyncIterator[bytes]:
        """The upstream's frames as Responses events, ending where the answer ends.

        `aiter_bytes` not `aiter_raw`: the request pinned `accept-encoding: identity`,
        so there should be nothing to decode, but an upstream that compressed anyway
        would otherwise be parsed as SSE and read as noise.

        A failure here cannot become a status: the Agent Runtime already has the 200 and
        is reading -- so it ends the stream instead, and the *absence* of
        `response.completed` is the report. That is not a swallowed error: the runtime
        turns a stream that closes without the terminator into a failed request, and the
        cause and the construct go to the operator in the line below. It is caught
        rather than allowed to propagate so that the failure the runtime sees is a clean
        end of stream instead of an ASGI-level abort, which reads to an operator as a
        defect in this service rather than as a fact about the answer.

        Nothing writes a marker yet. `Untranslatable` carries the closed-set cause and
        the detail a marker takes; the component that appends one is not on this path.
        """
        stream = MessagesStream()
        try:
            async for event in stream.translate(_frames(response.aiter_bytes())):
                yield _sse(event)
        except Untranslatable as exc:
            # The Session is named because this is the only record the Turn's failure
            # leaves: no status was sent and no marker is written yet, so an operator
            # correlating a tenant's report to a cause has this line and nothing else.
            _LOG.error(
                "session %s on model %s over %s: turn failed as %s: %s",
                turn.caller.session_id,
                entry.model,
                entry.wire.value,
                exc.cause.value,
                exc.detail,
            )
