"""The Anthropic wire's HTTP hop, graded on the request built and the bytes handed back.

Tier 1, no infrastructure. `httpx.MockTransport` stands in for Foundry and records the
request the real `httpx.AsyncClient` built, so what is asserted below is the actual
outbound wire and not a dict this test assembled. The broker is the real class over a
fake vault, because the credential swap is one of the properties under test and a
stubbed credential would prove the swap against itself.

The negative assertions are the load-bearing ones, and they are the mirror image of the
pass-through's. That handler has an allowlist of pod headers because it forwards a pod's
body; this one rewrites the body outright, so *no* pod header describes what leaves and
the whole outbound block is this service's own. A test that only checked the headers we
add would pass on the day a pod-supplied `x-api-key` started riding along beside them.

The last test drives the real ASGI app with both wires registered. It is the one place
here that grades dispatch rather than translation: the shape a model is served over
comes from its Routing Entry, and a test that exercised one wire alone could not tell a
correct dispatch from a build that had only one handler.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from uuid import UUID

import httpx
import pytest

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.session.session_token import mint_session_token
from managed_agent.gateway.model.anthropic_messages import (
    AnthropicMessagesHandler,
    census_of_tools_handed_to_the_model,
    upstream_url,
)
from managed_agent.gateway.model.anthropic_table import ANTHROPIC_VERSION
from managed_agent.gateway.model.credential_broker import ProviderCredentialBroker
from managed_agent.gateway.model.passthrough import ResponsesPassthrough
from managed_agent.gateway.model.router import (
    AuthScheme,
    CallerSession,
    GatewayRefusal,
    InboundTurn,
    ModelGateway,
    RoutingEntry,
    RoutingTable,
    SessionTokenVerifier,
    UpstreamWire,
    WireHandler,
    create_model_gateway_app,
)

_KEY = b"the-signing-key-both-gateways-hold"
_NOW_MS = 1_700_000_000_000
_SESSION = SessionId(UUID("11111111-1111-4111-8111-111111111111"))
_TENANT = TenantId(UUID("22222222-2222-4222-8222-222222222222"))
_CALLER = CallerSession(session_id=_SESSION, tenant_id=_TENANT)

_MESSAGES_MODEL = "gsds-claude-opus-4-6"
_RESPONSES_MODEL = "gpt-5-codex"
_FOUNDRY = "https://map-foundry.services.ai.azure.com/anthropic"

_POD_SUPPLIED_API_KEY = "POD-SUPPLIED-PROVIDER-KEY"
"""A provider key a Session pod attached itself, under the header this module presents
the *platform's* credential in. The pod is not trusted, so this is what a hostile or
merely misconfigured runtime sends, and it is the value that must not appear on the far
side of this hop."""

_POD_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"host", b"model-gateway.map.svc"),
    (b"authorization", b"Bearer the-pods-own-token"),
    (b"content-type", b"application/json"),
    (b"content-length", b"999999"),
    (b"accept", b"text/event-stream"),
    (b"accept-encoding", b"gzip"),
    (b"user-agent", b"codex_cli_rs/0.1.0"),
    (b"openai-beta", b"responses=experimental"),
    (b"x-codex-turn-state", b"state-1"),
    (b"anthropic-version", b"1999-01-01"),
    (b"anthropic-beta", b"pod-chosen-beta"),
    (b"x-api-key", _POD_SUPPLIED_API_KEY.encode()),
    (b"cookie", b"session=pods-own"),
    (b"x-header-nobody-named", b"arbitrary"),
)
"""What a pod can put on the inbound request. None of it describes the body that leaves.

`anthropic-version` and `anthropic-beta` are in here deliberately. They are the two a
pod could send that would look plausible on this hop, and letting either through would
put the wire version -- which decides how every construct in this wire's table is
read -- under the control of the thing this service does not trust.
"""


class _OneSecretVault:
    def __init__(self) -> None:
        self.reads: list[str] = []

    async def fetch(self, name: str) -> str:
        self.reads.append(name)
        return f"secret-for-{name}"


class _FrozenClock:
    def now_epoch_ms(self) -> int:
        return _NOW_MS


class _Chunks(httpx.AsyncByteStream):
    """An upstream body delivered in the pieces a socket would deliver it in.

    A stream rather than `content=bytes`, for two reasons. `content=bytes` hands back a
    response httpx already marks as consumed, so iterating it raises for a reason that
    exists only inside a test. And closure is the property this stands in for: a relayed
    upstream connection has to be released when the caller stops reading, and with a
    mock transport `closed` here is the only evidence that the handler's `finally` ran.
    """

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = tuple(chunks)
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _entry(
    *,
    base_url: str = _FOUNDRY,
    auth_scheme: AuthScheme = AuthScheme.API_KEY,
    query_params: tuple[tuple[str, str], ...] = (),
) -> RoutingEntry:
    return RoutingEntry(
        model=_MESSAGES_MODEL,
        wire=UpstreamWire.ANTHROPIC_MESSAGES,
        base_url=base_url,
        auth_scheme=auth_scheme,
        credential_name="map/dev/providers/anthropic",
        query_params=query_params,
    )


def _responses_body(**extra: object) -> bytes:
    return json.dumps(
        {
            "model": "claude-opus-5",
            "instructions": "be brief",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            **extra,
        }
    ).encode()


def _turn(
    *,
    body: bytes | None = None,
    headers: tuple[tuple[bytes, bytes], ...] = _POD_HEADERS,
) -> InboundTurn:
    return InboundTurn(
        caller=_CALLER,
        model="claude-opus-5",
        headers=headers,
        body=body if body is not None else _responses_body(),
    )


_TEXT_STREAM: tuple[Mapping[str, object], ...] = (
    {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 9}}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"input_tokens": 9, "output_tokens": 2},
    },
    {"type": "message_stop"},
)


def _sse(
    frames: Sequence[Mapping[str, object]], *, terminator: bytes = b"\n\n"
) -> bytes:
    return b"".join(
        b"event: "
        + str(frame["type"]).encode()
        + terminator[: len(terminator) // 2]
        + b"data: "
        + json.dumps(frame).encode()
        + terminator
        for frame in frames
    )


def _events(payload: bytes) -> list[dict[str, object]]:
    """Every event out of an SSE payload, read the way the Agent Runtime reads one."""
    out: list[dict[str, object]] = []
    for line in payload.split(b"\n"):
        if line.startswith(b"data:"):
            out.append(json.loads(line[len(b"data:") :].strip()))
    return out


def _handler(
    responder: httpx.MockTransport,
    *,
    vault: _OneSecretVault | None = None,
    max_tokens: int = 4096,
) -> AnthropicMessagesHandler:
    client = httpx.AsyncClient(transport=responder)
    broker = ProviderCredentialBroker(vault or _OneSecretVault(), _FrozenClock())
    return AnthropicMessagesHandler(client, broker, max_tokens=max_tokens)


def _serving(
    chunks: Sequence[bytes],
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
) -> tuple[httpx.MockTransport, list[httpx.Request], _Chunks]:
    """A transport answering with one canned response, recording what it was asked."""
    seen: list[httpx.Request] = []
    stream = _Chunks(chunks)

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            status,
            headers=dict(headers or {"content-type": "text/event-stream"}),
            stream=stream,
        )

    return httpx.MockTransport(respond), seen, stream


async def _drain(
    handler: AnthropicMessagesHandler, turn: InboundTurn, entry: RoutingEntry
) -> tuple[int, tuple[tuple[bytes, bytes], ...], bytes]:
    async with handler.open(turn, entry) as upstream:
        body = b"".join([chunk async for chunk in upstream.body])
        return upstream.status, upstream.headers, body


def test_the_path_is_appended_to_the_configured_prefix() -> None:
    assert (
        upstream_url(_entry())
        == "https://map-foundry.services.ai.azure.com/anthropic/v1/messages"
    )


def test_a_trailing_slash_on_the_prefix_does_not_double_the_separator() -> None:
    assert upstream_url(_entry(base_url=_FOUNDRY + "/")) == upstream_url(_entry())


def test_the_entrys_query_parameters_are_appended_once_and_encoded() -> None:
    url = upstream_url(_entry(query_params=(("api-version", "2024-10-01 preview"),)))

    assert url.endswith("/v1/messages?api-version=2024-10-01+preview")


def test_a_cap_that_is_not_a_cap_is_refused_at_construction() -> None:
    responder, _, _ = _serving([b""])
    for cap in (0, -1):
        with pytest.raises(ValueError, match="max_tokens"):
            _handler(responder, max_tokens=cap)


async def test_the_outbound_request_is_the_translated_body_at_the_messages_path() -> (
    None
):
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])

    await _drain(_handler(responder, max_tokens=777), _turn(), _entry())

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == upstream_url(_entry())
    sent = json.loads(seen[0].content)
    assert sent["model"] == _MESSAGES_MODEL
    assert sent["max_tokens"] == 777
    assert sent["stream"] is True
    assert sent["messages"][0]["role"] == "user"


async def test_the_census_counts_the_translated_tools_and_not_the_offered_ones(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Which side of the translation the census stands on is the whole point of it.

    A `tool_search` spec carries no `name` field, so an inbound-side census cannot see
    it at all -- and every other spec on that turn is one the runtime offered, which is
    the number that was already correct throughout the outage this census was written
    for. The tool arrived; this side's table dropped it. A census taken upstream of the
    step that loses things reports a healthy total while the model gets nothing.

    So this drives the real path and reads the name out of the log. The name can only be
    there if the line was taken after `_messages_body` ran.
    """
    responder, _, _ = _serving([_sse(_TEXT_STREAM)])
    body = _responses_body(
        tools=[{"type": "tool_search", "execution": "client", "parameters": {}}]
    )
    caplog.set_level(logging.INFO, logger="managed_agent.gateway.model")

    await _drain(_handler(responder), _turn(body=body), _entry())

    census = [
        one.getMessage()
        for one in caplog.records
        if one.name.endswith("anthropic_messages")
    ]
    assert len(census) == 1, census
    assert "tool_search" in census[0]
    assert str(_CALLER.session_id) in census[0]


async def test_the_deployment_name_comes_from_configuration_not_the_request() -> None:
    """A Foundry deployment's name need not equal the model id the tenant names, and
    which deployment a model reaches is its Routing Entry's answer."""
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])

    await _drain(_handler(responder), _turn(), _entry())

    assert json.loads(seen[0].content)["model"] == _MESSAGES_MODEL
    assert json.loads(seen[0].content)["model"] != "claude-opus-5"


async def test_the_wire_version_is_pinned_by_this_service() -> None:
    """The header is required and the Agent Runtime cannot supply it -- it speaks
    Responses and has no such field -- so a missing one is a 400 from the upstream."""
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])

    await _drain(_handler(responder), _turn(), _entry())

    assert seen[0].headers["anthropic-version"] == ANTHROPIC_VERSION
    assert ANTHROPIC_VERSION == "2023-06-01"


async def test_the_platforms_credential_is_the_only_key_that_leaves() -> None:
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])

    await _drain(_handler(responder), _turn(), _entry())

    assert seen[0].headers["x-api-key"] == "secret-for-map/dev/providers/anthropic"
    assert _POD_SUPPLIED_API_KEY not in str(dict(seen[0].headers))


async def test_the_credential_goes_out_in_the_scheme_its_entry_names() -> None:
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])

    await _drain(_handler(responder), _turn(), _entry(auth_scheme=AuthScheme.BEARER))

    assert seen[0].headers["authorization"] == (
        "Bearer secret-for-map/dev/providers/anthropic"
    )
    assert "x-api-key" not in seen[0].headers


async def test_no_header_the_pod_chose_crosses_this_hop() -> None:
    """The body that leaves is this service's own, so nothing the pod sent describes it.
    An allowlist would be a decision about which of a pod's Responses headers to put on
    an Anthropic request, and there is no such header."""
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])

    await _drain(_handler(responder), _turn(), _entry())
    arrived = {name.lower() for name in seen[0].headers}

    assert arrived <= {
        "anthropic-version",
        "content-type",
        "accept",
        "accept-encoding",
        "x-api-key",
        # The HTTP client's own, from the URL and the body it was handed. `user-agent`
        # and `connection` are named here rather than treated as ours because the client
        # supplies them for every request it builds; the assertion below is what says
        # the pod's value for one of them did not survive.
        "host",
        "content-length",
        "user-agent",
        "connection",
    }
    assert seen[0].headers["user-agent"] != "codex_cli_rs/0.1.0"
    assert seen[0].headers["accept-encoding"] == "identity"
    assert "anthropic-beta" not in arrived
    assert "cookie" not in arrived
    assert "openai-beta" not in arrived
    assert "x-codex-turn-state" not in arrived
    assert "x-header-nobody-named" not in arrived


async def test_the_pods_own_wire_version_does_not_override_the_pinned_one() -> None:
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])

    await _drain(_handler(responder), _turn(), _entry())

    assert seen[0].headers["anthropic-version"] == ANTHROPIC_VERSION


async def test_the_body_comes_back_as_events_the_runtime_dispatches_on() -> None:
    responder, _, _ = _serving([_sse(_TEXT_STREAM)])

    status, headers, body = await _drain(_handler(responder), _turn(), _entry())

    assert status == 200
    assert dict(headers)[b"content-type"].startswith(b"text/event-stream")
    assert [str(event["type"]) for event in _events(body)] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ]


async def test_the_terminator_reports_the_turn_as_finished_with_its_usage() -> None:
    responder, _, _ = _serving([_sse(_TEXT_STREAM)])

    _, _, body = await _drain(_handler(responder), _turn(), _entry())
    completed = _events(body)[-1]["response"]

    assert isinstance(completed, dict)
    assert completed["end_turn"] is True
    assert completed["usage"]["total_tokens"] == 11


async def test_an_event_split_across_two_chunks_is_still_one_event() -> None:
    """A frame is a unit of the protocol and not of the transport, so an event arriving
    in two reads must not be read as two events or as none."""
    whole = _sse(_TEXT_STREAM)
    cut = len(whole) // 2
    responder, _, _ = _serving([whole[:cut], whole[cut:]])

    _, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert len(_events(body)) == 5


async def test_events_delivered_one_byte_at_a_time_are_still_read_whole() -> None:
    whole = _sse(_TEXT_STREAM)
    responder, _, _ = _serving([whole[i : i + 1] for i in range(len(whole))])

    _, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert len(_events(body)) == 5


async def test_a_crlf_terminated_stream_is_read_the_same_way() -> None:
    """An intermediary may normalise line endings, and the two terminators are one
    protocol."""
    responder, _, _ = _serving([_sse(_TEXT_STREAM, terminator=b"\r\n\r\n")])

    _, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert len(_events(body)) == 5


async def test_comments_and_nameless_blocks_carry_no_frame() -> None:
    payload = b": keep alive\n\nevent: ping\n\n" + _sse(_TEXT_STREAM)
    responder, _, _ = _serving([payload])

    _, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert len(_events(body)) == 5


async def test_a_last_event_with_no_blank_line_after_it_is_not_lost() -> None:
    payload = _sse(_TEXT_STREAM).removesuffix(b"\n\n")
    responder, _, _ = _serving([payload])

    _, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert len(_events(body)) == 5


async def test_a_data_line_that_is_not_json_fails_the_turn_rather_than_being_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responder, _, _ = _serving([b"event: message_start\ndata: {not json\n\n"])

    with caplog.at_level(logging.ERROR):
        _, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert _events(body) == []
    assert "upstream_unclassified" in caplog.text


async def test_a_stream_that_fails_mid_answer_yields_no_terminator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MAP-A77 at the byte level: the Turn fails, and the absence of
    `response.completed` is how the runtime is told the answer is not whole."""
    truncated = (
        _TEXT_STREAM[0],
        _TEXT_STREAM[1],
        _TEXT_STREAM[2],
        _TEXT_STREAM[3],
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 4000},
        },
        {"type": "message_stop"},
    )
    responder, _, _ = _serving([_sse(truncated)])

    with caplog.at_level(logging.ERROR):
        status, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert status == 200
    kinds = [str(event["type"]) for event in _events(body)]
    assert "response.completed" not in kinds
    assert kinds[:2] == ["response.created", "response.output_item.added"]
    assert "upstream_truncated" in caplog.text
    assert "stop_reason.max_tokens" in caplog.text


async def test_a_stream_that_stops_without_its_terminator_yields_no_terminator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responder, _, _ = _serving([_sse(_TEXT_STREAM[:3])])

    with caplog.at_level(logging.ERROR):
        _, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert "response.completed" not in [str(event["type"]) for event in _events(body)]
    assert "upstream_truncated" in caplog.text


async def test_an_upstream_refusal_is_relayed_with_its_status_and_body() -> None:
    """The Anthropic error envelope carries `error.message`, the field the Agent Runtime
    reads out of a failed request -- so relaying it is what makes the failure land as a
    failed Turn rather than as an unparsed body."""
    refusal = json.dumps(
        {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    ).encode()
    responder, _, _ = _serving(
        [refusal],
        status=429,
        headers={"content-type": "application/json", "retry-after": "30"},
    )

    status, headers, body = await _drain(_handler(responder), _turn(), _entry())

    assert status == 429
    assert body == refusal
    assert dict(headers)[b"retry-after"] == b"30"


async def test_an_upstream_refusal_says_why_in_this_services_own_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A refusal costs the tenant a Turn, and the relayed body reaches only the pod.

    That pod is deleted when the Session ends, so before this line the whole account of
    a failed Turn was a status code in an access log. Nineteen live runs went by with a
    translation defect in front of them and every service reporting itself healthy.

    The type is logged apart from the message because the type is the provider's own
    vocabulary while the message quotes the request back -- and on this wire the request
    carries the tenant's document, so the message is capped and the type is not.
    """
    refusal = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages.1: unexpected `tool_use_id`",
            },
        }
    ).encode()
    responder, _, _ = _serving([refusal], status=400)
    caplog.set_level(logging.WARNING, logger="managed_agent.gateway.model")

    status, _, body = await _drain(_handler(responder), _turn(), _entry())

    assert (status, body) == (400, refusal)
    said = [one.getMessage() for one in caplog.records if "upstream refused" in one.msg]
    assert len(said) == 1, [one.getMessage() for one in caplog.records]
    assert "invalid_request_error" in said[0]
    assert "unexpected `tool_use_id`" in said[0]
    assert str(_CALLER.session_id) in said[0]


async def test_a_refusal_body_that_is_not_an_error_envelope_is_still_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A gateway in front of the provider can refuse in its own shape, or in none.

    The log line exists to explain a dead Turn, so a body it cannot parse must still
    produce one -- the raw text is worth more than silence, and a diagnostic that only
    works on well-formed input is missing exactly when it is needed.
    """
    responder, _, _ = _serving([b"<html>502 upstream</html>"], status=502)
    caplog.set_level(logging.WARNING, logger="managed_agent.gateway.model")

    await _drain(_handler(responder), _turn(), _entry())

    said = [one.getMessage() for one in caplog.records if "upstream refused" in one.msg]
    assert len(said) == 1
    assert "502 upstream" in said[0]
    assert "type=unknown" in said[0]


async def test_a_relayed_refusal_drops_the_other_connections_headers() -> None:
    responder, _, _ = _serving(
        [b"{}"],
        status=400,
        headers={
            "content-type": "application/json",
            "connection": "close",
            "transfer-encoding": "chunked",
        },
    )

    _, headers, _ = await _drain(_handler(responder), _turn(), _entry())
    names = {name.lower() for name, _ in headers}

    assert b"connection" not in names
    assert b"transfer-encoding" not in names
    assert b"content-type" in names


async def test_an_unreachable_upstream_is_a_refusal_and_not_an_exception() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    handler = _handler(httpx.MockTransport(unreachable))

    with pytest.raises(GatewayRefusal) as caught:
        await _drain(handler, _turn(), _entry())

    assert caught.value.status == 502
    assert _MESSAGES_MODEL not in caught.value.message
    assert "claude-opus-5" in caught.value.message


async def test_the_upstream_exchange_is_closed_when_the_relay_ends() -> None:
    responder, _, stream = _serving([_sse(_TEXT_STREAM)])

    await _drain(_handler(responder), _turn(), _entry())

    assert stream.closed


async def test_a_request_carrying_an_unclassified_field_is_refused_naming_it() -> None:
    """MAP-A80: the failure names the feature that could not be carried rather than
    reporting a generic upstream error."""
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])
    handler = _handler(responder)
    turn = _turn(body=_responses_body(a_field_nothing_here_reads={"x": 1}))

    with pytest.raises(GatewayRefusal) as caught:
        await _drain(handler, turn, _entry())

    assert caught.value.status == 400
    assert "request.a_field_nothing_here_reads" in caught.value.message
    assert seen == []


async def test_a_refusal_never_names_the_shape_the_model_is_served_over() -> None:
    """Which upstream serves a model is this service's configuration. A caller that
    learns it learns something it did not send."""
    responder, _, _ = _serving([_sse(_TEXT_STREAM)])
    turn = _turn(body=_responses_body(a_field_nothing_here_reads={"x": 1}))

    with pytest.raises(GatewayRefusal) as caught:
        await _drain(_handler(responder), turn, _entry())

    assert "anthropic_messages" not in caught.value.message
    assert _FOUNDRY not in caught.value.message


async def test_nothing_reaches_the_vault_for_a_request_that_cannot_be_translated() -> (
    None
):
    """The credential is fetched on a Session's behalf and shows up in the vault's audit
    trail, so a read for a request that was never going to leave is a false entry in
    somebody's log."""
    responder, _, _ = _serving([_sse(_TEXT_STREAM)])
    vault = _OneSecretVault()
    handler = _handler(responder, vault=vault)
    turn = _turn(body=_responses_body(a_field_nothing_here_reads={"x": 1}))

    with pytest.raises(GatewayRefusal):
        await _drain(handler, turn, _entry())

    assert vault.reads == []


async def test_a_body_that_is_not_a_responses_request_is_refused() -> None:
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])
    turn = _turn(body=json.dumps({"model": 7}).encode())

    with pytest.raises(GatewayRefusal) as caught:
        await _drain(_handler(responder), turn, _entry())

    assert caught.value.status == 400
    assert seen == []


async def test_a_request_that_did_not_ask_to_stream_is_refused() -> None:
    """Everything downstream of the translation builds an SSE response, so a request
    asking for a whole body would get a stream anyway."""
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])
    turn = _turn(body=_responses_body(stream=False))

    with pytest.raises(GatewayRefusal) as caught:
        await _drain(_handler(responder), turn, _entry())

    assert caught.value.status == 400
    assert seen == []


async def test_a_call_whose_arguments_will_not_parse_is_refused_naming_it() -> None:
    responder, seen, _ = _serving([_sse(_TEXT_STREAM)])
    turn = _turn(
        body=_responses_body(
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_bad",
                    "name": "f",
                    "arguments": "{not json",
                }
            ]
        )
    )

    with pytest.raises(GatewayRefusal) as caught:
        await _drain(_handler(responder), turn, _entry())

    assert caught.value.status == 400
    assert "arguments_not_json" in caught.value.message
    assert seen == []


async def test_each_model_is_served_over_the_wire_its_routing_entry_names() -> None:
    """MAP-A105, through the real route: two models on two wires, one process, and the
    shape each is served over comes from its own Routing Entry rather than from the
    request or from which handler happens to be registered first."""
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/v1/messages"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_Chunks([_sse(_TEXT_STREAM)]),
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_Chunks([b'data: {"type":"response.completed"}\n\n']),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    broker = ProviderCredentialBroker(_OneSecretVault(), _FrozenClock())
    handlers: dict[UpstreamWire, WireHandler] = {
        UpstreamWire.RESPONSES: ResponsesPassthrough(client, broker),
        UpstreamWire.ANTHROPIC_MESSAGES: AnthropicMessagesHandler(
            client, broker, max_tokens=4096
        ),
    }
    gateway = ModelGateway(
        table=RoutingTable(
            [
                RoutingEntry(
                    model=_RESPONSES_MODEL,
                    wire=UpstreamWire.RESPONSES,
                    base_url="https://api.openai.com/v1",
                    auth_scheme=AuthScheme.BEARER,
                    # The spelling every other Python file in this tree uses for the
                    # pass-through's fixture credential. One name spelled two ways means
                    # at most one of them names something real, and a guard in
                    # `tests/` fails on the pair rather than leaving it to production.
                    credential_name="map/upstream/openai",
                ),
                RoutingEntry(
                    model="claude-opus-5",
                    wire=UpstreamWire.ANTHROPIC_MESSAGES,
                    base_url=_FOUNDRY,
                    auth_scheme=AuthScheme.API_KEY,
                    credential_name="map/dev/providers/anthropic",
                ),
            ]
        ),
        handlers=handlers,
        tokens=SessionTokenVerifier(key=_KEY, clock=_FrozenClock()),
    )
    token = mint_session_token(
        session_id=_SESSION,
        tenant_id=_TENANT,
        expiry_epoch_s=_NOW_MS // 1000 + 60,
        key=_KEY,
    )
    app = create_model_gateway_app(gateway)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://gateway.invalid"
    ) as caller:
        for model in (_RESPONSES_MODEL, "claude-opus-5"):
            answer = await caller.post(
                "/v1/responses",
                content=_responses_body(model=model),
                headers={
                    "authorization": f"Bearer {token}",
                    "content-type": "application/json",
                },
            )
            assert answer.status_code == 200, (model, answer.text)

    assert [request.url.path for request in seen] == [
        "/v1/responses",
        "/anthropic/v1/messages",
    ]
    assert json.loads(seen[0].content)["model"] == _RESPONSES_MODEL
    assert json.loads(seen[1].content)["model"] == "claude-opus-5"
    assert "max_tokens" in json.loads(seen[1].content)
    assert "max_tokens" not in json.loads(seen[0].content)


def _a_turn_body(tools: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {
            "model": "gsds-claude-opus-4-6",
            "instructions": "do the errand",
            "input": [],
            "tools": tools,
        }
    ).encode()


def test_a_turn_carrying_no_tool_says_so_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The counterpart to the Tool Gateway's offer census, on the other side of the pod.

    That census says what the Gateway OFFERED. Nothing said what the runtime actually
    handed the model, and the gap between the two is where a granted tool goes missing:
    a Session whose Gateway logged `registered=1 offered=1` ran nineteen Turns in which
    the model behaved as though it had no such tool, and no line anywhere could tell a
    runtime that dropped the tool from a tenant who granted none.

    WARNING and not INFO, because zero tools on a wire whose whole purpose is tool use
    is the case worth waking up for, and it is the one the deployment's filter must not
    drop. A Session legitimately granted nothing still reads this line and still wants
    it: the platform cannot tell the two apart from here, and saying so loudly costs one
    line on a Turn that was never going to call a tool anyway.
    """
    caplog.set_level(logging.WARNING, logger="managed_agent.gateway.model")

    census_of_tools_handed_to_the_model(
        _a_turn_body([]), "gsds-claude-opus-4-6", _SESSION
    )

    assert [one.levelname for one in caplog.records] == ["WARNING"], caplog.records
    assert "was handed no tool" in caplog.records[0].getMessage()


def test_a_turn_carrying_tools_names_them(caplog: pytest.LogCaptureFixture) -> None:
    """Names, and not just a count, because the count cannot answer the real question.

    `tool_count=3` on a Session granted three tools and `tool_count=3` on a Session
    granted three DIFFERENT ones are the same line. The name is what an operator greps
    for and what the tenant's own Grant is written in, so a listing that dropped one
    tool and kept the rest is visible here and is invisible under a count.
    """
    caplog.set_level(logging.INFO, logger="managed_agent.gateway.model")

    census_of_tools_handed_to_the_model(
        _a_turn_body(
            [
                {"type": "function", "name": "ask_deepwiki", "parameters": {}},
                {"type": "function", "name": "shell", "parameters": {}},
            ]
        ),
        "gsds-claude-opus-4-6",
        _SESSION,
    )

    message = caplog.records[0].getMessage()
    assert "tool census" in message
    assert "ask_deepwiki" in message and "shell" in message


def test_a_body_this_wire_cannot_read_is_not_a_census_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A diagnostic that can fail the Turn it is diagnosing is worse than no diagnostic.

    The refusal for an unreadable body is already raised by `_messages_body`, which owns
    that decision and answers the caller properly. This runs before it and must stay a
    pure observation: it returns without a line rather than raising a second, competing
    error out of a code path whose only job is to describe what it saw.
    """
    caplog.set_level(logging.INFO, logger="managed_agent.gateway.model")

    census_of_tools_handed_to_the_model(b"{not json", "gsds-claude-opus-4-6", _SESSION)

    assert caplog.records == []
