"""The one inbound route, and the token that decides whose Turn a request is.

Tier 1, no infrastructure. The route, the verifier, the routing table and now the
minter are all the real things; the only fake left is the wire handler (which would
otherwise be a provider). Faking the handler is the point of the `WireHandler` seam:
the route's job is to establish identity, pick a shape and relay, and none of those
needs a real upstream to grade.

Nothing is faked in front of the verifier any more, and that is this file's largest
change. The layout it read before was keyed from an entry in AWS Secrets Manager, so a
fake vault stood in for that call -- and the token itself was minted *here*, by a
private reimplementation of a format nothing in `src/` produced. Those two facts
compounded: the happy case proved the verifier accepted what this file minted, and no
test could prove it accepted what a real pod carries, because nothing built one.

`mint_session_token` closes that. It is the same function
`control/pod_config/compiler.py` calls when it compiles a Session's configuration, so a
token minted below is the token a pod is actually started with, and
`test_a_token_the_control_plane_would_mint_reaches _the_handler` is the case that could
not exist before. Every rejection case stays a mutation of what that minter produced, so
a verifier that quietly accepted some other format fails the happy case first -- and now
a *minter* that drifted from the verifier fails it too, which is the property a
test-local minter could never have.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping, MutableMapping
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.session.session_token import mint_session_token
from managed_agent.gateway.model.router import (
    MAX_REQUEST_BODY_BYTES,
    AuthScheme,
    CallerSession,
    InboundTurn,
    ModelGateway,
    RoutingEntry,
    RoutingTable,
    SessionTokenVerifier,
    UpstreamResponse,
    UpstreamWire,
    create_model_gateway_app,
)

_SESSION = SessionId(UUID("11111111-1111-4111-8111-111111111111"))
_TENANT = TenantId(UUID("22222222-2222-4222-8222-222222222222"))
_KEY = b"the-signing-key-both-gateways-hold"
_NOW_MS = 1_700_000_000_000
_NOW_S = _NOW_MS // 1000

_RESPONSES_MODEL = "gpt-5-codex"
_MESSAGES_MODEL = "claude-opus-5"
_UNHANDLED_MODEL = "some-chat-model"
_MESSAGES_BASE_URL = "https://map-foundry.services.ai.azure.com/anthropic"
_UNHANDLED_BASE_URL = "https://vendor.example.com/v1"


def _mint(
    *,
    session_id: SessionId = _SESSION,
    tenant_id: TenantId = _TENANT,
    expiry_epoch_s: int = _NOW_S + 60,
    key: bytes = _KEY,
) -> str:
    """Sign a token through the same function the control plane compiles into a pod.

    Deliberately a thin pass-through rather than a local implementation of the layout.
    `mint_session_token` is what `control/pod_config/compiler.py` calls, so a token from
    here is byte-for-byte one a real pod carries, and the happy cases below can fail:
    a minter and a verifier that drift apart in either direction break them. A local
    re-implementation could only ever prove this file agrees with itself.
    """
    return mint_session_token(
        session_id=session_id,
        tenant_id=tenant_id,
        expiry_epoch_s=expiry_epoch_s,
        key=key,
    )


class _FixedClock:
    def __init__(self, now_ms: int = _NOW_MS) -> None:
        self.now_ms = now_ms

    def now_epoch_ms(self) -> int:
        return self.now_ms


class _Exchange:
    """One open upstream exchange, which closes only when somebody closes it.

    Written as an explicit `__aenter__`/`__aexit__` pair rather than with
    `@asynccontextmanager`, and that is the whole point of the class. An async generator
    also runs its `finally` when the interpreter finalises it, so a route that dropped
    the exchange instead of closing it still reported `closed == 1` once garbage
    collection caught up -- the assertion passed while the behaviour it names was gone.
    Nothing finalises a plain object's `__aexit__`, so `closed` here counts real closes.
    """

    def __init__(self, handler: _FakeHandler) -> None:
        self._handler = handler

    async def __aenter__(self) -> UpstreamResponse:
        return UpstreamResponse(
            status=self._handler.status,
            headers=self._handler.headers,
            body=self._handler.body(),
        )

    async def __aexit__(self, *exc_info: object) -> None:
        self._handler.closed += 1
        self._handler.order.append("close")


class _FakeHandler:
    """One wire shape, recorded rather than served.

    `hold` leaves the body stream open after its last chunk, which is how the
    client-hang-up case gets a stream that is still live when the caller goes away.

    `fails_after` makes the body raise once it has handed out that many chunks, and it
    is the one fault here that is neither synthetic nor rare: a real upstream stream
    raises `ReadTimeout` when an SSE response goes quiet -- which is the whole reason
    `MAP_UPSTREAM_READ_TIMEOUT_S` exists -- and `RemoteProtocolError` when a provider or
    a load balancer drops a response mid-body. `fails_after=0` is the same fault before
    any chunk went out; both are after the route returned, which is what distinguishes
    them from the header-block fault below.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        headers: tuple[tuple[bytes, bytes], ...] = (
            (b"content-type", b"text/event-stream"),
        ),
        chunks: tuple[bytes, ...] = (b"data: one\n\n", b"data: two\n\n"),
        hold: bool = False,
        fails_after: int | None = None,
    ) -> None:
        self.status = status
        self.headers = headers
        self.chunks = chunks
        self.hold = hold
        self.fails_after = fails_after
        self.opened: list[tuple[InboundTurn, RoutingEntry]] = []
        self.closed = 0
        self.forever = asyncio.Event()
        self.order: list[str] = []
        """Every chunk handed out and every close, in the order they happened.

        The order is the property, not just the count: an exchange closed the instant it
        opened is closed exactly once too, and a relay that did that would hand a real
        upstream a dead socket to read the body from."""

    def open(self, turn: InboundTurn, entry: RoutingEntry) -> _Exchange:
        self.opened.append((turn, entry))
        return _Exchange(self)

    async def body(self) -> AsyncIterator[bytes]:
        for handed, chunk in enumerate(self.chunks):
            if handed == self.fails_after:
                raise RuntimeError("the upstream stream died mid-body")
            self.order.append("chunk")
            yield chunk
        if self.fails_after == len(self.chunks):
            raise RuntimeError("the upstream stream died mid-body")
        if self.hold:
            await self.forever.wait()


def _table() -> RoutingTable:
    return RoutingTable(
        [
            RoutingEntry(
                model=_RESPONSES_MODEL,
                wire=UpstreamWire.RESPONSES,
                base_url="https://api.openai.com/v1",
                auth_scheme=AuthScheme.BEARER,
                credential_name="map/upstream/openai",
            ),
            RoutingEntry(
                model=_MESSAGES_MODEL,
                wire=UpstreamWire.ANTHROPIC_MESSAGES,
                base_url=_MESSAGES_BASE_URL,
                auth_scheme=AuthScheme.API_KEY,
                credential_name="map/upstream/foundry",
            ),
            RoutingEntry(
                model=_UNHANDLED_MODEL,
                wire=UpstreamWire.CHAT_COMPLETIONS,
                base_url=_UNHANDLED_BASE_URL,
                auth_scheme=AuthScheme.BEARER,
                credential_name="map/upstream/vendor",
            ),
        ]
    )


def _gateway(
    handlers: Mapping[UpstreamWire, Any],
    *,
    clock: _FixedClock | None = None,
) -> ModelGateway:
    return ModelGateway(
        table=_table(),
        handlers=handlers,
        tokens=SessionTokenVerifier(key=_KEY, clock=clock or _FixedClock()),
    )


def _client(gateway: ModelGateway) -> httpx.AsyncClient:
    app = create_model_gateway_app(gateway)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://gateway.invalid"
    )


def _body(model: str = _RESPONSES_MODEL, **extra: object) -> bytes:
    return json.dumps({"model": model, "input": [], **extra}).encode()


def _headers(token: str) -> dict[str, str]:
    return {"content-type": "application/json", "authorization": f"Bearer {token}"}


# --- the verifier on its own ------------------------------------------------------


async def test_a_signed_unexpired_token_names_its_session_and_its_tenant() -> None:
    verifier = SessionTokenVerifier(key=_KEY, clock=_FixedClock())

    assert await verifier.resolve(_mint()) == CallerSession(
        session_id=_SESSION, tenant_id=_TENANT
    )


def _flip_a_signature_byte(token: str) -> str:
    session, tenant, expiry, signature = token.split(".")
    flipped = "0" if signature[-1] != "0" else "1"
    return ".".join((session, tenant, expiry, signature[:-1] + flipped))


@pytest.mark.parametrize(
    ("name", "token"),
    [
        ("a flipped signature byte", _flip_a_signature_byte(_mint())),
        ("a signature from another key", _mint(key=b"not-the-signing-key")),
        ("no dots at all", "onepart"),
        ("three parts where four are required", ".".join(_mint().split(".")[:3])),
        ("five parts where four are required", _mint() + ".extra"),
        # The three parsed fields, each unparseable in turn, and each mutated *after*
        # a real mint so the signature is the thing that rejects it. A verifier that
        # parsed before it compared would pass these for a different reason and this
        # file could not tell the two apart.
        ("a session id that is not a uuid", "not-a-uuid." + _mint().partition(".")[2]),
        (
            "an expiry that is not a number",
            ".".join((str(_SESSION), str(_TENANT), "soon", "0" * 64)),
        ),
        ("an expiry at exactly now", _mint(expiry_epoch_s=_NOW_S)),
        ("an expiry already past", _mint(expiry_epoch_s=_NOW_S - 1)),
        ("an empty token", ""),
    ],
)
async def test_a_token_that_proves_nothing_names_no_session(
    name: str, token: str
) -> None:
    """Every rejection is the same answer, so nothing is learned from which one
    it was."""
    verifier = SessionTokenVerifier(key=_KEY, clock=_FixedClock())

    assert await verifier.resolve(token) is None, name


async def test_a_token_the_control_plane_would_mint_reaches_the_handler() -> None:
    """The case that could not exist while this file minted its own layout.

    Before, the happy path proved only that the verifier accepted what this module
    produced; nothing anywhere proved it accepted what a pod actually carries, because
    nothing in the tree minted one. The token below is minted by the control plane's
    own function with the control plane's own argument names, and it is asserted to get
    all the way past authentication to an opened upstream exchange -- so a verifier
    that agreed with a private test format and disagreed with the real one fails here.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})
    token = mint_session_token(
        session_id=_SESSION,
        tenant_id=_TENANT,
        expiry_epoch_s=_NOW_S + 3_600,
        key=_KEY,
    )

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses", content=_body(), headers=_headers(token)
        )

    assert answer.status_code == 200, answer.text
    assert len(handler.opened) == 1
    turn, _ = handler.opened[0]
    assert turn.caller == CallerSession(session_id=_SESSION, tenant_id=_TENANT)


async def test_the_key_is_held_rather_than_fetched_on_every_request() -> None:
    """This runs on every model request of every Turn, so it does no IO at all.

    Asserted structurally rather than by counting calls, because there is no longer a
    collaborator to count. The verifier holds the key as bytes, which is the property
    that matters: the previous verifier read it from a secret store behind a TTL, so a
    store outage or an expired window turned every model call in the fleet into a
    refusal. Nothing here can fail that way, and this test exists to notice if a fetch
    is ever reintroduced.
    """
    verifier = SessionTokenVerifier(key=_KEY, clock=_FixedClock())

    assert verifier.key == _KEY
    assert not [name for name in dir(verifier) if "vault" in name or "fetch" in name]


# --- the route --------------------------------------------------------------------


async def test_a_turn_is_relayed_with_the_upstreams_status_body_and_headers() -> None:
    """Nothing this service authored appears in what the Agent Runtime reads back."""
    handler = _FakeHandler(
        status=201,
        headers=(
            (b"content-type", b"text/event-stream"),
            (b"x-codex-primary-used-percent", b"12"),
        ),
        chunks=(b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"),
    )
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with (
        _client(gateway) as client,
        client.stream(
            "POST", "/v1/responses", content=_body(), headers=_headers(_mint())
        ) as response,
    ):
        status = response.status_code
        names = dict(response.headers)
        chunks = [chunk async for chunk in response.aiter_bytes() if chunk]

    assert status == 201
    assert b"".join(chunks) == b"data: a\n\ndata: b\n\ndata: c\n\n"
    assert names == {
        "content-type": "text/event-stream",
        "x-codex-primary-used-percent": "12",
    }
    assert handler.order == ["chunk", "chunk", "chunk", "close"], (
        "the exchange has to outlive the body: closed once, and after the last chunk"
    )


async def test_the_declared_shape_is_the_shape_that_serves_the_model() -> None:
    """A model on the translated shape never reaches the pass-through beside it."""
    passthrough = _FakeHandler()
    messages = _FakeHandler()
    gateway = _gateway(
        {
            UpstreamWire.RESPONSES: passthrough,
            UpstreamWire.ANTHROPIC_MESSAGES: messages,
        }
    )

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=_body(_MESSAGES_MODEL),
            headers=_headers(_mint()),
        )

    assert answer.status_code == 200
    assert not passthrough.opened
    assert [entry.base_url for _, entry in messages.opened] == [_MESSAGES_BASE_URL]
    assert [turn.model for turn, _ in messages.opened] == [_MESSAGES_MODEL]


async def test_the_body_reaches_the_handler_as_the_bytes_that_arrived() -> None:
    """Spelled by hand, so a re-serialisation inside the route would be visible.

    A body built with `json.dumps` cannot grade this: dumping what `json.loads` gives
    back reproduces it byte for byte, so a route that re-encoded what it parsed would
    pass. This one puts `model` in the middle and doubles a space, both of which an
    encoder normalises away -- and the guard's whole claim is that a field this route
    never reads reaches the handler exactly as written, which only holds if nothing
    between here and there re-encodes anything.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})
    sent = (
        b'{"input": [],  "model": "gpt-5-codex", "store": true,'
        b' "a_field_nothing_here_reads": {"deep": [1, 2]}}'
    )
    assert sent != json.dumps(json.loads(sent)).encode(), (
        "this body survives an encode unchanged, so it would grade nothing"
    )

    async with _client(gateway) as client:
        await client.post("/v1/responses", content=sent, headers=_headers(_mint()))

    assert [turn.body for turn, _ in handler.opened] == [sent]


async def test_an_unauthenticated_caller_gets_none_of_its_body_into_this_process() -> (
    None
):
    """Identity first, and the body only afterwards.

    The route once buffered the whole body on the line above the token check, so a body
    of any size from a caller with no valid token was fully resident before anything
    looked at the token. Starlette caps nothing, and this Deployment had no memory
    limit, so the kubelet would have reclaimed memory from co-tenant pods on the node.

    Graded by counting the bytes the ASGI app actually pulled off the stream rather than
    by timing or by memory, because a route that read the body and then discarded it is
    indistinguishable from one that never read it in any other way. A generator body is
    what makes the count meaningful: httpx sends it in chunks, and the `sent` counter
    stops wherever the app stopped asking.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})
    sent = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal sent
        for _ in range(64):
            sent += 1024
            yield b"x" * 1024

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=body(),
            headers={"content-type": "application/json", "authorization": "Bearer no"},
        )

    assert answer.status_code == 401
    assert sent == 0, f"{sent} bytes were read before the token was checked"
    assert not handler.opened


async def test_a_body_over_the_cap_is_refused_rather_than_buffered_whole() -> None:
    """The cap exists so the pod's memory request is a number rather than a hope.

    `content-length` is deliberately not what is checked -- the caller writes that
    header, so a cap keyed on it is a cap under the control of whoever it is for. This
    body declares nothing (httpx chunks a generator), and the refusal still lands.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})
    over = MAX_REQUEST_BODY_BYTES + 1

    async def body() -> AsyncIterator[bytes]:
        for _ in range(over // 65536 + 1):
            yield b"x" * 65536

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses", content=body(), headers=_headers(_mint())
        )

    assert answer.status_code == 413
    assert answer.json()["error"]["type"] == "invalid_request"
    assert not handler.opened


async def test_a_body_at_the_cap_is_served() -> None:
    """The boundary from the other side, so the cap cannot drift into refusing
    everything and still look like it works."""
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})
    padding = "y" * (MAX_REQUEST_BODY_BYTES - len(_body(_RESPONSES_MODEL, pad="")))

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=_body(_RESPONSES_MODEL, pad=padding),
            headers=_headers(_mint()),
        )

    assert answer.status_code == 200
    assert [len(turn.body) for turn, _ in handler.opened] == [MAX_REQUEST_BODY_BYTES]


async def test_a_body_naming_the_model_twice_is_refused_rather_than_resolved() -> None:
    """Which duplicate wins decides which credential is attached.

    `json.loads` keeps the last, so the route would attach the credential for the second
    `model` and forward the body byte-identical; an upstream whose parser keeps the
    first then serves the first model's request on the second model's credential, and
    the meter reading usage on the way back records the second. RFC 8259 leaves the case
    unspecified, which is exactly why this service cannot relay it.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=b'{"model": "' + _MESSAGES_MODEL.encode() + b'", "input": [],'
            b' "model": "' + _RESPONSES_MODEL.encode() + b'"}',
            headers=_headers(_mint()),
        )

    assert answer.status_code == 400
    assert answer.json()["error"]["type"] == "invalid_request"
    assert not handler.opened, (
        "the body reached a handler, so one of its two models picked the credential"
    )


async def test_a_duplicated_upstream_response_header_reaches_the_caller_twice() -> None:
    """The handler preserves the upstream's duplicates; the route must not collapse
    them.

    `headers=dict(...)` kept one value per name, which threw away all but the last -- so
    the care the handler takes over `multi_items()` bought nothing, and an upstream
    sending two `set-cookie` headers had one silently dropped on the way through.
    """
    handler = _FakeHandler(
        headers=(
            (b"content-type", b"text/event-stream"),
            (b"set-cookie", b"a=1"),
            (b"set-cookie", b"b=2"),
        )
    )
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses", content=_body(), headers=_headers(_mint())
        )

    assert answer.headers.get_list("set-cookie") == ["a=1", "b=2"]


async def test_a_non_ascii_upstream_header_relays_and_closes_the_exchange() -> None:
    """One header value above U+00FF answered 500 and leaked the upstream connection.

    Measured before the fix, one header varied and everything else identical: ASCII and
    latin-1 values both gave 200 with the exchange closed, and a UTF-8 value gave 500
    with it still open. The relay decoded the block with a charset httpx sniffed across
    the whole of it and re-encoded each value as latin-1, which raised on the value that
    had made it pick UTF-8 -- and that raise landed after the exchange was open and
    before the response that owns closing it existed, so nothing closed it.

    A provider emitting such a header on a class of responses -- a localised error
    detail, a `content-disposition` filename -- therefore failed every one of those
    Turns *and* leaked a connection out of the single process-lifetime client, until the
    pool was drained and this service relayed nothing at all.

    The closed count is the load-bearing assertion. A 200 alone would pass on a relay
    that answered correctly and held the socket.
    """
    detail = "caf\u00e9-\u4e2d\u6587"
    handler = _FakeHandler(
        headers=(
            (b"content-type", b"text/event-stream"),
            (b"x-upstream-detail", detail.encode("utf-8")),
        )
    )
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses", content=_body(), headers=_headers(_mint())
        )

    assert answer.status_code == 200
    assert answer.headers["x-upstream-detail"] == detail
    assert handler.closed == 1, "the upstream connection was left open"


class _HeadersThatCannotBeRead:
    """An upstream whose header block raises when the route reads it.

    Stands in for any failure between the moment the exchange is open and the moment
    Starlette holds a response that owns closing it. Nothing in the shipped code raises
    in that window any more -- the latin-1 encode that did was the bug above -- and that
    is exactly why the guard needs its own fault: a `try` that closes the exchange on
    the way out is only worth having if removing it fails a test.
    """

    def __iter__(self) -> AsyncIterator[tuple[bytes, bytes]]:
        raise RuntimeError("this header block cannot be read")


async def test_a_failure_after_the_upstream_opens_still_closes_the_exchange() -> None:
    """No path out of this route leaves the upstream connection open.

    The leak above was not really about encodings: it was that the window between
    `enter_async_context` and the returned response had no cleanup at all, so whatever
    raised in it kept a connection. Fixing only the encoding would leave the next line
    added there with the same property, and a connection held per failure against one
    process-lifetime client is a service that stops relaying rather than one that logs.
    """
    handler = _FakeHandler()
    handler.headers = cast(
        "tuple[tuple[bytes, bytes], ...]", _HeadersThatCannotBeRead()
    )
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    with pytest.raises(RuntimeError, match="cannot be read"):
        async with _client(gateway) as client:
            await client.post(
                "/v1/responses", content=_body(), headers=_headers(_mint())
            )

    assert handler.closed == 1, (
        "the request failed and the upstream connection was left open"
    )


async def _drive_one_turn(
    app: Any,
) -> tuple[list[MutableMapping[str, Any]], str | None]:
    """One POST through the ASGI callable, returning what was sent and what escaped.

    A callable and not a client, because the fault under test happens *after* the
    response was returned: an HTTP client turns that into a transport error of its own
    and hides whether the app raised, and it is the app's own unwinding that decides
    whether anything closes the exchange.
    """
    payload = _body()
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}

    sent: list[MutableMapping[str, Any]] = []

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", f"Bearer {_mint()}".encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("gateway.invalid", 443),
    }
    escaped: str | None = None
    try:
        async with asyncio.timeout(5):
            await app(scope, receive, send)
    except BaseException as exc:  # noqa: BLE001 - the fault under test is the subject
        escaped = type(exc).__name__
    return sent, escaped


@pytest.mark.parametrize("fails_after", [0, 1, 2])
async def test_a_stream_that_fails_mid_body_still_closes_the_exchange(
    fails_after: int,
) -> None:
    """The half of this window that a real upstream actually fails in.

    The guard above covers the window *before* the response exists, and needs an
    injected fault to do it because after the encoding fix nothing reachable raises
    there. This is the other half, and its fault needs no injecting: `aiter_raw` raises
    `ReadTimeout` when an SSE stream goes quiet -- `MAP_UPSTREAM_READ_TIMEOUT_S=240` is
    in the manifest for exactly that -- and `RemoteProtocolError` when a provider or a
    load balancer truncates a response. Both land past the `return`, where the route's
    `try` cannot see them.

    Measured before the fix, at every one of these three points: `exchange.closed=0`.
    The route's only cleanup there was `BackgroundTask(stack.aclose)`, and Starlette
    runs a background task only for a response whose stream *completed* -- so the
    handler's own `finally` never ran on a stream that failed. Three points and not one
    because the boundary matters: `fails_after=0` is before a byte of body went out, so
    nothing can rely on "the response is already in flight", and `2` is after the last
    chunk, where the generator is closing anyway.

    `closed == 1` and not `>= 1`: a route that closed on the way in *and* on the way out
    would satisfy a lower bound while handing a real upstream a dead socket to stream
    from, which is what `_Exchange` counts real closes to catch.
    """
    handler = _FakeHandler(fails_after=fails_after)
    app = create_model_gateway_app(_gateway({UpstreamWire.RESPONSES: handler}))

    sent, escaped = await _drive_one_turn(app)

    assert escaped == "RuntimeError", (
        f"the stream's own failure did not reach the caller at all: {escaped}"
    )
    assert handler.order.count("chunk") == fails_after, handler.order
    assert handler.closed == 1, (
        "the upstream stream failed mid-body and the exchange was left open -- "
        f"chunks handed out: {handler.order.count('chunk')}, closes: {handler.closed}"
    )
    assert handler.order[-1] == "close", handler.order


async def test_a_stream_that_completes_closes_the_exchange_once_and_not_twice() -> None:
    """The ordinary path, asserted on the count rather than only on the body.

    Two things now close this exchange -- the relay wrapper when the stream ends, and
    the background task Starlette runs afterwards -- and they are the same `aclose` on
    the same stack rather than two cleanups. If that stopped being true, this counts two
    closes on a healthy Turn, and a handler doing real work in `__aexit__` would do it
    twice.
    """
    handler = _FakeHandler()
    app = create_model_gateway_app(_gateway({UpstreamWire.RESPONSES: handler}))

    sent, escaped = await _drive_one_turn(app)

    assert escaped is None, escaped
    assert handler.closed == 1, f"closed {handler.closed} times, not once"
    assert handler.order == ["chunk", "chunk", "close"], handler.order


async def test_the_token_and_not_the_body_says_whose_turn_this_is() -> None:
    """`client_metadata` is the runtime's own tagging (ADR-007), never an identity.

    Attributing a Turn to a session id the pod wrote into its body would let a pod name
    a Session it does not run -- and spend, Grants and the Event Log all hang off that
    attribution.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})
    impostor = str(uuid4())

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=_body(
                _RESPONSES_MODEL,
                client_metadata={"session_id": impostor, "thread_id": str(uuid4())},
            ),
            headers=_headers(_mint()),
        )

    assert answer.status_code == 200
    assert [turn.caller for turn, _ in handler.opened] == [
        CallerSession(session_id=_SESSION, tenant_id=_TENANT)
    ]
    assert impostor != str(_SESSION)


@pytest.mark.parametrize(
    ("name", "headers"),
    [
        ("no authorization at all", {"content-type": "application/json"}),
        (
            "a scheme that is not bearer",
            {"content-type": "application/json", "authorization": "Basic abc"},
        ),
        (
            # A *valid* token under the wrong scheme. With a junk token the case proves
            # nothing: the token would fail to resolve and answer 401 anyway, so a route
            # that had stopped checking the scheme at all would still pass.
            "a good token presented under a scheme this service does not accept",
            {"content-type": "application/json", "authorization": f"Token {_mint()}"},
        ),
        (
            "bearer with nothing after it",
            {"content-type": "application/json", "authorization": "Bearer "},
        ),
        ("a token nothing signed", _headers("forged.token")),
        ("an expired token", _headers(_mint(expiry_epoch_s=_NOW_S - 1))),
    ],
)
async def test_a_request_that_proves_no_session_is_refused_unauthenticated(
    name: str, headers: dict[str, str]
) -> None:
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post("/v1/responses", content=_body(), headers=headers)

    assert answer.status_code == 401, name
    assert answer.json()["error"]["type"] == "unauthenticated"
    assert not handler.opened


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("not JSON", b"not json at all"),
        ("a JSON array", b"[]"),
        ("a JSON string", b'"gpt-5-codex"'),
        ("an object naming no model", b"{}"),
        ("an object whose model is not a string", b'{"model": 5}'),
        ("an object whose model is empty", b'{"model": ""}'),
    ],
)
async def test_a_body_that_names_no_model_is_a_bad_request(
    name: str, payload: bytes
) -> None:
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses", content=payload, headers=_headers(_mint())
        )

    assert answer.status_code == 400, name
    assert answer.json()["error"]["type"] == "invalid_request"
    assert not handler.opened


async def test_an_undeclared_model_is_refused_by_the_name_the_tenant_used() -> None:
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=_body("gpt-5-codex-preview"),
            headers=_headers(_mint()),
        )

    assert answer.status_code == 404
    assert "gpt-5-codex-preview" in answer.json()["error"]["message"]
    assert not handler.opened


async def test_a_shape_with_no_handler_refuses_without_naming_the_upstream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Which upstream refused is operator-visible in a log line and caller-visible
    nowhere. The log assertion is what keeps that negative diagnosable rather than
    merely quiet."""
    gateway = _gateway({UpstreamWire.RESPONSES: _FakeHandler()})

    with caplog.at_level(logging.ERROR, logger="managed_agent.gateway.model.router"):
        async with _client(gateway) as client:
            answer = await client.post(
                "/v1/responses",
                content=_body(_UNHANDLED_MODEL),
                headers=_headers(_mint()),
            )

    body = answer.text
    assert answer.status_code == 502
    assert _UNHANDLED_MODEL in answer.json()["error"]["message"]
    assert _UNHANDLED_BASE_URL not in body
    assert UpstreamWire.CHAT_COMPLETIONS.value not in body
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert _UNHANDLED_BASE_URL in logged
    assert UpstreamWire.CHAT_COMPLETIONS.value in logged


@pytest.mark.parametrize(
    ("octet", "named"),
    [(b"\r", "0x0d"), (b"\n", "0x0a"), (b"\x00", "0x00"), (b"\x7f", "0x7f")],
)
async def test_a_header_octet_that_cannot_be_in_a_header_is_refused(
    octet: bytes, named: str
) -> None:
    """The two octets a byte-faithful relay must not be faithful about.

    Once the request direction carries the caller's own octets, it carries whatever it
    is given -- and `httpx` does not check: measured, it builds a request with
    `b"a\r\nx-injected: yes"` in a header value without complaint. CR and LF end a
    header, so relaying either would make one pod-supplied value into a second header on
    the upstream hop, which is a worse outcome than the 500 this replaced.

    Nothing conformant puts one of these in the ASGI scope today, `h11` included, so
    this refuses nothing that can currently arrive. That is the point: the alternative
    is a filter whose safety is a property of the server in front of it rather than of
    itself.

    Refused in the envelope, naming the octet in hex. The octet's *number* and not the
    value it came in: the value is the pod's, and a message that echoed it would put pod
    bytes wherever this message is read.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=_body(),
            headers=_headers(_mint())
            | {"user-agent": f"codex{octet.decode('latin-1')}"},
        )

    assert answer.status_code == 400
    assert answer.json()["error"]["type"] == "invalid_request"
    assert named in answer.json()["error"]["message"]
    assert not handler.opened, (
        "the request reached a handler, so the octet was on its way to the upstream"
    )


async def test_a_header_whose_name_carries_a_control_octet_is_refused() -> None:
    """The name half of the rule above, and it is not decoration.

    A control octet in a *name* has one route through this filter that a name in the
    allowlist does not: the `x-codex-` prefix is open-ended by design, so a name
    spelled `x-codex-` then CRLF then `x-injected: yes` matches it -- measured,
    `_forwarded` returns True -- and would go out as a name plus a second header.
    A CR inside `user-agent` is refused by the allowlist and needs no help.

    Written because checking only the value survived mutation: with `for field in
    (value,)` the whole suite stayed green at 1493 passed. Every guard here for the
    value half was blind to the name half.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})
    app = create_model_gateway_app(gateway)
    payload = _body()
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}

    sent: list[MutableMapping[str, Any]] = []

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    # Written into the raw scope, because a client library will not build a header name
    # with a CR in it -- which is the whole reason this filter cannot rely on one.
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", f"Bearer {_mint()}".encode()),
            (b"x-codex-\r\nx-injected: yes", b"anything"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("gateway.invalid", 443),
    }

    async with asyncio.timeout(5):
        await app(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 400, start
    assert not handler.opened, (
        "the request reached a handler, so a name carrying CR was on its way out "
        "through the open-ended x-codex- prefix"
    )


@pytest.mark.parametrize("octet", [b"\t", b" "])
async def test_the_two_whitespace_octets_a_header_may_carry_still_travel(
    octet: bytes,
) -> None:
    """The refusal above is the C0 controls and DEL, not "anything unusual".

    HTAB and SP are legal in a field value (RFC 9110 §5.5), and a filter that refused
    them would refuse a `user-agent` with a space in it -- which is most of them.
    """
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=_body(),
            headers=_headers(_mint()) | {"user-agent": f"codex{octet.decode()}cli"},
        )

    assert answer.status_code == 200
    turn, _ = handler.opened[0]
    assert (b"user-agent", b"codex" + octet + b"cli") in turn.headers, turn.headers


async def test_an_inbound_turn_renders_neither_its_token_its_key_nor_its_body() -> None:
    """The one record in this module that carried three secrets under a default repr.

    `credential_broker.py` redacts both of its records for a stated reason -- redaction
    "closes the ordinary route by which secrets reach logs, an f-string somebody added
    while debugging". This record travels further than either of those: it is what
    this slice publishes to the wire handlers later slices write, and it carries the
    pod's own bearer token, whatever credential header the pod attached, *and* the
    request body, which is the tenant's conversation.

    Latent rather than live -- nothing in `src/` renders one today -- which is exactly
    why it is worth closing now. The line that does it will be an `_LOG.error("... %s",
    turn)` in a slice whose author never read this file's other two dataclasses.

    The header *names* stay, because they are what makes such a log line worth having
    and a name is not a secret. Every value goes.
    """
    pod_token = "POD-BEARER-TOKEN-VALUE"
    pod_key = "POD-SUPPLIED-PROVIDER-KEY"
    conversation = "the tenant said something private"
    turn = InboundTurn(
        caller=CallerSession(session_id=_SESSION, tenant_id=_TENANT),
        model=_RESPONSES_MODEL,
        headers=(
            (b"authorization", f"Bearer {pod_token}".encode()),
            (b"x-api-key", pod_key.encode()),
            (b"content-type", b"application/json"),
        ),
        body=json.dumps({"model": _RESPONSES_MODEL, "input": conversation}).encode(),
    )

    for rendering in (
        repr(turn),
        str(turn),
        f"{turn}",
        repr([turn]),
        repr({"t": turn}),
    ):
        assert pod_token not in rendering, rendering
        assert pod_key not in rendering, rendering
        assert conversation not in rendering, rendering
        assert "redacted" in rendering, rendering
        assert "authorization" in rendering, rendering
        assert _RESPONSES_MODEL in rendering, rendering


async def test_a_compressed_request_body_is_refused_by_naming_its_coding() -> None:
    """Nothing here decompresses one, and reading it as JSON would refuse it as
    malformed -- which is the same failure with the cause hidden."""
    handler = _FakeHandler()
    gateway = _gateway({UpstreamWire.RESPONSES: handler})

    async with _client(gateway) as client:
        answer = await client.post(
            "/v1/responses",
            content=_body(),
            headers=_headers(_mint()) | {"content-encoding": "zstd"},
        )

    assert answer.status_code == 415
    assert "zstd" in answer.json()["error"]["message"]
    assert not handler.opened


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/v1/responses/compact", 404),
        ("POST", "/responses", 404),
        ("GET", "/v1/responses", 405),
    ],
)
async def test_only_the_one_path_the_runtime_builds_is_served(
    method: str, path: str, expected: int
) -> None:
    """The runtime reaches `responses/compact` only when it believes it is talking to
    first-party OpenAI or Azure, which this service's provider entry is not."""
    gateway = _gateway({UpstreamWire.RESPONSES: _FakeHandler()})

    async with _client(gateway) as client:
        answer = await client.request(
            method, path, content=_body(), headers=_headers(_mint())
        )

    assert answer.status_code == expected


async def test_liveness_names_no_model_no_upstream_and_no_session() -> None:
    gateway = _gateway({UpstreamWire.RESPONSES: _FakeHandler()})

    async with _client(gateway) as client:
        answer = await client.get("/v1/healthz")

    assert answer.status_code == 200
    assert answer.json() == {"status": "ok"}


async def test_a_hang_up_while_a_chunk_is_in_flight_still_closes_the_exchange() -> None:
    """The one exit the relay wrapper cannot cover, and the reason two sites remain.

    A cancelled task does not run the `finally` of an async generator suspended at a
    `yield` -- the interpreter runs it at finalisation, which is garbage collection or
    loop shutdown, not now. So *where* the cancellation lands decides which cleanup can
    see it. In the hang-up test below the cancellation lands inside the generator, in
    `body`'s own await, so the wrapper's `finally` runs and the background task is
    redundant. Here `send` is what blocks, so the cancellation lands in Starlette's
    `stream_response` with the generator parked at a `yield`, and the background task is
    the only thing left that closes anything.

    Written because the background task turned out to be ungraded once the wrapper
    existed: replacing it with a no-op left the whole suite green at 1494 passed. The
    wrapper had taken over the one test that used to grade it, which is how a line stops
    being covered without anybody deleting a test.
    """
    handler = _FakeHandler()
    app = create_model_gateway_app(_gateway({UpstreamWire.RESPONSES: handler}))
    payload = _body()
    first_chunk_in_flight = asyncio.Event()
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        # Held until a body chunk is actually stuck in `send`, so the disconnect cannot
        # arrive before the state this test is about exists.
        await first_chunk_in_flight.wait()
        return {"type": "http.disconnect"}

    never = asyncio.Event()
    sent: list[MutableMapping[str, Any]] = []

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_in_flight.set()
            await never.wait()

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", f"Bearer {_mint()}".encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("gateway.invalid", 443),
    }

    async with asyncio.timeout(5):
        await app(scope, receive, send)

    assert handler.order.count("chunk") == 1, handler.order
    assert handler.closed == 1, (
        "the caller hung up while a chunk was stuck in send, and nothing closed the "
        f"upstream exchange: {handler.order}"
    )


async def test_the_upstream_exchange_is_closed_when_the_caller_hangs_up() -> None:
    """A pod that dies mid-stream must not leave the upstream connection open.

    Driven against the ASGI callable rather than a client, because a client that
    disconnects is exactly what a client library will not do for you: the disconnect
    message is the thing under test.
    """
    handler = _FakeHandler(chunks=(b"data: a\n\n",), hold=True)
    app = create_model_gateway_app(_gateway({UpstreamWire.RESPONSES: handler}))
    token = _mint()
    payload = _body()
    hung_up = asyncio.Event()
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await hung_up.wait()
        return {"type": "http.disconnect"}

    sent: list[MutableMapping[str, Any]] = []

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            hung_up.set()

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", f"Bearer {token}".encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("gateway.invalid", 443),
    }

    async with asyncio.timeout(5):
        await app(scope, receive, send)

    assert [m["type"] for m in sent][:2] == [
        "http.response.start",
        "http.response.body",
    ]
    assert handler.closed == 1, "the handler's context outlived the caller"
    assert not handler.forever.is_set()
