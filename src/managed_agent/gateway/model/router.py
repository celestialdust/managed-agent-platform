"""Which upstream serves a model, read from configuration and never inferred.

A model name is a routing key rather than a promise about what a model can do (ADR-010),
so a Routing Entry is the whole of what this platform knows about one: which of the
three request shapes its traffic takes outward, where that shape is sent, which
credential opens it, and how that credential is presented.

The lookup fails rather than falls back. There is no default entry, nothing infers a
shape from a base URL, and no second shape is attempted after one fails -- so this
service's behaviour never depends on something nobody configured, and a guess that
happened to work would be indistinguishable afterwards from a correct answer (ADR-016).

The table is parsed once into typed values, so every later reader holds a RoutingEntry
rather than a mapping that might be missing a key. A document naming a shape this build
does not have, or naming one model twice, is refused at that parse instead of at the
first Turn that needs it.

The import block below is the whole module's; the inbound surface in the second half of
this file adds none.
"""

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.ports import Clock
from managed_agent.core.session.session_token import (
    InvalidSessionToken,
    verify_session_token,
)

_LOG = logging.getLogger(__name__)


class UpstreamWire(StrEnum):
    """The three request shapes this service speaks outward.

    Responses is forwarded untranslated; the other two are translated in both directions
    by the handlers later slices register. The enum is the closed set -- an entry cannot
    name a fourth, and a build with no handler for one of these three refuses that entry
    rather than serving it down a shape it did not ask for.
    """

    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    CHAT_COMPLETIONS = "chat_completions"


class AuthScheme(StrEnum):
    """How an upstream expects its credential presented on the wire.

    Two forms, because two are what the upstreams in this plan document: a
    Responses-compatible upstream takes `Authorization: Bearer`, and an
    Anthropic-shaped deployment takes either that or a raw key in a header.
    A third form would be a guess about a provider nobody has configured.

    The header `API_KEY` renders is `x-api-key` and not `api-key`, and the difference is
    measured rather than chosen. Against the Foundry-hosted Anthropic route,
    `Authorization: Bearer` and `x-api-key` both authenticate and `api-key` -- Azure's
    own convention elsewhere on the same host -- answers **401** whose body reads
    "invalid subscription key", a message naming the wrong cause. `x-api-key` is also
    what first-party `api.anthropic.com` takes, so one header serves both and neither
    depends on which host the base URL names.
    """

    BEARER = "bearer"
    API_KEY = "api_key"

    def header_name(self) -> str:
        return "authorization" if self is AuthScheme.BEARER else "x-api-key"

    def header_value(self, secret: str) -> str:
        return f"Bearer {secret}" if self is AuthScheme.BEARER else secret


@dataclass(frozen=True, slots=True)
class RoutingEntry:
    """One model, and everything needed to serve it.

    `base_url` is a prefix, not a URL: each outward shape appends its own path to it,
    the way the Agent Runtime appends `responses` to the base URL it was given.
    `credential_name` is a vault entry's name and never its contents -- this record is
    configuration, and configuration is readable by anything that can read a ConfigMap.
    """

    model: str
    wire: UpstreamWire
    base_url: str
    auth_scheme: AuthScheme
    credential_name: str
    query_params: tuple[tuple[str, str], ...] = ()


class UnroutableModel(Exception):
    """No Routing Entry names this model, so nothing here can serve it."""

    def __init__(self, model: str) -> None:
        super().__init__(f"no routing entry for model {model!r}")
        self.model = model


class RoutingTable:
    """The declared model-to-upstream map, fixed for the process's life."""

    def __init__(self, entries: Sequence[RoutingEntry]) -> None:
        by_model: dict[str, RoutingEntry] = {}
        for entry in entries:
            if entry.model in by_model:
                raise ValueError(f"model {entry.model!r} has two routing entries")
            by_model[entry.model] = entry
        self._by_model: Mapping[str, RoutingEntry] = MappingProxyType(by_model)

    def entry_for(self, model: str) -> RoutingEntry:
        """The entry declared for this model. Never a default, never a nearest match."""
        entry = self._by_model.get(model)
        if entry is None:
            raise UnroutableModel(model)
        return entry

    def declared_models(self) -> frozenset[str]:
        return frozenset(self._by_model)


class _EntryDocument(BaseModel):
    """One entry in the shape configuration writes it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    wire: UpstreamWire
    base_url: str = Field(pattern=r"^https://")
    auth_scheme: AuthScheme
    credential_name: str = Field(min_length=1)
    query_params: dict[str, str] = Field(default_factory=dict)


class _TableDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: list[_EntryDocument] = Field(min_length=1)


def routing_table_from_json(document: bytes) -> RoutingTable:
    """Parse the configured table, or raise before the process serves anything.

    `base_url` must be https: the credential attached on the way out is the platform's
    own, and a plaintext hop would put it on the wire. Query parameters are an ordinary
    mapping because Azure's `api-version` is an ordinary query parameter to this runtime
    rather than a special case -- they are sorted into a tuple so two documents that
    differ only in key order produce the same entry.
    """
    parsed = _TableDocument.model_validate_json(document)
    return RoutingTable(
        [
            RoutingEntry(
                model=entry.model,
                wire=entry.wire,
                base_url=entry.base_url,
                auth_scheme=entry.auth_scheme,
                credential_name=entry.credential_name,
                query_params=tuple(sorted(entry.query_params.items())),
            )
            for entry in parsed.entries
        ]
    )


@dataclass(frozen=True, slots=True)
class CallerSession:
    """Which Session a request came from, established by its token and nothing else."""

    session_id: SessionId
    tenant_id: TenantId


@dataclass(frozen=True, slots=True)
class SessionTokenVerifier:
    """Reads a Session's identity off the token its pod was started with.

    This service never mints one. A Session token is signed by the control plane when
    the pod is placed, written into the compiled configuration the pod mounts, and read
    back here and at the Tool Gateway -- so one token, one layout and one key
    serve every hop a Session makes, and a Session carries a single credential
    rather than one per service it may reach.

    That is a correction rather than a preference. This class replaces a verifier of a
    *third* token layout -- a base64url payload and an HMAC over it, keyed from an entry
    in AWS Secrets Manager -- which nothing in this tree ever minted. Its own docstring
    said so, and said the answer was to stop using it: with no minter, `resolve`
    answered None for every request a real pod made and the Model Gateway answered
    401 to all of them. Two facts made that layout hard to fix in place rather than
    replace. The entry it read did not exist in the account, and this pod's IAM role
    could not read that prefix even had it existed, so the failure was a 503 on the
    first request and an `AccessDenied` in the log. Both dissolve here: the key
    arrives in the environment from the same Kubernetes Secret the control plane
    mints with and the Tool Gateway verifies with, which is where two of the three
    holders already kept it.

    Why an environment variable, in a service whose manifest otherwise refuses to
    name a Secret: the thing that manifest protects is the *upstream provider
    credential*, which this service exists to hold and fetches per TTL so that it
    expires. This is a different kind of value. It is a symmetric HMAC key shared
    with its minter, it proves only which Session is calling, and it is already an
    environment variable in the control plane and the Tool Gateway. Reading it from
    a fourth place would not make it less exposed; it would make this the one holder
    that disagrees, which is how the unminted layout came to exist. ADR-023.

    Holds the key for the process's life with no cache and no expiry window,
    because there is nothing to re-read: the key is a pod field fixed at admission,
    and a rotation replaces the pod. That removes the vault round trip, its TTL and
    the 503 the old verifier had to raise when the vault would not answer -- a
    failure mode this one cannot have.
    """

    key: bytes
    clock: Clock

    async def resolve(self, token: str) -> CallerSession | None:
        """The Session this token names, or None if it names none it can prove.

        Async because the inbound route awaits it and because the layout this
        replaces had to reach a vault. Nothing here blocks.

        Every rejection answers None rather than saying which check failed,
        matching `verify_session_token`, which raises one exception with no
        distinguishing detail for the same reason: a caller who learns *which* part
        was wrong learns whether a Session id exists.
        """
        try:
            context = verify_session_token(
                token, self.key, self.clock.now_epoch_ms() // 1000
            )
        except InvalidSessionToken:
            return None
        return CallerSession(session_id=context.session_id, tenant_id=context.tenant_id)


@dataclass(frozen=True, slots=True, repr=False)
class InboundTurn:
    """One model request as it arrived from a Session's pod.

    The headers are octets, for the same reason `UpstreamResponse`'s are: a header block
    carries no charset, and this record exists so a wire handler holds what arrived
    rather than what some codec made of it. The two directions disagreeing was the bug.
    The response direction was made byte-faithful and this one was left as text, so a
    pod-chosen octet above 0x7F reached an HTTP client that encodes a text header value
    as ASCII, raised there, and 500'd the Turn from outside the refusal envelope -- on
    the side this platform trusts least. Every octet in here is already one a header
    field may carry; `_inbound_headers` is where that is established.

    Redacted under repr and str, like the two records in `credential_broker.py` and for
    the same reason. This one holds the pod's own bearer token, whatever credential
    header the pod attached, and the request body -- which is the tenant's conversation.
    A dataclass's default repr puts all three into the next line somebody formats it
    into, and the line that does it will be an `_LOG.error("... %s", turn)` added by a
    slice that never read this file. So the repr carries the caller, the model, which
    header names arrived and how many body bytes there were, and none of their contents.
    """

    caller: CallerSession
    model: str
    headers: tuple[tuple[bytes, bytes], ...]
    body: bytes

    def __repr__(self) -> str:
        names = ",".join(name.decode("ascii", "replace") for name, _ in self.headers)
        return (
            f"InboundTurn(caller={self.caller!r}, model={self.model!r}, "
            f"header_names=[{names}], header_values=<redacted>, "
            f"body=<redacted, {len(self.body)} bytes>)"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True, slots=True)
class UpstreamResponse:
    """What an upstream answered, on its way back to the Agent Runtime.

    The headers are octets rather than text, and that is the whole reason this field is
    typed the way it is. A header block carries no charset, so an HTTP client that hands
    back strings picked one by sniffing the whole block at once -- and a single value
    above U+00FF made every value in that block decode as UTF-8, after which re-encoding
    them to relay raised on the one that arrived intact. Carrying the octets means
    nothing between the two sockets has to guess, and a field typed `bytes` cannot be
    read back through a codec somebody chose later.
    """

    status: int
    headers: tuple[tuple[bytes, bytes], ...]
    body: AsyncIterator[bytes]


class WireHandler(Protocol):
    """Serves one outward request shape."""

    def open(
        self, turn: InboundTurn, entry: RoutingEntry
    ) -> AbstractAsyncContextManager[UpstreamResponse]: ...


@dataclass(frozen=True)
class ModelGateway:
    """Everything the inbound surface needs, already wired.

    A shape with no handler is a missing key rather than a handler that raises, so a
    build that serves two shapes cannot silently serve a third by falling through to the
    one it has.
    """

    table: RoutingTable
    handlers: Mapping[UpstreamWire, WireHandler]
    tokens: SessionTokenVerifier


class GatewayRefusal(Exception):
    """A request this service will not serve, with the status the caller should see."""

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.message = message


def _refusal(exc: GatewayRefusal) -> JSONResponse:
    """Refuse in the envelope the Agent Runtime already parses.

    The closed, tenant-facing error set belongs to a different surface: this response
    goes to the Agent Runtime inside a pod, which turns a non-2xx into a failed request
    and reads the body's `error.message` as human text -- it scrapes a retry delay out
    of that text for a rate-limit error. Writing the refusal in the shape it already
    reads is what makes the failure land as a failed Turn instead of an unparsed body.

    The message names the model, which is the tenant's own word, and never the upstream,
    the shape or the base URL. Which upstream refused is operator-visible in the log
    line beside this and caller-visible nowhere.
    """
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"type": exc.kind, "message": exc.message}},
    )


def _bearer_token(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise GatewayRefusal(401, "unauthenticated", "a bearer token is required")
    return token


MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
"""The largest request body this service will hold in memory for one Turn.

Starlette imposes no cap of its own, so `Request.body()` buffers whatever arrives.
Without a bound, anything able to reach this Service could hand the process a
multi-gigabyte body and have it resident; with a bound, the pod's memory request is a
number derived from it rather than a hope. A Responses body is conversation state and
tool output, orders of magnitude under this, and a body over it is a mistake or an
attack either way.
"""


async def _bounded_body(request: Request) -> bytes:
    """The request body, refused past the cap and never decoded.

    The Agent Runtime may compress a request body. Nothing here decompresses one, and
    reading a compressed body as JSON would find no model and refuse it as malformed --
    which is the quiet version of the same failure. Naming the coding is the loud one,
    and it happens before any of the body is read.

    Read chunk by chunk rather than through `Request.body()` so the cap is enforced as
    the bytes arrive. `content-length` is not consulted: a caller writes that header, so
    trusting it would put the cap under the control of whoever the cap is for.
    """
    coding = request.headers.get("content-encoding", "identity").strip().lower()
    if coding not in ("", "identity"):
        raise GatewayRefusal(
            415, "invalid_request", f"content-encoding {coding} is not served here"
        )
    chunks: list[bytes] = []
    read = 0
    async for chunk in request.stream():
        read += len(chunk)
        if read > MAX_REQUEST_BODY_BYTES:
            raise GatewayRefusal(
                413,
                "invalid_request",
                f"the request body is larger than {MAX_REQUEST_BODY_BYTES} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


_ILLEGAL_FIELD_OCTETS = (
    frozenset(range(0x00, 0x09)) | frozenset(range(0x0A, 0x20)) | {0x7F}
)
"""The octets no HTTP header field may carry, as the complement of the ones it may.

RFC 9110 §5.5 allows SP, HTAB, `%x21-7E` and obs-text `%x80-FF` in a field value, which
leaves the C0 controls other than HTAB, plus DEL. Named as what is forbidden rather than
what is allowed because the forbidden set is the one with a consequence: CR and LF end a
header, so a value carrying either is two headers to whoever parses it next.
"""


def _inbound_headers(request: Request) -> tuple[tuple[bytes, bytes], ...]:
    """This request's header block as the octets that arrived, or a refusal.

    Octets, because a header block has no charset and this service relays rather than
    interprets. The bytes are what the ASGI server put in the scope, so nothing between
    the two sockets picks a codec -- and a value a pod chose cannot be one this service
    is unable to hand onward, which is what an earlier version turned into a 500 on
    every request carrying an octet above 0x7F in any forwarded header.

    Two octets are refused rather than relayed, and the refusal is the point. `httpx`
    does not validate a `bytes` header value at all -- measured: it builds a request
    with `b"a\\r\\nx-injected: yes"` in it without complaint -- so relaying octets
    faithfully means CR and LF would be relayed faithfully too, and a value carrying
    either is a second header on the upstream hop. Today no conformant parser puts one
    in the scope, `h11` included, so this refuses nothing that can currently arrive; it
    is here because the alternative is a filter whose safety is a property of the server
    in front of it rather than of itself.

    Checked over the whole block and not only the headers that travel. A name the
    allowlist drops is dropped either way, so refusing the request costs nothing real,
    and a rule that applies to every header is one a later wire handler cannot forget.
    """
    for name, value in request.headers.raw:
        for field in (name, value):
            illegal = next((b for b in field if b in _ILLEGAL_FIELD_OCTETS), None)
            if illegal is not None:
                raise GatewayRefusal(
                    400,
                    "invalid_request",
                    f"header {name.decode('ascii', 'replace')} carries the octet "
                    f"0x{illegal:02x}, which no header field may carry",
                )
    return tuple(request.headers.raw)


def _one_value_per_name(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, refusing one that names any member twice.

    RFC 8259 leaves duplicate member names unspecified, and parsers split on it: this
    one keeps the last, others keep the first. That is ordinarily somebody else's
    problem, but here the body picks which credential is attached -- a body reading
    `{"model": "a", ..., "model": "b"}` routes on `b`, is forwarded byte-identical, and
    an upstream taking the first serves `a`'s request on `b`'s credential while the
    meter reading usage on the way back records `b`.

    Refused rather than resolved, at any depth. A body whose meaning depends on which
    parser reads it has no single meaning to relay, and this service does not guess
    where a guess would be indistinguishable afterwards from a correct answer (ADR-016).
    """
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError("a JSON object names a member twice")
    return dict(pairs)


def _requested_model(body: bytes) -> str:
    """The `model` field, read without rewriting anything.

    The parse is discarded: what goes upstream is the bytes that arrived, so reading one
    field cannot cost another. A model name is a routing key and nothing more, which is
    why the value is not checked against anything here beyond being a non-empty string
    (ADR-010).
    """
    try:
        document = json.loads(body, object_pairs_hook=_one_value_per_name)
    except json.JSONDecodeError as exc:
        raise GatewayRefusal(
            400, "invalid_request", "the request body is not JSON"
        ) from exc
    except ValueError as exc:
        # `_one_value_per_name`'s refusal. A separate clause because the message above
        # would blame the syntax for a body that parses perfectly well and means two
        # things.
        raise GatewayRefusal(
            400, "invalid_request", "the request body names a field twice"
        ) from exc
    if not isinstance(document, dict):
        raise GatewayRefusal(
            400, "invalid_request", "the request body is not an object"
        )
    model = document.get("model")
    if not isinstance(model, str) or not model:
        raise GatewayRefusal(400, "invalid_request", "the request body names no model")
    return model


async def _relayed(
    body: AsyncIterator[bytes], stack: AsyncExitStack
) -> AsyncIterator[bytes]:
    """The upstream's chunks, closing the exchange when the stream ends however it ends.

    The stream failing is not an edge case, it is what the read timeout exists for: an
    SSE response that goes quiet trips `MAP_UPSTREAM_READ_TIMEOUT_S` mid-body, and a
    provider or a load balancer dropping a response gives `RemoteProtocolError` in the
    same place. Both raise out of `aiter_raw` *after* the route returned, which is past
    every `try` the route has -- and a `BackgroundTask` is run only for a response whose
    stream completed, so before this wrapper existed neither fault closed anything. The
    handler's own `finally` never ran, and for the `WireHandler` seam that is the whole
    contract: the meter slice reads consumption on the way back, and its cleanup is
    where a stream that ended without reporting any would be noticed.

    This is the same cleanup the route's own `except` closes and the same one the
    background task closes, not a second one. `AsyncExitStack.aclose` is idempotent, so
    the ordinary path -- stream exhausted here, then the background task -- closes once
    and the second call does nothing. Two call sites are the minimum: a generator
    suspended at a `yield` does not run its `finally` when the task reading it is
    cancelled, so the caller-hangs-up path still needs the background task, and a
    stream that raises still needs this.
    """
    try:
        async for chunk in body:
            yield chunk
    finally:
        await stack.aclose()


router = APIRouter(tags=["upstream"])


@router.post("/responses")
async def responses(request: Request) -> Response:
    """Relay one model request, having established whose Turn it is.

    The order is deliberate. Identity is settled first, and settled from the token
    alone, so an unauthenticated caller never gets a byte of its body into this
    process's memory -- an earlier version buffered the whole body on the line above the
    token check, which made "the body is not trusted until identity is settled" true of
    every use of the body except the one that costs memory. Then a coding this service
    cannot read is named before anything looks at the bytes, the read is bounded, and
    the shape is picked from the declared table rather than from the request. Nothing
    between here and the upstream reads a second field of the body.

    The exit stack outlives this function on purpose. `handler.open` holds a live
    connection, and the response body is still being read after this returns -- so one
    cleanup is reached from the three places this relay can end: a failure before the
    response exists closes it on the way out, the stream ending or failing closes it in
    `_relayed`, and the caller hanging up closes it in the background task. It is one
    stack and `aclose` is idempotent, so whichever gets there first is the one that
    closes and the others are free.
    """
    gateway: ModelGateway = request.app.state.gateway
    stack = AsyncExitStack()
    try:
        try:
            caller = await gateway.tokens.resolve(_bearer_token(request))
            if caller is None:
                raise GatewayRefusal(
                    401, "unauthenticated", "the token names no live Session"
                )
            body = await _bounded_body(request)
            model = _requested_model(body)
            try:
                entry = gateway.table.entry_for(model)
            except UnroutableModel as exc:
                raise GatewayRefusal(
                    404, "invalid_request", f"model {exc.model} is not configured"
                ) from exc
            handler = gateway.handlers.get(entry.wire)
            if handler is None:
                _LOG.error(
                    "no handler for wire %s serving model %s at %s",
                    entry.wire.value,
                    entry.model,
                    entry.base_url,
                )
                raise GatewayRefusal(
                    502, "server_error", f"model {model} cannot be served right now"
                )
            turn = InboundTurn(
                caller=caller,
                model=model,
                headers=_inbound_headers(request),
                body=body,
            )
            upstream = await stack.enter_async_context(handler.open(turn, entry))
            response = StreamingResponse(
                _relayed(upstream.body, stack),
                status_code=upstream.status,
                background=BackgroundTask(stack.aclose),
            )
            # `raw_headers` and not the `headers=` argument, which takes a mapping
            # and so keeps one value per name. The handler went to the trouble of
            # preserving the upstream's duplicates -- two `set-cookie` headers are
            # two headers -- and a mapping here threw all but the last away, which
            # made that care pointless and silently dropped one of every pair.
            # Assigned wholesale rather than appended: nothing StreamingResponse put
            # there is the upstream's, and the upstream's answer is what this relays.
            # The values are already the upstream's own octets, so nothing is encoded
            # here; only the name is lowercased, and a header name is an ASCII token
            # by definition.
            response.raw_headers = [
                (name.lower(), value) for name, value in upstream.headers
            ]
        except BaseException:
            # From `enter_async_context` onward the upstream exchange is open, and
            # until Starlette holds the response above, nothing else will close it: the
            # `BackgroundTask` runs only for a response that was returned, and
            # `_relayed` only for a stream that was started. Closed here and not in the
            # refusal branch below, so that every way out of this route passes one
            # cleanup rather than each way out carrying its own -- and closed rather
            # than left to the argument that nothing above can raise.
            await stack.aclose()
            raise
    except GatewayRefusal as exc:
        return _refusal(exc)
    return response


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness only. It names no model, no upstream and no Session."""
    return {"status": "ok"}


def create_model_gateway_app(gateway: ModelGateway) -> FastAPI:
    """The Model Gateway's ASGI app, taking an already-wired gateway.

    The `/v1` prefix is not taste. The Agent Runtime builds its request URL by
    concatenating the base URL its configuration names with the literal segment
    `responses`, so what this app serves has to be exactly the tail of the base URL an
    operator writes into that configuration. Everything the app needs is passed in, so a
    test drives the real route against fake handlers and the composition root stays the
    only place a concrete adapter is chosen.
    """
    app = FastAPI(title="Model Gateway", version="v1")
    app.state.gateway = gateway
    app.include_router(router, prefix="/v1")
    return app
