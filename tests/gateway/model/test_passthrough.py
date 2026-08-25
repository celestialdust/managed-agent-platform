"""The Responses shape forwarded as it arrived, graded byte for byte.

Tier 1, no infrastructure. `httpx.MockTransport` stands in for the upstream and records
the request the real `httpx.AsyncClient` built, so what is asserted below is the actual
outbound wire and not a mapping this test assembled. The broker is the real class
over a fake vault, because the credential swap is one of the properties under test
and a stubbed credential would prove the swap against itself.

The negative assertions are the load-bearing ones. A pass-through is defined by what it
does *not* do -- it does not drop a field it has no opinion about, does not honour
one, and does not carry the pod's own token outward -- and every one of those is
invisible unless a test looks for its absence.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

import httpx
import pytest

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.gateway.model.credential_broker import ProviderCredentialBroker
from managed_agent.gateway.model.passthrough import (
    _FORWARDED_NAMES as _ALLOWLIST,
)
from managed_agent.gateway.model.passthrough import (
    ResponsesPassthrough,
    upstream_url,
)
from managed_agent.gateway.model.passthrough import (
    _forwarded as _module_forwards,
)
from managed_agent.gateway.model.router import (
    AuthScheme,
    CallerSession,
    GatewayRefusal,
    InboundTurn,
    RoutingEntry,
    UpstreamWire,
)

_CALLER = CallerSession(
    session_id=SessionId(UUID("11111111-1111-4111-8111-111111111111")),
    tenant_id=TenantId(UUID("22222222-2222-4222-8222-222222222222")),
)
_BODY = (
    b'{"model":"gpt-5-codex","input":[],"store":false,'
    b'"prompt_cache_key":"abc","service_tier":"priority",'
    b'"include":["reasoning.encrypted_content"],'
    b'"client_metadata":{"session_id":"not-ours","thread_id":"nor-this"},'
    b'"a_field_nothing_here_reads":{"nested":[1,2,3]}}'
)

_POD_SUPPLIED_API_KEY = "POD-SUPPLIED-PROVIDER-KEY"
"""A provider key a Session pod attached itself, under the header this module uses to
present the *platform's* credential. The pod is not trusted, so this is what a hostile
or merely misconfigured runtime sends, and it is the value that must not appear on the
far side of this hop."""

_RUNTIME_HEADERS: tuple[tuple[str, str], ...] = (
    ("host", "model-gateway.map.svc"),
    ("authorization", "Bearer the-pods-own-token"),
    ("content-type", "application/json"),
    ("content-length", "999999"),
    ("connection", "close"),
    ("keep-alive", "timeout=5"),
    ("te", "trailers"),
    ("accept", "text/event-stream"),
    ("user-agent", "codex_cli_rs/0.1.0"),
    ("originator", "codex_cli_rs"),
    ("openai-beta", "responses=experimental"),
    ("x-codex-installation-id", "install-1"),
    ("x-codex-routing-hint", "hint-1"),
    ("x-codex-turn-state", "state-1"),
    ("x-codex-turn-metadata", "meta-1"),
    ("x-codex-parent-thread-id", "parent-1"),
    ("x-codex-window-id", "window-1"),
    ("x-codex-beta-features", "one,two"),
    ("session-id", "runtime-session"),
    ("thread-id", "runtime-thread"),
    ("x-client-request-id", "req-1"),
    ("x-oai-attestation", "attestation-blob"),
    ("x-openai-internal-codex-residency", "us"),
    ("x-openai-memgen-request", "true"),
    ("x-openai-subagent", "sub-1"),
    ("x-openai-internal-codex-responses-lite", "1"),
    ("x-responsesapi-include-timing-metrics", "1"),
    # The pod's own auth provider sends these two beside its bearer token. They select
    # an account and a routing plane for the *pod's* credential, which is not the one
    # that leaves this hop, so forwarding them would let a pod steer where the
    # platform's credential is spent.
    ("chatgpt-account-id", "pods-own-account"),
    ("x-openai-fedramp", "true"),
    # Everything below is what a pod can attach that this service must not relay. The
    # first is the load-bearing one: it is the header this module presents the
    # platform's own credential under, and forwarding it let a Turn be served on the
    # pod's key over this service's egress path while the meter on the way back
    # attributed it to the platform.
    ("x-api-key", _POD_SUPPLIED_API_KEY),
    ("cookie", "session=pods-own"),
    ("proxy-authorization", "Basic cG9kOnBvZA=="),
    ("x-forwarded-for", "10.0.0.9"),
    ("x-header-nobody-named", "arbitrary"),
)


_TRAVELS = frozenset(
    {
        "content-type",
        "accept",
        "user-agent",
        "originator",
        "openai-beta",
        "x-codex-installation-id",
        "x-codex-routing-hint",
        "x-codex-turn-state",
        "x-codex-turn-metadata",
        "x-codex-parent-thread-id",
        "x-codex-window-id",
        "x-codex-beta-features",
        "session-id",
        "thread-id",
        "x-client-request-id",
        "x-oai-attestation",
        "x-openai-internal-codex-residency",
        "x-openai-memgen-request",
        "x-openai-subagent",
        "x-openai-internal-codex-responses-lite",
        "x-responsesapi-include-timing-metrics",
    }
)
"""Of the inbound headers above, the ones that reach the upstream.

Every header `_RUNTIME_SETS_THESE` names is here, which is the transparency claim this
filter has to keep: an allowlist that dropped `x-client-request-id` would break the
runtime's request correlation quietly, and that is the failure mode an allowlist has
instead of the denylist's.

Spelled out here rather than imported from the module under test, so a name quietly
added to the implementation's allowlist does not also appear in what this file expects
to travel. That direction is the one that matters now: under a denylist the risk was a
header silently disappearing, and under an allowlist it is a header silently arriving.

Everything in `_RUNTIME_HEADERS` and not here must not cross this hop, and the test
below asserts that as a set difference rather than a second list -- so a header added to
the fixture with no decision about it fails rather than going ungraded.
"""

_RUNTIME_SETS_THESE: dict[str, str] = {
    # From the runtime's own header-name constants, not from prose about them. The
    # previous version of the allowlist was transcribed from
    # `research/model-wire-surface.md`'s twelve-name summary, and that summary omits
    # two of these -- so the fixture built from the same summary could not see the
    # omission and agreed with the bug. The right of each pair is where the runtime
    # sets it, under `.reference/codex/codex-rs/`.
    "originator": "login/src/auth/default_client.rs:335, every request",
    "user-agent": "login/src/auth/default_client.rs:336-338",
    "x-openai-internal-codex-residency": "login/src/auth/default_client.rs:341-349",
    "openai-beta": "core/src/client.rs:143",
    "x-codex-installation-id": "core/src/client.rs:144",
    "x-codex-routing-hint": "core/src/client.rs:145",
    "x-codex-turn-state": "core/src/client.rs:146",
    "x-codex-turn-metadata": "core/src/client.rs:147",
    "x-codex-parent-thread-id": "core/src/client.rs:148",
    "x-codex-window-id": "core/src/client.rs:149",
    "x-codex-beta-features": "core/src/client.rs:1977",
    "x-openai-memgen-request": "core/src/client.rs:150, inserted :769 and :786",
    "x-openai-subagent": "core/src/client.rs:151",
    "x-responsesapi-include-timing-metrics": "core/src/client.rs:152",
    "x-openai-internal-codex-responses-lite": "core/src/client.rs:159",
    "x-oai-attestation": "core/src/attestation.rs:7, inserted core/src/client.rs:628",
    "session-id": "codex-api/src/requests/headers.rs:7",
    "thread-id": "codex-api/src/requests/headers.rs:10",
    "x-client-request-id": "core/src/client.rs:1147",
    "accept": "codex-api/src/endpoint/responses.rs:149",
    "content-type": "set by the HTTP transport for a JSON body",
    "accept-encoding": "set by the HTTP transport; pinned by this hop",
    "authorization": "the pod's own credential; replaced, never forwarded",
    "chatgpt-account-id": "app-server-transport/.../remote_control/auth.rs:13",
    "x-openai-fedramp": "app-server-transport/.../remote_control/auth.rs:145",
}
"""Every request header the Agent Runtime can put on a Responses POST, with its source.

This is the enumeration the two lists above are graded against, and it exists because
they cannot grade each other. `_FORWARDED_NAMES` in the module and `_TRAVELS` here were
both transcribed from one prose bullet; when that bullet turned out to omit `originator`
-- which the runtime's client builder sets on every request it makes -- neither list
knew, and the fixture had no row for it, so it was ungraded in both directions at once.
Two copies of one wrong list agree perfectly.

So this one is transcribed from the runtime's own constants instead, produced by

    grep -rn 'const [A-Z_]*: &str' core/src/client.rs core/src/attestation.rs \
        login/src/auth/default_client.rs codex-api/src/requests/headers.rs

over `.reference/codex/codex-rs/`, plus the four an HTTP transport or an auth provider
adds rather than a named constant. A summary can omit a name; that grep cannot. It is
data and not an expectation -- what should happen to each of these is the two tests
below, and a name here with no decision recorded fails one of them.
"""


class _OneSecretVault:
    async def fetch(self, name: str) -> str:
        return f"secret-for-{name}"


class _FrozenClock:
    def now_epoch_ms(self) -> int:
        return 1_700_000_000_000


def _entry(
    *,
    base_url: str = "https://api.openai.com/v1",
    auth_scheme: AuthScheme = AuthScheme.BEARER,
    query_params: tuple[tuple[str, str], ...] = (),
) -> RoutingEntry:
    return RoutingEntry(
        model="gpt-5-codex",
        wire=UpstreamWire.RESPONSES,
        base_url=base_url,
        auth_scheme=auth_scheme,
        credential_name="map/upstream/openai",
        query_params=query_params,
    )


_RUNTIME_HEADER_OCTETS: tuple[tuple[bytes, bytes], ...] = tuple(
    (name.encode("ascii"), value.encode("ascii")) for name, value in _RUNTIME_HEADERS
)
"""The fixture above as what an inbound header block actually is.

Derived rather than spelled a second time: the readable version is about *which* headers
arrive and the filter's answer for each, and that reads worse with a `b` on every one of
seventy strings. A test whose subject is which octets travel spells its own bytes rather
than coming through here -- see the obs-text pair below."""


def _turn(
    *,
    body: bytes = _BODY,
    headers: tuple[tuple[bytes, bytes], ...] = _RUNTIME_HEADER_OCTETS,
) -> InboundTurn:
    return InboundTurn(caller=_CALLER, model="gpt-5-codex", headers=headers, body=body)


class _UpstreamBody(httpx.AsyncByteStream):
    """The upstream's body as a live stream that records having been closed.

    A stream rather than `content=bytes`, for two reasons. `content=bytes` hands back a
    response httpx already marks as consumed, so `aiter_raw` on it raises for a reason
    that exists only inside a test. And closure is the property this stands in for: a
    relayed upstream connection has to be released when the caller stops reading, and
    with a mock transport there is no socket to observe -- `closed` here is the only
    evidence that the relay's `finally` ran.
    """

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _Upstream:
    """A recording upstream: what it was asked, and what it hands back."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | list[tuple[bytes, bytes]] | None = None,
        chunks: tuple[bytes, ...] = (b"data: one\n\n", b"data: two\n\n"),
        raises: Exception | None = None,
    ) -> None:
        self.status = status
        self.headers: dict[str, str] | list[tuple[bytes, bytes]] = headers or {
            "content-type": "text/event-stream"
        }
        """Byte pairs where a test needs to choose the octets on the wire.

        A `dict[str, str]` here is httpx encoding the strings for the test, which is
        fine for an ASCII value and useless for the case where *which* octets arrive is
        the property under test."""
        self.raises = raises
        self.seen: list[httpx.Request] = []
        self.body = _UpstreamBody(chunks)

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.seen.append(request)
            request.read()
            if self.raises is not None:
                raise self.raises
            return httpx.Response(self.status, headers=self.headers, stream=self.body)

        return httpx.MockTransport(handle)

    def only(self) -> httpx.Request:
        assert len(self.seen) == 1, f"expected one outbound request, saw {self.seen}"
        return self.seen[0]


def _passthrough(
    upstream: _Upstream, client: httpx.AsyncClient
) -> ResponsesPassthrough:
    broker = ProviderCredentialBroker(_OneSecretVault(), _FrozenClock())
    return ResponsesPassthrough(client, broker)


async def _relay(
    upstream: _Upstream, entry: RoutingEntry, turn: InboundTurn | None = None
) -> tuple[int, tuple[tuple[bytes, bytes], ...], list[bytes]]:
    async with httpx.AsyncClient(transport=upstream.transport()) as client:
        handler = _passthrough(upstream, client)
        async with handler.open(turn or _turn(), entry) as response:
            chunks = [chunk async for chunk in response.body if chunk]
            return response.status, response.headers, chunks


# --- the URL ----------------------------------------------------------------------


def test_the_outward_url_is_the_base_url_plus_the_one_path_segment() -> None:
    assert upstream_url(_entry()) == "https://api.openai.com/v1/responses"
    assert (
        upstream_url(_entry(base_url="https://api.openai.com/v1/"))
        == "https://api.openai.com/v1/responses"
    )


def test_query_parameters_are_appended_once_and_percent_encoded() -> None:
    """This service is the client on this hop, so the encoding is its own to get right.

    The Agent Runtime concatenates the same shape without encoding anything; reproducing
    that would put a raw space or `&` from a ConfigMap straight onto the wire.
    """
    url = upstream_url(
        _entry(
            base_url="https://map.openai.azure.com/openai",
            query_params=(("api-version", "2025-04-01-preview"), ("x y", "a&b")),
        )
    )

    assert url == (
        "https://map.openai.azure.com/openai/responses"
        "?api-version=2025-04-01-preview&x+y=a%26b"
    )
    assert url.count("?") == 1


# --- the outbound request ---------------------------------------------------------


async def test_the_body_that_arrived_is_the_body_that_leaves() -> None:
    """Byte-identical, so a field nothing here understands cannot be lost or
    honoured."""
    upstream = _Upstream()

    await _relay(upstream, _entry())

    assert upstream.only().content == _BODY
    assert b"a_field_nothing_here_reads" in upstream.only().content
    assert b'"client_metadata"' in upstream.only().content


async def test_the_url_the_client_built_is_the_url_the_entry_declares() -> None:
    upstream = _Upstream()
    entry = _entry(query_params=(("api-version", "2025-04-01-preview"),))

    await _relay(upstream, entry)

    assert str(upstream.only().url) == upstream_url(entry)
    assert upstream.only().method == "POST"


async def test_the_pods_own_token_is_dropped_and_the_upstreams_is_attached() -> None:
    """The one header that changes, and the whole reason the credential stays here."""
    upstream = _Upstream()

    await _relay(upstream, _entry())

    sent = upstream.only().headers
    assert sent["authorization"] == "Bearer secret-for-map/upstream/openai"
    assert "the-pods-own-token" not in sent["authorization"]


async def test_an_api_key_upstream_gets_its_key_header_and_no_authorization() -> None:
    """`x-api-key`, measured: the Foundry Anthropic route 401s on Azure's `api-key`."""
    upstream = _Upstream()

    await _relay(upstream, _entry(auth_scheme=AuthScheme.API_KEY))

    sent = upstream.only().headers
    assert sent["x-api-key"] == "secret-for-map/upstream/openai"
    assert "authorization" not in sent


async def test_every_runtime_header_this_service_has_no_opinion_about_travels() -> None:
    """Including the whole `x-codex-*` family a provider is required to tolerate."""
    upstream = _Upstream()

    await _relay(upstream, _entry())

    sent = upstream.only().headers
    for name, value in _RUNTIME_HEADERS:
        if name not in _TRAVELS:
            continue
        assert sent[name] == value, name
    assert [name for name in sent if name.startswith("x-codex-")] == [
        "x-codex-installation-id",
        "x-codex-routing-hint",
        "x-codex-turn-state",
        "x-codex-turn-metadata",
        "x-codex-parent-thread-id",
        "x-codex-window-id",
        "x-codex-beta-features",
    ]


_DELIBERATELY_REFUSED = frozenset(
    {
        # Replaced, never forwarded: the platform's credential is what opens the far
        # side of this hop, and the pod's own is what must not reach it.
        "authorization",
        # The pod's own bearer-auth provider sends these two. They select an account
        # and a routing plane for the *pod's* credential, so forwarding them would let
        # a pod steer where the platform's credential is spent.
        "chatgpt-account-id",
        "x-openai-fedramp",
    }
)
"""The names in `_RUNTIME_SETS_THESE` this hop refuses on purpose, each with why.

Named here rather than inferred, so refusing one is a decision somebody wrote down and
the test below can tell a decision from an omission. That is the distinction the round
before this one had no way to make."""


def test_every_header_the_runtime_sends_is_forwarded_or_named_as_refused() -> None:
    """The guard the enumeration exists for: no runtime header goes undecided.

    Graded against `_RUNTIME_SETS_THESE`, which is transcribed from the runtime's own
    constants, rather than against `_TRAVELS`, which was transcribed from the same prose
    as the allowlist. That is the whole point -- the four headers this filter shipped
    without were invisible to a fixture built from the list they were missing from, so a
    complement test over that fixture was grading its own blind spot.

    Asserted through the module's own predicate, so a name covered by the `x-codex-`
    prefix counts as forwarded without this test having to know which mechanism let it
    through.
    """
    undecided = {
        name: source
        for name, source in _RUNTIME_SETS_THESE.items()
        if not _module_forwards(name.encode("ascii"))
        and name not in _DELIBERATELY_REFUSED
    }

    assert not undecided, (
        f"the runtime sends these and this hop neither forwards them nor names them as "
        f"refused: {undecided}"
    )


def test_no_name_in_the_allowlist_is_one_the_runtime_was_never_seen_to_send() -> None:
    """The other direction, and the one an allowlist needs most.

    A denylist's risk was a header silently disappearing; an allowlist's is a header
    silently arriving, and the arrival that mattered was `x-api-key` -- the name this
    module presents the platform's own credential under. Grading the allowlist as a
    subset of the enumeration means a name can only be added to it by first writing down
    where the runtime sets it, which is the step that was skipped.
    """
    unsourced = _ALLOWLIST - _RUNTIME_SETS_THESE.keys()

    assert not unsourced, (
        f"these names travel and nothing records the runtime sending them: "
        f"{sorted(unsourced)}"
    )


async def test_a_pod_supplied_api_key_never_reaches_the_upstream() -> None:
    """The header this module attaches the platform's own credential under.

    A pod that sends its own provider key as `x-api-key` had it forwarded verbatim
    *alongside* the platform's `Authorization`. On an upstream that prefers `x-api-key`
    -- and the Foundry Anthropic route this platform targets accepts both forms, which
    is why an operator would ever have two -- the Turn is then served on the pod's
    credential over this service's egress path, while the meter reading usage off the
    way back records it against the platform.

    Asserted on the value and not only the name, because the platform's own credential
    arrives under this same name when the entry is `api_key`: a test that asserted
    absence would fail for the wrong reason on half the routing table.
    """
    upstream = _Upstream()

    await _relay(upstream, _entry())

    sent = upstream.only().headers
    assert "x-api-key" not in sent
    assert _POD_SUPPLIED_API_KEY not in _every_value(sent)


async def test_the_platform_key_replaces_a_pod_supplied_one_of_the_same_name() -> None:
    """The `api_key` half of the case above: one `x-api-key` goes out, and it is
    ours."""
    upstream = _Upstream()

    await _relay(upstream, _entry(auth_scheme=AuthScheme.API_KEY))

    sent = upstream.only().headers
    assert sent["x-api-key"] == "secret-for-map/upstream/openai"
    assert _POD_SUPPLIED_API_KEY not in _every_value(sent)
    assert len(sent.get_list("x-api-key")) == 1, "two keys, and the upstream picks"


_REWRITTEN_BY_THIS_HOP = frozenset(
    {"host", "authorization", "content-length", "connection"}
)
"""Names present outbound whatever arrived, because this hop sets them: `host` from the
URL, `authorization` from the routing entry's credential, and `content-length` and
`connection` by httpx as it builds the request. Their *values* are graded elsewhere in
this file; the complement test below can only grade names, so it steps around these
four."""


def _every_value(headers: httpx.Headers) -> str:
    """Every outbound header value joined, for asserting a value appears nowhere.

    Not `str(headers)`: httpx's own repr replaces the value of anything it considers
    sensitive with `[secure]`, so a leaked credential would be invisible to a
    substring search over it -- the test would pass because the evidence was redacted.
    """
    return "\n".join(f"{name}: {value}" for name, value in headers.multi_items())


async def test_no_header_this_service_did_not_name_crosses_the_hop() -> None:
    """Default deny, asserted as the complement of the allowlist over the whole fixture.

    This is the property a denylist cannot have. A denylist forwards every name nobody
    thought of, so its coverage is a list of past mistakes; the failure that got through
    here was `x-api-key`, and `cookie`, `proxy-authorization` and
    `x-header-nobody-named` were never considered at all. Grading the complement means a
    fixture with no decision about it fails this test rather than travelling ungraded.
    """
    upstream = _Upstream()

    await _relay(upstream, _entry())

    sent = upstream.only().headers
    for name, _ in _RUNTIME_HEADERS:
        if name in _TRAVELS or name in _REWRITTEN_BY_THIS_HOP:
            continue
        assert name not in sent, (
            f"{name} reached the upstream; it is in neither the allowlist nor the "
            "set of headers this hop rewrites"
        )


async def test_the_headers_that_describe_the_inbound_hop_do_not_cross_to_this_one() -> (
    None
):
    """`host` is this hop's, `content-length` is recomputed, the rest do not travel.

    Forwarding an inbound `content-length` past a body about to be reframed is how a
    stream gets truncated, and forwarding the inbound `host` would send the
    gateway's own name to a provider that routes on it.

    `connection` is asserted by value rather than by absence, because httpx sets its own
    on every request it builds: the header is present either way, and the only
    thing worth knowing is whether the pod's `close` reached this hop and tore
    the connection down after one Turn.
    """
    upstream = _Upstream()

    await _relay(upstream, _entry())

    sent = upstream.only().headers
    assert sent["host"] == "api.openai.com"
    assert sent["content-length"] == str(len(_BODY))
    assert sent["connection"] != "close"
    assert "keep-alive" not in sent
    assert "te" not in sent


@pytest.mark.parametrize(
    ("asked", "expected"),
    [(None, "identity"), ("gzip", "gzip"), ("identity", "identity")],
)
async def test_the_content_coding_is_the_runtimes_own_or_identity(
    asked: str | None, expected: str
) -> None:
    """The response body is relayed undecoded, so a coding negotiated on the runtime's
    behalf would arrive declared and undecodable."""
    upstream = _Upstream()
    headers = tuple(
        (name, value)
        for name, value in _RUNTIME_HEADER_OCTETS
        if name != b"accept-encoding"
    )
    if asked is not None:
        headers = (*headers, (b"accept-encoding", asked.encode("ascii")))

    await _relay(upstream, _entry(), _turn(headers=headers))

    assert upstream.only().headers["accept-encoding"] == expected


# --- the response ----------------------------------------------------------------


async def test_the_response_chunks_come_back_in_order_and_unchanged() -> None:
    """The chunk boundaries, not only the bytes their concatenation adds up to.

    This body is `text/event-stream`, and the Agent Runtime acts on each event as it
    arrives. A relay that coalesced three chunks into one, or split one into bytes,
    would join to the same string while changing when the runtime sees anything --
    so asserting only the join would let a re-chunking relay pass.
    """
    upstream = _Upstream(chunks=(b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"))

    status, _, chunks = await _relay(upstream, _entry())

    assert status == 200
    assert chunks == [b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"]


async def test_an_upstreams_own_refusal_is_relayed_rather_than_rewritten() -> None:
    """Reading it and rewriting it would make this service the author of the message,
    and the slice that reads a refusal for a missing capability reads it out of that
    body."""
    upstream = _Upstream(
        status=429,
        headers={"content-type": "application/json", "retry-after": "3"},
        chunks=(b'{"error":{"message":"slow down, try again in 3s"}}',),
    )

    status, headers, chunks = await _relay(upstream, _entry())

    assert status == 429
    assert b"".join(chunks) == b'{"error":{"message":"slow down, try again in 3s"}}'
    assert (b"retry-after", b"3") in headers


async def test_the_response_headers_that_end_at_the_upstream_hop_are_not_relayed() -> (
    None
):
    upstream = _Upstream(
        headers={
            "content-type": "text/event-stream",
            "connection": "keep-alive",
            "x-codex-primary-used-percent": "12",
        }
    )

    _, headers, _ = await _relay(upstream, _entry())

    names = {name for name, _ in headers}
    assert b"content-type" in names
    assert b"x-codex-primary-used-percent" in names
    assert b"connection" not in names
    assert b"content-length" not in names


async def test_a_response_header_above_the_ascii_range_keeps_its_own_octets() -> None:
    """The octets the upstream chose, not a re-encoding of somebody's guess at them.

    A header block carries no charset, so an HTTP client handing back strings has to
    pick one by sniffing -- and httpx sniffs once for the whole block. A single value
    above U+00FF therefore changed how *every* value in that block was read, and
    re-encoding them as latin-1 to relay then raised on the one that had arrived intact:
    measured, the caller got 500 and the upstream connection was never closed, because
    the raise landed after the exchange was opened and before the response that owns
    closing it existed. So the relay carries bytes and decodes nothing.

    Two values, deliberately. The non-ASCII one is what the fix is about; the plain
    `content-type` beside it is what proves the claim about the block -- under the old
    decode it was the presence of the first that changed the reading of the second.
    """
    detail = b"caf\xc3\xa9-\xe4\xb8\xad\xe6\x96\x87"
    upstream = _Upstream(
        headers=[
            (b"content-type", b"text/event-stream"),
            (b"x-upstream-detail", detail),
        ]
    )

    _, headers, _ = await _relay(upstream, _entry())

    assert (b"x-upstream-detail", detail) in headers
    assert (b"content-type", b"text/event-stream") in headers


async def test_a_request_header_above_the_ascii_range_keeps_its_own_octets() -> None:
    """The other direction of the same property, and the one that was left crashing.

    Placed beside the response-direction test above on purpose. The bug was not that
    either direction had the wrong idea about encodings -- it was that one direction was
    fixed and the other was not, and the one left as text was the one whose author is a
    Session pod. `httpx` encodes a *text* header value as ASCII
    (`_normalize_header_value`), so one pod-chosen octet in `0x80-0xFF` in any forwarded
    header raised `UnicodeEncodeError` inside `build_request`: measured through the real
    route, `user-agent: codex-caf\xe9` gave a 500 carrying the server's plain-text body
    rather than the `error.message` envelope the Agent Runtime parses, plus a traceback
    per request -- while the same octet in a header the allowlist *drops* gave a clean
    200. The crash was caused by forwarding it.

    obs-text is what these octets are (RFC 9110 §5.5), and a relay has no business
    having an opinion about them: the allowlist already decided this header may
    travel, and past that decision the value is the runtime's own vocabulary.

    Two headers again, for the same reason as above: the plain one proves nothing
    else in the block was disturbed by the presence of the odd one.
    """
    upstream = _Upstream()
    detail = "caf\u00e9".encode()
    turn = _turn(
        headers=(
            (b"content-type", b"application/json"),
            (b"user-agent", detail),
            (b"x-codex-window-id", b"w\xff"),
        )
    )

    await _relay(upstream, _entry(), turn)

    sent = upstream.only().headers.raw
    assert (b"user-agent", detail) in sent, sent
    assert (b"x-codex-window-id", b"w\xff") in sent, sent
    assert (b"content-type", b"application/json") in sent, sent


async def test_a_credential_the_vault_will_not_hand_over_is_a_refusal() -> None:
    """A vault outage reaches the Agent Runtime as a failed Turn, not as a 500.

    Nothing between the vault and the route wrapped this, so an `AccessDenied` -- which
    the manifest's own OWED note says is the state of the deployed policy for one of
    the three entries it names -- left here as whatever botocore raised, became
    Starlette's plain-text `Internal Server Error`, and gave the operator a traceback
    per request instead of a refusal naming what could not be read.

    The refusal names neither the entry nor its value: the entry's name is this
    service's own configuration and the caller is inside a pod. Which entry failed is
    in the log line beside it, which is where an operator reads.
    """

    class _Refusing:
        async def fetch(self, name: str) -> str:
            raise PermissionError(f"AccessDenied reading {name}")

    broker = ProviderCredentialBroker(_Refusing(), _FrozenClock())

    with pytest.raises(GatewayRefusal) as refused:
        await broker.for_turn(_CALLER.session_id, _entry())

    assert refused.value.status == 503
    assert refused.value.kind == "server_error"
    assert "map/upstream/openai" not in refused.value.message
    assert "AccessDenied" not in refused.value.message


async def test_a_vault_value_that_cannot_be_a_header_is_refused_not_encoded() -> None:
    """A registration mistake named where it can still be named.

    Every credential a provider issues is an ASCII token, so a vault entry holding
    anything else is a mistake somebody made filing it -- the same class the vault
    adapter already refuses a non-string entry for. Encoding it under some other codec
    would hand the provider a credential that does not work and say nothing about why;
    raising out of `header()` would be a 500 on the platform's own misconfiguration.
    """

    class _NonAscii:
        async def fetch(self, name: str) -> str:
            return "secret-caf\u00e9"

    broker = ProviderCredentialBroker(_NonAscii(), _FrozenClock())

    with pytest.raises(GatewayRefusal) as refused:
        await broker.for_turn(_CALLER.session_id, _entry())

    assert refused.value.status == 503
    assert "caf" not in refused.value.message


async def test_an_unreachable_upstream_is_refused_without_naming_it() -> None:
    upstream = _Upstream(raises=httpx.ConnectError("no route to host"))
    entered = False

    with pytest.raises(GatewayRefusal) as refused:
        async with httpx.AsyncClient(transport=upstream.transport()) as client:
            handler = _passthrough(upstream, client)
            async with handler.open(_turn(), _entry()):
                entered = True

    assert not entered, "the exchange was yielded even though it never opened"
    assert refused.value.status == 502
    assert refused.value.kind == "server_error"
    assert "gpt-5-codex" in refused.value.message
    assert "api.openai.com" not in refused.value.message


async def test_the_upstream_response_is_closed_when_the_relay_leaves_its_context() -> (
    None
):
    """A caller that hangs up halfway must not leave the upstream connection open."""
    upstream = _Upstream(chunks=(b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"))

    async with httpx.AsyncClient(transport=upstream.transport()) as client:
        handler = _passthrough(upstream, client)
        async with handler.open(_turn(), _entry()) as response:
            first = await anext(response.body)
            assert first == b"data: a\n\n"
            assert not upstream.body.closed, "closed before the caller stopped reading"

        assert upstream.body.closed, "the upstream body outlived the exchange"
