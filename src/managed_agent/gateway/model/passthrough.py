"""The Responses shape, forwarded as it arrived.

A Responses-compatible upstream needs no translation, so this handler's whole job is to
be transparent about the *body*. The request body's bytes are the bytes that go out --
the router parsed them to read one field and threw the parse away, so a field nothing
here understands cannot be dropped by a round trip through a JSON library.

The headers are the opposite: nothing travels unless it is named. The Agent
Runtime's own vocabulary is named, including the whole `x-codex-*` family a provider
must tolerate, and the credential the routing entry declares replaces whatever the pod
sent -- which is what keeps the platform's credential on this side of the pod boundary
and the pod's own off the far side of it. `accept-encoding` is pinned to what was asked
for, and to `identity` when it asked for nothing, because the response body is relayed
without being decoded -- a content coding negotiated on the Agent Runtime's behalf would
arrive declared and undecodable.

The response direction is the reverse posture, deliberately: an upstream's headers are
relayed except the ones that describe its connection. The asymmetry is which side is
trusted. Nothing inside a Session pod is, so the request filter names what may leave;
the upstream is a service this platform chose, so its answer is relayed whole.

What is *not* asymmetric is the bytes. Both directions carry the octets that arrived,
because a header block has no charset and every guess at one here has been a bug.
`multi_items()` on the way back handed over strings httpx had decoded with a single
charset sniffed across the whole block, so one value above U+00FF changed how every
value in it was read; and on the way out a text value handed to httpx is encoded ASCII,
so one octet above 0x7F raised and 500'd the Turn. Fixing only the first left the crash
on the side that chooses the octet. A relay that never decodes cannot re-encode wrongly,
in either direction.

An upstream's own refusal is relayed with its status and its body untouched. Reading it
and rewriting it here would make this service the author of a message the upstream
wrote, and the slice that reads a refusal for a missing capability reads it out of that
body.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlencode

import httpx

from managed_agent.gateway.model.credential_broker import ProviderCredentialBroker
from managed_agent.gateway.model.router import (
    GatewayRefusal,
    InboundTurn,
    RoutingEntry,
    UpstreamResponse,
)

_LOG = logging.getLogger(__name__)

_RESPONSES_PATH = "responses"

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
"""Headers that describe one TCP connection. This hop is a different one, and forwarding
`content-length` from a body about to be reframed is how a stream ends up truncated."""

_HOP_BY_HOP_BYTES = frozenset(name.encode("ascii") for name in _HOP_BY_HOP)
"""The same names, as the wire spells them.

The response direction filters on raw bytes, so it compares against bytes. Decoding a
name in order to compare it would put a guessed codec on the one path that must not
guess -- and a header name is an ASCII token by definition, so there is nothing a
decode could tell this filter that the bytes do not."""

_FORWARDED_NAMES = frozenset(
    {
        "accept",
        "accept-encoding",
        "content-type",
        "user-agent",
        # The Agent Runtime's own turn headers, transcribed from the runtime's own
        # header-name constants -- `core/src/client.rs:143-160`,
        # `core/src/attestation.rs:7`, `login/src/auth/default_client.rs:42` and `:335`,
        # and `codex-api/src/requests/headers.rs:5-13`. Not from
        # `research/model-wire-surface.md`'s summary of them, which lists twelve where
        # the runtime sets sixteen: taking the twelve dropped `originator`, which the
        # runtime's client builder puts on every request it makes, and
        # `x-openai-memgen-request`. The `x-codex-*` family is covered by the prefix
        # below because that namespace is open-ended.
        "openai-beta",
        "originator",
        "session-id",
        "thread-id",
        "x-client-request-id",
        "x-oai-attestation",
        "x-openai-internal-codex-residency",
        "x-openai-internal-codex-responses-lite",
        "x-openai-memgen-request",
        "x-openai-subagent",
        "x-responsesapi-include-timing-metrics",
    }
)
"""The named inbound request headers that cross to the upstream hop.

An allowlist and not a denylist, and the difference is the whole security property of
this function. A denylist forwards every header nobody thought of, which is not a gap in
one list -- it is the default answer being "forward" for a class that includes every
credential form a provider will ever accept. That default is what sent a pod-supplied
`x-api-key` -- the exact header this module attaches the *platform's* credential
under -- straight through to the provider alongside the platform's own, letting a pod
have its Turn served on its own key over this service's egress path while the meter on
the way back attributed it here. It also forwarded `cookie`.

Under an allowlist the wrong answer is a header somebody has to deliberately add, and
the failure is a request the provider refuses rather than a credential nobody intended
to send. `accept-encoding` is listed and then pinned below, because the negotiated value
matters and the inbound one is where it comes from.

What an allowlist costs instead is a name nobody transcribed, and that failure is
silent: no line is logged for a header that does not travel, so the symptom is a
capability or quota difference at the provider with nothing here to trace it to. The
worst of the four this list was first written without is
`x-openai-internal-codex-residency` -- an operator who set a residency requirement had
it not declared on the wire, which is a data-residency control failing open. That is why
the entries above are transcribed from the runtime's own constants rather than from
prose about them: a summary can omit a name, and a grep over `const ... HEADER` cannot.
"""

_FORWARDED_PREFIXES = ("x-codex-",)
"""The one header namespace forwarded whole rather than by name.

The Agent Runtime sends seven `x-codex-*` headers today -- six named constants at
`core/src/client.rs:144-149` plus `x-codex-beta-features`, inserted as a literal at
`:1977` -- and the family is explicitly open-ended, a provider being required to
tolerate it, so naming its current members would make this file need editing every time
the runtime adds one. Two of those seven are absent from the prose summary that first
sourced the list above, which is the second reason not to source a filter from prose.
Every other allowed header is named exactly, because a prefix is a small denylist: it
forwards whatever lands inside it later.

Two headers the runtime can send are deliberately outside both sets.
`chatgpt-account-id` and `x-openai-fedramp` come from the pod's own bearer-auth provider
and select an account and a routing plane for the pod's own credential -- which is not
the credential that leaves here. Forwarding them would let a pod steer where the
*platform's* credential is spent.

No `anthropic-` names either. This handler serves the Responses wire only; the slice
that owns the Anthropic wire writes its own filter, and a rule here would be a rule for
traffic that never reaches this function.
"""


_FORWARDED_NAMES_BYTES = frozenset(name.encode("ascii") for name in _FORWARDED_NAMES)
_FORWARDED_PREFIXES_BYTES = tuple(p.encode("ascii") for p in _FORWARDED_PREFIXES)
"""The same two declarations, as the wire spells them.

Both directions now filter on the octets that arrived, so both compare against bytes,
and these are derived from the readable declarations above rather than written a second
time -- the same arrangement as `_HOP_BY_HOP_BYTES`, and for the same reason: two
spellings of one list is how a list comes to disagree with itself. A header name is an
ASCII token by definition, so the encode above cannot fail and nothing a decode could
tell this filter is missing from the bytes.
"""


def _forwarded(name: bytes) -> bool:
    """Whether one inbound header name crosses to the upstream hop.

    Default deny. A name matching neither the list nor a namespace does not travel,
    and `authorization` and `host` are absent from both on purpose -- the credential
    replaces the first, and httpx sets the second from the URL it is given.
    """
    return name in _FORWARDED_NAMES_BYTES or name.startswith(_FORWARDED_PREFIXES_BYTES)


def upstream_url(entry: RoutingEntry) -> str:
    """`{base_url}/responses`, with the entry's query parameters appended once.

    Percent-encoded here, unlike the Agent Runtime's own naive concatenation of the same
    shape: this service is the client on this hop, so the encoding is its responsibility
    rather than a quirk it has to reproduce.
    """
    url = f"{entry.base_url.rstrip('/')}/{_RESPONSES_PATH}"
    return f"{url}?{urlencode(entry.query_params)}" if entry.query_params else url


def _forward_headers(
    inbound: tuple[tuple[bytes, bytes], ...],
) -> dict[bytes, bytes]:
    """The inbound headers that cross to the upstream hop, lowercased.

    Default deny: a name the allowlist above does not recognise does not travel.

    Octets in and octets out, so a value the pod chose reaches the upstream as the bytes
    that arrived. That was true of the response direction and not of this one, and the
    asymmetry was backwards: httpx encodes a *text* header value as ASCII, so one octet
    above 0x7F in any forwarded header raised inside `build_request` and 500'd the Turn
    from outside the refusal envelope -- on the side this module's own docstring says
    nothing is trusted, and therefore on the side that gets to choose the octet. What is
    trusted here is not the value but the *name*: the allowlist has already decided this
    header may travel, and past that decision the value is the runtime's own vocabulary
    that this service has no opinion about.
    """
    forwarded = {
        name.lower(): value for name, value in inbound if _forwarded(name.lower())
    }
    forwarded.setdefault(b"accept-encoding", b"identity")
    return forwarded


class ResponsesPassthrough:
    """Serves UpstreamWire.RESPONSES: no translation, one header swapped."""

    def __init__(
        self, client: httpx.AsyncClient, broker: ProviderCredentialBroker
    ) -> None:
        self._client = client
        self._broker = broker

    @asynccontextmanager
    async def open(
        self, turn: InboundTurn, entry: RoutingEntry
    ) -> AsyncIterator[UpstreamResponse]:
        """Open the upstream exchange and hold it open while the body is read.

        A context manager rather than a coroutine because the body is a live stream: the
        connection has to outlive this call and has to close even when the Agent Runtime
        hangs up halfway through reading it.
        """
        credential = await self._broker.for_turn(turn.caller.session_id, entry)
        headers = _forward_headers(turn.headers)
        name, value = credential.header()
        headers[name] = value
        request = self._client.build_request(
            "POST", upstream_url(entry), headers=headers, content=turn.body
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.RequestError as exc:
            _LOG.warning(
                "upstream %s unreachable for model %s", entry.base_url, entry.model
            )
            raise GatewayRefusal(
                502, "server_error", f"model {entry.model} could not be reached"
            ) from exc
        try:
            yield UpstreamResponse(
                status=response.status_code,
                headers=tuple(
                    (name, value)
                    for name, value in response.headers.raw
                    if name.lower() not in _HOP_BY_HOP_BYTES
                ),
                body=response.aiter_raw(),
            )
        finally:
            await response.aclose()
