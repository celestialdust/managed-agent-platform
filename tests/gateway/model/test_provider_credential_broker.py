"""Fetching the upstream credential inside the Gateway, and never writing it down.

Tier 1, no infrastructure. The broker is the real class and only the vault and the clock
are fakes, because those two are the genuine boundaries -- one would be an AWS call, the
other the passage of time -- while the caching, the expiry and the redaction are the
behaviour under test.

Two assertions read `broker._cache` directly, which is otherwise not this suite's
business. They are the only way to tell "expired, and therefore refused" from "expired,
and therefore gone": the first is observable through `for_turn`, the second is a
statement about what is resident in memory after a rotation, and a revoked credential
still sitting in a process is exactly the failure those lines exist to catch. The same
reasoning put the same two lines in `tests/gateway/tool/test_credential_broker.py`.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from managed_agent.core.ids import SessionId
from managed_agent.gateway.model.credential_broker import (
    DEFAULT_CREDENTIAL_TTL_MS,
    ProviderCredentialBroker,
    UpstreamCredential,
)
from managed_agent.gateway.model.router import (
    AuthScheme,
    GatewayRefusal,
    RoutingEntry,
    UpstreamWire,
)

_SESSION = SessionId(UUID("11111111-1111-4111-8111-111111111111"))
_OTHER = SessionId(UUID("22222222-2222-4222-8222-222222222222"))
_SECRET = "s3cret"


class _RecordingVault:
    """A vault that answers a value derived from the name and remembers every ask.

    Derived rather than constant so that a fetch under the wrong entry's name hands back
    the wrong value and the assertion fails on the value, not only on the call count.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def fetch(self, name: str) -> str:
        self.asked.append(name)
        return f"{_SECRET}-for-{name}"


class _MovableClock:
    def __init__(self, now_ms: int = 1_000_000) -> None:
        self.now_ms = now_ms

    def now_epoch_ms(self) -> int:
        return self.now_ms


def _entry(
    *,
    credential_name: str = "map/upstream/openai",
    auth_scheme: AuthScheme = AuthScheme.BEARER,
) -> RoutingEntry:
    return RoutingEntry(
        model="gpt-5-codex",
        wire=UpstreamWire.RESPONSES,
        base_url="https://api.openai.com/v1",
        auth_scheme=auth_scheme,
        credential_name=credential_name,
    )


async def test_the_entry_named_by_the_route_is_the_entry_that_is_fetched() -> None:
    vault, clock = _RecordingVault(), _MovableClock()
    broker = ProviderCredentialBroker(vault, clock)

    credential = await broker.for_turn(_SESSION, _entry(auth_scheme=AuthScheme.API_KEY))

    assert vault.asked == ["map/upstream/openai"]
    assert credential.secret == f"{_SECRET}-for-map/upstream/openai"
    assert credential.scheme is AuthScheme.API_KEY


async def test_a_second_turn_inside_the_ttl_reads_nothing_and_past_it_reads_once() -> (
    None
):
    """The window bounds how long a rotated credential can keep being presented."""
    vault, clock = _RecordingVault(), _MovableClock()
    broker = ProviderCredentialBroker(vault, clock, ttl_ms=1_000)

    await broker.for_turn(_SESSION, _entry())
    clock.now_ms += 999
    await broker.for_turn(_SESSION, _entry())

    assert vault.asked == ["map/upstream/openai"]

    clock.now_ms += 1
    await broker.for_turn(_SESSION, _entry())

    assert vault.asked == ["map/upstream/openai", "map/upstream/openai"]


async def test_two_sessions_naming_one_entry_each_fetch_on_their_own_behalf() -> None:
    """Cached under the Session, so one Session's read is not another's.

    What this proves is that the cache does not collapse two Sessions into one read --
    which is the part this code controls. Whether the vault's own audit trail then names
    the Session is a property of the vault, and `CredentialVault.fetch` carries only a
    name, so nothing here can assert it.
    """
    vault, clock = _RecordingVault(), _MovableClock()
    broker = ProviderCredentialBroker(vault, clock)

    first = await broker.for_turn(_SESSION, _entry())
    second = await broker.for_turn(_OTHER, _entry())

    assert vault.asked == ["map/upstream/openai", "map/upstream/openai"]
    assert first.secret == second.secret


async def test_one_sessions_expired_entry_is_gone_and_not_merely_refused() -> None:
    """A stopped Session's credential does not sit in memory for the process's life."""
    vault, clock = _RecordingVault(), _MovableClock()
    broker = ProviderCredentialBroker(vault, clock, ttl_ms=1_000)

    await broker.for_turn(_SESSION, _entry())
    abandoned = (_SESSION, "map/upstream/openai")

    assert abandoned in broker._cache, "nothing was cached, so nothing below is checked"

    clock.now_ms += 1_001
    await broker.for_turn(_OTHER, _entry())

    assert abandoned not in broker._cache, (
        "an expired entry nobody asked for again is still resident, so the cache grows "
        "with every Session the process has ever served"
    )
    assert (_OTHER, "map/upstream/openai") in broker._cache


async def test_two_credential_names_under_one_session_are_two_entries() -> None:
    vault, clock = _RecordingVault(), _MovableClock()
    broker = ProviderCredentialBroker(vault, clock)

    openai = await broker.for_turn(_SESSION, _entry())
    foundry = await broker.for_turn(
        _SESSION, _entry(credential_name="map/upstream/foundry")
    )

    assert vault.asked == ["map/upstream/openai", "map/upstream/foundry"]
    assert openai.secret != foundry.secret


@pytest.mark.parametrize(
    "order",
    [
        (AuthScheme.BEARER, AuthScheme.API_KEY),
        (AuthScheme.API_KEY, AuthScheme.BEARER),
    ],
    ids=["bearer first", "api_key first"],
)
async def test_two_schemes_on_one_vault_name_each_get_their_own_header_form(
    order: tuple[AuthScheme, AuthScheme],
) -> None:
    """One secret, two Routing Entries, one Session, one TTL -- two header forms.

    An upstream that accepts both `Authorization: Bearer` and `x-api-key` is the reason
    an operator points two models at one vault entry, so this is the ordinary shape
    rather than a corner. A cache that held the whole credential under the entry's name
    served whichever scheme arrived first to both: the second entry's request went out
    under the first one's header and carried none of its own, for up to one TTL, and
    which way it broke depended on which model the Session asked for first.

    Both orders, because the defect was symmetric and a single order would leave half of
    it uncovered.
    """
    vault, clock = _RecordingVault(), _MovableClock()
    broker = ProviderCredentialBroker(vault, clock)

    forms = [
        (await broker.for_turn(_SESSION, _entry(auth_scheme=scheme))).header()[0]
        for scheme in order
    ]

    assert forms == [scheme.header_name().encode("ascii") for scheme in order]
    assert vault.asked == ["map/upstream/openai"], (
        "the scheme is configuration this process already holds, so distinguishing two "
        "of them must not cost a second vault read"
    )


async def test_a_cached_secret_does_not_render_its_value() -> None:
    """The cache entry, not only the credential handed out of it.

    `UpstreamCredential` redacts itself, which closes the route from a credential to a
    log line. The cache holds the same value in a different object, and a dataclass's
    default repr prints every field -- so a debug line formatting the cache would print
    what the credential refuses to.
    """
    vault, clock = _RecordingVault(), _MovableClock()
    broker = ProviderCredentialBroker(vault, clock)
    await broker.for_turn(_SESSION, _entry())

    cached = broker._cache[(_SESSION, "map/upstream/openai")]

    for rendering in (repr(cached), str(cached), f"{cached}", repr(broker._cache)):
        assert _SECRET not in rendering, rendering
        assert "redacted" in rendering, rendering


async def test_a_cache_that_never_expires_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ProviderCredentialBroker(_RecordingVault(), _MovableClock(), ttl_ms=0)
    with pytest.raises(ValueError, match="must be positive"):
        ProviderCredentialBroker(_RecordingVault(), _MovableClock(), ttl_ms=-1)


def test_the_default_window_is_a_positive_number_of_milliseconds() -> None:
    assert DEFAULT_CREDENTIAL_TTL_MS > 0


@pytest.mark.parametrize(
    ("scheme", "expected"),
    [
        (AuthScheme.BEARER, (b"authorization", f"Bearer {_SECRET}".encode("ascii"))),
        (AuthScheme.API_KEY, (b"x-api-key", _SECRET.encode("ascii"))),
    ],
)
def test_a_credential_becomes_exactly_one_header(
    scheme: AuthScheme, expected: tuple[bytes, bytes]
) -> None:
    """`x-api-key` and not `api-key`: the Foundry Anthropic route 401s on the latter."""
    assert UpstreamCredential(scheme=scheme, secret=_SECRET).header() == expected


def test_no_rendering_of_a_credential_carries_its_value() -> None:
    """repr, str and an f-string all lead somewhere that says nothing.

    Three routes because print, logging and f-strings reach a value by different ones,
    and a type that hides from one and not the others teaches a reader a rule that does
    not hold. `object.__format__` defers to `__str__`, which is what covers the third.
    """
    credential = UpstreamCredential(scheme=AuthScheme.BEARER, secret=_SECRET)

    renderings = [repr(credential), str(credential), f"{credential}"]

    for rendering in renderings:
        assert _SECRET not in rendering, rendering
        assert "redacted" in rendering, rendering
        assert AuthScheme.BEARER.value in rendering, rendering


# --- the two shapes a vault entry really arrives in --------------------------------


class _FixedVault:
    """A vault answering one literal, so a test can state the stored shape exactly."""

    def __init__(self, value: str) -> None:
        self.value = value

    async def fetch(self, name: str) -> str:
        return self.value


async def test_a_json_entry_hands_over_its_api_key_and_not_the_whole_document() -> None:
    """The shape the live Foundry entry is actually stored in.

    Measured on 2026-08-23: `map/dev/providers/anthropic` holds
    `{"api_key", "base_url", "model"}`, and the broker took the whole entry as the
    credential -- so `x-api-key` carried the entire JSON document and the upstream
    answered 401 "invalid subscription key or wrong API endpoint", naming a cause that
    was not the cause. Nothing in the tree noticed, because nothing in the tree ever
    made a real request to that upstream.

    The two sibling fields are asserted absent from the header rather than merely
    unread. A parse that returned the document with the key extracted, or that
    concatenated fields, would still pass an assertion that only checked the key is in
    there somewhere.
    """
    document = '{"api_key": "the-real-key", "base_url": "https://h/anthropic", '
    document += '"model": "gsds-claude-opus-4-6"}'
    broker = ProviderCredentialBroker(_FixedVault(document), _MovableClock())

    credential = await broker.for_turn(_SESSION, _entry(auth_scheme=AuthScheme.API_KEY))
    name, value = credential.header()

    assert credential.secret == "the-real-key"
    assert (name, value) == (b"x-api-key", b"the-real-key")
    assert b"base_url" not in value and b"model" not in value


async def test_a_bare_entry_is_the_credential_and_is_not_parsed() -> None:
    """The other legitimate shape, and it must not be touched.

    An operator who stored the credential alone gets it back byte for byte. This is the
    control on the case above: a parse that ran unconditionally would have to guess what
    a bare string means, and the guess is a credential going onto the wire.
    """
    broker = ProviderCredentialBroker(
        _FixedVault("sk-a-bare-credential"), _MovableClock()
    )

    credential = await broker.for_turn(_SESSION, _entry())

    assert credential.secret == "sk-a-bare-credential"


@pytest.mark.parametrize(
    ("name", "stored"),
    [
        ("a JSON object with no api_key", '{"key": "wrong-field-name"}'),
        ("a JSON object whose api_key is empty", '{"api_key": ""}'),
        ("a JSON object whose api_key is not a string", '{"api_key": 5}'),
        ("a JSON object whose api_key is null", '{"api_key": null}'),
        ("a JSON array", '{"a": 1'),
        ("an object that does not parse", '{"api_key": "x"'),
    ],
)
async def test_an_entry_that_opens_like_json_and_carries_no_key_is_refused(
    name: str, stored: str
) -> None:
    """Refused, and never fallen back to as an opaque credential.

    Falling back is the tempting behaviour and it is the wrong one: it would put a
    JSON document on the wire as a bearer, which is the defect this parse exists to fix,
    and the only report of it is a 401 from the upstream naming the wrong cause. A
    registration mistake has to fail where the entry's name is still in scope, which is
    here -- and the log line names the entry while the refusal to the caller does not.
    """
    broker = ProviderCredentialBroker(_FixedVault(stored), _MovableClock())

    with pytest.raises(GatewayRefusal) as refused:
        await broker.for_turn(_SESSION, _entry())

    assert refused.value.status == 503, name
    assert "map/upstream/openai" not in refused.value.message, (
        "the refusal named a vault entry to a caller inside a pod"
    )
    assert "api_key" not in refused.value.message


async def test_a_credential_that_cannot_be_a_header_is_refused_after_the_parse() -> (
    None
):
    """Order matters here, and this is what pins it.

    The ASCII check has to read the *extracted* credential, not the stored entry. A
    JSON entry whose sibling fields hold non-ASCII text -- a display name, a comment --
    would otherwise be refused for a value that never reaches a header, and the operator
    would be told a working credential cannot be one.
    """
    broker = ProviderCredentialBroker(
        _FixedVault('{"api_key": "fine", "note": "café"}'), _MovableClock()
    )

    credential = await broker.for_turn(_SESSION, _entry())

    assert credential.secret == "fine"

    refusing = ProviderCredentialBroker(
        _FixedVault('{"api_key": "café"}'), _MovableClock()
    )
    with pytest.raises(GatewayRefusal):
        await refusing.for_turn(_SESSION, _entry())
