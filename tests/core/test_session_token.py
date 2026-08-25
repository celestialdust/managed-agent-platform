"""What the token is, graded against the reader that has to accept it.

Every case here drives the real `verify_session_token`. A test that re-derived the HMAC
itself would assert that this file agrees with itself, which is the one thing worth
nothing: the format's whole purpose is that two services -- the control plane that mints
and the Tool Gateway that reads -- agree about one string.

The golden vector is a literal rather than a computation, for the same reason. Changing
the mint should fail a test, and a test that recomputes its own expected value changes
along with the thing it is supposed to pin.
"""

from __future__ import annotations

import hashlib
import hmac
import random
from uuid import UUID, uuid4

import pytest

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER,
    SESSION_TOKEN_HEADER_NAME,
    InvalidSessionToken,
    SessionContext,
    mint_session_token,
    verify_session_token,
)

KEY = b"a signing key that is thirty-two"
OTHER_KEY = b"a second key for the other hop!!"
SESSION = SessionId(UUID("11111111-2222-3333-4444-555555555555"))
TENANT = TenantId(UUID("66666666-7777-8888-9999-aaaaaaaaaaaa"))
EXPIRY = 1788664689
NOW = EXPIRY - 1

GOLDEN = (
    "11111111-2222-3333-4444-555555555555."
    "66666666-7777-8888-9999-aaaaaaaaaaaa."
    "1788664689."
    "e8d1a9844a80f3f38b4a2193ddcb9df5"
    "8e644f79beeae6c1634a7e3f60d0de7f"
)
"""The one string this format produces for one known input, as a literal.

Read off the reader that was already deployed before this module existed, so it pins
the move as a move: the mint is new code and this is the byte-for-byte output the
Tool Gateway has always accepted.
"""


def _signed(body: str) -> str:
    """Sign a body this module would never mint, so a malformed one can be presented.

    An independent second spelling of the signature, on purpose. The negative cases
    below need a token whose signature is valid over parts the mint cannot produce --
    a non-numeric expiry, an unhyphenated uuid -- and there is no way to get one from
    `mint_session_token`. It is not used to derive any expected value: the golden
    vector above is a literal precisely so that this helper cannot stand in for it.
    """
    return f"{body}.{hmac.new(KEY, body.encode(), hashlib.sha256).hexdigest()}"


def _minted(
    *,
    session_id: SessionId = SESSION,
    tenant_id: TenantId = TENANT,
    expiry_epoch_s: int = EXPIRY,
    key: bytes = KEY,
) -> str:
    return mint_session_token(
        session_id=session_id,
        tenant_id=tenant_id,
        expiry_epoch_s=expiry_epoch_s,
        key=key,
    )


def test_a_minted_token_is_the_exact_string_this_format_has_always_been() -> None:
    """The golden vector, as a literal.

    The two halves of this format run in two services and are deployed
    independently, so a change to the bytes is a change every deployed reader has to
    already accept. A test that recomputed the signature could not see such a change
    at all.
    """
    assert _minted() == GOLDEN


def test_a_minted_token_verifies_to_the_session_and_tenant_it_names() -> None:
    assert verify_session_token(_minted(), KEY, NOW) == SessionContext(
        session_id=SESSION, tenant_id=TENANT
    )


def test_a_token_signed_with_another_key_is_refused() -> None:
    """`MAP_SHIM_TOKEN_KEY` and `MAP_SESSION_TOKEN_KEY` are two keys for two hops, so a
    second key is not a hypothetical: a token minted for the shim hop must not open the
    Gateway hop."""
    with pytest.raises(InvalidSessionToken):
        verify_session_token(_minted(key=OTHER_KEY), KEY, NOW)


def test_an_expiry_equal_to_the_readers_now_is_refused() -> None:
    """The boundary, because `<=` and `<` are one character apart and this token cannot
    be refreshed -- the second it dies is the second the Session stops taking Turns."""
    token = _minted()
    with pytest.raises(InvalidSessionToken):
        verify_session_token(token, KEY, EXPIRY)
    assert verify_session_token(token, KEY, EXPIRY - 1).session_id == SESSION


def test_one_tenants_token_does_not_verify_as_another_tenants() -> None:
    """The tenant is read back off the token rather than assumed, and it is what the
    Gateway keys a registry lookup on. Two tenants naming one Session is not a real
    shape, but the token has to distinguish them anyway: it is the only thing the
    Gateway is told."""
    other_tenant = TenantId(uuid4())
    mine = _minted()
    theirs = _minted(tenant_id=other_tenant)

    assert verify_session_token(mine, KEY, NOW).tenant_id == TENANT
    assert verify_session_token(theirs, KEY, NOW).tenant_id == other_tenant
    assert mine != theirs


def test_every_byte_of_a_minted_token_is_one_an_http_header_may_carry() -> None:
    """The case that matters most, and the reason is measured rather than reasoned.

    The runtime that carries this header drops a value it cannot put in a header,
    warns into a log nobody reads, and sends the request anyway -- which arrives at
    the Gateway as an ordinary 401, indistinguishable from a wrong key or an expired
    token. So a mint that could emit one unusable byte would produce a silent failure
    with no line anywhere naming the cause.
    """
    rng = random.Random(20260823)
    for _ in range(128):
        token = _minted(
            session_id=SessionId(UUID(int=rng.getrandbits(128))),
            tenant_id=TenantId(UUID(int=rng.getrandbits(128))),
            expiry_epoch_s=rng.randrange(1, 2**40),
        )
        assert token == token.strip()
        for character in token:
            assert 0x20 <= ord(character) <= 0x7E, (
                f"{character!r} in {token!r} is not a byte an HTTP header value may "
                "carry, so the runtime would drop the whole header"
            )


def test_the_header_name_is_the_one_spelling_in_two_encodings() -> None:
    """One constant derived from the other, never two literals. A document naming one
    spelling and a middleware reading the other is a 401 on every call with nothing
    anywhere saying why."""
    assert SESSION_TOKEN_HEADER_NAME == "x-map-session"
    assert SESSION_TOKEN_HEADER == b"x-map-session"
    assert SESSION_TOKEN_HEADER_NAME.encode("ascii") == SESSION_TOKEN_HEADER


@pytest.mark.parametrize(
    "token",
    [
        "",
        "one",
        f"{SESSION}.{TENANT}",
        f"{SESSION}.{TENANT}.{EXPIRY}",
        f"{GOLDEN}.extra",
    ],
)
def test_a_token_missing_a_part_or_carrying_a_fifth_is_refused(token: str) -> None:
    with pytest.raises(InvalidSessionToken):
        verify_session_token(token, KEY, NOW)


def test_the_signature_covers_the_expiry() -> None:
    """The property the reader's docstring claims: a presenter cannot supply their own
    deadline and have it believed. Only part 2 is moved, so if the signature covered
    the first two parts only this would verify and return a context."""
    session, tenant, _, signature = _minted().split(".")
    later = ".".join((session, tenant, str(EXPIRY + 86_400), signature))

    with pytest.raises(InvalidSessionToken):
        verify_session_token(later, KEY, NOW)


def test_a_non_numeric_expiry_is_refused_rather_than_raising_valueerror() -> None:
    """Reached only with a signature over the malformed part, which is what the
    control plane would emit if it ever stringified something unexpected. The reader
    must answer with its own exception, because every caller catches that one."""
    with pytest.raises(InvalidSessionToken):
        verify_session_token(_signed(f"{SESSION}.{TENANT}.not-a-number"), KEY, NOW)


def test_the_same_identity_spelled_without_hyphens_is_a_different_token() -> None:
    """Measured leniency, asserted so the next reader does not treat a token as a
    canonical byte string.

    `UUID()` accepts the unhyphenated spelling, so two distinct strings verify to one
    `SessionContext`. Anything that used a token as a cache key, a rate-limit key or a
    replay-detection key would therefore be defeated by a re-spelling of the same
    identity -- and nothing in this format prevents that, because the mint is not the
    only thing that can produce a token the reader accepts.
    """
    canonical = _minted()
    session, tenant, expiry, _ = canonical.split(".")
    respelled = _signed(f"{session.replace('-', '')}.{tenant}.{expiry}")

    assert respelled != canonical
    assert verify_session_token(respelled, KEY, NOW) == verify_session_token(
        canonical, KEY, NOW
    )
