"""The reviewer credential: what it proves, and what it refuses to prove.

The load-bearing claim is not "a valid token verifies". It is that the reviewer family
and the Session family are **disjoint**: no Session token opens the audit surface and no
reviewer token opens a gateway. That is the whole security argument for reusing one
signing key, so it is asserted in both directions against the two real verifiers rather
than reasoned about -- a Session token is minted by `mint_session_token` here, not
hand-written, because a hand-written one proves only that this file agrees with itself.

The refusal table is parametrized over the reasons rather than written once against the
first one. Each reason is a separate decision about what must not be believed, and a
single case against the first would leave the rest graded by nothing -- which is how
four of five refusal reasons in `adapters/kubernetes/pod_runner.py` came to be
removable with the suite green.
"""

from __future__ import annotations

import hashlib
import hmac
import random
from collections.abc import Callable, Mapping
from typing import Final
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.requests import Request

from managed_agent.composition import Platform, build
from managed_agent.control.api.request.reviewer_auth import establish_reviewer_principal
from managed_agent.control.reviewers.audit_reader import REVIEWER_CLAIM, TENANT_CLAIM
from managed_agent.control.reviewers.token import (
    REVIEWER_AUDIENCE,
    HmacReviewerTokens,
    InvalidReviewerToken,
    NoReviewerKey,
    ReviewerAuthenticator,
    mint_reviewer_token,
    verify_reviewer_token,
)
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.session.session_token import (
    InvalidSessionToken,
    mint_session_token,
    verify_session_token,
)

KEY = b"a signing key that is thirty-two"
OTHER_KEY = b"a second key for the other hop!!"
REVIEWER = UUID("bbbbbbbb-cccc-dddd-eeee-ffffffffffff")
EXPIRY = 1788664689
NOW = EXPIRY - 1

_UNDIALLED = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/unused"
_KEY_ENV = "MAP_SHIM_TOKEN_KEY"

GOLDEN: Final = (
    "map-audit-reviewer."
    "bbbbbbbb-cccc-dddd-eeee-ffffffffffff."
    "1788664689."
    "36502aa6a11ead0a8e08c0e9e40a4aa6"
    "cbb4322ff7c8e69d53c493795db6159e"
)
"""The one string this format produces for one known input, as a literal.

A literal and not a recomputation, for the reason the Session format's golden vector is
one: a test that derives its own expected value changes silently along with the thing it
is supposed to pin, and this string is the only thing standing between an issued token
and a mint that quietly starts producing something else.
"""


class FrozenClock:
    """A clock the test moves, so an expiry can be crossed without waiting for one."""

    def __init__(self, epoch_s: int) -> None:
        self._epoch_s = epoch_s

    def now_epoch_ms(self) -> int:
        return self._epoch_s * 1000


def _a_platform(**overrides: object) -> Platform:
    """A `Platform` whose ports are all absent, for the paths that touch none.

    The reviewer authenticator is the only field these cases read. Every port is `None`
    rather than a fake, deliberately: a fake would answer if something reached for one,
    and what several cases here assert is that nothing does.
    """
    return Platform(
        event_log_append=None,  # type: ignore[arg-type]
        event_log_range=None,  # type: ignore[arg-type]
        definition_registry=None,  # type: ignore[arg-type]
        tool_registry=None,  # type: ignore[arg-type]
        session_registry=None,  # type: ignore[arg-type]
        webhooks=None,  # type: ignore[arg-type]
        environment_store=None,  # type: ignore[arg-type]
        turn_dispatch=None,  # type: ignore[arg-type]
        file_store=None,  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


def _app_wired_with(authenticator: ReviewerAuthenticator) -> FastAPI:
    """The least app the authenticator can read its collaborator off.

    It reaches the wired platform the way every route in the package does -- through
    `app.state` -- so a fake here would have to fake that lookup rather than the thing
    being tested. `Platform` is built with `None` ports because nothing on this path
    touches one; a port that were reached would raise rather than answer.
    """
    app = FastAPI()
    app.state.platform = _a_platform(reviewer_authenticator=authenticator)
    return app


def _signed(body: str) -> str:
    """Sign a body the mint would never produce, so a malformed token can be presented.

    An independent second spelling of the signature, deliberately. Several refusals
    below are only reachable with a *validly signed* token whose fields are wrong -- a
    non-numeric expiry, a reviewer that is not a uuid -- and there is no way to obtain
    one from `mint_reviewer_token`. It derives no expected value: `GOLDEN` is a literal
    precisely so this helper cannot stand in for it.
    """
    return f"{body}.{hmac.new(KEY, body.encode(), hashlib.sha256).hexdigest()}"


def _a_session_token() -> str:
    """A real Session token, minted by the real mint the pods are given."""
    return mint_session_token(
        session_id=SessionId(uuid4()),
        tenant_id=TenantId(uuid4()),
        expiry_epoch_s=EXPIRY,
        key=KEY,
    )


# --------------------------------------------------------------------------------------
# What the format is
# --------------------------------------------------------------------------------------


def test_a_minted_token_is_the_exact_string_this_format_has_always_been() -> None:
    assert (
        mint_reviewer_token(reviewer_id=REVIEWER, expiry_epoch_s=EXPIRY, key=KEY)
        == GOLDEN
    )


def test_a_minted_token_verifies_to_the_reviewer_it_names() -> None:
    """The parsed uuid comes back, not a bool: what the caller then holds is the
    identity itself, so there is no second place where an unparsed string could be
    trusted for having passed a check somewhere else."""
    verified = verify_reviewer_token(GOLDEN, KEY, NOW)

    assert verified == REVIEWER
    assert isinstance(verified, UUID)


def test_the_audience_is_the_first_field_and_is_covered_by_the_signature() -> None:
    """Relabelling a token for another surface must not survive.

    If the signature covered only the reviewer and the expiry, the audience would be a
    field a presenter could rewrite -- and the entire separation between the two token
    families would rest on nobody bothering to.
    """
    audience, reviewer, expiry, signature = GOLDEN.split(".")

    assert audience == REVIEWER_AUDIENCE
    relabelled = ".".join(("map-tool-gateway", reviewer, expiry, signature))
    with pytest.raises(InvalidReviewerToken):
        verify_reviewer_token(relabelled, KEY, NOW)


def test_the_signature_covers_the_expiry() -> None:
    """A presenter cannot supply their own deadline and have it believed. Only field 2
    moves here, so a signature over the first two fields would let this verify."""
    audience, reviewer, _, signature = GOLDEN.split(".")
    later = ".".join((audience, reviewer, str(EXPIRY + 86_400), signature))

    with pytest.raises(InvalidReviewerToken):
        verify_reviewer_token(later, KEY, NOW)


def test_an_expiry_equal_to_the_readers_now_is_refused() -> None:
    """The boundary, because `<=` and `<` are one character apart and there is no
    revocation -- the expiry is the only bound on a token that has leaked."""
    with pytest.raises(InvalidReviewerToken):
        verify_reviewer_token(GOLDEN, KEY, EXPIRY)
    assert verify_reviewer_token(GOLDEN, KEY, EXPIRY - 1) == REVIEWER


def test_every_byte_of_a_minted_token_is_one_an_http_header_may_carry() -> None:
    """A credential travels in `Authorization`, and a client handed a header value it
    cannot send drops the header, warns into a log nobody reads, and sends the request
    anyway -- arriving as an ordinary 401 with nothing naming the cause."""
    rng = random.Random(20260823)
    for _ in range(128):
        token = mint_reviewer_token(
            reviewer_id=UUID(int=rng.getrandbits(128)),
            expiry_epoch_s=rng.randrange(1, 2**40),
            key=KEY,
        )
        assert token == token.strip()
        for character in token:
            assert 0x20 <= ord(character) <= 0x7E, (
                f"{character!r} in {token!r} is not a byte an HTTP header value may "
                "carry, so the client would drop the whole header"
            )


# --------------------------------------------------------------------------------------
# The refusal table
# --------------------------------------------------------------------------------------

_REFUSED: Final[Mapping[str, Callable[[], str]]] = {
    "a session token, minted by the real mint a pod is given": _a_session_token,
    "a token signed with the other of the platform's two keys": lambda: (
        mint_reviewer_token(reviewer_id=REVIEWER, expiry_epoch_s=EXPIRY, key=OTHER_KEY)
    ),
    "a token whose signature has been altered": lambda: (
        GOLDEN[:-1] + ("0" if GOLDEN[-1] != "0" else "1")
    ),
    "no token at all": lambda: "",
    "a bare uuid with nothing signing it": lambda: str(REVIEWER),
    "the audience alone": lambda: REVIEWER_AUDIENCE,
    "three fields where four are required": lambda: ".".join(GOLDEN.split(".")[:3]),
    "a fifth field appended": lambda: f"{GOLDEN}.extra",
    "a validly signed token naming no audience this surface knows": lambda: _signed(
        f"map-tool-gateway.{REVIEWER}.{EXPIRY}"
    ),
    "a validly signed token whose reviewer is not a uuid": lambda: _signed(
        f"{REVIEWER_AUDIENCE}.not-a-uuid.{EXPIRY}"
    ),
    "a validly signed token whose expiry is not a number": lambda: _signed(
        f"{REVIEWER_AUDIENCE}.{REVIEWER}.whenever"
    ),
    "a token that expired a day ago": lambda: mint_reviewer_token(
        reviewer_id=REVIEWER, expiry_epoch_s=NOW - 86_400, key=KEY
    ),
}
"""Every reason a presented string must not become a reviewer identity.

A dict rather than a list so each reason names itself in the test id, and parametrized
over rather than sampled: each entry is a separate decision about what must not be
believed, and a case written against only the first would leave the other eleven graded
by nothing at all.
"""


@pytest.mark.parametrize("reason", sorted(_REFUSED), ids=lambda name: name)
def test_a_token_this_surface_must_not_believe_is_refused(reason: str) -> None:
    with pytest.raises(InvalidReviewerToken):
        verify_reviewer_token(_REFUSED[reason](), KEY, NOW)


@pytest.mark.parametrize("reason", sorted(_REFUSED), ids=lambda name: name)
def test_every_refusal_is_indistinguishable_from_every_other(reason: str) -> None:
    """One exception and one message for all twelve.

    A caller who learns *which* check failed learns whether a reviewer id exists and
    whether they hold the right key, one request at a time. So the refusals must be
    equal as strings, not merely the same class -- a message that interpolated the
    reason would still be the same exception type.
    """
    with pytest.raises(InvalidReviewerToken) as refused:
        verify_reviewer_token(_REFUSED[reason](), KEY, NOW)

    assert str(refused.value) == "invalid reviewer token"


# --------------------------------------------------------------------------------------
# The two families are disjoint, in both directions
# --------------------------------------------------------------------------------------


def test_a_session_token_does_not_open_the_audit_surface() -> None:
    """The direction that matters most, because a Session token is a file inside a
    tenant's own pod: the tenant's agent code can read it. If it verified here, any
    tenant would hold a credential for every other tenant's audit log.

    Both are signed with the same key on purpose, so nothing about this refusal comes
    from the signature -- it comes from the audience tag being field 0 of the signed
    message, which a Session token cannot carry because field 0 there is a uuid.
    """
    session_token = _a_session_token()

    assert verify_session_token(session_token, KEY, NOW) is not None
    with pytest.raises(InvalidReviewerToken):
        verify_reviewer_token(session_token, KEY, NOW)


def test_a_reviewer_token_does_not_open_a_gateway() -> None:
    """The other direction, and it is not symmetric with the first.

    A reviewer token's signature *does* verify under `verify_session_token`'s
    construction -- both HMAC the first three fields with the same key -- so what
    refuses it is the parse: field 0 is the audience literal and no uuid reads from it.
    Worth pinning precisely because it means the separation here rests on that parse
    and would be lost if the Session layout ever accepted a non-uuid in field 0.
    """
    with pytest.raises(InvalidSessionToken):
        verify_session_token(GOLDEN, KEY, NOW)


def test_the_two_mints_cannot_produce_one_string() -> None:
    """Structural, over many pairs rather than the one above: no input to either mint
    yields a token the other's verifier accepts, so the disjointness is a property of
    the formats and not of the values this file happened to choose."""
    rng = random.Random(20260824)
    for _ in range(64):
        expiry = rng.randrange(NOW + 1, 2**40)
        reviewer = mint_reviewer_token(
            reviewer_id=UUID(int=rng.getrandbits(128)), expiry_epoch_s=expiry, key=KEY
        )
        session = mint_session_token(
            session_id=SessionId(UUID(int=rng.getrandbits(128))),
            tenant_id=TenantId(UUID(int=rng.getrandbits(128))),
            expiry_epoch_s=expiry,
            key=KEY,
        )

        assert reviewer != session
        with pytest.raises(InvalidReviewerToken):
            verify_reviewer_token(session, KEY, NOW)
        with pytest.raises(InvalidSessionToken):
            verify_session_token(reviewer, KEY, NOW)


# --------------------------------------------------------------------------------------
# The authenticator, and how a process comes to hold one
# --------------------------------------------------------------------------------------


def test_the_configured_authenticator_reads_its_clock_rather_than_the_wall() -> None:
    """The clock is injected, so the expiry boundary is reachable without waiting.

    Seconds are derived from the millisecond port by division, so a token that dies
    mid-second dies for the whole second -- asserted rather than assumed, because a
    truncation that rounded the other way would extend every token by up to a second.
    """
    authenticator = HmacReviewerTokens(key=KEY, clock=FrozenClock(EXPIRY - 1))

    assert authenticator.reviewer_of(GOLDEN) == REVIEWER
    with pytest.raises(InvalidReviewerToken):
        HmacReviewerTokens(key=KEY, clock=FrozenClock(EXPIRY)).reviewer_of(GOLDEN)


def test_an_unconfigured_process_refuses_a_token_it_would_otherwise_accept() -> None:
    """The fail-safe default, graded against the one token that is genuinely valid.

    `NoReviewerKey` must refuse `GOLDEN` -- a real, unexpired, correctly signed token --
    because the only thing it is missing is a key. A default that accepted anything
    would accept everything.
    """
    with pytest.raises(InvalidReviewerToken):
        NoReviewerKey().reviewer_of(GOLDEN)


def test_a_platform_built_with_no_key_holds_an_authenticator_that_refuses() -> None:
    """The default on `Platform` itself, because that is what two dozen construction
    sites get. If it were a key rather than a refusing object every one of them would
    quietly become a reviewer-accepting deployment."""
    with pytest.raises(InvalidReviewerToken):
        _a_platform().reviewer_authenticator.reviewer_of(GOLDEN)


def test_the_root_builds_a_working_authenticator_from_the_key_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring, end to end from the variable the manifest sets.

    Without this the two halves could each be right and never meet: a verifier that
    works and a `build` that hands out `NoReviewerKey` produces a surface that refuses
    every real credential, which is the failure the audit endpoint already shipped with
    once.
    """
    monkeypatch.setenv(_KEY_ENV, KEY.decode())
    platform, engine = build(_UNDIALLED)

    assert (
        platform.reviewer_authenticator.reviewer_of(
            mint_reviewer_token(reviewer_id=REVIEWER, expiry_epoch_s=2**40, key=KEY)
        )
        == REVIEWER
    )
    assert engine is not None


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "set to the empty string"])
def test_a_key_that_is_absent_or_empty_leaves_an_authenticator_that_refuses(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """Empty is not a key, and this is the case that would be easy to get wrong.

    `hmac.new(b"", ...)` is a perfectly valid HMAC, so treating `""` as configured would
    build a verifier anybody can forge against -- while every probe reported the process
    configured, because the variable is set. The two arms are asserted separately
    because they arrive by different routes: an absent variable is an operator who has
    not set it, an empty one is a Secret key whose value is blank.
    """
    if value is None:
        monkeypatch.delenv(_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(_KEY_ENV, value)
    platform, _ = build(_UNDIALLED)

    assert isinstance(platform.reviewer_authenticator, NoReviewerKey)
    with pytest.raises(InvalidReviewerToken):
        platform.reviewer_authenticator.reviewer_of(GOLDEN)


def test_a_proven_reviewer_establishes_a_reviewer_and_no_tenant() -> None:
    """Both claims the gate downstream reads, from one credential.

    The reviewer claim is set to the parsed uuid; the tenant claim is left unset, and
    that absence is what the gate needs rather than an oversight -- a reviewer reads
    across every tenant, so there is no tenant for the credential to name, and a request
    that ended up carrying both is refused precisely so an audit read can never happen
    under a tenant's credential.

    Driving the dependency directly rather than through a client, because what is under
    test is which attributes it writes, and a 200 could not tell "set the reviewer
    claim" from "set both and got lucky about which one was read".
    """
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/audit/sessions/x/events",
            "headers": Headers({"authorization": f"Bearer {GOLDEN}"}).raw,
            "app": _app_wired_with(HmacReviewerTokens(key=KEY, clock=FrozenClock(NOW))),
        }
    )

    establish_reviewer_principal(request)

    assert getattr(request.state, REVIEWER_CLAIM, None) == REVIEWER
    assert not hasattr(request.state, TENANT_CLAIM)


@pytest.mark.parametrize(
    "presented",
    ["", f"Bearer {GOLDEN}x", "Bearer", f"Basic {GOLDEN}", GOLDEN],
    ids=[
        "no header at all",
        "a token whose signature was altered",
        "a scheme with no token behind it",
        "the right token under the wrong scheme",
        "a bare token with no scheme",
    ],
)
def test_a_request_that_proves_nothing_is_left_carrying_nothing(presented: str) -> None:
    """The refusing arm: the request comes out of the authenticator unchanged.

    That is the whole reason there is no error path here. Every one of these leaves the
    request exactly as a request with no credential at all, so all of them arrive at one
    refusal downstream and none of them can be told apart from outside.
    """
    headers = {"authorization": presented} if presented else {}
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/audit/sessions/x/events",
            "headers": Headers(headers).raw,
            "app": _app_wired_with(HmacReviewerTokens(key=KEY, clock=FrozenClock(NOW))),
        }
    )

    establish_reviewer_principal(request)

    assert not hasattr(request.state, REVIEWER_CLAIM)
    assert not hasattr(request.state, TENANT_CLAIM)


def test_the_field_is_declared_by_its_abstraction() -> None:
    """The surface reading this depends on "something can name the reviewer behind this
    token" and not on HMAC, a key or a clock -- so the annotation has to be the protocol
    and not either implementation."""
    assert Platform.__annotations__["reviewer_authenticator"] is ReviewerAuthenticator
