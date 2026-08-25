"""The credential a platform reviewer presents, and why no Session can forge one.

A reviewer reads any tenant's Event Log. A Session's pod reads model and tool
endpoints. Both authenticate to this platform with an HMAC over a dot-joined string,
both are signed with a symmetric key the control plane holds, and the two must never
be interchangeable -- a Session token that opened the audit surface would let a
tenant's own agent code read every other tenant's history, because that token is a
file inside the tenant's pod.

**The separation is in the signed bytes, not in a field somebody checks.** Field 0 of
a reviewer token is the literal `map-audit-reviewer`, and it is covered by the
signature. A Session token's field 0 is a uuid, so it can never satisfy the equality
below; a reviewer token handed to `verify_session_token` dies parsing
`UUID("map-audit-reviewer")`. Neither refusal depends on a reader remembering to look
at an audience, which is what makes the two token families disjoint rather than merely
distinguished. It also means a change to the Session layout -- another field, another
order -- cannot quietly make one accept the other, because the domain tag is the first
thing the message says.

Why not add an audience field to the Session layout instead: that layout is verified
by two deployed gateways, so moving it means every running Session's token stops
verifying. Domain separation costs nothing on that path and moves nothing.

**The key is the control plane's own, not the one the gateways share.** Two symmetric
keys exist and they are two deliberately (`deploy/k8s/control-plane.yaml`): one signs
the control-plane-to-shim bearer, the other signs the Session token that both gateways
verify. A reviewer token is signed with the *first*. The second is held by three
services, two of which only ever verify with it, so a compromise of either gateway
would hand over the ability to mint a reviewer token and read every tenant's audit
log. The first is held by the one service that also authorizes the audit read, so
holding it buys nothing that service could not already do. Do not merge the two keys
back into one on the grounds that both are HMAC keys the control plane holds.

Framework-free, like the Session layout it borrows its construction from: the mint and
the verify are functions of strings and bytes, so every case either refuses is
reachable in a test without an HTTP request. ADR-024 records the decision; ADR-023
records the Session layout this reuses.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Final, Protocol
from uuid import UUID

from managed_agent.core.ports import Clock

REVIEWER_AUDIENCE: Final = "map-audit-reviewer"
"""The domain tag every reviewer token carries as its first field, inside the signature.

Contains no dot, so it survives the split that reads the token, and is not a uuid, so
no Session token can present it. Changing this string invalidates every reviewer token
already issued, which is the intended effect of retiring an audience.
"""

_PARTS: Final = 4


class InvalidReviewerToken(Exception):
    """The token was absent, malformed, signed with another key, or past its expiry.

    One exception for every failure and no distinguishing detail, because the caller is
    an unauthenticated stranger: knowing *which* check failed tells them whether a
    reviewer id exists and whether they have the right key, one bit at a time.
    """


def mint_reviewer_token(*, reviewer_id: UUID, expiry_epoch_s: int, key: bytes) -> str:
    """Sign one reviewer's identity so `verify_reviewer_token` will read it back.

    Four dot-separated fields: the audience tag, the reviewer, the expiry in epoch
    seconds, and the hex HMAC-SHA256 over the first three rejoined by dots. The
    signature covers the audience and the expiry, so a presenter can neither relabel a
    token for another surface nor supply their own deadline and have it believed.

    Nothing here is escaped or quoted because nothing here can emit a byte that would
    need it -- a literal ASCII tag, a canonical uuid, decimal digits, dots and
    lowercase hex. That matters more than it looks: an HTTP client handed a header value
    it cannot send drops the header, warns into a log nobody reads, and sends the
    request anyway, which arrives as an ordinary 401 with nothing naming the cause.

    The expiry is absolute rather than a lifetime and no clock is read here, so the
    caller owns the clock exactly as the verifier does and two callers a second apart
    cannot mint tokens that die at different times.

    There is no revocation. A minted token is good until its expiry or until the key
    changes, so the expiry is the only bound on a leaked one -- mint short.
    """
    body = ".".join((REVIEWER_AUDIENCE, str(reviewer_id), str(expiry_epoch_s)))
    return f"{body}.{hmac.new(key, body.encode(), hashlib.sha256).hexdigest()}"


def verify_reviewer_token(token: str, key: bytes, now_epoch_s: int) -> UUID:
    """The reviewer this token names, or raise.

    Returns the parsed uuid rather than a bool, so what a caller holds afterwards is
    the identity itself and there is no second place where an unparsed string could be
    trusted for having passed a check somewhere else.

    The audience is compared first and with a plain `!=`: it is a public constant, so
    there is nothing about it to learn from how long the comparison takes, and refusing
    on it before the HMAC keeps a Session token from ever reaching the signature check
    at all. The signature is then compared with `compare_digest`, and it is checked
    before the expiry is parsed so that a presenter cannot supply their own expiry and
    have it read on the way to being rejected for something else.

    An expiry equal to `now_epoch_s` is refused: a token is valid strictly before the
    second it names.
    """
    parts = token.split(".")
    if len(parts) != _PARTS or parts[0] != REVIEWER_AUDIENCE:
        raise InvalidReviewerToken("invalid reviewer token")
    body = ".".join(parts[:3])
    expected = hmac.new(key, body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, parts[3]):
        raise InvalidReviewerToken("invalid reviewer token")
    try:
        reviewer_id = UUID(parts[1])
        expiry = int(parts[2])
    except ValueError as malformed:
        raise InvalidReviewerToken("invalid reviewer token") from malformed
    if expiry <= now_epoch_s:
        raise InvalidReviewerToken("invalid reviewer token")
    return reviewer_id


class ReviewerAuthenticator(Protocol):
    """Turns a presented token into a reviewer identity, or refuses.

    An interface rather than the function above so that the surface reading it depends
    on "something can name the reviewer behind this token" and not on HMAC, a key, or a
    clock. The two implementations below are the configured case and the unconfigured
    one, and the unconfigured one refuses -- which is what lets a process with no key
    start and serve every other route rather than failing to start.
    """

    def reviewer_of(self, token: str) -> UUID:
        """Raise `InvalidReviewerToken` unless `token` proves a reviewer identity."""
        ...


@dataclass(frozen=True, slots=True)
class HmacReviewerTokens:
    """Verifies with a key held for the process's life, against a clock it is handed.

    No cache and no re-read: the key arrives as a pod field fixed at admission, so a
    rotation replaces the pod and there is nothing to invalidate. The clock is injected
    for the ordinary reason -- a module that read the wall clock itself is a module
    whose expiry behaviour no test can move time for.

    Milliseconds divided down rather than a second clock, because `Clock` is the one
    time port this codebase has and a second one would be a second answer to what time
    it is.
    """

    key: bytes
    clock: Clock

    def reviewer_of(self, token: str) -> UUID:
        return verify_reviewer_token(token, self.key, self.clock.now_epoch_ms() // 1000)


@dataclass(frozen=True, slots=True)
class NoReviewerKey:
    """Refuses every token, for a process that was given no signing key.

    The safe default has to be a refusing object and not an empty key, because
    `hmac.new(b"", ...)` is perfectly valid: a process defaulted to `b""` would verify
    tokens anybody can mint, while looking configured and answering 200. This answers
    the same refusal as a wrong signature, so an unconfigured deployment is
    indistinguishable from a wrong credential to whoever is probing it, and the reason
    is an operator's to find in the manifest rather than a caller's to be told.
    """

    def reviewer_of(self, token: str) -> UUID:
        raise InvalidReviewerToken("invalid reviewer token")
