"""Establishes the platform reviewer on a request, from the token it presented.

This is the authentication half and `audit.py` holds the authorization half, because
they answer different questions and change for different reasons: *who is this* is a
question about a credential's signature, and *may they read this* is a question about
which principals a surface accepts. Folded together, widening one would silently widen
the other.

**A failed token establishes nothing rather than refusing here.** An absent header, an
unreadable one, a Session-audience token, a wrong signature, an expired token and a
process with no signing key all leave the request in exactly the state a request with
no credential is in -- so all six arrive at the same refusal, with the same code and
the same message, and a caller probing this surface learns nothing about which of the
six they hit. That swallow is the security property, not an unhandled error: the
refusal itself is not skipped, it is produced one layer up by the principal being
absent, which is where a request carrying no credential at all is already refused.

Scoped to the audit router rather than installed as application middleware, so no other
surface on this app can come to hold a reviewer principal -- a principal that reads
across every tenant should be reachable from exactly the routes that were designed for
it, and a router-level dependency makes a route added there later inherit it rather
than needing whoever adds it to remember.

Deliberately writes the principal onto request state instead of returning it. That keeps
the resolution downstream a function of two claim values rather than of a request, so
every case it refuses stays reachable in a test without constructing one. It is also
the only writer of that attribute anywhere in the package -- asserted structurally,
because anything else that wrote it would be a second, unsigned way to become a
reviewer.
"""

from fastapi import Request

from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.reviewers.audit_reader import REVIEWER_CLAIM
from managed_agent.control.reviewers.token import InvalidReviewerToken

_BEARER = "bearer"


def _presented_token(request: Request) -> str | None:
    """The bearer credential on this request, or None if it carries none.

    The scheme is compared lowercased because HTTP auth schemes are case-insensitive
    and `Bearer`, `bearer` and `BEARER` are one thing; a client that picked a different
    casing must not be told it has no credential when it has a valid one.
    """
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != _BEARER or not token:
        return None
    return token


def establish_reviewer_principal(request: Request) -> None:
    """Set this request's reviewer claim if, and only if, a token proves one.

    Returns nothing. Whether the request ends up authorized is decided downstream from
    the claims on it, so this either adds a proven identity or leaves the request
    untouched -- there is no third outcome and no value here for a caller to
    misinterpret.

    Establishes no tenant claim, and that absence is the point rather than an omission:
    a reviewer reads across every tenant, so there is no tenant for this credential to
    name, and a tenant claim arriving alongside a reviewer one is refused downstream
    precisely so that an audit read can never happen under a tenant's credential.
    """
    token = _presented_token(request)
    if token is None:
        return
    try:
        reviewer_id = platform_from_request(request).reviewer_authenticator.reviewer_of(
            token
        )
    except InvalidReviewerToken:
        return
    setattr(request.state, REVIEWER_CLAIM, reviewer_id)
