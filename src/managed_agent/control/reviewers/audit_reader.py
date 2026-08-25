"""Who may read the audit trail across tenants, and why that reader is not a tenant.

Two authorizations exist on this platform and they are not one authorization with a
wider setting. A tenant call reads one tenant's Sessions. A Control Plane call reads any
Session's Event Log. Keeping them apart is the whole content of this module: a reader
authorized differently is a second reason for the tenant surface to change, and folding
the two together is how a tenant boundary gets widened by accident.

A request arrives with a principal already established ahead of the router. Two claim
names are possible and this module treats them as disjoint: a platform reviewer is not a
tenant with more reach, and a tenant is not a reviewer with less.

`REVIEWER_CLAIM` is established by `control/api/request/reviewer_auth.py`, from a token
signed with a key only the control plane holds and tagged with an audience no Session
token carries. What that authenticator does on every failure is establish *nothing*, so
this module keeps refusing by default and every unproven request — no credential, a
tenant's own Session token, a forged signature, an expired one — reaches the same
refusal here. That direction matters on this surface more than on any other: a
permissive default would be an unauthenticated read of every tenant's history. The
tenant surface's own placeholder trusts an unauthenticated header
(`control/api/request/tenancy.py`); copying that here would have made any caller who
sets one header a reader of every tenant, which is a different order of mistake.

Refusing the both-claims case is the part that does work. A principal that were both
would let an audit read happen under a tenant's credential — the one thing this read is
defined by not doing — and it would do it invisibly, because the page it returned would
be indistinguishable from a legitimate one. So both claims together is an error rather
than a resolution in either direction, and neither claim is ever inferred from the
other.

Both refusals carry one code. Whether the caller tripped the missing-reviewer half or
the stray-tenant half is not something a caller branches on, and two codes would tell
someone probing this surface which half of the check they reached.

A PlatformReviewer carries an identifier and nothing else. Holding one confers no
ability to act as any tenant, because there is nothing in it to act with — which is the
structural half of the promise, the other half being that the router built on it
declares no write.

Framework-free on purpose: the parse below is a function of two claim values, so every
case it refuses is reachable in a test without constructing an HTTP request.
"""

from dataclasses import dataclass
from typing import Final
from uuid import UUID

REVIEWER_CLAIM: Final = "platform_reviewer_id"
"""The request-state name a platform reviewer is established under, ahead of the router.

Written by exactly one module, `control/api/request/reviewer_auth.py`, and by nothing
else -- a second writer would be a second way to become a reviewer, and the second one
would not be the one that checks a signature. That is asserted structurally rather than
left as a convention, because an extra writer is invisible in the diff that adds it.
"""

TENANT_CLAIM: Final = "tenant_id"
"""The request-state name an authenticated tenant will be established under.

Read here only in order to refuse it, and nothing establishes it yet — the tenant
surface parses an unauthenticated header instead. The reviewer authenticator will never
set it: a reviewer reads across every tenant and so names none, which is why a token
that proves one carries no tenant to establish. This check is therefore aimed at a
future authenticator that establishes both principals on one request, because that is
the change which would let an audit read happen under a tenant's credential.
"""

AUDIT_PRINCIPAL_UNRESOLVED: Final = "auth.audit_principal_unresolved"
"""The one refusal this component produces.

Deliberately not a member of the closed error set in `core/errors.py`. That set is the
published contract for what a call was refused *for*, and this says something else —
that the caller was never established as anyone — so it is spelled here, once, rather
than added to a set a caller is entitled to treat as complete. The tenant surface's own
authorization refusals (`request.tenant_missing`, `request.tenant_malformed`) sit
outside the set for the same reason and travel in the same `{"detail": {...}}` shape
(ADR-013).
"""


@dataclass(frozen=True, slots=True)
class PlatformReviewer:
    """A Control Plane reader: an identity, and no authority to act as anyone.

    One field, on purpose. Anything else it carried — a tenant, a token, a scope — would
    be a credential the audit path holds, and "holds no tenant credential" would then be
    a claim about how carefully every caller unpacks this rather than a property of the
    type. Frozen so a handler downstream cannot widen one it was handed.
    """

    reviewer_id: UUID


class AuditPrincipalRefused(Exception):
    """No platform reviewer was established, or one was established alongside a tenant.

    Carries a human-readable `reason` for whoever reads a log. The reason is not the
    contract and callers must not branch on it: the surface turns every instance of this
    into the same code, so the two halves of the check stay indistinguishable from
    outside.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def resolve_reviewer(reviewer_claim: object, tenant_claim: object) -> PlatformReviewer:
    """Parse the two claims into a reviewer, or refuse.

    Both parameters are `object` because they arrive from a request-state namespace that
    anything may write anything to; typing them narrowly would describe a guarantee that
    does not exist and would let a wrong type through unchecked. The reviewer claim is
    accepted only as a `UUID` — a string that merely looks like one is refused rather
    than parsed, because the thing that will set this claim is an authenticator, and an
    authenticator that hands over an unparsed string has not finished its job.

    The tenant check runs first and runs even when the reviewer claim is absent, so a
    request carrying a tenant and nothing else is refused for carrying a tenant rather
    than for lacking a reviewer. Both answers are the same code at the surface; order
    matters only in that neither claim can be satisfied by the presence of the other.
    """
    if tenant_claim is not None:
        raise AuditPrincipalRefused("this request carries a tenant principal")
    if not isinstance(reviewer_claim, UUID):
        raise AuditPrincipalRefused("this request carries no platform reviewer")
    return PlatformReviewer(reviewer_id=reviewer_claim)
