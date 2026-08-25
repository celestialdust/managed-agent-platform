"""Where a request's tenant comes from, for as long as nothing authenticates it.

**This is a placeholder and it trusts the caller completely.** The tenant is read from
the `X-Tenant-Id` request header, which anybody can set to anybody's id. Nothing signs
it, nothing checks it against a session or a key, and a caller who knows another
tenant's uuid can act as that tenant. It exists because the tenant-scoped tables and the
tenant-filtered reads are being built now and they need a tenant to key on, while the
authenticated claim that will supply one properly arrives later, with the auth
middleware that reads the tenant out of a signed token claim named `TENANT_CLAIM`
(planned as MAP-20).

Replacing it is meant to be a deletion, not an edit. Every route in this package reads
its tenant through this one function -- `test_tenancy.py` asserts that mechanically, by
scanning the package -- so removing it turns every call site into an import error rather
than into a route that quietly keeps trusting a header nobody checks any more.

An absent header is refused with 400 rather than defaulted. A default -- a constant, a
"first tenant", an empty uuid -- is the failure mode that survives the arrival of real
multi-tenancy: every call site keeps working and silently serves one tenant's data to
another. A refusal is loud at the one moment it is still cheap to fix.
"""

from uuid import UUID

from fastapi import Request

from managed_agent.control.api.refusals import Refusal
from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import TenantId

TENANT_HEADER = "X-Tenant-Id"
"""The header this placeholder reads. Named once so a test can assert it is the only
way a route in this package obtains a tenant."""


def unauthenticated_tenant_from_header(request: Request) -> TenantId:
    """Parse the tenant out of the request header, or refuse the request.

    Returns a `TenantId`, so a caller downstream holds a parsed value rather than a
    string it has to remember to validate. Raises `Refusal` -- 400,
    `request.tenant_missing` or `request.tenant_malformed` -- when the header is absent
    or is not a uuid. Both are the same kind of mistake, a caller that has not said who
    it is, and both are refused before anything is read or written. `Refusal` and not
    `HTTPException`, since 2026-08-24: a dependency cannot return a response, only
    raise, and the raised type is what carries the code into the one envelope.

    The name is deliberately ugly. It appears at every call site and in every stack
    trace, and it is the only warning a reader gets that the value it returns is
    asserted by whoever sent the request.
    """
    raw = request.headers.get(TENANT_HEADER)
    if raw is None:
        raise Refusal(ErrorCode.REQUEST_TENANT_MISSING, f"{TENANT_HEADER} is required")
    try:
        return TenantId(UUID(raw))
    except ValueError as malformed:
        raise Refusal(
            ErrorCode.REQUEST_TENANT_MALFORMED, f"{TENANT_HEADER} is not a uuid"
        ) from malformed
