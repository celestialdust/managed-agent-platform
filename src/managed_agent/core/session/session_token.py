"""The token a Session's pod presents to the Tool Gateway, minted and read back here.

One module for one wire format, because the two halves run in different services: the
control plane writes the token into a Session's compiled configuration and the Tool
Gateway reads it off every inbound request. Spelled separately in each package the
format has two definitions and no way to notice they have drifted -- and the drift is
invisible from both ends, because a token the Gateway cannot read is refused with the
same fixed 401 as a token nobody sent.

Here rather than under `gateway/tool/` because nothing under `control/` imports
`gateway/tool/`, and making it would pull FastAPI, Starlette and the MCP server library
into the control plane's import graph for three lines of HMAC.

**The token asserts an identity, not an authorization.** It says "I am this Session of
this tenant, until this second" and nothing else. The tool credential is still fetched
at the Gateway under the tenant this names and never enters the pod, and the registry
lookup a call is answered from is still the one keyed by that tenant. So what a pod
holds is a signature over its own two identifiers; what it does not hold is the key that
made it, and no pod can mint a token naming a tenant other than its own.

The reach is one tenant rather than one Session, and that is worth stating here because
the token names both. The Gateway resolves a tool against the tenant the token names and
consults no Grant on that path, so a token for one Session of a tenant reaches every
server that tenant registered. Narrowing that is a separate change to the Gateway; this
format does not widen it and cannot narrow it.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from managed_agent.core.ids import SessionId, TenantId

SESSION_TOKEN_HEADER: Final[bytes] = b"x-map-session"
"""The request header, in the bytes form an ASGI scope's header list carries."""

SESSION_TOKEN_HEADER_NAME: Final[str] = SESSION_TOKEN_HEADER.decode("ascii")
"""The same header as text, for the configuration document that names it as a key.

Derived rather than written twice: a document naming one spelling and a middleware
reading the other is a 401 on every call with nothing anywhere saying why.
"""

_PARTS: Final[int] = 4


@dataclass(frozen=True, slots=True)
class SessionContext:
    session_id: SessionId
    tenant_id: TenantId


class InvalidSessionToken(Exception):
    """The token was absent, malformed, not signed by us, or past its expiry."""


def mint_session_token(
    *,
    session_id: SessionId,
    tenant_id: TenantId,
    expiry_epoch_s: int,
    key: bytes,
) -> str:
    """Sign one Session's identity so the Tool Gateway will read it back.

    Four dot-separated parts: the Session, its tenant, the expiry in epoch seconds and
    the hex HMAC-SHA256 of the first three rejoined by dots. The signature covers the
    expiry, which is what lets the reader check the signature before it parses the
    expiry -- a presenter cannot supply their own deadline and have it believed.

    Everything emitted here is a legal HTTP header value by construction: two canonical
    UUIDs, decimal digits, dots and lowercase hex, 149 bytes for canonical ids. Nothing
    is escaped or quoted because nothing here can produce a byte that would need it, and
    that matters more than it looks -- the runtime carrying this header drops a value it
    cannot put in a header, warns into a log nobody reads, and sends the request anyway,
    which arrives as an ordinary 401.

    The expiry is absolute rather than a lifetime, and there is no clock in here, so two
    callers a second apart cannot mint tokens that differ in when they die. The caller
    owns the clock, exactly as the reader does.

    There is no refresh. A pod is started from a configuration compiled once and reads
    it once, so this expiry is a ceiling on how long that pod can use enterprise tools
    rather than a window that rolls forward.
    """
    parts = (str(session_id), str(tenant_id), str(expiry_epoch_s))
    signature = hmac.new(key, ".".join(parts).encode(), hashlib.sha256).hexdigest()
    return ".".join((*parts, signature))


def verify_session_token(token: str, key: bytes, now_epoch_s: int) -> SessionContext:
    """Read the Session and tenant out of a token the compiled configuration installed.

    Form: `<session uuid>.<tenant uuid>.<expiry epoch seconds>.<hex hmac-sha256 of the
    first three joined by dots>`. Verified with `compare_digest`, so a wrong signature
    costs the same time as a right one.

    Every failure raises the same exception with no distinguishing detail: a caller
    who learns *which* part was wrong learns whether a Session id exists. And the
    signature is checked before the expiry is parsed, so an attacker cannot supply their
    own expiry and have it believed on the way to being rejected for something else.
    """
    parts = token.split(".")
    if len(parts) != _PARTS:
        raise InvalidSessionToken("invalid session token")
    expected = hmac.new(key, ".".join(parts[:3]).encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, parts[3]):
        raise InvalidSessionToken("invalid session token")
    try:
        session_id = SessionId(UUID(parts[0]))
        tenant_id = TenantId(UUID(parts[1]))
        expiry = int(parts[2])
    except ValueError as exc:
        raise InvalidSessionToken("invalid session token") from exc
    if expiry <= now_epoch_s:
        raise InvalidSessionToken("invalid session token")
    return SessionContext(session_id=session_id, tenant_id=tenant_id)
