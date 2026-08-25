"""Holding tool credentials and attaching them on the outbound call.

The Tool Gateway runs outside the Session pod and is the only component that holds
a tool credential; the pod holds none (ADR-006). That is a property of where this
code runs, so nothing here tries to enforce it.

A fetched value becomes a `Secret`, and the only route out of one is `reveal`, which
is called twice in this file and nowhere else: once to write the value into an
environment mapping, once to write it into a header mapping. So a credential does not
reach a log line by being rendered into one, nor by a structured logger walking a
holder with `asdict`, nor by anything going through `pickle` -- `Secret` refuses all
three rather than relying on nobody trying. What it does not stop is
`getattr(secret, "_value")`, which is deliberate: no generic machinery does that by
accident, and a type cannot prevent code that reaches for the field by name.

What attaching means differs entirely by transport, which is why the two
attachments are two types with one method each rather than one type with a mode: a
stdio server is a child this service spawns, so its credential goes into that
child's environment, while a Streamable HTTP server is reached over the network, so
its credential goes into a request header. Handing a stdio attachment to the HTTP
branch does not type-check rather than half working.

The vault entry a registration names is composed under the tenant that registered
it -- see `vault_name`. Without that, `credential_ref` is text one tenant wrote
that can name another tenant's entry, and this service would fetch it.
"""

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, NoReturn

from managed_agent.core.ids import TenantId
from managed_agent.core.ports import CredentialVault
from managed_agent.core.registration.scope_binding import (
    StdioServer,
    StreamableHttpServer,
)
from managed_agent.core.vault_names import (
    TOOL_CREDENTIAL_PREFIX,
    VaultRefInvalid,
    scoped_vault_name,
)

VAULT_PREFIX: Final[str] = TOOL_CREDENTIAL_PREFIX
"""The prefix every entry this broker reads sits under, ahead of the tenant and the
ref. A ref that tries to climb out of it composes to a name no entry has.

An alias rather than the literal, since the control plane began writing entries this
reads: the two must compose the same string, so the string lives in `core.vault_names`
with the composition function and neither side owns it. The name is kept because it is
what this module's callers and tests already say, and because "the prefix this broker
reads" is a true thing to have a name for locally.
"""

HOLD_S: Final[float] = 300.0
"""How long a fetched value stays attachable.

It expires for one reason: a rotated or revoked credential has to stop being
attached without a deploy, and this is how long that takes.

**The window bounds attachment, not residency.** Nothing here runs at the instant a
window closes: there is no timer and no sweeper task, so an expired value is deleted
the next time this broker is asked for anything at all, and a process that reads once
and then goes idle holds what it read until it exits. The eviction covers every
expired entry rather than only the one being asked for, which is what keeps `_held`
bounded by what is in use rather than by everything ever read -- but that is a bound
on the dict, not a promise about when a revoked credential leaves memory.

Saying otherwise is what this docstring used to do, and a comment that states a
falsehood is a defect: a reader planning a revocation would have believed the value
was gone five minutes after rotation, when in fact only its attachment had stopped.
"""

VAULT_FETCH_TIMEOUT_S: Final[float] = 3.0
"""The deadline on one vault read.

The port declares no timeout and the adapter behind it does not build the client
that would carry one, so this is the only place a caller can bound the read. Three
seconds because the read happens inside the Gateway's own connection-startup
deadline (`mcp_proxy.GATEWAY_STARTUP_TIMEOUT_S`, measured at 5.0), and a bound that
is not strictly inside that one would never be the bound that fires. It is a literal
rather than an import because `mcp_proxy` imports this module and the reverse would be a
cycle; `tests/gateway/tool/test_credential_broker.py` compares the two values instead,
so the relation is asserted rather than restated in a comment.
"""


_Reason = Literal["malformed_ref", "unreadable", "unattachable"]
"""Why a registration's credential did not become an attachment.

Three cases and not a free string, so an arm that has to branch on one cannot be
written against a spelling nothing produces.
"""


class Secret:
    """A credential value that comes out through `reveal` and no other route.

    `__repr__` and `__str__` are both overridden because print, logging and
    f-strings reach them by different routes, and a value that hides from one and
    not the other is worse than one that hides from neither: it teaches a reader a
    rule that does not hold. An f-string goes through `object.__format__`, which
    defers to `__str__`, so the pair covers it.

    **Deliberately not a dataclass**, which is the whole reason this class is
    hand-written. Redacted rendering covers formatting, and formatting is not the
    only way a value reaches a log line: `dataclasses.asdict` and `astuple` walk a
    dataclass field by field and hand back the raw string, and a structured logger
    given an attachment does exactly that. `__reduce__` refuses for the same
    reason one layer down -- it is what `pickle` and `copy.deepcopy` both go
    through, so a Secret cannot be written to a cache or a queue either. What is
    left is `getattr(secret, "_value")`, which no generic machinery does by
    accident and no rule could prevent.

    `__slots__` rather than a `__dict__`, so `vars()` is a third walk that finds
    nothing.

    Equality is the identity default, so two Secrets holding one value compare
    unequal. Value equality would make this an oracle -- a holder could recover the
    value by comparing against guesses -- and nothing here keys anything by a
    credential. It stays hashable by identity, which discloses nothing.
    """

    __slots__ = ("_value",)

    _value: str

    def __init__(self, value: str) -> None:
        object.__setattr__(self, "_value", value)

    def reveal(self) -> str:
        """The value itself. Called in this file only, by the two attachments."""
        return self._value

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("a Secret is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("a Secret is immutable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("a Secret is not serialisable")

    def __repr__(self) -> str:
        return "Secret(<redacted>)"

    def __str__(self) -> str:
        return "<redacted>"


@dataclass(frozen=True, slots=True)
class StdioAttachment:
    """A credential bound for a spawned server's environment.

    `credential_ref` is carried so a refused attachment can name the registration
    line to change. It is the tenant's own text and holds no value.
    """

    credential_ref: str
    env_var: str
    secret: Secret

    def into_env(self, base: Mapping[str, str]) -> dict[str, str]:
        """A new environment for the child, carrying the credential.

        New rather than mutated: one child is spawned per registered stdio server,
        and an environment edited in place would carry one server's credential into
        the next spawn.

        A `base` that already names this variable is refused rather than
        overwritten. The base is a minimal environment built for this child, and
        the names in it are the ones a process needs to start. Measured, the SDK's
        `get_default_environment()` is exactly `HOME LOGNAME PATH SHELL TERM USER`,
        and every one of the six matches the pattern a registration's
        `credential_env_var` is validated against -- so a registration naming
        `PATH` would otherwise replace the child's search path with a credential
        and the child would fail to start for a reason unlike the cause.
        """
        if self.env_var in base:
            raise CredentialUnavailable(
                self.credential_ref,
                "unattachable",
                f"{self.env_var} is already set in the environment this child "
                f"would be given; a credential is never attached over a value",
            )
        return {**base, self.env_var: self.secret.reveal()}


@dataclass(frozen=True, slots=True)
class NoCredential:
    """What an HTTP server that authenticates nobody gets instead of a credential.

    A public MCP server is a real thing rather than a misconfiguration:
    `https://mcp.deepwiki.com/mcp` answers `initialize` over plain HTTPS with no header
    at all. Until this existed the registration model could not express one, because
    `credential_ref` was required -- so registering a public server meant naming a vault
    entry that had to be created and had to hold a value the far end ignores.

    A type rather than a `None` returned to the caller, so the one place that builds
    outbound headers keeps calling `into_headers` on whatever it was handed. A `None`
    there would put a branch on the credential path, and a branch on the credential path
    is a branch that can be written to skip the credential.

    Carries no `credential_ref`, because there is no registration text to name in a
    refusal: nothing here can fail.
    """

    def into_headers(self, base: Mapping[str, str]) -> dict[str, str]:
        """The request's headers, unchanged."""
        return dict(base)


@dataclass(frozen=True, slots=True)
class HttpAttachment:
    """A credential bound for an outbound request's headers.

    `credential_ref` is carried for the reason `StdioAttachment` carries one.
    """

    credential_ref: str
    header: str
    secret: Secret

    def into_headers(self, base: Mapping[str, str]) -> dict[str, str]:
        """New request headers carrying the credential.

        The vault entry holds the header value exactly as it goes on the wire, so a
        server wanting `Bearer <token>` has that stored. Nothing here derives a
        scheme from a header name: `Authorization` wants one and `X-Api-Key` does
        not, and guessing would produce a header that is wrong in a way only the far
        end can see.
        """
        if self.header in base:
            raise CredentialUnavailable(
                self.credential_ref,
                "unattachable",
                f"{self.header} is already set on this request; a credential is "
                f"never attached over an existing header",
            )
        return {**base, self.header: self.secret.reveal()}


class CredentialUnavailable(Exception):
    """The credential a registration named could not become an attachment.

    Three reasons, kept apart because what a reader does next differs: a ref that is
    not a well-formed name is a registration to fix, an unreadable entry is a vault
    to look at, and an attachment point already occupied is a registration to fix in
    a different line.

    `unattachable` is here rather than as a bare `ValueError` because of where the
    refusal ends up. `error_map` has an arm for this class and none for `ValueError`,
    so the collision refusal fell to `case _: return ErrorCode.INTERNAL` and every
    tool call under such a registration answered `platform.internal` -- paging the
    platform for a line only the tenant can change.

    Carries the ref and neither the value nor the composed key. The ref is text the
    tenant wrote, so echoing it discloses nothing; the composed key carries the
    tenant's own id and this message reaches a service log. The cause is chained, so
    the traceback keeps the diagnosis without this message quoting anything.

    `gateway/tool/error_map.py` has an arm for this class. Without one it would classify
    as `platform.internal`, which reports a tenant's own broken registration as a
    fault of the platform.
    """

    def __init__(self, credential_ref: str, reason: _Reason, detail: str = "") -> None:
        super().__init__(
            f"credential {credential_ref}: {reason}{f' -- {detail}' if detail else ''}"
        )
        self.credential_ref: str = credential_ref
        self.reason: _Reason = reason


def vault_name(tenant_id: TenantId, credential_ref: str) -> str:
    """The vault entry a tenant means when its registration names this ref.

    Composed under the tenant rather than trusted from the registration. Without
    that, one tenant could register a server naming another tenant's entry and the
    fetch would succeed; with it, such a ref composes to a name no entry has, so
    the escalation is not expressible rather than checked for.

    The composition itself is `core.vault_names.scoped_vault_name`, shared with the
    webhook signing path, because a second copy of it is a copy that can be
    weakened on one surface while a test on the other still passes. What this adds
    is the refusal type the Tool path needs: a malformed ref reaches a Session as
    `tool.unavailable`, which needs `CredentialUnavailable` and its reason, and
    `core/` has no business knowing that a Tool call is what asked.

    The ref's character set is checked at all because this is where it becomes a vault
    key -- `core/registration/scope_binding.py` accepts any non-empty string, which is
    right for a registration and not enough for a key.
    """
    try:
        return scoped_vault_name(VAULT_PREFIX, tenant_id, credential_ref)
    except VaultRefInvalid as invalid:
        raise CredentialUnavailable(credential_ref, "malformed_ref") from invalid


class ToolCredentialBroker:
    """Holds tool credentials and hands out attachments for the outbound call.

    One per process, shared by every Session the service serves. A connection to one
    registered server is opened once per Session -- `SessionUpstreams.session_for`
    caches it -- so a broker per Session would read the vault once per Session per
    server and hold nothing worth holding. Shared, one entry is read once per
    `HOLD_S` however many Sessions reach that server.

    Sharing across tenants is safe because the dict is keyed by the composed name,
    which carries the tenant: two tenants naming one ref are two entries here.

    A concurrent miss on one entry fetches twice and stores twice, and that is left
    alone. Both fetches read a value that was current when they read it, each caller
    attaches the value it read, and the second store costs one extra vault call --
    while a lock around the fetch would queue every server's opens behind whichever
    one missed.
    """

    def __init__(self, vault: CredentialVault, hold_s: float = HOLD_S) -> None:
        self._vault = vault
        self._hold_s = hold_s
        self._held: dict[str, tuple[float, Secret]] = {}

    async def for_stdio(
        self, tenant_id: TenantId, endpoint: StdioServer
    ) -> StdioAttachment:
        """The credential for a spawned server, bound to the variable it reads."""
        secret = await self._secret(tenant_id, endpoint.credential_ref)
        return StdioAttachment(
            endpoint.credential_ref, endpoint.credential_env_var, secret
        )

    async def for_http(
        self, tenant_id: TenantId, endpoint: StreamableHttpServer
    ) -> HttpAttachment | NoCredential:
        """The credential for an HTTP server, bound to the header it reads.

        A registration naming no `credential_ref` is a public server and gets
        `NoCredential`, which the caller uses the same way. The vault is not consulted
        at all in that case: reading it would be a call that can fail on behalf of a
        server that never needed it.
        """
        if endpoint.credential_ref is None:
            return NoCredential()
        secret = await self._secret(tenant_id, endpoint.credential_ref)
        return HttpAttachment(
            endpoint.credential_ref, endpoint.credential_header, secret
        )

    def _drop_expired(self, now: float) -> None:
        """Delete every entry whose window has closed, not only one.

        Deleted rather than merely refused: a value past its window that stays in
        this dict is a revoked credential the process is still holding.

        Every entry rather than the one being asked for, because the one being
        asked for is the one case that was never the problem -- it is about to be
        replaced by a fresh read. What lingers is an entry nobody asks for again,
        and with per-name eviction that is every rotated credential of every tenant
        that stopped using a server, held until the process exits while `_held`
        grows without bound behind it.

        A scan rather than a heap: one entry per tenant per server reached inside
        one window, so this is tens of keys walked on a path that is about to make
        a network call.
        """
        for name in [n for n, (until, _) in self._held.items() if until <= now]:
            del self._held[name]

    async def _secret(self, tenant_id: TenantId, credential_ref: str) -> Secret:
        name = vault_name(tenant_id, credential_ref)
        self._drop_expired(time.monotonic())
        held = self._held.get(name)
        if held is not None:
            return held[1]
        try:
            async with asyncio.timeout(VAULT_FETCH_TIMEOUT_S):
                raw = await self._vault.fetch(name)
        except Exception as exc:
            raise CredentialUnavailable(credential_ref, "unreadable") from exc
        secret = Secret(raw)
        # Stamped after the read returns, so a read that took two seconds does not
        # spend two seconds of the window it was fetched for.
        self._held[name] = (time.monotonic() + self._hold_s, secret)
        return secret
