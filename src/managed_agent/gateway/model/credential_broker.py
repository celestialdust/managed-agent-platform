"""Fetching the upstream provider credential, inside the Model Gateway and nowhere else.

A credential is fetched on a named Session's behalf and cached under that Session, so
every fetch is attributable to one Session in the vault's own audit trail rather than to
a service account that fetches constantly. The cache expires because a vault entry can
be rotated under us: the longest a rotated credential can keep being presented is one
TTL.

Nothing here writes a credential anywhere. UpstreamCredential redacts itself under both
repr and str, which closes the ordinary route by which secrets reach logs -- an f-string
somebody added while debugging -- and the value is handed out per outbound call rather
than as a header block for a whole exchange, so a caller cannot capture one and keep
using it.
"""

import json
import logging
from dataclasses import dataclass
from typing import Final

from managed_agent.core.ids import SessionId
from managed_agent.core.ports import Clock, CredentialVault
from managed_agent.gateway.model.router import AuthScheme, GatewayRefusal, RoutingEntry

_LOG = logging.getLogger(__name__)

DEFAULT_CREDENTIAL_TTL_MS = 300_000

_CREDENTIAL_FIELD: Final = "api_key"
"""The field a JSON-shaped vault entry must carry the credential under.

One spelling, and a JSON entry that does not carry it is refused rather than searched.
An operator stores a provider entry one of two ways -- the bare credential, or a JSON
object because the console's own form encourages one -- and both are legitimate. What
is not legitimate is guessing: an entry with two plausible fields, or one field under a
name nobody declared, would have this module pick a value that goes on the wire as a
credential, and the only report of a wrong pick is the upstream's 401.
"""


@dataclass(frozen=True, slots=True, repr=False)
class UpstreamCredential:
    """A provider credential together with the header form its upstream wants."""

    scheme: AuthScheme
    secret: str

    def header(self) -> tuple[bytes, bytes]:
        """The one header this credential becomes on an outbound request, as octets.

        Octets because that is what a header block is, and because this one is written
        into a block of the caller's own octets: a `str` here would be the single value
        on that block whose encoding was chosen by an HTTP client rather than stated
        somewhere a reader can find it.

        ASCII is the codec, and it cannot fail here: `for_turn` refuses a vault value
        that is not ASCII before it ever becomes one of these, so the encode below is
        working on a string already proven to survive it. That check is there rather
        than here because the vault read is where a registration mistake can still be
        named -- `Bearer ` plus the secret is too late to say which entry was wrong.
        """
        return (
            self.scheme.header_name().encode("ascii"),
            self.scheme.header_value(self.secret).encode("ascii"),
        )

    def __repr__(self) -> str:
        return f"UpstreamCredential(scheme={self.scheme.value}, secret=<redacted>)"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True, slots=True, repr=False)
class _CachedSecret:
    """One vault read, held until it expires.

    The secret and nothing else. An earlier version cached the whole
    `UpstreamCredential`, which folded the auth scheme -- configuration this process
    already holds, per Routing Entry -- into a value keyed by the vault entry's name
    alone. Two entries sharing one name then served each other's header form: the second
    one's request went out under the first one's scheme, with no header of its own.
    Caching only what was actually read from the vault makes that unrepresentable rather
    than merely tested for.

    Redacted under repr for the same reason `UpstreamCredential` is: a dataclass's
    default repr would print the secret into whatever line somebody formats it into.
    """

    secret: str
    expires_at_ms: int

    def __repr__(self) -> str:
        return f"_CachedSecret(secret=<redacted>, expires_at_ms={self.expires_at_ms})"

    def __str__(self) -> str:
        return self.__repr__()


class ProviderCredentialBroker:
    """Hands the Model Gateway the credential a Routing Entry names."""

    def __init__(
        self,
        vault: CredentialVault,
        clock: Clock,
        *,
        ttl_ms: int = DEFAULT_CREDENTIAL_TTL_MS,
    ) -> None:
        if ttl_ms <= 0:
            raise ValueError(
                "ttl_ms must be positive; a cache that never expires is not one"
            )
        self._vault = vault
        self._clock = clock
        self._ttl_ms = ttl_ms
        self._cache: dict[tuple[SessionId, str], _CachedSecret] = {}

    async def for_turn(
        self, session_id: SessionId, entry: RoutingEntry
    ) -> UpstreamCredential:
        """The credential for this Session's next outbound call.

        The scheme comes from `entry` on every call and is never read out of the cache,
        so the header form is this Routing Entry's own even when another entry sharing
        its `credential_name` warmed the cache first. Two entries on one vault name is
        the ordinary case, not a corner: one upstream that accepts both
        `Authorization: Bearer` and `x-api-key` is exactly why an operator would point
        two models at one secret.

        Only the vault read is cached, which is the only thing here that costs anything.

        Two ways a vault entry fails to become a credential, and both leave here as a
        refusal rather than as whatever was raised underneath. A vault that will not
        answer -- no such entry, no permission to read it, no answer at all -- is a 503:
        the Agent Runtime reads `error.message` out of the refusal envelope and turns a
        non-2xx into a failed Turn, so an outage that escaped as an exception reached it
        as an unparsed body and left the operator a stack trace per request. And a value
        that is not ASCII cannot be a header, so it is a registration mistake in the
        same class as the non-string entry the vault adapter already refuses; caught
        here, the log line can still say which entry was wrong.

        Neither refusal names the entry or the model's upstream to the caller, and
        neither can carry the value: the entry's name is this service's configuration,
        and the value is the thing this whole module exists to keep out of a message.
        A failure is not cached, so a vault that is failing is asked again on the next
        Turn -- how long to remember a failure is a decision nobody has made, and
        guessing one here would silently extend an outage past its cause.
        """
        now = self._clock.now_epoch_ms()
        self._drop_expired(now)
        key = (session_id, entry.credential_name)
        cached = self._cache.get(key)
        if cached is None:
            cached = _CachedSecret(
                await self._read(entry.credential_name), now + self._ttl_ms
            )
            self._cache[key] = cached
        return UpstreamCredential(scheme=entry.auth_scheme, secret=cached.secret)

    async def _read(self, credential_name: str) -> str:
        """One vault entry's value, or a refusal naming neither it nor its contents."""
        try:
            secret = await self._vault.fetch(credential_name)
        except Exception as exc:
            _LOG.error(
                "provider credential %s could not be read: %r", credential_name, exc
            )
            raise GatewayRefusal(
                503, "server_error", "this service cannot reach that model right now"
            ) from exc
        credential = self._credential_in(secret, credential_name)
        if not credential.isascii():
            _LOG.error(
                "provider credential %s holds a value that cannot be a header",
                credential_name,
            )
            raise GatewayRefusal(
                503, "server_error", "this service cannot reach that model right now"
            )
        return credential

    def _credential_in(self, secret: str, credential_name: str) -> str:
        """The credential out of a vault entry, whichever of the two shapes it is.

        A vault entry holds either the bare credential or a JSON object carrying it
        under `api_key`. Both are shapes an operator really produces: the second is what
        the Secrets Manager console's key/value form writes, and the platform does not
        get to dictate which one a credential arrives in.

        The whole entry used to be taken as the credential, and that was a defect
        measured against the live Foundry deployment on 2026-08-23: the entry
        `map/dev/providers/anthropic` holds
        `{"api_key": ..., "base_url": ..., "model": ...}`, so the header carried the
        entire JSON document and the upstream answered **401 "invalid subscription key
        or wrong API endpoint"** -- a message naming a cause that was not the cause. The
        defect was invisible from inside the platform because nothing in the tree ever
        made a real request to that upstream.

        Only `api_key` is looked for, and a JSON object without it is refused rather
        than searched for something plausible. A JSON object also carrying `base_url`
        and `model` is *not* read for those: they are the routing table's to declare,
        and reading them here would give one fact two sources free to disagree. If the
        two do disagree, the routing table is what this service acts on and the entry's
        copy is decoration -- which is worth knowing, because on that date the entry's
        `base_url` was right and the routing table's did not resolve at all.

        A value that merely starts with `{` is a JSON object as far as this is
        concerned. That is deliberate: a bare credential beginning with a brace is not a
        shape any provider issues, and treating a malformed object as an opaque
        credential would send a broken JSON document to an upstream as a bearer.
        """
        if not secret.lstrip().startswith("{"):
            return secret
        try:
            document = json.loads(secret)
        except json.JSONDecodeError as exc:
            _LOG.error(
                "provider credential %s opens with '{' but is not JSON", credential_name
            )
            raise GatewayRefusal(
                503, "server_error", "this service cannot reach that model right now"
            ) from exc
        field = document.get(_CREDENTIAL_FIELD) if isinstance(document, dict) else None
        if not isinstance(field, str) or not field:
            _LOG.error(
                "provider credential %s is a JSON object with no %s string",
                credential_name,
                _CREDENTIAL_FIELD,
            )
            raise GatewayRefusal(
                503, "server_error", "this service cannot reach that model right now"
            )
        return field

    def _drop_expired(self, now_ms: int) -> None:
        """Expired entries go on the way past rather than by a sweeper.

        The cache is keyed per Session, so without this it would grow with the number of
        Sessions the process has ever served instead of the number active within one
        TTL -- and a stopped Session's credential would sit in memory for the process's
        life.
        """
        for key in [k for k, v in self._cache.items() if v.expires_at_ms <= now_ms]:
            del self._cache[key]
