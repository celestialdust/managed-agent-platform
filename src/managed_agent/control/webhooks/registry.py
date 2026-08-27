"""What a tenant registers to be called back, and what the store must offer.

Two rules on the destination, both because the platform is the one that will make the
call. A callback is an outbound HTTP request the control plane issues from the control
plane's own network position; no Egress Policy is anywhere near it, since that is a
per-Session rule bounding a sandboxed process inside a pod and nothing here runs in a
pod. So the only thing standing between a registered string and a request to an internal
address is `parse_callback_url`.

https only: the callback carries a signature over its body, and a plaintext hop would
hand an observer the body and the signature together, which is everything needed to
replay it.

No address literal naming the machine or the network the control plane stands on. A
tenant that registers `https://169.254.169.254/...` is asking the control plane to fetch
its own instance metadata; a tenant that registers an internal service is asking it to
reach something no tenant can reach, and that is a request the control plane would
otherwise honour, because it holds the network position to do it.

What this deliberately does not do is resolve the hostname. DNS is not consulted, so a
name that resolves to a blocked address passes, and a name that resolves to a public
address today can be re-pointed tomorrow -- there is no check that survives that, only a
network position that does. The literal check still earns its place: it costs nothing
and it turns the careless registration into a refusal a tenant can read instead of a
request nobody meant to make.

`CallbackUrl` is what the parse produces, and every function downstream names it rather
than `str`. That is the point of parsing here: the store cannot be handed a destination
that was never checked, because there is no way to spell one.
"""

import ipaddress
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Final, NewType, Protocol
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.ids import TenantId
from managed_agent.core.vault_names import MAX_REF_LEN, VaultRefInvalid, parse_vault_ref
from managed_agent.core.vocabulary import WEBHOOK_ELIGIBLE

CallbackUrl = NewType("CallbackUrl", str)
"""An https destination that named no blocked address literal when it was parsed."""

BLOCKED_HOSTS: Final = frozenset({"localhost", "metadata.google.internal"})
"""Names that reach a blocked address without being one.

Not a substitute for the literal check in `parse_callback_url` -- a complement to it,
covering the two spellings that are conventions rather than addresses.
"""


class WebhookInvalid(Exception):
    """The registration names something this platform will not accept."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class EventTypeInvalid(WebhookInvalid):
    """One requested type this platform will not deliver, carried as a value.

    A subclass rather than another reason string, so a caller can put the offending name
    into a refusal's `detail` under its own key without recovering it from a sentence --
    which is free text a reader may reword, and would take the detail with it.
    """

    def __init__(self, event_type: str) -> None:
        super().__init__(
            f"{event_type} is not an event type a webhook may subscribe to"
        )
        self.event_type = event_type


def parse_callback_url(raw: str) -> CallbackUrl:
    """Turn a registered string into a destination, or raise `WebhookInvalid`.

    `is_global` is what does the work on the literal branch: it is False for loopback,
    link-local (which is where cloud instance metadata lives), private, reserved and
    unspecified addresses in both families at once, so this carries no list of ranges
    that would go stale.

    A host that is not an address literal is returned as given. That is the DNS gap the
    module docstring names, and it is deliberate rather than overlooked.
    """
    parts = urlsplit(raw)
    if parts.scheme != "https":
        raise WebhookInvalid("a callback url must be https")
    host = parts.hostname
    if not host:
        raise WebhookInvalid("a callback url must name a host")
    if host.lower() in BLOCKED_HOSTS:
        raise WebhookInvalid(f"{host} is not a callback destination")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return CallbackUrl(raw)
    if not address.is_global:
        raise WebhookInvalid(f"{host} is not a callback destination")
    return CallbackUrl(raw)


def parse_secret_ref(raw: str) -> str:
    """Return the reference unchanged, or raise `WebhookInvalid`.

    A reference is not a vault key: the dispatcher composes one from it and the tenant
    that registered it, and a reference that cannot be composed reaches the vault as no
    name at all. Refusing it here means the tenant reads the refusal at registration
    rather than watching callbacks silently go undelivered for a reason only a platform
    log would hold.

    This is a second check of a rule the dispatcher also applies, and deliberately so:
    the store already holds rows registered before this existed, so the composition has
    to refuse them too. Both checks run the same function, so there is one rule and not
    a boundary and a core that can drift.

    Returns `str` rather than a `SecretRef` of its own. A NewType here would spell a
    guarantee this does not make -- that whatever holds one is safe to hand to the vault
    -- when the thing that is safe to hand to the vault is the composed name and nothing
    else.
    """
    try:
        return parse_vault_ref(raw)
    except VaultRefInvalid:
        raise WebhookInvalid(
            f"a secret reference is up to {MAX_REF_LEN} characters of letters, "
            "digits, underscore, dot, dash and slash, starting with a letter or digit"
        ) from None


MAX_EVENT_TYPE_LEN: Final = 128
"""How long one requested type name may be.

Here because the field it bounds used to be an enum, which bounded itself. A set of free
strings does not, and every refusal on this route echoes the offending value back -- so
without a cap a tenant could choose how many bytes their own refusal costs to build and
to log. 128 is far above the longest name this vocabulary has any use for; it is a
ceiling on an abuse, not a budget anybody is meant to spend.
"""


def parse_event_types(raw: frozenset[str]) -> frozenset[str]:
    """Return the set unchanged, or raise `EventTypeInvalid` naming one type refused.

    Eligibility is read off the vocabulary registry rather than listed here, so a family
    that becomes deliverable needs no edit on this side and a type that never was cannot
    be admitted by a list that fell behind.

    Both kinds of bad type are one refusal. A published type that is not deliverable
    (`turn.message_delta` arrives once per token) and a name that is not a type at all
    are different mistakes, but the tenant's next move is the same in both cases --
    write a different type -- and one answer is what keeps the refusal from reporting
    which spellings the platform recognises.

    One name and not all of them, so the refusal is bounded by `MAX_EVENT_TYPE_LEN`
    rather than by how many types the request happened to carry. It is the lowest in
    sort order, so a request refused twice is refused the same way twice.

    Returns `frozenset[str]` rather than a type of its own. A NewType here would claim
    the value is safe to hand to the delivery path, and what makes a delivery safe is
    the tail reading only eligible types -- not this parse, which only stops a
    registration nobody could ever fire.
    """
    refused = sorted(name for name in raw if name not in WEBHOOK_ELIGIBLE)
    if refused:
        raise EventTypeInvalid(refused[0])
    return raw


class RegisterWebhook(BaseModel):
    """What a tenant sends. The url and the reference are still raw here.

    The parse is not a field validator on purpose: a validator failure is answered as
    the generic `request.invalid`, and every refusal this surface publishes carries a
    code of its own from the closed set. So the route parses and refuses, and what it
    hands onward is a `CallbackUrl`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1, max_length=2048)
    event_types: frozenset[Annotated[str, Field(max_length=MAX_EVENT_TYPE_LEN)]] = (
        Field(min_length=1)
    )
    secret_ref: str = Field(min_length=1, max_length=256)


@dataclass(frozen=True, slots=True)
class WebhookRecord:
    """One registration as the store holds it.

    There is no field here for a secret and there is not going to be one. The signing
    material lives in the credential vault under `secret_ref`, so a dump of this table,
    a log line carrying this record and the read-back a tenant gets are all the same
    thing: a reference.
    """

    id: UUID
    tenant_id: TenantId
    url: CallbackUrl
    event_types: frozenset[str]
    secret_ref: str
    created_at_ms: int


class WebhookView(BaseModel):
    """What a read of a registration returns. Structurally the record minus its tenant.

    The field list is closed and `extra="forbid"` holds it closed, so a field able to
    carry signing material cannot be added to a tenant-visible read by accident.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    url: str
    event_types: tuple[str, ...]
    secret_ref: str
    created_at_ms: int

    @classmethod
    def of(cls, record: WebhookRecord) -> "WebhookView":
        """Types come back sorted, so two reads of one registration are identical."""
        return cls(
            id=record.id,
            url=record.url,
            event_types=tuple(sorted(record.event_types)),
            secret_ref=record.secret_ref,
            created_at_ms=record.created_at_ms,
        )


class WebhookStore(Protocol):
    """What this slice needs from whatever holds registrations.

    Declared here and satisfied structurally, so nothing under `control/` names a
    concrete store. `watching` is on this port rather than on the delivery ledger
    because it is a query over registrations, and keeping it here is what lets the sweep
    hold one read-only view of them.
    """

    async def register(
        self,
        tenant_id: TenantId,
        url: CallbackUrl,
        event_types: frozenset[str],
        secret_ref: str,
    ) -> WebhookRecord:
        """Write one registration and return it as stored, with the id it was given."""
        ...

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        """This tenant's registrations, oldest first. Never another tenant's."""
        ...

    async def delete(self, webhook_id: UUID, tenant_id: TenantId) -> bool:
        """Remove one registration.

        False when this tenant has no registration by that id -- the same answer for an
        id that never existed and one belonging to somebody else, so a caller cannot use
        this surface to learn which ids exist elsewhere.
        """
        ...

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]:
        """This tenant's registrations naming this type. Empty is the common case."""
        ...
