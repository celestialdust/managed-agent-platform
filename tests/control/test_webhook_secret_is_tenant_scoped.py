"""The key a callback is signed under belongs to the tenant that registered it.

`secret_ref` is text a tenant wrote. Handed to the vault as-is it names any entry the
control plane can read, and the signature that comes back is an HMAC over a body the
registering tenant chose, posted to a url the registering tenant chose -- an offline
verification oracle on somebody else's credential. So the reference is composed under
the caller's tenant before it is a vault key, and the assertions here are about the name
the vault is *asked for* rather than about the value it answers with.

Every fake below records the names it was asked for. That is the whole point: a fake
that answered every name with one secret would let the cross-tenant read pass, because
the delivered bytes look identical either way.

No secret in this file is real. The one value is a literal placeholder generated in this
process, and it is never written to disk or into an assertion message.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import httpx

from managed_agent.control.webhooks.dispatcher import (
    SIGNATURE_HEADER,
    WEBHOOK_SECRET_PREFIX,
    SweepRun,
    WebhookDispatcher,
)
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import Seq, SessionId, TenantId, new_session_id
from managed_agent.core.vault_names import scoped_vault_name
from managed_agent.core.vocabulary import lifecycle
from managed_agent.gateway.tool.credential_broker import VAULT_PREFIX

SIGNING_PLACEHOLDER = "placeholder-whsec-" + secrets.token_hex(4)
"""A stand-in value, not a credential. Generated per run so it cannot be mistaken for
one that exists anywhere."""

_A_MOMENT = 1_000_000
"""An arbitrary instant, far enough above the sweep's safety lag that a window opens."""


@dataclass(frozen=True, slots=True)
class Row:
    """One event as the fold reads it. `payload` is empty and stays empty: nothing in
    this file asserts on delivered content, and an event body here would invite a reader
    to think one of these tests is about what a callback says."""

    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Candidate:
    tenant_id: TenantId
    session_id: SessionId
    seq: Seq
    type: str


@dataclass
class Log:
    """Sessions with their owning tenant, and the cross-Session tail the sweep reads."""

    owners: dict[SessionId, TenantId] = field(default_factory=dict)
    rows: dict[SessionId, list[Row]] = field(default_factory=dict)

    def stopped_session(self, tenant: TenantId) -> SessionId:
        session_id = new_session_id()
        self.owners[session_id] = tenant
        rows = self.rows.setdefault(session_id, [])
        for type_ in (lifecycle.SESSION_CREATED, lifecycle.SESSION_STOPPED):
            rows.append(Row(session_id, Seq(len(rows) + 1), type_))
        return session_id

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Row]:
        held = [r for r in self.rows.get(session_id, []) if start <= r.seq <= end]
        return held[:limit]

    async def lifecycle_events_between(
        self, types: Collection[str], from_ms: int, to_ms: int
    ) -> Sequence[Candidate]:
        return [
            Candidate(self.owners[sid], sid, row.seq, row.type)
            for sid, rows in self.rows.items()
            for row in rows
            if row.type in types
        ]


Claim = tuple[UUID, SessionId, Seq]


@dataclass(frozen=True, slots=True)
class Pending:
    """The retry join's row. Carries the tenant, because a retry composes a key too."""

    webhook_id: UUID
    tenant_id: TenantId
    url: str
    secret_ref: str
    session_id: SessionId
    event_type: str
    seq: Seq


@dataclass
class Store:
    """Registrations, the claim and the watermark, with the claim's real semantics."""

    hooks: list[WebhookRecord] = field(default_factory=list)
    attempts: dict[Claim, int] = field(default_factory=dict)
    delivered: dict[Claim, int] = field(default_factory=dict)
    pending: dict[Claim, Pending] = field(default_factory=dict)
    watermark: int = 0

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]:
        return [
            h
            for h in self.hooks
            if h.tenant_id == tenant_id and event_type in h.event_types
        ]

    async def scanned_through_ms(self) -> int:
        return self.watermark

    async def advance_scan_to(self, at_ms: int) -> None:
        self.watermark = max(self.watermark, at_ms)

    async def claim(
        self,
        webhook_id: UUID,
        session_id: SessionId,
        event_type: str,
        seq: Seq,
        max_attempts: int,
    ) -> int | None:
        key: Claim = (webhook_id, session_id, seq)
        if key in self.delivered or self.attempts.get(key, 0) >= max_attempts:
            return None
        self.attempts[key] = self.attempts.get(key, 0) + 1
        hook = next(h for h in self.hooks if h.id == webhook_id)
        self.pending[key] = Pending(
            webhook_id,
            hook.tenant_id,
            hook.url,
            hook.secret_ref,
            session_id,
            event_type,
            seq,
        )
        return self.attempts[key]

    async def mark_delivered(
        self, webhook_id: UUID, session_id: SessionId, seq: Seq, status: int
    ) -> None:
        key: Claim = (webhook_id, session_id, seq)
        self.delivered[key] = status
        self.pending.pop(key, None)

    async def undelivered(self, max_attempts: int, limit: int) -> Sequence[Pending]:
        owed = [
            row
            for key, row in self.pending.items()
            if key not in self.delivered and self.attempts[key] < max_attempts
        ]
        return owed[:limit]


@dataclass
class RecordingVault:
    """Answers the names it holds and records every name it was asked for.

    `asked` is the assertion surface for this whole file. `holds` is keyed by the
    *composed* name, so a fetch of an uncomposed reference misses and raises the way the
    real adapter's missing entry does.
    """

    holds: dict[str, str] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)

    async def fetch(self, name: str) -> str:
        self.asked.append(name)
        if name not in self.holds:
            raise KeyError(name)
        return self.holds[name]


@dataclass
class DenyingVault:
    """A vault that answers every name with a refusal rather than a miss."""

    asked: list[str] = field(default_factory=list)

    async def fetch(self, name: str) -> str:
        self.asked.append(name)
        raise PermissionError(f"not authorized to read {name}")


class Receiver:
    """A local in-process endpoint. Records every request it is posted. No socket."""

    def __init__(self) -> None:
        self.sent: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(request)
        return httpx.Response(200)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def _hook(tenant: TenantId, secret_ref: str, url: str) -> WebhookRecord:
    return WebhookRecord(
        id=uuid4(),
        tenant_id=tenant,
        url=CallbackUrl(url),
        event_types=frozenset({lifecycle.SESSION_STOPPED}),
        secret_ref=secret_ref,
        created_at_ms=0,
    )


def _sweep(
    store: Store,
    log: Log,
    vault: RecordingVault | DenyingVault,
    receiver: Receiver | None = None,
) -> SweepRun:
    client = (receiver or Receiver()).client()
    return asyncio.run(
        WebhookDispatcher(store, store, log, vault, client).sweep_once(_A_MOMENT)
    )


# --------------------------------------------------------------------------------
# The cross-tenant read.
# --------------------------------------------------------------------------------


def test_a_reference_naming_another_tenants_entry_is_never_the_name_fetched() -> None:
    """The reproduction. Tenant A registers a callback to a destination it controls and
    names tenant B's vault entry. B's entry must never be the name that is fetched."""
    attacker, victim = TenantId(uuid4()), TenantId(uuid4())
    victims_entry = f"map/tool-credential/{victim}/vendor/prod-token"

    vault = RecordingVault(holds={victims_entry: SIGNING_PLACEHOLDER})
    log = Log()
    log.stopped_session(attacker)
    store = Store(
        hooks=[_hook(attacker, victims_entry, "https://attacker.example/collect")]
    )
    receiver = Receiver()

    _sweep(store, log, vault, receiver)

    assert victims_entry not in vault.asked, (
        f"the dispatcher fetched an entry named by the victim's tenant id, which only "
        f"the victim may read: {vault.asked}"
    )
    assert all(str(attacker) in name for name in vault.asked), (
        f"{vault.asked} holds a name the calling tenant is not composed into, so a "
        "reference one tenant wrote is still reaching the vault unscoped"
    )
    assert not any(SIGNATURE_HEADER in r.headers for r in receiver.sent), (
        "a signature over the victim's credential was posted to the attacker"
    )


def test_the_composed_name_is_the_one_the_vault_is_asked_for() -> None:
    """Guard the guard. The assertions above are satisfied by a dispatcher that stopped
    fetching at all, and this is the only thing that tells the two apart."""
    tenant = TenantId(uuid4())
    log = Log()
    log.stopped_session(tenant)
    store = Store(hooks=[_hook(tenant, "signing-key", "https://hooks.example.com/x")])
    vault = RecordingVault(
        holds={f"map/webhook-secret/{tenant}/signing-key": SIGNING_PLACEHOLDER}
    )
    receiver = Receiver()

    _sweep(store, log, vault, receiver)

    assert vault.asked, "the dispatcher fetched nothing, so nothing here is checked"
    assert vault.asked == [f"map/webhook-secret/{tenant}/signing-key"], (
        f"asked for {vault.asked}, which is not the reference composed under the "
        "registering tenant beneath the webhook prefix"
    )
    assert len(receiver.sent) == 1, "the callback did not go out under the right name"


def test_a_reference_that_climbs_out_of_the_prefix_is_never_fetched() -> None:
    """A malformed reference is refused where the name is composed, so an escalation is
    inexpressible rather than checked for at the store."""
    tenant = TenantId(uuid4())
    for ref in ("../../other/entry", "a/../b", "/absolute/entry", "a b", "x" * 200):
        log = Log()
        log.stopped_session(tenant)
        store = Store(hooks=[_hook(tenant, ref, "https://hooks.example.com/x")])
        vault = RecordingVault()

        _sweep(store, log, vault)

        assert vault.asked == [], (
            f"{ref!r} composed to {vault.asked} and was fetched; a malformed "
            "reference is refused rather than passed through"
        )


# --------------------------------------------------------------------------------
# One tenant's bad reference costs that tenant, and nobody else.
# --------------------------------------------------------------------------------


def test_an_unreadable_reference_does_not_end_the_pass_for_other_tenants() -> None:
    """The fetch belongs inside the guarded region. Outside it, one bogus reference
    propagates out of `sweep_once` and every other tenant's callback goes undelivered
    for a reason that has nothing to do with its own registration."""
    broken, working = TenantId(uuid4()), TenantId(uuid4())
    log = Log()
    log.stopped_session(broken)
    log.stopped_session(working)
    store = Store(
        hooks=[
            _hook(broken, "no-such-entry", "https://a.example.com/x"),
            _hook(working, "signing-key", "https://b.example.com/x"),
        ]
    )
    vault = RecordingVault(
        holds={f"map/webhook-secret/{working}/signing-key": SIGNING_PLACEHOLDER}
    )
    receiver = Receiver()

    run = _sweep(store, log, vault, receiver)

    assert [str(r.url) for r in receiver.sent] == ["https://b.example.com/x"], (
        "the working tenant's callback did not go out, so one tenant's unreadable "
        "reference is still ending the pass for everybody"
    )
    assert len(run.failed) == 1, f"expected one failed delivery, got {run.failed}"
    assert run.failed[0].status is None, (
        "an unreadable reference reported a status, which means a request was made"
    )


def test_a_missing_entry_and_a_denied_one_are_one_outcome() -> None:
    """The vault adapter tells `KeyError` from a refusal on purpose, so a fetch this
    surface does not wrap makes it an existence oracle over every name in the vault."""
    tenant = TenantId(uuid4())

    def sweep(vault: RecordingVault | DenyingVault) -> SweepRun:
        log = Log()
        log.stopped_session(tenant)
        store = Store(
            hooks=[_hook(tenant, "signing-key", "https://hooks.example.com/x")]
        )
        return _sweep(store, log, vault)

    missing = sweep(RecordingVault())
    denied = sweep(DenyingVault())

    assert missing.failed and denied.failed, "neither case reached a delivery at all"
    assert [(d.delivered, d.status) for d in missing.failed] == [
        (d.delivered, d.status) for d in denied.failed
    ], (
        f"a missing entry gave {missing.failed} and a denied one gave {denied.failed}; "
        "a caller that can tell them apart can enumerate the vault"
    )


def test_no_tool_credential_can_be_reached_through_a_signing_reference() -> None:
    """A signing secret and a Tool credential sit under disjoint prefixes, so a
    reference registered for signing composes to a name in neither the registering
    tenant's Tool namespace nor anybody else's.

    Worth pinning because the delivered payload is not where this would show. A callback
    carries no credential either way; what a shared namespace would give away is the use
    of a Tool credential as the HMAC key, which is an unrate-limited offline oracle on
    it and leaves the payload looking exactly as it should.
    """
    tenant = TenantId(uuid4())
    composed = scoped_vault_name(WEBHOOK_SECRET_PREFIX, tenant, "signing-key")

    assert not composed.startswith(f"{VAULT_PREFIX}/"), (
        f"{composed} sits under the Tool credential prefix, so a signing reference can "
        "name a vendor token and have it used as an HMAC key"
    )
    assert not scoped_vault_name(VAULT_PREFIX, tenant, "x").startswith(
        f"{WEBHOOK_SECRET_PREFIX}/"
    ), "the two prefixes are not disjoint in the other direction either"


def test_the_retry_path_composes_under_the_tenant_too() -> None:
    """A retry reads the registration off the ledger's join rather than off the store,
    so the tenant has to travel on that row too or the second attempt is unscoped."""
    tenant = TenantId(uuid4())
    log = Log()
    session_id = log.stopped_session(tenant)
    store = Store(hooks=[_hook(tenant, "signing-key", "https://hooks.example.com/x")])
    hook = store.hooks[0]
    key: Claim = (hook.id, session_id, Seq(2))
    store.attempts[key] = 1
    store.pending[key] = Pending(
        hook.id,
        tenant,
        hook.url,
        hook.secret_ref,
        session_id,
        lifecycle.SESSION_STOPPED,
        Seq(2),
    )
    store.watermark = _A_MOMENT
    vault = RecordingVault(
        holds={f"map/webhook-secret/{tenant}/signing-key": SIGNING_PLACEHOLDER}
    )
    receiver = Receiver()

    _sweep(store, log, vault, receiver)

    assert vault.asked == [f"map/webhook-secret/{tenant}/signing-key"], (
        f"the retry asked for {vault.asked} rather than the composed name"
    )
    assert len(receiver.sent) == 1, "the retry did not deliver"
