"""Building, signing and delivering the one callback a lifecycle event earns.

A receiver has to answer two questions before acting: did this come from the platform,
and is it fresh. So the timestamp is signed along with the body and travels in its own
header -- a receiver checking only the body could be replayed a correctly-signed
callback for ever.

The body is built once, as bytes, and the same bytes are signed and sent. Serialising a
second time to sign would put two encoders in the path, and the first time they
disagreed about key order or unicode escaping every signature on the platform would fail
with nothing in a log to say why.

There is deliberately no `verify()` here. A receiver's check is a receiver's code, and a
copy of it shipped beside `signed_callback` would be dead in this codebase and would let
a test confirm a signature by re-running the bug that produced it. The tests recompute
the digest from the stdlib instead, which is the only version of that assertion worth
having.

Nothing in this module logs, prints or stores the signing secret. It is fetched from the
vault for one HMAC and is never placed in a payload, a header, an exception message or a
returned value.
"""

import hashlib
import hmac
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final, Protocol
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict

from managed_agent.control.webhooks.registry import WebhookRecord
from managed_agent.core.ids import Seq, SessionId, TenantId
from managed_agent.core.ports import CredentialVault
from managed_agent.core.vault_names import scoped_vault_name
from managed_agent.core.vocabulary import WEBHOOK_ELIGIBLE

WEBHOOK_SECRET_PREFIX: Final = "map/webhook-secret"
"""The segment every signing secret sits under, ahead of the tenant and the reference.

Its own prefix rather than the Tool credential's: a signing secret and a vendor token
are different kinds of thing, and one namespace would let a tenant sign its callbacks
under its own vendor token. Within one tenant, so not the cross-tenant read -- and a
confusion nothing needs.
"""

SIGNATURE_HEADER: Final = "x-managed-agent-signature"
TIMESTAMP_HEADER: Final = "x-managed-agent-timestamp"

SIGNATURE_SCHEME: Final = "v1"
"""Prefixed on every signature.

A later algorithm arrives beside this one under a second scheme name, so receivers
pinned to v1 keep verifying instead of failing every check at once on a day nobody
announced.
"""

SAFETY_LAG_MS: Final = 5_000
"""How far behind the present the window's leading edge sits.

Covers an append transaction -- one advisory lock, one select, one insert -- committing
after the tail read its timestamp, which is the only way a row is skipped rather than
merely re-read.
"""

MAX_WINDOW_MS: Final = 300_000
"""Widest span one pass reads.

Bounds the pass by time rather than by rows, so the watermark always advances by a
definite amount and a catch-up after an outage cannot livelock on a boundary where many
events share one timestamp.
"""

MAX_ATTEMPTS: Final = 5
RETRY_BATCH: Final = 100


class Callback(BaseModel):
    """What a callback says.

    The field list is closed and every member is something this platform issued: which
    registration fired, which Session, which event type happened, the sequence it
    happened at, and when this delivery was built. Nothing is read out of an event
    payload, a tool result or a model response, so there is no field a tool credential
    or an upstream token could travel in -- that is a property of this type rather than
    of a redaction pass somebody maintains.

    `event_type` is the type the event carries and not a state derived from it. A state
    is a fold over a whole log and two events can fold to the same one, so a callback
    naming a state cannot say which event caused it -- and the pair (session, sequence)
    below is what a receiver uses to go and read it.

    `seq` is here so the callback is useful without being large: a receiver reads the
    Event Log from that sequence, through the surface that authorizes it to, instead of
    being handed content over a channel that authorizes nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    webhook_id: UUID
    session_id: SessionId
    event_type: str
    seq: Seq
    delivered_at_ms: int


def signed_callback(
    secret: str, callback: Callback, at_ms: int
) -> tuple[bytes, dict[str, str]]:
    """The exact bytes to send, and the headers that authenticate them.

    Returns the body rather than taking one, so no caller can serialise again between
    signing and sending.

    The signed base string is the timestamp, a dot, then the body. The dot is not
    decoration: without it a timestamp of 1 with a body beginning "23" and a
    timestamp of 12 with a body beginning "3" produce the same bytes to sign,
    and two different callbacks under one valid signature is the whole failure
    mode a separator prevents.

    `secret` reaches nothing but `hmac.new`. It is not placed in the returned headers,
    is not part of the body, and is never rendered into a message.
    """
    body = callback.model_dump_json().encode()
    base = f"{at_ms}.".encode() + body
    digest = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return body, {
        "content-type": "application/json",
        TIMESTAMP_HEADER: str(at_ms),
        SIGNATURE_HEADER: f"{SIGNATURE_SCHEME}={digest}",
    }


class LifecycleCandidate(Protocol):
    """One deliverable event the tail found, with the tenant that owns the Session.

    Read-only members rather than plain annotations, for the reason `core.ports` gives
    for `EventRecord`: a plain annotation demands a settable attribute and so excludes
    every frozen implementation, which is what both the adapter's row and any honest
    test double are.

    `type` is on it because it is the whole answer the sweep needs from a candidate.
    Deriving it instead -- folding the Session's log and naming the state it arrived at
    -- was what this used to do, and it could not distinguish two events that fold to
    one state, which is exactly the pair a tenant most wants told apart.
    """

    @property
    def tenant_id(self) -> TenantId: ...

    @property
    def session_id(self) -> SessionId: ...

    @property
    def seq(self) -> Seq: ...

    @property
    def type(self) -> str: ...


class LifecycleScan(Protocol):
    """The cross-Session tail. One method, because one question is asked of the log."""

    async def lifecycle_events_between(
        self, types: Collection[str], from_ms: int, to_ms: int
    ) -> Sequence[LifecycleCandidate]:
        """Events of these types appended after `from_ms` and at or before `to_ms`.

        Half-open below and closed above, so consecutive windows cover the line between
        them exactly once.

        `types` is a `Collection` rather than a `Sequence` so the eligible set can be
        passed as the frozenset it is, with no order invented on the way in that would
        suggest one is meaningful.
        """
        ...


class WatchedWebhooks(Protocol):
    """The one question the sweep asks of the registrations.

    Narrower than `WebhookStore` on purpose: the dispatcher never registers, lists or
    deletes, so naming the wider port here would make every future method on it
    something the sweep formally depends on -- and would force a test double for the
    sweep to implement routes it never calls.
    """

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]: ...


class PendingDelivery(Protocol):
    """A claimed callback that has not been delivered and has attempts left."""

    @property
    def webhook_id(self) -> UUID: ...

    @property
    def tenant_id(self) -> TenantId: ...

    """Whose registration this is.

    Here so a retry composes its signing key the same way the first attempt did. A
    retry reads the registration off this row rather than off the store, so a tenant
    that did not travel on it would leave the second attempt naming an unscoped
    reference while the first was scoped.
    """

    @property
    def url(self) -> str: ...

    @property
    def secret_ref(self) -> str: ...

    @property
    def session_id(self) -> SessionId: ...

    @property
    def event_type(self) -> str: ...

    @property
    def seq(self) -> Seq: ...


class DeliveryLedger(Protocol):
    """The claim, the watermark, and the set still owed a callback."""

    async def scanned_through_ms(self) -> int:
        """The instant the tail has read through."""
        ...

    async def advance_scan_to(self, at_ms: int) -> None:
        """Move the watermark forward. Never moves it back, whoever calls it."""
        ...

    async def claim(
        self,
        webhook_id: UUID,
        session_id: SessionId,
        event_type: str,
        seq: Seq,
        max_attempts: int,
    ) -> int | None:
        """Take ownership of one attempt, or None when there is no attempt to take.

        None means the callback is already delivered, or its attempts are spent, or
        another dispatcher holds this attempt. One statement decides all three, because
        the case that matters is two dispatchers reaching the same event at once.

        A claim is identified by the registration, the Session and the sequence. The
        type travels with it because the row records what the callback said, but it is
        not what makes the claim unique: the sequence already is, and a type in the key
        would let one event be claimed twice under two spellings.
        """
        ...

    async def mark_delivered(
        self, webhook_id: UUID, session_id: SessionId, seq: Seq, status: int
    ) -> None:
        """Record that this callback landed, and stop it being retried."""
        ...

    async def undelivered(
        self, max_attempts: int, limit: int
    ) -> Sequence[PendingDelivery]:
        """Claimed callbacks still owed a delivery, fewest attempts first."""
        ...


@dataclass(frozen=True, slots=True)
class Delivery:
    """One attempt's outcome. `status` is None when the request got no answer at all.

    Carries the sequence as well as the type, because the type alone does not name one
    delivery: a Session can be suspended and resumed repeatedly, so a caller logging
    these would otherwise have several outcomes it could not tell apart.
    """

    webhook_id: UUID
    session_id: SessionId
    event_type: str
    seq: Seq
    delivered: bool
    status: int | None


@dataclass(frozen=True, slots=True)
class SweepRun:
    """What one pass did, so a caller can log it and decide whether to run again now."""

    delivered: tuple[Delivery, ...]
    failed: tuple[Delivery, ...]
    scanned_through_ms: int
    more_due: bool


_Attempted = tuple[UUID, SessionId, Seq]


class WebhookDispatcher:
    """Turns deliverable events into signed callbacks, one delivery each.

    Holds no schedule and reads no clock: `sweep_once` takes the instant, so whatever
    process runs it owns when it runs and there is no second answer to that question in
    the code.
    """

    def __init__(
        self,
        hooks: WatchedWebhooks,
        ledger: DeliveryLedger,
        scan: LifecycleScan,
        vault: CredentialVault,
        client: httpx.AsyncClient,
    ) -> None:
        self._hooks = hooks
        self._ledger = ledger
        self._scan = scan
        self._vault = vault
        self._client = client

    async def sweep_once(self, now_ms: int) -> SweepRun:
        """One pass: the new window, then whatever *earlier* attempts are still owed.

        "Earlier" is enforced rather than assumed. A callback attempted in this pass's
        window and refused by its receiver is still undelivered when the retry read
        runs, so without the set below one failing receiver would be posted to twice in
        a single pass and would burn its attempts at double rate.
        """
        frontier = now_ms - SAFETY_LAG_MS
        start = await self._ledger.scanned_through_ms()
        end = min(frontier, start + MAX_WINDOW_MS)
        outcomes: list[Delivery] = []
        attempted: set[_Attempted] = set()

        if end > start:
            for candidate in await self._scan.lifecycle_events_between(
                WEBHOOK_ELIGIBLE, start, end
            ):
                for hook in await self._hooks.watching(
                    candidate.tenant_id, candidate.type
                ):
                    attempted.add((hook.id, candidate.session_id, candidate.seq))
                    outcome = await self._attempt(
                        hook.id,
                        hook.tenant_id,
                        hook.url,
                        hook.secret_ref,
                        candidate.session_id,
                        candidate.type,
                        candidate.seq,
                        now_ms,
                    )
                    if outcome is not None:
                        outcomes.append(outcome)
            # After the window, not before: a crash mid-window leaves the watermark
            # where it was and the pass runs again over the same span, which the claim
            # absorbs. Moving it first would turn the same crash into callbacks nobody
            # ever gets.
            await self._ledger.advance_scan_to(end)

        for pending in await self._ledger.undelivered(MAX_ATTEMPTS, RETRY_BATCH):
            if (
                pending.webhook_id,
                pending.session_id,
                pending.seq,
            ) in attempted:
                continue
            outcome = await self._attempt(
                pending.webhook_id,
                pending.tenant_id,
                pending.url,
                pending.secret_ref,
                pending.session_id,
                pending.event_type,
                pending.seq,
                now_ms,
            )
            if outcome is not None:
                outcomes.append(outcome)

        return SweepRun(
            delivered=tuple(o for o in outcomes if o.delivered),
            failed=tuple(o for o in outcomes if not o.delivered),
            scanned_through_ms=max(end, start),
            more_due=end < frontier,
        )

    async def _attempt(
        self,
        webhook_id: UUID,
        tenant_id: TenantId,
        url: str,
        secret_ref: str,
        session_id: SessionId,
        event_type: str,
        seq: Seq,
        now_ms: int,
    ) -> Delivery | None:
        """Claim, sign and post one callback. None when this pass does not own the
        attempt.

        The claim runs before the secret is fetched and before anything is sent, so a
        registration nobody is owed costs no vault round trip and a losing dispatcher
        makes no request at all.

        `secret_ref` is text the registering tenant wrote, so it is composed under
        `tenant_id` rather than handed to the vault as a key. Handed over directly it
        names any entry this process can read, and what comes back is used as an HMAC
        key over a body and to a destination that same tenant chose -- an unrate-limited
        offline verification oracle on somebody else's credential.

        Every reason the secret does not arrive is one outcome: a spent attempt, no
        request made, the row left for the next pass. Three things ride on that being
        one outcome rather than several.

        It is *an* outcome, not an exception, so a single unreadable reference costs the
        tenant that wrote it and not the pass -- the caller reads it off
        `SweepRun.failed` exactly as it reads a receiver that was down, which is why
        swallowing here is not silence.

        It is *one* outcome, so this surface answers the same way for an entry that does
        not exist and an entry it may not read. The vault adapter distinguishes the two
        deliberately, and a caller able to tell them apart can enumerate every name in
        the vault a registration at a time.

        And it is one outcome for a malformed reference too, which is refused at the
        route as well. A row already in the store predates that check, so the refusal
        has to live where the key is composed and not only where it is registered.
        """
        if (
            await self._ledger.claim(
                webhook_id, session_id, event_type, seq, MAX_ATTEMPTS
            )
            is None
        ):
            return None
        try:
            secret = await self._vault.fetch(
                scoped_vault_name(WEBHOOK_SECRET_PREFIX, tenant_id, secret_ref)
            )
        except Exception:
            # Deliberately every exception, and deliberately not logged: the reason is
            # what a caller must not be able to tell apart, and a log line carrying it
            # would carry the composed name, which holds the tenant's own id.
            return Delivery(
                webhook_id, session_id, event_type, seq, delivered=False, status=None
            )
        body, headers = signed_callback(
            secret,
            Callback(
                webhook_id=webhook_id,
                session_id=session_id,
                event_type=event_type,
                seq=seq,
                delivered_at_ms=now_ms,
            ),
            now_ms,
        )
        try:
            response = await self._client.post(url, content=body, headers=headers)
        except httpx.HTTPError:
            # A receiver that is down is not a platform error and must not end the pass:
            # the attempt is spent, the row stays undelivered, and the next pass retries
            # it until MAX_ATTEMPTS. Every other webhook in this window still goes out.
            # The exception is swallowed rather than logged here because the caller gets
            # it as a value -- `SweepRun.failed` -- and a log line built from an httpx
            # error would carry the url, which is a tenant's.
            return Delivery(
                webhook_id, session_id, event_type, seq, delivered=False, status=None
            )
        if response.is_success:
            await self._ledger.mark_delivered(
                webhook_id, session_id, seq, response.status_code
            )
        return Delivery(
            webhook_id,
            session_id,
            event_type,
            seq,
            delivered=response.is_success,
            status=response.status_code,
        )
