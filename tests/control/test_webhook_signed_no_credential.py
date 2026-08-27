"""What leaves the platform when a Session appends an event somebody registered for.

A registration names event types, and the tail hands the dispatcher the event's own
type, so what a callback says about what happened is what the log said -- not a state
reconstructed from it. Two events that would fold to one state are two callbacks here,
which is the whole point of the key beneath them.

The credential fixtures are what make "no credential on the wire" checkable rather than
merely assertable: the Session's events carry a tool credential and an upstream
token in their payloads, the vault holds a third secret, and the tests scan the
delivered bytes for all three. Every secret here is generated in this process.

The delivery goes through a real `httpx.AsyncClient` on a `MockTransport` rather than a
poster fake, because the claim under test is about the bytes and headers that leave the
platform and a fake that recorded a call would let those be wrong while every assertion
passed. No test here opens a socket.

Digests are recomputed inline from `hmac` and `hashlib` rather than through
anything in the module under test: a signature checked by its own producer is
checked by whatever bug produced it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import json
import secrets
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.event_log_append import PostgresEventLogAppend
from managed_agent.adapters.postgres.event_log_range import (
    LifecycleRow,
    PostgresEventLogRange,
)
from managed_agent.adapters.postgres.session_registry import PostgresSessionRegistry
from managed_agent.adapters.postgres.webhook_store import PostgresWebhookStore
from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.dispatcher import (
    MAX_ATTEMPTS,
    MAX_WINDOW_MS,
    SAFETY_LAG_MS,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    Callback,
    DeliveryLedger,
    LifecycleCandidate,
    PendingDelivery,
    WatchedWebhooks,
    WebhookDispatcher,
    signed_callback,
)
from managed_agent.control.webhooks.registry import (
    MAX_EVENT_TYPE_LEN,
    CallbackUrl,
    RegisterWebhook,
    WebhookInvalid,
    WebhookRecord,
    WebhookStore,
    WebhookView,
    parse_callback_url,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import (
    Seq,
    SessionId,
    TenantId,
    new_definition_id,
    new_session_id,
)
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.session.projection import _TRANSITIONS
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vocabulary import WEBHOOK_ELIGIBLE, lifecycle, turn

TOOL_CREDENTIAL = "tool-cred-" + secrets.token_hex(8)
UPSTREAM_TOKEN = "sk-upstream-" + secrets.token_hex(8)
SIGNING_SECRET = "whsec-" + secrets.token_hex(8)
"""Generated per run and never written to disk, a log line or an assertion message.

Values rather than references because the point of two of them is to be *findable*: the
no-credential tests scan the delivered bytes for these exact strings, so they have to be
strings this process can compare against.
"""

_A_MOMENT = 1_000_000
"""An arbitrary instant, far enough above SAFETY_LAG_MS that a window opens at all."""


# --------------------------------------------------------------------------------
# Fakes. Every one of them is exercised by a test of its own before a delivery case
# leans on it, because a fake that lies is a green suite that proves nothing.
# --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FakeRow:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class FakeCandidate:
    tenant_id: TenantId
    session_id: SessionId
    seq: Seq
    type: str


@dataclass
class FakeLog:
    """One Event Log per Session, plus the cross-Session tail the dispatcher reads.

    There is no per-Session read on it, because the dispatcher has no reason to make
    one: the tail carries the event's type and its sequence, which is everything a
    callback says.
    """

    tenant: TenantId
    rows: dict[SessionId, list[FakeRow]] = field(default_factory=dict)

    def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        rows = self.rows.setdefault(session_id, [])
        seq = Seq(len(rows) + 1)
        rows.append(FakeRow(session_id, seq, type_, payload))
        return seq

    async def lifecycle_events_between(
        self, types: Collection[str], from_ms: int, to_ms: int
    ) -> Sequence[FakeCandidate]:
        # The window is ignored here on purpose: the adapter's half-open behaviour is
        # pinned against a real Postgres below, and mixing time into this fake would let
        # a delivery case fail for a reason that has nothing to do with delivery.
        return [
            FakeCandidate(self.tenant, row.session_id, row.seq, row.type)
            for rows in self.rows.values()
            for row in rows
            if row.type in types
        ]


Claim = tuple[UUID, SessionId, Seq]
"""What identifies one delivery: a registration, a Session, and where in its log.

Not the event's type. A Session is suspended and resumed as often as it likes, so a key
naming what happened rather than where would put two of those onto one row.
"""


@dataclass(frozen=True, slots=True)
class FakePending:
    """A claimed-but-undelivered callback, shaped like the retry join's row.

    The tenant is on it because the retry composes its own signing key from the tenant
    and the reference, exactly as the first attempt did.
    """

    webhook_id: UUID
    tenant_id: TenantId
    url: str
    secret_ref: str
    session_id: SessionId
    event_type: str
    seq: Seq


@dataclass
class FakeStore:
    """Registrations, the claim and the watermark, with the claim's real semantics.

    `claim` is written to the same rule as the SQL it stands in for -- an attempt is
    available only while the callback is undelivered and under the cap -- because every
    dedup and retry case in this file is a case about that rule, and a fake that merely
    counted calls would let all of them pass against a broken one.
    """

    hooks: list[WebhookRecord] = field(default_factory=list)
    attempts: dict[Claim, int] = field(default_factory=dict)
    delivered: dict[Claim, int] = field(default_factory=dict)
    pending: dict[Claim, FakePending] = field(default_factory=dict)
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
        self.pending[key] = FakePending(
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

    async def undelivered(self, max_attempts: int, limit: int) -> Sequence[FakePending]:
        owed = [
            row
            for key, row in self.pending.items()
            if key not in self.delivered and self.attempts[key] < max_attempts
        ]
        return owed[:limit]


@dataclass(frozen=True, slots=True)
class FakeVault:
    """Answers every reference with one secret. `fetches` is not counted here -- the
    test that cares about a vault round trip uses `ExplodingVault` instead, which cannot
    be satisfied by a miscount."""

    secret: str

    async def fetch(self, name: str) -> str:
        return self.secret


@dataclass(frozen=True, slots=True)
class ExplodingVault:
    """A vault that must never be reached."""

    async def fetch(self, name: str) -> str:
        raise AssertionError(
            "the dispatcher fetched a secret without owning an attempt"
        )


def _hook(
    tenant: TenantId,
    event_types: frozenset[str],
    url: str = "https://hooks.example.com/map",
) -> WebhookRecord:
    return WebhookRecord(
        id=uuid4(),
        tenant_id=tenant,
        url=CallbackUrl(url),
        event_types=event_types,
        secret_ref="webhook/1",
        created_at_ms=0,
    )


def _stopped_session(log: FakeLog) -> SessionId:
    """A Session that ran a tool and stopped, both credentials in its events."""
    session_id = new_session_id()
    log.append(session_id, lifecycle.SESSION_CREATED, {"definition_id": str(uuid4())})
    log.append(
        session_id,
        "tool.called",
        {"credential": TOOL_CREDENTIAL, "upstream_token": UPSTREAM_TOKEN},
    )
    log.append(session_id, lifecycle.SESSION_STOPPED, {})
    return session_id


def _twice_stopped_session(log: FakeLog) -> SessionId:
    """A Session created once and stopped twice: three events, one Session.

    The two stops are the reachable shape of "one state arrived at twice", and they are
    reachable rather than contrived: `control/session/lifecycle.py` documents the race
    that produces them, where two archive callers each fold before either appends and
    the second stop lands behind the first. Both fold to `STOPPED`, so a delivery keyed
    on the state a fold named can hold only one of them -- which is what this shape
    grades, and it grades the key rather than the tail.
    """
    session_id = new_session_id()
    log.append(session_id, lifecycle.SESSION_CREATED, {"definition_id": str(uuid4())})
    log.append(session_id, lifecycle.SESSION_STOPPED, {})
    log.append(session_id, lifecycle.SESSION_STOPPED, {})
    return session_id


class Recorder:
    """A local in-process endpoint. Records every request and answers as told."""

    def __init__(self, *answers: int) -> None:
        self.sent: list[httpx.Request] = []
        self._answers = list(answers) or [200]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(request)
        status = self._answers[min(len(self.sent) - 1, len(self._answers) - 1)]
        return httpx.Response(status)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def _dispatcher(
    store: FakeStore,
    log: FakeLog,
    client: httpx.AsyncClient,
    vault: FakeVault | ExplodingVault | None = None,
) -> WebhookDispatcher:
    return WebhookDispatcher(
        store, store, log, vault or FakeVault(SIGNING_SECRET), client
    )


# --------------------------------------------------------------------------------
# The fakes, held honest.
# --------------------------------------------------------------------------------


def test_the_fake_log_numbers_each_session_independently_from_one() -> None:
    log = FakeLog(tenant=TenantId(uuid4()))
    first, second = new_session_id(), new_session_id()

    assert log.append(first, "a", {}) == 1
    assert log.append(first, "b", {}) == 2
    assert log.append(second, "a", {}) == 1


async def test_the_fake_tail_returns_only_the_types_it_is_asked_for() -> None:
    log = FakeLog(tenant=TenantId(uuid4()))
    session_id = _stopped_session(log)

    found = await log.lifecycle_events_between(WEBHOOK_ELIGIBLE, 0, _A_MOMENT)

    assert [c.seq for c in found] == [1, 3], (
        "the tail returned the tool.called row, so a delivery test could pass for the "
        "wrong reason"
    )
    assert {c.session_id for c in found} == {session_id}


async def test_the_fake_claim_counts_up_and_stops_at_delivered_and_at_the_cap() -> None:
    tenant = TenantId(uuid4())
    hook = _hook(tenant, frozenset({lifecycle.SESSION_STOPPED}))
    store = FakeStore(hooks=[hook])
    session_id = new_session_id()

    async def take() -> int | None:
        return await store.claim(
            hook.id, session_id, lifecycle.SESSION_STOPPED, Seq(3), MAX_ATTEMPTS
        )

    assert await take() == 1
    assert await take() == 2
    await store.mark_delivered(hook.id, session_id, Seq(3), 200)
    assert await take() is None

    spent = FakeStore(hooks=[hook])
    taken = [
        await spent.claim(
            hook.id, session_id, lifecycle.SESSION_STOPPED, Seq(3), MAX_ATTEMPTS
        )
        for _ in range(MAX_ATTEMPTS + 1)
    ]
    assert taken == [*range(1, MAX_ATTEMPTS + 1), None]


async def test_the_fake_claim_keys_on_the_sequence_and_not_on_the_type() -> None:
    """The fake held to the rule the schema holds, before a delivery case leans on it.

    A fake keyed on the type would answer "already claimed" for the second of two
    events that share one, and every case below that reaches one state twice would go
    green over a dispatcher that delivers one callback where it owes two.
    """
    tenant = TenantId(uuid4())
    hook = _hook(tenant, frozenset({lifecycle.SESSION_STOPPED}))
    store = FakeStore(hooks=[hook])
    session_id = new_session_id()

    first = await store.claim(
        hook.id, session_id, lifecycle.SESSION_STOPPED, Seq(3), MAX_ATTEMPTS
    )
    again = await store.claim(
        hook.id, session_id, lifecycle.SESSION_STOPPED, Seq(7), MAX_ATTEMPTS
    )

    assert (first, again) == (1, 1)


# --------------------------------------------------------------------------------
# The parse at the boundary.
# --------------------------------------------------------------------------------


def test_a_public_https_destination_is_returned_unchanged() -> None:
    assert (
        parse_callback_url("https://hooks.example.com/x")
        == "https://hooks.example.com/x"
    )


def test_a_public_address_literal_is_accepted_rather_than_every_literal() -> None:
    assert parse_callback_url("https://93.184.216.34/x") == "https://93.184.216.34/x"


@pytest.mark.parametrize(
    "raw",
    [
        "http://hooks.example.com/x",
        "ftp://hooks.example.com/x",
        "hooks.example.com/x",
        "https://localhost/x",
        "https://LOCALHOST/x",
        "https://metadata.google.internal/computeMetadata/v1/",
        "https://127.0.0.1/x",
        "https://10.1.2.3/x",
        "https://192.168.0.5/x",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::1]/x",
        "https:///x",
    ],
)
def test_a_destination_this_platform_will_not_call_is_refused(raw: str) -> None:
    with pytest.raises(WebhookInvalid):
        parse_callback_url(raw)


def test_a_registration_refuses_a_stray_field_no_types_and_no_secret_ref() -> None:
    """`extra="forbid"` is what keeps a field carrying signing material out of the body.

    A model that ignored unknown fields would accept `secret` silently, and the tenant
    that sent one would believe the platform was holding it.
    """
    good: dict[str, Any] = {
        "url": "https://hooks.example.com/x",
        "event_types": [lifecycle.SESSION_STOPPED],
        "secret_ref": "signing-x",
    }
    parsed = RegisterWebhook.model_validate(good)
    assert parsed.event_types == frozenset({lifecycle.SESSION_STOPPED})

    for broken in (
        {**good, "secret": "hunter2"},
        {**good, "event_types": []},
        {**good, "secret_ref": ""},
    ):
        with pytest.raises(ValueError):
            RegisterWebhook.model_validate(broken)


def test_a_requested_type_longer_than_any_real_one_is_refused_by_the_model() -> None:
    """The bound the enum used to provide, restored now the field is free strings.

    Every refusal on this route echoes the offending value back, so an unbounded member
    here would let a tenant choose how many bytes their own refusal costs to build and
    to log. Refused at the model rather than in `parse_event_types`, because it is a
    property of the field's shape and holds whether or not the name is eligible.
    """
    good: dict[str, Any] = {
        "url": "https://hooks.example.com/x",
        "event_types": [lifecycle.SESSION_STOPPED],
        "secret_ref": "signing-x",
    }

    assert RegisterWebhook.model_validate(
        {**good, "event_types": ["e" * MAX_EVENT_TYPE_LEN]}
    ).event_types == frozenset({"e" * MAX_EVENT_TYPE_LEN}), (
        "a name at the limit was refused, so the bound is off by one"
    )
    with pytest.raises(ValueError):
        RegisterWebhook.model_validate(
            {**good, "event_types": ["e" * (MAX_EVENT_TYPE_LEN + 1)]}
        )


def test_a_stored_registration_is_frozen_and_names_no_secret_but_the_reference() -> (
    None
):
    record = _hook(TenantId(uuid4()), frozenset({lifecycle.SESSION_STOPPED}))

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.secret_ref = "other"  # type: ignore[misc]

    secret_ish = [
        name
        for name in record.__slots__
        if "secret" in name.lower() or "token" in name.lower()
    ]
    assert secret_ish == ["secret_ref"], (
        f"WebhookRecord carries {secret_ish}; the signing material lives in the vault "
        "under secret_ref and this record must stay a reference"
    )


def test_the_tenant_visible_read_has_five_fields_and_sorts_its_event_types() -> None:
    record = _hook(
        TenantId(uuid4()),
        frozenset({lifecycle.SESSION_STOPPED, lifecycle.SESSION_CREATED}),
    )

    view = WebhookView.of(record)

    assert view.event_types == (
        lifecycle.SESSION_CREATED,
        lifecycle.SESSION_STOPPED,
    )
    assert set(WebhookView.model_fields) == {
        "id",
        "url",
        "event_types",
        "secret_ref",
        "created_at_ms",
    }, "a field able to carry signing material was added to the tenant-visible read"


# --------------------------------------------------------------------------------
# The signature. The wrong-secret half is the one that fails on a no-op signer.
# --------------------------------------------------------------------------------


def _callback(**overrides: Any) -> Callback:
    base: dict[str, Any] = {
        "webhook_id": uuid4(),
        "session_id": new_session_id(),
        "event_type": lifecycle.SESSION_STOPPED,
        "seq": Seq(3),
        "delivered_at_ms": _A_MOMENT,
    }
    return Callback(**{**base, **overrides})


def test_the_body_is_the_callback_and_the_signature_covers_timestamp_and_body() -> None:
    callback = _callback()

    body, headers = signed_callback(SIGNING_SECRET, callback, _A_MOMENT)

    assert body == callback.model_dump_json().encode()
    expected = hmac.new(
        SIGNING_SECRET.encode(), f"{_A_MOMENT}.".encode() + body, hashlib.sha256
    ).hexdigest()
    assert headers[SIGNATURE_HEADER] == f"v1={expected}"
    assert headers[TIMESTAMP_HEADER] == str(_A_MOMENT)


def test_a_signature_computed_with_the_wrong_secret_does_not_verify() -> None:
    """The half that a no-op signer passes without.

    A `signed_callback` that ignored its secret entirely would satisfy every assertion
    in the test above, because that one recomputes the digest with the same secret it
    passed in. This one recomputes with a *different* secret and requires the two to
    differ, so a signature that does not actually depend on the key fails here.
    """
    callback = _callback()
    wrong_secret = "whsec-" + secrets.token_hex(8)

    _, headers = signed_callback(SIGNING_SECRET, callback, _A_MOMENT)
    body, _ = signed_callback(SIGNING_SECRET, callback, _A_MOMENT)

    forged = hmac.new(
        wrong_secret.encode(), f"{_A_MOMENT}.".encode() + body, hashlib.sha256
    ).hexdigest()
    assert headers[SIGNATURE_HEADER] != f"v1={forged}", (
        "a signature made with the registered secret verified against a different one, "
        "so the secret is not reaching the hmac"
    )
    assert hmac.compare_digest(
        headers[SIGNATURE_HEADER].removeprefix("v1="),
        hmac.new(
            SIGNING_SECRET.encode(), f"{_A_MOMENT}.".encode() + body, hashlib.sha256
        ).hexdigest(),
    )


def test_the_timestamp_and_every_byte_of_the_body_are_covered_by_the_signature() -> (
    None
):
    callback = _callback()
    _, at_one = signed_callback(SIGNING_SECRET, callback, _A_MOMENT)
    _, at_two = signed_callback(SIGNING_SECRET, callback, _A_MOMENT + 1)
    _, other_body = signed_callback(SIGNING_SECRET, _callback(seq=Seq(4)), _A_MOMENT)

    assert at_one[SIGNATURE_HEADER] != at_two[SIGNATURE_HEADER], (
        "only the timestamp changed and the signature did not, so a replay with "
        "a fresh timestamp would verify"
    )
    assert at_one[SIGNATURE_HEADER] != other_body[SIGNATURE_HEADER]


def test_the_callback_has_exactly_five_fields_and_refuses_anything_else() -> None:
    assert set(Callback.model_fields) == {
        "webhook_id",
        "session_id",
        "event_type",
        "seq",
        "delivered_at_ms",
    }
    with pytest.raises(ValueError):
        _callback(credential=TOOL_CREDENTIAL)
    with pytest.raises(ValueError):
        _callback(seq=0)


# --------------------------------------------------------------------------------
# The sweep.
# --------------------------------------------------------------------------------


# Types with a row in the projection's table that a tenant cannot put a callback on.
# Listed rather than derived, because each one is a decision somebody has to have taken,
# and the two halves of this set were decided for opposite reasons.
#
# The turn family began moving a Session's state when `RUNNING` came to mean "a Turn is
# executing", and making those three deliverable would add three types to the published
# webhook vocabulary -- which needs an API-version story rather than a `webhook=True`
# nobody discussed. They are candidates that have not been taken up.
#
# The two lifecycle types are the other case: they are not waiting on a decision, they
# are finished. A pod is leased for one Turn, so nothing suspends a Session and nothing
# resumes one (ADR-041). Their rows here and their declarations stay because the events
# a tenant's log already holds must keep folding and keep replaying; what a callback for
# either would buy is a delivery that is never coming.
#
# Neither half is silent -- `registry.py` refuses a registration naming an ineligible
# type and names it in the refusal, which is the cheap half of the trade `declare`
# describes.
_STATE_MOVING_BUT_NOT_DELIVERABLE = frozenset(
    {
        "turn.submitted",
        "turn.completed",
        "turn.failed",
        "session.suspended",
        "session.resumed",
    }
)


def test_every_state_moving_event_type_is_one_the_tail_reads() -> None:
    """The drift guard.

    A type that moves a Session's state and is not marked webhook-eligible would never
    be tailed, so the callback for it would simply never be sent -- with no error
    anywhere. Pinned against the projection's own table rather than against a list here,
    which is the only version of this assertion that can notice a new transition.

    It is one direction only, and deliberately: the tail is wider than this table --
    every eligible type is tailed whether or not it moves a state -- and requiring the
    two to be equal would refuse a deliverable event that leaves a Session where it was.

    The exception set is the second half of the guard rather than a hole in it. A new
    state-moving type has to be either deliverable or written down there with a reason,
    and either way somebody decides; what this refuses is the third option, where a type
    starts moving the state and nobody notices it cannot be subscribed to.
    """
    untailed = sorted(
        set(_TRANSITIONS) - WEBHOOK_ELIGIBLE - _STATE_MOVING_BUT_NOT_DELIVERABLE
    )
    assert untailed == [], (
        f"these event types move a Session's state and the webhook tail does not read "
        f"them: {untailed}. Declare them with webhook=True, or record them in "
        f"_STATE_MOVING_BUT_NOT_DELIVERABLE with the reason."
    )


def test_the_undeliverable_exceptions_are_all_really_state_moving() -> None:
    """The exception set cannot outlive what it excuses.

    A name left there after its type stopped moving the state -- or one that never did
    -- would quietly widen the guard's blind spot, so the set is checked against the
    projection's table in the other direction too.
    """
    stale = sorted(_STATE_MOVING_BUT_NOT_DELIVERABLE - set(_TRANSITIONS))
    assert stale == [], (
        f"these are excused from a rule they are not subject to: {stale}"
    )


async def test_a_stop_posts_one_callback_whose_signature_verifies() -> None:
    """One callback, counted -- not "at least one".

    The second sweep is what makes the count mean something: a dispatcher that delivered
    on every pass would satisfy "a callback was delivered" and fail here.
    """
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    session_id = _stopped_session(log)
    store = FakeStore(hooks=[_hook(tenant, frozenset({lifecycle.SESSION_STOPPED}))])
    recorder = Recorder(200)

    async with recorder.client() as client:
        dispatcher = _dispatcher(store, log, client)
        run = await dispatcher.sweep_once(now_ms=_A_MOMENT)
        again = await dispatcher.sweep_once(now_ms=_A_MOMENT + 60_000)

    assert len(recorder.sent) == 1, (
        f"one event produced {len(recorder.sent)} callbacks; the promise is exactly one"
    )
    assert (len(run.delivered), len(again.delivered)) == (1, 0)

    request = recorder.sent[0]
    payload = json.loads(request.content)
    assert payload["session_id"] == str(session_id)
    assert payload["event_type"] == lifecycle.SESSION_STOPPED
    assert payload["seq"] == 3

    timestamp = request.headers[TIMESTAMP_HEADER]
    expected = hmac.new(
        SIGNING_SECRET.encode(),
        f"{timestamp}.".encode() + request.content,
        hashlib.sha256,
    ).hexdigest()
    assert request.headers[SIGNATURE_HEADER] == f"v1={expected}"


async def test_no_credential_reaches_the_wire() -> None:
    """The Session's own events hold both credentials; the callback holds neither.

    Asserted over the bytes the transport was handed, not over the object that was
    built: a secret reaching the wire through a nested model or a `model_dump()`
    is still on the wire.
    """
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(hooks=[_hook(tenant, frozenset({lifecycle.SESSION_STOPPED}))])
    recorder = Recorder(200)

    async with recorder.client() as client:
        await _dispatcher(store, log, client).sweep_once(now_ms=_A_MOMENT)

    request = recorder.sent[0]
    # Headers as well as the body: a signing secret echoed into a header would satisfy
    # an assertion about the payload alone, and the claim is about what was delivered.
    wire = b"\n".join(
        [request.content, *(f"{k}: {v}".encode() for k, v in request.headers.items())]
    )
    for secret in (TOOL_CREDENTIAL, UPSTREAM_TOKEN, SIGNING_SECRET):
        assert secret.encode() not in wire, "a secret reached the callback"
    assert set(json.loads(request.content)) == {
        "webhook_id",
        "session_id",
        "event_type",
        "seq",
        "delivered_at_ms",
    }
    assert request.url.host == "hooks.example.com"


async def test_another_event_type_and_another_tenant_are_both_passed_over() -> None:
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(
        hooks=[
            _hook(tenant, frozenset({lifecycle.SESSION_SUSPENDED})),
            _hook(TenantId(uuid4()), frozenset({lifecycle.SESSION_STOPPED})),
        ]
    )
    recorder = Recorder(200)

    async with recorder.client() as client:
        run = await _dispatcher(store, log, client).sweep_once(now_ms=_A_MOMENT)

    assert recorder.sent == []
    assert run.delivered == ()


async def test_two_watched_event_types_get_one_callback_each_and_no_more() -> None:
    """The Session is created and then stops. Two events, two callbacks, and the
    second sweep adds none."""
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(
        hooks=[
            _hook(
                tenant,
                frozenset({lifecycle.SESSION_CREATED, lifecycle.SESSION_STOPPED}),
            )
        ]
    )
    recorder = Recorder(200)

    async with recorder.client() as client:
        dispatcher = _dispatcher(store, log, client)
        await dispatcher.sweep_once(now_ms=_A_MOMENT)
        await dispatcher.sweep_once(now_ms=_A_MOMENT + 60_000)

    delivered = [json.loads(r.content)["event_type"] for r in recorder.sent]
    assert sorted(delivered) == [
        lifecycle.SESSION_CREATED,
        lifecycle.SESSION_STOPPED,
    ]


async def test_one_state_reached_twice_is_called_back_for_each_event() -> None:
    """The reason this slice exists, at the surface a tenant actually sees.

    The Session is created and then stopped twice. Both stops fold to `STOPPED`, so a
    delivery keyed on the state the platform folded to could hold only one of them: the
    second would land on the first's row and be answered as a second attempt at a
    callback already delivered. The tenant would hear about one ending and never about
    the other, and no counter anywhere would move.

    Keyed on the sequence, the three events are three callbacks, each naming its own
    sequence -- and a second sweep adds none, so this is not a dispatcher that simply
    posts on every pass.

    This case used to be built from a create, a suspend and a resume, where the create
    and the resume were the pair that collided. Nothing suspends or resumes a Session
    any more (ADR-041), and neither type is deliverable, so the collision is now the one
    two stops make -- the same defect, reached through the race `_end_and_release`
    documents rather than through a pod's life.
    """
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    session_id = _twice_stopped_session(log)
    store = FakeStore(
        hooks=[
            _hook(
                tenant,
                frozenset({lifecycle.SESSION_CREATED, lifecycle.SESSION_STOPPED}),
            )
        ]
    )
    recorder = Recorder(200)

    async with recorder.client() as client:
        dispatcher = _dispatcher(store, log, client)
        await dispatcher.sweep_once(now_ms=_A_MOMENT)
        await dispatcher.sweep_once(now_ms=_A_MOMENT + 60_000)

    delivered = [json.loads(r.content) for r in recorder.sent]
    assert [(body["event_type"], body["seq"]) for body in delivered] == [
        (lifecycle.SESSION_CREATED, 1),
        (lifecycle.SESSION_STOPPED, 2),
        (lifecycle.SESSION_STOPPED, 3),
    ], (
        f"the tenant was told about {[b['seq'] for b in delivered]}. A second ending "
        "is a separate event at a separate sequence and owes a separate callback."
    )
    assert {body["session_id"] for body in delivered} == {str(session_id)}


async def test_a_window_inside_the_safety_lag_reads_and_moves_nothing() -> None:
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(
        hooks=[_hook(tenant, frozenset({lifecycle.SESSION_STOPPED}))],
        watermark=_A_MOMENT,
    )
    recorder = Recorder(200)

    async with recorder.client() as client:
        run = await _dispatcher(store, log, client).sweep_once(
            now_ms=_A_MOMENT + SAFETY_LAG_MS - 1
        )

    assert recorder.sent == []
    assert store.watermark == _A_MOMENT
    assert run.scanned_through_ms == _A_MOMENT


async def test_a_watermark_far_behind_advances_by_one_window_and_says_more_is_due() -> (
    None
):
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    store = FakeStore(watermark=_A_MOMENT)
    recorder = Recorder(200)
    now = _A_MOMENT + SAFETY_LAG_MS + 3_600_000

    async with recorder.client() as client:
        run = await _dispatcher(store, log, client).sweep_once(now_ms=now)

    assert run.scanned_through_ms == _A_MOMENT + MAX_WINDOW_MS
    assert run.more_due is True


async def test_a_lost_claim_costs_no_vault_read_and_no_request() -> None:
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    session_id = _stopped_session(log)
    hook = _hook(tenant, frozenset({lifecycle.SESSION_STOPPED}))
    store = FakeStore(hooks=[hook])
    # Another dispatcher already delivered this exact callback.
    store.delivered[(hook.id, session_id, Seq(3))] = 200
    recorder = Recorder(200)

    async with recorder.client() as client:
        run = await _dispatcher(store, log, client, ExplodingVault()).sweep_once(
            now_ms=_A_MOMENT
        )

    assert recorder.sent == []
    assert run.delivered == ()


async def test_an_unreachable_receiver_leaves_its_row_owed_and_the_pass_running() -> (
    None
):
    """A dead receiver is not a platform error, and must not stop the others."""
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(
        hooks=[
            _hook(
                tenant,
                frozenset({lifecycle.SESSION_STOPPED}),
                "https://down.example.com/x",
            ),
            _hook(
                tenant,
                frozenset({lifecycle.SESSION_STOPPED}),
                "https://up.example.com/x",
            ),
        ]
    )
    reached: list[str] = []

    def receive(request: httpx.Request) -> httpx.Response:
        reached.append(str(request.url.host))
        if request.url.host == "down.example.com":
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
        run = await _dispatcher(store, log, client).sweep_once(now_ms=_A_MOMENT)

    assert sorted(reached) == ["down.example.com", "up.example.com"]
    assert [d.status for d in run.failed] == [None]
    assert [d.delivered for d in run.delivered] == [True]
    assert len(await store.undelivered(MAX_ATTEMPTS, 100)) == 1


async def test_a_refusing_receiver_is_retried_once_per_pass_and_stops_at_the_cap() -> (
    None
):
    """One attempt per pass, and no more than MAX_ATTEMPTS in total.

    The per-pass count is the load-bearing half. The window loop and the retry loop both
    see this callback on the first pass, so without the dispatcher's own guard a failing
    receiver would be posted to twice in one sweep and burn its attempts at double rate.
    """
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(hooks=[_hook(tenant, frozenset({lifecycle.SESSION_STOPPED}))])
    recorder = Recorder(500)

    async with recorder.client() as client:
        dispatcher = _dispatcher(store, log, client)
        for pass_number in range(1, MAX_ATTEMPTS + 3):
            await dispatcher.sweep_once(now_ms=_A_MOMENT + 60_000 * pass_number)
            assert len(recorder.sent) == min(pass_number, MAX_ATTEMPTS), (
                f"after {pass_number} passes the receiver had been posted to "
                f"{len(recorder.sent)} times"
            )

    assert len(recorder.sent) == MAX_ATTEMPTS


async def test_a_receiver_that_answers_late_is_not_posted_to_again() -> None:
    """Fails, then succeeds. The success ends it -- the next pass sends nothing."""
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(hooks=[_hook(tenant, frozenset({lifecycle.SESSION_STOPPED}))])
    recorder = Recorder(503, 200)

    async with recorder.client() as client:
        dispatcher = _dispatcher(store, log, client)
        for pass_number in range(1, 5):
            await dispatcher.sweep_once(now_ms=_A_MOMENT + 60_000 * pass_number)

    assert len(recorder.sent) == 2


# --------------------------------------------------------------------------------
# The routes.
# --------------------------------------------------------------------------------


@dataclass
class InMemoryWebhooks:
    """A store with the real one's tenant scoping, so a route test can grade scoping."""

    rows: list[WebhookRecord] = field(default_factory=list)

    async def register(
        self,
        tenant_id: TenantId,
        url: CallbackUrl,
        event_types: frozenset[str],
        secret_ref: str,
    ) -> WebhookRecord:
        record = WebhookRecord(
            id=uuid4(),
            tenant_id=tenant_id,
            url=url,
            event_types=event_types,
            secret_ref=secret_ref,
            created_at_ms=len(self.rows),
        )
        self.rows.append(record)
        return record

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        return [r for r in self.rows if r.tenant_id == tenant_id]

    async def delete(self, webhook_id: UUID, tenant_id: TenantId) -> bool:
        keep = [
            r
            for r in self.rows
            if not (r.id == webhook_id and r.tenant_id == tenant_id)
        ]
        removed = len(keep) != len(self.rows)
        self.rows = keep
        return removed

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]:
        return [
            r
            for r in self.rows
            if r.tenant_id == tenant_id and event_type in r.event_types
        ]


class UnusedPort:
    """Every other port on the Platform. A route test that reached one is a bug."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"a webhook route reached {name} on another port")


def _platform(store: WebhookStore) -> Platform:
    unused: Any = UnusedPort()
    return Platform(
        event_log_append=unused,
        event_log_range=unused,
        definition_registry=unused,
        tool_registry=unused,
        session_registry=unused,
        webhooks=store,
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
    )


def _client(store: WebhookStore) -> TestClient:
    return TestClient(create_app(_platform(store)))


def _as(tenant: TenantId) -> dict[str, str]:
    return {TENANT_HEADER: str(tenant)}


def test_registering_returns_201_and_an_id_the_caller_did_not_send() -> None:
    store = InMemoryWebhooks()
    client = _client(store)
    tenant = TenantId(uuid4())

    response = client.post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "event_types": ["session.stopped"],
            "secret_ref": "signing-x",
        },
        headers=_as(tenant),
    )

    assert response.status_code == 201
    view = WebhookView.model_validate(response.json())
    assert view.id == store.rows[0].id
    assert view.event_types == (lifecycle.SESSION_STOPPED,)


def test_a_plaintext_destination_is_refused_by_code_and_writes_nothing() -> None:
    store = InMemoryWebhooks()

    response = _client(store).post(
        "/v1/webhooks",
        json={
            "url": "http://hooks.example.com/x",
            "event_types": ["session.stopped"],
            "secret_ref": "signing-x",
        },
        headers=_as(TenantId(uuid4())),
    )

    assert response.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    refusal = PublicErrorEnvelope.model_validate(response.json()).error
    assert refusal.code is ErrorCode.REQUEST_INVALID
    assert store.rows == []


def test_a_reference_that_is_not_a_vault_name_is_refused_at_registration() -> None:
    """The reference becomes a vault key only once the tenant is composed in front of
    it, so one that cannot be composed names no entry. Refused here rather than at
    delivery, where the tenant would see callbacks silently never arrive and the reason
    would sit in a platform log it cannot read.

    The refusal echoes the reference and not the url, so it names the field to change.
    """
    for ref in ("vault://x", "../../other", "has space", "", "x" * 300):
        store = InMemoryWebhooks()

        response = _client(store).post(
            "/v1/webhooks",
            json={
                "url": "https://hooks.example.com/x",
                "event_types": ["session.stopped"],
                "secret_ref": ref,
            },
            headers=_as(TenantId(uuid4())),
        )

        assert response.status_code in {
            STATUS_FOR[ErrorCode.REQUEST_INVALID],
            422,
        }, f"{ref!r} was accepted with {response.status_code}"
        assert store.rows == [], f"{ref!r} was written to the store"

    accepted = InMemoryWebhooks()
    assert (
        _client(accepted)
        .post(
            "/v1/webhooks",
            json={
                "url": "https://hooks.example.com/x",
                "event_types": ["session.stopped"],
                "secret_ref": "vendor/prod-token.v2",
            },
            headers=_as(TenantId(uuid4())),
        )
        .status_code
        == 201
    ), "a well-formed reference was refused, which is a check somebody deletes"


def test_a_refused_reference_names_itself_and_not_the_url() -> None:
    """Two fields are parsed at this route and each refusal has to say which one failed,
    or a tenant reads a refusal naming a url it got right."""
    store = InMemoryWebhooks()

    response = _client(store).post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "event_types": ["session.stopped"],
            "secret_ref": "vault://x",
        },
        headers=_as(TenantId(uuid4())),
    )

    detail = PublicErrorEnvelope.model_validate(response.json()).error.detail
    assert "secret_ref" in detail, f"the refusal names {sorted(detail)}"
    assert "url" not in detail, "the refusal blames the url, which was well-formed"


def test_a_registration_naming_an_ineligible_type_is_refused_and_names_it() -> None:
    """`turn.message_delta` is published, and it is not something to post.

    It arrives once per token. A registration for it would put one delivery row and one
    outbound request through the ledger per token generated, on the platform's retry
    budget and at somebody's endpoint -- which is discovered by the receiver rather than
    by the tenant who asked for it.

    The refusal names the type, because a request carrying several is otherwise a
    refusal the tenant cannot act on: they are told the registration was rejected and
    left to work out which of them did it.
    """
    store = InMemoryWebhooks()

    response = _client(store).post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "event_types": [turn.TURN_MESSAGE_DELTA],
            "secret_ref": "signing-x",
        },
        headers=_as(TenantId(uuid4())),
    )

    assert response.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    refusal = PublicErrorEnvelope.model_validate(response.json()).error
    assert refusal.code is ErrorCode.REQUEST_INVALID
    assert turn.TURN_MESSAGE_DELTA in json.dumps(refusal.detail), (
        f"the refusal detail is {refusal.detail} and does not name the type that was "
        "refused"
    )
    assert store.rows == [], "an ineligible type was written to the store"


def test_a_registration_naming_a_type_that_does_not_exist_is_refused() -> None:
    """A misspelling is refused rather than stored and never fired.

    Accepted, it is a registration that matches no event for the life of the
    platform -- and the tenant's evidence for that is silence, which is
    indistinguishable from a platform that is not delivering.
    """
    store = InMemoryWebhooks()

    response = _client(store).post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "event_types": ["session.stoped"],
            "secret_ref": "signing-x",
        },
        headers=_as(TenantId(uuid4())),
    )

    assert response.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert "session.stoped" in json.dumps(response.json())
    assert store.rows == []


def test_one_bad_type_beside_a_good_one_refuses_the_whole_registration() -> None:
    """Not a partial registration.

    Storing the eligible half would leave the tenant holding a registration that is not
    the one they asked for, and no answer anywhere saying so -- so the events they
    thought they had subscribed to would simply never arrive.
    """
    store = InMemoryWebhooks()

    response = _client(store).post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "event_types": [lifecycle.SESSION_STOPPED, turn.TURN_MESSAGE_DELTA],
            "secret_ref": "signing-x",
        },
        headers=_as(TenantId(uuid4())),
    )

    assert response.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert store.rows == []


def test_every_eligible_type_is_one_this_route_accepts() -> None:
    """The other direction, so the check cannot be tightened into refusing everything.

    A gate that refused every type would satisfy all three cases above. This is the one
    that fails if it does -- and it is driven off the registry rather than a list here,
    so a family that becomes deliverable is covered without an edit.
    """
    for eligible in sorted(WEBHOOK_ELIGIBLE):
        store = InMemoryWebhooks()

        response = _client(store).post(
            "/v1/webhooks",
            json={
                "url": "https://hooks.example.com/x",
                "event_types": [eligible],
                "secret_ref": "signing-x",
            },
            headers=_as(TenantId(uuid4())),
        )

        assert response.status_code == 201, (
            f"{eligible} is marked webhook-eligible and the route refused it with "
            f"{response.status_code}"
        )


def test_a_request_with_no_tenant_is_refused_before_the_store_is_reached() -> None:
    store = InMemoryWebhooks()

    response = _client(store).post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "event_types": ["session.stopped"],
            "secret_ref": "signing-x",
        },
    )

    assert response.status_code == 400
    assert store.rows == []


def test_a_list_holds_only_the_calling_tenants_registrations() -> None:
    store = InMemoryWebhooks()
    client = _client(store)
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    body = {
        "url": "https://hooks.example.com/x",
        "event_types": ["session.stopped"],
        "secret_ref": "signing-x",
    }
    client.post("/v1/webhooks", json=body, headers=_as(mine))
    client.post("/v1/webhooks", json=body, headers=_as(theirs))

    listed = client.get("/v1/webhooks", headers=_as(mine)).json()["webhooks"]

    assert [row["id"] for row in listed] == [str(store.rows[0].id)]


def test_deleting_answers_204_once_and_then_refuses_with_webhook_not_found() -> None:
    store = InMemoryWebhooks()
    client = _client(store)
    tenant = TenantId(uuid4())
    created = client.post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "event_types": ["session.stopped"],
            "secret_ref": "signing-x",
        },
        headers=_as(tenant),
    ).json()

    first = client.delete(f"/v1/webhooks/{created['id']}", headers=_as(tenant))
    second = client.delete(f"/v1/webhooks/{created['id']}", headers=_as(tenant))

    assert first.status_code == 204
    assert second.status_code == STATUS_FOR[ErrorCode.WEBHOOK_NOT_FOUND] == 404
    refusal = PublicErrorEnvelope.model_validate(second.json()).error
    assert refusal.code is ErrorCode.WEBHOOK_NOT_FOUND


def test_deleting_another_tenants_registration_refuses_and_leaves_it_readable() -> None:
    store = InMemoryWebhooks()
    client = _client(store)
    owner, thief = TenantId(uuid4()), TenantId(uuid4())
    created = client.post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "event_types": ["session.stopped"],
            "secret_ref": "signing-x",
        },
        headers=_as(owner),
    ).json()

    stolen = client.delete(f"/v1/webhooks/{created['id']}", headers=_as(thief))

    assert stolen.status_code == 404
    assert client.get("/v1/webhooks", headers=_as(owner)).json()["webhooks"] != []


def test_the_published_openapi_names_the_delete_refusal_and_its_envelope() -> None:
    app = create_app(_platform(InMemoryWebhooks()))
    responses = app.openapi()["paths"]["/v1/webhooks/{webhook_id}"]["delete"][
        "responses"
    ]

    assert "404" in responses
    assert "ErrorEnvelope" in json.dumps(responses["404"])


# --------------------------------------------------------------------------------
# Against a real PostgreSQL: the schema, the claim's race, and the tail's window.
# --------------------------------------------------------------------------------


@pytest.fixture
def store(engine: AsyncEngine) -> PostgresWebhookStore:
    return PostgresWebhookStore(engine)


async def test_the_registration_round_trips_and_is_scoped_to_its_tenant(
    store: PostgresWebhookStore,
) -> None:
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    event_types = frozenset({lifecycle.SESSION_STOPPED, lifecycle.SESSION_SUSPENDED})

    written = await store.register(
        mine, CallbackUrl("https://hooks.example.com/a"), event_types, "signing-a"
    )
    await store.register(
        theirs, CallbackUrl("https://hooks.example.com/b"), event_types, "signing-b"
    )

    listed = await store.list_for_tenant(mine)
    assert [r.id for r in listed] == [written.id]
    assert listed[0].event_types == event_types
    assert listed[0].created_at_ms > 0
    assert [r.id for r in await store.watching(mine, lifecycle.SESSION_STOPPED)] == [
        written.id
    ]
    assert await store.watching(mine, lifecycle.SESSION_CREATED) == [], (
        "a registration that named neither of these types was returned for one of them"
    )
    assert await store.watching(theirs, lifecycle.SESSION_STOPPED) != []


async def test_deleting_takes_the_delivery_rows_with_it_and_leaves_another_alone(
    store: PostgresWebhookStore, engine: AsyncEngine
) -> None:
    tenant = TenantId(uuid4())
    doomed = await store.register(
        tenant,
        CallbackUrl("https://hooks.example.com/doomed"),
        frozenset({lifecycle.SESSION_STOPPED}),
        "signing-a",
    )
    spared = await store.register(
        tenant,
        CallbackUrl("https://hooks.example.com/spared"),
        frozenset({lifecycle.SESSION_STOPPED}),
        "signing-b",
    )
    session_id = new_session_id()
    for hook in (doomed, spared):
        await store.claim(
            hook.id, session_id, lifecycle.SESSION_STOPPED, Seq(3), MAX_ATTEMPTS
        )

    assert await store.delete(doomed.id, tenant) is True
    assert await store.delete(doomed.id, tenant) is False
    assert await store.delete(spared.id, TenantId(uuid4())) is False

    async with engine.connect() as conn:
        remaining = (
            await conn.execute(
                sa.text(
                    "SELECT webhook_id FROM webhook_delivery WHERE session_id = :sid"
                ).bindparams(sa.bindparam("sid", type_=sa.Uuid())),
                {"sid": session_id},
            )
        ).scalars()
        assert [
            UUID(str(w)) for w in remaining if UUID(str(w)) in {doomed.id, spared.id}
        ] == [spared.id]


async def test_the_schema_refuses_no_types_a_plaintext_url_and_a_second_scan_row(
    engine: AsyncEngine,
) -> None:
    """Three constraints, each of which would otherwise be a check somebody can skip.

    The empty-array case is the one worth naming: `array_length(event_types, 1)` returns
    NULL for an empty array and a CHECK whose expression is NULL passes, so the obvious
    spelling admits exactly the row it was written to refuse. This is what proves the
    constraint on the table is the one that works -- and that it survived being rebuilt
    under the column's new name.
    """
    async with engine.begin() as conn:
        good = sa.text(
            "INSERT INTO webhook (id, tenant_id, url, event_types, secret_ref)"
            " VALUES (:wid, :tid, :url, :types, 'signing-x')"
        ).bindparams(
            sa.bindparam("wid", type_=sa.Uuid()),
            sa.bindparam("tid", type_=sa.Uuid()),
            sa.bindparam("types", type_=sa.ARRAY(sa.Text())),
        )
        for url, event_types in (
            ("https://hooks.example.com/x", []),
            ("http://hooks.example.com/x", [lifecycle.SESSION_STOPPED]),
        ):
            with pytest.raises(sa.exc.IntegrityError):
                async with engine.begin() as attempt:
                    await attempt.execute(
                        good,
                        {
                            "wid": uuid4(),
                            "tid": uuid4(),
                            "url": url,
                            "types": event_types,
                        },
                    )

    with pytest.raises(sa.exc.IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO webhook_scan (id, scanned_through_ms) VALUES (2, 0)"
                )
            )


async def test_the_watermark_is_seeded_at_the_present_and_never_moves_backwards(
    store: PostgresWebhookStore,
) -> None:
    """Seeded at now rather than 0: a zero would make the first sweep call back
    for every Session that reached a named state before anybody registered.
    """
    seeded = await store.scanned_through_ms()
    assert seeded > 1_700_000_000_000, (
        f"the watermark was seeded at {seeded}, which is not a recent epoch millisecond"
    )

    await store.advance_scan_to(seeded + 1_000)
    assert await store.scanned_through_ms() == seeded + 1_000
    await store.advance_scan_to(seeded)
    assert await store.scanned_through_ms() == seeded + 1_000


async def test_the_claim_counts_up_stops_at_delivered_and_stops_at_the_cap(
    store: PostgresWebhookStore,
) -> None:
    tenant = TenantId(uuid4())
    hook = await store.register(
        tenant,
        CallbackUrl("https://hooks.example.com/x"),
        frozenset({lifecycle.SESSION_STOPPED}),
        "signing-x",
    )
    session_id = new_session_id()

    async def take() -> int | None:
        return await store.claim(
            hook.id, session_id, lifecycle.SESSION_STOPPED, Seq(3), MAX_ATTEMPTS
        )

    assert [await take() for _ in range(MAX_ATTEMPTS + 1)] == [
        *range(1, MAX_ATTEMPTS + 1),
        None,
    ]

    other = new_session_id()
    assert (
        await store.claim(
            hook.id, other, lifecycle.SESSION_STOPPED, Seq(3), MAX_ATTEMPTS
        )
        == 1
    )
    owed = await store.undelivered(MAX_ATTEMPTS, 100)
    assert [
        (r.session_id, r.secret_ref, r.url) for r in owed if r.session_id == other
    ] == [(other, "signing-x", "https://hooks.example.com/x")]

    await store.mark_delivered(hook.id, other, Seq(3), 200)
    assert (
        await store.claim(
            hook.id, other, lifecycle.SESSION_STOPPED, Seq(3), MAX_ATTEMPTS
        )
        is None
    )
    assert [r.session_id for r in await store.undelivered(MAX_ATTEMPTS, 100)] != [other]


async def test_two_events_on_one_session_each_get_their_own_delivery(
    store: PostgresWebhookStore,
) -> None:
    """One Session, three lifecycle events, three deliveries owed, in front of the
    real database that decides it.

    This is the case the delivery key decides. A Session that is created and then
    stopped twice appends three events a tenant registered to hear about, and each is a
    separate thing that happened -- so each is owed a callback of its own. The second
    stop is the race `control/session/lifecycle.py` documents rather than a shape
    invented for this test: two archive callers each fold before either appends, and the
    losing append still lands.

    Keyed on the state a fold named, the third claim was not a new row: both stops
    arrive at STOPPED, so the third event landed on the row the second one inserted and
    was answered as a second *attempt* at a callback already delivered -- `[1, 1, 2]`
    rather than `[1, 1, 1]`. One ending reached the tenant, the other did not, and
    nothing anywhere recorded a delivery as missing.

    Keyed on the sequence, the three are three rows, because the sequence is what makes
    two events on one Session distinguishable at all.
    """
    tenant = TenantId(uuid4())
    hook = await store.register(
        tenant,
        CallbackUrl("https://hooks.example.com/twice-stopped"),
        frozenset({lifecycle.SESSION_CREATED, lifecycle.SESSION_STOPPED}),
        "signing-twice-stopped",
    )
    session_id = new_session_id()

    granted = [
        await store.claim(hook.id, session_id, event_type, Seq(seq), MAX_ATTEMPTS)
        for event_type, seq in (
            (lifecycle.SESSION_CREATED, 1),
            (lifecycle.SESSION_STOPPED, 2),
            (lifecycle.SESSION_STOPPED, 3),
        )
    ]

    assert granted == [1, 1, 1], (
        f"the three events were granted {granted}. Anything but a first attempt each "
        "means two events collapsed onto one delivery row, and the later one is "
        "delivered as a retry of the earlier or not at all."
    )


async def test_two_dispatchers_claiming_one_event_produce_exactly_one_winner(
    store: PostgresWebhookStore,
) -> None:
    """The race that decides whether "one callback" is true.

    Ten concurrent claims of one triple, each on its own connection. The primary
    key settles it; a read-then-write would let two callers see the row absent.
    """
    tenant = TenantId(uuid4())
    hook = await store.register(
        tenant,
        CallbackUrl("https://hooks.example.com/x"),
        frozenset({lifecycle.SESSION_STOPPED}),
        "signing-x",
    )
    session_id = new_session_id()

    taken = await asyncio.gather(
        *(
            store.claim(hook.id, session_id, lifecycle.SESSION_STOPPED, Seq(3), 1)
            for _ in range(10)
        ),
        return_exceptions=True,
    )

    won = [t for t in taken if t == 1]
    assert len(won) == 1, f"{len(won)} dispatchers claimed the same callback: {taken}"


async def test_the_tail_returns_each_lifecycle_event_once_with_its_owning_tenant(
    engine: AsyncEngine,
) -> None:
    """Two Sessions of two tenants, one window, and the half-open boundary.

    The boundary is the part that matters: consecutive windows must cover the
    instant between them exactly once. A repeat costs one refused claim; nothing
    recovers a skip.
    """
    ranges = PostgresEventLogRange(engine)
    appends = PostgresEventLogAppend(engine)
    sessions = PostgresSessionRegistry(engine)

    owners = [TenantId(uuid4()), TenantId(uuid4())]
    made: list[SessionId] = []
    for tenant in owners:
        session_id = new_session_id()
        await sessions.create(_session_record(session_id, tenant))
        await appends.append(session_id, lifecycle.SESSION_CREATED, {})
        await appends.append(session_id, "tool.called", {"credential": TOOL_CREDENTIAL})
        await appends.append(session_id, lifecycle.SESSION_STOPPED, {})
        made.append(session_id)

    mine = set(made)

    span = (
        sa.text(
            "SELECT"
            " (extract(epoch from min(appended_at)) * 1000)::bigint AS lo,"
            " (extract(epoch from max(appended_at)) * 1000)::bigint AS hi"
            " FROM event_log WHERE session_id = ANY(:ids)"
        )
        .bindparams(sa.bindparam("ids", type_=sa.ARRAY(sa.Uuid())))
        .columns(lo=sa.BigInteger(), hi=sa.BigInteger())
    )
    async with engine.connect() as conn:
        edges = (await conn.execute(span, {"ids": list(mine)})).one()

    low, high = int(edges.lo) - 1, int(edges.hi) + 1
    whole = [
        row
        for row in await ranges.lifecycle_events_between(WEBHOOK_ELIGIBLE, low, high)
        if row.session_id in mine
    ]

    assert len(whole) == 4, "each Session's created and stopped rows, and nothing else"
    assert {row.tenant_id for row in whole} == set(owners)
    for row in whole:
        assert row.seq in (1, 3), "the tool.called row was tailed"
    assert sorted({row.type for row in whole}) == [
        lifecycle.SESSION_CREATED,
        lifecycle.SESSION_STOPPED,
    ], (
        "the tail must hand back each event's own type -- it is what a registration is "
        "matched against, and nothing downstream can recover it"
    )

    # Two consecutive half-open windows over the same span cover it exactly once.
    midpoint = (low + high) // 2
    first = [
        (r.session_id, r.seq)
        for r in await ranges.lifecycle_events_between(WEBHOOK_ELIGIBLE, low, midpoint)
        if r.session_id in mine
    ]
    second = [
        (r.session_id, r.seq)
        for r in await ranges.lifecycle_events_between(WEBHOOK_ELIGIBLE, midpoint, high)
        if r.session_id in mine
    ]
    assert sorted(first + second) == sorted((r.session_id, r.seq) for r in whole)
    assert set(first).isdisjoint(second), "an event fell in both windows"


def _session_record(session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
    return SessionRecord(
        id=session_id,
        tenant_id=tenant_id,
        definition_id=new_definition_id(),
        definition_revision="1",
        grant=frozenset(),
        scope=(),
        budget_minor_units=1000,
        budget_currency="USD",
        retention_days=30,
    )


def test_the_adapters_satisfy_the_ports_the_dispatcher_names() -> None:
    """Structural conformance, asserted where mypy --strict also grades it.

    Cheap at run time and worth having anyway: mypy checks these assignments, and this
    fails with a name if a method is ever removed from an adapter that no call site in
    the suite happens to exercise.
    """
    engine: Any = None
    store = PostgresWebhookStore(engine)
    ranges = PostgresEventLogRange(engine)

    hooks: WatchedWebhooks = store
    ledger: DeliveryLedger = store
    registrations: WebhookStore = store
    candidate: type[LifecycleCandidate] = LifecycleRow
    pending: type[PendingDelivery] = type(
        FakePending(
            uuid4(),
            TenantId(uuid4()),
            "u",
            "r",
            new_session_id(),
            lifecycle.SESSION_STOPPED,
            Seq(1),
        )
    )

    assert (hooks, ledger, registrations, candidate, pending) is not None
    assert hasattr(ranges, "lifecycle_events_between")


class UnusedEnvironmentStore:
    """Satisfies the environment-store port and is never called.

    Raising rather than returning a harmless value: a test in this file that reached the
    environment store would be grading something this file does not grade, and a quiet
    stub would let it pass while doing so.
    """

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a test in this file registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        raise AssertionError("a test in this file resolved an environment")
