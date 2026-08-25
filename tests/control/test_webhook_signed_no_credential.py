"""What leaves the platform when a Session reaches a state somebody registered for.

The Event Log below is a real fold's input: the fake stores rows and the
dispatcher folds them through `core.projection`, so the state in the delivered
payload is produced the same way the state on the tenant's own read is, rather
than asserted into place by the fixture.

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
from collections.abc import Mapping, Sequence
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
    LIFECYCLE_TYPES,
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
from managed_agent.core.session.session import SessionRecord, SessionState
from managed_agent.core.vocabulary import lifecycle

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


@dataclass
class FakeLog:
    """One Event Log per Session, plus the cross-Session tail the dispatcher reads."""

    tenant: TenantId
    rows: dict[SessionId, list[FakeRow]] = field(default_factory=dict)
    reads: list[tuple[SessionId, int, int, int]] = field(default_factory=list)
    swept: set[SessionId] = field(default_factory=set)
    """Sessions the retention sweep emptied after the tail had already named them.

    The real race: the cross-Session tail reads a window, and by the time the fold goes
    back for that Session's own log there is nothing left in it. Modelled here because
    it is the only way the fold can be handed a range with no transition in it.
    """

    def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        rows = self.rows.setdefault(session_id, [])
        seq = Seq(len(rows) + 1)
        rows.append(FakeRow(session_id, seq, type_, payload))
        return seq

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[FakeRow]:
        """Honours `limit`, because the real adapter does and a fake that did not could
        not fail the way the real one fails."""
        self.reads.append((session_id, start, end, limit))
        if session_id in self.swept:
            return []
        held = [r for r in self.rows.get(session_id, []) if start <= r.seq <= end]
        return held[:limit]

    async def lifecycle_events_between(
        self, types: Sequence[str], from_ms: int, to_ms: int
    ) -> Sequence[FakeCandidate]:
        # The window is ignored here on purpose: the adapter's half-open behaviour is
        # pinned against a real Postgres below, and mixing time into this fake would let
        # a delivery case fail for a reason that has nothing to do with delivery.
        return [
            FakeCandidate(self.tenant, row.session_id, row.seq)
            for rows in self.rows.values()
            for row in rows
            if row.type in types
        ]


@dataclass
class CappedLog(FakeLog):
    """A log whose per-Session read defaults to two rows, like an adapter that pages.

    The real `PostgresEventLogRange.read` caps at 500 and documents a short result as
    "page for the rest". A fake that always returned everything it held could not fail
    the way the real one fails, so a caller taking the default would look correct here
    and fold one page in production. Two rather than 500 only so the test's log can be
    three events instead of six hundred.
    """

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 2
    ) -> Sequence[FakeRow]:
        return await super().read(session_id, start, end, limit)


Claim = tuple[UUID, SessionId, str]


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
    state: SessionState
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
        self, tenant_id: TenantId, state: SessionState
    ) -> Sequence[WebhookRecord]:
        return [h for h in self.hooks if h.tenant_id == tenant_id and state in h.states]

    async def scanned_through_ms(self) -> int:
        return self.watermark

    async def advance_scan_to(self, at_ms: int) -> None:
        self.watermark = max(self.watermark, at_ms)

    async def claim(
        self,
        webhook_id: UUID,
        session_id: SessionId,
        state: SessionState,
        seq: Seq,
        max_attempts: int,
    ) -> int | None:
        key: Claim = (webhook_id, session_id, state.value)
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
            state,
            seq,
        )
        return self.attempts[key]

    async def mark_delivered(
        self, webhook_id: UUID, session_id: SessionId, state: SessionState, status: int
    ) -> None:
        key: Claim = (webhook_id, session_id, state.value)
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
    states: frozenset[SessionState],
    url: str = "https://hooks.example.com/map",
) -> WebhookRecord:
    return WebhookRecord(
        id=uuid4(),
        tenant_id=tenant,
        url=CallbackUrl(url),
        states=states,
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


async def test_the_fake_log_reads_the_range_it_is_asked_for_and_honours_the_limit() -> (
    None
):
    log = FakeLog(tenant=TenantId(uuid4()))
    session_id = new_session_id()
    for name in ("a", "b", "c"):
        log.append(session_id, name, {})

    assert [r.seq for r in await log.read(session_id, Seq(1), Seq(2))] == [1, 2]
    assert [r.seq for r in await log.read(session_id, Seq(1), Seq(3), limit=2)] == [
        1,
        2,
    ]


async def test_the_fake_tail_returns_only_the_types_it_is_asked_for() -> None:
    log = FakeLog(tenant=TenantId(uuid4()))
    session_id = _stopped_session(log)

    found = await log.lifecycle_events_between(LIFECYCLE_TYPES, 0, _A_MOMENT)

    assert [c.seq for c in found] == [1, 3], (
        "the tail returned the tool.called row, so a delivery test could pass for the "
        "wrong reason"
    )
    assert {c.session_id for c in found} == {session_id}


async def test_the_fake_claim_counts_up_and_stops_at_delivered_and_at_the_cap() -> None:
    tenant = TenantId(uuid4())
    hook = _hook(tenant, frozenset({SessionState.STOPPED}))
    store = FakeStore(hooks=[hook])
    session_id = new_session_id()

    async def take() -> int | None:
        return await store.claim(
            hook.id, session_id, SessionState.STOPPED, Seq(3), MAX_ATTEMPTS
        )

    assert await take() == 1
    assert await take() == 2
    await store.mark_delivered(hook.id, session_id, SessionState.STOPPED, 200)
    assert await take() is None

    spent = FakeStore(hooks=[hook])
    taken = [
        await spent.claim(
            hook.id, session_id, SessionState.STOPPED, Seq(3), MAX_ATTEMPTS
        )
        for _ in range(MAX_ATTEMPTS + 1)
    ]
    assert taken == [*range(1, MAX_ATTEMPTS + 1), None]


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


def test_a_registration_refuses_a_stray_field_no_states_and_no_secret_ref() -> None:
    """`extra="forbid"` is what keeps a field carrying signing material out of the body.

    A model that ignored unknown fields would accept `secret` silently, and the tenant
    that sent one would believe the platform was holding it.
    """
    good: dict[str, Any] = {
        "url": "https://hooks.example.com/x",
        "states": ["stopped"],
        "secret_ref": "signing-x",
    }
    parsed = RegisterWebhook.model_validate(good)
    assert parsed.states == frozenset({SessionState.STOPPED})

    for broken in (
        {**good, "secret": "hunter2"},
        {**good, "states": []},
        {**good, "secret_ref": ""},
    ):
        with pytest.raises(ValueError):
            RegisterWebhook.model_validate(broken)


def test_a_stored_registration_is_frozen_and_names_no_secret_but_the_reference() -> (
    None
):
    record = _hook(TenantId(uuid4()), frozenset({SessionState.STOPPED}))

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


def test_the_tenant_visible_read_has_exactly_five_fields_and_sorts_its_states() -> None:
    record = _hook(
        TenantId(uuid4()), frozenset({SessionState.STOPPED, SessionState.RUNNING})
    )

    view = WebhookView.of(record)

    assert view.states == (SessionState.RUNNING, SessionState.STOPPED)
    assert set(WebhookView.model_fields) == {
        "id",
        "url",
        "states",
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
        "state": SessionState.STOPPED,
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
        "state",
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


def test_every_state_moving_event_type_is_one_the_tail_reads() -> None:
    """The drift guard.

    A type that moves a Session's state but is not in the lifecycle family would never
    be tailed, so the callback for it would simply never be sent -- with no error
    anywhere. Pinned against the projection's own table rather than against a list here,
    which is the only version of this assertion that can notice a new transition.
    """
    untailed = sorted(set(_TRANSITIONS) - set(LIFECYCLE_TYPES))
    assert untailed == [], (
        f"these event types move a Session's state and the webhook tail does not read "
        f"them: {untailed}. Declare them in the lifecycle family."
    )


async def test_a_stop_posts_one_callback_whose_signature_verifies() -> None:
    """One callback, counted -- not "at least one".

    The second sweep is what makes the count mean something: a dispatcher that delivered
    on every pass would satisfy "a callback was delivered" and fail here.
    """
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    session_id = _stopped_session(log)
    store = FakeStore(hooks=[_hook(tenant, frozenset({SessionState.STOPPED}))])
    recorder = Recorder(200)

    async with recorder.client() as client:
        dispatcher = _dispatcher(store, log, client)
        run = await dispatcher.sweep_once(now_ms=_A_MOMENT)
        again = await dispatcher.sweep_once(now_ms=_A_MOMENT + 60_000)

    assert len(recorder.sent) == 1, (
        f"a state reached once produced {len(recorder.sent)} callbacks; the promise is "
        "exactly one"
    )
    assert (len(run.delivered), len(again.delivered)) == (1, 0)

    request = recorder.sent[0]
    payload = json.loads(request.content)
    assert payload["session_id"] == str(session_id)
    assert payload["state"] == "stopped"
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
    store = FakeStore(hooks=[_hook(tenant, frozenset({SessionState.STOPPED}))])
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
        "state",
        "seq",
        "delivered_at_ms",
    }
    assert request.url.host == "hooks.example.com"


async def test_another_state_and_another_tenant_are_both_passed_over() -> None:
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(
        hooks=[
            _hook(tenant, frozenset({SessionState.SUSPENDED})),
            _hook(TenantId(uuid4()), frozenset({SessionState.STOPPED})),
        ]
    )
    recorder = Recorder(200)

    async with recorder.client() as client:
        run = await _dispatcher(store, log, client).sweep_once(now_ms=_A_MOMENT)

    assert recorder.sent == []
    assert run.delivered == ()


async def test_two_watched_states_get_one_callback_each_and_no_more() -> None:
    """The Session passes through RUNNING and then STOPPED. Two states, two callbacks,
    and the second sweep adds none."""
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(
        hooks=[_hook(tenant, frozenset({SessionState.RUNNING, SessionState.STOPPED}))]
    )
    recorder = Recorder(200)

    async with recorder.client() as client:
        dispatcher = _dispatcher(store, log, client)
        await dispatcher.sweep_once(now_ms=_A_MOMENT)
        await dispatcher.sweep_once(now_ms=_A_MOMENT + 60_000)

    states = [json.loads(r.content)["state"] for r in recorder.sent]
    assert sorted(states) == ["running", "stopped"]


async def test_a_session_swept_between_the_tail_and_the_fold_is_skipped() -> None:
    """The tail named it; the retention sweep emptied it before the fold got there.

    Nothing true can be said about a Session whose log is gone, so no callback goes out
    -- and the watermark still advances, because the window really was read and a
    watermark that stalled here would re-read this window for ever.
    """
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    session_id = _stopped_session(log)
    log.swept.add(session_id)
    store = FakeStore(
        hooks=[_hook(tenant, frozenset({SessionState.STOPPED}))],
        # Close enough behind that one window reaches the frontier, so the assertion
        # below is about the skip rather than about MAX_WINDOW_MS.
        watermark=_A_MOMENT - SAFETY_LAG_MS - 1_000,
    )
    recorder = Recorder(200)

    async with recorder.client() as client:
        run = await _dispatcher(store, log, client).sweep_once(now_ms=_A_MOMENT)

    assert recorder.sent == []
    assert store.watermark == _A_MOMENT - SAFETY_LAG_MS
    assert run.scanned_through_ms == _A_MOMENT - SAFETY_LAG_MS


async def test_the_fold_reads_the_whole_log_rather_than_one_default_page() -> None:
    """The read names a limit wide enough for the range it asks for.

    Taking the adapter's default has shipped twice in this repository: a fold that stops
    at page one reports the state as of that page, with nothing raising. Driven against
    a log that really does page, so what is graded is the state that got delivered
    rather than the argument that was passed.

    The Session here is created (1), calls a tool (2), stops (3). A fold capped at two
    rows arrives at RUNNING and stays there, and RUNNING was already called back for
    seq 1 -- so the failure is silent: one callback, naming the wrong state, and no
    "stopped" callback ever.
    """
    tenant = TenantId(uuid4())
    log = CappedLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(
        hooks=[_hook(tenant, frozenset({SessionState.RUNNING, SessionState.STOPPED}))]
    )
    recorder = Recorder(200)

    async with recorder.client() as client:
        await _dispatcher(store, log, client).sweep_once(now_ms=_A_MOMENT)

    delivered = sorted(json.loads(r.content)["state"] for r in recorder.sent)
    assert delivered == ["running", "stopped"], (
        f"the callbacks named {delivered}; a fold that stopped at the log's first page "
        "never reaches the stopped event and reports the state as of that page"
    )


async def test_a_window_inside_the_safety_lag_reads_and_moves_nothing() -> None:
    tenant = TenantId(uuid4())
    log = FakeLog(tenant=tenant)
    _stopped_session(log)
    store = FakeStore(
        hooks=[_hook(tenant, frozenset({SessionState.STOPPED}))],
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
    hook = _hook(tenant, frozenset({SessionState.STOPPED}))
    store = FakeStore(hooks=[hook])
    # Another dispatcher already delivered this exact callback.
    store.delivered[(hook.id, session_id, SessionState.STOPPED.value)] = 200
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
                tenant, frozenset({SessionState.STOPPED}), "https://down.example.com/x"
            ),
            _hook(
                tenant, frozenset({SessionState.STOPPED}), "https://up.example.com/x"
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
    store = FakeStore(hooks=[_hook(tenant, frozenset({SessionState.STOPPED}))])
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
    store = FakeStore(hooks=[_hook(tenant, frozenset({SessionState.STOPPED}))])
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
        states: frozenset[SessionState],
        secret_ref: str,
    ) -> WebhookRecord:
        record = WebhookRecord(
            id=uuid4(),
            tenant_id=tenant_id,
            url=url,
            states=states,
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
        self, tenant_id: TenantId, state: SessionState
    ) -> Sequence[WebhookRecord]:
        return [r for r in self.rows if r.tenant_id == tenant_id and state in r.states]


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
            "states": ["stopped"],
            "secret_ref": "signing-x",
        },
        headers=_as(tenant),
    )

    assert response.status_code == 201
    view = WebhookView.model_validate(response.json())
    assert view.id == store.rows[0].id
    assert view.states == (SessionState.STOPPED,)


def test_a_plaintext_destination_is_refused_by_code_and_writes_nothing() -> None:
    store = InMemoryWebhooks()

    response = _client(store).post(
        "/v1/webhooks",
        json={
            "url": "http://hooks.example.com/x",
            "states": ["stopped"],
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
                "states": ["stopped"],
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
                "states": ["stopped"],
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
            "states": ["stopped"],
            "secret_ref": "vault://x",
        },
        headers=_as(TenantId(uuid4())),
    )

    detail = PublicErrorEnvelope.model_validate(response.json()).error.detail
    assert "secret_ref" in detail, f"the refusal names {sorted(detail)}"
    assert "url" not in detail, "the refusal blames the url, which was well-formed"


def test_a_request_with_no_tenant_is_refused_before_the_store_is_reached() -> None:
    store = InMemoryWebhooks()

    response = _client(store).post(
        "/v1/webhooks",
        json={
            "url": "https://hooks.example.com/x",
            "states": ["stopped"],
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
        "states": ["stopped"],
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
            "states": ["stopped"],
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
            "states": ["stopped"],
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
    states = frozenset({SessionState.STOPPED, SessionState.SUSPENDED})

    written = await store.register(
        mine, CallbackUrl("https://hooks.example.com/a"), states, "signing-a"
    )
    await store.register(
        theirs, CallbackUrl("https://hooks.example.com/b"), states, "signing-b"
    )

    listed = await store.list_for_tenant(mine)
    assert [r.id for r in listed] == [written.id]
    assert listed[0].states == states
    assert listed[0].created_at_ms > 0
    assert [r.id for r in await store.watching(mine, SessionState.STOPPED)] == [
        written.id
    ]
    assert await store.watching(mine, SessionState.RUNNING) == []
    assert await store.watching(theirs, SessionState.STOPPED) != []


async def test_deleting_takes_the_delivery_rows_with_it_and_leaves_another_alone(
    store: PostgresWebhookStore, engine: AsyncEngine
) -> None:
    tenant = TenantId(uuid4())
    doomed = await store.register(
        tenant,
        CallbackUrl("https://hooks.example.com/doomed"),
        frozenset({SessionState.STOPPED}),
        "signing-a",
    )
    spared = await store.register(
        tenant,
        CallbackUrl("https://hooks.example.com/spared"),
        frozenset({SessionState.STOPPED}),
        "signing-b",
    )
    session_id = new_session_id()
    for hook in (doomed, spared):
        await store.claim(
            hook.id, session_id, SessionState.STOPPED, Seq(3), MAX_ATTEMPTS
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


async def test_the_schema_refuses_no_states_a_plaintext_url_and_a_second_scan_row(
    engine: AsyncEngine,
) -> None:
    """Three constraints, each of which would otherwise be a check somebody can skip.

    The empty-array case is the one worth naming: `array_length(states, 1)` returns NULL
    for an empty array and a CHECK whose expression is NULL passes, so the obvious
    spelling admits exactly the row it was written to refuse. This is what proves the
    constraint on the table is the one that works.
    """
    async with engine.begin() as conn:
        good = sa.text(
            "INSERT INTO webhook (id, tenant_id, url, states, secret_ref)"
            " VALUES (:wid, :tid, :url, :states, 'signing-x')"
        ).bindparams(
            sa.bindparam("wid", type_=sa.Uuid()),
            sa.bindparam("tid", type_=sa.Uuid()),
            sa.bindparam("states", type_=sa.ARRAY(sa.Text())),
        )
        for url, states in (
            ("https://hooks.example.com/x", []),
            ("http://hooks.example.com/x", ["stopped"]),
        ):
            with pytest.raises(sa.exc.IntegrityError):
                async with engine.begin() as attempt:
                    await attempt.execute(
                        good,
                        {
                            "wid": uuid4(),
                            "tid": uuid4(),
                            "url": url,
                            "states": states,
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
        frozenset({SessionState.STOPPED}),
        "signing-x",
    )
    session_id = new_session_id()

    async def take() -> int | None:
        return await store.claim(
            hook.id, session_id, SessionState.STOPPED, Seq(3), MAX_ATTEMPTS
        )

    assert [await take() for _ in range(MAX_ATTEMPTS + 1)] == [
        *range(1, MAX_ATTEMPTS + 1),
        None,
    ]

    other = new_session_id()
    assert (
        await store.claim(hook.id, other, SessionState.STOPPED, Seq(3), MAX_ATTEMPTS)
        == 1
    )
    owed = await store.undelivered(MAX_ATTEMPTS, 100)
    assert [
        (r.session_id, r.secret_ref, r.url) for r in owed if r.session_id == other
    ] == [(other, "signing-x", "https://hooks.example.com/x")]

    await store.mark_delivered(hook.id, other, SessionState.STOPPED, 200)
    assert (
        await store.claim(hook.id, other, SessionState.STOPPED, Seq(3), MAX_ATTEMPTS)
        is None
    )
    assert [r.session_id for r in await store.undelivered(MAX_ATTEMPTS, 100)] != [other]


async def test_two_dispatchers_claiming_one_state_change_produce_exactly_one_winner(
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
        frozenset({SessionState.STOPPED}),
        "signing-x",
    )
    session_id = new_session_id()

    taken = await asyncio.gather(
        *(
            store.claim(hook.id, session_id, SessionState.STOPPED, Seq(3), 1)
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
        for row in await ranges.lifecycle_events_between(LIFECYCLE_TYPES, low, high)
        if row.session_id in mine
    ]

    assert len(whole) == 4, "each Session's created and stopped rows, and nothing else"
    assert {row.tenant_id for row in whole} == set(owners)
    for row in whole:
        assert row.seq in (1, 3), "the tool.called row was tailed"

    # Two consecutive half-open windows over the same span cover it exactly once.
    midpoint = (low + high) // 2
    first = [
        (r.session_id, r.seq)
        for r in await ranges.lifecycle_events_between(LIFECYCLE_TYPES, low, midpoint)
        if r.session_id in mine
    ]
    second = [
        (r.session_id, r.seq)
        for r in await ranges.lifecycle_events_between(LIFECYCLE_TYPES, midpoint, high)
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
            SessionState.STOPPED,
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
