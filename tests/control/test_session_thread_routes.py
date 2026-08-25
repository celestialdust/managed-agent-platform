"""The five thread routes, driven through the real app over in-memory ports.

Tier 1. Every case goes through `create_app`, so a router that stops being mounted fails
here rather than passing against a FastAPI this file built for itself -- the defect that
left two wave-1 modules answering 404 for a whole wave while all their own tests passed.

The stubs for the ports this surface does not touch are imported from
`test_stream_resume` rather than copied. They raise on every method, so a route that
started resolving a definition or creating a Session would say so instead of passing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from uuid import uuid4

import httpx
import pytest
from test_stream_resume import (
    LiveResponse,
    OneOwnedSession,
    UnusedDefinitionRegistry,
    UnusedEnvironmentStore,
    UnusedToolRegistry,
    UnusedWebhooks,
    _scope,
)

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.threads import ThreadActivity
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, TenantId, new_session_id
from managed_agent.core.vocabulary import lifecycle, turn
from managed_agent.core.vocabulary import thread as thread_events

_OWNER = TenantId(uuid4())
_STRANGER = TenantId(uuid4())
_HEADERS = {TENANT_HEADER: str(_OWNER)}

_ROOT = "11111111-1111-5111-8111-111111111111"
_CHILD = "22222222-2222-5222-8222-222222222222"


@dataclass(frozen=True, slots=True)
class Row:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


class Log:
    """A contiguous in-memory log that carries payloads, which this surface reads.

    `FakeLog` in `test_stream_resume` numbers rows and stamps `{"n": seq}` as the
    payload, which is enough to grade framing and nothing here: every question on this
    surface is about `payload["thread_id"]`. So this one takes the payload from the
    caller and is otherwise the same shape.
    """

    def __init__(self) -> None:
        self.rows: list[Row] = []
        self._next = FIRST_SEQ

    def add(self, session_id: SessionId, type_: str, **payload: object) -> Seq:
        seq = Seq(self._next)
        self._next += 1
        self.rows.append(Row(session_id, seq, type_, dict(payload)))
        return seq

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Row]:
        span = [
            row
            for row in self.rows
            if row.session_id == session_id and start <= row.seq <= end
        ]
        return span[:limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Row]:
        for row in self.rows:
            if row.session_id == session_id and row.seq > after:
                yield row


class Appends:
    """Records what the archive route writes, and moves the index the way a store would.

    The mutation is the point. Without it the route's re-read after the append would
    return the pre-archive record, `archived_at_ms` would come back null on the call
    that just archived, and the test asserting otherwise would be grading a fake that
    forgot what it was told -- which is exactly the shape of an incomplete mock that
    passes.
    """

    def __init__(self, index: Index, log: Log) -> None:
        self.written: list[tuple[str, dict[str, object]]] = []
        self._index = index
        self._log = log

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.written.append((type_, payload))
        seq = self._log.add(session_id, type_, **payload)
        if type_ == thread_events.THREAD_ARCHIVED:
            self._index.archive(str(payload["thread_id"]), at_ms=_ARCHIVED_AT_MS)
        return seq


_ARCHIVED_AT_MS = 1_700_000_009_000


class Index:
    """The thread index, paging for real so the route's cursor logic is what is graded.

    Ordered by `started_seq` and exclusive on `after_seq`, because those are the two
    properties the port's docstring commits to and the route's `limit + 1` look-ahead
    depends on both. A fake that ignored `limit` would let `next_page` be wrong in
    either direction with nothing failing.
    """

    def __init__(self, *activities: ThreadActivity) -> None:
        self._activities = list(activities)

    def archive(self, thread_id: str, *, at_ms: int) -> None:
        self._activities = [
            replace(one, archived_at_ms=at_ms) if one.thread_id == thread_id else one
            for one in self._activities
        ]

    async def threads_of(
        self,
        session_id: SessionId,
        *,
        after_seq: Seq | None = None,
        limit: int = 25,
    ) -> Sequence[ThreadActivity]:
        ordered = sorted(self._activities, key=lambda one: one.started_seq)
        if after_seq is not None:
            ordered = [one for one in ordered if one.started_seq > after_seq]
        return ordered[:limit]

    async def thread_at(
        self, session_id: SessionId, thread_id: str
    ) -> ThreadActivity | None:
        return next(
            (one for one in self._activities if one.thread_id == thread_id), None
        )


def _activity(
    thread_id: str,
    *,
    parent: str | None = None,
    announced: bool = True,
    started_seq: int = 2,
    archived_at_ms: int | None = None,
    turn_ended: bool = True,
) -> ThreadActivity:
    return ThreadActivity(
        thread_id=thread_id,
        parent_thread_id=parent,
        was_announced=announced,
        started_seq=Seq(started_seq),
        created_at_ms=1_700_000_000_000 + started_seq,
        updated_at_ms=1_700_000_005_000 + started_seq,
        archived_at_ms=archived_at_ms,
        turn_ended=turn_ended,
    )


@dataclass(frozen=True, slots=True)
class Wired:
    app: object
    log: Log
    index: Index
    appends: Appends
    session_id: SessionId


def _wire(*activities: ThreadActivity, stopped: bool = False) -> Wired:
    """The real app over one owned Session whose log holds a creation event.

    A creation event is not decoration: `project` raises on a log without one, and the
    thread status depends on that fold, so a Session with an empty log would fail every
    route here with a 500 rather than answering.
    """
    session_id = new_session_id()
    log = Log()
    log.add(session_id, lifecycle.SESSION_CREATED)
    if stopped:
        log.add(session_id, lifecycle.SESSION_STOPPED)
    index = Index(*activities)
    appends = Appends(index, log)
    app = create_app(
        Platform(
            event_log_append=appends,
            event_log_range=log,
            definition_registry=UnusedDefinitionRegistry(),
            tool_registry=UnusedToolRegistry(),
            session_registry=OneOwnedSession(_OWNER, session_id),
            webhooks=UnusedWebhooks(),
            environment_store=UnusedEnvironmentStore(),
            turn_dispatch=NoPodTransport(),
            file_store=unconfigured_file_store(),
            session_threads=index,
        )
    )
    return Wired(app, log, index, appends, session_id)


def _client(wired: Wired) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=wired.app),  # type: ignore[arg-type]
        base_url="http://platform",
        headers=_HEADERS,
    )


async def test_a_sessions_threads_are_listed_oldest_first_with_their_parents() -> None:
    wired = _wire(
        _activity(_CHILD, parent=_ROOT, started_seq=7),
        _activity(_ROOT, started_seq=2),
    )
    async with _client(wired) as client:
        answer = await client.get(f"/v1/sessions/{wired.session_id}/threads")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert [one["id"] for one in body["data"]] == [_ROOT, _CHILD]
    assert body["data"][0]["parent_thread_id"] is None
    assert body["data"][1]["parent_thread_id"] == _ROOT
    assert body["next_page"] is None


async def test_a_thread_publishes_no_agent_no_stats_and_no_usage() -> None:
    """Absent rather than null: this platform does not publish those facts at all.

    Asserted as an exact key set rather than three absences, so a field added to the
    resource without a decision about it fails here.
    """
    wired = _wire(_activity(_ROOT))
    async with _client(wired) as client:
        answer = await client.get(f"/v1/sessions/{wired.session_id}/threads")
    assert set(answer.json()["data"][0]) == {
        "id",
        "session_id",
        "type",
        "parent_thread_id",
        "status",
        "created_at_ms",
        "updated_at_ms",
        "archived_at_ms",
    }


async def test_a_thread_the_runtime_never_announced_publishes_no_parent_field() -> None:
    """Absent, not null. Null is the root's answer and this thread is not the root.

    Measured against codex-cli 0.149.0 on 2026-08-24: one delegating Turn produced six
    threads and exactly one `thread.started`, so five of them had no parent pointer
    anywhere in the log. Publishing null for those would tell a caller looking for the
    root that it had found six.
    """
    wired = _wire(
        _activity(_ROOT, started_seq=2),
        _activity(_CHILD, announced=False, started_seq=7),
    )
    async with _client(wired) as client:
        answer = await client.get(f"/v1/sessions/{wired.session_id}/threads")
    root, child = answer.json()["data"]
    assert root["parent_thread_id"] is None, "the announced root lost its null"
    assert "parent_thread_id" not in child, child


async def test_an_unannounced_thread_is_still_read_and_archived_by_name() -> None:
    """It has events, so a caller can see it on the listing and must be able to act.

    A thread the runtime never announced is not a lesser thread: it produced output that
    reached a tenant. Refusing to address it would list something the API then denies.
    """
    wired = _wire(_activity(_CHILD, announced=False, turn_ended=True))
    async with _client(wired) as client:
        read = await client.get(f"/v1/sessions/{wired.session_id}/threads/{_CHILD}")
        archived = await client.post(
            f"/v1/sessions/{wired.session_id}/threads/{_CHILD}/archive"
        )
    assert read.status_code == 200, read.text
    assert "parent_thread_id" not in read.json()
    assert archived.status_code == 200, archived.text
    assert archived.json()["archived_at_ms"] == _ARCHIVED_AT_MS


async def test_a_session_that_never_delegated_answers_an_empty_page() -> None:
    wired = _wire()
    async with _client(wired) as client:
        answer = await client.get(f"/v1/sessions/{wired.session_id}/threads")
    assert answer.status_code == 200
    assert answer.json() == {"data": [], "next_page": None}


async def test_the_listing_pages_forward_and_the_last_page_offers_no_token() -> None:
    wired = _wire(
        _activity(_ROOT, started_seq=2), _activity(_CHILD, parent=_ROOT, started_seq=7)
    )
    async with _client(wired) as client:
        first = await client.get(
            f"/v1/sessions/{wired.session_id}/threads", params={"limit": 1}
        )
        assert first.status_code == 200
        token = first.json()["next_page"]
        assert token is not None, "a page with a thread behind it offered no token"
        second = await client.get(
            f"/v1/sessions/{wired.session_id}/threads",
            params={"limit": 1, "page": token},
        )
    assert [one["id"] for one in first.json()["data"]] == [_ROOT]
    assert [one["id"] for one in second.json()["data"]] == [_CHILD]
    assert second.json()["next_page"] is None


async def test_a_full_last_page_offers_no_token() -> None:
    """The look-ahead's whole purpose: a token that opens an empty page is unusable.

    Two threads, a limit of two. Without the extra row asked for and discarded, this
    answers a token, and a caller cannot tell an empty next page from one it has not
    read.
    """
    wired = _wire(
        _activity(_ROOT, started_seq=2), _activity(_CHILD, parent=_ROOT, started_seq=7)
    )
    async with _client(wired) as client:
        answer = await client.get(
            f"/v1/sessions/{wired.session_id}/threads", params={"limit": 2}
        )
    assert len(answer.json()["data"]) == 2
    assert answer.json()["next_page"] is None


async def test_a_page_token_this_surface_did_not_issue_is_refused() -> None:
    wired = _wire(_activity(_ROOT))
    async with _client(wired) as client:
        answer = await client.get(
            f"/v1/sessions/{wired.session_id}/threads", params={"page": "not-a-token"}
        )
    assert answer.status_code == 400
    assert answer.json()["error"]["code"] == ErrorCode.PAGINATION_CURSOR_INVALID.value


async def test_another_tenants_session_answers_as_though_it_did_not_exist() -> None:
    wired = _wire(_activity(_ROOT))
    async with _client(wired) as client:
        answer = await client.get(
            f"/v1/sessions/{wired.session_id}/threads",
            headers={TENANT_HEADER: str(_STRANGER)},
        )
    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == ErrorCode.SESSION_NOT_FOUND.value


async def test_one_thread_is_read_by_name() -> None:
    wired = _wire(_activity(_ROOT), _activity(_CHILD, parent=_ROOT, started_seq=7))
    async with _client(wired) as client:
        answer = await client.get(f"/v1/sessions/{wired.session_id}/threads/{_CHILD}")
    assert answer.status_code == 200
    assert answer.json()["id"] == _CHILD
    assert answer.json()["type"] == "session_thread"


async def test_a_thread_this_session_does_not_hold_is_not_found() -> None:
    wired = _wire(_activity(_ROOT))
    async with _client(wired) as client:
        answer = await client.get(f"/v1/sessions/{wired.session_id}/threads/{_CHILD}")
    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == ErrorCode.THREAD_NOT_FOUND.value


async def test_a_thread_whose_turn_is_open_reads_as_running() -> None:
    wired = _wire(_activity(_ROOT, turn_ended=False))
    async with _client(wired) as client:
        answer = await client.get(f"/v1/sessions/{wired.session_id}/threads/{_ROOT}")
    assert answer.json()["status"] == "running"


async def test_every_thread_of_a_stopped_session_reads_as_terminated() -> None:
    wired = _wire(_activity(_ROOT, turn_ended=False), stopped=True)
    async with _client(wired) as client:
        answer = await client.get(f"/v1/sessions/{wired.session_id}/threads/{_ROOT}")
    assert answer.json()["status"] == "terminated"


async def test_archiving_an_idle_thread_records_it_and_answers_the_new_state() -> None:
    wired = _wire(_activity(_ROOT, turn_ended=True))
    async with _client(wired) as client:
        answer = await client.post(
            f"/v1/sessions/{wired.session_id}/threads/{_ROOT}/archive"
        )
    assert answer.status_code == 200, answer.text
    assert answer.json()["archived_at_ms"] == _ARCHIVED_AT_MS
    assert answer.json()["status"] == "terminated"
    assert wired.appends.written == [
        (thread_events.THREAD_ARCHIVED, {"thread_id": _ROOT})
    ]


async def test_archiving_a_thread_whose_turn_is_still_running_is_refused() -> None:
    """And nothing is written. An archive already in the log would tell a consumer to
    stop reading before the events it was waiting for arrived."""
    wired = _wire(_activity(_ROOT, turn_ended=False))
    async with _client(wired) as client:
        answer = await client.post(
            f"/v1/sessions/{wired.session_id}/threads/{_ROOT}/archive"
        )
    assert answer.status_code == 409
    assert answer.json()["error"]["code"] == ErrorCode.THREAD_RUNNING.value
    assert wired.appends.written == []


async def test_archiving_twice_keeps_the_first_retirement() -> None:
    wired = _wire(_activity(_ROOT, turn_ended=True))
    async with _client(wired) as client:
        first = await client.post(
            f"/v1/sessions/{wired.session_id}/threads/{_ROOT}/archive"
        )
        second = await client.post(
            f"/v1/sessions/{wired.session_id}/threads/{_ROOT}/archive"
        )
    assert first.json()["archived_at_ms"] == second.json()["archived_at_ms"]
    assert len(wired.appends.written) == 1, wired.appends.written


async def test_archiving_a_thread_of_a_stopped_session_writes_nothing() -> None:
    """It is already terminated, so the archive has nothing left to guarantee."""
    wired = _wire(_activity(_ROOT, turn_ended=True), stopped=True)
    async with _client(wired) as client:
        answer = await client.post(
            f"/v1/sessions/{wired.session_id}/threads/{_ROOT}/archive"
        )
    assert answer.status_code == 200
    assert answer.json()["status"] == "terminated"
    assert wired.appends.written == []


async def test_a_threads_events_are_the_sessions_events_narrowed_to_it() -> None:
    wired = _wire(_activity(_ROOT), _activity(_CHILD, parent=_ROOT, started_seq=7))
    wired.log.add(wired.session_id, turn.TURN_MESSAGE_DELTA, thread_id=_ROOT, text="a")
    wired.log.add(wired.session_id, turn.TURN_MESSAGE_DELTA, thread_id=_CHILD, text="b")
    wired.log.add(wired.session_id, turn.TURN_MESSAGE_DELTA, thread_id=_ROOT, text="c")
    async with _client(wired) as client:
        answer = await client.get(
            f"/v1/sessions/{wired.session_id}/threads/{_CHILD}/events"
        )
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert [event["payload"]["text"] for event in body["events"]] == ["b"]
    assert body["from_seq"] == 1, "the span asked for is the span reported"


async def test_a_threads_event_span_reports_the_range_asked_for_not_the_count() -> None:
    """A short page over a wide span is the honest answer, and the reason is stated in
    the route: a page trimmed to a full count would hide how far through the log it got,
    and a caller could not tell a quiet thread from the end of the log."""
    wired = _wire(_activity(_ROOT))
    wired.log.add(wired.session_id, turn.TURN_MESSAGE_DELTA, thread_id=_ROOT, text="a")
    async with _client(wired) as client:
        answer = await client.get(
            f"/v1/sessions/{wired.session_id}/threads/{_ROOT}/events",
            params={"from_seq": 1, "to_seq": 50},
        )
    body = answer.json()
    assert (body["from_seq"], body["to_seq"]) == (1, 50)
    assert len(body["events"]) == 1


async def test_a_thread_event_read_on_an_unknown_thread_is_not_found() -> None:
    wired = _wire(_activity(_ROOT))
    async with _client(wired) as client:
        answer = await client.get(
            f"/v1/sessions/{wired.session_id}/threads/{_CHILD}/events"
        )
    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == ErrorCode.THREAD_NOT_FOUND.value


async def test_a_malformed_span_is_refused_before_the_thread_is_looked_up() -> None:
    """So the refusal says nothing about whether the thread exists."""
    wired = _wire(_activity(_ROOT))
    async with _client(wired) as client:
        known = await client.get(
            f"/v1/sessions/{wired.session_id}/threads/{_ROOT}/events",
            params={"from_seq": 9, "to_seq": 2},
        )
        unknown = await client.get(
            f"/v1/sessions/{wired.session_id}/threads/{_CHILD}/events",
            params={"from_seq": 9, "to_seq": 2},
        )
    assert known.status_code == unknown.status_code == 400
    assert known.json()["error"] == unknown.json()["error"]


async def test_the_thread_stream_carries_only_this_threads_events() -> None:
    """And keeps the Session's sequence as the SSE id, gaps included.

    The gaps are the contract: a resume position has to name a place in the log, so a
    thread-local ordinal could not be resolved after a retention sweep. A consumer
    treating consecutive ids as a completeness check would read every sibling's event as
    loss, which is why the route says so.
    """
    wired = _wire(_activity(_ROOT), _activity(_CHILD, parent=_ROOT, started_seq=7))
    wired.log.add(wired.session_id, turn.TURN_MESSAGE_DELTA, thread_id=_ROOT, text="a")
    mine = wired.log.add(
        wired.session_id, turn.TURN_MESSAGE_DELTA, thread_id=_CHILD, text="b"
    )
    wired.log.add(wired.session_id, turn.TURN_MESSAGE_DELTA, thread_id=_ROOT, text="c")
    later = wired.log.add(
        wired.session_id, turn.TURN_MESSAGE_DELTA, thread_id=_CHILD, text="d"
    )
    scope = _scope(
        f"/v1/sessions/{wired.session_id}/threads/{_CHILD}/stream", "", _HEADERS
    )
    async with LiveResponse(wired.app, scope) as live:  # type: ignore[arg-type]
        collected = await live.frames(2)
    assert [f"id: {mine}", f"id: {later}"] == [
        frame.splitlines()[0] for frame in collected
    ]
    texts = [json.loads(frame.split("data: ", 1)[1])["text"] for frame in collected]
    assert texts == ["b", "d"]


async def test_the_thread_stream_refuses_a_thread_this_session_does_not_hold() -> None:
    wired = _wire(_activity(_ROOT))
    scope = _scope(
        f"/v1/sessions/{wired.session_id}/threads/{_CHILD}/stream", "", _HEADERS
    )
    async with LiveResponse(wired.app, scope) as live:  # type: ignore[arg-type]
        assert live.status == 404


async def test_the_session_stream_still_carries_events_that_name_no_thread() -> None:
    """The control that stops the predicate from being applied to the wrong route.

    A Session's own lifecycle events belong to no thread, so a filter that leaked onto
    the Session stream would drop exactly the events that stream exists to deliver --
    and every thread test above would still pass.
    """
    wired = _wire(_activity(_ROOT))
    scope = _scope(f"/v1/sessions/{wired.session_id}/events/stream", "", _HEADERS)
    async with LiveResponse(wired.app, scope) as live:  # type: ignore[arg-type]
        collected = await live.frames(1)
    assert f"event: {lifecycle.SESSION_CREATED}" in collected[0]


@pytest.mark.parametrize(
    "path",
    [
        "threads",
        f"threads/{_ROOT}",
        f"threads/{_ROOT}/events",
    ],
)
async def test_every_read_refuses_a_session_it_cannot_see(path: str) -> None:
    """One case per route: the registry lookup is the only thing scoping any of them.

    The log and the thread index are both keyed by Session and carry no tenant.
    """
    wired = _wire(_activity(_ROOT))
    async with _client(wired) as client:
        answer = await client.get(
            f"/v1/sessions/{wired.session_id}/{path}",
            headers={TENANT_HEADER: str(_STRANGER)},
        )
    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == ErrorCode.SESSION_NOT_FOUND.value


async def test_archiving_refuses_a_session_it_cannot_see_and_writes_nothing() -> None:
    wired = _wire(_activity(_ROOT, turn_ended=True))
    async with _client(wired) as client:
        answer = await client.post(
            f"/v1/sessions/{wired.session_id}/threads/{_ROOT}/archive",
            headers={TENANT_HEADER: str(_STRANGER)},
        )
    assert answer.status_code == 404
    assert wired.appends.written == []


def test_a_pod_may_not_retire_a_thread() -> None:
    """The archive is the control plane's event, and the wire must not accept it.

    `ShimEventType` in `shim/serve.py` is a second closed set, checked again by the
    control plane when a pod's line arrives, and a type in it is one a Session pod can
    put on the wire. Retirement is a decision a tenant made through the API, so a pod
    that could append it could retire a thread the tenant is still reading.

    Asserted here rather than beside the wire, because this is where the claim is made:
    the archive route's contract is that it is the only writer of this event.
    """
    from managed_agent.core.vocabulary import is_published
    from managed_agent.session_shim.serve import SHIM_EVENT_TYPES

    assert is_published(thread_events.THREAD_ARCHIVED), (
        "an unpublished type is dropped by the stream, so no caller would ever see it"
    )
    assert thread_events.THREAD_ARCHIVED not in SHIM_EVENT_TYPES
