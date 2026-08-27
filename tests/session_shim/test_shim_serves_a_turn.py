"""A Turn reaching the runtime through the shim, and every way that is refused.

Tier 1 (local, no infrastructure). This drives both halves of the transport against each
other: `create_shim_app` over a real unix-socket Agent Runtime stand-in, and
`HttpPodDispatch` over an ASGI transport into that same app. One case therefore
exercises the real wire format in both directions rather than asserting that each half
agrees with a fake of the other.

**What is real here and what is not.** The WebSocket unix socket is real, the frames are
the runtime's own shape, the HTTP route is the real route and the appends are real
appends into an in-memory log. No pod is created by anything in this file: nothing in
this tree builds the `map-session` image and nothing implements the cluster `PodRunner`,
so "a Kubernetes pod answers on this address" is asserted nowhere and is not known.

Three shapes of case exist here because a guard without them is not a guard:

- Every refusal case that asserts the runtime was *not* touched goes on to touch it in
  the same test, so the negative assertion runs in a world where the positive event is
  possible.
- The wire type and the control plane's own check answer the same question about an
  event type, so each is pinned by a case the other cannot satisfy: one parses a line,
  one calls the check.
- The read deadline cannot be driven end to end without a real wedged pod, so it is
  proved in three linked pieces, each of which is real, and the seam between them is
  named rather than glossed.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Iterable, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fake_agent_runtime import FakeAgentRuntime, fake_agent_runtime
from fastapi import FastAPI
from pydantic import TypeAdapter, ValidationError
from websockets.asyncio.server import unix_serve

from managed_agent.control.files.output_shipout import (
    OutputNotRevisable,
    OutputsNotShippable,
)
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import Placement, PodPhase
from managed_agent.control.session.turn_dispatch import (
    TurnOutputNotRevisable,
    TurnUndeliverable,
)
from managed_agent.core.ids import Seq, SessionId, TurnId, new_session_id, new_turn_id
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import (
    PUBLISHED,
    lifecycle,
    thread,
    tool_call,
    tool_server,
    turn,
)
from managed_agent.session_shim.client import RuntimeConnection
from managed_agent.session_shim.pod_channel import (
    MAX_LINES_PER_TURN,
    HttpPodDispatch,
    refuse_an_event_the_pod_may_not_write,
    shim_token_for,
    shim_url_for,
)
from managed_agent.session_shim.serve import (
    READY_ROUTE,
    SHIM_EVENT_TYPES,
    SHIM_PORT,
    TURN_ROUTE,
    RunTurn,
    ServedSession,
    TurnLine,
    build_shim_app,
    connect_when_the_runtime_is_listening,
    create_shim_app,
)

_SRC = Path(__file__).resolve().parents[2] / "src" / "managed_agent"
_KEY = b"the platform's shim signing key, for tests only"
_THREAD = "thread-inside-the-pod"
_RUNTIME_TURN = "tn_fake_1"
_PROMPT = "summarise the findings"
_NAMESPACE = "map-sessions"

_LINE: TypeAdapter[TurnLine] = TypeAdapter(TurnLine)


# ------------------------------------------------------------------------------------
# Fixtures: an in-memory log, a cluster that answers one phase, a shim on a transport
# ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class InMemoryLog:
    """An `EventLogAppend` over a list, numbering its own appends from 1."""

    def __init__(self) -> None:
        self.written: list[Event] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.written.append(
            Event(session_id, Seq(len(self.written) + 1), type_, dict(payload))
        )
        return Seq(len(self.written))

    def types(self) -> list[str]:
        return [event.type for event in self.written]


class FixedPhase:
    """A cluster that reports one phase for every pod and starts nothing.

    `remove` records rather than refusing, which it used to do. Under ADR-041 a pod is
    leased for one Turn, so a dispatch that did NOT remove one would be the defect --
    the refusal was written when a pod outlived its Turn and it now asserts the
    opposite of the contract. The names are kept so a case that wants to grade the
    release can, and `phase` is deliberately left alone by it: every case here fixes
    the phase it wants and none is about what the cluster looks like afterwards.
    """

    def __init__(self, phase: PodPhase = PodPhase.RUNNING) -> None:
        self.phase = phase
        self.asked: list[str] = []
        self.removed: list[str] = []

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        raise AssertionError("dispatching a Turn tried to start a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        self.asked.append(pod_name)
        return self.phase

    async def remove(self, pod_name: str) -> None:
        self.removed.append(pod_name)


class Notified:
    def __init__(self) -> None:
        self.told: list[tuple[SessionId, TurnId]] = []

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        self.told.append((session_id, turn_id))


class SeamThatRaises:
    """A completion seam that took the Turn and then failed to make it durable."""

    def __init__(self, failure: Exception) -> None:
        self._failure = failure
        self.told: list[tuple[SessionId, TurnId]] = []

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        self.told.append((session_id, turn_id))
        raise self._failure


class ShimThatStreams(httpx.AsyncBaseTransport):
    """A stand-in for a pod's shim: one status line, then the lines it was given.

    Used where the case is about what the *control plane* does with a stream, including
    the streams a correct shim would never send -- which is the point, since the pod is
    the untrusted end and a compromised one is not bound by this repository's models.
    """

    def __init__(self, lines: Iterable[str], status: int = 200) -> None:
        self._lines = lines
        self._status = status
        self.requests: list[httpx.Request] = []
        self.deadlines: list[dict[str, float | None]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.deadlines.append(dict(request.extensions.get("timeout", {})))
        lines = self._lines

        async def body() -> AsyncIterator[bytes]:
            for line in lines:
                yield line.encode() + b"\n"

        return httpx.Response(
            self._status,
            headers={"content-type": "application/x-ndjson"},
            content=body(),
        )


class ShimThatCannotBeReached(httpx.AsyncBaseTransport):
    """A pod that went silent long enough for the read deadline to fire."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no byte arrived within the read deadline")


def _event(type_: str, **payload: object) -> str:
    return f'{{"kind":"event","type":"{type_}","payload":{_json(payload)}}}'


def _json(payload: dict[str, object]) -> str:
    items = ",".join(f'"{name}":"{value}"' for name, value in payload.items())
    return f"{{{items}}}"


_COMPLETED_LINE = '{"kind":"completed"}'


def _a_whole_turn(turn_id: TurnId) -> list[str]:
    return [
        _event(turn.TURN_STARTED, turn_id=str(turn_id)),
        _event(turn.TURN_MESSAGE_DELTA, turn_id=str(turn_id), text="the answer"),
        _event(turn.TURN_COMPLETED, turn_id=str(turn_id), text="the answer"),
        _COMPLETED_LINE,
    ]


class NeverPlaces:
    """The `SessionPods` seam, refusing every call.

    A Turn places a Session's pod only when `locate` answers ABSENT, and no case in this
    file reports that phase. Refusing rather than recording makes that a property this
    file asserts instead of one it merely happens to have: a case that started reaching
    placement fails here, naming it, rather than quietly exercising a collaborator
    nothing in this file was written to grade.
    """

    async def ensure_for(self, session_id: SessionId) -> None:
        raise AssertionError("a test in this file placed a Session's pod")


def _dispatch_over(
    transport: httpx.AsyncBaseTransport,
    log: InMemoryLog,
    notified: Notified | SeamThatRaises,
    phase: PodPhase = PodPhase.RUNNING,
) -> HttpPodDispatch:
    return HttpPodDispatch(
        placement=Placement(FixedPhase(phase)),
        pods=NeverPlaces(),
        log=log,
        on_completed=notified,
        namespace=_NAMESPACE,
        token_key=_KEY,
        transport=transport,
    )


# ------------------------------------------------------------------------------------
# Frames in the Agent Runtime's own shape
# ------------------------------------------------------------------------------------


def _started() -> dict[str, Any]:
    return {
        "method": "turn/started",
        "params": {"threadId": _THREAD, "turn": {"id": _RUNTIME_TURN, "items": []}},
    }


def _delta(text: str) -> dict[str, Any]:
    return {
        "method": "item/agentMessage/delta",
        "params": {
            "threadId": _THREAD,
            "turnId": _RUNTIME_TURN,
            "itemId": "item-1",
            "delta": text,
        },
    }


def _completed() -> dict[str, Any]:
    return {
        "method": "turn/completed",
        "params": {
            "threadId": _THREAD,
            "turn": {"id": _RUNTIME_TURN, "items": [], "status": "completed"},
        },
    }


@asynccontextmanager
async def _a_shim_serving(
    session_id: SessionId, token: str
) -> AsyncIterator[tuple[FastAPI, FakeAgentRuntime]]:
    """The real app, holding a real connection to a stand-in runtime next door."""
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            served = ServedSession(
                session_id=session_id,
                thread_id=_THREAD,
                connection=connection,
                token=token,
            )
            yield create_shim_app(served), runtime
        finally:
            await connection.close()


async def _answer_one_turn(runtime: FakeAgentRuntime, text: str = "the answer") -> None:
    """Wait for the shim to start a Turn, then send the frames that finish it."""
    await runtime.wait_until(
        lambda: "turn/start" in runtime.methods_received,
        "the shim never started a turn against the runtime",
    )
    for frame in (_started(), _delta(text), _completed()):
        await runtime.push(frame)


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pod.map-session"
    )


def _body(session_id: SessionId, turn_id: TurnId) -> dict[str, str]:
    return {
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "prompt": _PROMPT,
    }


_REFUSAL_DEADLINE_S = 5.0


async def _asking_to_be_refused(
    client: httpx.AsyncClient,
    session_id: SessionId,
    turn_id: TurnId,
    token: str | None = None,
) -> httpx.Response:
    """POST a Turn that must be refused, and require the answer promptly.

    Bounded, because a route that stopped refusing does not fail these cases -- it
    *accepts* the Turn and holds the response open until the runtime produces something,
    and in a refusal case nothing ever will. Deleting the token check turned this file
    into a suite that hung for two minutes and then had to be killed, which is the shape
    of a guard that cannot report. The deadline makes that a named failure in seconds.
    """
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    try:
        return await asyncio.wait_for(
            client.post(TURN_ROUTE, json=_body(session_id, turn_id), headers=headers),
            timeout=_REFUSAL_DEADLINE_S,
        )
    except TimeoutError:
        raise AssertionError(
            f"the route did not refuse this request within {_REFUSAL_DEADLINE_S}s; it "
            "accepted the Turn and is holding the stream open, which means the check "
            "that should have refused it is gone"
        ) from None


# ------------------------------------------------------------------------------------
# The wire: what a line may say
# ------------------------------------------------------------------------------------


def test_the_types_the_pod_may_write_are_the_turn_family_less_submitted() -> None:
    """The link between a `Literal` of strings and the published vocabulary.

    `Literal` takes literal strings, so `serve.ShimEventType` cannot be written in terms
    of the constants in `core/vocabulary/`. This is what makes the two one set: a turn
    type added to the vocabulary fails here rather than failing a Turn in a pod.

    `turn.submitted` is excluded and that is deliberate rather than an oversight. The
    control plane writes it when it admits a Turn and the pod never produces one, while
    a pod that could write it could plant an idempotency key -- and a later genuine
    submission under that key is then answered as a replay of a Turn that never ran.

    `tool.called` is admitted from OUTSIDE the turn family, and it is named here as one
    string rather than by admitting its family wholesale. The rest of the `tool` family
    is not the pod's to write -- `tool.progress` and the elicitation pair reach the log
    from a proxied server -- so admitting the family would hand the pod three types it
    has no business producing. Spelling the one member makes the next such admission a
    deliberate edit here.

    `tool.server_unavailable` is the third, named here for the same reason the other
    two are: it sits outside the turn family, so this comparison would not notice its
    absence from the wire set, and the pod's permission to write it would then be
    asserted in no test at all.

    `thread.started` is the second, admitted the same way and for the same reason, and
    its arrival is why this case is no longer described as the guard against a mapping
    that emits an extra type. It never was: a type in a family this comparison does not
    read changes neither side, so `thread.started` passed here and then killed every
    Turn in the cluster with a validation error from `TurnEventLine`. The guard that
    catches that shape is in `test_turn_runner.py` and derives its expectation from
    `_MAPPED` itself.
    """
    family = {name for name, in_family in PUBLISHED.items() if in_family == turn.FAMILY}

    assert family, "the turn family is empty, so this comparison asserts nothing"
    assert tool_call.TOOL_CALLED not in family, (
        "tool.called joined the turn family, so the assertion below no longer says "
        "anything about it and the pod's permission to write it is stated nowhere"
    )
    assert thread.THREAD_STARTED not in family, (
        "thread.started joined the turn family, so the assertion below no longer says "
        "anything about it and the pod's permission to write it is stated nowhere"
    )
    assert tool_server.TOOL_SERVER_UNAVAILABLE not in family, (
        "tool.server_unavailable joined the turn family, so the assertion below no "
        "longer says anything about it and the pod's permission to write it is stated "
        "nowhere"
    )
    expected = (
        (family - {turn.TURN_SUBMITTED})
        | {tool_call.TOOL_CALLED}
        | {thread.THREAD_STARTED}
        | {tool_server.TOOL_SERVER_UNAVAILABLE}
    )
    assert expected == SHIM_EVENT_TYPES


def test_a_line_naming_an_event_type_the_pod_may_not_write_is_not_a_line() -> None:
    """The wire type is the first of the two gates over what a pod may append.

    `session.stopped` is the case that matters: it folds a Session to STOPPED, so a pod
    able to write it can make its own Session refuse every later Turn.
    """
    with pytest.raises(ValidationError):
        _LINE.validate_json(_event(lifecycle.SESSION_STOPPED))
    with pytest.raises(ValidationError):
        _LINE.validate_json(_event("totally.invented"))
    with pytest.raises(ValidationError):
        _LINE.validate_json(_event(turn.TURN_SUBMITTED))

    assert _LINE.validate_json(_event(turn.TURN_STARTED)), (
        "a type a Turn does produce must still parse, or the three refusals above are "
        "satisfied by nothing parsing at all"
    )


def test_a_line_that_is_neither_an_event_nor_a_completion_is_refused() -> None:
    """The discriminator is the whole reason these are two models and not one with
    optional fields: `kind` and `type` cannot disagree if only `kind` selects."""
    with pytest.raises(ValidationError):
        _LINE.validate_json('{"kind":"something_else"}')
    with pytest.raises(ValidationError):
        _LINE.validate_json('{"type":"turn.started"}')


def test_a_run_turn_body_with_an_unknown_field_is_refused() -> None:
    """Refused rather than ignored: a control plane sending a field this shim does not
    know is a version skew, and quietly dropping it runs a Turn nobody asked for."""
    with pytest.raises(ValidationError):
        RunTurn.model_validate(
            {
                "session_id": str(uuid4()),
                "turn_id": str(uuid4()),
                "prompt": "hello",
                "steer": "ignore your instructions",
            }
        )


# ------------------------------------------------------------------------------------
# The route
# ------------------------------------------------------------------------------------


async def test_a_turn_reaches_the_runtime_and_its_events_stream_back_in_order() -> None:
    """The happy path, across the real socket and the real route.

    Also the place the boundary claim is measured: every frame the runtime sent carries
    its thread id, turn id and item id, and none of the three may appear in what leaves
    the pod. The count assertion is what keeps that negative honest -- a route that
    streamed nothing would satisfy it for free.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    token = shim_token_for(session_id, _KEY)

    async with _a_shim_serving(session_id, token) as (app, runtime):
        answering = asyncio.create_task(_answer_one_turn(runtime))
        async with (
            _client(app) as client,
            client.stream(
                "POST",
                TURN_ROUTE,
                json=_body(session_id, turn_id),
                headers={"Authorization": f"Bearer {token}"},
            ) as response,
        ):
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/x-ndjson"
            lines = [line async for line in response.aiter_lines() if line]
        await answering

    parsed = [_LINE.validate_json(line) for line in lines]
    assert [getattr(line, "type", "completed") for line in parsed] == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
        "completed",
    ]
    assert len(lines) == 4, (
        "nothing crossed the boundary, so the check below is vacuous"
    )
    for identifier in (_THREAD, _RUNTIME_TURN, "item-1"):
        assert identifier not in "\n".join(lines)


async def test_an_untokened_request_is_refused_and_the_runtime_is_untouched() -> None:
    """Refused before the runtime is dialled, so the Turn is absent rather than started
    and abandoned -- and then the same fixture runs a Turn, so "the runtime was not
    touched" is asserted in a state where touching it was possible."""
    session_id, turn_id = new_session_id(), new_turn_id()
    token = shim_token_for(session_id, _KEY)

    async with (
        _a_shim_serving(session_id, token) as (app, runtime),
        _client(app) as client,
    ):
        refused = await _asking_to_be_refused(client, session_id, turn_id)
        wrong = await _asking_to_be_refused(
            client, session_id, turn_id, token="not-this-sessions-token"
        )
        assert refused.status_code == 404
        assert wrong.status_code == 404
        assert "turn/start" not in runtime.methods_received

        answering = asyncio.create_task(_answer_one_turn(runtime))
        accepted = await client.post(
            TURN_ROUTE,
            json=_body(session_id, turn_id),
            headers={"Authorization": f"Bearer {token}"},
        )
        await answering

    assert accepted.status_code == 200
    assert "turn/start" in runtime.methods_received


async def test_another_sessions_turn_is_refused_exactly_as_an_untokened_one() -> None:
    """Byte-identical, which is what makes the refusal useless as an existence oracle.

    The second arm presents **this pod's own correct token** and names another Session,
    which is the input that separates the two checks. An earlier version presented the
    other Session's token instead: the token check refused that on its own, both arms
    answered identically for one reason, and deleting the Session comparison outright
    left the whole file green. The agent inside this pod is the caller that makes the
    distinction real -- it shares the pod with the token, so holding the token cannot be
    what decides which Session a Turn runs under.
    """
    served_id, other_id = new_session_id(), new_session_id()
    turn_id = new_turn_id()

    async with (
        _a_shim_serving(served_id, shim_token_for(served_id, _KEY)) as (app, runtime),
        _client(app) as client,
    ):
        untokened = await _asking_to_be_refused(client, other_id, turn_id)
        with_this_pods_token = await _asking_to_be_refused(
            client, other_id, turn_id, token=shim_token_for(served_id, _KEY)
        )
        assert "turn/start" not in runtime.methods_received

    assert untokened.status_code == with_this_pods_token.status_code == 404
    assert untokened.content == with_this_pods_token.content
    assert untokened.json()["code"] == "session.not_found"
    assert untokened.json()["detail"] == {"session_id": str(other_id)}


async def test_the_refusal_is_the_platform_envelope_not_some_other_shape() -> None:
    """Asserted here because nothing else reaches this module.

    `tests/core/test_closed_error_set.py` walks `control/api/` only, so the rule that
    every coded refusal comes from the published set is unenforced on this route. What
    that check would have said is said directly: the code is a member of the published
    enum, the status is the one the set assigns it, and the body has the envelope's
    three fields and nothing else.
    """
    session_id = new_session_id()

    async with (
        _a_shim_serving(session_id, shim_token_for(session_id, _KEY)) as (
            app,
            _,
        ),
        _client(app) as client,
    ):
        refused = await _asking_to_be_refused(client, new_session_id(), new_turn_id())

    body = refused.json()
    assert refused.status_code == 404
    assert set(body) == {"code", "message", "detail"}
    assert body["code"] == "session.not_found"
    assert body["message"]


async def test_a_non_ascii_bearer_token_is_refused_not_crashed_on() -> None:
    """The one refusal an HTTP client cannot even send, driven at the ASGI layer.

    Header values arrive latin-1-decoded, and `hmac.compare_digest` on two `str` values
    raises `TypeError` when either holds a character above 0x7F. Compared as `str` this
    route answers an unauthenticated caller with an unhandled exception -- a bare 500,
    outside the published set, reachable by anything on the cluster network. httpx
    refuses to put the byte on the wire, which is exactly why this drives the app
    directly instead.
    """
    session_id, turn_id = new_session_id(), new_turn_id()

    async with _a_shim_serving(session_id, shim_token_for(session_id, _KEY)) as (
        app,
        runtime,
    ):
        status, body = await _post_raw(
            app,
            headers=[
                (b"host", b"pod.map-session"),
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer \xfc\xfc\xfc"),
            ],
            payload=_body(session_id, turn_id),
        )
        assert "turn/start" not in runtime.methods_received

    assert status == 404, body
    assert b'"session.not_found"' in body


async def test_readiness_is_503_before_the_session_is_open_and_204_after() -> None:
    """The probe the headless Service publishes on.

    `build_shim_app` opens its Session in the lifespan, which is not run here, so its
    app is the not-ready state. uvicorn completes lifespan startup before it listens, so
    in a pod a probe in that window is refused at the socket instead -- both are
    not-ready, and this is the answer if the route is ever reached without a Session.
    """
    session_id = new_session_id()

    async with _client(build_shim_app()) as unopened:
        before = await unopened.get(READY_ROUTE)

    async with (
        _a_shim_serving(session_id, shim_token_for(session_id, _KEY)) as (
            app,
            _,
        ),
        _client(app) as opened,
    ):
        after = await opened.get(READY_ROUTE)

    assert before.status_code == 503
    assert after.status_code == 204


async def _post_raw(
    app: FastAPI, headers: list[tuple[bytes, bytes]], payload: dict[str, str]
) -> tuple[int, bytes]:
    """One POST straight into the ASGI app, with headers httpx would not send.

    Nothing between the caller and the route: an exception the route raises escapes
    here rather than being turned into a response, which is what lets a case assert
    that the route answered at all.
    """
    sent: list[dict[str, Any]] = []
    delivered = False

    async def receive() -> dict[str, Any]:
        """The body once, then a disconnect -- which is what a real server sends.

        Not "the request, forever". Starlette's streaming response calls this in a loop
        waiting for `http.disconnect`, and a coroutine that returns without awaiting
        anything never yields to the event loop: repeating the request turns that loop
        into a spin that no timeout can interrupt, because no timeout gets to run.
        """
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {
            "type": "http.request",
            "body": json.dumps(payload).encode(),
            "more_body": False,
        }

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(dict(message))

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": TURN_ROUTE,
        "raw_path": TURN_ROUTE.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("10.0.0.9", 55555),
        "server": ("10.0.0.8", SHIM_PORT),
    }
    try:
        await asyncio.wait_for(app(scope, receive, send), timeout=_REFUSAL_DEADLINE_S)
    except TimeoutError:
        raise AssertionError(
            f"the route did not answer within {_REFUSAL_DEADLINE_S}s; it accepted this "
            "request and is streaming a Turn, so whatever should have refused it is "
            "gone. Bounded because every caller of this helper expects a refusal, and "
            "an unbounded await here hangs the suite instead of failing it."
        ) from None
    start = [message for message in sent if message["type"] == "http.response.start"]
    body = b"".join(
        bytes(message.get("body", b""))
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert start, f"the route sent no response at all: {sent}"
    status: int = start[0]["status"]
    return status, body


# ------------------------------------------------------------------------------------
# Starting up beside a runtime that is not listening yet
# ------------------------------------------------------------------------------------


async def test_the_shim_waits_out_a_runtime_socket_that_appears_late() -> None:
    """The pod's own start-up race, which the shim loses on the first attempt.

    Both containers start together and the runtime binds its socket afterwards -- the
    manifest budgets thirty two-second attempts for it. A shim that dialled once would
    exit, and with `restartPolicy: Never` that pod never runs a Turn again.
    """
    directory = Path(tempfile.mkdtemp(prefix="map-"))
    socket_path = directory / "ctl.sock"
    try:
        connecting = asyncio.create_task(
            connect_when_the_runtime_is_listening(
                socket_path, attempts=200, pause_s=0.01
            )
        )
        await asyncio.sleep(0.1)
        assert not connecting.done(), (
            "the socket does not exist yet, so a connect that has already finished "
            "either failed or never dialled"
        )
        fake = FakeAgentRuntime(socket_path)
        async with unix_serve(fake.handle, str(socket_path)):
            connection = await asyncio.wait_for(connecting, timeout=5.0)
            await connection.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


async def test_a_runtime_socket_that_never_appears_fails_after_the_budget() -> None:
    """Waiting is bounded. A socket that has not appeared is eventually absent, and the
    honest end is the error the last attempt got rather than an unready pod forever."""
    with pytest.raises(OSError):
        await connect_when_the_runtime_is_listening(
            Path("/nonexistent/map/ctl.sock"), attempts=3, pause_s=0.001
        )


# ------------------------------------------------------------------------------------
# The address and the token, both computed
# ------------------------------------------------------------------------------------


def test_the_address_is_computed_from_the_pod_name_alone() -> None:
    """No table holds it, so nothing can disagree with the cluster about where a
    Session is. The name resolves only because the pod carries the matching subdomain
    and the headless Service exists -- `tests/deploy/` is what asserts both."""
    assert shim_url_for("map-session-abc", _NAMESPACE) == (
        f"http://map-session-abc.map-session.{_NAMESPACE}"
        f".svc.cluster.local:{SHIM_PORT}{TURN_ROUTE}"
    )


def test_two_sessions_get_different_tokens_and_neither_opens_the_other() -> None:
    """The blast radius argument, made testable.

    An agent that escapes its sandbox and reads its own token holds something that
    opens the shim it is already running inside, and nothing else.
    """
    one, other = new_session_id(), new_session_id()

    assert shim_token_for(one, _KEY) != shim_token_for(other, _KEY)
    assert shim_token_for(one, _KEY) == shim_token_for(one, _KEY)
    assert shim_token_for(one, _KEY) != shim_token_for(one, b"another signing key")


# ------------------------------------------------------------------------------------
# The dispatch
# ------------------------------------------------------------------------------------


async def test_a_dispatched_turn_runs_in_the_pod_and_lands_in_the_log() -> None:
    """Both halves at once: the control plane dials the real app, the app drives the
    real socket, and the events come back and are appended in arrival order.

    This is the case the platform did not have. Until the shim existed there was no
    listener in the pod, so `NoPodTransport` refused every Turn.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    log, notified = InMemoryLog(), Notified()

    async with _a_shim_serving(session_id, shim_token_for(session_id, _KEY)) as (
        app,
        runtime,
    ):
        answering = asyncio.create_task(_answer_one_turn(runtime))
        await _dispatch_over(httpx.ASGITransport(app=app), log, notified).dispatch(
            session_id, turn_id, _PROMPT
        )
        await answering

    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert log.written[-1].payload["text"] == "the answer"
    assert notified.told == [(session_id, turn_id)]
    assert {event.session_id for event in log.written} == {session_id}


async def test_a_pod_writing_a_lifecycle_event_stops_no_session() -> None:
    """The pod is the untrusted end, and this is what it would do with a free hand.

    `session.stopped` folds a Session to STOPPED, after which
    `control/api/routes/turns.py` refuses every later Turn for it. Nothing about the
    Session id is forged -- that comes from the caller -- so this reaches no other
    tenant; what it reaches is the pod's own Session, through the control plane's own
    credential.

    The event before the injection is appended and that is correct: it was a real mapped
    event. The Turn then ends as undeliverable, which is what closes it.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    log, notified = InMemoryLog(), Notified()
    await log.append(session_id, lifecycle.SESSION_CREATED, {})
    injected = ShimThatStreams(
        [
            _event(turn.TURN_STARTED, turn_id=str(turn_id)),
            _event(lifecycle.SESSION_STOPPED),
            _event(turn.TURN_COMPLETED, turn_id=str(turn_id)),
            _COMPLETED_LINE,
        ]
    )

    with pytest.raises(TurnUndeliverable):
        await _dispatch_over(injected, log, notified).dispatch(
            session_id, turn_id, _PROMPT
        )

    assert log.types() == [lifecycle.SESSION_CREATED, turn.TURN_STARTED]
    state, _ = project(log.written)
    assert state is not SessionState.STOPPED
    assert notified.told == []


def test_only_the_types_a_turn_produces_pass_the_second_gate() -> None:
    """The second gate, called directly.

    It cannot be reached from the wire, because the line model refuses the same types
    one step earlier -- so a case driving a stream would grade the model and report it
    as this. Two paths answering one question means a test through either covers
    neither; this is the input that separates them.
    """
    session_id = new_session_id()

    for allowed in SHIM_EVENT_TYPES:
        refuse_an_event_the_pod_may_not_write(allowed, session_id)

    for refused in (lifecycle.SESSION_STOPPED, turn.TURN_SUBMITTED, "totally.invented"):
        with pytest.raises(TurnUndeliverable, match="not one a Turn produces"):
            refuse_an_event_the_pod_may_not_write(refused, session_id)


async def test_a_stream_that_stops_mid_turn_closes_the_turn() -> None:
    """A stream with no terminal event means the pod went away with the Turn open.

    Returning quietly would leave a submission nobody ever explains: the log would say a
    Turn started and nothing would say it stopped. Raising is what makes the route close
    it as failed.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    log, notified = InMemoryLog(), Notified()
    stopped = ShimThatStreams([_event(turn.TURN_STARTED, turn_id=str(turn_id))])

    with pytest.raises(TurnUndeliverable, match="stopped mid-turn"):
        await _dispatch_over(stopped, log, notified).dispatch(
            session_id, turn_id, _PROMPT
        )

    assert log.types() == [turn.TURN_STARTED]
    assert notified.told == []


async def test_a_stream_past_the_line_cap_fails_the_turn_rather_than_hanging() -> None:
    """A pod that streams without stopping is bounded by a count, not by patience.

    The fixture stops just past the cap instead of running forever, and that is the
    load-bearing decision in this case rather than a convenience. Both the read loop and
    this transport are in-process and neither suspends, so a wall-clock deadline around
    the dispatch can never fire -- the event loop does not get control to run it. A
    truly endless fixture therefore hangs the suite instead of failing it, which is a
    guard that reports nothing.

    `match=` is what keeps the bounded fixture honest. With the cap removed the stream
    simply ends, and the dispatch still raises -- for the other reason, that no terminal
    event arrived. Matching the message is what tells the two apart.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    log, notified = InMemoryLog(), Notified()
    over_the_cap = ShimThatStreams(
        _event(turn.TURN_MESSAGE_DELTA, text="more")
        for _ in range(MAX_LINES_PER_TURN + 10)
    )

    with pytest.raises(TurnUndeliverable, match="streamed more than"):
        await _dispatch_over(over_the_cap, log, notified).dispatch(
            session_id, turn_id, _PROMPT
        )

    assert len(log.written) >= MAX_LINES_PER_TURN, (
        "the cap fired before the stream reached it, so this measures something else"
    )
    assert notified.told == []


async def test_the_rollout_is_notified_only_when_a_completion_line_arrived() -> None:
    """The completion line and the terminal event are different facts.

    A Turn the runtime reported as failed reaches `turn.failed` and a completion line,
    because the pod did finish it; a Turn whose stream ended at `turn.failed` with no
    completion line is a pod that went quiet, and shipping its state out is asking the
    wrong process.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    failed_line = _event(turn.TURN_FAILED, turn_id=str(turn_id), cause="runtime_lost")

    quiet_log, quiet_notified = InMemoryLog(), Notified()
    await _dispatch_over(
        ShimThatStreams([failed_line]), quiet_log, quiet_notified
    ).dispatch(session_id, turn_id, _PROMPT)

    finished_log, finished_notified = InMemoryLog(), Notified()
    await _dispatch_over(
        ShimThatStreams([failed_line, _COMPLETED_LINE]), finished_log, finished_notified
    ).dispatch(session_id, turn_id, _PROMPT)

    assert quiet_log.types() == finished_log.types() == [turn.TURN_FAILED]
    assert quiet_notified.told == []
    assert finished_notified.told == [(session_id, turn_id)]


async def test_the_dispatch_hands_its_transport_a_finite_read_deadline() -> None:
    """The first of the three pieces the wedged-pod bound is proved in.

    A pod that accepts a Turn and then sends nothing is bounded by no line count: the
    cap counts lines and there are none. With no read deadline the dispatch waits for
    the life of the process, and `control/api/routes/turns.py` awaits it inside the
    tenant's own POST -- so the tenant hangs too, holding a control-plane worker.

    In httpx a read deadline is inter-byte rather than total, which is why a Turn that
    legitimately runs for an hour with steady output is untouched by it.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    shim = ShimThatStreams(_a_whole_turn(turn_id))

    await _dispatch_over(shim, InMemoryLog(), Notified()).dispatch(
        session_id, turn_id, _PROMPT
    )

    assert shim.deadlines, "no request reached the transport, so nothing was measured"
    read = shim.deadlines[0]["read"]
    assert isinstance(read, float) and read > 0, (
        f"the dispatch handed its transport {shim.deadlines[0]}, so a pod that goes "
        "silent mid-Turn is bounded by nothing"
    )


async def test_a_read_deadline_is_what_bounds_a_peer_that_stops_sending() -> None:
    """The second piece, over a real socket rather than an in-process transport.

    An ASGI transport ignores timeouts entirely -- it never touches a socket -- so the
    case above can only show that the deadline is handed over. This shows what a handed-
    over read deadline does: against a server that answers its status line and then goes
    quiet, the read fails instead of waiting.
    """
    quiet = asyncio.Event()

    async def answer_then_go_quiet(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.read(1024)
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/x-ndjson\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        await writer.drain()
        await quiet.wait()
        writer.close()

    server = await asyncio.start_server(answer_then_go_quiet, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=0.25, write=2.0, pool=2.0)
        ) as client:
            with pytest.raises(httpx.ReadTimeout):
                async with client.stream(
                    "POST", f"http://127.0.0.1:{port}{TURN_ROUTE}", json={}
                ) as response:
                    async for _ in response.aiter_lines():
                        pass
    finally:
        quiet.set()
        server.close()
        await server.wait_closed()


async def test_a_read_that_times_out_closes_the_turn_rather_than_escaping() -> None:
    """The third piece: what the dispatch does with the error the other two produce.

    `TurnUndeliverable` is the only exception `control/api/routes/turns.py` catches, so
    an `httpx` error escaping from here would reach the tenant as a 500 with the Turn
    left open instead of a published refusal with the Turn recorded as failed.
    """
    with pytest.raises(TurnUndeliverable, match="could not be reached"):
        await _dispatch_over(
            ShimThatCannotBeReached(), InMemoryLog(), Notified()
        ).dispatch(new_session_id(), new_turn_id(), _PROMPT)


async def test_a_pod_that_is_not_running_is_refused_without_being_dialled() -> None:
    """The cluster's own answer, in milliseconds, not a connection timing out."""
    unreachable = ShimThatStreams([])

    with pytest.raises(TurnUndeliverable, match="is starting"):
        await _dispatch_over(
            unreachable, InMemoryLog(), Notified(), phase=PodPhase.STARTING
        ).dispatch(new_session_id(), new_turn_id(), _PROMPT)

    assert unreachable.requests == []


async def test_a_shim_that_refuses_the_turn_makes_it_undeliverable() -> None:
    """A shim answering the 404 it gives an unknown Session is a pod holding some other
    Session -- which is a Turn that did not run, not a Turn to retry silently."""
    refusing = ShimThatStreams([], status=404)

    with pytest.raises(TurnUndeliverable, match="answered 404"):
        await _dispatch_over(refusing, InMemoryLog(), Notified()).dispatch(
            new_session_id(), new_turn_id(), _PROMPT
        )


async def test_the_dispatch_presents_the_token_this_session_derives() -> None:
    """The header the shim's only check reads, measured off the request that was sent
    rather than asserted about the code that built it."""
    session_id, turn_id = new_session_id(), new_turn_id()
    shim = ShimThatStreams(_a_whole_turn(turn_id))

    await _dispatch_over(shim, InMemoryLog(), Notified()).dispatch(
        session_id, turn_id, _PROMPT
    )

    sent = shim.requests[0]
    assert sent.headers["authorization"] == f"Bearer {shim_token_for(session_id, _KEY)}"
    assert str(sent.url) == shim_url_for(f"map-session-{session_id}", _NAMESPACE)


def test_the_append_retry_has_one_definition_the_control_plane_uses() -> None:
    """One rule, one home, and every writer goes through it.

    Three modules append to a Session's Event Log without coordinating: the Turn runner
    in the pod, the Tool Gateway writing progress about a tool call that same Turn is
    still appending to, and the control plane recording what a pod streamed out. Each
    once carried its own copy of the retry with its own attempt count, which drifts
    silently -- both sides keep passing their own tests while one gives up sooner.

    Asserting the callers as well as the definition is what makes this catch the
    regression that actually happened: two copies existed and each side's suite was
    green, because neither module imported the other and nothing compared them.
    """
    sources = {
        module.name: module.read_text()
        for module in _SRC.rglob("*.py")
        if module.name != "__init__.py"
    }
    retriers = sorted(
        name for name, text in sources.items() if "except SequenceRace" in text
    )

    assert retriers == ["event_append.py"], (
        f"the lost-sequence retry is defined in {retriers}; it has one home and every "
        "other module must call `append_in_order` rather than write the loop again"
    )
    for caller in ("turn_runner.py", "pod_channel.py", "mcp_proxy.py"):
        assert "append_in_order" in sources[caller], (
            f"{caller} appends to the Event Log and does not call `append_in_order`, "
            "so it either lost the retry or grew a second copy under another name"
        )


async def test_a_ship_out_that_met_a_delivered_path_keeps_its_own_type() -> None:
    """The one completion-seam failure that does not become `TurnUndeliverable`.

    Everything this method awaits after the stream is collapsed into one type, and the
    reason is in `_record`'s docstring: the route catches exactly one thing, so an
    exception it does not know reaches the tenant as a bare 500 with no published code
    and no `turn.failed`. That collapse is right for every failure whose next move is
    the same, and wrong for the one whose next move differs -- an artifact path the
    agent rewrote is the tenant's to change, and 502 tells them to retry instead.

    So the translation is by type and not by message. Matching on the text would make
    the tenant's status depend on a string an operator is free to reword.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    log = InMemoryLog()
    collided = SeamThatRaises(
        OutputNotRevisable("report.md", "already delivered under that path")
    )

    with pytest.raises(TurnOutputNotRevisable) as refused:
        await _dispatch_over(
            ShimThatStreams(
                [_event(turn.TURN_COMPLETED, turn_id=str(turn_id)), _COMPLETED_LINE]
            ),
            log,
            collided,
        ).dispatch(session_id, turn_id, _PROMPT)

    assert refused.value.path == "report.md"
    assert collided.told == [(session_id, turn_id)]


async def test_every_other_way_ship_out_fails_still_becomes_undeliverable() -> None:
    """The other half, so the translation above cannot be a blanket re-raise.

    A length that did not match, a name the pod promised not to send, a store that
    refused the write: all of them are `OutputsNotShippable` too, none of them is
    something a tenant can act on, and all of them must keep arriving at the route as
    the one type it catches.
    """
    session_id, turn_id = new_session_id(), new_turn_id()
    log = InMemoryLog()
    truncated = SeamThatRaises(OutputsNotShippable("served fewer bytes than it listed"))

    with pytest.raises(TurnUndeliverable):
        await _dispatch_over(
            ShimThatStreams(
                [_event(turn.TURN_COMPLETED, turn_id=str(turn_id)), _COMPLETED_LINE]
            ),
            log,
            truncated,
        ).dispatch(session_id, turn_id, _PROMPT)
