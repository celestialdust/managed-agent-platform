"""MAP-A110: three calls take an engineer from nothing to an answer.

Tier 1 (testcontainers, real PostgreSQL 17). The real app, the real adapters, one real
database, and one scripted Session-shim.

What is real here and what is not is worth being exact about, because the scenario is a
promise about the whole path. Real: the definition registry, the Session registry, the
Event Log and its sequence, every route, the admission decision, the pod lookup, the
dispatch the serving process is actually built with (`HttpPodDispatch`), and the loop
that reads the pod's stream and appends what it carried. Scripted: the HTTP responses of
the Session-shim, one layer below. The dispatch's `transport` is pointed at a handler
that answers the shim's Turn route with the newline-delimited lines a shim streams, in
the shape `session_shim/serve.py` publishes -- and that shape is closed at both ends, so
a fake that drifted from the wire fails to parse here rather than passing.

What that leaves unproven is named rather than implied: that a shim inside a real pod
produces those bytes, having driven a real Agent Runtime over a unix socket to get them.
The first half of that is graded against the real shim app in `tests/session_shim/
test_shim_serves_a_turn.py`; the second is graded against a real pod on a real cluster
by the live tier under `tests/pod/`. Everything between the engineer's three calls and
the events in the log is exercised here.

The scenario's Given has grown one item since it was signed. A Session now names a
registered sandbox shape (`environment_id`), so a shape is registered here in setup
alongside the tool servers the Given already lists -- it is infrastructure a platform
team registers once, not a call the engineer makes per agent, and the count below
asserts exactly that by counting only what the engineer's own client sent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import Placement, PodPhase
from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, TurnId
from managed_agent.core.vocabulary import turn
from managed_agent.session_shim.pod_channel import HttpPodDispatch

_PIECES = ("cohort size was reported ", "in four ", "of the seven papers")
_ANSWER = "".join(_PIECES)
_SHA = "0" * 39 + "a"
_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
_NAMESPACE = "map-sessions"
_TOKEN_KEY = b"the key this deployment signs shim tokens with"
_THREAD = "9f2a6c1e-0f4d-5b3a-8c17-2d5e7f10ab34"
"""The thread id the shim attributes every event to, and it is the *platform's* own.

A uuid rather than a readable name, because a uuid is the only shape this field ever
carries on the wire: a shim publishes a `uuid5` it derived from the Agent Runtime's own
thread string and never that string itself, which is how ADR-007 gets attribution and
MAP-A10's no-runtime-identifier rule at the same time. Nothing here checks the
derivation -- it happens in the pod, and the guard on it is
`test_no_appended_payload_carries_a_runtime_identifier` in
`tests/session_shim/test_turn_runner.py`.
"""


class RunningPods:
    """A cluster in which the Session's pod is up.

    Starts nothing and removes nothing: a Turn is dispatched onto a pod that is
    already placed, and a dispatch that placed one would be doing another
    component's job.
    """

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        raise AssertionError("submitting a Turn tried to start a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.RUNNING

    async def remove(self, pod_name: str) -> None:
        """A no-op, where this used to refuse.

        Under ADR-041 a pod is leased for one Turn, so every dispatch releases one and
        the refusal written here asserted the opposite of the contract. Nothing is
        recorded because no case in this file grades which pod went -- the lease itself
        is graded in `tests/control/test_a_pod_is_leased_for_one_turn.py`, against a
        cluster whose phase actually reflects the removal.
        """


class NeverPlaces:
    """The `SessionPods` seam, refusing every call.

    A Turn places its own pod only when the cluster reports ABSENT or GONE, and the
    cluster above reports RUNNING for every pod. Refusing rather than recording makes
    that a property this file asserts instead of one it merely happens to have: the
    scenario is about three calls reaching an answer, and a placement quietly happening
    underneath would mean the scripted shim was answering for a pod nobody in this file
    decided existed. What a first Turn does about each phase is graded on its own in
    `tests/control/test_a_first_turn_places_the_session_pod.py`.
    """

    async def ensure_for(self, session_id: SessionId) -> None:
        raise AssertionError("submitting a Turn tried to place a pod")


class ScriptedShim:
    """The Session-shim's side of the Turn route: one Turn answered in three deltas.

    The lines are the shim's own shape rather than this test's invention --
    `session_shim/serve.py` publishes `TurnEventLine` and `TurnCompletedLine`, both
    closed to unknown fields and to event types outside the four a Turn produces, and
    the control plane validates every line against them. So a fake that drifted from
    the wire fails to parse rather than passing.

    The Turn id is read back out of the request rather than written as a literal beside
    it. The platform mints that id at admission and a shim echoes it into every payload
    it publishes; a literal here would let the two silently agree with each other and
    disagree with the platform, which is the one thing the case below cannot check for
    itself when it asserts the answer is filed under the Turn the caller was handed.

    The completion line is separate from the `turn.completed` event, and both are sent.
    They are different facts to the control plane: the event is what a tenant reads, and
    the line is what tells the dispatch the Turn *ended* rather than the stream stopping
    -- and only the second runs the ship-out seam.
    """

    def __init__(self) -> None:
        self.turns: list[dict[str, object]] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._answer)

    def _answer(self, request: httpx.Request) -> httpx.Response:
        asked = json.loads(request.read())
        self.turns.append(asked)
        return httpx.Response(200, content=self._stream(str(asked["turn_id"])))

    @staticmethod
    def _stream(turn_id: str) -> bytes:
        def event(type_: str, **fields: object) -> dict[str, object]:
            return {
                "kind": "event",
                "type": type_,
                "payload": {"turn_id": turn_id, "thread_id": _THREAD, **fields},
            }

        lines: list[dict[str, object]] = [event(turn.TURN_STARTED)]
        lines += [event(turn.TURN_MESSAGE_DELTA, text=piece) for piece in _PIECES]
        lines.append(event(turn.TURN_COMPLETED, text=_ANSWER))
        lines.append({"kind": "completed"})
        return "".join(json.dumps(line) + "\n" for line in lines).encode()


class NothingToShip:
    """MAP-13 owns what happens at a completed Turn; here it is only counted."""

    def __init__(self) -> None:
        self.told: list[tuple[SessionId, TurnId]] = []

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        self.told.append((session_id, turn_id))


class CountingTransport(httpx.ASGITransport):
    """An ASGI transport that remembers every request the engineer's client sent."""

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app=app)
        self.calls: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(f"{request.method} {request.url.path}")
        return await super().handle_async_request(request)


@pytest.fixture
async def wired(database_url: str) -> AsyncIterator[tuple[Platform, ScriptedShim]]:
    """The real platform, with the pod's own HTTP responses scripted one layer down.

    `build` already wired a dispatch, and it is replaced rather than reached into
    because the process that runs this test is handed no `PodRunner` and so gets
    `NoPodTransport` -- which refuses every Turn. What goes in its place is the same
    `HttpPodDispatch` a placing deployment is built with, differing from it in the one
    argument that exists to be different: `transport`, which points the Turn's HTTP
    call at the scripted shim above instead of at a pod's DNS name.

    The engine is disposed here rather than left to garbage collection: it owns a pool
    of 50 connections against one `max_connections`, and a leaked one fails a later
    test rather than this one.
    """
    platform, engine = build(database_url)
    shim = ScriptedShim()
    try:
        yield (
            replace(
                platform,
                turn_dispatch=HttpPodDispatch(
                    placement=Placement(RunningPods()),
                    pods=NeverPlaces(),
                    log=platform.event_log_append,
                    on_completed=NothingToShip(),
                    namespace=_NAMESPACE,
                    token_key=_TOKEN_KEY,
                    transport=shim.transport,
                ),
            ),
            shim,
        )
    finally:
        await engine.dispose()


def _definition_body() -> dict[str, object]:
    return {
        "name": "slr-reviewer",
        "instructions": "Extract findings and name the source document for each.",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SHA,
    }


def _create_body(definition_id: str, environment_id: str) -> dict[str, object]:
    return {
        "definition_id": definition_id,
        "environment_id": environment_id,
        "grant": [],
        "scope": {"repository": "acme/widgets"},
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 30,
    }


async def test_registering_opening_and_sending_reaches_a_domain_answer(
    wired: tuple[Platform, ScriptedShim],
) -> None:
    """MAP-A110, end to end, with the engineer's calls counted rather than trusted."""
    platform, shim = wired
    app = create_app(platform)
    tenant = str(uuid4())
    headers = {TENANT_HEADER: tenant}

    # Given: the sandbox shape a Session runs in, registered once by whoever runs the
    # platform. Sent on a client of its own so it cannot be mistaken for one of the
    # engineer's three calls.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://platform",
        headers=headers,
    ) as operator:
        registered = await operator.post(
            "/v1/environments",
            json={"name": "slr-sandbox", "runtime_image": _IMAGE, "denied_paths": []},
        )
        assert registered.status_code == 201, registered.text
        environment_id = str(registered.json()["id"])

    transport = CountingTransport(app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://platform", headers=headers
    ) as engineer:
        # 1. The agent definition.
        defined = await engineer.post("/v1/agents", json=_definition_body())
        assert defined.status_code == 201, defined.text
        definition_id = defined.json()["id"]

        # 2. A Session against it.
        created = await engineer.post(
            "/v1/sessions", json=_create_body(definition_id, environment_id)
        )
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]

        # 3. One Turn.
        answered = await engineer.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": "How many papers reported a cohort size?"},
            headers={"Idempotency-Key": "engineer-first-turn"},
        )
        assert answered.status_code == 202, answered.text
        turn_id = answered.json()["turn_id"]

        reached_in = list(transport.calls)

        # The 202 above means the Turn was accepted, not that it finished -- it runs on
        # a task the request never awaited. So the answer is waited for here, which is
        # what a tenant does too: they hold the event stream open. Awaiting the tasks
        # directly rather than polling because this test drives the app on its own
        # event loop, so the Turn is running on the same loop as this line.
        await asyncio.gather(*platform.background_turns.in_flight)

        # Reading the answer back is an observation, not a step towards it. Counted
        # after the fact so the three calls above stand on their own.
        page = await engineer.get(f"/v1/sessions/{session_id}/events?from_seq=1")
        assert page.status_code == 200, page.text

    assert reached_in == [
        "POST /v1/agents",
        "POST /v1/sessions",
        f"POST /v1/sessions/{session_id}/events",
    ], (
        "the engineer needed something other than the three calls the scenario names "
        f"to reach an answer: {reached_in}"
    )

    events = page.json()["events"]
    assert [event["type"] for event in events] == [
        "session.created",
        turn.TURN_SUBMITTED,
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert [event["seq"] for event in events] == [1, 2, 3, 4, 5, 6, 7]
    completed = events[-1]
    assert completed["payload"]["text"] == _ANSWER, (
        "the Turn did not come back with the domain answer the runtime produced"
    )
    assert completed["payload"]["turn_id"] == turn_id, (
        "the answer is filed under a different Turn than the one the caller was given"
    )
    assert shim.turns[0]["prompt"] == "How many papers reported a cohort size?", (
        "the engineer's own words did not reach the pod the Turn was carried to"
    )
    assert shim.turns[0]["turn_id"] == turn_id, (
        "the pod was asked to run a Turn under an id other than the one the caller was "
        "given, so nothing the pod streams back can be filed against their submission"
    )


async def test_a_retried_submission_does_not_run_the_turn_a_second_time(
    wired: tuple[Platform, ScriptedShim],
) -> None:
    """MAP-A9's idempotency half, over the real sequence rather than a fake one.

    The in-memory cases prove the algorithm; this proves it against the store that
    actually assigns the sequence, because the whole decision is "which submission got
    the lower number" and no fake can settle that question the way Postgres does.
    """
    platform, _ = wired
    app = create_app(platform)
    headers = {TENANT_HEADER: str(uuid4())}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://platform",
        headers=headers,
    ) as caller:
        registered = await caller.post(
            "/v1/environments",
            json={"name": "slr-sandbox", "runtime_image": _IMAGE, "denied_paths": []},
        )
        defined = await caller.post("/v1/agents", json=_definition_body())
        created = await caller.post(
            "/v1/sessions",
            json=_create_body(defined.json()["id"], str(registered.json()["id"])),
        )
        session_id = created.json()["id"]
        prompt = {"prompt": "How many papers reported a cohort size?"}
        key = {"Idempotency-Key": "engineer-first-turn"}
        path = f"/v1/sessions/{session_id}/events"
        first = await caller.post(path, json=prompt, headers=key)
        second = await caller.post(path, json=prompt, headers=key)

    assert first.status_code == 202
    assert second.status_code == 200, second.text
    assert second.json() == first.json()

    # Both submissions are answered before the Turn runs, so "the retry ran nothing" is
    # a claim about tasks that are still in flight at this line. Awaited, not slept on:
    # exactly one Turn must reach completion, and a count taken early would read zero
    # for the right answer and one for the wrong one at the wrong moment.
    await asyncio.gather(*platform.background_turns.in_flight)

    written = await platform.event_log_range.read(
        SessionId(UUID(str(created.json()["id"]))), FIRST_SEQ, Seq(100), limit=100
    )
    assert [row.type for row in written].count(turn.TURN_SUBMITTED) == 1, (
        "the retry recorded a second submission, so the log says two Turns were asked "
        "for and one of them was never run"
    )
    assert [row.type for row in written].count(turn.TURN_COMPLETED) == 1
