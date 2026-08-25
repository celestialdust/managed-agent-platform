"""MAP-A110: three calls take an engineer from nothing to an answer.

Tier 1 (testcontainers, real PostgreSQL 17). The real app, the real adapters, one real
database, and one scripted Agent Runtime.

What is real here and what is not is worth being exact about, because the scenario is a
promise about the whole path. Real: the definition registry, the Session registry, the
Event Log and its sequence, every route, the admission decision, the pod lookup, and the
loop that turns runtime notifications into published events. Not real: the transport
into the Session's pod. Nothing in this tree can open the Agent Runtime's unix socket
inside another pod, so the channel is scripted -- and the scripted frames are written in
the runtime's own shape, taken from the protocol source under `.reference/codex`, since
a fake shaped to match the code certifies the code against itself.

What that leaves unproven is named rather than implied: that a real pod answers these
frames. Everything between the engineer's three calls and the events in the log is
exercised.

The scenario's Given has grown one item since it was signed. A Session now names a
registered sandbox shape (`environment_id`), so a shape is registered here in setup
alongside the tool servers the Given already lists -- it is infrastructure a platform
team registers once, not a call the engineer makes per agent, and the count below
asserts exactly that by counting only what the engineer's own client sent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import Placement, PodBinding, PodPhase
from managed_agent.control.session.turn_dispatch import PodRuntime, PodTurnDispatch
from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, TurnId
from managed_agent.core.pod.repertoire import TurnStartRequest
from managed_agent.core.vocabulary import turn

_ANSWER = "cohort size was reported in four of the seven papers"
_SHA = "0" * 39 + "a"
_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
_THREAD = "thread-inside-the-pod"


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
        raise AssertionError("submitting a Turn tried to remove a pod")


class ScriptedRuntime:
    """One Agent Runtime, answering one Turn in three deltas.

    The frames are the runtime's, not this test's invention: `turn/started` and
    `turn/completed` carry `{threadId, turn}` with the status on `turn`, and an agent
    message delta carries its text at the top of the params beside three identifiers
    that must not cross into a published event.
    """

    def __init__(self) -> None:
        self.started: list[TurnStartRequest] = []

    async def start_turn(self, request: TurnStartRequest) -> str:
        self.started.append(request)
        return "runtime-turn-1"

    async def notifications(self) -> AsyncIterator[dict[str, Any]]:
        yield {
            "method": "turn/started",
            "params": {"threadId": _THREAD, "turn": {"id": "runtime-turn-1"}},
        }
        for piece in ("cohort size was reported ", "in four ", "of the seven papers"):
            yield {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": _THREAD,
                    "turnId": "runtime-turn-1",
                    "itemId": "item-1",
                    "delta": piece,
                },
            }
        yield {
            "method": "turn/completed",
            "params": {
                "threadId": _THREAD,
                "turn": {"id": "runtime-turn-1", "status": "completed"},
            },
        }


class ScriptedChannel:
    """Stands in for the transport into the pod, which does not exist."""

    def __init__(self, runtime: ScriptedRuntime) -> None:
        self.runtime = runtime

    async def open(self, binding: PodBinding) -> PodRuntime:
        return PodRuntime(connection=self.runtime, thread_id=_THREAD)


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
async def wired(database_url: str) -> AsyncIterator[tuple[Platform, ScriptedRuntime]]:
    """The real platform with the one seam that has no implementation scripted.

    The engine is disposed here rather than left to garbage collection: it owns a pool
    of 50 connections against one `max_connections`, and a leaked one fails a later
    test rather than this one.
    """
    platform, engine = build(database_url)
    runtime = ScriptedRuntime()
    try:
        yield (
            replace(
                platform,
                turn_dispatch=PodTurnDispatch(
                    Placement(RunningPods()),
                    ScriptedChannel(runtime),
                    platform.event_log_append,
                    NothingToShip(),
                ),
            ),
            runtime,
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
        "grant": ["fs.read"],
        "scope": {"repository": "acme/widgets"},
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 30,
    }


async def test_registering_opening_and_sending_reaches_a_domain_answer(
    wired: tuple[Platform, ScriptedRuntime],
) -> None:
    """MAP-A110, end to end, with the engineer's calls counted rather than trusted."""
    platform, runtime = wired
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
    assert runtime.started[0].input[0].text == (
        "How many papers reported a cohort size?"
    )


async def test_the_engineer_reads_no_runtime_identifier_anywhere(
    wired: tuple[Platform, ScriptedRuntime],
) -> None:
    """MAP-A10 over the one path that touches the runtime.

    Every frame the runtime sent carried its thread id, its own turn id and an item id.
    None of the three may appear in anything the tenant can read -- and this is the
    only test in the suite where a real Turn's events have been through the mapping
    with real runtime identifiers present to leak.
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
        environment_id = str(registered.json()["id"])
        defined = await caller.post("/v1/agents", json=_definition_body())
        created = await caller.post(
            "/v1/sessions",
            json=_create_body(defined.json()["id"], environment_id),
        )
        session_id = created.json()["id"]
        await caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": "How many papers reported a cohort size?"},
            headers={"Idempotency-Key": "engineer-first-turn"},
        )
        page = await caller.get(f"/v1/sessions/{session_id}/events?from_seq=1")

    body = page.text
    assert _THREAD not in body
    assert "runtime-turn-1" not in body
    assert "item-1" not in body


async def test_a_retried_submission_does_not_run_the_turn_a_second_time(
    wired: tuple[Platform, ScriptedRuntime],
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

    written = await platform.event_log_range.read(
        SessionId(UUID(str(created.json()["id"]))), FIRST_SEQ, Seq(100), limit=100
    )
    assert [row.type for row in written].count(turn.TURN_SUBMITTED) == 1, (
        "the retry recorded a second submission, so the log says two Turns were asked "
        "for and one of them was never run"
    )
    assert [row.type for row in written].count(turn.TURN_COMPLETED) == 1
