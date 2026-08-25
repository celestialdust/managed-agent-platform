"""One method per Repertoire entry, and no method that takes a method name.

These exercise the seven calls the handshake does not, against the stand-in runtime:
what goes on the wire, what comes back, and what a refusal does. The last case is the
shape rule — a public surface where every parameter is a declared params model has no
way to name a call the Repertoire omits (ADR-002).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fake_agent_runtime import fake_agent_runtime

from managed_agent.core.pod.repertoire import (
    REPERTOIRE,
    RepertoireMethod,
    TextInput,
    ThreadGoalSetRequest,
    ThreadReadRequest,
    ThreadResumeRequest,
    ThreadStartRequest,
    TurnInterruptRequest,
    TurnStartRequest,
    TurnSteerRequest,
)
from managed_agent.session_shim.client import RuntimeCallFailed, RuntimeConnection


def _a_thread_start() -> ThreadStartRequest:
    return ThreadStartRequest(
        cwd="/session/workspace",
        model="gpt-5",
        model_provider="map-model-gateway",
        permissions="map-session",
    )


async def test_a_thread_is_started_and_read_back_by_the_id_the_runtime_gave() -> None:
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            thread_id = await connection.start_thread(_a_thread_start())
            assert thread_id == "th_fake_1"
            read = await connection.read_thread(
                ThreadReadRequest(thread_id=thread_id, include_turns=True)
            )
            assert read.thread.id == thread_id
        finally:
            await connection.close()
        start = _frame(runtime.received, "thread/start")
        assert start["params"]["permissions"] == "map-session"
        assert "sandbox" not in start["params"]
        assert _frame(runtime.received, "thread/read")["params"] == {
            "threadId": "th_fake_1",
            "includeTurns": True,
        }


async def test_a_resumed_thread_sends_both_the_file_and_the_id() -> None:
    """The Agent Runtime prefers a non-empty path; the file is the recovered state."""
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            resumed = await connection.resume_thread(
                ThreadResumeRequest(
                    thread_id="th_old",
                    path="/var/lib/map/codex/rollout.jsonl",
                    permissions="map-session",
                )
            )
        finally:
            await connection.close()
    assert resumed == "th_fake_2"
    params = _frame(runtime.received, "thread/resume")["params"]
    assert params["path"] == "/var/lib/map/codex/rollout.jsonl"
    assert params["threadId"] == "th_old"


async def test_a_goal_is_set_and_reads_nothing_back() -> None:
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            await connection.set_goal(
                ThreadGoalSetRequest(
                    thread_id="th_fake_1",
                    objective="reconcile the ledger",
                    status="active",
                    token_budget=50_000,
                )
            )
        finally:
            await connection.close()
    params = _frame(runtime.received, "thread/goal/set")["params"]
    assert params["tokenBudget"] == 50_000
    assert params["status"] == "active"


async def test_a_turn_starts_and_its_events_arrive_in_the_order_they_were_sent() -> (
    None
):
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        events = connection.notifications()
        try:
            turn_id = await connection.start_turn(
                TurnStartRequest(
                    thread_id="th_fake_1", input=(TextInput(text="close the books"),)
                )
            )
            assert turn_id == "tn_fake_1"
            for index in range(3):
                await runtime.push(
                    {
                        "method": "item/agentMessage/delta",
                        "params": {"delta": str(index)},
                    }
                )
            arrived = [await _next(events) for _ in range(3)]
        finally:
            await events.aclose()
            await connection.close()
    assert [event["params"]["delta"] for event in arrived] == ["0", "1", "2"]
    assert _frame(runtime.received, "turn/start")["params"]["input"] == [
        {"type": "text", "text": "close the books"}
    ]


async def test_a_steer_written_for_a_finished_turn_fails_rather_than_landing() -> None:
    """The precondition is the whole point: a stale steer must not redirect the next
    Turn."""
    async with fake_agent_runtime() as runtime:
        runtime.errors["turn/steer"] = {"code": -32001, "message": "turn not active"}
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            with pytest.raises(RuntimeCallFailed) as raised:
                await connection.steer_turn(
                    TurnSteerRequest(
                        thread_id="th_fake_1",
                        expected_turn_id="tn_already_done",
                        input=(TextInput(text="actually, stop"),),
                    )
                )
        finally:
            await connection.close()
    assert raised.value.method is RepertoireMethod.TURN_STEER
    assert _frame(runtime.received, "turn/steer")["params"]["expectedTurnId"] == (
        "tn_already_done"
    )


async def test_an_interrupt_names_the_turn_it_stops() -> None:
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            await connection.interrupt_turn(
                TurnInterruptRequest(thread_id="th_fake_1", turn_id="tn_fake_1")
            )
        finally:
            await connection.close()
    assert _frame(runtime.received, "turn/interrupt")["params"] == {
        "threadId": "th_fake_1",
        "turnId": "tn_fake_1",
    }


def test_no_public_method_accepts_anything_but_a_declared_params_model() -> None:
    """A surface with no string parameter has no way to name an undeclared call.

    Stronger than looking for a parameter called `method`: any `str` parameter on this
    class would be a place a method name could arrive, however it were spelled.
    """
    declared = {entry.params_model for entry in REPERTOIRE.values()}
    taken: set[type[object]] = set()
    for name, member in inspect.getmembers(RuntimeConnection, inspect.isfunction):
        if name.startswith("_"):
            continue
        signature = inspect.signature(member, eval_str=True)
        for parameter in list(signature.parameters.values())[1:]:
            assert parameter.annotation in declared, (
                f"{name}({parameter.name}: {parameter.annotation}) is not a "
                "declared params model"
            )
            taken.add(parameter.annotation)
    handshake = {
        REPERTOIRE[RepertoireMethod.INITIALIZE].params_model,
        REPERTOIRE[RepertoireMethod.INITIALIZED].params_model,
    }
    assert taken == declared - handshake, (
        "the public surface should take every params model but the two connect() "
        "builds itself; an entry with no method here is a call nothing can make"
    )


def _frame(received: list[dict[str, Any]], method: str) -> dict[str, Any]:
    matched = [frame for frame in received if frame.get("method") == method]
    assert len(matched) == 1, f"{method} was sent {len(matched)} times, expected once"
    return matched[0]


async def _next(events: AsyncIterator[dict[str, Any]]) -> dict[str, Any]:
    return await asyncio.wait_for(anext(events), timeout=5.0)
