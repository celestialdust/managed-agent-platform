"""The connection itself: the unix socket, the handshake, and what closing does.

The server on the other end is a stand-in (see `fake_agent_runtime`), but the transport
is the real one — a WebSocket handshake over a unix socket, one JSON message per text
frame. A shim written against a raw JSON socket passes none of these.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from fake_agent_runtime import fake_agent_runtime

from managed_agent.control.pod_config.compiler import (
    CONTROL_SOCKET,
    compile_session_config,
)
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.pod.repertoire import RepertoireMethod, ThreadReadRequest
from managed_agent.core.registration.definition import (
    AgentDefinition,
    SkillsRevision,
)
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord
from managed_agent.session_shim.client import (
    CONTROL_SOCKET_PATH,
    RuntimeCallFailed,
    RuntimeConnection,
    RuntimeConnectionClosed,
)

# Where a Session pod reaches the Model Gateway. The `/v1` is load-bearing at both ends:
# the Agent Runtime POSTs `{base_url}/responses`, and the Gateway's router mounts
# `POST /v1/responses`.
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"

# The Gateway's signing key and the token's deadline, which the compiler takes from
# its caller and never defaults. Literals, so no case here can expire mid-run.
SESSION_TOKEN_KEY = b"a signing key that is thirty-two"
SESSION_TOKEN_EXPIRY = 4102444800


# The definition a Session pins, for the one field the compiler reads off it: the model.
# The provider is not here because it is not the definition's to name -- every model
# call leaves a Session pod through the Model Gateway.
A_DEFINITION = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)


def test_the_shim_reaches_the_socket_the_pod_actually_listens_on() -> None:
    """One source of truth, checked against the argv the pod is started with.

    The path is also the sandbox deny target, so a shim connecting somewhere else would
    be reachable by the confined agent as well as by the shim, and neither side would
    report anything wrong.
    """
    assert Path(CONTROL_SOCKET) == CONTROL_SOCKET_PATH
    compiled = compile_session_config(
        SessionRecord(
            id=new_session_id(),
            tenant_id=TenantId(uuid4()),
            definition_id=new_definition_id(),
            definition_revision="rev-1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=10_000,
            budget_currency="USD",
            retention_days=30,
        ),
        tool_gateway_url="http://tool-gateway.invalid",
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=_a_shape_that_narrows_nothing(),
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )
    argv = compiled.launch_argv
    assert argv[argv.index("--listen") + 1] == f"unix://{CONTROL_SOCKET_PATH}"


async def test_connect_completes_the_handshake_before_reporting_ready() -> None:
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        response = await connection.connect()
        try:
            assert response.user_agent == "fake-agent-runtime/1"
            await runtime.wait_until(
                lambda: len(runtime.received) >= 2, "the handshake's two frames"
            )
            assert runtime.methods_received[:2] == ["initialize", "initialized"]
            handshake = runtime.received[0]
            assert handshake["params"]["capabilities"] == {"experimentalApi": True}
            assert "id" in handshake
            assert "id" not in runtime.received[1], "a notification carries no id"
        finally:
            await connection.close()


async def test_connect_where_nothing_listens_raises_rather_than_reporting_ready() -> (
    None
):
    directory = Path(tempfile.mkdtemp(prefix="map-"))
    connection = RuntimeConnection(directory / "nobody.sock")
    with pytest.raises(OSError):
        await connection.connect()


async def test_a_call_before_connect_is_refused_and_writes_nothing() -> None:
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        with pytest.raises(RuntimeConnectionClosed):
            await connection.read_thread(
                ThreadReadRequest(thread_id="th_1", include_turns=False)
            )
        assert runtime.received == []


async def test_an_inbound_request_is_answered_not_handled_and_is_no_notification() -> (
    None
):
    """The inbound half is as closed as the outbound one.

    Approvals are off, so the Agent Runtime raises no request of its own; anything that
    does arrive is a capability nobody decided to support. It gets method-not-found, and
    it must not reach the notification stream, where a consumer would take it for an
    event of the Turn.
    """
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        events = connection.notifications()
        try:
            await runtime.push({"id": "srv-1", "method": "thread/approvalRequest"})
            await runtime.push({"method": "turn/started", "params": {}})
            first = await asyncio.wait_for(anext(events), timeout=5.0)
            assert first["method"] == "turn/started"
            await runtime.wait_until(
                lambda: any(f.get("id") == "srv-1" for f in runtime.received),
                "the shim's answer to the inbound request",
            )
            answers = [f for f in runtime.received if f.get("id") == "srv-1"]
            assert answers[0]["error"]["code"] == -32601
            assert "method" not in answers[0], "an answer is not a call of our own"
        finally:
            await events.aclose()
            await connection.close()


async def test_a_refused_call_renders_no_runtime_detail_but_keeps_it_pod_local() -> (
    None
):
    secret = "thread th_internal_9f2 absent from /var/lib/map/codex/rollout.jsonl"
    async with fake_agent_runtime() as runtime:
        runtime.errors["thread/read"] = {"code": -32000, "message": secret}
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            with pytest.raises(RuntimeCallFailed) as raised:
                await connection.read_thread(
                    ThreadReadRequest(thread_id="th_1", include_turns=False)
                )
        finally:
            await connection.close()
    failure = raised.value
    assert str(failure) == "thread/read failed with code -32000"
    assert "th_internal_9f2" not in str(failure)
    assert "th_internal_9f2" not in str(failure.args)
    assert failure.pod_local_detail == secret
    assert failure.method is RepertoireMethod.THREAD_READ


async def test_a_call_in_flight_when_the_socket_goes_fails_instead_of_hanging() -> None:
    """A waiter nobody will answer must not wait for the life of the pod."""
    async with fake_agent_runtime() as runtime:
        runtime.silent.add("thread/read")
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        call = asyncio.create_task(
            connection.read_thread(
                ThreadReadRequest(thread_id="th_1", include_turns=False)
            )
        )
        while "thread/read" not in runtime.methods_received:
            await asyncio.sleep(0.01)
        await connection.close()
        with pytest.raises(RuntimeConnectionClosed):
            await asyncio.wait_for(call, timeout=5.0)


def _a_shape_that_narrows_nothing() -> Environment:
    """A registered shape contributing an image and no extra deny rule.

    Compiling now names an environment, and these cases are about the platform's own
    floors rather than about a tenant's narrowing -- so the shape here adds nothing, and
    the compiled profile is exactly what the platform declares.
    """
    return Environment(
        id=new_environment_id(),
        tenant_id=TenantId(uuid4()),
        name="fixture",
        runtime_image="registry.map.internal/session@sha256:" + "a" * 64,
        denied_paths=(),
    )


async def test_closing_after_the_runtime_vanished_is_not_itself_a_failure() -> None:
    """A connection whose socket died is closable, and closing it raises nothing.

    The reader task ends on the socket dying, and `close` awaits it to be sure it has
    stopped -- so whatever the reader ended with is re-raised there unless the reader
    treats a dead socket as its ordinary terminal condition. It did not, and the shim
    calls `close` from its lifespan shutdown: a Session pod whose runtime exited
    without a closing handshake logged `Application shutdown failed. Exiting.` and was
    torn down hard, which cut the response stream the control plane was still reading
    and turned a Turn that had done its work into `runtime_lost`.

    The in-flight call is how this case knows the reader has already finished rather
    than been cancelled by `close`. Its `RuntimeConnectionClosed` is raised from the
    reader's own `finally`, so seeing it means the reader ran to the end -- without it
    `close` would cancel a reader still parked on `recv` and pass for the wrong reason.
    """
    async with fake_agent_runtime() as runtime:
        runtime.silent.add("thread/read")
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        call = asyncio.create_task(
            connection.read_thread(
                ThreadReadRequest(thread_id="th_1", include_turns=False)
            )
        )
        await runtime.wait_until(
            lambda: "thread/read" in runtime.methods_received,
            "the call never reached the runtime, so the socket died before it mattered",
        )
        await runtime.vanish()
        with pytest.raises(RuntimeConnectionClosed):
            await asyncio.wait_for(call, timeout=5.0)
        await connection.close()
