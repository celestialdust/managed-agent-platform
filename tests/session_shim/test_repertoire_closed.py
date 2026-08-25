"""Invariant I4: the Session-shim issues exactly the Repertoire, and nothing else.

Two halves, and neither is sufficient alone.

The measured half exercises every declared call against a real socket and reads the set
of requests off the frames that arrived. That is what "absent rather than refused" means
made checkable: a call outside the set leaves no frame behind, so the assertion is over
what the server *received* and not over what it answered.

The structural half reads the shim's own source and asserts what is not in it — a run
exercises the calls it happens to make, and can never show that no other call exists
in the module. Each structural case fails on an absence, so a future diff that adds a
method, spells one as a string, reopens a generic passthrough, or reaches an escape
hatch breaks it.

A negative assertion is satisfied most cheaply by the guarded thing being missing
altogether, so the measured half is also the positive one: it fails if the shim issues
nothing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import cast

import pytest
from fake_agent_runtime import fake_agent_runtime

from managed_agent.core.pod.repertoire import (
    REPERTOIRE,
    RepertoireEntry,
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
from managed_agent.session_shim import client as shim_client
from managed_agent.session_shim.client import NotInRepertoire, RuntimeConnection

_SOURCE = Path(shim_client.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)
_METHOD_SHAPED = re.compile(r"^[a-z][A-Za-z0-9]*(/[a-z][A-Za-z0-9_]*)+$")

# The connected client's escape-hatch surface: stable, emitting no approval request, and
# withdrawable by no configuration key. The Repertoire exists so these are unreachable
# from a pod by construction, which means their names must not occur in the shim at all.
_ESCAPE_HATCHES = ("thread/shellCommand", "command/exec", "fuzzyFileSearch", "fs/")


async def test_the_issued_request_set_is_the_repertoire_exactly() -> None:
    """Measured, not asserted: every frame the runtime received, against the contract.

    This is the positive half of the file. Each structural case below fails on an
    absence, and an absence is what a shim that issued nothing at all would also give —
    so the guard needs one case that fails when the calls stop being made.
    """
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            await connection.start_thread(
                ThreadStartRequest(
                    cwd="/session/workspace",
                    model="gpt-5",
                    model_provider="map-model-gateway",
                    permissions="map-session",
                )
            )
            await connection.resume_thread(
                ThreadResumeRequest(
                    thread_id="th_fake_1",
                    path="/var/lib/map/codex/rollout.jsonl",
                    permissions="map-session",
                )
            )
            await connection.read_thread(
                ThreadReadRequest(thread_id="th_fake_1", include_turns=False)
            )
            await connection.set_goal(
                ThreadGoalSetRequest(
                    thread_id="th_fake_1", objective="finish", status="active"
                )
            )
            await connection.start_turn(
                TurnStartRequest(thread_id="th_fake_1", input=(TextInput(text="go"),))
            )
            await connection.steer_turn(
                TurnSteerRequest(
                    thread_id="th_fake_1",
                    expected_turn_id="tn_fake_1",
                    input=(TextInput(text="turn left"),),
                )
            )
            await connection.interrupt_turn(
                TurnInterruptRequest(thread_id="th_fake_1", turn_id="tn_fake_1")
            )
        finally:
            await connection.close()
    issued = set(runtime.methods_received)
    assert issued == {method.value for method in RepertoireMethod}
    assert len(runtime.methods_received) == len(RepertoireMethod)


async def test_an_entry_naming_an_undeclared_method_writes_nothing_at_all() -> None:
    """Absent rather than refused, measured on the socket the runtime is listening on.

    The escape hatch cannot be named through the enum at all, so the attempt has to
    be forged past the type system to be made — and even forged, it is stopped before
    a byte is written. The assertion that matters is about the server's frames:
    nothing arrived for the Agent Runtime to accept or deny.
    """
    declared = REPERTOIRE[RepertoireMethod.THREAD_START]
    forged = RepertoireEntry(
        method=cast(RepertoireMethod, "thread/shellCommand"),
        params_model=declared.params_model,
        response_model=declared.response_model,
        experimental_fields=frozenset(),
    )
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            before = list(runtime.received)
            with pytest.raises(NotInRepertoire):
                await connection._write(
                    forged, declared.params_model.model_construct(), "forged-1"
                )
            assert runtime.received == before
        finally:
            await connection.close()
    assert "thread/shellCommand" not in runtime.methods_received


async def test_an_entry_the_repertoire_did_not_hand_out_is_refused() -> None:
    """Holding a matching method is not enough; it must be the declared entry itself.

    A hand-built entry can carry any params model under a declared method's name, which
    would send a body the Repertoire never described under a method it did.
    """
    declared = REPERTOIRE[RepertoireMethod.THREAD_START]
    forged = RepertoireEntry(
        method=declared.method,
        params_model=declared.params_model,
        response_model=declared.response_model,
        experimental_fields=frozenset(),
    )
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            before = list(runtime.received)
            with pytest.raises(NotInRepertoire):
                await connection._write(
                    forged, declared.params_model.model_construct(), "forged-2"
                )
            assert runtime.received == before
        finally:
            await connection.close()


def test_the_shim_names_the_whole_repertoire_and_nothing_beside_it() -> None:
    named = {
        RepertoireMethod[node.attr]
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "RepertoireMethod"
    }
    assert named == set(RepertoireMethod)


def test_no_method_name_is_spelled_as_a_bare_string() -> None:
    literals = [
        node.value
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert literals, "no string constants found; the source did not parse as expected"
    assert [text for text in literals if _METHOD_SHAPED.match(text)] == []


def test_the_escape_hatch_surface_is_absent_from_the_shim() -> None:
    assert [name for name in _ESCAPE_HATCHES if name in _SOURCE] == []


def test_the_only_writer_takes_an_entry_rather_than_a_name() -> None:
    write = next(
        node
        for node in ast.walk(_TREE)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_write"
    )
    annotation = write.args.args[1].annotation
    assert isinstance(annotation, ast.Name)
    assert annotation.id == "RepertoireEntry"


def test_every_send_goes_through_the_one_writer() -> None:
    """The check in `_write` is worth nothing if another line can reach the socket.

    Two `send` calls are expected and both are accounted for: the one in `_write`, and
    the method-not-found answer the reader gives an inbound request. A third would be an
    outbound path that skips the Repertoire check entirely.
    """
    senders = {
        enclosing.name
        for enclosing in ast.walk(_TREE)
        if isinstance(enclosing, ast.AsyncFunctionDef)
        for node in ast.walk(enclosing)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "send"
    }
    assert senders == {"_write", "_read_forever"}
