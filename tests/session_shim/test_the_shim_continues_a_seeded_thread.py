"""Whether the shim continues this Session's conversation or opens its first.

Tier 1 (local, no infrastructure), over the real JSON-RPC connection to the stand-in
runtime, so what is graded is the FRAME that reaches the runtime rather than a call
recorded against a mock. That matters here more than usual: `thread/start` and
`thread/resume` both come back with a thread id, and a shim that sent the wrong one
would look identical to every caller of `open_the_thread` -- and identical in every
status Kubernetes reports about the pod. The difference is only visible on the wire and
in the tenant's bill.

**What is being protected.** A resume that silently starts a new thread replays history
the Rollout's compaction checkpoints have already folded, charges the tenant for the
replay, and reports success (ADR-004). So the two cases below are not symmetric: the
first-placement case is the ordinary one, and the resume case is the one whose failure
mode is invisible.

The decision rests on a file, and these tests put a real one on disk through the seed's
own writer rather than composing a path. A test that wrote the path itself would keep
passing if the seed and the shim ever stopped spelling it the same way, which is the
one disagreement that puts every resuming Session on a fresh thread.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest
from fake_agent_runtime import fake_agent_runtime

from managed_agent.control.pod_config.compiler import PROFILE_NAME, WORKSPACE_ROOT
from managed_agent.session_shim import serve
from managed_agent.session_shim.client import RuntimeConnection
from managed_agent.session_shim.seed_rollout import write_seed
from managed_agent.session_shim.serve import open_the_thread

THREAD: Final = "0199c2f7-0000-7000-8000-00000000beef"
MODEL: Final = "gpt-5"
PROVIDER: Final = "map-model-gateway"

A_ROLLOUT: Final = (
    json.dumps({"type": "session_meta", "payload": {"id": THREAD}}).encode() + b"\n"
)


def _quiet(report: str) -> None:
    """The seed's own report sink, which these cases do not read."""


@pytest.fixture(autouse=True)
def _the_pods_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two values a first placement sends and a resume deliberately does not."""
    monkeypatch.setenv("MAP_MODEL", MODEL)
    monkeypatch.setenv("MAP_MODEL_PROVIDER", PROVIDER)


def _frame(received: list[dict[str, Any]], method: str) -> dict[str, Any]:
    matched = [frame for frame in received if frame.get("method") == method]
    assert len(matched) == 1, f"{method} was sent {len(matched)} times, expected once"
    return matched[0]


async def _open_against_a_runtime(home: Path) -> tuple[str, list[str]]:
    """Run the shim's one thread-opening decision against the stand-in runtime.

    Returns what it answered and every method it sent, so a case can assert on the call
    that was NOT made as easily as on the one that was.
    """
    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            opened = await open_the_thread(connection)
        finally:
            await connection.close()
        return opened, runtime.methods_received


# --------------------------------------------------------------------------------------
# A first placement
# --------------------------------------------------------------------------------------


async def test_a_pod_with_no_seeded_record_starts_this_session_s_first_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unchanged path, and the one every Session takes once.

    An empty runtime home is what the seed leaves behind on a first placement, so this
    is the state the shim actually meets rather than a state constructed for it.
    """
    monkeypatch.setattr(serve, "RUNTIME_HOME", tmp_path)

    opened, methods = await _open_against_a_runtime(tmp_path)

    assert opened == "th_fake_1"
    assert "thread/start" in methods
    assert "thread/resume" not in methods


async def test_a_first_thread_still_carries_the_model_and_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three fields a new thread cannot be opened without, on the wire.

    Asserted as a whole frame rather than field by field because this is the request a
    resume must NOT send: the next case reads the same params table and finds it
    absent, and the two together are what say the branch changed the call rather than
    just the value inside it.
    """
    monkeypatch.setattr(serve, "RUNTIME_HOME", tmp_path)

    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            await open_the_thread(connection)
        finally:
            await connection.close()
        params = _frame(runtime.received, "thread/start")["params"]

    assert params["cwd"] == WORKSPACE_ROOT
    assert params["model"] == MODEL
    assert params["permissions"] == PROFILE_NAME


# --------------------------------------------------------------------------------------
# A placement that continues a conversation
# --------------------------------------------------------------------------------------


async def test_a_pod_holding_a_seeded_record_continues_that_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap this slice closes, stated as the call that reaches the runtime.

    `thread/start` here would be the silent failure in full: a thread id comes back
    either way, the Turn runs, the tenant is billed for a replay of history the Rollout
    had already folded, and nothing anywhere reports a problem.
    """
    write_seed(
        tmp_path, A_ROLLOUT, datetime(2026, 8, 24, 9, 0, 0, tzinfo=UTC), report=_quiet
    )
    monkeypatch.setattr(serve, "RUNTIME_HOME", tmp_path)

    opened, methods = await _open_against_a_runtime(tmp_path)

    assert opened == "th_fake_2"
    assert "thread/resume" in methods
    assert "thread/start" not in methods


async def test_the_resume_names_the_seeded_file_and_the_thread_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both fields, because the runtime prefers the path and verifies it against the id.

    The path is read back off the disk the seed wrote to rather than composed here: the
    seed's writer and the shim's finder are the two halves of one spelling, and a test
    that wrote its own path would keep passing through exactly the drift that would put
    every resuming Session on a fresh thread.

    Neither the model nor the working directory is sent, and that is the protocol rather
    than an omission -- a resumed thread takes both from the record it is resuming.
    """
    seeded = write_seed(
        tmp_path, A_ROLLOUT, datetime(2026, 8, 24, 9, 0, 0, tzinfo=UTC), report=_quiet
    )
    monkeypatch.setattr(serve, "RUNTIME_HOME", tmp_path)

    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            await open_the_thread(connection)
        finally:
            await connection.close()
        params = _frame(runtime.received, "thread/resume")["params"]

    assert params["path"] == str(seeded)
    assert params["threadId"] == THREAD
    assert params["permissions"] == PROFILE_NAME
    assert "model" not in params
    assert "cwd" not in params


async def test_the_thread_resumed_under_is_the_records_own_and_not_the_filename_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read out of the `session_meta` line, which is what the runtime checks against.

    The two sources agree in every pod this platform builds, so the only way to tell
    which one is being read is to make them disagree. A shim parsing the filename would
    resume under a name the record contradicts, and the runtime would refuse a file it
    could otherwise have continued.
    """
    seeded = write_seed(
        tmp_path, A_ROLLOUT, datetime(2026, 8, 24, 9, 0, 0, tzinfo=UTC), report=_quiet
    )
    renamed = seeded.with_name(seeded.name.replace(THREAD, "not-the-thread-inside"))
    seeded.rename(renamed)
    monkeypatch.setattr(serve, "RUNTIME_HOME", tmp_path)

    async with fake_agent_runtime() as runtime:
        connection = RuntimeConnection(runtime.socket_path)
        await connection.connect()
        try:
            await open_the_thread(connection)
        finally:
            await connection.close()
        params = _frame(runtime.received, "thread/resume")["params"]

    assert params["threadId"] == THREAD
    assert params["path"] == str(renamed)
