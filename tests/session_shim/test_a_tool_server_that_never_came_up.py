"""A tool server that never started is announced, and a healthy one stays quiet.

Tier 1 (local, no infrastructure). The frames are scripted in the Agent Runtime's own
shape, copied from the wire test in its protocol source: a `mcpServer/startupStatus/
updated` notification carries `name`, `status` and an optional `error` beside each
other, with `threadId` nullable. A fake built from a guess would certify the guess, and
the guess that matters here is the one about where the failure text lives -- read from
the wrong key, every announcement would name a server and say nothing about why.

**Why this file exists at all.** A Session whose definition named a tool server ran a
live Turn in which the agent reported the tool was absent and asked how to proceed; the
Turn completed, the Tool Gateway logged no request from that pod, and the platform's
record was indistinguishable from a Turn that worked. The runtime had announced the
failure on this exact method. These cases are the listening.

Order is asserted where it matters. The announcement is appended in arrival order like
every other event, so a reader sees the server go down before the Turn that ran without
it -- which is the ordering that makes the log explain itself.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_turn_runner import (
    _THREAD,
    Notified,
    RecordingLog,
    Scripted,
    _completed,
    _delta,
    _started,
)

from managed_agent.core.ids import new_session_id, new_turn_id
from managed_agent.core.vocabulary import tool_server, turn
from managed_agent.session_shim.turn_runner import run_turn

_SERVER = "map_gateway"
_WHY = "MCP client for `map_gateway` timed out after 30 seconds."
_DOWN = tool_server.TOOL_SERVER_UNAVAILABLE


def _announcements(written: RecordingLog) -> list[Any]:
    """Every announcement this run appended, so a case asserts on shape not on count."""
    return [one for one in written.written if one.type == _DOWN]


def _startup(
    state: str, *, error: str | None = _WHY, name: str = _SERVER
) -> dict[str, Any]:
    """One startup transition in the runtime's shape.

    `error` defaults to text because the case that matters carries text; `None` omits
    the key entirely rather than writing an empty string, which is what a runtime
    reporting a state with nothing to explain actually sends.
    """
    frame: dict[str, Any] = {"threadId": None, "name": name, "status": state}
    if error is not None:
        frame["error"] = error
    return {"method": "mcpServer/startupStatus/updated", "params": frame}


async def _run(frames: list[dict[str, Any]]) -> RecordingLog:
    written = RecordingLog()
    await run_turn(
        new_session_id(),
        new_turn_id(),
        _THREAD,
        "do the errand",
        Scripted(frames),
        written,
        Notified(),
    )
    return written


@pytest.mark.anyio
async def test_a_server_that_failed_to_start_is_announced_with_the_reason() -> None:
    """The whole point: the tenant's grant went unhonoured and the log says so.

    The reason crosses as the runtime wrote it. Without it the event says a server is
    down and leaves the reader to guess between a timeout, a refused address and a
    rejected token -- three different people's problems, distinguishable only here.
    """
    written = await _run([_started(), _startup("failed"), _completed()])

    announced = _announcements(written)
    assert len(announced) == 1, written.types()
    assert announced[0].payload["server"] == _SERVER
    assert announced[0].payload["state"] == "failed"
    assert announced[0].payload["error"] == _WHY


@pytest.mark.anyio
async def test_it_lands_before_the_turn_that_ran_without_the_tool_completed() -> None:
    """Arrival order, because a reader uses it to explain the Turn below it.

    Asserted as positions in one list rather than as two separate presence checks: an
    announcement appended after the completion would still be in the log and would
    still read, to anyone scanning it, as something that happened afterwards.
    """
    frames = [_started(), _startup("failed"), _delta("no tool"), _completed()]
    types = (await _run(frames)).types()

    assert types.index(_DOWN) < types.index(turn.TURN_COMPLETED)


@pytest.mark.anyio
@pytest.mark.parametrize("state", ["starting", "ready"])
async def test_a_server_coming_up_or_already_up_announces_nothing(state: str) -> None:
    """The healthy path writes no row, and that is a decision rather than an oversight.

    The runtime reports every transition on this one method. Publishing them all would
    add a row per server per placement saying the expected thing happened, and the
    evidence a server is up is already in the log as the call that reached it.
    """
    written = await _run([_started(), _startup(state, error=None), _completed()])

    assert _DOWN not in written.types()


@pytest.mark.anyio
async def test_a_cancelled_startup_is_announced_too_and_says_which_it_was() -> None:
    """Cancelled leaves the Session without the tool exactly as failed does.

    One type rather than two, because a reader acting on either does the same thing;
    `state` keeps them apart for a reader diagnosing rather than reacting, since a
    startup abandoned while the pod was going away is not the same incident as one that
    could not reach its address.
    """
    written = await _run([_started(), _startup("cancelled", error=None), _completed()])

    announced = _announcements(written)
    assert len(announced) == 1, written.types()
    assert announced[0].payload["state"] == "cancelled"
    assert "error" not in announced[0].payload


@pytest.mark.anyio
async def test_a_frame_naming_no_server_announces_nothing() -> None:
    """A row naming no server cannot be acted on, so the absence is the honest record.

    An announcement with an empty `server` is worse than none: it tells a reader some
    server is down and gives them nothing to look at, and it would pass any assertion
    that merely counts announcements.
    """
    written = await _run([_started(), _startup("failed", name=""), _completed()])

    assert _DOWN not in written.types()
