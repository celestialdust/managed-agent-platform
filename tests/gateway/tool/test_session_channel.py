"""Progress out, a question out, and the answer back — all over one Event Log.

Tier 1 (local, no infrastructure). The log is in memory here, but it is a log with the
one property that matters to this module: it assigns the sequence itself and refuses a
writer that lost the race for the next one, which is the ordinary case rather than an
edge because the Session's own Turn runner is appending to the same log while a tool
call is open.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
from mcp.types import ElicitRequestFormParams, ElicitRequestURLParams

from managed_agent.core.ids import Seq, SessionId
from managed_agent.core.ports import SequenceRace
from managed_agent.gateway.tool.mcp_proxy import EventLogSessionChannel, ToolEventTypes

TYPES = ToolEventTypes(
    progress="tool.progress",
    elicitation_requested="tool.elicitation_requested",
    elicitation_answered="tool.elicitation_answered",
)


@dataclass(frozen=True, slots=True)
class Record:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


class FakeLog:
    """An in-memory Event Log that numbers its own events and can lose a race.

    `races` is how many appends are refused before one is accepted, which is what makes
    the retry observable: a channel that calls `append` bare fails the tool call it was
    reporting progress for, and the store's contract says the caller retries.
    """

    def __init__(self, races: int = 0) -> None:
        self.records: list[Record] = []
        self.attempts = 0
        self._races_left = races
        self._appended = asyncio.Event()

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.attempts += 1
        if self._races_left > 0:
            self._races_left -= 1
            raise SequenceRace("another writer took the next sequence")
        seq = len(self.records) + 1
        self.records.append(
            Record(session_id=session_id, seq=seq, type=type_, payload=payload)
        )
        self._appended.set()
        return seq

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Record]:
        seen = after
        while True:
            fresh = [r for r in self.records if r.seq > seen]
            for record in fresh:
                seen = record.seq
                yield record
            self._appended.clear()
            await self._appended.wait()

    def of_type(self, type_: str) -> list[Record]:
        return [r for r in self.records if r.type == type_]


def _channel(log: FakeLog, timeout_s: float = 5.0) -> EventLogSessionChannel:
    return EventLogSessionChannel(
        session_id=SessionId(uuid4()),
        append=log,
        events=log,  # type: ignore[arg-type]
        types=TYPES,
        timeout_s=timeout_s,
    )


def _form() -> ElicitRequestFormParams:
    return ElicitRequestFormParams(
        message="which invoice?",
        requested_schema={"type": "object", "properties": {"invoice": {}}},
    )


async def test_progress_appends_one_event_of_the_type_it_was_handed() -> None:
    log = FakeLog()

    await _channel(log).progress("call-1", 2.0, 3.0, "halfway")

    assert [r.type for r in log.records] == ["tool.progress"]
    assert log.records[0].payload == {
        "call_id": "call-1",
        "progress": 2.0,
        "total": 3.0,
        "message": "halfway",
    }


async def test_a_lost_sequence_race_is_retried_rather_than_failing_the_tool_call() -> (
    None
):
    """The store raises `SequenceRace` for the caller to ask again, and this asks.

    Unretried, the exception escapes the SDK's progress callback into `call_tool`'s
    broad handler and is classified `platform.internal` — so an ordinary concurrent
    write by the Session's own Turn runner would fail the whole tool call.
    """
    log = FakeLog(races=3)

    await _channel(log).progress("call-1", 1.0, None, None)

    assert log.attempts == 4
    assert len(log.of_type(TYPES.progress)) == 1


async def test_losing_every_attempt_is_raised_rather_than_swallowed() -> None:
    """Exhausting the retries is a real failure; a silently dropped event is worse."""
    log = FakeLog(races=99)

    with pytest.raises(SequenceRace):
        await _channel(log).progress("call-1", 1.0, None, None)

    assert log.records == []


async def test_a_question_is_answered_from_an_event_naming_the_same_elicitation() -> (
    None
):
    log = FakeLog()
    channel = _channel(log)

    async def answer() -> None:
        while not log.of_type(TYPES.elicitation_requested):
            await asyncio.sleep(0)
        asked = log.of_type(TYPES.elicitation_requested)[0]
        await log.append(
            asked.session_id,
            TYPES.elicitation_answered,
            {
                "elicitation_id": asked.payload["elicitation_id"],
                "action": "accept",
                "content": {"invoice": "INV-9"},
            },
        )

    replier = asyncio.create_task(answer())
    result = await channel.ask(_form())
    await replier

    assert result.action == "accept"
    assert result.content == {"invoice": "INV-9"}
    asked = log.of_type(TYPES.elicitation_requested)[0]
    assert asked.payload["requested_schema"] == _form().requested_schema
    assert asked.payload["message"] == "which invoice?"


async def test_the_question_write_is_retried_too() -> None:
    """The request event is a write to the same contended log as the progress one."""
    log = FakeLog(races=2)
    channel = _channel(log, timeout_s=0.05)

    result = await channel.ask(_form())

    assert result.action == "cancel"
    assert len(log.of_type(TYPES.elicitation_requested)) == 1


async def test_a_url_mode_question_is_declined_without_reaching_the_log() -> None:
    """Relaying a URL a third-party server chose would make this a redirector."""
    log = FakeLog()

    result = await _channel(log).ask(
        ElicitRequestURLParams(message="sign in", url="https://acme.example/oauth")
    )

    assert result.action == "decline"
    assert result.content is None
    assert log.records == []


async def test_an_answer_naming_a_different_question_does_not_end_the_wait() -> None:
    log = FakeLog()
    channel = _channel(log)

    async def answer_twice() -> None:
        while not log.of_type(TYPES.elicitation_requested):
            await asyncio.sleep(0)
        asked = log.of_type(TYPES.elicitation_requested)[0]
        await log.append(
            asked.session_id,
            TYPES.elicitation_answered,
            {"elicitation_id": "somebody-elses", "action": "accept", "content": {}},
        )
        await log.append(
            asked.session_id,
            TYPES.elicitation_answered,
            {
                "elicitation_id": asked.payload["elicitation_id"],
                "action": "accept",
                "content": {"invoice": "INV-2"},
            },
        )

    replier = asyncio.create_task(answer_twice())
    result = await channel.ask(_form())
    await replier

    assert result.content == {"invoice": "INV-2"}


async def test_an_unparseable_answer_is_skipped_rather_than_read_as_a_decline() -> None:
    """It may be answering a different question; declining would answer ours for it."""
    log = FakeLog()
    channel = _channel(log)

    async def answer() -> None:
        while not log.of_type(TYPES.elicitation_requested):
            await asyncio.sleep(0)
        asked = log.of_type(TYPES.elicitation_requested)[0]
        await log.append(
            asked.session_id, TYPES.elicitation_answered, {"nothing": "recognizable"}
        )
        await log.append(
            asked.session_id,
            TYPES.elicitation_answered,
            {"elicitation_id": asked.payload["elicitation_id"], "action": "decline"},
        )

    replier = asyncio.create_task(answer())
    result = await channel.ask(_form())
    await replier

    assert result.action == "decline"
    assert result.content is None


@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_a_refusing_answer_comes_back_as_that_action_with_no_content(
    action: str,
) -> None:
    log = FakeLog()
    channel = _channel(log)

    async def answer() -> None:
        while not log.of_type(TYPES.elicitation_requested):
            await asyncio.sleep(0)
        asked = log.of_type(TYPES.elicitation_requested)[0]
        await log.append(
            asked.session_id,
            TYPES.elicitation_answered,
            {
                "elicitation_id": asked.payload["elicitation_id"],
                "action": action,
                "content": {"ignored": "yes"},
            },
        )

    replier = asyncio.create_task(answer())
    result = await channel.ask(_form())
    await replier

    assert result.action == action
    assert result.content is None


async def test_an_unanswered_question_expires_as_cancel_and_writes_nothing_more() -> (
    None
):
    """The deadline is a constructor argument because the real one is fifty seconds.

    `asyncio.timeout` reads the running loop's clock, which no installed fixture can
    move, so a test bound to the module constant would be a fifty-second wall-clock
    wait rather than a test of the expiry.
    """
    log = FakeLog()

    result = await _channel(log, timeout_s=0.05).ask(_form())

    assert result.action == "cancel"
    assert [r.type for r in log.records] == [TYPES.elicitation_requested]


async def test_progress_reported_before_an_answer_is_read_back_ahead_of_it() -> None:
    log = FakeLog()
    channel = _channel(log)

    await channel.progress("call-1", 1.0, 3.0, "one")
    await channel.progress("call-1", 2.0, 3.0, "two")

    async def answer() -> None:
        while not log.of_type(TYPES.elicitation_requested):
            await asyncio.sleep(0)
        asked = log.of_type(TYPES.elicitation_requested)[0]
        await log.append(
            asked.session_id,
            TYPES.elicitation_answered,
            {"elicitation_id": asked.payload["elicitation_id"], "action": "accept"},
        )

    replier = asyncio.create_task(answer())
    await channel.ask(_form())
    await replier

    assert [r.type for r in log.records] == [
        TYPES.progress,
        TYPES.progress,
        TYPES.elicitation_requested,
        TYPES.elicitation_answered,
    ]
    assert [r.seq for r in log.records] == [1, 2, 3, 4]
