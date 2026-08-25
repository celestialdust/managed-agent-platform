"""Running one Turn: what gets appended, in what order, and what never crosses.

Tier 1 (local, no infrastructure). Realizes the running half of MAP-A110 -- a Turn runs
to completion and its events land in the Event Log in the order they arrived.

The connection is scripted rather than real, and the frames are scripted in the Agent
Runtime's own shape, taken from the protocol source under `.reference/codex`. That is
the whole value of a fake here: a fake built from a guess certifies the guess. The
completion frame is the one that matters -- its params are `{threadId, turn}` and the
status lives on `turn`, so a runner reading `params["status"]` finds nothing, treats
every failed Turn as a success, and passes any test whose frames were invented to match
the code.

Ordering is asserted as a list rather than as a set. The platform's sequence is assigned
by the append, so the order these appends happen in *is* the order a caller reads the
Turn in, and a reader of the Event Log cannot tell a reordered stream from a wrong one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4, uuid5

import pytest

from managed_agent.core.ids import Seq, SessionId, TurnId, new_session_id
from managed_agent.core.pod.repertoire import TurnStartRequest
from managed_agent.core.ports import SequenceRace
from managed_agent.core.vocabulary import tool_call, turn
from managed_agent.session_shim.turn_runner import _THREAD_NAMESPACE, run_turn

_THREAD = "thread-inside-the-pod"
_CHILD = "thread-of-the-subagent"
_RUNTIME_TURN = "runtime-turn-9"
_PROMPT = "summarise the findings"


@dataclass(frozen=True, slots=True)
class Appended:
    type: str
    payload: dict[str, object]


class RecordingLog:
    """Records what was appended, and hands out contiguous sequences."""

    def __init__(self, races_before_each_success: int = 0) -> None:
        self.written: list[Appended] = []
        self._races_left = races_before_each_success

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        if self._races_left:
            self._races_left -= 1
            raise SequenceRace("another writer took this sequence")
        self.written.append(Appended(type_, dict(payload)))
        return Seq(len(self.written))

    def types(self) -> list[str]:
        return [event.type for event in self.written]


class AlwaysRaces:
    """Never lets an append land, so the retry ceiling is reached."""

    def __init__(self) -> None:
        self.attempts = 0

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.attempts += 1
        raise SequenceRace("another writer took this sequence")


class Scripted:
    """A runtime that yields a prepared list of frames and then stops."""

    def __init__(self, frames: Sequence[dict[str, Any]]) -> None:
        self._frames = list(frames)
        self.started: list[TurnStartRequest] = []

    async def start_turn(self, request: TurnStartRequest) -> str:
        self.started.append(request)
        return _RUNTIME_TURN

    async def notifications(self) -> AsyncIterator[dict[str, Any]]:
        for frame in self._frames:
            yield frame


class Notified:
    def __init__(self) -> None:
        self.told: list[tuple[SessionId, TurnId]] = []

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        self.told.append((session_id, turn_id))


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


def _completed(
    status: str = "completed", *, thread_id: str | None = _THREAD
) -> dict[str, Any]:
    """A completion in the runtime's own shape: the status is a field of `turn`.

    `thread_id` names the thread whose turn finished, and it is a parameter because a
    completion arriving on this channel is not necessarily this Turn's: a spawned
    subagent runs its own turn on its own thread and reports it here too. The default is
    the root thread, so a case that says nothing about threads scripts the Turn's own
    ending.

    `None` omits the key entirely rather than writing an empty string. An absent
    `threadId` and an empty one are different frames to the code under test, and only
    the absent one is something a runtime dialect would actually send.
    """
    params: dict[str, Any] = {}
    if thread_id is not None:
        params["threadId"] = thread_id
    return {
        "method": "turn/completed",
        "params": {
            **params,
            "turn": {
                "id": _RUNTIME_TURN,
                "items": [],
                "status": status,
                "error": (
                    {"message": "the model refused"} if status == "failed" else None
                ),
            },
        },
    }


def _platform_thread(session_id: SessionId, runtime_id: str = _THREAD) -> str:
    """The identifier this platform issues for one runtime thread, re-derived here.

    Derived from the published namespace and the documented `session:runtime` form
    rather than read back off the event under test, which would assert that a value
    equals itself. What it deliberately does not do is call the module's own issuing
    function -- that would pass on a function returning its input, which is precisely
    the leak ADR-007 (MAP-A10) forbids.
    """
    return str(uuid5(_THREAD_NAMESPACE, f"{session_id}:{runtime_id}"))


async def _run(
    frames: Sequence[dict[str, Any]],
    log: RecordingLog | AlwaysRaces | None = None,
) -> tuple[RecordingLog | AlwaysRaces, Notified, SessionId, TurnId, Scripted]:
    written = RecordingLog() if log is None else log
    notified = Notified()
    session_id, turn_id = new_session_id(), TurnId(uuid4())
    runtime = Scripted(frames)
    await run_turn(session_id, turn_id, _THREAD, _PROMPT, runtime, written, notified)
    return written, notified, session_id, turn_id, runtime


async def test_a_whole_turn_lands_in_arrival_order_at_contiguous_sequences() -> None:
    log, notified, session_id, turn_id, runtime = await _run(
        [_started(), _delta("the "), _delta("domain "), _delta("answer"), _completed()]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert notified.told == [(session_id, turn_id)]
    assert runtime.started[0].thread_id == _THREAD
    assert runtime.started[0].input[0].text == _PROMPT


async def test_the_completed_event_carries_the_whole_answer_the_deltas_built() -> None:
    """One event holds the result, so a caller that did not hold the stream open still
    has it without reassembling three payloads in the right order."""
    log, _, session_id, turn_id, _ = await _run(
        [_started(), _delta("the "), _delta("domain "), _delta("answer"), _completed()]
    )

    assert isinstance(log, RecordingLog)
    # Exact equality kept, with the attribution ADR-007 requires added to it: the
    # Turn's ending is the root thread's event like every other, and equality is what
    # would catch a field appearing here that nothing decided to publish.
    assert log.written[-1].payload == {
        "turn_id": str(turn_id),
        "thread_id": _platform_thread(session_id),
        "text": "the domain answer",
    }


async def test_a_completion_reporting_failure_appends_a_failure_and_no_answer() -> None:
    """The status is read off `params.turn`, which is where the runtime puts it.

    Falsifiable, and this is the mutation that matters: reading `params["status"]`
    instead finds nothing on a real frame, so this Turn is recorded as `turn.completed`
    carrying whatever partial text arrived -- a failure served to the tenant as an
    answer, with `RUNTIME_REPORTED_FAILURE` left as dead code.
    """
    log, notified, session_id, turn_id, _ = await _run(
        [_started(), _delta("half an "), _completed(status="failed")]
    )

    assert isinstance(log, RecordingLog)
    assert log.types()[-1] == turn.TURN_FAILED
    assert log.written[-1].payload == {
        "turn_id": str(turn_id),
        "thread_id": _platform_thread(session_id),
        "cause": "runtime_reported_failure",
    }
    assert notified.told == [(session_id, turn_id)], (
        "a Turn that reached its completion notification is still a Turn the pod "
        "finished, so the collaborator that ships state out is still told"
    )


async def test_the_runtime_s_own_error_text_never_reaches_an_event() -> None:
    """A tenant sees a platform cause and never the runtime's own (ADR-013)."""
    log, _, _, _, _ = await _run([_completed(status="failed")])

    assert isinstance(log, RecordingLog)
    assert "the model refused" not in repr(log.written)


async def test_a_stream_ending_with_no_completion_fails_and_tells_nobody() -> None:
    """The pod is the thing that just went quiet, so asking it to ship state out is
    asking the wrong process."""
    log, notified, session_id, turn_id, _ = await _run([_started(), _delta("half an ")])

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_FAILED,
    ]
    assert log.written[-1].payload == {
        "turn_id": str(turn_id),
        "thread_id": _platform_thread(session_id),
        "cause": "runtime_lost",
    }
    assert notified.told == []


async def test_a_notification_with_no_entry_in_the_map_appends_nothing() -> None:
    """Closed by construction: an unmapped runtime event never reaches a tenant.

    The two frames here are real methods the runtime does emit and this platform has
    decided nothing about, plus one with no method at all -- which takes the same
    branch rather than a second one.
    """
    log, _, _, _, _ = await _run(
        [
            {"method": "thread/tokenUsage/updated", "params": {"threadId": _THREAD}},
            {"method": "item/reasoning/textDelta", "params": {"delta": "thinking"}},
            {"params": {"delta": "no method at all"}},
            _completed(),
        ]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [turn.TURN_COMPLETED]
    assert log.written[-1].payload["text"] == ""


async def test_no_appended_payload_carries_a_runtime_identifier() -> None:
    """MAP-A10's invariant, at the one place a runtime id could ride along.

    Every frame above carries the runtime's thread id and turn id, and the connection
    returns its own turn id from `start_turn`. None of the three may appear in anything
    appended: the map names the fields that cross, so a payload copied wholesale is the
    failure this asserts against.
    """
    log, _, _, _, _ = await _run([_started(), _delta("the answer"), _completed()])

    assert isinstance(log, RecordingLog)
    written = repr(log.written)
    assert _RUNTIME_TURN not in written
    assert _THREAD not in written
    assert "item-1" not in written


async def test_an_append_losing_two_sequence_races_still_produces_the_whole_run() -> (
    None
):
    """A lost race needs no recovery beyond asking again for a number.

    The store chooses the sequence, so a lifecycle transition landing mid-Turn simply
    takes the number this append wanted. What must not happen is a Turn losing an event
    because another writer was quicker.
    """
    log, _, _, _, _ = await _run(
        [_started(), _delta("the answer"), _completed()],
        log=RecordingLog(races_before_each_success=2),
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]


async def test_an_append_that_never_wins_is_raised_rather_than_swallowed() -> None:
    """Exhausting the attempts is a real failure and reaching the ceiling says so."""
    log = AlwaysRaces()

    with pytest.raises(SequenceRace, match="lost 8 sequence races"):
        await _run([_started(), _completed()], log=log)

    assert log.attempts == 8


async def test_the_turn_id_in_every_payload_is_the_platform_s_own() -> None:
    """The id a tenant holds is the one the platform minted, on every event."""
    log, _, _, turn_id, _ = await _run([_started(), _delta("the answer"), _completed()])

    assert isinstance(log, RecordingLog)
    assert {UUID(str(event.payload["turn_id"])) for event in log.written} == {turn_id}


# ------------------------------------------------------------------------------------
# What the agent DID: tool calls, which until this row nothing anywhere recorded
# ------------------------------------------------------------------------------------


def _item_completed(item: dict[str, Any]) -> dict[str, Any]:
    """One `item/completed` frame in the runtime's own shape, carrying any item.

    Built from the protocol source rather than from the code under test:
    `ItemCompletedNotification` is `{item, threadId, turnId, completedAtMs}` and
    `ThreadItem` is an externally tagged union, so the kind is `item["type"]` and the
    kind's own fields sit beside it rather than nested under a key. A helper that nested
    them would certify a runner that read them nested, and every real frame would then
    map to nothing.
    """
    return {
        "method": "item/completed",
        "params": {
            "item": item,
            "threadId": _THREAD,
            "turnId": _RUNTIME_TURN,
            "completedAtMs": 1_700_000_000_000,
        },
    }


def _mcp_tool_call(
    *, server: str = "deepwiki", tool: str = "ask_deepwiki", status: str = "completed"
) -> dict[str, Any]:
    """A finished MCP tool call item, with the fields a real one carries.

    `arguments` and `result` are present on purpose even though nothing reads them: a
    fixture that omitted them could not catch a runner that started copying them, and
    "the tenant's arguments are not in the Event Log" is one of the two claims the
    vocabulary module makes about this event.
    """
    return {
        "type": "mcpToolCall",
        "id": "item-tool-1",
        "server": server,
        "tool": tool,
        "status": status,
        "arguments": {"repoName": "acme/private", "question": "what is the key"},
        "result": {"content": [{"type": "text", "text": "the answer"}]},
        "durationMs": 1234,
        "error": None,
    }


async def test_a_finished_tool_call_is_recorded_as_the_event_the_tenant_can_read() -> (
    None
):
    """One `tool.called`, between the deltas, saying which tool on which server.

    The event a tenant asking "what did my agent do" reads. Before this row existed the
    honest answer was the agent's prose, which is a report by the thing being audited.
    """
    log, _, session_id, turn_id, _ = await _run(
        [
            _started(),
            _delta("calling out"),
            _item_completed(_mcp_tool_call()),
            _delta("done"),
            _completed(),
        ]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        tool_call.TOOL_CALLED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    recorded = next(
        one for one in log.written if one.type == tool_call.TOOL_CALLED
    ).payload
    # Exact equality kept, with the attribution ADR-007 requires added to it. The
    # equality is the guard against a payload copied wholesale out of the frame, so it
    # grows a key rather than becoming a subset check.
    assert recorded == {
        "turn_id": str(turn_id),
        "thread_id": _platform_thread(session_id),
        "server": "deepwiki",
        "tool": "ask_deepwiki",
        "status": "completed",
        "duration_ms": 1234,
    }


async def test_the_tenants_arguments_and_the_tools_answer_never_reach_the_log() -> None:
    """Asserted as the absence of the values, not of the key names.

    A key check passes for a runner that copied `arguments` into a field called
    something else, and the disclosure is the values leaving the pod rather than the
    spelling they left under. Both the argument text and the result text are searched
    for across the whole recorded payload.
    """
    log, _, _, _, _ = await _run(
        [_started(), _item_completed(_mcp_tool_call()), _completed()]
    )

    assert isinstance(log, RecordingLog)
    everything = repr([one.payload for one in log.written])
    for leaked in ("acme/private", "what is the key", "the answer"):
        assert leaked not in everything, (
            f"{leaked!r} reached the Event Log from a tool call's arguments or result; "
            "the log would then hold a copy of every payload an agent sent to a third "
            "party, on the platform's retention clock rather than the tenant's"
        )


async def test_a_failed_tool_call_is_recorded_rather_than_dropped() -> None:
    """A call that failed is the one a reader most needs, so it is not filtered.

    The status crosses verbatim. A runner that recorded only successes would leave a
    Session whose every tool call failed looking like a Session that called nothing --
    and "the agent never tried" and "the agent tried and could not reach it" send a
    reader to two different places.
    """
    log, _, _, _, _ = await _run(
        [_started(), _item_completed(_mcp_tool_call(status="failed")), _completed()]
    )

    assert isinstance(log, RecordingLog)
    statuses = [
        one.payload["status"]
        for one in log.written
        if one.type == tool_call.TOOL_CALLED
    ]
    assert statuses == ["failed"]


@pytest.mark.parametrize(
    "item",
    [
        {"type": "agentMessage", "id": "item-2", "text": "hello"},
        {"type": "commandExecution", "id": "item-3", "command": "ls ./files"},
        {"type": "reasoning", "id": "item-4", "summary": ["thinking"]},
        {"type": "fileChange", "id": "item-5", "changes": []},
        {"type": "webSearch", "id": "item-6"},
    ],
    ids=["agentMessage", "commandExecution", "reasoning", "fileChange", "webSearch"],
)
async def test_an_item_that_is_not_a_tool_call_appends_nothing(
    item: dict[str, Any],
) -> None:
    """`item/completed` fires for every kind of item, and only one kind is published.

    This is the case that makes the mapped-but-silent branch real. `item/completed` is
    the one entry in `_MAPPED` whose method does not by itself decide an event is owed,
    so a runner that read the item's kind one level too high -- or not at all -- would
    append a `tool.called` for every message the agent wrote, every command it ran and
    every file it touched, each with three empty strings in it. Parametrized because
    each of these is a kind the runtime really sends and a filter could admit any one of
    them alone.

    `commandExecution` is the member worth naming: a shell command is NOT recorded, and
    that is a stated gap rather than an accident. The vocabulary module says why -- a
    command line can carry a secret an argument list cannot, which is a different
    disclosure question from this event's.
    """
    log, _, _, _, _ = await _run([_started(), _item_completed(item), _completed()])

    assert isinstance(log, RecordingLog)
    assert log.types() == [turn.TURN_STARTED, turn.TURN_COMPLETED]


async def test_a_frame_whose_item_is_not_a_dictionary_appends_nothing() -> None:
    """A malformed frame is dropped, not turned into an event full of empty strings.

    Reachable rather than defensive: the runtime's protocol is a moving target, this pod
    is pinned to a codex version chosen at image build, and the failure mode of guessing
    is an Event Log full of `tool.called` rows naming no tool. Dropping is right here --
    the alternative is failing a Turn over a frame the platform does not publish.
    """
    log, _, _, _, _ = await _run(
        [
            _started(),
            {"method": "item/completed", "params": {"item": "not a dictionary"}},
            {"method": "item/completed", "params": {}},
            _completed(),
        ]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [turn.TURN_STARTED, turn.TURN_COMPLETED]


async def test_every_type_this_module_can_emit_is_one_the_pod_may_put_on_the_wire() -> (
    None
):
    """Every published type in `_MAPPED` is accepted by the shim's outbound wire.

    Two gates stand between a runtime frame and a tenant's Event Log: that mapping, and
    `serve.SHIM_EVENT_TYPES`, which the streaming route checks every line against. A
    type in the first and not the second is appended inside the pod and refused on its
    way out, and the refusal is a `ValidationError` from `TurnEventLine` that aborts the
    whole Turn -- not a dropped event, the whole Turn.

    **Derived from the map rather than naming a type, because the version that named one
    let exactly this through.** It asserted `tool.called` was admitted and nothing else,
    and its sibling in `test_shim_serves_a_turn.py` compared the wire set against the
    `turn` family plus that one name. `thread.started` is in the `thread` family, so it
    changed neither side of either check, passed both, and shipped in an image where
    every Turn died with that validation error. Iterating the map is the form that
    cannot be blind to a family nobody thought of.

    The emptiness assertion is the vacuity control: `all()` over an empty map is true,
    and this case's whole subject is the map's contents.
    """
    from managed_agent.session_shim.serve import SHIM_EVENT_TYPES
    from managed_agent.session_shim.turn_runner import _MAPPED

    emitted = {type_ for type_, _ in _MAPPED.values()}
    assert emitted, "the map is empty, so this asserts nothing"
    assert emitted <= SHIM_EVENT_TYPES, emitted - SHIM_EVENT_TYPES


# ------------------------------------------------------------------------------------
# Whose completion ends the Turn, once a Turn runs on more than one thread
# ------------------------------------------------------------------------------------


async def test_a_subagents_completion_does_not_end_the_turn_that_spawned_it() -> None:
    """A child's ending is not the root's, and the answer after it is still collected.

    The runtime gives a spawned subagent its own thread running its own turn, and that
    turn reports down the SAME notification channel this loop reads -- there is one
    channel per connection, not one per thread. So during a delegating Turn the first
    `turn/completed` to arrive is routinely a child's, and a loop that stopped at any
    completion would publish the root Turn as finished while the root agent was still
    talking.

    The deltas are scripted AFTER the child's completion on purpose, and the text
    assertion is what gives this case teeth: a types-only check would still pass for a
    runner that ended the Turn at the child's frame, because `turn.completed` would be
    appended either way -- just carrying half an answer.
    """
    log, notified, session_id, turn_id, _ = await _run(
        [
            _started(),
            _delta("delegating, "),
            _completed(thread_id=_CHILD),
            _delta("and the answer "),
            _delta("is seven"),
            _completed(),
        ]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert log.written[-1].payload["text"] == "delegating, and the answer is seven"
    assert notified.told == [(session_id, turn_id)]


async def test_a_subagents_completion_publishes_no_event_of_its_own() -> None:
    """A child's ending is skipped, not translated into a terminal event of its own.

    The only turn identifier this loop holds is the root Turn's, so any event appended
    for a child's completion would claim the ROOT Turn ended -- and a consumer reading
    the Event Log would see one Turn complete twice, the first copy carrying a partial
    answer. There is no id it could carry instead: a subagent's turn boundaries are not
    something this platform publishes, and what it does publish about a subagent is
    `thread.started` plus the deltas attributed to it.

    Two children rather than one, because "one extra event" and "one event per child"
    are different mistakes and only the second is visible with two. The delta between
    them is what makes the count assertion more than a restatement of the sequence: a
    runner that ended the Turn at the first child would also record exactly one
    `turn.completed`, and only its position in the sequence gives it away.
    """
    log, _, _, _, _ = await _run(
        [
            _started(),
            _completed(thread_id=_CHILD),
            _delta("the answer"),
            _completed(thread_id="thread-of-a-second-subagent"),
            _completed(),
        ]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert log.types().count(turn.TURN_COMPLETED) == 1


async def test_a_subagents_reported_failure_does_not_fail_the_turn() -> None:
    """A child that failed does not make the root Turn a failure.

    The runtime publishes no failure notification of its own -- a failed turn arrives as
    `turn/completed` with a `failed` status on its `turn` object -- so a child's failure
    reaches this loop through exactly the frame the root's success would. Read as the
    root's, it records `turn.failed` for a Turn that went on to answer, and the answer
    is then lost with it: `turn.failed` carries no text. Delegating to a subagent that
    fails and recovering from it is ordinary agent behaviour, not an error a tenant
    should be shown.
    """
    log, notified, session_id, turn_id, _ = await _run(
        [
            _started(),
            _completed(status="failed", thread_id=_CHILD),
            _delta("the subagent failed so I did it myself"),
            _completed(),
        ]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert log.written[-1].payload["text"] == "the subagent failed so I did it myself"
    assert notified.told == [(session_id, turn_id)]


async def test_the_root_threads_completion_still_ends_the_turn() -> None:
    """The filter narrows what ends the Turn; it does not stop the root from ending it.

    The frame after the completion is the falsifiable half. A filter matching nothing --
    one reading the wrong field, so that it never matched at all -- would leave the loop
    consuming, and this Turn would reach the end of the stream and be recorded as
    `runtime_lost` instead. Asserting that the trailing delta is absent says the loop
    genuinely stopped rather than merely arriving at the same first three events.
    """
    log, notified, session_id, turn_id, _ = await _run(
        [
            _started(),
            _completed(thread_id=_CHILD),
            _delta("the answer"),
            _completed(),
            _delta("said after the Turn was over"),
        ]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert "said after the Turn was over" not in repr(log.written)
    assert notified.told == [(session_id, turn_id)]


async def test_a_completion_naming_no_thread_is_read_as_this_turns_own() -> None:
    """A completion carrying no thread id ends the Turn, rather than being skipped.

    The deliberate opposite call from `thread/started`, where a frame that named no
    thread publishes nothing. There, believing the frame would record the WRONG thread
    as having started; here, disbelieving it ends no Turn at all -- the loop would
    consume to the end of the stream and record `runtime_lost`, for every Session, on
    any runtime dialect that omits the field. That is a total failure against a partial
    one, and it would take away the behaviour every single-agent Turn has relied on
    since before this filter existed.

    The trailing delta is what proves the Turn actually stopped here rather than
    reaching the same event count by running out of frames -- the two look alike from
    the type list alone, and only one of them says `runtime_lost`.
    """
    log, notified, session_id, turn_id, _ = await _run(
        [
            _started(),
            _delta("the answer"),
            _completed(thread_id=None),
            _delta("said after the Turn was over"),
        ]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]
    assert log.written[-1].payload["text"] == "the answer"
    assert "said after the Turn was over" not in repr(log.written)
    assert notified.told == [(session_id, turn_id)]
