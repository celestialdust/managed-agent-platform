"""Running one Turn against the Agent Runtime and recording what it did.

Ordering is most of the job. The platform's sequence is assigned by the append, so the
order a caller reads events in is the order this loop appends them in -- which is why
each append is awaited before the next notification is consumed. Consuming
concurrently would hand the order to the scheduler, and a reader of the Event Log
cannot tell a reordered stream from a wrong one.

Nothing off the runtime's wire is copied wholesale into an event. Each mapped
notification names the fields that cross into the platform's payload, so a runtime
thread id or turn id cannot ride along in a field nobody thought about, and a
notification with no entry in the map is dropped -- a decision taken here when the map
was written rather than at run time (ADR-013).

A completed Turn is the platform's durability boundary (ADR-004), and what happens at
it -- shipping the Rollout out of the pod -- has its own failure modes, so this loop
notifies a collaborator instead of doing it. A Turn that never reached a completion
notification does not notify: the pod is the thing that just went quiet, and asking it
to ship state out is asking the wrong process.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Final, Protocol
from uuid import UUID, uuid5

from managed_agent.core.ids import SessionId, TurnId
from managed_agent.core.pod.repertoire import TextInput, TurnStartRequest
from managed_agent.core.ports import EventLogAppend
from managed_agent.core.session.event_append import append_in_order
from managed_agent.core.vocabulary import thread, tool_call, tool_server, turn

_RUNTIME_TURN_COMPLETED: Final = "turn/completed"
_RUNTIME_STATUS_FAILED: Final = "failed"


@dataclass(frozen=True, slots=True)
class RuntimeNotification:
    """One notification off the runtime's wire, parsed into the two parts this uses.

    Parsed *here* rather than by the connection. `shim/client.py`'s `notifications()`
    yields the inbound frame uninterpreted, on purpose: mapping an Agent Runtime event
    onto this platform's vocabulary is this module's job, and a connection that also
    mapped would change both when the transport moved and when the vocabulary did.
    """

    method: str
    body: dict[str, object]


def _notification_of(frame: dict[str, Any]) -> RuntimeNotification:
    """One inbound frame as the pair the loop below reads.

    A frame with no `method` cannot be dispatched and is given the empty string, which
    matches no entry in the map and is therefore dropped by the same branch that drops
    an unmapped method -- one path for "this frame says nothing to us" rather than two.
    """
    params = frame.get("params")
    return RuntimeNotification(
        method=str(frame.get("method", "")),
        body=params if isinstance(params, dict) else {},
    )


class RuntimeConnection(Protocol):
    """The two things running a Turn needs of the runtime, and nothing more.

    Declared here rather than imported so this module depends on the calls it makes
    instead of on the shape of the client that makes them; the concrete client is built
    at the pod's entry point.

    **Both members are spelled as `shim/client.py` spells them, and that is the whole
    point of the structural declaration.** The shipped client has `start_turn(request:
    TurnStartRequest)` and `notifications()`; a draft of this file declared
    `start_turn(prompt: str)` and `notifications(runtime_turn_id)`, which the concrete
    class does not satisfy and cannot be made to satisfy --
    `tests/session_shim/test_repertoire_closed.py` forbids a bare `str` parameter on any
    public method of the client. A Protocol nothing implements is not a seam, it is a
    second interface with an adapter owed between them.
    """

    async def start_turn(self, request: TurnStartRequest) -> str:
        """Issue the runtime's turn start and return its own turn id."""

    def notifications(self) -> AsyncIterator[dict[str, Any]]:
        """Every inbound notification for this connection, in arrival order, unparsed.

        Not per-Turn, because the connection's queue is not per-Turn. A pod holds one
        Session and a Session has one Turn in flight, so every frame arriving while
        this loop runs belongs to the Turn it is running -- and the loop stops at the
        completion notification for the thread it started that Turn on, so a frame
        belonging to the next Turn is never consumed here. That is an invariant of the
        caller, not of the connection, which is why it is written down rather than
        enforced by a parameter the connection does not take.

        Nor is it per-thread. One Turn can run on several threads -- the root's plus one
        per subagent the runtime spawned -- and all of their turns report down this one
        channel, which is why the stop condition names a thread rather than being "a
        completion arrived".
        """


class TurnCompleted(Protocol):
    """Told once, after a Turn's last event is recorded."""

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None: ...


def _no_fields(body: dict[str, object]) -> dict[str, object]:
    return {}


def _delta_text(body: dict[str, object]) -> dict[str, object]:
    """The one field of an agent-message delta that crosses into the platform's event.

    `delta` sits at the top of the notification's params beside `threadId`, `turnId`
    and `itemId`; only the text crosses, so no runtime identifier can ride along.
    """
    return {"text": str(body.get("delta", ""))}


_RUNTIME_ITEM_MCP_TOOL_CALL: Final = "mcpToolCall"


def _tool_call_fields(body: dict[str, object]) -> dict[str, object] | None:
    """The fields of a finished MCP tool call, or None if this item is not one.

    `item/completed` fires for EVERY kind of item the runtime finishes -- agent
    messages, reasoning blocks, shell commands, file edits -- so this is the one entry
    in the map whose method does not by itself decide that an event is owed. Returning
    None is how it says so, and it is why an extractor may return None at all: the
    alternative was a second table of predicates beside this one, keyed on the same
    methods, which is two places to edit for one new item kind.

    The item's kind is its `type` field, because the runtime tags this union
    externally: an `mcpToolCall` arrives as `{"type": "mcpToolCall", "server": ...,
    "tool": ..., "status": ...}` rather than nested under a key. Read one level too
    high and every frame looks like no tool call at all, which is a silence no test
    that only scripts tool calls can see -- so a case here scripts an agent message and
    asserts nothing is appended.

    `arguments` and `result` are in the frame and are deliberately not read. The
    vocabulary module says why: they are the tenant's data, they are unbounded, and an
    Event Log holding them would retain a copy of every payload an agent ever sent to a
    third party on the platform's clock rather than the tenant's.
    """
    item = body.get("item")
    if not isinstance(item, dict):
        return None
    if item.get("type") != _RUNTIME_ITEM_MCP_TOOL_CALL:
        return None
    return {
        "server": str(item.get("server", "")),
        "tool": str(item.get("tool", "")),
        "status": str(item.get("status", "")),
        "duration_ms": item.get("durationMs"),
    }


_THREAD_NAMESPACE: Final = UUID("6f2a1c4e-8b3d-4f7a-9e10-2c5b7d0a3e41")
"""The namespace platform thread identifiers are minted in.

    A fixed literal rather than a derived value, because the identifier a tenant saw
    last week has to be the identifier they see today: deriving the namespace from
    anything that moves -- a version, a deployment, a config -- would silently re-mint
    every thread id the next time that thing changed, and a provenance identifier that
    changes is worse than none.

    Generated once and pasted in. It carries no meaning and is not a secret; what it
    buys is that a platform thread id is a uuid5 in OUR namespace, so two deployments
    naming the same runtime thread agree, and the same runtime thread met in two Turns
    is one thread to a reader.
"""

_RUNTIME_SERVER_NAME: Final = "name"
_RUNTIME_SERVER_STATE: Final = "status"
_RUNTIME_SERVER_ERROR: Final = "error"
_STATES_THAT_LEAVE_NO_SERVER: Final = frozenset({"failed", "cancelled"})


def _server_startup_fields(body: dict[str, object]) -> dict[str, object] | None:
    """A tool server that will not answer, and the runtime's reason, or None.

    The runtime reports every startup transition on one method -- `starting`, `ready`,
    `failed`, `cancelled` -- and only the two that leave the Session without the tool
    become events. Returning None for the other two is the documented way a mapped
    method carries no event, and it is what keeps a healthy placement from writing a
    row per server per Turn saying the expected thing happened.

    A frame naming no server carries no event either, and that is not the same caution.
    An announcement with an empty `server` tells a reader that something is down and
    gives them nothing to look at, which is worse than the absence it would replace --
    the same reason `_thread_started_fields` publishes nothing rather than a thread
    id it does not have.

    `error` is omitted when the runtime sent none rather than written as null. The two
    states differ in exactly this: a failure carries text explaining it and a
    cancellation usually carries nothing, so a key that is always present would make
    "the runtime had nothing to say" and "the runtime said nothing was wrong" the same
    payload.
    """
    state = body.get(_RUNTIME_SERVER_STATE)
    if state not in _STATES_THAT_LEAVE_NO_SERVER:
        return None
    server = body.get(_RUNTIME_SERVER_NAME)
    if not isinstance(server, str) or not server:
        return None
    fields: dict[str, object] = {"server": server, "state": state}
    why = body.get(_RUNTIME_SERVER_ERROR)
    if isinstance(why, str) and why:
        fields["error"] = why
    return fields


_RUNTIME_THREAD_ID: Final = "threadId"
_RUNTIME_THREAD: Final = "thread"
_RUNTIME_PARENT_THREAD_ID: Final = "parentThreadId"


def _attribution(body: dict[str, object]) -> dict[str, object]:
    """Which thread this frame came from, as a payload fragment, or nothing.

    Read here and not in each extractor because it is on every frame rather than on any
    particular kind of one -- and because a new row in the map below would otherwise
    have to remember it, which is exactly the kind of thing a new row does not remember.
    The module's docstring promises that adding a mapping is a new row and nothing else
    changes; attribution being central is what keeps that true.

    The key is OMITTED when the runtime sent no thread id, rather than written as an
    empty string. An empty id is a value that compares equal to another empty id, so two
    unattributed events would group into a thread that never existed; an absent key says
    the frame did not say.
    """
    found = body.get(_RUNTIME_THREAD_ID)
    if not isinstance(found, str) or not found:
        return {}
    return {"thread_id": found}


def _thread_started_fields(body: dict[str, object]) -> dict[str, object] | None:
    """The thread that just began and the thread that spawned it, or None.

    Read out of the frame's nested `thread` object rather than from beside it, and this
    is the trap the failure reader below documents for `turn`: the runtime tags these
    externally, so `{"method": "thread/started", "params": {"thread": {"id":
    "thr_123"}}}` is the documented shape and a read of `params.threadId` finds nothing
    on it. Every published event would then carry an empty thread id, and no test that
    scripts a top-level id could see it.

    The top-level pair is read only when there is no `thread` object at all, because
    both shapes exist in this protocol -- requests address a thread by a top-level
    `threadId` while notifications carry the object -- and a runtime speaking the flat
    dialect would otherwise publish nothing. What the fallback must not do is rescue a
    nested frame that gave no id: on a frame announcing a child the beside-it id is as
    likely to name the spawner, so reading it there records the wrong thread as having
    started rather than recording nothing.

    Returning None when neither yields an id is the documented way for a mapped method
    to carry no event, and it is the right answer here: a `thread.started` naming no
    thread is a node in the tree that can never be matched to the events it is supposed
    to group, which is worse than the absence it replaces.
    """
    nested = body.get(_RUNTIME_THREAD)
    # The dialect is decided by whether the object is THERE, not by whether it answered.
    # A frame that sent a `thread` object and no id inside it did not withhold that id
    # somewhere else, so falling through to the beside-it `threadId` would publish the
    # SPAWNER as the thread that started -- a wrong record rather than a missing one.
    # Measured: with the fallthrough, the case for a beginning that names no thread
    # published one anyway, naming the parent.
    if isinstance(nested, dict):
        started, parent = nested.get("id"), nested.get(_RUNTIME_PARENT_THREAD_ID)
    else:
        started = body.get(_RUNTIME_THREAD_ID)
        parent = body.get(_RUNTIME_PARENT_THREAD_ID)
    if not isinstance(started, str) or not started:
        return None
    return {
        "thread_id": started,
        "parent_thread_id": parent if isinstance(parent, str) and parent else None,
    }


def _issued(session_id: SessionId, payload: dict[str, object]) -> dict[str, object]:
    """Replace every runtime thread id in a payload with the one this platform issued.

    **This is the boundary, and it is here because two signed artifacts disagree without
    it.** ADR-007 (MAP-A10) requires that no Agent Runtime thread identifier
    reaches the caller and that "every identifier the caller sees is one the platform
    issued". ADR-007 requires every event to carry thread attribution so a tenant can
    tell which agent said what. Publishing the runtime's own string satisfies the second
    and breaks the first; publishing nothing satisfies the first and breaks the second.
    A platform-issued id derived from the runtime's satisfies both, because attribution
    needs the ids to be *distinct and stable*, never to be the runtime's.

    `uuid5` over `session_id:runtime_id` rather than a counter. A counter would need
    state that lives as long as the Session -- threads appear across Turns and a per-
    Turn counter would call two different threads "1" -- and it would make an id depend
    on the order frames arrived in. This depends on nothing but the two values, so the
    same runtime thread is the same platform thread in every Turn, in every process, on
    every replay.

    Scoped by Session, so the same runtime thread id reused under two Sessions yields
    two platform ids. The runtime mints these per process and this platform runs one per
    Session, so a collision across Sessions would otherwise merge two tenants' threads.

    Applied at the one place events are appended rather than inside each extractor. An
    extractor returns what the frame said; nothing published can carry a runtime value
    because everything published goes through here. The alternative -- each extractor
    minting its own -- is a rule every future row has to remember, and the row that
    forgets it leaks.
    """
    translated = dict(payload)
    for key in ("thread_id", "parent_thread_id"):
        runtime = translated.get(key)
        if isinstance(runtime, str) and runtime:
            translated[key] = str(uuid5(_THREAD_NAMESPACE, f"{session_id}:{runtime}"))
    return translated


_MAPPED: Final[
    dict[str, tuple[str, Callable[[dict[str, object]], dict[str, object] | None]]]
] = {
    "turn/started": (turn.TURN_STARTED, _no_fields),
    "item/agentMessage/delta": (turn.TURN_MESSAGE_DELTA, _delta_text),
    "item/completed": (tool_call.TOOL_CALLED, _tool_call_fields),
    "thread/started": (thread.THREAD_STARTED, _thread_started_fields),
    "mcpServer/startupStatus/updated": (
        tool_server.TOOL_SERVER_UNAVAILABLE,
        _server_startup_fields,
    ),
}
"""Runtime method -> the published type it becomes and the fields that cross into it.
Adding a mapping is a new row; nothing else in this module changes.

An extractor returning None means the frame carries no event even though its method is
mapped, which is what lets one row cover `item/completed` -- a method the runtime sends
for every kind of item it finishes, only one of which this platform publishes."""


def _completes_this_turn(body: dict[str, object], root_thread_id: str) -> bool:
    """Whether a `turn/completed` frame ends the Turn this loop is running.

    A `turn/completed` is not by itself this Turn's ending. The runtime runs a spawned
    subagent as its own thread with its own turn, and every one of those turns emits its
    own `turn/started` and `turn/completed` down the SAME notification channel -- there
    is one channel per connection, not one per thread. So the first completion to arrive
    during a delegating Turn is quite often a child's, and a loop that stopped at it
    would publish the root Turn as finished while the root agent was still talking: a
    truncated answer, or a child's failure recorded as the root's, with nobody left
    consuming the root's frames.

    Comparing against the thread the Turn was started on is what tells the two apart.
    The frame's `threadId` sits beside its `turn` object rather than inside it (the
    params are `{threadId, turn}`), and `root_thread_id` is the runtime's own id for the
    thread `start_turn` was called on, so the two are the same kind of string and
    compare directly. Neither value leaves the pod; this is a pod-local comparison and
    the ids published to a tenant are minted separately.

    A completion naming a DIFFERENT thread is skipped and the loop keeps consuming. It
    deliberately publishes nothing of its own: the only turn identifier this loop holds
    is the root Turn's, so any event appended for a child's completion would claim the
    root Turn ended -- a consumer would read one Turn completing twice. A subagent's
    turn boundaries are simply not something this platform publishes today; what it does
    publish about a subagent is `thread.started` and the deltas attributed to it.

    **A completion carrying NO thread id is treated as this Turn's.** The neighbouring
    `_thread_started_fields` learned the opposite lesson for its own frame -- there, a
    fallback rescuing a frame that said nothing would record the *wrong* thread as
    having started, so silence means drop. The trade here runs the other way, because
    the two mistakes are not the same size. Reading an unattributed completion as "not
    the root" ends no Turn at all: the loop would consume frames until the runtime
    connection dropped and then record every Turn as `runtime_lost`, for every Session,
    on any runtime dialect that omits the field. Reading it as the root is what every
    single-agent Turn has relied on since before this filter existed, and its worst case
    needs a runtime that both spawns subagents and omits thread ids from completions.
    Terminating is also the fail-safe direction for the pod itself: a Turn that never
    ends holds the pod open forever, and an unattributed frame carries nothing that
    could ever prove it belongs to a child.
    """
    named = body.get(_RUNTIME_THREAD_ID)
    if not isinstance(named, str) or not named:
        return True
    return named == root_thread_id


def _reported_failure(body: dict[str, object]) -> bool:
    """Whether a `turn/completed` frame is reporting a Turn that failed.

    The status is a field of the frame's `turn` object, not a sibling of it: the
    notification's params are `{threadId, turn}` and the Turn carries `status` plus an
    error populated only when that status is failed. Read one level too high the key is
    simply absent, so every failure would read as a success and the failure branch
    below would be unreachable -- a defect no test that scripts only successes can see.

    The runtime's own error is deliberately not read. A tenant sees a platform cause
    and never the runtime's text (ADR-013).
    """
    reported = body.get("turn")
    if not isinstance(reported, dict):
        return False
    return bool(reported.get("status") == _RUNTIME_STATUS_FAILED)


_PROGRESS_INTERVAL_S: Final = 30.0
"""How often a running Turn says what it is doing, whatever the runtime is doing.

Thirty seconds is chosen from the two costs it sits between, and neither is subtle. Too
long and the control plane waits that much longer before it can tell a wedged pod from
a busy one, because the sweep can only ever act on the last report it has. Too short
and a long Turn writes a row a second into a log the platform retains and every later
fold re-reads -- an hour-long Turn at this interval adds 120 rows, which is small
beside the per-token deltas the same Turn already writes.

**It is deliberately not derived from any deadline.** A reporting interval and a
give-up threshold are two different decisions, and tying them would mean that making
the platform more patient also made it blinder. The sweep decides how many missed
reports it will tolerate; this decides only how often the truth is available.
"""


@dataclass(slots=True)
class _Progress:
    """The monotonic facts one Turn has accumulated, as the ticker reads them.

    Mutable, and the one mutable thing in this module, because it is a counter: the
    loop bumps it and the ticker reads it, and a frozen value passed between them would
    be a snapshot that stopped being true the moment it was taken.

    Both counts only ever rise, which is what makes them usable as evidence of
    progress. A reader comparing two reports of the same Turn can conclude that work
    happened between them if either number moved, and that none did if neither moved --
    a conclusion no single report can support on its own.
    """

    frames: int = 0
    answer_bytes: int = 0
    last_frame_at: float = 0.0


async def _report_progress(
    session_id: SessionId,
    turn_id: TurnId,
    attribution: dict[str, object],
    progress: _Progress,
    log: EventLogAppend,
    interval_s: float,
) -> None:
    """Append what this Turn is doing, every `interval_s`, until cancelled.

    Runs as its own task beside the notification loop, which is the only place in this
    module that appends off the loop's own thread of control. The module docstring
    insists appends are awaited in order so a reader cannot see a reordered stream, and
    this does not weaken that: a progress report carries no part of the answer and
    holds no position in it, so where it lands among the deltas is not information. The
    ordering rule protects the sequence of the Turn's *content*, and this is not
    content.

    `idle_ms` is the report's most useful field and the one that needs saying out loud:
    it is how long the runtime has been silent, and a large value here is a **named**
    state rather than an absence. A Turn blocked on a slow model for ninety seconds
    reports that it has been blocked for ninety seconds, which is a healthy Turn
    describing itself -- the distinction the retired inter-byte deadline could not draw,
    because it inferred both liveness and progress from the same missing bytes.

    Cancellation is the only way out, and it is what the caller's `finally` does. A
    ticker outliving its Turn would append to a Turn that has already ended, for as long
    as the process ran.
    """
    while True:
        await asyncio.sleep(interval_s)
        idle_ms = int((time.monotonic() - progress.last_frame_at) * 1000)
        try:
            await append_in_order(
                log,
                session_id,
                turn.TURN_PROGRESS,
                {
                    "turn_id": str(turn_id),
                    **attribution,
                    "frames": progress.frames,
                    "answer_bytes": progress.answer_bytes,
                    "idle_ms": idle_ms,
                },
            )
        except Exception:  # noqa: BLE001 - see below; this must not end the Turn
            # A refused report is dropped and the next one is attempted. This is the
            # one place in this module where swallowing is right, and it is right for
            # a specific reason rather than for convenience: these reports are a
            # running commentary in which **only the newest one is read**
            # (`abandoned_turns.latest_idle_ms` scans backwards and stops at the first
            # match), so a report that never lands loses nothing the next one does not
            # carry thirty seconds later.
            #
            # What the alternative costs was measured rather than argued. This ran as
            # a bare `await` inside a bare `create_task` that nothing supervises, so a
            # single failed append ended the *task*, not the iteration: the Turn then
            # reported nothing for the rest of its life while working perfectly, which
            # is exactly the state the sweep's only live-pod signal cannot see -- and
            # since the hour-long ceiling was removed, nothing else closes it either.
            # Worse, the dead task held its exception until `run_turn`'s `finally`
            # awaited it, where `suppress` catches only `CancelledError`; a finished
            # Turn therefore raised out of `run_turn` and never appended
            # `turn.completed`. A transient blip on the control plane destroyed
            # completed work.
            #
            # `Exception` and not `BaseException`: `CancelledError` derives from
            # `BaseException`, and cancellation is the one way out of this loop that
            # must keep working -- the caller's `finally` depends on it.
            continue


async def run_turn(
    session_id: SessionId,
    turn_id: TurnId,
    thread_id: str,
    prompt: str,
    connection: RuntimeConnection,
    log: EventLogAppend,
    on_completed: TurnCompleted,
    progress_interval_s: float = _PROGRESS_INTERVAL_S,
) -> None:
    """Run one Turn and append every event it produced, in the order they arrived.

    The completed event carries the whole answer as well as the deltas that built it,
    so a caller that did not hold the stream open still has the result from one event
    instead of having to reassemble it.

    `thread_id` is the runtime's own id for the Session's root thread, returned by
    `start_thread` at the pod's entry point. It is required because `TurnStartRequest`
    carries it: a Turn is started *on a thread*, and there is no default. It is
    pod-local and never leaves the pod (ADR-007), which is why it is a plain `str`
    parameter here rather than anything this module stores or publishes.

    The runtime's own turn id comes back from `start_turn` and is deliberately unused
    below -- the loop stops at the completion notification for `thread_id` rather than
    filtering by turn. Keeping the call's return value named says what it is; discarding
    it silently would leave a reader wondering whether a filter was forgotten. Filtering
    on the thread instead is enough because a thread runs one turn at a time, and it is
    the filter the notification channel actually needs: the channel is shared by every
    thread in the Session, so a subagent's completion arrives here too.

    The Turn's own terminal event -- `turn.completed` or `turn.failed` -- is attributed
    to `thread_id` like every other event, and carries the platform's issued id for it
    rather than the runtime's string. These three appends are built here instead of
    being translated from a mapped frame, so the central attribution that every mapped
    event goes through does not reach them; without this they would be the only events
    of a Turn that did not say which agent produced them.
    """
    _runtime_turn_id = await connection.start_turn(
        TurnStartRequest(thread_id=thread_id, input=(TextInput(text=prompt),))
    )
    root_thread = _issued(session_id, {"thread_id": thread_id})
    answer: list[str] = []
    completed = False
    progress = _Progress(last_frame_at=time.monotonic())
    ticker = asyncio.create_task(
        _report_progress(
            session_id, turn_id, root_thread, progress, log, progress_interval_s
        )
    )
    try:
        completed = await _drive(
            session_id,
            turn_id,
            thread_id,
            root_thread,
            connection,
            log,
            answer,
            progress,
        )
    finally:
        # Cancelled and then awaited, so the task is finished before this returns. A
        # bare `cancel()` only requests it, and the report already inside its append
        # would land after the Turn's terminal event -- the one ordering this module
        # genuinely cannot allow, because a reader folding the log would see a Turn
        # still working after it had ended.
        ticker.cancel()
        with suppress(asyncio.CancelledError):
            await ticker

    if not completed:
        await append_in_order(
            log,
            session_id,
            turn.TURN_FAILED,
            {
                "turn_id": str(turn_id),
                **root_thread,
                "cause": turn.TurnFailureCause.RUNTIME_LOST.value,
            },
        )
        return
    await on_completed.turn_completed(session_id, turn_id)


async def _drive(
    session_id: SessionId,
    turn_id: TurnId,
    thread_id: str,
    root_thread: dict[str, object],
    connection: RuntimeConnection,
    log: EventLogAppend,
    answer: list[str],
    progress: _Progress,
) -> bool:
    """Consume the notification channel until this Turn ends. Returns whether it did.

    Split out of `run_turn` so the ticker's `finally` wraps one expression instead of
    the whole body: the loop below appends the Turn's terminal event, and that append
    has to happen while the ticker is still cancellable but before it is awaited.

    Every frame bumps `progress.frames`, mapped or not. That is deliberate and it is
    the widest signal available here -- a notification this module drops is still
    evidence the runtime is producing something, and dropping it from the count as well
    as from the log would make the shim blind to exactly the work it does not publish.
    """
    completed = False
    async for frame in connection.notifications():
        progress.frames += 1
        progress.last_frame_at = time.monotonic()
        note = _notification_of(frame)
        if note.method == _RUNTIME_TURN_COMPLETED:
            if not _completes_this_turn(note.body, thread_id):
                # A subagent finished, not this Turn. Nothing is appended: see
                # `_completes_this_turn` for why a child's ending gets no event of its
                # own rather than one carrying the root Turn's id.
                continue
            completed = True
            if _reported_failure(note.body):
                await append_in_order(
                    log,
                    session_id,
                    turn.TURN_FAILED,
                    {
                        "turn_id": str(turn_id),
                        **root_thread,
                        "cause": turn.TurnFailureCause.RUNTIME_REPORTED_FAILURE.value,
                    },
                )
            else:
                await append_in_order(
                    log,
                    session_id,
                    turn.TURN_COMPLETED,
                    {
                        "turn_id": str(turn_id),
                        **root_thread,
                        "text": "".join(answer),
                    },
                )
            break
        entry = _MAPPED.get(note.method)
        if entry is None:
            continue
        type_, carry = entry
        carried = carry(note.body)
        if carried is None:
            continue
        if type_ == turn.TURN_MESSAGE_DELTA:
            text = str(carried["text"])
            answer.append(text)
            progress.answer_bytes += len(text.encode())
        # `carried` is spread last so an extractor always outranks the central
        # attribution. That matters for `thread/started`, whose nested thread id is the
        # authoritative one and whose beside-it id may name the spawner instead.
        await append_in_order(
            log,
            session_id,
            type_,
            _issued(
                session_id,
                {"turn_id": str(turn_id), **_attribution(note.body), **carried},
            ),
        )

    return completed
