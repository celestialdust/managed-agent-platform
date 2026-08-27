"""A Turn runs to a terminal event whatever the submitting request does.

Three properties, and they are separable because they fail separately.

The first is that the POST returns before the Turn does. That is what the route has
advertised since it was written -- `status_code=202` -- and what it did not do: the
dispatch was awaited inline, so the promise of an accepted-and-running Turn was really
a completed one, and every consequence below followed from that.

The second is that the Turn's outcome is recorded by whatever runs the Turn, not by
whoever submitted it. This is the property with a trap in it: move the dispatch off the
request and leave the `except` blocks behind, and a Turn that fails after the response
has been sent records nothing at all -- which leaves the Session `RUNNING` with an open
Turn that refuses its next Turn and its archive, and is strictly worse than holding the
request open. So the failing endings are graded here more heavily than the passing one.

The third is that silence on the wire is not a failure. A gap between bytes used to end
a Turn at 120 seconds; an agent writing a file produces exactly that gap and was killed
by it in the field. Nothing bounds a Turn's elapsed time any more either -- the
hour-long total went on 2026-08-26, for the same reason one level up: an agent run has
no natural length, so a clock cannot tell a long one from a dead one. What is left on
this call is a read timeout no working pod can reach, guarding a half-open socket
rather than a Turn.
"""

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from managed_agent.control.session.turn_dispatch import (
    TurnOutputNotRevisable,
    TurnUndeliverable,
)
from managed_agent.control.session.turn_execution import (
    RUNTIME_SILENCE_DEADLINE_S,
    BackgroundTurns,
    run_turn,
    task_name,
)
from managed_agent.core.ids import Seq, SessionId, TurnId, new_session_id, new_turn_id
from managed_agent.core.vocabulary import turn
from managed_agent.session_shim import pod_channel
from managed_agent.session_shim.pod_channel import _SHIM_TIMEOUT

pytestmark = pytest.mark.anyio


@dataclass
class Appended:
    """One event somebody wrote, kept in the order it was written."""

    session_id: SessionId
    type: str
    payload: dict[str, object]


class RecordingLog:
    """An `EventLogAppend` that keeps what it was given."""

    def __init__(self) -> None:
        self.events: list[Appended] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.events.append(Appended(session_id, type_, dict(payload)))
        return Seq(len(self.events))

    def causes(self) -> list[str]:
        return [
            str(event.payload["cause"])
            for event in self.events
            if event.type == turn.TURN_FAILED
        ]


class Slow:
    """A dispatch that takes longer than it is given, and says whether it finished."""

    def __init__(self, takes_s: float) -> None:
        self._takes_s = takes_s
        self.finished = False

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        await asyncio.sleep(self._takes_s)
        self.finished = True


class Raises:
    """A dispatch that fails the way its caller asked it to."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        raise self._error


async def test_a_turn_that_fails_is_closed_by_whatever_ran_it() -> None:
    """The trap: the recording has to travel with the dispatch, or it happens nowhere.

    Nothing is awaiting `run_turn`'s result -- the request that asked for this Turn was
    answered before it started -- so an exception that escapes here is swallowed by
    asyncio and the Turn stays open for ever.
    """
    log = RecordingLog()
    session_id, turn_id = new_session_id(), new_turn_id()

    await run_turn(
        Raises(TurnUndeliverable("the pod is gone")),
        log,
        session_id,
        turn_id,
        "summarise it",
    )

    assert log.causes() == [turn.TurnFailureCause.POD_UNREACHABLE.value]
    assert log.events[0].payload["turn_id"] == str(turn_id)


async def test_a_turn_whose_artifact_collided_is_closed_with_its_own_cause() -> None:
    """The one failure that is not the platform's, and it keeps its published name."""
    log = RecordingLog()

    await run_turn(
        Raises(TurnOutputNotRevisable("report.md")),
        log,
        new_session_id(),
        new_turn_id(),
        "write it",
    )

    assert log.causes() == [turn.TurnFailureCause.OUTPUT_NOT_REVISABLE.value]


async def test_an_undeclared_failure_still_closes_the_turn() -> None:
    """An unexpected error used to become a 500; now it would become silence.

    Inside the request the route did not have to catch this: FastAPI answered 500 and a
    human saw it. A background task has no such reader, so an uncaught error would
    leave the Session wedged with nothing anywhere saying why.
    """
    log = RecordingLog()

    await run_turn(
        Raises(RuntimeError("the database went away")),
        log,
        new_session_id(),
        new_turn_id(),
        "summarise it",
    )

    assert log.causes() == [turn.TurnFailureCause.POD_UNREACHABLE.value]


async def test_a_slow_turn_is_carried_to_its_end_rather_than_given_up_on() -> None:
    """`run_turn` holds no deadline of its own, so a long Turn simply finishes.

    The inverse of what stood here until 2026-08-26, which injected a tiny deadline and
    asserted the dispatch was abandoned when it overran. That total bound was an hour in
    production and came within twenty-two minutes of ending a delegating review that
    was visibly healthy, with its longest phases still ahead and nothing shipping until
    the Turn boundary. An agent run has no natural length, so no wall-clock number here
    can tell a long Turn from a dead one.

    `deadline_s` is gone from the signature rather than defaulted to something large,
    which is what makes this permanent: a later caller cannot reintroduce the bound by
    passing a value, and this case would fail to even call `run_turn` if the parameter
    came back.

    What still ends a Turn is elsewhere and unaffected: `STUCK_IDLE_MS` reads what the
    pod reports about itself, and `RUNTIME_SILENCE_DEADLINE_S` bounds a socket that has
    gone quiet rather than a Turn that is thinking.
    """
    log = RecordingLog()
    slow = Slow(takes_s=0.2)

    await run_turn(slow, log, new_session_id(), new_turn_id(), "write it")

    assert slow.finished is True, (
        "the dispatch was abandoned before it finished, so something in this path "
        "still gives up on a Turn for taking too long"
    )
    assert log.causes() == []


def test_no_bound_on_this_call_can_be_reached_by_a_working_pod() -> None:
    """No inter-byte deadline survives, because silence is not evidence of death.

    Asserted on the wire timeout rather than on behaviour, because behaviour cannot
    reach it: proving the old 120-second gap no longer kills a Turn takes a test that
    is silent for longer than 120 seconds. What is checkable in milliseconds is that no
    bound below the total exists on this call.
    """
    # `httpx.Timeout.read` is optional, and None would mean "no read deadline at all".
    # That is not what was decided: a half-open socket with no deadline holds a
    # connection for the life of the process, so the bound is kept and set high.
    assert _SHIM_TIMEOUT.read is not None, (
        "the wire's read deadline was removed rather than raised; None hangs this call "
        "for the life of the process when a socket goes half-open"
    )
    assert _SHIM_TIMEOUT.read == RUNTIME_SILENCE_DEADLINE_S, (
        "the wire's read deadline is no longer the one constant that names this "
        "question, which is how the retired 120-second inter-byte deadline would come "
        "back -- as a second, smaller number written here directly"
    )
    assert RUNTIME_SILENCE_DEADLINE_S > 10 * 60, (
        "the wire's read deadline is low enough to fire on a working pod. A healthy "
        "Turn has been observed silent on this stream for about seven minutes, and an "
        "agent writing a large file produces exactly the signature of a dead one"
    )


def test_no_per_gap_deadline_is_reintroduced_under_another_name() -> None:
    """The Turn stream's `read` is the total deadline *by name*, not by value today.

    A value check alone passes the moment somebody reintroduces the retired bound as
    `read=_STREAM_IDLE_DEADLINE_S = 600.0` and sets the total to match -- and the
    decision was to retire the per-gap bound, not to make it larger. So this reads the
    source: the `read` keyword of the Turn's timeout must be the total deadline's own
    name, which a new constant cannot satisfy.

    Scoped to the Turn's timeout deliberately. `_FILE_TRANSFER_DEADLINE_S` and
    `_ROLLOUT_TRANSFER_DEADLINE_S` next to it are shorter on purpose and are not this
    defect: they bound transfers of a known quantity of bytes, where silence really
    does mean a stall, and no agent is thinking behind them.
    """
    source = Path(pod_channel.__file__).read_text()
    (assigned,) = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AnnAssign | ast.Assign)
        and "_SHIM_TIMEOUT"
        in ast.unparse(
            node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        )
    ]
    call = assigned.value
    assert isinstance(call, ast.Call)
    read = {word.arg: word.value for word in call.keywords}["read"]

    assert isinstance(read, ast.Name) and read.id == "RUNTIME_SILENCE_DEADLINE_S", (
        "the Turn's read deadline is written as "
        f"{ast.unparse(read)!r} rather than as RUNTIME_SILENCE_DEADLINE_S. A second "
        "constant here is the retired inter-byte deadline coming back under a new "
        "name; the decision was that this call carries no bound a working pod can "
        "reach."
    )


class LeasedPod:
    """A dispatch shaped like the shipped one: it releases its pod in a `finally`.

    Not a stand-in for `HttpPodDispatch` -- `test_a_pod_is_leased_for_one_turn.py`
    grades the real release against a cluster that models pods, and this file does not
    re-grade it. What this reproduces is the *shape* that makes a cancellation
    destructive: `HttpPodDispatch.dispatch` is `try: await self._carry(...) finally:
    await self._release_without_masking_the_turn(...)`, and a `finally` runs on
    `CancelledError` like any other ending. So a cancel delivered into a dispatch
    deletes a live pod mid-Turn.

    What the two cases below actually grade is whether a cancel can still be delivered
    there at all.
    """

    def __init__(self) -> None:
        self.held = True
        self.cancelled = False
        self.completed = False

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        try:
            await asyncio.sleep(0.05)
            self.completed = True
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.held = False


async def test_awaiting_a_dispatch_inside_a_request_is_what_a_disconnect_destroys() -> (
    None
):
    """The defect, pinned, so the case below is not vacuous.

    This is the old arrangement: the handler awaits the dispatch, so the cancellation
    an ASGI server delivers when a client hangs up propagates straight into it. The
    pod is released mid-Turn and no terminal event is ever written.
    """
    pod = LeasedPod()
    log = RecordingLog()

    async def handler() -> None:
        await run_turn(pod, log, new_session_id(), new_turn_id(), "summarise it")

    request = asyncio.create_task(handler())
    await asyncio.sleep(0)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert pod.cancelled is True
    assert pod.held is False, "the pod was released by the client hanging up"
    assert log.events == [], "and nothing was ever written to close the Turn"


async def test_a_turn_whose_submitting_request_is_abandoned_still_ends() -> None:
    """The fix: the request holds no reference to the Turn, so it cannot reach it.

    The same cancellation, delivered to the same handler, at the same moment. The Turn
    is now on a task the handler started and never awaited, so cancelling the handler
    cancels a coroutine that has already returned -- there is nothing left for the
    cancel to propagate into.

    Both halves are asserted because they fail separately: the Turn reaches a terminal
    event (here, completion), and the pod is not released by the abandonment.
    """
    pod = LeasedPod()
    log = RecordingLog()
    turns = BackgroundTurns()
    turn_id = new_turn_id()

    async def handler() -> None:
        turns.start(
            run_turn(pod, log, new_session_id(), turn_id, "summarise it"),
            name=task_name(turn_id),
        )
        # The handler is still running when the cancel arrives -- an ASGI server has
        # the response to serialize and write after the endpoint returns its value.
        # Without this the case would prove only that the handler is fast, and a cancel
        # landing on a task that already finished is not a cancel at all.
        await asyncio.sleep(0.02)

    request = asyncio.create_task(handler())
    await asyncio.sleep(0)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert pod.held is True, "the abandoned request released a pod that was still busy"
    await asyncio.gather(*turns.in_flight)

    assert pod.cancelled is False
    assert pod.completed is True
    assert log.causes() == [], "the Turn completed, so nothing closed it as failed"


async def test_a_turn_abandoned_mid_flight_that_then_fails_is_still_closed() -> None:
    """The disconnect and the failure together, which is the wedging case.

    A Turn nobody is waiting for that then fails is exactly where a `turn.failed` gets
    dropped, and a Session with an open Turn refuses its next Turn and its archive for
    ever. So the terminal event has to arrive after the request is gone.
    """
    log = RecordingLog()
    turns = BackgroundTurns()

    async def handler() -> None:
        turns.start(
            run_turn(
                Raises(TurnUndeliverable("the pod is gone")),
                log,
                new_session_id(),
                new_turn_id(),
                "summarise it",
            ),
            name="map-turn-under-test",
        )
        await asyncio.sleep(0.02)

    request = asyncio.create_task(handler())
    await asyncio.sleep(0)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    await asyncio.gather(*turns.in_flight)

    assert log.causes() == [turn.TurnFailureCause.POD_UNREACHABLE.value]


async def test_a_started_turn_is_tracked_and_named() -> None:
    """Tracked, so shutdown can find it; named, so an operator can see it."""
    turns = BackgroundTurns()
    log = RecordingLog()
    turn_id = new_turn_id()

    turns.start(
        run_turn(Slow(takes_s=0.05), log, new_session_id(), turn_id, "summarise it"),
        name=task_name(turn_id),
    )

    assert [task.get_name() for task in turns.in_flight] == [task_name(turn_id)]
    await asyncio.gather(*turns.in_flight)
    assert turns.in_flight == ()


async def test_shutdown_cancels_every_in_flight_turn_and_waits_for_it() -> None:
    """Cancelled *and* awaited. A cancel only asks; the await is what collects.

    Without the await the process exits with the task mid-Turn, which asyncio reports
    as "Task was destroyed but it is pending" -- the same reason `sweep_loop.sweeping`
    gives for awaiting its own tasks.
    """
    turns = BackgroundTurns()
    never = Slow(takes_s=30.0)

    turns.start(
        run_turn(never, RecordingLog(), new_session_id(), new_turn_id(), "summarise"),
        name="map-turn-under-test",
    )
    running = turns.in_flight
    await turns.aclose()

    assert never.finished is False
    assert [task.done() for task in running] == [True]
    assert turns.in_flight == ()


async def test_a_turn_that_was_never_started_leaves_nothing_to_shut_down() -> None:
    """`aclose` on an idle process is a no-op rather than an error."""
    await BackgroundTurns().aclose()


def test_the_task_name_carries_the_turn_it_is_running() -> None:
    """One spelling, because a test that wrote its own would grade its own string."""
    turn_id = TurnId(uuid4())
    assert task_name(turn_id) == f"map-turn-{turn_id}"
