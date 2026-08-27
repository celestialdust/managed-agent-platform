"""Running an admitted Turn to its end, and recording how it ended.

A Turn used to run inside the HTTP request that submitted it: the route awaited the
dispatch and turned whatever it raised into a refusal. That coupled a Turn's life to a
TCP connection the platform does not control, and it is where two production defects
came from -- a client that hung up destroyed a live pod mid-Turn, because the release
in `HttpPodDispatch.dispatch`'s `finally` runs on `CancelledError` like any other
ending; and the wire deadline that bounded the request had to be short enough for a
person to wait behind, which is how a 120-second gap between bytes came to kill agents
that were writing files.

So a Turn runs here instead, on a task nobody is awaiting, and the route returns the
202 it has always declared.

**The outcome is recorded here because there is nowhere else left.** This is the whole
risk of the move and it is worth stating plainly: the route could answer 202 and keep
its `except` blocks, and every Turn that failed after the response was sent would then
record nothing -- leaving the Session `RUNNING` with an open Turn, which refuses its
next Turn (`admit_turn` requires `state.accepts_a_turn()`), refuses its archive, and
pins the pod sweep. That is worse than holding the request open, so the recording
travels with the dispatch and not with the request.

The same reasoning is why the last `except` here is bare `Exception`. Inside a request
an undeclared error was survivable: FastAPI answered 500 and somebody saw it. On a task
nobody awaits, asyncio swallows it and the Session wedges in silence.

Provenance for the shape of a pod's life: ADR-041 and ADR-042.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any, Final

from managed_agent.control.session.abandoned_turns import RUNTIME_SILENCE_DEADLINE_MS
from managed_agent.control.session.turn_dispatch import (
    TurnDispatch,
    TurnOutputNotRevisable,
    TurnUndeliverable,
)
from managed_agent.core.ids import SessionId, TurnId
from managed_agent.core.ports import EventLogAppend
from managed_agent.core.vocabulary import turn

_LOG = logging.getLogger(__name__)

RUNTIME_SILENCE_DEADLINE_S: Final = RUNTIME_SILENCE_DEADLINE_MS / 1000
"""The wire's bound on a socket that has gone quiet, in seconds. Not a Turn's length.

Read the constant it comes from for why this is no longer the same question. In short:
this used to be `TURN_DEADLINE_S`, the dispatcher's total give-up point, and removing
that is the point of the change -- an agent run has no natural length, and a Turn was
observed thirty-eight minutes into one Turn with this number twenty-two
minutes away, and its longest phases still ahead of it.
"""


def task_name(turn_id: TurnId) -> str:
    """The asyncio task name a running Turn carries.

    A function rather than an f-string written at both ends, because this name is how a
    test observes that a Turn was started and how an operator reading a task dump finds
    which Turn is which. Two spellings would make the observation grade itself.
    """
    return f"map-turn-{turn_id}"


async def run_turn(
    dispatch: TurnDispatch,
    log: EventLogAppend,
    session_id: SessionId,
    turn_id: TurnId,
    prompt: str,
) -> None:
    """Carry one admitted Turn to the pod and close it however it ends.

    Never raises. Every ending is a closed Turn or a completed one, because the caller
    is a bare task and an exception escaping it reaches no reader at all.

    **Nothing here bounds how long the Turn may take, deliberately.** Until 2026-08-26
    this wrapped the dispatch in an hour, and the sweep applied the same hour from
    outside, so a Turn was killed at sixty minutes whatever its pod was doing -- which
    nearly happened, to a delegating run that was visibly healthy thirty-eight minutes
    in with its longest phases still ahead -- and nothing ships until the Turn boundary,
    so the hour would have taken all of it. An agent run has no natural length; a wedged
    pod is detected by what the pod reports about itself (`STUCK_IDLE_MS`), not by a
    clock this side of the wire.

    What that gives up was stated rather than hidden: a Turn stuck in **placement** --
    no pod, and no `turn.started` for the sweep's pod-gone signal to rest on -- had
    nothing ending it, and wanted a bound of its own. `AbandonedTurnSweeper` holds that
    bound now, `PLACEMENT_DEADLINE_MS`, and closes such a Turn `IT_NEVER_GOT_A_POD`. It
    is a different number from a Turn's length for the reason this paragraph gives: an
    agent run has no natural length, but a placement does -- the pod runner gives up
    after its own two timeouts, so past their sum nothing is still placing.

    The sentence above read "has nothing ending it" for a day after the bound was built,
    and that is why it now names the constant rather than describing it. A docstring
    claiming a case is unhandled while it is handled is read as a gap and reported as
    one; `tests/control/test_an_abandoned_turn_is_closed.py::
    test_the_turn_runner_names_the_bound_that_closes_a_stuck_placement` fails if the
    name goes missing again.
    """
    try:
        await dispatch.dispatch(session_id, turn_id, prompt)
    except TurnOutputNotRevisable as collided:
        # The Turn ran. The model answered, its events are in the log, and what failed
        # is storing a file it produced under a path it had already delivered -- so
        # this is the tenant's own doing and the platform has nothing to fix. The path
        # is the whole of the next move and it is the tenant's own string, so it goes
        # in this log; ADR-013 keeps the lane and the store it also names off any
        # tenant surface.
        _LOG.info(
            "turn %s for session %s rewrote a delivered artifact: %s",
            turn_id,
            session_id,
            collided,
        )
        # The path travels in the event, because the refusal it used to travel in no
        # longer exists. This Turn was answered 202 before it ran, so the 409 that
        # carried `detail.path` never happens -- and a Session that produced several
        # files leaves a tenant no other way to tell which one collided, which is the
        # whole of their next move. It is the tenant's own string, so ADR-013 does not
        # withhold it; what that record withholds is the lane and the store the
        # exception's text also names, and neither is copied here.
        await _close(
            log,
            session_id,
            turn_id,
            turn.TurnFailureCause.OUTPUT_NOT_REVISABLE,
            path=collided.path,
        )
    except TurnUndeliverable as undeliverable:
        # The words, not just the code. `session_pods.py` builds this message
        # deliberately -- to tell "this Session may not be resumed" from "that
        # environment is not registered" from "the image will not pull", three
        # different people's problems -- and for two days nothing printed it, so the
        # one thing that knew the cause never told anyone. This log is stderr, which no
        # tenant reads, so nothing is disclosed that the Event Log's cause withholds.
        _LOG.warning(
            "turn %s for session %s is undeliverable (%s): %s",
            turn_id,
            session_id,
            undeliverable.cause.value,
            undeliverable,
        )
        # The cause comes off the exception because only the site that raised it knows
        # which of four situations this was. Reading a fixed cause here is what made
        # `pod_unreachable` mean four things at once.
        await _close(log, session_id, turn_id, undeliverable.cause)
    except TimeoutError:
        # No longer this function's own deadline -- it has none. What reaches here is a
        # timeout from the wire beneath the dispatch, which means the socket produced
        # nothing for `RUNTIME_SILENCE_DEADLINE_S`. Still an ending, and still one the
        # Turn has to be closed for, but it is a transport fact rather than a verdict
        # on how long the agent was allowed to think.
        _LOG.warning(
            "turn %s for session %s produced nothing for %s seconds and was closed",
            turn_id,
            session_id,
            RUNTIME_SILENCE_DEADLINE_S,
        )
        await _close(
            log, session_id, turn_id, turn.TurnFailureCause.TURN_DEADLINE_EXCEEDED
        )
    except Exception:
        # Undeclared, and therefore the one worth the traceback. A `TurnDispatch` is
        # only allowed to raise the two types above; anything else is a defect in this
        # process rather than a fact about the pod, and it still has to close the Turn.
        _LOG.exception(
            "turn %s for session %s failed in a way the dispatch port does not declare",
            turn_id,
            session_id,
        )
        await _close(log, session_id, turn_id, turn.TurnFailureCause.POD_UNREACHABLE)


async def _close(
    log: EventLogAppend,
    session_id: SessionId,
    turn_id: TurnId,
    cause: turn.TurnFailureCause,
    path: str | None = None,
) -> None:
    """Append the `turn.failed` that ends this Turn, or say that it could not be.

    A failure to append is the one thing that leaves the Session wedged, so it is
    logged loudly and swallowed rather than re-raised at a caller that is a bare task.
    `AbandonedTurnSweeper` is what collects a Turn nothing in this process closed.

    `path` is omitted rather than written as null when the cause has no path to carry,
    so a reader can test for the key instead of learning which causes leave it empty.
    """
    payload: dict[str, object] = {
        "turn_id": str(turn_id),
        "cause": cause.value,
        "remedy": turn.REMEDY_FOR[cause],
    }
    if path is not None:
        payload["path"] = path
    try:
        await log.append(session_id, turn.TURN_FAILED, payload)
    except Exception:
        _LOG.exception(
            "turn %s for session %s failed with %s and the closing event could not be "
            "written; the sweep is what will close it",
            turn_id,
            session_id,
            cause.value,
        )


class BackgroundTurns:
    """The tasks carrying Turns that are no longer inside anybody's request.

    Holds a strong reference to every running task, which is not bookkeeping: asyncio
    keeps only a weak one, so a task nothing else holds can be garbage-collected
    mid-Turn. The done-callback is what keeps the set from growing for the life of the
    process.

    Takes no collaborators. Its single job is owning task lifetime, and what a task
    *does* arrives already built as a coroutine -- which is what lets it be a defaulted
    `Platform` field without the two dozen places that build a `Platform` having to
    hand it a dispatch and a log.
    """

    def __init__(self) -> None:
        self._running: set[asyncio.Task[None]] = set()

    def start(self, work: Coroutine[Any, Any, None], *, name: str) -> None:
        """Run this Turn on its own task and return immediately."""
        task = asyncio.create_task(work, name=name)
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    @property
    def in_flight(self) -> tuple[asyncio.Task[None], ...]:
        """The Turns running right now, as a snapshot that survives the set changing."""
        return tuple(self._running)

    async def aclose(self) -> None:
        """Stop every running Turn, and wait until each one has actually stopped.

        Cancelled **and** awaited, for the reason `sweep_loop.sweeping` gives: a cancel
        only requests the stop, so a process that returned without awaiting would exit
        with a task mid-Turn, which asyncio reports as "Task was destroyed but it is
        pending". `CancelledError` is suppressed per task because that is the answer a
        task gives when it honours the cancel.

        A cancelled Turn is left open on purpose. Recording an outcome would mean
        awaiting an append from inside a task that is being cancelled, where every
        further await is cancelled too, and the process is going away regardless.
        `AbandonedTurnSweeper` closes it from the replica that is still serving --
        which is the case that sweep exists for and states in its own first line.
        """
        stopping = self.in_flight
        for task in stopping:
            task.cancel()
        for task in stopping:
            with suppress(asyncio.CancelledError):
                await task
