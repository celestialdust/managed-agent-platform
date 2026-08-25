"""Running the platform's periodic sweeps inside the process that serves requests.

Two sweeps existed here, were tested, and were called by nothing in `src/`. What that
cost is not symmetrical. `POST /v1/webhooks` answered 201, wrote a row, and nothing ever
read it, so every registration was a promise to a tenant that no code kept. The pod
sweep's absence was quieter, because `archive_session` hands a pod back on the explicit
path -- what was lost was the backstop for the Sessions no archive call is coming for.

**In-process rather than a CronJob**, because the control plane already holds every
dependency both sweeps need: the engine, the cluster RBAC to delete a pod, the IRSA
identity the vault read is authorized by, and the vault client itself. A CronJob would
be a second image entrypoint, a second Role and a second manifest around the same two
calls, and it would still be this code.

**Started from the app's lifespan and from nowhere else.** A task created at import time
runs in whatever process imports the module -- a test collector, a linter, `--help` --
and one created inside a route handler is created once per request or never at all,
depending on traffic. The lifespan runs once per serving process, which is also the only
place with an exit hook to cancel the task from, so "is it running" and "was it stopped"
have one answer each.

**A tick that raises does not end the loop.** A sweeper that stops on its first
exception is worse than one that never started: the first few ticks make the wiring look
right, and the symptom arrives weeks later as deliveries that stopped for no reason
anybody changed. Every exception is logged with its traceback and the next tick runs on
schedule. `asyncio.CancelledError` is deliberately not caught -- it is a `BaseException`
rather than an `Exception`, and catching it is how a task refuses to shut down.

**Two replicas run this**, and whether that is safe is a property of each sweep rather
than of this loop. So it is carried per sweep, in `Sweep.lease`, where a new sweep has
to answer the question instead of inheriting somebody else's answer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Final, Protocol

_LOG = logging.getLogger(__name__)

SWEEP_INTERVAL_ENV_VAR: Final = "MAP_SWEEP_INTERVAL_S"
"""How long to leave between ticks, in whole seconds, with no default.

`MAP_`-prefixed for the reason every variable the composition root reads is: a generic
name is one a base image or a sidecar could set for its own reasons, which this platform
would then silently adopt as the rate at which it talks to the database and the cluster.

**No default, and that is the property worth being stubborn about.** A default would
read as configured from every angle -- the manifest would say nothing, the process would
start, the sweeps would run -- while the number that decides how long a tenant waits for
a callback would be whoever-wrote-this's guess, changeable only by a rebuild. That is
the same argument the Session-token lifetime makes about a security parameter, and a
delivery latency is the operator's to weigh for the same reason: they are the one who
has told a tenant what to expect.

**What the number trades.** Smaller: a callback goes out sooner and an abandoned pod is
reclaimed sooner, at the cost of one cross-Session query and one cluster read per tick
per replica. Larger: less load and a longer wait -- and above `MAX_WINDOW_MS` in
`webhook_dispatcher.py`, five minutes, the delivery tail stops being able to catch up
after an outage on any single tick, because one pass advances its watermark by at most
that span. Under that ceiling the choice is latency; over it, it is a backlog that only
becomes visible after downtime.
"""


def sweep_interval_from_env(source: Mapping[str, str] | None = None) -> int:
    """How many seconds to leave between ticks, refusing every value that is not one.

    Raises `KeyError` naming the variable when it is absent and `ValueError` naming it
    when it is present and unusable. Both reach an operator the way a missing Secret key
    does -- the process does not start -- rather than as a cadence nobody chose.

    Zero and negative are refused here instead of being left to `asyncio.sleep`. Zero is
    a loop that yields to the event loop and immediately sweeps again, which is a
    process holding the database and the cluster API busy for as long as it is up;
    a negative sleep returns instantly and is the same thing spelled differently.
    """
    raw = (os.environ if source is None else source)[SWEEP_INTERVAL_ENV_VAR]
    try:
        seconds = int(raw)
    except ValueError as unparseable:
        raise ValueError(
            f"{SWEEP_INTERVAL_ENV_VAR} is {raw!r}, which is not a whole number of "
            "seconds"
        ) from unparseable
    if seconds <= 0:
        raise ValueError(
            f"{SWEEP_INTERVAL_ENV_VAR} is {seconds}, so every sweep would start again "
            "the instant it finished and would hold the database and the cluster API "
            "busy for the life of the process"
        )
    return seconds


class SweepLease(Protocol):
    """Permission to be the one replica running a named sweep right now."""

    def held(self, name: str) -> AbstractAsyncContextManager[bool]:
        """Enter as True for the caller holding the lease, False for every other.

        A context manager rather than an acquire and a release, because what has to be
        impossible is holding the lease after the tick has ended: a pair of calls leaks
        it on any raise between them, and a leaked lease stops *every* replica from
        sweeping until something notices.

        False is not an error and must not raise. Losing the lease is the ordinary state
        of the replica that did not win, once per tick, for the life of the process.
        """
        ...


@dataclass(frozen=True, slots=True)
class Sweep:
    """One periodic sweep: what to run, what to call it, and who may run it at once."""

    name: str
    """Names the asyncio task and every log line about this sweep.

    A name and not an index, because it is what an operator reads in a traceback and
    what a test looks for when it asks whether this sweep was started and whether it was
    stopped.
    """

    run: Callable[[], Awaitable[object]]
    """One pass, taking nothing.

    Both sweeps this wires already decide their own scope from what they read, and
    neither is resumable, so a pass needs no arguments and returns nothing this loop can
    act on -- the return value is discarded. Typed as a plain callable rather than as a
    protocol both sweeps implement, because they do not: one reads a clock for itself
    and the other is handed the instant, and inventing a shared method name would mean
    editing two modules to satisfy a third.
    """

    lease: SweepLease | None
    """A lease when two replicas running this pass at once is a defect; None when it is
    not.

    Required rather than defaulted, in either direction. Defaulted to None, adding a
    sweep would inherit "concurrency is fine" by omission -- which is the assumption
    this field exists to stop anybody making silently. Defaulted to a lease, a sweep
    that is genuinely idempotent would pay a round trip and a held connection per tick
    to serialize something that did not need serializing, and the reader of the wiring
    could no longer tell which sweeps had actually been thought about.
    """


def task_name(sweep: str) -> str:
    """The asyncio task name a running sweep carries.

    A function rather than an f-string written at both ends, because this name is how a
    test observes that a sweep was started and that it was cancelled -- and two
    spellings of it would make a test that observes neither of those things pass.
    """
    return f"map-sweep-{sweep}"


@asynccontextmanager
async def sweeping(sweeps: Sequence[Sweep], every_s: float) -> AsyncIterator[None]:
    """Run every sweep on its own task for as long as the body runs, then stop them.

    One task per sweep rather than one task walking the list, so a sweep that is slow --
    a delivery batch against a receiver that is timing out -- delays only itself. A
    single task would make the pod sweep's cadence a function of how badly some tenant's
    endpoint is behaving.

    On exit every task is cancelled and then **awaited**. The await is the half that is
    easy to leave out and is not optional: a cancel only requests the stop, so a
    lifespan that returned without awaiting would let the process exit with the task
    mid-sweep, which asyncio reports as "Task was destroyed but it is pending" and which
    leaves a sweep half-done. `CancelledError` is suppressed per task because that is
    the answer a task gives when it honours the cancel; anything else it raises
    propagates, because a sweep failing on the way down is still a fact worth surfacing.
    """
    tasks = [
        asyncio.create_task(_ticking(sweep, every_s), name=task_name(sweep.name))
        for sweep in sweeps
    ]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task


async def _ticking(sweep: Sweep, every_s: float) -> None:
    """Run one sweep for ever on the interval, surviving whatever a pass raises.

    Sweeps first and sleeps afterwards, so a process that has just started serving has
    already swept once. A first tick one interval away would make every restart a gap in
    coverage as long as the interval, which on a rolling restart is when the backlog is
    largest.
    """
    while True:
        try:
            await _one_tick(sweep)
        except Exception:
            # Logged with the traceback and not re-raised: the next tick is worth more
            # than this one's failure. Never bare-`pass`ed either -- a sweep failing
            # every tick for a week has to be visible somewhere, and this line is the
            # only place it can be.
            _LOG.exception("the %s sweep raised; the next tick runs anyway", sweep.name)
        await asyncio.sleep(every_s)


async def _one_tick(sweep: Sweep) -> None:
    """One pass, under the lease when this sweep declares one.

    A sweep that loses its lease does nothing this tick and is not an error. It does not
    wait for the lease either: a blocking acquire would queue a second pass behind the
    first and run it over a window the winner had already delivered, which is the
    duplicate the lease exists to prevent, merely later.
    """
    if sweep.lease is None:
        await sweep.run()
        return
    async with sweep.lease.held(sweep.name) as mine:
        if mine:
            await sweep.run()
