"""Carrying an admitted Turn to the pod the Session runs in.

**Half of this is real and half of it refuses, and which is which is the point of this
docstring.** Locating the Session's pod is real: `PodTurnDispatch` asks `placement` for
the binding, and a pod that is not running is refused with `TurnUndeliverable` rather
than dispatched into. Running the Turn is real: given a connection to the Agent Runtime
it drives `run_turn`, which appends every event the Turn produced in arrival order.

What does not exist is the step between those two. The control plane runs outside the
Session's pod and the Agent Runtime listens on a unix socket *inside* it (ADR-001), so
there is no path from here to that socket, and no listener in the pod that would accept
a Turn on the control plane's behalf. That step is `PodChannel`, and the implementation
this process is wired with -- `NoPodTransport` -- refuses every Turn. It is not a stub
standing in for something almost finished: it is the honest answer, and it is the
fail-safe one, because a dispatch that quietly succeeded would leave the Event Log
saying a Turn was asked for and running while nothing ran it.

A refusal here is never silent. The Turn is already recorded as submitted by the time
dispatch is attempted, so the caller closes it with a `turn.failed` event naming
`pod_unreachable` -- the log then says a Turn was asked for and did not run, which is
the true reading of what happened.
"""

from dataclasses import dataclass
from typing import Protocol

from managed_agent.control.session.placement import Placement, PodBinding, PodPhase
from managed_agent.core.ids import SessionId, TurnId
from managed_agent.core.ports import EventLogAppend
from managed_agent.session_shim.turn_runner import (
    RuntimeConnection,
    TurnCompleted,
    run_turn,
)


class TurnUndeliverable(Exception):
    """The pod bound to this Session could not be reached.

    Raised by a `TurnDispatch` implementation and by nothing else. Every other failure
    an implementation meets is its own to translate; a caller of the port never sees a
    transport error, which is what keeps a runtime or cluster message off the tenant
    surface (ADR-013).
    """


class TurnDispatch(Protocol):
    """Carries an admitted Turn to the pod bound to the Session.

    Raises `TurnUndeliverable` when the pod cannot be reached. Returns when the Turn's
    events are recorded; there is nothing to return, because what the Turn produced is
    in the Event Log and reading it is a separate call.
    """

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PodRuntime:
    """A way to speak to the Agent Runtime in one pod, and the thread to speak about.

    The two travel together because neither is usable alone. A Turn is started *on a
    thread* and the thread id is minted inside the pod by `start_thread`; the control
    plane holds no thread id and must not (ADR-007), so whatever opens the connection
    is also the only thing that can say which thread this Session's Turns run on.
    """

    connection: RuntimeConnection
    thread_id: str


class PodChannel(Protocol):
    """Opens a way to speak to the Agent Runtime inside a placed pod.

    Raises `TurnUndeliverable` when it cannot. This is the seam the transport is
    missing at: see this module's docstring for why nothing in this tree implements it
    against a real pod.
    """

    async def open(self, binding: PodBinding) -> PodRuntime: ...


class PodTurnDispatch:
    """Locates the Session's pod and runs the Turn against the runtime inside it.

    The phase is checked before the channel is opened, so a Session whose pod is
    absent, starting or gone is refused by the cluster's own answer rather than by a
    connection timing out. That distinction matters to the caller only in how long it
    waits, which is reason enough: a tenant waiting out a socket timeout for a pod the
    cluster already knows is gone is a refusal delivered slowly.
    """

    def __init__(
        self,
        placement: Placement,
        channel: PodChannel,
        log: EventLogAppend,
        on_completed: TurnCompleted,
    ) -> None:
        self._placement = placement
        self._channel = channel
        self._log = log
        self._on_completed = on_completed

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        """Run one admitted Turn, or raise `TurnUndeliverable` having run nothing.

        Returns only once the Turn has finished and its events are appended. That is
        the shape the platform can honestly offer while the Turn runs on this side of
        the channel; a transport that handed the Turn to a process inside the pod would
        return as soon as the pod had accepted it, and the caller's contract -- a Turn
        is recorded before it is dispatched, and a dispatch that fails closes it --
        would be unchanged.
        """
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            raise TurnUndeliverable(
                f"the pod for session {session_id} is {binding.phase.value}"
            )
        runtime = await self._channel.open(binding)
        await run_turn(
            session_id,
            turn_id,
            runtime.thread_id,
            prompt,
            runtime.connection,
            self._log,
            self._on_completed,
        )


class NoPodTransport:
    """The dispatch this process is wired with: it refuses every Turn.

    Nothing in this tree can reach the Agent Runtime inside another pod, so this is
    what the composition root has to offer, and refusing is the only truthful thing it
    can do. Wired as a `TurnDispatch` rather than left as a `None` field so that the
    absence is a refusal a tenant is told about -- `Platform.turn_dispatch` has no
    default for the same reason, since a deploy with no dispatcher would otherwise look
    configured.
    """

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        raise TurnUndeliverable(
            "this process has no transport to a Session's pod, so no Turn can be "
            f"delivered; session {session_id} turn {turn_id} did not run"
        )
