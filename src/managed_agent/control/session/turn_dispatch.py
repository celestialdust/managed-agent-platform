"""The port an admitted Turn is carried to the Session's pod over, and its two failures.

Three things live here and nothing else: `TurnDispatch`, the port a route holds; the two
exceptions an implementation of it is allowed to fail with; and `NoPodTransport`, which
is what a deploy configured with no `PodRunner` gets.

**The implementation is not here.** `session_shim/pod_channel.py` holds
`HttpPodDispatch`, which locates the Session's pod, posts the Turn to the shim process
running inside it and appends the events the shim streams back. It lives beside the shim
it speaks to because the two are the ends of one wire, and a wire whose ends are
specified in two packages is one they can come to disagree about. What stays here is
only what a *caller* of the port needs, which is what lets a route import the vocabulary
of a dispatch without pulling the pod transport in behind it.

`NoPodTransport` refuses every Turn, and it is what `composition.build` wires when it is
handed no `PodRunner` -- a process with no cluster to place a Session's pod in, and so
with nothing for a Turn to run on. Refusing is the truthful answer there rather than a
placeholder standing in for something nearly finished, and it is the fail-safe one: a
dispatch that quietly succeeded would leave the Event Log saying a Turn was asked for
and running while nothing ran it. It is wired as a `TurnDispatch` rather than left as a
`None` field so that the absence is a refusal a tenant is told about --
`Platform.turn_dispatch` has no default for the same reason, since a deploy with no
dispatcher would otherwise look configured.

A refusal here is never silent. The Turn is already recorded as submitted by the time
dispatch is attempted, so the caller closes it with a `turn.failed` event naming
`pod_unreachable` -- the log then says a Turn was asked for and did not run, which is
the true reading of what happened.
"""

from typing import Protocol

from managed_agent.core.ids import SessionId, TurnId
from managed_agent.core.vocabulary import turn


class TurnUndeliverable(Exception):
    """The Turn could not be carried to a runtime, and `cause` says which way.

    Raised by a `TurnDispatch` implementation and by nothing else. Every other failure
    an implementation meets is its own to translate; a caller of the port never sees a
    transport error, which is what keeps a runtime or cluster message off the tenant
    surface (ADR-013).

    **The cause travels on the exception because only the raising site knows it.** One
    type covers a Session that may not be placed, a pod that never came up, a shim
    nothing answers on, and a shim that answered and declined -- and by the time this
    reaches the caller, the only thing left that could tell them apart is the message
    text, which is written for an operator and must not be parsed. A caller that
    guessed would guess the same thing every time, which is how one name came to cover
    four situations.

    Defaulted to `POD_UNREACHABLE`, which is what every site reported before the causes
    were split. Seventeen sites raise this type and only the dispatch path's carry a
    cause of their own, so the default is the old behaviour rather than a placeholder:
    an unconverted site reports what it always did instead of failing to construct.
    """

    def __init__(
        self,
        message: str,
        cause: turn.TurnFailureCause = turn.TurnFailureCause.POD_UNREACHABLE,
    ) -> None:
        super().__init__(message)
        self.cause = cause


class TurnOutputNotRevisable(Exception):
    """The Turn ran, and the agent rewrote an artifact it had already delivered.

    The port's second failure, and the only one that is not the platform's fault. It is
    raised after the Turn's events are appended -- the model answered, and the step that
    failed is storing what it produced -- so a caller that treats it as "the Turn did
    not run" is wrong about what happened as well as about what to do next.

    Declared here beside `TurnUndeliverable` rather than where it is first raised,
    because the two together are what a caller of `TurnDispatch` must handle, and a
    port whose failure vocabulary is spread across two packages is one a caller has to
    go looking for. `control/files/output_shipout.py` raises its own
    `OutputNotRevisable` at the collision and `session_shim/pod_channel.py` translates
    it here, which keeps the module that owns the bounds free of the pod wire.

    Carries the path because the refusal a tenant reads names it: a Session that
    produced several files leaves no way to tell which one collided, and which one it
    was is the whole of the next move.
    """

    def __init__(self, path: str) -> None:
        super().__init__(
            f"the artifact at {path!r} was already delivered under that path and "
            "cannot be revised"
        )
        self.path = path


class TurnDispatch(Protocol):
    """Carries an admitted Turn to the pod bound to the Session.

    Raises `TurnUndeliverable` when the pod cannot be reached, and
    `TurnOutputNotRevisable` when the Turn ran but what it produced could not be stored
    under the path the agent chose. Returns when the Turn's events are recorded; there
    is nothing to return, because what the Turn produced is in the Event Log and reading
    it is a separate call.
    """

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None: ...


class NoPodTransport:
    """What a deploy with no `PodRunner` gets: it refuses every Turn.

    A process built without one has no cluster to place a Session's pod in, so it has
    nowhere to carry a Turn and no way to find out where one went. Refusing is the only
    truthful answer available to it, and it is the one that fails at the Turn rather
    than at start-up -- `composition.build` says why that ordering is deliberate.

    Not on the wired path of a deploy that does have a runner. That process is built
    with `HttpPodDispatch` and this class is nowhere in it, which is what makes a
    refusal from here diagnostic: it means the process was configured without a runner,
    never that the transport failed.

    Wired as a `TurnDispatch` rather than left as a `None` field so that the absence is
    a refusal a tenant is told about -- `Platform.turn_dispatch` has no default for the
    same reason, since a deploy with no dispatcher would otherwise look configured.
    """

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        raise TurnUndeliverable(
            "this process has no transport to a Session's pod, so no Turn can be "
            f"delivered; session {session_id} turn {turn_id} did not run",
            turn.TurnFailureCause.NO_RUNTIME_CONFIGURED,
        )
