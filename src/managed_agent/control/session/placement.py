"""Where a Session's pod runs, and which pod a Session is bound to.

The binding is computed, never stored. A pod's name is a pure function of the Session's
identifier, and whether that pod is running is a question only the cluster can answer —
so no table holds a mutable pod reference and no second source can disagree about where
a Session is. Recovering the binding after a control-plane restart is arithmetic rather
than a lookup, and two controllers racing to place the same Session converge on one pod
instead of allocating two.

Placement policy belongs to the cluster scheduler. This module asks for a pod by name
and reports what came back; it never chooses a node, and a PodRunner that did would be
answering a question this platform does not ask.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.core.ids import SessionId
from managed_agent.core.ports import Clock

_LOG = logging.getLogger(__name__)

_POD_NAME_PREFIX: Final = "map-session-"


class PodPhase(StrEnum):
    """What the cluster says about a pod, reduced to the cases placement acts on."""

    ABSENT = "absent"
    STARTING = "starting"
    RUNNING = "running"
    GONE = "gone"


class PodNotStarted(Exception):
    """The pod for this Session could not be brought up, and here is why.

    Raised by a `PodRunner` implementation and by nothing else, carrying the cluster's
    own account of the failure -- an image that will not pull, a pod nothing can
    schedule, a name that does not match the configuration it was handed. A caller needs
    the reason and not only the fact: every one of those is fixed somewhere different,
    and a bare "it did not start" sends whoever is on call to read pod events by hand.

    Declared here rather than in the adapter that raises it, so a caller of this port
    can catch the port's failure by name without importing a cluster client -- which
    `control/` may not do at all.

    Deliberately not `TurnUndeliverable`. That names a Turn that could not be carried to
    a pod; this names a pod that never came up, which is a Session-lifecycle failure.
    Nothing calls `Placement.place` in this tree yet, so no route translates this into a
    published `ErrorCode` today. When one does: the unschedulable case is `OVERLOADED`
    and every other case is `INTERNAL`, both already in the closed set. Stated here
    because the mapping is a decision, and unbuilt because there is no caller to build
    it into -- not because it is hard.
    """


@dataclass(frozen=True, slots=True)
class PodBinding:
    """Which pod a Session is on right now.

    Carries no Agent Runtime identifier — not a thread id, not the read-only session tag
    the Agent Runtime emits. Those exist only inside the pod, and a binding holding one
    would put a runtime identifier a single serialization away from a tenant (ADR-007).
    """

    session_id: SessionId
    pod_name: str
    phase: PodPhase


@dataclass(frozen=True, slots=True)
class PlacedPod:
    """One Session pod the cluster is holding, as a sweep needs to see it.

    Carries the Session rather than the pod's name because the name is a pure function
    of the Session (`pod_name_for`) and the reverse is a parse that can fail -- so a
    lister that hands back a name would push that parse onto every consumer, and a
    consumer that got it wrong would be reasoning about the wrong Session. An
    implementation returns this only for a pod whose own label and name agree, which is
    the same consistency check `phase_of` applies; anything else in the namespace is not
    this platform's to describe.

    `created_at_ms` is the pod's, not the Session's. A sweep that deletes needs an age
    it can trust without reading any store: a Session's log can be behind, a pod's
    creation timestamp is set by the API server at admission and cannot be.
    """

    session_id: SessionId
    phase: PodPhase
    created_at_ms: int


@runtime_checkable
class PlacedPods(Protocol):
    """Every Session pod the cluster is currently holding.

    `runtime_checkable` so the composition root can narrow the cluster client it was
    handed, which is typed as `PodRunner` because that is all the Turn path needs. The
    real client has all four methods; a `PodRunner` double has three, and requiring the
    fourth on the parameter would make every three-method double in this suite a type
    error while proving nothing about the client a deployment actually wires. That check
    sees method names and not signatures, which is the same shallowness `core.ports`
    documents.

    Declared apart from `PodRunner` rather than added to it, and the split is not
    tidiness. `PodRunner` is satisfied by half a dozen doubles across this suite, none
    of which enumerates anything; folding this in would make each of them grow a method
    it never calls, and a double returning `[]` from a method a sweep reads as "nothing
    to reclaim" is a test that passes by saying the cluster is empty. A cluster client
    satisfies both by having all four methods, so the composition root wires one object.

    Asked of the cluster and never of a table, for the reason `placement.py` keeps no
    binding at all: what pods exist is the cluster's own answer, and a record of it is a
    second source free to name a pod that is gone or miss one that is not.
    """

    async def placed_pods(self) -> Sequence[PlacedPod]:
        """Every pod in this namespace that claims to be some Session's.

        Returns the whole namespace in one answer with no cap and no cursor. That is
        honest at this platform's scale rather than lazy: the ceiling is the node count
        times the per-node pod limit -- measured at seventeen on a `t3.medium`, against
        an autoscaler capped at four nodes -- so the collection is bounded by the
        cluster's own shape at somewhere under fifty rows, and a caller cannot ask for
        more pods than the cluster can hold.

        A pod whose label does not name a Session, or names one whose derived name is
        not this pod's, is absent from the result rather than described. Deciding that
        here rather than at every caller is what keeps a sweep from having to know the
        naming rule in order to be safe.
        """
        ...


class PodRunner(Protocol):
    """The cluster, as much of it as placement needs.

    Declared beside its one consumer rather than in a shared ports module: this module
    says what it requires and a concrete cluster client satisfies it, wired at the
    composition root like every other adapter.
    """

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        """Start the named pod if it is absent, and report its phase either way.

        Idempotent by contract: called for a pod already running it starts nothing and
        is not an error, which is what lets a resume and a retry take the same path.

        An implementation may wait, bounded, for the pod to finish starting before it
        answers -- a phase read the instant after a create is always the starting one,
        and this port offers no other way to wait. Raises `PodNotStarted` when the pod
        will not come up, rather than reporting a phase that invites the caller to poll
        something that is never going to change.
        """
        ...

    async def phase_of(self, pod_name: str) -> PodPhase:
        """Report the named pod's phase. ABSENT for a pod that was never created."""
        ...

    async def remove(self, pod_name: str) -> None:
        """Delete the named pod. Absent is success, so a repeated stop is not a
        failure."""
        ...


def pod_name_for(session_id: SessionId) -> str:
    """The one pod name a Session ever has.

    An RFC 1123 label by construction: the prefix plus a hyphenated uuid is 48
    characters, inside the 63-character limit, and holds only lowercase letters, digits
    and hyphens.
    """
    return f"{_POD_NAME_PREFIX}{session_id}"


@dataclass(frozen=True, slots=True)
class NodeHeadroom:
    """How many nodes can take a pod right now, and how many the cluster may ever add.

    Two numbers rather than one because they answer different questions and disagreeing
    is exactly what they are for. `schedulable` is what the cluster is holding open now;
    `ceiling` is what the autoscaler will grow it to unattended. A refusal to place a
    pod means something different depending on whether those two are equal.

    **Both are optional, and `None` never means zero.** Each is a separate call to the
    cluster and either can be refused on its own -- the node count is a cluster-scoped
    read and the ceiling is a namespaced one, so they are granted by different authority
    and can fail independently. Zero schedulable nodes is a cluster in serious trouble
    and "we were not allowed to count" is a deployment misconfiguration; publishing the
    second as the first would raise an alarm about the wrong thing, at the moment
    somebody is reading this to find out what is wrong.

    An implementation that cannot answer either one is expected to say so in the log,
    naming which read was refused, because the wire cannot: `None` reaches a reader
    identically whether the cluster refused, no autoscaler is deployed, or the runner
    makes no such call at all.
    """

    schedulable: int | None
    ceiling: int | None


@runtime_checkable
class ClusterHeadroom(Protocol):
    """The cluster's remaining room, asked of the cluster and of nothing else.

    Declared apart from `PodRunner` and narrowed with `isinstance`, for the reason
    `PlacedPods` gives one screen up and not for tidiness: `PodRunner` is satisfied by
    half a dozen doubles across this suite, and folding a node read into it would make
    every one of them a type error while proving nothing about the client a deployment
    wires. A real cluster client satisfies all three protocols by having the methods, so
    the composition root still wires one object.

    **Nothing in this repository implements this yet**, and a caller that cannot narrow
    to it reports no node numbers rather than guessing at them. The implementation is
    two read-only calls on the Kubernetes API -- list the nodes, and read the
    autoscaler Deployment's arguments -- and it belongs in the adapter beside the other
    four cluster reads.
    """

    async def node_headroom(self) -> NodeHeadroom:
        """Count the nodes that can take a pod, and read the ceiling that binds them.

        One method for both, rather than one each, because an operator compares them:
        two calls could straddle a scale-up and hand back a pair that was never true at
        any single instant, which is the one way this answer can mislead while every
        number in it is individually correct.

        The ceiling reported must be the one the *running* autoscaler is using, read
        from what that process was actually started with. A copy of the number held
        anywhere else -- a manifest, an environment variable, a constant in this tree --
        is a second spelling free to drift from the flag it claims to publish, and a
        ceiling that drifts is worse than no ceiling, because it reads as measured.
        """
        ...


@dataclass(frozen=True, slots=True)
class PlacementBacklog:
    """One backlog reading taken from the shared log rather than from a process.

    Three numbers and not `PlacementStats`, because this answers only the part that was
    wrong. The cluster half of that type is the same from every replica already.
    """

    turns_awaiting: int
    sessions_placing: int
    oldest_awaiting_at_ms: int | None


@runtime_checkable
class PlacementBacklogReader(Protocol):
    """The placement backlog as the whole platform's, read out of the event log.

    **Why this exists, measured rather than reasoned.** The count it replaces was per-
    process. On 2026-08-24 against `map-dev`, with six Sessions placing at once and two
    control-plane replicas, one replica reported `turns_awaiting_placement: 6` and the
    other reported `0` -- for the same six placements. A client's connection is sticky,
    so an operator's whole workload lands on one replica and an operator polling the
    other sees an idle platform. `PlacementWaits` states the cost as "roughly half the
    fleet's queue"; it is worse than half, because the error is not random.

    **Nothing new is written for this.** `PlacementWaits` argues against a store-backed
    count on the grounds that it would mean "a row per wait on the hot path plus a sweep
    for the rows a crashed process left behind". Both halves are already paid:
    `session.placing` is appended per waiting Turn, and what ends the wait is appended
    too -- the Turn's own `turn.started` or `turn.failed`. The terminal event IS the
    sweep, so there is no row to clean up and no new write to make.

    `within_s` is the caller's, and it is not a tuning knob. A `session.placing` with no
    terminal event and an age past the placement timeout is not a Turn that is waiting
    -- it is a Turn whose process died mid-placement, and its connection died with it.
    Counting those forever would make the gauge climb monotonically and never come back,
    which is the one failure that makes a depth gauge worth less than no gauge.

    `runtime_checkable`, following `PlacedPods` and `ClusterHeadroom` above: the
    composition root wires one object and `capacity()` narrows to what that object can
    actually answer, so a process assembled without a store reports the process-local
    count instead of failing. It compares method names and not signatures, which is what
    is asked here -- whether the collaborator can answer this at all.
    """

    async def placement_backlog(self, within_s: int, /) -> PlacementBacklog:
        """Turns still waiting for a pod, the Sessions they belong to, and the oldest.

        A Turn counts when the log holds a `session.placing` for it, no later
        `turn.started` or `turn.failed` on the same Session names the same Turn, and the
        placing event is younger than `within_s`.

        `oldest_awaiting_at_ms` is `None` exactly when the count is zero. It is the
        oldest live wait's append time in epoch milliseconds, which is what lets an
        operator tell a queue that is draining from one that is stuck at the same
        depth.
        """
        ...


@dataclass(frozen=True, slots=True)
class PlacementStats:
    """What the placement pipeline looks like right now, as an operator asks it.

    The first three are this process's own in-flight work and the last three are the
    cluster's. That split is worth stating because it is also the accuracy boundary: the
    cluster numbers are the same from whichever replica answers, and the queue numbers
    are only this replica's share of the waiting. `control/api/routes/capacity.py`
    carries what that costs a reader.

    `oldest_awaiting_placement_at_ms` is epoch milliseconds and not a formatted
    instant. Rendering belongs to the surface that publishes it, so the number stays
    comparable to every other clock read in this tree and only one module has to know
    what the wire format is.

    Every optional field is optional because the answer can be genuinely absent rather
    than zero: no Turn is waiting, or this deployment cannot read its cluster's nodes.
    Zero would be a measurement in both cases, and a false one.
    """

    turns_awaiting_placement: int
    sessions_placing: int
    oldest_awaiting_placement_at_ms: int | None
    session_pods_running: int
    nodes_schedulable: int | None
    node_ceiling: int | None


@runtime_checkable
class CapacityReport(Protocol):
    """Whatever can report the placement pipeline's shape, narrowed to that one read.

    **Exists so a route can reach the numbers without `Platform` growing a field.**
    `Placement` is already assembled into the serving process as
    `Platform.session_pod_release`, whose declared type names only `release` -- on
    purpose, so that no end-of-life path can reach `place` and revive a Session by
    accident. Narrowing that field to this protocol re-widens it by exactly one
    read-only method and nothing else: there is no way to place, remove or bind a pod
    through this.

    A process wired without a cluster holds `NoSessionPods` there, which does not
    satisfy this, and the capacity route reports the empty fleet rather than refusing.
    That is
    the accurate answer rather than a fallback, and it rests on an invariant
    `composition.build` already states: the only thing that creates a Session pod is
    built inside the same branch that wires the real release, so a process holding the
    default is one in which no Session pod can exist.
    """

    async def capacity(self) -> PlacementStats:
        """Every number the capacity surface publishes, read as one answer."""
        ...


class PlacementWait:
    """One Turn's stay in the placement queue, and how long it has lasted.

    Handed out by `Placement.awaiting` so the caller that measured a wait can report it
    after the wait is over. Reading the elapsed time off the register instead would not
    work: the register drops a wait the moment it ends, which is what keeps it bounded,
    so by the time a caller wants the number the entry it would look up is gone.

    **Carries no Turn identifier, and that absence is what keeps the seam narrow.**
    Nothing here is ever looked up by Turn -- the queue counts entries, groups them by
    Session, and takes the earliest start, and the one caller that wants a single wait's
    duration is holding the object already. So `SessionPods.ensure_for` did not have to
    grow a Turn argument to be counted, and the Turn id stays where it is genuinely
    needed: in the payload of the event the dispatch appends.

    `elapsed_ms` answers while the wait is open too, and that is deliberate rather than
    a side effect -- it is what lets the fleet aggregate say how long the unluckiest
    request has been waiting, for a request that has not finished waiting.
    """

    __slots__ = ("_clock", "_finished_ms", "session_id", "started_ms")

    def __init__(self, session_id: SessionId, started_ms: int, clock: Clock) -> None:
        self.session_id = session_id
        self.started_ms = started_ms
        self._clock = clock
        self._finished_ms: int | None = None

    def _finish(self) -> None:
        """Stop the clock on this wait. Called once, by the register, at exit."""
        if self._finished_ms is None:
            self._finished_ms = self._clock.now_epoch_ms()

    @property
    def elapsed_ms(self) -> int:
        """How long this wait lasted, or has lasted so far if it is still open.

        Floored at zero rather than returned as written. The two reads are of a wall
        clock, which can step backwards -- NTP correction, a suspended node resuming --
        and a negative duration published as a wait would be read as a clock this
        platform trusts rather than as one it does not control.
        """
        if self._finished_ms is not None:
            return max(0, self._finished_ms - self.started_ms)
        return max(0, self._clock.now_epoch_ms() - self.started_ms)


class PlacementWaits:
    """Which Turns are waiting for a pod in this process, and since when.

    **Mutable state, held deliberately, and the alternatives are worse.** What is
    waiting right now is not a fact any store holds: a Turn between "found no pod" and
    "has one" has appended nothing since its submission, and the thing that knows it is
    waiting is the coroutine that is blocked. Writing it down would mean a row per wait
    on the hot path plus a sweep for the rows a crashed process left behind, to answer a
    question that becomes false again in seconds.

    **What that costs, stated rather than left for a reader to discover.** This counts
    one process's waits. The serving Deployment runs two replicas, so an operator
    reading one of them sees roughly half the fleet's queue, and a restart resets the
    count to zero while the Turns that were waiting are still waiting -- they are on the
    connections that died with it, so they are no longer waiting on anything, which
    makes the reset correct rather than merely convenient.

    Bounded by construction: an entry exists only while a coroutine is inside the
    `awaiting` block, and the block removes it on the way out however it leaves --
    normally, by exception, or by cancellation. There is no path that adds without
    removing, which is what makes an unbounded dictionary safe here.

    **Keyed by a token this register mints, and not by Session or by Turn.** Two Turns
    of one Session can wait at once -- admission refuses a Session that will not take a
    Turn, not a Session that already has one open, so both reach the dispatch and both
    find no pod -- which rules out a Session key, since the second wait would overwrite
    the first and the depth would under-report exactly when it mattered. A Turn key
    would work and costs more than it is worth: it would put a Turn argument on
    `SessionPods.ensure_for` to carry an identifier nothing here reads back.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._waiting: dict[int, PlacementWait] = {}
        self._next_token = 0

    @asynccontextmanager
    async def awaiting(self, session_id: SessionId) -> AsyncIterator[PlacementWait]:
        """Count one Turn of this Session as waiting for a pod until the block exits.

        `try/finally` rather than a plain pair of statements, because every way out of
        that block has to remove the entry. A placement that raises is the common case
        rather than the exotic one -- an unschedulable pod, a definition that will not
        resolve, a Session that may not be resumed -- and a wait left in the register by
        one of those would inflate the queue depth for the life of the process, which is
        the one failure that makes this number worth less than no number.

        The token is minted before the first `await` in the block and never reused, so
        two coroutines entering concurrently cannot be handed the same key. It is an
        `int` that only grows, which is sound because it indexes nothing durable: the
        entry it keys lives for one placement, and the counter resets with the process
        that held them.
        """
        self._next_token += 1
        token = self._next_token
        wait = PlacementWait(session_id, self._clock.now_epoch_ms(), self._clock)
        self._waiting[token] = wait
        try:
            yield wait
        finally:
            wait._finish()
            self._waiting.pop(token, None)

    def snapshot(self) -> tuple[int, int, int | None]:
        """The three queue numbers, read together so they describe one instant.

        Returned as one tuple rather than by three methods for the reason
        `node_headroom` is one call: an operator compares them, and three reads
        interleaved with the placements they are counting could hand back a set no
        instant ever held -- two Turns waiting across three Sessions.

        The oldest instant is the earliest start still in the register, which is the
        Turn that has waited longest and has not stopped waiting. A Turn whose wait
        ended is not the unluckiest one; it is a finished one.
        """
        waiting = tuple(self._waiting.values())
        if not waiting:
            return 0, 0, None
        sessions = {wait.session_id for wait in waiting}
        return len(waiting), len(sessions), min(wait.started_ms for wait in waiting)


class _WallClock:
    """Wall-clock milliseconds since the epoch, for a `Placement` given no clock.

    Concrete here rather than imported, and duplicated rather than shared, on the same
    grounds this module already keeps its pod-naming rule local: what it holds is
    arithmetic every clock in this tree agrees on, not a rule two modules could disagree
    about. A default at all, rather than a required argument, because `Placement` is
    constructed by the composition root as `Placement(pod_runner)` and every test double
    in this suite builds it the same way -- so a required clock would be a signature
    change reaching two dozen call sites to supply a value all but one of them would
    write identically.
    """

    def now_epoch_ms(self) -> int:
        return time.time_ns() // 1_000_000


_BACKLOG_WINDOW_S: Final = 15 * 60
"""How far back a placement wait is still believed to be a wait.

A `session.placing` older than this with no terminal event beside it is not a Turn still
queued -- it is a Turn whose process died mid-placement, taking its connection with it.
Counting those would make the gauge climb and never fall, and a depth that only rises is
worse than no depth.

Fifteen minutes, which is comfortably past every bound on the placement path (the
unschedulable wait and the image-pull wait are both minutes) and comfortably short of an
hour. It does not need to be exact in either direction: too short under-reports a wait
nobody is still on the other end of, and too long over-reports one that has already
failed and been answered."""


class Placement:
    """Places a Session's pod, answers where it is, and reports the fleet's capacity.

    The capacity half is here rather than in a component of its own because it is the
    same subject seen from outside: this object is the only thing that both holds the
    cluster client and is reachable from the serving process's assembled ports, so a
    separate reporter would need either a second copy of the client or a new field on
    `Platform` to be reached from a route. What it adds is two queries and no new
    authority -- nothing below places, removes or changes anything.
    """

    def __init__(
        self,
        runner: PodRunner,
        waits: PlacementWaits | None = None,
        backlog: PlacementBacklogReader | None = None,
    ) -> None:
        self._runner = runner
        self._waits = waits if waits is not None else PlacementWaits(_WallClock())
        self._backlog = backlog

    async def place(self, compiled: CompiledConfig) -> PodBinding:
        """Ensure the pod for this compiled configuration's Session exists.

        The Session is taken from the configuration rather than passed beside it, so
        there is no call that starts one Session's pod from another Session's compiled
        documents — the pod would then run under the wrong Permission Profile and the
        wrong Tool Gateway identity, and nothing downstream would notice.

        The configuration itself passes through opaquely. This module's reason to change
        is the node tier; what the Agent Runtime is configured with is another
        component's business, and reading a field of it here would couple placement to a
        format it has no stake in.

        Raises `PodNotStarted` when the runner could not bring the pod up. The binding
        it would have returned does not exist, so there is nothing truthful to hand
        back; a caller that wants to know where a Session is without starting one asks
        `locate`.
        """
        pod_name = pod_name_for(compiled.session_id)
        phase = await self._runner.ensure(pod_name, compiled)
        return PodBinding(
            session_id=compiled.session_id, pod_name=pod_name, phase=phase
        )

    async def locate(self, session_id: SessionId) -> PodBinding:
        """Where the Session is now, asked of the cluster rather than of a record."""
        pod_name = pod_name_for(session_id)
        phase = await self._runner.phase_of(pod_name)
        return PodBinding(session_id=session_id, pod_name=pod_name, phase=phase)

    async def release(self, session_id: SessionId) -> None:
        """Give the pod back. The Session outlives it; the pod is the disposable one."""
        await self._runner.remove(pod_name_for(session_id))

    def awaiting(
        self, session_id: SessionId
    ) -> AbstractAsyncContextManager[PlacementWait]:
        """Count one Turn among the ones waiting for a pod, until the block exits.

        Delegated rather than exposing the register, so a caller says what it is doing
        -- this Turn is waiting -- instead of reaching through this object into a
        collection and mutating it. The register stays an implementation detail of how
        the count is kept, which is what lets it become something else later without
        every dispatch changing with it.

        Wraps the window in which a Turn has no pod, which is wider than the cluster
        call inside it: resolving the Session's environment, definition and skills
        happens in there too. That is the correct width rather than a convenient one --
        the number answers "how long before the model saw my Turn", and a tenant waiting
        on a definition lookup is waiting just as surely as one waiting on a node.
        """
        return self._waits.awaiting(session_id)

    async def capacity(self) -> PlacementStats:
        """The six numbers that say whether a slow Turn is queued or thinking.

        Assembled from two sources with different reliability, and the type carries the
        difference rather than this docstring alone: the queue numbers are this
        process's own and always answerable, and the cluster numbers are `None` when the
        runner this was built with cannot answer them.

        The pod count reads the cluster and filters to `RUNNING`, so it counts pods that
        can actually serve a Turn rather than pods that exist. A pod still starting is
        capacity that has been paid for and is not yet available, and counting it would
        make the fleet look able to serve Turns it would refuse -- the same distinction
        `PodTurnDispatch` makes before it dials one.

        A runner that cannot enumerate pods reports zero rather than `None`, and the
        asymmetry with the node fields is deliberate. `PlacedPods` is implemented by the
        cluster client every placing deployment wires, so a process that cannot answer
        it is a process that places no pods at all -- and zero running pods is then the
        true answer rather than an unknown one.
        """
        # The shared read when there is one, and the process-local register only when
        # there is not. They answer the same question with different reach, and the
        # difference is not a rounding: the register counts the waits this process is
        # holding open, so an operator whose connection is pinned to the other replica
        # reads an idle platform under full load. Measured, 2026-08-24: six placements,
        # one replica saying 6 and the other saying 0.
        #
        # The fallback is not a degraded mode anybody deploys into. Every serving
        # process is assembled with a store, so `_backlog` is None only in a test that
        # built `Placement` with a runner alone -- and there the register is the right
        # answer, because there is no log to read.
        if self._backlog is not None:
            standing = await self._backlog.placement_backlog(_BACKLOG_WINDOW_S)
            awaiting = standing.turns_awaiting
            sessions = standing.sessions_placing
            oldest_ms = standing.oldest_awaiting_at_ms
        else:
            awaiting, sessions, oldest_ms = self._waits.snapshot()
        running = 0
        if isinstance(self._runner, PlacedPods):
            running = sum(
                1
                for pod in await self._runner.placed_pods()
                if pod.phase is PodPhase.RUNNING
            )
        headroom: NodeHeadroom | None = None
        if isinstance(self._runner, ClusterHeadroom):
            headroom = await self._runner.node_headroom()
        else:
            # Logged, because the two ways the node fields come back empty are
            # indistinguishable on the wire and are different people's problems. This
            # branch is "this build cannot ask" -- a runner with no such method, a
            # wiring fact fixed by deploying a client that has one. The other is "the
            # cluster refused", which the implementation logs itself and is fixed with
            # RBAC. A reader who cannot tell them apart re-derives both from scratch.
            _LOG.info(
                "no node numbers: %s implements no cluster-headroom read, so "
                "nodes_schedulable and node_ceiling are reported as unknown",
                type(self._runner).__name__,
            )
        return PlacementStats(
            turns_awaiting_placement=awaiting,
            sessions_placing=sessions,
            oldest_awaiting_placement_at_ms=oldest_ms,
            session_pods_running=running,
            nodes_schedulable=headroom.schedulable if headroom else None,
            node_ceiling=headroom.ceiling if headroom else None,
        )
