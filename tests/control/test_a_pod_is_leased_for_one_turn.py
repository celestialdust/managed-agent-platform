"""A Session's pod lives exactly as long as the Turn it carries.

The property is about what is true *afterwards*, so every case here reads the phase
back out of a cluster that models one — a pod exists because something created it and
stops existing because something removed it — rather than asking a double what it was
told. A test that asserted `remove` had been called would keep passing over a runner
that removed the wrong pod, and the whole point of the lease is which pod goes.

Both endings are graded, because they are separate code paths and only one of them is
the happy one. A Turn that completes releases its pod; a Turn that fails releases its
pod; a Turn that never reached a running pod leaves nothing behind either. The failing
endings are the ones worth having a test for: a lease released only on success is a
leak whose size is the failure rate, and a platform under load fails more.

What this file does NOT cover is a control plane that dies mid-Turn. Nothing in this
process runs after that, so the release cannot happen here at all and the pod is
collected by the sweep in `control/session/reaper.py` instead. That is the reason the
sweep survives ADR-041 while `IDLE_GRACE_MS` does not.
"""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import Placement, PodPhase
from managed_agent.control.session.turn_dispatch import TurnUndeliverable
from managed_agent.core.ids import Seq, SessionId, TurnId, new_turn_id
from managed_agent.core.vocabulary import turn
from managed_agent.session_shim.pod_channel import HttpPodDispatch

_NAMESPACE = "map-test"
_KEY = b"a signing key that is thirty-two"


class OnePodCluster:
    """A `PodRunner` that holds the pods it was told to hold.

    The phase is derived from that set rather than fixed, which is what lets a case
    assert the lease: `ensure` puts a pod in, `remove` takes it out, and `phase_of`
    answers what a cluster would answer at that moment. A double returning a constant
    could not tell a released pod from a running one, which is the only distinction
    this file is about.

    RUNNING rather than STARTING because these cases need the Turn to reach the pod;
    the neighbouring file that grades placement uses STARTING for the opposite reason.
    """

    def __init__(self, *, holding: bool = False, terminating: bool = False) -> None:
        self.held: set[str] = set()
        self._holding = holding
        self._terminating = terminating
        self.removed: list[str] = []

    def place(self, pod_name: str) -> None:
        """Put a pod at this name, clearing whatever was on its way out first.

        This is what the real runner does rather than a convenience for the double: a
        pod's name is derived from its Session, so a create at an occupied name is
        refused by the API server -- the runner waits the old object out and only then
        creates. Modelling that here is what lets a case start from a terminating pod
        and still end with a Turn that ran.
        """
        self._terminating = False
        self.held.add(pod_name)

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        self.place(pod_name)
        return PodPhase.RUNNING

    async def phase_of(self, pod_name: str) -> PodPhase:
        if self._terminating:
            return PodPhase.GONE
        if self._holding:
            self.held.add(pod_name)
            self._holding = False
        return PodPhase.RUNNING if pod_name in self.held else PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        self.removed.append(pod_name)
        self.held.discard(pod_name)


class PlacesInto:
    """The `SessionPods` seam, placing into the cluster the case is reading.

    A Turn that finds no pod calls this, and what it must do is make the very pod the
    assertions below then look for. Wiring it to the same object rather than to a
    counter is what keeps the two halves of a case from being able to disagree.
    """

    def __init__(self, cluster: OnePodCluster, session_id: SessionId) -> None:
        self._cluster = cluster
        self._session_id = session_id

    async def ensure_for(self, session_id: SessionId) -> None:
        from managed_agent.control.session.placement import pod_name_for

        self._cluster.place(pod_name_for(session_id))


class QuietLog:
    """An `EventLogAppend` that keeps the types, numbering from one."""

    def __init__(self) -> None:
        self.types: list[str] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.types.append(type_)
        return Seq(len(self.types))


class NothingToDo:
    """A completion seam that does nothing, for the cases that grade the lease."""

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        return None


def _a_pod_streaming(turn_id: TurnId, *, completing: bool) -> httpx.MockTransport:
    """A pod answering the Turn route with one Turn, ended either way.

    The `{"kind": "completed"}` trailer is what tells the dispatch the stream ended on
    a completion rather than stopping, so omitting it is how a failed Turn is spelled.
    """
    lines = [
        json.dumps(
            {
                "kind": "event",
                "type": turn.TURN_COMPLETED if completing else turn.TURN_FAILED,
                "payload": (
                    {"turn_id": str(turn_id)}
                    if completing
                    else {"turn_id": str(turn_id), "cause": "model_error"}
                ),
            }
        )
    ]
    if completing:
        lines.append(json.dumps({"kind": "completed"}))
    body = ("\n".join(lines) + "\n").encode()
    return httpx.MockTransport(lambda request: httpx.Response(200, content=body))


def _unreachable() -> httpx.MockTransport:
    """A pod that cannot be dialled at all, which is the third ending."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to the shim")

    return httpx.MockTransport(refuse)


def _dispatch_over(
    cluster: OnePodCluster,
    session_id: SessionId,
    transport: httpx.AsyncBaseTransport,
    log: QuietLog,
) -> HttpPodDispatch:
    return HttpPodDispatch(
        placement=Placement(cluster),
        pods=PlacesInto(cluster, session_id),
        log=log,
        on_completed=NothingToDo(),
        namespace=_NAMESPACE,
        token_key=_KEY,
        transport=transport,
    )


@pytest.mark.asyncio
async def test_a_completed_turn_leaves_no_pod_behind() -> None:
    """The happy ending, and the one the decision is named for."""
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    cluster = OnePodCluster(holding=True)
    dispatch = _dispatch_over(
        cluster, session_id, _a_pod_streaming(turn_id, completing=True), QuietLog()
    )

    await dispatch.dispatch(session_id, turn_id, "a prompt")

    binding = await Placement(cluster).locate(session_id)
    assert binding.phase is PodPhase.ABSENT, (
        "the Turn ended and its pod is still there, so the Session is holding a slot "
        "between Turns -- which is the whole of what this decision removes"
    )


@pytest.mark.asyncio
async def test_a_failed_turn_leaves_no_pod_behind() -> None:
    """A Turn the model failed still ends, and its lease ends with it.

    Separate from the completed case because the two run different seams and only one
    of them is reached by the happy path. A lease released only on success leaks at the
    failure rate, which is highest exactly when the platform is under the most load.
    """
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    cluster = OnePodCluster(holding=True)
    dispatch = _dispatch_over(
        cluster, session_id, _a_pod_streaming(turn_id, completing=False), QuietLog()
    )

    await dispatch.dispatch(session_id, turn_id, "a prompt")

    binding = await Placement(cluster).locate(session_id)
    assert binding.phase is PodPhase.ABSENT


@pytest.mark.asyncio
async def test_a_turn_whose_pod_could_not_be_reached_leaves_no_pod_behind() -> None:
    """The transport failure, which raises past every seam and must still release.

    This is the ending a `finally` is for: the Turn does not complete, nothing is
    appended for it here, and the refusal travels to the route -- and the pod that was
    created for it is still a pod somebody has to pay for.
    """
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    cluster = OnePodCluster(holding=True)
    dispatch = _dispatch_over(cluster, session_id, _unreachable(), QuietLog())

    with pytest.raises(TurnUndeliverable):
        await dispatch.dispatch(session_id, turn_id, "a prompt")

    binding = await Placement(cluster).locate(session_id)
    assert binding.phase is PodPhase.ABSENT


@pytest.mark.asyncio
async def test_a_turn_that_had_to_place_its_own_pod_releases_that_pod() -> None:
    """The ordinary Turn under this decision: no pod exists, so every Turn places one.

    The case above starts from a pod that is already there, which is the state a Turn
    finds only when a previous release did not happen. This one is the shape every Turn
    takes once the lease holds, and it is the one that would leak on every Turn rather
    than on an unlucky one.
    """
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    cluster = OnePodCluster()
    dispatch = _dispatch_over(
        cluster, session_id, _a_pod_streaming(turn_id, completing=True), QuietLog()
    )

    await dispatch.dispatch(session_id, turn_id, "a prompt")

    assert cluster.removed, "the Turn placed a pod and released nothing"
    binding = await Placement(cluster).locate(session_id)
    assert binding.phase is PodPhase.ABSENT


@pytest.mark.asyncio
async def test_the_turn_that_places_its_own_pod_no_longer_says_it_resumed() -> None:
    """`session.resumed` is not appended, and under this decision it cannot be.

    Every Turn now takes the branch that used to mean "this Session is coming back from
    a suspension", so an append there would post a webhook callback per Turn to every
    endpoint a tenant registered for that type -- each one announcing a cold start that
    is now simply what a Turn is. The waiting signal a tenant actually needs is
    `session.placing`, which is appended before the wait rather than after it and is
    unaffected.
    """
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    cluster = OnePodCluster()
    log = QuietLog()
    dispatch = _dispatch_over(
        cluster, session_id, _a_pod_streaming(turn_id, completing=True), log
    )

    await dispatch.dispatch(session_id, turn_id, "a prompt")

    assert "session.resumed" not in log.types
    assert "session.placing" in log.types, (
        "the queued signal went with it, so a tenant watching the stream now learns "
        "nothing while its Turn waits for a node"
    )


@pytest.mark.asyncio
async def test_a_turn_over_a_pod_that_is_still_terminating_places_anyway() -> None:
    """The second Turn of a Session, arriving inside the previous pod's grace period.

    This is what the lease made ordinary. Deleting a pod is asynchronous -- the object
    keeps its name, stamped for deletion, while kubelet stops the containers -- so a
    Session that has taken a Turn before does not find ABSENT. It finds the pod its own
    last Turn released, which reads GONE.

    Refusing on that failed every second Turn submitted inside the grace window, and it
    reached the tenant as a 502 with `the pod for session ... is gone` in the control
    plane's log and nothing at all in the Event Log to explain it. The interactive case
    -- ask, read the answer, ask again -- is exactly the one it broke.

    Graded through to a completed Turn rather than stopping at the phase, because the
    two halves can each look right alone: placing over a terminating pod and then
    dispatching into the pod that is still going away would satisfy any assertion about
    placement and still lose the Turn.
    """
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    cluster = OnePodCluster(terminating=True)
    log = QuietLog()
    dispatch = _dispatch_over(
        cluster, session_id, _a_pod_streaming(turn_id, completing=True), log
    )

    await dispatch.dispatch(session_id, turn_id, "a prompt")

    assert turn.TURN_COMPLETED in log.types, (
        "a Turn that arrived while the last Turn's pod was still terminating was "
        "refused, so no Session can take two Turns inside one grace period"
    )
    assert "session.placing" in log.types
    binding = await Placement(cluster).locate(session_id)
    assert binding.phase is PodPhase.ABSENT, "its own pod outlived its own Turn"
