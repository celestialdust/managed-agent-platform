"""Each way a Turn can fail has its own name, and no two share one.

`pod_unreachable` was reported for at least four unrelated situations. A tenant counted
them from the outside: a missing internal CA, an MCP server answering 421 from a
loopback default, a genuinely dead pod, and an inter-byte deadline killing an agent that
was writing a file. One name for four causes is what turned a two-hour diagnosis into a
two-hour diagnosis, and the remedies disagree -- retrying is right for one of those and
wrong for the others.

So the property here is not "each cause has a name". It is **no two distinguishable
situations produce the same name**, which is the half a test can be vacuous about: a
file that only asserted each new cause is reachable would pass over an implementation
that answered the new cause for everything.

Every case builds the situation rather than raising the exception the situation would
raise, wherever the seam allows it -- a case that constructed `TurnUndeliverable` with
the cause it then asserts would be grading its own argument. Where the situation is a
cluster answer, the cluster double is what differs between cases.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from managed_agent.control.catalog.environments import UnknownEnvironment
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import Placement, PodPhase
from managed_agent.control.session.pods import FirstTurnPlacement, SessionPods
from managed_agent.control.session.turn_dispatch import (
    NoPodTransport,
    TurnUndeliverable,
)
from managed_agent.control.session.turn_execution import run_turn
from managed_agent.core.ids import Seq, SessionId, TurnId, new_session_id, new_turn_id
from managed_agent.core.vocabulary import turn
from managed_agent.session_shim.pod_channel import HttpPodDispatch

pytestmark = pytest.mark.anyio

_NAMESPACE = "map-test"
_KEY = b"a signing key that is thirty-two"
_CAUSE = turn.TurnFailureCause


class CollectingLog:
    """Keeps what was appended, and answers what closed the Turn."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, object]]] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.rows.append((type_, dict(payload)))
        return Seq(len(self.rows))

    def closing(self) -> dict[str, object]:
        failed = [payload for type_, payload in self.rows if type_ == turn.TURN_FAILED]
        assert failed, f"no turn.failed was appended; got {[t for t, _ in self.rows]}"
        return failed[-1]

    def cause(self) -> str:
        return str(self.closing()["cause"])


class Cluster:
    """A `PodRunner` answering one fixed phase, which is what a case varies."""

    def __init__(self, phase: PodPhase) -> None:
        self._phase = phase

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        return self._phase

    async def remove(self, pod_name: str) -> None:
        return None

    async def phase_of(self, pod_name: str) -> PodPhase:
        return self._phase


class WillNotPlace(FirstTurnPlacement):
    """The shipped `ensure_for`, with only the step underneath it made to fail.

    Subclassed rather than doubled, and constructed without the twelve collaborators
    the real one takes, because none of them is reached: what these cases grade is the
    translation `ensure_for` performs *around* `_place`, and that is the real method
    here. A hand-written double raising `TurnUndeliverable` with the cause the case then
    asserts would be grading its own argument -- which is the mistake this whole file
    is about.
    """

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def _place(self, session_id: SessionId) -> None:
        raise self._error


class NeverPlaces:
    """A `SessionPods` that succeeds without creating anything."""

    async def ensure_for(self, session_id: SessionId) -> None:
        return None


class NothingToShip:
    """A `TurnCompleted` seam with nothing to do, so no case grades a ship-out."""

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        return None


def _dispatch(
    *,
    phase: PodPhase,
    pods: SessionPods,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> HttpPodDispatch:
    """The shipped dispatch, differing per case only in the cluster and the shim."""
    transport = httpx.MockTransport(handler) if handler is not None else None
    return HttpPodDispatch(
        placement=Placement(Cluster(phase)),
        pods=pods,
        log=CollectingLog(),
        on_completed=NothingToShip(),
        namespace=_NAMESPACE,
        token_key=_KEY,
        transport=transport,
    )


async def _closed_by(dispatch: HttpPodDispatch) -> str:
    """Run one Turn through a real dispatch and hand back the cause that closed it."""
    log = CollectingLog()
    await run_turn(dispatch, log, new_session_id(), new_turn_id(), "summarise it")
    return log.cause()


async def test_a_session_that_may_not_be_placed_is_not_a_pod_that_is_unreachable() -> (
    None
):
    """The configuration this Session names is wrong, and retrying will not fix it."""
    dispatch = _dispatch(
        phase=PodPhase.ABSENT,
        pods=WillNotPlace(UnknownEnvironment("no such environment")),
    )

    assert await _closed_by(dispatch) == _CAUSE.SESSION_NOT_PLACEABLE.value


async def test_a_pod_that_was_asked_for_and_never_came_up_says_so() -> None:
    """Placement succeeded and the cluster still has no running pod.

    Distinct from the case above because the remedy is opposite: nothing about this
    Session's configuration is wrong, and this is the one worth retrying.
    """
    dispatch = _dispatch(phase=PodPhase.STARTING, pods=NeverPlaces())

    assert await _closed_by(dispatch) == _CAUSE.RUNTIME_DID_NOT_START.value


async def test_a_shim_that_cannot_be_dialled_is_what_pod_unreachable_now_means() -> (
    None
):
    """The cluster says the pod runs and nothing answers on its address.

    This is the only situation left under `pod_unreachable`, which is the point of the
    slice: the name is now true when it is used.
    """

    def refuses_to_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    dispatch = _dispatch(
        phase=PodPhase.RUNNING, pods=NeverPlaces(), handler=refuses_to_connect
    )

    assert await _closed_by(dispatch) == _CAUSE.POD_UNREACHABLE.value


async def test_a_shim_that_answers_and_declines_is_not_a_shim_that_is_absent() -> None:
    """Something is listening and it said no, which no retry changes."""

    def declines(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    dispatch = _dispatch(phase=PodPhase.RUNNING, pods=NeverPlaces(), handler=declines)

    assert await _closed_by(dispatch) == _CAUSE.RUNTIME_REFUSED_THE_TURN.value


async def test_a_turn_whose_wire_timed_out_says_that_and_not_pod_unreachable() -> None:
    """A timeout from beneath the dispatch gets its own name.

    Reported as `pod_unreachable` while the inter-byte deadline existed, which is the
    single most expensive collapse in this set: the pod was reachable, the agent was
    working, and the one remedy that name implies -- retry -- re-runs the same work
    into the same wall.

    The timeout is raised by the dispatch rather than induced by a deadline this test
    injects, because `run_turn` no longer holds one -- the hour-long total was removed
    on 2026-08-26 for killing healthy long Turns. What can still reach this handler is
    the wire's own `RUNTIME_SILENCE_DEADLINE_S`, and a socket that produced nothing for
    an hour raises exactly this, so raising it directly is the honest stand-in rather
    than a shortcut.
    """

    class TheWireGaveUp:
        async def dispatch(
            self, session_id: SessionId, turn_id: TurnId, prompt: str
        ) -> None:
            raise TimeoutError("the socket produced nothing for an hour")

    log = CollectingLog()
    await run_turn(
        TheWireGaveUp(),
        log,
        new_session_id(),
        new_turn_id(),
        "write it",
    )

    assert log.cause() == _CAUSE.TURN_DEADLINE_EXCEEDED.value


async def test_a_deploy_with_no_transport_says_so_rather_than_blaming_a_pod() -> None:
    """Every Turn in this process fails, and no tenant action changes that."""
    log = CollectingLog()

    await run_turn(
        NoPodTransport(), log, new_session_id(), new_turn_id(), "summarise it"
    )

    assert log.cause() == _CAUSE.NO_RUNTIME_CONFIGURED.value


async def test_no_two_of_these_situations_produce_the_same_cause() -> None:
    """The half that a per-cause test cannot assert on its own.

    A file of six cases each asserting its own cause passes over an implementation that
    answered one cause for all six, if the six were written to match it. This runs them
    together and asserts the set is as large as the list.
    """

    def refuses_to_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    def declines(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    causes = [
        await _closed_by(
            _dispatch(
                phase=PodPhase.ABSENT,
                pods=WillNotPlace(UnknownEnvironment("no such environment")),
            )
        ),
        await _closed_by(_dispatch(phase=PodPhase.STARTING, pods=NeverPlaces())),
        await _closed_by(
            _dispatch(
                phase=PodPhase.RUNNING, pods=NeverPlaces(), handler=refuses_to_connect
            )
        ),
        await _closed_by(
            _dispatch(phase=PodPhase.RUNNING, pods=NeverPlaces(), handler=declines)
        ),
    ]

    assert len(set(causes)) == len(causes), (
        f"four different failures produced {sorted(set(causes))}. A name shared by two "
        "situations is the defect this slice removes, not a tidy default."
    )


def test_every_cause_carries_a_remedy_and_no_two_remedies_are_the_same() -> None:
    """A name a tenant cannot act on is a name that has not helped them.

    Totality is asserted rather than the mapping's contents, so a cause added later
    fails here instead of silently reaching a tenant with no next move. The remedies
    must also differ: two causes with one sentence are two causes a reader cannot use,
    which is the same defect one name for two causes was.
    """
    remedies = {cause: turn.REMEDY_FOR[cause] for cause in turn.TurnFailureCause}

    assert len(set(remedies.values())) == len(remedies), (
        "two causes share a remedy sentence, so telling them apart buys a tenant "
        f"nothing: {sorted(remedies.values())}"
    )
    for cause, remedy in remedies.items():
        assert remedy.strip(), f"{cause.value} has no remedy"


def test_no_remedy_names_the_platform_underneath_the_session() -> None:
    """These sentences go to tenants, so ADR-013 binds them like any refusal.

    "runtime" is permitted and the rest are not: three published causes already carry
    the word, so it is vocabulary a consumer has, whereas a pod, a bucket, a lane or a
    cluster are the topology that record withholds. Asserted over the whole mapping
    rather than per sentence, so a remedy added later is covered without anybody
    remembering to extend this.
    """
    forbidden = ("pod", "bucket", "s3", "lane", "cluster", "kubernetes", "transport")
    for cause, remedy in turn.REMEDY_FOR.items():
        named = [word for word in forbidden if word in remedy.lower()]
        assert not named, (
            f"the remedy for {cause.value} names {named}, which is the platform's own "
            "topology rather than anything the tenant owns or can act on"
        )


async def test_the_closing_event_carries_the_remedy_the_tenant_acts_on() -> None:
    """The event log is the only surface a failed Turn reaches now.

    Written into the event rather than looked up from a live table, because the log is
    append-only and read long afterwards: a row carries the sentence that was true when
    it was written, and a mapping consulted later would answer for today's vocabulary
    about a failure from months ago.
    """
    log = CollectingLog()

    await run_turn(
        NoPodTransport(), log, new_session_id(), new_turn_id(), "summarise it"
    )

    closed = log.closing()
    assert closed["remedy"] == turn.REMEDY_FOR[_CAUSE.NO_RUNTIME_CONFIGURED]


def test_a_turn_undeliverable_that_names_no_cause_is_still_a_pod_unreachable() -> None:
    """The default is the old behaviour, so an unconverted raise site is not a crash.

    Seventeen sites raise this type and only the dispatch path's few carry a cause. The
    rest reach a tenant as they always did rather than as an exception nothing handles.
    """
    assert TurnUndeliverable("something").cause is _CAUSE.POD_UNREACHABLE


def test_the_published_cause_set_is_closed_and_ordered() -> None:
    """Members are appended, never inserted, because the published table is ordered.

    A member added in the middle re-orders a table somebody reads, and a consumer
    branching on this set learns of a new member from a release note rather than from a
    value it has no arm for.
    """
    assert [cause.value for cause in turn.TurnFailureCause] == [
        "runtime_reported_failure",
        "runtime_lost",
        "pod_unreachable",
        "output_not_revisable",
        "session_not_placeable",
        "runtime_did_not_start",
        "runtime_refused_the_turn",
        "turn_deadline_exceeded",
        "no_runtime_configured",
    ]
