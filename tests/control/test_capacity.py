"""The capacity surface reports a queue that is actually there, to a reader entitled
to it.

Tier 1: no cluster, no AWS, no network. Every number below comes from a fake runner or
from the in-process register, which is the whole point -- a capacity instrument that
needed a cluster to be tested is one that would go untested until the load test it
exists to make readable.

**The claim that matters is that the numbers move.** A field that is present and always
zero is worse than no field, because it reads as a measurement. So the cases here put
Turns into the placement window and assert the answer differs from the empty answer, in
each of the four ways it can differ: how many Turns wait, how many Sessions those
belong to, which one has waited longest, and how many pods can serve.

**What is deliberately NOT proven here.** That a real Kubernetes API answers a node
count or an autoscaler's arguments. The adapter's two reads are graded against fakes
standing in for the API client, which settles how each answer and each refusal is turned
into a number and settles nothing about whether the cluster grants the bindings they
need -- and a refused binding is why both fields can arrive null from a runner that
implements them. What is proven above that is the route publishing whatever the runner
returns, and publishing null rather than a stand-in when it cannot answer.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from kubernetes_asyncio.client.exceptions import ApiException

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import capacity
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.reviewers.token import (
    HmacReviewerTokens,
    mint_reviewer_token,
)
from managed_agent.control.session.lifecycle import NoSessionPods
from managed_agent.control.session.placement import (
    ClusterHeadroom,
    NodeHeadroom,
    PlacedPod,
    PlacedPods,
    Placement,
    PlacementStats,
    PlacementWaits,
    PodPhase,
)
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.core.ids import (
    Seq,
    SessionId,
    TurnId,
    new_session_id,
    new_turn_id,
)
from managed_agent.core.vocabulary import turn
from managed_agent.session_shim.pod_channel import HttpPodDispatch

_CAPACITY_PATH = "/v1/capacity"

_OLDEST = "oldest_awaiting_placement_at"

_REVIEWER_KEY = b"the control plane's own signing key"
_REVIEWER_EXPIRY = 4102444800
"""2100-01-01. Distant on purpose: the shipped authenticator reads the real clock, so a
credential meant to be valid has to outlive the calendar rather than this file's idea of
now."""

_NOW_MS = 1_700_000_000_000
"""A fixed instant, so no assertion below turns on how long a test took to run."""


class MovableClock:
    """A clock a test advances by hand, so an elapsed duration is a stated number.

    Every wait measured here is the difference between two reads of this, which is what
    lets a case assert `1500` rather than "some positive number" -- and "some positive
    number" is exactly the assertion that would pass on a broken measurement.
    """

    def __init__(self, now_ms: int = _NOW_MS) -> None:
        self.now_ms = now_ms

    def advance(self, by_ms: int) -> None:
        self.now_ms += by_ms

    def now_epoch_ms(self) -> int:
        return self.now_ms


class ClusterHoldingPods:
    """A runner that reports a fixed set of Session pods and answers no node question.

    Deliberately does NOT implement `node_headroom`, because that is the shape of every
    runner this repository actually has: the Kubernetes adapter has four methods and
    none of them reads a node. A double that answered would make the null case
    unreachable and would have this suite grading a capability nothing ships.
    """

    def __init__(self, phases: Sequence[PodPhase] = ()) -> None:
        self.pods = tuple(
            PlacedPod(session_id=new_session_id(), phase=phase, created_at_ms=_NOW_MS)
            for phase in phases
        )

    async def ensure(self, pod_name: str, compiled: Any) -> PodPhase:
        raise AssertionError("a test in this file placed a pod through the runner")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        raise AssertionError("a test in this file released a pod")

    async def placed_pods(self) -> Sequence[PlacedPod]:
        return self.pods


class ClusterThatKnowsItsNodes(ClusterHoldingPods):
    """A runner that can also answer both node questions.

    Stands in for the adapter method that does not exist yet, so the route's rendering
    of a *present* ceiling is graded rather than assumed. The two numbers it returns are
    the real cluster's measured pair -- four schedulable against a `--max-nodes-total`
    of four, while the nodegroup declares eight -- because the case worth rendering is
    the one an operator is actually going to read.
    """

    def __init__(
        self,
        phases: Sequence[PodPhase] = (),
        *,
        schedulable: int = 4,
        ceiling: int | None = 4,
    ) -> None:
        super().__init__(phases)
        self._headroom = NodeHeadroom(schedulable=schedulable, ceiling=ceiling)

    async def node_headroom(self) -> NodeHeadroom:
        return self._headroom


class RunnerWithNoEnumeration:
    """The three-method `PodRunner` half this suite is full of.

    Present so the "a runner that cannot enumerate pods reports zero" branch is reached
    by an object shaped like the doubles a reader will meet elsewhere, rather than by a
    contrivance.
    """

    async def ensure(self, pod_name: str, compiled: Any) -> PodPhase:
        raise AssertionError("a test in this file placed a pod through the runner")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        return None


class UnusedLog:
    """The Event Log ports, which no case in this file reads or writes."""

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        raise AssertionError("the capacity surface appended an event")

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Any]:
        raise AssertionError("the capacity surface read the Event Log")

    def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Any]:
        raise AssertionError("the capacity surface followed the Event Log")

    async def retained_floor(self, session_id: SessionId) -> Seq:
        raise AssertionError("the capacity surface asked for a retained floor")


class RefusesContact:
    """Every store the capacity path must not touch, as one refusing object.

    One class rather than five, because the assertion is the same for all of them and a
    reader counting them learns nothing: this surface names no tenant, so nothing here
    should ever be asked anything. Attribute access is what raises, so a method added to
    any of those ports later is refused too without this file changing.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the capacity surface reached a store: .{name}()")


def _platform(release: Any) -> Platform:
    """A platform holding nothing but the release the capacity route narrows.

    Every other port refuses contact, which is what makes "this surface reads the
    placement object and nothing else" an assertion rather than a claim in a docstring.
    """
    return Platform(
        event_log_append=UnusedLog(),
        event_log_range=UnusedLog(),
        definition_registry=RefusesContact(),
        tool_registry=RefusesContact(),
        session_registry=RefusesContact(),
        webhooks=RefusesContact(),
        environment_store=RefusesContact(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
        reviewer_authenticator=HmacReviewerTokens(
            key=_REVIEWER_KEY, clock=MovableClock()
        ),
        session_pod_release=release,
    )


def _app(release: Any) -> FastAPI:
    """The shipped app, as assembled, with nothing mounted by this file.

    Every case below goes through the same routing, dependencies and middleware a
    deployment serves, which is what makes the authorization cases mean anything: a
    router mounted here instead could carry its dependencies and still miss whatever
    `create_app` wraps around them. That the mount is present and singular is asserted
    separately, so a mount removed or duplicated fails as itself rather than as a
    confusing 404 in twenty other cases.
    """
    return create_app(_platform(release))


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://control")


def _reviewer() -> dict[str, str]:
    """A real credential: signed with the platform's key, unexpired, audit audience."""
    return {
        "Authorization": "Bearer "
        + mint_reviewer_token(
            reviewer_id=uuid.uuid4(),
            expiry_epoch_s=_REVIEWER_EXPIRY,
            key=_REVIEWER_KEY,
        )
    }


# --------------------------------------------------------------------------------------
# The register: the numbers move, and every exit path clears the entry
# --------------------------------------------------------------------------------------


async def test_an_idle_platform_and_a_queued_one_do_not_report_the_same_numbers() -> (
    None
):
    """The whole deliverable in one case: waiting looks different from not waiting.

    Asserted as two full tuples rather than field by field, so a change that fixed one
    number by breaking another cannot pass. The idle answer carries `None` for the
    oldest instant and not an epoch, because an empty queue has no oldest member.
    """
    clock = MovableClock()
    placement = Placement(ClusterHoldingPods(), PlacementWaits(clock))
    session_id = new_session_id()

    idle = await placement.capacity()
    async with placement.awaiting(session_id):
        queued = await placement.capacity()
    after = await placement.capacity()

    assert (idle.turns_awaiting_placement, idle.oldest_awaiting_placement_at_ms) == (
        0,
        None,
    )
    assert (queued.turns_awaiting_placement, queued.sessions_placing) == (1, 1)
    assert queued.oldest_awaiting_placement_at_ms == _NOW_MS
    assert (after.turns_awaiting_placement, after.oldest_awaiting_placement_at_ms) == (
        0,
        None,
    )


async def test_two_turns_of_one_session_are_two_waiting_and_one_placing() -> None:
    """The two counts are different numbers, and this is the case that separates them.

    Both Turns reach the dispatch and both find no pod, because admission refuses a
    Session that will not take a Turn rather than one that already has a Turn open. A
    register keyed by Session would answer 1 and 1 here -- under-reporting the depth at
    exactly the moment depth is the number somebody needs.
    """
    clock = MovableClock()
    placement = Placement(ClusterHoldingPods(), PlacementWaits(clock))
    one_session = new_session_id()

    async with placement.awaiting(one_session), placement.awaiting(one_session):
        both = await placement.capacity()

    assert both.turns_awaiting_placement == 2
    assert both.sessions_placing == 1


async def test_the_oldest_instant_is_the_earliest_wait_still_waiting() -> None:
    """The unluckiest request, and it stays the unluckiest when a later one finishes.

    The second Turn starts later and ends first, which is the ordering a naive
    implementation gets wrong: dropping to "the most recent start" or "the last one
    still open" would report the newer instant and make the worst wait invisible.
    """
    clock = MovableClock()
    placement = Placement(ClusterHoldingPods(), PlacementWaits(clock))
    first, second = new_session_id(), new_session_id()

    async with placement.awaiting(first):
        clock.advance(5_000)
        async with placement.awaiting(second):
            while_both_wait = await placement.capacity()
        while_only_the_older_waits = await placement.capacity()

    assert while_both_wait.oldest_awaiting_placement_at_ms == _NOW_MS
    assert while_both_wait.sessions_placing == 2
    assert while_only_the_older_waits.oldest_awaiting_placement_at_ms == _NOW_MS
    assert while_only_the_older_waits.sessions_placing == 1


@pytest.mark.parametrize("how_it_left", ["raised", "cancelled"])
async def test_a_wait_that_does_not_return_normally_still_leaves_the_queue(
    how_it_left: str,
) -> None:
    """A leaked entry would inflate the depth for the life of the process.

    Both abnormal exits are exercised because both are ordinary here rather than
    exotic. A placement raises whenever a pod will not schedule, a definition will not
    resolve or a Session may not be resumed; and it is cancelled whenever the tenant
    holding the Turn's HTTP connection hangs up, which under load is the common case.
    """
    clock = MovableClock()
    placement = Placement(ClusterHoldingPods(), PlacementWaits(clock))
    session_id = new_session_id()

    async def wait_and_fail() -> None:
        async with placement.awaiting(session_id):
            if how_it_left == "raised":
                raise RuntimeError("the pod would not schedule")
            await asyncio.sleep(3600)

    if how_it_left == "raised":
        with pytest.raises(RuntimeError):
            await wait_and_fail()
    else:
        task = asyncio.ensure_future(wait_and_fail())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    left_behind = await placement.capacity()
    assert left_behind.turns_awaiting_placement == 0
    assert left_behind.oldest_awaiting_placement_at_ms is None


# --------------------------------------------------------------------------------------
# The cluster half of the answer
# --------------------------------------------------------------------------------------


async def test_only_pods_that_can_serve_a_turn_are_counted_as_serving() -> None:
    """Capacity is what can take work, not what exists.

    A starting pod has been paid for and cannot take a Turn, and counting it would make
    the fleet look able to serve Turns it would refuse -- the same distinction the
    dispatch makes before it dials one. The fixture holds one pod in each phase, so a
    count of anything but 1 means some phase was admitted that should not have been.
    """
    cluster = ClusterHoldingPods(
        (PodPhase.RUNNING, PodPhase.STARTING, PodPhase.ABSENT, PodPhase.GONE)
    )
    placement = Placement(cluster, PlacementWaits(MovableClock()))

    assert (await placement.capacity()).session_pods_running == 1


async def test_a_runner_that_cannot_enumerate_pods_reports_zero_not_a_guess() -> None:
    """Zero rather than a crash, because such a process places no pods to count."""
    placement = Placement(RunnerWithNoEnumeration(), PlacementWaits(MovableClock()))
    stats = await placement.capacity()

    assert stats.session_pods_running == 0
    assert stats.nodes_schedulable is None
    assert stats.node_ceiling is None


async def test_the_node_numbers_are_the_runner_s_answer_and_never_invented() -> None:
    """Both directions, because the pair is the point.

    A runner that can answer has its numbers published verbatim -- including the case
    an operator cares about, a ceiling that binds below the nodegroup's declared
    maximum. A runner that cannot answer yields null. What must never appear is a number
    this platform made up: the pod count standing in for the node count, or a constant
    copied out of a manifest the serving process cannot read.
    """
    knows = Placement(
        ClusterThatKnowsItsNodes((PodPhase.RUNNING,), schedulable=4, ceiling=4),
        PlacementWaits(MovableClock()),
    )
    does_not = Placement(
        ClusterHoldingPods((PodPhase.RUNNING,)), PlacementWaits(MovableClock())
    )

    answered, unanswered = await knows.capacity(), await does_not.capacity()

    assert (answered.nodes_schedulable, answered.node_ceiling) == (4, 4)
    assert (unanswered.nodes_schedulable, unanswered.node_ceiling) == (None, None)
    assert answered.session_pods_running == unanswered.session_pods_running == 1


def test_the_cluster_client_this_platform_wires_can_answer_a_node_question() -> None:
    """The shipped adapter, not a double, because the claim is about what deploys.

    `KubernetesPodRunner` now satisfies all three narrowings, so a deployment's node
    fields are answered by real cluster reads rather than reported unknown. This
    replaces an earlier case that asserted the opposite and said in its own docstring
    that the honest edit on the day the adapter grew the method was to delete it rather
    than loosen it -- which is what happened.

    `issubclass` against a `runtime_checkable` Protocol sees method NAMES and not
    signatures, so this proves the method is there and proves nothing about what it
    returns. What it returns is graded below against a fake API, and what it needs from
    the cluster is graded by nothing here: RBAC lives in a manifest, and the honest
    place to assert a manifest is the deploy tier.
    """
    from managed_agent.adapters.kubernetes.pod_runner import KubernetesPodRunner

    assert issubclass(KubernetesPodRunner, ClusterHeadroom)
    assert issubclass(KubernetesPodRunner, PlacedPods)


# --------------------------------------------------------------------------------------
# The surface: who may read it, and what the numbers look like on the wire
# --------------------------------------------------------------------------------------


async def test_a_caller_who_proves_nothing_learns_nothing_about_the_fleet() -> None:
    """Deny by default, and a tenant header buys no more than an empty request does.

    The second half is the one worth asserting: the tenant surface parses an
    unauthenticated header, so a capacity route that accepted one would have made the
    fleet's shape readable by anyone who set a header -- how many Sessions hold pods and
    how close the cluster is to refusing.
    """
    placement = Placement(ClusterHoldingPods((PodPhase.RUNNING,)))
    async with _client(_app(placement)) as caller:
        anonymous = await caller.get(_CAPACITY_PATH)
        with_a_tenant = await caller.get(
            _CAPACITY_PATH, headers={TENANT_HEADER: str(uuid.uuid4())}
        )
        as_a_reviewer = await caller.get(_CAPACITY_PATH, headers=_reviewer())

    assert anonymous.status_code == 401
    assert with_a_tenant.status_code == 401
    assert as_a_reviewer.status_code == 200
    assert "placement_stats" not in anonymous.text
    assert "placement_stats" not in with_a_tenant.text


async def test_the_body_carries_the_numbers_the_placement_object_reported() -> None:
    """One queued Turn, one running pod, and a ceiling, rendered as a reader sees them.

    The instant goes out as an aware UTC timestamp rather than a bare number, so a
    consumer does not have to guess a zone, and the `type` discriminator lets it branch
    on a field it read rather than on the URL it called.
    """
    clock = MovableClock()
    placement = Placement(
        ClusterThatKnowsItsNodes((PodPhase.RUNNING, PodPhase.STARTING)),
        PlacementWaits(clock),
    )
    session_id = new_session_id()

    async with placement.awaiting(session_id), _client(_app(placement)) as caller:
        answer = await caller.get(_CAPACITY_PATH, headers=_reviewer())

    assert answer.status_code == 200
    body = answer.json()
    assert {name: body[name] for name in body if name != _OLDEST} == {
        "type": "placement_stats",
        "turns_awaiting_placement": 1,
        "sessions_placing": 1,
        "session_pods_running": 1,
        "nodes_schedulable": 4,
        "node_ceiling": 4,
    }
    # Compared as an instant rather than as a string, because the exact spelling of a
    # UTC offset is the serializer's business -- it emits `Z` where `isoformat` writes
    # `+00:00`, and a case pinned to either spelling would be asserting which library
    # rendered the field. What has to hold is that it names the right moment and names
    # its zone at all: a naive timestamp here would leave a consumer guessing.
    assert body[_OLDEST].endswith(("Z", "+00:00"))
    assert datetime.fromisoformat(body[_OLDEST]) == datetime.fromtimestamp(
        _NOW_MS / 1000, tz=UTC
    )


async def test_an_empty_queue_is_null_and_not_the_epoch() -> None:
    """A zero here would read as a Turn that has been waiting since 1970."""
    placement = Placement(ClusterHoldingPods(), PlacementWaits(MovableClock()))
    async with _client(_app(placement)) as caller:
        answer = await caller.get(_CAPACITY_PATH, headers=_reviewer())

    assert answer.json()["oldest_awaiting_placement_at"] is None
    assert answer.json()["turns_awaiting_placement"] == 0


async def test_a_process_that_places_no_pods_answers_rather_than_refusing() -> None:
    """The instrument has to produce numbers when something is already wrong.

    A control plane wired without a cluster holds `NoSessionPods`, which cannot report
    capacity -- and the honest answer is an empty fleet, not a 500, because the only
    thing that creates a Session pod is built inside the same branch that wires the real
    release. Refusing would take the instrument away at the moment it is wanted.
    """
    async with _client(_app(NoSessionPods())) as caller:
        answer = await caller.get(_CAPACITY_PATH, headers=_reviewer())

    assert answer.status_code == 200
    assert answer.json()["turns_awaiting_placement"] == 0
    assert answer.json()["session_pods_running"] == 0
    assert answer.json()["node_ceiling"] is None


async def test_the_shipped_app_serves_this_path_for_reading_and_nothing_else() -> None:
    """Mounted on the app a deployment runs, and answering GET alone.

    Every other case here reaches the route through the same `create_app`, so a mount
    that went missing would fail them all -- but it would fail them as twenty confusing
    404s. This is the case that fails as itself.

    Asserted by making requests rather than by walking `app.routes`, because that
    structure is a FastAPI internal that has already changed shape under this codebase
    once: included routers no longer appear there as flattened routes at all, so a guard
    reading it went quietly vacuous rather than red. A guard whose whole job is to catch
    a silent failure must not have a silent failure of its own.

    405 on every write verb is the half worth stating outright. Reading how full the
    cluster is opens no path to changing it, and a write added to this router later
    would answer something other than 405 here.
    """
    verbs = ("POST", "PUT", "PATCH", "DELETE")
    async with _client(_app(NoSessionPods())) as caller:
        read = await caller.get(_CAPACITY_PATH, headers=_reviewer())
        refused = {
            verb: (
                await caller.request(verb, _CAPACITY_PATH, headers=_reviewer())
            ).status_code
            for verb in verbs
        }

    assert read.status_code == 200, "the router is not mounted on the shipped app"
    assert refused == dict.fromkeys(verbs, 405)


def test_the_wire_shape_names_every_number_the_domain_type_carries() -> None:
    """No field is dropped on the way out, asserted rather than eyeballed.

    A number computed and never published is the failure this catches: the six fields
    exist because six questions could not be answered, and one silently missing from
    the response model is one of those questions still unanswered.
    """
    published = set(capacity.PlacementStatsView.model_fields) - {"type"}
    carried = {f.name for f in PlacementStats.__dataclass_fields__.values()}

    assert published == {
        name.removesuffix("_ms") if name.endswith("_at_ms") else name
        for name in carried
    }


# --------------------------------------------------------------------------------------
# The live path: the dispatch a deployment runs is the thing that counts the wait
# --------------------------------------------------------------------------------------


class ShimThatAcceptsAndFails(httpx.AsyncBaseTransport):
    """A pod that accepts the Turn and reports it failed, in one line.

    The Turn's own events are beside the point here -- what is under test is the queue
    depth during the placement that precedes them -- so this streams the shortest legal
    Turn rather than a realistic one. It has to be terminal: a stream ending with no
    terminal event is refused by the dispatch, which would fail this case for a reason
    that has nothing to do with the queue.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        line = json.dumps(
            {
                "kind": "event",
                "type": turn.TURN_FAILED,
                "payload": {"turn_id": str(uuid.uuid4()), "cause": "runtime_lost"},
            }
        )

        async def body() -> AsyncIterator[bytes]:
            yield line.encode() + b"\n"

        return httpx.Response(
            200, headers={"content-type": "application/x-ndjson"}, content=body()
        )


class CountingAppend:
    """An append that records nothing but the count, since order is graded elsewhere."""

    def __init__(self) -> None:
        self.appends = 0

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.appends += 1
        return Seq(self.appends)


class ClusterThatPlacesOnDemand:
    """ABSENT until a pod is asked for, RUNNING afterwards.

    Both answers are needed: the dispatch places only on ABSENT and re-reads the phase
    afterwards rather than trusting what placement returned, so a runner fixed at one
    phase could not reach both halves of the branch under test.
    """

    def __init__(self) -> None:
        self.placed = False

    async def ensure(self, pod_name: str, compiled: Any) -> PodPhase:
        raise AssertionError("the dispatch placed directly instead of via SessionPods")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.RUNNING if self.placed else PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        raise AssertionError("a test in this file released a pod")

    async def placed_pods(self) -> Sequence[PlacedPod]:
        return ()


class PodsThatWatchTheQueue:
    """The placement seam, reading the queue from inside its own placement.

    **This is what makes the instrumentation's position checkable rather than assumed.**
    Every other case in this file drives `Placement.awaiting` directly, which proves the
    register's arithmetic and proves nothing about whether the code a Turn runs through
    ever enters it. The wait is counted by the dispatch `composition.build` wires,
    around its call to this seam, so an observation taken in here comes from inside that
    window. If the wrap were missing, or had been left behind in the placement seam
    where it first went, this would see an empty queue.
    """

    def __init__(
        self, cluster: ClusterThatPlacesOnDemand, placement: Placement
    ) -> None:
        self._cluster = cluster
        self._placement = placement
        self.seen_from_inside: PlacementStats | None = None

    async def ensure_for(self, session_id: SessionId) -> None:
        self.seen_from_inside = await self._placement.capacity()
        self._cluster.placed = True


class UnusedCompletion:
    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        raise AssertionError("a test in this file completed a Turn")


async def test_the_dispatch_a_deployment_runs_is_what_counts_the_wait() -> None:
    """The queue reads 1 while a Turn is being placed, and 0 once it is not.

    Driven through `HttpPodDispatch` rather than `FirstTurnPlacement`, because that is
    where the wrap lives: it is the one place that both measures the wait and holds the
    Event Log needed to announce it, so counting anywhere else would either miss the
    announcement or double the count.

    The 0 afterwards is half the assertion. A wait left in the register would inflate
    the depth for the life of the process, and an inflated depth is the one failure that
    makes this number worth less than no number.
    """
    clock = MovableClock()
    cluster = ClusterThatPlacesOnDemand()
    placer = Placement(cluster, PlacementWaits(clock))
    pods = PodsThatWatchTheQueue(cluster, placer)

    await HttpPodDispatch(
        placement=placer,
        pods=pods,
        log=CountingAppend(),
        on_completed=UnusedCompletion(),
        namespace="map-test",
        token_key=b"a shim signing key",
        transport=ShimThatAcceptsAndFails(),
    ).dispatch(new_session_id(), new_turn_id(), "summarise it")

    from_inside = pods.seen_from_inside
    assert from_inside is not None, "the dispatch never reached the placement seam"
    assert from_inside.turns_awaiting_placement == 1
    assert from_inside.sessions_placing == 1
    assert from_inside.oldest_awaiting_placement_at_ms == _NOW_MS

    after = await placer.capacity()
    assert after.turns_awaiting_placement == 0
    assert after.oldest_awaiting_placement_at_ms is None


# --------------------------------------------------------------------------------------
# The cluster adapter's two reads, against fakes rather than a cluster
# --------------------------------------------------------------------------------------


def _node(
    *, ready: bool = True, unschedulable: bool = False, taints: Sequence[str] = ()
) -> Any:
    """One node, in as little detail as the count actually reads."""
    return SimpleNamespace(
        spec=SimpleNamespace(
            unschedulable=unschedulable,
            taints=[SimpleNamespace(key=key) for key in taints],
        ),
        status=SimpleNamespace(
            conditions=[
                SimpleNamespace(type="Ready", status="True" if ready else "False")
            ]
        ),
    )


def _deployment(*args_per_container: Sequence[str]) -> Any:
    return SimpleNamespace(
        spec=SimpleNamespace(
            template=SimpleNamespace(
                spec=SimpleNamespace(
                    containers=[
                        SimpleNamespace(args=list(args)) for args in args_per_container
                    ]
                )
            )
        )
    )


def _returning(value: Any) -> Any:
    """An async callable answering `value`, standing in for one API method."""

    async def call(*args: Any, **kwargs: Any) -> Any:
        return value

    return call


def test_a_node_counts_only_when_all_three_signals_agree_it_takes_work() -> None:
    """Three signals, any one of which disqualifies, because three actors write them.

    A cordon writes `spec.unschedulable`, the node controller writes the Ready
    condition, and the scheduler consults the taints. A count resting on one alone
    reports nodes as capacity the scheduler skips, which is the direction that hides a
    full cluster, and is why all three are read.
    """
    from managed_agent.adapters.kubernetes.pod_runner import _will_take_a_pod

    assert _will_take_a_pod(_node()) is True
    assert _will_take_a_pod(_node(ready=False)) is False
    assert _will_take_a_pod(_node(unschedulable=True)) is False
    assert _will_take_a_pod(_node(taints=["node.kubernetes.io/not-ready"])) is False


def test_a_taint_that_merely_dedicates_a_node_does_not_disqualify_it() -> None:
    """A taint is not in general a statement that a node is unusable.

    A dedicated nodegroup taints itself so only pods tolerating it land there, and
    counting such a node out would under-report a cluster somebody partitioned on
    purpose. Only the two lifecycle taints mean "not taking work".
    """
    from managed_agent.adapters.kubernetes.pod_runner import _will_take_a_pod

    assert _will_take_a_pod(_node(taints=["map.dedicated"])) is True


def test_a_node_with_no_ready_condition_is_not_counted_as_ready() -> None:
    """Unknown is not available, and treating it as available is how this drifts."""
    from managed_agent.adapters.kubernetes.pod_runner import _will_take_a_pod

    bare = SimpleNamespace(
        spec=SimpleNamespace(unschedulable=False, taints=None),
        status=SimpleNamespace(conditions=None),
    )
    assert _will_take_a_pod(bare) is False


def test_the_ceiling_is_read_from_the_flag_the_process_is_running() -> None:
    """The number in force, parsed off the running Deployment's own arguments.

    The real cluster's pair is the case worth reading: the autoscaler passes 4 while the
    nodegroup declares 8, so the smaller binds, and publishing it is what turns a
    refusal into a number somebody can see.
    """
    from managed_agent.adapters.kubernetes.pod_runner import _declared_node_ceiling

    real = _deployment(["--cloud-provider=aws", "--max-nodes-total=4", "--v=4"])
    assert _declared_node_ceiling(real) == 4
    assert _declared_node_ceiling(_deployment([], ["--max-nodes-total=6"])) == 6


def test_a_flag_that_only_looks_like_the_ceiling_is_not_read_as_it() -> None:
    """Anchored and whole, so a longer flag sharing the prefix is not mistaken for it.

    A flag passed as two list entries does not match either, and neither does one whose
    value carries anything after the digits. Not matching is the right answer for a
    spelling this cannot prove it understands: the alternative is publishing a bound
    read off the leading digits of a string it did not understand, which is a wrong
    number wearing the same shape as a right one.
    """
    from managed_agent.adapters.kubernetes.pod_runner import _declared_node_ceiling

    assert _declared_node_ceiling(_deployment(["--max-nodes-total-per-zone=9"])) is None
    assert _declared_node_ceiling(_deployment(["--max-nodes-total", "4"])) is None
    assert _declared_node_ceiling(_deployment(["--max-nodes-total=4,8"])) is None
    assert _declared_node_ceiling(_deployment(["--v=4"])) is None


async def test_a_refused_cluster_read_reports_unknown_and_never_zero() -> None:
    """A 403 degrades the field; it does not fail the read and does not read as zero.

    Zero schedulable nodes is a cluster in serious trouble; "we were not allowed to
    count" is a deployment misconfiguration. Publishing the second as the first raises
    an alarm about the wrong thing at exactly the moment somebody is reading this to
    find out what is wrong.

    Refused on one half only, because the two are granted by different authority -- the
    node count is cluster-scoped and the ceiling is namespaced -- so a cluster holding
    one binding and not the other still has to answer the one it can.
    """
    from managed_agent.adapters.kubernetes import pod_runner as adapter

    runner = adapter.KubernetesPodRunner(
        namespace="map-dev", token_key=b"k", manifest={}
    )

    @asynccontextmanager
    async def refused_core() -> AsyncIterator[Any]:
        raise ApiException(status=403, reason="Forbidden")
        yield None

    @asynccontextmanager
    async def answering_apps() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            read_namespaced_deployment=_returning(_deployment(["--max-nodes-total=4"]))
        )

    with (
        patch.object(adapter, "_core_api", refused_core),
        patch.object(adapter, "_apps_api", answering_apps),
    ):
        half_refused = await runner.node_headroom()

    assert half_refused.schedulable is None, "a refusal must not read as zero nodes"
    assert half_refused.ceiling == 4, "the permitted half still has to answer"


async def test_a_flagless_autoscaler_reports_unknown_rather_than_a_bound() -> None:
    """A Deployment passing no `--max-nodes-total` caps nothing this can publish.

    Such an autoscaler is bounded only by its nodegroup's own maximum, so publishing
    anything here would publish a bound that does not bind -- and the node count, which
    was permitted, still arrives.
    """
    from managed_agent.adapters.kubernetes import pod_runner as adapter

    runner = adapter.KubernetesPodRunner(
        namespace="map-dev", token_key=b"k", manifest={}
    )

    @asynccontextmanager
    async def one_ready_node() -> AsyncIterator[Any]:
        yield SimpleNamespace(list_node=_returning(SimpleNamespace(items=[_node()])))

    @asynccontextmanager
    async def flagless_apps() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            read_namespaced_deployment=_returning(_deployment(["--v=4"]))
        )

    with (
        patch.object(adapter, "_core_api", one_ready_node),
        patch.object(adapter, "_apps_api", flagless_apps),
    ):
        answered = await runner.node_headroom()

    assert answered.schedulable == 1
    assert answered.ceiling is None


def test_the_adapter_reads_the_autoscaler_in_this_platform_s_own_namespace() -> None:
    """`map-dev`, not `kube-system`, and the difference is the reported cause.

    The upstream example installs the autoscaler in `kube-system` and this deployment
    does not -- `deploy/k8s/cluster-autoscaler.yaml` puts it in the control plane's own
    namespace. A read addressed to the wrong one answers 404, which lands in the branch
    that means "no autoscaler is deployed", so the field would go quietly empty on a
    cluster that has one. Pinned against the manifest rather than against a literal, so
    moving the Deployment fails here instead of in the cluster.
    """
    import yaml

    from managed_agent.adapters.kubernetes.pod_runner import _AUTOSCALER_DEPLOYMENT

    manifest = Path(__file__).resolve().parents[2] / "deploy" / "k8s"
    documents = list(
        yaml.safe_load_all((manifest / "cluster-autoscaler.yaml").read_text())
    )
    deployments = [
        doc
        for doc in documents
        if isinstance(doc, dict) and doc.get("kind") == "Deployment"
    ]

    assert [d["metadata"]["name"] for d in deployments] == [_AUTOSCALER_DEPLOYMENT]
    assert [d["metadata"]["namespace"] for d in deployments] == ["map-dev"]


async def test_the_ceiling_read_is_addressed_to_the_namespace_this_runner_holds() -> (
    None
):
    """The read goes to the runner's own namespace, not to a namespace named in code.

    The test above pins the Deployment's name and location in the manifest; this pins
    where the call is actually sent, and the two are separate failures. A read addressed
    to a namespace written as a literal here would answer 404 in this deployment -- and
    404 lands in the branch meaning "no autoscaler is deployed", so the ceiling would go
    quietly empty on a cluster that has one. Silence is the worst outcome available: the
    field is read during an incident, when an absent number reads as an answer.

    Built for a namespace that appears nowhere else, so a literal cannot coincide with
    it.
    """
    from managed_agent.adapters.kubernetes import pod_runner as adapter

    addressed: dict[str, object] = {}

    async def record(*, name: str, namespace: str) -> Any:
        addressed.update(name=name, namespace=namespace)
        return _deployment(["--max-nodes-total=7"])

    @asynccontextmanager
    async def recording_apps() -> AsyncIterator[Any]:
        yield SimpleNamespace(read_namespaced_deployment=record)

    @asynccontextmanager
    async def no_nodes() -> AsyncIterator[Any]:
        yield SimpleNamespace(list_node=_returning(SimpleNamespace(items=[])))

    runner = adapter.KubernetesPodRunner(
        namespace="map-elsewhere", token_key=b"k", manifest={}
    )
    with (
        patch.object(adapter, "_core_api", no_nodes),
        patch.object(adapter, "_apps_api", recording_apps),
    ):
        answered = await runner.node_headroom()

    assert addressed == {
        "name": adapter._AUTOSCALER_DEPLOYMENT,
        "namespace": "map-elsewhere",
    }
    assert answered.ceiling == 7
