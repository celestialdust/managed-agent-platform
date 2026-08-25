"""The invoker: what actually runs the two periodic sweeps, and what stops them.

Tier 0 throughout, and the app-factory cases are too: `create_async_engine` resolves a
driver and builds a pool without dialling, so `build_app` runs to the end of its wiring
against a URL nothing answers. What a database would add here is the sweeps' own
behaviour, which is graded where it belongs -- their two modules, and the real-Postgres
file beside this one.

**The defect this file is against is an absence.** Both sweeps existed and were tested;
neither was called from anywhere in `src/`, so a webhook registration was accepted and
never delivered. A test that asserted "the scheduler called sweep" on a stand-in would
have gone green over that same absence the day somebody deleted the wiring, because the
stand-in is the thing being asked. So the cases below drive the **real**
`SessionPodReaper` over a cluster that really holds pods, and every claim that a tick
happened is asserted by asking that cluster what it still holds. A tick is observable
here as a pod that is gone.

`Platform.sweeps` is a collection whose members each encode a decision -- which pass
runs, and whether two replicas may run it at once -- so it is graded per member,
parametrized over the collection, rather than by a length check that the wrong pair of
sweeps would also satisfy.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
import yaml
from fastapi import FastAPI

from managed_agent import asgi
from managed_agent.adapters.kubernetes.pod_runner import KubernetesPodRunner
from managed_agent.composition import Platform
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import (
    PlacedPod,
    PlacedPods,
    Placement,
    PodPhase,
    pod_name_for,
)
from managed_agent.control.session.reaper import IDLE_GRACE_MS, SessionPodReaper
from managed_agent.control.sweep_loop import (
    SWEEP_INTERVAL_ENV_VAR,
    Sweep,
    SweepLease,
    sweep_interval_from_env,
    sweeping,
    task_name,
)
from managed_agent.control.webhooks.dispatcher import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    WebhookDispatcher,
)
from managed_agent.core.ids import Seq, SessionId, TenantId, new_session_id
from managed_agent.core.session.session import SessionState

_NOW_MS = 1_800_000_000_000
_A_TICK_S: Final = 0.01
"""Short enough that a test waits milliseconds for a tick, not seconds.

The interval is a parameter of `sweeping` rather than a constant it reads, which is what
makes this possible at all -- and is the same reason the deployed cadence can be an
operator's number instead of this file's.
"""

_UNDIALLED: Final = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/unused"
"""A URL nothing connects to. `create_async_engine` resolves the driver and builds a
pool without dialling, so `build_app` runs to the end of its wiring offline."""

_CONTROL_PLANE: Final = (
    Path(__file__).resolve().parents[2] / "deploy" / "k8s" / "control-plane.yaml"
)

_THE_WIRED_SWEEPS: Final = (
    ("webhook-delivery", True),
    ("session-pods", False),
)
"""Every pass the control plane is supposed to run, and whether it takes a lease.

The names are written out rather than read off the constants that produce them, because
a name here is also the advisory-lock key the lease is taken under: a rename that this
file followed silently would, during a rolling restart, put the old replica and the new
one on two different locks and hand both of them the same window.

`True` for the deliveries because the claim they rest on does not exclude a second
runner -- `tests/control/test_two_replicas_sweep_without_delivering_twice.py` puts that
in front of a real database rather than asserting it from the SQL. `False` for the pods
because every verdict there is a function of state both replicas read, a handback treats
an absent pod as success, and the one branch that writes goes through
`lifecycle._end_and_release`, whose own docstring records the redundant ending event as
reachable, bounded and accepted.
"""


# --------------------------------------------------------------------------------
# Doubles. Small on purpose: the sweep driving them is the real one.
# --------------------------------------------------------------------------------


@dataclass
class TrackedCluster:
    """A cluster that really holds pods, lists them and deletes them.

    Stands in for `KubernetesPodRunner`, which does all three. `ensure` and `phase_of`
    are here because `Placement` names them and are never reached: nothing in this file
    starts a pod.

    `fails_next` is how a tick is made to raise from inside the real sweep rather than
    from a stand-in wrapped around it. A cluster read is the first thing `sweep` does,
    so a failing one is a whole pass lost -- which is the shape of the failure that
    would kill a loop with no guard around its tick.
    """

    pods: dict[str, PlacedPod] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)
    fails_next: int = 0
    listings: int = 0

    def holding(self, session_id: SessionId, *, age_ms: int) -> None:
        self.pods[pod_name_for(session_id)] = PlacedPod(
            session_id=session_id,
            phase=PodPhase.RUNNING,
            created_at_ms=_NOW_MS - age_ms,
        )

    async def placed_pods(self) -> Sequence[PlacedPod]:
        self.listings += 1
        if self.fails_next > 0:
            self.fails_next -= 1
            raise RuntimeError("the cluster API refused this listing")
        return list(self.pods.values())

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        raise AssertionError("a test in this file started a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        raise AssertionError("a test in this file located a pod")

    async def remove(self, pod_name: str) -> None:
        self.deleted.append(pod_name)
        self.pods.pop(pod_name, None)


@dataclass
class EmptyLog:
    """A log holding nothing, for every Session anybody asks about.

    That is not a shortcut, it is the branch under test: a pod older than the grace
    period whose Session has no `session.created` anywhere is residue, and the sweep
    hands it back with `NO_SESSION_OWNS_IT`. It is the shortest real path from "a tick
    ran" to "a pod is gone", which is what makes it the one these cases drive.
    """

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        raise AssertionError("this sweep appended to a log it was told was empty")

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Any]:
        return []

    def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Any]:
        raise AssertionError("the sweep followed a log")

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return Seq(1)

    async def lifecycle_events_between(
        self, types: Sequence[str], from_ms: int, to_ms: int
    ) -> Sequence[object]:
        return []


class FrozenClock:
    def now_epoch_ms(self) -> int:
        return _NOW_MS


@dataclass
class FixedLease:
    """A lease that answers the same way every time, and counts its own exits.

    The count is what makes "released on the way out" checkable without a database: an
    exit missing for every enter is a lease held for the life of the process.
    """

    mine: bool
    entered: int = 0
    left: int = 0

    @asynccontextmanager
    async def held(self, name: str) -> AsyncIterator[bool]:
        self.entered += 1
        try:
            yield self.mine
        finally:
            self.left += 1


def _platform_of(app: FastAPI) -> Platform:
    """The wired platform an app was built from, as the type rather than as `Any`.

    `app.state` is untyped, so every reader of it needs this narrowing somewhere; here
    once rather than in six cases.
    """
    platform = app.state.platform
    assert isinstance(platform, Platform)
    return platform


def _the_pod_sweep(cluster: TrackedCluster, lease: SweepLease | None = None) -> Sweep:
    """The real reaper, wrapped as the scheduler takes it."""
    log = EmptyLog()
    return Sweep(
        name="session-pods",
        run=SessionPodReaper(
            pods=cluster,
            release=Placement(cluster),
            log=log,
            events=log,
            activity=log,
            clock=FrozenClock(),
        ).sweep,
        lease=lease,
    )


async def _until(claim: Callable[[], bool], *, within_s: float = 2.0) -> None:
    """Wait for a claim about the world to come true, or fail saying it did not.

    A deadline rather than a fixed sleep: a fixed sleep long enough to be reliable is
    long enough to be slow, and one short enough to be fast is flaky. The failure
    message is the useful half -- a test that timed out here is a sweep that never ran,
    which reads nothing like an assertion that fired.
    """
    deadline = asyncio.get_running_loop().time() + within_s
    while asyncio.get_running_loop().time() < deadline:
        if claim():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"{within_s}s passed and the sweep never made this true")


# --------------------------------------------------------------------------------
# The doubles, held honest.
# --------------------------------------------------------------------------------


async def test_the_tracked_cluster_lists_and_then_forgets_what_it_deletes() -> None:
    cluster = TrackedCluster()
    session_id = new_session_id()
    cluster.holding(session_id, age_ms=IDLE_GRACE_MS * 2)

    assert [p.session_id for p in await cluster.placed_pods()] == [session_id]

    await cluster.remove(pod_name_for(session_id))

    assert await cluster.placed_pods() == []
    assert cluster.deleted == [pod_name_for(session_id)]


async def test_the_tracked_cluster_raises_for_exactly_as_many_listings_as_asked() -> (
    None
):
    """`fails_next` has to run out, or a survival case cannot tell a loop that lived
    from one that died: both would report a pod that is still there."""
    cluster = TrackedCluster(fails_next=1)
    with pytest.raises(RuntimeError):
        await cluster.placed_pods()
    assert await cluster.placed_pods() == []


# --------------------------------------------------------------------------------
# The loop: it runs, it survives, it stops.
# --------------------------------------------------------------------------------


async def test_a_running_scheduler_reclaims_a_pod_no_session_owns() -> None:
    """The whole point, stated as an outcome: the pod is gone from the cluster.

    Nothing here asserts that a method was called. Replace the reaper's handback with a
    no-op, delete the `create_task` in `sweeping`, or start the loop with its sleep
    first, and this fails on the cluster's own answer.
    """
    cluster = TrackedCluster()
    session_id = new_session_id()
    cluster.holding(session_id, age_ms=IDLE_GRACE_MS * 2)

    async with sweeping([_the_pod_sweep(cluster)], every_s=_A_TICK_S):
        await _until(lambda: cluster.deleted == [pod_name_for(session_id)])

    assert await cluster.placed_pods() == []


async def test_a_pod_placed_a_moment_ago_survives_every_tick() -> None:
    """The scheduler runs the sweep; it does not widen it.

    Without this, the case above is satisfied by a loop that deletes every pod it can
    see -- and the guard that matters most in the sweep is the one that leaves a young
    pod alone.
    """
    cluster = TrackedCluster()
    cluster.holding(new_session_id(), age_ms=IDLE_GRACE_MS // 2)

    async with sweeping([_the_pod_sweep(cluster)], every_s=_A_TICK_S):
        await _until(lambda: cluster.listings >= 3)

    assert cluster.deleted == []


async def test_a_tick_that_raises_does_not_stop_the_sweep() -> None:
    """One pass fails; the next one still reclaims.

    The failure is raised from inside the real sweep -- the cluster read it starts with
    -- rather than by a wrapper, because the exception a live loop has to survive is the
    one its own work throws.

    Remove the `except Exception` in `_ticking` and this hangs to its deadline and fails
    saying the sweep never ran, which is exactly what a dead task looks like.
    """
    cluster = TrackedCluster(fails_next=1)
    session_id = new_session_id()
    cluster.holding(session_id, age_ms=IDLE_GRACE_MS * 2)

    async with sweeping([_the_pod_sweep(cluster)], every_s=_A_TICK_S):
        await _until(lambda: cluster.deleted == [pod_name_for(session_id)])

    assert cluster.listings >= 2, "the second tick is the one that did the work"


async def test_a_failing_tick_is_logged_with_its_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Surviving is half of it; a sweep failing every tick has to be findable.

    The traceback is asserted, not just the message: a log line naming the sweep and not
    the cause sends whoever is on call back to guessing.
    """
    cluster = TrackedCluster(fails_next=1)
    with caplog.at_level("ERROR", logger="managed_agent.control.sweep_loop"):
        async with sweeping([_the_pod_sweep(cluster)], every_s=_A_TICK_S):
            await _until(lambda: cluster.listings >= 2)

    failures = [r for r in caplog.records if r.levelname == "ERROR"]
    assert failures, "a tick raised and nothing said so"
    assert "session-pods" in failures[0].getMessage()
    assert failures[0].exc_info is not None, "logged without the traceback"


async def test_the_sweep_stops_when_the_scheduler_does() -> None:
    """No tick after the block ends, which is what a cancelled task means from outside.

    Asserted by counting listings across a wait *after* the context has closed, rather
    than by inspecting a task: a task that was cancelled and a task that was left
    running both exist as objects, and only one of them keeps sweeping.
    """
    cluster = TrackedCluster()
    async with sweeping([_the_pod_sweep(cluster)], every_s=_A_TICK_S):
        await _until(lambda: cluster.listings >= 2)

    stopped_at = cluster.listings
    await asyncio.sleep(_A_TICK_S * 20)

    assert cluster.listings == stopped_at


async def test_a_tick_still_running_at_shutdown_is_cancelled_and_awaited() -> None:
    """The half that is easy to leave out: the cancel is *awaited*.

    A sweep that never returns is the case that tells the two mistakes apart. Cancel
    without awaiting and the process exits with this pass mid-flight, which asyncio
    reports as "Task was destroyed but it is pending" and which leaves a sweep half
    done; await without cancelling and the lifespan never returns at all, so this test
    hangs. Both are observed here: the cancellation is recorded from inside the pass,
    and the block is required to have exited.
    """
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def _a_pass_that_never_finishes() -> object:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return None

    loop = asyncio.get_running_loop()
    started = loop.time()
    # The outer bound only keeps a broken shutdown from hanging this file for ever; the
    # assertion that decides the case is how long the exit actually took. Without the
    # `task.cancel()` the exit blocks until this deadline fires, and the deadline's own
    # cancellation then reaches the pass -- so `cancelled` would be set either way and
    # the elapsed time is the only thing that tells the two apart.
    async with asyncio.timeout(5):
        async with sweeping(
            [Sweep(name="never-ends", run=_a_pass_that_never_finishes, lease=None)],
            every_s=_A_TICK_S,
        ):
            await entered.wait()
    took = loop.time() - started

    assert cancelled.is_set(), (
        "the sweep in flight was not cancelled; a shutdown that only stops the loop "
        "between ticks leaves this pass running into process exit"
    )
    assert took < 1.0, (
        f"the lifespan took {took:.1f}s to exit over a pass that never returns, so "
        "nothing cancelled it -- what ended this was the outer deadline"
    )


async def test_one_slow_sweep_does_not_hold_up_the_other() -> None:
    """One task per sweep, and this is the property that buys.

    A single task walking the list would make the pod sweep's cadence a function of how
    badly some tenant's endpoint is behaving.
    """
    cluster = TrackedCluster()
    session_id = new_session_id()
    cluster.holding(session_id, age_ms=IDLE_GRACE_MS * 2)

    async def _a_pass_that_never_finishes() -> object:
        await asyncio.Event().wait()
        return None

    sweeps = [
        Sweep(name="never-ends", run=_a_pass_that_never_finishes, lease=None),
        _the_pod_sweep(cluster),
    ]
    async with sweeping(sweeps, every_s=_A_TICK_S):
        await _until(lambda: cluster.deleted == [pod_name_for(session_id)])


# --------------------------------------------------------------------------------
# The lease: a sweep that declares one runs only where it holds it.
# --------------------------------------------------------------------------------


async def test_a_sweep_that_holds_its_lease_runs() -> None:
    cluster = TrackedCluster()
    session_id = new_session_id()
    cluster.holding(session_id, age_ms=IDLE_GRACE_MS * 2)
    lease = FixedLease(mine=True)

    async with sweeping([_the_pod_sweep(cluster, lease)], every_s=_A_TICK_S):
        await _until(lambda: cluster.deleted == [pod_name_for(session_id)])


async def test_a_sweep_that_loses_its_lease_does_nothing_and_keeps_ticking() -> None:
    """The replica that did not win: no pass, no error, and it tries again next tick.

    Both halves matter. If losing the lease raised, the log would fill with a failure
    that is the normal state of one of two replicas; if losing it ended the loop, a
    replica that lost once would never sweep again -- and the one that always wins is
    the one that will be restarted first.
    """
    cluster = TrackedCluster()
    cluster.holding(new_session_id(), age_ms=IDLE_GRACE_MS * 2)
    lease = FixedLease(mine=False)

    async with sweeping([_the_pod_sweep(cluster, lease)], every_s=_A_TICK_S):
        await _until(lambda: lease.entered >= 3)

    assert cluster.listings == 0, "the sweep ran without holding its lease"
    assert cluster.deleted == []


async def test_the_lease_is_left_every_time_it_is_taken() -> None:
    """A lease kept past its tick stops every replica, not just this one."""
    lease = FixedLease(mine=True)
    cluster = TrackedCluster(fails_next=1)

    async with sweeping([_the_pod_sweep(cluster, lease)], every_s=_A_TICK_S):
        await _until(lambda: lease.entered >= 3)

    assert lease.left == lease.entered, (
        f"{lease.entered} leases taken and {lease.left} released, including the tick "
        "whose pass raised"
    )


# --------------------------------------------------------------------------------
# The interval: configured, or the process does not start.
# --------------------------------------------------------------------------------


def test_the_interval_is_read_from_the_environment() -> None:
    assert sweep_interval_from_env({SWEEP_INTERVAL_ENV_VAR: "30"}) == 30


def test_an_absent_interval_raises_naming_the_variable() -> None:
    """No default, so a deployment that forgot it fails at start-up rather than sweeping
    on a cadence nobody chose."""
    with pytest.raises(KeyError) as absent:
        sweep_interval_from_env({})
    assert absent.value.args[0] == SWEEP_INTERVAL_ENV_VAR


@pytest.mark.parametrize("raw", ["0", "-1", "", "x", "30s", "1.5"])
def test_an_interval_that_is_not_a_positive_whole_number_of_seconds_is_refused(
    raw: str,
) -> None:
    """Zero and negative are refused for the same reason as unparseable: each is a value
    an operator typed that produces a process nobody asked for -- here a loop that
    sweeps continuously for as long as it is up."""
    with pytest.raises(ValueError, match=SWEEP_INTERVAL_ENV_VAR):
        sweep_interval_from_env({SWEEP_INTERVAL_ENV_VAR: raw})


# --------------------------------------------------------------------------------
# The wiring: what the composition root hands over, and what starts it.
# --------------------------------------------------------------------------------


class AbsentCluster:
    """A cluster client that answers a phase, starts nothing, and can be enumerated.

    Four methods rather than the three a `PodRunner` double needs, because the fourth is
    what the composition root narrows on -- a three-method double reaches `build` and
    gets no pod sweep, which is the case the last test in this section covers.
    """

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        return PodPhase.ABSENT

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        return None

    async def placed_pods(self) -> Sequence[PlacedPod]:
        return []


class UnenumerableCluster:
    """The three-method `PodRunner` every other double in this suite is."""

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        return PodPhase.ABSENT

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        return None


@pytest.fixture
def a_placer_control_plane(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """The platform a deployed control plane builds, with a cluster that starts nothing.

    Every variable the placer's path reads is a stand-in: nothing in this section is
    about their values, only about what the wiring on the other side of them is.
    """
    monkeypatch.setenv("DATABASE_URL", _UNDIALLED)
    monkeypatch.setenv(SWEEP_INTERVAL_ENV_VAR, "30")
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    monkeypatch.setenv("MAP_SESSION_TOKEN_KEY", "a session-token signing key")
    monkeypatch.setenv("MAP_SESSION_TOKEN_LIFETIME_S", "3600")
    monkeypatch.setenv("MAP_TOOL_GATEWAY_URL", "http://tool-gateway.map-test/mcp")
    monkeypatch.setenv("MAP_MODEL_GATEWAY_URL", "http://model-gateway.map-test/v1")
    monkeypatch.setattr(asgi, "pod_runner_from_environment", AbsentCluster)
    return asgi.build_app()


@pytest.mark.parametrize(("name", "leased"), _THE_WIRED_SWEEPS)
def test_the_root_wires_this_sweep_and_answers_the_two_replica_question_for_it(
    name: str, leased: bool, a_placer_control_plane: FastAPI
) -> None:
    """Per member, because each member is two decisions and a count is neither.

    `len(sweeps) == 2` passes over the wrong pair, over one sweep wired twice, and over
    a delivery sweep that lost its lease in a refactor -- and the last of those is a
    duplicate callback to every tenant with nothing failing anywhere.
    """
    wired = {sweep.name: sweep for sweep in _platform_of(a_placer_control_plane).sweeps}
    assert name in wired, f"the root wired {sorted(wired)} and not {name}"
    assert (wired[name].lease is not None) is leased


def test_the_root_wires_no_sweep_it_was_not_asked_for(
    a_placer_control_plane: FastAPI,
) -> None:
    """The other direction of the case above: the parametrized set is the whole set."""
    wired = _platform_of(a_placer_control_plane).sweeps
    assert sorted(s.name for s in wired) == sorted(
        name for name, _ in _THE_WIRED_SWEEPS
    )


@pytest.mark.parametrize(("name", "_leased"), _THE_WIRED_SWEEPS)
async def test_the_lifespan_runs_this_sweep_while_serving_and_not_after(
    name: str, _leased: bool, a_placer_control_plane: FastAPI
) -> None:
    """A control plane that is serving requests is a control plane that is sweeping.

    The task is looked for by name because that is the only thing about a running sweep
    an outsider can see. Both edges are asserted from one enter/exit, since a sweep
    started and never stopped and a sweep never started are different defects that a
    single check for either one would confuse.
    """
    app = a_placer_control_plane
    running = app.router.lifespan_context(app)
    await running.__aenter__()
    try:
        during = {t.get_name() for t in asyncio.all_tasks()}
    finally:
        await running.__aexit__(None, None, None)
    after = {t.get_name() for t in asyncio.all_tasks()}

    assert task_name(name) in during, f"the lifespan started {sorted(during)}"
    assert task_name(name) not in after


async def test_a_control_plane_with_no_cluster_still_starts_and_still_delivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional capability stays optional.

    A sweep that required a cluster would make cluster access a start-up condition for
    the whole control plane -- so this asserts both halves: the process builds, and the
    pass that needs nothing from a cluster is still wired and still started.
    """
    monkeypatch.setenv("DATABASE_URL", _UNDIALLED)
    monkeypatch.setenv(SWEEP_INTERVAL_ENV_VAR, "30")
    monkeypatch.setattr(asgi, "pod_runner_from_environment", lambda: None)

    app = asgi.build_app()
    assert [s.name for s in _platform_of(app).sweeps] == ["webhook-delivery"]

    running = app.router.lifespan_context(app)
    await running.__aenter__()
    try:
        assert task_name("webhook-delivery") in {
            t.get_name() for t in asyncio.all_tasks()
        }
    finally:
        await running.__aexit__(None, None, None)


def test_a_cluster_client_that_cannot_be_enumerated_wires_no_pod_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the narrowing in the composition root actually skips.

    Stated as its own case rather than left implicit, because the skip is silent: a
    runner with three methods gets a control plane that places pods and never reclaims
    them. The case below is what keeps that from being reachable in a deployment.
    """
    monkeypatch.setenv("DATABASE_URL", _UNDIALLED)
    monkeypatch.setenv(SWEEP_INTERVAL_ENV_VAR, "30")
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    monkeypatch.setenv("MAP_SESSION_TOKEN_KEY", "a session-token signing key")
    monkeypatch.setenv("MAP_SESSION_TOKEN_LIFETIME_S", "3600")
    monkeypatch.setenv("MAP_TOOL_GATEWAY_URL", "http://tool-gateway.map-test/mcp")
    monkeypatch.setenv("MAP_MODEL_GATEWAY_URL", "http://model-gateway.map-test/v1")
    monkeypatch.setattr(asgi, "pod_runner_from_environment", UnenumerableCluster)

    wired = _platform_of(asgi.build_app()).sweeps
    assert [s.name for s in wired] == ["webhook-delivery"]


def test_the_cluster_client_a_deployment_wires_can_be_enumerated() -> None:
    """So the narrowing above never skips the pod sweep in production.

    Asserted against the class the composition root constructs rather than against an
    instance, because building one needs a manifest and a cluster and this claim needs
    neither: what is being checked is that the client has the method the sweep reads.
    A `PodRunner` that stopped having it would take the pod sweep out of every
    deployment and no other case in this tree would notice.
    """
    assert issubclass(KubernetesPodRunner, PlacedPods)


def test_the_deployed_manifest_names_the_interval_the_factory_refuses_to_start_without(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alarm and the thing it alarms about, in one case.

    This repository has already paid for a guard wired behind the absence it was
    guarding. So both halves are asserted here rather than in two files: the factory
    really does refuse without the variable, and the manifest that runs that factory
    really does carry it. Either alone is green while a deployment CrashLoops.
    """
    monkeypatch.setenv("DATABASE_URL", _UNDIALLED)
    monkeypatch.delenv(SWEEP_INTERVAL_ENV_VAR, raising=False)
    monkeypatch.setattr(asgi, "pod_runner_from_environment", lambda: None)
    with pytest.raises(KeyError) as absent:
        asgi.build_app()
    assert absent.value.args[0] == SWEEP_INTERVAL_ENV_VAR

    documents = [d for d in yaml.safe_load_all(_CONTROL_PLANE.read_text()) if d]
    container = documents[0]["spec"]["template"]["spec"]["containers"][0]
    declared = {entry["name"]: entry for entry in container["env"]}
    assert SWEEP_INTERVAL_ENV_VAR in declared, (
        f"{_CONTROL_PLANE.name} runs a factory that reads this variable and declares "
        f"{sorted(declared)}"
    )
    assert (
        sweep_interval_from_env(
            {SWEEP_INTERVAL_ENV_VAR: str(declared[SWEEP_INTERVAL_ENV_VAR]["value"])}
        )
        > 0
    )


# --------------------------------------------------------------------------------
# The delivery half, driven by the real dispatcher: a registration that was accepted
# and never delivered is the defect this whole file exists for.
# --------------------------------------------------------------------------------

_A_SIGNING_SECRET: Final = "a secret only the receiver and this platform hold"


@dataclass(frozen=True, slots=True)
class Owed:
    """A callback an earlier pass claimed and did not deliver, shaped like the join's
    row.

    The tenant travels on it because the retry composes its own vault key from the
    tenant and the reference, exactly as the first attempt did.
    """

    webhook_id: UUID
    tenant_id: TenantId
    url: str
    secret_ref: str
    session_id: SessionId
    state: SessionState
    seq: Seq


@dataclass
class Ledger:
    """The claim and the watermark, with the claim's real rule.

    An attempt is available while the callback is undelivered and under the cap, which
    is the rule the SQL implements and the rule
    `tests/control/test_two_replicas_sweep_without_delivering_twice.py` puts in front of
    a real database. A fake that merely counted calls would let a broken dispatcher
    deliver the same callback on every tick and this file would go green over it.

    `watermark` is set past the present on purpose, which closes the window pass and
    leaves the retry pass as the only thing that can post. That is the shorter of the
    two real routes to a delivery and it needs no clock to line up: the window's
    frontier sits five seconds behind now, so a test that wanted the other route would
    have to wait for it.
    """

    owed: dict[tuple[UUID, SessionId, str], Owed] = field(default_factory=dict)
    attempts: dict[tuple[UUID, SessionId, str], int] = field(default_factory=dict)
    delivered: dict[tuple[UUID, SessionId, str], int] = field(default_factory=dict)
    watermark: int = _NOW_MS * 2

    def already_claimed(self, row: Owed) -> None:
        key = (row.webhook_id, row.session_id, row.state.value)
        self.owed[key] = row
        self.attempts[key] = 1

    async def scanned_through_ms(self) -> int:
        return self.watermark

    async def advance_scan_to(self, at_ms: int) -> None:
        self.watermark = max(self.watermark, at_ms)

    async def claim(
        self,
        webhook_id: UUID,
        session_id: SessionId,
        state: SessionState,
        seq: Seq,
        max_attempts: int,
    ) -> int | None:
        key = (webhook_id, session_id, state.value)
        if key in self.delivered or self.attempts.get(key, 0) >= max_attempts:
            return None
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return self.attempts[key]

    async def mark_delivered(
        self, webhook_id: UUID, session_id: SessionId, state: SessionState, status: int
    ) -> None:
        key = (webhook_id, session_id, state.value)
        self.delivered[key] = status
        self.owed.pop(key, None)

    async def undelivered(self, max_attempts: int, limit: int) -> Sequence[Owed]:
        return [
            row
            for key, row in self.owed.items()
            if key not in self.delivered and self.attempts[key] < max_attempts
        ][:limit]


class UnreadScan:
    """The tail, which a closed window must never reach.

    Raising rather than returning nothing: an empty answer here would look the same as a
    window that was never opened, and the point of the watermark above is that it was
    never opened.
    """

    async def lifecycle_events_between(
        self, types: Sequence[str], from_ms: int, to_ms: int
    ) -> Sequence[Any]:
        raise AssertionError("the sweep read the tail over a closed window")

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Any]:
        raise AssertionError("the sweep folded a Session it had no candidate for")


@dataclass(frozen=True, slots=True)
class OneSecret:
    """A vault holding the signing secret under every name it is asked for."""

    async def fetch(self, name: str) -> str:
        return _A_SIGNING_SECRET


@dataclass
class Receiver:
    """A tenant's endpoint. Records what arrived and answers 200."""

    sent: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(request)
        return httpx.Response(200)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def _the_delivery_sweep(ledger: Ledger, receiver: Receiver) -> Sweep:
    """The real dispatcher, wrapped as the composition root wraps it."""
    dispatcher = WebhookDispatcher(
        UnwatchedRegistrations(), ledger, UnreadScan(), OneSecret(), receiver.client()
    )

    async def _run() -> object:
        return await dispatcher.sweep_once(_NOW_MS)

    return Sweep(name="webhook-delivery", run=_run, lease=None)


class UnwatchedRegistrations:
    """The registrations, which only the window pass consults. Closed window, no reads.

    A `[]` here would be a second way for a delivery to be missing, and this file has to
    be able to tell "nothing was owed" from "nothing was found".
    """

    async def watching(self, tenant_id: TenantId, state: SessionState) -> Sequence[Any]:
        raise AssertionError("the sweep matched registrations over a closed window")


def _an_owed_callback() -> Owed:
    return Owed(
        webhook_id=uuid4(),
        tenant_id=TenantId(uuid4()),
        url="https://hooks.example.com/a-tenants-endpoint",
        secret_ref="signing-a",
        session_id=new_session_id(),
        state=SessionState.STOPPED,
        seq=Seq(7),
    )


async def test_the_fake_ledger_stops_granting_a_claim_once_it_is_delivered() -> None:
    """The fake held honest before a delivery case leans on it.

    Without this rule the dispatcher would be handed a claim on every tick and the
    delivery case below would pass over a sweep that posts for ever.
    """
    ledger = Ledger()
    row = _an_owed_callback()
    key = (row.webhook_id, row.session_id, row.state.value)
    ledger.already_claimed(row)

    assert await ledger.claim(*key[:2], row.state, row.seq, 5) == 2
    await ledger.mark_delivered(row.webhook_id, row.session_id, row.state, 200)
    assert await ledger.claim(*key[:2], row.state, row.seq, 5) is None
    assert await ledger.undelivered(5, 100) == []


async def test_a_running_scheduler_delivers_the_callback_a_tenant_registered_for() -> (
    None
):
    """The worse half of the defect, as an outcome: the tenant's endpoint is called.

    Nothing here asserts that a method ran. The signature is recomputed from the stdlib
    over the bytes that actually arrived, so a delivery that reached the receiver with
    the wrong body, the wrong timestamp or no signature fails -- and the ledger is asked
    whether the callback is now delivered, so a post that was never recorded fails too.

    Remove the `create_task` in `sweeping`, or the sweep from `Platform.sweeps`, and
    this fails to its deadline with an empty receiver -- which is exactly the state the
    platform was in.
    """
    ledger, receiver = Ledger(), Receiver()
    row = _an_owed_callback()
    ledger.already_claimed(row)
    sweep = _the_delivery_sweep(ledger, receiver)

    async with sweeping([sweep], every_s=_A_TICK_S):
        await _until(lambda: len(receiver.sent) >= 1)

    posted = receiver.sent[0]
    assert str(posted.url) == row.url
    digest = hmac.new(
        _A_SIGNING_SECRET.encode(),
        f"{posted.headers[TIMESTAMP_HEADER]}.".encode() + posted.content,
        hashlib.sha256,
    ).hexdigest()
    assert posted.headers[SIGNATURE_HEADER] == f"v1={digest}"
    assert ledger.delivered == {(row.webhook_id, row.session_id, row.state.value): 200}


async def test_a_delivered_callback_is_not_posted_again_on_the_next_tick() -> None:
    """A loop that ran for ever over a delivered row would call a tenant every tick.

    The interval here is what makes this worth asserting: a scheduler is not one pass,
    and every dedup rule in the dispatcher is a rule about being called repeatedly.
    """
    ledger, receiver = Ledger(), Receiver()
    ledger.already_claimed(_an_owed_callback())
    sweep = _the_delivery_sweep(ledger, receiver)

    async with sweeping([sweep], every_s=_A_TICK_S):
        await _until(lambda: len(receiver.sent) >= 1)
        await asyncio.sleep(_A_TICK_S * 20)

    assert len(receiver.sent) == 1, (
        f"the tenant's endpoint was called {len(receiver.sent)} times for one callback"
    )
