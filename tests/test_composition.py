"""One place constructs a concrete adapter, and what it hands back really works.

Tier 1 (testcontainers, real PostgreSQL 17). Two properties: the Platform `build()`
returns is wired to adapters that write real rows, and no second place in the tree
constructs the engine — the second is a grep over the source, because the rule it
protects is "exactly one composition root" and a rule about the whole tree cannot be
checked from inside one module.
"""

import ast
import asyncio
import os
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent
from sqlalchemy.pool import QueuePool

from managed_agent.adapters.s3.evidence_store import EvidenceStorageUnconfigured
from managed_agent.composition import (
    Platform,
    build,
    pod_runner_from_environment,
)
from managed_agent.control.files.store import UPLOAD_BUCKET_ENV_VAR
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.lifecycle import NoSessionPods
from managed_agent.control.session.placement import PodPhase, pod_name_for
from managed_agent.control.session.threads import NoSessionThreads
from managed_agent.control.session.turn_dispatch import (
    NoPodTransport,
    TurnUndeliverable,
)
from managed_agent.control.skills.inventory import NoSkillInventory
from managed_agent.core.ids import new_session_id, new_turn_id
from managed_agent.core.ports import EventLogAppend, EventLogRange
from managed_agent.core.vfs.evidence import (
    THRESHOLD_ENV_VAR,
    CaptureThreshold,
    threshold_from_env,
)
from managed_agent.session_shim.pod_channel import HttpPodDispatch

_SRC = Path(__file__).parents[1] / "src"

# A URL nothing connects to. `create_async_engine` resolves the driver and builds a pool
# without dialling, and the cases below are about which transport `build` chose, so
# starting a container for them would buy nothing.
_UNDIALLED = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/unused"


async def test_build_returns_ports_bound_to_adapters_that_write(
    database_url: str,
) -> None:
    platform, engine = build(database_url)
    try:
        session_id = new_session_id()
        seq = await platform.event_log_append.append(
            session_id, "turn.started", {"from": "composition"}
        )
        written = await platform.event_log_range.read(session_id, seq, seq)
    finally:
        await engine.dispose()

    assert seq == 1
    assert [row.type for row in written] == ["turn.started"]
    assert [row.payload for row in written] == [{"from": "composition"}]


async def test_build_wires_a_real_thread_index_rather_than_the_stand_in() -> None:
    """The one field on `Platform` whose absence cannot fail, so nothing catches it.

    `NoSessionThreads` answers every read as though the Session had no threads, which is
    correct for a process with no store and indistinguishable from the truth about a
    Session that never delegated. So a composition root that forgot this field would
    serve the whole thread surface as empty pages and 404s, with every offline route
    test still passing -- they put a fake behind the port. This asserts the wiring
    itself, which is the only place the mistake would be visible.
    """
    platform, engine = build(_UNDIALLED)
    try:
        assert not isinstance(platform.session_threads, NoSessionThreads)
    finally:
        await engine.dispose()


async def test_build_wires_a_real_skill_inventory_rather_than_the_stand_in() -> None:
    """Asserted at the wiring, because a fake behind the port hides the mistake.

    Every offline test of the skill listing puts its own inventory behind this field, so
    all of them pass whether or not `build` fills it in -- which is exactly how two
    route modules answered 404 through the real app for a whole wave while their own
    tests were green. This is the one place a missing line is visible.

    Unlike the thread index above, forgetting this one would fail loudly at runtime
    rather than serving empty pages: `NoSkillInventory` refuses instead of answering
    empty. That makes the deployment's failure honest and this test's job smaller, but
    not zero -- a surface that refuses every listing is still a surface nobody can use,
    and finding out from a tenant is worse than finding out here.
    """
    platform, engine = build(_UNDIALLED)
    try:
        assert not isinstance(platform.skill_inventory, NoSkillInventory)
    finally:
        await engine.dispose()


def test_the_platform_exposes_the_two_log_ports_by_their_abstractions() -> None:
    annotations = Platform.__annotations__
    assert annotations["event_log_append"] is EventLogAppend
    assert annotations["event_log_range"] is EventLogRange


def test_build_reads_the_database_url_from_the_environment_when_given_none(
    monkeypatch: pytest.MonkeyPatch, database_url: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    platform, engine = build()
    assert isinstance(platform, Platform)
    assert engine.url.database is not None


def test_build_refuses_to_guess_a_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No default, no localhost fallback: an unconfigured process must not start."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(KeyError):
        build()


def test_only_the_composition_root_constructs_an_engine() -> None:
    offenders = [
        str(module.relative_to(_SRC))
        for module in _SRC.rglob("*.py")
        if module.name != "composition.py"
        and "create_async_engine" in module.read_text()
    ]
    assert offenders == [], f"a second composition root exists in {offenders}"


def test_nothing_under_core_imports_an_adapter() -> None:
    """The same rule as ruff's TID251 ban, checked without reading ruff's config.

    Kept alongside `ruff check .` rather than folded into it, and the difference is
    where each one can be switched off. The ban lives in `[tool.ruff.lint.per-file-
    ignores]`, where `core/**` is one line away from being exempt -- and once it is,
    `ruff check .` passes on a core module that imports an adapter and reports
    nothing. This walks the tree instead, so the only way to make it pass is for the
    import not to be there.

    It is a narrow gap and worth being honest about how narrow: an import from core
    into an adapter that imports core back is circular, and Python refuses it before
    any check here runs. What is left is the one-directional case -- a future adapter
    that does not import core -- which is exactly the case a config exemption would
    let through.
    """
    for module in (_SRC / "managed_agent" / "core").rglob("*.py"):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("managed_agent.adapters"), module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("managed_agent.adapters"), module


def test_the_composition_root_is_the_only_module_named_as_one() -> None:
    roots = [path for path in _SRC.rglob("composition.py")]
    assert len(roots) == 1, [str(path) for path in roots]
    assert os.path.basename(str(roots[0].parent)) == "managed_agent"


async def test_the_pool_itself_covers_the_planned_concurrency(
    database_url: str,
) -> None:
    """`pool_size` reaches the capacity plan, not just `pool_size + max_overflow`.

    The guard for a measured cliff, and the reason it checks `size()` alone. A
    connection handed out above `pool_size` is created for that one checkout and closed
    on return, so overflow does not stand in for pool: 50 concurrent appends measured
    132 ms at `size=40, overflow=10` and 20.5 ms at `size=50, overflow=0` -- the same
    ceiling of 50, 6x apart. An engine sized by its ceiling looks correct and is not,
    which is exactly the kind of defect a comment cannot hold on its own.

    50 is ADR-029's M1 plan for concurrent live Sessions. If that plan
    moves, this number moves with it -- the assertion is that the two agree, not that
    the number is 50 forever.
    """
    planned_concurrent_sessions = 50
    _, engine = build(database_url)
    try:
        pool = engine.pool
        # Narrowed rather than reached through the base class, and the narrowing is part
        # of the check: only a queueing pool has a size at all. A NullPool or a
        # StaticPool would satisfy `engine.pool` and keep no connections between
        # requests.
        assert isinstance(pool, QueuePool), (
            f"pool is a {type(pool).__name__}, which does not keep connections between "
            "checkouts; every append would open its own"
        )
        assert pool.size() >= planned_concurrent_sessions, (
            f"pool_size is {pool.size()}, below the "
            f"{planned_concurrent_sessions} concurrent Sessions ADR-029 "
            "provisions for; the shortfall is served by overflow connections that are "
            "opened and closed on every use"
        )
    finally:
        await engine.dispose()


def _the_placers_four_other_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set what `build` reads once it is handed a pod runner, so these cases reach it.

    A process with a runner compiles Session configurations, and the four values that
    takes -- both gateway addresses, the Session-token signing key and a token lifetime
    -- have no defaults on purpose. Set here as stand-ins, because nothing in this file
    is about their values; what is graded is the wiring on the other side of them.

    Not shared with the other files that need the same four lines. Two identical
    fixtures are a coincidence, and a shared module for them would couple files that
    are graded independently -- the same argument this repository already made about
    the `AbsentPod` doubles.
    """
    monkeypatch.setenv("MAP_SESSION_TOKEN_KEY", "a session-token signing key")
    monkeypatch.setenv("MAP_SESSION_TOKEN_LIFETIME_S", "3600")
    monkeypatch.setenv("MAP_TOOL_GATEWAY_URL", "http://tool-gateway.map-test/mcp")
    monkeypatch.setenv("MAP_MODEL_GATEWAY_URL", "http://model-gateway.map-test/v1")


class AbsentPod:
    """A cluster client that answers a phase and starts nothing.

    Stands in for the adapter nothing in this tree implements. That absence is the whole
    reason `build` takes this as an argument instead of constructing one: a `Placement`
    wired over a runner that cannot answer would fail at the first Turn rather than at
    start-up.

    The phase is now a constructor argument, and ABSENT is no longer the phase to
    dispatch against. A Turn that finds no pod places one, so ABSENT reaches the
    compilation path and the database behind it; GONE is refused by the transport
    itself, which is what the wiring case below is actually about. The default is
    unchanged so the sweep and the wiring cases that never dispatch are untouched.
    """

    def __init__(self, phase: PodPhase = PodPhase.ABSENT) -> None:
        self._phase = phase
        self.removed: list[str] = []

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        return self._phase

    async def phase_of(self, pod_name: str) -> PodPhase:
        return self._phase

    async def remove(self, pod_name: str) -> None:
        self.removed.append(pod_name)


async def test_a_platform_built_without_a_pod_runner_refuses_every_turn() -> None:
    """The default is a refusal, and it is honest rather than provisional.

    Nothing here can locate a Session's pod, so a dispatch that reported success would
    put a Turn in the Event Log that nothing ever ran.
    """
    platform, engine = build(_UNDIALLED)
    try:
        assert isinstance(platform.turn_dispatch, NoPodTransport)
        with pytest.raises(TurnUndeliverable):
            await platform.turn_dispatch.dispatch(
                new_session_id(), new_turn_id(), "summarise the findings"
            )
    finally:
        await engine.dispose()


async def test_a_platform_built_with_a_pod_runner_wires_no_nopodtransport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given something that can find a pod, the real transport is on the wired path.

    What this does not show is that a pod ever answers: no image in this tree builds
    `map-session` and the runner above is a stand-in, so this grades the wiring and
    nothing beyond it.

    Dispatched against a pod the cluster reports GONE rather than one it reports
    ABSENT, and the difference is the point: an absent pod is now *placed* -- a Turn
    that finds none compiles a configuration and creates one -- so ABSENT would send
    this case through the compilation path and into the undialled database, which is a
    different claim from the one it makes. GONE is still refused by the transport, and
    the refusal is what says the real transport rather than `NoPodTransport` answered.
    """
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    _the_placers_four_other_variables(monkeypatch)
    platform, engine = build(_UNDIALLED, pod_runner=AbsentPod(PodPhase.GONE))
    try:
        assert isinstance(platform.turn_dispatch, HttpPodDispatch)
        assert not isinstance(platform.turn_dispatch, NoPodTransport)
        with pytest.raises(TurnUndeliverable, match="is gone"):
            await platform.turn_dispatch.dispatch(
                new_session_id(), new_turn_id(), "summarise the findings"
            )
    finally:
        await engine.dispose()


async def test_a_platform_built_with_a_pod_runner_can_give_a_pod_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Releasing reaches the cluster client, and the default reaches nothing.

    Graded by what happens to the pod rather than by which object is attached. A field
    holding something of the right type that reaches no cluster looks identical from
    outside to a field nothing wired at all, and the difference shows up only much
    later, as a slot that never comes back. Both arms are here because the refusing
    default is itself a decision: a control plane with no placer must not report that
    it released a pod.
    """
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    _the_placers_four_other_variables(monkeypatch)
    session_id = new_session_id()
    runner = AbsentPod(PodPhase.RUNNING)
    platform, engine = build(_UNDIALLED, pod_runner=runner)
    try:
        await platform.session_pod_release.release(session_id)
    finally:
        await engine.dispose()
    assert runner.removed == [pod_name_for(session_id)]

    bare, bare_engine = build(_UNDIALLED)
    try:
        assert isinstance(bare.session_pod_release, NoSessionPods)
        await bare.session_pod_release.release(session_id)
    finally:
        await bare_engine.dispose()


def test_building_with_a_pod_runner_refuses_to_guess_a_signing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No default, for the reason `DATABASE_URL` has none.

    An empty signing key derives a token every pod can also derive, and the shim route's
    only check then passes for anything on the cluster network.
    """
    monkeypatch.delenv("MAP_SHIM_TOKEN_KEY", raising=False)
    _the_placers_four_other_variables(monkeypatch)
    with pytest.raises(KeyError, match="MAP_SHIM_TOKEN_KEY"):
        build(_UNDIALLED, pod_runner=AbsentPod())


_POD_RUNNER_METHODS = frozenset({"ensure", "phase_of", "remove"})


def _pod_runners_in(source: str) -> list[str]:
    """Every class in `source` that structurally satisfies `placement.PodRunner`.

    Structural rather than by base class, because `PodRunner` is a Protocol nothing
    inherits from: `AbsentPod` above satisfies it and names it nowhere, so a sweep
    looking for a base class would find neither it nor a real adapter. The Protocol's
    own declaration is skipped -- it defines all three methods by definition.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        if any(
            isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases
        ):
            continue
        methods = {
            member.name
            for member in node.body
            if isinstance(member, ast.AsyncFunctionDef | ast.FunctionDef)
        }
        if methods >= _POD_RUNNER_METHODS:
            found.append(node.name)
    return found


def test_the_sweep_finds_a_pod_runner_when_one_is_there() -> None:
    """The positive control for the case below.

    `AbsentPod` in this file satisfies `PodRunner` and names it nowhere, so a sweep that
    returned nothing here would return nothing over `src/` too and the case below would
    pass by looking at the wrong thing.
    """
    assert _pod_runners_in(Path(__file__).read_text()) == ["AbsentPod"]


def test_the_root_hands_over_an_implementation_now_that_the_tree_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep and the wiring have to agree, and this is what makes them.

    This case replaces the one that asserted None. That one held while nothing under
    `src/` implemented `PodRunner` and was written to fail the moment something did --
    which is what happened, so what it guarded now needs saying the other way round: an
    implementation exists, and a configured process must actually be handed one rather
    than being wired to `NoPodTransport` with a working adapter sitting unused beside
    it. That is the failure the original was aimed at, and it is the one that survived a
    whole slice before.

    `MAP_POD_MANIFEST` is what a deployment sets. Which variables are required, and what
    an unconfigured process gets instead, is `tests/adapters/test_pod_runner.py`'s --
    this asserts only that the root does not answer None while the tree can answer.
    """
    implementations = sorted(
        name
        for module in _SRC.rglob("*.py")
        for name in _pod_runners_in(module.read_text())
    )
    assert implementations, (
        "nothing under src/ implements PodRunner any more; if that is deliberate this "
        "case has to go back to asserting the root answers None"
    )
    monkeypatch.setenv(
        "MAP_POD_MANIFEST",
        str(Path(__file__).parents[1] / "deploy" / "k8s" / "session-pod.yaml"),
    )
    monkeypatch.setenv("MAP_NAMESPACE", "map-test")
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    assert pod_runner_from_environment() is not None, (
        f"{implementations} implements PodRunner now and the composition root still "
        "answers None: fill in its body"
    )


def test_the_platform_exposes_one_evidence_capture_reading_the_one_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The threshold is read here and nowhere else.

    Both capture points -- the Gateway's and the shim's -- get their threshold from this
    one construction. If either read the variable for itself, Evidence coverage would
    silently become a function of which tool ran, which is the one property a size-based
    rule exists to remove.
    """
    monkeypatch.setenv(THRESHOLD_ENV_VAR, "4096")
    platform, engine = build(_UNDIALLED)
    assert platform.evidence_capture.threshold == threshold_from_env()
    assert platform.evidence_capture.threshold == CaptureThreshold(4096)


def test_an_unconfigured_bucket_leaves_a_capture_point_that_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no bucket there is nowhere to put a payload, and the refusal is loud.

    The alternative is the one outcome this whole slice exists to prevent: a large
    result handed to the model with no Evidence written and nothing saying so. A failed
    tool call is recoverable; a missing audit record is not.
    """
    monkeypatch.delenv(UPLOAD_BUCKET_ENV_VAR, raising=False)
    platform, engine = build(_UNDIALLED)
    with pytest.raises(EvidenceStorageUnconfigured, match=UPLOAD_BUCKET_ENV_VAR):
        asyncio.run(_capture_something_large(platform))


async def _capture_something_large(platform: Platform) -> None:
    await platform.evidence_capture.apply(
        new_session_id(),
        "call-1",
        "acme__search",
        CallToolResult(content=[TextContent(type="text", text="x" * 200_000)]),
    )


def test_only_the_composition_root_constructs_the_evidence_blobs() -> None:
    """Invariant I13: the one place that knows the object store is S3."""
    offenders = [
        str(module.relative_to(_SRC))
        for module in _SRC.rglob("*.py")
        if module.name not in ("composition.py", "evidence_store.py")
        and "S3EvidenceBlobs(" in module.read_text()
    ]
    assert offenders == [], f"a second place constructs the evidence blobs: {offenders}"
