"""Whether a deployment that is supposed to place Session pods can.

Tier 1 for everything but the last three cases, which read the live cluster and are
marked `network`. SKIPPED MEANS THE CLUSTER WAS NOT ASKED.

The defect this file exists for: a control plane ran in map-dev accepting Sessions and
placing nothing, and every check in this suite was green throughout. The manifest's own
comment promised the process would CrashLoop the day something implemented PodRunner --
and it did not, because a second absence kept the first from being reached. An alarm
wired behind the thing it alarms about is not an alarm.

So none of the sets below is written down twice. The variables the placer requires are
derived by RUNNING the placer's own entry points against a cleared environment and
collecting what they demand, because a list in a test is a list somebody has to remember
to extend. The manifests graded are FOUND by scanning `deploy/k8s/` for the factory that
reaches those entry points, so a second control-plane-shaped workload is graded the day
it is added rather than the day somebody remembers it. The verbs the Role grants are
compared against the adapter's own `*_namespaced_*` call sites rather than against a
list. Every scan asserts it found something: a scan over nothing passes.

**Granularity.** The RBAC cases below assert per *rule*, per `(apiGroup, resource)` pair
and per binding subject -- never per manifest and never per role. A structural check
whose granularity is coarser than the mechanism's lets the compliant instances vouch for
the non-compliant ones: this repository has already paid for that once, with a file-
granular tenancy check over a route-granular mechanism, and 605 of 605 passed with an
ungated route in the tree.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest
import yaml

from managed_agent.adapters.kubernetes.pod_runner import KubernetesPodRunner
from managed_agent.composition import build, pod_runner_from_environment
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import PodPhase

_ROOT = Path(__file__).resolve().parents[2]
_K8S = _ROOT / "deploy" / "k8s"
_BOOTSTRAP = _K8S / "cluster-bootstrap.yaml"
_CONTROL_PLANE = _K8S / "control-plane.yaml"
_SESSION_POD = _K8S / "session-pod.yaml"
_ADAPTER = _ROOT / "src" / "managed_agent" / "adapters" / "kubernetes" / "pod_runner.py"

_FACTORY: Final = "managed_agent.asgi:build_app"
"""The one process entry point that reaches `pod_runner_from_environment`.

`composition:tool_gateway_app` and `composition:model_gateway_app` do not, so a manifest
running either of those is not a placer and is not graded here. Named as the thing a
container actually runs rather than as a filename, so a second workload wiring the same
factory is graded without anyone noticing it should be.
"""

_NAMESPACE: Final = "map-dev"

_UNDIALLED: Final = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/unused"
"""A URL nothing connects to. `create_async_engine` resolves the driver and builds a
pool without dialling, so `build` runs to the end of its wiring offline."""

_ATTEMPTS: Final = 40
"""How many times the derivation below may ask a function to keep failing.

Bounded because a loop that drives a function by its exceptions is a hang the moment it
meets one it does not understand, and a hanging test reads exactly like a slow one.
"""

_STAND_INS: Final = (str(_SESSION_POD), "1", "x")
"""Values tried, in order, for a variable the derivation has just discovered.

Their only job is to get the call past the line that reads the variable; nothing here
asserts anything about a value. Three shapes because the placer's variables are not one
kind -- a path that must parse as a manifest, a number that must be positive, and a
string that only has to be non-empty -- and a single stand-in cannot be all three.
"""


# --------------------------------------------------------------------------------------
# Derivations. Each one is here instead of a literal, and each says what a literal
# would cost.
# --------------------------------------------------------------------------------------


def _variables_the_placer_demands(monkeypatch: pytest.MonkeyPatch) -> frozenset[str]:
    """Every variable the placer's own code refuses to run without.

    Produced by asking it, not by listing names. The environment is cleared, each call
    is repeated until it stops raising for a missing or unusable value, and every name
    it raised about is collected. A variable added to `composition.py` tomorrow lands in
    this set with no edit here, which is the whole point -- the alternative is a literal
    that silently stops being complete.

    `MAP_POD_MANIFEST` is in the set by a different argument and it is asserted rather
    than assumed: with it unset the process is not a placer at all, so no later variable
    is ever reached. That is the switch, and a deployment meant to place pods that omits
    it gets the silence this file is named after. `test_an_unset_manifest_is_the_switch`
    is where that claim is checked.
    """
    for name in list(os.environ):
        if name.startswith("MAP_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAP_POD_MANIFEST", str(_SESSION_POD))
    demanded = {"MAP_POD_MANIFEST"}
    runner: object = None
    for entry in _placer_entry_points_lazy():
        runner = _drive(entry, runner, demanded, monkeypatch)
    return frozenset(demanded)


def _placer_entry_points_lazy() -> Iterator[str]:
    """The two calls a served process makes on its way to being a placer, in order.

    Both, not one, and the order is a dependency rather than a preference.
    `pod_runner_from_environment` decides *whether* this process places pods and reads
    the cluster client's own inputs; `build` is only asked for the four values a
    Session's configuration is compiled from once it has been handed a runner. A
    derivation over the first alone would miss four variables and would look complete --
    which is the exact shape of the defect this file exists for.
    """
    yield "pod_runner_from_environment"
    yield "build"


def _drive(
    entry: str,
    runner: object,
    demanded: set[str],
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Call one entry point until it stops complaining, collecting what it complained
    about.

    `KeyError` names a variable that is absent; `ValueError` means the stand-in for the
    variable most recently supplied was the wrong shape, and the next stand-in is tried.
    Anything else propagates: a derivation that swallowed an unexpected failure would
    quietly return a short set, and a short set is what this whole file is against.
    """
    last: str | None = None
    tried: dict[str, int] = {}
    for _ in range(_ATTEMPTS):
        try:
            if entry == "pod_runner_from_environment":
                runner = pod_runner_from_environment()
            else:
                assert isinstance(runner, KubernetesPodRunner), runner
                _, engine = build(_UNDIALLED, pod_runner=runner)
                engine.sync_engine.dispose()
        except KeyError as missing:
            last = str(missing.args[0])
            demanded.add(last)
            tried[last] = 0
            monkeypatch.setenv(last, _STAND_INS[0])
        except ValueError:
            assert last is not None, "a value was refused before any was supplied"
            tried[last] += 1
            assert tried[last] < len(_STAND_INS), (
                f"no stand-in this derivation knows gets past {last}; add one to "
                "_STAND_INS rather than writing the variable into a literal set"
            )
            monkeypatch.setenv(last, _STAND_INS[tried[last]])
        else:
            return runner
    raise AssertionError(
        f"{entry} still refused an environment after {_ATTEMPTS} rounds; this "
        "derivation is not converging and the set it returns would be short"
    )


def _documents(path: Path) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def _containers(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every container in a document that carries a pod template, init containers too.

    Init containers are included because a `secretKeyRef` or an `args` list under one is
    as real as under any other, and reading only `containers` is how a walk falls behind
    a manifest -- `session-pod.yaml` already declares initContainers.
    """
    spec = document.get("spec", {})
    pod = spec.get("template", {}).get(
        "spec", spec if document["kind"] == "Pod" else {}
    )
    return list(pod.get("initContainers", [])) + list(pod.get("containers", []))


def _manifests_that_wire_the_placer() -> tuple[Path, ...]:
    """Every manifest under deploy/k8s/ whose container runs the placer's factory.

    Found by reading the containers' `args` for `managed_agent.asgi:build_app`, which is
    the one entry point that reaches `pod_runner_from_environment`. Named that way
    rather than `control-plane.yaml` so that a second workload wiring the same factory
    is graded without anyone noticing it should be.
    """
    found = []
    for path in sorted(_K8S.glob("*.yaml")):
        for document in _documents(path):
            if any(
                _FACTORY in container.get("args", [])
                for container in _containers(document)
            ):
                found.append(path)
                break
    return tuple(found)


def _environment_of(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in container.get("env", [])}


def _placer_containers() -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, container)
        for path in _manifests_that_wire_the_placer()
        for document in _documents(path)
        for container in _containers(document)
        if _FACTORY in container.get("args", [])
    ]


def _secret_refs_for(variable: str) -> dict[str, tuple[str, str]]:
    """Every manifest naming that variable, mapped to the (Secret, key) it names.

    Keyed by manifest so a failure says which files disagree rather than only that they
    do. A manifest naming the variable with a literal value is excluded and would fail
    the literal-value allowlist in `test_control_plane_manifest.py` instead.
    """
    named: dict[str, tuple[str, str]] = {}
    for path in sorted(_K8S.glob("*.yaml")):
        for document in _documents(path):
            for container in _containers(document):
                entry = _environment_of(container).get(variable)
                if entry is None:
                    continue
                ref = entry.get("valueFrom", {}).get("secretKeyRef")
                if ref is not None:
                    named[path.name] = (ref["name"], ref["key"])
    return named


# --------------------------------------------------------------------------------------
# The variables
# --------------------------------------------------------------------------------------


def test_the_derivation_finds_more_than_the_one_variable_it_seeds_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vacuity control for every case below that uses the derived set.

    A derivation that converged immediately would return `{"MAP_POD_MANIFEST"}` and
    every assertion built on it would pass over a set of one. Stated as its own case
    with a floor rather than as a line inside another test, because it is the assertion
    that says the others are about something.
    """
    demanded = _variables_the_placer_demands(monkeypatch)
    assert len(demanded) >= 5, (
        f"the placer demanded only {sorted(demanded)}; the derivation is not reaching "
        "the code that reads them and every case below is passing over a short set"
    )


def test_an_unset_manifest_is_the_switch_and_not_merely_one_more_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no manifest named, no later variable is ever reached.

    This is why `MAP_POD_MANIFEST` is seeded into the derivation rather than discovered
    by it: its absence does not raise, it makes the process quietly not a placer. The
    pair below is the whole claim -- unset and the answer is None, set and it is a
    runner -- and it is what the guard in `pod_runner_from_environment` fires on.
    """
    for name in list(os.environ):
        if name.startswith("MAP_"):
            monkeypatch.delenv(name, raising=False)
    assert pod_runner_from_environment() is None

    monkeypatch.setenv("MAP_POD_MANIFEST", str(_SESSION_POD))
    monkeypatch.setenv("MAP_NAMESPACE", "map-test")
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a key")
    assert pod_runner_from_environment() is not None


def test_naming_a_namespace_with_no_manifest_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The state a deployed control plane was actually in, silently, for a day.

    Both of the placer's variables were absent, and the manifest's comment said the
    first one's absence would CrashLoop this process. It did not, because the second
    absence meant the first was never read. This is the case that fires on what is
    PRESENT: a process naming a namespace to place into with no manifest to place from
    has declared an intention it cannot act on.

    The message must name both variables, because an operator reading it has to know
    which of the two answers is the one they meant.
    """
    for name in list(os.environ):
        if name.startswith("MAP_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAP_NAMESPACE", _NAMESPACE)

    with pytest.raises(RuntimeError) as refused:
        pod_runner_from_environment()

    assert "MAP_NAMESPACE" in str(refused.value)
    assert "MAP_POD_MANIFEST" in str(refused.value)


def test_a_process_with_neither_variable_is_a_legitimate_non_placer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the guard, and the one that keeps it from being "refuse
    everything".

    A Tool Gateway, a Model Gateway, a local run and every test in this suite deploy
    with neither variable, and each is a process that legitimately places no pod. If
    this case ever fails, the guard above has stopped distinguishing a misconfigured
    placer from something that was never meant to be one, and it will have taken every
    other workload down with it.
    """
    for name in list(os.environ):
        if name.startswith("MAP_"):
            monkeypatch.delenv(name, raising=False)
    assert pod_runner_from_environment() is None


def test_a_zero_or_negative_token_lifetime_stops_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No floor over a compiled configuration reads the expiry, so this is where it is
    read.

    `check_floors` grades the token's shape and the two identifiers inside it, and every
    one of those holds for an expiry of zero. What that produces is a pod whose every
    tool call is answered with the Tool Gateway's fixed 401 -- the same answer an absent
    token gets -- for the pod's whole life, with nothing downstream able to name the
    cause. Refused at the one point where the variable can still be named.
    """
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a key")
    monkeypatch.setenv("MAP_SESSION_TOKEN_KEY", "another key")
    monkeypatch.setenv("MAP_TOOL_GATEWAY_URL", "http://tool-gateway/mcp")
    monkeypatch.setenv("MAP_MODEL_GATEWAY_URL", "http://model-gateway/v1")
    for bad in ("0", "-1"):
        monkeypatch.setenv("MAP_SESSION_TOKEN_LIFETIME_S", bad)
        with pytest.raises(ValueError, match="MAP_SESSION_TOKEN_LIFETIME_S"):
            build(_UNDIALLED, pod_runner=_AbsentPod())


class _AbsentPod:
    """A cluster that finds no pod and starts none. Enough to make `build` take the
    placer's branch, which is the only thing these cases need of it."""

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        raise AssertionError("a test in this file started a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        raise AssertionError("a test in this file located a pod")

    async def remove(self, pod_name: str) -> None:
        """A no-op, where this used to refuse.

        Under ADR-041 a pod is leased for one Turn, so every dispatch releases one and
        the refusal written here asserted the opposite of the contract. Nothing is
        recorded because no case in this file grades which pod went -- the lease itself
        is graded in `tests/control/test_a_pod_is_leased_for_one_turn.py`, against a
        cluster whose phase actually reflects the removal.
        """


def test_the_scan_finds_at_least_one_manifest_wiring_the_placer() -> None:
    """The other vacuity control, and it is not decoration.

    Every manifest case below iterates this scan. If the factory name in
    `control-plane.yaml`'s `args` were renamed, the scan would find nothing, every one
    of those cases would pass over an empty list, and the suite would go green having
    stopped checking the deployment entirely.
    """
    found = _manifests_that_wire_the_placer()
    assert found, (
        f"no manifest under {_K8S} runs {_FACTORY}, so every case reading this scan is "
        "asserting nothing"
    )


def test_every_manifest_that_wires_the_placer_declares_every_variable_it_demands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The committed manifest and the code that reads it, compared.

    Neither side is a list here: the variables come from running the placer, the
    manifests come from scanning for the factory. A fifth variable added to
    `composition.py` fails this with no edit to this file, and that is the property that
    makes it a guard rather than a second copy.
    """
    demanded = _variables_the_placer_demands(monkeypatch)
    containers = _placer_containers()
    assert containers, "the scan found no placer container"
    for path, container in containers:
        missing = sorted(demanded - set(_environment_of(container)))
        assert not missing, (
            f"{path.name} runs {_FACTORY} and declares none of {missing}. A process "
            "wired this way and missing one of these either refuses to start or "
            "accepts Sessions it can never place."
        )


def test_every_manifest_naming_the_session_token_key_names_one_secret_and_one_key() -> (
    None
):
    """The signing side and the verifying side read one value.

    The control plane signs a Session's compiled configuration with this key and the
    Tool Gateway verifies the token it produced. Two Secrets holding "the same" key are
    two things free to diverge, and divergence shows up as every Session's tool calls
    answering 401 with nothing naming a cause.

    The count assertion is not decoration. With one occurrence this passes trivially,
    and one occurrence is exactly the state before this slice -- the Tool Gateway's
    alone, with the signing side reading nothing at all.

    What this cannot say is that the bytes agree; it compares two *references*. Whether
    the values behind them are one value is settled where a token the control plane
    signed is put in front of the Gateway that verifies it.
    """
    named = _secret_refs_for("MAP_SESSION_TOKEN_KEY")
    assert len(named) >= 2, f"only {len(named)} manifest names it: {sorted(named)}"
    assert len(set(named.values())) == 1, named


def test_the_two_signing_keys_are_two_and_never_collapse_into_one() -> None:
    """One key for two hops in opposite directions makes either side's compromise the
    other's.

    The shim bearer authenticates the control plane *to* a pod; the Session token
    authenticates a pod *to* the Tool Gateway. A deployment that pointed both at one
    `(Secret, key)` would work, pass every other case in this file, and hand a pod the
    ability to mint the credential that commands it.
    """
    shim = set(_secret_refs_for("MAP_SHIM_TOKEN_KEY").values())
    session = set(_secret_refs_for("MAP_SESSION_TOKEN_KEY").values())
    assert shim, "no manifest names MAP_SHIM_TOKEN_KEY through a secretKeyRef"
    assert session, "no manifest names MAP_SESSION_TOKEN_KEY through a secretKeyRef"
    assert not (shim & session), (
        f"the two signing keys resolve to the same (Secret, key): {shim & session}"
    )


# --------------------------------------------------------------------------------------
# The pod manifest: one path, spelled in four places, compared rather than trusted
# --------------------------------------------------------------------------------------


def _platform() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "map_platform_for_placer_config", _ROOT / "deploy" / "platform.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_mounted_pod_manifest_path_is_the_one_the_deployment_actually_mounts() -> (
    None
):
    """One path, derived in four places, compared here rather than trusted anywhere.

    `MAP_POD_MANIFEST` names a file; a volumeMount names the directory it is under; the
    ConfigMap is generated with `--from-file`, which keys the entry by the source file's
    basename; and the volume names the ConfigMap. Four derivations of one contract, and
    nothing would fail if they disagreed -- the pod would start, the process would read
    a path that is not there, and the failure would arrive as an unreadable manifest
    with nothing naming a mount.
    """
    module = _platform()
    control_plane = next(w for w in module.WORKLOADS if w.component == "control-plane")
    generated = dict(control_plane.generated_config_maps)
    assert generated, "the control plane generates no ConfigMap, so nothing is mounted"

    documents = _documents(_CONTROL_PLANE)
    pod = documents[0]["spec"]["template"]["spec"]
    container = _containers(documents[0])[0]
    named = _environment_of(container)["MAP_POD_MANIFEST"]["value"]

    volumes = {v["name"]: v for v in pod["volumes"]}
    mounts = {m["mountPath"]: m for m in container["volumeMounts"]}
    holding = [mount for path, mount in mounts.items() if named.startswith(f"{path}/")]
    assert len(holding) == 1, (
        f"MAP_POD_MANIFEST is {named} and the container's mountPaths are "
        f"{sorted(mounts)}; exactly one must contain it"
    )
    volume = volumes[holding[0]["name"]]
    config_map = volume["configMap"]["name"]
    assert config_map in generated, (
        f"the volume mounts ConfigMap {config_map}, which deploy/platform.py does not "
        f"generate; it generates {sorted(generated)}"
    )
    source = Path(generated[config_map])
    assert Path(named).name == source.name, (
        f"MAP_POD_MANIFEST names {Path(named).name} and the ConfigMap is generated "
        f"from {source.name}; `--from-file` keys the entry by the basename, so the "
        "mounted file would be at a different path than the one the process reads"
    )
    assert holding[0].get("readOnly") is True, (
        "the pod-manifest mount is not readOnly. The process only reads it, and a "
        "volume mount root is created by kubelet as uid 0 -- fsGroup sets the group "
        "and never the owner -- so anything here that needed ownership would fail "
        "with an error naming nothing"
    )


def test_the_generated_config_map_is_not_image_substituted() -> None:
    """The Session-pod manifest reaches the cluster as its own bytes.

    `substituted()` rewrites the digest placeholder to the *platform* image for this
    commit. `session-pod.yaml` carries the same placeholder on all three of its
    containers, and `pod_runner._pod_for` rewrites it per Session to the registered
    Environment's digest. Passing it through `substituted()` on the way into the
    ConfigMap would bake the control plane's own image into every Session pod, and those
    pods would *start* -- running uvicorn against an image with no `codex` in it -- so
    the failure would arrive as a readiness timeout with nothing in it about an image.

    Two assertions, and the first alone would be a declaration compared to nothing. The
    placeholder must still be in the file, or this case is about nothing; and the loop
    that generates the ConfigMap must reach that file by PATH, through `--from-file`,
    with no call to `substituted` anywhere inside it. Parsed rather than grepped so a
    call in a neighbouring branch of `main` cannot fail this, and a call added inside
    the loop cannot escape it.
    """
    module = _platform()
    assert module.DIGEST_PLACEHOLDER in _SESSION_POD.read_text(), (
        "session-pod.yaml no longer carries the placeholder, so the claim this case "
        "makes about what must not be substituted is about nothing"
    )
    loop = _the_generated_config_map_loop()
    called = {
        node.func.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "substituted" not in called, (
        "deploy/platform.py calls substituted() inside the loop that generates the "
        "Session-pod ConfigMap, so every Session pod would be started on the control "
        "plane's own image -- which holds no codex -- and the failure would arrive as "
        "a readiness timeout naming nothing"
    )
    assert any(
        "--from-file=" in piece.value
        for node in ast.walk(loop)
        if isinstance(node, ast.JoinedStr)
        for piece in node.values
        if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
    ), (
        "the generated-ConfigMap loop no longer passes --from-file, so the bytes it "
        "sends are built in this process rather than read off disk and the assertion "
        "above no longer covers where they came from"
    )


def _the_generated_config_map_loop() -> ast.For:
    """The `for name, source in workload.generated_config_maps:` statement in `main`.

    Found by the thing it iterates rather than by position, so inserting a step above or
    below it in `main` moves nothing here. A missing loop fails rather than yielding an
    empty walk that every assertion over it would pass.
    """
    tree = ast.parse((_ROOT / "deploy" / "platform.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Attribute):
            continue
        if node.iter.attr == "generated_config_maps":
            return node
    raise AssertionError(
        "deploy/platform.py has no loop over generated_config_maps, so nothing "
        "generates the Session-pod ConfigMap and the control plane mounts an object "
        "this repository never creates"
    )


def test_the_control_socket_sits_below_every_volume_mount_root() -> None:
    """A socket in a mount root cannot be made private, and the error names nothing.

    The Agent Runtime calls `prepare_private_socket_directory()` on the parent of its
    `--listen` path: `mkdir(0700)`, and on EEXIST a `chmod(0700)` if the mode differs.
    Kubelet creates a volume mount root as **uid 0**; `fsGroup` sets the group and the
    setgid bit and never the owner, and there is no `fsUser`. So a `chmod` issued there
    by the pod's uid, with every capability dropped, is EPERM -- measured on a real
    node, under every seccomp profile including Unconfined. Writability was never the
    question; ownability was.

    One directory deeper the runtime creates the parent itself and owns it, so the chmod
    branch never runs. This asserts that depth: the socket's parent must not BE a mount
    root. Named as a guard rather than left to the comment in `session-pod.yaml`,
    because a comment that says a thing is true is not a mechanism.
    """
    from managed_agent.control.pod_config.compiler import CONTROL_SOCKET

    parent = str(Path(CONTROL_SOCKET).parent)
    roots = {
        mount["mountPath"].rstrip("/")
        for document in _documents(_SESSION_POD)
        for container in _containers(document)
        for mount in container.get("volumeMounts", [])
    }
    assert roots, "session-pod.yaml declares no volumeMounts, so this proves nothing"
    assert parent not in roots, (
        f"the control socket's parent {parent} IS a volume mount root. The runtime "
        "chmods that directory before binding and cannot own it, so the process dies "
        "with 'Operation not permitted' before it listens, under every profile."
    )


# --------------------------------------------------------------------------------------
# RBAC. Found by scanning, graded per rule.
# --------------------------------------------------------------------------------------


_PERMITTED_PAIRS: Final[dict[str, frozenset[tuple[str, str]]]] = {
    "control-plane": frozenset(
        {
            ("", "pods"),
            ("", "secrets"),
            # `get` on one Deployment by name: the autoscaler's, whose
            # `--max-nodes-total` is the node ceiling `GET /v1/capacity` reports.
            # Read rather than configured, because a number copied into this
            # platform's own settings is free to disagree with the flag that
            # actually bounds the cluster -- and the disagreement would show up as
            # an operator trusting a ceiling that was never enforced.
            ("apps", "deployments"),
        }
    ),
    "map-cluster-autoscaler": frozenset(
        {
            ("", "events"),
            ("", "endpoints"),
            ("", "pods/eviction"),
            ("", "pods/status"),
            ("", "nodes"),
            ("", "namespaces"),
            ("", "pods"),
            ("", "services"),
            ("", "replicationcontrollers"),
            ("", "persistentvolumeclaims"),
            ("", "persistentvolumes"),
            ("", "configmaps"),
            ("extensions", "replicasets"),
            ("extensions", "daemonsets"),
            ("extensions", "jobs"),
            ("apps", "statefulsets"),
            ("apps", "replicasets"),
            ("apps", "daemonsets"),
            ("batch", "jobs"),
            ("policy", "poddisruptionbudgets"),
            ("storage.k8s.io", "storageclasses"),
            ("storage.k8s.io", "csinodes"),
            ("storage.k8s.io", "csidrivers"),
            ("storage.k8s.io", "csistoragecapacities"),
            ("coordination.k8s.io", "leases"),
        }
    ),
    "cluster-autoscaler": frozenset(
        {("", "configmaps"), ("coordination.k8s.io", "leases")}
    ),
    # `list` on nodes, cluster-scoped, which is why it is a ClusterRole and not a
    # rule on the Role above: nodes are not namespaced, so a namespaced grant for
    # them cannot exist. It is the other half of the capacity surface's node
    # answer -- how many nodes are schedulable now, against the ceiling read from
    # the autoscaler's flag.
    "control-plane-counts-nodes": frozenset({("", "nodes")}),
}
"""Which (apiGroup, resource) pairs each declared role may touch.

A literal on purpose, and the scan below is what keeps it honest: every Role and
ClusterRole found anywhere under `deploy/k8s/` must be a key here, so a third identity
fails by name and by file instead of being graded by nothing. The scan discovers, the
literal is what somebody must extend, and the failure names the fix.

`tests/deploy/test_cluster_autoscaler.py` grades the autoscaler's own rules against its
own allowlist and is not edited. Two graders overlapping there is belt and braces rather
than a second source: that one answers "may this identity touch this resource" and this
one answers "is every identity graded at all".
"""


def _rbac_objects(kinds: frozenset[str]) -> list[tuple[Path, dict[str, Any]]]:
    """Every RBAC document of those kinds anywhere under deploy/k8s/, with its file.

    Found by `apiVersion`, so a new file carrying RBAC is covered the day it is
    committed. The file travels with the document because a failure has to say where to
    go, and a role name alone does not.
    """
    return [
        (path, document)
        for path in sorted(_K8S.glob("*.yaml"))
        for document in _documents(path)
        if str(document.get("apiVersion", "")).startswith("rbac.authorization.k8s.io/")
        and document["kind"] in kinds
    ]


def _roles() -> list[tuple[Path, dict[str, Any]]]:
    return _rbac_objects(frozenset({"Role", "ClusterRole"}))


def _bindings() -> list[tuple[Path, dict[str, Any]]]:
    return _rbac_objects(frozenset({"RoleBinding", "ClusterRoleBinding"}))


def _rules() -> list[tuple[Path, str, int, dict[str, Any]]]:
    """Every rule of every role, one entry each, with enough to name it in a failure.

    Flattened to the RULE rather than the role, because that is the granularity the
    mechanism has: a role's permissions are the union of its rules, and a check that
    looked at a role as a whole would let its compliant rules vouch for its
    non-compliant ones. This repository has already paid for a granularity mismatch of
    exactly that shape.
    """
    return [
        (path, role["metadata"]["name"], index, rule)
        for path, role in _roles()
        for index, rule in enumerate(role.get("rules", []))
    ]


def test_the_rbac_scan_finds_the_roles_that_are_there() -> None:
    """The vacuity control for every RBAC case below.

    A scan keyed on `apiVersion` returns nothing if the key is misspelled, and every
    case below would then pass over an empty list. Two roles is what this tree holds:
    the autoscaler's and the control plane's.
    """
    assert len(_roles()) >= 2, f"the scan found {len(_roles())} roles under {_K8S}"
    assert _rules(), "the scan found roles with no rules at all"


def test_every_role_declared_anywhere_under_deploy_is_graded_by_a_named_allowlist() -> (
    None
):
    """The coverage assertion, and the reason this is not a second copy of the
    autoscaler's grader.

    Nothing in this repository graded the autoscaler's RBAC at all until it was found by
    hand, and the fix went into the autoscaler's own file, reading that one manifest.
    This slice adds the second RBAC subject in the tree. A grader that reads one file is
    a grader against one file, which is the same sentence one column over -- so this one
    finds its subjects instead of being told them, and a third identity fails here by
    name and by file rather than being graded by nothing.
    """
    ungraded = sorted(
        f"{path.name}:{role['metadata']['name']}"
        for path, role in _roles()
        if role["metadata"]["name"] not in _PERMITTED_PAIRS
    )
    assert not ungraded, (
        f"{ungraded} declare RBAC that nothing here grades. Add each to "
        "_PERMITTED_PAIRS with the (apiGroup, resource) pairs it needs and a reason."
    )


def test_no_rule_grants_a_resource_outside_its_roles_allowlist() -> None:
    """Per rule and per pair, because that is the granularity a grant has.

    An allowlist rather than a ban list, for the reason the autoscaler's own grader
    gives: a ban list cannot cover what nobody thought of. `create` on `roles` or
    `escalate` on anything would be worse than reading Secrets, and no ban drafted today
    would name them.
    """
    for path, name, index, rule in _rules():
        permitted = _PERMITTED_PAIRS[name]
        granted = {
            (group, resource)
            for group in rule["apiGroups"]
            for resource in rule["resources"]
        }
        unexpected = sorted(granted - permitted)
        assert not unexpected, (
            f"{path.name}:{name} rule {index} grants {unexpected}, which nothing here "
            "permits"
        )


def test_no_rule_anywhere_uses_a_wildcard() -> None:
    """A `*` grants what an allowlist cannot see.

    Checked separately because a wildcard passes a pair allowlist trivially: a
    `("*", "*")` is one pair, and one pair is easy to add to a list of twenty-five
    without anyone noticing that it subsumes all of them.
    """
    for path, name, index, rule in _rules():
        for field in ("apiGroups", "resources", "verbs"):
            assert "*" not in rule.get(field, []), (
                f"{path.name}:{name} rule {index} uses a wildcard in {field}: {rule}"
            )


def test_no_rule_grants_get_on_secrets() -> None:
    """`("", "secrets")` is permitted for the control plane and the verb is the
    difference.

    Writing a Session's own material and reading this namespace's database credential
    are the same (apiGroup, resource) pair, so the pair allowlist cannot see between
    them. That is why this is its own case rather than a line inside it.

    It buys less than it reads like, and saying so here is the point: a PATCH response
    carries the modified object, so `patch` on secrets is already a read of every Secret
    in the namespace. Withholding `get` is worth doing and it is not a boundary. Nothing
    in this file may claim otherwise.
    """
    for path, name, index, rule in _rules():
        if "secrets" not in rule["resources"]:
            continue
        assert "get" not in rule["verbs"], (
            f"{path.name}:{name} rule {index} grants get on secrets"
        )
        assert "list" not in rule["verbs"] and "watch" not in rule["verbs"], (
            f"{path.name}:{name} rule {index} can enumerate this namespace's Secrets"
        )


def test_no_role_grants_services_because_nothing_calls_the_services_api() -> None:
    """The one grant it would have been reasonable to add and wrong to.

    A Session's shim is reached at a DNS name `shim_url_for` computes; the headless
    Service that makes that name resolve is applied by `deploy/bootstrap.py`, not by the
    control plane. Measured: no `*_namespaced_service` call exists anywhere in `src/`.
    Written as its own case because "the adapter does not call it" is a fact that can
    stop being true quietly, and then this fails and says which call made it false.

    Scoped to the roles this slice owns rather than to every role in the tree: the
    autoscaler legitimately reads Services, which is why it is a key in the allowlist
    above with that pair in it.
    """
    assert not [name for name in _verbs_the_adapter_calls() if name[0] == "services"], (
        "the placer now calls the services API; this Role must be widened deliberately"
    )
    for path, name, index, rule in _rules():
        if name != "control-plane":
            continue
        assert "services" not in rule["resources"], (
            f"{path.name}:{name} rule {index} grants services, which nothing calls"
        )


def test_every_binding_names_one_service_account_that_bootstrap_creates() -> None:
    """One subject per binding, and it is an identity this repository actually creates.

    Per SUBJECT, not per binding: a binding's subjects are a list, and a check that
    looked at the list as a whole would let the first subject vouch for a second one
    added beside it. A second subject hands the whole Role to whoever it names, which is
    the Kubernetes analogue of a trust policy with two principals.
    """
    created = {
        document["metadata"]["name"]
        for document in _documents(_BOOTSTRAP)
        if document["kind"] == "ServiceAccount"
    }
    assert created, f"{_BOOTSTRAP.name} creates no ServiceAccounts"
    assert _bindings(), "the scan found no RoleBindings at all"
    for path, binding in _bindings():
        subjects = binding["subjects"]
        assert len(subjects) == 1, (
            f"{path.name}:{binding['metadata']['name']} names {len(subjects)} "
            "subjects; each one holds the whole Role"
        )
        subject = subjects[0]
        assert subject["kind"] == "ServiceAccount", subject
        assert subject["namespace"] == _NAMESPACE, subject
        assert subject["name"] in created, (
            f"{path.name}:{binding['metadata']['name']} binds to ServiceAccount "
            f"{subject['name']}, which {_BOOTSTRAP.name} does not create"
        )


def test_every_binding_names_a_role_declared_in_the_same_file() -> None:
    """A `roleRef` to a role nothing declares binds to nothing and says nothing.

    Kubernetes accepts a RoleBinding whose roleRef names an absent Role: the binding is
    created and grants nothing, so the failure is a 403 from a workload rather than an
    error at apply time.
    """
    for path, binding in _bindings():
        declared = {
            document["metadata"]["name"]
            for document in _documents(path)
            if document["kind"] in ("Role", "ClusterRole")
        }
        ref = binding["roleRef"]["name"]
        assert ref in declared, (
            f"{path.name}:{binding['metadata']['name']} refers to {ref}, which "
            f"{path.name} does not declare (it declares {sorted(declared)})"
        )


# --------------------------------------------------------------------------------------
# The Role and the adapter, compared in both directions
# --------------------------------------------------------------------------------------


_CLIENT_VERB_FOR: Final[dict[str, str]] = {"read": "get"}
"""Where the generated Kubernetes client's method name differs from the API's verb.

`read_namespaced_pod` is a `get`. The one thing in the derivation below that somebody
had to know rather than read off the source, so it is written down instead of inferred.
"""


def _verbs_the_adapter_calls() -> frozenset[tuple[str, str]]:
    """Every (resource, verb) pair `pod_runner.py` actually exercises.

    Read off the adapter's own call sites -- every `<verb>_namespaced_<resource>`
    attribute named in the module -- and not typed out here, because a verb list in a
    test is a list that keeps agreeing with itself while the adapter grows a seventh
    call.

    Parsed with `ast` rather than matched with a regex, so a name inside a docstring or
    a comment is not a call. That distinction matters in this module: its docstrings
    name these methods repeatedly.
    """
    tree = ast.parse(_ADAPTER.read_text())
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parts = node.attr.split("_namespaced_")
        if len(parts) != 2 or not parts[1]:
            continue
        verb, resource = parts
        found.add((f"{resource}s", _CLIENT_VERB_FOR.get(verb, verb)))
    return frozenset(found)


_API_RECEIVERS: Final[frozenset[str]] = frozenset({"core", "apps"})
"""The local names `pod_runner.py` binds its two Kubernetes clients to.

Written down because it is the discriminator that makes the cluster-scoped scan below
derivable: `verb_resource` is a shape any attribute access could have, and only an
attribute of one of these is an API call. A rename in the adapter turns the scan up
empty, which its own vacuity control catches.
"""


def _cluster_verbs_the_adapter_calls() -> frozenset[tuple[str, str]]:
    """Every (resource, verb) pair the adapter calls on a CLUSTER-SCOPED API.

    The namespaced scan above is structurally blind to these, and the blindness had
    consequences: `core.list_node()` carries no `_namespaced_` because nodes are not
    namespaced, so the grant for it sat in a ClusterRole that nothing compared against
    any caller.

    Found by looking at attributes of the two client objects the adapter binds -- `core`
    and `apps` -- rather than by listing the cluster-scoped method names here. A list in
    a test is a list that keeps agreeing with itself while the adapter grows a call. The
    receiver name is the discriminator that makes this derivable at all: any attribute
    access in the module could look like `verb_resource`, and only these two are the
    API.
    """
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(_ADAPTER.read_text())):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in _API_RECEIVERS or "_namespaced_" in node.attr:
            continue
        verb, _, resource = node.attr.partition("_")
        if not resource:
            continue
        found.add((f"{resource}s", _CLIENT_VERB_FOR.get(verb, verb)))
    return frozenset(found)


def test_the_adapter_scan_finds_the_calls_that_are_there() -> None:
    """The vacuity control for both equalities below.

    A parse that found nothing would make an equality assert `frozenset() == <the
    Role>`, which fails -- but for the wrong reason, and with a message that sends
    somebody to the manifest. Asserted as the exact resource sets rather than as a
    count, because a count keeps passing when one resource is dropped and another added.
    """
    called = _verbs_the_adapter_calls()
    assert len(called) >= 6, f"the adapter scan found only {sorted(called)}"
    assert {resource for resource, _ in called} == {
        "pods",
        "secrets",
        "deployments",
    }, sorted(called)

    cluster = _cluster_verbs_the_adapter_calls()
    assert cluster == {("nodes", "list")}, sorted(cluster)


def test_the_control_plane_role_grants_exactly_what_the_adapter_calls() -> None:
    """Equality, not containment, and the assertion names both differences.

    A containment check in one direction alone is how a Role grows: each widening looks
    locally fine and nothing ever says the grant stopped matching the caller. A verb the
    Role grants and the adapter never calls is a grant nobody can justify; a call the
    Role does not cover is a 403 whose symptom is a placement that fails looking like a
    new bug. The failure this repository actually had was the second, in its most
    complete form: no Role at all.
    """
    granted = {
        (resource, verb)
        for _, name, _, rule in _rules()
        if name == "control-plane"
        for resource in rule["resources"]
        for verb in rule["verbs"]
    }
    called = _verbs_the_adapter_calls()
    assert granted == called, (
        f"granted but never called: {sorted(granted - called)}; "
        f"called but not granted: {sorted(called - granted)}"
    )


def test_the_cluster_role_grants_exactly_the_cluster_scoped_calls() -> None:
    """The cluster-scoped grant matches the cluster-scoped calls, both directions.

    A separate assertion from the Role's rather than a wider one, because the two grants
    are different objects with different blast radii. A namespaced Role reaches one
    namespace; a ClusterRole reaches the cluster, so a resource that drifts into this
    one is the more expensive of the two mistakes and deserves its own failure message.

    This is the assertion that did not exist when the node grant was written. Nothing
    compared the ClusterRole to any caller, so `list` on nodes was a grant whose only
    justification was the commit message that added it.
    """
    granted = {
        (resource, verb)
        for _, name, _, rule in _rules()
        if name == "control-plane-counts-nodes"
        for resource in rule["resources"]
        for verb in rule["verbs"]
    }
    assert granted, "no rule was found for control-plane-counts-nodes"
    called = _cluster_verbs_the_adapter_calls()
    assert granted == called, (
        f"granted but never called: {sorted(granted - called)}; "
        f"called but not granted: {sorted(called - granted)}"
    )


# --------------------------------------------------------------------------------------
# Live. A committed manifest and a running object are different facts.
# --------------------------------------------------------------------------------------

_UNREACHABLE: Final = (
    "connection refused",
    "was refused",
    "no configuration has been provided",
    "unable to connect",
)


def _kubectl(*args: str) -> dict[str, Any]:
    """Run kubectl and parse its JSON, skipping when there is no cluster.

    A missing cluster and a missing object are different outcomes and only the second is
    a failure. `kubectl` exits non-zero for both, so the message on stderr is the only
    thing that separates them.
    """
    done = subprocess.run(
        ["kubectl", *args, "-o", "json"], capture_output=True, text=True, timeout=60
    )
    if done.returncode != 0:
        lowered = done.stderr.lower()
        if any(marker in lowered for marker in _UNREACHABLE):
            pytest.skip(f"no reachable cluster: {done.stderr.strip()[:200]}")
        pytest.fail(f"kubectl {' '.join(args)} failed:\n{done.stderr.strip()}")
    parsed: dict[str, Any] = json.loads(done.stdout)
    return parsed


def _live_environment(component: str) -> dict[str, dict[str, Any]]:
    """The env entries of a running Deployment's first container, by name.

    Names and references only. No value behind a `secretKeyRef` is read: an entry
    contributes its name and the `(Secret, key)` it points at, and nothing else.
    """
    deployment = _kubectl("get", "deploy", component, "-n", _NAMESPACE)
    containers = deployment["spec"]["template"]["spec"]["containers"]
    return {entry["name"]: entry for entry in containers[0].get("env", [])}


@pytest.mark.network
def test_the_running_control_plane_declares_every_variable_the_placer_demands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read off the cluster, not off the file.

    A committed manifest and a running object are different facts, and this suite has
    been caught by that three times. A Deployment applied by hand, or applied from an
    older commit, satisfies every check above and places nothing.
    """
    demanded = _variables_the_placer_demands(monkeypatch)
    running = set(_live_environment("control-plane"))
    missing = sorted(demanded - running)
    assert not missing, (
        f"the control-plane Deployment running in {_NAMESPACE} declares none of "
        f"{missing}. It is accepting Sessions it cannot place."
    )


@pytest.mark.network
def test_the_running_control_plane_and_tool_gateway_resolve_one_signing_key() -> None:
    """Both Deployments' secretKeyRefs for MAP_SESSION_TOKEN_KEY name one pair.

    Reads the two references and never the value behind them. Whether the bytes agree is
    settled by the live checkpoint in `tests/pod/`, which puts a token the control plane
    signed in front of the Gateway that verifies it -- a reference and a value are two
    different claims, and only the second one matters to a Session.
    """
    refs = set()
    for component in ("control-plane", "tool-gateway"):
        entry = _live_environment(component).get("MAP_SESSION_TOKEN_KEY")
        assert entry is not None, (
            f"the running {component} declares no MAP_SESSION_TOKEN_KEY"
        )
        ref = entry["valueFrom"]["secretKeyRef"]
        refs.add((ref["name"], ref["key"]))
    assert len(refs) == 1, f"the two Deployments name {sorted(refs)}"


@pytest.mark.network
def test_the_deployed_environment_actually_produces_a_pod_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feed the running Deployment's own environment to the code that reads it.

    Every other case here compares names: the manifest declares X, the Deployment
    declares X. None of them says the environment ADDS UP to a placer, and that is the
    claim `control-plane.yaml` has been making since MAP-55 landed -- in a comment, with
    nothing able to fail.

    This takes the variable NAMES off the live Deployment, gives each one a value good
    enough to get past the line that reads it (the repository's own `session-pod.yaml`
    for the manifest path, a non-empty throwaway for each secretKeyRef, since the point
    is the SHAPE of the environment and not its contents) and asserts
    `pod_runner_from_environment()` returns a runner rather than None.

    Run against the environment as it stood before this slice this FAILS, returning
    None, which is the whole reason it is here. No value of any Secret is read: a
    secretKeyRef entry contributes only its name.
    """
    running = _live_environment("control-plane")
    for name in list(os.environ):
        if name.startswith("MAP_"):
            monkeypatch.delenv(name, raising=False)
    for name, entry in running.items():
        if not name.startswith("MAP_"):
            continue
        literal = entry.get("value")
        if name == "MAP_POD_MANIFEST":
            # A path inside the pod, which does not exist here. Substituted with the
            # repository's own copy of the file that ConfigMap is generated from, so
            # what is graded is that the variable is declared and parses -- not that
            # this machine holds the pod's filesystem.
            monkeypatch.setenv(name, str(_SESSION_POD))
        elif literal is not None:
            monkeypatch.setenv(name, literal)
        else:
            monkeypatch.setenv(name, "a throwaway standing in for a secret's value")

    assert pod_runner_from_environment() is not None, (
        "the control plane's deployed environment does not add up to a placer: "
        f"pod_runner_from_environment() returned None over {sorted(running)}"
    )
