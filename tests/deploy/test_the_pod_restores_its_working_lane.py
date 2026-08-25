"""The init container that puts a Session's `working` lane back, as the pod declares it.

Tier 1 (local, no infrastructure). Everything here reads `deploy/k8s/session-pod.yaml`
and the artifacts it has to agree with -- the image's venv path, the module the command
names, the readiness bound the adapter waits under. What it cannot see is a running pod;
what it can see is every way the manifest and the code could describe different things.

The manifest travels as a ConfigMap and the module travels in the image, so those two
land by different routes and can land in either order. The import check below is the one
that matters for that: a manifest naming a module the image does not carry is a pod
whose init container exits `No module named ...` for every tenant, and nothing else in
this suite would notice.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Final

import yaml

from managed_agent.adapters.kubernetes.pod_runner import _READY_TIMEOUT_SECONDS
from managed_agent.control.files.workspace_sync import WORKING_COUNT_LIMIT
from managed_agent.control.pod_config.compiler import WORKSPACE_ROOT
from managed_agent.session_shim.restore_working_lane import _CONCURRENT_FETCHES

_DEPLOY: Final[Path] = Path(__file__).resolve().parents[2] / "deploy"
_MANIFEST: Final[Path] = _DEPLOY / "k8s" / "session-pod.yaml"
_DOCKERFILE: Final[Path] = _DEPLOY / "docker" / "session.Dockerfile"
_POD: Final[dict[str, Any]] = yaml.safe_load(_MANIFEST.read_text())

RESTORE: Final = "restore-working-lane"
SEED: Final = "seed-runtime-home"

IMAGE_PULL_ALLOWANCE_SECONDS: Final = 60.0
"""What the readiness bound already spent on pulling the image before this slice.

Not measured here and not measurable here -- it is the figure
`pod_runner._READY_TIMEOUT_SECONDS` was built on when it was 180 s, restated so the
arithmetic below is about what the restore added rather than about the whole bound.
"""

SERIAL_ROUND_TRIP_SECONDS: Final = 0.02
"""ADR-030's stated cost of one object GET, used to price the restore's worst case.

The ADR's number rather than a measurement of this cluster, and the worst case rather
than the expected one: the fetch is concurrent, so this prices the restore as if that
concurrency were entirely lost. A bound that survives that survives losing either half
of the fix, which is the property the ADR asks for in as many words.
"""


def _init_containers() -> list[dict[str, Any]]:
    inits: list[dict[str, Any]] = list(_POD["spec"]["initContainers"])
    return inits


def _restore_container() -> dict[str, Any]:
    named = [c for c in _init_containers() if c["name"] == RESTORE]
    assert named, f"the pod manifest declares no {RESTORE!r} init container"
    return named[0]


def test_the_restore_runs_after_the_container_that_makes_the_sandbox_targets() -> None:
    """The ordering ADR-030 calls load-bearing, read off the mechanism rather than off a
    list of names.

    bwrap refuses to build a sandbox for this Session unless `<workspace>/.codex` and
    `<workspace>/.agents` exist as directories, and the seeding container's script is
    what makes them. So the restore cannot be folded into that script -- its last act is
    creating the paths the restore would write into -- and it cannot run before it.
    """
    made_here = [
        container["name"]
        for container in _init_containers()
        if f"{WORKSPACE_ROOT}/.codex" in "".join(container.get("args", ()))
    ]
    assert made_here == [SEED], (
        "no init container's script creates the workspace sandbox targets, so this "
        f"test is measuring nothing: {made_here}"
    )
    order = [container["name"] for container in _init_containers()]
    assert order.index(SEED) < order.index(RESTORE), (
        f"{RESTORE} runs before {SEED} has made the sandbox targets, so bwrap builds "
        f"no sandbox for this Session: {order}"
    )


def test_the_restore_is_its_own_container_and_not_a_regular_one() -> None:
    """An init container, so it finishes before the runtime and the shim start. As a
    regular container it would race the agent for the tree it is writing."""
    regular = [container["name"] for container in _POD["spec"]["containers"]]
    assert RESTORE not in regular
    assert RESTORE in [container["name"] for container in _init_containers()]


def test_the_command_names_a_module_the_image_can_actually_run() -> None:
    """The drift check between two artifacts that travel by different routes.

    The manifest ships as a ConfigMap; the module ships inside the image. A manifest
    naming a module the wheel does not carry is an init container that exits
    `No module named ...` on every placement, for every tenant -- and it is invisible in
    this suite unless something actually imports what the command names.
    """
    command = _restore_container()["command"]
    assert command[1] == "-m", f"not a `python -m` invocation: {command}"
    module = importlib.import_module(command[2])
    assert callable(module.main)


def test_the_interpreter_is_the_one_the_image_installs_the_wheel_into() -> None:
    """An absolute path in the manifest and an `ENV VIRTUAL_ENV` in the Dockerfile are
    two spellings of one directory. Disagreeing, they produce an init container that
    exits "no such file" -- which reads as a broken image, not as a wrong path."""
    declared = re.search(
        r"^ENV VIRTUAL_ENV=(\S+)", _DOCKERFILE.read_text(), re.MULTILINE
    )
    assert declared is not None, "the session image declares no VIRTUAL_ENV"
    assert _restore_container()["command"][0] == f"{declared.group(1)}/bin/python"


def test_it_mounts_the_workspace_it_writes_and_the_token_document() -> None:
    """Exactly two, and both are volumes the pod already declared.

    `compiled` is read-only because this only reads it, and the workspace is read-write
    because writing it is the whole job. Nothing here is a new volume, which is what
    keeps the closed set over volume backings in
    `tests/pod/test_pod_holds_no_credential` passing untouched: no credential enters
    this pod in order to serve the restore.
    """
    mounts = {mount["name"]: mount for mount in _restore_container()["volumeMounts"]}
    assert set(mounts) == {"workspace", "compiled"}
    assert mounts["compiled"]["readOnly"] is True
    assert mounts["compiled"]["mountPath"] == "/etc/map/compiled"
    assert mounts["workspace"]["mountPath"] == WORKSPACE_ROOT
    assert not mounts["workspace"].get("readOnly")
    assert not mounts["workspace"].get("subPath"), (
        "a subPath here would restore into one directory of the workspace rather than "
        "into the tree the agent works in"
    )
    declared = {volume["name"] for volume in _POD["spec"]["volumes"]}
    assert set(mounts) <= declared


def test_it_takes_no_environment_and_so_is_no_new_substitution_site() -> None:
    """Everything it needs is in the document it mounts. A variable here would be a
    fifth thing `adapters/kubernetes/pod_runner` has to substitute per Session, and a
    variable it does not know about is refused there rather than left unfilled."""
    container = _restore_container()
    assert "env" not in container
    assert "envFrom" not in container


def test_a_refusal_reaches_the_pod_status_and_not_only_the_container_log() -> None:
    """`_why_it_will_not_start` reads `terminated.message`, which under the default
    `File` policy is whatever the container wrote to /dev/termination-log and is empty
    for a process that only printed. FallbackToLogsOnError promotes the log into that
    field on a non-zero exit, so the reason the restore refused becomes the reason the
    tenant's placement failed instead of something only `kubectl logs` can answer."""
    assert _restore_container()["terminationMessagePolicy"] == "FallbackToLogsOnError"


def test_it_runs_under_the_same_floor_as_every_other_container_here() -> None:
    """Capabilities and no-new-privs are container-level and are NOT inherited from the
    pod's securityContext, so a container omitting the block runs with the runtime's
    default bounding set while its siblings run with none."""
    security = _restore_container()["securityContext"]
    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"]["drop"] == ["ALL"]
    assert "seccompProfile" not in security, (
        "only agent-runtime may widen the pod's seccomp floor: only it runs bwrap"
    )


def _probe_budget_seconds() -> float:
    """How long the manifest's own probes say a healthy pod may take to come up.

    Derived from the probes rather than restated, because the number
    `_READY_TIMEOUT_SECONDS` has to clear is whatever those probes currently allow --
    and a probe made more patient without moving the bound is a pod refused while it
    was still coming up correctly.
    """
    total = 0.0
    for container in _POD["spec"]["containers"]:
        for kind in ("startupProbe", "readinessProbe"):
            probe = container.get(kind)
            if probe is not None:
                total += probe["periodSeconds"] * probe["failureThreshold"]
    return total


def test_the_probe_budget_is_actually_read_from_the_manifest() -> None:
    """Guard the guard: a manifest whose probes stopped parsing would make the bound
    below trivially satisfied, and a zero budget is indistinguishable from a patient
    one."""
    assert _probe_budget_seconds() > 0


def test_the_ready_bound_covers_the_restore_even_with_no_concurrency_at_all() -> None:
    """The other half of ADR-030's arithmetic, and the reason 180 s became 300 s.

    Two changes were needed and either alone is a coin flip: the fetch is concurrent,
    and this bound moved. Priced here at the SERIAL cost of the ceiling, so the bound
    still holds if the concurrency is lost -- which is what makes this a guard on the
    bound rather than a restatement of it. An init container that overruns does not
    merely finish late: `ensure`'s cleanup deletes the pod.
    """
    worst_case = WORKING_COUNT_LIMIT * SERIAL_ROUND_TRIP_SECONDS
    spare = (
        _READY_TIMEOUT_SECONDS - _probe_budget_seconds() - IMAGE_PULL_ALLOWANCE_SECONDS
    )
    assert spare >= worst_case, (
        f"the ready bound leaves {spare:.0f}s after the probes and the image pull, and "
        f"restoring {WORKING_COUNT_LIMIT} objects one at a time costs {worst_case:.0f}s"
    )


def test_the_fetch_is_concurrent_enough_to_matter_at_the_ceiling() -> None:
    """The half of the fix that lives in the code rather than in the bound.

    Written against the ceiling because that is what makes the number meaningful: a
    concurrency of one restores 2048 objects in ~41s and a concurrency of sixteen does
    it in ~3s, against a spare budget this file computes above. Dropping the module's
    concurrency to one fails this and fails nothing else in the suite.
    """
    concurrent_cost = (
        WORKING_COUNT_LIMIT / _CONCURRENT_FETCHES
    ) * SERIAL_ROUND_TRIP_SECONDS
    assert concurrent_cost <= 10.0, (
        f"{_CONCURRENT_FETCHES} at a time restores the {WORKING_COUNT_LIMIT}-object "
        f"ceiling in {concurrent_cost:.0f}s, most of the budget the probes and the "
        "image pull leave"
    )
