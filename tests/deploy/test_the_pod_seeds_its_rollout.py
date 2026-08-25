"""The init container that puts a Session's Rollout back, as the pod declares it.

Tier 1 (local, no infrastructure). Everything here reads `deploy/k8s/session-pod.yaml`
and the artifacts it has to agree with -- the image's venv path, the module the command
names, the volume the seeded bytes are written into. What it cannot see is a running
pod; what it can see is every way the manifest and the code could describe different
things.

The manifest travels as a ConfigMap and the module travels in the image, so those two
land by different routes and can land in either order. The import check below is the one
that matters for that: a manifest naming a module the image does not carry is a pod
whose init container exits `No module named ...` for every tenant, and nothing else in
this suite would notice.

The separation checks are the other half, and they are why this is a container rather
than three more lines in the one before it. A reader of the manifest can see that the
container writing the workspace holds no mount of the runtime's home, and that the
container writing the runtime's home holds no mount of the workspace. That property is
worth having as something anybody can check by reading, and it stops being checkable the
moment one container does both jobs.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final

import yaml

from managed_agent.control.pod_config.compiler import CODEX_HOME, WORKSPACE_ROOT
from managed_agent.session_shim.seed_rollout import (
    _SEED_BUDGET_BYTES,
    RESUMING_ENV,
)

_DEPLOY: Final[Path] = Path(__file__).resolve().parents[2] / "deploy"
_MANIFEST: Final[Path] = _DEPLOY / "k8s" / "session-pod.yaml"
_DOCKERFILE: Final[Path] = _DEPLOY / "docker" / "session.Dockerfile"
_POD: Final[dict[str, Any]] = yaml.safe_load(_MANIFEST.read_text())

SEED_ROLLOUT: Final = "seed-rollout"
RESTORE_LANE: Final = "restore-working-lane"
SEED_HOME: Final = "seed-runtime-home"
HOME_VOLUME: Final = "codex-home"
WORKSPACE_VOLUME: Final = "workspace"

MEBIBYTE: Final = 1024 * 1024

VOLUME_SHARE_FOR_THE_SEED: Final = 4
"""How much of `codex-home` the seeded Rollout may claim: one part in this many.

A quarter. The rest is for what the runtime writes beside the file -- its own sqlite
state, measured in the manifest's own comment at about 5Mi after three Turns -- and for
the file's own growth, because the runtime keeps APPENDING to what the seed wrote. The
seeded length is a floor on what that file ends up occupying, never the total, so a
budget equal to the volume would guarantee eviction rather than merely risk it.

Its own constant here rather than folded into the assertion, so that raising the budget
is a decision taken against a stated share and not a number nudged until a test passed.
"""


def _init_containers() -> list[dict[str, Any]]:
    inits: list[dict[str, Any]] = list(_POD["spec"]["initContainers"])
    return inits


def _named(name: str) -> dict[str, Any]:
    named = [c for c in _init_containers() if c["name"] == name]
    assert named, f"the pod manifest declares no {name!r} init container"
    return named[0]


def _mounts(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {mount["name"]: mount for mount in container.get("volumeMounts", ())}


# --------------------------------------------------------------------------------------
# Where it runs in the sequence
# --------------------------------------------------------------------------------------


def test_the_seed_runs_after_both_containers_that_come_before_it() -> None:
    """By name and in order, because init containers run in sequence and this is last.

    After `seed-runtime-home`, which creates the runtime home this writes into -- a
    write into a directory that does not exist yet is the whole failure. And after
    `restore-working-lane`, because a workspace restore that is going to refuse should
    refuse before a second volume has been written: a refusal stops every container
    behind it, so the likelier and cheaper failure goes first.

    A count would say none of that. Three containers in the wrong order is three
    containers.
    """
    assert [c["name"] for c in _init_containers()] == [
        SEED_HOME,
        RESTORE_LANE,
        SEED_ROLLOUT,
    ]


def test_the_seed_is_an_init_container_and_not_a_regular_one() -> None:
    """A regular container would run beside the runtime rather than before it.

    The whole point is that the file is on disk before anything opens a thread. A
    container in `spec.containers` races the shim's lifespan, and the race it loses is
    the one where the Session silently starts fresh.
    """
    assert SEED_ROLLOUT not in {c["name"] for c in _POD["spec"]["containers"]}


# --------------------------------------------------------------------------------------
# Least privilege: neither container can write the other's tree
# --------------------------------------------------------------------------------------


def test_the_seed_mounts_the_runtime_home_it_writes_and_the_token_document() -> None:
    """Exactly what it needs, at the paths the code composes from.

    `codex-home` writable, because writing the Rollout under it is the job. `compiled`
    read-only, because the Gateway's URL and this Session's token are read out of that
    document and nothing here writes it.
    """
    mounts = _mounts(_named(SEED_ROLLOUT))
    assert mounts[HOME_VOLUME]["mountPath"] == CODEX_HOME
    assert mounts[HOME_VOLUME].get("readOnly") is not True
    assert mounts["compiled"]["readOnly"] is True


def test_neither_restoring_container_can_write_the_other_ones_tree() -> None:
    """The reason this is a sibling and not three more lines in the container above it.

    Stated as a pair of absences rather than as a list of mounts, because that is the
    property: the container that writes the workspace holds no handle on the runtime's
    record, and the container that writes the record holds no handle on the workspace.
    One container doing both jobs would hold both, and no reading of the manifest could
    tell you otherwise.
    """
    seeding = _mounts(_named(SEED_ROLLOUT))
    restoring = _mounts(_named(RESTORE_LANE))

    assert WORKSPACE_VOLUME not in seeding
    assert HOME_VOLUME not in restoring
    assert restoring[WORKSPACE_VOLUME]["mountPath"] == WORKSPACE_ROOT


def test_the_seed_holds_no_shim_token_and_no_extra_volume() -> None:
    """It adds no volume to this pod, which is what keeps the credential sweep true.

    The shim's bearer is mounted into the shim and no other container. This one reads a
    different token out of a document it already has, so a mount of `shim-token` here
    would be a second credential in a pod whose whole design is to hold none.
    """
    assert set(_mounts(_named(SEED_ROLLOUT))) == {HOME_VOLUME, "compiled"}


# --------------------------------------------------------------------------------------
# What it runs, and under what floor
# --------------------------------------------------------------------------------------


def test_the_command_names_a_module_the_image_can_actually_run() -> None:
    """The manifest and the image land by different routes and can land out of order.

    A ConfigMap naming a module the wheel does not carry is a pod that exits
    `No module named ...` for every tenant of the cluster, and the manifest alone cannot
    show it. Imported here rather than string-matched, so a module renamed in the
    package fails this rather than passing a spelling check against itself.
    """
    command = _named(SEED_ROLLOUT)["command"]
    assert command[1] == "-m"
    module = importlib.import_module(command[2])
    assert callable(module.main)


def test_the_interpreter_is_the_one_the_image_installs_the_wheel_into() -> None:
    """An absolute path, because no shell stands between kubelet and this process.

    Read against the Dockerfile rather than restated: a venv that moved would leave this
    naming a binary that is not there, and the error names the path rather than the
    move.
    """
    interpreter = _named(SEED_ROLLOUT)["command"][0]
    assert interpreter.endswith("/bin/python")
    assert interpreter.rsplit("/bin/python", 1)[0] in _DOCKERFILE.read_text()


def test_a_refusal_reaches_the_pod_status_and_not_only_the_container_log() -> None:
    """Without this the reason a placement failed is one `kubectl logs` away.

    `pod_runner._why_it_will_not_start` reads `terminated.message` into the error a
    tenant's placement fails with, and under the default `File` policy that field holds
    whatever the container wrote to /dev/termination-log -- which on a refusal is
    deliberately nothing, because the reason is on stderr. This policy is what promotes
    the log into that field.
    """
    assert _named(SEED_ROLLOUT)["terminationMessagePolicy"] == "FallbackToLogsOnError"


def test_it_runs_under_the_same_floor_as_every_other_container_here() -> None:
    """Container-level and never inherited: an init container that omits the block runs
    with the runtime's default capability set in its bounding set while its siblings run
    with none."""
    security = _named(SEED_ROLLOUT)["securityContext"]
    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"]["drop"] == ["ALL"]


def test_the_resume_variable_is_declared_so_the_adapter_can_fill_it() -> None:
    """The adapter refuses a manifest that does not declare this, so this is the half of
    that contract living in the file the adapter reads.

    The default is the safe one: an un-substituted pod asks for nothing and opens a
    thread, which is wrong only for a Session that has already run -- and such a pod
    cannot be placed at all, because the substitution that would have filled this is the
    one whose absence the adapter refuses.
    """
    declared = {e["name"]: e["value"] for e in _named(SEED_ROLLOUT)["env"]}
    assert declared == {RESUMING_ENV: "false"}


# --------------------------------------------------------------------------------------
# The budget, priced against the volume it is a budget for
# --------------------------------------------------------------------------------------


def _codex_home_limit_bytes() -> int:
    volumes = {v["name"]: v for v in _POD["spec"]["volumes"]}
    declared = str(volumes[HOME_VOLUME]["emptyDir"]["sizeLimit"])
    assert declared.endswith("Mi"), declared
    return int(declared.removesuffix("Mi")) * MEBIBYTE


def test_the_volume_this_writes_into_actually_declares_a_limit() -> None:
    """The floor under the arithmetic below, which no limit at all would satisfy.

    An `emptyDir` with no `sizeLimit` draws on the node's whole disk, so the comparison
    below would be against nothing and would pass for any budget. It is asserted rather
    than assumed because the manifest itself records that two of its volumes are
    deliberately unbounded -- this is not one of them, and that is the fact being held.
    """
    assert _codex_home_limit_bytes() > 0


def test_the_seed_budget_leaves_the_runtime_room_in_the_volume_they_share() -> None:
    """The seeded Rollout may claim one part in four of `codex-home`, and no more.

    Read off the manifest rather than restated, so raising the volume and raising the
    budget stay one decision instead of two files that agree today. What the other three
    parts are for is not slack: the runtime's own sqlite state lives here, and the
    runtime keeps APPENDING to the very file the seed wrote, so the seeded length is the
    floor of what that file occupies rather than the total.

    Falsifiable in the direction that matters -- raise `_SEED_BUDGET_BYTES` past a
    quarter of the volume and this fails, which is the edit that would otherwise ship a
    pod evicted mid-Turn by its own recovery.
    """
    limit = _codex_home_limit_bytes()
    assert limit // VOLUME_SHARE_FOR_THE_SEED >= _SEED_BUDGET_BYTES, (
        f"a seeded Rollout of {_SEED_BUDGET_BYTES} bytes is more than one part in "
        f"{VOLUME_SHARE_FOR_THE_SEED} of the {limit}-byte volume it shares with "
        "everything the runtime writes"
    )


def test_the_budget_is_large_enough_to_be_worth_having() -> None:
    """The other direction, because the assertion above is satisfied by zero.

    A budget of nothing refuses every resume, which passes a ceiling check perfectly and
    is the failure this whole path exists to prevent. One mebibyte is not a measurement
    of a Rollout -- nobody here has one -- it is the floor below which the ceiling has
    stopped being a ceiling and has become a refusal.
    """
    assert _SEED_BUDGET_BYTES >= MEBIBYTE
