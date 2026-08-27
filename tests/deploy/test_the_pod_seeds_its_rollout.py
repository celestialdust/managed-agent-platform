"""The init container that prepares a Session's runtime home, as the pod declares it.

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

The mount checks are the other half, and what they can say has narrowed. One init
container now does both jobs -- the shell that builds the runtime's home and the
sandbox targets, then the Python that puts the Rollout back under it -- so the mount set
is the union of what the two of them held separately, and what is still checkable by
reading is that the union grew by nothing else and still carries no second credential.

What stopped being checkable is the property two containers made visible: that the half
reaching the network held no handle on the tenant's durable tree. It was traded for one
fewer serial container start per placement. That is recorded here, in the file whose
assertion used to hold it, because a property that is gone reads exactly like a property
nobody thought of unless somebody says which it was.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final

import yaml

from managed_agent.control.pod_config.compiler import CODEX_HOME
from managed_agent.session_shim.seed_rollout import (
    _SEED_BUDGET_BYTES,
    RESUMING_ENV,
)

_DEPLOY: Final[Path] = Path(__file__).resolve().parents[2] / "deploy"
_MANIFEST: Final[Path] = _DEPLOY / "k8s" / "session-pod.yaml"
_DOCKERFILE: Final[Path] = _DEPLOY / "docker" / "session.Dockerfile"
_POD: Final[dict[str, Any]] = yaml.safe_load(_MANIFEST.read_text())

SEED: Final = "seed-runtime-home"
"""The one init container, named for the job that governs the other.

Both halves write the runtime's home: the shell creates it and copies `config.toml` in,
the Python writes the Rollout under the same root. `seed-rollout` named only the second
and conditional half, so the surviving name is the one that describes the container.
"""

HOME_VOLUME: Final = "codex-home"
WORKSPACE_VOLUME: Final = "workspace"
CONTROL_VOLUME: Final = "control"
COMPILED_VOLUME: Final = "compiled"
SESSION_SUBTREE: Final = "MAP_TENANT_ID/MAP_SESSION_ID"
"""The un-substituted `subPath` prefix `pod_runner` fills in per Session.

Restated rather than imported so that a manifest and an adapter agreeing with each
other by both being wrong still fails here.
"""

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


def _shell_body() -> str:
    """The one `/bin/sh -c` string the merged init container runs.

    Read out of `args` rather than `command`, which carries the interpreter and its
    flag. Asserted to be a single element because a second one would be `$0` and the
    positional parameters, which is a shape this manifest does not use and which would
    make the index arithmetic below read the wrong string.
    """
    container = _named(SEED)
    assert container["command"] == ["/bin/sh", "-c"]
    args = list(container["args"])
    assert len(args) == 1, args
    return str(args[0])


def _python_invocation() -> list[str]:
    """The `exec <interpreter> -m <module>` line, split, read out of the body itself.

    Parsed rather than restated. The interpreter and the module used to be elements of a
    `command` list, where every test below could read them off the manifest; the merge
    put them inside a shell string, and a test that answered from its own constant
    instead would pass against a manifest naming a module the image does not carry --
    which is the one failure this file exists to catch.
    """
    lines = [
        line.strip()
        for line in _shell_body().splitlines()
        if line.strip().startswith("exec ")
    ]
    assert len(lines) == 1, f"expected one exec line in the shell body, found {lines}"
    return lines[0].split()


# --------------------------------------------------------------------------------------
# Where it runs in the sequence
# --------------------------------------------------------------------------------------


def test_the_pod_declares_exactly_one_init_container() -> None:
    """By name rather than by count, because a count cannot say which one survived.

    The two that were here ran in sequence, and the sequence cost a container start and
    teardown on the critical path of every placement. They are one container now; the
    ordering they encoded is inside its body and is asserted below.
    """
    assert [c["name"] for c in _init_containers()] == [SEED]


def test_the_shell_half_runs_first_and_the_python_half_last() -> None:
    """The ordering two init containers used to give for free, now a property of a body.

    The shell creates `CODEX_HOME` and copies `config.toml` into it; the Python writes
    the Rollout under that same root. A write into a directory that does not exist yet
    is the whole failure, so the direction is asserted rather than assumed -- and it is
    asserted by POSITION, because a body where the two are interleaved would satisfy a
    mere containment check.
    """
    body = _shell_body()
    home = body.index("mkdir -p /var/lib/map/codex")
    python = body.index(" ".join(_python_invocation()))
    assert home < python, "the Python half runs before the root it writes into exists"
    assert body.rstrip().endswith(" ".join(_python_invocation())), (
        "a command after the Python half would run whether or not the seed refused, "
        "and its status -- not the seed's -- would be the container's"
    )


def test_either_half_failing_takes_the_pod_down() -> None:
    """Two containers failed independently; one container has to be made to.

    `set -eu` is what stops the shell from running on past a failed `mkdir` into a
    Python half with nothing under it, and `exec` is what makes the Python process the
    container itself -- so its non-zero status is the container's status rather than
    something a shell could swallow on the way out.
    """
    body = _shell_body()
    assert body.lstrip().startswith("set -eu"), (
        "without `set -eu` a failed mkdir or cp leaves the Python half running against "
        "a home that was never built, and the container still exits 0"
    )
    assert _python_invocation()[0] == "exec", (
        "without `exec` the Python half is a child of the shell, and what the "
        "container exits with is the shell's last status rather than the seed's"
    )


def test_the_seed_is_an_init_container_and_not_a_regular_one() -> None:
    """A regular container would run beside the runtime rather than before it.

    The whole point is that the file is on disk before anything opens a thread. A
    container in `spec.containers` races the shim's lifespan, and the race it loses is
    the one where the Session silently starts fresh.
    """
    assert SEED not in {c["name"] for c in _POD["spec"]["containers"]}


# --------------------------------------------------------------------------------------
# Least privilege: neither container can write the other's tree
# --------------------------------------------------------------------------------------


def test_the_seed_mounts_the_runtime_home_it_writes_and_the_token_document() -> None:
    """Exactly what it needs, at the paths the code composes from.

    `codex-home` writable, because writing the Rollout under it is the job. `compiled`
    read-only, because the Gateway's URL and this Session's token are read out of that
    document and nothing here writes it.
    """
    mounts = _mounts(_named(SEED))
    assert mounts[HOME_VOLUME]["mountPath"] == CODEX_HOME
    assert mounts[HOME_VOLUME].get("readOnly") is not True
    assert mounts[COMPILED_VOLUME]["readOnly"] is True


def test_the_workspace_handle_it_holds_reaches_one_Session_subtree_and_no_other() -> (
    None
):
    """What is left of the absence this file used to assert, and why it is not nothing.

    While the Rollout had a container of its own it mounted no workspace at all. Merging
    the two gave the half that reaches the network a read-write handle on the durable
    copy of a tenant's tree, and the only thing standing between that handle and every
    other tenant's tree is this `subPath`. So the prefix is asserted rather than the
    mount's presence: a mount of this volume whose `subPath` does not open at a Session
    is the volume ROOT, which `pod_runner._fill_sub_paths` refuses for the same reason.
    """
    mount = _mounts(_named(SEED))[WORKSPACE_VOLUME]
    assert str(mount["subPath"]).startswith(SESSION_SUBTREE)


def test_the_seed_holds_no_shim_token_and_no_extra_volume() -> None:
    """It adds no volume to this pod, which is what keeps the credential sweep true.

    The shim's bearer is mounted into the shim and no other container. This one reads a
    different token out of a document it already has, so a mount of `shim-token` here
    would be a second credential in a pod whose whole design is to hold none.

    The set is the union of what the two containers held before they were merged, and it
    is asserted CLOSED: a merge is the moment a mount arrives because it was convenient
    rather than because something needed it, and a closed set is what makes that arrival
    fail rather than pass.
    """
    assert set(_mounts(_named(SEED))) == {
        HOME_VOLUME,
        COMPILED_VOLUME,
        CONTROL_VOLUME,
        WORKSPACE_VOLUME,
    }


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
    invocation = _python_invocation()
    assert invocation[2] == "-m", invocation
    module = importlib.import_module(invocation[3])
    assert callable(module.main)
    assert len(invocation) == 4, (
        f"the seed is run with arguments nothing reads: {invocation}"
    )


def test_the_interpreter_is_the_one_the_image_installs_the_wheel_into() -> None:
    """An absolute path, because no shell stands between kubelet and this process.

    Read against the Dockerfile rather than restated: a venv that moved would leave this
    naming a binary that is not there, and the error names the path rather than the
    move.
    """
    interpreter = _python_invocation()[1]
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
    assert _named(SEED)["terminationMessagePolicy"] == "FallbackToLogsOnError"


def test_a_status_carrying_both_halves_still_says_which_of_them_refused() -> None:
    """One container, one termination message -- so the halves have to name themselves.

    While there were two containers `_why_it_will_not_start` named the failing one and
    that was the whole answer. There is one now, and its message is the tail of a log
    both halves write, so a bare `mkdir: ...` or a bare non-zero from `test` would say
    which command failed and not which job. Each half prefixes its own refusals instead.

    The Python half's prefix is `_to_stderr`'s and is asserted where that lives; this is
    the shell half, whose every refusal goes through one function so there is one place
    for the prefix to be.
    """
    body = _shell_body()
    assert f"{SEED}: " in body, (
        "the shell half writes refusals that do not name the job that refused"
    )


def test_the_shell_half_cannot_put_a_compiled_document_into_the_pod_status() -> None:
    """New risk from the merge, and the reason the policy above is now worth re-reading.

    `FallbackToLogsOnError` promotes this container's log tail into pod status, and pod
    status reaches a tenant through the error their placement fails with. The shell half
    did not carry that policy before the merge and now it does, while holding a mount of
    `compiled` -- the document that carries this Session's bearer token.

    So the body may move that document and may not READ it: `cp` copies bytes without
    printing them, and the readers below print what they are given. `set -x` is in the
    list because it needs no reader at all -- it would trace every expansion into the
    same stream.
    """
    commands = "\n".join(
        line for line in _shell_body().splitlines() if not line.strip().startswith("#")
    )
    forbidden = ("cat ", "head ", "tail ", "od ", "xxd ", "strings ", "set -x")
    found = sorted(word for word in forbidden if word in commands)
    assert not found, (
        f"the shell half runs {found}, and its output now reaches pod status: a line "
        "of /etc/map/compiled/config.toml there is this Session's token in an error "
        "handed to whoever asked for the pod"
    )


def test_it_runs_under_the_same_floor_as_every_other_container_here() -> None:
    """Container-level and never inherited: an init container that omits the block runs
    with the runtime's default capability set in its bounding set while its siblings run
    with none."""
    security = _named(SEED)["securityContext"]
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
    declared = {e["name"]: e["value"] for e in _named(SEED)["env"]}
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
