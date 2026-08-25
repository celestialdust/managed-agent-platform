"""The Session sandbox's seccomp profile, and the two manifests that place it.

Tier A reads JSON and YAML. It can say that the profile is fail-closed, that its base
allowlist is the container runtime's own rather than a hand-kept copy, that the only
syscalls it lifts above that base are the ones bubblewrap needs, and that the path
the pod names is the path the installer writes. It cannot say that a kernel accepts
the filter or that a sandbox builds under it -- no test in Tier A creates a pod.

Tier B does that, against the real cluster, and is skipped unless
MAP_CLUSTER_TESTS=1. Its five findings come in pairs, and the pairing is
load-bearing. "A write outside the Permission Profile was refused" is satisfied by a
sandbox that never ran at all -- measured: under RuntimeDefault, where bwrap cannot
create a namespace, that assertion passes and so does "nothing was partly written".
What distinguishes an enforcing boundary from a broken one is the positive beside it:
a confined command ran and its write landed. The same holds for the flag mask:
`unshare --user --uts` is refused here, and it is also refused under RuntimeDefault,
and a bare unshare(CLONE_NEWUTS) is refused by the kernel -- three paths to one
errno. It is evidence only next to the control that succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_FILE = _ROOT / "deploy" / "seccomp" / "session-sandbox.json"
_K8S = _ROOT / "deploy" / "k8s"

_PROFILE: dict[str, Any] = json.loads(_PROFILE_FILE.read_text())
_POD: dict[str, Any] = yaml.safe_load((_K8S / "session-pod.yaml").read_text())
_INSTALLER: dict[str, Any] = yaml.safe_load(
    (_K8S / "session-sandbox-seccomp-installer.yaml").read_text()
)
_SPIKE: dict[str, Any] = yaml.safe_load(
    (_ROOT / "deploy" / "spike" / "pod.yaml").read_text()
)

_SECCOMP_ROOT = "/var/lib/kubelet/seccomp"
"""kubelet's seccomp root on this cluster's nodes.

Not a choice this repository makes: kubelet resolves a Localhost profile against
`<--root-dir>/seccomp`, the flag is absent from the kubelet command line on both
nodes, and its default is /var/lib/kubelet. Read off the running kubelet rather than
off documentation.
"""

_RUNTIME_CONTAINER = "agent-runtime"

_SCRATCH_MOUNT = "/tmp"
"""Writable scratch the Tier B probe adds, and the pod now declares the same mount.

Read this before trusting a Tier B pass. The runtime's first act in building a sandbox
is to create a registry of synthetic bubblewrap mount targets under the system
temporary directory. `agent-runtime` is readOnlyRootFilesystem, so with nothing mounted
here that mkdir is EROFS, the runtime panics with `failed to create synthetic
bubblewrap mount registry ... Read-only file system`, and no confined command runs --
the same end state this slice's profile exists to fix, reached by a route that has
nothing to do with seccomp. That is what this mount was here to rule out, and for a
while it was the ONE PLACE THIS PROBE WAS WIDER THAN THE POD.

Measured on map-dev, three ways: with this mount the confined command runs; without
it, it panics; and pointing TMPDIR at the codex-home volume instead avoids the panic
but makes the runtime log `Refusing to create helper binaries under temporary dir`
and proceed degraded, so a spare env var is not the fix either.

The pod now mounts this too, so the probe is no longer wider than the pod in this
respect and a Tier B pass no longer hides a Session that cannot run a tool call. The
probe keeps its own volume rather than reading the pod's: it builds its own pod around
its own profile and its own writable root, and a mount it declares itself is one fewer
thing that has to be true of another file for this test to mean what it says.

What a Tier B pass still does NOT prove: that a pod built from session-pod.yaml runs a
confined command. This profile and these paths are not that profile and those paths --
`tests/pod/test_the_pod_materialises_its_sandbox_targets.py` is where that is measured.
"""

# containerd's default profile allows these 23 only when the container holds
# CAP_SYS_ADMIN, which a Session pod never does -- so under RuntimeDefault every one
# of them answers the default errno. Pinned here from
# contrib/seccomp/seccomp_default.go at v2.2.5 because the size of the set this
# profile lifts out of it is the whole security question, and 23 names is small
# enough to quote and stable enough to pin.
_CAP_SYS_ADMIN_ONLY = frozenset(
    {
        "bpf",
        "clone",
        "clone3",
        "fanotify_init",
        "fsconfig",
        "fsmount",
        "fsopen",
        "fspick",
        "lookup_dcookie",
        "mount",
        "mount_setattr",
        "move_mount",
        "open_tree",
        "perf_event_open",
        "quotactl",
        "quotactl_fd",
        "setdomainname",
        "sethostname",
        "setns",
        "syslog",
        "umount",
        "umount2",
        "unshare",
    }
)

_CONTAINERD_BASE_DIGEST = (
    "1fbc170ee2bbb543fb416ba53402dcb65c6af15cf337ca7dba5b79951f859853"
)
"""sha256 of the container runtime's base allowlist, newline-joined in source order.

Its source is the first `Names: []string{...}` block of
`contrib/seccomp/seccomp_default.go` in github.com/containerd/containerd at tag
`v2.2.5` -- the function that produces RuntimeDefault on this cluster's nodes.
Re-verify the claim by extracting those 363 names in source order, joining them with
newlines and taking sha256 of the result.

A digest rather than the names, because a second copy of the allowlist in this
repository is the drift it exists to detect: two lists that can disagree, with no way
to tell which one is lying.
"""

_NEWUTS = 0x04000000
_NEWCGROUP = 0x02000000


def _rules_naming(syscall: str) -> list[dict[str, Any]]:
    return [r for r in _PROFILE["syscalls"] if syscall in r["names"]]


def _allowed_unconditionally() -> frozenset[str]:
    """Every name this profile allows with no argument filter on it."""
    return frozenset(
        name
        for rule in _PROFILE["syscalls"]
        if rule["action"] == "SCMP_ACT_ALLOW" and "args" not in rule
        for name in rule["names"]
    )


def _base_rule_names() -> list[str]:
    """The inherited rule: the largest unconditional allow, unsplit and unreordered.

    Identified by size rather than by position, so reordering the file's rules does
    not move which one the provenance digest is taken over. Nothing else in the
    profile comes close -- the next largest names three syscalls.
    """
    largest = max(_PROFILE["syscalls"], key=lambda rule: len(rule["names"]))
    names: list[str] = largest["names"]
    return names


def _container(pod: dict[str, Any], name: str) -> dict[str, Any]:
    found = [c for c in pod["spec"]["containers"] if c["name"] == name]
    assert found, f"no container named {name}"
    first: dict[str, Any] = found[0]
    return first


def test_every_document_this_file_reads_parsed_into_something() -> None:
    """The positive half. Several cases below read discovered collections and an
    empty one would satisfy them."""
    assert len(_PROFILE["syscalls"]) >= 10
    assert len(_allowed_unconditionally()) >= 300
    assert _POD["kind"] == "Pod"
    assert _INSTALLER["kind"] == "DaemonSet"
    assert _SPIKE["kind"] == "Pod"


def test_the_profile_denies_every_syscall_it_does_not_name() -> None:
    """Fail-safe defaults: the default action is a refusal, and it is EPERM rather
    than a kill so a caller sees an errno it can report."""
    assert _PROFILE["defaultAction"] == "SCMP_ACT_ERRNO"
    assert _PROFILE["defaultErrnoRet"] == 1


def test_the_profile_covers_the_three_architectures_the_nodegroup_runs() -> None:
    """map-dev-nodes is AL2023_x86_64_STANDARD. The two 32-bit personalities are here
    because a 64-bit process can issue a syscall under either, and an architecture
    absent from this list is not filtered at all."""
    assert _PROFILE["architectures"] == [
        "SCMP_ARCH_X86_64",
        "SCMP_ARCH_X86",
        "SCMP_ARCH_X32",
    ]


def test_the_base_allowlist_is_the_runtimes_own_and_not_a_hand_kept_copy() -> None:
    """The provenance assertion, and the reason the rest of this file can be short.

    The base of this profile is not authored here: it is the container runtime's own
    default allowlist, so the security question is only ever "what did we add to it".
    A profile whose base had been edited -- one name dropped to shrink it, one added
    to make some tool work -- would answer that question against a moved baseline,
    and every delta assertion below would still pass.

    Held as a digest and not as a list. A second copy of 363 names in this repository
    is exactly the drift this detects.
    """
    base = _base_rule_names()
    digest = hashlib.sha256("\n".join(base).encode()).hexdigest()
    assert len(base) == 363, len(base)
    assert digest == _CONTAINERD_BASE_DIGEST, digest


def test_the_only_cap_sys_admin_syscalls_this_profile_lifts_are_two_mounts() -> None:
    """The security question, asked as a set operation.

    Under the container runtime's default these 23 are reachable only with
    CAP_SYS_ADMIN, which a Session pod drops. This profile allows two of them
    outright -- the mount pair bubblewrap uses to build its bind mounts -- and leaves
    19 refused. `clone` and `unshare` are in the set too and are allowed only under
    the mask the next case checks, which is why this reads the unconditional set and
    not every ALLOW rule.
    """
    lifted = _allowed_unconditionally() & _CAP_SYS_ADMIN_ONLY
    assert lifted == {"mount", "umount2"}, sorted(lifted)


def test_pivot_root_is_the_third_addition_and_it_is_allowed() -> None:
    """The positive half of the case above. pivot_root is in neither the runtime
    default's base list nor its CAP_SYS_ADMIN block -- the default refuses it with no
    rule at all -- so a set intersection cannot see it."""
    assert "pivot_root" in _allowed_unconditionally()


def test_the_unconditional_allow_set_is_the_size_it_was_measured_at() -> None:
    """371 = 363 (containerd v2.2.5 base) + 3 (kernel >= 4.8 arm) + 2 (amd64 arm)
    + 3 (mount, umount2, pivot_root, this slice's additions).

    A count rather than a list, and deliberately brittle: adding a syscall to make
    something work fails here, with this arithmetic in the message.
    """
    assert len(_allowed_unconditionally()) == 371


def test_clone_and_unshare_are_allowed_only_under_a_mask() -> None:
    """One rule for both, one condition on it, and the mask forbids the two
    namespaces the compiled bwrap argv does not ask for.

    MASKED_EQ reads `(arg0 & value) == valueTwo`, so `value` is the mask -- the same
    order containerd's own clone rule uses. Reversed, the rule permits nothing and
    the sandbox stops building.
    """
    for syscall in ("clone", "unshare"):
        rules = _rules_naming(syscall)
        allows = [r for r in rules if r["action"] == "SCMP_ACT_ALLOW"]
        assert len(allows) == 1, f"{syscall} has {len(allows)} ALLOW rules"
        args = allows[0]["args"]
        assert len(args) == 1
        assert args[0]["op"] == "SCMP_CMP_MASKED_EQ"
        assert args[0]["index"] == 0
        assert args[0]["value"] == _NEWUTS | _NEWCGROUP
        assert args[0]["valueTwo"] == 0


def test_clone3_answers_enosys_so_a_libc_falls_back_instead_of_failing() -> None:
    """ENOSYS and not EPERM, and it is not an accident: glibc issues clone3 and falls
    back to clone when the kernel says the call does not exist. EPERM is not that
    signal and fork would fail outright."""
    rules = _rules_naming("clone3")
    assert [r["action"] for r in rules] == ["SCMP_ACT_ERRNO"]
    assert rules[0]["errnoRet"] == 38


def test_no_rule_allows_any_syscall_on_the_denied_list() -> None:
    """The families this profile must never lift, whatever else it grows.

    Each is refused by the container runtime's default and measured still refused
    under this profile. Named rather than derived, because a derived list would need a
    second copy of the runtime default in this repository.
    """
    denied = {
        "acct",
        "add_key",
        "bpf",
        "chroot",
        "delete_module",
        "finit_module",
        "fsconfig",
        "fsmount",
        "fsopen",
        "fspick",
        "init_module",
        "io_uring_enter",
        "io_uring_register",
        "io_uring_setup",
        "kexec_load",
        "keyctl",
        "mount_setattr",
        "move_mount",
        "open_tree",
        "perf_event_open",
        "quotactl",
        "reboot",
        "request_key",
        "setdomainname",
        "sethostname",
        "setns",
        "swapoff",
        "swapon",
        "syslog",
        "userfaultfd",
    }
    for rule in _PROFILE["syscalls"]:
        if rule["action"] != "SCMP_ACT_ALLOW":
            continue
        overlap = denied & set(rule["names"])
        assert not overlap, f"{sorted(overlap)} allowed by {rule}"


def test_the_runtime_container_names_the_session_sandbox_profile() -> None:
    runtime = _container(_POD, _RUNTIME_CONTAINER)
    profile = runtime["securityContext"]["seccompProfile"]
    assert profile == {
        "type": "Localhost",
        "localhostProfile": "map/session-sandbox.json",
    }


def test_the_pod_level_floor_is_still_the_runtime_default() -> None:
    """Every container that does not override inherits this, and the two that do not
    override are the shim and the init container -- neither runs bwrap."""
    assert _POD["spec"]["securityContext"]["seccompProfile"] == {
        "type": "RuntimeDefault"
    }


def test_no_other_container_overrides_the_floor() -> None:
    """One container may be wider than the floor and it is named. A second override,
    or an Unconfined anywhere, would widen a container that runs no sandbox and would
    be invisible to a policy that reads only the pod."""
    every = _POD["spec"]["containers"] + _POD["spec"].get("initContainers", [])
    others = [c for c in every if c["name"] != _RUNTIME_CONTAINER]
    assert others, "nothing to check: this pod has only agent-runtime"
    for container in others:
        assert "seccompProfile" not in container.get("securityContext", {}), (
            f"{container['name']} overrides the pod's seccomp floor"
        )


def test_the_installer_writes_the_profile_at_the_path_the_pod_names() -> None:
    """The drift check. The pod's localhostProfile is relative to kubelet's seccomp
    root; the installer's hostPath is that root plus a directory, and its command
    writes one file into it. Composed, they must be the same string the pod asks
    for."""
    spec = _INSTALLER["spec"]["template"]["spec"]
    mount_dir: str = next(
        v["hostPath"]["path"] for v in spec["volumes"] if "hostPath" in v
    )
    relative = mount_dir.removeprefix(f"{_SECCOMP_ROOT}/")
    assert relative != mount_dir, f"{mount_dir} is not under {_SECCOMP_ROOT}"
    named = _container(_POD, _RUNTIME_CONTAINER)["securityContext"]
    asked = named["seccompProfile"]["localhostProfile"]
    assert f"{relative}/{_PROFILE_FILE.name}" == asked


def test_the_installer_reads_the_key_that_is_the_profile_files_basename() -> None:
    """The ConfigMap is created from deploy/seccomp/session-sandbox.json, so its key
    is that basename and the install command must name it. Renaming the profile file
    without touching the command fails here rather than on a node."""
    spec = _INSTALLER["spec"]["template"]["spec"]
    command: str = spec["containers"][0]["args"][0]
    assert f"/src/{_PROFILE_FILE.name}" in command
    assert command.count(_PROFILE_FILE.name) >= 2


def test_the_installer_needs_no_privilege_and_adds_no_capability() -> None:
    """It writes one file. A privileged pod on every node is a standing cost that
    this does not need, measured."""
    container = _INSTALLER["spec"]["template"]["spec"]["containers"][0]
    security = container["securityContext"]
    assert security.get("privileged") is None
    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"] == {"drop": ["ALL"]}
    assert security["seccompProfile"] == {"type": "RuntimeDefault"}


def test_the_installer_runs_where_a_session_pod_can_run() -> None:
    """A node the scheduler may place a Session pod on and the installer may not is a
    node whose Session pods never start."""
    assert (
        _INSTALLER["spec"]["template"]["spec"]["nodeSelector"]
        == _POD["spec"]["nodeSelector"]
    )


_GATE = "MAP_CLUSTER_TESTS"

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"the cluster proof is opt-in: set {_GATE}=1 to run it. It needs kubectl "
        "pointed at map-dev, creates a DaemonSet, a ConfigMap and a pod, and takes "
        "one to three minutes. SKIPPED MEANS NO KERNEL EVER LOADED THIS PROFILE -- "
        "every case above reads files only."
    ),
)

_PROBE_POD = "map-seccomp-proof"
_CONFIGMAP = "map-session-seccomp"
_DAEMONSET = "map-session-seccomp-installer"

_PROBE_SCRIPT = r"""
set -u
echo "seccomp-mode=$(awk '/^Seccomp:/{print $2}' /proc/self/status)"
mkdir -p /work/ws "$CODEX_HOME"
{
  printf 'default_permissions = "p"\n'
  printf '[permissions.p.filesystem]\n"/" = "read"\n"/work/ws" = "write"\n'
} > "$CODEX_HOME/config.toml"
codex sandbox -P p -- /bin/sh -c 'printf allowed > /work/ws/ok.txt' \
  && echo confined-command-runs=ok || echo confined-command-runs=FAIL
test "$(cat /work/ws/ok.txt 2>/dev/null)" = allowed \
  && echo confined-write-landed=ok || echo confined-write-landed=FAIL
codex sandbox -P p -- /bin/sh -c 'printf x > /work/outside.txt' 2>/dev/null \
  && echo write-outside-profile=ALLOWED || echo write-outside-profile=refused
unshare --user --map-root-user --mount --pid --ipc --net true 2>/dev/null \
  && echo sandbox-namespaces=ok || echo sandbox-namespaces=FAIL
unshare --user --map-root-user --uts true 2>/dev/null \
  && echo uts-namespace=ALLOWED || echo uts-namespace=refused
echo probe=complete
"""


def _kubectl(*argv: str, stdin: str | None = None) -> str:
    done = subprocess.run(
        ["kubectl", *argv],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if done.returncode != 0:
        pytest.fail(f"kubectl {' '.join(argv)} failed:\n{done.stderr}")
    return done.stdout


def _probe_manifest() -> str:
    """A pod shaped like session-pod.yaml where it matters: the same uid, the same
    dropped capabilities, the same pod-level floor, and the container-level override
    read out of session-pod.yaml itself rather than restated.

    CODEX_HOME is pointed into the writable volume rather than left at the image's own
    value, because the security context copied above carries readOnlyRootFilesystem
    and the runtime refuses to run at all when it cannot write its configuration tree.
    session-pod.yaml solves the same problem the same way -- an emptyDir mounted over
    CODEX_HOME -- so this is the real pod's shape and not a concession to the probe.

    The scratch mount is the one deliberate exception: see _SCRATCH_MOUNT, which says
    what the pod is missing and why the difference is not this slice's to close.
    """
    runtime = _container(_POD, _RUNTIME_CONTAINER)
    pod: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": _PROBE_POD},
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": _POD["spec"]["nodeSelector"],
            "automountServiceAccountToken": False,
            "securityContext": _POD["spec"]["securityContext"],
            "containers": [
                {
                    "name": "probe",
                    "image": _SPIKE["spec"]["containers"][0]["image"],
                    "imagePullPolicy": "Always",
                    "securityContext": runtime["securityContext"],
                    "command": ["/bin/sh", "-c", _PROBE_SCRIPT],
                    "env": [{"name": "CODEX_HOME", "value": "/work/codex"}],
                    "volumeMounts": [
                        {"name": "work", "mountPath": "/work"},
                        {"name": "scratch", "mountPath": _SCRATCH_MOUNT},
                    ],
                }
            ],
            "volumes": [
                {"name": "work", "emptyDir": {"sizeLimit": "256Mi"}},
                {"name": "scratch", "emptyDir": {"sizeLimit": "16Mi"}},
            ],
        },
    }
    dumped: str = yaml.safe_dump(pod)
    return dumped


@pytest.fixture(scope="module")
def profile_on_every_node() -> Iterator[None]:
    """Install the profile the way the installer's own comment says to, then take the
    two objects down again.

    The ConfigMap is generated from the profile file rather than from a manifest, so
    the documented procedure is the tested one. The file itself is left on the nodes:
    it is production state, and removing it would leave the cluster in a state where
    no Session pod can start.
    """
    generated = _kubectl(
        "create",
        "configmap",
        _CONFIGMAP,
        f"--from-file={_PROFILE_FILE}",
        "--dry-run=client",
        "-o",
        "yaml",
    )
    _kubectl("apply", "-f", "-", stdin=generated)
    _kubectl("apply", "-f", str(_K8S / "session-sandbox-seccomp-installer.yaml"))
    _kubectl("rollout", "status", f"ds/{_DAEMONSET}", "--timeout=180s")
    yield
    _kubectl("delete", "ds", _DAEMONSET, "--ignore-not-found")
    _kubectl("delete", "cm", _CONFIGMAP, "--ignore-not-found")


def _findings(pod: str) -> dict[str, str]:
    """Poll until the probe reaches a terminal phase, then split its transcript.

    Polls the phase rather than the log, because `kubectl logs` on a pod that has not
    started a container fails, and a failed kubectl here would be reported as a
    missing profile rather than as a pod that had not been scheduled yet.
    """
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        phase = _kubectl("get", "pod", pod, "-o", "jsonpath={.status.phase}")
        if phase in ("Succeeded", "Failed"):
            break
        time.sleep(3)
    transcript = _kubectl("logs", pod)
    found = dict(
        line.split("=", 1) for line in transcript.split() if line.count("=") == 1
    )
    assert found.get("probe") == "complete", transcript
    return found


@requires_the_cluster
def test_the_installer_covers_every_node_and_writes_the_whole_profile(
    profile_on_every_node: None,
) -> None:
    """Coverage from the DaemonSet's own status, content from the byte count.

    Both halves were arrived at by falsifying the obvious version of this test, which
    counted log lines starting with "installed". Measured: delete the `install` and
    `test -s` lines from the manifest so nothing whatever reaches the node, and that
    version still passes -- `wc -c` on a missing file fails inside the command
    substitution, `set -e` does not abort because `echo` itself succeeds, and the line
    "installed  bytes" still starts with "installed". The transcript read
    `/seccomp-root/map/session-sandbox.json: No such file or directory` directly above
    a passing assertion.

    So coverage is read off `status` rather than counted from logs -- a leftover pod
    from a previous generation is still terminating for a few seconds after a rollout
    and contributes its own log line, which made the counted version timing-dependent
    as well -- and content is the exact size of the file in this repository, which an
    absent, truncated or stale profile all fail.
    """
    nodes = _kubectl("get", "nodes", "-l", "map.role=session-pod", "-o", "name")
    eligible = len(nodes.split())
    assert eligible >= 1, "no node is labelled to run a Session pod"

    scheduled = _kubectl(
        "get",
        "ds",
        _DAEMONSET,
        "-o",
        "jsonpath={.status.desiredNumberScheduled} {.status.numberReady}",
    )
    desired, ready = (int(field) for field in scheduled.split())
    assert desired == eligible, f"{desired} scheduled for {eligible} eligible nodes"
    assert ready == desired, f"{ready} of {desired} installers ready"

    expected = f"installed {_PROFILE_FILE.stat().st_size} bytes"
    logs = _kubectl("logs", "-l", "map.role=seccomp-installer", "--tail=5")
    reported = [line for line in logs.splitlines() if line.startswith("installed")]
    assert reported, logs
    assert all(line.strip() == expected for line in reported), (expected, logs)


@requires_the_cluster
def test_a_confined_command_runs_and_the_boundary_holds(
    profile_on_every_node: None,
) -> None:
    """The checkpoint. Five findings, in pairs.

    confined-command-runs and confined-write-landed are the positives: bwrap built a
    sandbox and the write inside the Permission Profile reached the disk. Under
    RuntimeDefault both fail while write-outside-profile=refused still passes, which
    is why the negative is never read on its own.

    sandbox-namespaces and uts-namespace are the boundary and its control: the five
    namespaces the compiled bwrap argv asks for succeed, and the one it does not ask
    for is refused by this profile's mask -- a call that succeeds under Unconfined.
    """
    _kubectl("delete", "pod", _PROBE_POD, "--ignore-not-found")
    _kubectl("apply", "-f", "-", stdin=_probe_manifest())
    try:
        found = _findings(_PROBE_POD)
        # Read straight out of /proc rather than inferred: 2 is a filter installed,
        # 0 is Unconfined. It is the one finding that cannot be produced by a pod
        # running without the profile this slice ships.
        assert found["seccomp-mode"] == "2", found
        assert found["confined-command-runs"] == "ok", found
        assert found["confined-write-landed"] == "ok", found
        assert found["write-outside-profile"] == "refused", found
        assert found["sandbox-namespaces"] == "ok", found
        assert found["uts-namespace"] == "refused", found
    finally:
        _kubectl("delete", "pod", _PROBE_POD, "--ignore-not-found", "--wait=false")
