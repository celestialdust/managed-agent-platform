"""Put the cluster objects a Session pod needs in place, and say so only when they are.

Run it against a cluster that will host Sessions, from anywhere:

    uv run python deploy/bootstrap.py

Four objects, in this order: the namespace and the identities that live in it, the
headless Service that gives every Session pod its DNS record, the ConfigMap holding the
sandbox's seccomp profile, and the DaemonSet that copies that profile onto every node.
Each is applied declaratively, so running this twice is the same as running it once.
Watch what it prints rather than trusting it: on a second run the ConfigMap step says
`configured` and not `unchanged`, because the document generated for it carries
`creationTimestamp: null` and kubectl's client-side diff is therefore never empty. The
object does not move -- its `resourceVersion` is identical across the two runs, measured
-- so `configured` here means "no change", which is a thing only this file says.

Why a script and not `kubectl apply -f deploy/k8s/`. Three reasons, and each of them is
a thing a bare apply gets wrong rather than a convenience:

* Two of the manifests belong to other concerns and carry no namespace of their own, so
  the namespace can only arrive on the command line -- and `deploy/k8s/` also holds a
  per-Session pod template and a workload whose image is a placeholder, neither of which
  may be applied at all.
* The profile is a JSON file rather than a manifest, because its bytes are also read by
  a test and pasting them into a ConfigMap would be a second copy of them. Turning a
  file into a ConfigMap is `kubectl create`, which fails on its second run; the
  generate-then-apply form below converges.
* Finishing is not the same as working. `kubectl rollout status` exits 0 for a DaemonSet
  scheduled onto zero nodes, so the profile can be installed nowhere and every command
  here still succeed. `rollout_shortfall` is what refuses that.

What this does NOT do. It creates no Secret: `map-control-plane` and `map-tool-gateway`
hold values that are not in this repository and must not be, and the three per-Session
secrets are authored by the control plane when it places a pod. It does not label the
nodes the installer selects -- `map.role=session-pod` is a nodegroup property, an AWS
resource, and therefore Terraform's. It deploys no workload.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Fixed by IAM rather than chosen here: the three roles this platform's pods assume
# each trust exactly one `system:serviceaccount:map-dev:<name>` subject. See the header
# of deploy/k8s/cluster-bootstrap.yaml.
NAMESPACE: Final = "map-dev"

PROFILE_CONFIG_MAP: Final = "map-session-seccomp"
INSTALLER_DAEMONSET: Final = "map-session-seccomp-installer"

_IDENTITIES: Final = Path("deploy/k8s/cluster-bootstrap.yaml")
_SHIM_SERVICE: Final = Path("deploy/k8s/session-shim-service.yaml")
_PROFILE: Final = Path("deploy/seccomp/session-sandbox.json")
_INSTALLER: Final = Path("deploy/k8s/session-sandbox-seccomp-installer.yaml")

_ROLLOUT_TIMEOUT: Final = "180s"


@dataclass(frozen=True, slots=True)
class Step:
    """One change to the cluster, and the document to produce first if there is one.

    `generate` is `None` for a manifest that is already a file. It is a kubectl argv
    whose stdout becomes this step's stdin for the one object that is not a file.
    """

    describe: str
    argv: tuple[str, ...]
    generate: tuple[str, ...] | None = None


def required_inputs(root: Path) -> tuple[Path, ...]:
    """Every file this reads, resolved against a repository root."""
    return tuple(
        root / relative
        for relative in (_IDENTITIES, _SHIM_SERVICE, _PROFILE, _INSTALLER)
    )


def missing_inputs(root: Path) -> tuple[Path, ...]:
    """The required files that are not there.

    Checked before anything is applied, because a partial bootstrap is worse than none:
    a DaemonSet applied without its ConfigMap leaves pods stuck creating containers, and
    a Service applied without the namespace fails on a message about the namespace while
    the operator is reading about the Service.
    """
    return tuple(path for path in required_inputs(root) if not path.is_file())


def steps(root: Path) -> tuple[Step, ...]:
    """The four applies, in the order their dependencies impose.

    The namespace and its ServiceAccounts come first because everything after them
    is namespaced, and they share one file so that kubectl's own document ordering
    puts the namespace ahead of what lives in it. The ConfigMap precedes the DaemonSet
    that mounts it -- not for correctness, since kubelet retries a missing volume, but
    because the operator watching this run should not have to know that.
    """
    return (
        Step(
            describe=f"namespace {NAMESPACE} and the identities in it",
            argv=("kubectl", "apply", "-f", str(root / _IDENTITIES)),
        ),
        Step(
            describe="the headless Service every Session pod is addressed through",
            argv=(
                "kubectl",
                "apply",
                "-n",
                NAMESPACE,
                "-f",
                str(root / _SHIM_SERVICE),
            ),
        ),
        Step(
            describe=f"{PROFILE_CONFIG_MAP}, from the profile itself",
            generate=(
                "kubectl",
                "create",
                "configmap",
                PROFILE_CONFIG_MAP,
                f"--from-file={root / _PROFILE}",
                "--dry-run=client",
                "-o",
                "yaml",
            ),
            argv=("kubectl", "apply", "-n", NAMESPACE, "-f", "-"),
        ),
        Step(
            describe="the installer that copies the profile onto every node",
            argv=("kubectl", "apply", "-n", NAMESPACE, "-f", str(root / _INSTALLER)),
        ),
    )


def rollout_shortfall(status: Mapping[str, object]) -> str | None:
    """Say why a finished rollout does not mean the profile is installed, or `None`.

    Takes a DaemonSet's `.status`. Zero desired is the case worth a function: `kubectl
    rollout status` prints "successfully rolled out" and exits 0 for a DaemonSet that
    matched no node, so the exit code cannot tell an installed profile from an
    uninstalled one. The installer selects `map.role=session-pod`, which is on the
    live nodes and not on the nodegroup that made them, so a replaced node arrives
    without it -- and a Session pod naming a profile its node does not carry stays
    Pending for ever.
    """
    desired = status.get("desiredNumberScheduled", 0)
    ready = status.get("numberReady", 0)
    if desired == 0:
        return (
            "no node matched the installer's selector, so the profile is on no node; "
            "check that map.role=session-pod is on the nodes that run Sessions"
        )
    if ready != desired:
        return f"the profile is installed on {ready} of {desired} selected nodes"
    return None


def _daemonset_status() -> Mapping[str, object]:
    """Read the installer DaemonSet's status back out of the cluster."""
    read = subprocess.run(
        ("kubectl", "get", "ds", INSTALLER_DAEMONSET, "-n", NAMESPACE, "-o", "json"),
        check=True,
        capture_output=True,
        text=True,
    )
    parsed: Mapping[str, object] = json.loads(read.stdout)["status"]
    return parsed


def main() -> int:
    """Apply every step, then refuse to call it done unless the profile landed.

    Returns 1 without touching the cluster if any input is missing, and 1 after
    applying if the installer covered fewer nodes than it selected.
    """
    root = Path(__file__).resolve().parents[1]
    absent = missing_inputs(root)
    if absent:
        for path in absent:
            print(f"missing input: {path}", file=sys.stderr)
        print("nothing was applied", file=sys.stderr)
        return 1

    for step in steps(root):
        print(f"== {step.describe}")
        if step.generate is None:
            subprocess.run(step.argv, check=True)
            continue
        generated = subprocess.run(
            step.generate, check=True, capture_output=True, text=True
        )
        subprocess.run(step.argv, check=True, input=generated.stdout, text=True)

    subprocess.run(
        (
            "kubectl",
            "rollout",
            "status",
            f"ds/{INSTALLER_DAEMONSET}",
            "-n",
            NAMESPACE,
            f"--timeout={_ROLLOUT_TIMEOUT}",
        ),
        check=True,
    )
    shortfall = rollout_shortfall(_daemonset_status())
    if shortfall is not None:
        print(f"bootstrap incomplete: {shortfall}", file=sys.stderr)
        return 1
    print(f"bootstrap complete in namespace {NAMESPACE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
