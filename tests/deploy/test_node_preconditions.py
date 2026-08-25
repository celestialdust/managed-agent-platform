"""What has to already be true of a node before a Session pod can land on it.

Two preconditions, one file, because they are the same question asked about a node that
does not exist yet -- the one a scale-up, an AMI refresh or an instance failure brings
up tomorrow.

The first is the label. Every Session pod carries a `nodeSelector`, and a node without
the matching label is a node the scheduler will not place it on: the pod stays `Pending`
with no event on the pod itself, which is the least legible way for this to fail. A
label applied with `kubectl label nodes` satisfies the selector today and is a property
of the Node object rather than of the nodegroup that creates Node objects, so a
replacement node arrives without it. The nodegroup is therefore the only place the label
can be asserted from, and that is an AWS read rather than a Kubernetes one.

The second is `user.max_user_namespaces`. The Agent Runtime confines every tool
execution with bubblewrap, which takes a user namespace per confined command, so a node
whose ceiling is too low fails under load rather than at start-up. The floor is asserted
here and installed nowhere, because it is not a switch that was turned off -- it is the
kernel's own default, and that default is derived from RAM. Which is why the failure
message carries the derivation: `7524 < 15000` on its own does not tell the next reader
that the thing to change is the instance type.

Tiering. The cases that read manifests say nothing about any node and run by default.
The nodegroup's labels and a node's actual ceiling can only be read from the account, so
those are opt-in on `MAP_CLUSTER_TESTS=1` -- and a run that skipped them has measured no
node at all.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
_SESSION_POD = "k8s/session-pod.yaml"

# The labels the nodegroup that creates Session nodes is declared to carry. One fact in
# three places -- EKS holds it on `map-dev-nodes`, MAP-60's Terraform declares it, and a
# manifest's `nodeSelector` consumes it -- and nothing in the tree links the first two
# to the third. The offline case below compares this against every selector in
# `deploy/`; the cluster case compares it against what EKS actually reports, so it is a
# declaration under test rather than a constant nobody checks.
_NODEGROUP_LABELS = {"map.role": "session-pod"}

# The floor. It arrives from the sandbox spike, which wrote it as a
# `preBootstrapCommands` sysctl in `deploy/spike/nodegroup.yaml` -- against a nodegroup
# that has no launch template, so it never ran on any node. Both live nodes measure
# 15049 without it, so this is a minimum to hold to and not a value to install.
_FLOOR = 15000

_CLUSTER = "map-dev"
_NODEGROUP = "map-dev-nodes"


def _node_selector_anywhere_in(document: object) -> dict[str, str]:
    """The `nodeSelector` a document carries, at whatever depth it sits.

    Read by walking for the key rather than by indexing `spec.nodeSelector`, which is
    where it lives on a bare Pod and nowhere else. Every workload kind nests it under
    `spec.template.spec`, and a CronJob one level deeper again, so the indexing version
    of this swept the whole tree and saw only the two bare Pods -- while
    `k8s/session-sandbox-seccomp-installer.yaml:34` is a DaemonSet whose selector
    decides which nodes get the seccomp profile at all.

    The merge is safe *within* one document for a reason that does not depend on any
    other guard: one YAML document is one Kubernetes object, and one object carries at
    most one pod template, so there is at most one selector to find. An earlier version
    of this docstring justified the merge by "the per-file document-set guards forbid
    two pods in one manifest" -- which is false: only `control-plane.yaml` and
    `tool-gateway.yaml` have such a guard, and `model-gateway.yaml` is a live workload
    with three documents and none. Across documents is a real hazard and is handled by
    the caller, not here.
    """
    found: dict[str, str] = {}
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "nodeSelector" and isinstance(value, dict):
                found.update({str(k): str(v) for k, v in value.items()})
                continue
            found.update(_node_selector_anywhere_in(value))
    elif isinstance(document, list):
        for item in document:
            found.update(_node_selector_anywhere_in(item))
    return found


def _node_selectors() -> dict[str, dict[str, str]]:
    """Every non-empty `nodeSelector` under `deploy/`, keyed by relative path.

    A new manifest selecting on a label nobody declared on the nodegroup is the same
    defect as the one this file was written for, so the sweep is over the tree rather
    than over a list of files kept up to date by hand.

    `safe_load_all` and not `safe_load`: `k8s/tool-gateway.yaml` holds two documents,
    and the single-document loader raises `ComposerError` on it rather than returning
    the first one.

    Keyed by path, and a second selector in the same file **raises** rather than
    replacing the first. The version this replaces assigned once per document, so across
    documents the last one silently overwrote the earlier -- which made a bogus selector
    in document 1 of a two-document file invisible, and made the result depend on
    document order. Raising here rather than returning a richer key keeps every caller's
    lookup by filename working while making the collision impossible to miss.
    """
    found: dict[str, dict[str, str]] = {}
    for path in sorted(_DEPLOY.rglob("*.yaml")):
        for index, document in enumerate(yaml.safe_load_all(path.read_text())):
            selector = _node_selector_anywhere_in(document)
            if not selector:
                continue
            key = str(path.relative_to(_DEPLOY))
            if key in found:
                raise AssertionError(
                    f"deploy/{key} names a nodeSelector in more than one document -- "
                    f"{found[key]} and {selector} at document index {index}. Every one "
                    "has to be graded against the nodegroup's labels, and this mapping "
                    "holds one per file. Split the file or key this by document."
                )
            found[key] = selector
    return found


def _session_selector() -> dict[str, str]:
    """The `nodeSelector` every Session pod carries.

    Asserted rather than merely looked up: the two cluster cases enumerate nodes by this
    and compare the nodegroup's labels against it, so an empty result would let both of
    them pass while measuring nothing.
    """
    selectors = _node_selectors()
    assert _SESSION_POD in selectors, (
        f"deploy/{_SESSION_POD} names no spec.nodeSelector, so there is no label to "
        "look for on the nodegroup and nothing to enumerate nodes by. Selectors were "
        f"found in {sorted(selectors)}."
    )
    return selectors[_SESSION_POD]


def test_the_deploy_tree_yields_node_selectors_to_compare() -> None:
    """The positive control. Every case below iterates a discovered collection, and a
    parse that silently produced nothing would satisfy them by being empty."""
    assert _session_selector()
    assert len(_node_selectors()) >= 2


def test_every_node_selector_in_the_tree_is_a_label_the_nodegroup_declares() -> None:
    """A selector key the nodegroup does not set is a pod that is `Pending` for ever on
    every node the nodegroup will ever create."""
    for path, selector in sorted(_node_selectors().items()):
        undeclared = {
            key: value
            for key, value in selector.items()
            if _NODEGROUP_LABELS.get(key) != value
        }
        assert not undeclared, (
            f"deploy/{path} selects on {undeclared}, which the nodegroup is not "
            f"declared to carry -- it declares {_NODEGROUP_LABELS}. Either the "
            "nodegroup's labels gain the entry (and the account is updated to match) "
            "or the manifest stops selecting on it; a pod selecting on an undeclared "
            "label is unschedulable on any node the nodegroup creates."
        )


_GATE = "MAP_CLUSTER_TESTS"

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"the node preconditions are opt-in: set {_GATE}=1 to run them. They need "
        f"kubectl pointed at {_CLUSTER} and an AWS identity that can describe the "
        "nodegroup, and the floor case creates one short-lived pod per node. SKIPPED "
        "MEANS NO NODE AND NO NODEGROUP WAS READ -- every case above reads files only."
    ),
)

_PROBE_POD = "map-node-precondition-probe"

# `cat` and `grep` only, both proven present in this image on these nodes. MemTotal is
# printed but never parsed: it is evidence for the failure message, which has to say
# what the ceiling depends on, and a parser for it would be a second thing to get wrong.
_PROBE_SCRIPT = r"""
set -u
echo "max_user_namespaces=$(cat /proc/sys/user/max_user_namespaces)"
echo "threads_max=$(cat /proc/sys/kernel/threads-max)"
grep -E '^MemTotal:' /proc/meminfo
echo probe=complete
"""


def _run(argv: list[str], stdin: str | None = None) -> str:
    done = subprocess.run(
        argv, input=stdin, capture_output=True, text=True, timeout=300
    )
    if done.returncode != 0:
        pytest.fail(f"{' '.join(argv)} failed:\n{done.stdout}\n{done.stderr}")
    return done.stdout


def _probe_manifest(node: str) -> str:
    """A pod pinned to one node by name, shaped like a Session pod where that matters.

    `nodeName` rather than a `nodeSelector`: this has to reach a named node, including
    one whose labels are the thing in question.
    """
    return yaml.safe_dump(
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": _PROBE_POD},
            "spec": {
                "restartPolicy": "Never",
                "nodeName": node,
                "automountServiceAccountToken": False,
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 10001,
                    "runAsGroup": 10001,
                },
                "containers": [
                    {
                        "name": "probe",
                        "image": "public.ecr.aws/amazonlinux/amazonlinux:2023",
                        "command": ["/bin/sh", "-c"],
                        "args": [_PROBE_SCRIPT],
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                        },
                    }
                ],
            },
        }
    )


def _session_nodes() -> dict[str, str]:
    """Node name -> instance type, for every node a Session pod could be placed on."""
    expression = ",".join(f"{k}={v}" for k, v in sorted(_session_selector().items()))
    listed = json.loads(
        _run(["kubectl", "get", "nodes", "-l", expression, "-o", "json"])
    )
    return {
        item["metadata"]["name"]: item["metadata"]["labels"].get(
            "node.kubernetes.io/instance-type", "an unreported instance type"
        )
        for item in listed["items"]
    }


def _probe(node: str) -> tuple[dict[str, str], str]:
    """Run the probe on one node; return its `key=value` findings and its transcript.

    The delete is deliberately not routed through `_run`: a `pytest.fail` raised from a
    `finally` replaces the failure that got us there, so a cleanup that cannot reach the
    cluster must leave the pod behind and the real diagnosis intact.
    """
    _run(["kubectl", "delete", "pod", _PROBE_POD, "--ignore-not-found"])
    try:
        _run(["kubectl", "apply", "-f", "-"], stdin=_probe_manifest(node))
        _run(
            [
                "kubectl",
                "wait",
                f"pod/{_PROBE_POD}",
                "--for=jsonpath={.status.phase}=Succeeded",
                "--timeout=180s",
            ]
        )
        transcript = _run(["kubectl", "logs", f"pod/{_PROBE_POD}"])
    finally:
        subprocess.run(
            ["kubectl", "delete", "pod", _PROBE_POD, "--ignore-not-found"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    findings = dict(re.findall(r"^([a-z_]+)=(\S+)$", transcript, flags=re.MULTILINE))
    return findings, transcript


@requires_the_cluster
def test_the_nodegroup_declares_every_label_a_session_pod_selects_on() -> None:
    """The durable half of the label. `kubectl label nodes` satisfies the selector on
    the nodes that exist and says nothing about the next one, so the assertion is
    against the nodegroup -- the thing that will create it."""
    described = json.loads(
        _run(
            [
                "aws",
                "eks",
                "describe-nodegroup",
                "--cluster-name",
                _CLUSTER,
                "--nodegroup-name",
                _NODEGROUP,
            ]
        )
    )
    labels: dict[str, str] = described["nodegroup"]["labels"]
    missing = {
        key: value
        for key, value in _session_selector().items()
        if labels.get(key) != value
    }
    assert not missing, (
        f"nodegroup {_NODEGROUP} reports labels {labels}, so it does not declare "
        f"{missing} -- which every Session pod selects on. Any node it creates from "
        "here comes up without that label and every Session pod scheduled onto this "
        "cluster stays Pending for ever, with the error nowhere near the pod. A label "
        "applied with `kubectl label nodes` belongs to the Node object and is lost "
        "with it. Fix in place, without replacing a node -- the braces are required, "
        "`addOrUpdateLabels=k=v` is rejected as a string where a map is wanted:\n"
        f"  aws eks update-nodegroup-config --cluster-name {_CLUSTER} "
        f"--nodegroup-name {_NODEGROUP} --labels "
        "'addOrUpdateLabels={"
        + ",".join(f"{k}={v}" for k, v in sorted(missing.items()))
        + "}'"
    )


@requires_the_cluster
def test_every_node_a_session_pod_can_land_on_clears_the_userns_floor() -> None:
    nodes = _session_nodes()
    assert nodes, (
        f"no node carries {_session_selector()}, so there is nothing here to measure "
        "and this case would pass by iterating an empty list. Either the cluster has "
        "no Session-capable node -- in which case every Session pod is already "
        "unschedulable -- or the label is missing, which the case above covers."
    )
    for node, instance_type in sorted(nodes.items()):
        findings, transcript = _probe(node)
        assert findings.get("probe") == "complete", (
            f"the probe on {node} did not finish, so its readings cannot be trusted "
            f"as measurements of anything:\n{transcript}"
        )
        ceiling = int(findings["max_user_namespaces"])
        assert ceiling >= _FLOOR, (
            f"{node} ({instance_type}) reports "
            f"user.max_user_namespaces={ceiling}, under the floor of {_FLOOR} that a "
            "Session's per-command sandbox is sized against.\n\n"
            "Nothing was switched off here. That number is the kernel's own default "
            "and the default is derived from RAM, so it is the machine that is wrong "
            "rather than a permission. Measured on this cluster's AL2023 nodes "
            "(kernel 6.1.180-225.360.amzn2023): user.max_user_namespaces is exactly "
            "kernel.threads-max/2 -- 15049 = 30098/2 -- and threads-max is derived "
            "from total RAM at boot, 30098 x 32 pages = 3.67 GiB against a MemTotal "
            "of 3.75 GiB on a 4 GiB t3.medium. So a t3.medium clears the floor by 49 "
            "and no instance type under 4 GiB reaches it at all. The lever is the "
            "nodegroup's instance type, or a sysctl installed on the node through a "
            "launch template -- and the latter replaces the nodegroup.\n\n"
            f"the probe's transcript on {node}:\n{transcript}"
        )
