"""The cluster really holds the objects a Session pod cannot start without.

Two defects of the same shape were found on 2026-08-22 and neither was visible
to any existing check. `deploy/k8s/session-sandbox-seccomp-installer.yaml` had
never been applied, so the seccomp profile sat on the two nodes that happened to
exist with nothing to put it on a third. `deploy/k8s/cluster-bootstrap.yaml` had
never been applied, so the namespace every Session pod is placed into did not
exist. A manifest committed to the repo and a manifest running in the cluster
are different facts, and every check in this suite read only the former.

These read the live cluster, so they need credentials and are marked `network`.
They assert *prerequisites* rather than every file under `deploy/k8s/`:
`session-pod.yaml` is a per-Session template rendered and never applied as-is,
and `tool-gateway.yaml` names an image its ECR repository does not hold yet.
Asserting those were deployed would assert something false.

The profile check compares bytes, not mere presence, because the failure that
motivated it is a stale copy rather than a missing one — a hand-placed profile
and a reconciled one look identical until the file on disk changes and only one
of them follows.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.network

_PROFILE = Path("deploy/seccomp/session-sandbox.json")
_NAMESPACE = "map-dev"
_INSTALLER = "map-session-seccomp-installer"
_PROFILE_CONFIGMAP = "map-session-seccomp"

_UNREACHABLE = (
    "connection refused",
    "was refused",
    "no configuration has been provided",
    "unable to connect",
)


def _kubectl(*args: str) -> dict[str, object]:
    """Run kubectl and parse its JSON, skipping when there is no cluster.

    A missing cluster and a missing object are different outcomes and only the
    second is a failure. `kubectl` exits non-zero for both, so the message on
    stderr is the only thing that separates them.
    """
    done = subprocess.run(
        ["kubectl", *args, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if done.returncode != 0:
        lowered = done.stderr.lower()
        if any(marker in lowered for marker in _UNREACHABLE):
            pytest.skip(f"no reachable cluster: {done.stderr.strip()[:200]}")
        pytest.fail(f"kubectl {' '.join(args)} failed:\n{done.stderr.strip()}")
    parsed: dict[str, object] = json.loads(done.stdout)
    return parsed


def _the_one(kind: str, name: str) -> dict[str, object]:
    """Find a cluster-unique object by name without knowing its namespace.

    `kubectl get <kind> <name> -A` is rejected outright — a name and
    `--all-namespaces` cannot be combined — and hard-coding the namespace would
    make this check pass or fail on where somebody happened to apply the
    manifest rather than on whether it is running. A field selector searches
    every namespace and returns a list, so an empty list is the real "not
    deployed" signal this file exists to catch.
    """
    listing = _kubectl("get", kind, "-A", f"--field-selector=metadata.name={name}")
    items = listing.get("items")
    assert isinstance(items, list), f"unexpected kubectl output for {kind}/{name}"
    assert items, (
        f"no {kind} named {name} exists in any namespace. The manifest for it is "
        "committed under deploy/k8s/ but was never applied to this cluster."
    )
    assert len(items) == 1, (
        f"{len(items)} objects named {name} across namespaces; expected exactly "
        "one, so it is ambiguous which one a Session pod would resolve against"
    )
    first = items[0]
    assert isinstance(first, dict)
    return first


def test_the_namespace_every_session_pod_is_placed_into_exists() -> None:
    """Without it a Session pod cannot be created at all.

    The name is not free: three IAM roles' trust policies allow
    `sts:AssumeRoleWithWebIdentity` for subjects under
    `system:serviceaccount:map-dev:`, so a pod placed anywhere else holds a
    projected token no role will exchange.
    """
    ns = _kubectl("get", "namespace", _NAMESPACE)
    status = ns.get("status")
    assert isinstance(status, dict)
    assert status.get("phase") == "Active", (
        f"namespace {_NAMESPACE} is not Active: {status}"
    )


def test_the_seccomp_profile_reaches_every_node_not_merely_the_ones_that_existed() -> (
    None
):
    """A node that joins later must get the profile too.

    `numberReady` short of `desiredNumberScheduled` means at least one node is
    refusing Session pods, or running them against whatever profile it happens
    to have. Either way the fleet is not uniform, which is the one thing this
    DaemonSet exists to make true.
    """
    ds = _the_one("daemonset", _INSTALLER)
    status = ds.get("status")
    assert isinstance(status, dict)
    desired = status.get("desiredNumberScheduled")
    ready = status.get("numberReady")
    assert desired, f"the installer schedules onto no node: {status}"
    assert ready == desired, (
        f"the seccomp installer is ready on {ready} of {desired} nodes. A node "
        "without it refuses Session pods rather than running them unconfined, "
        "but it still breaks every Session placed there."
    )


def test_the_installed_profile_is_the_one_this_repo_declares() -> None:
    """Presence is not enough — a stale copy is the failure easiest to miss.

    The DaemonSet mounts this ConfigMap and copies it to the path kubelet
    resolves a `Localhost` profile against, so these bytes are what every
    Session container is actually filtered by. Parsed JSON is compared rather
    than raw text, so a trailing newline that changes no syscall rule does not
    fail the check.
    """
    holder = _the_one("configmap", _PROFILE_CONFIGMAP)
    data = holder.get("data")
    assert isinstance(data, dict), (
        f"ConfigMap {_PROFILE_CONFIGMAP} carries no data: {holder}"
    )
    installed = data.get(_PROFILE.name)
    assert installed is not None, (
        f"ConfigMap {_PROFILE_CONFIGMAP} has no key {_PROFILE.name}. The "
        "DaemonSet mounts it by that name, so the install silently copies "
        f"nothing. Keys present: {sorted(data)}"
    )
    assert json.loads(installed) == json.loads(_PROFILE.read_text()), (
        f"the profile in the cluster differs from {_PROFILE}. Recreate it:\n"
        f"  kubectl create configmap {_PROFILE_CONFIGMAP} "
        f"--from-file={_PROFILE} --dry-run=client -o yaml | kubectl apply -f -\n"
        f"  kubectl rollout restart ds/{_INSTALLER}"
    )


_BOOTSTRAP = Path("deploy/k8s/cluster-bootstrap.yaml")


def _manifests_bootstrap_applies() -> tuple[Path, ...]:
    """Every manifest path `deploy/bootstrap.py` hands to `kubectl apply`.

    Read out of that module's own constants, so a fourth file it starts applying is
    covered the day the constant is added. A list written here would be the thing that
    went stale -- it already had, silently, for `session-shim-service.yaml`.

    `_PROFILE` is deliberately excluded and named rather than skipped: it is the seccomp
    profile document, applied as data inside a ConfigMap the installer mounts, so it
    declares no Kubernetes object of its own and has no name to look up. The three
    below each declare objects.
    """
    spec = importlib.util.spec_from_file_location(
        "map_bootstrap_for_cluster_holds", Path("deploy/bootstrap.py").resolve()
    )
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bootstrap
    spec.loader.exec_module(bootstrap)
    paths = tuple(
        Path(str(getattr(bootstrap, name)))
        for name in ("_IDENTITIES", "_SHIM_SERVICE", "_INSTALLER")
    )
    assert len(paths) == 3, paths
    for path in paths:
        assert path.exists(), f"bootstrap.py names {path}, which is not in the tree"
    return paths


def _bootstrap_objects() -> list[tuple[str, str]]:
    """(kind, name) for every document in every manifest `deploy/bootstrap.py` applies.

    Read from bootstrap's own path constants rather than from a list here, because that
    is the difference this file exists to make and it was only half made. It read ONE of
    the three files bootstrap applies -- `cluster-bootstrap.yaml` -- and a `Service`
    declared in `session-shim-service.yaml`, applied by nothing, was invisible to it.

    That cost a whole diagnosis on 2026-08-23. The Service is the headless one every
    Session pod answers on, so with it absent a Session pod reached `2/2 Running` with
    its shim replying `/session/ready` 204 to the kubelet, and the Turn still failed
    `the shim for session ... could not be reached`: the control plane dials
    `<pod>.map-session.<ns>.svc.cluster.local`, which resolved to nothing. Every probe
    said the pod was healthy, because it was.

    Discovering the *documents* in one file while *listing* the files was the same
    looks-like-coverage shape this docstring's next paragraph was written about, one
    level up. Now both levels are discovered.

    The original note, kept because its lesson is the same one:

    Discovered rather than listed. The two cases in this file were written after two
    manifests that had never been applied, and a third one landed anyway on 2026-08-23:
    `ServiceAccount/model-gateway` was declared at `cluster-bootstrap.yaml:106` and did
    not exist, so MAP-69's first apply produced 0/2 replicas with
    `error looking up service account map-dev/model-gateway`. The checks above named the
    namespace, the DaemonSet and the ConfigMap **by hand**, and a ServiceAccount added
    to a file they already read was invisible to all three.

    So this reads the file. A fifth object added to it is covered the moment it is
    committed, which is the only version of this check that stops the class rather than
    the instance.
    """
    documents = [
        document
        for manifest in _manifests_bootstrap_applies()
        for document in yaml.safe_load_all(manifest.read_text())
    ]
    return [(d["kind"], d["metadata"]["name"]) for d in documents if d]


def test_every_object_the_bootstrap_manifest_declares_is_running() -> None:
    """Committed and applied are different facts, for every object in that file.

    Asserted one at a time rather than as a set difference so the failure names the
    object that is missing. Somebody reading this failure needs to know what to apply,
    not that a count disagreed.
    """
    declared = _bootstrap_objects()
    assert declared, f"{_BOOTSTRAP} declares nothing, so this proves nothing"
    for kind, name in declared:
        _the_one(kind, name)


def test_the_bootstrap_manifest_declares_what_the_hand_checks_assume() -> None:
    """The hand-named checks above and the discovered one must be about the same file.

    Not redundant with the case above: that one asserts the cluster matches the file,
    and this asserts the file still contains what the older checks assume. If the
    namespace were moved out of the bootstrap manifest into another one, the discovered
    check would silently stop covering it while continuing to pass, which is the
    looks-like-coverage failure this whole file exists to prevent.
    """
    kinds = {kind for kind, _ in _bootstrap_objects()}
    names = {name for _, name in _bootstrap_objects()}
    assert "Namespace" in kinds and _NAMESPACE in names, (
        f"{_BOOTSTRAP} no longer declares namespace {_NAMESPACE}; the hand-written "
        "check above is now asserting something this file does not promise"
    )
    assert "ServiceAccount" in kinds, (
        f"{_BOOTSTRAP} declares no ServiceAccount. It held four on 2026-08-23, and one "
        "of them being absent from the cluster is what this file was extended for."
    )


_SESSION_POD = Path("deploy/k8s/session-pod.yaml")


def _pod_manifest_config_map() -> str:
    """The ConfigMap name, read out of the script that creates it.

    Not a literal here, for the reason `test_control_plane_manifest.py` reads its
    namespace out of `deploy/bootstrap.py`: a name spelled in the applier and again in
    the checker is two spellings free to diverge, and the divergence would show up as
    this check passing against an object nothing mounts.
    """
    spec = importlib.util.spec_from_file_location(
        "map_platform_for_cluster_holds", Path("deploy/platform.py").resolve()
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    name: str = module.POD_MANIFEST_CONFIG_MAP
    return name


_CONTROL_PLANE = Path("deploy/k8s/control-plane.yaml")


def _platform() -> ModuleType:
    """`deploy/platform.py`, loaded by path.

    Loaded for `unrouted_service` so that "this Service routes to nothing" has ONE
    definition: the applier refuses a deploy on it, and the case below grades the
    cluster with it. A second reading written here would be free to call a Service
    healthy after the applier had stopped doing so, and that disagreement would surface
    as a deploy that refuses while every test passes.
    """
    spec = importlib.util.spec_from_file_location(
        "map_platform_for_service_routing", Path("deploy/platform.py").resolve()
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _control_plane_documents() -> list[dict[str, Any]]:
    documents = [d for d in yaml.safe_load_all(_CONTROL_PLANE.read_text()) if d]
    assert documents, f"{_CONTROL_PLANE} parsed into no documents"
    return documents


def _control_plane_deployment_spec() -> dict[str, Any]:
    """The Deployment's spec as this repository declares it, found by kind.

    By kind and not by index: the index is safe in the manifest's own test, which pins
    the document order, and this file has no business depending on that.
    """
    found = [d for d in _control_plane_documents() if d["kind"] == "Deployment"]
    assert len(found) == 1, f"{_CONTROL_PLANE} holds {len(found)} Deployments"
    spec: dict[str, Any] = found[0]["spec"]
    return spec


def test_the_control_plane_service_routes_to_a_ready_pod() -> None:
    """A Service can be committed, applied, and route to nothing.

    This is the one class of defect nothing local can grade. A selector matching no pod
    and a `targetPort` naming no container port are both well-formed YAML the API server
    accepts; a manifest test can only compare the manifest against itself. What
    disagrees is the cluster.

    It is the same defect as the two in this file's header, one object over.
    `session-shim-service.yaml` was committed and applied by nothing, and a Session pod
    at `2/2 Running` with its shim answering the kubelet failed every Turn as
    unreachable, because the name it published resolved to no endpoint. Every probe in
    that chain reported the pod healthy, because it was.

    The reading is delegated to `deploy/platform.py`'s `unrouted_service` rather than
    written again here, so the check that refuses a deploy and the check that grades the
    cluster cannot come to different conclusions from the same listing.

    Service names come out of the manifest, so a second Service added to that file is
    covered the day it is committed.
    """
    services = [
        str(document["metadata"]["name"])
        for document in _control_plane_documents()
        if document["kind"] == "Service"
    ]
    assert services, f"{_CONTROL_PLANE} declares no Service, so this grades nothing"
    unrouted_service = _platform().unrouted_service
    for name in services:
        _the_one("service", name)
        listing = _kubectl(
            "get",
            "endpointslice",
            "-n",
            _NAMESPACE,
            "-l",
            f"kubernetes.io/service-name={name}",
        )
        refusal = unrouted_service(name, listing)
        assert refusal is None, refusal


_ZONE_LABEL = "topology.kubernetes.io/zone"


def test_the_replicas_asked_for_are_running_and_spread_over_node_and_zone() -> None:
    """The replica count and both spreads, none of which anything local can settle.

    Three claims the manifest makes and cannot keep. The count: a Deployment running at
    fewer replicas than the file declares is a platform with the single point of failure
    the file says it does not have, and `kubectl get deploy` reporting `1/1` for a file
    that says 2 is what a stale apply looks like. The two spreads: the pod template asks
    the scheduler to keep the replicas on different nodes AND in different zones, with
    `whenUnsatisfiable: ScheduleAnyway` -- a PREFERENCE. Whether it was honoured is a
    decision the scheduler already made and no assertion over the manifest can reach.

    Both domains are graded, because on this cluster they are different partitions: the
    nodegroup runs several nodes per zone, so two pods on two nodes can be two pods in
    one zone. Grading only the node half would report a platform as spread while a
    single zone still held all of it.

    `min(replicas, domains available)` rather than the replica count outright: on a
    one-node or one-zone cluster co-location is the correct outcome, and asserting
    otherwise would fail on a fact about the cluster rather than about the platform.

    A failure here is not automatically a manifest defect. `ScheduleAnyway` yields to
    pressure, so co-location is the CORRECT scheduler decision when nothing else has
    room -- read node pressure before reading this as a bug.

    Pods being deleted are excluded. A rollout at `maxSurge: 0` never runs more pods
    than `replicas`, but a terminating pod still reports `Running` while carrying a
    `deletionTimestamp`, and counting one would put a node in this set that is on
    its way out of it.
    """
    declared = _control_plane_deployment_spec()
    desired = int(declared["replicas"])

    live = _the_one("deployment", "control-plane")
    running_spec = live["spec"]
    status = live["status"]
    assert isinstance(running_spec, dict) and isinstance(status, dict)
    assert int(running_spec["replicas"]) == desired, (
        f"the cluster's Deployment asks for {running_spec['replicas']} replicas and "
        f"{_CONTROL_PLANE} declares {desired}. It was changed and never applied."
    )
    assert status.get("availableReplicas") == desired, (
        f"{status.get('availableReplicas')} of {desired} replicas are available, "
        "so the platform has fewer serving pods than the file it came from declares"
    )

    selector = ",".join(
        f"{key}={value}" for key, value in declared["selector"]["matchLabels"].items()
    )
    pods = _kubectl("get", "pods", "-n", _NAMESPACE, "-l", selector)
    items = pods.get("items")
    assert isinstance(items, list) and items, f"no pod matches {selector}"
    alive = [
        pod
        for pod in items
        if pod["status"]["phase"] == "Running"
        and "deletionTimestamp" not in pod["metadata"]
    ]
    nodes = {pod["spec"]["nodeName"] for pod in alive}

    all_nodes = _kubectl("get", "nodes")
    listed = all_nodes.get("items")
    assert isinstance(listed, list) and listed, "the cluster reports no nodes"
    schedulable = [node for node in listed if not node["spec"].get("unschedulable")]
    zone_of = {
        node["metadata"]["name"]: node["metadata"]["labels"].get(_ZONE_LABEL)
        for node in schedulable
    }
    assert all(zone_of.values()), (
        f"a schedulable node carries no {_ZONE_LABEL} label: {zone_of}. The zone "
        "constraint keys on that label, and a node without it sits in a nameless "
        "domain where nothing can be skewed -- so the constraint is inert, not violated"
    )

    assert len(nodes) == min(desired, len(schedulable)), (
        f"{len(alive)} control-plane pods sit on {len(nodes)} of {len(schedulable)} "
        f"schedulable nodes: {sorted(nodes)}. At {desired} replicas and "
        f"{len(schedulable)} nodes they should occupy "
        f"{min(desired, len(schedulable))} -- fewer means a single node drain takes "
        "more of the API down than the topologySpreadConstraint asks for."
    )

    cordoned = sorted(node for node in nodes if node not in zone_of)
    assert not cordoned, (
        f"control-plane pods run on nodes that are not schedulable: {cordoned}. "
        "They serve now, and the scheduler will not put a replacement there, so the "
        "spread they currently show is not the spread this Deployment would get again."
    )
    zones = {str(zone_of[node]) for node in nodes}
    available = {str(zone) for zone in zone_of.values() if zone}
    assert len(zones) == min(desired, len(available)), (
        f"the control-plane pods occupy {sorted(zones)} of {sorted(available)}. At "
        f"{desired} replicas across {len(available)} zones they should occupy "
        f"{min(desired, len(available))}. Fewer means losing one availability zone "
        "takes more of the API down than the zone spread constraint asks for -- and it "
        "is the zone count, not the node count, that fixes the replica number in "
        "deploy/k8s/control-plane.yaml."
    )


def test_the_mounted_session_pod_manifest_is_the_one_this_repo_declares() -> None:
    """The control plane places pods from a file it mounts, so a stale copy is a stale
    Session pod.

    `deploy/platform.py` generates this ConfigMap from `session-pod.yaml` at apply time
    rather than committing a second document, so there is one description of a Session
    pod in the repository. That does not make the cluster's copy current: a ConfigMap
    generated once and left behind while the file moves is exactly the failure the
    seccomp-profile check above already exists for, one object over. Bytes are compared
    and not mere presence, for that reason.

    The name comes out of the applier rather than being written down again here, and
    the source file's basename is the key, because `--from-file` is what decides that.
    Both are contracts between the script that creates this object and the process that
    mounts it, and a contract restated is a contract that can be restated wrongly.
    """
    config_map = _pod_manifest_config_map()
    holder = _the_one("configmap", config_map)
    data = holder.get("data")
    assert isinstance(data, dict), f"ConfigMap {config_map} carries no data: {holder}"
    mounted = data.get(_SESSION_POD.name)
    assert mounted is not None, (
        f"ConfigMap {config_map} has no key {_SESSION_POD.name}. "
        "`--from-file` keys the entry by the source file's basename and "
        "MAP_POD_MANIFEST names that path, so the control plane reads nothing. "
        f"Keys present: {sorted(data)}"
    )
    assert yaml.safe_load(mounted) == yaml.safe_load(_SESSION_POD.read_text()), (
        f"the Session-pod manifest in the cluster differs from {_SESSION_POD}. Every "
        "pod the control plane places comes from those bytes. Regenerate it:\n"
        f"  kubectl create configmap {config_map} "
        f"--from-file={_SESSION_POD} --dry-run=client -o yaml | "
        "kubectl apply -n map-dev -f -"
    )
