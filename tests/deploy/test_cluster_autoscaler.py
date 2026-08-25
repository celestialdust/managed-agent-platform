"""The autoscaler's manifest, its role, and the one lifecycle line that keeps a
drift check from undoing its work.

Tier 1 (local; no cluster, no AWS, no network). Every assertion reads a file off
disk. What is graded is not style. Two facts make this component different from
everything else under `deploy/k8s/`: it is the only identity in the cluster that
can terminate an EC2 instance, and it is the only process that writes a number
Terraform also declares. Each case below is one of the ways those two turn into an
outage -- a credential that can drain the wrong group, a comparison that scales the
cluster back down, a flag copied from an example that evicts live Sessions.

NOT PROVEN by anything here, and neither gap is closable locally. That the
autoscaler ever scales: that needs a cluster, an unschedulable pod and an EC2
launch, and it is the checkpoint rather than a test. And that these documents are
ones the API server accepts: `kubectl apply --dry-run=server` established that
once, by hand, and `plan/MAP-65.md` quotes the output -- there is no schema
validator in this repository's offline suite, so a field name misspelled in a way
that still parses as YAML reaches `kubectl apply` uncaught.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "deploy" / "k8s" / "cluster-autoscaler.yaml"
_BOOTSTRAP = _ROOT / "deploy" / "k8s" / "cluster-bootstrap.yaml"
_SESSION_POD = _ROOT / "deploy" / "k8s" / "session-pod.yaml"
_POLICY = _ROOT / "deploy" / "iam" / "map-cluster-autoscaler.json"
_EKS = _ROOT / "deploy" / "terraform" / "eks.tf"
_IRSA = _ROOT / "deploy" / "terraform" / "irsa.tf"

# Every action this role may take, as a set rather than a list of things it may
# not. A ban list cannot cover what nobody has thought of, and the failure to
# guard against is a later edit adding one convenient action: ec2:RunInstances or
# eks:UpdateNodegroupConfig would each "work" and each widen this credential from
# changing a group's size to something else entirely. Every one below was
# simulated against the real account before it was written down.
_PERMITTED_ACTIONS = frozenset(
    {
        "autoscaling:DescribeAutoScalingGroups",
        "autoscaling:DescribeAutoScalingInstances",
        "autoscaling:DescribeLaunchConfigurations",
        "autoscaling:DescribeScalingActivities",
        "autoscaling:DescribeTags",
        "autoscaling:SetDesiredCapacity",
        "autoscaling:TerminateInstanceInAutoScalingGroup",
        "ec2:DescribeImages",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeLaunchTemplateVersions",
        "ec2:GetInstanceTypesFromInstanceRequirements",
        "eks:DescribeNodegroup",
    }
)

# The two that change something. Everything else in the set above is a read.
_CAPACITY_WRITES = frozenset(
    {
        "autoscaling:SetDesiredCapacity",
        "autoscaling:TerminateInstanceInAutoScalingGroup",
    }
)

_OWNING_TAG = "aws:ResourceTag/k8s.io/cluster-autoscaler/map-dev"


def _documents(path: Path) -> list[dict[str, Any]]:
    loaded = [d for d in yaml.safe_load_all(path.read_text()) if d]
    assert loaded, f"{path} parsed into no documents at all"
    return loaded


def _of_kind(path: Path, kind: str) -> list[dict[str, Any]]:
    return [d for d in _documents(path) if d["kind"] == kind]


def _deployment() -> dict[str, Any]:
    found = _of_kind(_MANIFEST, "Deployment")
    assert len(found) == 1, "one Deployment, or these assertions grade one of N"
    return found[0]


def _container() -> dict[str, Any]:
    # Annotated rather than inferred: the pod spec arrives from yaml.safe_load_all
    # as Any, so without this the return is Any and `mypy --strict` rejects it.
    pod: dict[str, Any] = _deployment()["spec"]["template"]["spec"]
    containers: list[dict[str, Any]] = pod["containers"]
    assert len(containers) == 1, "one container, for the same reason"
    return containers[0]


def _args() -> list[str]:
    return list(_container()["args"])


def _flag(name: str) -> str | None:
    """The **effective** value of `--name=value` in the container's args, or None.

    The last occurrence, not the first. Go's flag package takes the last one it
    parses, so a guard reading the first grades a value the process does not use --
    appending `--max-nodes-total=99` after the declared one changed the running
    controller's ceiling and left every assertion here green. Reading the last makes
    the guard agree with the parser; `test_no_flag_is_declared_twice` makes the
    disagreement impossible to reintroduce, because two spellings of one setting is a
    defect whichever one wins.
    """
    values = [arg.split("=", 1)[1] for arg in _args() if arg.startswith(f"--{name}=")]
    return values[-1] if values else None


def _flag_names() -> list[str]:
    """Every `--name` in the args, in order, including repeats."""
    return [
        arg.split("=", 1)[0].removeprefix("--")
        for arg in _args()
        if arg.startswith("--")
    ]


# strconv.ParseBool, which is what Go's flag package uses for a bool flag. Every
# spelling here is accepted silently; anything else makes the process exit at startup.
_GO_FALSE = frozenset({"0", "f", "F", "false", "FALSE", "False"})
_GO_TRUE = frozenset({"1", "t", "T", "true", "TRUE", "True"})


def _nodegroup_block() -> str:
    """The `aws_eks_node_group` resource's text, brace-matched from its header.

    Matched rather than sliced at a line number so the block survives edits above
    and below it, and so `lifecycle` found here provably belongs to the nodegroup
    rather than to the cluster resource in the same file.
    """
    text = _EKS.read_text()
    start = text.index('resource "aws_eks_node_group"')
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise AssertionError("the nodegroup block never closes")


def _size(name: str) -> int:
    found = re.search(rf"^\s*{name}\s*=\s*(\d+)\s*$", _nodegroup_block(), re.M)
    assert found, f"{name} is not declared in the nodegroup block"
    return int(found.group(1))


def _ignored_attributes() -> list[str]:
    """The entries of the nodegroup's `ignore_changes`, bracket-matched.

    Bracket-matched rather than captured with a `[^\\]]*` class, because the
    attribute names contain brackets themselves -- `scaling_config[0].desired_size`
    -- and a class that stops at the first `]` reads that entry as
    `scaling_config[0` and then passes a comparison meant to fail. Found the hard
    way: the first version of this file did exactly that.

    A regex, not `.index`, so `ignore_changes = all` fails on the assertion below
    with a sentence rather than raising ValueError out of a helper.
    """
    block = _nodegroup_block()
    opened = re.search(r"ignore_changes\s*=\s*\[", block)
    assert opened, "ignore_changes is absent, or is not a list literal"
    start = opened.end() - 1
    depth = 0
    for index in range(start, len(block)):
        if block[index] == "[":
            depth += 1
        elif block[index] == "]":
            depth -= 1
            if depth == 0:
                inner = block[start + 1 : index]
                return [entry.strip() for entry in inner.split(",") if entry.strip()]
    raise AssertionError("the ignore_changes list never closes")


def _autoscaler_account() -> dict[str, Any]:
    accounts = _of_kind(_BOOTSTRAP, "ServiceAccount")
    named = [a for a in accounts if a["metadata"]["name"] == "cluster-autoscaler"]
    assert len(named) == 1, "bootstrap does not create exactly one of these"
    return named[0]


def test_the_group_can_grow_at_all() -> None:
    """A ceiling equal to the floor is a constant, and an autoscaler pointed at a
    constant scales nothing while reporting that it is healthy. The numbers are
    eks.tf's argument and not this file's; what is asserted is the inequality this
    controller needs, plus a floor above zero -- see the scale-from-zero note in
    plan/MAP-65.md, where a group allowed to empty loses the node label five
    slices select on and stops scaling with no error that mentions it.
    """
    assert _size("max_size") > _size("min_size")
    assert _size("min_size") >= 2


def test_the_current_size_is_declared_autoscaler_owned() -> None:
    """The line that keeps a comparison from becoming an outage.

    The autoscaler moves `desired_size` by calling SetDesiredCapacity; the file
    declares it. Without `ignore_changes` the next apply -- including a routine
    one for an unrelated resource -- reconciles the account back to the file and
    drains whatever had been added, with the Sessions on it. Measured through
    `tools/terraform_drift.py` against the real account, with a third node the
    autoscaler had just added still running: with this line the gate answers
    `No changes` and exits 0; with it removed and nothing else changed the same
    account renders `~ desired_size = 3 -> 2` and the gate exits 2.
    """
    block = _nodegroup_block()
    assert "prevent_destroy = true" in block
    assert "scaling_config[0].desired_size" in block


def test_nothing_else_on_the_nodegroup_is_ignored() -> None:
    """`ignore_changes` is a hole in the comparison, so it is one attribute wide.

    `all`, or a second entry, would stop the gate reporting an instance type, a
    node label or a subnet somebody changed by hand -- the drift the whole
    Terraform directory exists to catch. min_size and max_size stay compared.
    """
    assert _ignored_attributes() == ["scaling_config[0].desired_size"]


def test_the_controller_will_not_grow_the_group_past_its_declared_ceiling() -> None:
    """Two ceilings, and the controller's must be the smaller one.

    The group's `max_size` is what anything may grow it to, a person running
    SetDesiredCapacity included. `--max-nodes-total` is what this controller does
    unattended, which matters because nothing in this cluster refuses a Session:
    there is no LimitRange and no ResourceQuota, and Session pods request nothing,
    so a loop that creates Sessions meets this number and nothing else. A flag
    above the group's ceiling is a bound that cannot be reached while reading like
    one that can.
    """
    total = _flag("max-nodes-total")
    assert total is not None, "--max-nodes-total is not set"
    ceiling, floor = _size("max_size"), _size("min_size")
    assert int(total) <= ceiling, (
        f"--max-nodes-total={total} is above the group's max_size {ceiling}, so it "
        "reads like a bound and cannot be reached"
    )
    assert int(total) >= floor, (
        f"--max-nodes-total={total} is below the group's min_size {floor}, so the "
        "controller can never add a node. It goes on reporting Ready 1/1 with its "
        "liveness probe passing while Session pods stay Pending for ever -- the "
        "pre-slice behaviour with a healthy-looking controller in front of it. This "
        "assertion was missing until 2026-08-23; `--max-nodes-total=1` passed."
    )


def test_a_node_running_a_session_is_never_drained_to_save_money() -> None:
    """Two flags that must stay true, read as values rather than matched as text.

    The upstream AWS example passes `--skip-nodes-with-local-storage=false`. Here
    that evicts live Sessions: session-pod.yaml declares emptyDir volumes, and
    emptyDir is exactly the local storage the default (true) refuses to drain.

    This asserted the **absence of two literal strings** until 2026-08-23, and Go
    parses a bool flag with `strconv.ParseBool`, which reads false from six
    spellings. Measured, one edit each, full deploy suite after every one:

        --skip-nodes-with-local-storage=false   -> 1 failed   (the only one caught)
        --skip-nodes-with-local-storage=0       -> 211 passed
        --skip-nodes-with-local-storage=f       -> 211 passed
        --skip-nodes-with-local-storage=F       -> 211 passed
        --skip-nodes-with-local-storage=FALSE   -> 211 passed
        --skip-nodes-with-local-storage=False   -> 211 passed

    Five ways to turn off the protection with the gate still green. This is the
    recurring defect of this project -- `docs/lessons.md`, "A guard written against
    one spelling is a guard against one spelling" -- and the cure it names is used
    here: read the value the program will read, and refuse everything that is not
    provably safe. An unparseable value fails too, because the process would exit at
    startup and a controller that never starts drains nothing but scales nothing
    either.
    """
    assert "emptyDir" in _SESSION_POD.read_text()
    for flag in ("skip-nodes-with-local-storage", "skip-nodes-with-system-pods"):
        value = _flag(flag)
        if value is None:
            continue  # absent means the default, which is true, which is the safe one
        assert value not in _GO_FALSE, (
            f"--{flag}={value} disables the protection: Go reads {sorted(_GO_FALSE)} "
            "as false. A node running live Sessions becomes a scale-down candidate."
        )
        assert value in _GO_TRUE, (
            f"--{flag}={value} is not a boolean Go can parse, so the controller exits "
            "at startup and nothing scales at all."
        )


def test_no_flag_is_declared_twice() -> None:
    """One setting, one place, because the parser takes the last and a reader the first.

    A repeated flag is not merely redundant: it is two answers to one question, and
    which one wins depends on argument order rather than on anything visible in the
    manifest. Whoever edits the first occurrence changes nothing and is told nothing.
    """
    names = _flag_names()
    repeated = sorted({name for name in names if names.count(name) > 1})
    assert not repeated, (
        f"{repeated} appear more than once in the container args. Go's flag package "
        "takes the last occurrence, so editing the first changes nothing silently."
    )


def test_the_controller_names_the_namespace_it_actually_runs_in() -> None:
    """`--namespace` defaults to kube-system and is not cosmetic.

    It is where the status ConfigMap and the leader-election Lease are written,
    and leader election is on by default in this release. A pod in map-dev with
    the default flag asks for a Lease in a namespace its Role does not cover and
    dies on that, with nothing in the message about scaling. Cross-read against
    bootstrap's Namespace rather than compared to a literal, so the three places
    that must agree cannot drift pairwise.
    """
    namespace = _deployment()["metadata"]["namespace"]
    assert _flag("namespace") == namespace
    assert _autoscaler_account()["metadata"]["namespace"] == namespace
    declared = [d["metadata"]["name"] for d in _of_kind(_BOOTSTRAP, "Namespace")]
    assert declared == [namespace], declared


def test_discovery_names_the_tags_the_group_carries_and_this_clusters_name() -> None:
    """Discovery by tag, with the cluster name read out of Terraform.

    A hardcoded group name would carry the nodegroup's uuid suffix, which a
    nodegroup replacement changes -- after which discovery finds nothing and the
    controller reports no group to scale. The cluster name is cross-read rather
    than repeated so a rename cannot leave this pointing at a cluster that is
    gone.
    """
    cluster = re.search(
        r'resource "aws_eks_cluster".*?^\s*name\s*=\s*"([^"]+)"',
        _EKS.read_text(),
        re.S | re.M,
    )
    assert cluster, "the cluster resource declares no name"
    assert _flag("node-group-auto-discovery") == (
        "asg:tag=k8s.io/cluster-autoscaler/enabled,"
        f"k8s.io/cluster-autoscaler/{cluster.group(1)}"
    )


def test_the_image_is_a_digest_and_its_minor_is_the_clusters() -> None:
    """A digest, and a release comment the cluster's version is compared to.

    A tag can be moved under a running Deployment; a digest cannot. A digest
    carries no version, so the release is named in a comment beside it -- and
    compared here, because the autoscaler states compatibility per Kubernetes
    minor and a cluster upgrade that leaves it behind is drift worth failing on.
    """
    image = _container()["image"]
    assert re.fullmatch(
        r"registry\.k8s\.io/autoscaling/cluster-autoscaler@sha256:[0-9a-f]{64}",
        image,
    ), image
    release = re.search(r"#\s*release v(\d+\.\d+)\.\d+", _MANIFEST.read_text())
    assert release, "no `# release vX.Y.Z` comment says what the digest is"
    version = re.search(
        r'resource "aws_eks_cluster".*?^\s*version\s*=\s*"([^"]+)"',
        _EKS.read_text(),
        re.S | re.M,
    )
    assert version, "the cluster resource declares no version"
    assert release.group(1) == version.group(1)


def test_the_pod_mounts_nothing_from_the_host_and_serves_nothing() -> None:
    """No volume, and no Service.

    The upstream example mounts the node's CA bundle over the container's trust
    store; this image already carries /etc/ssl/certs/ca-certificates.crt, checked
    inside it, so the mount buys nothing and puts a host path in the one pod that
    can terminate an instance. And a Service would give that pod an in-cluster
    name, which is a name something can call -- its metrics are reachable with
    `kubectl port-forward`.
    """
    spec = _deployment()["spec"]["template"]["spec"]
    assert spec.get("volumes", []) == []
    for container in spec["containers"]:
        assert container.get("volumeMounts", []) == []
    assert [d for d in _documents(_MANIFEST) if d["kind"] == "Service"] == []


def test_the_controller_cannot_evict_itself_mid_scale_down() -> None:
    """A Deployment's pod is controller-backed, so the autoscaler counts its own
    node as drainable. The annotation takes that node off the list; without it a
    scale-down can kill the process performing it, part-way through."""
    annotations = _deployment()["spec"]["template"]["metadata"]["annotations"]
    evict = "cluster-autoscaler.kubernetes.io/safe-to-evict"
    assert annotations[evict] == "false"


def test_the_pod_that_relieves_pressure_is_not_evicted_by_it() -> None:
    """Every Session pod is BestEffort, so a controller with no request of its own
    shares an eviction class with the load it exists to answer. A request and a
    priority class are what separate them; the numbers are upstream's and nothing
    here has measured this process."""
    requests = _container()["resources"]["requests"]
    assert requests["cpu"] and requests["memory"]
    spec = _deployment()["spec"]["template"]["spec"]
    assert spec["priorityClassName"] == "system-cluster-critical"


def test_the_role_can_change_capacity_only_in_this_clusters_groups() -> None:
    """The two write actions are the whole of this role's danger.

    Neither is granted unconditioned. The condition is a tag rather than the
    group's arn because the arn carries the nodegroup's uuid suffix, so a
    replacement would silently stop it matching and the failure would be an
    AccessDenied in a controller log rather than anything in Kubernetes. EKS
    re-applies the tag to whatever group it makes. Measured with the policy
    simulator: with the tag both are allowed, and with the tag absent or set to
    anything else both are implicitDeny.
    """
    document = json.loads(_POLICY.read_text())
    granting = [
        statement
        for statement in document["Statement"]
        if set(statement["Action"]) & _CAPACITY_WRITES
    ]
    assert granting, "nothing grants the two actions that change capacity"
    for statement in granting:
        actions = set(statement["Action"])
        assert actions <= _CAPACITY_WRITES, sorted(actions - _CAPACITY_WRITES)
        assert statement["Effect"] == "Allow"
        equals = statement.get("Condition", {}).get("StringEquals", {})
        assert equals == {_OWNING_TAG: "owned"}, equals


def test_the_role_grants_no_action_outside_a_written_down_set() -> None:
    """An allowlist, because a ban list cannot cover what nobody thought of.

    Also asserts the reads are not scoped to an arn: simulated against the
    group's arn every one of them comes back implicitDeny, which is the
    simulator's way of saying the action takes no resource. The one read that
    does take a resource -- eks:DescribeNodegroup -- is scoped to this cluster's
    nodegroups and is allowed there and denied elsewhere.
    """
    document = json.loads(_POLICY.read_text())
    granted = {a for s in document["Statement"] for a in s["Action"]}
    assert granted == _PERMITTED_ACTIONS, sorted(granted ^ _PERMITTED_ACTIONS)
    for statement in document["Statement"]:
        if "eks:DescribeNodegroup" in statement["Action"]:
            assert statement["Resource"].startswith(
                "arn:aws:eks:us-east-1:000000000000:nodegroup/map-dev/"
            )


def test_the_trust_condition_names_one_subject_and_pins_the_audience() -> None:
    """irsa.tf's header says why this is graded on its own: the condition decides
    WHICH service account may become the role. A second subject here would hand
    the one credential that can terminate an instance to whatever runs under it,
    with nothing in the repository comparing. The subject is cross-read from the
    ServiceAccount bootstrap creates, so the namespace cannot drift between the
    two files.
    """
    text = _IRSA.read_text()
    start = text.index('resource "aws_iam_role" "cluster_autoscaler"')
    block = text[start:]
    end = block.find("\nimport {")
    block = block if end == -1 else block[:end]
    namespace = _autoscaler_account()["metadata"]["namespace"]
    assert re.findall(r':sub"\s*=\s*"([^"]+)"', block) == [
        f"system:serviceaccount:{namespace}:cluster-autoscaler"
    ]
    assert re.findall(r':aud"\s*=\s*"([^"]+)"', block) == ["sts.amazonaws.com"]


# The Kubernetes half of this controller's credential. Nothing graded it until
# 2026-08-23, and the AWS half next door has carried an explicit allowlist since the
# slice was written -- with the docstring "An allowlist, because a ban list cannot
# cover what nobody thought of". The asymmetry was the whole defect: one credential,
# two halves, and the ungraded half is the one that can read Secrets. Five widenings
# were measured against the suite as it stood, each a one-line edit:
#
#   secrets: [get, list, watch] added to the ClusterRole  -> 1792 passed
#   a second binding subject (default/kube-system)        ->  211 passed
#   hostNetwork: true                                     ->  211 passed
#   privileged: true on the container                     ->  211 passed
#   serviceAccountName: tool-gateway                      ->  211 passed
#
# The first of those hands the autoscaler every Secret in the cluster,
# `map-control-plane` among them. The list below is what a cluster autoscaler needs in
# order to place a pod and drain a node; a genuine upstream addition is one line here
# and a reader who has to
# say why.
_PERMITTED_RESOURCES: Final = frozenset(
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
)


def _roles() -> list[dict[str, Any]]:
    return _of_kind(_MANIFEST, "ClusterRole") + _of_kind(_MANIFEST, "Role")


def _bindings() -> list[dict[str, Any]]:
    return _of_kind(_MANIFEST, "ClusterRoleBinding") + _of_kind(
        _MANIFEST, "RoleBinding"
    )


def test_the_controller_is_granted_no_kubernetes_resource_outside_the_allowlist() -> (
    None
):
    """Every (apiGroup, resource) it may touch is one somebody wrote down.

    An allowlist rather than a ban on `secrets`, for the same reason the AWS half of
    this credential uses one: a ban list cannot cover what nobody thought of. Reading
    Secrets is the consequence that makes this urgent, but `create` on `roles` or
    `escalate` on anything would be worse and no ban list drafted today would name them.
    """
    granted = {
        (group, resource)
        for role in _roles()
        for rule in role["rules"]
        for group in rule["apiGroups"]
        for resource in rule["resources"]
    }
    unexpected = sorted(granted - _PERMITTED_RESOURCES)
    assert not unexpected, (
        f"the autoscaler's RBAC grants {unexpected}, which nothing here permits. If "
        "this is a real upstream requirement, add it to _PERMITTED_RESOURCES with a "
        "reason. Note what this credential reaches: 'secrets' here means every Secret "
        "in the cluster, map-control-plane included."
    )


def test_no_rule_uses_a_wildcard() -> None:
    """A `*` grants what the allowlist above cannot see.

    Checked separately because a wildcard passes the allowlist trivially -- `("*", "*")`
    is one pair, and one pair is easy to add to a list of twenty-five without anyone
    noticing that it subsumes all of them.
    """
    for role in _roles():
        for rule in role["rules"]:
            for field in ("apiGroups", "resources", "verbs"):
                assert "*" not in rule.get(field, []), (
                    f"{role['metadata']['name']} uses a wildcard in {field}: {rule}"
                )


def test_each_binding_names_exactly_the_autoscaler_service_account() -> None:
    """One subject per binding, and it is this controller's own identity.

    A second subject hands evict-any-pod, update-any-node and patch-any-endpoints to
    whoever it names -- the Kubernetes analogue of a trust policy with two principals,
    which `irsa.tf` and its one-subject trust-condition test already defend on the AWS
    side. This is the same risk on the side nothing watched.
    """
    account = _deployment()["spec"]["template"]["spec"]["serviceAccountName"]
    namespace = _deployment()["metadata"]["namespace"]
    assert _bindings(), "the manifest declares no bindings at all"
    for binding in _bindings():
        subjects = binding["subjects"]
        assert subjects == [
            {"kind": "ServiceAccount", "name": account, "namespace": namespace}
        ], (
            f"{binding['metadata']['name']} names {subjects}; it must name exactly the "
            f"ServiceAccount {account} in {namespace}, which is the identity the "
            "Deployment actually runs as"
        )


def test_the_pod_runs_as_the_identity_its_irsa_role_trusts() -> None:
    """`serviceAccountName` is what selects the AWS role, so a wrong one is silent.

    Point the Deployment at another workload's ServiceAccount and it starts, stays
    Ready, assumes that workload's IRSA role instead of `map-cluster-autoscaler`, and
    fails only later on an AWS call -- with a message about the wrong role rather than
    about the wrong ServiceAccount. Cross-read against the bootstrap manifest so the
    two files cannot drift.
    """
    account = _deployment()["spec"]["template"]["spec"]["serviceAccountName"]
    assert account == _autoscaler_account()["metadata"]["name"]


def test_the_controller_takes_no_privilege_it_does_not_need() -> None:
    """It schedules pods and drains nodes; it needs no host and no root.

    Each of these passed the suite when flipped, so each is a real hole and not a
    hypothetical: `hostNetwork: true` puts it on the node's network namespace,
    `privileged: true` makes the container's other restrictions decorative.
    """
    pod = _deployment()["spec"]["template"]["spec"]
    assert not pod.get("hostNetwork"), "hostNetwork gives it the node's network"
    assert not pod.get("hostPID"), "hostPID gives it the node's process table"
    assert not pod.get("hostIPC")
    container = pod["containers"][0]
    security = container.get("securityContext", {})
    assert not security.get("privileged"), (
        "a privileged container makes runAsNonRoot, readOnlyRootFilesystem and the "
        "dropped capabilities on this same container decorative"
    )
    assert security.get("allowPrivilegeEscalation") is False
    assert security.get("readOnlyRootFilesystem") is True
    assert pod["securityContext"]["runAsNonRoot"] is True
