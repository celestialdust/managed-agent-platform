"""The default-deny policy set, and the switch without which it filters nothing.

TWO TIERS, and which one a case is in is the whole point of this file.

The cases up to `_GATE` read files. They grade the policy set against the manifests it
talks about -- that a selector matches a pod somebody declares, that a port named toward
a peer is a port that peer listens on, that every workload allowed to egress may still
resolve a name, that no address range opened for the database or the API server also
reaches a pod. All of that is checkable offline and none of it says anything about
whether a single packet is examined.

The cases after `_GATE` are the ones that can. A NetworkPolicy is accepted by the API
server whether or not anything enforces it, and on this cluster nothing does: the CNI
ships the agent that would and runs it switched off. So `test_the_cni_is_enforcing_
network_policy` FAILS TODAY, deliberately, and it is the only case here that can tell a
boundary from a description of one. A run that skipped it has measured nothing about
enforcement -- every case above it passes just as happily against a cluster where the
whole set is decoration.

WHY THE POD LABEL WALK IS LOCAL and not shared with
`test_the_manifest_set_is_deployable.py`, which walks for the same shape: that file's
walk answers "does this Service reach a pod", and it deliberately reads only `Service`
and `PodDisruptionBudget` selectors, so a NetworkPolicy's selectors are graded by
nothing there. Two small walks over one shape is the Rule of Three's second copy, and
the alternative today is a shared helper module for two callers.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest
import yaml

_ROOT: Final = Path(__file__).resolve().parents[2]
_K8S: Final = _ROOT / "deploy" / "k8s"
_POLICY_SET: Final = _K8S / "network-policies.yaml"
_NETWORK_TF: Final = _ROOT / "deploy" / "terraform" / "network.tf"
_VPC_CNI_TF: Final = _ROOT / "deploy" / "terraform" / "vpc_cni.tf"

_NAMESPACE: Final = "map-dev"
_FLOOR: Final = "default-deny"
"""The one policy here whose selector is empty, named because two cases turn on being
able to tell it from the others: an empty selector is the most permissive selector there
is to write by accident, and the deliberate one has to be identifiable so the accident
stays a failure."""

_GATEWAY_MANIFESTS: Final = ("tool-gateway.yaml", "model-gateway.yaml")
"""The two files whose pods a Session may reach. Files rather than labels, so the
comparison goes through each manifest's own `map.component` value: a policy that named
`map.component: tool-gatewya` would match no pod and fail, where a test restating the
label beside the policy would agree with the typo."""


def _module(name: str, relative: str) -> ModuleType:
    """`deploy/platform.py` by path. `deploy/` is not a package and not on sys.path."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def _platform() -> ModuleType:
    return _module("map_platform_netpol", "deploy/platform.py")


def _documents(path: Path) -> list[dict[str, Any]]:
    return [document for document in yaml.safe_load_all(path.read_text()) if document]


def _policies() -> dict[str, dict[str, Any]]:
    """Every policy in the set, name to `spec`, in file order."""
    return {
        str(document["metadata"]["name"]): dict(document["spec"])
        for document in _documents(_POLICY_SET)
    }


_POD_TEMPLATE_KINDS: Final = frozenset(
    {"Deployment", "DaemonSet", "Job", "StatefulSet"}
)


class _Pod:
    """One pod shape declared under `deploy/k8s/`: where it came from, its labels, its
    ports.

    Labels are the TEMPLATE's and not the enclosing object's, because those are two
    places free to disagree -- a Deployment labelled `map.component: x` whose template
    is not creates pods carrying no such label, and a selector written against the
    object's label then matches nothing while every document reads correctly.
    """

    def __init__(
        self, where: str, labels: dict[str, str], ports: frozenset[int]
    ) -> None:
        self.where = where
        self.labels = labels
        self.ports = ports


def _pods() -> list[_Pod]:
    """Every pod shape under `deploy/k8s/`, with its labels and its container ports.

    A `CronJob` raises rather than being skipped: its template is one level deeper than
    every other kind, so a silent skip would be a pod that no case in this file grades
    while the sweep still reports success.
    """
    found: list[_Pod] = []
    for path in sorted(_K8S.glob("*.yaml")):
        for document in _documents(path):
            kind = str(document["kind"])
            if kind == "CronJob":
                raise AssertionError(
                    f"{path.name} declares a CronJob, whose pod template is at "
                    "spec.jobTemplate.spec.template. This walk would read the wrong "
                    "place; teach it that path before adding one."
                )
            if kind == "Pod":
                metadata = dict(document["metadata"])
                specification = dict(document["spec"])
            elif kind in _POD_TEMPLATE_KINDS:
                template = document["spec"]["template"]
                metadata = dict(template.get("metadata") or {})
                specification = dict(template["spec"])
            else:
                continue
            ports = {
                int(entry["containerPort"])
                for group in ("initContainers", "containers")
                for container in specification.get(group) or []
                for entry in container.get("ports") or []
            }
            found.append(
                _Pod(
                    where=f"{path.name}:{kind}/{document['metadata']['name']}",
                    labels={
                        str(key): str(value)
                        for key, value in (metadata.get("labels") or {}).items()
                    },
                    ports=frozenset(ports),
                )
            )
    return found


def _matching(selector: dict[str, str]) -> list[_Pod]:
    """Every pod shape in the tree whose labels satisfy this equality selector.

    A subset comparison, which is what Kubernetes does: a selector may name fewer labels
    than the pod carries.
    """
    return [
        pod
        for pod in _pods()
        if all(pod.labels.get(key) == value for key, value in selector.items())
    ]


def _rules(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every rule in one policy, tagged `ingress` or `egress`."""
    return [
        (direction, dict(rule))
        for direction in ("ingress", "egress")
        for rule in spec.get(direction) or []
    ]


def _peers(direction: str, rule: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(peer) for peer in rule.get("to" if direction == "egress" else "from") or []
    ]


def _ports(rule: dict[str, Any]) -> list[int]:
    return [int(entry["port"]) for entry in rule.get("ports") or []]


def _blocks(rule: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    return [
        dict(peer["ipBlock"]) for peer in _peers(direction, rule) if "ipBlock" in peer
    ]


def _subnet_cidrs() -> list[str]:
    """The pod subnets, read out of `network.tf` rather than written down here.

    These are the two subnets the nodegroup runs in, so every pod in this cluster takes
    its address from one of them. What they are used for below is checking that a rule
    written to reach the public internet carves the cluster's own addresses out of
    itself, and the VPC's own CIDR is not declared anywhere in this tree -- so the
    derivation goes through what is: a carve-out that does not contain these two
    contains no pod.
    """
    found = re.findall(r'cidr_block\s*=\s*"([0-9./]+)"', _NETWORK_TF.read_text())
    return [cidr for cidr in found if cidr != "0.0.0.0/0"]


def _v4(cidr: str) -> ipaddress.IPv4Network:
    """One CIDR from the policy set as a network, refusing anything that is not IPv4.

    The narrow type is not a convenience: `subnet_of` compares two networks of the same
    family and refuses a mixed pair, so a v6 entry would raise here rather than being
    compared against a v4 carve-out and quietly answering False. This cluster is v4 --
    the CNI runs with `ENABLE_IPv6=false` -- and the VPC CNI applies a policy's v4 rules
    or its v6 rules by the cluster's family, never both, so a v6 entry in this set would
    be a rule nothing enforces.
    """
    return ipaddress.IPv4Network(cidr)


_IMDS: Final = ipaddress.IPv4Address("169.254.169.254")
"""The instance metadata service. Not a repository value and not a cluster value -- the
same address on every EC2 instance there has ever been, which is why it can be a literal
here. A pod that reaches it can ask for the node's instance-role credentials, which on
this cluster are the nodegroup's."""


# --- Tier A: the set as text, against the manifests it talks about. ------------------


def test_the_set_yields_policies_and_pods_to_grade() -> None:
    """The positive control. Every case below iterates a discovered collection, and a
    parse that quietly produced nothing would satisfy all of them by being empty."""
    policies = _policies()
    pods = _pods()
    assert len(policies) >= 7, sorted(policies)
    assert len(pods) >= 6, [pod.where for pod in pods]
    assert sum(len(_rules(spec)) for spec in policies.values()) >= 15


def test_every_document_is_a_network_policy_in_this_platforms_namespace() -> None:
    """A policy is namespaced, and one applied into the wrong namespace selects the
    wrong pods -- or none, which is the same object doing nothing. Written on each
    document rather than left to the command line, so `kubectl apply -f` from any
    context puts the set where it belongs."""
    for document in _documents(_POLICY_SET):
        where = f"{document['kind']}/{document['metadata'].get('name')}"
        assert document["kind"] == "NetworkPolicy", where
        assert document["apiVersion"] == "networking.k8s.io/v1", where
        assert document["metadata"].get("namespace") == _NAMESPACE, where


def test_only_the_floor_selects_every_pod() -> None:
    """`podSelector: {}` selects every pod in the namespace. That is exactly right for
    the blanket deny and it is the worst possible accident in a policy that allows
    something: a rule meant for one workload would hand its allowances to every pod
    here, including a Session's."""
    empty = sorted(
        name for name, spec in _policies().items() if not spec["podSelector"]
    )
    assert empty == [_FLOOR], (
        f"{empty} select every pod in {_NAMESPACE} through an empty podSelector. Only "
        f"{_FLOOR} may, because it allows nothing; any other policy with an empty "
        "selector grants its rules to every pod in the namespace."
    )


def test_the_floor_denies_both_directions_and_allows_nothing() -> None:
    """A policy naming a direction in `policyTypes` and carrying no rule for it denies
    that direction outright. Both are named, because a floor that covered only ingress
    would leave a new workload free to dial anything in the cluster."""
    floor = _policies()[_FLOOR]
    assert sorted(floor["policyTypes"]) == ["Egress", "Ingress"], floor
    assert not _rules(floor), (
        f"{_FLOOR} carries rules, so it is not a floor -- whatever it allows, it "
        "allows to every pod in this namespace"
    )


def test_every_policy_selects_a_pod_this_tree_declares() -> None:
    """A policy whose selector matches nothing is valid, listed, and inert.

    The same silently-inert shape as a Service selector matching no pod, and worse in
    one way: a Service that routes nowhere fails visibly at the first request, while a
    policy that selects nothing fails by protecting nothing, which nothing reports.
    """
    for name, spec in sorted(_policies().items()):
        if name == _FLOOR:
            continue
        selector = dict(spec["podSelector"]["matchLabels"])
        assert _matching(selector), (
            f"policy {name} selects {selector} and no pod declared under deploy/k8s/ "
            f"carries those labels, so it constrains nothing. Pods in the tree: "
            f"{ {pod.where: pod.labels for pod in _pods()} }"
        )


def test_every_peer_selector_matches_a_pod_this_tree_declares() -> None:
    """An allow rule naming labels no pod carries permits nothing.

    This is the failure mode of a typo in a rule rather than in a selector, and it is
    quieter: `map.component: tool-gatewya` in a Session pod's egress reads as a
    well-formed permission and takes every Session's tool calls out, with the refusal
    arriving as a connection timeout to a name that resolved.
    """
    for name, spec in sorted(_policies().items()):
        for direction, rule in _rules(spec):
            for peer in _peers(direction, rule):
                if "podSelector" not in peer:
                    continue
                selector = dict(peer["podSelector"]["matchLabels"])
                if "namespaceSelector" in peer:
                    # A peer in another namespace -- CoreDNS. Its pods are not declared
                    # in this tree, so there is nothing here to compare them against and
                    # the DNS case below is what grades this rule instead.
                    continue
                assert _matching(selector), (
                    f"policy {name}: a {direction} rule allows {selector}, which no "
                    "pod declared under deploy/k8s/ carries, so it permits nothing"
                )


def test_every_workload_allowed_to_egress_may_still_resolve_a_name() -> None:
    """DNS, on both protocols, for every policy that restricts egress at all.

    Every other egress rule in this file names a host: the database DSN, the AWS
    endpoints, both gateways' Service names, a Session pod's per-pod record. A policy
    that restricts egress and forgets port 53 breaks all of them at once and reports it
    as whatever was being resolved being unreachable -- which reads as the database
    being down, or the gateway being down, and not as this file.

    TCP as well as UDP, because a response over 512 bytes is retried over TCP and this
    namespace's own names already answer with several records.
    """
    for name, spec in sorted(_policies().items()):
        if name == _FLOOR or "Egress" not in spec["policyTypes"]:
            continue
        allowed = {
            (str(entry["protocol"]), int(entry["port"]))
            for _, rule in _rules(spec)
            for peer in _peers("egress", rule)
            if "namespaceSelector" in peer
            for entry in rule.get("ports") or []
        }
        assert {("UDP", 53), ("TCP", 53)} <= allowed, (
            f"policy {name} restricts egress and allows {sorted(allowed)} toward "
            "another namespace, so it does not allow both DNS protocols to CoreDNS. "
            "Every name this workload dials resolves there first."
        )


def test_every_port_named_toward_a_peer_is_one_that_pod_listens_on() -> None:
    """A port in a rule has to be the port the receiving container is bound to.

    WHICH POD OWNS THE PORT DEPENDS ON THE DIRECTION, and getting that backwards is the
    easy mistake. On an `egress` rule the port belongs to the PEER -- it is where this
    pod is dialling. On an `ingress` rule it belongs to the policy's OWN pods -- it is
    what they are listening on, and the peer is merely who may connect. So a gateway's
    ingress rule from a Session pod names 8080 because the gateway listens on 8080, not
    because a Session pod does.

    Derived from each manifest's `containerPort` rather than restated, so renumbering a
    container fails here instead of leaving a rule pointing at a port nothing serves.

    Peers in another namespace are skipped for the reason the case above gives.
    """
    for name, spec in sorted(_policies().items()):
        for direction, rule in _rules(spec):
            peers = _peers(direction, rule)
            if any("namespaceSelector" in peer for peer in peers):
                continue
            if direction == "ingress":
                listeners = _matching(dict(spec["podSelector"]["matchLabels"]))
            else:
                listeners = [
                    pod
                    for peer in peers
                    if "podSelector" in peer
                    for pod in _matching(dict(peer["podSelector"]["matchLabels"]))
                ]
            if not listeners:
                continue
            served = {port for pod in listeners for port in pod.ports}
            for port in _ports(rule):
                assert port in served, (
                    f"policy {name}: a {direction} rule names port {port}, and the "
                    f"pods it applies to ({[pod.where for pod in listeners]}) declare "
                    f"{sorted(served)}. A rule naming a port nothing is bound to "
                    "allows nothing, and one naming the wrong port allows the wrong "
                    "thing."
                )


def test_no_container_port_in_the_tree_falls_inside_an_address_range_allow() -> None:
    """The rules that name a CIDR are for the database, the API server and AWS, and each
    one also reaches any POD at that address range and port. That is only acceptable
    while nothing in this platform listens on those ports, and this is what keeps that a
    checked fact rather than a sentence in a comment.

    A workload added on 443 would silently become reachable from the control plane and
    the autoscaler; one on 5432 from every process that talks to the database. Either is
    a widening nobody chose, arriving with a manifest that says nothing about network
    policy.
    """
    opened = {
        port
        for spec in _policies().values()
        for direction, rule in _rules(spec)
        if _blocks(rule, direction)
        for port in _ports(rule)
    }
    assert opened, "no CIDR-shaped allow was found, so this case checked nothing"
    for pod in _pods():
        overlap = sorted(pod.ports & opened)
        assert not overlap, (
            f"{pod.where} listens on {overlap}, which an address-range rule in the "
            f"policy set opens ({sorted(opened)}). Those rules exist to reach the "
            "database, the Kubernetes API and AWS, and every pod at that port comes "
            "with them. Give this workload a port outside that set, or narrow the rule "
            "to the address it is actually for."
        )


def test_every_allow_that_leaves_the_vpc_carves_out_what_is_inside_it() -> None:
    """`cidr: 0.0.0.0/0` means every address there is, and the `except` list is the one
    thing stopping "may reach AWS" from also meaning "may reach the control plane, the
    database, and every other tenant's Session".

    Three carve-outs are required and each is derived rather than restated. The pod
    subnets come out of `network.tf`, and an `except` entry has to CONTAIN them -- so
    the VPC's own CIDR satisfies this without this file needing to know it. Link-local
    is required because that is where the node's own instance credentials answer. And
    every single address the same policy reaches through a narrower rule has to be
    excluded too: a blanket rule that also covered it would make the narrow rule
    decoration, and the narrow rule is where somebody wrote down what this workload is
    allowed to talk to.
    """
    seen = 0
    for name, spec in sorted(_policies().items()):
        singles = [
            network
            for direction, rule in _rules(spec)
            for block in _blocks(rule, direction)
            if (network := _v4(block["cidr"])).prefixlen == 32
        ]
        for direction, rule in _rules(spec):
            for block in _blocks(rule, direction):
                if _v4(block["cidr"]).prefixlen != 0:
                    continue
                seen += 1
                carved = [_v4(entry) for entry in block.get("except") or []]
                for cidr in _subnet_cidrs():
                    subnet = _v4(cidr)
                    assert any(subnet.subnet_of(entry) for entry in carved), (
                        f"policy {name}: an open {direction} rule excludes {carved} "
                        f"and nothing there contains {subnet}, which is a subnet every "
                        "pod in this cluster takes its address from. As written the "
                        "rule reaches them."
                    )
                assert any(_IMDS in entry for entry in carved), (
                    f"policy {name}: an open {direction} rule excludes {carved}, none "
                    f"of which contains {_IMDS} -- the instance metadata service, "
                    "where the node's own IAM credentials answer. These pods take "
                    "credentials from a projected web-identity token and have no "
                    "reason to reach it."
                )
                for single in singles:
                    assert any(single.subnet_of(entry) for entry in carved), (
                        f"policy {name} reaches {single} through a rule of its own and "
                        f"also through an open rule excluding only {carved}. The "
                        "narrow rule is then decoration, and the record of what this "
                        "workload may talk to is gone."
                    )
    assert seen, "no open address-range rule was found, so this case checked nothing"


def test_the_control_plane_admits_no_pod_in_this_cluster() -> None:
    """The tenant API authenticates nothing: it takes its tenant from a request header
    whose value is asserted by whoever sent the request, so any pod that can reach it
    can read and write every tenant's Sessions. Declaring `Ingress` and carrying no
    ingress rule is what denies every pod -- and an operator still reaches it, because
    `kubectl port-forward` arrives from the node rather than from a pod.

    ADR-022 rests on a Session's pod being unable to name this address. This is the half
    that stops it reaching one it guessed.
    """
    control_plane = _policies()["control-plane"]
    assert "Ingress" in control_plane["policyTypes"], control_plane
    assert not control_plane.get("ingress"), (
        "the control plane admits a peer: "
        f"{control_plane.get('ingress')}. Nothing in this cluster is supposed to dial "
        "an API that authenticates nobody."
    )


def test_a_session_pod_may_reach_exactly_the_two_gateways() -> None:
    """The strictest document in the set, asserted as an equality rather than as a list
    of things it does not do.

    Equality is what makes this hold against a rule somebody adds later: a case that
    asserted only that the control plane is absent would pass when the Kubernetes API,
    another Session, or the public internet is added. The set is compared through the
    manifests' own labels, so the two gateway files decide what it contains.
    """
    session = _policies()["session-pod"]
    reachable = {
        pod.where
        for _, rule in _rules(session)
        for peer in _peers("egress", rule)
        if "podSelector" in peer and "namespaceSelector" not in peer
        for pod in _matching(dict(peer["podSelector"]["matchLabels"]))
    }
    expected = {
        pod.where for pod in _pods() if pod.where.split(":")[0] in _GATEWAY_MANIFESTS
    }
    assert expected, "neither gateway manifest yielded a pod, so this compared nothing"
    assert reachable == expected, (
        f"a Session pod's egress reaches {sorted(reachable)}; the two gateways are "
        f"{sorted(expected)}. Anything else in that set is a pod running a tenant's "
        "model-authored code being allowed to dial something nobody decided it could."
    )
    assert not [
        block
        for direction, rule in _rules(session)
        for block in _blocks(rule, direction)
    ], (
        "a Session pod's policy names an address range. Every destination it has is a "
        "pod in this namespace, and a CIDR here is either the internet or an in-VPC "
        "address, both of which this pod is not supposed to have."
    )


# --- Tier A, continued: the applier's refusal, and the Terraform that lifts it. ------


@pytest.mark.parametrize("value", ["1", "t", "T", "TRUE", "true", "True"])
def test_every_spelling_of_on_that_the_flags_parser_accepts_reads_as_on(
    value: str,
) -> None:
    """The agent is a Go program and its boolean flag is read with `strconv.ParseBool`,
    which accepts six spellings of true. A check that matched one of them would call the
    other five off -- which is the defect `docs/lessons.md` records twice and
    `tests/test_guards_read_values_not_spellings.py` exists to stop recurring. The
    spellings are in the parameter list rather than inside an assertion for that guard's
    own reason: a literal flag-and-boolean inside an `assert` is the shape it refuses.
    """
    assert _platform().enforcing_network_policy((f"--enable-network-policy={value}",))


@pytest.mark.parametrize("value", ["0", "f", "F", "FALSE", "false", "False", "yes", ""])
def test_no_other_value_reads_as_on(value: str) -> None:
    """The six false spellings, and two the parser rejects outright. A value `ParseBool`
    refuses makes the agent exit at start-up, which is a nodeagent in CrashLoopBackOff
    and no enforcement either -- so anything that is not provably on is off."""
    assert not _platform().enforcing_network_policy(
        (f"--enable-network-policy={value}",)
    )


def test_the_flag_named_with_no_value_reads_as_on_and_absent_reads_as_off() -> None:
    """Two shapes that are not `flag=value`, and they go opposite ways for reasons that
    are not symmetric. A Go boolean flag named alone IS true, so reading it as off would
    invent a state the program does not have. A flag that is not there at all is off,
    because that is the fail-safe direction: the caller uses this to decide whether a
    policy set is decoration, and "we could not find the flag" must never be reported as
    "it is enforcing".

    The third list is built above the assertion rather than inside it, because it
    carries another of the nodeagent's boolean flags as noise and a flag-and-boolean
    literal inside an `assert` is the shape
    `tests/test_guards_read_values_not_spellings.py` refuses -- rightly, since that is
    how a guard comes to cover one spelling of six.
    """
    enforcing = _platform().enforcing_network_policy
    other_flags = ("--enable-ipv6=false", "--log-level=debug")
    assert enforcing(("--enable-network-policy",))
    assert not enforcing(())
    assert not enforcing(other_flags)


def test_the_last_occurrence_of_the_flag_is_the_one_that_counts() -> None:
    """Matching the parser, which takes a repeated flag's last value. A reading that
    took the first would call a DaemonSet enforcing on the strength of an argument the
    program itself overrode."""
    enforcing = _platform().enforcing_network_policy
    on, off = "--enable-network-policy=true", "--enable-network-policy=false"
    assert not enforcing((on, off))
    assert enforcing((off, on))


def test_a_policy_set_the_cni_ignores_is_a_refusal_and_an_enforced_one_is_not() -> None:
    """The applier's own check, and the case is written so its two arms can disagree.

    Three inputs and three different answers from one function: policies with the flag
    off is a refusal, the SAME policies with the flag on is not, and no policies at all
    is not -- because a namespace with no NetworkPolicy is unfiltered and says so, which
    is the state this cluster is in today. A case that only asserted the refusal would
    pass just as well against a function that refused everything.
    """
    inert = _platform().inert_network_policies
    off, on = ("--enable-network-policy=false",), ("--enable-network-policy=true",)
    policies = ("control-plane", "default-deny")

    refusal = inert(policies, off)
    assert refusal is not None
    assert "control-plane" in refusal and "default-deny" in refusal
    assert "vpc_cni.tf" in refusal, (
        "the refusal does not name where the switch is, so whoever meets it has to go "
        f"and find out: {refusal}"
    )
    assert inert(policies, on) is None
    assert inert((), off) is None


def test_the_terraform_declares_the_key_that_turns_enforcement_on() -> None:
    """The value, not the presence of the key. `enableNetworkPolicy` is typed by the
    add-on's configuration schema as a string with boolean format, so the JSON has to
    carry `"true"` and not a bare `true` -- and a declaration reading `"false"` would
    parse, plan and apply while leaving every policy in the tree inert."""
    text = _VPC_CNI_TF.read_text()
    assert 'addon_name   = "vpc-cni"' in text, (
        f"{_VPC_CNI_TF.name} declares no vpc-cni add-on, so nothing here turns "
        "enforcement on"
    )
    found = re.search(r'enableNetworkPolicy\s*=\s*"([^"]*)"', text)
    assert found is not None, (
        f"{_VPC_CNI_TF.name} names no enableNetworkPolicy value. That key is the whole "
        "reason this file exists; without it the add-on is adopted and nothing changes."
    )
    assert found.group(1) in {"1", "t", "T", "TRUE", "true", "True"}, (
        f"{_VPC_CNI_TF.name} sets enableNetworkPolicy to {found.group(1)!r}, which is "
        "not a value the agent reads as on"
    )


def test_the_addon_cannot_take_the_pod_network_away_when_it_is_destroyed() -> None:
    """Two guards, against two different mistakes, and both are needed.

    `preserve` decides what a destroy leaves behind: without it, removing this resource
    deletes the CNI DaemonSet, and no node can then give a new pod an address. With it,
    the Kubernetes objects stay and only EKS's management of them ends -- which is the
    state this cluster is in today, so it is a state it can live in.

    `prevent_destroy` stops the plan being made at all, turning it into exit 1 with the
    plan still printed. The first makes the destroy harmless; the second makes it loud.
    """
    text = _VPC_CNI_TF.read_text()
    assert "preserve = true" in text, (
        f"{_VPC_CNI_TF.name} does not set preserve, so `terraform destroy` on this "
        "resource deletes the CNI DaemonSet and nothing schedulable starts on any node"
    )
    assert "prevent_destroy = true" in text, (
        f"{_VPC_CNI_TF.name} does not set prevent_destroy, so a plan that replaces "
        "this add-on renders as an ordinary change rather than as an error"
    )


# --- Tier B: the cluster, which is the only thing that can tell a filter from a note. -

_GATE = "MAP_CLUSTER_TESTS"

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"the policy set's enforcement is opt-in: set {_GATE}=1 to run it. It needs "
        "kubectl pointed at map-dev and reads two objects. SKIPPED MEANS NOTHING WAS "
        "MEASURED ABOUT ENFORCEMENT -- every case above passes identically against a "
        "cluster where the whole policy set is inert, because that is what a cluster "
        "with a NetworkPolicy and no enforcement looks like from a file."
    ),
)


def _kubectl(*argv: str) -> str:
    done = subprocess.run(
        ["kubectl", *argv], capture_output=True, text=True, timeout=120
    )
    if done.returncode != 0:
        pytest.fail(f"kubectl {' '.join(argv)} failed:\n{done.stderr}")
    return done.stdout


@requires_the_cluster
def test_the_cni_is_enforcing_network_policy() -> None:
    """The only case in this file that can tell a boundary from a description of one.

    THIS FAILS UNTIL THE ADD-ON IS APPLIED, and that is the point rather than a defect
    in the test. Everything else here reads files, and a NetworkPolicy is accepted by
    the API server whether or not anything enforces it -- so without this case a
    complete default-deny policy set and no enforcement at all is a green suite.

    The container is found BY NAME and its arguments are asserted non-empty before the
    flag is read, so a green result means "the enforcing container's own arguments say
    on" rather than "we looked somewhere and found nothing to object to". `aws-node`
    carries two containers and the enforcing one is second today; an index would keep
    working right up until the order changed, and would then read the wrong container's
    arguments, which is a reading that looks like an answer.
    """
    described = json.loads(
        _kubectl("get", "daemonset", "aws-node", "-n", "kube-system", "-o", "json")
    )
    containers = {
        str(container["name"]): tuple(
            str(argument) for argument in container.get("args") or ()
        )
        for container in described["spec"]["template"]["spec"]["containers"]
    }
    arguments = containers.get("aws-eks-nodeagent", ())
    assert arguments, (
        "the aws-eks-nodeagent container was not found in kube-system/aws-node, or it "
        f"runs with no arguments. Containers there: {sorted(containers)}. Nothing "
        "about enforcement has been measured -- do not read this as it being off."
    )
    assert _platform().enforcing_network_policy(arguments), (
        "the CNI is NOT enforcing network policy, so every NetworkPolicy in this "
        "cluster is accepted, listed and applied to no packet. The nodeagent runs with "
        f"{list(arguments)}.\n\n"
        "Nothing is broken and nothing was switched off by hand: the CNI on this "
        "cluster is the install-time default -- `aws eks list-addons` does not list "
        "vpc-cni and the aws-node DaemonSet is labelled managed-by Helm -- and that "
        "default ships the enforcing agent turned off. NETWORK_POLICY_ENFORCING_MODE "
        "on the same DaemonSet is not this: it says what happens to a pod's traffic "
        "while its policies are being programmed, not whether they are.\n\n"
        "The lever is deploy/terraform/vpc_cni.tf, which declares the vpc-cni add-on "
        "with enableNetworkPolicy. Applying it rolls aws-node on every node, so pod "
        "networking is briefly unavailable for a pod being created on a node whose "
        "aws-node is restarting. Read that file's header before applying it to a "
        "cluster that is serving."
    )


@requires_the_cluster
def test_every_policy_this_repo_declares_is_in_the_cluster() -> None:
    """A manifest in the repository and a manifest in the cluster are different facts.

    This project has paid for that gap twice -- a headless Service committed and
    applied by nothing, so a healthy Session pod was unaddressable, and a ServiceAccount
    declared and never applied, so a Deployment sat at 0/2. A policy set is the same
    shape with a worse failure: nothing reports the absence, and the boundary is simply
    not there.

    Nothing in this repository applies this file. Until something does, the command is
    `kubectl apply -f deploy/k8s/network-policies.yaml`, and this case is what says
    whether it has been run.
    """
    listed = json.loads(
        _kubectl("get", "networkpolicy", "-n", _NAMESPACE, "-o", "json")
    )
    live = {str(item["metadata"]["name"]) for item in listed["items"]}
    declared = set(_policies())
    assert declared <= live, (
        f"{sorted(declared - live)} are declared in deploy/k8s/network-policies.yaml "
        f"and are not in namespace {_NAMESPACE}. The cluster holds {sorted(live)}. "
        "Apply the set: kubectl apply -f deploy/k8s/network-policies.yaml"
    )
