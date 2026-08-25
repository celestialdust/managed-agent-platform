"""The control plane's manifest and its migration Job, graded on what makes them work.

Tier 1, local, no cluster. Nothing here validates the documents against the real
Kubernetes schemas: `kubeconform` is not installed and is not a row in
`environment.md`, so a tactic naming it would be a check that silently never runs, and a
field name misspelled in a way that still parses as YAML reaches `kubectl apply`
uncaught.

Three cases are cross-reads rather than assertions about a literal, and they are the
ones that matter most. The probe path is read off the app, because a manifest naming a
path the code stopped serving is a pod that never becomes ready. The namespace is read
off `deploy/bootstrap.py`, because three copies of a name fixed by three IAM trust
policies is two copies too many. The object bucket is read off
`deploy/iam/map-control-plane.json`, because a bucket named here without a grant on the
same bucket is an AccessDenied on every upload and reads as good configuration.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final
from urllib.parse import urlsplit

import pytest
import yaml

from managed_agent.control.api.routes.health import HEALTH_ROUTE

_ROOT = Path(__file__).resolve().parents[2]
_K8S = _ROOT / "deploy" / "k8s"
_DEPLOYMENT = _K8S / "control-plane.yaml"
_JOB = _K8S / "schema-migration.yaml"
_POLICY = _ROOT / "deploy" / "iam" / "map-control-plane.json"

PLACEHOLDER: Final = "sha256:" + "0" * 64
ALLOWED_LITERAL_VARIABLES: Final = frozenset(
    {
        "MAP_NAMESPACE",
        "MAP_POD_MANIFEST",
        "MAP_TOOL_GATEWAY_URL",
        "MAP_MODEL_GATEWAY_URL",
        "MAP_SESSION_TOKEN_LIFETIME_S",
        "MAP_OBJECT_BUCKET",
        "MAP_ROLLOUT_BUCKET",
        "MAP_SWEEP_INTERVAL_S",
    }
)
"""The environment variables in this manifest that may carry a `value:`.

Addresses, paths, bucket names and durations -- a namespace, a mounted file, the two
gateway Services, the bucket uploads and Evidence go to, the bucket Rollouts go to, how
long a Session's token stays valid, and how long the in-process sweeps wait between
passes. Named rather than counted, deliberately: a count written here is a second copy
of `len(ALLOWED_LITERAL_VARIABLES)`, and this docstring carried two of them that
disagreed -- one updated when an entry was added and one not, with the stale number in
the more quotable sentence. A reader who wants the count reads the set.

None of them is a credential. A bucket name is a name, and what makes reaching it
possible is a role rather than this string; every one of them is a fact an operator has
to be able to read out of the file they applied.

`tests/deploy/test_tool_gateway_manifest.py` bans literals outright, and its reasoning
is right for a manifest whose two variables are both credentials: "the difference
between a name and a value is the whole of whether this file can leak a credential." An
allowlist keeps that force -- anything not named here still has to be a secretKeyRef --
without pushing an address into a Secret, which would hide a non-secret in the one place
nobody can read.

**A list this long looks like one somebody maintains, and it stays an allowlist
anyway.** The two directions fail differently: an allowlist that
has fallen behind refuses a correct manifest, loudly, at the gate; a denylist of
credential-shaped names that has fallen behind admits a new credential as a literal,
silently, into the one file whose whole job is to hold names and not values. Fail-safe
defaults decide it, because the thing guarded is credential disclosure. This list is not
closed and is not claimed to be.
"""

# The uid the platform image bakes a passwd entry for. A numeric runAsUser with no entry
# in the image has no HOME; the Session image's is 10001 and this must not be it.
SHIPPING_UID: Final = 10002


def _load(path: Path) -> list[dict[str, Any]]:
    documents = [d for d in yaml.safe_load_all(path.read_text()) if d]
    assert documents, f"{path} parsed into no documents at all"
    return documents


def _container(path: Path) -> dict[str, Any]:
    only = _load(path)[0]["spec"]["template"]["spec"]["containers"]
    assert len(only) == 1, "one container, or the assertions below grade one of N"
    return dict(only[0])


def _only(path: Path, kind: str) -> dict[str, Any]:
    """The single document of one kind in a manifest.

    Found by kind rather than by index, unlike `_container` above, which takes document
    0. Both are correct and they are correct for different reasons: the Deployment is
    pinned to position 0 by the case below and by a reader in another file, so indexing
    it is safe; a Service that moved would be found here and misgraded there.
    """
    found = [document for document in _load(path) if document["kind"] == kind]
    assert len(found) == 1, f"{path} holds {len(found)} {kind} documents; expected 1"
    return dict(found[0])


def _pod_selector() -> dict[str, str]:
    """The labels this Deployment selects its own pods by.

    Asserts the Deployment's two halves agree before returning, and that is the whole
    reason this is a function. A Deployment whose `selector.matchLabels` and its pod
    template's labels disagree owns no pod at all -- and every "this selector matches
    the Deployment's pods" assertion below would then compare two things against a set
    nothing is in, and pass.
    """
    spec = _only(_DEPLOYMENT, "Deployment")["spec"]
    selector = dict(spec["selector"]["matchLabels"])
    labels = dict(spec["template"]["metadata"]["labels"])
    assert selector, "the Deployment selects on no label at all"
    assert selector == labels, (
        f"the Deployment selects {selector} and labels its pods {labels}. A Deployment "
        "whose two halves disagree owns no pod, and the selector checks below would be "
        "comparing against an empty set"
    )
    return selector


def _service_port_named(name: str) -> int:
    """The port of the one Service under deploy/k8s/ carrying this name.

    Swept over the directory rather than indexed to a file, so the answer follows a
    Service that moves between manifests. Exactly one port and exactly one Service, or
    this fails: the caller below compares numbers, and a number picked out of several
    would be comparing whichever came first.
    """
    found = [
        port["port"]
        for path in sorted(_K8S.glob("*.yaml"))
        for document in _load(path)
        if document["kind"] == "Service" and document["metadata"]["name"] == name
        for port in document["spec"]["ports"]
    ]
    assert len(found) == 1, f"{name}: found {len(found)} Service ports, expected 1"
    return int(found[0])


def _bootstrap() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "map_bootstrap_for_control_plane", _ROOT / "deploy" / "bootstrap.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_file_holds_the_deployment_first_then_a_service_and_a_budget() -> None:
    """The document set AND its order, because the order is load-bearing.

    Two readers take document 0 of this file as the Deployment instead of searching for
    it by kind: `_container` above, and
    tests/deploy/test_the_placer_is_configured_wherever_it_is_deployed.py's
    `_documents(_CONTROL_PLANE)[0]`. Putting the Service first would have them grade a
    Service as a Deployment, and they would fail somewhere that says nothing about
    document order.

    This case replaces `test_the_file_holds_exactly_one_deployment_and_no_service`,
    which asserted the opposite and was right when it was written. Its citation of
    ADR-022:95-98 was accurate and stays accurate -- neither a Service nor a
    NetworkPolicy existed, so "the boundary was assumed, not built" -- but the
    conclusion drawn from it here does not survive `replicas` leaving 1.
    `kubectl port-forward deploy/control-plane` reaches ONE arbitrary pod of the set, so
    with more than one replica the alternative to a Service is not "one fewer address to
    defend"; it is no correct address at all. The authentication gap the old case named
    is entirely unchanged, which is why the next case asserts `type`.
    """
    assert [d["kind"] for d in _load(_DEPLOYMENT)] == [
        "Deployment",
        "Service",
        "PodDisruptionBudget",
    ]


def test_the_service_is_cluster_ip_and_nothing_here_publishes_it_outward() -> None:
    """The one field between an unauthenticated API and the internet.

    control/api/app.py declares no middleware and no Depends, and every gated route
    takes its tenant from a request header whose own docstring says the value "is
    asserted by whoever sent the request". A LoadBalancer or a NodePort here would
    therefore serve every tenant's Sessions, Turns and uploaded files to anybody who can
    set a header.

    An EQUALITY on `type`, not only the absence of a string. An absence is satisfied by
    the field being gone for any reason at all -- a document that lost its whole `spec`
    would pass a scan for "type: LoadBalancer" and fail this. The text scan is kept
    beside the equality as a second net over the file rather than over one document,
    because a value that can be copied out of a comment is a value.
    """
    assert _only(_DEPLOYMENT, "Service")["spec"]["type"] == "ClusterIP"
    text = _DEPLOYMENT.read_text()
    for forbidden in (
        "type: LoadBalancer",
        "type: NodePort",
        "kind: Ingress",
        "externalIPs",
    ):
        assert forbidden not in text, forbidden


def test_the_service_selects_the_pods_this_deployment_owns() -> None:
    """A Service whose selector matches no pod is valid YAML and routes to nothing.

    Compared against the Deployment in the same file rather than against a literal, so
    renaming the component label moves both sides together and renaming one side fails
    here.

    The failure prevented is not hypothetical at the Service level in this project: a
    Session pod at `2/2 Running` with its shim answering the kubelet failed every Turn
    as unreachable, because the Service publishing its address had been applied by
    nothing. `kubectl apply` cannot tell an address that routes from one that does not,
    and neither can `kubectl rollout status`.
    """
    assert _only(_DEPLOYMENT, "Service")["spec"]["selector"] == _pod_selector()


def test_the_service_routes_to_the_port_the_readiness_probe_proves_alive() -> None:
    """A named targetPort is resolved against the pod's own containerPort names.

    One that names nothing yields an endpoint with no port rather than an endpoint to
    the wrong port, so the address resolves, accepts a connection and carries no
    traffic. Read off the container instead of written down.

    Equal to the readiness probe's port and not merely present among the container's,
    which is the stronger of the two: a Service publishing a port whose health nothing
    checks would send traffic to a listener that may never have come up.
    """
    container = _container(_DEPLOYMENT)
    declared = {port["name"] for port in container["ports"]}
    ports = _only(_DEPLOYMENT, "Service")["spec"]["ports"]
    assert len(ports) == 1, f"one Service port, or this grades one of {len(ports)}"
    assert ports[0]["targetPort"] in declared, (ports[0], declared)
    assert ports[0]["targetPort"] == container["readinessProbe"]["httpGet"]["port"]


_PORTLESS_URL_VARIABLES: Final = ("MAP_TOOL_GATEWAY_URL", "MAP_MODEL_GATEWAY_URL")
"""The in-cluster addresses this manifest hands out, which carry no port.

Named here because the case below needs somewhere to start, and two is the whole set as
of this commit -- `MAP_TOOL_GATEWAY_URL` and `MAP_MODEL_GATEWAY_URL` are the only
`http://` values in `ALLOWED_LITERAL_VARIABLES`, checked at the moment this was written
by reading the manifest's env block. A third would not be covered until it is added, and
the case asserts the count so that adding one to the manifest and not here is visible.
"""


def test_the_service_port_is_the_one_this_platforms_portless_urls_reach() -> None:
    """Derived from the addresses this Deployment already hands out, not chosen here.

    Nothing in this tree dials the control plane by name today, so no compiled URL pins
    this number the way every compiled config.toml pins model-gateway.yaml's. What pins
    it instead is the convention: both in-cluster URLs this manifest sets carry no port,
    a URL with no port is port 80, and the Services behind them are read out of
    deploy/k8s/ here rather than restated -- so this fails if the convention itself
    moves.

    A third platform Service on any other number would mean the one natural spelling of
    its own address, `http://control-plane.<namespace>.svc.cluster.local/v1/...`,
    reaches nothing -- while reading as perfectly good configuration.
    """
    environment = {entry["name"]: entry for entry in _container(_DEPLOYMENT)["env"]}
    ports: dict[str, int] = {}
    for variable in _PORTLESS_URL_VARIABLES:
        url = urlsplit(environment[variable]["value"])
        assert url.port is None, (
            f"{variable} now carries an explicit port ({url.port}), so it no longer "
            "shows that this platform addresses its own services on the default port"
        )
        assert url.hostname, variable
        ports[variable] = _service_port_named(url.hostname.split(".")[0])
    assert len(ports) == len(_PORTLESS_URL_VARIABLES), ports
    assert set(ports.values()) == {80}, (
        f"the platform's portless URLs reach {ports}, so 80 is no longer the number "
        "this convention means, and the control-plane Service no longer follows from it"
    )
    assert _only(_DEPLOYMENT, "Service")["spec"]["ports"][0]["port"] == 80


_SPREAD_DOMAINS: Final = frozenset(
    {"kubernetes.io/hostname", "topology.kubernetes.io/zone"}
)
"""The two failure domains the replica count is argued against, so both need covering.

The count is justified on ZONES -- the nodegroup spans exactly two subnets, so two is
the largest number of replicas each holding a zone nothing else can take with it. The
drain hazard that motivated more than one replica is per NODE. On this cluster those are
not the same partition: it ran four nodes across two zones on 2026-08-23, so a hostname
constraint alone permits both replicas in one zone, and a zone constraint alone permits
both on one node inside a zone.
"""


def test_the_replicas_cannot_all_land_on_one_node_or_in_one_zone() -> None:
    """Each maxSkew must be small enough that all the replicas together violate it.

    Asserted as a relation to `replicas` rather than as the literal 1, because the
    property is not "the number is 1" -- it is "all of them in one domain is a
    violation". Every replica in one domain is a skew of `replicas`, so any maxSkew at
    or above that permits exactly the state the constraint exists to forbid, while
    reading like a constraint.

    Both domains, and one constraint per domain. A set missing either leaves one of the
    two arguments for this Deployment's shape unenforced, and the gap is invisible: a
    hostname-only spread reads as "the replicas are kept apart" while permitting both of
    them in the zone whose loss the replica count was chosen to survive.

    Why any explicit constraint is needed: kube-scheduler's defaults are maxSkew 3 on
    hostname and 5 on zone (its PodTopologySpread `defaultConstraints`, read from the
    upstream default -- EKS does not expose its scheduler configuration, so neither
    number is measured on this cluster). Two pods together against zero everywhere else
    is a skew of 2, inside both.

    ScheduleAnyway is pinned because DoNotSchedule is the wrong trade and the difference
    is invisible in a diff: DoNotSchedule leaves the second replica Pending whenever
    only one node or zone can take it, which is exactly when the availability this buys
    is scarce. Nothing local can grade the effect of either -- a spread constraint's
    outcome is a scheduler's decision, and only a live cluster disagrees with a
    well-formed one.
    """
    spec = _only(_DEPLOYMENT, "Deployment")["spec"]
    constraints = spec["template"]["spec"]["topologySpreadConstraints"]
    keys = [constraint["topologyKey"] for constraint in constraints]
    assert len(keys) == len(set(keys)), f"a domain is constrained twice: {keys}"
    assert set(keys) == set(_SPREAD_DOMAINS), (
        f"the spread constraints cover {sorted(keys)} and the replica count is argued "
        f"against {sorted(_SPREAD_DOMAINS)}. A domain with no constraint is one where "
        "every replica may sit together, which is the state the count was chosen to "
        "avoid."
    )
    for constraint in constraints:
        where = constraint["topologyKey"]
        assert int(constraint["maxSkew"]) < int(spec["replicas"]), (
            f"maxSkew {constraint['maxSkew']} on {where} against replicas "
            f"{spec['replicas']}: every replica in one {where} is a skew of replicas, "
            "so this constraint permits the very state it exists to forbid"
        )
        assert constraint["whenUnsatisfiable"] == "ScheduleAnyway", where
        assert constraint["labelSelector"]["matchLabels"] == _pod_selector(), where


def test_the_budget_keeps_one_pod_without_blocking_every_drain_for_ever() -> None:
    """minAvailable must be at least 1 and below `replicas`, and both halves matter.

    Below 1 the budget refuses nothing. At or above `replicas` it permits nothing: every
    eviction of every pod violates it, so a node drain stalls indefinitely on `Cannot
    evict pod as it would violate the pod's disruption budget` and the cluster cannot be
    maintained. Between them is the only interval where a budget does what it is for.

    `maxUnavailable` is refused rather than accepted as equivalent. At two replicas the
    two spellings permit the same thing, and they diverge the moment somebody scales
    this Deployment to one: `maxUnavailable: 1` lets the last pod go, and
    `minAvailable: 1` blocks the drain instead.

    What this cannot grade: whether an eviction is actually refused. That is a live
    admission decision, and a budget selecting the wrong pods reports `expectedPods: 0`
    and permits everything -- which is why the selector has a case of its own above.
    """
    replicas = int(_only(_DEPLOYMENT, "Deployment")["spec"]["replicas"])
    budget = _only(_DEPLOYMENT, "PodDisruptionBudget")["spec"]
    assert "maxUnavailable" not in budget, (
        "maxUnavailable and minAvailable spell the same permission at two replicas and "
        "diverge at one, where maxUnavailable: 1 lets the last pod go"
    )
    assert 1 <= int(budget["minAvailable"]) < replicas, (budget, replicas)
    assert budget["selector"]["matchLabels"] == _pod_selector()


def test_the_pod_runs_as_a_non_root_user_the_image_has_a_passwd_entry_for() -> None:
    security = _load(_DEPLOYMENT)[0]["spec"]["template"]["spec"]["securityContext"]
    assert security["runAsNonRoot"] is True
    assert security["runAsUser"] == SHIPPING_UID
    assert security["seccompProfile"]["type"] == "RuntimeDefault"


def test_the_container_cannot_escalate_write_its_root_or_keep_a_capability() -> None:
    security = _container(_DEPLOYMENT)["securityContext"]
    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"]["drop"] == ["ALL"]


def test_the_pod_takes_none_of_the_host() -> None:
    pod = _load(_DEPLOYMENT)[0]["spec"]["template"]["spec"]
    assert "hostNetwork" not in pod
    assert "hostPID" not in pod
    assert "privileged" not in _container(_DEPLOYMENT)["securityContext"]


@pytest.mark.parametrize("path", [_DEPLOYMENT, _JOB])
def test_only_the_one_allowlisted_variable_carries_a_literal_value(path: Path) -> None:
    env = _container(path)["env"]
    assert env, "no environment entries found, so this assertion grades nothing"
    for entry in env:
        if entry["name"] in ALLOWED_LITERAL_VARIABLES:
            assert "value" in entry, (
                f"{entry['name']} is allowlisted and is not literal"
            )
            continue
        assert "value" not in entry, f"{entry['name']} carries a literal value"
        assert "secretKeyRef" in entry["valueFrom"], entry["name"]


def test_the_namespace_it_names_is_the_one_bootstrap_names() -> None:
    """Fixed by three IAM trust policies, not chosen here. A pod anywhere else holds a
    projected token no role exchanges, and that fails at the vault rather than here."""
    expected = _bootstrap().NAMESPACE
    for path in (_DEPLOYMENT, _JOB):
        for document in _load(path):
            assert document["metadata"]["namespace"] == expected, path

    declared = {
        entry["value"]
        for entry in _container(_DEPLOYMENT)["env"]
        if entry["name"] == "MAP_NAMESPACE"
    }
    assert declared == {expected}, (
        "composition.py's default namespace is 'default', which resolves to no pod; "
        "the manifest is where that is corrected and it must agree with bootstrap"
    )


def _actions_of(statement: dict[str, Any]) -> list[str]:
    """A statement's actions, whether it spells them as one string or as a list.

    Both spellings are legal IAM and mean the same thing, so a reader of this policy
    must not be able to disable the cases below by rewriting a list of one as a bare
    string.
    """
    action = statement["Action"]
    return [action] if isinstance(action, str) else list(action)


def _s3_statements() -> list[dict[str, Any]]:
    """Every statement in this role's policy that grants anything in S3.

    Found by scanning the actions rather than by index or by Sid, so it follows a
    statement that is reordered or renamed.
    """
    return [
        dict(statement)
        for statement in json.loads(_POLICY.read_text())["Statement"]
        if any(action.startswith("s3:") for action in _actions_of(statement))
    ]


def _s3_statement() -> dict[str, Any]:
    """The one statement that grants actions on OBJECTS in the bucket.

    Separated from the bucket-level grant by its Resource rather than by its Sid,
    because the two are different kinds of permission and IAM enforces the difference:
    an object action needs `arn:...:bucket/*` and a bucket action needs
    `arn:...:bucket`, and each is inert on the other's ARN. A policy that spelled
    `s3:ListBucket` against `/*` would read as a granted listing and grant nothing --
    which is a failure that looks like a working policy until something tries to list.

    Exactly one, or this fails: the cases below describe "the object grant" as a single
    thing, and picking the first of several would grade whichever the file happened to
    list first while the rest went unread.
    """
    statements = [
        statement
        for statement in _s3_statements()
        if str(statement["Resource"]).endswith("/*")
    ]
    assert len(statements) == 1, (
        f"{_POLICY.relative_to(_ROOT)} holds {len(statements)} statements granting S3 "
        "actions; the cases below are written about one"
    )
    return statements[0]


def test_the_object_bucket_it_names_is_the_one_this_role_is_granted() -> None:
    """Read out of the policy, not written down here a second time.

    The two halves fail differently and both fail late. With no variable the upload
    surface answers 500 naming the variable, which is at least a diagnosis. With a
    variable naming a bucket this role has no grant on, every upload is an AccessDenied
    from botocore -- and the manifest reads as perfectly good configuration, because
    nothing in it can be wrong on its own. Only the pair is right or wrong, so only the
    pair is asserted.

    Derived rather than pinned also gives this case its own control: a policy whose S3
    statement was dropped or re-scoped fails here, instead of leaving an assertion that
    compares the manifest against a bucket nothing grants.
    """
    granted = _s3_statement()["Resource"]
    assert granted.startswith("arn:aws:s3:::") and granted.endswith("/*"), (
        f"{granted} is not an objects-in-one-bucket ARN, so the bucket the manifest "
        "must name cannot be read out of it"
    )
    bucket = granted.removeprefix("arn:aws:s3:::").removesuffix("/*")

    named = {
        entry["value"]
        for entry in _container(_DEPLOYMENT)["env"]
        if entry["name"] == "MAP_OBJECT_BUCKET"
    }
    assert named == {bucket}, (
        f"the manifest names {named or 'no object bucket'} and this role is granted "
        f"objects in {bucket}; unset, POST /v1/files answers 500 storage_unconfigured, "
        "and pointed anywhere else it answers AccessDenied"
    )


def test_the_grant_behind_that_bucket_reads_and_writes_and_cannot_delete() -> None:
    """Exactly the two calls the upload path makes, and no third.

    Asserted as an equality rather than as two `in` checks, because the claim the
    manifest makes beside that variable is about what this role CANNOT do: an uploaded
    file is referenced by a Session's mount long after the upload call returned, so a
    Delete reachable from this process is a tenant's file removable while something
    still points at it. An `in` pair would keep passing after somebody widened the
    statement, which is the only way that claim can stop being true.

    A widening that is genuinely wanted fails here and is meant to: the Evidence sweep
    that shares this bucket needs s3:DeleteObject, and wiring it into this process is a
    decision somebody takes in the policy with this case in front of them rather than
    one that arrives as a passing test. `s3:ListBucket` was exactly such a widening,
    taken on 2026-08-24 for the Session VFS -- and it is NOT here, because it is a
    bucket action and belongs on the bucket ARN, which is the case below.
    """
    granted = set(_actions_of(_s3_statement()))
    assert granted == {"s3:GetObject", "s3:PutObject"}, (
        f"this role is granted {sorted(granted)} in S3. The manifest says beside "
        "MAP_OBJECT_BUCKET that the grant is read-and-write and carries no Delete; "
        f"{sorted(granted - {'s3:GetObject', 's3:PutObject'})} is more than that"
    )


def test_the_bucket_itself_is_listable_and_nothing_more() -> None:
    """The one bucket-level grant, taken deliberately for the Session VFS.

    `list_lane` is a `ListObjectsV2` under a Session's prefix, which is `s3:ListBucket`
    on the bucket -- not on its objects. Without it a lane cannot be enumerated, so a
    working tree cannot be materialized back into a resumed pod, and the four-operation
    port the VFS declares has one operation that always fails.

    **Asserted as an equality, and on this ARN rather than that one.** A `ListBucket`
    written against `arn:...:bucket/*` is the mistake this case exists to catch: it
    parses, it deploys, it reads in a policy as a granted listing, and it grants
    nothing. The failure surfaces at the first call as an AccessDenied that looks like a
    missing grant rather than a misplaced one.

    What this must not become is the door to the rest of the bucket. `s3:ListBucket`
    with no condition lists every key in it, including Evidence and every other tenant's
    uploads. That is scoped in the CALLER rather than in the policy -- `lane_prefix`
    composes the tenant into the prefix, so a control plane asking for another tenant's
    objects would have to build a prefix it has no code path to build. Stated plainly
    because the alternative is an `s3:prefix` condition, which cannot be written here:
    one policy would have to enumerate every tenant.
    """
    listing = [
        statement
        for statement in _s3_statements()
        if not str(statement["Resource"]).endswith("/*")
    ]
    assert len(listing) == 1, (
        f"{len(listing)} statements grant a bucket-level S3 action; the case below "
        "describes the bucket grant as a single thing"
    )
    granted = set(_actions_of(listing[0]))
    assert granted == {"s3:ListBucket"}, (
        f"this role is granted {sorted(granted)} on the bucket itself. The VFS needs "
        "exactly ListBucket to enumerate a lane; anything else here is reach over the "
        "whole bucket that no code path in this process asks for"
    )
    objects = str(_s3_statement()["Resource"])
    assert str(listing[0]["Resource"]) == objects.removesuffix("/*"), (
        f"the bucket grant names {listing[0]['Resource']} and the object grant names "
        f"{objects}; a ListBucket on any other bucket is a listing of somebody else's"
    )


def test_no_statement_anywhere_in_this_policy_can_delete_an_object() -> None:
    """The manifest's claim, checked over the WHOLE policy rather than one statement.

    `test_the_grant_behind_that_bucket_reads_and_writes_and_cannot_delete` asserts an
    equality over the object statement, which was the whole policy's S3 surface while
    there was one S3 statement. There are two now, so that case has stopped being a
    claim about the role and become a claim about one statement in it -- and a Delete
    added in a third statement would leave it green.

    That matters more here than it would elsewhere: a sealed lane's immutability rests
    on `IfNoneMatch` refusing an overwrite, and an overwrite is the only way to destroy
    a stored object at all WHEN THERE IS NO DELETE. A Delete reachable from this process
    unmakes the guarantee without touching a line of the code that states it.
    """
    granted = {
        action for statement in _s3_statements() for action in _actions_of(statement)
    }
    deletes = {action for action in granted if "Delete" in action}
    assert deletes == set(), (
        f"this role is granted {sorted(deletes)}. A sealed lane's immutability is a "
        "conditional put plus the absence of a delete; with one, the seal is a "
        "convention rather than a guarantee"
    )


def test_both_probes_name_a_path_the_app_actually_serves_without_a_tenant() -> None:
    """Read off the code rather than written down twice. A probe holds no tenant header,
    so it can only reach a route that requires none."""
    container = _container(_DEPLOYMENT)
    for probe in ("readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"]["path"] == f"/v1{HEALTH_ROUTE}"
        assert container[probe]["httpGet"]["port"] == "http"
    assert {port["name"] for port in container["ports"]} == {"http"}


@pytest.mark.parametrize("path", [_DEPLOYMENT, _JOB])
def test_the_image_is_the_digest_placeholder_and_not_a_tag(path: Path) -> None:
    """A tag is a name for whatever was pushed last; a digest is a name for bytes. The
    placeholder is what deploy/platform.py substitutes and what it refuses to apply
    without."""
    image = _container(path)["image"]
    assert image == f"map/control-plane@{PLACEHOLDER}", image
    assert ":latest" not in image


def test_the_job_names_the_same_image_placeholder_as_the_deployment() -> None:
    """One image, or the Job upgrades to a schema this code does not read."""
    assert _container(_JOB)["image"] == _container(_DEPLOYMENT)["image"]


# Measured from inside a control-plane pod against the live map-dev instance,
# not read off the parameter group's formula:
#
#     show max_connections                = 400
#     show superuser_reserved_connections =   3
#     select rolsuper ... current_user    -> false   (map_app is not a superuser)
#
# so 397 are available. Two earlier figures were wrong and both were DERIVED:
# 225 from the parameter group's formula at db.t4g.small (47 too high, and it
# passed a demand of 180 against a real 178), and 362 from doubling the measured
# 181 when the instance was resized to db.t4g.medium. The instance answers 400.
# Re-measure this after any instance-class change; do not compute it.
_USABLE_CONNECTIONS = 397

# Backends this platform does not open -- RDS's own monitoring, an operator's
# psql. Ten were in use with the platform idle; fifteen is that with a little
# room. Subtracting it makes the remainder the platform's own budget rather
# than the instance's total.
_RESERVED_FOR_OTHERS = 15


def _peak_pods(spec: dict[str, Any]) -> int:
    """The most pods of one Deployment that can exist at once, rollout included.

    Steady state is `replicas`. During a rolling update Kubernetes may start
    extra pods up to `maxSurge`, which defaults to **25% rounded up** when the
    manifest is silent -- and silence is the common case, so a guard reading
    only `replicas` would understate every Deployment that had not thought
    about it. A percentage resolves against `replicas`; an integer is taken as
    given.
    """
    replicas = int(spec["replicas"])
    rolling = (spec.get("strategy") or {}).get("rollingUpdate") or {}
    surge = rolling.get("maxSurge", "25%")
    if isinstance(surge, str) and surge.endswith("%"):
        extra = math.ceil(replicas * int(surge[:-1]) / 100)
    else:
        extra = int(surge)
    return replicas + extra


def _database_workloads() -> dict[str, int]:
    """Every workload under deploy/k8s/ that opens a connection, by peak pods.

    Discovered by reading each manifest for the database Secret rather than by
    naming the two that exist today. The earlier guard named
    `control-plane.yaml` and `tool-gateway.yaml` as literals, so a third
    database-using workload would have raised the real total while the check
    kept reporting the old one. A workload counts when any container names the
    `database-url` key of Secret `map-control-plane`, which is the only way to
    reach the database from a pod.

    A `Job` counts as one pod: alembic opens a single connection, no pool.
    """
    found: dict[str, int] = {}
    for manifest in sorted(_K8S.glob("*.yaml")):
        for doc in yaml.safe_load_all(manifest.read_text()):
            if not doc or doc.get("kind") not in {"Deployment", "Job"}:
                continue
            spec = doc["spec"]
            opens_a_connection = any(
                (variable.get("valueFrom") or {}).get("secretKeyRef", {}).get("key")
                == "database-url"
                for container in spec["template"]["spec"].get("containers", [])
                for variable in container.get("env", [])
            )
            if opens_a_connection:
                name = f"{manifest.name}:{doc['metadata']['name']}"
                found[name] = _peak_pods(spec) if doc["kind"] == "Deployment" else 1
    return found


def test_the_replica_count_and_surge_fit_the_databases_connection_ceiling() -> None:
    """Every database-using workload's peak pods times the pool must fit 178.

    The arithmetic is ADR-022:89-93's; the ceiling is measured rather than
    derived from the parameter group's formula, because the derived figure was
    47 too high and let through a configuration that already exceeded the
    instance.

    Peak, not steady state: a Deployment that does not pin `maxSurge` gets 25%
    during a rollout, and the surging pods open pools of their own while the
    ones they replace still hold theirs. Past the ceiling the instance answers
    `FATAL: too many connections`, which environment.md's postgres row records
    as "a total outage, as the designed response to load" -- so the failure
    mode of getting this wrong is not degradation.
    """
    from managed_agent import composition

    per_process = composition._POOL_SIZE + composition._MAX_OVERFLOW
    workloads = _database_workloads()
    assert workloads, "no workload reads the database Secret; this guard found nothing"

    mine = f"{_DEPLOYMENT.name}:control-plane"
    assert mine in workloads, (
        f"the discovery did not find {mine}, so this sum is being computed without the "
        f"one Deployment this file is about. Found: {sorted(workloads)}"
    )
    assert workloads[mine] >= int(
        _only(_DEPLOYMENT, "Deployment")["spec"]["replicas"]
    ), (
        f"{mine} contributes {workloads[mine]} peak pods, fewer than the replicas it "
        f"declares: {workloads}"
    )

    budget = _USABLE_CONNECTIONS - _RESERVED_FOR_OTHERS
    peak = sum(workloads.values())
    assert peak * per_process <= budget, (
        f"peak pool is {peak * per_process} connections against a budget of "
        f"{budget} ({_USABLE_CONNECTIONS} usable less {_RESERVED_FOR_OTHERS} "
        f"for backends this platform does not open); {per_process} per process "
        f"across {peak} peak pods: {workloads}"
    )


def test_more_than_one_replica_serves_and_a_rollout_opens_no_extra_pool() -> None:
    """More than one pod, and no surge, so a rollout adds no connections.

    `> 1` and not `== 2`, and that shape is inherited rather than invented: the literal
    it replaces was `replicas == 1`, and the note beside it recorded why the literal was
    a problem even when it was true. With a literal asserted here, the ceiling
    arithmetic above became unreachable from this file -- raising `replicas` failed on
    the literal and never reached the sum. `> 1` keeps both halves live: a count the
    database cannot hold passes here and fails there, which is where that answer
    belongs.

    What `> 1` is for. One pod is a single point of failure for the whole platform's
    tenant-facing API: a node drain, an eviction or an OOMKill takes the API down, and a
    rollout at one replica and maxSurge 0 has a window with no serving pod at all.

    maxSurge 0 costs no availability above one replica. With maxUnavailable 1 a rollout
    takes one pod down and leaves the rest serving, so the pin costs capacity for the
    length of a rollout instead of a gap -- which is what it cost at one replica, and
    what this file used to say.
    """
    spec = _only(_DEPLOYMENT, "Deployment")["spec"]
    assert int(spec["replicas"]) > 1, (
        "one replica is one node drain away from no tenant-facing API at all"
    )
    assert spec["strategy"]["rollingUpdate"]["maxSurge"] == 0
    assert int(spec["strategy"]["rollingUpdate"]["maxUnavailable"]) >= 1, (
        "Kubernetes rejects a RollingUpdate with maxSurge and maxUnavailable both at "
        "zero, because such a rollout can never take a step"
    )


def test_the_upload_spool_has_somewhere_writable() -> None:
    """The upload route declares an UploadFile, so Starlette writes the body into a
    SpooledTemporaryFile that spills to disk before the route function runs. With a
    read-only root and no writable /tmp, a large upload fails with an OSError rather
    than the named refusal the file store gives."""
    mounts = {m["name"]: m for m in _container(_DEPLOYMENT)["volumeMounts"]}
    volumes = {
        v["name"]: v
        for v in _load(_DEPLOYMENT)[0]["spec"]["template"]["spec"]["volumes"]
    }
    assert mounts["scratch"]["mountPath"] == "/tmp"
    assert "emptyDir" in volumes["scratch"]
    assert "medium" not in volumes["scratch"]["emptyDir"], (
        "a memory-backed spool counts a 100 MiB upload against this container's memory "
        "limit"
    )


def test_the_job_runs_the_migration_runner_from_the_working_directory_it_needs() -> (
    None
):
    """Measured: alembic resolves script_location against the working directory, so from
    anywhere else it answers `CommandError: Path doesn't exist: migrations`."""
    container = _container(_JOB)
    assert container["workingDir"] == "/opt/map"
    assert container["command"][:2] == ["alembic", "-c"]
    assert container["command"][-2:] == ["upgrade", "head"]


def test_the_job_and_the_deployment_read_the_same_secret_key() -> None:
    """One database, one credential. Two keys would be two things to rotate and a second
    way for the runner and the reader to reach different databases."""

    def key_of(path: Path) -> tuple[str, str]:
        entry = next(
            e for e in _container(path)["env"] if e["name"].endswith("DATABASE_URL")
        )
        reference = entry["valueFrom"]["secretKeyRef"]
        return reference["name"], reference["key"]

    assert key_of(_JOB) == key_of(_DEPLOYMENT)


def test_the_job_does_not_retry() -> None:
    """A Job that succeeded on its second attempt reports Complete, and the transient
    failure is then something only the pod list remembers."""
    assert _load(_JOB)[0]["spec"]["backoffLimit"] == 0
    assert _load(_JOB)[0]["spec"]["template"]["spec"]["restartPolicy"] == "Never"
