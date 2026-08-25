"""`deploy/platform.py`, graded on its pure functions and its declarations.

Tier 1, and it reaches no cluster, no registry and no account. Every case here drives a
function that takes its inputs as arguments -- which is why `substituted`,
`rotating_credential` and `deployment_shortfall` take a string, a username and a status
mapping rather than reading any of them. NOT PROVEN by anything here: that `kubectl
apply` is accepted, that the Job completes, or that a pod becomes ready. Those are the
row's checkpoint and need the cluster.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final, NamedTuple

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def _platform() -> ModuleType:
    return _module("map_platform", "deploy/platform.py")


def test_the_namespace_it_reads_is_the_one_bootstrap_names() -> None:
    """Read out of cluster-bootstrap.yaml rather than held as a constant, so there are
    two readers of one file instead of three copies of one name."""
    bootstrap = _module("map_bootstrap_for_platform", "deploy/bootstrap.py")
    assert _platform().namespace(_ROOT) == bootstrap.NAMESPACE


def test_every_workload_names_a_manifest_that_exists() -> None:
    for workload in _platform().WORKLOADS:
        assert (_ROOT / workload.manifest).is_file(), workload.manifest
        if workload.schema_job is not None:
            assert (_ROOT / workload.schema_job).is_file(), workload.schema_job


def test_every_variable_a_workload_requires_is_one_its_manifest_declares() -> None:
    """The applier's own check, run at the gate instead of at the cluster.

    `undeclared_variables` is what refuses an apply, and until now nothing ran it
    offline. So a `required_variables` entry naming a variable no manifest declares --
    a rename on one side, a copy-paste of the wrong spelling -- was a green repository
    and a refusal at the moment somebody was deploying, which is the worst place to
    learn it.

    Parametrized over `WORKLOADS` rather than written per component, because each entry
    in each of those tuples encodes a decision about what that process cannot start
    without. A test naming three of them by hand grades three and silently stops
    covering the fourth.

    This does NOT check that a Secret holds the key, or that the value is right. It
    checks the one thing the manifest can be wrong about on its own: naming the
    variable at all.
    """
    platform = _platform()
    assert platform.WORKLOADS, "no workloads found, so this assertion grades nothing"
    graded = 0
    for workload in platform.WORKLOADS:
        assert platform.undeclared_variables(_ROOT, workload) == (), workload.component
        graded += len(workload.required_variables)
    assert graded, "no workload requires any variable, so this grades nothing"


_KINDS_CARRYING_A_POD: Final = frozenset(
    {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "Job", "Pod"}
)
"""Kinds whose spec must contain a pod, so a missing one is malformed.

`Pod` is in the set and is the one member with no `template` below it -- its `spec`
*is* the pod spec. It was missing when this set was written, so a bare Pod returned the
value meaning "this kind has no pod at all", and `deploy/k8s/session-pod.yaml:15` is
a bare Pod. A Pod appended to any manifest with no per-file document-set guard was
therefore skipped in silence.
"""


def _pod_spec(document: dict[str, Any]) -> dict[str, Any] | None:
    """The pod spec of one manifest document, or None when the kind has no pod at all.

    A workload manifest holds its Service in the same file, and a Service reads no
    environment, so it cannot reference a Secret and is skipped -- by kind, which is
    what makes the skip safe. The version this replaces skipped on the *absence* of a
    template, which is the same thing to a Service and to a Deployment somebody
    malformed: both returned None, both were skipped, and a secretKeyRef nobody
    declared would ride into the cluster with the caller told nothing. The pre-flight
    check that reads these declarations is the last thing between a stale declaration
    and a pod in CreateContainerConfigError, so this raises rather than skips.

    CronJob nests its template one level deeper than every other kind; the indexing is
    deliberate everywhere below, because a KeyError naming the missing key is the
    outcome this function exists to produce.
    """
    kind = document["kind"]
    # The kind decides first, and `spec` is only read after. A ConfigMap has no `spec`
    # key at all -- model-gateway.yaml carries one -- so indexing before the kind check
    # would raise on a document that has nothing to do with pods, which is the same
    # fail-loud-on-the-wrong-thing this function exists to avoid in the other direction.
    if kind not in _KINDS_CARRYING_A_POD and kind != "CronJob":
        return None
    spec: dict[str, Any] = document["spec"]
    if kind == "CronJob":
        nested: dict[str, Any] = spec["jobTemplate"]["spec"]["template"]["spec"]
        return nested
    if kind == "Pod":
        return spec
    pod: dict[str, Any] = spec["template"]["spec"]
    return pod


_WHOLE_SECRET_FORMS: Final = frozenset({"secretRef", "secret", "imagePullSecrets"})
"""Forms by which a manifest depends on a Secret without naming a key inside it.

`envFrom[].secretRef` imports every key as an environment variable, `volumes[].secret`
mounts every key as a file, and `imagePullSecrets` hands the whole thing to the kubelet.
The declaration these tests grade is a set of `(name, key)` pairs, which cannot express
"every key, whichever those turn out to be" -- so a manifest using one of these forms is
refused here rather than silently contributing nothing to the comparison. Whoever needs
one extends the declaration first; the refusal names that.
"""


class SecretReferences(NamedTuple):
    """What one manifest document says about the Secrets it reads.

    Three fields rather than two because `optional: true` is a third state, and the
    version of this that had two fields forced a caller to treat it as one of the
    other two -- both of which are wrong for it.
    """

    required: set[tuple[str, str]]
    optional: set[tuple[str, str]]
    whole: list[str]


def _secret_references(document: dict[str, Any]) -> SecretReferences:
    """Every `(secret, key)` pair one document reads, and any whole-secret form in it.

    Found by walking the document for the *shape* of a reference rather than by
    indexing the path to one. That distinction is the whole point: the version this
    replaces read `spec.template.spec.containers[].env[]`, so a `secretKeyRef` under
    `initContainers`, under `ephemeralContainers`, or reached through any nesting the
    Kubernetes API adds later was invisible to it -- and `session-pod.yaml:48` proves
    this repository already writes initContainers. A walk has no list of places to keep
    up to date, so it cannot fall behind one.

    What remains enumerated is the set of reference *shapes*, and that set is fixed by
    the Kubernetes API rather than by whoever edits a manifest here. The companion test
    refuses the shapes this cannot read, so the enumeration fails loudly instead of
    quietly returning less than everything.
    """
    pairs: set[tuple[str, str]] = set()
    soft: set[tuple[str, str]] = set()
    whole: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "secretKeyRef":
                    if not isinstance(value, dict):
                        # Not valid Kubernetes, and the one shape where silence would be
                        # the outcome: no pair to compare and no refusal either. This
                        # guard runs before `kubectl apply` gets to reject it, so it has
                        # to be the thing that speaks.
                        whole.append(key)
                        continue
                    if value.get("optional") is True:
                        # A reference the pod starts without. Counting it as a hard
                        # dependency leaves no correct move: declared, `absent_secrets`
                        # blocks the whole apply over a Secret whose absence is harmless
                        # by design; undeclared, the equality below fails. Excluded here
                        # and reported by the companion test, so it is visible rather
                        # than merely uncounted.
                        soft.add((value["name"], value["key"]))
                        continue
                    pairs.add((value["name"], value["key"]))
                    continue
                if key in _WHOLE_SECRET_FORMS:
                    whole.append(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(document)
    return SecretReferences(required=pairs, optional=soft, whole=whole)


_PODS_THE_WORKLOADS_CARRY: Final = frozenset(
    {
        ("deploy/k8s/control-plane.yaml", "Deployment"),
        ("deploy/k8s/schema-migration.yaml", "Job"),
        ("deploy/k8s/tool-gateway.yaml", "Deployment"),
        ("deploy/k8s/model-gateway.yaml", "Deployment"),
    }
)
"""Every (manifest, kind) pair the applier's workloads carry a pod in.

An exact set and not a count. The count this replaces was `>= 3` against a real four,
so it had one case of headroom -- and dropping `Job` from `_KINDS_CARRYING_A_POD`, which
reinstates the whole blindness this file was rewritten to close, left the file passing.
A number cannot distinguish a kind that stopped being recognised from a manifest
somebody added, and this pairing can: adding a workload fails here and says so, which is
the cheap half of a change that should not be silent anyway.
"""


def test_every_kind_that_carries_a_pod_really_does() -> None:
    """A workload manifest document whose kind promises a pod must hold one.

    `_pod_spec` raises rather than returning None for these, so this is the case that
    exercises the raising. It is separate from the secret comparison because that walks
    for a shape and would not notice a malformed template at all.
    """
    module = _platform()
    seen: set[tuple[str, str]] = set()
    for workload in module.WORKLOADS:
        paths = [workload.manifest] + (
            [workload.schema_job] if workload.schema_job else []
        )
        for path in paths:
            for document in yaml.safe_load_all((_ROOT / path).read_text()):
                if document is None:
                    continue
                pod = _pod_spec(document)
                if pod is None:
                    continue
                seen.add((str(path), str(document["kind"])))
                assert pod["containers"], (
                    f"{path}: {document['kind']} has no containers"
                )
    assert seen == _PODS_THE_WORKLOADS_CARRY, (
        "the set of pod-carrying documents across every workload manifest is not the "
        f"set this file expects. Found {sorted(seen)}, expected "
        f"{sorted(_PODS_THE_WORKLOADS_CARRY)}. A kind that disappeared from "
        "_KINDS_CARRYING_A_POD stops being graded here, and a count with headroom "
        "cannot tell that from a manifest somebody added."
    )


def test_no_workload_manifest_reads_a_secret_by_a_form_the_comparison_cannot_read() -> (
    None
):
    """The declaration is `(name, key)` pairs, so a whole-secret read is refused.

    Without this, adding `envFrom: [{secretRef: ...}]` to a container contributes no
    pair, the equality below still holds, and the pre-flight check reports nothing about
    a Secret the pod cannot start without. The refusal is the fail-safe direction: a
    form nobody has taught the comparison is treated as unreadable rather than as
    absent.
    """
    module = _platform()
    for workload in module.WORKLOADS:
        paths = [workload.manifest] + (
            [workload.schema_job] if workload.schema_job else []
        )
        for path in paths:
            for document in yaml.safe_load_all((_ROOT / path).read_text()):
                if document is None:
                    continue
                whole = _secret_references(document).whole
                assert not whole, (
                    f"{path} uses {sorted(set(whole))}, which names a Secret without "
                    "naming a key inside it. The pre-flight check verifies (name, key) "
                    "pairs, so this form contributes nothing to that comparison and "
                    "the pod would reach CreateContainerConfigError, check silent. "
                    "Extend Workload.secrets to carry a whole-secret dependency first."
                )


def test_an_optional_secret_reference_is_not_declared_as_a_hard_dependency() -> None:
    """A `secretKeyRef` marked `optional: true` must stay out of `workload.secrets`.

    The pod starts without it, by design. `absent_secrets` treats every declared pair as
    a precondition and blocks the *whole* apply when one is missing -- "a partial apply
    is worse than none" -- so declaring an optional reference turns a harmless absence
    into a refused deploy. Excluding it from the equality check is therefore only half
    the answer; this is the other half, and without it the exclusion is invisible and a
    future optional reference could be declared with nothing objecting.

    **This currently holds over an empty set**: no manifest under `deploy/k8s/` marks
    any reference optional today, so the loop below runs zero times. That is stated
    rather than hidden, because a vacuous assertion reads like a passing one -- and the
    second assertion is what makes the vacuity itself visible if it ever stops being
    true.
    """
    module = _platform()
    optional_seen: set[tuple[str, str]] = set()
    for workload in module.WORKLOADS:
        paths = [workload.manifest] + (
            [workload.schema_job] if workload.schema_job else []
        )
        for path in paths:
            for document in yaml.safe_load_all((_ROOT / path).read_text()):
                if document is None:
                    continue
                references = _secret_references(document)
                optional_seen |= references.optional
                wrongly_declared = references.optional & set(workload.secrets)
                assert not wrongly_declared, (
                    f"{path} reads {sorted(wrongly_declared)} optional: true, and "
                    f"{workload.component} declares it. absent_secrets would refuse "
                    "the whole apply over a Secret this pod starts fine without."
                )
    assert not optional_seen or all(
        pair not in set(sum((list(w.secrets) for w in module.WORKLOADS), []))
        for pair in optional_seen
    ), sorted(optional_seen)


def test_every_secret_a_manifest_reads_is_declared_on_its_workload() -> None:
    """The declaration is what the pre-flight check reads, so a manifest that added a
    secretKeyRef nobody declared would be applied and then sit in
    CreateContainerConfigError."""
    module = _platform()
    for workload in module.WORKLOADS:
        paths = [workload.manifest] + (
            [workload.schema_job] if workload.schema_job else []
        )
        referenced = set()
        for path in paths:
            for document in yaml.safe_load_all((_ROOT / path).read_text()):
                if document is None:
                    continue
                referenced |= _secret_references(document).required
        assert referenced == set(workload.secrets), workload.component
        if workload.database_secret is not None:
            assert workload.database_secret in workload.secrets, (
                f"{workload.component}'s database_secret is not one of the pairs the "
                "pre-flight check verifies, so the DSN it inspects is not one the "
                "manifest reads"
            )


def test_a_workload_with_no_database_secret_reads_no_database_url() -> None:
    """`database_secret is None` is a claim, and this is what makes it one.

    A workload declaring None skips the rotating-credential check entirely, and it is
    also the workload a reader will leave out of the connection-ceiling arithmetic the
    manifests do against one `max_connections` -- both on the strength of "it opens no
    database". A manifest that grew a `DATABASE_URL` while its entry still said None
    would be a third pool of 60 connections nothing counted.
    """
    module = _platform()
    for workload in module.WORKLOADS:
        if workload.database_secret is not None:
            continue
        for document in yaml.safe_load_all((_ROOT / workload.manifest).read_text()):
            pod = (document or {}).get("spec", {}).get("template", {}).get("spec")
            if pod is None:
                continue
            for container in pod["containers"]:
                for entry in container.get("env", []):
                    assert not entry["name"].endswith("DATABASE_URL"), (
                        f"{workload.component} declares no database_secret and its "
                        f"manifest reads {entry['name']}"
                    )
                assert "envFrom" not in container, (
                    f"{workload.component} declares no database_secret and its "
                    "manifest takes a whole Secret or ConfigMap as environment, so "
                    "what it reads cannot be read off this file"
                )


def test_substitution_replaces_the_placeholder_with_a_real_reference() -> None:
    module = _platform()
    text = f"image: map/control-plane@{module.DIGEST_PLACEHOLDER}\n"
    done = module.substituted(
        text, "map/control-plane", "registry/map/control-plane@sha256:" + "a" * 64
    )
    assert module.DIGEST_PLACEHOLDER not in done
    assert "sha256:" + "a" * 64 in done


def test_substitution_refuses_a_manifest_with_no_placeholder() -> None:
    """A digest hand-edited in once would otherwise be applied for ever, running
    whatever that commit built long after the tree moved on."""
    module = _platform()
    with pytest.raises(RuntimeError, match="not at its placeholder"):
        module.substituted(
            "image: map/control-plane@sha256:" + "b" * 64, "map/control-plane", "x"
        )


def test_substitution_refuses_a_manifest_that_still_holds_a_placeholder() -> None:
    module = _platform()
    text = (
        f"image: map/control-plane@{module.DIGEST_PLACEHOLDER}\n"
        f"image: map/tool-gateway@{module.DIGEST_PLACEHOLDER}\n"
    )
    with pytest.raises(RuntimeError, match="survived substitution"):
        module.substituted(text, "map/control-plane", "registry/x@sha256:" + "a" * 64)


@pytest.mark.parametrize(
    "dsn,master,expected",
    [
        ("postgresql://mapadmin:x@host:5432/managed_agent", "mapadmin", True),
        ("postgresql://map_app:x@host:5432/managed_agent", "mapadmin", False),
        ("postgresql+asyncpg://mapadmin@host/managed_agent", "mapadmin", True),
        ("postgresql://host:5432/managed_agent", "mapadmin", False),
    ],
)
def test_the_master_user_is_recognised_in_every_dsn_spelling(
    dsn: str, master: str, expected: bool
) -> None:
    """The refusal exists because that password rotates every 7 days (measured) and this
    role cannot read the rotated value, so the workload serves and then stops."""
    assert _platform().rotating_credential(dsn, master) is expected


def test_a_deployment_scaled_to_zero_is_a_shortfall() -> None:
    """`kubectl rollout status` prints "successfully rolled out" for zero replicas, so
    the exit code cannot tell a running workload from an absent one."""
    assert _platform().deployment_shortfall({}, 0) is not None


def test_a_deployment_short_of_its_own_replica_count_is_a_shortfall() -> None:
    assert _platform().deployment_shortfall({"availableReplicas": 1}, 2) is not None


def test_a_deployment_at_its_replica_count_is_not() -> None:
    assert _platform().deployment_shortfall({"availableReplicas": 1}, 1) is None


def test_above_one_replica_a_serving_but_incomplete_deployment_is_short() -> None:
    """The state that does not exist at one replica, and does at two.

    At `desired == 1` the outcomes are "nothing is serving" and "everything is". At 2
    there is a third -- one pod up, one not -- where the workload IS serving and the
    redundancy the second replica was added for is absent. Refusing it is the intent: a
    deploy one pod short has not delivered what the manifest asked for, and the one
    moment somebody is reading the output is the moment to say so.

    The message is asserted to carry both counts, because "1 of 2" is the whole content
    of the refusal. A message reading only "not available" would leave the reader unable
    to tell a total outage from a missing replica, which are different pages to open.
    """
    module = _platform()
    assert module.deployment_shortfall({"availableReplicas": 2}, 2) is None
    short = module.deployment_shortfall({"availableReplicas": 1}, 2)
    assert short is not None
    assert "1 of 2" in short, short


def test_every_service_in_a_workload_manifest_is_one_the_applier_checks() -> None:
    """The Service names come out of the manifests, so a new one is covered on commit.

    Compared against a second traversal of the same files rather than against a list: a
    hand-written expectation here would go on covering the old set while passing, which
    is the failure `declared_services` exists to avoid one layer down. The two
    traversals differ -- one asks each workload, the other reads each workload's
    manifest -- so a `declared_services` that stopped finding a document shows up as a
    difference rather than as agreement.
    """
    module = _platform()
    checked = {
        (workload.manifest.name, name)
        for workload in module.WORKLOADS
        for name in module.declared_services(_ROOT, workload)
    }
    present = {
        (workload.manifest.name, document["metadata"]["name"])
        for workload in module.WORKLOADS
        for document in yaml.safe_load_all((_ROOT / workload.manifest).read_text())
        if document and document["kind"] == "Service"
    }
    assert present, "no workload manifest declares a Service, so this grades nothing"
    assert checked == present, (
        f"the applier checks {sorted(checked)} and the manifests declare "
        f"{sorted(present)}. A Service the applier does not check is an address that "
        "can be applied while routing to nothing, with the deploy reporting success."
    )


def test_a_service_with_no_endpoint_slices_at_all_routes_nowhere() -> None:
    """The case the whole function exists for, and it must never answer None.

    `{"items": []}` is what `kubectl get endpointslice -l ...` returns for a Service
    whose selector matches no ready pod -- and a Service like that applies cleanly,
    passes `kubectl rollout status`, and leaves `deployment_shortfall` green, because
    both of those read the Deployment.
    """
    assert _platform().unrouted_service("control-plane", {"items": []}) is not None


def test_a_service_whose_only_endpoint_is_unready_routes_nowhere() -> None:
    """A slice can exist and still carry nothing that traffic may go to."""
    listing = {
        "items": [
            {
                "endpoints": [
                    {"addresses": ["10.0.0.1"], "conditions": {"ready": False}}
                ],
                "ports": [{"name": "http", "port": 8080}],
            }
        ]
    }
    assert _platform().unrouted_service("control-plane", listing) is not None


def test_a_service_with_a_ready_endpoint_and_a_resolved_port_routes() -> None:
    """The positive case, drawn in the shape a real listing takes.

    Without this the file grades only refusals, and a function that answered "routes
    nowhere" unconditionally would satisfy every other case here.
    """
    listing = {
        "items": [
            {
                "endpoints": [
                    {"addresses": ["10.0.0.1"], "conditions": {"ready": True}}
                ],
                "ports": [{"name": "http", "port": 8080}],
            }
        ]
    }
    assert _platform().unrouted_service("control-plane", listing) is None


def test_an_endpoint_with_no_ready_condition_counts_as_ready() -> None:
    """The EndpointSlice API's own rule, and reading it the other way breaks deploys.

    `conditions.ready` is a nullable field and absent means true. Counting a missing
    field as not-ready would refuse every healthy deploy against a server that omits
    it -- a refusal about our parser, arriving as a claim about the cluster.
    """
    listing = {
        "items": [
            {
                "endpoints": [{"addresses": ["10.0.0.1"], "conditions": {}}],
                "ports": [{"name": "http", "port": 8080}],
            }
        ]
    }
    assert _platform().unrouted_service("control-plane", listing) is None


def test_a_ready_endpoint_with_no_resolved_port_routes_nowhere() -> None:
    """The second net, over a `targetPort` that names no container port.

    Weaker than the endpoint half and said so where it is defined: whether Kubernetes
    actually produces a portless slice for an unresolvable named `targetPort` is not
    measured. What this pins is that a listing in that shape is refused rather than read
    as healthy.
    """
    listing = {
        "items": [
            {
                "endpoints": [
                    {"addresses": ["10.0.0.1"], "conditions": {"ready": True}}
                ],
                "ports": [{"name": "http"}],
            }
        ]
    }
    assert _platform().unrouted_service("control-plane", listing) is not None


@pytest.mark.parametrize("listing", [{}, {"items": None}, {"items": "0"}])
def test_a_listing_that_is_not_a_kubectl_listing_raises_instead_of_reading_as_empty(
    listing: dict[str, Any],
) -> None:
    """An empty listing and a malformed one must not become the same answer.

    `kubectl get -o json` always carries an `items` list, even when it is empty, so the
    absence of one means the input is not a listing. Reading that as "no endpoints"
    would report a healthy Service as broken with nothing to say which of the two
    happened -- and this refusal stops a deploy, so the distinction is the difference
    between a real defect and a parser bug wearing its clothes.
    """
    with pytest.raises(RuntimeError):
        _platform().unrouted_service("control-plane", listing)


def test_the_apply_path_actually_calls_the_routing_check() -> None:
    """A pure function nobody calls is a check that does not run.

    `main` is read for the CALL rather than the file for the name, because an import or
    a definition satisfies a name search while nothing runs -- `docs/lessons.md` records
    that exact substitution passing a guard whose behaviour had been removed. The
    module's own docstring lists this as one of the things it does, so an available but
    uncalled function would make that list false.
    """
    body = inspect.getsource(_platform().main)
    assert "declared_services(root, workload)" in body, body[-2000:]
    assert "_routing_refusal(root, service)" in body, body[-2000:]


def test_it_applies_no_manifest_bootstrap_already_applies() -> None:
    """Two appliers reaching for one object is two places a change has to be made, and
    the one that is not run wins the next time somebody runs it."""
    bootstrap = _module("map_bootstrap_for_overlap", "deploy/bootstrap.py")
    # Any twelve digits: `steps()` substitutes what it is handed into the IRSA
    # annotations, and nothing here reads the result.
    built = bootstrap.steps(_ROOT, "210987654321")
    already = {
        Path(token).name
        for step in built
        for token in step.argv
        if token.endswith(".yaml")
    } | {
        # A step applying text this process produced says `-f -`, so its file is not in
        # argv and would silently leave this set -- taking the overlap it is meant to
        # catch with it.
        step.source.name
        for step in built
        if step.source is not None
    }
    for workload in _platform().WORKLOADS:
        assert workload.manifest.name not in already
        if workload.schema_job is not None:
            assert workload.schema_job.name not in already
