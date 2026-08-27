"""Every object a manifest depends on is declared here or produced by something named.

`docs/lessons.md` records this guard as buildable and unbuilt, twice. The defect behind
it: applying `deploy/k8s/` the obvious way left the cluster broken, because a DaemonSet
mounted a ConfigMap that only a `kubectl create configmap` in a **header comment**
would have made -- and `kubectl apply -f` does not read header comments. The
generalisation recorded there: if applying a directory the obvious way leaves the
cluster broken, the manifest set is not deployable however well documented it is.

The same class has since cost a second deploy: `ServiceAccount/model-gateway` was
declared in `cluster-bootstrap.yaml` and never applied, and the Deployment sat at 0/2.

So this compares two sets that a human otherwise compares by remembering. What a
manifest depends on is found by **walking each document for the shape of a
dependency** rather than by indexing a path to one, because a path list is a list
somebody has to keep current while the manifests are edited by somebody else.

Three dependencies are legitimately not declared here, and each is admitted only by
importing the constant its producer uses -- never by restating a name. A restated
name is free to diverge from the thing it names, and the way it diverges is that the
guard keeps passing after the producer stops producing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest
import yaml

from managed_agent.adapters.kubernetes.pod_runner import _SECRET_FILES

_ROOT: Final = Path(__file__).resolve().parents[2]
_K8S: Final = _ROOT / "deploy" / "k8s"
_SESSION_POD: Final = _K8S / "session-pod.yaml"


def _module(name: str, relative: str) -> ModuleType:
    """A module under `deploy/` loaded by path.

    `deploy/` is not a package and is not on `sys.path`; the wheel packages
    `src/managed_agent` only. Registered in `sys.modules` before execution because a
    dataclass defined in the module resolves `sys.modules[cls.__module__]` while it is
    being built, and a module absent from that mapping fails with an error naming
    `NoneType` rather than naming the import.
    """
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, relative
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DEPENDENCY_SHAPES: Final[dict[str, tuple[str, str]]] = {
    "configMap": ("ConfigMap", "name"),
    "secret": ("Secret", "secretName"),
    "secretKeyRef": ("Secret", "name"),
    "secretRef": ("Secret", "name"),
    "configMapKeyRef": ("ConfigMap", "name"),
    "configMapRef": ("ConfigMap", "name"),
}
"""The keys by which a pod spec names an object it cannot start without.

Fixed by the Kubernetes API rather than by whoever edits a manifest here, which is what
makes enumerating them different from enumerating the places they appear. A shape the
API adds later is a gap; a *location* the API adds later is not, because the walk below
has no locations in it.
"""


def _dependencies_of(document: object) -> set[tuple[str, str]]:
    """Every `(kind, name)` one document depends on, at whatever depth it sits."""
    found: set[tuple[str, str]] = set()
    if isinstance(document, dict):
        for key, value in document.items():
            shape = _DEPENDENCY_SHAPES.get(key)
            if shape is not None and isinstance(value, dict):
                kind, field = shape
                if field in value:
                    found.add((kind, str(value[field])))
                    continue
            found |= _dependencies_of(value)
    elif isinstance(document, list):
        for item in document:
            found |= _dependencies_of(item)
    return found


def _documents() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(_K8S.glob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text()):
            if document:
                out.append(document)
    return out


def _declared() -> set[tuple[str, str]]:
    return {(d["kind"], d["metadata"]["name"]) for d in _documents()}


def _referenced() -> dict[tuple[str, str], set[str]]:
    """Each dependency, mapped to the manifest filenames that name it."""
    found: dict[tuple[str, str], set[str]] = {}
    for path in sorted(_K8S.glob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text()):
            if not document:
                continue
            for dependency in _dependencies_of(document):
                found.setdefault(dependency, set()).add(path.name)
    return found


def _produced_by_bootstrap() -> set[tuple[str, str]]:
    """What `deploy/bootstrap.py` creates that no manifest here declares.

    The name is imported from the module that creates it, so renaming the constant moves
    both sides at once and renaming only one side fails this file.
    """
    bootstrap = _module("_deploy_bootstrap", "deploy/bootstrap.py")
    name: str = bootstrap.PROFILE_CONFIG_MAP
    return {("ConfigMap", name)}


def _rewritten_per_session() -> set[tuple[str, str]]:
    """The Secret names in `session-pod.yaml` that are templates, not dependencies.

    `pod_runner._create` overwrites `volume["secret"]["secretName"]` with a
    per-Session name before the pod is submitted, so the literal in the manifest is
    never the name of anything that has to exist. Derived from the manifest's own
    volume list intersected with the volumes the runner knows how to fill, so a volume
    the runner does *not* fill stays a real dependency and is reported as one.
    """
    documents = [d for d in yaml.safe_load_all(_SESSION_POD.read_text()) if d]
    templates: set[tuple[str, str]] = set()
    for document in documents:
        if document.get("kind") != "Pod":
            continue
        for volume in document["spec"].get("volumes", []):
            secret = volume.get("secret")
            if secret is None or volume["name"] not in _SECRET_FILES:
                continue
            templates.add(("Secret", str(secret["secretName"])))
    return templates


def _supplied_by_an_operator() -> set[tuple[str, str]]:
    """Secrets the applier will not proceed without, whose values are not in this repo.

    Taken from `platform.WORKLOADS` rather than listed here: `absent_secrets` reads
    that same declaration before applying anything, so a Secret named here and not
    there would be one this file excuses and the applier never checks.
    """
    platform = _module("_deploy_platform", "deploy/platform.py")
    return {
        ("Secret", secret)
        for workload in platform.WORKLOADS
        for secret, _key in workload.secrets
    }


def _generated_by_the_applier() -> set[tuple[str, str]]:
    """ConfigMaps `deploy/platform.py` builds from a file in this repo at apply time.

    The Session-pod manifest reaches the cluster this way rather than as a second
    committed document, so the repository holds one description of a Session pod and the
    control plane mounts the bytes this directory already carries. `main()` creates
    exactly what `WORKLOADS` declares, which is why this reads that declaration instead
    of naming the ConfigMap again: a name excused here and absent there is one nothing
    generates, and the excuse would outlive the producer in silence.
    """
    platform = _module("_deploy_platform", "deploy/platform.py")
    return {
        ("ConfigMap", name)
        for workload in platform.WORKLOADS
        for name, _source in workload.generated_config_maps
    }


def _accounted_for() -> set[tuple[str, str]]:
    return (
        _declared()
        | _produced_by_bootstrap()
        | _rewritten_per_session()
        | _supplied_by_an_operator()
        | _generated_by_the_applier()
    )


def test_the_comparison_has_two_non_empty_sides() -> None:
    """Both sets are non-empty and each exception resolver found something.

    Without this the whole file passes on an empty walk, which is the state it is in on
    any day the shape table stops matching the manifests -- and an empty walk is
    indistinguishable from a clean tree by every other assertion here.
    """
    assert len(_declared()) >= 15, sorted(_declared())
    assert len(_referenced()) >= 5, sorted(_referenced())
    assert _produced_by_bootstrap()
    assert _rewritten_per_session()
    assert _supplied_by_an_operator()
    assert _generated_by_the_applier()


def test_every_object_a_manifest_depends_on_is_declared_or_produced() -> None:
    """A dependency nothing declares and nothing produces wedges the pod that mounts it.

    The failure it prevents is `CreateContainerConfigError` or
    `error looking up service account`, both of which read as a slow start rather than a
    missing input, and both of which have cost a deploy in this project already.
    """
    accounted = _accounted_for()
    referenced = _referenced()
    orphans = {k: v for k, v in referenced.items() if k not in accounted}
    assert not orphans, (
        "these objects are mounted or read by a manifest and are neither declared in "
        f"deploy/k8s/ nor produced by anything this repository runs: "
        f"{ {f'{k[0]}/{k[1]}': sorted(v) for k, v in sorted(orphans.items())} }. "
        "Applying deploy/k8s/ would leave the pod naming one wedged. Either declare "
        "it, or have something in this repository create it and admit it here by "
        "importing that producer's own constant."
    )


def test_the_per_session_secret_names_really_are_rewritten() -> None:
    """The template exception holds only while the runner overwrites the name.

    Asserted against the runner's source because the alternative is trusting a
    docstring: if the rewrite is ever removed, `session-pod.yaml`'s literal Secret
    names become real dependencies of a Secret nothing creates, and the exception
    above would go on excusing them.
    """
    source = (_ROOT / "src/managed_agent/adapters/kubernetes/pod_runner.py").read_text()
    assert 'volume["secret"]["secretName"] = _secret_name(' in source, (
        "pod_runner no longer rewrites the manifest's secretName per Session, so the "
        "literal names in session-pod.yaml are dependencies rather than templates and "
        "_rewritten_per_session() is excusing objects that must now exist"
    )
    assert _rewritten_per_session() == {
        ("Secret", "map-session-compiled-config"),
        ("Secret", "map-session-requirements"),
        ("Secret", "map-session-shim-token"),
    }, sorted(_rewritten_per_session())


_POD_TEMPLATE_KINDS: Final = frozenset(
    {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "Job"}
)
"""Kinds that create pods from a `spec.template`.

`Pod` is deliberately absent and handled on its own below: its labels are at
`metadata.labels`, not under a template. `CronJob` is absent because none exists here
and its template sits one level deeper (`spec.jobTemplate.spec.template`) -- the walk
would read the wrong place rather than nothing, so it is refused by the completeness
check rather than quietly skipped.
"""


def _pod_label_sets() -> dict[str, dict[str, str]]:
    """Every label set a pod under deploy/k8s/ is created with, by where it came from.

    The template's labels and NOT the object's own. They are different places and they
    are free to disagree: a Deployment labelled `map.component: x` whose pod template is
    not creates pods carrying no such label, and a Service selecting on it routes to
    nothing while every document in the file reads correctly.

    Raises on a `CronJob`, whose template is one level deeper than every other kind. A
    silent skip there would be a pod nothing in this file grades -- the same
    looks-like-coverage shape the rest of this module is about.
    """
    found: dict[str, dict[str, str]] = {}
    for path in sorted(_K8S.glob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text()):
            if not document:
                continue
            kind = document["kind"]
            if kind == "CronJob":
                raise AssertionError(
                    f"{path.name} declares a CronJob, whose pod template is at "
                    "spec.jobTemplate.spec.template. This walk would read the wrong "
                    "place; teach it that path before adding one."
                )
            where = f"{path.name}:{kind}/{document['metadata']['name']}"
            if kind == "Pod":
                found[where] = dict(document["metadata"].get("labels") or {})
            elif kind in _POD_TEMPLATE_KINDS:
                template = document["spec"]["template"]
                found[where] = dict(template.get("metadata", {}).get("labels") or {})
    return found


def _label_selectors() -> dict[str, dict[str, str]]:
    """Every selector under deploy/k8s/ that has to match a pod, by where it came from.

    Two kinds, and the two spell it differently. A Service's `spec.selector` is a flat
    map of equalities. A PodDisruptionBudget's is a LabelSelector, whose equality half
    is `spec.selector.matchLabels`. A budget written with `matchExpressions` instead is
    refused rather than skipped: this comparison cannot express a set-based selector,
    and a selector it silently ignored would be one nothing checks.

    A Deployment's own `spec.selector` is not here. It is compared against its own
    template by the Deployment's manifest test, where the two live in one document; the
    interesting case in THIS file is the selector whose pods are somewhere else
    entirely -- `session-shim-service.yaml` selects pods declared in `session-pod.yaml`.
    """
    found: dict[str, dict[str, str]] = {}
    for path in sorted(_K8S.glob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text()):
            if not document:
                continue
            kind = document["kind"]
            where = f"{path.name}:{kind}/{document['metadata']['name']}"
            if kind == "Service":
                found[where] = dict(document["spec"].get("selector") or {})
            elif kind == "PodDisruptionBudget":
                selector = document["spec"]["selector"]
                assert "matchExpressions" not in selector, (
                    f"{where} selects with matchExpressions, which this comparison "
                    "cannot read. Ignoring it would leave a budget nothing grades."
                )
                found[where] = dict(selector.get("matchLabels") or {})
    return found


def test_every_selector_that_must_match_a_pod_matches_one() -> None:
    """A Service or a budget selecting labels no pod carries is silently inert.

    Both are valid YAML the API server accepts without complaint. A Service whose
    selector matches nothing publishes a name that resolves and then connects to
    nothing, which is not hypothetical here: a Session pod at `2/2 Running` with its
    shim answering the kubelet failed every Turn as unreachable because the Service
    publishing its address had been applied by nothing, and every probe in the chain
    reported fine. A budget whose selector matches nothing reports `expectedPods: 0` and
    permits every eviction -- protecting nothing while reading as protection.

    A subset comparison and not equality, because that is what Kubernetes does: a
    selector may name fewer labels than the pod carries. Compared across the whole
    directory rather than per file, because a selector and the pods it selects need not
    share a manifest.

    An EMPTY selector is refused separately. `spec.selector: {}` on a Service selects
    every pod in the namespace, and it is a subset of every label set, so it would pass
    the loop below while being the most wrong answer available.

    Floors, so the sweep cannot pass on an empty walk: 7 pod label sets and 5 selectors
    were found when this was written -- one Pod, one DaemonSet, one Job, four
    Deployments; four Services and one PodDisruptionBudget.
    """
    pods = _pod_label_sets()
    selectors = _label_selectors()
    assert len(pods) >= 6, sorted(pods)
    assert len(selectors) >= 4, sorted(selectors)

    for where, selector in sorted(selectors.items()):
        assert selector, (
            f"{where} selects on no label at all, which selects every pod in the "
            "namespace rather than none"
        )
        matched = [
            pod
            for pod, labels in pods.items()
            if all(labels.get(key) == value for key, value in selector.items())
        ]
        assert matched, (
            f"{where} selects {selector} and no pod declared under deploy/k8s/ carries "
            f"those labels. Pod label sets in the tree: {pods}"
        )


@pytest.mark.parametrize(
    "shape",
    sorted(_DEPENDENCY_SHAPES),
)
def test_each_dependency_shape_is_one_the_walk_can_read(shape: str) -> None:
    """Every shape in the table yields a dependency when it appears at any depth.

    A table entry with the wrong field name reads as "this shape is covered" and
    contributes nothing, which is the failure the walk exists to avoid. Driven through a
    document nested four levels deep, because top-level is the one depth the broken
    version of this walk got right.
    """
    kind, field = _DEPENDENCY_SHAPES[shape]
    buried = {
        "spec": {"template": {"spec": {"containers": [{shape: {field: "probe-name"}}]}}}
    }
    assert _dependencies_of(buried) == {(kind, "probe-name")}


def _manifests_something_applies() -> set[Path]:
    """Every file under `deploy/k8s/` that an automated path puts into the cluster.

    Read off the two appliers' own public surfaces rather than listed here, for the
    reason `_produced_by_bootstrap` gives: a list restated in a test is free to diverge
    from the thing it names, and the way it diverges is that the guard keeps passing
    after the applier stops applying.

    A generated ConfigMap counts. `session-pod.yaml` is never `kubectl apply`d -- it is
    the body of a ConfigMap built from the file at apply time -- but the question here
    is whether editing the file changes the cluster, and for that one it does.
    """
    bootstrap = _module("_deploy_bootstrap", "deploy/bootstrap.py")
    platform = _module("_deploy_platform", "deploy/platform.py")
    applied: set[Path] = {(_ROOT / path).resolve() for path in bootstrap.APPLIES}
    for workload in platform.WORKLOADS:
        applied.add((_ROOT / workload.manifest).resolve())
        if workload.schema_job is not None:
            applied.add((_ROOT / workload.schema_job).resolve())
        applied.update((_ROOT / path).resolve() for path in workload.companions)
        applied.update(
            (_ROOT / source).resolve() for _, source in workload.generated_config_maps
        )
    return applied


def test_every_manifest_in_this_tree_is_applied_by_something() -> None:
    """A manifest no applier names reaches the cluster only when somebody remembers.

    The third instance of the class this file's header describes, and the first one
    where the missing object is a security boundary rather than a mount. The two
    recorded above were a ConfigMap made only by a header comment and a ServiceAccount
    declared and never applied; both announced themselves, because a pod stuck at
    ContainerCreating and a Deployment at 0/2 are things somebody notices.

    **This shape announces nothing.** `deploy/k8s/network-policies.yaml` declares a
    default-deny and six exceptions; absent from the cluster, every pod talks to every
    pod and every manifest still reads as though it could not. Nothing is unready,
    nothing restarts, and the tests that grade the file's *contents* stay green --
    which is exactly the state this repository was in: the CNI enforcing, the policy
    set written, and the namespace holding none of it.

    So the guard is about the applier, not the cluster. `test_network_policies.py`'s
    `test_every_policy_this_repo_declares_is_in_the_cluster` says whether the set is
    there right now and needs a cluster to ask; this says whether anything will put it
    back, and it runs offline on every change.
    """
    declared = {path.resolve() for path in _K8S.glob("*.yaml")}
    applied = _manifests_something_applies()
    orphaned = sorted(path.name for path in declared - applied)
    assert not orphaned, (
        f"{orphaned} sit in deploy/k8s/ and no applier names them, so editing one "
        "changes nothing until a person runs kubectl by hand. Name it in "
        "deploy/bootstrap.py's APPLIES if it stands the cluster up once, or in a "
        "Workload's companions in deploy/platform.py if it should be re-asserted every "
        "release."
    )


_MAKEFILE: Final = _ROOT / "Makefile"


def _prerequisites(target: str) -> list[str]:
    """The prerequisites of one Makefile target, in the order make runs them."""
    for line in _MAKEFILE.read_text().splitlines():
        if line.startswith(f"{target}:"):
            return line.split(":", 1)[1].split("##")[0].split()
    pytest.fail(f"the Makefile declares no target {target!r}")


def _recipe(target: str) -> str:
    """Every recipe line of one Makefile target, joined."""
    lines: list[str] = []
    collecting = False
    for line in _MAKEFILE.read_text().splitlines():
        if not collecting:
            collecting = line.startswith(f"{target}:")
            continue
        if line.startswith("\t"):
            lines.append(line)
        elif line.strip():
            break
    return "\n".join(lines)


def test_a_deploy_checks_its_inputs_before_it_spends_anything() -> None:
    """An input the last step needs is refused at the first, not after the push.

    On 2026-08-26 `make deploy` built the platform image, pushed it to three ECR
    repositories, created the cluster objects, rolled the control plane and rolled the
    Tool Gateway -- and then refused, because `MAP_FOUNDRY_RESOURCE` was unset and the
    Model Gateway's routing table needs it. Nothing was wrong with the refusal: a
    Foundry resource names one company's Azure account, so it cannot be defaulted or
    derived, and a wrong one would send a tenant's prompts to somebody else's endpoint.
    What was wrong was that it came last, leaving two workloads on the new image and one
    on the old -- a half-deployed platform, which is the state a deploy exists to avoid.

    So the assertion is about ORDER, not about the variable. `deploy-inputs` has to run
    before `image`, because `image` is the first step that spends anything: everything
    after it is a push or a rollout, and every one of those is harder to undo than a
    refusal.

    The variable is read off `deploy/platform.py`'s own constant rather than spelled
    here, for the reason this file's header gives about restated names: spelled twice,
    a rename leaves the Makefile checking a variable nothing reads any more while this
    case goes on passing, which is precisely the failure both are here to prevent.

    Not a test of `make`. It reads the Makefile as a document, so it runs offline and on
    every change, including on a machine with no cluster and no Docker.
    """
    platform = _module("_deploy_platform", "deploy/platform.py")
    steps = _prerequisites("deploy")

    assert "deploy-inputs" in steps, (
        f"the deploy target's prerequisites are {steps}, and none of them checks the "
        "inputs. A deploy that discovers a missing input after the image is pushed "
        "leaves the platform half-rolled."
    )
    assert "image" in steps, (
        f"the deploy target's prerequisites are {steps} and 'image' is not among them, "
        "so this case can no longer tell where the spending starts -- re-point it at "
        "whatever the first irreversible step has become."
    )
    assert steps.index("deploy-inputs") < steps.index("image"), (
        f"deploy runs {steps}, which checks its inputs after 'image' has already built "
        "and pushed. Move deploy-inputs in front of it: a refusal is free until "
        "something has been pushed or rolled."
    )
    assert platform.FOUNDRY_RESOURCE_VAR in _recipe("deploy-inputs"), (
        f"deploy-inputs does not check {platform.FOUNDRY_RESOURCE_VAR}, which "
        "deploy/platform.py's with_foundry refuses the Model Gateway without -- at the "
        "last step of the chain. It is knowable at the first."
    )
