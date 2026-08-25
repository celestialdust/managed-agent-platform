"""The cluster objects a Session pod needs before it exists, in the order they arrive.

Tier 1 (local, no cluster). Every assertion here reads YAML off disk or calls a pure
function; nothing creates an object, so nothing here shows that a real API server admits
what `deploy/bootstrap.py` sends. What it does cover is the three ways this particular
job goes wrong silently, each of which was measured against `map-dev` before it was
written down:

* The namespace is not a free variable. Four IAM roles in the account already carry
  `sts:AssumeRoleWithWebIdentity` trust conditions naming one ServiceAccount each, and
  all four name the same namespace. A bootstrap that picks a different one produces
  pods whose projected token no role trusts, and the failure surfaces as an
  `AccessDenied` from the vault rather than as anything about namespaces.

* A ServiceAccount a manifest names and nothing creates is a hard admission failure, not
  a warning: `pods "x" is forbidden: error looking up service account
  default/tool-gateway: serviceaccount "tool-gateway" not found`. So the set of
  ServiceAccounts the manifests name has to equal the set bootstrap creates, and that
  equality has to be checked by something rather than remembered.

* `kubectl rollout status` on a DaemonSet scheduled onto **no** node prints
  "successfully rolled out" and exits 0. Measured. So a bootstrap whose success
  condition is that exit code reports a profile installed on every node while
  installing it on none -- which is what a node that lost its `map.role=session-pod`
  label produces.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_K8S = _ROOT / "deploy" / "k8s"
_MANIFEST = _K8S / "cluster-bootstrap.yaml"

# Measured with one `aws iam get-role` per role, profile `map-dev-agent`, account
# 062677866851 -- the first three on 2026-08-22, the fourth when it was created.
# Each of the four roles trusts exactly one subject, and the namespace component is
# the same in all four -- which is what makes it a fact about the account rather
# than a preference this repository gets to hold. The fourth is the autoscaler's and
# is the reason it is not in kube-system: a kube-system subject makes this set
# disagree with itself, which the next test catches by name.
_TRUSTED_SUBJECTS: Final = {
    "map-cluster-autoscaler": "system:serviceaccount:map-dev:cluster-autoscaler",
    "map-control-plane": "system:serviceaccount:map-dev:control-plane",
    "map-model-gateway": "system:serviceaccount:map-dev:model-gateway",
    "map-tool-gateway": "system:serviceaccount:map-dev:tool-gateway",
}

_ROLE_ANNOTATION: Final = "eks.amazonaws.com/role-arn"


def _bootstrap() -> ModuleType:
    """Import `deploy/bootstrap.py` by path.

    `deploy/` is not an importable package -- it holds manifests, and the one Python
    file in it is a deployment entry point rather than part of the shipped wheel, which
    is why it is not under `src/`. Loaded as `tests/tools/test_plan_waves.py` loads the
    other non-package module in this repository.
    """
    spec = importlib.util.spec_from_file_location(
        "map_bootstrap", _ROOT / "deploy" / "bootstrap.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _documents() -> list[dict[str, Any]]:
    parsed = yaml.safe_load_all(_MANIFEST.read_text())
    return [document for document in parsed if document is not None]


def _of_kind(kind: str) -> list[dict[str, Any]]:
    return [document for document in _documents() if document.get("kind") == kind]


def _service_account_names_in(manifest: Path) -> set[str]:
    """Every `serviceAccountName` any pod spec in this file asks for.

    Walked rather than read at a fixed depth: a Pod carries one at `spec`, a Deployment
    at `spec.template.spec`, and a CronJob two levels below that again. A reader of this
    test does not have to know which shape each manifest is.
    """
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            name = node.get("serviceAccountName")
            if isinstance(name, str):
                found.add(name)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for document in yaml.safe_load_all(manifest.read_text()):
        walk(document)
    return found


def test_the_manifest_parsed_and_has_documents_to_examine() -> None:
    """The positive control. Several assertions below read a discovered collection and a
    file that failed to parse into one would satisfy them by being empty."""
    kinds = [document["kind"] for document in _documents()]
    assert len(kinds) >= 2
    assert "Namespace" in kinds


def test_the_namespace_is_the_one_the_accounts_iam_roles_already_pinned() -> None:
    namespaces = {subject.split(":")[2] for subject in _TRUSTED_SUBJECTS.values()}
    assert len(namespaces) == 1, f"the three roles disagree: {namespaces}"
    declared = [document["metadata"]["name"] for document in _of_kind("Namespace")]
    assert declared == [namespaces.pop()]


def test_every_service_account_is_annotated_with_the_role_that_trusts_it() -> None:
    """Without this annotation the pod-identity webhook injects no role, so the pod
    holds a token nothing exchanges. With the *wrong* one it holds a token that role
    refuses. Both fail at the vault rather than here, which is why it is pinned."""
    accounts = _of_kind("ServiceAccount")
    assert accounts, "no ServiceAccount declared"
    for account in accounts:
        namespace = account["metadata"]["namespace"]
        name = account["metadata"]["name"]
        subject = f"system:serviceaccount:{namespace}:{name}"
        trusting = [
            role for role, trusted in _TRUSTED_SUBJECTS.items() if trusted == subject
        ]
        assert trusting, f"no role in the account trusts {subject}"
        annotation = account["metadata"]["annotations"][_ROLE_ANNOTATION]
        assert annotation.endswith(f":role/{trusting[0]}"), annotation


def test_every_service_account_a_manifest_names_is_created_by_bootstrap() -> None:
    """The guard that keeps this file from rotting. A slice landing a new workload
    manifest names its identity and cannot create it -- nothing under `deploy/k8s/`
    carries a ServiceAccount but this file -- so the day `model-gateway.yaml` or
    `control-plane.yaml` arrives, this fails and names what is missing."""
    created = {document["metadata"]["name"] for document in _of_kind("ServiceAccount")}
    for manifest in sorted(_K8S.glob("*.yaml")):
        if manifest == _MANIFEST:
            continue
        for wanted in _service_account_names_in(manifest):
            assert wanted in created, (
                f"{manifest.name} names {wanted}; nothing creates it"
            )


def test_bootstrap_checks_for_every_file_it_will_apply() -> None:
    """Fail-closed, and it is not hypothetical: two of the four inputs are landing on
    another branch. A bootstrap that applied what it found and skipped what it did not
    would leave a cluster with a DaemonSet and no profile for it to install."""
    module = _bootstrap()
    required = set(module.required_inputs(_ROOT))
    applied = {
        Path(step.argv[index + 1])
        for step in module.steps(_ROOT)
        for index, token in enumerate(step.argv)
        if token == "-f" and step.argv[index + 1] != "-"
    }
    generated = {
        Path(token.split("=", 1)[1])
        for step in module.steps(_ROOT)
        for token in (step.generate or ())
        if token.startswith("--from-file=")
    }
    assert applied | generated == required


def test_missing_inputs_names_each_absent_file() -> None:
    module = _bootstrap()
    empty = Path("/nonexistent-root-for-this-test")
    assert set(module.missing_inputs(empty)) == set(module.required_inputs(empty))


def test_the_profile_config_map_is_generated_and_applied_rather_than_created() -> None:
    """`kubectl create configmap --from-file` exits 1 with `already exists` on a second
    run -- measured -- so a bootstrap that used it could be run exactly once. Generating
    the document and applying it converges instead, and keeps the profile's bytes in one
    file rather than pasting them into a manifest."""
    module = _bootstrap()
    generating = [step for step in module.steps(_ROOT) if step.generate is not None]
    assert len(generating) == 1
    step = generating[0]
    assert step.generate[:2] == ("kubectl", "create")
    assert "--dry-run=client" in step.generate
    assert step.argv[:2] == ("kubectl", "apply")
    assert step.argv[-2:] == ("-f", "-")


def test_bootstrap_applies_no_manifest_that_belongs_to_a_workload() -> None:
    """`session-pod.yaml` is a template the control plane compiles per Session and names
    per Session; applying it here would create one pod called `map-session` that no
    Session owns. `tool-gateway.yaml` is a workload whose image is still
    `PLACEHOLDER_ECR` and whose factory does not exist. Bootstrap is the objects those
    two need, not those two."""
    module = _bootstrap()
    applied = {
        Path(token).name
        for step in module.steps(_ROOT)
        for token in step.argv
        if token.endswith(".yaml")
    }
    assert "session-pod.yaml" not in applied
    assert "tool-gateway.yaml" not in applied


def test_every_namespaced_step_names_the_same_namespace() -> None:
    """Two of the three manifests bootstrap applies are owned by other slices and
    carry no namespace of their own, so theirs can only arrive on the command line. A
    step that forgot it would land in `default`, where the headless Service publishes
    addresses for pods in a namespace nothing dials."""
    module = _bootstrap()
    namespace = _of_kind("Namespace")[0]["metadata"]["name"]
    for step in module.steps(_ROOT):
        for argv in (step.argv, step.generate or ()):
            if "-n" in argv:
                assert argv[argv.index("-n") + 1] == namespace, step.describe


def test_a_daemonset_that_covers_no_node_is_a_shortfall() -> None:
    """Measured against `map-dev`: a DaemonSet whose nodeSelector matches nothing
    reports `desiredNumberScheduled: 0` and `kubectl rollout status` prints
    "successfully rolled out" and exits 0. So the exit code is not the oracle; this
    is."""
    module = _bootstrap()
    assert module.rollout_shortfall({"desiredNumberScheduled": 0, "numberReady": 0})


def test_a_daemonset_short_of_its_own_node_count_is_a_shortfall() -> None:
    module = _bootstrap()
    assert module.rollout_shortfall({"desiredNumberScheduled": 2, "numberReady": 1})


def test_a_daemonset_ready_on_every_node_it_selected_is_not_a_shortfall() -> None:
    """The control for the two above: without it they are satisfied by a function that
    calls everything a shortfall."""
    module = _bootstrap()
    assert (
        module.rollout_shortfall({"desiredNumberScheduled": 2, "numberReady": 2})
        is None
    )
