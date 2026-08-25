"""The Model Gateway's manifest, graded on the one thing it exists to prevent.

Tier 1 (local, no cluster). This is the only component that holds a provider
credential, so every assertion below is about a way that credential could end up
written down: an env var, a mounted Secret, a value pasted into the routing table,
or a container permissive enough that reading the process's memory is somebody's
option.

NOT PROVEN by anything here: nothing validates these documents against the real
Kubernetes schemas. `kubeconform` is not installed and is not an attested tool for this
repo, so a tactic naming it would be a check that silently never runs; a field name
misspelled in a way that still parses as YAML reaches `kubectl apply` uncaught.

Also not proven: that the vault entries `routing.json` names are readable by the
role this pod assumes. `deploy/iam/map-model-gateway.json` grants
`secretsmanager:GetSecretValue` on `map/dev/providers/*` and nothing else, and that file
is Terraform's under ADR-021 rather than this slice's. The paths test below pins every
provider entry inside that prefix, and no other entry is named from anywhere in this
manifest -- the session-token signing key, which used to be the exception, arrives from
a Kubernetes Secret instead (ADR-023).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import httpx
import pytest
import yaml

from managed_agent import composition
from managed_agent.control.pod_config.model_binding import MODEL_PROVIDER_WIRE
from managed_agent.core.ids import SessionId
from managed_agent.gateway.model.anthropic_table import ANTHROPIC_VERSION
from managed_agent.gateway.model.credential_broker import ProviderCredentialBroker
from managed_agent.gateway.model.router import (
    MAX_REQUEST_BODY_BYTES,
    UpstreamWire,
    create_model_gateway_app,
    routing_table_from_json,
)

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "deploy" / "k8s" / "model-gateway.yaml"
_BOOTSTRAP = _ROOT / "deploy" / "k8s" / "cluster-bootstrap.yaml"
_POLICY = _ROOT / "deploy" / "iam" / "map-model-gateway.json"

_READABLE_PREFIX = "map/dev/providers/"


PLACEHOLDER: Final = "sha256:" + "0" * 64

IN_CLUSTER_BASE_URL: Final = "http://model-gateway.map-dev.svc.cluster.local/v1"
"""The address every compiled `config.toml` gives the Agent Runtime as its `base_url`.

Written out here so the composition below has something to be equal *to*. It is not the
source of the value -- an Environment record carries it and an operator registers it --
which is exactly why it is worth pinning: the manifest is the only thing that can make
this string resolve, and nothing else in the tree compares the two.
"""


def _platform() -> ModuleType:
    """`deploy/platform.py`, loaded by path.

    It is a script rather than a package member, so there is no import for it. Loaded
    here rather than reimplemented because this file's subject is whether the applier
    and the manifest agree, and a second reading of the manifest could only agree with
    itself.
    """
    spec = importlib.util.spec_from_file_location(
        "map_platform_for_model_gateway", _ROOT / "deploy" / "platform.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_LOOKS_LIKE_A_SECRET = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9._-]{16,}"),
)


def _documents() -> list[dict[str, Any]]:
    loaded = [d for d in yaml.safe_load_all(_MANIFEST.read_text()) if d is not None]
    assert loaded, f"{_MANIFEST} parsed into no documents at all"
    return loaded


def _of_kind(kind: str) -> dict[str, Any]:
    found = [d for d in _documents() if d["kind"] == kind]
    assert len(found) == 1, f"expected exactly one {kind}, found {len(found)}"
    return found[0]


def _container() -> dict[str, Any]:
    containers = _of_kind("Deployment")["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1, "one container, or the assertions below grade one of N"
    only = containers[0]
    assert isinstance(only, dict)
    return only


def _routing_document() -> bytes:
    document = _of_kind("ConfigMap")["data"]["routing.json"]
    assert isinstance(document, str), "routing.json is not text in the ConfigMap"
    return document.encode()


def _config_map(name: str) -> dict[str, Any]:
    """The ConfigMap document a volume names, or a failure saying it is not here.

    A ConfigMap holding this service's own configuration belongs in this file; the one
    Secret the pod mounts is deliberately created outside the repo. So a `configMap`
    volume naming a document this file does not hold is a pod that never leaves
    ContainerCreating, and there is nowhere else it could legitimately have come from.
    """
    found = [
        d
        for d in _documents()
        if d["kind"] == "ConfigMap" and d["metadata"]["name"] == name
    ]
    assert len(found) == 1, (
        f"a volume mounts ConfigMap {name!r} and this file holds {len(found)} "
        "documents by that name, so the pod would wait for one that never appears"
    )
    return found[0]


def _volume_mounts() -> list[dict[str, Any]]:
    return list(_container()["volumeMounts"])


def _rebuild_the_mounted_filesystem(root: Path) -> None:
    """Recreate, under `root`, the files this pod's volumes put in its container.

    The point is to stop the boot test below from supplying the value it is grading. A
    test cannot write to `/etc/map`, so the alternative was to override
    `MAP_ROUTING_TABLE_PATH` with a temp path -- which made the manifest's own
    declaration of that one variable invisible to the only guard that reads it, and let
    five start-up-fatal edits through. Here the world is adapted to the manifest
    instead: every mount is created where the manifest says it is, each ConfigMap's keys
    are written as files inside it, and the env values keep the manifest's own
    directories and filenames (rebased only at the leading `/`).

    A `secret` volume gets its directory and no contents. Its one member is the serving
    certificate, which is created outside this repo, and nothing the factory reads is in
    it -- uvicorn opens those files, and uvicorn is not started here.
    """
    volumes = {
        volume["name"]: volume
        for volume in _of_kind("Deployment")["spec"]["template"]["spec"]["volumes"]
    }
    assert _volume_mounts(), "no volumeMounts, so nothing below is reconstructed"
    for mount in _volume_mounts():
        volume = volumes.get(mount["name"])
        assert volume is not None, (
            f"volumeMount {mount['name']!r} names no volume this pod declares"
        )
        directory = root / mount["mountPath"].lstrip("/")
        directory.mkdir(parents=True, exist_ok=True)
        if "configMap" not in volume:
            continue
        for filename, content in _config_map(volume["configMap"]["name"])[
            "data"
        ].items():
            (directory / filename).write_text(content)


def _environment_rebased_under(root: Path) -> dict[str, str]:
    """This container's env block, with absolute paths moved under a real directory.

    Names and values are the manifest's own; only the leading `/` moves. Every coupling
    the boot test depends on survives that -- which variable, which directory, which
    filename -- so a value pointing outside a declared mountPath still points outside
    the reconstructed one. A value that is not a path does not start with `/` and
    travels untouched, and a variable whose value comes from a Secret carries the
    stand-in `_container_env` supplies -- what the boot test needs of it is that it is
    present, not what it says.
    """
    return {
        name: (str(root / value.lstrip("/")) if value.startswith("/") else value)
        for name, value in _container_env().items()
    }


def _absolute_paths_this_container_names() -> list[tuple[str, str]]:
    """Every absolute filesystem path in this container's env block and its args."""
    found = [
        (f"env {name}", value)
        for name, value in _container_env().items()
        if value.startswith("/")
    ]
    for arg in _container()["args"]:
        flag, separator, value = arg.partition("=")
        if separator and value.startswith("/"):
            found.append((f"arg {flag}", value))
    return found


def test_the_file_holds_the_three_objects_this_service_is() -> None:
    """Its identity is not among them: a ServiceAccount a workload manifest declares is
    one `cluster-bootstrap.yaml` does not create, and admission rejects the pod outright
    rather than warning."""
    assert [d["kind"] for d in _documents()] == ["ConfigMap", "Deployment", "Service"]


def test_the_identity_this_pod_assumes_carries_this_services_own_iam_role() -> None:
    """Which account, not merely one that exists.

    Set membership was the wrong grade and it hid a real drift: this file names two
    ServiceAccounts, so `serviceAccountName: model-gateway` could become `tool-gateway`
    and stay a member. Nothing else in the tree would object either -- the bootstrap's
    own test already asks whether every account a manifest names is created, so this one
    asking the same question a second time was the weaker of two copies.

    The drift it let through runs in both directions at once. `map-tool-gateway`'s
    policy grants no `map/dev/providers/*`, so the Gateway loses the one credential it
    exists to hold and fails at the first Turn with an AccessDenied that names a vault
    path rather than this line; and it grants `s3:DeleteObject` on the evidence bucket,
    which the Model Gateway has no business holding at all.

    Graded on the annotation rather than on the name, because the annotation is the fact
    that decides which role the projected token is exchanged for. The account name is
    just what points at it.
    """
    roles = {
        d["metadata"]["name"]: d["metadata"]["annotations"][
            "eks.amazonaws.com/role-arn"
        ]
        for d in yaml.safe_load_all(_BOOTSTRAP.read_text())
        if d is not None and d.get("kind") == "ServiceAccount"
    }
    pod = _of_kind("Deployment")["spec"]["template"]["spec"]
    account = pod["serviceAccountName"]

    assert account in roles, sorted(roles)
    assert roles[account].endswith("/map-model-gateway"), roles[account]


def test_every_document_sits_in_the_namespace_the_accounts_iam_roles_pinned() -> None:
    """A pod anywhere else holds a projected token no role will exchange, and the
    failure surfaces as an AccessDenied from the vault rather than about namespaces."""
    namespaces = {
        d["metadata"]["name"]
        for d in yaml.safe_load_all(_BOOTSTRAP.read_text())
        if d is not None and d.get("kind") == "Namespace"
    }

    assert len(namespaces) == 1, namespaces
    for document in _documents():
        assert document["metadata"]["namespace"] == next(iter(namespaces))


# --- the credential's absence ------------------------------------------------------


def test_the_only_secret_this_container_reaches_for_is_the_one_it_verifies_with() -> (
    None
):
    """A literal, or a `secretKeyRef` naming a pair the applier already checks.

    This used to be `set(entry) == {"name", "value"}` for every entry -- no Secret
    referenced by any route -- on the reasoning that an env var is readable by anything
    in the container for the process's whole life, which is strictly worse than a value
    fetched per TTL and never stored. That reasoning is right about the thing it was
    written for and wrong as a blanket rule, and the difference is what this narrower
    version encodes.

    What it was written for is the **provider credential**: a bearer that reaches an
    external provider, is selected per tenant, and costs money. Nothing about that
    changes -- it is still fetched from the vault per TTL and still never appears here,
    which is what the vault-shape scan below and the routing-table checks above hold.

    What it wrongly also forbade is the **session-token signing key**, a symmetric HMAC
    key this service only ever *verifies* with. It authorizes no outward call and buys
    nothing; the most a reader of it can do is mint a token naming a Session, which the
    control plane holding the same key already does. And it has to be byte-identical to
    the key the control plane mints with, because a verifier keyed from anywhere else
    answers 401 to every model call in the fleet -- which is exactly what this service
    did until 2026-08-23, when the key it read came from a vault entry nothing minted
    against. Reading it from the same Kubernetes Secret the control plane and the Tool
    Gateway read is what makes that drift impossible rather than merely unlikely.

    The allowlist is `deploy/platform.py`'s own `secrets` declaration rather than a name
    written here, which stops this from becoming a second place the pair is spelled. It
    also means the pair is covered by `absent_secrets`, so a Secret or key missing from
    the cluster fails the apply rather than the first request.
    """
    container = _container()
    env = container["env"]
    module = _platform()
    workload = next(w for w in module.WORKLOADS if w.component == "model-gateway")
    permitted = {(secret, key) for secret, key in workload.secrets}

    assert env, "no environment entries found, so this assertion grades nothing"
    assert "envFrom" not in container, (
        "envFrom pulls in every key of a Secret at once, so nothing here could say "
        "which values the container ends up holding"
    )
    referenced = set()
    for entry in env:
        if "value" in entry:
            assert set(entry) == {"name", "value"}, entry
            assert isinstance(entry["value"], str), entry["name"]
            continue
        assert set(entry) == {"name", "valueFrom"}, entry
        assert set(entry["valueFrom"]) == {"secretKeyRef"}, entry
        ref = entry["valueFrom"]["secretKeyRef"]
        assert set(ref) == {"name", "key"}, ref
        referenced.add((str(ref["name"]), str(ref["key"])))

    assert referenced, (
        "no secretKeyRef found, so the allowlist below grades nothing -- if the last "
        "one went away, restore the stricter all-literals assertion instead"
    )
    assert referenced <= permitted, (
        f"{sorted(referenced - permitted)} is read from a Secret that "
        "deploy/platform.py does not check exists before applying this workload"
    )


def test_this_deployment_mounts_no_secret_at_all() -> None:
    """No Secret arrives as a *file*, which is a separate route from the env block.

    The one Secret this container reads arrives as an environment variable and is
    accounted for by the test above. A mounted Secret would be a second route with a
    second set of names, invisible to that accounting -- so this stays absolute even
    though the env assertion no longer is.

    Stronger than the assertion this replaces, which allowed exactly one -- a serving
    certificate for the TLS listener the manifest's header explains is gone. That Secret
    also existed in no cluster and is created by nothing in this repository, so a volume
    naming it was a pod that would have sat in ContainerCreating; an allowlist of one
    name could not see that, and "no Secret" does not need to.
    """
    pod = _of_kind("Deployment")["spec"]["template"]["spec"]

    assert [v for v in pod["volumes"] if "secret" in v] == []


def test_the_routing_table_parses_into_the_table_the_service_reads() -> None:
    """Parsed by the real parser: a ConfigMap this build would refuse at start-up is a
    Deployment that crash-loops with the reason only in its own logs."""
    table = routing_table_from_json(_routing_document())

    assert table.declared_models(), "an empty table would answer 404 for every model"
    shapes = {table.entry_for(m).wire for m in table.declared_models()}
    # Every shape declared here must be one a handler exists for. This read
    # `RESPONSES in shapes` while that was the only registered wire; naming a
    # particular wire stopped being the property once MAP-23 added the Anthropic
    # translator, and it made deleting the last `responses` model look like a
    # regression when it is a routing choice. A wire declared with no handler
    # behind it is the real defect -- the gateway accepts the model and fails
    # mid-stream. The served set is asserted exactly, against the built app, in
    # `test_this_manifests_own_environment_boots_the_factory_it_names`.
    assert shapes <= {UpstreamWire.RESPONSES, UpstreamWire.ANTHROPIC_MESSAGES}, (
        f"{sorted(w.value for w in shapes)} declares a wire no handler serves, so "
        "the gateway accepts that model and then fails mid-stream"
    )


def test_every_declared_credential_is_a_vault_name_and_never_a_value() -> None:
    table = routing_table_from_json(_routing_document())

    for model in sorted(table.declared_models()):
        name = table.entry_for(model).credential_name
        assert name.startswith(_READABLE_PREFIX), (
            f"{model} names {name}, outside the prefix "
            f"{_POLICY.relative_to(_ROOT)} grants this role"
        )
        assert "\n" not in name and len(name) < 128, name


def test_the_policy_this_test_pins_against_still_grants_that_prefix() -> None:
    """The control for the test above: without this, renaming the prefix in the policy
    would leave that assertion passing against a prefix nothing grants."""
    granted = json.loads(_POLICY.read_text())["Statement"][0]["Resource"]

    assert granted.endswith(f":secret:{_READABLE_PREFIX}*"), granted


def test_nothing_anywhere_in_this_manifest_looks_like_a_credential() -> None:
    """A broad sweep rather than a field list, because the point is that no *line* of
    this file is a secret however somebody later chooses to add one."""
    text = _MANIFEST.read_text()

    for pattern in _LOOKS_LIKE_A_SECRET:
        assert not pattern.search(text), pattern.pattern


# --- the container ----------------------------------------------------------------


def test_the_pod_runs_as_a_non_root_user() -> None:
    security = _of_kind("Deployment")["spec"]["template"]["spec"]["securityContext"]

    assert security["runAsNonRoot"] is True
    assert security["runAsUser"] != 0


def test_the_container_cannot_escalate_write_its_root_or_keep_a_capability() -> None:
    security = _container()["securityContext"]

    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"]["drop"] == ["ALL"]
    assert "privileged" not in security


def test_the_container_asks_for_memory_and_names_a_limit_to_be_killed_at() -> None:
    """A pod with no `resources` block is BestEffort, and BestEffort at `replicas: 2`
    means the kubelet reclaims memory from whatever else is on the node rather than from
    the process that grew. The limit is the number this container dies at instead."""
    resources = _container()["resources"]

    assert resources["requests"]["memory"]
    assert resources["requests"]["cpu"]
    assert resources["limits"]["memory"]
    assert "cpu" not in resources["limits"], (
        "a CPU limit throttles a latency-sensitive relay rather than shedding load"
    )


_MEBIBYTE = 1024 * 1024


def _quantity_in_bytes(quantity: str) -> int:
    """A Kubernetes memory quantity as a number. Mi and Gi only: all this file uses."""
    suffixes = {"Mi": _MEBIBYTE, "Gi": _MEBIBYTE * 1024}
    for suffix, multiplier in suffixes.items():
        if quantity.endswith(suffix):
            return int(quantity.removesuffix(suffix)) * multiplier
    raise AssertionError(f"memory quantity {quantity!r} uses a suffix this cannot read")


def test_the_body_cap_and_this_pods_memory_are_one_decision() -> None:
    """The three numbers the `resources` comment says it derived, and their coupling.

    All three survived mutation before this test existed: the cap 8 MiB -> 8 GiB and the
    limit 512Mi -> 1Mi each passed the whole suite. Both cap tests in
    `tests/gateway/model/` size their request body *from the constant they grade*, so
    they prove a cap is enforced at whatever value it holds and never that it is 8 MiB
    -- and the comment at `model-gateway.yaml` derives this container's memory from that
    exact number, which is a derivation with nothing holding either end.

    Graded two ways on purpose. The literals, so changing a number somebody decided
    is a deliberate edit to this test rather than a silent one. And the inequalities
    comment's own reasoning rests on -- a memory request that cannot hold even one body
    is not sized for bodies, and a limit at or below the request is not a headroom -- so
    the two files cannot drift apart even if both literals move together.
    """
    resources = _container()["resources"]
    request = _quantity_in_bytes(resources["requests"]["memory"])
    limit = _quantity_in_bytes(resources["limits"]["memory"])

    assert MAX_REQUEST_BODY_BYTES == 8 * _MEBIBYTE, (
        "the pod's memory is derived from this number; moving it moves the manifest"
    )
    assert resources["requests"]["memory"] == "256Mi"
    assert resources["limits"]["memory"] == "512Mi"
    assert request > MAX_REQUEST_BODY_BYTES, (
        f"a {request}-byte request cannot hold one {MAX_REQUEST_BODY_BYTES}-byte body, "
        "so it was not sized from the cap the comment says it was"
    )
    assert limit > request, "the limit is the headroom above the request, not below it"


def test_this_service_is_two_pods_so_losing_one_is_not_losing_the_gateway() -> None:
    """`replicas: 2` survived being set to `0` -- a Deployment that serves nothing.

    Two and not one because every Turn of every Session goes through this Service, so at
    one replica a rolling update or a node drain takes the whole Model Gateway down.
    `tool-gateway.yaml` is two for the same reason. And the number is load-bearing for
    the `resources` comment in this file, which reasons about what the kubelet does "at
    replicas: 2".
    """
    assert _of_kind("Deployment")["spec"]["replicas"] == 2


def test_nothing_this_container_mounts_is_writable() -> None:
    """`readOnly: true` on the routing mount survived being set to `false`.

    `readOnlyRootFilesystem: true` is already set, which makes the mounts the only
    places in this container a process could write at all -- so a mount left writable
    is not one writable path among many, it is the only one. Neither of the two holds
    anything this process writes: a ConfigMap and a serving certificate.

    Asserted over every mount rather than the two by name, so a third added later is
    covered without this test being edited.
    """
    assert _volume_mounts(), "no volumeMounts, so this asserts nothing"
    writable = [m["name"] for m in _volume_mounts() if m.get("readOnly") is not True]

    assert not writable, (
        f"these mounts are writable in a container whose root is not: {writable}"
    )


def test_the_service_port_is_the_default_port_for_the_scheme_it_serves() -> None:
    """`port: 443` survived being set to `80`, and the two are not interchangeable here.

    This container terminates TLS itself, so the Service in front of it carries HTTPS. A
    base URL an operator writes for the Agent Runtime names a host and a path and
    ordinarily no port -- `https://model-gateway.map-dev.svc.cluster.local/v1` -- and a
    client then connects to 443. At `port: 80` that URL reaches nothing, and a plain
    `http://` one reaches a TLS listener with a plaintext request.

    Read off the probes' own `scheme` rather than written down as 443, so this is the
    coupling and not a third copy of the number. `tool-gateway.yaml` is `port: 80` under
    the same rule: its probes name no scheme, so they are HTTP.
    """
    default_for = {"HTTPS": 443, "HTTP": 80}
    schemes = {
        _container()[probe]["httpGet"].get("scheme", "HTTP")
        for probe in ("readinessProbe", "livenessProbe")
    }

    assert len(schemes) == 1, f"the two probes disagree about the scheme: {schemes}"
    expected = default_for[schemes.pop()]
    assert [p["port"] for p in _of_kind("Service")["spec"]["ports"]] == [expected]


def test_the_pod_takes_none_of_the_host() -> None:
    pod = _of_kind("Deployment")["spec"]["template"]["spec"]

    assert "hostNetwork" not in pod
    assert "hostPID" not in pod


@pytest.mark.parametrize("probe", ["readinessProbe", "livenessProbe"])
def test_both_probes_reach_a_path_the_app_serves_over_the_scheme_it_binds(
    probe: str,
) -> None:
    """Read off the app rather than written down twice: a manifest naming a path the
    code stopped serving is a pod that never becomes ready, and a probe declaring a
    scheme the server does not speak is a probe that fails while the process is healthy.

    `scheme` is absent on purpose and that is asserted rather than left implicit --
    kubelet defaults an httpGet probe to HTTP, and this container is told to serve HTTP.
    The TLS half is the pair below: uvicorn ignores a certificate with no key, so both
    flags or neither is the only coherent state, and here it has to be neither.
    """
    served = _paths_the_app_answers_a_get_on()
    http_get = _container()[probe]["httpGet"]
    declared = {port["name"] for port in _container()["ports"]}

    assert http_get["path"] in served, sorted(served)
    assert "scheme" not in http_get, (
        "the probe names a scheme; the container is told to serve plain HTTP"
    )
    assert http_get["port"] in declared
    assert _tls_flags_uvicorn_was_given() == set(), (
        "uvicorn is told to terminate TLS while every caller addresses this over http://"
    )


def _tls_flags_uvicorn_was_given() -> set[str]:
    """Which of the two TLS flags the container's args actually carry.

    Kept as a helper although the expected answer is now empty, because the empty answer
    is the assertion: this service is addressed by `http://model-gateway.<ns>.svc.
    cluster.local/v1`, the base_url `control/pod_config/compiler.py` renders into every
    compiled config.toml, and a plaintext request to a TLS socket does not negotiate.
    Naming both flags rather than grepping for `ssl` means a certificate added without a
    key is caught by the same rule.
    """
    wanted = {"--ssl-certfile", "--ssl-keyfile"}
    return {
        flag for arg in _container()["args"] for flag in [arg.partition("=")[0]]
    } & wanted


def _paths_the_app_answers_a_get_on() -> set[str]:
    """The GET paths the real app serves, read off its own OpenAPI document.

    Off the document rather than off `app.routes`, because this FastAPI flattens
    nothing: an included router is one opaque `_IncludedRouter` entry in that
    list, so walking it finds `/docs` and `/openapi.json` and none of the routes
    that matter -- which reads as "the app serves no health path" rather than as
    "this test looked in the wrong place".

    The gateway passed in is `None`: only the route table is read, and nothing behind it
    is reached by generating a schema.
    """
    document = create_model_gateway_app(cast(Any, None)).openapi()
    return {
        path for path, operations in document["paths"].items() if "get" in operations
    }


def test_every_absolute_path_this_container_names_sits_under_a_volume_it_mounts() -> (
    None
):
    """A manifest is a graph of names that must agree, not a list of fields.

    Each of these paths is asserted somewhere else in isolation and every one of them
    was free to disagree with the mountPath it depends on: `--ssl-certfile` moved to
    `/etc/map/nowhere/tls.crt` left every gate green and a uvicorn that refuses to bind.
    An assertion per field cannot catch a disagreement between two of them, so this is
    the pairwise rule stated once for every path the container names.
    """
    mounted = [mount["mountPath"] for mount in _volume_mounts()]
    named = _absolute_paths_this_container_names()

    assert named, "no absolute paths found, so this assertion grades nothing"
    for label, path in named:
        assert any(
            path == mount or path.startswith(f"{mount.rstrip('/')}/")
            for mount in mounted
        ), f"{label} names {path}, which is under none of {mounted}"


def test_the_container_port_is_the_port_the_args_tell_uvicorn_to_bind() -> None:
    """The number the process listens on and the number the cluster routes to.

    Two declarations of one fact in one file, and the probes and the Service both follow
    the declared one: raising `containerPort` to 9443 while uvicorn still binds 8443
    passed every gate and produced a pod whose probes and Service hit a closed port.
    """
    bound = [
        arg.partition("=")[2]
        for arg in _container()["args"]
        if arg.startswith("--port=")
    ]

    assert len(bound) == 1, _container()["args"]
    assert [port["containerPort"] for port in _container()["ports"]] == [int(bound[0])]


def test_the_service_targets_the_port_by_the_name_the_container_declares() -> None:
    """A number repeated in two documents is free to disagree with itself."""
    declared = {port["name"] for port in _container()["ports"]}

    for port in _of_kind("Service")["spec"]["ports"]:
        assert port["targetPort"] in declared


def test_the_service_selects_the_pods_this_deployment_labels() -> None:
    deployment = _of_kind("Deployment")

    assert (
        _of_kind("Service")["spec"]["selector"]
        == deployment["spec"]["selector"]["matchLabels"]
    )


def test_the_image_is_the_digest_placeholder_and_not_a_tag() -> None:
    """A tag is a name for whatever was pushed last; a digest is a name for bytes.

    The placeholder is what `deploy/platform.py` substitutes and what `substituted`
    refuses to apply without, so a digest hand-edited in here once would otherwise run
    that commit's bytes for ever.

    The three negatives are asserted over the **whole text** and not over the image
    field, which is deliberate and costs the manifest its right to quote its own history
    in a comment. A value sitting in a comment is one line away from being a value: this
    file's previous image reference was produced by copying a placeholder somebody else
    had written down, and the repository name in it was wrong. The sibling manifest
    quotes the old spelling and this one points at it instead.
    """
    text = _MANIFEST.read_text()

    assert _container()["image"] == f"map/model-gateway@{PLACEHOLDER}"
    assert "PLACEHOLDER_ECR" not in text
    assert ":latest" not in text
    assert "map-model-gateway:" not in text


def test_the_address_this_service_answers_to_is_the_one_a_compiled_config_names() -> (
    None
):
    """The Service, the port and the app's prefix compose the base_url, or nothing does.

    The Agent Runtime builds its request URL by concatenating the configured base_url
    with the literal segment `responses`, and it is given no port -- so the three facts
    that have to agree are the Service's name-and-namespace, its port being the one a
    portless URL means, and the app's prefix being the tail of that base_url. Each of
    them is asserted somewhere else in isolation and all three were free to disagree:
    port 443 with the same everything else passed every other case in this file and was
    a Service the platform does not address.

    `MODEL_PROVIDER_WIRE` is the cross-read that makes the last segment more than a
    literal typed twice -- it is the value the compiler writes as `wire_api`, and the
    runtime appends the same word to the base_url it was given.
    """
    service = _of_kind("Service")
    ports = service["spec"]["ports"]
    assert len(ports) == 1, ports
    assert ports[0]["port"] == 80, (
        "a compiled base_url carries no port, and a URL with no port is port 80"
    )

    host = (
        f"{service['metadata']['name']}."
        f"{service['metadata']['namespace']}.svc.cluster.local"
    )
    served = create_model_gateway_app(cast(Any, None)).openapi()["paths"]
    prefixes = {path.rsplit("/", 1)[0] for path in served}
    assert prefixes == {"/v1"}, sorted(served)

    assert f"http://{host}{prefixes.pop()}" == IN_CLUSTER_BASE_URL
    assert f"{IN_CLUSTER_BASE_URL}/{MODEL_PROVIDER_WIRE}" == (
        f"http://{host}/v1/responses"
    )
    assert "/v1/responses" in served


def test_no_environment_variable_names_a_vault_entry_at_all() -> None:
    """Two locations name vault entries, and the env block is not one of them.

    `deploy/platform.py` learns which entries to ask the account about from the routing
    table and from `vault_variables`. A third location -- an env variable naming another
    entry -- would be invisible to it, and the failure that causes arrives hours later
    as an AccessDenied naming a secret path.

    Until 2026-08-23 there was exactly one such variable, `MAP_POD_TOKEN_KEY_NAME`, and
    it named the signing key. It is gone: the key is a Kubernetes Secret now, read from
    the environment as a value rather than as a name to look up, and the reasoning is
    in `test_the_only_secret_this_container_reaches_for_is_the_one_it_verifies_with`.
    So the honest assertion is now that the set is **empty**, which is a stronger claim
    than the one it replaces and cannot pass vacuously the way "every one of them is
    declared" would with none of them to check.

    `vault_variables` stays on the `Workload` dataclass and is still exercised by
    `test_the_applier_refuses_a_manifest_whose_key_variable_was_renamed`. It is the
    mechanism for a variable of this shape, and the mechanism is worth keeping against
    the day one comes back; what must not come back silently is a variable using it
    without the applier being told.

    The shape is `map/dev/<something>/<something>`, which is what every entry in this
    account is named and what both `deploy/iam/map-model-gateway.json` and
    `deploy/terraform/secrets.tf` are written against.
    """
    module = _platform()
    workload = next(w for w in module.WORKLOADS if w.component == "model-gateway")
    vault_shaped = re.compile(r"\Amap/dev/[^/\s]+/[^/\s]+\Z")

    named = {
        name for name, value in _container_env().items() if vault_shaped.match(value)
    }

    assert named == set(workload.vault_variables), (
        f"{sorted(named ^ set(workload.vault_variables))} name vault entries the "
        "manifest and deploy/platform.py disagree about"
    )
    assert named == set(), (
        f"{sorted(named)} names a vault entry from the env block; the provider "
        "credentials belong in routing.json and nothing else here names one"
    )


def test_the_applier_asks_the_account_about_exactly_the_names_this_file_declares() -> (
    None
):
    """One reading of the manifest, not two.

    The applier's refusal is only worth its exit code if the names it asks about are the
    names the pod reads: a check built on a hand-kept list would keep passing after a
    model was added to the routing table, which is the case this service will actually
    grow. Compared against the real parser's own answer rather than against a literal,
    for the same reason.
    """
    module = _platform()
    workload = next(w for w in module.WORKLOADS if w.component == "model-gateway")
    table = routing_table_from_json(_routing_document())

    declared = module.declared_vault_entries(_ROOT, workload)
    names = {name for _why, name in declared}

    assert names == {
        table.entry_for(model).credential_name for model in table.declared_models()
    }
    assert all(why for why, _name in declared), (
        "a refusal that cannot say what named an entry leaves somebody grepping for a "
        "secret path"
    )


def test_the_applier_refuses_a_manifest_whose_key_variable_was_renamed() -> None:
    """A location that is not there must raise, not check nothing.

    This is the failure mode an allowlist of names has: rename the variable and the
    check keeps passing while the entry it was written for goes unasked-about. Driven
    against the real manifest with one field of the declaration changed, because the
    manifest itself must keep the name it has.
    """
    module = _platform()
    workload = next(w for w in module.WORKLOADS if w.component == "model-gateway")
    renamed = replace(workload, vault_variables=("MAP_POD_TOKEN_KEY_NAME_TYPO",))

    with pytest.raises(RuntimeError, match="MAP_POD_TOKEN_KEY_NAME_TYPO"):
        module.declared_vault_entries(_ROOT, renamed)


_A_STAND_IN_FOR_A_SECRET: Final = "a-value-the-cluster-supplies-and-this-file-cannot"


def _container_env() -> dict[str, str]:
    """Every variable the container is started with, and something to put in each.

    A `secretKeyRef` entry gets a stand-in rather than being dropped. Dropping it would
    silently shrink the set every caller here scans -- the absolute-path check, the
    vault-shape check, the boot test -- so a variable moved from a literal to a Secret
    would fall out of all three at once and nothing would say so. The name is what they
    need; only the boot test needs a value at all, and only that it is present.
    """
    declared: dict[str, str] = {}
    for entry in _container()["env"]:
        if "value" in entry:
            declared[str(entry["name"])] = str(entry["value"])
        else:
            declared[str(entry["name"])] = _A_STAND_IN_FOR_A_SECRET
    return declared


def test_the_args_name_a_factory_the_composition_root_actually_exposes() -> None:
    """`uvicorn --factory` resolves this string at start-up, so a name that is not there
    is a crash-loop whose reason is in the container's own log and nowhere else."""
    args = _container()["args"]
    target = "managed_agent.composition:model_gateway_app"

    assert target in args
    assert "--factory" in args
    assert callable(getattr(composition, target.split(":")[1]))


def test_this_manifests_own_environment_boots_the_factory_it_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The env block and the factory are driven against each other, not read separately.

    Every variable the factory reads has no default and raises when absent, which is the
    right behaviour and also the reason a name that drifted between this file and
    `composition.py` would surface only as a container that exits at start-up with a
    `KeyError` in its own log. Here it surfaces as this test.

    Nothing is overridden. The container's volumes are reconstructed under `tmp_path`
    and the env block is taken verbatim with only the leading `/` rebased, so the
    manifest's own value for every variable is the value the factory reads -- including
    the one that names a path, which an earlier version of this test set itself and
    therefore could not grade. Deleting or renaming that variable, pointing it outside
    the mounted volume, moving the volumeMount away from it, or naming a ConfigMap this
    file does not hold each fail here now.

    Nothing dials: the vault client and the HTTP client are both constructed lazily.
    """
    _rebuild_the_mounted_filesystem(tmp_path)
    for name, value in _environment_rebased_under(tmp_path).items():
        monkeypatch.setenv(name, value)

    app = composition.model_gateway_app()

    assert set(app.openapi()["paths"]) == {"/v1/responses", "/v1/healthz"}
    # Two of the three wires, and the assertion is on the exact set rather than on
    # membership: a wire registered without a handler that can serve it is a model this
    # gateway accepts and then fails mid-stream. It read `[RESPONSES]` until MAP-23
    # shipped the Anthropic translator, which is the slice this line was waiting for.
    # Chat Completions is the one still absent, and a model declared on it is refused
    # loudly rather than sent a Responses body to an endpoint that does not accept one.
    assert set(app.state.gateway.handlers) == {
        UpstreamWire.RESPONSES,
        UpstreamWire.ANTHROPIC_MESSAGES,
    }, "a wire is registered here only when a handler exists that can serve it"
    assert app.state.gateway.table.declared_models() == frozenset(
        routing_table_from_json(_routing_document()).declared_models()
    )


# --- the one thing no local assertion can do: ask the upstream ----------------------


_UPSTREAM_GATE: Final = "MAP_MODEL_GATEWAY_UPSTREAM"

requires_the_upstream = pytest.mark.skipif(
    os.environ.get(_UPSTREAM_GATE) != "1",
    reason=(
        f"the upstream proof is opt-in: set {_UPSTREAM_GATE}=1 to run it. It needs "
        "AWS credentials that can read map/dev/providers/*, outbound network, and a "
        "few tokens of real provider quota per model. SKIPPED MEANS NO UPSTREAM WAS "
        "ASKED, AND EVERY ASSERTION IN THIS FILE ABOVE IS SATISFIED BY A ROUTING "
        "TABLE THAT REACHES NOTHING."
    ),
)


def _vault_value(name: str) -> str:
    """One vault entry, read through the AWS CLI rather than through the adapter.

    The CLI and not `SecretsManagerVault` on purpose. This test's subject is whether the
    *deployed configuration* names a reachable upstream, and routing that read through
    the platform's own adapter would make an adapter defect look like a configuration
    defect. The value is never returned to a caller that prints it.
    """
    completed = subprocess.run(
        [
            "aws",
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            name,
            "--region",
            "us-east-1",
            "--query",
            "SecretString",
            "--output",
            "text",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"could not read {name}: {completed.stderr.strip()[:300]}"
    )
    return completed.stdout.strip()


@requires_the_upstream
@pytest.mark.parametrize(
    "model", sorted(routing_table_from_json(_routing_document()).declared_models())
)
def test_every_routed_model_is_one_its_upstream_actually_answers(model: str) -> None:
    """The guard that would have caught three defects at once, and did not exist.

    On 2026-08-23 the Anthropic entry in this manifest named the host
    `map-foundry.services.ai.azure.com`, which does not resolve, and the model
    `claude-opus-5`, which this account never deployed -- and the credential broker
    handed the whole JSON vault entry to the upstream as a bearer. Three independent
    defects on one code path, and **every** assertion in this file passed through all of
    them, because each one is a claim about text that is well formed. A hostname that
    does not exist parses as a hostname. A model nobody deployed is a string. A JSON
    document is ASCII and fits in a header.

    Nothing local can disagree with any of that. The only thing that can is the
    upstream, which is why this test makes a real request rather than checking a shape.
    It is parametrised over the table's own models, so a model added to the manifest is
    probed the day it is added rather than the day somebody remembers.

    Deliberately probing with the **broker's** extraction of the credential and the
    **routing entry's** own base URL: those two are what a real Turn uses, so a probe
    that reconstructed either from the vault entry's convenience copies would pass while
    the deployment failed. That is the exact failure mode this replaces -- the vault
    entry's `base_url` was correct on that date and the routing table's was not, so a
    probe trusting the entry would have reported everything healthy.

    Opt-in because it costs real provider quota and needs credentials no local run has.
    It is not a substitute for the assertions above; it is the one that can fail for a
    reason none of them can see.
    """
    entry = routing_table_from_json(_routing_document()).entry_for(model)
    if entry.wire is not UpstreamWire.ANTHROPIC_MESSAGES:
        pytest.skip(
            f"{model} is on wire {entry.wire.value}, whose probe body this test does "
            "not know how to build yet -- SKIPPED MEANS THIS MODEL WAS NOT ASKED. "
            "Whether its credential exists at all is the sibling test's question, "
            "which is not skipped."
        )
    broker = ProviderCredentialBroker(
        cast(Any, _AVaultReturningOneValue(_vault_value(entry.credential_name))),
        cast(Any, _AStoppedClock()),
    )
    credential = asyncio.run(broker.for_turn(SessionId(uuid.uuid4()), entry))
    header_name, header_value = credential.header()
    url = f"{entry.base_url.rstrip('/')}/v1/messages"
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with: ready"}],
        }
    ).encode()
    # `anthropic-version` is imported from the module that puts it on a real request
    # rather than spelled again here. It is required and not optional -- without it the
    # route answers 400 -- and the Agent Runtime never sends it, so whatever forwards a
    # Turn onto this wire has to. Two spellings of a wire-version pin is how a probe
    # comes to certify a version production does not send.
    headers = {
        "content-type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
        header_name.decode("ascii"): header_value.decode("ascii"),
    }

    with httpx.Client(timeout=60.0) as client:
        answer = client.post(url, content=body, headers=headers)

    assert answer.status_code == 200, (
        f"{model} at {entry.base_url} answered {answer.status_code}: "
        f"{answer.text[:400]}"
    )
    payload = answer.json()
    assert payload.get("content"), payload
    assert payload.get("stop_reason"), payload


class _AVaultReturningOneValue:
    """The value already read, handed to the broker so the broker does the extraction.

    The point is that the probe above goes through the broker's own parse rather than
    around it. A probe that pulled `api_key` out of the entry itself would have been
    green on 2026-08-23 while every real request carried the whole JSON document.
    """

    def __init__(self, value: str) -> None:
        self._value = value

    async def fetch(self, name: str) -> str:
        return self._value


class _AStoppedClock:
    def now_epoch_ms(self) -> int:
        return 0


@requires_the_upstream
def test_every_credential_this_table_names_is_one_the_account_holds() -> None:
    """The cheap half of the question above, asked without spending any quota.

    Split from the live probe on purpose, because the two fail for different reasons and
    a reader needs to know which. This one asks only whether the vault entry *exists*; a
    model whose wire has no probe body still has its credential checked here, so a
    missing entry cannot hide behind a skip.

    It found one the first time it ran. On 2026-08-23 this table named
    `map/dev/providers/openai` and the account held no such entry, so every
    `gpt-5-codex` call would answer 503 -- on the one wire this build actually serves.
    `deploy/platform.py` already refuses to apply a workload naming an entry the
    account lacks, and the Deployment is running, so the entry went away after the
    apply. That is the gap: the applier checks once and nothing checks again.

    One `list-secrets` rather than a `describe-secret` per name, so an absent entry
    and a read this role may not make are told apart by the same call.
    """
    table = routing_table_from_json(_routing_document())
    named = {
        table.entry_for(model).credential_name for model in table.declared_models()
    }
    completed = subprocess.run(
        [
            "aws",
            "secretsmanager",
            "list-secrets",
            "--region",
            "us-east-1",
            "--query",
            "SecretList[].Name",
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.strip()[:300]
    held = set(json.loads(completed.stdout))

    assert named, "the table names no credentials, so this assertion grades nothing"
    assert named <= held, (
        f"{sorted(named - held)} is named by deploy/k8s/model-gateway.yaml and is not "
        "in the account, so every call to that model answers 503"
    )
