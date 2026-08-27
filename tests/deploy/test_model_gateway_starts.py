"""The Model Gateway as a running service: the applier's refusals, and then the pod.

Three tiers, and the gate on each says what a skipped run does not prove.

Tier 1 is local. It drives the factory the manifest names against an environment that is
missing one variable at a time, and it drives `deploy/platform.py`'s two new pure
functions -- the ones that decide whether a vault entry a manifest names is in the
account. Nothing here reaches a cluster, a registry or AWS.

Tier 2 puts the Deployment in `map-dev` and asks it from another pod. It is skipped
unless `MAP_CLUSTER_TESTS=1`, the gate `tests/deploy/test_tool_gateway_starts.py` and
`tests/deploy/test_node_preconditions.py` already use. A SKIPPED RUN SAYS NOTHING ABOUT
THE CLUSTER. Its cases are a positive and a negative through one probe mechanism in one
run, because a negative alone cannot tell a service that refused a call from a probe
that never reached one.

Tier 3 is the wire: a pod running the **Session** image, asking the address every
compiled `config.toml` gives the Agent Runtime as its `base_url`, and then the Gateway's
own log read back for that request. It needs `MAP_SESSION_IMAGE` as well as the gate.

WHAT NONE OF THIS PROVES: that a Turn works. The Agent Runtime reaches this service
unauthenticated -- `control/pod_config/compiler.py` writes no credential field into the
provider table, deliberately, because a Session pod holds no provider secret -- so the
strongest thing on the wire today is a request that arrives and is refused. Two further
refusals sit in front of it inside the pod, owned by MAP-67 and MAP-68, and neither is
this slice's. So tier 3 asserts that the address resolves and that the Gateway logged
the request; it asserts nothing about a model answering.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast
from unittest import mock

import pytest
import yaml
from fastapi.testclient import TestClient

from managed_agent import composition
from managed_agent.gateway.model.router import (
    ModelGateway,
    SessionTokenVerifier,
    create_model_gateway_app,
    routing_table_from_json,
)

_ROOT: Final = Path(__file__).resolve().parents[2]
_MANIFEST: Final = _ROOT / "deploy" / "k8s" / "model-gateway.yaml"

_GATE: Final = "MAP_CLUSTER_TESTS"
_SESSION_IMAGE: Final = "MAP_SESSION_IMAGE"
_PROBE_POD: Final = "map-modelgw-probe"
_COMPONENT: Final = "model-gateway"

UNAUTHENTICATED_BODY: Final = {
    "error": {"type": "unauthenticated", "message": "a bearer token is required"}
}
"""What this service answers a request that carries no bearer token.

A literal, because the cluster tiers below read it off a wire and have no way to import
it -- and the first case in this file is what keeps the literal true, by driving the
real route and comparing. Asserting the body and not only the 401 is the difference
between "the Gateway refused this" and "something between here and the Gateway
refused this": an ingress, a proxy or a sidecar answers 401 too.
"""


def _platform() -> ModuleType:
    """`deploy/platform.py`, loaded by path -- it is a script, so there is no import."""
    spec = importlib.util.spec_from_file_location(
        "map_platform_for_model_gateway_starts", _ROOT / "deploy" / "platform.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workload() -> Any:
    module = _platform()
    return next(w for w in module.WORKLOADS if w.component == _COMPONENT)


def _documents() -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all(_MANIFEST.read_text()) if d]


def _of_kind(kind: str) -> dict[str, Any]:
    found = [d for d in _documents() if d["kind"] == kind]
    assert len(found) == 1, f"expected exactly one {kind}, found {len(found)}"
    return found[0]


def _container() -> dict[str, Any]:
    containers = _of_kind("Deployment")["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1
    return dict(containers[0])


_A_STAND_IN_FOR_A_SECRET: Final = "a-value-the-cluster-supplies-and-this-file-cannot"


def _environment_the_manifest_declares() -> dict[str, str]:
    """Every variable the container is started with, and something to put in each.

    Two kinds of entry, and they differ in whether the manifest holds the value. A
    literal `value` is the manifest's own and is returned as written. A `secretKeyRef`
    names a Kubernetes Secret this file cannot read and should not want to, so it gets
    a stand-in: what the callers below need is the *name* and the fact that the factory
    refuses when it is absent, neither of which depends on the real bytes.

    Both kinds are returned from one dict on purpose. The callers parametrise over the
    manifest's own env block precisely so that a variable added there is covered the
    day it is added -- and a variable moved from a literal to a Secret, which is what
    happened to the signing key, must not silently drop out of that coverage.
    """
    declared: dict[str, str] = {}
    for entry in _container()["env"]:
        name = str(entry["name"])
        if "value" in entry:
            declared[name] = str(entry["value"])
            continue
        assert "secretKeyRef" in entry["valueFrom"], entry
        declared[name] = _A_STAND_IN_FOR_A_SECRET
    return declared


def base_url_the_manifest_publishes() -> str:
    """The in-cluster address this manifest makes answerable, composed from itself.

    Derived rather than typed, so a Service renamed, moved to another namespace, or
    given a port other than the one a portless URL means moves this string and fails
    whatever depends on it. `tests/deploy/test_model_gateway_manifest.py` is where the
    composed value is pinned against the literal every compiled `config.toml` carries;
    here it is simply what the probes address.
    """
    service = _of_kind("Service")
    ports = service["spec"]["ports"]
    assert len(ports) == 1 and ports[0]["port"] == 80, ports
    host = (
        f"{service['metadata']['name']}."
        f"{service['metadata']['namespace']}.svc.cluster.local"
    )
    served = create_model_gateway_app(cast(Any, None)).openapi()["paths"]
    prefixes = {path.rsplit("/", 1)[0] for path in served}
    assert prefixes == {"/v1"}, sorted(served)
    return f"http://{host}{prefixes.pop()}"


# --- tier 1: the factory, and the applier's refusals --------------------------------


class _AStoppedClock:
    def now_epoch_ms(self) -> int:
        return 0


def test_the_fixed_body_this_file_asserts_on_the_wire_is_the_one_the_app_writes() -> (
    None
):
    """The literal above, compared against the real route once, locally.

    The two cluster tiers cannot import a constant out of the process they are talking
    to, so they compare bytes to a literal. This is what stops that literal from
    drifting: change the refusal in `gateway/model/router.py` and this fails here,
    naming the two places that then disagree, rather than in a cluster run somebody may
    not make.
    """
    gateway = ModelGateway(
        table=routing_table_from_json(
            _of_kind("ConfigMap")["data"]["routing.json"].encode()
        ),
        handlers={},
        tokens=SessionTokenVerifier(
            key=b"a-key-nothing-here-signs-with", clock=cast(Any, _AStoppedClock())
        ),
    )
    with TestClient(create_model_gateway_app(gateway)) as client:
        answer = client.post("/v1/responses", json={"model": "gpt-5-codex"})

    assert answer.status_code == 401
    assert answer.json() == UNAUTHENTICATED_BODY


@pytest.mark.parametrize("absent", sorted(_environment_the_manifest_declares()))
def test_the_factory_raises_naming_the_variable_that_is_absent(
    absent: str, tmp_path: Path
) -> None:
    """No defaults, and the message says which one.

    Parametrised over the manifest's own env block rather than over a list written here,
    so a variable added to the manifest is covered the day it is added and a variable
    the factory stopped reading fails this instead of being quietly carried.

    Every value is the manifest's own except the routing-table path, which is redirected
    at a copy of the manifest's own ConfigMap document written under `tmp_path`. The
    factory reads that file before it reads the last variable, so a path pointing
    nowhere would make three of the four cases fail on a `FileNotFoundError` and prove
    nothing about which name was missing.
    """
    table = tmp_path / "routing.json"
    table.write_text(_of_kind("ConfigMap")["data"]["routing.json"])
    declared = _environment_the_manifest_declares()
    partial = {
        name: (str(table) if value.startswith("/") else value)
        for name, value in declared.items()
        if name != absent
    }
    with (
        mock.patch.dict(os.environ, partial, clear=True),
        pytest.raises(KeyError, match=absent),
    ):
        composition.model_gateway_app()


def test_an_entry_the_account_holds_is_not_reported_and_one_it_does_not_is() -> None:
    """Both arms in one case, because a refusal list that is always empty refuses
    nothing and a refusal list that is always full refuses everything.

    The names are this manifest's real ones, so an entry renamed in the manifest moves
    this case rather than leaving it asserting against a name nothing declares.
    """
    module = _platform()
    declared = module.declared_vault_entries(_ROOT, _workload())
    names = [name for _why, name in declared]

    assert module.unreachable_vault_entries(declared, frozenset(names)) == ()

    lines = module.unreachable_vault_entries(declared, frozenset(names[1:]))
    assert len(lines) == 1
    assert names[0] in lines[0]
    assert declared[0][0] in lines[0], (
        "the refusal names the entry but not what named it, so a reader is left "
        "grepping for a secret path"
    )


def test_the_applier_names_every_routed_credential_and_nothing_else() -> None:
    """Every upstream credential the table routes to, checked before the apply.

    A check that read only the env block would apply a Deployment that answers every
    model call with an AccessDenied, and that failure arrives at the first real request
    rather than during the rollout.

    The signing key is deliberately *not* in this set and the second assertion holds
    that line. It is a Kubernetes Secret key rather than a vault entry, mounted by the
    manifest and read from the environment at startup, so a missing one crashes the
    container and the rollout stalls where somebody is watching. Re-declaring it here
    would put a second name on the same fact and give the two copies room to disagree.
    """
    module = _platform()
    workload = _workload()
    table = routing_table_from_json(
        _of_kind("ConfigMap")["data"]["routing.json"].encode()
    )

    names = {name for _why, name in module.declared_vault_entries(_ROOT, workload)}

    for model in table.declared_models():
        assert table.entry_for(model).credential_name in names, model
    assert not [name for name in names if "signing-key" in name or "token" in name], (
        "a signing key came back as a vault entry, so it is now declared twice"
    )


def test_a_routing_table_the_manifest_does_not_hold_is_refused() -> None:
    """The second location that must raise rather than check nothing.

    Same shape as the renamed variable: a declaration pointing at a document this file
    does not carry would otherwise leave every routed credential unasked-about, and the
    manifest would apply. Driven by moving the declaration rather than the manifest,
    which keeps the table it has.
    """
    module = _platform()
    with pytest.raises(RuntimeError, match="expected 1"):
        module.declared_vault_entries(
            _ROOT, replace(_workload(), routing_table_key="not-a-key.json")
        )


def test_an_empty_routing_table_cannot_reach_the_credential_scan_at_all() -> None:
    """The guarantee `declared_vault_entries` leans on, asserted rather than assumed.

    A table with no entries is the document that would make "every credential this
    workload names is in the account" pass by naming none -- the vacuous-guard shape.
    There is no branch for it in the applier because the parser refuses it first, and
    this is what makes that reasoning checkable: delete the schema's minimum and this
    fails, pointing at the applier comment that cites it.
    """
    with pytest.raises(ValueError, match="at least 1 item"):
        routing_table_from_json(b'{"entries": []}')


# --- tier 2 and 3: the cluster ------------------------------------------------------


requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"the cluster proof is opt-in: set {_GATE}=1 to run it. It needs kubectl "
        "against map-dev, the Deployment applied by `deploy/platform.py "
        "model-gateway`, and it creates and deletes one probe pod. SKIPPED MEANS THE "
        "CLUSTER WAS NOT CONSULTED."
    ),
)

requires_the_session_image = pytest.mark.skipif(
    os.environ.get(_GATE) != "1" or not os.environ.get(_SESSION_IMAGE),
    reason=(
        f"the wire proof needs {_GATE}=1 and {_SESSION_IMAGE} set to a digest-pinned "
        "reference in map/session-shim, because what it proves is that a pod running "
        "the Session image reaches this Service. SKIPPED MEANS THE WIRE WAS NOT READ."
    ),
)


def _kubectl(*argv: str) -> str:
    done = subprocess.run(
        ("kubectl", "-n", "map-dev", *argv), capture_output=True, text=True
    )
    if done.returncode != 0:
        pytest.fail(f"kubectl {' '.join(argv)} failed:\n{done.stderr}")
    return done.stdout


def _finished(timeout_s: float = 180.0) -> str:
    """Wait for the probe pod to stop running, and return the phase it stopped in.

    Polled rather than `kubectl wait --for=condition=Ready=false`, which a *pending* pod
    also satisfies -- so that spelling can return before the container has run and hand
    the caller an empty log to parse.
    """
    deadline = time.monotonic() + timeout_s
    phase = ""
    while time.monotonic() < deadline:
        phase = _kubectl(
            "get", "pod", _PROBE_POD, "-o", "jsonpath={.status.phase}"
        ).strip()
        if phase in ("Succeeded", "Failed"):
            return phase
        time.sleep(2)
    pytest.fail(f"{_PROBE_POD} was still {phase or 'unscheduled'} after {timeout_s}s")


def _probe(url: str, image: str, method: str = "GET") -> tuple[int, str]:
    """Ask the Service from inside the cluster, from a pod running `image`.

    The image is a parameter rather than a constant because the two tiers ask different
    questions with the same mechanism: tier 2 asks whether the Service answers and uses
    whatever the Deployment is running, and tier 3 asks whether a **Session** pod can
    reach it and so has to be the Session image. Both carry httpx, so this needs no
    second image, no curl, and no registry the cluster cannot reach.

    The answer is parsed out of the pod's own stdout as JSON, so a pod that failed to
    start or an httpx exception fails this rather than reading as a status of zero.

    IT WEARS `map.role: session-pod`, and that is a correctness requirement rather than
    decoration. Since 2026-08-26 this namespace is default-deny, and the Model Gateway's
    ingress admits exactly one client: a Session pod -- which is the point of that
    document, since this is the process holding the provider credential no pod holds. An
    unlabelled probe is refused twice over, by the floor and by that ingress, so before
    the label this measured a caller production never has. It also collapses the tier 2
    / tier 3 distinction above in the one way that matters: both tiers now dial as the
    client the Service actually serves, and the image is still what separates them.
    """
    program = (
        "import httpx,sys,json;"
        f"r=httpx.request({method!r},{url!r},timeout=10);"
        "print(json.dumps({'status':r.status_code,'body':r.text[:200]}))"
    )
    _kubectl("delete", "pod", _PROBE_POD, "--ignore-not-found")
    _kubectl(
        "run",
        _PROBE_POD,
        "--image",
        image,
        "--labels",
        "map.role=session-pod",
        "--restart=Never",
        "--command",
        "--",
        "python",
        "-c",
        program,
    )
    try:
        phase = _finished()
        logs = _kubectl("logs", _PROBE_POD)
        assert phase == "Succeeded", f"the probe pod {phase}; its output was:\n{logs}"
        answer = json.loads(logs.strip().splitlines()[-1])
        return int(answer["status"]), str(answer["body"])
    finally:
        _kubectl("delete", "pod", _PROBE_POD, "--ignore-not-found", "--wait=false")


def _serving_image() -> str:
    described = json.loads(_kubectl("get", "deploy", _COMPONENT, "-o", "json"))
    image = described["spec"]["template"]["spec"]["containers"][0]["image"]
    return str(image)


@requires_the_cluster
def test_the_deployment_is_available_at_its_own_replica_count() -> None:
    """`kubectl rollout status` exits 0 for a Deployment scaled to zero, so the count is
    read off the object rather than off an exit code."""
    described = json.loads(_kubectl("get", "deploy", _COMPONENT, "-o", "json"))
    desired = described["spec"]["replicas"]

    assert desired > 0
    assert described["status"].get("availableReplicas") == desired, described["status"]


@requires_the_cluster
def test_a_pod_in_the_namespace_gets_two_hundred_from_the_health_path() -> None:
    """Through the Service name and its port 80, which is what an in-cluster caller
    uses -- not the container's 8080. A test that reached the pod IP would pass with the
    Service misconfigured.

    This is also the positive control for the two refusal cases below: without it, a 401
    could be a refusal or could be a probe that never reached the Service at all.
    """
    status, _ = _probe(f"{base_url_the_manifest_publishes()}/healthz", _serving_image())

    assert status == 200


@requires_the_cluster
def test_the_health_path_is_under_the_prefix_and_not_at_the_root() -> None:
    """`/healthz` at the root is 404 here, and that is the shape rather than a defect.

    `create_model_gateway_app` mounts everything under `/v1` because the Agent Runtime
    builds its request URL by concatenating the configured base_url with the literal
    `responses`, so the app's paths have to be the tail of the base_url an operator
    registers. A second health route at the root would be an unprefixed surface on the
    same process, and this asserts there is not one -- if a later slice adds one
    deliberately, this is where that decision has to be argued rather than absorbed.
    """
    root = base_url_the_manifest_publishes().removesuffix("/v1")

    status, _ = _probe(f"{root}/healthz", _serving_image())

    assert status == 404, (
        "the root now answers a health path as well as /v1/healthz; two health "
        "surfaces on one process is a decision, not a fix"
    )


@requires_the_cluster
@pytest.mark.parametrize("header", ["", "x-map-session"])
def test_a_model_call_with_no_bearer_token_is_refused_at_the_front_door(
    header: str,
) -> None:
    """The strongest thing reachable today, and it is a negative.

    It proves the token check is on the wired path of the real process rather than only
    in a unit test: a Gateway wired without it would try to route the body and answer
    404 for an unconfigured model or 502 for a wire with no handler.

    The parametrisation is the two spellings of "unauthenticated" this platform uses.
    `x-map-session` is the header the **Tool** Gateway reads, and a pod that sent only
    that one would be a pod using the wrong service's credential shape; this service
    reads `authorization: Bearer`, so both arms are refused and refused identically.
    """
    url = f"{base_url_the_manifest_publishes()}/responses"
    program_url = f"{url}?probe={header or 'none'}"

    status, body = _probe(program_url, _serving_image(), method="POST")

    assert status == 401
    assert json.loads(body) == UNAUTHENTICATED_BODY


@requires_the_session_image
def test_a_pod_running_the_session_image_reaches_the_gateway_on_the_wire() -> None:
    """The wire, proven by the Gateway's own log rather than by the absence of an error.

    The address is the one every compiled `config.toml` names, composed from the
    manifest, with `responses` appended the way the Agent Runtime appends it. The pod is
    the Session image, so what is exercised is the image a Session actually runs
    resolving that name and connecting to it.

    A unique query string is what makes the log line this request's rather than any
    request's: two replicas serve this Service and the probe lands on one of them, so
    the logs are read by label across both.

    WHAT THIS IS NOT: a Turn. Nothing here starts the Agent Runtime. The runtime's own
    path to this address is blocked inside the pod by refusals MAP-67 and MAP-68 own,
    and it would arrive unauthenticated even past them, so the honest claim is that the
    address resolves and the Gateway received the request -- which is the hop that did
    not exist before this slice.
    """
    marker = uuid.uuid4().hex
    url = f"{base_url_the_manifest_publishes()}/responses?probe={marker}"

    status, body = _probe(url, os.environ[_SESSION_IMAGE], method="POST")

    assert status == 401
    assert json.loads(body) == UNAUTHENTICATED_BODY

    logged = _kubectl(
        "logs", "-l", f"map.component={_COMPONENT}", "--tail=200", "--prefix"
    )
    assert marker in logged, (
        "the probe got an answer but no replica logged the request, so what answered "
        f"is not this Deployment. Last 200 lines per replica:\n{logged}"
    )
