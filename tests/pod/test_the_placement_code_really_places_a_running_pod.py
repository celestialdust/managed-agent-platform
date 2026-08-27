"""The production placement code creates a Session pod in map-dev that reaches RUNNING.

Skipped unless MAP_CLUSTER_TESTS=1, and deselected unless the mark expression admits
`network`. SKIPPED OR DESELECTED MEANS NO POD WAS PLACED. The documented invocation is
`MAP_CLUSTER_TESTS=1 pytest tests/pod -m "image or not image"` -- one `-m`, because a
second one replaces the first rather than adding to it.

Why this exists beside `test_the_control_plane_places_a_session_pod.py`. That file
drives the REST API of the **deployed** control plane, which is the right thing to grade
and is currently blocked on two counts: its Deployment is missing `MAP_POD_MANIFEST` and
`MAP_SHIM_TOKEN_KEY`, and the image it runs predates the placement adapter entirely
(`ModuleNotFoundError: No module named 'managed_agent.adapters.kubernetes'`). Both are
fixed by a redeploy that the applier refuses until `map-control-plane/shim-token-key`
exists.

So that file cannot answer whether the *placement code* works, and until this one was
written nothing could. The record claimed for several sessions that it was unprovable;
it was not. What placement needs is the code, the committed manifest, a ServiceAccount
that may create pods, and an API server -- the Secret is only how the deployed
Deployment obtains a key, and this drives the runner with a key of its own.

NO KEY VALUE IS PRINTED OR ASSERTED ON. The shim-token key is `os.urandom(32)` and dies
with the process. The Session-token key is read from the cluster because the deployed
Tool Gateway verifies against it -- a key invented here produces `HTTP 401: invalid
session token` at the runtime's MCP handshake and the pod never starts, which is that
gateway's authentication working. It is passed to the compiler and to nothing else.

What this file does NOT show is a Turn *completing*, and the reason changed on
2026-08-23. The paragraph that stood here described a compiled provider block naming
`env_key = "MAP_POD_TOKEN"` against a variable no manifest declared, and two open
escalations about a vault entry holding the signing key. **None of that is still true**,
and the correction is worth keeping rather than overwriting silently: `env_key` is gone
(the bearer rides in `http_headers` instead), the escalations were closed by deleting
the code that needed them, and the entry `map/dev/platform/pod-token-signing-key` is not
read by anything any more. A reader acting on the old text would go looking for a
variable and a secret that no longer figure in this.

What is true is one layer further out. The compiled document now carries a token the
Model Gateway can actually verify, and the Anthropic upstream the routing table names
answers HTTP 200 when asked directly. What is missing is the **translator**: a model
declared on the `anthropic_messages` wire has no registered handler, so the Gateway
refuses it loudly rather than sending a Responses body to an endpoint that does not
accept one. That refusal is deliberate. Until the handler exists a Turn cannot complete
on that model, and `test_a_turn_is_accepted_and_streams_to_a_terminal_line` says in one
place what has to change when one does.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import (
    NAMESPACE,
    forwarded,
    internal_ca_of_the_cluster,
    shim_dial,
)

from managed_agent.adapters.kubernetes.pod_runner import KubernetesPodRunner
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.pod_config import compiler as config_compiler
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import (
    SessionId,
    TenantId,
    new_definition_id,
)
from managed_agent.core.registration.definition import AgentDefinition, SkillsRevision
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord

_ROOT: Final = Path(__file__).resolve().parents[2]
_MANIFEST: Final = _ROOT / "deploy" / "k8s" / "session-pod.yaml"
_REGION: Final = "us-east-1"
_REPOSITORY: Final = "map/session-shim"
_GATEWAY_SECRET: Final = ("map-tool-gateway", "session-token-key")
_SHIM_PORT: Final = 8081
_EXPIRY: Final = 4102444800

requires_the_cluster = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="places a real pod in map-dev; set MAP_CLUSTER_TESTS=1 to run",
)
pytestmark = [pytest.mark.network, requires_the_cluster]


def _run(*argv: str) -> str:
    """One command, refused loudly rather than returning an empty string.

    `check=True` because the failure mode this guards is a probe that reports nothing
    and reads as a pass: an empty answer from `kubectl` looks the same as an empty
    cluster.
    """
    done = subprocess.run(argv, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _aws(*argv: str) -> Any:
    return json.loads(_run("aws", *argv, "--output", "json"))


def _kubectl(*argv: str) -> str:
    return _run("kubectl", "-n", NAMESPACE, *argv)


def _session_image() -> str:
    """The newest digest in the Session repository, as the push script prints it.

    Newest-push rather than a pinned tag, matching the other cluster files: the
    manifest's own reference is sixty-four zeros, a placeholder that resolves to no
    image anywhere.
    """
    account = _aws("sts", "get-caller-identity")["Account"]
    details = _aws("ecr", "describe-images", "--repository-name", _REPOSITORY)[
        "imageDetails"
    ]
    newest = max(details, key=lambda one: str(one["imagePushedAt"]))
    return (
        f"{account}.dkr.ecr.{_REGION}.amazonaws.com/"
        f"{_REPOSITORY}@{newest['imageDigest']}"
    )


def _session_token_key() -> bytes:
    """The key the DEPLOYED Tool Gateway verifies Session tokens with.

    Read from the cluster rather than invented, because the runtime's MCP handshake
    dials that gateway during startup and a token it refuses stops the pod before it
    runs. Never printed and never asserted on; `composition.py` does
    `os.environ[...].encode()`, so the bytes of the Secret's decoded value are the key
    with no second decode.
    """
    name, key = _GATEWAY_SECRET
    encoded = _kubectl("get", "secret", name, "-o", f"jsonpath={{.data.{key}}}")
    assert encoded, (
        f"Secret {name} has no key {key}; the deployed gateway cannot verify"
    )
    return base64.b64decode(encoded)


_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080


def _register_a_session(base: str, image: str) -> tuple[SessionId, TenantId]:
    """An environment, a definition and a Session through the API, and their ids.

    This file places its own pod and does not need the control plane to do it -- but
    the pod's runtime dials the Tool Gateway during start-up, and the Gateway reads the
    Session's row to learn its Grant. A Session id this file invented has no row, the
    Gateway answers the MCP handshake with an error, and the runtime treats that server
    as required: `Failed to create session: required MCP servers failed to initialize`,
    and the pod exits 3 before it serves anything.

    So the Session is real and the placement is still this file's own. An empty Grant is
    correct here and is not what used to break it: the control plane's own cases create
    Sessions with `grant: []` and their pods start. What the Gateway refuses is a
    Session it has never heard of, which is the right refusal and the wrong input.
    """
    headers = {TENANT_HEADER: str(TenantId(uuid4()))}
    with httpx.Client(base_url=base, timeout=60.0, headers=headers) as caller:
        environment = caller.post(
            "/v1/environments",
            json={"name": f"placement-{uuid4().hex[:8]}", "runtime_image": image},
        )
        assert environment.status_code == 201, environment.text
        definition = caller.post(
            "/v1/agents",
            json={
                "name": f"placement-{uuid4().hex[:8]}",
                "instructions": "Reply with exactly: ok",
                "model": "gsds-claude-opus-4-6",
                "skills_repository": "git@github.com:acme/skills.git",
                "skills_revision": "0" * 39 + "a",
            },
        )
        assert definition.status_code == 201, definition.text
        session = caller.post(
            "/v1/sessions",
            json={
                "definition_id": definition.json()["id"],
                "environment_id": environment.json()["id"],
                "grant": [],
                "scope": {},
                "budget_minor_units": 500,
                "budget_currency": "USD",
                "retention_days": 30,
            },
        )
        assert session.status_code == 201, session.text
    return SessionId(UUID(session.json()["id"])), TenantId(UUID(headers[TENANT_HEADER]))


def _compiled(
    image: str, token_key: bytes, session_id: SessionId, tenant_id: TenantId
) -> config_compiler.CompiledConfig:
    """Documents from the production compiler, never hand-written.

    Hand-written TOML was tried and the runtime refused it with `thread/start failed
    with code -32600` -- an Invalid Request caused by the probe, not the platform. A
    copy of a compiled document is free to differ from what the compiler emits, and the
    difference is what the runtime rejects.
    """
    record = SessionRecord(
        id=session_id,
        tenant_id=tenant_id,
        definition_id=new_definition_id(),
        definition_revision="rev-1",
        grant=frozenset(),
        scope=(),
        budget_minor_units=10_000,
        budget_currency="USD",
        retention_days=1,
    )
    return config_compiler.compile_session_config(
        record,
        tool_gateway_url=f"http://tool-gateway.{NAMESPACE}.svc.cluster.local/mcp",
        model_gateway_url=f"http://model-gateway.{NAMESPACE}.svc.cluster.local/v1",
        definition=AgentDefinition(
            name="map-placement-check",
            instructions="Reply with exactly: ok",
            # The one model this account has actually deployed, and the reason this
            # probe can complete a Turn at all. `gpt-5-codex` stood here and routes to
            # `map/dev/providers/openai`, a vault entry that exists in no account: every
            # Turn died at the credential fetch with a 503, and the "does not complete"
            # assertion below was measuring an absent secret rather than the platform.
            model="gsds-claude-opus-4-6",
            skills_repository="git@github.com:acme/skills.git",
            skills_revision=SkillsRevision("0" * 39 + "a"),
        ),
        environment=Environment(
            id=new_environment_id(),
            tenant_id=record.tenant_id,
            name="map-placement-check",
            runtime_image=image,
            denied_paths=(),
        ),
        session_token_key=token_key,
        session_token_expiry_epoch_s=_EXPIRY,
    )


@contextmanager
def _placed() -> Iterator[tuple[str, config_compiler.CompiledConfig, str]]:
    """A Session pod placed by the real runner, deleted however this block exits.

    Yields the pod name, the configuration it was started from, and the phase `ensure`
    reported. Deletion is in a `finally` and removes the pod only: `_create` makes the
    pod own its three Secrets, so they go with it -- which is itself checked below
    rather than assumed.

    The runner signs with the cluster's own CA, from the same expression `shim_dial`
    reads to decide how to dial. Left to its default the runner holds no CA, places a
    pod serving plain HTTP, and the dial -- reading the deployed Secret, which does hold
    one -- speaks TLS at it. That mismatch surfaces as `record layer failure`, which
    names neither side of it.
    """
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id, tenant_id = _register_a_session(base, image)
    compiled = _compiled(image, _session_token_key(), session_id, tenant_id)
    runner = KubernetesPodRunner.from_manifest_file(
        _MANIFEST,
        namespace=NAMESPACE,
        token_key=os.urandom(32),
        internal_ca=internal_ca_of_the_cluster(),
    )
    pod_name = pod_name_for(compiled.session_id)
    try:
        phase = asyncio.run(runner.ensure(pod_name, compiled))
        yield pod_name, compiled, str(phase)
    finally:
        subprocess.run(
            ("kubectl", "-n", NAMESPACE, "delete", "pod", pod_name, "--wait=false"),
            capture_output=True,
            text=True,
        )


def _bearer(pod_name: str) -> str:
    """This Session's own shim token, out of the Secret the runner created.

    Read rather than re-derived: re-deriving it here would need this file to hold the
    signing key and the token format, and a second implementation of a bearer is a
    second thing that can be right about a token the pod refuses.
    """
    encoded = _kubectl(
        "get", "secret", f"{pod_name}-shim-token", "-o", "jsonpath={.data.token}"
    )
    assert encoded, f"the runner created no shim-token Secret for {pod_name}"
    return base64.b64decode(encoded).decode().strip()


def test_the_placement_code_puts_a_running_pod_in_the_namespace() -> None:
    """The whole objective's first half, and the thing nothing else here can answer.

    Asserts on the pod object rather than on any log line. `turns.py` appends
    `pod_unreachable` for every dispatch failure and drops the phase (ADR-013), so an
    event log cannot tell a control plane that placed a pod from one that placed
    nothing. The object is the only discriminator.
    """
    with _placed() as (pod_name, _compiled_config, phase):
        assert phase == "running", (
            f"ensure reported {phase!r} for {pod_name}. RUNNING means the shim "
            "opened its thread against the runtime and answered its readiness "
            "probe; anything else means it did not, and "
            "`kubectl logs -c session-shim` says why."
        )
        listed = _kubectl("get", "pod", pod_name, "-o", "jsonpath={.status.phase}")
        assert listed == "Running", f"{pod_name} is {listed!r} to the API server"
        ready = _kubectl(
            "get",
            "pod",
            pod_name,
            "-o",
            "jsonpath={range .status.containerStatuses[*]}{.name}={.ready} {end}",
        )
        assert "session-shim=true" in ready and "agent-runtime=true" in ready, (
            f"both containers must be ready, got {ready!r}"
        )


def test_the_pod_owns_its_secrets_so_deleting_it_leaves_nothing() -> None:
    """Three Secrets exist while the pod does, and none outlive it.

    `_create` sets the pod as owner as its third step. Untested, a rename of that step
    leaves one Secret per Session accumulating in the namespace forever, each holding a
    token and a compiled configuration -- which is a slow leak of exactly the material
    this platform is careful about elsewhere.
    """
    with _placed() as (pod_name, _config, _phase):
        during = _kubectl("get", "secrets", "-o", "name")
        mine = [line for line in during.splitlines() if pod_name in line]
        assert len(mine) == 3, f"expected 3 Secrets for {pod_name}, got {mine}"
    for _ in range(60):
        after = _kubectl("get", "secrets", "-o", "name")
        if not [line for line in after.splitlines() if pod_name in line]:
            break
        time.sleep(1)
    else:
        raise AssertionError(
            f"Secrets for {pod_name} outlived the pod, so the ownership step in "
            "_create no longer takes effect and every Session leaks three Secrets"
        )


def test_a_turn_is_accepted_and_streams_to_a_terminal_line() -> None:
    """The pod serves a Turn, calls a real model through the Gateway, and completes it.

    This is the whole platform on one request. The pod authenticates the Turn against a
    token the control plane would mint, the Agent Runtime inside it POSTs the Model
    Gateway, the Gateway translates a Responses body onto the Anthropic Messages wire,
    fetches the provider credential under its own IRSA identity, and Foundry answers --
    and the answer streams back out as events. Nothing here is stubbed and nothing is
    local: a failure at any of those hops is a failure of this test.

    It asserted the opposite until 2026-08-23 -- that the Turn does NOT complete -- with
    a docstring saying to tighten it rather than delete it once one did. This is that
    tightening. What made the difference was not one fix: the Gateway verified a token
    layout nothing minted, the routing table's Anthropic host did not resolve, its model
    named an undeployed deployment, the whole JSON vault entry went out as the
    credential, no handler was registered for the wire the table declared, the pod had
    no DNS record of its own, and this probe named a model whose credential exists in no
    account. Each of those was invisible to every other assertion in the tree.

    `turn.completed` and more than three lines, which is the pair the loose version
    could not tell apart from a broken platform.
    """
    with _placed() as (pod_name, compiled, phase):
        assert phase == "running", phase
        # The dial is decided from the cluster's own Secret, so this case follows the
        # platform into and out of mTLS instead of pinning a scheme. A forward lands on
        # `127.0.0.1` and the pod's certificate names it by its in-cluster address, so
        # the SNI name `shim_dial` carries is what makes verification pass -- against
        # the real name, not by turning the check off.
        dial = shim_dial(pod_name)
        with forwarded(f"pod/{pod_name}", _SHIM_PORT) as forward:
            base = dial.base(forward)
            token = _bearer(pod_name)
            body = {
                "session_id": str(compiled.session_id),
                "turn_id": str(uuid4()),
                "prompt": "Reply with exactly: ok",
            }
            lines: list[dict[str, Any]] = []
            with (
                httpx.Client(timeout=180.0, verify=dial.verify) as client,
                client.stream(
                    "POST",
                    f"{base}/session/turn",
                    json=body,
                    headers={"Authorization": f"Bearer {token}"},
                    extensions=dial.extensions,
                ) as response,
            ):
                assert response.status_code == 200, (
                    f"the shim refused the Turn with {response.status_code}: "
                    f"{response.read().decode()[:400]}"
                )
                for line in response.iter_lines():
                    if line:
                        lines.append(json.loads(line))

    types = [one.get("type") for one in lines if one.get("kind") == "event"]
    assert "turn.started" in types, (
        f"the pod accepted the Turn and never started it; lines were {lines}"
    )
    assert "turn.failed" not in types, (
        f"the Turn reached the model hop and failed there; lines were {lines}"
    )
    assert "turn.completed" in types, (
        f"the Turn started and never completed; lines were {lines}"
    )
    # More than three, because turn.started / turn.failed / completed is exactly three
    # and is what a Turn that never reached a model produced. A count alone would be a
    # weak assertion; paired with turn.completed above it says the model answered with
    # something rather than merely that the Turn ended well.
    assert len(lines) > 3, f"nothing came back from the model; lines were {lines}"
    assert lines[-1].get("kind") == "completed", (
        f"the stream did not end on a terminal line; last was {lines[-1]}"
    )
