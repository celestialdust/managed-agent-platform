"""The Tool Gateway as a running service: the factory, and then the pod.

Two tiers.

Tier 1 builds the app from a stubbed environment. It needs no database -- `build`
creates an `AsyncEngine`, which does not connect until something awaits a query, and
nothing here does -- and no cluster. What it proves is that the factory the manifest
names exists, returns an app serving the two paths the manifest and the compiled config
depend on, and refuses to start when either variable it needs is absent.

Tier 2 puts it in `map-dev` and asks another pod. It is skipped unless
`MAP_CLUSTER_TESTS=1`, the gate `tests/deploy/test_node_preconditions.py` and
`tests/deploy/test_session_sandbox_seccomp.py` already use. A SKIPPED RUN SAYS NOTHING
ABOUT THE CLUSTER.

Tier 2's two cases are a positive and a negative through the one probe mechanism, in the
one run, on purpose. A negative alone cannot tell a service that refused a call from a
probe that never reached one: both look like "no 200". The health case is what makes the
401 mean the middleware answered.

NOT PROVEN by either tier: that a tool call works. Nothing mints an `x-map-session`
token (`verify_session_token` has no signer in `src/`), the compiled config emits no
header for one, no tool credential exists in the vault under any prefix, and the prefix
the broker composes is not the prefix the role can read. Those are plan/MAP-64.md's
blockers 2 and 3 and they are why tier 2's positive case is the health path and its
negative case is a 401.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI
from starlette.routing import Route

from managed_agent import composition
from managed_agent.core import vocabulary
from managed_agent.core.vfs.session_vfs import SessionFiles
from managed_agent.core.vocabulary import tool_in_flight
from managed_agent.gateway.tool.rollout_seed import SessionRollouts
from managed_agent.gateway.tool.server import (
    MCP_PATH,
    GatewaySessions,
    create_gateway_app,
)

_ENVIRONMENT = {
    "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/none",
    "MAP_SESSION_TOKEN_KEY": "not-a-real-key",
    # Required here and resolved with `.get` in `build`, which is not a contradiction:
    # `build` serves the control plane too, where an absent bucket makes one upload
    # surface refuse legibly. This process captures Evidence for every tool call, and
    # an absent bucket there means the upstream runs, answers, and the capture then
    # fails -- so the model is told the tool broke and answers from memory. Measured on
    # a live Turn. The parametrized case below is what grades the difference.
    "MAP_OBJECT_BUCKET": "not-a-real-bucket",
    # Required here and resolved with `.get` in `build` for the same reason as the
    # bucket above, and the consequence of getting it wrong is the worst of the three.
    # Absent, this service would answer "no Rollout" to every pod seeding one, every
    # resuming Session would open a FRESH thread over a conversation that already
    # exists, and the platform would replay history the runtime's compaction
    # checkpoints have folded -- charging the tenant for the replay and reporting
    # success. The parametrized case below is what makes that a start-up refusal.
    "MAP_ROLLOUT_BUCKET": "not-a-real-bucket",
}

_GATE = "MAP_CLUSTER_TESTS"
_NAMESPACE = "map-dev"
_SERVICE_URL = f"http://tool-gateway.{_NAMESPACE}.svc.cluster.local"
_PROBE_POD = "map-toolgw-probe"


def test_the_factory_returns_an_app_serving_the_health_path_and_the_mcp_path() -> None:
    """Both paths, because the manifest depends on one and every Session depends on the
    other, and an app serving only the first passes every probe and answers no call."""
    with mock.patch.dict(os.environ, _ENVIRONMENT, clear=True):
        app = composition.tool_gateway_app()

    served = {route.path for route in app.routes if isinstance(route, Route)}
    assert MCP_PATH in served
    assert "/healthz" in served


@pytest.mark.parametrize("absent", sorted(_ENVIRONMENT))
def test_the_factory_raises_naming_the_variable_that_is_absent(absent: str) -> None:
    """No defaults, and the message says which one. A signing key defaulting to empty
    bytes would verify a token every pod could also mint, and the MCP route's only check
    would pass for anyone on the cluster network."""
    partial = {k: v for k, v in _ENVIRONMENT.items() if k != absent}
    with (
        mock.patch.dict(os.environ, partial, clear=True),
        pytest.raises(KeyError, match=absent),
    ):
        composition.tool_gateway_app()


def test_the_signing_key_is_not_the_shim_key() -> None:
    """Two keys for two hops in opposite directions. Setting only the shim's must not
    start this service, because one key for both would make either side's compromise the
    other's."""
    with (
        mock.patch.dict(
            os.environ,
            {
                "DATABASE_URL": _ENVIRONMENT["DATABASE_URL"],
                # Present so that it is not what raises: this case is about the two
                # signing keys, and a refusal naming the bucket would pass the assertion
                # below for the wrong reason while proving nothing about either key.
                "MAP_OBJECT_BUCKET": _ENVIRONMENT["MAP_OBJECT_BUCKET"],
                "MAP_SHIM_TOKEN_KEY": "k",
            },
            clear=True,
        ),
        pytest.raises(KeyError, match="MAP_SESSION_TOKEN_KEY"),
    ):
        composition.tool_gateway_app()


def test_the_three_event_types_come_from_the_published_set() -> None:
    """Not from three strings in composition.py. A service holding its own copies can
    emit a name the closed set does not carry, which is the one thing a closed set
    prevents.

    The source scan is the half that catches a regression the wiring cannot: a factory
    that stopped importing the module and wrote the strings inline would still hand the
    Gateway three correct names today, and would keep handing them after the published
    set moved.
    """
    for name in (
        tool_in_flight.TOOL_PROGRESS,
        tool_in_flight.TOOL_ELICITATION_REQUESTED,
        tool_in_flight.TOOL_ELICITATION_ANSWERED,
    ):
        assert vocabulary.is_published(name), name

    source = Path(str(composition.__file__)).read_text()
    for literal in (
        "tool.progress",
        "tool.elicitation_requested",
        "tool.elicitation_answered",
    ):
        assert literal not in source, f"{literal} is a second copy of a published name"


def test_the_names_the_factory_hands_the_gateway_are_the_published_ones() -> None:
    """Read off the `GatewaySessions` the factory really built.

    The source scan above proves the strings are not written in the file; this proves
    the published ones reached the Gateway. Neither is enough alone -- a factory could
    import the module, hand in three empty strings, and pass the scan -- and this is the
    half the checkpoint names, because what the service emits is what it was handed.

    `create_gateway_app` is wrapped rather than replaced: the real one still runs on the
    real object, and the wrapper only keeps a reference to the argument the factory
    chose to pass. There is no accessor for it, which is right for a service that has no
    reason to publish its own event vocabulary at runtime.
    """
    real = create_gateway_app
    passed: list[GatewaySessions] = []

    def capture(
        sessions: GatewaySessions,
        token_key: bytes,
        files: SessionFiles,
        rollouts: SessionRollouts,
    ) -> FastAPI:
        passed.append(sessions)
        return real(sessions, token_key, files, rollouts)

    with (
        mock.patch.dict(os.environ, _ENVIRONMENT, clear=True),
        mock.patch.object(composition, "create_gateway_app", capture),
    ):
        composition.tool_gateway_app()

    assert len(passed) == 1
    types = passed[0]._types
    assert (
        types.progress,
        types.elicitation_requested,
        types.elicitation_answered,
    ) == (
        tool_in_flight.TOOL_PROGRESS,
        tool_in_flight.TOOL_ELICITATION_REQUESTED,
        tool_in_flight.TOOL_ELICITATION_ANSWERED,
    )


requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"the cluster proof is opt-in: set {_GATE}=1 to run it. It needs kubectl "
        "against map-dev, the Deployment applied by `deploy/platform.py "
        "tool-gateway`, and it creates and deletes one probe pod. SKIPPED MEANS THE "
        "CLUSTER WAS NOT CONSULTED."
    ),
)


def _kubectl(*argv: str, stdin: str | None = None) -> str:
    done = subprocess.run(
        ("kubectl", "-n", _NAMESPACE, *argv),
        input=stdin,
        capture_output=True,
        text=True,
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


def _probe(path: str, method: str = "GET") -> tuple[int, str]:
    """Ask the Service from inside the cluster, using the image that is already there.

    The probe pod runs the platform image, because httpx is a runtime dependency of it
    -- so this needs no second image, no curl, and no registry the cluster cannot reach.
    It is deleted first and after, and `--wait=false` on the last delete so a slow
    teardown does not fail a case that already answered.

    The answer is parsed out of the pod's own stdout as JSON. A pod that failed to
    start, or an httpx exception, therefore fails this rather than reading as a status
    of zero.
    """
    image = json.loads(_kubectl("get", "deploy", "tool-gateway", "-o", "json"))["spec"][
        "template"
    ]["spec"]["containers"][0]["image"]
    program = (
        "import httpx,sys,json;"
        f"r=httpx.request({method!r},{_SERVICE_URL + path!r},timeout=10);"
        "print(json.dumps({'status':r.status_code,'body':r.text[:200]}))"
    )
    _kubectl("delete", "pod", _PROBE_POD, "--ignore-not-found")
    _kubectl(
        "run",
        _PROBE_POD,
        "--image",
        image,
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


@requires_the_cluster
def test_the_deployment_is_available_at_its_own_replica_count() -> None:
    """`kubectl rollout status` exits 0 for a Deployment scaled to zero, so the count is
    read off the object rather than off an exit code."""
    described = json.loads(_kubectl("get", "deploy", "tool-gateway", "-o", "json"))
    desired = described["spec"]["replicas"]
    assert desired > 0
    assert described["status"].get("availableReplicas") == desired, described["status"]


@requires_the_cluster
def test_a_pod_in_the_namespace_gets_two_hundred_from_the_health_path() -> None:
    """Through the Service name and its port 80, which is what an in-cluster caller uses
    -- not the container's 8080. A test that reached the pod IP would pass with the
    Service misconfigured.

    This is also the positive control for the case below: without it, a 401 could be a
    refusal or could be a probe that never reached the Service at all.
    """
    status, _ = _probe("/healthz")
    assert status == 200


@requires_the_cluster
def test_a_call_with_no_session_token_is_refused_at_the_front_door() -> None:
    """The strongest thing reachable today, and it is a negative. It proves the
    middleware is on the wired path in the real process rather than only in a unit test:
    a Gateway that had been wired without it would answer this with an MCP protocol
    error instead of a 401.

    The body is asserted too, because a 401 from an ingress, a proxy or a sidecar would
    satisfy the status alone. This one is the fixed body gateway/tool/server.py writes,
    which is deliberately not an ErrorEnvelope -- a pod reads it, not a tenant.
    """
    status, body = _probe(MCP_PATH, method="POST")
    assert status == 401
    assert "invalid session token" in body
