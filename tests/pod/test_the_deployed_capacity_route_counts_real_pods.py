"""`GET /v1/capacity` answered by the deployed control plane, with real numbers in it.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed and
nothing here is evidence of anything on that run.

Two of this route's seven fields are reads against the Kubernetes API rather than
against the database: `session_pods_running` lists pods in the namespace, and
`nodes_schedulable` lists nodes. Both ship `null` when the read is refused, and until
this file existed both had only ever been seen as `null` -- the RBAC for them was
committed, graded in both directions by a manifest scan, and never once exercised by a
live call. The reason was not the RBAC. The route requires a platform reviewer, no test
held one against a deployment, and every attempt reached the fail-safe 401 instead.

**The counting case is the one that matters.** A route returning two hard-coded integers
passes every "is this an int" assertion ever written. So the pod count is read twice,
with a Session placed between the readings, and what is asserted is the *difference*.
`nodes_schedulable` cannot be moved that way -- adding a node is the autoscaler's
decision and takes a minute -- so it is graded against the ceiling beside it, which
comes from a different source entirely.

NO KEY OR TOKEN VALUE IS PRINTED OR ASSERTED ON. A reviewer token has to be minted to
reach this route at all, and the key it is minted with is read out of the deployed
Secret into a local and never rendered: not into an assertion message, not into a repr,
not into a failure. What the assertions read is a status code and six numbers.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from collections.abc import Iterator
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import NAMESPACE, forwarded, kubectl

from managed_agent.control.reviewers.token import mint_reviewer_token
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import SessionId

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_SECRET: Final = "map-control-plane"
_KEY_FIELD: Final = "shim-token-key"
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_TOKEN_LIFETIME_S: Final = 600
_TURN_DEADLINE_S: Final = 600
_SUBMIT_TIMEOUT_S: Final = 900
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

requires_the_cluster = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places a real pod and calls a real model",
)


def _reviewer_header() -> dict[str, str]:
    """A reviewer Authorization header, minted with the deployed key.

    The key is read out of the running Secret rather than passed in, because a key this
    test carried of its own would mint a token the deployment cannot verify -- and the
    resulting 401 is indistinguishable from the route being broken. It is decoded into a
    local and never rendered: the assertion below reports whether the field was empty,
    not what was in it.

    Minted here rather than by a fixture so the refusal case can run without one."""
    encoded = kubectl(
        "get", "secret", _SECRET, "-o", f"jsonpath={{.data.{_KEY_FIELD}}}"
    )
    key = base64.b64decode(encoded)
    assert key, f"{_SECRET} carries no {_KEY_FIELD}, so no reviewer can be minted"
    token = mint_reviewer_token(
        reviewer_id=uuid4(),
        expiry_epoch_s=int(time.time()) + _TOKEN_LIFETIME_S,
        key=key,
    )
    return {"authorization": f"Bearer {token}"}


def _capacity(base: str, header: dict[str, str]) -> dict[str, Any]:
    """One reading of the route, asserted 200 before it is returned.

    The status is checked here and not in each case, so a deployment that refuses the
    route fails once with its body attached rather than four times with a `KeyError`."""
    answered = httpx.get(f"{base}/v1/capacity", headers=header, timeout=60)
    assert answered.status_code == 200, answered.text
    body: dict[str, Any] = answered.json()
    return body


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned."""
    done = subprocess.run(
        (
            "aws",
            "ecr",
            "describe-images",
            "--repository-name",
            _REPOSITORY,
            "--region",
            _REGION,
            "--output",
            "json",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    details = json.loads(done.stdout)["imageDetails"]
    assert details, f"{_REPOSITORY} holds no images, so no Session pod can start"
    newest = max(details, key=lambda one: str(one["imagePushedAt"]))
    return (
        f"{newest['registryId']}.dkr.ecr.{_REGION}.amazonaws.com/"
        f"{_REPOSITORY}@{newest['imageDigest']}"
    )


def _created(answered: httpx.Response) -> dict[str, Any]:
    assert answered.status_code == 201, answered.text
    body: dict[str, Any] = answered.json()
    return body


def _place_one_session(base: str, tenant_id: str) -> SessionId:
    """One Session with one Turn submitted, returning its id. Does not wait.

    The submission is not awaited to completion because what this file counts is a pod
    that exists, and a pod exists as soon as placement is done -- waiting for the model
    to answer adds a round trip to every run for a number no case here reads. The Turn
    is submitted rather than skipped because placement is a first-Turn step: a Session
    created and never run has no pod to count.

    A one-word prompt for the same reason. Nothing reads the answer."""
    stamp = uuid4().hex[:8]
    caller = httpx.Client(
        base_url=base, timeout=_SUBMIT_TIMEOUT_S, headers={TENANT_HEADER: tenant_id}
    )
    with caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"capacity-{stamp}", "runtime_image": _session_image()},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"capacity-{stamp}",
                    "instructions": "Answer in one word.",
                    "model": _MODEL,
                    "skills_repository": "git@github.com:acme/skills.git",
                    "skills_revision": "0" * 39 + "a",
                    "skills": [],
                    "tool_servers": [],
                },
            )
        )
        session = _created(
            caller.post(
                "/v1/sessions",
                json={
                    "definition_id": definition["id"],
                    "environment_id": environment["id"],
                    "budget_minor_units": 500_000,
                    "budget_currency": "USD",
                    "retention_days": 1,
                },
            )
        )
        session_id = SessionId(UUID(session["id"]))
        submitted = caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": "Reply with the single word ready and nothing else."},
            headers={"Idempotency-Key": uuid4().hex},
        )
    assert submitted.status_code == 202, submitted.text
    return session_id


def _running(session_id: SessionId) -> bool:
    """Whether this Session's pod is in `Running`, by asking the cluster.

    Asked of the cluster rather than of the route under test. A poll that waited on
    `session_pods_running` to rise would be waiting for the number it then asserts,
    which passes on a route that returns a rising constant and fails on nothing.

    `check=False` so an absent pod reads as not-running rather than raising: this is
    called in a loop whose whole purpose is the window before the pod exists."""
    phase = kubectl(
        "get",
        "pod",
        pod_name_for(session_id),
        "-o",
        "jsonpath={.status.phase}",
        check=False,
    ).strip()
    return phase == "Running"


def _clean_up(session_id: SessionId) -> None:
    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


@pytest.fixture(scope="module")
def readings() -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """The route before a Session is placed and again once its pod is Running.

    Module-scoped, so one placement serves every case. The pair is the unit rather than
    two separate fixtures because the finding is the difference between them, and two
    fixtures could sample it either side of a pod that had already gone.

    The teardown deletes the pod and its Secrets in a `finally`, including when the wait
    above fails. A run that gave up on placement still created a Session the control
    plane places a pod for."""
    header = _reviewer_header()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        before = _capacity(base, header)
        session_id = _place_one_session(base, str(uuid4()))
        try:
            deadline = time.monotonic() + _TURN_DEADLINE_S
            while not _running(session_id) and time.monotonic() < deadline:
                time.sleep(2)
            assert _running(session_id), (
                f"no pod for session {session_id} reached Running in "
                f"{_TURN_DEADLINE_S}s, so nothing here can be counted"
            )
            yield before, _capacity(base, header)
        finally:
            _clean_up(session_id)


@requires_the_cluster
def test_both_cluster_reads_come_back_as_numbers(
    readings: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Neither cluster read is `null`, in either reading.

    `null` is what this route ships when the Kubernetes read is refused, and it is the
    only value either field had ever been observed to hold before this file: the RBAC
    was committed and graded by a manifest scan, and no live call had ever been made
    because the route needs a reviewer and no test held one.

    Both readings, not just the second. A route whose first read fails and whose second
    succeeds is a route that works only once something warmed it up, which is worth
    failing on rather than averaging over."""
    before, after = readings
    for reading in (before, after):
        assert isinstance(reading["session_pods_running"], int), reading
        assert isinstance(reading["nodes_schedulable"], int), reading


@requires_the_cluster
def test_a_placed_pod_is_counted_rather_than_a_constant_returned(
    readings: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The pod count went up by exactly one. This is the case that has teeth.

    Every other assertion in this file is satisfied by a route returning two hard-coded
    integers. This one is not, and `+ 1` rather than `>` is deliberate: a count that
    jumped by three while one pod was placed is counting something other than this
    namespace's Session pods, and `>` would call that agreement.

    Exact rather than tolerant, which makes this case sensitive to another run placing a
    pod at the same moment. That is accepted: the live suite is run by one person
    against one dev cluster, and a flake here is a true statement about what the cluster
    held."""
    before, after = readings
    assert after["session_pods_running"] == before["session_pods_running"] + 1, (
        before,
        after,
    )


@requires_the_cluster
def test_the_node_count_sits_under_the_ceiling_it_is_published_beside(
    readings: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The node count is a real number inside the bound published next to it.

    `nodes_schedulable` cannot be moved the way the pod count can -- adding a node is
    the autoscaler's decision and takes about a minute -- so what grades it is the
    ceiling beside it, which comes from a different source entirely: the count is a list
    call against the API server, the ceiling is read off the autoscaler's own
    configuration. Two numbers from two places agreeing is a weaker claim than a
    difference, and it is the claim available here.

    At least one, because a cluster answering zero schedulable nodes while a pod of ours
    is Running on one of them is reporting something impossible."""
    _, after = readings
    assert isinstance(after["node_ceiling"], int), after
    assert 1 <= after["nodes_schedulable"] <= after["node_ceiling"], after


@requires_the_cluster
def test_the_route_refuses_a_caller_who_is_only_a_tenant() -> None:
    """A tenant header is not a reviewer, and the route says so at 401.

    The fail-safe direction, live. This is also the case that explains the whole file:
    every earlier attempt to read this route against a deployment landed here, and a 401
    from a route that needs a credential looks exactly like a route that is broken. Its
    own mint is skipped, which is what keeps it independent of the fixture."""
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        refused = httpx.get(
            f"{base}/v1/capacity",
            headers={TENANT_HEADER: str(uuid4())},
            timeout=60,
        )

    assert refused.status_code == 401, refused.text
    assert refused.json()["error"]["code"] == "auth.audit_principal_unresolved"


@requires_the_cluster
def test_the_namespace_the_counts_are_taken_over_is_the_deployed_one() -> None:
    """The counts above are over `map-dev` and not some other namespace.

    A one-line control, and not a redundant one: every number in this file is a count
    over whatever namespace `cluster_access` is pointed at, and a run against a
    different one would produce the same shaped numbers and grade nothing about the
    deployment the rest of the suite is talking about."""
    assert NAMESPACE == "map-dev"
