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

**The second reading is taken while the Turn is still running, and it has to be.** A pod
is leased for exactly one Turn and given back when that Turn ends (ADR-041), and the API
holds the submission's response until then -- so `session_pods_running` is now the count
of Turns actually in flight, and outside that window there is no pod of ours to count.
This file used to submit the Turn, wait for the API to answer and then start looking for
a Running pod, which under the lease is looking for one the platform has already
deleted. The submission therefore runs on a thread that is not joined until the second
reading has been taken.

That is also the sharper claim. The field counts the same objects by the same
arithmetic and means something else for it: while a Session held a pod for its whole
life the number read as headroom, and under the lease every running pod is already
carrying a Turn, so it reads as utilisation. The difference this file asserts is
therefore one concurrent Turn rather than one more Session that could take one.

**Which is why the wait below is for RUNNING and must stay that way.** A pod that is
still starting belongs to a Turn that is queued, and that Turn is already counted in
`turns_awaiting_placement`; relaxing this to "the pod exists" would take the second
reading while the same Turn was being counted in two fields an operator reads against
each other, and the `+ 1` would then be measuring the race rather than the count.

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
from concurrent.futures import Future, ThreadPoolExecutor
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


def _register_a_session(base: str, tenant_id: str) -> SessionId:
    """An environment, a definition and a Session, over the REST API. Places nothing.

    Split from the submission below because the two now have to happen on different
    threads. Registering a Session creates no pod -- placement is a step a Turn pays for
    -- so this part can block the caller, and the part that cannot is the Turn.

    Everything here is through the API, which is what makes the pod counted later the
    deployed process's doing rather than this run's."""
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
        return SessionId(UUID(session["id"]))


_ENDED: Final = frozenset({"turn.completed", "turn.failed"})
"""Either of which means the Turn is over and its pod has been released."""


def _submit_a_turn(base: str, tenant_id: str, session_id: SessionId) -> None:
    """Submit one Turn and block until it has ended.

    Run on a thread by its only caller, and that is the whole shape of this file now.
    The lease deletes the pod when the Turn ends, so a caller that waited for the Turn
    on the main thread would have nothing left to count by the time it looked.

    **The block is this function's own polling, not the API's.** The route held its
    response until the dispatch returned once, and this simply returned the POST. It
    answers 202 the moment the Turn is admitted and runs it on a background task now, so
    a caller that returns on the POST reports the Turn finished while the pod is still
    being placed -- and the fixture's teardown then deletes the pod out from under the
    Turn, which reaches the log as `runtime_did_not_start: the pod ... is gone`. The
    Turn's end is only readable out of its Session's events.

    A one-word prompt because nothing reads the answer. What is being counted is that a
    pod is up and serving, and the cheapest Turn that gets one there is the right one to
    ask for."""
    caller = httpx.Client(
        base_url=base, timeout=_SUBMIT_TIMEOUT_S, headers={TENANT_HEADER: tenant_id}
    )
    with caller:
        submitted = caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": "Reply with the single word ready and nothing else."},
            headers={"Idempotency-Key": uuid4().hex},
        )
        assert submitted.status_code == 202, submitted.text
        deadline = time.monotonic() + _SUBMIT_TIMEOUT_S
        while time.monotonic() < deadline:
            answered = caller.get(f"/v1/sessions/{session_id}/events")
            assert answered.status_code == 200, answered.text
            seen = [one["type"] for one in answered.json()["events"]]
            if _ENDED.intersection(seen):
                return
            time.sleep(2)
        raise AssertionError(
            f"the Turn on session {session_id} did not end in {_SUBMIT_TIMEOUT_S}s; "
            f"its events were {seen}"
        )


def _running(session_id: SessionId) -> bool:
    """Whether this Session's pod is one the route would count, by asking the cluster.

    Asked of the cluster rather than of the route under test. A poll that waited on
    `session_pods_running` to rise would be waiting for the number it then asserts,
    which passes on a route that returns a rising constant and fails on nothing.

    **Every container ready, and not the pod phase alone.** The route counts what
    `_phase_of` calls RUNNING, which is the phase *and* every container ready -- a
    Session is served by the runtime and the shim together and neither serves a Turn
    alone. Reading the phase by itself made this poll a weaker condition than the thing
    it was gating, so it returned during the seconds when the pod says Running and its
    containers are still coming up, and the reading taken right after found the route
    counting nothing. Measured on `map-dev`: `session_pods_running` 0 alongside
    `sessions_placing` 1, which is the control plane still inside its placement wait and
    is the route being right rather than late.

    That gap was always there and used to be invisible: a pod outlived the Turn that
    placed it, so anything reading afterwards found it long since ready. Under the lease
    the pod exists only while its Turn does, and the window this poll opens is the
    whole of what there is to measure.

    The emptiness check is why this is not a bare `all()`. `all(())` is true, so a pod
    whose container statuses have not been reported yet -- which is every pod for its
    first moment -- would otherwise read as ready, which is the same bug one layer down.

    `check=False` so an absent pod reads as not-running rather than raising: this is
    called in a loop whose whole purpose is the window before the pod exists. Under the
    lease there is a window after it too -- the pod is deleted when its Turn ends -- so
    a False here is "not yet" or "not any more", and only the caller, which knows
    whether the Turn is still in flight, can tell those apart."""
    reading = kubectl(
        "get",
        "pod",
        pod_name_for(session_id),
        "-o",
        "jsonpath={.status.phase}|{.status.containerStatuses[*].ready}",
        check=False,
    ).strip()
    phase, _, readiness = reading.partition("|")
    ready = readiness.split()
    return phase == "Running" and bool(ready) and all(r == "true" for r in ready)


def _clean_up(session_id: SessionId) -> None:
    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


def _await_running_while_in_flight(session_id: SessionId, flying: Future[None]) -> None:
    """Block until this Session's pod is Running, while its Turn is still being carried.

    Polled against the submission and not against a clock alone, because the pod belongs
    to that Turn: it appears once the placement is done and is deleted the moment the
    Turn ends. A wait that outlived the submission would spend the rest of its deadline
    looking for a pod that is correctly gone, and then report the platform as having
    placed nothing.

    Fails here rather than handing back a verdict. If no pod of ours was ever Running
    there is nothing for the route to count, and the difference the cases below assert
    would be a difference between two readings of an unchanged cluster -- which is
    exactly what a route returning a constant produces."""
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        if _running(session_id):
            return
        if flying.done():
            # Whatever the submission did comes first. A Turn refused at admission and a
            # Turn whose pod never started leave the same absence here and are two
            # different people's problems; this re-raises the first, names the second.
            flying.result(timeout=0)
            pytest.fail(
                f"the Turn for session {session_id} ended before its pod was ever "
                "Running, so there was never a pod of ours here to count. The Turn was "
                "answered, so what failed is the pod starting rather than the Turn "
                "being accepted."
            )
        time.sleep(2)
    pytest.fail(
        f"no pod for session {session_id} reached Running in {_TURN_DEADLINE_S}s, so "
        "nothing here can be counted"
    )


@pytest.fixture(scope="module")
def readings() -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """The route before a Session takes a Turn, and again while that Turn is running.

    Module-scoped, so one Turn serves every case. The pair is the unit rather than two
    separate fixtures because the finding is the difference between them, and two
    fixtures could sample it either side of a pod that had already gone -- which under
    the lease is no longer a hypothetical: the pod goes away by itself, on the
    platform's schedule rather than on this file's.

    The submission is joined before anything is yielded, so no Turn is still running
    while the cases read. Otherwise the teardown would delete a pod out from under a
    Turn the platform was still carrying and this file would be the cause of a failure
    it then reported.

    The teardown deletes the pod and its Secrets in a `finally`, including when the wait
    above fails. Ordinarily the lease has already deleted all four; what this covers is
    the run that died mid-Turn, which leaves a pod nothing else will reap."""
    header = _reviewer_header()
    tenant_id = str(uuid4())
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        before = _capacity(base, header)
        session_id = _register_a_session(base, tenant_id)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                flying = pool.submit(_submit_a_turn, base, tenant_id, session_id)
                _await_running_while_in_flight(session_id, flying)
                after = _capacity(base, header)
                flying.result(timeout=_SUBMIT_TIMEOUT_S + 60)
            yield before, after
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
