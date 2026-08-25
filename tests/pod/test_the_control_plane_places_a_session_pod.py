"""A Session pod, in the real map-dev namespace, created by the deployed control plane.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NO POD WAS PLACED -- every other check
over this path reads YAML or drives a fake runner, and neither can say whether the API
server accepted a create from this ServiceAccount.

Nothing here creates a pod itself. It drives the REST API through a port-forward to the
deployed control plane, exactly as a tenant would, and then looks in the namespace for
an object whose name is derived from the Session id it was handed. That is what "the
control plane put it there" means, and it is why the name is computed rather than
searched for: a leftover `map-session-*` from another run would satisfy a search.

Two findings here are absences, and each has a control in the same run. That no SECOND
pod appears is paired with the first Turn, where one MUST appear. That the Tool Gateway
accepts the token the control plane signed is paired with a token minted here under a
throwaway key, which MUST be refused -- an endpoint that accepts everything and an
endpoint that verifies are indistinguishable from one probe.

What this file does not show is a Turn *finishing*. It waits for the pod object and not
for RUNNING, so every case here costs one placement and no image pull and no model call.
Whether a Turn completes -- and whether two tenants' Turns stay apart -- is
`test_two_tenants_run_at_once_through_the_deployed_api.py`, which is built to wait.

Three of these cases asserted the opposite until 2026-08-23: that the Turn ends in
`turn.failed` and the API answers 502. That was correct, and for a reason that has gone
away. The Environment named an image that resolves nowhere, on the reasoning that a pod
does not have to start for an object's existence to be decided -- and then the deployed
control plane started actually running the placement code, which DELETES a pod that will
not start along with its three Secrets rather than leaving a Session holding an object
kubelet retries for ever. So the fake image left nothing here to look at, and three
cases failed naming causes that were not the cause. A real image is resolved at run time
now and the cost that old comment named is accepted: this run depends on ECR holding a
Session image.

NO VALUE OF ANY KEY IS PRINTED OR ASSERTED ON. The Session token is lifted out of a
Secret and handed to an HTTP client; the throwaway key is this file's own and is
compared to nothing. Every assertion here is on a status code, a name or a count.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import tomllib
from collections.abc import Iterator
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import NAMESPACE, forwarded, kubectl

from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER_NAME,
    mint_session_token,
)

_GATE: Final = "MAP_CLUSTER_TESTS"
_CONTROL_PLANE_PORT: Final = 8080
_TENANT: Final = "11111111-1111-4111-8111-111111111111"

_REGION: Final = "us-east-1"
_REPOSITORY: Final = "map/session-shim"


def _session_image() -> str:
    """The newest digest in the Session repository, resolved at run time.

    This was a digest-shaped reference to `registry.map.internal` that was never pulled,
    on the reasoning that the pod did not have to start for anything here to be decided.
    That reasoning stopped being true on 2026-08-23, and the way it stopped is worth
    keeping: the placer deletes a pod that will not start, together with its three
    Secrets, so that a Session is never left holding an object kubelet will retry for
    ever. Once the deployed control plane actually ran that code, an image that resolves
    nowhere left NOTHING for this file to look at -- the pod object, the Secrets and the
    Session token all went away microseconds before the assertions read for them, and
    three cases failed while reporting causes that were not the cause.

    So a real image, and the cost the old comment named is real and accepted: this run
    now depends on ECR holding a Session image. It is resolved rather than pinned for
    the same reason as in the two files beside this one -- a digest typed into a test
    pins the run to whatever the registry held that day.
    """
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


_PLACEMENT_DEADLINE_S: Final = 120.0
"""How long the placement behind a 202 is given to produce a pod object.

Covers the compile, three Secret creates and one pod create against the real API server,
plus whatever the control plane's own queue is doing. It does NOT cover an image pull --
nothing here waits for RUNNING -- so a value in minutes would only ever be spent on a
control plane that is not going to place at all.
"""

_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")
"""The three per-Session Secrets the placer writes, one per secret volume it mounts.

Asserted alongside the pod rather than instead of it. A pod with no Secrets is what a
create half-succeeding against a Role that grants pods and not secrets looks like, and
an existence check on the pod alone would call that a pass.
"""

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"the placement proof is opt-in: set {_GATE}=1 to run it. It needs kubectl "
        "pointed at map-dev and drives the deployed control plane over a port-forward. "
        "It creates one Session pod and its three Secrets, and deletes them. "
        "SKIPPED MEANS NO POD WAS PLACED."
    ),
)

pytestmark = [pytest.mark.network, requires_the_cluster]


# ---------------------------------------------------------------------------
# Cluster plumbing
# ---------------------------------------------------------------------------


def _exists(kind: str, name: str) -> bool:
    """Whether the namespace holds that object right now.

    A `get` on one name rather than a listing filtered afterwards, so this cannot answer
    yes because something else in the namespace shares a prefix.
    """
    done = subprocess.run(
        ["kubectl", "-n", NAMESPACE, "get", kind, name, "-o", "name"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return done.returncode == 0


def _pod_uid(name: str) -> str | None:
    """The pod's UID, or None if no pod of that name exists.

    A UID and not a name, because a name cannot answer the question. `pod_name_for` is a
    function of the Session id, so a placement that deleted the pod and made another
    would produce the same name twice and a name comparison would call that idempotent.
    The UID is assigned by the API server per object and changes when the object does.

    This replaced a helper that collected every `map-session-*` pod in the namespace and
    compared the whole set before and after. That reading was wrong in a way worth
    naming: it fails whenever anything else is running a Session, which on a platform
    whose objective is serving many tenants at once is the normal condition rather than
    interference. It failed exactly that way the first time a second cluster case ran in
    the same session.
    """
    listed = kubectl(
        "get", "pod", name, "-o", "jsonpath={.metadata.uid}", check=False
    ).strip()
    return listed or None


def _session_token_of(session_id: SessionId) -> str:
    """The `x-map-session` header out of that Session's own compiled Secret.

    Read out of the object the control plane wrote, because that is the only place the
    token it signed exists -- the compiler mints it and hands it straight to the Secret.
    What comes back is passed to an HTTP client and to nothing else: never printed,
    never logged, never compared to a key.
    """
    encoded = kubectl(
        "get",
        "secret",
        f"{pod_name_for(session_id)}-compiled",
        "-o",
        r"jsonpath={.data.config\.toml}",
    )
    assert encoded, "the compiled Secret carries no config.toml"
    document = tomllib.loads(base64.b64decode(encoded).decode())
    for server in document.get("mcp_servers", {}).values():
        headers = server.get("http_headers", {})
        if SESSION_TOKEN_HEADER_NAME in headers:
            token: str = headers[SESSION_TOKEN_HEADER_NAME]
            return token
    pytest.fail(f"no {SESSION_TOKEN_HEADER_NAME} in the compiled document")


# ---------------------------------------------------------------------------
# Driving the API exactly as a tenant would
# ---------------------------------------------------------------------------


def _client(base: str, timeout: int = 60) -> httpx.Client:
    return httpx.Client(
        base_url=base, timeout=timeout, headers={TENANT_HEADER: _TENANT}
    )


def _register_a_session(base: str) -> SessionId:
    """An environment, an agent definition and a Session, through the REST API only.

    Nothing here touches the database or the cluster directly. That is the whole shape
    of this file's claim: the objects it then finds in `map-dev` were put there by the
    deployed process, because this run has no other way to have created them.
    """
    with _client(base) as caller:
        environment = caller.post(
            "/v1/environments",
            json={
                "name": f"placement-{uuid4().hex[:8]}",
                "runtime_image": _session_image(),
            },
        )
        assert environment.status_code == 201, environment.text
        definition = caller.post(
            "/v1/agents",
            json={
                "name": f"placement-{uuid4().hex[:8]}",
                "instructions": "this agent never runs; the pod is the finding",
                # The model this account has actually deployed. `gpt-5-codex` stood
                # here and routes to a vault entry that exists in no account, which
                # fails a Turn at the credential fetch -- invisible while no pod could
                # start, and the whole finding once one could.
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
        return SessionId(UUID(session.json()["id"]))


def _take_a_turn(base: str, session_id: SessionId) -> httpx.Response:
    """Submit one Turn and return whatever the API answered.

    202: the API admits a Turn and dispatches it behind the response, so the outcome
    arrives in the Event Log rather than on this connection. This used to expect 502 --
    a pod from an image that resolves nowhere is never reachable -- and the expectation
    was correct for as long as the placement never worked. It is a placement failure
    now, not an ordinary state, so it is asserted at the call sites rather than here.
    """
    with _client(base, timeout=300) as caller:
        return caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": "Reply with exactly one word: acknowledged"},
            headers={"Idempotency-Key": uuid4().hex},
        )


def _event_types(base: str, session_id: SessionId) -> list[str]:
    with _client(base) as caller:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
    assert answered.status_code == 200, answered.text
    events: list[dict[str, Any]] = answered.json()["events"]
    return [event["type"] for event in events]


def _clean_up(session_id: SessionId) -> None:
    """Delete the pod and its three Secrets. The namespace is left as it was found."""
    pod = pod_name_for(session_id)
    kubectl("delete", "pod", pod, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{pod}-{suffix}", "--ignore-not-found", check=False
        )


# ---------------------------------------------------------------------------
# One run, read four ways
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def placed() -> Iterator[tuple[str, SessionId, httpx.Response]]:
    """One Session, one Turn, and the port-forward the cases below keep reading through.

    Module-scoped because placing a pod costs a compilation, three Secret creates and a
    pod create against the real API server, and because these cases are four readings of
    ONE run rather than four runs: "no second pod appeared" only means anything about
    the same Session the first Turn placed one for.
    """
    with forwarded("deploy/control-plane", _CONTROL_PLANE_PORT) as base:
        session_id = _register_a_session(base)
        try:
            answered = _take_a_turn(base, session_id)
            _await_the_pod(session_id, answered)
            yield base, session_id, answered
        finally:
            _clean_up(session_id)


def _await_the_pod(session_id: SessionId, answered: httpx.Response) -> None:
    """Block until the Session's pod object exists, or fail saying it never did.

    Necessary because the API answers 202 and places behind the response. Every case
    below reads objects in the namespace, and without this they race the dispatch: the
    pod would be absent for the first few hundred milliseconds and "the control plane
    placed nothing" is what that looks like. It did not use to race, because the old
    expectation was a synchronous 502 that only came back after the placement had been
    attempted and cleaned up.

    Waits for the object and not for RUNNING. The object is what these cases are about,
    and requiring RUNNING would make every one of them pay an image pull for a fact that
    another file grades against a Turn that completes.
    """
    deadline = time.monotonic() + _PLACEMENT_DEADLINE_S
    pod = pod_name_for(session_id)
    while time.monotonic() < deadline:
        if _pod_uid(pod) is not None:
            return
        time.sleep(1)
    pytest.fail(
        f"the API answered {answered.status_code} and no pod {pod} appeared in "
        f"{_PLACEMENT_DEADLINE_S}s. A control plane that admits a Turn and places "
        "nothing is the state this file exists to end."
    )


def test_the_pod_exists_and_the_control_plane_is_what_created_it(
    placed: tuple[str, SessionId, httpx.Response],
) -> None:
    """The whole slice, in one object's existence.

    Before this slice every deployment change could be in place and this would still
    fail, because nothing in `src/` called `Placement.place` or `compile_session_config`
    at all. The name is `pod_name_for(session_id)` computed from the id the API just
    returned, so nothing this run did not cause can satisfy it.

    The three Secrets are asserted with the pod rather than in a case of their own: a
    pod whose secret volumes have nothing behind them can never mount, kubelet retries
    for ever, and an existence check on the pod alone would call that a pass.
    """
    _, session_id, _ = placed
    pod = pod_name_for(session_id)
    assert _exists("pod", pod), (
        f"no pod {pod} in {NAMESPACE}. The control plane accepted the Session and the "
        "Turn and placed nothing, which is the state this slice exists to end."
    )
    missing = [s for s in _SECRET_SUFFIXES if not _exists("secret", f"{pod}-{s}")]
    assert not missing, (
        f"the pod exists and {missing} do not. Its secret volumes have nothing behind "
        "them, so kubelet retries the mount for ever and the pod can never start."
    )


def test_the_turn_is_admitted_and_recorded_before_anything_is_dispatched(
    placed: tuple[str, SessionId, httpx.Response],
) -> None:
    """The Turn is admitted, and the log records the admission independently of it.

    Two facts, and the second is why this is worth a case. The API answers 202, and the
    Event Log holds `session.created` then `turn.submitted` -- so the admission is
    durable rather than only a status code on one connection. A control plane that
    answered 202 and wrote nothing would pass a status-code assertion and lose the Turn.

    Nothing here waits for an outcome. Whether the Turn completes is graded in
    `test_two_tenants_run_at_once_through_the_deployed_api.py`, which is built to wait;
    asserting it here would make every case in this file pay a model round trip for a
    finding another file already carries.

    This asserted 502 and `turn.failed` until 2026-08-23, correctly, for as long as the
    Environment named an image that resolves nowhere. That is a placement failure now
    and not the ordinary path.
    """
    base, session_id, first = placed
    assert first.status_code == 202, first.text
    types = _event_types(base, session_id)
    assert types[0] == "session.created", types
    assert "turn.submitted" in types, types


def test_the_tool_gateway_accepts_the_token_the_control_plane_signed(
    placed: tuple[str, SessionId, httpx.Response],
) -> None:
    """Ask the process that decides, rather than compare two spellings.

    The control plane signs a Session's compiled document with `MAP_SESSION_TOKEN_KEY`
    and the Tool Gateway verifies with the variable of the same name. Whether both
    resolve to the same BYTES is not a question any YAML comparison can answer: two
    references can name one `(Secret, key)` while the cluster holds something
    unexpected behind it, and only the verifier knows.

    The control in the same run is a token minted here under a key this file invented,
    which must be refused. Without it, an endpoint that accepts everything and an
    endpoint that verifies are indistinguishable from one probe.
    """
    _, session_id, _ = placed
    signed = _session_token_of(session_id)
    forged = mint_session_token(
        session_id=session_id,
        tenant_id=TenantId(UUID(_TENANT)),
        expiry_epoch_s=int(time.time()) + 3600,
        key=b"a key this test invented; the cluster has never seen it",
    )
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    accept = "application/json, text/event-stream"
    with (
        forwarded("svc/tool-gateway", 80) as gateway,
        httpx.Client(base_url=gateway, timeout=60) as caller,
    ):
        refused = caller.post(
            "/mcp",
            json=body,
            headers={SESSION_TOKEN_HEADER_NAME: forged, "Accept": accept},
        )
        accepted = caller.post(
            "/mcp",
            json=body,
            headers={SESSION_TOKEN_HEADER_NAME: signed, "Accept": accept},
        )
    assert refused.status_code == 401, (
        "the Tool Gateway accepted a token signed with a key it has never seen, so "
        f"nothing below this line proves anything: {refused.status_code}"
    )
    assert accepted.status_code != 401, (
        "the Tool Gateway refused the token the control plane signed. Either the two "
        "MAP_SESSION_TOKEN_KEY references name one (Secret, key) and the bytes behind "
        "them disagree, or the control plane read a different Secret than its manifest "
        "says it does."
    )


def test_a_second_turn_reuses_the_pod_the_first_one_placed(
    placed: tuple[str, SessionId, httpx.Response],
) -> None:
    """One Session, one pod object, whatever a caller does.

    The branch under test is idempotence: `ensure_for` finds the pod it made a moment
    ago rather than making another. A placement that ignored what already exists would
    replace the object, and the replacement would carry the SAME NAME -- `pod_name_for`
    is a function of the Session id -- so this compares UIDs, which the API server
    assigns per object.

    ADR-004's other rule, that a Session which HAS completed a Turn never gets a fresh
    pod, is graded in the Tier-1 file. This Session has completed none.

    Scoped to this Session's own pod. It used to compare every `map-session-*` pod in
    the namespace before and after, which is a reading that fails whenever any other
    Session is running -- the normal condition on the platform this is meant to grade.
    """
    base, session_id, _ = placed
    pod = pod_name_for(session_id)
    before = _pod_uid(pod)
    assert before is not None, f"no pod {pod} before the second Turn"
    second = _take_a_turn(base, session_id)
    assert second.status_code == 202, second.text
    after = _pod_uid(pod)
    assert after == before, (
        f"a second Turn replaced the pod object for {session_id}: uid {before} -> "
        f"{after}. One Session has one pod, whatever a caller does."
    )
