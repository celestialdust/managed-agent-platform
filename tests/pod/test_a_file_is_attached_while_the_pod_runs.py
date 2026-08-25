"""A file attached to a Session that is already running, and the agent reads it.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

`POST /v1/sessions/{id}/resources` has a branch no offline test can reach. A Session
that has completed a Turn holds a pod, so the route pushes the bytes down the
authenticated hop before it appends anything -- and `build` wires the real placement
object only where a pod runner is configured, which an offline test never has. Every
offline case is therefore the no-pod branch or the refusal that proves the push is
attempted first. This is the case where a file actually arrives.

**The witness is the agent, because nothing else can be.** The pod has no exec and no
attach -- there are zero call sites for either in `src/` -- and its own listing route
serves the output directory rather than the input one. So the only reader of `./files/`
outside the container is the model inside it, and the proof that bytes arrived is a
token the model could not have produced any other way: the nonce is generated here, is
in the file's *contents* and not its name, and the file itself is uploaded after the
first Turn has already ended.

One Session, two Turns, module-scoped so the pair runs once. Turn one exists to produce
the pod and the `turn.completed` the route branches on; it also reads the file the
Session was created with, so a failure in the second Turn cannot be blamed on file
delivery in general. Turn two reads only the file that arrived while the pod was
running.

NO KEY OR TOKEN VALUE IS PRINTED OR ASSERTED ON.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import forwarded, kubectl

from managed_agent.control.api.routes.resources import (
    FILES_MOUNT_PATH,
    REASON_MOUNT_PATH_FIXED,
)
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import SessionId

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_TURN_DEADLINE_S: Final = 600
_SUBMIT_TIMEOUT_S: Final = 900
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

requires_the_cluster = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places real pods and calls a real model",
)


def _nonce(prefix: str) -> str:
    """Upper-case hex behind a word, because the model has to reproduce it exactly."""
    return f"{prefix}-{uuid4().hex[:12].upper()}"


def _doc(nonce: str) -> bytes:
    """The document the agent must read, with the nonce inside it and not in its name.

    Not in the name on purpose: the file name is in the prompt, so a nonce there would
    be quotable by an agent that never opened the file."""
    return (
        "# Field note\n\n"
        "This document exists so that one token can only be quoted by something that "
        "opened it.\n\n"
        f"    reference code: {nonce}\n"
    ).encode()


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned.

    Newest-push, matching the files beside this one: a digest written in here would pin
    the run to whatever ECR held on the day somebody typed it."""
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


def _client(base: str, tenant_id: str, timeout: int = 90) -> httpx.Client:
    return httpx.Client(
        base_url=base, timeout=timeout, headers={TENANT_HEADER: tenant_id}
    )


def _created(answered: httpx.Response) -> dict[str, Any]:
    assert answered.status_code == 201, answered.text
    body: dict[str, Any] = answered.json()
    return body


def _upload(base: str, tenant_id: str, name: str, nonce: str) -> dict[str, Any]:
    """Upload one document. 200 and 201 both accepted, matching the file beside this.

    The upload route's status is not what any assertion here is about, and pinning it to
    one value would make this fixture fail for a reason no test in this file names.
    """
    with _client(base, tenant_id) as caller:
        answered = caller.post(
            "/v1/files", files={"file": (name, _doc(nonce), "text/markdown")}
        )
    assert answered.status_code in (200, 201), answered.text
    body: dict[str, Any] = answered.json()
    return body


def _prompt(name: str) -> str:
    """One demand, quotable, and nothing else asked for.

    "Reply with nothing else" because the assertion is a substring check and the failure
    that matters is the token being absent -- extra prose costs tokens and adds nothing
    a reader of a failure would use."""
    return (
        f"Read the file ./files/{name} and quote the reference code it contains, "
        "exactly as written. Reply with nothing else."
    )


def _submit(base: str, tenant_id: str, session_id: SessionId, prompt: str) -> None:
    with _client(base, tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
        answered = caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": prompt},
            headers={"Idempotency-Key": uuid4().hex},
        )
    assert answered.status_code == 202, answered.text


def _events(base: str, tenant_id: str, session_id: SessionId) -> list[dict[str, Any]]:
    with _client(base, tenant_id) as caller:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
    assert answered.status_code == 200, answered.text
    listed: list[dict[str, Any]] = answered.json()["events"]
    return listed


def _await_terminal(
    base: str, tenant_id: str, session_id: SessionId, already: int
) -> list[dict[str, Any]]:
    """Poll until one more Turn has closed than had closed before, either way.

    Counted rather than matched on presence, because this Session runs two Turns: a wait
    for "any terminal event" returns immediately on the second call, having seen the
    first Turn's completion, and the second Turn's answer would be read out of a log it
    is not in yet.

    Both terminal types, not only the good one. A poll waiting for `turn.completed`
    alone sits out its whole deadline on a Turn that failed in the first second and then
    reports a timeout, which sends the reader after the wrong thing."""
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        closed = [
            one for one in events if one["type"] in ("turn.completed", "turn.failed")
        ]
        if len(closed) > already:
            return events
        time.sleep(3)
    events = _events(base, tenant_id, session_id)
    pytest.fail(
        f"session {session_id} closed no turn beyond {already} in "
        f"{_TURN_DEADLINE_S}s; the log was {[one['type'] for one in events]}"
    )


def _said_since(events: list[dict[str, Any]], after_seq: int) -> str:
    """Everything the agent said after a given sequence, as one string.

    Bounded below by the first Turn's last sequence, which is what keeps the two Turns'
    answers apart. Without it the second assertion passes on a Session whose second Turn
    said nothing at all, because the first Turn's deltas are still in the log.

    Assembled from the deltas rather than read off `turn.completed`, so what is graded
    is what the tenant streamed."""
    return "".join(
        str(one["payload"].get("text", ""))
        for one in events
        if one["type"] == "turn.message_delta" and int(one["seq"]) > after_seq
    )


def _highest_seq(events: list[dict[str, Any]]) -> int:
    return max(int(one["seq"]) for one in events)


def _pod_uid(session_id: SessionId) -> str:
    """The cluster's own identity for this Session's pod, or "" if there is none.

    A uid rather than a name, because the name is derived from the Session id and so is
    identical across a delete and a re-place -- which is the one thing this has to be
    able to tell apart. A uid is minted per object and never reused.

    Empty string for an absent pod rather than a raise: `jsonpath` on a missing object
    exits non-zero, and this is asked twice around a step whose failure should be
    reported by the assertion that reads it rather than by the read.
    """
    return kubectl(
        "get",
        "pod",
        pod_name_for(session_id),
        "-o",
        "jsonpath={.metadata.uid}",
        check=False,
    ).strip()


def _clean_up(session_id: SessionId) -> None:
    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


@dataclass(frozen=True, slots=True)
class _Run:
    base: str
    tenant_id: str
    session_id: SessionId
    created_with: dict[str, Any]
    attached: dict[str, Any]
    created_nonce: str
    attached_nonce: str
    attach_response: httpx.Response
    first_turn_said: str
    second_turn_said: str
    pod_before_attach: str
    pod_after_second_turn: str
    definition_id: str
    environment_id: str
    listed: list[dict[str, Any]]


@pytest.fixture(scope="module")
def run() -> Iterator[_Run]:
    """One Session, two Turns, one attach between them. Module-scoped.

    The order is the whole fixture. The file attached in the middle is *uploaded* in the
    middle too, after the first Turn has already ended -- so its bytes did not exist
    when the pod was created and cannot have arrived with it, and the only path they
    could have taken is the hop the attach route pushes down.

    The teardown deletes the pod and its Secrets whatever happened, including a failure
    during submission. A run that died there still created a Session the control plane
    places a pod for, and three aborted runs once left forty-two pods squatting the
    namespace, after which the next run's scheduling refusal read as the cluster being
    out of capacity."""
    tenant_id = str(uuid4())
    stamp = uuid4().hex[:8]
    created_nonce = _nonce("BEFORE")
    attached_nonce = _nonce("AFTER")
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        first = _upload(base, tenant_id, "brief.md", created_nonce)
        with _client(base, tenant_id) as caller:
            environment = _created(
                caller.post(
                    "/v1/environments",
                    json={"name": f"attach-{stamp}", "runtime_image": image},
                )
            )
            definition = _created(
                caller.post(
                    "/v1/agents",
                    json={
                        "name": f"attach-{stamp}",
                        "instructions": (
                            "Files attached to your Session are in ./files/ relative "
                            "to your working directory. Read what you are asked to "
                            "read and quote exactly."
                        ),
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
                        "file_ids": [first["id"]],
                        "budget_minor_units": 500_000,
                        "budget_currency": "USD",
                        "retention_days": 1,
                    },
                )
            )
        session_id = SessionId(UUID(session["id"]))
        try:
            _submit(base, tenant_id, session_id, _prompt("brief.md"))
            after_one = _await_terminal(base, tenant_id, session_id, already=0)
            first_said = _said_since(after_one, after_seq=0)
            boundary = _highest_seq(after_one)

            # Read before the attach and again at the end. A pod deleted and
            # re-placed between the two Turns would be built from the log, which folds
            # the attach -- so the file would arrive by a path this file is not testing
            # and every assertion below would still pass.
            pod_before = _pod_uid(session_id)

            # Uploaded only now, so the bytes did not exist when the pod was created
            # and cannot have arrived with it.
            second = _upload(base, tenant_id, "appendix.md", attached_nonce)
            with _client(base, tenant_id) as caller:
                attach = caller.post(
                    f"/v1/sessions/{session_id}/resources",
                    json={"file_id": second["id"], "type": "file"},
                )
                shown = caller.get(f"/v1/sessions/{session_id}/resources")
            assert shown.status_code == 200, shown.text

            _submit(base, tenant_id, session_id, _prompt("appendix.md"))
            after_two = _await_terminal(base, tenant_id, session_id, already=1)
            yield _Run(
                base=base,
                tenant_id=tenant_id,
                session_id=session_id,
                created_with=first,
                attached=second,
                created_nonce=created_nonce,
                attached_nonce=attached_nonce,
                attach_response=attach,
                first_turn_said=first_said,
                second_turn_said=_said_since(after_two, after_seq=boundary),
                pod_before_attach=pod_before,
                pod_after_second_turn=_pod_uid(session_id),
                definition_id=str(definition["id"]),
                environment_id=str(environment["id"]),
                listed=shown.json()["data"],
            )
        finally:
            _clean_up(session_id)


@requires_the_cluster
def test_the_attach_was_accepted_against_a_session_holding_a_pod(run: _Run) -> None:
    """201, and the resource it answers with is the file that was named.

    A 201 alone would be satisfied by a route that appended nothing and pushed nothing,
    which is why the two assertions below it exist -- but this one has to come first,
    because a non-201 here makes every later failure in this file a consequence rather
    than a finding."""
    assert run.attach_response.status_code == 201, run.attach_response.text
    assert run.attach_response.json()["id"] == run.attached["id"]
    assert run.attach_response.json()["filename"] == "appendix.md"


@requires_the_cluster
def test_the_first_turn_read_the_file_the_session_was_created_with(run: _Run) -> None:
    """The control: file delivery at creation works on this deployment.

    Without it, a missing token in the second Turn is ambiguous between "the attach
    route did not deliver" and "this pod cannot read ./files/ at all". This is the
    second reading ruled out, on the same pod, in the same run."""
    assert run.created_nonce in run.first_turn_said, run.first_turn_said


@requires_the_cluster
def test_the_agent_read_the_file_that_arrived_while_the_pod_was_running(
    run: _Run,
) -> None:
    """The finding: a file that arrived after the pod was running was read.

    This is the branch no offline test reaches. The token is in a file uploaded after
    the first Turn closed, so nothing about the pod's creation could have carried it,
    and the agent has no network path to the object store -- the pod holds no cloud
    identity (ADR-004). The only way this string reaches the model is the control plane
    pushing the bytes down the shim hop when the attach was accepted."""
    assert run.attached_nonce in run.second_turn_said, run.second_turn_said


@requires_the_cluster
def test_one_pod_served_both_turns_so_the_file_arrived_by_the_push(run: _Run) -> None:
    """The same pod object, before the attach and after the second Turn.

    This is what makes the test above a finding rather than a coincidence. A pod deleted
    and re-placed between the two Turns would be built from `_creation_facts`, which
    folds `session.file_attached` -- so the file would reach the workspace by the
    placement path, the agent would quote the nonce, and nothing would have exercised
    the push at all. The two readings being equal rules that out.

    A uid and not a name: the name is derived from the Session id, so it survives a
    delete and a re-place unchanged and would compare equal across exactly the event
    this exists to detect. Non-empty is asserted too, because two absent pods also
    compare equal.
    """
    assert run.pod_before_attach, "no pod existed after the first Turn closed"
    assert run.pod_after_second_turn == run.pod_before_attach


@requires_the_cluster
def test_the_resource_list_carries_both_in_the_order_they_arrived(run: _Run) -> None:
    """The list route folds creation and the attach, in log order.

    The order is the order the workspace is written in, and it is asserted rather than
    membership-checked for that reason. This also exercises the fold against a real log
    that has a Turn's worth of events between the two it cares about -- offline the two
    are adjacent."""
    assert [str(one["id"]) for one in run.listed] == [
        str(run.created_with["id"]),
        str(run.attached["id"]),
    ]


@requires_the_cluster
def test_the_deployed_route_refuses_a_mount_path_it_cannot_write(run: _Run) -> None:
    """The refusal and the acceptance, against the deployed route.

    Both halves in one case because the same file id is used for each: a route that
    refused every `mount_path` would satisfy the first assertion, and one that ignored
    the field would satisfy the second. Together they say the comparison happens.

    The refused value is upstream's own default, which is the one a client written
    against their surface actually sends."""
    third = _upload(run.base, run.tenant_id, "elsewhere.md", _nonce("NEVER"))
    with _client(run.base, run.tenant_id) as caller:
        refused = caller.post(
            f"/v1/sessions/{run.session_id}/resources",
            json={
                "file_id": third["id"],
                "type": "file",
                "mount_path": "/mnt/session/uploads/x",
            },
        )
        allowed = caller.post(
            f"/v1/sessions/{run.session_id}/resources",
            json={
                "file_id": third["id"],
                "type": "file",
                "mount_path": FILES_MOUNT_PATH,
            },
        )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID], refused.text
    assert refused.json()["error"]["detail"]["reason"] == REASON_MOUNT_PATH_FIXED
    assert allowed.status_code == 201, allowed.text


@requires_the_cluster
def test_the_deployed_route_refuses_a_name_the_workspace_already_holds(
    run: _Run,
) -> None:
    """A second `appendix.md` is refused, live, under its own code.

    The name is already in the workspace because this run put it there through the route
    under test, which is what makes this the real collision rather than a fixture of
    one. The receiver renames atomically into one flat directory, so accepting it would
    replace a file the agent has already read with different bytes and nothing would
    record the moment."""
    twin = _upload(run.base, run.tenant_id, "appendix.md", _nonce("TWIN"))
    with _client(run.base, run.tenant_id) as caller:
        refused = caller.post(
            f"/v1/sessions/{run.session_id}/resources",
            json={"file_id": twin["id"], "type": "file"},
        )

    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == ErrorCode.RESOURCE_FILENAME_ATTACHED.value


@requires_the_cluster
def test_the_deployed_listing_pages_backward(run: _Run) -> None:
    """`prev_page` on the deployed listing walks back to the page it came from.

    Here rather than in a file of its own because this run has a tenant with Sessions in
    it and a port-forward already open; the property is graded offline in
    `tests/control/test_sessions_page_backward.py`, and what this adds is that the
    deployed build emits the field at all -- a store without the capability omits it,
    and the whole three-state contract is invisible to a test that never asks a real
    deployment.

    Creates the second Session it needs rather than skipping without one. The version
    that skipped passed on every run and graded nothing, which is the shape of a case
    that looks like coverage: a walk backward needs two pages, and the other Session in
    this tenant is the one the rest of the file spends twenty seconds building.
    """
    with _client(run.base, run.tenant_id) as caller:
        # A second Session, created and never run. A collection of one has no page to
        # walk back to, and a case that skipped there would leave the field untested on
        # every deployment -- which is what happened until this create was added.
        # Placement is a first-Turn step, so a bare create costs no pod.
        _created(
            caller.post(
                "/v1/sessions",
                json={
                    "definition_id": run.definition_id,
                    "environment_id": run.environment_id,
                    "budget_minor_units": 1_000,
                    "budget_currency": "USD",
                    "retention_days": 1,
                },
            )
        )
        first = caller.get("/v1/sessions?limit=1")
        assert first.status_code == 200, first.text
        assert first.json()["prev_page"] is None, first.text
        token = first.json()["next_page"]
        assert token is not None, first.text
        second = caller.get(f"/v1/sessions?limit=1&page={token}")
        assert second.status_code == 200, second.text
        back = caller.get(f"/v1/sessions?limit=1&page={second.json()['prev_page']}")

    assert back.status_code == 200, back.text
    # Asserted non-empty because the comparison below is between two lists: a walk that
    # returned nothing at both ends would satisfy it while having paged over no rows.
    assert len(first.json()["sessions"]) == 1, first.text
    assert [one["id"] for one in back.json()["sessions"]] == [
        one["id"] for one in first.json()["sessions"]
    ]
    assert back.json()["prev_page"] is None
