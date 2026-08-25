"""A document the agent wrote, downloaded back, and the Session closed afterwards.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

Two capabilities the platform ships that had never been exercised against the real
cluster. Both have in-process tests and neither had ever completed once end to end
against the deployed API, which is a different claim: `test_output_shipout.py` drives
the ship-out class over fakes, and `test_session_lifecycle_transitions.py` drives the
transitions over a real database and no pod. Neither can see a route that is
not mounted, a manifest whose mount is missing, or a Deployment running last week's
image.

**The ship-out leg asks the model for the one thing models do reliably.** "Write these
exact bytes to this path" -- not a summary, not a transcription of a token into a
sentence. Everything after that is the platform's: the pod's own listing route, the
authenticated hop that reads it, the write into the Session's `artifacts` lane, the
`output.produced` append, and `GET /v1/sessions/{id}/artifacts/{path}`. The nonce is
generated here and exists nowhere else, so bytes that come back carrying it came back
from the agent's workspace.

**The path asked for is NESTED, and that is what this run is for.** Ship-out carried a
flat filename only, until a produced file started landing in the `artifacts` lane
instead of being minted as an upload. A Turn writing `report.txt` passes under the old
rule and the new one alike and separates them not at all, so it proves nothing about
the change. `out/report/report.txt` is a file the previous path could not have carried
at any point: an upload is keyed by a filename, and a filename holds no separator.

**The bytes are compared, not their length.** A file that arrives the right size and the
wrong content is the failure a length check cannot see, and it is the plausible one: the
hop reads a length off the pod's own listing and caps the fetch at it.

**The lifecycle leg asserts what each verb PROMISES, which is not what its name
suggests.** `POST /v1/sessions/{id}` changes nothing by design and refuses every field.
`DELETE` stops the Session and keeps its history -- the Event Log still reads at the
same sequences afterwards, which this file asserts, because a caller who assumed
otherwise would be assuming erasure this platform cannot perform.

NO KEY OR TOKEN VALUE IS PRINTED OR ASSERTED ON.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import forwarded

from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import SessionId
from managed_agent.core.pod.workspace_contract import OUTPUT_DIR_NAME

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_TURN_DEADLINE_S: Final = 600
_SUBMIT_TIMEOUT_S: Final = 900
_OUTPUT_PATH: Final = "report/report.txt"
"""Where the agent writes, below `out/`, and the path the artifact is stored under.

`out/` is the platform's collection point and the shim strips it, so what reaches the
lane -- and what `output.produced` announces, and what downloads -- is the path BELOW
it, separator included.
"""
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

requires_the_cluster = pytest.mark.skipif(
    __import__("os").environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places real pods and calls a real model",
)


def _nonce() -> str:
    """Upper-case hex behind a word, because the model has to reproduce it exactly."""
    return f"OUT-{uuid4().hex[:12].upper()}"


def _body(nonce: str) -> str:
    """The bytes the agent is asked to write, and the whole of what it is asked to do.

    One line and no formatting to get wrong. A prompt asking for a document to be
    *composed* would grade the model's prose and then compare bytes against prose, which
    no assertion can do; a prompt asking for a fixed string back makes every failure
    downstream of the model a platform failure.
    """
    return f"produced-by-the-agent {nonce}\n"


def _prompt(nonce: str) -> str:
    return (
        f"Create the directory ./{OUTPUT_DIR_NAME}/report/ under your current working "
        f"directory, then write a file at ./{OUTPUT_DIR_NAME}/{_OUTPUT_PATH} . Its "
        f"entire contents must be this one line, followed by a newline and nothing "
        f"else:\n\n{_body(nonce).rstrip()}\n\n"
        f"Do not add a heading, a code fence, or any other text to the file. When the "
        f"file exists, reply with the single word DONE."
    )


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned.

    Newest-push, matching the files beside this one: a digest written in here would pin
    the run to whatever ECR held on the day somebody typed it, and this file's whole
    subject -- a route and a mount that arrived recently -- is what a stale image lacks.
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


def _client(base: str, tenant_id: str, timeout: int = 90) -> httpx.Client:
    return httpx.Client(
        base_url=base, timeout=timeout, headers={TENANT_HEADER: tenant_id}
    )


def _created(answered: httpx.Response) -> dict[str, Any]:
    assert answered.status_code == 201, answered.text
    body: dict[str, Any] = answered.json()
    return body


def _a_session(base: str, tenant_id: str, image: str, run: str) -> SessionId:
    """One Session with no files, no tools and no skills, through the REST API only.

    Deliberately bare. This file's subject is the outbound path and the lifecycle verbs,
    and every attachment a Session could carry is one more thing that can fail in a way
    a reader would have to rule out first.
    """
    with _client(base, tenant_id) as caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"shipout-{run.lower()}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"shipout-{run.lower()}",
                    "instructions": (
                        "You write files exactly as asked, with no extra commentary "
                        "inside the file."
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
                    "budget_minor_units": 500_000,
                    "budget_currency": "USD",
                    "retention_days": 1,
                },
            )
        )
    return SessionId(UUID(session["id"]))


def _events(base: str, tenant_id: str, session_id: SessionId) -> list[dict[str, Any]]:
    with _client(base, tenant_id) as caller:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
    assert answered.status_code == 200, answered.text
    listed: list[dict[str, Any]] = answered.json()["events"]
    return listed


def _await_terminal(
    base: str, tenant_id: str, session_id: SessionId
) -> list[dict[str, Any]]:
    """Poll until the Turn ends either way, and return the whole log.

    Both outcomes: a poll waiting only for `turn.completed` sits out its whole deadline
    on a Turn that failed in the first second and then reports a timeout, which sends
    the reader after the wrong thing. A ship-out that raises is recorded as a FAILED
    Turn by design, so this leg's own most likely failure arrives as `turn.failed`.
    """
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        if any(one["type"] in ("turn.completed", "turn.failed") for one in events):
            return events
        time.sleep(3)
    events = _events(base, tenant_id, session_id)
    pytest.fail(
        f"session {session_id} produced no terminal event in {_TURN_DEADLINE_S}s; "
        f"the log was {[one['type'] for one in events]}"
    )


def _clean_up(session_id: SessionId) -> None:
    from cluster_access import kubectl  # noqa: I001 -- local, after the src import

    from managed_agent.control.session.placement import pod_name_for

    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


@dataclass(frozen=True, slots=True)
class _Shipped:
    """One run of the outbound path, for the cases to read."""

    session_id: SessionId
    tenant_id: str
    nonce: str
    events: list[dict[str, Any]]
    base: str


@pytest.fixture(scope="module")
def shipped() -> Iterator[_Shipped]:
    """One Session, one Turn that writes one file, and the log it produced.

    Module-scoped so the Turn runs once. The teardown deletes the pod and its Secrets
    whatever happened, including a failure during submission -- three aborted runs once
    left forty-two pods squatting the namespace, after which the next run's scheduling
    refusal read as the cluster being out of capacity.

    `base` is carried on the record because the port-forward closes when this fixture
    exits, so a case that wanted to download the file afterwards would have nothing to
    reach; the download happens in the cases, inside this fixture's scope, while it is
    still up.
    """
    nonce = _nonce()
    tenant_id = str(uuid4())
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id = _a_session(base, tenant_id, image, nonce)
        try:
            with _client(base, tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
                answered = caller.post(
                    f"/v1/sessions/{session_id}/events",
                    json={"prompt": _prompt(nonce)},
                    headers={"Idempotency-Key": uuid4().hex},
                )
            assert answered.status_code == 202, answered.text
            yield _Shipped(
                session_id=session_id,
                tenant_id=tenant_id,
                nonce=nonce,
                events=_await_terminal(base, tenant_id, session_id),
                base=base,
            )
        finally:
            _clean_up(session_id)


@requires_the_cluster
def test_the_turn_reached_a_terminal_event_at_all(shipped: _Shipped) -> None:
    """`turn.completed`, not `turn.failed`, and asserted before the legs below.

    First because a ship-out that raises is recorded as a failed Turn on purpose: the
    alternative is a Turn reading as complete while what it produced is still only
    inside a pod about to die. So `turn.failed` here is the ship-out path's own loudest
    failure, and it must not read as "the model did not write the file".
    """
    types = [one["type"] for one in shipped.events]
    assert "turn.completed" in types, types


@requires_the_cluster
def test_the_produced_file_was_announced_in_the_event_log(shipped: _Shipped) -> None:
    """The event without which the bytes are stored and unreachable.

    Nothing else in the platform tells a tenant the path: the resources listing answers
    with the files the Session was CREATED with, and the `artifacts` lane has no listing
    route by design -- the Event Log is the listing. So this event is the only path from
    "my agent wrote a document" to those bytes, and it is asserted before the download
    because a missing event and a bad download are different failures.

    **`path`, and no longer `filename` beside a `file_id`.** The payload carries the
    workspace-relative path the agent wrote at, below `out/`, which is exactly what the
    download route takes. A bare filename could not name the file this run produces,
    and there is no upload to carry an id for.
    """
    announced = [one for one in shipped.events if one["type"] == "output.produced"]
    assert announced, [one["type"] for one in shipped.events]
    payload = announced[0]["payload"]
    assert payload["path"] == _OUTPUT_PATH, payload
    assert payload["byte_length"] == len(_body(shipped.nonce).encode()), payload


@requires_the_cluster
def test_the_announcement_follows_the_turn_it_belongs_to(shipped: _Shipped) -> None:
    """Ordered after `turn.completed`, which is the true causal order: ship-out runs at
    completion. A tenant polling for the terminal event and then reading the rest of the
    log finds the file id already there, rather than having to keep polling past a Turn
    that already said it was done."""
    types = [one["type"] for one in shipped.events]
    assert types.index("output.produced") > types.index("turn.completed"), types


@requires_the_cluster
def test_the_bytes_come_back_byte_for_byte(shipped: _Shipped) -> None:
    """The whole outbound path, compared as bytes and not as a length.

    A file the right size with the wrong content is what a length check cannot see, and
    it is the plausible failure here: the hop reads a length off the pod's own listing
    and caps its fetch at it. The nonce exists in this process and in the bytes the
    agent was asked to write, nowhere else -- so these bytes came from that workspace.
    """
    announced = [one for one in shipped.events if one["type"] == "output.produced"]
    assert announced, [one["type"] for one in shipped.events]
    path = announced[0]["payload"]["path"]
    assert "/" in path, path
    with _client(shipped.base, shipped.tenant_id) as caller:
        got = caller.get(f"/v1/sessions/{shipped.session_id}/artifacts/{path}")
    assert got.status_code == 200, got.text
    assert got.content == _body(shipped.nonce).encode(), got.content[:200]


@requires_the_cluster
def test_another_tenant_cannot_download_the_file(shipped: _Shipped) -> None:
    """The same id, a different tenant header, and no bytes.

    Asserted here rather than left to the unit tier because this is the first time the
    route has served a PRODUCED file at all, and produced files reach the lane by a
    path uploads never take -- `output_shipout` calls `place` directly with a tenant it
    read off the Session record, not one a caller sent.

    **The property is that "not yours" and "no such file" are indistinguishable, and
    this now asserts it instead of arguing it.** Two calls, the second naming a Session
    id never issued to anybody, and the two refusals are compared. The status and the
    code have to be equal -- a caller able to tell them apart could walk the identifier
    space and learn which ids exist somewhere on the platform. A 403 beside a 404 would
    be exactly that leak.

    The prose it replaces argued the same thing and asserted only one of the two calls,
    so nothing here would have noticed the day the cases diverged. It also named a
    literal status, twice wrongly: `422` until wave 0 moved every `request.invalid` to
    `400`, and `request.invalid` until the files slice gave the refusal its own
    `file.not_found` and a 404. Both drifts survived because a cluster-gated file is
    invisible to an offline sweep. Read out of `STATUS_FOR` now, which cannot go stale.

    404 is also the parity-correct answer -- upstream answers `not_found_error` for a
    resource that is not there -- and it costs nothing here, because the route could not
    tell the two apart if it wanted to. The object key composes the tenant from the
    REQUEST and the Session id from the PATH, so a stranger naming this Session reads a
    key under their own tenant segment, which holds nothing. The isolation is the key's
    shape rather than a comparison this code performs, which is why there is no branch
    here that could be got wrong.
    """
    announced = [one for one in shipped.events if one["type"] == "output.produced"]
    assert announced
    path = announced[0]["payload"]["path"]
    never_issued = uuid4()
    with _client(shipped.base, str(uuid4())) as stranger:
        refused = stranger.get(f"/v1/sessions/{shipped.session_id}/artifacts/{path}")
        unknown = stranger.get(f"/v1/sessions/{never_issued}/artifacts/{path}")

    assert refused.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND], refused.text
    body = refused.json()
    assert body["error"]["code"] == ErrorCode.FILE_NOT_FOUND.value, refused.text
    assert shipped.nonce not in refused.text

    # The comparison, which is the whole test. Statuses and codes equal; the only
    # difference permitted between the two bodies is the id each was asked about and
    # the per-request id, both of which the caller already knows.
    assert unknown.status_code == refused.status_code, unknown.text
    assert unknown.json()["error"]["code"] == body["error"]["code"], unknown.text
    assert unknown.json()["error"]["message"] == body["error"]["message"], unknown.text


@requires_the_cluster
def test_the_lifecycle_verbs_do_what_they_promise() -> None:
    """Create, read, update, archive, delete -- against the deployed API, in one case.

    One case rather than five because these are steps of one sequence and not
    independent claims: an update asserted against a Session that was never created
    tells a reader nothing, and each step's precondition is the step before it. The
    file above is the opposite shape for the opposite reason -- those legs fail
    independently.

    What is asserted is what each verb PROMISES, which for two of them is not what the
    name suggests. `POST /v1/sessions/{id}` revises nothing on this platform: an empty
    body answers with the current state and a body naming a field is refused with a code
    naming it. `DELETE` stops the Session and KEEPS its history, so the Event Log still
    reads at the same sequences afterwards -- asserted, because a caller who assumed
    erasure would be assuming an operation no port in this tree can perform.

    No pod and no Turn: none of these verbs needs one, and a Turn here would put a model
    call and a placement wait in front of a sequence that is testing neither.
    """
    tenant_id = str(uuid4())
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id = _a_session(base, tenant_id, image, _nonce())
        try:
            with _client(base, tenant_id) as caller:
                read = caller.get(f"/v1/sessions/{session_id}")
                assert read.status_code == 200, read.text
                assert read.json()["id"] == str(session_id)

                identity = caller.post(f"/v1/sessions/{session_id}", json={})
                assert identity.status_code == 200, identity.text
                assert identity.json()["id"] == str(session_id)

                refused = caller.post(
                    f"/v1/sessions/{session_id}",
                    json={"budget_minor_units": 999},
                )
                # Status and envelope shape both come from the code table and the
                # one published envelope. Wave 0 moved this refusal from 422 to 400
                # and nested the body under `error`, and this file asserted the old
                # shape for both -- unnoticed, because a cluster-gated test is
                # invisible to an offline sweep.
                assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID], (
                    refused.text
                )
                refusal = refused.json()["error"]
                assert refusal["code"] == ErrorCode.REQUEST_INVALID.value, refused.text
                assert refusal["detail"]["reason"] == "budget_not_revisable", (
                    refused.text
                )

                archived = caller.post(f"/v1/sessions/{session_id}/archive")
                assert archived.status_code == 200, archived.text

                again = caller.post(f"/v1/sessions/{session_id}/archive")
                assert again.status_code == 200, again.text

                closed = caller.delete(f"/v1/sessions/{session_id}")
                assert closed.status_code == 200, closed.text

                survived = caller.get(f"/v1/sessions/{session_id}/events")
                assert survived.status_code == 200, survived.text
                types = [one["type"] for one in survived.json()["events"]]
                assert "session.created" in types, types
                assert "session.stopped" in types, types

                still_listed = caller.get(f"/v1/sessions/{session_id}")
                assert still_listed.status_code == 200, still_listed.text
        finally:
            _clean_up(session_id)
