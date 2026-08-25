"""The agent's working tree, kept in the object store after its pod is gone.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

The sibling of `test_a_session_ships_out_a_document_and_then_closes.py` on the other
lane. That file proves a *deliverable* survives its pod; this one proves the tree the
agent was working IN survives it, which is a different transfer over a different route
with a different write mode -- `artifacts` is sealed and written once, `working` is
mutable and rewritten every Turn.

**One Turn, and the reason is a platform limit rather than a choice.** The claim most
worth making live would be the second Turn's: that a sync re-reads only what changed.
It cannot be made here. `Placement.place_resuming` raises unconditionally, so a second
Turn submitted to a Session that has already completed one is refused at the door with
`turn.undeliverable` -- measured, not assumed, and recorded in `docs/progress.md`. The
diff is graded instead by `tests/control/test_workspace_sync.py`, which counts what the
pod was asked for; when resume is built, the second Turn belongs here.

**What this DOES settle that no fake can.** That the transfer runs against a real pod
over the real shim routes, that the write lands under this Session's `working` prefix in
the real bucket, that it is announced as a *replace* rather than a place -- the mutable
lane's write mode, not the sealed one's -- and that whatever a real Codex runtime leaves
lying in a workspace does not drag the reserved roots along with it.

**The nonce is generated here and exists nowhere else**, so bytes carrying it came back
from this agent's workspace and not from a previous run's leftovers in the same bucket.

NO KEY OR TOKEN VALUE IS PRINTED OR ASSERTED ON.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import forwarded

from managed_agent.core.ids import SessionId
from managed_agent.core.pod.workspace_contract import NOT_SYNCED
from managed_agent.core.vfs.session_vfs import WORKING

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_TURN_DEADLINE_S: Final = 600
_SUBMIT_TIMEOUT_S: Final = 900
_KEPT: Final = "analysis.py"
_SECOND: Final = "notes.txt"
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

requires_the_cluster = pytest.mark.skipif(
    __import__("os").environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places real pods and calls a real model",
)


def _bucket() -> str:
    """The object bucket the deployed control plane writes, read off its own manifest.

    Read from the running Deployment rather than named here, so this file cannot pass by
    inspecting a bucket the control plane is not actually using.

    A Deployment carrying no bucket FAILS here rather than skipping. A control plane
    with nowhere to put a lane is a broken deployment, not a reason this file has
    nothing to say -- and a skip would report that breakage as a green line with a
    number nobody reads. It is also not a precondition this file could build: the
    variable belongs to the deployment, and a test that set one would be grading a
    bucket the running control plane is not writing to.
    """
    from cluster_access import kubectl

    listed = kubectl(
        "get",
        "deploy",
        "control-plane",
        "-o",
        "jsonpath={.spec.template.spec.containers[0].env}",
    )
    for one in json.loads(listed):
        if one.get("name") == "MAP_OBJECT_BUCKET":
            found: str = one["value"]
            return found
    raise AssertionError(
        "the deployed control plane names no MAP_OBJECT_BUCKET, so it has nowhere to "
        "write a lane; fix the deployment rather than this file"
    )


def _aws(*argv: str) -> str:
    """One `aws` call, returning its stdout.

    Shelled out rather than run through a client library because the layer rules keep
    an object-store SDK inside `adapters/`, and a test that reached for one here would
    be the second place in the tree that knows how to talk to S3.
    """
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        pytest.fail(f"{' '.join(argv)} failed:\n{done.stderr}")
    return done.stdout


def _lane_keys(bucket: str, tenant_id: str, session_id: SessionId) -> dict[str, bytes]:
    """Every object in this Session's working lane, by path below the lane prefix.

    The bodies are read, not just the keys, because the claim this file makes is about
    bytes: a lane holding the right key with a previous Turn's contents is precisely
    what a broken diff produces, and a key listing cannot see it.
    """
    prefix = f"sessions/{tenant_id}/{session_id}/{WORKING.directory}/"
    listed = json.loads(
        _aws(
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--region",
            _REGION,
            "--output",
            "json",
        )
        or "{}"
    )
    found: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for one in listed.get("Contents", ()):
            key = str(one["Key"])
            into = Path(scratch) / "object"
            _aws(
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--region",
                _REGION,
                str(into),
            )
            found[key[len(prefix) :]] = into.read_bytes()
    return found


def _nonce() -> str:
    return f"WORK-{uuid4().hex[:12].upper()}"


def _prompt(nonce: str) -> str:
    """Two files at the workspace root, and the whole of what the agent is asked to do.

    At the ROOT and not under `out/`, because a file under `out/` is carried by the
    ship-out path this file is not about: a case that wrote there would pass whether or
    not the working lane existed. Two files rather than one so that a sync which
    transferred something -- but not everything -- is distinguishable from one that
    worked, which a single file cannot show.
    """
    return (
        f"In your current working directory -- NOT in any subdirectory -- create two "
        f"files. The first is named {_KEPT} and its entire contents must be this one "
        f"line followed by a newline:\n\nfirst {nonce}\n\nThe second is named "
        f"{_SECOND} and its entire contents must be this one line followed by a "
        f"newline:\n\nsecond {nonce}\n\nAdd no heading, no code fence and no other "
        f"text to either file. When both exist, reply with the single word DONE."
    )


def _session_image() -> str:
    """The newest digest in the Session repository, for the reason its sibling gives."""
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
    """One bare Session: no files, no tools, no skills."""
    with _client(base, tenant_id) as caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"working-{run.lower()}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"working-{run.lower()}",
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

    Both outcomes, for the reason the ship-out file gives: a poll waiting only for
    `turn.completed` sits out its whole deadline on a Turn that failed in the first
    second and then reports a timeout, which sends the reader after the wrong thing. A
    sync that raises is recorded as a FAILED Turn by design, so this module's own most
    likely failure arrives as `turn.failed`.
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
class _Kept:
    """One Turn of one Session, and what the working lane held afterwards."""

    session_id: SessionId
    tenant_id: str
    nonce: str
    events: list[dict[str, Any]]
    lane: dict[str, bytes]


@pytest.fixture(scope="module")
def kept() -> Iterator[_Kept]:
    """One Session, one Turn, and the lane read after it.

    Module-scoped so the Turn and its model call run once. The teardown deletes the pod
    and its Secrets whatever happened, for the reason its sibling gives: three aborted
    runs once left forty-two pods squatting the namespace, after which the next run's
    scheduling refusal read as the cluster being out of capacity.

    The lane is read out of the bucket rather than through the API on purpose. There is
    no route that lists a working lane -- it is the platform's own resume state and not
    a tenant-facing collection -- so the bucket is the only place to see it, and reading
    it there makes this a check on the STORED bytes rather than on a listing that could
    agree with a broken write.
    """
    nonce = _nonce()
    tenant_id = str(uuid4())
    image = _session_image()
    bucket = _bucket()
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
            events = _await_terminal(base, tenant_id, session_id)
            yield _Kept(
                session_id=session_id,
                tenant_id=tenant_id,
                nonce=nonce,
                events=events,
                lane=_lane_keys(bucket, tenant_id, session_id),
            )
        finally:
            _clean_up(session_id)


@requires_the_cluster
def test_the_turn_reached_a_terminal_event_at_all(kept: _Kept) -> None:
    """`turn.completed`, not `turn.failed`, and asserted before the legs below.

    A sync that raises is recorded as a FAILED Turn by design, the same as a ship-out
    that raises: the alternative is a Turn reading as complete while the tree it left
    behind is still only inside a pod about to die. So `turn.failed` here is this
    module's own loudest failure, and reading it first saves the reader from diagnosing
    an empty lane that is empty because the transfer never ran.
    """
    kinds = [one["type"] for one in kept.events]
    assert "turn.completed" in kinds, kinds
    assert "turn.failed" not in kinds, kinds


@requires_the_cluster
def test_the_files_the_agent_left_at_its_root_are_in_the_working_lane(
    kept: _Kept,
) -> None:
    """The bytes, compared exactly, carrying a nonce that exists nowhere else.

    The whole of the outward claim in one assertion: files the agent wrote OUTSIDE
    `out/` -- which is the half of the workspace the ship-out path was never for -- are
    in the object store under this Session's `working` prefix, with the bytes the agent
    wrote and not merely with the right names and lengths.
    """
    assert kept.lane.get(_KEPT) == f"first {kept.nonce}\n".encode()
    assert kept.lane.get(_SECOND) == f"second {kept.nonce}\n".encode()


@requires_the_cluster
def test_the_lane_was_written_as_a_replace_and_not_as_a_place(kept: _Kept) -> None:
    """`vfs.object_replaced` on `working`, and never `vfs.object_placed` on it.

    The two lanes differ in lifecycle and not only in name: `artifacts` is sealed and
    refuses a second write to a path, `working` is mutable and requires one. Which verb
    was used is the observable difference, and getting it wrong would pass every byte
    assertion in this file for exactly one Turn and then refuse the next one.
    """
    working = [
        one
        for one in kept.events
        if str(one.get("payload", {}).get("lane")) == WORKING.directory
    ]
    assert working, [one["type"] for one in kept.events]
    assert {one["type"] for one in working} == {"vfs.object_replaced"}, working
    named = {str(one["payload"]["relative"]) for one in working}
    assert {_KEPT, _SECOND} <= named, named


@requires_the_cluster
def test_the_reserved_roots_are_not_in_the_lane(kept: _Kept) -> None:
    """`files/`, `out/` and `.map/lib` stay out, on a real pod's real tree.

    The offline cases assert this against a tmp_path the test itself laid out, so they
    can only find a bug in the predicate. This one runs against whatever a real Codex
    runtime actually leaves in a workspace, which is the version of the claim that can
    still be surprised -- by a runtime that moved its state, or by a mount that arrived
    somewhere nobody expected.
    """
    for path in kept.lane:
        for reserved in NOT_SYNCED:
            assert not path.startswith(f"{reserved}/"), (path, reserved)
            assert path != reserved, path


@requires_the_cluster
def test_nothing_was_left_behind_at_the_ceiling(kept: _Kept) -> None:
    """No `vfs.working_lane_partial`, because this workspace is far under the bound.

    Asserted so that a run whose lane looked thin has one fewer explanation to chase: a
    partial sync SAYS it was partial, and the absence of that event means the lane holds
    everything the pod offered rather than an arbitrary prefix of it.
    """
    kinds = [one["type"] for one in kept.events]
    assert "vfs.working_lane_partial" not in kinds, kinds
