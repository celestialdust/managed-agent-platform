"""A Session's files survive the death of the pod that wrote them, with no restore.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

**The claim ADR-035 is, stated so it can fail.** Before the mount, a Session's working
files lived on the pod's own disk and the platform copied them to the bucket at Turn
boundaries; a pod that died between boundaries took whatever it had written with it, and
a resume restored the last copy rather than the work. The mount removes the copy: the
pod writes S3 Files directly, so the durable store and the working directory are the
same bytes and there is nothing to restore because nothing was ever a copy.

That claim is invisible to every offline test, because faking a filesystem proves only
that the code reads the path it was told to. It is also invisible to
`test_a_reaped_session_takes_another_turn.py`, which proves a Session with no pod gets a
new one and says nothing about what is on that pod's disk when it arrives.

**Why the browse route is read with the pod gone and no Turn in between.** That is the
step that makes this evidence rather than a story. Read while the pod runs, the file
could be coming off the pod's own disk. Read after a second Turn, a restore could have
put it back. Read in the gap -- pod confirmed gone, no Turn submitted -- the only thing
that can still be holding those bytes is the durable filesystem, because nothing else
exists at that moment. `GET /v1/sessions/{id}/workspace` answers from the control
plane's own mount of the same volume, so the read crosses a different process, a
different pod and a different mount than the write did.

**Nothing here deletes the pod any more, and the gap got wider rather than narrower.** A
pod is leased for exactly one Turn and given back when that Turn ends (ADR-041), so the
write Turn's pod is gone the moment that Turn is over -- this file used to manufacture
that state with `kubectl delete` and now only waits for it and grades that it happened.
The consequence for what can be observed is not symmetric, and it costs one claim: there
is no longer any moment at which the writing pod is alive and its Turn is over, so the
read this file used to take "while the pod is alive" cannot be taken at all without
racing a running Turn. That read is gone rather than weakened, and the pair it fed --
the same bytes either side of the pod's death -- is re-aimed at the pair that still
exists: the read taken in the gap, and the read taken after a second Turn has come and
gone. A restore step would have to show itself across a Turn boundary, and that is a
boundary this file still crosses.

**The nonce is generated per run and read back out of the file.** A fixed string would
pass against a workspace left by a previous run, which is the failure this whole file
would be least able to notice.

**Both prompts name a goal and never a command.** An earlier draft said "run this exact
shell command: cat durable.txt" and the Turn failed -- the runtime refused it, and the
model reported the approval policy as blocking it rather than reaching for a tool that
would have worked. That refusal is the sandbox behaving as designed and is graded in
`test_a_confined_command_runs_in_a_session_pod.py`; dictating a mechanism here only made
this file fail for a reason it is not about. What it has to observe is that the second
pod can see the first pod's bytes, and which tool the agent picks to see them with is
not part of that claim.

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

from managed_agent.core.ids import SessionId

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_TURN_DEADLINE_S: Final = 600
_SUBMIT_TIMEOUT_S: Final = 900
_POD_GONE_DEADLINE_S: Final = 180
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

_FILENAME: Final = "durable.txt"
"""The file the agent writes, at the top of its workspace.

Deliberately not under `files/` or `out/`: those two lanes carry contract meaning --
`files/` is what the platform put there and `out/` is what gets shipped out -- and a
file in either could plausibly be explained by the lane machinery rather than by the
mount. A bare file at the root of the workspace is explained by nothing but the volume.
"""

requires_the_cluster = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places real pods and calls a real model",
)

pytestmark = [pytest.mark.network, requires_the_cluster]


def _session_image() -> str:
    """The newest digest in the Session repository, for the reason its siblings give."""
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
    """One bare Session: no files, no tools, no skills.

    Bare on purpose. Attaching an input file would put bytes in the workspace that the
    platform delivered, and this case has to be able to say that the only thing which
    could have written `durable.txt` is the agent.
    """
    with _client(base, tenant_id) as caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"durable-{run.lower()}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"durable-{run.lower()}",
                    "instructions": (
                        "You use the shell to do exactly what you are asked, then "
                        "reply with the requested word and no commentary."
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


def _submit(base: str, tenant_id: str, session_id: SessionId, prompt: str) -> None:
    with _client(base, tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
        answered = caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": prompt},
            headers={"Idempotency-Key": uuid4().hex},
        )
    assert answered.status_code == 202, answered.text


def _terminal(events: list[dict[str, Any]]) -> int:
    return sum(one["type"] in ("turn.completed", "turn.failed") for one in events)


def _await_terminal(
    base: str, tenant_id: str, session_id: SessionId, already: int
) -> list[dict[str, Any]]:
    """Poll until one MORE Turn has ended either way, and return the whole log."""
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        if _terminal(events) > already:
            return events
        time.sleep(3)
    events = _events(base, tenant_id, session_id)
    pytest.fail(
        f"session {session_id} produced no terminal event beyond {already} in "
        f"{_TURN_DEADLINE_S}s; the log was {[one['type'] for one in events]}"
    )


def _browse(
    base: str, tenant_id: str, session_id: SessionId, path: str = ""
) -> httpx.Response:
    """One read of the browse route, returned unasserted so a case can grade the code.

    Unasserted because two of the cases below care about a status this helper would
    otherwise have decided was a failure.
    """
    suffix = f"/{path}" if path else ""
    with _client(base, tenant_id) as caller:
        return caller.get(f"/v1/sessions/{session_id}/workspace{suffix}")


def _the_pod_after_the_turn(session_id: SessionId) -> str:
    """What the cluster says about this Session's pod once its Turn has ended.

    Returns the last listing seen -- empty once the pod is gone -- rather than
    asserting, so a case can grade it by name. This file's whole argument is about the
    gap after the writer stops existing, and if the writer had not stopped existing the
    reads below would still pass while measuring a live pod's disk. That is the one
    failure worth stating in its own case rather than raising from a helper.

    Waited on rather than read once, and the window it absorbs is real rather than a
    tolerance. Releasing a pod issues a delete; deletion is asynchronous, and
    `session-pod.yaml` gives the pod ten seconds of grace to stop. For that long the
    object is still listed, stamped for deletion, and the container may still be
    running -- which is precisely the state the reads below must not be taken in, since
    a live writer is the explanation this file exists to exclude.
    """
    from managed_agent.control.session.placement import pod_name_for

    name = pod_name_for(session_id)
    listed = f"{name}: never read"
    deadline = time.monotonic() + _POD_GONE_DEADLINE_S
    while time.monotonic() < deadline:
        listed = kubectl(
            "get", "pod", name, "--ignore-not-found", "-o", "name", check=False
        ).strip()
        if not listed:
            break
        time.sleep(2)
    return listed


def _clean_up(session_id: SessionId) -> None:
    from managed_agent.control.session.placement import pod_name_for

    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


@dataclass(frozen=True, slots=True)
class _Outlived:
    """One Session, one nonce, and the reads taken after its pod stopped existing."""

    session_id: SessionId
    tenant_id: str
    nonce: str
    after_first: list[dict[str, Any]]
    pod_after_the_write: str
    listed_once_gone: httpx.Response
    read_once_gone: httpx.Response
    after_second: list[dict[str, Any]]
    read_after_second_turn: httpx.Response


@pytest.fixture(scope="module")
def outlived() -> Iterator[_Outlived]:
    """Write a nonce, wait for the lease to end, read with nothing running, then again.

    Module-scoped so two model calls and two cold starts happen once. The order is the
    argument and is not rearrangeable. The reads after `_the_pod_after_the_turn` are
    what separate "the file is on a disk" from "a file on a disk that outlived its pod";
    the second Turn is what shows the agent gets it back rather than merely that an
    operator can see it; and the read taken after that Turn is what a restore step would
    have to survive, since a Turn boundary is the only place one could run.
    """
    tenant_id = str(uuid4())
    nonce = uuid4().hex
    image = _session_image()
    run = uuid4().hex[:8]
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id = _a_session(base, tenant_id, image, run)
        try:
            _submit(
                base,
                tenant_id,
                session_id,
                f"Create a file called {_FILENAME} in your working directory whose "
                f"entire contents are the text {nonce} with no newline, no quotes and "
                f"nothing else. Then reply with the single word WROTE.",
            )
            after_first = _await_terminal(base, tenant_id, session_id, already=0)

            pod_after_the_write = _the_pod_after_the_turn(session_id)

            listed_once_gone = _browse(base, tenant_id, session_id)
            read_once_gone = _browse(base, tenant_id, session_id, _FILENAME)

            _submit(
                base,
                tenant_id,
                session_id,
                f"Read the file {_FILENAME} in your working directory and reply "
                f"with its exact contents and nothing else.",
            )
            after_second = _await_terminal(
                base, tenant_id, session_id, already=_terminal(after_first)
            )
            yield _Outlived(
                session_id=session_id,
                tenant_id=tenant_id,
                nonce=nonce,
                after_first=after_first,
                pod_after_the_write=pod_after_the_write,
                listed_once_gone=listed_once_gone,
                read_once_gone=read_once_gone,
                after_second=after_second,
                read_after_second_turn=_browse(base, tenant_id, session_id, _FILENAME),
            )
        finally:
            _clean_up(session_id)


def _answer(events: list[dict[str, Any]]) -> str:
    return "".join(
        str(one["payload"].get("text", ""))
        for one in events
        if one["type"] == "turn.message_delta"
    )


def test_the_write_turn_completed(outlived: _Outlived) -> None:
    """The baseline, asserted first so a failure below is attributable.

    If the Turn that was supposed to write the file never completed, every case below is
    diagnosing an empty workspace rather than a workspace that did not survive.
    """
    kinds = [one["type"] for one in outlived.after_first]
    assert "turn.completed" in kinds, kinds
    assert "turn.failed" not in kinds, kinds


def test_the_writing_pod_stopped_existing_when_its_turn_ended(
    outlived: _Outlived,
) -> None:
    """The premise the two reads below rest on, and under the lease it is measured
    rather than manufactured.

    A pod is leased for one Turn, so the pod that wrote the nonce is gone once the write
    Turn is over -- this file no longer deletes it. Asserted by name because of how it
    fails: a lease that stopped releasing leaves the writer running, and every read
    below would then pass while measuring a live pod's own disk, which is the one
    explanation this file exists to exclude.

    An empty listing and not a phase. `kubectl get` prints nothing at all for an object
    that is gone, while a pod still terminating is still listed, so the empty string is
    the only reading that excludes both.
    """
    assert outlived.pod_after_the_write == "", (
        f"the write Turn ended and the cluster still listed "
        f"{outlived.pod_after_the_write} {_POD_GONE_DEADLINE_S}s later, so the reads "
        f"below could be coming off the writing pod's own disk"
    )


def test_the_file_is_still_there_with_the_pod_gone_and_no_turn_in_between(
    outlived: _Outlived,
) -> None:
    """The case this file exists for.

    Nothing is running that could be serving these bytes from memory or from a
    container's own layer: the pod is confirmed gone, and no Turn has been submitted
    since, so no placement, no restore and no seed has had an opportunity to put
    anything back. What answers is the durable filesystem, because at this instant it is
    the only thing left holding the workspace.

    This is also the read that carries what a separate case used to: that the control
    plane reads the workspace through its own mount rather than out of the agent's
    container. Two processes, two pods and two mounts of one volume were the point of
    that case, and they are all still true here -- with the writer additionally gone,
    which is strictly the stronger observation.

    Before ADR-035 this read would have 404'd or served whatever the last Turn-boundary
    copy contained, which for a file written mid-Turn is nothing.
    """
    assert outlived.listed_once_gone.status_code == 200, outlived.listed_once_gone.text
    names = [entry["name"] for entry in outlived.listed_once_gone.json()["entries"]]
    assert _FILENAME in names, names


def test_the_bytes_survived_and_not_merely_the_name(outlived: _Outlived) -> None:
    """The nonce itself, read back with the pod gone.

    A directory listing proves an inode; this proves the contents. The nonce is minted
    per run, so it also excludes the one explanation a listing cannot: that the entry
    came from some earlier run's workspace under a reused Session id.
    """
    assert outlived.read_once_gone.status_code == 200, outlived.read_once_gone.text
    assert outlived.nonce in outlived.read_once_gone.text, (
        f"expected the run's nonce in the file served after the pod died; got "
        f"{outlived.read_once_gone.text[:200]!r}"
    )


def test_a_whole_turn_came_and_went_without_reconstructing_the_file(
    outlived: _Outlived,
) -> None:
    """Same bytes either side of a Turn, which is what "no restore step" means.

    A platform that restored from the bucket could also pass the case above while
    serving a slightly different file -- an earlier snapshot, or a copy whose trailing
    bytes were never flushed. This is where such a step would have to show itself: a
    whole Turn has run between these two reads, with a pod placed for it and released
    afterwards, and a Turn boundary is the only place a restore or a copy-back could
    plausibly run. Equality across it says nothing was reconstructed, because there was
    nothing to reconstruct from.

    The pair used to be the two sides of the pod's death. Under the lease there is no
    moment at which the writing pod is alive and its Turn is over, so that pair cannot
    be taken; this one crosses the same machinery -- `seed-runtime-home` and a fresh
    mount -- and can.
    """
    assert outlived.read_after_second_turn.status_code == 200, (
        outlived.read_after_second_turn.text
    )
    assert outlived.read_after_second_turn.text == outlived.read_once_gone.text


def test_the_agent_read_its_own_file_back_after_the_cold_start(
    outlived: _Outlived,
) -> None:
    """The tenant-visible form of the claim: the agent, not just an operator, gets it.

    The cases above prove the bytes are durable and readable by the control plane. This
    proves the second pod mounts the same subtree the first one wrote -- the `subPath`
    substitution -- which is the half a browse-route read cannot reach.
    """
    kinds = [one["type"] for one in outlived.after_second]
    assert "turn.failed" not in kinds, kinds
    assert kinds.count("turn.completed") == 2, kinds
    assert outlived.nonce in _answer(outlived.after_second), _answer(
        outlived.after_second
    )


def test_the_pod_really_was_cold_started_a_second_time(outlived: _Outlived) -> None:
    """Two placements, so the second Turn genuinely ran on a pod that did not exist.

    Without this the whole file degrades quietly: if the lease had not released the
    first pod, the second Turn would have run warm on it and every assertion above would
    still pass while testing nothing. The case above reads the cluster for that; this
    reads the platform's own account of it, and the two failing together is the
    difference between a pod that lingered and a placement the control plane never
    announced.

    `session.placing` is appended only where a Turn found this Session's pod absent, so
    counting two is a statement about what the platform observed and not merely that two
    Turns ran. It used to be counted alongside `session.resumed`; nothing appends that
    any more, because under a per-Turn lease every Turn is the absent case.
    """
    kinds = [one["type"] for one in outlived.after_second]
    assert kinds.count("session.placing") == 2, kinds
