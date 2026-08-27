"""A Session whose pod is gone takes another Turn, against the real cluster.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

**The claim no offline case can make.** Every unit test of the placement path fakes the
thing that actually has to work: that a pod which no longer exists is noticed as absent
and a new one is placed for a Session the platform has already seen. A fake placement
returns whatever it was told to return, so it can only prove the code branches the way
its author expected. This one takes two real Turns on one real Session with the pod
genuinely absent in between, which is the version of the claim that can still be
surprised -- by a scheduler that will not admit the second pod, by a shim token minted
for a pod that is gone, or by a `subPath` that mounts the second pod over the wrong
Session's workspace.

**Nothing here deletes the pod, and that is what changed.** An earlier draft did,
because a pod then outlived its Turn and the only other way to reach this state was to
wait out a fifteen-minute idle grace, which no case can do. A pod is now leased for
exactly one Turn and given back when that Turn ends (ADR-041), so "a Session whose pod
is gone" is simply what a Session is between Turns. The deletion was a stand-in for
production and it is gone; the state it used to manufacture is now the platform's own
behaviour, and `_the_pod_after_the_turn` measures it instead of causing it. That
measurement is asserted in a case of its own rather than buried in the fixture, because
a lease that stopped releasing would otherwise surface as a second Turn that ran warm.

**`session.placing`, twice, is the evidence.** It is appended by the dispatch that finds
this Session's pod absent, before it starts waiting for one, once per Turn that pays for
a placement. Seeing it TWICE on one Session says the platform observed this Session to
have no pod on each of two Turns and built one both times. This file used to count it
paired with `session.resumed`; nothing appends that any more, because under the lease
every Turn takes the branch it announced, so an event meaning "coming back from a
suspension" would fire on every Turn -- and it is webhook-eligible, which would make it
a callback per Turn to every endpoint a tenant registered.

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

requires_the_cluster = pytest.mark.skipif(
    __import__("os").environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places real pods and calls a real model",
)


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
    """One bare Session: no files, no tools, no skills."""
    with _client(base, tenant_id) as caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"resume-{run.lower()}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"resume-{run.lower()}",
                    "instructions": "You answer in one word and add no commentary.",
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
    """One Turn, asserted accepted at the door.

    The status is checked here rather than left to the poll below because the defect
    this file exists for used to surface exactly here: a Session the platform would not
    resume refused its second Turn with `turn.undeliverable`, and a poll that only
    waited for a terminal event would report that refusal as a six-hundred-second
    timeout, sending the reader after the pod instead of the state machine.
    """
    with _client(base, tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
        answered = caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": prompt},
            headers={"Idempotency-Key": uuid4().hex},
        )
    assert answered.status_code == 202, answered.text


def _await_terminal(
    base: str, tenant_id: str, session_id: SessionId, already: int
) -> list[dict[str, Any]]:
    """Poll until one MORE Turn has ended either way, and return the whole log.

    `already` is how many terminal events the log carried before this Turn was
    submitted, and counting past it is what makes this reusable for the second Turn: a
    poll that stopped at "a terminal event exists" would return the first Turn's
    completion instantly and report the second Turn as finished before it started.
    """
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


def _terminal(events: list[dict[str, Any]]) -> int:
    return sum(one["type"] in ("turn.completed", "turn.failed") for one in events)


def _the_pod_after_the_turn(session_id: SessionId) -> str:
    """What the cluster says about this Session's pod once its Turn has ended.

    Returns the last listing seen -- empty once the pod is gone -- rather than
    asserting, so the caller can grade it in a named case instead of a fixture dying
    inside a helper. A fixture that failed here would report "no pod" as a collection
    error over five cases at once, none of them naming the lease.

    Polled rather than read once, and the window it absorbs is real rather than a
    tolerance. Releasing a pod issues a delete; deletion is asynchronous, and
    `session-pod.yaml` gives the pod ten seconds of grace to stop. For that long the
    object is still listed, stamped for deletion -- which is the lease working, not
    failing, so a single read taken the instant a Turn completes would fail this file
    for the ordinary case.

    Waiting also decides WHICH placement path the second Turn takes, and that is worth
    naming because it is not the commoner one. A Turn arriving inside the grace window
    reads the stamped pod as GONE, and `_carry` treats GONE as a cue to place exactly as
    it treats ABSENT -- it waits the old pod out and places in its place. A Turn
    arriving after the window reads ABSENT. Both append `session.placing` and both end
    on a new pod, so every count below holds either way; this file waits so that it
    grades the ABSENT path deterministically rather than racing a ten-second window it
    could not land in reliably. The GONE path is the one an interactive tenant actually
    takes and it is not graded here.
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
class _Resumed:
    """One Session, two Turns, and what the cluster held between them."""

    session_id: SessionId
    tenant_id: str
    after_first: list[dict[str, Any]]
    pod_between_turns: str
    after_second: list[dict[str, Any]]


@pytest.fixture(scope="module")
def resumed() -> Iterator[_Resumed]:
    """Two Turns on one Session, with the lease taking the pod back in between.

    Module-scoped so two model calls and two cold starts happen once. The teardown
    deletes the pod and its Secrets whatever happened, for the reason its siblings give:
    aborted runs leave pods squatting the namespace, after which the next run's
    scheduling refusal reads as the cluster being out of capacity. It stays even though
    the lease should have left nothing to delete -- a teardown that trusts the thing the
    file is grading is a teardown that leaks on exactly the runs that found a defect.
    """
    tenant_id = str(uuid4())
    image = _session_image()
    run = uuid4().hex[:8]
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id = _a_session(base, tenant_id, image, run)
        try:
            _submit(base, tenant_id, session_id, "Reply with the single word FIRST.")
            after_first = _await_terminal(base, tenant_id, session_id, already=0)

            pod_between_turns = _the_pod_after_the_turn(session_id)

            _submit(base, tenant_id, session_id, "Reply with the single word SECOND.")
            after_second = _await_terminal(
                base, tenant_id, session_id, already=_terminal(after_first)
            )
            yield _Resumed(
                session_id=session_id,
                tenant_id=tenant_id,
                after_first=after_first,
                pod_between_turns=pod_between_turns,
                after_second=after_second,
            )
        finally:
            _clean_up(session_id)


@requires_the_cluster
def test_the_first_turn_completed_and_cold_started_its_pod(resumed: _Resumed) -> None:
    """The baseline, asserted before the resume so a failure is attributable.

    Both halves matter. If the first Turn failed, everything below is diagnosing a
    Session that never worked rather than one that would not take a second Turn. And if
    the first Turn did NOT announce a placement, this Session found a pod it should not
    have had, which would make the second placement below unattributable -- there would
    be no way to tell a Session that cold-started twice from one that cold-started once
    and was counted twice.
    """
    kinds = [one["type"] for one in resumed.after_first]
    assert "turn.completed" in kinds, kinds
    assert "turn.failed" not in kinds, kinds
    assert kinds.count("session.placing") == 1, kinds


@requires_the_cluster
def test_the_first_turn_gave_its_pod_back_when_it_ended(resumed: _Resumed) -> None:
    """The premise every case below rests on, and it is the platform's doing not this
    file's.

    A pod is leased for one Turn, so when the first Turn reached a terminal event its
    pod stopped existing. Asserted in its own case rather than inside the fixture
    because of how it fails: a lease that stopped releasing leaves a running pod, the
    second Turn finds it, and the placement counts below come up one short -- which
    reads as a Session that would not cold-start rather than as a pod nobody gave back.

    An empty listing and not a phase. `kubectl get` prints nothing at all for an object
    that is gone, and a pod still terminating is still listed, so the empty string is
    the only reading that excludes both.
    """
    assert resumed.pod_between_turns == "", (
        f"the first Turn ended and the cluster still listed {resumed.pod_between_turns}"
        f" {_POD_GONE_DEADLINE_S}s later, so the second Turn below ran warm and says "
        f"nothing about placement"
    )


@requires_the_cluster
def test_the_second_turn_was_accepted_and_completed(resumed: _Resumed) -> None:
    """The defect, stated as the thing that used to be impossible.

    `SUSPENDED` refused every Turn and nothing transitioned out of it, so a Session
    whose pod had been reclaimed was finished for good -- its next submission was
    refused at the door. Two completions on one Session, with the pod deleted between
    them, is that refusal being gone.

    Admission and completion are asserted separately because they fail differently and
    the difference is the diagnosis. A second `turn.submitted` says `accepts_a_turn()`
    let the Turn in, which is the half the state merge is responsible for; a second
    `turn.completed` says the pod that had to be built for it actually ran the work.
    A refusal at the door produces neither, and a placement that failed produces only
    the first -- so counting both tells the reader which layer broke without a rerun.
    """
    kinds = [one["type"] for one in resumed.after_second]
    assert kinds.count("turn.submitted") == 2, kinds
    assert kinds.count("turn.completed") == 2, kinds
    assert "turn.failed" not in kinds, kinds


@requires_the_cluster
def test_the_pod_was_cold_started_a_second_time(resumed: _Resumed) -> None:
    """Two placements on one Session, which is what says the second Turn built a pod
    rather than finding one.

    `session.placing` is appended only inside the branch a Turn reaches when the phase
    it read was `ABSENT`, and before it starts waiting -- so a second one cannot be
    produced by a Turn that found a pod running. Counting two is therefore a stronger
    statement than "the Turn worked": it says the platform observed this Session's pod
    to be missing and built another one for it.

    This is where the file used to also count `session.resumed`. Under the lease every
    Turn is the absent case, so that event would say nothing this one does not, and
    nothing appends it.
    """
    kinds = [one["type"] for one in resumed.after_second]
    assert kinds.count("session.placing") == 2, kinds


@requires_the_cluster
def test_each_placement_names_the_turn_that_paid_for_it(resumed: _Resumed) -> None:
    """The payload's only field, and the reason it carries one.

    A tenant watching the stream sees a Turn that is taking a long time and cannot
    otherwise tell waiting for a node from the model thinking. The Turn id is what
    attributes the wait to the Turn that is paying for it, and it is the only field
    `SessionPlacing` carries. Two placements naming two DIFFERENT Turns is what makes
    that field worth carrying; two naming the same Turn would mean the second event was
    a copy rather than an event -- which under a per-Turn lease is the shape a retry
    loop or a duplicated dispatch would take.
    """
    named = [
        str(one["payload"]["turn_id"])
        for one in resumed.after_second
        if one["type"] == "session.placing"
    ]
    assert len(named) == 2, named
    assert len(set(named)) == 2, named


@requires_the_cluster
def test_the_session_never_reported_a_state_that_refuses_a_turn(
    resumed: _Resumed,
) -> None:
    """The trap state, asserted absent from the tenant-visible stream.

    `session.suspended` is still a declared and published type -- the reaper appends it
    when it reclaims a pod, and that is correct. What must never appear is a Session
    reading as unable to take a Turn while a Turn is running on it, which is what the
    old fold produced. Neither Turn here was reaped, so neither event belongs in this
    log, and one appearing would mean something reclaimed a pod mid-Turn.
    """
    kinds = [one["type"] for one in resumed.after_second]
    assert "session.suspended" not in kinds, kinds
    assert "session.stopped" not in kinds, kinds
