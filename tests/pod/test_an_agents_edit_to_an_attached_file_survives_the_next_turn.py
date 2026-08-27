"""An agent edits a file attached to its Session, and the next Turn leaves it alone.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

**The defect this exists for was silent data loss, and it is assembled out of three
facts that are each correct on their own.** A pod is leased for exactly one Turn
(ADR-041), so every Turn is a placement; `control/session/pods.py` calls
`AttachedFiles.place_for` after every `place`, so every placement re-pushes the
Session's WHOLE attachment set before the agent is given an instruction; and the
workspace lives on a mounted volume that outlives the pod (ADR-035), so those bytes land
on a file the agent may have spent the previous Turn editing. Until commit 9ae122b the
shim's file route ended in an unconditional `partial.replace(final)`, and the three
facts together meant the start of every Turn overwrote the previous Turn's work -- with
the Turn reporting success, the tenant told nothing, and no event recording that
anything had been taken away. `session_shim/serve.py::place_a_file` now returns 204 for
a name the workspace already holds, which is what makes re-delivery idempotent instead
of destructive.

**Why the offline test of that route cannot reach this.**
`tests/session_shim/test_shim_places_a_file.py` drives the route in process and proves
the skip, which is the unit of the fix and not the property. The three facts that turn a
re-push into data loss are a lease that ends at a Turn boundary, a volume that outlives
the pod, and a placement path that fires again on the next Turn; none of the three is in
that process, and the route there is called by the test rather than by a placement. A
Session running a stale shim image would pass every offline gate in this repo and still
overwrite, because the code that decides is inside the image.

**The witness is a nonce the model wrote, and it is read back through a different
process.** The nonce is minted per run and the agent is asked to append it to the
attached file, so nothing in this file ever writes those bytes -- a fixture that planted
them through some other door would be grading that door. It is then read through the
control plane's own read-only mount of the same volume
(`GET /v1/sessions/{id}/workspace/...`), the read
`test_a_workspace_outlives_the_pod_that_wrote_it.py` establishes. It has to be that
read rather than an exec into the pod: between Turns this Session owns no pod to exec
into, and the whole question is what is on the volume in that gap and after the next
placement crosses it.

**One Session, two Turns, module-scoped, and the order is the argument.** The first Turn
makes the edit. The read taken after it is what says there was an edit to lose. The
second Turn is a fresh pod and therefore a fresh placement -- which is graded rather
than assumed, because a Turn that somehow reused a pod would place nothing, re-push
nothing, and pass every case below while proving none of them. The read taken after that
Turn is the finding.

**What this file can observe and what it cannot, stated rather than implied.** The PUT
the placement makes is not visible from outside the pod: the fixed route answers 204 for
a skip and 204 for a write, the pod is gone before anything here could read its log, and
no event records a file push. So the chain graded here is that the placement path ran a
second time (`session.placing` twice) and did not refuse (the Turn completed, and
`place_for` raising fails the Turn before it starts), which together mean every attached
file was pushed again. That is an inference about one line of control-plane code and is
the one link below that is reasoned rather than measured.

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

from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import SessionId
from managed_agent.core.pod.workspace_contract import INPUT_DIR_NAME

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

_FILENAME: Final = "notebook.md"
"""The one file this Session attaches, and the one the agent is asked to edit.

Attached rather than created by the agent, which is the whole point: a file the agent
made itself is never re-pushed, so it is not the file the defect could reach. This name
is what the placement writes to on every Turn.
"""

requires_the_cluster = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places real pods and calls a real model",
)


def _nonce(prefix: str) -> str:
    """Upper-case hex behind a word, because the model has to reproduce it exactly."""
    return f"{prefix}-{uuid4().hex[:12].upper()}"


def _doc(nonce: str) -> bytes:
    """The document the tenant attaches, carrying a code of its own.

    The attached document's code matters as much as the appended one. It is what the
    re-push would restore, so a file still holding it *and* the agent's line is a file
    that was added to; a file holding only it is a file that was replaced -- which is
    precisely the failure, and the two are told apart by looking for both.
    """
    return (
        "# Working notebook\n\n"
        "This document is attached to the session and the agent is asked to add to "
        "it.\n\n"
        f"    reference code: {nonce}\n"
    ).encode()


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned.

    Newest-push, matching the files beside this one: a digest written in here would pin
    the run to whatever ECR held on the day somebody typed it. It matters more here than
    in most of these files -- the line under test is inside this image, so a run against
    a stale digest grades the defect rather than the fix and says nothing either way.
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


def _upload(base: str, tenant_id: str, name: str, nonce: str) -> dict[str, Any]:
    """Upload the document. 200 and 201 both accepted, matching the files beside this.

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


def _edit_prompt(line: str) -> str:
    """One demand, a goal rather than a command, and nothing else asked for.

    A goal and not a shell command for the reason
    `test_a_workspace_outlives_the_pod_that_wrote_it.py` records: an earlier draft
    elsewhere dictated the exact command and the runtime refused it, which failed the
    Turn for a reason that file was not about. Which tool the agent reaches for to
    append a line is not part of this claim.

    "Append" and "create no other file" are both spelled out because the cheap way to
    satisfy a vague version of this is to write a new file, and a new file is never
    re-pushed -- so an agent that did that would leave this run grading nothing.
    """
    return (
        f"The file ./{INPUT_DIR_NAME}/{_FILENAME} already exists. Append one new line "
        f"to the end of it, reading exactly: {line}. Leave everything already in that "
        "file exactly as it is, and create no other file. Then reply with the single "
        "word APPENDED and nothing else."
    )


def _read_back_prompt() -> str:
    """The second Turn's demand: read, quote, and change nothing.

    Read-only in words as well as in intent, because two cases below compare the file
    either side of this Turn and an agent that tidied up would fail them for something
    that is not the property. The revision code is asked for by that name: the attached
    document does not contain one, so an agent handed a file the placement had
    overwritten has nothing to quote and says so.
    """
    return (
        f"Read the file ./{INPUT_DIR_NAME}/{_FILENAME} and reply with the revision "
        "code it now contains, exactly as written. Do not create, modify or delete "
        "any file. Reply with nothing else."
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


def _terminal(events: list[dict[str, Any]]) -> int:
    return sum(one["type"] in ("turn.completed", "turn.failed") for one in events)


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
    reports a timeout, which sends the reader after the wrong thing.
    """
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        if _terminal(events) > already:
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
    answers apart. Without it the second Turn's assertion passes on a Session whose
    second Turn said nothing at all, because the first Turn's deltas are still in the
    log -- and the first Turn was handed the nonce in its own prompt, so it is exactly
    the place a stray copy of the token would be found.
    """
    return "".join(
        str(one["payload"].get("text", ""))
        for one in events
        if one["type"] == "turn.message_delta" and int(one["seq"]) > after_seq
    )


def _highest_seq(events: list[dict[str, Any]]) -> int:
    return max(int(one["seq"]) for one in events)


def _file(
    base: str, tenant_id: str, session_id: SessionId, relative: str
) -> httpx.Response:
    """One read of the workspace file route, returned unasserted.

    Unasserted so the case that grades it says what the status was worth; a helper that
    raised would turn a 404 into a fixture error naming this line rather than a failure
    naming the file that went missing.
    """
    with _client(base, tenant_id) as caller:
        return caller.get(f"/v1/sessions/{session_id}/workspace/{relative}")


def _listing(
    base: str, tenant_id: str, session_id: SessionId, path: str
) -> httpx.Response:
    """One read of the workspace listing route, returned unasserted, same reason."""
    with _client(base, tenant_id) as caller:
        return caller.get(f"/v1/sessions/{session_id}/workspace", params={"path": path})


def _shape(listed: httpx.Response) -> list[tuple[str, str, int | None]]:
    """A directory reduced to what a re-push would change: name, kind and length.

    `modified_at` is deliberately left out. It would be the sharpest witness of a
    rewrite, and it is also the one field whose stability across two reads is a property
    of an S3 Files mount's attribute handling rather than of this platform -- so a case
    resting on it would fail for a reason this file cannot diagnose. The length carries
    the same claim here without that risk: the agent's line makes the file longer than
    the document that would be restored over it.
    """
    return [
        (str(entry["name"]), str(entry["kind"]), entry["byte_length"])
        for entry in listed.json()["entries"]
    ]


def _the_pod_after_the_first_turn(session_id: SessionId) -> str:
    """What the cluster says about this Session's pod once its Turn has ended.

    Returns the last listing seen -- empty once the pod is gone -- rather than
    asserting, so the case that grades it can say in its own name what an empty string
    is worth: the second Turn is a second placement only if the first pod was given
    back, and if it was not then nothing below was re-pushed at all.

    Waited on rather than read once, and the window it absorbs is real rather than a
    tolerance. The lease is released after the Turn's completion has been appended, and
    deletion is asynchronous on top of that, so for the pod's grace period the object is
    still listed, stamped for deletion, and its container may still be running.

    An empty listing rather than a phase. `kubectl get` prints nothing at all for an
    object that is gone, while a pod still terminating is still listed under its own
    name, so the empty string is the only reading that excludes both.
    """
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
    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


@dataclass(frozen=True, slots=True)
class _Run:
    """One Session, one edit, and the workspace read either side of the next Turn."""

    session_id: SessionId
    attached_nonce: str
    edited_nonce: str
    after_first: list[dict[str, Any]]
    read_after_the_edit: httpx.Response
    listed_after_the_edit: httpx.Response
    pod_between_the_turns: str
    after_second: list[dict[str, Any]]
    second_turn_said: str
    read_after_the_placement: httpx.Response
    listed_after_the_placement: httpx.Response


@pytest.fixture(scope="module")
def run() -> Iterator[_Run]:
    """Attach a file, have the agent edit it, then run a second Turn over the top.

    Module-scoped so two model calls and two cold starts happen once. The reads are
    taken at the two moments that matter and nowhere else: after the editing Turn's pod
    is confirmed gone, which is what says an edit existed on the volume with nothing
    running to hold it there; and after the next Turn, which is the placement that used
    to overwrite it.

    The teardown deletes the pod and its Secrets whatever happened, including a failure
    during submission. A run that died there still created a Session the control plane
    places a pod for, and three aborted runs once left forty-two pods squatting the
    namespace, after which the next run's scheduling refusal read as the cluster being
    out of capacity.
    """
    tenant_id = str(uuid4())
    stamp = uuid4().hex[:8]
    attached_nonce = _nonce("ATTACHED")
    edited_nonce = _nonce("EDIT")
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        uploaded = _upload(base, tenant_id, _FILENAME, attached_nonce)
        with _client(base, tenant_id) as caller:
            environment = _created(
                caller.post(
                    "/v1/environments",
                    json={"name": f"edit-{stamp}", "runtime_image": image},
                )
            )
            definition = _created(
                caller.post(
                    "/v1/agents",
                    json={
                        "name": f"edit-{stamp}",
                        "instructions": (
                            f"Files attached to your Session are in "
                            f"./{INPUT_DIR_NAME}/ relative to your working directory. "
                            "You may edit them. Do exactly what you are asked and "
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
                        "file_ids": [uploaded["id"]],
                        "budget_minor_units": 500_000,
                        "budget_currency": "USD",
                        "retention_days": 1,
                    },
                )
            )
        session_id = SessionId(UUID(session["id"]))
        relative = f"{INPUT_DIR_NAME}/{_FILENAME}"
        try:
            _submit(
                base,
                tenant_id,
                session_id,
                _edit_prompt(f"revision code: {edited_nonce}"),
            )
            after_first = _await_terminal(base, tenant_id, session_id, already=0)
            boundary = _highest_seq(after_first)

            # Waited for rather than read, and both reads below are taken on the far
            # side of it. A pod still standing here is one whose container could be
            # serving these bytes and one the next Turn might reuse, and either reading
            # empties the case this file exists for.
            pod_between_the_turns = _the_pod_after_the_first_turn(session_id)

            read_after_the_edit = _file(base, tenant_id, session_id, relative)
            listed_after_the_edit = _listing(
                base, tenant_id, session_id, INPUT_DIR_NAME
            )

            _submit(base, tenant_id, session_id, _read_back_prompt())
            after_second = _await_terminal(
                base, tenant_id, session_id, already=_terminal(after_first)
            )
            yield _Run(
                session_id=session_id,
                attached_nonce=attached_nonce,
                edited_nonce=edited_nonce,
                after_first=after_first,
                read_after_the_edit=read_after_the_edit,
                listed_after_the_edit=listed_after_the_edit,
                pod_between_the_turns=pod_between_the_turns,
                after_second=after_second,
                second_turn_said=_said_since(after_second, after_seq=boundary),
                read_after_the_placement=_file(base, tenant_id, session_id, relative),
                listed_after_the_placement=_listing(
                    base, tenant_id, session_id, INPUT_DIR_NAME
                ),
            )
        finally:
            _clean_up(session_id)


@requires_the_cluster
def test_the_turn_that_was_asked_to_edit_the_file_completed(run: _Run) -> None:
    """The baseline, asserted first so a failure below is attributable.

    If the Turn that was supposed to append the line never completed, every case below
    is diagnosing a file nothing edited rather than an edit that did not survive.
    """
    kinds = [one["type"] for one in run.after_first]
    assert "turn.completed" in kinds, kinds
    assert "turn.failed" not in kinds, kinds


@requires_the_cluster
def test_the_agent_added_its_line_to_the_attached_file_itself(run: _Run) -> None:
    """There was an edit to lose, and it was made to the file the placement writes.

    Both codes, and the pair is the assertion rather than either half. The agent's code
    says something was appended; the attached document's own code, still present beside
    it, says the appending happened *in* the attached file rather than in a new file the
    agent created at some other name -- and a new file is never re-pushed, so an agent
    that took that route would leave every case below passing with nothing at stake.

    Read through the control plane's mount with the editing pod already gone, so what
    answers is the volume rather than a container that could still be holding the bytes.
    """
    assert run.read_after_the_edit.status_code == 200, run.read_after_the_edit.text
    body = run.read_after_the_edit.text
    assert run.edited_nonce in body, body[:400]
    assert run.attached_nonce in body, body[:400]


@requires_the_cluster
def test_the_pod_that_made_the_edit_was_given_back_before_the_next_turn(
    run: _Run,
) -> None:
    """The premise the finding rests on: the next Turn had no pod to inherit.

    A pod is leased for one Turn and given back when that Turn ends, which is what makes
    the next Turn a placement at all. Asserted by name because of how it fails: a lease
    that stopped releasing would leave the second Turn running warm on the first pod,
    pushing nothing and re-writing nothing, and every case below would pass while
    grading a code path that never ran.

    An empty listing and not a phase, for the reason the helper gives.
    """
    assert run.pod_between_the_turns == "", (
        f"the editing Turn ended and the cluster still listed "
        f"{run.pod_between_the_turns} {_POD_GONE_DEADLINE_S}s later, so the next Turn "
        "may have reused that pod and re-pushed nothing"
    )


@requires_the_cluster
def test_the_next_turn_really_was_a_second_placement(run: _Run) -> None:
    """The platform's own account of what the case above read off the cluster.

    `session.placing` is appended only where a Turn found this Session's pod absent, so
    counting two is a statement about what the control plane observed rather than that
    two Turns ran. The two failing together is the difference between a pod that
    lingered and a placement the platform never announced.

    A completed second Turn is asserted in the same case, and it carries the half of
    the premise that cannot be seen directly. `place_for` is called after `place` and
    before the Turn is dispatched, and it raises rather than returning when a file
    cannot be delivered -- so a Turn that reached completion is a Turn whose whole
    attachment set was pushed again. The push itself is invisible from out here: the
    route answers 204 for a skip and 204 for a write, and no event records either.
    """
    kinds = [one["type"] for one in run.after_second]
    assert "turn.failed" not in kinds, kinds
    assert kinds.count("turn.completed") == 2, kinds
    assert kinds.count("session.placing") == 2, kinds


@requires_the_cluster
def test_the_agents_edit_survived_the_next_turns_placement(run: _Run) -> None:
    """The finding: the re-pushed attachment did not take the agent's work with it.

    This is the read that used to come back without the code. The second Turn placed a
    fresh pod and its placement re-pushed the document the tenant uploaded -- bytes that
    do not contain this token -- over a name the workspace already held, and before the
    fix the file the agent had edited was replaced by them, silently, with the Turn
    reporting success.

    Read after the Turn rather than during it, because the placement runs after the pod
    is ready and before the agent is given an instruction: the moment the overwrite used
    to happen is inside the second Turn, and a read taken once that Turn has closed is
    on the far side of it.
    """
    assert run.read_after_the_placement.status_code == 200, (
        run.read_after_the_placement.text
    )
    assert run.edited_nonce in run.read_after_the_placement.text, (
        f"the agent's line was in {_FILENAME} after the first Turn and is not there "
        f"after the second Turn's placement; the file now reads "
        f"{run.read_after_the_placement.text[:400]!r}"
    )


@requires_the_cluster
def test_the_placement_left_the_file_byte_for_byte_alone(run: _Run) -> None:
    """Not merely that the token is there, but that nothing about the file moved.

    The case above would also be satisfied by a placement that rewrote the file into
    something that still happened to carry the token -- a merge, a restore of an
    interleaved copy, an append of the original document underneath the agent's line.
    Equality across the Turn says the route did nothing at all to it, which is what
    skipping means and is the only outcome that leaves a tenant's edit exactly as the
    tenant's agent left it.

    Meaningful only alongside the case that says an edit landed. Two identical reads of
    a file nothing ever touched would satisfy this on their own.

    A failure here has two readings and the message names both, because one of them is
    not about the property: either the placement rewrote the file, or the second Turn's
    agent did -- and it was asked in as many words not to.
    """
    assert run.read_after_the_placement.status_code == 200, (
        run.read_after_the_placement.text
    )
    assert run.read_after_the_placement.text == run.read_after_the_edit.text, (
        f"{_FILENAME} changed across the second Turn; either its placement rewrote the "
        "file or that Turn's agent did, and the prompt asked it to change nothing"
    )


@requires_the_cluster
def test_the_placement_put_no_second_copy_beside_it(run: _Run) -> None:
    """The whole attachment directory, either side of the placement, unchanged.

    This closes the one explanation the two cases above cannot: that the re-push
    happened and landed somewhere else, under a scratch name or a name of its own, in
    which case the edit would survive for a reason that has nothing to do with the route
    skipping. A new entry, a vanished one, or a changed length anywhere in this
    directory is that explanation showing itself.

    It also catches the route's own `.<name>.partial` scratch file surviving a push,
    which is the shape a half-written transfer leaves behind.
    """
    assert run.listed_after_the_edit.status_code == 200, run.listed_after_the_edit.text
    assert run.listed_after_the_placement.status_code == 200, (
        run.listed_after_the_placement.text
    )
    before = _shape(run.listed_after_the_edit)
    # Presence, not exclusivity. Requiring `before` to hold nothing else would fail on
    # a `.bak` or an editor swap file the Turn-1 agent left behind -- a model obeying
    # an instruction is not a guarantee -- and it would buy nothing: the explanation it
    # was written to exclude, a re-push that landed under some other name, shows up in
    # the equality below as an entry `after` has and `before` does not.
    assert _FILENAME in [name for name, _, _ in before], before
    assert _shape(run.listed_after_the_placement) == before


@requires_the_cluster
def test_the_next_turns_agent_read_its_predecessors_work_and_not_the_original(
    run: _Run,
) -> None:
    """The tenant-visible form of the claim, and the one that needs no mount at all.

    Every case above reads the volume through the control plane. This reads it through
    the agent, which is the only witness that says the *pod* mounts the file the edit
    survived in -- the placement writes through the shim container's narrowed mount and
    the agent works through the runtime container's whole one, and a fix that saved the
    bytes somewhere the agent could not see them would satisfy everything else here.

    The revision code is a token the attached document does not contain, so an agent
    handed a file the placement had overwritten could not produce it. Bounded to what
    was said after the first Turn's last sequence, because the first Turn was handed
    this token in its own prompt and is where a stray copy of it would be.
    """
    assert run.edited_nonce in run.second_turn_said, run.second_turn_said
