"""Many produced files, over several Turns, against the deployed cluster.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

**The defect this exists for presented as a phase-eight bug.** A run that produces a
handful of files per Turn shipped successfully seven times and lost the eighth Turn's
output after the expensive work was done. The cause was one constant doing two jobs:
the pod truncated its own walk of `out/` at the number of files one Turn *transfers*,
so a Session already holding that many never got the files it had just written into a
listing at all, and the filter that weighs already-delivered paths out of the transfer
bound cannot weigh a path it was never shown. Counted whole, a bound the docstring
described as per-Turn was a budget on distinct paths for the life of the Session -- and
because no route deletes a file out of a pod, the refusal repeated on every Turn after
it and the Session could never deliver anything again.

Every part of that is invisible to the in-process suite in one specific way, which is
why this file is here. `tests/control/test_output_shipout.py` drives the ship-out class
over a fake pod, so the listing it weighs is the listing the test wrote; the walk that
produced the defect lives in `session_shim/serve.py`, inside the pod image, and the two
halves only meet on a real cluster. A run against a stale Session image would also pass
every offline gate and still truncate at the old bound.

**Three Turns and one Session, because the defect is between Turns.** The first writes
a batch, the second writes another batch without the first having been removed, and the
third rewrites a path already delivered. One Turn proves nothing about a cumulative
bound; a fresh Session per Turn proves less than that.

**The bytes are compared, not their count.** A run that stores the right number of
files with the wrong contents is the failure a count cannot see, and it is the plausible
one here: the hop reads a length off the pod's own listing and caps its fetch at it.

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

from managed_agent.control.files.output_shipout import OUTPUT_COUNT_LIMIT
from managed_agent.core.ids import SessionId
from managed_agent.core.pod.workspace_contract import OUTPUT_DIR_NAME
from managed_agent.core.vocabulary import turn

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_TURN_DEADLINE_S: Final = 900
_SUBMIT_TIMEOUT_S: Final = 1200
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

_PER_BATCH: Final = 60
"""How many files each Turn writes.

Above the bound that used to govern the whole Session and far below the one that now
governs a Turn, which is what makes the run a test of the fix rather than of either
ceiling. Two batches of this is a hundred and twenty paths standing at once, which the
old code could not have held at any point in a Session's life; neither Turn on its own
is over the current per-Turn bound, so nothing here is expected to ship partially and
`output.partial` appearing at all is a failure.

Sixty rather than five hundred because every file is a real object written into a real
bucket over a real port-forward, and the run costs three model Turns either way. What
the number has to be is *more than the old cap*, not close to the new one.
"""

_BATCHES: Final = ("phase-a", "phase-b")

requires_the_cluster = pytest.mark.skipif(
    __import__("os").environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places real pods and calls a real model",
)


def _nonce() -> str:
    """Upper-case hex behind a word, because the model has to reproduce it exactly."""
    return f"MANY-{uuid4().hex[:12].upper()}"


def _relative(batch: str, index: int) -> str:
    """The lane-relative path one file lands at: below `out/`, separator included."""
    return f"{batch}/item-{index:03d}.txt"


def _body(nonce: str, batch: str, index: int) -> bytes:
    """What one file must contain, derived so every file's bytes differ from every
    other's. A batch that wrote the same line into sixty files would pass a byte
    comparison while the platform delivered one file sixty times."""
    return f"{nonce} {batch} {index:03d}\n".encode()


def _write_prompt(nonce: str, batch: str) -> str:
    """One shell command, given literally, rather than a description of the result.

    A prompt that describes sixty files leaves the model composing a loop, and a run
    that fails then has failed at the model rather than at the platform -- which is the
    one failure this file cannot tell apart from the one it is looking for. Handing over
    the command makes every failure downstream of it a platform failure.

    POSIX shell only: no `seq`, no bashisms, no process substitution. The pod's image is
    not this repository's to assume a shell of.
    """
    return (
        "Run exactly this shell command in your current working directory, then reply "
        "with the single word DONE and nothing else.\n\n"
        f"mkdir -p ./{OUTPUT_DIR_NAME}/{batch} && i=1; while [ $i -le {_PER_BATCH} ]; "
        f'do n=$(printf "%03d" $i); printf "{nonce} {batch} %s\\n" "$n" > '
        f"./{OUTPUT_DIR_NAME}/{batch}/item-$n.txt; i=$((i+1)); done\n\n"
        "Do not write any other file, and do not delete anything."
    )


def _rewrite_prompt(nonce: str) -> str:
    """Ask for the one thing the artifacts lane refuses: a delivered path, rewritten.

    Given as a command for the reason above and for one more. The workspace contract
    now tells the agent that a produced path is written once, so a model asked in prose
    to revise a file it has already produced may well decline -- which would be the
    contract working and would leave this case asserting nothing. A literal command
    takes that judgement out of the run.
    """
    target = _relative(_BATCHES[0], 1)
    return (
        "Run exactly this shell command in your current working directory, then reply "
        "with the single word DONE and nothing else. Ignore the guidance about writing "
        "each produced path only once; this is a deliberate test of what happens when "
        "it is not followed.\n\n"
        f'printf "{nonce} revised\\n" > ./{OUTPUT_DIR_NAME}/{target}\n\n'
        "Do not write any other file."
    )


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned.

    Newest-push, matching the files beside this one. It matters more here than in most:
    the walk this file exercises lives in the Session image, so a run against a stale
    one would truncate at the old bound and report the defect as unfixed.
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

    Deliberately bare: this file's subject is the outbound path across Turns, and every
    attachment a Session could carry is one more thing a reader would have to rule out.
    """
    with _client(base, tenant_id) as caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"many-{run.lower()}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"many-{run.lower()}",
                    "instructions": (
                        "You run the shell commands you are given, exactly as given, "
                        "and add nothing of your own to any file."
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
    base: str, tenant_id: str, session_id: SessionId, already: int
) -> list[dict[str, Any]]:
    """Poll until this Turn ends either way, and return the whole log.

    `already` is how many terminal events the log held before this Turn was submitted,
    which is what makes the wait about *this* Turn: a Session on its third Turn already
    carries two `turn.completed` events, and a poll looking for one at all returns
    instantly having measured nothing.

    Both outcomes are terminal. A ship-out that raises is recorded as a FAILED Turn by
    design, and waiting only for the completion would sit out the whole deadline on a
    Turn that ended in the first second -- which sends the reader after the wrong thing.
    """
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        if _terminals(events) > already:
            return events
        time.sleep(3)
    events = _events(base, tenant_id, session_id)
    pytest.fail(
        f"session {session_id} produced no terminal event in {_TURN_DEADLINE_S}s; "
        f"the log was {[one['type'] for one in events]}"
    )


def _terminals(events: list[dict[str, Any]]) -> int:
    return sum(one["type"] in ("turn.completed", "turn.failed") for one in events)


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
class _Produced:
    """One Session's worth of runs, for the cases to read."""

    session_id: SessionId
    tenant_id: str
    nonce: str
    events: list[dict[str, Any]]
    rewrite: httpx.Response
    base: str


@pytest.fixture(scope="module")
def produced() -> Iterator[_Produced]:
    """One Session, two writing Turns and one rewriting Turn, and the log they produced.

    Module-scoped so the three Turns run once. The teardown deletes the pod and its
    Secrets whatever happened, including a failure part-way through: three aborted runs
    once left forty-two pods squatting the namespace, after which the next run's
    scheduling refusal read as the cluster being out of capacity.

    The third Turn is waited out like the other two, and its refusal is read out of the
    log. Its response used to be the whole subject of a case here, because the route
    held its answer until the dispatch returned and the 409 carried the colliding path
    in `detail`. It answers 202 at admission now, so that response says only that the
    Turn was admitted -- and the refusal, with the path, travels in `turn.failed`.

    `base` is carried on the record because the port-forward closes when this fixture
    exits, so a case wanting to download a file afterwards would have nothing to reach.
    """
    nonce = _nonce()
    tenant_id = str(uuid4())
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        session_id = _a_session(base, tenant_id, image, nonce)
        try:
            events: list[dict[str, Any]] = []
            for batch in _BATCHES:
                with _client(base, tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
                    answered = caller.post(
                        f"/v1/sessions/{session_id}/events",
                        json={"prompt": _write_prompt(nonce, batch)},
                        headers={"Idempotency-Key": uuid4().hex},
                    )
                assert answered.status_code == 202, (batch, answered.text)
                events = _await_terminal(
                    base, tenant_id, session_id, _terminals(events)
                )
            with _client(base, tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
                rewrite = caller.post(
                    f"/v1/sessions/{session_id}/events",
                    json={"prompt": _rewrite_prompt(nonce)},
                    headers={"Idempotency-Key": uuid4().hex},
                )
            assert rewrite.status_code == 202, rewrite.text
            events = _await_terminal(base, tenant_id, session_id, _terminals(events))
            yield _Produced(
                session_id=session_id,
                tenant_id=tenant_id,
                nonce=nonce,
                events=events,
                rewrite=rewrite,
                base=base,
            )
        finally:
            _clean_up(session_id)


@requires_the_cluster
def test_neither_writing_turn_failed(produced: _Produced) -> None:
    """No `turn.failed` until the rewrite, and asserted before anything below it.

    A ship-out that raises is recorded as a failed Turn on purpose -- the alternative is
    a Turn reading as complete while what it produced is still only inside a pod about
    to die -- so `turn.failed` here is the outbound path's own loudest failure, and it
    must not be read as "the model did not write the files". This is also exactly where
    the defect used to land: the SECOND Turn failed, having produced its files.

    **The two are not exclusive, which is what this asserts around.** A Turn refused by
    ship-out carries BOTH events, and the third Turn of this run is exactly that shape.
    So the claim is about `turn.failed`: there is one, it belongs to the rewrite, and it
    comes after every announcement the two writing Turns made.

    **Why the third Turn still carries `turn.completed` at all**, since 2026-08-26 the
    marker is withheld when the completion seam fails: the seam ships the Rollout first
    and stops at the first stage that raises, so a refusal from the outputs stage proves
    the Rollout was stored. That Turn did reach the durability boundary the marker
    claims, and the thing that failed is the agent rewriting a path it had already
    delivered -- the tenant's own doing, answered 409. Withholding for that would fail a
    Turn that worked, which is why the withholding is narrower than the seam.
    """
    types = [one["type"] for one in produced.events]
    assert types.count("turn.failed") == 1, types
    assert types.count("turn.completed") == len(_BATCHES) + 1, types
    assert types.index("turn.failed") > _index_of_last(types, "output.produced"), types


def _index_of_last(types: list[str], wanted: str) -> int:
    return len(types) - 1 - types[::-1].index(wanted)


@requires_the_cluster
def test_every_file_of_both_batches_was_announced(produced: _Produced) -> None:
    """A hundred and twenty paths, each announced exactly once.

    Exactly once matters in both directions. Missing is the defect this file exists
    for; twice would mean the already-delivered filter stopped weighing what the lane
    holds, which costs a re-upload of everything on every Turn for the life of the
    Session and shows up nowhere else -- the bucket holds the right bytes either way.
    """
    announced = [
        str(one["payload"]["path"])
        for one in produced.events
        if one["type"] == "output.produced"
    ]
    expected = [
        _relative(batch, index)
        for batch in _BATCHES
        for index in range(1, _PER_BATCH + 1)
    ]

    assert sorted(announced) == sorted(expected), (
        f"announced {len(announced)} of {len(expected)}; "
        f"missing {sorted(set(expected) - set(announced))[:10]}"
    )


@requires_the_cluster
def test_the_second_turn_shipped_only_what_it_added(produced: _Produced) -> None:
    """The bound is on what a Turn ADDS, which is the whole of the fix.

    Nothing empties the agent's output directory between Turns and nothing should, so
    the pod re-offers every file it has ever produced on every Turn. Counted whole, the
    per-Turn bound was a budget on distinct paths for the Session's life. This asserts
    the count directly: the second Turn's announcements are the second batch and no part
    of the first, which is only true if the already-delivered filter really is reading
    back what the first Turn wrote.

    The window is submission to submission and not completion to completion, because
    ship-out runs AFTER the Turn's own completion event -- so a window opening at the
    first `turn.completed` contains the first Turn's sixty announcements, which is what
    this case is trying to prove are absent.
    """
    types = [one["type"] for one in produced.events]
    submissions = [i for i, one in enumerate(types) if one == "turn.submitted"]
    assert len(submissions) == len(_BATCHES) + 1, types
    second, third = submissions[1], submissions[2]
    after = [
        str(one["payload"]["path"])
        for one in produced.events[second:third]
        if one["type"] == "output.produced"
    ]

    assert len(after) == _PER_BATCH, after[:10]
    assert all(path.startswith(f"{_BATCHES[1]}/") for path in after), after[:10]


@requires_the_cluster
def test_neither_turn_was_partial(produced: _Produced) -> None:
    """Sixty is far below what one Turn ships, so a partial here means a bound moved.

    Asserted rather than assumed because `output.partial` is not a failure and would not
    stop the run: a Session silently shipping in fragments passes every case above,
    slowly, and this is the only thing that would say so.
    """
    partials = [one for one in produced.events if one["type"] == "output.partial"]
    assert partials == [], (partials, OUTPUT_COUNT_LIMIT)


@requires_the_cluster
def test_the_bytes_come_back_byte_for_byte(produced: _Produced) -> None:
    """Every one of the hundred and twenty, downloaded and compared as bytes.

    A file the right size with the wrong content is what a count cannot see, and it is
    the plausible failure: the hop reads a length off the pod's own listing and caps its
    fetch at it. The nonce exists in this process and in the bytes the agent was told to
    write, nowhere else, so bytes carrying it came back from that workspace.

    Every file rather than a sample, because a truncated walk loses a *contiguous tail*
    of a sorted listing -- a sample drawn from the front is exactly the sample that
    would not notice.
    """
    wrong: list[str] = []
    with _client(produced.base, produced.tenant_id) as caller:
        for batch in _BATCHES:
            for index in range(1, _PER_BATCH + 1):
                path = _relative(batch, index)
                got = caller.get(f"/v1/sessions/{produced.session_id}/artifacts/{path}")
                if got.status_code != 200 or got.content != _body(
                    produced.nonce, batch, index
                ):
                    wrong.append(f"{path} -> {got.status_code} {got.content[:60]!r}")

    assert wrong == [], (
        f"{len(wrong)} of {len(_BATCHES) * _PER_BATCH} wrong: {wrong[:5]}"
    )


@requires_the_cluster
def test_rewriting_a_delivered_path_is_refused_and_the_path_is_named(
    produced: _Produced,
) -> None:
    """The refusal a tenant can act on, and the one this surface used to get wrong.

    It answered 502 `turn.undeliverable` under cause `pod_unreachable`, which told the
    caller the platform had failed and that a retry was the move. Both are wrong: the
    pod was reachable, the seal on the artifacts lane refused the write, and a retry
    re-runs an agent that writes the same path again. The colliding path is what makes
    the next move -- write it under a different name -- readable from the refusal alone.

    **Read out of `turn.failed`, not off the submission**, because the submission no
    longer carries it: the route answers 202 at admission and runs the Turn on a
    background task, so the 409 that once held `detail.path` never happens. The path
    travels in the closing event instead. A run in which the model simply declined to
    overwrite the file produces no `turn.failed` at all, which is why the absence is
    graded separately from the cause.
    """
    failures = [one for one in produced.events if one["type"] == "turn.failed"]
    assert failures, (
        "the Turn ended without a failure, so either the agent declined to overwrite "
        "the file or the lane accepted a revision it must refuse; the log read "
        f"{[one['type'] for one in produced.events]}"
    )
    payload = failures[-1]["payload"]
    assert payload["cause"] == turn.TurnFailureCause.OUTPUT_NOT_REVISABLE.value, payload
    assert payload["path"] == _relative(_BATCHES[0], 1), payload


@requires_the_cluster
def test_the_refused_rewrite_left_the_delivered_bytes_alone(
    produced: _Produced,
) -> None:
    """The seal's actual promise, which the refusal alone does not prove.

    A refusal that had already replaced the object would be worse than an acceptance: a
    tenant holding a digest they checked earlier would find the bytes changed under a
    path the platform had just told them could not be revised.
    """
    path = _relative(_BATCHES[0], 1)
    with _client(produced.base, produced.tenant_id) as caller:
        got = caller.get(f"/v1/sessions/{produced.session_id}/artifacts/{path}")

    assert got.status_code == 200, got.text
    assert got.content == _body(produced.nonce, _BATCHES[0], 1), got.content[:200]
