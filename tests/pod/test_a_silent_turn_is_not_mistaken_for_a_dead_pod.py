"""A Turn that puts nothing on the wire for minutes finishes, against a real pod.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

**The defect this exists for cost a tenant three Turns and one wrong diagnosis.** The
platform read a gap between output bytes as a dead pod and killed the Turn at 120
seconds, reporting `pod_unreachable`. An agent writing a file emits nothing for as long
as the write takes, so the busiest a Session ever gets looked exactly like death. The
tenant cut the same payload into smaller pieces and it walked straight through --
nothing had got smaller except the silence. Worse than the lost Turns: because one cause
name covered four unrelated failures, a falsifiable theory about *which* step was dying
looked confirmed when it was wrong, and the tenant trimmed an argument that was never
the problem.

**Why this cannot be an in-process test, and why the offline suite missing it was not an
oversight.** `tests/control/test_a_turn_outlives_its_request.py` proves the property
against fakes, and its dispatch returns when the test says it returns. The gap that
killed real Turns is a gap in an HTTP read across a port-forward into a pod, produced by
a subprocess in that pod doing work; a fake cannot produce it because a fake has no
socket to go quiet on. The two halves only meet on a real cluster, and a control plane
still carrying a per-gap deadline would pass every offline gate and fail here.

**Silence is produced deterministically rather than by writing something large.** A big
file is how the defect was found, but its quiet stretch depends on disk, on the model's
choice of how to write it, and on how much the agent narrates -- a test that has to
generate several megabytes to be valid is a test that goes green for the wrong reason on
a fast day. A shell command that sleeps produces the identical shape on the wire, a tool
call that starts and then says nothing, for a duration the test picks. What is being
tested is the platform's reading of silence, and the silence does not care what made it.

**The sleep is longer than the deadline that was retired, not longer than the one that
replaced it.** 150 seconds is comfortably past the old 120 and nowhere near the hour a
Turn is now allowed, which is what makes a pass here evidence about the change rather
than about either bound. A failure at roughly 125 seconds is the old deadline still
live in the cluster -- most likely a control plane that was not rolled.

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
from cluster_access import forwarded

from managed_agent.core.ids import SessionId

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_TURN_DEADLINE_S: Final = 900
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

_SILENCE_S: Final = 150
"""How long the Turn's tool call says nothing.

Past the retired 120-second inter-byte deadline by a margin wider than scheduling noise,
and far below the hour a Turn is now given. A smaller number would pass under the old
code too and prove nothing; a much larger one would spend the run's time buying no extra
confidence, because the property under test is "no bound below the total exists", and a
bound that does not fire at 150 seconds is not going to fire at 300.
"""

_RETIRED_DEADLINE_S: Final = 120
"""The bound this test exists to prove is gone.

Named rather than written into the assertion message as a literal, so that a reader who
finds this test failing at about this many seconds knows immediately what they are
looking at: not a slow cluster, but a control plane still running the old code.
"""

_SUBMIT_ACK_S: Final = 30
"""How long the POST may take to come back with its 202.

Generous by two orders of magnitude against what the route now does, and still far below
`_SILENCE_S`, which is the only comparison that matters. The route has declared
`status_code=202` since it was written and used to hold the connection for the whole
Turn anyway; a POST that takes longer than the Turn's own silence is that old behaviour,
whatever number it comes back with.
"""

requires_the_cluster = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places a real pod and calls a real model",
)

_PROMPT: Final = (
    "Run exactly this one shell command and nothing else, then tell me the word it "
    f"printed: sleep {_SILENCE_S} && echo finished-after-the-silence"
)


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned.

    Newest-push, matching the files beside this one. It used to be true that a stale
    Session image could not invalidate this case -- the silence is produced by a shell
    command and the retired deadline lived in the control plane -- and resolving was
    only about not failing on a pod that will not start. The progress cases at the foot
    of this file ended that: the emitter runs *in the pod*, so they grade the image this
    resolves, and an environment pinned to a digest built before the emitter reports
    nothing at all. That is not a defect and the assertion says so, because a tenant may
    pin an old digest forever and nothing will ever move them off it.
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

    Deliberately bare. The subject here is what the platform does with a quiet stream,
    and every attachment a Session could carry is one more thing a reader would have to
    rule out before believing the result.
    """
    with _client(base, tenant_id) as caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"silent-{run.lower()}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"silent-{run.lower()}",
                    "instructions": (
                        "You run the shell commands you are given, exactly as given, "
                        "and you do not narrate while a command is running."
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


def _terminals(events: list[dict[str, Any]]) -> int:
    return sum(one["type"] in ("turn.completed", "turn.failed") for one in events)


def _await_terminal(
    base: str, tenant_id: str, session_id: SessionId
) -> list[dict[str, Any]]:
    """Poll until this Turn ends either way, and return the whole log.

    Both endings are terminal and both are read. Waiting only for the completion would
    sit out the full deadline on a Turn that failed in the first second, which sends the
    reader after a timeout when what happened was a refusal -- and a refusal is the
    interesting outcome here, because its cause names what went wrong.
    """
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        if _terminals(events) > 0:
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
class _Quiet:
    """One Turn that went quiet, and the timings taken around it."""

    session_id: SessionId
    events: list[dict[str, Any]]
    submit_status: int
    submit_seconds: float
    turn_seconds: float


@pytest.fixture(scope="module")
def quiet() -> Iterator[_Quiet]:
    """Place a Session, submit one deliberately silent Turn, and time both halves.

    Module-scoped because the run costs a real pod, a real model call and two and a half
    minutes of deliberate waiting. Every case below reads this one result rather than
    paying for it again.
    """
    run = uuid4().hex[:8]
    tenant_id = str(uuid4())
    image = _session_image()
    session_id: SessionId | None = None
    try:
        with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
            session_id = _a_session(base, tenant_id, image, run)

            began = time.monotonic()
            with _client(base, tenant_id) as caller:
                answered = caller.post(
                    f"/v1/sessions/{session_id}/events",
                    json={"prompt": _PROMPT},
                    headers={"Idempotency-Key": uuid4().hex},
                )
            acknowledged = time.monotonic()

            # Checked here rather than left to the poll, and this file paid for the
            # rule rather than inheriting it. The first run of this case omitted the
            # `Idempotency-Key` header above, the route answered 400 in milliseconds
            # naming the missing field, and because nothing read that status the run
            # went on to poll an empty log for the full 900 seconds and reported "no
            # terminal event" -- fifteen minutes spent turning a precise answer into a
            # vague one. Every refusal this route can give is instant and says what is
            # wrong; a poll converts all of them into the same timeout.
            assert answered.status_code == 202, (
                "the submission was refused, so no Turn ever ran and the poll below "
                f"would spend {_TURN_DEADLINE_S}s discovering an empty log. The route "
                f"said: {answered.text}"
            )

            events = _await_terminal(base, tenant_id, session_id)
            ended = time.monotonic()

        yield _Quiet(
            session_id=session_id,
            events=events,
            submit_status=answered.status_code,
            submit_seconds=acknowledged - began,
            turn_seconds=ended - began,
        )
    finally:
        if session_id is not None:
            _clean_up(session_id)


@requires_the_cluster
def test_the_turn_survived_its_own_silence(quiet: _Quiet) -> None:
    """The Turn completed, having said nothing for longer than the retired deadline.

    This is the whole file in one assertion. A `turn.failed` here is the defect back,
    and its cause is printed rather than summarised because after the split each cause
    sends a reader somewhere different.
    """
    types = [one["type"] for one in quiet.events]
    failed = [one for one in quiet.events if one["type"] == "turn.failed"]
    causes = [one.get("payload", {}).get("cause") for one in failed]
    assert "turn.completed" in types, (
        f"the Turn did not complete after {quiet.turn_seconds:.0f}s; "
        f"causes were {causes}, and the log was {types}"
    )
    assert not failed, f"the Turn failed with {causes}; the log was {types}"


@requires_the_cluster
def test_the_turn_really_did_outlast_the_retired_deadline(quiet: _Quiet) -> None:
    """The run is only evidence if it actually spent longer than 120 seconds quiet.

    Without this, a cluster that somehow answered in ten seconds would pass the case
    above having tested nothing -- the agent could have declined to run the command, or
    the shell could have rejected it. Asserting the elapsed time is what makes the
    completion mean the silence was survived rather than avoided.
    """
    assert quiet.turn_seconds > _RETIRED_DEADLINE_S, (
        f"the Turn finished in {quiet.turn_seconds:.0f}s, which is inside the "
        f"{_RETIRED_DEADLINE_S}s deadline this case exists to outlast -- so it proves "
        "nothing. The agent most likely did not run the command it was given."
    )


@requires_the_cluster
def test_the_submission_was_acknowledged_long_before_the_turn_ended(
    quiet: _Quiet,
) -> None:
    """The POST returned its 202 while the Turn was still running.

    The route has advertised `status_code=202` since it was written and awaited the
    whole dispatch inline anyway, so the promise was a completed Turn wearing an
    accepted Turn's status code. The comparison that settles it is not the absolute
    number but the ratio: an acknowledgement that came back before the Turn's own
    silence had finished cannot have waited for the Turn.
    """
    assert quiet.submit_status == 202, (
        f"submission answered {quiet.submit_status} rather than 202"
    )
    assert quiet.submit_seconds < _SUBMIT_ACK_S, (
        f"the POST took {quiet.submit_seconds:.0f}s to acknowledge, which is the "
        "dispatch still being awaited inside the request"
    )
    assert quiet.submit_seconds < quiet.turn_seconds, (
        f"the POST took {quiet.submit_seconds:.0f}s and the Turn took "
        f"{quiet.turn_seconds:.0f}s; an acknowledgement is not supposed to wait for "
        "the work it acknowledges"
    )


_MINIMUM_REPORTS: Final = 2
"""How many progress reports this Turn's silence must have produced to mean anything.

The shim reports every thirty seconds and the runtime is quiet for `_SILENCE_S`, so a
healthy run produces four or five. Two is asserted instead, and the gap is deliberate:
the run is timed against a live cluster whose placement latency is not fixed, and a
case that fails when a report lands a few seconds late is a case that gets muted. Two
is the smallest number that still separates the two failures worth catching -- zero
means the type never traversed the route at all, and one means it was emitted once and
the ticker died, which is the same silence this file exists to end wearing a single
report as a disguise.
"""

_ABSENT: Final = -1
"""What a missing numeric field reads as, so its absence is loud rather than plausible.

Zero would be a defensible default and is the wrong one here: an `idle_ms` of zero is
a value a healthy report could genuinely carry, so a wired-to-nothing field and a
just-ticked one would be indistinguishable. A negative number is a value no counter in
this payload can hold, which is what makes it evidence.
"""

_LIVELY_IDLE_MS: Final = _RETIRED_DEADLINE_S * 1000
"""The ceiling `idle_ms` must stay under while the *published* stream is quiet.

This constant is the finding, and it is the opposite of what this case was first
written to assert. The expectation going in was that a Turn told to `sleep 150` would
report an `idle_ms` climbing towards 150_000, because the pod was understood to be
silent. The first honest run said otherwise: five reports carrying
`[23784, 23781, 21074, 18839, 16407]`, never once above twenty-four seconds.

The reason is that the two streams are not the same stream. `idle_ms` is measured from
the last frame the shim received from the runtime (`turn_runner.py:596-598`, set before
any mapping, so it counts frames that are dropped rather than published), while the
retired inter-byte deadline watched what the pod *wrote out*. The runtime talks to the
shim throughout a long shell command; the event log goes quiet because almost none of
that is publishable. So the pod being killed at 120 seconds was never a pod that had
stopped -- it was a live process whose evidence of life was not on the wire anybody was
watching.

Asserting the ceiling rather than a floor is what turns that into a guard: if `idle_ms`
ever exceeds the deadline this file exists to retire, either the runtime genuinely
stopped talking, or the field has been repointed at the published stream and has become
a second copy of the signal it was built to replace.
"""


def _reports_of_the_only_turn(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """This Turn's progress reports, with the one-Turn assumption made checkable.

    Filtering on `type` alone is what the cases here used to do, and it is right only
    because this fixture submits exactly one Turn. That is a property of the fixture and
    not of the platform: a Session's log holds every Turn it has ever run, `frames` is a
    per-Turn counter that restarts at each one, and a peer session measuring a real
    two-Turn workload saw it go 45 -> 123 and then back to 56. So a session-scoped
    `frames[-1] > frames[0]` is false on the second Turn of any real run, and a case
    asserting it would be passing by luck rather than by construction.

    Grouping and then insisting on a single group is what turns that luck into a check:
    if this fixture ever grows a second Turn, this fails and names the count, instead of
    quietly comparing counters from two different Turns.
    """
    reports = [one for one in events if one["type"] == "turn.progress"]
    turns = {str(one.get("payload", {}).get("turn_id")) for one in reports}
    assert len(turns) <= 1, (
        f"reports from {len(turns)} Turns are mixed together: {turns}. `frames` "
        "restarts per Turn, so comparing across them is meaningless -- scope by "
        "turn_id."
    )
    return reports


@requires_the_cluster
def test_a_progress_report_traversed_the_route_and_kept_coming(quiet: _Quiet) -> None:
    """`turn.progress` reached the tenant's event log, more than once, from a real pod.

    Everything else in this file grades the deadline. This grades the path: the shim
    emits the type, `TurnEventLine` carries it, the streaming route admits it,
    `pod_channel` appends it, and a tenant reading the log gets it. Each of those was
    read and argued before this ran, and reading is the weaker evidence -- the repo has
    a record of a type that passed the allowlist and then killed every Turn in the
    cluster on a validation error one step further in.
    """
    reports = _reports_of_the_only_turn(quiet.events)
    types = [one["type"] for one in quiet.events]

    assert len(reports) >= _MINIMUM_REPORTS, (
        f"the Turn was quiet for {quiet.turn_seconds:.0f}s and produced "
        f"{len(reports)} progress report(s); the log was {types}. Zero means the type "
        "did not survive the trip out of the pod, and the pod image is the first place "
        "to look -- an environment pinned to a digest built before the emitter is "
        "silent by construction and is not a defect."
    )


@requires_the_cluster
def test_a_progress_report_carries_the_silence_rather_than_a_zero(
    quiet: _Quiet,
) -> None:
    """`idle_ms` measured the real silence, and the counts are monotonic.

    Split from the case above because "the type arrived" and "the type was true" fail
    for different reasons and send a reader to different files -- the first to the
    route, the second to the ticker. A report that arrives carrying zeroes would pass
    the first case completely.
    """
    reports = _reports_of_the_only_turn(quiet.events)
    assert reports, "no progress report to inspect -- see the case above"

    idles = [int(one.get("payload", {}).get("idle_ms", _ABSENT)) for one in reports]
    frames = [int(one.get("payload", {}).get("frames", _ABSENT)) for one in reports]

    assert _ABSENT not in idles + frames, (
        f"a report arrived without idle_ms or frames: idles={idles}, frames={frames}. "
        "This line exists because the monotonic check below passes on a list of "
        "sentinels -- a field that is entirely absent is sorted, so without this the "
        "strongest-looking assertion in the case is the one that cannot fail. That is "
        "not hypothetical: the first live run of this case read the event body under "
        "the wrong key and every value came back a sentinel."
    )
    assert max(idles) < _LIVELY_IDLE_MS, (
        f"idle_ms reached {max(idles)}ms across {idles} while the published stream was "
        f"quiet, which is past the {_RETIRED_DEADLINE_S}s deadline this file retires. "
        "Either the runtime really did stop talking to the shim, or idle_ms is now "
        "measuring what the pod published rather than what it received -- and in that "
        "second case it has quietly become the signal it was built to replace."
    )
    assert frames == sorted(frames), (
        f"frame counts went backwards across the Turn: {frames}. They are a running "
        "total, and a reader comparing two reports concludes work happened only "
        "because the number cannot fall."
    )
    assert frames[-1] > frames[0], (
        f"frames did not move across the whole Turn: {frames}. Sortedness alone is "
        "satisfied by a counter stuck at one value, so it cannot tell a runtime that "
        "worked throughout from one that emitted nothing after the first report -- "
        "which is exactly the wedge this signal exists to make visible."
    )


@requires_the_cluster
def test_no_progress_report_landed_after_the_turn_had_ended(quiet: _Quiet) -> None:
    """The ticker was awaited, not merely asked to stop.

    `run_turn` cancels the reporting task and then awaits it, so a report already
    inside its append finishes before the Turn's terminal event is written. A bare
    `cancel()` would leave that report to land afterwards, and a reader folding the log
    forward would see a Turn still working after it had ended -- the one ordering the
    shim genuinely cannot allow. Cheap to assert here and impossible to notice by hand.
    """
    types = [one["type"] for one in quiet.events]
    assert "turn.completed" in types, "the Turn did not complete -- see the first case"

    ended = types.index("turn.completed")
    after = [one for one in types[ended + 1 :] if one == "turn.progress"]
    assert not after, (
        f"{len(after)} progress report(s) landed after turn.completed; the log was "
        f"{types}. The ticker outlived its Turn."
    )
