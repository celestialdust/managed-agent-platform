"""Two tenants, two Sessions, two Turns at once against the deployed control plane.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

Why this is a separate file from the two beside it. `test_the_control_plane_places_a_
session_pod.py` grades whether the deployed control plane can create the objects a
Session needs, and deliberately registers an image that is never pulled: its findings
are about what the API server accepted, so a pod that starts would be beside the point
and would cost every run a real image pull. `test_the_placement_code_really_places_a_
running_pod.py` drives the placement code directly with a key of its own, which is how a
Turn was first made to complete. Neither can say what this one does: that the platform
serves more than one tenant at the same time without their work meeting.

**What this proves.** Two Sessions, under two different tenants, each get their own pod;
both pods are RUNNING in the same observation; both Turns complete through the real
model; and each Session's answer is its own. The prompts ask for different words, so an
answer landing in the wrong Event Log is a failure with a name rather than a suspicion.

**What it does not prove, stated because the difference is easy to lose.** It does not
show the two Turns were in the model at the same instant -- both pods are observed
Running together, but the Turns are submitted asynchronously and either could finish
first. It does not show the platform holds up at any scale beyond two. And two is chosen
because it is the smallest number that can go wrong in the way that matters, not because
it is a load figure.

**The hazard this exists for is real and was live in this repository.** Until 2026-08-23
a Session pod carried `subdomain` and no `hostname`, so no pod had a DNS record of its
own; the headless Service's bare name resolved instead, to one arbitrary Session pod. A
platform that fell back to it would deliver one tenant's Turn into another tenant's pod
and report success. `docs/lessons.md` carries that entry. This file is the case that
would have failed loudly rather than a comment saying it cannot happen.

NO KEY OR TOKEN VALUE IS PRINTED OR ASSERTED ON. This file mints nothing and reads no
Secret; every assertion is on an event type, an answer string, a pod name or a count.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import NAMESPACE, forwarded, kubectl

from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import SessionId

_GATE: Final = "MAP_CLUSTER_TESTS"
_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080

_REGION: Final = "us-east-1"
_REPOSITORY: Final = "map/session-shim"

_MODEL: Final = "gsds-claude-opus-4-6"
"""The model this account has actually deployed, so a Turn can reach one.

The other entry in the routing table names a credential that exists in no account, and
a Session bound to it fails at the vault read with a 503 -- a failure with nothing to do
with this file's subject that would look exactly like one that does.
"""

_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

_ANSWERS: Final = (
    "alpaca",
    "walrus",
    "cobra",
    "ferret",
    "marmot",
    "toucan",
    "gibbon",
    "narwhal",
    "gecko",
    "lemur",
    "otter",
    "badger",
    "puffin",
    "ocelot",
    "tapir",
    "quokka",
    "manatee",
    "civet",
    "kestrel",
    "wombat",
    "caribou",
    "ibex",
    "jackal",
    "heron",
)
"""One word per concurrent tenant, and they must all be different.

Distinctness is the whole mechanism: each tenant asks for its own word, so an answer
landing in the wrong Event Log is a failure naming both words rather than a suspicion.
Two tenants sharing a word would make the crossed-answer check pass vacuously, which
`test_the_words_are_all_different` refuses.

Single common nouns, because the model has to reproduce one exactly and a token it would
spell two ways would fail this test for reasons that are not the platform's.
"""

_CONCURRENCY_ENV: Final = "MAP_CONCURRENT_SESSIONS"
_DEFAULT_CONCURRENCY: Final = 2
"""How many tenants run at once, and why the default is the smallest interesting number.

Two is the smallest count that can exhibit the failure this file exists for, and it is
the default because every cluster run pays for this: at twenty-four the case takes about
a minute and leaves twenty-four pods to reap, which is not a price to charge somebody
checking one unrelated thing.

Raise it with the environment variable to measure capacity rather than correctness.
Measured on 2026-08-23 against `map-dev` this way: 2, 6, 12 and 24 tenants all
completed with no crossed answers, the last in 57s across four nodes.

Twenty-four FAILED twice before that, and neither failure was concurrency. The first was
the placer refusing `Unschedulable` instead of letting the autoscaler add a node; the
second an autoscaled node whose seccomp profile had not landed on it yet. Both are fixed
and both are in `docs/lessons.md`. The figure is what the platform does now.

The ceiling is the cluster's, not the platform's: `cluster-autoscaler.yaml` caps the
nodegroup at four nodes and a t3.medium holds seventeen pods, so somewhere above forty
concurrent Sessions this stops being a test of the platform. Nothing here checks that
bound, so a very large value fails as a capacity refusal naming the shortfall.
"""


def _answers() -> tuple[str, ...]:
    """The distinct words for this run, as many as the concurrency asks for."""
    asked = int(os.environ.get(_CONCURRENCY_ENV, _DEFAULT_CONCURRENCY))
    assert asked >= 2, f"{_CONCURRENCY_ENV}={asked} is not a test of concurrency"
    assert asked <= len(_ANSWERS), (
        f"{_CONCURRENCY_ENV}={asked} needs {asked} distinct words and this file holds "
        f"{len(_ANSWERS)}. Add more rather than repeating one: two tenants sharing a "
        "word make the crossed-answer check pass without checking anything."
    )
    return _ANSWERS[:asked]


_SUBMIT_TIMEOUT_S: Final = 660
"""How long this client waits for the API to answer a Turn submission.

Larger than it looks like it should be, and the reason is a fact about the platform
rather than about this test: a Session's FIRST Turn is answered only once its pod has
been placed, so the response is held for the whole placement. On an autoscaled cluster
that includes waiting for a node -- measured at 1m0s for the ASG plus the image pull --
so the adapter's own bounds add to about ten minutes in the worst case and this must
exceed them or the client gives up on a placement that was going to succeed.

It did, at 120 s: twenty-four Sessions submitted at once, the autoscaler added two
nodes, and this client gave up while the platform was still working. That is worth
knowing about the API's shape, not only about this constant.
"""

_TURN_DEADLINE_S: Final = 420
"""How long one Turn is given to reach a terminal event.

Generous on purpose. It covers a cold image pull on a node that has never run the
Session image, the runtime opening its thread, and one model round trip -- and the first
of those has been measured at over two minutes. A tighter bound turns a slow pull into a
failure that reads as the model never answering.
"""

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"{_GATE}=1 not set. This case places two pods in the real {NAMESPACE} "
        "namespace and calls a real model; it must not run because somebody typed "
        "pytest."
    ),
)


@dataclass(frozen=True, slots=True)
class _Tenant:
    """One tenant, the word its Turn asks for, and the ids the API handed back.

    A frozen record rather than four parallel lists, because the whole subject here is
    which value belongs to which tenant: a test that tracked them separately could pass
    while pairing them wrongly, which is the exact defect it is looking for.
    """

    tenant_id: str
    answer: str
    session_id: SessionId
    turn_id: str = ""
    """Empty until the Turn is submitted, because the record exists before it is.

    The record has to exist first: it names the pod the cleanup deletes, and a run that
    dies during submission still made Sessions the control plane will place pods for.
    Empty rather than None so the leak check below reads the same either way -- an id
    nothing was given cannot appear in another tenant's log, and `"" not in str(log)` is
    false for every log, which would pass vacuously. The check skips an empty one and
    says so.
    """

    @property
    def pod_name(self) -> str:
        return pod_name_for(self.session_id)


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned.

    Newest-push, matching the two files beside this one: the pod manifest's own
    reference is a placeholder that resolves nowhere, and a digest written into this
    file would pin the run to whatever ECR held on the day somebody typed it.
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


def _client(base: str, tenant_id: str, timeout: int = 60) -> httpx.Client:
    return httpx.Client(
        base_url=base, timeout=timeout, headers={TENANT_HEADER: tenant_id}
    )


def _register(base: str, tenant_id: str, image: str) -> SessionId:
    """An environment, a definition and a Session for one tenant, over the REST API.

    Nothing here touches the database or the cluster. That is the shape of the claim:
    the pods found below were put there by the deployed process, because this run has no
    other way to have created them.
    """
    with _client(base, tenant_id) as caller:
        environment = caller.post(
            "/v1/environments",
            json={"name": f"concurrent-{uuid4().hex[:8]}", "runtime_image": image},
        )
        assert environment.status_code == 201, environment.text
        definition = caller.post(
            "/v1/agents",
            json={
                "name": f"concurrent-{uuid4().hex[:8]}",
                "instructions": "Answer with the single word you are asked for.",
                "model": _MODEL,
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
                "budget_minor_units": 10_000,
                "budget_currency": "USD",
                "retention_days": 1,
            },
        )
        assert session.status_code == 201, session.text
        return SessionId(UUID(session.json()["id"]))


def _submit(base: str, who: _Tenant) -> str:
    """Submit one Turn asking for one word, and return the turn id it was given.

    202, not 200: the API accepts a Turn and dispatches it behind the response, so the
    answer arrives in the Event Log rather than on this connection. A 200 here would
    mean the API had changed shape under this file.
    """
    with _client(base, who.tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
        answered = caller.post(
            f"/v1/sessions/{who.session_id}/events",
            json={"prompt": f"Reply with exactly one word: {who.answer}"},
            headers={"Idempotency-Key": uuid4().hex},
        )
    assert answered.status_code == 202, answered.text
    turn_id: str = answered.json()["turn_id"]
    return turn_id


def _events(base: str, tenant_id: str, session_id: SessionId) -> list[dict[str, Any]]:
    with _client(base, tenant_id) as caller:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
    assert answered.status_code == 200, answered.text
    listed: list[dict[str, Any]] = answered.json()["events"]
    return listed


def _terminal(events: list[dict[str, Any]]) -> str | None:
    """The name of the terminal Turn event if one has been appended, else None.

    Both outcomes are looked for, not only the good one. A poll that waited for
    `turn.completed` alone would sit out its whole deadline on a Turn that failed in the
    first second and then report a timeout, which names the wrong cause.
    """
    for event in events:
        if event["type"] in ("turn.completed", "turn.failed"):
            return str(event["type"])
    return None


def _await_terminal(base: str, who: _Tenant) -> list[dict[str, Any]]:
    """Poll one Session's Event Log until its Turn ends, and return the whole log.

    Polling rather than the streaming route, because what is being graded is the
    durable record: a stream shows what one connection was told, and the Event Log is
    what the tenant can read back afterwards. Also, two streams held open from one test
    process would make the failure of either read as a failure of both.
    """
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, who.tenant_id, who.session_id)
        if _terminal(events) is not None:
            return events
        time.sleep(3)
    events = _events(base, who.tenant_id, who.session_id)
    pytest.fail(
        f"tenant {who.tenant_id} session {who.session_id} produced no terminal event "
        f"in {_TURN_DEADLINE_S}s; the log was "
        f"{[one['type'] for one in events]}"
    )


def _pod_phases(names: list[str]) -> dict[str, str]:
    """The phase of each named pod in one API call, or absent from the mapping.

    ONE call, deliberately. Two calls could see one pod Running and then the other
    Running after the first had gone, which is two Sessions in sequence -- exactly what
    this file must not mistake for two at once.
    """
    listed = kubectl(
        "get",
        "pods",
        "-o",
        "jsonpath={range .items[*]}{.metadata.name}={.status.phase}{'\\n'}{end}",
    )
    seen = dict(
        line.split("=", 1) for line in listed.splitlines() if line.count("=") == 1
    )
    return {name: seen[name] for name in names if name in seen}


def _clean_up(who: _Tenant) -> None:
    kubectl(
        "delete", "pod", who.pod_name, "--ignore-not-found", "--wait=false", check=False
    )
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete",
            "secret",
            f"{who.pod_name}-{suffix}",
            "--ignore-not-found",
            check=False,
        )


@requires_the_cluster
def test_two_tenants_take_a_turn_at_once_and_neither_gets_the_others_answer() -> None:
    """Two tenants, two pods running together, two answers that do not cross.

    The two Turns are submitted from two threads so that neither waits on the other's
    acceptance, and the pods are then observed in ONE `kubectl get pods` call, so
    "together" is one reading of the cluster rather than two readings stitched into a
    claim.

    Each prompt asks for a different single word. That is what makes a crossed answer a
    failure with a name: an assertion that both Turns merely completed would pass on a
    platform that served both of them out of one pod and gave both tenants the same
    reply.
    """
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        tenants = tuple((str(uuid4()), answer) for answer in _answers())
        registered = [
            (tenant_id, answer, _register(base, tenant_id, image))
            for tenant_id, answer in tenants
        ]
        # Every Session this run created, named before the first Turn is submitted, so
        # the cleanup below covers a run that dies DURING submission. It did not: an
        # earlier version built this list from the submissions' own results, so a client
        # timeout on one submission skipped the cleanup for all of them and left pods
        # squatting the namespace. Forty-two accumulated that way across three aborted
        # runs and the next run could not schedule at all -- reported, correctly, as the
        # cluster being out of capacity, which sent the reader after the wrong thing.
        who = [
            _Tenant(tenant_id=tenant_id, answer=answer, session_id=session_id)
            for tenant_id, answer, session_id in registered
        ]
        # Distinct pods, asserted before anything waits on them. Two Sessions sharing
        # one pod name would make every check below pass by reading one pod twice, and
        # it is the shape a pod name derived from something other than the Session id
        # would take -- so it is worth a line even though `pod_name_for` is a hash of
        # the id and cannot collide in practice.
        assert len({one.pod_name for one in who}) == len(who), who

        try:
            with ThreadPoolExecutor(max_workers=len(who)) as pool:
                turn_ids = list(pool.map(lambda one: _submit(base, one), who))
            who = [
                replace(one, turn_id=turn_id)
                for one, turn_id in zip(who, turn_ids, strict=True)
            ]
            _await_both_running(who)
            logs = _await_both_terminal(base, who)
            _assert_each_answer_is_its_own(who, logs)
        finally:
            for one in who:
                _clean_up(one)


def _await_both_running(who: list[_Tenant]) -> dict[str, str]:
    """Wait until ONE reading of the cluster shows every pod Running, and return it.

    Fails the case rather than returning a verdict, because there is nothing a caller
    could usefully do with "not yet" -- the deadline has already passed by then. The
    return value is the reading itself, for a caller that wants to say more about it.

    The single-reading requirement is the whole point and is enforced in `_pod_phases`,
    not here: two readings could each show one pod Running and together show two
    Sessions in sequence, which is exactly what this file must not accept as two at
    once.
    """
    names = [one.pod_name for one in who]
    deadline = time.monotonic() + _TURN_DEADLINE_S
    seen: dict[str, str] = {}
    while time.monotonic() < deadline:
        seen = _pod_phases(names)
        if len(seen) == len(names) and set(seen.values()) == {"Running"}:
            return seen
        time.sleep(3)
    pytest.fail(
        f"the {len(names)} session pods were never Running in one reading; "
        f"the last was {seen}"
    )


def _await_both_terminal(
    base: str, who: list[_Tenant]
) -> dict[str, list[dict[str, Any]]]:
    """Each Session's Event Log, once every Turn has ended.

    The two waits run in threads rather than one after the other. Sequentially, the
    second Session's Turn would be finishing while this process blocked on the first,
    and a deadline that is generous per Turn would become a deadline for both.
    """
    with ThreadPoolExecutor(max_workers=len(who)) as pool:
        logs = list(pool.map(lambda one: _await_terminal(base, one), who))
    return {one.tenant_id: log for one, log in zip(who, logs, strict=True)}


def _assert_each_answer_is_its_own(
    who: list[_Tenant], logs: dict[str, list[dict[str, Any]]]
) -> None:
    """Every Turn completed, and no Session's log carries another's word or turn id.

    Three separate claims, and the third is the one the other two cannot make. Both
    Turns completing says the platform works twice. Each log holding its own word says
    the answers did not cross. Neither log holding the other's turn id says the Event
    Logs themselves are not shared -- a leak that would show up as a tenant reading
    another tenant's work rather than as a wrong answer.
    """
    words = {one.answer for one in who}
    for one in who:
        log = logs[one.tenant_id]
        types = [event["type"] for event in log]
        assert "turn.completed" in types, (
            f"tenant {one.tenant_id} asked for {one.answer!r} and its Turn did not "
            f"complete; the log was {types}"
        )
        text = " ".join(
            str(event.get("payload", {}).get("text", "")) for event in log
        ).lower()
        assert one.answer in text, (
            f"tenant {one.tenant_id} asked for {one.answer!r} and the answer did not "
            f"contain it; the text was {text!r}"
        )
        for foreign in words - {one.answer}:
            assert foreign not in text, (
                f"tenant {one.tenant_id} asked for {one.answer!r} and its answer "
                f"carries {foreign!r}, which belongs to another tenant: {text!r}"
            )
        for other in who:
            if other is one:
                continue
            if not other.turn_id:
                continue
            assert other.turn_id not in str(log), (
                f"tenant {one.tenant_id}'s Event Log carries turn {other.turn_id}, "
                f"which was submitted by tenant {other.tenant_id}"
            )


def test_the_words_the_tenants_ask_for_are_all_different() -> None:
    """The distinctness the crossed-answer check rests on, graded without a cluster.

    Not behind the cluster gate, deliberately. Two tenants sharing a word would make
    `_assert_each_answer_is_its_own` pass while checking nothing -- the set difference
    it iterates would be empty -- and that is a defect somebody introduces while adding
    a word, at a keyboard, with no cluster in reach.
    """
    assert len(set(_ANSWERS)) == len(_ANSWERS), sorted(_ANSWERS)
    assert all(word.isalpha() and word.islower() for word in _ANSWERS), _ANSWERS
