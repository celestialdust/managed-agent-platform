"""A Session pod, in the real map-dev namespace, created by the deployed control plane.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NO POD WAS PLACED -- every other check
over this path reads YAML or drives a fake runner, and neither can say whether the API
server accepted a create from this ServiceAccount.

Nothing here creates a pod itself. It drives the REST API through a port-forward to the
deployed control plane, exactly as a tenant would, and then looks in the namespace for
an object whose name is derived from the Session id it was handed. That is what "the
control plane put it there" means, and it is why the name is computed rather than
searched for: a leftover `map-session-*` from another run would satisfy a search.

One finding here is an absence and it has its control in the same run: that the Tool
Gateway accepts the token the control plane signed is paired with a token minted here
under a throwaway key, which MUST be refused -- an endpoint that accepts everything and
an endpoint that verifies are indistinguishable from one probe.

**A pod exists only while its Turn is in flight, and that is why this file watches
rather than looks afterwards.** A pod is leased for exactly one Turn and given back when
that Turn ends (ADR-041), and `POST /v1/sessions/{id}/events` holds its response open
until the Turn is over -- so at the moment a submission answers, the pod and its three
Secrets have already been deleted. Every case below reads a snapshot a watcher took
while the submission was still in flight; none of them reads the namespace for itself.
Looking after the answer is what these cases used to do, and under the lease that finds
an absence which is correct rather than a control plane that placed nothing.

What this file still does not do is wait for RUNNING. The watcher stops at the pod
object, so no case here spends anything on a fact another file grades. The submission
underneath it is a different matter, and the cost is real and unavoidable: the response
is held for the whole Turn, so the one Turn this file takes pays an image pull and a
model round trip whether or not anything here reads the answer. That is the argument for
taking exactly one. Whether a Turn completes -- and whether two tenants' Turns stay
apart -- is `test_two_tenants_run_at_once_through_the_deployed_api.py`, which is built
to wait.

**The fourth case was reversed rather than deleted, and the second Turn it takes is
submitted immediately.** It asserted that a second Turn REUSES the pod the first one
placed -- one Session, one pod object, whatever a caller does -- which was the
platform's rule until ADR-041 withdrew it. The same instrument now reads the other way:
both Turns' pods carry one name, so only the UID the API server assigns per object can
say whether the second Turn got a pod of its own.

Submitting it with no wait in between is what keeps this from duplicating
`test_a_reaped_session_takes_another_turn.py`. A Turn sent straight after the last one
arrives while the previous pod is still inside its grace period, listed and stamped for
deletion, which reads as GONE -- and a Turn that finds GONE is placed by waiting that
pod out and creating in its place. That file waits the window out on purpose, so that it
grades the ABSENT path deterministically instead of racing ten seconds, and its own
docstring records that the GONE path is the one an interactive tenant takes and is not
graded there. It is graded here. Refusing that Turn is what the platform did until it
was fixed, and it broke every second Turn a tenant sent promptly.

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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
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
"""How long the watcher looks for a pod object before giving up on the Turn placing one.

Covers the compile, three Secret creates and one pod create against the real API server,
plus whatever the control plane's own queue is doing. It does NOT cover an image pull --
nothing here waits for RUNNING -- so a value in minutes would only ever be spent on a
control plane that is not going to place at all.

It bounds the watcher and not the submission. The submission is bounded by
`_SUBMIT_TIMEOUT_S`, which has to be far larger for a reason that belongs to the
platform rather than to this file.

A second Turn's placement can also include waiting the previous pod out -- the adapter
allows a minute for that -- so this has to exceed that wait plus the creates, and two
minutes does.
"""

_SUBMIT_TIMEOUT_S: Final = 660
"""How long this client waits for a submitted Turn to end.

Larger than it looks like it should be, because it covers a placement, a cold image pull
and a model round trip end to end. On an autoscaled cluster the placement alone has been
measured at a minute for the node plus fifteen seconds for the pull.

It used to be the HTTP timeout on a route that held its response for all of that. The
route answers 202 immediately now, so this is the polling deadline in
`_wait_out_the_turn` instead -- the same span, waited for in a different place.

It was 300, chosen when this file expected a synchronous refusal rather than a Turn. A
client that gives up here aborts a Turn the platform was going to finish, and the pod it
leaves behind is deleted by the cleanup rather than by the lease.
"""

_TERMINAL_DEADLINE_S: Final = _SUBMIT_TIMEOUT_S + 60.0
"""How long the watcher waits for the submission thread after it has taken its snapshot.

Longer than the client's own timeout by a margin, because the thread ends when that
timeout fires: a deadline shorter than it would report this file as hanging when what
is actually happening is an HTTP client giving up on schedule.
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

    Presence, and which object it was. A name cannot carry the second half:
    `pod_name_for` is a function of the Session id, so every pod this Session ever gets
    is called the same thing, and under a lease it gets one per Turn. The UID is
    assigned by the API server per object, so recording it says WHICH pod was seen and
    not merely that something by that name was there -- which is what makes a snapshot
    taken during a Turn attributable to that Turn afterwards.

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


def _session_token_of(session_id: SessionId) -> str | None:
    """The `x-map-session` header out of that Session's own compiled Secret.

    Read out of the object the control plane wrote, because that is the only place the
    token it signed exists -- the compiler mints it and hands it straight to the Secret.
    What comes back is passed to an HTTP client and to nothing else: never printed,
    never logged, never compared to a key.

    Called only from inside a Turn, because that Secret is deleted when the Turn ends.
    None rather than a failure when there is nothing to read, and the caller can tell
    the two absences apart without this saying which: whether the Secret existed at all
    is recorded beside the token, so an empty answer with the Secret present means a
    compiled document that carries no such header.
    """
    encoded = kubectl(
        "get",
        "secret",
        f"{pod_name_for(session_id)}-compiled",
        "-o",
        r"jsonpath={.data.config\.toml}",
        check=False,
    ).strip()
    if not encoded:
        return None
    document = tomllib.loads(base64.b64decode(encoded).decode())
    for server in document.get("mcp_servers", {}).values():
        headers = server.get("http_headers", {})
        if SESSION_TOKEN_HEADER_NAME in headers:
            token: str = headers[SESSION_TOKEN_HEADER_NAME]
            return token
    return None


def _secrets_present(pod: str) -> tuple[str, ...]:
    """Which of the three per-Session Secrets are in the namespace right now.

    Returned as the suffixes that ARE there rather than the ones that are not, so one
    reading answers both questions a caller has: a Turn in flight wants everything
    present, and a Turn that has ended wants nothing left.
    """
    return tuple(one for one in _SECRET_SUFFIXES if _exists("secret", f"{pod}-{one}"))


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


_ENDED = frozenset({"turn.completed", "turn.failed"})
"""The two events either of which means a Turn is over and the Session is free again.

Read out of the log rather than off the submission, because since `2316348` the API
answers 202 the moment a Turn is admitted and runs it on a background task. There is no
longer a response to wait on, and the only thing that says a Turn ended is one of these
appearing in its Session's events.
"""


def _take_a_turn(base: str, session_id: SessionId) -> httpx.Response:
    """Submit one Turn and return the submission's answer, once the Turn has ended.

    Blocks for the whole Turn, which is why every caller here runs it on a thread --
    but the block is this function's own polling now, not the API's. Until `2316348`
    the route held its response until the dispatch returned, and this could simply
    return the POST. It starts a background task and answers 202 immediately now, so a
    caller that returns on the POST is handed a Turn that has not begun: the pod is
    still initialising, and a second Turn submitted straight after is refused 409
    because the first is genuinely still running. Both of those were read as platform
    defects before the waiting moved here.

    A non-202 answer is returned as it stands. It is a refusal, there is no Turn to wait
    for, and the call sites grade the status line themselves.
    """
    with _client(base, timeout=_SUBMIT_TIMEOUT_S) as caller:
        answered = caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": "Reply with exactly one word: acknowledged"},
            headers={"Idempotency-Key": uuid4().hex},
        )
        if answered.status_code != 202:
            return answered
        _wait_out_the_turn(caller, session_id, str(answered.json()["turn_id"]))
        return answered


def _wait_out_the_turn(
    caller: httpx.Client, session_id: SessionId, turn_id: str
) -> None:
    """Poll this Session's events until THIS Turn ends, or fail saying where it got to.

    Keyed on the turn id and not merely on the event type, because a Session that has
    already taken a Turn carries that Turn's terminal event for ever -- so a check for
    "is there a `turn.completed` in here" is satisfied by the previous Turn before this
    one has started, which is precisely the second-Turn case this file exists to grade.
    """
    deadline = time.monotonic() + _SUBMIT_TIMEOUT_S
    seen: list[str] = []
    while time.monotonic() < deadline:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
        if answered.status_code == 200:
            events = [
                event
                for event in answered.json()["events"]
                if str((event.get("payload") or {}).get("turn_id")) == turn_id
            ]
            seen = [str(event["type"]) for event in events]
            if _ENDED.intersection(seen):
                return
        time.sleep(2)
    pytest.fail(
        f"turn {turn_id} on session {session_id} never ended inside "
        f"{_SUBMIT_TIMEOUT_S:.0f}s. Its events were {seen or 'none at all'}."
    )


def _event_types(base: str, session_id: SessionId) -> list[str]:
    with _client(base) as caller:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
    assert answered.status_code == 200, answered.text
    events: list[dict[str, Any]] = answered.json()["events"]
    return [event["type"] for event in events]


def _clean_up(session_id: SessionId) -> None:
    """Delete the pod and its three Secrets. The namespace is left as it was found.

    Ordinarily a no-op now, and kept for the runs where it is not. The lease deletes all
    four objects when a Turn ends, so a run that completed leaves nothing here to
    collect -- but a run that died mid-Turn, or one whose client gave up on a submission
    the platform was still carrying, leaves a pod nothing else will reap. Every delete
    tolerates absent, so the ordinary case costs four refusals nobody reads.
    """
    pod = pod_name_for(session_id)
    kubectl("delete", "pod", pod, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{pod}-{suffix}", "--ignore-not-found", check=False
        )


# ---------------------------------------------------------------------------
# One run, watched twice, read four ways
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Watched:
    """What one Turn had in the namespace while it ran, and what it left behind.

    A record taken as the Turn happened rather than a set of live reads, because the
    objects it describes exist only for the length of that Turn and the submission's own
    answer arrives after they are gone. Every field but the last was read at one
    sighting, so a case that pairs two of them is comparing two facts about one moment
    instead of two moments -- which, under a lease, is the difference between "this pod
    has no Secrets" and "somebody looked either side of the handback".
    """

    answered: httpx.Response
    """What the API finally said, which is after the Turn ended and the pod went."""

    pod_uid: str | None
    """The pod's UID at the first sighting, or None if none appeared while it ran.

    A UID rather than a bool, because it names the object that was seen. The pod's name
    is a function of the Session id and so is the same for every pod this Session is
    ever given, which makes "a pod called this was there" a weaker statement than it
    reads as -- and the failure message wants to be able to say which one.
    """

    secrets_present: tuple[str, ...]
    """Which of the three Secrets existed at that same sighting."""

    session_token: str | None
    """The `x-map-session` header out of the compiled Secret. Never printed."""


def _watch_one_turn(
    base: str, session_id: SessionId, ignoring: str | None = None
) -> _Watched:
    """Submit one Turn and read the namespace while that Turn is still being carried.

    The submission runs on a thread of its own, and that is the whole reason this
    function exists. The API holds its response until the Turn is over, and the lease
    deletes the pod and its Secrets before it answers -- so called inline there is no
    moment left at which any of this can be looked at. Everything the cases below read
    is gathered here, between the placement and the handback.

    The pod and its Secrets are read in one breath, and the ordering is the invariant
    being graded: the placer creates the three Secrets and then the pod, so a pod
    sighted with a Secret missing is a pod whose volume can never mount.

    `ignoring` is the UID of a pod this Turn must NOT be credited with, and it is what
    makes a second Turn observable at all. A Turn submitted straight after another finds
    the previous pod still listed, stamped for deletion and working through its grace
    period -- at the SAME NAME, because the name is a function of the Session. A watcher
    without this would latch onto that object within a second, report the Turn as having
    been placed a pod, and hand back the previous Turn's UID -- which then reads as one
    pod serving two Turns, the exact defect this file used to assert was impossible.

    Returns only once the submission has answered, so the caller is handed a Turn that
    is over and a namespace the platform has already been given back.
    """
    pod = pod_name_for(session_id)
    with ThreadPoolExecutor(max_workers=1) as pool:
        flying = pool.submit(_take_a_turn, base, session_id)
        uid, present, token = _watch_for_the_pod(pod, session_id, flying, ignoring)
        answered = flying.result(timeout=_TERMINAL_DEADLINE_S)
    return _Watched(
        answered=answered, pod_uid=uid, secrets_present=present, session_token=token
    )


def _watch_for_the_pod(
    pod: str,
    session_id: SessionId,
    flying: Future[httpx.Response],
    ignoring: str | None,
) -> tuple[str | None, tuple[str, ...], str | None]:
    """Poll until a pod that is not `ignoring` appears, then read it and its Secrets.

    Stops the moment the submission has answered, because after that the lease has
    already taken the pod back and a poll that kept going would spend its whole deadline
    on a namespace that is correctly empty.

    The Secrets are read only once a pod this Turn owns has been sighted, and that
    ordering is what makes them attributable. A pod being replaced has its Secrets
    deleted with it and the next set minted before the next pod is created, so a reading
    taken at the sight of the NEW pod cannot be describing the old one's.

    Reports nothing found rather than failing. Which case that is a finding for depends
    on what the caller was looking for -- the first Turn owes a pod, the second owes a
    different one -- and a helper that failed here would report all of them as the same
    thing and name none of them.
    """
    deadline = time.monotonic() + _PLACEMENT_DEADLINE_S
    while time.monotonic() < deadline:
        uid = _pod_uid(pod)
        if uid is not None and uid != ignoring:
            return uid, _secrets_present(pod), _session_token_of(session_id)
        if flying.done():
            return None, (), None
        time.sleep(1)
    return None, (), None


@dataclass(frozen=True, slots=True)
class _Run:
    """One Session's two watched Turns, and the forward the cases read the API through.

    A record rather than a tuple because the two Turns are told apart by which one they
    are and by nothing else -- both carry the same fields, over the same Session, under
    the same pod name -- and a case that unpacked them positionally could compare the
    first with itself and still pass.
    """

    base: str
    session_id: SessionId
    first: _Watched
    second: _Watched | None
    """None when the first Turn placed nothing, so the second was never attempted."""

    first_pod_at_second_submit: str | None
    """The first Turn's pod as the second Turn was submitted, or None if already gone.

    Not asserted on, and recorded because it says which of two placement paths this run
    actually took. A pod still listed here is stamped for deletion and reads as GONE, so
    the second Turn is placed by waiting that pod out; a pod already collected reads as
    ABSENT and is placed straight away. Both are correct and both end on a new pod, so
    no assertion turns on it -- but a run that never lands in the ten-second window
    graded the path its sibling file already grades, and a reader deserves to know which
    happened rather than infer it.
    """


@pytest.fixture(scope="module")
def placed() -> Iterator[_Run]:
    """One Session, two watched Turns back to back, and the forward cases read through.

    Module-scoped because these cases are four readings of ONE run rather than four runs
    of their own, and because each Turn is expensive in a way it did not use to be: the
    API holds its response until the Turn ends, so every submission pays a placement, an
    image pull and a model round trip.

    **Nothing waits between the two Turns, and that is the point of the second one.**
    The first Turn's pod is deleted as its response is written and then lingers for its
    grace period, so a Turn submitted immediately arrives while that pod is still
    listed. That is what an interactive tenant does -- ask, read the answer, ask again
    -- and it is the placement path a Turn takes when the pod it is replacing is still
    terminating. Waiting here would produce the other path, which
    `test_a_reaped_session_takes_another_turn.py` already grades deterministically and
    says in as many words that it grades instead of this one.

    The second Turn is attempted only if the first placed a pod. Without it there is
    nothing to compare against, and a run whose control plane places nothing would
    otherwise spend another eleven minutes finding that out twice.
    """
    with forwarded("deploy/control-plane", _CONTROL_PLANE_PORT) as base:
        session_id = _register_a_session(base)
        try:
            first = _watch_one_turn(base, session_id)
            still_there = None
            second = None
            if first.pod_uid is not None:
                # Read BEFORE the second submission and not after it: this is the only
                # moment at which "was the old pod still there when the next Turn
                # arrived" is answerable, and a read taken afterwards would be
                # describing whatever the second Turn had by then done about it.
                still_there = _pod_uid(pod_name_for(session_id))
                second = _watch_one_turn(base, session_id, ignoring=first.pod_uid)
            yield _Run(
                base=base,
                session_id=session_id,
                first=first,
                second=second,
                first_pod_at_second_submit=still_there,
            )
        finally:
            _clean_up(session_id)


def test_the_pod_exists_and_the_control_plane_is_what_created_it(
    placed: _Run,
) -> None:
    """The whole slice, in one object's existence, read while the Turn held it.

    Before this slice every deployment change could be in place and this would still
    fail, because nothing in `src/` called `Placement.place` or `compile_session_config`
    at all. The name is `pod_name_for(session_id)` computed from the id the API just
    returned, so nothing this run did not cause can satisfy it.

    The three Secrets are asserted with the pod rather than in a case of their own: a
    pod whose secret volumes have nothing behind them can never mount, kubelet retries
    for ever, and an existence check on the pod alone would call that a pass. Both
    halves come from one sighting, which is what keeps the pairing meaningful now that
    the pod is leased -- read at two moments, "the pod is here and its Secrets are not"
    is also what a correct handback looks like from the wrong side of it.
    """
    watched = placed.first
    pod = pod_name_for(placed.session_id)
    assert watched.pod_uid is not None, (
        f"no pod {pod} was ever in {NAMESPACE} while its Turn was in flight, and the "
        f"API then answered {watched.answered.status_code}. The control plane accepted "
        "the Session and the Turn and placed nothing, which is the state this slice "
        "exists to end."
    )
    missing = [s for s in _SECRET_SUFFIXES if s not in watched.secrets_present]
    assert not missing, (
        f"the pod exists and {missing} do not. Its secret volumes have nothing behind "
        "them, so kubelet retries the mount for ever and the pod can never start."
    )


def test_the_turn_is_admitted_and_recorded_before_anything_is_dispatched(
    placed: _Run,
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
    first = placed.first.answered
    assert first.status_code == 202, first.text
    types = _event_types(placed.base, placed.session_id)
    assert types[0] == "session.created", types
    assert "turn.submitted" in types, types


def test_the_tool_gateway_accepts_the_token_the_control_plane_signed(
    placed: _Run,
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

    The token is lifted during the Turn and presented after it, because the Secret
    holding it is deleted with the pod while the answer this case wants is not. Nothing
    about that weakens the question: the gateway's middleware decides on the signature
    and the expiry alone -- it reads no pod, no Session and no cluster -- and the
    deployed lifetime is a day, so a token minted minutes ago is exactly as acceptable
    to it as one minted while its pod was up. What would weaken it is presenting a token
    that had expired, and this file cannot run for long enough to do that.
    """
    session_id = placed.session_id
    signed = placed.first.session_token
    assert signed is not None, (
        "no Session token was read while the Turn was in flight, so there is nothing "
        "to present. The compiled Secret was "
        + (
            "in the namespace and carried no such header"
            if "compiled" in placed.first.secrets_present
            else "not in the namespace at all"
        )
    )
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


def test_a_second_turn_is_given_a_pod_of_its_own(placed: _Run) -> None:
    """A Turn arriving on the heels of the last one is placed a pod, not handed the old.

    This case asserted the opposite until ADR-041 -- one Session, one pod object,
    whatever a caller does -- and what it graded, `ensure_for` finding the pod it made a
    moment ago rather than making another, is the rule that record withdrew. The
    instrument survives the reversal exactly: both Turns' pods carry the SAME NAME
    because the name is a function of the Session id, so only the UID the API server
    assigns per object can tell a new pod from the old one still standing.

    **What makes this worth a case rather than a duplicate is WHEN the second Turn is
    submitted.** Nothing waits, so it arrives while the first pod is still working
    through its grace period, listed and stamped for deletion. That pod reads as GONE,
    and a Turn that finds GONE is placed by waiting it out and creating in its place --
    a different path from the one a Turn takes when the object has already been
    collected. `test_a_reaped_session_takes_another_turn.py` waits the window out
    deliberately, so that it can grade the second path without racing a ten-second
    window, and its own docstring records that the first path is the one an interactive
    tenant takes and is not graded there. This is where it is graded.

    Refusing a Turn in that window is not hypothetical: it is what the platform did
    until it was fixed, as a 502 reading "the pod for session ... is gone", and it
    failed every second Turn a tenant sent promptly. So the assertions here are the
    shape of that regression -- a Turn answered, a pod that is new, and Secrets behind
    its volumes.

    Which path a given run took is reported and not asserted. Landing inside a
    ten-second window is not something a test can promise, and both paths end on a new
    pod, so an assertion on the window would fail runs where the platform was merely
    quick.
    """
    first, second = placed.first, placed.second
    assert first.pod_uid is not None, (
        "the first Turn placed no pod, so the second was never submitted and there is "
        "nothing here to compare. What failed is above this case, not in it."
    )
    assert second is not None, "the second Turn was not attempted"
    arrived = (
        "while the first pod was still terminating"
        if placed.first_pod_at_second_submit == first.pod_uid
        else "after the first pod had already been collected"
    )
    assert second.answered.status_code == 202, (
        f"a second Turn submitted {arrived} was answered "
        f"{second.answered.status_code}: {second.answered.text}"
    )
    assert second.pod_uid is not None, (
        f"a second Turn on session {placed.session_id}, submitted {arrived}, was "
        "answered 202 and no pod of its own ever appeared. Every Turn pays its own "
        "placement, so a second Turn placing nothing is a Session usable exactly once."
    )
    missing = [s for s in _SECRET_SUFFIXES if s not in second.secrets_present]
    assert not missing, (
        f"the second Turn's pod exists and {missing} do not, so its secret volumes "
        "have nothing behind them and it can never start. A pod that replaces another "
        "takes the old one's Secrets with it, so this is the reading that says the "
        "replacement minted its own rather than inheriting a deletion."
    )
    assert second.pod_uid != first.pod_uid, (
        f"the second Turn on session {placed.session_id} ran on the pod the first one "
        f"placed: uid {first.pod_uid} twice. A pod is leased for one Turn, so a UID "
        "surviving into the next one is a pod that was never given back."
    )
