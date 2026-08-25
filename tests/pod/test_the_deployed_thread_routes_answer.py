"""The five thread routes, against the control plane running in the cluster.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

**The offline route tests cannot reach what this reaches.** Every one of them puts a
hand-written thread index behind the port, so the facts under assertion are facts a fake
was told. Here the threads come out of the real `event_log` table, their timestamps are
the rows' own `appended_at`, and the identifiers were minted by the shim in a pod during
a real Turn. Three things can only be wrong here: the SQL, the migration's index, and
whether the composition root wired the adapter at all -- and that last one fails
silently, because a `Platform` without the index answers an empty listing, which is also
the honest answer for a Session that never delegated.

That is why the first case asserts a thread is **present**. An empty page would pass
every other assertion in this file by making them vacuous.

**A single-agent Turn is enough.** The runtime announces the root thread before the Turn
starts, measured 2026-08-24, so one ordinary Turn produces exactly one thread with a
null parent -- which is all five routes' subject. Delegation is not exercised here,
and cannot be: `spawn_agent` refuses every model this platform routes.

Two Sessions, not one. The archive is a write, and a retirement in the middle of a
module-scoped Session would make every later case depend on the order it ran in.

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
from managed_agent.core.vocabulary import thread, turn

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_TURN_DEADLINE_S: Final = 600
_SUBMIT_TIMEOUT_S: Final = 900
_STREAM_READ_S: Final = 30
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")
_PROMPT: Final = "Reply with exactly the word ok and nothing else."
_DELEGATING_PROMPT: Final = (
    "Delegate this to a subagent: count the vowels in the word REFRIGERATOR and "
    "report the number. Use a subagent rather than doing it yourself, then tell me "
    "the number it reported."
)

requires_the_cluster = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places a real pod and calls a real model",
)


def _session_image() -> str:
    """The newest digest in the Session repository, resolved rather than pinned."""
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


@dataclass(frozen=True, slots=True)
class Ran:
    """One Session that has closed a Turn, and how to keep talking to it."""

    base: str
    tenant_id: str
    session_id: SessionId
    events: tuple[dict[str, Any], ...]


def _events(base: str, tenant_id: str, session_id: SessionId) -> list[dict[str, Any]]:
    with _client(base, tenant_id) as caller:
        answered = caller.get(f"/v1/sessions/{session_id}/events")
    assert answered.status_code == 200, answered.text
    listed: list[dict[str, Any]] = answered.json()["events"]
    return listed


def _await_terminal(
    base: str, tenant_id: str, session_id: SessionId
) -> list[dict[str, Any]]:
    """Poll until the Turn closes either way, then return the whole log.

    Both terminal types. A poll waiting only for success sits out its whole deadline on
    a Turn that failed in the first second and then reports a timeout, which sends the
    reader after the wrong thing.
    """
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        events = _events(base, tenant_id, session_id)
        if any(
            one["type"] in (turn.TURN_COMPLETED, turn.TURN_FAILED) for one in events
        ):
            return events
        time.sleep(3)
    events = _events(base, tenant_id, session_id)
    pytest.fail(
        f"session {session_id} closed no turn in {_TURN_DEADLINE_S}s; the log was "
        f"{[one['type'] for one in events]}"
    )


def _clean_up(session_id: SessionId) -> None:
    """Delete the pod and its Secrets. Three aborted runs once left forty-two pods
    squatting the namespace, after which the next run's scheduling refusal read as the
    cluster being out of capacity."""
    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


def _run_one_turn(
    base: str, image: str, *, prompt: str = _PROMPT, multiagent: bool = False
) -> Iterator[Ran]:
    """A fresh tenant, a Session under a one-off definition, and one closed Turn.

    A tenant of its own per call, so the two fixtures below cannot see each other's
    Sessions and a listing assertion cannot be satisfied by the wrong one.
    """
    tenant_id = str(uuid4())
    stamp = uuid4().hex[:8]
    with _client(base, tenant_id) as caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={"name": f"threads-{stamp}", "runtime_image": image},
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"threads-{stamp}",
                    "instructions": "Answer briefly and exactly as asked.",
                    "model": _MODEL,
                    "skills_repository": "git@github.com:acme/skills.git",
                    "skills_revision": "0" * 39 + "a",
                    "skills": [],
                    "tool_servers": [],
                    **(
                        {"multiagent": {"enabled": True, "max_depth": 2}}
                        if multiagent
                        else {}
                    ),
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
    session_id = SessionId(UUID(session["id"]))
    try:
        with _client(base, tenant_id, timeout=_SUBMIT_TIMEOUT_S) as caller:
            submitted = caller.post(
                f"/v1/sessions/{session_id}/events",
                json={"prompt": prompt},
                headers={"Idempotency-Key": uuid4().hex},
            )
        assert submitted.status_code == 202, submitted.text
        closed = _await_terminal(base, tenant_id, session_id)
        yield Ran(base, tenant_id, session_id, tuple(closed))
    finally:
        _clean_up(session_id)


@pytest.fixture(scope="module")
def forwarded_base() -> Iterator[str]:
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        yield base


@pytest.fixture(scope="module")
def image() -> str:
    return _session_image()


@pytest.fixture(scope="module")
def read_only(forwarded_base: str, image: str) -> Iterator[Ran]:
    """The Session every read case shares. Nothing here retires anything on it."""
    yield from _run_one_turn(forwarded_base, image)


@pytest.fixture(scope="module")
def retirable(forwarded_base: str, image: str) -> Iterator[Ran]:
    """A second Session, whose thread the archive cases are free to retire."""
    yield from _run_one_turn(forwarded_base, image)


@pytest.fixture(scope="module")
def delegating(forwarded_base: str, image: str) -> Iterator[Ran]:
    """A third Session, whose Turn runs on more than one thread.

    Worth a whole extra Turn and pod. The four read routes are indistinguishable on a
    single-agent Session between a listing folded from the log and one folded from the
    runtime's announcements -- there is one thread and it is announced. Only a Turn that
    opens several threads separates them, and the first version of this surface got it
    wrong in exactly that gap: it listed one thread of six.
    """
    yield from _run_one_turn(
        forwarded_base, image, prompt=_DELEGATING_PROMPT, multiagent=True
    )


def _threads(ran: Ran, **params: int | str) -> dict[str, Any]:
    with _client(ran.base, ran.tenant_id) as caller:
        answered = caller.get(f"/v1/sessions/{ran.session_id}/threads", params=params)
    assert answered.status_code == 200, answered.text
    body: dict[str, Any] = answered.json()
    return body


def _root_id(ran: Ran) -> str:
    listed = _threads(ran)["data"]
    assert listed, (
        "the deployed listing named no thread for a Session that ran a Turn. Either "
        "the shim published no thread.started, or the composition root left "
        "Platform.session_threads as the refusing stand-in -- which answers an empty "
        "page and looks like success"
    )
    return str(listed[0]["id"])


@requires_the_cluster
def test_the_turn_closed_so_everything_below_is_about_a_session_that_ran(
    read_only: Ran,
) -> None:
    """First, because a failed Turn makes every case below vacuous."""
    closed = [
        one["type"]
        for one in read_only.events
        if one["type"] in (turn.TURN_COMPLETED, turn.TURN_FAILED)
    ]
    assert closed == [turn.TURN_COMPLETED], [one["type"] for one in read_only.events]


@requires_the_cluster
def test_the_deployed_listing_names_the_root_thread(read_only: Ran) -> None:
    """One thread, no parent, and the identifier the log actually carries.

    Compared against the `thread.started` event rather than merely checked for shape:
    the listing folds the log, so an id here that the log does not hold would mean the
    fold invented one -- and a listing of plausible identifiers is worse than none at
    all, because every later call would 404 for a reason nothing explains.
    """
    body = _threads(read_only)
    assert len(body["data"]) == 1, body
    only = body["data"][0]
    announced = next(
        one for one in read_only.events if one["type"] == thread.THREAD_STARTED
    )
    assert only["id"] == announced["payload"]["thread_id"]
    assert only["parent_thread_id"] is None
    assert only["type"] == "session_thread"
    assert body["next_page"] is None


@requires_the_cluster
def test_the_deployed_listing_accepts_a_limit(read_only: Ran) -> None:
    """The bound is a published query parameter, so a 422 on it would be a defect.

    A one-thread Session cannot page, so this grades the parameter and the look-ahead
    rather than the walk: asking for one of one must still answer no next page, because
    the extra row the route asks for and discards is not there.
    """
    body = _threads(read_only, limit=1)
    assert len(body["data"]) == 1
    assert body["next_page"] is None


@requires_the_cluster
def test_the_deployed_thread_carries_real_timestamps(read_only: Ran) -> None:
    """The two fields the offline suite cannot produce, because they come from SQL.

    `EventRecord` carries no `appended_at`, which is the whole reason the thread index
    is a query of its own rather than a fold over the log. Only a real row can show that
    the query reads the column it claims to.
    """
    only = _threads(read_only)["data"][0]
    assert only["created_at_ms"] > 1_700_000_000_000, only
    assert only["updated_at_ms"] >= only["created_at_ms"], only
    assert only["archived_at_ms"] is None, only


@requires_the_cluster
def test_the_deployed_thread_is_read_by_name_and_agrees_with_the_listing(
    read_only: Ran,
) -> None:
    with _client(read_only.base, read_only.tenant_id) as caller:
        answered = caller.get(
            f"/v1/sessions/{read_only.session_id}/threads/{_root_id(read_only)}"
        )
    assert answered.status_code == 200, answered.text
    assert answered.json() == _threads(read_only)["data"][0]


@requires_the_cluster
def test_the_deployed_thread_reads_idle_once_its_turn_has_closed(
    read_only: Ran,
) -> None:
    """Not running, and that is the fact the SQL has to get right.

    `turn_ended` is answered by matching the thread's own `turn_id` against a terminal
    turn event in the same log. A query that missed the match reports every finished
    thread as running for ever, and no offline case can tell, because offline the
    boolean is whatever the fake was handed.
    """
    assert _threads(read_only)["data"][0]["status"] == "idle"


@requires_the_cluster
def test_a_thread_id_this_session_never_held_is_not_found(read_only: Ran) -> None:
    absent = str(uuid4())
    with _client(read_only.base, read_only.tenant_id) as caller:
        answered = caller.get(f"/v1/sessions/{read_only.session_id}/threads/{absent}")
    assert answered.status_code == 404, answered.text
    assert answered.json()["error"]["code"] == "thread.not_found"


@requires_the_cluster
def test_the_deployed_thread_events_are_the_ones_that_name_it(read_only: Ran) -> None:
    """Non-empty, all attributed to this thread, and a subset of the Session's log.

    Non-empty first. A predicate that matched nothing would satisfy "all of them carry
    this id" trivially, which is the shape of a filter that silently drops everything.
    """
    root = _root_id(read_only)
    with _client(read_only.base, read_only.tenant_id) as caller:
        answered = caller.get(
            f"/v1/sessions/{read_only.session_id}/threads/{root}/events"
        )
    assert answered.status_code == 200, answered.text
    narrowed = answered.json()["events"]
    assert narrowed, "the thread that produced the whole Turn has no events"
    assert all(one["payload"].get("thread_id") == root for one in narrowed), narrowed
    seqs = {one["seq"] for one in narrowed}
    assert seqs < {one["seq"] for one in read_only.events}, (
        "every event of the Session was attributed to this thread, so the predicate "
        "narrowed nothing -- the Session's own creation and placement carry no thread"
    )


@requires_the_cluster
def test_the_deployed_thread_stream_replays_that_threads_events(
    read_only: Ran,
) -> None:
    """The stream opens and delivers, with the Session's sequence as the SSE id.

    The Turn is already closed, so what arrives is the backlog above position zero
    rather than a live tail -- which is the half of the route that can be asserted
    without racing a model. The ids are the Session's, so they skip: the numbers not
    seen belong to events that name no thread.
    """
    root = _root_id(read_only)
    frames: list[str] = []
    followed = f"/v1/sessions/{read_only.session_id}/threads/{root}/stream"
    with (
        _client(read_only.base, read_only.tenant_id, timeout=_STREAM_READ_S) as caller,
        caller.stream("GET", followed) as answered,
    ):
        assert answered.status_code == 200
        for line in answered.iter_lines():
            if line.startswith("id: "):
                frames.append(line)
            if len(frames) >= 2:
                break
    assert frames, "the thread stream opened and sent no event"
    numbers = [int(line.removeprefix("id: ")) for line in frames]
    assert numbers == sorted(numbers), numbers


@requires_the_cluster
def test_retiring_a_deployed_thread_is_recorded_and_is_idempotent(
    retirable: Ran,
) -> None:
    """The one write on this surface, end to end: refuse nothing, record once, re-read.

    The second call is the case with teeth. The route reads the thread, sees an archive
    already there and answers without appending -- so a second timestamp equal to the
    first is the only evidence that the log holds one retirement rather than two, and a
    log with two could not say which one counted.
    """
    root = _root_id(retirable)
    with _client(retirable.base, retirable.tenant_id) as caller:
        first = caller.post(
            f"/v1/sessions/{retirable.session_id}/threads/{root}/archive"
        )
        second = caller.post(
            f"/v1/sessions/{retirable.session_id}/threads/{root}/archive"
        )
        read_back = caller.get(f"/v1/sessions/{retirable.session_id}/threads/{root}")
    assert first.status_code == 200, first.text
    assert first.json()["archived_at_ms"] is not None, first.text
    assert first.json()["status"] == "terminated"
    assert second.json()["archived_at_ms"] == first.json()["archived_at_ms"]
    assert read_back.json() == second.json()


@requires_the_cluster
def test_the_retirement_is_an_event_on_the_sessions_own_log(retirable: Ran) -> None:
    """One `thread.archived`, naming the thread, appended by the control plane.

    Read back through the Session's event surface rather than trusted from the archive
    route's own answer: the route computes its reply from a re-read, so a reply carrying
    a timestamp proves the row exists but not that the row is on this Session's log
    under the published type -- which is what a consumer following the stream sees.
    """
    root = _root_id(retirable)
    archived = [
        one
        for one in _events(retirable.base, retirable.tenant_id, retirable.session_id)
        if one["type"] == thread.THREAD_ARCHIVED
    ]
    assert len(archived) == 1, archived
    assert archived[0]["payload"] == {"thread_id": root}


@requires_the_cluster
def test_the_deployed_listing_names_every_thread_that_produced_an_event(
    delegating: Ran,
) -> None:
    """The case the single-agent Session cannot make: more threads than announcements.

    Measured 2026-08-24 against codex-cli 0.149.0 -- one delegating Turn produced six
    thread identifiers and exactly one `thread.started`, because the runtime announces
    the thread it opens for the Turn and says nothing when it opens one for a spawned
    agent. A listing derived from the announcements returned one row and 404'd for the
    other five while their events sat in the log.

    Asserted as set equality against the log rather than as a count, so a listing that
    happened to return the right number of the wrong threads fails. And the announced
    thread must be among them with its parent published as null: it is the root, and it
    is the one thread a consumer has to be able to pick out.
    """
    carried = {
        str(one["payload"]["thread_id"])
        for one in delegating.events
        if isinstance(one["payload"].get("thread_id"), str)
    }
    announced = [
        one for one in delegating.events if one["type"] == thread.THREAD_STARTED
    ]
    assert len(carried) > len(announced), (
        "this Turn ran no more threads than were announced, so it cannot tell the two "
        f"derivations apart; threads {sorted(carried)}, announcements {len(announced)}"
    )
    listed = _threads(delegating)["data"]
    assert {str(one["id"]) for one in listed} == carried
    roots = [one for one in listed if one.get("parent_thread_id", "absent") is None]
    assert len(roots) == 1, roots
    assert roots[0]["id"] == announced[0]["payload"]["thread_id"]
