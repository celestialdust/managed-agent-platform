"""A multiagent Turn, and whether the tenant can tell which agent said what.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN -- no pod was placed, no
model was called, and nothing here is evidence of anything on that run.

**This is the first multiagent Turn this project has ever run.** `multiagent.enabled` is
per agent definition and defaults false, so every live run before this one was a single
agent, and no real `thread/started` frame had ever reached `shim/turn_runner.py`. Every
offline case for thread attribution scripts the shape the runtime's documentation gives
-- `params.thread.id`, nested -- and a flat fallback exists because that documentation
is the only witness to either arrangement. This is the case that can contradict it.

**What the first run measured, on 2026-08-24 against codex-cli 0.149.0.** Two documented
assumptions turned out false and one thing worked exactly as designed.

The runtime announces the ROOT thread, before the Turn starts, with no parent. The
vocabulary module said the opposite -- that only spawned threads are announced, so an
absent `thread.started` identifies the root -- and that is corrected there.

`spawn_agent` refuses a model that is not in the runtime binary's own catalogue:
*"Unknown model `gsds-claude-opus-4-6` for spawn_agent. Available models: gpt-5.6-sol,
gpt-5.6-terra, gpt-5.6-luna, gpt-5.5, gpt-5.2"*. Every model this platform routes is a
custom name the Model Gateway resolves, and the five it offers are OpenAI models this
account has no upstream for. So no delegation could complete here, and the cases about
the answer were strict xfails naming that reason -- written to fail, loudly, on the day
it was fixed. **All three have now fired**: the session image bakes a catalogue naming
every routed model, and the last of them passed on 2026-08-24.

And the thing that worked: the subagent's `turn/started` arrived on a DIFFERENT thread
and the platform gave it a different identifier. Before this slice that event was
indistinguishable from the root agent's, which is exactly why the failure above was
legible -- the log showed a second thread starting and then the Turn failing, rather
than one agent inexplicably stopping. That is the case ADR-007 made for attribution, met
on the first run.

NO KEY OR TOKEN VALUE IS PRINTED OR ASSERTED ON.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
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

_TURN_DEADLINE_S: Final = 900
_SUBMIT_TIMEOUT_S: Final = 1200
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")
_MAX_DEPTH: Final = 2

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


_PROMPT: Final = (
    "Delegate this to a subagent: count the vowels in the word REFRIGERATOR and "
    "report the number. Use a subagent rather than doing it yourself, then tell me "
    "the number it reported."
)


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

    Both terminal types, because a multiagent Turn is the most likely one so far to
    fail: it spends more, runs longer and exercises a runtime path nothing here has
    taken. A poll waiting only for success would sit out its whole deadline on a Turn
    that failed in the first second and then report a timeout, sending the reader after
    the wrong thing.

    A longer deadline than the single-agent files use, since the root agent's Turn does
    not close until its subagent's work is done.
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
    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


@pytest.fixture(scope="module")
def events() -> Iterator[list[dict[str, Any]]]:
    """One Session under a definition with delegation enabled, and one Turn.

    `multiagent` is the field that makes this file different from every other live one,
    and `max_depth` is 2 -- the smallest value above the default that permits a delegate
    at all. Deeper would multiply pods and spend for nothing: one level is enough to
    produce a parent and a child, which is the whole subject.

    A larger budget than the other live files, because a delegating Turn pays for two
    agents' tokens and a Turn refused for budget would look exactly like a Turn that
    declined to delegate.

    The teardown deletes the pod and its Secrets in a `finally`. Three aborted runs once
    left forty-two pods squatting the namespace, after which the next run's scheduling
    refusal read as the cluster being out of capacity.
    """
    tenant_id = str(uuid4())
    stamp = uuid4().hex[:8]
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        with _client(base, tenant_id) as caller:
            environment = _created(
                caller.post(
                    "/v1/environments",
                    json={"name": f"multi-{stamp}", "runtime_image": image},
                )
            )
            definition = _created(
                caller.post(
                    "/v1/agents",
                    json={
                        "name": f"multi-{stamp}",
                        "instructions": (
                            "You coordinate work. When asked to delegate, spawn a "
                            "subagent and report what it tells you."
                        ),
                        "model": _MODEL,
                        "skills_repository": "git@github.com:acme/skills.git",
                        "skills_revision": "0" * 39 + "a",
                        "skills": [],
                        "tool_servers": [],
                        "multiagent": {"enabled": True, "max_depth": _MAX_DEPTH},
                    },
                )
            )
            session = _created(
                caller.post(
                    "/v1/sessions",
                    json={
                        "definition_id": definition["id"],
                        "environment_id": environment["id"],
                        "budget_minor_units": 2_000_000,
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
                    json={"prompt": _PROMPT},
                    headers={"Idempotency-Key": uuid4().hex},
                )
            assert submitted.status_code == 202, submitted.text
            yield _await_terminal(base, tenant_id, session_id)
        finally:
            _clean_up(session_id)


@requires_the_cluster
def test_the_multiagent_turn_reached_a_terminal_event(
    events: list[dict[str, Any]],
) -> None:
    """The Turn completed rather than failed. First, because everything else follows.

    A failed Turn makes every case below a statement about a Turn that did not happen,
    so this one names the outcome and prints the whole type sequence when it is wrong.

    **This was a strict xfail and it passed on 2026-08-24, which is why it is now an
    assertion.** It did not start passing because delegation started working -- it does
    not, and the case at the bottom of this file records that it does not. It started
    passing because `turn_runner.py` stopped ending a Turn on the first `turn/completed`
    to arrive on any thread. Before that, a spawned thread's refusal reported itself as
    the root Turn's failure: the child's completion carried `status: failed`, the loop
    read it as this Turn's, and the platform published `turn.failed` while the root
    agent was still working -- which it then went on to finish successfully. The failure
    this file measured yesterday was the platform's, not the runtime's.
    """
    closed = [
        one["type"]
        for one in events
        if one["type"] in (turn.TURN_COMPLETED, turn.TURN_FAILED)
    ]
    assert closed == [turn.TURN_COMPLETED], [one["type"] for one in events]


@requires_the_cluster
def test_the_root_thread_is_announced_before_the_turn_starts(
    events: list[dict[str, Any]],
) -> None:
    """A `thread.started` arrives, it precedes `turn.started`, and its parent is null.

    This is the corrected assumption, pinned. The vocabulary module claimed the runtime
    announces a thread only when it spawns one, so a consumer identifying the root by
    the ABSENCE of a `thread.started` would have found no root at all. Ordering is
    asserted rather than mere presence, because "announced before the Turn" is the part
    that makes this the root rather than a child that happened to have no parent
    recorded.
    """
    types = [one["type"] for one in events]
    assert thread.THREAD_STARTED in types, types
    assert types.index(thread.THREAD_STARTED) < types.index(turn.TURN_STARTED), types
    first = next(one for one in events if one["type"] == thread.THREAD_STARTED)
    assert first["payload"]["parent_thread_id"] is None, first


@requires_the_cluster
def test_a_delegating_turn_produces_more_than_one_thread(
    events: list[dict[str, Any]],
) -> None:
    """Two distinct thread identifiers on one Turn. This is the whole slice, live.

    Before this slice the subagent's `turn/started` carried no attribution and was
    indistinguishable from the root agent's, so a tenant saw one voice and a reader of
    the log saw one agent stop for no reason. Two ids is the smallest observable
    difference that says otherwise, and it holds even though the delegation then failed
    -- the runtime opened the child's thread before refusing the model, so the events
    that prove attribution works arrive whether or not the work does.

    Counted over every event carrying an id rather than over `thread.started` alone,
    because on this run the second thread was never announced: `spawn_agent` failed
    after the child's `turn/started` and before any `thread/started` for it. Grouping by
    attribution is what finds it, which is the mechanism ADR-007 asked for.
    """
    identities = {
        str(one["payload"]["thread_id"])
        for one in events
        if one["payload"].get("thread_id")
    }
    assert len(identities) >= 2, [
        (one["type"], one["payload"].get("thread_id")) for one in events
    ]


@requires_the_cluster
def test_every_event_the_tenant_reads_says_which_thread_produced_it(
    events: list[dict[str, Any]],
) -> None:
    """Every event that came from the runtime says which thread produced it.

    This is the shipped defect, live: subagent text arrived under the same method the
    root agent uses, so a tenant saw one undifferentiated voice with no boundary in it.
    Only runtime-sourced types are examined -- a control-plane event like
    `turn.submitted` has no thread and never did.

    The `assert from_the_runtime` line is the vacuity control. A Session whose Turn
    produced no runtime events at all would otherwise pass an emptiness check about
    attribution.
    """
    from_the_runtime = [
        one
        for one in events
        if one["type"]
        in (turn.TURN_STARTED, turn.TURN_MESSAGE_DELTA, thread.THREAD_STARTED)
    ]
    assert from_the_runtime, [one["type"] for one in events]
    unattributed = [
        one for one in from_the_runtime if not one["payload"].get("thread_id")
    ]
    assert unattributed == [], unattributed


@requires_the_cluster
def test_every_thread_identifier_is_one_the_platform_issued(
    events: list[dict[str, Any]],
) -> None:
    """Every thread identifier a tenant can read is a uuid5 this platform minted.

    ADR-007 (MAP-A10), live: no Agent Runtime thread identifier reaches the
    caller, and every identifier the caller sees is one the platform issued. The
    runtime's own values are opaque strings of its choosing and would not parse as a
    uuid5, so this is the assertion that distinguishes a translated id from a copied one
    without needing to know what the runtime sent.

    Version 5 and not merely well-formed: a v4 here would mean a fresh value per event,
    which destroys grouping while still looking like an identifier the platform issued.
    """
    seen = [
        str(one["payload"][key])
        for one in events
        for key in ("thread_id", "parent_thread_id")
        if one["payload"].get(key)
    ]
    assert seen, [one["type"] for one in events]
    for value in seen:
        assert UUID(value).version == 5, value


@requires_the_cluster
def test_the_answer_came_back_and_names_the_count(
    events: list[dict[str, Any]],
) -> None:
    """The work got done and the answer is in the stream. Not by a subagent.

    REFRIGERATOR has five vowels. Asserted because attribution can be perfect on a
    stream that says nothing useful, and nothing above would catch that.

    **A strict xfail until 2026-08-24, when it passed.** What the log showed: the root
    agent tried to spawn five times, was refused each time with the runtime's own
    catalogue message, told the tenant so in as many words, and then counted the vowels
    itself. So this passes without delegation, and the case below is the one that holds
    the line on whether delegation happened.

    Either spelling accepted, since which one a model uses is not this platform's
    concern.
    """
    said = "".join(
        str(one["payload"].get("text", ""))
        for one in events
        if one["type"] == turn.TURN_MESSAGE_DELTA
    )
    assert said.strip(), [one["type"] for one in events]
    assert "5" in said or "five" in said.lower(), said


@requires_the_cluster
def test_a_spawned_agent_produced_output_of_its_own(
    events: list[dict[str, Any]],
) -> None:
    """The claim the two retired xfails used to carry: a subagent actually spoke.

    Structural rather than textual, so it cannot flake on how a model phrases a refusal.
    A thread that starts and says nothing is a spawn the runtime refused before
    sampling, and for as long as that was every thread but the root this was a strict
    xfail naming the reason.

    **It passed on 2026-08-24, which is why the marker is gone.** What changed is that
    the session image now bakes a model catalogue at `/opt/codex/models.json` carrying
    an entry per routed model, so `spawn_agent` resolves the name a Session is bound to
    instead of refusing it. The entry is cloned from the one runtime model that reaches
    the multi-agent backend, minus the donor's identity and minus the request wire the
    Model Gateway does not implement -- inheriting that wire is what made the model
    reach the provider with no tools bound and answer by describing calls it never made.
    """
    root = next(
        one["payload"]["thread_id"]
        for one in events
        if one["type"] == thread.THREAD_STARTED
        and one["payload"]["parent_thread_id"] is None
    )
    spoke = {
        one["payload"].get("thread_id")
        for one in events
        if one["type"] == turn.TURN_MESSAGE_DELTA
    }
    assert spoke - {root}, (
        "every delta on this Turn came from the root thread, so no spawned agent ever "
        f"produced output; threads that started: "
        f"{sorted({one['payload'].get('thread_id') for one in events})}"
    )


@requires_the_cluster
def test_the_runtime_announces_only_the_root_thread(
    events: list[dict[str, Any]],
) -> None:
    """One `thread.started` for a Turn that ran several threads. This is a live finding.

    Measured 2026-08-24 against codex-cli 0.149.0: six distinct thread identifiers
    carried events and exactly one `thread.started` was published -- the root's. So the
    runtime announces the thread it opens for the Turn and says nothing when it opens
    one for a spawned agent, so the only witness to a child thread is the
    `turn/started` that arrives on it.

    Pinned here because a whole surface depends on it. `GET /v1/sessions/{id}/threads`
    derives its listing from the events that carry a thread id, and the first version of
    it derived the listing from the announcements instead -- which listed one thread of
    six and 404'd for the other five while their events sat in the log. If the runtime
    starts announcing spawned threads, this fails, and the derivation can be simplified
    back with the parent pointer that would then be available.
    """
    announced = [one for one in events if one["type"] == thread.THREAD_STARTED]
    carried = {
        one["payload"]["thread_id"]
        for one in events
        if isinstance(one["payload"].get("thread_id"), str)
    }
    assert len(announced) == 1, [one["payload"] for one in announced]
    assert len(carried) > len(announced), sorted(carried)
    assert announced[0]["payload"]["thread_id"] in carried
