"""The whole tenant-facing surface, once, over one real app and one real database.

Tier 1 (testcontainers, real PostgreSQL 17). Every other test in this suite exercises
one slice: it builds the routes that slice owns over fakes for everything else, which is
what makes those tests fast and precise. The cost is that nothing checks the *seams
between* them, and the seams are where a wiring mistake lives. Three examples this file
covers and no per-slice test can:

- `tests/control/test_sessions.py` resolves definitions through `AlwaysResolves()`, so
  every Session in that file is created against an id that was never registered. Whether
  `POST /v1/sessions` can resolve an id that `POST /v1/agents` actually wrote — through
  `PostgresDefinitionRegistry`, over the real column types — is not asserted anywhere.
- The event a Session's creation appends is written by `sessions.py` and read by
  `events.py`, which agree about the payload keys only by both being correct. Nothing
  reads back what the create path wrote through the route that serves it.
- The four ports come out of `composition.build` bound to five adapters sharing one
  engine and one connection pool. Every route being reachable on one app over one pool
  is a property of the composition root, and `tests/test_composition.py` checks that
  wiring without driving a request through it.

So this walks one tenant's path end to end and then checks that a second tenant
standing beside it sees none of the first's work. It is deliberately a single long test
rather than several: the value is in the *carry* — the id minted at step 2 being the id
resolved at step 3 and the id listed at step 6 — and splitting it into independent cases
would need each one to rebuild the prior state from fakes, which is the thing being
avoided.

It asserts on response bodies rather than on the database, because a body is what a
tenant can observe and a row is not.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.composition import build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER

_SHA = "0" * 39 + "a"

_DEEPWIKI = {
    "transport": "streamable_http",
    "url": "https://mcp.deepwiki.com",
    "credential_ref": "vault/acme/deepwiki",
}


def _server_body() -> dict[str, object]:
    return {
        "server_name": "deepwiki",
        "endpoint": _DEEPWIKI,
        "tools": [
            {
                "name": "ask_question",
                "remote_name": "ask_question",
                "parameters": {"repoName": "string", "question": "string"},
                "scope_bindings": [{"dimension": "repository", "argument": "repoName"}],
            }
        ],
    }


def _definition_body() -> dict[str, object]:
    return {
        "name": "slr-reviewer",
        "instructions": "Extract findings and name the source document for each.",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SHA,
        "tool_servers": ["deepwiki"],
    }


_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
"""A digest-pinned image, because a registered shape refuses anything else."""


async def _an_environment(caller: httpx.AsyncClient) -> str:
    """A sandbox shape registered through the route, over the real adapter.

    Registered rather than faked because that is the seam this file exists for: the
    create path resolves this id through `PostgresEnvironmentStore`, over the real uuid
    and json columns, against a row another route wrote.
    """
    registered = await caller.post(
        "/v1/environments",
        json={
            "name": "slr-sandbox",
            "runtime_image": _FIXTURE_IMAGE,
            "denied_paths": ["/session/workspace/secrets"],
        },
    )
    assert registered.status_code == 201, registered.text
    return str(registered.json()["id"])


def _create_body(definition_id: str, environment_id: str) -> dict[str, object]:
    return {
        "definition_id": definition_id,
        "environment_id": environment_id,
        "grant": ["fs.read"],
        "scope": {"repository": "acme/widgets"},
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 30,
    }


@pytest.fixture
async def app(database_url: str) -> AsyncIterator[FastAPI]:
    """The real app over the real adapters, on the migrated container database.

    The engine is disposed here rather than left to garbage collection: it owns a pool
    of 50 connections against one `max_connections`, and a test that leaked one would
    fail a later test rather than itself.
    """
    platform, engine = build(database_url)
    try:
        yield create_app(platform)
    finally:
        await engine.dispose()


def _caller(app: FastAPI, tenant: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://platform",
        headers={TENANT_HEADER: tenant},
    )


async def test_a_tenant_registers_creates_reads_and_lists_while_another_sees_none(
    app: FastAPI,
) -> None:
    """One tenant's whole path, then a second tenant beside it seeing nothing of it."""
    mine, theirs = str(uuid4()), str(uuid4())

    async with _caller(app, mine) as caller:
        # 1. A tool server and its catalog.
        registered = await caller.post("/v1/mcp_servers", json=_server_body())
        assert registered.status_code == 201, registered.text

        # 2. A definition naming that server. The id comes back from the registry
        #    rather than being chosen by the caller, which is what makes step 3 a real
        #    resolve instead of a coincidence.
        defined = await caller.post("/v1/agents", json=_definition_body())
        assert defined.status_code == 201, defined.text
        definition_id = defined.json()["id"]
        assert defined.json()["revision"] == 1

        # 3. A Session against that definition. This is the assertion no per-slice test
        #    makes: the create path resolving an id another route wrote, through the
        #    Postgres adapter, over the real uuid column.
        environment_id = await _an_environment(caller)
        read_back = await caller.get(f"/v1/environments/{environment_id}")
        assert read_back.status_code == 200, read_back.text
        assert read_back.json()["denied_paths"] == ["/session/workspace/secrets"], (
            "the shape did not survive the round trip through the real json column"
        )

        created = await caller.post(
            "/v1/sessions", json=_create_body(definition_id, environment_id)
        )
        assert created.status_code == 201, created.text
        body = created.json()
        session_id, first_seq = body["id"], body["seq"]
        assert body["state"] == "running"
        assert first_seq == 1, (
            f"the creation event landed at seq {first_seq}. Sequence numbers are "
            "per-Session and this is the Session's first event, so anything but 1 "
            "means the sequence is shared across Sessions or the log was pre-seeded."
        )

        # 4. The Session reads back by its own id.
        fetched = await caller.get(f"/v1/sessions/{session_id}")
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["id"] == session_id

        # 5. The Event Log serves back what the create path appended. The payload keys
        #    are the contract between two modules that never import each other.
        events = await caller.get(f"/v1/sessions/{session_id}/events?from_seq=1")
        assert events.status_code == 200, events.text
        page = events.json()
        assert page["session_id"] == session_id
        assert [event["seq"] for event in page["events"]] == [1]
        creation = page["events"][0]
        assert creation["type"] == "session.created"
        assert creation["payload"]["definition_id"] == definition_id, (
            "the id the create route wrote into the event payload is not the id the "
            "definition was registered as. These are two modules agreeing by both "
            "being right; nothing else checks that they do."
        )
        assert creation["payload"]["definition_revision"] == 1

        # 6. The listing carries the same Session, under the four published keys.
        listed = await caller.get("/v1/sessions")
        assert listed.status_code == 200, listed.text
        rows = listed.json()["sessions"]
        assert [row["id"] for row in rows] == [session_id]
        assert set(rows[0]) == {
            "id",
            "definition_id",
            "definition_revision",
            "created_at_ms",
        }
        assert rows[0]["definition_id"] == definition_id

    # 7. A second tenant, on the same app and the same database, sees none of it. Each
    #    of these is a different code path to the same rule, which is why all four are
    #    here rather than one standing in for the rest.
    async with _caller(app, theirs) as other:
        empty = await other.get("/v1/sessions")
        assert empty.json()["sessions"] == []
        assert (await other.get(f"/v1/sessions/{session_id}")).status_code == 404
        assert (await other.get(f"/v1/sessions/{session_id}/events")).status_code == 404
        collides = await other.post("/v1/mcp_servers", json=_server_body())
        assert collides.status_code == 201, (
            "a second tenant could not register the server name the first tenant "
            f"holds. Got {collides.status_code}: {collides.text}. Server names are "
            "scoped per tenant, so this is a 409 only if the uniqueness constraint "
            "lost its tenant column."
        )


async def test_a_session_against_an_unregistered_definition_writes_nothing(
    app: FastAPI,
) -> None:
    """The refusal, and the half of it that matters: no Session exists afterwards.

    Separate from the journey above because it needs a tenant with no prior Sessions for
    the listing assertion to mean anything — "the list is empty" is only evidence that
    nothing was written if it was empty before.

    404 with a flat `code`, not 422 with a nested one: `core/errors.py` publishes
    `DEFINITION_NOT_FOUND` at 404, and a route emitting it at 422 was contradicting the
    published status map. The body is flat because that is what `ErrorEnvelope`
    serializes to; `{"detail": {...}}` is FastAPI's wrapper around an `HTTPException`,
    and this route no longer raises one.
    """
    tenant = str(uuid4())
    async with _caller(app, tenant) as caller:
        before = await caller.get("/v1/sessions")
        assert before.json()["sessions"] == []

        never_registered = str(uuid4())
        # A real environment, so the definition is what this create is refused for. The
        # create path resolves the sandbox shape first, and a body naming neither would
        # come back as `environment.not_found` -- a true answer to a different question.
        refused = await caller.post(
            "/v1/sessions",
            json=_create_body(never_registered, await _an_environment(caller)),
        )
        assert refused.status_code == 404, refused.text
        assert refused.json()["error"]["code"] == "definition.not_found"

        after = await caller.get("/v1/sessions")
        assert after.json()["sessions"] == [], (
            "a create call that was refused with 404 left a Session behind. A refusal "
            "that stores the row anyway is worse than no check at all, because the "
            "tenant believes it was rejected."
        )


async def test_fifty_concurrent_creates_all_succeed_over_one_pool(
    app: FastAPI,
) -> None:
    """The capacity the pool was sized for, driven concurrently through the real routes.

    `composition.py` sets `pool_size=50` against a measured cliff: at 50 concurrent
    appends the median was 20.5 ms, and at `pool_size=40, max_overflow=10` — the same
    ceiling of 50, reached with overflow instead of pool — it was 132 ms, because a
    connection handed out above `pool_size` is created for one checkout and closed on
    return. That measurement was taken against the adapter directly. Nothing drove the
    same concurrency through the app, so nothing checked that fifty requests in flight
    over one engine all get served rather than one of them finding the pool empty and
    timing out.

    Both assertions below would still hold if the requests had run one after another, so
    this test cannot prove they overlapped — what it can do is fail if concurrency
    breaks them, which is the direction that matters.

    They are not equally load-bearing, and it is worth writing down which is which.
    `seq == 1` is real: replacing the append's `WHERE session_id = :sid` with
    `WHERE (session_id = :sid OR TRUE)` — a global sequence instead of a per-Session one
    — fails this test with "50 Sessions got a first sequence other than 1 (e.g. 2)". The
    distinct-ids assertion is defensive rather than load-bearing: making
    `new_session_id` return a constant does fail the test, but through `session_pkey`'s
    `UniqueViolationError` surfacing as an unhandled `IntegrityError` long before the
    assertion is reached. The primary key is the real guard against a duplicate id; this
    assertion can only fire if duplicates were somehow *persisted*, which that
    constraint prevents. It is kept because a future schema change that dropped the
    constraint would make it the guard, and because an unhandled IntegrityError is a
    worse diagnostic than a sentence naming the mint.
    """
    tenant = str(uuid4())
    async with _caller(app, tenant) as caller:
        defined = await caller.post("/v1/agents", json=_definition_body())
        assert defined.status_code == 201, defined.text
        definition_id = defined.json()["id"]

        body = _create_body(definition_id, await _an_environment(caller))
        responses = await asyncio.gather(
            *(caller.post("/v1/sessions", json=body) for _ in range(50))
        )

    failures = [r for r in responses if r.status_code != 201]
    assert not failures, (
        f"{len(failures)} of 50 concurrent creates failed. First: "
        f"{failures[0].status_code} {failures[0].text}. `pool_size` is 50 and is meant "
        "to cover this exactly, so a timeout here means the pool is smaller than the "
        "concurrency it was sized for, or a connection is being held across requests."
    )

    bodies = [r.json() for r in responses]
    ids = {b["id"] for b in bodies}
    assert len(ids) == 50, (
        f"50 creates produced {len(ids)} distinct Session ids. Ids are minted per "
        "call, so a collision means the mint is not per-call."
    )
    off_by_sequence = [b for b in bodies if b["seq"] != 1]
    assert not off_by_sequence, (
        f"{len(off_by_sequence)} Sessions got a first sequence other than 1 "
        f"(e.g. {off_by_sequence[0]['seq']}). Sequences are per-Session, so a number "
        "above 1 on a Session's first event means the sequence is shared across "
        "Sessions -- which under concurrency is a correctness bug, not a slow path."
    )
