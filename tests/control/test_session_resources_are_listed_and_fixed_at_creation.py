"""A Session's files can be read back, and no single resource can be replaced.

Tier 1 (testcontainers, real PostgreSQL 17). The tier is load-bearing for two of the
properties here rather than incidental. The list route reads the `session.created`
payload, and the payload under test is the one the **create route actually writes** --
hand-building it would make the two halves agree with each other and prove nothing
about either. And the "creation event no longer retained" case is arranged the way a
retention sweep produces it, with a DELETE against the real table, which a fake log
keyed on a dict cannot represent.

Only the uploaded-file storage stands in, because the alternative is an S3 bucket. The
`FileStore` above it is the real one, so the digest and the length a list entry carries
are the ones the upload path computed rather than values a fake invented.

**What this file no longer claims.** Its title once ended "cannot be changed after it is
created", and the refusal it grades was read as covering the whole idea. It does not:
`POST .../resources/{resource_id}` is upstream's *update a resource* and is what refuses
here, and the collection route beside it now accepts an attach --
`test_a_file_is_attached_to_a_running_session.py` grades that, and `gap.md` D1 records
why the arguments against it were wrong. The tests below are unchanged and still
correct, because replacing one resource and adding one are different verbs; only the
title's reach was too wide.

Driven with `AsyncClient` over `ASGITransport` rather than `TestClient`, for the reason
`test_sessions_tenant_scope.py` gives: an async engine's pooled connections belong to
the loop that opened them, and `TestClient` runs the app in a loop of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.files import (
    REASON_FILE_NOT_FOUND,
    REASON_STORAGE_UNCONFIGURED,
)
from managed_agent.control.api.routes.resources import (
    REASON_CREATION_RECORD_NOT_RETAINED,
    REASON_CREATION_RECORD_UNREADABLE,
    REASON_RESOURCES_FIXED_AT_CREATION,
    AttachedResourceView,
)
from managed_agent.control.files.store import (
    FileId,
    FileStore,
    FileWindow,
    UploadedFile,
    UploadedFileStorage,
    UploadSizeLimit,
    unconfigured_file_store,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vocabulary import lifecycle, turn

_SKILLS_SHA = "0" * 39 + "b"
_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "b" * 64

_MARKER = b"resource-body-marker"
"""A body distinctive enough that finding it in a response is proof, not coincidence.

The list route must serve a file's *facts* and never its bytes, and the only way to
assert that over a JSON document is to look for the bytes. A body of `b"x" * 10` would
appear inside a digest by accident sooner or later.
"""


class FakeStorage:
    """One dict of objects and one of rows, keyed the way the real store keys them.

    Copied from `test_file_upload_download.py` rather than shared, which is the same
    trade that file records for the Platform roster: there is no shared fixtures module
    under `tests/control/`, and a helper package introduced for one class would be a new
    import surface for every test file in the directory.

    Three of the four operations below raise rather than answer. They are on the port
    because the file family grew a listing and a delete, and nothing in this file
    performs either -- a fake that answered them with an empty page or a silent success
    would let a later change reach this file's subject through a path it never meant to
    exercise. The fifth, `deletion_recorded`, is answered, because the download route
    this file drives asks it on every request.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.rows: dict[FileId, UploadedFile] = {}

    async def write(self, key: str, body: bytes) -> None:
        self.objects[key] = body

    async def read_bytes(self, key: str) -> bytes | None:
        return self.objects.get(key)

    async def record(self, file: UploadedFile) -> None:
        self.rows[file.id] = file

    async def lookup(self, tenant_id: TenantId, file_id: FileId) -> UploadedFile | None:
        found = self.rows.get(file_id)
        return found if found is not None and found.tenant_id == tenant_id else None

    async def page(
        self, tenant_id: TenantId, window: FileWindow, limit: int
    ) -> tuple[UploadedFile, ...]:
        raise AssertionError("this file exercises no listing")

    async def erase(self, key: str) -> None:
        raise AssertionError("this file exercises no deletion")

    async def record_deletion_unless_held(
        self, tenant_id: TenantId, file_id: FileId
    ) -> int:
        raise AssertionError("this file exercises no deletion")

    async def deletion_recorded(self, file_id: FileId) -> bool:
        """Answered for real, because the download route asks it on every request.

        Nothing here deletes, so it is always false. Answering rather than raising is
        what lets this file keep asserting that a listed resource id is the id that
        downloads the file -- the check sits on that path now.
        """
        return False


_PORT: UploadedFileStorage = FakeStorage()
"""Graded against the port by mypy --strict rather than by a runtime check, because the
Protocol is not runtime_checkable and an isinstance against one that was would compare
method names and not signatures."""


@dataclass(frozen=True)
class Wired:
    """The wired platform, a client onto it, and the engine its connections belong to.

    The engine comes along because two cases here arrange states no route can produce:
    a creation event that a retention sweep removed, and one whose payload this platform
    did not write. Both are a DELETE against `event_log`, which is the operation the
    table deliberately leaves open.
    """

    platform: Platform
    client: AsyncClient
    engine: AsyncEngine


@pytest.fixture
async def wired(database_url: str) -> AsyncIterator[Wired]:
    platform, engine = build(database_url)
    # Only the storage under `FileStore` is replaced. `build` binds it to a real bucket
    # named by the environment, and an offline test has none -- the unconfigured
    # stand-in refuses every call, so an upload could not happen at all and no Session
    # could name a file. Everything above it is the wired object.
    stored = replace(
        platform, file_store=FileStore(FakeStorage(), UploadSizeLimit(4096))
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app(stored)),
            base_url="http://tenant",
        ) as client:
            yield Wired(platform=stored, client=client, engine=engine)
    finally:
        await engine.dispose()


def _without_request_id(response: Response) -> str:
    """One response's body text with its own per-request id masked out.

    The id is minted per request, before any handler decides anything, so no two
    responses carry the same one and it reports nothing about either. It has to come out
    before two refusals can be compared byte for byte: left in, every pair differs on it
    and the comparison loses the power to detect the differences it exists to detect.
    """
    body: dict[str, object] = response.json()
    return response.text.replace(str(body["request_id"]), "<request>")


def _headers(tenant: uuid.UUID) -> dict[str, str]:
    return {TENANT_HEADER: str(tenant)}


async def _upload(
    client: AsyncClient, tenant: uuid.UUID, name: str
) -> dict[str, object]:
    """One file through the real upload route, and what that route recorded about it."""
    stored = await client.post(
        "/v1/files",
        files={"file": (name, _MARKER + name.encode(), "text/plain")},
        headers=_headers(tenant),
    )
    assert stored.status_code == 201, stored.text
    recorded: dict[str, object] = stored.json()
    return recorded


async def _a_session(
    client: AsyncClient, tenant: uuid.UUID, file_ids: list[str]
) -> str:
    """A Session created through the real route, holding exactly these file ids.

    Every dependency is registered through its own route rather than written into a
    table, so the ids the create call names are ones the platform issued to this tenant
    -- which is what the create path re-checks before it appends anything.
    """
    headers = _headers(tenant)
    registered = await client.post(
        "/v1/agents",
        json={
            "name": f"resources-fixture-{uuid.uuid4()}",
            "instructions": "irrelevant to these tests",
            "model": "gpt-5-codex",
            "skills_repository": "git@github.com:acme/skills.git",
            "skills_revision": _SKILLS_SHA,
        },
        headers=headers,
    )
    assert registered.status_code == 201, registered.text
    shape = await client.post(
        "/v1/environments",
        json={"name": "resources-fixture", "runtime_image": _FIXTURE_IMAGE},
        headers=headers,
    )
    assert shape.status_code == 201, shape.text
    created = await client.post(
        "/v1/sessions",
        json={
            "definition_id": registered.json()["id"],
            "environment_id": shape.json()["id"],
            "file_ids": file_ids,
            "grant": [],
            "scope": {},
            "budget_minor_units": 500,
            "budget_currency": "USD",
            "retention_days": 7,
        },
        headers=headers,
    )
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


_DROP_CREATION_EVENT = sa.text(
    "DELETE FROM event_log WHERE session_id = :id AND type = :type"
).bindparams(sa.bindparam("id", type_=sa.Uuid()), sa.bindparam("type", type_=sa.Text()))


async def _forget_the_creation_event(engine: AsyncEngine, session_id: str) -> None:
    """Remove the row a retention sweep would remove, and nothing else.

    A DELETE and not an UPDATE, because the table refuses an UPDATE outright (migration
    0001) -- which is also why the malformed-payload case below has to append a fresh
    creation event rather than edit the one that is there.
    """
    async with engine.begin() as connection:
        await connection.execute(
            _DROP_CREATION_EVENT,
            {"id": uuid.UUID(session_id), "type": lifecycle.SESSION_CREATED},
        )


async def test_a_session_lists_exactly_the_files_it_was_created_with(
    wired: Wired,
) -> None:
    """The ids, all of them, in the order the create call named -- and no others.

    Order is asserted rather than membership because it is the order the files are
    written into the pod: a list that sorted them would describe a workspace the agent
    does not have.
    """
    tenant = uuid.uuid4()
    first = await _upload(wired.client, tenant, "first.txt")
    second = await _upload(wired.client, tenant, "second.txt")
    third = await _upload(wired.client, tenant, "third.txt")
    named = [str(third["id"]), str(first["id"]), str(second["id"])]
    session_id = await _a_session(wired.client, tenant, named)

    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )

    assert listed.status_code == 200, listed.text
    assert [entry["id"] for entry in listed.json()["data"]] == named


@pytest.mark.parametrize("field", sorted(AttachedResourceView.model_fields))
async def test_every_field_of_a_listed_resource_carries_the_recorded_fact(
    wired: Wired, field: str
) -> None:
    """One assertion per member of the view, parametrized over the view itself.

    Each field of `AttachedResourceView` is a separate decision about what a caller is
    told, and a test that compared whole dictionaries would pass while one of them was
    wired to the wrong source. Driving the parametrize off `model_fields` is what makes
    a field added later fail here rather than ship ungraded: the expectation below has
    no entry for a name it does not know, and raises `KeyError`.

    `type` is the one field with no counterpart on the upload, because it says what kind
    of resource this is rather than anything about the file.
    """
    tenant = uuid.uuid4()
    uploaded = await _upload(wired.client, tenant, "notes.txt")
    session_id = await _a_session(wired.client, tenant, [str(uploaded["id"])])

    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )

    assert listed.status_code == 200, listed.text
    entry = listed.json()["data"][0]
    expected = "file" if field == "type" else uploaded[field]
    assert entry[field] == expected


async def test_the_listed_resource_id_is_the_id_that_downloads_the_file(
    wired: Wired,
) -> None:
    """The list mints no identifier of its own, and this is what that buys a caller.

    If the resource id were a separate value there would have to be a mapping from it to
    a file id stored somewhere, and that store would be a second record of which files a
    Session holds. Asserted end to end rather than by comparing two strings: the id from
    the list is handed straight to the download route.
    """
    tenant = uuid.uuid4()
    uploaded = await _upload(wired.client, tenant, "report.txt")
    session_id = await _a_session(wired.client, tenant, [str(uploaded["id"])])
    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )
    assert listed.status_code == 200, listed.text

    resource_id = listed.json()["data"][0]["id"]
    downloaded = await wired.client.get(
        f"/v1/files/{resource_id}/content", headers=_headers(tenant)
    )

    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == _MARKER + b"report.txt"


async def test_the_list_serves_a_files_facts_and_never_its_bytes(wired: Wired) -> None:
    """No entry carries content, and no entry carries a field nobody declared.

    Both halves matter. The bytes are already a resource of their own, so serving them
    here would be a second answer about one object -- and a field appearing that the
    view does not declare would mean the response is being assembled somewhere other
    than through the model that documents it.
    """
    tenant = uuid.uuid4()
    uploaded = await _upload(wired.client, tenant, "secret.txt")
    session_id = await _a_session(wired.client, tenant, [str(uploaded["id"])])

    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )

    assert listed.status_code == 200, listed.text
    assert _MARKER.decode() not in listed.text
    assert set(listed.json()["data"][0]) == set(AttachedResourceView.model_fields)


async def test_a_session_created_with_no_files_lists_nothing(wired: Wired) -> None:
    """An empty list and a 200, not a 404: the Session is there and holds nothing."""
    tenant = uuid.uuid4()
    session_id = await _a_session(wired.client, tenant, [])

    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )

    assert listed.status_code == 200, listed.text
    assert listed.json() == {"data": []}


async def test_a_creation_event_that_names_no_files_at_all_lists_nothing(
    wired: Wired,
) -> None:
    """A Session created before the field existed holds no files, and says so.

    Absent is not the same as unreadable and is not an error -- which is the reason the
    payload is parsed through a model with a default rather than indexed. Arranged by
    removing the creation event and appending one without the key, because the create
    route always writes it and no route can produce this payload today.
    """
    tenant = uuid.uuid4()
    session_id = await _a_session(wired.client, tenant, [])
    await _forget_the_creation_event(wired.engine, session_id)
    await wired.platform.event_log_append.append(
        SessionId(uuid.UUID(session_id)),
        lifecycle.SESSION_CREATED,
        {"environment_id": str(uuid.uuid4())},
    )

    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )

    assert listed.status_code == 200, listed.text
    assert listed.json() == {"data": []}


async def test_listing_another_tenants_session_is_refused_not_answered(
    wired: Wired,
) -> None:
    """And refused in the same words as an id nobody ever created.

    The Event Log is keyed by Session and carries no tenant, so the read this route does
    succeeds against anybody's Session -- the registry is the only thing standing
    between a stranger and another tenant's file ids. The two bodies are compared rather
    than only their statuses, because a difference of one word is enough to turn the
    refusal into an oracle for which Session ids exist. Each body echoes back the id its
    own request named, so that one value is masked before they are compared -- a caller
    already knows the id it sent, and leaving it in would make two identical refusals
    look different.
    """
    owner, stranger = uuid.uuid4(), uuid.uuid4()
    uploaded = await _upload(wired.client, owner, "private.txt")
    session_id = await _a_session(wired.client, owner, [str(uploaded["id"])])
    invented = uuid.uuid4()

    theirs = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(stranger)
    )
    nobodys = await wired.client.get(
        f"/v1/sessions/{invented}/resources", headers=_headers(stranger)
    )

    assert theirs.status_code == STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]
    assert theirs.json()["error"]["code"] == ErrorCode.SESSION_NOT_FOUND.value
    assert str(uploaded["id"]) not in theirs.text
    assert theirs.status_code == nobodys.status_code
    theirs_body = _without_request_id(theirs).replace(session_id, "<asked-for>")
    nobodys_body = _without_request_id(nobodys).replace(str(invented), "<asked-for>")
    assert theirs_body == nobodys_body


async def test_listing_without_saying_who_is_asking_is_refused(wired: Wired) -> None:
    """The gate sits on the router, so it applies to a route nobody remembered to gate.

    Driven rather than read off the source: the structural scan in `test_tenancy.py`
    proves the declaration exists, and this proves the declaration is the thing FastAPI
    resolves.
    """
    tenant = uuid.uuid4()
    session_id = await _a_session(wired.client, tenant, [])

    listed = await wired.client.get(f"/v1/sessions/{session_id}/resources")

    assert listed.status_code == 400
    assert listed.json()["error"]["code"] == "request.tenant_missing"


async def test_a_session_whose_creation_event_is_gone_is_refused_not_emptied(
    wired: Wired,
) -> None:
    """410 and a reason, because "no files" would be a claim this route cannot make.

    Creation appends the event before it writes the registry row, so a Session the
    registry knows had a creation event -- its absence means the row is no longer
    retained. Answering `{"data": []}` would report a tenant's attached files as never
    having existed, which is the one wrong answer that looks like a right one.
    """
    tenant = uuid.uuid4()
    uploaded = await _upload(wired.client, tenant, "swept.txt")
    session_id = await _a_session(wired.client, tenant, [str(uploaded["id"])])
    await _forget_the_creation_event(wired.engine, session_id)

    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )

    assert listed.status_code == STATUS_FOR[ErrorCode.EVENT_RANGE_EXPIRED]
    body = listed.json()
    assert body["error"]["code"] == ErrorCode.EVENT_RANGE_EXPIRED.value
    assert body["error"]["detail"]["reason"] == REASON_CREATION_RECORD_NOT_RETAINED


async def test_a_creation_payload_this_platform_cannot_read_is_refused(
    wired: Wired,
) -> None:
    """An unreadable file list is refused rather than reported as an empty one.

    The distinction this proves is the reason the payload is parsed instead of indexed:
    a hand-rolled read cannot tell "names no files" from "names them in a form this code
    did not write", and reporting the first when the second is true tells a tenant their
    documents are gone.
    """
    tenant = uuid.uuid4()
    session_id = await _a_session(wired.client, tenant, [])
    await _forget_the_creation_event(wired.engine, session_id)
    await wired.platform.event_log_append.append(
        SessionId(uuid.UUID(session_id)),
        lifecycle.SESSION_CREATED,
        {"file_ids": "notes.txt"},
    )

    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )

    assert listed.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    body = listed.json()
    assert body["error"]["code"] == ErrorCode.INTERNAL.value
    assert body["error"]["detail"]["reason"] == REASON_CREATION_RECORD_UNREADABLE


async def test_a_file_the_store_no_longer_has_is_the_platforms_fault(
    wired: Wired,
) -> None:
    """Named as internal rather than as a bad request, and the file is named.

    The create route checked every one of these ids while the tenant was on the
    connection and an uploaded file's row is never rewritten, so an id here resolving to
    nothing is something the platform lost -- not something the caller got wrong.
    """
    tenant = uuid.uuid4()
    session_id = await _a_session(wired.client, tenant, [])
    lost = uuid.uuid4()
    await _forget_the_creation_event(wired.engine, session_id)
    await wired.platform.event_log_append.append(
        SessionId(uuid.UUID(session_id)),
        lifecycle.SESSION_CREATED,
        {"file_ids": [str(lost)]},
    )

    listed = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )

    assert listed.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    body = listed.json()
    assert body["error"]["detail"]["reason"] == REASON_FILE_NOT_FOUND
    assert body["error"]["detail"]["file_id"] == str(lost)


async def test_a_process_with_no_object_store_says_so_rather_than_listing_nothing(
    wired: Wired,
) -> None:
    """The one refusal that is about this process and not about this Session.

    A second app is built over the same log and registry with the unconfigured store
    behind it, which is what a deployment missing its bucket has. The Session was
    created through the configured one, so the ids in its record are real and only the
    read of them fails.
    """
    tenant = uuid.uuid4()
    uploaded = await _upload(wired.client, tenant, "notes.txt")
    session_id = await _a_session(wired.client, tenant, [str(uploaded["id"])])
    bucketless = create_app(
        replace(wired.platform, file_store=unconfigured_file_store())
    )

    async with AsyncClient(
        transport=ASGITransport(app=bucketless), base_url="http://tenant"
    ) as caller:
        listed = await caller.get(
            f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
        )

    assert listed.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    assert listed.json()["error"]["detail"]["reason"] == REASON_STORAGE_UNCONFIGURED


async def test_attaching_a_resource_after_creation_is_refused(wired: Wired) -> None:
    """The refusal a caller reads, and it names where files are chosen instead.

    A published code with the reason in `detail`: the closed set carries no member for
    "this resource set does not change", and adding one is a version event. So a caller
    branches on `request.invalid` and then on the reason, which is the shape the file
    surface established.
    """
    tenant = uuid.uuid4()
    uploaded = await _upload(wired.client, tenant, "late.txt")
    session_id = await _a_session(wired.client, tenant, [])

    refused = await wired.client.post(
        f"/v1/sessions/{session_id}/resources/{uploaded['id']}",
        headers=_headers(tenant),
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    body = refused.json()
    assert body["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert body["error"]["detail"]["reason"] == REASON_RESOURCES_FIXED_AT_CREATION
    # The create call's own field, by name. A refusal that says "not allowed" and does
    # not say where the thing IS allowed sends the caller to the docs for the answer.
    assert "file_ids" in body["error"]["message"]


async def test_the_refusal_does_not_turn_on_whether_a_turn_has_completed(
    wired: Wired,
) -> None:
    """Both sides of the recovery boundary get the identical answer.

    This is the assertion that pins which of the three readings of this verb was chosen.
    A completed Turn is the platform's durability boundary (ADR-004) and it is the line
    the pod-placement path already turns on, so "refuse only after a Turn has run" was
    the available alternative. It is not what happens: the set is fixed from creation,
    so the answer is the same before the first Turn and after one.
    """
    tenant = uuid.uuid4()
    uploaded = await _upload(wired.client, tenant, "mid-run.txt")
    session_id = await _a_session(wired.client, tenant, [])

    before = await wired.client.post(
        f"/v1/sessions/{session_id}/resources/{uploaded['id']}",
        headers=_headers(tenant),
    )
    await wired.platform.event_log_append.append(
        SessionId(uuid.UUID(session_id)), turn.TURN_COMPLETED, {}
    )
    after = await wired.client.post(
        f"/v1/sessions/{session_id}/resources/{uploaded['id']}",
        headers=_headers(tenant),
    )

    assert before.status_code == after.status_code
    assert _without_request_id(before) == _without_request_id(after)
    assert (
        after.json()["error"]["detail"]["reason"] == REASON_RESOURCES_FIXED_AT_CREATION
    )


async def test_the_refusal_is_the_same_whatever_it_is_pointed_at(wired: Wired) -> None:
    """One answer for a Session you own, a stranger's, and one nobody created.

    The refusal reads nothing, so it cannot vary -- and that is the property worth
    asserting rather than the implementation detail behind it: a verb that answered
    differently for a Session that exists would be a way to find out which ids do.
    """
    owner, stranger = uuid.uuid4(), uuid.uuid4()
    mine = await _a_session(wired.client, owner, [])
    invented, resource = uuid.uuid4(), uuid.uuid4()

    answers = [
        await wired.client.post(
            f"/v1/sessions/{target}/resources/{resource}", headers=_headers(caller)
        )
        for target, caller in ((mine, owner), (mine, stranger), (invented, stranger))
    ]

    assert {answer.status_code for answer in answers} == {
        STATUS_FOR[ErrorCode.REQUEST_INVALID]
    }
    assert {answer.json()["error"]["message"] for answer in answers} == {
        answers[0].json()["error"]["message"]
    }


async def test_the_refusal_appends_nothing_to_the_session_log(wired: Wired) -> None:
    """No event, of any type, from a call that did nothing.

    The Event Log is append-only and `session.created` already says what this Session
    holds, so an event here would be a second record of one fact -- and a payload
    rewritten in place would not be an event at all, it would be a rewrite of a log the
    store refuses to rewrite. The whole log is compared, not just its length, so an
    appended-then-something-else-removed sequence cannot pass.
    """
    tenant = uuid.uuid4()
    uploaded = await _upload(wired.client, tenant, "ignored.txt")
    session_id = SessionId(uuid.UUID(await _a_session(wired.client, tenant, [])))
    before = [
        (event.seq, event.type, event.payload)
        for event in await wired.platform.event_log_range.read(session_id, 1, 2**62)
    ]

    await wired.client.post(
        f"/v1/sessions/{session_id}/resources/{uploaded['id']}",
        headers=_headers(tenant),
    )

    after = [
        (event.seq, event.type, event.payload)
        for event in await wired.platform.event_log_range.read(session_id, 1, 2**62)
    ]
    assert after == before


async def test_refusing_without_saying_who_is_asking_happens_first(
    wired: Wired,
) -> None:
    """An unauthenticated caller does not even reach the refusal.

    Worth its own case rather than folded into the gate test above, because this route
    takes no tenant of its own: the gate is on the router, and nothing in the handler
    would notice its absence.
    """
    refused = await wired.client.post(
        f"/v1/sessions/{uuid.uuid4()}/resources/{uuid.uuid4()}"
    )

    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "request.tenant_missing"
