"""Attaching one more file to a Session that is already running.

Tier 1 (testcontainers, real PostgreSQL 17), and the tier is load-bearing rather than
incidental. What this route decides it decides by folding a Session's Event Log -- what
it already holds, whether a Turn is open, whether it would still take one -- and the log
it folds here is the one the real create, archive and append paths wrote. A fake log
keyed on a dict would let this file's fixtures and this file's subject agree with each
other and prove nothing about either.

Only the uploaded-file storage stands in, because the alternative is an S3 bucket. The
`FileStore` above it is real, so a filename collision is decided on the name the upload
path recorded and a byte ledger is priced from the length it computed.

**`session_attachments` is the refusing default here, and that is deliberate.** `build`
wires the real one only where a pod runner is configured, and an offline test has none.
So every case below is either a Session with no pod yet -- where the route appends and
the placement path delivers later, which is the ordinary path -- or a Session past its
first Turn, where the refusal proves the push is attempted before anything is recorded.
What no offline test can show is bytes arriving in a pod; `tests/pod/` is where that
lives.

Driven with `AsyncClient` over `ASGITransport` rather than `TestClient`, for the reason
`test_sessions_tenant_scope.py` gives: an async engine's pooled connections belong to
the loop that opened them, and `TestClient` runs the app in a loop of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import resources as resources_module
from managed_agent.control.api.routes.files import REASON_FILE_NOT_FOUND
from managed_agent.control.api.routes.resources import (
    FILES_MOUNT_PATH,
    REASON_CREATION_RECORD_NOT_RETAINED,
    REASON_FILE_BUDGET_EXHAUSTED,
    REASON_MOUNT_PATH_FIXED,
)
from managed_agent.control.files.store import (
    FileId,
    FileStore,
    FileWindow,
    UploadedFile,
    UploadedFileStorage,
    UploadSizeLimit,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vocabulary import PUBLISHED, lifecycle, resource, turn

_SKILLS_SHA = "0" * 39 + "c"
_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "c" * 64


class FakeStorage:
    """One dict of objects and one of rows, keyed the way the real store keys them.

    Copied from the two neighbouring files rather than shared, which is the trade they
    record: there is no shared fixtures module under `tests/control/`, and a helper
    package introduced for one class would be a new import surface for every file here.

    Two differences from the copy next door, and both are the subject. Deletion is
    answered rather than refused, because the attach route asks `deletion_recorded` on
    every call and one case here deletes a file and then attaches it. Listing and
    erasing still raise: nothing here performs either, and a fake that answered them
    would let a later change reach this file's subject down a path it never meant to
    exercise.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.rows: dict[FileId, UploadedFile] = {}
        self.deleted: set[FileId] = set()

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
        self.objects.pop(key, None)

    async def record_deletion_unless_held(
        self, tenant_id: TenantId, file_id: FileId
    ) -> int:
        """Records the tombstone and answers "no Session holds it", which is zero.

        The real store writes and counts in one statement, and the count is the whole
        guard: a non-zero answer means the write was rolled back. This cannot count,
        because counting means reading the Sessions that name the file and this holds no
        log -- so it models the branch where nothing holds it, and one test in this file
        deletes exactly one file, which is a file no Session names.

        Getting this wrong was informative rather than academic: a first version
        returned `1`, which is what a real store answers when a live Session holds the
        file, and the delete route refused with `file.in_use` before the case under test
        ran at all. A stub whose return value is not the contract's value fails
        somewhere else and reads as a defect in whatever it reaches.
        """
        self.deleted.add(file_id)
        return 0

    async def deletion_recorded(self, file_id: FileId) -> bool:
        return file_id in self.deleted


_PORT: UploadedFileStorage = FakeStorage()
"""Graded against the port by mypy --strict, not by an isinstance: the Protocol is not
runtime_checkable, and one that was would compare method names and not signatures."""


@dataclass(frozen=True)
class Wired:
    platform: Platform
    client: AsyncClient
    engine: AsyncEngine
    storage: FakeStorage


@pytest.fixture
async def wired(database_url: str) -> AsyncIterator[Wired]:
    platform, engine = build(database_url)
    storage = FakeStorage()
    stored = replace(platform, file_store=FileStore(storage, UploadSizeLimit(4096)))
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app(stored)),
            base_url="http://tenant",
        ) as client:
            yield Wired(platform=stored, client=client, engine=engine, storage=storage)
    finally:
        await engine.dispose()


def _headers(tenant: uuid.UUID) -> dict[str, str]:
    return {TENANT_HEADER: str(tenant)}


def _without_request_id(response: Response) -> str:
    """One response's text with its per-request id masked, so two can be compared."""
    body: dict[str, object] = response.json()
    return response.text.replace(str(body["request_id"]), "<request>")


async def _upload(
    client: AsyncClient, tenant: uuid.UUID, name: str, body: bytes = b"contents"
) -> dict[str, object]:
    stored = await client.post(
        "/v1/files",
        files={"file": (name, body, "text/plain")},
        headers=_headers(tenant),
    )
    assert stored.status_code == 201, stored.text
    recorded: dict[str, object] = stored.json()
    return recorded


async def _a_session(
    client: AsyncClient, tenant: uuid.UUID, file_ids: list[str]
) -> str:
    """A Session through the real routes, holding exactly these file ids."""
    headers = _headers(tenant)
    registered = await client.post(
        "/v1/agents",
        json={
            "name": f"attach-fixture-{uuid.uuid4()}",
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
        json={"name": "attach-fixture", "runtime_image": _FIXTURE_IMAGE},
        headers=headers,
    )
    assert shape.status_code == 201, shape.text
    created = await client.post(
        "/v1/sessions",
        json={
            "definition_id": registered.json()["id"],
            "environment_id": shape.json()["id"],
            "file_ids": file_ids,
            "grant": ["fs.read"],
            "scope": {},
            "budget_minor_units": 500,
            "budget_currency": "USD",
            "retention_days": 7,
        },
        headers=headers,
    )
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


async def _attach(
    wired: Wired,
    tenant: uuid.UUID,
    session_id: str,
    file_id: str,
    mount_path: str | None = None,
) -> Response:
    body: dict[str, object] = {"file_id": file_id, "type": "file"}
    if mount_path is not None:
        body["mount_path"] = mount_path
    return await wired.client.post(
        f"/v1/sessions/{session_id}/resources", json=body, headers=_headers(tenant)
    )


async def _listed(wired: Wired, tenant: uuid.UUID, session_id: str) -> list[str]:
    seen = await wired.client.get(
        f"/v1/sessions/{session_id}/resources", headers=_headers(tenant)
    )
    assert seen.status_code == 200, seen.text
    rows: list[dict[str, object]] = seen.json()["data"]
    return [str(row["id"]) for row in rows]


async def _append(
    wired: Wired, session_id: str, type_: str, payload: dict[str, object]
) -> None:
    """One event straight into the log, for the states no route will produce.

    A submitted Turn nothing closes, and an idle-timeout suspension: the first needs a
    dispatch this process cannot make and the second needs a reaper's clock. Both are
    appends the real paths make, made here directly, which is why they go through
    `event_log_append` rather than an INSERT -- the sequence and the ordering trigger
    are part of what the fold reads.
    """
    await wired.platform.event_log_append.append(
        SessionId(uuid.UUID(session_id)), type_, payload
    )


# --- the vocabulary the route needs ----------------------------------------------


def test_exactly_one_published_type_says_a_resource_was_attached() -> None:
    """The published attach type exists, is the only one in its family, and is used.

    This replaces a guard that asserted the opposite -- that no published type said a
    resource was attached -- and that guard was right for as long as nothing could
    attach one. Its own docstring named the condition for reversing it: the type had to
    arrive with the delivery path that makes it true. So the inverse is asserted here
    rather than the assertion simply deleted, because a published type nothing emits is
    still the defect the first one was pointing at (ADR-013): it becomes part of the API
    version whether or not anything produces it.

    The second half reads the route's source, which is the same shape the router-mount
    guard uses and for the same reason: whether a module *names* a type is a property of
    the file, and every behavioural test in this file would still pass against a route
    that appended some other type entirely -- it would just report a Session holding
    nothing.
    """
    declared = sorted(
        name for name, family in PUBLISHED.items() if family == resource.FAMILY
    )

    assert declared == [resource.SESSION_FILE_ATTACHED], declared
    assert (
        resource.SESSION_FILE_ATTACHED
        in Path(resources_module.__file__ or "").read_text()
    ), (
        "the attach type is published and the route that must emit it does not name "
        "it, so the API version declares an event nothing can produce"
    )


# --- the ordinary path ------------------------------------------------------------


async def test_an_attached_file_joins_what_the_session_holds(wired: Wired) -> None:
    """The whole of the ordinary path: 201, the resource it made, and the list after.

    The response body is compared as one value rather than field by field, and the
    digest and length in it came from the upload route rather than from this test -- so
    a route filling either in from the request would show here.

    The list is asserted too, and that is the half that matters: a 201 that appended
    nothing looks identical from the response alone.
    """
    tenant = uuid.uuid4()
    first = await _upload(wired.client, tenant, "brief.txt")
    second = await _upload(wired.client, tenant, "appendix.txt")
    session_id = await _a_session(wired.client, tenant, [str(first["id"])])

    added = await _attach(wired, tenant, session_id, str(second["id"]))

    assert added.status_code == 201, added.text
    assert added.json() == {
        "id": str(second["id"]),
        "type": "file",
        "filename": "appendix.txt",
        "media_type": "text/plain",
        "byte_length": second["byte_length"],
        "content_sha256": second["content_sha256"],
    }
    assert await _listed(wired, tenant, session_id) == [
        str(first["id"]),
        str(second["id"]),
    ]


async def test_two_attaches_are_listed_in_the_order_they_were_made(
    wired: Wired,
) -> None:
    """Creation's file first, then each attach in the order it was made.

    The two attaches are made in the reverse of their alphabetical and their upload
    order, so a fold that sorted the set -- by name, by id, by anything but the log --
    comes out different. The order is the order the pod's workspace is written in, so a
    list sorted another way would describe a workspace laid out differently from the one
    the agent sees.
    """
    tenant = uuid.uuid4()
    held = await _upload(wired.client, tenant, "held.txt")
    later = await _upload(wired.client, tenant, "later.txt")
    latest = await _upload(wired.client, tenant, "latest.txt")
    session_id = await _a_session(wired.client, tenant, [str(held["id"])])

    for added in (latest, later):
        landed = await _attach(wired, tenant, session_id, str(added["id"]))
        assert landed.status_code == 201, landed.text

    assert await _listed(wired, tenant, session_id) == [
        str(held["id"]),
        str(latest["id"]),
        str(later["id"]),
    ]


async def test_a_session_created_holding_nothing_can_be_attached_to(
    wired: Wired,
) -> None:
    """A Session created with no files is a Session that can be given one.

    Worth its own case because the fold distinguishes "the creation event names no
    files" from "there is no creation event", and the two answers are an empty tuple and
    `None`. Collapsing them would refuse this attach with a retention error.
    """
    tenant = uuid.uuid4()
    only = await _upload(wired.client, tenant, "only.txt")
    session_id = await _a_session(wired.client, tenant, [])

    added = await _attach(wired, tenant, session_id, str(only["id"]))

    assert added.status_code == 201, added.text
    assert await _listed(wired, tenant, session_id) == [str(only["id"])]


# --- mount_path -------------------------------------------------------------------


async def test_the_one_writable_directory_may_be_named(wired: Wired) -> None:
    """Naming the one directory this platform writes to is accepted, not refused.

    The other half of the parametrised refusal below. Without it, a route that refused
    every `mount_path` would pass that one -- and a caller sending the field with the
    value this API documents would be unable to attach anything.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "named.txt")
    session_id = await _a_session(wired.client, tenant, [])

    added = await _attach(
        wired, tenant, session_id, str(file["id"]), mount_path=FILES_MOUNT_PATH
    )

    assert added.status_code == 201, added.text


@pytest.mark.parametrize(
    "asked",
    [
        "/mnt/session/uploads/f",
        "/session/workspace",
        "/session/workspace/files/",
        "files",
        "",
    ],
)
async def test_any_other_mount_path_is_refused_rather_than_ignored(
    wired: Wired, asked: str
) -> None:
    """Every other value is refused, including the ones that look close enough.

    Upstream's own default is in this list, which is the point: a client written against
    their surface sends `/mnt/session/uploads/<file_id>` and has to be told, not quietly
    written somewhere else. `/session/workspace/files/` with a trailing slash and
    `files` relative are here because a comparison written with `startswith` or a
    normalising `Path` would accept them and the receiver would still write one flat
    name.

    The list is asserted empty afterwards: a refusal that appended first would be
    invisible in the response.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "elsewhere.txt")
    session_id = await _a_session(wired.client, tenant, [])

    refused = await _attach(
        wired, tenant, session_id, str(file["id"]), mount_path=asked
    )

    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert refused.json()["error"]["detail"]["reason"] == REASON_MOUNT_PATH_FIXED
    assert await _listed(wired, tenant, session_id) == []


async def test_a_refused_mount_path_reaches_no_store_at_all(wired: Wired) -> None:
    """The refusal comes before any store is touched, so neither id has to exist.

    A Session id nobody created and a file id nobody uploaded. If the mount check ran
    after the ownership read this would be a 404 instead, which is a worse answer: the
    caller would fix the id, try again, and only then learn the real problem.
    """
    tenant = uuid.uuid4()

    refused = await _attach(
        wired, tenant, str(uuid.uuid4()), str(uuid.uuid4()), mount_path="/tmp"
    )

    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["detail"]["reason"] == REASON_MOUNT_PATH_FIXED


# --- the collision ----------------------------------------------------------------


async def test_a_second_file_of_the_same_name_is_refused(wired: Wired) -> None:
    """Two different files with one name: the second is refused, not written over.

    The bodies differ so the two really are two files, and the ids are asserted
    different before anything else -- an upload path that deduplicated on name would
    make this test pass while proving nothing.

    409 and not 400: the request is well formed and would be accepted against a
    different Session. What it collides with is a directory, and the pod's receiver
    renames atomically, so honouring it would replace the earlier file with no record of
    the moment.
    """
    tenant = uuid.uuid4()
    held = await _upload(wired.client, tenant, "report.txt", b"the first one")
    twin = await _upload(wired.client, tenant, "report.txt", b"a different one")
    assert held["id"] != twin["id"]
    session_id = await _a_session(wired.client, tenant, [str(held["id"])])

    refused = await _attach(wired, tenant, session_id, str(twin["id"]))

    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == ErrorCode.RESOURCE_FILENAME_ATTACHED.value
    assert await _listed(wired, tenant, session_id) == [str(held["id"])]


async def test_attaching_the_same_file_twice_is_the_same_refusal(
    wired: Wired,
) -> None:
    """The same file twice collides with itself, and the second call is refused.

    **Not idempotent, on purpose.** A retry that answered 201 again would be harmless
    here and would need the route to distinguish "the same file" from "a different file
    of the same name" -- which is a second rule, on a path where the first one already
    gives the caller a correct and readable answer. The refusal names the filename, so a
    caller that retried a timed-out call learns its first call landed.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "once.txt")
    session_id = await _a_session(wired.client, tenant, [])
    first = await _attach(wired, tenant, session_id, str(file["id"]))
    assert first.status_code == 201, first.text

    again = await _attach(wired, tenant, session_id, str(file["id"]))

    assert again.status_code == 409, again.text
    assert await _listed(wired, tenant, session_id) == [str(file["id"])]


# --- the byte ledger --------------------------------------------------------------


async def test_the_ledger_counts_what_the_session_already_holds(
    wired: Wired, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N attaches each under the budget do not jointly exceed it.

    The budget is patched down to twenty bytes, because the real one is 256 MiB and a
    test that moved that many bytes through an ASGI transport would be measuring the
    transport. Patched on the route's own module rather than on `session_files`, because
    the route reads the name it imported and that is the read under test.

    Ten held plus ten fits, and one more byte does not. Without the ledger every one of
    these is one file under twenty and all three are accepted -- and an `emptyDir` over
    its `sizeLimit` is enforced by kubelet eviction rather than `ENOSPC`, so the pod
    would disappear mid-Turn for a reason visible only in node events.
    """
    monkeypatch.setattr(resources_module, "WORKSPACE_FILE_BUDGET_BYTES", 20)
    tenant = uuid.uuid4()
    held = await _upload(wired.client, tenant, "a.txt", b"0123456789")
    fits = await _upload(wired.client, tenant, "b.txt", b"0123456789")
    over = await _upload(wired.client, tenant, "c.txt", b"0")
    session_id = await _a_session(wired.client, tenant, [str(held["id"])])

    fitted = await _attach(wired, tenant, session_id, str(fits["id"]))
    assert fitted.status_code == 201, fitted.text
    refused = await _attach(wired, tenant, session_id, str(over["id"]))

    assert refused.status_code == 400, refused.text
    detail = refused.json()["error"]["detail"]
    assert detail["reason"] == REASON_FILE_BUDGET_EXHAUSTED
    assert await _listed(wired, tenant, session_id) == [
        str(held["id"]),
        str(fits["id"]),
    ]


async def test_one_attach_under_the_budget_is_not_refused_for_the_others(
    wired: Wired, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the case above: the same budget, and an attach that fits.

    Without this, a route that refused every attach once the budget was patched would
    pass the ledger test. It is the same two files, minus the third.
    """
    monkeypatch.setattr(resources_module, "WORKSPACE_FILE_BUDGET_BYTES", 20)
    tenant = uuid.uuid4()
    held = await _upload(wired.client, tenant, "a.txt", b"0123456789")
    fits = await _upload(wired.client, tenant, "b.txt", b"0123456789")
    session_id = await _a_session(wired.client, tenant, [str(held["id"])])

    added = await _attach(wired, tenant, session_id, str(fits["id"]))

    assert added.status_code == 201, added.text


# --- the file itself --------------------------------------------------------------


async def test_a_file_this_tenant_does_not_hold_is_a_404(wired: Wired) -> None:
    """A file id this tenant does not hold is a 404, and one refusal covers both ways.

    Another tenant's real file and an id nobody ever uploaded get the same status, the
    same code and the same sentence. Two distinguishable answers would let a caller
    holding an id learn from the refusal whether it names somebody else's file.
    """
    tenant = uuid.uuid4()
    other = uuid.uuid4()
    theirs = await _upload(wired.client, other, "theirs.txt")
    session_id = await _a_session(wired.client, tenant, [])

    refused = await _attach(wired, tenant, session_id, str(theirs["id"]))
    never = await _attach(wired, tenant, session_id, str(uuid.uuid4()))

    assert refused.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert refused.json()["error"]["code"] == ErrorCode.FILE_NOT_FOUND.value
    assert refused.json()["error"]["detail"]["reason"] == REASON_FILE_NOT_FOUND
    assert never.status_code == refused.status_code
    assert _without_request_id(never).replace(
        str(uuid.UUID(str(never.json()["error"]["detail"]["file_id"]))), "<file>"
    ) == _without_request_id(refused).replace(str(theirs["id"]), "<file>")


async def test_a_deleted_file_is_refused_as_gone(wired: Wired) -> None:
    """A deleted file is 410, and the attach is not recorded.

    The deletion goes through the real delete route, so what this asserts is that the
    two surfaces agree: the row survives a deletion by design -- which is why the id
    still resolves -- and the attach has to ask a second question to find out. A route
    that only called `describe` would accept this and put a document the platform has
    recorded as gone into a running workspace.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "withdrawn.txt")
    session_id = await _a_session(wired.client, tenant, [])
    erased = await wired.client.delete(
        f"/v1/files/{file['id']}", headers=_headers(tenant)
    )
    assert erased.status_code in (200, 204), erased.text

    refused = await _attach(wired, tenant, session_id, str(file["id"]))

    assert refused.status_code == STATUS_FOR[ErrorCode.FILE_DELETED]
    assert refused.json()["error"]["code"] == ErrorCode.FILE_DELETED.value
    assert await _listed(wired, tenant, session_id) == []


# --- the Session's own state ------------------------------------------------------


async def test_an_attach_to_another_tenants_session_is_one_404(wired: Wired) -> None:
    """Another tenant's Session and a Session nobody created are one 404.

    Compared byte for byte with the ids masked, not merely on the status. The Event Log
    is keyed by Session and holds no tenant, so an attach that folded before it checked
    ownership would succeed against somebody else's Session. A refusal that differed
    between these two would also tell a caller holding an id which of the cases it was
    in.
    """
    mine = uuid.uuid4()
    theirs = uuid.uuid4()
    file = await _upload(wired.client, mine, "mine.txt")
    not_mine = await _a_session(wired.client, theirs, [])
    never = str(uuid.uuid4())

    refused = await _attach(wired, mine, not_mine, str(file["id"]))
    absent = await _attach(wired, mine, never, str(file["id"]))

    assert refused.status_code == 404, refused.text
    assert absent.status_code == 404
    assert _without_request_id(refused).replace(not_mine, "<id>") == (
        _without_request_id(absent).replace(never, "<id>")
    )


async def test_an_attach_to_an_archived_session_is_refused(wired: Wired) -> None:
    """An archived Session accepts no file, and the refusal names the state.

    Archived through the real route, so the `session.stopped` event is the one that path
    appends. 409 rather than 404: the id is right and the Session is deliberately past
    taking work, so 404 would invite the caller to go looking for a mistake in their id.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "late.txt")
    session_id = await _a_session(wired.client, tenant, [])
    stopped = await wired.client.post(
        f"/v1/sessions/{session_id}/archive", headers=_headers(tenant)
    )
    assert stopped.status_code == 200, stopped.text

    refused = await _attach(wired, tenant, session_id, str(file["id"]))

    assert refused.status_code == 409, refused.text
    assert (
        refused.json()["error"]["code"] == ErrorCode.SESSION_NOT_ACCEPTING_TURNS.value
    )
    assert refused.json()["error"]["detail"]["state"] == "stopped"
    assert await _listed(wired, tenant, session_id) == []


async def test_an_attach_to_a_reaped_session_is_the_same_refusal(
    wired: Wired,
) -> None:
    """A reaped Session is the same refusal, and needs no rule of its own.

    Reaping suspends, and a suspended Session's pod is gone: `place_resuming` raises
    unconditionally, so no future Turn will ever read a file attached now. That is why
    one rule -- "would this Session still take a Turn" -- covers archiving and reaping
    both, and why the state travels in the detail so a caller can tell which it hit.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "reaped.txt")
    session_id = await _a_session(wired.client, tenant, [])
    await _append(
        wired,
        session_id,
        lifecycle.SESSION_SUSPENDED,
        {"stop_reason": lifecycle.StopReason.IDLE_TIMEOUT.value},
    )

    refused = await _attach(wired, tenant, session_id, str(file["id"]))

    assert refused.status_code == 409, refused.text
    assert (
        refused.json()["error"]["code"] == ErrorCode.SESSION_NOT_ACCEPTING_TURNS.value
    )
    assert refused.json()["error"]["detail"]["state"] == "suspended"


async def test_an_attach_while_a_turn_is_open_is_refused(wired: Wired) -> None:
    """A Turn in flight refuses the attach, and the refusal names the Turn.

    The runtime already holds its prompt, so a file landing now may or may not be read
    and no record afterwards could say which. The Turn id is in the detail because the
    caller's next move is to interrupt that Turn, and a refusal that did not name it
    would leave them guessing which.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "midturn.txt")
    session_id = await _a_session(wired.client, tenant, [])
    turn_id = str(uuid.uuid4())
    await _append(wired, session_id, turn.TURN_SUBMITTED, {"turn_id": turn_id})

    refused = await _attach(wired, tenant, session_id, str(file["id"]))

    assert refused.status_code == STATUS_FOR[ErrorCode.SESSION_TURN_IN_FLIGHT]
    assert refused.json()["error"]["code"] == ErrorCode.SESSION_TURN_IN_FLIGHT.value
    assert refused.json()["error"]["detail"]["turn_id"] == turn_id
    assert await _listed(wired, tenant, session_id) == []


async def test_the_same_attach_is_accepted_once_that_turn_is_closed(
    wired: Wired,
) -> None:
    """The same attach is accepted once the Turn is closed. The control for above.

    Closed with `turn.failed` rather than `turn.completed`, which is deliberate:
    `open_turn` reads both terminal types, and a version that watched only completions
    would leave every Session that had ever failed a Turn unable to accept a file for
    ever. It also keeps this case on the no-pod branch, so what it grades is the refusal
    lifting and not delivery.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "after.txt")
    session_id = await _a_session(wired.client, tenant, [])
    turn_id = str(uuid.uuid4())
    await _append(wired, session_id, turn.TURN_SUBMITTED, {"turn_id": turn_id})
    await _append(wired, session_id, turn.TURN_FAILED, {"turn_id": turn_id})

    added = await _attach(wired, tenant, session_id, str(file["id"]))

    assert added.status_code == 201, added.text


# --- delivery, and the order it happens in ----------------------------------------


async def test_a_session_past_its_first_turn_needs_a_pod_that_will_take_the_file(
    wired: Wired,
) -> None:
    """Past its first Turn the bytes go first, and a failed push records nothing.

    A completed Turn means `FirstTurnPlacement` will not place a pod again, so the file
    has to go down now or never. This deployment has no pod runner, so the refusing
    default answers, and the 502 is the honest reading: the platform could not put the
    file where the agent would look.

    The assertion that matters is the empty list afterwards. It is the whole of the
    push-before-append order: appending first would leave a Session whose record names a
    file its pod does not have, with nothing that would ever push it, and the 502 would
    have told the caller to retry a call that had already changed the record.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "delivered.txt")
    session_id = await _a_session(wired.client, tenant, [])
    turn_id = str(uuid.uuid4())
    await _append(wired, session_id, turn.TURN_SUBMITTED, {"turn_id": turn_id})
    await _append(wired, session_id, turn.TURN_COMPLETED, {"turn_id": turn_id})

    refused = await _attach(wired, tenant, session_id, str(file["id"]))

    assert refused.status_code == STATUS_FOR[ErrorCode.TURN_UNDELIVERABLE]
    assert refused.json()["error"]["code"] == ErrorCode.TURN_UNDELIVERABLE.value
    assert await _listed(wired, tenant, session_id) == [], (
        "the push was refused and the attach was recorded anyway, so this session's "
        "record names a file its pod does not have and nothing will ever push it"
    )


async def test_the_creation_event_being_gone_refuses_the_attach(wired: Wired) -> None:
    """A Session whose creation event retention removed cannot be added to.

    The same DELETE a retention sweep performs -- a DELETE and not an UPDATE, because
    the table refuses an UPDATE outright. Refused rather than treated as a Session
    holding nothing: creation appends that event before it writes the registry row, so a
    row without the event means the event was retained once, and attaching to a set that
    cannot be read would put a file beside documents nobody can enumerate.
    """
    tenant = uuid.uuid4()
    file = await _upload(wired.client, tenant, "orphan.txt")
    session_id = await _a_session(wired.client, tenant, [])
    async with wired.engine.begin() as connection:
        await connection.execute(
            sa.text(
                "DELETE FROM event_log WHERE session_id = :id AND type = :type"
            ).bindparams(
                sa.bindparam("id", type_=sa.Uuid()),
                sa.bindparam("type", type_=sa.Text()),
            ),
            {"id": uuid.UUID(session_id), "type": lifecycle.SESSION_CREATED},
        )

    refused = await _attach(wired, tenant, session_id, str(file["id"]))

    assert refused.status_code == STATUS_FOR[ErrorCode.EVENT_RANGE_EXPIRED]
    assert (
        refused.json()["error"]["detail"]["reason"]
        == REASON_CREATION_RECORD_NOT_RETAINED
    )
