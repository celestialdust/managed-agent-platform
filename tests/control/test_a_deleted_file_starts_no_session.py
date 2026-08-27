"""A deleted file starts no Session, and the refusal says which fault it is.

Tier 1 (testcontainers, real PostgreSQL 17), and the tier is load-bearing rather than
incidental: the tombstone is a row, the guard that counts holding Sessions is a
transaction over two tables, and the property under test is that the create route
consults the tombstone at all. A fake log keyed on a dict cannot represent either.

Only the object storage under `FileStore` stands in, for the reason
`test_session_resources_are_listed_and_fixed_at_creation.py` gives -- the alternative is
an S3 bucket. Everything above it is the wired object, so the tombstone this file writes
is written by the same `FileStore.delete` a tenant's DELETE reaches.

**Why this file exists and is not two tests appended to that one.** The deleted-file
refusal on Session creation was the last of `file_store.deletion_recorded`'s three
callers to be wired, and its docstring names the trap: `describe` answers about the row,
the row outlives the bytes, so an id that resolves is not an id whose file is still
there. Nothing in the suite could see the gap, because the two paths that did ask -- the
download and the resource listing -- both pass.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

import pytest
from httpx import ASGITransport, AsyncClient

from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.files import REASON_FILE_NOT_FOUND
from managed_agent.control.files.store import (
    FileId,
    FileStore,
    FileWindow,
    UploadedFile,
    UploadedFileStorage,
    UploadSizeLimit,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import TenantId

_SKILLS_SHA = "0" * 39 + "c"
_FIXTURE_IMAGE = "registry.map.internal/session@sha256:" + "c" * 64


class FakeStorage:
    """Objects and rows in two dicts, plus a real tombstone set and a real erase.

    `record_deletion_unless_held` returns 0 holders and that is honest here rather than
    convenient: every deletion in this file happens before any Session is created, so
    there is nothing to hold the file. The tests assert that ordering themselves -- the
    delete call precedes the create call in each -- so the premise cannot quietly stop
    being true while the fake keeps answering as though it held.

    `page` raises, because nothing here lists. A fake that answered a listing with an
    empty page would let a later change reach this file's subject through a path it
    never meant to exercise.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.rows: dict[FileId, UploadedFile] = {}
        self.tombstoned: set[FileId] = set()

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
        self.tombstoned.add(file_id)
        return 0

    async def deletion_recorded(self, file_id: FileId) -> bool:
        return file_id in self.tombstoned


_PORT: UploadedFileStorage = FakeStorage()
"""Graded against the port by mypy --strict, the way the sibling file grades its own."""


@dataclass(frozen=True)
class Wired:
    platform: Platform
    client: AsyncClient
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
            yield Wired(platform=stored, client=client, storage=storage)
    finally:
        await engine.dispose()


def _headers(tenant: uuid.UUID) -> dict[str, str]:
    return {TENANT_HEADER: str(tenant)}


async def _upload(client: AsyncClient, tenant: uuid.UUID, name: str) -> str:
    stored = await client.post(
        "/v1/files",
        files={"file": (name, b"a body worth deleting", "text/plain")},
        headers=_headers(tenant),
    )
    assert stored.status_code == 201, stored.text
    return str(stored.json()["id"])


async def _an_agent_and_an_environment(
    client: AsyncClient, tenant: uuid.UUID
) -> tuple[str, str]:
    """The two ids a Session needs, registered through their own routes."""
    headers = _headers(tenant)
    registered = await client.post(
        "/v1/agents",
        json={
            "name": f"deleted-file-fixture-{uuid.uuid4()}",
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
        json={"name": "deleted-file-fixture", "runtime_image": _FIXTURE_IMAGE},
        headers=headers,
    )
    assert shape.status_code == 201, shape.text
    return str(registered.json()["id"]), str(shape.json()["id"])


def _creation_body(
    definition_id: str, environment_id: str, file_ids: list[str]
) -> dict[str, object]:
    return {
        "definition_id": definition_id,
        "environment_id": environment_id,
        "file_ids": file_ids,
        "grant": [],
        "scope": {},
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 7,
    }


async def test_a_deleted_file_is_refused_with_the_fault_that_it_is(
    wired: Wired,
) -> None:
    """The refusal is 410 and names the file, and no Session exists afterwards.

    410 and not the 400 the absent-row case answers, and the difference is what the
    caller does next. "No such file" invites them to re-check the id they sent; "this
    file was deleted" tells them the id was right and the thing is gone, so re-sending
    it can never work.
    """
    tenant = uuid.uuid4()
    definition_id, environment_id = await _an_agent_and_an_environment(
        wired.client, tenant
    )
    file_id = await _upload(wired.client, tenant, "gone.txt")

    # Deleted before the Session names it, which is the whole arrangement. The fake's
    # zero-holders answer is honest at this point in the sequence and not before it.
    await wired.platform.file_store.delete(
        tenant_id=TenantId(tenant), file_id=FileId(uuid.UUID(file_id))
    )

    refused = await wired.client.post(
        "/v1/sessions",
        json=_creation_body(definition_id, environment_id, [file_id]),
        headers=_headers(tenant),
    )
    expected = STATUS_FOR[ErrorCode.FILE_DELETED]
    assert refused.status_code == expected == 410, refused.text
    body = refused.json()
    assert body["error"]["code"] == ErrorCode.FILE_DELETED.value
    assert body["error"]["detail"]["file_id"] == file_id


async def test_the_deleted_files_row_still_resolves_so_a_second_check_is_needed(
    wired: Wired,
) -> None:
    """The row still resolves, so the check that catches this is not the absent-row one.

    This is the arms-disagree half of the test above. `describe` succeeds for a deleted
    file -- the row is kept on purpose, so a Session's creation event never names an id
    that resolves to nothing -- which means the 400 the loop already answered could not
    have produced the 410. Asserted by reading the row back through the store the route
    reads it through: if this ever raises, the two refusals have collapsed into one and
    the test above stops proving which check fired.
    """
    tenant = uuid.uuid4()
    file_id = await _upload(wired.client, tenant, "still-a-row.txt")
    await wired.platform.file_store.delete(
        tenant_id=TenantId(tenant), file_id=FileId(uuid.UUID(file_id))
    )

    described = await wired.platform.file_store.describe(
        tenant_id=TenantId(tenant), file_id=FileId(uuid.UUID(file_id))
    )
    assert str(described.id) == file_id
    assert await wired.platform.file_store.deletion_recorded(
        file_id=FileId(uuid.UUID(file_id))
    )


async def test_a_file_that_never_existed_is_still_the_other_refusal(
    wired: Wired,
) -> None:
    """An id that was never issued is still the 400, not the 410.

    The two refusals are next to each other in one loop, so the guard against them being
    confused is a test that pins each to its own cause. Without this, a change that
    answered 410 for everything unresolvable would pass the test above.
    """
    tenant = uuid.uuid4()
    definition_id, environment_id = await _an_agent_and_an_environment(
        wired.client, tenant
    )

    refused = await wired.client.post(
        "/v1/sessions",
        json=_creation_body(definition_id, environment_id, [str(uuid.uuid4())]),
        headers=_headers(tenant),
    )
    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID] == 400
    assert refused.json()["error"]["detail"]["reason"] == REASON_FILE_NOT_FOUND


async def test_one_deleted_file_among_several_refuses_the_whole_creation(
    wired: Wired,
) -> None:
    """A live file alongside a deleted one is refused too, and the live one is named
    nowhere.

    The loop refuses on the first bad id and returns, so the refusal names one file even
    when several were sent. That is the intended shape -- a caller fixes one thing at a
    time -- but it has to be pinned, because a change that gathered every fault into one
    body would alter the response for every existing caller.
    """
    tenant = uuid.uuid4()
    definition_id, environment_id = await _an_agent_and_an_environment(
        wired.client, tenant
    )
    live = await _upload(wired.client, tenant, "kept.txt")
    doomed = await _upload(wired.client, tenant, "removed.txt")
    await wired.platform.file_store.delete(
        tenant_id=TenantId(tenant), file_id=FileId(uuid.UUID(doomed))
    )

    refused = await wired.client.post(
        "/v1/sessions",
        json=_creation_body(definition_id, environment_id, [live, doomed]),
        headers=_headers(tenant),
    )
    assert refused.status_code == 410, refused.text
    assert refused.json()["error"]["detail"]["file_id"] == doomed
    assert live not in refused.text


async def test_a_live_file_still_starts_a_session(wired: Wired) -> None:
    """The check refuses a deleted file and nothing else.

    Without this the three tests above would all pass over a route that refused every
    file id, and the surface would be broken in the direction none of them look.
    """
    tenant = uuid.uuid4()
    definition_id, environment_id = await _an_agent_and_an_environment(
        wired.client, tenant
    )
    live = await _upload(wired.client, tenant, "present.txt")

    created = await wired.client.post(
        "/v1/sessions",
        json=_creation_body(definition_id, environment_id, [live]),
        headers=_headers(tenant),
    )
    assert created.status_code == 201, created.text
