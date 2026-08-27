"""Uploading before any Session exists, downloading byte-for-byte, refusing the rest.

The body used throughout carries NUL bytes, a high byte and a UTF-8 sequence, because
"byte-for-byte" is only tested by bytes that a text-shaped code path would damage.
"""

import pathlib
from collections.abc import AsyncIterator, Mapping, Sequence
from urllib.parse import unquote
from uuid import UUID, uuid4

import aioboto3  # type: ignore[import-untyped]
import httpx
import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.s3.uploaded_file import S3UploadedFiles
from managed_agent.composition import Platform, build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import files
from managed_agent.control.files.store import (
    DEFAULT_MAX_UPLOAD_BYTES,
    MAX_CONFIGURABLE_UPLOAD_BYTES,
    MAX_UPLOAD_ENV_VAR,
    UPLOAD_BUCKET_ENV_VAR,
    FileId,
    FileStore,
    FileWindow,
    InvalidUploadFilename,
    NoUploadBucket,
    UploadedFile,
    UploadedFileCorrupt,
    UploadedFileNotFound,
    UploadedFileStorage,
    UploadedFileVanished,
    UploadSizeLimit,
    UploadStorageUnconfigured,
    UploadTooLarge,
    content_digest,
    new_file_id,
    parse_upload_filename,
    read_within_limit,
    unconfigured_file_store,
    upload_limit_from_env,
    upload_object_key,
)
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import DefinitionId, Seq, SessionId, TenantId
from managed_agent.core.ports import EventRecord, Resolution, SessionListing
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord

AWKWARD_BODY = b"\x00\xff head\xe2\x80\x94tail \x00 end"


@pytest.mark.parametrize("size", [1, 4096, MAX_CONFIGURABLE_UPLOAD_BYTES])
def test_a_limit_inside_the_permitted_range_constructs(size: int) -> None:
    assert UploadSizeLimit(size).byte_length == size


@pytest.mark.parametrize("size", [0, -1, MAX_CONFIGURABLE_UPLOAD_BYTES + 1])
def test_a_limit_outside_the_permitted_range_is_refused(size: int) -> None:
    """A limit above what one PUT can create would admit an upload the object store
    refuses at the end of the transfer, which is the worst moment to find out."""
    with pytest.raises(ValueError):
        UploadSizeLimit(size)


def test_a_limit_admits_exactly_up_to_itself() -> None:
    limit = UploadSizeLimit(64)

    assert limit.admits(0)
    assert limit.admits(64)
    assert not limit.admits(65)


def test_an_unset_variable_gives_the_default_limit() -> None:
    assert upload_limit_from_env({}).byte_length == DEFAULT_MAX_UPLOAD_BYTES


def test_a_configured_limit_is_read_from_the_environment() -> None:
    assert upload_limit_from_env({MAX_UPLOAD_ENV_VAR: "4096"}).byte_length == 4096


@pytest.mark.parametrize("raw", ["nonsense", "0", "-1", ""])
def test_a_present_but_unusable_limit_raises_rather_than_falling_back(raw: str) -> None:
    """Falling back would run a deployment that meant to cap uploads at one mebibyte at
    the default instead, and nothing would say so until a bill arrived."""
    with pytest.raises(ValueError) as refused:
        upload_limit_from_env({MAX_UPLOAD_ENV_VAR: raw})

    assert MAX_UPLOAD_ENV_VAR in str(refused.value)


@pytest.mark.parametrize("name", ["notes.txt", "a b ünï.txt", "日本語.txt", "x" * 255])
def test_a_name_that_survives_a_path_and_a_header_is_accepted(name: str) -> None:
    """A non-ASCII name is deliberately accepted: a platform that cannot take a name in
    a non-Latin script is broken, not safe. The download header carries it instead."""
    assert parse_upload_filename(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "x" * 256,
        ".",
        "..",
        "a/b.txt",
        "a\\b.txt",
        'a"b.txt',
        "a\nb.txt",
        "a\x00b.txt",
        "a\x7fb.txt",
    ],
)
def test_a_name_that_would_not_survive_is_refused(name: str) -> None:
    with pytest.raises(InvalidUploadFilename):
        parse_upload_filename(name)


def test_a_digest_is_lower_hex_and_changes_with_one_flipped_bit() -> None:
    digest = content_digest(AWKWARD_BODY)

    assert len(digest) == 64
    assert digest == digest.lower()
    assert digest != content_digest(AWKWARD_BODY[:-1] + b"\x01")


def test_two_tenants_over_one_file_id_address_two_objects() -> None:
    """The tenant is a path segment, so a key built for one tenant can never address
    another's object even if the row were wrong."""
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    file_id = new_file_id()

    ours = upload_object_key(mine, file_id)
    yours = upload_object_key(theirs, file_id)

    assert ours != yours
    assert ours.startswith("uploads/")
    assert yours.startswith("uploads/")
    assert ours.replace(str(mine), str(theirs)) == yours


def a_record(tenant: TenantId | None = None) -> UploadedFile:
    """One well-formed record over AWKWARD_BODY, for the cases that need a valid one."""
    return UploadedFile(
        id=new_file_id(),
        tenant_id=tenant if tenant is not None else TenantId(uuid4()),
        filename="notes.txt",
        media_type="text/plain",
        byte_length=len(AWKWARD_BODY),
        content_sha256=content_digest(AWKWARD_BODY),
    )


@pytest.mark.parametrize(
    "override",
    [
        {"unknown": "field"},
        {"content_sha256": content_digest(b"x").upper()},
        {"byte_length": -1},
        {"media_type": ""},
        {"filename": "a/b.txt"},
    ],
)
def test_a_record_the_store_could_not_honour_is_refused(
    override: dict[str, object],
) -> None:
    """Parsed from a mapping rather than constructed, because two of these cases —
    an unknown field and a name that is really a path — are refusals the constructor's
    own type signature already forbids, so only the parse can be asked about them."""
    with pytest.raises(ValueError):
        UploadedFile.model_validate({**a_record().model_dump(), **override})


def test_a_record_cannot_be_rewritten_after_it_is_made() -> None:
    """The bytes behind a row cannot change, so neither may the length or the hash
    recorded for them: a download that disagrees is a fault, not a new version."""
    record = a_record()

    with pytest.raises(ValueError):
        record.byte_length = 0


def test_a_records_object_key_is_the_one_the_key_function_builds() -> None:
    record = a_record()

    assert record.object_key == upload_object_key(record.tenant_id, record.id)


class FakeStorage:
    """One dict of objects and one of rows, keyed the way the real store keys them."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.rows: dict[FileId, UploadedFile] = {}
        self.tombstones: set[FileId] = set()
        self.calls: list[str] = []

    async def write(self, key: str, body: bytes) -> None:
        self.calls.append("write")
        self.objects[key] = body

    async def read_bytes(self, key: str) -> bytes | None:
        return self.objects.get(key)

    async def record(self, file: UploadedFile) -> None:
        self.calls.append("record")
        self.rows[file.id] = file

    async def lookup(self, tenant_id: TenantId, file_id: FileId) -> UploadedFile | None:
        found = self.rows.get(file_id)
        return found if found is not None and found.tenant_id == tenant_id else None

    async def deletion_recorded(self, file_id: FileId) -> bool:
        """Answered for real, because the download route asks it on every request.

        Nothing here deletes, so it is always false -- and that is the point of
        answering it rather than raising: every download below goes through the check,
        so the check being on the path is exercised by the whole file.
        """
        return file_id in self.tombstones

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


_PORT: UploadedFileStorage = FakeStorage()
"""Graded against the port by mypy --strict rather than by a runtime check: the Protocol
is not runtime_checkable, and isinstance against one that was would compare method names
and not signatures, which is the half that actually drifts."""

_UNCONFIGURED_PORT: UploadedFileStorage = NoUploadBucket()
"""The production stand-in, graded against the same port by the same means."""


async def test_the_fake_answers_the_way_the_real_storage_must() -> None:
    storage = FakeStorage()
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    file_id = new_file_id()
    record = UploadedFile(
        id=file_id,
        tenant_id=mine,
        filename="notes.txt",
        media_type="text/plain",
        byte_length=len(AWKWARD_BODY),
        content_sha256=content_digest(AWKWARD_BODY),
    )
    await storage.write(record.object_key, AWKWARD_BODY)
    await storage.record(record)

    assert await storage.read_bytes(record.object_key) == AWKWARD_BODY
    assert await storage.read_bytes(upload_object_key(theirs, file_id)) is None
    assert await storage.lookup(mine, file_id) == record
    assert await storage.lookup(theirs, file_id) is None
    assert await storage.lookup(mine, new_file_id()) is None


async def _yield(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def test_a_body_arriving_in_pieces_is_buffered_whole() -> None:
    assert (
        await read_within_limit(_yield(b"one", b"two", b"three"), UploadSizeLimit(1024))
        == b"onetwothree"
    )


async def test_an_empty_body_buffers_to_no_bytes() -> None:
    assert await read_within_limit(_yield(), UploadSizeLimit(1024)) == b""


async def test_a_body_landing_exactly_on_the_limit_is_admitted() -> None:
    assert await read_within_limit(_yield(b"x" * 64), UploadSizeLimit(64)) == b"x" * 64


async def test_the_read_stops_at_the_first_byte_past_the_limit() -> None:
    """It refuses without pulling the next chunk, which is what bounds the cost of one
    request at the limit plus one chunk however much the client meant to send."""
    pulled = []

    async def counted() -> AsyncIterator[bytes]:
        for index in range(10):
            pulled.append(index)
            yield b"x" * 32

    with pytest.raises(UploadTooLarge) as refused:
        await read_within_limit(counted(), UploadSizeLimit(33))

    assert refused.value.limit_bytes == 33
    assert refused.value.received_at_least_bytes == 64
    assert pulled == [0, 1]


async def test_storing_writes_the_object_before_the_row() -> None:
    """A crash between them then costs storage, never the tenant's file."""
    storage = FakeStorage()
    store = FileStore(storage, UploadSizeLimit(1024))

    record = await store.store(
        tenant_id=TenantId(uuid4()),
        filename="notes.txt",
        media_type="application/pdf",
        chunks=_yield(AWKWARD_BODY),
    )

    assert storage.calls == ["write", "record"]
    assert record.byte_length == len(AWKWARD_BODY)
    assert record.content_sha256 == content_digest(AWKWARD_BODY)
    assert storage.objects[record.object_key] == AWKWARD_BODY


async def test_a_stored_file_comes_back_with_the_octets_it_went_in_with() -> None:
    storage = FakeStorage()
    store = FileStore(storage, UploadSizeLimit(1024))
    tenant = TenantId(uuid4())
    record = await store.store(
        tenant_id=tenant,
        filename="notes.txt",
        media_type="application/pdf",
        chunks=_yield(AWKWARD_BODY),
    )

    fetched, body = await store.fetch(tenant_id=tenant, file_id=record.id)

    assert body == AWKWARD_BODY
    assert fetched == record


async def test_another_tenants_file_is_not_found() -> None:
    storage = FakeStorage()
    store = FileStore(storage, UploadSizeLimit(1024))
    record = await store.store(
        tenant_id=TenantId(uuid4()),
        filename="notes.txt",
        media_type="text/plain",
        chunks=_yield(AWKWARD_BODY),
    )

    with pytest.raises(UploadedFileNotFound):
        await store.fetch(tenant_id=TenantId(uuid4()), file_id=record.id)


async def test_an_identifier_nothing_wrote_is_not_found_the_same_way() -> None:
    store = FileStore(FakeStorage(), UploadSizeLimit(1024))

    with pytest.raises(UploadedFileNotFound):
        await store.fetch(tenant_id=TenantId(uuid4()), file_id=new_file_id())


async def test_a_row_whose_object_went_away_is_a_fault_not_an_empty_file() -> None:
    storage = FakeStorage()
    store = FileStore(storage, UploadSizeLimit(1024))
    tenant = TenantId(uuid4())
    record = await store.store(
        tenant_id=tenant,
        filename="notes.txt",
        media_type="text/plain",
        chunks=_yield(AWKWARD_BODY),
    )
    del storage.objects[record.object_key]

    with pytest.raises(UploadedFileVanished):
        await store.fetch(tenant_id=tenant, file_id=record.id)


async def test_bytes_that_do_not_match_the_recorded_hash_are_refused() -> None:
    """The hash check is what makes byte-for-byte a property the platform enforces
    rather than one it hopes the object store preserved."""
    storage = FakeStorage()
    store = FileStore(storage, UploadSizeLimit(1024))
    tenant = TenantId(uuid4())
    record = await store.store(
        tenant_id=tenant,
        filename="notes.txt",
        media_type="text/plain",
        chunks=_yield(AWKWARD_BODY),
    )
    storage.objects[record.object_key] = b"different bytes entirely"

    with pytest.raises(UploadedFileCorrupt):
        await store.fetch(tenant_id=tenant, file_id=record.id)


async def test_a_process_with_no_bucket_refuses_both_operations_by_name() -> None:
    store = unconfigured_file_store()

    with pytest.raises(UploadStorageUnconfigured) as storing:
        await store.store(
            tenant_id=TenantId(uuid4()),
            filename="notes.txt",
            media_type="text/plain",
            chunks=_yield(b"x"),
        )
    with pytest.raises(UploadStorageUnconfigured) as fetching:
        await store.fetch(tenant_id=TenantId(uuid4()), file_id=new_file_id())

    assert "MAP_OBJECT_BUCKET" in str(storing.value)
    assert "MAP_OBJECT_BUCKET" in str(fetching.value)


_INSERT_ROW = sa.text(
    "INSERT INTO uploaded_file"
    " (id, tenant_id, filename, media_type, byte_length, content_sha256)"
    " VALUES (:id, :tenant, :filename, :media_type, :len, :hex)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
)


def _row(**overrides: object) -> dict[str, object]:
    return {
        "id": uuid4(),
        "tenant": uuid4(),
        "filename": "notes.txt",
        "media_type": "text/plain",
        "len": len(AWKWARD_BODY),
        "hex": content_digest(AWKWARD_BODY),
        **overrides,
    }


@pytest.mark.parametrize(
    "name", ["notes.txt", "a b ünï.txt", "日本語.txt", "held—together.txt"]
)
async def test_the_store_admits_a_name_the_parser_admits(
    engine: AsyncEngine, name: str
) -> None:
    """The check constraint and `_safe_filename` have to agree, or the platform accepts
    a name at one layer that the other refuses. Non-ASCII letters pass at both."""
    async with engine.begin() as conn:
        await conn.execute(_INSERT_ROW, _row(filename=name))


@pytest.mark.parametrize("name", ["a/b.txt", "a\\b.txt", 'a"b.txt', "a\nb.txt", ""])
async def test_the_store_refuses_a_name_the_parser_refuses(
    engine: AsyncEngine, name: str
) -> None:
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_INSERT_ROW, _row(filename=name))


@pytest.mark.parametrize(
    "bad",
    [
        {"hex": content_digest(b"x").upper()},
        {"hex": "not-a-hash"},
        {"len": -1},
    ],
)
async def test_a_row_the_record_type_would_refuse_is_refused_by_the_store_too(
    engine: AsyncEngine, bad: dict[str, object]
) -> None:
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_INSERT_ROW, _row(**bad))


async def test_one_identifier_addresses_one_file_for_the_whole_platform(
    engine: AsyncEngine,
) -> None:
    """The key is the id alone. A second tenant's row under the same id would make the
    identifier ambiguous rather than private; the tenant is enforced in every WHERE."""
    shared = uuid4()
    async with engine.begin() as conn:
        await conn.execute(_INSERT_ROW, _row(id=shared))

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_INSERT_ROW, _row(id=shared))


async def test_two_tenants_each_keep_their_own_row(engine: AsyncEngine) -> None:
    mine, theirs = uuid4(), uuid4()
    async with engine.begin() as conn:
        await conn.execute(_INSERT_ROW, _row(tenant=mine))
        await conn.execute(_INSERT_ROW, _row(tenant=theirs))

    async with engine.connect() as conn:
        found = await conn.execute(
            sa.text(
                "SELECT count(*) FROM uploaded_file WHERE tenant_id IN (:a, :b)"
            ).bindparams(
                sa.bindparam("a", type_=sa.Uuid()), sa.bindparam("b", type_=sa.Uuid())
            ),
            {"a": mine, "b": theirs},
        )
    assert found.scalar_one() == 2


async def test_an_update_is_refused_rather_than_silently_ignored(
    engine: AsyncEngine,
) -> None:
    """The hash a download is checked against must not be movable by a hand-run
    statement. A rule with DO INSTEAD NOTHING would report success to the writer."""
    written = uuid4()
    async with engine.begin() as conn:
        await conn.execute(_INSERT_ROW, _row(id=written))

    with pytest.raises(DBAPIError):
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE uploaded_file SET content_sha256 = :hex WHERE id = :id"
                ).bindparams(sa.bindparam("id", type_=sa.Uuid())),
                {"hex": content_digest(b"other"), "id": written},
            )


def shipped_storage(engine: AsyncEngine) -> UploadedFileStorage:
    """The real adapter, returned under the port's type.

    Annotated as `UploadedFileStorage` rather than as its own class, so `mypy --strict`
    grades the whole of `S3UploadedFiles` — its object half included — against the
    interface `FileStore` actually drives. That is the only grading the bucket half of
    this adapter gets in this slice: `testcontainers.community.minio` imports `minio`,
    which this tree does not install, so `put_object`, `get_object` and the `NoSuchKey`
    arm are structurally checked and never executed. The relational half below is
    exercised for real.

    The bucket name is never dialled by anything these tests call, so any string does.
    """
    return S3UploadedFiles(aioboto3.Session(), "no-bucket-is-reached-here", engine)


@pytest.mark.parametrize("name", ["notes.txt", "a b ünï.txt", "日本語.txt"])
async def test_a_recorded_row_reads_back_as_the_record_that_was_written(
    engine: AsyncEngine, name: str
) -> None:
    storage = shipped_storage(engine)
    tenant = TenantId(uuid4())
    written = UploadedFile(
        id=new_file_id(),
        tenant_id=tenant,
        filename=name,
        media_type="application/pdf",
        byte_length=len(AWKWARD_BODY),
        content_sha256=content_digest(AWKWARD_BODY),
    )

    await storage.record(written)

    assert await storage.lookup(tenant, written.id) == written


async def test_the_tenant_is_in_the_where_clause_and_not_left_to_the_caller(
    engine: AsyncEngine,
) -> None:
    """A filter every caller has to remember is one a caller will eventually forget,
    and that forgetting is a cross-tenant read."""
    storage = shipped_storage(engine)
    written = a_record()
    await storage.record(written)

    assert await storage.lookup(TenantId(uuid4()), written.id) is None


async def test_an_identifier_nothing_recorded_reads_as_absent(
    engine: AsyncEngine,
) -> None:
    storage = shipped_storage(engine)

    assert await storage.lookup(TenantId(uuid4()), new_file_id()) is None


async def test_recording_one_identifier_twice_is_refused_by_the_store(
    engine: AsyncEngine,
) -> None:
    storage = shipped_storage(engine)
    written = a_record()
    await storage.record(written)

    with pytest.raises(IntegrityError):
        await storage.record(written)


class UnusedLog:
    """Satisfies both log ports and is never called: this file tests the file surface.

    Raising rather than returning a harmless value. A case here that reached the Event
    Log would be grading something this file does not grade, and a quiet stub would let
    it do so without saying anything.
    """

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        raise AssertionError("a file test appended to the Event Log")

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]:
        raise AssertionError("a file test read the Event Log")

    async def follow(
        self, session_id: SessionId, after: Seq
    ) -> AsyncIterator[EventRecord]:
        raise AssertionError("a file test followed the Event Log")
        yield  # pragma: no cover - unreachable, and what makes this a generator

    async def retained_floor(self, session_id: SessionId) -> Seq:
        raise AssertionError("a file test asked the Event Log's retained floor")


class UnusedDefinitions:
    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        raise AssertionError("a file test registered a definition")

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        raise AssertionError("a file test resolved a definition")

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        raise AssertionError("a file test listed a definition's versions")

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        raise AssertionError("a file test read a definition revision")

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a file test retired a definition version")


class UnusedTools:
    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        raise AssertionError("a file test registered an MCP server")

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        raise AssertionError("a file test looked up a registered tool")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        raise AssertionError("a file test listed a tenant's tools")


class UnusedSessions:
    async def create(self, record: SessionRecord) -> None:
        raise AssertionError("a file test wrote a Session registry row")

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        raise AssertionError("a file test fetched a Session registry row")

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        raise AssertionError("a file test paged the Session registry")


class UnusedWebhooks:
    async def register(
        self,
        tenant_id: TenantId,
        url: CallbackUrl,
        event_types: frozenset[str],
        secret_ref: str,
    ) -> WebhookRecord:
        raise AssertionError("a file test registered a webhook")

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        raise AssertionError("a file test listed a tenant's webhooks")

    async def delete(self, webhook_id: UUID, tenant_id: TenantId) -> bool:
        raise AssertionError("a file test deleted a webhook")

    async def watching(
        self, tenant_id: TenantId, event_type: str
    ) -> Sequence[WebhookRecord]:
        raise AssertionError("a file test asked what watches a state")


class UnusedEnvironments:
    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("a file test registered an environment")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        raise AssertionError("a file test resolved an environment")


def a_platform(store: FileStore) -> Platform:
    """The real frozen dataclass, with a raising stand-in behind every other port.

    Nine keywords because `Platform` has nine fields and none has a default. The eight
    stand-ins are the roster `tests/control/test_app.py` already carries, copied rather
    than reinvented — there is no shared Platform builder under `tests/control/`, and
    `tests/conftest.py` provides only database fixtures.
    """
    unused = UnusedLog()
    return Platform(
        event_log_append=unused,
        event_log_range=unused,
        definition_registry=UnusedDefinitions(),
        tool_registry=UnusedTools(),
        session_registry=UnusedSessions(),
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironments(),
        turn_dispatch=NoPodTransport(),
        file_store=store,
    )


def build_app(store: FileStore) -> FastAPI:
    """The real app factory over the real router. Nothing stands in for authentication.

    `create_app` and not a hand-built FastAPI: this is what makes the file notice if the
    `include_router` line is dropped, which every test below would otherwise survive by
    mounting the router itself.
    """
    return create_app(a_platform(store))


def client_for(app: FastAPI, tenant: UUID | None = None) -> httpx.AsyncClient:
    """A client that names its tenant on every request, or one that never names it.

    The tenant rides in the header the routes' dependency reads. A client built with
    None is therefore exactly the request that arrives with no tenant at all: there is
    no middleware to leave out and no principal to unset.
    """
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://s",
        headers={TENANT_HEADER: str(tenant)} if tenant is not None else {},
    )


async def test_a_file_uploaded_before_any_session_downloads_byte_for_byte() -> None:
    storage = FakeStorage()
    app = build_app(FileStore(storage, UploadSizeLimit(1024)))
    async with client_for(app, uuid4()) as caller:
        created = await caller.post(
            "/v1/files",
            files={"file": ("notes.txt", AWKWARD_BODY, "application/pdf")},
        )
        assert created.status_code == 201
        body = created.json()
        assert body["content_sha256"] == content_digest(AWKWARD_BODY)
        assert body["byte_length"] == len(AWKWARD_BODY)

        fetched = await caller.get(f"/v1/files/{body['id']}/content")

    assert fetched.status_code == 200
    assert fetched.content == AWKWARD_BODY
    assert fetched.headers["x-content-type-options"] == "nosniff"
    assert fetched.headers["content-disposition"].startswith("attachment;")
    # Bytes first, row second: an interrupted upload costs storage, never the file.
    assert storage.calls == ["write", "record"]


async def test_a_zero_byte_upload_round_trips_as_a_file_and_not_as_an_absence() -> None:
    app = build_app(FileStore(FakeStorage(), UploadSizeLimit(1024)))
    async with client_for(app, uuid4()) as caller:
        created = await caller.post(
            "/v1/files", files={"file": ("empty.bin", b"", "text/plain")}
        )
        assert created.json()["byte_length"] == 0
        assert created.json()["content_sha256"] == content_digest(b"")

        fetched = await caller.get(f"/v1/files/{created.json()['id']}/content")

    assert fetched.status_code == 200
    assert fetched.content == b""


@pytest.mark.parametrize("name", ["notes.txt", "my notes.txt", "日本語.txt"])
async def test_the_name_survives_the_round_trip_whatever_alphabet_it_is_in(
    name: str,
) -> None:
    """The third case raised UnicodeEncodeError out of the app before RFC 6266.

    Header values are latin-1 on the wire and the parse at upload admits any non-control
    character, so this is not a hypothetical: the exception escaped the ASGI app rather
    than becoming a 500, and no assertion about the body was ever reached.
    """
    app = build_app(FileStore(FakeStorage(), UploadSizeLimit(1024)))
    async with client_for(app, uuid4()) as caller:
        created = await caller.post(
            "/v1/files", files={"file": (name, AWKWARD_BODY, "text/plain")}
        )
        assert created.status_code == 201
        fetched = await caller.get(f"/v1/files/{created.json()['id']}/content")

    assert fetched.content == AWKWARD_BODY
    disposition = fetched.headers["content-disposition"]
    encoded = disposition.split("filename*=UTF-8''")[1]
    assert unquote(encoded) == name
    assert (f'filename="{name}"' in disposition) is name.isascii()


async def test_an_upload_over_the_configured_size_is_refused_and_stores_nothing() -> (
    None
):
    storage = FakeStorage()
    app = build_app(FileStore(storage, UploadSizeLimit(64)))
    async with client_for(app, uuid4()) as caller:
        refused = await caller.post(
            "/v1/files", files={"file": ("big.bin", b"x" * 65, "text/plain")}
        )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    envelope = refused.json()
    assert envelope["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert envelope["error"]["detail"]["reason"] == files.REASON_UPLOAD_TOO_LARGE
    assert envelope["error"]["detail"]["max_bytes"] == 64
    assert (storage.objects, storage.rows) == ({}, {})


async def test_a_body_landing_exactly_on_the_limit_is_admitted_by_the_route() -> None:
    app = build_app(FileStore(FakeStorage(), UploadSizeLimit(64)))
    async with client_for(app, uuid4()) as caller:
        created = await caller.post(
            "/v1/files", files={"file": ("exact.bin", b"x" * 64, "text/plain")}
        )

    assert created.status_code == 201
    assert created.json()["byte_length"] == 64


async def test_a_name_that_is_really_a_path_is_refused_and_stores_nothing() -> None:
    storage = FakeStorage()
    app = build_app(FileStore(storage, UploadSizeLimit(1024)))
    async with client_for(app, uuid4()) as caller:
        refused = await caller.post(
            "/v1/files", files={"file": ("a/b.txt", b"x", "text/plain")}
        )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert refused.json()["error"]["detail"]["reason"] == files.REASON_FILENAME_INVALID
    assert (storage.objects, storage.rows) == ({}, {})


async def test_a_foreign_id_is_indistinguishable_from_one_that_never_existed() -> None:
    storage = FakeStorage()
    owner, stranger = uuid4(), uuid4()
    async with client_for(
        build_app(FileStore(storage, UploadSizeLimit(1024))), owner
    ) as theirs:
        created = await theirs.post(
            "/v1/files", files={"file": ("notes.txt", AWKWARD_BODY, "text/plain")}
        )
    file_id = created.json()["id"]

    async with client_for(
        build_app(FileStore(storage, UploadSizeLimit(1024))), stranger
    ) as caller:
        foreign = await caller.get(f"/v1/files/{file_id}/content")
        absent = await caller.get(f"/v1/files/{uuid4()}/content")

    assert (foreign.status_code, foreign.json()["error"]["code"]) == (
        absent.status_code,
        absent.json()["error"]["code"],
    )
    assert foreign.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert foreign.json()["error"]["code"] == ErrorCode.FILE_NOT_FOUND.value


async def test_a_caller_that_names_no_tenant_reaches_no_store() -> None:
    storage = FakeStorage()
    app = build_app(FileStore(storage, UploadSizeLimit(1024)))
    async with client_for(app) as caller:
        response = await caller.post(
            "/v1/files", files={"file": ("notes.txt", b"x", "text/plain")}
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request.tenant_missing"
    assert (storage.objects, storage.rows, storage.calls) == ({}, {}, [])


async def test_a_download_whose_object_changed_underneath_it_is_a_fault() -> None:
    """A wrong-bytes answer would be worse than a refusal: the caller cannot tell."""
    storage = FakeStorage()
    app = build_app(FileStore(storage, UploadSizeLimit(1024)))
    async with client_for(app, uuid4()) as caller:
        created = await caller.post(
            "/v1/files", files={"file": ("notes.txt", AWKWARD_BODY, "text/plain")}
        )
        key = next(iter(storage.objects))
        storage.objects[key] = b"substituted"
        fetched = await caller.get(f"/v1/files/{created.json()['id']}/content")

    assert fetched.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    assert fetched.json()["error"]["code"] == ErrorCode.INTERNAL.value
    assert fetched.json()["error"]["detail"]["reason"] == files.REASON_FILE_UNREADABLE


async def test_a_process_with_no_bucket_refuses_and_names_the_variable() -> None:
    """The wiring `build()` produces when MAP_OBJECT_BUCKET is unset, exercised.

    Asserted through the route rather than against `NoUploadBucket` directly, because
    the property is that the refusal reaches a caller as a published code with a
    readable reason — an unhandled RuntimeError would satisfy any test of the storage
    alone.
    """
    app = build_app(unconfigured_file_store())
    async with client_for(app, uuid4()) as caller:
        refused = await caller.post(
            "/v1/files", files={"file": ("notes.txt", b"x", "text/plain")}
        )

    assert refused.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    assert refused.json()["error"]["code"] == ErrorCode.INTERNAL.value
    assert (
        refused.json()["error"]["detail"]["reason"] == files.REASON_STORAGE_UNCONFIGURED
    )
    assert "MAP_OBJECT_BUCKET" in refused.json()["error"]["message"]


async def test_no_answer_this_surface_gives_names_the_bucket_or_the_object_key() -> (
    None
):
    """The identifier is the only handle a tenant gets. A key or a bucket in a body
    would hand out the storage layout, and a tenant segment with it."""
    storage = FakeStorage()
    app = build_app(FileStore(storage, UploadSizeLimit(64)))
    async with client_for(app, uuid4()) as caller:
        created = await caller.post(
            "/v1/files", files={"file": ("notes.txt", AWKWARD_BODY, "text/plain")}
        )
        refused = await caller.post(
            "/v1/files", files={"file": ("big.bin", b"x" * 65, "text/plain")}
        )

    key = next(iter(storage.objects))
    for body in (created.text, refused.text):
        assert key not in body
        assert "uploads/" not in body
        assert "bucket" not in body.lower()


def test_this_surface_emits_only_published_codes() -> None:
    """The codes this surface reaches for exist, and it added none of them.

    Asserted by name rather than as a whole-set enumeration. A copy of the member list
    here would be a second source of truth for the published contract, and it would go
    red on any later slice that legitimately adds a member, for a reason that has
    nothing to do with files. The exact-membership guard belongs with the enum;
    `tests/control/test_closed_error_set.py` already walks this route file for a code
    invented in place.
    """
    assert STATUS_FOR[ErrorCode.REQUEST_INVALID] == 400
    assert STATUS_FOR[ErrorCode.INTERNAL] == 500
    assert STATUS_FOR[ErrorCode.FILE_NOT_FOUND] == 404
    assert STATUS_FOR[ErrorCode.FILE_DELETED] == 410
    assert STATUS_FOR[ErrorCode.FILE_IN_USE] == 409


def test_every_file_route_is_published_under_the_version_prefix() -> None:
    """All five, and by verb rather than by path alone.

    `/v1/files` and `/v1/files/{file_id}` each carry two operations, so a path check
    would pass with the listing or the delete missing -- which is the mistake worth
    catching, since both were added after the path already existed.
    """
    paths = build_app(unconfigured_file_store()).openapi()["paths"]

    assert set(paths["/v1/files"]) == {"post", "get"}
    assert set(paths["/v1/files/{file_id}"]) == {"get", "delete"}
    assert set(paths["/v1/files/{file_id}/content"]) == {"get"}


async def test_a_process_started_with_no_bucket_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build()` reads the bucket with `.get`, so its absence is not a start-up failure.

    Ten call sites build a Platform with only a database URL, and a required read would
    turn every one of them into a KeyError — which would make one surface's
    configuration a start-up condition for the whole control plane. The file surface
    refuses at its first request instead, naming the variable.

    No connection is opened here: `create_async_engine` builds a pool lazily, and the
    assertion below never reaches the database.
    """
    monkeypatch.delenv(UPLOAD_BUCKET_ENV_VAR, raising=False)
    platform, engine = build("postgresql+asyncpg://nobody@127.0.0.1:1/none")
    try:
        with pytest.raises(UploadStorageUnconfigured) as refused:
            await platform.file_store.fetch(
                tenant_id=TenantId(uuid4()), file_id=new_file_id()
            )
    finally:
        await engine.dispose()

    assert UPLOAD_BUCKET_ENV_VAR in str(refused.value)


def test_the_object_store_client_is_constructed_at_the_composition_root_only() -> None:
    """Invariant I13: one place knows the storage is S3, so swapping it is one edit.

    Read from the source text because the property is about what is *written* in the
    tree — a second module constructing its own client would import and run perfectly.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "managed_agent"
    constructors = sorted(
        str(module.relative_to(root))
        for module in root.rglob("*.py")
        if "S3UploadedFiles(" in module.read_text()
    )

    assert constructors == ["composition.py"], (
        f"S3UploadedFiles is constructed in {constructors}; only the composition root "
        "may name a concrete adapter"
    )
