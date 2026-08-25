"""Listing a tenant's files, and deleting one without losing what named it.

Three routes in one file because they are one decision. A delete writes a tombstone and
erases the object; the listing and the metadata read are what make that decision visible
-- the file goes absent from one and answers 410 from the other. Tested apart, the three
could drift into disagreeing about whether a file exists, which is the defect worth
catching here and the one no single-route test can see.

Two layers, deliberately. What `FileStore` decides -- the write order, the refusals, the
one extra row -- is exercised over a fake, because none of it needs a database. What SQL
decides -- which rows a page holds, which Sessions still hold a file -- is exercised
against real PostgreSQL, because a fake agreeing with a query proves only that the fake
was written from the same idea.
"""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import aioboto3  # type: ignore[import-untyped]
import httpx
import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.s3.uploaded_file import (
    _DELETED,
    _LIVE_SESSIONS_HOLDING,
    _TOMBSTONE,
    S3UploadedFiles,
)
from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.files import (
    DEFAULT_FILE_PAGE_SIZE,
    MAX_FILE_PAGE_SIZE,
    REASON_TWO_ANCHORS,
)
from managed_agent.control.files.store import (
    FileId,
    FileStore,
    FileWindow,
    UploadedFile,
    UploadedFileInUse,
    UploadedFileNotFound,
    UploadedFileStorage,
    UploadSizeLimit,
    new_file_id,
    unconfigured_file_store,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vocabulary import lifecycle


async def _one(body: bytes) -> AsyncIterator[bytes]:
    yield body


class Storage:
    """Objects, rows and tombstones in three containers, keyed as the real store keys.

    `page` mirrors the adapter's ordering rather than inventing one: insertion order is
    upload order, a page reads newest first, and a tombstoned row is still a position
    but never a result. Where the two could drift, the adapter is exercised for real
    further down -- what these tests are about is what `FileStore` decides on top of
    either.

    `erase_raises` is how the half-finished delete is reached. It is a field rather than
    a subclass because the failure is a property of one call in one test, and a second
    class would put the interesting line a scroll away from the assertion about it.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.rows: dict[FileId, UploadedFile] = {}
        self.tombstones: set[FileId] = set()
        self.holders: dict[FileId, int] = {}
        self.calls: list[str] = []
        self.erase_raises: Exception | None = None

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

    async def page(
        self, tenant_id: TenantId, window: FileWindow, limit: int
    ) -> tuple[UploadedFile, ...]:
        everything = reversed(list(self.rows.values()))
        held = [row for row in everything if row.tenant_id == tenant_id]
        anchor = window.before_id if window.walks_backward else window.after_id
        if anchor is not None:
            positions = [row.id for row in held]
            # An anchor this tenant does not hold yields no rows rather than raising,
            # which is what the adapter does: the subquery that reads the anchor's
            # position returns NULL, so the row comparison is NULL and matches nothing.
            # Mirrored here so removing the store's anchor check shows up as the empty
            # page a caller would really see instead of as this fake blowing up.
            if anchor not in positions:
                return ()
            at = positions.index(anchor)
            held = held[:at][::-1] if window.walks_backward else held[at + 1 :]
        if window.scope_id is not None:
            held = [
                row for row in held if row.produced_in_session_id == window.scope_id
            ]
        return tuple(row for row in held if row.id not in self.tombstones)[:limit]

    async def erase(self, key: str) -> None:
        self.calls.append("erase")
        if self.erase_raises is not None:
            raise self.erase_raises
        self.objects.pop(key, None)

    async def record_deletion_unless_held(
        self, tenant_id: TenantId, file_id: FileId
    ) -> int:
        """The adapter's transaction, as its outcome rather than as a transaction.

        What a caller can observe of the real one is exactly this: either the count is
        zero and the tombstone is there, or the count is not and nothing was written. A
        fake that recorded first and then reported a count would let a rolled-back
        delete pass as a written one.
        """
        self.calls.append("record_deletion_unless_held")
        holding = self.holders.get(file_id, 0)
        if holding == 0:
            self.tombstones.add(file_id)
        return holding

    async def deletion_recorded(self, file_id: FileId) -> bool:
        return file_id in self.tombstones


_PORT: UploadedFileStorage = Storage()
"""Graded against the port by mypy --strict rather than by a runtime check: the Protocol
is not runtime_checkable, and an isinstance against one that was would compare method
names and not signatures, which is the half that actually drifts."""


async def stored(
    store: FileStore,
    tenant: TenantId,
    *,
    name: str = "notes.txt",
    body: bytes = b"contents",
    session: SessionId | None = None,
) -> UploadedFile:
    return await store.store(
        tenant_id=tenant,
        filename=name,
        media_type="text/plain",
        chunks=_one(body),
        produced_in_session_id=session,
    )


def a_store(storage: Storage) -> FileStore:
    return FileStore(storage, UploadSizeLimit(4096))


# --------------------------------------------------------------------------------------
# What the store decides
# --------------------------------------------------------------------------------------


async def test_a_page_holds_what_was_asked_for_and_says_when_more_remain() -> None:
    """The extra row is the whole answer to "is there another page".

    Asserted at both ends of the walk, because a limit that were off by one would show
    up as a page that is right and a `has_more` that is wrong -- or the reverse, which
    is worse: a caller told the collection is exhausted while a row is still waiting.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = TenantId(uuid4())
    written = [await stored(store, tenant, name=f"{n}.txt") for n in range(3)]

    first = await store.page(tenant_id=tenant, window=FileWindow(), limit=2)
    whole = await store.page(tenant_id=tenant, window=FileWindow(), limit=3)

    assert [row.id for row in first.files] == [written[2].id, written[1].id]
    assert first.has_more
    assert [row.id for row in whole.files] == [row.id for row in reversed(written)]
    assert not whole.has_more


async def test_a_deleted_file_is_absent_from_the_listing() -> None:
    """The one place a deletion is silent rather than reported.

    The collection answers "what do I have", and a deleted file is not had. A row here
    would make the listing the one surface where a tenant can still see what they asked
    to be rid of.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = TenantId(uuid4())
    kept = await stored(store, tenant, name="kept.txt")
    gone = await stored(store, tenant, name="gone.txt")

    await store.delete(tenant_id=tenant, file_id=gone.id)
    page = await store.page(tenant_id=tenant, window=FileWindow(), limit=10)

    assert [row.id for row in page.files] == [kept.id]


async def test_an_anchor_this_tenant_does_not_hold_is_refused_not_answered_empty() -> (
    None
):
    """An empty page looks like a finished walk, which is the wrong answer to a typo.

    Both cases are asserted because they are the two ways a caller gets here and the
    dangerous one is the second: an id belonging to somebody else must not silently read
    as the end of this tenant's collection.
    """
    storage = Storage()
    store = a_store(storage)
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    await stored(store, mine)
    yours = await stored(store, theirs)

    with pytest.raises(UploadedFileNotFound):
        await store.page(
            tenant_id=mine, window=FileWindow(after_id=new_file_id()), limit=10
        )
    with pytest.raises(UploadedFileNotFound):
        await store.page(tenant_id=mine, window=FileWindow(after_id=yours.id), limit=10)


async def test_a_backward_page_reads_in_the_same_order_as_a_forward_one() -> None:
    """Both directions present newest first; only the walk runs the other way.

    This is the property a caller relies on to page backwards at all. The rows come out
    of the store walking away from the anchor -- oldest first -- and are turned round
    here, so a client rendering a page does not have to know which way it asked.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = TenantId(uuid4())
    written = [await stored(store, tenant, name=f"{n}.txt") for n in range(5)]

    forward = await store.page(
        tenant_id=tenant, window=FileWindow(after_id=written[3].id), limit=2
    )
    backward = await store.page(
        tenant_id=tenant, window=FileWindow(before_id=written[0].id), limit=2
    )

    assert [row.id for row in forward.files] == [written[2].id, written[1].id]
    assert [row.id for row in backward.files] == [written[2].id, written[1].id]
    assert forward.has_more and backward.has_more


async def test_a_window_naming_both_anchors_cannot_be_built() -> None:
    """Refused where the value is made, so no caller of the store can ask for it.

    A page starting in two places has no meaning to answer, and choosing one anchor for
    the caller hands back a page they did not ask for and cannot tell apart from one
    they did.
    """
    with pytest.raises(ValueError):
        FileWindow(after_id=new_file_id(), before_id=new_file_id())


async def test_a_delete_writes_the_tombstone_before_it_erases_the_object() -> None:
    """The order is the safety argument, so the order is what is asserted.

    Erasing first leaves the bytes gone while every read still says the file is live,
    and nothing anywhere recording that a deletion was even attempted -- so no retry
    can finish it. This way round, a failure between the two leaves a file that reads
    as deleted and a retry that completes it.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = TenantId(uuid4())
    written = await stored(store, tenant)
    storage.calls.clear()

    await store.delete(tenant_id=tenant, file_id=written.id)

    assert storage.calls == ["record_deletion_unless_held", "erase"]
    assert storage.objects == {}
    assert written.id in storage.rows


async def test_a_failed_erase_leaves_a_file_that_reads_as_deleted() -> None:
    """The half-finished state, and the retry that finishes it.

    Three claims, and the third is the one that makes the order defensible: the
    tombstone is written, the bytes are still there, and nothing has told the caller the
    delete succeeded -- the exception propagates. The retry absorbs the tombstone
    conflict and erases the object, so the state converges rather than needing an
    operator.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = TenantId(uuid4())
    written = await stored(store, tenant)
    storage.erase_raises = RuntimeError("the bucket refused")

    with pytest.raises(RuntimeError):
        await store.delete(tenant_id=tenant, file_id=written.id)

    assert await store.deletion_recorded(file_id=written.id)
    assert storage.objects != {}

    storage.erase_raises = None
    await store.delete(tenant_id=tenant, file_id=written.id)

    assert storage.objects == {}
    assert await store.deletion_recorded(file_id=written.id)


async def test_a_delete_is_refused_while_a_session_that_has_not_stopped_holds_it() -> (
    None
):
    """Nothing is written when the guard fires, which is what makes it a guard.

    A refusal that had already tombstoned the file would leave the Session running
    against a file the platform reports as deleted -- the refusal would be advice rather
    than a decision.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = TenantId(uuid4())
    written = await stored(store, tenant)
    storage.holders[written.id] = 2

    with pytest.raises(UploadedFileInUse) as refused:
        await store.delete(tenant_id=tenant, file_id=written.id)

    assert refused.value.session_count == 2
    assert not await store.deletion_recorded(file_id=written.id)
    assert storage.objects != {}


async def test_deleting_a_file_nobody_uploaded_is_a_lookup_failure() -> None:
    storage = Storage()
    store = a_store(storage)

    with pytest.raises(UploadedFileNotFound):
        await store.delete(tenant_id=TenantId(uuid4()), file_id=new_file_id())


async def test_another_tenants_file_cannot_be_deleted() -> None:
    """The same refusal as an id that never existed, and for the same reason.

    Two answers here would let a caller walk the identifier space with DELETE and learn
    which ids exist somewhere on the platform -- the read routes are careful about this
    and the delete has to be too, or the probe just moves to the other verb.
    """
    storage = Storage()
    store = a_store(storage)
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    yours = await stored(store, theirs)

    with pytest.raises(UploadedFileNotFound):
        await store.delete(tenant_id=mine, file_id=yours.id)

    assert not await store.deletion_recorded(file_id=yours.id)
    assert storage.objects != {}


# --------------------------------------------------------------------------------------
# What SQL decides, against real PostgreSQL
# --------------------------------------------------------------------------------------

_UPLOAD_ROW = sa.text(
    "INSERT INTO uploaded_file"
    " (id, tenant_id, filename, media_type, byte_length, content_sha256,"
    "  produced_in_session_id, uploaded_at)"
    " VALUES (:id, :tenant, :filename, 'text/plain', 3, :hex, :produced, :at)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("produced", type_=sa.Uuid()),
    # Declared and parsed, not passed as text. asyncpg refuses a string for a
    # `timestamptz` outright -- `expected a datetime.date or datetime.datetime instance`
    # -- so the readable ISO spelling the tests below use has to become a datetime here.
    sa.bindparam("at", type_=sa.TIMESTAMP(timezone=True)),
)

_SESSION_ROW = sa.text(
    "INSERT INTO session"
    " (id, tenant_id, definition_id, definition_revision, grant_tools, scope,"
    "  budget_minor_units, budget_currency, retention_days)"
    " VALUES (:id, :tenant, :definition, 'rev-1', :grant, :scope, 100, 'USD', 7)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("definition", type_=sa.Uuid()),
    sa.bindparam("grant", type_=sa.JSON()),
    sa.bindparam("scope", type_=sa.JSON()),
)

_EVENT_ROW = sa.text(
    "INSERT INTO event_log (session_id, seq, type, payload)"
    " VALUES (:session_id, :seq, :type, :payload)"
).bindparams(
    sa.bindparam("session_id", type_=sa.Uuid()),
    sa.bindparam("payload", type_=sa.JSON()),
)

_TOMBSTONE_ROW = sa.text(
    "INSERT INTO uploaded_file_deletion (file_id) VALUES (:file_id)"
).bindparams(sa.bindparam("file_id", type_=sa.Uuid()))

_HASH = "0" * 64


def shipped(engine: AsyncEngine) -> UploadedFileStorage:
    """The real adapter under the port's type, over a bucket nothing here dials.

    Every test below drives only its relational half, so the bucket name is a string
    and no object call is made -- the same trade `test_file_upload_download.py` records
    for its own use of this adapter. Annotated as the port so `mypy --strict` grades
    the whole class against the interface `FileStore` actually drives.
    """
    return S3UploadedFiles(aioboto3.Session(), "bucket-nothing-dials", engine)


async def uploaded(
    engine: AsyncEngine,
    tenant: TenantId,
    *,
    at: str,
    name: str = "notes.txt",
    produced_in: SessionId | None = None,
) -> FileId:
    """One row, with its upload moment chosen rather than taken from the clock.

    The listing orders by `(uploaded_at, id)`, so a test about ordering has to be able
    to set the timestamp: rows written in one transaction share `now()` to the
    microsecond and would order by id alone, which is exactly the tie the second key
    exists for and not the case being tested.
    """
    file_id = new_file_id()
    async with engine.begin() as conn:
        await conn.execute(
            _UPLOAD_ROW,
            {
                "id": file_id,
                "tenant": tenant,
                "filename": name,
                "hex": _HASH,
                "produced": produced_in,
                "at": datetime.fromisoformat(at),
            },
        )
    return file_id


async def a_session(
    engine: AsyncEngine, tenant: TenantId, *, holding: tuple[FileId, ...], stopped: bool
) -> SessionId:
    """A Session that names these files, and either has stopped or has not.

    Written as the row plus the log entries rather than through the API, because what
    is under test is the query that reads them -- a creation going through `POST
    /v1/sessions` would drag its whole validation path into a test about a count.
    """
    session_id = SessionId(uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            _SESSION_ROW,
            {
                "id": session_id,
                "tenant": tenant,
                "definition": uuid4(),
                "grant": [],
                "scope": {},
            },
        )
        await conn.execute(
            _EVENT_ROW,
            {
                "session_id": session_id,
                "seq": 1,
                "type": lifecycle.SESSION_CREATED,
                "payload": {"file_ids": [str(one) for one in holding]},
            },
        )
        if stopped:
            await conn.execute(
                _EVENT_ROW,
                {
                    "session_id": session_id,
                    "seq": 2,
                    "type": lifecycle.SESSION_STOPPED,
                    "payload": {},
                },
            )
    return session_id


async def tombstoned(engine: AsyncEngine, file_id: FileId) -> None:
    async with engine.begin() as conn:
        await conn.execute(_TOMBSTONE_ROW, {"file_id": file_id})


async def test_the_listing_reads_newest_first_and_only_this_tenants(
    engine: AsyncEngine,
) -> None:
    """Both halves of the WHERE clause, in the query that has to get both right.

    The tenant is a term in the statement rather than a filter applied afterwards, so
    another tenant's row is absent rather than fetched and dropped. Asserted alongside
    the order because the two share one statement and a rewrite of either touches it.
    """
    storage = shipped(engine)
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    older = await uploaded(engine, mine, at="2026-01-01T00:00:00Z", name="older.txt")
    newer = await uploaded(engine, mine, at="2026-01-02T00:00:00Z", name="newer.txt")
    await uploaded(engine, theirs, at="2026-01-03T00:00:00Z", name="not-mine.txt")

    page = await storage.page(mine, FileWindow(), 10)

    assert [row.id for row in page] == [newer, older]


async def test_the_listing_leaves_out_a_file_with_a_tombstone(
    engine: AsyncEngine,
) -> None:
    """The anti-join, which is what makes a deleted file absent rather than filtered.

    In SQL rather than in Python because the limit is applied in SQL: a page that
    fetched twenty rows and then dropped the deleted ones would hand back short pages
    and a `has_more` computed from the wrong count.
    """
    storage = shipped(engine)
    tenant = TenantId(uuid4())
    kept = await uploaded(engine, tenant, at="2026-01-01T00:00:00Z", name="kept.txt")
    gone = await uploaded(engine, tenant, at="2026-01-02T00:00:00Z", name="gone.txt")
    await tombstoned(engine, gone)

    page = await storage.page(tenant, FileWindow(), 10)

    assert [row.id for row in page] == [kept]
    assert await storage.deletion_recorded(gone)
    assert not await storage.deletion_recorded(kept)


async def test_the_scope_filter_answers_with_one_sessions_output(
    engine: AsyncEngine,
) -> None:
    """`scope_id` is a predicate on the one column that tells the two kinds apart.

    A tenant's upload has no scope, so it is absent from a scoped page; that is the
    whole reason `produced_in_session_id` is recorded at insert time rather than
    derived later.
    """
    storage = shipped(engine)
    tenant = TenantId(uuid4())
    session = SessionId(uuid4())
    await uploaded(engine, tenant, at="2026-01-01T00:00:00Z", name="sent.txt")
    made = await uploaded(
        engine, tenant, at="2026-01-02T00:00:00Z", name="made.txt", produced_in=session
    )

    scoped = await storage.page(tenant, FileWindow(scope_id=session), 10)
    everything = await storage.page(tenant, FileWindow(), 10)

    assert [row.id for row in scoped] == [made]
    assert len(everything) == 2
    assert scoped[0].produced_in_session_id == session


async def test_an_anchor_positions_a_page_in_both_directions(
    engine: AsyncEngine,
) -> None:
    """The row comparison against `(uploaded_at, id)`, walked each way from one anchor.

    The backward page comes back oldest first, because it walks away from its anchor --
    that is the contract `FileStore.page` reverses, and asserting it unreversed here is
    what would catch the adapter and the store disagreeing about which end the extra row
    is at.
    """
    storage = shipped(engine)
    tenant = TenantId(uuid4())
    written = [
        await uploaded(engine, tenant, at=f"2026-01-0{n}T00:00:00Z", name=f"{n}.txt")
        for n in range(1, 6)
    ]

    forward = await storage.page(tenant, FileWindow(after_id=written[3]), 2)
    backward = await storage.page(tenant, FileWindow(before_id=written[1]), 2)

    assert [row.id for row in forward] == [written[2], written[1]]
    assert [row.id for row in backward] == [written[2], written[3]]


async def test_a_deleted_files_row_is_still_a_position_to_page_from(
    engine: AsyncEngine,
) -> None:
    """A walk begun before a deletion carries on across it.
    The row survives the delete, so it still marks a place in the order even though it
    can never be a result. Refusing the anchor instead would break exactly the caller
    who is doing the right thing -- paging steadily through a collection that changed
    underneath them.
    them.
    """
    storage = shipped(engine)
    tenant = TenantId(uuid4())
    old = await uploaded(engine, tenant, at="2026-01-01T00:00:00Z", name="old.txt")
    middle = await uploaded(engine, tenant, at="2026-01-02T00:00:00Z", name="mid.txt")
    await uploaded(engine, tenant, at="2026-01-03T00:00:00Z", name="new.txt")
    await tombstoned(engine, middle)

    page = await storage.page(tenant, FileWindow(after_id=middle), 10)

    assert [row.id for row in page] == [old]


async def test_a_session_that_has_not_stopped_counts_as_holding_its_files(
    engine: AsyncEngine,
) -> None:
    """The count the delete guard turns into a 409, read out of the Event Log.

    There is no join table to count: a Session's attached ids live only in its
    `session.created` payload, so the query reads the payload and asks whether a
    `session.stopped` ever followed.
    """
    storage = shipped(engine)
    tenant = TenantId(uuid4())
    held = await uploaded(engine, tenant, at="2026-01-01T00:00:00Z")
    loose = await uploaded(engine, tenant, at="2026-01-01T00:00:01Z")
    await a_session(engine, tenant, holding=(held,), stopped=False)
    await a_session(engine, tenant, holding=(held,), stopped=False)

    assert await storage.record_deletion_unless_held(tenant, held) == 2
    assert not await storage.deletion_recorded(held)
    assert await storage.record_deletion_unless_held(tenant, loose) == 0
    assert await storage.deletion_recorded(loose)


async def test_a_stopped_session_no_longer_holds_its_files(
    engine: AsyncEngine,
) -> None:
    """A terminated Session does not hold the delete, and the tombstone is why.

    Its history keeps the id and the tombstone is what tells that history what happened,
    so there is nothing left for the guard to protect -- no pod will be placed for it
    again.
    """
    storage = shipped(engine)
    tenant = TenantId(uuid4())
    held = await uploaded(engine, tenant, at="2026-01-01T00:00:00Z")
    await a_session(engine, tenant, holding=(held,), stopped=True)

    assert await storage.record_deletion_unless_held(tenant, held) == 0
    assert await storage.deletion_recorded(held)


async def test_another_tenants_session_naming_the_id_is_not_counted(
    engine: AsyncEngine,
) -> None:
    """`event_log` carries no tenant, so the count joins `session` to get one.
    A Session naming another tenant's file cannot be created, so this is a second lock
    on a shut door. It is asserted anyway because it is what keeps the count from
    depending on that: a delete must not be refused by somebody else's Session.
    """
    storage = shipped(engine)
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    held = await uploaded(engine, mine, at="2026-01-01T00:00:00Z")
    await a_session(engine, theirs, holding=(held,), stopped=False)

    assert await storage.record_deletion_unless_held(mine, held) == 0
    assert await storage.deletion_recorded(held)


async def test_the_store_over_the_real_adapter_pages_and_reports_deletion(
    engine: AsyncEngine,
) -> None:
    """The two halves joined: `FileStore`'s decisions over the real queries.
    The fake above and the adapter above are each exercised alone, which leaves one gap
    between them -- the extra row, the reversal and the anchor lookup running against
    SQL rather than against a dict. This is that gap, and it is why the numbers here
    are chosen to make `has_more` true.
    """
    store = FileStore(shipped(engine), UploadSizeLimit(4096))
    tenant = TenantId(uuid4())
    written = [
        await uploaded(engine, tenant, at=f"2026-02-0{n}T00:00:00Z", name=f"{n}.txt")
        for n in range(1, 4)
    ]

    first = await store.page(tenant_id=tenant, window=FileWindow(), limit=2)
    backward = await store.page(
        tenant_id=tenant, window=FileWindow(before_id=written[0]), limit=2
    )

    assert [row.id for row in first.files] == [written[2], written[1]]
    assert first.has_more
    assert [row.id for row in backward.files] == [written[2], written[1]]


_TOMBSTONE_COUNT = sa.text(
    "SELECT count(*) FROM uploaded_file_deletion WHERE file_id = :file_id"
).bindparams(sa.bindparam("file_id", type_=sa.Uuid()))

_COUNT_ARGS = {
    "created_type": lifecycle.SESSION_CREATED,
    "stopped_type": lifecycle.SESSION_STOPPED,
}


async def test_a_refused_delete_over_the_real_store_writes_no_tombstone(
    engine: AsyncEngine,
) -> None:
    """The rollback, proven through the real store rather than through the fake.

    Three claims, and the second is the one the transaction buys: the count comes back,
    no tombstone survives, and the file is still in the listing. A guard that counted
    and wrote in separate transactions would leave the row behind on the refusal, so
    every later read would report a file the platform never agreed to delete.

    The erase is never reached, because `delete` raises first. That the bytes survive a
    refusal is asserted over the fake, where an object store exists to check.
    """
    store = FileStore(shipped(engine), UploadSizeLimit(4096))
    tenant = TenantId(uuid4())
    held = await uploaded(engine, tenant, at="2026-03-01T00:00:00Z")
    await a_session(engine, tenant, holding=(held,), stopped=False)

    with pytest.raises(UploadedFileInUse) as refused:
        await store.delete(tenant_id=tenant, file_id=held)

    assert refused.value.session_count == 1
    assert not await store.deletion_recorded(file_id=held)
    page = await store.page(tenant_id=tenant, window=FileWindow(), limit=10)
    assert [row.id for row in page.files] == [held]


async def test_a_create_committing_inside_the_transaction_is_still_counted(
    engine: AsyncEngine,
) -> None:
    """The interleaving the inversion closes, forced rather than hoped for.

    Driven through the adapter's own statements on two connections, with the
    transaction boundaries under the test's control, because the adapter method is
    atomic by design and cannot be interrupted from outside. Importing the statements
    rather than retyping the SQL is what keeps this a test of the real query: a rewrite
    of either statement is exercised here.

    The order is the one that used to lose. The tombstone is written and not yet
    committed, so a session creation checking it -- on its own connection, which is how
    `sessions.py` checks -- does not see it. That create then commits. Under READ
    COMMITTED the count statement takes a fresh snapshot, so it sees the create and the
    transaction rolls back. Before the inversion the count had already been taken by
    this point and the delete would have gone through."""
    tenant = TenantId(uuid4())
    held = await uploaded(engine, tenant, at="2026-03-02T00:00:00Z")

    async with engine.connect() as deleting:
        transaction = await deleting.begin()
        await deleting.execute(_TOMBSTONE, {"file_id": str(held)})
        async with engine.connect() as creating:
            unseen = await creating.scalar(_DELETED, {"file_id": str(held)})
        await a_session(engine, tenant, holding=(held,), stopped=False)
        counted = await deleting.scalar(
            _LIVE_SESSIONS_HOLDING,
            {"tenant": str(tenant), "file_id": str(held), **_COUNT_ARGS},
        )
        await transaction.rollback()

    assert unseen is None
    assert counted == 1
    async with engine.connect() as conn:
        assert await conn.scalar(_TOMBSTONE_COUNT, {"file_id": held}) == 0


async def test_a_create_committing_after_the_count_is_the_race_that_remains(
    engine: AsyncEngine,
) -> None:
    """The interleaving that remains, reproduced on purpose.

    This asserts a defect is reachable, which is deliberate. A race nobody can see is a
    race nobody closes, and the alternative to an executable record of it is a
    paragraph in a docstring that stops being true without anything failing.

    The losing order: the tombstone is written, the count is taken and sees nothing,
    the create commits, and only then does the tombstone commit. Neither transaction
    ever saw the other, because under READ COMMITTED they conflict on nothing -- this
    one writes the tombstone and reads the log, the create writes the log and reads the
    tombstone. That is write skew, and snapshot isolation does not prevent it.

    It cannot be closed from this side. Session creation is three transactions -- the
    tombstone check, the event append, the registry write -- so no lock taken here is
    still held when the create commits. Closing it needs the create made atomic, or
    SERIALIZABLE on both sides.

    The failure it produces: a Session that holds a file whose bytes are gone, which
    dies at its next pod placement. That is the symptom the whole in-use guard exists
    to prevent, arriving through the one door left open."""
    tenant = TenantId(uuid4())
    held = await uploaded(engine, tenant, at="2026-03-03T00:00:00Z")

    async with engine.connect() as deleting:
        transaction = await deleting.begin()
        await deleting.execute(_TOMBSTONE, {"file_id": str(held)})
        counted = await deleting.scalar(
            _LIVE_SESSIONS_HOLDING,
            {"tenant": str(tenant), "file_id": str(held), **_COUNT_ARGS},
        )
        await a_session(engine, tenant, holding=(held,), stopped=False)
        await transaction.commit()

    assert counted == 0
    async with engine.connect() as conn:
        assert await conn.scalar(_TOMBSTONE_COUNT, {"file_id": held}) == 1
        holders = await conn.scalar(
            _LIVE_SESSIONS_HOLDING,
            {"tenant": str(tenant), "file_id": str(held), **_COUNT_ARGS},
        )

    assert holders == 1, (
        "a live Session holds a file this platform has recorded as deleted. If this "
        "assertion fails, the race described on FileStore.delete has been closed -- "
        "invert this test to assert the refusal instead of deleting it, so the "
        "guarantee stays covered."
    )


async def test_the_adapter_writes_the_tombstone_before_it_counts(
    engine: AsyncEngine,
) -> None:
    """The order inside the transaction, read off the statements actually sent.

    Instrumentation rather than a black-box assertion, and the reason is the same one
    that makes this race hard: no observation from outside a committed transaction can
    tell the two orders apart. The test above proves that writing first is the order
    which catches an interleaved create; this one proves the adapter uses that order.
    Without it, reverting to count-first passes every other test in this file --
    measured, not supposed.

    Nothing is stubbed. The listener records the real SQL on its way to real
    PostgreSQL, and the statements are matched by their opening words rather than
    compared whole, so rewording either query does not fail this.
    """
    sent: list[str] = []

    def record(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        sent.append(" ".join(statement.split()))

    tenant = TenantId(uuid4())
    held = await uploaded(engine, tenant, at="2026-03-04T00:00:00Z")
    sa.event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        await shipped(engine).record_deletion_unless_held(tenant, held)
    finally:
        sa.event.remove(engine.sync_engine, "before_cursor_execute", record)

    guarding = [
        statement
        for statement in sent
        if "uploaded_file_deletion" in statement or "event_log" in statement
    ]

    assert guarding[0].startswith("INSERT INTO uploaded_file_deletion")
    assert guarding[1].startswith("SELECT count(*) FROM event_log")


# --------------------------------------------------------------------------------------
# What a caller sees
# --------------------------------------------------------------------------------------


class Unused:
    """Every port but the file store raises. These routes reach no other one.
    One class with `__getattr__` rather than a roster of named stand-ins, which is the
    shape `test_the_schema_matches_the_answers.py` uses: what matters is that a reach
    for anything else fails loudly and names what was reached for.
    """

    def __getattr__(self, name: str) -> Any:
        async def refuse(*args: object, **kwargs: object) -> Any:
            raise AssertionError(f"a file route reached {name}")

        return refuse


def build_app(store: FileStore) -> FastAPI:
    """The real app factory over the real router. Nothing stands in for authentication.
    `create_app` and not a hand-built FastAPI, so this file notices if the
    `include_router` line is dropped -- which every test below would otherwise survive
    by mounting the router itself.
    """
    unused = Unused()
    return create_app(
        Platform(
            event_log_append=unused,
            event_log_range=unused,
            definition_registry=unused,
            tool_registry=unused,
            session_registry=unused,
            webhooks=unused,
            environment_store=unused,
            turn_dispatch=unused,
            file_store=store,
        )
    )


def client_for(app: FastAPI, tenant: UUID) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://s",
        headers={TENANT_HEADER: str(tenant)},
    )


async def test_the_listing_answers_data_first_id_last_id_and_has_more() -> None:
    """The published envelope, by name.
    These four names are the reference's, not this repository's -- every other
    collection here pages by an opaque `page` cursor and a `next_page`. A client
    generated from the documented surface reads these, so the field names are the
    contract and are asserted as such.
    as such.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = uuid4()
    written = [await stored(store, TenantId(tenant), name=f"{n}.txt") for n in range(3)]

    async with client_for(build_app(store), tenant) as caller:
        page = await caller.get("/v1/files", params={"limit": 2})

    body = page.json()
    assert page.status_code == 200
    assert [row["id"] for row in body["data"]] == [
        str(written[2].id),
        str(written[1].id),
    ]
    assert body["first_id"] == str(written[2].id)
    assert body["last_id"] == str(written[1].id)
    assert body["has_more"] is True


async def test_an_empty_listing_carries_null_anchors_rather_than_none() -> None:
    """A caller that walked to the end reads the same two fields it read all along.

    An absent field leaves a consumer's own default to say what happened; a null says
    there was nothing to anchor on, which is the true answer.
    """
    store = a_store(Storage())

    async with client_for(build_app(store), uuid4()) as caller:
        page = await caller.get("/v1/files")

    assert page.json() == {
        "data": [],
        "first_id": None,
        "last_id": None,
        "has_more": False,
    }


async def test_a_produced_files_metadata_carries_the_session_that_made_it() -> None:
    """`scope` is how a caller tells a file the agent wrote from one they sent.
    Both rows are read, because the informative half is the null: an uploaded file has
    no scope, and a caller reading a mixed page needs the two to be distinguishable
    without knowing which is which in advance.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = uuid4()
    session = SessionId(uuid4())
    sent = await stored(store, TenantId(tenant), name="sent.txt")
    made = await stored(store, TenantId(tenant), name="made.txt", session=session)

    async with client_for(build_app(store), tenant) as caller:
        one = await caller.get(f"/v1/files/{made.id}")
        other = await caller.get(f"/v1/files/{sent.id}")

    assert one.json()["scope"] == {"id": str(session), "type": "session"}
    assert one.json()["type"] == "file"
    assert one.json()["downloadable"] is True
    assert other.json()["scope"] is None


async def test_metadata_reports_what_was_recorded_without_reading_the_object() -> None:
    """The point of the route: a hundred-mebibyte upload's filename for no transfer.

    Asserted by emptying the object store and reading the metadata anyway. If this route
    ever grew a fetch, this is the test that would fail rather than a timeout in
    production.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = uuid4()
    written = await stored(store, TenantId(tenant), body=b"some bytes")
    storage.objects.clear()

    async with client_for(build_app(store), tenant) as caller:
        read = await caller.get(f"/v1/files/{written.id}")

    assert read.status_code == 200
    assert read.json()["filename"] == "notes.txt"
    assert read.json()["byte_length"] == len(b"some bytes")
    assert read.json()["content_sha256"] == written.content_sha256


async def test_deleting_answers_the_published_shape_and_repeats_it() -> None:
    """200 with a body, and the same body twice.
    A 204 may carry no body at all, so the status follows the shape rather than the
    verb. The repeat is the property a caller retrying a timeout depends on: they must
    not be told the file was never there.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = uuid4()
    written = await stored(store, TenantId(tenant))

    async with client_for(build_app(store), tenant) as caller:
        first = await caller.delete(f"/v1/files/{written.id}")
        again = await caller.delete(f"/v1/files/{written.id}")

    assert first.status_code == 200
    assert first.json() == {"id": str(written.id), "type": "file_deleted"}
    assert again.status_code == 200
    assert again.json() == first.json()


async def test_a_deleted_file_answers_410_and_leaves_the_listing() -> None:
    """The three surfaces after a delete, which is the whole point of the tombstone.
    410 and not 404, because the identifier is not wrong: it named a file and the file
    was deliberately removed. A tenant who deleted a file to honour a deletion request
    needs the platform to say so -- a 404 leaves them unable to tell it from a typo in
    the id they kept.
    they kept.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = uuid4()
    written = await stored(store, TenantId(tenant))

    async with client_for(build_app(store), tenant) as caller:
        await caller.delete(f"/v1/files/{written.id}")
        metadata = await caller.get(f"/v1/files/{written.id}")
        content = await caller.get(f"/v1/files/{written.id}/content")
        listing = await caller.get("/v1/files")

    for gone in (metadata, content):
        assert gone.status_code == STATUS_FOR[ErrorCode.FILE_DELETED] == 410
        assert gone.json()["error"]["code"] == ErrorCode.FILE_DELETED.value
        assert gone.json()["error"]["detail"]["file_id"] == str(written.id)
    assert listing.json()["data"] == []


async def test_a_foreign_deleted_file_is_404_and_not_410() -> None:
    """Ownership is proven before the tombstone is read, and the order is the point.
    The reverse order answers 410 for another tenant's deleted file, which tells a
    caller that an id exists somewhere on the platform -- exactly the probe the
    identical not-found bodies exist to prevent. Both reads are checked, because both
    had to get the order right independently.
    """
    storage = Storage()
    store = a_store(storage)
    owner, stranger = uuid4(), uuid4()
    written = await stored(store, TenantId(owner))
    async with client_for(build_app(store), owner) as theirs:
        await theirs.delete(f"/v1/files/{written.id}")

    async with client_for(build_app(store), stranger) as caller:
        metadata = await caller.get(f"/v1/files/{written.id}")
        content = await caller.get(f"/v1/files/{written.id}/content")
        never = await caller.get(f"/v1/files/{uuid4()}")

    for refused in (metadata, content, never):
        assert refused.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND] == 404
        assert refused.json()["error"]["code"] == ErrorCode.FILE_NOT_FOUND.value


async def test_a_delete_refused_for_a_live_session_says_how_many_hold_it() -> None:
    """409 with a count, and nothing removed.

    A count and not a list of ids: the count is what a caller acts on -- end those
    Sessions and retry -- and a list grows with the tenant's history, so the body would
    have no ceiling. The file is still readable afterwards, which is what makes this a
    refusal rather than a warning.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = uuid4()
    written = await stored(store, TenantId(tenant))
    storage.holders[written.id] = 3

    async with client_for(build_app(store), tenant) as caller:
        refused = await caller.delete(f"/v1/files/{written.id}")
        still_there = await caller.get(f"/v1/files/{written.id}")

    assert refused.status_code == STATUS_FOR[ErrorCode.FILE_IN_USE] == 409
    assert refused.json()["error"]["code"] == ErrorCode.FILE_IN_USE.value
    assert refused.json()["error"]["detail"]["sessions_holding"] == 3
    assert still_there.status_code == 200


async def test_naming_both_anchors_is_refused_and_names_them_both() -> None:
    """A page cannot start in two places, and the refusal says which two were sent.

    Reported as a request refusal rather than by picking one, because a caller who sent
    both has lost track of which way they are walking and a chosen anchor is a page they
    cannot tell from the one they meant.
    """
    store = a_store(Storage())
    one, other = uuid4(), uuid4()

    async with client_for(build_app(store), uuid4()) as caller:
        refused = await caller.get(
            "/v1/files", params={"after_id": str(one), "before_id": str(other)}
        )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    detail = refused.json()["error"]["detail"]
    assert detail["reason"] == REASON_TWO_ANCHORS
    assert (detail["after_id"], detail["before_id"]) == (str(one), str(other))


async def test_an_unknown_anchor_is_a_cursor_refusal_and_not_an_empty_page() -> None:
    """An empty page reads as a finished walk, which is the wrong answer to a typo.

    Refused as a cursor rather than as a missing file, because the id arrived as a
    position in a query string and not as the thing being asked for -- a 404 on a
    collection reads as the collection being gone.
    """
    store = a_store(Storage())
    unknown = uuid4()

    async with client_for(build_app(store), uuid4()) as caller:
        refused = await caller.get("/v1/files", params={"after_id": str(unknown)})

    assert refused.status_code == STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]
    assert refused.json()["error"]["code"] == ErrorCode.PAGINATION_CURSOR_INVALID.value
    assert refused.json()["error"]["detail"]["anchor_id"] == str(unknown)


@pytest.mark.parametrize("limit", [0, MAX_FILE_PAGE_SIZE + 1])
async def test_a_limit_outside_the_published_range_is_refused(limit: int) -> None:
    """The ceiling is the reference's 1000 rather than this repository's usual 100.

    A client generated from the documented surface sends `limit=1000` and expects it
    served, so the boundary is asserted from both sides: the published maximum is
    accepted and one past it is not.
    """
    store = a_store(Storage())

    async with client_for(build_app(store), uuid4()) as caller:
        refused = await caller.get("/v1/files", params={"limit": limit})
        admitted = await caller.get("/v1/files", params={"limit": MAX_FILE_PAGE_SIZE})

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert admitted.status_code == 200


async def test_the_default_page_size_is_the_published_twenty() -> None:
    """Asserted through the route, because the constant alone proves nothing about it.

    A default declared and then not wired into the parameter is exactly the defect this
    catches: twenty-one files, no `limit`, and the page has to stop at twenty.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = uuid4()
    for n in range(DEFAULT_FILE_PAGE_SIZE + 1):
        await stored(store, TenantId(tenant), name=f"{n}.txt")

    async with client_for(build_app(store), tenant) as caller:
        page = await caller.get("/v1/files")

    assert len(page.json()["data"]) == DEFAULT_FILE_PAGE_SIZE
    assert page.json()["has_more"] is True


async def test_a_process_with_no_bucket_refuses_all_three_and_names_the_variable() -> (
    None
):
    """The wiring `build()` produces with MAP_OBJECT_BUCKET unset, on the new routes.
    Asserted through the routes rather than against `NoUploadBucket`, because the
    property is that the refusal reaches a caller as a published code with a readable
    reason -- an unhandled RuntimeError would satisfy any test of the storage alone.
    """
    app = build_app(unconfigured_file_store())
    file_id = uuid4()

    async with client_for(app, uuid4()) as caller:
        answers = (
            await caller.get("/v1/files"),
            await caller.get(f"/v1/files/{file_id}"),
            await caller.delete(f"/v1/files/{file_id}"),
        )

    for refused in answers:
        assert refused.status_code == STATUS_FOR[ErrorCode.INTERNAL]
        assert refused.json()["error"]["code"] == ErrorCode.INTERNAL.value
        assert "MAP_OBJECT_BUCKET" in refused.json()["error"]["message"]


async def test_no_answer_these_routes_give_names_the_bucket_or_the_object_key() -> None:
    """The identifier is the only handle a tenant gets.
    A key or a bucket in a body hands out the storage layout and the tenant segment
    inside it. The listing and the metadata read are the two new bodies carrying stored
    facts, so they are the two that could leak one.
    """
    storage = Storage()
    store = a_store(storage)
    tenant = uuid4()
    written = await stored(store, TenantId(tenant))

    async with client_for(build_app(store), tenant) as caller:
        listing = await caller.get("/v1/files")
        metadata = await caller.get(f"/v1/files/{written.id}")

    key = next(iter(storage.objects))
    for body in (listing.text, metadata.text):
        assert key not in body
        assert "uploads/" not in body
        assert "bucket" not in body.lower()
