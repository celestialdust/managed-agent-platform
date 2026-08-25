"""The lane store writes the bytes, then the event that says the bytes exist.

The object half runs against a fake `.client("s3")`, which is the whole of the surface
`S3LaneBlobs` touches, so the conditional write and the pagination are exercised for
real rather than asserted about in prose. `testcontainers.community.minio` imports
`minio`, which this tree does not install; the fake is what is available and it is
enough, because what it stands in for is S3's own answer to a condition, and that answer
is a documented error code rather than a behaviour worth a container.

**The header is asserted, not assumed.** A sealed lane's immutability rests entirely on
`IfNoneMatch` reaching the service: without it every `place` is an ordinary overwrite
and the whole lifecycle is a comment. So one case reads the request the fake received.
That is the difference between testing the guarantee and testing the code's opinion of
itself.

The order is the other guarantee. Bytes first means an event always names an object that
is there; the reverse leaves bytes no provenance mentions, which is recoverable and
self-clearing under the Session's own prefix. An event naming bytes that were never
stored is a record that lies about a produced artifact, so the ordering is asserted
directly from a shared journal rather than inferred from a failure.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from types import TracebackType
from typing import Any, Self
from uuid import uuid4

import aioboto3  # type: ignore[import-untyped]
import pytest

from managed_agent.adapters.s3.session_vfs import (
    _ALREADY_THERE,
    S3LaneBlobs,
    SessionVfsStore,
    UnconfiguredSessionVfs,
)
from managed_agent.core.ids import Seq, SessionId, TenantId
from managed_agent.core.ports import EventLogAppend
from managed_agent.core.vfs.session_vfs import (
    Lane,
    LaneBlobs,
    MutableFile,
    MutableLane,
    ObjectAlreadyPresent,
    SealedFile,
    SealedLane,
    SessionFiles,
    SourceRef,
    VfsFile,
    VfsUnconfigured,
    lane_prefix,
)
from managed_agent.core.vfs.vfs_provenance import provenance
from managed_agent.core.vocabulary.vfs import OBJECT_PLACED, OBJECT_REPLACED

A_TENANT = TenantId(uuid4())
A_SESSION = SessionId(uuid4())

ARTIFACTS = SealedLane("kept")
WORKING = MutableLane("scratchpad")
LANES: tuple[Lane, ...] = (ARTIFACTS, WORKING)
"""Example lanes, declared here because the platform declares none.

The store's behaviour splits exactly two ways -- a conditional write for a sealed lane
and an unconditional one for a mutable lane -- so two examples exercise every branch
this adapter has. A third would add a case and no coverage. These names are local
fixtures and not a platform default; the module deliberately names no lane."""


class _ClientError(Exception):
    """A botocore client error, as this adapter reads it: by its response shape.

    The adapter never imports the client's exception classes -- it reads
    `response["Error"]["Code"]` -- so a fake that carries that shape is the genuine
    boundary and not an approximation of one.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _NoSuchKey(Exception):
    pass


class _Exceptions:
    ClientError = _ClientError
    NoSuchKey = _NoSuchKey


class _Body:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def read(self) -> bytes:
        return self._body


class _Paginator:
    def __init__(self, objects: dict[str, bytes], page_size: int) -> None:
        self._objects = objects
        self._page_size = page_size

    async def paginate(
        self, *, Bucket: str, Prefix: str
    ) -> AsyncIterator[dict[str, object]]:
        matched = sorted(key for key in self._objects if key.startswith(Prefix))
        for start in range(0, max(len(matched), 1), self._page_size):
            page = matched[start : start + self._page_size]
            yield {
                "Contents": [
                    {"Key": key, "Size": len(self._objects[key])} for key in page
                ]
            }


class _Client:
    exceptions = _Exceptions

    def __init__(
        self,
        objects: dict[str, bytes],
        puts: list[dict[str, object]],
        page_size: int,
        fail_code: str | None = None,
    ) -> None:
        self._objects = objects
        self._puts = puts
        self._page_size = page_size
        self._fail_code = fail_code

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        IfNoneMatch: str | None = None,
    ) -> None:
        self._puts.append({"Key": Key, "IfNoneMatch": IfNoneMatch})
        if self._fail_code is not None:
            raise _ClientError(self._fail_code)
        if IfNoneMatch == "*" and Key in self._objects:
            raise _ClientError("PreconditionFailed")
        self._objects[Key] = Body

    async def get_object(self, *, Bucket: str, Key: str) -> dict[str, _Body]:
        if Key not in self._objects:
            raise _NoSuchKey(Key)
        return {"Body": _Body(self._objects[Key])}

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self._objects, self._page_size)


class FakeS3Session:
    """`.client("s3")` and nothing else, because that is all the adapter calls.

    This type-checks under `mypy --strict` only because `aioboto3` is untyped, so
    `aioboto3.Session` resolves to `Any` where the adapter declares it.
    """

    def __init__(self, page_size: int = 1000, fail_code: str | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts: list[dict[str, object]] = []
        self._page_size = page_size
        self._fail_code = fail_code

    def client(self, name: str) -> _Client:
        assert name == "s3"
        return _Client(self.objects, self.puts, self._page_size, self._fail_code)


class RecordingLog:
    """An `EventLogAppend` that numbers appends and keeps them for the fold."""

    def __init__(self, journal: list[str] | None = None) -> None:
        self.appended: list[tuple[str, dict[str, object]]] = []
        self._journal = journal

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        if self._journal is not None:
            self._journal.append(f"event:{type_}")
        self.appended.append((type_, payload))
        return Seq(len(self.appended))


class JournallingBlobs:
    """A `LaneBlobs` that records the order it was called in, and stores nothing."""

    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    async def put_new(self, key: str, body: bytes) -> None:
        self._journal.append("bytes")

    async def put(self, key: str, body: bytes) -> None:
        self._journal.append("bytes")

    async def get(self, key: str) -> bytes | None:
        return None

    async def list_prefix(self, prefix: str) -> Sequence[Any]:
        return ()


class RefusingBlobs:
    """A `LaneBlobs` whose writes fail, to prove no event is appended when they do."""

    async def put_new(self, key: str, body: bytes) -> None:
        raise RuntimeError("the bucket refused")

    async def put(self, key: str, body: bytes) -> None:
        raise RuntimeError("the bucket refused")

    async def get(self, key: str) -> bytes | None:
        return None

    async def list_prefix(self, prefix: str) -> Sequence[Any]:
        return ()


_BLOBS_PORT: LaneBlobs = JournallingBlobs([])
"""Graded against the port by `mypy --strict` rather than by a runtime check: an
isinstance against a runtime_checkable Protocol compares method names and not
signatures, which is the half that actually drifts."""


def shipped_blobs() -> LaneBlobs:
    """The real S3 adapter under the port's type.

    Annotated as the port rather than as its own class, so `mypy --strict` grades the
    whole of `S3LaneBlobs` against the interface the store actually drives. That is the
    only grading its four methods get here beyond the fake; the bucket is never dialled.
    """
    return S3LaneBlobs(aioboto3.Session(), "no-bucket-is-reached-here")


def a_store(fake: FakeS3Session, log: EventLogAppend) -> SessionFiles:
    """The store under the port's type, so the port is what the cases drive."""
    return SessionVfsStore(S3LaneBlobs(fake, "a-bucket"), log)


def a_file(lane: Lane, relative: str = "a-file.txt") -> VfsFile:
    if isinstance(lane, MutableLane):
        return MutableFile(A_TENANT, A_SESSION, lane, relative)
    return SealedFile(A_TENANT, A_SESSION, lane, relative)


def test_there_are_lanes_to_grade() -> None:
    """Guard the guard: the per-lane cases below are parametrized over `LANES`."""
    assert len(LANES) >= 2
    assert any(isinstance(lane, MutableLane) for lane in LANES)
    assert any(isinstance(lane, SealedLane) for lane in LANES)


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
async def test_a_write_in_any_lane_is_readable_at_its_own_key(lane: Lane) -> None:
    fake = FakeS3Session()
    store = a_store(fake, RecordingLog())
    file = a_file(lane)

    stored = await store.place(file, b"the bytes")

    assert stored.key == file.key
    assert fake.objects[file.key] == b"the bytes"
    assert await store.read(file) == b"the bytes"


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
async def test_no_lane_lets_place_overwrite_what_is_already_there(lane: Lane) -> None:
    """The conditional write, graded in every lane rather than in the first.

    `place` is the only write a sealed lane has, so this is that lane's immutability. In
    the mutable lane it says the caller wanted `replace` and reached for the creating
    call -- the same refusal, a different meaning, and both are decisions.
    """
    fake = FakeS3Session()
    store = a_store(fake, RecordingLog())
    file = a_file(lane)
    await store.place(file, b"first")

    with pytest.raises(ObjectAlreadyPresent):
        await store.place(file, b"second")

    assert fake.objects[file.key] == b"first", "the refused write must not have landed"


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
async def test_place_sends_the_condition_that_makes_the_refusal_the_stores(
    lane: Lane,
) -> None:
    """Without `IfNoneMatch` on the wire, every `place` is an ordinary overwrite.

    Read off the request the client received. The immutability of a sealed lane is this
    header and nothing else -- a prior existence check would be a race, and the code
    being careful is not a property the bucket enforces.
    """
    fake = FakeS3Session()
    await a_store(fake, RecordingLog()).place(a_file(lane), b"the bytes")

    assert fake.puts == [{"Key": a_file(lane).key, "IfNoneMatch": "*"}]


async def test_replace_overwrites_and_sends_no_condition() -> None:
    """The mutable lane's write is unconditional; that is what makes it the mutable
    lane."""
    fake = FakeS3Session()
    store = a_store(fake, RecordingLog())
    file = MutableFile(A_TENANT, A_SESSION, WORKING, "draft.md")
    await store.place(file, b"first")

    stored = await store.replace(file, b"second")

    assert fake.objects[file.key] == b"second"
    assert stored.key == file.key
    assert fake.puts[-1] == {"Key": file.key, "IfNoneMatch": None}


async def test_the_bytes_are_written_before_the_event_that_claims_they_exist() -> None:
    journal: list[str] = []
    store = SessionVfsStore(JournallingBlobs(journal), RecordingLog(journal))

    await store.place(a_file(ARTIFACTS), b"the report")

    assert journal == ["bytes", f"event:{OBJECT_PLACED}"]


async def test_a_replace_orders_its_two_writes_the_same_way() -> None:
    journal: list[str] = []
    store = SessionVfsStore(JournallingBlobs(journal), RecordingLog(journal))

    await store.replace(MutableFile(A_TENANT, A_SESSION, WORKING, "d.md"), b"x")

    assert journal == ["bytes", f"event:{OBJECT_REPLACED}"]


async def test_a_write_that_never_landed_appends_no_event() -> None:
    """The failure that must stay impossible: an event naming bytes nobody stored."""
    log = RecordingLog()
    store = SessionVfsStore(RefusingBlobs(), log)

    with pytest.raises(RuntimeError):
        await store.place(a_file(ARTIFACTS), b"the report")

    assert log.appended == [], "an event was recorded for bytes that were never written"


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
async def test_what_a_write_appends_folds_to_the_provenance_of_what_it_wrote(
    lane: Lane,
) -> None:
    """The adapter's payload and the projection's reader, joined end to end.

    Each side has its own cases; this is the one that fails if they stop agreeing, which
    is the failure neither side's own tests can see.
    """
    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class Rec:
        session_id: SessionId
        seq: Seq
        type: str
        payload: dict[str, object]

    log = RecordingLog()
    store = a_store(FakeS3Session(), log)
    source = SourceRef(
        relative="raw.json",
        digest=(
            await store.place(a_file(ARTIFACTS, "raw.json"), b"the raw evidence")
        ).digest,
    )

    stored = await store.place(a_file(lane, "out.md"), b"the output", sources=(source,))

    folded = provenance(
        [
            Rec(A_SESSION, Seq(n + 1), type_, payload)
            for n, (type_, payload) in enumerate(log.appended)
        ]
    )
    record = folded[(lane.directory, "out.md")]
    assert record.digest == stored.digest
    assert record.sources == (source,)


async def test_a_missing_object_reads_as_nothing_rather_than_raising() -> None:
    store = a_store(FakeS3Session(), RecordingLog())
    assert await store.read(a_file(ARTIFACTS, "never-written.md")) is None


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
async def test_listing_a_lane_returns_its_own_objects_with_the_prefix_stripped(
    lane: Lane,
) -> None:
    """A lane listing must not include a sibling lane's objects.

    Written per lane because the prefix is composed per lane, and a listing that read
    one segment too high would return the whole Session in every lane while a
    single-lane case still passed.
    """
    fake = FakeS3Session()
    store = a_store(fake, RecordingLog())
    for other in LANES:
        await store.place(a_file(other, "deep/one.txt"), b"xy")

    entries = await store.list_lane(A_TENANT, A_SESSION, lane)

    assert [entry.relative for entry in entries] == ["deep/one.txt"]
    assert [entry.byte_length for entry in entries] == [2]


async def test_a_listing_longer_than_one_page_returns_every_object() -> None:
    """Pagination, because a lane has no bound and the API answers a page at a time.

    An unpaginated pass would return the first page and report success, which is a
    listing that is wrong in exactly the case a real lane reaches.
    """
    fake = FakeS3Session(page_size=2)
    store = a_store(fake, RecordingLog())
    for n in range(5):
        await store.place(a_file(WORKING, f"file-{n}.txt"), b"x")

    entries = await store.list_lane(A_TENANT, A_SESSION, WORKING)

    assert sorted(entry.relative for entry in entries) == [
        f"file-{n}.txt" for n in range(5)
    ]


async def test_a_zero_byte_object_naming_the_prefix_itself_is_not_an_entry() -> None:
    """Some tools write one to make a prefix look like a directory."""
    fake = FakeS3Session()
    prefix = lane_prefix(A_TENANT, A_SESSION, WORKING)
    fake.objects[prefix] = b""
    store = a_store(fake, RecordingLog())

    assert await store.list_lane(A_TENANT, A_SESSION, WORKING) == []


@pytest.mark.parametrize("code", sorted(_ALREADY_THERE))
async def test_every_code_meaning_the_key_was_taken_reads_as_that(code: str) -> None:
    """Both codes, graded one at a time.

    `PreconditionFailed` is the ordinary answer; `ConditionalRequestConflict` is a lost
    race between two conditional writes. Removing either from the set would leave that
    case surfacing as an unclassified infrastructure error, and a single-code case would
    not notice.
    """

    blobs = S3LaneBlobs(FakeS3Session(fail_code=code), "a-bucket")
    with pytest.raises(ObjectAlreadyPresent):
        await blobs.put_new("sessions/t/s/evidence/a.txt", b"x")


async def test_a_service_error_that_is_not_about_the_key_is_not_swallowed() -> None:
    """`AccessDenied` reported as "already present" would read as a benign refusal."""

    blobs = S3LaneBlobs(FakeS3Session(fail_code="AccessDenied"), "a-bucket")
    with pytest.raises(_ClientError):
        await blobs.put_new("sessions/t/s/evidence/a.txt", b"x")


_UNCONFIGURED_CALLS: tuple[str, ...] = ("place", "replace", "read", "list_lane")
"""Every method of the refusing default, so each is graded rather than the first.

Each one encodes the same decision -- that an unwired VFS refuses instead of answering
emptily -- and a method that quietly started returning `None` or `[]` would make a
Session with no durable store look like one that produced nothing.
"""


@pytest.mark.parametrize("call", _UNCONFIGURED_CALLS)
async def test_the_unconfigured_vfs_refuses_every_call(call: str) -> None:
    vfs: SessionFiles = UnconfiguredSessionVfs()
    file = MutableFile(A_TENANT, A_SESSION, WORKING, "a.txt")
    attempts: dict[str, Callable[[], Awaitable[object]]] = {
        "place": lambda: vfs.place(file, b"x"),
        "replace": lambda: vfs.replace(file, b"x"),
        "read": lambda: vfs.read(file),
        "list_lane": lambda: vfs.list_lane(A_TENANT, A_SESSION, WORKING),
    }

    with pytest.raises(VfsUnconfigured):
        await attempts[call]()


def test_the_unconfigured_default_covers_every_method_of_the_port() -> None:
    """The list above is complete against the port, not against what was remembered."""
    declared = {name for name in vars(SessionFiles) if not name.startswith("_")} - {
        "place",
        "replace",
        "read",
        "list_lane",
    }
    assert declared == set(), f"{declared} is on the port and ungraded above"


def test_the_shipped_adapter_satisfies_the_port() -> None:
    assert isinstance(shipped_blobs(), LaneBlobs)
