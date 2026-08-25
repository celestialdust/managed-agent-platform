"""The Evidence store writes the bytes before the row that claims they exist.

Tier 1 (testcontainers, real PostgreSQL 17) for the ledger half. The object half runs
against an in-memory `EvidenceBlobs`, which is the declared port and therefore the
genuine boundary: `testcontainers.community.minio` imports `minio`, which this tree does
not install, so `S3EvidenceBlobs.put`, `.get` and `.delete_prefix` are graded
structurally by `mypy --strict` through `shipped_blobs` below and never executed. What
IS exercised for real is the thing this module owns -- the ordering of the two writes
and what each of them records.

The order is the guarantee. Bytes first means a ledger row always points at an object
that is there; the reverse failure leaves bytes with no row, which is harmless and
self-clearing because they sit under the Session's own prefix and the expiry sweep takes
them with everything else. A row with no bytes is a record that lies, and there is no
sequence of calls here that can produce one.
"""

from __future__ import annotations

from uuid import uuid4

import aioboto3  # type: ignore[import-untyped]
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.s3.evidence_store import EvidenceStore, S3EvidenceBlobs
from managed_agent.core.ids import SessionId
from managed_agent.core.vfs.evidence import (
    CaptureAsEvidence,
    CaptureContext,
    CapturePoint,
    CaptureThreshold,
    EvidenceBlobs,
    ReturnInline,
    decide,
    digest_of,
    evidence_object_key,
)

THRESHOLD = CaptureThreshold(100)

_READ = (
    sa.text(
        "SELECT hash_algorithm, hash_hex, object_key, byte_length, threshold_bytes,"
        " passed_through_bytes, capture_point, tool_name, truncated_at_runtime_cap"
        " FROM evidence_capture WHERE session_id = :sid AND call_id = :call_id"
    )
    .bindparams(sa.bindparam("sid", type_=sa.Uuid()))
    .columns(
        hash_algorithm=sa.Text(),
        hash_hex=sa.Text(),
        object_key=sa.Text(),
        byte_length=sa.BigInteger(),
        threshold_bytes=sa.BigInteger(),
        passed_through_bytes=sa.BigInteger(),
        capture_point=sa.Text(),
        tool_name=sa.Text(),
        truncated_at_runtime_cap=sa.Boolean(),
    )
)

_COUNT = sa.text(
    "SELECT count(*) FROM evidence_capture WHERE session_id = :sid"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))


class MemoryBlobs:
    """The object store as a dict. Real enough: keys in, bytes out, prefix delete."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, body: bytes) -> str:
        self.objects[key] = body
        return key

    async def get(self, key: str) -> bytes:
        return self.objects[key]

    async def delete_prefix(self, prefix: str) -> int:
        doomed = [k for k in self.objects if k.startswith(prefix)]
        for key in doomed:
            del self.objects[key]
        return len(doomed)


class RefusingBlobs(MemoryBlobs):
    """An object store that will not accept a write, to test what the ledger does."""

    async def put(self, key: str, body: bytes) -> str:
        raise RuntimeError("the bucket refused this write")


def shipped_blobs() -> EvidenceBlobs:
    """The real S3 adapter under the port's type.

    Annotated as the port rather than as its own class so `mypy --strict` grades the
    whole of `S3EvidenceBlobs` against the interface `EvidenceStore` actually drives.
    That is the only grading its three methods get here; the bucket is never dialled.
    """
    return S3EvidenceBlobs(aioboto3.Session(), "no-bucket-is-reached-here")


def a_context(session: SessionId, call_id: str = "call-1") -> CaptureContext:
    return CaptureContext(
        session_id=session,
        call_id=call_id,
        tool_name="acme__search",
        capture_point=CapturePoint.TOOL_GATEWAY,
    )


def a_capture(payload: bytes, passed_through_bytes: int = 0) -> CaptureAsEvidence:
    decision = decide(payload, THRESHOLD, passed_through_bytes=passed_through_bytes)
    assert isinstance(decision, CaptureAsEvidence), "payload is below the threshold"
    return decision


async def test_the_adapter_shipped_satisfies_the_port_it_is_wired_behind() -> None:
    """Structural only -- see the module docstring on why the bucket is not reached."""
    assert isinstance(shipped_blobs(), S3EvidenceBlobs)


async def test_a_capture_writes_the_bytes_and_a_row_that_names_them(
    engine: AsyncEngine,
) -> None:
    blobs = MemoryBlobs()
    store = EvidenceStore(blobs, engine)
    session = SessionId(uuid4())
    payload = b"p" * 200
    decision = a_capture(payload, passed_through_bytes=4_096)

    ref = await store.record_captured(a_context(session), payload, decision)

    assert ref.object_key.endswith(decision.digest.hex)
    assert list(blobs.objects) == [ref.object_key]
    async with engine.connect() as conn:
        row = (await conn.execute(_READ, {"sid": session, "call_id": "call-1"})).one()
    assert row.hash_hex == decision.digest.hex
    assert row.hash_algorithm == "sha256"
    assert row.object_key == ref.object_key
    assert row.byte_length == 200
    assert row.threshold_bytes == 100
    assert row.passed_through_bytes == 4_096
    assert row.capture_point == "tool-gateway"
    assert row.tool_name == "acme__search"
    assert row.truncated_at_runtime_cap is False


async def test_rehashing_what_a_reader_downloads_reproduces_the_recorded_digest(
    engine: AsyncEngine,
) -> None:
    """The whole of the hash contract: an auditor who fetches the object and hashes it
    either gets the recorded value back or has found a real discrepancy."""
    blobs = MemoryBlobs()
    store = EvidenceStore(blobs, engine)
    session = SessionId(uuid4())
    payload = bytes(range(256)) * 4

    ref = await store.record_captured(a_context(session), payload, a_capture(payload))

    downloaded = await blobs.get(ref.object_key)
    assert downloaded == payload
    assert digest_of(downloaded) == ref.digest
    async with engine.connect() as conn:
        row = (await conn.execute(_READ, {"sid": session, "call_id": "call-1"})).one()
    assert digest_of(downloaded).hex == row.hash_hex
    assert digest_of(downloaded).byte_length == row.byte_length


async def test_two_sessions_capturing_one_payload_get_two_objects(
    engine: AsyncEngine,
) -> None:
    """Identical bytes are deliberately stored twice. The per-Session prefix is what
    lets one Session's expiry delete exactly its own Evidence with a prefix delete,
    instead of a reference count nobody is maintaining."""
    blobs = MemoryBlobs()
    store = EvidenceStore(blobs, engine)
    one, two = SessionId(uuid4()), SessionId(uuid4())
    payload = b"shared" * 100

    first = await store.record_captured(a_context(one), payload, a_capture(payload))
    second = await store.record_captured(a_context(two), payload, a_capture(payload))

    assert first.digest == second.digest
    assert first.object_key != second.object_key
    assert len(blobs.objects) == 2

    assert await blobs.delete_prefix(f"evidence/{one}/") == 1
    assert await blobs.get(second.object_key) == payload


async def test_a_session_prefix_delete_takes_every_object_it_holds(
    engine: AsyncEngine,
) -> None:
    blobs = MemoryBlobs()
    store = EvidenceStore(blobs, engine)
    session = SessionId(uuid4())
    for n in range(3):
        payload = f"result-{n}".encode() * 40
        await store.record_captured(
            a_context(session, f"call-{n}"), payload, a_capture(payload)
        )

    assert len(blobs.objects) == 3
    assert await blobs.delete_prefix(f"evidence/{session}/") == 3
    assert blobs.objects == {}


async def test_an_object_write_that_fails_leaves_no_row_claiming_it(
    engine: AsyncEngine,
) -> None:
    """A row with no bytes is a record that lies. Bytes first is what makes that
    unreachable -- the ledger insert is never attempted."""
    store = EvidenceStore(RefusingBlobs(), engine)
    session = SessionId(uuid4())
    payload = b"q" * 200

    with pytest.raises(RuntimeError, match="refused this write"):
        await store.record_captured(a_context(session), payload, a_capture(payload))

    async with engine.connect() as conn:
        assert await conn.scalar(_COUNT, {"sid": session}) == 0


async def test_an_inline_return_gets_a_row_and_no_object(engine: AsyncEngine) -> None:
    """The row is what lets a reviewer who finds no Evidence for a call see that the
    output was small, rather than that the record is missing."""
    blobs = MemoryBlobs()
    store = EvidenceStore(blobs, engine)
    session = SessionId(uuid4())

    await store.record_inline(
        a_context(session),
        ReturnInline(byte_length=12, threshold=100, passed_through_bytes=204_800),
    )

    assert blobs.objects == {}
    async with engine.connect() as conn:
        row = (await conn.execute(_READ, {"sid": session, "call_id": "call-1"})).one()
    assert row.hash_hex is None
    assert row.hash_algorithm is None
    assert row.object_key is None
    assert row.byte_length == 12
    assert row.threshold_bytes == 100
    assert row.passed_through_bytes == 204_800, (
        "the row would read as a small output, which is what the count is for"
    )
    assert row.truncated_at_runtime_cap is False


async def test_a_capture_the_runtime_had_already_cut_says_so_in_its_row(
    engine: AsyncEngine,
) -> None:
    """Evidence with a tail missing must never be presented as complete."""
    store = EvidenceStore(MemoryBlobs(), engine)
    session = SessionId(uuid4())
    payload = b"r" * 200

    ref = await store.record_captured(
        CaptureContext(
            session_id=session,
            call_id="call-1",
            tool_name="builtin__bash",
            capture_point=CapturePoint.SESSION_SHIM,
        ),
        payload,
        a_capture(payload),
        truncated_at_runtime_cap=True,
    )

    assert ref.truncated_at_runtime_cap is True
    async with engine.connect() as conn:
        row = (await conn.execute(_READ, {"sid": session, "call_id": "call-1"})).one()
    assert row.truncated_at_runtime_cap is True
    assert row.capture_point == "session-shim"


async def test_recording_one_call_twice_is_refused_by_the_store(
    engine: AsyncEngine,
) -> None:
    """Two rows for one call would disagree and nothing would say which describes the
    bytes the model was given. The object write is idempotent by construction -- the key
    is the digest of the body -- so the retry rewrites identical bytes over themselves
    and the ledger is what refuses."""
    blobs = MemoryBlobs()
    store = EvidenceStore(blobs, engine)
    session = SessionId(uuid4())
    payload = b"s" * 200

    await store.record_captured(a_context(session), payload, a_capture(payload))
    with pytest.raises(IntegrityError):
        await store.record_captured(a_context(session), payload, a_capture(payload))

    assert list(blobs.objects) == [
        evidence_object_key(session, a_capture(payload).digest)
    ]
    async with engine.connect() as conn:
        assert await conn.scalar(_COUNT, {"sid": session}) == 1
