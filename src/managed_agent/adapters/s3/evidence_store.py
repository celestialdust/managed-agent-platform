"""The captured payload, and the ledger row recording that a capture happened.

Evidence is one durable fact spread across two stores — the payload in the object store,
addressed by its own digest, and a row in the relational store saying that a capture
happened, at which point, over which threshold, and how many octets of the same result
went back to the caller unweighed. Both writes are ordered here rather than at the
callers because the order is the guarantee, and an order enforced at two call sites is
an order nobody owns. The directory names the store holding the bytes, which is
the substantive half; keeping the ledger write in the same class is what makes it
impossible to call one half without the other.

Bytes first, row second. A ledger row therefore always points at an object that is
there. The reverse failure — bytes written and no row — is harmless and self-clearing,
because the object sits under the Session's own prefix and the expiry sweep takes it
with everything else; a row with no bytes is a record that lies, and there is no
sequence of calls here that can produce one.

The object write is unconditional and needs no compare-and-set. The key is the digest
of the body, so writing a key that already exists writes the identical bytes over
themselves — there is nothing an overwrite here can destroy.

The bind types are declared because a textual statement carries no column metadata for
SQLAlchemy to infer from, so the driver receives whatever Python object the caller
handed it. Under asyncpg a uuid passed as a string is parsed anyway and does match —
the honest reason to declare it is that the declaration keeps this adapter correct
under a driver that does not parse for you, and that
`tests/adapters/test_statements_declare_their_types.py` requires it.
"""

import aioboto3  # type: ignore[import-untyped]
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.control.files.store import UPLOAD_BUCKET_ENV_VAR
from managed_agent.core.vfs.evidence import (
    CaptureAsEvidence,
    CaptureContext,
    EvidenceBlobs,
    EvidenceRef,
    EvidenceStorageUnconfigured,
    ReturnInline,
    evidence_object_key,
    evidence_vfs_path,
)

_INSERT_CAPTURED = sa.text(
    "INSERT INTO evidence_capture (session_id, call_id, tool_name, capture_point,"
    " byte_length, threshold_bytes, passed_through_bytes, hash_algorithm, hash_hex,"
    " object_key, truncated_at_runtime_cap)"
    " VALUES (:sid, :call_id, :tool, :point, :len, :threshold, :through, :algo, :hex,"
    " :key, :cut)"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))

_INSERT_INLINE = sa.text(
    "INSERT INTO evidence_capture (session_id, call_id, tool_name, capture_point,"
    " byte_length, threshold_bytes, passed_through_bytes, truncated_at_runtime_cap)"
    " VALUES (:sid, :call_id, :tool, :point, :len, :threshold, :through, false)"
).bindparams(sa.bindparam("sid", type_=sa.Uuid()))


class UnconfiguredEvidence:
    """Both Evidence ports when nothing is wired behind them: every call refuses.

    Not a fallback that lets the work proceed. A capture point wired over this fails the
    tool call, which is what makes the one outcome this design exists to prevent
    unreachable — a large result handed to the model with no Evidence behind it and
    nothing saying so. A failed tool call is recoverable; a missing audit record is not.

    An inline return refuses too, and that is deliberate rather than strict for its own
    sake: the ledger row recording that output was below the threshold is what lets a
    reviewer tell "small" from "not recorded", so a call completed without one has left
    no honest trace either.

    Reading and sweeping refuse for the same reason rather than answering empty: "there
    is no bucket" and "the bucket holds nothing" are different facts, and an expiry
    sweep that reported zero deletions against an unconfigured store would look like a
    Session with no Evidence to remove.

    Both ports, in one class, because there is nothing to distinguish: the object half
    and the ledger half are unwired together or wired together, and two classes saying
    the same sentence would let a future wiring satisfy one and not the other.
    """

    def _refuse(self) -> EvidenceStorageUnconfigured:
        # The same variable an upload reads. Evidence and uploaded files share one
        # bucket and are separated by prefix, so there is one name to set and one thing
        # for an operator to get wrong; naming the upload variable in an Evidence
        # failure is correct rather than a leak of somebody else's concern.
        return EvidenceStorageUnconfigured(
            f"{UPLOAD_BUCKET_ENV_VAR} is not set, so tool output cannot be captured as"
            " Evidence"
        )

    async def put(self, key: str, body: bytes) -> str:
        raise self._refuse()

    async def get(self, key: str) -> bytes:
        raise self._refuse()

    async def delete_prefix(self, prefix: str) -> int:
        raise self._refuse()

    async def record_captured(
        self,
        ctx: CaptureContext,
        payload: bytes,
        decision: CaptureAsEvidence,
        truncated_at_runtime_cap: bool = False,
    ) -> EvidenceRef:
        raise self._refuse()

    async def record_inline(self, ctx: CaptureContext, decision: ReturnInline) -> None:
        raise self._refuse()


class S3EvidenceBlobs:
    """One bucket of content-addressed payloads, read and swept by prefix."""

    def __init__(self, session: aioboto3.Session, bucket: str) -> None:
        self._session = session
        self._bucket = bucket

    async def put(self, key: str, body: bytes) -> str:
        async with self._session.client("s3") as s3:
            await s3.put_object(Bucket=self._bucket, Key=key, Body=body)
        return key

    async def get(self, key: str) -> bytes:
        async with self._session.client("s3") as s3:
            response = await s3.get_object(Bucket=self._bucket, Key=key)
            body: bytes = await response["Body"].read()
        return body

    async def delete_prefix(self, prefix: str) -> int:
        """Remove every object under one prefix, returning how many went.

        Paginated because a Session's Evidence has no bound, and `list_objects_v2`
        answers a thousand keys at a time; a single unpaginated pass would silently
        leave the rest behind and report success.
        """
        deleted = 0
        async with self._session.client("s3") as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if keys:
                    await s3.delete_objects(
                        Bucket=self._bucket, Delete={"Objects": keys}
                    )
                    deleted += len(keys)
        return deleted


class EvidenceStore:
    """Records one capture decision in both stores, payload first, ledger row second."""

    def __init__(self, blobs: EvidenceBlobs, engine: AsyncEngine) -> None:
        self._blobs = blobs
        self._engine = engine

    async def record_captured(
        self,
        ctx: CaptureContext,
        payload: bytes,
        decision: CaptureAsEvidence,
        truncated_at_runtime_cap: bool = False,
    ) -> EvidenceRef:
        key = evidence_object_key(ctx.session_id, decision.digest)
        await self._blobs.put(key, payload)
        async with self._engine.begin() as conn:
            await conn.execute(
                _INSERT_CAPTURED,
                {
                    "sid": ctx.session_id,
                    "call_id": ctx.call_id,
                    "tool": ctx.tool_name,
                    "point": str(ctx.capture_point),
                    "len": decision.digest.byte_length,
                    "threshold": decision.threshold,
                    "through": decision.passed_through_bytes,
                    "algo": decision.digest.algorithm,
                    "hex": decision.digest.hex,
                    "key": key,
                    "cut": truncated_at_runtime_cap,
                },
            )
        return EvidenceRef(
            session_id=ctx.session_id,
            digest=decision.digest,
            capture_point=ctx.capture_point,
            object_key=key,
            vfs_path=evidence_vfs_path(decision.digest),
            truncated_at_runtime_cap=truncated_at_runtime_cap,
        )

    async def record_inline(self, ctx: CaptureContext, decision: ReturnInline) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                _INSERT_INLINE,
                {
                    "sid": ctx.session_id,
                    "call_id": ctx.call_id,
                    "tool": ctx.tool_name,
                    "point": str(ctx.capture_point),
                    "len": decision.byte_length,
                    "threshold": decision.threshold,
                    "through": decision.passed_through_bytes,
                },
            )
