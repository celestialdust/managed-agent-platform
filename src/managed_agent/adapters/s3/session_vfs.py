"""A Session's lanes over one S3 bucket, and the event that records each write.

Two classes with two jobs. `S3LaneBlobs` is the bucket and nothing else.
`SessionVfsStore` orders the two durable writes a VFS write consists of -- the object,
then the event saying it exists -- because the order is the guarantee, and an order
enforced at each call site is an order nobody owns. That is the shape `EvidenceStore` in
this same directory already has, for the same reason.

**Bytes first, event second.** An event therefore always names an object that is there.
The reverse failure -- bytes stored and no event -- leaves an object no provenance
mentions, which is recoverable by listing the lane and is self-clearing under the
Session's own prefix sweep. An event naming bytes that were never stored is a record
that lies about a produced artifact, and there is no sequence of calls here that can
write one.

**A sealed lane's immutability is enforced by S3, not by this code being careful.**
`put_new` sends a conditional write -- `IfNoneMatch: "*"` -- so the store itself refuses
a second write to an occupied key and the refusal does not depend on a prior existence
check that a concurrent writer could invalidate. That matters more than the tidiness of
it: the control plane's grant carries `s3:GetObject` and `s3:PutObject` and deliberately
no `s3:DeleteObject`, so overwriting is the only way to destroy a stored object at all,
and a conditional put is exactly the operation that closes it. The condition costs no
additional IAM action -- it is still `s3:PutObject`.

`list_prefix` is the one method here that needs `s3:ListBucket` on the bucket, which is
an action the control plane's role does not hold today. It is written as the port
declares it rather than omitted, because a lane that cannot be listed is a missing grant
and not a missing feature.
"""

from collections.abc import Sequence

import aioboto3  # type: ignore[import-untyped]

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.ports import EventLogAppend
from managed_agent.core.session.event_append import append_in_order
from managed_agent.core.vfs.evidence import digest_of
from managed_agent.core.vfs.session_vfs import (
    Lane,
    LaneBlobs,
    LaneEntry,
    ObjectAlreadyPresent,
    SourceRef,
    StoredObject,
    VfsFile,
    VfsUnconfigured,
    lane_prefix,
)
from managed_agent.core.vfs.vfs_provenance import written_payload
from managed_agent.core.vocabulary.vfs import OBJECT_PLACED

_ALREADY_THERE: frozenset[str] = frozenset(
    {"PreconditionFailed", "ConditionalRequestConflict"}
)
"""The service error codes that mean a conditional write stored nothing.

Two, not one. `PreconditionFailed` is the ordinary answer when the key is already
occupied; `ConditionalRequestConflict` is what S3 returns when two conditional writes to
one key race each other. Both mean the same thing to a caller -- this write did not
happen because something else is there -- and treating only the first as the refusal
would surface a lost race as an unclassified infrastructure error.
"""


def _service_error_code(exc: BaseException) -> str:
    """The AWS service error code an exception carries, or "" when it carries none.

    Read off the exception rather than matched by type, for the reason
    `adapters/secrets/vault.py` gives about its own copy: the shape
    `response["Error"]["Code"]` is what every botocore client error carries, and reading
    it needs no import of the client's exception classes -- which matters here because
    this tree declares the packages it imports and `botocore` is not among them.

    A second copy of four lines rather than an import across two adapter families. The
    Rule of Three says the third copy is when to extract; the second is when to say
    where the other one is, which this does.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return ""
    error = response.get("Error")
    return error.get("Code", "") if isinstance(error, dict) else ""


class S3LaneBlobs:
    """One bucket of lane objects: conditional create, overwrite, read and list."""

    def __init__(self, session: aioboto3.Session, bucket: str) -> None:
        self._session = session
        self._bucket = bucket

    async def put_new(self, key: str, body: bytes) -> None:
        """Store `body` at `key` only if the key holds nothing.

        Raises `ObjectAlreadyPresent` rather than letting the service error out, so a
        caller that has just been refused an overwrite reads a platform word for it and
        no infrastructure vocabulary reaches a decision that is about the platform.
        """
        async with self._session.client("s3") as s3:
            try:
                await s3.put_object(
                    Bucket=self._bucket, Key=key, Body=body, IfNoneMatch="*"
                )
            except s3.exceptions.ClientError as exc:
                if _service_error_code(exc) in _ALREADY_THERE:
                    raise ObjectAlreadyPresent(key) from exc
                raise

    async def get(self, key: str) -> bytes | None:
        async with self._session.client("s3") as s3:
            try:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
            except s3.exceptions.NoSuchKey:
                return None
            body: bytes = await response["Body"].read()
        return body

    async def list_prefix(self, prefix: str) -> Sequence[LaneEntry]:
        """Every object under `prefix`, with the prefix stripped from each name.

        Paginated because a lane has no bound and `list_objects_v2` answers a thousand
        keys at a time; a single unpaginated pass would silently leave the rest behind
        and report success. That failure has already been made once in this directory,
        which is why the note is here rather than assumed.

        A key that is exactly the prefix is skipped. Some tools create a zero-byte
        object to make a prefix look like a directory, and reporting it as a lane entry
        would put a file with an empty name in a listing.
        """
        found: list[LaneEntry] = []
        async with self._session.client("s3") as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    relative = str(obj["Key"])[len(prefix) :]
                    if relative:
                        found.append(
                            LaneEntry(relative=relative, byte_length=int(obj["Size"]))
                        )
        return found


class SessionVfsStore:
    """Writes a lane object and then the event recording it, in that order."""

    def __init__(self, blobs: LaneBlobs, log: EventLogAppend) -> None:
        self._blobs = blobs
        self._log = log

    async def place(
        self, file: VfsFile, body: bytes, sources: Sequence[SourceRef] = ()
    ) -> StoredObject:
        digest = digest_of(body)
        await self._blobs.put_new(file.key, body)
        await append_in_order(
            self._log,
            file.session_id,
            OBJECT_PLACED,
            written_payload(file.lane, file.relative, digest, sources),
        )
        return StoredObject(key=file.key, digest=digest)

    async def read(self, file: VfsFile) -> bytes | None:
        return await self._blobs.get(file.key)

    async def list_lane(
        self, tenant_id: TenantId, session_id: SessionId, lane: Lane
    ) -> Sequence[LaneEntry]:
        return await self._blobs.list_prefix(lane_prefix(tenant_id, session_id, lane))


class UnconfiguredSessionVfs:
    """The VFS when no bucket is wired behind it: every call refuses.

    Not a fallback that lets the work proceed. A Session whose durable filesystem is
    unwired must not appear to have one -- an agent told its artifact was stored, with
    nothing behind the call, has been told a lie it cannot detect and the user is
    promised a file that does not exist. A refused write is recoverable; a silently
    dropped artifact is the failure this whole layer exists to remove.

    Reading and listing refuse for the same reason rather than answering empty. "There
    is no bucket" and "the lane holds nothing" are different facts, and a listing that
    reported zero objects against an unconfigured store would read as a Session that
    produced nothing.
    """

    def _refuse(self) -> VfsUnconfigured:
        return VfsUnconfigured(
            "no object bucket is wired behind the Session VFS, so a lane has nowhere"
            " to live"
        )

    async def place(
        self, file: VfsFile, body: bytes, sources: Sequence[SourceRef] = ()
    ) -> StoredObject:
        raise self._refuse()

    async def read(self, file: VfsFile) -> bytes | None:
        raise self._refuse()

    async def list_lane(
        self, tenant_id: TenantId, session_id: SessionId, lane: Lane
    ) -> Sequence[LaneEntry]:
        raise self._refuse()
