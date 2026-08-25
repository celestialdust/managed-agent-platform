"""Files with an identity of their own: their names, their digests, and their size cap.

Everything this store holds is a tenant's *upload*: an input supplied before any Session
exists, so it cannot be addressed under a Session and is reached by the identifier the
upload gave back.

**A file the agent wrote is no longer one of these, and `produced_in_session_id` is what
is left of when it was.** Ship-out used to mint an upload for each produced file and
stamp that column with the Session; it now places the file into that Session's
`artifacts` lane instead, because an upload is keyed by a *filename* and a filename
cannot hold a separator -- so a deliverable like `report/fig1.png` had nowhere to go
under the old arrangement and was simply never offered. The column and the `scope_id`
filter over it are kept rather than dropped: rows written before the change carry real
values and are still a tenant's files, and dropping a column is a migration over live
objects for no gain. Nothing writes it any more, so on a deployment with no such rows
that filter matches nothing.

Nothing here is revisable. A key is derived from an identifier issued once and the
object under it is written once -- which is the same property the `artifacts` lane has,
and is why a produced file could move there without becoming a different kind of thing.

Nothing here imports a web framework, a database driver or an object-store client. The
size limit, the name rules, the object key and the order the two writes happen in are
all decisions, and a decision that can only be exercised through an HTTP request is a
decision nobody can test cheaply. The storage this module needs is a Protocol declared
beside the two operations and satisfied at the composition root.

The identifier type is declared here rather than in core.ids because a file id is not
part of the domain those ids describe — no Session, Turn or Grant refers to one — and
because that module has a single writer. If a later feature makes a file id a domain
identity, the move is one NewType.
"""

import hashlib
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Final, NewType, Protocol
from uuid import UUID, uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from managed_agent.core.ids import SessionId, TenantId

FileId = NewType("FileId", UUID)
"""The identifier an upload hands back. Opaque, and the only way a file is addressed."""


def new_file_id() -> FileId:
    return FileId(uuid4())


UPLOAD_KEY_ROOT: Final[str] = "uploads"
DEFAULT_MEDIA_TYPE: Final[str] = "application/octet-stream"

DEFAULT_MAX_UPLOAD_BYTES: Final[int] = 100 * 1024 * 1024
"""What a tenant may send when nothing is configured.

A hundred mebibytes rather than something larger because the body is held whole in
memory while it is hashed and written, so the figure is also the per-upload memory cost
of a concurrent request. Raising it past what a process can hold means writing the
object in parts, which is a different write path and not a bigger number here.
"""

MAX_CONFIGURABLE_UPLOAD_BYTES: Final[int] = 5 * 1024 * 1024 * 1024
"""The largest object one PUT can create. A limit configured above this would admit an
upload the object store refuses at the end of the transfer, which is the worst moment to
find out — so the ceiling is enforced when the configuration is read, not when a tenant
hits it."""

MAX_UPLOAD_ENV_VAR: Final[str] = "MAP_MAX_UPLOAD_BYTES"

_MAX_FILENAME_BYTES: Final[int] = 255


class UploadTooLarge(ValueError):
    """More bytes arrived than the configured limit admits.

    Carries what was configured and how much had arrived when the read stopped. The
    second number is a floor rather than the body's real length: the read stops at the
    first byte past the limit and never learns how much more was coming, which is the
    whole reason it stops there.
    """

    def __init__(self, limit_bytes: int, received_at_least_bytes: int) -> None:
        super().__init__(
            f"{received_at_least_bytes} bytes arrived; "
            f"the configured limit is {limit_bytes}"
        )
        self.limit_bytes = limit_bytes
        self.received_at_least_bytes = received_at_least_bytes


class InvalidUploadFilename(ValueError):
    """A name that would not survive being written into a header or a mount path."""


@dataclass(frozen=True, slots=True)
class UploadSizeLimit:
    """The configured ceiling on one upload, proven in range at construction."""

    byte_length: int

    def __post_init__(self) -> None:
        if not 1 <= self.byte_length <= MAX_CONFIGURABLE_UPLOAD_BYTES:
            raise ValueError(
                f"upload limit {self.byte_length} is outside "
                f"1..{MAX_CONFIGURABLE_UPLOAD_BYTES}"
            )

    def admits(self, byte_length: int) -> bool:
        return byte_length <= self.byte_length


def upload_limit_from_env(source: Mapping[str, str] | None = None) -> UploadSizeLimit:
    """Read the configured limit, or fall back to the default.

    A value that is present and unusable raises rather than falling back. A fallback
    there would mean a deployment that meant to cap uploads at one mebibyte and mistyped
    it would run at the default instead, and nothing would say so until a bill arrived.
    """
    raw = (source if source is not None else os.environ).get(MAX_UPLOAD_ENV_VAR)
    if raw is None:
        return UploadSizeLimit(DEFAULT_MAX_UPLOAD_BYTES)
    try:
        return UploadSizeLimit(int(raw))
    except ValueError as exc:
        raise ValueError(f"{MAX_UPLOAD_ENV_VAR}={raw!r}: {exc}") from exc


def _safe_filename(raw: str) -> str:
    """Parse the name a tenant sent, or refuse it.

    Every rule is about somewhere the name is later written verbatim. It is the leaf a
    mount path is built from later, where a separator or a dot-dot component names a
    different file than the one uploaded; it is stored, where a NUL truncates it for
    whatever reads it next; and it goes into a Content-Disposition header, where a quote
    or a newline ends the field early and starts an attacker-chosen one.

    What it deliberately does **not** refuse is a non-ASCII name, and that has a cost
    paid elsewhere: header values are latin-1 on the wire, so a name holding an em-dash
    or a CJK character cannot be written into `filename="..."` at all. Refusing such a
    name here would be the cheaper fix and the wrong one — a platform that cannot
    accept a name in a non-Latin script is broken, not safe.
    `content_disposition` in the download route carries it instead, which is where
    the encoding actually belongs.

    A name is a leaf, never a path: a caller that sent "a/b.txt" is refused rather than
    silently given "b.txt", because a caller whose name was quietly changed cannot match
    what it uploaded against what it later lists.
    """
    if not raw:
        raise InvalidUploadFilename("a filename is not empty")
    if len(raw.encode("utf-8")) > _MAX_FILENAME_BYTES:
        raise InvalidUploadFilename(
            f"{len(raw.encode('utf-8'))} bytes; a filename holds at most "
            f"{_MAX_FILENAME_BYTES}"
        )
    if raw in {".", ".."}:
        raise InvalidUploadFilename(f"{raw!r} names a directory, not a file")
    for forbidden in ("/", "\\", '"'):
        if forbidden in raw:
            raise InvalidUploadFilename(f"a filename carries no {forbidden!r}")
    if any(character < " " or character == "\x7f" for character in raw):
        raise InvalidUploadFilename("a filename carries no control character")
    return raw


UploadFilename = Annotated[str, AfterValidator(_safe_filename)]
"""A tenant-supplied name, proven safe to write into a path and a header."""


def parse_upload_filename(raw: str) -> UploadFilename:
    """Parse a name that did not arrive through a model boundary.

    Raises InvalidUploadFilename rather than a pydantic ValidationError, because the
    caller is a route holding a multipart part's filename and it has to turn the refusal
    into one named refusal of its own rather than into a validation report about a
    field.
    """
    return _safe_filename(raw)


def content_digest(body: bytes) -> str:
    """SHA-256, lowercase hex, over the octets exactly as they will be stored.

    Recorded at upload and checked again on every download, which is what makes
    "byte-for-byte" a property the platform enforces rather than one it hopes the object
    store preserved.
    """
    return hashlib.sha256(body).hexdigest()


def upload_object_key(tenant_id: TenantId, file_id: FileId) -> str:
    """Where one uploaded file's bytes live.

    The tenant is a path segment rather than only a column, so one tenant's uploads are
    a listable, deletable prefix without a reference count nobody is keeping — and so a
    key built for one tenant can never address another's object even if the row were
    wrong.
    """
    return f"{UPLOAD_KEY_ROOT}/{tenant_id}/{file_id}"


class UploadedFile(BaseModel):
    """One uploaded file, as the platform recorded it. Never rewritten.

    The bytes behind it cannot change: the key is derived from an identifier issued
    once, and nothing writes that key a second time. So the length and the hash recorded
    here describe those bytes for as long as the row exists, and a download that
    disagrees with them is a fault rather than a newer version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: FileId
    tenant_id: TenantId
    filename: UploadFilename
    media_type: str = Field(min_length=1, max_length=255)
    byte_length: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    produced_in_session_id: SessionId | None = None
    """The Session whose agent wrote this file, or None when a tenant uploaded it.

    The one field that separates "what I sent" from "what the agent made", and it has to
    be written at the moment the row is: this table refuses UPDATE by trigger
    (migration 0010), so a listing that later wanted to tell them apart could not
    backfill it. Record it here or lose the fact.

    Nullable rather than two tables, because a caller holds one identifier space and
    downloads through one route whichever it is. Two tables would make `GET
    /v1/files/{id}/content` two lookups over one id, and an id that resolved in neither
    indistinguishable from an id that resolved in the wrong one.
    """

    @property
    def object_key(self) -> str:
        return upload_object_key(self.tenant_id, self.id)


class UploadedFileNotFound(LookupError):
    """No file with that identifier is readable by the tenant that asked.

    Deliberately one condition rather than two. A file this tenant never uploaded and a
    file another tenant did are the same answer, because two answers would let a caller
    walk the identifier space and learn which ids exist somewhere on the platform.
    """


class UploadedFileVanished(RuntimeError):
    """The row exists and the object behind it does not."""


class UploadedFileCorrupt(RuntimeError):
    """The stored bytes do not hash to what was recorded when they were written."""


class UploadStorageUnconfigured(RuntimeError):
    """This process has no object bucket, so no file can be stored or read."""


class UploadedFileInUse(Exception):
    """A Session that has not stopped still holds this file, so it may not be deleted.

    Carries how many hold it rather than which ones. The count is what a caller acts on
    -- end those Sessions and retry -- while a list of ids grows with the tenant's
    history, and a refusal whose body has no ceiling is one no client can budget for.
    """

    def __init__(self, session_count: int) -> None:
        super().__init__(f"sessions still holding this file: {session_count}")
        self.session_count = session_count


@dataclass(frozen=True, slots=True)
class FileWindow:
    """Which of a tenant's files a listing asks for, and where its page starts.

    The two anchors are mutually exclusive, and that is enforced here rather than at the
    route so no caller of the store can ask for a page starting in two places. Naming
    both is a caller who has lost track of which way they are walking; picking one for
    them hands back a page they did not ask for.

    An anchor is a file id and not an opaque cursor, because that is the shape the
    listing publishes. It costs a lookup that has to happen anyway -- an id has to be
    proven to belong to this tenant before it can name a position -- and it lets a
    caller resume from any row it is already holding.
    """

    scope_id: SessionId | None = None
    """Only files this Session's agent produced. None asks for every file.

    The filter reads `produced_in_session_id`, which nothing writes any more -- ship-out
    places a produced file into the Session's `artifacts` lane instead. So this narrows
    to rows written before that change, and finds nothing where there are none. What a
    Session produced today is answered by its Event Log, one `output.produced` per file
    carrying the path that downloads it.
    """

    after_id: FileId | None = None
    before_id: FileId | None = None

    def __post_init__(self) -> None:
        if self.after_id is not None and self.before_id is not None:
            raise ValueError(
                "a page starts after one file or before another, never both"
            )

    @property
    def walks_backward(self) -> bool:
        """Whether this page is taken from the anchor towards newer files."""
        return self.before_id is not None


@dataclass(frozen=True, slots=True)
class FilePage:
    """One page of a tenant's files, and whether the walk has further to go.

    `has_more` is about the direction this page was taken in, which is the only thing
    one extra row can answer: a forward page says whether older files remain, a backward
    one whether newer files do. Read as "more files exist somewhere" it would be wrong
    on every backward page that started at the newest file.
    """

    files: tuple[UploadedFile, ...]
    has_more: bool


class UploadedFileStorage(Protocol):
    """The two places one uploaded file lives, behind one port.

    Bytes and row are one port rather than two because they are only ever written
    together and in one order, and a caller holding two ports could do them in the other
    one. The tenant is a parameter of the lookup rather than a filter the caller applies
    afterwards, so an implementation that ignored it would fail the cross-tenant case
    rather than leaving every caller to remember the comparison.

    The media type is carried on the row and not on the object. One place to read it
    from means a download cannot be served as one type by the store and another by the
    record.
    """

    async def write(self, key: str, body: bytes) -> None: ...

    async def read_bytes(self, key: str) -> bytes | None:
        """The object's bytes, or None if the key holds nothing."""

    async def record(self, file: UploadedFile) -> None: ...

    async def lookup(
        self, tenant_id: TenantId, file_id: FileId
    ) -> UploadedFile | None: ...

    async def page(
        self, tenant_id: TenantId, window: FileWindow, limit: int
    ) -> tuple[UploadedFile, ...]:
        """At most `limit` of this tenant's undeleted files, walking from the anchor.

        Ordered newest first, except on a backward page, which comes back oldest first
        because it walks away from its anchor. Handing rows back in walk order rather
        than in presentation order is what lets the caller drop the one extra row it
        asked for from the right end; reversing here would hide which end that is.

        A deleted file is absent from the rows. An anchor naming one is still a
        position, so a walk begun before a deletion carries on across it.
        """

    async def erase(self, key: str) -> None:
        """Remove one object. A key that holds nothing is success, not a fault --
        there is nothing left to remove and no caller can act on the difference."""

    async def record_deletion_unless_held(
        self, tenant_id: TenantId, file_id: FileId
    ) -> int:
        """Write the tombstone and count the Sessions holding the file, atomically.

        Returns how many of this tenant's Sessions hold it and have not stopped. Zero
        means the tombstone is committed; any other number means the write was rolled
        back and nothing was recorded.

        **One operation and not two, because the order of the two is the guard.** A
        caller holding a count and a write could take the count first, which leaves a
        Session created between them holding a file the platform has since called
        deleted. Writing first and counting second means a create that commits before
        the count is seen by it, and one that starts after the write commits sees the
        tombstone -- so the two cannot both miss each other except in the one
        interleaving named on `FileStore.delete`, which this shape cannot reach either.

        A command that returns a value, which is the exception to command-query
        separation this codebase makes deliberately: it is a compare-and-set, and the
        thing a caller has to know is whether the set happened.

        Idempotent on the write. A repeat absorbs the conflict and leaves the first
        moment, so a caller retrying a timeout is answered about when the file stopped
        being usable rather than about their retry.
        """

    async def deletion_recorded(self, file_id: FileId) -> bool:
        """Whether a tombstone exists for this file."""


async def read_within_limit(
    chunks: AsyncIterator[bytes], limit: UploadSizeLimit
) -> bytes:
    """Buffer an upload, refusing as soon as it crosses the limit.

    It stops at the first byte past the limit rather than reading to the end and
    measuring, which bounds what one request costs at the limit plus one chunk however
    much the client intended to send. The consequence is that the refusal cannot say how
    large the body really was — only that it was at least this large — and that is the
    trade being made.

    This bounds what the platform *holds in this process*, not what it *receives*: by
    the time a route calls this, the multipart parser has already taken the whole body
    off the network and spooled it. Bounding the transfer itself is the ingress's job,
    and it is configured there rather than pretended at here.
    """
    buffered = bytearray()
    async for chunk in chunks:
        buffered += chunk
        if not limit.admits(len(buffered)):
            raise UploadTooLarge(limit.byte_length, len(buffered))
    return bytes(buffered)


class FileStore:
    """Storing an uploaded file, reading one back with its bytes checked, deleting one.

    Command and query are separate: store() writes and returns what it wrote, fetch()
    reads and writes nothing. Neither reads or repairs on behalf of the other.

    delete() is the third kind and the only one that removes anything. What it removes
    is the bytes; the metadata row survives it, because the id outlives the object --
    the argument is in full on delete() itself.
    """

    def __init__(self, storage: UploadedFileStorage, limit: UploadSizeLimit) -> None:
        self._storage = storage
        self._limit = limit

    @property
    def limit(self) -> UploadSizeLimit:
        return self._limit

    async def store(
        self,
        *,
        tenant_id: TenantId,
        filename: UploadFilename,
        media_type: str,
        chunks: AsyncIterator[bytes],
        produced_in_session_id: SessionId | None = None,
    ) -> UploadedFile:
        """Take one upload and make it addressable.

        Bytes first, row second: a crash between them leaves an object nothing points
        at, which costs storage. The other order leaves an identifier the tenant is
        holding that fetches nothing, which costs the tenant their file.

        `produced_in_session_id` names the Session whose agent wrote this file, and
        defaults to None because the upload route is now the only caller that reaches
        here -- the answer there is "no Session; a tenant sent it". Ship-out was the
        second caller and places into the `artifacts` lane instead, so nothing in the
        running system passes this today. It is kept because the column and the
        `scope_id` filter over it are kept, and a filter over a column no code can write
        would be a worse thing to leave behind than an argument nobody currently sends.

        One consequence worth stating where it changed: the configured size limit is no
        longer a second ceiling on a produced file. It bounded one while ship-out came
        through here; a produced file is now bounded only by the Turn's own output
        budget.
        """
        body = await read_within_limit(chunks, self._limit)
        record = UploadedFile(
            id=new_file_id(),
            tenant_id=tenant_id,
            filename=filename,
            media_type=media_type,
            byte_length=len(body),
            content_sha256=content_digest(body),
            produced_in_session_id=produced_in_session_id,
        )
        await self._storage.write(record.object_key, body)
        await self._storage.record(record)
        return record

    async def describe(self, *, tenant_id: TenantId, file_id: FileId) -> UploadedFile:
        """One uploaded file's row, without reading its bytes.

        Exists so a caller that only needs to know a file is there does not pay for a
        100 MiB download to find out. The distinction matters at exactly one point: a
        Session naming files it wants attached is checked at creation, where the tenant
        is still on the connection, and the bytes are not needed until a pod exists.

        No hash check, because there is nothing to check -- and that is the honest
        boundary of what this proves. It says the row is there and this tenant owns it;
        `fetch` is what says the object behind it still reproduces the digest recorded
        at upload.
        """
        record = await self._storage.lookup(tenant_id, file_id)
        if record is None:
            raise UploadedFileNotFound(str(file_id))
        return record

    async def fetch(
        self, *, tenant_id: TenantId, file_id: FileId
    ) -> tuple[UploadedFile, bytes]:
        """One uploaded file and its bytes, checked against the hash recorded for them.

        The row is read first and the key is built from it, so the bytes fetched are the
        bytes that row describes and never an object addressed from the request. The
        hash check is what "byte-for-byte" means here: the read either reproduces the
        digest taken at upload, or it fails rather than handing back a plausible file.
        """
        record = await self._storage.lookup(tenant_id, file_id)
        if record is None:
            raise UploadedFileNotFound(str(file_id))
        body = await self._storage.read_bytes(record.object_key)
        if body is None:
            raise UploadedFileVanished(f"{record.object_key} holds nothing")
        actual = content_digest(body)
        if actual != record.content_sha256:
            raise UploadedFileCorrupt(
                f"{record.object_key} hashes to {actual}; "
                f"{record.content_sha256} was recorded"
            )
        return record, body

    async def page(
        self, *, tenant_id: TenantId, window: FileWindow, limit: int
    ) -> FilePage:
        """One page of what this tenant has, and whether the walk goes further.

        A deleted file is absent, because this listing answers "what do I have" and a
        deleted file is not had. It is the one place a deletion is invisible rather than
        reported: every read addressed at an id says the file was deleted, and only the
        collection leaves it out.

        The anchor is resolved before the page is read, and an anchor this tenant does
        not hold is refused rather than answered with an empty page. An empty page looks
        like a finished walk, so a caller who mistyped an id -- or held one belonging to
        somebody else -- would conclude the collection was exhausted and stop.

        One row more than was asked for is read, and that row is the whole answer to "is
        there another page". It is dropped from the end the walk was heading towards,
        which is why the storage hands rows back walked rather than presented.
        """
        for anchor in (window.after_id, window.before_id):
            if anchor is None:
                continue
            if await self._storage.lookup(tenant_id, anchor) is None:
                raise UploadedFileNotFound(str(anchor))
        walked = await self._storage.page(tenant_id, window, limit + 1)
        shown = walked[:limit]
        return FilePage(
            files=tuple(reversed(shown)) if window.walks_backward else shown,
            has_more=len(walked) > limit,
        )

    async def deletion_recorded(self, *, file_id: FileId) -> bool:
        """Whether this file's bytes have been deleted.

        Asked separately rather than folded into `describe`, and the reason is what
        breaks if it is folded in. Two callers outside this module -- Session creation
        and a Session's resource listing -- catch only the absent-row refusal, so a
        `describe` that raised on a tombstone would reach both as an unhandled
        exception: a stopped Session whose file was later deleted would make its own
        resource listing fail. The row is still there and `describe` still answers about
        the row; whoever needs the second fact asks for it.

        Not tenant-scoped, and it does not need to be. Every caller has already resolved
        the row under a tenant, so an id arriving here is one that tenant holds. Scoping
        it again would mean the tombstone carrying a tenant column -- a copy of
        `uploaded_file`'s own fact, free to disagree with it.
        """
        return await self._storage.deletion_recorded(file_id)

    async def delete(self, *, tenant_id: TenantId, file_id: FileId) -> None:
        """Delete the bytes, keep the row, and record when it stopped being usable.

        **The row stays because the id outlives the object.** A Session's creation event
        names the files it was created with, so dropping the row would leave that
        history naming an id that resolves to nothing -- which reads as the platform
        having lost the data rather than as the tenant having deleted it. Those two look
        identical from outside and mean opposite things.

        **Refused while a Session that has not stopped holds the file.** The placement
        path reads these bytes into the pod, so a Session mid-run whose file vanished
        fails at its next placement with the cause three layers from the symptom. A
        tenant honouring a deletion request can end a Session; they cannot debug a
        placement failure.

        **The tombstone is written before the count is taken, and both happen inside one
        transaction that rolls back if the count is not zero.** Counting first leaves a
        window in which a Session created after the count still commits before the
        tombstone, and the file it holds is deleted underneath it. Writing first closes
        that direction: a create committing before the count is seen by it, and a create
        starting after the transaction commits reads the tombstone and is refused.

        **What that does not close, stated plainly rather than left to be discovered.**
        Under READ COMMITTED the two transactions never conflict -- this one writes the
        tombstone and reads the log, session creation writes the log and reads the
        tombstone -- so a create whose tombstone read precedes this write AND whose
        commit follows this count is seen by neither. That is textbook write skew, and
        it survives because session creation is three transactions rather than one: no
        lock this side can take is still held when the create commits. Closing it needs
        the create made atomic, or SERIALIZABLE on both sides. Until then the residual
        is one interleaving of a few milliseconds, and the failure it produces is a
        Session that cannot place its pod -- see
        `tests/control/test_file_listing_and_deletion.py`, which reproduces it
        deliberately so that closing the race breaks a test rather than going unnoticed.

        **Object last, after the transaction commits.** A rolled-back tombstone must not
        leave the bytes gone, which is the one outcome nothing can recover from. Erasing
        first would delete the bytes while every read still said the file was live: a
        download answering a fault rather than "deleted", with nothing recording that a
        deletion was attempted and no retry able to finish it. This way round, a crash
        between the commit and the erase leaves a file that already reads as deleted and
        bytes still in the bucket -- and the caller is not told otherwise, because this
        returns only after the erase. They see a fault, retry, the tombstone insert
        absorbs its own conflict, and the retry erases the object. That state is safe
        and it converges.

        Idempotent by construction, so the caller who retried a timeout is answered the
        same as the caller who did not.
        """
        record = await self._storage.lookup(tenant_id, file_id)
        if record is None:
            raise UploadedFileNotFound(str(file_id))
        holding = await self._storage.record_deletion_unless_held(tenant_id, file_id)
        if holding:
            raise UploadedFileInUse(holding)
        await self._storage.erase(record.object_key)


UPLOAD_BUCKET_ENV_VAR: Final[str] = "MAP_OBJECT_BUCKET"
"""The one bucket this platform writes. Read at the composition root only."""


class NoUploadBucket:
    """The storage a process gets when no object bucket is configured.

    Every method raises, so an unconfigured deployment refuses uploads at the first
    request with the missing variable's name in the message. The alternative — reading
    the variable at start-up and raising — would take the whole control plane down over
    a resource only one surface uses, and would make `build()` unusable for the ten
    files that call it with only a database URL. Named rather than written as a lambda
    so that the gap is greppable and a reader of the wiring can see it is deliberate;
    this is the same shape `NoPodTransport` uses for a process with no pod runner.
    """

    async def write(self, key: str, body: bytes) -> None:
        raise UploadStorageUnconfigured(f"{UPLOAD_BUCKET_ENV_VAR} is not set")

    async def read_bytes(self, key: str) -> bytes | None:
        raise UploadStorageUnconfigured(f"{UPLOAD_BUCKET_ENV_VAR} is not set")

    async def record(self, file: UploadedFile) -> None:
        raise UploadStorageUnconfigured(f"{UPLOAD_BUCKET_ENV_VAR} is not set")

    async def lookup(self, tenant_id: TenantId, file_id: FileId) -> UploadedFile | None:
        raise UploadStorageUnconfigured(f"{UPLOAD_BUCKET_ENV_VAR} is not set")

    async def page(
        self, tenant_id: TenantId, window: FileWindow, limit: int
    ) -> tuple[UploadedFile, ...]:
        raise UploadStorageUnconfigured(f"{UPLOAD_BUCKET_ENV_VAR} is not set")

    async def erase(self, key: str) -> None:
        raise UploadStorageUnconfigured(f"{UPLOAD_BUCKET_ENV_VAR} is not set")

    async def record_deletion_unless_held(
        self, tenant_id: TenantId, file_id: FileId
    ) -> int:
        raise UploadStorageUnconfigured(f"{UPLOAD_BUCKET_ENV_VAR} is not set")

    async def deletion_recorded(self, file_id: FileId) -> bool:
        raise UploadStorageUnconfigured(f"{UPLOAD_BUCKET_ENV_VAR} is not set")


def unconfigured_file_store() -> FileStore:
    """A `FileStore` that refuses everything, for a process with no bucket."""
    return FileStore(NoUploadBucket(), UploadSizeLimit(DEFAULT_MAX_UPLOAD_BYTES))
