"""The five routes of the file resource: upload, list, read, download, delete.

Upload is multipart because that is what an upload of one file is: the bytes and the
name they were given travel together in one request, and a name carried in a header
beside a raw body would be a second convention for the same thing.

**A delete removes the bytes and keeps the row**, and the whole argument for that is on
`FileStore.delete`. What it means at this surface is the part a caller sees: after a
delete both reads of that id answer 410 rather than 404, because a tenant who deleted a
file to honour a deletion request needs the platform to say the deletion happened -- a
404 leaves them unable to tell it from an id they mistyped. The listing is the one place
a deleted file is simply absent: it answers "what do I have", and a deleted file is not
had.

**The listing pages unlike every other collection here**, by `after_id`/`before_id` and
`first_id`/`last_id`/`has_more` rather than by the opaque `page` cursor the Sessions and
Skills listings take. That inconsistency is the published surface's own and is copied
deliberately: a client generated from it walks this collection by those names and could
not walk it at all if this one route paged our way.

The refusals here are values rather than exceptions, in the envelope every other refusal
on this surface uses. Reaching a file that is not yours and reaching one that never
existed produce the identical body, so an identifier space cannot be probed for what
exists.

This module owns who may ask and how an answer is shaped. What may be stored, how large
it may be and whether the bytes came back intact are decided in control.file_store,
which needs no HTTP request to exercise.
"""

from collections.abc import AsyncIterator
from typing import Annotated, Final, Literal
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.files.store import (
    DEFAULT_MEDIA_TYPE,
    FileId,
    FileWindow,
    InvalidUploadFilename,
    UploadedFile,
    UploadedFileCorrupt,
    UploadedFileInUse,
    UploadedFileNotFound,
    UploadedFileVanished,
    UploadStorageUnconfigured,
    UploadTooLarge,
    parse_upload_filename,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import SessionId, TenantId

router = APIRouter(tags=["files"])

READ_CHUNK_BYTES: Final[int] = 1024 * 1024
"""How much of an upload is pulled at a time. It is also how far past the
configured limit a refused upload can be buffered before it is refused."""

REASON_UPLOAD_TOO_LARGE: Final[str] = "upload_too_large"
REASON_FILENAME_INVALID: Final[str] = "filename_invalid"
REASON_FILE_NOT_FOUND: Final[str] = "file_not_found"
REASON_FILE_UNREADABLE: Final[str] = "file_unreadable"
REASON_STORAGE_UNCONFIGURED: Final[str] = "storage_unconfigured"
"""Named in `detail` because no published code says which of these it was.

Four are emitted here and `REASON_FILE_NOT_FOUND` is emitted only by
`control/api/routes/resources.py`, which imports it rather than respelling it -- a
missing file is a coded refusal on this surface now that `file.not_found` exists, and
stays a `reason` there because that route is reporting a row the platform itself lost. A
caller branches on the code and then on this.
"""

REASON_TWO_ANCHORS: Final[str] = "page_anchors_conflict"
"""A listing naming both `after_id` and `before_id`. See `list_files`."""

DEFAULT_FILE_PAGE_SIZE: Final[int] = 20
MAX_FILE_PAGE_SIZE: Final[int] = 1000
"""How many files one page carries by default and at most.

Both are the figures the published listing states, rather than the 25 and 100 the
Sessions and Skills listings here use. A client generated from that surface sends
`limit=1000` and expects it to be served, and a lower ceiling would refuse a request the
contract admits -- so the two numbers are copied even though they make this collection
page differently from its neighbours.
"""


class UploadedFileView(BaseModel):
    """What an upload hands back, and what the file is addressed by afterwards."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: FileId
    filename: str
    media_type: str
    byte_length: int
    content_sha256: str


class FileScope(BaseModel):
    """The Session a produced file belongs to. Null on a file a tenant uploaded.

    An object carrying the kind beside the id rather than a bare session id, because it
    is the other half of the `scope_id` filter: a caller reading a page tells a file the
    agent wrote from one the tenant sent by whether this is here, and does not have to
    know that a null means the second.

    **Null on every row a current deployment writes.** Ship-out no longer mints an
    upload for a produced file; it places one into the Session's `artifacts` lane, so
    only rows written before that change carry a scope. The field stays in the published
    shape because those rows still list, and because withdrawing a field from a response
    breaks a caller that reads it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SessionId
    type: Literal["session"] = "session"


class FileMetadata(BaseModel):
    """One file's recorded facts, without its bytes.

    The field names are this platform's own, and that is a deliberate departure from the
    published shape, which spells the middle two `mime_type` and `size_bytes`. They
    match what `POST /v1/files` hands back and what `GET /v1/sessions/{id}/resources`
    lists, so a caller holding all three responses of one family sees one name per fact.
    Matching the reference here would give them two names for one fact across three
    routes, which is a worse trade than one field a generated client maps once.

    `downloadable` is always true, and it is carried anyway because the published shape
    carries it. There is no answer this platform gives where it would be false: a
    deleted file is absent from the listing and answers 410 when read by id, so no
    caller ever holds metadata describing bytes that are gone. If a future state can be
    reached where a row is readable and its object is not, this is the field that
    reports it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: FileId
    type: Literal["file"] = "file"
    filename: str
    media_type: str
    byte_length: int
    content_sha256: str
    downloadable: bool = True
    scope: FileScope | None = None


class FilePageView(BaseModel):
    """One page of a tenant's files, in the shape the published listing takes.

    `first_id` and `last_id` are null on an empty page rather than absent, so a caller
    that walked to the end reads two fields that say there was nothing left to anchor on
    -- and reads them in the same place it read them on every earlier page.

    `has_more` answers the direction the page was taken in: forward, whether older files
    remain; backward, whether newer ones do. One extra row cannot answer anything wider
    than that, and a field that claimed to would be wrong on every backward page taken
    from the newest file.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: tuple[FileMetadata, ...]
    first_id: FileId | None
    last_id: FileId | None
    has_more: bool


class DeletedFile(BaseModel):
    """What a delete hands back: the id that went, and that it was a file.

    No timestamp and no flag, because the published shape carries neither. The moment is
    recorded in the store -- a deletion is a fact this platform keeps -- and what a
    caller needs from the response is that the id they named is the id that went.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: FileId
    type: Literal["file_deleted"] = "file_deleted"


def _metadata(record: UploadedFile) -> FileMetadata:
    """One row as both the listing and the metadata read publish it.

    One function so the two routes cannot describe the same file differently -- a caller
    that read a page and then read one of its rows by id would otherwise be able to see
    the two disagree.
    """
    return FileMetadata(
        id=record.id,
        filename=record.filename,
        media_type=record.media_type,
        byte_length=record.byte_length,
        content_sha256=record.content_sha256,
        scope=(
            None
            if record.produced_in_session_id is None
            else FileScope(id=record.produced_in_session_id)
        ),
    )


def content_disposition(filename: str) -> str:
    """`attachment` plus the name, in the one form an HTTP header can carry it.

    Header values are encoded latin-1 on the wire, and the parse at upload admits any
    non-control character — so a name holding an em-dash or a CJK character cannot go
    into `filename="..."` at all: the server raises UnicodeEncodeError while encoding
    the header and the response never leaves. Measured, not reasoned about.

    So the name always travels percent-encoded in `filename*`, which RFC 6266 defines
    for exactly this. The quoted `filename=` is emitted beside it only for an ASCII
    name, as the fallback for a client that reads only that form; building it by
    concatenation is safe only because the parse at upload removed every quote,
    backslash and control character.
    """
    quoted = quote(filename, safe="")
    if filename.isascii():
        return f"attachment; filename=\"{filename}\"; filename*=UTF-8''{quoted}"
    return f"attachment; filename*=UTF-8''{quoted}"


async def _chunks_of(upload: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await upload.read(READ_CHUNK_BYTES):
        yield chunk


@router.post(
    "/files",
    status_code=status.HTTP_201_CREATED,
    response_model=UploadedFileView,
    responses={
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def upload_a_file(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    file: Annotated[UploadFile, File()],
) -> UploadedFileView | JSONResponse:
    """Store one file and hand back the identifier it is addressed by from now on.

    No Session is named and none needs to exist. That is the point of the resource: the
    file is uploaded first and a Session that will read it is created afterwards.
    """
    store = platform_from_request(request).file_store
    try:
        filename = parse_upload_filename(file.filename or "")
    except InvalidUploadFilename as bad:
        return refuse(
            ErrorCode.REQUEST_INVALID, str(bad), reason=REASON_FILENAME_INVALID
        )
    try:
        record = await store.store(
            tenant_id=tenant_id,
            filename=filename,
            media_type=file.content_type or DEFAULT_MEDIA_TYPE,
            chunks=_chunks_of(file),
        )
    except UploadTooLarge as refused:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            f"this upload exceeds the configured maximum of "
            f"{refused.limit_bytes} bytes",
            reason=REASON_UPLOAD_TOO_LARGE,
            max_bytes=refused.limit_bytes,
            received_at_least_bytes=refused.received_at_least_bytes,
        )
    except UploadStorageUnconfigured as unconfigured:
        return refuse(
            ErrorCode.INTERNAL,
            str(unconfigured),
            reason=REASON_STORAGE_UNCONFIGURED,
        )
    return UploadedFileView(
        id=record.id,
        filename=record.filename,
        media_type=record.media_type,
        byte_length=record.byte_length,
        content_sha256=record.content_sha256,
    )


@router.get(
    "/files/{file_id}/content",
    response_class=Response,
    responses={
        STATUS_FOR[ErrorCode.FILE_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.FILE_DELETED]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def download_file_content(
    file_id: FileId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> Response:
    """The bytes exactly as they were uploaded, or a refusal.

    Served as an attachment and marked unsniffable because the media type is whatever
    the uploading caller declared: rendered inline, a file uploaded as HTML would run
    script on this platform's origin.

    **The deletion is checked after ownership and before the bytes, and both halves of
    that order matter.** Checking it first would answer 410 for another tenant's deleted
    file, which tells a caller that an id exists somewhere -- exactly the probe the
    identical not-found bodies exist to prevent. Checking it after the fetch would be
    too late: by then the object is gone, so `fetch` raises the vanished-object fault
    and the caller is told the platform lost their file rather than that they deleted
    it.

    The price is that the row is read twice, once to prove ownership and once inside
    `fetch` to build the object key. The alternative -- fetch first, and read the
    tombstone only when the object turns out to be missing -- costs one read instead of
    two and buys a download that still serves bytes for a file `GET /v1/files/{file_id}`
    already answers 410 for, for as long as a failed erase leaves the object in place.
    Two reads of one file disagreeing about whether it exists is worse than a second
    SELECT.
    """
    store = platform_from_request(request).file_store
    try:
        await store.describe(tenant_id=tenant_id, file_id=file_id)
        if await store.deletion_recorded(file_id=file_id):
            return refuse(
                ErrorCode.FILE_DELETED,
                "that file was deleted; its bytes are gone and this identifier "
                "serves none",
                file_id=str(file_id),
            )
        record, body = await store.fetch(tenant_id=tenant_id, file_id=file_id)
    except UploadedFileNotFound:
        return refuse(
            ErrorCode.FILE_NOT_FOUND,
            "no file with that identifier is readable by this tenant",
            file_id=str(file_id),
        )
    except UploadStorageUnconfigured as unconfigured:
        return refuse(
            ErrorCode.INTERNAL,
            str(unconfigured),
            reason=REASON_STORAGE_UNCONFIGURED,
            file_id=str(file_id),
        )
    except (UploadedFileVanished, UploadedFileCorrupt) as broken:
        return refuse(
            ErrorCode.INTERNAL,
            str(broken),
            reason=REASON_FILE_UNREADABLE,
            file_id=str(file_id),
        )
    return Response(
        content=body,
        media_type=record.media_type,
        headers={
            "content-disposition": content_disposition(record.filename),
            "x-content-type-options": "nosniff",
        },
    )


@router.get(
    "/files",
    response_model=FilePageView,
    responses={
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def list_files(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    scope_id: SessionId | None = None,
    after_id: FileId | None = None,
    before_id: FileId | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_FILE_PAGE_SIZE)] = DEFAULT_FILE_PAGE_SIZE,
) -> FilePageView | JSONResponse:
    """One page of the files this tenant has, newest first.

    A deleted file is absent, and this is the only route where a deletion is silent
    rather than reported. The collection answers "what do I have"; a deleted file is not
    had, and a row here saying otherwise would make the listing the one place a tenant
    could still see what they asked to be rid of.

    `scope_id` narrows the page to rows carrying that Session in `scope`. It is a filter
    and not a sub-collection: such a file is addressed by the same identifier space and
    downloaded through the same route as an uploaded one, so it is the same listing with
    a predicate rather than a listing of its own. It finds only files produced before
    ship-out moved to the `artifacts` lane, for the reason `FileScope` above gives. What
    a Session produced today is in its Event Log, one `output.produced` per file, and
    downloads from `GET /v1/sessions/{id}/artifacts/{path}`.

    An anchor names a position rather than carrying one. That costs a lookup -- the id
    has to be proven to be this tenant's before it can name anything -- and an id that
    is not is refused rather than answered with an empty page, because an empty page
    looks like a walk that finished. A caller who mistyped an id would otherwise
    conclude the collection was exhausted.

    An anchor may name a deleted file. Its row is still there, so it still marks a
    position, and a walk begun before a deletion continues across it instead of failing
    on the one row that moved.
    """
    if after_id is not None and before_id is not None:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            "a page starts after one file or before another, never both",
            reason=REASON_TWO_ANCHORS,
            after_id=str(after_id),
            before_id=str(before_id),
        )
    anchor = after_id if after_id is not None else before_id
    store = platform_from_request(request).file_store
    try:
        page = await store.page(
            tenant_id=tenant_id,
            window=FileWindow(
                scope_id=scope_id, after_id=after_id, before_id=before_id
            ),
            limit=limit,
        )
    except UploadedFileNotFound:
        return refuse(
            ErrorCode.PAGINATION_CURSOR_INVALID,
            "the file this page is anchored on is not one this tenant holds; walk "
            "from the beginning by naming neither anchor",
            anchor_id=str(anchor),
        )
    except UploadStorageUnconfigured as unconfigured:
        return refuse(
            ErrorCode.INTERNAL,
            str(unconfigured),
            reason=REASON_STORAGE_UNCONFIGURED,
        )
    return FilePageView(
        data=tuple(_metadata(record) for record in page.files),
        first_id=page.files[0].id if page.files else None,
        last_id=page.files[-1].id if page.files else None,
        has_more=page.has_more,
    )


@router.get(
    "/files/{file_id}",
    response_model=FileMetadata,
    responses={
        STATUS_FOR[ErrorCode.FILE_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.FILE_DELETED]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def read_file_metadata(
    file_id: FileId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> FileMetadata | JSONResponse:
    """One file's recorded facts, without paying for its bytes.

    Exists because the only way to learn anything about a file used to be to download
    it, which for a hundred-mebibyte upload means transferring all of it to read a
    filename. Nothing here opens the object, so nothing here can say whether the bytes
    still hash to what was recorded -- that is `GET /v1/files/{file_id}/content`'s
    answer and it costs the transfer.

    410 rather than 404 once the file is deleted. The identifier is not wrong and the
    tenant's memory of it is not wrong either: it named a file, and the file was
    deliberately removed. A 404 would leave a tenant who deleted a file to honour a
    deletion request unable to tell that it happened from a typo in the id they kept.

    Ownership is proven before the tombstone is read, so another tenant's deleted file
    is a 404 like any other id they do not hold. The reverse order would answer 410 and
    tell them an id exists somewhere on the platform.
    """
    store = platform_from_request(request).file_store
    try:
        record = await store.describe(tenant_id=tenant_id, file_id=file_id)
        if await store.deletion_recorded(file_id=file_id):
            return refuse(
                ErrorCode.FILE_DELETED,
                "that file was deleted; what survives is a record that this "
                "identifier existed, not the file it named",
                file_id=str(file_id),
            )
    except UploadedFileNotFound:
        return refuse(
            ErrorCode.FILE_NOT_FOUND,
            "no file with that identifier is readable by this tenant",
            file_id=str(file_id),
        )
    except UploadStorageUnconfigured as unconfigured:
        return refuse(
            ErrorCode.INTERNAL,
            str(unconfigured),
            reason=REASON_STORAGE_UNCONFIGURED,
            file_id=str(file_id),
        )
    return _metadata(record)


@router.delete(
    "/files/{file_id}",
    response_model=DeletedFile,
    responses={
        STATUS_FOR[ErrorCode.FILE_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.FILE_IN_USE]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def delete_a_file(
    file_id: FileId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> DeletedFile | JSONResponse:
    """Delete one file's bytes. 200 with a body, not 204.

    The published shape is `{"id": ..., "type": "file_deleted"}`, and a 204 may carry no
    body at all -- so the status follows the shape rather than the verb.

    **What is deleted is the bytes.** The metadata row survives, because a Session's
    creation event names the files it was created with and an id that resolved to
    nothing would make that history read as the platform having lost the data rather
    than as the tenant having deleted it. `FileStore.delete` carries the argument and
    the write order that makes a failure between the two steps safe.

    **Refused with 409 while a Session that has not stopped holds the file**, because
    the alternative is a route that breaks a running agent: its next pod placement fails
    on a file that is not there, three layers from anything naming this request. The
    refusal counts the Sessions rather than listing them -- the count is what a caller
    acts on, and a list has no ceiling. A stopped Session does not hold the delete: its
    history keeps the id, and the tombstone is what tells that history what happened.

    Deleting twice answers the same body twice. The tombstone is one row per file, so
    the second insert conflicts and is absorbed; a caller retrying a timeout must not be
    told the file was never there. The retry also re-attempts the object deletion, which
    is what makes a failure between the two writes something a retry finishes.
    """
    store = platform_from_request(request).file_store
    try:
        await store.delete(tenant_id=tenant_id, file_id=file_id)
    except UploadedFileNotFound:
        return refuse(
            ErrorCode.FILE_NOT_FOUND,
            "no file with that identifier is readable by this tenant",
            file_id=str(file_id),
        )
    except UploadedFileInUse as held:
        return refuse(
            ErrorCode.FILE_IN_USE,
            "sessions that have not stopped still hold this file; end them and "
            "retry, or the agent reading it fails at its next pod placement",
            file_id=str(file_id),
            sessions_holding=held.session_count,
        )
    except UploadStorageUnconfigured as unconfigured:
        return refuse(
            ErrorCode.INTERNAL,
            str(unconfigured),
            reason=REASON_STORAGE_UNCONFIGURED,
            file_id=str(file_id),
        )
    return DeletedFile(id=file_id)
