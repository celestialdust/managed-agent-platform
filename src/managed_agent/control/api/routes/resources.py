"""Listing what a Session holds, and attaching one more file to a Session.

A Session's resources are the files it was created with, plus any attached since. Both
live only in its Event Log -- `SessionRecord` has no such field and the `session` table
no such column -- so the list is a fold over the log rather than a read of a column, and
the log's order is the answer's order. Each entry carries the file's recorded facts and
none of its bytes: the bytes are already reachable at `GET /v1/files/{id}/content`, and
a second way to download them would be a second answer about one object.

**The resource id is the file id.** Nothing here mints a separate identifier, because a
separate one needs a mapping held somewhere, and that store would be a second record of
which files a Session holds -- free to disagree with the event that already says so. The
consequence is deliberate and useful: an id from this list is the id that downloads the
file.

**Two POSTs, and they answer differently on purpose.** `POST .../resources` attaches one
more file to a Session already created. `POST .../resources/{resource_id}` -- upstream's
*update a resource* -- refuses every call, because a file resource has nothing an update
could act on: `UploadedFile` is frozen, its row is never rewritten, and its object key
derives from an identifier issued once, so there is no field a caller could name. The
verb that exists elsewhere for this is for a mounted repository, whose mutable part is
the credential it is cloned with, and no credential travels with a file here.

The attach appends `session.file_attached`, so what a Session holds is the creation
event **plus** those. That is a real change from what this module used to claim, and the
claim it replaces was checkable and checked: a test asserted that no published event
type said a resource was attached, and its own docstring said adding one had to arrive
with the delivery path that makes it true. It has. What the log still guarantees is the
property that mattered -- the set only ever grows, every addition carries its own
sequence, and nothing rewrites what an earlier Turn saw. A Turn that ran before an
attach did not see that file, and the log says when.

The three earlier arguments against accepting an attach are recorded as wrong in
`gap.md` D1, measured rather than reasoned away. The load-bearing one was that an
accepted file could not be delivered: the *call site* that pushes files into a pod is
creation-only, but the *transport* is not -- `PodFilePlacement.place_file` PUTs to a
running pod's file route and needs nothing but a running pod. So delivery here has two
cases and no new machinery. A Session whose first Turn has not completed has no pod yet
and the placement path reads this event when it places one; a Session past that point
has a pod, and the bytes go down the same authenticated hop a moment later.

`mount_path` is refused rather than accepted and ignored. Upstream defaults it to
`/mnt/session/uploads/<file_id>` and this platform writes to
`/session/workspace/files/<filename>`: the pod's receiver refuses every path separator
and the volume is mounted with `subPath: files`, so any other location is not merely a
different default but physically unwritable. Accepting the field would make the platform
report a location it did not use.

Provenance for the placement boundary the third reason rests on: docs/adr/ADR-004.
"""

from typing import Annotated, Final, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from managed_agent.composition import Platform
from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.api.routes.files import (
    REASON_FILE_NOT_FOUND,
    REASON_STORAGE_UNCONFIGURED,
)
from managed_agent.control.files.attachments import (
    WORKSPACE_FILE_BUDGET_BYTES,
    FilesNotPlaceable,
)
from managed_agent.control.files.store import (
    FileId,
    UploadedFile,
    UploadedFileNotFound,
    UploadStorageUnconfigured,
)
from managed_agent.control.session.lifecycle import open_turn, whole_log
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.pod.workspace_contract import INPUT_DIR_NAME
from managed_agent.core.ports import (
    EventRecord,
    SessionNotVisible,
)
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import lifecycle, resource, turn

router = APIRouter(
    tags=["resources"],
    # On the router, so a route added here later inherits the gate rather than needing
    # whoever adds it to remember. The refusal below wants no tenant of its own and is
    # still gated by this: a caller who has not said who it is learns nothing at all,
    # which is the same order every other route in this package answers in.
    dependencies=[Depends(unauthenticated_tenant_from_header)],
)

REASON_RESOURCES_FIXED_AT_CREATION: Final[str] = "resources_fixed_at_creation"
REASON_MOUNT_PATH_FIXED: Final[str] = "mount_path_not_configurable"
REASON_FILE_BUDGET_EXHAUSTED: Final[str] = "session_file_budget_exhausted"
REASON_CREATION_RECORD_NOT_RETAINED: Final[str] = "session_created_not_retained"
REASON_CREATION_RECORD_UNREADABLE: Final[str] = "session_created_payload_unreadable"
"""The five things this module names in `detail` rather than in a code.

The published set is closed and carries no member for any of them (ADR-013), so each
travels as a `reason` under a code a caller can already branch on -- the shape
`control/api/routes/files.py` established for its own five. The two file reasons this
module also emits are imported from there rather than respelled, so one string cannot
become two spellings of one condition.
"""


class AttachedResourceView(BaseModel):
    """One file a Session holds, addressed by the id that also downloads it.

    `type` is carried even though `file` is the only kind this platform attaches. A
    caller of a resource list branches on the kind before reading kind-specific fields,
    and a list that omits the discriminator forces every caller to assume one -- an
    assumption that becomes wrong silently the first time a second kind appears.

    The digest and the length are the ones recorded at upload. They describe the bytes
    for as long as the row exists, so a caller can compare them against a download
    without asking this route for bytes it deliberately does not serve.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: FileId
    type: Literal["file"]
    filename: str
    media_type: str
    byte_length: int
    content_sha256: str


class AttachedResourceList(BaseModel):
    """Everything one Session holds, in the order it was created with.

    Wrapped in `data` rather than returned as a bare array, which is the shape every
    other list on this surface uses and the one that leaves room for a paging field
    without changing the type of the response.

    The order is the order the create call named, because that is the order the files
    are written into the pod -- a list sorted some other way here would describe a
    workspace laid out differently from the one the agent sees.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: tuple[AttachedResourceView, ...]


class _CreationFileIds(BaseModel):
    """The one field of a `session.created` payload this module reads.

    Parsed rather than picked out by hand, so the three cases separate themselves: the
    key absent is an empty tuple, a well-formed list is the ids in order, and anything
    else raises. That third case matters and is why this is a model at all -- reading
    the ids by hand makes "a list this code cannot parse" indistinguishable from "this
    Session attaches nothing", and answering the second when the first is true tells a
    tenant their files are gone.

    `extra="ignore"` because a creation payload carries eight other keys this module has
    no business in; forbidding them would refuse every real event.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    file_ids: tuple[FileId, ...] = ()


FILES_MOUNT_PATH: Final[str] = f"/session/workspace/{INPUT_DIR_NAME}"
"""The one directory an attached file can land in, as a caller would name it.

Built from `INPUT_DIR_NAME` rather than written out, because the pod's receiver resolves
the same constant and the volume is mounted with `subPath` on it. A second spelling here
would be a path this route advertised and the pod did not serve.

A caller may name this and nothing else. It is not a default this route fills in and
then ignores: the receiver refuses every path separator in a filename and the mount
makes any other directory unreachable, so accepting another value would report a
location the bytes did not go to.
"""


def _held_file_ids(events: list[EventRecord]) -> tuple[FileId, ...] | None:
    """Every file this Session holds, in the order it came to hold them, or None.

    None means the log carries no `session.created`, which is a different answer from
    the empty tuple and must not be collapsed into it. Creation appends that event
    before it writes the registry row, so a registry row exists only where the event
    once did -- its absence therefore means retention has passed over it, and reporting
    "no resources" would tell a tenant their files never existed.

    Reads the WHOLE log rather than stopping at the creation event, and that is the cost
    of letting the set grow: an attach is an event later in the same log, so a walk that
    stopped at the first match would report the set as it was at creation and call it
    current. The version this replaces stopped early and was correct while the set could
    not change.

    A second `session.created` is ignored rather than merged. Nothing appends one and
    the append path would have to be broken to produce one, so the first is the creation
    and a later one is corruption -- merging its ids would silently double a Session's
    holdings rather than leave the anomaly visible in the log.

    Raises `ValidationError` when a payload names its files in a form this module cannot
    read. Deliberately not caught here: the caller turns it into an internal refusal
    naming the Session, and swallowing it would make an unreadable payload look like an
    empty one.
    """
    held: tuple[FileId, ...] | None = None
    for event in events:
        if event.type == lifecycle.SESSION_CREATED and held is None:
            held = _CreationFileIds.model_validate(event.payload).file_ids
        elif event.type == resource.SESSION_FILE_ATTACHED and held is not None:
            # `held is not None` is required by the type, not by a runtime worry:
            # `held` is None until the creation event is met, and `(*held, ...)` on it
            # would not type-check. So there is no version of this branch without the
            # narrowing, and the behaviour it picks -- discard an attach read ahead of
            # creation -- matches `session_pods.py`, where the creation event's
            # assignment discards it and no guard is needed at all.
            attached = resource.SessionFileAttached.model_validate(event.payload)
            held = (*held, FileId(UUID(attached.file_id)))
    return held


def _view(record: UploadedFile) -> AttachedResourceView:
    return AttachedResourceView(
        id=record.id,
        type="file",
        filename=record.filename,
        media_type=record.media_type,
        byte_length=record.byte_length,
        content_sha256=record.content_sha256,
    )


@router.get(
    "/sessions/{session_id}/resources",
    response_model=AttachedResourceList,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.EVENT_RANGE_EXPIRED]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def list_resources(
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> AttachedResourceList | JSONResponse:
    """What this Session holds now: one entry per attached file, and no bytes.

    Who owns the Session is settled **before** the log is touched. The Event Log is
    keyed by Session and carries no tenant, so reading somebody else's Session succeeds
    and hands back the ids of their files. One 404 covers "no such Session" and "not
    yours", so a caller holding an id cannot learn which it is.

    A Session the registry knows and whose creation event is missing is **refused rather
    than answered empty**. Creation appends the event before it writes the registry row,
    so a row exists only where an event once did -- the event being absent therefore
    means it is no longer retained, and answering "no resources" would report a
    Session's files as never having existed.

    An id in that payload that the file store cannot resolve is this platform's fault
    and not the caller's, which is why it comes back as an internal refusal naming the
    file rather than as the invalid-request the upload surface answers a caller-supplied
    id with. The create call checked every one of these ids while the tenant was still
    on the connection, and an uploaded file's row is never rewritten, so reaching this
    means something below lost a row.

    `describe` and not `fetch`: what a resource list needs is the recorded facts, and
    fetching would download every attached file to answer a question about their names.

    The whole log, not the creation event alone. `POST .../resources` appends
    `session.file_attached`, so the set grows, and a read that stopped at creation would
    report what the Session started with as what it holds. That costs what `GET
    /v1/sessions/{id}` already pays and already names as the part that will not scale.
    """
    platform = platform_from_request(request)
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return refuse(
            ErrorCode.SESSION_NOT_FOUND,
            f"no session {session_id} is visible to this tenant",
            session_id=str(session_id),
        )
    try:
        held = _held_file_ids(await whole_log(platform.event_log_range, session_id))
    except ValidationError:
        return refuse(
            ErrorCode.INTERNAL,
            "this session's creation event names its files in a form this platform "
            "cannot read, so the set it holds cannot be reported",
            reason=REASON_CREATION_RECORD_UNREADABLE,
            session_id=str(session_id),
        )
    if held is None:
        return refuse(
            ErrorCode.EVENT_RANGE_EXPIRED,
            "this session's creation event is no longer retained, so what it was "
            "created holding cannot be read back",
            reason=REASON_CREATION_RECORD_NOT_RETAINED,
            session_id=str(session_id),
        )
    views: list[AttachedResourceView] = []
    for file_id in held:
        try:
            record = await platform.file_store.describe(
                tenant_id=tenant_id, file_id=file_id
            )
        except UploadedFileNotFound:
            return refuse(
                ErrorCode.INTERNAL,
                "this session was created holding a file the store no longer has a "
                "record of",
                reason=REASON_FILE_NOT_FOUND,
                session_id=str(session_id),
                file_id=str(file_id),
            )
        except UploadStorageUnconfigured as unconfigured:
            return refuse(
                ErrorCode.INTERNAL,
                str(unconfigured),
                reason=REASON_STORAGE_UNCONFIGURED,
                session_id=str(session_id),
                file_id=str(file_id),
            )
        views.append(_view(record))
    return AttachedResourceList(data=tuple(views))


class AttachResource(BaseModel):
    """The body of an attach: which file, and where it must not go.

    `type` is required and has one member. A caller that omits it is refused rather than
    defaulted to `file`, because the field is upstream's discriminator over three kinds
    and this platform serves one -- a caller whose client sends `memory_store` should
    learn that here rather than have it read as a file.

    `mount_path` is optional and, when given, must be the one directory this platform
    writes to. Compared rather than ignored for the reason the module docstring gives:
    the pod's receiver cannot write anywhere else, so any other value is a location the
    response would claim and the bytes would not occupy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_id: FileId
    type: Literal["file"]
    mount_path: str | None = Field(default=None)


@router.post(
    "/sessions/{session_id}/resources",
    status_code=status.HTTP_201_CREATED,
    response_model=AttachedResourceView,
    responses={
        STATUS_FOR[ErrorCode.SESSION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.SESSION_TURN_IN_FLIGHT]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.FILE_DELETED]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.EVENT_RANGE_EXPIRED]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.TURN_UNDELIVERABLE]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def attach_resource(
    body: AttachResource,
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> AttachedResourceView | JSONResponse:
    """Attach one more file to a Session, and record that it now holds it.

    **The bytes go down before the event goes in, and that order is deliberate.** The
    append is the commit: after it, the list route says this Session holds the file and
    the placement path will deliver it. If the push fails first, nothing was appended
    and the caller has a refusal it can retry. The other order leaves a Session whose
    record names a file its pod does not have, and nothing would ever push it again --
    placement runs once per Session. The residue of the order chosen is a file in a
    workspace the log does not name, which no prompt points at and a retry overwrites
    with identical bytes.

    **Delivery has two cases and the branch mirrors the placement path's own.** A
    Session that has completed no Turn has no pod; `FirstTurnPlacement` places one and
    reads this event when it does, so pushing here would be a push to nothing. A Session
    past that point has a pod and the file goes down now. The predicate is
    `turn.completed` because that is exactly what the placement path branches on --
    deriving it another way here would be a second answer to "will placement run again"
    and free to disagree.

    **Refused unless the Session would accept a Turn.** A stopped Session accepts no
    event at all; a suspended one accepts no Turn either, so no future Turn will ever
    read the file. Both are 409 naming the state rather than a 201 for a document
    nothing will open. This is where the reaped-Session refusal lives and it needs no
    case of its own -- reaping suspends.

    **Refused while a Turn is in flight.** The runtime already holds its prompt, so a
    file landing mid-Turn may or may not be read and no record afterwards could say
    which. The caller interrupts and attaches after, which is the order archiving
    requires for the same reason.

    **The byte ledger is folded before any bytes are read.** Every file the Session
    holds is priced from its `uploaded_file` row, and that sum plus the incoming length
    is compared against the workspace budget before a single object is fetched. Pricing
    from the row is exact rather than approximate: `byte_length` and `content_sha256`
    are taken from one buffer in one expression at upload, and a fetch whose bytes
    disagree with the digest raises -- so bytes that could differ from the recorded
    length cannot be summed, because the read that would reveal the difference refuses
    first. The placement call still accumulates as it reads, which is what catches a row
    whose object vanished between the fold and the push.

    **A filename already in the workspace is refused, not overwritten.** The receiver
    renames atomically into one flat directory, so honouring a collision would replace
    the earlier file and no record anywhere would name the moment. Compared on the
    filename and not the file id, because two different uploads can carry one name and
    it is the name that collides on disk.

    Ownership is settled before the log is touched, for the reason the list route gives:
    the Event Log is keyed by Session and holds no tenant, so an attach aimed at another
    tenant's id would otherwise succeed and change their Session.
    """
    if body.mount_path is not None and body.mount_path != FILES_MOUNT_PATH:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            f"an attached file is written to {FILES_MOUNT_PATH} and nowhere else, so "
            "mount_path can only name that directory or be omitted",
            reason=REASON_MOUNT_PATH_FIXED,
            mount_path=body.mount_path,
        )
    platform = platform_from_request(request)
    try:
        await platform.session_registry.fetch(session_id, tenant_id)
    except SessionNotVisible:
        return refuse(
            ErrorCode.SESSION_NOT_FOUND,
            f"no session {session_id} is visible to this tenant",
            session_id=str(session_id),
        )
    events = await whole_log(platform.event_log_range, session_id)
    try:
        held = _held_file_ids(events)
    except ValidationError:
        return refuse(
            ErrorCode.INTERNAL,
            "this session's creation event names its files in a form this platform "
            "cannot read, so nothing can be added to a set it cannot read",
            reason=REASON_CREATION_RECORD_UNREADABLE,
            session_id=str(session_id),
        )
    if held is None:
        return refuse(
            ErrorCode.EVENT_RANGE_EXPIRED,
            "this session's creation event is no longer retained, so what it holds "
            "cannot be read and nothing can be added to it",
            reason=REASON_CREATION_RECORD_NOT_RETAINED,
            session_id=str(session_id),
        )
    # `project` cannot raise here: `_held_file_ids` returned a tuple, which means it met
    # the creation event, which is the one thing `project` refuses a log for.
    state, _ = project(events)
    if state is not SessionState.RUNNING:
        return refuse(
            ErrorCode.SESSION_NOT_ACCEPTING_TURNS,
            f"session {session_id} is {state.value} and will run no further Turn, so a "
            "file attached now would never be read",
            session_id=str(session_id),
            state=state.value,
        )
    if (running := open_turn(events)) is not None:
        return refuse(
            ErrorCode.SESSION_TURN_IN_FLIGHT,
            "a Turn is running on this session; interrupt it before attaching a file",
            session_id=str(session_id),
            turn_id=str(running),
        )
    ledger = 0
    taken: set[str] = set()
    for file_id in held:
        priced = await _describe_or_refuse(platform, tenant_id, session_id, file_id)
        if isinstance(priced, JSONResponse):
            return priced
        ledger += priced.byte_length
        taken.add(priced.filename)
    if await platform.file_store.deletion_recorded(file_id=body.file_id):
        return refuse(
            ErrorCode.FILE_DELETED,
            "that file was deleted, so attaching it would put a document in this "
            "session's workspace that the platform has recorded as gone",
            session_id=str(session_id),
            file_id=str(body.file_id),
        )
    try:
        incoming = await platform.file_store.describe(
            tenant_id=tenant_id, file_id=body.file_id
        )
    except UploadedFileNotFound:
        return refuse(
            ErrorCode.FILE_NOT_FOUND,
            "this tenant holds no file of that id, so there is nothing to attach",
            reason=REASON_FILE_NOT_FOUND,
            session_id=str(session_id),
            file_id=str(body.file_id),
        )
    except UploadStorageUnconfigured as unconfigured:
        return refuse(
            ErrorCode.INTERNAL,
            str(unconfigured),
            reason=REASON_STORAGE_UNCONFIGURED,
            session_id=str(session_id),
            file_id=str(body.file_id),
        )
    if incoming.filename in taken:
        return refuse(
            ErrorCode.RESOURCE_FILENAME_ATTACHED,
            f"this session already holds a file named {incoming.filename!r}, and the "
            "pod's workspace is one flat directory, so attaching this one would "
            "replace it",
            session_id=str(session_id),
            file_id=str(body.file_id),
            filename=incoming.filename,
        )
    if ledger + incoming.byte_length > WORKSPACE_FILE_BUDGET_BYTES:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            f"this session holds {ledger} bytes of files and this one is "
            f"{incoming.byte_length} more, over the {WORKSPACE_FILE_BUDGET_BYTES} its "
            "pod's workspace can hold beside what the agent writes",
            reason=REASON_FILE_BUDGET_EXHAUSTED,
            session_id=str(session_id),
            file_id=str(body.file_id),
        )
    if any(event.type == turn.TURN_COMPLETED for event in events):
        try:
            await platform.session_attachments.place_for(
                session_id, tenant_id, (body.file_id,), ledger
            )
        except FilesNotPlaceable as undelivered:
            return refuse(
                ErrorCode.TURN_UNDELIVERABLE,
                f"this session's pod would not take the file: {undelivered}",
                session_id=str(session_id),
                file_id=str(body.file_id),
            )
    await platform.event_log_append.append(
        session_id,
        resource.SESSION_FILE_ATTACHED,
        resource.SessionFileAttached(file_id=str(body.file_id)).model_dump(),
    )
    return _view(incoming)


async def _describe_or_refuse(
    platform: Platform,
    tenant_id: TenantId,
    session_id: SessionId,
    file_id: FileId,
) -> UploadedFile | JSONResponse:
    """One held file's recorded facts, or the refusal the attach route answers with.

    Both failures are this platform's rather than the caller's, which is why both are
    internal refusals naming the file: the ids being priced came out of this Session's
    own log, every one was checked while a tenant was on the connection, and an
    `uploaded_file` row is never rewritten. Reaching either means something below lost a
    row or a bucket.

    Returns the response rather than raising, so the route reads as a straight line of
    refusals in the order it checks them -- the shape `list_resources` uses for the same
    two conditions.
    """
    try:
        return await platform.file_store.describe(tenant_id=tenant_id, file_id=file_id)
    except UploadedFileNotFound:
        return refuse(
            ErrorCode.INTERNAL,
            "this session holds a file the store no longer has a record of, so what it "
            "already holds cannot be priced",
            reason=REASON_FILE_NOT_FOUND,
            session_id=str(session_id),
            file_id=str(file_id),
        )
    except UploadStorageUnconfigured as unconfigured:
        return refuse(
            ErrorCode.INTERNAL,
            str(unconfigured),
            reason=REASON_STORAGE_UNCONFIGURED,
            session_id=str(session_id),
            file_id=str(file_id),
        )


@router.post(
    "/sessions/{session_id}/resources/{resource_id}",
    responses={STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope}},
)
async def refuse_to_change_a_resource(
    session_id: SessionId, resource_id: FileId
) -> JSONResponse:
    """Refuse, and name the create call as the place a Session's files are chosen.

    The same refusal for every caller, every Session and every resource id, because
    nothing about it depends on any of the three: the module docstring gives the three
    reasons, and none of them is a fact about a particular Session. So nothing is read
    before answering -- no registry row, no log, no file. A refusal that varied would be
    a way to ask whether a Session id or a resource id exists, from a verb that does
    nothing.

    Both path segments are still parsed as the identifiers they name. A malformed one is
    a different mistake -- the caller has not addressed anything -- and the framework
    answers it by naming the field, the way every other route in this package does.

    Nothing is appended. The Event Log is append-only and `session.created` already says
    what this Session holds, so a second record here would be a second answer to one
    question. Its absence is what makes the list route above the whole truth.
    """
    return refuse(
        ErrorCode.REQUEST_INVALID,
        "a Session's resources are fixed when it is created: name every file in the "
        "create call's file_ids, or create a new Session to change the set",
        reason=REASON_RESOURCES_FIXED_AT_CREATION,
        session_id=str(session_id),
        resource_id=str(resource_id),
    )
