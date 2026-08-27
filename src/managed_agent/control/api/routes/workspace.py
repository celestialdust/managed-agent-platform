"""Browsing a Session's live workspace: what is there, and one file's bytes.

**Two routes, because there are two questions and only one of them has an answer the
Event Log already holds.** `control/api/routes/artifacts.py` serves what a Turn
*declared* -- sealed, digestible, enumerated by one `output.produced` event per file.
This serves what is *on disk*: mutable, live, and enumerable because a directory is
enumerable. ADR-038 decides they are separate doors and says why one route with a flag
would be worse -- the artifacts lane's seal is a promise about the past, and a route
that sometimes keeps it and sometimes does not keeps it never.

**This does not contradict `artifacts.py`'s refusal to carry a listing, and the two are
adjacent enough to be read as though it did.** That refusal is about the *declared* set
having exactly one authoritative enumeration: a second answer to "what did this Session
make" would be free to disagree with the Event Log and unable to say which Turn made
what. The listing here enumerates a directory rather than a declaration, and answers a
question the log cannot answer at all -- what is on disk that the agent never declared.
Nothing here is a second answer to anything the log already answers.

**Everything is reachable through this route: `.env`, `.git`, `node_modules`, every
dotfile.** That is the route rather than a leak in it, and it is precisely why this is a
separate door with its own authorization rather than a widening of the deliverable one.
Do not add a filter here. A file the agent wrote and did not declare is the one thing
this route exists to reach, and the recovery case ADR-038 was accepted for -- a failed
or interrupted Turn whose work survives on the mount -- is entirely made of such files.

**Why this reads the file system and not the bucket, which would have been easier.**
Under ADR-036 the workspace lives on an S3 Files file system synchronised to the
platform bucket, and the control plane already talks to that bucket. Reading the bucket
prefix would need no mount and no new containment. It was rejected because S3 Files'
export is not prompt, and a route that served the bucket while calling itself live would
be lying in three separate ways a caller could not detect:

- Export is **batched behind a window of write inactivity**, not behind a fixed delay:
  a modified file is exported only once writes to it have stopped for sixty seconds. The
  documented example is a file appended to every thirty seconds for five minutes, whose
  export begins at minute six -- so the lag is the length of the write activity, not a
  bound on it. An agent appending to a log across a Turn is exactly that shape, and a
  Turn is when somebody most wants to see what the workspace holds.
- Export failures are **per file and invisible from the bucket side**, surfaced as an
  extended attribute on the file system plus a CloudWatch metric. A file that failed to
  export (`PathTooLong`, `S3AccessDenied`, and the rest) is simply absent from the
  bucket, and a bucket reader cannot tell that from a file that was never written.
- A directory rename is **instant on the file system and gradual in the bucket**: S3
  has no atomic rename, so every object is copied to its new key and the old one is
  deleted, and "both directories will be visible on the S3 bucket until the rename is
  fully completed". A listing taken from the bucket in that window describes a tree that
  never existed on disk.

None of this is a rule left undeclared in `deploy/terraform/session_vfs.tf`. A
synchronisation configuration has two kinds of rule -- import and expiration -- and no
export rule exists to write: export is automatic and unconditional. It is simply not
prompt, and automatic is what makes that easy to mistake for prompt.

So the bucket is a durable copy of the workspace and is not the workspace. Only the file
system is authoritative, which is why this route needs the control plane to have its own
read-only mount of the same file system rather than a second reader of the bucket.

**The mount root is read from the environment rather than taken off `Platform`, and that
is a compromise rather than a design.** A mount is a process-level deployment fact, like
the pod manifest path `pod_runner_from_environment` reads, and not a port with an
adapter behind it -- but the composition root is still where a deployment fact should be
turned into a value, and it is not here because this route was built in a slice that
does not own that file. Moving it is a field on `Platform` and a changed `Depends`; the
containment and the parse below do not move with it.

**Path containment here is this module's own and cannot be borrowed from the lane
grammar.** `parse_relative_path` governs what may be composed into an object key, and it
is the reason `artifacts.py` needs no traversal defence of its own. It cannot serve this
route for two independent reasons. It refuses a leading dot, so `.env` and `.git/config`
-- the files this route exists for -- compose to no path at all under it. And it refuses
`..` as a *substring*, which it can afford because no lane path wants to spell `a..b`,
while a workspace holds whatever the agent named and a real file must not be unreachable
because of how it is spelled. Widening it is not the way to get this: it is shared by
every lane, and the properties it holds for a composed key are the ones this route does
not need. So `parse_workspace_path` below is per-segment, and containment is finished by
resolving the composed path and requiring the result to be inside the Session's own
subtree -- which is the only check that sees a symlink, since a symlink is invisible to
every lexical rule and is a live escape on a file system in a way it never is in a
bucket.

**Cross-tenant isolation is the composed path's shape rather than a check made here.**
The subtree is `<mount>/<tenant>/<session>`, composed from the tenant on the *request*
and the Session id in the *path*, so a caller naming another tenant's Session reaches a
directory under their own tenant segment, which does not exist. There is no arrangement
of inputs that reads another tenant's files, because the tenant component is never taken
from the caller's claim about which Session this is. That is the same rule, and the same
reason, as `artifacts.py` -- and it is why the containment check below resolves against
the *Session's* subtree and not against the mount root: every tenant is under one root,
so a check written against the root would resolve a symlink into a neighbour's
workspace, find it comfortably inside the mount, and serve it.

The consequence, deliberately: a Session that does not exist and one that exists with
nothing at that path answer identically, and so does a path that resolved outside the
subtree. Separating them turns this route into a probe -- for which Session ids are
real, and for what is on the file system outside the caller's own tree.
"""

from __future__ import annotations

import os
import stat as stat_module
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.refusals import Refusal, refuse
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.api.routes.files import content_disposition
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import SessionId, TenantId

router = APIRouter(
    tags=["workspace"],
    dependencies=[Depends(unauthenticated_tenant_from_header)],
)

WORKSPACE_MOUNT_ENV_VAR: Final[str] = "MAP_SESSION_WORKSPACE_MOUNT"
"""Where this process has the Session VFS mounted, if it has it mounted at all.

**It must name the directory that holds `<tenant_id>/<session_id>` subtrees directly.**
Mounted through the access point Session pods use -- `fsap-091907e0af59c1b50`, rooted at
`/workspaces` -- that is the mount point itself, because the access point's root is
already the `workspaces/` key prefix. Mounted without that access point this must
include the `workspaces` segment, since the whole bucket is then in view.

Stating it here rather than composing a literal `workspaces/` below is what keeps the
two mount styles from needing two versions of this route, and it leaves the choice with
the deployment that made it.
"""

REASON_WORKSPACE_PATH_INVALID: Final[str] = "workspace_path_invalid"
REASON_WORKSPACE_NOT_MOUNTED: Final[str] = "workspace_not_mounted"
REASON_WORKSPACE_MOUNT_MISCONFIGURED: Final[str] = "workspace_mount_misconfigured"
REASON_NOT_A_DIRECTORY: Final[str] = "workspace_path_not_a_directory"
REASON_NOT_A_FILE: Final[str] = "workspace_path_not_a_file"
"""The five conditions named in `detail` rather than in a code of their own.

The published code set is closed (ADR-013) and carries no member for any of them, so
each travels as a `reason` under a code a caller can already branch on -- the shape
`control/api/routes/files.py` established for its own five and `artifacts.py` reuses.
"""

MAX_WORKSPACE_PATH_BYTES: Final[int] = 4096
"""Longest workspace-relative path this route will compose.

Linux `PATH_MAX`, less the room the mount root and the Session's own prefix take. An
over-long path cannot name a file that exists -- the file system would have refused to
create it -- so refusing it here costs nothing and keeps a caller-controlled string from
being handed to `os` at whatever length it arrived.
"""

MAX_SEGMENT_BYTES: Final[int] = 255
"""Longest single path component. Linux `NAME_MAX`, and also the limit S3 Files
documents for a path component it can synchronise, so a longer one names nothing."""


class WorkspacePathInvalid(Exception):
    """The path is not one this route can contain, so it composes to nothing.

    Carries the path and never the composed one: the path is text the caller sent, so
    echoing it discloses nothing, while a composed path carries the tenant's own id and
    this message reaches a service log. `VfsPathInvalid` withholds the same thing for
    the same reason.
    """

    def __init__(self, relative: str) -> None:
        super().__init__(f"{relative!r} is not a path inside a Session's workspace")
        self.relative: str = relative


def parse_workspace_path(relative: str) -> tuple[str, ...]:
    """The path's segments, once it is a path that cannot leave a subtree.

    Raises `WorkspacePathInvalid` when it is not. The empty string parses to no segments
    and names the workspace root, which is what a caller listing a Session for the first
    time asks for.

    Per **segment** and not per substring, which is the whole difference from
    `parse_relative_path` and is deliberate rather than a relaxation. `..` is refused as
    a complete component because that is the component the kernel resolves upward; a
    file the agent named `a..b` is refused by nothing here, because it is a file that
    exists and this route's promise is that what is on disk is reachable.

    What is refused, and why each one is a real escape rather than tidiness:

    - A **leading separator**, which makes the path absolute and would discard the
      Session's prefix entirely when joined -- `Path("/mnt/x") / "/etc/passwd"` is
      `/etc/passwd`, silently, with no error anywhere.
    - A `..` **component**, which resolves above the subtree.
    - A `.` **component** and an **empty** one, neither of which addresses anything the
      other spellings do not, and both of which make two different strings name one file
      -- so a rule applied to one spelling is escaped by the other.
    - A **NUL**, which terminates the string for the C call underneath and would open a
      different file than the one the rest of the path named.
    - A component or a path **longer than the file system can hold**, which names
      nothing and is refused before it reaches `os`.

    Symlinks are not this function's business and cannot be: nothing about the *text* of
    a path says whether a component of it is a link. `_contained_in` below is what sees
    them, by resolving.
    """
    if len(relative.encode()) > MAX_WORKSPACE_PATH_BYTES or "\x00" in relative:
        raise WorkspacePathInvalid(relative)
    if not relative:
        return ()
    if relative.startswith("/"):
        raise WorkspacePathInvalid(relative)
    segments = tuple(relative.split("/"))
    for segment in segments:
        if (
            not segment
            or segment in (".", "..")
            or len(segment.encode()) > MAX_SEGMENT_BYTES
        ):
            raise WorkspacePathInvalid(relative)
    return segments


class WorkspaceMountMisconfigured(ValueError):
    """The mount is named by configuration and the name is not usable.

    A `ValueError` because that is what it is, and its own type because the dependency
    below has to catch exactly this and turn it into a refusal -- catching `ValueError`
    there would swallow whatever else one day raises one and report it as a bad mount.

    Carries the offending value for a log. It is a deployment's own configuration rather
    than anything a tenant sent, so it is safe there and useless without it -- but it
    does not go on the wire, for the reason the closed code set exists: a caller learns
    that this platform is misconfigured, not how.
    """

    def __init__(self, raw: str) -> None:
        super().__init__(f"{WORKSPACE_MOUNT_ENV_VAR}={raw!r} is not an absolute path")
        self.raw: str = raw


def workspace_root_in(env: Mapping[str, str]) -> Path | None:
    """The mount root this configuration names, or None when it names none.

    A value that is present and unusable raises rather than falling back to None. A
    deployment that meant to name a mount and mistyped it would otherwise serve every
    request as though it had no mount at all, which reads to a tenant as "your work is
    gone" -- and nothing would say the configuration was at fault.
    `upload_limit_from_env` refuses a mistyped value for the same reason.

    A relative path is refused rather than resolved, because it would resolve against
    whatever directory the process happened to start in. The same configuration would
    then read a different tree depending on how the process was launched, and the
    containment below it would be defending a subtree of somewhere nobody chose.

    Separate from the dependency below only because FastAPI reads a dependency's
    signature to decide what to take off the request, and a `Mapping` parameter there
    would be resolved as a request body on a GET. This is the half a test can call.
    """
    raw = env.get(WORKSPACE_MOUNT_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    root = Path(raw)
    if not root.is_absolute():
        raise WorkspaceMountMisconfigured(raw)
    return root


def mounted_workspace_root() -> Path | None:
    """The mount root for this process, read fresh per request.

    Per request rather than at import, so a test can substitute one through
    `dependency_overrides` and so a process whose configuration is corrected does not
    have to be rebuilt to notice. The read is one `dict` lookup.

    There is no default, and that is the safety rather than an omission. Any default --
    a conventional mount path, the working directory -- is the failure that survives
    deployment: a control plane whose mount never came up would read some unrelated
    directory and report its contents to a tenant as their agent's workspace.

    A misconfigured mount becomes a `Refusal` here rather than travelling out as the
    `ValueError` it is. A dependency cannot return a response, and an exception that
    escapes one is rendered by the framework at a status and in a body the published set
    never agreed to -- so it would reach a caller as the one refusal on this surface
    shaped unlike every other. It is deliberately *not* reported as an unmounted
    workspace: there is a mount configured, it is wrong, and telling an operator "there
    is no mount" sends them to add one that is already there.
    """
    try:
        return workspace_root_in(os.environ)
    except WorkspaceMountMisconfigured as broken:
        raise Refusal(
            ErrorCode.INTERNAL,
            "this deployment's session workspace mount is misconfigured",
            reason=REASON_WORKSPACE_MOUNT_MISCONFIGURED,
        ) from broken


EntryKind = Literal["file", "directory", "symlink", "other"]
"""What a directory entry may be, named once because two places decide it.

The classifier below produces one of these and the published model declares them, and a
literal written out twice is a pair that agrees until somebody adds a fifth kind to one.
"""


class WorkspaceEntry(BaseModel):
    """One thing found in a directory: what it is called, what it is, and how big.

    `kind` distinguishes a symlink from what it points at rather than resolving it, and
    that is load-bearing rather than descriptive. A link may point out of the subtree,
    in which case this route will not serve it -- so reporting it as the file it targets
    would describe something the caller cannot read, and reporting that file's size
    would disclose a fact about a file outside the subtree without ever serving it.

    `byte_length` is null for everything that is not a regular file. A directory's own
    `st_size` is the size of its directory entry, which reads as the size of what is
    inside it and is not; a symlink's is the length of its target path, which is a fact
    about the link's text.

    `modified_at` is here because the case ADR-038 was accepted for is diagnosing a Turn
    that failed, and the first question that asks is which files it had got to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: EntryKind
    byte_length: int | None
    modified_at: datetime


class WorkspaceListing(BaseModel):
    """What one directory held at the moment it was read, and nothing more than that.

    **This is not a digest and must not be mistaken for one.** It reads a mutable file
    system, so two listings of one directory may differ and neither is a record of what
    the Session produced. The path is echoed back because a caller holding several of
    these needs to know which directory each describes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    entries: list[WorkspaceEntry]


def _subtree_of(mount: Path, tenant_id: TenantId, session_id: SessionId) -> Path:
    """Where one Session's workspace sits under the mount: tenant, then Session.

    The same two segments a Session pod is handed as its `subPath`, in the same order,
    so the directory this reads and the directory the agent writes are one directory. In
    bucket keys that is `workspaces/<tenant>/<session>/`; the `workspaces` segment is
    the access point's own root and therefore sits *above* the mount, which is why it is
    not composed here -- see `WORKSPACE_MOUNT_ENV_VAR`.

    **Not `lane_prefix`.** That composes `sessions/<tenant>/<session>/<lane>/`, where
    the durable lanes live in the bucket, and that is a different tree from this one:
    the `sessions/` prefix is root-owned and a Session pod cannot write it. Reusing it
    here would compose a path that exists and is not the workspace.

    Safe to interpolate without a parse because both ids are `NewType`s over `UUID`,
    which FastAPI parses out of the request before this is reached. Neither can carry a
    separator or a dot segment, so neither can change the shape of the result.
    """
    return mount / str(tenant_id) / str(session_id)


def _contained_in(subtree: Path, segments: tuple[str, ...]) -> Path | None:
    """The composed path with every symlink resolved, if that lands inside the subtree.

    None when it does not, and None is the answer for a broken link and a resolution
    loop as well -- all three mean the same thing to a caller, which is that nothing
    readable is there.

    **This is the check `parse_workspace_path` cannot make.** The path that names a
    symlink is perfectly well-formed; the escape is on disk, not in the request. So the
    composed path is resolved -- which follows every link in every component -- and the
    result is required to be under the subtree. A link inside the subtree resolves to
    somewhere still inside it and is served; a link out of it is not.

    The subtree is resolved too, so the comparison is between two paths that have both
    had their links followed. Comparing a resolved target against an unresolved root
    would fail for every request the moment the mount path itself contained a link.

    **What this does not close, stated rather than implied.** The resolution and the
    read that follows it are separate calls, so a component swapped for a symlink in
    between is not defended against here. Closing it means resolving the path a
    component at a time through `openat`, re-deciding containment at each link -- a path
    resolver written in this module, for a race whose only runner is the Session's own
    agent, against a request it cannot see coming.
    """
    base = subtree.resolve()
    try:
        resolved = base.joinpath(*segments).resolve()
    except OSError:
        return None
    return resolved if resolved.is_relative_to(base) else None


def _kind_and_size(entry: os.DirEntry[str]) -> tuple[EntryKind, int | None]:
    """What a directory entry is, decided without following it anywhere."""
    if entry.is_symlink():
        return "symlink", None
    if entry.is_dir(follow_symlinks=False):
        return "directory", None
    if entry.is_file(follow_symlinks=False):
        return "file", entry.stat(follow_symlinks=False).st_size
    return "other", None


def _entries_in(directory: Path) -> list[WorkspaceEntry]:
    """Everything in that directory, sorted by name.

    Sorted so two reads of an unchanged directory agree: `scandir` returns whatever
    order the file system holds, and a listing that reordered itself between requests
    would read as the tree having changed when it had not.

    An entry that disappears between the scan and its `stat` is dropped rather than
    reported. This reads a tree the agent is writing, and a file that was deleted while
    this listing was being taken was not there at the moment it was read -- which is
    exactly what this listing claims to describe and no more.
    """
    found: list[WorkspaceEntry] = []
    with os.scandir(directory) as scan:
        for entry in scan:
            try:
                kind, size = _kind_and_size(entry)
                modified = entry.stat(follow_symlinks=False).st_mtime
            except OSError:
                continue
            found.append(
                WorkspaceEntry(
                    name=entry.name,
                    kind=kind,
                    byte_length=size,
                    modified_at=datetime.fromtimestamp(modified, tz=UTC),
                )
            )
    return sorted(found, key=lambda found_entry: found_entry.name)


def _absent(session_id: SessionId) -> JSONResponse:
    """The one refusal for everything that is not there, whatever the reason.

    Absent, another tenant's, never existed, or resolved outside the subtree -- one body
    for all four. The module docstring says why they are not separated: any distinction
    makes this route a probe, either for which Session ids are real or for what lies
    outside the caller's own tree.
    """
    return refuse(
        ErrorCode.FILE_NOT_FOUND,
        "nothing is at that path in a session workspace readable by this tenant",
        session_id=str(session_id),
    )


def _unmounted(session_id: SessionId) -> JSONResponse:
    """This process has no workspace mount, which is not the same as an empty workspace.

    A 404 here would tell a tenant their agent's work is gone -- a lie they cannot
    detect and would act on by re-running a Turn that already succeeded. `artifacts.py`
    refuses an unconfigured object store in the same shape and for the same reason.
    """
    return refuse(
        ErrorCode.INTERNAL,
        "no session workspace is mounted in this process, so nothing can be read",
        reason=REASON_WORKSPACE_NOT_MOUNTED,
        session_id=str(session_id),
    )


def _malformed(session_id: SessionId, bad: WorkspacePathInvalid) -> JSONResponse:
    """A path this route cannot contain, told apart from a path that holds nothing.

    Different facts and answered differently on purpose: a caller whose path was
    malformed has something to fix, and one told "not found" would go looking for a file
    that was never the problem.
    """
    return refuse(
        ErrorCode.REQUEST_INVALID,
        str(bad),
        reason=REASON_WORKSPACE_PATH_INVALID,
        session_id=str(session_id),
    )


@router.get(
    "/sessions/{session_id}/workspace",
    response_model=WorkspaceListing,
    responses={
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.FILE_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def list_a_workspace_directory(
    session_id: SessionId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    mount: Annotated[Path | None, Depends(mounted_workspace_root)],
    path: Annotated[str, Query()] = "",
) -> WorkspaceListing | JSONResponse:
    """What that directory held when it was read. The root when `path` is empty.

    **The directory travels as a query parameter while the file below travels in the
    path, and the asymmetry is deliberate.** A `:path` parameter is greedy to the end of
    the URL, so a suffix that told the two operations apart -- the `/content` shape
    `files.py` uses -- cannot exist after one. Putting the discriminator in front
    instead would shadow a real directory of that name. So the collection is addressed
    as a collection and a member by its path, which also gives the commonest request,
    the workspace root, a URL with nothing in it to get wrong.

    **Two routes rather than one that returns either shape.** The bodies are a JSON
    listing and an octet stream, and one operation carrying both cannot be described in
    the published document -- which is what a client is generated from, so a caller
    would be generated code that parses JSON and is handed a file.

    A path that names a file is refused rather than redirected. The caller asked the
    wrong door about a path that exists, and answering "not found" would send them
    looking for a file they can see in the listing they just read.
    """
    if mount is None:
        return _unmounted(session_id)
    try:
        segments = parse_workspace_path(path)
    except WorkspacePathInvalid as bad:
        return _malformed(session_id, bad)
    target = _contained_in(_subtree_of(mount, tenant_id, session_id), segments)
    if target is None or not target.exists():
        return _absent(session_id)
    if not target.is_dir():
        return refuse(
            ErrorCode.REQUEST_INVALID,
            "that path is not a directory, so it has no listing",
            reason=REASON_NOT_A_DIRECTORY,
            session_id=str(session_id),
        )
    return WorkspaceListing(path=path, entries=_entries_in(target))


@router.get(
    "/sessions/{session_id}/workspace/{path:path}",
    response_class=Response,
    responses={
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.FILE_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def read_a_workspace_file(
    session_id: SessionId,
    path: str,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    mount: Annotated[Path | None, Depends(mounted_workspace_root)],
) -> Response:
    """The file's bytes as they are on disk right now, or a refusal.

    Served as an attachment and marked unsniffable, for the reason `artifacts.py` gives
    and more strongly: nothing in a workspace was ever declared as anything at all, so a
    file the model happened to name `.html` is script that would otherwise run on this
    platform's origin. The `filename` in the disposition is the last segment, because
    that is what a browser writes to disk and a browser cannot be asked to create a
    directory.

    **Streamed from the file rather than read into a buffer**, which is the one place
    this route must differ from the artifacts one. That lane holds what ship-out placed
    and ship-out is budgeted; a workspace holds whatever the agent wrote, including a
    checkpoint or an archive that would take the serving process down if a request could
    make it resident. Streaming is also what lets this promise reach *everything* on
    disk, as ADR-038 requires -- a size cap would keep the promise for small files and
    quietly withdraw it for the ones a tenant most wants back.

    **Only a regular file.** A workspace holds sockets and FIFOs an agent made for its
    own use, and opening a FIFO blocks until a writer arrives -- so a route that opened
    whatever the path named would hang a worker on a file that is not a file. The type
    is decided by `lstat` before anything is opened, so nothing is attempted first.
    """
    if mount is None:
        return _unmounted(session_id)
    try:
        segments = parse_workspace_path(path)
    except WorkspacePathInvalid as bad:
        return _malformed(session_id, bad)
    target = _contained_in(_subtree_of(mount, tenant_id, session_id), segments)
    if target is None:
        return _absent(session_id)
    try:
        found = target.lstat()
    except OSError:
        return _absent(session_id)
    if not stat_module.S_ISREG(found.st_mode):
        return refuse(
            ErrorCode.REQUEST_INVALID,
            "that path is not a regular file, so it has no bytes to serve",
            reason=REASON_NOT_A_FILE,
            session_id=str(session_id),
        )
    return FileResponse(
        target,
        media_type="application/octet-stream",
        headers={
            "content-disposition": content_disposition(segments[-1]),
            "x-content-type-options": "nosniff",
        },
    )
