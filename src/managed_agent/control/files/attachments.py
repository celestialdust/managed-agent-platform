"""Putting a Session's attached files into its pod, before its first Turn runs.

The pod holds no cloud identity and cannot read the object store itself (ADR-004), so
the file a tenant uploaded travels the other way: the control plane, which already has
the credential and already has an authenticated hop to the pod, reads the bytes and
pushes them down it. Nothing new is trusted with anything.

This is deliberately NOT the Session VFS mount. That is an NFS mount over the platform
bucket, declared in `deploy/terraform/session_vfs.tf`, and it cannot be created in this
account today -- S3 Files builds its EventBridge rules as the calling identity and that
identity is denied `events:ListRules`, which needs someone with IAM admin. The two are
not alternatives that were weighed: the mount makes the whole bucket visible to the pod
and would replace this, and until it exists this is how a tenant's file reaches an agent
at all. What this does not give is durability of what the agent WRITES, which is the
mount's other half and is untouched here.

Two refusals, and both are the same shape: a Session whose attached file did not arrive
must not run. The agent would report that it cannot find the document, the tenant would
read that as the platform losing their upload, and nothing anywhere would say otherwise.
That is why the byte budget is checked before the first push rather than per file -- a
Session that is going to fail on its fourth file should not have run with three.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from managed_agent.control.files.store import (
    FileId,
    UploadedFile,
    UploadedFileCorrupt,
    UploadedFileNotFound,
    UploadedFileVanished,
)
from managed_agent.core.ids import SessionId, TenantId

WORKSPACE_FILE_BUDGET_BYTES = 256 * 1024 * 1024
"""How many bytes of attached files one Session may carry, in total.

A quarter of the workspace volume's 1Gi `sizeLimit`, which is the number this has to sit
under and the reason it is expressed against it rather than chosen. The remaining
three quarters are the agent's to write into: a Session whose attachments filled the
volume would start, read its files, and then fail on its first output with a disk-full
error that names nothing about attachments.

An `emptyDir` `sizeLimit` is enforced by eviction, not by a write refusal -- the kubelet
notices the volume is over and evicts the pod. So the failure this bound prevents is not
a tidy `ENOSPC`: it is a Session pod disappearing mid-Turn for a reason that appears in
node events and nowhere a tenant can see.

Not the same number as the per-file upload cap (100 MiB) and not derived from it. Two
files at the cap fit; three do not, and that is the intended shape -- the cap is about
one request's size and this is about one pod's disk.
"""


class FilesNotPlaceable(Exception):
    """A Session's attached files cannot be delivered, so the Session must not run.

    One type for every reason, because the caller does the same thing with all of them
    and the difference belongs in the message. The reasons are not interchangeable to a
    reader, though: an id that does not resolve is the tenant's mistake, a vanished
    object or a hash mismatch is the platform's, and an over-budget total is a limit.
    All three end with the same words -- no pod, no Turn -- which is the only outcome
    that does not lie to somebody.
    """


class UploadedFileReader(Protocol):
    """The one thing this module needs of the file store: read a file back.

    A port of one method rather than the concrete `FileStore`, because this module is
    the high-level half and should not know that the store also uploads, enforces a size
    limit, or hashes on the way in. `FileStore` satisfies it structurally, so the
    composition root passes the real one and nothing adapts anything.
    """

    async def fetch(
        self, *, tenant_id: TenantId, file_id: FileId
    ) -> tuple[UploadedFile, bytes]: ...


class FilePlacement(Protocol):
    """Whatever puts one file into a Session's pod.

    A port rather than the concrete client, so this module names no transport. The
    implementation is `shim.pod_channel.PodFilePlacement`, which computes the pod's
    address and its bearer token from what the cluster answers.
    """

    async def place_file(
        self, session_id: SessionId, name: str, body: bytes, /
    ) -> None: ...


@runtime_checkable
class SessionAttachments(Protocol):
    """Whatever can put a file into a Session's pod, named as one method.

    `AttachedFiles` below satisfies it, and so does the refusing stand-in beside it. It
    exists because two callers now want this and they are in different layers: the
    placement path holds the concrete object, and a route holds it off `Platform` -- and
    `control/api/` may not depend on how a file reaches a pod any more than it depends
    on which cluster answers.
    """

    async def place_for(
        self,
        session_id: SessionId,
        tenant_id: TenantId,
        file_ids: tuple[FileId, ...],
        already_held_bytes: int = 0,
    ) -> tuple[str, ...]: ...


class UnconfiguredAttachments:
    """Refuses every placement. What a `Platform` built without a pod runner holds.

    A refusing default rather than `None`, so no caller has to test the field before
    using it and no caller can forget to. It raises the same type a real placement
    raises for a file it cannot deliver, which is what lets one `except` clause at the
    call site cover both -- a process with no pods and a pod that would not take the
    bytes are the same answer to a tenant: the file is not there, so nothing claims it
    is.

    An empty set is still refused rather than answered `()`. `AttachedFiles` returns
    early for one, and matching that here would make "this deployment places no files"
    look like success for exactly the call that asks for none -- the one case where the
    difference is invisible and so the one where a wiring mistake would survive.
    """

    async def place_for(
        self,
        session_id: SessionId,
        tenant_id: TenantId,
        file_ids: tuple[FileId, ...],
        already_held_bytes: int = 0,
    ) -> tuple[str, ...]:
        raise FilesNotPlaceable(
            f"this deployment places no Session pods, so the {len(file_ids)} file(s) "
            f"attached to session {session_id} have nowhere to be written"
        )


class AttachedFiles:
    """Reads a Session's attached files and writes them into its pod.

    Two collaborators and no state, so a single instance serves every Session. The file
    store is the same one `POST /v1/files` wrote through, which is what makes "the file
    the tenant uploaded" and "the file the agent reads" the same bytes rather than two
    reads that could disagree.
    """

    def __init__(self, files: UploadedFileReader, placement: FilePlacement) -> None:
        self._files = files
        self._placement = placement

    async def place_for(
        self,
        session_id: SessionId,
        tenant_id: TenantId,
        file_ids: tuple[FileId, ...],
        already_held_bytes: int = 0,
    ) -> tuple[str, ...]:
        """Put every one of these files in the pod, or refuse having placed none.

        Returns the names as they now exist in the workspace, in the order given, so a
        caller can record what a Session was started with.

        **Every file is read before any is pushed.** The alternative interleaves them
        and leaves a Session holding a prefix of its attachments when the fourth id does
        not resolve -- which starts, reads three files, and is wrong in a way no error
        describes. Reading first also means the byte budget is checked against what was
        actually read rather than against what the rows claimed.

        The cost is stated rather than hidden: the whole set is in this worker's memory
        at once, which is what `WORKSPACE_FILE_BUDGET_BYTES` bounds. A Session attaching
        nothing does no work and asks the store nothing -- there is no question to put.

        `already_held_bytes` is what the Session's pod holds from earlier calls, and it
        starts the running total rather than being compared against separately.
        Defaulted to zero because the creation path is one call for the whole set and
        there is nothing earlier; a late attach folds the figure out of the Session's
        log and passes it, and without it N late attaches each under the budget would
        jointly exceed it -- and an `emptyDir` over its `sizeLimit` is enforced by
        kubelet eviction rather than by `ENOSPC`, so the pod would disappear mid-Turn
        for a reason visible only in node events.

        A caller that folds the figure from a log **must still let this accumulate**.
        The fold prices each held file from its `uploaded_file` row, which is what a
        successful fetch proves the bytes hash to; what the fold cannot see is a row
        whose object vanished between the fold and the push, and the read below is what
        catches that.
        """
        if not file_ids:
            return ()
        bodies: list[tuple[str, bytes]] = []
        total = already_held_bytes
        for file_id in file_ids:
            record, body = await self._read(session_id, tenant_id, file_id)
            total += len(body)
            if total > WORKSPACE_FILE_BUDGET_BYTES:
                raise FilesNotPlaceable(
                    f"the files attached to session {session_id} total more than "
                    f"{WORKSPACE_FILE_BUDGET_BYTES} bytes, which is what its pod's "
                    "workspace can hold beside what the agent writes"
                    + (
                        ""
                        if not already_held_bytes
                        else f" ({already_held_bytes} bytes of it already placed)"
                    )
                )
            bodies.append((str(record.filename), body))
        for name, body in bodies:
            await self._placement.place_file(session_id, name, body)
        return tuple(name for name, _ in bodies)

    async def _read(
        self, session_id: SessionId, tenant_id: TenantId, file_id: FileId
    ) -> tuple[UploadedFile, bytes]:
        """One file, with every way the store can refuse turned into this module's own.

        Re-raised rather than allowed through, because the caller of `place_for` catches
        one type: a `LookupError` escaping from here reaches a tenant as an unexplained
        500 with no Turn closed behind it. The cause is chained, so the message an
        operator reads still names the object key or the digest.
        """
        try:
            return await self._files.fetch(tenant_id=tenant_id, file_id=file_id)
        except UploadedFileNotFound as absent:
            raise FilesNotPlaceable(
                f"session {session_id} attaches file {file_id}, which this tenant does "
                "not hold; the Session would start with the document missing"
            ) from absent
        except (UploadedFileVanished, UploadedFileCorrupt) as broken:
            raise FilesNotPlaceable(
                f"session {session_id} attaches file {file_id} and its bytes could not "
                f"be read back as recorded: {broken}"
            ) from broken
