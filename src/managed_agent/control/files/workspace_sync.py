"""Keeping the agent's working tree after the pod that held it is gone.

A Session's workspace is `emptyDir` in its pod: created with the pod and destroyed with
it. `out/` is shipped to the `artifacts` lane at a completed Turn, so a deliverable
survives -- but the tree the agent was working IN does not, and that tree is what a
resumed Session would need to carry on rather than start over.

This module copies it into the Session's `working` lane at a completed Turn.

**Sync, not mount, and the difference is where the writes go.** ADR-006 rejected
"everything remote" because it pays remote latency for data whose defining property is
that it does not need to survive, and that reasoning holds for every intermediate write
the agent makes. So scratch stays pod-local, no write leaves the pod mid-Turn, and this
runs once at the boundary where a Turn is already durable. A design in which the agent's
writes travelled to an object store as it made them is the road ADR-006 refused.

**Diff, do not re-upload.** The pod's listing carries a digest per file, so this uploads
only the paths whose bytes differ from what the lane already holds. A Session whose
workspace did not change between two Turns costs one listing and one lane listing, and
transfers nothing. That is what makes this affordable to run at *every* completion
rather than occasionally -- and "occasionally" has no correct schedule, because the Turn
that mattered is the one before the pod died.

**What this does NOT do yet: put the tree back.** Restoring a working lane into a pod
needs a write mount of the workspace root, and the shim mounts the workspace
`subPath: files` deliberately -- see `session_shim/serve.WORKSPACE_FILES`, which
explains that a mount of the whole workspace would let whatever reaches that port write
anything the agent later reads. Widening it to serve a restore would trade a permanent
capability for an operation that happens once per placement, which is the wrong shape:
the restore should run in an init container, before the runtime starts and with no port
listening, and that is a pod-manifest change with an ADR owed.

The second blocker was that no Session which had completed a Turn was re-placed at
all, so this lane was written and read by nothing. Both are closed now: the restore runs
as the `restore-working-lane` init container (ADR-030), and the placement path compiles
a second pod for a Session that has already run rather than refusing (ADR-031). What is
still missing is narrower and belongs to the lifecycle rather than to placement -- a
`SUSPENDED` Session accepts no Turn, so nothing reaches placement to ask for that pod.

**And one thing it does not carry: a path whose first character is a dot.** The lane
grammar requires an alphanumeric first character so a path cannot address its lane's own
prefix, so a dotfile at the workspace root composes to no key and is left in the pod.
What that costs is the agent's ordinary root dotfiles -- `.gitignore`, `.env`,
`.python-version`, `.pytest_cache`, `.venv` -- and not the agent's own conversation and
tool state, which is not in the workspace at all. The runtime keeps that under
`CODEX_HOME`, a volume of its own, and the Rollout ships out at a completed Turn by its
own route rather than through any lane (`session_shim/serve.ROLLOUT_ROUTE`).
`<workspace>/.codex` is an empty directory the pod creates only so the sandbox has a
target to deny, and nothing writes it. Settle the dotfiles with the restore, not before
it: the fix is a spelling for such a path inside the lane, and choosing one now would be
choosing it blind.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from managed_agent.control.files.output_shipout import ProducedFile
from managed_agent.core.ids import Seq, SessionId, TenantId, TurnId
from managed_agent.core.ports import EventLogAppend, EventRecord
from managed_agent.core.session.event_append import append_in_order
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vfs.session_vfs import (
    WORKING,
    MutableFile,
    SourceRef,
    StoredObject,
)
from managed_agent.core.vfs.vfs_provenance import provenance
from managed_agent.core.vocabulary import vfs

WORKING_COUNT_LIMIT = 2048
"""How many changed working files one Turn will sync.

Far above `OUTPUT_COUNT_LIMIT`, and the ratio is the point rather than the number. A
produced file is a deliverable and a handful is the expected shape; a working tree is a
project, and a project with two thousand files is ordinary. The bound exists so a
runaway generator cannot make one Turn's sync unbounded, not to express an opinion about
how many files an agent ought to have.
"""

WORKING_BUDGET_BYTES = 256 * 1024 * 1024
"""How many bytes of changed working files one Turn will sync, in total.

Larger than `OUTPUT_BUDGET_BYTES` for the same reason the count is: this is a tree
rather than a document. It bounds what one control-plane worker holds over one Turn --
each file is pulled into memory, written and released, so the peak is one file and this
is what the whole set may cost.

Only files that CHANGED count against it, because only those are transferred. A large
workspace nobody is editing costs nothing after its first sync, which is the common
shape of a long Session.
"""

_UNBOUNDED_END = Seq(2**63 - 1)

_log = logging.getLogger(__name__)
"""This module's own logger. The composition root wires no Python logger --
`log` there is the Event Log append port -- so the operator-facing half of a
partial sync is taken from the standard hierarchy rather than injected."""


class WorkspaceReader(Protocol):
    """Whatever can list a pod's working tree and read one file out of it.

    The pod-wire half, declared here so this module names nothing on that wire.
    `session_shim/pod_channel.py` satisfies it.
    """

    async def list_workspace(
        self, session_id: SessionId, /
    ) -> tuple[ProducedFile, ...]: ...

    async def fetch_workspace_file(
        self, session_id: SessionId, name: str, limit_bytes: int, /
    ) -> bytes | None: ...


class WorkingLane(Protocol):
    """The mutable lane this writes, narrowed to the one call it makes.

    `MutableFile` and not the `VfsFile` union: a sealed lane cannot be rewritten and
    this only ever rewrites, so a Protocol admitting both would let a caller be wired to
    a lane whose semantics refuse the one operation this performs.
    """

    async def replace(
        self,
        file: MutableFile,
        body: bytes,
        sources: Sequence[SourceRef] = (),
    ) -> StoredObject: ...


class SessionReader(Protocol):
    async def read(self, session_id: SessionId) -> SessionRecord: ...


class SessionLogReader(Protocol):
    """The one read this makes of the Event Log, and not the whole range port.

    `EventLogRange` also carries `follow` and `retained_floor`, and this module calls
    neither. Narrowed for the reason every Protocol here is narrowed: a double built to
    exercise this class should have to implement what this class uses, not what the
    port's other consumers use, or the cost of testing a caller rises with every
    capability added for somebody else.
    """

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]: ...


@dataclass(frozen=True, slots=True)
class _Bounded:
    """What one Turn's sync will take, and what it will not."""

    taken: tuple[ProducedFile, ...]
    left_behind: int
    bytes_taken: int
    bytes_offered: int


class SyncWorkingLaneAtTurnCompletion:
    """Copies the agent's working tree into the `working` lane at a completed Turn.

    Last in `EachAtTurnCompletion`, deliberately. The Rollout is the Session's resume
    state and a produced file is the tenant's document; this is what makes a future
    resume better. A seam that raises stops the ones behind it, so the seam whose loss
    costs least goes behind the ones whose loss costs more.

    **Nothing here raises for a workspace that is too large.** A Turn that did good work
    in a big tree is not a failed Turn, and refusing one for its size would make the
    platform's own limit read as the agent's error. The sync takes what fits, in path
    order so two runs over one tree take the same subset, and appends
    `vfs.working_lane_partial` saying what it left.
    """

    def __init__(
        self,
        workspace: WorkspaceReader,
        lane: WorkingLane,
        sessions: SessionReader,
        events: EventLogAppend,
        log_range: SessionLogReader,
    ) -> None:
        self._workspace = workspace
        self._lane = lane
        self._sessions = sessions
        self._events = events
        self._log_range = log_range

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        listed = await self._workspace.list_workspace(session_id)
        if not listed:
            return
        tenant_id = (await self._sessions.read(session_id)).tenant_id
        held = await self._digests_already_stored(session_id)
        changed = _within_the_ceiling(
            tuple(
                one
                for one in sorted(listed, key=lambda f: f.name)
                if _has_changed(one, held.get(one.name))
            )
        )
        for file in changed.taken:
            await self._sync_one(session_id, tenant_id, file)
        if changed.left_behind:
            await self._say_the_lane_is_partial(session_id, changed)

    async def _digests_already_stored(
        self, session_id: SessionId
    ) -> dict[str, str | None]:
        """What the log says each working-lane path currently hashes to.

        **The log and not the bucket, because a lane listing carries no digest.** That
        is deliberate at the store -- filling the field would make listing a lane cost a
        download of it -- and `LaneEntry`'s own docstring names the provenance fold as
        where a caller goes instead. This is that caller.

        `.hex` and not the digest object. `EvidenceDigest` is a model carrying the
        algorithm and the covered length beside the hex, so stringifying it yields a
        repr that matches no pod's report -- and a comparison that never matches is a
        diff that silently degrades to re-uploading the whole tree every Turn. That
        failure is invisible in the bucket, which holds the right bytes either way; only
        a count of what was transferred shows it, which is what the tests assert on.

        Failing safe is the property that makes it usable. Retention can drop the events
        below a Session's floor, and a path whose write fell off the bottom simply has
        no entry here, so it is re-uploaded. The wrong answer is one transfer; the wrong
        answer in the other direction would be an agent's file silently never stored.
        """
        folded = provenance(await self._whole_log(session_id))
        return {
            record.relative: record.digest.hex
            for record in folded.values()
            if record.lane == WORKING.directory
        }

    async def _whole_log(self, session_id: SessionId) -> list[EventRecord]:
        """Every event of one Session, across as many reads as it takes.

        The port caps a read and promises nothing about how much of a range comes back,
        so a short page means "read again" and never "the range is empty". Reads from
        just past the highest sequence seen and stops on an empty page, which terminates
        for any page size because each non-empty page raises the cursor above its own
        highest sequence and a log is finite.

        The third copy of this walk -- `control/session/pods.py` and
        `control/api/routes/sessions.py` hold the other two, and the note there says why
        they were left as two. Three is where that argument stops holding, and pulling
        them together is a refactor that should arrive on its own commit rather than
        riding in with this.
        """
        events: list[EventRecord] = []
        cursor = 0
        while True:
            page = await self._log_range.read(
                session_id, Seq(cursor + 1), _UNBOUNDED_END
            )
            if not page:
                return events
            events.extend(page)
            cursor = max(event.seq for event in page)

    async def _sync_one(
        self, session_id: SessionId, tenant_id: TenantId, file: ProducedFile
    ) -> None:
        """One working file into the lane, or nothing if it went away.

        `replace` and not `place`: the working lane is mutable by construction, and a
        Session's second Turn re-offers every file its first Turn left. A conditional
        create would refuse all of them, which is the shape the `artifacts` lane wants
        and the exact opposite of what this one is for.

        A file that vanished between the listing and the fetch is skipped rather than
        refused, matching ship-out: the agent shares this filesystem, and a scratch file
        it removed after the Turn ended is not a loss.

        The length is NOT compared against the listing, and that is the difference from
        ship-out. A working file that changed while being read is still the agent's own
        state and is worth keeping; a produced file that arrives at the wrong length is
        a document the tenant would be handed in a form nobody wrote. The fetch is still
        capped at the listed length, so a file that grew cannot spend the whole budget.
        """
        body = await self._workspace.fetch_workspace_file(
            session_id, file.name, file.byte_length
        )
        if body is None:
            return
        await self._lane.replace(
            MutableFile(
                tenant_id=tenant_id,
                session_id=session_id,
                lane=WORKING,
                relative=file.name,
            ),
            body,
        )

    async def _say_the_lane_is_partial(
        self, session_id: SessionId, changed: _Bounded
    ) -> None:
        _log.warning(
            "session %s has a workspace larger than one Turn syncs; kept %d of %d "
            "changed paths (%d of %d bytes)",
            session_id,
            len(changed.taken),
            len(changed.taken) + changed.left_behind,
            changed.bytes_taken,
            changed.bytes_offered,
        )
        await append_in_order(
            self._events,
            session_id,
            vfs.WORKING_LANE_PARTIAL,
            {
                "paths_synced": len(changed.taken),
                "paths_left": changed.left_behind,
                "bytes_synced": changed.bytes_taken,
                "path_ceiling": WORKING_COUNT_LIMIT,
                "byte_ceiling": WORKING_BUDGET_BYTES,
            },
        )


def _has_changed(file: ProducedFile, stored: str | None) -> bool:
    """Whether this path has to be uploaded, given what the log says the lane holds.

    **A missing digest on either side means "upload it", never "assume unchanged".** A
    pod that reports no digest is running an older shim image, and reading its silence
    as "nothing changed" would freeze the lane at whatever it held when that pod
    started -- a wrong answer that looks exactly like a working one. A path with no
    provenance is one the lane has never been told about, or one whose write fell below
    the retained floor. Both resolve to the expensive, correct branch.
    """
    if stored is None or file.content_sha256 is None:
        return True
    return file.content_sha256 != stored


def _within_the_ceiling(changed: tuple[ProducedFile, ...]) -> _Bounded:
    """As much of `changed` as one Turn will sync, and how much it turned away.

    Takes a prefix of the sorted list rather than, say, the smallest files first. A
    prefix is stable -- two runs over one unchanged tree take the same subset, so a
    workspace over the ceiling converges on having its first N paths durable instead of
    thrashing between arbitrary halves of itself.

    A single file larger than the whole budget is left behind rather than allowed
    through, and it is the one case where the ceiling costs something real: that file
    can never sync. It is reported in the same event as everything else left behind,
    which is the only place a tenant could learn it.
    """
    taken: list[ProducedFile] = []
    bytes_taken = 0
    for file in changed:
        if len(taken) >= WORKING_COUNT_LIMIT:
            break
        if bytes_taken + file.byte_length > WORKING_BUDGET_BYTES:
            break
        taken.append(file)
        bytes_taken += file.byte_length
    return _Bounded(
        taken=tuple(taken),
        left_behind=len(changed) - len(taken),
        bytes_taken=bytes_taken,
        bytes_offered=sum(file.byte_length for file in changed),
    )
