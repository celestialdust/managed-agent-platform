"""Getting a file the agent WROTE out of the pod before the pod is allowed to die.

A Session's workspace is an `emptyDir` (`deploy/k8s/session-pod.yaml`). Nothing about it
outlives the pod, so until this existed a tenant who asked an agent to produce a
document got a completed Turn and no document: the bytes were written, the Turn was
recorded, and the reaper took the volume with the pod.

This is the outbound mirror of `session_files.py`. That module reads the object store
and pushes a tenant's upload down the authenticated hop into the pod; this one pulls
what the agent left back up the same hop and puts it in the store. The direction is
the same in both and it is not a style choice: the pod is given no cloud identity
(ADR-004), so nothing in it can read or write a bucket, and every byte crossing that
boundary crosses because the control plane came and moved it.

It is the same journey the Rollout already makes -- fetch off the pod's disk at a
completed Turn, put in the object store -- and it deliberately makes the same choices
where the reasons carry over. Where it differs, the difference is stated at the point it
is made.

Nothing here imports a web framework, an object-store client or the pod wire. The three
collaborators are Protocols declared below and satisfied elsewhere, which is what keeps
this module -- where every bound and every refusal lives -- exercisable without an HTTP
request or a bucket.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from managed_agent.control.files.lane_digests import (
    SessionLogReader,
    digest_differs,
    digests_in_lane,
)
from managed_agent.control.files.store import content_digest
from managed_agent.core.ids import SessionId, TenantId, TurnId
from managed_agent.core.pod.workspace_contract import is_a_produced_path
from managed_agent.core.ports import EventLogAppend
from managed_agent.core.session.event_append import append_in_order
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vfs.session_vfs import (
    ARTIFACTS,
    ObjectAlreadyPresent,
    SealedFile,
    SourceRef,
    StoredObject,
    VfsFile,
)
from managed_agent.core.vocabulary import output

_LOG = logging.getLogger(__name__)
"""This module's own logger, for the operator half of a partial ship-out.

The composition root wires no Python logger -- its `log` is the Event Log append port --
so this is taken from the standard hierarchy rather than injected. Nothing here logs a
path or a byte of a tenant's document; the counts and the Session id are what an
operator needs and all they get.
"""

OUTPUT_TREE_LIMIT = 2048
"""How many files the pod will enumerate under its output directory, at most.

**Two bounds and not one, because listing a file and shipping a file cost different
things.** This bounds a walk of the pod's own disk and the size of the listing that
crosses the wire; `OUTPUT_COUNT_LIMIT` below bounds how many files one Turn transfers.
They were one constant until 2026-08-25, and that is precisely what made the transfer
bound cumulative: the pod stopped scanning at the transfer bound, so a Session already
holding that many had the files it added this Turn fall outside what the pod would even
report. The filter that weighs already-delivered paths out of the transfer bound cannot
weigh a path it was never shown.

The scan stops once it has this many *plus one*, and that extra entry is the signal
rather than a payload: it lets the far end tell "the tree holds exactly what I will
enumerate" from "it holds more", without the pod having read a directory of unknown size
to count.

Two thousand and forty-eight, and the ratio to the transfer bound is the point rather
than the number. Being over this one is
the worse of the two failures: past it the walk yields a sorted prefix and nothing
else, so a file whose name sorts beyond the cut is invisible on this Turn and on every
Turn after it. Being over the transfer bound is recoverable by comparison -- that tail
stays on the pod and the next Turn reaches further into it.

Which is why the signal is read rather than merely available. `_within_the_ceilings`
is told whether the listing came back over this bound and `output.partial` carries it,
on every Turn it is true and not only on the Turn a tail also existed. There is nothing
the platform can do about a tree this size -- it will not walk further, and it cannot
delete out of a pod -- and everything the tenant can, once they are told.

It is not higher, and the reason is the pod's own clock rather than the wire. Every file
this walk lists is read and hashed on every Turn (`_digest_of` in `shim/serve.py`), so
the ceiling is also what a Session pays per Turn for the rest of its life. Two thousand
small documents is milliseconds; two hundred thousand would be a Turn spent hashing.
"""

OUTPUT_COUNT_LIMIT = 500
"""How many produced files one completed Turn may ship.

Declared here rather than beside the route that reports them, because it is a bound on
what a control-plane worker will take rather than a fact about the pod.

**Five hundred, raised from thirty-two on 2026-08-25, and the reason for the old number
had stopped being true.** Thirty-two was chosen for an agent that writes a document:
one to a handful of files at a working root, and a tree with more than thirty-two files
directly in it read as a build directory nobody asked to keep. Agents that fan a
literature review out to a file per paper, or a report out to a file per section,
produce hundreds in one Turn and are not build directories, and thirty-two turned them
into a Turn that shipped a third of its work.

It is not the bound that keeps the transfer finite -- `OUTPUT_BUDGET_BYTES` is, and it
did not move. What this one bounds is the *count*: five hundred sequential fetch-and-
store round trips, one per file, inside the tenant's own request. At an in-cluster fetch
plus an object-store write that is tens of seconds on a Turn that produced five hundred
files, which is a Turn that has already been running for minutes. Raising it further
buys less: the byte budget would bind first for anything but very small files, and the
round trips are what a tenant waits through.

It is deliberately far below `OUTPUT_TREE_LIMIT`, and the gap is what makes this a bound
on one Turn rather than on the Session -- see that constant. Moving this one up without
moving that one is what tightens the gap, and
`test_the_enumeration_bound_is_well_above_what_one_turn_transfers` is what notices.

**It bounds what one Turn ADDS, not what the Session holds**, and that distinction was
missing until 2026-08-25. Nothing empties the agent's output directory between Turns, so
the pod re-offers every file it has ever produced on every Turn; counted whole, this was
a budget on distinct paths for the life of the Session, and a run writing four files a
Turn shipped seven times and was refused on the eighth having done nothing different.
Worse, there is no route that deletes a file out of a pod, so that refusal repeated on
every Turn after it and the Session could never deliver again. `_not_yet_delivered` is
what makes the count this docstring always claimed the count that is taken.

Being over it no longer fails the Turn either. Ship-out takes this many in sorted
order, leaves the rest on the pod for the next Turn, and appends `output.partial`
naming what it left and which ceiling bound it. The objection the old refusal rested
on was silence rather than partiality -- a tenant handed some of what they produced
cannot tell which the platform dropped *if nothing says so* -- and the event says so.
"""

OUTPUT_BUDGET_BYTES = 64 * 1024 * 1024
"""How many bytes of produced files one completed Turn may ship, in total.

A quarter of `session_files.WORKSPACE_FILE_BUDGET_BYTES`, and the ratio is the reason
rather than the number. That budget is spent once per Session, when its attachments are
placed; this one is spent **once per Turn**, and a Session runs many Turns, so the
per-Turn figure has to be the smaller of the two or a long Session's ship-outs dominate
what a control-plane worker holds.

What it bounds is exactly that: this worker's memory. Each file is pulled off the pod
into memory and handed to the file store, which buffers it again to hash it, so the peak
is one file rather than the whole set -- but the set is what a Turn is allowed to cost,
and an unbounded one is a control-plane replica killed by the OOM reaper while serving
one tenant's Session.

Spent against the lengths the listing reported rather than against what arrives, so the
set a Turn will take is decided before the first byte moves. Files are taken in sorted
order until the next one would cross this line, and the rest are left on the pod with
`output.partial` naming them -- so a Turn that cannot afford its fourth file still ships
three, which is the opposite of the old shape, where one file over the line cost every
file beside it.

One case this cannot repair, stated because only the event makes it visible: a single
file larger than this whole budget fits in no prefix and is therefore left behind on
every Turn, forever.
"""


@dataclass(frozen=True, slots=True)
class ProducedFile:
    """One file the agent produced: where it sits in the tree, and how big it is.

    `name` is a path relative to the directory ship-out is reading from, so it may carry
    separators -- `report/fig1.png` is one file, at that path, one directory down. The
    field kept its name through that widening because it is the wire's field name too,
    and renaming it here alone would leave the two spellings to be reconciled by whoever
    next read `pod_channel.py`.

    The length is carried because it is what the transfer is checked against. A produced
    file is not append-only the way a Rollout is, so a body that arrives short has no
    torn-tail reading available: it is a partial document, and storing one under the
    whole document's name is a lie the tenant has no way to detect.

    Declared here rather than in the shim's wire models so this module names nothing on
    the pod wire. `shim/pod_channel.py` parses the wire shape and hands back these.
    """

    name: str
    byte_length: int
    content_sha256: str | None = None
    """What the pod said the bytes hash to, or None if it does not report digests.

    None is not "no opinion, proceed": it is "this pod predates the field". A Session's
    pod runs the shim image it started with, so a control plane that required this would
    fail every Turn already in flight the moment it rolled. Verified when present,
    skipped when absent, and the absence is what a rolling deploy looks like rather than
    a hole a caller chose.
    """


class OutputsNotShippable(Exception):
    """What the agent produced cannot be made durable, so the Turn did not achieve it.

    One type for every reason whose *caller* acts the same on it, which is all of them
    but one: `HttpPodDispatch` turns whatever the completion seam raises into
    `TurnUndeliverable` and `control/api/routes/turns.py` records the Turn as failed.
    The reasons are not interchangeable to a reader, though -- a body that did not match
    its listed length, bytes that changed between the listing and the fetch, and a name
    the pod promised not to send are three different problems -- so each carries its own
    message.

    The exception is `OutputNotRevisable` below, which subclasses this. It is split off
    because the caller does *not* do the same thing with it: it is the tenant's own
    doing, it answers 409 rather than 502, and the move it invites is the opposite of a
    retry.
    """


class OutputNotRevisable(OutputsNotShippable):
    """The agent wrote a produced path it had already delivered, with other bytes.

    The one refusal in this module the tenant caused and can fix, which is why it is
    the one with a type of its own. Everything else raised here is the pod failing to
    serve what it listed, and a tenant's move for all of those is to try the Turn
    again; trying again here re-runs an agent that writes the same path a second time.

    A subclass rather than a separate hierarchy so a caller that knows only
    `OutputsNotShippable` still catches it -- and `pod_channel.py`, which translates
    this one to a 409 and collapses the rest into `TurnUndeliverable`, tells them apart
    by type rather than by matching the message. Matching the message would make a
    tenant-visible status depend on how an operator worded a sentence.

    The path is an attribute because the refusal envelope two hops up puts it in
    `detail`; recovering it by parsing the message would tie that field to the prose.
    """

    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


class WorkspaceOutputs(Protocol):
    """Getting at what one Session's agent has written, without naming a transport.

    The concrete implementation is `shim/pod_channel.py`'s `PodOutputFetch`, which
    computes the pod's address and the Session's bearer token from what the cluster
    answers. A Protocol rather than that class for the reason `RolloutFetch` is one:
    `placement` sits on the far side of this seam and importing it here would close a
    cycle back through `control/session/placement.py`.

    The direction is the point of the shape, exactly as it is for a Rollout. State
    leaves a pod because the control plane came and read it, never because the pod
    pushed: a Protocol shaped the other way would need a durable write credential inside
    the least-trusted process in the platform, and the per-Session token that would
    authorize it is a constant for the Session's life with no nonce and no expiry
    (ADR-022).
    """

    async def list_outputs(self, session_id: SessionId, /) -> tuple[ProducedFile, ...]:
        """What that Session's agent has produced. Empty when its pod holds nothing.

        May answer with more entries than `OUTPUT_COUNT_LIMIT`; that is how "the
        workspace holds more than we will take" is reported, and the caller decides
        what it costs.
        """

    async def fetch_output(
        self, session_id: SessionId, name: str, limit_bytes: int, /
    ) -> bytes | None:
        """That file's bytes, or None when the name no longer names a plain file.

        Refuses rather than truncating past `limit_bytes`, so a pod that under-reported
        a length in its listing cannot spend more of this worker's memory than the
        budget the listing was checked against.
        """


class ArtifactLane(Protocol):
    """The two lane operations shipping a produced file needs, and no third.

    `place` and not `replace`, because the destination is sealed: a Turn's artifact is
    written once and the store refuses a second write to the same key. `read` is here
    only for what that refusal costs -- see `_ship_one` -- and is deliberately not a
    listing: what this module ships is decided by what the pod offers, never by what the
    lane already holds.

    A narrow Protocol rather than the concrete `SessionVfsStore`, matching how
    `session_files.py` asks the file store for a read: this module is the high-level
    half and has no business knowing the store also lists lanes, or that the event
    recording each write is appended by the same call. `SessionVfsStore` satisfies it
    structurally, so the composition root passes the real one and nothing adapts
    anything.
    """

    async def place(
        self, file: VfsFile, body: bytes, sources: Sequence[SourceRef] = ()
    ) -> StoredObject:
        """Write a new object. Raises `ObjectAlreadyPresent` if the key is occupied."""

    async def read(self, file: VfsFile) -> bytes | None:
        """The object's bytes, or None when the lane holds nothing at that path."""


class SessionCreationFacts(Protocol):
    """A Session's record by id alone, for the one field this needs: its tenant.

    Keyed by the Session and taking no tenant, which is worth being exact about because
    every tenant-scoped read of that store takes one. The tenant argument is what a
    *caller-supplied* id is filtered by, so another tenant's Session is absent from an
    answer rather than fetched and then dropped. Nothing is caller-supplied here: this
    runs on the completion of a Turn the platform itself admitted, for a Session it
    already resolved, and the record is never handed back out -- its `tenant_id` is only
    used to scope the write that follows. A tenant argument here would have to be
    invented, and an invented tenant is a filter that always agrees with itself.
    """

    async def read(self, session_id: SessionId) -> SessionRecord: ...


class TurnCompletionSeam(Protocol):
    """Whatever a completed Turn owes. Satisfied by this module's class and by
    `rollout_sync.ShipOutAtTurnCompletion`, neither of which imports the other."""

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None: ...


class EachAtTurnCompletion:
    """Every seam a completed Turn owes, awaited in the order they were given.

    Ordered, and the order is a decision rather than an artefact of a loop. The
    Rollout's ship-out goes first: it is the Session's resume state, so losing it makes
    every later Turn impossible, while losing a produced document costs that document. A
    seam that raises stops the ones behind it, so putting the recoverable loss second is
    what keeps a failure to ship one file from also costing the Session its ability to
    run again.

    A variadic composite rather than a field on either seam, because neither owns the
    other and a seam that reached for a sibling would decide the order somewhere no
    reader of the composition root can see it.
    """

    def __init__(self, *seams: TurnCompletionSeam) -> None:
        self._seams = seams

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        for seam in self._seams:
            await seam.turn_completed(session_id, turn_id)


@dataclass(frozen=True, slots=True)
class _Bounded:
    """What one Turn's ship-out takes of what it was offered, and what it leaves.

    `left_behind` counts only paths the pod actually listed. `tree_truncated` says the
    listing was not the whole tree, so there are further paths this number does not
    include and cannot -- counting them would mean the pod reading a directory of
    unknown size, which is what `OUTPUT_TREE_LIMIT` exists to prevent. The two fields
    together are the honest answer: this many are late, and there may be more.
    """

    taken: tuple[ProducedFile, ...]
    left_behind: int
    bytes_taken: int
    tree_truncated: bool


def _within_the_ceilings(
    undelivered: Sequence[ProducedFile], tree_truncated: bool
) -> _Bounded:
    """As many new files as one Turn ships, and how many it turned away.

    **A prefix of the sorted list rather than, say, the smallest files first.** A prefix
    is stable, so a Session over the ceiling converges on having its first N paths
    durable instead of thrashing between arbitrary halves of itself -- and the paths it
    did not take are offered again next Turn, where the ones already delivered cost
    nothing and the same prefix rule reaches further into the tail.

    **Both ceilings are checked per file rather than over the set**, which is what makes
    a partial possible at all: the old shape summed the whole set and refused, so one
    file over the budget cost every file beside it. Here the loop stops adding when the
    next file would cross either line, and everything it already took still ships.

    A single file larger than the whole byte budget is left behind rather than allowed
    through, and it is the one case this costs something real: no prefix containing it
    ever fits, so it never ships. `output.partial` is the only place a tenant could
    learn that, which is why it is appended for every non-empty tail rather than only
    for the ones that will drain.

    The only walk of this shape left in the tree. There was a near-copy in the
    working-lane sync, deliberately kept separate because the ceilings and the announced
    counts differed; that module is gone with the lane (ADR-035), so the question of
    folding them is closed rather than answered.
    """
    taken: list[ProducedFile] = []
    bytes_taken = 0
    for file in sorted(undelivered, key=lambda one: one.name):
        if len(taken) >= OUTPUT_COUNT_LIMIT:
            break
        if bytes_taken + file.byte_length > OUTPUT_BUDGET_BYTES:
            break
        taken.append(file)
        bytes_taken += file.byte_length
    return _Bounded(
        taken=tuple(taken),
        left_behind=len(undelivered) - len(taken),
        bytes_taken=bytes_taken,
        tree_truncated=tree_truncated,
    )


class ShipOutOutputsAtTurnCompletion:
    """Puts every file the agent produced this Turn into the object store.

    **A failed ship-out raises, and the Turn's tenant-visible events have already been
    appended.** That is the Rollout path's choice and the reason carries over intact: a
    Turn that reads as complete while what it produced is still only inside a pod about
    to be allowed to die is the exact lie this whole module exists to remove. ADR-004
    already puts the Event-Log-ahead-of-the-durable-state divergence in writing, so the
    honest outcome is a dispatch that fails loudly over one that swallows the error.

    The cost of that choice is real and is bounded deliberately. A transient
    object-store failure turns a Turn that did run into a Turn recorded as failed, which
    a tenant may retry and pay for twice -- so `EachAtTurnCompletion` runs the Rollout's
    ship-out first, and a failure here therefore never costs the Session its resume
    state. What it costs is one Turn's record, not the Session.

    **A pod holding nothing does nothing**, and unlike a Rollout that is the common case
    rather than a guard: most Turns answer in text and write no file, and there is no
    stored object here for an empty answer to overwrite. Nothing is asked of the file
    store, the Session registry or the Event Log for such a Turn.

    **Each file shipped appends `output.produced`, and without it the bytes are
    unreachable.** Nothing else in the platform tells the tenant which path a Turn wrote
    to: `GET /v1/sessions/{id}/resources` lists what the Session was created with, and
    there is deliberately no route that lists a lane. So the append is part of shipping
    rather than a notification beside it -- a stored file nobody was told about is the
    same outcome as one that never left the pod, an AWS bill worse.

    That is the second event each shipped file produces. The first is
    `vfs.object_placed`, appended by the lane store inside `place`, recording that the
    object exists and what it hashes to; this one says the object is a deliverable.
    Neither is derivable from the other, because the same lane write happens for things
    nobody asked for.

    The append is per file and follows that file's write, so a run that raises partway
    has appended exactly the files that are durable. The alternative -- one event after
    the loop -- would name files that shipped and files that did not with no way to tell
    which, and the loop above already refuses every bound before the first transfer
    precisely so that a partial ship-out is rare rather than routine.
    """

    def __init__(
        self,
        outputs: WorkspaceOutputs,
        artifacts: ArtifactLane,
        sessions: SessionCreationFacts,
        events: EventLogAppend,
        log_range: SessionLogReader,
    ) -> None:
        self._outputs = outputs
        self._artifacts = artifacts
        self._sessions = sessions
        self._events = events
        self._log_range = log_range

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        """Ship what this Session's agent produced, or refuse having shipped none.

        Every refusal is decided before the first transfer, which is the shape
        `session_files.place_for` takes and for the same reason: a partial ship-out is
        the worst outcome available, because a tenant holding three of five documents
        cannot tell which two the platform dropped or that it dropped any.

        The names are re-parsed here even though the pod filtered them, because the pod
        is the untrusted end: `parse_upload_filename` is what the platform records a
        file under, and a name that fails it is a pod sending something its own listing
        promised not to. That is a lying pod rather than a badly named document, which
        is why it raises rather than being skipped -- a real pod never produces it.

        **That check runs over everything the pod offered, and it runs first.** A
        compromised pod is a different kind of problem from a Turn that produced too
        much, and ordering the bounds ahead of it would let a pod hide an illegal path
        behind a listing large enough to be refused on its count. Both refuse the Turn,
        so the ordering costs nothing but which reason is reported -- and the reason is
        the only thing an operator reading it has.

        **The bounds are then weighed over what is NOT already delivered**, which is
        what makes them per-Turn; `_not_yet_delivered` carries why. A Turn that produced
        nothing new returns having asked nothing of the pod, the bucket or the Session
        registry -- the same shape a Turn that produced nothing at all takes, and for
        the same reason: there is no work in it.

        **Being over those bounds ships part and appends `output.partial`; it does not
        fail the Turn.** That is the sibling sync's choice and the reason carries over
        with one addition. A Turn that did the work and wrote the documents must not be
        failed for how many of them there were -- and here it must especially not be,
        because nothing removes a file from a pod, so a refusal repeats on every Turn
        after it and the Session can never deliver again. The objection the old refusal
        was built on was silence, not partiality: a tenant handed some of what they
        produced cannot tell which the platform dropped *if nothing says so*. The event
        says so, and names the ceiling that bound it.

        What still fails the Turn is the set of things a later Turn cannot repair: a pod
        offering a path its own listing rule forbids, a body that arrives at the wrong
        length or the wrong digest, and an artifact rewritten under a path already
        delivered. None of those is a quantity, and none of them drains.
        """
        produced = await self._outputs.list_outputs(session_id)
        if not produced:
            return
        for file in produced:
            _refuse_a_path_the_pod_promised_not_to_send(session_id, file.name)
        bounded = _within_the_ceilings(
            await self._not_yet_delivered(session_id, produced),
            tree_truncated=len(produced) > OUTPUT_TREE_LIMIT,
        )
        if bounded.taken:
            tenant_id = (await self._sessions.read(session_id)).tenant_id
            for file in bounded.taken:
                await self._ship_one(session_id, tenant_id, file)
        if bounded.left_behind or bounded.tree_truncated:
            await self._say_what_did_not_fit(session_id, bounded)

    async def _not_yet_delivered(
        self, session_id: SessionId, produced: Sequence[ProducedFile]
    ) -> tuple[ProducedFile, ...]:
        """The files whose bytes this Session's artifacts lane does not already hold.

        **This is what makes the two bounds above per-Turn rather than per-Session.**
        Nothing empties the agent's output directory between Turns and nothing should,
        so the pod re-offers every file it has ever produced on every Turn after the
        first. Counted whole, the bound is a budget on distinct paths for the life of
        the Session: a run writing four files a Turn ships seven times and is refused on
        the eighth, having done nothing different. Counted here, only what this Turn
        actually added is weighed, which is what both bounds' own docstrings say they
        weigh.

        It also removes the transfer. A path already delivered with identical bytes is
        dropped before anything is fetched from the pod or read out of the bucket, so a
        Turn that produced nothing new costs one log fold instead of a fetch and a
        download per file it has ever produced.

        **A path already delivered with DIFFERENT bytes is kept, not dropped.** That is
        the agent rewriting a delivered artifact, the seal says it cannot be recorded,
        and `_ship_one` is where it is refused saying so. Dropping it here would be the
        silent wrong answer -- a Turn reporting success having stored neither version.

        The fold fails safe, so a path whose write has fallen below the log's retained
        floor reads as undelivered and is transferred again. `_ship_one` then finds the
        key occupied and settles it on the bytes, which is the slow path and the correct
        one.
        """
        delivered = await digests_in_lane(
            self._log_range, session_id, ARTIFACTS.directory
        )
        return tuple(
            file
            for file in produced
            if digest_differs(file.content_sha256, delivered.get(file.name))
        )

    async def _say_what_did_not_fit(
        self, session_id: SessionId, bounded: _Bounded
    ) -> None:
        """Announce the tail this Turn did not take, and log it for an operator.

        Both audiences, because the two learn different things from it. The tenant gets
        the counts and the ceilings, which is what tells them a document is coming
        rather than lost; the platform's own log gets the same numbers against a
        Session id, which is what turns "one tenant's file is late" into a pattern.

        **Appended for a truncated tree even when nothing was left behind**, and that is
        the case worth being deliberate about. A Session past the enumeration bound goes
        on taking Turns, and on most of them everything the listing showed is already
        delivered -- every ceiling here satisfied, `paths_left` zero, and by every
        measure this code has of its own work the Turn went perfectly. The files past
        the cut are still unreachable. Announced only on the Turn a tail existed, the
        one report would be the Turn it first appeared on and a tenant not watching then
        would never hear of it again.
        """
        _LOG.warning(
            "session %s produced more than one Turn ships; took %d of %d new paths "
            "(%d bytes) against ceilings %d and %d, tree truncated: %s",
            session_id,
            len(bounded.taken),
            len(bounded.taken) + bounded.left_behind,
            bounded.bytes_taken,
            OUTPUT_COUNT_LIMIT,
            OUTPUT_BUDGET_BYTES,
            bounded.tree_truncated,
        )
        await append_in_order(
            self._events,
            session_id,
            output.OUTPUT_PARTIAL,
            {
                "paths_shipped": len(bounded.taken),
                "paths_left": bounded.left_behind,
                "bytes_shipped": bounded.bytes_taken,
                "path_ceiling": OUTPUT_COUNT_LIMIT,
                "byte_ceiling": OUTPUT_BUDGET_BYTES,
                "tree_ceiling": OUTPUT_TREE_LIMIT,
                "tree_truncated": bounded.tree_truncated,
            },
        )

    async def _ship_one(
        self,
        session_id: SessionId,
        tenant_id: TenantId,
        file: ProducedFile,
    ) -> None:
        """One produced file, checked against the length its listing reported.

        **The fetch is capped at that same length rather than at the whole budget.** It
        is the tightest cap available and it is free, because the listing already said
        how big the file is -- so a pod that under-reports a length to slip a large
        transfer past the budget check above is refused at the wire instead.

        A body *shorter* than the listing is refused too, and that is the half a cap
        cannot do. A produced file is not append-only, so short bytes are a document
        that was being rewritten while it was read, and storing that under the whole
        document's name is undetectable afterwards: the store hashes what it is given,
        so the digest would certify the truncation.

        **The length is checked and then the digest, and the second catches what the
        first cannot.** A file rewritten to the same length between the listing and the
        fetch passes every length check there is; only the content settles it. The pod
        reports the digest it computed while listing, so this compares two moments on
        the pod's own filesystem rather than trusting one of them -- and a mismatch is a
        refusal rather than a repair, because nothing here knows which of the two bodies
        the tenant meant. A pod that reports no digest is not refused, for the reason
        `ProducedFile.content_sha256` gives.

        **A file that has vanished is skipped, not refused.** The pod answers "no plain
        file there" for a name it listed a moment earlier when the agent unlinked it
        after its Turn ended, and there is no document in that to lose.

        **The destination is a sealed lane, so the second Turn of every Session lands
        on an occupied key.** Nothing clears the agent's output directory between Turns
        -- and nothing should, because a route that reached into a pod to delete a
        tenant's files to fix a control-plane problem would also destroy the tenant's
        expectation that `out/` accumulates across a Session. So a Turn that produced
        nothing new re-offers the previous Turn's files, and `place` refuses every one
        of them. Left unhandled that fails a Turn that did nothing wrong, on the second
        Turn, for every Session that ever produced a file.

        `_is_already_delivered` settles it by reading what is stored. Identical bytes
        mean this exact artifact is already durable and already announced, so the ship
        succeeded before and succeeds now, and nothing is appended -- an event per Turn
        naming an unchanged file would tell a tenant polling the log that their document
        was produced again on a Turn that never touched it. Different bytes at the same
        path are a real collision: the agent rewrote a delivered artifact, the seal says
        that cannot be recorded, and the Turn is refused saying so.
        """
        body = await self._outputs.fetch_output(session_id, file.name, file.byte_length)
        if body is None:
            return
        if len(body) != file.byte_length:
            raise OutputsNotShippable(
                f"session {session_id} listed {file.name!r} as {file.byte_length} "
                f"bytes and served {len(body)}, so what arrived is not the document"
            )
        if (
            file.content_sha256 is not None
            and content_digest(body) != file.content_sha256
        ):
            raise OutputsNotShippable(
                f"session {session_id} served {file.name!r} at the length it listed "
                "but not the content it listed, so the bytes changed between the two "
                "and what arrived is not the document that was offered"
            )
        target = SealedFile(
            tenant_id=tenant_id,
            session_id=session_id,
            lane=ARTIFACTS,
            relative=file.name,
        )
        try:
            await self._artifacts.place(target, body)
        except ObjectAlreadyPresent as occupied:
            if await self._is_already_delivered(target, body):
                return
            raise OutputNotRevisable(
                file.name,
                f"session {session_id} produced {file.name!r} again with different "
                "bytes, and an artifact already delivered under that path cannot be "
                "revised -- write the new version under a different path",
            ) from occupied
        await append_in_order(
            self._events,
            session_id,
            output.OUTPUT_PRODUCED,
            {"path": file.name, "byte_length": len(body)},
        )

    async def _is_already_delivered(self, target: SealedFile, body: bytes) -> bool:
        """Whether the lane already holds exactly these bytes at exactly this path.

        Compares the bytes rather than a digest of them, which is what the port makes
        available and is also strictly the stronger check: two digests agreeing is
        evidence the bodies agree, and the bodies agreeing is the fact itself. There is
        no `head` on `LaneBlobs` that could answer this without the download, and adding
        one to save a fetch on a path that only runs when a Turn produced nothing new
        would be buying speed with a fifth operation on a port whose whole claim is that
        it has four.

        The cost is stated where it bites: this holds the stored object and the fetched
        one at once, so a re-offered 64 MiB artifact peaks at twice that in this worker.
        The per-Turn budget already bounds the set, and the peak is one file rather than
        the set, so the ceiling is `OUTPUT_BUDGET_BYTES` doubled and not multiplied by
        the file count.

        A read that answers None while `place` said the key was occupied is a delete
        between the two calls -- which nothing in this platform holds the grant to do --
        so it is reported as a real collision rather than as a match. Answering "already
        delivered" for an object that is not there would drop the file silently.
        """
        stored = await self._artifacts.read(target)
        return stored is not None and stored == body


def _refuse_a_path_the_pod_promised_not_to_send(
    session_id: SessionId, relative: str
) -> None:
    """Raise unless this path is one the pod's own listing rule would have offered.

    Re-parsed rather than trusted. The shim filters its listing by `is_a_produced_path`
    and this checks the same predicate, so a path arriving here that fails it is not a
    document with an awkward name -- it is a pod sending something its own listing said
    it would not, and the honest reading of that is a compromised pod rather than a file
    to quietly drop.

    Checked for every file before the first transfer, for the reason the count and byte
    bounds are: a Turn that is going to be refused on its fourth file should not have
    shipped three, because a tenant holding three of five documents cannot tell which
    two the platform dropped or that it dropped any.

    Refuses rather than returning a parsed value, because there is nothing to return:
    the path is a `str` already and `SealedFile` parses it again at construction. What
    this adds over that parse is the dotted-segment rule and a message naming the pod.
    """
    if not is_a_produced_path(relative):
        raise OutputsNotShippable(
            f"the pod for session {session_id} offered {relative!r} as a produced "
            "file, which is not a path this platform's own listing rule would have "
            "offered: it must be relative, carry no dotted segment and no `..`"
        )
