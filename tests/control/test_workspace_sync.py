"""The agent's working tree, kept after the pod that held it is gone.

Every case here drives `SyncWorkingLaneAtTurnCompletion` over a fake pod and a fake
bucket with the REAL `SessionVfsStore` between them, so the key composition, the
`vfs.object_replaced` append and the mutable-lane semantics are the shipping code's own
rather than a double's imitation of them.

**The diff is proved by what was transferred, never by reading the code.** Every case
that claims "only what changed was uploaded" asserts on `pod.fetched` -- the list of
paths the pod was actually asked for -- because a sync that re-read every file and then
wrote the same bytes back would satisfy any assertion made about the bucket's contents.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest
from test_output_shipout import FakeSessions

from managed_agent.adapters.s3.session_vfs import SessionVfsStore
from managed_agent.control.files.output_shipout import ProducedFile
from managed_agent.control.files.store import content_digest
from managed_agent.control.files.workspace_sync import (
    WORKING_BUDGET_BYTES,
    WORKING_COUNT_LIMIT,
    SyncWorkingLaneAtTurnCompletion,
)
from managed_agent.core.ids import Seq, SessionId, TenantId, TurnId, new_session_id
from managed_agent.core.ports import EventRecord
from managed_agent.core.vfs.session_vfs import WORKING, LaneEntry
from managed_agent.core.vocabulary import vfs

pytestmark = pytest.mark.anyio

_TURN = TurnId(uuid4())
_SCRIPT = b"print('the agent wrote this')\n"


class FakeLaneBlobs:
    """One dict of objects, keyed exactly as the real bucket keys them.

    A fake at the *bucket* and not at the lane, so everything above it here is the real
    `SessionVfsStore`: the key composition, the `vfs.object_replaced` append and the
    digest it records are the shipping code's own rather than a double's imitation.

    Not `test_output_shipout.FakeLaneBlobs`, and the difference is the point. That one
    raises on `put` -- "a sealed lane is never rewritten" -- a correct guard for the
    artifacts lane and exactly wrong here: rewriting is what the working lane is for,
    and a Session's second Turn rewrites every file its first Turn left. Sharing one
    double would mean weakening that guard to serve this file.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put_new(self, key: str, body: bytes) -> None:
        raise AssertionError("the working lane creates through replace, not place")

    async def put(self, key: str, body: bytes) -> None:
        self.objects[key] = body

    async def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    async def list_prefix(self, prefix: str) -> Sequence[LaneEntry]:
        raise AssertionError("the diff reads provenance, not a lane listing")


class FakeWorkspace:
    """A working tree that answers a fixed listing and hands back fixed bodies.

    `fetched` is the whole point of this double. It records every path the sync actually
    asked the pod for, which is the only evidence that a second sync transferred less
    than a first one -- comparing bucket contents cannot tell a skipped upload from a
    re-upload of identical bytes.
    """

    def __init__(
        self,
        listing: tuple[ProducedFile, ...],
        served: dict[str, bytes] | None = None,
    ) -> None:
        self.listing = listing
        self._served = (
            served
            if served is not None
            else {f.name: b"x" * f.byte_length for f in listing}
        )
        self.fetched: list[str] = []

    async def list_workspace(
        self, session_id: SessionId, /
    ) -> tuple[ProducedFile, ...]:
        return self.listing

    async def fetch_workspace_file(
        self, session_id: SessionId, name: str, limit_bytes: int, /
    ) -> bytes | None:
        self.fetched.append(name)
        return self._served.get(name)


class ReplayingLog:
    """One Session's log, appended to and read back, as the real pair of ports is.

    Append and range over ONE list, because the sync writes provenance through the lane
    store and then reads it back on the next Turn: two doubles that did not share
    storage would make every second sync look like a first one, which is exactly the
    behaviour under test.
    """

    def __init__(self) -> None:
        self.records: list[EventRecord] = []
        self.appended: list[tuple[str, dict[str, object]]] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        seq = Seq(len(self.records) + 1)
        self.appended.append((type_, payload))
        self.records.append(
            _Record(session_id=session_id, seq=seq, type=type_, payload=payload)
        )
        return seq

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> list[EventRecord]:
        found = [r for r in self.records if start <= r.seq <= end]
        return found[:limit]


class _Record:
    """One event as `EventRecord` reads it: whose, where it sits, and what it says.

    `session_id` is carried even though every case here runs one Session at a time,
    because the port declares it and a double that drops a field is a double that would
    let a caller reading the wrong Session's events pass its tests.
    """

    def __init__(
        self,
        session_id: SessionId,
        seq: Seq,
        type: str,
        payload: dict[str, object],
    ) -> None:
        self.session_id = session_id
        self.seq = seq
        self.type = type
        self.payload = payload


def _a_sync(
    workspace: FakeWorkspace,
    tenant_id: TenantId,
    blobs: FakeLaneBlobs | None = None,
    log: ReplayingLog | None = None,
) -> tuple[SyncWorkingLaneAtTurnCompletion, FakeLaneBlobs, ReplayingLog]:
    """A sync over a fake bucket with the real lane store between them.

    `blobs` and `log` are passed in by the cases that run two Turns against one Session:
    the bucket is the durable state and the log is where the digests it was written with
    live, so a second sync built over fresh ones would be a first sync wearing a name.
    """
    bucket = blobs if blobs is not None else FakeLaneBlobs()
    replaying = log if log is not None else ReplayingLog()
    store = SessionVfsStore(bucket, replaying)
    return (
        SyncWorkingLaneAtTurnCompletion(
            workspace, store, FakeSessions(tenant_id), replaying, replaying
        ),
        bucket,
        replaying,
    )


def _listed(name: str, body: bytes) -> ProducedFile:
    return ProducedFile(
        name=name, byte_length=len(body), content_sha256=content_digest(body)
    )


async def test_a_working_file_reaches_the_lane_under_the_session_and_tenant() -> None:
    """The key is composed by the real store, and asserted as a literal.

    Written out rather than rebuilt from `lane_prefix`, for the reason the ship-out
    cases give: a key recomposed from the same helper the code used agrees with it by
    construction and would keep agreeing after somebody reordered the two identifiers.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    workspace = FakeWorkspace(
        (_listed("analysis.py", _SCRIPT),), {"analysis.py": _SCRIPT}
    )
    sync, bucket, _ = _a_sync(workspace, tenant_id)

    await sync.turn_completed(session_id, _TURN)

    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/working/analysis.py": _SCRIPT
    }


async def test_a_nested_working_path_keeps_its_shape() -> None:
    """A working tree is a tree. The separator is the whole reason for the lane path."""
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    module = b"def helper():\n    return 1\n"
    workspace = FakeWorkspace(
        (_listed("src/pkg/helper.py", module),), {"src/pkg/helper.py": module}
    )
    sync, bucket, _ = _a_sync(workspace, tenant_id)

    await sync.turn_completed(session_id, _TURN)

    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/working/src/pkg/helper.py": module
    }


async def test_a_second_turn_that_changed_nothing_transfers_nothing() -> None:
    """The claim the diff exists to make, asserted on the transfer not the bucket.

    Two Turns over one Session, sharing the bucket AND the log -- the log is where the
    digest the first write recorded lives, so a fresh one would make the second sync a
    first sync. `pod.fetched` is empty after the second Turn: the sync did not even ask
    for the bytes, which is the only way to tell "uploaded the same thing again" from
    "recognised it had them".
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    listing = (_listed("analysis.py", _SCRIPT),)
    first = FakeWorkspace(listing, {"analysis.py": _SCRIPT})
    sync, bucket, log = _a_sync(first, tenant_id)
    await sync.turn_completed(session_id, _TURN)
    assert first.fetched == ["analysis.py"]

    second = FakeWorkspace(listing, {"analysis.py": _SCRIPT})
    again, _, _ = _a_sync(second, tenant_id, bucket, log)
    await again.turn_completed(session_id, TurnId(uuid4()))

    assert second.fetched == []
    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/working/analysis.py": _SCRIPT
    }


async def test_a_second_turn_transfers_only_the_file_whose_bytes_changed() -> None:
    """One of three edited, one transfer. The byte count, not the code.

    The two unchanged files are left alone and the edited one is re-read, so what the
    second Turn cost is proportional to what the agent did rather than to how large its
    workspace is. That is the property that makes syncing at EVERY completion
    affordable, which matters because the Turn that mattered is the one before the pod
    died and no schedule can know which that was.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    one, two, three = b"first\n", b"second\n", b"third\n"
    bodies = {"a.py": one, "b.py": two, "c.py": three}
    listing = tuple(_listed(name, body) for name, body in bodies.items())
    first = FakeWorkspace(listing, dict(bodies))
    sync, bucket, log = _a_sync(first, tenant_id)
    await sync.turn_completed(session_id, _TURN)
    assert sorted(first.fetched) == ["a.py", "b.py", "c.py"]

    edited = b"second, rewritten\n"
    after = dict(bodies) | {"b.py": edited}
    second = FakeWorkspace(
        tuple(_listed(name, body) for name, body in after.items()), after
    )
    again, _, _ = _a_sync(second, tenant_id, bucket, log)
    await again.turn_completed(session_id, TurnId(uuid4()))

    assert second.fetched == ["b.py"]
    assert bucket.objects[f"sessions/{tenant_id}/{session_id}/working/b.py"] == edited


async def test_a_pod_reporting_no_digest_is_re_read_rather_than_assumed_unchanged() -> (
    None
):
    """Silence is not "nothing changed"; it is an older shim image.

    Reading a missing digest as unchanged would freeze the lane at whatever it held when
    that pod started -- a wrong answer indistinguishable from a working one, and one
    that costs the agent every edit it made afterwards. So the fallback is the
    expensive, correct branch, and this asserts the transfer happened twice.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    silent = (ProducedFile(name="analysis.py", byte_length=len(_SCRIPT)),)
    assert silent[0].content_sha256 is None
    first = FakeWorkspace(silent, {"analysis.py": _SCRIPT})
    sync, bucket, log = _a_sync(first, tenant_id)
    await sync.turn_completed(session_id, _TURN)

    second = FakeWorkspace(silent, {"analysis.py": _SCRIPT})
    again, _, _ = _a_sync(second, tenant_id, bucket, log)
    await again.turn_completed(session_id, TurnId(uuid4()))

    assert second.fetched == ["analysis.py"]


async def test_a_file_that_vanished_between_the_listing_and_the_fetch_is_skipped() -> (
    None
):
    """Scratch the agent removed after its Turn ended is not a loss, nor a fault."""
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    workspace = FakeWorkspace((_listed("gone.tmp", b"x" * 4),), {})
    sync, bucket, log = _a_sync(workspace, tenant_id)

    await sync.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert log.appended == []


async def test_an_empty_workspace_reads_the_registry_and_the_lane_not_at_all() -> None:
    """A pod that is gone lists nothing, and nothing is the correct amount of work."""
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    workspace = FakeWorkspace(())
    sessions = FakeSessions(tenant_id)
    bucket = FakeLaneBlobs()
    log = ReplayingLog()
    sync = SyncWorkingLaneAtTurnCompletion(
        workspace, SessionVfsStore(bucket, log), sessions, log, log
    )

    await sync.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert sessions.read_for == []


async def test_a_workspace_over_the_path_ceiling_is_synced_in_part_and_says_so() -> (
    None
):
    """A big workspace does not fail a Turn that did good work. It is reported.

    The three claims that matter, together: the files that fit ARE durable, the count is
    exactly the ceiling, and the log carries `vfs.working_lane_partial` naming what was
    left. A refusal here would make the platform's own limit read as the agent's error,
    and silence would read as the platform having lost work.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    over = WORKING_COUNT_LIMIT + 3
    bodies = {f"f{index:05d}.py": b"y" for index in range(over)}
    workspace = FakeWorkspace(
        tuple(_listed(name, body) for name, body in bodies.items()), bodies
    )
    sync, bucket, log = _a_sync(workspace, tenant_id)

    await sync.turn_completed(session_id, _TURN)

    assert len(workspace.fetched) == WORKING_COUNT_LIMIT
    assert len(bucket.objects) == WORKING_COUNT_LIMIT
    partial = [one for one in log.appended if one[0] == vfs.WORKING_LANE_PARTIAL]
    assert len(partial) == 1, log.appended
    assert partial[0][1]["paths_synced"] == WORKING_COUNT_LIMIT
    assert partial[0][1]["paths_left"] == 3
    assert partial[0][1]["path_ceiling"] == WORKING_COUNT_LIMIT


async def test_an_oversized_workspace_converges_across_turns() -> None:
    """The ceiling delays a file; it does not exclude it forever.

    This is the property a "take an arbitrary subset" ceiling would not have, and it
    falls out of the diff rather than being arranged. The first Turn syncs a sorted
    prefix and reports the rest. On the second Turn every path it already stored is
    unchanged, so none of them spends any of the budget -- and the three it left behind
    are the only changed paths there are, so they go. The lane ends up holding the whole
    tree, having never transferred more than one Turn's worth at a time.

    A subset chosen arbitrarily each Turn would instead thrash: every Turn would store a
    different portion, each one evicting nothing and re-transferring much, and the lane
    would never hold a consistent picture of anything.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    over = WORKING_COUNT_LIMIT + 3
    bodies = {f"f{index:05d}.py": b"y" for index in range(over)}
    listing = tuple(_listed(name, body) for name, body in bodies.items())
    first = FakeWorkspace(listing, dict(bodies))
    sync, bucket, log = _a_sync(first, tenant_id)
    await sync.turn_completed(session_id, _TURN)
    assert len(first.fetched) == WORKING_COUNT_LIMIT
    assert first.fetched[0].endswith("f00000.py")

    second = FakeWorkspace(listing, dict(bodies))
    again, _, _ = _a_sync(second, tenant_id, bucket, log)
    await again.turn_completed(session_id, TurnId(uuid4()))

    assert second.fetched == [f"f{index:05d}.py" for index in range(2048, over)]
    assert len(bucket.objects) == over
    assert not [one for one in log.appended[-3:] if one[0] == vfs.WORKING_LANE_PARTIAL]


async def test_a_workspace_over_the_byte_ceiling_stops_at_the_budget() -> None:
    """Bytes bound it as well as count, because one large file is the other shape.

    Two files, each over half the budget, so the count is nowhere near its ceiling and
    the bytes are past theirs. The second is left behind and reported.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    half = WORKING_BUDGET_BYTES // 2 + 1
    listing = (
        ProducedFile(name="a.bin", byte_length=half, content_sha256="a" * 64),
        ProducedFile(name="b.bin", byte_length=half, content_sha256="b" * 64),
    )
    workspace = FakeWorkspace(listing, {"a.bin": b"", "b.bin": b""})
    sync, _, log = _a_sync(workspace, tenant_id)

    await sync.turn_completed(session_id, _TURN)

    assert workspace.fetched == ["a.bin"]
    partial = [one for one in log.appended if one[0] == vfs.WORKING_LANE_PARTIAL]
    assert len(partial) == 1
    assert partial[0][1]["paths_left"] == 1


async def test_the_lane_written_is_the_working_lane_and_not_the_artifacts_lane() -> (
    None
):
    """The two lanes are not interchangeable and this is the guard on which one is used.

    A produced file is sealed and a working file is rewritten every Turn; writing this
    tree into `artifacts` would refuse on the second Turn of every Session and would
    hand the tenant the agent's scratch as deliverables.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    workspace = FakeWorkspace(
        (_listed("analysis.py", _SCRIPT),), {"analysis.py": _SCRIPT}
    )
    sync, bucket, _ = _a_sync(workspace, tenant_id)

    await sync.turn_completed(session_id, _TURN)

    assert all(f"/{WORKING.directory}/" in key for key in bucket.objects)
    assert not any("/artifacts/" in key for key in bucket.objects)
