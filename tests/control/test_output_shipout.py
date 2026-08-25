"""What a completed Turn does with the files its agent wrote.

Two halves, and the split is deliberate. The first drives the ship-out class against a
fake pod and the *real* `FileStore`, because every bound and every refusal is a decision
in that class and none of them needs an HTTP request to exercise. The second drives the
real `PodOutputFetch` against the real shim app over an ASGI transport, so the wire, the
caps and the length check are graded against the route that actually serves them rather
than against a fake that agrees with them by construction.

What neither half shows is that a Kubernetes pod ever answers on this address. Nothing
in this tree places one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import aioboto3  # type: ignore[import-untyped]
import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.s3.session_vfs import SessionVfsStore
from managed_agent.adapters.s3.uploaded_file import S3UploadedFiles
from managed_agent.control.files.output_shipout import (
    OUTPUT_BUDGET_BYTES,
    OUTPUT_COUNT_LIMIT,
    EachAtTurnCompletion,
    OutputsNotShippable,
    ProducedFile,
    ShipOutOutputsAtTurnCompletion,
    WorkspaceOutputs,
)
from managed_agent.control.files.store import (
    DEFAULT_MEDIA_TYPE,
    FileId,
    FileStore,
    FileWindow,
    UploadedFile,
    UploadSizeLimit,
    content_digest,
    new_file_id,
)
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import Placement, PodPhase
from managed_agent.control.session.turn_dispatch import TurnUndeliverable
from managed_agent.core.ids import (
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    TurnId,
    new_session_id,
)
from managed_agent.core.ports import EventLogAppend
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vfs.session_vfs import (
    LaneBlobs,
    LaneEntry,
    ObjectAlreadyPresent,
)
from managed_agent.core.vocabulary import output, vfs
from managed_agent.session_shim.client import RuntimeConnection
from managed_agent.session_shim.pod_channel import PodOutputFetch, shim_token_for
from managed_agent.session_shim.serve import ServedSession, create_shim_app

_THREAD = "0199c4de-6f2a-7b81-9c3d-4e5f60718293"
_KEY = b"a signing key for these cases only"
_NAMESPACE = "map-test"
_REPORT = b"# Report\n\nWhat the agent found.\n"
_TURN = TurnId(uuid4())


class FakeStorage:
    """One dict of objects and one of rows, keyed the way the real store keys them.

    The four operations below raise rather than answer. They are on the port because the
    file family grew a listing and a delete, and nothing in this file performs either --
    a fake that answered them with an empty page or a silent success would let a later
    change reach this file's subject through a path it never meant to exercise.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.rows: dict[FileId, UploadedFile] = {}

    async def write(self, key: str, body: bytes) -> None:
        self.objects[key] = body

    async def read_bytes(self, key: str) -> bytes | None:
        return self.objects.get(key)

    async def record(self, file: UploadedFile) -> None:
        self.rows[file.id] = file

    async def lookup(self, tenant_id: TenantId, file_id: FileId) -> UploadedFile | None:
        found = self.rows.get(file_id)
        return found if found is not None and found.tenant_id == tenant_id else None

    async def page(
        self, tenant_id: TenantId, window: FileWindow, limit: int
    ) -> tuple[UploadedFile, ...]:
        raise AssertionError("this file exercises no listing")

    async def erase(self, key: str) -> None:
        raise AssertionError("this file exercises no deletion")

    async def record_deletion_unless_held(
        self, tenant_id: TenantId, file_id: FileId
    ) -> int:
        raise AssertionError("this file exercises no deletion")

    async def deletion_recorded(self, file_id: FileId) -> bool:
        raise AssertionError("this file exercises no deletion")


class FakeLaneBlobs:
    """One dict of objects, keyed exactly as the real bucket keys them.

    A fake at the *bucket*, not at the lane, so everything above it in this file is the
    real `SessionVfsStore`: the key composition, the conditional create, and the
    `vfs.object_placed` append are the shipping code's own rather than a double's
    imitation of them. What a double at the lane would have proved is that ship-out
    calls a method; what this proves is that a Turn's artifact lands at a key and cannot
    be written twice.

    `put_new` raises the way S3's conditional write does, which is the behaviour every
    re-ship case in this file turns on. `put` is present because the port declares it
    and asserts instead of answering: a sealed lane has no rewrite, so a ship-out that
    reached for one would be reaching past its own guarantee, and a fake that quietly
    succeeded would let it.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put_new(self, key: str, body: bytes) -> None:
        if key in self.objects:
            raise ObjectAlreadyPresent(key)
        self.objects[key] = body

    async def put(self, key: str, body: bytes) -> None:
        raise AssertionError("a sealed lane is never rewritten")

    async def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    async def list_prefix(self, prefix: str) -> Sequence[LaneEntry]:
        raise AssertionError("this file exercises no listing")


_BLOBS_PORT: LaneBlobs = FakeLaneBlobs()
"""Graded against the port by mypy --strict, for the reason `_PORT` below is."""


class FakeSessions:
    """The one read the ship-out makes of the Session registry, and a record of it."""

    def __init__(self, tenant_id: TenantId) -> None:
        self._tenant_id = tenant_id
        self.read_for: list[SessionId] = []

    async def read(self, session_id: SessionId) -> SessionRecord:
        self.read_for.append(session_id)
        return SessionRecord(
            id=session_id,
            tenant_id=self._tenant_id,
            definition_id=DefinitionId(uuid4()),
            definition_revision="1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=1000,
            budget_currency="USD",
            retention_days=30,
        )


class FakePod:
    """A workspace that answers a fixed listing and hands back fixed bodies.

    `served` maps a name to what the pod will send for it, which is not necessarily the
    same as what its listing claimed: that gap is how a pod that under-reports or
    over-reports a length is expressed here. A name absent from `served` answers None,
    which is the route's "that name no longer holds a plain file".
    """

    def __init__(
        self,
        listing: tuple[ProducedFile, ...],
        served: dict[str, bytes] | None = None,
    ) -> None:
        self._listing = listing
        self._served = (
            served
            if served is not None
            else {f.name: b"x" * f.byte_length for f in listing}
        )
        self.fetched: list[tuple[str, int]] = []

    async def list_outputs(self, session_id: SessionId, /) -> tuple[ProducedFile, ...]:
        return self._listing

    async def fetch_output(
        self, session_id: SessionId, name: str, limit_bytes: int, /
    ) -> bytes | None:
        self.fetched.append((name, limit_bytes))
        body = self._served.get(name)
        if body is None:
            return None
        if len(body) > limit_bytes:
            raise TurnUndeliverable(f"{name!r} is larger than {limit_bytes} bytes")
        return body


_PORT: WorkspaceOutputs = FakePod(())
"""Graded against the port by mypy --strict rather than by a runtime check: the Protocol
is not runtime_checkable, and isinstance against one that was would compare method names
and not signatures, which is the half that actually drifts."""


class FakeLog:
    """The Event Log, recording what was appended and in what order.

    A list rather than a counter, because the order is a claim this module makes: the
    append for one file follows that file's store, so a run that raises partway has
    appended exactly the files that are durable. A counter could not tell that from
    appending them all at the end.
    """

    def __init__(self) -> None:
        self.appended: list[tuple[str, dict[str, object]]] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.appended.append((type_, payload))
        return Seq(len(self.appended))


_LOG_PORT: EventLogAppend = FakeLog()
"""Graded against the port by mypy --strict, for the reason `_PORT` above is."""


def _a_ship_out(
    pod: FakePod, tenant_id: TenantId, blobs: FakeLaneBlobs | None = None
) -> tuple[ShipOutOutputsAtTurnCompletion, FakeLaneBlobs, FakeSessions, FakeLog]:
    """A ship-out over a fake bucket, with the real lane store between them.

    `blobs` is passed in by the cases that run two Turns against one Session: the bucket
    IS the durable state a second Turn collides with, so a second ship-out built over a
    fresh one would find an empty lane and prove nothing about a re-ship.
    """
    bucket = FakeLaneBlobs() if blobs is None else blobs
    sessions = FakeSessions(tenant_id)
    log = FakeLog()
    return (
        ShipOutOutputsAtTurnCompletion(
            pod, SessionVfsStore(bucket, log), sessions, log
        ),
        bucket,
        sessions,
        log,
    )


def _announcements(log: FakeLog) -> list[Mapping[str, object]]:
    """Just the `output.produced` payloads, in order.

    Filtered rather than indexed, because each shipped file now appends two events --
    `vfs.object_placed` from the lane store and this one from ship-out -- and a case
    about what the tenant is told should not also be asserting how many events the layer
    below it wrote.
    """
    produced = output.OUTPUT_PRODUCED
    return [payload for type_, payload in log.appended if type_ == produced]


async def test_a_file_the_agent_wrote_lands_in_the_sessions_artifacts_lane() -> None:
    """The whole feature in one case: bytes in the pod, object at a key in the lane.

    **The key is written out as a literal rather than composed from `lane_prefix`.** A
    key rebuilt here from the same parts the code builds it from agrees with itself by
    construction and would keep agreeing after somebody reordered the tenant and the
    Session -- which is the one part of this layout that carries a guarantee, because a
    tenant-first prefix is what makes a tenant-wide sweep a prefix operation and what
    makes another tenant's object unreachable from a Session id alone.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    pod = FakePod(
        (ProducedFile(name="report.md", byte_length=len(_REPORT)),),
        {"report.md": _REPORT},
    )
    ship_out, bucket, _, _ = _a_ship_out(pod, tenant_id)

    await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/artifacts/report.md": _REPORT
    }


async def test_a_deliverable_one_directory_down_keeps_its_shape() -> None:
    """The acceptance case for the whole change: a nested path, stored nested.

    A flat file passing proves nothing this change made -- it shipped before. What could
    not happen before is `report/fig1.png`: the destination was a flat upload namespace
    keyed by a filename, and a filename cannot hold a separator, so the shim filtered
    such a file out of its own listing and the tenant simply never received it. A report
    with its figures beside it is the ordinary shape of a deliverable, and this is the
    case that says it survives.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    figure = b"\x89PNG\r\n\x1a\n and then some"
    pod = FakePod(
        (
            ProducedFile(name="report/fig1.png", byte_length=len(figure)),
            ProducedFile(name="report/index.md", byte_length=len(_REPORT)),
        ),
        {"report/fig1.png": figure, "report/index.md": _REPORT},
    )
    ship_out, bucket, _, log = _a_ship_out(pod, tenant_id)

    await ship_out.turn_completed(session_id, _TURN)

    root = f"sessions/{tenant_id}/{session_id}/artifacts"
    assert bucket.objects == {
        f"{root}/report/fig1.png": figure,
        f"{root}/report/index.md": _REPORT,
    }
    assert [one["path"] for one in _announcements(log)] == [
        "report/fig1.png",
        "report/index.md",
    ]


async def test_a_second_turn_that_changed_nothing_ships_and_announces_nothing() -> None:
    """Without this, the second Turn of every Session that ever produced a file fails.

    Nothing clears the agent's output directory between Turns, and nothing should: a
    route that reached into a pod to delete a tenant's files would destroy the
    expectation that `out/` accumulates across a Session. So Turn 2 re-offers Turn 1's
    files at the same paths, and a sealed lane refuses every one of them. Against the
    upload store this used to be a harmless duplicate row; against a seal it is an
    unhandled refusal on a Turn that did nothing wrong.

    Announcing nothing is the second half and is not merely tidiness. An
    `output.produced` per Turn naming an unchanged file tells a tenant polling the log
    that their document was produced again on a Turn that never touched it.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    listing = (ProducedFile(name="report.md", byte_length=len(_REPORT)),)
    bodies = {"report.md": _REPORT}
    bucket = FakeLaneBlobs()

    first, _, _, first_log = _a_ship_out(FakePod(listing, bodies), tenant_id, bucket)
    await first.turn_completed(session_id, _TURN)
    assert len(_announcements(first_log)) == 1

    second, _, _, second_log = _a_ship_out(FakePod(listing, bodies), tenant_id, bucket)
    await second.turn_completed(session_id, TurnId(uuid4()))

    assert _announcements(second_log) == []
    assert second_log.appended == []
    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/artifacts/report.md": _REPORT
    }


async def test_a_second_turn_that_rewrote_a_delivered_artifact_is_refused() -> None:
    """The other side of the case above, and the reason it cannot just be swallowed.

    Identical bytes at an occupied key mean the artifact is already delivered. Different
    bytes mean the agent revised something a tenant may already have downloaded and
    checked against a recorded digest -- and the seal's whole promise is that the digest
    stays true. Treating both as "already shipped" would drop the new version silently,
    which is the failure mode the seal exists to make impossible.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    bucket = FakeLaneBlobs()
    first, _, _, _ = _a_ship_out(
        FakePod(
            (ProducedFile(name="report.md", byte_length=len(_REPORT)),),
            {"report.md": _REPORT},
        ),
        tenant_id,
        bucket,
    )
    await first.turn_completed(session_id, _TURN)

    revised = b"# Report\n\nWhat the agent found, on reflection.\n"
    second, _, _, log = _a_ship_out(
        FakePod(
            (ProducedFile(name="report.md", byte_length=len(revised)),),
            {"report.md": revised},
        ),
        tenant_id,
        bucket,
    )

    with pytest.raises(OutputsNotShippable, match="cannot be revised"):
        await second.turn_completed(session_id, TurnId(uuid4()))

    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/artifacts/report.md": _REPORT
    }
    assert log.appended == []


async def test_the_shipped_file_is_announced_with_the_path_that_downloads_it() -> None:
    """Without this event the bytes are stored and unreachable, which is the point.

    Nothing else in the platform tells the tenant which path a Turn wrote to: the
    resources listing answers with the files the Session was CREATED with, and there is
    deliberately no route that lists a lane. So the path in this payload is the only
    route from "my agent wrote a document" to those bytes.

    Asserted over the whole appended list rather than over a filtered one, because the
    ORDER of the two events matters and is not obvious: `vfs.object_placed` comes first,
    from inside `place`, so the announcement is never in the log ahead of the object it
    names. A reader folding this log to answer "does this artifact exist" never sees a
    Turn where it does not.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    pod = FakePod(
        (ProducedFile(name="report.md", byte_length=len(_REPORT)),),
        {"report.md": _REPORT},
    )
    ship_out, _, _, log = _a_ship_out(pod, tenant_id)

    await ship_out.turn_completed(session_id, _TURN)

    assert [type_ for type_, _ in log.appended] == [
        vfs.OBJECT_PLACED,
        output.OUTPUT_PRODUCED,
    ]
    assert _announcements(log) == [{"path": "report.md", "byte_length": len(_REPORT)}]


async def test_two_files_are_announced_separately_in_the_order_they_shipped() -> None:
    """One event per file, not one per Turn, and the order is the order they were made
    durable -- so a tenant that wanted the second document is not folding a list out of
    a single event, and a run that raised partway names exactly what survived."""
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    listing = (
        ProducedFile(name="one.md", byte_length=3),
        ProducedFile(name="two.md", byte_length=5),
    )
    pod = FakePod(listing, {"one.md": b"aaa", "two.md": b"bbbbb"})
    ship_out, _, _, log = _a_ship_out(pod, tenant_id)

    await ship_out.turn_completed(session_id, _TURN)

    assert [one["path"] for one in _announcements(log)] == ["one.md", "two.md"]
    assert [one["byte_length"] for one in _announcements(log)] == [3, 5]


async def test_a_turn_that_produced_nothing_announces_nothing() -> None:
    """The absence of the event is what means "no document", so it has to be an absence.

    An `output.produced` carrying zero files, or a count of them, would make a tenant
    parse a payload to learn something the empty log already says -- and would make the
    common case (a Turn that answers in text) append an event on every Turn.
    """
    ship_out, _, _, log = _a_ship_out(FakePod(()), TenantId(uuid4()))

    await ship_out.turn_completed(new_session_id(), _TURN)

    assert log.appended == []


async def test_a_file_that_vanished_is_not_announced() -> None:
    """A name the pod listed and then could not serve is skipped, and skipping it must
    also skip the announcement: an event naming a file id that was never stored sends a
    tenant to `GET /v1/files/{id}/content` for a 404 they cannot explain."""
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    pod = FakePod((ProducedFile(name="gone.md", byte_length=4),), {})
    ship_out, bucket, _, log = _a_ship_out(pod, tenant_id)

    await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert log.appended == []


async def test_two_sessions_of_one_tenant_do_not_share_an_artifact_path() -> None:
    """What the tenant/Session key layout buys, asserted rather than assumed.

    The path an agent writes is text the model chose, so two Sessions of one tenant
    routinely produce `report.md`. If the Session were not in the key, the second
    Session's ship-out would land on the first's object -- and against a SEALED lane
    that is not a silent overwrite but a refused Turn, so the failure would present as
    one tenant's Session being broken by an unrelated one.

    This is also the negative half of the re-ship cases above: the same path colliding
    is correct WITHIN a Session and must not happen ACROSS two.
    """
    tenant_id = TenantId(uuid4())
    one, two = new_session_id(), new_session_id()
    bucket = FakeLaneBlobs()
    listing = (ProducedFile(name="report.md", byte_length=len(_REPORT)),)

    for session_id in (one, two):
        ship_out, _, _, _ = _a_ship_out(
            FakePod(listing, {"report.md": _REPORT}), tenant_id, bucket
        )
        await ship_out.turn_completed(session_id, _TURN)

    assert sorted(bucket.objects) == sorted(
        (
            f"sessions/{tenant_id}/{one}/artifacts/report.md",
            f"sessions/{tenant_id}/{two}/artifacts/report.md",
        )
    )


async def test_a_tenants_own_upload_still_names_no_session() -> None:
    """The negative half of the field above, through the path that has always existed.

    Without this, a column that was always populated would satisfy the case above and
    would tell a listing nothing: the distinction only exists if an upload leaves it
    null.
    """
    tenant_id = TenantId(uuid4())
    storage = FakeStorage()
    store = FileStore(storage, UploadSizeLimit(OUTPUT_BUDGET_BYTES))

    async def _body() -> AsyncIterator[bytes]:
        yield b"the tenant sent this"

    record = await store.store(
        tenant_id=tenant_id,
        filename="brief.md",
        media_type="text/markdown",
        chunks=_body(),
    )
    assert record.produced_in_session_id is None


async def test_a_turn_that_produced_nothing_asks_the_store_and_registry_nothing() -> (
    None
):
    """Most Turns answer in text, so this is the common path rather than a guard.

    The registry read is asserted absent as well as the write, because a ship-out that
    resolved the tenant before finding out there was nothing to ship would put a
    database round trip on every Turn the platform runs.
    """
    session_id = new_session_id()
    pod = FakePod(())
    ship_out, bucket, sessions, log = _a_ship_out(pod, TenantId(uuid4()))

    await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert sessions.read_for == []
    assert pod.fetched == []


async def test_more_files_than_the_limit_refuses_and_ships_none_of_them() -> None:
    """The count bound, exceeded on purpose so the refusal is measured.

    Nothing stored is the half that matters. A ship-out that refused after storing the
    first thirty-two would leave a tenant holding an arbitrary subset of what the agent
    produced, with no way to know which files are missing or that any are.
    """
    session_id = new_session_id()
    listing = tuple(
        ProducedFile(name=f"file-{index:03d}.md", byte_length=1)
        for index in range(OUTPUT_COUNT_LIMIT + 1)
    )
    pod = FakePod(listing)
    ship_out, bucket, sessions, log = _a_ship_out(pod, TenantId(uuid4()))

    with pytest.raises(OutputsNotShippable, match="more than"):
        await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert sessions.read_for == []
    assert pod.fetched == []


async def test_exactly_the_limit_is_admitted() -> None:
    """The boundary on the other side of the refusal above.

    Without this the bound could be off by one in the direction that refuses a
    legitimate Turn, and the refusal case would pass either way.
    """
    session_id = new_session_id()
    listing = tuple(
        ProducedFile(name=f"file-{index:03d}.md", byte_length=1)
        for index in range(OUTPUT_COUNT_LIMIT)
    )
    ship_out, bucket, _, _ = _a_ship_out(FakePod(listing), TenantId(uuid4()))

    await ship_out.turn_completed(session_id, _TURN)

    assert len(bucket.objects) == OUTPUT_COUNT_LIMIT


async def test_more_bytes_than_the_budget_refuses_before_any_transfer() -> None:
    """The byte bound, and it is checked against the listing rather than the transfer.

    That ordering is the point: two files each inside the budget can exceed it together,
    and a ship-out that discovered this on the second file would already have stored the
    first. `pod.fetched` being empty is what says nothing was pulled at all.
    """
    session_id = new_session_id()
    half = OUTPUT_BUDGET_BYTES // 2 + 1
    listing = (
        ProducedFile(name="a.bin", byte_length=half),
        ProducedFile(name="b.bin", byte_length=half),
    )
    ship_out, bucket, sessions, _ = _a_ship_out(FakePod(listing, {}), TenantId(uuid4()))

    with pytest.raises(OutputsNotShippable, match="bytes"):
        await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert sessions.read_for == []


async def test_each_fetch_is_capped_at_the_length_its_own_listing_reported() -> None:
    """The tightest cap available, and it is free because the listing already said.

    Capping at the whole budget instead would let a pod that reported a one-byte file
    send sixty-four mebibytes and be inside its cap the whole way. What is asserted is
    the limit each fetch was given, per file, so a single shared cap fails this.
    """
    session_id = new_session_id()
    listing = (
        ProducedFile(name="small.md", byte_length=3),
        ProducedFile(name="larger.md", byte_length=11),
    )
    pod = FakePod(listing, {"small.md": b"abc", "larger.md": b"hello world"})
    ship_out, _, _, log = _a_ship_out(pod, TenantId(uuid4()))

    await ship_out.turn_completed(session_id, _TURN)

    assert pod.fetched == [("small.md", 3), ("larger.md", 11)]


async def test_a_body_shorter_than_its_listing_is_refused_rather_than_stored() -> None:
    """The half a cap cannot do, and the one that would be undetectable afterwards.

    The store hashes whatever it is handed, so a truncated document would be recorded
    with a digest that certifies the truncation: every later download would check out
    and the tenant would read half a report as a whole one.
    """
    session_id = new_session_id()
    pod = FakePod(
        (ProducedFile(name="report.md", byte_length=len(_REPORT)),),
        {"report.md": _REPORT[:10]},
    )
    ship_out, bucket, _, _ = _a_ship_out(pod, TenantId(uuid4()))

    with pytest.raises(OutputsNotShippable, match="not the document"):
        await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {}


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.md",
        "sub/../../escaped.md",
        "/etc/passwd",
        "..",
        ".",
        "",
        ".codex/state.json",
        "out/.map/lib/thing.py",
        "report/.hidden/fig.png",
        "trailing/",
        "double//segment.md",
        'quote"d.md',
        "back\\.md",
    ],
)
async def test_a_path_the_pod_should_not_have_offered_refuses_the_turn(
    name: str,
) -> None:
    """A pod sending one of these is not a pod with an awkwardly named document.

    The shim filters its own listing by the same rule -- literally the same function --
    so every one of these arriving here means the process on the other end is not the
    one this platform placed. Each entry encodes its own decision, which is why they are
    parametrized one case per name rather than asserted in a loop:

    The first three escape the lane. `..` at the front, `..` reached through a real
    directory, and an absolute path all compose a key outside this Session's prefix, and
    the middle one is the case a rule that only inspected the first segment would let
    through -- exactly the shape a nested path makes newly reachable.

    `..`, `.` and the empty name compose a key naming a prefix rather than an object.

    The three dotted ones are the rule the lane grammar cannot express: a dot is legal
    inside a path, so `.codex/state.json` parses as a lane path and is runtime scratch,
    `out/.map/lib/thing.py` is an installed dependency tree, and
    `report/.hidden/fig.png` is the same rule at the second segment rather than the
    first.

    A trailing separator and a doubled one both compose to a key no read reconstructs
    from the path that produced it.

    The quote and the backslash are what survives of the upload grammar's protections
    that the lane grammar keeps: both would go verbatim into a Content-Disposition
    header.

    Refusing rather than skipping is the deliberate choice, and the reason it does not
    wedge a Session is the filter at the pod: a real workspace cannot produce this.
    """
    session_id = new_session_id()
    pod = FakePod((ProducedFile(name=name, byte_length=1),), {name: b"x"})
    ship_out, bucket, _, log = _a_ship_out(pod, TenantId(uuid4()))

    with pytest.raises(OutputsNotShippable):
        await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert log.appended == []
    assert pod.fetched == []


@pytest.mark.parametrize(
    "name", ["report.md", "report/fig1.png", "a/b/c/deep.txt", "with.dots/in.name.md"]
)
async def test_a_path_a_real_workspace_produces_is_admitted(name: str) -> None:
    """The other side of the refusals above, and the half that would go unnoticed.

    A guard list is only meaningful against the paths it must NOT refuse. Without this,
    tightening the rule until it refused everything would leave every case above green
    and this whole route silently dead -- which is close to what the flat-only rule was
    doing to `report/fig1.png` before this change.

    `with.dots/in.name.md` is here because the dotted-segment rule is about a leading
    dot, and a rule written as "no dot anywhere" would pass every refusal case above
    while refusing the ordinary name of a figure.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    pod = FakePod((ProducedFile(name=name, byte_length=1),), {name: b"x"})
    ship_out, bucket, _, log = _a_ship_out(pod, tenant_id)

    await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/artifacts/{name}": b"x"
    }
    assert [one["path"] for one in _announcements(log)] == [name]


async def test_a_file_that_vanished_between_the_listing_and_the_fetch_is_skipped() -> (
    None
):
    """The one absence that is not a failure, and the others still ship.

    An agent that tidies up after its own Turn unlinks a file it just listed. There is
    no document in that to lose, and refusing would fail a Turn over a file the agent
    itself deleted -- while the report beside it, which the tenant actually asked for,
    would go unshipped.
    """
    session_id = new_session_id()
    listing = (
        ProducedFile(name="scratch.tmp", byte_length=5),
        ProducedFile(name="report.md", byte_length=len(_REPORT)),
    )
    pod = FakePod(listing, {"report.md": _REPORT})
    ship_out, bucket, _, log = _a_ship_out(pod, TenantId(uuid4()))

    await ship_out.turn_completed(session_id, _TURN)

    assert [key.rsplit("/", 1)[-1] for key in bucket.objects] == ["report.md"]
    assert [one["path"] for one in _announcements(log)] == ["report.md"]


async def test_one_file_larger_than_the_whole_turns_budget_is_refused() -> None:
    """The per-Turn budget is now the only ceiling on a produced file, so it has to
    catch the single-file case as well as the several-files-together one.

    **This case replaces one that no longer has a subject, and the difference is a real
    capability change worth stating.** A produced file used to be minted as an upload,
    so `UploadSizeLimit` -- the bound on what a tenant may POST -- applied to it as a
    second, possibly tighter ceiling: a deployment capping uploads at a mebibyte capped
    agent output there too. A lane has no such per-object bound, so what governs now is
    `OUTPUT_BUDGET_BYTES` alone. That is the more honest place for the rule (what an
    agent may produce is not the same question as what a tenant may upload) and it is
    strictly more permissive, so it is written down rather than left to be discovered.

    The refusal lands before any transfer, which is what `pod.fetched` says: a file the
    platform will not keep should not first be pulled into this worker's memory.
    """
    session_id = new_session_id()
    pod = FakePod(
        (ProducedFile(name="big.bin", byte_length=OUTPUT_BUDGET_BYTES + 1),), {}
    )
    ship_out, bucket, sessions, log = _a_ship_out(pod, TenantId(uuid4()))

    with pytest.raises(OutputsNotShippable, match="bytes"):
        await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert log.appended == []
    assert pod.fetched == []
    assert sessions.read_for == []


class Recording:
    """A completion seam that records that it ran, or raises instead."""

    def __init__(self, log: list[str], label: str, fails: bool = False) -> None:
        self._log = log
        self._label = label
        self._fails = fails

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        self._log.append(self._label)
        if self._fails:
            raise RuntimeError(self._label)


async def test_the_seams_a_completed_turn_owes_run_in_the_order_they_were_given() -> (
    None
):
    """Order is a decision here, not an artefact of iteration.

    The Rollout's ship-out is meant to go first, because losing a Session's resume state
    costs every later Turn while losing a document costs that document. A composite that
    iterated a set, or a dict, would satisfy a test that only checked both ran.
    """
    ran: list[str] = []
    seams = EachAtTurnCompletion(Recording(ran, "rollout"), Recording(ran, "outputs"))

    await seams.turn_completed(new_session_id(), _TURN)

    assert ran == ["rollout", "outputs"]


async def test_a_seam_that_raises_stops_the_ones_behind_it() -> None:
    """Which is why the recoverable loss is placed second and not first.

    If this did not hold, the order above would buy nothing: a failure to ship one
    document would be followed by the Rollout's ship-out regardless and the ordering
    would be decoration.
    """
    ran: list[str] = []
    seams = EachAtTurnCompletion(
        Recording(ran, "first", fails=True), Recording(ran, "second")
    )

    with pytest.raises(RuntimeError, match="first"):
        await seams.turn_completed(new_session_id(), _TURN)

    assert ran == ["first"]


class FixedPhase:
    """A cluster reporting one phase for every pod, and starting nothing."""

    def __init__(self, phase: PodPhase = PodPhase.RUNNING) -> None:
        self._phase = phase

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        raise AssertionError("shipping outputs tried to start a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        return self._phase

    async def remove(self, pod_name: str) -> None:
        raise AssertionError("shipping outputs tried to remove a pod")


def _a_pod_serving(
    monkeypatch: pytest.MonkeyPatch, produced: Path, session_id: SessionId
) -> httpx.AsyncBaseTransport:
    """The real shim app for one Session, reachable over an ASGI transport.

    `WORKSPACE_READ_ROOT` is redirected on the module the routes read it from, because
    the pod path is a deployment constant everywhere except under a test. The runtime
    connection is constructed and never dialled: neither output route touches it.
    """
    monkeypatch.setattr(
        "managed_agent.session_shim.serve.WORKSPACE_READ_ROOT", produced
    )
    app = create_shim_app(
        ServedSession(
            session_id=session_id,
            thread_id=_THREAD,
            connection=RuntimeConnection(produced / "never-dialled.sock"),
            token=shim_token_for(session_id, _KEY),
        )
    )
    return httpx.ASGITransport(app=app)


def _a_fetch(transport: httpx.AsyncBaseTransport) -> PodOutputFetch:
    return PodOutputFetch(Placement(FixedPhase()), _NAMESPACE, _KEY, transport)


async def test_the_real_route_and_the_real_fetch_agree_about_one_produced_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves at once: the route lists and serves, the fetch reads and parses.

    A fake pod agrees with the control plane by construction. This is the case that
    would catch the two ends disagreeing about the route, the JSON shape or the status
    the listing answers.

    **It is also the guard that the two spellings of SHA-256 agree.** The shim hashes in
    chunks with `hashlib` and does not import `content_digest`, deliberately, so that a
    pod-side module owes nothing to a control-plane one for a hash. Two spellings of one
    rule are two rules that can drift, and this compares them over real bytes read by
    the real route -- so a change to either side that moved the hex fails here.
    """
    session_id = new_session_id()
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    fetch = _a_fetch(_a_pod_serving(monkeypatch, produced, session_id))

    listed = await fetch.list_outputs(session_id)

    assert listed == (
        ProducedFile(
            name="report.md",
            byte_length=len(_REPORT),
            content_sha256=content_digest(_REPORT),
        ),
    )
    assert await fetch.fetch_output(session_id, "report.md", len(_REPORT)) == _REPORT


async def test_a_file_that_grew_past_its_listed_length_is_refused_on_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap against the real route, exceeded rather than described.

    The file is listed, then grown, then fetched under the length the listing gave -- so
    the route serves more bytes than the cap admits and the read stops. Without the cap
    this would put an arbitrary number of bytes into a control-plane worker on the word
    of the pod's own listing.
    """
    session_id = new_session_id()
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    fetch = _a_fetch(_a_pod_serving(monkeypatch, produced, session_id))
    listed = await fetch.list_outputs(session_id)
    (produced / "report.md").write_bytes(_REPORT + b"x" * 4096)

    with pytest.raises(TurnUndeliverable, match="larger than"):
        await fetch.fetch_output(session_id, "report.md", listed[0].byte_length)


async def test_a_file_that_shrank_is_carried_back_short_for_the_caller_to_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wire cannot tell a short body from a small file, so the caller checks.

    This is the seam between the two halves of this file: `PodOutputFetch` returns what
    the route served, and the ship-out class is what compares it against the listing.
    Asserting the short read here is what makes the refusal there reachable in
    production rather than only against a fake.
    """
    session_id = new_session_id()
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    fetch = _a_fetch(_a_pod_serving(monkeypatch, produced, session_id))
    listed = await fetch.list_outputs(session_id)
    (produced / "report.md").write_bytes(b"short")

    body = await fetch.fetch_output(session_id, "report.md", listed[0].byte_length)

    assert body == b"short"
    assert body is not None and len(body) != listed[0].byte_length


async def test_a_pod_that_is_not_running_lists_nothing_rather_than_failing_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Turn cannot have completed on a pod that is gone, so nothing is there to ship.

    Raising here would fail a Turn that did complete and whose events are already in the
    log, over a pod that died after its last event.
    """
    session_id = new_session_id()
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    monkeypatch.setattr(
        "managed_agent.session_shim.serve.WORKSPACE_READ_ROOT", produced
    )
    fetch = PodOutputFetch(
        Placement(FixedPhase(PodPhase.GONE)),
        _NAMESPACE,
        _KEY,
        _NeverDialled(),
    )

    assert await fetch.list_outputs(session_id) == ()


async def test_a_pod_that_vanished_part_way_through_refuses_rather_than_half_shipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asymmetry with the case above, and the reason for it.

    By the time a fetch runs, this Session's files have been listed and some may already
    be stored. A ship-out that stops half way leaves a tenant holding three documents of
    five with nothing saying so, which is worse than a Turn recorded as failed.
    """
    session_id = new_session_id()
    fetch = PodOutputFetch(
        Placement(FixedPhase(PodPhase.GONE)), _NAMESPACE, _KEY, _NeverDialled()
    )

    with pytest.raises(TurnUndeliverable, match="part way through"):
        await fetch.fetch_output(session_id, "report.md", 10)


async def test_a_listing_that_is_not_one_closes_the_turn_rather_than_reading_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "I could not read that" is not "there was nothing there".

    Reading an unparseable listing as empty would drop every document a Turn produced,
    silently, through the one channel nobody would think to check -- and the pod is the
    untrusted end of this connection.
    """
    session_id = new_session_id()

    async def _not_a_listing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"files": "not a list"}')

    fetch = _a_fetch(httpx.MockTransport(_not_a_listing))

    with pytest.raises(TurnUndeliverable, match="listing"):
        await fetch.list_outputs(session_id)


async def test_a_listing_padded_past_its_cap_is_refused_on_the_way_in() -> None:
    """The listing has its own bound, derived from the count limit rather than chosen.

    A pod padding its JSON with whitespace would otherwise put an unbounded body into a
    control-plane worker before anything could look at how many entries it held.
    """
    session_id = new_session_id()

    async def _enormous(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b" " * (2 * 1024 * 1024) + b"{}")

    fetch = _a_fetch(httpx.MockTransport(_enormous))

    with pytest.raises(TurnUndeliverable, match="output listing"):
        await fetch.list_outputs(session_id)


class _NeverDialled(httpx.AsyncBaseTransport):
    """A transport that fails the test if anything is sent over it."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the pod was dialled at {request.url}")


def _shipped_storage(engine: AsyncEngine) -> S3UploadedFiles:
    """The real adapter against a real database. No object call is made by these cases.

    The bucket name is never dialled by anything below, so any string does: what is
    graded here is the column, the statement and the migration.
    """
    return S3UploadedFiles(aioboto3.Session(), "no-bucket-is-reached-here", engine)


@pytest.mark.parametrize("produced_in", [True, False])
async def test_which_session_produced_a_file_survives_the_real_round_trip(
    engine: AsyncEngine, produced_in: bool
) -> None:
    """Both values of the one field, against the real column the real migration adds.

    Parametrized over the two because each is a separate claim and neither implies the
    other: a statement that dropped the column would pass the null case, and one that
    wrote a constant would pass the populated one. The row is compared whole, so a value
    that came back as a string where a `SessionId` was written fails here rather than at
    whatever reads it next.
    """
    tenant = TenantId(uuid4())
    session_id = new_session_id() if produced_in else None
    written = UploadedFile(
        id=new_file_id(),
        tenant_id=tenant,
        filename="report.md",
        media_type=DEFAULT_MEDIA_TYPE,
        byte_length=len(_REPORT),
        content_sha256=content_digest(_REPORT),
        produced_in_session_id=session_id,
    )

    storage = _shipped_storage(engine)
    await storage.record(written)

    read_back = await storage.lookup(tenant, written.id)
    assert read_back == written
    assert read_back is not None
    assert read_back.produced_in_session_id == session_id


async def test_a_body_matching_the_listed_digest_is_stored() -> None:
    """The pod reports a digest, the bytes hash to it, and the file lands.

    The happy half of the pair below. Asserted separately from the refusal because a
    verification that refused everything would pass the refusal case alone, and this is
    the case that says the check can be satisfied at all.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    pod = FakePod(
        (
            ProducedFile(
                name="report.md",
                byte_length=len(_REPORT),
                content_sha256=content_digest(_REPORT),
            ),
        ),
        {"report.md": _REPORT},
    )
    ship_out, bucket, _, _ = _a_ship_out(pod, tenant_id)

    await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/artifacts/report.md": _REPORT
    }


async def test_a_body_at_the_listed_length_but_not_the_listed_digest_is_refused() -> (
    None
):
    """The failure a length check cannot see, and the reason the digest is carried.

    Both bodies are the same number of bytes, so every length assertion in this file
    passes on this input. What differs is the content -- a document rewritten between
    the moment the pod listed it and the moment the control plane fetched it. Nothing
    here knows which of the two the tenant meant, so it is a refusal rather than a
    repair, and nothing is stored under a name whose bytes are in doubt.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    rewritten = bytes(len(_REPORT))
    assert len(rewritten) == len(_REPORT) and rewritten != _REPORT
    pod = FakePod(
        (
            ProducedFile(
                name="report.md",
                byte_length=len(_REPORT),
                content_sha256=content_digest(_REPORT),
            ),
        ),
        {"report.md": rewritten},
    )
    ship_out, bucket, _, log = _a_ship_out(pod, tenant_id)

    with pytest.raises(OutputsNotShippable, match="not the document"):
        await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {}
    assert _announcements(log) == []


async def test_a_pod_that_reports_no_digest_still_ships() -> None:
    """A Session's pod runs the shim image it started with, so absence is not a fault.

    A control plane that required the digest would fail every Turn already in flight the
    moment it rolled -- the pods answering those Turns were started before the field
    existed and will answer without it for the rest of their Sessions. So `None` means
    "this pod does not report digests" and the file ships on its length alone, which is
    exactly what it shipped on before the field was added.

    Written as its own case rather than left to the twenty-odd listings above that omit
    the field, because those omit it incidentally and this asserts it on purpose: if the
    default ever became a computed digest, they would keep passing and this would fail.
    """
    session_id, tenant_id = new_session_id(), TenantId(uuid4())
    listed = ProducedFile(name="report.md", byte_length=len(_REPORT))
    assert listed.content_sha256 is None
    ship_out, bucket, _, _ = _a_ship_out(
        FakePod((listed,), {"report.md": _REPORT}), tenant_id
    )

    await ship_out.turn_completed(session_id, _TURN)

    assert bucket.objects == {
        f"sessions/{tenant_id}/{session_id}/artifacts/report.md": _REPORT
    }
