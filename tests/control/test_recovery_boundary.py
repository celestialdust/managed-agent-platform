"""The recovery boundary, exercised over a fake bucket and hand-built Rollout bytes.

Each case is one sentence of ADR-004 turned into an assertion. The bytes are built here
rather than produced by a real Agent Runtime because the property under test is the cut
and the round trip, and a real one would make the input unreproducible run to run. What
the real runtime writes is a live question this slice cannot settle -- no test in this
repository has run the binary -- and the lifecycle names below are the ones
`research/plan-pass-rollout-and-resume.md` cites with a line number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

import httpx
import pytest

from managed_agent.composition import _BUCKET_ENV, _RolloutNotYetShipped, build
from managed_agent.control.files.rollout_sync import (
    TURN_CLOSED_ABORTED,
    TURN_CLOSED_COMPLETE,
    TURN_OPENED,
    MalformedRollout,
    RolloutSync,
    ShipOutAtTurnCompletion,
    rollout_key,
    truncate_at_last_completed_turn,
)
from managed_agent.control.files.store import UPLOAD_KEY_ROOT
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import (
    Placement,
    PodPhase,
)
from managed_agent.control.session.turn_dispatch import TurnDispatch, TurnUndeliverable
from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, TurnId, new_turn_id
from managed_agent.core.session.markers import DiscardCause, WorkDiscarded, discard
from managed_agent.core.vocabulary import turn
from managed_agent.core.vocabulary.marker import WORK_DISCARDED
from managed_agent.session_shim.client import RuntimeConnection
from managed_agent.session_shim.pod_channel import (
    HttpPodDispatch,
    PodRolloutFetch,
    shim_token_for,
)
from managed_agent.session_shim.serve import ServedSession, create_shim_app

_THREAD = "0199c4de-6f2a-7b81-9c3d-4e5f60718293"
_KEY = b"a signing key for these cases only"
_NAMESPACE = "map-sessions"


def _line(kind: str, payload: dict[str, object], ordinal: int) -> bytes:
    return json.dumps(
        {
            "timestamp": "2026-08-22T10:00:00.000Z",
            "ordinal": ordinal,
            "type": kind,
            "payload": payload,
        }
    ).encode("utf-8")


def _meta(ordinal: int = 0) -> bytes:
    return _line("session_meta", {"id": str(uuid4())}, ordinal)


def _event(name: str, ordinal: int) -> bytes:
    return _line("event_msg", {"type": name}, ordinal)


def _said(text: str, ordinal: int) -> bytes:
    return _line("response_item", {"type": "message", "text": text}, ordinal)


def _rollout(*lines: bytes) -> bytes:
    return b"\n".join(lines) + b"\n"


_EM_DASH = "—"
_EM_DASH_BYTES = _EM_DASH.encode("utf-8")


def _cut_mid_character(kind: str, ordinal: int) -> bytes:
    """A line severed inside a multi-byte character, as a half-flushed write leaves one.

    `ensure_ascii=False` because that is what the runtime's serializer does: serde_json
    emits non-ASCII raw, so this em-dash is three bytes on disk and a write that stopped
    between them leaves bytes no decoder will accept. The other helpers here go through
    the default, which escapes it to an ASCII sequence -- a tail built that way is torn
    for the parser and perfectly decodable, which is the case this one is not.
    """
    whole = json.dumps(
        {
            "timestamp": "2026-08-22T10:00:00.000Z",
            "ordinal": ordinal,
            "type": kind,
            "payload": {"type": "message", "text": f"usage {_EM_DASH} 1200 tokens"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    assert _EM_DASH_BYTES in whole, "the serializer escaped what this case needs raw"
    return whole[: whole.index(_EM_DASH_BYTES) + 2]


class InMemoryRolloutStore:
    """A dict standing in for the bucket, recording puts so replacement is observable.

    It offers no delete, which is the point: `RolloutSync` holds only this surface, so a
    version of it that tried to remove an object would not type-check.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts: list[str] = []

    async def put(self, key: str, body: bytes) -> None:
        self.objects[key] = body
        self.puts.append(key)

    async def get(self, key: str) -> bytes | None:
        return self.objects.get(key)


class FixedFetch:
    """A `RolloutFetch` answering with what a pod would be holding."""

    def __init__(self, body: bytes | None) -> None:
        self._body = body
        self.asked: list[SessionId] = []

    async def fetch_rollout(self, session_id: SessionId, /) -> bytes | None:
        self.asked.append(session_id)
        return self._body


class UnreachablePod(Exception):
    """Stands in for whatever the pod wire raises when a read cannot be completed."""


class FailingFetch:
    """A `RolloutFetch` that cannot answer at all."""

    async def fetch_rollout(self, session_id: SessionId, /) -> bytes | None:
        raise UnreachablePod(str(session_id))


class BucketRefusedTheWrite(Exception):
    """Stands in for whatever the object-store client raises when a PUT is refused.

    Deliberately not `TurnUndeliverable` and not an `httpx` error: those are the two
    shapes the dispatch already knows how to translate, and a store failure is neither.
    botocore's own is `ClientError`, which shares no base with either.
    """


class RefusedByThePod:
    """A `RolloutFetch` that already speaks the port's own failure.

    What `PodRolloutFetch` raises when the pod answers a status the fetch will not read
    as a Rollout -- a refused token, a body past the cap.
    """

    async def fetch_rollout(self, session_id: SessionId, /) -> bytes | None:
        raise TurnUndeliverable(
            f"the shim for session {session_id} answered 403 for its rollout"
        )


class UnreachableAtTheFetch:
    """A `RolloutFetch` whose read of the pod fails at the transport.

    `httpx.ReadError` stands for the whole family: a reset connection, a pod that
    stopped answering inside the fetch's own deadline. `dispatch` already translates
    these accurately, so the completion seam must not relabel them.
    """

    async def fetch_rollout(self, session_id: SessionId, /) -> bytes | None:
        raise httpx.ReadError("the connection went away mid-read")


class StoreThatCannotWrite:
    """A `RolloutObjectStore` whose bucket refuses every write."""

    async def put(self, key: str, body: bytes) -> None:
        raise BucketRefusedTheWrite(key)

    async def get(self, key: str) -> bytes | None:
        return None


class CountingLog:
    """An `EventLogAppend` recording the types it was given, numbering from 1."""

    def __init__(self) -> None:
        self.types: list[str] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.types.append(type_)
        return Seq(len(self.types))


def _a_pod_streaming_one_completed_turn(turn_id: TurnId) -> httpx.MockTransport:
    """A pod that answers the Turn route with one completed Turn and stops.

    Hand-built rather than served by `create_shim_app` because what is under test is
    what the dispatch does *after* the stream ends, and driving the real shim would need
    a real Agent Runtime on the other end of a socket.
    """
    body = (
        json.dumps(
            {
                "kind": "event",
                "type": turn.TURN_COMPLETED,
                "payload": {"turn_id": str(turn_id)},
            }
        )
        + "\n"
        + json.dumps({"kind": "completed"})
        + "\n"
    ).encode()

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.MockTransport(answer)


# ------------------------------------------------------------------------------------
# The cut
# ------------------------------------------------------------------------------------


def test_the_lifecycle_names_are_exactly_the_ones_research_measured() -> None:
    """A widening here is a fail-open on the one guarantee this slice makes.

    `turn_started` / `turn_complete` / `turn_aborted` are cited with file and line at
    `research/plan-pass-rollout-and-resume.md` lines 6 and 98. Accepting a name that is
    not a Turn boundary would cut after an unfinished Turn and hand a partial tail back.
    """
    assert {"turn_started"} == TURN_OPENED
    assert {"turn_complete"} == TURN_CLOSED_COMPLETE
    assert {"turn_aborted"} == TURN_CLOSED_ABORTED


def test_completed_turns_survive_and_the_partial_tail_does_not() -> None:
    restored = truncate_at_last_completed_turn(
        _rollout(
            _meta(),
            _event("turn_started", 1),
            _said("first answer", 2),
            _event("turn_complete", 3),
            _event("turn_started", 4),
            _said("half an answer", 5),
        )
    )
    assert b"first answer" in restored.body
    assert b"half an answer" not in restored.body
    assert restored.completed_turns == 1
    assert restored.partial_turn_dropped is True
    assert restored.dropped_lines == 2


def test_the_completion_line_itself_is_kept_so_the_turn_is_not_rerun() -> None:
    restored = truncate_at_last_completed_turn(
        _rollout(_meta(), _event("turn_started", 1), _event("turn_complete", 2))
    )
    assert restored.body.splitlines()[-1] == _event("turn_complete", 2)
    assert restored.partial_turn_dropped is False


def test_an_aborted_turn_is_dropped_and_reads_as_a_turn_that_was_lost() -> None:
    """An abort is not a completion, so the cut lands before the Turn that aborted."""
    restored = truncate_at_last_completed_turn(
        _rollout(
            _meta(),
            _event("turn_complete", 1),
            _event("turn_started", 2),
            _event("turn_aborted", 3),
        )
    )
    assert restored.completed_turns == 1
    assert restored.partial_turn_dropped is True
    assert restored.dropped_lines == 2


def test_an_abort_with_no_opener_after_the_cut_reads_as_a_turn_that_was_lost() -> None:
    """`TURN_CLOSED_ABORTED`'s one behavioural job, with nothing else able to do it.

    The case above this builds `turn_started, turn_aborted`, so the opener alone carries
    the `any()` and dropping the abort from `_TURN_OPENERS` leaves it green. Here the
    dropped region holds the abort and no opener at all -- the shape a record has when
    the cut lands between a Turn's start and its abort -- so the abort is the only line
    that can report the Turn as lost.
    """
    restored = truncate_at_last_completed_turn(
        _rollout(_meta(), _event("turn_complete", 1), _event("turn_aborted", 2))
    )
    assert restored.partial_turn_dropped is True
    assert restored.dropped_lines == 1


def test_trailing_lines_that_are_not_a_turn_do_not_read_as_a_lost_turn() -> None:
    restored = truncate_at_last_completed_turn(
        _rollout(_meta(), _event("turn_complete", 1), _event("token_count", 2))
    )
    assert restored.partial_turn_dropped is False
    assert restored.dropped_lines == 1


def test_a_torn_last_line_is_dropped_rather_than_fatal() -> None:
    """Ship-out reads the file while the runtime may still be appending to it."""
    torn = _rollout(_meta(), _event("turn_complete", 1)) + b'{"timestamp":"2026-'
    restored = truncate_at_last_completed_turn(torn)
    assert restored.completed_turns == 1
    assert restored.dropped_lines == 1


def test_a_tail_torn_inside_a_character_is_dropped_and_not_raised_at_the_caller() -> (
    None
):
    """The tear that a real Rollout gets, and the one that used to be permanent.

    `json.loads` on bytes decodes before it parses, so this tail raises
    `UnicodeDecodeError` and not `JSONDecodeError` -- a distinction no other case in
    this file can see, because every tail they build is pure ASCII. The stored object is
    only replaced by the next completed Turn, and that Turn needs a resume of its own,
    so a restore that raised here left the Session unable to start ever again.
    """
    torn = _rollout(_meta(), _event("turn_complete", 1)) + _cut_mid_character(
        "response_item", 2
    )

    restored = truncate_at_last_completed_turn(torn)

    assert restored.completed_turns == 1
    assert restored.dropped_lines == 1
    assert restored.body.splitlines()[-1] == _event("turn_complete", 1)


def test_a_header_torn_inside_a_character_is_refused_as_a_malformed_rollout() -> None:
    """The one line whose tearing is fatal, and the exception has to be this platform's.

    A caller cannot tell a `UnicodeDecodeError` escaping from here apart from a bug in
    this module; `MalformedRollout` says the bytes were not a Rollout, which is the true
    reading and the one the recovery path is written to handle.
    """
    with pytest.raises(MalformedRollout):
        truncate_at_last_completed_turn(_cut_mid_character("session_meta", 0) + b"\n")


def test_a_session_with_no_completed_turn_keeps_only_its_header() -> None:
    header = _meta()
    restored = truncate_at_last_completed_turn(
        _rollout(header, _event("turn_started", 1), _said("interrupted", 2))
    )
    assert restored.body == header + b"\n"
    assert restored.completed_turns == 0
    assert restored.partial_turn_dropped is True


def test_bytes_that_do_not_open_with_the_header_are_refused() -> None:
    with pytest.raises(MalformedRollout):
        truncate_at_last_completed_turn(_rollout(_event("turn_complete", 1)))


def test_no_lines_at_all_is_refused_rather_than_restored_as_empty() -> None:
    with pytest.raises(MalformedRollout):
        truncate_at_last_completed_turn(b"\n  \n")


# ------------------------------------------------------------------------------------
# Ship-out and restore
# ------------------------------------------------------------------------------------


def test_the_key_gives_every_session_its_own_object() -> None:
    """The literal shape, asserted literally, because this f-string is the separation.

    Every other case in this file reaches the store *through* `rollout_key`, so all of
    them stay self-consistent under a key function that separates nothing -- including
    the two that assert on `store.puts` and `store.objects`, and including
    `len(store.objects) == 1`, which holds while one Session is in play. A key that
    dropped the Session would make tenant A's completed Turn overwrite tenant B's
    Rollout, and B's next resume would restore A's entire agent conversation.
    """
    one = SessionId(uuid4())
    another = SessionId(uuid4())

    assert rollout_key(one) == f"rollouts/{one}/rollout.jsonl"
    assert rollout_key(one) != rollout_key(another)


def test_a_rollout_key_cannot_address_an_uploaded_file() -> None:
    """One bucket may hold both, and only the prefixes keep them apart.

    `MAP_ROLLOUT_BUCKET` and the upload bucket's variable are separate names, so
    nothing stops an operator pointing them at one bucket -- and one bucket carrying
    prefixes is what the provisioned environment has. A Rollout written under the
    upload prefix would sit in the key space of a tenant-addressable file surface.
    """
    session_id = SessionId(uuid4())

    assert rollout_key(session_id).startswith("rollouts/")
    assert not rollout_key(session_id).startswith(f"{UPLOAD_KEY_ROOT}/")


async def test_a_resume_gets_back_what_the_last_completed_turn_shipped() -> None:
    store = InMemoryRolloutStore()
    sync = RolloutSync(store)
    session_id = SessionId(uuid4())

    await sync.ship_out(session_id, _rollout(_meta(), _event("turn_complete", 1)))
    await sync.ship_out(
        session_id,
        _rollout(_meta(), _event("turn_complete", 1), _event("turn_complete", 2)),
    )

    restored = await sync.restore_for_resume(session_id)
    assert restored is not None
    assert restored.completed_turns == 2
    assert store.puts == [rollout_key(session_id), rollout_key(session_id)]
    assert len(store.objects) == 1


async def test_a_session_that_never_completed_a_turn_restores_nothing() -> None:
    sync = RolloutSync(InMemoryRolloutStore())
    assert await sync.restore_for_resume(SessionId(uuid4())) is None


async def test_a_restore_leaves_the_stored_rollout_where_it_was() -> None:
    """MAP-A8: the record stays readable. Nothing on either path removes an object."""
    store = InMemoryRolloutStore()
    sync = RolloutSync(store)
    session_id = SessionId(uuid4())
    body = _rollout(_meta(), _event("turn_complete", 1))

    await sync.ship_out(session_id, body)
    await sync.restore_for_resume(session_id)

    assert store.objects[rollout_key(session_id)] == body


async def test_a_completed_turn_ships_what_the_pod_was_holding() -> None:
    store = InMemoryRolloutStore()
    session_id = SessionId(uuid4())
    body = _rollout(_meta(), _event("turn_complete", 1))
    fetch = FixedFetch(body)
    completion = ShipOutAtTurnCompletion(fetch, RolloutSync(store))

    await completion.turn_completed(session_id, new_turn_id())

    assert store.objects == {rollout_key(session_id): body}
    assert fetch.asked == [session_id], "the pod is asked for its own Session's bytes"


async def test_a_pod_holding_no_rollout_does_not_overwrite_a_good_one() -> None:
    """An empty object here would destroy the only thing recovery reads."""
    store = InMemoryRolloutStore()
    session_id = SessionId(uuid4())
    good = _rollout(_meta(), _event("turn_complete", 1))
    await RolloutSync(store).ship_out(session_id, good)

    completion = ShipOutAtTurnCompletion(FixedFetch(None), RolloutSync(store))
    await completion.turn_completed(session_id, new_turn_id())

    assert store.objects[rollout_key(session_id)] == good
    assert store.puts == [rollout_key(session_id)]


async def test_a_pod_answering_zero_bytes_does_not_overwrite_a_good_one_either() -> (
    None
):
    """The case above answers None. This one answers `b""`, and the two mean the same
    thing about the pod while being different values in Python -- which is the whole
    defect. `PodRolloutFetch` returns the body the route gave it, and a route that
    answers 200 over a file the runtime created and has not flushed gives it zero bytes.

    A guard written as `body is None` admits those bytes, the put replaces a good stored
    Rollout with nothing, and every later resume of the Session reads a Rollout with no
    lines and refuses. The stored object is only replaced by the next completed Turn,
    and that Turn needs a resume, so the Session never comes back.
    """
    store = InMemoryRolloutStore()
    session_id = SessionId(uuid4())
    good = _rollout(_meta(), _event("turn_complete", 1))
    await RolloutSync(store).ship_out(session_id, good)

    completion = ShipOutAtTurnCompletion(FixedFetch(b""), RolloutSync(store))
    await completion.turn_completed(session_id, new_turn_id())

    assert store.objects[rollout_key(session_id)] == good
    assert store.puts == [rollout_key(session_id)], "the empty body reached the bucket"


async def test_a_stored_rollout_of_zero_bytes_restores_none_rather_than_raising() -> (
    None
):
    """What a Session wedged by the defect above needs in order to come back.

    Written into the store directly, because ship-out now refuses to put these bytes --
    so the only way this object exists is a process that shipped before that guard, and
    those objects are in the bucket already. A restore that raised `MalformedRollout`
    here would leave the Session unable to start ever again, since only a completed Turn
    replaces the object and a completed Turn needs a resume. None is the honest reading:
    zero bytes are not a Rollout, and no Turn of this Session is recoverable from them.
    """
    store = InMemoryRolloutStore()
    session_id = SessionId(uuid4())
    store.objects[rollout_key(session_id)] = b""

    assert await RolloutSync(store).restore_for_resume(session_id) is None


async def test_a_pod_lost_after_a_turn_completed_leaves_a_record_that_reads_clean() -> (
    None
):
    """The dominant recovery path, and what the cut can and cannot say about it.

    Ship-out fires at Turn 1's completion while the runtime is idle, so the stored
    record ends on that completion line. Turn 2 starts and the pod dies: no Turn
    completed, so nothing ships -- `store.puts` holding one key is that fact, not a
    narration of it. Turn 2 is therefore absent from the bytes a resume reads, and the
    result reads as a clean record: `partial_turn_dropped` False and `dropped_lines`
    zero on precisely the path a discard marker is owed for.

    Frozen here so nobody can make `partial_turn_dropped` claim otherwise. It cannot be
    computed from these bytes -- the evidence is not in them -- so a version of it that
    answered True here would be guessing, and a caller wiring a discard marker off it
    appends nothing. `completed_turns` reconciled against the caller's own count of
    completed Turns in the Event Log is what names the lost Turn.
    """
    store = InMemoryRolloutStore()
    sync = RolloutSync(store)
    session_id = SessionId(uuid4())
    at_turn_one = _rollout(
        _meta(),
        _event("turn_started", 1),
        _said("first answer", 2),
        _event("turn_complete", 3),
    )
    completion = ShipOutAtTurnCompletion(FixedFetch(at_turn_one), sync)

    await completion.turn_completed(session_id, new_turn_id())

    restored = await sync.restore_for_resume(session_id)
    assert restored is not None
    assert store.puts == [rollout_key(session_id)]
    assert restored.completed_turns == 1
    assert restored.partial_turn_dropped is False
    assert restored.dropped_lines == 0


async def test_a_fetch_that_cannot_answer_is_not_swallowed() -> None:
    """A Turn that reads as durable while its bytes are still only inside a pod about
    to be allowed to die is the one outcome worse than a loud failure."""
    store = InMemoryRolloutStore()
    completion = ShipOutAtTurnCompletion(FailingFetch(), RolloutSync(store))

    with pytest.raises(UnreachablePod):
        await completion.turn_completed(SessionId(uuid4()), new_turn_id())

    assert store.puts == []


def test_what_the_cut_reports_composes_into_the_marker_a_resume_owes() -> None:
    """The two halves type-check together and the payload survives the round trip.

    This slice appends no marker -- the sequence the interrupted Turn began at is read
    on the resume path, which is another slice's -- so what is guarded here is that the
    fact this slice produces is the fact `markers.discard` takes. The defect class it
    exists for is a plan that composed two handlers whose signatures could not compose,
    where the missing argument was the boundary.

    **`completed_turns` is the fact the marker's condition is computed from, and
    `partial_turn_dropped` is free text.** Both are asserted here for that reason and in
    that order. The flag reads False on the dominant pod-lost path -- the case below
    this one freezes it -- so a resuming slice that made it the condition would append
    nothing exactly when a marker is owed; the count reconciled against its own log is
    what names the loss. These bytes happen to carry both, which is what lets one case
    pin both shapes without asserting that either one alone decides anything.
    """
    restored = truncate_at_last_completed_turn(
        _rollout(
            _meta(),
            _event("turn_complete", 1),
            _event("turn_started", 2),
            _said("half an answer", 3),
        )
    )
    assert restored.completed_turns == 1
    assert restored.partial_turn_dropped is True

    type_, payload = discard(
        DiscardCause.POD_LOST,
        f"the pod ended before this Turn was made durable; the resumed record holds "
        f"{restored.completed_turns} completed Turns and dropped "
        f"{restored.dropped_lines} lines after the last one",
        FIRST_SEQ,
    )

    assert type_ == WORK_DISCARDED
    parsed = WorkDiscarded.model_validate(payload)
    assert parsed.cause is DiscardCause.POD_LOST
    assert str(restored.completed_turns) in parsed.detail
    assert str(restored.dropped_lines) in parsed.detail
    assert parsed.discarded_from == FIRST_SEQ


# ------------------------------------------------------------------------------------
# The pull: the control plane reads the bytes out of the pod
# ------------------------------------------------------------------------------------


class NeverPlaces:
    """The `SessionPods` seam, refusing every call.

    A Turn places a Session's pod only when `locate` answers ABSENT, and no case in this
    file reports that phase. Refusing rather than recording makes that a property this
    file asserts instead of one it merely happens to have: a case that started reaching
    placement fails here, naming it, rather than quietly exercising a collaborator
    nothing in this file was written to grade.
    """

    async def ensure_for(self, session_id: SessionId) -> None:
        raise AssertionError("a test in this file placed a Session's pod")


class FixedPhase:
    """A cluster reporting one phase for every pod, and starting nothing."""

    def __init__(self, phase: PodPhase = PodPhase.RUNNING) -> None:
        self._phase = phase
        self.asked: list[str] = []

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        raise AssertionError("reading a rollout tried to start a pod")

    async def phase_of(self, pod_name: str) -> PodPhase:
        self.asked.append(pod_name)
        return self._phase

    async def remove(self, pod_name: str) -> None:
        raise AssertionError("reading a rollout tried to remove a pod")


class NeverDialled(httpx.AsyncBaseTransport):
    """A transport that fails the test if anything is sent over it."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the pod was dialled at {request.url}")


def _a_pod_serving(
    monkeypatch: pytest.MonkeyPatch, home: Path, session_id: SessionId, token_key: bytes
) -> httpx.AsyncBaseTransport:
    """The real shim app for one Session, reachable over an ASGI transport.

    `RUNTIME_HOME` is redirected on the module the route reads it from, because the pod
    path is a deployment constant everywhere except under a test. The connection is
    constructed and never dialled: the Rollout route touches the runtime not at all.
    """
    monkeypatch.setattr("managed_agent.session_shim.serve.RUNTIME_HOME", home)
    app = create_shim_app(
        ServedSession(
            session_id=session_id,
            thread_id=_THREAD,
            connection=RuntimeConnection(home / "never-dialled.sock"),
            token=shim_token_for(session_id, token_key),
        )
    )
    return httpx.ASGITransport(app=app)


def _wrote_a_rollout(home: Path, body: bytes) -> None:
    path = home / "sessions/2026/08/22" / f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


async def test_the_pull_returns_exactly_the_bytes_the_pods_file_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves at once: the real route serves the file, the real fetch reads it.

    What this does not show is that a Kubernetes pod ever answers on this address --
    nothing in this tree places one.
    """
    session_id = SessionId(uuid4())
    body = _rollout(_meta(), _event("turn_complete", 1), _event("turn_started", 2))
    _wrote_a_rollout(tmp_path, body)

    fetch = PodRolloutFetch(
        Placement(FixedPhase()),
        _NAMESPACE,
        _KEY,
        _a_pod_serving(monkeypatch, tmp_path, session_id, _KEY),
    )
    assert await fetch.fetch_rollout(session_id) == body


async def test_a_completed_turn_ships_the_pods_bytes_cut_at_the_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The slice's checkpoint, end to end over the wire it will really use.

    The stored object holds the tail the pod was still writing -- ship-out preserves
    what the pod had -- and the restore is where the cut happens, so a resume gets bytes
    ending on a completion line.
    """
    session_id = SessionId(uuid4())
    live = _rollout(
        _meta(),
        _event("turn_started", 1),
        _said("first answer", 2),
        _event("turn_complete", 3),
        _event("turn_started", 4),
        _said("half an answer", 5),
    )
    _wrote_a_rollout(tmp_path, live)
    store = InMemoryRolloutStore()
    sync = RolloutSync(store)
    completion = ShipOutAtTurnCompletion(
        PodRolloutFetch(
            Placement(FixedPhase()),
            _NAMESPACE,
            _KEY,
            _a_pod_serving(monkeypatch, tmp_path, session_id, _KEY),
        ),
        sync,
    )

    await completion.turn_completed(session_id, new_turn_id())

    assert store.objects[rollout_key(session_id)] == live
    restored = await sync.restore_for_resume(session_id)
    assert restored is not None
    assert b"half an answer" not in restored.body
    assert restored.body.splitlines()[-1] == _event("turn_complete", 3)
    assert restored.completed_turns == 1
    assert restored.partial_turn_dropped is True


@pytest.mark.parametrize("phase", [PodPhase.ABSENT, PodPhase.STARTING, PodPhase.GONE])
async def test_a_pod_that_is_not_running_is_not_dialled_at_all(
    phase: PodPhase,
) -> None:
    """A Turn cannot have completed on a pod that is gone, so there are no bytes to
    ship and the honest answer is None rather than a failure for a Turn that finished.

    The cluster is still asked, which is what makes the absent dial a decision rather
    than a fetch that did nothing.

    Every phase that is not RUNNING, and not GONE alone: `phase_of` answers ABSENT for
    a pod that was never created, which is what a deleted pod reports and therefore the
    dominant state after an eviction. A guard narrowed to GONE would dial a pod that is
    not there for exactly that case.
    """
    cluster = FixedPhase(phase)
    fetch = PodRolloutFetch(Placement(cluster), _NAMESPACE, _KEY, NeverDialled())

    assert await fetch.fetch_rollout(SessionId(uuid4())) is None
    assert len(cluster.asked) == 1


async def test_a_pod_holding_no_file_yet_answers_none_and_not_empty_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 204 the route answers has to arrive here as None. Read as zero bytes it
    would overwrite a good stored Rollout with nothing."""
    session_id = SessionId(uuid4())
    fetch = PodRolloutFetch(
        Placement(FixedPhase()),
        _NAMESPACE,
        _KEY,
        _a_pod_serving(monkeypatch, tmp_path, session_id, _KEY),
    )
    assert await fetch.fetch_rollout(session_id) is None


async def test_a_zero_byte_file_leaves_the_stored_rollout_where_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path the live defect ran down, over the real route and the real fetch.

    The file exists and holds nothing, which is the pod between the runtime creating its
    record and flushing the first line into it. Everything here is the shipped code
    except the bytes on disk: the route decides the status, the fetch reads it, and
    ship-out decides whether to put. Two guards have to hold for the stored object to
    survive -- the route's, and the control plane's own -- because the route stats the
    file and `FileResponse` stats it again, so the file can empty between them.
    """
    session_id = SessionId(uuid4())
    good = _rollout(_meta(), _event("turn_complete", 1))
    store = InMemoryRolloutStore()
    sync = RolloutSync(store)
    await sync.ship_out(session_id, good)
    _wrote_a_rollout(tmp_path, b"")

    completion = ShipOutAtTurnCompletion(
        PodRolloutFetch(
            Placement(FixedPhase()),
            _NAMESPACE,
            _KEY,
            _a_pod_serving(monkeypatch, tmp_path, session_id, _KEY),
        ),
        sync,
    )
    await completion.turn_completed(session_id, new_turn_id())

    assert store.objects[rollout_key(session_id)] == good
    assert store.puts == [rollout_key(session_id)]
    restored = await sync.restore_for_resume(session_id)
    assert restored is not None, "the good record survived and still restores"
    assert restored.completed_turns == 1


async def test_a_rollout_past_the_cap_is_refused_rather_than_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap is lowered rather than the file grown: the shipped bound is 256 MiB and
    a test that wrote that much would be measuring the disk. What is exercised is the
    real accumulation loop and the real refusal.
    """
    session_id = SessionId(uuid4())
    body = _rollout(_meta(), _event("turn_complete", 1))
    _wrote_a_rollout(tmp_path, body)
    monkeypatch.setattr(
        "managed_agent.session_shim.pod_channel.ROLLOUT_FETCH_LIMIT_BYTES",
        len(body) - 1,
    )

    fetch = PodRolloutFetch(
        Placement(FixedPhase()),
        _NAMESPACE,
        _KEY,
        _a_pod_serving(monkeypatch, tmp_path, session_id, _KEY),
    )
    with pytest.raises(TurnUndeliverable, match="larger than"):
        await fetch.fetch_rollout(session_id)


async def test_a_token_derived_for_another_session_is_refused_by_the_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pod's own check is what stops this, and the refusal has to reach the caller.

    Driven by giving the fetch a different signing key, so the token it derives is not
    the one this pod was mounted with -- which is the shape a control plane holding the
    wrong key has. A refusal read as "no rollout yet" would silently stop shipping
    every Session's bytes.
    """
    session_id = SessionId(uuid4())
    body = _rollout(_meta(), _event("turn_complete", 1))
    _wrote_a_rollout(tmp_path, body)

    fetch = PodRolloutFetch(
        Placement(FixedPhase()),
        _NAMESPACE,
        b"a key this pod was never mounted with",
        _a_pod_serving(monkeypatch, tmp_path, session_id, _KEY),
    )
    with pytest.raises(TurnUndeliverable, match="404"):
        await fetch.fetch_rollout(session_id)


# ------------------------------------------------------------------------------------
# Placing a pod that continues a thread
# ------------------------------------------------------------------------------------


class RecordingCluster:
    """A cluster that can place a pod and remembers every pod it was asked for."""

    def __init__(self) -> None:
        self.ensured: list[str] = []

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        self.ensured.append(pod_name)
        return PodPhase.RUNNING

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.RUNNING if pod_name in self.ensured else PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        return None


@runtime_checkable
class PodPlacement(Protocol):
    """The placement surface the components around `Placement` depend on.

    Declared locally rather than imported: what is pinned here is that the concrete
    `Placement` answers to it, and that a class missing a method does not.

    It carried a fourth method, `place_resuming`, for as long as a resume was a
    refusal. It does not now, and the removal is the point rather than an omission --
    there is ONE way a Session's pod is placed, and a resuming Session goes through it
    with one compiled value different. A second entry point would be a second place the
    compile inputs are assembled, and the two would be free to assemble them differently
    for the same Session.
    """

    async def place(self, compiled: CompiledConfig) -> object: ...

    async def locate(self, session_id: SessionId) -> object: ...

    async def release(self, session_id: SessionId) -> None: ...


class PlacementMissingRelease:
    """A placement short one method, so the conformance check below can fail."""

    async def place(self, compiled: CompiledConfig) -> object:
        return None

    async def locate(self, session_id: SessionId) -> object:
        return None


async def test_a_resume_places_through_the_one_path_every_placement_goes_through(
    tmp_path: Path,
) -> None:
    """There is no second entry point, and asking for one is a type error.

    `Placement` grew a `place_resuming` while a resume was a refusal, and the refusal
    was the whole of its body. Now that a resuming Session is compiled and placed like
    any other, a surviving second method would be a second way to assemble the compile
    inputs for one Session -- and the two would be free to disagree about the
    Environment revision, the definition or the skills that Session pinned.

    Asserted as an absence because that is the property: nothing may call it, and no
    caller may be written against it. The cluster below *can* place a pod, so this is
    not passing because placement is broken.
    """
    cluster = RecordingCluster()
    placement = Placement(cluster)

    assert not hasattr(placement, "place_resuming")
    assert (await placement.locate(SessionId(uuid4()))).phase is PodPhase.ABSENT
    assert cluster.ensured == []


def test_the_concrete_placement_answers_to_the_surface_its_callers_declare() -> None:
    """`runtime_checkable` plus `isinstance` checks method **names** only, never
    signatures, so this proves the names exist and proves nothing about their arguments.
    The negative half is what keeps this from passing for a Protocol nothing could
    fail."""
    assert isinstance(Placement(RecordingCluster()), PodPlacement)
    assert not isinstance(PlacementMissingRelease(), PodPlacement)


# ------------------------------------------------------------------------------------
# What a failed ship-out does to the Turn that triggered it
# ------------------------------------------------------------------------------------


async def test_a_bucket_that_refuses_the_write_fails_the_turn_as_undeliverable() -> (
    None
):
    """A ship-out failure has to leave the dispatch as the one exception its port names.

    `TurnDispatch` says a caller of the port never sees a transport error, and
    `control/api/routes/turns.py` catches exactly `TurnUndeliverable` -- appending
    `turn.failed` and answering with the published `turn.undeliverable` code. A store
    error escaping raw is caught by nothing: the tenant gets a bare 500 carrying no code
    from the closed set, and the Event Log gets no record that the Turn produced an
    answer nothing can resume from.

    The `turn.completed` append is asserted to have happened first, because that is the
    divergence ADR-004 puts in writing: the Event Log may be one Turn ahead of the
    resume state, and failing loudly here is the deliberate alternative to a Turn that
    reads as durable while its bytes are only inside a pod about to be allowed to die.
    """
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    log = CountingLog()
    dispatch = HttpPodDispatch(
        placement=Placement(FixedPhase()),
        pods=NeverPlaces(),
        log=log,
        on_completed=ShipOutAtTurnCompletion(
            FixedFetch(_rollout(_meta(), _event("turn_complete", 1))),
            RolloutSync(StoreThatCannotWrite()),
        ),
        namespace=_NAMESPACE,
        token_key=_KEY,
        transport=_a_pod_streaming_one_completed_turn(turn_id),
    )

    with pytest.raises(TurnUndeliverable, match=str(session_id)):
        await dispatch.dispatch(session_id, turn_id, "a prompt")

    assert log.types == [turn.TURN_COMPLETED], (
        "the tenant-visible append happened first"
    )


async def test_a_pod_refusal_at_the_fetch_reaches_the_caller_as_itself() -> None:
    """The completion seam has two failures that say different things, and the
    translation above must not flatten one into the other.

    A fetch refusal is already `TurnUndeliverable` carrying what the pod answered. Left
    to the broad handler it would come out saying the rollout "could not be stored",
    which tells an operator the object store failed when it was never reached -- and the
    store is the one thing that is fine in this case.
    """
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    dispatch = HttpPodDispatch(
        placement=Placement(FixedPhase()),
        pods=NeverPlaces(),
        log=CountingLog(),
        on_completed=ShipOutAtTurnCompletion(
            RefusedByThePod(), RolloutSync(StoreThatCannotWrite())
        ),
        namespace=_NAMESPACE,
        token_key=_KEY,
        transport=_a_pod_streaming_one_completed_turn(turn_id),
    )

    with pytest.raises(TurnUndeliverable, match="answered 403 for its rollout"):
        await dispatch.dispatch(session_id, turn_id, "a prompt")


async def test_a_transport_failure_at_the_fetch_is_not_reported_as_a_store_one() -> (
    None
):
    """The same rule as the case above, for the other shape `dispatch` already handles.

    An `httpx` error from the fetch is what a pod that stopped answering looks like, and
    `dispatch`'s own handler names it correctly. Swallowing it into the store's message
    would send an operator to the bucket for a pod problem.
    """
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    dispatch = HttpPodDispatch(
        placement=Placement(FixedPhase()),
        pods=NeverPlaces(),
        log=CountingLog(),
        on_completed=ShipOutAtTurnCompletion(
            UnreachableAtTheFetch(), RolloutSync(StoreThatCannotWrite())
        ),
        namespace=_NAMESPACE,
        token_key=_KEY,
        transport=_a_pod_streaming_one_completed_turn(turn_id),
    )

    with pytest.raises(TurnUndeliverable, match="could not be reached"):
        await dispatch.dispatch(session_id, turn_id, "a prompt")


async def test_a_completed_turn_whose_bytes_do_reach_the_bucket_does_not_raise() -> (
    None
):
    """The positive half, so the case above is not passing on a dispatch that raises
    for every completed Turn. Same transport, same log, a store that accepts."""
    session_id = SessionId(uuid4())
    turn_id = new_turn_id()
    store = InMemoryRolloutStore()
    body = _rollout(_meta(), _event("turn_complete", 1))
    dispatch = HttpPodDispatch(
        placement=Placement(FixedPhase()),
        pods=NeverPlaces(),
        log=CountingLog(),
        on_completed=ShipOutAtTurnCompletion(FixedFetch(body), RolloutSync(store)),
        namespace=_NAMESPACE,
        token_key=_KEY,
        transport=_a_pod_streaming_one_completed_turn(turn_id),
    )

    await dispatch.dispatch(session_id, turn_id, "a prompt")

    assert store.objects[rollout_key(session_id)] == body


# ------------------------------------------------------------------------------------
# The wiring
# ------------------------------------------------------------------------------------

# A URL nothing connects to. `create_async_engine` resolves the driver and builds a pool
# without dialling, and these cases are about which completion seam `build` chose.
_UNDIALLED = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/unused"


def _the_placers_four_other_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set what `build` reads once it is handed a pod runner, so these cases reach it.

    A process with a runner compiles Session configurations, and the four values that
    takes -- both gateway addresses, the Session-token signing key and a token lifetime
    -- have no defaults on purpose. Set here as stand-ins, because nothing in this file
    is about their values; what is graded is the wiring on the other side of them.

    Not shared with the other files that need the same four lines. Two identical
    fixtures are a coincidence, and a shared module for them would couple files that
    are graded independently -- the same argument this repository already made about
    the `AbsentPod` doubles.
    """
    monkeypatch.setenv("MAP_SESSION_TOKEN_KEY", "a session-token signing key")
    monkeypatch.setenv("MAP_SESSION_TOKEN_LIFETIME_S", "3600")
    monkeypatch.setenv("MAP_TOOL_GATEWAY_URL", "http://tool-gateway.map-test/mcp")
    monkeypatch.setenv("MAP_MODEL_GATEWAY_URL", "http://model-gateway.map-test/v1")


def _seam_of(dispatch: TurnDispatch) -> object:
    """What `build` put on the dispatch's Turn-completion seam.

    Read off the private attribute deliberately. The seam is a constructor argument and
    not a field of `Platform` -- `Platform` is frozen with no defaults, so a new field
    would be an edit to every site that builds one -- so the wiring is only observable
    here, and a wiring nothing observes is a wiring that can silently regress.
    """
    assert isinstance(dispatch, HttpPodDispatch)
    return dispatch._on_completed


async def test_without_a_bucket_a_completed_turn_still_ships_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bucket name guessed from a default would write every tenant's resume state
    somewhere nobody chose, so an unconfigured process keeps the seam that visibly
    drops a completion rather than one that looks configured."""
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    _the_placers_four_other_variables(monkeypatch)
    monkeypatch.delenv("MAP_ROLLOUT_BUCKET", raising=False)

    platform, engine = build(_UNDIALLED, pod_runner=RecordingCluster())
    try:
        assert isinstance(_seam_of(platform.turn_dispatch), _RolloutNotYetShipped)
    finally:
        await engine.dispose()


async def test_a_bucket_puts_the_real_ship_out_on_the_completion_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    _the_placers_four_other_variables(monkeypatch)
    monkeypatch.delenv("MAP_ROLLOUT_BUCKET", raising=False)

    platform, engine = build(
        _UNDIALLED, pod_runner=RecordingCluster(), rollout_bucket="a-bucket"
    )
    try:
        assert isinstance(_seam_of(platform.turn_dispatch), ShipOutAtTurnCompletion)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
async def test_a_blank_bucket_name_is_no_bucket_and_not_a_bucket_called_nothing(
    blank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty or whitespace variable is an operator who meant to set it and did not.

    `is not None` admits both, and what gets built is a real `S3RolloutStore` addressing
    a bucket whose name is blank -- so every ship-out fails at the AWS call instead of
    at start-up, and by then the Turn is already appended. The seam that visibly ships
    nothing is the honest answer to a name nobody supplied, and it is the same answer
    the unset case already gets.
    """
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    _the_placers_four_other_variables(monkeypatch)
    monkeypatch.setenv(_BUCKET_ENV, blank)

    platform, engine = build(_UNDIALLED, pod_runner=RecordingCluster())
    try:
        assert isinstance(_seam_of(platform.turn_dispatch), _RolloutNotYetShipped)
    finally:
        await engine.dispose()


async def test_the_bucket_is_read_from_the_environment_when_the_argument_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`os.environ.get` and not `os.environ[...]`: the case above sets no bucket and
    must keep building, so an absent name is a seam that ships nothing rather than a
    control plane that will not start."""
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    _the_placers_four_other_variables(monkeypatch)
    monkeypatch.setenv(_BUCKET_ENV, "a-bucket-from-the-environment")

    platform, engine = build(_UNDIALLED, pod_runner=RecordingCluster())
    try:
        assert isinstance(_seam_of(platform.turn_dispatch), ShipOutAtTurnCompletion)
    finally:
        await engine.dispose()


def test_the_bucket_variable_is_namespaced_to_this_platform() -> None:
    """A generic name is one a base image or a sidecar could set for its own reasons,
    which this platform would then silently adopt as the place tenants' resume state
    goes."""
    assert _BUCKET_ENV.startswith("MAP_")
