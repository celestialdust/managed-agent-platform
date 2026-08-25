"""Provenance is folded out of the log, and the fold agrees with what writes it.

The load-bearing case here is the round trip. `written_payload` builds the event body
and `provenance` reads it, and they are one module precisely so they cannot drift; a
test that hand-wrote the payload it expected to read would grade the reader against this
file's idea of the shape rather than against the writer's, and would keep passing
through exactly the rename that breaks production. So every case that needs a payload
builds it with the shipped builder, and the round trip is parametrized over `LANES` so a
lane whose writes cannot be read back fails by name.

What is deliberately *not* here is a file. There is no `manifest.json` to assert on,
because the facts one would carry are these events -- so the assertions are about what
the log says, and there is no second artifact that could disagree with it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

import pytest

from managed_agent.core.ids import Seq, SessionId
from managed_agent.core.vfs.evidence import digest_of
from managed_agent.core.vfs.session_vfs import (
    Lane,
    MutableLane,
    SealedLane,
    SourceRef,
)
from managed_agent.core.vfs.vfs_provenance import provenance, written_payload
from managed_agent.core.vocabulary.vfs import OBJECT_PLACED, OBJECT_REPLACED

ARTIFACTS = SealedLane("kept")
WORKING = MutableLane("scratchpad")
LANES: tuple[Lane, ...] = (ARTIFACTS, WORKING)
"""Example lanes, declared here because the platform declares none.

One sealed and one mutable, which is the whole distinction a provenance fold can see:
a sealed lane's object appears once, and a mutable lane's object appears again. The
local names are kept short for the cases below and are not platform lanes."""

A_SESSION = SessionId(uuid4())


@dataclass(frozen=True, slots=True)
class Event:
    """An `EventRecord` that is frozen, which the port admits on purpose.

    The port's members are read-only properties rather than plain annotations so an
    immutable implementation satisfies it; this is that implementation, and its being
    accepted by `mypy --strict` is part of what the port is asserting.
    """

    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


def placed(
    seq: int,
    lane: Lane,
    relative: str,
    body: bytes,
    sources: tuple[SourceRef, ...] = (),
) -> Event:
    return Event(
        A_SESSION,
        Seq(seq),
        OBJECT_PLACED,
        written_payload(lane, relative, digest_of(body), sources),
    )


def rewritten(seq: int, lane: Lane, relative: str, body: bytes) -> Event:
    return Event(
        A_SESSION,
        Seq(seq),
        OBJECT_REPLACED,
        written_payload(lane, relative, digest_of(body)),
    )


def test_there_are_lanes_to_grade() -> None:
    """Guard the guard: the round trip below is parametrized over `LANES`."""
    assert len(LANES) >= 2
    assert any(isinstance(lane, MutableLane) for lane in LANES)
    assert any(isinstance(lane, SealedLane) for lane in LANES)


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
def test_a_write_in_any_lane_reads_back_with_the_digest_it_was_written_with(
    lane: Lane,
) -> None:
    body = b"the produced bytes"
    folded = provenance([placed(1, lane, "report.md", body)])

    record = folded[(lane.directory, "report.md")]
    assert record.digest == digest_of(body)
    assert record.digest.byte_length == len(body)
    assert record.placed_at_seq == 1
    assert record.revisions == 0
    assert record.sources == ()


def test_the_sources_a_write_declared_survive_the_round_trip() -> None:
    """The manifest's `sources[].sha256`, as a projection rather than a file."""
    source = SourceRef(relative="raw.json", digest=digest_of(b"the raw evidence"))
    folded = provenance(
        [placed(1, ARTIFACTS, "report.md", b"the report", sources=(source,))]
    )

    record = folded[(ARTIFACTS.directory, "report.md")]
    assert record.sources == (source,)
    assert record.sources[0].digest == digest_of(b"the raw evidence")


def test_a_source_digest_that_no_longer_matches_its_file_is_detectable() -> None:
    """This is what recording the source's digest at the write is *for*.

    The artifact says what its input hashed to when it was made. The input is then
    rewritten. Nothing in the record changes -- and comparing the two is what tells a
    reader the derivation no longer describes the bytes it was made from. A mutable
    manifest that was updated in place at the rewrite would have destroyed exactly this.
    """
    was = digest_of(b"the raw evidence")
    folded = provenance(
        [
            placed(
                1,
                ARTIFACTS,
                "report.md",
                b"the report",
                sources=(SourceRef(relative="draft.json", digest=was),),
            ),
            rewritten(2, WORKING, "draft.json", b"the raw evidence, revised"),
        ]
    )

    recorded = folded[(ARTIFACTS.directory, "report.md")].sources[0].digest
    now = folded[(WORKING.directory, "draft.json")].digest
    assert recorded == was
    assert recorded != now, "a rewritten source must be distinguishable from its record"


def test_a_rewrite_bumps_the_revision_count_and_keeps_the_placement() -> None:
    folded = provenance(
        [
            placed(1, WORKING, "draft.md", b"first"),
            rewritten(4, WORKING, "draft.md", b"second"),
            rewritten(7, WORKING, "draft.md", b"third"),
        ]
    )

    record = folded[(WORKING.directory, "draft.md")]
    assert record.placed_at_seq == 1
    assert record.last_written_seq == 7
    assert record.revisions == 2
    assert record.digest == digest_of(b"third"), (
        "the current bytes are the current hash"
    )


def test_a_rewrite_whose_placement_was_expired_reports_no_placement() -> None:
    """A read above the retained floor is a real read, not a broken one.

    `None` rather than the rewrite's own sequence: backfilling it would manufacture a
    placement the log does not carry, and "placed at 9" would then be indistinguishable
    from a genuine placement at 9.
    """
    folded = provenance([rewritten(9, WORKING, "draft.md", b"only this survived")])

    record = folded[(WORKING.directory, "draft.md")]
    assert record.placed_at_seq is None
    assert record.last_written_seq == 9
    assert record.revisions == 1


def test_an_event_type_the_fold_does_not_know_changes_nothing() -> None:
    """Totality: a delivery or approval act can be a third type later with no edit
    here."""
    folded = provenance(
        [
            Event(A_SESSION, Seq(1), "session.created", {}),
            placed(2, ARTIFACTS, "report.md", b"body"),
            Event(A_SESSION, Seq(3), "turn.completed", {"whatever": True}),
        ]
    )
    assert set(folded) == {(ARTIFACTS.directory, "report.md")}


def test_a_log_that_produced_nothing_folds_to_nothing() -> None:
    assert provenance([]) == {}
    assert provenance([Event(A_SESSION, Seq(1), "session.created", {})]) == {}


def test_one_path_in_two_lanes_is_two_objects() -> None:
    """The key is the pair, so a lane is not something a path can be confused across."""
    folded = provenance(
        [
            placed(1, ARTIFACTS, "notes.md", b"promised"),
            placed(2, WORKING, "notes.md", b"scratch"),
        ]
    )
    assert folded[(ARTIFACTS.directory, "notes.md")].digest == digest_of(b"promised")
    assert folded[(WORKING.directory, "notes.md")].digest == digest_of(b"scratch")


def test_a_write_with_no_sources_carries_no_sources_field_at_all() -> None:
    """Absent, not `[]`. An event says a writer declared something only when it did."""
    assert "sources" not in written_payload(WORKING, "a.txt", digest_of(b"x"))
    assert "sources" in written_payload(
        WORKING, "a.txt", digest_of(b"x"), (SourceRef("b.txt", digest_of(b"y")),)
    )


_BROKEN: tuple[tuple[str, dict[str, object]], ...] = (
    ("no lane", {"relative": "a.txt", "digest": digest_of(b"x").model_dump()}),
    ("no path", {"lane": WORKING.directory, "digest": digest_of(b"x").model_dump()}),
    ("no digest", {"lane": WORKING.directory, "relative": "a.txt"}),
    (
        "sources not a list",
        {
            "lane": WORKING.directory,
            "relative": "a.txt",
            "digest": digest_of(b"x").model_dump(),
            "sources": {"relative": "b.txt"},
        },
    ),
    (
        "a source with no digest",
        {
            "lane": WORKING.directory,
            "relative": "a.txt",
            "digest": digest_of(b"x").model_dump(),
            "sources": [{"relative": "b.txt"}],
        },
    ),
)
"""Payloads a writer should never produce, one per way of being malformed.

Each is graded on its own so a check that stops covering one of them says which. The
fold raises rather than skipping: a provenance view that quietly omitted an object would
report a produced artifact as never produced, which is the failure the record exists to
prevent.
"""


@pytest.mark.parametrize(
    ("what", "payload"), _BROKEN, ids=[case[0] for case in _BROKEN]
)
def test_a_malformed_payload_is_refused_and_names_the_event(
    what: str, payload: Mapping[str, object]
) -> None:
    with pytest.raises(ValueError, match="seq 5"):
        provenance([Event(A_SESSION, Seq(5), OBJECT_PLACED, dict(payload))])
