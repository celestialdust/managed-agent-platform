"""The fold behind ship-out's "already delivered" question, graded against a paging log.

**Every case here runs against a log whose page cap is two rows.** That is the whole
point of the file. This repository has shipped the capped-read defect three times --
`docs/lessons.md` carries all three -- and the first guard written for it asserted that
the caller passed a wide enough `limit`, which the adapter's 500-row default satisfies
for any log short enough to write in a test. It passed against the very mutation it
existed to catch.

So the fake below defaults to **two** rows and the cases assert on the paths delivered,
not on the arguments passed. A fold that reads one page reports the state as of that
page, which reads exactly like the state as of the log and is wrong only for the
Sessions old enough for it to matter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pytest

from managed_agent.control.files.lane_digests import (
    SessionLogReader,
    digests_in_lane,
)
from managed_agent.core.ids import Seq, SessionId, new_session_id
from managed_agent.core.ports import EventRecord
from managed_agent.core.vfs.evidence import digest_of
from managed_agent.core.vfs.session_vfs import ARTIFACTS, SealedLane
from managed_agent.core.vocabulary import vfs

A_SECOND_LANE: Final = SealedLane("evidence")
"""Some lane that is not `ARTIFACTS`, for the case that grades lane separation.

Declared here rather than taken from `session_vfs.LANES`, which holds exactly one lane
now that the workspace is a mounted volume (ADR-035). The separation this file grades is
a property of the fold, not of how many lanes the platform happens to write today, so
the case must keep working the moment a second one is declared -- and it would silently
stop grading anything if it were parametrized over a one-member tuple.
"""

_PAGE: Final = 2
"""How many rows this file's log hands back per read, unless a case says otherwise.

Two, so that every fold in this file needs several reads to finish. A cap that any of
these cases could satisfy in one read is a cap that grades nothing -- and 500, the real
adapter's default, is exactly such a cap for every log short enough to build in a test.
"""


class _Row:
    """One event as `EventRecord` reads it.

    A class rather than the real record type because the real one is built by an
    adapter from a database row, and nothing here has a database. It carries
    `session_id` even though every case runs one Session, because the port declares
    the field and a double that drops one would let a caller reading the wrong
    Session's events pass.
    """

    def __init__(
        self, session_id: SessionId, seq: Seq, type: str, payload: dict[str, object]
    ) -> None:
        self.session_id = session_id
        self.seq = seq
        self.type = type
        self.payload = payload


class CappedLog:
    """A log that hands back at most `page` rows per read, defaulting to two.

    **A wide `limit` does not defeat it, and that is the honest shape.** A real adapter
    caps at its own maximum page, so a caller asking for a billion rows still gets one
    page -- and a fold that relied on asking for enough would pass its tests and
    truncate in production. Only paging until the log is exhausted works against this,
    which is the guarantee these cases exist to hold.
    """

    def __init__(self, session_id: SessionId, page: int = _PAGE) -> None:
        self._session_id = session_id
        self._page = page
        self.records: list[EventRecord] = []
        self.reads: list[tuple[Seq, Seq, int]] = []

    def placed(self, lane: str, relative: str, body: bytes) -> None:
        seq = Seq(len(self.records) + 1)
        self.records.append(
            _Row(
                session_id=self._session_id,
                seq=seq,
                type=vfs.OBJECT_PLACED,
                payload={
                    "lane": lane,
                    "relative": relative,
                    "digest": digest_of(body).model_dump(),
                    "byte_length": len(body),
                },
            )
        )

    def noise(self, type_: str) -> None:
        """An event the fold does not know, to prove the fold is total over them."""
        seq = Seq(len(self.records) + 1)
        self.records.append(
            _Row(session_id=self._session_id, seq=seq, type=type_, payload={"any": "x"})
        )

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]:
        self.reads.append((start, end, limit))
        found = [r for r in self.records if start <= r.seq <= end]
        return found[: min(limit, self._page)]


_PORT: SessionLogReader = CappedLog(new_session_id())
"""Graded against the Protocol by mypy --strict rather than at run time: the Protocol is
not runtime_checkable, and an isinstance against one that was would compare method names
and not signatures, which is the half that drifts."""


async def test_the_fold_reads_past_the_first_page() -> None:
    """Five writes over a two-row page. A one-page fold sees the first two.

    The assertion is on the paths delivered, never on the `limit` the caller passed. An
    assertion about the argument is satisfiable by a default, which is exactly how the
    third occurrence of this defect got past its own guard.
    """
    session_id = new_session_id()
    log = CappedLog(session_id)
    for index in range(5):
        log.placed(ARTIFACTS.directory, f"report{index}.md", f"body {index}".encode())

    found = await digests_in_lane(log, session_id, ARTIFACTS.directory)

    assert sorted(found) == [f"report{index}.md" for index in range(5)]
    assert len(log.reads) > 1, "the whole log came back in one read; the cap is inert"


async def test_a_path_carries_the_digest_of_its_own_bytes() -> None:
    """The value is what a pod's own SHA-256 of the same bytes would be.

    Compared against `digest_of` rather than against a literal: the comparison a caller
    makes is between this and a hex string a pod computed, so a value that were the
    model's repr instead of its `hex` field would never match anything and would degrade
    silently to re-transferring every file every Turn.
    """
    session_id = new_session_id()
    log = CappedLog(session_id)
    log.placed(ARTIFACTS.directory, "report.md", b"# Report\n")

    found = await digests_in_lane(log, session_id, ARTIFACTS.directory)

    assert found["report.md"] == digest_of(b"# Report\n").hex


async def test_one_lanes_digests_do_not_leak_into_anothers() -> None:
    """Two lanes write `notes.md`, and they are two objects at two keys.

    A fold that ignored the lane would answer the other lane's digest for the artifacts
    lane's path, and ship-out would read that as "already delivered" and drop a document
    that had never been stored.
    """
    session_id = new_session_id()
    log = CappedLog(session_id)
    log.placed(ARTIFACTS.directory, "notes.md", b"delivered")
    log.placed(A_SECOND_LANE.directory, "notes.md", b"draft")

    artifacts = await digests_in_lane(log, session_id, ARTIFACTS.directory)
    other = await digests_in_lane(log, session_id, A_SECOND_LANE.directory)

    assert artifacts["notes.md"] == digest_of(b"delivered").hex
    assert other["notes.md"] == digest_of(b"draft").hex


async def test_events_the_fold_does_not_know_are_walked_past() -> None:
    """A Session's log is mostly Turn events, and the fold has to survive them.

    Interleaved rather than appended, so a fold that stopped at the first unknown type
    would lose the writes after it rather than all of them -- the failure that would
    look like "some files re-upload sometimes".
    """
    session_id = new_session_id()
    log = CappedLog(session_id)
    log.placed(ARTIFACTS.directory, "first.md", b"one")
    log.noise("turn.started")
    log.noise("turn.completed")
    log.placed(ARTIFACTS.directory, "second.md", b"two")

    found = await digests_in_lane(log, session_id, ARTIFACTS.directory)

    assert sorted(found) == ["first.md", "second.md"]


async def test_a_session_that_wrote_nothing_folds_to_nothing() -> None:
    """Empty, and not an error. Ship-out asks this before its first Turn ever ships."""
    session_id = new_session_id()

    assert await digests_in_lane(CappedLog(session_id), session_id, "artifacts") == {}


@pytest.mark.parametrize("page", [1, 2, 3, 7, 500])
async def test_the_fold_is_the_same_answer_at_every_page_size(page: int) -> None:
    """Whatever the adapter's cap turns out to be, the answer is the whole log.

    Parametrized over page sizes on both sides of the row count, because a walk that
    terminates on a *short* page rather than an empty one is correct for exactly the
    sizes that divide it evenly and silently truncates for the rest.
    """
    session_id = new_session_id()
    log = CappedLog(session_id, page=page)
    for index in range(6):
        log.placed(ARTIFACTS.directory, f"f{index}.md", f"{index}".encode())

    assert len(await digests_in_lane(log, session_id, ARTIFACTS.directory)) == 6
