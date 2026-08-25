"""What one submission gets, and that a second one never buys a second Turn.

Tier 1 (local, no infrastructure). Realizes the admission half of MAP-A9: a Turn sent
to a stopped Session is refused and no work begins, and re-submitting one key inside a
Session does not start a second Turn.

Two hazards drive most of this file. A client whose submission times out retries, and
the retry must get the Turn it already has rather than a new one. And two submissions
can be in flight at once, each folding a log that does not yet contain the other -- so
the sixteen-way case below is not a stress test, it is the case the algorithm exists
for.

The fake log **pages**. `EventLogRange.read` returns at most `limit` records and treats
a short result as "page for the rest", and this repository has shipped a caller taking
that default three times (MAP-3, MAP-7, MAP-51). A fake that always returned everything
it held could not fail that way, so `NarrowPages` below defaults to two rows: against
it, a read that names no limit sees two events of a longer log and answers wrongly. The
assertions are on the verdicts delivered, never on the arguments passed, because an
assertion about an argument can be satisfied by a default.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from uuid import UUID

import pytest

from managed_agent.control.session.lifecycle import (
    TurnAdmitted,
    TurnRefused,
    TurnReplayed,
    admit_turn,
)
from managed_agent.core.ids import FIRST_SEQ, Seq, SessionId, new_session_id
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import turn

_KEY = "retry-key-0001"
_PROMPT = "summarise the findings"


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


class InMemoryLog:
    """Both log ports over one list, counting appends and honouring `limit`.

    Every operation yields to the event loop before it answers. That is what lets the
    concurrency case below have sixteen submissions genuinely overlapping: a fake that
    never awaited would run each `admit_turn` to completion before the next one
    started, and the interleaving the algorithm is written for would never happen.
    """

    default_page = 500

    def __init__(self) -> None:
        self._events: list[Event] = []
        self.appends = 0

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        await asyncio.sleep(0)
        self.appends += 1
        return self.add(session_id, type_, payload)

    def add(
        self,
        session_id: SessionId,
        type_: str,
        payload: dict[str, object] | None = None,
    ) -> Seq:
        """Append without going through the port, to stand in for another writer."""
        seq = Seq(len(self._events) + 1)
        self._events.append(Event(session_id, seq, type_, dict(payload or {})))
        return seq

    def types(self) -> list[str]:
        return [event.type for event in self._events]

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int | None = None
    ) -> Sequence[Event]:
        await asyncio.sleep(0)
        span = [
            event
            for event in self._events
            if event.session_id == session_id and start <= event.seq <= end
        ]
        return span[: self.default_page if limit is None else limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Event]:
        for event in self._events:
            if event.session_id == session_id and event.seq > after:
                yield event

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return FIRST_SEQ


class NarrowPages(InMemoryLog):
    """The same log with a two-row default page, as a capped adapter behaves.

    The cap is expressed as the *default* rather than as a ceiling the fake enforces
    regardless, because that is what the real port does: `read` returns at most `limit`
    records and `limit` defaults to 500. A caller that names a limit wide enough for
    its span gets its span; a caller that names none gets a page. Enforcing a hard cap
    here instead would fail the callers that page correctly *and* the callers that
    prove their limit covers the range, and only one of those is a defect.
    """

    default_page = 2


def _running(log: InMemoryLog) -> SessionId:
    session_id = new_session_id()
    log.add(session_id, "session.created")
    return session_id


async def test_a_running_session_admits_a_turn_and_the_submission_lands() -> None:
    log = InMemoryLog()
    session_id = _running(log)

    admission = await admit_turn(session_id, _KEY, _PROMPT, log, log)

    assert isinstance(admission, TurnAdmitted)
    assert admission.seq == 2
    assert log.types() == ["session.created", turn.TURN_SUBMITTED]


async def test_the_recorded_submission_carries_the_key_the_turn_id_and_the_prompt() -> (
    None
):
    """The event is the record of the submission, so a later reader needs all three.

    The key in particular: the whole idempotency answer is read back out of this
    payload, so a submission recorded without it would make every retry a new Turn with
    nothing failing.
    """
    log = InMemoryLog()
    session_id = _running(log)

    admission = await admit_turn(session_id, _KEY, _PROMPT, log, log)

    assert isinstance(admission, TurnAdmitted)
    recorded = list(await log.read(session_id, Seq(2), Seq(2)))[0]
    assert recorded.type == turn.TURN_SUBMITTED
    assert recorded.payload["idempotency_key"] == _KEY
    assert recorded.payload["prompt"] == _PROMPT
    assert UUID(str(recorded.payload["turn_id"])) == admission.turn_id


async def test_the_same_key_a_second_time_replays_and_appends_nothing() -> None:
    log = InMemoryLog()
    session_id = _running(log)
    first = await admit_turn(session_id, _KEY, _PROMPT, log, log)
    assert isinstance(first, TurnAdmitted)
    appends_after_the_first = log.appends

    second = await admit_turn(session_id, _KEY, _PROMPT, log, log)

    assert second == TurnReplayed(turn_id=first.turn_id, seq=first.seq)
    assert log.appends == appends_after_the_first, (
        "the retry appended a second submission. The Turn it names is the first one, "
        "so the extra event is a Turn the log says was asked for and nothing ran."
    )


async def test_a_different_key_in_the_same_session_is_a_different_turn() -> None:
    """The key is scoped to the Session and is not a lock on it.

    Without this, an implementation that replayed *any* prior submission would pass
    every case above -- and a Session would be able to run exactly one Turn ever.
    """
    log = InMemoryLog()
    session_id = _running(log)
    first = await admit_turn(session_id, _KEY, _PROMPT, log, log)
    second = await admit_turn(session_id, "another-key-0002", _PROMPT, log, log)

    assert isinstance(first, TurnAdmitted)
    assert isinstance(second, TurnAdmitted)
    assert second.turn_id != first.turn_id


async def test_one_key_under_two_sessions_names_two_unrelated_submissions() -> None:
    """The same string is looked up inside one Session and never across them."""
    log = InMemoryLog()
    mine, theirs = _running(log), _running(log)

    first = await admit_turn(mine, _KEY, _PROMPT, log, log)
    second = await admit_turn(theirs, _KEY, _PROMPT, log, log)

    assert isinstance(first, TurnAdmitted)
    assert isinstance(second, TurnAdmitted)
    assert second.turn_id != first.turn_id


@pytest.mark.parametrize(
    ("event", "state"),
    [
        ("session.stopped", SessionState.STOPPED),
        ("session.suspended", SessionState.SUSPENDED),
    ],
)
async def test_a_session_that_takes_no_turn_is_refused_and_nothing_is_written(
    event: str, state: SessionState
) -> None:
    log = InMemoryLog()
    session_id = _running(log)
    log.add(session_id, event)

    admission = await admit_turn(session_id, _KEY, _PROMPT, log, log)

    assert admission == TurnRefused(state=state)
    assert log.appends == 0, "a refused submission was recorded as one"


async def test_a_retry_that_crosses_a_stop_still_gets_back_the_turn_it_started() -> (
    None
):
    """The idempotency check runs before the state check, and this is why.

    A submission that succeeded and then timed out on the wire must not come back as a
    refusal: the client cannot tell that apart from a submission that never happened,
    and would have no way to learn the Turn it already owns.
    """
    log = InMemoryLog()
    session_id = _running(log)
    first = await admit_turn(session_id, _KEY, _PROMPT, log, log)
    assert isinstance(first, TurnAdmitted)
    log.add(session_id, "session.stopped")

    assert await admit_turn(session_id, _KEY, _PROMPT, log, log) == TurnReplayed(
        turn_id=first.turn_id, seq=first.seq
    )


class StopsUnderTheAppend(InMemoryLog):
    """Something else stops the Session in the window between the fold and the append.

    The narrowest window there is, and the one no read before the append can see.
    """

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.add(session_id, "session.stopped")
        return await super().append(session_id, type_, payload)


async def test_a_stop_landing_between_the_fold_and_the_append_refuses_the_turn() -> (
    None
):
    """The first read said running. The settled prefix is what catches it.

    Without the second fold the Turn would be admitted and dispatched onto a Session
    that is no longer running, and the log would carry a submission followed by a stop
    with nothing saying the Turn never ran.
    """
    log = StopsUnderTheAppend()
    session_id = _running(log)

    admission = await admit_turn(session_id, _KEY, _PROMPT, log, log)

    assert admission == TurnRefused(state=SessionState.STOPPED)


async def test_sixteen_submissions_of_one_key_buy_exactly_one_turn() -> None:
    """MAP-A9's other half, settled by sequence rather than by a lock.

    Every submitter re-reads a prefix that ends at its own sequence, so a later racer
    always sees an earlier one and an earlier racer never sees a later one. The two
    facts asserted are the ones that matter: exactly one Turn is dispatchable, and all
    sixteen callers were told the same turn id, so no client walks away holding a Turn
    nobody will run.
    """
    log = InMemoryLog()
    session_id = _running(log)

    verdicts = await asyncio.gather(
        *(admit_turn(session_id, _KEY, _PROMPT, log, log) for _ in range(16))
    )

    admitted = [v for v in verdicts if isinstance(v, TurnAdmitted)]
    replayed = [v for v in verdicts if isinstance(v, TurnReplayed)]
    assert len(admitted) == 1, f"{len(admitted)} of 16 submissions each bought a Turn"
    assert len(replayed) == 15
    assert {v.turn_id for v in replayed} == {admitted[0].turn_id}


async def test_the_key_lookup_reads_past_the_first_page_of_a_long_log() -> None:
    """A submission beyond one page is found before a second one is written.

    The append count is the assertion that bites, and the reason is worth writing
    down. Replace the whole-log read with a single default read and the *verdict* is
    still right -- the settled re-read after the append sees the earlier submission and
    reports the replay. What it cannot undo is the append it already made. So the
    truncated version answers correctly and writes a `turn.submitted` per retry
    forever: a log filling with Turns that were asked for and never ran, and the reader
    of that log has no way to tell them from real ones. Verified: the mutation leaves
    the two assertions above green and fails this one with 2 appends.
    """
    log = NarrowPages()
    session_id = _running(log)
    log.add(session_id, "turn.filler")
    first = await admit_turn(session_id, _KEY, _PROMPT, log, log)
    assert isinstance(first, TurnAdmitted)
    assert first.seq == 3, "the submission has to sit outside the first page"

    assert await admit_turn(session_id, _KEY, _PROMPT, log, log) == TurnReplayed(
        turn_id=first.turn_id, seq=first.seq
    )
    assert log.appends == 1, (
        "the retry recorded a second submission, so the key lookup did not see the "
        "first one -- it sits past the first page of the log"
    )


async def test_the_state_fold_reads_past_the_first_page_of_a_long_log() -> None:
    """A stop beyond one page still refuses, rather than folding a stale page.

    Falsifiable: a single default read sees `session.created` and the filler, reports
    RUNNING, and dispatches a Turn onto a stopped Session.
    """
    log = NarrowPages()
    session_id = _running(log)
    log.add(session_id, "turn.filler")
    log.add(session_id, "session.stopped")

    admission = await admit_turn(session_id, _KEY, _PROMPT, log, log)

    assert admission == TurnRefused(state=SessionState.STOPPED)
    assert log.appends == 0


async def test_the_settled_read_covers_the_prefix_it_just_appended_into() -> None:
    """The second read names a limit, and a long enough log is what proves it.

    Falsifiable two ways, both real. Dropping `limit=seq` returns the first two rows,
    which do not contain the append -- the assertion that the winner exists fires and
    the tenant gets a 500. Widening the read to the head instead of stopping at the
    append is worse and silent: two racers would each see the other and both stand
    down, so no Turn would run at all.
    """
    log = NarrowPages()
    session_id = _running(log)
    log.add(session_id, "turn.filler")
    log.add(session_id, "turn.filler")

    admission = await admit_turn(session_id, _KEY, _PROMPT, log, log)

    assert isinstance(admission, TurnAdmitted)
    assert admission.seq == 4
