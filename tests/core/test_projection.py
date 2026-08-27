"""A Session's state is a fold over its own log, and nothing else.

Tier 1 (local, no infrastructure). What is graded here is that the fold is ordered,
refuses to invent, and answers the one question `SessionState` asks: is a Turn
executing right now. The answer depends on the order the events are read in (so the
state really is "what the log says" rather than a set-membership test), a log with no
creation event raises instead of defaulting to a state the log never recorded, and a
stop is the end of the fold rather than one more move in it.
"""

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from managed_agent.core.ids import Seq, SessionId
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState

_SESSION = SessionId(uuid4())


@dataclass(frozen=True, slots=True)
class Event:
    """A stand-in for a stored event, carrying only what the fold reads.

    A hand-rolled record rather than the Postgres adapter's Row: the projection is pure
    and this test has no business needing a database to grade it.
    """

    seq: Seq
    type: str
    session_id: SessionId = _SESSION
    payload: dict[str, object] = field(default_factory=dict)


def _log(*types: str) -> list[Event]:
    """Number a run of event types from 1, the way a Session's log is numbered."""
    return [Event(seq=Seq(index), type=type_) for index, type_ in enumerate(types, 1)]


def test_a_created_session_is_idle() -> None:
    """Alive and at rest. Nothing is executing on a Session that has only just begun."""
    assert project(_log("session.created")) == (SessionState.IDLE, 1)


def test_a_submitted_turn_is_the_one_thing_that_makes_a_session_running() -> None:
    state, seq = project(_log("session.created", "turn.submitted"))
    assert state is SessionState.RUNNING
    assert seq == 2


def test_a_completed_turn_returns_the_session_to_idle() -> None:
    """Finishing work leaves the Session takeable, not finished."""
    state, _ = project(_log("session.created", "turn.submitted", "turn.completed"))
    assert state is SessionState.IDLE


def test_a_failed_turn_returns_the_session_to_idle() -> None:
    """A Turn that failed ends the Turn, not the Session. The next one may succeed."""
    state, _ = project(_log("session.created", "turn.submitted", "turn.failed"))
    assert state is SessionState.IDLE


def test_a_suspended_session_is_idle_because_the_pod_is_a_separate_axis() -> None:
    """The suspension says a pod was handed back, which is not a fact about the Session.

    A Session outlives every pod it is given, so reclaiming one moves it to rest rather
    than out of service -- the next Turn asks for a pod and gets one.
    """
    state, seq = project(_log("session.created", "session.suspended"))
    assert state is SessionState.IDLE
    assert seq == 2


def test_a_resumed_session_is_idle_for_the_same_reason() -> None:
    """A pod arriving moves the Session no more than a pod leaving did.

    Both events describe the pod, and the pair is only readable as a pair if neither
    claims to change whether the Session will work again.
    """
    state, _ = project(_log("session.created", "session.suspended", "session.resumed"))
    assert state is SessionState.IDLE


def test_a_pod_placed_during_a_turn_leaves_the_session_running() -> None:
    """The one moment `session.resumed` can be appended, and the trap it sets.

    A pod is placed lazily, by the Turn that finds none -- so the placement event lands
    *between* that Turn's submission and its completion, and there is no other moment it
    can land at. A row folding it to `IDLE` would therefore read every cold-started
    Session as at rest while its agent was executing, and `accepts_a_turn()` is `is
    IDLE`, so the next submission would be admitted onto a Session already working.

    That is what the table's `None` for this type buys, and it is why the pod pair is
    not symmetric: a reclaim is only ever appended with no Turn open, so `IDLE` is a
    true statement there and would be a false one here.
    """
    state, _ = project(_log("session.created", "turn.submitted", "session.resumed"))
    assert state is SessionState.RUNNING


def test_a_stopped_session_reads_as_stopped_and_takes_no_turn() -> None:
    state, _ = project(_log("session.created", "session.stopped"))
    assert state is SessionState.STOPPED
    assert not state.accepts_a_turn()


def test_a_stop_absorbs_every_event_that_lands_behind_it() -> None:
    """The one asymmetry in the fold, and it is load-bearing rather than tidy.

    A submission can legitimately land after a stop: the admission path appends before
    it re-reads, so a submitter racing an archive leaves its `turn.submitted` in a log
    that has already stopped. Under a rule where the last row simply wins, that event
    would read the Session back to `RUNNING` -- a stopped Session that accepts Turns
    again, produced by a race nothing refuses. So the stop ends the fold.
    """
    state, seq = project(
        _log("session.created", "session.stopped", "turn.submitted", "turn.completed")
    )
    assert state is SessionState.STOPPED
    assert seq == 4, "the sequence still advances; only the state stops moving"


def test_a_stop_absorbs_a_suspension_that_races_it() -> None:
    """The same race from the sweep's side: it folds, an archive stops, then it appends.

    Without the absorbing rule the suspension would report the archived Session as
    merely at rest, and the next Turn would be admitted onto a Session the tenant had
    ended.
    """
    state, _ = project(_log("session.created", "session.stopped", "session.suspended"))
    assert state is SessionState.STOPPED


def test_the_order_events_are_read_in_decides_the_answer() -> None:
    """The same events in the other order give the other state.

    This is the difference between folding a log forward and asking which events a log
    contains; only the first can tell a Session running a Turn from one that has
    finished it.
    """
    finished, _ = project(_log("session.created", "turn.submitted", "turn.completed"))
    started, _ = project(_log("session.created", "turn.completed", "turn.submitted"))
    assert finished is SessionState.IDLE
    assert started is SessionState.RUNNING


def test_an_event_type_the_table_has_no_row_for_moves_no_state() -> None:
    """A type outside the folded set is read, counted, and otherwise ignored.

    The fold is no longer total over every type -- the turn family moves it now -- but
    it is still total over the families that say nothing about whether work is running,
    which is what lets those grow without this function being edited.
    """
    state, seq = project(
        _log("session.created", "turn.started", "turn.message_delta", "tool.called")
    )
    assert state is SessionState.IDLE
    assert seq == 4


def test_a_log_with_no_creation_event_refuses_to_invent_a_state() -> None:
    with pytest.raises(ValueError, match="session.created"):
        project(_log("turn.started"))


def test_an_empty_log_refuses_rather_than_reporting_a_state() -> None:
    """A Session with no events does not exist, and a default here would invent one."""
    with pytest.raises(ValueError):
        project([])
