"""A Session's state is a fold over its own log, and nothing else.

Tier 1 (local, no infrastructure). What is graded here is that the fold is total,
ordered and refuses to invent: an event type it has no case for advances the
sequence and changes nothing (so a later slice adding a type cannot break a state
read), the answer depends on the order the events are read in (so the state really
is "what the log says" rather than a set-membership test), and a log with no
creation event raises instead of defaulting to a state the log never recorded.
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


def test_a_created_session_is_running() -> None:
    assert project(_log("session.created")) == (SessionState.RUNNING, 1)


def test_a_suspended_session_reads_as_suspended() -> None:
    state, seq = project(_log("session.created", "session.suspended"))
    assert state is SessionState.SUSPENDED
    assert seq == 2


def test_a_stopped_session_reads_as_stopped_and_takes_no_turn() -> None:
    state, _ = project(_log("session.created", "session.stopped"))
    assert state is SessionState.STOPPED
    assert not state.accepts_a_turn()


def test_a_resumed_session_is_running_again() -> None:
    state, seq = project(
        _log("session.created", "session.suspended", "session.resumed")
    )
    assert state is SessionState.RUNNING
    assert seq == 3


def test_the_order_events_are_read_in_decides_the_answer() -> None:
    """The same events in the other order give the other state.

    This is the difference between folding a log forward and asking which events a log
    contains; only the first can tell a Session that was suspended from one that was
    suspended and then resumed.
    """
    suspended, _ = project(
        _log("session.created", "session.resumed", "session.suspended")
    )
    resumed, _ = project(
        _log("session.created", "session.suspended", "session.resumed")
    )
    assert suspended is SessionState.SUSPENDED
    assert resumed is SessionState.RUNNING


def test_an_unknown_event_type_advances_the_sequence_and_moves_no_state() -> None:
    """A type the table has no case for is read, counted, and otherwise ignored.

    This is what lets a later slice add an event type without editing this function: the
    fold is total over types it has never heard of, so a state read cannot break on one.
    """
    state, seq = project(_log("session.created", "turn.started", "turn.completed"))
    assert state is SessionState.RUNNING
    assert seq == 3


def test_a_log_with_no_creation_event_refuses_to_invent_a_state() -> None:
    with pytest.raises(ValueError, match="session.created"):
        project(_log("turn.started"))


def test_an_empty_log_refuses_rather_than_reporting_a_state() -> None:
    """A Session with no events does not exist, and a default here would invent one."""
    with pytest.raises(ValueError):
        project([])
