"""The turn family: what it publishes, what a submission may carry, and the fold.

Tier 1 (local, no infrastructure). Three properties.

The five types are published under one family, so a consumer can select the turn events
out of a mixed log by family rather than by a list of names it has to keep in step.

`TurnSubmitted` parses rather than validates: the value that comes out is proof the key
was well formed and the prompt was not empty, so nothing downstream re-checks either.

And the projection stays total over the new types. `core/session/projection.py` is not
edited by this family and claims a type it has no case for advances the sequence and
changes nothing; five new types is the first time that claim is asked to hold for a
whole family at once, so it is exercised rather than assumed.
"""

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from managed_agent.core import vocabulary
from managed_agent.core.ids import Seq, SessionId, new_session_id, new_turn_id
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import turn

_TURN_TYPES = (
    turn.TURN_SUBMITTED,
    turn.TURN_STARTED,
    turn.TURN_MESSAGE_DELTA,
    turn.TURN_COMPLETED,
    turn.TURN_FAILED,
)


@dataclass(frozen=True, slots=True)
class Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object] = field(default_factory=dict)


def _a_submission(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "turn_id": str(new_turn_id()),
        "idempotency_key": "retry-key-0001",
        "prompt": "summarise the findings",
    }
    body.update(overrides)
    return body


def test_the_five_turn_types_are_published_under_the_turn_family() -> None:
    """All five, named individually, so a type dropped from the module fails here."""
    assert turn.FAMILY == "turn"
    assert [vocabulary.PUBLISHED.get(type_) for type_ in _TURN_TYPES] == ["turn"] * 5


def test_a_failure_is_its_own_type_rather_than_a_field_on_the_completion() -> None:
    """The one mistake the split makes impossible.

    A consumer that handles `turn.completed` cannot read a failed Turn as a success by
    forgetting a field, because a failed Turn never arrives as `turn.completed`.
    """
    assert turn.TURN_COMPLETED != turn.TURN_FAILED
    assert "status" not in turn.TurnSubmitted.model_fields


def test_the_failure_causes_are_exactly_the_three_ways_a_turn_produces_no_answer() -> (
    None
):
    """Closed, and closed at three: a fourth member is a published-vocabulary change."""
    assert [cause.value for cause in turn.TurnFailureCause] == [
        "runtime_reported_failure",
        "runtime_lost",
        "pod_unreachable",
    ]


def test_a_well_formed_submission_parses() -> None:
    submission = turn.TurnSubmitted.model_validate(_a_submission())

    assert submission.idempotency_key == "retry-key-0001"
    assert submission.prompt == "summarise the findings"


def test_a_submission_refuses_a_field_nobody_declared() -> None:
    """A caller that misspelled a field would otherwise believe it had set one."""
    with pytest.raises(ValidationError):
        turn.TurnSubmitted.model_validate(_a_submission(priority="high"))


def test_a_submission_refuses_an_empty_prompt() -> None:
    """An empty Turn is a Turn the runtime would answer with nothing, at full cost."""
    with pytest.raises(ValidationError):
        turn.TurnSubmitted.model_validate(_a_submission(prompt=""))


@pytest.mark.parametrize("key", ["short12", "", "has spaces here", "x" * 256])
def test_a_submission_refuses_a_key_outside_the_one_rule(key: str) -> None:
    """Seven characters, none, whitespace, and one past the ceiling.

    The rule lives on `IdempotencyKey` and both the header and this payload annotate
    it, so a key this refuses is a key the route refuses too -- which is what stops a
    value being accepted at the door and rejected by the log that has to store it.
    """
    with pytest.raises(ValidationError):
        turn.TurnSubmitted.model_validate(_a_submission(idempotency_key=key))


def test_a_parsed_submission_cannot_be_rewritten() -> None:
    """It is read back by every later admission decision for this Session."""
    submission = turn.TurnSubmitted.model_validate(_a_submission())

    with pytest.raises(ValidationError):
        submission.turn_id = new_turn_id()


def test_the_projection_stays_total_over_the_whole_turn_family() -> None:
    """Five new types advance the sequence and change the state not at all.

    Written as a whole family in one log rather than one type per case, because the
    property is about the fold's default arm and a single unknown type exercises it as
    well as five do -- what five buys is that no one of them quietly grew a transition.
    """
    session_id = new_session_id()
    log = [Event(session_id, Seq(1), "session.created")]
    log += [
        Event(session_id, Seq(index), type_)
        for index, type_ in enumerate(_TURN_TYPES, start=2)
    ]

    assert project(log) == (SessionState.RUNNING, 6)


def test_a_turn_family_event_does_not_undo_a_stop() -> None:
    """The other half of totality: the fold's answer still comes from the lifecycle.

    Without this, a transition accidentally added for a turn type could move a stopped
    Session back to running and the test above would not notice -- it starts from
    `session.created` and expects RUNNING either way.
    """
    session_id = new_session_id()
    log = [
        Event(session_id, Seq(1), "session.created"),
        Event(session_id, Seq(2), "session.stopped"),
        Event(session_id, Seq(3), turn.TURN_SUBMITTED),
        Event(session_id, Seq(4), turn.TURN_COMPLETED),
    ]

    assert project(log) == (SessionState.STOPPED, 4)
