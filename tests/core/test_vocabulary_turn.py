"""The turn family: what it publishes, what a submission may carry, and the fold.

Tier 1 (local, no infrastructure). Three properties.

Every type is published under one family, so a consumer can select the turn events out
of a mixed log by family rather than by a list of names it has to keep in step. The
count is deliberately not written down here any more: a test that asserts "there are
five" is edited to say six by whoever adds the sixth, which is the same hand that was
supposed to be checked.

`TurnSubmitted` parses rather than validates: the value that comes out is proof the key
was well formed and the prompt was not empty, so nothing downstream re-checks either.

And the fold reads only the types that open or close a Turn.
`core/session/projection.py` folds `turn.submitted` to `RUNNING` and both terminal types
back to `IDLE`, because `RUNNING` means a Turn is executing and only the turn family
knows when one is. The rest -- `turn.started`, `turn.message_delta` and `turn.progress`
-- move nothing: they report from inside a Turn that is already open, and a state that
changed on them would be saying the same thing twice. Which types move the fold is
asserted here rather than assumed, because it is the whole reason this family and that
fold are coupled at all.
"""

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from managed_agent.core import vocabulary
from managed_agent.core.ids import Seq, SessionId, new_session_id, new_turn_id
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import turn

_NAMED_TURN_TYPES = (
    turn.TURN_SUBMITTED,
    turn.TURN_STARTED,
    turn.TURN_MESSAGE_DELTA,
    turn.TURN_COMPLETED,
    turn.TURN_FAILED,
    turn.TURN_PROGRESS,
)
"""Each type spelled out, so one dropped from the module fails a test rather than
silently shrinking the set every other check here derives."""

_TURN_TYPES = tuple(
    type_ for type_, family in vocabulary.PUBLISHED.items() if family == turn.FAMILY
)
"""The family as the registry actually holds it, which is what the totality checks
below must be written against.

This used to be the hand-written tuple above, and the fold's table asserted itself
equal to it -- so the claim "every published type is graded here" held only for as
long as somebody remembered to edit both. It was not a hypothetical: `turn.progress`
was added to the family and every check in this file stayed green without grading it.
Derived here so a new member of the family fails the fold's table until somebody has
decided what it does to a Session's state."""


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


def test_every_named_turn_type_is_published_under_the_turn_family() -> None:
    """Each named individually, so a type dropped from the module fails here.

    The two sets are compared as well as looked up: the named tuple catches a type
    that left the module, and the equality catches one that joined the family without
    being named here -- which the previous count-based assertion could not, because a
    sixth published type simply was not in the list it counted.
    """
    assert turn.FAMILY == "turn"
    published = [vocabulary.PUBLISHED.get(type_) for type_ in _NAMED_TURN_TYPES]
    assert published == ["turn"] * len(_NAMED_TURN_TYPES)
    assert set(_NAMED_TURN_TYPES) == set(_TURN_TYPES)


def test_a_failure_is_its_own_type_rather_than_a_field_on_the_completion() -> None:
    """The one mistake the split makes impossible.

    A consumer that handles `turn.completed` cannot read a failed Turn as a success by
    forgetting a field, because a failed Turn never arrives as `turn.completed`.
    """
    assert turn.TURN_COMPLETED != turn.TURN_FAILED
    assert "status" not in turn.TurnSubmitted.model_fields


def test_the_failure_causes_are_exactly_the_ways_a_turn_produces_no_answer() -> None:
    """Closed: a member added here is a published-vocabulary change, and is one edit.

    Which is what this guard is for. It failed on 2026-08-25 when
    `output_not_revisable` was added, and that failure is the guard working -- a
    consumer branching on this set learns of a new member from a release note, never
    from a value it has no arm for.

    The order is asserted along with the membership, because `StrEnum` preserves
    declaration order and the published documentation is generated from it: a member
    inserted in the middle re-orders a table somebody reads.
    """
    assert [cause.value for cause in turn.TurnFailureCause] == [
        "runtime_reported_failure",
        "runtime_lost",
        "pod_unreachable",
        "output_not_revisable",
        "session_not_placeable",
        "runtime_did_not_start",
        "runtime_refused_the_turn",
        "turn_deadline_exceeded",
        "no_runtime_configured",
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


def test_the_fold_reads_a_submission_and_both_closures_and_nothing_else() -> None:
    """Which of the five move the state, asserted one type at a time.

    A Session's state answers "is a Turn executing right now", so exactly the events
    that open and close a Turn may move it. `turn.started` and `turn.message_delta` are
    the interesting rows: both arrive while a Turn is open and both must leave the
    state where the submission put it, because a fold that reset on progress would
    report a Session as idle in the middle of its own work.

    Written per type rather than as one family log, which the previous version of this
    test could afford when the answer was "none of them move it". Now that three do,
    running them together would let a wrong row hide behind a right one that follows.
    """
    moves: dict[str, SessionState] = {
        turn.TURN_SUBMITTED: SessionState.RUNNING,
        turn.TURN_STARTED: SessionState.RUNNING,
        turn.TURN_MESSAGE_DELTA: SessionState.RUNNING,
        turn.TURN_COMPLETED: SessionState.IDLE,
        turn.TURN_FAILED: SessionState.IDLE,
        # Arrives on a timer for as long as a Turn is running, so like the two rows
        # above it must leave the state where the submission put it. A fold that reset
        # on a progress report would call a Session idle every thirty seconds in the
        # middle of its own work.
        turn.TURN_PROGRESS: SessionState.RUNNING,
    }
    assert set(moves) == set(_TURN_TYPES), "every published type is graded here"

    session_id = new_session_id()
    for type_, expected in moves.items():
        log = [
            Event(session_id, Seq(1), "session.created"),
            Event(session_id, Seq(2), turn.TURN_SUBMITTED),
            Event(session_id, Seq(3), type_),
        ]
        assert project(log) == (expected, 3), type_


def test_a_turn_family_event_does_not_undo_a_stop() -> None:
    """A stop outranks every Turn event that lands behind it.

    This guard mattered more once the fold started reading these types. While no turn
    type moved the state it was a check on a row nobody had written; now that
    `turn.submitted` means `RUNNING`, a submission racing an archive would read a
    stopped Session as running again if the stop were not the end of the fold. That
    race is reachable -- the admission path appends its submission before it re-reads
    the log -- so this is a live case rather than a hypothetical one.
    """
    session_id = new_session_id()
    log = [
        Event(session_id, Seq(1), "session.created"),
        Event(session_id, Seq(2), "session.stopped"),
        Event(session_id, Seq(3), turn.TURN_SUBMITTED),
        Event(session_id, Seq(4), turn.TURN_COMPLETED),
    ]

    assert project(log) == (SessionState.STOPPED, 4)
