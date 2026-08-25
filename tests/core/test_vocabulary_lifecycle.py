"""The projection's transition table and the published vocabulary do not drift apart.

Tier 1 (local, no infrastructure). Two directions, and both matter. A table entry
whose event type nothing declares is a transition that can never fire — nothing
may emit an unpublished type, so the row is dead and the state it leads to is
unreachable. A declared lifecycle type that no table entry reads is an event a
tenant can be sent that moves the state nowhere. The two files are edited by
different slices, which is exactly why the agreement between them needs a test
rather than a convention.

The stop reasons below are graded the same way and for the same reason. Each member
encodes a decision — which of the two ending events carries it — and that decision is
made in a `Literal` on a payload model, so a member added to the enum without being
admitted by either model would be a reason nothing can ever append. One assertion per
member, parametrized over the enum, because a single assertion over the whole set is
satisfied by whichever member happens to be checked first.
"""

from typing import get_args

import pytest

from managed_agent.core import vocabulary
from managed_agent.core.session.projection import _TRANSITIONS
from managed_agent.core.vocabulary import lifecycle
from managed_agent.core.vocabulary.lifecycle import StopReason


def _session_transitions() -> set[str]:
    return {name for name in _TRANSITIONS if name.startswith("session.")}


def test_every_session_transition_the_projection_reads_is_published() -> None:
    """A row here with no declared type is a state the fold can never arrive at."""
    undeclared = {
        name for name in _session_transitions() if not vocabulary.is_published(name)
    }
    assert undeclared == set(), (
        f"transition table reads unpublished types: {sorted(undeclared)}"
    )


def test_every_lifecycle_type_moves_the_state_somewhere() -> None:
    """A published lifecycle event the fold ignores is a transition nobody wired."""
    declared = {
        value
        for name, value in vars(lifecycle).items()
        if name.isupper() and name != "FAMILY" and isinstance(value, str)
    }
    assert declared, "the lifecycle family declares no event types"
    assert declared <= set(_TRANSITIONS), (
        f"unread lifecycle types: {sorted(declared - set(_TRANSITIONS))}"
    )


def test_the_lifecycle_family_is_registered_under_its_own_name() -> None:
    assert vocabulary.PUBLISHED[lifecycle.SESSION_CREATED] == lifecycle.FAMILY


def _reasons_of(
    model: type[lifecycle.SessionSuspended | lifecycle.SessionStopped],
) -> set[StopReason]:
    """The stop reasons one payload model's `Literal` admits.

    Read off the annotation rather than restated here, so this file holds no second copy
    of the partition. A test that listed the members itself would agree with itself
    after somebody widened a `Literal` and forgot the other one.
    """
    return set(get_args(model.model_fields["stop_reason"].annotation))


@pytest.mark.parametrize("reason", list(StopReason), ids=lambda r: r.value)
def test_every_stop_reason_is_carried_by_exactly_one_ending_event(
    reason: StopReason,
) -> None:
    """A reason both events admit, or neither does, is a reason nothing can append.

    Neither is the defect that matters: a member sitting in the enum that no payload
    model will accept is a value a consumer can read in the docs, branch on, and never
    see. Both is the other direction — a reason that could mean "parked" or "finished"
    depending on which event carried it, which is two facts wearing one name.
    """
    carried_by = [
        event
        for event, model in (
            (lifecycle.SESSION_SUSPENDED, lifecycle.SessionSuspended),
            (lifecycle.SESSION_STOPPED, lifecycle.SessionStopped),
        )
        if reason in _reasons_of(model)
    ]
    assert len(carried_by) == 1, (
        f"{reason.value} is carried by {carried_by or 'no ending event'}; every stop "
        "reason belongs to exactly one of session.suspended and session.stopped"
    )


@pytest.mark.parametrize("reason", list(StopReason), ids=lambda r: r.value)
def test_a_stop_reason_is_refused_by_the_event_that_does_not_carry_it(
    reason: StopReason,
) -> None:
    """The partition is enforced at run time and not only by the type checker.

    `mypy --strict` already refuses the wrong pairing at every call site in `src/`, and
    that is the guard that matters. This one exists because the payloads are also parsed
    back out of a JSONB column by anything reading a log, where no annotation is
    present — so the model has to refuse the pairing on its own.
    """
    for model in (lifecycle.SessionSuspended, lifecycle.SessionStopped):
        if reason in _reasons_of(model):
            assert model(stop_reason=reason).stop_reason is reason  # type: ignore[arg-type]
            continue
        with pytest.raises(ValueError):
            model(stop_reason=reason)  # type: ignore[arg-type]
