"""The stream family adds exactly one name to the published set, and one payload shape.

Tier 1 (local, no infrastructure). The point of the file is the *count*. Every name a
tenant can be sent is fixed for the life of an API version (ADR-013), so a transport
that quietly grew a second event type would be an unversioned addition that no consumer
can branch on -- and the transport is where that is most tempting, because a frame feels
like plumbing rather than vocabulary. Asserting the family holds one type is what makes
the second one fail here instead of shipping.

`test_vocabulary.py` already grades every family for registration and family agreement.
What is left, and lives here, is this family's own count and the shape of what its one
frame carries.
"""

import pytest
from pydantic import ValidationError

from managed_agent.core import vocabulary
from managed_agent.core.errors import ErrorCode
from managed_agent.core.vocabulary import stream


def _declared_types() -> dict[str, str]:
    """The event-type constants this module exposes, by constant name.

    Read off the module rather than filtered out of the registry, so a constant assigned
    without going through `declare` is counted here and then fails the published check.
    """
    return {
        name: value
        for name, value in vars(stream).items()
        if name.isupper()
        and name != "FAMILY"
        and isinstance(value, str)
        and "." in value
    }


def test_the_family_declares_exactly_one_event_type() -> None:
    """One frame this surface originates; every other name on it is a log row's own."""
    assert _declared_types() == {"STREAM_ERROR": "stream.error"}


def test_the_one_type_is_published_under_this_family() -> None:
    assert vocabulary.is_published(stream.STREAM_ERROR)
    assert vocabulary.PUBLISHED[stream.STREAM_ERROR] == stream.FAMILY
    assert stream.FAMILY == "stream"


def test_the_error_frame_carries_the_floor_a_caller_can_resume_from() -> None:
    error = stream.StreamError(
        code=ErrorCode.EVENT_RANGE_EXPIRED,
        message="sequence 3 has expired; this Session retains from 7",
        retained_floor=7,
    )

    assert error.retained_floor == 7
    assert error.code == ErrorCode.EVENT_RANGE_EXPIRED


def test_an_unknown_field_is_refused_rather_than_ignored() -> None:
    """A misspelled field would otherwise be dropped and the frame sent without it."""
    with pytest.raises(ValidationError):
        stream.StreamError(
            code=ErrorCode.EVENT_RANGE_EXPIRED,
            message="expired",
            retained_floor=7,
            retained_flor=7,  # type: ignore[call-arg]
        )


def test_the_frame_cannot_be_rewritten_after_it_is_built() -> None:
    """Frozen because a refusal is a value: the code that says why must not be edited
    between construction and the wire."""
    error = stream.StreamError(
        code=ErrorCode.EVENT_RANGE_EXPIRED, message="expired", retained_floor=7
    )

    with pytest.raises(ValidationError):
        error.code = ErrorCode.INTERNAL


@pytest.mark.parametrize("floor", [0, -1])
def test_a_floor_that_is_not_a_sequence_number_is_refused(floor: int) -> None:
    """Sequences run from 1, so 0 is not a lower floor -- it is an off-by-one that would
    tell a caller it may resume from a position that cannot exist."""
    with pytest.raises(ValidationError):
        stream.StreamError(
            code=ErrorCode.EVENT_RANGE_EXPIRED, message="expired", retained_floor=floor
        )


def test_an_empty_message_is_refused() -> None:
    """The message is the human half of the refusal; an empty one is a refusal that
    explains nothing to whoever is reading the log."""
    with pytest.raises(ValidationError):
        stream.StreamError(
            code=ErrorCode.EVENT_RANGE_EXPIRED, message="", retained_floor=7
        )


def test_a_code_outside_the_published_closed_set_is_refused() -> None:
    """The frame cannot become a second, unversioned place to invent a refusal code."""
    with pytest.raises(ValidationError):
        stream.StreamError(
            code="event_log.gone_missing",  # type: ignore[arg-type]
            message="expired",
            retained_floor=7,
        )
