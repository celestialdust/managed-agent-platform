"""The sequence type refuses the values that would become a gap.

Tier 1 (local, no infrastructure). These assertions are about the type alone: a
Session's sequence is contiguous from 1, so 0, a negative, a float and a bool are each a
parse error at the boundary rather than a row in the store.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from managed_agent.core.ids import (
    FIRST_SEQ,
    Seq,
    new_session_id,
    new_turn_id,
)

_seq = TypeAdapter(Seq)


def test_seq_accepts_one() -> None:
    assert _seq.validate_python(1) == 1


def test_first_seq_is_one() -> None:
    assert FIRST_SEQ == 1
    assert _seq.validate_python(FIRST_SEQ) == 1


@pytest.mark.parametrize("bad", [0, -1])
def test_seq_rejects_below_one(bad: int) -> None:
    with pytest.raises(ValidationError):
        _seq.validate_python(bad)


def test_seq_rejects_bool_under_strict() -> None:
    """True is 1 to Python and a bug to us: a flag where a sequence belongs."""
    with pytest.raises(ValidationError):
        _seq.validate_python(True)


def test_seq_rejects_float_under_strict() -> None:
    with pytest.raises(ValidationError):
        _seq.validate_python(1.0)


def test_new_ids_are_distinct() -> None:
    assert new_session_id() != new_session_id()
    assert new_turn_id() != new_turn_id()
