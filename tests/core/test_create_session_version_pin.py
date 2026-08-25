"""What `CreateSession` accepts as a version pin, and what it refuses at the boundary.

A file of its own beside `test_session.py`, because every case here is about one field
and the reason it is annotated the way it is.

The case worth the file is `true`. `bool` is a subclass of `int`, so a plain `int`
annotation accepts `true` and stores `1` -- measured on pydantic 2.13.4, and `ge=1` does
not refuse it either, because 1 is greater than or equal to 1. A caller who sent `true`
would be pinned to revision 1 having written no pin, and would go on running revision 1
through every later edit with no error anywhere. Nothing downstream could detect it: by
the time the handler sees the value it is an ordinary `1`.

So these are not tests of pydantic. They are the record of why the annotation is
`StrictInt` and not `int`, in a form that fails if somebody simplifies it.
"""

import pytest
from pydantic import ValidationError

from managed_agent.core.session.session import CreateSession


def _body(**overrides: object) -> dict[str, object]:
    return {
        "definition_id": "3b1f0c8a-0000-4000-8000-000000000001",
        # Required, and about a different axis than the version pin these cases grade:
        # a Session names the sandbox shape it runs in, and there is no default one.
        "environment_id": "3b1f0c8a-0000-4000-8000-000000000002",
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 30,
    } | overrides


def test_no_version_and_an_explicit_null_both_mean_the_current_revision() -> None:
    """Two spellings of "I have no opinion", and neither is revision 1."""
    assert CreateSession.model_validate(_body()).definition_version is None
    assert (
        CreateSession.model_validate(_body(definition_version=None)).definition_version
        is None
    )


def test_a_version_number_is_kept_exactly() -> None:
    assert (
        CreateSession.model_validate(_body(definition_version=2)).definition_version
        == 2
    )


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (True, "bool is a subclass of int: a plain `int` field stores this as 1"),
        ("2", "a numeric string would be coerced by a non-strict field"),
        (1.0, "a float that compares equal to 1 is not a revision number"),
    ],
)
def test_a_value_that_is_not_an_integer_is_refused_rather_than_coerced(
    value: object, why: str
) -> None:
    with pytest.raises(ValidationError) as refused:
        CreateSession.model_validate(_body(definition_version=value))

    assert refused.value.errors()[0]["type"] == "int_type", why


@pytest.mark.parametrize("value", [0, -1])
def test_a_revision_below_one_is_refused(value: int) -> None:
    """Revisions are numbered from 1, so 0 is not an older version -- it is a bug."""
    with pytest.raises(ValidationError) as refused:
        CreateSession.model_validate(_body(definition_version=value))

    assert refused.value.errors()[0]["type"] == "greater_than_equal"


def test_a_misspelled_version_field_is_named_rather_than_read_as_unpinned() -> None:
    """`extra="forbid"` is what turns a typo into an error instead of a silent default.

    Without it a caller who wrote `versoin` would be told nothing while the platform
    ran the newest revision -- the exact behaviour they were trying to avoid.
    """
    with pytest.raises(ValidationError) as refused:
        CreateSession.model_validate(_body(versoin=2))

    assert refused.value.errors()[0]["type"] == "extra_forbidden"
