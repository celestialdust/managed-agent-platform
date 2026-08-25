"""The mechanism that decides what may cross a translated wire, graded on its defaults.

Tier 1, no infrastructure. Every assertion here is about the mechanism rather than about
any wire's knowledge: the tables under test are built in the test body, so a row this
file relies on cannot be quietly changed by the slice that owns a real wire.

The negative assertions carry the weight. What this mechanism is for is the construct
nobody thought about, and the only way to grade that is to hand it a name no table has
and require the failing answer -- a positive test over classified constructs would pass
unchanged on the day the default flipped to permissive.

`_TABLES` is reached into directly by one fixture. The registry is process-global by
design, so a test that installs a table has to put the process back the way it found it;
doing that through a public "unregister" would put a hole in the mechanism for the sake
of the tests that grade it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from managed_agent.core.session.markers import DiscardCause
from managed_agent.gateway.model import classify as classify_module
from managed_agent.gateway.model.classify import (
    Classification,
    Disposition,
    Untranslatable,
    WireNotClassified,
    carry,
    classified,
    classify,
    register_table,
)
from managed_agent.gateway.model.router import UpstreamWire


@pytest.fixture
def scratch() -> Iterator[UpstreamWire]:
    """A wire whose table this test owns, with the global registry restored after.

    `RESPONSES` is the wire no module installs a table for -- it is forwarded rather
    than translated -- so borrowing it cannot collide with a real wire's rows.
    """
    saved = dict(classify_module._TABLES)
    try:
        yield UpstreamWire.RESPONSES
    finally:
        classify_module._TABLES.clear()
        classify_module._TABLES.update(saved)


def test_a_construct_with_no_row_is_classified_as_failing(
    scratch: UpstreamWire,
) -> None:
    register_table(scratch, [Classification("known", Disposition.TRANSLATED, "kept")])

    row = classify(scratch, "nobody.thought.about.this")

    assert row.disposition is Disposition.FAILS
    assert row.cause is DiscardCause.UPSTREAM_UNCLASSIFIED


def test_the_failing_default_is_not_stored_where_a_commit_could_flip_it(
    scratch: UpstreamWire,
) -> None:
    """The unclassified answer is built on demand, so no table can hold a row saying an
    unclassified construct is acceptable."""
    register_table(scratch, [Classification("known", Disposition.TRANSLATED, "kept")])

    assert "nobody.thought.about.this" not in classified(scratch)
    answer = classify(scratch, "nobody.thought.about.this")
    assert answer.disposition is Disposition.FAILS


def test_carry_raises_on_an_unclassified_construct_and_names_it(
    scratch: UpstreamWire,
) -> None:
    register_table(scratch, [Classification("known", Disposition.TRANSLATED, "kept")])

    with pytest.raises(Untranslatable) as caught:
        carry(scratch, "server_tool_use")

    assert caught.value.cause is DiscardCause.UPSTREAM_UNCLASSIFIED
    assert "server_tool_use" in caught.value.detail
    assert scratch.value in caught.value.detail


def test_carry_hands_back_only_the_two_dispositions_a_caller_may_act_on(
    scratch: UpstreamWire,
) -> None:
    register_table(
        scratch,
        [
            Classification("kept", Disposition.TRANSLATED, "has a counterpart"),
            Classification("lost", Disposition.DROPPED, "harmless to lose"),
        ],
    )

    assert carry(scratch, "kept") is Disposition.TRANSLATED
    assert carry(scratch, "lost") is Disposition.DROPPED


def test_a_failing_row_without_a_cause_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="marker cause"):
        Classification("x", Disposition.FAILS, "fails for a reason")


def test_a_carried_row_with_a_cause_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="writes no marker"):
        Classification(
            "x", Disposition.DROPPED, "harmless", DiscardCause.UPSTREAM_TRUNCATED
        )


@pytest.mark.parametrize("why", ["", "   ", "\n"])
def test_a_row_with_no_reason_cannot_be_built(why: str) -> None:
    with pytest.raises(ValueError, match="no reason"):
        Classification("x", Disposition.TRANSLATED, why)


def test_a_wire_cannot_be_given_two_tables(scratch: UpstreamWire) -> None:
    register_table(scratch, [Classification("a", Disposition.DROPPED, "harmless")])

    with pytest.raises(RuntimeError, match="already has a classification table"):
        register_table(scratch, [Classification("b", Disposition.DROPPED, "harmless")])


def test_one_construct_cannot_be_classified_twice(scratch: UpstreamWire) -> None:
    with pytest.raises(ValueError, match="classified twice"):
        register_table(
            scratch,
            [
                Classification("a", Disposition.TRANSLATED, "one reading"),
                Classification("a", Disposition.DROPPED, "another reading"),
            ],
        )


def test_a_wire_with_no_table_raises_rather_than_answering_unknown() -> None:
    """A missing table is a wiring fault in this process, not a stream of upstream
    surprises, and reporting it as the latter sends somebody looking in the wrong
    place."""
    with pytest.raises(WireNotClassified, match=UpstreamWire.CHAT_COMPLETIONS.value):
        classified(UpstreamWire.CHAT_COMPLETIONS)

    with pytest.raises(WireNotClassified):
        classify(UpstreamWire.CHAT_COMPLETIONS, "anything")


def test_an_installed_table_cannot_be_mutated(scratch: UpstreamWire) -> None:
    register_table(scratch, [Classification("a", Disposition.DROPPED, "harmless")])
    table = classified(scratch)
    smuggled = Classification("b", Disposition.TRANSLATED, "smuggled in")

    with pytest.raises(TypeError):
        table["b"] = smuggled  # type: ignore[index]


def test_a_row_that_does_not_fail_cannot_be_raised(scratch: UpstreamWire) -> None:
    with pytest.raises(ValueError, match="does not fail"):
        Untranslatable(scratch, Classification("a", Disposition.DROPPED, "harmless"))


def test_one_occurrence_particulars_ride_beside_the_rows_own_reason(
    scratch: UpstreamWire,
) -> None:
    row = Classification(
        "stream.error",
        Disposition.FAILS,
        "an error frame means the answer stopped",
        DiscardCause.UPSTREAM_REFUSED,
    )

    detail = Untranslatable(scratch, row, note="overloaded_error: try later").detail

    assert "an error frame means the answer stopped" in detail
    assert "overloaded_error: try later" in detail
