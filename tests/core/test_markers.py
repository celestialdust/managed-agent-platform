"""Discarded work stays in the log, and a marker after it is what makes it discarded.

Tier 1 (local, no infrastructure). Everything graded here is pure, so a database would
only slow it down; the store's own guarantees are graded in `tests/adapters/`.

Three things are load-bearing and each is asserted so that removing the behaviour breaks
the assertion. The cause set is closed, so a consumer branching on it cannot be handed a
value the platform never published. The marker sits outside the stretch it declares, so
it survives the fold it causes -- which is the difference between work that was thrown
away and work that never happened. And the effective history really is computed rather
than stored: the last test here reads one log two ways and gets two different Session
states out of it, which cannot pass if the marker is ignored.
"""

import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from managed_agent.core import vocabulary
from managed_agent.core.ids import Seq, SessionId
from managed_agent.core.ports import EventRecord
from managed_agent.core.session.markers import (
    DiscardCause,
    DiscardedSpan,
    Outcome,
    WorkDiscarded,
    discard,
    discarded_spans,
    effective,
    is_discarded,
    outcome_of,
)
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import SessionState
from managed_agent.core.vocabulary import marker
from managed_agent.core.vocabulary.marker import FAMILY, WORK_DISCARDED

_SESSION = SessionId(uuid4())


@dataclass(frozen=True, slots=True)
class Event:
    """A stand-in for a stored event, carrying only what these readers look at.

    Hand-rolled rather than the Postgres adapter's row: every function under test is
    pure, and a fixture that reached for a database would be asserting the adapter.
    """

    seq: Seq
    type: str
    session_id: SessionId = _SESSION
    payload: dict[str, object] = field(default_factory=dict)


def _event(seq: int, type_: str) -> Event:
    return Event(seq=Seq(seq), type=type_)


def _marker(
    seq: int,
    discarded_from: int,
    cause: DiscardCause = DiscardCause.INTERRUPTED,
    detail: str = "the caller interrupted the turn",
) -> Event:
    """A marker event built through `discard`, so the tests exercise the real door.

    Constructing the payload by hand here would let the tests keep passing after
    `discard` started producing a shape no reader accepts.
    """
    type_, payload = discard(cause, detail, Seq(discarded_from))
    return Event(seq=Seq(seq), type=type_, payload=payload)


def _seqs(events: Sequence[EventRecord]) -> list[int]:
    return [event.seq for event in events]


# --- the family is published, and the registry found it without being told ------------


def test_the_marker_type_is_published_under_the_marker_family() -> None:
    assert vocabulary.is_published(WORK_DISCARDED)
    assert vocabulary.PUBLISHED[WORK_DISCARDED] == FAMILY
    assert FAMILY == "marker"


def test_importing_the_registry_alone_publishes_the_marker() -> None:
    """A fresh interpreter that imports only the registry still knows the type.

    Run out of process on purpose: inside this test session `markers.py` has already
    been imported by the module header above, so an in-process check could not tell
    discovery from that import having done the work.
    """
    probe = (
        "import sys\n"
        "from managed_agent.core import vocabulary\n"
        "assert 'managed_agent.core.session.markers' not in sys.modules, 'dragged in'\n"
        "print(vocabulary.PUBLISHED['marker.work_discarded'])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "marker"


def test_the_registry_names_none_of_the_families_it_discovers() -> None:
    """Adding a family is a new file, never an edit here -- so the edit is impossible.

    Asserted over every family rather than this one, because the property is about the
    registry and a check that only mentioned `marker` would go stale on the next family.

    Matched on word boundaries rather than as a bare substring. A plain `in` test reads
    the family `turn` out of the word "Returns", which the registry's own docstring
    contains -- a false failure that says the registry hand-names a family it does not
    mention at all. The boundaries lose nothing real: a family named by hand appears as
    `"turn"`, as `.turn` inside an import path, or after `== `, and a non-word character
    sits on each side of it in every one of those.
    """
    families = set(vocabulary.PUBLISHED.values())
    assert FAMILY in families, "discovery did not run; the check below would be vacuous"

    registry = Path(vocabulary.__file__).read_text(encoding="utf-8")
    named = {
        family
        for family in families
        if re.search(rf"\b{re.escape(family)}\b", registry)
    }
    assert named == set(), f"the registry names families by hand: {sorted(named)}"


def test_every_type_the_marker_family_declares_is_one_the_fold_honours() -> None:
    """A published marker type no reader acts on is a stretch nobody ever leaves out.

    The family module and the reader are two files carrying one piece of knowledge
    between them, which is drift by construction -- the same shape
    `tests/core/test_vocabulary_lifecycle.py` guards for the lifecycle family, where a
    transition table read two types no module could publish. A second marker type
    declared without teaching `discarded_spans` about it fails here, in the slice that
    adds it, rather than reading as a feature nothing implements.
    """
    declared = {
        value
        for name, value in vars(marker).items()
        if name.isupper() and name != "FAMILY" and isinstance(value, str)
    }
    assert declared, "the marker family declares no event types"
    assert declared == {WORK_DISCARDED}, (
        f"declared but not honoured by the fold: {sorted(declared - {WORK_DISCARDED})}"
    )


# --- the cause set is closed, and the coarse outcome beside it ------------------------


def test_every_cause_has_an_outcome() -> None:
    """Total over the closed set: a cause added without an outcome is a KeyError.

    The non-empty assertion is what stops this certifying nothing: every claim below
    lives inside the loop, so an empty set would pass it without executing a line.
    """
    assert list(DiscardCause), "no causes to check"
    for cause in DiscardCause:
        assert isinstance(outcome_of(cause), Outcome)


def test_a_fault_and_a_bound_working_are_told_apart() -> None:
    """The split is asserted by name, so moving a cause between them fails here."""
    failed = {cause for cause in DiscardCause if outcome_of(cause) is Outcome.FAILED}
    stopped = {cause for cause in DiscardCause if outcome_of(cause) is Outcome.STOPPED}
    assert failed == {
        DiscardCause.POD_LOST,
        DiscardCause.UPSTREAM_UNTRANSLATABLE,
        DiscardCause.UPSTREAM_UNCLASSIFIED,
        DiscardCause.UPSTREAM_REFUSED,
        DiscardCause.UPSTREAM_TRUNCATED,
        DiscardCause.USAGE_UNREPORTED,
    }
    assert stopped == {
        DiscardCause.INTERRUPTED,
        DiscardCause.SUPERSEDED,
        DiscardCause.ROLLED_BACK,
        DiscardCause.TURN_CEILING_EXCEEDED,
    }


def test_a_cause_outside_the_set_is_refused() -> None:
    with pytest.raises(ValueError):
        DiscardCause("budget_exhausted")


def test_no_cause_names_a_budget() -> None:
    """The spend gate is checked between Turns, so a Budget stop discards nothing.

    A Budget cause here would invite a caller to write a marker covering a stretch that
    was never abandoned -- the Turn it followed ran to completion.
    """
    assert list(DiscardCause), "no causes to check"
    assert [cause for cause in DiscardCause if "budget" in cause.value] == []


# --- the payload, and the one door for writing it -------------------------------------


def test_discard_returns_the_published_type_and_a_reparsable_payload() -> None:
    type_, payload = discard(DiscardCause.POD_LOST, "node drained", Seq(7))
    assert type_ == WORK_DISCARDED
    assert vocabulary.is_published(type_)
    reparsed = WorkDiscarded.model_validate(payload)
    assert reparsed.cause is DiscardCause.POD_LOST
    assert reparsed.detail == "node drained"
    assert reparsed.discarded_from == 7


def test_the_payload_survives_a_round_trip_through_json() -> None:
    """The column is jsonb, so an unserialized enum fails at the store."""
    _, payload = discard(DiscardCause.SUPERSEDED, "a later turn replaced it", Seq(4))
    assert (
        WorkDiscarded.model_validate(json.loads(json.dumps(payload))).discarded_from
        == 4
    )


def test_an_empty_detail_is_refused() -> None:
    """A marker whose free text is blank names nothing, which is the half a cause cannot
    carry: which of two same-cause discards this one was."""
    with pytest.raises(ValidationError):
        discard(DiscardCause.INTERRUPTED, "", Seq(2))


def test_a_detail_past_the_cap_is_refused() -> None:
    with pytest.raises(ValidationError):
        discard(DiscardCause.INTERRUPTED, "x" * 2001, Seq(2))


def test_an_unknown_cause_string_is_refused() -> None:
    with pytest.raises(ValidationError):
        WorkDiscarded.model_validate(
            {"cause": "budget_exhausted", "detail": "why", "discarded_from": 2}
        )


def test_an_extra_payload_key_is_refused() -> None:
    """A misspelled field would otherwise be dropped, and the marker would read as
    though the writer never set it."""
    with pytest.raises(ValidationError):
        WorkDiscarded.model_validate(
            {
                "cause": DiscardCause.INTERRUPTED.value,
                "detail": "why",
                "discarded_from": 2,
                "discarded_to": 5,
            }
        )


def test_a_discarded_from_of_zero_is_refused() -> None:
    """0 is not a smaller sequence, it is an index/count confusion."""
    with pytest.raises(ValidationError):
        discard(DiscardCause.INTERRUPTED, "why", Seq(0))


def test_a_boolean_is_not_a_sequence() -> None:
    """`True` equals 1 and is not a sequence; strict parsing is what refuses it."""
    with pytest.raises(ValidationError):
        WorkDiscarded.model_validate(
            {
                "cause": DiscardCause.INTERRUPTED.value,
                "detail": "why",
                "discarded_from": True,
            }
        )


def test_the_payload_cannot_be_rewritten() -> None:
    marker = WorkDiscarded(
        cause=DiscardCause.POD_LOST, detail="drained", discarded_from=3
    )
    with pytest.raises(ValidationError):
        marker.detail = "something else"


# --- the span a marker declares, with the marker itself outside it -------------------


def test_a_span_covers_its_first_and_excludes_its_marker() -> None:
    span = DiscardedSpan(
        first=Seq(3), marker_seq=Seq(6), cause=DiscardCause.INTERRUPTED, detail="why"
    )
    assert [seq for seq in range(1, 8) if span.covers(Seq(seq))] == [3, 4, 5]


def test_a_span_that_starts_at_its_own_marker_is_refused() -> None:
    """An empty stretch is a marker declaring nothing, which no reader can act on."""
    with pytest.raises(ValueError, match="at or after"):
        DiscardedSpan(
            first=Seq(6),
            marker_seq=Seq(6),
            cause=DiscardCause.INTERRUPTED,
            detail="why",
        )


def test_a_span_that_starts_after_its_own_marker_is_refused() -> None:
    with pytest.raises(ValueError, match="at or after"):
        DiscardedSpan(
            first=Seq(7),
            marker_seq=Seq(6),
            cause=DiscardCause.INTERRUPTED,
            detail="why",
        )


def test_a_span_cannot_be_rewritten() -> None:
    span = DiscardedSpan(
        first=Seq(3), marker_seq=Seq(6), cause=DiscardCause.INTERRUPTED, detail="why"
    )
    with pytest.raises(AttributeError):
        span.first = Seq(1)  # type: ignore[misc]


def test_is_discarded_asks_every_span() -> None:
    spans = (
        DiscardedSpan(
            first=Seq(2), marker_seq=Seq(4), cause=DiscardCause.INTERRUPTED, detail="a"
        ),
        DiscardedSpan(
            first=Seq(7), marker_seq=Seq(9), cause=DiscardCause.SUPERSEDED, detail="b"
        ),
    )
    assert [seq for seq in range(1, 11) if is_discarded(spans, Seq(seq))] == [
        2,
        3,
        7,
        8,
    ]


# --- the forward fold, which is what reports the work as discarded --------------------


def test_the_discarded_stretch_is_left_out_and_the_marker_survives() -> None:
    log = [
        _event(1, "turn.started"),
        _event(2, "turn.output"),
        _event(3, "turn.output"),
        _event(4, "turn.output"),
        _event(5, "turn.output"),
        _marker(6, discarded_from=3),
    ]
    assert _seqs(effective(log)) == [1, 2, 6]


def test_reading_the_log_leaves_the_discarded_work_in_it() -> None:
    """The other half of append-only: the stretch is left out of the view, not the log.

    A reader that could not still see 3, 4 and 5 could not tell work that was abandoned
    from work that never happened, and telling those apart is why the log keeps both.
    """
    log = [
        _event(1, "turn.started"),
        _event(2, "turn.output"),
        _event(3, "turn.output"),
        _event(4, "turn.output"),
        _event(5, "turn.output"),
        _marker(6, discarded_from=3),
    ]
    before = list(log)
    effective(log)
    assert log == before
    assert _seqs(log) == [1, 2, 3, 4, 5, 6]


def test_a_page_starting_inside_the_stretch_still_leaves_it_out() -> None:
    """The stretch is read off the payload, not off what precedes the marker.

    A caller paging by sequence can be handed a window whose first event is already
    inside a discarded stretch; adjacency would tell it nothing, and this is why
    `discarded_from` is carried explicitly.
    """
    page = [_event(4, "turn.output"), _event(5, "turn.output"), _marker(6, 3)]
    assert _seqs(effective(page)) == [6]


def test_a_stretch_whose_marker_is_not_in_the_page_still_reads_as_current() -> None:
    """The converse of the case above, and the reason the limit is in the docstring.

    Nothing in a log says work was discarded until the marker saying so is read, so a
    page ending before the marker cannot know. That is inherent to reading forward, not
    a defect -- it is pinned because the consequence is silent: the page comes back
    looking exactly like current work, with no error and no short result to notice. A
    caller that pages carries `discarded_spans` forward rather than calling `effective`
    per page.
    """
    page = [_event(1, "turn.started"), _event(2, "turn.output")]
    assert effective(page) == tuple(page)

    whole = [*page, _marker(3, discarded_from=1)]
    assert _seqs(effective(whole)) == [3]


def test_a_later_marker_swallows_an_earlier_marker_and_its_stretch() -> None:
    """Stretches nest: a rollback discards Turns that carry markers of their own."""
    log = [
        _event(1, "turn.started"),
        _event(2, "turn.output"),
        _event(3, "turn.output"),
        _event(4, "turn.output"),
        _event(5, "turn.output"),
        _marker(6, discarded_from=3),
        _event(7, "turn.started"),
        _event(8, "turn.output"),
        _marker(
            9, discarded_from=2, cause=DiscardCause.ROLLED_BACK, detail="rolled back"
        ),
    ]
    assert _seqs(effective(log)) == [1, 9]


def test_a_log_with_no_marker_comes_back_unchanged() -> None:
    log = [_event(1, "session.created"), _event(2, "turn.started")]
    assert effective(log) == tuple(log)


def test_two_stretches_keep_their_own_causes_rather_than_being_merged() -> None:
    """A merged stretch loses the cause behind each part of it."""
    log = [
        _event(1, "turn.started"),
        _marker(2, discarded_from=1, cause=DiscardCause.INTERRUPTED, detail="first"),
        _event(3, "turn.started"),
        _marker(4, discarded_from=3, cause=DiscardCause.POD_LOST, detail="second"),
    ]
    spans = discarded_spans(log)
    assert [
        (span.first, span.marker_seq, span.cause, span.detail) for span in spans
    ] == [
        (1, 2, DiscardCause.INTERRUPTED, "first"),
        (3, 4, DiscardCause.POD_LOST, "second"),
    ]


def test_a_marker_moves_the_state_fold_nowhere() -> None:
    """`project` has no case for a marker, so it advances the sequence and nothing else.

    That totality is what lets this slice add an event type without editing the fold,
    and it is exactly what a later diff could break -- so it is asserted by comparing
    against the same log with the marker events taken out.
    """
    log = [
        _event(1, "session.created"),
        _event(2, "turn.started"),
        _event(3, "turn.output"),
        _marker(4, discarded_from=2),
        _event(5, "session.suspended"),
    ]
    without = [event for event in log if event.type != WORK_DISCARDED]
    assert project(log) == project(without) == (SessionState.SUSPENDED, 5)


def test_the_forward_read_reports_discarded_work_as_discarded_not_current() -> None:
    """One log, read two ways, gives two different Session states.

    This is the checkpoint itself. The raw stream says the Session is suspended, because
    the suspension is written there and nothing is ever removed. The effective history
    says it is running, because the marker after the suspension declares it discarded.
    A reader that ignored the marker would get the same answer twice, so this is the
    assertion that cannot pass unless the marker is honoured.
    """
    log = [
        _event(1, "session.created"),
        _event(2, "session.suspended"),
        _marker(3, discarded_from=2, cause=DiscardCause.ROLLED_BACK, detail="undone"),
    ]
    raw_state, raw_last = project(log)
    live_state, live_last = project(effective(log))

    assert raw_state is SessionState.SUSPENDED
    assert live_state is SessionState.RUNNING
    assert raw_last == live_last == 3


def test_a_span_cannot_start_below_the_first_sequence() -> None:
    """`DiscardedSpan` refuses a `first` below 1, and refuses it here not upstream.

    The annotation on `first` is `Seq`, which is a pydantic annotation sitting on a
    dataclass field -- so it validates nothing on this path, and the type reads as a
    guarantee it does not give. Every instance built today comes from a `WorkDiscarded`
    whose `discarded_from` pydantic really did validate, but that is a fact about the
    one current caller and not about this type. A second caller, or a test fixture,
    would be accepted while folding away a sequence that cannot exist.
    """
    with pytest.raises(ValueError, match="below the first sequence"):
        DiscardedSpan(
            first=Seq(0), marker_seq=Seq(5), cause=DiscardCause.POD_LOST, detail=""
        )

    # The ordering check is separate and still fires: neither guard shadows the other.
    with pytest.raises(ValueError, match="at or after the marker itself"):
        DiscardedSpan(
            first=Seq(5), marker_seq=Seq(5), cause=DiscardCause.POD_LOST, detail=""
        )
