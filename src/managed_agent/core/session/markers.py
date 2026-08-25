"""Markers: the record that work already in the Event Log was thrown away.

The log is append-only, so nothing written is removed when the work it describes is
abandoned. What follows the abandoned stretch instead is a marker naming where that
stretch began, a machine-readable cause, and free text beside it -- and a reader's
current view is the raw stream with every marked stretch left out. That raw-stream /
effective-history split is the one the Agent Runtime already uses internally, kept here
rather than reinvented once per consumer (ADR-008).

The cause is closed and the free text is not, and that division is the point. Consumers
branch on the cause whether or not it is committed, so it is committed; nobody branches
on prose, so prose is where one discard's particulars go.

There is deliberately no Budget cause here. The spend gate is checked between Turns and
never interrupts a running one, so a Budget stop has no work to discard and is recorded
as an event of its own rather than as a marker.

This is also not the platform's error set. An error refuses a call a caller just made; a
marker says that work already recorded is no longer current. They are read by different
people at different times, and errors.py owns the former.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.ids import FIRST_SEQ, Seq
from managed_agent.core.ports import EventRecord
from managed_agent.core.vocabulary.marker import WORK_DISCARDED


class Outcome(StrEnum):
    """Whether a discard was a bound doing its job or something going wrong.

    It sits beside the cause because "fault, or bound" is the coarse question every
    reader asks first, and answering that by enumerating causes means each cause added
    later silently joins whichever branch forgot it.
    """

    STOPPED = "stopped"
    FAILED = "failed"


class DiscardCause(StrEnum):
    """Why a stretch of log was discarded. Closed: nothing outside it is writable."""

    POD_LOST = "pod_lost"
    """The pod ended mid-Turn. Synthesized on resume from the log being ahead of the
    rollout, because the process that would have reported it is the one that died."""

    INTERRUPTED = "interrupted"
    """Something outside the Turn asked it to stop before it finished."""

    SUPERSEDED = "superseded"
    """A later Turn took this one's place before it finished."""

    ROLLED_BACK = "rolled_back"
    """An explicit rollback removed the stretch from the effective history."""

    TURN_CEILING_EXCEEDED = "turn_ceiling_exceeded"
    """The Turn reached the platform's own per-Turn token bound and was cut off."""

    UPSTREAM_UNTRANSLATABLE = "upstream_untranslatable"
    """A construct could not be carried faithfully across the Upstream Wire in use."""

    UPSTREAM_UNCLASSIFIED = "upstream_unclassified"
    """The upstream sent a completion signal absent from the classification table."""

    UPSTREAM_REFUSED = "upstream_refused"
    """The upstream declined the request, including for a capability it lacks."""

    UPSTREAM_TRUNCATED = "upstream_truncated"
    """The upstream stopped at a limit of its own, not at the end of the answer."""

    USAGE_UNREPORTED = "usage_unreported"
    """The upstream's stream ended without reporting what it consumed."""


_OUTCOME: Final[Mapping[DiscardCause, Outcome]] = {
    DiscardCause.POD_LOST: Outcome.FAILED,
    DiscardCause.INTERRUPTED: Outcome.STOPPED,
    DiscardCause.SUPERSEDED: Outcome.STOPPED,
    DiscardCause.ROLLED_BACK: Outcome.STOPPED,
    DiscardCause.TURN_CEILING_EXCEEDED: Outcome.STOPPED,
    DiscardCause.UPSTREAM_UNTRANSLATABLE: Outcome.FAILED,
    DiscardCause.UPSTREAM_UNCLASSIFIED: Outcome.FAILED,
    DiscardCause.UPSTREAM_REFUSED: Outcome.FAILED,
    DiscardCause.UPSTREAM_TRUNCATED: Outcome.FAILED,
    DiscardCause.USAGE_UNREPORTED: Outcome.FAILED,
}

_uncovered = set(DiscardCause) - set(_OUTCOME)
if _uncovered:
    raise RuntimeError(
        f"DiscardCause with no Outcome: {sorted(c.value for c in _uncovered)}"
    )


def outcome_of(cause: DiscardCause) -> Outcome:
    """Whether this cause is a bound working or a fault. Total over the closed set."""
    return _OUTCOME[cause]


class WorkDiscarded(BaseModel):
    """A marker's payload: where the discarded stretch began, why, and in words.

    `discarded_from` is explicit rather than inferred from whatever precedes the marker,
    because a marker can land in a page while the stretch it covers began in an earlier
    one -- and a stretch the reader has to guess at is a stretch two readers guess
    differently.

    `detail` is required and non-empty. A marker whose free text is blank names nothing,
    and the naming is the half of this structure a cause cannot carry: which of two
    same-cause discards this one was.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cause: DiscardCause
    detail: str = Field(min_length=1, max_length=2000)
    discarded_from: Seq


def discard(
    cause: DiscardCause, detail: str, discarded_from: Seq
) -> tuple[str, dict[str, object]]:
    """Build the (type, payload) pair to hand to the Event Log's append port.

    One door for writing a marker, so the type string and the payload shape are settled
    here instead of at each of the several components that discard work. The caller does
    not choose where the marker lands; the append path assigns it, so the stretch is
    closed by wherever it lands, and a marker for unappended work cannot be written.

    Parsing happens here rather than at the store. `discarded_from` is a plain `int`
    at this call site, and the annotation on `Seq` is erased -- it validates nothing
    until a pydantic model reads a field of that type. Constructing the model below is
    what turns a bad sequence into a refusal instead of a row.
    """
    payload: dict[str, object] = WorkDiscarded(
        cause=cause, detail=detail, discarded_from=discarded_from
    ).model_dump(mode="json")
    return WORK_DISCARDED, payload


@dataclass(frozen=True, slots=True)
class DiscardedSpan:
    """One marked-off stretch: `first` up to but not including `marker_seq`.

    The marker sits outside the stretch it declares, so it survives its own fold. A
    reader that dropped the marker as well could not tell work that was abandoned from
    work that never happened, and telling those apart is the whole reason the log keeps
    both.
    """

    first: Seq
    marker_seq: Seq
    cause: DiscardCause
    detail: str

    def __post_init__(self) -> None:
        # Both halves are checked here rather than left to the annotations, for the
        # reason `core/ids.py` spells out: `Seq` is a pydantic annotation and this is a
        # dataclass, so `first: Seq` validates nothing on this path. Every instance
        # built today comes from a `WorkDiscarded` whose `discarded_from` pydantic did
        # validate -- but that is a property of the one current caller, not of this
        # type, and `DiscardedSpan(first=0, marker_seq=5)` would otherwise be accepted
        # and fold away a sequence that cannot exist.
        if self.first < FIRST_SEQ:
            raise ValueError(
                f"marker at {self.marker_seq} discards from {self.first}, below"
                f" the first sequence {FIRST_SEQ}; sequences begin at {FIRST_SEQ}"
            )
        if self.first >= self.marker_seq:
            raise ValueError(
                f"marker at {self.marker_seq} discards from {self.first},"
                " which is at or after the marker itself"
            )

    def covers(self, seq: Seq) -> bool:
        return self.first <= seq < self.marker_seq


def discarded_spans(events: Iterable[EventRecord]) -> tuple[DiscardedSpan, ...]:
    """Read the log forward and collect every stretch a marker declares.

    Stretches may overlap and nest -- a rollback discards Turns that carry markers of
    their own -- and nothing here is merged, because a merged stretch loses the cause
    behind each part of it.
    """
    spans: list[DiscardedSpan] = []
    for event in events:
        if event.type != WORK_DISCARDED:
            continue
        payload = WorkDiscarded.model_validate(event.payload)
        spans.append(
            DiscardedSpan(
                first=payload.discarded_from,
                marker_seq=event.seq,
                cause=payload.cause,
                detail=payload.detail,
            )
        )
    return tuple(spans)


def is_discarded(spans: Sequence[DiscardedSpan], seq: Seq) -> bool:
    return any(span.covers(seq) for span in spans)


def effective(events: Sequence[EventRecord]) -> tuple[EventRecord, ...]:
    """The raw stream with every discarded stretch left out.

    Takes a Sequence rather than an iterator on purpose: a marker is read after the work
    it discards, so the events are walked twice -- once to find the stretches, once to
    leave them out. A marker that declares a stretch survives it; a marker that falls
    inside a larger stretch goes with it.

    Markers apply only within the events handed in, and that limit is stated because it
    is silent. A caller that pages gets a partial answer from any page ending before a
    marker: the stretch reads as current, since nothing read so far says otherwise. It
    is inherent to reading a log forward rather than a fault of this function -- but a
    paging caller therefore carries the spans from `discarded_spans` forward across
    pages instead of calling this once per page and trusting the answer.
    """
    spans = discarded_spans(events)
    return tuple(event for event in events if not is_discarded(spans, event.seq))
