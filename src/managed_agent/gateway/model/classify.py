"""What may cross a translated Upstream Wire, decided once and read at run time.

Every construct either side of a translated wire can carry is classified by one
question: would dropping or flattening it let the Agent Runtime believe a Turn ended
normally when it did not? A construct that fails that question fails the Turn and is
named in a marker; one that passes is translated, or dropped with nothing recorded at
all. A construct nobody classified is treated as though it had failed the question,
because the permissive outcome is the one that writes a falsehood into an append-only
log, and that outcome has to be chosen deliberately rather than arrived at by omission
(ADR-009).

The mechanism is here and the per-wire knowledge is not. Each wire module owns its own
table and installs it when it is imported, so a wire is added by writing a file rather
than by adding a branch to this one. A wire whose table was never installed raises
instead of classifying everything as unknown: that is a wiring fault in this process,
and reporting it as a stream of upstream surprises would send somebody looking in the
wrong place.

`UpstreamWire` is imported rather than declared. The set of shapes this service speaks
is one closed set and it already lives beside the Routing Entry that names a member of
it; a second spelling here would be two enums for one decision, and the day they
disagreed would be the day a table was installed under a member no entry can name.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from managed_agent.core.session.markers import DiscardCause
from managed_agent.gateway.model.router import UpstreamWire


class Disposition(StrEnum):
    """What becomes of a construct that reaches a translator."""

    TRANSLATED = "translated"
    DROPPED = "dropped"
    FAILS = "fails"


@dataclass(frozen=True, slots=True)
class Classification:
    """One construct's settled disposition, and the reason it is that one.

    `cause` is present exactly when the disposition is FAILS. A failing construct has to
    hand a marker a cause from the closed set, and a construct that is carried has no
    marker to write -- pairing the two in one type is what stops either half being
    forgotten.

    `why` is prose, for whoever reads a marker or reviews this table later, and it is
    required. A row with no reason cannot be re-derived when the wire moves under it,
    and a row nobody can re-derive is a row nobody dares change.
    """

    construct: str
    disposition: Disposition
    why: str
    cause: DiscardCause | None = None

    def __post_init__(self) -> None:
        fails = self.disposition is Disposition.FAILS
        if fails and self.cause is None:
            raise ValueError(
                f"{self.construct}: a failing construct needs a marker cause"
            )
        if not fails and self.cause is not None:
            raise ValueError(f"{self.construct}: a carried construct writes no marker")
        if not self.why.strip():
            raise ValueError(f"{self.construct}: classified with no reason")


class WireNotClassified(RuntimeError):
    """No table was installed for this wire: a wiring fault, not upstream data."""


class Untranslatable(Exception):
    """A construct that cannot cross this wire faithfully.

    Carries the closed-set cause a marker takes and the free text that names the
    construct, so the Turn's failure path writes the marker out of this and invents no
    wording of its own. `note` is where one occurrence's particulars go -- an upstream
    error's own message, say -- since the row's `why` is the same for every occurrence
    of that construct.

    `detail` names the wire, which makes it operator-facing rather than caller-facing.
    Which shape a model is served over is this service's configuration, so a caller that
    learns it learns something about the platform it did not send; a refusal built from
    one of these has to take the construct and the reason and leave the prefix behind.
    """

    def __init__(
        self, wire: UpstreamWire, classification: Classification, note: str = ""
    ) -> None:
        if classification.cause is None:
            raise ValueError(
                f"{classification.construct} does not fail; it cannot be raised"
            )
        self.wire = wire
        self.classification = classification
        self.cause: DiscardCause = classification.cause
        detail = f"{wire.value}: {classification.construct}: {classification.why}"
        self.detail = f"{detail} ({note})" if note else detail
        super().__init__(self.detail)


_TABLES: dict[UpstreamWire, Mapping[str, Classification]] = {}
"""Every installed wire table, keyed by the wire that installed it.

Process-global and written exactly once per wire, while that wire's module is imported.
It is mutable state, which is why the only writer refuses a second write: a table that
could be replaced at run time is a table whose contents depend on import order, and the
contents are what decides whether a Turn is allowed to look finished.
"""


def register_table(wire: UpstreamWire, rows: Iterable[Classification]) -> None:
    """Install one wire's table. Called once, while that wire's module is imported.

    Two refusals, and both are about a table meaning one thing. A second table for a
    wire already registered would silently replace knowledge somebody is reading, and
    two rows for one construct would leave the answer to whichever the loop saw last.
    """
    if wire in _TABLES:
        raise RuntimeError(f"{wire.value} already has a classification table")
    table: dict[str, Classification] = {}
    for row in rows:
        if row.construct in table:
            raise ValueError(f"{wire.value}: {row.construct} classified twice")
        table[row.construct] = row
    _TABLES[wire] = MappingProxyType(table)


def classified(wire: UpstreamWire) -> Mapping[str, Classification]:
    """Every construct this wire has a row for. The exhaustiveness tests read this."""
    table = _TABLES.get(wire)
    if table is None:
        raise WireNotClassified(f"no classification table installed for {wire.value}")
    return table


def classify(wire: UpstreamWire, construct: str) -> Classification:
    """The disposition of one construct on one wire.

    A construct with no row gets a failing classification built here rather than stored,
    so no table can contain a row that says an unclassified construct is acceptable.
    """
    row = classified(wire).get(construct)
    if row is not None:
        return row
    return Classification(
        construct=construct,
        disposition=Disposition.FAILS,
        why="no classification row exists for this construct on this wire",
        cause=DiscardCause.UPSTREAM_UNCLASSIFIED,
    )


def carry(wire: UpstreamWire, construct: str) -> Disposition:
    """Classify, and raise on anything that fails.

    The one door a translator goes through. A call site never sees a FAILS row, so it
    cannot read one and carry on: what comes back is TRANSLATED or DROPPED, or nothing
    comes back.
    """
    row = classify(wire, construct)
    if row.disposition is Disposition.FAILS:
        raise Untranslatable(wire, row)
    return row.disposition
