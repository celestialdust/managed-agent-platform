"""Guard on the spike record: a filled-in verdict and no blank measurement.

This record is the answer to the question the rest of the plan is forbidden to
assume, and later slices read numbers straight out of its Measured table. A blank
cell there is worse than a missing file: the file's absence stops a slice, and a
blank cell is read as a value and quietly propagates as one.

So two properties are enforced. The Verdict must be one of exactly three strings,
because a verdict written as prose can be read either way by whoever needs it to
come out their way. And every Measured row must carry both a value and the
command or probe section it came from, because a number with no provenance cannot
be re-derived when somebody doubts it a month from now.

The template's own instructional comments are stripped before any of this is
checked — the point is to fail on a table nobody filled in, not on the template
that was correctly filled in over.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RECORD = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "features"
    / "managed-agent-platform"
    / "spike-bubblewrap.md"
)

PERMITTED_VERDICTS = (
    "boundary holds",
    "boundary does not hold",
    "holds with the conditions below",
)


def _strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _section(text: str, heading: str) -> str:
    """Return the body under `## heading`, up to the next `## ` heading.

    Sliced on the heading rather than parsed as markdown because the only
    structure this needs is "which lines belong to this section", and a parser
    would be a dependency bought for one question.
    """
    pattern = rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)"
    found = re.search(pattern, text, flags=re.M | re.S)
    if found is None:
        raise AssertionError(f"the record has no '## {heading}' section")
    return found.group(1)


def _table_rows(section_text: str) -> list[list[str]]:
    """Return the data rows of the first pipe table in a section, cells trimmed.

    Skips the header row and the `|---|` separator, and drops the empty strings
    that split() produces either side of the leading and trailing pipes.
    """
    rows: list[list[str]] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= {"-", ":"} and c for c in cells):
            continue
        rows.append(cells)
    return rows[1:]


@pytest.fixture(scope="module")
def record() -> str:
    if not RECORD.exists():
        pytest.fail(f"the spike record does not exist at {RECORD}")
    return _strip_html_comments(RECORD.read_text())


def test_the_verdict_is_one_of_the_three_permitted_strings(record: str) -> None:
    verdict = _section(record, "Verdict").strip()
    assert verdict, "the Verdict section is empty"
    assert verdict.splitlines()[0].strip() in PERMITTED_VERDICTS, (
        f"the verdict reads {verdict.splitlines()[0].strip()!r}; it must be exactly "
        f"one of {PERMITTED_VERDICTS} so it cannot be read two ways"
    )


def test_no_measured_cell_is_blank(record: str) -> None:
    rows = _table_rows(_section(record, "Measured"))
    assert rows, "the Measured table has no rows"
    blank = [row[0] for row in rows if any(not cell for cell in row)]
    assert not blank, (
        f"these Measured rows have a blank cell: {blank}. A later slice reads these "
        "as values, so a blank one propagates as a number nobody measured."
    )


def test_every_measured_row_names_where_the_number_came_from(record: str) -> None:
    """A quantity with no provenance cannot be re-derived when it is doubted.

    Checked as a column width rather than a content match: the record is prose and
    the phrasing is the author's, but a row that has dropped the 'How' column
    entirely has dropped the only thing that makes the number checkable.
    """
    rows = _table_rows(_section(record, "Measured"))
    short = [row[0] for row in rows if len(row) < 3]
    assert not short, f"these Measured rows carry no 'How' cell: {short}"
