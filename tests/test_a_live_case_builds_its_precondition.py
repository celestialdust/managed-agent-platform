"""No live case skips itself. A precondition is built, not branched on.

`tests/pod/` runs against a real cluster, and every case in it is gated by a
`requires_the_cluster` skipif on `MAP_CLUSTER_TESTS` -- defined in two of the files and
imported by the rest. That gate is the one honest skip in this tree: the cluster either
is reachable or is not, and no fixture can build one.

Every OTHER skip in that directory is a precondition the fixture could have created. One
was written -- a backward-paging case skipped when its tenant held a single Session, in
a tenant the same fixture creates -- and it skipped on every run that would ever be
made, reporting `7 passed, 1 skipped` three times before anybody asked which one. A skip
whose condition the test controls is not a conditional; it is a case that has been
deleted and still shows up in the count.

So the rule is mechanical rather than a matter of judgement, because judgement is what
produced the case above: `tests/pod/` contains no `pytest.skip` call. A live case that
wants two Sessions creates the second one.
"""

from pathlib import Path
from typing import Final

_LIVE_SUITE: Final = Path(__file__).parent / "pod"
_FORBIDDEN: Final = "pytest.skip("
_GATE: Final = "requires_the_cluster"


def _cases_that_skip_themselves(directory: Path) -> list[str]:
    """Every line under `directory` holding a `pytest.skip` call, as `file:line`.

    A string match on the source rather than a runtime count of skips, because the
    defect is invisible at runtime: a case that always skips and a case that skips today
    produce the same green line, and the one that always skips is the one worth
    catching. The shared gate is spelled `pytest.mark.skipif` and does not match.

    Taking the directory as an argument is what lets the control below plant a skip and
    watch this flag it. A hard-coded path would leave the matcher itself ungraded."""
    return sorted(
        f"{path.name}:{number}"
        for path in directory.glob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if _FORBIDDEN in line
    )


def test_no_live_case_skips_itself_on_a_precondition_it_could_build() -> None:
    """No `pytest.skip` call anywhere under `tests/pod/`.

    The message names the two ways out, since a reader who has just tripped this wants
    to know which of them applies rather than that a rule exists."""
    offenders = _cases_that_skip_themselves(_LIVE_SUITE)
    assert offenders == [], (
        "these live cases skip themselves; build the precondition instead, or if the "
        f"skip is genuinely about the cluster use the shared gate: {offenders}"
    )


def test_the_scan_flags_a_planted_skip(tmp_path: Path) -> None:
    """The control: a planted skip is found, and a clean file beside it is not.

    Both halves, because either alone is satisfied by a broken matcher -- one that flags
    everything passes the first, and one that flags nothing passes an assertion that
    only counts. The line number is asserted too, so a scan that reported the right file
    for the wrong reason is not read as agreement.

    This is what the earlier version of this file got wrong. It controlled for vacuity
    by taking an inventory of the live directory, which grades whether the files exist
    and never touches the question of whether the search works."""
    (tmp_path / "test_planted.py").write_text(
        "import pytest\n\n\ndef test_one() -> None:\n"
        '    pytest.skip("a precondition this case could have built")\n'
    )
    (tmp_path / "test_clean.py").write_text(
        "def test_two() -> None:\n    assert True\n"
    )

    assert _cases_that_skip_themselves(tmp_path) == ["test_planted.py:5"]


def test_the_scan_is_pointed_at_the_live_suite() -> None:
    """The directory being scanned is the gated live suite and not some other one.

    Without this the rule above passes loudest on the day somebody moves `tests/` and
    the glob starts matching nothing. A floor rather than an exact count because live
    files get added, and the gate name rather than a file list because a file list is a
    second copy of the directory that is free to disagree with it.

    Counts only the files that carry the gate. Two files in there are offline by design
    -- one reads a manifest, one is a container build behind `pytest.mark.image` -- and
    an assertion that every file is gated would be false the moment a third arrives."""
    gated = [
        path.name for path in _LIVE_SUITE.glob("test_*.py") if _GATE in path.read_text()
    ]
    assert len(gated) >= 8, gated
