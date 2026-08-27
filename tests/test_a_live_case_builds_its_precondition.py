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

import ast
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


def _targets_of(node: ast.AnnAssign | ast.Assign) -> list[ast.expr]:
    """What one assignment binds, annotated or not."""
    return [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)


def _grants_naming_a_bare_tool(directory: Path) -> list[str]:
    """Every Grant under `directory` that names a registered tool's bare name.

    A tool's identity is the pair `(server, tool)`, joined into one advertised name
    before the model or the Grant ever sees it. So a Grant element that is a string
    literal, or a reference to a module-level string constant -- which is how a bare
    `_TOOL: Final = "ask_deepwiki"` reaches one -- names a tool nothing resolves. A name
    built at run time from the server that was just registered does not match, and that
    is the only shape that can be right.

    Nothing refuses such a Grant: the create route stores it unread, the Session starts
    with no tools, and the first sign is the model reporting mid-Turn that a tool is not
    available -- after a pod has been placed and paid for.
    """
    offenders: list[str] = []
    for path in sorted(directory.glob("*.py")):
        module = ast.parse(path.read_text())
        constants = {
            target.id
            for node in module.body
            if isinstance(node, ast.AnnAssign | ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for target in _targets_of(node)
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(module):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if not (isinstance(key, ast.Constant) and key.value == "grant"):
                    continue
                if not isinstance(value, ast.List):
                    continue
                for element in value.elts:
                    bare = (
                        isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    ) or (isinstance(element, ast.Name) and element.id in constants)
                    if bare:
                        offenders.append(f"{path.name}:{element.lineno}")
    return offenders


def test_no_live_case_grants_a_tool_by_its_bare_name() -> None:
    """Every Grant in the live suite names what the model is actually shown."""
    offenders = _grants_naming_a_bare_tool(_LIVE_SUITE)
    assert offenders == [], (
        "these Grants name a tool's bare name; the platform advertises "
        "`<server>__<tool>`, so build the name with `advertised_name_for` from the "
        f"server this case registered: {offenders}"
    )


def test_the_grant_scan_flags_a_planted_bare_name(tmp_path: Path) -> None:
    """The control: both shapes of the mistake are caught and the right one is not.

    Three files rather than two, because a matcher that flags every Grant passes a
    one-file control and would condemn the correct spelling along with the wrong ones.
    """
    (tmp_path / "test_literal.py").write_text('X = {"grant": ["tavily_search"]}\n')
    (tmp_path / "test_constant.py").write_text(
        '_TOOL = "ask_deepwiki"\nX = {"grant": [_TOOL]}\n'
    )
    (tmp_path / "test_built.py").write_text(
        'X = {"grant": [advertised_name_for(server, _TOOL)]}\n'
    )

    assert _grants_naming_a_bare_tool(tmp_path) == [
        "test_constant.py:2",
        "test_literal.py:1",
    ]


_RUNNER: Final = "KubernetesPodRunner"
_INVENTED: Final = "new_session_id"


def _cases_that_invent_the_session_they_place(directory: Path) -> list[str]:
    """Every module that places a pod with the real runner and invents its Session.

    A pod's runtime dials the Tool Gateway while it starts, and the Gateway reads that
    Session's row to learn its Grant. A Session id no route ever issued has no row, the
    handshake fails, and the runtime treats that server as required -- so the pod exits
    3 and the case reports `PodNotStarted`, which names neither the Session nor the
    Gateway. The Session has to be one the API issued; the placement can still be the
    case's own.

    Both names on the module, not on one statement, because the two are usually many
    lines apart -- the runner in a fixture, the record in a compiler helper.
    """
    return sorted(
        path.name
        for path in directory.glob("*.py")
        if _RUNNER in (source := path.read_text()) and f"{_INVENTED}(" in source
    )


def test_no_live_case_places_a_pod_for_a_session_it_invented() -> None:
    offenders = _cases_that_invent_the_session_they_place(_LIVE_SUITE)
    assert offenders == [], (
        "these cases place a real pod for a Session the platform never issued, so the "
        f"Tool Gateway refuses its handshake and the pod exits: {offenders}"
    )


def test_the_invented_session_scan_flags_a_planted_case(tmp_path: Path) -> None:
    """The control: a module needs BOTH names to be flagged, and neither alone."""
    (tmp_path / "test_both.py").write_text(
        "runner = KubernetesPodRunner()\nrecord = new_session_id()\n"
    )
    (tmp_path / "test_places_only.py").write_text("runner = KubernetesPodRunner()\n")
    (tmp_path / "test_invents_only.py").write_text("record = new_session_id()\n")

    assert _cases_that_invent_the_session_they_place(tmp_path) == ["test_both.py"]


_SUBMITS: Final = '"prompt"'
_TERMINALS: Final = ("turn.completed", "TURN_COMPLETED")


def _cases_that_read_the_submission_as_the_answer(directory: Path) -> list[str]:
    """Every module that submits a Turn and never looks for the event that ends it.

    `POST /v1/sessions/{id}/events` answers 202 on admission and runs the Turn on a
    background task. So the response says the Turn was accepted and nothing else: a case
    that reads it as the Turn's end carries on while the pod is still being placed, and
    its own teardown then deletes the pod out from under the Turn. That has surfaced
    three separate ways -- a second Turn refused 409 because the first was genuinely
    still running, two pods never seen Running at once, and a Turn closed
    `runtime_did_not_start: the pod ... is gone` -- and none of the three names the
    submission.

    Matched on the terminal event's name in any spelling, because the wait itself has no
    single shape: some cases poll a helper, one counts terminals it has already seen,
    one runs the wait on a thread. What they cannot do without is naming what they are
    waiting for.

    Over `test_*.py` and not every module, because `cluster_access.py` beside them is a
    helper: it names the route without being the case that submits to it, and a case is
    what the rule is about.
    """
    return sorted(
        path.name
        for path in directory.glob("test_*.py")
        if _SUBMITS in (source := path.read_text())
        and "/events" in source
        and not any(one in source for one in _TERMINALS)
    )


def test_no_live_case_reads_the_submission_as_the_turns_answer() -> None:
    offenders = _cases_that_read_the_submission_as_the_answer(_LIVE_SUITE)
    assert offenders == [], (
        "these cases submit a Turn and never wait for it to end; the route answers 202 "
        f"at admission, so the Turn is only over when its log says so: {offenders}"
    )


def test_the_submission_scan_flags_a_planted_case(tmp_path: Path) -> None:
    """The control: submitting without waiting is flagged, waiting either way is not."""
    (tmp_path / "test_no_wait.py").write_text(
        'r = post("/v1/sessions/1/events", json={"prompt": "hi"})\n'
    )
    (tmp_path / "test_waits.py").write_text(
        'r = post("/v1/sessions/1/events", json={"prompt": "hi"})\n'
        'while "turn.completed" not in seen:\n    pass\n'
    )
    (tmp_path / "test_waits_by_name.py").write_text(
        'r = post("/v1/sessions/1/events", json={"prompt": "hi"})\n'
        "while turn.TURN_COMPLETED not in seen:\n    pass\n"
    )

    assert _cases_that_read_the_submission_as_the_answer(tmp_path) == [
        "test_no_wait.py"
    ]
