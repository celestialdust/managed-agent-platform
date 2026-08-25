"""The tree `tools/publish.py` would push carries no part of the design record.

`.gitignore` keeps `docs/`, `CONTEXT.md`, `STATE.md` and the agent instructions out of
the index, and that is the whole protection until somebody types `git add -f`, or checks
out a commit from before the ignore rules existed. Either way the path is tracked again,
and nothing about the tree looks wrong. This asserts the property the ignore rules are
only a means to: whatever reached the index, the published tree does not carry it.

**The strip is graded against a tree that actually carries one.** Against HEAD it
removes nothing -- the ignore rules already did that -- so a case that only checked HEAD
would assert a property holding for free and would go on passing with the strip deleted.
Both mutations were tried and both passed before this file built the excluded path in.
So the first case here hands `snapshot_tree` a tree with a design artefact in it and
checks it does not come back out, and checks the result is byte-identical to the clean
snapshot.

The artefact is built as a loose blob and placed with `update-index --cacheinfo` rather
than by adding a real file, because none of the excluded paths exists in a fresh clone
-- that is the point of them -- so a case that force-added one from the working tree
would pass here and fail in CI.

`snapshot_tree()` is imported rather than driven through the command line, which is the
opposite of how `tools/skip_gates.py` is exercised, and for one reason: the CLI refuses
a dirty working tree, so a developer with an unsaved edit would see this fail for a
reason that is not about publishing. The function writes through a scratch index of its
own and touches neither the real index nor the working tree, so it is safe to call at
any time.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))

from publish import EXCLUDED, is_a_rejected_push, snapshot_tree  # noqa: E402

# Present in every published tree. Without these, a set-membership assertion about what
# is absent would also hold over an empty tree.
_MUST_SHIP = ("README.md", "pyproject.toml", "src", "tests", "deploy", "migrations")

_A_PATH_UNDER_AN_EXCLUDED_PREFIX = "docs/a-design-note-this-test-invented.md"

_MAY_NAME_AN_EXCLUDED_PATH = frozenset(
    {
        # Creates a `docs/` of its own inside `tmp_path` to prove the image build
        # ignores one. It reads nothing from the repository's.
        "tests/deploy/test_session_image_reaches_the_registry.py",
        # `.claude/skills/<name>/SKILL.md` is this platform's domain vocabulary, not a
        # path in this repository: it is where a *tenant's* uploaded repository keeps
        # its skills, and the file posts one to `/v1/skills/repository`. The collision
        # with the agent-instruction directory excluded here is in the name only.
        "tests/control/test_skill_listing.py",
        # Grades the exclusion list, so it invents a path under one on purpose --
        # `_A_PATH_UNDER_AN_EXCLUDED_PREFIX`, which exists in no tree but the fixture's.
        "tests/test_publish_withholds_the_design_record.py",
    }
)
"""Published tests allowed to write one of these names, each with the reason.

Every entry costs a reader the question "does this one really not read the record?", so
the reason is written next to it rather than left to be re-derived. A test that reaches
this list because it genuinely reads the design record is in the wrong list: it belongs
in `EXCLUDED`, with the document it grades.
"""


def _git(*argv: str, index: str | None = None) -> str:
    environment = None
    if index is not None:
        environment = {**os.environ, "GIT_INDEX_FILE": index}
    return subprocess.run(
        ["git", "-C", str(_ROOT), *argv],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout


def _paths_in(tree: str) -> tuple[str, ...]:
    listing = _git("ls-tree", "-r", "--name-only", tree)
    return tuple(line for line in listing.splitlines() if line)


def _excluded_paths_in(tree: str) -> list[str]:
    return sorted(
        path
        for path in _paths_in(tree)
        if any(path == name or path.startswith(f"{name}/") for name in EXCLUDED)
    )


def _a_tree_carrying_a_design_artefact() -> str:
    """HEAD's tree plus one file under an excluded prefix, written to no branch.

    The blob is hashed from a string rather than read off disk: every excluded path is
    absent from a fresh clone, so there is nothing to add there.
    """
    blob = subprocess.run(
        ["git", "-C", str(_ROOT), "hash-object", "-w", "--stdin"],
        input="a design note that must never be published\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as scratch:
        index = str(Path(scratch) / "index")
        _git("read-tree", "HEAD", index=index)
        _git(
            "update-index",
            "--add",
            "--cacheinfo",
            f"100644,{blob},{_A_PATH_UNDER_AN_EXCLUDED_PREFIX}",
            index=index,
        )
        return _git("write-tree", index=index).strip()


def test_a_design_artefact_in_the_index_is_stripped_from_the_snapshot() -> None:
    carrying = _a_tree_carrying_a_design_artefact()
    assert _A_PATH_UNDER_AN_EXCLUDED_PREFIX in _paths_in(carrying), (
        "the fixture did not build the case it claims to; nothing below grades anything"
    )

    published = snapshot_tree(carrying)

    assert not _excluded_paths_in(published), (
        f"{_excluded_paths_in(published)} would be published. A path under an excluded "
        "prefix reached the index and tools/publish.py did not strip it."
    )
    assert published == snapshot_tree(), (
        "the snapshot differs depending on whether a design artefact was tracked, so "
        "what gets published is not decided by the exclusion list alone"
    )


def test_the_snapshot_of_head_carries_no_excluded_path() -> None:
    """The live case: whatever is tracked right now, none of it is the design record."""
    leaked = _excluded_paths_in(snapshot_tree())
    assert not leaked, (
        f"{len(leaked)} design-record path(s) would be published: {leaked}. They are "
        "gitignored, so something force-added them or they predate the ignore rules."
    )


def test_the_snapshot_still_carries_what_ships() -> None:
    published = _paths_in(snapshot_tree())
    missing = [
        name
        for name in _MUST_SHIP
        if not any(path == name or path.startswith(f"{name}/") for path in published)
    ]
    assert not missing, f"the published tree is missing {missing}"


def _string_constants(source: str) -> list[str]:
    return [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _names_an_excluded_path(value: str) -> bool:
    """Whether a string constant reads as a path at or under an excluded prefix.

    Prefix-matching on the literal, not on a resolved path, because that is the only
    form available: the path is assembled at runtime from `Path(...).parents[2]` and a
    chain of `/` operands, so nothing static knows what it resolves to. What is
    available is the pieces, and a test that names `"docs"` at all is the signal.
    """
    return any(value == name or value.startswith(f"{name}/") for name in EXCLUDED)


def test_no_published_test_reads_a_path_the_publish_withholds() -> None:
    """A test that ships must be runnable in a clone of what ships.

    This is the guard for a failure that reached CI. `tests/spike/test_report_shape.py`
    and `tests/spike/test_record_complete.py` grade a document under `docs/`, not
    published, so on a runner they failed ten times at "the spike record does not exist"
    -- for the absence of a file rather than for anything about the code.

    The scan that was supposed to have found them beforehand was a regex over single
    lines, and both files build their path across several:

        RECORD = (
            Path(__file__).resolve().parents[2]
            / "docs"
            ...

    So this walks the syntax tree instead. Every string constant in every published test
    is checked against the exclusion list, which catches the fragment `"docs"` wherever
    in an expression it sits. It costs a parse of each test file and nothing else.

    It over-reports by construction, because a literal is all it can see: three
    published tests name one of these strings and read nothing, including this one. They
    sit in `_MAY_NAME_AN_EXCLUDED_PATH` with the reason beside each. Over-reporting is
    the right direction for this guard -- a false positive costs a line and a sentence,
    and a false negative is ten setup errors on a runner.
    """
    offenders: list[str] = []
    for path in sorted(_ROOT.glob("tests/**/*.py")):
        relative = path.relative_to(_ROOT).as_posix()
        if _names_an_excluded_path(relative) or relative in _MAY_NAME_AN_EXCLUDED_PATH:
            continue
        named = sorted(
            {
                v
                for v in _string_constants(path.read_text())
                if _names_an_excluded_path(v)
            }
        )
        if named:
            offenders.append(f"{relative} names {named}")

    assert not offenders, (
        "these test files ship but read something the publish withholds, so they fail "
        "a clone for the absence of a document, not for anything about the code. So "
        "exclude the test along with what it grades, or stop it reading the record:\n  "
        + "\n  ".join(offenders)
    )


# Git's own words, copied out of a push this repository actually made. Both were
# produced against the real remote rather than written from memory: the first by
# pushing a commit the branch was already past, the second by pushing after somebody
# had committed on the mirror through the web editor.
_BEHIND = """To https://github.com/example/managed-agent-platform.git
 ! [rejected]        45e9aba -> main (non-fast-forward)
error: failed to push some refs to 'https://github.com/example/managed-agent-platform.git'
hint: Updates were rejected because a pushed branch tip is behind its remote
hint: counterpart. If you want to integrate the remote changes, use 'git pull'
hint: before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details."""

_DIVERGED = """To https://github.com/example/managed-agent-platform.git
 ! [rejected]        public -> main (fetch first)
error: failed to push some refs to 'https://github.com/example/managed-agent-platform.git'
hint: Updates were rejected because the remote contains work that you do not
hint: have locally. This is usually caused by another repository pushing to
hint: the same ref. If you want to integrate the remote changes, use
hint: 'git pull' before pushing again."""


def test_a_push_rejected_for_a_moved_branch_is_recognised() -> None:
    """Both spellings, because they are the same situation and take the same answer.

    The mirror is a squashed snapshot with no ancestry in common with this history, so
    git's advice in both messages -- `git pull` -- is the one thing that must not be
    done: it would drag the mirror's commits in here and a merge would put the withheld
    design record back into the tree the next snapshot writes. `publish.py` prints its
    own instructions instead, and this is what routes it there.
    """
    assert is_a_rejected_push(_BEHIND)
    assert is_a_rejected_push(_DIVERGED)


def test_a_push_that_failed_for_another_reason_is_not_recognised() -> None:
    """Everything else re-raises, and this is the half that keeps that true.

    A push also fails with no network, no credential, and against a protection rule.
    Reporting one of those as "somebody edited the mirror" sends the reader to
    reconcile a file that was never out of step, and the real cause -- an expired
    token, a rule they need an owner to change -- goes unread.
    """
    other = (
        "fatal: could not read Username for 'https://github.com': terminal prompts "
        "disabled",
        "fatal: unable to access 'https://github.com/example/x.git/': Could not "
        "resolve host: github.com",
        """To https://github.com/example/x.git
 ! [remote rejected] public -> main (protected branch hook declined)
error: failed to push some refs to 'https://github.com/example/x.git'""",
    )
    for stderr in other:
        assert not is_a_rejected_push(stderr), stderr
