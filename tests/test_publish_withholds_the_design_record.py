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

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))

from publish import EXCLUDED, snapshot_tree  # noqa: E402

# Present in every published tree. Without these, a set-membership assertion about what
# is absent would also hold over an empty tree.
_MUST_SHIP = ("README.md", "pyproject.toml", "src", "tests", "deploy", "migrations")

_A_PATH_UNDER_AN_EXCLUDED_PREFIX = "docs/a-design-note-this-test-invented.md"


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
