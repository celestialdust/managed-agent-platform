"""Push the shippable tree to a public remote, leaving the internal record behind.

This repository keeps its planning and decision record on disk but out of git: `docs/`,
`STATE.md` and `CONTEXT.md` are gitignored and untracked. None of it is part of what
somebody deploying or extending the platform needs, and some of it names internal
accounts, incidents and people's working assumptions. So the published tree is the
source, the tests, the deployment and the README -- and nothing else.

**Why a squashed snapshot and not a push of this branch.** Those paths were tracked
until they were removed from the index, and history does not forget: every commit before
that one still carries the whole record, so `git log -p` on a pushed branch publishes
exactly what the ignore rules withhold from its tip. A snapshot has one commit and one
tree, so what is published is what `--dry-run` prints and nothing else is reachable.

**Why an exclusion list as well as the ignore rules.** The two guard different things.
`.gitignore` stops a path being *added*; `git add -f` walks straight past it, and so
does a path committed before the rule existed. This list is applied to the snapshot's
own index, so a design artefact that reached the tree by either route still does not
reach the remote. On a clean tree it removes nothing, which is the expected `--dry-run`
reading.

Nothing here rewrites history, moves a branch you work on, or touches the working tree.
It writes a tree through a scratch index, commits it on `refs/heads/public`, and pushes
that. The internal repository is unchanged either way, and running it twice adds one
commit to `public` rather than duplicating anything.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parent.parent

# The design record, and the one test that reads it. `tests/test_gates.py` asserts that
# every skip gate in the suite appears in `docs/test-gates.md`; with that file gone the
# test fails for the absence of a document rather than for anything about the code, so
# it leaves with what it grades. The gate table itself is reproduced in README.md, which
# is where a reader outside this repository would look for it.
#
# `CLAUDE.md` and `AGENTS.md` go too, and not because they are secret: every pointer in
# them resolves into `docs/`, so published beside a tree with no `docs/` they are a page
# of dangling references, which is worse than their absence.
#
# The two spike guards leave for the same reason `tests/test_gates.py` does: they grade
# the spike record's Verdict and Measured table, so without `docs/` they fail a clone
# for the absence of a document. `tests/spike/test_spike_image.py` stays -- it reads
# `deploy/spike` and grades the image that gets built, which ships.
EXCLUDED: Final[tuple[str, ...]] = (
    ".claude",
    "AGENTS.md",
    "CLAUDE.md",
    "CONTEXT.md",
    "STATE.md",
    "docs",
    "tests/spike/test_record_complete.py",
    "tests/spike/test_report_shape.py",
    "tests/test_gates.py",
)

_SNAPSHOT_BRANCH: Final = "public"


def _git(*argv: str, index: str | None = None) -> str:
    environment = dict(os.environ)
    if index is not None:
        environment["GIT_INDEX_FILE"] = index
    done = subprocess.run(
        ("git", "-C", str(_ROOT), *argv),
        capture_output=True,
        text=True,
        env=environment,
    )
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(argv)} failed: {done.stderr.strip()}")
    return done.stdout


def tracked_under(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Every tracked path at or under one of `prefixes`, as git spells it."""
    listing = _git("ls-files", "-z", "--", *prefixes)
    return tuple(sorted(path for path in listing.split("\0") if path))


def snapshot_tree(start: str = "HEAD") -> str:
    """Write `start`'s tree minus the excluded paths, and return its object id.

    Through a scratch index rather than the repository's own, so the working tree and
    the real index are untouched and this is safe to run mid-edit -- though the caller
    refuses a dirty tree anyway, for a different reason.

    `start` is a parameter so the strip can be graded rather than assumed. Against HEAD
    it removes nothing today -- the ignore rules already kept every design artefact out
    of the index -- so a test that only ever passed HEAD would assert a property that
    holds for free and would go on passing with the strip deleted. A test hands in a
    tree that *does* carry one and checks it does not come back out.
    """
    with tempfile.TemporaryDirectory() as scratch:
        index = str(Path(scratch) / "index")
        _git("read-tree", start, index=index)
        _git(
            "rm",
            "--cached",
            "-r",
            "-q",
            "--ignore-unmatch",
            "--",
            *EXCLUDED,
            index=index,
        )
        return _git("write-tree", index=index).strip()


def _remote_url(remote: str) -> str | None:
    try:
        return _git("remote", "get-url", remote).strip()
    except RuntimeError:
        return None


def _existing_snapshot() -> str | None:
    try:
        return _git("rev-parse", "--verify", f"refs/heads/{_SNAPSHOT_BRANCH}").strip()
    except RuntimeError:
        return None


def _report(tree: str) -> None:
    withheld = tracked_under(EXCLUDED)
    kept = len(tracked_under((".",))) - len(withheld)
    print(f"snapshot of {_git('rev-parse', '--short', 'HEAD').strip()} -> tree {tree}")
    print(f"  publishing {kept} files")
    if withheld:
        print(
            f"  stripping {len(withheld)} tracked file(s) under: {', '.join(EXCLUDED)}"
        )
        for path in withheld:
            print(f"    - {path}")
    else:
        # The expected reading on a clean tree: the ignore rules already kept every
        # design artefact out of the index, so the strip found nothing left to do.
        print(f"  nothing to strip -- none of {', '.join(EXCLUDED)} is tracked")
    print("  top level of the published tree:")
    for line in _git("ls-tree", "--name-only", tree).splitlines():
        print(f"    {line}")


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the tree and describe it, but create no commit and push nothing",
    )
    parser.add_argument(
        "--remote",
        default=os.environ.get("PUBLISH_REMOTE", "origin"),
        help="the git remote to push to (default: origin, or $PUBLISH_REMOTE)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="the branch on that remote to push to (default: main)",
    )
    options = parser.parse_args(argv)

    # A snapshot names a commit, so an uncommitted edit would ship under a sha that does
    # not contain it -- the same rule the image push scripts apply to a tag, and for the
    # same reason: afterwards nobody can tell which bytes were published.
    dirty = _git("status", "--porcelain").strip()
    if dirty:
        print(
            "refusing: the working tree is dirty, so the snapshot would name a",
            file=sys.stderr,
        )
        print(
            "commit that is not what is published. Commit or stash first:",
            file=sys.stderr,
        )
        print(dirty, file=sys.stderr)
        return 1

    tree = snapshot_tree()
    _report(tree)

    if options.dry_run:
        print("\ndry run: nothing was committed and nothing was pushed")
        return 0

    url = _remote_url(options.remote)
    if url is None:
        print(
            f"\nrefusing: no git remote named {options.remote!r}. Create the "
            "repository and point this at it, then run again:",
            file=sys.stderr,
        )
        print(
            "  gh repo create <name> --private --source=. --remote=origin --push=false",
            file=sys.stderr,
        )
        return 1

    parent = _existing_snapshot()
    message = f"Managed Agent Platform — {_git('rev-parse', 'HEAD').strip()[:12]}"
    argv_commit = ["commit-tree", tree, "-m", message]
    if parent is not None:
        argv_commit += ["-p", parent]
    commit = _git(*argv_commit).strip()
    _git("update-ref", f"refs/heads/{_SNAPSHOT_BRANCH}", commit)
    print(f"\ncommitted {commit[:12]} on {_SNAPSHOT_BRANCH}; pushing to {url}")
    try:
        _git(
            "push",
            options.remote,
            f"refs/heads/{_SNAPSHOT_BRANCH}:refs/heads/{options.branch}",
        )
    except RuntimeError as rejected:
        if not is_a_rejected_push(str(rejected)):
            raise
        _report_a_diverged_mirror(options.remote, options.branch)
        return 1
    print(f"pushed to {options.remote}/{options.branch}")
    return 0


def is_a_rejected_push(stderr: str) -> bool:
    """Whether git refused a push because the branch moved under us.

    Matched on the parenthesised reason git puts on the `! [rejected]` line rather
    than on the hint paragraph below it, because the hints are prose that git has
    reworded before and the reasons are a fixed vocabulary. Two of them mean this:
    `non-fast-forward` when the pushed tip is behind, `fetch first` when the two have
    diverged. Both are the same situation here -- somebody committed on the mirror --
    and both need the same answer.

    Anything else re-raises. A push can also fail for no network, no credential, or a
    branch protection rule, and reporting one of those as "somebody edited the mirror"
    would send the reader to fix a file that is already correct.
    """
    return "(non-fast-forward)" in stderr or "(fetch first)" in stderr


def _report_a_diverged_mirror(remote: str, branch: str) -> None:
    """Say what a rejected push means here, because git's own hint is wrong for it.

    Git suggests `git pull`. That is the right advice for a branch and the wrong advice
    for this: the published branch is a squashed mirror with no ancestry in common with
    the working history, so a pull would drag the mirror's commits into the source
    repository and a merge would put the withheld design record back into the tree the
    next snapshot writes.

    A rejection means somebody committed on the mirror -- most often an edit made
    through the GitHub web editor. That commit is real work and this tool cannot carry
    it back, because the snapshot is built from HEAD's tree and simply overwrites
    whatever the branch held. So the fix has two halves and both are the operator's:
    make the same edit here, then move the local snapshot ref forward to what the
    remote already has, which is what turns the next push back into a fast-forward.
    """
    print(
        f"\nrefusing: {remote}/{branch} has a commit this snapshot is not built on.\n"
        "Somebody committed on the published mirror -- usually a web edit. Do NOT run\n"
        "`git pull`, whatever git just suggested: the mirror shares no history with\n"
        "this repository, and merging it would put the withheld design record back.\n"
        "\nRead what changed, make the same change here, then point the snapshot ref\n"
        "at the remote so the next push fast-forwards:\n"
        f"  git fetch {remote} {branch}\n"
        f"  git log -p refs/heads/{_SNAPSHOT_BRANCH}..FETCH_HEAD\n"
        "  # ...apply that edit to the source and commit it...\n"
        f"  git update-ref refs/heads/{_SNAPSHOT_BRANCH} FETCH_HEAD\n"
        "  make publish",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
