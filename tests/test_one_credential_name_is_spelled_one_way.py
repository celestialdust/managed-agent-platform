"""One credential name is spelled one way, wherever this tree writes it.

The defect: `tests/gateway/model/test_gateway_inbound.py` spelled the pod-token
entry `map/pod-token-signing-key` while the manifest, the IAM policy, the
applier's existence check and every other reference spelled it
`map/dev/platform/pod-token-signing-key`. It could not fail there, because
that file's minter and its verifier share the one constant -- so the two
agree with each other and neither agrees with the account.

`docs/lessons.md` records four separate instances of a value spelled two
ways, and the cure recorded there is not a list of the names in use: a list
is maintained by whoever remembers this guard, while the literals are
written by whoever does not.

**This file's first version got the cure wrong**, and how it was wrong is
worth keeping. It asserted a *shape* -- `map/<stage>/<service>/<name>`, four
segments -- on the premise that every entry in the account has it. Run
against the tree it failed on thirteen literals, almost all legitimate:
`map/webhook-secret` and `map/tool-credential` are prefixes that production
code joins a tenant id onto, `map/upstream/foundry` and its siblings are
routing fixtures, and several were fragments of a concatenated image
reference. There is no single shape. The premise had been invented rather
than measured, which is the defect `docs/lessons.md` records as *"a
configuration floor was written against a URL the platform does not use"* --
recorded one hour before this file repeated it.

So this compares what the defect actually was. Group every `map/...` literal
by its **last path segment**, and treat a segment reached by two different
paths as a defect. That needs no list of names, makes no claim about segment
counts, and stays correct when a slice adds an entry at a depth nobody
anticipated: a two-segment prefix and a four-segment entry both pass, and
only a genuine disagreement fails.

Scanned by walking the syntax tree for string **literals**, not by grepping
for `map/`. A grep reports every prose mention in a docstring, and a guard
that cries wolf is one somebody switches off.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[1]
_TREES: Final = ("src", "tests", "deploy")
_PREFIX: Final = "map/"


def _string_literals(path: Path) -> set[str]:
    """Every string constant in one Python file, at any depth.

    Includes docstrings, deliberately: a docstring naming an entry by a spelling nothing
    else uses misleads a reader as much as code doing it, and this check is cheap enough
    that there is no reason to exempt them.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


def _paths_by_leaf() -> dict[str, dict[str, set[str]]]:
    """Each final path segment, mapped to the full paths using it and where they appear.

    Excluded before comparison rather than inside it, so a failure lists only things
    somebody has to fix: a literal carrying whitespace is prose; one ending
    in `*` is an IAM resource pattern; one ending in `/` is a prefix a caller
    appends to; and one containing `@` or `:` is a fragment of a concatenated
    image reference, not a name.
    """
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for tree in _TREES:
        for path in sorted(_ROOT.joinpath(tree).rglob("*.py")):
            for literal in _string_literals(path):
                if not literal.startswith(_PREFIX) or literal == _PREFIX:
                    continue
                if literal.endswith(("*", "/")) or any(c.isspace() for c in literal):
                    continue
                if "@" in literal or ":" in literal:
                    continue
                leaf = literal.rsplit("/", 1)[-1]
                if not leaf:
                    continue
                grouped[leaf][literal].add(str(path.relative_to(_ROOT)))
    return grouped


def test_the_scan_finds_names_to_compare() -> None:
    """Enough literals for the comparison to mean something.

    This is what fails if the prefix changes or the trees move. Without it, a
    rename turns this file into a pass over zero literals -- the vacuous state
    `docs/lessons.md` records six times over.
    """
    grouped = _paths_by_leaf()
    assert len(grouped) >= 5, (
        f"found only {len(grouped)} distinct {_PREFIX!r} leaf names under "
        f"{list(_TREES)}: {sorted(grouped)}. Either the naming convention "
        "changed, in which case update the prefix here, or the scan is looking "
        "in the wrong place and checking nothing."
    )


def test_no_credential_leaf_name_is_reached_by_two_different_paths() -> None:
    """Two spellings of one name mean at least one names nothing that exists.

    The failure this prevents is the quiet one: a test whose minter and
    verifier share the constant passes forever while naming an entry that
    cannot exist, and the mismatch surfaces only in production as
    `AccessDeniedException` on a path nobody recognises.
    """
    split = {
        leaf: {spelling: sorted(where) for spelling, where in spellings.items()}
        for leaf, spellings in _paths_by_leaf().items()
        if len(spellings) > 1
    }
    assert not split, (
        "these names are spelled more than one way in this tree, so at most "
        f"one of each pair names something that exists: {split}. Pick the "
        "spelling the deployed manifests and IAM policies use and make the "
        "others match it -- a test whose minter and verifier share a constant "
        "cannot detect the disagreement itself."
    )


def test_the_comparison_would_have_caught_the_defect_it_was_written_for() -> None:
    """Falsifies the grouping against the exact pair that motivated this file.

    Asserted over the real grouping function rather than a hand-built dict,
    because the thing that could be wrong is the grouping -- a version keying
    on the whole path instead of the leaf would pass every assertion above and
    catch nothing.
    """
    leaf = "pod-token-signing-key"
    # Assembled, never written as one literal: a literal of the wrong spelling would be
    # found by the scan below and this file would fail on its own example.
    was = _PREFIX + leaf
    now = _PREFIX + "dev/platform/" + leaf
    assert was.rsplit("/", 1)[-1] == leaf and now.rsplit("/", 1)[-1] == leaf
    assert was != now
    grouped = _paths_by_leaf()
    assert leaf in grouped, (
        f"{leaf!r} is no longer written anywhere under {list(_TREES)}, so this file's "
        "motivating case is gone. That is fine, but check the comparison still has "
        "something to compare before trusting it."
    )
    assert set(grouped[leaf]) == {now}, (
        f"expected {leaf!r} to be spelled only {now!r}, found {sorted(grouped[leaf])}"
    )
