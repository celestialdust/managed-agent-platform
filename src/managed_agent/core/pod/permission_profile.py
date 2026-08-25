"""Filesystem rules bounding what a confined command in a Session's pod may touch.

Every rule is an absolute path prefix and never a glob. A glob-form deny rule makes
the runtime scan the tree with ripgrep while it compiles the sandbox argv, and an
expansion past the runtime's 8192-match cap is fatal — the confined command then does
not run at all, so the same rule set can pass on one node and brick another. Over a
remote-backed mount that scan is also a network directory walk. A prefix costs
neither (ADR-006).

`_GLOB_METACHARACTERS` is not a guess about what the runtime treats as a glob. Its
classifier is `path.chars().any(|ch| matches!(ch, '*' | '?' | '[' | ']'))`, so these
four characters and no others turn a filesystem table key into a pattern entry. A
path this module accepts is therefore one the runtime parses as a concrete path.

The older coarse sandbox settings cannot be expressed here, and that is the point: no
field takes a sandbox mode, and there is no writable root that is not itself a rule,
so a configuration assembled out of this module cannot quietly fall back to them
(ADR-005).

Rules validate in __post_init__ rather than through a separate checker, so a
constructed rule is proof its path was parsed. Nothing downstream re-checks it and
nothing downstream can skip the check.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

_GLOB_METACHARACTERS = frozenset("*?[]")
_BUILTIN_PARENTS = frozenset({":read-only", ":workspace"})
_RESERVED_PROFILE_NAME = "filesystem"


class FsAccess(StrEnum):
    """Access under a path prefix.

    Precedence is deny > write > read, and a more specific prefix beats a broader
    one, so a profile paints an area and carves holes in it. That resolution is the
    runtime's and is deliberately not reimplemented here: nothing in this module
    weighs two rules against each other, because a profile declaring two rules over
    one path is refused rather than resolved.
    """

    READ = "read"
    WRITE = "write"
    DENY = "deny"


def path_spelling_error(path: str) -> str | None:
    """Why this is not the one spelling a rule path is accepted in, or None.

    Absolute, no glob metacharacter, no trailing separator except on the root, no
    empty component and no `.` or `..` component. Returns the reason as a fragment a
    caller completes rather than raising, because two callers want two failures out of
    it: `FsRule` refuses a value at construction with `ValueError`, and the compiled
    document's floors refuse a rendered document with `FloorViolation`. One rule, two
    renderings; a second copy of the rule would be free to drift from this one, and
    the floor's whole reason for grading the rendered text is that the Python value
    and the document can disagree.

    `/run/codex/` and `/run/./codex` name one directory and are two strings, and a
    rule written in a spelling nothing else uses is a rule whose target the sandbox
    looks for at a path the pod never created.
    """
    if not path.startswith("/"):
        return "must be absolute"
    if _GLOB_METACHARACTERS & set(path):
        return "must be a prefix, not a glob"
    if path != "/" and path.endswith("/"):
        return "must not end in a separator"
    if "//" in path or any(part in {".", ".."} for part in path.split("/")):
        return "must be normalised"
    return None


def _components(path: str) -> tuple[str, ...]:
    """The path as the sequence of names a path comparison actually walks.

    Empty segments and `.` segments carry no name and are dropped -- `/a//b` and
    `/a/./b` walk the same two names as `/a/b`. A `..` segment is KEPT as a name of
    its own and never resolved: resolving it would need the filesystem, which is in a
    pod that does not exist where this runs, and guessing at it would turn a
    comparison into a claim about an inode. A leading `/` is kept as its own leading
    element so an absolute path is never a prefix of a relative one, or the reverse.
    """
    named = tuple(part for part in path.split("/") if part not in ("", "."))
    return (("/",) + named) if path.startswith("/") else named


def is_strictly_under(child: str, parent: str) -> bool:
    """True when `child` names something inside `parent` and is not `parent` itself.

    Compares the two as sequences of path components rather than as strings, and that
    is the whole design of it. A string prefix test answers a question about spelling
    when the question is about paths: `/run/codexx` starts with `/run/codex` and is
    not inside it, `parent + "/"` spells `"//"` for the root and so makes the root
    nobody's parent, and `/run//codex/x` does not start with `/run/codex/` while
    naming a path that is plainly inside it. Every one of those is a wrong answer a
    prefix test gives confidently. Comparing components gives all three correctly and
    needs no list of the spellings somebody might write.

    That last case is why this is not merely a tidier prefix test. This repository's
    most expensive recurring defect is a guard that grades one spelling of a value
    while the thing that consumes it accepts several, and a deny set reaches a
    filesystem-path consumer that does not care how many separators were typed. A
    guard that did care would report a clean deny set over a pair the sandbox treats
    as nested.

    `PurePosixPath.is_relative_to` is not used and would be wrong here: it collapses a
    `.` component while keeping `..` and a leading `//`, so it normalises some
    spellings and not others, and it reports a path as relative to itself.

    A symlink is out of reach and is not claimed: this runs where no pod exists, so
    two spellings that are one inode cannot be told apart. What covers that is the
    init container's `-L` walk of every component of the control-socket path and the
    runtime treating a deny path across a writable symlink as fatal (ADR-012).
    """
    ancestor = _components(parent)
    descendant = _components(child)
    return len(descendant) > len(ancestor) and descendant[: len(ancestor)] == ancestor


def nested_deny_pairs(denied: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Every (ancestor, descendant) pair among these paths, ancestor first.

    Returns the pairs rather than a bool so a caller can name both paths in its
    refusal. A caller that only needed a yes/no would still have to re-derive the pair
    to say anything useful, and the two derivations could disagree.
    """
    paths = tuple(denied)
    return tuple(
        (ancestor, descendant)
        for ancestor in paths
        for descendant in paths
        if is_strictly_under(descendant, ancestor)
    )


@dataclass(frozen=True, slots=True)
class FsRule:
    """One absolute path prefix and the access permitted beneath it.

    A path is accepted only in the one spelling the runtime compares against:
    absolute, no glob metacharacter, no trailing separator, no empty or dot
    component. The normalisation checks are not tidiness. `/run/codex/` and
    `/run/./codex` name one directory and are two strings, and a rule written in a
    spelling nothing else uses is a rule whose target the sandbox looks for at a path
    the pod never created.
    """

    path: str
    access: FsAccess

    def __post_init__(self) -> None:
        reason = path_spelling_error(self.path)
        if reason is not None:
            raise ValueError(f"rule path {reason}: {self.path!r}")


@dataclass(frozen=True, slots=True)
class PermissionProfile:
    """A named set of rules layered on a built-in parent.

    ':danger-full-access' is not an accepted parent. The runtime rejects it outright,
    and a profile reaching for it would be asking for the substrate this platform
    does not use — better refused here, where the message can say so.

    The name 'filesystem' is refused for a rendering reason rather than a runtime
    one: the managed document puts this profile's body at `[permissions.<name>]` and
    the non-weakenable deny-read list at `[permissions.filesystem]`. A profile named
    'filesystem' renders those two into one TOML table, and the collision parses.

    Two rules over one path are refused rather than merged: the runtime resolves that
    pair by its own precedence, and a caller who wrote both almost certainly meant
    one.
    """

    name: str
    extends: str
    rules: tuple[FsRule, ...]

    def __post_init__(self) -> None:
        if self.name.startswith(":") or self.name == _RESERVED_PROFILE_NAME:
            raise ValueError(f"profile name is reserved by the runtime: {self.name!r}")
        if self.extends not in _BUILTIN_PARENTS:
            raise ValueError(f"parent must be one of {sorted(_BUILTIN_PARENTS)}")
        if not self.rules:
            raise ValueError("an empty profile keeps access restricted and only warns")
        paths = [r.path for r in self.rules]
        if len(set(paths)) != len(paths):
            raise ValueError(f"two rules over one path: {sorted(paths)}")

    def denied(self) -> tuple[str, ...]:
        """The paths this profile denies, in declaration order."""
        return tuple(r.path for r in self.rules if r.access is FsAccess.DENY)

    def writable(self) -> tuple[str, ...]:
        """The paths this profile makes writable, in declaration order.

        The positive half of the question `denied()` answers, and the floor check
        needs both: a profile that denies everything the checkpoint names and grants
        nothing satisfies every absence and runs no agent.
        """
        return tuple(r.path for r in self.rules if r.access is FsAccess.WRITE)
