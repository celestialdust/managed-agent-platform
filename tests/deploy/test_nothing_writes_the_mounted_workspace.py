"""No object key this platform composes lands inside the mounted workspace.

**This is the case that makes the mount safe, and the failure it guards is silent.**
The S3 Files file system is over `aws_s3_bucket.platform` -- the SAME bucket the VFS
lanes, the evidence store, the uploads and the Rollouts all write to. One bucket, one
key namespace, and the mount is a window onto one prefix of it. So "the pod's workspace
is durable" is not a property of the mount; it is a property of the mount's prefix being
one nothing else writes.

What happens when something else does write it is the part worth spelling out. S3 Files
settles a bucket-versus-file-system conflict in the BUCKET's favour: the file system's
copy is moved to `.s3files-lost+found-<fs-id>`, which sits ABOVE the access point root
and is therefore invisible to the pod, to the browse route, and to every listing either
of them can make. An agent's live working tree is discarded, no call fails, nothing is
logged, and the tenant sees an empty workspace. That is exactly the shape of failure a
test has to catch, because no operator will.

Until ADR-035 the platform did write there, through `control/files/workspace_sync.py`
at every Turn boundary. Deleting that module is what closed the hole; this file is what
keeps it closed, because the next writer will not be called `workspace_sync`.

Two cases, and they fail on different mistakes. The first calls every key builder the
tree has and checks where each one lands -- it catches a builder whose prefix is changed
into a collision. The second scans the source for the prefix as a written string, which
is what a NEW builder would have to contain, and is the half that survives the list of
builders going stale.

That list is `tests/object_key_builders.py` rather than a table in this file, because
`test_the_object_grant_matches_the_keys_the_code_writes.py` grades the same builders
from the other side -- it asks whether IAM grants every prefix they compose. Two copies
of the list would diverge in the direction that reads as safe: a builder added here and
forgotten there is one whose writes AWS refuses, while both files pass.

Tier 1 (local, no cluster). The mount root is read out of `session_vfs.tf` rather than
written here, so this cannot go on passing against an access point that has moved.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

import pytest
from object_key_builders import KEYS as _KEYS

_ROOT: Final = Path(__file__).resolve().parents[2]
_SRC: Final = _ROOT / "src"
_TERRAFORM: Final = _ROOT / "deploy" / "terraform" / "session_vfs.tf"


def _mount_key_root() -> str:
    """The bucket key prefix the Session workspace mount is a window onto.

    Read out of the access point's `root_directory` rather than written here. The two
    are the same fact -- S3 Files maps a POSIX path under the access point root onto a
    bucket key under the same prefix -- and a copy of it in this file would go on
    passing on the day somebody moves the access point, which is the one day this case
    has to fail.
    """
    block = re.search(
        r'resource\s+"aws_s3files_access_point"\s+"session_workspaces"\s*\{'
        r"(?P<body>.*?)\n\}",
        _TERRAFORM.read_text(),
        re.DOTALL,
    )
    assert block, "session_vfs.tf declares no `session_workspaces` access point"
    path = re.search(
        r'root_directory\s*\{[^}]*?path\s*=\s*"(?P<path>[^"]+)"',
        block.group("body"),
        re.DOTALL,
    )
    assert path, "the `session_workspaces` access point declares no root_directory path"

    root = path.group("path").strip("/")
    assert root, "the access point roots at `/`, so every key is inside the mount"
    return f"{root}/"


def test_there_are_key_builders_to_grade() -> None:
    """Guard the guard: the case below is parametrized over `_KEYS`.

    Also that each one composes something. A builder returning an empty string would
    satisfy "does not start with the mount prefix" while composing no key at all, which
    is the way this case would pass by grading nothing.
    """
    assert len(_KEYS) >= 4, "a key builder has gone missing from this list"
    composed = {name: build() for name, build in _KEYS.items()}
    for name, key in composed.items():
        assert key.strip("/"), f"{name} composes an empty key"
        assert "/" in key, f"{name} composes {key!r}, which is not a key under a prefix"
    assert len(set(composed.values())) == len(composed), (
        f"two builders compose the same key: {composed}"
    )


@pytest.mark.parametrize("name", sorted(_KEYS))
def test_no_key_the_platform_composes_lands_inside_the_mounted_workspace(
    name: str,
) -> None:
    """Each builder, against the one prefix it must never reach.

    Asserted per builder rather than over the set, so a refusal names which writer
    moved. The four live in four modules that do not import each other, and nothing but
    this compares them.
    """
    key = _KEYS[name]()
    root = _mount_key_root()

    assert not key.startswith(root), (
        f"{name} composes {key!r}, which is inside the mounted workspace ({root!r})."
        " S3 Files resolves a bucket-versus-file-system conflict in the bucket's"
        " favour and moves the file system's copy to `.s3files-lost+found-<fs-id>`,"
        " above the access point root -- so this write would silently discard an"
        " agent's live working tree"
    )


def _string_literals(source: str) -> list[str]:
    """Every string constant in a module that the code could actually use.

    Prose is excluded, because prose about the mount is exactly what a reader needs --
    `control/api/routes/workspace.py` explains the prefix at length and must go on being
    able to. What must not appear is the prefix in a string the code USES.

    **The rule is "a bare string statement", not "the first statement of a body".** This
    tree documents module-level constants the way it documents functions, with a string
    on the line after the assignment, and those are the strings most likely to discuss a
    key prefix -- `WORKSPACE_MOUNT_ENV_VAR` is one, and it is the whole reason this is
    spelled out. A rule that only skipped a body's first statement would report that
    paragraph as a writer of the mount, which is the reverse of true: it is the constant
    that keeps this route reading the POSIX mount instead of composing a key at all.

    An `ast.Expr` wrapping a string is a value computed and thrown away, so it is never
    a string the code uses, wherever it appears. That is the property being relied on.
    """
    tree = ast.parse(source)
    prose = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in prose
    ]


def test_no_module_in_the_tree_writes_the_mount_prefix_into_a_string() -> None:
    """The half that survives `_KEYS` going stale.

    A new writer aimed at the mount has to name the prefix somewhere, and the only place
    left is a string literal -- so this scans every one of them in `src/` and refuses
    the
    prefix outright. It is deliberately blunt: there is no legitimate reason for any
    module here to compose a key under the mount, because everything that reads the
    workspace reads the POSIX mount instead (`MAP_SESSION_WORKSPACE_MOUNT`), and
    everything that writes durable bytes writes another prefix entirely.

    The consequence of the bluntness is stated rather than hidden: the day a module
    legitimately needs this string, this case fails and somebody has to decide whether
    what they are adding is a second writer into the mount. That is the conversation
    this file exists to force.
    """
    root = _mount_key_root()
    segment = root.strip("/")
    offenders = [
        (path.relative_to(_ROOT), literal)
        for path in sorted(_SRC.rglob("*.py"))
        for literal in _string_literals(path.read_text())
        if root in literal or literal.strip("/") == segment
    ]

    assert not offenders, (
        f"these string literals name the mounted workspace prefix {root!r}:"
        f" {offenders}. Nothing may compose a bucket key there -- read the mount"
        " instead"
    )
