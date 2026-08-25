"""The lane namespace: what a key composes to, and which lane a rewrite can name.

Two properties are graded here and they are graded differently, because they hold for
different reasons.

The **key composition** is ordinary runtime behaviour and is asserted by calling it.
Every case that composes a key is parametrized over `LANES` itself rather than over a
list written in this file. Each lane encodes a decision -- that it exists, and whether
it can be rewritten -- and a copy of the collection beside the test would be free to
fall behind the module, leaving a lane added later graded by nothing. That has already
happened once in this repo, to a set of refusal reasons in `pod_runner.py`, four of
whose five members were reachable by no assertion at all.

The **lane lifecycle** is a property of the signatures, so it is asserted against the
signatures. `replace` naming `MutableFile` is what makes overwriting a sealed lane
inexpressible, and `mypy --strict` over the whole tree is what enforces it at every call
site; the cases below grade the claim those two rest on -- that the annotation really is
the narrow type, and that a `MutableFile`'s lane field really is the one lane kind that
permits a rewrite. An assertion that merely called `replace` with a sealed file would be
testing Python's willingness to ignore annotations, which is not the guarantee.
"""

from typing import get_type_hints
from uuid import uuid4

import pytest

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vfs.session_vfs import (
    Lane,
    LaneNameInvalid,
    MutableFile,
    MutableLane,
    SealedFile,
    SealedLane,
    SessionFiles,
    VfsFile,
    VfsPathInvalid,
    lane_prefix,
    parse_lane_name,
    parse_relative_path,
)

A_TENANT = TenantId(uuid4())
A_SESSION = SessionId(uuid4())

A_SEALED_LANE = SealedLane("kept")
A_MUTABLE_LANE = MutableLane("scratchpad")
LANES: tuple[Lane, ...] = (A_SEALED_LANE, A_MUTABLE_LANE)
"""Example lanes, declared here because the platform declares none.

Two, and only two, because two lane *kinds* is the whole of what this module decides.
While four named lanes lived in the module, parametrizing over them graded a taxonomy;
now that a caller declares the set, a third example would grade nothing a second sealed
lane does not already grade. Deliberately not `evidence`/`artifacts`/`working`: a reader
who saw those words here would take them for a platform default, which is exactly the
thing that was removed.
"""


def a_file(lane: Lane, relative: str = "a-file.txt") -> VfsFile:
    """One file in whichever lane, under the right path type for that lane's kind.

    The narrowing is what a caller does once and the type system carries afterwards. It
    is here so the parametrized cases below can hold any lane in one variable while the
    write cases still get a path type whose lane field is the narrow one.
    """
    if isinstance(lane, MutableLane):
        return MutableFile(A_TENANT, A_SESSION, lane, relative)
    return SealedFile(A_TENANT, A_SESSION, lane, relative)


def test_there_are_lanes_of_both_kinds_to_grade() -> None:
    """Guard the guard: every case below is parametrized over `LANES`, so an empty
    collection would pass all of them by running none.

    Both kinds and not a count. A count was the right guard while the module declared
    the set; now that a caller does, a count would only assert that this file's own
    fixture list is the length this file wrote it -- which is nothing. What the cases
    below need is that each kind is represented, because the two kinds are what the
    module actually distinguishes.
    """
    assert any(isinstance(lane, SealedLane) for lane in LANES), (
        f"{LANES} has no sealed lane"
    )
    assert any(isinstance(lane, MutableLane) for lane in LANES), (
        f"{LANES} has no mutable lane"
    )


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
def test_every_lane_composes_under_its_own_tenant_and_session(lane: Lane) -> None:
    prefix = lane_prefix(A_TENANT, A_SESSION, lane)
    assert prefix == f"sessions/{A_TENANT}/{A_SESSION}/{lane.directory}/"
    assert a_file(lane).key == prefix + "a-file.txt"


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
def test_every_lane_puts_the_tenant_above_the_session(lane: Lane) -> None:
    """Tenant first, so a tenant's whole VFS is one prefix.

    Asserted per lane rather than once, because the ordering is what makes a tenant-wide
    sweep a prefix operation, and a lane composed by some other route would not have it.
    """
    key = a_file(lane).key
    assert key.index(str(A_TENANT)) < key.index(str(A_SESSION))


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
def test_no_lane_lets_one_tenants_path_reach_another_tenants_key(lane: Lane) -> None:
    """The relative path is text the agent wrote; the tenant is composed in, not
    compared.

    Two tenants sharing a Session id and a path get two keys, and neither key mentions
    the other tenant. That is the property `core/vault_names.py` exists to hold for
    vault entries, and the reason it is asserted again here is that a second surface
    composing its own name is a second place the guarantee can be weakened while the
    first surface's tests still pass -- which is exactly how the webhook path once
    shipped with no composition at all.
    """
    other = TenantId(uuid4())
    mine = a_file(lane).key
    theirs = (
        MutableFile(other, A_SESSION, lane, "a-file.txt").key
        if isinstance(lane, MutableLane)
        else SealedFile(other, A_SESSION, lane, "a-file.txt").key
    )
    assert mine != theirs
    assert str(other) not in mine
    assert str(A_TENANT) not in theirs


_ESCAPES = (
    "../evidence/other",
    "a/../../b",
    "..",
    "/absolute",
    "",
    "a//b",
    "trailing/",
    ".hidden",
)
"""Paths that must not compose to a key. Each is a distinct way out of the lane.

Parametrized rather than checked as a group so a refusal that stops covering one of them
names which one. `.hidden` is here because the first character is required to be
alphanumeric: a path beginning with a dot would otherwise address the lane's own prefix.
"""


@pytest.mark.parametrize("relative", _ESCAPES)
@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
def test_no_lane_composes_a_key_from_a_path_that_leaves_it(
    lane: Lane, relative: str
) -> None:
    with pytest.raises(VfsPathInvalid):
        a_file(lane, relative)


@pytest.mark.parametrize("relative", _ESCAPES)
def test_the_parser_refuses_the_same_paths_the_path_types_do(relative: str) -> None:
    """The one function both path types call, graded directly.

    Both `SealedFile` and `MutableFile` parse in `__post_init__`, so a refusal weakened
    here weakens both at once -- which is the point of there being one parser, and the
    reason it is worth an assertion of its own rather than only through the types.
    """
    with pytest.raises(VfsPathInvalid) as refused:
        parse_relative_path(relative)
    assert refused.value.relative == relative


def test_a_refusal_names_the_path_and_never_the_composed_key() -> None:
    """The path is the tenant's own text; a composed key carries the tenant's id.

    This message reaches a service log, so echoing the path discloses nothing while
    echoing a key would put a tenant id there.
    """
    with pytest.raises(VfsPathInvalid) as refused:
        parse_relative_path("../escape")
    assert str(A_TENANT) not in str(refused.value)


@pytest.mark.parametrize("relative", ("a-file.txt", "deep/nested/path.json", "a.b-c_d"))
def test_an_ordinary_path_composes(relative: str) -> None:
    assert parse_relative_path(relative) == relative


def test_a_path_at_the_length_limit_composes_and_one_past_it_does_not() -> None:
    from managed_agent.core.vfs.session_vfs import MAX_RELATIVE_LEN

    assert parse_relative_path("a" * MAX_RELATIVE_LEN)
    with pytest.raises(VfsPathInvalid):
        parse_relative_path("a" * (MAX_RELATIVE_LEN + 1))


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
def test_every_lane_is_one_kind_or_the_other_and_never_both(lane: Lane) -> None:
    """The kinds are the lifecycle, so each lane belongs to exactly one.

    A lane that satisfied neither type would be one no write method accepts; a lane that
    somehow satisfied both would be one `replace` accepts, which is the whole thing the
    split prevents.
    """
    assert isinstance(lane, SealedLane) != isinstance(lane, MutableLane)


_NOT_ONE_DIRECTORY = (
    "..",
    "../other-tenant",
    "a/b",
    "/absolute",
    "",
    ".hidden",
    "a..b",
    "Upper",
    "-leading-dash",
)
"""Names that must not become a lane. Each is a distinct way out of one directory.

Parametrized rather than checked as a group, so a refusal that stops covering one of
them fails on that one instead of hiding behind the others. The first four are the ones
that compose a key outside this Session; the rest are refused to keep a lane name one
plain lowercase directory, because a name that differs from another only in case is a
name that is one object on S3 and one file on a case-insensitive mount.
"""


@pytest.mark.parametrize("directory", _NOT_ONE_DIRECTORY)
@pytest.mark.parametrize("kind", (SealedLane, MutableLane))
def test_neither_lane_kind_can_be_declared_under_a_name_that_is_not_one_directory(
    kind: type[SealedLane] | type[MutableLane], directory: str
) -> None:
    """Both kinds, because the validation is on each and a caller reaches for either.

    This case exists because of what the platform stopped deciding. While the lane set
    was four constants in the module, an unvalidated name was safe by accident -- nobody
    could write one. Now that a caller declares the set, the name is the leading
    segments of an object key composed from input, and `..` in it is one tenant's key
    inside another's.
    """
    with pytest.raises(LaneNameInvalid):
        kind(directory)


@pytest.mark.parametrize("directory", _NOT_ONE_DIRECTORY)
def test_the_lane_parser_refuses_the_same_names_the_lane_kinds_do(
    directory: str,
) -> None:
    """The parser and the constructors cannot drift, because one calls the other.

    Asserted anyway: a caller validating a declaration before building anything reaches
    the parser directly, and a refusal it did not share would let that caller accept a
    name the constructor then rejects at a point with no declaration in hand.
    """
    with pytest.raises(LaneNameInvalid):
        parse_lane_name(directory)


@pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.directory)
def test_a_declared_lane_keeps_the_name_it_was_declared_under(lane: Lane) -> None:
    """The accepting half. Without it every case above passes on a parser that refuses
    everything, which is the failure mode a refusal test cannot see on its own."""
    assert parse_lane_name(lane.directory) == lane.directory


def test_replace_accepts_only_a_file_in_the_rewritable_lane() -> None:
    """The signature is where overwriting a sealed lane becomes inexpressible.

    `mypy --strict` runs over this whole tree, so a call site passing a `SealedFile`
    here fails the type gate. What this asserts is the claim that gate rests on: that
    the annotation is the narrow type and not the union.
    """
    assert get_type_hints(SessionFiles.replace)["file"] is MutableFile


def test_place_accepts_a_file_in_any_lane() -> None:
    """Creating is legal everywhere; only rewriting is narrowed."""
    assert get_type_hints(SessionFiles.place)["file"] == VfsFile


def test_a_rewritable_file_cannot_be_typed_over_a_sealed_lane() -> None:
    """The other half of the same guarantee.

    Narrowing `replace` would buy nothing if a `MutableFile` could be built over
    `EVIDENCE`; its lane field is annotated with the mutable kind, so it cannot.
    """
    assert get_type_hints(MutableFile)["lane"] is MutableLane
    assert get_type_hints(SealedFile)["lane"] is SealedLane
