"""The lane namespace: what a key composes to, and that nothing can rewrite one.

Two properties are graded here and they are graded differently, because they hold for
different reasons.

The **key composition** is ordinary runtime behaviour and is asserted by calling it.
Every case that composes a key is parametrized over `LANES` itself rather than over a
list written in this file. Each lane encodes a decision -- that it exists, and whether
it can be rewritten -- and a copy of the collection beside the test would be free to
fall behind the module, leaving a lane added later graded by nothing. That has already
happened once in this repo, to a set of refusal reasons in `pod_runner.py`, four of
whose five members were reachable by no assertion at all.

The **seal** is a property of the surface rather than of any call, so it is asserted
against the surface. There is no method here that overwrites -- `replace` and the
mutable lane kind it took went with the workspace mount (ADR-035) -- and what makes that
a guarantee rather than a coincidence is that no signature admits one. So the cases
below assert on absence: that neither the store protocol nor the blob port grew a
rewriting method back. An assertion that called something and checked it failed would be
grading an implementation; the guarantee is that there is nothing to call.
"""

from typing import get_type_hints
from uuid import uuid4

import pytest

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vfs.session_vfs import (
    Lane,
    LaneBlobs,
    LaneNameInvalid,
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
A_SECOND_LANE = SealedLane("scratchpad")
LANES: tuple[Lane, ...] = (A_SEALED_LANE, A_SECOND_LANE)
"""Example lanes, declared here because the platform declares only one.

Two, because a key composed from a lane has to be graded against more than one lane --
a composition that ignored its argument passes every case run against a single lane. A
third would grade nothing the second does not, now that the kinds are one kind.

Declared locally rather than parametrized over `session_vfs.LANES`, which holds exactly
one member. Deliberately not `evidence`/`artifacts`/`working`: a reader who saw those
words here would take them for a platform default, which is exactly the thing that was
removed.
"""


def a_file(lane: Lane, relative: str = "a-file.txt") -> VfsFile:
    """One file in whichever lane, for the parametrized cases below."""
    return SealedFile(A_TENANT, A_SESSION, lane, relative)


def test_there_are_distinct_lanes_to_grade() -> None:
    """Guard the guard: every case below is parametrized over `LANES`, so an empty
    collection would pass all of them by running none.

    Distinct, and not merely two. What the cases below need is that a key composed for
    one lane can be told from a key composed for another -- two entries spelled the same
    would satisfy a count and grade nothing.
    """
    assert len(LANES) >= 2, f"{LANES} cannot grade a per-lane composition"
    assert len({lane.directory for lane in LANES}) == len(LANES), (
        f"{LANES} holds the same lane twice"
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
    theirs = SealedFile(other, A_SESSION, lane, "a-file.txt").key
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
    """The one function every path type calls, graded directly.

    `SealedFile` parses in `__post_init__` and calls nothing else, so a refusal weakened
    here is weakened everywhere at once -- which is the point of there being one
    parser, and the reason it is worth an assertion of its own rather than only through
    the type.
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
def test_a_lane_cannot_be_declared_under_a_name_that_is_not_one_directory(
    directory: str,
) -> None:
    """Asserted on the constructor, because that is where a caller declares one.

    This case exists because of what the platform stopped deciding. While the lane set
    was four constants in the module, an unvalidated name was safe by accident -- nobody
    could write one. Now that a caller declares the set, the name is the leading
    segments of an object key composed from input, and `..` in it is one tenant's key
    inside another's.
    """
    with pytest.raises(LaneNameInvalid):
        SealedLane(directory)


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


_A_REWRITE_BY_ANY_NAME = ("replace", "put", "overwrite", "update")
"""Names a rewriting method would plausibly arrive under.

A closed list and not a heuristic. What this guards is the *reintroduction* of an
overwrite, and reintroduction comes with a name somebody chose -- `replace` and `put`
are the two that were actually here, and the other two are what the same method gets
called when it comes back under a different word. A case asserting only that `replace`
is absent is one a rename walks straight past.

Deliberately excluding `write` and `save`, which read as "create" at least as often as
"overwrite". A name this list claims is always a rewrite has to be one, or the first
honest rename of `place` fails a case about something else entirely.
"""


@pytest.mark.parametrize("name", _A_REWRITE_BY_ANY_NAME)
def test_the_store_offers_no_way_to_overwrite_a_stored_object(name: str) -> None:
    """The seal is the absence of a method, so absence is what is asserted.

    `SessionFiles.replace` and `LaneBlobs.put` both existed while a mutable lane did.
    Both are gone, and what makes that a guarantee rather than a gap is that nothing
    above the adapter can express an overwrite at all: an artifact's recorded digest is
    a claim about bytes, and it is worth exactly as much as the narrowest write surface
    underneath it.

    Asserted on both protocols, because a rewrite reintroduced at either level is a
    rewrite. The port is where a caller would reach for one; the blob surface is where
    one could be added without any caller changing.
    """
    assert not hasattr(SessionFiles, name), (
        f"SessionFiles.{name} can overwrite a stored object; the lane seal is gone"
    )
    assert not hasattr(LaneBlobs, name), (
        f"LaneBlobs.{name} can overwrite a stored object; the lane seal is gone"
    )


def test_the_one_write_takes_a_file_in_any_lane() -> None:
    """`place` is the whole write surface, and it is not narrowed to a lane.

    Creating is legal in every lane -- what is refused is a second write to a key that
    already holds bytes, and the store refuses that conditionally rather than this
    signature refusing it by type.
    """
    assert get_type_hints(SessionFiles.place)["file"] == VfsFile


def test_a_stored_file_is_typed_over_a_sealed_lane_and_nothing_else() -> None:
    """The other half of the same guarantee.

    Removing every overwrite buys nothing if a file type could later be built over a
    lane kind that permits one. There is one kind, and this is the assertion that fails
    if a second is introduced and this type widened to admit it.
    """
    assert get_type_hints(SealedFile)["lane"] is SealedLane
