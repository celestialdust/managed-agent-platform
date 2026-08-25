"""Grades the rule and profile types on what they refuse to construct.

Tier 1 (local, no infrastructure). These are the types a compiled configuration is
assembled out of, and their whole value is in the constructions they make unavailable: a
glob-form rule, a relative path, a profile extending the full-access built-in. So most
cases here assert a refusal, and each names the reason in its message rather than only
the exception type -- for a constraint an author will be tempted to work around, the
sentence is the deliverable.

The accepting cases are not filler. A type that refused everything would satisfy
every refusal below, so each family of refusals is paired with the construction that
must still succeed.
"""

import pytest

from managed_agent.core.pod.permission_profile import (
    FsAccess,
    FsRule,
    PermissionProfile,
)


@pytest.mark.parametrize(
    "path", ["session/workspace", "", "./workspace", "~/workspace"]
)
def test_a_rule_path_that_is_not_absolute_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        FsRule(path=path, access=FsAccess.DENY)


@pytest.mark.parametrize(
    "path",
    [
        "/session/**/*.env",
        "/session/workspace/*",
        "/session/workspace/?",
        "/session/[abc]",
        "/session/]",
    ],
)
def test_a_glob_form_rule_path_is_refused(path: str) -> None:
    """The four characters here are exactly the runtime's own glob classifier.

    A glob deny rule makes it walk the tree while it compiles the sandbox argv, and an
    expansion past its match cap stops every command in the Session -- so the same rule
    set passes on one node and bricks another.
    """
    with pytest.raises(ValueError, match="not a glob"):
        FsRule(path=path, access=FsAccess.DENY)


@pytest.mark.parametrize("path", ["/session/workspace/", "/run/"])
def test_a_trailing_separator_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="must not end in a separator"):
        FsRule(path=path, access=FsAccess.DENY)


@pytest.mark.parametrize("path", ["/run/./codex", "/run/../codex", "/run//codex"])
def test_an_unnormalised_rule_path_is_refused(path: str) -> None:
    """Two spellings of one directory are two strings to whatever compares them."""
    with pytest.raises(ValueError, match="must be normalised"):
        FsRule(path=path, access=FsAccess.DENY)


@pytest.mark.parametrize("path", ["/", "/run/codex", "/run/codex/app-server.sock"])
def test_a_plain_absolute_prefix_is_accepted(path: str) -> None:
    assert FsRule(path=path, access=FsAccess.READ).path == path


def test_the_root_prefix_is_not_mistaken_for_a_trailing_separator() -> None:
    """`/` both starts and ends with a separator, so the two checks must not collide."""
    assert FsRule(path="/", access=FsAccess.READ).path == "/"


def _rules() -> tuple[FsRule, ...]:
    return (
        FsRule(path="/session/workspace", access=FsAccess.WRITE),
        FsRule(path="/run/codex", access=FsAccess.DENY),
    )


@pytest.mark.parametrize("name", [":x", ":read-only", "filesystem"])
def test_a_reserved_profile_name_is_refused(name: str) -> None:
    """'filesystem' collides with the managed deny-read table when rendered."""
    with pytest.raises(ValueError, match="reserved by the runtime"):
        PermissionProfile(name=name, extends=":read-only", rules=_rules())


@pytest.mark.parametrize("parent", [":danger-full-access", ":nonesuch", "map-session"])
def test_a_parent_outside_the_two_built_ins_is_refused(parent: str) -> None:
    """The runtime rejects ':danger-full-access' too; refusing here says why."""
    with pytest.raises(ValueError, match="parent must be one of"):
        PermissionProfile(name="map-session", extends=parent, rules=_rules())


def test_an_empty_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="empty profile"):
        PermissionProfile(name="map-session", extends=":read-only", rules=())


def test_two_rules_over_one_path_are_refused_rather_than_merged() -> None:
    doubled = (
        FsRule(path="/run/codex", access=FsAccess.READ),
        FsRule(path="/run/codex", access=FsAccess.DENY),
    )
    with pytest.raises(ValueError, match="two rules over one path"):
        PermissionProfile(name="map-session", extends=":read-only", rules=doubled)


def test_a_well_formed_profile_is_accepted_over_either_built_in() -> None:
    for parent in (":read-only", ":workspace"):
        profile = PermissionProfile(name="map-session", extends=parent, rules=_rules())
        assert profile.extends == parent


def test_denied_and_writable_report_their_own_halves_in_declaration_order() -> None:
    profile = PermissionProfile(
        name="map-session",
        extends=":read-only",
        rules=(
            FsRule(path="/session/workspace", access=FsAccess.WRITE),
            FsRule(path="/etc/codex", access=FsAccess.DENY),
            FsRule(path="/usr/share", access=FsAccess.READ),
            FsRule(path="/run/codex", access=FsAccess.DENY),
        ),
    )

    assert profile.denied() == ("/etc/codex", "/run/codex")
    assert profile.writable() == ("/session/workspace",)
