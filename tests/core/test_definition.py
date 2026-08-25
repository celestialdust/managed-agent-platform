"""What a declarative agent definition accepts, and what it refuses at the boundary.

Pure parsing, so no database and no container: every property here is a property of
the model itself, and a test that needed a store to grade one would be grading the
store.

The refusals are the point rather than the happy path. A definition is what an
engineer writes instead of code, so a mistake in one is a mistake in the agent, and
the only place it can be caught cheaply is the moment it arrives.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from managed_agent.core.registration.definition import (
    AgentDefinition,
    MultiAgentPosture,
)

_SHA = "0" * 39 + "a"


def _body(**overrides: object) -> dict[str, object]:
    """A well-formed submission, with named fields replaced.

    Written as a dict rather than by constructing and copying a model, because half
    these tests are about inputs no model can hold.
    """
    return {
        "name": "slr-reviewer",
        "instructions": "Extract findings and name the source document for each.",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SHA,
    } | overrides


def test_a_well_formed_definition_parses_into_a_typed_value() -> None:
    """The boundary hands back a value whose type proves it was checked."""
    definition = AgentDefinition.model_validate(_body())

    assert definition.name == "slr-reviewer"
    assert definition.skills_revision == _SHA
    assert definition.tool_servers == frozenset()
    assert definition.multiagent == MultiAgentPosture(enabled=False, max_depth=1)


def test_an_unknown_field_is_refused_and_the_error_names_it() -> None:
    """A misspelled field is refused rather than dropped.

    Silently ignoring it is the harmful case: the tenant believes they set a posture
    and the platform runs the default, and nothing anywhere reports a disagreement.
    """
    with pytest.raises(ValidationError) as raised:
        AgentDefinition.model_validate(_body(multiagnet={"enabled": True}))

    assert "multiagnet" in str(raised.value), (
        "the refusal does not name the offending field, so a tenant cannot tell "
        f"which of their keys was wrong: {raised.value}"
    )


def test_a_branch_name_is_refused_with_the_reason_a_branch_cannot_pin() -> None:
    """`main` is refused, and the message explains the reason rather than the shape.

    A pattern mismatch alone tells a tenant their string is the wrong shape, which
    they can see. What they cannot see is that a branch is refused *on purpose* --
    that a moving reference would let the definition change under a Session already
    running on it. Without that sentence the obvious next move is to look for the
    flag that turns the check off.
    """
    with pytest.raises(ValidationError) as raised:
        AgentDefinition.model_validate(_body(skills_revision="main"))

    message = str(raised.value)
    assert "moving reference" in message, (
        "the refusal does not say why a branch is refused, so the reader's next move "
        f"is to look for the switch that disables it: {message}"
    )


@pytest.mark.parametrize(
    ("revision", "why"),
    [
        ("0" * 39, "a 39-character id is one short of a full sha and is refused"),
        ("0" * 41, "a 41-character id is refused"),
        ("A" * 40, "an uppercase sha is not the form git prints and is refused"),
        ("v1.2.0", "a tag moves and is refused"),
        ("", "an empty revision is refused"),
        (" " + "0" * 39, "a padded id is refused rather than stripped"),
    ],
)
def test_anything_that_is_not_a_full_lowercase_sha_is_refused(
    revision: str, why: str
) -> None:
    with pytest.raises(ValidationError, match="moving reference"):
        AgentDefinition.model_validate(_body(skills_revision=revision))
    assert why  # the parametrize label is the documentation


def test_a_model_name_no_catalog_knows_parses_without_error() -> None:
    """No catalog is consulted here, and that is the decision rather than an omission.

    A model name is a routing key. A capability mismatch surfaces as a failed Turn
    naming the missing feature; a catalog checked at registration goes stale in a way
    a tenant cannot distinguish from the failure it exists to prevent (ADR-010).
    """
    definition = AgentDefinition.model_validate(
        _body(model="a-model-nobody-has-heard-of-9000")
    )

    assert definition.model == "a-model-nobody-has-heard-of-9000"


def test_a_definition_cannot_be_mutated_after_it_is_parsed() -> None:
    """Parsed once and thereafter read-only.

    A definition is versioned rather than edited, and a value that could be edited in
    place after parsing would make the revision number describe something other than
    what is running.
    """
    definition = AgentDefinition.model_validate(_body())

    with pytest.raises(ValidationError):
        definition.model = "something-else"


@pytest.mark.parametrize("depth", [0, -1, 5])
def test_a_multiagent_depth_outside_the_permitted_range_is_refused(depth: int) -> None:
    """Depth 1 is a single agent; the ceiling bounds how deep a tree may nest."""
    with pytest.raises(ValidationError):
        AgentDefinition.model_validate(_body(multiagent={"max_depth": depth}))


def test_tool_servers_parse_into_a_set_that_cannot_be_added_to() -> None:
    """The tool list is a set because order carries no meaning, and frozen because a
    definition that grew a tool after parsing would widen what an agent may reach
    without any revision recording it."""
    definition = AgentDefinition.model_validate(
        _body(tool_servers=["crossref", "crossref", "openalex"])
    )

    assert definition.tool_servers == frozenset({"crossref", "openalex"})
    assert isinstance(definition.tool_servers, frozenset)


@pytest.mark.parametrize(
    ("field", "padded_value"),
    [
        ("name", " slr-reviewer"),
        ("name", "slr-reviewer "),
        ("name", " slr-reviewer "),
        ("instructions", " Extract findings"),
        ("instructions", "Extract findings "),
        ("model", " gpt-5-codex"),
        ("model", "gpt-5-codex "),
        ("model", " gpt-5-codex "),
    ],
)
def test_whitespace_padded_strings_are_refused_at_the_boundary(
    field: str, padded_value: str
) -> None:
    """Leading or trailing whitespace is refused rather than stripped.

    A padded value stored as-is would fail when rendered into config.toml and compared
    against routing tables -- the failure would surface as a Session that will not
    compile, far from where the bad value was registered. Refusing at the boundary
    surfaces the error immediately and ensures what is stored is exactly what was sent.
    """
    with pytest.raises(ValidationError) as raised:
        AgentDefinition.model_validate(_body(**{field: padded_value}))

    message = str(raised.value)
    assert "whitespace" in message, (
        f"the refusal does not mention whitespace, so a reader cannot tell what is "
        f"wrong with their input: {message}"
    )
