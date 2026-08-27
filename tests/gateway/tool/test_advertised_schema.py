"""The schema a tool is offered under, once its Scope-bound arguments are gone.

Tier 1 (local, no infrastructure). The subject is a pure function over an upstream's
own JSON Schema, so every case here states a property about the schema that leaves
rather than about the listing that carried it. `test_proxy_listing.py` holds the other
half -- that the listing path actually calls this -- because a perfect function nothing
reaches advertises nothing.

Two properties are asserted apart throughout, because either alone is satisfiable by
abandoning the other: the bound argument is gone, **and** everything the upstream
declared beside it is still there. A function that answered `{"type": "object"}` for
every tool would pass every removal case here and leave the model unable to call
anything.

The drift cases are not defensive decoration. A registration is written once and the
server behind it keeps shipping: the argument a binding names can stop being declared,
stop being required, or start being spelled some way this platform never anticipated,
and all of that arrives at listing time. A listing that raises over it is a Session with
no tools at all, which is a far worse answer than a schema with nothing removed.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from managed_agent.core.registration.scope_binding import ScopeBinding
from managed_agent.gateway.tool.advertised_schema import without_bound_arguments

REPOSITORY: Final = (ScopeBinding(dimension="repository", argument="repo_name"),)

TWO_BINDINGS: Final = (
    ScopeBinding(dimension="repository", argument="repo_name"),
    ScopeBinding(dimension="branch", argument="ref"),
)


def _schema(
    properties: dict[str, Any] | None = None, required: list[str] | None = None
) -> dict[str, Any]:
    """An upstream's schema, carrying only what the case under test is about."""
    schema: dict[str, Any] = {"type": "object"}
    if properties is not None:
        schema["properties"] = properties
    if required is not None:
        schema["required"] = required
    return schema


def test_a_bound_argument_is_not_a_property_of_the_offered_schema() -> None:
    """The whole point: the model is never shown the field the platform fills.

    A model that can read the argument out of the schema can supply it, and a model
    that supplies it gets a refusal instead of an answer -- so leaving it declared
    turns a boundary the platform holds into a rule the model has to be told about
    twice.
    """
    offered = without_bound_arguments(
        _schema({"repo_name": {"type": "string"}, "question": {"type": "string"}}),
        REPOSITORY,
    )

    assert "repo_name" not in offered["properties"]


def test_the_arguments_no_binding_names_are_still_offered() -> None:
    """Removal is aimed, not a wipe.

    Asserted separately because the failure it guards against passes the case above:
    a function that returned a schema declaring no properties at all would strip every
    bound argument correctly and take the model's actual question with it.
    """
    offered = without_bound_arguments(
        _schema({"repo_name": {"type": "string"}, "question": {"type": "string"}}),
        REPOSITORY,
    )

    assert offered["properties"] == {"question": {"type": "string"}}


def test_a_required_bound_argument_leaves_the_required_list_with_it() -> None:
    """The hard case the ADR names, and it fails in a direction nobody would look.

    A schema that requires a property it no longer declares is one no caller can
    satisfy: a runtime validating arguments before the call refuses every call to this
    tool, and a model reading the schema is told to supply a field it has no
    declaration for. The tool would be advertised and uncallable -- which reads as the
    platform being broken rather than as a tool being narrowed.
    """
    offered = without_bound_arguments(
        _schema(
            {"repo_name": {"type": "string"}, "question": {"type": "string"}},
            required=["repo_name", "question"],
        ),
        REPOSITORY,
    )

    assert offered["required"] == ["question"]


def test_the_arguments_the_upstream_required_beside_it_stay_required() -> None:
    """The other half of the same edit, and the direction that loses a promise.

    A function that dropped `required` wholesale would pass the case above. What it
    would cost is every other required argument: the model is no longer told the
    question is mandatory, and the failure moves from a schema the runtime can check
    to an upstream refusing a call that arrived incomplete.
    """
    offered = without_bound_arguments(
        _schema(
            {"repo_name": {"type": "string"}, "question": {"type": "string"}},
            required=["repo_name", "question"],
        ),
        REPOSITORY,
    )

    assert "question" in offered["required"]


def test_a_required_list_the_removal_empties_is_dropped_rather_than_left_empty() -> (
    None
):
    """An empty `required` is legal in 2020-12 and was not in draft 4.

    The Agent Runtime is not the only reader of this schema and this platform does not
    choose the validator behind it. Leaving `"required": []` behind on a tool whose one
    argument was bound puts a keyword that some drafts refuse into every listing, for
    no gain -- the absent keyword says exactly what the empty list says.
    """
    offered = without_bound_arguments(
        _schema({"repo_name": {"type": "string"}}, required=["repo_name"]),
        REPOSITORY,
    )

    assert "required" not in offered


@pytest.mark.parametrize("binding", TWO_BINDINGS, ids=lambda b: b.argument)
def test_every_binding_the_tool_declares_is_removed(binding: ScopeBinding) -> None:
    """Parametrized over the bindings themselves, so dropping one cannot stay green.

    A tool declares as many bindings as it likes, and a loop that removed the first
    and returned would be graded identically by a case written against a single
    binding. What that failure looks like in production is one bound argument still
    advertised on a tool with two.
    """
    offered = without_bound_arguments(
        _schema(
            {
                "repo_name": {"type": "string"},
                "ref": {"type": "string"},
                "question": {"type": "string"},
            },
            required=["repo_name", "ref", "question"],
        ),
        TWO_BINDINGS,
    )

    assert binding.argument not in offered["properties"]
    assert binding.argument not in offered["required"]


def test_the_upstreams_own_schema_object_is_left_as_it_was() -> None:
    """The upstream's `Tool` outlives this call, and the next listing re-reads it.

    A removal done in place edits the object the MCP client cached for the connection,
    so the second listing of a Session would be stripping an already-stripped schema --
    green either way -- and any other reader of that tool would see a schema this
    process quietly rewrote. Nested containers are the part that actually bites: a
    top-level copy that shares `properties` with the original still empties it.
    """
    upstream = _schema(
        {"repo_name": {"type": "string"}, "question": {"type": "string"}},
        required=["repo_name", "question"],
    )

    without_bound_arguments(upstream, REPOSITORY)

    assert upstream["properties"] == {
        "repo_name": {"type": "string"},
        "question": {"type": "string"},
    }
    assert upstream["required"] == ["repo_name", "question"]


def test_an_argument_the_upstream_stopped_declaring_is_not_an_error() -> None:
    """Drift, and the reason this is handled here rather than at the call.

    A registration is written once against a schema that keeps moving. When the server
    renames or drops the argument a binding names, there is nothing to remove -- and
    the listing has to carry on, because the alternative is that one stale binding
    empties the catalogue for every tool the tenant registered. The clamp still writes
    the value on the outbound call, and the server refuses it there if it must.
    """
    offered = without_bound_arguments(
        _schema({"question": {"type": "string"}}, required=["question"]), REPOSITORY
    )

    assert offered["properties"] == {"question": {"type": "string"}}
    assert offered["required"] == ["question"]


def test_a_schema_that_declares_no_properties_is_returned_as_it_came() -> None:
    """`{"type": "object", "additionalProperties": true}` -- a real registered shape.

    The conformance stdio server's `echo_arguments` declares exactly this, under the
    clamp's own real-call suite. There is no property to remove and the schema forbids
    nothing, so stripping cannot stop a model supplying the field here. That is the
    limit the ADR records rather than a gap: stripping is the ergonomics, and the
    refusal on the call path is the enforcement. What this pins is that the limit is
    reached quietly -- no raise, and no structure invented to have something to remove.
    """
    upstream: dict[str, Any] = {"type": "object", "additional_properties": True}

    assert without_bound_arguments(upstream, REPOSITORY) == upstream


def test_an_argument_required_without_being_declared_still_leaves_required() -> None:
    """A schema that requires what it does not declare, which servers really send.

    The two keywords drift apart on their own, and a removal driven off `properties`
    alone would leave the bound argument required and uncallable -- the exact failure
    the required case above exists to prevent, arriving by the one route that case
    cannot see.
    """
    offered = without_bound_arguments(
        _schema({"question": {"type": "string"}}, required=["repo_name", "question"]),
        REPOSITORY,
    )

    assert offered["required"] == ["question"]


@pytest.mark.parametrize(
    "malformed",
    [
        {"type": "object", "properties": ["repo_name"]},
        {"type": "object", "properties": {"repo_name": {}}, "required": "repo_name"},
        {"type": "object", "properties": None, "required": None},
    ],
    ids=["properties-is-a-list", "required-is-a-string", "both-are-null"],
)
def test_a_schema_shaped_unlike_json_schema_is_passed_through_untouched(
    malformed: dict[str, Any],
) -> None:
    """Nothing here validates the upstream, and a listing must not die trying.

    These are not hypothetical shapes so much as the space of what arrives when a
    registered server is written by somebody else: `input_schema` is typed
    `dict[str, Any]` and the MCP client checks nothing below the top level. A
    `TypeError` raised in the middle of the listing loop is caught by the per-server
    handler and recorded as an upstream failure -- so a schema this platform could not
    read would be reported as a server that could not be reached, and every other tool
    on that server would vanish with it.
    """
    unchanged = {key: value for key, value in malformed.items()}

    assert without_bound_arguments(malformed, REPOSITORY) == unchanged
