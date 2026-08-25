"""A tool declares an argument its Session Scope narrows, or it cannot be registered.

Tier 1, no infrastructure. This is where the platform's core promise is mechanically
checked: an unbound tool is reachable at the full breadth of a tenant's data by every
Session whose Grant names it, so the refusal *is* the enforcement and it has to land
while the declaration is being parsed. Anything that admitted such a tool -- a warning,
a nullable field, an advisory flag -- puts the enforcement somewhere later, where the
component doing the enforcing is the tenant's own server.

Every refusal here is asserted on its *reason text*, not merely on the fact that
something raised. A `ValidationError` proves the model refused; it does not prove the
model refused for the reason under test, and a check that fires for the wrong reason
passes exactly as loudly as one that fires for the right one.

The fixtures use the coordinates of this project's own MCP server rather than an
invented one, so the shape being asserted is a shape a real registration takes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from managed_agent.core.registration.scope_binding import (
    ParameterType,
    RegisteredTool,
    ScopeBinding,
    ServerRegistration,
    StreamableHttpServer,
    ToolRegistration,
)

DEEPWIKI_ENDPOINT: dict[str, object] = {
    "transport": "streamable_http",
    "url": "https://mcp.deepwiki.com",
    "credential_ref": "vault/acme/deepwiki",
}


def a_tool(**overrides: object) -> dict[str, object]:
    """`ask_question` as deepwiki really declares it, narrowed by repository."""
    return {
        "name": "ask_question",
        "remote_name": "ask_question",
        "parameters": {"repoName": "string", "question": "string"},
        "scope_bindings": [{"dimension": "repository", "argument": "repoName"}],
    } | overrides


def a_registration(**overrides: object) -> dict[str, object]:
    return {
        "server_name": "deepwiki",
        "endpoint": DEEPWIKI_ENDPOINT,
        "tools": [a_tool()],
    } | overrides


def _reason(refused: pytest.ExceptionInfo[ValidationError]) -> str:
    """The message text, so a test can assert *why* the model refused.

    Asserting only that something raised passes just as loudly when the wrong check
    fired, and these checks refuse six different things about one declaration.
    """
    return str(refused.value)


def test_a_tool_binding_a_declared_string_argument_parses() -> None:
    tool = ToolRegistration.model_validate(a_tool())

    assert tool.name == "ask_question"
    assert tool.scope_bindings == (
        ScopeBinding(dimension="repository", argument="repoName"),
    )
    assert tool.parameters["repoName"] is ParameterType.STRING


def test_a_tool_with_no_scope_binding_is_refused_and_the_reason_names_the_tool() -> (
    None
):
    """ADR-003's rule, at the only moment it can still be enforced cheaply."""
    with pytest.raises(ValidationError) as refused:
        ToolRegistration.model_validate(a_tool(scope_bindings=[]))

    reason = _reason(refused)
    assert "ask_question" in reason, reason
    assert "no Scope Binding declared" in reason, reason


def test_a_binding_on_an_undeclared_argument_names_the_tool_and_the_argument() -> None:
    """The commonest mistake is a typo in the argument name, and it is invisible.

    A refusal that said only "inexpressible" would leave a person comparing two
    identical-looking strings; naming the argument is what makes the message actionable.
    """
    with pytest.raises(ValidationError) as refused:
        ToolRegistration.model_validate(
            a_tool(scope_bindings=[{"dimension": "repository", "argument": "repoNmae"}])
        )

    reason = _reason(refused)
    assert "ask_question" in reason, reason
    assert "repoNmae" in reason, reason
    assert "does not declare" in reason, reason


def test_a_binding_on_a_non_string_argument_names_the_type_it_was_declared_as() -> None:
    """A Scope value is written into the argument, so the argument has to hold text."""
    with pytest.raises(ValidationError) as refused:
        ToolRegistration.model_validate(
            a_tool(
                parameters={"repoName": "string", "page": "integer"},
                scope_bindings=[{"dimension": "repository", "argument": "page"}],
            )
        )

    reason = _reason(refused)
    assert "page" in reason, reason
    assert "integer" in reason, reason


@pytest.mark.parametrize(
    "declared", ["integer", "number", "boolean", "object", "array"]
)
def test_no_type_but_string_can_carry_a_scope_value(declared: str) -> None:
    """Asserted over the whole enum rather than one member of it.

    A check written as `!= INTEGER` would pass the single-case test above and admit
    every other type, so the closed set is walked instead of sampled.
    """
    with pytest.raises(ValidationError) as refused:
        ToolRegistration.model_validate(
            a_tool(
                parameters={"field": declared},
                scope_bindings=[{"dimension": "repository", "argument": "field"}],
            )
        )

    assert declared in _reason(refused)


def test_two_bindings_on_one_argument_are_refused() -> None:
    """Which dimension narrows the argument would otherwise depend on list order."""
    with pytest.raises(ValidationError) as refused:
        ToolRegistration.model_validate(
            a_tool(
                scope_bindings=[
                    {"dimension": "repository", "argument": "repoName"},
                    {"dimension": "organisation", "argument": "repoName"},
                ]
            )
        )

    reason = _reason(refused)
    assert "repoName" in reason, reason
    assert "bound twice" in reason, reason


def test_two_bindings_on_one_dimension_are_refused() -> None:
    """One Scope value cannot be written into two arguments and still be one narrowing.

    Distinct from the case above -- there the argument repeats, here the dimension does
    -- and each is caught by its own set, so removing either check leaves the other
    passing.
    """
    with pytest.raises(ValidationError) as refused:
        ToolRegistration.model_validate(
            a_tool(
                parameters={"repoName": "string", "question": "string"},
                scope_bindings=[
                    {"dimension": "repository", "argument": "repoName"},
                    {"dimension": "repository", "argument": "question"},
                ],
            )
        )

    reason = _reason(refused)
    assert "repository" in reason, reason
    assert "bound twice" in reason, reason


def test_a_registration_repeating_a_tool_name_is_refused_naming_the_name() -> None:
    """Two tools of one name inside one registration have no Grant that tells them
    apart, and the store's key would refuse the second with a constraint name instead
    of the tool's."""
    with pytest.raises(ValidationError) as refused:
        ServerRegistration.model_validate(
            a_registration(
                tools=[a_tool(), a_tool(remote_name="askQuestionV2")],
            )
        )

    reason = _reason(refused)
    assert "ask_question" in reason, reason
    assert "repeat" in reason, reason


def test_a_registration_with_no_tools_is_refused() -> None:
    """A server offering nothing is a row no Grant can ever name."""
    with pytest.raises(ValidationError):
        ServerRegistration.model_validate(a_registration(tools=[]))


def test_a_registration_carries_no_reference_to_an_agent_definition() -> None:
    """One registration serves every definition naming the server, so there is no
    per-definition copy of it free to drift.

    Asserted over the field names rather than in prose, because the property is that
    the *type* cannot express such a reference -- with `extra="forbid"`, a submission
    trying to add one is refused rather than dropped.
    """
    assert set(ServerRegistration.model_fields) == {
        "server_name",
        "endpoint",
        "tools",
    }
    assert ServerRegistration.model_config["extra"] == "forbid"


def test_two_tools_of_one_registration_may_bind_different_dimensions() -> None:
    """Uniqueness is per tool, not per registration: two tools naming one dimension is
    the normal case, and a check scoped to the registration would forbid it."""
    registration = ServerRegistration.model_validate(
        a_registration(
            tools=[
                a_tool(),
                a_tool(
                    name="read_wiki_contents",
                    remote_name="read_wiki_contents",
                    parameters={"repoName": "string"},
                ),
            ]
        )
    )

    assert [tool.name for tool in registration.tools] == [
        "ask_question",
        "read_wiki_contents",
    ]
    assert {tool.scope_bindings[0].dimension for tool in registration.tools} == {
        "repository"
    }


def test_a_registered_tool_round_trips_through_the_json_shape_a_row_holds() -> None:
    """What the store writes is what `RegisteredTool` parses back.

    The registry writes `parameters` and `scope_bindings` as JSON documents and reads
    them back into this model, so the two shapes have to agree. Asserted here, where
    it costs no database, and again against real PostgreSQL where the columns are.
    """
    tool = ToolRegistration.model_validate(a_tool())

    read_back = RegisteredTool.model_validate(
        {
            "name": tool.name,
            "remote_name": tool.remote_name,
            "parameters": {
                argument: declared.value
                for argument, declared in tool.parameters.items()
            },
            "scope_bindings": [
                binding.model_dump(mode="json") for binding in tool.scope_bindings
            ],
            "server_name": "deepwiki",
            "endpoint": DEEPWIKI_ENDPOINT,
        }
    )

    assert read_back.parameters == tool.parameters
    assert read_back.scope_bindings == tool.scope_bindings
    assert isinstance(read_back.endpoint, StreamableHttpServer)
    assert read_back.endpoint.url == "https://mcp.deepwiki.com"
