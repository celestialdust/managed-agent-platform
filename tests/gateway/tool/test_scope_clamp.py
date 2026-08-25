"""Narrowing a tool call to the Session's Scope, and refusing the call that cannot be.

Tier 1 (local, no infrastructure). The subject is a pure function, so these are the
tests that can state the property directly rather than through a server: what the
outbound call carries, and what happens when the Scope has nothing to narrow with.

Two properties are asserted separately because either one alone is satisfiable by
abandoning the other. A function that refused everything would satisfy every refusal
case; a function that returned the arguments untouched would satisfy every pass-through
case. Neither is the clamp.

The bindings cases are parametrized over the tool's own `scope_bindings` rather than
over a list written beside it. A tool declares as many bindings as it likes and the
clamp has to apply *every* one -- a check that drives the function through the first
binding grades the function, not the collection, and the way that failure shows up is
that a second binding can be dropped with nothing going red.
"""

from __future__ import annotations

from typing import Final
from uuid import uuid4

import pytest

from managed_agent.core.ids import SessionId
from managed_agent.core.registration.scope_binding import (
    ParameterType,
    RegisteredTool,
    ScopeBinding,
    StreamableHttpServer,
)
from managed_agent.gateway.tool.scope_clamp import OutOfScope, narrow

SESSION: Final = SessionId(uuid4())


def _tool(*bindings: ScopeBinding, **parameters: ParameterType) -> RegisteredTool:
    """A registered tool carrying exactly the bindings and parameters named."""
    declared = dict(parameters) or {"repo_name": ParameterType.STRING}
    return RegisteredTool(
        name="ask_question",
        remote_name="ask_question",
        parameters=declared,
        scope_bindings=bindings
        or (ScopeBinding(dimension="repository", argument="repo_name"),),
        server_name="deepwiki",
        endpoint=StreamableHttpServer(
            transport="streamable_http", url="https://mcp.deepwiki.com/mcp"
        ),
    )


def test_the_scope_value_replaces_what_the_model_asked_for() -> None:
    """The whole point: the model does not get to choose the bound argument.

    If a value the model supplied could survive here, the narrowing would be advice
    rather than a boundary, and a model that guessed a different repository would be
    served it.
    """
    narrowed = narrow(
        _tool(),
        {"repository": "acme/widgets"},
        {"repo_name": "someone-else/private", "question": "what is this"},
    )

    assert not isinstance(narrowed, OutOfScope)
    assert narrowed["repo_name"] == "acme/widgets"


def test_an_argument_no_binding_names_is_carried_through_untouched() -> None:
    """The clamp narrows the bound arguments and is not a filter on the rest.

    Asserted because the failure it guards against passes every other test here: a
    function that returned only the bound arguments would narrow correctly and strip
    the model's actual question on the way.
    """
    narrowed = narrow(
        _tool(),
        {"repository": "acme/widgets"},
        {"repo_name": "acme/widgets", "question": "what is this"},
    )

    assert not isinstance(narrowed, OutOfScope)
    assert narrowed["question"] == "what is this"


def test_a_bound_argument_the_model_omitted_is_supplied_by_the_scope() -> None:
    """Written in, not merely overwritten.

    A clamp that only rewrote arguments already present would leave a model free to
    widen the call by omitting the bound one -- the argument would be absent from the
    outbound call and the server would apply its own default, which is the whole of
    the tenant's data on every tool this platform brokers.
    """
    narrowed = narrow(_tool(), {"repository": "acme/widgets"}, {"question": "what"})

    assert not isinstance(narrowed, OutOfScope)
    assert narrowed["repo_name"] == "acme/widgets"


def test_the_arguments_the_caller_passed_are_not_mutated() -> None:
    """The caller's mapping is left as it was; the narrowed call is a new object.

    `McpProxy.call_tool` receives its arguments from the MCP server layer and the
    failure path re-reads them, so a clamp that rewrote them in place would make the
    record of what the model asked for say what the platform decided instead.
    """
    asked = {"repo_name": "someone-else/private", "question": "what is this"}

    narrow(_tool(), {"repository": "acme/widgets"}, asked)

    assert asked["repo_name"] == "someone-else/private"


_TWO_BINDINGS: Final = _tool(
    ScopeBinding(dimension="repository", argument="repo_name"),
    ScopeBinding(dimension="branch", argument="ref"),
    repo_name=ParameterType.STRING,
    ref=ParameterType.STRING,
    question=ParameterType.STRING,
)


@pytest.mark.parametrize(
    "binding", _TWO_BINDINGS.scope_bindings, ids=lambda b: b.dimension
)
def test_every_binding_the_tool_declares_is_applied(binding: ScopeBinding) -> None:
    """Parametrized over the tool's own bindings, so dropping one cannot stay green."""
    scope = {"repository": "acme/widgets", "branch": "release-2"}

    narrowed = narrow(
        _TWO_BINDINGS,
        scope,
        {"repo_name": "wrong/repo", "ref": "wrong-branch", "question": "what"},
    )

    assert not isinstance(narrowed, OutOfScope)
    assert narrowed[binding.argument] == scope[binding.dimension]


def test_a_dimension_the_session_scope_does_not_carry_refuses_the_call() -> None:
    """Fail-safe: a call that cannot be narrowed is not made.

    The registration was accepted on the promise that this tool could be narrowed. A
    Session whose Scope is silent on the dimension cannot keep that promise, and the
    two ways to proceed are to refuse or to send the call at the full breadth of the
    tenant's data. Refusing is the one that matches what the tenant was told.
    """
    refused = narrow(_tool(), {}, {"question": "what is this"})

    assert isinstance(refused, OutOfScope)
    assert refused.dimension == "repository"
    assert refused.tool_name == "ask_question"


def test_a_scope_that_narrows_only_some_of_the_bindings_refuses_the_whole_call() -> (
    None
):
    """A partially narrowed call is a widened call, so it is refused rather than sent.

    This is the case a clamp written as a loop with a `continue` gets wrong, and it is
    the dangerous direction: every binding the Scope does carry gets applied, the call
    looks narrowed at the point it goes out, and the one dimension nobody supplied is
    the one running at the full breadth of the tenant's data.
    """
    refused = narrow(
        _TWO_BINDINGS,
        {"repository": "acme/widgets"},
        {"repo_name": "acme/widgets", "ref": "anything", "question": "what"},
    )

    assert isinstance(refused, OutOfScope)
    assert refused.dimension == "branch"


def test_the_refusal_names_no_scope_value() -> None:
    """Nothing the refusal carries lets a model learn what the Scope holds.

    A model that could read a Scope value out of a refusal could also probe for one:
    call, read the refusal, adjust, call again. That is why a disagreement between the
    model's argument and the Scope is answered by overwriting rather than refusing --
    and it is only worth anything if the refusal that remains says nothing either.
    """
    refused = narrow(
        _TWO_BINDINGS,
        {"repository": "acme/widgets"},
        {"repo_name": "acme/widgets", "ref": "anything", "question": "what"},
    )

    assert isinstance(refused, OutOfScope)
    assert "acme/widgets" not in repr(refused)
