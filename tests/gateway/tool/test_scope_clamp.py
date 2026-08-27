"""Narrowing a tool call to the Session's Scope, and refusing the call that cannot be.

Tier 1 (local, no infrastructure). The subject is a pure function, so these are the
tests that can state the property directly rather than through a server: what the
outbound call carries, and what happens when the Scope has nothing to narrow with.

Two properties are asserted separately because either one alone is satisfiable by
abandoning the other. A function that refused everything would satisfy every refusal
case; a function that returned the arguments untouched would satisfy every pass-through
case. Neither is the clamp.

A bound argument the caller supplied is now a refusal rather than an overwrite, and the
cases for it come in a pair that has to be read together. Refusing the value that
*disagrees* with the Scope is the easy half and is not worth much on its own -- a model
learns from which guesses are refused, which is the oracle the whole design closes. So
the case that supplies the Scope's own correct value is asserted beside it, because the
refusal has to be indifferent to what was supplied for any of this to hold.

The bindings cases are parametrized over the tool's own `scope_bindings` rather than
over a list written beside it. A tool declares as many bindings as it likes and the
clamp has to apply *every* one -- a check that drives the function through the first
binding grades the function, not the collection, and the way that failure shows up is
that a second binding can be dropped with nothing going red.
"""

from __future__ import annotations

import logging
from typing import Final
from uuid import uuid4

import pytest

from managed_agent.core.ids import SessionId
from managed_agent.core.registration.advertised_name import advertised_name_for
from managed_agent.core.registration.scope_binding import (
    ParameterType,
    RegisteredTool,
    ScopeBinding,
    StreamableHttpServer,
)
from managed_agent.gateway.tool.scope_clamp import (
    ArgumentNotOffered,
    OutOfScope,
    ScopeRefusal,
    narrow,
)

SESSION: Final = SessionId(uuid4())
CLAMP_LOGGER: Final = "managed_agent.gateway.tool.scope_clamp"
PROBED: Final = "someone-else/private"
"""What a model supplies when it is guessing at the Scope rather than working."""


def _tool(*bindings: ScopeBinding, **parameters: ParameterType) -> RegisteredTool:
    """A registered tool carrying exactly the bindings and parameters named."""
    declared = dict(parameters) or {"repo_name": ParameterType.STRING}
    return RegisteredTool(
        name="ask_question",
        advertised_name=advertised_name_for("deepwiki", "ask_question"),
        remote_name="ask_question",
        parameters=declared,
        scope_bindings=bindings
        or (ScopeBinding(dimension="repository", argument="repo_name"),),
        server_name="deepwiki",
        endpoint=StreamableHttpServer(
            transport="streamable_http", url="https://mcp.deepwiki.com/mcp"
        ),
    )


def test_a_bound_argument_the_caller_supplied_refuses_the_call() -> None:
    """The model does not get to choose the bound argument -- and is told so.

    This replaces an overwrite. Sending the call with the Scope's value substituted
    was cheaper and never failed a call that would have worked, and what it cost was
    a sentence nobody can say to a tenant afterwards: the model asked to act on A and
    the platform acted on B. On a write or a delete that is not a defect anyone can
    explain after the fact.

    The refusal names the argument, because the field is not in the schema this
    Gateway advertises and a caller that supplied it anyway needs to know which one to
    drop.
    """
    refused = narrow(
        _tool(),
        {"repository": "acme/widgets"},
        {"repo_name": PROBED, "question": "what is this"},
    )

    assert isinstance(refused, ArgumentNotOffered)
    assert refused.tool_name == "ask_question"
    assert refused.dimension == "repository"
    assert refused.argument == "repo_name"


def test_supplying_the_scopes_own_value_is_refused_exactly_the_same() -> None:
    """The half that makes the other half worth having, and it looks like pedantry.

    A clamp that refused only a *disagreeing* value would read as more forgiving and
    would rebuild the oracle this design exists to close: call with a guess, read
    whether the refusal came back, and walk to the Scope's value one guess at a time.
    The refusal is therefore indifferent to what was supplied -- it is about the field
    having been supplied at all, which is a protocol violation because the field was
    never offered.

    A test that only ever supplied a wrong value could not tell those two designs
    apart, and the wrong one passes every other case in this file.
    """
    refused = narrow(
        _tool(),
        {"repository": "acme/widgets"},
        {"repo_name": "acme/widgets", "question": "what is this"},
    )

    assert isinstance(refused, ArgumentNotOffered)
    assert refused.argument == "repo_name"


def test_the_refusal_carries_no_value_while_the_log_line_carries_the_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One event, two audiences, and only one of them may be told what was attempted.

    The refusal goes back to the model. The line goes to the platform's own log, which
    no model and no tenant reads, and it is the only place the attempted value is
    allowed to exist -- it is what separates a model probing for the Scope from a
    caller carrying an argument that has simply gone stale. Asserted together because
    each is satisfiable by breaking the other: dropping the value everywhere passes the
    first assertion, and carrying it on the returned object passes the second.
    """
    with caplog.at_level(logging.WARNING, logger=CLAMP_LOGGER):
        refused = narrow(_tool(), {"repository": "acme/widgets"}, {"repo_name": PROBED})

    assert isinstance(refused, ArgumentNotOffered)
    assert PROBED not in repr(refused)
    assert PROBED in caplog.text


def test_the_log_line_names_the_tool_and_the_dimension_that_was_probed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A value alone says nothing an operator can act on.

    The reader of this line is looking at one pod's log across every Session it serves.
    Without the tool and the dimension, a repeated attempt is an unattributed string:
    they cannot tell which registration governs it, which tenant vocabulary it belongs
    to, or whether two lines are one model probing one dimension or two models probing
    two.
    """
    with caplog.at_level(logging.WARNING, logger=CLAMP_LOGGER):
        narrow(_tool(), {"repository": "acme/widgets"}, {"repo_name": PROBED})

    assert "ask_question" in caplog.text
    assert "repository" in caplog.text


def test_a_scope_the_session_lacks_refuses_before_a_supplied_argument_does() -> None:
    """Two refusals can be true at once, and which one is returned is a decision.

    A call whose Scope cannot narrow it is refused for a reason that is about the
    Session rather than about the caller's arguments, and it holds however the call
    was written -- so it is answered first. The alternative reads worse than it
    sounds: a Session missing the dimension would be told to drop an argument, would
    drop it, and would then be refused again for the reason that was true all along.
    """
    refused = narrow(_tool(), {}, {"repo_name": PROBED})

    assert isinstance(refused, OutOfScope)
    assert refused.dimension == "repository"


def test_a_scope_dimension_no_binding_names_refuses_the_call() -> None:
    """The fail-open half, closed. This is the direction that ran wide silently.

    A Session scoped to a repository *and* an environment, calling a tool that binds
    only the repository, used to go out narrowed on the repository and wide on the
    environment -- across every environment the credential could reach, while the call
    looked fully narrowed at the point it left. The module's own docstring argued that
    exact case in the opposite direction and did not govern it.

    The refusal names the dimension nothing could hold the call to, which is the
    tenant's own word out of their own Session, and it names no value: the reason a
    Scope value may never ride a refusal does not change with which way the mismatch
    runs.
    """
    refused = narrow(
        _tool(),
        {"repository": "acme/widgets", "environment": "production"},
        {"question": "what is this"},
    )

    assert isinstance(refused, OutOfScope)
    assert refused.dimension == "environment"
    assert refused.cause is ScopeRefusal.DIMENSION_NOT_BOUND
    assert "production" not in repr(refused)
    assert "acme/widgets" not in repr(refused)


def test_both_directions_of_the_mismatch_come_back_as_one_shape() -> None:
    """Two causes, one type -- which is what keeps the caller's branch single.

    The ADR is explicit that this reuses `OutOfScope` rather than adding a type, and
    the reason is at the other end: a second refusal type is a second arm every caller
    has to grow, and a caller that forgets it falls through to whatever the default is.
    Here the default is sending the call. So the cause is a field on one shape, and the
    only thing that varies is the sentence the tenant reads.
    """
    lacks = narrow(_tool(), {}, {"question": "what"})
    unbound = narrow(_tool(), {"repository": "acme/widgets", "environment": "prod"}, {})

    assert isinstance(lacks, OutOfScope)
    assert isinstance(unbound, OutOfScope)
    assert lacks.cause is not unbound.cause


def test_a_dimension_the_tool_binds_is_answered_before_one_it_does_not() -> None:
    """Both causes are true at once here, and only one sentence can go back.

    A Session scoped `{topic}` calling a tool that binds `region` trips both: the
    binding's dimension is not in the Scope, and the Scope's dimension is not bound.
    That pair is not hypothetical -- it is what the live tier already asserts a refusal
    for, naming `region`, the dimension the *registration* declared.

    So the binding's own dimension is answered first, and the ordering is a decision
    rather than an accident of which loop runs. It keeps this new rule from re-labelling
    a refusal that already had a settled answer: every Session that was being refused
    yesterday is refused tomorrow for the same stated reason, and the new cause speaks
    only where the old one had nothing to say.
    """
    refused = narrow(
        _tool(
            ScopeBinding(dimension="region", argument="repo_name"),
            repo_name=ParameterType.STRING,
        ),
        {"topic": "general"},
        {"question": "what"},
    )

    assert isinstance(refused, OutOfScope)
    assert refused.dimension == "region"
    assert refused.cause is ScopeRefusal.DIMENSION_NOT_IN_SCOPE


def test_a_scope_that_cannot_hold_the_call_is_answered_before_the_arguments_are_read(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Session's shape settles before this particular call's shape, as before.

    A supplied bound argument is a fact about how one call was written; an unbindable
    Scope dimension is a fact about the Session and the registration, true of every
    call this Session will ever make to this tool. Answering the argument first would
    send a caller to fix the one thing that was not going to help: it would drop the
    argument, call again, and be refused for the reason that held all along.

    The log is asserted empty of the clamp's own line for the same reason: a refusal
    that never read the arguments has no attempt to record, and a line here would put
    a value in the log for a call whose arguments were never the problem.
    """
    with caplog.at_level(logging.WARNING, logger=CLAMP_LOGGER):
        refused = narrow(
            _tool(),
            {"repository": "acme/widgets", "environment": "prod"},
            {"repo_name": PROBED},
        )

    assert isinstance(refused, OutOfScope)
    assert refused.cause is ScopeRefusal.DIMENSION_NOT_BOUND
    assert PROBED not in caplog.text


def test_which_unbound_dimension_is_named_does_not_depend_on_how_it_was_written() -> (
    None
):
    """One Scope, two orders, one answer -- so a refusal is a fact, not a draw.

    A Session's Scope is stored as an ordered tuple of pairs, so two Sessions bounded
    to the very same thing can carry it written either way round. Naming whichever
    unbound dimension came first would make the sentence a tenant reads depend on the
    order their create call happened to serialize, and two Sessions with identical
    Scopes would be told to fix different things.
    """
    written_one_way = narrow(
        _tool(), {"repository": "acme/widgets", "alpha": "1", "zulu": "2"}, {}
    )
    written_the_other = narrow(
        _tool(), {"zulu": "2", "repository": "acme/widgets", "alpha": "1"}, {}
    )

    assert isinstance(written_one_way, OutOfScope)
    assert isinstance(written_the_other, OutOfScope)
    assert written_one_way.dimension == written_the_other.dimension


def test_an_argument_no_binding_names_is_carried_through_untouched() -> None:
    """The clamp narrows the bound arguments and is not a filter on the rest.

    Asserted because the failure it guards against passes every other test here: a
    function that returned only the bound arguments would narrow correctly and strip
    the model's actual question on the way.
    """
    narrowed = narrow(
        _tool(), {"repository": "acme/widgets"}, {"question": "what is this"}
    )

    assert not isinstance(narrowed, OutOfScope | ArgumentNotOffered)
    assert narrowed["question"] == "what is this"


def test_a_bound_argument_the_model_omitted_is_supplied_by_the_scope() -> None:
    """Written in, not merely overwritten.

    A clamp that only rewrote arguments already present would leave a model free to
    widen the call by omitting the bound one -- the argument would be absent from the
    outbound call and the server would apply its own default, which is the whole of
    the tenant's data on every tool this platform brokers.
    """
    narrowed = narrow(_tool(), {"repository": "acme/widgets"}, {"question": "what"})

    assert not isinstance(narrowed, OutOfScope | ArgumentNotOffered)
    assert narrowed["repo_name"] == "acme/widgets"


def test_the_arguments_the_caller_passed_are_not_mutated() -> None:
    """The caller's mapping is left as it was; the narrowed call is a new object.

    `McpProxy.call_tool` receives its arguments from the MCP server layer and the
    failure path re-reads them, so a clamp that wrote the Scope's value in place would
    make the record of what the model asked for say what the platform decided instead.
    The case is written around the argument the model *omitted*, which is the only
    write the clamp still makes: supplying it is now a refusal, and a refusal writes
    nothing anywhere.
    """
    asked: dict[str, object] = {"question": "what is this"}

    narrow(_tool(), {"repository": "acme/widgets"}, asked)

    assert asked == {"question": "what is this"}


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

    narrowed = narrow(_TWO_BINDINGS, scope, {"question": "what"})

    assert not isinstance(narrowed, OutOfScope | ArgumentNotOffered)
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
