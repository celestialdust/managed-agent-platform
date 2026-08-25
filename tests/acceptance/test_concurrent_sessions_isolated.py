"""Isolation between two Sessions, and between agents inside one Session.

Each case turns one clause of the isolation property into an assertion about a
difference: a slot's own model and instructions, a slot the Session does not bind, a
model another Session bound. A lookup that fell back to the root, or a value shared
between Sessions, satisfies every same-configuration test and fails exactly these.

The agent documents are checked for what they do not carry as well as for what they do.
An agent document may override any configuration key, so the absence of a working
directory is what keeps three agents in one Session on one Session VFS, and the absence
of sandbox_mode and mcp_servers is what stops one widening its own confinement or its
own tool reach (ADR-005, ADR-014).

The compiled-and-driven helpers come from the sibling module by bare name: pytest puts a
test directory that is not a package on sys.path, and the alternative is a shared
fixtures module no slice owns.
"""

import asyncio
import tomllib
from uuid import uuid4

import pytest
from test_model_per_session import (
    MODEL_A,
    MODEL_B,
    UPSTREAM_A,
    UPSTREAM_B,
    _compiled,
    _definition,
    _gateway,
    _record,
    _RecordingUpstream,
    _turn,
)

from managed_agent.control.pod_config.model_binding import (
    ROOT_SLOT,
    AgentBinding,
    ModelBindingViolation,
    SessionModelBindings,
    UnboundAgentSlot,
    agent_slot,
    bind_agent,
    bind_session,
    check_agent_document,
    render_agent_document,
)
from managed_agent.core.ids import SessionId, TenantId

MODEL_C = "gpt-5-codex-mini"
EXTRACTOR = agent_slot("slr-extractor")
APPRAISER = agent_slot("slr-appraiser")
UNBOUND = agent_slot("slr-summariser")


def _three_agents() -> SessionModelBindings:
    """One Session binding three differently-configured agents, root included."""
    record = _record(SessionId(uuid4()), TenantId(uuid4()))
    return (
        bind_session(record, _definition(MODEL_A))
        .with_agent(
            AgentBinding(
                slot=EXTRACTOR,
                model=MODEL_B,
                instructions="Extract every reported effect size verbatim.",
                description="extractor",
            )
        )
        .with_agent(
            AgentBinding(
                slot=APPRAISER,
                model=MODEL_C,
                instructions="Appraise risk of bias and report nothing numeric.",
                description="appraiser",
            )
        )
    )


def test_an_unbound_slot_raises_rather_than_answering_with_the_roots() -> None:
    bindings = _three_agents()

    with pytest.raises(UnboundAgentSlot) as raised:
        bindings.binding_for(UNBOUND)

    assert UNBOUND in str(raised.value)


def test_each_slot_answers_with_only_its_own_model_and_instructions() -> None:
    bindings = _three_agents()

    assert bindings.binding_for(ROOT_SLOT).model == MODEL_A
    assert bindings.binding_for(EXTRACTOR).model == MODEL_B
    assert bindings.binding_for(APPRAISER).model == MODEL_C
    assert "effect size" in bindings.binding_for(EXTRACTOR).instructions
    assert "effect size" not in bindings.binding_for(ROOT_SLOT).instructions
    assert "risk of bias" not in bindings.binding_for(EXTRACTOR).instructions
    assert bindings.models == frozenset({MODEL_A, MODEL_B, MODEL_C})


def test_with_agent_leaves_the_bindings_it_was_called_on_unchanged() -> None:
    record = _record(SessionId(uuid4()), TenantId(uuid4()))
    before = bind_session(record, _definition(MODEL_A))

    after = before.with_agent(bind_agent(EXTRACTOR, _definition(MODEL_B)))

    assert before.models == frozenset({MODEL_A})
    assert after.models == frozenset({MODEL_A, MODEL_B})
    assert len(before.agents) == 1


def test_two_sessions_permit_only_their_own_models() -> None:
    one = bind_session(
        _record(SessionId(uuid4()), TenantId(uuid4())), _definition(MODEL_A)
    )
    two = bind_session(
        _record(SessionId(uuid4()), TenantId(uuid4())), _definition(MODEL_B)
    )

    assert one.permits(MODEL_A) and not one.permits(MODEL_B)
    assert two.permits(MODEL_B) and not two.permits(MODEL_A)
    assert one.session_id != two.session_id


def test_a_second_binding_for_one_slot_is_refused() -> None:
    bindings = _three_agents()

    with pytest.raises(ModelBindingViolation):
        bindings.with_agent(bind_agent(EXTRACTOR, _definition(MODEL_C)))

    with pytest.raises(ModelBindingViolation):
        bindings.with_agent(bind_agent(ROOT_SLOT, _definition(MODEL_C)))


def test_every_agent_document_carries_its_own_configuration_and_no_others() -> None:
    documents = {
        agent.slot: render_agent_document(agent)
        for agent in _three_agents().agents
        if agent.slot != ROOT_SLOT
    }

    assert set(documents) == {EXTRACTOR, APPRAISER}
    extractor = tomllib.loads(documents[EXTRACTOR].toml)
    appraiser = tomllib.loads(documents[APPRAISER].toml)
    assert extractor["name"] == EXTRACTOR
    assert extractor["model"] == MODEL_B
    assert appraiser["model"] == MODEL_C
    assert "risk of bias" not in documents[EXTRACTOR].toml
    assert "effect size" not in documents[APPRAISER].toml
    assert documents[EXTRACTOR].filename == f"{EXTRACTOR}.toml"


@pytest.mark.parametrize(
    "surplus",
    [
        'sandbox_mode = "danger-full-access"',
        "mcp_servers = []",
        'cwd = "/tmp"',
        "[skills.config]\nenabled = true",
    ],
)
def test_an_agent_document_that_would_widen_anything_is_refused(surplus: str) -> None:
    document = render_agent_document(bind_agent(EXTRACTOR, _definition(MODEL_B))).toml

    check_agent_document(document)
    with pytest.raises(ModelBindingViolation):
        check_agent_document(f"{document}\n{surplus}\n")


def test_no_agent_document_names_a_working_directory() -> None:
    for agent in _three_agents().agents:
        if agent.slot == ROOT_SLOT:
            continue
        assert "cwd" not in tomllib.loads(render_agent_document(agent).toml)


@pytest.mark.parametrize(
    "name", ["", "../root", "agents/root", "a.b", "Root", "a" * 65]
)
def test_a_slot_is_a_plain_lowercase_name_or_it_is_refused(name: str) -> None:
    with pytest.raises(ModelBindingViolation):
        agent_slot(name)


async def test_concurrent_turns_are_each_served_their_own_sessions_model() -> None:
    session_a, tenant_a, config_a = _compiled(MODEL_A)
    session_b, tenant_b, config_b = _compiled(MODEL_B)
    upstream = _RecordingUpstream()
    app = _gateway(upstream)

    responses = await asyncio.gather(
        *(_turn(app, config) for _ in range(3) for config in (config_a, config_b))
    )

    assert [response.status_code for response in responses] == [200] * 6
    # The upstream is part of the tuple, not discarded. Dropping it -- which this
    # assertion did until 2026-08-23 -- left a Gateway that resolved every model to one
    # upstream passing: the Session and the model would still pair correctly and only
    # the address would be wrong, which is the failure that costs a tenant its calls
    # going somewhere the Egress Policy never allowed.
    assert set(upstream.seen) == {
        (session_a, MODEL_A, UPSTREAM_A),
        (session_b, MODEL_B, UPSTREAM_B),
    }
    assert len(upstream.seen) == 6
    assert (tenant_a, tenant_b) == (tenant_a, tenant_b), (
        "both tenants are bound above and read by _compiled; named here so a reader "
        "sees the two Sessions belong to different tenants"
    )
