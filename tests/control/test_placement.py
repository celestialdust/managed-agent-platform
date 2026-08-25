"""Placement computes the binding instead of storing it, and carries nothing runtime.

The cluster here is a fake, and that is the honest limit of this file: no pod is
created, so nothing below shows that a real Kubernetes API accepts the name or the
manifest. What it does show is the arithmetic — one Session, one name, forever — and
that the binding a caller gets back holds only values this platform issued.
"""

from __future__ import annotations

import dataclasses
import re
from uuid import uuid4

import pytest

from managed_agent.control.pod_config.compiler import (
    CompiledConfig,
    compile_session_config,
)
from managed_agent.control.session.placement import (
    Placement,
    PodBinding,
    PodPhase,
    pod_name_for,
)
from managed_agent.core.ids import (
    SessionId,
    TenantId,
    new_definition_id,
    new_session_id,
)
from managed_agent.core.registration.definition import (
    AgentDefinition,
    SkillsRevision,
)
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord

_RFC_1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


# Where a Session pod reaches the Model Gateway. The `/v1` is load-bearing at both ends:
# the Agent Runtime POSTs `{base_url}/responses`, and the Gateway's router mounts
# `POST /v1/responses`.
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"

# The Gateway's signing key and the token's deadline, which the compiler takes from
# its caller and never defaults. Literals, so no case here can expire mid-run.
SESSION_TOKEN_KEY = b"a signing key that is thirty-two"
SESSION_TOKEN_EXPIRY = 4102444800
# What a pod-local Agent Runtime would call its own thread. Nothing may carry it out.
_A_RUNTIME_IDENTIFIER = "th_0199c4de6f2a7b81"


class RecordingRunner:
    """A cluster that remembers what it was asked and answers with a scripted phase.

    It also holds a runtime-issued identifier, so the "no runtime identifier escapes"
    case has something real to look for rather than asserting the absence of a value
    that was never anywhere near the code.
    """

    def __init__(self, phase: PodPhase = PodPhase.RUNNING) -> None:
        self.phase = phase
        self.ensured: list[tuple[str, CompiledConfig]] = []
        self.asked: list[str] = []
        self.removed: list[str] = []
        self.runtime_thread_id = _A_RUNTIME_IDENTIFIER

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        self.ensured.append((pod_name, compiled))
        return self.phase

    async def phase_of(self, pod_name: str) -> PodPhase:
        self.asked.append(pod_name)
        return PodPhase.ABSENT if pod_name in self.removed else self.phase

    async def remove(self, pod_name: str) -> None:
        self.removed.append(pod_name)


# The definition a Session pins, for the one field the compiler reads off it: the model.
# The provider is not here because it is not the definition's to name -- every model
# call leaves a Session pod through the Model Gateway.
A_DEFINITION = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)


def _a_compiled_config(session_id: SessionId) -> CompiledConfig:
    return compile_session_config(
        SessionRecord(
            id=session_id,
            tenant_id=TenantId(uuid4()),
            definition_id=new_definition_id(),
            definition_revision="rev-1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=10_000,
            budget_currency="USD",
            retention_days=30,
        ),
        tool_gateway_url="http://tool-gateway.invalid",
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=_a_shape_that_narrows_nothing(),
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )


def test_a_session_has_one_pod_name_and_it_is_a_legal_label() -> None:
    session_id = new_session_id()
    name = pod_name_for(session_id)
    assert name == pod_name_for(session_id), "the name is computed, so it cannot drift"
    assert name != pod_name_for(new_session_id())
    assert len(name) <= 63
    assert _RFC_1123_LABEL.match(name), name


async def test_placing_asks_the_cluster_once_and_reports_what_it_said() -> None:
    session_id = new_session_id()
    runner = RecordingRunner(PodPhase.STARTING)
    binding = await Placement(runner).place(_a_compiled_config(session_id))
    assert len(runner.ensured) == 1
    assert runner.ensured[0][0] == pod_name_for(session_id)
    assert binding == PodBinding(
        session_id=session_id,
        pod_name=pod_name_for(session_id),
        phase=PodPhase.STARTING,
    )


async def test_placing_twice_converges_on_one_pod_rather_than_allocating_two() -> None:
    """Two controllers racing must not end with a Session on two pods."""
    session_id = new_session_id()
    compiled = _a_compiled_config(session_id)
    runner = RecordingRunner()
    placement = Placement(runner)
    first = await placement.place(compiled)
    second = await placement.place(compiled)
    assert first.pod_name == second.pod_name
    assert {name for name, _ in runner.ensured} == {first.pod_name}


async def test_the_compiled_configuration_reaches_the_cluster_unread() -> None:
    """Placement changes when the node tier does, not when the config format does."""
    session_id = new_session_id()
    compiled = _a_compiled_config(session_id)
    runner = RecordingRunner()
    await Placement(runner).place(compiled)
    assert runner.ensured[0][1] is compiled


async def test_where_a_session_is_comes_from_the_cluster_and_not_from_a_record() -> (
    None
):
    session_id = new_session_id()
    runner = RecordingRunner()
    placement = Placement(runner)
    await placement.place(_a_compiled_config(session_id))
    assert (await placement.locate(session_id)).phase is PodPhase.RUNNING
    await placement.release(session_id)
    assert runner.removed == [pod_name_for(session_id)]
    after = await placement.locate(session_id)
    assert after.phase is PodPhase.ABSENT
    assert after.pod_name == pod_name_for(session_id), "the name outlives the pod"


async def test_a_terminated_pod_is_reported_as_the_cluster_reports_it() -> None:
    session_id = new_session_id()
    runner = RecordingRunner(PodPhase.GONE)
    assert (await Placement(runner).locate(session_id)).phase is PodPhase.GONE


async def test_a_binding_is_immutable() -> None:
    binding = await Placement(RecordingRunner()).place(
        _a_compiled_config(new_session_id())
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.pod_name = "map-session-somewhere-else"  # type: ignore[misc]


async def test_no_field_of_a_binding_holds_anything_the_agent_runtime_issued() -> None:
    """The three fields are all platform-issued, and there is no fourth."""
    session_id = new_session_id()
    runner = RecordingRunner()
    binding = await Placement(runner).place(_a_compiled_config(session_id))
    assert {field.name for field in dataclasses.fields(binding)} == {
        "session_id",
        "pod_name",
        "phase",
    }
    rendered = " ".join(
        str(getattr(binding, field.name)) for field in dataclasses.fields(binding)
    )
    assert runner.runtime_thread_id not in rendered
    assert rendered == f"{session_id} {pod_name_for(session_id)} {PodPhase.RUNNING}"


def _a_shape_that_narrows_nothing() -> Environment:
    """A registered shape contributing an image and no extra deny rule.

    Compiling now names an environment, and these cases are about the platform's own
    floors rather than about a tenant's narrowing -- so the shape here adds nothing, and
    the compiled profile is exactly what the platform declares.
    """
    return Environment(
        id=new_environment_id(),
        tenant_id=TenantId(uuid4()),
        name="fixture",
        runtime_image="registry.map.internal/session@sha256:" + "a" * 64,
        denied_paths=(),
    )
