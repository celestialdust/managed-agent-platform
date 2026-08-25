"""The model choice, followed from a Session's creation facts to the upstream reached.

Two Sessions are compiled from definitions differing in one field, and each one's Turn
is driven through the real Model Gateway to a recording upstream. The model each request
names is read out of that Session's compiled configuration, so a document naming the
wrong model fails here rather than agreeing with a constant this file also wrote.

Nothing is registered and no store is involved: the definition and the Session record
are built here, because what is under test is the path from a pinned definition to an
upstream, and a registry in the middle would only restate the definition registry's own
tests.

**The bearer is not minted here.** It is read out of the provider table in the
configuration the compiler just wrote, which is the same document a pod mounts, and
handed to the Gateway as `authorization`. That is the whole point of this file now.
The version before it hand-minted a token of a layout nothing in `src/` produced, so
it could prove a token verifies under its own key and nothing about whether a pod
ever holds one -- and the answer at the time was that it did not: the provider table
named `env_key = "MAP_POD_TOKEN"`, `deploy/k8s/session-pod.yaml` fills no such
variable, and against real `map-dev` the runtime sent the Gateway no request at all.

So a break anywhere along the hop fails here: the compiler not writing a header, writing
a token the verifier's key does not check out, writing one for the wrong Session, or the
two ends disagreeing about the layout. The one thing this still cannot prove is that the
real Agent Runtime reads `http_headers` off a provider table and attaches it --
`tests/pod/` owns that, against the real image.

The signing key is generated per run and lives only in this process. It stands in
for the Kubernetes Secret `map-tool-gateway/session-token-key`, which does exist in
the cluster and which the control plane, the Tool Gateway and now the Model Gateway
all read (ADR-023).
"""

import os
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.control.pod_config.compiler import compile_session_config
from managed_agent.control.pod_config.model_binding import (
    MODEL_PROVIDER_AUTH_HEADER,
    MODEL_PROVIDER_ID,
    ModelBindingViolation,
    bind_session,
    render_model_selection,
)
from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.registration.definition import (
    AgentDefinition,
    MultiAgentPosture,
)
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.session.session import SessionRecord
from managed_agent.gateway.model.router import (
    AuthScheme,
    InboundTurn,
    ModelGateway,
    RoutingEntry,
    RoutingTable,
    SessionTokenVerifier,
    UpstreamResponse,
    UpstreamWire,
    create_model_gateway_app,
)

GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"
TOOL_GATEWAY_URL = "https://tool-gateway.map.svc.cluster.local/mcp"
MODEL_A = "gpt-5-codex"
MODEL_B = "claude-sonnet-4-5-foundry"

UPSTREAM_A = "https://upstream-a.invalid"
UPSTREAM_B = "https://upstream-b.invalid"
"""The two upstreams the routing tables in this file send the two models to.

Named rather than repeated as literals so a test can assert *which* upstream a Turn
reached. A Gateway that resolved every model to one address still pairs each Session
with the right model, so an assertion that reads only the model cannot see it.
"""

NOW_MS = 1_760_000_000_000

# Generated per run and never persisted. Stands in for the Kubernetes Secret
# `map-tool-gateway/session-token-key`: the control plane signs a Session token with it
# and both gateways verify with it, so one key here is not a simplification of the
# deployment but a copy of its shape.
_SESSION_TOKEN_KEY = os.urandom(32)
_SESSION_TOKEN_EXPIRY_S = 4102444800

_A_TOKEN = "11111111-1111-1111-1111-111111111111.2222.9999999999.abc123"
"""A stand-in for the two cases that render a fragment without compiling a Session.

Deliberately not minted with `_SESSION_TOKEN_KEY`, because neither case that uses it
reads the token back: one asks whether the rendered `[agents]` table names a bound
model and the other asks whether a malformed `base_url` is refused, and both would
answer the same for any string. Minting a real one here would make those two cases look
like they proved something about the token, which the cases that compile a whole Session
already prove properly by reading the bearer out of the document.
"""
_RUNTIME_IMAGE = f"map-session@sha256:{'a' * 64}"


class _Clock:
    def now_epoch_ms(self) -> int:
        return NOW_MS


@dataclass
class _RecordingUpstream:
    """One wire handler, recording which Session asked for which model, and where."""

    seen: list[tuple[SessionId, str, str]] = field(default_factory=list)

    @asynccontextmanager
    async def open(
        self, turn: InboundTurn, entry: RoutingEntry
    ) -> AsyncIterator[UpstreamResponse]:
        self.seen.append((turn.caller.session_id, turn.model, entry.base_url))
        yield UpstreamResponse(
            status=200,
            headers=((b"content-type", b"text/event-stream"),),
            body=_one_event(),
        )


async def _one_event() -> AsyncIterator[bytes]:
    yield b"data: {}\n\n"


def _definition(model: str) -> AgentDefinition:
    """One definition body. Two Sessions differ in this one field and nothing else."""
    return AgentDefinition(
        name="slr-extractor",
        instructions="Extract findings and name the source document for each.",
        model=model,
        skills_repository="https://git.internal/skills.git",
        skills_revision="a" * 40,
        tool_servers=frozenset({"clinical-corpus"}),
        multiagent=MultiAgentPosture(enabled=True, max_depth=2),
    )


def _record(session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
    return SessionRecord(
        id=session_id,
        tenant_id=tenant_id,
        definition_id=DefinitionId(uuid4()),
        definition_revision="1",
        grant=frozenset({"clinical-corpus.search"}),
        scope=(("study_id", "S-4471"),),
        budget_minor_units=50_000,
        budget_currency="USD",
        retention_days=30,
    )


def _environment() -> Environment:
    """A registered shape that narrows nothing, so only the model varies here."""
    return Environment(
        id=EnvironmentId(uuid4()),
        tenant_id=TenantId(uuid4()),
        name="plain",
        runtime_image=_RUNTIME_IMAGE,
        denied_paths=(),
    )


def _compiled(model: str) -> tuple[SessionId, TenantId, str]:
    """A Session's id, its tenant, and the configuration its pod would start with."""
    session_id = SessionId(uuid4())
    tenant_id = TenantId(uuid4())
    record = _record(session_id, tenant_id)
    compiled = compile_session_config(
        record,
        tool_gateway_url=TOOL_GATEWAY_URL,
        model_gateway_url=GATEWAY_URL,
        environment=_environment(),
        definition=_definition(model),
        session_token_key=_SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=_SESSION_TOKEN_EXPIRY_S,
    )
    return session_id, tenant_id, compiled.config_toml


def _model_named_by(config_toml: str) -> str:
    return str(tomllib.loads(config_toml)["model"])


def _session_token_in(config_toml: str) -> str:
    """The token itself, with the scheme stripped off the header value."""
    scheme, _, token = _bearer_the_pod_would_send(config_toml).partition(" ")
    assert scheme == "Bearer", f"the header carries {scheme!r}, not a Bearer scheme"
    return token


def _bearer_the_pod_would_send(config_toml: str) -> str:
    """The `authorization` value a pod running this configuration puts on a model call.

    Read out of the document rather than rebuilt, so nothing this file wrote can
    stand in for what the compiler wrote. A missing table, a missing header or a
    renamed key raises a KeyError here, which is the correct outcome: a pod given such
    a document reaches the Model Gateway with no credential and every Turn fails.
    """
    providers = tomllib.loads(config_toml)["model_providers"]
    entry = providers[MODEL_PROVIDER_ID]
    return str(entry["http_headers"][MODEL_PROVIDER_AUTH_HEADER])


def _gateway(upstream: _RecordingUpstream) -> FastAPI:
    table = RoutingTable(
        (
            RoutingEntry(
                model=MODEL_A,
                wire=UpstreamWire.RESPONSES,
                base_url=UPSTREAM_A,
                auth_scheme=AuthScheme.BEARER,
                credential_name="map/dev/providers/upstream-a",
            ),
            RoutingEntry(
                model=MODEL_B,
                wire=UpstreamWire.RESPONSES,
                base_url=UPSTREAM_B,
                auth_scheme=AuthScheme.API_KEY,
                credential_name="map/dev/providers/upstream-b",
            ),
        )
    )
    return create_model_gateway_app(
        ModelGateway(
            table=table,
            handlers={UpstreamWire.RESPONSES: upstream},
            tokens=SessionTokenVerifier(key=_SESSION_TOKEN_KEY, clock=_Clock()),
        )
    )


async def _turn(app: FastAPI, config_toml: str) -> httpx.Response:
    """One model call, made the way a pod holding this configuration would make it.

    The model and the bearer both come out of the document, so this takes no arguments
    that could disagree with it.
    """
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://gateway.invalid"
    ) as client:
        return await client.post(
            "/v1/responses",
            json={
                "model": _model_named_by(config_toml),
                "input": [{"type": "text", "text": "go"}],
            },
            headers={
                MODEL_PROVIDER_AUTH_HEADER: _bearer_the_pod_would_send(config_toml)
            },
        )


def test_the_compiled_configuration_names_only_the_model_its_session_bound() -> None:
    _, _, config_a = _compiled(MODEL_A)
    _, _, config_b = _compiled(MODEL_B)

    assert _model_named_by(config_a) == MODEL_A
    assert _model_named_by(config_b) == MODEL_B
    assert MODEL_B not in config_a
    assert MODEL_A not in config_b


def test_the_provider_block_points_the_pod_at_the_model_gateway() -> None:
    _, _, config = _compiled(MODEL_A)
    document = tomllib.loads(config)
    providers = document["model_providers"]

    assert list(providers) == [MODEL_PROVIDER_ID]
    assert document["model_provider"] == MODEL_PROVIDER_ID
    assert providers[MODEL_PROVIDER_ID]["base_url"] == GATEWAY_URL
    assert providers[MODEL_PROVIDER_ID]["wire_api"] == "responses"
    assert providers[MODEL_PROVIDER_ID]["http_headers"] == {
        MODEL_PROVIDER_AUTH_HEADER: f"Bearer {_session_token_in(config)}"
    }
    assert "openai_base_url" not in config
    assert "chatgpt_base_url" not in config


def test_the_url_the_runtime_builds_from_that_block_is_the_one_this_app_serves() -> (
    None
):
    """The two ends of the model hop, compared rather than each asserted alone.

    The runtime joins its base URL to the literal segment `responses`, trimming one
    trailing slash and nothing else. The route the Gateway registers is the other half,
    and it is read off the app rather than written down again -- a base URL and a route
    that each look right separately are what a 404 from a healthy service is made of.
    """
    _, _, config = _compiled(MODEL_A)
    base = str(tomllib.loads(config)["model_providers"][MODEL_PROVIDER_ID]["base_url"])
    built = f"{base.rstrip('/')}/responses"
    served = create_model_gateway_app(None).openapi()["paths"]  # type: ignore[arg-type]

    assert built.endswith("/v1/responses")
    assert built[len("http://model-gateway.map-dev.svc.cluster.local") :] in served


def test_a_subagent_spawned_with_no_model_falls_back_to_a_bound_one() -> None:
    record = _record(SessionId(uuid4()), TenantId(uuid4()))
    bindings = bind_session(record, _definition(MODEL_A))
    document = tomllib.loads(
        render_model_selection(
            bindings, gateway_base_url=GATEWAY_URL, session_token=_A_TOKEN
        )
    )

    assert bindings.permits(document["agents"]["default_subagent_model"])
    assert document["agents"]["enabled"] is True


def test_the_model_selection_leaves_the_configuration_floors_standing() -> None:
    _, _, config = _compiled(MODEL_A)
    document = tomllib.loads(config)

    assert "sandbox_mode" not in config
    assert "sandbox_workspace_write" not in config
    assert len(document["mcp_servers"]) == 1


@pytest.mark.parametrize(
    "base_url",
    [
        "model-gateway.map-dev.svc.cluster.local/v1",
        "http://model-gateway.map-dev.svc.cluster.local",
        "http://model-gateway.map-dev.svc.cluster.local/",
        "http://model-gateway.map-dev.svc.cluster.local/v2/",
    ],
)
def test_a_base_url_that_would_miss_the_responses_path_is_refused(
    base_url: str,
) -> None:
    record = _record(SessionId(uuid4()), TenantId(uuid4()))
    bindings = bind_session(record, _definition(MODEL_A))

    with pytest.raises(ModelBindingViolation):
        render_model_selection(
            bindings, gateway_base_url=base_url, session_token=_A_TOKEN
        )


async def test_each_turn_is_served_by_the_model_its_session_named() -> None:
    session_a, tenant_a, config_a = _compiled(MODEL_A)
    session_b, tenant_b, config_b = _compiled(MODEL_B)
    upstream = _RecordingUpstream()
    app = _gateway(upstream)

    first = await _turn(app, config_a)
    second = await _turn(app, config_b)

    assert (first.status_code, second.status_code) == (200, 200)
    assert upstream.seen == [
        (session_a, MODEL_A, UPSTREAM_A),
        (session_b, MODEL_B, UPSTREAM_B),
    ]


async def test_a_turn_whose_bearer_was_signed_by_another_key_reaches_no_upstream() -> (
    None
):
    """The positive control for every 200 above: the same request, one input changed."""
    session_id, tenant_id, config = _compiled(MODEL_A)
    upstream = _RecordingUpstream()
    app = create_model_gateway_app(
        ModelGateway(
            table=RoutingTable(
                (
                    RoutingEntry(
                        model=MODEL_A,
                        wire=UpstreamWire.RESPONSES,
                        base_url=UPSTREAM_A,
                        auth_scheme=AuthScheme.BEARER,
                        credential_name="map/dev/providers/upstream-a",
                    ),
                )
            ),
            handlers={UpstreamWire.RESPONSES: upstream},
            tokens=SessionTokenVerifier(key=os.urandom(32), clock=_Clock()),
        )
    )

    response = await _turn(app, config)

    assert response.status_code == 401
    assert upstream.seen == []


async def test_an_undeclared_model_fails_the_turn_rather_than_falling_back() -> None:
    session_id, tenant_id, config = _compiled("a-model-nobody-declared")
    upstream = _RecordingUpstream()
    app = _gateway(upstream)

    response = await _turn(app, config)

    assert response.status_code == 404
    assert "a-model-nobody-declared" in response.json()["error"]["message"]
    assert upstream.seen == []
