"""The baked catalogue is what makes `multiagent.enabled` do anything.

The runtime resolves a subagent's model by **exact** match against a model catalogue
and refuses the spawn when the name is absent. Its own catalogue is compiled into its
binary, names eight models, and names none of this platform's -- so before this work the
field was accepted, stored, and inert: an agent asked to delegate was refused five times
out of five and did the whole task itself.

The document itself is built into the runtime image by `tools/bake_model_catalog.py`,
which asserts everything about its *contents* at build time, and
`tests/deploy/test_session_image_runs_both_halves.py` guards the two facts that span
this module and the image -- that the Dockerfile bakes to the path named here, and that
no volume mounts over it.

So what is left for this module is the one half neither of those can see: that a
compiled Session actually points the runtime at the catalogue. That failure is the
quietest of the three, because nothing about it looks broken -- the pod starts, the
Session runs, and every spawn is refused, which is precisely the defect this work exists
to remove, returning under a different cause.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from managed_agent.control.pod_config.compiler import (
    RUNTIME_CATALOG_PATH,
    CompiledConfig,
    FloorViolation,
    check_floors,
    compile_session_config,
)
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.registration.definition import (
    AgentDefinition,
    MultiAgentPosture,
)
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord

GATEWAY_URL = "https://tool-gateway.map.internal/mcp"
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"
_TOKEN_KEY = b"a signing key that is thirty-two"
_EXPIRY = 4102444800
_MODEL = "gpt-5-codex"
_ROOT = Path(__file__).resolve().parents[2]


def _definition(*, delegates: bool) -> AgentDefinition:
    return AgentDefinition(
        name="slr-reviewer",
        instructions="Extract findings and name the source for each.",
        model=_MODEL,
        skills_repository="git@github.com:acme/skills.git",
        skills_revision="0" * 39 + "a",
        multiagent=MultiAgentPosture(enabled=delegates),
    )


def _compiled(*, delegates: bool = True) -> CompiledConfig:
    return compile_session_config(
        SessionRecord(
            id=new_session_id(),
            tenant_id=TenantId(uuid4()),
            definition_id=new_definition_id(),
            definition_revision="rev-1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=10_000,
            budget_currency="USD",
            retention_days=30,
        ),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=_definition(delegates=delegates),
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="fixture",
            runtime_image="registry.map.internal/session@sha256:" + "a" * 64,
            denied_paths=(),
        ),
        session_token_key=_TOKEN_KEY,
        session_token_expiry_epoch_s=_EXPIRY,
    )


@pytest.mark.parametrize("delegates", [True, False])
def test_every_session_points_the_runtime_at_the_catalogue(delegates: bool) -> None:
    """The managed document names the catalogue whether or not this Session delegates.

    Unconditional on purpose, and both halves are parametrized so neither can regress
    alone. The runtime only reads the catalogue because this line names it, and the
    catalogue is where the raised output cap lives -- a per-entry field -- so a Session
    compiled without the line keeps the runtime's 10,000-token default and leaves the
    Evidence capture's margin undefined even though it never spawns anything (ADR-020).
    """
    compiled = _compiled(delegates=delegates)
    assert f"model_catalog_json = {RUNTIME_CATALOG_PATH!r}".replace("'", '"') in (
        compiled.requirements_toml
    )


def test_the_catalogue_line_is_a_requirement_not_a_default() -> None:
    """It rides `requirements.toml`, which sets the value *and* pins it.

    `config.toml` would be a starting point a later layer inside the pod could replace,
    and replacing it swaps the list of models the agent may delegate to. The runtime
    applies requirements to the parsed configuration before it loads the catalogue and
    then refuses any layer that disagrees, which is why the distinction is load-bearing
    rather than stylistic.
    """
    compiled = _compiled()
    assert "model_catalog_json" in compiled.requirements_toml
    assert "model_catalog_json" not in compiled.config_toml


def test_a_session_that_does_not_name_the_catalogue_is_refused() -> None:
    """The floor fires, so a Session that would silently refuse every spawn cannot ship.

    Written by removing the line from a real compilation rather than by hand-building a
    document, so the case stays reachable if the emitter's spelling changes.
    """
    compiled = _compiled()
    stripped = re.sub(r"(?m)^model_catalog_json = .*\n", "", compiled.requirements_toml)
    assert stripped != compiled.requirements_toml
    with pytest.raises(FloorViolation, match="model catalogue"):
        check_floors(replace(compiled, requirements_toml=stripped))


def test_the_compiler_carries_no_catalogue_of_its_own() -> None:
    """No per-Session copy, so the Secret it used to compete for is untouched.

    The document runs to a few hundred kilobytes and a Kubernetes Secret is capped at
    1 MiB across every key together, shared with this Session's skill bundle. Baking it
    removes that competition outright rather than bounding it, and this asserts the
    removal: a `CompiledConfig` field holding the catalogue would put the bytes back.
    """
    compiled = _compiled()
    fields: dict[str, Any] = {
        name: getattr(compiled, name) for name in compiled.__slots__
    }
    assert not [
        name
        for name, value in fields.items()
        if isinstance(value, str) and '"models"' in value
    ], "a compiled field carries the catalogue, which the image is supposed to hold"


def test_the_image_bakes_the_catalogue_where_the_compiler_names_it() -> None:
    """The image build and the compiled document have to mean the same file.

    Nothing links them at runtime. The bake script takes its destination as an argument
    and imports nothing from this package, so agreement rests entirely on the value the
    Dockerfile passes -- and a disagreement is not a degraded Session but a pod that
    dies at configuration load, because the runtime reads this path while loading its
    configuration and treats an unreadable one as fatal.

    So the Dockerfile's own `--out` is parsed rather than a second copy of the path
    being asserted here. Pinning the constant against a literal would pass happily
    while the build wrote somewhere else, which is the only failure this needs to catch.
    """
    dockerfile = _ROOT / "deploy" / "docker" / "session.Dockerfile"
    baked = re.findall(r"--out\s+(\S+)", dockerfile.read_text())
    assert baked, f"{dockerfile} runs no bake step with an --out path"
    assert set(baked) == {RUNTIME_CATALOG_PATH}, (
        f"the image bakes the catalogue to {sorted(set(baked))} while the compiled "
        f"requirements name {RUNTIME_CATALOG_PATH!r}; the runtime reads that path "
        "while loading its configuration, so every pod on this image fails to start"
    )


def test_no_volume_mounts_over_the_catalogue() -> None:
    """The catalogue's path must not sit under anything the pod mounts.

    This is the whole reason the path is `/opt/codex` and not `/etc/codex`. A Secret
    volume replaces its mount point's contents wholesale, and the requirements Secret is
    mounted at `/etc/codex` -- so a file baked into the image at that path is invisible
    to the process that needs it. Invisible in the worst way, too: the runtime's own
    error is that the path does not exist, which reads exactly like an image build that
    never ran, and the mistake is invisible in a diff because both halves look right on
    their own.
    """
    manifest = yaml.safe_load(
        (_ROOT / "deploy" / "k8s" / "session-pod.yaml").read_text()
    )
    runtime = [
        c for c in manifest["spec"]["containers"] if c["name"] == "agent-runtime"
    ]
    assert runtime, "the manifest declares no 'agent-runtime' container"
    catalogue = Path(RUNTIME_CATALOG_PATH)
    for mount in (str(m["mountPath"]) for m in runtime[0]["volumeMounts"]):
        covered = Path(mount) == catalogue or Path(mount) in catalogue.parents
        assert not covered, (
            f"{mount!r} is mounted over {RUNTIME_CATALOG_PATH!r}, so the baked "
            "catalogue is unreadable and the runtime fails at configuration load"
        )
