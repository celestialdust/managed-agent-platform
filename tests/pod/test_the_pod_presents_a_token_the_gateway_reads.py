"""Whether the compiled document actually makes the runtime send the token.

Nothing else in the tree can answer this, and the reason is measured: `codex
app-server` connects to no MCP server at start-up, so a Session pod's startupProbe,
its readiness and a `kubectl exec` into it all pass on a pod whose header is wrong.
Parsing the document proves less than nothing here -- the runtime accepts a misspelled
key, drops a value it cannot put in a header, and sends the request either way, and
the Gateway answers all three with the same fixed 401 it answers a request that
carried nothing.

So this drives the real `codex` binary, from the real Session image, against the real
`SessionTokenMiddleware`, and reads the wire. `codex mcp list --json` performs an
authenticated probe request to the configured URL carrying the configured headers, which
is what makes this reachable with no model credential, no OAuth and no cluster.

It carries `@pytest.mark.image`: `addopts` deselects it from the default run and
`pytest -m image` selects it. READ THAT AS: A DEFAULT RUN OF THIS FILE SAYS NOTHING
ABOUT THE WIRE. It needs a Docker daemon and it builds the Session image.

The container runs as uid 10001 with `/etc/codex` bind-mounted read-only, which is what
the pod does, and the requirements document is the one the compiler renders -- because
that document carries the managed MCP allowlist, and a header added to `config.toml`
must not make the identity stop matching. Two cases below are a matched pair on exactly
that: one asserts the server is still enabled, and one changes the identity URL and
asserts it is disabled. Without the second, "enabled" would prove only that the
allowlist is never read -- the reading that already cost this project a wrong
conclusion once.

What this file cannot show is a Turn. Nothing in `src/` calls the compiler, no pod
runner is wired into `composition.build()`, and a real tool call needs a model
credential. A green run here says the document is right, not that a Session can use a
tool.
"""

from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest

from managed_agent.control.pod_config.compiler import (
    GATEWAY_SERVER_ID,
    CompiledConfig,
    compile_session_config,
)
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.registration.definition import AgentDefinition, SkillsRevision
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER_NAME,
    mint_session_token,
)

_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DOCKERFILE: Final[Path] = _ROOT / "deploy" / "docker" / "session.Dockerfile"

CODEX_VERSION: Final[str] = "0.149.0"
PLATFORM: Final[str] = "linux/amd64"
IMAGE_TAG: Final[str] = "map-session:map66"

TOKEN_KEY: Final[bytes] = b"a signing key that is thirty-two"
OTHER_KEY: Final[bytes] = b"a different key, thirty-two bytes"

# Absolute seconds, not offsets from a clock. `_FUTURE` outlives any run of this suite
# and `_PAST` is behind every one, so neither case can flip on the day it is run.
_FUTURE: Final[int] = 4102444800
_PAST: Final[int] = 1000000000

# The listener the container talks to, inside the container's own network namespace. The
# port is arbitrary; what matters is that the URL the document names and the URL the
# middleware listens on are one string, built from these two constants.
_PORT: Final[int] = 18080
GATEWAY_URL: Final[str] = f"http://127.0.0.1:{_PORT}/mcp"
MODEL_GATEWAY_URL: Final[str] = "http://model-gateway.invalid/v1"

A_DEFINITION: Final[AgentDefinition] = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)

GATE = f'''
"""The Tool Gateway's admission check, in front of a recorder that answers 200.

Imported from the installed package inside the image, so what refuses a request here is
the same class the deployed Gateway wraps its MCP app in -- not a re-implementation of
it, which would grade this file against itself.
"""

import sys

import uvicorn

from managed_agent.gateway.tool.server import SessionTokenMiddleware


async def recorder(scope, receive, send):
    """Record that the middleware let a request through, and answer 200."""
    if scope["type"] != "http":
        return
    print(f"PASSED {{scope['method']}} {{scope['path']}}", flush=True)
    body = b'{{"ok":true}}'
    await send(
        {{
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }}
    )
    await send({{"type": "http.response.body", "body": body}})


uvicorn.run(
    SessionTokenMiddleware(recorder, sys.argv[1].encode()),
    host="127.0.0.1",
    port={_PORT},
    log_level="info",
)
'''

# One shell program for every case, so no case differs from another in how it waits, how
# it copies, or where it reads. `set -eu` deliberately does NOT cover the codex call: a
# refused probe is an outcome to read, not an error to abort on.
_IN_CONTAINER: Final[str] = """
set -eu
export CODEX_HOME=/tmp/ch
mkdir -p "$CODEX_HOME"
cp /etc/codex/config.toml "$CODEX_HOME/config.toml"
python /etc/codex/gate.py "$1" > /tmp/gate.log 2>&1 &
i=0
while [ "$i" -lt 60 ]; do
  grep -q "Uvicorn running" /tmp/gate.log && break
  i=$((i + 1))
  sleep 0.25
done
if ! grep -q "Uvicorn running" /tmp/gate.log; then
  echo "GATE_NEVER_STARTED"
  cat /tmp/gate.log
  exit 3
fi
set +e
codex mcp list --json > /tmp/mcp.json 2>&1
echo "CODEX_RC=$?"
echo "--- MCP LIST ---"
cat /tmp/mcp.json
echo "--- GATE LOG ---"
cat /tmp/gate.log
"""


def _record() -> SessionRecord:
    return SessionRecord(
        id=new_session_id(),
        tenant_id=TenantId(uuid4()),
        definition_id=new_definition_id(),
        definition_revision="rev-1",
        grant=frozenset(),
        scope=(),
        budget_minor_units=10_000,
        budget_currency="USD",
        retention_days=30,
    )


def _compiled(*, key: bytes = TOKEN_KEY, expiry: int = _FUTURE) -> CompiledConfig:
    """One Session's real documents, from the real compiler.

    The key and the expiry are parameters because two of the cases below need a token
    this compiler would happily produce and the Gateway must still refuse -- one signed
    by another key, one already dead. Neither is a mutation of the rendered text: they
    are what the compiler emits when its caller passes the wrong thing, which is the
    failure actually worth grading.
    """
    return compile_session_config(
        _record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="fixture",
            runtime_image="registry.map.internal/session@sha256:" + "a" * 64,
            denied_paths=(),
        ),
        definition=A_DEFINITION,
        session_token_key=key,
        session_token_expiry_epoch_s=expiry,
    )


@dataclass(frozen=True, slots=True)
class Wire:
    """What one container run showed: what codex made of the server, what arrived."""

    stdout: str

    @property
    def reached_the_recorder(self) -> bool:
        """Whether the middleware let anything through to the app behind it."""
        return "PASSED GET /mcp" in self.stdout

    @property
    def refusals(self) -> int:
        return self.stdout.count("401 Unauthorized")

    @property
    def enabled(self) -> bool:
        return '"enabled": true' in self.stdout

    @property
    def disabled_reason(self) -> str | None:
        for line in self.stdout.splitlines():
            if '"disabled_reason"' in line:
                return line.split(":", 1)[1].strip().rstrip(",").strip('"')
        return None


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=1800)


@pytest.fixture(scope="module")
def image() -> str:
    """Build the Session image from the working tree and yield its tag.

    Built here rather than reused from another module's tag because this file's whole
    subject is a module that has to be *inside* the image: a stale image would import an
    older `managed_agent`, and the middleware under test would not be the one in this
    diff.

    --platform linux/amd64 because that is what the nodegroup runs, so the binary
    exercised is the one a node would pull.
    """
    built = _run(
        [
            "docker",
            "build",
            "--platform",
            PLATFORM,
            "--build-arg",
            f"CODEX_VERSION={CODEX_VERSION}",
            "-f",
            str(_DOCKERFILE),
            "-t",
            IMAGE_TAG,
            str(_ROOT),
        ]
    )
    if built.returncode != 0:
        pytest.fail(
            f"docker build failed rc={built.returncode}\n{built.stderr[-4000:]}"
        )
    return IMAGE_TAG


def _on_the_wire(
    image: str,
    etc: Path,
    *,
    config_toml: str,
    requirements_toml: str,
    gate_key: bytes = TOKEN_KEY,
) -> Wire:
    """Run one probe: two documents at `/etc/codex`, the gate inside, codex against it.

    The gate's key is separate from whatever signed the token in `config_toml`, so a
    signature mismatch can be staged from either side.
    """
    (etc / "config.toml").write_text(config_toml)
    (etc / "requirements.toml").write_text(requirements_toml)
    (etc / "gate.py").write_text(GATE)

    done = _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            PLATFORM,
            "-v",
            f"{etc}:/etc/codex:ro",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            _IN_CONTAINER,
            "sh",
            gate_key.decode(),
        ]
    )
    assert "GATE_NEVER_STARTED" not in done.stdout, done.stdout
    assert done.returncode == 0, f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"
    return Wire(stdout=done.stdout)


@pytest.mark.image
def test_the_container_runs_as_the_uid_the_pod_runs_as(image: str) -> None:
    """The positive control on the environment every case below is measured in.

    A probe that silently ran as root would say nothing about a pod: the documents are
    mounted read-only and `CODEX_HOME` is a path root can write and uid 10001 cannot.
    """
    identity = _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            PLATFORM,
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "id -u",
        ]
    )
    assert identity.returncode == 0, identity.stderr
    assert identity.stdout.strip() == "10001"


@pytest.mark.image
def test_the_compiled_document_makes_the_runtime_send_a_token_the_gateway_accepts(
    image: str, tmp_path: Path
) -> None:
    """The checkpoint: real codex, the compiler's two documents, the real middleware.

    Both halves are asserted: the request reached the app behind the middleware, and
    nothing was refused. A run in which the middleware let a request through *and* also
    refused one would mean the probe made two different requests, which would make every
    negative below ambiguous.
    """
    compiled = _compiled()
    wire = _on_the_wire(
        image,
        tmp_path,
        config_toml=compiled.config_toml,
        requirements_toml=compiled.requirements_toml,
    )

    assert wire.reached_the_recorder, wire.stdout
    assert wire.refusals == 0, wire.stdout
    assert "200 OK" in wire.stdout, wire.stdout


@pytest.mark.image
def test_the_same_document_with_the_header_removed_is_refused(
    image: str, tmp_path: Path
) -> None:
    """The negative control, from the same document with the one line stripped.

    Without this the positive above is satisfied by a middleware that lets everything
    through, which is precisely the failure a fixed 401 makes invisible.
    """
    compiled = _compiled()
    stripped = "\n".join(
        line
        for line in compiled.config_toml.splitlines()
        if not line.startswith("http_headers =")
    )
    assert stripped != compiled.config_toml, "the header line was not found to remove"

    wire = _on_the_wire(
        image,
        tmp_path,
        config_toml=stripped,
        requirements_toml=compiled.requirements_toml,
    )

    assert not wire.reached_the_recorder, wire.stdout
    assert wire.refusals >= 1, wire.stdout


@pytest.mark.image
def test_a_token_signed_with_a_different_key_is_refused_on_the_wire(
    image: str, tmp_path: Path
) -> None:
    """The signature is checked, not merely the header's presence.

    Staged the way it would really happen: the control plane compiles with a key the
    Gateway does not hold. The document is well-formed and every floor passes.
    """
    compiled = _compiled(key=OTHER_KEY)
    wire = _on_the_wire(
        image,
        tmp_path,
        config_toml=compiled.config_toml,
        requirements_toml=compiled.requirements_toml,
        gate_key=TOKEN_KEY,
    )

    assert not wire.reached_the_recorder, wire.stdout
    assert wire.refusals >= 1, wire.stdout


@pytest.mark.image
def test_an_expired_token_is_refused_on_the_wire(image: str, tmp_path: Path) -> None:
    """The expiry, end to end, because the whole 14-day argument rests on it.

    This is also what the ceiling costs: the Gateway is a `required` server, so a
    Session whose token has expired does not lose one tool -- it stops taking Turns.
    """
    compiled = _compiled(expiry=_PAST)
    wire = _on_the_wire(
        image,
        tmp_path,
        config_toml=compiled.config_toml,
        requirements_toml=compiled.requirements_toml,
    )

    assert not wire.reached_the_recorder, wire.stdout
    assert wire.refusals >= 1, wire.stdout


@pytest.mark.image
def test_the_runtime_still_considers_the_gateway_enabled(
    image: str, tmp_path: Path
) -> None:
    """Adding a header to `config.toml` does not disturb the managed allowlist.

    The identity in `requirements.toml` is matched on the URL alone, so this should
    hold -- and "should" is why the paired case below exists.
    """
    compiled = _compiled()
    wire = _on_the_wire(
        image,
        tmp_path,
        config_toml=compiled.config_toml,
        requirements_toml=compiled.requirements_toml,
    )

    assert wire.enabled, wire.stdout
    assert wire.disabled_reason in (None, "null"), wire.stdout


@pytest.mark.image
def test_a_requirements_document_naming_another_url_disables_the_server(
    image: str, tmp_path: Path
) -> None:
    """The planted control for the case above, and it is not optional.

    "Enabled" is also what a runtime that never opened `requirements.toml` would report,
    so on its own it proves the allowlist is *not* read just as well as that it is.
    Changing the one URL the identity matches on must flip the server to disabled and
    name the file -- and probing the wrong file is the specific mistake that produced a
    meaningless `invalid transport` here once: `identity` is a `requirements.toml` key,
    and `/etc/codex/requirements.toml` is where the loader reads it.
    """
    compiled = _compiled()
    elsewhere = compiled.requirements_toml.replace(
        GATEWAY_URL, "http://tool-gateway.somewhere-else.invalid/mcp"
    )
    assert elsewhere != compiled.requirements_toml, "the identity URL was not found"

    wire = _on_the_wire(
        image,
        tmp_path,
        config_toml=compiled.config_toml,
        requirements_toml=elsewhere,
    )

    assert not wire.enabled, wire.stdout
    assert wire.disabled_reason == "requirements (/etc/codex/requirements.toml)", (
        wire.stdout
    )


@pytest.mark.image
def test_a_misspelled_header_key_reaches_the_wire_with_no_header_at_all(
    image: str, tmp_path: Path
) -> None:
    """Why the compile-time floor exists, shown rather than argued.

    `headers` instead of `http_headers` is accepted by the configuration parser, leaves
    the server enabled with no warning, and sends no header -- so from outside the pod
    it is indistinguishable from a document that named none. The floor in
    `config_compiler` is the only thing that catches it before a pod is created; this
    case is what says the runtime will not.
    """
    compiled = _compiled()
    misspelled = compiled.config_toml.replace("http_headers = ", "headers = ")
    assert misspelled != compiled.config_toml

    wire = _on_the_wire(
        image,
        tmp_path,
        config_toml=misspelled,
        requirements_toml=compiled.requirements_toml,
    )

    assert wire.enabled, "a misspelled key is expected to parse and be inert"
    assert not wire.reached_the_recorder, wire.stdout
    assert wire.refusals >= 1, wire.stdout


def test_the_document_this_file_mounts_is_the_compilers_own() -> None:
    """The one case here that runs in the default gate, and it grades this file.

    Every case above mounts `_compiled()`'s output, so if that stopped carrying a token
    -- or started carrying one for a different Session -- the image cases would still be
    marked `image`, still be deselected, and nothing would say so. This runs always.

    The token is compared against the mint rather than parsed apart, because what the
    image cases need is the exact string the Gateway will accept: a value that merely
    looks token-shaped would pass a structural check and fail on the wire, where the
    failure costs a container run to diagnose.
    """
    record = _record()
    compiled = compile_session_config(
        record,
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="fixture",
            runtime_image="registry.map.internal/session@sha256:" + "a" * 64,
            denied_paths=(),
        ),
        definition=A_DEFINITION,
        session_token_key=TOKEN_KEY,
        session_token_expiry_epoch_s=_FUTURE,
    )
    parsed = tomllib.loads(compiled.config_toml)
    server = parsed["mcp_servers"][GATEWAY_SERVER_ID]
    headers = server["http_headers"]

    assert set(headers) == {SESSION_TOKEN_HEADER_NAME}
    assert headers[SESSION_TOKEN_HEADER_NAME] == mint_session_token(
        session_id=record.id,
        tenant_id=record.tenant_id,
        expiry_epoch_s=_FUTURE,
        key=TOKEN_KEY,
    )
    assert server["url"] == GATEWAY_URL
    assert server["required"] is True, (
        "the Gateway is a required server, which is what makes an expired token stop "
        "the Session rather than lose one tool"
    )
