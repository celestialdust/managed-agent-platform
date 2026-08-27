"""Shared pieces for the Tool Gateway's unit tier: a counting vault and a real server.

The stdio server is a separate process speaking the real protocol down a pipe, so
nothing here is a fake the proxy could see through. What it cannot catch is a quirk of
somebody else's stdio implementation — it is written to the same SDK the Gateway is.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final
from uuid import UUID

from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.registration.scope_binding import StdioServer
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vfs.evidence import (
    CaptureAsEvidence,
    CaptureContext,
    CaptureThreshold,
    EvidenceRef,
    ReturnInline,
    evidence_object_key,
    evidence_vfs_path,
)
from managed_agent.gateway.tool.credential_broker import ToolCredentialBroker
from managed_agent.gateway.tool.evidence_capture import EvidenceCapture

STDIO_SERVER: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "conformance"
    / "mcp"
    / "servers"
    / "stdio_server.py"
)

SCHEMA_STDIO_SERVER: Final[Path] = (
    Path(__file__).resolve().parent / "schema_stdio_server.py"
)
"""A second real server, whose one tool declares an output schema.

Separate from the conformance server rather than a sixth tool on it, because a declared
output schema is a property of the *listing* and the caller's MCP client validates every
result of every tool it listed. Adding the schema to a tool on the shared server would
put that validation in the path of the fidelity, progress and elicitation tests, which
are about something else.
"""

CREDENTIAL_REF: Final[str] = "conformance/stdio-token"
CREDENTIAL_VALUE: Final[str] = "s3cr3t-conformance-value"
CREDENTIAL_ENV_VAR: Final[str] = "MAP_CONFORMANCE_TOKEN"

TENANT: Final[TenantId] = TenantId(UUID("11111111-1111-4111-8111-111111111111"))
"""The tenant every unit-tier registration in this package belongs to.

Fixed rather than fresh per test, because the vault key a fetch is asserted against is
composed from it and a random tenant would make that assertion unreadable in a failure
message.
"""


class CountingVault:
    """A `CredentialVault` that answers one secret and counts who asked.

    The count is how a test tells one spawned child from two: the proxy fetches exactly
    once per transport it opens, and the process it spawns hands back no handle to count
    directly.
    """

    def __init__(self, value: str = CREDENTIAL_VALUE) -> None:
        self.value = value
        self.fetches: list[str] = []

    async def fetch(self, name: str) -> str:
        self.fetches.append(name)
        return self.value


def broker(vault: CountingVault) -> ToolCredentialBroker:
    """A real broker over a counting vault.

    The broker is the real class and only the vault is a fake, so what these tests
    exercise is the composition and the hold rather than a stand-in for them. The vault
    is the genuine boundary: it is the thing that would otherwise be an AWS call.
    """
    return ToolCredentialBroker(vault=vault)


def stdio_endpoint(*args: str) -> StdioServer:
    """A registration pointing at the conformance stdio server, run by this venv."""
    return StdioServer(
        transport="stdio",
        command=sys.executable,
        args=(str(STDIO_SERVER), *args),
        credential_ref=CREDENTIAL_REF,
        credential_env_var=CREDENTIAL_ENV_VAR,
    )


def schema_stdio_endpoint() -> StdioServer:
    """A registration pointing at the server whose tool declares an output schema."""
    return StdioServer(
        transport="stdio",
        command=sys.executable,
        args=(str(SCHEMA_STDIO_SERVER),),
        credential_ref=CREDENTIAL_REF,
        credential_env_var=CREDENTIAL_ENV_VAR,
    )


class CountingEvidence:
    """An `EvidenceRecorder` that keeps every classification and stores the bytes.

    Here rather than in each test file because two modules in this package need one and
    a third does not exist yet. It is a fake at the port -- the real
    `tool_gateway.evidence_capture.EvidenceCapture` sits in front of it -- so what these
    tests exercise is the capture decision and the substitution, with only the two
    stores behind the port stood in for.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.captured: list[tuple[CaptureContext, CaptureAsEvidence]] = []
        self.inline: list[tuple[CaptureContext, ReturnInline]] = []

    async def record_captured(
        self,
        ctx: CaptureContext,
        payload: bytes,
        decision: CaptureAsEvidence,
        truncated_at_runtime_cap: bool = False,
    ) -> EvidenceRef:
        key = evidence_object_key(ctx.session_id, decision.digest)
        self.objects[key] = payload
        self.captured.append((ctx, decision))
        return EvidenceRef(
            session_id=ctx.session_id,
            digest=decision.digest,
            capture_point=ctx.capture_point,
            object_key=key,
            vfs_path=evidence_vfs_path(decision.digest),
            truncated_at_runtime_cap=truncated_at_runtime_cap,
        )

    async def record_inline(self, ctx: CaptureContext, decision: ReturnInline) -> None:
        self.inline.append((ctx, decision))


UNIT_TIER_THRESHOLD: Final[CaptureThreshold] = CaptureThreshold(64 * 1024)
"""A threshold above anything the conformance stdio server returns.

Deliberately the production default rather than something tiny: these tests are about
proxy fidelity, and every one of them would otherwise assert against a reference
sentence instead of the server's own result. A test that wants a capture sets its own.
"""

OVERSIZE_BODY_BYTES: Final[int] = 100_000
"""How large a body a test asks for when it wants one the capture point must store.

Above `UNIT_TIER_THRESHOLD`, and below Linux's ceiling on a single string handed to a
child process -- which is the half nobody expects. The only size lever either stdio
server offers is the credential, and a credential reaches the child in its environment,
so on Linux the whole body has to fit in one `envp` string. `MAX_ARG_STRLEN` is 32
pages: 128 KiB wherever a page is 4 KiB, which is every runner this suite has met.

Measured, not reasoned about. These tests asked for 200_000 and passed for months on
macOS, which has no per-string limit, then failed on the first Linux CI run with
`OSError: [Errno 7] Argument list too long` out of `subprocess.Popen`. It surfaces to
the caller as `tool.unavailable` -- an upstream that would not start -- so it reads as a
Gateway defect and is not one.

Anything past ~128 KiB brings that back. A case that needs a genuinely larger payload
belongs in `test_evidence_capture.py`, which calls the capture point directly and spawns
nothing.
"""


def capture(
    recorder: CountingEvidence | None = None,
    threshold: CaptureThreshold = UNIT_TIER_THRESHOLD,
) -> EvidenceCapture:
    """The real capture point over a counting recorder."""
    return EvidenceCapture(recorder or CountingEvidence(), threshold)


class FixedScope:
    """A `SessionScopeReader` that answers one Scope and counts the reads.

    Every registered tool declares a Scope Binding, so every proxy that calls one has
    to be given a Scope carrying that binding's dimension -- which is why this appears
    in every test here that makes a call, rather than only in the ones about clamping.
    The default names `account`, the dimension this package's registrations bind.

    The Grant is positional and defaults to empty, which the proxy reads as *no tools*.
    That is deliberately the inconvenient default: a case that calls a tool has to name
    the tool in its Grant, so the authorization it is relying on is written in the case
    rather than inherited from a permissive fixture nobody reads.

    The read count is not decoration. The proxy is meant to read a Session's Scope once
    and hold it for its life, and a proxy that re-read it per call would put a database
    round trip on the hot path with every assertion in this package still green.
    """

    def __init__(self, *grant: str, **scope: str) -> None:
        self.scope: dict[str, str] = scope or {"account": "the-tenants-own-account"}
        self.grant = frozenset(grant)
        self.reads = 0

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        self.reads += 1
        return SessionRecord(
            id=session_id,
            tenant_id=tenant_id,
            definition_id=DefinitionId(UUID("22222222-2222-4222-8222-222222222222")),
            definition_revision="0" * 40,
            grant=self.grant,
            scope=tuple(self.scope.items()),
            budget_minor_units=1_000,
            budget_currency="USD",
            retention_days=1,
        )
