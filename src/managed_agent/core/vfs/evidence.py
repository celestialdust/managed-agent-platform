"""When tool output becomes Evidence, and exactly what its hash covers.

Two things are pinned here because a consumer has to reproduce them. The threshold is a
single value that every capture point reads, so whether a given result is Evidence does
not depend on which tool produced it or which code path classified it. The digest is
SHA-256 over the captured octets as they are stored and as they are later served — no
framing, no re-encoding, no trailing newline — so an auditor who downloads the bytes and
hashes them either gets the recorded value back or has found a real discrepancy.

The threshold is held well below the Agent Runtime's configured output cap, and the
relationship is enforced rather than hoped for. The runtime discards bytes past its cap
where the output is collected, so nothing downstream can recover them; keeping the two
rules a factor of two apart means a result small enough to be returned inline was never
anywhere near being truncated either, and the two rules can never fire on one call
(ADR-019).
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from os import environ
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.ids import SessionId

HASH_ALGORITHM: Final[Literal["sha256"]] = "sha256"
"""The one digest algorithm Evidence is addressed by and verified against."""

RUNTIME_OUTPUT_CAP_BYTES: Final[int] = 5_000_000
"""How much of one tool's output the Agent Runtime is configured to retain.

Decimal 5 MB rather than 5 MiB: the number is a per-concurrent-call memory budget in a
pod tier that is sized by memory, and rounding it to a power of two would imply a
precision the measurement behind it does not have. It is raised from the runtime's own
1 MiB default so that a large result reaches a capture point before the runtime drops
its tail.
"""

MAX_THRESHOLD_BYTES: Final[int] = RUNTIME_OUTPUT_CAP_BYTES // 2
"""The largest threshold that keeps the two rules from ever firing on one call.

A factor of two, not a rounded-down margin. Anything at or above this and a single
result could be both small enough to be returned inline and large enough to have
already lost its tail at the runtime's cap — and a reader has no way to tell an intact
inline result from a truncated one.
"""

DEFAULT_THRESHOLD_BYTES: Final[int] = 65_536
"""What the threshold is when the deployment sets nothing.

64 KiB is a starting value, not a measurement: it is far enough below the runtime's cap
to leave the margin above intact and far enough above a typical structured result that
an ordinary call is not diverted. The number the platform runs on is configured.
"""

THRESHOLD_ENV_VAR: Final[str] = "EVIDENCE_CAPTURE_THRESHOLD_BYTES"


class CapturePoint(StrEnum):
    """Where a piece of Evidence was captured, and therefore what it is worth.

    TOOL_GATEWAY output was captured by infrastructure the pod cannot reach.
    SESSION_SHIM output was born inside the pod and captured by platform code running
    there, so a pod compromised deeply enough to subvert the shim can subvert its own
    record. The two do not carry the same assurance, and every stored piece of Evidence
    says which it is rather than leaving a reviewer to infer it from the tool name
    (ADR-019).

    These two spellings are a stored column value and a check constraint in migration
    0007, not a display label. Renaming one renames data.
    """

    TOOL_GATEWAY = "tool-gateway"
    SESSION_SHIM = "session-shim"


@dataclass(frozen=True, slots=True)
class CaptureThreshold:
    """The one size at or above which output is captured instead of returned inline."""

    byte_length: int

    def __post_init__(self) -> None:
        if not 1 <= self.byte_length <= MAX_THRESHOLD_BYTES:
            raise ValueError(
                f"capture threshold {self.byte_length} is outside"
                f" 1..{MAX_THRESHOLD_BYTES}; a threshold at or near the runtime's"
                " output cap lets one result be both uncaptured and truncated"
            )

    def captures(self, size: int) -> bool:
        """Whether output of this many bytes is Evidence. At the threshold exactly, it
        is."""
        return size >= self.byte_length


def threshold_from_env(env: Mapping[str, str] | None = None) -> CaptureThreshold:
    """Read the one threshold, from the one variable, for every capture point.

    Both capture points call this and nothing else. If they ever read two values,
    Evidence coverage silently becomes a function of which tool ran — which is the one
    property the size-based rule exists to remove.
    """
    source = environ if env is None else env
    raw = source.get(THRESHOLD_ENV_VAR)
    if raw is None:
        return CaptureThreshold(DEFAULT_THRESHOLD_BYTES)
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{THRESHOLD_ENV_VAR}={raw!r} is not an integer") from exc
    return CaptureThreshold(parsed)


class EvidenceDigest(BaseModel):
    """A captured payload's identity: the algorithm, the digest, and the length covered.

    The length is part of the identity rather than a convenience. A reader served a
    short body would otherwise hash fewer bytes and report a bare mismatch, with no way
    to tell a corrupted object from a truncated read of an intact one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["sha256"] = HASH_ALGORITHM
    hex: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)


def digest_of(payload: bytes) -> EvidenceDigest:
    """Hash exactly the octets given, and nothing else.

    The covered range is the whole payload as captured: the bytes written to the object
    store and the bytes any later reader is served. Nothing is prepended, appended or
    re-encoded between hashing and storing, so re-hashing what a reader downloads
    reproduces this value byte for byte.
    """
    return EvidenceDigest(
        hex=hashlib.sha256(payload).hexdigest(), byte_length=len(payload)
    )


@dataclass(frozen=True, slots=True)
class ReturnInline:
    """Output below the threshold: the agent gets the bytes, and no Evidence is
    written."""

    byte_length: int
    threshold: int
    passed_through_bytes: int


@dataclass(frozen=True, slots=True)
class CaptureAsEvidence:
    """Output at or above the threshold: these bytes are stored and referenced."""

    digest: EvidenceDigest
    threshold: int
    passed_through_bytes: int


Disposition = ReturnInline | CaptureAsEvidence


def decide(
    payload: bytes, threshold: CaptureThreshold, *, passed_through_bytes: int
) -> Disposition:
    """Classify one tool result. The hashing happens here, at the moment of the
    decision.

    Deciding and hashing in one place is what keeps the recorded digest and the stored
    bytes from drifting apart: there is no route to the storing branch that does not
    already hold the digest of exactly this payload, so no caller can store one payload
    while recording the hash of another.

    `passed_through_bytes` counts the octets of the same result that this point does not
    weigh and hands back untouched. It takes no part in the decision -- the size rule is
    over the weighed payload, and a threshold applied to bytes that are never captured
    would divert a call while leaving the bytes that caused the diversion on the wire.
    It is required rather than defaulted because zero is a claim: a row asserting that
    nothing else reached the caller, written by a caller who simply did not look, is the
    one shape of lie this ledger exists to make impossible.
    """
    if not threshold.captures(len(payload)):
        return ReturnInline(
            byte_length=len(payload),
            threshold=threshold.byte_length,
            passed_through_bytes=passed_through_bytes,
        )
    return CaptureAsEvidence(
        digest=digest_of(payload),
        threshold=threshold.byte_length,
        passed_through_bytes=passed_through_bytes,
    )


EVIDENCE_MOUNT: Final[str] = "/session/evidence"


def evidence_object_key(session_id: SessionId, digest: EvidenceDigest) -> str:
    """Where a captured payload's bytes live in the object store.

    Content-addressed under a per-Session prefix, which deliberately stores identical
    bytes twice when two Sessions capture them. The prefix is what lets one Session's
    expiry delete exactly its own Evidence with a prefix delete; a globally deduplicated
    key would turn every expiry into a reference count nobody is maintaining, and the
    first missed decrement deletes Evidence some other Session's audit trail points at.
    """
    return f"evidence/{session_id}/{digest.algorithm}-{digest.hex}"


def evidence_vfs_path(digest: EvidenceDigest) -> str:
    """Where the same bytes appear inside the tree the agent reads.

    One file name per digest under the Session's own evidence mount, so the path a
    result references is the path the agent can open, and two different payloads can
    never collide on one name. The mount's sibling directories are not this module's to
    name.
    """
    return f"{EVIDENCE_MOUNT}/{digest.algorithm}-{digest.hex}"


class EvidenceRef(BaseModel):
    """One stored piece of Evidence, as everything downstream refers to it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: SessionId
    digest: EvidenceDigest
    capture_point: CapturePoint
    object_key: str
    vfs_path: str
    truncated_at_runtime_cap: bool = False
    """True when the Agent Runtime had already cut this output at its own cap before
    capture.

    Always False for output captured outside the pod, where nothing has passed through
    the runtime yet. It exists on the shared type because a capture made inside the pod
    can be cut, and Evidence with a tail missing must never be presented as complete.
    """


@dataclass(frozen=True, slots=True)
class CaptureContext:
    """Who produced one result, and at which of the two points it is classified."""

    session_id: SessionId
    call_id: str
    tool_name: str
    capture_point: CapturePoint


class EvidenceStorageUnconfigured(RuntimeError):
    """No object bucket is named, so a payload has nowhere to be written.

    Defined beside the two ports rather than in the adapter that first raised it,
    because it is part of what a caller must be prepared for when it holds one of these
    Protocols -- and a caller cannot import the adapter. The layer linter bans
    `managed_agent.adapters` outside `adapters/` and `composition.py`, so an exception
    living there is one no upstream handler is allowed to name; the Tool Gateway's error
    map could only catch it as a bare `RuntimeError` and reported an unset bucket as the
    registered server failing a call that server had in fact completed.

    A `RuntimeError` rather than a domain error because it is a deployment that is
    wrong, not a request: no argument a tenant could send makes it go away, and nothing
    inside a running process can correct it.
    """


class EvidenceBlobs(Protocol):
    """The three object-store operations Evidence needs, and no others."""

    async def put(self, key: str, body: bytes) -> str: ...

    async def get(self, key: str) -> bytes: ...

    async def delete_prefix(self, prefix: str) -> int: ...


class EvidenceRecorder(Protocol):
    """Durably recording one capture decision, whichever point reached it.

    Both capture points call this and neither knows how many stores sit behind it. That
    is what keeps the payload write and the ledger write one ordered operation instead
    of two that every caller has to sequence identically and one of them eventually will
    not.
    """

    async def record_captured(
        self,
        ctx: CaptureContext,
        payload: bytes,
        decision: CaptureAsEvidence,
        truncated_at_runtime_cap: bool = False,
    ) -> EvidenceRef: ...

    async def record_inline(
        self, ctx: CaptureContext, decision: ReturnInline
    ) -> None: ...
