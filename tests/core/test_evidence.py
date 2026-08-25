"""The capture rule, the digest contract, and the key layout Evidence is addressed by.

These are the two numbers and the one algorithm somebody outside this codebase has to
reproduce, so they are asserted against values written here rather than against the
module's own constants where a wrong constant would agree with itself. The empty
SHA-256 and the digest of `b"abc"` are the two published vectors; a refactor that
changed what the digest covers would have to change them too.
"""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from managed_agent.core.ids import SessionId
from managed_agent.core.vfs.evidence import (
    DEFAULT_THRESHOLD_BYTES,
    EVIDENCE_MOUNT,
    HASH_ALGORITHM,
    MAX_THRESHOLD_BYTES,
    RUNTIME_OUTPUT_CAP_BYTES,
    THRESHOLD_ENV_VAR,
    CaptureAsEvidence,
    CaptureContext,
    CapturePoint,
    CaptureThreshold,
    EvidenceBlobs,
    EvidenceDigest,
    EvidenceRecorder,
    EvidenceRef,
    ReturnInline,
    decide,
    digest_of,
    evidence_object_key,
    evidence_vfs_path,
    threshold_from_env,
)

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def a_session() -> SessionId:
    return SessionId(uuid4())


def a_digest(payload: bytes = b"payload") -> EvidenceDigest:
    return digest_of(payload)


def test_the_algorithm_is_named_and_is_sha256() -> None:
    """A consumer reproducing the digest needs the name, not just the length."""
    assert HASH_ALGORITHM == "sha256"


def test_the_threshold_ceiling_is_half_the_runtime_cap() -> None:
    """The margin is checked rather than trusted.

    The runtime discards past its own cap where the output is collected, so a threshold
    at or near that cap would let one result be both uncaptured and truncated -- and
    the lost tail is unrecoverable by anything downstream (ADR-019).
    """
    assert RUNTIME_OUTPUT_CAP_BYTES == 5_000_000
    assert MAX_THRESHOLD_BYTES == RUNTIME_OUTPUT_CAP_BYTES // 2
    assert DEFAULT_THRESHOLD_BYTES < MAX_THRESHOLD_BYTES


@pytest.mark.parametrize("size", [1, MAX_THRESHOLD_BYTES])
def test_a_threshold_inside_the_margin_constructs(size: int) -> None:
    assert CaptureThreshold(size).byte_length == size


@pytest.mark.parametrize("size", [0, -1, RUNTIME_OUTPUT_CAP_BYTES])
def test_a_threshold_outside_the_margin_is_refused(size: int) -> None:
    with pytest.raises(ValueError, match=str(MAX_THRESHOLD_BYTES)):
        CaptureThreshold(size)


def test_at_the_threshold_exactly_the_output_is_evidence() -> None:
    """The boundary is stated once, here, because both capture points read it."""
    threshold = CaptureThreshold(100)
    assert threshold.captures(99) is False
    assert threshold.captures(100) is True
    assert threshold.captures(101) is True


def test_an_unset_variable_gives_the_default_threshold() -> None:
    assert threshold_from_env({}) == CaptureThreshold(DEFAULT_THRESHOLD_BYTES)


def test_a_set_variable_is_the_threshold() -> None:
    assert threshold_from_env({THRESHOLD_ENV_VAR: "4096"}).byte_length == 4096


def test_a_variable_that_is_not_a_number_names_itself_in_the_refusal() -> None:
    """A start-up failure has to say which variable to fix."""
    with pytest.raises(ValueError, match=THRESHOLD_ENV_VAR):
        threshold_from_env({THRESHOLD_ENV_VAR: "nonsense"})


def test_a_variable_above_the_ceiling_is_refused() -> None:
    with pytest.raises(ValueError, match=str(MAX_THRESHOLD_BYTES)):
        threshold_from_env({THRESHOLD_ENV_VAR: str(MAX_THRESHOLD_BYTES + 1)})


def test_the_environment_is_read_when_no_mapping_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production call takes no argument, so that path needs its own assertion."""
    monkeypatch.setenv(THRESHOLD_ENV_VAR, "8192")
    assert threshold_from_env().byte_length == 8192
    monkeypatch.delenv(THRESHOLD_ENV_VAR)
    assert threshold_from_env().byte_length == DEFAULT_THRESHOLD_BYTES


def test_the_empty_payload_hashes_to_the_published_empty_digest() -> None:
    """A published vector, so a change to what the digest covers cannot pass
    silently."""
    empty = digest_of(b"")
    assert empty.hex == EMPTY_SHA256
    assert empty.byte_length == 0
    assert empty.algorithm == "sha256"


def test_the_digest_covers_exactly_the_octets_given() -> None:
    """No framing, no re-encoding, no trailing newline."""
    assert digest_of(b"abc").hex == hashlib.sha256(b"abc").hexdigest()
    assert digest_of(b"abc\n").hex != digest_of(b"abc").hex


def test_the_length_is_part_of_the_identity() -> None:
    """A short body would otherwise report a bare mismatch, with no way to tell a
    corrupted object from a truncated read of an intact one."""
    assert digest_of(b"0123456789").byte_length == 10


@pytest.mark.parametrize(
    "bad",
    [
        {"hex": EMPTY_SHA256.upper(), "byte_length": 0},
        {"hex": EMPTY_SHA256[:63], "byte_length": 0},
        {"hex": EMPTY_SHA256, "byte_length": -1},
        {"hex": EMPTY_SHA256, "byte_length": 0, "algorithm": "sha1"},
        {"hex": EMPTY_SHA256, "byte_length": 0, "extra": "x"},
    ],
)
def test_a_digest_that_no_reader_could_reproduce_is_refused(
    bad: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        EvidenceDigest(**bad)


def test_a_digest_cannot_be_edited_after_it_is_built() -> None:
    with pytest.raises(ValidationError):
        a_digest().hex = EMPTY_SHA256


def test_below_the_threshold_the_bytes_go_back_and_nothing_is_hashed() -> None:
    decided = decide(b"x" * 99, CaptureThreshold(100), passed_through_bytes=0)
    assert decided == ReturnInline(
        byte_length=99, threshold=100, passed_through_bytes=0
    )


def test_at_the_threshold_the_decision_already_carries_the_digest() -> None:
    """Deciding and hashing in one place is what keeps the recorded digest and the
    stored bytes from drifting apart: there is no route to the storing branch that does
    not already hold the digest of exactly this payload."""
    payload = b"y" * 100
    decided = decide(payload, CaptureThreshold(100), passed_through_bytes=0)
    assert isinstance(decided, CaptureAsEvidence)
    assert decided.digest == digest_of(payload)
    assert decided.threshold == 100


def test_the_two_capture_points_are_spelled_the_way_the_store_expects() -> None:
    """These two strings are a stored value and a check constraint, not a label."""
    assert CapturePoint.TOOL_GATEWAY.value == "tool-gateway"
    assert CapturePoint.SESSION_SHIM.value == "session-shim"
    assert str(CapturePoint.TOOL_GATEWAY) == "tool-gateway"


def test_one_digest_under_two_sessions_differs_only_in_the_session_segment() -> None:
    """The per-Session prefix is what lets one Session's expiry delete exactly its own
    Evidence with a prefix delete, instead of a reference count nobody is keeping."""
    digest = a_digest()
    one, two = a_session(), a_session()
    first = evidence_object_key(one, digest)
    second = evidence_object_key(two, digest)

    assert first.startswith("evidence/")
    assert second.startswith("evidence/")
    assert first != second
    assert first.replace(str(one), "") == second.replace(str(two), "")
    assert first.endswith(f"{digest.algorithm}-{digest.hex}")


def test_the_key_is_inside_the_prefix_a_session_expiry_sweeps() -> None:
    session = a_session()
    key = evidence_object_key(session, a_digest())
    assert key.startswith(f"evidence/{session}/")


def test_the_path_the_agent_opens_is_under_the_session_evidence_mount() -> None:
    digest = a_digest()
    path = evidence_vfs_path(digest)
    assert path.startswith(f"{EVIDENCE_MOUNT}/")
    assert path.endswith(digest.hex)


def test_two_distinct_payloads_never_collide_on_one_path() -> None:
    assert evidence_vfs_path(digest_of(b"one")) != evidence_vfs_path(digest_of(b"two"))


def test_a_reference_refuses_an_unknown_field_and_refuses_assignment() -> None:
    ref = EvidenceRef(
        session_id=a_session(),
        digest=a_digest(),
        capture_point=CapturePoint.TOOL_GATEWAY,
        object_key="evidence/x/sha256-y",
        vfs_path=f"{EVIDENCE_MOUNT}/sha256-y",
    )
    assert ref.truncated_at_runtime_cap is False
    with pytest.raises(ValidationError):
        ref.object_key = "somewhere else"
    with pytest.raises(ValidationError):
        EvidenceRef(
            session_id=a_session(),
            digest=a_digest(),
            capture_point=CapturePoint.TOOL_GATEWAY,
            object_key="k",
            vfs_path="p",
            unheard_of=1,  # type: ignore[call-arg]
        )


def test_a_capture_context_cannot_be_built_without_saying_which_point_it_is() -> None:
    """A reviewer must never have to infer the assurance from the tool name."""
    with pytest.raises(TypeError):
        CaptureContext(  # type: ignore[call-arg]
            session_id=a_session(), call_id="c", tool_name="t"
        )


class InMemoryEvidence:
    """A fake standing in for both ports at once, so the pair is graded together."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.inline: list[tuple[CaptureContext, ReturnInline]] = []

    async def put(self, key: str, body: bytes) -> str:
        self.objects[key] = body
        return key

    async def get(self, key: str) -> bytes:
        return self.objects[key]

    async def delete_prefix(self, prefix: str) -> int:
        doomed = [k for k in self.objects if k.startswith(prefix)]
        for key in doomed:
            del self.objects[key]
        return len(doomed)

    async def record_captured(
        self,
        ctx: CaptureContext,
        payload: bytes,
        decision: CaptureAsEvidence,
        truncated_at_runtime_cap: bool = False,
    ) -> EvidenceRef:
        key = evidence_object_key(ctx.session_id, decision.digest)
        await self.put(key, payload)
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


async def test_an_in_memory_fake_satisfies_both_ports() -> None:
    """Structural, so `mypy --strict` grades the fake against the protocols and this
    body proves the annotations are not merely accepted but usable."""
    fake = InMemoryEvidence()
    blobs: EvidenceBlobs = fake
    recorder: EvidenceRecorder = fake
    session = SessionId(UUID("22222222-2222-4222-8222-222222222222"))
    payload = b"z" * 100
    decision = decide(payload, CaptureThreshold(100), passed_through_bytes=0)
    assert isinstance(decision, CaptureAsEvidence)

    ref = await recorder.record_captured(
        CaptureContext(
            session_id=session,
            call_id="call-1",
            tool_name="acme__search",
            capture_point=CapturePoint.TOOL_GATEWAY,
        ),
        payload,
        decision,
    )

    assert await blobs.get(ref.object_key) == payload
    assert digest_of(await blobs.get(ref.object_key)) == ref.digest
    assert await blobs.delete_prefix(f"evidence/{session}/") == 1
