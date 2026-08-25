"""Large enterprise output is captured before the Agent Runtime is handed anything.

The point of the position is that nothing derived from the bytes reaches the model until
the record is durable, so these tests assert on ordering as well as on content: the fake
recorder below records whether `apply` had returned by the time the capture landed, and
the answer has to be no.

What crosses on a capture is a short sentence naming the path, the length and the
digest. Two things fall out and neither is incidental -- the payload cannot be re-sent
in a context every Turn, and the runtime's own output cap never fires on an enterprise
result, because what the runtime receives is a few hundred bytes whatever the
registered server returned.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Final, get_args
from uuid import uuid4

import pytest
from mcp.types import (
    AudioContent,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from managed_agent.core.ids import SessionId
from managed_agent.core.vfs.evidence import (
    EVIDENCE_MOUNT,
    CaptureAsEvidence,
    CaptureContext,
    CapturePoint,
    CaptureThreshold,
    EvidenceRef,
    ReturnInline,
    digest_of,
    evidence_object_key,
    evidence_vfs_path,
)
from managed_agent.gateway.tool.evidence_capture import EvidenceCapture

THRESHOLD = CaptureThreshold(1_000)

MARKER: Final[str] = "M" * 500
"""Bulk small enough to keep a result inline, large enough that a zero count is a fail.

Every block in the table below carries it, so each of those cases makes the same
assertion: this block's octets reached the count, rather than being weighed as
nothing because no branch here names its type.
"""

NON_TEXT_BLOCKS: Final[Mapping[str, ContentBlock]] = {
    "image": ImageContent(type="image", data=MARKER, mime_type="image/png"),
    "audio": AudioContent(type="audio", data=MARKER, mime_type="audio/wav"),
    "resource_link": ResourceLink(
        type="resource_link", uri="acme://index", name="index", description=MARKER
    ),
    "resource": EmbeddedResource(
        type="resource",
        resource=TextResourceContents(uri="acme://report", text=MARKER),
    ),
}
"""One block of every content type the protocol defines that is not text.

Keyed by the protocol's own discriminator, so the test asserting this table is complete
reads as the protocol reads. Every entry is a type the capture point has no branch for
-- it branches on text and nothing else -- which is the property being pinned: two of
these four were the only ones any case here ever built, and a rule that quietly weighed
the other two as nothing would have gone unnoticed.
"""


class RecordingRecorder:
    """A recorder that keeps the bytes it was given and notes when it was called.

    `returned_before_apply` is the ordering assertion's evidence: the capture point sets
    it once `apply` is about to hand a result back, so a recorder that fired afterwards
    would see it already True.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.captured: list[tuple[CaptureContext, CaptureAsEvidence]] = []
        self.inline: list[tuple[CaptureContext, ReturnInline]] = []
        self.apply_had_returned_when_recorded: list[bool] = []
        self.apply_has_returned = False

    async def record_captured(
        self,
        ctx: CaptureContext,
        payload: bytes,
        decision: CaptureAsEvidence,
        truncated_at_runtime_cap: bool = False,
    ) -> EvidenceRef:
        self.apply_had_returned_when_recorded.append(self.apply_has_returned)
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
        self.apply_had_returned_when_recorded.append(self.apply_has_returned)
        self.inline.append((ctx, decision))


def a_capture() -> tuple[RecordingRecorder, EvidenceCapture]:
    recorder = RecordingRecorder()
    return recorder, EvidenceCapture(recorder, THRESHOLD)


async def apply(
    capture: EvidenceCapture,
    result: CallToolResult,
    recorder: RecordingRecorder | None = None,
    session: SessionId | None = None,
) -> CallToolResult:
    returned = await capture.apply(
        session or SessionId(uuid4()), "call-1", "acme__search", result
    )
    if recorder is not None:
        recorder.apply_has_returned = True
    return returned


def text_result(body: str, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=body)], is_error=is_error
    )


def only_text(result: CallToolResult) -> list[str]:
    return [b.text for b in result.content if isinstance(b, TextContent)]


async def test_a_large_result_comes_back_as_a_reference_and_not_as_the_bytes() -> None:
    recorder, capture = a_capture()
    body = "x" * 5_000

    returned = await apply(capture, text_result(body), recorder)

    assert len(returned.content) == 1
    sentence = only_text(returned)[0]
    assert body not in sentence
    assert str(len(body)) in sentence
    assert EVIDENCE_MOUNT in sentence
    assert hashlib.sha256(body.encode()).hexdigest() in sentence
    assert len(sentence) < 500, "a reference must not itself be a large payload"


async def test_the_bytes_the_reference_names_are_the_bytes_that_were_stored() -> None:
    """The whole of the hash contract at this capture point: the digest in the sentence
    the model reads is the digest of the object an auditor can download."""
    recorder, capture = a_capture()
    body = "y" * 5_000

    returned = await apply(capture, text_result(body), recorder)

    (ctx, decision) = recorder.captured[0]
    stored = recorder.objects[evidence_object_key(ctx.session_id, decision.digest)]
    assert stored == body.encode()
    assert digest_of(stored) == decision.digest
    assert digest_of(stored).hex in only_text(returned)[0]


async def test_the_record_is_durable_before_the_result_is_handed_back() -> None:
    """Move the capture after the hand-back and every other test here still passes."""
    recorder, capture = a_capture()

    await apply(capture, text_result("z" * 5_000), recorder)

    assert recorder.apply_had_returned_when_recorded == [False]


async def test_a_small_result_comes_back_exactly_as_the_server_sent_it() -> None:
    recorder, capture = a_capture()
    original = text_result("small enough")

    returned = await apply(capture, original, recorder)

    assert returned == original
    assert recorder.objects == {}
    assert recorder.captured == []


async def test_a_small_result_still_gets_a_row_recording_its_size() -> None:
    """A reviewer who finds no Evidence for a call has to see that it was small rather
    than that the record is missing."""
    recorder, capture = a_capture()

    await apply(capture, text_result("tiny"), recorder)

    (ctx, decision) = recorder.inline[0]
    assert decision == ReturnInline(
        byte_length=4, threshold=1_000, passed_through_bytes=0
    )
    assert ctx.capture_point is CapturePoint.TOOL_GATEWAY
    assert ctx.tool_name == "acme__search"
    assert ctx.call_id == "call-1"
    assert recorder.apply_had_returned_when_recorded == [False]


@pytest.mark.parametrize(
    ("size", "captured"), [(999, False), (1_000, True), (1_001, True)]
)
async def test_the_boundary_is_at_the_threshold_and_includes_it(
    size: int, captured: bool
) -> None:
    recorder, capture = a_capture()

    await apply(capture, text_result("a" * size), recorder)

    assert bool(recorder.captured) is captured
    assert bool(recorder.inline) is not captured


async def test_every_text_block_is_weighed_together_and_replaced_together() -> None:
    """Nothing that was weighed is left on the wire, and nothing left on the wire went
    unweighed."""
    recorder, capture = a_capture()
    result = CallToolResult(
        content=[
            TextContent(type="text", text="head" * 200),
            TextContent(type="text", text="tail" * 200),
        ]
    )

    returned = await apply(capture, result, recorder)

    (ctx, decision) = recorder.captured[0]
    stored = recorder.objects[evidence_object_key(ctx.session_id, decision.digest)]
    assert stored == ("head" * 200 + "\n" + "tail" * 200).encode()
    assert len(returned.content) == 1


async def test_a_non_text_block_survives_a_capture_untouched() -> None:
    """An image carries its own size discipline in the protocol, and folding one into a
    text digest would produce bytes no reader could reconstruct."""
    recorder, capture = a_capture()
    image = ImageContent(type="image", data="Zm9v", mime_type="image/png")
    result = CallToolResult(content=[TextContent(type="text", text="q" * 5_000), image])

    returned = await apply(capture, result, recorder)

    assert returned.content[1] == image
    assert len(returned.content) == 2
    assert "q" * 5_000 not in only_text(returned)[0]
    (ctx, decision) = recorder.captured[0]
    stored = recorder.objects[evidence_object_key(ctx.session_id, decision.digest)]
    assert stored == ("q" * 5_000).encode()


async def test_a_result_weighed_to_the_last_octet_reports_none_passed_through() -> None:
    """The positive control for the two cases below.

    Both of them assert that a count is large. A count that were large for every result
    would satisfy them while telling a reviewer nothing, so one case has to pin the
    other end: a result made only of text is weighed to the last octet, and the row says
    so.
    """
    recorder, capture = a_capture()

    await apply(capture, text_result("tiny"), recorder)

    assert recorder.inline[0][1].passed_through_bytes == 0


async def test_bulk_this_point_does_not_weigh_does_not_read_as_small() -> None:
    """An eleven-byte row about a call that handed over 200 KB is worse than no row.

    A reviewer who finds no Evidence for a call reads this row to learn why, and the
    weighed octets alone answer with the size of the part that was looked at. An
    embedded resource is passed through by design -- folding it into a text digest would
    produce bytes no reader could reconstruct -- so the row has to say that it went
    through, or it asserts that this call's output was below the threshold when 200 KB
    of it was not.
    """
    recorder, capture = a_capture()
    bulk = "T" * 200_000
    result = CallToolResult(
        content=[
            TextContent(type="text", text="small notes"),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(uri="acme://report", text=bulk),
            ),
        ]
    )

    returned = await apply(capture, result, recorder)

    assert recorder.captured == [], "the weighed part is below the threshold"
    decision = recorder.inline[0][1]
    assert decision.byte_length == len("small notes")
    assert decision.byte_length < decision.threshold
    assert decision.passed_through_bytes > len(bulk), (
        "the row says nothing of the 200 KB it handed the caller"
    )
    assert bulk in "".join(
        block.resource.text
        for block in returned.content
        if isinstance(block, EmbeddedResource)
        and isinstance(block.resource, TextResourceContents)
    )


async def test_a_captured_row_says_how_much_went_through_beside_it() -> None:
    """A capture bounds the text, not the blocks that survive it.

    The image goes back with the reference sentence, so the same question a reviewer has
    about an inline row -- how much reached the model -- has an answer here too.
    """
    recorder, capture = a_capture()
    image = ImageContent(type="image", data="B" * 240_000, mime_type="image/png")
    result = CallToolResult(content=[TextContent(type="text", text="q" * 5_000), image])

    await apply(capture, result, recorder)

    decision = recorder.captured[0][1]
    assert decision.digest.byte_length == 5_000
    assert decision.passed_through_bytes > len(image.data)


async def test_every_block_that_passes_through_is_counted_and_not_just_one() -> None:
    """A result whose bulk is spread over several blocks is not one block's worth.

    The two cases above hold for a result with exactly one block this point does not
    weigh, and a count that read the first of them -- or the last, or the largest --
    satisfies both while filing a three-document result at the size of one document.
    The row is the reviewer's answer to "how much reached the model", so it has to be
    the whole of what reached the model rather than a sample of it.

    All three carry the same bulk, so summing them is the only reading that clears the
    assertion: drop any one and the count falls a third short.
    """
    recorder, capture = a_capture()
    each = "T" * 60_000
    result = CallToolResult(
        content=[
            TextContent(type="text", text="small notes"),
            ImageContent(type="image", data=each, mime_type="image/png"),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(uri="acme://first", text=each),
            ),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(uri="acme://second", text=each),
            ),
        ]
    )

    await apply(capture, result, recorder)

    decision = recorder.inline[0][1]
    assert decision.byte_length == len("small notes")
    assert decision.passed_through_bytes > 3 * len(each), (
        "the row reports one block's octets for a call that handed over three"
    )


@pytest.mark.parametrize(
    "block", list(NON_TEXT_BLOCKS.values()), ids=list(NON_TEXT_BLOCKS)
)
async def test_a_content_type_this_point_has_no_branch_for_is_still_measured(
    block: ContentBlock,
) -> None:
    """The rule is "everything that is not text", not a list of remembered types.

    A type missing from the count is worse than a type missing from the payload: the
    block still reaches the model, and the row then says this call's output was small
    while being silent about the part of it nobody weighed.
    """
    recorder, capture = a_capture()
    result = CallToolResult(
        content=[TextContent(type="text", text="small notes"), block]
    )

    await apply(capture, result, recorder)

    assert recorder.inline[0][1].passed_through_bytes >= len(MARKER)


def test_the_table_of_non_text_types_is_every_one_the_protocol_defines() -> None:
    """The case above is only as wide as that table, and the protocol can widen.

    A sixth content block arriving with an SDK upgrade would be counted by the rule
    either way -- that is the point of the rule -- but with no case saying so, and the
    next reader would have no way to tell a type that was thought about from one that
    was not. Failing here is how the upgrade earns a case instead of a guess.
    """
    assert {type(block) for block in NON_TEXT_BLOCKS.values()} | {TextContent} == set(
        get_args(ContentBlock)
    )


async def test_structured_content_is_weighed_stored_and_removed() -> None:
    """Serialized beside the text rather than left out: a result whose bulk is
    structured would otherwise be measured as though it were empty."""
    recorder, capture = a_capture()
    rows = {"rows": [{"n": i} for i in range(400)]}
    result = CallToolResult(content=[], structured_content=rows)

    returned = await apply(capture, result, recorder)

    assert returned.structured_content is None
    (ctx, decision) = recorder.captured[0]
    stored = recorder.objects[evidence_object_key(ctx.session_id, decision.digest)]
    assert stored == json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    assert decision.digest.byte_length == len(stored)


async def test_two_dictionaries_differing_only_in_key_order_produce_one_digest() -> (
    None
):
    """A digest a reader reproduces must not depend on a dictionary's iteration
    order."""
    first_recorder, first = a_capture()
    second_recorder, second = a_capture()
    padding = "p" * 1_000
    forwards = {"alpha": padding, "beta": padding}
    backwards = {"beta": padding, "alpha": padding}

    await apply(first, CallToolResult(content=[], structured_content=forwards))
    await apply(second, CallToolResult(content=[], structured_content=backwards))

    assert first_recorder.captured[0][1].digest == second_recorder.captured[0][1].digest


async def test_a_failed_call_stays_a_failed_call_after_a_capture() -> None:
    """`is_error` is how the Agent Runtime tells a tool that failed from one that did
    not; a capture that dropped it would turn a failure into an answer."""
    recorder, capture = a_capture()

    returned = await apply(capture, text_result("e" * 5_000, is_error=True), recorder)

    assert returned.is_error is True


async def test_a_call_still_waiting_on_input_is_not_reported_as_finished() -> None:
    """`result_type` defaults to "complete", so a captured result rebuilt from its parts
    rather than copied silently answers a call the server said was not done."""
    recorder, capture = a_capture()
    result = CallToolResult(
        content=[TextContent(type="text", text="i" * 5_000)],
        result_type="input_required",
    )

    returned = await apply(capture, result, recorder)

    assert returned.result_type == "input_required"


async def test_the_protocol_metadata_a_server_attached_survives_a_capture() -> None:
    """This module has an opinion about the payload and about nothing else, so a field
    it does not understand must reach the runtime with the value the server sent."""
    recorder, capture = a_capture()
    result = CallToolResult(
        content=[TextContent(type="text", text="j" * 5_000)],
        _meta={"upstream-trace": "abc123"},
    )

    returned = await apply(capture, result, recorder)

    assert returned.meta == {"upstream-trace": "abc123"}


async def test_an_empty_result_is_returned_inline_and_recorded_as_empty() -> None:
    recorder, capture = a_capture()

    returned = await apply(capture, CallToolResult(content=[]), recorder)

    assert returned.content == []
    assert recorder.inline[0][1].byte_length == 0
    assert recorder.objects == {}


async def test_the_capture_point_recorded_is_the_gateway_and_not_the_shim() -> None:
    """Evidence captured outside the pod carries a stronger guarantee than Evidence
    captured within a Session, and the row has to say which this is."""
    recorder, capture = a_capture()

    await apply(capture, text_result("g" * 5_000), recorder)

    assert recorder.captured[0][0].capture_point is CapturePoint.TOOL_GATEWAY
    assert recorder.captured[0][0].capture_point.value == "tool-gateway"


async def test_the_session_the_capture_is_filed_under_is_the_calling_session() -> None:
    recorder, capture = a_capture()
    session = SessionId(uuid4())

    await apply(capture, text_result("h" * 5_000), recorder, session=session)

    assert recorder.captured[0][0].session_id == session
    assert list(recorder.objects)[0].startswith(f"evidence/{session}/")
