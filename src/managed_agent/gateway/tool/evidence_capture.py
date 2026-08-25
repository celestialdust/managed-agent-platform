"""Classifying a registered server's result before the Agent Runtime ever sees it.

This runs at the Tool Gateway, outside the pod, on the way back from a registered MCP
server — the only point an enterprise result passes that the pod cannot reach, which is
what makes Evidence captured here stronger than Evidence captured within a Session.

What the payload covers is deliberate. Text blocks carry a tool's bulk output and are
joined in order; structured content is serialized beside them with sorted keys and no
spaces, so the digest a reader reproduces does not depend on a dictionary's iteration
order. Both are measured, both are stored, and when the result is captured both are
removed from what goes back — so nothing that was weighed is left on the wire. Non-text
blocks pass through untouched: an image or an embedded resource carries its own size
discipline in the protocol, and folding one into a text digest would produce bytes no
reader could reconstruct.

So something *is* left on the wire unweighed, and the ledger says how much. A row that
recorded only the weighed octets would answer "was this call's output small?" with the
size of the part it happened to look at — and a 200 KB embedded resource beside eleven
bytes of text would be filed as eleven bytes, below the threshold, which is a worse
record than none because it answers the reviewer's question wrongly. Every row carries
the octets that went back unweighed alongside the octets that were weighed, so the two
questions a reviewer has are two numbers rather than one number and an assumption.

Below the threshold nothing is substituted. The result goes back exactly as the server
sent it and the ledger still gets a row recording its size and the threshold it was
measured against, so a reviewer who finds no Evidence for a call can see that it was
small rather than that the record is missing.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from mcp.types import CallToolResult, ContentBlock, TextContent, Tool

from managed_agent.core.ids import SessionId
from managed_agent.core.vfs.evidence import (
    CaptureContext,
    CapturePoint,
    CaptureThreshold,
    EvidenceRecorder,
    EvidenceRef,
    ReturnInline,
    decide,
)


def advertisable(tool: Tool) -> Tool:
    """One registered tool as this Gateway may honestly offer it to the Agent Runtime.

    An output schema is dropped, and that is the whole of it. A declared output schema
    is a promise that every successful call returns structured content conforming to it,
    and the MCP client enforces that promise for the caller without being asked: it
    caches the schema from the listing and revalidates every non-error result, raising
    when structured content is absent. A capture replaces the result — the payload not
    crossing is the point of it — so the promise is one this Gateway cannot keep for any
    tool whose output might be large, which is every tool. Forwarding the schema anyway
    turns each large result from a schema-declaring server into a raised exception
    on the runtime side instead of an Evidence reference, and turns the declaration
    into an opt-out of capture that the upstream, not the platform, controls.

    Declining to advertise it is the fail-safe of the two. The schema is optional in the
    protocol, a client given none simply does not validate, and structured content still
    reaches the agent untouched on every result small enough to be returned inline. What
    is lost is a shape hint the model would have had. Keeping the hint would mean either
    leaving bulk output on the wire whenever a server declares a schema, or
    advertising a schema of this Gateway's own composition — a promise about a shape
    two parties would have to agree on, checkable here against exactly one client.
    """
    return tool.model_copy(update={"output_schema": None})


@dataclass(frozen=True, slots=True)
class _Measured:
    """One result split by whether this capture point weighs a part of it or not.

    The blocks that pass through and the count of their octets come out of one
    reading of one result, so the number the ledger records and the blocks the caller
    receives cannot describe different things.
    """

    payload: bytes
    passed_through: Sequence[ContentBlock]
    passed_through_bytes: int


def _measure(result: CallToolResult) -> _Measured:
    """Weigh the text and the structured content; count what goes back unweighed.

    The unweighed count is each surviving block's own JSON, which is the form it travels
    in, so the number is the octets that block costs the caller rather than a guess at
    the size of whatever it holds — a base64 image is counted at its encoded length
    because that is what crosses. Every block that is not text is counted by the same
    rule, so a content type this codebase has never seen is measured rather than
    silently weighing nothing.
    """
    weighed: list[str] = []
    passed: list[ContentBlock] = []
    for block in result.content:
        if isinstance(block, TextContent):
            weighed.append(block.text)
        else:
            passed.append(block)
    if result.structured_content is not None:
        weighed.append(
            json.dumps(result.structured_content, sort_keys=True, separators=(",", ":"))
        )
    return _Measured(
        payload="\n".join(weighed).encode("utf-8"),
        passed_through=passed,
        passed_through_bytes=sum(
            len(block.model_dump_json(by_alias=True).encode("utf-8"))
            for block in passed
        ),
    )


def _reference_block(ref: EvidenceRef) -> TextContent:
    """What the model reads in place of a large result.

    A plain sentence rather than a JSON envelope: the Agent Runtime hands tool content
    to the model as text, and a shape the model must parse before it can act is a shape
    it can get wrong. The path is absolute and readable from inside the sandbox.
    """
    return TextContent(
        type="text",
        text=(
            f"This tool returned {ref.digest.byte_length} bytes, at or above the"
            " platform's capture threshold, so its output was written into the"
            f" session's own file tree instead of being returned here. Read it at"
            f" {ref.vfs_path}. Its {ref.digest.algorithm} digest is {ref.digest.hex}."
        ),
    )


class EvidenceCapture:
    """The Tool Gateway's capture point: one threshold, applied to every result."""

    def __init__(self, recorder: EvidenceRecorder, threshold: CaptureThreshold) -> None:
        self._recorder = recorder
        self._threshold = threshold

    @property
    def threshold(self) -> CaptureThreshold:
        """The size this point captures at. Readable so the wiring can be checked.

        Both capture points must be constructed from one reading of the one variable,
        and a threshold nobody can observe is a threshold nothing can assert agreement
        on. A query with no side effects: this cannot be set.
        """
        return self._threshold

    async def apply(
        self,
        session_id: SessionId,
        call_id: str,
        tool_name: str,
        result: CallToolResult,
    ) -> CallToolResult:
        """Return what the Agent Runtime should receive, once the record is durable.

        Everything this awaits happens before the returned value exists, so there is no
        ordering for a caller to preserve: by the time a result is handed back, the
        bytes behind it are stored and the ledger row is written.
        """
        ctx = CaptureContext(
            session_id=session_id,
            call_id=call_id,
            tool_name=tool_name,
            capture_point=CapturePoint.TOOL_GATEWAY,
        )
        measured = _measure(result)
        decision = decide(
            measured.payload,
            self._threshold,
            passed_through_bytes=measured.passed_through_bytes,
        )
        if isinstance(decision, ReturnInline):
            await self._recorder.record_inline(ctx, decision)
            return result
        ref = await self._recorder.record_captured(ctx, measured.payload, decision)
        # Copied rather than rebuilt, so every field this module has no opinion about
        # survives with the value the server sent. Two of them are load-bearing:
        # `is_error` is how the Agent Runtime tells a tool that failed from one that
        # did not, and `result_type` can say `input_required`, which a reconstruction
        # would quietly reset to `complete` and turn a half-finished call into an
        # answer. Only the two things that were weighed are cleared.
        #
        # Clearing `structured_content` is legal only because `advertisable` above keeps
        # this Gateway from ever declaring an output schema for the tool. The two belong
        # together: restore the schema to the listing and this line becomes a raised
        # `RuntimeError` in the caller's MCP client for every large result.
        return result.model_copy(
            update={
                "content": [_reference_block(ref), *measured.passed_through],
                "structured_content": None,
            }
        )
