"""The Messages response stream, read out as the events the Agent Runtime dispatches on.

Stateful because the two streams carry the same information at different granularities:
this side opens a content block, streams deltas into it and closes it, while the runtime
wants a whole item on `response.output_item.done` -- so a block's identity and its
accumulated text or partial JSON have to be held between frames.

The terminator is synthesized, because the Messages stream ends at `message_stop` and
the runtime turns a stream closing without `response.completed` into an error. That is
only safe because every construct meaning "did not finish" has already failed the Turn
before this gets there: a stop reason that is not the end of the answer, an error frame,
a block nobody classified, or the bytes simply running out. Read the other way round,
the absence of a terminator *is* how this module reports failure, so nothing here may
yield one on a path that has not seen the whole answer.

A streamed tool argument yields no event of its own. The runtime explicitly ignores
`response.function_call_arguments.delta`, so there is nowhere for a partial call to go
until it is whole.

This module holds no HTTP client and imports none. Frames arrive as already-parsed JSON
objects, because the runtime dispatches on the `type` inside `data:` and not on the SSE
`event:` name, so the event name carries nothing this needs -- and the module that owns
the socket is the only one that has to know that.
"""

import json
import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Final

from managed_agent.gateway.model.anthropic_request import (
    SEARCH_TOOL,
    encode_thinking,
    unfold_tool_name,
)
from managed_agent.gateway.model.anthropic_table import WIRE
from managed_agent.gateway.model.classify import (
    Disposition,
    Untranslatable,
    carry,
    classify,
)

_LOG = logging.getLogger(__name__)

_USAGE_FIELDS: Final = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
"""The usage counts this wire reads.

The upstream reports more than these -- measured, it also sends `service_tier`,
`inference_geo` and a nested `cache_creation` breakdown -- and a field absent from this
tuple is ignored rather than being a surprise: the Responses usage shape has four
numbers, and reading a fifth would mean inventing somewhere to put it.
"""


@dataclass(frozen=True, slots=True)
class _OpenBlock:
    """A content block that has started and not yet stopped."""

    kind: str
    item_id: str
    call_id: str
    name: str


def _unclassified(construct: str) -> Untranslatable:
    """The failure for a frame shape this wire's table has no row for, and should not.

    `classify` synthesizes the failing row rather than reading one, so this is the
    fail-safe default doing its job rather than a gap somebody should fill: the
    alternative to failing here is inventing the state the frame refers to.
    """
    return Untranslatable(WIRE, classify(WIRE, construct))


class MessagesStream:
    """One Messages SSE stream, translated once.

    One instance per Turn: it holds that Turn's open blocks, its accumulated content and
    its usage. Translating twice through one instance is refused rather than allowed to
    fold two Turns' counts into whichever completed last: the state is per-Turn, so the
    object is too.
    """

    def __init__(self) -> None:
        self._open: dict[int, _OpenBlock] = {}
        self._parts: dict[int, list[str]] = {}
        self._signature: dict[int, str] = {}
        self._usage: dict[str, int] = {}
        self._response_id = ""
        self._stop_reason = ""
        self._closed = False
        self._started = False

    async def translate(
        self, frames: AsyncIterator[Mapping[str, object]]
    ) -> AsyncIterator[dict[str, object]]:
        """Yield the runtime's events for one upstream stream.

        An error frame is classified by hand and not through `carry`, because its own
        type and message are the one thing worth putting in the marker beside the row's
        reason -- every other failing construct is fully described by which construct it
        was.

        A frame is read, translated and yielded before the next is read, so one that
        fails on its first frame has read exactly one. That is the difference between
        failing and failing after paying for the whole answer anyway.
        """
        if self._started:
            raise RuntimeError("this MessagesStream has already translated a stream")
        self._started = True
        async for frame in frames:
            kind = str(frame.get("type", ""))
            if kind == "error":
                raise Untranslatable(
                    WIRE, classify(WIRE, "stream.error"), note=_error_note(frame)
                )
            if carry(WIRE, f"stream.{kind}") is Disposition.DROPPED:
                continue
            for event in self._on(kind, frame):
                yield event
        if not self._closed:
            raise Untranslatable(WIRE, classify(WIRE, "stream.no_terminator"))

    def _on(
        self, kind: str, frame: Mapping[str, object]
    ) -> Iterator[dict[str, object]]:
        if kind == "message_start":
            yield from self._on_message_start(frame)
        elif kind == "content_block_start":
            yield from self._on_block_start(frame)
        elif kind == "content_block_delta":
            yield from self._on_block_delta(frame)
        elif kind == "content_block_stop":
            yield from self._on_block_stop(frame)
        elif kind == "message_delta":
            self._on_message_delta(frame)
        elif kind == "message_stop":
            yield self._completed()
            self._closed = True

    def _on_message_start(
        self, frame: Mapping[str, object]
    ) -> Iterator[dict[str, object]]:
        message = _mapping(frame.get("message"))
        self._response_id = str(message.get("id", ""))
        self._absorb_usage(message.get("usage"))
        yield {"type": "response.created", "response": {"id": self._response_id}}

    def _on_block_start(
        self, frame: Mapping[str, object]
    ) -> Iterator[dict[str, object]]:
        index = _index(frame)
        block = _mapping(frame.get("content_block"))
        kind = str(block.get("type", ""))
        carry(WIRE, f"block.{kind}")
        self._open[index] = _OpenBlock(
            kind=kind,
            item_id=f"{self._response_id}-{index}",
            call_id=str(block.get("id", "")),
            name=str(block.get("name", "")),
        )
        # The opening frame may already carry content. Seeding from it rather than
        # starting empty is what stops the first characters of an answer being lost with
        # nothing reporting a loss.
        seed = str(
            block.get("text") or block.get("thinking") or block.get("data") or ""
        )
        self._parts[index] = [seed] if seed else []
        self._signature[index] = str(block.get("signature", ""))
        yield {"type": "response.output_item.added", "item": self._item(index)}

    def _on_block_delta(
        self, frame: Mapping[str, object]
    ) -> Iterator[dict[str, object]]:
        index = _index(frame)
        if index not in self._open:
            raise _unclassified("stream.content_block_delta_before_start")
        delta = _mapping(frame.get("delta"))
        kind = str(delta.get("type", ""))
        if carry(WIRE, f"delta.{kind}") is Disposition.DROPPED:
            return
        if kind == "signature_delta":
            self._signature[index] += str(delta.get("signature", ""))
            return
        if kind == "input_json_delta":
            self._parts[index].append(str(delta.get("partial_json", "")))
            return
        if kind == "thinking_delta":
            text = str(delta.get("thinking", ""))
            self._parts[index].append(text)
            yield {
                "type": "response.reasoning_text.delta",
                "delta": text,
                "content_index": index,
            }
            return
        text = str(delta.get("text", ""))
        self._parts[index].append(text)
        yield {
            "type": "response.output_text.delta",
            "delta": text,
            "item_id": self._open[index].item_id,
        }

    def _on_block_stop(
        self, frame: Mapping[str, object]
    ) -> Iterator[dict[str, object]]:
        index = _index(frame)
        if index not in self._open:
            raise _unclassified("stream.content_block_stop_before_start")
        yield {"type": "response.output_item.done", "item": self._item(index)}
        self._open.pop(index, None)
        self._parts.pop(index, None)
        self._signature.pop(index, None)

    def _on_message_delta(self, frame: Mapping[str, object]) -> None:
        delta = _mapping(frame.get("delta"))
        stop = delta.get("stop_reason")
        if isinstance(stop, str) and stop:
            carry(WIRE, f"stop_reason.{stop}")
            self._stop_reason = stop
        self._absorb_usage(frame.get("usage"))

    def _item(self, index: int) -> dict[str, object]:
        """Render an open block as the runtime's own item shape.

        A thinking block goes back inside its carrier verbatim rather than field by
        field, because the signature is only valid over the block as it arrived --
        reassembling one from parts would produce something the upstream refuses on the
        next Turn.
        """
        block = self._open[index]
        body = "".join(self._parts[index])
        if block.kind == "tool_use" and block.name == SEARCH_TOOL:
            return _search_call(block.item_id, block.call_id, body)
        if block.kind == "tool_use":
            namespace, name = unfold_tool_name(block.name)
            call: dict[str, object] = {
                "type": "function_call",
                "id": block.item_id,
                "call_id": block.call_id,
                "name": name,
                "arguments": body or "{}",
            }
            if namespace is not None:
                # Only when the name carried one. The runtime reads an absent namespace
                # and its own name for "no namespace" identically, so sending the
                # default spelling here would add a field that changes nothing -- and
                # sending it on a call the model made against a plain tool would claim
                # a grouping that tool never had.
                call["namespace"] = namespace
            return call
        if block.kind in ("thinking", "redacted_thinking"):
            verbatim: dict[str, object] = {"type": block.kind}
            if block.kind == "thinking":
                verbatim["thinking"] = body
                verbatim["signature"] = self._signature[index]
            else:
                verbatim["data"] = body
            return {
                "type": "reasoning",
                "id": block.item_id,
                "summary": [],
                "encrypted_content": encode_thinking(verbatim),
            }
        return {
            "type": "message",
            "id": block.item_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": body}],
        }

    def _absorb_usage(self, usage: object) -> None:
        """Take the latest counts reported for this message.

        Usage arrives twice -- input-side counts on the opening frame, then a cumulative
        object on each `message_delta`, so a later report replaces an earlier one field
        by field rather than being added to it.
        """
        if not isinstance(usage, Mapping):
            return
        for field in _USAGE_FIELDS:
            value = usage.get(field)
            if isinstance(value, int):
                self._usage[field] = value

    def _completed(self) -> dict[str, object]:
        """The terminator the runtime insists on, built from the last frame.

        `total_tokens` is summed, not read: on this wire `input_tokens` counts only
        the uncached remainder, so the prompt's size is that plus both cache figures,
        and a total copied from `input_tokens` would under-report every cached Turn. The
        reasoning detail is zero, not guessed -- thinking is billed as output here
        and no separate count is reported.
        """
        if self._open:
            _LOG.warning(
                "message_stop arrived with %d content block(s) still open",
                len(self._open),
            )
        prompt = self._usage.get("input_tokens", 0)
        cached = self._usage.get("cache_read_input_tokens", 0)
        written = self._usage.get("cache_creation_input_tokens", 0)
        output = self._usage.get("output_tokens", 0)
        return {
            "type": "response.completed",
            "response": {
                "id": self._response_id,
                "end_turn": self._stop_reason == "end_turn",
                "usage": {
                    "input_tokens": prompt,
                    "input_tokens_details": {
                        "cached_tokens": cached,
                        "cache_write_tokens": written,
                    },
                    "output_tokens": output,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": prompt + cached + written + output,
                },
            },
        }


def _search_call(item_id: str, call_id: str, arguments: str) -> dict[str, object]:
    """A call on the search tool, as the item the runtime routes it by.

    Not a `function_call`. The runtime dispatches on the item type and its search
    handler accepts one payload shape, returning `Fatal error: tool_search handler
    received unsupported payload` for every other -- so the wrong item type does not
    degrade the call, it kills it, and the model is told its search found nothing.

    `arguments` is a real object here, where `function_call` carries a JSON-encoded
    string. An unparsable body becomes an empty object rather than a refusal: the
    handler answers that with `query must not be empty` through the channel the model
    reads, so the model searches again, where failing the stream would end the Turn.

    `execution` is stated because the item's shape requires it and the answer is not in
    doubt: this tool runs inside the runtime, over its own registry of deferred tools,
    and no request for it ever leaves the pod.
    """
    try:
        parsed = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        parsed = {}
    return {
        "type": "tool_search_call",
        "id": item_id,
        "call_id": call_id,
        "execution": "client",
        "arguments": parsed if isinstance(parsed, dict) else {},
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _index(frame: Mapping[str, object]) -> int:
    value = frame.get("index")
    return value if isinstance(value, int) else 0


def _error_note(frame: Mapping[str, object]) -> str:
    """The upstream's own words about its own failure, bounded.

    Bounded because it lands in a marker's free text, which is capped, and an upstream
    that answers with a page of prose should not be the reason a marker cannot be
    written.
    """
    error = _mapping(frame.get("error"))
    return f"{error.get('type', 'unknown')}: {error.get('message', '')}"[:400]
