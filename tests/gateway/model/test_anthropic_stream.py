"""Messages frames to the events the Agent Runtime parses, graded on the event list.

Tier 1, no infrastructure and no HTTP: frames are fed in as already-parsed objects,
because that is the seam the translator has -- the handler owns the SSE bytes.

The failing cases are the reason this file exists, and each asserts two things rather
than one. That the Turn fails is the easy half; that **no** `response.completed` was
yielded before it is the half that matters, because the runtime treats that event as
proof the answer is whole. A translator that failed loudly *after* emitting a terminator
would have already written the falsehood the whole classification table exists to
prevent.

The read counts are the other load-bearing assertion. A stream that fails on its first
frame and then drains the rest has still spent the tokens and still delayed the failure,
so "failed" and "failed after reading everything anyway" are graded apart.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence

import pytest

from managed_agent.core.session.markers import DiscardCause
from managed_agent.gateway.model.anthropic_request import decode_thinking
from managed_agent.gateway.model.anthropic_stream import MessagesStream
from managed_agent.gateway.model.classify import Untranslatable


async def _feed(
    frames: Sequence[Mapping[str, object]],
) -> AsyncIterator[Mapping[str, object]]:
    for frame in frames:
        yield frame


async def _events(
    frames: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [event async for event in MessagesStream().translate(_feed(frames))]


def _start(usage: Mapping[str, object] | None = None) -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {"id": "msg_01", "usage": dict(usage or {"input_tokens": 10})},
    }


def _stop(reason: str = "end_turn", usage: object = None) -> list[dict[str, object]]:
    delta: dict[str, object] = {
        "type": "message_delta",
        "delta": {"stop_reason": reason},
    }
    if usage is not None:
        delta["usage"] = usage
    return [delta, {"type": "message_stop"}]


def _text_block(index: int, *, chunks: Sequence[str]) -> list[dict[str, object]]:
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
        *[
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": chunk},
            }
            for chunk in chunks
        ],
        {"type": "content_block_stop", "index": index},
    ]


def _kinds(events: Sequence[Mapping[str, object]]) -> list[str]:
    return [str(event["type"]) for event in events]


async def test_a_text_stream_becomes_the_runtimes_own_event_sequence() -> None:
    events = await _events([_start(), *_text_block(0, chunks=["Hel", "lo"]), *_stop()])

    assert _kinds(events) == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ]
    assert [
        event["delta"]
        for event in events
        if event["type"] == "response.output_text.delta"
    ] == [
        "Hel",
        "lo",
    ]
    done = events[4]["item"]
    assert isinstance(done, dict)
    assert done["type"] == "message"
    assert done["role"] == "assistant"
    assert done["content"] == [{"type": "output_text", "text": "Hello"}]


async def test_the_created_event_carries_the_upstreams_own_response_id() -> None:
    events = await _events([_start(), *_stop()])

    assert events[0] == {"type": "response.created", "response": {"id": "msg_01"}}


async def test_a_normal_completion_reports_the_turn_as_finished() -> None:
    events = await _events(
        [_start(), *_text_block(0, chunks=["x"]), *_stop("end_turn")]
    )
    completed = events[-1]["response"]

    assert isinstance(completed, dict)
    assert completed["end_turn"] is True
    assert _kinds(events).count("response.completed") == 1


async def test_a_turn_that_stopped_for_a_tool_is_not_reported_as_finished() -> None:
    events = await _events(
        [_start(), *_text_block(0, chunks=["x"]), *_stop("tool_use")]
    )
    completed = events[-1]["response"]

    assert isinstance(completed, dict)
    assert completed["end_turn"] is False


async def test_a_streamed_tool_call_yields_nothing_until_it_is_whole() -> None:
    """The runtime ignores a partial-arguments event, so there is nowhere for half a
    call to go; the whole call ships when the block stops."""
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "call_9", "name": "read_file"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'th":"/x"}'},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop("tool_use"),
    ]

    events = await _events(frames)

    assert _kinds(events) == [
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    ]
    item = events[2]["item"]
    assert isinstance(item, dict)
    assert item == {
        "type": "function_call",
        "id": "msg_01-0",
        "call_id": "call_9",
        "name": "read_file",
        "arguments": '{"path":"/x"}',
    }


async def test_a_search_call_comes_back_as_a_search_call_and_not_a_function_call() -> (
    None
):
    """The runtime routes on the item type, and its search handler is fatal on a miss.

    codex dispatches `tool_search_call` to a handler that accepts one payload shape and
    returns `Fatal error: tool_search handler received unsupported payload` for any
    other. A `tool_use` named `tool_search` rendered as a `function_call` therefore does
    not degrade -- it kills the call, and the model is told its search came back empty.

    Measured on the cluster: the model searched, got nothing back, wrote "I searched
    extensively and it's simply not present", and finished the errand without the tool.
    Offering the tool on the way out is only half of carrying it.

    `arguments` is an object here and a JSON-encoded string on `function_call`. The two
    items disagree on that field, which is why this cannot be one branch with a
    different type string.
    """
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "s1", "name": "tool_search"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"deep'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'wiki"}'},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop("tool_use"),
    ]

    events = await _events(frames)

    item = events[2]["item"]
    assert item == {
        "type": "tool_search_call",
        "id": "msg_01-0",
        "call_id": "s1",
        "execution": "client",
        "arguments": {"query": "deepwiki"},
    }


async def test_a_search_call_with_unparsable_arguments_still_reaches_its_handler() -> (
    None
):
    """An empty object, because the handler answers that one and refuses a string.

    `query must not be empty` comes back through `RespondToModel`, which the model reads
    and can act on -- it searches again. The alternative shapes are both worse: a
    JSON-encoded string is the payload the handler calls fatal, and refusing the whole
    stream would end a Turn over one malformed call.
    """
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "s2", "name": "tool_search"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop("tool_use"),
    ]

    events = await _events(frames)

    assert events[2]["item"]["arguments"] == {}  # type: ignore[index]


async def test_a_tool_call_with_no_arguments_ships_an_empty_object() -> None:
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "c", "name": "n"},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop("tool_use"),
    ]

    events = await _events(frames)
    item = events[2]["item"]

    assert isinstance(item, dict)
    assert item["arguments"] == "{}"


async def test_a_returned_namespaced_call_carries_its_namespace_back() -> None:
    """The upstream offers one flat list, so a namespaced tool travels out under a name
    with the namespace folded in -- and the runtime resolves a call by the pair, so the
    name has to be split back or the call names a tool that side has never heard of."""
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_9",
                "name": "mcp__map_tool_gateway-ask_deepwiki",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"repoName":"a/b"}'},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop("tool_use"),
    ]

    events = await _events(frames)
    item = events[2]["item"]

    assert isinstance(item, dict)
    assert item == {
        "type": "function_call",
        "id": "msg_01-0",
        "call_id": "call_9",
        "name": "ask_deepwiki",
        "namespace": "mcp__map_tool_gateway",
        "arguments": '{"repoName":"a/b"}',
    }


async def test_a_plain_call_carries_no_namespace_field_at_all() -> None:
    """The runtime reads an absent namespace and its own default spelling identically,
    so sending the default would add a field that changes nothing -- and sending it on a
    call against a flat tool would claim a grouping that tool never had."""
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "c", "name": "exec_command"},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop("tool_use"),
    ]

    events = await _events(frames)
    item = events[2]["item"]

    assert isinstance(item, dict)
    assert "namespace" not in item


async def test_a_thinking_block_comes_back_with_its_signature_intact() -> None:
    """The signature is only valid over the block as it arrived, so the carrier holds
    the block verbatim rather than being reassembled from parts."""
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "step one "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "step two"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "AbC/123+=="},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop(),
    ]

    events = await _events(frames)

    assert _kinds(events) == [
        "response.created",
        "response.output_item.added",
        "response.reasoning_text.delta",
        "response.reasoning_text.delta",
        "response.output_item.done",
        "response.completed",
    ]
    item = events[4]["item"]
    assert isinstance(item, dict)
    assert item["type"] == "reasoning"
    assert item["summary"] == []
    carrier = item["encrypted_content"]
    assert isinstance(carrier, str)
    assert decode_thinking(carrier) == {
        "type": "thinking",
        "thinking": "step one step two",
        "signature": "AbC/123+==",
    }


async def test_a_redacted_thinking_block_round_trips_its_opaque_payload() -> None:
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque-blob"},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop(),
    ]

    events = await _events(frames)
    item = events[2]["item"]

    assert isinstance(item, dict)
    carrier = item["encrypted_content"]
    assert isinstance(carrier, str)
    assert decode_thinking(carrier) == {
        "type": "redacted_thinking",
        "data": "opaque-blob",
    }


async def test_the_later_usage_report_replaces_the_earlier_one_field_by_field() -> None:
    """`input_tokens` counts only the uncached remainder here, so the total is that plus
    both cache figures plus output -- a total copied from `input_tokens` would
    under-report every cached Turn."""
    events = await _events(
        [
            _start({"input_tokens": 10, "cache_read_input_tokens": 5}),
            *_text_block(0, chunks=["x"]),
            *_stop(
                usage={
                    "input_tokens": 10,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 3,
                    # Fields this upstream also reports, measured live. Nothing here
                    # reads them and nothing may trip over them.
                    "service_tier": "standard",
                    "inference_geo": "us",
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 3,
                        "ephemeral_1h_input_tokens": 0,
                    },
                }
            ),
        ]
    )
    response = events[-1]["response"]

    assert isinstance(response, dict)
    assert response["usage"] == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 3},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 25,
    }


async def test_a_stream_reporting_no_usage_at_all_still_totals_to_zero() -> None:
    events = await _events(
        [{"type": "message_start", "message": {"id": "m"}}, *_stop()]
    )
    response = events[-1]["response"]

    assert isinstance(response, dict)
    assert response["usage"]["total_tokens"] == 0


async def test_a_keep_alive_and_a_citation_yield_nothing_and_the_turn_completes() -> (
    None
):
    """MAP-A79: a construct whose loss cannot make an unfinished Turn look finished is
    dropped with nothing recorded, and the Turn completes normally."""
    frames: list[Mapping[str, object]] = [
        _start(),
        {"type": "ping"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {"type": "ping"},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": {"type": "web"}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop(),
    ]

    events = await _events(frames)

    assert _kinds(events) == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ]


async def _raises(
    frames: Sequence[Mapping[str, object]],
) -> tuple[Untranslatable, list[dict[str, object]]]:
    """Drive the translator to its failure, keeping the events it managed to yield."""
    seen: list[dict[str, object]] = []
    with pytest.raises(Untranslatable) as caught:
        async for event in MessagesStream().translate(_feed(frames)):
            seen.append(event)
    return caught.value, seen


@pytest.mark.parametrize(
    ("reason", "cause"),
    [
        ("max_tokens", DiscardCause.UPSTREAM_TRUNCATED),
        ("model_context_window_exceeded", DiscardCause.UPSTREAM_TRUNCATED),
        ("stop_sequence", DiscardCause.UPSTREAM_TRUNCATED),
        ("refusal", DiscardCause.UPSTREAM_REFUSED),
        ("pause_turn", DiscardCause.UPSTREAM_UNTRANSLATABLE),
    ],
)
async def test_a_stop_reason_that_is_not_the_end_of_the_answer_fails_the_turn(
    reason: str, cause: DiscardCause
) -> None:
    """MAP-A77: the Turn fails, the construct is named, and no event claims it
    completed."""
    error, seen = await _raises(
        [_start(), *_text_block(0, chunks=["par"]), *_stop(reason)]
    )

    assert error.cause is cause
    assert f"stop_reason.{reason}" in error.detail
    assert "response.completed" not in _kinds(seen)


async def test_an_error_frame_fails_the_turn_carrying_the_upstreams_own_words() -> None:
    error, seen = await _raises(
        [
            _start(),
            *_text_block(0, chunks=["half an ans"]),
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "try again later"},
            },
            {"type": "message_stop"},
        ]
    )

    assert error.cause is DiscardCause.UPSTREAM_REFUSED
    assert "overloaded_error" in error.detail
    assert "try again later" in error.detail
    assert "response.completed" not in _kinds(seen)


async def test_an_error_frame_with_no_error_object_still_fails_the_turn() -> None:
    error, _ = await _raises([_start(), {"type": "error"}])

    assert error.cause is DiscardCause.UPSTREAM_REFUSED


async def test_a_server_side_tool_block_fails_as_unclassified() -> None:
    """MAP-A78: a construct nobody classified fails rather than being dropped."""
    error, seen = await _raises(
        [
            _start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "s",
                    "name": "search",
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
    )

    assert error.cause is DiscardCause.UPSTREAM_UNCLASSIFIED
    assert "block.server_tool_use" in error.detail
    assert "response.completed" not in _kinds(seen)


async def test_a_returned_name_the_fold_cannot_reverse_fails_the_turn() -> None:
    """`a-b-c` could have been offered as three different namespace/name pairs, and
    picking one would aim the call at a tool the model was never shown -- which runs the
    wrong tool rather than running none. No terminator precedes the failure."""
    error, seen = await _raises(
        [
            _start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_9",
                    "name": "mcp__x-ask-deepwiki",
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
    )

    assert error.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE
    assert "block.tool_use.name_not_flattened" in error.detail
    assert "response.completed" not in _kinds(seen)


async def test_a_server_side_fallback_block_fails_on_what_it_costs() -> None:
    error, _ = await _raises(
        [
            _start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "fallback", "model": "another"},
            },
        ]
    )

    assert error.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE
    assert "block.fallback" in error.detail


async def test_a_stream_that_stops_without_its_terminator_fails_the_turn() -> None:
    error, seen = await _raises([_start(), *_text_block(0, chunks=["cut off here"])])

    assert error.cause is DiscardCause.UPSTREAM_TRUNCATED
    assert "stream.no_terminator" in error.detail
    assert "response.completed" not in _kinds(seen)


async def test_an_empty_stream_fails_rather_than_completing_silently() -> None:
    error, seen = await _raises([])

    assert error.cause is DiscardCause.UPSTREAM_TRUNCATED
    assert seen == []


async def test_a_delta_for_a_block_that_never_opened_fails_the_turn() -> None:
    """Frames out of this order are not a shape this wire has a row for, and the
    alternative to failing is inventing the block the delta belongs to."""
    error, _ = await _raises(
        [
            _start(),
            {
                "type": "content_block_delta",
                "index": 3,
                "delta": {"type": "text_delta", "text": "orphan"},
            },
        ]
    )

    assert error.cause is DiscardCause.UPSTREAM_UNCLASSIFIED


async def test_a_stop_for_a_block_that_never_opened_fails_the_turn() -> None:
    error, _ = await _raises([_start(), {"type": "content_block_stop", "index": 7}])

    assert error.cause is DiscardCause.UPSTREAM_UNCLASSIFIED


async def test_an_upstream_that_does_not_speak_this_wire_fails_on_its_first_frame() -> (
    None
):
    """MAP-A106: a Routing Entry declared this wire and the upstream answered in
    another. The failure names the mismatch, and the read count is what separates
    "failed" from "failed after reading the whole stream anyway"."""
    reads = 0
    recorded: tuple[Mapping[str, object], ...] = (
        {"type": "response.created", "response": {"id": "resp_upstream"}},
        {"type": "response.output_text.delta", "delta": "not this wire"},
        {"type": "response.completed"},
    )

    async def frames() -> AsyncIterator[Mapping[str, object]]:
        nonlocal reads
        for frame in recorded:
            reads += 1
            yield frame

    seen: list[dict[str, object]] = []
    with pytest.raises(Untranslatable) as caught:
        async for event in MessagesStream().translate(frames()):
            seen.append(event)

    assert caught.value.cause is DiscardCause.UPSTREAM_UNCLASSIFIED
    assert "anthropic_messages" in caught.value.detail
    assert "stream.response.created" in caught.value.detail
    assert reads == 1
    assert seen == []


async def test_two_blocks_in_one_message_each_get_their_own_item() -> None:
    events = await _events(
        [
            _start(),
            *_text_block(0, chunks=["first"]),
            *_text_block(1, chunks=["second"]),
            *_stop(),
        ]
    )
    done = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
    ]

    assert [item["id"] for item in done] == ["msg_01-0", "msg_01-1"]  # type: ignore[index]
    assert [item["content"][0]["text"] for item in done] == ["first", "second"]  # type: ignore[index]


async def test_a_block_opened_with_seed_text_keeps_it() -> None:
    """The opening frame may already carry content, and dropping it would lose the
    first characters of an answer without anything reporting a loss."""
    frames: list[Mapping[str, object]] = [
        _start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "seeded"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "-more"},
        },
        {"type": "content_block_stop", "index": 0},
        *_stop(),
    ]

    events = await _events(frames)
    item = events[3]["item"]

    assert isinstance(item, dict)
    assert item["content"] == [{"type": "output_text", "text": "seeded-more"}]


async def test_a_stream_is_spent_once_and_will_not_translate_a_second_time() -> None:
    """One instance per Turn: it holds that Turn's open blocks and its usage, so reusing
    one would fold two Turns' counts into whichever completed last."""
    stream = MessagesStream()
    frames = [_start(), *_stop()]

    assert [event async for event in stream.translate(_feed(frames))]

    with pytest.raises(RuntimeError, match="already"):
        async for _ in stream.translate(_feed(frames)):
            pass
