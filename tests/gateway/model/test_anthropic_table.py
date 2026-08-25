"""The Anthropic wire's table, graded for exhaustiveness rather than for taste.

Tier 1, no infrastructure. The inventories below -- the item types the Agent Runtime can
emit, the stream events and delta kinds this upstream can send, the stop reasons it can
report -- are written out here so that the table and the thing the table is about are
two independent statements. A test that read the inventory out of the table would say
only that the table equals itself.

The three absences at the bottom are as load-bearing as the presences. An Azure-hosted
Foundry deployment refuses the server-side tool features by design, so one of those
blocks arriving means the deployment is not what its Routing Entry says -- and failing
as unclassified is the right answer to that rather than a gap somebody should fill in.
"""

from __future__ import annotations

import importlib

from managed_agent.core.session.markers import DiscardCause
from managed_agent.gateway.model.anthropic_table import WIRE
from managed_agent.gateway.model.classify import Disposition, classified, classify

RUNTIME_ITEM_TYPES = frozenset(
    {
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_search_call",
        "tool_search_output",
        "local_shell_call",
        "web_search_call",
        "image_generation_call",
        "additional_tools",
        "agent_message",
        "compaction",
        "compaction_trigger",
        "context_compaction",
    }
)
"""Every `type` string the Agent Runtime can put in the `input` array of a request."""

STREAM_EVENTS = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
        "error",
    }
)
"""Every event name the Messages streaming response can carry."""

DELTA_KINDS = frozenset(
    {
        "text_delta",
        "thinking_delta",
        "signature_delta",
        "input_json_delta",
        "citations_delta",
    }
)

STOP_REASONS = frozenset(
    {
        "end_turn",
        "tool_use",
        "max_tokens",
        "model_context_window_exceeded",
        "stop_sequence",
        "refusal",
        "pause_turn",
    }
)

SERVER_SIDE_TOOL_BLOCKS = frozenset(
    {"server_tool_use", "mcp_tool_use", "web_search_tool_result", "mcp_tool_result"}
)
"""Blocks a Foundry deployment answers 400 for. Deliberately absent from the table."""


def test_every_item_type_the_runtime_emits_has_a_row() -> None:
    rows = classified(WIRE)
    missing = {kind for kind in RUNTIME_ITEM_TYPES if f"item.{kind}" not in rows}
    assert not missing


def test_every_stream_event_has_a_row() -> None:
    rows = classified(WIRE)
    missing = {name for name in STREAM_EVENTS if f"stream.{name}" not in rows}
    assert not missing


def test_every_delta_kind_has_a_row() -> None:
    rows = classified(WIRE)
    missing = {kind for kind in DELTA_KINDS if f"delta.{kind}" not in rows}
    assert not missing


def test_a_stream_that_stops_without_its_terminator_is_classified() -> None:
    """Not an event name -- the absence of one -- and it still needs a row, because
    "the bytes ran out" is the failure a synthesized terminator would paper over."""
    row = classify(WIRE, "stream.no_terminator")

    assert row.disposition is Disposition.FAILS
    assert row.cause is DiscardCause.UPSTREAM_TRUNCATED


def test_every_stop_reason_has_a_row_and_only_two_of_them_are_carried() -> None:
    """Five of the seven mean the answer stopped somewhere; two mean it finished."""
    rows = {reason: classify(WIRE, f"stop_reason.{reason}") for reason in STOP_REASONS}

    carried = {
        reason
        for reason, row in rows.items()
        if row.disposition is not Disposition.FAILS
    }
    assert carried == {"end_turn", "tool_use"}
    assert all(row.construct in classified(WIRE) for row in rows.values())


def test_exactly_seven_item_types_are_translated() -> None:
    """Seven kinds of item cross, and one sub-construct of `message` crosses with them.

    The extra entry is not an eighth item type. `item.message.role_system_level` is a
    developer- or system-role message, which is an `item.message` whose content this
    wire carries in the top-level `system` field instead of in the messages array -- so
    it is translated, and it is translated into a different part of the request from
    every other row here.

    Named in this set rather than filtered out of it, because the alternative is a test
    that grades item types and silently stops grading sub-constructs.

    This set has now grown twice, both times because a DROPPED row's stated reason was
    false and the set looked complete while it stood -- which is the whole argument for
    spelling the members out rather than asserting a count. The three added this time
    are the deferred-discovery items, and their absence cost every granted MCP tool on
    nineteen consecutive live Turns.
    """
    translated = {
        construct.removeprefix("item.")
        for construct, row in classified(WIRE).items()
        if construct.startswith("item.") and row.disposition is Disposition.TRANSLATED
    }

    assert translated == {
        "message",
        "message.role_system_level",
        "reasoning",
        "function_call",
        "function_call_output",
        "tool_search_call",
        "tool_search_output",
        "additional_tools",
    }


def test_only_the_schemaless_tool_kind_cannot_cross() -> None:
    """The three sit together and look alike, and one of them cannot cross.

    A namespace is pure grouping around ordinary function tools, so unwrapping it loses
    nothing the model needs. `custom` is a free-text tool with no schema, so there is
    nothing on this side for it to become and carrying it would mean offering a tool
    that cannot be called.

    `tool_search` was dropped beside it on the reading that it "asks the upstream to
    discover tools mid-turn". That reading was measurably wrong and is corrected here:
    `create_tool_search_tool` sets `execution: "client"` and the handler is a local BM25
    search over the runtime's own registry, so the provider is never asked anything. The
    cost of the wrong reading was not a missing niche feature -- codex defers every MCP
    tool behind this one and cannot be configured not to, so dropping it dropped the
    tenant's whole granted catalogue on every Turn.
    """
    assert classify(WIRE, "tool.namespace").disposition is Disposition.TRANSLATED
    assert classify(WIRE, "tool.tool_search").disposition is Disposition.TRANSLATED
    assert classify(WIRE, "tool.custom").disposition is Disposition.DROPPED


def test_an_unfoldable_tool_name_fails_rather_than_being_offered() -> None:
    """A name that cannot be folded and split back has no safe spelling on this wire:
    two pairs would share one offered name, and a returned call would land on whichever
    tool the split guessed at."""
    for construct in (
        "tool.name_not_flattenable",
        "tool.name_too_long",
        "block.tool_use.name_not_flattened",
    ):
        row = classify(WIRE, construct)
        assert row.disposition is Disposition.FAILS, construct
        assert row.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE, construct


def test_a_server_side_tool_block_fails_as_unclassified() -> None:
    for block in sorted(SERVER_SIDE_TOOL_BLOCKS):
        row = classify(WIRE, f"block.{block}")
        assert row.disposition is Disposition.FAILS, block
        assert row.cause is DiscardCause.UPSTREAM_UNCLASSIFIED, block


def test_a_fallback_fails_on_what_it_costs_not_on_where_it_stopped() -> None:
    """A fallback bills per attempt in `usage.iterations[]`, which the Responses usage
    shape has nowhere to put, so flattening it under-reports what the Turn cost."""
    row = classify(WIRE, "block.fallback")

    assert row.disposition is Disposition.FAILS
    assert row.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE


def test_the_keep_alive_is_dropped_with_nothing_recorded() -> None:
    assert classify(WIRE, "stream.ping").disposition is Disposition.DROPPED
    assert classify(WIRE, "delta.citations_delta").disposition is Disposition.DROPPED


def test_importing_the_table_module_twice_installs_one_table() -> None:
    before = classified(WIRE)
    importlib.import_module("managed_agent.gateway.model.anthropic_table")

    assert classified(WIRE) is before


def test_every_row_carries_a_reason_somebody_can_re_derive() -> None:
    for construct, row in classified(WIRE).items():
        assert row.why.strip(), construct
        assert row.construct == construct
