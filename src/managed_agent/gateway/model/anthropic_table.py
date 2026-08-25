"""What crosses the Anthropic Messages wire, construct by construct.

The two formats do not cover each other, and the gaps differ in consequence rather than
in size: a lost `ping` changes nothing, a lost citation weakens an answer, and a
flattened `pause_turn` records a Turn as finished that is still waiting. Only the third
kind is worth failing over, which is why one rule sorts all of them and why nothing here
is decided per request (ADR-009).

Terse on purpose -- the reasoning shape is one rule applied seventy-five times, and each
row's `why` says only what that rule decided for that construct. Four groups earn a
second look.

`tool.namespace` is carried by folding, not by grouping: this side's tools are one flat
list, so a namespace's members are offered individually under names its own name is
folded into, and a returned call is split back. That is why the two names beside it fail
rather than being repaired -- a fold that cannot be reversed uniquely would offer two
tools under one name, and a call landing on whichever of them the split guessed at is
worse than a call that never ran.

`tool.tool_search` sits in the same group and is carried, having been dropped here for
a while on the reading that it asks the upstream to discover tools mid-turn. It does
not: it executes on the client, over the runtime's own registry, and the provider is
told nothing. The distinction is load-bearing rather than pedantic, because the runtime
puts no MCP tool in the request's tool list -- it offers this one tool and defers the
rest -- so this row and the three items beside it (`tool_search_call`,
`tool_search_output`, `additional_tools`) are the only way a granted tool reaches a
model on this wire at all. While they were dropped, nineteen consecutive live Turns
were served a model that could see its granted tool named in prose and could not call
it.

`block.fallback` fails on the *spend* property rather than the completion one: a
server-side fallback bills per attempt in `usage.iterations[]`, which the Responses
usage shape has nowhere to put, and ADR-016 widened the fail-safe to cover exactly that.

The seven `stop_reason` values are classified one at a time, because five of the seven
mean the answer stopped somewhere and only two mean it finished.

And the server-side tool blocks -- `server_tool_use`, `mcp_tool_use`, the
`*_tool_result` family -- deliberately have **no** rows. An Azure-hosted Foundry
deployment returns `400 Bad Request` for those features by design, so one arriving means
the deployment is not what its Routing Entry says; failing as unclassified is the right
answer to that rather than a gap in this table.

This module holds knowledge and no behaviour, which is why it is not the translator. The
table is installed on import, so importing either direction's translator is enough to
have the rows in place, and nothing has to remember to call an initialiser.
"""

from typing import Final

from managed_agent.core.session.markers import DiscardCause
from managed_agent.gateway.model.classify import (
    Classification,
    Disposition,
    register_table,
)
from managed_agent.gateway.model.router import UpstreamWire

WIRE: Final = UpstreamWire.ANTHROPIC_MESSAGES
"""The wire these rows belong to, named once so no call site spells it out again."""

ANTHROPIC_VERSION: Final = "2023-06-01"
"""The dated API version this wire's translation was written against.

A version pin, not a guess and not a default. The Messages endpoint requires the header
on every request and answers 400 without it, and the Agent Runtime cannot supply it --
it speaks Responses and has no such field -- so the value has to be stated somewhere in
this service. Here rather than in the handler, because the request and response shapes
this table classifies are the shapes *this version* defines: moving the pin without
re-reading the rows would leave the table describing a wire that had changed underneath
it.
"""


def _t(construct: str, why: str) -> Classification:
    return Classification(construct, Disposition.TRANSLATED, why)


def _d(construct: str, why: str) -> Classification:
    return Classification(construct, Disposition.DROPPED, why)


def _f(construct: str, cause: DiscardCause, why: str) -> Classification:
    return Classification(construct, Disposition.FAILS, why, cause)


ROWS: Final[tuple[Classification, ...]] = (
    # ---- the response stream's own frames ----------------------------------------
    _t("stream.message_start", "opens the response and carries the input-side usage"),
    _t("stream.content_block_start", "opens one output item"),
    _t("stream.content_block_delta", "carries one item's incremental content"),
    _t("stream.content_block_stop", "closes one output item"),
    _t("stream.message_delta", "carries the stop reason and the cumulative usage"),
    _t("stream.message_stop", "the frame response.completed is synthesized from"),
    _d("stream.ping", "a keep-alive with no counterpart; nothing downstream reads it"),
    _f(
        "stream.error",
        DiscardCause.UPSTREAM_REFUSED,
        "an error frame mid-stream means the answer stopped, and the runtime has "
        "no arm for one, so carrying on ends in a synthesized completion for a "
        "Turn that never finished",
    ),
    _f(
        "stream.no_terminator",
        DiscardCause.UPSTREAM_TRUNCATED,
        "the byte stream ended with no message_stop, so where the answer stopped is "
        "unknowable",
    ),
    # ---- delta kinds inside content_block_delta -----------------------------------
    _t("delta.text_delta", "becomes response.output_text.delta"),
    _t("delta.thinking_delta", "becomes response.reasoning_text.delta"),
    _t(
        "delta.signature_delta",
        "held, and re-emitted inside the block's opaque carrier",
    ),
    _t(
        "delta.input_json_delta",
        "accumulated; the whole call ships when the block stops",
    ),
    _d(
        "delta.citations_delta",
        "no counterpart; a lost citation weakens an answer without changing "
        "whether the Turn finished",
    ),
    # ---- output content blocks ----------------------------------------------------
    _t("block.text", "becomes a message item carrying one output_text"),
    _t(
        "block.thinking",
        "becomes a reasoning item whose carrier holds the block verbatim",
    ),
    _t(
        "block.redacted_thinking",
        "the same carrier; the payload is opaque on both sides",
    ),
    _t(
        "block.tool_use",
        "becomes a function_call whose arguments are the re-encoded input",
    ),
    _f(
        "block.tool_use.name_not_flattened",
        DiscardCause.UPSTREAM_UNTRANSLATABLE,
        "a returned name this wire never offered does not split back into a namespace "
        "and a name, and choosing one would aim the call at a tool the model was never "
        "shown",
    ),
    _f(
        "block.fallback",
        DiscardCause.UPSTREAM_UNTRANSLATABLE,
        "a server-side fallback bills per attempt in usage.iterations[], which the "
        "Responses usage shape cannot carry, so flattening it under-reports what the "
        "Turn cost",
    ),
    # ---- stop reasons, one row per value ------------------------------------------
    _t(
        "stop_reason.end_turn",
        "the model finished; the terminator carries end_turn true",
    ),
    _t(
        "stop_reason.tool_use",
        "the model wants a tool; the Turn continues at the runtime",
    ),
    _f(
        "stop_reason.max_tokens",
        DiscardCause.UPSTREAM_TRUNCATED,
        "the answer was cut at a token cap and is not the whole answer",
    ),
    _f(
        "stop_reason.model_context_window_exceeded",
        DiscardCause.UPSTREAM_TRUNCATED,
        "the prompt outgrew the window, so part of the question was answered or none",
    ),
    _f(
        "stop_reason.stop_sequence",
        DiscardCause.UPSTREAM_TRUNCATED,
        "this wire sends no stop_sequences, so a stop at one came from elsewhere",
    ),
    _f(
        "stop_reason.refusal",
        DiscardCause.UPSTREAM_REFUSED,
        "a refusal arrives as HTTP 200 and would otherwise read as a short answer",
    ),
    _f(
        "stop_reason.pause_turn",
        DiscardCause.UPSTREAM_UNTRANSLATABLE,
        "the pause-and-resume protocol has no counterpart at all, so a paused Turn "
        "would be recorded as a finished one",
    ),
    # ---- request envelope fields the runtime sends --------------------------------
    _t("request.stream", "the runtime hardcodes it true and this wire sends it true"),
    _t("request.instructions", "becomes the top-level system field as one text block"),
    _t("request.tool_choice", "the plain string becomes tool_choice's object form"),
    _t(
        "request.parallel_tool_calls",
        "false becomes tool_choice.disable_parallel_tool_use",
    ),
    _t(
        "request.text.format",
        "the output schema becomes output_config.format; the strict and name keys "
        "beside it are Responses-only and are left off",
    ),
    _d("request.text.verbosity", "no counterpart; verbosity cannot end a Turn early"),
    _t("request.reasoning.summary", "asking for a summary becomes thinking.display"),
    _d(
        "request.reasoning.effort",
        "both effort scales are closed and neither documents a correspondence; "
        "guessing one changes how much the model thinks with nobody choosing",
    ),
    _d(
        "request.reasoning.context",
        "names how much reasoning the server replays, which this wire replays itself "
        "from the carrier",
    ),
    _d(
        "request.store",
        "Messages is stateless, so store:false asserts what is already true",
    ),
    _d(
        "request.include",
        "asks the server for encrypted reasoning; this wire returns thinking blocks "
        "unconditionally, so the ask is already satisfied",
    ),
    _d(
        "request.prompt_cache_key",
        "one opaque key against per-block breakpoints; this wire places the "
        "breakpoints itself",
    ),
    _d(
        "request.client_metadata",
        "two dozen keys naming platform internals; forwarding them would put our own "
        "identifiers on a provider's wire",
    ),
    _d("request.stream_options", "its one field is sent only to first-party OpenAI"),
    _d(
        "request.service_tier",
        "a request field here against a usage response field there; not two ends "
        "of one mapping",
    ),
    # ---- tool specifications ------------------------------------------------------
    _t(
        "tool.function",
        "parameters becomes input_schema; the rest of the object matches",
    ),
    _d(
        "tool.custom",
        "a freeform tool is constrained by a grammar rather than a schema and its call "
        "arrives as raw text, and this side has no field for either; the cost is that "
        "the runtime offers apply_patch this way and its own base_instructions teach "
        "the model to use it, so the model arrives instructed in an editing tool it "
        "was never handed -- accepted because shell editing through exec_command "
        "covers the capability, and paid every Turn rather than once",
    ),
    _t(
        "tool.namespace",
        "this side offers one flat list, so each member becomes its own tool under a "
        "name the namespace is folded into and the returned call is split back",
    ),
    _f(
        "tool.name_not_flattenable",
        DiscardCause.UPSTREAM_UNTRANSLATABLE,
        "a name outside [A-Za-z0-9_] cannot be folded into a namespaced one without "
        "two pairs sharing an offered name, and a returned call the split cannot aim "
        "at one tool would run whichever tool it landed on",
    ),
    _f(
        "tool.name_too_long",
        DiscardCause.UPSTREAM_UNTRANSLATABLE,
        "the folded name passes this side's 128-character ceiling, and shortening it "
        "to fit is what makes two tools share one name",
    ),
    _t(
        "tool.tool_search",
        "the runtime defers every MCP tool behind this one and offers it in their "
        "place, so dropping it drops the whole granted catalogue; it executes on the "
        'client (`execution: "client"`, a local BM25 search over the runtime\'s own '
        "registry) and asks the provider for nothing, so it carries as a function tool",
    ),
    _d(
        "tool.web_search",
        "the option set carries no counterpart, and the server-side tool it names is "
        "refused by design on an Azure-hosted deployment",
    ),
    # ---- conversation items the runtime replays -----------------------------------
    _t("item.message", "becomes a user or assistant message with translated blocks"),
    _t(
        "item.message.role_system_level",
        "a developer- or system-role message is system-level content, and this wire "
        "takes system-level content in the top-level `system` field rather than in the "
        "messages array -- so it is carried there, after the instructions, in the "
        "order it arrived",
    ),
    _d(
        "item.message.role_not_conversational",
        "the messages array admits user and assistant only, and the two system-level "
        "roles are carried in `system` by the row above; a role that is neither has no "
        "field on this wire to go in. This row's earlier rationale said the runtime "
        "used another role only for a prefix carrier whose content it also sent in "
        "`instructions` -- that was false, and while it stood the runtime's skills "
        "catalogue was dropped on every Turn with nothing anywhere saying so",
    ),
    _t(
        "item.reasoning",
        "the carrier is unpacked back into the block it was written from",
    ),
    _d(
        "item.reasoning.no_carried_block",
        "a reasoning item with no carrier this wire wrote -- plaintext reasoning, or a "
        "carrier from another wire -- has no block to restore",
    ),
    _t("item.function_call", "becomes a tool_use block in an assistant message"),
    _f(
        "item.function_call.arguments_not_json",
        DiscardCause.UPSTREAM_UNTRANSLATABLE,
        "tool_use.input is an object and the arguments string will not parse into one, "
        "so the request would be refused upstream and the Turn would never run",
    ),
    _t("item.function_call_output", "becomes a tool_result block in a user message"),
    _d(
        "item.custom_tool_call",
        "the return leg of a tool this wire does not offer; the runtime has the item "
        "for it, so the drop is the tool decision reaching its second leg rather than "
        "a limit of this side -- carry tool.custom and this row is what has to move "
        "with it",
    ),
    _d(
        "item.custom_tool_call_output",
        "the other half of a call this wire drops",
    ),
    _t(
        "item.tool_search_call",
        "a tool_use block, like any other call; its arguments arrive as an object "
        "rather than the JSON-encoded string a function_call carries",
    ),
    _t(
        "item.tool_search_output",
        "a tool_result block, and its `tools` are folded into the request's tool "
        "array -- a discovered tool left inside the item is one the model was told "
        "about and still cannot call",
    ),
    _d(
        "item.local_shell_call",
        "a shell call the runtime already ran and recorded itself",
    ),
    _d("item.web_search_call", "open_page and find_in_page have no counterpart action"),
    _d("item.image_generation_call", "no counterpart item type"),
    _t(
        "item.additional_tools",
        "carries a tool list inline; this wire sends tools in the top-level array, so "
        "the list is folded into it rather than dropped with the item",
    ),
    _d("item.agent_message", "author and recipient routing has no counterpart"),
    _d(
        "item.compaction",
        "an encrypted history summary this wire cannot re-encode; the history it "
        "stands for is shorter, never falsely finished",
    ),
    _d("item.compaction_trigger", "a request control, not a durable item"),
    _d(
        "item.context_compaction",
        "as compaction: an encrypted summary with no counterpart",
    ),
    # ---- content blocks inside those items ----------------------------------------
    _t("content.input_text", "becomes a text block"),
    _t("content.output_text", "becomes a text block"),
    _t("content.input_image.data_url", "becomes an image block with a base64 source"),
    _d(
        "content.input_image.remote_url",
        "the runtime prepares images into data URLs, so a bare remote URL is not a "
        "shape it emits and no source form is graded for one",
    ),
    _d("content.input_audio", "no counterpart content type"),
    _d(
        "content.encrypted_content",
        "an opaque channel for cross-agent content this wire cannot re-encode",
    ),
)

register_table(WIRE, ROWS)
