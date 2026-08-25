"""One Turn's request, rewritten from the Responses shape into the Messages shape.

The Agent Runtime speaks exactly one protocol and builds its request URL from a
compile-time path constant, so a Foundry-hosted Anthropic model cannot be reached
without rewriting the body. Nothing here is decided per request: every field either has
a row in this wire's table or fails the Turn, so what this module contributes is the
mechanical part -- which field becomes which, and where the cache breakpoints go.

Two shape differences do most of the work. Tool results are top-level items on the Agent
Runtime's side and `tool_result` blocks inside a *user* message here, so an output has
to be attached to a message rather than appended as one. And `max_tokens` is required
on every Messages request while the runtime's request has no field for it at all, so it
is a parameter of this function: there is nothing to translate, and a default here would
be a cap on every Turn that nobody chose.

This module holds no HTTP client and imports none. It builds a body and returns it, so
there is no endpoint it could ask what it is -- which is the property that makes a
model's wire a thing its Routing Entry declares rather than one this service detects.
"""

import base64
import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from managed_agent.core.ids import SessionId
from managed_agent.gateway.model.anthropic_table import WIRE
from managed_agent.gateway.model.classify import (
    Disposition,
    Untranslatable,
    carry,
    classify,
)

_LOG = logging.getLogger(__name__)

_DATA_URL: Final = re.compile(
    r"^data:(?P<media_type>image/[A-Za-z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=]+)$"
)

_TOOL_NAME_FOLD: Final = "-"
"""The character that joins a namespace to a tool name in the one name this wire offers.

A hyphen because the runtime sanitizes both halves of a namespaced tool name to
`[A-Za-z0-9_]` before either reaches here, so a hyphen cannot occur inside a half while
this side's tool-name charset still accepts it. That is what makes the join point
unique, and a unique join point is what makes the split on the way back recover the
pair that was sent instead of guessing where it was.
"""

_TOOL_NAME_HALF: Final = re.compile(r"^[A-Za-z0-9_]+$")
"""What each half must be for the fold to be reversible. Deliberately narrower than the
tool-name charset this side accepts: the difference is exactly the fold character."""

_TOOL_NAME_LIMIT: Final = 128
"""The longest tool name this side accepts. A folded name over it is refused rather than
shortened, because every shortening maps two long pairs onto one offered name."""

_DEFAULT_TOOL_NAMESPACE: Final = "functions"
"""The runtime's name for "no namespace". A call under it is a plain tool call, so it
folds to the bare name -- the same name a call carrying no namespace field at all folds
to, which is what keeps one tool from being offered under two spellings."""

SEARCH_TOOL: Final = "tool_search"
"""The runtime's name for the tool it defers every other tool behind.

It is both a `type` and a name here, because the spec that offers it carries no `name`
field -- the runtime addresses it by its type. Deferral is not optional in codex-rs
0.149.0: the flag that would turn it off (`tool_search_always_defer_mcp_tools`) is at
stage `Removed`, which makes the config parser skip the key rather than honour it. So
this tool absent from a request is every MCP tool the tenant granted absent from it.
"""

_CARRIER: Final = "anthropic-thinking-v1"
"""The tag inside the opaque reasoning carrier this wire writes.

It rides *inside* the payload rather than beside it so that a carrier written by
something else -- another wire, or a plaintext reasoning item -- is recognisable as
foreign instead of being decoded into a block this side would then sign wrongly.
"""

_Message = tuple[str, list[dict[str, object]]]
"""A drafted message before rendering: its role, and its blocks in order."""


def _breakpoint() -> dict[str, object]:
    """A fresh cache_control annotation. Fresh, so no two blocks share one object."""
    return {"type": "ephemeral"}


def encode_thinking(block: Mapping[str, object]) -> str:
    """Pack a thinking block into the one opaque string the runtime round-trips.

    The runtime carries reasoning across Turns as an `encrypted_content` string it never
    looks inside, while this wire needs the thinking text and its cryptographic
    `signature` back verbatim or the upstream rejects the block. Because the runtime
    treats the string as opaque, this wire owns its format -- which is what lets a
    signature survive a round trip through a protocol that has no field for one.
    """
    packed = json.dumps({"v": _CARRIER, "block": dict(block)}, separators=(",", ":"))
    return base64.b64encode(packed.encode()).decode()


def decode_thinking(carrier: str) -> dict[str, object] | None:
    """Unpack what `encode_thinking` wrote, or None when something else wrote it.

    Every way the string can fail to be one of ours answers None rather than raising:
    a reasoning item whose carrier this wire did not write has no block to restore, and
    that is a drop rather than a fault. `validate=True` on the decode is what makes a
    string that merely looks like base64 answer None instead of decoding to noise.
    """
    try:
        parsed = json.loads(base64.b64decode(carrier.encode(), validate=True))
    except ValueError:
        # Covers json.JSONDecodeError and binascii.Error, both ValueError subclasses.
        return None
    if not isinstance(parsed, dict) or parsed.get("v") != _CARRIER:
        return None
    block = parsed.get("block")
    return block if isinstance(block, dict) else None


class ResponsesTurn(BaseModel):
    """The part of the runtime's request this wire reads.

    Extra keys are allowed rather than forbidden, and that is not laxness: the runtime
    adds request fields on its own schedule, and a field this translator does not read
    is classified by name in this wire's table rather than by whether Pydantic had heard
    of it. Forbidding extras would turn a runtime upgrade into a rejected request before
    the classification ever ran, which is a worse failure and a less informative one.

    `stream` is declared rather than left to the extras because its value is read and
    not merely classified. Every other declared field is here because the translation
    needs it.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    model: str
    stream: bool = True
    instructions: str = ""
    input: tuple[dict[str, object], ...] = ()
    tools: tuple[dict[str, object], ...] = ()
    tool_choice: str = "auto"
    parallel_tool_calls: bool = True
    reasoning: dict[str, object] | None = None
    text: dict[str, object] | None = None


def to_messages_request(
    turn: ResponsesTurn,
    *,
    deployment: str,
    max_tokens: int,
    session_id: SessionId,
) -> dict[str, object]:
    """Build one Turn's Messages request body.

    `deployment` rather than `turn.model`: on Foundry the model field takes a deployment
    name that need not equal the model id -- measured, a deployment named
    `gsds-claude-opus-4-6` answers with `model: claude-opus-4-6` -- and which deployment
    a model reaches is its Routing Entry's answer, not this module's.

    Cache breakpoints are placed here rather than carried across, because the runtime's
    whole cache-control surface is one opaque key and this side has only per-block
    breakpoints. Three are placed -- the last tool, the system block, the end of the
    conversation -- in the render order that defines the cache key, leaving one of the
    four unspent so an automatic breakpoint added later does not make the request
    refusable.

    Raises `Untranslatable` and builds nothing partial. A body missing a construct is a
    body that asks a different question than the one the tenant asked, so there is no
    half-translated request to send.
    """
    for extra in sorted(turn.model_extra or {}):
        carry(WIRE, f"request.{extra}")
    carry(WIRE, "request.stream")
    if turn.stream is not True:
        # The row above states the invariant this branch enforces: the runtime hardcodes
        # `stream` true, and everything downstream of here builds an SSE response. A
        # request asking for a whole body would get a stream anyway, so the construct
        # nobody classified is refused rather than answered in the wrong shape.
        carry(WIRE, "request.stream.not_true")
    body: dict[str, object] = {
        "model": deployment,
        "max_tokens": max_tokens,
        "messages": _messages(turn.input),
        "stream": True,
    }
    if system := _system(turn.instructions, _system_level_blocks(turn.input)):
        body["system"] = system
    offered = [*turn.tools, *_discovered_tool_specs(turn.input)]
    translated = _tools(offered)
    _record_tool_census(offered, translated, turn.input, session_id)
    if translated:
        body["tools"] = translated
        body["tool_choice"] = _tool_choice(turn.tool_choice, turn.parallel_tool_calls)
    if (thinking := _thinking(turn.reasoning)) is not None:
        body["thinking"] = thinking
    if (output_config := _output_config(turn.text)) is not None:
        body["output_config"] = output_config
    return body


_SYSTEM_LEVEL_ROLES: Final = ("developer", "system")
"""The roles whose content is system-level rather than a turn in the conversation.

Both, because the runtime has used each of them for the same purpose and this side
cannot tell which it will send next. `developer` is what it uses today for its
world-state sections; `system` is the older spelling of the same idea and costs nothing
to accept.
"""


def _system(
    instructions: str, system_level: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    """The system field: the turn's instructions, then any system-level messages.

    Two sources and one field, because this wire has one place for system-level content
    and the runtime sends it in two. `instructions` is the stable part -- an agent's
    definition, unchanged for the life of the Session -- and the messages are the
    volatile part, re-rendered by the runtime every Turn as its view of the world
    changes.

    That ordering is also what makes the cache breakpoint worth placing where it is. The
    breakpoint marks the end of the cacheable prefix, so it sits on the instructions
    block: everything up to and including the agent's definition is identical between
    Turns and is served from cache, and the block that changes every Turn sits after it
    where it cannot invalidate anything. Putting it on the last block instead would make
    every Turn a cache miss on the whole system field.

    A Turn with no instructions and no system-level messages returns an empty list, and
    the caller omits the field rather than sending an empty one: a `system` present and
    empty is a different request from one carrying no `system` at all.
    """
    blocks: list[dict[str, object]] = []
    if instructions:
        carry(WIRE, "request.instructions")
        blocks.append(
            {"type": "text", "text": instructions, "cache_control": _breakpoint()}
        )
    blocks.extend(system_level)
    return blocks


def _system_level_blocks(
    items: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Every system-level message's content, flattened into system blocks, in order.

    A second pass over the same input `_messages` walks, rather than a value threaded
    out of it. The two answers go into two different fields of the request and are built
    from the same items by different rules; one function returning both would be a
    function whose caller has to know which half is which.

    **This is the fix for a defect that was silent in exactly the way this module's
    classification table exists to prevent.** The runtime delivers its skills catalogue
    -- the list of skills the model is allowed to know it has -- as a developer-role
    message, and this side dropped every message whose role was not `user` or
    `assistant`. So a Session with a skill delivered into its pod, readable on disk,
    answered that it had no skills, and the only trace was a classification row whose
    stated reason for the drop was untrue. Nothing failed and nothing logged.

    No cache breakpoint on these blocks. They are the volatile half of the system field
    by definition -- the runtime re-renders them per Turn -- so a breakpoint here would
    be spent on a prefix that never repeats.
    """
    blocks: list[dict[str, object]] = []
    for item in items:
        if str(item.get("type", "")) != "message":
            continue
        if str(item.get("role", "")) not in _SYSTEM_LEVEL_ROLES:
            continue
        blocks.extend(_blocks(item.get("content")))
    return blocks


def fold_tool_name(namespace: str | None, name: str) -> str:
    """The one name this wire offers for a tool the runtime addresses as a pair.

    The runtime groups a tool under a namespace and sends the two halves as separate
    fields; this side takes a flat list of tools with one name each. They are joined
    here rather than at each call site because the request's tool list and the replayed
    calls in its history have to agree on the answer down to the character: a call whose
    name was folded differently from the tool it names is a name the model was never
    offered, which reads downstream as the model inventing a tool.

    A half the fold cannot reverse is refused instead of escaped. Escaping would work,
    but it would give one pair two spellings, and the runtime sanitizes both halves to
    `[A-Za-z0-9_]` before they arrive -- so a half that needs escaping means something
    upstream stopped sanitizing, which is worth a marker rather than a quiet repair.

    A tool with no namespace goes through here too, and is held to the same charset.
    Not for its own sake -- it needs no joining -- but because the split on the way back
    reads one name and cannot ask whether it was folded: a plain tool named with the
    fold character in it would come back split into a namespace and a name, aimed at a
    pair that does not exist. Refusing to offer it is the only way the split stays
    total. The runtime sanitizes every namespaced name it sends, so this bites only a
    name that reached the wire unsanitized.
    """
    unfoldable = classify(WIRE, "tool.name_not_flattenable")
    if not _TOOL_NAME_HALF.match(name):
        raise Untranslatable(WIRE, unfoldable, note=f"tool {name!r}")
    if namespace is None or namespace in ("", _DEFAULT_TOOL_NAMESPACE):
        return name
    if not _TOOL_NAME_HALF.match(namespace):
        raise Untranslatable(WIRE, unfoldable, note=f"namespace {namespace!r}")
    folded = f"{namespace}{_TOOL_NAME_FOLD}{name}"
    if len(folded) > _TOOL_NAME_LIMIT:
        raise Untranslatable(WIRE, classify(WIRE, "tool.name_too_long"), note=folded)
    return folded


def unfold_tool_name(offered: str) -> tuple[str | None, str]:
    """Recover the namespace and name that `fold_tool_name` joined into one.

    Total, and it never guesses. A name with no fold character was offered without a
    namespace and goes back the same way -- including one the model invented, which the
    runtime is the right place to reject as unknown. A name folded at exactly one point
    into two reversible halves splits there. Anything else is a name this wire did not
    offer *and* cannot read, and picking a namespace for it would aim the call at a tool
    the model was never shown, so it fails with a marker instead.
    """
    if _TOOL_NAME_FOLD not in offered:
        return None, offered
    namespace, _, name = offered.partition(_TOOL_NAME_FOLD)
    if _TOOL_NAME_HALF.match(namespace) and _TOOL_NAME_HALF.match(name):
        return namespace, name
    raise Untranslatable(
        WIRE, classify(WIRE, "block.tool_use.name_not_flattened"), note=offered
    )


def _tools(specs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The offered tools, dropped ones absent, the last of the survivors marked.

    An empty result is the signal the caller uses to leave both `tools` and
    `tool_choice` off the body: this side refuses a `tool_choice` beside no tools, so a
    turn whose every tool was dropped must not carry a choice about them.

    A namespace spec contributes one entry per member rather than one entry, so "the
    last of the survivors" is the last member of the last namespace. That is still the
    last tool as the wire will read it, which is what the breakpoint is for -- it caches
    the whole tool list, and there is nothing about a namespace boundary for it to sit
    on.

    A name is offered once. The upstream refuses a tool list carrying a duplicate name,
    and the caller feeds this both the runtime's offered specs and every spec the
    conversation's history discovered -- so one tool found on two Turns, or found once
    and also offered outright, arrives here twice by construction. First wins: the
    earlier spec is the one the model has already seen a call against.
    """
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for spec in specs:
        kind = str(spec.get("type", ""))
        if carry(WIRE, f"tool.{kind}") is Disposition.DROPPED:
            continue
        if kind == "namespace":
            candidates = _folded_members(spec)
        elif kind == SEARCH_TOOL:
            candidates = [_offered_tool(SEARCH_TOOL, spec)]
        else:
            folded = fold_tool_name(None, str(spec.get("name", "")))
            candidates = [_offered_tool(folded, spec)]
        for one in candidates:
            name = str(one["name"])
            if name in seen:
                continue
            seen.add(name)
            out.append(one)
    if out:
        out[-1] = {**out[-1], "cache_control": _breakpoint()}
    return out


def _discovered_tool_specs(
    items: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Every tool spec the conversation's history carries inline, in the order found.

    Two items carry a tool list inside themselves rather than in the request's tool
    array: `tool_search_output`, which is what the runtime's local search returns, and
    `additional_tools`, which the runtime uses to hand over a group mid-conversation.
    This wire has one place to put a tool -- the top-level array -- so a spec left
    inside its item is a tool the model has been told the shape of and cannot call. It
    answers by describing the call in prose, and the Turn completes having run nothing.

    Deferred discovery is how every MCP tool arrives, so this is not an edge: without
    it the granted catalogue reaches the model as text and never as a callable tool.

    Ordering matters only against the deduplication in `_tools`: these are appended
    after the runtime's own offered specs, so a tool offered outright keeps the spec
    the model has already been calling it under.
    """
    out: list[Mapping[str, object]] = []
    for item in items:
        if str(item.get("type", "")) not in ("tool_search_output", "additional_tools"):
            continue
        found = item.get("tools")
        if not isinstance(found, list):
            continue
        out.extend(one for one in found if isinstance(one, Mapping))
    return out


def _folded_members(spec: Mapping[str, object]) -> list[dict[str, object]]:
    """One namespace's members, each offered as a tool of its own.

    Each member is classified by its own `type` against the same rows a top-level tool
    of that type uses, so a member kind this wire drops at the top level is dropped
    here too and a kind nobody classified fails here too. One namespace whose every
    member is dropped contributes nothing, which is the same outcome as offering it with
    no members and cheaper to read.

    The namespace's own description is offered to any member that has none of its own.
    It is the only place the model learns which server a tool came from, and a member
    with no description at all is a tool the model has no reason to pick.
    """
    namespace = str(spec.get("name", ""))
    shared = str(spec.get("description", ""))
    members = spec.get("tools")
    out: list[dict[str, object]] = []
    for member in members if isinstance(members, list) else ():
        if not isinstance(member, Mapping):
            continue
        if carry(WIRE, f"tool.{member.get('type', '')}") is Disposition.DROPPED:
            continue
        out.append(
            _offered_tool(
                fold_tool_name(namespace, str(member.get("name", ""))),
                member,
                fallback_description=shared,
            )
        )
    return out


def _offered_tool(
    name: str, spec: Mapping[str, object], *, fallback_description: str = ""
) -> dict[str, object]:
    """One tool as this side reads it, under the name already folded for it."""
    tool: dict[str, object] = {
        "name": name,
        "description": str(spec.get("description", "")) or fallback_description,
        "input_schema": spec.get("parameters") or {"type": "object", "properties": {}},
    }
    if spec.get("strict") is True:
        tool["strict"] = True
    return tool


def _dropped_names(specs: Sequence[Mapping[str, object]]) -> list[str]:
    """The offered tools this wire does not carry, by name.

    By type where a spec has no name, which is not an edge case: `tool_search` and
    `web_search` are addressed by their type and carry no `name` field at all, so a
    census that only read `name` would report the two most interesting absences as
    empty strings.

    A namespace's members are not walked. A namespace is carried, so it never appears
    here, and a member dropped inside a carried namespace is a different fact from a
    whole tool kind this wire refuses -- one this line would flatten into the other.
    """
    dropped = []
    for spec in specs:
        kind = str(spec.get("type", ""))
        if classify(WIRE, f"tool.{kind}").disposition is not Disposition.DROPPED:
            continue
        dropped.append(str(spec.get("name", "")) or kind)
    return sorted(dropped)


def _record_tool_census(
    specs: Sequence[Mapping[str, object]],
    translated: Sequence[Mapping[str, object]],
    items: Sequence[Mapping[str, object]],
    session_id: SessionId,
) -> None:
    """Say how many tools the runtime offered for this Turn and how many survived.

    A request that reaches the model carrying no tools does not fail. The model is told
    about its tools in the instructions either way, so it answers by writing tool calls
    as prose, and the Turn completes with a fluent message that ran nothing and quoted
    numbers it invented. That is indistinguishable, from the outside, from a Turn that
    genuinely had nothing to call -- which is why it went unnoticed across sixteen live
    cases until somebody read a request body.

    So a Turn whose tools survive is counted, and every shape of a request that reaches
    the model carrying none is a warning. The first shape is every offered tool dropped,
    which is this building a body asking a different question than the runtime asked.
    `specs` counts top-level entries, so a namespace counts once here and contributes
    many to `translated`; the two numbers are a census of each end, not a subtraction.

    The second shape is no top-level tools at all, beside an `additional_tools` item
    carrying the list inline. The wire table drops that item because this wire sends
    tools in the top-level array, which is true of the ordinary Responses shape and
    false of the one a `use_responses_lite` model asks for. When both are true at once
    the request is honestly untranslatable, and saying so is the difference between a
    minute and an afternoon.

    The third shape offers nothing anywhere, and it is the one that used to return in
    silence. It reads as a Turn that simply wanted no tools, which is presumably why it
    was left alone, and it is equally what a runtime that never listed its tool
    catalogue sends -- the body left behind by a Session granted a tool the model then
    told the tenant does not exist looks exactly like this, and nothing else here
    records that it happened.

    All three are warnings rather than informational lines, and the level is the guard
    rather than a label on it. This service runs under `uvicorn`, whose logging config
    names only its own three loggers and leaves the root at WARNING holding no handler,
    so a `managed_agent` record below WARNING reaches `logging.lastResort` and is
    dropped -- the healthy count below has never once appeared in a deployed log. What
    the level costs is noise on a Turn that legitimately wanted no tools, and this
    platform has none of those: the runtime's built-in tools are the agent's hands and
    are granted by default (ADR-018), so a Turn offering nothing is a catalogue that
    never arrived rather than one nobody asked for.

    Dropped tools are named, and the surviving ones are only counted. The asymmetry is
    the point: a carried tool is named by the census on the other side of this hop, and
    a dropped one is named nowhere else at all -- it leaves no trace in the body, so a
    reader holding "14 offered, 13 translated" has to know the wire table by heart to
    work out which one went and whether that was meant. `apply_patch` is dropped on
    every Turn this platform serves, deliberately, and the runtime's own system prompt
    teaches the model to use it; a reader deserves to see that stated rather than
    reconstruct it.

    This reverses an earlier line here that logged kinds and withheld names, on the
    grounds that a name can carry a tenant's namespace. It can, and the census on the
    model side already emits exactly those names for the same reason an operator needs
    them -- so the withholding bought nothing and cost the one fact this line exists to
    carry. A tool name is a name the tenant chose, not tenant content; the schema beside
    it is the part that is never logged.

    Every line carries the Session, which is not decoration: this gateway serves every
    Session in the namespace at once, so a census nobody can attribute is a line that
    cannot be joined to the one the model side writes for the same request, and cannot
    be told apart from another tenant's. That was found by a test reading these lines
    back out of the deployed log and matching nothing.
    """
    inline = sum(1 for item in items if item.get("type") == "additional_tools")
    if not specs:
        if inline:
            _LOG.warning(
                "tool census: session=%s no top-level tools, but %d additional_tools "
                "item(s) carry a list inline -- this wire reads the top-level array, "
                "so the model is reaching the provider with nothing bound and can only "
                "describe calls it cannot make",
                session_id,
                inline,
            )
        else:
            _LOG.warning(
                "tool census: session=%s the runtime offered no tools at all, so this "
                "request carries none and the model can only describe calls it cannot "
                "make; built-in tools are granted by default here, so this is a "
                "catalogue that never arrived rather than a Turn that wanted none",
                session_id,
            )
        return
    kinds = sorted({str(spec.get("type", "")) for spec in specs})
    if translated:
        _LOG.info(
            "tool census: session=%s %d specs offered, %d translated, kinds=%s, "
            "dropped=%s",
            session_id,
            len(specs),
            len(translated),
            kinds,
            _dropped_names(specs),
        )
        return
    _LOG.warning(
        "tool census: session=%s all %d offered specs were dropped, so this request "
        "carries no tools and the model can only describe calls it cannot make; "
        "kinds=%s",
        session_id,
        len(specs),
        kinds,
    )


def _tool_choice(choice: str, parallel: bool) -> dict[str, object]:
    """The runtime sends a bare string where this side takes an object.

    The parallel-call switch is a modifier on that object rather than a field of its
    own, which is why the two are read together here instead of at two call sites.
    """
    carry(WIRE, "request.tool_choice")
    carry(WIRE, "request.parallel_tool_calls")
    out: dict[str, object] = {"type": choice}
    if not parallel:
        out["disable_parallel_tool_use"] = True
    return out


def _thinking(reasoning: Mapping[str, object] | None) -> dict[str, object] | None:
    """Map the reasoning request onto this side's thinking union.

    Only the summary preference crosses. `display: "omitted"` is not thinking switched
    off -- the block still opens, takes its signature and closes -- which is what keeps
    a Turn's reasoning round-trippable even when the tenant asked not to be shown it.
    """
    if reasoning is None:
        return None
    for field in sorted(reasoning):
        carry(WIRE, f"request.reasoning.{field}")
    summary = reasoning.get("summary")
    display = "summarized" if isinstance(summary, str) and summary else "omitted"
    return {"type": "adaptive", "display": display}


def _output_config(text: Mapping[str, object] | None) -> dict[str, object] | None:
    """The structured-output ask, carrying the schema and nothing beside it.

    `strict` and `name` sit next to the schema on the Responses side and have no
    counterpart here, so they are left off rather than folded into a field that would
    then mean something else.
    """
    if text is None:
        return None
    for field in sorted(text):
        carry(WIRE, f"request.text.{field}")
    fmt = text.get("format")
    if not isinstance(fmt, Mapping):
        return None
    return {"format": {"type": "json_schema", "schema": fmt.get("schema") or {}}}


def _messages(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The conversation, item by item, as messages this side accepts.

    `dropped_calls` is what keeps the two halves of a tool call decided together. A
    `tool_result` whose `tool_use` is absent from the history is refused upstream, so an
    output whose call was dropped is dropped with it rather than each half being judged
    alone -- which is a rule about the pair and cannot be expressed in either row.
    """
    dropped_calls: set[str] = set()
    drafted: list[_Message] = []
    for item in items:
        kind = str(item.get("type", ""))
        if carry(WIRE, f"item.{kind}") is Disposition.DROPPED:
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                dropped_calls.add(call_id)
            continue
        if kind == "message":
            role = str(item.get("role", ""))
            if role in _SYSTEM_LEVEL_ROLES:
                # Carried by `_system_level_blocks` into the request's `system` field,
                # so it is skipped here rather than dropped. Two passes over one list is
                # the price of the two fields being built by different rules.
                carry(WIRE, "item.message.role_system_level")
                continue
            if role not in ("user", "assistant"):
                carry(WIRE, "item.message.role_not_conversational")
                continue
            if blocks := _blocks(item.get("content")):
                drafted.append((role, blocks))
        elif kind == "function_call":
            drafted.append(("assistant", [_tool_use(item)]))
        elif kind == "function_call_output":
            if str(item.get("call_id", "")) not in dropped_calls:
                drafted.append(("user", [_tool_result(item)]))
        elif kind == "tool_search_call":
            drafted.append(("assistant", [_tool_search_use(item)]))
        elif kind == "tool_search_output":
            drafted.append(("user", [_tool_search_result(item)]))
        elif kind == "reasoning":
            if (block := _restored_thinking(item)) is not None:
                drafted.append(("assistant", [block]))
    return _rendered(drafted)


def _rendered(drafted: Sequence[_Message]) -> list[dict[str, object]]:
    """Serialize the drafted messages, merging by role and marking the prefix end.

    Consecutive messages of one role are merged into a single message, because this side
    reads a run of them as one turn: every `tool_result` answering one assistant turn
    must sit in a single user message, and every parallel `tool_use` in a single
    assistant message. The runtime emits each call and each result as its own top-level
    item, so a turn with two parallel calls drafts four messages here and must send two.

    The breakpoint then goes on the last block of the last message, so every Turn
    re-reads the whole conversation before it out of cache and pays a write only for
    what it added. Merging happens first: the mark belongs on the last block as the wire
    will see it, not on the last block of a message that is about to be joined to
    another.
    """
    merged: list[_Message] = []
    for role, blocks in drafted:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, [*merged[-1][1], *blocks])
        else:
            merged.append((role, list(blocks)))
    if merged and merged[-1][1]:
        role, blocks = merged[-1]
        merged[-1] = (
            role,
            [*blocks[:-1], {**blocks[-1], "cache_control": _breakpoint()}],
        )
    return [{"role": role, "content": blocks} for role, blocks in merged]


def _blocks(content: object) -> list[dict[str, object]]:
    """One item's content blocks, translated. A block with no counterpart is absent."""
    out: list[dict[str, object]] = []
    for block in content if isinstance(content, list) else ():
        if not isinstance(block, Mapping):
            continue
        kind = str(block.get("type", ""))
        if kind == "input_image":
            if (image := _image(str(block.get("image_url", "")))) is not None:
                out.append(image)
            continue
        if carry(WIRE, f"content.{kind}") is Disposition.DROPPED:
            continue
        # The only carried non-image content types are the two text ones, which is why
        # one text block serves both of them.
        out.append({"type": "text", "text": str(block.get("text", ""))})
    return out


def _image(image_url: str) -> dict[str, object] | None:
    """Turn the runtime's image string into an image block, or drop it.

    The runtime carries an image as one `image_url` string, prepared locally into a data
    URL; this side takes a structured `source`. A string that is not a data URL is
    therefore not a shape the runtime produces, and rather than assert a source form no
    transcript grades, it is dropped -- a lost image cannot make an unfinished Turn look
    finished, which is the whole test.
    """
    match = _DATA_URL.match(image_url)
    if match is None:
        carry(WIRE, "content.input_image.remote_url")
        return None
    carry(WIRE, "content.input_image.data_url")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": match["media_type"],
            "data": match["data"],
        },
    }


def _tool_use(item: Mapping[str, object]) -> dict[str, object]:
    """A `function_call` item as a `tool_use` block, its arguments parsed to an object.

    `function_call` carries its arguments as a JSON-encoded string and `tool_use.input`
    is a real object, so the string is parsed here rather than forwarded. A string that
    will not parse into an object cannot be carried at all: the upstream refuses the
    request, and a refused request is a Turn that never ran. It fails with a marker
    naming the call instead of being sent to be rejected.

    The name goes through the same fold the tool list does. A namespaced call replayed
    under its bare name would name no tool in the request it travels in, and a model
    reading its own history under a name nothing offers is being shown a tool call it
    could not have made.
    """
    raw = item.get("arguments")
    call_id = str(item.get("call_id", ""))
    row = classify(WIRE, "item.function_call.arguments_not_json")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw else {}
    except json.JSONDecodeError as exc:
        raise Untranslatable(WIRE, row, note=call_id) from exc
    if not isinstance(parsed, dict):
        raise Untranslatable(WIRE, row, note=call_id)
    namespace = item.get("namespace")
    return {
        "type": "tool_use",
        "id": call_id,
        "name": fold_tool_name(
            namespace if isinstance(namespace, str) else None,
            str(item.get("name", "")),
        ),
        "input": parsed,
    }


def _tool_result(item: Mapping[str, object]) -> dict[str, object]:
    """A `function_call_output` item as a `tool_result` block.

    The runtime's output payload is either a bare string or a list of content items, and
    it reports failure as `success: false` where this side reads `is_error: true`. The
    absent-means-success reading is deliberate: only an explicit false marks an error,
    so an output that omits the field is not reported to the model as having failed.
    """
    output = item.get("output")
    content = (
        [{"type": "text", "text": output}]
        if isinstance(output, str)
        else _blocks(output)
    )
    result: dict[str, object] = {
        "type": "tool_result",
        "tool_use_id": str(item.get("call_id", "")),
        "content": content,
    }
    if item.get("success") is False:
        result["is_error"] = True
    return result


def _tool_search_use(item: Mapping[str, object]) -> dict[str, object]:
    """A `tool_search_call` item as a `tool_use` block.

    It cannot share `_tool_use`. That builder parses `arguments` from a JSON-encoded
    string, which is the shape `function_call` uses; this item carries `arguments` as a
    real object, and a string parse of an object yields nothing -- the call would replay
    with an empty input, so the history would show the model searching for nothing and
    then acting on what it found.

    The name is the type, not a field: the item carries no `name`, and the tool it
    names is offered under that same constant, so the two agree by construction.
    """
    arguments = item.get("arguments")
    return {
        "type": "tool_use",
        "id": str(item.get("call_id", "")),
        "name": SEARCH_TOOL,
        "input": arguments if isinstance(arguments, Mapping) else {},
    }


def _tool_search_result(item: Mapping[str, object]) -> dict[str, object]:
    """A `tool_search_output` item as a `tool_result` block, naming what it found.

    The specs themselves are not repeated here -- `_discovered_tool_specs` has already
    put them in the request's tool array, which is the only place this wire lets a model
    call one from. What the result carries is the names, so the assistant turn that
    follows reads as an answer to the search rather than an empty result the model has
    to guess the meaning of.

    An empty search says so in words. A `tool_result` with no content is refused
    upstream, and "no tool matched" is also the honest answer to give a model that is
    about to decide whether to search again.
    """
    found = item.get("tools")
    names = sorted(
        str(one["name"])
        for one in (found if isinstance(found, list) else ())
        if isinstance(one, Mapping) and isinstance(one.get("name"), str)
    )
    joined = ", ".join(names)
    text = f"Found {len(names)} tools: {joined}" if names else "No tool found."
    return {
        "type": "tool_result",
        "tool_use_id": str(item.get("call_id", "")),
        "content": [{"type": "text", "text": text}],
    }


def _restored_thinking(item: Mapping[str, object]) -> dict[str, object] | None:
    """The thinking block a reasoning item carries, or None if it carries none of ours.

    Plaintext reasoning and a carrier from another wire both land here, and both are
    drops: there is no block to restore, and inventing one would produce a signature the
    upstream refuses on the next Turn.
    """
    carrier = item.get("encrypted_content")
    if not isinstance(carrier, str) or not carrier:
        carry(WIRE, "item.reasoning.no_carried_block")
        return None
    block = decode_thinking(carrier)
    if block is None:
        carry(WIRE, "item.reasoning.no_carried_block")
        return None
    return block
