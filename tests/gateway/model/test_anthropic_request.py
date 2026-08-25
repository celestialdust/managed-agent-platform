"""Responses request to Messages request, graded on the body that would go out.

Tier 1, no infrastructure and no HTTP. What is asserted is the dict the translator
builds, because that dict is the whole of what the upstream will see -- the handler adds
headers and a URL and changes not a byte of it.

Two families of assertion carry the weight. The first is that nothing is invented: the
top-level key set is checked against the fields the Messages API defines, so a key this
translator makes up is a failure here rather than a 400 in production. The second is the
pairing rule -- a `tool_result` whose `tool_use` is absent is refused upstream, so an
output whose call was dropped has to be dropped with it, and parallel calls and their
results have to land in one message each.
"""

from __future__ import annotations

import re
from typing import Final
from uuid import UUID

import pytest

from managed_agent.core.ids import SessionId
from managed_agent.core.session.markers import DiscardCause
from managed_agent.gateway.model.anthropic_request import (
    ResponsesTurn,
    decode_thinking,
    encode_thinking,
    fold_tool_name,
    to_messages_request,
    unfold_tool_name,
)
from managed_agent.gateway.model.classify import Untranslatable

UPSTREAM_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
"""The pattern the Messages API applies to a tool's `name`, written out here.

Deliberately a second, independent copy of the rule rather than an import of the
translator's own constants: a folded name checked against the regex that produced it
proves nothing, while one checked against the upstream's published pattern proves the
name would be accepted. A name outside this is a 400 on the request, which is a Turn
that never ran.
"""

MESSAGES_TOP_LEVEL = frozenset(
    {
        "model",
        "max_tokens",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "thinking",
        "output_config",
        "stream",
        "metadata",
        "stop_sequences",
        "temperature",
        "top_k",
        "top_p",
        "service_tier",
    }
)
"""Every top-level field the Messages request body defines. Nothing else may appear."""


def _turn(**fields: object) -> ResponsesTurn:
    return ResponsesTurn.model_validate({"model": "claude-opus-5", **fields})


_SESSION: Final = SessionId(UUID("22222222-2222-4222-8222-222222222222"))


def _built(turn: ResponsesTurn, *, max_tokens: int = 4096) -> dict[str, object]:
    return to_messages_request(
        turn,
        deployment="gsds-claude-opus-4-6",
        max_tokens=max_tokens,
        session_id=_SESSION,
    )


def _annotations(body: object) -> int:
    """Every cache_control annotation anywhere in the built body."""
    if isinstance(body, dict):
        here = 1 if "cache_control" in body else 0
        return here + sum(_annotations(value) for value in body.values())
    if isinstance(body, list):
        return sum(_annotations(item) for item in body)
    return 0


def test_the_envelope_names_the_deployment_and_the_given_cap() -> None:
    body = _built(
        _turn(
            instructions="be brief",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            ],
        ),
        max_tokens=1234,
    )

    assert body["model"] == "gsds-claude-opus-4-6"
    assert body["max_tokens"] == 1234
    assert body["stream"] is True
    assert body["system"] == [
        {"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


def test_no_top_level_field_the_messages_api_does_not_define() -> None:
    body = _built(
        _turn(
            instructions="s",
            input=[{"type": "message", "role": "user", "content": []}],
            tools=[{"type": "function", "name": "t", "parameters": {}}],
            reasoning={"summary": "auto"},
            text={"format": {"schema": {"type": "object"}}},
            store=False,
            stream=True,
            include=["reasoning.encrypted_content"],
            prompt_cache_key="k",
            service_tier="priority",
        )
    )

    assert set(body) <= MESSAGES_TOP_LEVEL


def test_a_function_tools_parameters_become_its_input_schema() -> None:
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    body = _built(
        _turn(
            tools=[
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "read one",
                    "parameters": schema,
                }
            ]
        )
    )

    assert body["tools"] == [
        {
            "name": "read_file",
            "description": "read one",
            "input_schema": schema,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_only_the_last_tool_carries_a_breakpoint() -> None:
    body = _built(
        _turn(
            tools=[
                {"type": "function", "name": "a", "parameters": {}},
                {"type": "function", "name": "b", "parameters": {}},
            ]
        )
    )
    tools = body["tools"]

    assert isinstance(tools, list)
    assert "cache_control" not in tools[0]
    assert "cache_control" in tools[1]


def test_a_full_turn_spends_three_of_the_four_breakpoints() -> None:
    """Three placed -- last tool, system block, end of the conversation -- leaving one
    unspent so an automatic breakpoint added later cannot make the request refusable."""
    body = _built(
        _turn(
            instructions="be brief",
            tools=[{"type": "function", "name": "a", "parameters": {}}],
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        )
    )

    assert _annotations(body) == 3


def test_a_dropped_tool_is_absent_and_nothing_is_raised() -> None:
    body = _built(
        _turn(
            tools=[
                {"type": "custom", "name": "grammar", "parameters": {}},
                {"type": "function", "name": "kept", "parameters": {}},
            ]
        )
    )
    tools = body["tools"]

    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools] == ["kept"]


def test_the_bare_tool_choice_string_becomes_the_object_form() -> None:
    body = _built(_turn(tools=[{"type": "function", "name": "a", "parameters": {}}]))

    assert body["tool_choice"] == {"type": "auto"}


def test_serial_tool_calls_are_asked_for_as_a_modifier_on_tool_choice() -> None:
    body = _built(
        _turn(
            tools=[{"type": "function", "name": "a", "parameters": {}}],
            parallel_tool_calls=False,
        )
    )

    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_no_tool_choice_travels_when_no_tool_does() -> None:
    """The field is meaningless without tools and this side refuses it beside an empty
    list, so a turn that offers nothing must not carry a choice about it."""
    body = _built(_turn(tools=[{"type": "custom", "name": "dropped"}]))

    assert "tools" not in body
    assert "tool_choice" not in body


def test_a_namespaced_tool_is_offered_under_a_name_carrying_its_namespace() -> None:
    """Every MCP tool arrives inside a namespace spec, and this side has no grouping to
    put one in -- so a namespace that is not folded into the flat list is a granted tool
    the model is never offered."""
    body = _built(
        _turn(
            tools=[
                {
                    "type": "namespace",
                    "name": "mcp__map_tool_gateway",
                    "description": "Tools for working with deepwiki.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "ask_deepwiki",
                            "description": "Ask a question about a repository.",
                            "strict": False,
                            "parameters": {
                                "type": "object",
                                "properties": {"repoName": {"type": "string"}},
                            },
                        }
                    ],
                }
            ]
        )
    )
    tools = body["tools"]

    assert isinstance(tools, list)
    assert tools == [
        {
            "name": "mcp__map_tool_gateway-ask_deepwiki",
            "description": "Ask a question about a repository.",
            "input_schema": {
                "type": "object",
                "properties": {"repoName": {"type": "string"}},
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_every_member_of_a_namespace_is_offered_and_a_dropped_kind_is_not() -> None:
    """Members are classified by their own `type` against the rows a top-level tool of
    that type uses, so the two levels cannot disagree about a kind."""
    body = _built(
        _turn(
            tools=[
                {
                    "type": "namespace",
                    "name": "ns",
                    "description": "",
                    "tools": [
                        {"type": "function", "name": "one", "parameters": {}},
                        {"type": "custom", "name": "freeform", "parameters": {}},
                        {"type": "function", "name": "two", "parameters": {}},
                    ],
                }
            ]
        )
    )
    tools = body["tools"]

    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools] == ["ns-one", "ns-two"]


def test_a_member_with_no_description_of_its_own_borrows_the_namespaces() -> None:
    """It is the only place the model learns which server a tool came from, and a tool
    with no description at all is one the model has no reason to pick."""
    body = _built(
        _turn(
            tools=[
                {
                    "type": "namespace",
                    "name": "ns",
                    "description": "Tools in the ns namespace.",
                    "tools": [{"type": "function", "name": "bare", "parameters": {}}],
                }
            ]
        )
    )
    tools = body["tools"]

    assert isinstance(tools, list)
    assert tools[0]["description"] == "Tools in the ns namespace."


def test_the_breakpoint_lands_on_the_last_member_not_the_last_spec() -> None:
    """One spec becomes several tools, so "the last tool" moved. The breakpoint caches
    the whole list and has nothing about a namespace boundary to sit on."""
    body = _built(
        _turn(
            tools=[
                {"type": "function", "name": "flat", "parameters": {}},
                {
                    "type": "namespace",
                    "name": "ns",
                    "description": "",
                    "tools": [
                        {"type": "function", "name": "one", "parameters": {}},
                        {"type": "function", "name": "two", "parameters": {}},
                    ],
                },
            ]
        )
    )
    tools = body["tools"]

    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools] == ["flat", "ns-one", "ns-two"]
    assert [("cache_control" in tool) for tool in tools] == [False, False, True]


def test_the_shape_the_runtime_actually_sends_is_one_namespace_per_mcp_tool() -> None:
    """Measured against the runtime's own construction, which wraps each MCP tool in a
    namespace of its own rather than grouping a server's tools into one -- so a server
    with three tools arrives as three namespaces holding one member each, and the
    breakpoint has to survive that too."""
    body = _built(
        _turn(
            tools=[
                {
                    "type": "namespace",
                    "name": "mcp__map_tool_gateway",
                    "description": "Tools for working with deepwiki.",
                    "tools": [{"type": "function", "name": name, "parameters": {}}],
                }
                for name in (
                    "ask_deepwiki",
                    "read_wiki_contents",
                    "read_wiki_structure",
                )
            ]
        )
    )
    tools = body["tools"]

    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools] == [
        "mcp__map_tool_gateway-ask_deepwiki",
        "mcp__map_tool_gateway-read_wiki_contents",
        "mcp__map_tool_gateway-read_wiki_structure",
    ]
    assert [("cache_control" in tool) for tool in tools] == [False, False, True]


def test_the_runtimes_own_grouping_of_plain_tools_offers_them_under_bare_names() -> (
    None
):
    """One of the runtime's two tool-list shapes gathers every ordinary tool into a
    namespace it names `functions`, which is its spelling for "no namespace" -- folding
    that in would offer every tool under a second name nothing resolves."""
    body = _built(
        _turn(
            tools=[
                {
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [
                        {"type": "function", "name": "exec_command", "parameters": {}},
                        {"type": "function", "name": "apply_patch", "parameters": {}},
                    ],
                }
            ]
        )
    )
    tools = body["tools"]

    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools] == ["exec_command", "apply_patch"]


def test_a_namespace_whose_every_member_is_dropped_offers_nothing() -> None:
    body = _built(
        _turn(
            tools=[
                {
                    "type": "namespace",
                    "name": "ns",
                    "description": "",
                    "tools": [{"type": "custom", "name": "freeform"}],
                }
            ]
        )
    )

    assert "tools" not in body
    assert "tool_choice" not in body


def test_a_replayed_namespaced_call_is_named_exactly_as_the_tool_it_calls() -> None:
    """The pair is what matters, not either half. A call folded differently from the
    tool list it travels with names no offered tool, and a model reading its own history
    under a name nothing offers is being shown a call it could not have made."""
    body = _built(
        _turn(
            tools=[
                {
                    "type": "namespace",
                    "name": "mcp__map_tool_gateway",
                    "description": "",
                    "tools": [
                        {"type": "function", "name": "ask_deepwiki", "parameters": {}}
                    ],
                }
            ],
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "ask_deepwiki",
                    "namespace": "mcp__map_tool_gateway",
                    "arguments": '{"repoName":"a/b"}',
                }
            ],
        )
    )
    tools = body["tools"]
    messages = body["messages"]

    assert isinstance(tools, list)
    assert isinstance(messages, list)
    offered = {str(tool["name"]) for tool in tools}
    called = str(messages[0]["content"][0]["name"])
    assert called in offered, f"{called} names no offered tool"


def test_a_plain_call_folds_the_same_whether_it_names_the_default_namespace() -> None:
    """The runtime spells "no namespace" three ways -- absent, empty, and `functions` --
    and one tool offered under two names is a tool the model can call wrongly."""
    for namespace in (None, "", "functions"):
        item: dict[str, object] = {
            "type": "function_call",
            "call_id": "c",
            "name": "exec_command",
            "arguments": "{}",
        }
        if namespace is not None:
            item["namespace"] = namespace
        body = _built(_turn(input=[item]))
        messages = body["messages"]

        assert isinstance(messages, list)
        assert messages[0]["content"][0]["name"] == "exec_command", namespace


@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        ("mcp__map_tool_gateway", "ask_deepwiki"),
        ("a", "b"),
        ("a_b", "c"),
        ("a", "b_c"),
        ("_", "_"),
        ("Mixed_CASE_9", "tool_9"),
        ("0", "0"),
        ("n" * 60, "t" * 67),
    ],
)
def test_folding_a_pair_and_reading_it_back_returns_that_pair(
    namespace: str, name: str
) -> None:
    assert unfold_tool_name(fold_tool_name(namespace, name)) == (namespace, name)


@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        ("mcp__map_tool_gateway", "ask_deepwiki"),
        ("a", "b"),
        ("_", "_"),
        ("0", "0"),
        ("Mixed_CASE_9", "tool_9"),
        ("n" * 60, "t" * 67),
    ],
)
def test_a_folded_name_is_one_the_upstream_would_accept(
    namespace: str, name: str
) -> None:
    """Graded against the Messages API's own name pattern, not against the translator's
    constants. A folded name the upstream rejects is a 400 on the whole request, so
    every tool in the list is lost rather than the one that could not be spelled."""
    folded = fold_tool_name(namespace, name)

    assert UPSTREAM_TOOL_NAME.match(folded), folded


def test_no_two_pairs_share_one_offered_name() -> None:
    """`a_b`/`c` and `a`/`b_c` concatenate to the same string without a separator, which
    is the collision the fold character exists to prevent."""
    pairs = [
        ("a_b", "c"),
        ("a", "b_c"),
        ("a_b_c", "d"),
        ("mcp__x", "y"),
        ("mcp", "__x_y"),
    ]
    folded = [fold_tool_name(namespace, name) for namespace, name in pairs]

    assert len(set(folded)) == len(pairs)
    assert [unfold_tool_name(one) for one in folded] == pairs


@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        ("has-fold", "t"),
        ("ns", "has-fold"),
        ("ns", ""),
        ("ns", "has.dot"),
        ("has.dot", "t"),
        ("ns", "has space"),
    ],
)
def test_a_pair_the_fold_cannot_reverse_is_refused_not_escaped(
    namespace: str, name: str
) -> None:
    """Escaping would carry the pair but give it two spellings. The runtime sanitizes
    both halves before they arrive, so one needing an escape means something upstream
    stopped sanitizing -- worth a marker rather than a quiet repair."""
    with pytest.raises(Untranslatable) as raised:
        fold_tool_name(namespace, name)

    assert raised.value.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE
    assert "tool.name_not_flattenable" in raised.value.detail


def test_a_plain_tool_named_with_the_fold_character_is_refused_too() -> None:
    """It needs no joining, but the split on the way back reads one name and cannot ask
    whether it was folded -- so offering `my-tool` would bring back a call aimed at a
    namespace `my` that does not exist. Refusing is what keeps the split total."""
    with pytest.raises(Untranslatable) as raised:
        _built(_turn(tools=[{"type": "function", "name": "my-tool", "parameters": {}}]))

    assert raised.value.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE
    assert "tool.name_not_flattenable" in raised.value.detail


def test_a_folded_name_past_the_ceiling_is_refused_rather_than_shortened() -> None:
    """Two long pairs shortened to fit become one offered name, and a call landing on
    whichever tool won that race is worse than a call that never ran."""
    assert len(fold_tool_name("n" * 60, "t" * 67)) == 128

    with pytest.raises(Untranslatable) as raised:
        fold_tool_name("n" * 60, "t" * 68)

    assert raised.value.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE
    assert "tool.name_too_long" in raised.value.detail


def test_a_name_with_no_fold_reads_back_as_a_call_on_no_namespace() -> None:
    assert unfold_tool_name("exec_command") == (None, "exec_command")
    assert unfold_tool_name("anything the model invented") == (
        None,
        "anything the model invented",
    )


@pytest.mark.parametrize("offered", ["a-b-c", "-b", "a-", "-", "a-b.c"])
def test_a_name_the_fold_cannot_reverse_fails_rather_than_being_guessed_at(
    offered: str,
) -> None:
    """Picking one of the ways it could split would aim the call at a tool the model was
    never shown, which runs the wrong tool instead of running none."""
    with pytest.raises(Untranslatable) as raised:
        unfold_tool_name(offered)

    assert raised.value.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE
    assert "block.tool_use.name_not_flattened" in raised.value.detail


def test_asking_for_a_reasoning_summary_becomes_the_thinking_display() -> None:
    assert _built(_turn(reasoning={"summary": "auto"}))["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert _built(_turn(reasoning={"effort": "high"}))["thinking"] == {
        "type": "adaptive",
        "display": "omitted",
    }


def test_an_output_schema_becomes_the_output_config_format() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    body = _built(
        _turn(text={"format": {"name": "a", "strict": True, "schema": schema}})
    )

    assert body["output_config"] == {
        "format": {"type": "json_schema", "schema": schema}
    }


def test_a_function_call_becomes_a_tool_use_whose_input_is_an_object() -> None:
    body = _built(
        _turn(
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"/tmp/x"}',
                }
            ]
        )
    )

    assert body["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "read_file",
                    "input": {"path": "/tmp/x"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]


def test_arguments_that_will_not_parse_fail_rather_than_being_sent_to_be_refused() -> (
    None
):
    with pytest.raises(Untranslatable) as caught:
        _built(
            _turn(
                input=[
                    {
                        "type": "function_call",
                        "call_id": "call_bad",
                        "name": "f",
                        "arguments": "{not json",
                    }
                ]
            )
        )

    assert caught.value.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE
    assert "call_bad" in caught.value.detail


def test_arguments_that_parse_to_something_other_than_an_object_also_fail() -> None:
    with pytest.raises(Untranslatable) as caught:
        _built(
            _turn(
                input=[
                    {
                        "type": "function_call",
                        "call_id": "call_list",
                        "name": "f",
                        "arguments": "[1, 2]",
                    }
                ]
            )
        )

    assert caught.value.cause is DiscardCause.UPSTREAM_UNTRANSLATABLE


def test_a_function_call_output_becomes_a_user_tool_result() -> None:
    body = _built(
        _turn(
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "the file said hello",
                },
            ]
        )
    )
    messages = body["messages"]

    assert isinstance(messages, list)
    assert messages[1]["role"] == "user"
    result = messages[1]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "call_1"
    assert result["content"] == [{"type": "text", "text": "the file said hello"}]
    assert "is_error" not in result


def test_a_failed_output_is_marked_as_an_error() -> None:
    body = _built(
        _turn(
            input=[
                {
                    "type": "function_call",
                    "call_id": "c",
                    "name": "f",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "c",
                    "output": "boom",
                    "success": False,
                },
            ]
        )
    )
    messages = body["messages"]

    assert isinstance(messages, list)
    assert messages[1]["content"][0]["is_error"] is True


def test_parallel_calls_and_their_results_each_land_in_one_message() -> None:
    """This side requires every tool_result for one assistant turn to sit in a single
    user message, and every parallel tool_use in a single assistant message, so two
    drafted messages of the same role are one message on the wire."""
    body = _built(
        _turn(
            input=[
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "f",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "b",
                    "name": "g",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "a", "output": "one"},
                {"type": "function_call_output", "call_id": "b", "output": "two"},
            ]
        )
    )
    messages = body["messages"]

    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["assistant", "user"]
    assert [block["id"] for block in messages[0]["content"]] == ["a", "b"]
    assert [block["tool_use_id"] for block in messages[1]["content"]] == ["a", "b"]


def test_an_output_whose_call_was_dropped_is_dropped_with_it() -> None:
    """A tool_result with no tool_use before it is refused upstream, so the two halves
    are decided together rather than each on its own."""
    body = _built(
        _turn(
            input=[
                {"type": "local_shell_call", "call_id": "shell_1"},
                {"type": "function_call_output", "call_id": "shell_1", "output": "out"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "and now"}],
                },
            ]
        )
    )
    messages = body["messages"]

    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["content"][0]["text"] == "and now"


def test_a_data_url_image_becomes_a_base64_source() -> None:
    body = _built(
        _turn(
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,iVBORw0KGgo=",
                        }
                    ],
                }
            ]
        )
    )
    block = body["messages"][0]["content"][0]  # type: ignore[index]

    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"] == "iVBORw0KGgo="


def test_a_remote_url_image_and_an_audio_block_are_absent_with_nothing_raised() -> None:
    body = _built(
        _turn(
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://example/x.png"},
                        {"type": "input_audio", "data": "..."},
                        {"type": "input_text", "text": "look"},
                    ],
                }
            ]
        )
    )
    blocks = body["messages"][0]["content"]  # type: ignore[index]

    assert [block["type"] for block in blocks] == ["text"]


def test_a_role_the_messages_array_does_not_admit_is_dropped() -> None:
    body = _built(
        _turn(
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "prefix"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "real"}],
                },
            ]
        )
    )
    messages = body["messages"]

    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["user"]


def test_a_request_field_nobody_classified_fails_the_turn() -> None:
    with pytest.raises(Untranslatable) as caught:
        _built(_turn(a_field_nothing_here_reads={"nested": [1, 2, 3]}))

    assert caught.value.cause is DiscardCause.UPSTREAM_UNCLASSIFIED
    assert "request.a_field_nothing_here_reads" in caught.value.detail


def test_a_dropped_request_field_does_not_fail_the_turn() -> None:
    body = _built(
        _turn(
            store=False,
            include=["reasoning.encrypted_content"],
            prompt_cache_key="k",
            client_metadata={"session_id": "not-ours"},
            stream_options={"include_usage": True},
            service_tier="priority",
        )
    )

    assert set(body) <= MESSAGES_TOP_LEVEL


def test_a_thinking_block_survives_a_round_trip_through_the_carrier() -> None:
    block = {
        "type": "thinking",
        "thinking": "step one, step two",
        "signature": "AbC/123+xyz==",
    }

    assert decode_thinking(encode_thinking(block)) == block


def test_a_carrier_written_by_something_else_is_not_mistaken_for_one_of_ours() -> None:
    assert decode_thinking("not base64 at all") is None
    assert decode_thinking("") is None
    assert decode_thinking(encode_thinking({"type": "thinking"})[:-4] + "AAAA") is None


def test_a_reasoning_item_is_unpacked_back_into_the_block_it_came_from() -> None:
    block = {"type": "thinking", "thinking": "held", "signature": "sig"}
    body = _built(
        _turn(
            input=[
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": encode_thinking(block),
                }
            ]
        )
    )
    restored = body["messages"][0]["content"][0]  # type: ignore[index]

    assert restored["thinking"] == "held"
    assert restored["signature"] == "sig"


def test_a_reasoning_item_with_no_carrier_of_ours_is_dropped() -> None:
    body = _built(
        _turn(
            input=[
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "reasoning", "id": "rs_2", "encrypted_content": "foreign"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                },
            ]
        )
    )
    messages = body["messages"]

    assert isinstance(messages, list)
    assert len(messages) == 1


def test_the_translator_does_not_mutate_the_turn_it_was_given() -> None:
    items = ({"type": "message", "role": "user", "content": []},)
    turn = _turn(input=list(items))

    _built(turn)

    assert turn.input == items


# ------------------------------------------------------------------------------------
# System-level messages: the skills catalogue, which this wire used to throw away
# ------------------------------------------------------------------------------------


def _developer(text: str) -> dict[str, object]:
    """One developer-role message in the shape the Agent Runtime sends it.

    The runtime delivers its world-state sections this way -- the skills catalogue among
    them -- as an `item.message` with role `developer` and `input_text` content. Written
    from the runtime's shape rather than from what the code reads, because a fixture
    built to match the code certifies the code.
    """
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": text}],
    }


_CATALOGUE = "<skills_instructions>brief-summary: write a brief</skills_instructions>"

_USER_HELLO: dict[str, object] = {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "hello"}],
}
"""One ordinary turn, so a case can show a system-level message beside a real one."""


def test_a_developer_message_reaches_the_model_in_the_system_field() -> None:
    """The defect this exists for: it used to reach the model nowhere at all.

    A developer-role message is system-level content, and the Messages API has no
    developer role -- so this side dropped it, and the runtime's skills catalogue went
    with it. A Session whose skill was delivered into its pod and readable on disk
    answered that it had no skills, and the only trace was a classification row.

    Asserted on the text landing in `system` rather than on a block count, because the
    claim is that the model is told; a block present with the wrong content would
    satisfy a count.
    """
    body = _built(
        _turn(instructions="You are careful.", input=[_developer(_CATALOGUE)])
    )

    system = body["system"]
    assert isinstance(system, list)
    assert [str(block["text"]) for block in system] == [
        "You are careful.",
        _CATALOGUE,
    ]


def test_a_developer_message_is_not_also_a_turn_in_the_conversation() -> None:
    """One place, not two. Carried into `system` AND left in `messages` would show the
    model its own catalogue twice and, worse, show it once in the tenant's voice --
    which is an injection surface, because a developer block is exactly what a prompt
    would forge to look like.
    """
    body = _built(_turn(input=[_developer(_CATALOGUE), _USER_HELLO]))

    messages = body["messages"]
    assert isinstance(messages, list)
    assert [one["role"] for one in messages] == ["user"]
    assert _CATALOGUE not in str(messages)


def test_a_system_role_message_is_carried_the_same_way() -> None:
    """`system` is the older spelling of `developer` and reaches the same field.

    Both are accepted because the runtime has used each for this purpose and a wire
    translator cannot know which it will be sent next. A test naming only `developer`
    would grade one of the two decisions the constant encodes.
    """
    item = {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "be brief"}],
    }
    body = _built(_turn(input=[item]))

    system = body["system"]
    assert isinstance(system, list)
    assert [str(block["text"]) for block in system] == ["be brief"]


def test_several_system_level_messages_arrive_in_the_order_they_were_sent() -> None:
    """Order preserved, because the runtime's later section supersedes its earlier one.

    The runtime re-renders a world-state section each Turn and replays the history, so
    two catalogues can be in one request; the model reading them out of order would act
    on the stale one. A set-based or dict-based collection would pass a single-message
    case and lose this.
    """
    body = _built(
        _turn(
            input=[
                _developer("first: two skills"),
                _USER_HELLO,
                _developer("second: three skills"),
            ]
        )
    )

    system = body["system"]
    assert isinstance(system, list)
    assert [str(block["text"]) for block in system] == [
        "first: two skills",
        "second: three skills",
    ]


def test_the_volatile_half_of_the_system_field_carries_no_breakpoint() -> None:
    """The breakpoint stays on the instructions, which is what makes the cache work.

    A cache breakpoint marks the end of the cacheable prefix. The instructions are
    identical between Turns and the catalogue is re-rendered every Turn, so a breakpoint
    on the catalogue would put a block that never repeats inside the cached prefix and
    make every Turn a full miss on the system field. There are only four breakpoints in
    a request and three are already spent.
    """
    body = _built(
        _turn(instructions="You are careful.", input=[_developer(_CATALOGUE)])
    )

    system = body["system"]
    assert isinstance(system, list)
    assert "cache_control" in system[0]
    assert "cache_control" not in system[1]


def test_a_system_level_message_alone_still_produces_a_system_field() -> None:
    """No instructions and a catalogue is a real request, not an empty system field.

    An agent definition with empty instructions is allowed, and its Session still has a
    skills catalogue. A `_system` that returned early on empty instructions would drop
    the catalogue for exactly those Sessions -- the same defect again, narrowed to a
    subset nobody would think to test.
    """
    body = _built(_turn(input=[_developer(_CATALOGUE)]))

    system = body["system"]
    assert isinstance(system, list)
    assert [str(block["text"]) for block in system] == [_CATALOGUE]


def test_a_turn_with_neither_omits_the_system_field_rather_than_sending_it_empty() -> (
    None
):
    """`system: []` is a different request from one with no `system`, so neither is sent
    by accident. This is the case that would have passed vacuously before the field
    could have two sources."""
    body = _built(_turn(input=[_USER_HELLO]))

    assert "system" not in body


def test_a_role_that_is_neither_conversational_nor_system_level_is_still_dropped() -> (
    None
):
    """The widening is two roles, not every role.

    `tool` is the shape this guards: a role this wire has no field for must not silently
    become system-level content, because system content is the most trusted text in the
    request and a role nobody classified would be promoted straight into it.
    """
    item = {
        "type": "message",
        "role": "tool",
        "content": [{"type": "input_text", "text": "unexpected"}],
    }
    body = _built(_turn(input=[item, _USER_HELLO]))

    assert "system" not in body
    assert "unexpected" not in str(body)


def test_a_tool_list_arriving_only_inline_is_bound_rather_than_reported() -> None:
    """A request whose tool list arrives inline must not build a toolless body.

    This wire sends tools in the top-level array, which for a while was read as reason
    to drop the item that carries them inline -- true for the ordinary Responses shape
    and false for the one a `use_responses_lite` model asks for, where the inline item
    is the only copy. The body was still built, so all that stood between that and a
    Turn full of invented tool calls was a warning nobody was reading at the time.

    It is bound now, so the warning has nothing left to say and this pins the binding
    instead. A Turn offering no tools anywhere is still reported, by the test below.
    """
    turn = _turn(
        input=[
            {"type": "additional_tools", "tools": [{"type": "function", "name": "ls"}]},
            {"type": "message", "role": "user", "content": "hello"},
        ]
    )

    body = _built(turn)

    tools = body["tools"]
    assert isinstance(tools, list)
    assert [one["name"] for one in tools] == ["ls"]


def test_a_turn_offering_no_tools_anywhere_is_reported_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The quietest toolless shape has to speak, and speak where the log can hear it.

    No top-level tools and no inline item builds the same body as a Turn whose tools
    were every one dropped: the model reaches the provider with nothing bound and can
    only write its calls as prose. It is also what a runtime that never listed its
    tool catalogue sends, which is a live failure with no evidence behind it while
    this branch stays silent.

    The level is asserted, not just the line. Under this deployment's logging config
    the root logger holds no handler and sits at WARNING, so a census emitted lower is
    a guard that exists only in the source -- which is why the whole record list is
    compared rather than searched.
    """
    turn = _turn(input=[{"type": "message", "role": "user", "content": "hello"}])

    with caplog.at_level("INFO"):
        body = _built(turn)

    assert "tools" not in body
    assert [record.levelname for record in caplog.records] == ["WARNING"], [
        (r.levelname, r.getMessage()) for r in caplog.records
    ]
    assert "offered no tools" in caplog.records[0].getMessage()


def test_a_turn_whose_tools_survive_is_counted_and_not_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A healthy Turn is counted and nothing else, so a warning keeps meaning something.

    Every warning above fires on a request that carries no tools. If the ordinary case
    warned too, the level would stop separating anything and the one line worth reading
    would arrive in a stream nobody reads.
    """
    turn = _turn(
        input=[{"type": "message", "role": "user", "content": "hello"}],
        tools=[{"type": "function", "name": "ls", "parameters": {}}],
    )

    with caplog.at_level("INFO"):
        body = _built(turn)

    assert "tools" in body
    assert [record.levelname for record in caplog.records] == ["INFO"], [
        (r.levelname, r.getMessage()) for r in caplog.records
    ]


_APPLY_PATCH_SPEC: Final = {
    "type": "custom",
    "name": "apply_patch",
    "description": (
        "The apply_patch tool can be used to edit files. This is a FREEFORM tool, so "
        "do not wrap the patch in JSON."
    ),
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: begin_patch hunk+ end_patch",
    },
}
"""The runtime's own file-editing tool, in the shape it puts on the wire.

Not a fixture invented here: the runtime declares apply_patch as a freeform tool whose
call is constrained by a Lark grammar rather than a JSON schema, and offers it on every
Turn without being asked to.
"""


def test_the_runtimes_own_editing_tool_is_dropped_on_purpose(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """apply_patch does not reach the model, and that is a decision, not an oversight.

    The runtime offers this tool unprompted and its base_instructions teach the model to
    use it, so every Turn this platform serves reaches the provider with the model
    instructed in an editing tool it was never handed. Nothing breaks, because editing
    also works through the shell -- which is exactly why the absence is invisible and
    why it is pinned here instead of left to be re-derived.

    A grammar-constrained tool has no counterpart on this side: there is no field for a
    Lark definition and no item that carries a call back as raw text. Carrying it means
    building both, so anyone who does has to come here and say so deliberately.
    """
    turn = _turn(
        input=[{"type": "message", "role": "user", "content": "hello"}],
        tools=[
            {"type": "function", "name": "ls", "parameters": {}},
            _APPLY_PATCH_SPEC,
        ],
    )

    with caplog.at_level("INFO"):
        body = _built(turn)

    tools = body["tools"]
    assert isinstance(tools, list)
    assert [one["name"] for one in tools] == ["ls"]

    census = caplog.records[0].getMessage()
    assert "dropped=['apply_patch']" in census, census
    assert f"session={_SESSION}" in census, census


def test_every_census_shape_names_the_session_it_is_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A census nobody can attribute is a line that cannot be used.

    This gateway serves every Session in the namespace at once, so an unattributed
    census can neither be joined to the line the model side writes for the same request
    nor be told apart from another tenant's -- which makes it unreadable in exactly the
    situation it exists for. Found by a live test that read these lines back out of the
    deployed log and matched none of them.

    All four shapes, because the three that warn need this more than the one that
    counts: a warning is read while something is wrong, and "some Session reached the
    model with no tools" is not an answer anybody can act on.
    """
    shapes: list[dict[str, object]] = [
        {"input": [{"type": "message", "role": "user", "content": "hi"}]},
        {
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "additional_tools", "tools": []},
            ]
        },
        {
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [_APPLY_PATCH_SPEC],
        },
        {
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "ls", "parameters": {}}],
        },
    ]

    for shape in shapes:
        caplog.clear()
        with caplog.at_level("INFO"):
            _built(_turn(**shape))
        census = caplog.records[0].getMessage()
        assert f"session={_SESSION}" in census, census


def test_a_dropped_tool_that_has_no_name_is_named_by_its_kind(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The two most interesting absences carry no name field at all.

    web_search is addressed by its type and has no name, so a census reading `name`
    alone would report it as an empty string -- the one shape where the reader most
    needs to be told which tool went missing, rendered as nothing.
    """
    turn = _turn(
        input=[{"type": "message", "role": "user", "content": "hello"}],
        tools=[
            {"type": "function", "name": "ls", "parameters": {}},
            {"type": "web_search"},
        ],
    )

    with caplog.at_level("INFO"):
        _built(turn)

    census = caplog.records[0].getMessage()
    assert "dropped=['web_search']" in census, census


# ---- deferred MCP tools, which is how every MCP tool actually arrives -------------


_SEARCH_SPEC: Final = {
    "type": "tool_search",
    "execution": "client",
    "description": "Search for tools",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

_DISCOVERED: Final = {
    "type": "function",
    "name": "deepwiki__ask_deepwiki",
    "description": "Ask a repository a question.",
    "defer_loading": True,
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
}

_FOUND: Final = {
    "type": "tool_search_output",
    "call_id": "s1",
    "status": "completed",
    "execution": "client",
    "tools": [_DISCOVERED],
}

_CALLED: Final = {
    "type": "tool_search_call",
    "call_id": "s1",
    "execution": "client",
    "arguments": {"query": "deepwiki"},
}


def test_the_search_tool_is_offered_since_mcp_tools_defer_behind_it() -> None:
    """Dropping this tool is dropping every MCP tool the tenant granted.

    The runtime does not put a granted MCP tool in the request's tool list. It offers
    `tool_search` and defers the rest, and in codex-rs 0.149.0 that is not a setting:
    `tool_search_always_defer_mcp_tools` defaults true and its stage is `Removed`, so
    the config parser skips the key rather than honouring it. This wire dropped
    `tool_search` on the reading that it "asks the upstream to discover tools mid-turn".
    It does not -- `create_tool_search_tool` sets `execution: "client"` and the handler
    is a local BM25 search over the runtime's own registry, so nothing about it reaches
    a provider. The measured cost of that reading was every granted tool missing from
    nineteen consecutive live Turns.
    """
    body = _built(_turn(tools=[_SEARCH_SPEC]))

    tools = body["tools"]
    assert isinstance(tools, list)
    assert [one["name"] for one in tools] == ["tool_search"]
    assert tools[0]["input_schema"] == _SEARCH_SPEC["parameters"]


def test_a_discovered_tool_becomes_callable_on_the_turn_that_replays_it() -> None:
    """Discovery is worthless unless the discovered tool is then bound.

    A `tool_search_output` carries the specs it found. This wire sends tools in the
    top-level array and nowhere else, so a discovered tool left inside the item is one
    the model has just been told about and still cannot call -- the model then writes
    the call as prose and the Turn completes having run nothing.
    """
    body = _built(_turn(tools=[_SEARCH_SPEC], input=[_CALLED, _FOUND]))

    tools = body["tools"]
    assert isinstance(tools, list)
    assert "deepwiki__ask_deepwiki" in [one["name"] for one in tools]


def test_the_search_call_and_its_output_replay_as_a_call_and_a_result() -> None:
    """The history has to read as a tool call, or the next Turn contradicts itself.

    `tool_search_call` carries its arguments as a real object, unlike `function_call`
    which carries a JSON-encoded string -- so it needs its own builder rather than the
    one beside it, and a builder that assumed the string shape would send `{}`.
    """
    body = _built(_turn(tools=[_SEARCH_SPEC], input=[_CALLED, _FOUND]))

    messages = body["messages"]
    assert isinstance(messages, list)
    call = messages[0]["content"][0]
    assert call["type"] == "tool_use"
    assert call["name"] == "tool_search"
    assert call["id"] == "s1"
    assert call["input"] == {"query": "deepwiki"}
    result = messages[1]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "s1"


def test_a_tool_discovered_twice_is_offered_once() -> None:
    """Two Turns of discovery must not put one name in the array twice.

    The upstream refuses a tool list carrying a duplicate name, so a Session that
    searched for the same tool on two Turns would fail on the second -- a defect that
    appears only after a conversation gets long enough to repeat itself.
    """
    again = {**_FOUND, "call_id": "s2"}
    body = _built(_turn(tools=[_SEARCH_SPEC], input=[_FOUND, again]))

    tools = body["tools"]
    assert isinstance(tools, list)
    names = [one["name"] for one in tools]
    assert names.count("deepwiki__ask_deepwiki") == 1


def test_tools_carried_inline_by_additional_tools_are_bound_too() -> None:
    """The same rule, on the other item that carries a tool list inline.

    The census already warned when this item appeared beside an empty top-level array,
    which is how the shape was known about; warning was all it did.
    """
    inline = {"type": "additional_tools", "role": "system", "tools": [_DISCOVERED]}
    body = _built(_turn(tools=[_SEARCH_SPEC], input=[inline]))

    tools = body["tools"]
    assert isinstance(tools, list)
    assert "deepwiki__ask_deepwiki" in [one["name"] for one in tools]
