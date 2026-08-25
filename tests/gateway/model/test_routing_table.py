"""The declared model-to-upstream table: parsed once, and never guessed at.

Tier 1, no infrastructure. Nothing here reaches a network or a vault -- the table is a
pure function of the document it was handed, which is the whole point of parsing it once
at start-up rather than reading a mapping per Turn.

Every refusal below is a refusal at parse time. That is what is being graded: a document
naming a shape this build does not have, or naming one model twice, has to fail before
the process serves anything, because the alternative is discovering it on the first Turn
of whichever tenant happened to name that model.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from managed_agent.gateway.model.router import (
    AuthScheme,
    RoutingEntry,
    RoutingTable,
    UnroutableModel,
    UpstreamWire,
    routing_table_from_json,
)

_THREE_WIRES: dict[str, list[dict[str, object]]] = {
    "entries": [
        {
            "model": "gpt-5-codex",
            "wire": "responses",
            "base_url": "https://api.openai.com/v1",
            "auth_scheme": "bearer",
            "credential_name": "map/upstream/openai",
            "query_params": {},
        },
        {
            "model": "claude-opus-5",
            "wire": "anthropic_messages",
            "base_url": "https://map-foundry.services.ai.azure.com/anthropic",
            "auth_scheme": "api_key",
            "credential_name": "map/upstream/foundry",
            "query_params": {},
        },
        {
            "model": "some-chat-model",
            "wire": "chat_completions",
            "base_url": "https://vendor.example.com/v1",
            "auth_scheme": "bearer",
            "credential_name": "map/upstream/vendor",
            "query_params": {"api-version": "2025-04-01-preview"},
        },
    ]
}


def _document(payload: object) -> bytes:
    return json.dumps(payload).encode()


def _table() -> RoutingTable:
    return routing_table_from_json(_document(_THREE_WIRES))


def test_each_declared_model_reaches_the_upstream_its_entry_names() -> None:
    """Three models on three different shapes, each answered with its own entry."""
    table = _table()

    assert table.entry_for("gpt-5-codex") == RoutingEntry(
        model="gpt-5-codex",
        wire=UpstreamWire.RESPONSES,
        base_url="https://api.openai.com/v1",
        auth_scheme=AuthScheme.BEARER,
        credential_name="map/upstream/openai",
    )
    assert table.entry_for("claude-opus-5").wire is UpstreamWire.ANTHROPIC_MESSAGES
    assert table.entry_for("claude-opus-5").auth_scheme is AuthScheme.API_KEY
    assert table.entry_for("some-chat-model").wire is UpstreamWire.CHAT_COMPLETIONS
    assert table.entry_for("some-chat-model").query_params == (
        ("api-version", "2025-04-01-preview"),
    )


def test_an_undeclared_model_is_refused_by_name_and_not_served_by_a_default() -> None:
    """There is no nearest match and no default entry, so the lookup raises."""
    table = _table()

    with pytest.raises(UnroutableModel) as refused:
        table.entry_for("gpt-5-codex-preview")

    assert refused.value.model == "gpt-5-codex-preview"
    assert "gpt-5-codex-preview" in str(refused.value)


def test_two_entries_for_one_model_are_refused_rather_than_one_winning() -> None:
    """Silently keeping the last would route a model by document order."""
    twice = RoutingEntry(
        model="gpt-5-codex",
        wire=UpstreamWire.RESPONSES,
        base_url="https://api.openai.com/v1",
        auth_scheme=AuthScheme.BEARER,
        credential_name="map/upstream/openai",
    )
    other = RoutingEntry(
        model="gpt-5-codex",
        wire=UpstreamWire.RESPONSES,
        base_url="https://elsewhere.example.com/v1",
        auth_scheme=AuthScheme.BEARER,
        credential_name="map/upstream/elsewhere",
    )

    with pytest.raises(ValueError, match="two routing entries"):
        RoutingTable([twice, other])


@pytest.mark.parametrize(
    ("mutation", "why"),
    [
        ({"wire": "grpc"}, "a shape this build has no handler for"),
        ({"base_url": "http://api.openai.com/v1"}, "a plaintext hop"),
        ({"auth_scheme": "x-api-key"}, "a credential form nobody implements"),
        ({"credential_name": ""}, "an entry naming no vault entry"),
        ({"model": ""}, "an entry naming no model"),
        ({"nonsense": "field"}, "a key nothing reads"),
    ],
)
def test_a_document_this_build_cannot_serve_is_refused_at_the_parse(
    mutation: dict[str, object], why: str
) -> None:
    entry = {**_THREE_WIRES["entries"][0], **mutation}

    with pytest.raises(ValidationError):
        routing_table_from_json(_document({"entries": [entry]}))

    assert why, "the parametrisation names why, so a failure says which case broke"


def test_a_table_with_no_entries_is_refused() -> None:
    """A process that parsed an empty table would answer 404 for every model."""
    with pytest.raises(ValidationError):
        routing_table_from_json(_document({"entries": []}))


def test_a_body_that_is_not_the_table_document_is_refused() -> None:
    with pytest.raises(ValidationError):
        routing_table_from_json(b"{}")


def test_query_parameter_key_order_does_not_change_the_parsed_entry() -> None:
    """Two documents that differ only in key order describe one upstream.

    Order-dependent entries would compare unequal after a ConfigMap edit that changed
    nothing, and the outbound query string would reorder with it.
    """
    forward = dict(_THREE_WIRES["entries"][2])
    forward["query_params"] = {"a": "1", "b": "2"}
    backward = dict(_THREE_WIRES["entries"][2])
    backward["query_params"] = {"b": "2", "a": "1"}

    one = routing_table_from_json(_document({"entries": [forward]}))
    two = routing_table_from_json(_document({"entries": [backward]}))

    assert one.entry_for("some-chat-model") == two.entry_for("some-chat-model")
    assert one.entry_for("some-chat-model").query_params == (("a", "1"), ("b", "2"))


def test_the_declared_model_set_cannot_be_edited_by_whoever_reads_it() -> None:
    """The table is fixed for the process's life, so its answer is not a live view."""
    table = _table()
    declared = table.declared_models()

    assert declared == {"gpt-5-codex", "claude-opus-5", "some-chat-model"}
    assert isinstance(declared, frozenset)
    with pytest.raises(AttributeError):
        declared.add("smuggled")  # type: ignore[attr-defined]
    assert table.declared_models() == declared
