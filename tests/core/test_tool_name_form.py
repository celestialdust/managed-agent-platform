"""The committed form of a registered name, and the byte budget it is committed to.

Tier 1, no infrastructure. What is being graded is the *shape* the platform promises a
Grant: the Agent Runtime rewrites names it is handed, and the whole reason these
patterns are narrow is that every transformation it applies is the identity function
over a name that already matches them. A pattern loosened by one character class is a
name the runtime is free to rewrite, and a Grant written against the original then
resolves to nothing.

The arithmetic is asserted rather than assumed. The tool-name length limit is derived
from the runtime's 128-byte ceiling minus a reserve for the prefix it prepends, and
those three numbers only mean anything together -- a reserve raised without lowering the
limit silently permits a name that cannot be qualified.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from managed_agent.core.registration.scope_binding import (
    MAX_QUALIFIED_TOOL_NAME_BYTES,
    MAX_TOOL_NAME_BYTES,
    QUALIFICATION_RESERVE_BYTES,
    ServerEndpoint,
    ServerName,
    StdioServer,
    StreamableHttpServer,
    ToolName,
    qualification_fits,
)


class _Named(BaseModel):
    """A model declaring the annotated types, which is the only place they bite.

    `ToolName("...")` at a call site is `str("...")` -- `Annotated` is erased at
    runtime. So the constraint is exercised the way production exercises it: as a field
    pydantic parses.
    """

    tool: ToolName = "search"
    server: ServerName = "deepwiki"


class _Endpoint(BaseModel):
    endpoint: ServerEndpoint


def _parse_endpoint(**payload: object) -> StdioServer | StreamableHttpServer:
    return _Endpoint.model_validate({"endpoint": payload}).endpoint


def test_the_reserve_and_the_limit_add_up_to_the_runtime_ceiling() -> None:
    assert (
        MAX_TOOL_NAME_BYTES + QUALIFICATION_RESERVE_BYTES
        == MAX_QUALIFIED_TOOL_NAME_BYTES
    )
    assert MAX_TOOL_NAME_BYTES == 96


def test_a_tool_name_at_the_byte_limit_parses() -> None:
    at_limit = "a" * MAX_TOOL_NAME_BYTES

    assert _Named(tool=at_limit).tool == at_limit


def test_a_tool_name_one_byte_over_the_limit_is_refused() -> None:
    """96 is the last name that can be qualified inside the runtime's 128 bytes.

    A 97-character name would be truncated by the runtime rather than refused here, and
    a truncated name is a name a Grant cannot address.
    """
    with pytest.raises(ValidationError):
        _Named(tool="a" * (MAX_TOOL_NAME_BYTES + 1))


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("Search", "uppercase survives the sanitizer, so Search and search are two"),
        ("read-wiki", "a hyphen is rewritten to an underscore by the runtime"),
        ("2fast", "a leading digit reads as a sanitized name rather than a chosen one"),
        ("read wiki", "a space is rewritten to an underscore by the runtime"),
        ("_private", "a leading underscore collides with a sanitized punctuation name"),
        ("", "an empty name addresses nothing"),
    ],
)
def test_a_tool_name_the_runtime_would_rewrite_is_refused(name: str, why: str) -> None:
    with pytest.raises(ValidationError):
        _Named(tool=name)
    assert why, "every case states the transformation it is protecting the name from"


def test_a_lowercase_underscored_tool_name_parses() -> None:
    """The three names the project's own MCP server offers all pass unchanged."""
    for name in ("read_wiki_structure", "read_wiki_contents", "ask_question"):
        assert _Named(tool=name).tool == name


@pytest.mark.parametrize("name", ["Deepwiki", "-lead", "9lives", "a" * 64])
def test_a_server_name_outside_the_committed_form_is_refused(name: str) -> None:
    with pytest.raises(ValidationError):
        _Named(server=name)


def test_a_server_name_may_carry_a_hyphen_where_a_tool_name_may_not() -> None:
    """A server name never reaches the model, so the sanitizer never sees it.

    It is the tool name the runtime qualifies and shows; the server name here is the
    handle an agent definition uses, and hyphenating it costs nothing.
    """
    assert _Named(server="acme-crm").server == "acme-crm"
    with pytest.raises(ValidationError):
        _Named(tool="acme-crm")


def test_qualification_fits_at_the_reserve_and_not_one_byte_past_it() -> None:
    """`mcp__` + name + `__` is seven bytes of overhead against the 32-byte reserve."""
    largest = QUALIFICATION_RESERVE_BYTES - len("mcp____")
    assert largest == 25

    assert qualification_fits("g" * largest)
    assert not qualification_fits("g" * (largest + 1))


def test_a_stdio_body_carrying_a_url_is_refused() -> None:
    """The union is exclusive, so nothing downstream decides which half to believe."""
    with pytest.raises(ValidationError):
        _parse_endpoint(
            transport="stdio",
            command="npx",
            credential_ref="vault/acme/crm",
            credential_env_var="CRM_TOKEN",
            url="https://mcp.deepwiki.com",
        )


def test_a_streamable_http_endpoint_must_be_https() -> None:
    with pytest.raises(ValidationError):
        _parse_endpoint(
            transport="streamable_http",
            url="http://mcp.deepwiki.com",
            credential_ref="vault/acme/deepwiki",
        )


def test_the_union_picks_its_type_from_the_transport_tag_alone() -> None:
    over_http = _parse_endpoint(
        transport="streamable_http",
        url="https://mcp.deepwiki.com",
        credential_ref="vault/acme/deepwiki",
    )
    over_stdio = _parse_endpoint(
        transport="stdio",
        command="npx",
        args=["-y", "mcp-remote"],
        credential_ref="vault/acme/deepwiki",
        credential_env_var="DEEPWIKI_TOKEN",
    )

    assert isinstance(over_http, StreamableHttpServer)
    assert isinstance(over_stdio, StdioServer)
    assert over_stdio.args == ("-y", "mcp-remote")


def test_an_http_endpoint_defaults_its_credential_header() -> None:
    """The common case needs no answer, and naming one stays available."""
    endpoint = _parse_endpoint(
        transport="streamable_http",
        url="https://mcp.deepwiki.com",
        credential_ref="vault/acme/deepwiki",
    )

    assert isinstance(endpoint, StreamableHttpServer)
    assert endpoint.credential_header == "Authorization"


def test_a_credential_header_that_is_not_a_header_name_is_refused() -> None:
    """`credential_env_var` is a pattern and its sibling was `min_length=1`, which
    accepts a name no HTTP request can carry.

    Not an injection: measured against httpx2 2.12.0, h11 refuses each of these at
    serialization with `Illegal header name` and nothing reaches the wire. What it costs
    is a registration that can never work, discovered on every tool call rather than
    once at registration -- the tenant sees `tool.unavailable` for ever and the reason
    sits in a platform log it cannot read.

    The permitted set is RFC 7230's `tchar` rather than something narrower, because a
    check that refuses a legal header name is a check somebody deletes.
    """
    for header in ("X-Api-Key: z\r\nX-Injected", "Ok\nBad", "  ", "", "a" * 65):
        with pytest.raises(ValidationError):
            _parse_endpoint(
                transport="streamable_http",
                url="https://mcp.deepwiki.com",
                credential_ref="vault/acme/deepwiki",
                credential_header=header,
            )

    for header in ("Authorization", "X-Api-Key", "x_api_key", "X-Api-Key1"):
        parsed = _parse_endpoint(
            transport="streamable_http",
            url="https://mcp.deepwiki.com",
            credential_ref="vault/acme/deepwiki",
            credential_header=header,
        )
        assert isinstance(parsed, StreamableHttpServer)
        assert parsed.credential_header == header


def test_an_endpoint_carries_a_vault_reference_and_never_a_secret() -> None:
    """`credential_ref` is required, so a registration cannot omit where the secret is.

    The field is a name, and there is no field on either transport that could hold the
    value itself -- `extra="forbid"` is what makes that true rather than conventional.
    """
    for model in (StdioServer, StreamableHttpServer):
        assert "credential_ref" in model.model_fields
        assert model.model_config["extra"] == "forbid"
        assert not any(
            "secret" in field or "token" in field or field.endswith("_value")
            for field in model.model_fields
        ), f"{model.__name__} has a field that could hold a credential value"
