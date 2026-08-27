"""A registered server's failure becomes a published code, and never its own words.

Tier 1 (local, no infrastructure). The exception shapes exercised here are the ones a
real SDK produces, not tidied stand-ins: a transport failure arrives inside one
`ExceptionGroup` while opening and two while calling, so a classifier that matched the
bare type would fall through every arm and record a refused connection as a fault of
this platform.
"""

from __future__ import annotations

import logging

import httpx2
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CONNECTION_CLOSED,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    REQUEST_TIMEOUT,
)

from managed_agent.core.errors import ErrorCode
from managed_agent.core.ports import SessionNotVisible
from managed_agent.core.vfs.evidence import EvidenceStorageUnconfigured
from managed_agent.gateway.tool import error_map
from managed_agent.gateway.tool.scope_clamp import ScopeRefusal

_LEAKY = (
    "psycopg2: FATAL password authentication failed for user 'acme_ro' "
    "at internal-host-7.corp:5432"
)


def _grouped(exc: Exception, depth: int) -> Exception:
    """`exc` inside `depth` nested `ExceptionGroup`s, the way anyio delivers it."""
    wrapped: Exception = exc
    for _ in range(depth):
        wrapped = ExceptionGroup("tg", [wrapped])
    return wrapped


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (MCPError(code=REQUEST_TIMEOUT, message="x"), ErrorCode.TOOL_TIMED_OUT),
        (TimeoutError("x"), ErrorCode.TOOL_TIMED_OUT),
        (
            _grouped(MCPError(code=REQUEST_TIMEOUT, message="x"), 2),
            ErrorCode.TOOL_TIMED_OUT,
        ),
        (MCPError(code=INVALID_PARAMS, message="x"), ErrorCode.REQUEST_INVALID),
        (MCPError(code=METHOD_NOT_FOUND, message="x"), ErrorCode.TOOL_UNAVAILABLE),
        (MCPError(code=CONNECTION_CLOSED, message="x"), ErrorCode.TOOL_UNAVAILABLE),
        (_grouped(httpx2.ConnectError("refused"), 1), ErrorCode.TOOL_UNAVAILABLE),
        (FileNotFoundError(2, "No such file or directory"), ErrorCode.TOOL_UNAVAILABLE),
        (httpx2.ReadTimeout("slow"), ErrorCode.TOOL_TIMED_OUT),
        (RuntimeError("nobody knows"), ErrorCode.INTERNAL),
        # Not an upstream failure at all: the Scope read a tool call makes before it
        # opens anything raises this, and it arrives on the same path. Its own code
        # rather than `INTERNAL`, which would report a swept Session as a malfunction
        # of this platform.
        (
            SessionNotVisible("11111111-1111-4111-8111-111111111111"),
            ErrorCode.SESSION_NOT_FOUND,
        ),
    ],
)
def test_each_measured_failure_shape_becomes_its_published_code(
    exc: BaseException, expected: ErrorCode
) -> None:
    assert error_map.classify(exc) is expected


def test_an_upstream_internal_error_is_the_registering_teams_fault_not_ours() -> None:
    """-32603 from upstream is `tool.unavailable`, never `platform.internal`.

    The distinction is who gets paged. A server reporting its own internal error is the
    registering team's problem, and recording it as this platform's sends an operator to
    read logs of a service that did nothing wrong.
    """
    upstream = MCPError(code=INTERNAL_ERROR, message="upstream blew up")

    assert error_map.classify(upstream) is ErrorCode.TOOL_UNAVAILABLE


def test_an_unrecognized_companion_cannot_mask_a_recognized_cause() -> None:
    """A group ordered by failure time routinely puts a teardown ahead of the cause."""
    group = ExceptionGroup(
        "tg", [RuntimeError("stream teardown"), httpx2.ConnectError("refused")]
    )

    assert error_map.classify(group) is ErrorCode.TOOL_UNAVAILABLE


def test_the_upstreams_own_words_reach_the_log_and_not_the_envelope(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The whole point of the module: an operator gets the text, the model does not."""
    with caplog.at_level(logging.ERROR, logger="managed_agent.gateway.tool.error_map"):
        envelope = error_map.record(MCPError(code=0, message=_LEAKY), "billing_lookup")

    rendered = envelope.model_dump_json()
    assert "internal-host-7.corp" not in rendered
    assert "acme_ro" not in rendered
    assert "billing_lookup" in rendered

    correlation_id = envelope.detail["correlation_id"]
    assert isinstance(correlation_id, str)
    logged = "\n".join(
        record.getMessage() + (record.exc_text or "") for record in caplog.records
    )
    assert _LEAKY in caplog.text
    assert correlation_id in logged


def test_a_refusal_leaves_as_a_failed_tool_call_carrying_the_whole_envelope() -> None:
    envelope = error_map.refusal(ErrorCode.TOOL_NOT_GRANTED, "ghost_tool", "abc123")

    result = error_map.as_tool_result(envelope)

    assert result.is_error is True
    assert len(result.content) == 1
    only = result.content[0]
    assert only.type == "text"
    assert only.text == envelope.model_dump_json()


def test_a_read_refusal_puts_the_envelope_in_data_and_a_fixed_sentence_in_message() -> (
    None
):
    envelope = error_map.refusal(ErrorCode.TOOL_NOT_GRANTED, "db://acme/x", "abc123")

    error = error_map.as_mcp_error(envelope)

    assert error.message == "the platform refused this read"
    assert error.data == envelope.model_dump(mode="json")
    assert "db://acme/x" not in error.message


def test_the_two_scope_refusals_share_a_code_and_do_not_share_a_sentence() -> None:
    """One shape, two messages -- the clause of ADR-034 that is easiest to half-build.

    A Scope and a tool's bindings can fail to meet in two directions, and a tenant's
    next move differs by direction: one is fixed in the Scope a Session is created
    with, the other by a Binding the registration never declared. Both are refused
    under one code so a caller keeps a single branch, which is the half that a shared
    envelope gets right by construction -- and the half it silently gets wrong is the
    text, because reusing one sentence for both compiles, passes every code-level
    check, and tells half of its readers to go and fix something that is already
    correct.

    Neither may name a value. `dimension` here is a term out of the tenant's own
    registration or their own create call; the Scope's value behind it is what a model
    probing for the Session's bounds is trying to read.
    """
    lacks = error_map.out_of_scope(
        "repo_map", "repository", ScopeRefusal.DIMENSION_NOT_IN_SCOPE
    )
    unbound = error_map.out_of_scope(
        "repo_map", "repository", ScopeRefusal.DIMENSION_NOT_BOUND
    )

    assert lacks.code is unbound.code is ErrorCode.TOOL_OUT_OF_SCOPE
    # One tool and one dimension across both, so the only thing left that can differ
    # is the sentence. Given each its own dimension, this comparison passes on the
    # dimension alone and would go on passing with both causes sharing a sentence --
    # which is the failure it exists to catch.
    assert lacks.message != unbound.message
    assert "repository" in lacks.message and "repository" in unbound.message
    assert "repo_map" in lacks.message and "repo_map" in unbound.message


def test_every_code_this_module_can_produce_is_a_member_of_the_published_set() -> None:
    """Guarded against an unpublished code being introduced by a mapping edit.

    Read off the mapping itself plus the fallbacks, rather than off a list written here
    that would go stale the moment a row is added.
    """
    producible = set(error_map._JSON_RPC_TO_CODE.values()) | {
        error_map.classify(exc)
        for exc in (
            TimeoutError("x"),
            httpx2.ConnectError("x"),
            RuntimeError("x"),
            MCPError(code=-31999, message="a code nothing maps"),
        )
    }

    assert producible, "nothing was collected, so the membership check below is vacuous"
    assert producible <= set(ErrorCode)


def test_an_unconfigured_evidence_store_is_not_reported_as_the_servers_failure() -> (
    None
):
    """An unset `MAP_OBJECT_BUCKET` is this platform's fault and must read as one.

    The Evidence store refuses when nothing is wired behind it, and that refusal reaches
    this module on the call path because capture sits inside the same `try` as the call.
    Left to fall through to the default arm it took `refusal`'s envelope, whose text is
    fixed: "the registered server behind {subject} did not complete the call". That
    sentence is false in both halves -- the registered server completed the call, and
    what failed afterwards was this platform's own storage -- so it sends a tenant to
    debug a healthy service of their own.

    `SessionNotVisible` already leaves by a different door for the same reason. This is
    the second case of it, and the two are told apart at the point of recording rather
    than by the code, because both are honestly `INTERNAL`-shaped to a caller.
    """
    envelope = error_map.record(
        EvidenceStorageUnconfigured("MAP_OBJECT_BUCKET is not set"), "some_tool"
    )

    assert "the registered server" not in envelope.message


def test_an_unconfigured_evidence_store_still_names_what_the_caller_asked_for(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The subject and a correlation id survive the different door.

    A refusal that dropped the subject would leave a tenant holding a failure with
    nothing to match it against, and one that dropped the correlation id would leave an
    operator unable to find the logged cause -- which is the whole reason the text is
    allowed to say so little.
    """
    with caplog.at_level(logging.ERROR):
        envelope = error_map.record(
            EvidenceStorageUnconfigured("MAP_OBJECT_BUCKET is not set"), "some_tool"
        )

    assert envelope.detail["subject"] == "some_tool"
    assert envelope.detail["correlation_id"]
    assert "upstream MCP failure" not in caplog.text
