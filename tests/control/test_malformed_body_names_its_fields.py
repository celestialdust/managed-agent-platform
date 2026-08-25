"""What a caller learns when its body was rejected before any route ran.

This is the most common refusal this API produces and the one a caller is likeliest to
hit while integrating, so what the body carries decides whether the next attempt is a
fix or a guess.

Two properties, pulling against each other, and both are asserted here because
satisfying either alone is easy:

1. **The rejected field is named.** A refusal saying only that one thing was wrong
   leaves the caller to find which, and there is nothing in the response to find it
   with.
2. **Nothing the caller submitted is echoed.** Pydantic's own diagnostics carry an
   `input` key holding the offending VALUE. On a field named `api_key` or `token` that
   value is a credential, and a response body ends up in access logs, in error trackers,
   and in whatever the caller prints while debugging. A refusal that leaks it is worse
   than one that says nothing.

The `loc` path is what makes both possible at once: it holds field names and array
indices and never values, so it is the one part of a validation failure that is safe to
publish.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.refusals import (
    _MOST_FIELDS_NAMED,
    _MOST_REASONS_GIVEN,
    _fields_named,
    _reasons_given,
)
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import TenantId

A_SECRET = "sk-this-value-must-never-appear-in-a-response"


class Unused:
    """Every port raises: a body is rejected before a route reads one.

    So a case here that passed by reaching a store would be asserting that store's
    behaviour, and would keep passing after the validation handler was deleted.
    """

    def __getattr__(self, name: str) -> Any:
        async def refuse(*args: object, **kwargs: object) -> Any:
            raise AssertionError(f"a malformed body reached {name}")

        return refuse


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    platform = Platform(
        event_log_append=Unused(),
        event_log_range=Unused(),
        definition_registry=Unused(),
        tool_registry=Unused(),
        session_registry=Unused(),
        webhooks=Unused(),
        environment_store=Unused(),
        turn_dispatch=Unused(),
        file_store=unconfigured_file_store(),
        skill_store=Unused(),
    )
    async with AsyncClient(
        transport=ASGITransport(app=create_app(platform)),
        base_url="http://control-plane",
        headers={TENANT_HEADER: str(TenantId(uuid.uuid4()))},
    ) as made:
        yield made


# ---------------------------------------------------------------------------------
# The two properties, over the wire
# ---------------------------------------------------------------------------------


async def test_a_rejected_body_names_the_fields_it_rejected(
    client: AsyncClient,
) -> None:
    """The caller's next move is to fix a field, so the field has to be in the body.

    An earlier version of this handler sent `problem_count` alone. It was accurate and
    it was not actionable: a caller learned that one thing was wrong about a body with
    six fields in it.
    """
    refused = await client.post("/v1/sessions", json={})

    assert refused.status_code == 400, refused.text
    body: dict[str, Any] = refused.json()
    assert body["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    named = body["error"]["detail"]["fields"]
    assert "environment_id" in named, named
    assert "definition_id" in named, named


async def test_a_submitted_value_is_never_echoed_back(client: AsyncClient) -> None:
    """**The case worth breaking the build over.**

    Pydantic's error list carries the offending value under `input`. Forwarding that
    list is the obvious implementation and it puts whatever the caller sent into the
    response
    -- so a wrong-typed `api_key` comes back in full, into every log holding a response
    body. Asserted against the whole response text rather than one field, because the
    value must not appear anywhere in it: not in `detail`, not in the message, not in a
    field name.
    """
    refused = await client.post(
        "/v1/sessions", json={"environment_id": {"api_key": A_SECRET}}
    )

    assert refused.status_code == 400
    assert A_SECRET not in refused.text, (
        "a value the caller submitted came back in the refusal, so any credential sent "
        "to the wrong field is now in every log that holds a response body"
    )


async def test_the_count_and_the_names_are_both_present(client: AsyncClient) -> None:
    """Both, because each answers a question the other does not.

    The count is the whole answer to "was that all of it"; the names are the answer to
    "which". With the list capped, a caller holding fewer names than problems needs the
    count to know the list was cut.
    """
    refused = await client.post("/v1/sessions", json={})

    detail = refused.json()["error"]["detail"]
    assert isinstance(detail["problem_count"], int)
    assert detail["problem_count"] >= 1
    assert isinstance(detail["fields"], str)


# ---------------------------------------------------------------------------------
# The path-to-name reduction, at the unit
# ---------------------------------------------------------------------------------


def test_the_request_part_is_dropped_from_the_name() -> None:
    """`body.environment_id` reads as though `body` were a field. It is not: it is which
    part of the request the field was in, and the caller already knows that."""
    assert _fields_named([{"loc": ("body", "environment_id")}]) == ["environment_id"]
    assert _fields_named([{"loc": ("query", "limit")}]) == ["limit"]


def test_a_body_that_is_not_an_object_is_named_body() -> None:
    """The one case where the request part IS the answer: there is no field to name,
    and dropping the only segment would produce a refusal naming nothing at all."""
    assert _fields_named([{"loc": ("body",)}]) == ["body"]


def test_an_index_survives_so_the_caller_knows_which_element() -> None:
    """ "one of your skills is wrong" is not actionable over a list of forty."""
    assert _fields_named([{"loc": ("body", "skills", 0, "name")}]) == ["skills.0.name"]


def test_one_field_failing_two_rules_is_named_once() -> None:
    """It is one field to fix. Naming it twice implies two."""
    twice = [
        {"loc": ("body", "limit")},
        {"loc": ("body", "limit")},
    ]
    assert _fields_named(twice) == ["limit"]


def test_the_order_does_not_depend_on_the_order_they_failed_in() -> None:
    """Sorted, so two requests failing the same way produce the same string -- which is
    what lets a caller compare two refusals, and a test assert on one."""
    assert _fields_named([{"loc": ("body", "b")}, {"loc": ("body", "a")}]) == ["a", "b"]


def test_the_list_is_capped_and_says_how_many_it_left_out() -> None:
    """An unbounded join is a caller-controlled string in a response: a body with a
    thousand bad fields would otherwise come back naming a thousand of them. Capped, and
    the remainder is stated -- a truncated list with no sign of truncation would read as
    the complete answer."""
    many = [{"loc": ("body", f"field{index:03d}")} for index in range(40)]

    named = _fields_named(many)

    assert len(named) == _MOST_FIELDS_NAMED + 1
    assert named[-1] == f"and {40 - _MOST_FIELDS_NAMED} more"
    assert named[0] == "field000"


def test_a_failure_carrying_no_location_names_nothing_rather_than_raising() -> None:
    """A validation error with no `loc` is not a shape this codebase produces, and the
    reduction still has to answer: a refusal handler that raises turns a 400 into a 500
    and loses the refusal entirely."""
    assert _fields_named([{}]) == []
    assert _fields_named([]) == []


# ---------------------------------------------------------------------------------
# Why it was rejected, for the rejections whose reason this codebase wrote
# ---------------------------------------------------------------------------------

A_TYPO_REASON = (
    "Value error, tool ask_question: the Scope Binding for dimension repository "
    "names argument repoNmae, which this tool does not declare"
)


def test_a_validator_this_repo_wrote_has_its_reason_published() -> None:
    """A field name is not always enough, and this is the case that proves it.

    A tool binding argument `repoNmae` when the tool declares `repoName` is refused, and
    the tenant's next move is to find a one-character typo. Told only that `tools.0` was
    wrong, they are comparing two identical-looking strings across a catalogue. The
    validator's own message names the tool and the argument, which is the whole reason
    that message exists.
    """
    raised = [
        {"type": "value_error", "loc": ("body", "tools", 0), "msg": A_TYPO_REASON}
    ]

    given = _reasons_given(raised)

    assert len(given) == 1
    assert "ask_question" in given[0]
    assert "repoNmae" in given[0]


def test_a_built_in_check_contributes_no_reason() -> None:
    """Pydantic's own `msg` for a built-in check is a generic sentence.

    "Field required" beside a field named `environment_id` says nothing the field name
    did not, and publishing it would put the framework's wording on this API's surface,
    where a caller could come to match on it.
    """
    built_in = [
        {"type": "missing", "loc": ("body", "x"), "msg": "Field required"},
        {"type": "string_type", "loc": ("body", "y"), "msg": "Input should be a str"},
        {
            "type": "extra_forbidden",
            "loc": ("body", "z"),
            "msg": "Extra inputs are not permitted",
        },
    ]

    assert _reasons_given(built_in) == []


def test_the_frameworks_own_prefix_is_stripped() -> None:
    """`Value error, ` is the wrapper's, not the validator's.

    A caller reading a message that opens by naming the framework learns something about
    this platform's implementation instead of about their own request.
    """
    given = _reasons_given(
        [{"type": "value_error", "msg": "Value error, the domain must be lower case"}]
    )

    assert given == ["the domain must be lower case"]


def test_one_reason_said_twice_is_said_once() -> None:
    """Two entries carrying one sentence is one thing to fix."""
    twice = [
        {"type": "value_error", "msg": "Value error, no Scope Binding declared"},
        {"type": "value_error", "msg": "Value error, no Scope Binding declared"},
    ]

    assert _reasons_given(twice) == ["no Scope Binding declared"]


def test_the_reasons_are_capped_and_say_how_many_they_left_out() -> None:
    """Fewer than the field cap on purpose: a reason is a sentence, and five sentences
    is already a caller reading rather than scanning. Truncation is stated, because a
    cut list with no sign of cutting reads as the whole answer."""
    many = [
        {"type": "value_error", "msg": f"Value error, reason {index}"}
        for index in range(12)
    ]

    given = _reasons_given(many)

    assert len(given) == _MOST_REASONS_GIVEN + 1
    assert given[-1] == f"and {12 - _MOST_REASONS_GIVEN} more"


def test_an_entry_with_no_message_contributes_nothing_rather_than_a_blank() -> None:
    """A blank entry in the joined string reads as a reason nobody wrote down."""
    assert _reasons_given([{"type": "value_error", "msg": ""}]) == []
    assert _reasons_given([{"type": "value_error"}]) == []


async def test_a_built_in_only_rejection_carries_no_reasons_key_at_all(
    client: AsyncClient,
) -> None:
    """Absent, not present and empty.

    A blank `reasons` says "there was no reason", which is a different claim from "every
    rejection here was a built-in check whose reason is its own field name". The same
    distinction this codebase draws when it omits the sandbox document's network keys
    rather than writing them off.
    """
    refused = await client.post("/v1/sessions", json={})

    assert "reasons" not in refused.json()["error"]["detail"]


def test_the_submitted_value_is_ignored_even_on_an_entry_carrying_one() -> None:
    """**Asserted at the unit, because the wire cannot reach this case.**

    Every entry Pydantic produces carries an `input` key holding the value the caller
    submitted. `_reasons_given` reads `msg` and must never read `input` -- and the only
    reliable way to check that is to hand it an entry where the two differ, because a
    body that triggers a validator this repo wrote AND carries a credential in the same
    field is a body no route here accepts.

    An earlier version of this test went over the wire with a secret in two fields. It
    passed and it was worthless: those fields fail a built-in uuid check, so no reason
    was published and neither branch ran. Measured by breaking it -- appending `input`
    to the published message left all eighteen assertions green.
    """
    carrying = [
        {
            "type": "value_error",
            "loc": ("body", "tools", 0),
            "msg": "Value error, tool ask_question: argument repoNmae is undeclared",
            "input": {"credential": A_SECRET},
        }
    ]

    given = _reasons_given(carrying)

    assert given == ["tool ask_question: argument repoNmae is undeclared"]
    assert A_SECRET not in " ".join(given), (
        "the submitted value reached the published reason, so a credential sent to a "
        "field a validator rejects would land in every log holding a response body"
    )


async def test_a_value_is_not_echoed_on_the_path_the_wire_can_reach(
    client: AsyncClient,
) -> None:
    """The over-the-wire half, honest about which half it is.

    This body fails built-in checks, so it exercises the `fields` path and not the
    reasons path. Worth asserting on its own -- it is the commonest refusal this API
    sends -- and it is not evidence about `input`, which the unit test above covers.
    """
    refused = await client.post(
        "/v1/sessions", json={"definition_id": A_SECRET, "environment_id": A_SECRET}
    )

    assert A_SECRET not in refused.text
    assert "reasons" not in refused.json()["error"]["detail"], (
        "a built-in check published a reason, so this test no longer covers the path "
        "its docstring claims"
    )
