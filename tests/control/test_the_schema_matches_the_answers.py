"""The published document promises only what the app answers.

**Nothing else in this suite can catch a defect here.** Every other test exercises the
app; a client is generated from `/openapi.json`. So a document that disagrees with the
app is a defect in every SDK built from it and green everywhere in this repository.

The concrete case: FastAPI attaches a `422` carrying its own `HTTPValidationError` to
every operation that declares a body or a typed parameter. That was right until the
refusal envelope landed. Now `RequestValidationError` is answered as `400` with
`PublicErrorEnvelope`, and no member of the published `ErrorCode` set maps to `422` at
all -- so the framework's automatic entry described a response this app cannot produce,
on 28 operations, in a shape nothing here sends. A generated client was wrong about the
commonest refusal on the API: wrong status, wrong body, wrong field to read the code out
of.

The assertions are relational rather than absolute. None says "28" or names a route: the
document is compared against `STATUS_FOR`, which is the app's own table, so both move
together and neither can be updated alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.schema import _FRAMEWORK_SCHEMAS
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope


class Unused:
    """Every port raises. Building the document reads routes, never a store."""

    def __getattr__(self, name: str) -> Any:
        async def refuse(*args: object, **kwargs: object) -> Any:
            raise AssertionError(f"building the schema reached {name}")

        return refuse


def _document() -> dict[str, Any]:
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
    document: dict[str, Any] = create_app(platform).openapi()
    return document


def _operations() -> list[tuple[str, str, dict[str, Any]]]:
    found = [
        (verb.upper(), path, operation)
        for path, operations in _document()["paths"].items()
        for verb, operation in operations.items()
        if isinstance(operation, dict)
    ]
    assert found, "no operations in the document; every test here would pass vacuously"
    return found


ANSWERABLE = frozenset(str(status) for status in STATUS_FOR.values())
"""Every status some published code carries, as OpenAPI spells statuses.

Read off the app's own table rather than listed, so a code whose status changes moves
this set with it and the two cannot be updated separately.
"""


def test_no_operation_promises_a_status_no_published_code_can_carry() -> None:
    """**The assertion that would have caught the 422.**

    A documented status the app cannot produce is worse than an undocumented one: a
    client author writes a branch for it, the branch is never taken, and the refusal
    that does arrive falls through to whatever the generator emitted for the unexpected
    case.

    2xx and 3xx are exempt: those are successes, which no `ErrorCode` describes.
    """
    promised_but_impossible: dict[str, list[str]] = {}
    for verb, path, operation in _operations():
        for status in operation.get("responses", {}):
            if not status.isdigit() or int(status) < 400:
                continue
            if status not in ANSWERABLE:
                promised_but_impossible.setdefault(status, []).append(f"{verb} {path}")

    assert promised_but_impossible == {}, (
        "the document promises statuses no published code carries: "
        f"{ {s: len(v) for s, v in promised_but_impossible.items()} }. A client "
        "generated from this branches on a response that never arrives."
    )


def test_every_documented_refusal_carries_this_apis_envelope() -> None:
    """One shape on the wire has to be one shape in the document.

    A refusal documented with the framework's own model tells a client author to read
    `detail` where this API puts `error`, so the code is unreachable from the shape the
    document describes.
    """
    envelope = f"#/components/schemas/{PublicErrorEnvelope.__name__}"
    wrong: list[str] = []
    for verb, path, operation in _operations():
        for status, response in operation.get("responses", {}).items():
            if not status.isdigit() or int(status) < 400:
                continue
            if envelope not in str(response):
                wrong.append(f"{verb} {path} -> {status}")

    assert wrong == [], (
        f"these refusals are documented in some other shape: {wrong}. Every refusal "
        "this API sends is a PublicErrorEnvelope."
    )


@pytest.mark.parametrize("name", _FRAMEWORK_SCHEMAS)
def test_the_frameworks_validation_schemas_are_gone(name: str) -> None:
    """Both of them, and the second is the one that hid.

    They reference each other -- `HTTPValidationError` holds an array of
    `ValidationError` -- so a single pass over a reference scan taken once at the start
    removes the first and keeps the second forever, because what kept the second alive
    was inside the definition just deleted. That is what the first version of the
    remover did, and this parametrised case is what said so.
    """
    assert name not in _document()["components"]["schemas"]


def test_nothing_still_points_at_a_schema_that_was_removed() -> None:
    """A dangling `$ref` stops a client generator rather than warning it.

    So removal is conditional on nothing referencing the name, and this asserts the
    condition held -- a document that removed a schema something still names is worse
    than one carrying an unused definition.
    """
    document = _document()
    defined = set(document["components"]["schemas"])
    text = str(document)

    dangling = [
        name
        for name in _FRAMEWORK_SCHEMAS
        if name not in defined and f"#/components/schemas/{name}" in text
    ]

    assert dangling == [], f"these are referenced but no longer defined: {dangling}"


def test_the_malformed_refusal_is_documented_at_the_status_it_arrives_at() -> None:
    """The positive half.

    Removing a wrong promise and adding none would leave the commonest refusal on this
    API undocumented, which is a smaller defect and still one.
    """
    expected = str(STATUS_FOR[ErrorCode.REQUEST_INVALID])
    taking_a_body = [
        f"{verb} {path}"
        for verb, path, operation in _operations()
        if "requestBody" in operation
    ]
    assert taking_a_body, "no operation declares a body; this test proves nothing"

    missing = [
        f"{verb} {path}"
        for verb, path, operation in _operations()
        if "requestBody" in operation and expected not in operation.get("responses", {})
    ]

    assert missing == [], (
        f"these take a body and do not document a {expected}: {missing}. Every one of "
        "them can be refused by the validation handler."
    )


def test_the_documented_status_follows_the_table_rather_than_a_literal() -> None:
    """A guard on this file, not on the app.

    Every assertion above reads `STATUS_FOR`. If `REQUEST_INVALID` moves, they all move
    with it and none of them notices -- which is correct for them and would leave the
    document free to keep the old status. This is the one place that pins the two
    together: the document must offer the table's status for a body it can reject.
    """
    document = _document()
    sessions = document["paths"]["/v1/sessions"]["post"]["responses"]
    assert str(STATUS_FOR[ErrorCode.REQUEST_INVALID]) in sessions
    assert "422" not in sessions, (
        "the framework's status is back on an operation, so the correction is no "
        "longer applied to every route"
    )
