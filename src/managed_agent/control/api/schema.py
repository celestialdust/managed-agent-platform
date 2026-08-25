"""The published OpenAPI document, corrected where the framework guesses wrong.

A generated client is written from this document and not from the code, so a promise
here that the app does not keep is a defect in every SDK built from it -- and it is the
kind nothing in this repository would notice, because the tests exercise the app.

**FastAPI adds a `422` with its own `HTTPValidationError` to every operation that
declares a body or a typed parameter.** That was true of this API until the refusal
envelope landed: `RequestValidationError` is now answered as `400` carrying
`PublicErrorEnvelope`, and no member of the published `ErrorCode` set maps to `422` at
all. So the framework's automatic entry describes a response this app cannot produce, on
28 of its operations, and it describes it in a shape nothing here sends. A client
generated from it is wrong about the commonest refusal on the API: wrong status, wrong
body, and wrong about which field to read the code out of.

Corrected here, once, rather than by declaring `responses=` on every route. A schema
correction a route has to remember is one some route will not have -- and the next route
added would inherit the framework's guess again with nothing to say so. This is the same
reason the request-id envelope and the beta-header check are middleware.

What this does NOT do is invent responses. It removes a promise the app cannot keep and
substitutes the one it does keep for the same cause; an operation that already declares
its own `400` keeps it, because that declaration names a specific code and is more
informative than the generic one.
"""

from __future__ import annotations

from typing import Any, Final

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope

_FRAMEWORK_VALIDATION_STATUS: Final = "422"
"""What FastAPI attaches by itself.

A string because OpenAPI keys its responses map by string.
"""

_MALFORMED_STATUS: Final = str(STATUS_FOR[ErrorCode.REQUEST_INVALID])
"""What this app actually answers, read from the table rather than typed again.

Derived so that a future change to `REQUEST_INVALID`'s status moves the schema with it.
Typed here as a literal, the document and the app would disagree the day it moved, and
the disagreement would be invisible from either side.
"""

_ENVELOPE_NAME: Final = PublicErrorEnvelope.__name__

_FRAMEWORK_SCHEMAS: Final = ("HTTPValidationError", "ValidationError")
"""Component schemas that exist only to describe the response being removed."""

_DESCRIPTION: Final = (
    "The request could not be accepted as written. `error.detail` names the "
    "rejected fields, and carries the validator's own reason where this API "
    "wrote one."
)


def _envelope_response() -> dict[str, Any]:
    return {
        "description": _DESCRIPTION,
        "content": {
            "application/json": {
                "schema": {"$ref": f"#/components/schemas/{_ENVELOPE_NAME}"}
            }
        },
    }


def _correct_the_malformed_response(document: dict[str, Any]) -> dict[str, Any]:
    """Replace the framework's invented 422 with the refusal this app sends.

    An operation that already declares its own `400` keeps it: that declaration was
    written by a route that knows which code it sends, and overwriting it with the
    generic description would lose information. The `422` goes either way, because no
    published code carries that status and a document offering both would leave a client
    author to guess which one arrives.
    """
    for operations in document.get("paths", {}).values():
        for operation in operations.values():
            if not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            if responses.pop(_FRAMEWORK_VALIDATION_STATUS, None) is None:
                continue
            responses.setdefault(_MALFORMED_STATUS, _envelope_response())
    return document


def _drop_schemas_nothing_references(document: dict[str, Any]) -> dict[str, Any]:
    """Remove the framework's validation schemas once nothing points at them.

    Conditional on a reference scan rather than unconditional, and the condition is
    the point: these names are FastAPI's, so a future version of it may reference them
    from somewhere this module does not know about. Deleting a schema a `$ref` still
    names produces a document that fails validation -- worse than an unused
    definition, because a client generator stops rather than warns.

    Repeated until nothing more can go, because these schemas reference EACH OTHER:
    `HTTPValidationError` holds an array of `ValidationError`. A single pass over a
    scan taken once at the start deletes the first and keeps the second forever, since
    the reference that kept it alive was inside the definition just removed. Measured
    -- that is exactly what the first version of this did.
    """
    schemas = document.get("components", {}).get("schemas", {})
    removing = True
    while removing:
        removing = False
        for name in _FRAMEWORK_SCHEMAS:
            if name not in schemas:
                continue
            elsewhere = {key: value for key, value in schemas.items() if key != name}
            pointed_at = f"#/components/schemas/{name}"
            if pointed_at in str(document.get("paths", {})) or pointed_at in str(
                elsewhere
            ):
                continue
            del schemas[name]
            removing = True
    return document


def publish_the_schema_the_app_answers_with(app: FastAPI) -> None:
    """Install the corrected document as this app's `openapi()`.

    Cached on `app.openapi_schema` the way FastAPI's own implementation caches, so the
    corrections are applied once rather than per request to `/openapi.json`.
    """

    def corrected() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        document = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        document = _correct_the_malformed_response(document)
        document = _drop_schemas_nothing_references(document)
        app.openapi_schema = document
        return document

    app.openapi = corrected  # type: ignore[method-assign]
