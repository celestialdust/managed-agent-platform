"""One refusal, one request id, and the two handlers that leave no response unenveloped.

**Why this module exists at all.** Thirteen route files each carried a private `_refuse`
with an identical body, each writing the same envelope from the same two lookups. That
is thirteen places to change when the envelope changes, and thirteen chances to change
twelve of them -- and the envelope did change, on 2026-08-24, from this platform's own
`{code, message, detail}` to the nested shape a consumer's SDK reads.

**Why the nesting rather than one shape or the other.** `ErrorCode` has 24 members and
says what exactly was wrong; the published `type` has 8 and says what kind of thing was
wrong. A client generated against the Managed Agents documentation can classify any
refusal from here without knowing this platform's vocabulary, and a client that does
know it still gets `error.code` to branch on. Publishing only the coarse class would
delete every distinction a caller can act on -- `session.not_accepting_turns` and
`tool.name_conflict` are both `invalid_request_error` and want opposite responses.

**Why the handlers are here and not only the helper.** A refusal this platform *decides*
to return was already enveloped by the thirteen copies. Two refusals it does not decide
were not, and they were the ones a consumer met first:

- `RequestValidationError`, which FastAPI answers itself with `422` and a body of
  `{"detail": [...]}`. Nothing in this codebase produced that body and nothing could
  intercept it, so the most common refusal on any API -- a malformed request -- was the
  one shaped least like every other.
- An unhandled exception, which Starlette answers with a bare `500` and the string
  `Internal Server Error`. A caller retrying on that has no request id to report.

Registering both here keeps the guarantee stateable in one sentence: **every non-2xx
response from this app carries this envelope and a request id.** A guarantee holding for
the refusals we author and not the ones the framework authors is not a guarantee, and it
fails on exactly the inputs a new integrator sends.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from managed_agent.core.errors import STATUS_FOR, ErrorCode, public_envelope

REQUEST_ID_HEADER: Final = "X-Request-Id"
"""The header every response carries the id on, success or refusal.

On the success path too, and that is the point rather than symmetry: a caller reporting
"the Session I created never ran" is describing a 201, and an id that appeared only on
failures could not be quoted for it.
"""

_REQUEST_ID_PREFIX: Final = "req_"
_UNKNOWN_REQUEST: Final = "req_unattributed"
"""Used only when a refusal is built outside a request's lifetime.

Never expected on the wire. It exists so `refuse()` stays a pure function of its
arguments -- a missing id is a wrong id in a log lookup, and raising here would turn a
refusal into a 500 while reporting nothing about the original fault.
"""


def new_request_id() -> str:
    """A fresh opaque id for one request.

    `uuid4().hex` and not a counter or a timestamp: an id a caller can quote must not
    also tell them how many requests this platform has served, and two processes behind
    one Service must not mint the same id.
    """
    return _REQUEST_ID_PREFIX + uuid.uuid4().hex


_CURRENT_REQUEST_ID: ContextVar[str] = ContextVar(
    "map_request_id", default=_UNKNOWN_REQUEST
)
"""The id of the request being served on this task.

**A ContextVar and not a parameter, and the reason is the diff it avoids.** Thirteen
route files refuse, at roughly sixty call sites, and almost none of them declare a
`Request`: they take the path and body parameters they actually use. Passing the id
explicitly would mean adding a parameter to every route that can refuse -- changing
signatures whose shape is otherwise the documentation of what that route reads -- so
that a value none of them inspect could be carried through.

Safe because the unit of isolation matches the unit of work: the server runs each
request in its own task, a `ContextVar` set inside that task is invisible to every
other, and `contextvars` propagates into tasks a handler spawns. What it must never
become is a channel for anything else. One value, written in one place, read in one
place.

The default is deliberate rather than a fallback nobody meant. A route exercised
directly by a unit test has no middleware and therefore no id, and a refusal that raised
`LookupError` while reporting a refusal would replace a legible 400 with an illegible
500 -- turning the absence of a diagnostic aid into a fault worse than the one it
described.
"""


def request_id_of(request: Request) -> str:
    """The id assigned to this request, read off the request itself.

    Kept for the handlers below, which are handed a `Request` and should not depend on
    the ambient value when the explicit one is in front of them.
    """
    found = getattr(request.state, "request_id", None)
    return found if isinstance(found, str) and found else current_request_id()


def current_request_id() -> str:
    """The id for the request on this task, or the unattributed marker."""
    return _CURRENT_REQUEST_ID.get()


def refuse(code: ErrorCode, message: str, **detail: str | int) -> JSONResponse:
    """One refusal, at the status and class the published set assigns that code.

    Neither the status nor the class is a parameter. Both are looked up, so no route can
    answer with a status the contract does not give the code it is naming, and no route
    can pair a code with a class that contradicts it.

    This signature is exactly the one the thirteen private copies had, which is what let
    them be replaced by an import rather than a rewrite of every call site.
    """
    envelope = public_envelope(
        code=code,
        message=message,
        request_id=current_request_id(),
        detail=dict(detail),
    )
    return JSONResponse(
        status_code=STATUS_FOR[code],
        content=envelope.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: envelope.request_id},
    )


class Refusal(Exception):
    """A refusal raised rather than returned, for code that cannot return a response.

    **A raisable twin of `refuse`, because a dependency cannot return a response.** The
    tenant-header check, the audit principal check and the cursor parser all run as
    FastAPI dependencies: they decide a request is unacceptable before any route body
    executes, and the only way out of a dependency is to raise. They previously raised
    `HTTPException` with a `detail` dict that happened to carry a `code` key, which
    FastAPI renders as `{"detail": {...}}` -- a third body shape on the same API, and
    the one a caller met most often, since a missing tenant header is the first mistake
    anyone makes.

    Carrying an `ErrorCode` rather than a status and a string is what makes it
    equivalent to `refuse`: the status and the published class are looked up from the
    code in both paths, so a dependency and a route refusing for the same reason cannot
    answer differently.
    """

    def __init__(self, code: ErrorCode, message: str, **detail: str | int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail: dict[str, str | int] = dict(detail)


def _enveloped(
    request: Request, code: ErrorCode, message: str, **detail: str | int
) -> JSONResponse:
    """`refuse` for the two exception handlers, which are given the `Request`.

    Separate from `refuse` because an exception handler can run after the middleware's
    `finally` has reset the ContextVar, so the ambient value is not trustworthy there
    and `request.state` is.
    """
    envelope = public_envelope(
        code=code,
        message=message,
        request_id=request_id_of(request),
        detail=dict(detail),
    )
    return JSONResponse(
        status_code=STATUS_FOR[code],
        content=envelope.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: envelope.request_id},
    )


_MOST_FIELDS_NAMED: Final = 10
"""How many field names one refusal will list. See `_fields_named`."""


def _fields_named(problems: Sequence[Any]) -> list[str]:
    """Which fields a validation failure named, as a bounded, sorted, de-duplicated
    list.

    The leading segment of a Pydantic `loc` is the request part -- `body`, `query`,
    `path` -- and it is dropped: the caller named the field, not the part, and
    `body.environment_id` reads as though `body` were a field. A `loc` of just that
    segment (a body that is not an object at all) keeps it, because there is no
    field to name and "body" is the true answer.

    Indices are kept as written, so `skills.0.name` says which element. Sorted so
    two requests failing on the same fields produce the same string, and de-
    duplicated because one field failing two rules is one field to fix.
    """
    paths: set[str] = set()
    for problem in problems:
        location = tuple(str(part) for part in problem.get("loc", ()))
        if len(location) > 1:
            location = location[1:]
        if location:
            paths.add(".".join(location))
    listed = sorted(paths)
    if len(listed) <= _MOST_FIELDS_NAMED:
        return listed
    remaining = len(listed) - _MOST_FIELDS_NAMED
    return [*listed[:_MOST_FIELDS_NAMED], f"and {remaining} more"]


_VALIDATOR_RAISED: Final = "value_error"
"""Pydantic's `type` for an error a validator in this codebase raised."""

_PYDANTIC_PREFIX: Final = "Value error, "

_MOST_REASONS_GIVEN: Final = 5
"""How many validator messages one refusal carries. Fewer than the field cap: a
message is a sentence, and five sentences is already a caller reading rather than
scanning."""


def _reasons_given(problems: Sequence[Any]) -> list[str]:
    """Why each rejection happened, for the rejections whose reason this codebase wrote.

    Pydantic tags an error raised by a `@model_validator` or `@field_validator` as
    `value_error`, and its `msg` is the string that validator raised -- authored here,
    in this repository, by code that chose what to name. Those are published. Every
    other `type` is a built-in check (`missing`, `extra_forbidden`, `string_type`,
    `int_parsing`) whose `msg` is Pydantic's own generic sentence, carrying no
    information a field name does not already give.

    **This is the difference between a refusal a caller can act on and one it
    cannot.** A tool registration binding argument `repoNmae` when the tool declares
    `repoName` is refused, and the tenant's next move is to find a one-character typo.
    A response naming only the field `tools.0` leaves them comparing two identical-
    looking strings across a catalogue. The validator's own message names the tool and
    the argument, which is why it exists.

    `input` is never published, from any entry, and that is the line that matters: it
    holds the value the caller SUBMITTED, so a wrong-typed credential field would come
    back in full. Note what this means about the change as a whole -- before the
    envelope existed this handler forwarded the entire error list, `msg` and `input`
    and `type` and `ctx` together. Publishing `msg` alone is strictly less than what
    shipped then, and the part that was actually dangerous stays withheld.

    Pydantic's `Value error, ` prefix is stripped. It is the wrapper's, not the
    validator's, and a caller reading a message that starts by naming the framework
    learns something about our implementation instead of about their request.
    """
    said: list[str] = []
    for problem in problems:
        if problem.get("type") != _VALIDATOR_RAISED:
            continue
        message = str(problem.get("msg", ""))
        if message.startswith(_PYDANTIC_PREFIX):
            message = message[len(_PYDANTIC_PREFIX) :]
        if message and message not in said:
            said.append(message)
    if len(said) <= _MOST_REASONS_GIVEN:
        return said
    remaining = len(said) - _MOST_REASONS_GIVEN
    return [*said[:_MOST_REASONS_GIVEN], f"and {remaining} more"]


def install_request_envelope(app: FastAPI) -> None:
    """Mint an id for every request, and envelope the two refusals we do not author.

    Called once from `app.py`. Middleware rather than a dependency because a dependency
    does not run for a request that fails validation before the route is entered, which
    is precisely the case the validation handler below exists for.
    """

    @app.middleware("http")
    async def _carry_a_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        assigned = new_request_id()
        request.state.request_id = assigned
        token = _CURRENT_REQUEST_ID.set(assigned)
        try:
            response = await call_next(request)
        finally:
            _CURRENT_REQUEST_ID.reset(token)
        response.headers[REQUEST_ID_HEADER] = assigned
        return response

    @app.exception_handler(RequestValidationError)
    async def _malformed(request: Request, exc: RequestValidationError) -> Response:
        """A body or parameter the route could not accept, as `request.invalid`.

        400 rather than FastAPI's 422, and the reason is not tidiness: an SDK
        written against a 400-for-malformed contract treats an unexpected 422 as a
        transport fault rather than a refusal it can read a code out of.

        **The field names go out; nothing else from the framework's diagnostics
        does.** Which field was rejected is the only part a caller can act on, and
        an earlier version of this handler sent only a count -- which told a caller
        that one thing was wrong and left it to guess which, on the most common
        refusal this API produces. What is withheld is withheld for a reason and not
        for tidiness: Pydantic's `msg`, `type` and `ctx` would make this envelope's
        `detail` a published copy of its diagnostic shape, and its `input` echoes
        the value submitted -- which on a credential or a token field would write a
        secret into every log that holds a response body. A `loc` path carries field
        names and array indices only, so it is the one part that is safe to publish.

        Capped, and the count is sent alongside so a caller can see the list was
        cut. An unbounded join is a caller-controlled string in a response: a body
        with a thousand bad fields would otherwise return a thousand field names.
        """
        problems = exc.errors()
        detail: dict[str, str | int] = {
            "problem_count": len(problems),
            "fields": ", ".join(_fields_named(problems)),
        }
        # Absent rather than empty when no validator spoke. A key present and blank
        # reads as "there was no reason", which is a different claim from "every
        # rejection here was a built-in check whose reason is its field name".
        said = _reasons_given(problems)
        if said:
            detail["reasons"] = "; ".join(said)
        return _enveloped(
            request,
            ErrorCode.REQUEST_INVALID,
            "the request could not be accepted as written",
            **detail,
        )

    @app.exception_handler(Refusal)
    async def _raised(request: Request, exc: Exception) -> Response:
        """A refusal a dependency raised, rendered as if a route had returned it.

        Raised by a dependency, so it never reached a route body. Rendered here through
        exactly the same lookup a route's `refuse()` uses, which is the point: the
        caller cannot tell whether a refusal was decided before or inside the handler,
        and should not be able to.
        """
        assert isinstance(exc, Refusal)
        return _enveloped(request, exc.code, exc.message, **exc.detail)

    @app.exception_handler(StarletteHTTPException)
    async def _router_refusal(request: Request, exc: Exception) -> Response:
        """An unknown path or a method the route does not accept.

        The router's own refusals, which no code here raises: an unknown path and a
        wrong verb on a known one. Starlette answers both with `{"detail": "Not
        Found"}`, so before this handler existed the two most likely first requests from
        a new integrator -- a typo'd path, and a GET where a POST was wanted -- returned
        a body shaped unlike every documented refusal.

        Mapped onto codes rather than passed through, so `detail` never carries the
        framework's own wording. A 401 raised elsewhere as a plain `HTTPException` lands
        on `platform.internal` rather than a guessed class, because inventing an
        authentication refusal from a bare status is how a caller ends up told to re-
        authenticate for a fault that had nothing to do with credentials.
        """
        assert isinstance(exc, StarletteHTTPException)
        by_status = {
            404: ErrorCode.REQUEST_ROUTE_NOT_FOUND,
            405: ErrorCode.REQUEST_METHOD_NOT_ALLOWED,
        }
        code = by_status.get(exc.status_code, ErrorCode.INTERNAL)
        message = (
            "no route serves this path"
            if code is ErrorCode.REQUEST_ROUTE_NOT_FOUND
            else "this route does not accept that method"
            if code is ErrorCode.REQUEST_METHOD_NOT_ALLOWED
            else "the request could not be completed"
        )
        return _enveloped(request, code, message)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> Response:
        """Anything that escaped a route, as `platform.internal` with an id to quote.

        The exception is not described in the response, on purpose: a stack detail, a
        module path or a driver's message is exactly what the closed code set exists to
        keep off the wire. The request id is what makes this actionable -- it is the one
        string a caller can send us that turns "it broke" into a log lookup.
        """
        return _enveloped(
            request,
            ErrorCode.INTERNAL,
            "the request could not be completed",
        )
