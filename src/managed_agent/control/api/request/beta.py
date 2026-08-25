"""The `anthropic-beta` header: which wire shape a caller is asking for.

A schema version in a header exists so a caller can pin the shape of the responses it
parses, and so a server can serve two shapes at once while callers migrate between
them. This build serves exactly one shape, which makes both halves of that useful and
neither of them free.

**A value this build does not serve is refused.** Answering a caller that pinned a shape
with a different shape is the whole failure the header exists to prevent, and it fails
silently: the response parses far enough to look like an answer and then differs
somewhere the caller did not check.

**An absent header is served the one shape there is.** That is safe only while there is
one, and it stops being safe the moment a second is added -- at which point a request
naming no version is ambiguous and has to be refused rather than guessed at. That
transition is not left to be remembered: `SERVED` is a set, and
`test_absence_stops_being_answerable_once_a_second_version_exists` fails when a second
member is added while absence is still accepted. The guard is the reason absence is
allowed at all.

Nothing here authenticates or authorises. The header names a schema, not a permission,
and a caller that sends the right one is no more trusted for it.

Their surface uses a *second* header value for memory endpoints
(`agent-memory-2026-07-22`). This platform serves no memory endpoints -- declined at the
Plan gate and excluded by instruction -- so that value is not served, and a caller
sending it is refused by the same rule as any other value: it names a shape this build
does not answer in.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Set
from typing import Final

from fastapi import FastAPI, Request, Response

from managed_agent.control.api.refusals import Refusal, refuse
from managed_agent.core.errors import ErrorCode

BETA_HEADER: Final = "anthropic-beta"

MANAGED_AGENTS_2026_04_01: Final = "managed-agents-2026-04-01"

SERVED: Final[Set[str]] = frozenset({MANAGED_AGENTS_2026_04_01})
"""Every schema version this build answers in.

A set with one member rather than a scalar, deliberately. The transition that matters is
one member to two, and a scalar makes that transition an edit to a type instead of an
edit to a value -- so the guard test cannot be written against it, and the day a second
shape ships nobody is told that absence has become ambiguous.
"""


def _refuse_a_shape_we_do_not_answer_in(asked_for: str) -> Refusal:
    """The refusal for a version this build does not serve.

    The message names what was asked for and what is served, because the caller's next
    move is to change its own pin and it cannot do that from "unsupported" alone.
    """
    return Refusal(
        ErrorCode.REQUEST_BETA_UNSUPPORTED,
        f"this build does not serve schema version {asked_for!r}; it serves "
        + ", ".join(sorted(SERVED)),
        asked_for=asked_for,
        served=", ".join(sorted(SERVED)),
    )


def version_asked_for(request: Request) -> str:
    """The schema version this request is to be answered in.

    Raises `Refusal` when the header names a version this build does not serve. An
    absent header resolves to the one served version -- see the module docstring for why
    that is safe only while there is one, and for the test that fails when it stops
    being.

    A header carrying several comma-separated values is refused unless every one of them
    is served. Serving the subset we recognise and ignoring the rest would answer a
    caller that asked for two shapes with one, which is the same silent mismatch as
    answering the wrong single shape.
    """
    raw = request.headers.get(BETA_HEADER)
    if raw is None:
        (only,) = SERVED if len(SERVED) == 1 else (None,)
        if only is None:  # pragma: no cover -- the guard test owns this transition
            raise Refusal(
                ErrorCode.REQUEST_BETA_UNSUPPORTED,
                f"this build serves more than one schema version, so {BETA_HEADER} "
                "is required; it serves " + ", ".join(sorted(SERVED)),
                served=", ".join(sorted(SERVED)),
            )
        return only
    asked = [value.strip() for value in raw.split(",") if value.strip()]
    if not asked:
        raise _refuse_a_shape_we_do_not_answer_in(raw)
    for value in asked:
        if value not in SERVED:
            raise _refuse_a_shape_we_do_not_answer_in(value)
    return asked[0]


def install_beta_header(app: FastAPI) -> None:
    """Check the header on every request, and say on every response what was served.

    Middleware rather than a per-route dependency for the same reason the envelope is
    middleware: a check a route has to remember is a check some route will not have, and
    a schema version that applies to some routes is not a schema version.

    The response carries the served version back. A caller that sent no header learns
    which shape it got without having to know what this build's default is, and a caller
    that sent one gets it echoed, which is what lets a proxy in between cache on it.
    """

    @app.middleware("http")
    async def _answer_in_one_shape(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Caught here rather than left to `@app.exception_handler(Refusal)`, because
        # that handler cannot see it. Measured: Starlette's exception handlers live in
        # `ExceptionMiddleware`, which sits INSIDE every user middleware, so an
        # exception raised in one propagates outward past it and leaves through
        # `ServerErrorMiddleware` -- the request fails with a traceback and no envelope
        # at all, not with the 500 envelope you would expect. The shape is still built
        # once: `refuse` is the same function every route calls, so this is one more
        # caller of the single envelope rather than a second envelope.
        #
        # This middleware must be installed BEFORE `install_request_envelope`, which
        # makes it the INNER of the two -- last registered is outermost. That is what
        # puts the request id on the ContextVar by the time `refuse` reads it. Installed
        # the other way round, every refusal from here would be attributed to
        # `req_unattributed`.
        try:
            served = version_asked_for(request)
        except Refusal as refusal:
            return refuse(refusal.code, refusal.message, **refusal.detail)
        response = await call_next(request)
        response.headers[BETA_HEADER] = served
        return response
