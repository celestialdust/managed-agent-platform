"""Serving a Session back the workspace tree its own earlier Turns left behind.

The `working` lane is written at the end of every completed Turn and, until this module,
was read by nothing: a Session's workspace survived its pod and no pod was ever given
one back. This is the read half of that round trip, and it lives on the Tool Gateway
rather than on the Session shim because the shim mounts the workspace `subPath: files`
-- a narrowing that keeps the pod's one outward-facing process from writing over what
the agent later reads, and which is not worth widening for an operation that happens
once per placement (ADR-030).

**Who is calling is decided by the token and by nothing else.** Neither route takes a
tenant, a Session, or anything else naming whose lane to read: both compose their key
from the `SessionContext` the token middleware verified. A pod holding a token for
Session A therefore has no arrangement of inputs -- no path, no query, no header --
that reads Session B's lane, because the two identifying segments of the key are never
taken from the request. That is the whole security argument for exposing this to a
confined agent, which reaches the Gateway with the same token its pod does.

**A path that will not parse is absent, not malformed.** `parse_relative_path` refuses
`..`, an empty segment, a leading separator and a trailing one, so a path it rejects
composes to no key in this lane and the lane therefore holds nothing at it. Answering
404 is the true answer and keeps the parser un-widened; widening it to serve a restore
would trade the guarantee that a composed key cannot leave this Session's lane for the
convenience of one caller.

**The ceiling is the client's, not this route's.** ADR-030 bounds a restore at 2048
objects and 256 MiB, and the init container that fetches the lane is what enforces it.
This route reports the lane honestly, however large it has grown -- a second
enforcement point here would be a second number to keep in step with
`WORKING_COUNT_LIMIT`, and the day they disagreed the lane would list objects no
restore would fetch while nothing said which bound had refused.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Final

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import request_response
from starlette.types import ASGIApp

from managed_agent.core.session.session_token import SessionContext
from managed_agent.core.vfs.session_vfs import (
    WORKING,
    MutableFile,
    SessionFiles,
    VfsPathInvalid,
)

LANE_LISTING_PATH: Final[str] = "/v1/session/working-lane"
"""Where the whole listing of the calling Session's working lane is served."""

LANE_OBJECT_PATH: Final[str] = "/v1/session/working-lane/{path:path}"
"""Where one object of that lane is served, addressed by its lane-relative path.

`{path:path}` rather than `{path}` because a lane path carries separators: the whole
point of a lane keyed by path is that a workspace arrives with its shape intact, and a
single-segment parameter would make `src/main.py` unaddressable.
"""

_log = logging.getLogger(__name__)


def working_lane_endpoints(
    files: SessionFiles, calling_session: Callable[[], SessionContext]
) -> Sequence[tuple[str, ASGIApp]]:
    """The two working-lane paths and the ASGI app each one serves, unwrapped.

    Unwrapped deliberately: the caller puts the token middleware around each of these,
    so the Gateway has one place where what is behind the token check is decided rather
    than two modules each asserting they wrapped themselves.

    `calling_session` is a callable rather than a `SessionContext` because these
    endpoints outlive any one request -- they are built once at start-up and read the
    caller per call. Passing the context itself would bind every response to whichever
    pod happened to be first through the door.
    """

    async def serve_listing(_request: Request) -> Response:
        """Every object in the calling Session's lane, with the size of each.

        Sizes and not digests, because that is what `list_lane` answers: filling a
        digest field here would cost a download of the whole lane to describe it.
        The client needs the size anyway -- it is what a restore checks its byte
        budget against before it starts fetching.
        """
        who = calling_session()
        entries = await files.list_lane(who.tenant_id, who.session_id, WORKING)
        if not entries:
            # At WARNING because an empty lane and a lane this process could not read
            # produce the same well-formed `{"objects": []}`, and downstream there is
            # nothing left to tell them apart: the restore materializes an empty
            # workspace and the agent reports a file missing with no line anywhere
            # saying why. This deployment's root logger sits at WARNING with no
            # handler, so a line below that level would exist only in this source.
            _log.warning(
                "the working lane of session %s holds no objects, so a restore from"
                " this listing materializes an empty workspace",
                who.session_id,
            )
        return JSONResponse(
            {
                "objects": [
                    {"path": entry.relative, "size": entry.byte_length}
                    for entry in entries
                ]
            }
        )

    async def serve_object(request: Request) -> Response:
        """One object's bytes, or the same refusal a path holding nothing gets.

        `MutableFile` parses the path at construction, so holding one is proof the
        composed key is inside this Session's own lane. A path it refuses is answered
        as absent rather than as malformed: such a path composes to no key here, so
        the lane really does hold nothing at it, and saying so is what lets the parser
        stay as narrow as the write side needs it.
        """
        who = calling_session()
        try:
            target = MutableFile(
                tenant_id=who.tenant_id,
                session_id=who.session_id,
                lane=WORKING,
                relative=str(request.path_params["path"]),
            )
        except VfsPathInvalid:
            return _absent()
        body = await files.read(target)
        if body is None:
            return _absent()
        return Response(content=body, media_type="application/octet-stream")

    return (
        (LANE_LISTING_PATH, request_response(serve_listing)),
        (LANE_OBJECT_PATH, request_response(serve_object)),
    )


def _absent() -> Response:
    """The one refusal this surface gives, for both ways a path can hold nothing.

    Deliberately not an `ErrorEnvelope`, for the reason the token refusal beside it is
    not one: that vocabulary is published to tenants, and this response is read by a
    Session's own pod. A code no tenant can ever observe does not belong in a set the
    platform commits to.
    """
    return JSONResponse(
        {"error": "no object at that path in this session's working lane"},
        status_code=404,
    )
