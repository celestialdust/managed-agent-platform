"""Downloading one file a Session's agent produced, by the path the agent wrote it at.

**This route is what makes a produced file reachable, and before it there was one that
did the same job differently.** Ship-out used to mint an *upload* for each produced file
and announce its identifier, so a tenant downloaded their agent's work from
`GET /v1/files/{id}/content` -- the same door their own uploads come back through. Two
things were wrong with that and only one of them is cosmetic. The cosmetic one: a
produced file was indistinguishable from an uploaded one in a tenant's file list, so
which Turn made what was a correlation the tenant had to do. The load-bearing one: an
upload is keyed by a *filename*, and a filename cannot contain a separator -- so
`out/report/fig1.png` had nowhere to go, and the ship-out path simply never offered it.

A Session's `artifacts` lane is keyed by a path instead, so a deliverable that is a tree
-- a report with its figures, a page with its assets -- arrives with its shape intact.
This is the door onto that lane.

**Cross-tenant isolation here is the key's shape rather than a check this code makes.**
The object key is composed as `sessions/<tenant>/<session>/artifacts/<path>` from the
tenant on the *request* and the Session id in the *path*, so a caller naming another
tenant's Session composes a key under their own tenant segment, which holds nothing, and
is answered "no artifact there". There is no arrangement of inputs that reads another
tenant's object, because the tenant component is never taken from the caller's claim
about which Session this is. That is `lane_prefix`'s whole reason for composing the
tenant in rather than comparing it afterwards.

The consequence is that a Session that does not exist and one that exists and holds
nothing at that path answer identically, and that is deliberate rather than lazy: the
alternative distinguishes them, which turns this route into a probe for whether a
Session id is real. The identical refusals `control/api/routes/files.py` gives for
another tenant's file exist for the same reason.

**No listing beside it.** What a Session produced is already in its Event Log, one
`output.produced` per file with the path that downloads it, in Turn order -- and a
listing would be a second answer to "what did this Session make", free to disagree with
the log and unable to say which Turn made what. A tenant polls the log for
`turn.completed` already; the paths arrive in the same stream.
"""

from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request, Response

from managed_agent.control.api.refusals import refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.api.routes.files import content_disposition
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vfs.session_vfs import (
    ARTIFACTS,
    SealedFile,
    VfsPathInvalid,
    VfsUnconfigured,
)

router = APIRouter(
    tags=["artifacts"],
    dependencies=[Depends(unauthenticated_tenant_from_header)],
)

REASON_ARTIFACT_PATH_INVALID: Final[str] = "artifact_path_invalid"
REASON_ARTIFACT_STORE_UNCONFIGURED: Final[str] = "artifact_store_unconfigured"
"""The two conditions named in `detail` rather than in a code of their own.

The published code set is closed (ADR-013) and carries no member for either, so each
travels as a `reason` under a code a caller can already branch on -- the shape
`control/api/routes/files.py` established for its own five.
"""


@router.get(
    "/sessions/{session_id}/artifacts/{path:path}",
    response_class=Response,
    responses={
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.FILE_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def download_an_artifact(
    session_id: SessionId,
    path: str,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> Response:
    """The artifact's bytes exactly as the agent wrote them, or a refusal.

    Served as an attachment and marked unsniffable, for the reason ship-out stores every
    produced file under the platform's default media type: nothing declared a type -- an
    agent writes bytes, not a `Content-Type` -- and rendering a file inline because the
    model happened to name it `.html` is how an agent's output runs script on this
    platform's origin. The `filename` in the disposition is the path's last segment,
    because that is what a browser will write to disk and a browser cannot be asked to
    create a directory.

    **The path is parsed and not merely passed through.** `SealedFile` parses it at
    construction, which is the check that matters, but its failure is a `ValueError`
    reaching a route -- so it is caught here and turned into the refusal it is: a path
    carrying `..`, a leading separator or an empty segment composes to no key at all,
    and telling the caller their path is malformed is a different fact from telling
    them nothing is stored at it.

    **A 404 covers three cases on purpose**: no such Session, a Session belonging to
    somebody else, and a Session of this tenant's that holds nothing at that path. The
    module docstring says why they are not separated.
    """
    try:
        target = SealedFile(
            tenant_id=tenant_id,
            session_id=session_id,
            lane=ARTIFACTS,
            relative=path,
        )
    except VfsPathInvalid as malformed:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            str(malformed),
            reason=REASON_ARTIFACT_PATH_INVALID,
            session_id=str(session_id),
        )
    try:
        body = await platform_from_request(request).session_artifacts.read(target)
    except VfsUnconfigured as unconfigured:
        return refuse(
            ErrorCode.INTERNAL,
            str(unconfigured),
            reason=REASON_ARTIFACT_STORE_UNCONFIGURED,
            session_id=str(session_id),
        )
    if body is None:
        return refuse(
            ErrorCode.FILE_NOT_FOUND,
            "no artifact is stored at that path in a session readable by this tenant",
            session_id=str(session_id),
        )
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "content-disposition": content_disposition(path.rsplit("/", 1)[-1]),
            "x-content-type-options": "nosniff",
        },
    )
