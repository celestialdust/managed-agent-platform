"""Serving a Session back the conversation its own earlier Turns wrote.

The Rollout is the Agent Runtime's own resume state. It is shipped out of a pod at
every completed Turn and, until this route, was read by nothing on the way in: a
Session's record survived its pod and no pod was ever given one back, so a Session that
lost its pod could only be started fresh -- replaying history its compaction
checkpoints had already folded, at the tenant's cost, while reporting success.

This is the read half of that round trip, and it sits on the Tool Gateway for the same
reason the working-lane routes do: the pod's egress is kube-dns and the two gateways,
the Gateway already holds the bucket grant and already verifies the `x-map-session`
token, and the init container that seeds the file already has that token in the
`compiled` volume. Nothing is minted, no arrow is opened (ADR-030, ADR-031).

**Who is calling is decided by the token and by nothing else.** The route takes no
tenant, no Session and no path -- there is nothing in a request for a caller to vary.
The Session whose Rollout comes back is the one in the verified `SessionContext`, so a
pod holding a token for Session A has no arrangement of inputs that reads Session B's
conversation. That is the whole security argument for a surface reachable by a confined
agent, and it is the same one the working-lane routes rest on.

**The bytes are cut back to the last completed Turn before they leave.** That cut is
the recovery boundary and belongs to the control plane's own module, which is why this
route asks for a restored Rollout rather than for an object: a route that served raw
stored bytes would put the boundary in two places, and the copy that drifted would hand
a resumed Session a torn tail (ADR-004).

**The ceiling is the client's, not this route's.** What a pod may write into its
runtime home is bounded by that volume, and the init container refuses a body over its
own budget before it writes a byte. A second bound here would be a second number to
keep in step, and the day they disagreed the Gateway would refuse a Rollout the pod
would have accepted while nothing said which bound spoke.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final, Protocol

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import request_response
from starlette.types import ASGIApp

from managed_agent.core.ids import SessionId
from managed_agent.core.session.session_token import SessionContext

ROLLOUT_SEED_PATH: Final[str] = "/v1/session/rollout"
"""Where the calling Session's restored Rollout is served, whole and at one path.

One object and no listing, unlike the working lane beside it: a Session has exactly one
Rollout, replaced wholesale at each completed Turn, so there is nothing to enumerate and
no path for a caller to compose.
"""

_NDJSON: Final[str] = "application/x-ndjson"

_log = logging.getLogger(__name__)


class RestoredBody(Protocol):
    """The one thing this route needs off a restored Rollout: the bytes to serve.

    A structural read-only view rather than the control plane's own `RestoredRollout`,
    so that nothing under `gateway/` imports from `control/`. Every other module in this
    service depends on `core` alone, and that is worth keeping true for a value this
    route reads one field of -- the cut counts and the dropped-line tally beside it are
    the shipping seam's business and reach no pod.

    A property rather than an attribute because the value satisfying it is a frozen
    dataclass, whose fields a mutable protocol attribute would not accept.
    """

    @property
    def body(self) -> bytes: ...


class SessionRollouts(Protocol):
    """Where one Session's resume state is read back from, already cut to a boundary.

    Keyed by the Session alone and taking no tenant, which is exactly the shape of the
    stored object: one Rollout per Session, and the Session segment of its key is the
    whole separation between one tenant's conversation and another's. A tenant argument
    here would be one this route had to invent, and an invented tenant is a filter that
    always agrees with itself -- the caller's tenancy is already settled by the token.

    No write and no delete. This service reads a Session's record back to a pod and has
    no business replacing or removing one: ship-out belongs to the control plane and
    expiry belongs to the retention sweep, and a Protocol offering either would make
    this the second process able to take a Session's ability to run again.
    """

    async def restore_for_resume(
        self, session_id: SessionId
    ) -> RestoredBody | None: ...


def rollout_seed_endpoint(
    rollouts: SessionRollouts, calling_session: Callable[[], SessionContext]
) -> tuple[str, ASGIApp]:
    """The seed path and the ASGI app serving it, unwrapped.

    Unwrapped for the reason the working-lane endpoints are: the caller puts the token
    middleware around it, so what sits behind the token check is decided in one place
    rather than asserted by each module about itself.

    `calling_session` is a callable and not a context because this endpoint is built
    once at start-up and outlives every request; binding a context here would answer
    every pod with whichever one came through the door first.
    """

    async def serve_rollout(_request: Request) -> Response:
        """This Session's Rollout, or 204 when no Turn of it ever completed.

        **204 and not 404, and not 200 over an empty body.** A first placement is the
        common case and is not an error, so a status meaning "absent" would put every
        Session's ordinary start-up on an error path. And a 200 carrying zero bytes is
        the dangerous one: the caller would write an empty file, the runtime would be
        asked to resume from a Rollout with no lines, and a record with no lines is the
        one shape its own reader treats as a hard error -- a pod that fails to start
        for a Session that simply has not run yet.
        """
        who = calling_session()
        restored = await rollouts.restore_for_resume(who.session_id)
        if restored is None:
            # At INFO rather than WARNING, and the difference from the working lane's
            # empty listing beside it is deliberate: a Session with nothing stored is
            # the ordinary first placement here, where an empty working lane is at
            # least as likely to be a lane this process could not read. The caller
            # does not act on absence blindly either -- it knows whether this Session
            # has run before, and refuses when the two disagree.
            _log.info(
                "session %s has no stored Rollout, so a pod seeded from this answer "
                "opens a new thread",
                who.session_id,
            )
            return Response(status_code=204)
        return Response(content=restored.body, media_type=_NDJSON)

    return ROLLOUT_SEED_PATH, request_response(serve_rollout)
