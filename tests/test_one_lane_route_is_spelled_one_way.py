"""One route, spelled in two services, held together by a test rather than an import.

The Tool Gateway declares the working-lane route; the `restore-working-lane` init
container dials it. They cannot share the constant. The gateway module pulls in
Starlette and the MCP server library and costs about 0.458s to import, against
roughly 0.187s for the whole restore module -- and that module runs in an init
container in front of every pod start, whose common path is one listing and an exit.
`core/session/session_token.py` records the same rule for the same reason: a format
shared across services lives where importing it does not drag another service's
dependency graph along.

So the literal is written twice on purpose, which is exactly the shape that goes wrong
quietly -- the route is renamed on one side, the other still compiles, and the failure
is a 404 inside an init container at placement time. A test may import anything, so the
agreement is held here instead. This file is the sibling of
`tests/test_one_credential_name_is_spelled_one_way.py` and exists for the same reason.
"""

from __future__ import annotations

from managed_agent.gateway.tool.working_lane import (
    LANE_LISTING_PATH,
    LANE_OBJECT_PATH,
)
from managed_agent.session_shim.restore_working_lane import LANE_ROUTE


def test_the_client_dials_the_path_the_gateway_serves() -> None:
    assert LANE_ROUTE == LANE_LISTING_PATH, (
        "the restore init container dials a listing path the Tool Gateway does not "
        f"serve: it asks for {LANE_ROUTE!r} and the route table declares "
        f"{LANE_LISTING_PATH!r}"
    )


def test_an_object_hangs_off_that_same_listing_path() -> None:
    """The client builds an object url as the listing path plus the relative path.

    Asserted against the gateway's `{path:path}` declaration rather than against a
    second copy of the join, so renaming the route on either side fails here rather
    than at a pod's placement.
    """
    hung_off_the_listing = f"{LANE_ROUTE}/{{path:path}}"
    assert hung_off_the_listing == LANE_OBJECT_PATH, (
        f"the object route is {LANE_OBJECT_PATH!r}, which is not the listing path "
        f"{LANE_ROUTE!r} with a path parameter hung off it -- the client composes "
        "object urls that way and would ask for something that does not route"
    )
