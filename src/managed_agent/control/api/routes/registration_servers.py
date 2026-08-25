"""POST /v1/mcp_servers — the only place an MCP server and its tools are accepted.

The two refusals arrive by different paths on purpose. A Scope Binding no argument can
express is refused while the body is being parsed, because the declaration carries
everything the check needs and a type that admitted it would let it reach the store. A
name already registered cannot be known at parse time, so the registry raises it and
this is where it becomes a response.

The response echoes the names the registration claimed rather than an identifier. A
definition names a server by name and a Grant names a tool by name, so the names are the
handles a caller uses afterwards, and the row ids stay internal (ADR-007).
"""

from collections.abc import Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from managed_agent.control.api.refusals import Refusal
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import TenantId
from managed_agent.core.registration.scope_binding import (
    NameAlreadyRegistered,
    RegisteredNameKind,
    ServerRegistration,
)

router = APIRouter(tags=["registration"])

_NAME_TAKEN: Final[Mapping[RegisteredNameKind, ErrorCode]] = {
    "server": ErrorCode.TOOL_SERVER_NAME_TAKEN,
    "tool": ErrorCode.TOOL_NAME_TAKEN,
}
"""Which refusal a taken name of each kind is.

Two codes and not one formatted string, which is what this replaces. A code assembled
at the call site (`tool_registration.{kind}_name_already_registered`) cannot be a member
of a closed enum, so it was a value callers could receive and no `match` could cover --
and the two cases genuinely differ for the caller: a taken SERVER name means this
registration was already made, while a taken TOOL name means two servers are offering
one name to the same Grant. A `Mapping` keyed by the `Literal` means adding a third kind
fails `mypy --strict` here rather than emitting an unmapped code.
"""
"""The refusal code, one member per kind of name.

A literal here rather than a member of the platform's closed `ErrorCode` set, because
that set does not exist yet -- it arrives with the slice that builds the error envelope.
It is a template in one place so that folding both members into the closed set later is
one edit.
"""


class ServerRegistered(BaseModel):
    """What a tenant gets back: the handles it will use afterwards.

    A definition names the server by `server_name` and a Grant names a tool by the names
    in `tools`, so echoing them is what lets a caller check that what it registered is
    what it can now address.
    """

    server_name: str
    tools: tuple[str, ...]


@router.post("/mcp_servers", status_code=status.HTTP_201_CREATED)
async def register(
    body: ServerRegistration,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> ServerRegistered:
    """Register a server and every tool it offers, or refuse the whole registration.

    409 on a name this tenant already holds, naming which names and which kind. Refused
    whole rather than partially: dropping the colliding tools and keeping the rest would
    leave the tenant believing it registered a catalog it did not, and disambiguating
    the name would silently re-point every Grant already written against it.

    A tool whose Scope Binding no argument can express never reaches here -- the body
    annotation refuses it with 400 before this runs. That is the point rather than an
    omission: a second check in the handler would be a second answer to the same
    question, free to disagree with the first.
    """
    try:
        await platform_from_request(request).tool_registry.register(tenant_id, body)
    except NameAlreadyRegistered as exc:
        raise Refusal(
            _NAME_TAKEN[exc.kind],
            f"{', '.join(exc.names)} is already registered for this tenant. "
            f"A name is never disambiguated behind the Grants written against "
            f"it, so this registration was refused whole.",
            names=", ".join(exc.names),
        ) from exc
    return ServerRegistered(
        server_name=body.server_name,
        tools=tuple(tool.name for tool in body.tools),
    )
