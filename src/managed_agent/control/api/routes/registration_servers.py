"""POST and GET on /v1/mcp_servers — the tenant's MCP catalog, written and read.

POST is the only place an MCP server and its tools are accepted. GET answers the
question the tenant otherwise has no way to ask: which names may a Grant contain.

The two refusals arrive by different paths on purpose. A Scope Binding no argument can
express is refused while the body is being parsed, because the declaration carries
everything the check needs and a type that admitted it would let it reach the store. A
name already registered cannot be known at parse time, so the registry raises it and
this is where it becomes a response.

The response echoes the names the registration claimed rather than an identifier. A
definition names a server by name and a Grant names a tool by name, so the names are the
handles a caller uses afterwards, and the row ids stay internal (ADR-007).
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, ConfigDict

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


class ToolInCatalog(BaseModel):
    """One tool, under both the names that matter when a Grant is written.

    `name` is what the tenant called it when registering; `advertised_name` is what a
    Grant has to say and what the model is shown. They differ by the server prefix, and
    a caller cannot derive the second from the first without knowing how this platform
    joins the two -- which is the whole reason this view carries both instead of one.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    advertised_name: str


class ServerInCatalog(BaseModel):
    """One registered server and the tools behind it."""

    model_config = ConfigDict(frozen=True)

    server_name: str
    tools: tuple[ToolInCatalog, ...]


class Catalog(BaseModel):
    """Every server a tenant holds. Empty when it has registered nothing."""

    model_config = ConfigDict(frozen=True)

    servers: tuple[ServerInCatalog, ...]


@router.get("/mcp_servers", response_model=Catalog)
async def catalog(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> Catalog:
    """What this tenant may name in a Grant, grouped under the server offering it.

    Scoped by `tool_registry.list_for_tenant(tenant_id)`, which takes the tenant into
    the query rather than filtering rows after the fact, so another tenant's catalog is
    never read and an empty answer never means "hidden". No limit is named because that
    read declares none: the registry lists a tenant's own rows whole, and the count is
    bounded by what that tenant registered.

    Grouped rather than flat because the POST beside it registers one server at a time,
    so a caller comparing what it registered against what it can now address is reading
    the same shape twice. Servers come back in name order and tools in the order the
    registry lists them, which is tool-name order -- a stable read, so a caller diffing
    two responses sees only what really changed.

    Carries neither the endpoint nor the scope bindings nor the remote name. A Grant is
    written from the two names here and nothing else, and the endpoint holds a url and
    a `credential_ref` -- a vault name that buys nothing alone, which is exactly what
    makes handing it back easy to wave through. Fail-safe default: a read gives up what
    its question needs, and this question does not need those.
    """
    tools = await platform_from_request(request).tool_registry.list_for_tenant(
        tenant_id
    )
    behind: defaultdict[str, list[ToolInCatalog]] = defaultdict(list)
    for tool in tools:
        behind[tool.server_name].append(
            ToolInCatalog(name=tool.name, advertised_name=tool.advertised_name)
        )
    return Catalog(
        servers=tuple(
            ServerInCatalog(server_name=name, tools=tuple(behind[name]))
            for name in sorted(behind)
        )
    )
