"""Storing MCP server registrations and reading back the tools they offer.

Registering is one transaction over two tables, because a half-written registration
advertises tools whose server nobody can reach. Names are checked for a collision before
anything is inserted, and the two checks each earn their place: the tool check reports
*every* taken name rather than the first insert to fail, and the server check running
first is what makes a retry of a registration that already succeeded answer "this server
is registered" instead of sending the caller after a tool conflict that does not exist.
Neither check is the guarantee, though -- the store's constraints are. A concurrent
registration slipping between a check and its insert fails on the constraint, and the
`except IntegrityError` arms turn that into the same refusal rather than letting it take
the name.

Rows are parsed back into the types the registration was parsed into, so nothing hands
the Tool Gateway a shape that was only ever checked on the way in.

Bind parameter types are declared because these are textual statements and SQLAlchemy
has no column metadata to infer from: without one asyncpg receives a bare Python object
and refuses a `dict` for a `jsonb` column outright, while a uuid passed as text is
compared against whatever punctuation it happened to be spelled with. `.columns(...)` on
the reads is the same declaration in the other direction, and it is worth being precise
about what it buys here: measured against this dialect it changes nothing,
because SQLAlchemy's asyncpg driver installs its own json codec and the three JSON
columns already arrive decoded. It is declared anyway, as the statement of what these
columns are for a driver that does not -- but a reader should not believe it is what
makes the reads work today, because deleting it breaks no test and changes no value.
"""

from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.core.ids import TenantId
from managed_agent.core.registration.advertised_name import advertised_name_for
from managed_agent.core.registration.scope_binding import (
    NameAlreadyRegistered,
    RegisteredTool,
    ServerRegistration,
    UnknownTool,
)

# The join is on the server id alone. It needs no tenant term because the tool row is
# already filtered by tenant and the composite foreign key makes a tool and its server
# share one -- the constraint is what lets the read stay this short.
_TOOL_COLUMNS = (
    "SELECT t.name AS name, t.advertised_name AS advertised_name,"
    " t.remote_name AS remote_name,"
    " t.parameters AS parameters, t.scope_bindings AS scope_bindings,"
    " s.server_name AS server_name, s.endpoint AS endpoint"
    " FROM registered_tool t JOIN tool_server s ON s.id = t.server_id"
)

_READ_TYPES = {
    "name": sa.Text(),
    "advertised_name": sa.Text(),
    "remote_name": sa.Text(),
    "parameters": sa.JSON(),
    "scope_bindings": sa.JSON(),
    "server_name": sa.Text(),
    "endpoint": sa.JSON(),
}

_TENANT = sa.bindparam("tenant", type_=sa.Uuid())

_TAKEN_SERVER_NAME = sa.text(
    "SELECT server_name FROM tool_server"
    " WHERE tenant_id = :tenant AND server_name = :server_name"
).bindparams(_TENANT)

# On the advertised name and not on `name`, which is what per-server scoping means
# here: `search` behind two servers is two rows and no collision, while two rows that
# would advertise one string is still the collision the runtime cannot survive.
_TAKEN_TOOL_NAMES = sa.text(
    "SELECT advertised_name FROM registered_tool"
    " WHERE tenant_id = :tenant AND advertised_name = ANY(:names)"
).bindparams(_TENANT, sa.bindparam("names", type_=sa.ARRAY(sa.Text())))

_INSERT_SERVER = sa.text(
    "INSERT INTO tool_server (id, tenant_id, server_name, endpoint)"
    " VALUES (:id, :tenant, :server_name, :endpoint)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    _TENANT,
    sa.bindparam("endpoint", type_=sa.JSON()),
)

_INSERT_TOOL = sa.text(
    "INSERT INTO registered_tool"
    " (tenant_id, name, advertised_name, server_id, remote_name, parameters,"
    " scope_bindings)"
    " VALUES (:tenant, :name, :advertised_name, :server_id, :remote_name,"
    " :parameters, :scope_bindings)"
).bindparams(
    _TENANT,
    sa.bindparam("server_id", type_=sa.Uuid()),
    sa.bindparam("parameters", type_=sa.JSON()),
    sa.bindparam("scope_bindings", type_=sa.JSON()),
)

# Keyed on the advertised name because that is what the caller has: the Gateway is
# handed one string by the model and has no second field carrying the server.
_LOOKUP = (
    sa.text(
        f"{_TOOL_COLUMNS} WHERE t.tenant_id = :tenant AND t.advertised_name = :name"
    )
    .bindparams(_TENANT)
    .columns(**_READ_TYPES)
)

_LIST = (
    sa.text(f"{_TOOL_COLUMNS} WHERE t.tenant_id = :tenant ORDER BY t.name")
    .bindparams(_TENANT)
    .columns(**_READ_TYPES)
)


class PostgresToolRegistry:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None:
        """Write the server and every tool it offers, or write none of them.

        Raises `NameAlreadyRegistered` when this tenant already holds the server name,
        or when any tool would advertise a name it already advertises, naming which. A
        tool name this tenant uses behind a *different* server is not a collision: the
        pair identifies the tool, and only the joined form has to stay unique.

        The refusal is whole rather than partial: a registration that dropped its
        colliding tools and kept the rest would leave a tenant believing it registered
        a catalog it did not.

        The server name is checked before the tool names, and the order is the answer to
        a case where both are taken: re-submitting a registration that already succeeded
        collides on everything, and "this server is registered" is the true reading of
        that while "this tool name is taken" is not.
        """
        server_id = uuid4()
        async with self._engine.begin() as conn:
            taken_server = (
                await conn.execute(
                    _TAKEN_SERVER_NAME,
                    {"tenant": tenant_id, "server_name": registration.server_name},
                )
            ).scalar_one_or_none()
            if taken_server is not None:
                raise NameAlreadyRegistered("server", (str(taken_server),))

            names = [
                advertised_name_for(registration.server_name, tool.name)
                for tool in registration.tools
            ]
            taken_tools = (
                (
                    await conn.execute(
                        _TAKEN_TOOL_NAMES, {"tenant": tenant_id, "names": names}
                    )
                )
                .scalars()
                .all()
            )
            if taken_tools:
                raise NameAlreadyRegistered("tool", tuple(sorted(taken_tools)))

            try:
                await conn.execute(
                    _INSERT_SERVER,
                    {
                        "id": server_id,
                        "tenant": tenant_id,
                        "server_name": registration.server_name,
                        "endpoint": registration.endpoint.model_dump(mode="json"),
                    },
                )
            except IntegrityError as exc:
                # The check above lost a race with a concurrent registration. Reported
                # as the same refusal rather than as an integrity error, because the
                # caller's situation is identical either way and it must not retry: the
                # name is taken, and it will still be taken next time.
                raise NameAlreadyRegistered(
                    "server", (registration.server_name,)
                ) from exc

            for tool in registration.tools:
                try:
                    await conn.execute(
                        _INSERT_TOOL,
                        {
                            "tenant": tenant_id,
                            "name": tool.name,
                            "advertised_name": advertised_name_for(
                                registration.server_name, tool.name
                            ),
                            "server_id": server_id,
                            "remote_name": tool.remote_name,
                            "parameters": {
                                argument: declared.value
                                for argument, declared in tool.parameters.items()
                            },
                            "scope_bindings": [
                                binding.model_dump(mode="json")
                                for binding in tool.scope_bindings
                            ],
                        },
                    )
                except IntegrityError as exc:
                    # Named by the advertised form, because the bare name is not what
                    # collided: the caller may hold no other `search`, and being told
                    # `search` is taken would send them looking for one. The advertised
                    # name says which server already offers it.
                    raise NameAlreadyRegistered(
                        "tool",
                        (advertised_name_for(registration.server_name, tool.name),),
                    ) from exc

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        """The tool of that advertised name and the server behind it, this tenant only.

        `tool_name` is the *advertised* name -- `<server>__<tool>` -- because that is
        the only name the caller has. The Tool Gateway is handed one string by the model
        and no second field carrying the server, which is the whole reason the joined
        form exists.

        Raises `UnknownTool` when the tenant has none of that name. One refusal covers
        both "never registered" and "another tenant's", so holding a name confirms
        nothing about anyone else's catalog.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(_LOOKUP, {"tenant": tenant_id, "name": tool_name})
            ).one_or_none()
        if row is None:
            raise UnknownTool(tool_name)
        return RegisteredTool.model_validate(dict(row._mapping))

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[RegisteredTool]:
        """Every tool this tenant has registered, ordered by the name a Grant names."""
        async with self._engine.connect() as conn:
            rows = await conn.execute(_LIST, {"tenant": tenant_id})
            return [RegisteredTool.model_validate(dict(row._mapping)) for row in rows]
