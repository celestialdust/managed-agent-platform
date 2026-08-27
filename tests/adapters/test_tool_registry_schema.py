"""The store, not the application, is what refuses an illegal tool registration row.

Tier 1 (testcontainers, real PostgreSQL 17). Asserted with raw SQL rather than through
the registry adapter, because the point of each of these is that the guarantee survives
a writer that never loads our code -- a psql session, a later slice's adapter, a
migration somebody writes in a hurry.

A Grant names a tool by the name written here. That is the whole reason these
constraints exist: a name handed out twice makes two tools answer to one Grant, a name
rewritten in place breaks every Grant already written against it, and a tool with no
Scope Binding is reachable at the full breadth of a tenant's data by every Session whose
Grant names it.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

_DEEPWIKI: dict[str, object] = {
    "transport": "streamable_http",
    "url": "https://mcp.deepwiki.com",
    "credential_ref": "vault/acme/deepwiki",
    "credential_header": "Authorization",
}
_BINDING: list[dict[str, str]] = [{"dimension": "repository", "argument": "repoName"}]

_INSERT_SERVER = sa.text(
    "INSERT INTO tool_server (id, tenant_id, server_name, endpoint)"
    " VALUES (:id, :tenant, :server_name, :endpoint)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("endpoint", type_=sa.JSON()),
)

_INSERT_TOOL = sa.text(
    "INSERT INTO registered_tool"
    " (tenant_id, name, advertised_name, server_id, remote_name, parameters,"
    " scope_bindings)"
    " VALUES (:tenant, :name, :advertised, :server_id, :remote_name, :parameters,"
    " :bindings)"
).bindparams(
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("server_id", type_=sa.Uuid()),
    sa.bindparam("parameters", type_=sa.JSON()),
    sa.bindparam("bindings", type_=sa.JSON()),
)


async def _server(
    engine: AsyncEngine, tenant_id: uuid.UUID, server_name: str = "deepwiki"
) -> uuid.UUID:
    server_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            _INSERT_SERVER,
            {
                "id": server_id,
                "tenant": tenant_id,
                "server_name": server_name,
                "endpoint": _DEEPWIKI,
            },
        )
    return server_id


async def _tool(
    engine: AsyncEngine,
    tenant_id: uuid.UUID,
    server_id: uuid.UUID,
    name: str = "ask_question",
    bindings: object = None,
    server_name: str = "deepwiki",
) -> None:
    """Insert one tool row directly, joining its advertised name the way the app does.

    `server_name` is passed rather than read back from `server_id`, because this file
    grades constraints and a helper that queried for it would make every case depend on
    a read the constraint under test does not involve. The cases that need the two to
    disagree pass it explicitly.
    """
    async with engine.begin() as conn:
        await conn.execute(
            _INSERT_TOOL,
            {
                "tenant": tenant_id,
                "name": name,
                "advertised": f"{server_name}__{name}",
                "server_id": server_id,
                "remote_name": name,
                "parameters": {"repoName": "string"},
                "bindings": _BINDING if bindings is None else bindings,
            },
        )


async def test_one_tenant_cannot_register_one_server_name_twice(
    engine: AsyncEngine,
) -> None:
    """The second registration is refused rather than becoming a second server.

    Two servers of one name under one tenant give an agent definition naming that server
    two things to reach, and nothing in the definition can say which.
    """
    tenant_id = uuid.uuid4()
    await _server(engine, tenant_id)

    with pytest.raises(IntegrityError):
        await _server(engine, tenant_id)


async def test_two_tenants_may_each_register_the_same_server_name(
    engine: AsyncEngine,
) -> None:
    """The uniqueness is per tenant, not global.

    A global one would make the first tenant to register `deepwiki` the only tenant that
    ever can, and would leak that somebody had.
    """
    first, second = uuid.uuid4(), uuid.uuid4()

    await _server(engine, first)
    await _server(engine, second)

    async with engine.connect() as conn:
        registered = await conn.scalar(
            sa.text(
                "SELECT count(*) FROM tool_server"
                " WHERE server_name = 'deepwiki' AND tenant_id = ANY(:tenants)"
            ).bindparams(sa.bindparam("tenants", type_=sa.ARRAY(sa.Uuid()))),
            {"tenants": [first, second]},
        )

    assert registered == 2


async def test_a_second_server_may_offer_a_tool_name_the_first_already_uses(
    engine: AsyncEngine,
) -> None:
    """The key is `(tenant, server, name)`, so the same bare name behind two servers is
    two rows.

    `search`, `list_issues` and `get_file` are named the same by everyone, and until
    `0032` a tenant could hold exactly one of each across its whole catalogue. A tool is
    identified by its (server, tool) pair now -- the same shape the upstream Managed
    Agents API uses, where an `mcp_tool_use` block carries `name` and `server_name` as
    separate fields.
    """
    tenant_id = uuid.uuid4()
    first_server = await _server(engine, tenant_id, "deepwiki")
    second_server = await _server(engine, tenant_id, "acme-wiki")

    await _tool(engine, tenant_id, first_server, server_name="deepwiki")
    await _tool(engine, tenant_id, second_server, server_name="acme-wiki")

    async with engine.connect() as conn:
        held = await conn.scalar(
            sa.text(
                "SELECT count(*) FROM registered_tool"
                " WHERE tenant_id = :tenant AND name = 'ask_question'"
            ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
            {"tenant": tenant_id},
        )
    assert held == 2


async def test_two_tools_advertising_one_name_to_a_tenant_are_refused(
    engine: AsyncEngine,
) -> None:
    """What per-server scoping did not loosen, held by the index that replaced the key.

    The Agent Runtime is handed one namespace for a tenant's whole catalogue, and it
    appends a SHA1-derived suffix when two names it receives would sanitize to one -- so
    a Grant written against the original stops resolving. Tenant-unique *advertised*
    names are what leave that suffix nothing to disambiguate, which is why the
    constraint moved to `advertised_name` rather than being dropped.

    Forced here by writing the collision directly, which is the only way to reach the
    index: the adapter checks for a taken advertised name before it inserts, so through
    the app this arrives as a refusal rather than as an integrity error. Both matter --
    the check is the legible refusal, the index is what holds under a concurrent second
    registration.
    """
    tenant_id = uuid.uuid4()
    first_server = await _server(engine, tenant_id, "deepwiki")
    second_server = await _server(engine, tenant_id, "acme-wiki")
    await _tool(engine, tenant_id, first_server, server_name="deepwiki")

    with pytest.raises(IntegrityError):
        await _tool(engine, tenant_id, second_server, server_name="deepwiki")


async def test_two_tenants_may_each_register_a_tool_of_the_same_name(
    engine: AsyncEngine,
) -> None:
    first, second = uuid.uuid4(), uuid.uuid4()

    await _tool(engine, first, await _server(engine, first))
    await _tool(engine, second, await _server(engine, second))

    async with engine.connect() as conn:
        registered = await conn.scalar(
            sa.text(
                "SELECT count(*) FROM registered_tool"
                " WHERE name = 'ask_question' AND tenant_id = ANY(:tenants)"
            ).bindparams(sa.bindparam("tenants", type_=sa.ARRAY(sa.Uuid()))),
            {"tenants": [first, second]},
        )

    assert registered == 2


async def test_a_tool_row_with_no_scope_binding_is_refused_by_the_store(
    engine: AsyncEngine,
) -> None:
    """The boundary refuses this too. Both are needed and neither is the other.

    The boundary's refusal is what explains itself to the person who wrote the
    declaration; this one is what holds against every writer that never passes the
    boundary, which is the only reason the rule is a guarantee rather than a habit.
    """
    tenant_id = uuid.uuid4()
    server_id = await _server(engine, tenant_id)

    with pytest.raises(IntegrityError):
        await _tool(engine, tenant_id, server_id, bindings=[])


async def test_a_tool_row_naming_another_tenants_server_is_refused(
    engine: AsyncEngine,
) -> None:
    """The foreign key is composite, so a tool cannot straddle two tenants.

    A key on `server_id` alone would let one tenant's tool point at another tenant's
    server row -- and the endpoint, including the credential reference, comes from that
    row.
    """
    owner, stranger = uuid.uuid4(), uuid.uuid4()
    owners_server = await _server(engine, owner)

    with pytest.raises(IntegrityError):
        await _tool(engine, stranger, owners_server)


@pytest.mark.parametrize(
    ("table", "assignment", "key"),
    [
        ("tool_server", "server_name = 'renamed'", "server_name"),
        ("registered_tool", "name = 'renamed'", "name"),
    ],
)
async def test_an_update_raises_rather_than_quietly_changing_nothing(
    engine: AsyncEngine, table: str, assignment: str, key: str
) -> None:
    """Refused, not absorbed -- and the row is unchanged afterwards.

    A rewrite rule doing nothing would satisfy the second half of that and fail the
    first, telling a writer its edit succeeded while the stored name stayed put. Both
    halves are asserted because either alone admits the wrong implementation.
    """
    tenant_id = uuid.uuid4()
    server_id = await _server(engine, tenant_id)
    await _tool(engine, tenant_id, server_id)

    with pytest.raises(DBAPIError) as refused:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(  # noqa: S608 - table and assignment are test literals
                    f"UPDATE {table} SET {assignment} WHERE tenant_id = :tenant"
                ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
                {"tenant": tenant_id},
            )

    assert "append-only" in str(refused.value), str(refused.value)

    async with engine.connect() as conn:
        unchanged = await conn.scalar(
            sa.text(  # noqa: S608 - table and key are test literals
                f"SELECT {key} FROM {table} WHERE tenant_id = :tenant"
            ).bindparams(sa.bindparam("tenant", type_=sa.Uuid())),
            {"tenant": tenant_id},
        )

    assert unchanged != "renamed", (
        f"{table}.{key} was rewritten; every Grant written against the old name now "
        "resolves to nothing"
    )


async def test_neither_table_has_a_column_naming_an_agent_definition(
    engine: AsyncEngine,
) -> None:
    """One registration serves every definition naming the server.

    A per-definition column is how the same server ends up configured twice, with the
    two copies free to drift -- so the schema has nowhere to put one.
    """

    def _columns(connection: Any) -> dict[str, set[str]]:
        inspector = sa.inspect(connection)
        return {
            table: {column["name"] for column in inspector.get_columns(table)}
            for table in ("tool_server", "registered_tool")
        }

    async with engine.connect() as conn:
        columns = await conn.run_sync(_columns)

    assert columns["tool_server"], "the inspector found no tool_server columns at all"
    assert columns["registered_tool"], "the inspector found no registered_tool columns"
    for table, names in columns.items():
        assert not any("definition" in name or "agent" in name for name in names), (
            f"{table} carries a per-definition column: {sorted(names)}"
        )
