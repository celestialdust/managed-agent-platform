"""tool_server and registered_tool: MCP server registrations and the tools they offer.

A Grant names a tool by the name registered here, so a name rewritten in place would
break every Grant already written against it and a name handed out twice would make two
tools answer to one Grant. Both facts point at the same schema: the tenant-facing name
is the key, so a repeat is a constraint violation rather than a second row, and there is
no update path at all.

An UPDATE raises rather than being absorbed. A rewrite rule doing nothing would leave
the stored registration correct while reporting success to whoever tried to change it,
which is the shape of failure discovered a month later by someone wondering why their
edit did not take. Migrations 0001 and 0004 made this call and stated it in the same
terms; this follows them rather than reaching a third answer for one question.

The JSON columns are `jsonb` rather than `json`. `json` keeps the submitted text and
re-parses it on every access, and cannot be indexed or queried inside -- and the check
constraint below queries inside one of them. Converting later is a full table rewrite
under an ACCESS EXCLUSIVE lock, which is the cost migration 0003 paid because the first
migration did not choose it.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_server",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("server_name", sa.Text(), nullable=False),
        sa.Column("endpoint", postgresql.JSONB(), nullable=False),
        sa.Column(
            "registered_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        # No agent-definition column, deliberately: a definition names a server by name,
        # so this one row is what every definition naming it reaches.
        sa.UniqueConstraint(
            "tenant_id", "server_name", name="tool_server_name_once_per_tenant"
        ),
        # The pair a tool row's composite foreign key points at, so a tool and the
        # server it belongs to can never end up under two different tenants.
        sa.UniqueConstraint("id", "tenant_id", name="tool_server_id_with_tenant"),
    )
    op.create_table(
        "registered_tool",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("remote_name", sa.Text(), nullable=False),
        sa.Column("parameters", postgresql.JSONB(), nullable=False),
        sa.Column("scope_bindings", postgresql.JSONB(), nullable=False),
        # The tenant-facing name is the key. A registration claiming a name this tenant
        # already uses violates it rather than being disambiguated into a second tool.
        sa.PrimaryKeyConstraint("tenant_id", "name"),
        sa.ForeignKeyConstraint(
            ["server_id", "tenant_id"],
            ["tool_server.id", "tool_server.tenant_id"],
            name="registered_tool_belongs_to_its_server",
        ),
        # The Scope-binding rule in the store and not only at the boundary: there is no
        # path by which a tool without a Scope Binding becomes a row (ADR-003).
        sa.CheckConstraint(
            "jsonb_array_length(scope_bindings) >= 1",
            name="registered_tool_has_a_scope_binding",
        ),
    )
    op.create_index("registered_tool_by_server", "registered_tool", ["server_id"])
    op.execute(
        """
        CREATE FUNCTION tool_server_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'tool_server is append-only: server % may not be updated -- register '
                'a new server instead',
                OLD.id
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER tool_server_no_update BEFORE UPDATE ON tool_server
        FOR EACH ROW EXECUTE FUNCTION tool_server_refuse_update();
        """
    )
    op.execute(
        """
        CREATE FUNCTION registered_tool_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'registered_tool is append-only: tool % may not be updated -- every '
                'Grant naming it was written against what it says now',
                OLD.name
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER registered_tool_no_update BEFORE UPDATE ON registered_tool
        FOR EACH ROW EXECUTE FUNCTION registered_tool_refuse_update();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS registered_tool_no_update ON registered_tool;")
    op.execute("DROP FUNCTION IF EXISTS registered_tool_refuse_update();")
    op.execute("DROP TRIGGER IF EXISTS tool_server_no_update ON tool_server;")
    op.execute("DROP FUNCTION IF EXISTS tool_server_refuse_update();")
    op.drop_index("registered_tool_by_server", table_name="registered_tool")
    op.drop_table("registered_tool")
    op.drop_table("tool_server")
