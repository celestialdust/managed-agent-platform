"""agent_definition: one row per (definition, revision). Never updated in place.

A definition is what an engineer writes instead of code, and a Session pins the revision
it resolved at creation. Both facts point at the same schema decision: `(id, revision)`
is the key, so registering again writes a new row rather than replacing one a running
Session may already be running on. Nothing here is ever edited, so the table has no
`updated_at` and no version column beyond the one in the key.

An UPDATE raises rather than being absorbed. A rewrite rule doing nothing would leave
the stored revision correct while reporting success to whoever tried to change it, which
is the shape of failure discovered a month later by someone wondering why their edit did
not take. Migration 0001 made this call for `event_log` and stated it in the same terms;
this follows it rather than reaching a second answer for the same question.

`body` is `jsonb` rather than `json`. `json` keeps the submitted text and re-parses
it on every access, and cannot be indexed or queried inside. Later work reads
fields out of this column, and `json` -> `jsonb` on a populated table is a full
rewrite holding an ACCESS EXCLUSIVE lock -- which is the cost migration 0003 paid
on `event_log.payload` because the first migration did not choose it.

`skills_revision` is duplicated out of `body` into its own column, and that is
deliberate rather than an oversight. The pin is what a later slice has to check a
publication gate against and what an operator has to search by, and neither is a thing
to do by digging into a document column. `body` remains the record of what the tenant
submitted; the column is the one field the platform itself queries on.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_definition",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("body", postgresql.JSONB(), nullable=False),
        # CHAR(40) rather than Text: a full lowercase sha is exactly forty characters,
        # and a fixed width is the store restating a constraint the boundary already
        # parses for. Neither is enough alone -- the boundary refuses a branch name of
        # the wrong length, this refuses a forty-character string that reached the table
        # without passing the boundary.
        sa.Column("skills_revision", sa.CHAR(40), nullable=False),
        sa.Column(
            "registered_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # (id, revision) is the key, so registering again writes a new revision rather
        # than replacing one a running Session may have pinned.
        sa.PrimaryKeyConstraint("id", "revision"),
        sa.CheckConstraint("revision >= 1", name="agent_definition_revision_from_one"),
    )
    # Leads with tenant_id, which the primary key cannot serve: the key leads with id,
    # so listing one tenant's definitions has no id to give it.
    op.create_index(
        "agent_definition_by_tenant", "agent_definition", ["tenant_id", "id"]
    )
    op.execute(
        """
        CREATE FUNCTION agent_definition_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'agent_definition is append-only: definition % revision % may not be '
                'updated -- register a new revision instead',
                OLD.id, OLD.revision
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER agent_definition_no_update BEFORE UPDATE ON agent_definition
        FOR EACH ROW EXECUTE FUNCTION agent_definition_refuse_update();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS agent_definition_no_update ON agent_definition;")
    op.execute("DROP FUNCTION IF EXISTS agent_definition_refuse_update();")
    op.drop_index("agent_definition_by_tenant", table_name="agent_definition")
    op.drop_table("agent_definition")
