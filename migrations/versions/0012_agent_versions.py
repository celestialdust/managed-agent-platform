"""agent_version_archive: one row per retired (definition, revision). Never rewritten.

Deliberately a table of its own rather than a column on `agent_definition`. Migration
0004 put a `BEFORE UPDATE` trigger on that table which *raises* -- so a retirement
written as a column update does not merely fail to take, it cannot be recorded there at
all. A second, append-only table is the only shape available, and it is also the one to
want: it keeps `agent_definition` immutable and makes a retirement the same kind of
thing as every other record here, a row that was written once.

There is no tenant column. Which tenant owns a revision is already a fact of
`agent_definition`, and a copy of it here would be free to disagree with the row it
describes. The insert that writes this table selects its key out of `agent_definition`
under a tenant predicate instead, so a cross-tenant retirement writes nothing at all.

The composite foreign key does real work: it makes retiring a revision nobody registered
a store-level refusal rather than a row describing nothing.

Revision ID: 0012
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op

revision = "0012"
# Branch point, not "0012 minus one". Revision ids come from the plan and slices land in
# wave order rather than numeric order; this one is built in wave 5 sub-wave 1, where
# the merged head is 0006.
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_version_archive",
        sa.Column("definition_id", sa.Uuid(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column(
            "archived_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # One row per retired version: a second retirement of the same version is a
        # conflict the insert absorbs, not a second retirement.
        sa.PrimaryKeyConstraint("definition_id", "revision"),
        # (id, revision) is agent_definition's primary key, so this pair can only name
        # a version that was actually registered.
        sa.ForeignKeyConstraint(
            ["definition_id", "revision"],
            ["agent_definition.id", "agent_definition.revision"],
            name="agent_version_archive_names_a_registered_version",
        ),
    )
    # An UPDATE raises rather than being absorbed, which is the mechanism migration 0001
    # settled for every append-only table here: a rewrite rule doing nothing would leave
    # the stored row correct while reporting success to whoever tried to change it.
    op.execute(
        """
        CREATE FUNCTION agent_version_archive_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'agent_version_archive is append-only: definition % revision % may not '
                'be updated',
                OLD.definition_id, OLD.revision
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER agent_version_archive_no_update
        BEFORE UPDATE ON agent_version_archive
        FOR EACH ROW EXECUTE FUNCTION agent_version_archive_refuse_update();
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS agent_version_archive_no_update"
        " ON agent_version_archive;"
    )
    op.execute("DROP FUNCTION IF EXISTS agent_version_archive_refuse_update();")
    op.drop_table("agent_version_archive")
