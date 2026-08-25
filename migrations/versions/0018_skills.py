"""skill and skill_repository_file: the two ways a skill reaches an agent. Append-only.

Two tables because the two routes are addressed differently and nothing about them
overlaps. `skill` holds one uploaded skill per platform-minted id, and an agent
definition attaches it by that id. `skill_repository_file` holds the skill directory of
one checkout, keyed by the `(repository, revision)` pair a definition already pins, and
an agent gets all of them or none. Folding them into one table would need a nullable
origin, which is two shapes in one relation and a constraint for every question about
which shape a row is.

Both are append-only, and for the same reason the definition rows are: a Session
resolves its skills once, when it is created, and goes on reading exactly the bytes it
resolved to. A skill that could be edited in place would change what a running agent
does with nothing anywhere recording that it changed. A new body is a new upload with a
new id, and a new commit is a new revision.

`body` is the whole `SKILL.md` as submitted rather than the parsed fields, so what the
runtime reads is byte-for-byte what was validated. `name` and `description` are stored
beside it as columns even though both are derivable from `body`, because they are what a
collision is detected on and what a listing shows, and re-parsing a document to answer
either is a parser in the read path.

Text, not a bucket. A `SKILL.md` is capped at 32 KiB at the door, so the largest agent
resolves to half a megabyte -- small enough that a row is the honest place for it, and
an object store would add a second system that can be up while this one is down for a
value neither larger nor more streamable than an agent's instructions, which have lived
in `agent_definition.body` since migration 0004.

An UPDATE is refused by a BEFORE UPDATE trigger that raises, which is the mechanism
migration 0001 chose and stated: a rewrite rule with DO INSTEAD NOTHING leaves the
stored row correct while reporting success to the writer that tried to change it.

Revision ID: 0018
Revises: 0007
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0018"
# Not "0018 minus one". Revision ids come from the plan and slices land in wave order
# rather than numeric order, so the parent is whatever head was live when this merged.
down_revision = "0007"
branch_labels = None
depends_on = None

_TABLES = ("skill", "skill_repository_file")


def upgrade() -> None:
    op.create_table(
        "skill",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "uploaded_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        # Deliberately NOT unique on (tenant_id, name). Two uploads of a skill called
        # `pdf-report` are two skills, and which one an agent runs is whichever id it
        # attached -- that is what makes an upload immutable and a revision safe to
        # replace by attaching a different id. The collision that does matter is two
        # skills resolving to one name *within one definition*, and that is refused
        # where the definition is resolved, because a repository can bring one too and
        # no constraint here can see that.
        sa.CheckConstraint("name <> ''", name="skill_name_is_not_blank"),
        sa.CheckConstraint("description <> ''", name="skill_description_is_not_blank"),
        sa.CheckConstraint("body <> ''", name="skill_body_is_not_blank"),
    )
    # Every read is `WHERE tenant_id = ... AND id = ANY(...)`, and the tenant is a term
    # in the query rather than a filter applied to fetched rows. Leading with the tenant
    # so the index serves that shape, and so a scan can never cross tenants even when
    # the planner would rather use the primary key.
    op.create_index("skill_by_tenant", "skill", ["tenant_id", "id"])
    op.create_table(
        "skill_repository_file",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("repository", sa.Text(), nullable=False),
        sa.Column("revision", sa.CHAR(40), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "submitted_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # One row per skill per checkout, which is also what makes a resubmission of the
        # same commit a no-op rather than a second copy: the insert conflicts here and
        # does nothing.
        sa.PrimaryKeyConstraint("tenant_id", "repository", "revision", "name"),
        sa.CheckConstraint(
            "name <> ''", name="skill_repository_file_name_is_not_blank"
        ),
        sa.CheckConstraint(
            "description <> ''", name="skill_repository_file_description_is_not_blank"
        ),
        sa.CheckConstraint(
            "body <> ''", name="skill_repository_file_body_is_not_blank"
        ),
    )
    # Written out twice rather than looped, and the reason is a check rather than a
    # style: `tests/test_migrations.py` reads these statements out of the *source* to
    # confirm every append-only table in the tree refuses an UPDATE the same way. An
    # f-string interpolating the table name produces source the pattern cannot see, so a
    # loop here would leave both of these tables unguarded by the check that exists to
    # notice exactly that.
    op.execute(
        """
        CREATE FUNCTION skill_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'skill is append-only: the body of skill % may not be changed under a '
                'Session that already resolved to it -- a new body is a new upload',
                OLD.id
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER skill_no_update BEFORE UPDATE ON skill
        FOR EACH ROW EXECUTE FUNCTION skill_refuse_update();
        """
    )
    op.execute(
        """
        CREATE FUNCTION skill_repository_file_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'skill_repository_file is append-only: revision % is a commit and its '
                'skills may not be changed under it -- a new body is a new commit',
                OLD.revision
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER skill_repository_file_no_update BEFORE UPDATE
        ON skill_repository_file
        FOR EACH ROW EXECUTE FUNCTION skill_repository_file_refuse_update();
        """
    )


def downgrade() -> None:
    """Remove everything `upgrade` created, both functions included.

    A function is a schema object of its own: a downgrade that dropped only the tables
    would leave them behind and the next `upgrade` would fail on `CREATE FUNCTION ...
    already exists`. Dropping a table takes its trigger and its index with it, so those
    are named only for the trigger, which must go before the function it calls.
    """
    for table in _TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_no_update ON {table};")
        op.execute(f"DROP FUNCTION IF EXISTS {table}_refuse_update();")
    op.drop_index("skill_by_tenant", table_name="skill")
    op.drop_table("skill_repository_file")
    op.drop_table("skill")
