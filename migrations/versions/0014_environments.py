"""environment: registered sandbox shapes. Written once, never updated in place.

The update rule is the schema half of what the type guarantees. An environment that
could be edited would give the Session created on Tuesday a different shape from the one
created on Monday under the same id, and "two Sessions named one shape" would become a
claim about the clock. A new shape gets a new id.

An UPDATE is refused by a BEFORE UPDATE trigger that raises, which is the mechanism
migration 0001 chose and stated: a rewrite rule with DO INSTEAD NOTHING leaves the
stored row correct while reporting success to the writer that tried to change it.

There are deliberately no revisions in the manner of the agent-definition registry.
Nothing asks a Session to pin an environment revision, and a version nobody selects is a
column that can only drift.

Revision ID: 0014
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014"
# Not "0014 minus one". Revision ids come from the plan and slices land in wave order
# rather than numeric order, so the parent is whatever head was live when this merged --
# 0017 here, which is a higher number than the revision this file declares.
down_revision = "0017"
branch_labels = None
depends_on = None

_NOW_MS = sa.text("(extract(epoch from now()) * 1000)::bigint")


def upgrade() -> None:
    op.create_table(
        "environment",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("runtime_image", sa.Text(), nullable=False),
        # Not nullable and with no server default: a shape that denies nothing is an
        # empty array, so an absent list can never be read as an unwritten one.
        sa.Column("denied_paths", sa.JSON(), nullable=False),
        sa.Column(
            "created_at_ms", sa.BigInteger(), server_default=_NOW_MS, nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        # The digest pin holds in the store as well as in the parse. A row written by
        # anything but the parse still cannot carry a floating tag, and a tag is what
        # would let one id mean two different sets of bytes.
        sa.CheckConstraint(
            "runtime_image LIKE '%@sha256:%'",
            name="environment_image_digest_pinned",
        ),
        sa.CheckConstraint("length(name) > 0", name="environment_name_present"),
    )
    # Every read of this table is "this id, for this tenant"; the primary key covers the
    # id and the tenant is a predicate on top of it, so no second index earns its cost.
    op.execute(
        """
        CREATE FUNCTION environment_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'environment is append-only: the shape registered under % may not be '
                'changed -- two Sessions naming one id must run in one shape',
                OLD.id
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER environment_no_update BEFORE UPDATE ON environment
        FOR EACH ROW EXECUTE FUNCTION environment_refuse_update();
        """
    )


def downgrade() -> None:
    """Remove everything `upgrade` created, the function included.

    The function is dropped explicitly because it is a schema object of its own:
    `drop_table` takes the table's trigger with it but leaves the function behind, and
    the *next* upgrade then fails on `CREATE FUNCTION ... already exists`. That is the
    rollback nobody discovers until the incident -- roll forward, roll back, fix, roll
    forward again, and the last step is where it breaks. The trigger is named anyway so
    it goes before the function it calls, which keeps this correct if the table drop
    ever moves.
    """
    op.execute("DROP TRIGGER IF EXISTS environment_no_update ON environment;")
    op.execute("DROP FUNCTION IF EXISTS environment_refuse_update();")
    op.drop_table("environment")
