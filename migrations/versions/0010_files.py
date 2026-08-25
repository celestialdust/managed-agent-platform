"""uploaded_file: one row per file a tenant uploaded, before any Session exists.

Revision ID: 0010 Revises: 0014
"""

import sqlalchemy as sa
from alembic import op

revision = "0010"
# Branch point, not "0010 minus one". Revision ids come from the plan and slices land in
# wave order rather than numeric order, so the chain here is not sorted by file number:
# 0001 -> 0002 -> 0003 -> 0004 -> 0005 -> 0006 -> 0012 -> 0016 -> 0017 -> 0014. The live
# head is 0014, a higher number than this file declares, and 0007 through 0011 do not
# exist yet. Pointing this at 0009 would name a parent that is not there and
# `alembic upgrade head` would refuse.
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "uploaded_file",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("byte_length", sa.BigInteger(), nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column(
            "uploaded_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # The id alone, not (tenant_id, id). An id is issued once and addresses one file
        # for the whole platform, so a second tenant's row under the same id would make
        # the identifier ambiguous rather than private. The tenant is enforced in the
        # WHERE clause of every read instead, which is what makes a foreign id absent.
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "byte_length >= 0", name="uploaded_file_length_non_negative"
        ),
        sa.CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'", name="uploaded_file_hash_is_lower_hex"
        ),
        sa.CheckConstraint("length(filename) > 0", name="uploaded_file_named"),
        # The store's half of the same rule `_safe_filename` enforces at the boundary.
        # Non-ASCII letters pass here on purpose and are carried by the download's RFC
        # 6266 header; separators, quotes and control characters do not.
        sa.CheckConstraint(
            "filename !~ '[/\\\\\"[:cntrl:]]'", name="uploaded_file_name_is_a_leaf"
        ),
    )
    # The bytes behind a row cannot change: the key is derived from an id issued once
    # and is written once. So the length and the hash describe those bytes permanently,
    # and there is no legitimate UPDATE. Leaving that to application code would leave a
    # hand-run statement able to move the hash a download is checked against.
    #
    # A BEFORE UPDATE trigger that raises, not a rewrite rule: migration 0001 settled
    # that, and `test_every_append_only_table_refuses_an_update_the_same_way` fails on a
    # second mechanism appearing in the tree.
    op.execute(
        """
        CREATE FUNCTION uploaded_file_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                '% is append-only and may not be updated', TG_TABLE_NAME
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER uploaded_file_no_update BEFORE UPDATE ON uploaded_file
        FOR EACH ROW EXECUTE FUNCTION uploaded_file_refuse_update();
        """
    )


def downgrade() -> None:
    # The function is a schema object of its own and `drop_table` does not take it. A
    # downgrade that leaves it behind passes its own round-trip and fails the *next*
    # upgrade with "already exists" -- measured: DuplicateFunctionError.
    op.execute("DROP TRIGGER IF EXISTS uploaded_file_no_update ON uploaded_file;")
    op.execute("DROP FUNCTION IF EXISTS uploaded_file_refuse_update();")
    op.drop_table("uploaded_file")
