"""evidence_capture: one row per tool result that passed a capture point.

Revision ID: 0007 Revises: 0010
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
# Branch point, not "0007 minus one". Revision ids come from the plan and slices
# land in wave order rather than numeric order, so the chain on disk is not
# ascending: it runs 0006 -> 0012 -> 0016 -> 0017 -> 0014 -> 0010, and 0010 is the
# head.
#
# The plan's step file said 0013, which is MAP-45's, and MAP-45 is deferred -- so it
# named a parent that will never exist and alembic would have failed at the first
# upgrade. It was corrected to the live head in the step file itself. Re-point it at
# merge if another migration lands first; `tools/plan_waves.py` reports the head to use
# and names this file when the parent it finds is not in the shipping set.
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evidence_capture",
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("call_id", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("capture_point", sa.Text(), nullable=False),
        sa.Column("byte_length", sa.BigInteger(), nullable=False),
        sa.Column("threshold_bytes", sa.BigInteger(), nullable=False),
        # The octets of the same result that the capture point did not weigh and handed
        # back untouched. Without it `byte_length` is read as the size of the call's
        # output when it is only the size of the part that was measured, and a row
        # saying "eleven bytes, below the threshold" about a result that carried a
        # 200 KB embedded resource beside those eleven bytes answers a reviewer's
        # question wrongly rather than not at all.
        #
        # No server default, unlike `truncated_at_runtime_cap` below. False there is a
        # fact about a payload that has not passed through the runtime yet; zero here
        # would be an assertion that nothing else reached the caller, and a writer who
        # simply did not look would be making it silently.
        sa.Column("passed_through_bytes", sa.BigInteger(), nullable=False),
        sa.Column("hash_algorithm", sa.Text(), nullable=True),
        sa.Column("hash_hex", sa.Text(), nullable=True),
        sa.Column("object_key", sa.Text(), nullable=True),
        sa.Column(
            "truncated_at_runtime_cap",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "captured_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # One row per call, and the Session leads the key so an expiry sweep deletes a
        # Session's whole ledger on one index. A second capture of the same call is a
        # primary-key violation rather than a second row disagreeing with the first.
        sa.PrimaryKeyConstraint("session_id", "call_id"),
        # The size rule and the presence of a digest are one fact, so they are one
        # constraint. A row cannot record an inline return while carrying a hash, and
        # cannot record a capture without one.
        sa.CheckConstraint(
            "(byte_length >= threshold_bytes) = (hash_hex IS NOT NULL)",
            name="evidence_capture_hash_iff_at_threshold",
        ),
        sa.CheckConstraint(
            "(hash_hex IS NULL) = (object_key IS NULL)"
            " AND (hash_hex IS NULL) = (hash_algorithm IS NULL)",
            name="evidence_capture_reference_is_whole",
        ),
        sa.CheckConstraint(
            "hash_hex IS NULL OR hash_hex ~ '^[0-9a-f]{64}$'",
            name="evidence_capture_hash_is_lower_hex",
        ),
        sa.CheckConstraint(
            "byte_length >= 0 AND threshold_bytes >= 1 AND passed_through_bytes >= 0",
            name="evidence_capture_sizes_are_positive",
        ),
        sa.CheckConstraint(
            "capture_point IN ('tool-gateway', 'session-shim')",
            name="evidence_capture_point_is_known",
        ),
        # Nothing that was returned inline can be cut at a cap it never reached.
        sa.CheckConstraint(
            "NOT truncated_at_runtime_cap OR hash_hex IS NOT NULL",
            name="evidence_capture_truncation_implies_capture",
        ),
    )
    # Evidence is write-once, and the rule is enforced rather than documented: a
    # captured payload is addressed by the hash of its own bytes, so a row whose hash
    # could be rewritten would be a row that stops describing the object it points at.
    #
    # A BEFORE UPDATE trigger that raises, not a rewrite rule: migration 0001 settled
    # that, and `test_every_append_only_table_refuses_an_update_the_same_way` fails on a
    # second mechanism appearing in the tree.
    op.execute(
        """
        CREATE FUNCTION evidence_capture_refuse_update() RETURNS trigger AS $$
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
        CREATE TRIGGER evidence_capture_no_update BEFORE UPDATE ON evidence_capture
        FOR EACH ROW EXECUTE FUNCTION evidence_capture_refuse_update();
        """
    )


def downgrade() -> None:
    # A trigger and its function, in that order, and both by the names `upgrade` used.
    # The plan's snippet dropped a RULE that was never created and left the function
    # behind, which passes its own round trip and fails the NEXT upgrade with
    # DuplicateFunctionError -- the shape of a rollback nobody discovers until the
    # incident that needs it. `drop_table` takes the trigger with the table but not the
    # function, which is a schema object of its own.
    op.execute("DROP TRIGGER IF EXISTS evidence_capture_no_update ON evidence_capture;")
    op.execute("DROP FUNCTION IF EXISTS evidence_capture_refuse_update();")
    op.drop_table("evidence_capture")
