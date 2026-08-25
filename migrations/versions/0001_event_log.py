"""event_log: append-only, per-Session contiguous sequence.

Revision ID: 0001

Every guarantee this table makes is made by the store rather than by the code that
writes to it, because seven components append to one Session's log and a rule that lives
in one of them is not a rule. The pair `(session_id, seq)` is the primary key, so a
duplicate sequence is a constraint violation rather than a second row; the check keeps
the sequence from starting at 0; and a trigger refuses an UPDATE to a row that already
exists.

DELETE is deliberately left open: expiry is a real operation this platform performs (the
retention sweep), and a table that cannot forget cannot honour a retention policy.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "event_log",
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "appended_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # The pair is the primary key, so a duplicate sequence is a constraint violation
        # rather than a second row. This is what makes a race fail loudly (ADR-008).
        sa.PrimaryKeyConstraint("session_id", "seq"),
        sa.CheckConstraint("seq >= 1", name="event_log_seq_from_one"),
    )
    # An UPDATE is refused, not absorbed. A rewrite rule doing nothing would leave the
    # earlier events correct while reporting success to the writer that tried to change
    # them, and MAP-A45 requires the attempt to be refused rather than silently ignored
    # — so this raises, in the integrity-violation class, and the writer sees an error.
    op.execute(
        """
        CREATE FUNCTION event_log_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'event_log is append-only: session % seq % may not be updated',
                OLD.session_id, OLD.seq
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER event_log_no_update BEFORE UPDATE ON event_log
        FOR EACH ROW EXECUTE FUNCTION event_log_refuse_update();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS event_log_no_update ON event_log;")
    op.execute("DROP FUNCTION IF EXISTS event_log_refuse_update();")
    op.drop_table("event_log")
