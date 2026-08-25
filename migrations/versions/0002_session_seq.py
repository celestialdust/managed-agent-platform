"""Record how far each Session's sequence has gone, so a sweep cannot rewind it.

The append path derived the next sequence from `max(seq)` over surviving rows. That is
correct until rows stop surviving: the migration before this one deliberately leaves
DELETE open for the retention sweep, and once a sweep removes a Session's whole log,
`max(seq)` is null and the sequence restarts at 1 -- reissuing identities that were
already handed out. The sequence is the SSE event id, so a consumer resuming at "last
event id 3" is then given a *different* event 3. Not a gap, which the primary key
catches loudly, but a duplicate identity across time, which nothing catches at all.

This table is the high-water mark. It is not a cache of `max(seq)` and it is not
authoritative for what the log contains: it says how far the numbering has gone, which
is a different fact and the only one that survives its rows being deleted. It is also
what lets `retained_floor` tell a fully-swept log from one that was never written --
without it both answer 1, which is the one distinction that method exists to draw.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_seq",
        sa.Column("session_id", sa.Uuid(), primary_key=True),
        # The sequence the *next* append will use. A Session with no appends has no row
        # rather than a row holding 1: absence is what distinguishes "never written"
        # from "written and swept", and a default row would erase it.
        sa.Column("next_seq", sa.BigInteger(), nullable=False),
        sa.CheckConstraint(
            "next_seq >= 2",
            name="session_seq_next_seq_is_past_the_first",
        ),
    )
    # Backfill, so this revision is correct on a database that already holds events
    # rather than only on an empty one. Grouped from the log itself because that is the
    # only place the information exists before this table.
    op.execute(
        "INSERT INTO session_seq (session_id, next_seq)"
        " SELECT session_id, max(seq) + 1 FROM event_log GROUP BY session_id"
    )


def downgrade() -> None:
    # Dropping this loses the high-water marks of any Session whose rows were swept, and
    # they cannot be recovered from the log. Stated rather than guarded: a downgrade is
    # a deliberate act and refusing it would leave no way back at all.
    op.drop_table("session_seq")
