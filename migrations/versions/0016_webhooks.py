"""webhook, webhook_delivery, webhook_scan: registrations, the once-per-state claim,
and the instant the cross-Session tail has read the Event Log through.

The claim table is the load-bearing one. "One callback per state change" is the whole
promise, and the cheapest way to hold it under two dispatchers, a restart mid-post and a
re-read of an overlapping window is to make it a primary key: the triple
`(webhook_id, session_id, state)` can exist once, so whichever pass inserts it owns the
delivery and every other pass is told no by the store rather than by a check it might
skip.

`webhook_delivery` is the one table here that is **not** append-only, so it takes no
update trigger: an attempt count that rises and a delivered timestamp that is written
once the receiver answers are the whole reason the row exists. `webhook` and
`webhook_scan` are not append-only either -- a registration is deleted rather than
edited, and the watermark's single row is what advances.

`webhook_scan` is seeded at the current instant rather than at zero. A platform that ran
before anybody registered a webhook has a history of Sessions that already stopped, and
a watermark of zero would make the first sweep deliver a callback for every one of them.

The states column is `text[]` rather than jsonb. The sweep asks this table one question
-- which of this tenant's registrations name this state -- which is
`:state = ANY(states)` over an array, and a GIN index covers that directly with no
second expression to keep in step with the column.

Revision ID: 0016
Revises: 0012
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016"
# Not "0016 minus one". Revision ids come from the plan and slices land in wave order
# rather than numeric order, so the parent is whatever head was live when this merged --
# 0012 here, which is a higher number than three of the revisions before it.
down_revision = "0012"
branch_labels = None
depends_on = None

_NOW_MS = sa.text("(extract(epoch from now()) * 1000)::bigint")


def upgrade() -> None:
    op.create_table(
        "webhook",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("states", sa.ARRAY(sa.Text()), nullable=False),
        # A name in the credential vault, never the secret itself. Every operator with a
        # psql prompt can read this row; none of them can sign a callback from what it
        # holds.
        sa.Column("secret_ref", sa.Text(), nullable=False),
        sa.Column(
            "created_at_ms", sa.BigInteger(), server_default=_NOW_MS, nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        # `cardinality` and not `array_length(states, 1) >= 1`. Measured: for an empty
        # array `array_length` returns NULL, a CHECK whose expression is NULL
        # passes, and the constraint would admit exactly the row it was written
        # to refuse.
        # `cardinality` returns 0 there, so the comparison is a real one.
        sa.CheckConstraint("cardinality(states) >= 1", name="webhook_states_nonempty"),
        # A second spelling of the parse rule in webhook_registry.py, on purpose. The
        # parse is what tells a tenant why their url was refused; this is what makes a
        # plaintext callback destination unwritable by anything, including a future
        # second writer. `starts_with` rather than `LIKE 'https://%'` so no literal `%`
        # travels through a driver that reads one as a parameter marker.
        sa.CheckConstraint("starts_with(url, 'https://')", name="webhook_url_is_https"),
    )
    op.create_index("webhook_by_tenant", "webhook", ["tenant_id"])
    op.create_index("webhook_states_gin", "webhook", ["states"], postgresql_using="gin")

    op.create_table(
        "webhook_delivery",
        sa.Column("webhook_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        # The sequence the state was reached at. Carried so a retry can rebuild the
        # identical callback without going back to the Event Log, whose window the sweep
        # has left behind.
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("delivered_at_ms", sa.BigInteger(), nullable=True),
        # The receiver's HTTP response code on the most recent attempt, kept for
        # an operator reading why a destination is not answering.
        sa.Column("last_response_code", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["webhook_id"], ["webhook.id"], ondelete="CASCADE"),
        # This is the "one callback" guarantee. A constraint rather than a check in the
        # dispatcher because two dispatchers racing one state change is the case that
        # matters, and only the store can decide that race.
        sa.PrimaryKeyConstraint("webhook_id", "session_id", "state"),
        sa.CheckConstraint("attempts >= 1", name="webhook_delivery_attempted"),
    )
    # The retry pass reads exactly this partial index, and a row leaves it the moment it
    # is delivered -- so the set the retry scans stays the size of what is currently
    # failing rather than the size of everything ever sent.
    op.create_index(
        "webhook_delivery_pending",
        "webhook_delivery",
        ["attempts"],
        postgresql_where=sa.text("delivered_at_ms IS NULL"),
    )

    op.create_table(
        "webhook_scan",
        sa.Column("id", sa.SmallInteger(), nullable=False),
        sa.Column("scanned_through_ms", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("id = 1", name="webhook_scan_single_row"),
    )
    # Seeded at now, not at zero: the Event Log predates this table, and a watermark of
    # zero would make the first sweep call back for every Session that ever reached a
    # named state.
    op.execute(
        sa.text(
            "INSERT INTO webhook_scan (id, scanned_through_ms)"
            " VALUES (1, (extract(epoch from now()) * 1000)::bigint)"
        )
    )


def downgrade() -> None:
    op.drop_table("webhook_scan")
    op.drop_index("webhook_delivery_pending", table_name="webhook_delivery")
    op.drop_table("webhook_delivery")
    op.drop_index("webhook_states_gin", table_name="webhook")
    op.drop_index("webhook_by_tenant", table_name="webhook")
    op.drop_table("webhook")
