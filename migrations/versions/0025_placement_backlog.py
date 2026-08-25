"""A partial index over the placement events, so a backlog read is not a scan.

`GET /v1/capacity` answered a per-process queue depth until this landed, and the number
could be flatly wrong rather than merely imprecise: measured against `map-dev` on
2026-08-24, one replica reported `turns_awaiting_placement: 6` while the other reported
`0` for the same six placements. A client's connection is sticky, so all of one
operator's work lands on one replica and an operator polling the other sees nothing.

The fix reads the backlog out of the log instead, which is shared. Nothing new is
written for it: `session.placing` was already appended per waiting Turn, and what ends
the wait was already appended too -- the Turn's own `turn.started` or `turn.failed`. So
the count is a query over rows that exist rather than a new write on the hot path.

This index is what keeps that query off a full scan. Partial on the one type the query
selects, so it holds one row per placement wait ever recorded and not one per event --
on this log that is a small fraction of the table, and the index stays small as the log
grows with Turn traffic.

Ordered by `appended_at` because the query is bounded by time. A wait older than the
placement timeout is not a live wait; it is a wait whose process died, and counting it
forever would make the gauge climb monotonically and never come back down. The bound is
the caller's, so it is not written into the schema.

No index is added for the terminating lookup and none is needed. That lookup is
`session_id = ? AND seq > ?`, which is a range scan on the table's own primary key --
`(session_id, seq)` -- so it is already an index seek per candidate.
"""

from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None

_INDEX = "event_log_placement_waits"


def upgrade() -> None:
    op.execute(
        f"CREATE INDEX {_INDEX} ON event_log (appended_at) "
        "WHERE type = 'session.placing';"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX};")
