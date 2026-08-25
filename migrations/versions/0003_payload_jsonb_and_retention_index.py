"""Store the payload as jsonb, and index the column the retention sweep filters on.

Two changes that are cheap now and expensive later, for the same reason: both rewrite or
rebuild against however many rows exist when they run, and right now that is none.

`payload` was `json`, which PostgreSQL stores as the original text and re-parses on
every access. `jsonb` parses once at write time and is the only one of the two that can
be indexed or queried inside. MAP-6's markers and MAP-19's evidence references both need
to look inside this column, and `json` -> `jsonb` on a populated `event_log` is a full
table rewrite holding an ACCESS EXCLUSIVE lock.

`appended_at` had no index while the migration that created it promises a retention
sweep -- "expiry is a real operation this platform performs". `WHERE appended_at < now()
- interval '30 days'` was a sequential scan whose cost tracks the *whole* table rather
than the rows being expired, so the sweep would get slower exactly as the log it exists
to trim grew.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # USING is required: PostgreSQL will not implicitly cast json to jsonb.
    op.execute(
        "ALTER TABLE event_log ALTER COLUMN payload TYPE jsonb USING payload::jsonb"
    )
    # Not covered by the primary key, which leads with session_id -- a sweep filters on
    # time across every Session, so it has no session_id to give the index.
    op.create_index("event_log_appended_at_idx", "event_log", ["appended_at"])


def downgrade() -> None:
    op.drop_index("event_log_appended_at_idx", table_name="event_log")
    op.execute(
        "ALTER TABLE event_log ALTER COLUMN payload TYPE json USING payload::json"
    )
