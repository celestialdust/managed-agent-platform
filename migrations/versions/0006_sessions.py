"""session: the facts fixed when a Session is created. Never updated in place.

Two questions cannot be answered by folding one Session's Event Log, because both range
*across* Sessions rather than within one: which Sessions a tenant has, and in what order
they were created. This table exists for those two. Everything it holds is a creation
fact, so it takes the same update rule the Event Log, the definition registry and the
tool registry take -- written once, never altered.

There is no state, status or pod column, and their absence is the design rather than an
omission. State is folded from the Event Log on every read, so a column here would be a
second source free to disagree with the log; the pod binding is genuinely mutable and
lives with whatever owns placement. With both gone, nothing in this table can change,
which is what makes the update rule cost nothing.

An UPDATE raises rather than being absorbed. A rewrite rule with DO INSTEAD NOTHING
would leave the stored row correct while reporting success to whoever tried to change
it, which is the shape of failure discovered a month later by someone wondering why
their edit did not take. Migrations 0001, 0004 and 0005 each made this call and stated
it in the same terms; this follows them rather than reaching a fourth answer.

`created_at_ms` is epoch milliseconds in a `bigint` rather than a timestamp, because the
list route hands this value back to the caller inside a cursor and takes it again on the
next page. An integer survives that round trip exactly; a timestamp rendered into a
string and parsed back is a place for a lost sub-millisecond to skip a row silently.

`definition_revision` is text and not an integer. It mirrors `SessionRecord`, whose
field is a `str`, and a column that stores the field's own type is the one that cannot
lose a value the type admits -- the definition registry numbers revisions today, and
`"rev-1"` is a value the record accepts.

The two JSON columns are `jsonb` rather than `json`. `json` keeps the submitted text and
re-parses it on every access, and cannot be indexed or queried inside; converting later
is a full table rewrite under an ACCESS EXCLUSIVE lock, which is the cost migration 0003
paid on `event_log.payload` because the first migration did not choose it.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("definition_id", sa.Uuid(), nullable=False),
        sa.Column("definition_revision", sa.Text(), nullable=False),
        # Named grant_tools rather than grant: GRANT is a SQL keyword, and a quoted
        # column name has to be quoted in every statement touching it.
        #
        # Not nullable and with no server default: a Grant that was never written is
        # impossible, so an empty Grant is an empty array and reads as nothing rather
        # than as everything.
        sa.Column("grant_tools", postgresql.JSONB(), nullable=False),
        sa.Column("scope", postgresql.JSONB(), nullable=False),
        sa.Column("budget_minor_units", sa.BigInteger(), nullable=False),
        sa.Column("budget_currency", sa.CHAR(3), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=False),
        sa.Column(
            "created_at_ms",
            sa.BigInteger(),
            server_default=sa.text("(extract(epoch from now()) * 1000)::bigint"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        # A zero budget is not a Session that can do nothing, it is a Session nobody
        # meant to create; a zero retention is a log that expires the moment it is
        # written. The boundary refuses both, and this is what holds against a writer
        # that never passes the boundary.
        sa.CheckConstraint("budget_minor_units > 0", name="session_budget_positive"),
        sa.CheckConstraint("retention_days > 0", name="session_retention_positive"),
        # `NOT NULL` on a jsonb column does not say what these two constraints say,
        # and believing it does is the trap. A JSON `null` is a perfectly good jsonb
        # *value*, so `grant_tools` can be the literal `null` while the column is not
        # null -- and it reads back as Python `None`, where the reader expects a list
        # and every consumer downstream has to guess whether an undecided Grant means
        # no tools or all of them. Measured, not assumed: binding `None` through a
        # `sa.JSON()` parameter writes exactly that, and the insert succeeded without
        # these two constraints.
        sa.CheckConstraint(
            "jsonb_typeof(grant_tools) = 'array'", name="session_grant_is_an_array"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(scope) = 'object'", name="session_scope_is_an_object"
        ),
    )
    # The paging order is (created_at_ms, id) descending within one tenant, and this
    # index is that order. The id is in the key because two Sessions can share a
    # millisecond, and a page boundary between two equal keys is exactly where a keyset
    # walk repeats a row or skips one.
    op.create_index(
        "session_by_tenant_creation",
        "session",
        ["tenant_id", sa.text("created_at_ms DESC"), sa.text("id DESC")],
    )
    op.execute(
        """
        CREATE FUNCTION session_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'session is append-only: session % may not be updated -- every field '
                'of it is a fact fixed when it was created',
                OLD.id
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER session_no_update BEFORE UPDATE ON session
        FOR EACH ROW EXECUTE FUNCTION session_refuse_update();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS session_no_update ON session;")
    op.execute("DROP FUNCTION IF EXISTS session_refuse_update();")
    op.drop_index("session_by_tenant_creation", table_name="session")
    op.drop_table("session")
