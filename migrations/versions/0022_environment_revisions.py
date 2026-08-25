"""Environments become updatable and retirable, on the mechanisms agents already use.

An Environment was permanent. `POST /v1/environments/{id}` had nowhere to write: this
table refuses an UPDATE by raising, so an edit was not merely unbuilt but unexpressible.
Anthropic's surface has both an update and an archive on an Environment, and their own
reference is explicit that they keep no history of one -- it tells the caller to "keep
your own record of the changes". This is that record.

**An update is a new revision, which is what keeps 0014's invariant true rather than
abandoning it.** The trigger installed there refuses a rewrite with the words "two
Sessions naming one id must run in one shape", and that is still the guarantee worth
having: an Environment is the sandbox a Session runs inside, so an id that quietly meant
a wider sandbox on Tuesday than it did on Monday would make an audit of what a Session
was allowed to do unanswerable. The primary key becomes `(id, revision)`, exactly as
`agent_definition`'s did in 0012, and a Session pins the revision it resolved at
creation. Two Sessions naming one id still run in one shape -- the shape that id had
when each was created.

The trigger's message is replaced rather than kept, because it would now be false in a
way that misleads: a reader hitting it would conclude an Environment cannot be changed,
when what cannot happen is a revision being rewritten after something pinned it. The
function is replaced in place, so the trigger built on it needs no change.

**The `server_default` of 1 on `revision` stays**, and an earlier draft of this
migration dropped it on an argument that does not survive being checked. That argument
was: a surviving default lets a writer omit the column, land on revision 1 forever, and
see a primary-key conflict the first time a tenant edits anything. The conflict is real
and is the reason to keep it -- a writer that omits the revision fails loudly on the
second write to one id, which is what it should do, and that second write was already a
conflict before this migration because the key was `id` alone. Dropped, the default
instead breaks every insert that exists today: the store does not name the column, so a
NOT NULL with nothing behind it refuses the first Environment anybody registers. Sixty-
three tests said so.

**`environment_archive` follows `agent_archive` in 0021**, down to having no foreign
key: this table's `id` is no longer unique on its own, so a reference to it has no
unique target to name. Existence is established by the route, which selects the
Environment under a tenant predicate before inserting. One row per Environment rather
than per revision, because archiving is a fact about the Environment -- retiring
revisions one at a time would leave the next edit free to create a live one, and
terminal is the property being copied.

**A delete needs no schema at all**, and that is worth stating so the absence does not
read as an oversight. `DELETE /v1/environments/{id}` is permitted only while no Session
references the Environment, so the guard is what protects the record and a tombstone
would add nothing: there is no Session whose history a hard delete could make
unreadable. The `BEFORE UPDATE` trigger does not touch DELETE, so the statement is
already allowed.

The local prose guides say that delete answers 204, twice and in two files, and both are
wrong -- their reference types the response `BetaEnvironmentDeleteResponse` and returns
`{"id": ..., "type": "environment_deleted"}`. A body is not a 204. Noted here because it
was the only status code stated anywhere in those guides, so a reader who trusts them
arrives at this table expecting a route that sends nothing back.

Revision ID: 0022 Revises: 0021
"""

import sqlalchemy as sa
from alembic import op

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None

_REVISION_MESSAGE = """
CREATE OR REPLACE FUNCTION environment_refuse_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION
        'environment is append-only: revision % of the shape registered under % may '
        'not be changed -- an edit is a new revision, because a Session pinned this '
        'one and must keep running in the shape it resolved',
        OLD.revision, OLD.id
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
"""

_ORIGINAL_MESSAGE = """
CREATE OR REPLACE FUNCTION environment_refuse_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION
        'environment is append-only: the shape registered under % may not be '
        'changed -- two Sessions naming one id must run in one shape',
        OLD.id
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
"""


_ARCHIVE_TRIGGER = """
CREATE TRIGGER environment_archive_no_update BEFORE UPDATE ON environment_archive
FOR EACH ROW EXECUTE FUNCTION environment_archive_refuse_update();
"""


def upgrade() -> None:
    op.add_column(
        "environment",
        sa.Column(
            "revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.create_check_constraint(
        "environment_revision_is_positive", "environment", "revision >= 1"
    )
    op.drop_constraint("environment_pkey", "environment", type_="primary")
    op.create_primary_key("environment_pkey", "environment", ["id", "revision"])
    op.execute(_REVISION_MESSAGE)
    op.create_table(
        "environment_archive",
        sa.Column("environment_id", sa.Uuid(), nullable=False),
        sa.Column(
            "archived_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # One row per Environment, so a repeated archive is a conflict the insert
        # absorbs and the route answers with the ORIGINAL timestamp. A fresh one would
        # say the Environment was retired at the moment of the retry, which is a false
        # fact about when it stopped being referenceable.
        sa.PrimaryKeyConstraint("environment_id"),
    )
    op.execute(
        """
        CREATE FUNCTION environment_archive_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'environment_archive is append-only and an archive is terminal: '
                'environment % may not be updated',
                OLD.environment_id
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(_ARCHIVE_TRIGGER)


def downgrade() -> None:
    """Undo both halves, and restore 0014's message rather than leaving 0022's behind.

    A downgrade that dropped the column but kept the replaced function would leave a
    trigger raising about a `revision` column that no longer exists, so the next
    attempted UPDATE would fail on the message instead of on the rule -- a rollback that
    looks complete and breaks the thing it restored.

    Reverting the primary key needs the extra revisions gone first: several rows can
    share an `id` now, and a key on `(id)` alone cannot be created over them. Deleting
    them is what a downgrade of this migration means -- the edits it made possible are
    the thing being removed -- and it is why this direction loses data where the upgrade
    lost none.
    """
    op.execute(
        "DROP TRIGGER IF EXISTS environment_archive_no_update ON environment_archive;"
    )
    op.execute("DROP FUNCTION IF EXISTS environment_archive_refuse_update();")
    op.drop_table("environment_archive")
    op.execute("DELETE FROM environment WHERE revision > 1;")
    op.drop_constraint("environment_pkey", "environment", type_="primary")
    op.create_primary_key("environment_pkey", "environment", ["id"])
    op.drop_constraint("environment_revision_is_positive", "environment", type_="check")
    op.drop_column("environment", "revision")
    op.execute(_ORIGINAL_MESSAGE)
