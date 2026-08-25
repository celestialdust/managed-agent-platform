"""agent_archive: one row per retired agent. The agent, not one of its revisions.

Distinct from `agent_version_archive` beside it, and the distinction is the whole reason
this table exists. That one retires a single revision so no new Session resolves to it
while other revisions stay live. This one retires the agent: every revision at once,
permanently, with no revision left to resolve to. A caller could approximate the second
by retiring every revision one at a time, and the result would not be the same thing --
a later `POST /v1/agents/{id}/versions` would succeed and the agent would be live again.
Retirement has to be a fact about the agent for it to be terminal.

**Terminal, and that is a copied decision rather than a chosen one.** Anthropic's
surface has no unarchive and no delete for an agent, so archive is the only end state an
agent has. A reversible archive is a different feature with a different failure mode: it
makes "this agent is retired" a claim with a lifetime, so everything that read it has to
read it again.

The shape follows `agent_version_archive`: an append-only table with a rewrite-refusing
trigger, not a nullable column on `agent_definition`. Migration 0004 put a raising
`BEFORE UPDATE` trigger on that table, so a retirement written as a column update cannot
be recorded there at all. A second table is the only shape available, and it is also the
one to want, because it keeps the definition immutable and makes a retirement the same
kind of thing as every other record here.

`archived_at` carries the timestamp because the API publishes it: their agent object
exposes `archived_at` as a nullable ISO 8601 string, null while live. The absence of a
row here is what renders as null, and a row's timestamp is what renders when there is
one. Not a boolean -- a boolean cannot answer "when", and their object does.

**No foreign key, and the reason is worth stating because a reader will look for one.**
`agent_version_archive` has one, pointing at `agent_definition(id, revision)`, which
works because that pair is that table's primary key. This table's key is `id` alone, and
`id` alone is not unique there -- it cannot be, since holding several revisions of one
agent is the whole point of that table. So there is no unique target for a reference to
name, and no index added here would create one: an index on `(id)` would not be unique
either, and the primary-key index on `(id, revision)` already serves a lookup by `id`.
Existence is therefore established by the route, which selects the agent under a tenant
predicate before inserting.

No tenant column, for the reason 0012 gives: which tenant owns an agent is already a
fact of `agent_definition`, and a copy here would be free to disagree with the row it
describes. That same select is what scopes the retirement, so a cross-tenant call writes
nothing at all.

Revision ID: 0021 Revises: 0020
"""

import sqlalchemy as sa
from alembic import op

revision = "0021"
# The live head when this was written; see 0010's comment on why this chain is not
# sorted by file number.
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_archive",
        sa.Column("definition_id", sa.Uuid(), nullable=False),
        sa.Column(
            "archived_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # One row per retired agent, so a second archive of the same agent is a conflict
        # the insert absorbs rather than a second retirement. That is what makes the
        # route idempotent: a caller whose first call timed out retries and gets the
        # ORIGINAL timestamp back, not a fresh one. A fresh timestamp would say the
        # agent was retired at the moment of the retry, which is a different and false
        # fact about when anything referencing it stopped being resolvable.
        sa.PrimaryKeyConstraint("definition_id"),
    )
    # An UPDATE raises rather than being absorbed, which is the mechanism migration 0001
    # settled for every append-only table here: a rewrite rule doing nothing would leave
    # the stored row correct while reporting success to whoever tried to change it. Here
    # it is also what makes the archive terminal -- with no unarchive to build, the only
    # way back is to rewrite or delete this row, and the first of those now raises.
    op.execute(
        """
        CREATE FUNCTION agent_archive_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'agent_archive is append-only and an archive is terminal: agent % may '
                'not be updated',
                OLD.definition_id
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER agent_archive_no_update BEFORE UPDATE ON agent_archive
        FOR EACH ROW EXECUTE FUNCTION agent_archive_refuse_update();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS agent_archive_no_update ON agent_archive;")
    op.execute("DROP FUNCTION IF EXISTS agent_archive_refuse_update();")
    op.drop_table("agent_archive")
