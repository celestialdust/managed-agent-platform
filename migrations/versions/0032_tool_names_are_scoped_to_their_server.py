"""A tool name becomes unique per server, and the name the model sees carries both.

Until this revision a tool's name was the primary key together with its tenant, so a
tenant could hold exactly one tool called `search` no matter how many MCP servers it had
registered. Two servers offering a tool of the same name -- which is ordinary; `search`,
`list_issues` and `get_file` are named the same by everyone -- meant the second
registration was refused outright, and there is no route that renames or removes the
first. The tenant's only way forward was a name nobody wanted, `search_2`, chosen once
and permanent.

**The identity of a tool is the pair `(server, tool)`**, which is also how the upstream
Managed Agents API models it: an `mcp_tool_use` block carries `name` and `server_name`
as separate fields, per-tool configuration is keyed by the bare name *inside* a toolset
that already names one server, and a tool name alone is never asked to be unique. So the
primary key becomes `(tenant_id, server_id, name)` and `name` goes back to being the
bare name the server itself reports.

**`advertised_name` is what the pair collapses to for the one layer that cannot carry a
pair.** The Agent Runtime is configured with a single MCP server -- the Tool Gateway --
so every tool a tenant has reaches the model through one namespace, as one string, with
no second field to put the server in. The Gateway joins the two as `<server>__<tool>`
and resolves an incoming call by looking that string up whole.

**Its unique index is not bookkeeping; it is the guarantee the naming scheme rests on.**
The runtime rewrites the names it is handed: it qualifies each as
`mcp__<server>__<tool>`, maps every character outside `[a-zA-Z0-9_]` to `_`, and appends
a SHA1-derived twelve-hex suffix when two names would sanitize to one. A Grant written
against a name that later acquires such a suffix resolves to nothing. Registered names
are shaped so the sanitizer is the identity function over them, and it is *tenant-unique
advertised names* that leave the collision suffix with nothing to disambiguate.
Per-server uniqueness alone would not do that. So the constraint that used to sit on
`name` moves to `advertised_name` rather than being dropped -- the tenant-wide guarantee
is unchanged, it now applies to the string that actually reaches the runtime.

That is also why the backfill can be a plain expression and needs no de-duplication
pass: every existing `name` was already unique within its tenant, so every
`<server>__<name>` it produces is too.

The column is written rather than generated, because the expression spans two tables --
`server_name` lives on `tool_server` -- and PostgreSQL admits no generated column that
does. The adapter computes it on insert; this index is what makes a wrong computation a
refused write instead of a duplicate the Gateway would resolve arbitrarily.
"""

import sqlalchemy as sa
from alembic import op

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


# Both tables refuse every UPDATE by trigger (0005, 0006), so the two backfills below
# each have to lift their table's refusal for the length of one statement. That is not a
# loophole in the invariant; it is the one place the invariant cannot be stated, and the
# reason is different for each table.
#
# `registered_tool_no_update` says "every Grant naming it was written against what it
# says now". `advertised_name` does not exist until three lines above the backfill,
# so no Grant can have been written against it, and `name` is not touched. The trigger
# cannot express "except a column added in this transaction", and it refuses per row --
# so against a registry that has ever held a tool the statement raises and the migration
# dies. It did: the first deploy to a cluster with 208 registered tools failed here.
#
# `session_no_update` protects a Grant from being widened. The second backfill widens
# nothing: it rewrites `ask` to `deepwiki__ask`, which is the same tool under the name
# the Gateway will now compare. Leaving it out is what would change a Grant -- to the
# empty one, for every Session that had a tool, silently.
#
# Both are re-enabled inside the same transaction the migration runs in, so no other
# writer can reach either table while the refusal is lifted.
_LIFT_THE_TOOL_REFUSAL = (
    "ALTER TABLE registered_tool DISABLE TRIGGER registered_tool_no_update"
)
_RESTORE_THE_TOOL_REFUSAL = (
    "ALTER TABLE registered_tool ENABLE TRIGGER registered_tool_no_update"
)
_LIFT_THE_SESSION_REFUSAL = "ALTER TABLE session DISABLE TRIGGER session_no_update"
_RESTORE_THE_SESSION_REFUSAL = "ALTER TABLE session ENABLE TRIGGER session_no_update"


def _rewrite_every_grant(matched_on: str, rewritten_to: str) -> str:
    """Rewrite each Grant entry that names one of its tenant's tools, and only those.

    `matched_on` is the tool column an entry is looked up by and `rewritten_to` is the
    one it becomes, so the upgrade and the downgrade are the same statement with the two
    swapped rather than two statements that can drift.

    **An entry matching no tool is carried through untouched, by `COALESCE`.** It named
    nothing before and names nothing after, and inventing a qualified form for it is the
    only edit here that could widen a Grant: a tenant could later register a tool whose
    advertised name is the string that was invented.

    `WITH ORDINALITY` and the `ORDER BY` keep the entries in the order they were stored.
    Membership is what the Gateway reads, so order decides nothing -- but a Grant that
    comes back reordered makes every later diff of the column unreadable.
    """
    return f"""
        UPDATE session s
           SET grant_tools = (
                   SELECT jsonb_agg(
                              COALESCE(t.{rewritten_to}, g.entry) ORDER BY g.ordinality
                          )
                     FROM jsonb_array_elements_text(s.grant_tools)
                          WITH ORDINALITY AS g(entry, ordinality)
                     LEFT JOIN registered_tool t
                            ON t.tenant_id = s.tenant_id AND t.{matched_on} = g.entry
               )
         WHERE jsonb_typeof(s.grant_tools) = 'array'
           AND jsonb_array_length(s.grant_tools) > 0
    """


def upgrade() -> None:
    op.add_column(
        "registered_tool", sa.Column("advertised_name", sa.Text(), nullable=True)
    )
    op.execute(_LIFT_THE_TOOL_REFUSAL)
    op.execute(
        """
        UPDATE registered_tool t
           SET advertised_name = s.server_name || '__' || t.name
          FROM tool_server s
         WHERE s.id = t.server_id
        """
    )
    op.execute(_RESTORE_THE_TOOL_REFUSAL)
    op.alter_column("registered_tool", "advertised_name", nullable=False)
    # After the backfill, so the joined string has one definition rather than two.
    op.execute(_LIFT_THE_SESSION_REFUSAL)
    op.execute(_rewrite_every_grant(matched_on="name", rewritten_to="advertised_name"))
    op.execute(_RESTORE_THE_SESSION_REFUSAL)
    # Dropped and rebuilt rather than extended: the old key IS the constraint being
    # removed, and a tenant holding one `search` is exactly what it enforced.
    op.drop_constraint("registered_tool_pkey", "registered_tool", type_="primary")
    op.create_primary_key(
        "registered_tool_pkey",
        "registered_tool",
        ["tenant_id", "server_id", "name"],
    )
    op.create_unique_constraint(
        "registered_tool_advertises_one_name_per_tenant",
        "registered_tool",
        ["tenant_id", "advertised_name"],
    )


def downgrade() -> None:
    # Not reversible without loss, and the failure is loud rather than silent. Two tools
    # named `search` behind different servers are exactly what this revision made
    # possible, and the old key cannot hold both -- so a downgrade over data written
    # since the upgrade fails on the primary key rather than choosing which tool to
    # drop. A tenant whose rows all predate the upgrade downgrades cleanly.
    # Before the column goes, or there is nothing left to match a Grant entry against
    # and every Grant stays qualified against tools that are bare again -- which is the
    # upgrade's own failure mode, arriving by the other direction.
    op.execute(_LIFT_THE_SESSION_REFUSAL)
    op.execute(_rewrite_every_grant(matched_on="advertised_name", rewritten_to="name"))
    op.execute(_RESTORE_THE_SESSION_REFUSAL)
    op.drop_constraint(
        "registered_tool_advertises_one_name_per_tenant",
        "registered_tool",
        type_="unique",
    )
    op.drop_constraint("registered_tool_pkey", "registered_tool", type_="primary")
    op.create_primary_key(
        "registered_tool_pkey", "registered_tool", ["tenant_id", "name"]
    )
    op.drop_column("registered_tool", "advertised_name")
