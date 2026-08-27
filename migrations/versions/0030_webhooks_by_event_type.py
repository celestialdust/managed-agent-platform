"""Webhooks subscribe to event types, and a delivery is keyed by the sequence it is for.

Two changes, and they are one change. A registration's filter stops being a set of
Session states and becomes a set of event type names, so `webhook.states` becomes
`webhook.event_types`; and the delivery ledger stops being keyed by the state a fold
named and becomes keyed by the sequence the event sits at, so `webhook_delivery`'s
primary key moves from `(webhook_id, session_id, state)` to `(webhook_id, session_id,
seq)` and its `state` column becomes `event_type`.

**Why the key had to move.** `(webhook_id, session_id, state)` can hold one row per
state a Session ever reaches, and a Session reaches the same state more than once: it is
created, suspended, and resumed, and the first and third both arrive at RUNNING. The
third event therefore lands on the row the first one inserted and is answered as another
*attempt* at a callback that was already delivered -- so a tenant hears about the
suspend, never hears about the resume, and nothing records a delivery as missing. The
sequence is what makes two events on one Session distinguishable, which is why it is the
key. It is not a new column: `0016_webhooks.py` has carried it since the table existed,
so this re-keys onto a column that is already there and already NOT NULL.

`event_type` stays on the row beside the key rather than in it. A retry rebuilds the
callback from this row, and the type is part of what the callback says.

**The index and the check are rebuilt rather than left under their old names.**
PostgreSQL rewrites an index definition and a check expression when the column beneath
them is renamed, but it does not rename either object, so leaving them would leave two
things called `webhook_states_*` over a column called `event_types` -- and the next
person to read the schema would have to work out which of the two names was the lie.

**The rows already stored are translated, because a rename moves a column and leaves
what is in it.** A registration written before this holds `running` and `stopped` in a
column the sweep now queries with event type names, so it matches nothing and stops
firing -- with no error, no refusal and no row anywhere saying a callback was owed. The
only evidence a tenant gets is an endpoint that went quiet, which is what a broken
delivery path looks like too.

The translation is the inverse of the fold that produced those states, so a subscription
keeps meaning what it meant: `running` becomes `session.created` **and**
`session.resumed`, because both folded to RUNNING and both used to produce a callback
for a tenant watching it. That is why the mapping is one-to-many in this direction and
why it cannot be run backwards element by element.

Half of that mapping subscribes a tenant to a type nothing currently emits: no code
appends `session.resumed` today. That is deliberate and not a loose end to tidy. It is
the type a pod rehydration will announce, so a registration translated here starts
firing when that lands instead of having been quietly narrowed on the way past -- and
narrowing it now would drop half of what the tenant asked for at the one moment nobody
would be watching for it. Do not "fix" this by deleting the `session.resumed` row from
the table below (ADR-032).

`webhook_delivery` is translated from the Event Log instead, joined on the
`(session_id, seq)` the row already carries -- the map cannot answer here, since
`running` is both a create and a resume and guessing would put a type on the wire naming
something that did not happen. Only rows still owed a delivery are rewritten: a
delivered row records what was *sent*, and what was sent was the old spelling.

**The downgrade can refuse, and that is the honest outcome.** Going back re-creates a
primary key that admits one row per state, and the rows this schema was made to allow
are exactly the ones that collide under it. It fails loudly on the duplicate rather than
picking a row to discard: which delivery record to lose is not a decision a migration
can make.

Revision ID: 0030
Revises: 0029
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None

_FOLD = (
    "(VALUES"
    " ('running', 'session.created'),"
    " ('running', 'session.resumed'),"
    " ('suspended', 'session.suspended'),"
    " ('stopped', 'session.stopped')"
    ") AS fold(state, event_type)"
)
"""`projection._TRANSITIONS`, written out as rows so SQL can join on it.

A copy of that table and deliberately a frozen one. A migration has to keep saying what
was true on the day it ran, so importing the live mapping would let a later edit to the
projection silently change what this already-applied revision claims it did.
"""


def _retype_registrations(from_column: str, to_column: str, on: str) -> sa.TextClause:
    """Rewrite each registration's array through the fold, in either direction.

    `coalesce(..., element)` keeps anything the join does not recognise rather than
    dropping it, so a row already holding the target vocabulary survives untouched and a
    value neither side knows is left for a person to find instead of deleted.

    `DISTINCT` matters going back: two types fold to one state, so a registration naming
    both would otherwise store it twice.
    """
    return sa.text(
        f"UPDATE webhook w SET {from_column} = translated.names FROM ("
        f" SELECT w2.id, array_agg(DISTINCT coalesce(fold.{to_column}, element)) AS"
        f" names FROM webhook w2, unnest(w2.{from_column}) AS element"
        f" LEFT JOIN {_FOLD} ON fold.{on} = element GROUP BY w2.id"
        ") AS translated WHERE w.id = translated.id"
    )


def upgrade() -> None:
    op.alter_column("webhook", "states", new_column_name="event_types")
    op.execute(_retype_registrations("event_types", "event_type", on="state"))
    # After the translation, so the index is built once over the values it will be
    # queried with rather than over the ones being replaced.
    op.drop_index("webhook_states_gin", table_name="webhook")
    op.create_index(
        "webhook_event_types_gin", "webhook", ["event_types"], postgresql_using="gin"
    )
    op.drop_constraint("webhook_states_nonempty", "webhook", type_="check")
    # `cardinality` and not `array_length(event_types, 1) >= 1`, for the reason 0016
    # measured: `array_length` answers NULL for an empty array, a CHECK whose expression
    # is NULL passes, and the constraint would admit exactly the row it refuses.
    op.create_check_constraint(
        "webhook_event_types_nonempty",
        "webhook",
        sa.text("cardinality(event_types) >= 1"),
    )

    op.drop_constraint("webhook_delivery_pkey", "webhook_delivery", type_="primary")
    op.alter_column("webhook_delivery", "state", new_column_name="event_type")
    # Only what is still owed. A delivered row is the record of a callback that really
    # was sent under the old spelling, and rewriting it would make the ledger describe
    # a delivery nobody received.
    op.execute(
        sa.text(
            "UPDATE webhook_delivery d SET event_type = e.type FROM event_log e"
            " WHERE e.session_id = d.session_id AND e.seq = d.seq"
            " AND d.delivered_at_ms IS NULL AND d.event_type <> e.type"
        )
    )
    op.create_primary_key(
        "webhook_delivery_pkey",
        "webhook_delivery",
        ["webhook_id", "session_id", "seq"],
    )


def downgrade() -> None:
    op.drop_constraint("webhook_delivery_pkey", "webhook_delivery", type_="primary")
    op.alter_column("webhook_delivery", "event_type", new_column_name="state")
    op.execute(
        sa.text(
            f"UPDATE webhook_delivery d SET state = fold.state FROM {_FOLD}"
            " WHERE fold.event_type = d.state AND d.delivered_at_ms IS NULL"
        )
    )
    # Refuses on a Session that reached one state twice -- see the module docstring.
    # The translation above is what creates that collision, by folding a create and a
    # resume back onto one name.
    op.create_primary_key(
        "webhook_delivery_pkey",
        "webhook_delivery",
        ["webhook_id", "session_id", "state"],
    )

    op.execute(_retype_registrations("event_types", "state", on="event_type"))
    op.drop_constraint("webhook_event_types_nonempty", "webhook", type_="check")
    op.drop_index("webhook_event_types_gin", table_name="webhook")
    op.alter_column("webhook", "event_types", new_column_name="states")
    op.create_index("webhook_states_gin", "webhook", ["states"], postgresql_using="gin")
    op.create_check_constraint(
        "webhook_states_nonempty", "webhook", sa.text("cardinality(states) >= 1")
    )
