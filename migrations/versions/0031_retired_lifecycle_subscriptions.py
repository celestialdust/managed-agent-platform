"""Two event types stop being subscribable, and the registrations naming them go too.

A pod is leased for one Turn and destroyed when that Turn ends (ADR-041), so a Session
is never suspended between Turns and never resumes. `session.suspended` and
`session.resumed` have no producer left anywhere in the tree. Both stay *declared*,
because the rows already in tenants' Event Logs have to keep folding and keep replaying,
and a read surface drops an undeclared type in silence rather than refusing it. What
ends is eligibility: neither is a type a callback may be registered for any more.

**That flip does not reach the rows already stored, which is what this revision is
for.** `webhook.event_types` is a `text[]` and nothing has ever checked its contents
against the published vocabulary -- not the column, not a constraint, not the store. The
register route parses against the vocabulary, so nothing new gets in and everything
already in stays. And these are not hypothetical: 0030 wrote `session.resumed` into
every registration that had asked for `running`, and `session.suspended` into every one
that had asked for `suspended`, deliberately, with a note saying not to take them out.
That note was written while a pod rehydration was still expected to append the type.
There is no rehydration now.

**A registration left holding only these two is deleted, and that is the decision this
revision has to defend.** The CHECK on the column requires at least one name, so such a
row cannot simply be emptied; the answers are to delete it, or to leave it whole and
naming a type nothing will ever append.

Leaving it is the worse of the two, and 0030's own reasoning is what says so. That
revision translated `running` and `stopped` rather than letting a rename strand them, on
the grounds that a tenant's only evidence of a stranded subscription is an endpoint that
goes quiet -- which is indistinguishable from a delivery path that is broken. A row
surviving here is stranded in exactly that way: the tail scans only eligible types, so
it can never fire; the register route refuses both names, so it cannot be written again
as it stands; and a tenant reading their registrations back is shown a callback that is
coming. Deleting it is what turns that silence into something they meet -- a row missing
from their listing, and a re-registration answered by a refusal that names the type.

What the deletion costs is real and bounded. `webhook_delivery` cascades from
`webhook.id`, so a callback still owed to such a registration is cancelled rather than
retried; those are attempts at events already in the past, for a subscription with no
future. Nothing unrecoverable goes: the signing material lives in the credential vault
under `secret_ref` and this table holds only the reference, and the destination is the
tenant's own URL -- so re-registering it against a type that still fires is one call.

**The downgrade does nothing, and says so rather than guessing.** Which registrations
named these types is recorded nowhere once they are stripped, and re-adding the names to
every row that survives would subscribe tenants to types they never asked for. A
rollback past this point therefore leaves the stripped subscriptions narrowed and the
deleted ones gone: the release being rolled back to appends both types again and
delivers them to nobody who used to be listening. That is a loss, and it is written down
here rather than papered over -- refusing the rollback instead would make an incident
worse for the sake of a subscription list.

Revision ID: 0031
Revises: 0030
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None

_RETIRED = "ARRAY['session.resumed', 'session.suspended']::text[]"
"""The two names, written out rather than imported from the vocabulary.

Frozen for the same reason 0030 freezes its copy of the projection's fold: an
already-applied revision has to keep saying what it did on the day it ran, and a later
edit to the declarations must not silently change the claim.
"""


def upgrade() -> None:
    # Two statements and no ordering between them: the update's second condition
    # excludes exactly the rows the delete takes, so neither can see the other's work.
    #
    # `array_remove` twice rather than a rebuild through `unnest`, because it takes
    # every occurrence and leaves the surviving order alone -- so a row that changes is
    # a row that really did name one of these. `&&` and `<@` are both index-driven
    # against `webhook_event_types_gin`.
    op.execute(
        sa.text(
            "UPDATE webhook SET event_types ="
            " array_remove(array_remove(event_types, 'session.resumed'),"
            " 'session.suspended')"
            f" WHERE event_types && {_RETIRED} AND NOT (event_types <@ {_RETIRED})"
        )
    )
    # Whatever would have been left empty. See the module docstring: the CHECK refuses
    # an empty array, and a row naming only these can never fire, can never be written
    # again, and still reads back to its owner as a live callback.
    op.execute(sa.text(f"DELETE FROM webhook WHERE event_types <@ {_RETIRED}"))


def downgrade() -> None:
    """Deliberately empty. The module docstring says what is not restored, and why."""
