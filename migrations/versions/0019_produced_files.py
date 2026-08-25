"""uploaded_file.produced_in_session_id: which files the agent made rather than got.

Revision ID: 0019 Revises: 0018
"""

import sqlalchemy as sa
from alembic import op

revision = "0019"
# The live head when this was written. Revision ids come from the plan and slices land
# in wave order rather than numeric order, so the chain here is not sorted by file
# number -- see 0010's comment. This one happens to be: the head is 0018.
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable, and NULL means "a tenant uploaded this". Two tables would have been the
    # other shape and the wrong one: a caller holds one identifier and downloads through
    # one route, so splitting the rows would make GET /v1/files/{id}/content two lookups
    # over one id, with "not found in either" indistinguishable from "found in the wrong
    # one".
    #
    # It has to be written at insert time or never. This table refuses UPDATE by trigger
    # (migration 0010), so nothing can come back later and say which rows an agent
    # produced -- there is no statement that would be allowed to.
    #
    # No foreign key to `session`. The retention sweep removes a Session's rows on its
    # own schedule and a produced file outlives the Session that made it: a tenant
    # downloading a document six months later must not find that row cascade-deleted, or
    # the Session's own delete refused. What this column is for is telling one kind of
    # row from the other, which needs the value and not referential integrity.
    op.add_column(
        "uploaded_file",
        sa.Column("produced_in_session_id", sa.Uuid(), nullable=True),
    )
    # No index. Nothing queries by this column yet -- the download route reads one row
    # by its primary key -- and an index shaped for a listing nobody has written would
    # be a guess at that listing's WHERE clause. The column carries the fact; the slice
    # that adds the listing adds the index its own query needs.


def downgrade() -> None:
    op.drop_column("uploaded_file", "produced_in_session_id")
