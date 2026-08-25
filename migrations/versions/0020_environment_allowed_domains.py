"""environment.allowed_domains: which names a Session's own commands may reach.

Revision ID: 0020 Revises: 0019
"""

import sqlalchemy as sa
from alembic import op

revision = "0020"
# The live head when this was written; see 0010's comment on why this chain is not
# sorted by file number.
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NOT NULL with a server default of the empty array, and the default is the whole
    # point rather than a convenience for existing rows. This column is the one field on
    # an Environment that WIDENS what a Session can do: empty means the sandbox keeps
    # egress off, and a non-empty list turns it on and confines it to those names. So
    # every row written before this column existed has to read as "no network", and a
    # nullable column would leave NULL to be interpreted -- by each reader, one of whom
    # would eventually read it as "unset, so unrestricted".
    #
    # JSON to match `denied_paths` beside it. An array column would be the better fit
    # for a list of names and would still be the wrong choice: the two lists are read by
    # one store through one row mapper, and a table where one list is JSON and the other
    # is text[] makes that mapper carry two shapes for one idea.
    op.add_column(
        "environment",
        sa.Column(
            "allowed_domains",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )
    # No CHECK constraint on the contents, unlike the digest pin above it. The domain
    # rules this platform enforces are a spelling grammar -- lower case, at least two
    # labels, a wildcard only in the leading label, nothing under this cluster's own
    # suffixes -- and expressing that in SQL would be a second copy of a grammar that
    # lives in `core/registration/environment.py`, free to disagree with it and applied
    # only to rows this application did not write. The digest pin is one LIKE over one
    # scalar and earns its place; this would not.
    #
    # No index. Nothing queries by this column: every read of this table is one row by
    # its primary key, and this field is carried on that row rather than searched.


def downgrade() -> None:
    op.drop_column("environment", "allowed_domains")
