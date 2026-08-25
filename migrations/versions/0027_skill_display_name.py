"""A skill's human label, which the create door accepts and nothing could store.

Anthropic's `POST /v1/skills` takes an optional `display_name` beside the bundle, and
their read answers it back. This platform's create door had no field for it and this
table had no column, so a caller sending one either had it rejected by `extra="forbid"`
or -- worse, had the field been added without this -- accepted and dropped.

Nullable, and it stays nullable rather than defaulting to the skill's `name`. A label
somebody chose and a directory name parsed out of frontmatter are different facts, and
defaulting one to the other would make "this skill has no label" unrepresentable: every
row would carry a label, and no reader could tell an author's choice from a fallback.
Null is the honest value for every skill registered before this column existed.

Not indexed. Nothing looks a skill up by its label -- the listing orders by `name` and
every read is by id -- so an index here would be a write cost with no reader.
"""

import sqlalchemy as sa
from alembic import op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("skill", sa.Column("display_name", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("skill", "display_name")
