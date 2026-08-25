"""An id for every skill that arrived by repository submission, so reads can name one.

This platform has two write doors for skills, and only one of them minted an id. An
upload got a `skill` row with a primary key, and everything a tenant can do afterwards
is addressed by it: read it back, delete it, list it, add a version to it. A repository
submission wrote `skill_repository_file` rows keyed by `(tenant, repository, revision,
name)` and no id at all -- so a team whose CI submits its skills could register them and
then had **no way to enumerate, read, or retire any of them.** The listing served the
uploaded half and said so in its docstring; the read and the delete took a uuid that a
repository skill did not have.

This table is the missing half of the id space. One row per repository skill, giving it
the same kind of handle an uploaded skill has had all along, so one collection can serve
both and the origin -- rather than the presence of an id -- decides which operations
apply to a row.

**A separate table rather than a column on `skill_repository_file`, because that table
refuses UPDATE.** An append-only trigger stands over it (see `0018`), which is the
property that makes a resubmitted commit a no-op instead of a second copy. Adding a
column is DDL and would be permitted, but backfilling the rows already there is an
UPDATE and the trigger would refuse it -- correctly. Appending a row per skill breaks
nothing and needs no exception carved into a guard that exists to have none.

**`uuid5` over the four key columns, not `uuid4`.** A resubmission of the same commit
must land on the same id. The file table already makes that resubmission a no-op by
conflicting on its primary key; a random id minted here would hand the same unchanged
skill a second identity, and a definition pinning the first would look retired. Derived
from the tenant as well as the repository, so two tenants submitting the same commit of
the same public repository get different ids and neither can name the other's row.

The id is **stored rather than derived on read**, even though it is a pure function of
the key. A read arrives holding the id and nothing else: recomputing the function
forwards cannot answer which of a tenant's rows produced it, and there is no reverse.
Unique in both directions -- the id is the primary key, and the key columns carry their
own unique constraint -- so neither a duplicate id nor a second id for one skill can be
written.

Backfilled in Python rather than in SQL. `uuid5` needs a hash function this database
does not have without an extension, and requiring `uuid-ossp` on every deployment to
compute four hundred identifiers once is a dependency bought at the wrong price.
"""

from uuid import UUID, uuid5

import sqlalchemy as sa
from alembic import op

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None

_NAMESPACE = UUID("b47ac10b-58cc-4372-a567-0e02b2c3d479")
"""The namespace repository-skill ids are minted in.

A fixed literal, pasted in once and never derived. The id a tenant read last week has to
be the id they read today, and a namespace computed from anything that moves -- a
version, a deployment, a revision of this file -- would silently re-mint every id the
next time that thing changed. It carries no meaning and is not a secret.

Deliberately not the namespace thread identifiers use (`shim/turn_runner.py`). Two id
spaces minted in one namespace could collide only if they hashed the same string, which
they cannot, but sharing it would mean a future change to either one's input format has
to reason about the other's.
"""


def _id_for(tenant_id: object, repository: str, revision_: str, name: str) -> UUID:
    """The id one repository skill is addressed by, from the four columns that key it.

    Separator is `\\n`, which none of the four inputs can contain: a tenant id is a
    uuid, a revision is 40 hex characters, and a skill name matched
    `SKILL_NAME_PATTERN` before it was stored. A separator a value could contain would
    let two distinct skills hash one string -- `repo="a", name="b/c"` against
    `repo="a/b", name="c"` -- and the collision would surface as one skill shadowing
    another rather than as an error.
    """
    return uuid5(_NAMESPACE, "\n".join([str(tenant_id), repository, revision_, name]))


def upgrade() -> None:
    op.create_table(
        "skill_repository_id",
        sa.Column("skill_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("repository", sa.Text(), nullable=False),
        sa.Column("revision", sa.CHAR(40), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "assigned_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("skill_id"),
        sa.UniqueConstraint(
            "tenant_id",
            "repository",
            "revision",
            "name",
            name="skill_repository_id_one_per_skill",
        ),
    )
    # Leading with the tenant for the reason `skill_by_tenant` does: every read has the
    # tenant as a term in the query rather than as a filter over fetched rows, so a scan
    # cannot cross tenants even where the planner would rather use the primary key.
    op.create_index(
        "skill_repository_id_by_tenant",
        "skill_repository_id",
        ["tenant_id", "name"],
    )
    # Written out rather than looped, matching `0018`: `tests/test_migrations.py` reads
    # these statements out of this file's source to confirm every append-only table
    # refuses an UPDATE the same way, and an f-string interpolating the table name
    # produces source that check cannot see.
    op.execute(
        """
        CREATE FUNCTION skill_repository_id_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'skill_repository_id is append-only; an id assignment is never moved';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER skill_repository_id_no_update
        BEFORE UPDATE ON skill_repository_id
        FOR EACH ROW EXECUTE FUNCTION skill_repository_id_refuse_update();
        """
    )
    _backfill()


def _backfill() -> None:
    """Assign an id to every repository skill already stored.

    Read in full rather than in batches. One row per skill per checkout is a count in
    the thousands at the outside, and a batched backfill would need a cursor that
    survives a failure mid-migration -- complexity bought for a table this size is
    complexity that only ever runs untested.
    """
    connection = op.get_bind()
    existing = connection.execute(
        sa.text(
            "SELECT tenant_id, repository, revision, name FROM skill_repository_file"
        )
    ).fetchall()
    if not existing:
        return
    connection.execute(
        sa.text(
            "INSERT INTO skill_repository_id "
            "(skill_id, tenant_id, repository, revision, name) "
            "VALUES (:skill_id, :tenant_id, :repository, :revision, :name)"
        ),
        [
            {
                "skill_id": _id_for(
                    row.tenant_id, row.repository, row.revision, row.name
                ),
                "tenant_id": row.tenant_id,
                "repository": row.repository,
                "revision": row.revision,
                "name": row.name,
            }
            for row in existing
        ],
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS skill_repository_id_no_update ON skill_repository_id;"
    )
    op.execute("DROP FUNCTION IF EXISTS skill_repository_id_refuse_update();")
    op.drop_index("skill_repository_id_by_tenant", table_name="skill_repository_id")
    op.drop_table("skill_repository_id")
