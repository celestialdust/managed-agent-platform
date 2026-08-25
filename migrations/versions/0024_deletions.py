"""An uploaded file and a skill can be deleted. Both as tombstones, and both for one
reason.

`DELETE /v1/files/{id}` was the sharpest absence on the whole surface: an uploaded file
could never be removed, which makes a tenant unable to honour a deletion request about
data they put here. That is a retention problem rather than an ergonomic one. `DELETE
/v1/skills/{id}` is the smaller sibling of the same gap.

**Neither is a DELETE of the row, and the reason is the same in both cases: the id
outlives the object.** A Session's `session.created` event names the files it was
created with, and an agent definition's `skills_revision` digest is taken over the
skills it resolved. Removing the row would leave those records naming an id that
resolves to nothing -- which reads as the platform having lost the data, not as the
tenant having deleted it. Those two look identical from outside and mean opposite
things, so the row stays and a tombstone says when it stopped being usable.

**For a file, the tombstone and the bytes part company, and that is the point.** The row
here is metadata -- a filename, a media type, a length, a digest. The personal data is
the bytes, and those are in the object store. A deletion request is about the bytes, so
the route deletes the object and writes this row; what survives is a record that an id
existed, was this long, and is gone. A tombstone that left the bytes in place would be a
deletion in name only, which is the worst of the available outcomes: the tenant has been
told the data is gone and it is not.

**A delete is refused while a live Session references the file**, and that guard is what
keeps this from being a way to break a running agent. A Session mid-run whose file
vanished would fail at its next pod placement, with the cause three layers away from the
symptom. The refusal is a 409 naming the Session, and a tenant honouring a deletion
request ends the Session first -- which is a step they can take, unlike debugging a
placement failure. Terminated Sessions do not hold the delete: their history keeps the
id, and this row is what tells that history what happened.

Revision ID: 0024 Revises: 0023
"""

import sqlalchemy as sa
from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None

# One statement per entry, because asyncpg refuses a prepared statement carrying
# several commands: a single string holding all four raises PostgresSyntaxError at the
# first `op.execute`. A tuple rather than one blob split on `;`, since a `;` inside a
# `$$` function body is not a statement boundary.
#
# At column zero so each `CREATE TRIGGER <name> BEFORE UPDATE` fits on one line, which
# is what `tests/test_migrations.py` matches in the source. Both names are too long to
# survive being indented inside `op.execute`, and a guard that check cannot see is a
# guard nobody counts.
_REFUSE_UPDATE: tuple[str, ...] = (
    """
CREATE FUNCTION uploaded_file_deletion_refuse_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION
        'uploaded_file_deletion is append-only and a deletion is terminal: file % may '
        'not be updated -- the bytes are already gone from the object store',
        OLD.file_id
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
""",
    """
CREATE TRIGGER uploaded_file_deletion_no_update BEFORE UPDATE ON uploaded_file_deletion
FOR EACH ROW EXECUTE FUNCTION uploaded_file_deletion_refuse_update();
""",
    """
CREATE FUNCTION skill_deletion_refuse_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION
        'skill_deletion is append-only and a deletion is terminal: skill % may not be '
        'updated',
        OLD.skill_id
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
""",
    """
CREATE TRIGGER skill_deletion_no_update BEFORE UPDATE ON skill_deletion
FOR EACH ROW EXECUTE FUNCTION skill_deletion_refuse_update();
""",
)

_DROP_GUARDS: tuple[str, ...] = (
    "DROP TRIGGER IF EXISTS skill_deletion_no_update ON skill_deletion;",
    "DROP FUNCTION IF EXISTS skill_deletion_refuse_update();",
    "DROP TRIGGER IF EXISTS uploaded_file_deletion_no_update"
    " ON uploaded_file_deletion;",
    "DROP FUNCTION IF EXISTS uploaded_file_deletion_refuse_update();",
)


def upgrade() -> None:
    op.create_table(
        "uploaded_file_deletion",
        sa.Column("file_id", sa.Uuid(), nullable=False),
        sa.Column(
            "deleted_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # One row per file, so a repeated delete is a conflict the insert absorbs and
        # the route answers 204 twice with the same recorded moment. A second row would
        # date the deletion to the retry, and a caller retrying a timeout would move
        # the answer to "when did this data go" every time they asked.
        sa.PrimaryKeyConstraint("file_id"),
        # `uploaded_file`'s key is `id` alone, so this reference has a unique target
        # and is declared -- a tombstone for a file that was never uploaded is
        # impossible rather than merely unlikely.
        sa.ForeignKeyConstraint(["file_id"], ["uploaded_file.id"]),
    )
    op.create_table(
        "skill_deletion",
        sa.Column("skill_id", sa.Uuid(), nullable=False),
        sa.Column(
            "deleted_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("skill_id"),
        sa.ForeignKeyConstraint(["skill_id"], ["skill.id"]),
    )
    for statement in _REFUSE_UPDATE:
        op.execute(statement)


def downgrade() -> None:
    """Drop the guards before the tables, and the functions with them.

    Same reason as every downgrade in this tree: `drop_table` takes the trigger and
    leaves the function, so the next upgrade fails on `CREATE FUNCTION ... already
    exists` -- the rollback that is only ever found during an incident.
    """
    for statement in _DROP_GUARDS:
        op.execute(statement)
    op.drop_table("skill_deletion")
    op.drop_table("uploaded_file_deletion")
