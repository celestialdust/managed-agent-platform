"""Skills get versions, and a version carries the sibling files a skill names.

Two absences, and the second is already producing wrong behaviour rather than merely
missing behaviour. A skill that regresses cannot be reverted, because there is nothing
to revert to: `POST /v1/skills` stores one row and the row cannot be rewritten. And a
skill is a single body, so it cannot carry the files it names. Anthropic's published
`pdf` skill tells the model to read `forms.md` and `reference.md`; uploaded here, those
references point at files that do not exist, and the model is instructed to consult them
anyway. `tests/fixtures/anthropic_pdf_skill.py` records that skill verbatim along with
the six distributions it names.

**A version is a microsecond timestamp, minted by the platform.** Their reference is
explicit and repeats it on all five version pages: "Each version is identified by a Unix
epoch timestamp", with the sample value `"1759178010641129"` -- sixteen digits, so
microseconds. Published as a string, and stored here as the integer it is. Stored as
text it would order lexicographically, which agrees with numeric order only while every
value has the same width, and the widths already disagree: their older examples show
`"1"`. An ordering that is correct until a digit is added is the kind that breaks years
later with no change to blame.

**A collision is possible and is the route's problem, named here so it is not a
surprise.** Two versions of one skill minted in the same microsecond conflict on this
key. The insert fails rather than silently overwriting, which is the right half to get
right; the route retries with the next microsecond. A sequence would avoid it and would
cost the published contract -- the value has to be a timestamp because that is what
their clients read.

**No separate opaque version id, and that is a deliberate divergence.** Their version
object carries both an `id` (`skillver_...`) and a `version`, and their own delete
endpoint then returns the *version string* in the `id` field -- so even upstream treats
the version as the identifier. Nothing in this repository publishes prefixed ids; every
id here is a bare UUID. Minting a second identifier for one row would add a convention
this platform does not have, to carry information `(skill_id, version)` already carries.

**`directory` is extracted from the bundle, not chosen by the caller.** A version's
content is a zip, and their version object publishes `directory` as "the top-level
directory name that was extracted from the uploaded files". It is stored rather than
recomputed because recomputing it means re-reading the file set on every read of the
version, and because the name the files were extracted under is a fact about that upload
-- a later change to how we pick it must not retroactively rename what a Session already
ran.

**The existing rows are backfilled as one version each, and `skill.body` stays where it
is.** That leaves one body recorded twice, which is worth being uneasy about and is safe
for a specific reason rather than a hopeful one: `skill` refuses an UPDATE, so
`skill.body` can never change, and the copy here is of a value that is already frozen.
Two records of one fact drift when either can move; neither can. The alternative -- no
backfill -- would have `GET /v1/skills/{id}/versions` answer empty for a skill whose
`latest_version` names a version, which is a contradiction a caller sees rather than an
untidiness a reader sees. The backfilled version is the upload's own timestamp in
microseconds, so it is the true moment rather than the migration's.

**`latest_version` is `max(version)` and is not stored.** Their reference publishes it
as "the most recent version of the skill that has been created" and offers no promote or
activate endpoint anywhere across nine skills pages, so latest is a derived fact about
the set. Stored, it would need an UPDATE, which every table here refuses.

**`skill_version_file` is why versions are worth having at all**, and its path column
carries the check that matters: a sibling file is written into the skill's directory in
a Session's workspace, so a path that escapes that directory escapes into the workspace.
Absolute paths and any path containing `..` are refused by the store as well as by the
parse, because the rule protects a filesystem and a row written by anything but our
parse would reach that filesystem the same way.

**Deleting a version is a tombstone, not a DELETE.** A version can be pinned: an agent
definition carries a `skills_revision` digest over the set it resolved, and a Session
created under that definition has a history that names it. Removing the row would make
that history unreadable -- the digest would resolve to nothing, and the record would say
the Session ran skills that cannot be produced. So the row stays and the tombstone is
what stops new resolution. Their delete answers `{id, type}` with no `deleted_at`
anywhere in the response, which says nothing about what the store keeps: it is a wire
shape, and this is the record behind it.

Revision ID: 0023 Revises: 0022
"""

import sqlalchemy as sa
from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None

# One statement per entry, because asyncpg refuses a prepared statement carrying
# several commands: a single string holding all six raises PostgresSyntaxError at
# the first `op.execute`, and the migration fails on syntax rather than on schema.
# A tuple rather than one blob split on `;`, since a `;` inside a `$$` function
# body is not a statement boundary and splitting there would cut the body in half.
#
# At column zero so each `CREATE TRIGGER <name> BEFORE UPDATE` fits on one line,
# which is what `tests/test_migrations.py` matches in the *source*. A name long
# enough to wrap when indented inside `op.execute` is a guard that check cannot
# see -- present, correct, and uncounted. Literal strings rather than an f-string
# over the table names, for the reason 0018 gives: interpolated source is source
# that check cannot read.
_REFUSE_UPDATE: tuple[str, ...] = (
    """
CREATE FUNCTION skill_version_refuse_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION
        'skill_version is append-only: version % of skill % may not be changed -- a '
        'correction is a new version, because a definition pinned this one',
        OLD.version, OLD.skill_id
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
""",
    """
CREATE TRIGGER skill_version_no_update BEFORE UPDATE ON skill_version
FOR EACH ROW EXECUTE FUNCTION skill_version_refuse_update();
""",
    """
CREATE FUNCTION skill_version_file_refuse_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION
        'skill_version_file is append-only: % in version % of skill % may not be '
        'changed -- the model is told to read this file and must read one thing',
        OLD.path, OLD.version, OLD.skill_id
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
""",
    """
CREATE TRIGGER skill_version_file_no_update BEFORE UPDATE ON skill_version_file
FOR EACH ROW EXECUTE FUNCTION skill_version_file_refuse_update();
""",
    """
CREATE FUNCTION skill_version_retire_refuse_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION
        'skill_version_retirement is append-only and a retirement is terminal: '
        'version % of skill % may not be updated',
        OLD.version, OLD.skill_id
        USING ERRCODE = 'restrict_violation';
END;
$$ LANGUAGE plpgsql;
""",
    """
CREATE TRIGGER skill_version_retire_no_update BEFORE UPDATE ON skill_version_retirement
FOR EACH ROW EXECUTE FUNCTION skill_version_retire_refuse_update();
""",
)

_DROP_GUARDS: tuple[str, ...] = (
    "DROP TRIGGER IF EXISTS skill_version_retire_no_update"
    " ON skill_version_retirement;",
    "DROP FUNCTION IF EXISTS skill_version_retire_refuse_update();",
    "DROP TRIGGER IF EXISTS skill_version_file_no_update ON skill_version_file;",
    "DROP FUNCTION IF EXISTS skill_version_file_refuse_update();",
    "DROP TRIGGER IF EXISTS skill_version_no_update ON skill_version;",
    "DROP FUNCTION IF EXISTS skill_version_refuse_update();",
)

# Every skill stored before versions existed becomes its own first version, timestamped
# from the upload it already records rather than from now() -- a version dated today
# would date the skill to the migration instead of to the upload, and `latest_version`
# is read from this column.
#
# `directory` is the skill's own name, because a skill uploaded as one body was never
# extracted from a bundle and has no other candidate. It is the name the model would
# have seen the file under.
_BACKFILL = """
INSERT INTO skill_version
    (skill_id, version, name, description, body, directory, created_at)
SELECT
    id,
    (extract(epoch from uploaded_at) * 1000000)::bigint,
    name,
    description,
    body,
    name,
    uploaded_at
FROM skill;
"""


def upgrade() -> None:
    op.create_table(
        "skill_version",
        sa.Column("skill_id", sa.Uuid(), nullable=False),
        # Microseconds since the epoch. See the docstring on why this is an integer
        # while the wire value is its decimal string.
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("directory", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("skill_id", "version"),
        # `skill`'s key is `id` alone and stays that way, so unlike the archive tables
        # in 0021 and 0022 this reference has a unique target and is declared. It is
        # what makes a version of a skill that was never uploaded impossible rather
        # than merely unlikely.
        sa.ForeignKeyConstraint(["skill_id"], ["skill.id"]),
        # Rejects a version minted before 2001-09-09, which in microseconds is where
        # sixteen digits begin. The value this actually catches is a *millisecond*
        # timestamp written by mistake: those are three digits shorter, so they land in
        # 1970 and would sort before every real version rather than failing.
        sa.CheckConstraint(
            "version >= 1000000000000000", name="skill_version_is_microseconds"
        ),
        sa.CheckConstraint("name <> ''", name="skill_version_name_is_not_blank"),
        sa.CheckConstraint(
            "description <> ''", name="skill_version_description_is_not_blank"
        ),
        sa.CheckConstraint("body <> ''", name="skill_version_body_is_not_blank"),
        # The directory is joined onto a workspace path, so it is one path segment and
        # nothing else. A separator or a `..` in it would place the whole extracted
        # bundle somewhere the caller chose.
        sa.CheckConstraint(
            "directory <> '' AND directory NOT LIKE '%/%'"
            " AND directory NOT LIKE '%..%'",
            name="skill_version_directory_is_one_segment",
        ),
    )
    op.create_table(
        "skill_version_file",
        sa.Column("skill_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        # One row per path per version, so a bundle naming one path twice is a conflict
        # rather than two rows of which the reader picks whichever came back first.
        # Which of the two the model would have read is not a question worth having an
        # answer to.
        sa.PrimaryKeyConstraint("skill_id", "version", "path"),
        sa.ForeignKeyConstraint(
            ["skill_id", "version"],
            ["skill_version.skill_id", "skill_version.version"],
        ),
        sa.CheckConstraint("path <> ''", name="skill_version_file_path_is_not_blank"),
        # The path is joined onto the version's directory inside a Session's workspace.
        # An absolute path ignores that directory and a `..` climbs out of it, so both
        # are refused here as well as at the parse: this constraint is what holds for a
        # row our parse did not write, and the filesystem it protects cannot tell the
        # difference between the two writers.
        sa.CheckConstraint(
            "path NOT LIKE '/%' AND path NOT LIKE '%..%'",
            name="skill_version_file_path_stays_inside",
        ),
        sa.CheckConstraint("body <> ''", name="skill_version_file_body_is_not_blank"),
    )
    op.create_table(
        "skill_version_retirement",
        sa.Column("skill_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column(
            "retired_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # One row per version, so a repeated delete is a conflict the insert absorbs
        # and the route answers the same way twice.
        sa.PrimaryKeyConstraint("skill_id", "version"),
        sa.ForeignKeyConstraint(
            ["skill_id", "version"],
            ["skill_version.skill_id", "skill_version.version"],
        ),
    )
    for statement in _REFUSE_UPDATE:
        op.execute(statement)
    op.execute(_BACKFILL)


def downgrade() -> None:
    """Drop the guards before the tables, and the functions with them.

    `drop_table` takes a trigger with it and leaves the function behind, so a downgrade
    that dropped only tables would make the next upgrade fail on `CREATE FUNCTION ...
    already exists`. That is the rollback nobody finds until an incident -- roll back,
    fix, roll forward, and the last step is where it breaks.
    """
    for statement in _DROP_GUARDS:
        op.execute(statement)
    op.drop_table("skill_version_retirement")
    op.drop_table("skill_version_file")
    op.drop_table("skill_version")
