"""skill_eval_run: one row per graded (tenant, repository, revision). Never rewritten.

There is no baseline table. A repository's baseline per skill is max(score) over the
accepted runs, which these rows already hold; a stored copy would be a second source for
a fact this one derives exactly, free to drift from the runs it summarises with no way
to tell which of the two was lying.

scores and regressions are JSONB rather than JSON because the baseline read expands the
scores object with jsonb_each_text and the check constraints below call jsonb_typeof and
jsonb_array_length, none of which the json type supports.

An UPDATE is refused by a BEFORE UPDATE trigger that raises, which is the mechanism
migration 0001 chose and stated: a rewrite rule with DO INSTEAD NOTHING leaves the
stored row correct while reporting success to the writer that tried to change it.

Revision ID: 0017
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0017"
# Not "0017 minus one". Revision ids come from the plan and slices land in wave order
# rather than numeric order, so the parent is whatever head was live when this merged.
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "skill_eval_run",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("repository", sa.Text(), nullable=False),
        sa.Column("revision", sa.CHAR(40), nullable=False),
        sa.Column("scores", postgresql.JSONB(), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("regressions", postgresql.JSONB(), nullable=False),
        sa.Column(
            "submitted_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # One grading per revision. A second submission conflicts here rather than
        # writing a second row, which is what stops a refused revision being re-rolled
        # until it passes.
        sa.PrimaryKeyConstraint("tenant_id", "repository", "revision"),
        # `NOT NULL` on a jsonb column does not say what these two say: a JSON `null`,
        # a number and a string are all perfectly good jsonb *values*, and each would
        # make the length and expansion below fail somewhere far from here.
        sa.CheckConstraint(
            "jsonb_typeof(scores) = 'object' AND scores <> '{}'::jsonb",
            name="skill_eval_run_scores_is_a_non_empty_object",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(regressions) = 'array'",
            name="skill_eval_run_regressions_is_an_array",
        ),
        # Accepted exactly when nothing regressed. Held in the store so a writer that
        # gets it wrong produces an error, not a row that reads as an acceptance nobody
        # granted.
        sa.CheckConstraint(
            "accepted = (jsonb_array_length(regressions) = 0)",
            name="skill_eval_run_accepted_iff_no_regressions",
        ),
    )
    # The baseline read filters on accepted and groups within one repository; a partial
    # index keeps it off the refused rows, which are the ones that accumulate on a
    # repository somebody is actively fixing.
    op.create_index(
        "skill_eval_run_accepted_by_repository",
        "skill_eval_run",
        ["tenant_id", "repository"],
        postgresql_where=sa.text("accepted"),
    )
    op.execute(
        """
        CREATE FUNCTION skill_eval_run_refuse_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'skill_eval_run is append-only: the grade recorded for revision % may '
                'not be changed -- a verdict a caller can re-roll is not a gate',
                OLD.revision
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER skill_eval_run_no_update BEFORE UPDATE ON skill_eval_run
        FOR EACH ROW EXECUTE FUNCTION skill_eval_run_refuse_update();
        """
    )


def downgrade() -> None:
    """Remove everything `upgrade` created, the function included.

    The function is dropped explicitly because it is a schema object of its own: a
    downgrade that dropped only the table would leave it behind, and the next `upgrade`
    would fail on `CREATE FUNCTION ... already exists`. Dropping the table takes its
    trigger and its index with it, so those two are named only for the trigger, which
    must go before the function it calls.
    """
    op.execute("DROP TRIGGER IF EXISTS skill_eval_run_no_update ON skill_eval_run;")
    op.execute("DROP FUNCTION IF EXISTS skill_eval_run_refuse_update();")
    op.drop_index("skill_eval_run_accepted_by_repository", table_name="skill_eval_run")
    op.drop_table("skill_eval_run")
