"""Storing agent definitions and resolving the revision a Session pins.

`resolve` returns the latest revision's body together with the number it landed on. A
Session records that number, so a definition registered later cannot change what a
running Session is. Both halves come back because either alone is a half-answer: a body
with no number cannot be pinned, and a number with no body cannot be run.

The bind parameter types are declared rather than left to the driver, and that is not
decoration. These are textual statements, so SQLAlchemy has no column metadata to infer
from and asyncpg receives whatever Python object it is handed: a `dict` for a `jsonb`
column is refused outright, and a `str` for a `uuid` column is refused or, worse,
compared against a differently-punctuated spelling of the same id. `.columns(...)` on
the read is the same requirement in the other direction, and its effect here is
**measured to be nothing**: delete it and no test fails and no value changes, because
the asyncpg dialect decodes `jsonb` on its own. It stays as insurance against a driver
that does not, not as the mechanism that makes the read work. The sentence that used
to sit here claimed the opposite -- inherited from the bind-parameter rule above it,
which is real, and never checked against this module.

A concurrent second registration of the same id is caught by the primary key rather than
prevented by a lock. That is the right trade here and not the same trade the Event Log
append makes: sequences there must be contiguous, so a burnt number is a permanent gap
and the lock is load-bearing. Revisions only have to be distinct and increasing, the
route mints a fresh id for every first registration, and a loser sees an integrity error
rather than a wrong answer.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.types import TypeEngine

from managed_agent.control.catalog.definitions import AgentRecord
from managed_agent.control.skills.evaluation import (
    Baseline,
    EvalFacts,
    Grade,
    Regression,
    RunRecord,
)
from managed_agent.core.ids import DefinitionId, TenantId
from managed_agent.core.ports import UnknownDefinition
from managed_agent.core.registration.definition import AgentDefinition, VersionFact

# The next revision is computed from the rows in the same statement that writes one, so
# there is no window between reading the number and using it. An aggregate with no GROUP
# BY returns one row even over no rows, which is what makes the first registration land
# on 1 without a separate "does this exist" query.
#
# `max(revision)` is taken across the id without filtering on tenant, deliberately: the
# revision numbers of one id are one sequence, and a tenant-filtered max would let two
# tenants each believe they hold revision 1 of the same id. Reads are tenant-filtered;
# the numbering is not.
_INSERT = sa.text(
    "INSERT INTO agent_definition (id, tenant_id, revision, body, skills_revision)"
    " SELECT :id, :tenant, coalesce(max(revision), 0) + 1, :body, :skills"
    " FROM agent_definition WHERE id = :id"
    " RETURNING revision"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("body", type_=sa.JSON()),
)

_RESOLVE_LATEST = (
    sa.text(
        "SELECT revision, body FROM agent_definition"
        " WHERE id = :id AND tenant_id = :tenant ORDER BY revision DESC LIMIT 1"
    )
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant", type_=sa.Uuid()),
    )
    .columns(revision=sa.Integer(), body=sa.JSON())
)


# Registered and retired in one read, so the two cannot be observed out of step. A LEFT
# JOIN rather than two queries: a revision retired between them would come back as live,
# and a Session would start on a version somebody had just withdrawn.
_LIST_VERSIONS = (
    sa.text(
        "SELECT d.revision AS revision, (a.revision IS NOT NULL) AS archived"
        " FROM agent_definition d"
        " LEFT JOIN agent_version_archive a"
        " ON a.definition_id = d.id AND a.revision = d.revision"
        " WHERE d.id = :id AND d.tenant_id = :tenant"
        " ORDER BY d.revision"
    )
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant", type_=sa.Uuid()),
    )
    .columns(revision=sa.Integer(), archived=sa.Boolean())
)

_READ_VERSION = (
    sa.text(
        "SELECT body FROM agent_definition"
        " WHERE id = :id AND tenant_id = :tenant AND revision = :revision"
    )
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant", type_=sa.Uuid()),
    )
    .columns(body=sa.JSON())
)

# The key is selected out of `agent_definition` under the tenant predicate rather than
# taken from the caller, which is what makes retiring another tenant's version write
# nothing without a second round trip: the SELECT returns no row, so the INSERT inserts
# none. `ON CONFLICT DO NOTHING` is what makes a repeat idempotent, and `RETURNING` then
# yields a row only on the call that actually wrote one.
_ARCHIVE_VERSION = sa.text(
    "INSERT INTO agent_version_archive (definition_id, revision)"
    " SELECT id, revision FROM agent_definition"
    " WHERE id = :id AND tenant_id = :tenant AND revision = :revision"
    " ON CONFLICT DO NOTHING"
    " RETURNING revision"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
)


# An agent is a stack of revisions, and the two aggregates below are what turn the stack
# back into one thing: the highest revision is its current shape, and the earliest
# `registered_at` is when it came into being. Both in one CTE rather than two reads, so
# a registration landing between them cannot produce a version from after the read and
# a creation time from before it.
#
# The join back onto `agent_definition` carries no tenant term and does not need one:
# the CTE is already tenant-scoped, and `(id, revision)` is that table's primary key, so
# the pair it selects names exactly one row.
_AGENT_ONE_ROW_PER_AGENT = (
    "WITH agent AS ("
    " SELECT id, max(revision) AS revision, min(registered_at) AS created_at"
    " FROM agent_definition WHERE tenant_id = :tenant"
)
"""The head both whole-agent reads share, up to the point their filters differ.

Two statements rather than one with a switch, because the page walks a keyset and the
single read is a primary-key lookup -- one plan for both would be the wrong plan for
one of them. Shared as text so the fold that defines "an agent" is written once: a page
whose `version` meant something different from a read's would be a bug no assertion
about either alone could see.
"""

_AGENT_TAIL = (
    " SELECT a.id AS id, a.revision AS revision, a.created_at AS created_at,"
    " d.body AS body, r.archived_at AS archived_at"
    " FROM agent a"
    " JOIN agent_definition d ON d.id = a.id AND d.revision = a.revision"
    " LEFT JOIN agent_archive r ON r.definition_id = a.id"
)
"""The columns and joins both whole-agent reads share, once their filters have run."""

_AGENT_COLUMNS: dict[str, TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "revision": sa.Integer(),
    "created_at": sa.TIMESTAMP(timezone=True),
    "body": sa.JSON(),
    "archived_at": sa.TIMESTAMP(timezone=True),
}
"""What both whole-agent reads return. `body` is the one that has to be declared -- a
jsonb column with no type comes back as JSON text -- and the rest are declared beside it
so the shape is stated in one place rather than half-stated in two.

Worth knowing what the two adapter guards can and cannot see here.
`test_statements_declare_their_types.py` reads each statement's own source text, so it
sees the filters written inline below and not the fold these constants hold -- the
`SELECT` it looks for is in a name rather than in the segment. The fold is shared for
the reason the constants exist: what an agent's `version` and `created_at` ARE is one
rule, and two copies of it are free to disagree about which revision a page reports and
which one a read does. Declaring the returned types once, here, is the other half of
that trade."""

# Every optional filter is folded into a `coalesce` against an infinity rather than
# written as an `IS NULL OR ...` branch. The keyset boundary needs a sentinel either
# way -- there is no "after nothing" a row comparison can be omitted for -- so writing
# all three the same way makes the WHERE clause three rules a reader can read straight
# down instead of three conditionals to combine. `-infinity` and `infinity` are real
# timestamptz values, not strings PostgreSQL happens to accept.
#
# `CAST(:x AS ...)` and not `:x::...`: SQLAlchemy's own bind-parameter scan refuses to
# see a name followed by a colon, so the postfix cast would leave `:after_at` as
# literal text in the statement and the parameter unbound.
#
# The keyset is `(created_at, id)` and not `created_at` alone. Two agents registered in
# the same microsecond would otherwise share a position, and a page boundary landing
# between them repeats one row or drops the other.
_AGENT_PAGE = (
    sa.text(
        _AGENT_ONE_ROW_PER_AGENT + " GROUP BY id)" + _AGENT_TAIL + ""
        " WHERE (CAST(:include_archived AS boolean) OR r.archived_at IS NULL)"
        " AND a.created_at >="
        " coalesce(CAST(:created_from AS timestamptz), '-infinity')"
        " AND a.created_at <="
        " coalesce(CAST(:created_to AS timestamptz), 'infinity')"
        " AND (a.created_at, a.id) <"
        " (coalesce(CAST(:after_at AS timestamptz), 'infinity'),"
        " coalesce(CAST(:after_id AS uuid),"
        " '00000000-0000-0000-0000-000000000000'))"
        " ORDER BY a.created_at DESC, a.id DESC"
        " LIMIT :limit"
    )
    .bindparams(
        sa.bindparam("tenant", type_=sa.Uuid()),
        sa.bindparam("after_id", type_=sa.Uuid()),
        sa.bindparam("after_at", type_=sa.TIMESTAMP(timezone=True)),
        sa.bindparam("created_from", type_=sa.TIMESTAMP(timezone=True)),
        sa.bindparam("created_to", type_=sa.TIMESTAMP(timezone=True)),
        sa.bindparam("include_archived", type_=sa.Boolean()),
    )
    .columns(**_AGENT_COLUMNS)
)

_AGENT_READ = (
    sa.text(_AGENT_ONE_ROW_PER_AGENT + " AND id = :id GROUP BY id)" + _AGENT_TAIL)
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant", type_=sa.Uuid()),
    )
    .columns(**_AGENT_COLUMNS)
)

# Reads the retirement it just wrote, or the one that was already there, in one
# statement -- which is what lets the route be idempotent without a read-then-write
# race. The two branches are mutually exclusive by PostgreSQL's own rules rather than
# by luck: a data-modifying CTE's rows are invisible to the rest of the statement's
# snapshot, so the plain SELECT sees a row only when the retirement predates this call,
# and `ON CONFLICT DO NOTHING` returns one only when this call is what wrote it.
#
# That exclusivity is why a repeat gets the ORIGINAL timestamp. A fresh one would claim
# the agent was retired at the moment of a retry, which is a different and false fact
# about when anything referencing it stopped being resolvable.
#
# `owned` is what scopes the write. The key is selected out of `agent_definition` under
# a tenant predicate rather than taken from the caller, so retiring another tenant's
# agent inserts nothing and reads nothing back -- no second round trip, and no branch
# here that could be forgotten.
_ARCHIVE_AGENT = sa.text(
    "WITH owned AS ("
    " SELECT DISTINCT id FROM agent_definition"
    " WHERE id = :id AND tenant_id = :tenant"
    "), written AS ("
    " INSERT INTO agent_archive (definition_id) SELECT id FROM owned"
    " ON CONFLICT DO NOTHING RETURNING archived_at"
    ")"
    " SELECT archived_at FROM written"
    " UNION ALL"
    " SELECT r.archived_at FROM agent_archive r"
    " JOIN owned o ON o.id = r.definition_id"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
)

# `HAVING` is the whole of the optimistic-concurrency check, and it is what makes the
# check and the write one statement: an aggregate with no GROUP BY produces a single
# row, `HAVING` decides whether the SELECT feeding the INSERT yields it, and a caller
# working from a stale number therefore writes nothing rather than writing a revision
# the platform then has to take back.
#
# Over no rows -- an id nobody registered, or one belonging to another tenant --
# `max(revision)` is NULL and `NULL = :expected` is NULL, so the HAVING is not
# satisfied and nothing is written. Both refusals fall out of the same clause.
#
# The max is tenant-filtered here while `_INSERT` above deliberately does not filter
# it. The two are not in tension: an id is minted per creation and belongs to one
# tenant, so on the rows this can insert the filter changes no number, and on the rows
# it must not insert it is what refuses them.
_REGISTER_AT_REVISION = sa.text(
    "INSERT INTO agent_definition (id, tenant_id, revision, body, skills_revision)"
    " SELECT :id, :tenant, max(revision) + 1, :body, :skills"
    " FROM agent_definition WHERE id = :id AND tenant_id = :tenant"
    " HAVING max(revision) = :expected"
    " RETURNING revision"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("body", type_=sa.JSON()),
)


# The baseline per skill is derived, not stored. `DISTINCT ON (s.key)` with this ORDER
# BY takes the highest score each skill has reached on an accepted run; the two trailing
# sort terms decide a tie in favour of the *earliest* run to reach it, so
# `set_by_revision` names the revision that set the bar rather than the last one to
# match it, and two runs sharing a timestamp resolve the same way on every read.
#
# `jsonb_each_text` is why the column is jsonb and not json: json cannot be expanded.
_EVAL_BASELINES = (
    sa.text(
        "SELECT DISTINCT ON (s.key) s.key AS skill, s.value::int AS score,"
        " r.revision AS set_by_revision"
        " FROM skill_eval_run r, jsonb_each_text(r.scores) AS s"
        " WHERE r.tenant_id = :tenant AND r.repository = :repository AND r.accepted"
        " ORDER BY s.key, s.value::int DESC, r.submitted_at ASC, r.revision ASC"
    )
    .bindparams(sa.bindparam("tenant", type_=sa.Uuid()))
    .columns(skill=sa.Text(), score=sa.Integer(), set_by_revision=sa.Text())
)

# `scores` and `regressions` are a mapping and a list of mappings, and a textual
# statement carries no column metadata for asyncpg to infer from -- both are refused on
# the first insert without the declaration below. The names are domain words rather than
# `payload` or `body`, which is exactly why the declaration has to be deliberate here.
_INSERT_EVAL_RUN = (
    sa.text(
        "INSERT INTO skill_eval_run"
        " (tenant_id, repository, revision, scores, accepted, regressions)"
        " VALUES (:tenant, :repository, :revision, :scores, :accepted, :regressions)"
        " ON CONFLICT (tenant_id, repository, revision) DO NOTHING"
        " RETURNING accepted, regressions"
    )
    .bindparams(
        sa.bindparam("tenant", type_=sa.Uuid()),
        sa.bindparam("scores", type_=sa.JSON()),
        sa.bindparam("regressions", type_=sa.JSON()),
    )
    .columns(accepted=sa.Boolean(), regressions=sa.JSON())
)

_READ_EVAL_RUN = (
    sa.text(
        "SELECT accepted, regressions FROM skill_eval_run"
        " WHERE tenant_id = :tenant AND repository = :repository"
        " AND revision = :revision"
    )
    .bindparams(sa.bindparam("tenant", type_=sa.Uuid()))
    .columns(accepted=sa.Boolean(), regressions=sa.JSON())
)

# One read for both facts. Enrollment counts every run and acceptance only the accepted
# ones, so a repository that failed its first submission is still under the gate.
_EVAL_FACTS = (
    sa.text(
        "SELECT count(*) > 0 AS repository_enrolled,"
        " count(*) FILTER (WHERE accepted AND revision = :revision) > 0"
        " AS revision_accepted"
        " FROM skill_eval_run WHERE tenant_id = :tenant AND repository = :repository"
    )
    .bindparams(sa.bindparam("tenant", type_=sa.Uuid()))
    .columns(repository_enrolled=sa.Boolean(), revision_accepted=sa.Boolean())
)


def _regression_as_json(regression: Regression) -> dict[str, object]:
    """One regression as the object the `regressions` array holds.

    `scored` is written as JSON null when the run omitted the skill entirely, so the
    stored array has one shape for both kinds of regression and the reader below needs
    no branch. The distinction survives: null is not a score, and the reader below
    brings it back as `None`.
    """
    return {
        "skill": regression.skill,
        "baseline": regression.baseline,
        "scored": regression.scored,
        "set_by_revision": regression.set_by_revision,
    }


def _regression_from_json(stored: object) -> Regression:
    """One element of the stored `regressions` array, back as a typed value.

    A missing `scored` key reads the same as a stored null, which is what keeps rows
    written before that field was always emitted readable.
    """
    assert isinstance(stored, dict)
    scored = stored.get("scored")
    return Regression(
        skill=str(stored["skill"]),
        baseline=int(stored["baseline"]),
        scored=None if scored is None else int(scored),
        set_by_revision=str(stored["set_by_revision"]),
    )


def _run_record(
    accepted: object, regressions: object, first_grading: bool
) -> RunRecord:
    """The stored verdict for one revision, whichever statement returned it.

    Built from the columns rather than from the `Grade` the caller passed in, so a
    repeat submission is answered out of the row and never out of a second grading.
    """
    assert isinstance(regressions, list)
    return RunRecord(
        accepted=bool(accepted),
        regressions=tuple(_regression_from_json(item) for item in regressions),
        first_grading=first_grading,
    )


@dataclass(frozen=True, slots=True)
class Resolved:
    """What a definition resolved to, and the revision number that identifies it."""

    definition: AgentDefinition
    revision: int


class PostgresDefinitionRegistry:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        """Write a new revision of `definition_id` and return the number it was given.

        Returned rather than derived by the caller: a caller counting its own
        registrations is right until anything else registers, and the number is what a
        Session pins.

        Raises `sqlalchemy.exc.IntegrityError` if a concurrent writer took the same
        revision. The caller may retry; it must never pick a revision itself, because a
        chosen revision is how one gets overwritten.
        """
        async with self._engine.begin() as conn:
            row = await conn.execute(
                _INSERT,
                {
                    "id": definition_id,
                    "tenant": tenant_id,
                    "body": definition.model_dump(mode="json"),
                    "skills": definition.skills_revision,
                },
            )
            return int(row.scalar_one())

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolved:
        """The latest revision of a tenant's definition, and its number.

        Raises `UnknownDefinition` when the tenant has no such definition -- refused
        rather than answered with a default-shaped one, which would start a Session with
        no instructions and show the tenant an agent that does nothing instead of a
        refusal naming what is missing.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _RESOLVE_LATEST, {"id": definition_id, "tenant": tenant_id}
                )
            ).one_or_none()
        if row is None:
            raise UnknownDefinition(str(definition_id))
        return Resolved(AgentDefinition.model_validate(row.body), int(row.revision))

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> tuple[VersionFact, ...]:
        """Every revision this tenant owns of one agent, ascending, with its state.

        Empty both for an id nobody registered and for an id belonging to somebody
        else. The tenant is a term in the query rather than a check performed on the
        result, so another tenant's rows are absent rather than fetched and dropped --
        a filter that runs in the store cannot be forgotten at a call site.
        """
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    _LIST_VERSIONS, {"id": definition_id, "tenant": tenant_id}
                )
            ).all()
        return tuple(
            VersionFact(revision=int(row.revision), archived=bool(row.archived))
            for row in rows
        )

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        """One exact revision's body, or None when this tenant has no such revision.

        `None` rather than a raise, because the caller reaching this has already chosen
        the revision from `list_versions`: an absent body there is a missing row rather
        than a missing version, and it is worth telling those apart upstream.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _READ_VERSION,
                    {"id": definition_id, "tenant": tenant_id, "revision": revision},
                )
            ).one_or_none()
        return None if row is None else AgentDefinition.model_validate(row.body)

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        """Retire one revision; True when this call is what retired it.

        False covers two states a caller acts on identically and neither of which
        writes anything: the version was already retired, and the version is not this
        tenant's.

        Retirement is a row in a table of its own rather than a column here.
        `agent_definition` refuses an UPDATE by trigger, so a retirement written as a
        column update would not quietly fail to take -- it could not be recorded at all.
        """
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    _ARCHIVE_VERSION,
                    {"id": definition_id, "tenant": tenant_id, "revision": revision},
                )
            ).one_or_none()
        return row is not None

    async def page_agents(
        self,
        tenant_id: TenantId,
        *,
        include_archived: bool,
        created_from: datetime | None,
        created_to: datetime | None,
        after: tuple[datetime, DefinitionId] | None,
        limit: int,
    ) -> tuple[AgentRecord, ...]:
        """One page of a tenant's agents, newest first.

        Newest first because a caller looking for an agent it just made should find it
        on the first page, and because the walk then never has to revisit a page it has
        already read: a registration that lands mid-walk appears ahead of the position
        the caller holds rather than inside it.

        `after` is the last row of the previous page and is exclusive. Both halves are
        needed -- two agents registered in the same microsecond share a timestamp, and a
        boundary naming only the timestamp cannot say which of them the caller already
        has.

        `include_archived` is a parameter rather than a default because the two answers
        are both correct for different callers: a console listing wants live agents, and
        an audit of what a tenant ever had wants all of them. Which one is the default
        is the surface's decision, not this one's.

        Returns at most `limit` rows and says nothing about whether more exist. The
        caller asks for one more than it means to show and reads the answer off the
        count, which is the only way to know without a second query.
        """
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    _AGENT_PAGE,
                    {
                        "tenant": tenant_id,
                        "include_archived": include_archived,
                        "created_from": created_from,
                        "created_to": created_to,
                        "after_at": None if after is None else after[0],
                        "after_id": None if after is None else after[1],
                        "limit": limit,
                    },
                )
            ).all()
        return tuple(
            AgentRecord(
                definition_id=DefinitionId(row.id),
                version=int(row.revision),
                created_at=row.created_at,
                archived_at=row.archived_at,
                definition=AgentDefinition.model_validate(row.body),
            )
            for row in rows
        )

    async def read_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> AgentRecord | None:
        """One agent as a whole, or None when this tenant has no agent by that id.

        A retired agent is returned rather than withheld, with `archived_at` set.
        Retirement makes an agent unusable, not invisible: a caller holding a Session
        that resolved it needs to be able to find out what it was and why it can no
        longer be started, and answering "no such agent" would send them looking for a
        typo in an id that is correct.

        None both for an id nobody registered and for one belonging to somebody else.
        Telling those apart would let anyone holding an id learn from the answer whether
        it names another tenant's agent.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _AGENT_READ, {"id": definition_id, "tenant": tenant_id}
                )
            ).one_or_none()
        if row is None:
            return None
        return AgentRecord(
            definition_id=definition_id,
            version=int(row.revision),
            created_at=row.created_at,
            archived_at=row.archived_at,
            definition=AgentDefinition.model_validate(row.body),
        )

    async def archive_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> datetime | None:
        """Retire the agent and say when; None when the agent is not this tenant's.

        Retirement is terminal and there is no way back: `agent_archive` refuses an
        UPDATE by raising and nothing deletes from it, so this is the last state
        transition the agent has.

        A second call is not an error and does not write a second row. It returns the
        timestamp the FIRST call wrote, which is the fact a retry needs: a fresh
        timestamp would say the agent was retired at the moment of the retry, and
        anything reasoning about when references to it stopped resolving would be wrong
        by the length of the outage.

        None means the id is not this tenant's -- either nobody registered it or
        somebody else did. Nothing was written in that case, because the row it would
        have written is selected out of `agent_definition` under a tenant predicate.
        """
        async with self._engine.begin() as conn:
            return (
                await conn.execute(
                    _ARCHIVE_AGENT, {"id": definition_id, "tenant": tenant_id}
                )
            ).scalar_one_or_none()

    async def register_at_revision(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
        expected: int,
    ) -> int | None:
        """Append a revision only while `expected` is still the newest number.

        The comparison and the write are one statement, so there is no window between
        them. That is the difference between this and a read followed by `register`: two
        callers who both read revision 3 would both append, and the second would
        silently overwrite the first's intent with a revision built from a body that
        never saw it.

        None covers three outcomes a caller acts on identically -- a stale `expected`,
        an id that is not this tenant's, and a concurrent writer that took the next
        number first. The third arrives as an integrity error rather than as an empty
        result, because both writers computed the same revision and the primary key is
        what refuses the loser; it is caught here so a lost race reads as a version
        conflict rather than as a fault.
        """
        try:
            async with self._engine.begin() as conn:
                row = (
                    await conn.execute(
                        _REGISTER_AT_REVISION,
                        {
                            "id": definition_id,
                            "tenant": tenant_id,
                            "body": definition.model_dump(mode="json"),
                            "skills": definition.skills_revision,
                            "expected": expected,
                        },
                    )
                ).one_or_none()
        except IntegrityError:
            return None
        return None if row is None else int(row.revision)

    async def eval_baselines(
        self, tenant_id: TenantId, repository: str
    ) -> tuple[Baseline, ...]:
        """The bar every skill in one repository currently has to clear.

        Derived from the accepted runs rather than read from a baseline table, because
        the runs already hold the fact exactly and a stored summary of them would be a
        second source free to disagree with the rows it summarises.

        `set_by_revision` names the revision that *set* the bar, not the last one to
        match it: the ORDER BY breaks a tie on score by the earliest submission, so a
        later run scoring exactly the same leaves the attribution where it was. The
        trailing sort on the revision itself makes two runs sharing a timestamp resolve
        the same way on every read instead of arbitrarily.

        Empty for a repository nobody has submitted a run for, which is the same answer
        as for a repository that does not exist -- nothing here can distinguish the two,
        and no caller can act differently on them.
        """
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    _EVAL_BASELINES, {"tenant": tenant_id, "repository": repository}
                )
            ).all()
        return tuple(
            Baseline(
                skill=str(row.skill),
                score=int(row.score),
                set_by_revision=str(row.set_by_revision),
            )
            for row in rows
        )

    async def record_eval_run(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        scores: Mapping[str, int],
        graded: Grade,
    ) -> RunRecord:
        """Write this grading, or report the one already recorded for that revision.

        The refused runs are written too. A refusal that left no row would let the same
        revision be submitted again later against a baseline that had since moved, and
        the second answer would silently replace the first.

        A repeat submission takes the ON CONFLICT DO NOTHING branch and the verdict
        comes back out of the stored row, so the grade this call computed is discarded.
        That is what makes the gate un-re-rollable: a CI job retried for reasons that
        have nothing to do with the code under grading gets the original answer rather
        than a second opinion. A conflict rather than a caught integrity error, because
        a retried job is an expected call and not an exception.

        Both statements run in one transaction, so the row read on the conflict branch
        is never one a concurrent writer has yet to commit.
        """
        parameters: dict[str, object] = {
            "tenant": tenant_id,
            "repository": repository,
            "revision": revision,
            "scores": dict(scores),
            "accepted": graded.accepted,
            "regressions": [_regression_as_json(r) for r in graded.regressions],
        }
        async with self._engine.begin() as conn:
            inserted = (await conn.execute(_INSERT_EVAL_RUN, parameters)).one_or_none()
            if inserted is not None:
                return _run_record(inserted.accepted, inserted.regressions, True)
            held = (await conn.execute(_READ_EVAL_RUN, parameters)).one()
        return _run_record(held.accepted, held.regressions, False)

    async def eval_facts(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> EvalFacts:
        """Whether the gate applies to this repository, and whether it took this
        revision.

        Both facts come from one read, so a pin cannot see an enrollment without the
        acceptance that arrived with it.

        Enrollment counts **every** run while acceptance counts only the accepted ones.
        That asymmetry is the point: submitting a failing run turns the gate on for the
        repository, so the way past it cannot be to fail once and then stop calling.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _EVAL_FACTS,
                    {
                        "tenant": tenant_id,
                        "repository": repository,
                        "revision": revision,
                    },
                )
            ).one()
        return EvalFacts(
            repository_enrolled=bool(row.repository_enrolled),
            revision_accepted=bool(row.revision_accepted),
        )
