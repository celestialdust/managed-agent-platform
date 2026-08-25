"""The `environment` table: one sandbox shape written, read, listed, retired, removed.

Every statement here is keyed by the tenant except one, and the exception is named
below. On a read the tenant is a WHERE clause rather than a check applied to what came
back, so another tenant's shape is absent from the result instead of fetched and then
dropped -- there is no moment at which this process holds a row it is not entitled to.

**A read resolves the LATEST revision, and that is the change everything else here is
built around.** The primary key is `(id, revision)`, so an id stands for a sequence of
rows rather than for one row, and "the row with this id" stopped being a question with a
single answer the moment an edit could append. Every read below therefore orders by
revision and takes one, or groups by id and takes the maximum. A statement that selected
on the id alone would return whichever row the plan reached first -- correct for as long
as nobody edits anything, and then silently wrong.

`insert` still does not name `revision`, so a create lands on 1 through the column's
server default. That is deliberate rather than left over: the route mints a fresh id per
registration, so the only revision it can mean is the first, and a second write to one
id through this method is a primary-key conflict rather than an edit that got in through
the wrong door. `insert_revision` is the door for an edit.

Reads return rows as mappings rather than parsed shapes. Parsing lives in one place
above this, so a rule added there applies to a row written before it existed rather than
to whichever adapter remembered to re-check.

**`sessions_referencing` is the one statement with no tenant term, and it reads a
different table.** A Session records the Environment it was created in nowhere but its
own `session.created` event, so the count a delete is refused on has to be read out of
`event_log`. It is left unscoped on purpose: an Environment id is only resolvable by the
tenant who owns it, so every Session naming one is that tenant's, and a tenant predicate
here could only ever make the number *smaller* than the set of Sessions a delete would
strand. The number is about rows the caller has already proved they own, and it is a
count rather than a list -- a list of Session ids is unbounded.

Bind parameter types are declared because these are textual statements: SQLAlchemy has
no column metadata to infer from, so asyncpg receives whatever Python object it was
handed, and a `list` bound to a json column is refused on the first insert.
`.columns(...)` is the same declaration in the other direction. Measured against this
dialect the result-type declarations change nothing today -- asyncpg decodes json on its
own -- but the repository requires them and a driver without that decoding would make
them load-bearing.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.core.ids import TenantId
from managed_agent.core.registration.environment import Environment, EnvironmentId
from managed_agent.core.vocabulary import lifecycle

_ROW_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "tenant_id": sa.Uuid(),
    "name": sa.Text(),
    "runtime_image": sa.Text(),
    "denied_paths": sa.JSON(),
    "allowed_domains": sa.JSON(),
    "revision": sa.Integer(),
    "archived_at": sa.TIMESTAMP(timezone=True),
}

# The listing carries one column the single read does not, and it is the position a page
# resumes after rather than a fact about the shape. See `_PAGE_HEAD` for which of an
# id's several `created_at_ms` values it is.
_LISTING_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    **_ROW_TYPES,
    "created_at_ms": sa.BigInteger(),
}

_SHAPE_COLUMNS = (
    "e.id, e.tenant_id, e.name, e.runtime_image, e.denied_paths, e.allowed_domains,"
    " e.revision, a.archived_at"
)
"""The select list both reads share, spelled once.

Two transcriptions of it would be free to diverge, and the shape they would diverge in
is the one that costs the most: a page whose rows carry a key the single read does not
makes the parse above hand back two different values for one Environment.
"""

_INSERT = sa.text(
    "INSERT INTO environment"
    " (id, tenant_id, name, runtime_image, denied_paths, allowed_domains)"
    " VALUES (:id, :tenant_id, :name, :runtime_image, :denied_paths,"
    " :allowed_domains)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant_id", type_=sa.Uuid()),
    sa.bindparam("denied_paths", type_=sa.JSON()),
    sa.bindparam("allowed_domains", type_=sa.JSON()),
)

# The next revision is computed in the same statement that writes it, so there is no
# window between reading the number and using it -- the pattern `agent_definition` uses
# in 0012. An aggregate with no GROUP BY returns one row over no rows, so this would
# write revision 1 for an id nobody registered -- which is why the route establishes
# existence first, under the tenant predicate this max deliberately does not apply.
#
# `max(revision)` is taken across the id and not across the pair, for that same reason
# spelled the other way: the revisions of one id are one sequence, and a tenant-filtered
# maximum would let two tenants each believe they hold revision 1 of one id.
_INSERT_REVISION = sa.text(
    "INSERT INTO environment"
    " (id, tenant_id, name, runtime_image, denied_paths, allowed_domains, revision)"
    " SELECT :id, :tenant_id, :name, :runtime_image, :denied_paths, :allowed_domains,"
    " coalesce(max(revision), 0) + 1 FROM environment WHERE id = :id"
    " RETURNING revision"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant_id", type_=sa.Uuid()),
    sa.bindparam("denied_paths", type_=sa.JSON()),
    sa.bindparam("allowed_domains", type_=sa.JSON()),
)

# `ORDER BY revision DESC LIMIT 1` is the whole of what makes this a read of an
# Environment rather than a read of one of its revisions.
_FETCH = (
    sa.text(
        f"SELECT {_SHAPE_COLUMNS}"
        " FROM environment e"
        " LEFT JOIN environment_archive a ON a.environment_id = e.id"
        " WHERE e.id = :id AND e.tenant_id = :tenant_id"
        " ORDER BY e.revision DESC LIMIT 1"
    )
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant_id", type_=sa.Uuid()),
    )
    .columns(**_ROW_TYPES)
)

# The same read, at a revision the caller names instead of at the newest one. This is
# what honours a Session's pin: the placement path reads the number out of the creation
# event and asks for exactly that row, so an edit that landed afterwards changes nothing
# about a Session already running.
#
# No archive predicate, deliberately. Retirement means no NEW Session may reference the
# Environment, and their own semantics are explicit that sessions in flight continue --
# so a pod for a Session created before the retirement is still placed. Refusing here
# would make archiving a way to kill live work.
_FETCH_REVISION = (
    sa.text(
        f"SELECT {_SHAPE_COLUMNS}"
        " FROM environment e"
        " LEFT JOIN environment_archive a ON a.environment_id = e.id"
        " WHERE e.id = :id AND e.tenant_id = :tenant_id AND e.revision = :revision"
    )
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant_id", type_=sa.Uuid()),
    )
    .columns(**_ROW_TYPES)
)

# The grouped subquery answers both halves of "which row stands for this id": the
# revision to show, and the millisecond the id was FIRST registered, which is what the
# page is ordered and resumed by. Ordering on the shown revision's own `created_at_ms`
# would move an Environment up the list when it was edited, and a keyset walk over a
# key that moves repeats one row and skips another.
#
# `include_archived` is a bound boolean rather than a second pair of statements: it is a
# filter value, unlike the keyset below, which is a position and must not be admitted as
# a nullable parameter.
_PAGE_HEAD = (
    f"SELECT {_SHAPE_COLUMNS}, newest.created_at_ms"
    " FROM environment e"
    " JOIN (SELECT id, max(revision) AS revision, min(created_at_ms) AS created_at_ms"
    " FROM environment WHERE tenant_id = :tenant_id GROUP BY id) newest"
    " ON newest.id = e.id AND newest.revision = e.revision"
    " LEFT JOIN environment_archive a ON a.environment_id = e.id"
    " WHERE e.tenant_id = :tenant_id"
    " AND (:include_archived OR a.environment_id IS NULL)"
)
_PAGE_TAIL = " ORDER BY newest.created_at_ms DESC, e.id DESC LIMIT :limit"

_FIRST_PAGE = (
    sa.text(f"{_PAGE_HEAD}{_PAGE_TAIL}")
    .bindparams(
        sa.bindparam("tenant_id", type_=sa.Uuid()),
        sa.bindparam("include_archived", type_=sa.Boolean()),
    )
    .columns(**_LISTING_TYPES)
)

# The keyset comparison is a row constructor rather than the three-way OR it expands
# into, which is the form `session_registry.py` uses and for the same reason.
_PAGE_AFTER = (
    sa.text(
        f"{_PAGE_HEAD} AND (newest.created_at_ms, e.id) < (:after_ms, :after_id)"
        f"{_PAGE_TAIL}"
    )
    .bindparams(
        sa.bindparam("tenant_id", type_=sa.Uuid()),
        sa.bindparam("include_archived", type_=sa.Boolean()),
        sa.bindparam("after_ms", type_=sa.BigInteger()),
        sa.bindparam("after_id", type_=sa.Uuid()),
    )
    .columns(**_LISTING_TYPES)
)

# The id is selected out of `environment` under the tenant predicate rather than taken
# from the caller, so retiring another tenant's Environment writes nothing without a
# second round trip: the SELECT returns no row and the INSERT inserts none.
# `ON CONFLICT DO NOTHING` is what makes a repeat idempotent, and `DISTINCT` is what
# keeps a table of several revisions from offering one id several times over.
_ARCHIVE = sa.text(
    "INSERT INTO environment_archive (environment_id)"
    " SELECT DISTINCT id FROM environment"
    " WHERE id = :id AND tenant_id = :tenant_id"
    " ON CONFLICT (environment_id) DO NOTHING"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant_id", type_=sa.Uuid()),
)

# Read after the insert rather than returned by it, because what the caller needs is the
# ORIGINAL timestamp and `RETURNING` yields a row only on the call that wrote one. The
# join is the tenant predicate: without it this would answer about another tenant's
# retirement for anyone holding the id.
_ARCHIVED_AT = (
    sa.text(
        "SELECT a.archived_at FROM environment_archive a"
        " JOIN environment e ON e.id = a.environment_id"
        " WHERE a.environment_id = :id AND e.tenant_id = :tenant_id LIMIT 1"
    )
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant_id", type_=sa.Uuid()),
    )
    .columns(archived_at=sa.TIMESTAMP(timezone=True))
)

# Every revision, in one statement, because a delete of an Environment is not a delete
# of a revision: leaving the earlier ones would leave the id resolvable at a shape older
# than the one the caller asked to remove.
_DELETE = sa.text(
    "DELETE FROM environment WHERE id = :id AND tenant_id = :tenant_id"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant_id", type_=sa.Uuid()),
)

# Run only after the delete above removed something, so it needs no tenant term of its
# own -- the row it removes belongs to an Environment this transaction has just proved
# was the caller's. Removed rather than kept: a record that an Environment was retired,
# for an Environment that no longer exists, answers a question nobody can ask.
_DELETE_ARCHIVE = sa.text(
    "DELETE FROM environment_archive WHERE environment_id = :id"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

# Sessions that named this Environment and have not stopped. `NOT EXISTS` rather than a
# join on the stop event, so a Session with several events contributes one row and not
# one per event; and the absence of a stop rather than the presence of a run, because a
# suspended Session is still a Session that can be resumed into this shape.
#
# A Session whose `session.created` event has aged out of retention is not counted, and
# that is the one direction this can be wrong in. It is stated rather than hidden: the
# same sweep that removed the event removed the only record of which Environment that
# Session named, so no query over this schema can answer for it.
_SESSIONS_REFERENCING = sa.text(
    "SELECT count(*) FROM event_log created"
    " WHERE created.type = :created_type"
    " AND created.payload->>'environment_id' = :environment_id"
    " AND NOT EXISTS (SELECT 1 FROM event_log ended"
    " WHERE ended.session_id = created.session_id AND ended.type = :stopped_type)"
).bindparams(sa.bindparam("environment_id", type_=sa.Text()))

_MAX_PAGE = 500
"""The largest page this adapter will serve.

A read with no ceiling materialises every row it matches into one list before the caller
sees the first. A larger `limit` is refused rather than reduced, for the reason
`session_registry.py` gives: a reduced page is a short page, and a short page is how
this port says the walk is over.
"""


class PostgresEnvironmentStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    def _values(self, environment: Environment) -> dict[str, object]:
        """One shape as bind parameters, for the two statements that write one.

        Shared so a column added to the table is bound identically by a create and by an
        edit. Two transcriptions would let an edit quietly drop a field and write it as
        its default, which reads as a tenant having cleared something they never named.
        """
        return {
            "id": environment.id,
            "tenant_id": environment.tenant_id,
            "name": environment.name,
            "runtime_image": environment.runtime_image,
            "denied_paths": list(environment.denied_paths),
            "allowed_domains": list(environment.allowed_domains),
        }

    async def insert(self, environment: Environment, /) -> None:
        """Write one shape as the first revision of a new id.

        Raises `sqlalchemy.exc.IntegrityError` if that id already holds revision 1. The
        caller mints a fresh id per registration, so a collision is a store fault and
        never an edit: an edit is `insert_revision`, which numbers itself, and an INSERT
        that silently became one would be the exact thing that makes an id stop meaning
        one shape.
        """
        async with self._engine.begin() as conn:
            await conn.execute(_INSERT, self._values(environment))

    async def insert_revision(self, environment: Environment, /) -> int:
        """Append the next revision of an id that already exists, and say which.

        Raises `sqlalchemy.exc.IntegrityError` when two edits of one id race: both
        compute the same next number and the second one to commit is refused. That is
        the outcome to want -- the alternative is one edit overwriting the other, which
        this table's own trigger exists to make impossible.
        """
        async with self._engine.begin() as conn:
            written = await conn.scalar(_INSERT_REVISION, self._values(environment))
        return int(str(written))

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        """The latest revision of one of this tenant's shapes, or None.

        None covers both "no such id" and "that id belongs to somebody else", because
        those two answers are the same answer: a caller that could tell them apart could
        enumerate other tenants' ids by asking.

        The row carries `revision` and `archived_at` beside the shape's own columns. The
        second is null for an Environment nobody has retired, which is not the same as
        absent -- an absent key would mean this store never asked.
        """
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        _FETCH, {"id": environment_id, "tenant_id": tenant_id}
                    )
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else dict(row)

    async def fetch_revision(
        self,
        environment_id: EnvironmentId,
        tenant_id: TenantId,
        revision: int,
        /,
    ) -> Mapping[str, object] | None:
        """One named revision of one of this tenant's shapes, or None.

        The read a pin needs. `fetch` answers "what does this id mean now", which is the
        right question for a caller about to start something and the wrong one for a
        caller resuming something already started -- an edit would silently move it into
        a sandbox nobody agreed to.

        None covers "no such id", "not this tenant's" and "no such revision", which are
        one answer for the reason `fetch` gives, plus a third of the same kind: a caller
        able to tell a missing revision from a missing id could count another tenant's
        edits.

        A retired Environment answers normally. Archiving refuses new Sessions and does
        not stop Sessions already running, so a pod for one of those is still placed.
        """
        async with self._engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        _FETCH_REVISION,
                        {
                            "id": environment_id,
                            "tenant_id": tenant_id,
                            "revision": revision,
                        },
                    )
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else dict(row)

    async def page(
        self,
        tenant_id: TenantId,
        after: tuple[int, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Mapping[str, object]]:
        """One page of this tenant's Environments, newest first, one row per id.

        The row shown for an id is its latest revision, and the position the page is
        ordered and resumed by is when that id was first registered -- so an edit
        changes what a row says and never where it sits.

        `after` is the `(created_at_ms, id)` of the last row the caller holds. A `limit`
        outside `1..500` is refused rather than clamped: a clamped page is a short page,
        and a short page is how this port says the walk is over.
        """
        if limit < 1 or limit > _MAX_PAGE:
            raise ValueError(
                f"page limit {limit} for tenant {tenant_id} is outside 1..{_MAX_PAGE}. "
                "Refused rather than clamped, because a clamped page is short and a "
                "short page means the walk is over."
            )
        parameters: dict[str, object] = {
            "tenant_id": tenant_id,
            "limit": limit,
            "include_archived": include_archived,
        }
        statement = _FIRST_PAGE
        if after is not None:
            statement = _PAGE_AFTER
            parameters["after_ms"] = after[0]
            parameters["after_id"] = after[1]
        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement, parameters)).mappings().all()
        return [dict(row) for row in rows]

    async def archive(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> datetime | None:
        """Retire this Environment, and return when it was retired.

        Returns the timestamp of the FIRST retirement, so a client whose call timed out
        retries and is told the same moment rather than a fresh one. None means the id
        is not this tenant's, or nothing was ever registered under it -- one answer, for
        the reason `fetch` gives.

        Both statements run in one transaction, so the read cannot land between the
        insert and a concurrent delete and answer about a row that is gone.
        """
        keys = {"id": environment_id, "tenant_id": tenant_id}
        async with self._engine.begin() as conn:
            await conn.execute(_ARCHIVE, keys)
            when = await conn.scalar(_ARCHIVED_AT, keys)
        if when is None:
            return None
        assert isinstance(when, datetime)
        return when

    async def delete(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> bool:
        """Remove every revision of this id, and say whether anything was removed.

        False for an id that was already gone and for one that is not this tenant's.
        Whoever calls this has already established that no Session references the
        Environment, which is what makes a hard delete safe: there is no history a
        removal could make unreadable.

        The retirement row goes with it, and only if the shape rows went first -- so a
        caller who is not the owner cannot reach it, and no record survives claiming an
        Environment that no longer exists was retired.
        """
        async with self._engine.begin() as conn:
            removed = await conn.execute(
                _DELETE, {"id": environment_id, "tenant_id": tenant_id}
            )
            if removed.rowcount == 0:
                return False
            await conn.execute(_DELETE_ARCHIVE, {"id": environment_id})
        return True

    async def sessions_referencing(self, environment_id: EnvironmentId, /) -> int:
        """How many Sessions created in this Environment have not stopped.

        Read out of the Event Log because that is the only place the pairing is written:
        the `session` row records what a Session was created with and the Environment is
        not one of those columns, so the `session.created` payload is the record.

        Counts a suspended Session, which is not an oversight: suspension is reversible,
        so such a Session can be resumed into this shape and a delete would strand it.
        """
        async with self._engine.connect() as conn:
            counted = await conn.scalar(
                _SESSIONS_REFERENCING,
                {
                    "created_type": lifecycle.SESSION_CREATED,
                    "stopped_type": lifecycle.SESSION_STOPPED,
                    "environment_id": str(environment_id),
                },
            )
        return int(str(counted))
