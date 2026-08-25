"""Storing the facts fixed when a Session is created, and finding Sessions across them.

Two questions cannot be answered by folding one Session's Event Log, because both range
over Sessions rather than within one: which Sessions a tenant has, and in what order
they were created. This table answers those two and nothing else -- a Session's current
state is still a fold over its log on every read, and there is no state column here for
a reader to reach for instead.

The creation facts appear both in the `session.created` event and in this row, and
neither can drift from the other: both are written once from one parsed value in one
request, the log is append-only, and this table's trigger refuses an UPDATE outright.

Every statement carries `tenant_id`, and it is a term in the query rather than a check
performed on the way out. A filter that runs in the store cannot be forgotten by a
caller, and it is what makes another tenant's Session *absent* from a result rather than
fetched and then dropped.

Ordering is `(created_at_ms, id)` descending and paging is keyset rather than offset, so
a Session created during a walk lands ahead of the walk and cannot shift a page already
handed out. The pair is the key because two Sessions can share a millisecond, and a page
boundary falling between two equal keys is where an offset walk repeats a row or skips
one.

Bind parameter types are declared because these are textual statements and SQLAlchemy
has no column metadata to infer from: without one asyncpg receives a bare Python
object, refuses a `dict` bound to `jsonb` on the first insert, and compares a uuid
passed as text against whatever punctuation that spelling happened to use.

`.columns(...)` on the reads is the same declaration in the other direction, and it is
worth being exact about what it buys here: measured against this dialect it changes
nothing, because SQLAlchemy's asyncpg driver installs its own json codec and both JSON
columns already arrive decoded. Deleting it breaks no test and alters no value. It is
declared anyway -- the repository requires it, and a driver without that codec would
make it load-bearing -- but a reader should not believe it is what makes these reads
work today.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.ports import SessionNotVisible
from managed_agent.core.session.session import SessionRecord

_INSERT = sa.text(
    "INSERT INTO session"
    " (id, tenant_id, definition_id, definition_revision, grant_tools, scope,"
    "  budget_minor_units, budget_currency, retention_days)"
    " VALUES (:id, :tenant, :definition, :revision, :grant, :scope,"
    "  :budget, :currency, :retention)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("definition", type_=sa.Uuid()),
    sa.bindparam("grant", type_=sa.JSON()),
    sa.bindparam("scope", type_=sa.JSON()),
)

# Annotated rather than inferred: the values are five different TypeEngine subclasses
# and their join is a base too wide for `.columns(**...)` to accept.
_RECORD_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "tenant_id": sa.Uuid(),
    "definition_id": sa.Uuid(),
    "definition_revision": sa.Text(),
    "grant_tools": sa.JSON(),
    "scope": sa.JSON(),
    "budget_minor_units": sa.BigInteger(),
    "budget_currency": sa.Text(),
    "retention_days": sa.Integer(),
}

_FETCH = (
    sa.text(
        "SELECT id, tenant_id, definition_id, definition_revision, grant_tools, scope,"
        " budget_minor_units, budget_currency, retention_days FROM session"
        " WHERE id = :id AND tenant_id = :tenant"
    )
    .bindparams(
        sa.bindparam("id", type_=sa.Uuid()),
        sa.bindparam("tenant", type_=sa.Uuid()),
    )
    .columns(**_RECORD_TYPES)
)

_READ = (
    sa.text(
        "SELECT id, tenant_id, definition_id, definition_revision, grant_tools, scope,"
        " budget_minor_units, budget_currency, retention_days FROM session"
        " WHERE id = :id"
    )
    .bindparams(sa.bindparam("id", type_=sa.Uuid()))
    .columns(**_RECORD_TYPES)
)
"""The one statement here with no `tenant_id` term, and `read` below says why.

Kept as its own statement rather than as `_FETCH` with a nullable bind, because a
tenant-scoped read and a tenant-free one are two different capabilities and a single
statement whose scoping depends on whether an argument happened to be None is one a
caller can widen by passing nothing.
"""

_PAGE_HEAD = (
    "SELECT id, definition_id, definition_revision, created_at_ms FROM session"
    " WHERE tenant_id = :tenant"
)
_PAGE_TAIL = " ORDER BY created_at_ms DESC, id DESC LIMIT :limit"

_LISTING_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "definition_id": sa.Uuid(),
    "definition_revision": sa.Text(),
    "created_at_ms": sa.BigInteger(),
}

# Two statements rather than one taking a nullable keyset, which is the call
# `event_log_range.py` already makes for its bounded and unbounded reads. One statement
# would need an `:after_ms IS NULL OR ...` arm, so every first page would carry a
# comparison against a parameter that is not a position, and a reader could not tell
# which of the two shapes any given execution was.
_FIRST_PAGE = (
    sa.text(f"{_PAGE_HEAD}{_PAGE_TAIL}")
    .bindparams(sa.bindparam("tenant", type_=sa.Uuid()))
    .columns(**_LISTING_TYPES)
)

# The keyset comparison is a row constructor rather than the three-way OR it expands
# into, so PostgreSQL can walk `session_by_tenant_creation` straight to the boundary
# instead of filtering after the fact.
#
# There are no `::bigint` / `::uuid` casts on the parameters, and their absence is
# load-bearing rather than tidying. SQLAlchemy's `text()` will not see `:name` when a
# `::` cast follows it -- its bind pattern ends in a negative lookahead for a colon --
# so `:after_ms::bigint` is not a parameter at all: `.bindparams()` raises
# `ArgumentError` naming it, and without that call the literal text reaches the driver.
# The declared bindparam types below are what the casts were reaching for, carried
# through the one channel that works here.
_PAGE_AFTER = (
    sa.text(
        f"{_PAGE_HEAD} AND (created_at_ms, id) < (:after_ms, :after_id){_PAGE_TAIL}"
    )
    .bindparams(
        sa.bindparam("tenant", type_=sa.Uuid()),
        sa.bindparam("after_ms", type_=sa.BigInteger()),
        sa.bindparam("after_id", type_=sa.Uuid()),
    )
    .columns(**_LISTING_TYPES)
)

# The same keyset walk in the other direction, and three things make it one.
#
# `>=` rather than `>`, and the boundary row is INCLUDED. The position handed in is the
# page's own oldest row -- not the row before it -- because that is the only key a
# caller can name without having read the page it is asking for. A forward cursor is the
# last row of the page BEFORE the one it opens, so walking back from that same key with
# `>` would return that page minus its final row and one row from further up: a window
# shifted by one, which drifts a little further every page. Inclusive, the same key
# names exactly the page the caller came from.
#
# ASC rather than DESC, so the `LIMIT` keeps the rows ADJACENT to the position rather
# than the newest in the whole collection. A DESC sort with the same WHERE would return
# the top of the tenant's list on every backward page -- which looks like a page, and is
# a different page every time the tenant creates a Session.
#
# So the caller gets them oldest-first and reverses. The reversal is the route's, not
# this statement's: a store that answered in presentation order would put the extra row
# the route reads to detect a further page at whichever end the flip moved it to, and
# the route would have to know which. Walked order out, presentation order decided
# above.
_PAGE_ENDING_AT = (
    sa.text(
        f"{_PAGE_HEAD} AND (created_at_ms, id) >= (:oldest_ms, :oldest_id)"
        " ORDER BY created_at_ms ASC, id ASC LIMIT :limit"
    )
    .bindparams(
        sa.bindparam("tenant", type_=sa.Uuid()),
        sa.bindparam("oldest_ms", type_=sa.BigInteger()),
        sa.bindparam("oldest_id", type_=sa.Uuid()),
    )
    .columns(**_LISTING_TYPES)
)

# The largest page this adapter will serve. A read with no ceiling materialises every
# row it matches into one list before the caller sees the first, which for the Event Log
# was measured at 203.8 ms and ~17 MB against 50,000 rows, and 3.5 ms with a cap. A
# tenant's Session count grows the same way and has no natural bound.
#
# A larger `limit` is refused rather than reduced. Silently reducing it would hand back
# a page shorter than the caller asked for, and a short page is precisely how this port
# says "the walk is over" -- so the reduction would read as an ending and the caller
# would stop, having seen part of its Sessions and no sign that it had.
_MAX_PAGE = 500


def _record_from(row: sa.Row[Any]) -> SessionRecord:
    """One `session` row as the record it was created from.

    Shared by the two reads below rather than written out in each, because what a row
    means is one piece of knowledge: a second transcription is free to widen a type or
    forget the sort below, and the two reads would then hand back records that compare
    unequal for the same stored Session.
    """
    return SessionRecord(
        id=SessionId(row.id),
        tenant_id=TenantId(row.tenant_id),
        definition_id=DefinitionId(row.definition_id),
        definition_revision=str(row.definition_revision),
        grant=frozenset(row.grant_tools),
        # Sorted so two reads of one row are equal values: SessionRecord is frozen
        # and compared by value, and a mapping's iteration order is not part of the
        # row it was stored as.
        scope=tuple(sorted(dict(row.scope).items())),
        budget_minor_units=int(row.budget_minor_units),
        budget_currency=str(row.budget_currency),
        retention_days=int(row.retention_days),
    )


@dataclass(frozen=True, slots=True)
class SessionListing:
    """One row of a tenant's list, carrying the keyset the next page starts from.

    Frozen and slotted like every other row type here: a listing is a snapshot of
    creation facts, and nothing that holds one has any business editing it.
    """

    id: SessionId
    definition_id: DefinitionId
    definition_revision: str
    created_at_ms: int


class PostgresSessionRegistry:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create(self, record: SessionRecord) -> None:
        """Write the row that makes a Session findable.

        The Grant is sorted and the Scope written as a mapping, so two writes of one
        record produce equal stored documents: a `frozenset` and a `dict` both have an
        iteration order that is not part of the value, and `fetch` compares whole
        records.

        Raises `sqlalchemy.exc.IntegrityError` if the id is already taken. Not caught
        here, because ids are minted rather than chosen and a collision is a broken
        generator rather than a caller mistake -- turning it into a domain refusal would
        report a bug as a normal outcome.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                _INSERT,
                {
                    "id": record.id,
                    "tenant": record.tenant_id,
                    "definition": record.definition_id,
                    "revision": record.definition_revision,
                    "grant": sorted(record.grant),
                    "scope": dict(record.scope),
                    "budget": record.budget_minor_units,
                    "currency": record.budget_currency,
                    "retention": record.retention_days,
                },
            )

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        """That tenant's Session of that id, as the record it was created from.

        Raises `SessionNotVisible` for both "no such Session" and "belongs to another
        tenant", and the two are one refusal on purpose: distinguishable answers would
        let a caller holding an id learn whether it names another tenant's Session, by
        the shape of the refusal alone.

        Refused rather than answered with nothing, which is the same choice the Event
        Log range read makes for an inverted range: an empty answer is indistinguishable
        from a legitimate one, and a caller acting on it goes wrong quietly.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(_FETCH, {"id": session_id, "tenant": tenant_id})
            ).one_or_none()
        if row is None:
            raise SessionNotVisible(str(session_id))
        return _record_from(row)

    async def read(self, session_id: SessionId) -> SessionRecord:
        """That Session's creation facts, whoever owns it. Raises `SessionNotVisible`.

        The one read here that names no tenant, which is worth the paragraph it costs
        because every other statement in this module carries one and that is what makes
        another tenant's Session absent from an answer rather than fetched and dropped.

        The rule that protects is about *caller-supplied* ids: a caller holding an id
        must not learn from a refusal whether it names somebody else's Session. This
        answers no caller. Its one consumer is the control plane's own placement path,
        reached from a Turn route that has already fetched this Session for the
        authenticated tenant, and what it returns is used to scope every resolution
        that follows -- the record's own `tenant_id` is the tenant those reads use. A
        tenant argument here could only be one the placement path invented, and an
        invented tenant is a filter that agrees with itself by construction.

        `SessionNotVisible` for an absent row, the same refusal `fetch` gives, because
        the absence is the same absence; a second exception type would be a second
        thing every caller handles identically.
        """
        async with self._engine.connect() as conn:
            row = (await conn.execute(_READ, {"id": session_id})).one_or_none()
        if row is None:
            raise SessionNotVisible(str(session_id))
        return _record_from(row)

    async def page(
        self,
        tenant_id: TenantId,
        after: tuple[int, UUID] | None,
        limit: int,
    ) -> Sequence[SessionListing]:
        """One page of a tenant's Sessions, newest first, after the given keyset.

        `after` is the `(created_at_ms, id)` of the last row the caller already holds,
        as one value rather than two nullable arguments -- half a keyset is not a
        position, and a pair of parameters admits the half-set state as a value.

        At most `limit` rows come back and a short page means there is nothing further
        below it. A `limit` outside `1..500` is refused rather than clamped: a clamped
        page is a short page, a short page is how this method says the walk is over, and
        a caller reading one as the end would stop early with no sign that it had.
        """
        if limit < 1 or limit > _MAX_PAGE:
            raise ValueError(
                f"page limit {limit} for tenant {tenant_id} is outside 1..{_MAX_PAGE}. "
                "Refused rather than clamped, because a clamped page is short and a "
                "short page means the walk is over."
            )
        parameters: dict[str, object] = {"tenant": tenant_id, "limit": limit}
        if after is None:
            statement = _FIRST_PAGE
        else:
            statement = _PAGE_AFTER
            parameters["after_ms"] = after[0]
            parameters["after_id"] = after[1]
        async with self._engine.connect() as conn:
            result = await conn.execute(statement, parameters)
            return [
                SessionListing(
                    id=SessionId(row.id),
                    definition_id=DefinitionId(row.definition_id),
                    definition_revision=str(row.definition_revision),
                    created_at_ms=int(row.created_at_ms),
                )
                for row in result
            ]

    async def page_ending_at(
        self,
        tenant_id: TenantId,
        oldest: tuple[int, UUID],
        limit: int,
    ) -> Sequence[SessionListing]:
        """The page of a tenant's Sessions whose OLDEST row is the one given.

        Named for what it answers rather than for the direction it walks, because the
        direction is not the interesting half: `page` above takes the row before the
        page it returns, this takes the last row OF the page it returns, and only the
        second can be asked for by a caller holding a forward cursor. That cursor is the
        previous page's final row, so handing it back here reproduces that page exactly
        -- which is what paging backward means and why no separate "one before this" key
        has to be invented.

        Rows come back oldest-first -- walked order, not presentation order -- and are
        NOT flipped here. The caller asks for one row more than it will show, to learn
        whether the walk continues past this page, and that extra row is the newest of
        them; a store that pre-flipped would move it to the other end and make its
        position this store's business rather than the caller's.

        `oldest` is required rather than nullable, and that is the difference from
        `page`. There is no backward walk from nowhere: `None` would have to mean the
        oldest page, which is a forward walk from the far end and a different question.
        A caller with no position asks `page`.

        The same `1..500` refusal as `page`, for the same reason -- a clamped page is a
        short page, and a short page is how both of these say the walk is over.
        """
        if limit < 1 or limit > _MAX_PAGE:
            raise ValueError(
                f"page limit {limit} for tenant {tenant_id} is outside 1..{_MAX_PAGE}. "
                "Refused rather than clamped, because a clamped page is short and a "
                "short page means the walk is over."
            )
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _PAGE_ENDING_AT,
                {
                    "tenant": tenant_id,
                    "limit": limit,
                    "oldest_ms": oldest[0],
                    "oldest_id": oldest[1],
                },
            )
            return [
                SessionListing(
                    id=SessionId(row.id),
                    definition_id=DefinitionId(row.definition_id),
                    definition_revision=str(row.definition_revision),
                    created_at_ms=int(row.created_at_ms),
                )
                for row in result
            ]
