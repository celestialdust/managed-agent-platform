"""Reading a tenant's whole skill inventory out of Postgres, across both doors.

Three statements and no writes but one. The write is the id assignment a repository
submission needs, which is an INSERT of rows whose contents are a pure function of their
own key -- so it conflicts harmlessly on a resubmission and needs no read to be correct.
It gets one anyway, and the reason is in `assign_repository_ids`.

The listing is a `UNION ALL` over two tables rather than two pages the caller merges. A
merge in the caller cannot be made total without over-fetching both sides, and the
failure mode of getting it wrong is a listing that silently drops one origin's tail: a
caller walks to what looks like the end and has seen part of what it holds, with nothing
saying so. One keyset over the union has one end.

Every statement carries `tenant_id` as a term rather than as a filter over rows that
came back, which is what makes another tenant's skill absent from the answer instead of
fetched and dropped. On the repository side that term is applied to the id table and the
file table is reached by joining all four key columns, so a row can only be assembled
from two rows that already agree about whose it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.control.skills.inventory import (
    RepositorySkillHeld,
    RepositorySkillRow,
    SkillRow,
    UploadedSkillRow,
)
from managed_agent.core.ids import SkillId, TenantId
from managed_agent.core.registration.skill import repository_skill_id

_TENANT = sa.bindparam("tenant", type_=sa.Uuid())
_REPOSITORY = sa.bindparam("repository", type_=sa.Text())
_REVISION = sa.bindparam("revision", type_=sa.CHAR(40))

# ON CONFLICT DO NOTHING is the idempotency. The conflicting row necessarily holds the
# same id, because the id is a `uuid5` of the four columns the unique constraint is on
# -- so there is no branch here for "already assigned" and no second statement racing
# this one to find out.
_ASSIGN = sa.text(
    "INSERT INTO skill_repository_id"
    " (skill_id, tenant_id, repository, revision, name)"
    " VALUES (:skill_id, :tenant, :repository, :revision, :name)"
    " ON CONFLICT DO NOTHING"
).bindparams(
    sa.bindparam("skill_id", type_=sa.Uuid()),
    _TENANT,
    _REPOSITORY,
    _REVISION,
    sa.bindparam("name", type_=sa.Text()),
)

# The verification read. `= ANY(:names)` rather than an IN list built per call, so one
# prepared statement serves every arity and no submission's size changes the SQL.
_ASSIGNED = (
    sa.text(
        "SELECT name, skill_id FROM skill_repository_id"
        " WHERE tenant_id = :tenant AND repository = :repository"
        " AND revision = :revision AND name = ANY(:names)"
    )
    .bindparams(
        _TENANT,
        _REPOSITORY,
        _REVISION,
        sa.bindparam("names", type_=sa.ARRAY(sa.Text())),
    )
    .columns(name=sa.Text(), skill_id=sa.Uuid())
)

# The join is on all four key columns rather than on a surrogate, because there is no
# surrogate: `skill_repository_file` is keyed by the four and this table exists to give
# that key a handle. Reaching the file row any other way would mean trusting one of the
# four to be redundant.
_FILE_JOIN: Final = (
    " FROM skill_repository_id i"
    " JOIN skill_repository_file f"
    " ON f.tenant_id = i.tenant_id AND f.repository = i.repository"
    " AND f.revision = i.revision AND f.name = i.name"
)

_READ_ONE = (
    sa.text(
        "SELECT i.skill_id, f.name, f.description, f.body,"
        " f.repository, CAST(f.revision AS text) AS revision"
        f"{_FILE_JOIN}"
        " WHERE i.tenant_id = :tenant AND i.skill_id = :id"
    )
    .bindparams(_TENANT, sa.bindparam("id", type_=sa.Uuid()))
    .columns(
        skill_id=sa.Uuid(),
        name=sa.Text(),
        description=sa.Text(),
        body=sa.Text(),
        repository=sa.Text(),
        revision=sa.Text(),
    )
)

# `CAST(NULL AS text)` rather than `NULL::text`, and rather than a bare `NULL`. A bare
# NULL leaves the union's column type to be inferred from whichever arm Postgres reads
# first, which is not a thing to depend on; the `::` spelling collides with the bind
# pattern `text()` uses, which ends in a negative lookahead for a colon and would leave
# the cast unparsed.
#
# `revision` is CHAR(40) on its own table and text on the other arm, so it is cast on
# both sides of the union rather than only where it is null -- two arms disagreeing
# about a column's type is an error at plan time, and the cast that fixes it belongs
# where a reader can see both.
_UPLOADED_ARM: Final = (
    "SELECT id AS skill_id, name, description, display_name,"
    " CAST(NULL AS text) AS repository, CAST(NULL AS text) AS revision,"
    " 'upload' AS origin"
    " FROM skill WHERE tenant_id = :tenant"
)

_REPOSITORY_ARM: Final = (
    "SELECT i.skill_id, f.name, f.description,"
    " CAST(NULL AS text) AS display_name,"
    " f.repository, CAST(f.revision AS text) AS revision,"
    " 'repository' AS origin"
    f"{_FILE_JOIN}"
    " WHERE i.tenant_id = :tenant"
)

_UNION: Final = f"SELECT * FROM ({_UPLOADED_ARM} UNION ALL {_REPOSITORY_ARM}) inventory"
_PAGE_TAIL: Final = " ORDER BY name, skill_id LIMIT :limit"

_ROW_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "skill_id": sa.Uuid(),
    "name": sa.Text(),
    "description": sa.Text(),
    "display_name": sa.Text(),
    "repository": sa.Text(),
    "revision": sa.Text(),
    "origin": sa.Text(),
}

# Two statements rather than one taking a nullable keyset, which is the call every other
# paged read in this package makes. One statement would need an `:after_name IS NULL OR
# ...` arm, so every first page would carry a comparison against a parameter that is not
# a position and a reader could not tell which shape an execution was.
_FIRST_PAGE = sa.text(f"{_UNION}{_PAGE_TAIL}").bindparams(_TENANT).columns(**_ROW_TYPES)

# A row constructor rather than the three-way OR it expands into, so the boundary is one
# comparison. Strictly greater than, so the row the caller already holds is never handed
# out twice. No casts on the parameters, for the reason the union's casts are spelled
# with `CAST`: `:after_id::uuid` is not a parameter at all under `text()`'s bind
# pattern.
_PAGE_AFTER = (
    sa.text(f"{_UNION} WHERE (name, skill_id) > (:after_name, :after_id){_PAGE_TAIL}")
    .bindparams(
        _TENANT,
        sa.bindparam("after_name", type_=sa.Text()),
        sa.bindparam("after_id", type_=sa.Uuid()),
    )
    .columns(**_ROW_TYPES)
)

_MAX_PAGE: Final = 1024
"""The largest page this adapter will serve.

The same number `skill_registry.py` uses for the uploaded-only walk, and for the same
reason: the listing route asks for one row past the page to learn whether another
exists, so a bound equal to the published cap would refuse the one input the published
cap says is legal. Sharing the number rather than importing it, because these are two
reads whose bounds happen to agree today and nothing says they must -- a shared constant
would make a change to one silently move the other.
"""


class IdAssignmentDrifted(Exception):
    """A stored repository-skill id is not the one this code computes for that skill.

    The one failure that cannot be recovered from and must not be papered over.
    Migration `0028` backfilled every repository skill that existed when the table was
    added, using a namespace literal duplicated in `control/skills/inventory.py` because
    a migration cannot import from `src/`. If those two ever diverge, the rows are still
    there and every lookup computes a different id -- so a skill reads as missing rather
    than as broken, and a tenant is told their skill does not exist.

    Raised rather than logged, and raised on the write path where the disagreement is
    first visible, because the alternative is a submission that reports ids no read will
    ever resolve.
    """


class PostgresSkillInventory:
    """The inventory reads, against a real database.

    One engine and no other state, so a single instance serves every tenant. Satisfies
    `control.skill_inventory.SkillInventory` structurally; the composition root passes
    this and nothing adapts anything.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def assign_repository_ids(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        names: Sequence[str],
    ) -> tuple[tuple[str, SkillId], ...]:
        """Give every skill of this checkout its id, and prove the stored ids agree.

        The ids are computed here rather than read back, because they are a pure
        function of the key: a row that already exists holds the same id this computes,
        since the unique constraint is on exactly the four columns the function hashes.
        So the INSERT conflicts harmlessly and the answer is the computed pairs either
        way.

        **The verification read is not redundant, and it is the reason this method is
        not three lines.** The one thing determinism cannot prove is that the namespace
        and the fold in `control/skills/inventory.py` still match the ones migration
        `0028` backfilled with. If they have drifted, every backfilled row is
        unreachable and nothing else in the system can notice -- the row is present, the
        id computed for it is different, and the skill reads as absent. One SELECT per
        submission buys a loud failure instead of a silent one, on the path where the
        disagreement first becomes visible.

        An empty submission is answered without touching the database. Postgres would
        take an empty executemany and an `= ANY('{}')` happily, but a checkout with no
        skills is a submission the door already refuses and the round trip buys nothing.
        """
        if not names:
            return ()
        assigned = tuple(
            (name, repository_skill_id(tenant_id, repository, revision, name))
            for name in names
        )
        async with self._engine.begin() as conn:
            await conn.execute(
                _ASSIGN,
                [
                    {
                        "skill_id": skill_id,
                        "tenant": tenant_id,
                        "repository": repository,
                        "revision": revision,
                        "name": name,
                    }
                    for name, skill_id in assigned
                ],
            )
            stored = {
                str(row.name): SkillId(row.skill_id)
                for row in (
                    await conn.execute(
                        _ASSIGNED,
                        {
                            "tenant": tenant_id,
                            "repository": repository,
                            "revision": revision,
                            "names": [name for name, _ in assigned],
                        },
                    )
                ).all()
            }
        drifted = [
            (name, expected, stored.get(name))
            for name, expected in assigned
            if stored.get(name) != expected
        ]
        if drifted:
            name, expected, found = drifted[0]
            raise IdAssignmentDrifted(
                f"skill {name!r} of {repository} at {revision} is stored under id "
                f"{found} and this code computes {expected}. The id namespace or its "
                "fold has diverged from the one migration 0028 backfilled with, so "
                f"{len(drifted)} skill(s) of this checkout cannot be read back."
            )
        return assigned

    async def repository_skill_at(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> RepositorySkillHeld | None:
        """One repository skill in full, or None when this tenant holds no such id.

        The body is fetched here and nowhere else on this surface. This is the one read
        where it is worth the bytes: the caller named exactly this skill and there is
        one of it, where a listing would move most of a megabyte per page to drop it.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(_READ_ONE, {"tenant": tenant_id, "id": skill_id})
            ).one_or_none()
        if row is None:
            return None
        return RepositorySkillHeld(
            skill_id=SkillId(row.skill_id),
            name=str(row.name),
            description=str(row.description),
            body=str(row.body),
            repository=str(row.repository),
            revision=str(row.revision),
        )

    async def page(
        self,
        tenant_id: TenantId,
        after: tuple[str, SkillId] | None,
        limit: int,
    ) -> tuple[SkillRow, ...]:
        """One page across both origins, ordered by name then id.

        Neither index serves this sort: `skill_by_tenant` is `(tenant_id, id)` and
        `skill_repository_id_by_tenant` is `(tenant_id, name)`, so the union is sorted
        after it is assembled. That is the honest cost of ordering a named inventory by
        its names, over one tenant's rows rather than the table's, and an index on
        `(tenant_id, name, id)` on each side is what to add if it ever shows up in a
        plan.

        A `limit` outside the window raises rather than being clamped. A clamped page is
        a short page, and a short page is how this read says the walk is over -- so
        clamping would tell a caller it had seen everything it holds.

        The row's type carries its origin, decided here from the discriminator the union
        selects. An unexpected value raises rather than defaulting to one of the two:
        the column is written by this module's own SQL, so a third value means the
        statement and this function have been changed apart, and guessing which arm it
        belongs to would put a repository skill in the collection as an upload.
        """
        if limit < 1 or limit > _MAX_PAGE:
            raise ValueError(
                f"page limit {limit} for tenant {tenant_id} is outside 1..{_MAX_PAGE}. "
                "Refused rather than clamped, because a clamped page is short and a "
                "short page means the walk is over."
            )
        parameters: dict[str, object] = {"tenant": tenant_id, "limit": limit}
        statement = _FIRST_PAGE
        if after is not None:
            statement = _PAGE_AFTER
            parameters["after_name"] = after[0]
            parameters["after_id"] = after[1]
        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement, parameters)).all()
        return tuple(_row_of(row) for row in rows)


def _row_of(row: Any) -> SkillRow:
    """One result row as the type its origin says it is.

    Split out so the two arms are read side by side: the uploaded row carries a label
    and no checkout, the repository row carries a checkout and no label, and neither
    carries a field belonging to the other.
    """
    origin = str(row.origin)
    if origin == "upload":
        return UploadedSkillRow(
            skill_id=SkillId(row.skill_id),
            name=str(row.name),
            description=str(row.description),
            display_name=(None if row.display_name is None else str(row.display_name)),
        )
    if origin == "repository":
        return RepositorySkillRow(
            skill_id=SkillId(row.skill_id),
            name=str(row.name),
            description=str(row.description),
            repository=str(row.repository),
            revision=str(row.revision),
        )
    raise ValueError(
        f"the inventory union selected origin {origin!r}, which this reader has no "
        "arm for. The discriminator is written by this module's own SQL, so a "
        "value it does not know means the statement and the reader were changed "
        "apart."
    )
