"""The skill store over Postgres: uploaded skills, and one checkout's skill directory.

Two relations, two access shapes, one adapter. Nothing here decides anything -- what a
valid skill is was settled where it was parsed, and which skills a definition resolves
to is settled in `control/skills/registry.py`. This reads and writes rows.

The tenant is a term in every WHERE clause rather than a check applied to rows that came
back. That is what makes another tenant's skill id read as absent instead of forbidden,
so a caller holding an id learns nothing from the answer.

Both tables are append-only in the schema, which is what lets the two writes here be
plain inserts with no read-then-write anywhere. An upload gets a fresh id and cannot
collide. A repository submission conflicts on its primary key and does nothing, so a
retried CI job is a no-op rather than an error or a duplicate -- and the row count it
reports is how the caller tells its own retry from a first submission.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.control.skills.registry import (
    SkillHeld,
    SkillListing,
    SkillRecord,
    SkillVersionBundle,
    SkillVersionCollision,
    SkillVersionFile,
    SkillVersionRecord,
)
from managed_agent.core.ids import SkillId, TenantId
from managed_agent.core.registration.skill import ValidatedSkill

_TENANT = sa.bindparam("tenant", type_=sa.Uuid())

_ADD_SKILL = sa.text(
    "INSERT INTO skill (id, tenant_id, name, description, body, display_name)"
    " VALUES (:id, :tenant, :name, :description, :body, :display_name)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    _TENANT,
    sa.bindparam("name", type_=sa.Text()),
    sa.bindparam("description", type_=sa.Text()),
    sa.bindparam("body", type_=sa.Text()),
    # Nullable, and passed rather than defaulted: a skill registered without a
    # label stores NULL, which is a different fact from a label that happens to
    # equal the name. Defaulting it in SQL would make the two indistinguishable.
    sa.bindparam("display_name", type_=sa.Text()),
)

# `= ANY(:ids)` rather than an `IN` list built per call, so one prepared statement
# serves every arity and no request shape changes the SQL.
_READ_SKILLS = (
    sa.text(
        "SELECT id, name, description, body FROM skill"
        " WHERE tenant_id = :tenant AND id = ANY(:ids) ORDER BY name, id"
    )
    .bindparams(_TENANT, sa.bindparam("ids", type_=sa.ARRAY(sa.Uuid())))
    .columns(id=sa.Uuid(), name=sa.Text(), description=sa.Text(), body=sa.Text())
)

# The listing read. `body` is deliberately absent from the SELECT: it is the one column
# that can be 32 KiB, nothing on the listing surface shows it, and fetching it would
# move most of a megabyte per page out of the database in order to be dropped.
_PAGE_HEAD = (
    "SELECT id, name, description, display_name FROM skill WHERE tenant_id = :tenant"
)
_PAGE_TAIL = " ORDER BY name, id LIMIT :limit"

_LISTING_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "name": sa.Text(),
    "description": sa.Text(),
    "display_name": sa.Text(),
}

# Two statements rather than one taking a nullable keyset, which is the call the session
# registry already makes for the same shape. One statement would need an
# `:after_name IS NULL OR ...` arm, so every first page would carry a comparison against
# a parameter that is not a position, and a reader could not tell which shape any given
# execution was.
_FIRST_PAGE = (
    sa.text(f"{_PAGE_HEAD}{_PAGE_TAIL}").bindparams(_TENANT).columns(**_LISTING_TYPES)
)

# A row constructor rather than the three-way OR it expands into, so the boundary is one
# comparison rather than a filter applied after the fact. Strictly greater than, so the
# row the caller already holds is never handed out a second time.
#
# No `::text` or `::uuid` casts on the parameters, and their absence is load-bearing:
# SQLAlchemy's `text()` bind pattern ends in a negative lookahead for a colon, so
# `:after_id::uuid` is not a parameter at all and `.bindparams()` raises naming it. The
# declared types below are what a cast would have been reaching for.
_PAGE_AFTER = (
    sa.text(f"{_PAGE_HEAD} AND (name, id) > (:after_name, :after_id){_PAGE_TAIL}")
    .bindparams(
        _TENANT,
        sa.bindparam("after_name", type_=sa.Text()),
        sa.bindparam("after_id", type_=sa.Uuid()),
    )
    .columns(**_LISTING_TYPES)
)

# The largest page this adapter will serve. A read with no ceiling materialises every
# matching row before the caller sees the first, and a tenant's uploaded-skill count has
# no natural bound: a skill is immutable and never deleted, so every edit of one leaves
# another row here for good.
#
# 1024 rather than the 1000 the tenant surface publishes, and the gap is arithmetic
# rather than slack. The listing route asks for `limit + 1` -- one probe row past the
# page, to learn whether another page exists without a second query -- so a bound equal
# to the published cap would refuse the one input the published cap says is legal. This
# was 500 while the surface published 100, which hid the relationship; raising the
# published cap to the reference's 1000 exposed it. `_MAX_VERSION_PAGE` below is 1024
# against a published 1000 for exactly this reason, and the two are the same rule.
_MAX_PAGE = 1024

# ON CONFLICT DO NOTHING is the idempotency, and `RETURNING name` is how many rows it
# actually wrote: a conflicting row returns nothing, so the count distinguishes a first
# submission from a retry without a second query racing the first.
_ADD_REPOSITORY_SKILL = sa.text(
    "INSERT INTO skill_repository_file"
    " (tenant_id, repository, revision, name, description, body)"
    " VALUES (:tenant, :repository, :revision, :name, :description, :body)"
    " ON CONFLICT DO NOTHING RETURNING name"
).bindparams(
    _TENANT,
    sa.bindparam("repository", type_=sa.Text()),
    sa.bindparam("revision", type_=sa.CHAR(40)),
    sa.bindparam("name", type_=sa.Text()),
    sa.bindparam("description", type_=sa.Text()),
    sa.bindparam("body", type_=sa.Text()),
)

_READ_REPOSITORY_SKILLS = (
    sa.text(
        "SELECT name, description, body FROM skill_repository_file"
        " WHERE tenant_id = :tenant AND repository = :repository"
        " AND revision = :revision ORDER BY name"
    )
    .bindparams(
        _TENANT,
        sa.bindparam("repository", type_=sa.Text()),
        sa.bindparam("revision", type_=sa.CHAR(40)),
    )
    .columns(name=sa.Text(), description=sa.Text(), body=sa.Text())
)


# The version reads and writes. `skill_version` and its two companions are keyed by
# `(skill_id, version)` and carry no tenant column, so every statement below reaches the
# tenant through `skill` -- either as a join or as the SELECT an INSERT draws its row
# from. That keeps the property the rest of this adapter has: another tenant's skill id
# is absent from the answer rather than fetched and dropped, and no statement here can
# be made to touch a row by supplying an id alone.

_ID = sa.bindparam("id", type_=sa.Uuid())
_VERSION = sa.bindparam("version", type_=sa.BigInteger())

# `latest_version` is a correlated subquery rather than a stored column, because every
# table behind it refuses an UPDATE and a stored pointer would need one. It excludes
# retired versions, so retiring the newest moves this to the newest survivor and
# retiring all of them leaves it null -- a skill that is still readable and no longer
# resolvable, which is the honest description of that state.
_READ_SKILL = (
    sa.text(
        "SELECT s.id, s.name, s.description, s.display_name,"
        " (SELECT max(v.version) FROM skill_version v"
        " LEFT JOIN skill_version_retirement r"
        " ON r.skill_id = v.skill_id AND r.version = v.version"
        " WHERE v.skill_id = s.id AND r.skill_id IS NULL) AS latest_version,"
        " (d.skill_id IS NOT NULL) AS deleted"
        " FROM skill s LEFT JOIN skill_deletion d ON d.skill_id = s.id"
        " WHERE s.tenant_id = :tenant AND s.id = :id"
    )
    .bindparams(_TENANT, _ID)
    .columns(
        id=sa.Uuid(),
        name=sa.Text(),
        description=sa.Text(),
        display_name=sa.Text(),
        latest_version=sa.BigInteger(),
        deleted=sa.Boolean(),
    )
)

# INSERT ... SELECT rather than VALUES, so the tenant is a term in the write as well as
# in the reads: a skill id belonging to somebody else selects no row and writes nothing.
# ON CONFLICT DO NOTHING is what makes a retried delete answer the same way twice with
# the moment the first one recorded.
_DELETE_SKILL = sa.text(
    "INSERT INTO skill_deletion (skill_id)"
    " SELECT id FROM skill WHERE id = :id AND tenant_id = :tenant"
    " ON CONFLICT DO NOTHING"
).bindparams(_ID, _TENANT)

_ADD_VERSION = sa.text(
    "INSERT INTO skill_version"
    " (skill_id, version, name, description, body, directory)"
    " SELECT id, :version, :name, :description, :body, :directory"
    " FROM skill WHERE id = :id AND tenant_id = :tenant"
).bindparams(
    _ID,
    _TENANT,
    _VERSION,
    sa.bindparam("name", type_=sa.Text()),
    sa.bindparam("description", type_=sa.Text()),
    sa.bindparam("body", type_=sa.Text()),
    sa.bindparam("directory", type_=sa.Text()),
)

# Plain VALUES here and not INSERT ... SELECT: the composite foreign key points at the
# version row this transaction just wrote, and that row was written under the tenant
# term above, so a file cannot be attached to a version the tenant does not own.
_ADD_VERSION_FILE = sa.text(
    "INSERT INTO skill_version_file (skill_id, version, path, body)"
    " VALUES (:id, :version, :path, :body)"
).bindparams(
    _ID,
    _VERSION,
    sa.bindparam("path", type_=sa.Text()),
    sa.bindparam("body", type_=sa.Text()),
)

_UNIQUE_VIOLATION = "23505"
"""PostgreSQL's SQLSTATE for a duplicate key.

Read off the driver's error rather than inferred from the exception class, because
`IntegrityError` also covers the check constraints on `path` and `directory`. Those are
not collisions and must not be retried: retrying one would mint a new version, fail the
same way, and turn a bad bundle into sixteen refused writes and a 500.
"""

# The version listing. `body` is absent from the SELECT for the reason it is absent from
# the skill listing: it is the one column that can be 32 KiB and no listing shows it.
# Newest first, because the reason to read this list is to find what to go back to.
_VERSION_PAGE_HEAD = (
    "SELECT v.version, v.name, v.description, v.directory FROM skill_version v"
    " JOIN skill s ON s.id = v.skill_id"
    " LEFT JOIN skill_version_retirement r"
    " ON r.skill_id = v.skill_id AND r.version = v.version"
    " WHERE s.tenant_id = :tenant AND v.skill_id = :id AND r.skill_id IS NULL"
)
_VERSION_PAGE_TAIL = " ORDER BY v.version DESC LIMIT :limit"

_VERSION_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "version": sa.BigInteger(),
    "name": sa.Text(),
    "description": sa.Text(),
    "directory": sa.Text(),
}

_FIRST_VERSION_PAGE = (
    sa.text(f"{_VERSION_PAGE_HEAD}{_VERSION_PAGE_TAIL}")
    .bindparams(_TENANT, _ID)
    .columns(**_VERSION_TYPES)
)

# Strictly less than, because the walk descends: the row the caller already holds is
# never handed out a second time. One comparison and no tiebreak, since a version is
# unique within a skill -- which is why this listing needs no composite cursor and the
# skill listing does.
_VERSION_PAGE_AFTER = (
    sa.text(f"{_VERSION_PAGE_HEAD} AND v.version < :after{_VERSION_PAGE_TAIL}")
    .bindparams(_TENANT, _ID, sa.bindparam("after", type_=sa.BigInteger()))
    .columns(**_VERSION_TYPES)
)

# One version, retired or not, and the tombstone comes back with it. A retirement is
# read as a left join rather than filtered out, because the caller has to be able to
# tell a version that was retired from one that was never written.
_READ_VERSION = (
    sa.text(
        "SELECT v.version, v.name, v.description, v.directory,"
        " (r.skill_id IS NOT NULL) AS retired"
        " FROM skill_version v JOIN skill s ON s.id = v.skill_id"
        " LEFT JOIN skill_version_retirement r"
        " ON r.skill_id = v.skill_id AND r.version = v.version"
        " WHERE s.tenant_id = :tenant AND v.skill_id = :id AND v.version = :version"
    )
    .bindparams(_TENANT, _ID, _VERSION)
    .columns(**_VERSION_TYPES, retired=sa.Boolean())
)

# The same row with the document, for the archive download and for nothing else.
_READ_VERSION_BODY = (
    sa.text(
        "SELECT v.version, v.name, v.description, v.directory, v.body,"
        " (r.skill_id IS NOT NULL) AS retired"
        " FROM skill_version v JOIN skill s ON s.id = v.skill_id"
        " LEFT JOIN skill_version_retirement r"
        " ON r.skill_id = v.skill_id AND r.version = v.version"
        " WHERE s.tenant_id = :tenant AND v.skill_id = :id AND v.version = :version"
    )
    .bindparams(_TENANT, _ID, _VERSION)
    .columns(**_VERSION_TYPES, body=sa.Text(), retired=sa.Boolean())
)

# Ordered by path so the same version always produces the same archive, which is what
# lets a caller compare two downloads of one version and get the same answer.
_READ_VERSION_FILES = (
    sa.text(
        "SELECT f.path, f.body FROM skill_version_file f"
        " JOIN skill s ON s.id = f.skill_id"
        " WHERE s.tenant_id = :tenant AND f.skill_id = :id"
        " AND f.version = :version ORDER BY f.path"
    )
    .bindparams(_TENANT, _ID, _VERSION)
    .columns(path=sa.Text(), body=sa.Text())
)

_RETIRE_VERSION = sa.text(
    "INSERT INTO skill_version_retirement (skill_id, version)"
    " SELECT v.skill_id, v.version FROM skill_version v"
    " JOIN skill s ON s.id = v.skill_id"
    " WHERE s.tenant_id = :tenant AND v.skill_id = :id AND v.version = :version"
    " ON CONFLICT DO NOTHING"
).bindparams(_TENANT, _ID, _VERSION)

_MAX_VERSION_PAGE = 1024
"""The largest version page this adapter will serve.

Above the 1000 the tenant surface publishes, plus the one extra row that surface asks
for to decide whether another page exists. Same shape as `_MAX_PAGE`: the published
bound is the tighter one, so a caller learns it from a 400 naming the field rather than
from a refusal down here.
"""


class PostgresSkillRegistry:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add_skill(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        skill: ValidatedSkill,
        *,
        display_name: str | None,
    ) -> None:
        """Store one uploaded skill under the id the platform minted for it.

        No conflict is possible and none is handled: the id is fresh, and two uploads
        naming the same skill are two skills with two ids on purpose.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                _ADD_SKILL,
                {
                    "id": skill_id,
                    "tenant": tenant_id,
                    "name": skill.name,
                    "description": skill.description,
                    "body": skill.text,
                    "display_name": display_name,
                },
            )

    async def read_skills(
        self, tenant_id: TenantId, skill_ids: Sequence[SkillId]
    ) -> tuple[SkillRecord, ...]:
        """The records among these ids this tenant holds, ordered by name then id.

        An empty request is answered without a query. Postgres would take `= ANY('{}')`
        happily and return nothing, but the round trip buys nothing and a definition
        attaching no skills is the common case.

        Ordered by id after name so two skills a tenant gave the same name still come
        back in a fixed order -- the name alone is not unique in this table, and that
        is deliberate.
        """
        if not skill_ids:
            return ()
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    _READ_SKILLS, {"tenant": tenant_id, "ids": list(skill_ids)}
                )
            ).all()
        return tuple(
            SkillRecord(
                skill_id=SkillId(row.id),
                skill=ValidatedSkill(
                    name=str(row.name),
                    description=str(row.description),
                    text=str(row.body),
                ),
            )
            for row in rows
        )

    async def page_uploaded_skills(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillListing, ...]:
        """One page of this tenant's uploaded skills, ordered by name then id.

        The tenant is a term in both statements rather than a filter over rows that came
        back, which is what makes another tenant's skill absent from the page instead of
        fetched and dropped. `skill_by_tenant` leads with `tenant_id`, so the scan
        cannot cross a tenant even where the planner would rather use the primary key.
        That index does not serve the sort, so the sort is over the one tenant's rows --
        the honest cost of ordering a named inventory by its names, and an index on
        `(tenant_id, name, id)` is what to add if it ever shows up in a plan.

        A `limit` outside the window raises rather than being clamped. A clamped page is
        a short page and a short page is how this read says the walk is over, so
        clamping would tell a caller it had seen everything.
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
        return tuple(
            SkillListing(
                skill_id=SkillId(row.id),
                name=str(row.name),
                description=str(row.description),
                # Left as None rather than coerced through `str`, which would turn
                # an unlabelled skill into the string "None" on the wire.
                display_name=(
                    None if row.display_name is None else str(row.display_name)
                ),
            )
            for row in rows
        )

    async def set_repository_skills(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        skills: Sequence[ValidatedSkill],
    ) -> int:
        """Record this checkout's skill directory; report how many rows were new.

        One transaction for the whole directory, so a submission that fails part way
        leaves the pair holding either the previous submission or nothing -- never half
        a skill set, which would resolve to an agent silently missing skills.

        A conflicting row is left exactly as it is rather than overwritten. The pair
        names a commit, and a commit's skills do not change; a second submission
        claiming otherwise is a retry, and the first answer stands.
        """
        written = 0
        async with self._engine.begin() as conn:
            for skill in skills:
                result = await conn.execute(
                    _ADD_REPOSITORY_SKILL,
                    {
                        "tenant": tenant_id,
                        "repository": repository,
                        "revision": revision,
                        "name": skill.name,
                        "description": skill.description,
                        "body": skill.text,
                    },
                )
                written += len(result.all())
        return written

    async def read_repository_skills(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> tuple[ValidatedSkill, ...]:
        """This checkout's skills, ordered by name; empty when none was submitted.

        Empty is an answer rather than a refusal. A definition may pin a repository
        nobody has submitted skills for, which is every definition registered before
        this table existed, and an agent with no repository skills is a real thing to
        want.
        """
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    _READ_REPOSITORY_SKILLS,
                    {
                        "tenant": tenant_id,
                        "repository": repository,
                        "revision": revision,
                    },
                )
            ).all()
        return tuple(
            ValidatedSkill(
                name=str(row.name),
                description=str(row.description),
                text=str(row.body),
            )
            for row in rows
        )

    async def read_skill(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> SkillHeld | None:
        """One skill with its live-version pointer, or None if this tenant holds none.

        A deleted skill still answers. Whether a deletion is a refusal or a no-op is the
        caller's question -- a read refuses, a repeated delete does not -- and a store
        that answered None for both would make the two indistinguishable here.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(_READ_SKILL, {"tenant": tenant_id, "id": skill_id})
            ).one_or_none()
        if row is None:
            return None
        latest = row.latest_version
        return SkillHeld(
            skill_id=SkillId(row.id),
            name=str(row.name),
            description=str(row.description),
            display_name=(None if row.display_name is None else str(row.display_name)),
            latest_version=None if latest is None else int(latest),
            deleted=bool(row.deleted),
        )

    async def delete_skill(self, tenant_id: TenantId, skill_id: SkillId) -> None:
        """Write the tombstone, leaving an existing one exactly as it is.

        No read first and no report of what happened. The route has already read the
        skill in order to refuse an id this tenant does not hold, and the only other
        outcome -- already deleted -- is answered identically on purpose.
        """
        async with self._engine.begin() as conn:
            await conn.execute(_DELETE_SKILL, {"tenant": tenant_id, "id": skill_id})

    async def add_skill_version(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        version: int,
        skill: ValidatedSkill,
        directory: str,
        files: Sequence[SkillVersionFile],
    ) -> None:
        """Write the version and its files in one transaction, or report a collision.

        Only the version insert is translated into `SkillVersionCollision`, and only on
        a duplicate key. A check constraint refusing a path or a directory travels on as
        the integrity error it is: it will fail the same way at every microsecond, so
        retrying it would spend the caller's whole retry budget and answer 500 for a
        bundle that should have been refused at the parse.
        """
        async with self._engine.begin() as conn:
            try:
                await conn.execute(
                    _ADD_VERSION,
                    {
                        "tenant": tenant_id,
                        "id": skill_id,
                        "version": version,
                        "name": skill.name,
                        "description": skill.description,
                        "body": skill.text,
                        "directory": directory,
                    },
                )
            except IntegrityError as clash:
                if getattr(clash.orig, "sqlstate", None) != _UNIQUE_VIOLATION:
                    raise
                raise SkillVersionCollision(
                    f"version {version} of skill {skill_id} already exists; two "
                    "versions were minted inside one microsecond"
                ) from clash
            for one in files:
                await conn.execute(
                    _ADD_VERSION_FILE,
                    {
                        "id": skill_id,
                        "version": version,
                        "path": one.path,
                        "body": one.text,
                    },
                )

    async def page_skill_versions(
        self, tenant_id: TenantId, skill_id: SkillId, after: int | None, limit: int
    ) -> tuple[SkillVersionRecord, ...]:
        """One page of this skill's live versions, newest first.

        A `limit` outside the window raises rather than being clamped, for the reason
        `page_uploaded_skills` gives: a clamped page is short, and a short page is how
        this read says the walk is over.
        """
        if limit < 1 or limit > _MAX_VERSION_PAGE:
            raise ValueError(
                f"version page limit {limit} for skill {skill_id} is outside "
                f"1..{_MAX_VERSION_PAGE}. Refused rather than clamped, because a "
                "clamped page is short and a short page means the walk is over."
            )
        parameters: dict[str, object] = {
            "tenant": tenant_id,
            "id": skill_id,
            "limit": limit,
        }
        statement = _FIRST_VERSION_PAGE
        if after is not None:
            statement = _VERSION_PAGE_AFTER
            parameters["after"] = after
        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement, parameters)).all()
        return tuple(
            SkillVersionRecord(
                version=int(row.version),
                name=str(row.name),
                description=str(row.description),
                directory=str(row.directory),
                retired=False,
            )
            for row in rows
        )

    async def read_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionRecord | None:
        """One version and whether it is retired, reading no file body.

        The document is not read either. This answers a metadata read, and the one
        caller that needs the bytes asks for the bundle instead.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _READ_VERSION,
                    {"tenant": tenant_id, "id": skill_id, "version": version},
                )
            ).one_or_none()
        if row is None:
            return None
        return SkillVersionRecord(
            version=int(row.version),
            name=str(row.name),
            description=str(row.description),
            directory=str(row.directory),
            retired=bool(row.retired),
        )

    async def read_skill_version_bundle(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionBundle | None:
        """One version with its document and every sibling file.

        Two statements on one connection rather than a join, because a join would repeat
        the 32 KiB document once per file and this read exists to move bytes.
        """
        async with self._engine.connect() as conn:
            found = (
                await conn.execute(
                    _READ_VERSION_BODY,
                    {"tenant": tenant_id, "id": skill_id, "version": version},
                )
            ).one_or_none()
            if found is None:
                return None
            rows = (
                await conn.execute(
                    _READ_VERSION_FILES,
                    {"tenant": tenant_id, "id": skill_id, "version": version},
                )
            ).all()
        return SkillVersionBundle(
            record=SkillVersionRecord(
                version=int(found.version),
                name=str(found.name),
                description=str(found.description),
                directory=str(found.directory),
                retired=bool(found.retired),
            ),
            skill_md=str(found.body),
            files=tuple(
                SkillVersionFile(path=str(row.path), text=str(row.body)) for row in rows
            ),
        )

    async def retire_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> None:
        """Write the tombstone, leaving an existing one and its moment as they are.

        A version this tenant does not hold selects no row and writes nothing, which is
        the same answer a repeated retirement gets. Both are cases the route has already
        decided: it read the version first in order to refuse an unknown one.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                _RETIRE_VERSION,
                {"tenant": tenant_id, "id": skill_id, "version": version},
            )
