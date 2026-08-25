"""The `vault` and `vault_credential` tables: two names written, read, listed, retired.

**Nothing this module reads or writes holds a secret value.** The value behind a
credential lives in Secrets Manager under a key composed from the two names below, and a
copy here would be a second place to leak it from and a second place a rotation could
miss. Every statement exchanges names, ids, kinds and timestamps -- what comes back is
enough to manage a credential and never enough to use one.

Every statement is keyed by the tenant, and the tenant is a WHERE term rather than a
check applied to what came back: another tenant's row is absent from the result instead
of fetched and then dropped, so there is no moment at which this process holds a row it
is not entitled to. That is also why the credential read never joins to its vault for
the tenant -- `vault_credential` carries its own `tenant_id`, and a predicate one hop
away is one a query can forget while still returning rows.

**A duplicate name is refused by the unique index, in the INSERT.** `ON CONFLICT ON
CONSTRAINT ... DO NOTHING RETURNING id` returns no row when the name is taken, which is
one round trip with no window between a check and a write for a concurrent writer to
slip into. The conflict target is the constraint by name rather than its columns,
because that is what keeps a primary-key collision out of the same arm: the caller mints
a fresh uuid per write, so a collision on `id` is a store fault and reporting it as a
name the tenant chose badly would send them to rename something that was never the
problem.

**Listings are keyset walks over `(created_at DESC, id DESC)`, and the archived filter
is two statements rather than one bound boolean.** That is what keeps migration 0029's
partial index reachable once a pooled connection's prepared statement stops being
re-planned -- see `_LIVE`, which carries the measurement. The four forms each listing
needs are built once at import and chosen by a lookup, so the keyset comparison and the
ordering are spelled once: two transcriptions of a keyset rule are free to disagree, and
a walk over a key two statements order differently repeats one row and steps over
another.

Bind parameter types are declared because these are textual statements: SQLAlchemy has
no column metadata to infer from, so asyncpg receives whatever Python object it was
handed. `.columns(...)` is the same declaration in the other direction.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.core.ids import CredentialId, TenantId, VaultId
from managed_agent.core.ports import CredentialNameTaken, VaultNameTaken
from managed_agent.core.vault_catalogue import Credential, CredentialKind, Vault

_VAULT_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "tenant_id": sa.Uuid(),
    "name": sa.Text(),
    "created_at": sa.TIMESTAMP(timezone=True),
    "archived_at": sa.TIMESTAMP(timezone=True),
}

_CREDENTIAL_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "vault_id": sa.Uuid(),
    "tenant_id": sa.Uuid(),
    "name": sa.Text(),
    "kind": sa.Text(),
    "created_at": sa.TIMESTAMP(timezone=True),
    "value_written_at": sa.TIMESTAMP(timezone=True),
    "archived_at": sa.TIMESTAMP(timezone=True),
}

_VAULT_COLUMNS = "id, tenant_id, name, created_at, archived_at"
_CREDENTIAL_COLUMNS = (
    "id, vault_id, tenant_id, name, kind, created_at, value_written_at, archived_at"
)

_ID = sa.bindparam("id", type_=sa.Uuid())
_TENANT_ID = sa.bindparam("tenant_id", type_=sa.Uuid())
_VAULT_ID = sa.bindparam("vault_id", type_=sa.Uuid())
_AT = sa.bindparam("at", type_=sa.TIMESTAMP(timezone=True))
_CREATED_AT = sa.bindparam("created_at", type_=sa.TIMESTAMP(timezone=True))
_VALUE_WRITTEN_AT = sa.bindparam("value_written_at", type_=sa.TIMESTAMP(timezone=True))
_AFTER = (
    sa.bindparam("after_at", type_=sa.TIMESTAMP(timezone=True)),
    sa.bindparam("after_id", type_=sa.Uuid()),
)

_LIVE = " AND archived_at IS NULL"
"""The default listing's filter, as a literal rather than a bound parameter.

Measured against real PostgreSQL 17 with `enable_seqscan` off, because the first version
of this comment claimed more than was true. Under a **custom** plan -- the parameter's
value known while planning -- `(:include_archived OR archived_at IS NULL)` folds to
`archived_at IS NULL` and plans `vault_credential_live_by_tenant` exactly as this
literal does. The two are equal there, and a bound boolean would cost nothing.

They part under a **generic** plan, which is what a pooled connection reaches once
PostgreSQL stops re-planning a prepared statement it has already run several times. The
parameter cannot be folded, so the query's predicate no longer implies the index's, and
the listing drops to a bitmap scan over `credential_name_is_one_per_vault` plus a sort
-- reading the archive alongside the live rows and losing the ordering the index already
had. The literal keeps the backward index scan in both plans.

That is the whole reason the archived filter is two statements here and one bound
boolean in `environment_store.py`: the index it has to stay on is partial, and that one
is not.
"""

_KEYSET = " AND (created_at, id) < (:after_at, :after_id)"
"""Where the next page resumes, as a row constructor rather than the three-way OR it
expands into -- the form `session_registry.py` uses, and for the same reason: one
comparison the planner can drive an index with, and one place the tie-break on `id`
is spelled."""

_ORDER = " ORDER BY created_at DESC, id DESC LIMIT :limit"
"""Newest first, with `id` breaking a tie.

The tie-break is not decoration. Both tables default `created_at` to `now()`, which is
the transaction's start time, so two rows written in one transaction carry the identical
timestamp -- and a keyset walk whose key is not unique either repeats a row or steps
over one.
"""

_MAX_PAGE = 500
"""The largest page this adapter will serve.

A read with no ceiling materialises every row it matches into one list before the caller
sees the first. A larger `limit` is refused rather than reduced, for the reason
`environment_store.py` gives: a clamped page is a short page, and a short page is how
this port says the walk is over.
"""

# `created_at` is written from the object rather than left to the column's default. The
# port hands in a whole `Vault` and returns nothing, so a server-assigned timestamp
# would be a value the caller holds a different copy of and has no way to learn -- it
# would read its own object back changed. The default stays useful for the doors that do
# not come through here: a backfill, a repair script.
#
# `archived_at` is not written, whatever the object carries. Archiving is an act with
# its own statement below, and admitting it as a field here would let a create produce a
# vault that was retired before it existed.
_INSERT_VAULT = sa.text(
    "INSERT INTO vault (id, tenant_id, name, created_at)"
    " VALUES (:id, :tenant_id, :name, :created_at)"
    " ON CONFLICT ON CONSTRAINT vault_name_is_one_per_tenant DO NOTHING"
    " RETURNING id"
).bindparams(_ID, _TENANT_ID, _CREATED_AT)

_INSERT_CREDENTIAL = sa.text(
    "INSERT INTO vault_credential"
    " (id, vault_id, tenant_id, name, kind, created_at, value_written_at)"
    " VALUES (:id, :vault_id, :tenant_id, :name, :kind, :created_at,"
    " :value_written_at)"
    " ON CONFLICT ON CONSTRAINT credential_name_is_one_per_vault DO NOTHING"
    " RETURNING id"
).bindparams(_ID, _VAULT_ID, _TENANT_ID, _CREATED_AT, _VALUE_WRITTEN_AT)

# No archive predicate on either fetch: a retired name stays readable. A Session that
# used a credential still has a name to report rather than a dangling id, and a tenant
# that archived something by mistake can still see what they archived.
_FETCH_VAULT = (
    sa.text(
        f"SELECT {_VAULT_COLUMNS} FROM vault WHERE id = :id AND tenant_id = :tenant_id"
    )
    .bindparams(_ID, _TENANT_ID)
    .columns(**_VAULT_TYPES)
)

_FETCH_CREDENTIAL = (
    sa.text(
        f"SELECT {_CREDENTIAL_COLUMNS} FROM vault_credential"
        " WHERE id = :id AND tenant_id = :tenant_id"
    )
    .bindparams(_ID, _TENANT_ID)
    .columns(**_CREDENTIAL_TYPES)
)

_VAULT_PAGE = f"SELECT {_VAULT_COLUMNS} FROM vault WHERE tenant_id = :tenant_id"

_VAULT_PAGE_LIVE = (
    sa.text(f"{_VAULT_PAGE}{_LIVE}{_ORDER}")
    .bindparams(_TENANT_ID)
    .columns(**_VAULT_TYPES)
)
_VAULT_PAGE_LIVE_AFTER = (
    sa.text(f"{_VAULT_PAGE}{_LIVE}{_KEYSET}{_ORDER}")
    .bindparams(_TENANT_ID, *_AFTER)
    .columns(**_VAULT_TYPES)
)
_VAULT_PAGE_ALL = (
    sa.text(f"{_VAULT_PAGE}{_ORDER}").bindparams(_TENANT_ID).columns(**_VAULT_TYPES)
)
_VAULT_PAGE_ALL_AFTER = (
    sa.text(f"{_VAULT_PAGE}{_KEYSET}{_ORDER}")
    .bindparams(_TENANT_ID, *_AFTER)
    .columns(**_VAULT_TYPES)
)

_CREDENTIAL_PAGE = (
    f"SELECT {_CREDENTIAL_COLUMNS} FROM vault_credential"
    " WHERE tenant_id = :tenant_id AND vault_id = :vault_id"
)

_CREDENTIAL_PAGE_LIVE = (
    sa.text(f"{_CREDENTIAL_PAGE}{_LIVE}{_ORDER}")
    .bindparams(_TENANT_ID, _VAULT_ID)
    .columns(**_CREDENTIAL_TYPES)
)
_CREDENTIAL_PAGE_LIVE_AFTER = (
    sa.text(f"{_CREDENTIAL_PAGE}{_LIVE}{_KEYSET}{_ORDER}")
    .bindparams(_TENANT_ID, _VAULT_ID, *_AFTER)
    .columns(**_CREDENTIAL_TYPES)
)
_CREDENTIAL_PAGE_ALL = (
    sa.text(f"{_CREDENTIAL_PAGE}{_ORDER}")
    .bindparams(_TENANT_ID, _VAULT_ID)
    .columns(**_CREDENTIAL_TYPES)
)
_CREDENTIAL_PAGE_ALL_AFTER = (
    sa.text(f"{_CREDENTIAL_PAGE}{_KEYSET}{_ORDER}")
    .bindparams(_TENANT_ID, _VAULT_ID, *_AFTER)
    .columns(**_CREDENTIAL_TYPES)
)

# Keyed by `(include_archived, resuming after a key)`. A lookup rather than a branch so
# the four forms sit beside each other and a fifth cannot be added in one family and
# forgotten in the other.
_VAULT_PAGES: Mapping[tuple[bool, bool], sa.TextualSelect] = {
    (False, False): _VAULT_PAGE_LIVE,
    (False, True): _VAULT_PAGE_LIVE_AFTER,
    (True, False): _VAULT_PAGE_ALL,
    (True, True): _VAULT_PAGE_ALL_AFTER,
}

_CREDENTIAL_PAGES: Mapping[tuple[bool, bool], sa.TextualSelect] = {
    (False, False): _CREDENTIAL_PAGE_LIVE,
    (False, True): _CREDENTIAL_PAGE_LIVE_AFTER,
    (True, False): _CREDENTIAL_PAGE_ALL,
    (True, True): _CREDENTIAL_PAGE_ALL_AFTER,
}

# `archived_at IS NULL` in the WHERE is what makes the timestamp the moment of the FIRST
# archive: a repeat matches nothing and moves nothing, so a client whose call timed out
# and retried does not overwrite when it happened with when it asked again.
_ARCHIVE_VAULT = sa.text(
    "UPDATE vault SET archived_at = now()"
    " WHERE id = :id AND tenant_id = :tenant_id AND archived_at IS NULL"
).bindparams(_ID, _TENANT_ID)

_ARCHIVE_CREDENTIAL = sa.text(
    "UPDATE vault_credential SET archived_at = now()"
    " WHERE id = :id AND tenant_id = :tenant_id AND archived_at IS NULL"
).bindparams(_ID, _TENANT_ID)

_DELETE_VAULT = sa.text(
    "DELETE FROM vault WHERE id = :id AND tenant_id = :tenant_id"
).bindparams(_ID, _TENANT_ID)

_DELETE_CREDENTIAL = sa.text(
    "DELETE FROM vault_credential WHERE id = :id AND tenant_id = :tenant_id"
).bindparams(_ID, _TENANT_ID)

_MARK_VALUE_WRITTEN = sa.text(
    "UPDATE vault_credential SET value_written_at = :at"
    " WHERE id = :id AND tenant_id = :tenant_id"
).bindparams(_ID, _TENANT_ID, _AT)


class PostgresVaultStore:
    """A tenant's vaults and credential names, over one engine.

    Both families on one class because they are one port: a credential is addressed
    through its vault, and every read of one is scoped by the tenant that owns the
    other. Two classes would put that tenant predicate in two places, and the predicate
    is what stops a tenant reading somebody else's row.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def insert_vault(self, vault: Vault, /) -> None:
        """Write one vault, live.

        Raises `VaultNameTaken` when this tenant already holds a vault of that name.
        Refused rather than disambiguated: a credential's ref is `<vault>/<credential>`,
        so two vaults of one name would make every ref under either of them resolve to
        whichever the Tool Gateway reached first.

        `created_at` is written from the object; `archived_at` is not written at all --
        see the comment on `_INSERT_VAULT` for both.
        """
        async with self._engine.begin() as conn:
            written = (
                await conn.execute(
                    _INSERT_VAULT,
                    {
                        "id": vault.id,
                        "tenant_id": vault.tenant_id,
                        "name": vault.name,
                        "created_at": vault.created_at,
                    },
                )
            ).scalar_one_or_none()
        if written is None:
            raise VaultNameTaken(vault.name)

    async def fetch_vault(
        self, vault_id: VaultId, tenant_id: TenantId, /
    ) -> Vault | None:
        """One of this tenant's vaults, archived or not, or None.

        None covers "no such id" and "that id is somebody else's" alike, because those
        two answers are the same answer: a caller able to tell them apart could
        enumerate another tenant's vault ids by asking.

        An archived vault comes back normally. Archiving stops a vault being used, and
        hiding it here would leave a tenant holding an id they can neither use nor look
        at -- and a Session that named one of its credentials with no name to report.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _FETCH_VAULT, {"id": vault_id, "tenant_id": tenant_id}
                )
            ).one_or_none()
        return None if row is None else _vault(row)

    async def page_vaults(
        self,
        tenant_id: TenantId,
        after: tuple[datetime, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Vault]:
        """One page of this tenant's vaults, newest first.

        `after` is the `(created_at, id)` of the last row the caller holds, and the walk
        resumes strictly below it -- so a vault created while the caller is part-way
        through the walk sorts above the position they are at and is simply not seen,
        rather than shifting every remaining row by one.

        A `limit` outside `1..500` is refused rather than clamped: a clamped page is a
        short page, and a short page is how this port says the walk is over.
        """
        _refuse_a_limit_outside_the_bound(limit, tenant_id)
        parameters: dict[str, object] = {"tenant_id": tenant_id, "limit": limit}
        if after is not None:
            parameters["after_at"], parameters["after_id"] = after
        statement = _VAULT_PAGES[(include_archived, after is not None)]
        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement, parameters)).all()
        return [_vault(row) for row in rows]

    async def archive_vault(self, vault_id: VaultId, tenant_id: TenantId, /) -> bool:
        """Retire one vault, and say whether this call is what retired it.

        False for an id that is not this tenant's, for one nobody registered, and for a
        vault that was already archived. The first two are one answer on purpose; the
        third joins them because the alternative is an error describing the state the
        caller asked for, which forces every retry into a read-then-write race.

        The credentials in the vault are left alone. A vault is a namespace rather than
        a container, and cascading would archive names the tenant never named -- with no
        way back, since there is no unarchive.
        """
        async with self._engine.begin() as conn:
            archived = await conn.execute(
                _ARCHIVE_VAULT, {"id": vault_id, "tenant_id": tenant_id}
            )
        return archived.rowcount == 1

    async def delete_vault(self, vault_id: VaultId, tenant_id: TenantId, /) -> bool:
        """Remove one vault, and say whether there was one to remove.

        False for an id that never existed and for one belonging to somebody else
        alike: the statement matches on both columns, so the two cases are the same
        zero-row result and no caller can tell them apart.

        Raises `sqlalchemy.exc.IntegrityError` when the vault still holds credentials.
        The foreign key names `(vault.id, vault.tenant_id)` with no cascade, so a vault
        cannot be removed out from under rows that address themselves through its name
        -- the caller establishes it is empty first, and this is what catches the write
        that raced that check. Cascading instead would delete credential rows while
        their values stayed in Secrets Manager, orphaned under a key nothing can compose
        any more.
        """
        async with self._engine.begin() as conn:
            removed = await conn.execute(
                _DELETE_VAULT, {"id": vault_id, "tenant_id": tenant_id}
            )
        return removed.rowcount == 1

    async def insert_credential(self, credential: Credential, /) -> None:
        """Write one credential's name and shape, live. No value passes here.

        Raises `CredentialNameTaken` when the vault already holds that name. Scoped to
        the vault and not to the tenant: the same spelling in two of a tenant's vaults
        is two different refs, and refusing it would make a vault a weaker namespace
        than the ref format says it is.

        Raises `sqlalchemy.exc.IntegrityError` when `vault_id` names no vault, or names
        one belonging to another tenant. The second is the composite foreign key doing
        its job -- there is no pair of values satisfying it that crosses a tenant, so a
        cross-tenant credential is unrepresentable rather than merely refused.

        `value_written_at` is written from the object, beside `created_at`. They are
        equal on a create and diverge on the first rotation, which is the whole reason
        the column exists.
        """
        async with self._engine.begin() as conn:
            written = (
                await conn.execute(
                    _INSERT_CREDENTIAL,
                    {
                        "id": credential.id,
                        "vault_id": credential.vault_id,
                        "tenant_id": credential.tenant_id,
                        "name": credential.name,
                        "kind": credential.kind.value,
                        "created_at": credential.created_at,
                        "value_written_at": credential.value_written_at,
                    },
                )
            ).scalar_one_or_none()
        if written is None:
            raise CredentialNameTaken(credential.vault_id, credential.name)

    async def fetch_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> Credential | None:
        """One of this tenant's credentials, archived or not, or None.

        Read by id and tenant with no join to the vault, which is what the denormalised
        `tenant_id` on the row is for: the tenant predicate is load-bearing for
        isolation, so it belongs on the table being read rather than one hop away where
        a query can drop it and still return rows.

        None covers "no such id" and "not this tenant's", for the reason `fetch_vault`
        gives.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    _FETCH_CREDENTIAL, {"id": credential_id, "tenant_id": tenant_id}
                )
            ).one_or_none()
        return None if row is None else _credential(row)

    async def page_credentials(
        self,
        vault_id: VaultId,
        tenant_id: TenantId,
        after: tuple[datetime, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Credential]:
        """One page of one vault's credentials, newest first. No values.

        Empty for a vault that does not exist, one belonging to somebody else, and one
        holding nothing -- the same three-into-one answer the fetches give, reached the
        same way and for the same reason.

        `after` and `limit` behave as they do for `page_vaults`, and the default listing
        is the read migration 0029's partial index was built for.
        """
        _refuse_a_limit_outside_the_bound(limit, tenant_id)
        parameters: dict[str, object] = {
            "tenant_id": tenant_id,
            "vault_id": vault_id,
            "limit": limit,
        }
        if after is not None:
            parameters["after_at"], parameters["after_id"] = after
        statement = _CREDENTIAL_PAGES[(include_archived, after is not None)]
        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement, parameters)).all()
        return [_credential(row) for row in rows]

    async def archive_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> bool:
        """Stop this credential being attachable, and say whether this call did it.

        False for an id that is not this tenant's, for one nobody registered, and for a
        credential already archived -- one answer, for the reason `archive_vault` gives.

        The value in Secrets Manager is untouched. Removing it is `delete_credential`'s
        job, and the two are separate acts: a tenant who archived a name by mistake has
        nothing to recover if archiving had erased the value too.
        """
        async with self._engine.begin() as conn:
            archived = await conn.execute(
                _ARCHIVE_CREDENTIAL, {"id": credential_id, "tenant_id": tenant_id}
            )
        return archived.rowcount == 1

    async def delete_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> bool:
        """Remove one credential's row, and say whether there was one to remove.

        False for an id that never existed and for one belonging to somebody else
        alike. Only the row goes: erasing the value behind it is a separate call to a
        separate port, and the caller sequences the two -- this returning True is how it
        knows the name was its own to erase under.
        """
        async with self._engine.begin() as conn:
            removed = await conn.execute(
                _DELETE_CREDENTIAL, {"id": credential_id, "tenant_id": tenant_id}
            )
        return removed.rowcount == 1

    async def mark_value_written(
        self, credential_id: CredentialId, tenant_id: TenantId, at: datetime, /
    ) -> bool:
        """Record when the value behind this name was last written.

        The timestamp is taken from the caller rather than from `now()`, so it is the
        moment the vault accepted the value and not the moment this row was updated --
        the two statements are separate round trips, and a rotation that succeeded and
        then reported a later time would have this column disagree with the vault's own.

        Matches an archived credential as well as a live one. Archiving decides whether
        a value may be *attached*, and refusing here would fold "not yours" and "already
        archived" into the same `False` -- which leaves the caller no message to send.
        """
        async with self._engine.begin() as conn:
            marked = await conn.execute(
                _MARK_VALUE_WRITTEN,
                {"id": credential_id, "tenant_id": tenant_id, "at": at},
            )
        return marked.rowcount == 1


def _refuse_a_limit_outside_the_bound(limit: int, tenant_id: TenantId) -> None:
    """Raise `ValueError` unless `limit` is within `1..500`.

    Shared by both listings because the reason is one reason and the number is one
    number: a page size this adapter would clamp is a page the caller reads as the last
    one. Two copies could be given two ceilings, and the walk that broke would be the
    one over the table with more rows in it.
    """
    if limit < 1 or limit > _MAX_PAGE:
        raise ValueError(
            f"page limit {limit} for tenant {tenant_id} is outside 1..{_MAX_PAGE}. "
            "Refused rather than clamped, because a clamped page is short and a "
            "short page means the walk is over."
        )


def _vault(row: Any) -> Vault:
    """One `vault` row as the domain object, parsed rather than cast.

    Parsed on the way out even though it was parsed on the way in, because the two rules
    are deliberately not the same rule: 0029's check constraint refuses only what would
    make a credential's ref ambiguous -- a `/`, a `..`, a leading punctuation character
    -- while `VaultName` also fixes the case and the length. So a row written through
    another door can satisfy the database and fail here, and failing here is the answer
    to want: the alternative hands a caller an object whose type says it was validated
    when it was not.
    """
    return Vault(
        id=VaultId(UUID(str(row.id))),
        tenant_id=TenantId(UUID(str(row.tenant_id))),
        name=str(row.name),
        created_at=row.created_at,
        archived_at=row.archived_at,
    )


def _credential(row: Any) -> Credential:
    """One `vault_credential` row as the domain object, values and all -- of which
    there are none.

    `kind` is re-read into the enum rather than trusted as text. The column's check
    constraint and the enum are one closed set today, and a kind the enum has not heard
    of is a credential nothing can attach: caught here, at a read the tenant is waiting
    on, rather than inside the Tool Gateway one hop from the model.
    """
    return Credential(
        id=CredentialId(UUID(str(row.id))),
        vault_id=VaultId(UUID(str(row.vault_id))),
        tenant_id=TenantId(UUID(str(row.tenant_id))),
        name=str(row.name),
        kind=CredentialKind(str(row.kind)),
        created_at=row.created_at,
        value_written_at=row.value_written_at,
        archived_at=row.archived_at,
    )
