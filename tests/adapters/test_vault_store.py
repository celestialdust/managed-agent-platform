"""Holding a tenant's vault and credential names, and what PostgreSQL guarantees.

Tier 1 (testcontainers, real PostgreSQL 17), against the adapter rather than raw SQL,
because the adapter is what a caller uses.

Real PostgreSQL and not a fake, because every property asserted below belongs to the
database rather than to the code: the unique index that turns a second vault of one name
into a refusal, the check constraint that refuses a `/` in a name whatever door the
write came through, the tenant term in the WHERE clause that makes another tenant's row
absent rather than forbidden, and the keyset comparison that walks a table without
repeating a row or stepping over one. A dict keyed by id would pass a version of all
four while proving none of them.

No test here writes a credential value, because no method under test accepts one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.vault_store import _MAX_PAGE, PostgresVaultStore
from managed_agent.core.ids import (
    TenantId,
    VaultId,
    new_credential_id,
    new_vault_id,
)
from managed_agent.core.ports import (
    CredentialNameTaken,
    VaultCatalogue,
    VaultNameTaken,
)
from managed_agent.core.vault_catalogue import Credential, CredentialKind, Vault

_EPOCH = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _tenant() -> TenantId:
    return TenantId(uuid.uuid4())


def _vault(tenant: TenantId, name: str, *, at: datetime = _EPOCH) -> Vault:
    """A live vault, id minted here the way the route mints one."""
    return Vault(id=new_vault_id(), tenant_id=tenant, name=name, created_at=at)


def _credential(
    vault: Vault,
    name: str,
    *,
    at: datetime = _EPOCH,
    kind: CredentialKind = CredentialKind.STATIC_BEARER,
) -> Credential:
    """A live credential in `vault`, created and last written at the same instant."""
    return Credential(
        id=new_credential_id(),
        vault_id=vault.id,
        tenant_id=vault.tenant_id,
        name=name,
        kind=kind,
        created_at=at,
        value_written_at=at,
    )


def test_the_adapter_is_the_port(engine: AsyncEngine) -> None:
    """The store satisfies `VaultCatalogue`, signatures included.

    The annotation is the assertion and `mypy --strict` is what checks it -- which is
    the half that matters, because every method on that port is positional-only and a
    keyword-taking implementation would satisfy `isinstance` and then fail at the one
    call site in `composition.py`. The runtime check below covers the other half: a
    method missing outright.
    """
    store: VaultCatalogue = PostgresVaultStore(engine)
    assert isinstance(store, VaultCatalogue)


async def test_a_vault_and_a_credential_round_trip(engine: AsyncEngine) -> None:
    """What comes back is what went in, as the domain object rather than a row.

    Including `created_at`: the port hands in a whole object and returns nothing, so a
    timestamp assigned by the column's default would leave the caller holding a copy
    that differs from the stored row with no call that would tell them.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    credential = _credential(vault, "github-token")

    await store.insert_vault(vault)
    await store.insert_credential(credential)

    assert await store.fetch_vault(vault.id, tenant) == vault
    assert await store.fetch_credential(credential.id, tenant) == credential


async def test_a_credentials_ref_is_its_two_names_joined(engine: AsyncEngine) -> None:
    """The stored names compose the ref the Tool Gateway already resolves.

    Asserted through what was read back rather than through what was written, because
    the addressing scheme is only true if the round trip preserves both halves: a store
    that lower-cased, trimmed or re-encoded either name would put the credential at a
    ref no Tool registration names.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    await store.insert_vault(vault)
    await store.insert_credential(_credential(vault, "github-token"))

    listed = await store.page_credentials(vault.id, tenant, None, 10, False)

    assert [row.ref(vault.name) for row in listed] == ["prod/github-token"]


async def test_another_tenants_rows_are_absent_rather_than_forbidden(
    engine: AsyncEngine,
) -> None:
    """The tenant is a term in the query, not a filter applied to what came back.

    So a caller holding an id learns nothing from the answer about whether it names
    somebody else's vault -- and the two reads answer alike, which is what stops the
    credential read from being the softer of the two.
    """
    store = PostgresVaultStore(engine)
    owner, stranger = _tenant(), _tenant()
    vault = _vault(owner, "prod")
    credential = _credential(vault, "github-token")
    await store.insert_vault(vault)
    await store.insert_credential(credential)

    assert await store.fetch_vault(vault.id, stranger) is None
    assert await store.fetch_credential(credential.id, stranger) is None
    assert await store.page_credentials(vault.id, stranger, None, 10, False) == []


async def test_a_second_vault_of_one_name_is_refused(engine: AsyncEngine) -> None:
    """One name per tenant, so a ref composed from it names one vault.

    Refused rather than disambiguated: two vaults called `prod` would make every
    `prod/<credential>` ref resolve to whichever the Gateway reached first, and the
    tenant would have no way to see which.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    await store.insert_vault(_vault(tenant, "prod"))

    with pytest.raises(VaultNameTaken) as refusal:
        await store.insert_vault(_vault(tenant, "prod"))

    assert refusal.value.name == "prod"


async def test_one_vault_name_per_tenant_and_not_per_platform(
    engine: AsyncEngine,
) -> None:
    """Two tenants may both call a vault `prod`.

    The unique index is on `(tenant_id, name)`, and the pairing matters: a platform-wide
    name would make one tenant's registration decide what another tenant may call
    theirs, which is a cross-tenant coupling nothing about the ref format asks for.
    """
    store = PostgresVaultStore(engine)
    first, second = _tenant(), _tenant()

    await store.insert_vault(_vault(first, "prod"))
    await store.insert_vault(_vault(second, "prod"))

    assert (await store.page_vaults(first, None, 10, False))[0].tenant_id == first
    assert (await store.page_vaults(second, None, 10, False))[0].tenant_id == second


async def test_a_second_credential_of_one_name_in_one_vault_is_refused(
    engine: AsyncEngine,
) -> None:
    """The name is unique inside the vault, and free again in the next one.

    Both halves in one test because they are one claim: the vault is the namespace. If
    the second insert were refused too, a vault would be a weaker namespace than the ref
    format says it is, and a tenant with `staging` and `prod` copies of one integration
    could not name them alike.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    prod, staging = _vault(tenant, "prod"), _vault(tenant, "staging")
    await store.insert_vault(prod)
    await store.insert_vault(staging)
    await store.insert_credential(_credential(prod, "github-token"))

    with pytest.raises(CredentialNameTaken) as refusal:
        await store.insert_credential(_credential(prod, "github-token"))

    assert refusal.value.vault_id == prod.id
    assert refusal.value.name == "github-token"
    await store.insert_credential(_credential(staging, "github-token"))


async def test_the_database_refuses_a_slash_in_a_vault_name(
    engine: AsyncEngine,
) -> None:
    """A name that would make a ref ambiguous is refused by the table, not the type.

    Built with `model_construct` so the boundary's own pattern is bypassed, which is the
    whole point: the route is where a tenant reads the refusal and the constraint is
    where the invariant is kept, and a second write door -- a backfill, a repair script
    -- reaches the constraint and not the route. With `/` admitted, vault `a/b`
    credential `c` and vault `a` credential `b/c` compose to one vault key and one of a
    tenant's credentials silently shadows another.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    ambiguous = Vault.model_construct(
        id=new_vault_id(),
        tenant_id=tenant,
        name="acme/prod",
        created_at=_EPOCH,
        archived_at=None,
    )

    with pytest.raises(IntegrityError) as refusal:
        await store.insert_vault(ambiguous)

    assert "vault_name_is_one_segment" in str(refusal.value)


async def test_the_database_refuses_a_slash_in_a_credential_name(
    engine: AsyncEngine,
) -> None:
    """The same rule on the other half of the ref, kept by the other table.

    Held on both names or held on neither: the ref is the join of the two, so a
    constraint on one alone leaves exactly the ambiguity it was written to prevent.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    await store.insert_vault(vault)
    ambiguous = Credential.model_construct(
        id=new_credential_id(),
        vault_id=vault.id,
        tenant_id=tenant,
        name="github/token",
        kind=CredentialKind.STATIC_BEARER,
        created_at=_EPOCH,
        value_written_at=_EPOCH,
        archived_at=None,
    )

    with pytest.raises(IntegrityError) as refusal:
        await store.insert_credential(ambiguous)

    assert "credential_name_is_one_segment" in str(refusal.value)


async def test_an_archived_credential_leaves_the_listing_and_stays_readable(
    engine: AsyncEngine,
) -> None:
    """Archiving hides a name from the default listing and never loses it.

    All three readings in one test because they are one behaviour: the row must leave
    the listing a tenant browses, come back when they ask for the archive, and still
    answer a fetch -- a Session that used the credential has a name to report rather
    than a dangling id, and there is no unarchive to recover it with.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    kept = _credential(vault, "kept", at=_EPOCH)
    retired = _credential(vault, "retired", at=_EPOCH + timedelta(seconds=1))
    await store.insert_vault(vault)
    await store.insert_credential(kept)
    await store.insert_credential(retired)

    assert await store.archive_credential(retired.id, tenant) is True

    live = await store.page_credentials(vault.id, tenant, None, 10, False)
    everything = await store.page_credentials(vault.id, tenant, None, 10, True)
    fetched = await store.fetch_credential(retired.id, tenant)

    assert [row.id for row in live] == [kept.id]
    assert [row.id for row in everything] == [retired.id, kept.id]
    assert fetched is not None
    assert fetched.archived_at is not None


async def test_an_archived_vault_leaves_the_listing_and_stays_readable(
    engine: AsyncEngine,
) -> None:
    """The same for a vault, and its credentials are left where they are.

    Not a cascade: a vault is a namespace rather than a container, so archiving it must
    not retire names the tenant never named -- with no unarchive, a cascade would be
    unrecoverable.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    credential = _credential(vault, "github-token")
    await store.insert_vault(vault)
    await store.insert_credential(credential)

    assert await store.archive_vault(vault.id, tenant) is True

    assert await store.page_vaults(tenant, None, 10, False) == []
    assert [row.id for row in await store.page_vaults(tenant, None, 10, True)] == [
        vault.id
    ]
    still_there = await store.fetch_credential(credential.id, tenant)
    assert still_there is not None
    assert still_there.archived_at is None


async def test_archiving_twice_says_which_call_did_it(engine: AsyncEngine) -> None:
    """True from the call that archived, False from a repeat -- and the time stands.

    The repeat is answered rather than refused, so a client whose call timed out can
    retry; and because the second write matches nothing, `archived_at` keeps the moment
    the archive actually happened instead of the moment somebody asked again.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    await store.insert_vault(vault)

    assert await store.archive_vault(vault.id, tenant) is True
    first = await store.fetch_vault(vault.id, tenant)
    assert await store.archive_vault(vault.id, tenant) is False
    second = await store.fetch_vault(vault.id, tenant)

    assert first is not None and second is not None
    assert first.archived_at == second.archived_at


async def test_another_tenant_cannot_archive_or_delete(engine: AsyncEngine) -> None:
    """False, and nothing written -- the same False a missing id gets.

    The two cases are deliberately one answer: a caller able to tell "no such id" from
    "not yours" could probe another tenant's ids by watching which refusal came back.
    """
    store = PostgresVaultStore(engine)
    owner, stranger = _tenant(), _tenant()
    vault = _vault(owner, "prod")
    credential = _credential(vault, "github-token")
    await store.insert_vault(vault)
    await store.insert_credential(credential)

    assert await store.archive_credential(credential.id, stranger) is False
    assert await store.delete_credential(credential.id, stranger) is False
    assert await store.archive_vault(vault.id, stranger) is False
    assert await store.delete_vault(vault.id, stranger) is False
    assert await store.archive_vault(VaultId(uuid.uuid4()), owner) is False

    survivor = await store.fetch_credential(credential.id, owner)
    assert survivor is not None
    assert survivor.archived_at is None


async def test_deleting_a_credential_removes_the_row(engine: AsyncEngine) -> None:
    """True once, then False, and the name is gone from every read.

    Deleting is what a tenant does to take the name back; archiving is what they do to
    stop it being attachable. The two are separate acts, so this must actually remove
    the row rather than stamp it.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    credential = _credential(vault, "github-token")
    await store.insert_vault(vault)
    await store.insert_credential(credential)

    assert await store.delete_credential(credential.id, tenant) is True
    assert await store.delete_credential(credential.id, tenant) is False
    assert await store.fetch_credential(credential.id, tenant) is None
    assert await store.page_credentials(vault.id, tenant, None, 10, True) == []


async def test_a_vault_holding_credentials_will_not_be_deleted(
    engine: AsyncEngine,
) -> None:
    """The foreign key refuses it: a credential cannot outlive the vault addressing it.

    A cascade here would delete rows whose values still sit in Secrets Manager under a
    key nothing can compose any more. The caller empties the vault first; this is what
    catches the write that raced that check.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    credential = _credential(vault, "github-token")
    await store.insert_vault(vault)
    await store.insert_credential(credential)

    with pytest.raises(IntegrityError):
        await store.delete_vault(vault.id, tenant)

    assert await store.fetch_vault(vault.id, tenant) is not None

    assert await store.delete_credential(credential.id, tenant) is True
    assert await store.delete_vault(vault.id, tenant) is True


async def test_a_credential_cannot_be_written_into_another_tenants_vault(
    engine: AsyncEngine,
) -> None:
    """The composite foreign key makes a cross-tenant credential unrepresentable.

    Refused by `(vault_id, tenant_id)` referencing `(vault.id, vault.tenant_id)` rather
    than by a check in the route: there is no pair of values that satisfies the
    reference and crosses a tenant, so no write door can produce one.
    """
    store = PostgresVaultStore(engine)
    owner, stranger = _tenant(), _tenant()
    vault = _vault(owner, "prod")
    await store.insert_vault(vault)
    smuggled = Credential(
        id=new_credential_id(),
        vault_id=vault.id,
        tenant_id=stranger,
        name="github-token",
        kind=CredentialKind.STATIC_BEARER,
        created_at=_EPOCH,
        value_written_at=_EPOCH,
    )

    with pytest.raises(IntegrityError):
        await store.insert_credential(smuggled)


async def test_a_kind_the_closed_set_refuses_is_not_reported_as_a_taken_name(
    engine: AsyncEngine,
) -> None:
    """An unknown kind raises the integrity error itself, never `CredentialNameTaken`.

    The distinction the route depends on. `insert_credential` absorbs exactly one
    constraint -- `credential_name_is_one_per_vault`, named as the ON CONFLICT target --
    so the foreign key, the name check and this kind check each surface as themselves.
    A blanket `except IntegrityError` mapped onto "name taken" would tell a tenant to
    rename a credential whose name was never the problem, and they would rename it and
    be refused again.

    The kind is carried by a throwaway enum rather than by a bare string, because the
    adapter reads `.value` off it: the point is a row whose kind the database has not
    heard of, not a caller passing the wrong Python type.
    """

    class _RetiredKind(StrEnum):
        OAUTH_REFRESH = "oauth_refresh"

    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    await store.insert_vault(vault)
    unplaceable = Credential.model_construct(
        id=new_credential_id(),
        vault_id=vault.id,
        tenant_id=tenant,
        name="github-token",
        kind=_RetiredKind.OAUTH_REFRESH,
        created_at=_EPOCH,
        value_written_at=_EPOCH,
        archived_at=None,
    )

    with pytest.raises(IntegrityError) as refusal:
        await store.insert_credential(unplaceable)

    assert "credential_kind_is_a_known_one" in str(refusal.value)
    assert not isinstance(refusal.value, CredentialNameTaken)


async def test_marking_a_value_written_moves_only_that_timestamp(
    engine: AsyncEngine,
) -> None:
    """A rotation updates `value_written_at` and leaves `created_at` where it was.

    The pair is what lets a tenant looking at a credential that stopped working tell
    "rotated an hour ago" from "untouched since March" -- which is the only thing the
    platform can offer them, since it never returns the value itself.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    credential = _credential(vault, "github-token")
    rotated_at = _EPOCH + timedelta(days=30)
    await store.insert_vault(vault)
    await store.insert_credential(credential)

    assert await store.mark_value_written(credential.id, tenant, rotated_at) is True
    assert (
        await store.mark_value_written(
            credential.id, _tenant(), rotated_at + timedelta(days=1)
        )
        is False
    )

    read = await store.fetch_credential(credential.id, tenant)
    assert read is not None
    assert read.value_written_at == rotated_at
    assert read.created_at == credential.created_at


async def test_a_keyset_walk_sees_every_credential_exactly_once(
    engine: AsyncEngine,
) -> None:
    """Paging two at a time over five rows yields all five, newest first, no repeats.

    The property the walk exists for. A page boundary is where an ordering that is not
    total shows up: the same row served twice, or one stepped over, and neither raises.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    await store.insert_vault(vault)
    written = [
        _credential(vault, f"token-{index}", at=_EPOCH + timedelta(seconds=index))
        for index in range(5)
    ]
    for credential in written:
        await store.insert_credential(credential)

    walked: list[Credential] = []
    after: tuple[datetime, uuid.UUID] | None = None
    while True:
        page = await store.page_credentials(vault.id, tenant, after, 2, False)
        walked.extend(page)
        if len(page) < 2:
            break
        after = (page[-1].created_at, page[-1].id)

    assert [row.id for row in walked] == [row.id for row in reversed(written)]


async def test_a_keyset_walk_separates_rows_written_at_one_instant(
    engine: AsyncEngine,
) -> None:
    """Three vaults sharing a timestamp are still walked one at a time.

    `created_at` alone is not a total order -- a batch write gives several rows the
    identical instant -- so the page resumes on `(created_at, id)`. Without the
    tie-break a walk with `limit` 1 would serve one row forever or skip its neighbours.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    written = [_vault(tenant, f"vault-{index}", at=_EPOCH) for index in range(3)]
    for vault in written:
        await store.insert_vault(vault)

    walked: list[Vault] = []
    after: tuple[datetime, uuid.UUID] | None = None
    while True:
        page = await store.page_vaults(tenant, after, 1, False)
        walked.extend(page)
        if not page:
            break
        after = (page[-1].created_at, page[-1].id)

    assert sorted(row.id for row in walked) == sorted(row.id for row in written)


@pytest.mark.parametrize("limit", [0, -1, _MAX_PAGE + 1])
async def test_a_limit_outside_the_bound_is_refused_rather_than_clamped(
    engine: AsyncEngine, limit: int
) -> None:
    """`ValueError`, not a smaller page.

    A clamped page is a short page, and a caller walking a keyset reads a short page as
    the end of the walk -- so clamping would end the walk early and silently, which is
    the failure that looks like missing data rather than like a bug.
    """
    store = PostgresVaultStore(engine)
    tenant = _tenant()
    vault = _vault(tenant, "prod")
    await store.insert_vault(vault)

    with pytest.raises(ValueError, match="outside 1..500"):
        await store.page_vaults(tenant, None, limit, False)
    with pytest.raises(ValueError, match="outside 1..500"):
        await store.page_credentials(vault.id, tenant, None, limit, False)
