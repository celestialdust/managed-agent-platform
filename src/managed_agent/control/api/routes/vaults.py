"""Registering a tenant's own tool credentials, and never handing one back.

The surface a tenant reaches before it registers an MCP server that needs
authenticating. A vault is a namespace, a credential is a name and a shape inside one,
and the value is the one thing here that arrives and never leaves: it goes to the
credential vault on the way in and no route below returns it, redacts it, or reveals a
substring of it. `Credential` carries no value field for that reason and
`CredentialView` does not add one -- a field a reader can see is a field somebody later
populates behind a flag.

**What a create returns that makes the surface usable at all** is the `ref` --
`<vault name>/<credential name>` -- because that string, and nothing else here, is what
the tenant then writes into a `POST /v1/mcp_servers` registration. Without it a tenant
holds two ids and has to guess that the join is a slash. The ref is composed by
`credential_ref` rather than spelled here, so this surface and the Tool Gateway's
resolution cannot drift into two spellings of one address.

**The vault key is composed from `core`, not from the Gateway.** The prefix lives at
`core.vault_names.TOOL_CREDENTIAL_PREFIX` precisely because two processes under two IAM
roles compose it -- the Gateway to read a credential on an outbound call, this route to
write one when a tenant registers it. Reaching into `gateway/` from a control-plane
route would invert that: one package's internal detail would become the other's
contract, and the write would land wherever the Gateway's copy happened to say.

**Order of writes, and which way each one fails.** A credential is two writes -- the
value into the vault, the row into the catalogue -- and they cannot be one. So the
question is only which residue is survivable:

- *Value first, then row.* A crash between them leaves a vault entry no row names. It
  is absent from every listing, and the tenant's next attempt at the same create
  overwrites it. Invisible, and retrying repairs it.
- *Row first, then value.* A crash between them leaves a row the tenant can see, whose
  `ref` they will paste into a registration, behind which there is nothing. Every tool
  call authenticated with it fails at somebody else's MCP server, with a message about
  somebody else's MCP server -- one hop past anything this platform can explain.

The second is the failure this whole module exists to prevent, so the value goes first.
A delete runs the same argument backwards and lands on the opposite order for a reason
that is not symmetry: **an erase needs the vault's name to compose the key, and the
delete is what destroys that name.** Forgetting first would strand a live, attachable
value with nothing left that could address it. So a delete erases, then forgets.

**Archived is read-only, and there is no unarchive.** A write against an archived vault
is refused rather than applied to a row the tenant believes is retired, and creating a
credential inside one is such a write.

Statuses here are ours, taken from `STATUS_FOR` and chosen nowhere in this file.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Final
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from managed_agent.control.api.refusals import Refusal, refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import (
    CredentialId,
    TenantId,
    VaultId,
    new_credential_id,
    new_vault_id,
)
from managed_agent.core.ports import (
    CredentialNameTaken,
    CredentialVaultWriter,
    VaultCatalogue,
    VaultNameTaken,
)
from managed_agent.core.vault_catalogue import (
    Credential,
    CredentialKind,
    CredentialName,
    CredentialSubmission,
    Vault,
    VaultName,
    credential_ref,
)
from managed_agent.core.vault_names import TOOL_CREDENTIAL_PREFIX, scoped_vault_name

router = APIRouter(tags=["vaults"])

VAULT_ABSENT: Final = "no vault with that id is registered"
CREDENTIAL_ABSENT: Final = "no credential with that id is in that vault"
"""The one sentence each absent id is refused with.

Written once because each is answered for situations that must not be told apart: no
such id, an id another tenant registered, and an id this call has just deleted. A second
wording is a second answer, and the difference between two answers is an existence
oracle over ids nobody is entitled to probe for.
"""

DEFAULT_PAGE_SIZE: Final = 25
MAX_PAGE_SIZE: Final = 100

MAX_CREDENTIALS_PER_VAULT: Final = 20
"""The most credentials one vault may hold, matching the surface this one mirrors.

A ceiling rather than a storage limit: nothing here would strain at a thousand. It is
published by the API being mirrored, so a tenant porting a working integration finds
the same number here -- and a platform that accepted the twenty-first would be handing
back a vault that cannot be represented on the surface it claims parity with.

Per vault, and a tenant may create as many vaults as they like, so this bounds no
tenant's total. That is what makes it a shape rather than a quota."""
"""How many rows one page may hold.

Bounded because an unbounded page is a whole-collection read wearing a limit parameter.
"""

_SCAN_PAGE: Final = 100
"""How many rows one step of an internal walk reads.

`_walk_credentials` covers a whole vault, so this is a round-trip size rather than a
bound on the answer.
"""

_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    STATUS_FOR[ErrorCode.VAULT_NOT_FOUND]: {"model": PublicErrorEnvelope}
}
"""The refusal every route here shares, declared once for the published document.

Annotated because FastAPI's `responses` parameter is `dict[int | str, dict[str, Any]]`
and a bare literal assigned to a name infers something narrower. `VAULT_NOT_FOUND` and
`CREDENTIAL_NOT_FOUND` are both 404, so one key covers both and the envelope is the
same shape either way.
"""

_ARCHIVED: dict[int | str, dict[str, Any]] = {
    STATUS_FOR[ErrorCode.VAULT_ARCHIVED]: {"model": PublicErrorEnvelope}
}


def catalogue_of(request: Request) -> VaultCatalogue:
    """Where this process keeps vault and credential rows.

    A dependency rather than a read inside each handler, so the eleven routes below name
    what they need in their signature and a process wired without a catalogue refuses at
    the first call rather than at an attribute lookup halfway through a write.
    """
    return platform_from_request(request).vault_catalogue


def writer_of(request: Request) -> CredentialVaultWriter:
    """Where a submitted value goes, and the only thing on this surface that puts one
    there.

    Typed at the writing half of the vault port, which has no `fetch`. That is what
    makes "no route here returns a value" a property of the code rather than a promise:
    there is no call available in this module that could read one back.
    """
    return platform_from_request(request).credential_writer


class InvalidCursor(Exception):
    """The caller sent something that is not a cursor this surface issued."""


@dataclass(frozen=True, slots=True)
class Cursor:
    """A position in one creation-ordered walk: the instant, and the row at it.

    Both halves are needed. Two rows can be written in one microsecond, and a position
    naming only the instant cannot say which of them the caller already holds -- so a
    page boundary landing between them repeats one row and drops the other.

    One type for both walks here, and deliberately not shared with the `Cursor` in
    `environments.py` or `session_list.py`. Those hold a typed id of their own
    collection, and a single type spanning collections would make a token issued by one
    walk a position the other accepts. Within this module the two walks are keyed by the
    same pair and are both narrowed by the path before the cursor is read at all -- a
    vault walk by the tenant, a credential walk by the tenant and the vault -- so a
    token carried from one to the other names a place, never a row the caller may not
    see. The worst it can do is start a page somewhere the caller did not mean.

    `|` separates the halves rather than `.`, because an ISO-8601 instant contains a dot
    before its microseconds and a position whose timestamp had to be reassembled from
    two fragments is a parse waiting to be got subtly wrong. Neither an instant nor a
    uuid can contain a pipe.
    """

    created_at: datetime
    row_id: UUID

    def encode(self) -> str:
        """The position as a token, base64url with its padding stripped.

        Padding is stripped so the token carries no `=`, which would be percent-encoded
        in a query string and come back looking different from what was issued.
        """
        raw = f"{self.created_at.isoformat()}|{self.row_id}".encode()
        return urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> "Cursor":
        """Parse a token back into a position, or raise `InvalidCursor`.

        Everything that is not a token this surface issued is one refusal. There is no
        partial reading -- a token whose instant parses and whose id does not names no
        row, and treating half of it as a position would start the next page somewhere
        the caller never was.
        """
        try:
            padded = token + "=" * (-len(token) % 4)
            text = urlsafe_b64decode(padded.encode()).decode()
            stamp, separator, identifier = text.partition("|")
            if not separator:
                # Raised inside the `try` and deliberately not a `ValueError`: this
                # module's own refusal type passes straight through the clause below
                # rather than being caught and re-raised as itself.
                raise InvalidCursor(token)
            return cls(datetime.fromisoformat(stamp), UUID(identifier))
        except ValueError as exc:
            # binascii.Error and UnicodeDecodeError are both ValueError, so one clause
            # covers bad base64, bad utf-8, a malformed instant and a malformed uuid.
            raise InvalidCursor(token) from exc


def _position(page: str | None) -> tuple[datetime, UUID] | None:
    """The store-level position a `page` parameter names, or None for the start.

    A cursor this surface did not issue is refused rather than treated as the start of
    the collection. Starting over on a bad cursor silently hands back the newest page
    again, which reads as the walk having looped rather than failed.
    """
    if page is None:
        return None
    try:
        cursor = Cursor.decode(page)
    except InvalidCursor as exc:
        raise Refusal(
            ErrorCode.PAGINATION_CURSOR_INVALID,
            "cursor was not issued by this surface",
        ) from exc
    return (cursor.created_at, cursor.row_id)


def _entry_name(tenant_id: TenantId, vault_name: str, credential_name: str) -> str:
    """The vault key holding one credential's value, for this tenant.

    Composed here from `core` and never from `gateway.tool.credential_broker`, whose
    `VAULT_PREFIX` is an alias of the same constant. The prefix, the tenant and the ref
    are joined by `scoped_vault_name`, so the name written here is the name the Tool
    Gateway resolves -- and a ref that tried to climb out of the tenant's segment
    raises there rather than composing to somebody else's entry.
    """
    return scoped_vault_name(
        TOOL_CREDENTIAL_PREFIX,
        tenant_id,
        credential_ref(vault_name, credential_name),
    )


class CreateVault(BaseModel):
    """A vault being registered. One field, because a vault is only a namespace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: VaultName


class RotateCredential(BaseModel):
    """A new value for a credential that already exists.

    Only the value. Not the name, because the name is half the `ref` every registration
    already written against this credential names -- accepting a new one here would
    silently re-point them at nothing. Not the kind either: the kind decides where the
    value is attached, and a bearer token quietly becoming an environment variable is a
    tool call that fails at the far end for a reason nothing here recorded. Both are
    changed by deleting this credential and registering another, which is the act that
    tells the tenant their registrations must move too.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: SecretStr = Field(min_length=1)


class VaultView(BaseModel):
    """One vault read back. It carries no tenant: the caller is the tenant."""

    model_config = ConfigDict(frozen=True)

    id: VaultId
    name: VaultName
    created_at: datetime
    archived_at: datetime | None
    """When this vault was retired, or null while it is live. There is no unarchive, so
    a non-null value here is terminal."""

    @classmethod
    def of(cls, vault: Vault) -> "VaultView":
        return cls(
            id=vault.id,
            name=vault.name,
            created_at=vault.created_at,
            archived_at=vault.archived_at,
        )


class CredentialView(BaseModel):
    """One credential read back: what it is called, what shape it is, and **no value**.

    The value is absent rather than redacted, and the absence is the feature. A redacted
    field teaches every reader that the value is here and merely hidden, which is the
    premise a later change needs in order to reveal it under some flag. `last_four` is
    refused on the same ground -- it is a real substring of a real credential, handed to
    anybody who can name the tenant.

    `ref` is here because it is the reason a tenant called this route: it is the exact
    string they paste into an MCP server registration. It is derived from the vault's
    name at read time rather than stored, so it cannot go stale against the vault.
    """

    model_config = ConfigDict(frozen=True)

    id: CredentialId
    vault_id: VaultId
    name: CredentialName
    kind: CredentialKind
    ref: str
    created_at: datetime
    value_written_at: datetime
    """When the value behind this name was last written, which is not when the row was
    created once a rotation has happened. It decides nothing -- it is here so a tenant
    whose credential stopped working can tell "rotated an hour ago" from "untouched
    since March" without this platform showing them a value it never returns."""

    archived_at: datetime | None

    @classmethod
    def of(cls, credential: Credential, vault_name: str) -> "CredentialView":
        return cls(
            id=credential.id,
            vault_id=credential.vault_id,
            name=credential.name,
            kind=credential.kind,
            ref=credential.ref(vault_name),
            created_at=credential.created_at,
            value_written_at=credential.value_written_at,
            archived_at=credential.archived_at,
        )


class VaultPage(BaseModel):
    """One page of vaults, and where the page after it starts.

    `data` and `next_page` are the names the rest of this API's list responses use, so a
    client written against one reads this one. `next_page` is null at the end of the
    walk rather than a token leading somewhere empty, so a caller stops on a field it
    can read instead of on a wasted round trip.
    """

    model_config = ConfigDict(frozen=True)

    data: tuple[VaultView, ...]
    next_page: str | None


class CredentialPage(BaseModel):
    """One page of one vault's credentials. No value appears in any element."""

    model_config = ConfigDict(frozen=True)

    data: tuple[CredentialView, ...]
    next_page: str | None


async def _walk_credentials(
    catalogue: VaultCatalogue, vault_id: VaultId, tenant_id: TenantId
) -> AsyncIterator[Credential]:
    """Every credential in one vault, archived included, in the catalogue's order."""
    after: tuple[datetime, UUID] | None = None
    while True:
        rows = await catalogue.page_credentials(
            vault_id, tenant_id, after, _SCAN_PAGE, True
        )
        for row in rows:
            yield row
        if len(rows) < _SCAN_PAGE:
            return
        after = (rows[-1].created_at, rows[-1].id)


@dataclass(frozen=True, slots=True)
class _VaultContents:
    """What one walk of a vault answers, for a create that has two questions.

    Both facts come from the same walk because the create needs both before it writes
    anything, and a vault holding hundreds of credentials would otherwise be paged
    twice per request to answer them separately.
    """

    name_is_taken: bool
    held: int


async def _scan_for_create(
    catalogue: VaultCatalogue, vault_id: VaultId, tenant_id: TenantId, name: str
) -> _VaultContents:
    """Whether that vault holds a credential of that name, and how many it holds.

    **This is not how the 409 is produced -- `CredentialNameTaken` is.** It is here to
    stop a refused create from destroying the credential it collided with, and that
    danger is specific to this surface: the vault key is composed from the tenant and
    the two names and from nothing else, so a create naming an existing credential
    composes *the same key*. With the value written before the row, the catalogue's
    refusal would arrive one step too late -- the tenant would receive a 409 and the
    credential they already had would be sitting behind the value from the request that
    was just rejected.

    So the ordinary duplicate is stopped before anything is written, and the narrow race
    two simultaneous creates can still win is caught from the catalogue below. There is
    no such hazard on a vault, which holds no value, and no scan there.

    The count comes back from the same walk for the ceiling check, and it counts
    archived credentials too. Archiving revokes nothing and frees nothing: an archived
    row still holds its name against a new create and its value is still attachable, so
    a ceiling that ignored it would be a ceiling a tenant could walk past by archiving.
    """
    held = 0
    taken = False
    async for credential in _walk_credentials(catalogue, vault_id, tenant_id):
        held += 1
        taken = taken or credential.name == name
    return _VaultContents(name_is_taken=taken, held=held)


async def _vault_or_refusal(
    catalogue: VaultCatalogue, vault_id: VaultId, tenant_id: TenantId
) -> Vault | JSONResponse:
    """One vault under the tenant predicate, or the refusal every route shares.

    The tenant is a term in the store's query rather than something compared against
    what came back, so there is no moment at which this process holds a row it is not
    entitled to.
    """
    vault = await catalogue.fetch_vault(vault_id, tenant_id)
    if vault is None:
        return refuse(ErrorCode.VAULT_NOT_FOUND, VAULT_ABSENT, vault_id=str(vault_id))
    return vault


def _name_taken_refusal(name: str, vault_id: VaultId) -> JSONResponse:
    return refuse(
        ErrorCode.CREDENTIAL_NAME_TAKEN,
        "this vault already holds a credential of that name, and a second one would "
        "make the ref composed from it address two credentials",
        name=name,
        vault_id=str(vault_id),
    )


def _archived_refusal(vault: Vault) -> JSONResponse:
    return refuse(
        ErrorCode.VAULT_ARCHIVED,
        "that vault was retired, and a retirement is terminal, so nothing can be "
        "written to it or registered inside it",
        vault_id=str(vault.id),
    )


# --------------------------------------------------------------------------------------
# Vaults
# --------------------------------------------------------------------------------------


@router.post(
    "/vaults",
    status_code=status.HTTP_201_CREATED,
    response_model=VaultView,
    responses={STATUS_FOR[ErrorCode.VAULT_NAME_TAKEN]: {"model": PublicErrorEnvelope}},
)
async def create_vault(
    body: CreateVault,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
) -> VaultView | JSONResponse:
    """Register a namespace for this tenant's credentials.

    The id is minted here rather than accepted from the caller, so registering twice
    makes two vaults rather than overwriting one -- and overwriting one would re-point
    every ref composed from its name.

    A name this tenant already holds is refused. Disambiguating it silently is what
    cannot happen: a ref is `<vault>/<credential>`, so two vaults of one name would make
    one ref address two credentials, resolved inside the Tool Gateway where nothing can
    ask which was meant.
    """
    vault = Vault(
        id=new_vault_id(),
        tenant_id=tenant_id,
        name=body.name,
        created_at=datetime.now(UTC),
    )
    try:
        await catalogue.insert_vault(vault)
    except VaultNameTaken as taken:
        return refuse(
            ErrorCode.VAULT_NAME_TAKEN,
            "this tenant already holds a vault of that name, and a name is never "
            "disambiguated behind the refs composed from it",
            name=taken.name,
        )
    return VaultView.of(vault)


@router.get(
    "/vaults",
    response_model=VaultPage,
    responses={
        STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]: {"model": PublicErrorEnvelope}
    },
)
async def list_vaults(
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
    page: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    include_archived: bool = False,
) -> VaultPage | JSONResponse:
    """One page of this tenant's vaults, and nothing about what is inside them.

    A vault is a namespace rather than a container, so listing what is in one is a
    separate read with its own page -- a tenant with four hundred credentials in one
    vault does not get them as a side effect of listing vaults.

    Archived vaults are absent unless asked for, and that direction is the useful one:
    an archived vault takes no new credential, so a default listing carrying them would
    put entries in front of a caller that every write would refuse.

    The store is asked for one row more than will be returned. That extra row is the
    whole answer to "is there another page", and it is why `next_page` is null rather
    than a token leading somewhere empty.
    """
    rows = await catalogue.page_vaults(
        tenant_id, _position(page), limit + 1, include_archived
    )
    shown = rows[:limit]
    more = len(rows) > limit
    return VaultPage(
        data=tuple(VaultView.of(row) for row in shown),
        next_page=(
            Cursor(shown[-1].created_at, shown[-1].id).encode()
            if more and shown
            else None
        ),
    )


@router.get("/vaults/{vault_id}", response_model=VaultView, responses=_NOT_FOUND)
async def read_vault(
    vault_id: VaultId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
) -> VaultView | JSONResponse:
    """One vault as it stands.

    A retired vault reads back normally with `archived_at` set. That is the point of
    retirement being a fact on the resource rather than a deletion: a caller who
    archived one by mistake can still see what they had.
    """
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    return VaultView.of(found)


@router.post(
    "/vaults/{vault_id}/archive", response_model=VaultView, responses=_NOT_FOUND
)
async def archive_vault(
    vault_id: VaultId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
) -> VaultView | JSONResponse:
    """Retire a vault: no new credential, no rotation, contents still readable.

    Retiring twice is not an error, and the response is what makes that observable: the
    second call answers 200 carrying the timestamp of the FIRST retirement. A refusal
    there would push every client into a read-then-write race to avoid it, and a fresh
    timestamp would claim the vault stopped accepting writes at the moment of a retry --
    a false fact about when the refusals began. That is why an already-archived vault
    returns here without a second write rather than being archived again.

    Archiving revokes nothing. The values behind this vault's credentials stay where
    they are and stay attachable, because a registration resolves a ref through the Tool
    Gateway without consulting these rows at all. Revoking is what DELETE does, and the
    difference is deliberate: a tenant tidying a namespace must not silently break every
    Session already authenticating with what is in it.

    There is no unarchive, on this surface or any other.
    """
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    if found.archived_at is not None:
        return VaultView.of(found)
    if not await catalogue.archive_vault(vault_id, tenant_id):
        # Read, then written, and the write found nothing: a delete landed in between.
        # Answered as absent, which is what it now is.
        return refuse(ErrorCode.VAULT_NOT_FOUND, VAULT_ABSENT, vault_id=str(vault_id))
    retired = await catalogue.fetch_vault(vault_id, tenant_id)
    if retired is None:
        return refuse(ErrorCode.VAULT_NOT_FOUND, VAULT_ABSENT, vault_id=str(vault_id))
    return VaultView.of(retired)


@router.delete(
    "/vaults/{vault_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    # Explicitly no response model. FastAPI otherwise derives one from the return
    # annotation and refuses the route at import time, because a 204 may carry no body;
    # the unhappy path is a JSONResponse declared under `responses` instead.
    response_model=None,
    responses=_NOT_FOUND,
)
async def delete_vault(
    vault_id: VaultId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
    writer: Annotated[CredentialVaultWriter, Depends(writer_of)],
) -> Response | JSONResponse:
    """Remove a vault, erasing every value it held.

    **Erase first, then forget, and the order is forced rather than chosen.** The key a
    value sits under is composed from the vault's name and the credential's name, so
    deleting the rows first destroys the only thing that can address the values -- and
    what would be left is a live, attachable secret nothing remaining could name or
    reach. A tenant who then registered a vault of the same name would resurrect every
    one of them.

    An erase that fails partway leaves rows whose values are gone. Those credentials
    fail at the far end, loudly, and the tenant's retry of this same call erases the
    rest and removes the rows -- `erase` is idempotent, so a retry repairs rather than
    compounds. That is the residue worth having; the other order's residue is a secret
    that outlives every record of itself.

    **The credential rows go too, one at a time, and the vault cannot be removed until
    they have.** `vault_credential`'s foreign key names `(vault.id, vault.tenant_id)`
    with no cascade, so `delete_vault` refuses a vault anything still points at -- the
    store will not orphan a row, and this is where the emptying happens instead. It is
    also why the erase and the row removal are paired per credential rather than done in
    two passes: a pass that erased everything and then failed to remove a row would
    leave that row addressing a key whose value is gone, and the pairing bounds that to
    the one credential the failure happened on.

    Refuses rather than answering 204 for an id this tenant does not hold, because a
    tenant that mistyped an id and got 204 would believe it had revoked credentials it
    had not.
    """
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    async for credential in _walk_credentials(catalogue, vault_id, tenant_id):
        await writer.erase(_entry_name(tenant_id, found.name, credential.name))
        await catalogue.delete_credential(credential.id, tenant_id)
    if not await catalogue.delete_vault(vault_id, tenant_id):
        # Another delete of the same id won the race. Its loser is told the same thing
        # a caller naming an id nobody registered is told.
        return refuse(ErrorCode.VAULT_NOT_FOUND, VAULT_ABSENT, vault_id=str(vault_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------------------


@router.post(
    "/vaults/{vault_id}/credentials",
    status_code=status.HTTP_201_CREATED,
    response_model=CredentialView,
    responses={
        **_NOT_FOUND,
        **_ARCHIVED,
        STATUS_FOR[ErrorCode.CREDENTIAL_NAME_TAKEN]: {"model": PublicErrorEnvelope},
    },
)
async def create_credential(
    vault_id: VaultId,
    body: CredentialSubmission,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
    writer: Annotated[CredentialVaultWriter, Depends(writer_of)],
) -> CredentialView | JSONResponse:
    """Register a credential and return the `ref` a Tool registration will name it by.

    **The value goes to the vault before the row goes to the catalogue.** A crash
    between them leaves a vault entry no row names -- absent from every listing, and
    overwritten by the tenant's next attempt at the same create. The other order leaves
    a row the tenant can see and whose ref they will paste into a registration, with
    nothing behind it, so every tool call using it fails at somebody else's MCP server.

    The value is read out of its `SecretStr` on exactly one line below, and that is the
    only line in this module that can produce it. It reaches the writer and nothing
    else: not the response, not a log record, not the refusals above, which carry ids
    and names only.

    Creating inside an archived vault is refused. A retirement is terminal, and a
    credential registered into one would be a ref the tenant could use while the vault
    holding it is one nobody may write to.
    """
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    if found.archived_at is not None:
        return _archived_refusal(found)
    contents = await _scan_for_create(catalogue, vault_id, tenant_id, body.name)
    if contents.name_is_taken:
        return _name_taken_refusal(body.name, vault_id)
    if contents.held >= MAX_CREDENTIALS_PER_VAULT:
        # Ahead of the `put` below, and that ordering is the whole of this check being
        # correct rather than merely present. The vault key is composed from the tenant
        # and the two names, so a refusal that arrived after the write would leave a
        # value at a name whose row was never created: unreachable through this API,
        # absent from every count, and still held at the vault.
        return refuse(
            ErrorCode.VAULT_FULL,
            f"this vault already holds {contents.held} credentials, which is the "
            f"most one vault may hold. Delete one, or create another vault -- the "
            f"ceiling is per vault and a tenant may hold many.",
            vault_id=str(vault_id),
            held=contents.held,
        )
    now = datetime.now(UTC)
    credential = Credential(
        id=new_credential_id(),
        vault_id=vault_id,
        tenant_id=tenant_id,
        name=body.name,
        kind=body.kind,
        created_at=now,
        value_written_at=now,
    )
    await writer.put(
        _entry_name(tenant_id, found.name, body.name), body.value.get_secret_value()
    )
    try:
        await catalogue.insert_credential(credential)
    except CredentialNameTaken as taken:
        # The scan above found the name free and this one lost the race to it. The
        # value written a moment ago now sits behind the winner's row, which is the one
        # outcome this ordering cannot prevent -- both writes were this tenant's own,
        # in flight together, and the alternative order strands a row with no value.
        return _name_taken_refusal(taken.name, vault_id)
    return CredentialView.of(credential, found.name)


@router.get(
    "/vaults/{vault_id}/credentials",
    response_model=CredentialPage,
    responses={
        **_NOT_FOUND,
        STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]: {"model": PublicErrorEnvelope},
    },
)
async def list_credentials(
    vault_id: VaultId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
    page: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    include_archived: bool = False,
) -> CredentialPage | JSONResponse:
    """One page of a vault's credentials, each with its ref and none with its value.

    The vault is read first for its name, which every `ref` on this page is composed
    from, and that read is also what refuses an id this tenant does not hold -- so a
    caller cannot page an unknown vault and read emptiness as an answer about it.
    """
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    rows = await catalogue.page_credentials(
        vault_id, tenant_id, _position(page), limit + 1, include_archived
    )
    shown = rows[:limit]
    more = len(rows) > limit
    return CredentialPage(
        data=tuple(CredentialView.of(row, found.name) for row in shown),
        next_page=(
            Cursor(shown[-1].created_at, shown[-1].id).encode()
            if more and shown
            else None
        ),
    )


async def _credential_or_refusal(
    catalogue: VaultCatalogue,
    vault_id: VaultId,
    credential_id: CredentialId,
    tenant_id: TenantId,
) -> Credential | JSONResponse:
    """One credential of one vault under the tenant predicate, or a refusal.

    The vault in the path is checked against the row rather than trusted, because the
    catalogue addresses a credential by its own id: without this, a credential could be
    read, rotated or deleted through any vault id the tenant happens to hold, and the
    path would be saying something untrue about where the credential lives.

    A credential in another of this tenant's vaults is refused exactly as one that does
    not exist is. Telling them apart would answer "that id is yours, but not here",
    which is a fact the caller can already establish by naming the right vault.
    """
    credential = await catalogue.fetch_credential(credential_id, tenant_id)
    if credential is None or credential.vault_id != vault_id:
        return refuse(
            ErrorCode.CREDENTIAL_NOT_FOUND,
            CREDENTIAL_ABSENT,
            credential_id=str(credential_id),
            vault_id=str(vault_id),
        )
    return credential


@router.get(
    "/vaults/{vault_id}/credentials/{credential_id}",
    response_model=CredentialView,
    responses=_NOT_FOUND,
)
async def read_credential(
    vault_id: VaultId,
    credential_id: CredentialId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
) -> CredentialView | JSONResponse:
    """One credential's name, shape and ref. Never its value, on this route or any
    other."""
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    credential = await _credential_or_refusal(
        catalogue, vault_id, credential_id, tenant_id
    )
    if isinstance(credential, JSONResponse):
        return credential
    return CredentialView.of(credential, found.name)


@router.post(
    "/vaults/{vault_id}/credentials/{credential_id}",
    response_model=CredentialView,
    responses={**_NOT_FOUND, **_ARCHIVED},
)
async def rotate_credential(
    vault_id: VaultId,
    credential_id: CredentialId,
    body: RotateCredential,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
    writer: Annotated[CredentialVaultWriter, Depends(writer_of)],
) -> CredentialView | JSONResponse:
    """Replace the value behind a credential, at the same name and the same ref.

    **Rotation is an overwrite at one key, and that is the whole feature.** The ref does
    not change, so every registration already naming it goes on working and starts using
    the new value without being touched. A rotation that minted a new name would leave
    the old value live at the old key with nothing left to erase it, which is the
    opposite of what a rotation is for.

    The value goes to the vault before `value_written_at` is moved, for the reason the
    create gives: a timestamp claiming a write that did not happen would tell a tenant
    debugging a failing credential that they had already fixed it.

    Refused against an archived vault, because that is a write and a retirement is
    terminal.
    """
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    if found.archived_at is not None:
        return _archived_refusal(found)
    credential = await _credential_or_refusal(
        catalogue, vault_id, credential_id, tenant_id
    )
    if isinstance(credential, JSONResponse):
        return credential
    await writer.put(
        _entry_name(tenant_id, found.name, credential.name),
        body.value.get_secret_value(),
    )
    written_at = datetime.now(UTC)
    if not await catalogue.mark_value_written(credential_id, tenant_id, written_at):
        # The row went away between the read and the stamp. The value is written and
        # nothing names it, which is the residue this surface tolerates -- see the
        # module docstring -- so the honest answer is that the credential is absent.
        return refuse(
            ErrorCode.CREDENTIAL_NOT_FOUND,
            CREDENTIAL_ABSENT,
            credential_id=str(credential_id),
            vault_id=str(vault_id),
        )
    rotated = credential.model_copy(update={"value_written_at": written_at})
    return CredentialView.of(rotated, found.name)


@router.post(
    "/vaults/{vault_id}/credentials/{credential_id}/archive",
    response_model=CredentialView,
    responses=_NOT_FOUND,
)
async def archive_credential(
    vault_id: VaultId,
    credential_id: CredentialId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
) -> CredentialView | JSONResponse:
    """Retire one credential, leaving it readable and its ref composable.

    Idempotent on the same terms a vault's archive is: a second call answers with the
    FIRST retirement's timestamp rather than refusing or re-stamping, so a retried
    request cannot claim the credential was retired later than it was.

    Archiving revokes nothing, exactly as a vault's archive does not. Use DELETE to
    revoke; this marks a row.
    """
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    credential = await _credential_or_refusal(
        catalogue, vault_id, credential_id, tenant_id
    )
    if isinstance(credential, JSONResponse):
        return credential
    if credential.archived_at is not None:
        return CredentialView.of(credential, found.name)
    if not await catalogue.archive_credential(credential_id, tenant_id):
        return refuse(
            ErrorCode.CREDENTIAL_NOT_FOUND,
            CREDENTIAL_ABSENT,
            credential_id=str(credential_id),
            vault_id=str(vault_id),
        )
    retired = await catalogue.fetch_credential(credential_id, tenant_id)
    if retired is None:
        return refuse(
            ErrorCode.CREDENTIAL_NOT_FOUND,
            CREDENTIAL_ABSENT,
            credential_id=str(credential_id),
            vault_id=str(vault_id),
        )
    return CredentialView.of(retired, found.name)


@router.delete(
    "/vaults/{vault_id}/credentials/{credential_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    # No response model, for the reason the vault delete gives.
    response_model=None,
    responses=_NOT_FOUND,
)
async def delete_credential(
    vault_id: VaultId,
    credential_id: CredentialId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    catalogue: Annotated[VaultCatalogue, Depends(catalogue_of)],
    writer: Annotated[CredentialVaultWriter, Depends(writer_of)],
) -> Response | JSONResponse:
    """Revoke a credential: erase the value, then remove the row.

    **A delete that only removed the row would be a revocation that did not happen.**
    The Tool Gateway resolves a ref straight to a vault key and never consults these
    rows, so a value left behind goes on authenticating outbound calls for as long as
    any registration names it -- and the tenant, seeing the credential gone from every
    listing, would believe otherwise.

    Erase before remove, because the key is composed from the vault's name and this
    credential's name and removing the row is what destroys the second half of it. If
    the removal then fails, the tenant retries this call: `erase` is idempotent and the
    row goes on the second pass.

    Deleting from an archived vault is allowed, and it is the one write that is. A
    retirement stops a tenant adding to a namespace; it is not a reason to refuse them
    the ability to revoke what is already in it.

    **The 204 means the value is gone from the vault, not that every call has stopped.**
    The Tool Gateway holds a value it has read for `ToolCredentialBroker.HOLD_S`, so a
    Session that reached this server inside that window goes on presenting the old value
    until the hold lapses -- five minutes at the current setting. That is long enough to
    matter for the reason a credential usually gets deleted in a hurry, and it is
    written here because this route is where a tenant decides they are done: nobody
    revoking a leaked key thinks to go and read a broker's cache policy first. Rotating
    the credential at the far end is what stops a call now; this stops the platform
    handing the value out again.
    """
    found = await _vault_or_refusal(catalogue, vault_id, tenant_id)
    if isinstance(found, JSONResponse):
        return found
    credential = await _credential_or_refusal(
        catalogue, vault_id, credential_id, tenant_id
    )
    if isinstance(credential, JSONResponse):
        return credential
    await writer.erase(_entry_name(tenant_id, found.name, credential.name))
    if not await catalogue.delete_credential(credential_id, tenant_id):
        # Another delete won the race. Its loser is told what a caller naming an id
        # nobody registered is told, and the value is gone either way.
        return refuse(
            ErrorCode.CREDENTIAL_NOT_FOUND,
            CREDENTIAL_ABSENT,
            credential_id=str(credential_id),
            vault_id=str(vault_id),
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
