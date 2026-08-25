"""What a tenant registers when it brings a tool credential of its own.

Two names and a kind. **No type in this module holds a secret value** -- the value
arrives at the boundary, goes to the vault, and is never carried by a domain object,
which is what keeps it out of a repr, a structured log record and a stored row. The
inbound request type below is the one place a value appears, and it appears as a
`SecretStr` so that even there it does not render.

The pair of names is the whole design. `vault_names.scoped_vault_name` already composes
`map/tool-credential/<tenant>/<ref>` and the Tool Gateway already resolves it, so this
does not invent a second addressing scheme beside the working one: a credential's ref is
just `<vault>/<credential>`, and every existing resolution path reads it unchanged.
`api-parity.md` §C5 records that as the requirement -- add the registering half, leave
the resolving half alone -- and joining two names is what makes it literally true.

The separator is why both names are one segment. `parse_vault_ref` admits `/` inside a
ref, so if a name could contain one, vault `a/b` credential `c` and vault `a` credential
`b/c` would compose to the same vault key and one of a tenant's credentials would
silently shadow another. Migration `0029` refuses it in the database and this refuses it
at the boundary, where the tenant reads the refusal.

`CredentialKind` is a closed set of two because two is what the platform can attach
today, and each maps to one transport's attachment point rather than to a preference: a
bearer token becomes a request header on a Streamable HTTP server, an environment
variable becomes a variable in a spawned stdio server's environment. OAuth with
automatic refresh -- the third kind Anthropic offers -- is deliberately absent rather
than accepted and ignored. It needs a refresh loop, a token store and a clock, none of
which exist here, and a member admitted now would answer 201 to a registration that can
never be attached.
"""

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from managed_agent.core.ids import CredentialId, TenantId, VaultId

MAX_NAME_LEN: Final[int] = 48
"""Longest either name may be.

Both are joined into a ref bounded by `vault_names.MAX_REF_LEN` (128), and that ref is
composed under a prefix and a uuid before it becomes a Secrets Manager name. Two names
of 48 plus a separator is 97, which leaves the composed ref inside its own bound with
room that does not have to be recomputed every time the prefix changes.
"""

_NAME_PATTERN: Final[str] = rf"^[a-z0-9][a-z0-9_-]{{0,{MAX_NAME_LEN - 1}}}$"
"""What either name may spell: one segment, no separator, no dot.

Narrower than `vault_names._REF_PATTERN`, which admits `/` and `.` because it validates
a whole ref. Neither is admitted in a *component* of one: `/` is the separator and would
make the ref ambiguous, and `.` is refused because `..` is what `parse_vault_ref` treats
as an escape attempt -- a name that cannot contain a dot cannot contribute half of one
across the join.

Lowercase for the reason `ToolName` is lowercase: a store that treats `Prod` and `prod`
as one name while this treats them as two would let a tenant believe it had rotated a
credential it had not.
"""

VaultName = Annotated[str, Field(pattern=_NAME_PATTERN)]
CredentialName = Annotated[str, Field(pattern=_NAME_PATTERN)]


class CredentialKind(StrEnum):
    """Where a credential's value may be attached, named by what the value is.

    A closed set, and closed in the database too (`0029`'s check constraint), so the
    column can hold only a kind this platform has a place for.

    **Nothing reads it to decide where a value goes, and that is worth saying plainly
    because an earlier version of this docstring said the opposite.** It claimed the
    kind was read inside the Tool Gateway at the moment of a tool call. It is not: the
    attachment point comes from the *server* registration's transport -- `for_stdio`
    binds to `credential_env_var`, `for_http` to `credential_header` -- and the broker
    never sees this enum. So the two members are a label the tenant writes and reads
    back, matching the surface this one mirrors, and a credential whose kind disagrees
    with the transport of the server it is attached to is accepted by both ends and
    fails at the far one.

    Making it decide something means checking the two agree where the tenant is still
    on the phone, at `POST /v1/mcp_servers`. That check has to resolve a
    `credential_ref` to this row, and reading a ref is confined by
    `test_no_other_module_reads_a_credential_ref` to the two modules that compose a
    vault key from it -- a rule that closed a real cross-tenant read. So the check is
    not a small addition to the route; it needs either that rule widened with an
    argument for why resolving is not composing, or a port shaped so the route never
    holds the ref. Left undone rather than done through the guard.
    """

    STATIC_BEARER = "static_bearer"
    """The complete value of a request header, on a Streamable HTTP server.

    Spelled `static_bearer` because that is the value the modelled surface publishes
    for this kind, and a parity surface differing only in the spelling of a
    discriminator is the most annoying kind of incompatibility -- it type-checks, it
    reads correctly, and it fails at the one boundary a client cannot inspect.

    The complete value, not the token: the vault entry holds `Bearer abc123` rather than
    `abc123`, because `Authorization` wants the scheme and `X-Api-Key` does not, and a
    platform that derived the scheme from the header name would produce a header that is
    wrong in a way only the far end can see. `HttpAttachment.into_headers` documents the
    same rule from the other side, and this is where the tenant is told it.
    """

    ENVIRONMENT_VARIABLE = "environment_variable"
    """The value of one variable in a spawned stdio server's environment."""


def credential_ref(vault_name: str, credential_name: str) -> str:
    """The ref a Tool registration names to reach this credential.

    A function rather than an f-string at three call sites, because the join *is* the
    addressing scheme: the route that returns a ref to a tenant, the writer that
    composes the vault key, and any test that checks the two agree must produce one
    string, and a second spelling of the separator would put a credential at a name
    nothing reads.

    Unvalidated on purpose. Both names were parsed into `VaultName` and `CredentialName`
    before they could reach here, and `scoped_vault_name` parses the joined ref again on
    the way to a key. A third check here would be a third answer to a question already
    answered twice.
    """
    return f"{vault_name}/{credential_name}"


class Vault(BaseModel):
    """A tenant's named collection of credentials, as it is read back.

    Carries no credentials. A vault is a namespace rather than a container: listing what
    is in one is a separate read with its own page, so a vault with four hundred
    credentials is not a response body.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: VaultId
    tenant_id: TenantId
    name: VaultName
    created_at: datetime
    archived_at: datetime | None = None


class Credential(BaseModel):
    """One credential's name and shape, never its value.

    This is the whole response body of every read on the credential routes, and the
    value is absent rather than redacted. A redacted field teaches a reader that the
    value is *here and hidden*, which invites a later change to reveal it under some
    flag; an absent field says the value went somewhere else and this is not the place
    to ask. `last_four`-style disclosure is refused on the same ground -- it is a real
    substring of a real credential, handed out on a read that needs no authentication
    beyond the tenant header.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: CredentialId
    vault_id: VaultId
    tenant_id: TenantId
    name: CredentialName
    kind: CredentialKind
    created_at: datetime
    value_written_at: datetime
    archived_at: datetime | None = None

    def ref(self, vault_name: str) -> str:
        """The ref a Tool registration names to reach this credential.

        Takes the vault's name rather than reading it, because this object does not
        carry one: the row holds `vault_id`, and a name copied onto it at read time
        would be a second copy free to go stale against a vault that was renamed.
        """
        return credential_ref(vault_name, self.name)


class CredentialSubmission(BaseModel):
    """A credential being written: the shape, and the value, for as long as it takes.

    The one type here that touches a value, and it holds a `SecretStr` rather than a
    `str`. That is not decoration: this object is a request body, and a request body
    reaches an exception message, a validation error and a debug log by routes nobody
    writes deliberately. `SecretStr` renders as `**********` through every one of them,
    and the value comes out only where somebody typed `get_secret_value()`.

    `value` is required and non-empty. An empty credential is the failure mode worth
    refusing loudest: it writes a vault entry that exists, so every existence check
    passes, and every call authenticated with it fails at the far end with a message
    about the far end. `gap.md` names the same shape as out of scope -- an empty secret
    container created to satisfy an existence-only check -- and this is where a tenant
    would otherwise create one by accident.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: CredentialName
    kind: CredentialKind
    value: SecretStr = Field(min_length=1)


class VaultNotConfigured(RuntimeError):
    """This deployment was built with no vault behind it, and something asked for one.

    A platform fault rather than a tenant one, and it surfaces as one: the request was
    well-formed and would have worked against a correctly wired process. Raising is the
    point -- the alternative is a route that quietly answers 200 to a credential write
    that went nowhere, which is the single worst outcome on this surface because every
    later symptom appears at the far end of somebody else's MCP server.
    """


class UnconfiguredVaultCatalogue:
    """Refuses every read and write. What a `Platform` built with no database holds.

    A refusing default rather than `None`, so no caller tests the field before using it
    and no caller can forget to -- the same shape `UnconfiguredAttachments` uses, and
    for the same reason. `Platform` is constructed in roughly two dozen places, so a
    required field would break all of them at once, including branches being written
    right now where the field does not exist yet.

    Every method of the port is written out rather than caught by a `__getattr__`. The
    dynamic version was tried first, on the reasoning that eleven stubs is eleven
    chances to leave one out -- and `mypy --strict` refused it, because a class whose
    members are all `object` does not satisfy the Protocol. That refusal is the answer
    to the reasoning: the type checker is exactly the thing that catches a missing
    stub, so the explicit form is the one whose totality is verified rather than
    asserted, and a method added to the port fails here at the assignment in
    `composition.py` rather than at the first request that calls it.
    """

    def _refuse(self, method: str) -> VaultNotConfigured:
        return VaultNotConfigured(
            f"this deployment has no vault catalogue, so {method}() cannot answer; "
            "a credential registered here would be recorded nowhere"
        )

    async def insert_vault(self, vault: Vault, /) -> None:
        raise self._refuse("insert_vault")

    async def fetch_vault(
        self, vault_id: VaultId, tenant_id: TenantId, /
    ) -> Vault | None:
        raise self._refuse("fetch_vault")

    async def page_vaults(
        self,
        tenant_id: TenantId,
        after: tuple[datetime, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Vault]:
        raise self._refuse("page_vaults")

    async def archive_vault(self, vault_id: VaultId, tenant_id: TenantId, /) -> bool:
        raise self._refuse("archive_vault")

    async def delete_vault(self, vault_id: VaultId, tenant_id: TenantId, /) -> bool:
        raise self._refuse("delete_vault")

    async def insert_credential(self, credential: Credential, /) -> None:
        raise self._refuse("insert_credential")

    async def fetch_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> Credential | None:
        raise self._refuse("fetch_credential")

    async def page_credentials(
        self,
        vault_id: VaultId,
        tenant_id: TenantId,
        after: tuple[datetime, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Credential]:
        raise self._refuse("page_credentials")

    async def archive_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> bool:
        raise self._refuse("archive_credential")

    async def delete_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> bool:
        raise self._refuse("delete_credential")

    async def mark_value_written(
        self, credential_id: CredentialId, tenant_id: TenantId, at: datetime, /
    ) -> bool:
        raise self._refuse("mark_value_written")


class UnconfiguredCredentialWriter:
    """Refuses every write. What a `Platform` built with no secret store holds."""

    async def put(self, name: str, value: str) -> None:
        raise VaultNotConfigured(
            "this deployment has no credential vault, so nothing can be written at "
            f"{name}; accepting the value here would report success for a credential "
            "that no tool call could ever attach"
        )

    async def erase(self, name: str) -> None:
        raise VaultNotConfigured(
            "this deployment has no credential vault, so nothing at "
            f"{name} can be erased; reporting success would report a revocation "
            "that did not happen"
        )
