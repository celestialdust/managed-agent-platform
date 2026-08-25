"""The `/v1/vaults` surface, over the real routes, against in-memory ports.

Tier 1: the real router, the real refusal envelope, real HTTP through an ASGI
transport, and two fakes that keep the guarantees the database keeps. No cluster and no
container -- what is graded here is the route: which status comes back, what the body
says, and, on every write, what the two ports actually hold afterwards.

**The fakes are fakes and not mocks.** The catalogue keeps two dicts and raises
`VaultNameTaken` / `CredentialNameTaken` exactly where `vault_name_is_one_per_tenant`
and `credential_name_is_one_per_vault` would fire, which is what the port promises a
caller -- so the 409s below are produced by the same path production produces them by,
not by a branch the test arranged. The writer keeps a `name -> value` dict, so
"the value was written", "it was written under the key the Tool Gateway reads" and "it
is gone" are read off the store rather than asserted about a call.

The claim this file exists for, which no other test in the tree can make, is the
negative one: **a credential's value never comes back**. It is asserted against the raw
response text of every route on the surface -- including the refusals, where a
framework's own diagnostics are what would echo a submitted body -- because a stray
field on a model is invisible to an assertion that reads a parsed body key by key.

The seam test is the other one worth naming. The control plane writes a vault key and
the Tool Gateway reads one, in two processes under two IAM roles, and nothing at
runtime compares them. `test_the_key_written_is_the_key_the_gateway_would_read`
composes both and asserts they are one string, so a change to either composition fails
here rather than at somebody else's MCP server.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.exc import IntegrityError

# `test_app`'s stand-ins, imported rather than rebuilt, for the reason
# `test_environment_lifecycle.py` gives: every name there is module-level and public,
# `tests/` carries no `__init__.py`, and pytest's own sys.path entry is what makes the
# bare name work. Nine of `Platform`'s ports are irrelevant here and every one of them
# raises if a route touches it, which is the assertion that this surface reads nothing
# but the two ports it declares.
from test_app import (
    UnusedEnvironmentStore,
    UnusedLog,
    UnusedRegistry,
    UnusedSessionRegistry,
    UnusedToolRegistry,
    UnusedWebhooks,
)

from managed_agent.composition import Platform
from managed_agent.control.api import refusals
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import vaults
from managed_agent.control.api.routes.vaults import MAX_CREDENTIALS_PER_VAULT
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import CredentialId, TenantId, VaultId, new_vault_id
from managed_agent.core.ports import CredentialNameTaken, VaultNameTaken
from managed_agent.core.vault_catalogue import Credential, CredentialKind, Vault
from managed_agent.gateway.tool.credential_broker import vault_name

SECRET = "Bearer sk-live-" + "9f3c1d7b2a" * 3
"""The value submitted throughout, long and distinctive enough that a substring of it
in any response body is unambiguous. Never a real credential and never printed."""

OTHER_SECRET = "Bearer sk-live-" + "0e5a8c4f6b" * 3


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeVaultCatalogue:
    """The catalogue port over two dicts, keeping the database's two unique names.

    Every read takes the tenant as a query term rather than checking it afterwards, the
    way the port says it must, so there is no point in this object at which a caller's
    row set contains somebody else's row.
    """

    def __init__(self) -> None:
        self.vaults: dict[VaultId, Vault] = {}
        self.credentials: dict[CredentialId, Credential] = {}

    async def insert_vault(self, vault: Vault, /) -> None:
        taken = any(
            row.tenant_id == vault.tenant_id and row.name == vault.name
            for row in self.vaults.values()
        )
        if taken:
            raise VaultNameTaken(vault.name)
        self.vaults[vault.id] = vault

    async def fetch_vault(
        self, vault_id: VaultId, tenant_id: TenantId, /
    ) -> Vault | None:
        found = self.vaults.get(vault_id)
        return found if found is not None and found.tenant_id == tenant_id else None

    async def page_vaults(
        self,
        tenant_id: TenantId,
        after: tuple[datetime, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Vault]:
        rows = [
            row
            for row in self.vaults.values()
            if row.tenant_id == tenant_id
            and (include_archived or row.archived_at is None)
        ]
        return _page(rows, after, limit, lambda row: (row.created_at, row.id))

    async def archive_vault(self, vault_id: VaultId, tenant_id: TenantId, /) -> bool:
        found = await self.fetch_vault(vault_id, tenant_id)
        if found is None:
            return False
        self.vaults[vault_id] = found.model_copy(
            update={"archived_at": datetime.now(UTC)}
        )
        return True

    async def delete_vault(self, vault_id: VaultId, tenant_id: TenantId, /) -> bool:
        """Refuse a vault that still holds credentials, the way the real store does.

        This fake used to cascade -- delete the vault and drop its credential rows with
        it -- and that modelled a database nobody deployed. `vault_credential`'s foreign
        key names `(vault.id, vault.tenant_id)` with no cascade, so `PostgresVaultStore.
        delete_vault` raises `IntegrityError` here and
        `test_a_vault_that_still_holds_a_credential_cannot_be_deleted` pins it. Every
        route case below was green against the cascade while the deployed path answered
        500, which is the whole argument for a fake refusing exactly where its subject
        refuses.
        """
        if await self.fetch_vault(vault_id, tenant_id) is None:
            return False
        if any(row.vault_id == vault_id for row in self.credentials.values()):
            raise IntegrityError(
                "DELETE FROM vault", {}, Exception("vault_credential_vault_id_fkey")
            )
        del self.vaults[vault_id]
        return True

    async def insert_credential(self, credential: Credential, /) -> None:
        taken = any(
            row.vault_id == credential.vault_id and row.name == credential.name
            for row in self.credentials.values()
        )
        if taken:
            raise CredentialNameTaken(credential.vault_id, credential.name)
        self.credentials[credential.id] = credential

    async def fetch_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> Credential | None:
        found = self.credentials.get(credential_id)
        return found if found is not None and found.tenant_id == tenant_id else None

    async def page_credentials(
        self,
        vault_id: VaultId,
        tenant_id: TenantId,
        after: tuple[datetime, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Credential]:
        rows = [
            row
            for row in self.credentials.values()
            if row.vault_id == vault_id
            and row.tenant_id == tenant_id
            and (include_archived or row.archived_at is None)
        ]
        return _page(rows, after, limit, lambda row: (row.created_at, row.id))

    async def archive_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> bool:
        found = await self.fetch_credential(credential_id, tenant_id)
        if found is None:
            return False
        self.credentials[credential_id] = found.model_copy(
            update={"archived_at": datetime.now(UTC)}
        )
        return True

    async def delete_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> bool:
        if await self.fetch_credential(credential_id, tenant_id) is None:
            return False
        del self.credentials[credential_id]
        return True

    async def mark_value_written(
        self, credential_id: CredentialId, tenant_id: TenantId, at: datetime, /
    ) -> bool:
        found = await self.fetch_credential(credential_id, tenant_id)
        if found is None:
            return False
        self.credentials[credential_id] = found.model_copy(
            update={"value_written_at": at}
        )
        return True


def _page(
    rows: list[Any],
    after: tuple[datetime, UUID] | None,
    limit: int,
    position: Any,
) -> list[Any]:
    """One page of `rows` in creation order, starting strictly after `after`.

    Ordered by the pair rather than by the instant alone, because two rows written in
    one microsecond would otherwise have no order at all and a page boundary between
    them would repeat one and drop the other -- which is the defect the cursor's second
    half exists to prevent, so the fake must not paper over it.
    """
    ordered = sorted(rows, key=position)
    if after is not None:
        ordered = [row for row in ordered if position(row) > after]
    return ordered[:limit]


class FakeCredentialWriter:
    """The writing half of the vault, as a `name -> value` dict.

    It holds values, which is what makes "the value reached the vault under this exact
    key" and "the value is gone" readable facts rather than claims about calls. It has
    no `fetch`, exactly as the port has none, so nothing in a test can hand a value back
    to a route either.
    """

    def __init__(self) -> None:
        self.entries: dict[str, str] = {}
        self.erased: list[str] = []
        self.fail_next_erase = False
        """Make one erase fail, so a caller's half-finished delete can be graded.

        A vault delete erases many values and then removes the rows, and the interesting
        question is what it leaves behind when it stops partway. Without a way to make
        one call fail, that path is reachable only by an outage.
        """

    async def put(self, name: str, value: str) -> None:
        self.entries[name] = value

    async def erase(self, name: str) -> None:
        if self.fail_next_erase:
            self.fail_next_erase = False
            raise RuntimeError(f"the vault refused to erase {name}")
        self.entries.pop(name, None)
        self.erased.append(name)


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Harness:
    app: FastAPI
    catalogue: FakeVaultCatalogue
    writer: FakeCredentialWriter
    tenant: TenantId


def a_harness() -> Harness:
    """The vault router over one platform, refused the way the real app refuses.

    The envelope install is load-bearing rather than decoration, and it goes before the
    router because that is the order it covers it in. Two refusals exercised below are
    *raised* rather than returned -- a missing tenant header refuses in a dependency,
    and a body FastAPI rejects never enters a route at all -- so without these handlers
    a hand-mounted router would pass every route-level test here while disagreeing with
    the deployed app on both.
    """
    catalogue = FakeVaultCatalogue()
    writer = FakeCredentialWriter()
    unused = UnusedLog()
    app = FastAPI()
    app.state.platform = Platform(
        event_log_append=unused,
        event_log_range=unused,
        definition_registry=UnusedRegistry(),
        tool_registry=UnusedToolRegistry(),
        session_registry=UnusedSessionRegistry(),
        webhooks=UnusedWebhooks(),
        environment_store=UnusedEnvironmentStore(),
        turn_dispatch=NoPodTransport(),
        file_store=unconfigured_file_store(),
        vault_catalogue=catalogue,
        credential_writer=writer,
    )
    refusals.install_request_envelope(app)
    app.include_router(vaults.router, prefix="/v1")
    return Harness(app=app, catalogue=catalogue, writer=writer, tenant=a_tenant())


def a_tenant() -> TenantId:
    return TenantId(uuid4())


def caller(app: FastAPI, tenant: TenantId) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://platform",
        headers={TENANT_HEADER: str(tenant)},
    )


def _submission(**overrides: Any) -> dict[str, Any]:
    """A credential body, so a case names only the field it is about."""
    return {
        "name": "prod-token",
        # The member rather than its spelling, so a rename of the closed set fails at
        # import here instead of arriving as a puzzling 400 from a field check.
        "kind": CredentialKind.STATIC_BEARER.value,
        "value": SECRET,
        **overrides,
    }


async def _a_vault(client: httpx.AsyncClient, name: str = "vendor") -> dict[str, Any]:
    response = await client.post("/v1/vaults", json={"name": name})
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


async def _a_credential(
    client: httpx.AsyncClient, vault_id: str, **overrides: Any
) -> dict[str, Any]:
    response = await client.post(
        f"/v1/vaults/{vault_id}/credentials", json=_submission(**overrides)
    )
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


# --------------------------------------------------------------------------------------
# The ref, and the key behind it
# --------------------------------------------------------------------------------------


async def test_a_created_credential_answers_with_the_ref_a_registration_names() -> None:
    """The `ref` is the whole point of the response: it is what the tenant pastes into
    a `POST /v1/mcp_servers` body, and without it they would have two ids and a guess.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client, name="vendor")
        credential = await _a_credential(client, vault["id"], name="prod-token")
    assert credential["ref"] == "vendor/prod-token"


async def test_the_ref_a_read_answers_is_the_ref_the_create_answered() -> None:
    """A read, a list and a rotate must not describe one credential three ways.

    A tenant that wrote down the ref from a create and later compared it with a listing
    must not find a difference that only means two code paths built the string.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        created = await _a_credential(client, vault["id"])
        read = await client.get(f"/v1/vaults/{vault['id']}/credentials/{created['id']}")
        listed = await client.get(f"/v1/vaults/{vault['id']}/credentials")
        rotated = await client.post(
            f"/v1/vaults/{vault['id']}/credentials/{created['id']}",
            json={"value": OTHER_SECRET},
        )
    assert read.json()["ref"] == created["ref"]
    assert listed.json()["data"][0]["ref"] == created["ref"]
    assert rotated.json()["ref"] == created["ref"]


async def test_the_key_written_is_the_key_the_gateway_would_read() -> None:
    """The seam: two processes compose this name and nothing at runtime compares them.

    The control plane writes the entry when a tenant registers a credential; the Tool
    Gateway composes the same name to fetch it on an outbound call, under a different
    IAM role in a different process. If the two ever disagree the write lands where no
    read looks, and the only symptom is an authentication failure at somebody else's
    MCP server. So both are composed here and asserted to be one string.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client, name="vendor")
        credential = await _a_credential(client, vault["id"], name="prod-token")
    expected = vault_name(harness.tenant, credential["ref"])
    assert list(harness.writer.entries) == [expected]
    assert expected.startswith("map/tool-credential/")
    assert str(harness.tenant) in expected


# --------------------------------------------------------------------------------------
# The value never comes back
# --------------------------------------------------------------------------------------


async def test_no_response_on_this_surface_carries_the_value() -> None:
    """**The claim this file exists for**, asserted against raw response text.

    Against the text and not a parsed body, because the failure being guarded is a
    field nobody meant to add -- a `value`, a `last_four`, a model that gained one
    -- and an assertion that walks known keys cannot see a key it does not know to
    look for. Every route that can answer with a credential is driven, plus a refusal,
    because a refusal is where a framework's own diagnostics would echo a submitted
    body.
    """
    harness = a_harness()
    seen: list[httpx.Response] = []
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        created = await client.post(
            f"/v1/vaults/{vault['id']}/credentials", json=_submission()
        )
        seen.append(created)
        credential_id = created.json()["id"]
        base = f"/v1/vaults/{vault['id']}/credentials/{credential_id}"
        seen.append(await client.get(f"/v1/vaults/{vault['id']}/credentials"))
        seen.append(await client.get(base))
        seen.append(await client.post(base, json={"value": OTHER_SECRET}))
        seen.append(await client.post(f"{base}/archive"))
        seen.append(await client.get(f"/v1/vaults/{vault['id']}"))
        seen.append(await client.get("/v1/vaults"))
        # A refusal, and the one whose body is built by the framework rather than by
        # this codebase: an unknown `kind` alongside a perfectly good value.
        seen.append(
            await client.post(
                f"/v1/vaults/{vault['id']}/credentials",
                json=_submission(name="other", kind="oauth_refresh"),
            )
        )
        # And a refusal this codebase authors, carrying a name that was submitted
        # beside the value.
        seen.append(
            await client.post(
                f"/v1/vaults/{vault['id']}/credentials", json=_submission()
            )
        )
    for response in seen:
        assert SECRET not in response.text, response.request.url
        assert OTHER_SECRET not in response.text, response.request.url
        assert "sk-live" not in response.text, response.request.url


async def test_the_credential_body_declares_no_value_field() -> None:
    """The absence is structural, not a filter one route remembered to apply.

    Read off the published schema rather than off one response, so a field added to the
    model fails here even if no test happened to drive the route that would return it.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        schema = (await client.get("/openapi.json")).json()
    fields = schema["components"]["schemas"]["CredentialView"]["properties"]
    assert "value" not in fields
    assert not [name for name in fields if "secret" in name or "last" in name]


# --------------------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------------------


async def test_another_tenants_vault_reads_as_one_that_was_never_registered() -> None:
    """A distinguishable answer would be an existence oracle over other tenants' ids."""
    harness = a_harness()
    stranger = a_tenant()
    async with caller(harness.app, harness.tenant) as owner:
        vault = await _a_vault(owner)
    async with caller(harness.app, stranger) as other:
        theirs = await other.get(f"/v1/vaults/{vault['id']}")
        invented = await other.get(f"/v1/vaults/{uuid4()}")
    assert theirs.status_code == STATUS_FOR[ErrorCode.VAULT_NOT_FOUND]
    assert theirs.json()["error"]["code"] == ErrorCode.VAULT_NOT_FOUND.value
    assert theirs.json()["error"]["message"] == invented.json()["error"]["message"]


async def test_another_tenants_credential_reads_as_absent() -> None:
    harness = a_harness()
    stranger = a_tenant()
    async with caller(harness.app, harness.tenant) as owner:
        vault = await _a_vault(owner)
        credential = await _a_credential(owner, vault["id"])
    async with caller(harness.app, stranger) as other:
        response = await other.get(
            f"/v1/vaults/{vault['id']}/credentials/{credential['id']}"
        )
    assert response.status_code == STATUS_FOR[ErrorCode.VAULT_NOT_FOUND]


async def test_a_credential_is_not_reachable_through_another_of_its_vaults() -> None:
    """The vault in the path is checked against the row rather than trusted.

    Without it a credential could be read, rotated or revoked through any vault id the
    tenant happens to hold, and the path would be stating something untrue about where
    the credential lives -- which matters because the vault's name is half its ref.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        first = await _a_vault(client, name="vendor")
        second = await _a_vault(client, name="other")
        credential = await _a_credential(client, first["id"])
        response = await client.get(
            f"/v1/vaults/{second['id']}/credentials/{credential['id']}"
        )
    assert response.status_code == STATUS_FOR[ErrorCode.CREDENTIAL_NOT_FOUND]
    assert response.json()["error"]["code"] == ErrorCode.CREDENTIAL_NOT_FOUND.value


# --------------------------------------------------------------------------------------
# Names are one per scope
# --------------------------------------------------------------------------------------


async def test_a_second_vault_of_one_name_is_refused() -> None:
    """Two vaults of one name would make one ref address two credentials."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        await _a_vault(client, name="vendor")
        again = await client.post("/v1/vaults", json={"name": "vendor"})
    assert again.status_code == STATUS_FOR[ErrorCode.VAULT_NAME_TAKEN]
    assert again.json()["error"]["code"] == ErrorCode.VAULT_NAME_TAKEN.value
    assert len(harness.catalogue.vaults) == 1


async def test_one_name_per_tenant_and_not_per_platform() -> None:
    """Two tenants may each hold a vault called `vendor`: the key scopes by tenant."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as owner:
        await _a_vault(owner, name="vendor")
    async with caller(harness.app, a_tenant()) as other:
        response = await other.post("/v1/vaults", json={"name": "vendor"})
    assert response.status_code == 201


async def test_a_second_credential_of_one_name_in_one_vault_is_refused() -> None:
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        await _a_credential(client, vault["id"], name="prod-token")
        again = await client.post(
            f"/v1/vaults/{vault['id']}/credentials", json=_submission(name="prod-token")
        )
    assert again.status_code == STATUS_FOR[ErrorCode.CREDENTIAL_NAME_TAKEN]
    assert again.json()["error"]["code"] == ErrorCode.CREDENTIAL_NAME_TAKEN.value
    assert len(harness.catalogue.credentials) == 1


async def test_a_refused_duplicate_does_not_overwrite_the_credential_it_hit() -> None:
    """**The refused create must not destroy what it collided with.**

    The vault key is composed from the tenant and the two names and from nothing else,
    so a create naming a credential that already exists composes *the same key*. The
    value goes to the vault before the row goes to the catalogue -- so if the collision
    were only discovered by the catalogue, the tenant would be handed a 409 while the
    credential they already had sat behind the value from the request just rejected: a
    silent rotation performed by a refusal.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        credential = await _a_credential(client, vault["id"], name="prod-token")
        again = await client.post(
            f"/v1/vaults/{vault['id']}/credentials",
            json=_submission(name="prod-token", value=OTHER_SECRET),
        )
    key = vault_name(harness.tenant, credential["ref"])
    assert again.status_code == STATUS_FOR[ErrorCode.CREDENTIAL_NAME_TAKEN]
    assert harness.writer.entries[key] == SECRET


async def test_an_archived_vault_still_holds_its_name() -> None:
    """A retired vault's name is not released, because the row is still there and the
    database's uniqueness does not exempt it. A route that scanned only live rows would
    answer "that name is free" about a name the insert is about to be refused for."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client, name="vendor")
        await client.post(f"/v1/vaults/{vault['id']}/archive")
        again = await client.post("/v1/vaults", json={"name": "vendor"})
    assert again.status_code == STATUS_FOR[ErrorCode.VAULT_NAME_TAKEN]


# --------------------------------------------------------------------------------------
# Archived is read-only
# --------------------------------------------------------------------------------------


async def test_a_credential_cannot_be_registered_in_an_archived_vault() -> None:
    """Creating inside a retired vault is a write, and a retirement is terminal.

    The refusal must also leave the vault untouched: a 409 that had already written the
    value would be a secret sitting in the vault for a registration that was refused.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        await client.post(f"/v1/vaults/{vault['id']}/archive")
        response = await client.post(
            f"/v1/vaults/{vault['id']}/credentials", json=_submission()
        )
    assert response.status_code == STATUS_FOR[ErrorCode.VAULT_ARCHIVED]
    assert response.json()["error"]["code"] == ErrorCode.VAULT_ARCHIVED.value
    assert harness.writer.entries == {}
    assert harness.catalogue.credentials == {}


async def test_a_rotation_into_an_archived_vault_is_refused() -> None:
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        credential = await _a_credential(client, vault["id"])
        await client.post(f"/v1/vaults/{vault['id']}/archive")
        response = await client.post(
            f"/v1/vaults/{vault['id']}/credentials/{credential['id']}",
            json={"value": OTHER_SECRET},
        )
    assert response.status_code == STATUS_FOR[ErrorCode.VAULT_ARCHIVED]
    assert list(harness.writer.entries.values()) == [SECRET]


async def test_retiring_a_vault_twice_answers_with_the_first_retirement() -> None:
    """A fresh timestamp on a retry would claim the vault stopped accepting writes at
    the moment of the retry -- a false fact about when the refusals began."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        first = await client.post(f"/v1/vaults/{vault['id']}/archive")
        second = await client.post(f"/v1/vaults/{vault['id']}/archive")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["archived_at"] == second.json()["archived_at"]
    assert first.json()["archived_at"] is not None


async def test_an_archived_vault_is_absent_from_a_listing_unless_asked_for() -> None:
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client, name="vendor")
        await _a_vault(client, name="live")
        await client.post(f"/v1/vaults/{vault['id']}/archive")
        default = await client.get("/v1/vaults")
        asked = await client.get("/v1/vaults", params={"include_archived": True})
    assert [row["name"] for row in default.json()["data"]] == ["live"]
    assert sorted(row["name"] for row in asked.json()["data"]) == ["live", "vendor"]


# --------------------------------------------------------------------------------------
# Writing, rotating and revoking
# --------------------------------------------------------------------------------------


async def test_the_value_reaches_the_vault_and_the_row_reaches_the_catalogue() -> None:
    """Both halves of the create landed, and the value is the one submitted."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        credential = await _a_credential(client, vault["id"])
    key = vault_name(harness.tenant, credential["ref"])
    assert harness.writer.entries[key] == SECRET
    assert len(harness.catalogue.credentials) == 1


async def test_a_rotation_overwrites_at_the_same_name() -> None:
    """**Rotation is an overwrite at one key, and that is the whole feature.**

    A rotation that minted a second key would leave the old value live at the old key
    with nothing left to erase it, and every registration already naming the ref would
    go on attaching the value the tenant believes they replaced.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        credential = await _a_credential(client, vault["id"])
        before = credential["value_written_at"]
        response = await client.post(
            f"/v1/vaults/{vault['id']}/credentials/{credential['id']}",
            json={"value": OTHER_SECRET},
        )
    key = vault_name(harness.tenant, credential["ref"])
    assert response.status_code == 200
    assert list(harness.writer.entries) == [key]
    assert harness.writer.entries[key] == OTHER_SECRET
    assert response.json()["ref"] == credential["ref"]
    assert response.json()["value_written_at"] >= before


async def test_an_empty_value_is_refused_and_writes_nothing() -> None:
    """An empty credential passes every existence check and fails at the far end."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        response = await client.post(
            f"/v1/vaults/{vault['id']}/credentials", json=_submission(value="")
        )
    assert response.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert harness.writer.entries == {}


async def test_deleting_a_credential_erases_its_value() -> None:
    """**A delete that only removed the row would be a revocation that did not happen.**

    The Tool Gateway resolves a ref straight to a vault key and never consults these
    rows, so a value left behind goes on authenticating outbound calls while the tenant,
    seeing the credential gone from every listing, believes otherwise.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        credential = await _a_credential(client, vault["id"])
        key = vault_name(harness.tenant, credential["ref"])
        assert harness.writer.entries[key] == SECRET
        response = await client.delete(
            f"/v1/vaults/{vault['id']}/credentials/{credential['id']}"
        )
    assert response.status_code == 204
    assert harness.writer.entries == {}
    assert harness.writer.erased == [key]
    assert harness.catalogue.credentials == {}


async def test_deleting_a_vault_erases_every_value_it_held() -> None:
    """Rows removed with values left behind would be secrets outliving every record of
    themselves -- and a tenant registering a vault of the same name again would
    resurrect all of them."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client, name="vendor")
        first = await _a_credential(client, vault["id"], name="prod-token")
        second = await _a_credential(client, vault["id"], name="staging-token")
        response = await client.delete(f"/v1/vaults/{vault['id']}")
    assert response.status_code == 204
    assert harness.writer.entries == {}
    assert sorted(harness.writer.erased) == sorted(
        vault_name(harness.tenant, row["ref"]) for row in (first, second)
    )
    assert harness.catalogue.vaults == {}


async def test_deleting_a_vault_this_tenant_does_not_hold_erases_nothing() -> None:
    """A tenant that mistyped an id and got 204 would believe it had revoked
    credentials it had not."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as owner:
        vault = await _a_vault(owner)
        await _a_credential(owner, vault["id"])
    async with caller(harness.app, a_tenant()) as stranger:
        response = await stranger.delete(f"/v1/vaults/{vault['id']}")
    assert response.status_code == STATUS_FOR[ErrorCode.VAULT_NOT_FOUND]
    assert harness.writer.erased == []
    assert len(harness.writer.entries) == 1


async def test_a_deleted_credential_is_gone_from_every_read() -> None:
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        credential = await _a_credential(client, vault["id"])
        base = f"/v1/vaults/{vault['id']}/credentials/{credential['id']}"
        await client.delete(base)
        read = await client.get(base)
        listed = await client.get(f"/v1/vaults/{vault['id']}/credentials")
        again = await client.delete(base)
    assert read.status_code == STATUS_FOR[ErrorCode.CREDENTIAL_NOT_FOUND]
    assert listed.json()["data"] == []
    assert again.status_code == STATUS_FOR[ErrorCode.CREDENTIAL_NOT_FOUND]


async def test_retiring_a_credential_leaves_it_readable_and_revokes_nothing() -> None:
    """Archive marks a row; DELETE revokes. Conflating them would either strand live
    secrets or break every Session already authenticating with one."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        credential = await _a_credential(client, vault["id"])
        base = f"/v1/vaults/{vault['id']}/credentials/{credential['id']}"
        first = await client.post(f"{base}/archive")
        second = await client.post(f"{base}/archive")
        read = await client.get(base)
    assert first.json()["archived_at"] is not None
    assert first.json()["archived_at"] == second.json()["archived_at"]
    assert read.json()["archived_at"] == first.json()["archived_at"]
    assert harness.writer.erased == []


# --------------------------------------------------------------------------------------
# Paging
# --------------------------------------------------------------------------------------


async def test_a_walk_covers_every_vault_exactly_once() -> None:
    """The store is asked for one row more than is returned, and that extra row is the
    whole answer to "is there another page"."""
    harness = a_harness()
    stamp = datetime(2026, 8, 25, 12, tzinfo=UTC)
    for index in range(7):
        await harness.catalogue.insert_vault(
            Vault(
                id=new_vault_id(),
                tenant_id=harness.tenant,
                name=f"vendor-{index}",
                created_at=stamp + timedelta(seconds=index),
            )
        )
    seen: list[str] = []
    async with caller(harness.app, harness.tenant) as client:
        page: str | None = None
        for _ in range(10):
            params: dict[str, Any] = {"limit": 3}
            if page is not None:
                params["page"] = page
            body = (await client.get("/v1/vaults", params=params)).json()
            seen.extend(row["name"] for row in body["data"])
            page = body["next_page"]
            if page is None:
                break
    assert sorted(seen) == [f"vendor-{index}" for index in range(7)]
    assert len(seen) == len(set(seen))


async def test_a_cursor_this_surface_did_not_issue_is_refused() -> None:
    """Starting over on a bad cursor would silently hand back the newest page again,
    which reads as the walk having looped rather than failed."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        vaults_page = await client.get("/v1/vaults", params={"page": "not-a-cursor"})
        credentials_page = await client.get(
            f"/v1/vaults/{vault['id']}/credentials", params={"page": "not-a-cursor"}
        )
    for response in (vaults_page, credentials_page):
        assert response.status_code == STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]
        assert (
            response.json()["error"]["code"]
            == ErrorCode.PAGINATION_CURSOR_INVALID.value
        )


async def test_a_page_larger_than_the_surface_publishes_is_refused() -> None:
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        response = await client.get("/v1/vaults", params={"limit": 101})
    assert response.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert "limit" in response.json()["error"]["detail"]["fields"]


async def test_listing_the_credentials_of_a_vault_nobody_registered_is_refused() -> (
    None
):
    """Emptiness is not an answer about a vault this tenant does not hold: a caller
    could otherwise page an unknown id and read the empty page as a fact about it."""
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        response = await client.get(f"/v1/vaults/{uuid4()}/credentials")
    assert response.status_code == STATUS_FOR[ErrorCode.VAULT_NOT_FOUND]


# --------------------------------------------------------------------------------------
# The names the boundary refuses
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "vendor/prod",
        "vendor.prod",
        "Vendor",
        "-vendor",
        "v" * 49,
        "",
    ],
)
async def test_a_name_that_would_make_a_ref_ambiguous_is_refused(name: str) -> None:
    """A `/` in either name would make `<vault>/<credential>` address two things.

    Vault `a/b` credential `c` and vault `a` credential `b/c` compose to one key, so one
    of a tenant's credentials would silently shadow another -- resolved inside the Tool
    Gateway, where nothing can ask which was meant. A `.` is refused because `..` is
    what `parse_vault_ref` treats as an escape attempt, and a name that cannot hold a
    dot cannot contribute half of one across the join.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        response = await client.post("/v1/vaults", json={"name": name})
    assert response.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert harness.catalogue.vaults == {}


async def test_a_request_with_no_tenant_header_is_refused_before_anything_is_read() -> (
    None
):
    harness = a_harness()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=harness.app), base_url="http://platform"
    ) as client:
        response = await client.post("/v1/vaults", json={"name": "vendor"})
    assert response.status_code == STATUS_FOR[ErrorCode.REQUEST_TENANT_MISSING]
    assert harness.catalogue.vaults == {}


# --------------------------------------------------------------------------------------
# Deleting a vault that is not empty
# --------------------------------------------------------------------------------------


async def test_deleting_a_vault_removes_the_credentials_it_holds() -> None:
    """A vault holding credentials deletes, values and rows together.

    The foreign key on `vault_credential` names `(vault.id, vault.tenant_id)` with no
    cascade, so the store refuses to remove a vault that still has rows pointing at it
    -- `test_a_vault_that_still_holds_a_credential_cannot_be_deleted` in the adapter
    suite pins that. The route is therefore what empties it, and it has to do so in the
    order the module docstring forces: a value is erased before the row naming it goes,
    because the key that value sits under is composed from the two names and deleting
    the row destroys the only thing that can address it.

    This case did not exist while the route was written, and the fake it runs against
    used to cascade. Both had to be wrong at once for the deployed path to answer 500
    on a delete with a green suite behind it.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        first = await _a_credential(client, vault["id"], name="prod-token")
        second = await _a_credential(client, vault["id"], name="staging-token")

        answered = await client.delete(f"/v1/vaults/{vault['id']}")

    assert answered.status_code == 204, answered.text
    assert harness.catalogue.vaults == {}, "the vault outlived its own delete"
    assert harness.catalogue.credentials == {}, (
        "the vault is gone and its credential rows are not, so a tenant listing vaults "
        "sees nothing while rows remain addressing an id no vault has"
    )
    assert harness.writer.entries == {}, (
        f"{len(harness.writer.entries)} value(s) outlived the vault that named them. "
        "The key is composed from the vault's name and the credential's name, so "
        "nothing remaining can address them -- and a tenant registering a vault of the "
        "same name would resurrect every one. Refs affected: "
        f"{sorted(harness.writer.entries)}"
    )
    assert first["id"] != second["id"]


async def test_a_vault_delete_that_cannot_finish_leaves_the_vault_standing() -> None:
    """A failed erase refuses, and does not remove the vault behind the failure.

    The residue worth having is rows whose values are gone: those credentials fail at
    the far end, loudly, and the tenant's retry erases the rest -- `erase` is
    idempotent, so a retry repairs rather than compounds. The residue worth refusing is
    a vault removed while a value it named is still live in the store, because nothing
    left can compose the key that value sits under.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        await _a_credential(client, vault["id"], name="prod-token")
        harness.writer.fail_next_erase = True

        with pytest.raises(RuntimeError):
            await client.delete(f"/v1/vaults/{vault['id']}")

    assert vault["id"] in [str(one) for one in harness.catalogue.vaults], (
        "the erase failed and the vault was removed anyway, so the value still in the "
        "store is one nothing can name"
    )


# --------------------------------------------------------------------------------------
# The ceiling on one vault
# --------------------------------------------------------------------------------------


async def test_a_vault_holds_no_more_credentials_than_the_published_ceiling() -> None:
    """The twenty-first create is refused, and the twentieth is not.

    Both halves, because either alone is satisfiable by breaking the other: a route
    that refused every create would pass the refusal assertion, and one that refused
    none would pass the acceptance. The number itself is asserted against
    `MAX_CREDENTIALS_PER_VAULT` rather than against a literal `20` written here -- a
    copy is free to fall behind the constant, and then this reads as a ceiling test
    while grading a number nothing enforces.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        for index in range(MAX_CREDENTIALS_PER_VAULT):
            await _a_credential(client, vault["id"], name=f"token-{index}")

        refused = await client.post(
            f"/v1/vaults/{vault['id']}/credentials", json=_submission(name="one-more")
        )

    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == ErrorCode.VAULT_FULL.value


async def test_a_full_vault_refuses_before_it_writes_the_value() -> None:
    """Nothing reaches the secret store on the way to the ceiling refusal.

    The order matters here for the same reason it does on a duplicate name: this
    surface writes the value before the row, and the key is composed from the tenant
    and the two names alone. A refusal that arrived after the write would leave a
    value sitting at a name whose row was never created -- unreachable, uncounted, and
    still costing money at the vault.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        vault = await _a_vault(client)
        for index in range(MAX_CREDENTIALS_PER_VAULT):
            await _a_credential(client, vault["id"], name=f"token-{index}")
        held = len(harness.writer.entries)

        await client.post(
            f"/v1/vaults/{vault['id']}/credentials", json=_submission(name="one-more")
        )

    assert len(harness.writer.entries) == held


async def test_the_ceiling_is_per_vault_and_not_per_tenant() -> None:
    """A full vault does not stop the tenant's next vault from taking a credential.

    Asserted because the cheapest wrong implementation counts what the tenant holds
    rather than what this vault holds, and every other case here passes under it.
    """
    harness = a_harness()
    async with caller(harness.app, harness.tenant) as client:
        full = await _a_vault(client, name="full")
        for index in range(MAX_CREDENTIALS_PER_VAULT):
            await _a_credential(client, full["id"], name=f"token-{index}")
        roomy = await _a_vault(client, name="roomy")

        accepted = await client.post(
            f"/v1/vaults/{roomy['id']}/credentials", json=_submission(name="first")
        )

    assert accepted.status_code == 201, accepted.text
