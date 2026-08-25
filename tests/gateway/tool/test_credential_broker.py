"""Composing a vault key under the calling tenant, and holding what it read.

Tier 1, no infrastructure. The broker is the real class throughout and only the vault is
a fake, because the vault is the genuine boundary -- the thing that would otherwise be
an AWS call -- while the composition and the hold are the behaviour under test.

The cross-tenant test is the reason this file exists. Two tenants registering the same
`credential_ref` must read two different entries, and the vault below answers a
different value per name so that a leak shows up as the wrong value rather than as an
assertion nobody wrote.
"""

from __future__ import annotations

import asyncio
import re
from uuid import UUID

import pytest
from pydantic import ValidationError

from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import TenantId
from managed_agent.core.registration.scope_binding import (
    StdioServer,
    StreamableHttpServer,
)
from managed_agent.gateway.tool import error_map
from managed_agent.gateway.tool.credential_broker import (
    HOLD_S,
    VAULT_FETCH_TIMEOUT_S,
    VAULT_PREFIX,
    CredentialUnavailable,
    ToolCredentialBroker,
    vault_name,
)
from managed_agent.gateway.tool.mcp_proxy import GATEWAY_STARTUP_TIMEOUT_S

_ONE = TenantId(UUID("11111111-1111-4111-8111-111111111111"))
_OTHER = TenantId(UUID("22222222-2222-4222-8222-222222222222"))
_REF = "vendor/prod-token"
_VALUE = "s3cr3t-broker-value"


class _PerNameVault:
    """A vault answering a value derived from the name it was asked for.

    Derived rather than constant, so a fetch under the wrong tenant's key hands back
    the wrong tenant's value and the assertion fails on the value rather than on a
    count nobody checked.
    """

    def __init__(self) -> None:
        self.fetches: list[str] = []

    async def fetch(self, name: str) -> str:
        self.fetches.append(name)
        return f"value-for::{name}"


class _OneValueVault:
    def __init__(self, value: str = _VALUE) -> None:
        self.value = value
        self.fetches: list[str] = []

    async def fetch(self, name: str) -> str:
        self.fetches.append(name)
        return self.value


class _RotatingVault:
    """A vault whose answer changes between reads, so a stale hold is visible."""

    def __init__(self) -> None:
        self.fetches: list[str] = []

    async def fetch(self, name: str) -> str:
        self.fetches.append(name)
        return f"value-{len(self.fetches)}"


class _RaisingVault:
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.fetches: list[str] = []

    async def fetch(self, name: str) -> str:
        self.fetches.append(name)
        raise self._error


class _AnswersThenFails:
    """A vault that answers once and then stops, so a stale hold has something to be
    served instead of the failure -- which is the mistake being tested for."""

    def __init__(self) -> None:
        self.fetches: list[str] = []

    async def fetch(self, name: str) -> str:
        self.fetches.append(name)
        if len(self.fetches) > 1:
            raise RuntimeError("the vault is down")
        return _VALUE


class _SlowVault:
    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.fetches: list[str] = []

    async def fetch(self, name: str) -> str:
        self.fetches.append(name)
        await asyncio.sleep(self._delay)
        return _VALUE


def _stdio(ref: str = _REF) -> StdioServer:
    return StdioServer(
        transport="stdio",
        command="/bin/true",
        credential_ref=ref,
        credential_env_var="MAP_TOKEN",
    )


def _http(ref: str = _REF) -> StreamableHttpServer:
    return StreamableHttpServer(
        transport="streamable_http",
        url="https://tool.example.invalid/mcp",
        credential_ref=ref,
        credential_header="X-Api-Key",
    )


def test_the_composed_name_is_the_prefix_the_tenant_and_the_ref() -> None:
    """Pinned as a literal here, which is the one place this format is written down --
    so a `vault_name` that returned the bare ref fails on this line."""
    assert vault_name(_ONE, _REF) == (
        f"map/tool-credential/11111111-1111-4111-8111-111111111111/{_REF}"
    )
    assert vault_name(_ONE, _REF).startswith(f"{VAULT_PREFIX}/")


async def test_a_stdio_registration_gets_the_value_under_the_variable_it_reads() -> (
    None
):
    vault = _OneValueVault()

    attached = await ToolCredentialBroker(vault).for_stdio(_ONE, _stdio())

    assert attached.into_env({}) == {"MAP_TOKEN": _VALUE}
    assert vault.fetches == [vault_name(_ONE, _REF)]


async def test_an_http_registration_gets_the_value_under_the_header_it_reads() -> None:
    vault = _OneValueVault()

    attached = await ToolCredentialBroker(vault).for_http(_ONE, _http())

    assert attached.into_headers({}) == {"X-Api-Key": _VALUE}
    assert vault.fetches == [vault_name(_ONE, _REF)]


async def test_two_tenants_naming_one_ref_read_two_entries() -> None:
    """The escalation this whole module exists to make inexpressible. Before the
    composition existed, `credential_ref` was text one tenant wrote and the fetch went
    to whatever name it said -- so two tenants naming one ref read one entry."""
    vault = _PerNameVault()
    broker = ToolCredentialBroker(vault)

    mine = await broker.for_http(_ONE, _http())
    theirs = await broker.for_http(_OTHER, _http())

    assert vault.fetches == [vault_name(_ONE, _REF), vault_name(_OTHER, _REF)]
    assert vault.fetches[0] != vault.fetches[1]
    assert mine.into_headers({})["X-Api-Key"] == f"value-for::{vault_name(_ONE, _REF)}"
    assert theirs.into_headers({})["X-Api-Key"] == (
        f"value-for::{vault_name(_OTHER, _REF)}"
    )


@pytest.mark.parametrize(
    "ref",
    [
        "../other-tenant/token",
        "x/../../map/tool-credential/OTHER/token",
        "a..b",
        "/leading-slash",
        "",
        "has space",
        "x" * 129,
    ],
)
async def test_a_ref_that_could_climb_out_of_the_prefix_is_refused_unread(
    ref: str,
) -> None:
    """`a..b` is admitted by the character pattern alone, which is why `..` is refused
    as a substring rather than as a leading path segment: what this promises is that
    the composed name cannot leave the tenant's prefix, and that has to survive the
    store being swapped for a path-like one."""
    vault = _OneValueVault()

    with pytest.raises(CredentialUnavailable) as raised:
        await ToolCredentialBroker(vault).for_http(
            _ONE, _http().model_copy(update={"credential_ref": ref})
        )

    assert raised.value.reason == "malformed_ref"
    assert vault.fetches == []


async def test_a_malformed_ref_is_refused_by_vault_name_itself() -> None:
    with pytest.raises(CredentialUnavailable) as raised:
        vault_name(_ONE, "../other/token")

    assert raised.value.credential_ref == "../other/token"
    assert raised.value.reason == "malformed_ref"


async def test_an_unreadable_entry_becomes_one_failure_type_chained_to_its_cause() -> (
    None
):
    missing = KeyError(vault_name(_ONE, _REF))
    vault = _RaisingVault(missing)

    with pytest.raises(CredentialUnavailable) as raised:
        await ToolCredentialBroker(vault).for_http(_ONE, _http())

    assert raised.value.reason == "unreadable"
    assert raised.value.credential_ref == _REF
    assert raised.value.__cause__ is missing


async def test_the_message_carries_neither_the_value_nor_the_composed_name() -> None:
    """The ref is text the tenant wrote, so echoing it discloses nothing; the composed
    key carries the tenant's own id and this message reaches a service log."""
    vault = _RaisingVault(RuntimeError(f"the vault said {_VALUE}"))

    with pytest.raises(CredentialUnavailable) as raised:
        await ToolCredentialBroker(vault).for_http(_ONE, _http())

    message = str(raised.value)
    assert _VALUE not in message
    assert vault_name(_ONE, _REF) not in message
    assert _REF in message


async def test_a_vault_that_will_not_answer_in_time_fails_rather_than_hangs() -> None:
    vault = _SlowVault(VAULT_FETCH_TIMEOUT_S * 4)
    broker = ToolCredentialBroker(vault)

    with pytest.raises(CredentialUnavailable) as raised:
        async with asyncio.timeout(VAULT_FETCH_TIMEOUT_S * 3):
            await broker.for_http(_ONE, _http())

    assert raised.value.reason == "unreadable"


def test_the_fetch_deadline_is_strictly_inside_the_gateway_s_startup_deadline() -> None:
    """Imported from both modules rather than restated, so the relation is asserted.
    A bound outside the Gateway's own connection-startup deadline would never be the
    bound that fires, and the constant is a literal there only because importing it
    the other way would be a cycle."""
    assert 0.0 < VAULT_FETCH_TIMEOUT_S < GATEWAY_STARTUP_TIMEOUT_S


async def test_a_held_credential_is_read_once_for_however_many_asks() -> None:
    vault = _OneValueVault()
    broker = ToolCredentialBroker(vault)

    await broker.for_http(_ONE, _http())
    await broker.for_stdio(_ONE, _stdio())

    assert vault.fetches == [vault_name(_ONE, _REF)]
    assert HOLD_S > 0.0


async def test_a_credential_past_its_window_is_read_again_and_the_new_value_wins() -> (
    None
):
    """The window exists for one reason: a rotated or revoked credential has to stop
    being attached without a deploy."""
    vault = _RotatingVault()
    broker = ToolCredentialBroker(vault, hold_s=0.0)

    first = await broker.for_http(_ONE, _http())
    second = await broker.for_http(_ONE, _http())

    assert vault.fetches == [vault_name(_ONE, _REF)] * 2
    assert first.into_headers({})["X-Api-Key"] == "value-1"
    assert second.into_headers({})["X-Api-Key"] == "value-2"


async def test_an_expired_entry_is_replaced_rather_than_left_in_the_process() -> None:
    """`is not`, not `not in`: `_secret` deletes the stale entry and immediately stores
    the fresh read, so the key is present and only its value has moved. What must not
    survive is the expired Secret object -- a revoked credential still resident."""
    vault = _RotatingVault()
    broker = ToolCredentialBroker(vault, hold_s=0.0)
    name = vault_name(_ONE, _REF)

    await broker.for_http(_ONE, _http())
    stale = broker._held[name][1]
    await broker.for_http(_ONE, _http())

    assert broker._held[name][1] is not stale
    assert broker._held[name][1].reveal() == "value-2"


async def test_an_expired_entry_goes_when_anything_is_asked_for_not_only_itself() -> (
    None
):
    """The eviction covers every expired entry and not just the one being asked for.

    Only the re-asked path was exercised before, and lazy eviction passes that test
    while leaving a rotated credential nobody asks for again resident for the process
    lifetime -- and `_held` growing without bound alongside it. One tenant's ask is
    enough to clear another's expired entry, which is what bounds the dict by what is
    in use rather than by everything ever read.
    """
    vault = _PerNameVault()
    broker = ToolCredentialBroker(vault, hold_s=0.0)
    abandoned = vault_name(_ONE, _REF)

    await broker.for_http(_ONE, _http())
    assert abandoned in broker._held, "nothing was held, so nothing below is checked"
    await broker.for_http(_OTHER, _http())

    assert abandoned not in broker._held, (
        "an expired entry nobody asked for again is still resident; eviction runs "
        "only for the name being read, so a revoked credential stays in this process"
    )


async def test_the_window_bounds_attachment_and_not_residency() -> None:
    """The limitation the docstring must keep stating. Nothing runs at the instant a
    window closes, so a broker asked nothing more still holds what it read -- and a
    test that did not pin this would let the claim drift back to "deleted when the
    window closes", which is what it used to say and was not true."""
    vault = _OneValueVault()
    broker = ToolCredentialBroker(vault, hold_s=0.0)

    await broker.for_http(_ONE, _http())

    assert vault_name(_ONE, _REF) in broker._held, (
        "an entry left after its window closed with no further ask; if a sweeper was "
        "added, HOLD_S's docstring owes the stronger claim and this test owes deletion"
    )


async def test_an_expired_entry_is_not_served_when_the_re_read_fails() -> None:
    """Falling back to the expired value would be the whole window undone: a revoked
    credential would keep being attached for as long as the vault stayed unhappy."""
    vault = _AnswersThenFails()
    broker = ToolCredentialBroker(vault, hold_s=0.0)
    first = await broker.for_http(_ONE, _http())
    assert first.into_headers({})["X-Api-Key"] == _VALUE

    with pytest.raises(CredentialUnavailable) as raised:
        await broker.for_http(_ONE, _http())

    assert raised.value.reason == "unreadable"
    assert len(vault.fetches) == 2


@pytest.mark.parametrize("reason", ["malformed_ref", "unreadable"])
def test_an_unreadable_credential_is_the_registration_s_problem(reason: str) -> None:
    """Without an arm for this class it classifies as `platform.internal` -- which
    reports a tenant's own broken registration as a fault of the platform."""
    failure = CredentialUnavailable(_REF, reason)  # type: ignore[arg-type]

    assert error_map.classify(failure) is ErrorCode.TOOL_UNAVAILABLE
    assert error_map.classify(failure).value == "tool.unavailable"


def test_the_module_is_the_only_place_the_ref_pattern_is_written_down() -> None:
    """A sanity check on the pattern itself, so the refusals above are refusals of the
    shapes intended rather than of everything."""
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", _REF)
    assert vault_name(_ONE, "a") == f"{VAULT_PREFIX}/{_ONE}/a"
    assert vault_name(_ONE, "x" * 128).endswith("x" * 128)


async def test_a_public_server_gets_no_header_and_the_vault_is_never_asked() -> None:
    """A registration naming no credential is a public server, not a broken one.

    The vault assertion is the one that matters. A branch that returned an empty
    attachment *after* reading the vault would pass the header assertion and still fail
    every public server whose tenant happens to hold no vault entry -- which is every
    tenant with a public server, since there is nothing for them to store.
    """
    vault = _RaisingVault(AssertionError("a public server must not read the vault"))
    public = StreamableHttpServer(
        transport="streamable_http", url="https://mcp.example.invalid/mcp"
    )

    attached = await ToolCredentialBroker(vault).for_http(_ONE, public)

    assert attached.into_headers({"accept": "text/event-stream"}) == {
        "accept": "text/event-stream"
    }
    assert vault.fetches == []


def test_naming_a_header_without_a_credential_is_refused_at_registration() -> None:
    """The one combination that cannot be honoured, refused where it is written.

    A header named for a credential that does not exist would produce an outbound call
    carrying no header while the registration says which header it carries. Left to the
    Tool Gateway it is a 401 from the far end, and the registration that caused it looks
    correct.
    """
    with pytest.raises(ValidationError) as raised:
        StreamableHttpServer(
            transport="streamable_http",
            url="https://mcp.example.invalid/mcp",
            credential_header="X-Api-Key",
        )

    assert "names no credential_ref" in str(raised.value)


def test_the_default_header_alone_does_not_make_a_registration_illegal() -> None:
    """`credential_header` has a real default, so absence must be read from the set.

    A validator keyed on the header's *value* rather than on whether it was set would
    refuse every public server, because the default `Authorization` is indistinguishable
    from a caller who typed it.
    """
    assert (
        StreamableHttpServer(
            transport="streamable_http", url="https://mcp.example.invalid/mcp"
        ).credential_ref
        is None
    )
