"""Reading one credential out of a Secrets Manager client, and the four ways it fails.

Tier 1, no AWS. The adapter is handed a client factory rather than building one, so the
whole of it is exercisable against a fake carrying the single method the port needs --
which is the point of that shape, and asserting it here is what keeps the shape.

Two of the assertions below are about what the module does *not* do. It names no
third-party package, so the port stays importable in a process that has no AWS SDK
installed; and no failure path it takes puts a credential into an exception message,
because that message reaches a service log.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import pytest

from managed_agent.adapters.secrets.vault import SecretsClient, SecretsManagerVault
from managed_agent.core.ports import CredentialVault

_MODULE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "managed_agent"
    / "adapters"
    / "secrets"
    / "vault.py"
)

_ENTRY = "map/tool-credential/tenant/ref"
_VALUE = "s3cr3t-vault-value"


class _ServiceError(Exception):
    """A stand-in for a botocore client error, carrying the shape one carries.

    The real class is not imported for the same reason the adapter does not import it:
    the client is built elsewhere, so its exception classes are not this layer's to
    name. What the adapter reads is `response["Error"]["Code"]`, and that is all this
    has to carry for the read to be the real read.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": "irrelevant"}}


class _Answers:
    """A client answering one canned response, or raising one canned exception."""

    def __init__(
        self,
        response: Mapping[str, object] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._response = response
        self._raises = raises
        self.asked: list[str] = []

    async def get_secret_value(self, *, SecretId: str) -> Mapping[str, object]:
        self.asked.append(SecretId)
        if self._raises is not None:
            raise self._raises
        assert self._response is not None
        return self._response


def _vault_over(client: _Answers) -> SecretsManagerVault:
    @asynccontextmanager
    async def factory() -> AsyncIterator[SecretsClient]:
        yield client

    def enter() -> AbstractAsyncContextManager[SecretsClient]:
        return factory()

    return SecretsManagerVault(enter)


async def test_a_string_entry_comes_back_as_its_value() -> None:
    client = _Answers({"SecretString": _VALUE})

    assert await _vault_over(client).fetch(_ENTRY) == _VALUE
    assert client.asked == [_ENTRY]


@pytest.mark.parametrize(
    "code", ["ResourceNotFoundException", "InvalidRequestException"]
)
async def test_a_missing_entry_is_a_key_error_naming_it(code: str) -> None:
    """A secret scheduled for deletion answers with the second code, and to a caller
    that is the same fact as the first: there is no value to attach."""
    vault = _vault_over(_Answers(raises=_ServiceError(code)))

    with pytest.raises(KeyError) as raised:
        await vault.fetch(_ENTRY)

    assert raised.value.args == (_ENTRY,)


async def test_another_service_error_is_not_translated_into_a_missing_entry() -> None:
    """A vault that would not answer is a different fact from one with no such entry,
    and collapsing the two would have a caller retry a registration that is fine."""
    original = _ServiceError("ThrottlingException")
    vault = _vault_over(_Answers(raises=original))

    with pytest.raises(_ServiceError) as raised:
        await vault.fetch(_ENTRY)

    assert raised.value is original


async def test_an_error_carrying_no_service_code_is_re_raised_as_itself() -> None:
    original = ValueError("something else entirely")
    vault = _vault_over(_Answers(raises=original))

    with pytest.raises(ValueError) as raised:
        await vault.fetch(_ENTRY)

    assert raised.value is original


async def test_a_binary_entry_is_refused_rather_than_decoded() -> None:
    """Every credential this platform attaches goes into an environment variable or an
    HTTP header, both text, so a binary entry is a registration mistake and picking an
    encoding for it would hide that mistake behind a credential that does not work."""
    vault = _vault_over(_Answers({"SecretBinary": b"\xff\xfe"}))

    with pytest.raises(ValueError, match=_ENTRY):
        await vault.fetch(_ENTRY)


async def test_no_failure_path_puts_the_value_into_its_message() -> None:
    """The value is in the response on the success path only; every raise below happens
    with it either absent or unread, and this is the assertion that keeps it so."""
    failures: list[BaseException] = []
    for client in (
        _Answers(raises=_ServiceError("ResourceNotFoundException")),
        _Answers(raises=_ServiceError("ThrottlingException")),
        _Answers({"SecretBinary": _VALUE.encode()}),
        _Answers({"SecretString": 17}),
    ):
        with pytest.raises(Exception) as raised:  # noqa: B017, PT011
            await _vault_over(client).fetch(_ENTRY)
        failures.append(raised.value)

    assert len(failures) == 4
    for failure in failures:
        assert _VALUE not in str(failure)
        assert _VALUE not in repr(failure)


def _unreachable() -> AbstractAsyncContextManager[SecretsClient]:
    raise AssertionError("the port check never enters the client")


def test_the_adapter_satisfies_the_port_it_is_written_behind() -> None:
    """Checked by assignment rather than by `isinstance`, so `mypy --strict` grades the
    method's whole signature and not merely that a name of that spelling exists."""
    port: CredentialVault = SecretsManagerVault(_unreachable)

    assert isinstance(port, SecretsManagerVault)


def test_the_module_imports_nothing_outside_the_standard_library() -> None:
    """The AWS SDK is not imported here, which is what lets this port be imported in a
    process that has none installed and driven in a test that mocks nothing."""
    roots = set()
    for node in ast.walk(ast.parse(_MODULE.read_text())):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    assert roots, f"no imports found in {_MODULE}; this walk reads nothing"
    assert roots <= sys.stdlib_module_names, sorted(roots - sys.stdlib_module_names)
