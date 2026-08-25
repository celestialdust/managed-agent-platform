"""Reading one credential out of AWS Secrets Manager, and putting one in.

Two classes, one per direction, because two processes hold them: the Tool Gateway
reads and the control plane writes, and neither has any use for the other half.
Splitting them is what lets a control plane be handed something with no `fetch` on
it at all. They share this module because they share the one reason to change --
"where credentials are kept" -- and would otherwise be two files that must move
together.

Neither holds anything, caches anything, or decides when a value stops being usable
-- whoever asked owns those questions.

A missing entry raises KeyError. The banned-api rule in `pyproject.toml` bans
`managed_agent.adapters` everywhere except `adapters/` itself, `composition.py` and
`tests/`, so no consumer of this port -- nothing under `gateway/tool/`, `control/` or
`gateway/model/` -- can import this module to name an exception class it raises. A
caller that must tell "no such entry" from "the vault would not answer" therefore needs
a type it can name without importing this file, and a keyed store raising a stdlib
lookup error is that type.

No deadline is set here, because the object that carries one on this path is the
client and this module does not build it. Two bounds are owed elsewhere and both
are named so neither is discovered missing: the factory passed in must carry a
`botocore.config.Config` with connect and read timeouts and a retry cap (its
defaults are 60s, 60s and botocore's own retry resolution), and the bound a caller
can enforce without the client is an `asyncio.timeout` around `fetch`, which
`gateway/tool/credential_broker.py` sets.

A value read here appears in nothing but the return: no log line, no exception
message, no repr. An exception raised here names the entry and never its contents.
"""

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Final, Protocol

_MISSING_ENTRY_CODES: Final[frozenset[str]] = frozenset(
    {
        "ResourceNotFoundException",
        # A secret scheduled for deletion answers with this instead. To a caller
        # that is the same fact -- there is no value to attach -- so it is not a
        # separate outcome.
        "InvalidRequestException",
    }
)


class SecretsClient(Protocol):
    """The one call this adapter makes on a Secrets Manager client.

    Declared here rather than imported so nothing in this module depends on an AWS
    package: the client is constructed at the composition root with every other
    concrete thing, and this module can be driven by a fake carrying this single
    method.
    """

    async def get_secret_value(self, *, SecretId: str) -> Mapping[str, object]: ...


class SecretsWriteClient(Protocol):
    """The three calls the writing half makes.

    A second Protocol rather than three more methods on `SecretsClient`, so a fake
    standing in for the reader is not obliged to implement writes it will never be
    asked for -- and, more to the point, so the reading half cannot acquire a write
    by being handed a client that happens to have one.
    """

    async def create_secret(
        self, *, Name: str, SecretString: str
    ) -> Mapping[str, object]: ...

    async def put_secret_value(
        self, *, SecretId: str, SecretString: str
    ) -> Mapping[str, object]: ...

    async def delete_secret(
        self, *, SecretId: str, RecoveryWindowInDays: int
    ) -> Mapping[str, object]: ...


def _service_error_code(exc: BaseException) -> str:
    """The AWS service error code an exception carries, or "" when it carries none.

    Read off the exception rather than matched by type, because this module is
    handed a client and never builds one, so that client's exception classes are
    not ours to import. The shape read here is the `response["Error"]["Code"]`
    every botocore client error carries.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return ""
    error = response.get("Error")
    return error.get("Code", "") if isinstance(error, dict) else ""


class SecretsManagerVault:
    """The credential-vault port over AWS Secrets Manager.

    A client is entered per fetch. Whoever asked holds what it reads for minutes,
    so this runs a handful of times per entry per process, and a long-lived client
    would hand someone a connection to close in exchange for saving a setup that
    nothing waits on.
    """

    def __init__(
        self,
        client_factory: Callable[[], AbstractAsyncContextManager[SecretsClient]],
    ) -> None:
        self._client_factory = client_factory

    async def fetch(self, name: str) -> str:
        """The current value of one vault entry.

        Raises KeyError when no entry of that name exists, and ValueError when the
        entry holds bytes: every credential this platform attaches goes into an
        environment variable or an HTTP header, both of which are text, so a binary
        entry is a registration mistake and picking an encoding for it would hide
        that mistake behind a credential that does not work.
        """
        async with self._client_factory() as client:
            try:
                response = await client.get_secret_value(SecretId=name)
            except Exception as exc:
                if _service_error_code(exc) in _MISSING_ENTRY_CODES:
                    raise KeyError(name) from exc
                raise
        value = response.get("SecretString")
        if not isinstance(value, str):
            raise ValueError(f"vault entry {name} holds no string value")
        return value


_ALREADY_EXISTS: Final[str] = "ResourceExistsException"

RECOVERY_WINDOW_DAYS: Final[int] = 7
"""How long an erased entry stays recoverable.

Seven rather than zero. A tenant deleting the wrong credential is a mistake somebody
can undo within a week; a tenant deleting the right one gets what they asked for
either way, because Secrets Manager refuses to return the value of an entry pending
deletion. So the window costs nothing an attacker can use and buys back the one
irreversible operation on this surface.

Not thirty, which is the service default: a rotated-and-erased credential sitting
recoverable for a month is a month in which a compromised value is still restorable
by anyone who can call the API.
"""


class SecretsManagerVaultWriter:
    """The credential-vault writing port over AWS Secrets Manager.

    Holds no reading capability, which is the whole reason it is a separate class.

    A client is entered per call, matching the reader above and for the same reason:
    writes happen when a tenant registers or rotates a credential, which is rare
    enough that a long-lived client would trade a connection to close for a setup
    nothing waits on.

    A value handled here appears in no log line, no exception message and no repr. The
    exceptions raised name the entry and never its contents.
    """

    def __init__(
        self,
        client_factory: Callable[[], AbstractAsyncContextManager[SecretsWriteClient]],
    ) -> None:
        self._client_factory = client_factory

    async def put(self, name: str, value: str) -> None:
        """Write this value at this name, whether or not anything is there.

        Create-then-put rather than describe-then-branch, because the check and the
        write would be two calls with a gap between them: two tenants registering the
        same name concurrently would both see "absent" and both create, and one of
        them gets `ResourceExistsException` from `create_secret` anyway. Handling that
        exception *is* the concurrency-safe branch, so asking first buys a round trip
        and a race.

        This makes rotation and creation one operation. That is deliberate: a rotate
        that could only target an existing entry, and a create that could only target
        a missing one, would make the caller's correctness depend on knowing which
        state the vault was in -- and the caller is a route handler whose own database
        row is the thing that says whether this credential is new. Whichever it says,
        the vault ends up holding this value at this name.
        """
        async with self._client_factory() as client:
            try:
                await client.create_secret(Name=name, SecretString=value)
            except Exception as exc:
                if _service_error_code(exc) != _ALREADY_EXISTS:
                    raise
                await client.put_secret_value(SecretId=name, SecretString=value)

    async def erase(self, name: str) -> None:
        """Stop this entry's value being readable, recoverably.

        A missing entry is not an error. The caller is deleting a credential it has a
        row for, and a row without a vault entry is a state this platform can reach
        honestly -- a write that failed after the row was committed, or an entry erased
        twice by a retried request. Raising here would turn a retry into a failure and
        leave the row undeletable, which is the worse of the two outcomes by a wide
        margin.
        """
        async with self._client_factory() as client:
            try:
                await client.delete_secret(
                    SecretId=name, RecoveryWindowInDays=RECOVERY_WINDOW_DAYS
                )
            except Exception as exc:
                if _service_error_code(exc) not in _MISSING_ENTRY_CODES:
                    raise
