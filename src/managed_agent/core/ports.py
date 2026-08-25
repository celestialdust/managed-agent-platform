"""The abstract ports every adapter satisfies. Nothing here imports infrastructure.

Appending and range-reading are separate ports over the same store because they scale in
opposite directions: the append path serializes per Session to keep the sequence
contiguous, while a range read has no ordering obligation at all and can be served by a
replica. One port would force a replica-backed reader to carry the writer's lock.

Each port is `runtime_checkable` so a caller can assert conformance where the binding
happens rather than discovering a missing method at the first call. That check is
shallow by design — it sees names, not signatures — so it catches the adapter that never
grew a method and not the one that grew it wrong; the adapter's own tests cover the
second.
"""

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from managed_agent.core.ids import (
    CredentialId,
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    VaultId,
)
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.scope_binding import (
    RegisteredTool,
    ServerRegistration,
)
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vault_catalogue import Credential, Vault


@runtime_checkable
class EventRecord(Protocol):
    """What an appended event looks like once the store has numbered it.

    The members are read-only rather than settable, which is what an append-only log
    means expressed as a type: nothing that holds one of these may rewrite the sequence
    or the payload of an event already written. Declaring them as plain annotations
    would demand a settable attribute and so exclude every immutable implementation —
    the frozen record the Postgres reader returns among them.
    """

    @property
    def session_id(self) -> SessionId: ...

    @property
    def seq(self) -> Seq: ...

    @property
    def type(self) -> str: ...

    @property
    def payload(self) -> dict[str, object]: ...


@runtime_checkable
class EventLogAppend(Protocol):
    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        """Append one event and return the sequence it was given.

        Raises SequenceRace when another writer took the next sequence first. The caller
        retries; it must never choose a sequence itself, because a chosen sequence is
        how a gap gets written.
        """
        ...


@runtime_checkable
class EventLogRange(Protocol):
    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]:
        """Events of one Session with `start <= seq <= end`, in sequence order.

        Both ends are inclusive, and **at most `limit` records come back**. A short
        result therefore means "page for the rest", not "the range is empty" -- a caller
        that needs the whole span reads again from just past the highest sequence it saw
        and stops only on an empty page.

        The cap and the ordering are stated here rather than left to each adapter
        because a caller cannot page correctly without both, and this signature is all a
        caller sees. Omitting them cost one silent wrong answer already: a state fold
        stopped at the adapter's undeclared page cap and reported a stale state, with
        every test passing, because nothing in the contract said a full result was not
        the whole range.

        An empty result is not an error. A range past the head of the log, or over a
        Session that has written nothing, is legitimately empty -- that is how a caller
        paging forward learns it has reached the end.
        """
        ...

    def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[EventRecord]:
        """Yield events as they are appended, starting strictly after `after`."""
        ...

    async def retained_floor(self, session_id: SessionId) -> Seq:
        """The lowest sequence still retained. A read below this is a distinct refusal
        rather than an empty range, so a caller can tell expiry from emptiness."""
        ...


@runtime_checkable
class ObjectStore(Protocol):
    async def put(self, key: str, body: bytes) -> str: ...
    async def get(self, key: str) -> bytes: ...
    async def delete_prefix(self, prefix: str) -> int: ...


@runtime_checkable
class CredentialVault(Protocol):
    async def fetch(self, name: str) -> str: ...


@runtime_checkable
class CredentialVaultWriter(Protocol):
    """Putting a credential into the vault, and taking one out. Never reading one.

    Deliberately not `CredentialVault` with two more methods. The component that
    accepts a tenant's credential and the component that attaches it to an outbound
    call are different processes with different IAM roles, and the accepting one has
    no business reading a value back -- it already had the value once, at the moment
    the tenant sent it, and every later read would be a capability it does not need.

    Split as a type rather than only in IAM so the restriction survives a policy edit:
    a control plane holding one of these has no `fetch` to call, so code that would
    read a tenant's tool credential does not compile rather than failing at AWS. The
    two halves are enforced in two places that fail differently, which is the point.

    `erase` rather than `delete`, because Secrets Manager's own deletion is scheduled
    rather than immediate and a caller that believed otherwise would report a
    credential gone while it was still readable for the recovery window. What the name
    promises is that the value stops being attachable, which is what the adapter
    delivers.
    """

    async def put(self, name: str, value: str) -> None: ...
    async def erase(self, name: str) -> None: ...


@runtime_checkable
class VaultCatalogue(Protocol):
    """Where a tenant's vaults and credential *names* are kept. No value passes here.

    Both families on one port rather than two, because a credential is addressed
    through its vault and every read of one is scoped by the tenant that owns the
    other. Two ports would mean two places holding the same tenant predicate, and the
    predicate is what stops a tenant reading another's row.

    Every method takes the tenant, and takes it as a query term rather than as
    something to check against what came back: a row belonging to somebody else is
    absent from the result instead of fetched and then dropped, so there is no moment
    at which this process holds a row it is not entitled to.

    `archive_*` and `delete_*` return whether they matched, not the row. A caller that
    got `False` cannot tell "no such id" from "not yours", and that is the intended
    answer -- distinguishing them would let a tenant probe for the existence of
    another tenant's ids.
    """

    async def insert_vault(self, vault: Vault, /) -> None: ...
    async def fetch_vault(
        self, vault_id: VaultId, tenant_id: TenantId, /
    ) -> Vault | None: ...
    async def page_vaults(
        self,
        tenant_id: TenantId,
        after: tuple[datetime, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Vault]: ...
    async def archive_vault(
        self, vault_id: VaultId, tenant_id: TenantId, /
    ) -> bool: ...
    async def delete_vault(self, vault_id: VaultId, tenant_id: TenantId, /) -> bool: ...

    async def insert_credential(self, credential: Credential, /) -> None: ...
    async def fetch_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> Credential | None: ...
    async def page_credentials(
        self,
        vault_id: VaultId,
        tenant_id: TenantId,
        after: tuple[datetime, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Credential]: ...
    async def archive_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> bool: ...
    async def delete_credential(
        self, credential_id: CredentialId, tenant_id: TenantId, /
    ) -> bool: ...
    async def mark_value_written(
        self, credential_id: CredentialId, tenant_id: TenantId, at: datetime, /
    ) -> bool: ...


@runtime_checkable
class Clock(Protocol):
    def now_epoch_ms(self) -> int: ...


@runtime_checkable
class Resolution(Protocol):
    """A definition body paired with the revision number it was read from."""

    @property
    def definition(self) -> AgentDefinition: ...

    @property
    def revision(self) -> int: ...


@runtime_checkable
class DefinitionRegistry(Protocol):
    """Where agent definitions are stored, and where a Session's pin is resolved.

    `register` returns the revision number it wrote, and `resolve` returns a body
    together with the revision it came from. The number is returned rather than derived
    by the caller because a caller counting its own registrations is right only until
    something else registers, and that number is what a Session pins for its whole life.
    """

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int: ...

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        """The latest revision visible to that tenant.

        Raises `UnknownDefinition` when there is none. Refused rather than answered
        with a default-shaped definition, which would start a Session with no
        instructions and show the tenant an agent that does nothing instead of a
        refusal naming what is missing.
        """
        ...

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        """Every revision of one agent this tenant owns, ascending, with its state.

        Empty means the agent does not exist *for this caller*, and that is one answer
        for an id nobody registered and an id belonging to somebody else -- the same
        reason `UnknownDefinition` covers both.

        One call rather than "which revisions are there" plus "which of them are
        retired", so a caller cannot read the two at different instants and act on a
        pair that was never simultaneously true.
        """
        ...

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        """One exact revision's body, or None when this tenant has no such revision.

        Addressed by number rather than "latest", which is what lets a Session created
        months ago keep reading the bytes it resolved to.
        """
        ...

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        """Retire one revision. True when this call retired it, False when it already
        was -- and False, writing nothing, when the revision is not this tenant's.

        Idempotent rather than refusing a repeat: a caller retrying a retirement wants
        it retired, and a refusal there forces every caller into a read-then-write race
        to avoid an error that describes the state it asked for.
        """
        ...


@runtime_checkable
class ToolRegistry(Protocol):
    """Where MCP server registrations are stored and where a tool name is resolved.

    Every seam this process depends on is declared in this one module, so a reader
    answering "what can the platform talk to" reads one file rather than hunting the
    domain modules for scattered Protocols. `DefinitionRegistry` above is the same shape
    -- it exchanges `AgentDefinition` and still lives here, importing it -- and one seam
    declared somewhere else would make the collection stop being the answer.

    `register` writes a whole registration or none of it: a half-written one advertises
    tools whose server nobody can reach. It raises `NameAlreadyRegistered` rather than
    disambiguating a taken name, because a Grant already written against that name would
    then silently resolve to a different tool.
    """

    async def register(
        self, tenant_id: TenantId, registration: ServerRegistration
    ) -> None: ...

    async def lookup(self, tenant_id: TenantId, tool_name: str) -> RegisteredTool:
        """The tool of that name, with the server behind it. Raises `UnknownTool`."""
        ...

    async def list_for_tenant(
        self, tenant_id: TenantId
    ) -> Sequence[RegisteredTool]: ...


class SequenceRace(Exception):
    """Another writer took the sequence this append tried to claim."""


class VaultNameTaken(Exception):
    """This tenant already holds a vault of that name.

    Declared here rather than in the adapter that raises it, for the reason
    `UnknownDefinition` is: an exception a port promises to raise is part of that
    port's interface, and the control plane has to catch this one while the layer
    rule forbids it from importing an adapter.

    Raised rather than prevented by a read-before-write, which is what the route
    would otherwise have to do. That check is a check-then-act race: two requests
    naming one name both find it free and both insert, and the unique constraint
    then fires on whichever loses -- so the scan does not remove the case, it only
    removes the route's ability to answer it as a 409. The constraint is the
    decider either way; this type is how its verdict reaches the tenant.

    Carries the name and never the composed vault key. The name is text the tenant
    wrote, so echoing it discloses nothing; the composed key carries the tenant's own
    id and this message reaches a service log.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"a vault named {name!r} is already registered")
        self.name: str = name


class CredentialNameTaken(Exception):
    """This vault already holds a credential of that name.

    Separate from `VaultNameTaken` because what the tenant does next differs -- one
    is a second vault of a name they hold, the other is a second credential inside a
    vault -- and because a single type would force the route to inspect a field to
    decide which refusal to answer with.

    Carries the vault it was raised against, so a tenant writing to several vaults in
    one script can tell which one refused.
    """

    def __init__(self, vault_id: VaultId, name: str) -> None:
        super().__init__(f"a credential named {name!r} already exists in that vault")
        self.vault_id: VaultId = vault_id
        self.name: str = name


class UnknownDefinition(Exception):
    """No definition with that id is visible to that tenant.

    Defined here rather than in the adapter that raises it, because an exception a port
    promises to raise is part of that port's interface -- and because the control plane
    has to catch it, while the layer rule forbids the control plane from importing an
    adapter. The same reasoning put `SequenceRace` here.

    One exception covers both "never registered" and "belongs to another tenant".
    Telling them apart would confirm the existence of another tenant's definition to
    anyone holding its id.
    """


@runtime_checkable
class SessionListing(Protocol):
    """One row of a tenant's Session list, and the key the next page starts from.

    Read-only members for the same reason `EventRecord`'s are: nothing holding one of
    these may rewrite a creation fact, and plain annotations would demand a settable
    attribute and so exclude every frozen implementation.

    There is deliberately no state here. A Session's state is a fold over its own Event
    Log, so carrying it on a list row would mean one fold per row on a read whose job is
    to help a caller *find* a Session; a caller that wants state reads the Session.
    """

    @property
    def id(self) -> SessionId: ...

    @property
    def definition_id(self) -> DefinitionId: ...

    @property
    def definition_revision(self) -> str: ...

    @property
    def created_at_ms(self) -> int: ...


@runtime_checkable
class SessionRegistry(Protocol):
    """Where a Session's creation facts are stored, and how a tenant finds its own.

    Every method takes the tenant, and the tenant is a term in the store's own query
    rather than a check the caller performs afterwards. That is what makes another
    tenant's Session *absent* from a result instead of fetched and then dropped: a
    filter that runs in the store cannot be forgotten at a call site.
    """

    async def create(self, record: SessionRecord) -> None: ...

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        """That tenant's Session of that id. Raises `SessionNotVisible`."""
        ...

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[SessionListing]:
        """One page of that tenant's Sessions, newest first, after the given key.

        `after` is the `(created_at_ms, id)` of the last row the caller already holds,
        as one value rather than two nullable arguments -- half a key is not a position.
        `None` starts at the newest.

        **At most `limit` rows come back and a short page means the walk is over.** A
        `limit` the adapter will not serve is refused rather than quietly reduced: a
        reduced page is indistinguishable from an exhausted one, and a caller reading a
        short page as the end would stop early and never learn it had.
        """
        ...


@runtime_checkable
class SessionsWalkedBackward(SessionRegistry, Protocol):
    """A `SessionRegistry` that can also hand back a page already walked past.

    Separate from `SessionRegistry` rather than a method on it, and separate so that a
    store which cannot do this is not a broken implementation of the port -- it is a
    store without this capability, and the route asks with `isinstance` before offering
    the caller a way to use it. That is the difference between a deployment that pages
    backward and one that does not, expressed as a type rather than as a
    `NotImplementedError` a caller finds at runtime.

    The narrowing also keeps the existing port honest. `SessionRegistry.page` is
    implemented by hand in sixteen test files, and widening it would have made every one
    of them fail to conform for want of a method their test never calls.
    """

    async def page_ending_at(
        self, tenant_id: TenantId, oldest: tuple[int, UUID], limit: int
    ) -> Sequence[SessionListing]:
        """That tenant's page whose OLDEST row is the key given, oldest-first.

        The key is the page's own last row, inclusive, which is exactly what a forward
        cursor holds -- so a caller that walked forward to a page can name the page it
        came from without having kept it. Rows arrive oldest-first, in the order the
        walk visited them, and the caller reverses; a store answering in presentation
        order would decide where the caller's one extra look-ahead row sits.

        **At most `limit` rows come back and a short page means there is nothing further
        back.** A `limit` the adapter will not serve is refused, not reduced, for the
        reason `page` gives.
        """
        ...


class SessionNotVisible(Exception):
    """No Session with that id belongs to that tenant.

    One exception covers both "no such Session" and "not yours". Two distinguishable
    answers would turn a read into an existence oracle: a caller holding an id could
    learn from the shape of the refusal whether it names another tenant's Session.

    Here rather than in the adapter that raises it, because the route that turns it into
    a 404 lives in `control/` and the layer rule forbids `control/` from importing
    `managed_agent.adapters`. An exception a port promises to raise is part of that
    port's interface, which is the same reasoning that put `SequenceRace` and
    `UnknownDefinition` here.
    """
