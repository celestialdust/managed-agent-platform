"""Which revision a named agent version resolves to, and what a whole agent is.

Two altitudes, in one file because the second is what the first is a slice of. The
functions at the top answer "which revision does this reference resolve to", which is
the question a Session creation asks. The types at the bottom answer "what is this
agent", which is the question the agent surface asks -- and its answer is a fold over
the same revisions, so splitting them would put one concept's two halves in two files
that have to agree about what a revision means.

Nothing here performs I/O or names a web framework, and nothing here imports a store —
the round trip is the `DefinitionRegistry` port, so every refusal below is reachable
from an ordinary function call rather than only from an HTTP request. The cases worth
grading are the awkward ones (a pin on a retired revision, an unpinned reference whose
newest revision was withdrawn, an agent with nothing live left), and here they are unit
cases.

There is deliberately no parser. The two shapes a caller can ask for reach the platform
as two fields on `CreateSession` — an id, and a revision number or nothing — where
pydantic performs every refusal at the boundary. Building a value out of them is one
line at one call site, which is fewer moving parts than a parser plus the domain errors
it would have to raise for refusals the boundary already answers 400 for.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from managed_agent.core.ids import DefinitionId, TenantId
from managed_agent.core.ports import DefinitionRegistry
from managed_agent.core.registration.definition import AgentDefinition, VersionFact


@dataclass(frozen=True, slots=True)
class AgentReference:
    """An agent, and either an exact revision of it or no opinion about which.

    `version is None` is "the current one" and is not a default standing in for
    revision 1: the two behave differently the moment a second revision exists, which
    is the only time either matters.
    """

    definition_id: DefinitionId
    version: int | None


class UnknownAgentVersion(Exception):
    """No such revision is registered for that agent under that tenant.

    One exception for "this agent does not exist", "this agent is somebody else's" and
    "this agent has no revision by that number". Telling them apart would let anyone
    holding an id learn from the refusal whether it names another tenant's agent.
    """


class AgentVersionArchived(Exception):
    """That revision exists and was retired. It starts no new Session.

    Carries the revision because the caller's own number and the refused one are not
    always the same: an unpinned reference is refused by the newest revision, which the
    caller never named.
    """

    def __init__(self, revision: int) -> None:
        super().__init__(f"version {revision} is archived")
        self.revision = revision


def choose_revision(facts: Sequence[VersionFact], reference: AgentReference) -> int:
    """The revision this reference resolves to, or the reason it resolves to none.

    An unpinned reference takes the highest *live* revision rather than the highest
    one. Withdrawing a bad edit is the reason to retire a revision at all, and a
    reference that kept pointing at the withdrawn one would make the withdrawal do
    nothing.

    A pinned reference is never redirected. If the pinned revision is retired the
    answer is a refusal and never the next revision down: a Session silently moved onto
    different instructions is precisely what a pin is written to prevent, and it would
    be invisible — the Session would run, and run something else.

    Assumes `facts` holds each revision at most once, which is the store's own
    guarantee: `(id, revision)` is `agent_definition`'s primary key.
    """
    if not facts:
        raise UnknownAgentVersion("this agent has no registered version")
    if reference.version is None:
        live = [fact.revision for fact in facts if not fact.archived]
        if not live:
            raise AgentVersionArchived(max(fact.revision for fact in facts))
        return max(live)
    for fact in facts:
        if fact.revision == reference.version:
            if fact.archived:
                raise AgentVersionArchived(fact.revision)
            return fact.revision
    raise UnknownAgentVersion(f"version {reference.version} is not registered")


@dataclass(frozen=True, slots=True)
class ResolvedVersion:
    """A definition body and the revision number it came from, together.

    Either alone is a half-answer: a body with no number cannot be pinned, and a number
    with no body cannot be run.
    """

    definition: AgentDefinition
    revision: int


async def resolve_reference(
    store: DefinitionRegistry, tenant_id: TenantId, reference: AgentReference
) -> ResolvedVersion:
    """The definition a reference resolves to, and the revision it landed on.

    Raises `UnknownAgentVersion` or `AgentVersionArchived`; both mean no Session may be
    started, and they are distinct because a caller acts on them differently — one is a
    typo or a stale id, the other is a revision that was deliberately withdrawn.

    Two reads rather than one join, because the choice between revisions is a pure
    function of the version facts and is worth testing as one. The second read is a
    primary-key lookup on a call that is already about to write a Session.

    A body that is absent after the facts named its revision is reported as unknown
    rather than defaulted. `agent_definition` has no delete path, so that combination
    means the row went missing, and inventing a revision here would hide it.
    """
    facts = await store.list_versions(reference.definition_id, tenant_id)
    revision = choose_revision(facts, reference)
    body = await store.read_version(reference.definition_id, tenant_id, revision)
    if body is None:
        raise UnknownAgentVersion(f"version {revision} vanished between reads")
    return ResolvedVersion(definition=body, revision=revision)


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """One agent as a whole, rather than one of its revisions.

    An agent is a stack of immutable revisions, and three of the four fields here are
    facts about the stack rather than about any row in it. `version` is the newest
    revision's number, which is what a caller compares against to update safely.
    `created_at` is the oldest revision's timestamp, because that is when the agent came
    into being -- the newest revision's timestamp is when it was last edited, which is a
    different fact and the wrong one to publish as a creation time. `archived_at` is
    null for a live agent and is a timestamp for a retired one, and retirement is
    terminal, so a non-null value here never goes back to null.

    `definition` is the newest revision's body. Deliberately the newest and not the one
    a running Session pinned: this is what a caller sees when it reads the agent back,
    and a Session goes on reading the exact revision it resolved to.
    """

    definition_id: DefinitionId
    version: int
    created_at: datetime
    archived_at: datetime | None
    definition: AgentDefinition


@runtime_checkable
class AgentLifecycle(Protocol):
    """The whole-agent reads and writes, as the surface that serves them needs them.

    Beside `DefinitionRegistry` rather than folded into it, and satisfied structurally
    by the same object: every method here is about the agent that the revisions in that
    port are revisions *of*, so a second store would be a second thing to keep in step
    with the first. `runtime_checkable` so `agent_lifecycle_of` below can narrow a
    registry typed as the revision port. That check sees method names and not
    signatures, which is the same shallowness `core.ports` documents -- it catches the
    object that never grew these methods, not the one that grew them wrong.
    """

    async def page_agents(
        self,
        tenant_id: TenantId,
        *,
        include_archived: bool,
        created_from: datetime | None,
        created_to: datetime | None,
        after: tuple[datetime, DefinitionId] | None,
        limit: int,
    ) -> tuple[AgentRecord, ...]:
        """One page of a tenant's agents, newest first, filtered as asked."""
        ...

    async def read_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> AgentRecord | None:
        """One agent, or None when this tenant has no agent by that id."""
        ...

    async def archive_agent(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> datetime | None:
        """Retire the agent and say when; None when the agent is not this tenant's.

        Idempotent, and the timestamp it returns on a repeat is the ORIGINAL one.
        """
        ...

    async def register_at_revision(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
        expected: int,
    ) -> int | None:
        """Append a revision only while `expected` is still the newest number.

        None when it is not, which covers both a caller working from a stale read and a
        concurrent writer that got there first.
        """
        ...

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        """Append the next revision unconditionally, and report its number."""
        ...


def agent_lifecycle_of(registry: object) -> AgentLifecycle:
    """Narrow the wired definition registry to the whole-agent surface.

    A type narrowing rather than a validation: the registry comes from the composition
    root and no request can influence which object is there, so a registry without these
    methods is a wiring mistake and not something a caller did.

    It raises rather than degrading to "this tenant has no agents". A registry that
    cannot answer whether an agent is retired has not said it is live -- it has said
    nothing -- and serving the second as the first is how a retired agent starts work
    again. A `TypeError` surfaces as a 500 on the first request, which is loud at the
    one moment it is still cheap.
    """
    if not isinstance(registry, AgentLifecycle):
        raise TypeError(
            f"{type(registry).__name__} is wired as the definition registry but does "
            "not answer the whole-agent reads, so an agent could be read back or "
            "edited without its retirement ever being consulted"
        )
    return registry
