"""Reading an agent back, editing it, and retiring it for good.

Four routes over the stack of revisions `agent_definition` holds. Three of them treat
that stack as one thing -- an agent has a current shape, a creation time, and either a
retirement or none -- and the fourth adds to it. What makes them a module of their own
rather than an addition to `agent_versions.py` beside it is the unit they address: that
one addresses a revision by number, and every route here addresses the agent.

The distinction is sharpest at archive. Retiring a *version* withdraws one revision so
no new Session resolves to it while its siblings stay live, and it is reversible in
effect -- a later revision can be appended and the agent goes on working. Retiring the
*agent* is terminal: `agent_archive` refuses an UPDATE by raising, nothing deletes from
it, and this module refuses every write to an agent that has a row there. A caller
cannot approximate one with the other in either direction.

An update writes a new revision rather than rewriting the current one, because that is
the only thing this store can do -- `agent_definition` refuses an UPDATE by trigger. The
consequence is worth stating plainly rather than treating as an implementation detail:
an edit is additive, a Session that already resolved revision N goes on reading exactly
the bytes it resolved to, and the `version` in the response is the number the edit
landed on.

Optimistic concurrency is by version and not by content. A caller that re-sends the
values already stored is still refused if the version it holds is stale, because the
question the check answers is "did anything change under you since you read", and a
caller who has not seen the intervening edit cannot know whether re-sending its own body
undoes something.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Final
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from managed_agent.control.api.refusals import Refusal, refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.catalog.definitions import (
    AgentRecord,
    agent_lifecycle_of,
)
from managed_agent.control.skills.evaluation import (
    Standing,
    skill_evals_of,
    standing_of,
)
from managed_agent.control.skills.registry import SkillsUnresolvable, read_attached
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import DefinitionId, TenantId
from managed_agent.core.registration.definition import AgentDefinition

router = APIRouter(tags=["agents"])

DEFAULT_PAGE_SIZE: Final = 20
MAX_PAGE_SIZE: Final = 100
"""How many agents one page may hold.

Bounded because an unbounded page is a whole-collection read wearing a limit parameter:
the store materialises every matching row before the caller sees the first. Twenty by
default rather than the twenty-five `GET /v1/sessions` uses, because these are the
numbers the surface this one is modelled on publishes and a caller writing against both
should not have to remember which collection changed the default.
"""


class InvalidCursor(Exception):
    """The caller sent something that is not a cursor this surface issued."""


@dataclass(frozen=True, slots=True)
class Cursor:
    """A position in one tenant's creation-ordered agent list.

    Both halves are needed. Two agents can be registered in the same microsecond, and a
    position naming only the timestamp cannot say which of them the caller already has
    -- so a page boundary landing between them would repeat one row or drop the other.

    The timestamp is carried as its own ISO 8601 spelling rather than as a count of
    milliseconds. A count would have to round, and a rounded position compares against
    the stored microsecond as though it were earlier -- which repeats the row it was
    meant to skip past.
    """

    created_at: datetime
    agent_id: DefinitionId

    def encode(self) -> str:
        """The position as a token, base64url with its padding stripped.

        Padding is stripped so the token carries no `=`, which would be percent-encoded
        in a query string and come back looking different from what was issued.

        The separator is `|` because it appears in neither half: an ISO 8601 timestamp
        holds `-`, `:`, `.` and `+`, and a uuid holds `-`.
        """
        raw = f"{self.created_at.isoformat()}|{self.agent_id}".encode()
        return urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> "Cursor":
        """Parse a token back into a position, or raise `InvalidCursor`.

        Everything that is not a token this surface issued is one refusal. There is no
        partial reading -- a token whose timestamp parses and whose id does not names no
        row, and treating half of it as a position would start the next page somewhere
        the caller never was.
        """
        try:
            padded = token + "=" * (-len(token) % 4)
            text = urlsafe_b64decode(padded.encode()).decode()
            moment, _, identifier = text.partition("|")
            return cls(datetime.fromisoformat(moment), DefinitionId(UUID(identifier)))
        except ValueError as exc:
            # binascii.Error and UnicodeDecodeError are both ValueError, so one clause
            # covers bad base64, bad utf-8, an unparseable timestamp and a malformed
            # uuid. A token with no separator needs no check of its own: `partition`
            # leaves the id half empty and `UUID("")` raises.
            raise InvalidCursor(token) from exc


class Agent(AgentDefinition):
    """One agent as a caller reads it: its current shape plus who and when it is.

    A subclass rather than a wrapper with a nested `definition`, so the configuration
    fields sit at the top level where the surface this one is modelled on puts them, and
    so the field list is written once. Re-declaring `name`, `model` and the rest here
    would be a second copy of the definition's shape, free to fall behind it the next
    time a field is added.

    `version` is the newest revision's number and is the value to send back on an
    update. `created_at` is when the agent came into being rather than when it was last
    edited -- the two differ the moment anything is edited, and a caller sorting by
    creation wants the first.

    `archived_at` is null while the agent is live and is a timestamp once it is retired.
    Null and not absent, because a caller reading this field is asking a question with
    two real answers; an absent field would leave "not retired" and "this response does
    not say" looking identical.
    """

    id: DefinitionId
    version: int
    created_at: datetime
    archived_at: datetime | None


class AgentPage(BaseModel):
    """One page of agents, and where the page after it starts.

    `data` rather than `agents`, and `next_page` rather than a wrapped cursor object,
    because those are the names the surface this one is modelled on uses and a generated
    client reads the array off the key it was generated against.

    `next_page` is null at the end of the walk rather than a token leading somewhere
    empty, so a caller stops on a field it can read instead of on a wasted round trip.
    There is deliberately no `prev_page`: paging backward needs the store to answer "the
    rows before this position", which it cannot, and a field emitted as null on every
    page would state that the caller is on the first one.
    """

    model_config = ConfigDict(frozen=True)

    data: list[Agent]
    next_page: str | None


class AgentUpdate(AgentDefinition):
    """A whole new shape for an agent, and optionally the version it replaces.

    A full definition and not a patch, because the store writes whole revisions: a
    partial update would have to merge the submitted fields onto the stored ones, and
    what that means for a set-valued field like `skills` -- replace, union, or
    difference -- is a decision nothing here is entitled to make silently. A caller
    reads the agent, changes what it wants, and sends the result back.

    `version` is the agent's current version and is optional. Supplied, the update is
    refused unless it still matches, which is what lets a caller edit without first
    proving nothing changed under it. Omitted, the update applies unconditionally --
    last write wins, and that is the caller's choice to make rather than a default this
    surface imposes.
    """

    version: int | None = Field(default=None, ge=1)

    def as_definition(self) -> AgentDefinition:
        """The definition half alone, with the concurrency token dropped.

        Converted rather than passed through, and the conversion is load-bearing: the
        store writes whatever body it is handed, `AgentDefinition` forbids unknown
        fields, and a stored body carrying a stray `version` key would be refused by
        every later read of it. The failure would surface as an agent that cannot be
        read back, far from the update that wrote it.

        Iterating the model rather than dumping it, because `model_dump()` serialises
        and `skills` is a `frozenset[SkillAttachment]`: dumping turns each attachment
        into a `dict` and then tries to rebuild the set from them, which raises
        `TypeError: unhashable type: 'dict'`. An update attaching no skill dumps an
        empty set and never hashes anything, so the bug is invisible until a caller
        attaches one. Iteration yields the field values as they are -- the attachments
        stay models, and `model_validate` passes them straight through.
        """
        return AgentDefinition.model_validate(
            {name: value for name, value in self if name != "version"}
        )


def _as_agent(record: AgentRecord, archived_at: datetime | None) -> Agent:
    """One stored agent as the object a caller reads.

    `archived_at` is passed in rather than taken from the record because the archive
    route knows a newer answer than its own pre-read did: it has just written the
    retirement, or absorbed a repeat and been handed the original timestamp. Every other
    caller passes the record's own value.

    `dict(record.definition)` and not `model_dump()`, for the reason spelled out in
    `AgentUpdate.as_definition`: dumping a `frozenset[SkillAttachment]` rebuilds the set
    out of dicts and raises `TypeError: unhashable type: 'dict'`. Both read routes go
    through here, so that raised a 500 for every definition attaching a skill -- and the
    listing failed whole rather than per row, leaving a tenant holding one such agent
    unable to read any of them. Iteration hands over the field values untouched, and
    `Agent` declares the same fields it inherits from `AgentDefinition`, so they
    validate as themselves.
    """
    return Agent(
        id=record.definition_id,
        version=record.version,
        created_at=record.created_at,
        archived_at=archived_at,
        **dict(record.definition),
    )


_NOT_YOURS: dict[int | str, dict[str, Any]] = {
    STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]: {"model": PublicErrorEnvelope}
}
"""The refusal every route addressing one agent shares, declared once for the schema.

One code answers "no such agent" and "not yours". Distinguishing them would let anyone
holding an id learn from the refusal whether it names another tenant's agent.

Annotated because FastAPI's `responses` parameter is `dict[int | str, dict[str, Any]]`
and a bare literal assigned to a name infers a narrower type than that.
"""

_UPDATE_REFUSALS: dict[int | str, dict[str, Any]] = {
    **_NOT_YOURS,
    STATUS_FOR[ErrorCode.AGENT_ARCHIVED]: {"model": PublicErrorEnvelope},
    STATUS_FOR[ErrorCode.DEFINITION_INVALID]: {"model": PublicErrorEnvelope},
}
"""What an update can refuse. `AGENT_ARCHIVED` and `AGENT_VERSION_CONFLICT` share a
status, so the schema lists it once; the code in the envelope is what tells them
apart."""


@router.get("/agents", response_model=AgentPage)
async def list_agents(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    page: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    include_archived: bool = False,
    created_from: Annotated[
        AwareDatetime | None, Query(alias="created_at[gte]")
    ] = None,
    created_to: Annotated[AwareDatetime | None, Query(alias="created_at[lte]")] = None,
) -> AgentPage | JSONResponse:
    """One page of the calling tenant's agents, newest first.

    Retired agents are absent unless asked for. An agent is retired to get it out of the
    way, so a listing that kept showing it would leave the retirement doing nothing a
    caller could see -- and the tenant who needs to audit what they once had asks for it
    by name with `include_archived`.

    The store is asked for one row more than will be returned. That extra row is the
    whole answer to "is there another page", and it is why `next_page` is null rather
    than a token leading somewhere empty.

    The two `created_at` bounds are inclusive and either may be omitted. Both require an
    offset -- `AwareDatetime` refuses a bare local time at the boundary, naming the
    field -- because a bound with no offset is a bound whose meaning depends on where
    the reader is standing, and the rows it selects would differ by hours.

    Another tenant's agents are absent rather than filtered out here: the tenant is a
    term in the store's own query, so there is no point in this function where a
    cross-tenant row exists and has to be dropped.
    """
    after: tuple[datetime, DefinitionId] | None = None
    if page is not None:
        try:
            position = Cursor.decode(page)
        except InvalidCursor as exc:
            raise Refusal(
                ErrorCode.PAGINATION_CURSOR_INVALID,
                "cursor was not issued by this surface",
            ) from exc
        after = (position.created_at, position.agent_id)

    registry = agent_lifecycle_of(platform_from_request(request).definition_registry)
    rows = await registry.page_agents(
        tenant_id,
        include_archived=include_archived,
        created_from=created_from,
        created_to=created_to,
        after=after,
        limit=limit + 1,
    )
    shown = rows[:limit]
    more = len(rows) > limit
    return AgentPage(
        data=[_as_agent(record, record.archived_at) for record in shown],
        next_page=(
            Cursor(shown[-1].created_at, shown[-1].definition_id).encode()
            if more
            else None
        ),
    )


@router.get("/agents/{agent_id}", response_model=Agent, responses=_NOT_YOURS)
async def read_agent(
    agent_id: DefinitionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> Agent | JSONResponse:
    """One agent, retired or not.

    A retired agent answers 200 with `archived_at` set rather than 404 or 409. Retiring
    makes an agent unusable, not non-existent: the caller's id is correct, and a refusal
    here would send them looking for a mistake they did not make. What the retirement
    refuses is every *write*, and each of those says so with its own code.
    """
    registry = agent_lifecycle_of(platform_from_request(request).definition_registry)
    record = await registry.read_agent(agent_id, tenant_id)
    if record is None:
        return refuse(
            ErrorCode.DEFINITION_NOT_FOUND,
            "no agent with that id is registered to this tenant",
            agent_id=str(agent_id),
        )
    return _as_agent(record, record.archived_at)


@router.post("/agents/{agent_id}", response_model=Agent, responses=_UPDATE_REFUSALS)
async def update_agent(
    agent_id: DefinitionId,
    body: AgentUpdate,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> Agent | JSONResponse:
    """Replace an agent's shape, and report the revision the edit landed on.

    Every refusal leaves no revision behind, and the two that are decided here are
    decided before anything is written. They are ordered by how little the caller can do
    about them: an id that does not resolve, then a retirement that is permanent, then
    the two gates a caller fixes by changing what they sent.

    A retired agent is refused rather than edited. Archive is terminal, and an edit
    after it would make the retirement reversible in effect -- the agent would have a
    newer revision than the one that was current when it was retired, and anything
    reasoning about what a retired agent was would be reading an edit made afterwards.

    That refusal names the agent and not the moment it was retired. `archived_at` is on
    the agent object, where it is rendered once by the response model; writing it into a
    refusal `detail` means formatting the same instant a second way, and the two
    spellings -- `...Z` from the model, `...+00:00` from `isoformat` -- are the same
    timestamp described differently on one API. A caller that needs the moment reads the
    agent.

    The `version` check compares numbers, not bodies. A caller re-sending exactly what
    is stored is still refused when the version it holds is stale, because what it is
    telling us is which state it read, and it has not read the state its write would
    land on. The refusal names the version the platform holds, so the retry is a re-read
    rather than a guess.

    The CI eval gate and the attached skill ids are checked here as well as on
    `POST /v1/agents` and `POST /v1/agents/{id}/versions`, and they have to be: this
    route accepts a whole definition, so it accepts a `skills_revision` and a `skills`
    array, and a gate held on the other two doors alone would be walked around by
    updating an agent instead of creating or versioning one.

    The version is never compared here. It is passed to the store as a condition on the
    write, so the comparison and the append are one statement and there is no window
    between them -- and a comparison here as well would be a second answer to the same
    question, shadowing the one that is load-bearing. Measured, and that is why it is
    gone: deleting the comparison that used to sit above left all 49 of this slice's
    tests green, because the store refused every case it had been refusing.
    """
    platform = platform_from_request(request)
    registry = agent_lifecycle_of(platform.definition_registry)
    record = await registry.read_agent(agent_id, tenant_id)
    if record is None:
        return refuse(
            ErrorCode.DEFINITION_NOT_FOUND,
            "no agent with that id is registered to this tenant",
            agent_id=str(agent_id),
        )
    if record.archived_at is not None:
        return refuse(
            ErrorCode.AGENT_ARCHIVED,
            "this agent was retired and accepts no further edits",
            agent_id=str(agent_id),
        )
    definition = body.as_definition()
    facts = await skill_evals_of(platform.definition_registry).eval_facts(
        tenant_id, definition.skills_repository, definition.skills_revision
    )
    if standing_of(facts) is Standing.BLOCKED:
        return refuse(
            ErrorCode.DEFINITION_SKILLS_REVISION_NOT_ACCEPTED,
            "this skills revision has not cleared its repository's CI eval "
            "baseline, so no agent may be pinned to it",
            field="skills_revision",
            skills_repository=definition.skills_repository,
            skills_revision=definition.skills_revision,
        )
    try:
        await read_attached(platform.skill_store, tenant_id, definition)
    except SkillsUnresolvable as unresolvable:
        return refuse(
            ErrorCode.DEFINITION_INVALID,
            unresolvable.message,
            field="skills",
            **unresolvable.detail,
        )
    if body.version is None:
        written: int | None = await registry.register(agent_id, tenant_id, definition)
    else:
        written = await registry.register_at_revision(
            agent_id, tenant_id, definition, body.version
        )
    if written is None:
        return refuse(
            ErrorCode.AGENT_VERSION_CONFLICT,
            "the version supplied is not this agent's current version, so this "
            "update was written against a state that has since changed",
            agent_id=str(agent_id),
            version=record.version,
        )
    return _as_agent(
        AgentRecord(
            definition_id=agent_id,
            version=written,
            created_at=record.created_at,
            archived_at=None,
            definition=definition,
        ),
        None,
    )


@router.post("/agents/{agent_id}/archive", response_model=Agent, responses=_NOT_YOURS)
async def archive_agent(
    agent_id: DefinitionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> Agent | JSONResponse:
    """Retire an agent for good, and return it as it now stands.

    There is no route back. Nothing here unarchives and nothing deletes an agent, so
    this is the last state transition an agent has -- which is why the whole agent is
    returned rather than an acknowledgement: the response is the last full description
    of it that will ever change.

    Retiring twice is a success and returns the timestamp of the FIRST retirement. The
    caller's intent is satisfied either way, and a caller whose first call timed out is
    retrying rather than making a second request -- answering 409 would turn a lost
    response into an error about a state that is exactly what they asked for. The
    original timestamp and not a fresh one, because when the agent stopped being
    startable is a fact about the first call.

    A Session already running keeps running. This writes one row saying the agent is
    retired; the checks that read it run when something new is created, and a Session
    that has already resolved its revision never asks again.
    """
    registry = agent_lifecycle_of(platform_from_request(request).definition_registry)
    record = await registry.read_agent(agent_id, tenant_id)
    if record is None:
        return refuse(
            ErrorCode.DEFINITION_NOT_FOUND,
            "no agent with that id is registered to this tenant",
            agent_id=str(agent_id),
        )
    archived_at = await registry.archive_agent(agent_id, tenant_id)
    if archived_at is None:
        # The read above found the agent, so this can only mean it stopped being this
        # tenant's between the two calls -- which nothing can currently cause, since
        # ownership never changes and nothing deletes a definition. Reported as the
        # same refusal rather than asserted, because the alternative is a 500 for a
        # caller whose id was fine.
        return refuse(
            ErrorCode.DEFINITION_NOT_FOUND,
            "no agent with that id is registered to this tenant",
            agent_id=str(agent_id),
        )
    return _as_agent(record, archived_at)
