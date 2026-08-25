"""The six things a tenant does to a sandbox shape: create, read, list, update, archive,
delete.

A registration is refused here, at the moment the tenant who wrote the shape is present,
and not later inside a compilation whose failure would surface to whoever opened a
Session. That is the whole reason the parse runs on the way in as well as on the way
out.

A read of another tenant's id answers exactly as a read of an id nobody registered does:
a distinguishable answer is an existence oracle over other tenants' ids. Every route
here holds to that, which is why an update, an archive and a delete all begin by reading
the shape under the tenant predicate rather than by writing and inspecting what the
write touched.

**An update is a new revision, never a rewrite**, and that is what keeps the guarantee
this resource was built on rather than trading it away. The guarantee is that two
Sessions naming one id run in one shape; an id that quietly meant a wider sandbox on
Tuesday than on Monday would make "what was this Session allowed to do" unanswerable
after the fact. So an edit appends, a Session records the revision it resolved, and the
old revision stays exactly as the Sessions holding it left it.

**An archive is terminal and a delete is real.** Archiving refuses new Sessions and
further edits while leaving the shape readable; there is no unarchive, because a
reversible retirement is a different feature with a different failure mode. Deleting
removes every revision, and what makes that safe is the refusal in front of it: an
Environment no Session references has no history a removal could make unreadable.

Statuses on this surface are ours. Anthropic's reference states none anywhere -- its
pages have no error section by construction -- so every refusal below takes its status
from `STATUS_FOR` and nothing here chooses one.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Final, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.refusals import Refusal, refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.catalog.environments import (
    EnvironmentLifecycle,
    ResolvedEnvironment,
    UnknownEnvironment,
    list_environments,
    parse_environment,
    resolve_environment_revision,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import TenantId
from managed_agent.core.registration.environment import (
    CreateEnvironment,
    EnvironmentId,
    new_environment_id,
)

router = APIRouter(tags=["environments"])

REASON_ABSENT: Final = "no environment with that id is registered"
"""The one sentence every route here refuses an unknown id with.

Written once because it is answered for three different situations -- no such id, an id
another tenant registered, an id this call has just deleted -- and the three must not be
told apart. A second wording is a second answer, and the difference between two answers
is the oracle.
"""

DEFAULT_PAGE_SIZE: Final = 25
MAX_PAGE_SIZE: Final = 100
"""How many Environments one page may hold.

Bounded because an unbounded page is a whole-collection read wearing a limit parameter.
The adapter refuses anything above 500 outright; this is the tighter bound the tenant
surface publishes, so a caller learns it from a 400 naming the field rather than from a
500.
"""

_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]: {"model": PublicErrorEnvelope}
}
"""The refusal four routes here share, declared once for the published document.

Annotated because FastAPI's `responses` parameter is `dict[int | str, dict[str, Any]]`
and a bare literal assigned to a name infers something narrower.
"""


class InvalidCursor(Exception):
    """The caller sent something that is not a cursor this surface issued."""


@dataclass(frozen=True, slots=True)
class Cursor:
    """A position in one tenant's registration-ordered Environment list.

    Both halves are needed. Two Environments can be registered in one millisecond, and
    a position naming only the millisecond cannot say which of them the caller already
    holds -- so a page boundary landing between them repeats one row and drops one.

    Not the `Cursor` in `session_list.py`, and deliberately not shared with it. That one
    holds a `SessionId`; a single type over both would make a position in one collection
    a value the other accepts, so a token issued by a Session walk would be read here as
    a place in a list of shapes, and page from an id that names nothing.
    """

    created_at_ms: int
    environment_id: EnvironmentId

    def encode(self) -> str:
        """The position as a token, base64url with its padding stripped.

        Padding is stripped so the token carries no `=`, which would be percent-encoded
        in a query string and come back looking different from what was issued.
        """
        raw = f"{self.created_at_ms}.{self.environment_id}".encode()
        return urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> "Cursor":
        """Parse a token back into a position, or raise `InvalidCursor`.

        Everything that is not a token this surface issued is one refusal. There is no
        partial reading -- a token whose millisecond parses and whose id does not names
        no row, and treating half of it as a position would start the next page
        somewhere the caller never was.
        """
        try:
            padded = token + "=" * (-len(token) % 4)
            text = urlsafe_b64decode(padded.encode()).decode()
            milliseconds, _, identifier = text.partition(".")
            return cls(int(milliseconds), EnvironmentId(UUID(identifier)))
        except ValueError as exc:
            # binascii.Error and UnicodeDecodeError are both ValueError, so one clause
            # covers bad base64, bad utf-8, a non-numeric half and a malformed uuid.
            raise InvalidCursor(token) from exc


class EnvironmentRegistered(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: EnvironmentId


class EnvironmentView(BaseModel):
    """A shape read back. It carries no tenant: the caller is the tenant."""

    model_config = ConfigDict(frozen=True)

    id: EnvironmentId
    name: str
    runtime_image: str
    denied_paths: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    """Read back because it is the one field here that grants something. A caller who
    sent the wrong list has an agent that can reach the wrong place, and no other route
    would tell them."""

    revision: int
    """Which revision this is, which Anthropic's own resource does not carry.

    Published because their reference tells the caller to "keep your own record of the
    changes" and this platform keeps one, so withholding the number would leave a tenant
    unable to answer the two questions it exists for: whether the edit they just sent
    landed, and which shape a Session that pinned a revision is actually running in.
    """

    archived_at: datetime | None
    """When this Environment was retired, or null while it is live.

    Rendered with an explicit UTC offset (`+00:00`) rather than `Z`. Both are RFC 3339
    and this is what the stored timestamp serialises to; a consumer parsing one parses
    the other.
    """


class EnvironmentPage(BaseModel):
    """One page of Environments, and where the page after it starts.

    `data` and `next_page` are the names Anthropic's list response uses, so a client
    generated against their documentation reads this one. A local prose guide says the
    cursor parameters are `after_id`/`before_id`; that guide is wrong, and the reference
    page it purports to describe has `page` and `next_page`.

    `next_page` is null at the end of the walk rather than a token leading somewhere
    empty, so a caller stops on a field it can read instead of on a wasted round trip.
    There is deliberately no `prev_page`: paging backward needs the store to answer "the
    rows BEFORE this position", which it cannot, and a field emitted as null on every
    page would state something false about being on the first one.
    """

    model_config = ConfigDict(frozen=True)

    data: tuple[EnvironmentView, ...]
    next_page: str | None


class EnvironmentDeleted(BaseModel):
    """The body a delete answers with, which is why a delete is a 200 and not a 204.

    Two local prose guides say this route returns 204. Both are wrong: the reference
    types the response `BetaEnvironmentDeleteResponse` and shows `{"id": ...,
    "type": "environment_deleted"}`, and a body is not a 204.

    `type` is a constant, and it earns its place by making the body self-describing in a
    log or a webhook payload where nothing says which call produced it.
    """

    model_config = ConfigDict(frozen=True)

    id: EnvironmentId
    type: Literal["environment_deleted"] = "environment_deleted"


def environment_lifecycle(request: Request) -> EnvironmentLifecycle:
    """The wired store, widened to the surface the lifecycle routes need.

    `Platform.environment_store` is typed at the two-method port a Session's create and
    a pod's placement use, which is as wide as those two need and narrower than listing,
    editing, retiring and deleting need. The composition root wires the whole store, so
    this asks the object in front of it rather than asking for a second field on the
    Platform.

    A store that cannot do the rest is a fault in the composition root and not something
    a request can cause, so it refuses as `platform.internal` -- enveloped, with a
    request id to quote, rather than as a bare framework 500.
    """
    store = platform_from_request(request).environment_store
    if not isinstance(store, EnvironmentLifecycle):
        raise Refusal(
            ErrorCode.INTERNAL,
            "this app was wired with an environment store that cannot list, edit, "
            "retire or delete a shape",
        )
    return store


def _view_of(resolved: ResolvedEnvironment) -> EnvironmentView:
    """One resolved shape as the body every route here answers with.

    Written once so the four routes that return an Environment cannot describe one
    differently. They are reached by different acts -- a read, an edit, a retirement, a
    page -- and a caller comparing what two of them said about one id must not find a
    difference that only means two functions built the body.
    """
    environment = resolved.environment
    return EnvironmentView(
        id=environment.id,
        name=environment.name,
        runtime_image=environment.runtime_image,
        denied_paths=environment.denied_paths,
        allowed_domains=environment.allowed_domains,
        revision=resolved.revision,
        archived_at=resolved.archived_at,
    )


@router.post(
    "/environments",
    status_code=status.HTTP_201_CREATED,
    response_model=EnvironmentRegistered,
    responses={STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope}},
)
async def register(
    body: CreateEnvironment,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> EnvironmentRegistered | JSONResponse:
    """Register a sandbox shape and return the id Sessions will name it by.

    The id is minted here rather than accepted from the caller, so registering twice
    makes two shapes rather than overwriting one -- which is what the store refuses
    anyway, and refusing it at a route is a worse error message than never producing it.

    The shape is parsed before anything is written, and a refusal carries a code from
    the published set rather than pydantic's field error, because what is being refused
    is a rule about paths and images rather than a field's type.
    """
    try:
        environment = parse_environment(
            environment_id=new_environment_id(),
            tenant_id=tenant_id,
            name=body.name,
            runtime_image=body.runtime_image,
            denied_paths=body.denied_paths,
            allowed_domains=body.allowed_domains,
        )
    except ValueError as refused:
        return refuse(ErrorCode.REQUEST_INVALID, str(refused))
    await platform_from_request(request).environment_store.insert(environment)
    return EnvironmentRegistered(id=environment.id)


@router.get(
    "/environments",
    response_model=EnvironmentPage,
    responses={
        STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]: {"model": PublicErrorEnvelope}
    },
)
async def list_registered(
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    store: Annotated[EnvironmentLifecycle, Depends(environment_lifecycle)],
    page: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    include_archived: bool = False,
) -> EnvironmentPage | JSONResponse:
    """One page of the calling tenant's Environments, newest first, one row per id.

    One row per id and not one per revision: what a tenant is listing is the names they
    can start a Session under, and an id is one name however many times it was edited.
    Each row shows the latest revision.

    Archived Environments are absent unless asked for, and that direction is the useful
    one: an archived shape starts no Session, so a list carrying it by default would put
    entries in front of a caller that every create would refuse.

    The store is asked for one row more than will be returned. That extra row is the
    whole answer to "is there another page", and it is why `next_page` is null rather
    than a token leading somewhere empty.

    Another tenant's Environments are absent rather than filtered out here: the tenant
    is a term in the store's own query, so there is no point in this function at which a
    cross-tenant row exists and has to be dropped.

    A cursor this surface did not issue is refused rather than treated as the start of
    the collection. Starting over on a bad cursor would silently hand back the newest
    page again, which reads as the walk having looped rather than failed.
    """
    after: tuple[int, UUID] | None = None
    if page is not None:
        try:
            position = Cursor.decode(page)
        except InvalidCursor as exc:
            raise Refusal(
                ErrorCode.PAGINATION_CURSOR_INVALID,
                "cursor was not issued by this surface",
            ) from exc
        after = (position.created_at_ms, position.environment_id)
    rows = await list_environments(store, tenant_id, after, limit + 1, include_archived)
    shown = rows[:limit]
    more = len(rows) > limit
    last = shown[-1] if shown else None
    return EnvironmentPage(
        data=tuple(_view_of(row.resolved) for row in shown),
        next_page=(
            Cursor(last.created_at_ms, last.resolved.environment.id).encode()
            if more and last is not None
            else None
        ),
    )


@router.get(
    "/environments/{environment_id}",
    response_model=EnvironmentView,
    responses=_NOT_FOUND,
)
async def read(
    environment_id: EnvironmentId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> EnvironmentView | JSONResponse:
    """The latest revision of one registered shape, as it stands.

    Reads through the same parse a registration goes through, so a row that no longer
    satisfies today's rules raises rather than being handed back -- a shape that would
    not be accepted now is not a shape a Session should be started in.

    A retired Environment reads back normally, with `archived_at` set. That is the
    point of retirement being a fact on the resource rather than a deletion: the caller
    who archived one by mistake can still see what they had, and the one who never
    archived anything can tell the two states apart without a second call.
    """
    try:
        resolved = await resolve_environment_revision(
            platform_from_request(request).environment_store,
            environment_id,
            tenant_id,
        )
    except UnknownEnvironment:
        return refuse(ErrorCode.ENVIRONMENT_NOT_FOUND, REASON_ABSENT)
    return _view_of(resolved)


@router.post(
    "/environments/{environment_id}",
    response_model=EnvironmentView,
    responses={
        **_NOT_FOUND,
        STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.ENVIRONMENT_ARCHIVED]: {"model": PublicErrorEnvelope},
    },
)
async def update(
    environment_id: EnvironmentId,
    body: CreateEnvironment,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    store: Annotated[EnvironmentLifecycle, Depends(environment_lifecycle)],
) -> EnvironmentView | JSONResponse:
    """Write the next revision of a shape, and answer with the shape as it now stands.

    The whole body is sent and the whole shape is replaced, which is what `POST` on this
    resource means upstream too. There is no partial edit and that is not a shortcut: a
    field left out of a revision would have to be filled in from the revision below it,
    so a caller who dropped `allowed_domains` by accident would silently keep an egress
    grant they had meant to remove.

    Existence is settled first, under the tenant predicate, because the write numbers
    itself from whatever rows already carry this id -- so without the read an id nobody
    registered would be written as a brand new Environment at revision 1, by a call that
    reads as an edit and answers 200 either way.

    An archived Environment refuses. Retirement is terminal, and an edit that landed on
    one would produce a revision no Session can ever be started in -- accepted, stored,
    and unusable.

    The new shape is parsed before anything is written, so a refusal leaves no revision
    behind and the id goes on meaning what it meant.
    """
    try:
        current = await resolve_environment_revision(store, environment_id, tenant_id)
    except UnknownEnvironment:
        return refuse(ErrorCode.ENVIRONMENT_NOT_FOUND, REASON_ABSENT)
    if current.archived:
        return refuse(
            ErrorCode.ENVIRONMENT_ARCHIVED,
            "that environment was retired, and a retirement is terminal, so its "
            "shape can no longer be changed",
            environment_id=str(environment_id),
        )
    try:
        revised = parse_environment(
            environment_id=environment_id,
            tenant_id=tenant_id,
            name=body.name,
            runtime_image=body.runtime_image,
            denied_paths=body.denied_paths,
            allowed_domains=body.allowed_domains,
        )
    except ValueError as refused:
        return refuse(ErrorCode.REQUEST_INVALID, str(refused))
    revision = await store.insert_revision(revised)
    return _view_of(
        ResolvedEnvironment(environment=revised, revision=revision, archived_at=None)
    )


@router.post(
    "/environments/{environment_id}/archive",
    response_model=EnvironmentView,
    responses=_NOT_FOUND,
)
async def archive(
    environment_id: EnvironmentId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    store: Annotated[EnvironmentLifecycle, Depends(environment_lifecycle)],
) -> EnvironmentView | JSONResponse:
    """Retire an Environment: no new Session, no further edit, history intact.

    Nothing about a Session already running changes here. This writes one row saying the
    Environment is retired; the checks that read it run when a Session is created and
    when a shape is edited, and a Session that has already resolved never asks again --
    which is what lets a shape be withdrawn without stopping work in flight.

    Retiring twice is not an error, and the response is what makes that observable: the
    second call answers 200 carrying the timestamp of the FIRST retirement. A refusal
    there would push every client into a read-then-write race to avoid it, and a fresh
    timestamp would claim the Environment stopped being referenceable at the moment of a
    retry -- a false fact about when new Sessions began being refused.

    There is no unarchive, on this surface or any other. A reversible retirement is a
    different feature: it would mean a Session could be created in a shape that had been
    withdrawn and restored, and nothing in the log would say which of those it was.
    """
    try:
        resolved = await resolve_environment_revision(store, environment_id, tenant_id)
    except UnknownEnvironment:
        return refuse(ErrorCode.ENVIRONMENT_NOT_FOUND, REASON_ABSENT)
    retired_at = await store.archive(environment_id, tenant_id)
    if retired_at is None:
        # The read above found it and the write found nothing, so a delete landed in
        # between. Answered as absent, which is what it now is.
        return refuse(ErrorCode.ENVIRONMENT_NOT_FOUND, REASON_ABSENT)
    return _view_of(
        ResolvedEnvironment(
            environment=resolved.environment,
            revision=resolved.revision,
            archived_at=retired_at,
        )
    )


@router.delete(
    "/environments/{environment_id}",
    response_model=EnvironmentDeleted,
    responses={
        **_NOT_FOUND,
        STATUS_FOR[ErrorCode.ENVIRONMENT_IN_USE]: {"model": PublicErrorEnvelope},
    },
)
async def delete(
    environment_id: EnvironmentId,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    store: Annotated[EnvironmentLifecycle, Depends(environment_lifecycle)],
) -> EnvironmentDeleted | JSONResponse:
    """Remove every revision of a shape no Session is running in.

    Refused while any Session that has not stopped names this Environment, and the count
    travels in the refusal so a caller knows whether they are waiting on one Session or
    a hundred. A count and not a list of ids: the list is unbounded, and a refusal whose
    size depends on how much work a tenant has is a refusal that stops being readable
    exactly when it matters.

    That refusal is what makes a hard delete safe rather than a data-loss hazard. An
    Environment nothing references has no Session whose history a removal could make
    unreadable, which is why there is no tombstone table -- one would record that a
    shape used to exist for the benefit of nobody who can ask.

    Deleting an Environment that is already gone is a 404 and not a second 200, and this
    is where a delete and an archive part company. An archive answers about a fact that
    is still there to be read; a delete removed the row, so a repeat has nothing to
    report and a 200 would claim this call deleted something it did not.

    An archived Environment may still be deleted. Retirement stops new Sessions; it is
    not a reason to keep rows nobody can reach.
    """
    try:
        await resolve_environment_revision(store, environment_id, tenant_id)
    except UnknownEnvironment:
        return refuse(ErrorCode.ENVIRONMENT_NOT_FOUND, REASON_ABSENT)
    holding = await store.sessions_referencing(environment_id)
    if holding:
        return refuse(
            ErrorCode.ENVIRONMENT_IN_USE,
            "sessions that have not stopped are still running in this environment, "
            "so deleting it would leave them naming a shape that does not exist",
            environment_id=str(environment_id),
            sessions=holding,
        )
    if not await store.delete(environment_id, tenant_id):
        # Read, then counted, then gone: another delete of the same id won the race.
        # Reported as absent, which is the same answer that call's loser deserves.
        return refuse(ErrorCode.ENVIRONMENT_NOT_FOUND, REASON_ABSENT)
    return EnvironmentDeleted(id=environment_id)
