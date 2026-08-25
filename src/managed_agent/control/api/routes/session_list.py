"""GET /v1/sessions — one tenant's Sessions, newest first, a page at a time.

A listing row deliberately carries no state. A Session's state is a fold over its own
Event Log, so putting it on a page of twenty-five rows would be twenty-five folds on a
read whose job is to help a caller *find* a Session; a caller that wants state reads the
Session itself.

The cursor is opaque and the caller round-trips it unread. What it encodes is the
store's ordering key, and a caller that parsed it would be depending on that ordering
— which is the coupling an opaque token exists to prevent. Opaque here rather than the
sequence-range paging the Event Log uses, because a sequence range needs a sequence and
a collection of Sessions has none: there is no number a caller could name that the
platform guarantees.

The direction of travel is inside the token too. There is one page parameter on the
wire, so a `prev_page` handed back to it would otherwise walk forward and return the
page the caller was already on -- a loop, and one that looks like paging. A second query
parameter naming a direction would work and was not taken: it makes two fields that must
agree, and a caller pairing yesterday's token with the other direction gets a page from
neither walk.

Nothing here authenticates. The tenant arrives from the placeholder in `tenancy.py`,
which trusts a header, and no decision record picks an authentication mechanism yet.
What *is* decided here is the failure direction: a request carrying no tenant is refused
rather than defaulted, because an unscoped read of this collection is every tenant's
Sessions at once.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Final
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from managed_agent.control.api.refusals import Refusal
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.ports import SessionListing, SessionsWalkedBackward

router = APIRouter(tags=["sessions"])

CURSOR_INVALID: Final = ErrorCode.PAGINATION_CURSOR_INVALID.value
SESSION_NOT_FOUND: Final = ErrorCode.SESSION_NOT_FOUND.value
"""The two refusal codes this slice can produce.

`ErrorCode` now carries both of these, and each literal is kept only because it is the
name a test imports to assert the wire string. Neither is the source of truth: the
source is the enum member beside it, and the assignment below derives from it so the two
cannot drift.
"""

DEFAULT_PAGE_SIZE: Final = 25
MAX_PAGE_SIZE: Final = 100
"""How many Sessions one page may hold.

Bounded because an unbounded page is a whole-collection read wearing a limit parameter:
the store materialises every matching row before the caller sees the first, and a
tenant's Session count has no natural ceiling. The adapter refuses anything above 500
outright; this is the tighter bound the tenant surface publishes, so a caller learns it
from a 400 naming the field rather than from a 500.
"""


class InvalidCursor(Exception):
    """The caller sent something that is not a cursor this surface issued."""


class Walk(StrEnum):
    """Which way a cursor is pointing.

    The values are the mark that goes in the token, and forward's is empty on purpose: a
    forward token encodes to the same bytes it did before this field existed, so every
    cursor already in a caller's hands keeps working and there is nothing to migrate.

    Two members rather than two cursor types. A union of `Forward` and `Backward` would
    also remove the illegal state, and buys nothing here: neither shape can hold a
    position the other cannot, the only difference is which comparison the store makes,
    and both halves would need their own copy of `encode` and `decode`.
    """

    FORWARD = ""
    BACKWARD = "b"


@dataclass(frozen=True, slots=True)
class Cursor:
    """A position in one tenant's creation-ordered Session list.

    Both halves are needed. Two Sessions can share a millisecond, and a position naming
    only the millisecond cannot say which of them the caller already has — so a page
    boundary landing between them would repeat one row or drop the other.
    """

    created_at_ms: int
    session_id: SessionId
    walk: Walk = Walk.FORWARD
    """Which way the page this token opens is walked. See the module docstring."""

    def encode(self) -> str:
        """The position as a token, base64url with its padding stripped.

        Padding is stripped so the token carries no `=`, which would be percent-encoded
        in a query string and come back looking different from what was issued.
        """
        mark = "" if self.walk is Walk.FORWARD else f".{self.walk.value}"
        raw = f"{self.created_at_ms}.{self.session_id}{mark}".encode()
        return urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> "Cursor":
        """Parse a token back into a position, or raise `InvalidCursor`.

        Everything that is not a token this surface issued is one refusal. There is no
        partial reading — a token whose millisecond parses and whose id does not names
        no row, and treating half of it as a position would start the next page
        somewhere the caller never was.

        A missing separator needs no check of its own, and one was written here and
        then removed rather than left as reassurance. When the separator is absent
        `partition` returns the whole string as the first part and `""` as the third,
        and `UUID("")` raises — so a token with no dot is already refused, by the parse
        that has to happen anyway. An explicit `if not separator: raise` could never be
        the branch that refused anything: dead in one direction and redundant in the
        other, which is the shape of a guard that reads as protection and is not.
        """
        try:
            padded = token + "=" * (-len(token) % 4)
            text = urlsafe_b64decode(padded.encode()).decode()
            milliseconds, _, rest = text.partition(".")
            identifier, _, mark = rest.partition(".")
            # `Walk("")` is FORWARD, so a two-field token -- every token issued before
            # the direction existed -- reads as forward without a branch for it. An
            # unknown mark raises ValueError from the enum lookup and is refused by the
            # clause below, alongside bad base64 and a malformed uuid.
            return cls(int(milliseconds), SessionId(UUID(identifier)), Walk(mark))
        except ValueError as exc:
            # binascii.Error and UnicodeDecodeError are both ValueError, so one clause
            # covers bad base64, bad utf-8, a non-numeric half and a malformed uuid.
            raise InvalidCursor(token) from exc


class SessionListed(BaseModel):
    """One Session as it appears in a list. No state: see the module docstring."""

    id: SessionId
    definition_id: DefinitionId
    definition_revision: str
    created_at_ms: int


class SessionPage(BaseModel):
    """One page of Sessions, and where the pages either side of it start.

    Both tokens are null at the end of their walk rather than a token leading somewhere
    empty, so a caller stops on a field it can read instead of on a wasted round trip.

    `prev_page` is ABSENT, not null, from a deployment whose store cannot walk backward
    -- and null only where the caller is demonstrably on the first page. Three states,
    because there are three things to say and a present-and-null field can only say two
    of them: a deployment that cannot answer would otherwise report every page as the
    first one. An absent field leaves a consumer's own default to mean "unknown", which
    is the same reason the compiled sandbox document omits its network keys instead of
    writing them off.

    Neither token costs a query. Going forward, the page is fetched one row longer than
    it is shown and that extra row is the whole answer to "is there another"; the
    previous page is named by the very cursor the caller arrived on, since a forward
    cursor is the last row of the page before. Going backward the extra row does the
    same job at the other end.
    """

    sessions: list[SessionListed]
    next_page: str | None
    prev_page: str | None = None
    """Defaulted so it can be left UNSET, which is how it serialises away.

    The default is never the value a response carries: the route passes this field
    explicitly on every deployment that can page backward, null included, and omits the
    argument entirely on one that cannot. Pydantic tracks which of those happened, and
    the route is declared `response_model_exclude_unset`, so an omitted argument is an
    absent key rather than a null one.
    """


def _listed(row: SessionListing) -> SessionListed:
    """One store row as one wire row.

    Shared by the two directions rather than written out in each. What a listing row
    means on the wire is one piece of knowledge, and a second transcription is free to
    drop a field that only one of the two walks would then be missing.
    """
    return SessionListed(
        id=row.id,
        definition_id=row.definition_id,
        definition_revision=row.definition_revision,
        created_at_ms=row.created_at_ms,
    )


@router.get("/sessions", response_model_exclude_unset=True)
async def list_sessions(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    page: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
) -> SessionPage:
    """One page of the calling tenant's Sessions, newest first.

    The store is asked for one row more than will be returned, in whichever direction
    the cursor points. That extra row is the whole answer to "is there another page that
    way", and it is why the token for it is null rather than one leading somewhere empty
    -- a caller stops when the field is null instead of on a wasted round trip.

    The page a caller came FROM needs no row at all. A forward cursor is the last row of
    the previous page, so that same key, read backward and inclusively, reproduces that
    page exactly -- which is why `page_ending_at` takes the page's own oldest row rather
    than the row before it.

    Another tenant's Sessions are absent rather than filtered out here: the tenant is a
    term in the store's own query, so there is no point in this function where a cross-
    tenant row exists and has to be dropped.

    A cursor this surface did not issue is refused with 400 rather than treated as the
    start of the collection. Starting over on a bad cursor would silently hand a caller
    the newest page again, which reads as the walk having looped rather than failed. A
    backward cursor arriving at a deployment that cannot walk backward is the same
    refusal for the same reason -- this surface did not issue it, because a store
    without that capability never puts a `prev_page` on a response at all.
    """
    store = platform_from_request(request).session_registry

    position: Cursor | None = None
    if page is not None:
        try:
            position = Cursor.decode(page)
        except InvalidCursor as exc:
            raise Refusal(
                ErrorCode.PAGINATION_CURSOR_INVALID,
                "cursor was not issued by this surface",
            ) from exc

    if position is not None and position.walk is Walk.BACKWARD:
        if not isinstance(store, SessionsWalkedBackward):
            raise Refusal(
                ErrorCode.PAGINATION_CURSOR_INVALID,
                "this deployment does not page backward, so it issued no such cursor",
            )
        here = (position.created_at_ms, position.session_id)
        walked = await store.page_ending_at(tenant_id, here, limit + 1)
        # Oldest-first out of the store. The rows nearest the position are the page --
        # `walked[limit]` is the row just past it, present only when a further page
        # exists -- and the flip into newest-first presentation happens here.
        shown = list(reversed(walked[:limit]))
        # The page after this one begins strictly after this page's oldest row, and the
        # read was inclusive, so that row IS the position. Naming the position rather
        # than `walked[0]` also covers the page whose boundary row has since gone: the
        # key stays a valid place to walk forward from even when nothing sits on it.
        next_page: str | None = Cursor(*here, Walk.FORWARD).encode()
        # Everything the store returned past this page: one row when a further page
        # exists, empty when this is the newest. Sliced rather than indexed so the two
        # cases are one expression rather than a length test and a lookup that have
        # to agree.
        beyond = walked[limit:]
        prev_page = (
            Cursor(beyond[0].created_at_ms, beyond[0].id, Walk.BACKWARD).encode()
            if beyond
            else None
        )
        return SessionPage(
            sessions=[_listed(row) for row in shown],
            next_page=next_page,
            prev_page=prev_page,
        )

    after = None if position is None else (position.created_at_ms, position.session_id)
    rows = await store.page(tenant_id, after, limit + 1)
    shown = list(rows[:limit])
    next_page = (
        Cursor(shown[-1].created_at_ms, shown[-1].id, Walk.FORWARD).encode()
        if len(rows) > limit
        else None
    )
    listed = [_listed(row) for row in shown]
    if not isinstance(store, SessionsWalkedBackward):
        # Left unset so it serialises away entirely. A null here would tell every caller
        # of every page that it was on the first one.
        return SessionPage(sessions=listed, next_page=next_page)
    return SessionPage(
        sessions=listed,
        next_page=next_page,
        # Null and not a token when there is no cursor: with no position the caller is
        # at the newest row, which is the one place this can be said without a read.
        prev_page=(
            None
            if position is None
            else Cursor(
                position.created_at_ms, position.session_id, Walk.BACKWARD
            ).encode()
        ),
    )
