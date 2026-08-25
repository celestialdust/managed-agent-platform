"""The five things a tenant does to a Session: create, read, archive, update, delete.

A read returns state folded from the log rather than a stored column, so what a
tenant reads and what happened cannot disagree. The cost is a range read per state
read, and the recorded answer for a log long enough that this stops being cheap is a
snapshot plus a tail — as a throwaway cache, never a second source.

Who owns a Session is the one fact the log cannot answer: it is keyed by Session and
carries no tenant, so a fold over somebody else's Session succeeds and hands back their
state. Every route therefore goes through the Session registry for ownership — create to
write it, the other four to check it before folding or appending anything.

Three of the five end or change a Session and only two outcomes exist behind them.
`archive` and `delete` reach the one terminal transition this platform has, and `update`
reaches none: nothing a Session was created with can be rewritten, so that route's whole
body is a refusal plus the identity case. Each route's own docstring says what it does
and does not accomplish, because the difference between them is not visible in what they
append.

The returned id is the platform's own. No Agent Runtime identifier appears in this
response or in any other (ADR-007).
"""

from collections.abc import Awaitable, Callable
from typing import Annotated, Final, assert_never

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from managed_agent.control.api.refusals import Refusal, refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.api.routes.files import REASON_FILE_NOT_FOUND
from managed_agent.control.catalog.definitions import (
    AgentReference,
    AgentVersionArchived,
    UnknownAgentVersion,
    resolve_reference,
)
from managed_agent.control.catalog.environments import (
    UnknownEnvironment,
    resolve_environment_revision,
)
from managed_agent.control.files.store import FileId, UploadedFileNotFound
from managed_agent.control.session.lifecycle import (
    ArchiveOutcome,
    ArchiveRefused,
    SessionAlreadyArchived,
    SessionArchived,
    archive_session,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import (
    Seq,
    SessionId,
    TenantId,
    TurnId,
    new_session_id,
)
from managed_agent.core.ports import (
    EventLogRange,
    EventRecord,
    SessionNotVisible,
)
from managed_agent.core.session.projection import project
from managed_agent.core.session.session import (
    CreateSession,
    SessionRecord,
    SessionState,
    UpdateSession,
)
from managed_agent.core.vocabulary import lifecycle

router = APIRouter(tags=["sessions"])

# A range read has to name an upper bound, and the head of a log has no number until it
# is written, so every read to the head has to name one that cannot be reached. Far
# above any sequence a Session can reach and inside the column's signed 64-bit range.
#
# Kept wide open rather than narrowed to one page's worth: a retention sweep raises a
# Session's lowest surviving sequence, so a first window of a few hundred could fall
# entirely below the survivors and read as an empty log.
_UNBOUNDED_END: Seq = 2**62

SESSION_TURN_IN_FLIGHT: Final = ErrorCode.SESSION_TURN_IN_FLIGHT.value
"""A Turn is running, so the Session was not closed. Archive and delete both send it.

**Folded into the published `ErrorCode` set on 2026-08-24.** This constant used to hold
a literal, and its own docstring explained that the code was outside the enum on
purpose: there was no member meaning this, and minting one was a version event belonging
to whoever owned `core/errors.py`. That reason expired when the same change that closed
the set added nine other members -- and until it did, this was the last hand-built
envelope in the tree and the last code reaching a caller that the enum had never heard
of.

Kept as a name, deriving from the member, because it is what tests import to assert the
wire string. It is not the source of truth; the member is.
"""

REASON_GRANT_NOT_REVISABLE: Final = "grant_not_revisable"
REASON_BUDGET_NOT_REVISABLE: Final = "budget_not_revisable"
"""Why the update route refused, named in `detail` beside `ErrorCode.REQUEST_INVALID`.

Two reasons rather than one, because the two fields are permanently refused for
different causes and a caller that wants to know which wall it hit gets a different
answer. They travel in `detail` rather than becoming codes of their own, which is the
shape `files.py` established: the published set is closed, it has no member for either,
and adding one is a version event under ADR-013.
"""


class SessionCreated(BaseModel):
    id: SessionId
    state: SessionState
    seq: Seq


class SessionView(BaseModel):
    id: SessionId
    state: SessionState
    seq: Seq


async def _whole_log(port: EventLogRange, session_id: SessionId) -> list[EventRecord]:
    """Every event of one Session, across as many reads as it takes.

    The port takes a range and promises nothing about how much of it comes back in one
    answer, while the adapter behind it caps a read and treats a short result as "page
    for
    the rest". A caller that needs the whole log therefore has to page, and the state
    fold
    does need the whole log: folding one page reports the state as of that page, which
    for
    a stopped Session five hundred events ago is the word "running" and a resume
    position
    far behind the head — wrong, and silently so.

    Reads from just past the highest sequence seen so far until a page comes back empty.
    That terminates for any page size, one included, because each non-empty page raises
    the cursor above its own highest sequence and a log is finite. Taking the highest
    rather than the last element keeps termination independent of whether the port
    happens to return a page in order.
    """
    events: list[EventRecord] = []
    cursor = 0
    while True:
        page = await port.read(session_id, Seq(cursor + 1), _UNBOUNDED_END)
        if not page:
            return events
        events.extend(page)
        cursor = max(event.seq for event in page)


async def _settle_ownership(
    lookup: Callable[[SessionId, TenantId], Awaitable[SessionRecord]],
    session_id: SessionId,
    tenant_id: TenantId,
) -> None:
    """Raise 404 unless that Session is this tenant's. Returns nothing on success.

    Four routes here need this before they fold or append anything, and the reason is
    the same for all four: the Event Log is keyed by Session and holds no tenant, so a
    fold or a stop aimed at another tenant's id would succeed and act on their Session.
    The registry is the only thing that knows the owner, so it is asked first.

    One 404 covers "no such Session" and "not yours". Two distinguishable answers would
    let a caller holding an id learn from the refusal whether it names somebody else's
    Session, which turns every route here into an existence oracle.

    **It takes the lookup itself rather than the registry, so that every call site
    spells `session_registry.fetch` out loud.** That is not a style preference: the
    scoping guard in `tests/control/test_tenancy.py` reads this module's source and
    fails any route file that names `platform.event_log_range` without also naming
    `session_registry.fetch`, because a route that takes a tenant and never uses it is
    the hole that got written into three plans in a row. A helper holding the port
    behind a parameter name would satisfy that guard once, here, and leave it blind to
    a fifth route added later with no check at all.

    The record it fetches is deliberately dropped rather than returned. Nothing in this
    module reads a creation fact -- the update route refuses without consulting one, and
    the state routes fold the log -- so handing one back would invite a later reader to
    treat this as a lookup and start trusting a field of it as current.
    """
    try:
        await lookup(session_id, tenant_id)
    except SessionNotVisible as invisible:
        raise Refusal(
            ErrorCode.SESSION_NOT_FOUND,
            f"no session {session_id} is visible to this tenant",
        ) from invisible


def _turn_in_flight(session_id: SessionId, turn_id: TurnId, verb: str) -> JSONResponse:
    """Refuse a close because a Turn is open, naming the Turn to interrupt.

    Written once and sent by both closing routes. `verb` is the word the caller used, so
    the sentence names the call that was refused rather than a generic one -- a client
    that sent a delete is not told about archiving.

    Through `refuse` rather than building a response, which is the whole of what changed
    here: this function used to assemble `{code, message, detail}` by hand, and it was
    the one place in the tree still doing so after thirteen others were consolidated. A
    caller of a closing route got a body shaped unlike every other refusal this API
    sends, and no route-level test could see it because the test asserted the shape this
    function produced.
    """
    return refuse(
        ErrorCode.SESSION_TURN_IN_FLIGHT,
        f"a Turn is running on this session; interrupt it before {verb}",
        session_id=str(session_id),
        turn_id=str(turn_id),
    )


def _closed_or_refused(
    session_id: SessionId, outcome: ArchiveOutcome, verb: str
) -> SessionView | JSONResponse:
    """Turn the one terminal transition's outcome into the one answer both verbs give.

    A Session that was already stopped answers exactly as one this call stopped: the
    same body, at the sequence of the stop that is in the log. That is what makes both
    closing routes idempotent -- a client whose first call timed out retries and cannot
    put two stops in one log.

    The match has an `assert_never` tail rather than a trailing else, so mypy --strict
    fails the build if `archive_session` grows a fourth outcome with no arm here. An
    outcome that fell through would be one this function silently reported as closed.
    """
    match outcome:
        case ArchiveRefused(turn_id=running):
            return _turn_in_flight(session_id, running, verb)
        case SessionArchived(seq=seq) | SessionAlreadyArchived(seq=seq):
            return SessionView(id=session_id, state=SessionState.STOPPED, seq=seq)
        case _:
            assert_never(outcome)


@router.post(
    "/sessions",
    status_code=status.HTTP_201_CREATED,
    response_model=SessionCreated,
    responses={
        STATUS_FOR[ErrorCode.DEFINITION_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.DEFINITION_VERSION_ARCHIVED]: {
            "model": PublicErrorEnvelope
        },
        # `environment.not_found` is 404 as well, so it shares the entry above rather
        # than adding one -- two keys of the same status would silently be one key. The
        # same holds for `environment.archived` against the 409 below it.
        STATUS_FOR[ErrorCode.FILE_DELETED]: {"model": PublicErrorEnvelope},
    },
)
async def create(
    body: CreateSession,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SessionCreated | JSONResponse:
    """Create a Session: append the event that makes it exist, then record its owner.

    The state returned is RUNNING because that is what the event just appended folds to
    — a Session is ready for a Turn the moment it is created.

    Two writes, and their order is the answer to what a failure between them leaves
    behind. The event is what makes the Session exist and what every state read folds;
    the registry row is what makes it findable and says which tenant owns it. Appending
    first means a crash in between leaves a Session nobody can list or read — inert, and
    invisible in the direction that matters. The other order leaves one that lists and
    reads as though it were real while its log is empty, which is worse: a caller can
    reach it.

    The two cannot drift. Both are written from one parsed body in one request, and both
    stores refuse an update, so there is no later moment at which either could change.

    The Grant is sorted before it is written. It arrives as a set, which has no
    order, and an event payload is read back verbatim by everything downstream — so
    writing it unsorted would make the same request produce different bytes on
    different runs.

    The definition is resolved **before** the append, and the revision it resolved to
    goes into the payload. Two reasons, and the ordering matters as much as the value.
    The revision is written down because a definition registered later must not change
    what this Session already is — resolving again on every Turn would mean exactly
    that. And it is resolved first because a Session whose creation event exists and
    whose definition does not is the worst of both: addressable, folding to RUNNING, and
    unable to run. An unresolvable definition therefore appends nothing at all.

    Which revision that is depends on whether the caller pinned one. With no pin the
    newest revision that has not been withdrawn is taken, so an edit reaches the next
    Session; with a pin, exactly that revision, and a refusal if it was withdrawn
    rather than a quiet slide onto the revision below it. Once resolved, the number is
    written down and never consulted again for this Session — which is the whole of why
    retiring a revision stops a new Session and does not touch a running one.

    The environment is resolved **before** the definition and so before both writes, for
    the same reason and one more: it is the shape the pod is started from, and a Session
    whose creation event exists while the shape it names does not is addressable, folds
    to RUNNING, and can never be placed. An unregistered environment therefore appends
    nothing, writes no registry row, and places no pod.

    The environment's **revision** is written into the payload beside the definition's,
    and for the identical reason: an Environment can now be edited, so an id alone
    stopped saying which shape a Session runs in. Without the number, two Sessions
    naming one id across an edit would be running in different sandboxes with nothing
    anywhere recording that they differed — which is precisely the invariant the
    Environment table's append-only trigger was installed to protect. Resolving takes
    the newest revision, that number is written down, and the shape this Session is
    entitled to is fixed from here on.

    An archived environment refuses. Retirement is terminal and its whole content is
    "no new Session in this shape", so the refusal belongs at the one route that starts
    one. A Session already running in a retired Environment is untouched — it resolved
    its revision when it was created and never asks again.

    Four refusals reach a caller from here and they are different acts, so they carry
    different codes: an environment id that resolves to nothing, an environment that
    resolved and was retired, a definition id or version number that resolves to
    nothing, and a version that resolved and was deliberately withdrawn. A malformed
    version, or a body with no environment id at all, is none of them — those never
    reach this function, because `CreateSession` refuses them and the framework's
    rejection is answered as 400 naming the field.
    """
    platform = platform_from_request(request)
    try:
        # Resolved before anything else this handler does, and the ordering is the
        # guarantee: a refusal raised after the append or after the registry write would
        # leave a Session behind that names a shape the platform cannot produce.
        #
        # What is consumed here is the revision, which goes into the payload below. The
        # shape itself is not -- this route compiles no configuration and places no pod
        # -- so when a compilation call arrives here it takes `.environment` off this
        # same value rather than resolving a second time.
        environment = await resolve_environment_revision(
            platform.environment_store, body.environment_id, tenant_id
        )
    except UnknownEnvironment:
        return refuse(
            ErrorCode.ENVIRONMENT_NOT_FOUND,
            "no environment with that id is registered",
        )
    if environment.archived:
        return refuse(
            ErrorCode.ENVIRONMENT_ARCHIVED,
            "that environment was retired, so no new Session may be started in it",
            environment_id=str(body.environment_id),
        )
    reference = AgentReference(body.definition_id, body.definition_version)
    try:
        resolved = await resolve_reference(
            platform.definition_registry, tenant_id, reference
        )
    except UnknownAgentVersion:
        return refuse(
            ErrorCode.DEFINITION_NOT_FOUND,
            "no such agent definition version is registered to this tenant",
            definition_id=str(body.definition_id),
        )
    except AgentVersionArchived as archived:
        return refuse(
            ErrorCode.DEFINITION_VERSION_ARCHIVED,
            "that version of the agent definition was retired and starts no Session",
            definition_id=str(body.definition_id),
            version=archived.revision,
        )
    # Checked here, where the caller is still on the connection, and the answer is
    # permanent: an uploaded file's row is never rewritten and the object behind it is
    # never written twice, so an id that resolves now resolves when the pod is placed.
    # The alternative is a Session that is accepted and then fails its first Turn as
    # undeliverable -- a refusal that names the wrong thing, arriving whenever the
    # tenant next tried to use what they thought they had created.
    #
    # `describe` and not `fetch`: what is in question is whether the row is there, and
    # the bytes are not needed until a pod exists. Fetching would download every
    # attached file to answer a question about their ids.
    for attached in body.file_ids:
        try:
            await platform.file_store.describe(
                tenant_id=tenant_id, file_id=FileId(attached)
            )
        except UploadedFileNotFound:
            return refuse(
                ErrorCode.REQUEST_INVALID,
                "no file with that identifier is readable by this tenant, so this "
                "Session would start with the document missing",
                reason=REASON_FILE_NOT_FOUND,
                file_id=str(attached),
            )
        # A second question, because `describe` cannot answer this one. A deleted
        # file keeps its row -- a Session's creation event names the ids it was
        # created with, so dropping the row would leave that history pointing at
        # nothing -- and `describe` answers about the row. So the id above resolves
        # and the bytes behind it are gone.
        #
        # 410 rather than the 400 above, and the difference is what the caller does
        # next. "No such file" invites them to check the id they sent; "this file
        # was deleted" tells them the id was right and the thing is gone, so
        # re-sending it will never work.
        if await platform.file_store.deletion_recorded(file_id=FileId(attached)):
            return refuse(
                ErrorCode.FILE_DELETED,
                "that file was deleted, so this Session would start with the "
                "document missing",
                file_id=str(attached),
            )
    session_id = new_session_id()
    seq = await platform.event_log_append.append(
        session_id,
        lifecycle.SESSION_CREATED,
        {
            "definition_id": str(body.definition_id),
            "definition_revision": resolved.revision,
            "environment_id": str(body.environment_id),
            "environment_revision": environment.revision,
            "file_ids": [str(one) for one in body.file_ids],
            "grant": sorted(body.grant),
            "scope": body.scope,
            "budget_minor_units": body.budget_minor_units,
            "budget_currency": body.budget_currency,
            "retention_days": body.retention_days,
        },
    )
    await platform.session_registry.create(
        SessionRecord(
            id=session_id,
            tenant_id=tenant_id,
            definition_id=body.definition_id,
            definition_revision=str(resolved.revision),
            grant=body.grant,
            scope=tuple(sorted(body.scope.items())),
            budget_minor_units=body.budget_minor_units,
            budget_currency=body.budget_currency,
            retention_days=body.retention_days,
        )
    )
    return SessionCreated(id=session_id, state=SessionState.RUNNING, seq=seq)


@router.get("/sessions/{session_id}")
async def read(
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SessionView:
    """Read a Session by folding its whole log forward.

    The whole log, from sequence 1, because the fold has no starting state to resume
    from — that is the cost of keeping no stored copy, and it is deliberate. The
    sequence returned is the last one read, which is what a caller resumes a stream
    from.

    Reading the whole log takes as many round trips as the log has pages, which is the
    part of this that will not scale. The recorded answer when it stops fitting the
    interactive budget is a snapshot plus a tail, kept as a throwaway cache and never as
    a second source.

    Who owns the Session is settled **before** the fold, and the fold is not what
    settles it — the Event Log is keyed by Session and holds no tenant, so folding a
    Session belonging to somebody else would succeed and return that Session's state.
    The registry is the only thing here that knows the owner, so it is asked first and a
    Session it will not show this tenant is never read at all.

    A Session that is not this tenant's is **refused**, not answered with an empty or
    default-shaped state. One 404 covers both "no such Session" and "not yours": two
    distinguishable answers would let a caller holding an id learn from the refusal
    whether it names another tenant's Session.
    """
    platform = platform_from_request(request)
    await _settle_ownership(platform.session_registry.fetch, session_id, tenant_id)
    events = await _whole_log(platform.event_log_range, session_id)
    state, seq = project(events)
    return SessionView(id=session_id, state=state, seq=seq)


@router.post("/sessions/{session_id}/archive", response_model=SessionView)
async def archive(
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SessionView | JSONResponse:
    """Archive a Session: accept no further event, keep its history, hand its pod back.

    Two things happen and only one of them is this route's promise. The promise is that
    the Session stops accepting events, which is the `session.stopped` append and is
    what makes a later Turn refuse with the state rather than run. Handing the pod back
    is the consequence that matters operationally -- the cluster holds roughly
    forty-five Session pods and nothing else in this tree ever gave one up -- and it is
    deliberately *not* part of what a 200 here claims: a Turn still open across the
    append leaves the pod for the sweep to reclaim, and the archive is complete anyway.

    History is preserved and is not deleted. `delete` below reaches this same transition
    for the same reason: removing a Session's history is not an operation this platform
    has, so archiving is the whole of what either verb accomplishes. That route's
    docstring carries the reasoning and the list of what survives, and a reader
    comparing the two should start there: they are told apart by what they promise a
    caller rather than by what they append.

    Idempotent. A second call appends nothing and answers with the same view, so a
    client that retried a timed-out archive cannot put two stops in one log.

    Refused while a Turn is running, and that refusal is folded out of the Turn events
    rather than read off the Session's state -- `RUNNING` here means "would accept a
    Turn", so refusing on it would refuse every Session that was archivable. The caller
    interrupts the Turn and archives after it, which is the order the API this mirrors
    requires for the same reason.

    The refusal's code is **not** in the published closed set. Adding a member is a
    version event under ADR-013 and `core/errors.py` is not this slice's to edit, so it
    is spelled as a module constant here and inventoried in
    `tests/control/test_closed_error_set.py` -- the same path `session_list.py` took for
    its paging refusal, and the one that test's own failure message asks for. Folding it
    in is one edit by whoever owns that file.

    Ownership is settled before anything is folded or appended, for the reason the read
    route gives: the Event Log is keyed by Session and holds no tenant, so archiving
    another tenant's Session would succeed and stop it. One 404 covers "no such Session"
    and "not yours", so a caller holding an id cannot learn which it is.
    """
    platform = platform_from_request(request)
    await _settle_ownership(platform.session_registry.fetch, session_id, tenant_id)
    outcome = await archive_session(
        session_id,
        platform.event_log_append,
        platform.event_log_range,
        platform.session_pod_release,
    )
    return _closed_or_refused(session_id, outcome, "archiving")


@router.post(
    "/sessions/{session_id}",
    response_model=SessionView,
    responses={STATUS_FOR[ErrorCode.REQUEST_INVALID]: {"model": PublicErrorEnvelope}},
)
async def update(
    body: UpdateSession,
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SessionView | JSONResponse:
    """Change nothing about a Session, and say precisely why each field cannot change.

    **Nothing a Session was created with is revisable on this platform, so this route
    has no success path that changes anything.** A body naming a field is refused; an
    empty body is answered with the Session's current state, which is the identity ask
    and the only one that can succeed. The route exists so that a caller reaching for
    the update verb gets a coded refusal naming the field and the reason, instead of a
    405 reading like a routing accident or a 200 that changed nothing without saying so.

    The Grant is refused for two independent reasons, and either alone would settle it.
    A revision would have to be a `session.updated` event, because the Session's own row
    refuses UPDATE in the store itself -- the trigger raises rather than absorbing the
    write -- and that event type is not in the published closed vocabulary, where adding
    one is a version change (ADR-013). And a revised Grant would take effect nowhere
    even if it were recorded: the Grant is enforced at the Tool Gateway (ADR-014), and
    nothing in this tree reads `SessionRecord.grant` at all today, so the field is
    written at creation and consulted by no comparison. A route that accepted a revision
    would be promising an enforcement change that no component performs.

    The Budget is refused for the same store reason and one of its own: no component
    measures a Session's spend against `budget_minor_units`, so raising or lowering a
    ceiling nothing compares against would change a stored number and no behaviour.

    A title, a metadata map and a vault list -- the rest of what the surface this
    platform mirrors lets a caller change -- are refused by `UpdateSession`'s
    `extra="forbid"` as a 400 naming the field, and that is the honest answer rather
    than a coded one. There is no title column, no metadata column and no vault anywhere
    in this platform, so there is no concept here to refuse and nothing to say about it
    beyond "no such field".

    **The field refusals are checked before the Session's state, and the order is a
    decision.** "This field is not revisable" is a property of the platform and holds in
    every state, so reporting the state first would send a caller to un-archive and
    retry a call that would then fail for the permanent reason anyway. The two fields
    are checked in a fixed order so that a body naming both gets a stable answer.

    Ownership is settled first all the same, before either check, so that a caller
    holding a stranger's id learns 404 rather than learning from a field refusal that
    the Session exists.
    """
    platform = platform_from_request(request)
    await _settle_ownership(platform.session_registry.fetch, session_id, tenant_id)
    if body.grant is not None:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            "a Session's Grant is fixed at creation: revising it would need an event "
            "type the published vocabulary does not carry, and nothing here reads the "
            "Grant, so the revision would take effect nowhere",
            reason=REASON_GRANT_NOT_REVISABLE,
            session_id=str(session_id),
        )
    if body.budget_minor_units is not None:
        return refuse(
            ErrorCode.REQUEST_INVALID,
            "a Session's Budget is fixed at creation: the row refuses an update in the "
            "store itself, and nothing here measures spend against the ceiling, so a "
            "new one would change a stored number and no behaviour",
            reason=REASON_BUDGET_NOT_REVISABLE,
            session_id=str(session_id),
        )
    events = await _whole_log(platform.event_log_range, session_id)
    state, seq = project(events)
    return SessionView(id=session_id, state=state, seq=seq)


@router.delete("/sessions/{session_id}", response_model=SessionView)
async def delete(
    session_id: SessionId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SessionView | JSONResponse:
    """Close a Session for good: no further event, no pod. **The history survives.**

    What a caller gets from this and cannot undo: the Session accepts no further event,
    so it will never run another Turn, and its pod is handed back -- with it goes every
    byte of the sandbox that was not synced out at the last completed Turn (ADR-004).
    Both are permanent; nothing in this platform starts a stopped Session again.

    What a caller does **not** get, and this is the whole reason to read this before
    reaching for the verb: their data is not gone. After this answers 200, the Session's
    whole Event Log is still readable at the same sequence numbers, `GET
    /v1/sessions/{id}` still answers, and the Session still appears in `GET
    /v1/sessions`. What the Session captured -- its Evidence, its Rollout, its VFS -- is
    still held.

    That is not a shortcut taken here; it is what this platform can do. Removing a
    Session means removing its record, its events and the state its sandbox held, and no
    port in this tree can do any of the three: the Session registry offers `create`,
    `fetch` and `page` and no delete; neither Event Log port removes a row; and the
    Rollout store has no delete on purpose, because expiry is owned by the retention
    sweep and a second remover could take a Rollout away while the sweep still believed
    the Session was inside its window. That retention sweep is the one sanctioned
    removal path in this platform and it is not built -- so a `DELETE` claiming erasure
    would be a false statement to a tenant about their own data, which is worse than a
    verb that says exactly what it did.

    What this **does** contribute to erasure is the part that is not a no-op. A Session
    that still accepts events has a log whose newest row keeps moving forward, so a
    retention window measured from a row's age never closes over the whole of it.
    Closing the Session fixes the last row, and from then on its history has an expiry
    that arrives. That is the sense in which this is a delete on an append-only log: it
    cannot remove the history, but it is what lets retention eventually do so.

    Reaching the same transition as `archive` is therefore deliberate rather than an
    unfinished shortcut, and the log cannot tell the two calls apart -- both leave one
    `session.stopped` carrying the `archived` reason. Recording the difference would
    mean a `session.deleted` type in the published closed vocabulary, which ADR-013
    makes a version change, and that vocabulary is not this slice's to widen. Until it
    is, the two verbs differ in what they promise and not in what they write.

    Idempotent, and a second call is not an error. It appends nothing, answers with the
    same view at the same sequence, and still hands the pod back -- which is the value
    of retrying a call that died between its append and its handback.

    Refused while a Turn is running, with the Turn named so the caller knows what to
    interrupt. Nothing is appended and the pod is kept on that path: a Turn a tenant is
    waiting on does not lose its pod to a close, however the close was spelled.
    """
    platform = platform_from_request(request)
    await _settle_ownership(platform.session_registry.fetch, session_id, tenant_id)
    # The same transition `archive` reaches, called by its own name rather than through
    # a `delete_session` that would forward to it. A wrapper would suggest the two verbs
    # diverge somewhere below this line, and they do not.
    outcome = await archive_session(
        session_id,
        platform.event_log_append,
        platform.event_log_range,
        platform.session_pod_release,
    )
    return _closed_or_refused(session_id, outcome, "deleting")
