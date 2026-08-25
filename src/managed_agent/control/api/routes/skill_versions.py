"""The five version routes under /v1/skills/{id}, and the lookup the six of them share.

**Two absences, and the second was already producing wrong behaviour.** A skill that
regresses could not be reverted, because `POST /v1/skills` writes one immutable row and
there was nothing to go back to. And a skill was a single body, so it could not carry
the files it names: Anthropic's published `pdf` skill tells the model to read `forms.md`
and `reference.md`, and uploaded through that door those references point at files that
do not exist while the model is still instructed to consult them.
`tests/fixtures/anthropic_pdf_skill.py` records that document verbatim along with the
six distributions it names.

**A version is a microsecond timestamp the platform mints.** Published as a string, held
as the integer it is -- `migrations/versions/0023_skill_versions.py` records why the
column is not text. Two versions of one skill minted inside the same microsecond collide
on the key, and the store raises rather than overwriting; `create_skill_version` mints
the next microsecond and writes again. Nothing here accepts a caller-supplied version:
there is no way to ask for one, which is what keeps the value meaning the moment it was
written.

**What a version's content is and where its `directory` comes from are decided in
`skill_bundles.py`**, not here. What matters at this level is that the caller chooses
neither: `name`, `description` and `directory` are all read out of the uploaded bundle,
so the record and the document cannot disagree about what the skill says it does or
where the model will look for its files.

**A retirement is a tombstone and the version stays readable.** `GET .../versions/{v}`
answers 200 for a retired version and says so; only resolution into a new Session
refuses. A version can be pinned -- an agent definition carries a digest over the set it
resolved -- so deleting the row would leave that history naming something that resolves
to nothing, which reads as the platform having lost the data rather than as the tenant
having retired it. The listing excludes retired versions, because it answers "what can
this skill still be resolved to".

**The newest version may be retired**, and `latest_version` then names the newest
survivor. A regression is exactly the case retirement exists for, so forbidding it would
disable the feature in its motivating case; retire every version and the skill is
readable, has a null `latest_version`, and resolves to nothing.
"""

from __future__ import annotations

import re
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import Annotated, Any, Final, Literal

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile
from fastapi import status as http_status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.refusals import Refusal, refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.api.routes.skill_bundles import (
    SkillBundle,
    archive_of,
    parse_bundle,
)
from managed_agent.control.skills.inventory import SkillInventory
from managed_agent.control.skills.registry import (
    SkillHeld,
    SkillStore,
    SkillVersionCollision,
    SkillVersionRecord,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import SkillId, TenantId

version_routes = APIRouter(tags=["skills"])
"""The five routes below, for `skills.py` to fold into the router `create_app` mounts.

**Deliberately not named `router`, and the name is the documentation.** A module-level
`router` under `control/api/` is this repository's marker for a mount point:
`tests/control/test_every_router_is_mounted.py` reads the name out of the source
and demands an
`include_router` for it inside `create_app`. This module is not a mount point -- it is a
fragment of the `/v1/skills` surface, folded in by `skills.py` so that prefix has one
router -- and a `router` here would be a name that guard is right to refuse.
"""

ECHOED_VERSION_CHARS: Final[int] = 64
"""How much of an unusable version string a refusal quotes back.

Bounded because the value is the caller's: an unbounded echo puts a
caller-controlled string of any length into a response body and into every log
that holds one. Sixty-four is long enough to see the mistake in.
"""

REASON_VERSION_UNMINTABLE: Final[str] = "version_unmintable"
"""Named in `detail` for the one fault here that is the platform's, not the caller's:
every microsecond this route was willing to try was already taken.

The message says which of the two causes it was, following the precedent
`REASON_STORAGE_UNCONFIGURED` sets in `files.py`: a named internal fault publishes what
it is, and only an *unhandled* exception falls back to the generic sentence. Naming it
is what turns "it broke" into "the clock stopped" for whoever reads the response, and it
reveals nothing a stack detail would -- the two causes are the only two there are.

500 rather than 529, and that is the judgement rather than a default. 529 tells a caller
to retry, and no legitimate request rate can produce this: they would have to be writing
versions faster than one per microsecond. With a working clock the condition is
essentially impossible, which leaves a clock that is not advancing -- and that does not
fix itself on a retry a caller was invited to make.
"""

REASON_CURSOR_INVALID: Final[str] = "cursor_invalid"
"""Spelled the same way `skills.py` spells it, for the same reason: the published code
set has no paging family, so the branchable half of a bad cursor lives in `detail`."""

DEFAULT_VERSION_PAGE_SIZE: Final[int] = 20
MAX_VERSION_PAGE_SIZE: Final[int] = 1000
"""How many versions one page may hold, published to match the reference's own bound.

Wider than the skill listing's 100, and the cost is real rather than assumed: a version
row carries no body but does carry a description, which is capped at 1024 characters, so
a thousand rows is a response of roughly a megabyte. That is the price of the published
number. A caller walking the list should ask for the default and follow `next_page`.
"""

MINT_ATTEMPTS: Final[int] = 16
"""How many microseconds `create_skill_version` will walk forward through a collision.

Sixteen consecutive occupied microseconds means sixteen versions of one skill were
written inside sixteen microseconds, or that the clock is not advancing. Neither is a
refusal a caller can act on, so the loop gives up rather than spinning: an unbounded
retry inside a request handler is a handler that never answers.
"""

MIN_VERSION: Final[int] = 1_000_000_000_000_000
"""The smallest version the store will hold -- where sixteen digits begin, which in
microseconds is 2001. What it actually excludes is a *millisecond* timestamp written by
mistake: three digits shorter, so it lands in 1970 and would sort before every real
version instead of failing."""

MAX_VERSION: Final[int] = 2**63 - 1
"""What the column holds. A longer run of digits is refused here rather than sent to
Postgres, which would answer with a driver error a caller cannot read."""

_VERSION_DIGITS: Final = re.compile(r"^[0-9]{16,19}$")
"""What a version looks like on the wire. Anchored digits rather than `str.isdigit`,
which is true of superscripts and other numerals `int()` then refuses."""

_UPLOADED_BUNDLE: Final = "the uploaded bundle"
"""What a refusal calls a document that arrived with no path we could name."""


def mint_version() -> int:
    """The current moment as microseconds since the epoch.

    A function rather than an expression inside the route so a test can hold the clock
    still and force the collision the store is built to refuse. `time.time_ns` rather
    than `time.time`, because a float loses microsecond resolution somewhere in the
    2030s and the value being minted is a key.
    """
    return time.time_ns() // 1_000


def _parsed_version(text: str) -> int | None:
    """The version this string names, or None when no version could be spelled that way.

    One place, because two callers ask the same question about the same string -- a path
    segment and a cursor's payload -- and two answers could disagree about whether a
    nineteen-digit value is a version.
    """
    if not _VERSION_DIGITS.match(text):
        return None
    found = int(text)
    if found < MIN_VERSION or found > MAX_VERSION:
        return None
    return found


def version_from_path(version: str) -> int:
    """The version named in the path, or a refusal saying none is spelled that way.

    A dependency rather than a check inside each of the three routes taking a version,
    because a dependency cannot return a response and this is exactly what `Refusal`
    exists for: a caller cannot tell whether the refusal was decided before the handler
    or inside it, and should not be able to.

    404 and not 400. A caller asking for `/versions/latest` is asking for a version that
    does not exist rather than sending a malformed request -- no version has that
    name and none ever can -- and `latest` is the plausible guess, since that is the
    word `SkillAttachment.version` accepts.
    """
    found = _parsed_version(version)
    if found is None:
        raise Refusal(
            ErrorCode.SKILL_VERSION_NOT_FOUND,
            "a version is the Unix epoch timestamp in microseconds that the platform "
            "minted for it, so no version is named by this string; read them from "
            "GET /v1/skills/{id}/versions rather than constructing one",
            version=version[:ECHOED_VERSION_CHARS],
        )
    return found


def version_cursor(version: int) -> str:
    """A page position as a token, base64url with its padding stripped.

    Encoded rather than sent as the bare version so a caller follows `next_page` instead
    of assembling one, which is what lets the boundary change without breaking a walk.
    Padding is stripped because a `=` is percent-encoded in a query string and comes
    back looking unlike what was issued.
    """
    return urlsafe_b64encode(str(version).encode()).decode().rstrip("=")


def version_at(token: str) -> int | None:
    """The position a token names, or None when it is not a token this surface issued.

    Nothing partial: a token that decodes to something that is not a version names no
    row, and reading it as the start of the collection would restart the walk at the top
    without saying so.
    """
    try:
        padded = token + "=" * (-len(token) % 4)
        text = urlsafe_b64decode(padded.encode()).decode()
    except ValueError:
        # binascii.Error and UnicodeDecodeError are both ValueError, so one clause
        # covers bad base64 and bad utf-8.
        return None
    return _parsed_version(text)


class SkillVersionView(BaseModel):
    """One version on the wire: which one, what its document says, and where it went.

    `version` is a string carrying a decimal integer, which is the reference's own shape
    and worth keeping even though the value is numeric: a client that parsed it into a
    64-bit int would be right today and wrong the day the field carries anything else,
    and nothing about it is arithmetic.

    No creation timestamp. The version is one -- it is the microsecond it was written --
    so a second field holding that moment would be a second record of one fact.

    `retired` is on every surface that returns this shape, including the listing, where
    it reads false on every row because the listing excludes retired versions. One shape
    for one resource: a client parses a version the same way wherever it read it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    name: str
    description: str
    directory: str
    retired: bool


class SkillVersionPage(BaseModel):
    """One page of a skill's live versions, and where the page after it starts.

    `data` rather than a named collection, because that is the key this collection is
    published under. `has_more` is required and not optional, which is the one place the
    version listing's shape differs from the file listing's -- copying the optional form
    would let a consumer treat an absent field as "no more" and stop a walk early.

    The two cannot disagree: both come from the one extra row the store was asked for,
    so `next_page` is null exactly when `has_more` is false.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: list[SkillVersionView]
    next_page: str | None
    has_more: bool


class SkillVersionDeleted(BaseModel):
    """What a retirement answers: the version that was retired, and what happened.

    **`id` carries the version string and not the skill's id**, which looks like a
    mistake and is the reference's own shape -- its Returns block redefines `id` on
    this one response as "Version identifier for the skill". Copied rather than
    corrected, because a client generated from that document reads this field expecting
    the version, and a platform that helpfully returned the skill id would break it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: Literal["skill_version_deleted"] = "skill_version_deleted"


async def skill_or_refusal(
    store: SkillStore,
    inventory: SkillInventory,
    tenant_id: TenantId,
    skill_id: SkillId,
) -> SkillHeld | JSONResponse:
    """The uploaded skill, or the refusal standing in: 409 a commit's, 410 deleted, 404
    unknown.

    Three refusals rather than one, because each says something different about the id
    and a caller acts differently on each.

    **409 for a repository skill, and this is the one refusal about a skill that
    exists.** Every route below writes or retires a version, and a repository skill's
    version *is* the commit it was submitted from -- a version row written against one
    would be a body this platform minted sitting under an id whose content is fixed by a
    commit, and resolution would then have two answers to what the skill says. So the id
    resolves, the skill is real, and the operation does not apply to it. 404 would say
    the caller is wrong about the id when they are not; 400 would say the request is
    malformed when it is well-formed and merely aimed at the wrong kind of skill.

    A deleted skill's id is one the platform *did* hold and deliberately stopped
    serving, and a tenant who honoured a deletion request needs to see that it happened
    rather than be told to look again. An unknown id is the caller being wrong about it,
    and that answer invites them to check it.

    Another tenant's skill is unknown rather than forbidden, at both doors. The tenant
    is a term in each store's own query, so there is no point in this function where a
    cross-tenant row exists to be refused. That matters more for the repository half
    than the uploaded one: a repository id is a `uuid5` of its four key columns, so
    anyone who knows the repository, the revision and the name can derive another
    tenant's id without ever having seen it, and the answer must reveal nothing about
    whether the row is there.

    **The inventory is consulted only when the uploaded half misses.** A repository id
    and an uploaded id cannot be the same value, so the order costs nothing in
    correctness; it saves a second query on every request that names an uploaded skill,
    which is all five routes' normal case.

    Lives here rather than in `skills.py` because all its callers are the routes below.
    `read_skill` in `skills.py` deliberately does *not* use it: that route can serve a
    repository skill, so it must not inherit the 409.
    """
    held = await store.read_skill(tenant_id, skill_id)
    if held is None:
        from_commit = await inventory.repository_skill_at(tenant_id, skill_id)
        if from_commit is not None:
            return refuse(
                ErrorCode.SKILL_OWNED_BY_COMMIT,
                "this skill came from a repository checkout, so its content is fixed "
                f"by revision {from_commit.revision} of {from_commit.repository} and "
                "this platform neither versions nor deletes it. Stop using it by not "
                "pinning that revision, or submit a new commit",
                skill_id=str(skill_id),
                repository=from_commit.repository,
                revision=from_commit.revision,
            )
        return refuse(
            ErrorCode.SKILL_NOT_FOUND,
            "no skill with that id is registered to this tenant",
            skill_id=str(skill_id),
        )
    if held.deleted:
        return refuse(
            ErrorCode.SKILL_DELETED,
            "this skill was deleted; its id is kept so a Session's history can still "
            "name what it ran, and nothing new resolves through it",
            skill_id=str(skill_id),
        )
    return held


async def write_version(
    store: SkillStore,
    tenant_id: TenantId,
    skill_id: SkillId,
    bundle: SkillBundle,
) -> int | JSONResponse:
    """Mint a version for this bundle and write it, or return the refusal for why not.

    One function because three doors now write a version and all three have to mint it
    the same way: `POST /v1/skills/{id}/versions`, and both shapes of `POST /v1/skills`,
    which mints version one so a freshly created skill resolves to something. A second
    copy of this loop would be a second answer to "what happens when two versions land
    in one microsecond", and the two would be free to diverge on the one input nobody
    exercises by hand.

    The version cannot be asked for. Two minted inside one microsecond collide on the
    store's key and the store raises rather than overwriting -- overwriting would
    rewrite a version something may already have resolved, and skipping would report a
    version that was never written -- so this walks forward a microsecond at a time. The
    value stays a true timestamp to within the number of collisions, which is the trade
    a sequence would have avoided at the cost of the published contract: the value has
    to be a timestamp because that is what their clients read.

    Returns the refusal rather than raising it, because both callers are route handlers
    that already return a `JSONResponse` on their other refusal paths and neither is a
    dependency. The caller checks the type; there is no path on which a version was
    written and a refusal came back.
    """
    version = mint_version()
    for _ in range(MINT_ATTEMPTS):
        try:
            await store.add_skill_version(
                tenant_id,
                skill_id,
                version,
                bundle.skill,
                bundle.directory,
                bundle.files,
            )
            return version
        except SkillVersionCollision:
            version += 1
    # Enveloped rather than raised, and `platform.internal` rather than a refusal of
    # the request: the caller's bundle was fine and there is nothing for them to
    # change. What this says is that the clock is not advancing, and the request id in
    # the envelope is what turns that into a log lookup.
    return refuse(
        ErrorCode.INTERNAL,
        f"{MINT_ATTEMPTS} consecutive microseconds are already taken by versions "
        f"of skill {skill_id}, so either that many were written in that many "
        "microseconds or this platform's clock is not advancing",
        reason=REASON_VERSION_UNMINTABLE,
        attempts=MINT_ATTEMPTS,
    )


def _view_of(record: SkillVersionRecord) -> SkillVersionView:
    """One stored version as the wire shape, with the version rendered as its string."""
    return SkillVersionView(
        version=str(record.version),
        name=record.name,
        description=record.description,
        directory=record.directory,
        retired=record.retired,
    )


_SKILL_REFUSALS: Final[dict[int | str, dict[str, Any]]] = {
    STATUS_FOR[ErrorCode.SKILL_NOT_FOUND]: {"model": PublicErrorEnvelope},
    STATUS_FOR[ErrorCode.SKILL_DELETED]: {"model": PublicErrorEnvelope},
    STATUS_FOR[ErrorCode.SKILL_OWNED_BY_COMMIT]: {"model": PublicErrorEnvelope},
}
"""What every route here can answer besides its own. `SKILL_VERSION_NOT_FOUND` shares a
status with `SKILL_NOT_FOUND`, so the three version reads declare nothing extra;
`SKILL_OWNED_BY_COMMIT` is here because the shared lookup answers it for all five."""


@version_routes.post(
    "/skills/{skill_id:uuid}/versions",
    status_code=http_status.HTTP_201_CREATED,
    response_model=SkillVersionView,
    responses={
        **_SKILL_REFUSALS,
        STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope},
    },
)
async def create_skill_version(
    skill_id: SkillId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    files: Annotated[list[UploadFile], File()],
) -> SkillVersionView | JSONResponse:
    """Write a new version of this skill from an uploaded bundle, and name it.

    The version is minted here and cannot be asked for. Two of them minted inside one
    microsecond collide on the store's key, and the store raises rather than overwriting
    -- overwriting would rewrite a version something may already have resolved, and
    skipping would answer 201 for a version that was never written. So this walks
    forward a microsecond at a time. The value stays a true timestamp to within the
    number of collisions, which is the trade a sequence would have avoided at the cost
    of the published contract: the value has to be a timestamp because that is what
    their clients read.

    A deleted skill refuses. Writing a version of a skill whose id no longer resolves
    would leave a row nothing can reach, and the tenant who deleted it has been told the
    skill is gone.

    The uploaded document decides the version's `name`, so a version may rename the
    skill. That is not guarded against here, because the delivery path already refuses
    two skills resolving to one name at the door of the Session -- which is where the
    collision is decidable and where the refusal can name both sides of it.
    """
    platform = platform_from_request(request)
    store = platform.skill_store
    held = await skill_or_refusal(store, platform.skill_inventory, tenant_id, skill_id)
    if isinstance(held, JSONResponse):
        return held
    bundle = await parse_bundle(files)
    written = await write_version(store, tenant_id, skill_id, bundle)
    if isinstance(written, JSONResponse):
        return written
    version = written
    return SkillVersionView(
        version=str(version),
        name=bundle.skill.name,
        description=bundle.skill.description,
        directory=bundle.directory,
        retired=False,
    )


@version_routes.get(
    "/skills/{skill_id:uuid}/versions",
    response_model=SkillVersionPage,
    responses={
        **_SKILL_REFUSALS,
        STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]: {"model": PublicErrorEnvelope},
    },
)
async def list_skill_versions(
    skill_id: SkillId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    page: str | None = None,
    limit: Annotated[
        int, Query(ge=1, le=MAX_VERSION_PAGE_SIZE)
    ] = DEFAULT_VERSION_PAGE_SIZE,
) -> SkillVersionPage | JSONResponse:
    """One page of this skill's live versions, newest first.

    Retired versions are absent and there is no parameter to ask for them. The listing
    answers "what can this skill still be resolved to", and a retirement is the
    statement that this one cannot; a retired version is still readable one at a time,
    by the version a pinned definition already names.

    The store is asked for one row more than will be returned. That extra row is the
    whole answer to "is there another page", which is why `next_page` is null rather
    than a token leading somewhere empty.

    A cursor this surface did not issue is refused rather than read as the start of the
    collection. Starting over would hand a caller the first page again, which reads as
    the walk having looped rather than failed.
    """
    platform = platform_from_request(request)
    store = platform.skill_store
    held = await skill_or_refusal(store, platform.skill_inventory, tenant_id, skill_id)
    if isinstance(held, JSONResponse):
        return held
    after: int | None = None
    if page is not None:
        after = version_at(page)
        if after is None:
            return refuse(
                ErrorCode.PAGINATION_CURSOR_INVALID,
                "cursor was not issued by this surface; walk the listing from the "
                "beginning by omitting it rather than by constructing one",
                reason=REASON_CURSOR_INVALID,
            )
    rows = await store.page_skill_versions(tenant_id, skill_id, after, limit + 1)
    shown = rows[:limit]
    more = len(rows) > limit
    return SkillVersionPage(
        data=[_view_of(row) for row in shown],
        next_page=version_cursor(shown[-1].version) if more else None,
        has_more=more,
    )


@version_routes.get(
    "/skills/{skill_id:uuid}/versions/{version}",
    response_model=SkillVersionView,
    responses=_SKILL_REFUSALS,
)
async def read_skill_version(
    skill_id: SkillId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    version: Annotated[int, Depends(version_from_path)],
) -> SkillVersionView | JSONResponse:
    """One version, whether or not it has been retired, with the retirement reported.

    A retired version answers 200 rather than 410, and that is the difference between a
    retirement and a deletion. A version can be pinned -- an agent definition carries a
    digest over the set it resolved, and a Session created under that definition has a
    history naming it -- so the record has to stay readable or that history stops being
    readable with it. What a retirement stops is resolution into a new Session, and
    `retired` is how a caller sees that before creating one.
    """
    platform = platform_from_request(request)
    store = platform.skill_store
    held = await skill_or_refusal(store, platform.skill_inventory, tenant_id, skill_id)
    if isinstance(held, JSONResponse):
        return held
    record = await store.read_skill_version(tenant_id, skill_id, version)
    if record is None:
        return refuse(
            ErrorCode.SKILL_VERSION_NOT_FOUND,
            "this skill has no version by that name",
            skill_id=str(skill_id),
            version=str(version),
        )
    return _view_of(record)


@version_routes.delete(
    "/skills/{skill_id:uuid}/versions/{version}",
    response_model=SkillVersionDeleted,
    responses=_SKILL_REFUSALS,
)
async def retire_skill_version(
    skill_id: SkillId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    version: Annotated[int, Depends(version_from_path)],
) -> SkillVersionDeleted | JSONResponse:
    """Stop this version resolving into new Sessions, and keep it readable.

    Retiring an already-retired version answers exactly the same way, with the moment
    the first retirement recorded. A caller retrying a request that timed out must not
    be told the retry failed, and must not move the recorded moment either -- which is
    what a second row would do to the answer to "when did this stop being usable".

    The newest version may be retired, and `latest_version` then names the newest
    survivor. A regression is precisely the case retirement exists for, so refusing it
    would disable the feature exactly where it is needed; retire all of them and the
    skill is still readable, has a null `latest_version`, and resolves to nothing.
    """
    platform = platform_from_request(request)
    store = platform.skill_store
    held = await skill_or_refusal(store, platform.skill_inventory, tenant_id, skill_id)
    if isinstance(held, JSONResponse):
        return held
    record = await store.read_skill_version(tenant_id, skill_id, version)
    if record is None:
        return refuse(
            ErrorCode.SKILL_VERSION_NOT_FOUND,
            "this skill has no version by that name, so there is nothing to retire",
            skill_id=str(skill_id),
            version=str(version),
        )
    await store.retire_skill_version(tenant_id, skill_id, version)
    return SkillVersionDeleted(id=str(version))


@version_routes.get(
    "/skills/{skill_id:uuid}/versions/{version}/content",
    response_class=Response,
    responses=_SKILL_REFUSALS,
)
async def download_skill_version(
    skill_id: SkillId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    version: Annotated[int, Depends(version_from_path)],
) -> Response:
    """This version's whole bundle as a zip archive, rooted at its directory.

    The endpoint that makes a version's content readable at all. Every other read here
    answers with four columns; the files are the reason versions exist, and without this
    a caller could see that `forms.md` was uploaded and never get it back.

    A retired version downloads. The archive is how a pinned version's content stays
    reachable, and a definition that pinned one has a history that names it.

    Served as an attachment and marked unsniffable. The media type is this platform's
    rather than the caller's, so the sniffing risk is smaller than it is for an uploaded
    file -- but the bytes inside are still the tenant's, and a browser that decided this
    was HTML would run their script on this platform's origin.
    """
    platform = platform_from_request(request)
    store = platform.skill_store
    held = await skill_or_refusal(store, platform.skill_inventory, tenant_id, skill_id)
    if isinstance(held, JSONResponse):
        return held
    bundle = await store.read_skill_version_bundle(tenant_id, skill_id, version)
    if bundle is None:
        return refuse(
            ErrorCode.SKILL_VERSION_NOT_FOUND,
            "this skill has no version by that name, so there is no archive to build",
            skill_id=str(skill_id),
            version=str(version),
        )
    # Built by concatenation without quoting, which is safe for exactly one reason: the
    # name matched `SKILL_NAME_PATTERN` at the parse, so it holds no quote, no
    # backslash and no control character, and the version is decimal digits.
    filename = f"{bundle.record.name}-{bundle.record.version}.zip"
    return Response(
        content=archive_of(bundle),
        media_type="application/zip",
        headers={
            "content-disposition": f'attachment; filename="{filename}"',
            "x-content-type-options": "nosniff",
        },
    )
