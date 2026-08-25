"""The skill surface: the two write doors, the listing, and one skill read or deleted.

One skill per upload, addressed afterwards by the id this mints. One whole skill
directory per repository submission, addressed afterwards by the `(repository,
revision)` pair an agent definition already pins. Nothing else registers a skill, and
neither door accepts a body it has not parsed.

The five version routes live in `skill_versions.py` and are folded into this router
rather than mounted separately, so `/v1/skills` has one router and the composition root
needs no second line. Split by size and not by subject: seven routes and their wire
models in one file is past what this repository keeps in one file, and the version
routes are the half that reads as its own thing.

The listing exists to give an upload's id back. Before it, that 201 was the only record
of a skill anywhere a tenant could reach: the body stayed stored, immutable and
undeletable, and no request could name it again -- so a lost response was a lost skill
and the only remedy was to upload the document a second time and hold two.

Every refusal at the two write doors is a 400 out of the parse rather than an error
envelope, and that is the design rather than a shortcut. What makes a skill
unregisterable is a property of the document -- no frontmatter, no description, a name
that is not a legal directory, a path nothing scans -- so the parse is the whole of the
decision and the message names the skill and the reason. A second check inside the
handler would be a second answer to "is this skill well-formed", free to disagree with
the first.

The repository door reaches that parse through a model annotation, so FastAPI answers
before the handler runs. The create door parses in the handler, because it dispatches on
the content type and an annotation would have committed FastAPI to one of the two before
this code could look. The property the annotation was there for is kept rather than
traded away: one parse, before any store is touched, and the refusal raised as the same
validation failure the annotation raised so a caller cannot tell which door decided.

The listing has no document to parse and refuses in the published closed set instead;
the one thing it can be asked that it cannot answer is a cursor it did not issue.

No route in this file or in `skill_versions.py` touches a Session, so none appends to
the Event Log. The writes all run before any definition can name what they registered,
which is before any Session exists.

All of them are scoped through the same unauthenticated tenant placeholder as every
other route in this package. A skill is written by a domain team and submitted by that
team's CI, which is the same principal that already submits that repository's eval
scores; inventing a separate one here would decide an authorization question no decision
record has opened.

**`POST /v1/skills` takes a multipart bundle or a JSON document, and dispatches on the
content type.** Multipart is the shape the reference publishes and the shape a generated
client sends, so a create call from one used to be refused as malformed -- no skill
could be created by a caller who had not been written against this platform
specifically. JSON stays because it works and because this package's own repository door
and every test here speak it; removing a working door is not what parity asks for.

Two parses and one refusal set, which is the property that matters. What makes a
document unregisterable is a property of the document -- no frontmatter, no description,
a name that is not a legal directory -- so it cannot depend on which content type
carried it. A caller must not be able to get a malformed skill stored by re-sending it
the other way, and the JSON door's refusal is raised as the same validation failure the
annotation used to raise so both doors answer through one handler.

The multipart parse is `skill_bundles.parse_bundle` -- the same one the version-create
route uses -- rather than a second parser written here. A second one would be a second
answer to what a path inside a bundle may be, and this surface has exactly one place
that decides that.

**Both doors mint version 1**, so a skill this API creates always resolves to something.
The reference says a Skill holds at least one version, and a client doing create then
read then use-the-latest used to be handed null. A multipart create writes the parts it
was sent; a JSON create writes a version holding just the document, filed under the
skill's own name because a flat submission names no directory of its own.

The skill row is written before the version and the order is forced rather than chosen:
`skill_version.skill_id` is a foreign key onto `skill.id`, so there is no sequence in
which the version could come first. They are two writes, so a failure between them
leaves a skill with no version -- and that is a state this surface already represents
and already publishes, because it is the state every skill registered before versions
existed is in. `latest_version` reads null, the version listing is empty, and the remedy
is `POST /v1/skills/{id}/versions`. What the order rules out is the opposite failure, a
version row referring to a skill that was never recorded, which no read could explain.
One transaction would be better and needs a store method that writes both; there is
none, and inventing one is not this file's decision to make.
"""

import json
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from typing import Annotated, Final, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from managed_agent.control.api.refusals import Refusal, refuse
from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.tenancy import unauthenticated_tenant_from_header
from managed_agent.control.api.routes.skill_bundles import (
    REASON_BUNDLE_INVALID,
    SkillBundle,
    bundle_of_document,
    parse_bundle,
)
from managed_agent.control.api.routes.skill_versions import (
    version_routes,
    write_version,
)
from managed_agent.control.skills.inventory import (
    RepositorySkillRow,
    SkillOrigin,
    UploadedSkillRow,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorEnvelope
from managed_agent.core.ids import SkillId, TenantId, new_skill_id
from managed_agent.core.registration.definition import SkillsRevision
from managed_agent.core.registration.skill import (
    SkillType,
    ValidatedSkill,
    scan_tree,
)

router = APIRouter(tags=["skills"])

# Folded in here rather than mounted in `app.py`, so `/v1/skills` is served by one
# router and adding a route to that prefix is a change in one place. The version routes
# are registered ahead of everything below, which nothing depends on: no path here is
# ambiguous against another, because `{skill_id:uuid}` cannot match `repository`,
# `evals` or `baselines`.
router.include_router(version_routes)

_UPLOADED = "the uploaded SKILL.md"
"""What the refusal calls a document that arrived with no path around it."""

_FILES_FIELD: Final = "files"
"""The multipart field name the reference's create call sends its parts under.

Theirs, not ours -- `-F files=...` is what an SDK emits -- and the same name
`POST /v1/skills/{id}/versions` already takes, so one caller does not have to learn two
spellings for the same thing on the same resource.
"""

DOCUMENT_FIELD: Final = "skill_md"
"""The JSON door's only field: the whole `SKILL.md` including its frontmatter.

Taken whole rather than as a name and a description beside a body, because that is what
is stored and what the runtime reads -- separate fields would let the request and the
document disagree about what the skill is called, and only one of the two can be
delivered.
"""

_DISPLAY_NAME: Final = "display_name"
"""The reference's optional human label, taken by both create doors and stored.

Distinct from `name`, which is parsed out of the frontmatter and is also a directory and
a delivery path. This one is prose a person chose, is not sent to the model, and has no
grammar to satisfy -- so the two cannot be folded into one field even though both are
"what the skill is called".

Absent means absent. It is never defaulted to `name`, which would make "this skill has
no label" unrepresentable: every row would carry one and no reader could tell an
author's choice from a fallback.
"""

MAX_DISPLAY_NAME_CHARS: Final[int] = 256
"""How long a label may be.

Bounded because it is published on every listing row, and a page holds up to
`MAX_PAGE_SIZE` of them -- an unbounded label makes an unbounded response out of a
collection whose whole point is that it can be walked in fixed-size pages. Longer than
any label a person writes and far shorter than the 1024 a description gets, because a
label is a few words by construction: it exists to be read in a list.
"""


_MULTIPART: Final = "multipart/form-data"
"""Matched as a prefix of the content type, never compared whole: the header carries a
boundary parameter after the media type, so an equality test never fires."""

REASON_CURSOR_INVALID: Final[str] = "cursor_invalid"
"""Named in `detail` because the published code set is closed and has no paging family.

A caller branches on `request.invalid` and then on this, which is where anything
branchable belongs (ADR-013). The alternative was a code of this route's own, and a code
invented in a route is an unversioned addition to the published contract.
"""

DEFAULT_PAGE_SIZE: Final[int] = 25
MAX_PAGE_SIZE: Final[int] = 1000
"""How many skills one page may hold, matching the reference's own bound.

Bounded at all because an unbounded page is a whole-collection read wearing a limit
parameter, and a tenant's uploaded-skill count has no ceiling -- a skill is immutable
and never deleted, so every edit of one leaves the previous one in the collection for
good.

**This number is not free to move on its own, and the constraint is arithmetic rather
than taste.** The route asks the store for `limit + 1` -- one probe row past the page,
so that "is there another page" is answered without a second query -- so the adapter's
own bound has to be at least one greater than whatever is published here. `_MAX_PAGE` in
`adapters/postgres/skill_registry.py` is 1024 for that reason, and
`MAX_VERSION_PAGE_SIZE` sits under `_MAX_VERSION_PAGE` the same way.

Raising this past the adapter's bound would not fail anything until a caller actually
sent the published maximum, and it would fail as a 500 -- the surface answering that its
own documented limit is a server fault. That is the failure the two constants exist in
relation to prevent, and it stayed invisible for as long as the adapter's 500 sat
against a published 100: the gap was wide enough to hide the rule.
`test_skill_listing.py` pins the relationship arithmetically rather than trusting this
paragraph, because a comment cannot fail.
"""


class InvalidCursor(Exception):
    """The caller sent something that is not a cursor this surface issued."""


@dataclass(frozen=True, slots=True)
class SkillCursor:
    """A position in one tenant's name-ordered list of uploaded skills.

    Both halves are needed. Two uploads may deliberately carry the same name, and a
    position naming only the name cannot say which of them the caller already has -- so
    a page boundary landing between them would repeat one row or drop the other.

    The id leads in the encoded form and the name follows, which is the one ordering
    that survives a name containing a dot: `SKILL_NAME_PATTERN` admits one and a uuid
    does not, so splitting on the first separator always cuts in the right place.
    """

    name: str
    skill_id: SkillId

    def encode(self) -> str:
        """The position as a token, base64url with its padding stripped.

        Padding is stripped so the token carries no `=`, which would be percent-encoded
        in a query string and come back looking different from what was issued.
        """
        raw = f"{self.skill_id}.{self.name}".encode()
        return urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> "SkillCursor":
        """Parse a token back into a position, or raise `InvalidCursor`.

        Everything that is not a token this surface issued is one refusal. There is no
        partial reading -- a token whose id parses and whose name is empty names no row,
        and no stored skill has a blank name, so reading it as a position would restart
        the walk at the top of the collection without saying so.
        """
        try:
            padded = token + "=" * (-len(token) % 4)
            text = urlsafe_b64decode(padded.encode()).decode()
            identifier, _, name = text.partition(".")
            if not name:
                raise InvalidCursor(token)
            return cls(name=name, skill_id=SkillId(UUID(identifier)))
        except ValueError as exc:
            # binascii.Error and UnicodeDecodeError are both ValueError, so one clause
            # covers bad base64, bad utf-8 and a malformed uuid.
            raise InvalidCursor(token) from exc


class SkillSource(BaseModel):
    """Whose catalogue a skill belongs to, as the object the reference publishes.

    **An object holding one field rather than the bare string it wraps**, and the
    nesting is the contract rather than a preference. A client generated from the
    reference evaluates `skill.source.type`; handed a string it raises instead of
    reading the wrong answer, so publishing the enum directly is a break and not a
    cosmetic difference.

    Always `custom` here. `anthropic` names Anthropic's pre-built catalogue -- pdf,
    docx, xlsx, pptx -- which are bodies this platform does not hold and which
    `SkillAttachment` already refuses at the parse, naming the catalogue in the refusal.
    A field with one possible value still earns publishing when its absence fails a
    consumer's parse.

    Declared once and used by both the read and the listing row, which is the point. The
    same fact used to be published under two names on two payloads -- `source` on one
    and a bare `type` on the other -- which is two places for one hardcoded value to be
    changed and one chance to change only one of them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: SkillType = SkillType.CUSTOM


CUSTOM_SOURCE: Final = SkillSource()
"""The one source value this platform can hold, built once rather than per row.

Safe to share because the model is frozen: a page of five hundred rows would otherwise
allocate five hundred identical immutable objects to say one thing.
"""


class SkillRegistered(BaseModel):
    """What an uploader gets back: the id to attach by, and what the platform read.

    `name` and `description` come back because they were read out of the document
    rather than sent, and an uploader who mistyped a name would otherwise find out from
    an agent that never invokes the skill.

    `version` is always `latest` and is returned anyway, so the value an attachment may
    carry is one a caller has seen rather than one they have to know. It is the name of
    a resolution rule and not the version that was just minted: `SkillAttachment`
    accepts only `latest`, deliberately, because a Session resolves to whatever the
    newest live version is at the moment it starts. The minted version itself is read
    back from `GET /v1/skills/{id}` as `latest_version`, or from the version listing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SkillId
    name: str
    description: str
    version: str = "latest"


class UploadedSkillRead(BaseModel):
    """One uploaded skill read back by its id: what it says, and which version is
    current.

    `latest_version` is the greatest version nobody has retired, as a string, or null
    once every version has been -- a skill that is still readable and resolves to
    nothing. No endpoint sets it: the reference offers neither a promote nor an activate
    anywhere across its skills pages, so latest is derived from the set rather than
    pointed at, and a stored pointer would need an UPDATE that every table behind it
    refuses.

    `display_name` is the tenant's own label, or null for a skill nobody gave one.
    **Null here is a fact and not a gap**: it means no label was set, which is why the
    create door never defaults it to `name`. Defaulting would make the two
    indistinguishable, and a caller building an interface would have no way to tell a
    chosen label from a directory name it happened to be handed. Every skill registered
    before the column existed reads null too, and those two nulls are deliberately one
    value -- "nobody labelled this" is true of both, and a third state saying which era
    a row is from would be a fact about our migrations rather than about the skill.

    `source` and `origin` are published in exactly the shapes the listing row publishes
    them. That sameness is the decision: this once published `source` as a bare enum
    while the listing published the same fact as a field called `type`, so one value had
    three spellings across two payloads and a client could not parse a skill the same
    way wherever it read one.

    No body. The document and the files beside it are read per version, from
    `GET /v1/skills/{id}/versions/{version}/content`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SkillId
    name: str
    description: str
    display_name: str | None
    latest_version: str | None
    source: SkillSource = CUSTOM_SOURCE
    origin: Literal[SkillOrigin.UPLOAD] = SkillOrigin.UPLOAD


class RepositorySkillRead(BaseModel):
    """One repository skill read back by the id its checkout's submission assigned it.

    **The body is served here and on no listing row.** The caller asked for exactly this
    skill and there is one of it, where a page of bodies is a page nobody can walk. It
    is served at all because there is nowhere else to get it: a repository skill has no
    version archive, since its content is whatever the commit holds.

    **`latest_version` is null, and the `(repository, revision)` pair beside it is what
    actually pins the body.** Reporting the revision there was the alternative and it is
    worse: `latest_version` is a version string this platform minted, every version
    route parses one as sixteen to nineteen digits, and a caller handed a
    forty-character commit id would carry it straight to a route that refuses it -- a
    published value that no endpoint on this surface accepts. Null plus the commit says
    the same thing without inviting that. It is present rather than absent because the
    field is in the published shape for a skill and a client reads it; what it means
    here is "this platform minted no version for this skill", which is exactly true.

    No `display_name`, absent rather than null, for the reason `RepositorySkillListed`
    omits it: a checkout submits a directory rather than a labelled skill, so the label
    is a field that does not apply and a null would invite a caller to set one through a
    door that does not exist.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SkillId
    name: str
    description: str
    body: str
    repository: str
    revision: str
    latest_version: None = None
    source: SkillSource = CUSTOM_SOURCE
    origin: Literal[SkillOrigin.REPOSITORY] = SkillOrigin.REPOSITORY


SkillRead = Annotated[
    UploadedSkillRead | RepositorySkillRead, Field(discriminator="origin")
]
"""One skill read back, of whichever origin, discriminated by `origin`.

The same union the listing rows form, for the same reason: the two origins carry
different facts, and one shape with every difference nullable cannot say which of "not
applicable" and "not set" a null means.
"""


class SkillDeleted(BaseModel):
    """What a delete answers: the id that is gone, and what happened to it.

    `{id, type}` and nothing else, which is the reference's own shape for this
    operation. No timestamp -- the one field a caller might expect and the one the
    reference does not send. The moment is kept in the store, where a repeated delete
    reads the first one back rather than moving it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SkillId
    type: Literal["skill_deleted"] = "skill_deleted"


class RepositorySkills(BaseModel):
    """One checkout's whole skill directory, submitted by whoever holds the checkout.

    `files` is the tree as paths to text. Which of those paths hold a skill is decided
    by `scan_tree`, not by the submitter, and a path that looks like a skill and is not
    one refuses the submission rather than being passed over -- a submission that
    quietly registered nothing is what this endpoint exists to make impossible.

    `revision` is the same pinned 40-character commit id an `AgentDefinition` carries,
    reusing that model's type so the string a definition pins and the string a
    submission is filed under cannot drift into two shapes that never match. A branch
    is refused for the reason it is refused there: the skills of a moving reference
    would change under a Session already running on them, and a merged pull request
    nobody reviewed would change what a registered agent does.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1)
    revision: SkillsRevision
    files: dict[str, str] = Field(min_length=1)

    @property
    def skills(self) -> tuple[ValidatedSkill, ...]:
        """The skills in this tree. Cannot raise once an instance exists."""
        return scan_tree(self.files)

    @model_validator(mode="after")
    def _scan(self) -> "RepositorySkills":
        scan_tree(self.files)
        return self


class RepositorySkillsRecorded(BaseModel):
    """What CI gets back: which skills this checkout has, and whether they were new.

    `newly_recorded` is 0 when every skill was already filed under this exact commit,
    which is how a retried job tells its own retry from a first submission. A commit's
    skills do not change, so the first submission's answer stands and the retry writes
    nothing.

    `skill_ids` maps each name to the id that skill is addressed by everywhere else --
    the read, the listing, and the refusals that say it may not be deleted or versioned.
    Without it this door registered skills that nothing could then name: every other
    route on this surface takes a uuid, and a repository skill did not have one until
    `migrations/versions/0028` gave it one.

    **A map beside the ordered names rather than replacing them, and the redundancy is
    deliberate.** `skills` carries the order the platform read the directory in, which a
    map cannot; `skill_ids` carries identity, which is looked up by name and has no
    order to preserve. They cannot disagree -- both are built from the one list the
    submission produced -- and the names are published under both only because each
    carries a fact the other cannot. Replacing `skills` outright was the tidier option
    and would have broken a published field that callers already read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str
    revision: str
    skills: list[str]
    skill_ids: dict[str, str]
    newly_recorded: int


class UploadedSkillListed(BaseModel):
    """One skill that arrived by upload, as a listing row.

    No `skill_md`, and that is the decision the row is shaped by. A body is up to 32 KiB
    and a tenant's collection has no ceiling, so a page carrying bodies is a page nobody
    can walk -- and `name` and `description` were read out of the document at upload and
    stored as columns beside it precisely so this read opens no document.

    **A body is read back per version, not per skill.** The document and the files
    beside it come from `GET /v1/skills/{id}/versions/{version}/content`, which serves
    them as an archive.

    `display_name` is here and not only on the read, which is the point of having it: a
    label exists to be shown in a list, and a listing that made a caller fetch every row
    individually to learn what to call it would leave the field doing no work.

    `origin` is a `Literal` rather than the enum, which is what makes this and
    `RepositorySkillListed` a discriminated union on the wire: a client switches on it
    and a generated one gets two concrete shapes instead of one shape with every
    checkout column nullable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SkillId
    name: str
    description: str
    display_name: str | None
    source: SkillSource = CUSTOM_SOURCE
    origin: Literal[SkillOrigin.UPLOAD] = SkillOrigin.UPLOAD


class RepositorySkillListed(BaseModel):
    """One skill that arrived by repository submission, as a listing row.

    Carries the checkout it came from, because that pair is how a definition names it
    and is the only thing telling two submissions of one skill at different commits
    apart.

    **No `display_name`, and it is absent rather than null.** A checkout submits a
    directory, not a labelled skill, so the label is a field that does not apply here --
    and a null would say "nobody set one", which invites a caller to offer to set it
    through a door that does not exist. That distinction is the whole reason this is a
    union and not one row with every column nullable: one nullable shape cannot tell
    "not applicable" from "not set", and a caller reading a null `revision` on an
    uploaded skill would conclude it came from a commit nobody recorded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SkillId
    name: str
    description: str
    repository: str
    revision: str
    source: SkillSource = CUSTOM_SOURCE
    origin: Literal[SkillOrigin.REPOSITORY] = SkillOrigin.REPOSITORY


SkillListed = Annotated[
    UploadedSkillListed | RepositorySkillListed, Field(discriminator="origin")
]
"""One listing row, of whichever origin, discriminated by `origin`.

Mirrors the store's own `SkillRow` union rather than flattening it. Flattening would
mean one row with `display_name`, `repository` and `revision` all nullable, and a caller
that forgot an arm would read a `None` meaning "not applicable" here and "not set"
there.
"""


def _listed(row: UploadedSkillRow | RepositorySkillRow) -> SkillListed:
    """One stored row as its wire shape, matched on which kind of row it is.

    A match rather than a constructor with optional arguments, so adding a third origin
    is a new arm the type checker demands rather than another nullable field nobody
    notices is always null.
    """
    match row:
        case UploadedSkillRow():
            return UploadedSkillListed(
                id=row.skill_id,
                name=row.name,
                description=row.description,
                display_name=row.display_name,
            )
        case RepositorySkillRow():
            return RepositorySkillListed(
                id=row.skill_id,
                name=row.name,
                description=row.description,
                repository=row.repository,
                revision=row.revision,
            )


class SkillPage(BaseModel):
    """One page of a tenant's uploaded skills, and where the page after it starts.

    `next_page` is null at the end of the walk rather than a token leading somewhere
    empty, so a caller stops on a field it can read instead of on a wasted round trip.

    `data` and not `skills`, because that is the key every collection on this API
    publishes its rows under and the key a generated client reads. A collection
    answering under its own name hands that client nothing, so the walk stops before it
    starts.

    **`has_more` is kept even though the reference does not send it, and that is a
    decision rather than an oversight.** An extra field breaks no generated client -- a
    consumer ignores what it was not told about -- and this collection's own published
    shape carries it, as does the version listing beside it. Deleting it to match the
    reference field-for-field would remove something callers already read in order to
    gain nothing. The two cannot disagree: both are computed from the one extra row the
    store was asked for, so a reader branching on either gets the same answer. A caller
    walking the listing should follow `next_page`; `has_more` is for one that only wants
    to know whether the collection is exhausted.

    There is deliberately no `prev_page`. Paging backward needs the store to answer "the
    rows BEFORE this position", which it cannot, and a field emitted as null on every
    page would state that the caller is on the first page. An absent field leaves a
    consumer's own default to say "unknown"; a present-and-null one says something
    false.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: list[SkillListed]
    next_page: str | None
    has_more: bool


@dataclass(frozen=True, slots=True)
class SkillCreation:
    """What either create door produces: the bundle, and the label sent beside it.

    A pair rather than a `display_name` field on `SkillBundle`, because a bundle is also
    what `POST /v1/skills/{id}/versions` parses and that door has no label to carry. A
    field that is always None on one of two callers is a field whose meaning depends on
    who is holding it.

    The label is `str | None` and None is a real value meaning the caller gave none --
    not a placeholder for "not yet read". Both doors settle it before this exists.
    """

    bundle: SkillBundle
    display_name: str | None


def _label_from(raw: object) -> str | None:
    """One submitted label checked into a stored one, or the refusal for why not.

    Shared by both doors so a label acceptable through one cannot be refused by the
    other -- a field one door took and the other did not would make the two doors two
    surfaces, and a caller would have to know which to use in order to set a label.

    A non-string is refused rather than coerced. `str(3)` is a label the tenant did not
    write, and it would be read back by whoever wonders why their skill is called "3".
    Multipart cannot deliver a non-string at all, so this arm exists for the JSON door;
    it is shared anyway, because the alternative is two answers to what a label is.

    Whitespace-only is refused for the reason an empty file in a bundle is: a label that
    renders as nothing is indistinguishable in a listing from having set none, and the
    caller believes they set one. Untrimmed otherwise -- the tenant's spacing is theirs.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise Refusal(
            ErrorCode.REQUEST_INVALID,
            f"{_DISPLAY_NAME!r} is text or absent, and this request sent a "
            f"{type(raw).__name__}; it is not coerced, because a label nobody wrote is "
            "one somebody will later have to explain",
            reason=REASON_BUNDLE_INVALID,
            field=_DISPLAY_NAME,
        )
    if not raw.strip():
        raise Refusal(
            ErrorCode.REQUEST_INVALID,
            f"{_DISPLAY_NAME!r} is blank, and a blank label reads in a listing exactly "
            "like a skill that was never given one; omit the field instead",
            reason=REASON_BUNDLE_INVALID,
            field=_DISPLAY_NAME,
        )
    if len(raw) > MAX_DISPLAY_NAME_CHARS:
        raise Refusal(
            ErrorCode.REQUEST_INVALID,
            f"{_DISPLAY_NAME!r} is {len(raw)} characters and may be at most "
            f"{MAX_DISPLAY_NAME_CHARS}; it is published on every row of a listing "
            "page, so an unbounded label is an unbounded response",
            reason=REASON_BUNDLE_INVALID,
            field=_DISPLAY_NAME,
        )
    return raw


async def _from_json(request: Request) -> SkillCreation:
    """The JSON door's document and label, or the refusal for why not.

    Parsed here rather than through a body annotation, because an annotation commits
    FastAPI to one content type before the handler runs and there would then be nothing
    left to dispatch on. What the annotation was there for is kept: one parse, before
    any store is touched, and a refusal naming what is wrong with the document.

    **Every refusal is a `Refusal`, and the document's own refusal comes from the same
    conversion the multipart door uses.** That is what makes the two doors one refusal
    set rather than two sets that currently agree: a malformed document produces one
    object, built in one place, so no rearrangement of these two functions can leave a
    document acceptable to one door and not the other.

    Two keys and no others, checked by hand because there is no model here to carry
    `extra="forbid"`. An unrecognised key is refused rather than ignored for the reason
    the multipart door refuses one: a field accepted and dropped is the
    accept-it-and-deliver-nothing failure this surface exists to prevent.
    """
    raw = await request.body()
    try:
        payload = json.loads(raw)
    except ValueError as unreadable:
        raise Refusal(
            ErrorCode.REQUEST_INVALID,
            "the request body is not JSON, and this door takes either a JSON object "
            f"carrying {DOCUMENT_FIELD!r} or a multipart form of files",
            reason=REASON_BUNDLE_INVALID,
        ) from unreadable
    if not isinstance(payload, dict):
        raise Refusal(
            ErrorCode.REQUEST_INVALID,
            f"the request body is a JSON {type(payload).__name__} and this door takes "
            f"an object carrying {DOCUMENT_FIELD!r}",
            reason=REASON_BUNDLE_INVALID,
        )
    unexpected = sorted(set(payload) - {DOCUMENT_FIELD, _DISPLAY_NAME})
    if unexpected:
        raise Refusal(
            ErrorCode.REQUEST_INVALID,
            f"this door takes {DOCUMENT_FIELD!r} and {_DISPLAY_NAME!r} and nothing "
            f"else; it was also sent {', '.join(unexpected)!r}",
            reason=REASON_BUNDLE_INVALID,
            field=unexpected[0],
        )
    document = payload.get(DOCUMENT_FIELD)
    if not isinstance(document, str) or not document:
        raise Refusal(
            ErrorCode.REQUEST_INVALID,
            f"{DOCUMENT_FIELD!r} is the whole SKILL.md including its frontmatter, as a "
            "non-empty string. It is taken whole rather than as a name and a "
            "description, so the request and the document cannot disagree about what "
            "the skill is called -- only one of the two could be delivered",
            reason=REASON_BUNDLE_INVALID,
            field=DOCUMENT_FIELD,
        )
    return SkillCreation(
        bundle=bundle_of_document(document, source=_UPLOADED),
        display_name=_label_from(payload.get(_DISPLAY_NAME)),
    )


async def _from_multipart(request: Request) -> SkillCreation:
    """The multipart door's parts and label, or the refusal for why not.

    **A form field this route does not recognise is refused rather than ignored**, which
    is the whole reason this reads the form's keys at all. The JSON door would get that
    from a model's `extra="forbid"`; multipart has no such default, so an unknown part
    is silently dropped -- and a field accepted and dropped is the
    accept-it-and-deliver-nothing failure this surface exists to prevent, worse than a
    refusal because the caller believes the value was set.

    A `files` part that arrived without a filename is refused here rather than inside
    the parse. Starlette hands such a part over as a string rather than an upload, so it
    never reaches the code that would have said the filename is the path; without this
    it would be dropped from the list and the bundle would look one file short of what
    was sent.

    The label arrives as an ordinary form field and goes through the same `_label_from`
    the JSON door uses. A multipart field is always text, so the non-string arm there
    cannot fire from this side -- it is shared anyway, because two checks would be two
    answers to what a label is.
    """
    async with request.form() as form:
        unexpected = sorted(set(form.keys()) - {_FILES_FIELD, _DISPLAY_NAME})
        if unexpected:
            raise Refusal(
                ErrorCode.REQUEST_INVALID,
                f"this door takes multipart parts named {_FILES_FIELD!r} and an "
                f"optional {_DISPLAY_NAME!r}; it was also sent "
                f"{', '.join(unexpected)!r}",
                reason=REASON_BUNDLE_INVALID,
                field=unexpected[0],
            )
        parts = form.getlist(_FILES_FIELD)
        # A part with a filename arrives as an upload and one without arrives as a plain
        # string, so the string case *is* the nameless-part case. Tested against `str`
        # rather than against `UploadFile` because the form yields Starlette's class and
        # `fastapi.UploadFile` is a subclass of it -- an isinstance test the other way
        # round is false for every part and refuses every legal bundle.
        if any(isinstance(part, str) for part in parts):
            raise Refusal(
                ErrorCode.REQUEST_INVALID,
                f"one of the {_FILES_FIELD!r} parts arrived with no filename, and the "
                "filename is the path the file is delivered at; send each file as its "
                "own part with the path it should have inside the skill directory",
                reason=REASON_BUNDLE_INVALID,
            )
        return SkillCreation(
            bundle=await parse_bundle(
                [part for part in parts if not isinstance(part, str)]
            ),
            display_name=_label_from(form.get(_DISPLAY_NAME)),
        )


async def _created_skill(request: Request) -> SkillCreation:
    """Whichever of the two doors this request came through, as one value.

    Dispatched on the content type, and multipart is recognised by prefix because the
    header carries a boundary parameter after it. Anything else is read as JSON,
    including a request with no content type at all: that is what a caller sending a
    bare body means, and it is the door this route had before multipart was added.

    One return type on purpose. Everything downstream -- the skill row, the first
    version, the response -- is written from a `SkillCreation` and cannot tell which
    door produced it, which is what keeps the two doors from drifting into two
    behaviours.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.startswith(_MULTIPART):
        return await _from_multipart(request)
    return await _from_json(request)


@router.post(
    "/skills",
    status_code=status.HTTP_201_CREATED,
    response_model=SkillRegistered,
    responses={STATUS_FOR[ErrorCode.INTERNAL]: {"model": PublicErrorEnvelope}},
)
async def upload_skill(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SkillRegistered | JSONResponse:
    """Store one skill with its first version, and report the id to attach it by.

    Takes a multipart bundle or a JSON document; `_created_bundle` decides which and
    both end up here holding the same value. No body annotation, because an annotation
    would commit FastAPI to one content type before this function could look.

    The id is minted here rather than taken from the uploader, for the reason `POST
    /v1/agents` mints a definition id: an uploader-chosen id could aim an upload at one
    another tenant already holds.

    Two creates of a skill with the same name are two skills with two ids, and that is
    deliberate rather than unhandled. A stored skill's row is immutable, because a
    Session resolves its skills once and then goes on reading exactly what it resolved
    to; the way to change what a skill says is a new version, and the way to replace one
    outright is a new id attached by a new definition revision.

    Version one is written after the skill row and the order is the foreign key's, not a
    preference -- see this module's docstring for what a failure between the two writes
    leaves behind and why that state is one the surface can already describe. A refusal
    from the version write is returned as it stands: the skill row exists, so answering
    201 would report a create that did not finish.
    """
    created = await _created_skill(request)
    skill_id = new_skill_id()
    store = platform_from_request(request).skill_store
    await store.add_skill(
        tenant_id, skill_id, created.bundle.skill, display_name=created.display_name
    )
    written = await write_version(store, tenant_id, skill_id, created.bundle)
    if isinstance(written, JSONResponse):
        return written
    return SkillRegistered(
        id=skill_id,
        name=created.bundle.skill.name,
        description=created.bundle.skill.description,
    )


@router.post(
    "/skills/repository",
    status_code=status.HTTP_201_CREATED,
    response_model=RepositorySkillsRecorded,
)
async def record_repository_skills(
    body: RepositorySkills,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> RepositorySkillsRecorded:
    """File one checkout's skill directory under the pair an agent definition pins.

    This is what gives `skills_repository` and `skills_revision` something to resolve
    to. The platform does not go and read the repository: the Managed Agents API can
    scan a mounted checkout at session start because its sandbox has the repository and
    the network to fetch it, and its own documentation says that discovery does not run
    for self-hosted sandboxes, which is what this platform is. So the skills come from
    whoever already has the checkout -- the same CI that submits that repository's eval
    scores -- and the pin then means exactly what it always claimed to: the skills of
    that commit.

    Filing skills for a commit is not the same as clearing the eval gate and this route
    deliberately does not consult it. The gate decides whether a revision may be
    *pinned by a definition*, and it is enforced at the two doors where a definition
    names one. A submission here writes what a commit contains, which is a fact about
    the commit rather than a permission; refusing it on a failing gate would leave the
    revision's skills unrecorded and the gate's own refusal unexplainable.
    """
    skills = body.skills
    platform = platform_from_request(request)
    written = await platform.skill_store.set_repository_skills(
        tenant_id, body.repository, body.revision, skills
    )
    # After the files, and the order matters on a retry: assigning an id for a skill
    # whose body was never stored would publish an id that resolves to nothing. The
    # assignment is idempotent on the same four columns the file table keys on, so a
    # resubmission returns the ids already held and writes neither table.
    assigned = await platform.skill_inventory.assign_repository_ids(
        tenant_id, body.repository, body.revision, [skill.name for skill in skills]
    )
    return RepositorySkillsRecorded(
        repository=body.repository,
        revision=body.revision,
        skills=[skill.name for skill in skills],
        skill_ids={name: str(skill_id) for name, skill_id in assigned},
        newly_recorded=written,
    )


@router.get(
    "/skills",
    response_model=SkillPage,
    responses={
        STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]: {"model": PublicErrorEnvelope}
    },
)
async def list_skills(
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
    page: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    source: SkillType | None = None,
) -> SkillPage | JSONResponse:
    """One page of every skill this tenant holds, of either origin, ordered by name.

    **Both write doors, in one collection, since `migrations/versions/0028`.** This
    listed only the uploaded half for as long as a repository skill had no id: they are
    addressed by the `(repository, revision)` pair a definition pins, so paging them
    beside uploaded skills would have meant a row with a nullable id and a nullable pair
    -- two shapes in one payload. 0028 gave every repository skill an id derived from
    its four key columns, so the id space is whole and one collection can serve both.

    What replaced the nullable row is a discriminated union rather than a wider row.
    `origin` says which door wrote a row, and each arm carries only the fields that door
    produces: an upload has a `display_name`, a repository skill has its checkout, and
    neither carries the other's as a null. The presence of an id is no longer what tells
    the two apart, which is what made one collection possible; `origin` is, and it also
    decides what a caller may then do -- an upload takes versions and a delete, a
    repository skill takes neither, because its body is fixed by the commit.

    The walk is one ordered read over both tables rather than a page from each merged
    here. Merging cannot be made total without over-fetching both sides, and the failure
    is silent: a caller walks to what looks like the end having seen part of what it
    holds. The keyset is `(name, id)` over the union, which is a total order because an
    id is unique across both origins.

    `?source=` filters on the reference's axis and not on that one. `source` says whose
    catalogue a body belongs to, `origin` says which door wrote the row, and the two are
    unrelated: every skill this platform holds is `custom` whichever door it came by.
    **`?source=anthropic` therefore answers an empty page rather than a refusal.** It is
    a legal value of its own enum -- a filter that rejects one of those is a filter that
    lies about what it accepts -- and the true answer for a catalogue this platform
    holds none of is nothing. A value the enum does not hold is a malformed request,
    which is a different thing and is refused as one.

    Another tenant's skills are absent rather than filtered out here: the tenant is a
    term in the store's own query, so there is no point in this function where a
    cross-tenant row exists and has to be dropped.

    The store is asked for one row more than will be returned. That extra row is the
    whole answer to "is there another page", and it is why `next_page` is null rather
    than a token leading somewhere empty.

    A cursor this surface did not issue is refused rather than treated as the start of
    the collection. Starting over on a bad cursor would silently hand a caller the first
    page again, which reads as the walk having looped rather than failed.
    """
    after: tuple[str, SkillId] | None = None
    if page is not None:
        try:
            position = SkillCursor.decode(page)
        except InvalidCursor:
            return refuse(
                ErrorCode.PAGINATION_CURSOR_INVALID,
                "cursor was not issued by this surface; walk the listing from the "
                "beginning by omitting it rather than by constructing one",
                reason=REASON_CURSOR_INVALID,
            )
        after = (position.name, position.skill_id)

    if source is not None and source is not SkillType.CUSTOM:
        # Answered without asking the store, because the answer does not depend on what
        # it holds: nothing here is Anthropic's catalogue and nothing ever will be. A
        # query would fetch rows in order to drop all of them, and the empty page it
        # produced would be indistinguishable from this one.
        return SkillPage(data=[], next_page=None, has_more=False)

    rows = await platform_from_request(request).skill_inventory.page(
        tenant_id, after, limit + 1
    )
    rows_shown = rows[:limit]
    more = len(rows) > limit
    return SkillPage(
        data=[_listed(row) for row in rows_shown],
        next_page=(
            SkillCursor(
                name=rows_shown[-1].name, skill_id=rows_shown[-1].skill_id
            ).encode()
            if more
            else None
        ),
        has_more=more,
    )


@router.get(
    "/skills/{skill_id:uuid}",
    response_model=SkillRead,
    responses={
        STATUS_FOR[ErrorCode.SKILL_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.SKILL_DELETED]: {"model": PublicErrorEnvelope},
    },
)
async def read_skill(
    skill_id: SkillId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SkillRead | JSONResponse:
    """One uploaded skill, by the id `POST /v1/skills` minted for it.

    The read that closes what the listing only narrowed. A caller holding an id could
    already page the collection until the row appeared, which is a walk whose cost grows
    with everything the tenant has ever uploaded -- for a lookup by primary key.

    **`{skill_id:uuid}` and not `{skill_id}`, and that is load-bearing rather than
    tidy.** This router is mounted ahead of the skill-eval router, and FastAPI matches
    in registration order, so a bare placeholder would swallow `GET
    /v1/skills/baselines` and answer a malformed-request refusal for a route that
    exists and works. The convertor makes the match fail instead, so the request
    falls through to the literal path. A skill id that is not a uuid gets the
    router's own not-found, which is the status this route would have answered
    anyway.

    **Either origin, and the uploaded half is tried first.** A caller holding an id has
    no way to know which door minted it, so consulting one half and answering 404 would
    be wrong about every skill from the other. Uploaded first because that is the half
    that can be deleted: a tombstoned skill has to answer 410 rather than fall through
    to a repository lookup that would answer 404 and undo the distinction. An id neither
    half holds is one 404, not two refusals a caller would have to tell apart.

    Written against `read_skill` directly rather than through `skill_or_refusal`, which
    the version routes share. That helper now answers 409 for a repository id, because a
    version route cannot act on one -- and this route can, so it must not inherit that
    refusal.

    A deleted skill answers 410 and not 404. The id was real and the platform
    deliberately stopped serving it, which is what a tenant honouring a deletion request
    has to be able to see; 404 would send them looking for a mistake in the id. Only an
    uploaded skill can be in that state: a repository skill cannot be deleted at all.

    An uploaded skill's body does not come back, unchanged from the listing -- it is
    read per version, from `GET /v1/skills/{id}/versions/{version}/content`. A
    repository skill's does, because it has no version archive to be read from instead:
    its content is whatever the pinned commit holds.
    """
    platform = platform_from_request(request)
    held = await platform.skill_store.read_skill(tenant_id, skill_id)
    if held is not None:
        if held.deleted:
            return refuse(
                ErrorCode.SKILL_DELETED,
                "this skill was deleted; its id is kept so a Session's history can "
                "still name what it ran, and nothing new resolves through it",
                skill_id=str(skill_id),
            )
        return UploadedSkillRead(
            id=held.skill_id,
            name=held.name,
            description=held.description,
            display_name=held.display_name,
            latest_version=(
                None if held.latest_version is None else str(held.latest_version)
            ),
        )
    from_commit = await platform.skill_inventory.repository_skill_at(
        tenant_id, skill_id
    )
    if from_commit is None:
        return refuse(
            ErrorCode.SKILL_NOT_FOUND,
            "no skill with that id is registered to this tenant, by upload or by "
            "repository submission",
            skill_id=str(skill_id),
        )
    return RepositorySkillRead(
        id=from_commit.skill_id,
        name=from_commit.name,
        description=from_commit.description,
        body=from_commit.body,
        repository=from_commit.repository,
        revision=from_commit.revision,
    )


@router.delete(
    "/skills/{skill_id:uuid}",
    response_model=SkillDeleted,
    responses={
        STATUS_FOR[ErrorCode.SKILL_NOT_FOUND]: {"model": PublicErrorEnvelope},
        STATUS_FOR[ErrorCode.SKILL_OWNED_BY_COMMIT]: {"model": PublicErrorEnvelope},
    },
)
async def delete_skill(
    skill_id: SkillId,
    request: Request,
    tenant_id: Annotated[TenantId, Depends(unauthenticated_tenant_from_header)],
) -> SkillDeleted | JSONResponse:
    """Stop serving this skill, keeping the id so a Session's history stays readable.

    A tombstone and not a DELETE of the row. A Session created under a definition that
    attached this skill has a history naming it, and an agent definition carries a
    digest over the skills it resolved; removing the row would leave both pointing at an
    id that resolves to nothing, which reads as the platform having lost the data rather
    than as the tenant having deleted it. Those two look identical from outside and mean
    opposite things.

    **The one read of a deleted skill that does not refuse.** Deleting twice answers the
    same way, with the moment the first delete recorded, because a caller retrying a
    request that timed out must not be told the retry failed. So this route reads the
    row itself rather than through `skill_or_refusal`, which answers 410 for the one
    case this route has to accept.

    The skill's versions need not be retired first. Requiring it would make a deletion a
    walk of unknown length, and the versions of a deleted skill are already unreachable:
    every route that reads one goes through the shared lookup and answers 410.

    **A repository skill refuses with 409, and it is the one id here that resolves and
    cannot be deleted.** Its content is fixed by the commit it was submitted from, and a
    definition pins `(repository, revision)`; removing the row would leave an
    already-registered definition unresolvable while the commit it names still sits in
    the repository -- the platform reporting lost data for a skill that is exactly where
    CI put it. There is no tombstone that would help, because the thing a tombstone
    records is a decision this surface is not the owner of. The way to stop using one is
    to stop pinning that revision, and the refusal says so rather than leaving a caller
    to work it out. 404 was the alternative and it lies: the id is real and the tenant
    holds it.

    What this does *not* do is refuse while something still uses the skill. A definition
    that attached it keeps resolving, because `read_attached` reads the skill row and
    the row is still there -- which is the same reason the history stays readable. That
    is a difference from `DELETE /v1/files/{id}`, where the bytes leave the object store
    and a live Session would fail at its next pod placement.
    """
    platform = platform_from_request(request)
    store = platform.skill_store
    held = await store.read_skill(tenant_id, skill_id)
    if held is None:
        from_commit = await platform.skill_inventory.repository_skill_at(
            tenant_id, skill_id
        )
        if from_commit is not None:
            return refuse(
                ErrorCode.SKILL_OWNED_BY_COMMIT,
                "this skill came from a repository checkout, so it is owned by "
                f"revision {from_commit.revision} of {from_commit.repository} rather "
                "than by this surface. Deleting the row would leave a definition "
                "pinning that revision unresolvable while the commit it names still "
                "exists, which reads as the platform having lost the skill. Stop "
                "using it by not pinning that revision",
                skill_id=str(skill_id),
                repository=from_commit.repository,
                revision=from_commit.revision,
            )
        return refuse(
            ErrorCode.SKILL_NOT_FOUND,
            "no skill with that id is registered to this tenant",
            skill_id=str(skill_id),
        )
    await store.delete_skill(tenant_id, skill_id)
    return SkillDeleted(id=skill_id)
