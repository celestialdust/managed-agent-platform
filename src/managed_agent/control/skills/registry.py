"""What the platform holds a skill for, and which skills one definition resolves to.

Two routes reach a Session and this module is where they meet. A definition attaches
skills by id, and it pins a `(skills_repository, skills_revision)` whose whole skill
directory it also gets. Both are read here, at the moment a Session is created, and
reduced to the one thing the pod can be handed: an ordered set of files.

Neither route trusts a runtime to go and look. The Managed Agents API can scan a
mounted repository at session start because its sandbox has the repository and the
network to fetch it; the note in that documentation saying repository discovery does
not work for self-hosted sandboxes is describing us. So a repository's skills are
submitted from the checkout that holds them, by the same actor that already submits
that repository's eval scores, and the platform holds what was submitted. The pin
therefore means what it always claimed to mean -- the skills of exactly that commit --
without anything in the control plane needing a git client or a tenant's credential.

A pair nobody has submitted for resolves to no skills rather than to a refusal, which
matches how the eval gate treats a repository nobody has graded: the mechanism turns on
for a pair with its first submission. Refusing instead would refuse every definition
registered before this existed, and an agent with no repository skills is a real thing
to want.

One place this deliberately behaves differently from the API it copies. There, a
repository skill and an attached skill may share a name and both stay available, each
announced under its own path. Here every skill is delivered into one flat directory, so
two with one name are two files at one path and the second would silently replace the
first. That is refused, and it is refused at the door of the Session rather than
discovered as a missing skill inside the pod.

An attached skill resolves to a *version*, and a version is a directory rather than a
document. So what leaves here is not one file per skill: a version's `SKILL.md` arrives
with every file stored beside it, which is the only reason a skill that tells the model
to read `forms.md` finds `forms.md` there. Two consequences worth stating once. The
delivered directory is the name the version's own document declares rather than the name
the skill was uploaded under, because a version may rename its skill and the runtime
reads the frontmatter it announces the directory from. And the flat directory's
collision comes back one level down: two files in two different skills' trees can still
fold to one entry in the Secret that carries them, so paths are refused as well as
names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from managed_agent.core.ids import SkillId, TenantId
from managed_agent.core.registration.definition import AgentDefinition
from managed_agent.core.registration.skill import (
    MAX_SKILLS_PER_AGENT,
    SKILL_DELIVERY_MAX_BYTES,
    PinnedSkillVersion,
    SkillFile,
    SkillVersionPin,
    ValidatedSkill,
    delivery_key,
    delivery_path,
    skill_files,
)


@dataclass(frozen=True, slots=True)
class SkillRecord:
    """One uploaded skill as the store holds it: its id and the skill itself.

    The id is separate from the skill because it is the platform's, not the tenant's --
    the tenant supplies a name and gets back an id to attach by. Two uploads with the
    same name are two records with two ids, and which one an agent runs is whichever
    one it attached.
    """

    skill_id: SkillId
    skill: ValidatedSkill


@dataclass(frozen=True, slots=True)
class SkillListing:
    """One uploaded skill as a listing shows it: the id to attach by, and what it says.

    Deliberately not a `SkillRecord`. A record carries the whole `SKILL.md`, which is
    capped at 32 KiB and exists to be delivered into a Session; a listing answers "what
    do I hold", and a page of twenty-five records would move most of a megabyte out of
    the database so that the surface could drop all of it. `name` and `description` are
    stored as columns beside the body for exactly this read, so answering it re-parses
    no document.

    `display_name` is the label a tenant may give a skill for people to read, and it is
    a different thing from `name`: the name is the identifier the runtime announces a
    directory as and matched `SKILL_NAME_PATTERN`, while this is free prose and is never
    sent to the model. Null for a skill nobody labelled, which is most of them and every
    one that predates the column. Not defaulted, because a listing that forgot to read
    the column would otherwise publish null for every row and nothing would say so.
    """

    skill_id: SkillId
    name: str
    description: str
    display_name: str | None


@dataclass(frozen=True, slots=True)
class SkillHeld:
    """One uploaded skill as a read of it answers: what it says, and its two states.

    `latest_version` is the greatest version nobody has retired, or None once every
    version has been. Computed by the read rather than stored: a stored pointer would
    need an UPDATE and every table behind this refuses one, and "the most recent
    version" is a fact about the set rather than about any row in it.

    `deleted` is a state and not an absence. The row survives a delete so a Session's
    history can still name the skill it ran; what the flag changes is that every read
    of the skill now refuses -- a different answer from "no such skill", which has to
    stay different, because one says the caller is wrong about the id and the other
    says the platform did what it was asked.

    `display_name` is the tenant's own label for the skill, held apart from `name` and
    null for a skill nobody labelled. Two fields rather than one because they answer to
    different rules and different readers: `name` matched `SKILL_NAME_PATTERN` and
    becomes a directory and a Secret key, while this is free prose a person reads in a
    listing and the model is never shown. Required rather than defaulted for the reason
    `SkillListing`'s is -- a read that forgot the column would report every skill as
    unlabelled, which is indistinguishable from the truth.
    """

    skill_id: SkillId
    name: str
    description: str
    display_name: str | None
    latest_version: int | None
    deleted: bool


@dataclass(frozen=True, slots=True)
class SkillVersionFile:
    """One sibling file of a version, at a path under the version's directory.

    Relative, and that is the whole safety property. The path is joined onto a
    directory inside a Session's workspace, so a leading separator or a `..` would put
    the file where the uploader chose rather than where the platform did. Both are
    refused where the bundle is parsed and again by the store's own check, because the
    filesystem cannot tell which of the two writers produced the row.
    """

    path: str
    text: str


@dataclass(frozen=True, slots=True)
class SkillVersionRecord:
    """One version as a read answers: which one, what it says, whether it still runs.

    `name` and `description` are this version's own, read out of the `SKILL.md` that
    arrived with it rather than off the skill. That is what makes a version a version:
    the document changed, and so did what the model is told the skill does.

    No creation timestamp, because the version is one. It is minted as microseconds
    since the epoch at the moment of the write, so a column holding that moment
    separately would be a second record of one fact, free to disagree with the first.
    """

    version: int
    name: str
    description: str
    directory: str
    retired: bool


@dataclass(frozen=True, slots=True)
class SkillVersionBundle:
    """A version and every file in it, which an archive download is built from.

    `skill_md` is the document itself and is deliberately not repeated in `files`: it
    is a column of the version row rather than one of the siblings, so there is no
    second copy to disagree with it about what the skill says.
    """

    record: SkillVersionRecord
    skill_md: str
    files: tuple[SkillVersionFile, ...]


class SkillsUnresolvable(Exception):
    """A definition names skills that cannot be turned into files, and why.

    Raised where a Session resolves rather than where the definition was written,
    because both causes depend on what the store holds: an id can be attached and later
    not be there, and two names collide only once both have been looked up.

    `detail` is flat -- `str` and `int` values only -- because it goes into the error
    envelope, whose detail may not nest. Where the cause is a set of things, one of them
    is named and the count comes with it.

    Carries no error code of its own, and that is a decision rather than an omission.
    Every refusal raised here is caught by whichever door was resolving: the two
    registration doors envelope it as `definition.invalid` with `field="skills"`, and a
    Session's placement carries the words into its own failure and publishes the one
    cause that surface publishes. A code minted here would therefore be a second name
    for a refusal those doors have already named, and the two would eventually be
    reported for the same cause. What travels is the message and the flat detail, which
    is what a caller reads either way.
    """

    def __init__(self, message: str, **detail: str | int) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class SkillVersionCollision(Exception):
    """Two versions of one skill were minted inside the same microsecond.

    Its own exception rather than the store's integrity error, because the answer to it
    is to mint again, and a caller that had to recognise a driver's error class in order
    to retry would be reaching through the port at the one moment that matters.
    """


class SkillStore(Protocol):
    """What a tenant's skills need held for them: storing, listing, and resolving.

    A port of its own rather than more methods on the definition registry, and the line
    is worth stating because the eval gate went the other way. Whether a pinned
    revision cleared its gate is a property *of the pin* and belongs with whatever
    resolves the pin. A skill body is a separate entity with its own id, its own
    lifetime and its own upload door, and a definition can be resolved without reading
    one.

    Every read takes the tenant. Not as a filter applied afterwards -- the tenant is a
    term in the query, so a skill id belonging to somebody else reads as absent rather
    than as forbidden, and a caller holding an id learns nothing from the refusal.
    """

    async def add_skill(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        skill: ValidatedSkill,
        *,
        display_name: str | None,
    ) -> None:
        """Store one uploaded skill under an id the platform minted, with its label.

        `display_name` is the tenant's optional label, and None is a real value meaning
        they gave none. Keyword-only and *not* defaulted here, which is the one place
        that matters: a label a caller forgot to pass would be stored as absent and read
        back as absent, so the tenant's field would have gone nowhere and every test
        would still pass. Requiring it at the port makes forgetting a type error at the
        call site instead.

        Keyword-only because the argument sits beside a `ValidatedSkill` that already
        carries a name and a description; positionally, a label and a body are two
        strings in a row that nothing would catch being swapped.

        An implementation may give it a default -- a default is a widening, so it still
        satisfies this -- and a store whose own callers predate labels is welcome to.
        What the port fixes is that no caller reaching a store *through* this type can
        drop a label by leaving it out.
        """
        ...

    async def read_skills(
        self, tenant_id: TenantId, skill_ids: Sequence[SkillId]
    ) -> tuple[SkillRecord, ...]:
        """The records among these ids that this tenant holds, ordered by skill name.

        Ids that do not resolve are absent from the result rather than reported. The
        caller knows what it asked for and can say which is missing; a store that
        raised on the first one could only ever name one.
        """
        ...

    async def page_uploaded_skills(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillListing, ...]:
        """One page of this tenant's uploaded skills, ordered by name then id.

        The order is the one `read_skills` already returns, so the store has a single
        answer to "in what order do this tenant's skills come back" rather than one per
        read. It is also a total order over the table -- the id breaks the tie that a
        name cannot, because two uploads may share a name on purpose -- which is what a
        page boundary needs: a key that repeats would repeat a row or drop one.

        `after` is the `(name, id)` of the last row the caller already holds, as one
        value rather than two nullable arguments, because half a keyset is not a
        position and a pair of parameters admits the half-set state as a value.

        Only the skills that arrived by id. A repository submission is addressed by the
        `(repository, revision)` pair a definition pins and has no id at all, so it is
        not a row in a collection of ids; `read_repository_skills` is where those are
        read, by the pair that names them.

        At most `limit` rows come back, and a short page means there is nothing after
        it. A `limit` the store will not serve is refused rather than clamped: a clamped
        page is a short page, and a short page is how this method says the walk is over.
        """
        ...

    async def read_skill(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> SkillHeld | None:
        """One uploaded skill, or None when this tenant holds none under that id.

        None covers both "never uploaded" and "belongs to somebody else", because the
        tenant is a term in the query and there is no moment at which the other
        tenant's row exists to be told apart. A deleted skill still answers, with
        `deleted` set: what a deletion means differs by caller, and every caller here
        means something different by it.
        """
        ...

    async def delete_skill(self, tenant_id: TenantId, skill_id: SkillId) -> None:
        """Record that this skill is gone, keeping the moment the first delete named.

        Idempotent, because a caller retrying a delete that timed out must not be told
        the second attempt failed -- and must not move the recorded moment either,
        which is what a second row would do. A skill this tenant does not hold writes
        nothing and reports nothing: the caller has already read it and refused.
        """
        ...

    async def add_skill_version(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        version: int,
        skill: ValidatedSkill,
        directory: str,
        files: Sequence[SkillVersionFile],
    ) -> None:
        """Write one version and its files, or raise `SkillVersionCollision`.

        `version` is minted by the caller as microseconds since the epoch, so two
        versions of one skill minted inside one microsecond collide on the key. The
        collision is raised rather than absorbed: overwriting would rewrite a version
        something may already have resolved, and skipping would answer 201 for a
        version that was never written. The caller mints the next microsecond.

        The version row and its files are one write. Half a bundle is a skill whose
        `SKILL.md` names files that are not there, which is the failure versions exist
        to end.
        """
        ...

    async def page_skill_versions(
        self, tenant_id: TenantId, skill_id: SkillId, after: int | None, limit: int
    ) -> tuple[SkillVersionRecord, ...]:
        """One page of this skill's live versions, newest first.

        Retired versions are absent. The listing answers "what can this skill be
        resolved to", and a retirement is the statement that this one no longer can.

        Newest first because the reason to read the list is to find what to go back to.
        `after` is the version of the last row the caller holds, which is a total order
        by itself: a version is unique within a skill, so no tiebreak is needed and a
        page boundary can neither repeat a row nor drop one.

        A `limit` the store will not serve is refused rather than clamped, for the
        reason `page_uploaded_skills` refuses one: a clamped page is a short page, and
        a short page is how this method says the walk is over.
        """
        ...

    async def read_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionRecord | None:
        """One version, retired or not, or None when this skill has no such version.

        A retired version answers rather than reading as absent, and its tombstone
        comes back with it. A version can be pinned -- an agent definition carries a
        digest over the set it resolved -- so the history naming it has to stay
        readable; what a retirement stops is resolution into a new Session.
        """
        ...

    async def read_skill_version_bundle(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionBundle | None:
        """One version with its whole content, for a caller downloading the archive.

        Separate from `read_skill_version` because this reads every file body and that
        one reads none. A listing that fetched bodies would move a whole bundle out of
        the database in order to publish four columns of it.
        """
        ...

    async def retire_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> None:
        """Record that this version may no longer resolve, keeping the first moment.

        Idempotent for the reason `delete_skill` is: a retried retirement is the same
        retirement, and a second row would date it to the retry.
        """
        ...

    async def set_repository_skills(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        skills: Sequence[ValidatedSkill],
    ) -> int:
        """Record this checkout's skill directory, and say how many rows it wrote.

        Idempotent per `(tenant, repository, revision)`: a revision is immutable, so a
        second submission of the same commit is a retry rather than an edit, and it
        writes nothing and reports 0.
        """
        ...

    async def read_repository_skills(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> tuple[ValidatedSkill, ...]:
        """This checkout's skills, ordered by name; empty when none was submitted."""
        ...


class UnconfiguredSkills:
    """A skill store that refuses every call, for a platform assembled without one.

    Every method raises, and that is the safe direction rather than the inconvenient
    one. A store that answered "no skills" would make an unwired platform
    indistinguishable from a tenant who attached none: every definition would resolve,
    every Session would start, and every agent would quietly have no skills -- which is
    the exact failure this whole module exists to end.
    """

    async def add_skill(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        skill: ValidatedSkill,
        *,
        display_name: str | None,
    ) -> None:
        raise self._unconfigured()

    async def read_skills(
        self, tenant_id: TenantId, skill_ids: Sequence[SkillId]
    ) -> tuple[SkillRecord, ...]:
        raise self._unconfigured()

    async def page_uploaded_skills(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillListing, ...]:
        raise self._unconfigured()

    async def set_repository_skills(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        skills: Sequence[ValidatedSkill],
    ) -> int:
        raise self._unconfigured()

    async def read_repository_skills(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> tuple[ValidatedSkill, ...]:
        raise self._unconfigured()

    async def read_skill(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> SkillHeld | None:
        raise self._unconfigured()

    async def delete_skill(self, tenant_id: TenantId, skill_id: SkillId) -> None:
        raise self._unconfigured()

    async def add_skill_version(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        version: int,
        skill: ValidatedSkill,
        directory: str,
        files: Sequence[SkillVersionFile],
    ) -> None:
        raise self._unconfigured()

    async def page_skill_versions(
        self, tenant_id: TenantId, skill_id: SkillId, after: int | None, limit: int
    ) -> tuple[SkillVersionRecord, ...]:
        raise self._unconfigured()

    async def read_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionRecord | None:
        raise self._unconfigured()

    async def read_skill_version_bundle(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionBundle | None:
        raise self._unconfigured()

    async def retire_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> None:
        raise self._unconfigured()

    @staticmethod
    def _unconfigured() -> RuntimeError:
        return RuntimeError(
            "no skill store is wired into this platform, so no skill can be uploaded "
            "or resolved; a Session started now would run an agent with none of the "
            "skills its definition names"
        )


async def read_attached(
    store: SkillStore, tenant_id: TenantId, definition: AgentDefinition
) -> tuple[ValidatedSkill, ...]:
    """The skills this definition attaches by id, refusing one the tenant does not hold.

    Not treated as "then attach nothing": the definition asserts the skill is there,
    and the honest answer to a false assertion is a refusal naming the id. An id that
    does not resolve is the whole of the accept-it-and-deliver-nothing failure, moved
    up one level.

    Separate from `resolve` because it is the half whose answer is permanent. A stored
    skill is immutable and never deleted, so an id that resolves when a definition is
    registered still resolves when a Session starts -- which is what makes this the
    right check to run at the registration door, where the tenant is still on the
    connection. None of `resolve`'s other refusals can be settled that early, and each
    for the same kind of reason: a repository submission may arrive after the definition
    does, and a version may be written or retired after it too.

    A definition that attaches nothing is answered without asking the store anything.
    Not an optimisation -- there is no question to ask, and a store consulted about an
    empty set of ids could only answer emptily. It is also what keeps a definition that
    names no skill working on a platform with no skill store wired, which is every
    definition written before this existed, while one that *does* attach a skill still
    fails loudly there rather than resolving quietly to nothing.
    """
    return tuple(
        record.skill for record in await _attached_records(store, tenant_id, definition)
    )


async def _attached_records(
    store: SkillStore, tenant_id: TenantId, definition: AgentDefinition
) -> tuple[SkillRecord, ...]:
    """The records this definition attaches, with their ids, refusing a missing one.

    The ids come back because resolution needs them: which version a skill delivers is
    a property of the attachment, and the attachment names the skill by id. Kept
    separate from `read_attached` so the refusal for an id nobody uploaded is written
    once -- the registration door and the Session door have to give the same answer to
    the same wrong id, and two copies of that check are two answers waiting to differ.

    **An attachment id addresses the upload door only.** A repository skill has an id
    of its own now, so this needs saying rather than being left to be inferred from
    which table happens to be read. Attaching one by id would mean the same skill could
    arrive twice at once -- once by id, once because the definition pins the commit that
    brought it -- and the merge would report that as two skills colliding on one name,
    which is false and unfixable by the tenant. The commit is how that route is
    attached: pinning it delivers the checkout's whole skill directory, which is the
    unit that route submits in.

    So the refusal below says what an id attaches and how the other route is attached,
    and deliberately does *not* say the tenant holds no such skill. That claim would be
    false for the id of a skill their own CI submitted, and a platform contradicting its
    own listing is worse than one that answers narrowly. Telling the two apart would
    take a repository-skill lookup through this port, and the id of a skill that this
    port cannot read is not a distinction it can honestly draw.
    """
    wanted = _attached_ids(definition)
    if not wanted:
        return ()
    records = await store.read_skills(tenant_id, sorted(wanted))
    if len(records) != len(wanted):
        missing = sorted(wanted - {record.skill_id for record in records})
        raise SkillsUnresolvable(
            f"{len(missing)} attached skill id(s) name no skill this platform can "
            "attach, so this definition names a skill that would never reach the "
            "agent. An id attaches a skill uploaded with POST /v1/skills; a skill a "
            "repository submission brought is attached by pinning its commit in "
            "skills_repository and skills_revision, which delivers that checkout's "
            "whole skill directory",
            skill_id=str(missing[0]),
            unresolved_skills=len(missing),
        )
    return records


async def resolve(
    store: SkillStore, tenant_id: TenantId, definition: AgentDefinition
) -> tuple[SkillFile, ...]:
    """Every file that delivers this definition's skills, or a refusal saying why not.

    Reads both routes and merges them, and this is what a Session is started from.

    An attached skill resolves to a *version* -- either the one the attachment pins or
    the greatest one nobody has retired -- and a version is a whole directory rather
    than a single document. That is why this is the one place the sibling files a
    `SKILL.md` names actually reach an agent: the document says to read `forms.md` and
    this is what puts `forms.md` beside it.

    A skill with no versions at all resolves to its own stored body as the single
    `SKILL.md`. Every skill uploaded before versions existed has none, so refusing them
    would take every agent already running off the air; the fallback is what lets this
    change ship rather than a convenience.

    A repository skill has no versions and cannot have any -- it is addressed by the
    commit it was submitted from, and a commit is already a version -- so that route
    still resolves to one document per skill.

    Every refusal below is a Session that would otherwise start with a skill missing,
    or with a skill the caller did not ask for. They arrive in three groups. The
    attachments are refused for an id nobody uploaded and for one skill attached at two
    versions. A version is refused for not existing, for having been retired, and for
    holding a file at a path that would not land under its own directory. The merged set
    is refused for holding too many skills, two skills of one name, two files of one
    delivery entry, or more bytes than the Secret carries.

    None of them can be decided any earlier than here, which is why they are here rather
    than at the registration door: versions are written and retired after a definition
    is registered, two names collide only once both have been looked up, and the
    parse-time bound on `skills` sees the attached ones while a repository brings more.

    A deleted skill still resolves, and that is the delete route's decision rather than
    an oversight here: it publishes that a definition which attached a skill keeps
    working, for the same reason a Session's history stays readable. Nothing in this
    function reads `deleted`.
    """
    delivered = [
        await _deliver_attached(store, tenant_id, record, pin)
        for record, pin in await _attached_with_pins(store, tenant_id, definition)
    ]
    from_repository = await store.read_repository_skills(
        tenant_id, definition.skills_repository, definition.skills_revision
    )
    return _merge(delivered, from_repository)


@dataclass(frozen=True, slots=True)
class _Delivered:
    """One resolved skill: the name it is delivered under, and every file of it.

    `name` is not always the name the skill was uploaded under. A version's document
    decides its own name, so a version may rename the skill, and the delivered
    directory has to be the name the delivered `SKILL.md` declares -- the runtime
    announces the directory and reads the frontmatter, and a skill whose two disagree
    is one the runtime refuses to announce. It follows that the name collision below is
    over these names rather than over the store's.
    """

    name: str
    origin: str
    files: tuple[SkillFile, ...]


async def _attached_with_pins(
    store: SkillStore, tenant_id: TenantId, definition: AgentDefinition
) -> tuple[tuple[SkillRecord, SkillVersionPin], ...]:
    """Each attached skill paired with the version its attachment asks for.

    Ordered by the store's own order, which is by name, so the same definition resolves
    to the same bytes on every read.

    One skill attached twice at two versions is refused rather than merged. The parse
    keeps both -- they are genuinely two different grants and not a repeated one, which
    is why deduplicating them there would be wrong -- but both would be delivered into
    `skills/<name>/`, so one version's files would replace the other's. It is decided
    here and not at the parse, even though a definition alone is enough to see it,
    because every other "two things at one delivery path" refusal is decided here: two
    places deciding what collides is two answers that can disagree about the same
    definition.

    Refused after the ids are read, so a definition that is wrong in both ways is told
    about the id first. That is the fact a caller can act on without knowing anything
    about versions, and it is the same answer the registration door already gave them.
    """
    records = await _attached_records(store, tenant_id, definition)
    pins: dict[SkillId, SkillVersionPin] = {}
    for attachment in definition.skills:
        skill_id = SkillId(attachment.skill_id)
        held = pins.get(skill_id)
        if held is not None and held != attachment.version:
            first, second = sorted([held.wire, attachment.version.wire])
            raise SkillsUnresolvable(
                f"skill {skill_id} is attached twice, at version {first} and at "
                f"version {second}; a skill is delivered into one directory, so the "
                "two versions' files would land on the same paths and one would "
                "silently replace the other",
                skill_id=str(skill_id),
                first_version=first,
                second_version=second,
            )
        pins[skill_id] = attachment.version
    return tuple((record, pins[record.skill_id]) for record in records)


async def _deliver_attached(
    store: SkillStore,
    tenant_id: TenantId,
    record: SkillRecord,
    pin: SkillVersionPin,
) -> _Delivered:
    """The files one attached skill delivers, at the version its attachment asks for.

    `latest` is resolved through `read_skill`, which is where the rest of this platform
    reads it from: the store derives "the greatest version nobody has retired" in one
    correlated subquery, and the skill read publishes that same number as
    `latest_version`. Deriving it a second time here -- newest first, take one, skip the
    retired -- would be a second rule free to disagree with the published one, and the
    two would disagree exactly when it mattered, on the skill whose newest version was
    just retired.

    A skill with no live version at all falls back to its stored body. That covers two
    states that are one answer: a skill uploaded before versions existed, and a skill
    every version of which has been retired. The second is the honest reading of a null
    `latest_version` -- the skill is still readable and its original body is still what
    the id was registered with.

    **A pinned version that has been retired is refused, not served.** The choice is
    forced by what a retirement already means everywhere else on this surface: the
    version listing excludes retired versions because it answers "what can this skill
    be resolved to", `latest` skips them, and the single-version read and the archive
    download both keep answering. Retired therefore means readable and not resolvable,
    and serving one here would be the one place that sentence stopped being true.

    The case for serving it is that a pin is a promise of exactly these bytes, and
    breaking it strands a definition that did nothing wrong. That case loses on who the
    retirement was aimed at. A version is retired because it regressed, and the
    definitions that pinned the regressed version are precisely the ones still getting
    it -- so serving them would make the retire route a button that changes nothing in
    its motivating case, while leaving the operator no way to stop the bad version short
    of editing every tenant's definitions. A refusal is also the recoverable direction:
    the tenant pins another version or `latest` and is running again, and the bytes are
    still downloadable in the meantime. Serving it silently is not recoverable, because
    nobody finds out.
    """
    if isinstance(pin, PinnedSkillVersion):
        version = pin.version
    else:
        held = await store.read_skill(tenant_id, record.skill_id)
        latest = None if held is None else held.latest_version
        if latest is None:
            return _Delivered(
                name=record.skill.name,
                origin="attached",
                files=skill_files([record.skill]),
            )
        version = latest
    bundle = await store.read_skill_version_bundle(tenant_id, record.skill_id, version)
    if bundle is None:
        raise SkillsUnresolvable(
            f"skill {record.skill_id} has no version {version}, and a pin is not "
            "answered with the latest version instead: a caller who pinned a version "
            "and was served another one has no way to tell, which is the whole reason "
            "to pin. Read the versions this skill has from "
            "GET /v1/skills/{id}/versions",
            skill_id=str(record.skill_id),
            version=str(version),
        )
    if bundle.record.retired:
        raise SkillsUnresolvable(
            f"version {version} of skill {record.skill_id} has been retired, so it no "
            "longer resolves into a new Session -- which is what retiring a version "
            "is for, and this definition pins the version it was aimed at. It stays "
            "readable and its archive stays downloadable; pin a version the listing "
            "still shows, or 'latest'",
            skill_id=str(record.skill_id),
            version=str(version),
        )
    return _Delivered(
        name=bundle.record.name,
        origin="attached",
        files=_version_files(bundle),
    )


def _version_files(bundle: SkillVersionBundle) -> tuple[SkillFile, ...]:
    """One version as delivered files: its document, and every file stored beside it.

    Delivered under the name the version's own document declares, not the name the
    skill was uploaded under, because the runtime announces the directory and reads the
    frontmatter -- a directory that disagreed with the name inside it is a skill the
    runtime will not announce.

    The `SKILL.md` comes from the bundle's own column and is deliberately not looked for
    among the siblings, which is the same split the store holds it in: one record of
    what the skill says, so there is no second copy to disagree.

    A sibling path is refused if it could be delivered anywhere but under this skill's
    directory. The upload door refuses these paths and so does the store, and this is
    the third check on purpose: the path is joined onto a directory inside a Session's
    workspace, and a row on its own does not say which writer produced it.
    """
    name = bundle.record.name
    files = [SkillFile(relative_path=delivery_path(name), text=bundle.skill_md)]
    for sibling in bundle.files:
        _refuse_an_undeliverable_path(sibling.path, bundle.record)
        files.append(
            SkillFile(
                relative_path=delivery_path(name, sibling.path), text=sibling.text
            )
        )
    return tuple(files)


def _refuse_an_undeliverable_path(path: str, record: SkillVersionRecord) -> None:
    """Refuse a stored sibling path that would not land under the skill's directory.

    Three ways a path fails, and they are one question asked three ways: does joining
    this onto the skill's directory produce a file inside it. An absolute path ignores
    the directory, a `..` climbs out of it, and a backslash is a separator on the other
    kind of filesystem. An empty or `.` segment is refused as well -- not for escaping,
    but because `a//b` and `./a` are second spellings of a path already in the set, and
    a set with two spellings of one path has an order-dependent answer to which file
    the model reads.
    """
    segments = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or ".." in path
        or any(segment in ("", ".") for segment in segments)
    ):
        raise SkillsUnresolvable(
            f"version {record.version} of skill {record.name!r} holds a file at "
            f"{path!r}, which is not a path it can be delivered at: a skill's files "
            "are written by joining the path onto the skill's own directory inside the "
            "session, so this one would land outside it or under a second spelling of "
            "a path already there",
            skill=record.name,
            version=str(record.version),
            path=path,
        )


def _attached_ids(definition: AgentDefinition) -> frozenset[SkillId]:
    """The ids this definition attaches, as the platform's id type.

    The attachment holds a plain `UUID` because that is what parsing a request body can
    produce; `SkillId` is the name the store is addressed by, and the conversion is a
    no-op at runtime. Doing it in one function keeps the cast out of the two places
    that would otherwise both need it and could disagree about which type they hold.
    """
    return frozenset(SkillId(attachment.skill_id) for attachment in definition.skills)


def _merge(
    attached: Sequence[_Delivered], from_repository: Sequence[ValidatedSkill]
) -> tuple[SkillFile, ...]:
    """One delivery set from the two routes, refusing a set a Session cannot carry.

    Four checks, in the order a tenant can act on them. The count first, so a definition
    that is both over the limit and ambiguous is told about the limit -- the count is
    the fact a tenant can act on without knowing what the other route brought. Then the
    name, which is the collision one flat delivery directory has always had. Then the
    paths, which is the same collision one level down and is new: a version brings a
    tree, and two files in different trees can still become one entry. Then the weight,
    last because it is the one that depends on every byte of everything else.

    Ordered by delivery path rather than by name. That is the same order for the skills
    that have one file each, and it is the only total order once a skill has several --
    so the same definition still resolves to the same bytes, which is what lets one
    Session's delivery be compared with the last one's.
    """
    delivered = [
        *attached,
        *(
            _Delivered(name=skill.name, origin="repository", files=skill_files([skill]))
            for skill in from_repository
        ),
    ]
    total = len(delivered)
    if total > MAX_SKILLS_PER_AGENT:
        raise SkillsUnresolvable(
            f"this definition resolves to {total} skills and a Session carries at "
            f"most {MAX_SKILLS_PER_AGENT}; every one of them is announced to the model "
            "on every turn, which is what the limit pays for",
            resolved_skills=total,
            limit=MAX_SKILLS_PER_AGENT,
        )
    seen: dict[str, str] = {}
    for one in delivered:
        if one.name in seen:
            raise SkillsUnresolvable(
                f"two skills resolve to the name {one.name!r}, one "
                f"{seen[one.name]} and one {one.origin}; both would be delivered to "
                f"{delivery_path(one.name)} and one would silently replace the other",
                skill=one.name,
                first_origin=seen[one.name],
                second_origin=one.origin,
            )
        seen[one.name] = one.origin
    files = sorted(
        (file for one in delivered for file in one.files),
        key=lambda file: file.relative_path,
    )
    _refuse_two_files_at_one_key(files)
    _refuse_more_than_a_secret_holds(files)
    return tuple(files)


def _refuse_two_files_at_one_key(files: Sequence[SkillFile]) -> None:
    """Refuse two delivery paths that become one entry where the files are written.

    A skill reaches a Session as an entry in a Kubernetes Secret, and a Secret key
    admits no `/`, so the separator is folded away on the way in. `a/b.md` and `a_b.md`
    are two paths the upload door accepts and one entry afterwards -- and the one that
    lost is a file the model was told to read.

    Refused here as well as where the Secret is written, which is deliberate rather
    than redundant. This side can name both paths in a message the tenant reads and can
    do it before a pod is attempted; that side is the last thing between a manifest and
    the API server. Neither is the only check, because a set assembled here is not the
    only thing that reaches that writer.
    """
    seen: dict[str, str] = {}
    for file in files:
        key = delivery_key(file.relative_path)
        first = seen.get(key)
        if first is not None:
            raise SkillsUnresolvable(
                f"{first} and {file.relative_path} are two files and one delivery "
                "entry: a skill's files reach the session as keys of a Kubernetes "
                "Secret, which cannot hold a '/', so these two become one and the "
                "second would silently replace the first",
                first_path=first,
                second_path=file.relative_path,
            )
        seen[key] = file.relative_path


def _refuse_more_than_a_secret_holds(files: Sequence[SkillFile]) -> None:
    """Refuse a delivery set too heavy for the Secret that carries it.

    The bound that replaced the count. Sixteen skills at 32 KiB was half a megabyte
    while a skill was one document; a skill now resolves to a version carrying up to
    thirty-two documents, so the count bounds sixteen megabytes and a Secret holds one.
    Counting the bodies is what actually bounds the object.

    Named in bytes, with the total in the message, because a tenant told only that the
    set is too big cannot tell whether to drop a skill or trim a file -- and the number
    is the only thing that says which.
    """
    total = sum(len(file.text.encode()) for file in files)
    if total > SKILL_DELIVERY_MAX_BYTES:
        heaviest = max(files, key=lambda file: len(file.text.encode()))
        raise SkillsUnresolvable(
            f"this definition's skills are {total} bytes together and a Session "
            f"carries at most {SKILL_DELIVERY_MAX_BYTES}; they are delivered inside "
            "one Kubernetes Secret, which is capped at 1 MiB for all of them and the "
            f"compiled configuration together. The largest is {heaviest.relative_path}",
            resolved_bytes=total,
            limit=SKILL_DELIVERY_MAX_BYTES,
            largest_path=heaviest.relative_path,
        )
