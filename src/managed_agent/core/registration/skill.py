"""What a skill is, read off a `SKILL.md`, and which paths in a tree hold one.

Nothing here reads or writes anything. A tree of paths to text goes in and typed
skills come out, or a refusal naming the skill and the reason does. That is the whole
point of the split: the moment a skill can be refused is the moment somebody submitted
it, when the submitter is still on the other end of the connection and can read why.
A skill validated any later is validated inside a pod, where a malformed one is
indistinguishable from a missing one and the only symptom is an agent that says it has
no skills.

The `SKILL.md` format is not ours. It is the open format both hosts here read -- the
frontmatter block with `name` and `description`, then instructions -- so a skill
authored for either one is readable by the other, and neither host's own directory
convention is a property of the file. That is why two roots are scanned rather than
one: `.claude/skills`, which is where the Managed Agents API discovers repository
skills, and `.agents/skills`, which is where codex looks. Refusing one of the two
would refuse a correctly authored skill for naming its directory after the other host,
which helps nobody. Both land in the same place once delivered, because the file is the
same file.

Unknown frontmatter keys are ignored rather than refused, which is the one permissive
choice in this file. The format is an open standard with optional keys (`license`,
tool allowlists) that grow without us, the text is delivered verbatim so the runtime
reads whatever it understands, and forbidding the ones we do not know would refuse
valid skills to no end. What is NOT permissive: a matched path that will not parse
refuses the whole submission rather than being skipped, and a `SKILL.md` sitting at a
path that looks like a skill and is not one refuses it too. A submission that quietly
registered nothing is the failure this module exists to make impossible.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid5

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    model_validator,
)

from managed_agent.core.ids import SkillId

SKILL_NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
"""The grammar of a skill's name, which is also a directory name and a file path.

Lowercase, starting alphanumeric, and no slash, dot-dot or space: the name is
interpolated into a delivery path and into a Kubernetes Secret key, so a name that can
traverse or that a key cannot hold is refused here rather than at the point where it
would become somebody else's malformed manifest.
"""

_NAME = re.compile(SKILL_NAME_PATTERN)

SKILL_NAME_MAX_CHARS = 128
"""The ceiling on a skill's name, which the pattern alone does not give.

The pattern says which characters; this says how many, and it is not cosmetic. The name
is interpolated into a Kubernetes Secret key on the way to a Session and a Secret key is
capped at 253 characters, so an unbounded name would produce a Secret the API server
refuses -- a pod that never starts, for a reason no tenant can see. 128 is also what the
eval gate's `SkillName` allows, and the two have to agree or a name could clear one and
be refused by the other.
"""

SKILL_FILE = "SKILL.md"
"""The one filename a skill directory is recognised by. Not ours to choose."""

SKILL_ROOTS = (".claude/skills", ".agents/skills")
"""The two repository directories scanned, in the order a listing reports them.

Order is fixed rather than incidental: two roots may hold the same skill name, and
which one wins has to be the same answer on every scan of the same tree.
"""

_SKILL_PATH = re.compile(
    r"^(?:" + "|".join(re.escape(root) for root in SKILL_ROOTS) + r")/([^/]+)/"
    r"" + re.escape(SKILL_FILE) + r"$"
)

# Skills reach a Session as entries in a Kubernetes Secret, and a Secret is capped at
# 1 MiB for all of its keys together -- the compiled `requirements.toml` included. So
# something has to bound what a definition resolves to, and the three numbers below are
# that bound. A pod whose Secret is over the limit does not start, and the reason is
# three layers below anything a tenant sees.
#
# The count and the per-file size no longer multiply into the answer. They did when a
# skill was one file: 16 skills x 32 KiB was 512 KiB, half the ceiling. A skill now
# resolves to a version, and a version carries up to 32 files of that size, so the
# product is 16 MiB -- sixteen times a Secret. The bound that actually pays for the
# delivery is therefore the total, `SKILL_DELIVERY_MAX_BYTES`, and the count now pays
# for something else, which its own docstring says.
SKILL_MD_MAX_BYTES = 32 * 1024
"""The ceiling on one skill's text, in bytes of UTF-8. See the budget above."""

SKILL_DELIVERY_MAX_BYTES = 512 * 1024
"""The ceiling on everything one Session's skills weigh together, in bytes of UTF-8.

Half the Secret's 1 MiB, and the halving is the arithmetic rather than caution. The
bodies are handed over as `stringData`, so the API server base64s them and 512 KiB of
text is about 683 KiB of object; the keys are charged for too, and 16 skills x 32
files at a 256-character path is another 128 KiB or so; `requirements.toml` is in the
same Secret. A little over 800 KiB of a 1 MiB ceiling leaves the compiled configuration
room to grow without a release quietly starting to produce pods that never schedule.

Bounded on the total rather than per skill because that is what the Secret bounds. A
per-skill ceiling large enough for a real version bundle multiplies by the count into
something no Secret holds, and one small enough to multiply safely would refuse a
single legitimate bundle.
"""

MAX_SKILLS_PER_AGENT = 16
"""The ceiling on how many skills one agent definition may attach.

No longer what keeps the Secret inside 1 MiB -- `SKILL_DELIVERY_MAX_BYTES` is, because
one skill now brings a whole version's files rather than one document. What this count
still pays for is the other cost of a skill, which bytes do not measure: every attached
skill's name and description are announced to the model on every turn, at up to
`DESCRIPTION_MAX_CHARS` each, so sixteen of them is around 16 KiB of context an agent
spends before it has read anything. It is also the only one of the two bounds a parse
can refuse, because it is a fact about the definition rather than about what the store
holds -- so a tenant learns this number on the connection instead of watching a Session
fail to start.

Far below the 500 the Managed Agents API allows, and the difference is not a policy
disagreement -- that platform mounts a filesystem and this one packs a Secret.
"""

DESCRIPTION_MAX_CHARS = 1024
"""The ceiling on a description, which is loaded into every session's context.

A description is the sentence a host matches a task against, not documentation. Hosts
truncate long ones, so a description over this length is not refused for being
expensive -- it is refused because past here the tenant and the host disagree about
what the skill says it does.
"""


class SkillMalformed(ValueError):
    """One skill that cannot be registered, and the reason in the tenant's terms.

    A `ValueError` so a pydantic validator can raise it straight into a 400 whose
    body carries this message as its published reason. `skill` is the path it was read
    from rather than its declared
    name, because a skill whose frontmatter will not parse has no name yet, and the
    path is the thing the submitter can go and open.
    """

    def __init__(self, skill: str, reason: str) -> None:
        super().__init__(f"{skill}: {reason}")
        self.skill = skill
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ValidatedSkill:
    """A skill whose `SKILL.md` was parsed, not merely accepted.

    Holding this value is the proof: there is no path to one that skipped the parse,
    which is why nothing downstream re-checks the frontmatter. `text` is the whole
    original document rather than a re-rendering of the fields, so what the runtime
    reads is byte-for-byte what the tenant wrote -- a re-render would drop every
    optional key this module deliberately does not understand.
    """

    name: str
    description: str
    text: str


def parse_skill_md(text: str, *, source: str) -> ValidatedSkill:
    """Read one `SKILL.md` into a typed skill, or refuse it saying which and why.

    `source` names the file for the refusal only; nothing about it is checked here, so
    this is also the function that grades a single uploaded document with no tree
    around it.

    The frontmatter is parsed as YAML rather than scanned line by line, because a
    description is prose and prose contains colons, quotes and commas. A hand-rolled
    reader would accept a description the runtime reads differently, which is worse
    than refusing it: the tenant and the host would disagree about what the skill says
    it does and nothing would report the disagreement.
    """
    if len(text.encode()) > SKILL_MD_MAX_BYTES:
        raise SkillMalformed(
            source,
            f"a SKILL.md may be at most {SKILL_MD_MAX_BYTES} bytes; skills are "
            "delivered to the session inside a Kubernetes Secret, which is capped at "
            "1 MiB for all of them together",
        )
    frontmatter, body = _split_frontmatter(text, source=source)
    name = _required_string(frontmatter, "name", source=source)
    if not _NAME.match(name) or len(name) > SKILL_NAME_MAX_CHARS:
        raise SkillMalformed(
            source,
            f"name {name!r} is not usable as a skill name: it becomes a directory and "
            f"a secret key on the way to the session, so it must match "
            f"{SKILL_NAME_PATTERN} and be at most {SKILL_NAME_MAX_CHARS} characters",
        )
    description = _required_string(frontmatter, "description", source=source)
    if len(description) > DESCRIPTION_MAX_CHARS:
        raise SkillMalformed(
            source,
            f"description is {len(description)} characters and the limit is "
            f"{DESCRIPTION_MAX_CHARS}; a host matches a task against this sentence "
            "and truncates a longer one, so past here it reads differently than "
            "you wrote it",
        )
    if not body.strip():
        raise SkillMalformed(
            source,
            "there are no instructions under the frontmatter, so this skill would be "
            "announced to the agent and then tell it nothing",
        )
    return ValidatedSkill(name=name, description=description, text=text)


def scan_tree(files: Mapping[str, str]) -> tuple[ValidatedSkill, ...]:
    """Every skill in a repository tree, refusing the tree if any of it is wrong.

    Returned sorted by name, so the same tree always yields the same order and the
    delivery built from it is the same bytes.

    Three ways this refuses, and each is a case that would otherwise register nothing
    and say nothing. A matched `SKILL.md` that will not parse refuses the tree rather
    than being dropped from it. A `SKILL.md` at a path with a `skills` directory
    somewhere in it that is not one of the recognised layouts refuses the tree too --
    that is the near miss (`skills/x/SKILL.md` outside `.claude`, a bare
    `.claude/skills/SKILL.md`, one nested a level too deep), and every one of them is a
    tenant who believes they shipped a skill. A tree with no skill at all is refused
    last, because a submission that registers nothing was a mistake by whoever sent it.

    A `SKILL.md` with no `skills` directory anywhere in its path is ignored in silence,
    and that is the deliberate limit of the near-miss check: a vendored or documentary
    one is not a claim to be a skill and refusing it would refuse the tree over a file
    nobody meant us to read.

    Two roots may hold the same name. The winner is the first root in `SKILL_ROOTS`,
    which is why that order is fixed -- merging them would have to choose whose
    instructions to keep, and announcing both would deliver two files to one path.

    Two passes, and the split is what makes the paragraph above true rather than
    aspirational. The near misses are refused first, over every path, so a tree
    containing one is refused whether or not the skills beside it happen to parse.
    Then the skills are collected in root precedence order -- `_root_rank` before the
    path, not the path alone, because sorting by path puts `.agents` ahead of `.claude`
    and would silently hand the tie to whichever root sorts lower rather than to the
    one this module says wins.
    """
    for path in sorted(files):
        _refuse_a_near_miss(path)
    found: dict[str, ValidatedSkill] = {}
    for path in sorted(files, key=lambda candidate: (_root_rank(candidate), candidate)):
        matched = _SKILL_PATH.match(path)
        if matched is None:
            continue
        skill = parse_skill_md(files[path], source=path)
        directory = matched.group(1)
        if skill.name != directory:
            raise SkillMalformed(
                path,
                f"the frontmatter calls this skill {skill.name!r} but its directory is "
                f"{directory!r}; the two have to agree, because the directory is what "
                "the runtime announces and the frontmatter is what it reads",
            )
        found.setdefault(skill.name, skill)
    if not found:
        raise SkillMalformed(
            "the submitted tree",
            "holds no skill: a skill is "
            + " or ".join(f"{root}/<name>/{SKILL_FILE}" for root in SKILL_ROOTS),
        )
    return tuple(found[name] for name in sorted(found))


def _root_rank(path: str) -> int:
    """Which root this path sits under, as its index in `SKILL_ROOTS`.

    A path under no root ranks past the last one, so it sorts after every real skill.
    Nothing depends on where it lands -- the collecting pass skips it -- but a total
    order keeps the sort deterministic rather than dependent on how many roots there
    happen to be.
    """
    for rank, root in enumerate(SKILL_ROOTS):
        if path.startswith(f"{root}/"):
            return rank
    return len(SKILL_ROOTS)


def _refuse_a_near_miss(path: str) -> None:
    """Refuse a `SKILL.md` that is shaped like a skill and is not one.

    The test is a `skills` path segment, which is what every documented near miss has
    and what an unrelated file does not: it catches the missing `.claude`, the missing
    skill directory and the extra nesting level with one rule, and passes over a
    `SKILL.md` that never claimed to be discoverable.

    A path that really is a skill is the first thing let through, because this runs over
    every path in the tree rather than only the leftovers -- and a recognised layout
    obviously has a `skills` segment in it.
    """
    if _SKILL_PATH.match(path) is not None:
        return
    if path.rsplit("/", 1)[-1] != SKILL_FILE or "skills" not in path.split("/"):
        return
    raise SkillMalformed(
        path,
        "is not a path a skill is loaded from, so it would be submitted and never "
        "reach the agent; the recognised layouts are "
        + " and ".join(f"{root}/<name>/{SKILL_FILE}" for root in SKILL_ROOTS),
    )


def _split_frontmatter(text: str, *, source: str) -> tuple[dict[str, object], str]:
    """The frontmatter mapping and the instructions under it.

    A missing or unterminated block is refused rather than treated as a skill with no
    metadata, because a host that cannot read a name and a description does not
    announce the skill at all -- the tenant's file is present and the agent still has
    nothing.
    """
    if not text.startswith("---\n"):
        raise SkillMalformed(
            source,
            "does not open with a `---` frontmatter block, so no host can read its "
            "name or description and none will announce it",
        )
    parts = text.split("\n---", 2)
    if len(parts) < 2:
        raise SkillMalformed(
            source, "opens a `---` frontmatter block and never closes it"
        )
    loaded = yaml.safe_load(parts[0].removeprefix("---\n"))
    if not isinstance(loaded, dict):
        raise SkillMalformed(
            source,
            "has a frontmatter block that is not a set of `key: value` entries",
        )
    return {str(key): value for key, value in loaded.items()}, parts[1]


def _required_string(
    frontmatter: Mapping[str, object], key: str, *, source: str
) -> str:
    """One frontmatter entry that has to be present, a string, and not blank.

    Blank counts as absent on purpose. `description:` with nothing after it parses as
    None and `description: ""` parses as empty, and both are a tenant who meant to
    write one -- answering "missing" tells them what to do, where "wrong type" does
    not.
    """
    value = frontmatter.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillMalformed(
            source,
            f"frontmatter has no usable `{key}`, and a skill without one is never "
            "announced to the agent",
        )
    return value.strip()


class SkillType(StrEnum):
    """Where a skill an agent attaches came from.

    Both members are the Managed Agents API's spelling, so a caller who knows that API
    writes our payload without a translation step. Only one of them resolves here --
    see `_REFUSE_ANTHROPIC`.
    """

    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


_REFUSE_ANTHROPIC = (
    "type 'anthropic' names Anthropic's pre-built skill catalogue (pdf, docx, xlsx, "
    "pptx). Those are skill bodies Anthropic ships and this platform does not hold: "
    "the runtime here is codex, and an attachment of that type would be stored and "
    "deliver nothing, which is worse than a refusal because the agent would simply "
    "report having no skill. Author the skill, upload it with POST /v1/skills, and "
    "attach the id it returns as type 'custom'"
)
"""Why the type parses and never resolves.

Refused rather than answered with a body of our own under the same id. A caller writing
`skill_id: "pdf"` means Anthropic's pdf skill, and serving different instructions under
that name would be a silent behavioural substitution -- the same class of defect as
accepting the field and doing nothing, only harder to notice. A refusal is also the
reversible direction: the day this platform ships a body for a reserved id, the id
starts resolving and nothing that worked before changes. The opposite move cannot be
undone quietly.
"""

SKILL_VERSION_LATEST = "latest"
"""The one word this field takes that is not a number, and what it means.

Not a version and never resolved to a stored one at parse time: it means "whichever
version is greatest and nobody has retired", which is a fact about the set of versions
at the moment a Session starts rather than about the definition. A definition that said
`latest` in January and starts a Session in June gets June's answer, which is the whole
reason a caller writes the word instead of a number.
"""

_VERSION_DIGITS = re.compile(r"^[0-9]{16,19}$")
"""What a version looks like on the wire.

An anchored run of ASCII digits rather than `str.isdigit`, which is true of superscripts
and other numerals that `int()` then refuses -- so the check and the conversion would
disagree, and the disagreement would surface as an unhandled error rather than as the
refusal below.
"""

# The window a version has to land in, and why it is written here rather than imported.
# A version is minted as microseconds since the epoch, so sixteen digits is 2001 and
# nineteen is what a signed 64-bit column holds. What the lower bound actually excludes
# is a *millisecond* timestamp sent by mistake: three digits shorter, so it would land
# in 1970 and sort before every real version instead of failing.
#
# `control/api/routes/skill_versions.py` asks the same question of a path segment and a
# page cursor, and answers it with the same two numbers. It is not imported because this
# module is the core of the platform and that one is a route module: importing it here
# would point the dependency from the domain at the HTTP surface, and its answer is a
# `Refusal` carrying a status code, which is not a thing a definition's parse can raise.
# The shape is published in both places and has to agree; the duplication is the price
# of the direction of the arrow.
_VERSION_MIN = 1_000_000_000_000_000
_VERSION_MAX = 2**63 - 1

_ECHOED_VERSION_CHARS = 64
"""How much of an unusable version string the refusal quotes back.

Bounded because the value is the caller's: an unbounded echo puts a caller-controlled
string of any length into a response body and into every log that holds one.
"""


@dataclass(frozen=True, slots=True)
class LatestSkillVersion:
    """The pin that names no version: whichever one is current when a Session starts.

    A value with no fields rather than a `None` or a magic string, so the two things
    this field can mean are two types. A resolver holding one cannot read it as a
    number, and nothing has to reserve a sentinel integer that a real version would
    eventually collide with.
    """

    @property
    def wire(self) -> str:
        """What this pin was written as, and is published as."""
        return SKILL_VERSION_LATEST


@dataclass(frozen=True, slots=True)
class PinnedSkillVersion:
    """One particular version, by the number the platform minted for it.

    Holding this is the proof that the caller's string was read: it is a version-shaped
    number, in the window a version can occupy. Whether that version *exists* is not a
    question this value answers -- it cannot be, because a definition is parsed once and
    versions are written and retired afterwards -- and it is answered where a Session
    resolves, which is the only place the answer is still true when it is used.
    """

    version: int

    @property
    def wire(self) -> str:
        """The decimal string this version is published as.

        A string on the wire and an int in here, which is the reference's own shape.
        The value is a key rather than a quantity: nothing does arithmetic on it, and a
        client that parsed it into a 64-bit integer would be right today and wrong the
        day the field carries anything else.
        """
        return str(self.version)


SkillVersionPin = LatestSkillVersion | PinnedSkillVersion
"""Which version of a skill an attachment resolves to: a particular one, or current.

Two types rather than one nullable field, so there is no third state to handle. Every
value of this type is a decision, and the resolver's branch on it is exhaustive by
construction.
"""


def _parse_a_version_pin(value: Any) -> Any:
    """Turn the wire's version string into the pin it names, or refuse it.

    Runs before the field is coerced, and returns whatever it cannot recognise as one
    of the two pins untouched, so a value of some other type is refused by the core
    schema with its own type error rather than by this function guessing at what the
    caller meant. An already-parsed pin passes through, which is what lets a definition
    be built in Python as well as read off a request.
    """
    if isinstance(value, LatestSkillVersion | PinnedSkillVersion):
        return value
    if not isinstance(value, str):
        return value
    if value == SKILL_VERSION_LATEST:
        return LatestSkillVersion()
    if _VERSION_DIGITS.match(value) and _VERSION_MIN <= int(value) <= _VERSION_MAX:
        return PinnedSkillVersion(version=int(value))
    raise ValueError(
        f"{value[:_ECHOED_VERSION_CHARS]!r} does not name a version of a skill. A "
        "version is the moment the platform minted it, as microseconds since the Unix "
        "epoch, so it is a run of 16 to 19 digits and never an ordinal like '2' or a "
        "date. Read the versions a skill has from GET /v1/skills/{id}/versions and "
        f"send one of those, or send {SKILL_VERSION_LATEST!r} for the greatest version "
        "nobody has retired"
    )


def _version_on_the_wire(pin: SkillVersionPin) -> str:
    """The published spelling of a pin, for the serialiser to hand back.

    A function rather than the attribute inline, because a serialiser has to be a
    callable with a declared return type for the schema to say `string`.
    """
    return pin.wire


SkillVersionField = Annotated[
    SkillVersionPin,
    BeforeValidator(_parse_a_version_pin),
    PlainSerializer(_version_on_the_wire, return_type=str),
    WithJsonSchema({"type": "string"}),
]
"""The `version` field: parsed into a pin, published as the string it arrived as.

The json schema is overridden because the field's *type* is two dataclasses and its
*wire form* is a string. Without this the published schema would describe the internal
representation, and a generated client would send an object to a field that takes a
word.
"""


class SkillAttachment(BaseModel):
    """One skill an agent definition attaches, by id.

    The wire shape is the Managed Agents API's `skills` array entry, field for field,
    so a caller who knows that API knows this one. One value does not resolve and is
    refused rather than silently reinterpreted: `type` must be `custom`.

    `version` takes what that API takes -- the word `latest`, or a version the
    platform minted -- and comes out as a `SkillVersionPin`, so which of the two a
    caller meant is a type and not a string a later reader has to interpret again.
    Whether the pinned version exists is deliberately not checked here. It cannot be:
    a definition is parsed once, versions are written and retired afterwards, and an
    answer computed now would be stale by the time a Session used it. It is checked
    where a Session resolves, which is also where a missing one can be refused with
    the skill's name in the message.

    The version's grammar is a field validator rather than a model one, and the
    difference is what the refusal says. A field validator names `version` and quotes
    the offending value; a model validator in `after` mode would only run once every
    field had already coerced, so a payload with two mistakes in it would report the
    other one and leave this field looking accepted.

    `skill_id` parses to a `UUID` because that is what this platform's ids are -- the
    same bare form `DefinitionId` already takes on the wire. It is deliberately not
    Anthropic's `skill_01...` spelling: the id is opaque to a caller either way, and
    matching the platform's own id convention matters more than matching another
    platform's cosmetics.

    The refusal of `anthropic` runs in `before` mode, over the submitted mapping, so it
    is reached before `skill_id` is coerced. In `after` mode the coercion would run
    first and `{"type": "anthropic", "skill_id": "pdf"}` -- the exact payload from
    Anthropic's own documentation, and so the most likely one to arrive -- would be
    answered with a uuid parse error instead of the explanation it needs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: SkillType
    skill_id: UUID
    version: SkillVersionField = Field(
        default=LatestSkillVersion(),
        # The default is a value, not a string, so the published schema would otherwise
        # show the dataclass's empty object as the default of a string field.
        json_schema_extra={"default": SKILL_VERSION_LATEST},
    )

    @model_validator(mode="before")
    @classmethod
    def _refuse_a_type_this_platform_cannot_deliver(cls, data: Any) -> Any:
        """Refuse `anthropic` before any field is coerced, and say what to do instead.

        Non-mapping input is passed through for the core schema to refuse with its own
        type error: this validator knows about skill catalogues, not about shapes.
        """
        if isinstance(data, Mapping) and data.get("type") == SkillType.ANTHROPIC:
            raise ValueError(_REFUSE_ANTHROPIC)
        return data


def delivery_path(name: str, subpath: str = SKILL_FILE) -> str:
    """Where one of a skill's files sits, relative to whichever skill root is mounted.

    Relative on purpose, and no absolute root appears anywhere in this package. The
    runtime scans several roots and they are not equally durable -- one of them is
    marked deprecated in the runtime's own source -- so which root a Session mounts is
    a deployment decision that belongs with whatever builds the mount. Naming one here
    would put that decision in two files free to disagree.

    The name is matched against `SKILL_NAME_PATTERN` before it gets here, which admits
    no separator, no leading dot and no `..`, so the directory this builds is always
    exactly one segment under `skills/` however the root ends up being chosen.

    `subpath` defaults to the `SKILL.md` because that is every skill's one certain
    file, and a version's sibling files are the reason it is a parameter at all: a
    document that tells the model to read `forms.md` needs `forms.md` beside it, and
    "beside" is this function. Unlike the name, a subpath is a string a tenant uploaded,
    so it is *not* trusted here -- it is refused where the delivery set is assembled,
    which is the layer that knows what the whole set contains and can therefore name
    both halves of a collision. This function joins; it does not vouch.
    """
    return f"skills/{name}/{subpath}"


_REPOSITORY_SKILL_NAMESPACE = UUID("b47ac10b-58cc-4372-a567-0e02b2c3d479")
"""The namespace repository-skill ids are minted in.

A fixed literal, pasted in once and never derived. The id a tenant read last week has to
be the id they read today, and a namespace computed from anything that moves -- a
version, a deployment, a revision of this file -- would silently re-mint every id the
next time that thing changed. It carries no meaning and is not a secret.

Deliberately not the namespace thread identifiers are minted in. Two id spaces sharing
one namespace could collide only if they hashed the same string, which they cannot, but
sharing it would mean a future change to either one's input format has to reason about
the other's.
"""


def repository_skill_id(
    tenant_id: UUID, repository: str, revision: str, name: str
) -> SkillId:
    """The id one repository skill is addressed by, from the four columns that key it.

    A repository skill arrives without an id: it is submitted as a whole checkout and
    keyed by `(tenant, repository, revision, name)`, which is why it could be resolved
    into a Session and not read, listed or retired one at a time. This is the id that
    closes that, and it is derived rather than minted at random so that the same skill
    of the same commit is the same id however many times it is submitted -- a
    resubmission is a retry, and a retry that renamed every id would break every link
    a tenant had already been handed.

    The tenant is one of the four inputs, for the reason every read of this store has a
    tenant term: two tenants submitting the same commit of the same public repository
    hold two different skills, and neither may name the other's row.

    The separator is a newline, and none of the four inputs can contain one -- a tenant
    id is a uuid, a revision is 40 hex characters, and a name matched
    `SKILL_NAME_PATTERN` before it was stored. A separator a value *could* contain would
    let two distinct skills fold into one string (`repository="a", name="b/c"` against
    `repository="a/b", name="c"` under a `/`), and the collision would surface as one
    skill shadowing another rather than as an error.

    **This is a second copy of `_id_for` in
    `migrations/versions/0028_skill_repository_ids.py`, and the namespace and the fold
    must stay byte-for-byte identical to it.** A migration cannot import from `src/`, so
    one of the two has to be a copy; what makes the copy load-bearing is that the
    migration *backfilled* every repository skill that existed when it ran. A drift here
    would therefore not produce merely a different new id -- it would orphan every id
    already in the table, and the symptom is a listing whose rows cannot be read back
    individually. `tests/core/test_skill.py` asserts the two against each other by
    loading the migration's own source, so a drift fails a test rather than a tenant.
    """
    return SkillId(
        uuid5(
            _REPOSITORY_SKILL_NAMESPACE,
            "\n".join([str(tenant_id), repository, revision, name]),
        )
    )


DELIVERY_KEY_SEPARATOR = "_"
"""What a `/` in a delivery path becomes on the way into a Kubernetes Secret key."""


def delivery_key(relative_path: str) -> str:
    """The one flat name a delivery path collapses to, for finding a collision.

    A Secret key admits `[-._a-zA-Z0-9]` and no `/`, so a nested delivery path cannot be
    a key as written and the separator is folded away. Two paths that differ only where
    that folding happens -- `a/b.md` and `a_b.md` -- are therefore one key and one file,
    and the second silently replaces the first.

    Here so the set can be refused before it is built. The adapter that writes the
    Secret folds the same way and refuses the same collision, and the duplication is
    deliberate: this side can name the two skills and the two paths in a message a
    tenant reads, and that side is the last thing standing between a manifest and the
    API server. Neither is the only check.

    Not the Secret key itself -- the prefix, the suffix and the character legality of a
    key belong to the adapter that writes one, and a second opinion about those here
    would be a second thing to keep in step. What this answers is only "do two of these
    paths become one entry".
    """
    return relative_path.replace("/", DELIVERY_KEY_SEPARATOR)


@dataclass(frozen=True, slots=True)
class SkillFile:
    """One file to place in a Session, at a path relative to the system config root.

    A pair rather than a mapping entry because the value it travels in is a frozen,
    slotted dataclass that therefore has a `__hash__`, and a `Mapping` field would make
    the whole of it unhashable.
    """

    relative_path: str
    text: str


def skill_files(skills: Sequence[ValidatedSkill]) -> tuple[SkillFile, ...]:
    """The files that deliver these skills, ordered by name.

    Ordered so the same set of skills produces the same bytes every time, which is what
    lets a Session's delivery be compared against the one before it.
    """
    return tuple(
        SkillFile(relative_path=delivery_path(skill.name), text=skill.text)
        for skill in sorted(skills, key=lambda s: s.name)
    )
