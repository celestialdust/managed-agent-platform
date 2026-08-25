"""A skill version's content on the way in and on the way out: multipart, then zip.

One module because it is one fact read two ways. A version's files arrive as multipart
parts and leave as a zip archive rooted at the version's directory, and the two have to
agree about what a path means -- a parse that admitted `a/../b` and a packer that wrote
it as an entry would put a tenant's file outside the directory the Session mounted. Held
apart from the routes because the routes decide who may ask; this decides what the bytes
are, and neither needs to know how the other refuses.

**A part's filename is its path inside the bundle**, and that is a decision rather than
a reading of the reference, which gives this endpoint no body parameters at all -- its
only evidence is a `-F files='["Example data"]'` example, a form field with no filename.
A part with no filename cannot say where its contents go, and a version whose files have
no paths cannot deliver the `forms.md` that Anthropic's published `pdf` skill tells the
model to read. So a nameless part is refused rather than stored somewhere invented.

**Every path rule here is one the store also enforces, and the duplication is the
point.** A path is joined onto a directory inside a Session's workspace, so a path that
escapes the directory escapes into the workspace -- and the filesystem cannot tell a row
this parse wrote from one written by a migration, a psql session, or a later adapter.
Refusing in both places means neither is the only thing standing between a bundle and
the disk. Where this is stricter than the store it is stricter on purpose, and the
reason is written where the rule is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Final
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from starlette.datastructures import UploadFile

from managed_agent.control.api.refusals import Refusal
from managed_agent.control.skills.registry import SkillVersionBundle, SkillVersionFile
from managed_agent.core.errors import ErrorCode
from managed_agent.core.registration.skill import (
    SKILL_FILE,
    SKILL_MD_MAX_BYTES,
    SkillMalformed,
    ValidatedSkill,
    delivery_key,
    parse_skill_md,
)

REASON_BUNDLE_INVALID: Final[str] = "skill_bundle_invalid"
"""Named in `detail` so a caller can branch on "the upload was wrong" without reading
prose. The published code set has no member for a malformed multipart bundle and a code
invented in a route is an unversioned addition to the contract, so the distinction goes
where anything branchable goes (ADR-013)."""

MAX_BUNDLE_FILES: Final[int] = 32
"""How many files one version may carry.

Chosen against the ceiling on the far end rather than picked: each file is bounded by
`SKILL_MD_MAX_BYTES`, and thirty-two of those is exactly the 1 MiB a Kubernetes Secret
holds, which is how a skill reaches a Session. A bundle that cannot be delivered is
refused at the door rather than stored and found undeliverable later.
"""

MAX_BUNDLE_PATH_CHARS: Final[int] = 256
"""How long a path inside a bundle may be. Bounded because the path is echoed in the
refusal that rejects it, and an unbounded echo is a caller-controlled response body."""

_UPLOADED_BUNDLE: Final = "the uploaded bundle"
"""What a refusal calls a document that arrived with no path we could name."""


@dataclass(frozen=True, slots=True)
class SkillBundle:
    """One parsed multipart upload: the document, where it goes, and its siblings.

    A value rather than three loose arguments, because it is what the parse produces:
    holding one is the proof that the paths were checked, the document parsed and the
    directory settled, so nothing downstream re-decides any of it.
    """

    skill: ValidatedSkill
    directory: str
    files: tuple[SkillVersionFile, ...]


def _refuse_bundle(message: str, **detail: str | int) -> Refusal:
    """One refusal for every way a bundle can be wrong, built in one place.

    Returned rather than raised so each call site reads `raise _refuse_bundle(...)`,
    which keeps the raise visible where it happens instead of hidden one frame down.
    """
    return Refusal(
        ErrorCode.REQUEST_INVALID, message, reason=REASON_BUNDLE_INVALID, **detail
    )


def _bundle_path(raw: str) -> str:
    """One part's filename as a path inside the bundle, or a refusal.

    Every rule here is the one the store's own check enforces, and the duplication is
    deliberate: the path is joined onto a directory inside a Session's workspace, so a
    row that escapes the directory escapes into the workspace -- and the filesystem
    cannot tell a row this parse wrote from one written by anything else. Refusing in
    both places means neither is the only thing standing between a bundle and the disk.

    Stricter than the store in two ways it can afford to be. An empty or `.` segment is
    refused, so `a//b` and `./a` cannot arrive as two spellings of one path that the
    primary key would treat as different files. And a control character is refused,
    because a path is written into an archive entry and read back by a shell.
    """
    if not raw:
        raise _refuse_bundle(
            "one of the uploaded parts carries no filename, and the filename is the "
            "path the file is delivered at; send each file as its own part with the "
            "path it should have inside the skill directory"
        )
    if len(raw) > MAX_BUNDLE_PATH_CHARS:
        raise _refuse_bundle(
            f"a path inside a bundle is at most {MAX_BUNDLE_PATH_CHARS} characters",
            path_length=len(raw),
        )
    if raw.startswith("/") or "\\" in raw or ".." in raw:
        raise _refuse_bundle(
            f"{raw!r} is not a path this bundle can carry: it is delivered by joining "
            "the path onto a directory inside the session's workspace, so an absolute "
            "path ignores that directory and a '..' climbs out of it",
            path=raw[:MAX_BUNDLE_PATH_CHARS],
        )
    if any(segment in ("", ".") for segment in raw.split("/")):
        raise _refuse_bundle(
            f"{raw!r} has an empty or '.' path segment, which is a second spelling of "
            "another path in the same bundle; send each file at one path",
            path=raw[:MAX_BUNDLE_PATH_CHARS],
        )
    if any(character < " " or character == "\x7f" for character in raw):
        raise _refuse_bundle(
            "a path inside a bundle carries no control character; it is written into "
            "an archive entry and read back by a shell"
        )
    return raw


async def _text_of(upload: UploadFile, path: str) -> str:
    """One part's contents as text, or a refusal naming what it could not be read as.

    Text and not bytes, because a skill's files are what the model reads and they are
    delivered as entries in a Kubernetes Secret. So a part that is not UTF-8 is refused
    rather than stored base64-encoded and delivered as something the model cannot read;
    the accept-it-and-deliver-nothing failure this whole slice exists to remove.

    A NUL is refused separately even though it survives UTF-8, because Postgres will not
    hold one in a text column: without this the refusal would arrive as a driver error
    through the 500 handler, naming nothing the caller could fix.
    """
    raw = await upload.read()
    if len(raw) > SKILL_MD_MAX_BYTES:
        raise _refuse_bundle(
            f"{path!r} is {len(raw)} bytes and a file in a skill bundle may be at most "
            f"{SKILL_MD_MAX_BYTES}; the bundle is delivered inside one Kubernetes "
            "Secret, which is capped at 1 MiB for all of it",
            path=path,
            byte_length=len(raw),
        )
    if not raw.strip():
        raise _refuse_bundle(
            f"{path!r} is empty, and a file the model is told to read must have "
            "something in it",
            path=path,
        )
    try:
        text = raw.decode()
    except UnicodeDecodeError as undecodable:
        raise _refuse_bundle(
            f"{path!r} is not UTF-8 text; a skill's files are delivered as entries in "
            "a Kubernetes Secret and read by the model, so a file that is not text "
            "cannot be delivered as one",
            path=path,
        ) from undecodable
    if "\x00" in text:
        raise _refuse_bundle(
            f"{path!r} carries a NUL byte, which no text column can hold",
            path=path,
        )
    return text


def _rooted(read: Mapping[str, str]) -> tuple[str | None, Mapping[str, str]]:
    """The bundle's top-level directory and its paths relative to it, or a refusal.

    None means the paths are flat and the caller has to supply the directory from
    somewhere else; `_parse_bundle` takes the skill's own name there.

    A bundle mixing a rooted path with a flat one is refused rather than guessed at,
    and so is one naming two roots. Either could be read two ways -- flatten everything,
    or invent a root -- and both readings change where the model looks for the file. A
    caller who zipped one directory sends one root, which is what the reference means by
    "the top-level directory name that was extracted from the uploaded files".
    """
    roots = {path.split("/", 1)[0] for path in read if "/" in path}
    flat = sorted(path for path in read if "/" not in path)
    if roots and flat:
        raise _refuse_bundle(
            f"this bundle has {flat[0]!r} at the top level and {sorted(roots)[0]!r} as "
            "a directory; send either one directory holding every file, or every file "
            "at the top level, so there is one answer to where the skill lives",
            top_level_file=flat[0],
        )
    if len(roots) > 1:
        raise _refuse_bundle(
            f"this bundle has {len(roots)} top-level directories "
            f"({', '.join(sorted(roots)[:2])}); a version is one skill directory",
            directory_count=len(roots),
        )
    if not roots:
        return None, read
    root = roots.pop()
    return root, {path[len(root) + 1 :]: text for path, text in read.items()}


def _refuse_colliding_deliveries(entries: Mapping[str, str]) -> None:
    """Refuse a bundle holding two paths that become one file once delivered.

    A skill's files reach a Session as entries in a Kubernetes Secret, whose keys hold
    no `/`, so `delivery_key` folds the separator away -- and that fold is not
    injective. `a/b.md` and `a_b.md` are two distinct paths in this bundle and one key
    in the Secret, so delivering both writes one over the other and the model reads
    whichever arrived last. Neither path is malformed on its own, which is why no rule
    above catches this: it is a property of the pair.

    **The fold is imported and the refusal is local, and that split is deliberate.** How
    a path becomes a key is one fact and lives in one place, so this cannot drift from
    the encoding it is predicting. Whether a set of paths is acceptable is checked here
    *and* where the Secret is built, for the reason every path rule in this module is
    checked twice -- see the note at the top. What this side buys is the message: both
    paths are still in hand, so the refusal names them. By the time the manifest is
    assembled the only remaining fact is that one key was written twice, and a tenant
    reading that has to work backwards to find out which two files they sent.

    Checked over the paths *relative to the version's directory*, because those are what
    the store holds and what delivery joins onto it. The directory is a single segment
    by construction, so it contributes the same prefix to every key and can neither
    bring two distinct relative paths together nor keep two colliding ones apart.
    """
    claimed: dict[str, str] = {}
    for path in sorted(entries):
        key = delivery_key(path)
        if key in claimed:
            raise _refuse_bundle(
                f"{claimed[key]!r} and {path!r} are two paths in this bundle and one "
                "file once delivered: a skill's files arrive as entries in a "
                "Kubernetes Secret, whose keys hold no '/', so both collapse to "
                f"{key!r} and one would silently replace the other",
                path=path[:MAX_BUNDLE_PATH_CHARS],
                collides_with=claimed[key][:MAX_BUNDLE_PATH_CHARS],
            )
        claimed[key] = path


def bundle_of_document(skill_md: str, *, source: str) -> SkillBundle:
    """One bare `SKILL.md` as a bundle of one file, or the refusal a bundle would get.

    Here rather than in the route that needs it, so the JSON create door and the
    multipart one refuse a malformed document through *the same* conversion rather than
    through two that agree today. The refusal a caller receives -- its code, its
    branchable `reason`, and the sentence naming what is wrong with the frontmatter --
    is produced once, so there is no arrangement of the two doors in which one accepts a
    document the other rejects.

    `source` is the caller's, because it is what the message calls the thing that failed
    to parse and the two doors are handed different things: one a document, one a file
    inside a bundle. Naming it accurately is worth more than making the two sentences
    identical, and nothing branches on prose.

    The directory is the skill's own name. A bare document names no path and a version
    has to be filed somewhere; the name is the only answer the document itself supplies,
    and it matched `SKILL_NAME_PATTERN`, so it is a legal directory by construction.
    """
    try:
        skill = parse_skill_md(skill_md, source=source)
    except SkillMalformed as malformed:
        raise _refuse_bundle(str(malformed)) from malformed
    return SkillBundle(skill=skill, directory=skill.name, files=())


async def parse_bundle(uploads: Sequence[UploadFile]) -> SkillBundle:
    """Read the multipart upload into a version, or refuse it saying what is wrong.

    **Typed on Starlette's `UploadFile` and not FastAPI's, which is the wider of the
    two.** FastAPI's is a subclass carrying the validation hooks an annotation needs, so
    a route that declares `list[UploadFile]` still satisfies this; what a caller reading
    the form itself gets back from `request.form()` is the base class, and it is not an
    instance of the subclass. Narrowing to FastAPI's would have made this unusable from
    the one caller that dispatches on content type -- and nothing here touches anything
    but `filename` and `read()`, both of which the base class defines.

    In the handler rather than in a model validator, which is where `POST /v1/skills`
    puts the equivalent parse. The difference is not style: reading an upload is
    awaitable and a pydantic validator is not, so there is no annotation that could hold
    this. What is kept is the property that made the annotation worth having -- one
    parse, before any store is touched, and a refusal naming the file and the reason.

    The `SKILL.md` is read out of the bundle rather than taken as a field, so `name` and
    `description` come from inside the document the way the reference says. It is
    stored as the version's own column and deliberately not repeated among the siblings:
    one record of what the skill says, which cannot disagree with itself.
    """
    if not uploads:
        raise _refuse_bundle(
            "a version carries at least one file, and one of them is the SKILL.md; "
            "send them as multipart/form-data parts named 'files'"
        )
    if len(uploads) > MAX_BUNDLE_FILES:
        raise _refuse_bundle(
            f"this bundle has {len(uploads)} files and a version may carry at most "
            f"{MAX_BUNDLE_FILES}; they are delivered together inside one Kubernetes "
            "Secret, which is what the limit pays for",
            file_count=len(uploads),
        )
    read: dict[str, str] = {}
    for upload in uploads:
        path = _bundle_path(upload.filename or "")
        if path in read:
            raise _refuse_bundle(
                f"{path!r} arrived twice in one bundle, and which of the two the model "
                "would have read is not a question worth having an answer to",
                path=path,
            )
        read[path] = await _text_of(upload, path)
    root, entries = _rooted(read)
    _refuse_colliding_deliveries(entries)
    if SKILL_FILE not in entries:
        raise _refuse_bundle(
            f"this bundle has no {SKILL_FILE}, so nothing in it says what the skill is "
            f"called or what it does; the paths sent were {', '.join(sorted(entries))}"
        )
    try:
        skill = parse_skill_md(entries[SKILL_FILE], source=_UPLOADED_BUNDLE)
    except SkillMalformed as malformed:
        raise _refuse_bundle(str(malformed)) from malformed
    return SkillBundle(
        skill=skill,
        # A bundle-derived root is one path segment by construction -- it is what a
        # split on '/' produced -- and `_bundle_path` has already refused a '..'
        # anywhere in it. A name-derived one matched `SKILL_NAME_PATTERN`, which admits
        # no separator and no dot-dot. So both satisfy the store's own directory check
        # without a third place deciding what a legal directory is.
        directory=root if root is not None else skill.name,
        files=tuple(
            SkillVersionFile(path=path, text=text)
            for path, text in sorted(entries.items())
            if path != SKILL_FILE
        ),
    )


def archive_of(bundle: SkillVersionBundle) -> bytes:
    """One version as a zip, rooted at its directory, the same bytes every time.

    Every entry's timestamp is the version itself -- the microsecond it was written,
    truncated to the two-second resolution a zip entry holds -- rather than the moment
    of the download. So two downloads of one version produce identical archives, which
    is what lets a caller compare a digest of one against a digest of another instead of
    having to unpack both to find out whether anything changed.

    The `SKILL.md` is written from the version's own column and the siblings from their
    rows, which is the same split the store holds them in: one copy of the document, so
    the archive cannot carry two that disagree.
    """
    stamp = datetime.fromtimestamp(bundle.record.version // 1_000_000, tz=UTC)
    when = (stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second)
    root = bundle.record.directory
    entries = {f"{root}/{SKILL_FILE}": bundle.skill_md} | {
        f"{root}/{one.path}": one.text for one in bundle.files
    }
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name in sorted(entries):
            # A ZipInfo rather than `writestr(name, ...)`, which stamps the entry with
            # the current time and would make every download of one version a different
            # file. Compression is set on the info because that is where `writestr`
            # reads it from when it is handed one.
            entry = ZipInfo(filename=name, date_time=when)
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, entries[name])
    return buffer.getvalue()
