"""What a `SKILL.md` has to be, which paths hold one, and what an agent may attach.

Tier 1 (local, no infrastructure). Every property here is a property of the parse, so a
test that needed a store to grade one would be grading the store.

The refusals are the subject rather than the happy path, and each of them is a case that
would otherwise reach a pod and be indistinguishable from a skill that was never
registered: the agent reports having no skill and nothing anywhere says why. Every
refusal is asserted to name the offending skill *and* the reason, because a refusal that
names neither leaves the submitter guessing which of their files was wrong.

The two parametrized cases are collections whose members each encode a decision -- the
ways a document can fail to be a skill, and the paths that look like a skill and are
not. Written as one assertion per member so that removing a member breaks a test, which
is what a `for` loop over the same list would not do.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from managed_agent.core.registration.definition import AgentDefinition
from managed_agent.core.registration.skill import (
    DESCRIPTION_MAX_CHARS,
    MAX_SKILLS_PER_AGENT,
    SKILL_MD_MAX_BYTES,
    SKILL_NAME_MAX_CHARS,
    SKILL_ROOTS,
    LatestSkillVersion,
    PinnedSkillVersion,
    SkillAttachment,
    SkillMalformed,
    SkillType,
    delivery_key,
    delivery_path,
    parse_skill_md,
    repository_skill_id,
    scan_tree,
    skill_files,
)

_SHA = "0" * 39 + "a"

_A_VERSION = 1774000000000000
"""One version, spelled the way the platform mints them: microseconds since the epoch.

Sixteen digits, which is a moment in 2026. A shorter run of digits is a millisecond
timestamp written by mistake and is refused, which one of the cases below is about.
"""


def _skill_md(
    name: str = "pdf-report", description: str = "Build a PDF report."
) -> str:
    """A well-formed document, with the two required frontmatter entries replaced."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\nDo the thing.\n"


def _definition(**overrides: object) -> AgentDefinition:
    body: dict[str, object] = {
        "name": "slr-reviewer",
        "instructions": "Extract findings and name the source document for each.",
        "model": "gpt-5-codex",
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _SHA,
    } | overrides
    return AgentDefinition.model_validate(body)


def test_a_well_formed_skill_parses_into_a_typed_value() -> None:
    """The boundary hands back a value whose type proves the frontmatter was read."""
    skill = parse_skill_md(_skill_md(), source="upload")

    assert skill.name == "pdf-report"
    assert skill.description == "Build a PDF report."
    assert skill.text == _skill_md(), (
        "the stored text is not the submitted document byte for byte, so an optional "
        "frontmatter key this platform does not understand would be dropped on the way "
        "to a runtime that does"
    )


def test_an_unknown_frontmatter_key_is_kept_rather_than_refused() -> None:
    """The format is an open standard and its optional keys grow without us.

    The one permissive choice in the parse, and it is safe because the document is
    delivered verbatim: the runtime reads whatever it understands and this platform
    does not have to be taught each new key before a tenant may use it.
    """
    text = (
        "---\nname: pdf-report\ndescription: Build a PDF report.\n"
        "license: Apache-2.0\n---\n\nDo the thing.\n"
    )

    skill = parse_skill_md(text, source="upload")

    assert "license: Apache-2.0" in skill.text


@pytest.mark.parametrize(
    ("text", "reason_fragment", "why"),
    [
        (
            "name: pdf-report\ndescription: x\n",
            "frontmatter",
            "a document with no frontmatter block at all is refused",
        ),
        (
            "---\nname: pdf-report\ndescription: x\n",
            "never closes",
            "an unterminated frontmatter block is refused rather than read to the end",
        ),
        (
            "---\njust a string\n---\nbody\n",
            "not a set of `key: value`",
            "frontmatter that is not a mapping is refused",
        ),
        (
            "---\ndescription: x\n---\nbody\n",
            "`name`",
            "a missing name is refused, and the refusal names the missing key",
        ),
        (
            "---\nname: pdf-report\n---\nbody\n",
            "`description`",
            "a missing description is refused",
        ),
        (
            "---\nname: pdf-report\ndescription:\n---\nbody\n",
            "`description`",
            "a blank description is refused as missing rather than as a wrong type",
        ),
        (
            "---\nname: PDF-Report\ndescription: x\n---\nbody\n",
            "not usable as a skill name",
            "an uppercase name is refused: it becomes a directory and a secret key",
        ),
        (
            "---\nname: ../escape\ndescription: x\n---\nbody\n",
            "not usable as a skill name",
            "a name that could traverse out of the delivery directory is refused",
        ),
        (
            "---\nname: has space\ndescription: x\n---\nbody\n",
            "not usable as a skill name",
            "a name with a space is refused: a secret key cannot hold one",
        ),
        (
            "---\nname: pdf-report\ndescription: x\n---\n\n   \n",
            "no instructions",
            "a skill with no body is refused rather than announced and then silent",
        ),
    ],
)
def test_a_document_that_is_not_a_skill_is_refused_saying_which_and_why(
    text: str, reason_fragment: str, why: str
) -> None:
    with pytest.raises(SkillMalformed) as raised:
        parse_skill_md(text, source="the-uploaded-file")

    assert raised.value.skill == "the-uploaded-file", (
        "the refusal does not name the file, so a submitter with a tree of them cannot "
        f"tell which one was wrong: {raised.value}"
    )
    assert reason_fragment in raised.value.reason, (
        f"the refusal does not explain what is wrong: {raised.value}"
    )
    assert why  # the parametrize label is the documentation


def test_an_over_long_name_is_refused_because_a_secret_key_is_bounded() -> None:
    """The pattern says which characters and the bound says how many.

    Not cosmetic: the name is interpolated into a Kubernetes Secret key, and a key over
    253 characters is refused by the API server -- a pod that never starts, for a reason
    no tenant could see.
    """
    with pytest.raises(SkillMalformed, match="not usable as a skill name"):
        parse_skill_md(
            _skill_md(name="a" * (SKILL_NAME_MAX_CHARS + 1)), source="upload"
        )


def test_an_over_long_description_is_refused_with_the_limit_in_the_message() -> None:
    with pytest.raises(SkillMalformed) as raised:
        parse_skill_md(
            _skill_md(description="x" * (DESCRIPTION_MAX_CHARS + 1)), source="upload"
        )

    assert str(DESCRIPTION_MAX_CHARS) in raised.value.reason, (
        f"the refusal does not say what the limit is: {raised.value}"
    )


def test_an_over_large_document_is_refused_because_it_travels_in_a_secret() -> None:
    """The byte bound is a delivery constraint, and the message has to say so.

    A submitter told only "too long" will try to raise the limit. Told that skills ride
    in a Kubernetes Secret capped at 1 MiB for all of them together, they know the
    number is paid for by something.
    """
    padding = "x" * SKILL_MD_MAX_BYTES

    with pytest.raises(SkillMalformed) as raised:
        parse_skill_md(_skill_md() + padding, source="upload")

    assert "Secret" in raised.value.reason, (
        f"the refusal does not say what the size limit is paying for: {raised.value}"
    )


def test_both_documented_skill_roots_are_scanned() -> None:
    """One skill authored for each host, and both are found.

    The `SKILL.md` format is the same open format either way; only the directory
    differs. Refusing one root would refuse a correctly authored skill for naming its
    directory after the other host, which helps nobody -- and this platform sits
    between an
    Anthropic-shaped API and a codex runtime, so both arrive.
    """
    tree = {
        f"{SKILL_ROOTS[0]}/from-claude/SKILL.md": _skill_md(name="from-claude"),
        f"{SKILL_ROOTS[1]}/from-codex/SKILL.md": _skill_md(name="from-codex"),
    }

    found = scan_tree(tree)

    assert [skill.name for skill in found] == ["from-claude", "from-codex"]


def test_a_file_that_is_not_a_skill_is_passed_over_in_silence() -> None:
    """Everything in the tree that never claimed to be a skill is ignored.

    The near-miss check below is deliberately narrow for this reason: refusing a tree
    over a vendored or documentary `SKILL.md` would refuse it over a file nobody meant
    the platform to read.
    """
    tree = {
        f"{SKILL_ROOTS[0]}/pdf-report/SKILL.md": _skill_md(),
        "README.md": "# hello",
        "vendor/some-package/SKILL.md": "not even close to valid",
    }

    assert [skill.name for skill in scan_tree(tree)] == ["pdf-report"]


@pytest.mark.parametrize(
    ("path", "why"),
    [
        (
            f"{SKILL_ROOTS[0]}/SKILL.md",
            "a SKILL.md with no skill directory around it is refused",
        ),
        (
            f"{SKILL_ROOTS[0]}/tools/pdf-report/SKILL.md",
            "one nested a level too deep is refused",
        ),
        (
            "skills/pdf-report/SKILL.md",
            "a skills directory outside .claude is refused: the documented mistake",
        ),
        (
            f"packages/web/{SKILL_ROOTS[0]}/pdf-report/SKILL.md",
            "a skills directory below the repository root is refused",
        ),
    ],
)
def test_a_path_that_looks_like_a_skill_and_is_not_refuses_the_tree(
    path: str, why: str
) -> None:
    """Each of these is a tenant who believes they shipped a skill.

    Passing them over would register nothing and say nothing, which is the whole failure
    this module exists to end. Refused with the layouts that *are* scanned named in the
    message, so the submitter can see the difference rather than guess at it.
    """
    with pytest.raises(SkillMalformed) as raised:
        scan_tree({path: _skill_md()})

    assert raised.value.skill == path
    assert SKILL_ROOTS[0] in raised.value.reason, (
        "the refusal does not say which layouts are scanned, so the submitter's next "
        f"move is to guess: {raised.value}"
    )
    assert why  # the parametrize label is the documentation


def test_a_tree_with_no_skill_at_all_is_refused() -> None:
    """A submission that would register nothing was a mistake by whoever sent it."""
    with pytest.raises(SkillMalformed, match="holds no skill"):
        scan_tree({"README.md": "# hello"})


def test_a_malformed_skill_refuses_the_whole_tree_rather_than_being_dropped() -> None:
    """One bad file is not silently skipped in favour of the good ones.

    Dropping it would hand back a partial skill set that looks complete, and the agent
    would be missing exactly the skill nobody was told about.
    """
    tree = {
        f"{SKILL_ROOTS[0]}/pdf-report/SKILL.md": _skill_md(),
        f"{SKILL_ROOTS[0]}/broken/SKILL.md": "no frontmatter here",
    }

    with pytest.raises(SkillMalformed) as raised:
        scan_tree(tree)

    assert "broken" in raised.value.skill


def test_a_name_disagreeing_with_its_directory_is_refused() -> None:
    """The directory is what the runtime announces and the frontmatter is what it reads.

    Left to disagree, the two would describe different skills and only one of them could
    be delivered.
    """
    with pytest.raises(SkillMalformed) as raised:
        scan_tree(
            {f"{SKILL_ROOTS[0]}/pdf-report/SKILL.md": _skill_md(name="something-else")}
        )

    assert "pdf-report" in raised.value.reason
    assert "something-else" in raised.value.reason


def test_one_name_in_two_roots_resolves_to_the_first_root_every_time() -> None:
    """Two roots may hold one name, and the winner has to be the same answer always.

    Merging them would have to choose whose instructions to keep and announcing both
    would deliver two files to one path, so the order of `SKILL_ROOTS` decides it.
    """
    tree = {
        f"{SKILL_ROOTS[0]}/pdf-report/SKILL.md": _skill_md(description="from claude"),
        f"{SKILL_ROOTS[1]}/pdf-report/SKILL.md": _skill_md(description="from codex"),
    }

    found = scan_tree(tree)

    assert len(found) == 1
    assert found[0].description == "from claude"


def test_the_delivery_path_is_relative_and_holds_no_root() -> None:
    """No absolute root appears in this package, and that is the decision.

    The runtime scans several roots and they are not equally durable, so which one a
    Session mounts belongs with whatever builds the mount. Naming one here would put
    that decision in two files free to disagree.
    """
    path = delivery_path("pdf-report")

    assert path == "skills/pdf-report/SKILL.md"
    assert not path.startswith("/"), (
        "the delivery path is absolute, so this package has taken a decision about "
        "which of the runtime's skill roots is mounted"
    )


def test_delivery_is_ordered_by_name_so_the_same_skills_are_the_same_bytes() -> None:
    """Ordered output is what lets one Session's delivery be compared with the last."""
    a = parse_skill_md(_skill_md(name="alpha"), source="a")
    z = parse_skill_md(_skill_md(name="zulu"), source="z")

    assert [f.relative_path for f in skill_files([z, a])] == [
        "skills/alpha/SKILL.md",
        "skills/zulu/SKILL.md",
    ]


@pytest.mark.parametrize(
    ("attachment", "accepted", "why"),
    [
        (
            {"type": "custom", "skill_id": "1e8f5a3c-0000-4000-8000-000000000001"},
            True,
            "a custom skill attached by the id this platform minted resolves",
        ),
        (
            {
                "type": "custom",
                "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
                "version": "latest",
            },
            True,
            "the version field is accepted at its only honourable value",
        ),
        (
            {"type": "anthropic", "skill_id": "pdf"},
            False,
            "anthropic's pre-built catalogue is refused: we hold none of those bodies",
        ),
        (
            {"type": "anthropic", "skill_id": "1e8f5a3c-0000-4000-8000-000000000001"},
            False,
            "the anthropic type is refused on its own, not by its id failing to parse",
        ),
        (
            {
                "type": "custom",
                "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
                "version": str(_A_VERSION),
            },
            True,
            "a version the platform minted is accepted, and pins that version",
        ),
        (
            {
                "type": "custom",
                "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
                "version": "2",
            },
            False,
            "a bare ordinal is refused: no version of anything is ever named '2'",
        ),
        (
            {
                "type": "custom",
                "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
                "version": str(_A_VERSION // 1000),
            },
            False,
            "a millisecond timestamp is refused rather than pinned to 1970",
        ),
        (
            {
                "type": "custom",
                "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
                "version": "LATEST",
            },
            False,
            "the word is one spelling and not a family of them",
        ),
        (
            {"type": "plugin", "skill_id": "1e8f5a3c-0000-4000-8000-000000000001"},
            False,
            "a type neither platform defines is refused",
        ),
    ],
)
def test_each_skill_attachment_type_and_version_is_decided_one_way(
    attachment: dict[str, str], accepted: bool, why: str
) -> None:
    """Every member of the wire's `type` and `version` space, one assertion each.

    Both are collections whose members each encode a decision, so each needs its own
    assertion: a loop over the same list would keep passing after a member was dropped,
    and a dropped member is a value that starts being accepted silently.
    """
    if accepted:
        assert SkillAttachment.model_validate(attachment).type is SkillType.CUSTOM
    else:
        with pytest.raises(ValidationError):
            SkillAttachment.model_validate(attachment)
    assert why  # the parametrize label is the documentation


def test_the_anthropic_refusal_explains_itself_and_says_what_to_do_instead() -> None:
    """The documented payload gets an explanation, not a uuid parse error.

    `{"type": "anthropic", "skill_id": "pdf"}` is the literal example in Anthropic's own
    documentation, so it is the payload most likely to arrive. A caller told only that
    `pdf` is not a uuid would go looking for the wrong thing entirely; what they need to
    know is that the catalogue does not exist here and what the alternative is.
    """
    with pytest.raises(ValidationError) as raised:
        SkillAttachment.model_validate({"type": "anthropic", "skill_id": "pdf"})

    message = str(raised.value)
    assert "pre-built" in message, (
        f"the refusal does not say what type 'anthropic' names: {message}"
    )
    assert "POST /v1/skills" in message, (
        f"the refusal does not say how to attach the skill instead: {message}"
    )
    assert "codex" in message, (
        "the refusal does not say why this platform holds none of those bodies: "
        f"{message}"
    )


_TENANT = uuid.UUID("2b1c9d4e-0000-4000-8000-00000000000a")
_REVISION = "0" * 39 + "a"


def _migration_0028() -> Any:
    """The migration that backfilled repository-skill ids, loaded from its source.

    Loaded by path rather than imported: a migration's module name starts with a digit,
    so no import statement can name it, and `migrations/` is not a package on the path.
    Nothing in it runs -- `upgrade()` is never called -- so this reads the two values
    the backfill used and no more.
    """
    path = Path("migrations/versions/0028_skill_repository_ids.py").resolve()
    spec = importlib.util.spec_from_file_location("skill_repository_ids_0028", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_repository_skills_id_is_the_one_the_migration_already_wrote() -> None:
    """The two copies of this function have to agree, so the test is the two of them.

    The migration cannot import from `src/`, so the fold is written twice, and the
    duplication is only safe while it is checked. It backfilled every repository skill
    that existed when it ran, so a drift here does not produce a wrong new id -- it
    orphans every id already in the table, and the symptom is a listing whose rows
    cannot be read back one at a time. Asserted against the migration's own source
    rather than against a golden value, because a golden value copied from the same
    keyboard as the code proves only that the copy happened.
    """
    migration = _migration_0028()

    for repository, name in (
        ("git@github.com:acme/skills.git", "pdf-report"),
        ("https://example.invalid/x.git", "citation-check"),
    ):
        assert repository_skill_id(
            _TENANT, repository, _REVISION, name
        ) == migration._id_for(_TENANT, repository, _REVISION, name)


def test_a_repository_skills_id_is_a_function_of_all_four_columns() -> None:
    """Change any one of the four and the id changes, which is what keys the row.

    The tenant is in it for the reason the store's every read has a tenant term: two
    tenants submitting the same commit of the same public repository hold two skills,
    and neither may name the other's row.
    """
    base = repository_skill_id(
        _TENANT, "git@github.com:acme/skills.git", _REVISION, "pdf-report"
    )

    assert base == repository_skill_id(
        _TENANT, "git@github.com:acme/skills.git", _REVISION, "pdf-report"
    ), "the id is not stable for one key, so a row could not be read back by it"
    other_tenant = uuid.UUID("2b1c9d4e-0000-4000-8000-00000000000b")
    assert base != repository_skill_id(
        other_tenant, "git@github.com:acme/skills.git", _REVISION, "pdf-report"
    )
    assert base != repository_skill_id(
        _TENANT, "git@github.com:acme/other.git", _REVISION, "pdf-report"
    )
    assert base != repository_skill_id(
        _TENANT, "git@github.com:acme/skills.git", "1" * 40, "pdf-report"
    )
    assert base != repository_skill_id(
        _TENANT, "git@github.com:acme/skills.git", _REVISION, "citation-check"
    )


def test_two_repository_skills_cannot_be_folded_into_one_string() -> None:
    """The separator is the whole argument, so the pair it protects is the test.

    A separator one of the inputs could contain would let two distinct skills hash the
    same string, and the collision would surface as one skill shadowing another rather
    than as an error. These two keys differ only in where the boundary between
    repository and name falls, which is exactly the pair a `/` would have merged.
    """
    assert repository_skill_id(
        _TENANT, "acme/skills", _REVISION, "pdf-report"
    ) != repository_skill_id(_TENANT, "acme", _REVISION, "skills")


def test_a_pinned_version_comes_out_as_a_value_that_names_which_version() -> None:
    """The parse produces the version, not the string it was written as.

    This is the whole difference between a pin that works and a field that goes
    nowhere: a resolver handed `PinnedSkillVersion(...)` cannot mistake it for the
    word `latest`, and nothing downstream has to re-read the caller's string to find
    out which of the two it meant.
    """
    attachment = SkillAttachment.model_validate(
        {
            "type": "custom",
            "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
            "version": str(_A_VERSION),
        }
    )

    assert attachment.version == PinnedSkillVersion(version=_A_VERSION)


def test_latest_comes_out_as_its_own_value_rather_than_as_a_number() -> None:
    """The two states cannot be held as one, which is why neither can be mistaken.

    `latest` is not a version and has no number; a resolver that received one would
    have to invent a sentinel for it, and every sentinel is a number that eventually
    collides with a real version.
    """
    attachment = SkillAttachment.model_validate(
        {"type": "custom", "skill_id": "1e8f5a3c-0000-4000-8000-000000000001"}
    )

    assert attachment.version == LatestSkillVersion()


def test_a_version_goes_back_onto_the_wire_as_the_string_it_arrived_as() -> None:
    """A definition is read back and re-registered, so the shape has to round trip.

    The typed pin is an internal representation. What a caller sent and what a caller
    reads back has to be the published string, or an agent read from this platform
    could not be posted back to it.
    """
    body = {
        "type": "custom",
        "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
        "version": str(_A_VERSION),
    }

    dumped = SkillAttachment.model_validate(body).model_dump(mode="json")

    assert dumped["version"] == str(_A_VERSION)
    assert SkillAttachment.model_validate(dumped).version == PinnedSkillVersion(
        version=_A_VERSION
    )


def test_a_string_no_version_is_spelled_by_is_refused_saying_what_one_is() -> None:
    """A caller who guessed the shape is told where the real ones are published.

    `1`, `v2` and a date are all things a caller reaches for when a field is called
    `version`, and none of them names anything here. The refusal has to say what a
    version actually is, or the caller's next attempt is another guess.
    """
    with pytest.raises(ValidationError) as raised:
        SkillAttachment.model_validate(
            {
                "type": "custom",
                "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
                "version": "v2",
            }
        )

    message = str(raised.value)
    assert "microseconds" in message, (
        f"the refusal does not say what a version is: {message}"
    )
    assert "/versions" in message, (
        f"the refusal does not say where the real versions are published: {message}"
    )


def test_an_over_long_version_string_is_not_echoed_back_whole() -> None:
    """The echoed value is the caller's, so the echo is bounded.

    An unbounded one puts a caller-controlled string of any length into a response
    body and into every log that holds one.
    """
    with pytest.raises(ValidationError) as raised:
        SkillAttachment.model_validate(
            {
                "type": "custom",
                "skill_id": "1e8f5a3c-0000-4000-8000-000000000001",
                "version": "9" * 4096,
            }
        )

    assert "9" * 4096 not in str(raised.value)


def test_the_same_skill_pinned_at_two_versions_is_two_attachments() -> None:
    """Two pins are two grants, where two spellings of one pin are one.

    `skills` is a frozenset, so what counts as the same attachment is decided by the
    parsed value. Two versions of one skill are genuinely two things to deliver -- and
    they collide on one delivery path, which is the merge's refusal rather than this
    one's.
    """
    one = {"type": "custom", "skill_id": "1e8f5a3c-0000-4000-8000-000000000001"}

    definition = _definition(
        skills=[
            dict(one, version=str(_A_VERSION)),
            dict(one, version=str(_A_VERSION + 1)),
            dict(one, version=str(_A_VERSION)),
        ]
    )

    assert len(definition.skills) == 2


def test_a_versions_sibling_file_is_delivered_inside_the_skills_directory() -> None:
    """`forms.md` reaches the agent at the path the `SKILL.md` tells it to read.

    A skill's document names its siblings relative to itself, so a sibling delivered
    anywhere but beside it is a document instructing the model to read a file that is
    not there -- which is the defect the whole version mechanism exists to remove.
    """
    assert delivery_path("pdf-report", "forms.md") == "skills/pdf-report/forms.md"
    assert (
        delivery_path("pdf-report", "reference/fonts.md")
        == "skills/pdf-report/reference/fonts.md"
    )
    assert delivery_path("pdf-report") == "skills/pdf-report/SKILL.md", (
        "the default subpath is no longer the SKILL.md, so every caller that delivers "
        "one skill file now has to name it"
    )


def test_two_paths_that_flatten_to_one_key_are_visible_as_one_key() -> None:
    """The collision a nested delivery path can reintroduce, made checkable.

    A file reaches a Session as an entry in a Kubernetes Secret, whose keys admit no
    `/`, so the delivery path is flattened on the way in. Two paths that differ only
    where that flattening happens are one entry, and the second would silently replace
    the first -- so the set has to be refused before it is built rather than after one
    of the files has gone missing.
    """
    assert delivery_key("skills/pdf/a/b.md") == delivery_key("skills/pdf/a_b.md")
    assert delivery_key("skills/pdf/a/b.md") != delivery_key("skills/pdf/a/c.md")


def test_a_definition_attaches_no_skill_unless_it_says_so() -> None:
    """The field is additive: every definition written before it stays valid."""
    assert _definition().skills == frozenset()


def test_the_same_skill_attached_twice_is_the_same_grant_twice() -> None:
    """Deduplicated rather than refused, for the reason `tool_servers` is a set.

    The order a tenant lists attachments in carries no meaning and a repeat is not a
    second grant, so the submission is not refused over one.
    """
    one = {"type": "custom", "skill_id": "1e8f5a3c-0000-4000-8000-000000000001"}

    definition = _definition(skills=[one, dict(one), dict(one, version="latest")])

    assert len(definition.skills) == 1


def test_more_attachments_than_a_session_can_carry_are_refused_at_the_parse() -> None:
    """The count bound is enforced where the definition arrives, not in a pod.

    A tenant learns the number instead of watching a Session fail to start, and the
    reason it exists is the Secret the skills travel in rather than an opinion about how
    many skills an agent should have.
    """
    too_many = [
        {"type": "custom", "skill_id": str(uuid.uuid4())}
        for _ in range(MAX_SKILLS_PER_AGENT + 1)
    ]

    with pytest.raises(ValidationError):
        _definition(skills=too_many)


def test_a_definition_cannot_have_its_skills_changed_after_it_is_parsed() -> None:
    """Frozen for the reason the rest of the definition is: a revision describes what
    is actually running, and a value editable after parsing would not."""
    definition = _definition(
        skills=[{"type": "custom", "skill_id": "1e8f5a3c-0000-4000-8000-000000000001"}]
    )

    with pytest.raises(ValidationError):
        definition.skills = frozenset()
