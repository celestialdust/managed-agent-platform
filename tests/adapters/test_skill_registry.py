"""Holding a skill, and the guarantees that are PostgreSQL's rather than the code's.

Tier 1 (testcontainers, real PostgreSQL 17), against the adapter rather than raw SQL
wherever the adapter is what a caller uses.

Real Postgres and not a fake, because every property here is one no stand-in could
demonstrate: the primary key that turns a resubmitted commit into a no-op instead of a
second copy, the trigger that refuses an UPDATE to a skill a Session already resolved,
and the tenant term in the WHERE clause that makes another tenant's id absent rather
than forbidden. A dict keyed by id would pass a version of all three while proving none
of them.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.skill_registry import (
    _MAX_PAGE,
    PostgresSkillRegistry,
)
from managed_agent.control.skills.registry import SkillListing
from managed_agent.core.ids import SkillId, TenantId, new_skill_id
from managed_agent.core.registration.skill import ValidatedSkill, parse_skill_md

_REPOSITORY = "git@github.com:acme/skills.git"
_SHA = "0" * 39 + "a"
_OTHER_SHA = "f" * 39 + "e"


def _skill(name: str, description: str = "Build a PDF report.") -> ValidatedSkill:
    """A parsed skill, built through the parser so its text is a real document."""
    return parse_skill_md(
        f"---\nname: {name}\ndescription: {description}\n---\n\nDo the thing.\n",
        source=f"{name}/SKILL.md",
    )


async def test_an_uploaded_skill_round_trips_byte_for_byte(engine: AsyncEngine) -> None:
    """What comes back is the document that went in, not a re-rendering of its fields.

    The whole point of storing the text rather than the parsed name and description: a
    re-render would drop every optional frontmatter key this platform deliberately does
    not understand, and the runtime that does understand them would never see them.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    skill_id = new_skill_id()
    original = parse_skill_md(
        "---\nname: pdf-report\ndescription: Build one.\nlicense: Apache-2.0\n"
        "---\n\nDo the thing.\n",
        source="upload",
    )

    await registry.add_skill(tenant, skill_id, original, display_name=None)
    read = await registry.read_skills(tenant, [skill_id])

    assert len(read) == 1
    assert read[0].skill_id == skill_id
    assert read[0].skill == original


async def test_another_tenants_skill_is_absent_rather_than_forbidden(
    engine: AsyncEngine,
) -> None:
    """The tenant is a term in the query, not a filter applied to rows that came back.

    So a caller holding an id learns nothing from the answer about whether it names
    somebody else's skill -- the same posture `definition.not_found` takes for the same
    reason.
    """
    registry = PostgresSkillRegistry(engine)
    owner = TenantId(uuid.uuid4())
    stranger = TenantId(uuid.uuid4())
    skill_id = new_skill_id()
    await registry.add_skill(owner, skill_id, _skill("pdf-report"), display_name=None)

    assert await registry.read_skills(stranger, [skill_id]) == ()
    assert len(await registry.read_skills(owner, [skill_id])) == 1


async def test_an_id_nobody_stored_is_missing_from_the_result_rather_than_raising(
    engine: AsyncEngine,
) -> None:
    """The store reports what it holds and the caller says which id is missing.

    A store that raised on the first unresolvable id could only ever name one of them,
    where the caller knows the whole set it asked for and can count.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    held = new_skill_id()
    await registry.add_skill(tenant, held, _skill("pdf-report"), display_name=None)

    read = await registry.read_skills(tenant, [held, new_skill_id()])

    assert [one.skill_id for one in read] == [held]


async def test_reading_no_ids_at_all_is_answered_without_a_query(
    engine: AsyncEngine,
) -> None:
    """A definition attaching nothing is the common case and needs no round trip."""
    assert (
        await PostgresSkillRegistry(engine).read_skills(TenantId(uuid.uuid4()), [])
        == ()
    )


async def test_two_uploads_with_one_name_are_two_skills(engine: AsyncEngine) -> None:
    """Deliberately not unique on (tenant, name), and this is what that buys.

    A stored skill is immutable, because a Session resolves once and then reads exactly
    what it resolved to. So the way to change a skill is to upload the new body and
    attach the new id -- which is a change to the definition, and gets a definition
    revision recording it. A uniqueness constraint here would make that impossible and
    force an in-place edit instead.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    first, second = new_skill_id(), new_skill_id()
    await registry.add_skill(
        tenant, first, _skill("pdf-report", "the first body"), display_name=None
    )
    await registry.add_skill(
        tenant, second, _skill("pdf-report", "the second body"), display_name=None
    )

    read = await registry.read_skills(tenant, [first, second])

    assert len(read) == 2
    assert {one.skill.description for one in read} == {
        "the first body",
        "the second body",
    }


async def test_a_stored_skill_cannot_be_edited_under_a_session_that_resolved_it(
    engine: AsyncEngine,
) -> None:
    """The append-only trigger, refused rather than silently ignored.

    A rewrite rule with DO INSTEAD NOTHING would leave the row correct while reporting
    success to the writer that tried to change it, which is the one thing the platform's
    append-only tables may not do -- so the mechanism is a BEFORE UPDATE trigger that
    raises, and this is what proves it raises.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    skill_id = new_skill_id()
    await registry.add_skill(tenant, skill_id, _skill("pdf-report"), display_name=None)

    with pytest.raises(DBAPIError) as refused:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text("UPDATE skill SET body = :body WHERE id = :id").bindparams(
                    sa.bindparam("body", type_=sa.Text()),
                    sa.bindparam("id", type_=sa.Uuid()),
                ),
                {"body": "something else entirely", "id": skill_id},
            )

    assert "append-only" in str(refused.value), str(refused.value)


async def test_a_repository_submission_reads_back_ordered_by_name(
    engine: AsyncEngine,
) -> None:
    """Ordered so the same commit always delivers the same bytes in the same order."""
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())

    written = await registry.set_repository_skills(
        tenant,
        _REPOSITORY,
        _SHA,
        [_skill("zulu"), _skill("alpha"), _skill("mike")],
    )
    read = await registry.read_repository_skills(tenant, _REPOSITORY, _SHA)

    assert written == 3
    assert [one.name for one in read] == ["alpha", "mike", "zulu"]


async def test_resubmitting_a_commit_writes_nothing_and_keeps_the_first_answer(
    engine: AsyncEngine,
) -> None:
    """The primary key is the idempotency, and the count is how a retry knows itself.

    A commit's skills do not change, so a second submission is a retried CI job rather
    than an edit. The first body stands -- a submission that overwrote it could change
    what a running agent does with nothing recording that it changed.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    await registry.set_repository_skills(
        tenant, _REPOSITORY, _SHA, [_skill("pdf-report", "the original body")]
    )

    again = await registry.set_repository_skills(
        tenant, _REPOSITORY, _SHA, [_skill("pdf-report", "a different body")]
    )
    read = await registry.read_repository_skills(tenant, _REPOSITORY, _SHA)

    assert again == 0
    assert [one.description for one in read] == ["the original body"]


async def test_two_revisions_of_one_repository_hold_different_skills(
    engine: AsyncEngine,
) -> None:
    """The pin is per commit, so a later commit's skills do not reach an earlier pin.

    This is what makes a pinned revision worth pinning: a Session running the old
    revision goes on resolving the old skill set after a new commit is submitted.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    await registry.set_repository_skills(
        tenant, _REPOSITORY, _SHA, [_skill("pdf-report")]
    )
    await registry.set_repository_skills(
        tenant, _REPOSITORY, _OTHER_SHA, [_skill("pdf-report"), _skill("new-skill")]
    )

    assert [
        one.name
        for one in await registry.read_repository_skills(tenant, _REPOSITORY, _SHA)
    ] == ["pdf-report"]
    assert [
        one.name
        for one in await registry.read_repository_skills(
            tenant, _REPOSITORY, _OTHER_SHA
        )
    ] == ["new-skill", "pdf-report"]


async def test_another_tenants_repository_submission_is_not_visible(
    engine: AsyncEngine,
) -> None:
    """Two tenants may name the same repository URL and the same commit.

    A shared open-source skills repository is the obvious case, and each tenant's
    submission is theirs: the tenant is part of the key, not a filter over one shared
    row.
    """
    registry = PostgresSkillRegistry(engine)
    owner = TenantId(uuid.uuid4())
    stranger = TenantId(uuid.uuid4())
    await registry.set_repository_skills(
        owner, _REPOSITORY, _SHA, [_skill("pdf-report")]
    )

    assert await registry.read_repository_skills(stranger, _REPOSITORY, _SHA) == ()


async def test_a_pair_nobody_submitted_for_reads_empty_rather_than_raising(
    engine: AsyncEngine,
) -> None:
    """Empty is the true answer for a repository nobody has submitted skills for.

    Every definition registered before this table existed pins one, and refusing them
    all would refuse every agent on the platform.
    """
    assert (
        await PostgresSkillRegistry(engine).read_repository_skills(
            TenantId(uuid.uuid4()), _REPOSITORY, _SHA
        )
        == ()
    )


async def test_a_repository_submission_cannot_be_edited_in_place(
    engine: AsyncEngine,
) -> None:
    """The second append-only table, refusing an UPDATE by the same mechanism."""
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    await registry.set_repository_skills(
        tenant, _REPOSITORY, _SHA, [_skill("pdf-report")]
    )

    with pytest.raises(DBAPIError) as refused:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE skill_repository_file SET body = :body"
                    " WHERE tenant_id = :tenant"
                ).bindparams(
                    sa.bindparam("body", type_=sa.Text()),
                    sa.bindparam("tenant", type_=sa.Uuid()),
                ),
                {"body": "something else entirely", "tenant": tenant},
            )

    assert "append-only" in str(refused.value), str(refused.value)


async def test_a_blank_body_is_refused_by_the_schema_and_not_only_by_the_parser(
    engine: AsyncEngine,
) -> None:
    """The check constraint is the guarantee; the parser is the good manners.

    A skill with no instructions is announced to the agent and then tells it nothing.
    The parser refuses one at the door, and this is the floor under that -- a writer
    that got past the parser some other way still cannot store one.
    """
    tenant = TenantId(uuid.uuid4())

    with pytest.raises(DBAPIError):
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO skill (id, tenant_id, name, description, body)"
                    " VALUES (:id, :tenant, 'pdf-report', 'a description', '')"
                ).bindparams(
                    sa.bindparam("id", type_=sa.Uuid()),
                    sa.bindparam("tenant", type_=sa.Uuid()),
                ),
                {"id": SkillId(uuid.uuid4()), "tenant": tenant},
            )


# Every walk below is bounded. A keyset boundary written inclusively makes a walk that
# reads pages forever, and an unbounded loop turns that into a test that hangs rather
# than one that fails.
_PAGE_BUDGET = 20


async def _walk(
    registry: PostgresSkillRegistry, tenant: TenantId, page_size: int
) -> list[SkillListing]:
    """Every row of one tenant's listing, read a page at a time, duplicates kept.

    Duplicates are kept rather than collapsed into a set, because a boundary that hands
    the same row out twice has to be able to fail the count -- a set comparison would
    absorb exactly the defect the walk exists to catch.
    """
    rows: list[SkillListing] = []
    after: tuple[str, SkillId] | None = None
    for _ in range(_PAGE_BUDGET):
        page = await registry.page_uploaded_skills(tenant, after, page_size)
        rows.extend(page)
        if len(page) < page_size:
            return rows
        after = (page[-1].name, page[-1].skill_id)
    raise AssertionError(
        f"the walk read {_PAGE_BUDGET} pages without a short one; the keyset boundary "
        "is repeating rows or never advancing"
    )


async def test_the_listing_holds_only_the_asking_tenants_uploaded_skills(
    engine: AsyncEngine,
) -> None:
    """The tenant is a term in the paging SQL, graded with two tenants in the table.

    One tenant proves nothing here. A statement with no tenant term returns exactly the
    right rows when only one tenant has uploaded anything, and every other property of
    the read -- the order, the page size, the keyset -- keeps holding. It takes a second
    tenant's row in the same table for the missing term to change an answer.
    """
    registry = PostgresSkillRegistry(engine)
    owner = TenantId(uuid.uuid4())
    stranger = TenantId(uuid.uuid4())
    await registry.add_skill(owner, new_skill_id(), _skill("mine"), display_name=None)
    await registry.add_skill(
        stranger, new_skill_id(), _skill("theirs"), display_name=None
    )

    ours = await registry.page_uploaded_skills(owner, None, 10)
    theirs = await registry.page_uploaded_skills(stranger, None, 10)

    assert [one.name for one in ours] == ["mine"]
    assert [one.name for one in theirs] == ["theirs"]


async def test_a_tenant_who_has_uploaded_nothing_reads_an_empty_page(
    engine: AsyncEngine,
) -> None:
    """Empty rather than a refusal: every tenant has a collection and most are empty."""
    assert (
        await PostgresSkillRegistry(engine).page_uploaded_skills(
            TenantId(uuid.uuid4()), None, 10
        )
        == ()
    )


async def test_the_listing_is_ordered_by_name_then_id(engine: AsyncEngine) -> None:
    """The same order `read_skills` returns, so the store has one answer, not two.

    The id after the name is not decoration. This table is deliberately not unique on
    `(tenant, name)`, so a name alone is not a position in the collection and a page
    boundary between two skills of one name could not say which one the caller held.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    twins = sorted([new_skill_id(), new_skill_id()])
    for skill_id in reversed(twins):
        await registry.add_skill(
            tenant, skill_id, _skill("pdf-report"), display_name=None
        )
    await registry.add_skill(tenant, new_skill_id(), _skill("alpha"), display_name=None)
    await registry.add_skill(tenant, new_skill_id(), _skill("zulu"), display_name=None)

    listed = await registry.page_uploaded_skills(tenant, None, 10)

    assert [one.name for one in listed] == ["alpha", "pdf-report", "pdf-report", "zulu"]
    assert [one.skill_id for one in listed[1:3]] == twins


async def test_the_keyset_walk_reads_every_uploaded_skill_exactly_once(
    engine: AsyncEngine,
) -> None:
    """One row per page, so every boundary in the collection is actually exercised.

    A page big enough to hold the whole collection has no boundary to get wrong, which
    is why the page size here is 1: an inclusive comparison repeats a row on every step
    and a comparison on the wrong column drops one.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    names = ["mike", "alpha", "zulu", "bravo", "yankee"]
    for name in names:
        await registry.add_skill(
            tenant, new_skill_id(), _skill(name), display_name=None
        )

    walked = await _walk(registry, tenant, 1)

    assert [one.name for one in walked] == sorted(names)


async def test_the_walk_steps_past_two_skills_that_share_a_name(
    engine: AsyncEngine,
) -> None:
    """Both ids come back exactly once, which is what the id in the keyset buys.

    Two uploads of one name are two skills on purpose. A boundary keyed on the name
    alone would either hand the first one out again forever or skip the second.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    both = [new_skill_id(), new_skill_id()]
    for skill_id in both:
        await registry.add_skill(
            tenant, skill_id, _skill("pdf-report"), display_name=None
        )

    walked = await _walk(registry, tenant, 1)

    assert sorted(one.skill_id for one in walked) == sorted(both)


async def test_a_repository_submission_is_not_a_row_in_the_uploaded_listing(
    engine: AsyncEngine,
) -> None:
    """The two relations do not leak into each other, which is the listing's boundary.

    A repository's skills have no id and are addressed by the `(repository, revision)`
    pair a definition pins. If they appeared here they would appear without the one
    field the collection is addressed by.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    await registry.set_repository_skills(
        tenant, _REPOSITORY, _SHA, [_skill("from-the-repository")]
    )
    await registry.add_skill(
        tenant, new_skill_id(), _skill("uploaded"), display_name=None
    )

    listed = await registry.page_uploaded_skills(tenant, None, 10)

    assert [one.name for one in listed] == ["uploaded"]


async def test_a_skills_label_is_stored_and_read_back_on_both_surfaces(
    engine: AsyncEngine,
) -> None:
    """The label survives the write and appears on the read and on the listing.

    Both surfaces, because they select it separately: the read is a column on `skill`
    and the listing is a column on the page statement, so one of them carrying it is not
    the property. The column existed for a while with no path to it -- nothing in `src/`
    mentioned it -- which is the state this asserts is over.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    skill_id = new_skill_id()

    await registry.add_skill(
        tenant, skill_id, _skill("pdf-report"), display_name="PDF tools"
    )

    held = await registry.read_skill(tenant, skill_id)
    listed = await registry.page_uploaded_skills(tenant, None, 10)
    assert held is not None
    assert held.display_name == "PDF tools"
    assert [one.display_name for one in listed] == ["PDF tools"]


async def test_a_skill_registered_without_a_label_holds_null_not_its_name(
    engine: AsyncEngine,
) -> None:
    """No label is None, and specifically not the skill's own name.

    Defaulting the label to the name would make "this skill has no label" impossible to
    represent: every row would carry one and no reader could tell an author's choice
    from a fallback. It also must not be the string "None", which is what a `str()`
    coercion over a null column produces and what a wire payload would then publish.
    """
    registry = PostgresSkillRegistry(engine)
    tenant = TenantId(uuid.uuid4())
    skill_id = new_skill_id()

    await registry.add_skill(tenant, skill_id, _skill("pdf-report"), display_name=None)

    held = await registry.read_skill(tenant, skill_id)
    assert held is not None
    assert held.display_name is None


@pytest.mark.parametrize(
    "limit", [0, -1, _MAX_PAGE + 1], ids=["zero", "negative", "past the cap"]
)
async def test_a_limit_outside_the_window_is_refused_rather_than_clamped(
    engine: AsyncEngine, limit: int
) -> None:
    """Each end of the window, because a bound on one end is not a bound.

    Clamped rather than refused, a limit past the cap would produce a page shorter than
    the caller asked for -- and a short page is how this read says the walk is over, so
    the caller would stop having seen part of what it holds and no sign that it had.
    `limit=0` clamped upward would be the same lie from the other end.

    Both the over-cap input and the expected message are derived from `_MAX_PAGE` rather
    than written out. They were literals -- `501` and `outside 1..500` -- and all three
    cases went red the moment the bound moved, including the two that had nothing to do
    with the cap: the shared `match` string is what they had in common. A test that
    pins a constant by copying it fails on the change it was meant to survive.
    """
    with pytest.raises(ValueError, match=f"outside 1..{_MAX_PAGE}"):
        await PostgresSkillRegistry(engine).page_uploaded_skills(
            TenantId(uuid.uuid4()), None, limit
        )
