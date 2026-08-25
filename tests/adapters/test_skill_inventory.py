"""One inventory over two tables, and the properties only real Postgres can show.

Tier 1 (testcontainers, real PostgreSQL 17), driven through the adapter rather than raw
SQL wherever the adapter is what a caller uses.

Real Postgres and not a fake, because every property that matters here is a property of
the statements. The keyset has to stay total across a page boundary that falls between
the two origins, which is a fact about `ORDER BY` over a `UNION ALL` and about the two
arms agreeing on column types -- a dict of rows sorted in Python would demonstrate a
version of it while proving none of it. The tenant term has to make another tenant's
skill absent rather than filtered, which is a fact about the WHERE clause. And the id
assignment has to conflict harmlessly on a resubmission, which is the unique
constraint's job and not the code's.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.skill_inventory import (
    _MAX_PAGE,
    IdAssignmentDrifted,
    PostgresSkillInventory,
)
from managed_agent.adapters.postgres.skill_registry import PostgresSkillRegistry
from managed_agent.control.skills.inventory import (
    RepositorySkillRow,
    UploadedSkillRow,
)
from managed_agent.core.ids import SkillId, TenantId, new_skill_id
from managed_agent.core.registration.skill import (
    ValidatedSkill,
    parse_skill_md,
    repository_skill_id,
)

_REPOSITORY = "git@github.com:acme/skills.git"
_OTHER_REPOSITORY = "git@github.com:acme/other-skills.git"
_SHA = "0" * 39 + "a"
_OTHER_SHA = "f" * 39 + "e"


def _skill(name: str) -> ValidatedSkill:
    """A parsed skill, built through the parser so its text is a real document."""
    return parse_skill_md(
        f"---\nname: {name}\ndescription: Build a {name} report.\n---\n\nDo it.\n",
        source=f"{name}/SKILL.md",
    )


async def _submit(
    engine: AsyncEngine, tenant: TenantId, names: list[str], revision: str = _SHA
) -> tuple[tuple[str, SkillId], ...]:
    """Put a checkout's skills in the file table and assign their ids.

    Both halves, because an id row without its file row is a state no door can produce
    and a test built on it would prove nothing about the join.
    """
    await PostgresSkillRegistry(engine).set_repository_skills(
        tenant, _REPOSITORY, revision, [_skill(name) for name in names]
    )
    return await PostgresSkillInventory(engine).assign_repository_ids(
        tenant, _REPOSITORY, revision, names
    )


async def test_a_resubmitted_checkout_gets_the_ids_it_had(engine: AsyncEngine) -> None:
    """Assigning twice returns the same ids and leaves one row per skill.

    This is the property the whole id scheme rests on. A `uuid4` here would hand an
    unchanged skill a second identity on every CI re-run, and a definition pinning the
    first would resolve to nothing while the commit it names still exists.

    The row count is asserted as well as the ids, because equal ids alone would also be
    consistent with two rows holding the same id -- which the primary key forbids, and
    which is worth reading off the table rather than trusting.
    """
    tenant = TenantId(uuid.uuid4())

    first = await _submit(engine, tenant, ["pdf", "docx"])
    second = await _submit(engine, tenant, ["pdf", "docx"])

    assert first == second
    async with engine.connect() as conn:
        held = (
            await conn.execute(
                sa.text(
                    "SELECT count(*) AS n FROM skill_repository_id"
                    " WHERE tenant_id = :tenant"
                ),
                {"tenant": tenant},
            )
        ).one()
    assert held.n == 2


async def test_two_tenants_submitting_one_commit_get_different_ids(
    engine: AsyncEngine,
) -> None:
    """The same skill of the same public repository is two rows with two ids.

    Without the tenant in the hash, one tenant's submission would compute the id another
    tenant's row already holds -- and since the id is what a read is addressed by, the
    second tenant would read the first one's skill.
    """
    one, other = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())

    mine = await _submit(engine, one, ["pdf"])
    theirs = await _submit(engine, other, ["pdf"])

    assert mine[0][1] != theirs[0][1]


async def test_a_stored_id_the_code_cannot_recompute_is_refused_not_returned(
    engine: AsyncEngine,
) -> None:
    """A drifted namespace fails loudly on the write rather than silently on every read.

    Forced by writing a row under an id nothing computes, which is exactly the state a
    changed namespace literal or a changed separator would leave behind for every skill
    migration 0028 backfilled. Without the verification read this method would return
    the id it computed, the caller would publish it, and every later lookup would answer
    "no such skill" for a row sitting in the table.

    The assertion is on the raise and on both ids appearing in the message: an operator
    reading this needs to know which id is stored and which is computed, or they cannot
    tell a namespace change from a fold change.
    """
    tenant = TenantId(uuid.uuid4())
    await PostgresSkillRegistry(engine).set_repository_skills(
        tenant, _REPOSITORY, _SHA, [_skill("pdf")]
    )
    stranded = SkillId(uuid.uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO skill_repository_id"
                " (skill_id, tenant_id, repository, revision, name)"
                " VALUES (:id, :tenant, :repository, :revision, 'pdf')"
            ),
            {
                "id": stranded,
                "tenant": tenant,
                "repository": _REPOSITORY,
                "revision": _SHA,
            },
        )

    with pytest.raises(IdAssignmentDrifted) as refused:
        await PostgresSkillInventory(engine).assign_repository_ids(
            tenant, _REPOSITORY, _SHA, ["pdf"]
        )

    message = str(refused.value)
    assert str(stranded) in message
    assert str(repository_skill_id(tenant, _REPOSITORY, _SHA, "pdf")) in message


async def test_an_empty_submission_asks_the_database_nothing(
    engine: AsyncEngine,
) -> None:
    """No names is answered without a round trip, and writes nothing.

    Asserted on the table rather than only on the return value, because an empty tuple
    would also come back from a statement that ran and matched nothing.
    """
    tenant = TenantId(uuid.uuid4())

    assigned = await PostgresSkillInventory(engine).assign_repository_ids(
        tenant, _REPOSITORY, _SHA, []
    )

    assert assigned == ()
    async with engine.connect() as conn:
        held = (
            await conn.execute(
                sa.text(
                    "SELECT count(*) AS n FROM skill_repository_id"
                    " WHERE tenant_id = :tenant"
                ),
                {"tenant": tenant},
            )
        ).one()
    assert held.n == 0


async def test_a_repository_skill_is_read_back_by_the_id_it_was_assigned(
    engine: AsyncEngine,
) -> None:
    """The read resolves the id to the file row, body and checkout included.

    The body is the field this read exists for -- the listing deliberately does not
    carry one -- and the checkout is what a definition pins, so both are asserted rather
    than just the name.
    """
    tenant = TenantId(uuid.uuid4())
    (_, skill_id), *_ = await _submit(engine, tenant, ["pdf"])

    held = await PostgresSkillInventory(engine).repository_skill_at(tenant, skill_id)

    assert held is not None
    assert held.name == "pdf"
    assert held.repository == _REPOSITORY
    assert held.revision == _SHA
    assert "Do it." in held.body


async def test_another_tenants_id_reads_as_absent_rather_than_forbidden(
    engine: AsyncEngine,
) -> None:
    """A foreign id is None, and the tenant is a term in the query rather than a filter.

    None and not a refusal, because a refusal that distinguished "not yours" from "no
    such skill" would answer a question about somebody else's data. The id is real and
    resolvable -- for its owner -- which is what makes this the case a filter applied
    after the fetch would get wrong.
    """
    owner, stranger = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())
    (_, skill_id), *_ = await _submit(engine, owner, ["pdf"])
    inventory = PostgresSkillInventory(engine)

    assert await inventory.repository_skill_at(owner, skill_id) is not None
    assert await inventory.repository_skill_at(stranger, skill_id) is None


async def test_the_listing_carries_both_origins_as_their_own_types(
    engine: AsyncEngine,
) -> None:
    """An upload and a repository skill come back on one page, typed by origin.

    The two doors were separate collections and the uploaded one was all a tenant could
    list, so a team whose CI submitted its skills could see none of them. One page over
    both is the whole point, and the types are what let a caller tell which operations
    apply without inspecting for a missing field.
    """
    tenant = TenantId(uuid.uuid4())
    await PostgresSkillRegistry(engine).add_skill(
        tenant, new_skill_id(), _skill("aardvark"), display_name=None
    )
    await _submit(engine, tenant, ["zebra"])

    page = await PostgresSkillInventory(engine).page(tenant, None, 10)

    assert [type(row) for row in page] == [UploadedSkillRow, RepositorySkillRow]
    assert [row.name for row in page] == ["aardvark", "zebra"]
    uploaded, from_repository = page
    assert isinstance(from_repository, RepositorySkillRow)
    assert from_repository.repository == _REPOSITORY
    assert from_repository.revision == _SHA
    assert uploaded.origin == "upload"
    assert from_repository.origin == "repository"


async def test_a_page_boundary_between_the_origins_repeats_nothing_and_drops_nothing(
    engine: AsyncEngine,
) -> None:
    """The keyset stays total where the page break falls between the two tables.

    This is the failure this port exists to prevent, and it is the one a caller merging
    two pages itself gets wrong: walked one origin at a time, the tail of whichever side
    ran out first goes missing, and the caller sees a short page and stops. Names are
    chosen so the sort interleaves the origins rather than grouping them, so a walk that
    silently switched tables at a boundary would be visible in the order.

    Every page is size two against six rows, so the walk crosses a boundary three times
    and at least one of those falls mid-origin.
    """
    tenant = TenantId(uuid.uuid4())
    registry = PostgresSkillRegistry(engine)
    for name in ["alpha", "gamma", "epsilon"]:
        await registry.add_skill(
            tenant, new_skill_id(), _skill(name), display_name=None
        )
    await _submit(engine, tenant, ["beta", "delta", "zeta"])
    inventory = PostgresSkillInventory(engine)

    walked: list[str] = []
    after: tuple[str, SkillId] | None = None
    while True:
        page = await inventory.page(tenant, after, 2)
        if not page:
            break
        walked.extend(row.name for row in page)
        after = (page[-1].name, page[-1].skill_id)

    assert walked == ["alpha", "beta", "delta", "epsilon", "gamma", "zeta"]
    assert len(walked) == len(set(walked))


async def test_the_listing_holds_no_other_tenants_rows(engine: AsyncEngine) -> None:
    """A second tenant's skills of both origins are absent from the page.

    Both origins, because the tenant term is applied in two places -- once per arm of
    the union -- and one of them being right is not the property.
    """
    mine, theirs = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())
    registry = PostgresSkillRegistry(engine)
    await registry.add_skill(
        mine, new_skill_id(), _skill("mine-uploaded"), display_name=None
    )
    await _submit(engine, mine, ["mine-from-ci"])
    await registry.add_skill(
        theirs, new_skill_id(), _skill("theirs-uploaded"), display_name=None
    )
    await _submit(engine, theirs, ["theirs-from-ci"])

    page = await PostgresSkillInventory(engine).page(mine, None, 10)

    assert sorted(row.name for row in page) == ["mine-from-ci", "mine-uploaded"]


async def test_two_submissions_of_one_skill_at_two_commits_are_two_rows(
    engine: AsyncEngine,
) -> None:
    """The same skill name at two revisions lists twice, with different ids.

    The revision is part of the identity, so a repository skill is not one row that
    moves as commits land -- it is a row per commit, which is what lets a definition
    pinning an older revision keep resolving after a newer one is submitted.
    """
    tenant = TenantId(uuid.uuid4())
    (_, older), *_ = await _submit(engine, tenant, ["pdf"], revision=_SHA)
    (_, newer), *_ = await _submit(engine, tenant, ["pdf"], revision=_OTHER_SHA)

    page = await PostgresSkillInventory(engine).page(tenant, None, 10)

    assert older != newer
    assert [row.name for row in page] == ["pdf", "pdf"]
    assert sorted(str(row.skill_id) for row in page) == sorted([str(older), str(newer)])


@pytest.mark.parametrize(
    "limit", [0, -1, _MAX_PAGE + 1], ids=["zero", "negative", "past the cap"]
)
async def test_a_limit_outside_the_window_is_refused_rather_than_clamped(
    engine: AsyncEngine, limit: int
) -> None:
    """Each end of the window, because a bound on one end is not a bound.

    Clamped rather than refused, a limit past the cap would produce a page shorter than
    the caller asked for -- and a short page is how this read says the walk is over, so
    the caller would stop having seen part of what it holds with no sign that it had.

    Both the over-cap input and the expected message derive from `_MAX_PAGE`, so moving
    the bound does not turn this red for a reason that has nothing to do with the bound.
    """
    with pytest.raises(ValueError, match=f"outside 1..{_MAX_PAGE}"):
        await PostgresSkillInventory(engine).page(TenantId(uuid.uuid4()), None, limit)


async def test_an_id_row_whose_file_row_is_missing_lists_nothing(
    engine: AsyncEngine,
) -> None:
    """A half-written pair contributes no row rather than a row with no body.

    Unreachable through the doors -- `assign_repository_ids` is called after the files
    are written -- but reachable by a partial failure between the two, and this pins
    which way it fails. An inner join drops it; an outer join would have produced a
    listing row for a skill with no description, which reads as a corrupted skill rather
    than as one that was never finished.
    """
    tenant = TenantId(uuid.uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO skill_repository_id"
                " (skill_id, tenant_id, repository, revision, name)"
                " VALUES (:id, :tenant, :repository, :revision, 'orphan')"
            ),
            {
                "id": repository_skill_id(tenant, _OTHER_REPOSITORY, _SHA, "orphan"),
                "tenant": tenant,
                "repository": _OTHER_REPOSITORY,
                "revision": _SHA,
            },
        )

    page = await PostgresSkillInventory(engine).page(tenant, None, 10)

    assert page == ()
