"""The store, not the application, is what keeps a bundle inside the directory it names.

Tier 1 (testcontainers, real PostgreSQL 17). Asserted with raw SQL rather than through
the adapter, because the point of each of these is that the guarantee survives a writer
that never loads our code -- a psql session, a later slice's adapter, a migration
somebody writes in a hurry.

**The two path checks are the ones that matter most and they are checked first.** A
version's files are written into the skill's directory inside a Session's workspace, so
a path that escapes that directory escapes into the workspace, and a directory that is
not one segment relocates the whole bundle. `skill_bundles.py` refuses both at the
parse; these are what hold for a row the parse never saw, and the filesystem cannot tell
the two writers apart.

The last few cases go through `PostgresSkillRegistry` rather than raw SQL, and they are
here rather than beside the adapter's other tests because what they grade belongs to
this schema: the adapter turns a duplicate key into `SkillVersionCollision` so the route
can retry, and turns a *check* violation into nothing at all so the route cannot retry
it forever. Both readings depend on the driver reporting a SQLSTATE, which is a property
of asyncpg against these constraints and not of the adapter's code.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.skill_registry import PostgresSkillRegistry
from managed_agent.control.skills.registry import (
    SkillVersionCollision,
    SkillVersionFile,
)
from managed_agent.core.ids import SkillId, TenantId
from managed_agent.core.registration.skill import ValidatedSkill

_MICROSECOND = 1_759_178_010_641_129
"""The sample version from the reference, used wherever the value itself is not
the subject. Sixteen digits, which is what makes it microseconds and not
milliseconds."""

_UPLOAD = sa.text(
    "INSERT INTO skill (id, tenant_id, name, description, body)"
    " VALUES (:id, :tenant, 'pdf', 'Build a report.', 'body')"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()), sa.bindparam("tenant", type_=sa.Uuid())
)

_WRITE_VERSION = sa.text(
    "INSERT INTO skill_version"
    " (skill_id, version, name, description, body, directory)"
    " VALUES (:id, :version, 'pdf', 'Build a report.', 'body', :directory)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("version", type_=sa.BigInteger()),
    sa.bindparam("directory", type_=sa.Text()),
)

_WRITE_FILE = sa.text(
    "INSERT INTO skill_version_file (skill_id, version, path, body)"
    " VALUES (:id, :version, :path, 'how to do it')"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("version", type_=sa.BigInteger()),
    sa.bindparam("path", type_=sa.Text()),
)

_RETIRE = sa.text(
    "INSERT INTO skill_version_retirement (skill_id, version) VALUES (:id, :version)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()), sa.bindparam("version", type_=sa.BigInteger())
)

_DELETE_SKILL = sa.text(
    "INSERT INTO skill_deletion (skill_id) VALUES (:id)"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_COUNT_VERSIONS = sa.text(
    "SELECT count(*) FROM skill_version WHERE skill_id = :id"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))

_COUNT_RETIREMENTS = sa.text(
    "SELECT count(*) FROM skill_version_retirement WHERE skill_id = :id"
).bindparams(sa.bindparam("id", type_=sa.Uuid()))


async def _upload(engine: AsyncEngine) -> tuple[uuid.UUID, uuid.UUID]:
    """One skill row, so a version has something to hang off.

    Returns the pair every statement below needs: the tenant, then the skill.
    """
    tenant, skill_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(_UPLOAD, {"id": skill_id, "tenant": tenant})
    return tenant, skill_id


async def _write_version(
    engine: AsyncEngine,
    skill_id: uuid.UUID,
    version: int = _MICROSECOND,
    directory: str = "pdf",
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            _WRITE_VERSION,
            {"id": skill_id, "version": version, "directory": directory},
        )


async def _counted(
    engine: AsyncEngine, statement: sa.TextClause, id_: uuid.UUID
) -> int:
    async with engine.connect() as conn:
        return int(await conn.scalar(statement, {"id": id_}) or 0)


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../escaped.md", "notes/../../escaped.md", "/", ".."],
)
async def test_a_file_path_that_leaves_the_directory_is_refused_by_the_store(
    engine: AsyncEngine, path: str
) -> None:
    """The check that holds when the parse is not the writer.

    Each of these is joined onto the version's directory inside a Session's workspace,
    so each one names a file outside the subtree the Session mounted. The application
    refuses them too; this is what stands between a row nobody parsed and that
    filesystem, and there is nothing else in the path.
    """
    _, skill_id = await _upload(engine)
    await _write_version(engine, skill_id)

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                _WRITE_FILE,
                {"id": skill_id, "version": _MICROSECOND, "path": path},
            )


async def test_a_blank_file_path_or_body_is_refused(engine: AsyncEngine) -> None:
    """An empty path names the directory itself; an empty body is a file to read that
    says nothing."""
    _, skill_id = await _upload(engine)
    await _write_version(engine, skill_id)

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                _WRITE_FILE, {"id": skill_id, "version": _MICROSECOND, "path": ""}
            )
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO skill_version_file (skill_id, version, path, body)"
                    " VALUES (:id, :version, 'forms.md', '')"
                ).bindparams(
                    sa.bindparam("id", type_=sa.Uuid()),
                    sa.bindparam("version", type_=sa.BigInteger()),
                ),
                {"id": skill_id, "version": _MICROSECOND},
            )


@pytest.mark.parametrize("directory", ["pdf/nested", "..", "a..b", "", "/pdf"])
async def test_a_directory_that_is_not_one_path_segment_is_refused(
    engine: AsyncEngine, directory: str
) -> None:
    """The directory is joined onto a workspace path, so it is one segment and nothing
    else.

    A separator or a `..` in it would place the whole extracted bundle wherever the
    caller chose, which is a larger escape than any single file's path can manage: every
    file in the version goes with it.
    """
    _, skill_id = await _upload(engine)

    with pytest.raises(IntegrityError):
        await _write_version(engine, skill_id, directory=directory)


async def test_a_version_shorter_than_sixteen_digits_is_refused(
    engine: AsyncEngine,
) -> None:
    """What this actually catches is a millisecond timestamp written by mistake.

    Those are three digits shorter, so they land in 1970 and would sort before every
    real version -- a wrong `latest_version` rather than a failed write, which is the
    kind of defect that surfaces as "the rollback picked the wrong body".
    """
    _, skill_id = await _upload(engine)
    milliseconds = _MICROSECOND // 1_000

    with pytest.raises(IntegrityError):
        await _write_version(engine, skill_id, version=milliseconds)

    await _write_version(engine, skill_id, version=_MICROSECOND)
    assert await _counted(engine, _COUNT_VERSIONS, skill_id) == 1


async def test_two_versions_at_one_microsecond_are_refused_by_the_key(
    engine: AsyncEngine,
) -> None:
    """The collision the route retries. Refused rather than overwriting, which is the
    half that has to be right: an overwrite would rewrite a version something already
    resolved."""
    _, skill_id = await _upload(engine)
    await _write_version(engine, skill_id)

    with pytest.raises(IntegrityError):
        await _write_version(engine, skill_id)

    assert await _counted(engine, _COUNT_VERSIONS, skill_id) == 1


async def test_one_path_twice_in_one_version_is_refused_by_the_key(
    engine: AsyncEngine,
) -> None:
    """Two rows at one path would leave the model reading whichever came back first."""
    _, skill_id = await _upload(engine)
    await _write_version(engine, skill_id)
    async with engine.begin() as conn:
        await conn.execute(
            _WRITE_FILE,
            {"id": skill_id, "version": _MICROSECOND, "path": "forms.md"},
        )

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                _WRITE_FILE,
                {"id": skill_id, "version": _MICROSECOND, "path": "forms.md"},
            )


async def test_a_version_or_a_file_or_a_retirement_of_nothing_is_refused(
    engine: AsyncEngine,
) -> None:
    """Three foreign keys, each making a row about nothing impossible.

    Without them the tables would hold a version of a skill that was never uploaded, a
    file belonging to a version that was never written, and a retirement of a version
    that does not exist -- and every read joining against them would silently skip it.
    """
    _, skill_id = await _upload(engine)
    await _write_version(engine, skill_id)

    with pytest.raises(IntegrityError):
        await _write_version(engine, uuid.uuid4())
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                _WRITE_FILE,
                {"id": skill_id, "version": _MICROSECOND + 1, "path": "forms.md"},
            )
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_RETIRE, {"id": skill_id, "version": _MICROSECOND + 1})


async def test_retiring_a_version_twice_is_one_row_and_a_refusal(
    engine: AsyncEngine,
) -> None:
    """What lets the adapter's ON CONFLICT DO NOTHING mean "already retired".

    The key is the pair, so a second retirement is a conflict rather than a second row
    -- which is what keeps the recorded moment the first one's instead of the retry's.
    """
    _, skill_id = await _upload(engine)
    await _write_version(engine, skill_id)
    async with engine.begin() as conn:
        await conn.execute(_RETIRE, {"id": skill_id, "version": _MICROSECOND})

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_RETIRE, {"id": skill_id, "version": _MICROSECOND})

    assert await _counted(engine, _COUNT_RETIREMENTS, skill_id) == 1


async def test_deleting_a_skill_twice_is_one_row_and_a_refusal(
    engine: AsyncEngine,
) -> None:
    """Same shape as the retirement, and for the same reason: the moment must not move.

    A second row would date the deletion to the retry, so a caller retrying a timeout
    would move the answer to "when did this data go" every time they asked.
    """
    _, skill_id = await _upload(engine)
    async with engine.begin() as conn:
        await conn.execute(_DELETE_SKILL, {"id": skill_id})

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_DELETE_SKILL, {"id": skill_id})
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(_DELETE_SKILL, {"id": uuid.uuid4()})


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE skill_version SET body = 'edited' WHERE skill_id = :id",
        "UPDATE skill_version_file SET body = 'edited' WHERE skill_id = :id",
        "UPDATE skill_version_retirement SET retired_at = now() WHERE skill_id = :id",
        "UPDATE skill_deletion SET deleted_at = now() WHERE skill_id = :id",
    ],
)
async def test_an_update_raises_rather_than_being_quietly_ignored(
    engine: AsyncEngine, statement: str
) -> None:
    """Refused loudly, which is the mechanism this whole tree settled on in 0001.

    A rewrite rule that did nothing would leave the stored row correct and report
    success to whoever tried to change it -- the failure found a month later by somebody
    wondering why their edit never took.
    """
    _, skill_id = await _upload(engine)
    await _write_version(engine, skill_id)
    async with engine.begin() as conn:
        await conn.execute(
            _WRITE_FILE,
            {"id": skill_id, "version": _MICROSECOND, "path": "forms.md"},
        )
        await conn.execute(_RETIRE, {"id": skill_id, "version": _MICROSECOND})
        await conn.execute(_DELETE_SKILL, {"id": skill_id})

    with pytest.raises(DBAPIError) as raised:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(statement).bindparams(sa.bindparam("id", type_=sa.Uuid())),
                {"id": skill_id},
            )

    assert "append-only" in str(raised.value)


async def test_the_adapter_reports_a_duplicate_version_as_a_collision(
    engine: AsyncEngine,
) -> None:
    """The translation the route's retry depends on, against the real driver.

    Read off the SQLSTATE the driver reports rather than off the exception class, which
    also covers the check constraints. If asyncpg stopped reporting one, the route would
    answer 500 for a case it is built to survive -- and nothing above this line could
    tell.
    """
    tenant, skill_id = await _upload(engine)
    store = PostgresSkillRegistry(engine)
    skill = ValidatedSkill(name="pdf", description="Build a report.", text="body")
    written = (
        TenantId(tenant),
        SkillId(skill_id),
        _MICROSECOND,
        skill,
        "pdf",
        (SkillVersionFile(path="forms.md", text="how to do it"),),
    )
    await store.add_skill_version(*written)

    with pytest.raises(SkillVersionCollision):
        await store.add_skill_version(*written)


async def test_the_adapter_does_not_report_a_bad_directory_as_a_collision(
    engine: AsyncEngine,
) -> None:
    """A check violation travels on as itself, because retrying it would never succeed.

    Reported as a collision it would cost the route its whole retry budget and answer
    500 for a bundle that should have been refused at the parse -- sixteen writes and no
    explanation. This is the discrimination the SQLSTATE buys.
    """
    tenant, skill_id = await _upload(engine)
    store = PostgresSkillRegistry(engine)
    skill = ValidatedSkill(name="pdf", description="Build a report.", text="body")

    with pytest.raises(IntegrityError):
        await store.add_skill_version(
            TenantId(tenant), SkillId(skill_id), _MICROSECOND, skill, "pdf/nested", ()
        )


async def test_the_adapter_reads_latest_as_the_greatest_live_version(
    engine: AsyncEngine,
) -> None:
    """Derived over the versions nobody retired, which is what makes retiring the newest
    a usable move rather than a way to break the skill.

    Also the read that has to answer for a skill this tenant does not hold: absent
    rather than forbidden, so a caller holding an id learns nothing from the answer.
    """
    tenant, skill_id = await _upload(engine)
    store = PostgresSkillRegistry(engine)
    await _write_version(engine, skill_id, version=_MICROSECOND)
    await _write_version(engine, skill_id, version=_MICROSECOND + 1)

    held = await store.read_skill(TenantId(tenant), SkillId(skill_id))
    assert held is not None
    assert held.latest_version == _MICROSECOND + 1
    assert held.deleted is False

    await store.retire_skill_version(
        TenantId(tenant), SkillId(skill_id), _MICROSECOND + 1
    )
    await store.delete_skill(TenantId(tenant), SkillId(skill_id))

    after = await store.read_skill(TenantId(tenant), SkillId(skill_id))
    assert after is not None
    assert after.latest_version == _MICROSECOND
    assert after.deleted is True
    assert await store.read_skill(TenantId(uuid.uuid4()), SkillId(skill_id)) is None


async def test_the_adapter_reads_a_version_back_with_its_files(
    engine: AsyncEngine,
) -> None:
    """The round trip the archive is built from, and the listing that excludes a
    retirement.

    The bundle read is the one place a file body leaves the database, so it is also the
    only place the split between the document and its siblings is observable: the
    `SKILL.md` comes back as the version's own column and never as one of the files.
    """
    tenant, skill_id = await _upload(engine)
    store = PostgresSkillRegistry(engine)
    skill = ValidatedSkill(name="pdf", description="Build a report.", text="# pdf\n")
    await store.add_skill_version(
        TenantId(tenant),
        SkillId(skill_id),
        _MICROSECOND,
        skill,
        "pdf",
        (
            SkillVersionFile(path="reference.md", text="the reference"),
            SkillVersionFile(path="forms.md", text="the forms"),
        ),
    )

    bundle = await store.read_skill_version_bundle(
        TenantId(tenant), SkillId(skill_id), _MICROSECOND
    )

    assert bundle is not None
    assert bundle.skill_md == "# pdf\n"
    assert [one.path for one in bundle.files] == ["forms.md", "reference.md"]
    assert bundle.record.directory == "pdf"

    page = await store.page_skill_versions(
        TenantId(tenant), SkillId(skill_id), None, 10
    )
    assert [row.version for row in page] == [_MICROSECOND]
    await store.retire_skill_version(TenantId(tenant), SkillId(skill_id), _MICROSECOND)
    assert (
        await store.page_skill_versions(TenantId(tenant), SkillId(skill_id), None, 10)
        == ()
    )
    still = await store.read_skill_version(
        TenantId(tenant), SkillId(skill_id), _MICROSECOND
    )
    assert still is not None and still.retired is True
