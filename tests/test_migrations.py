"""The migration runner creates the schema and takes it back down again.

Tier 1 (testcontainers, real PostgreSQL 17). This is the one path that will actually
create these tables in a deployed database, and until this file existed it was the one
path with no coverage: `tests/conftest.py` drives `upgrade()` through a hand-built
`MigrationContext` so the suite stays independent of a runner config, which leaves the
runner itself, `alembic.ini`, `migrations/env.py` and `downgrade()` all unexercised.

Driven through `alembic.command` against a `Config(root / "alembic.ini")` rather than
through a subprocess, because that is the API five later slices already call — so this
exercises the same entry point they will, not a shell equivalent of it.

Its own container, not the session-scoped one: the shared database is already migrated,
and `downgrade base` against it would drop the tables every other test is using.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.postgres import PostgresContainer

_ROOT = Path(__file__).resolve().parents[1]
_URL_ENVS = ("MAP_DATABASE_URL", "DATABASE_URL")
_URL_ENV = _URL_ENVS[0]


@pytest.fixture
def unmigrated_url() -> Iterator[str]:
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as postgres:
        yield postgres.get_connection_url()


def _config() -> Config:
    return Config(_ROOT / "alembic.ini")


def _tables(url: str) -> set[str]:
    """Table names as the database reports them, over the same driver the app uses."""

    async def _read() -> set[str]:
        engine = create_async_engine(url)
        try:
            async with engine.connect() as conn:
                return set(
                    await conn.run_sync(lambda c: sa.inspect(c).get_table_names())
                )
        finally:
            await engine.dispose()

    return asyncio.run(_read())


def test_the_runner_refuses_to_run_without_a_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No default URL, and the refusal is loud.

    A runner that falls back to a localhost default, in an environment where something
    happens to answer on 5432, migrates the wrong database and reports success.
    """
    for name in _URL_ENVS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match=_URL_ENV):
        command.current(_config())


def test_the_runner_falls_back_to_the_application_url(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The variable the deployment actually sets is enough on its own.

    Two names exist so a deployment can hand the runner a DDL-capable role without
    granting that to every request handler. Only the application's name is provisioned,
    though, so a runner that insisted on the other one would refuse to start in the
    environment it was built for -- with a message naming a variable nobody was told to
    set. Asserted against a real upgrade rather than against `_url()`, because the
    fallback has to survive the whole path from `alembic.ini` to a created table.
    """
    monkeypatch.delenv("MAP_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", unmigrated_url)

    command.upgrade(_config(), "head")
    assert "event_log" in _tables(unmigrated_url), (
        "DATABASE_URL alone did not reach the runner; a deployment setting only the "
        "variable environment.md provisions would fail on an unset one"
    )


def test_upgrade_head_then_downgrade_base_round_trips(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`upgrade head` builds the schema and `downgrade base` removes it.

    The round trip is the point rather than the upgrade alone: a `downgrade()` nothing
    has ever run is a rollback that does not work, discovered during the incident that
    needs it. Checked against `information_schema` rather than against alembic's exit
    status, because the runner reporting success is itself what is under test.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    assert "event_log" not in _tables(unmigrated_url)

    command.upgrade(_config(), "head")
    assert "event_log" in _tables(unmigrated_url), (
        "upgrade head left no event_log table"
    )

    command.downgrade(_config(), "base")
    remaining = _tables(unmigrated_url)
    assert "event_log" not in remaining, (
        f"downgrade base left event_log behind; remaining {remaining}. A downgrade "
        "that does not remove what its upgrade created is not a rollback."
    )


def test_the_schema_can_be_rebuilt_after_a_full_downgrade(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`upgrade head` still works on a database that has been taken back to base.

    The half of the round trip the test above cannot see. `drop_table` removes a table's
    triggers and indexes, so a `downgrade()` that forgets them still leaves an empty
    database and passes -- but a `CREATE FUNCTION` is a schema object of its own, and a
    downgrade that leaves one behind fails the *next* upgrade with "already exists".

    That is precisely the shape of a rollback nobody discovers until the incident: roll
    forward, hit a problem, roll back, fix the problem, roll forward again -- and the
    third step is where it breaks. Migration 0017 shipped with that defect in its first
    draft and this is the check that would have caught it.

    Asserted over the whole table set rather than one name, so it holds for whatever the
    tree grows next without anyone remembering to extend it.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)

    command.upgrade(_config(), "head")
    built = _tables(unmigrated_url)
    command.downgrade(_config(), "base")
    command.upgrade(_config(), "head")

    assert _tables(unmigrated_url) == built, (
        "rebuilding after a downgrade did not produce the same schema. A downgrade "
        "that leaves a function, a type or a rule behind fails the next upgrade rather "
        "than the one that wrote it."
    )


def test_the_payload_column_is_jsonb_and_the_sweep_column_is_indexed(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two properties that are cheap to hold now and a table rewrite to fix later.

    `jsonb` because MAP-6's markers and MAP-19's evidence references have to query
    inside this column, and `json` cannot be indexed or queried inside — it stores the
    original text and re-parses on every access. An index on `appended_at` because the
    migration that created it promises a retention sweep, and without one that sweep
    scans the whole table: its cost would track the log's total size rather than the
    rows it expires, getting slower exactly as the thing it exists to trim grows.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    command.upgrade(_config(), "head")

    async def _inspect() -> tuple[str, set[str]]:
        engine = create_async_engine(unmigrated_url)
        try:
            async with engine.connect() as conn:
                column_type = await conn.scalar(
                    sa.text(
                        "SELECT data_type FROM information_schema.columns"
                        " WHERE table_name = 'event_log' AND column_name = 'payload'"
                    )
                )
                indexed = await conn.run_sync(
                    lambda c: {
                        tuple(ix["column_names"] or ())
                        for ix in sa.inspect(c).get_indexes("event_log")
                    }
                )
                # `column_names` admits None for an expression index; only real
                # column names answer the question this test asks.
                return str(column_type), {
                    c for cols in indexed for c in cols if c is not None
                }
        finally:
            await engine.dispose()

    column_type, indexed_columns = asyncio.run(_inspect())
    assert column_type == "jsonb", (
        f"payload is {column_type!r}; json cannot be indexed or queried inside, and "
        "converting it once the table holds rows is a full rewrite under an ACCESS "
        "EXCLUSIVE lock"
    )
    assert "appended_at" in indexed_columns, (
        f"appended_at is not indexed (indexed columns: {sorted(indexed_columns)}); the "
        "retention sweep would scan the whole table"
    )


def test_the_revision_history_is_one_unbranched_chain() -> None:
    """Exactly one head, and every revision reachable from it.

    The guard for a defect that is invisible until deploy time and was live in this
    repository. Fifteen slices each add a table, each numbers its revision by hand,
    and the plan hands out those numbers as a global sequence -- so a slice that adds
    an unplanned migration silently claims a number a later slice was told to use. Two
    files then declare the same `revision`, or two declare the same `down_revision`,
    and alembic has two heads. `upgrade head` fails outright on the first; on the
    second it applies one branch and leaves the other's tables absent while reporting
    success.

    Read from the script directory rather than from a database, so it holds with
    nothing running and fails in CI rather than during a deploy. `iterate_revisions`
    is what makes this a reachability check and not just a count: a history could have
    one head and still leave an orphan whose parent no longer exists.
    """
    from alembic.script import ScriptDirectory

    scripts = ScriptDirectory.from_config(_config())
    heads = scripts.get_heads()
    assert len(heads) == 1, (
        f"alembic reports {len(heads)} heads: {sorted(heads)}. Two slices numbered "
        "their revision the same way, or two descend from one parent; `upgrade head` "
        "cannot apply both."
    )

    reachable = {rev.revision for rev in scripts.iterate_revisions(heads[0], "base")}
    every = {rev.revision for rev in scripts.walk_revisions()}
    orphans = every - reachable
    assert not orphans, (
        f"revisions unreachable from the single head: {sorted(orphans)}. These exist "
        "as files and will never be applied, so the tables they create are missing in "
        "every environment while nothing reports an error."
    )


def test_every_append_only_table_refuses_an_update_the_same_way() -> None:
    """One mechanism for refusing an UPDATE, across every migration in the tree.

    Fifteen slices each add a table and several are append-only, so each one answers, by
    hand, a question that was settled in migration 0001: how does an append-only table
    refuse an UPDATE. 0001 chose a `BEFORE UPDATE` trigger that raises, and said in a
    comment why not a rewrite rule -- a rule with `DO INSTEAD NOTHING` leaves the
    earlier rows correct while reporting **success** to the writer that tried to change
    them, and MAP-A45 requires the attempt to be refused rather than silently ignored.

    Nothing enforced that, and nothing could have from the outside: the two mechanisms
    differ only in what the *writer* is told, so any test asserting the stored row is
    unchanged passes under both. A sweep of the plan found the rule form in thirteen
    step files, seventeen occurrences, and not one step file anywhere proposing the
    trigger. The single slice that got it right is the one that deviated from its own
    step file and asked.

    So this asserts the *consistency* rather than the choice: every append-only table
    refuses an UPDATE by the same mechanism, and more than one mechanism in the tree is
    the failure. That framing is the whole point -- it fails on the **new** migration,
    which means it does not require its author to have read the old one, and not reading
    the old one is exactly what happened. Naming the trigger form directly would work
    today and would be one more decision recorded where only a reader finds it.
    """
    mechanisms: dict[str, set[str]] = {}
    for path in sorted((_ROOT / "migrations" / "versions").glob("*.py")):
        sql = path.read_text()
        if re.search(r"CREATE RULE \w+ AS ON UPDATE", sql):
            mechanisms.setdefault("rule-do-instead-nothing", set()).add(path.name)
        if re.search(r"CREATE TRIGGER \w+ BEFORE UPDATE", sql):
            mechanisms.setdefault("before-update-trigger", set()).add(path.name)

    assert mechanisms, (
        "no migration refuses an UPDATE by any mechanism. Either this tree has no "
        "append-only table yet, or every guard has been dropped -- and the second is "
        "indistinguishable from the first to this check, so it asserts presence too "
        "rather than passing vacuously on an empty result."
    )
    assert len(mechanisms) == 1, (
        "append-only tables refuse an UPDATE two different ways: "
        + "; ".join(f"{k} in {sorted(v)}" for k, v in sorted(mechanisms.items()))
        + ". Migration 0001 settled this and wrote down why: a rule with DO INSTEAD "
        "NOTHING reports success to the writer it silently ignored, which is the one "
        "thing MAP-A45 forbids. Follow whichever mechanism the existing migrations use."
    )


def test_a_migration_says_on_its_own_output_which_revisions_it_applied(
    unmigrated_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`upgrade head` names every revision it ran, on stderr, where a Job reads logs.

    This guards a defect measured in the cluster, not a hypothetical: the `map-dev`
    migration Job's pod logs were **0 bytes** after applying eleven revisions, because
    `alembic.ini` carries a complete logging configuration and `migrations/env.py`
    never loaded it. A run that migrated a production schema and a run that did nothing
    produced byte-identical output, so the only evidence of what had happened to the
    database was an exit code.

    Asserted on the revision identifiers rather than on a line count or a substring
    like "Running upgrade", because the identifiers are what a person reading a Job's
    log actually needs -- which revisions ran -- and because a formatter change should
    not fail this while a silent migration passes it. Every revision in the tree must
    appear: checking only the head would pass a run that logged its last step and
    swallowed the ten before it.
    """
    from alembic.script import ScriptDirectory

    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    expected = {
        revision.revision
        for revision in ScriptDirectory.from_config(_config()).walk_revisions()
    }
    assert expected, "no revisions in the tree, so this proves nothing"

    command.upgrade(_config(), "head")
    emitted = capsys.readouterr().err

    missing = sorted(r for r in expected if r not in emitted)
    assert not missing, (
        f"upgrade head applied {len(expected)} revisions and named "
        f"{len(expected) - len(missing)} of them on stderr; {missing} left no trace. A "
        "migration Job whose logs do not say what it did is indistinguishable from one "
        "that did nothing."
    )
