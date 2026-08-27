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
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.postgres import PostgresContainer

from managed_agent.core.vocabulary import WEBHOOK_ELIGIBLE

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


def _write(url: str, statements: list[tuple[str, dict[str, object]]]) -> None:
    """Run statements against a database part-way up the chain.

    Textual and unhelpfully raw on purpose: what these insert is a row in the shape a
    *previous* revision stored, and the adapters only know how to write the current one.
    """

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                for sql, params in statements:
                    await conn.execute(sa.text(sql), params)
        finally:
            await engine.dispose()

    asyncio.run(_run())


def _read(url: str, sql: str, params: dict[str, object]) -> object:
    async def _run() -> object:
        engine = create_async_engine(url)
        try:
            async with engine.connect() as conn:
                return (await conn.execute(sa.text(sql), params)).scalar_one()
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def test_a_registration_written_before_the_rename_still_matches_its_events(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rename moves the column and leaves what is in it, which is the whole problem.

    A registration stored before 0030 holds Session *states* -- `running`, `stopped` --
    and after the rename those sit in a column the sweep now queries with event *type*
    names. Nothing matches, so the registration stops firing: no error, no refusal, no
    row anywhere saying a callback was owed. The tenant's evidence is an endpoint that
    went quiet, which is indistinguishable from a platform that is not delivering.

    `running` becomes two types and that is correct rather than a widening: a tenant who
    asked to hear about a Session running was told on create and on resume, because both
    folded to RUNNING. The translation is the inverse of the fold, so what they were
    subscribed to is exactly what they stay subscribed to.

    **Stops at 0030 rather than running to head**, because it grades that revision's
    translation and 0031 later takes the `session.resumed` half back out -- for a reason
    that has nothing to do with the fold. Asserting the two revisions' combined output
    here would put two decisions behind one assertion, and a failure would not say which
    of them moved. The end-to-end outcome for this same legacy row is graded by the case
    that follows.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    command.upgrade(_config(), "0029")
    registration = uuid4()
    _write(
        unmigrated_url,
        [
            (
                "INSERT INTO webhook (id, tenant_id, url, states, secret_ref)"
                " VALUES (:wid, :tid, 'https://hooks.example.com/legacy',"
                " ARRAY['running','stopped'], 'signing-legacy')",
                {"wid": registration, "tid": uuid4()},
            )
        ],
    )

    command.upgrade(_config(), "0030")

    carried = _read(
        unmigrated_url,
        "SELECT event_types FROM webhook WHERE id = :wid",
        {"wid": registration},
    )
    assert sorted(carried) == [  # type: ignore[call-overload]
        "session.created",
        "session.resumed",
        "session.stopped",
    ], (
        f"the registration now names {carried}, which the tail matches nothing "
        "against. A rename that leaves state names in a column read as event types is "
        "a subscription that has silently stopped."
    )


def test_a_subscription_to_a_type_nothing_appends_is_stripped_or_deleted(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0031, end to end, from the rows 0030 actually wrote.

    Both registrations below are written in the pre-0030 spelling because that is where
    these subscriptions came from: 0030 translated `running` into `session.created`
    **and** `session.resumed`, and `suspended` into `session.suspended`, and nothing has
    ever checked that column against the published vocabulary. So a tenant is holding a
    subscription to a type this platform will never append again, and did not ask for
    the half of it that is now dead.

    The two rows take the two different endings, and the difference is the whole case. A
    registration with something left keeps it and loses only what cannot fire. A
    registration with nothing left is removed, because the alternative is a row that
    reads back to its owner as a live callback, refuses to be registered again, and can
    never deliver -- a subscription that has silently stopped, which is the failure
    0030's own translation was written to prevent.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    command.upgrade(_config(), "0029")
    mixed, dead = uuid4(), uuid4()
    _write(
        unmigrated_url,
        [
            (
                "INSERT INTO webhook (id, tenant_id, url, states, secret_ref)"
                " VALUES (:wid, :tid, 'https://hooks.example.com/mixed',"
                " ARRAY['running','stopped'], 'signing-mixed')",
                {"wid": mixed, "tid": uuid4()},
            ),
            (
                "INSERT INTO webhook (id, tenant_id, url, states, secret_ref)"
                " VALUES (:wid, :tid, 'https://hooks.example.com/dead',"
                " ARRAY['suspended'], 'signing-dead')",
                {"wid": dead, "tid": uuid4()},
            ),
        ],
    )

    command.upgrade(_config(), "head")

    kept = _read(
        unmigrated_url,
        "SELECT event_types FROM webhook WHERE id = :wid",
        {"wid": mixed},
    )
    assert sorted(kept) == ["session.created", "session.stopped"], (  # type: ignore[call-overload]
        f"the registration now names {kept}. A tenant who asked to hear about a "
        "Session running keeps the half of that which still happens, and loses only "
        "the half nothing will ever append."
    )

    survivors = _read(
        unmigrated_url,
        "SELECT count(*) FROM webhook WHERE id = :wid",
        {"wid": dead},
    )
    assert survivors == 0, (
        "a registration whose every event type is one nothing appends was left in "
        "place. It cannot fire, it cannot be registered again, and it reads back to "
        "its owner as a callback that is coming."
    )


def test_no_stored_subscription_survives_naming_a_type_a_tenant_cannot_register(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap that made 0031 necessary, turned into something that fails out loud.

    Nothing ties `webhook.event_types` to the published vocabulary -- not the column,
    not a constraint, not the store, and the one route that parses runs in front of new
    registrations only. So a type leaving `WEBHOOK_ELIGIBLE` strands every stored
    subscription naming it: the tail no longer scans for it, the row can never fire, and
    a tenant's only evidence is an endpoint that goes quiet. Nothing reports it, which
    is how `session.resumed` sat in live registrations from 0030 until it was found by
    reading the code.

    This is the assertion that would have failed on the day the eligibility flipped, and
    it stays failing until the migration chain catches up. Read against the **live**
    registry rather than a list frozen here, which is the opposite of what the migration
    bodies do and is deliberate: a revision has to keep saying what it did on the day it
    ran, and a guard has to say what is true now, or it cannot notice the next
    retirement.

    Seeded through the pre-0030 spelling so the row arrives the way production's did --
    written by a translation, not by the door that parses.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    command.upgrade(_config(), "0029")
    _write(
        unmigrated_url,
        [
            (
                "INSERT INTO webhook (id, tenant_id, url, states, secret_ref)"
                " VALUES (:wid, :tid, 'https://hooks.example.com/every-state',"
                " ARRAY['running','suspended','stopped'], 'signing-every-state')",
                {"wid": uuid4(), "tid": uuid4()},
            )
        ],
    )

    command.upgrade(_config(), "head")

    stored = _read(
        unmigrated_url,
        "SELECT array_agg(DISTINCT name) FROM webhook, unnest(event_types) AS name",
        {},
    )
    assert stored, "no registration survived, so the assertion below proves nothing"
    unregisterable = sorted(set(stored) - WEBHOOK_ELIGIBLE)  # type: ignore[call-overload]
    assert unregisterable == [], (
        f"these are stored in webhook.event_types and the register route would refuse "
        f"them: {unregisterable}. A type that stops being eligible needs a migration "
        "stripping it, or every subscription naming it goes quiet with no error."
    )


def test_a_callback_still_owed_across_the_rename_is_rebuilt_with_the_events_own_type(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undelivered row is retried after the deploy, so its type reaches a tenant.

    Recovered by joining the Event Log on `(session_id, seq)` rather than mapped from
    the state, because the delivery row already records exactly which event it was for
    and the map cannot answer: `running` is both a create and a resume, and guessing
    would put a type on the wire that names something that did not happen.

    A row already delivered keeps what it says. It is a record of what was sent, and
    what was sent was the old spelling -- rewriting it would make the ledger claim a
    callback nobody received.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    command.upgrade(_config(), "0029")
    registration, session = uuid4(), uuid4()
    _write(
        unmigrated_url,
        [
            (
                "INSERT INTO webhook (id, tenant_id, url, states, secret_ref)"
                " VALUES (:wid, :tid, 'https://hooks.example.com/owed',"
                " ARRAY['stopped'], 'signing-owed')",
                {"wid": registration, "tid": uuid4()},
            ),
            (
                "INSERT INTO event_log (session_id, seq, type, payload)"
                " VALUES (:sid, 3, 'session.stopped', '{}'::jsonb),"
                " (:sid, 2, 'session.suspended', '{}'::jsonb)",
                {"sid": session},
            ),
            (
                "INSERT INTO webhook_delivery"
                " (webhook_id, session_id, state, seq, attempts, delivered_at_ms)"
                " VALUES (:wid, :sid, 'stopped', 3, 1, NULL),"
                " (:wid, :sid, 'suspended', 2, 1, 1700000000000)",
                {"wid": registration, "sid": session},
            ),
        ],
    )

    command.upgrade(_config(), "head")

    owed = _read(
        unmigrated_url,
        "SELECT event_type FROM webhook_delivery"
        " WHERE webhook_id = :wid AND session_id = :sid AND seq = 3",
        {"wid": registration, "sid": session},
    )
    assert owed == "session.stopped", (
        f"the callback still owed would be retried naming {owed!r}, which is not an "
        "event type this platform publishes"
    )

    already_sent = _read(
        unmigrated_url,
        "SELECT event_type FROM webhook_delivery"
        " WHERE webhook_id = :wid AND session_id = :sid AND seq = 2",
        {"wid": registration, "sid": session},
    )
    assert already_sent == "suspended", (
        "a delivered row was rewritten; it records what was sent, and what was sent "
        "was the old spelling"
    )


def test_a_rollback_hands_a_registration_back_the_states_it_was_written_with(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rolling back has to undo the translation too, or the column outlives its data.

    A deploy that goes out and comes back leaves `webhook.states` holding event type
    names, which the previous release matches nothing against -- the same silently-dead
    subscription as the forward gap, arrived at from the other side and at the worst
    possible moment, since a rollback is already an incident.

    Two types fold onto one state going back, which is what the `DISTINCT` in 0030 is
    for: `session.created` and `session.resumed` both become `running`. A registration
    naming both cannot reach this point any more -- 0031 strips the second on the way up
    and cannot put it back on the way down -- so what is graded here is that the round
    trip still hands back the states the row was written with, not that the duplicate is
    collapsed.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    command.upgrade(_config(), "0029")
    registration = uuid4()
    _write(
        unmigrated_url,
        [
            (
                "INSERT INTO webhook (id, tenant_id, url, states, secret_ref)"
                " VALUES (:wid, :tid, 'https://hooks.example.com/rollback',"
                " ARRAY['running','stopped'], 'signing-rollback')",
                {"wid": registration, "tid": uuid4()},
            )
        ],
    )

    command.upgrade(_config(), "head")
    command.downgrade(_config(), "0029")

    handed_back = _read(
        unmigrated_url,
        "SELECT states FROM webhook WHERE id = :wid",
        {"wid": registration},
    )
    assert sorted(handed_back) == ["running", "stopped"], (  # type: ignore[call-overload]
        f"the rollback left the registration naming {handed_back}, which the release "
        "being rolled back to reads as Session states and matches nothing against"
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


def _seed_a_tenant_with_one_tool_and_one_grant(
    url: str, *, tenant: object, server: object, session: object
) -> None:
    """One server, one tool, one Session whose Grant names that tool, as 0031 stored it.

    Written raw for the reason `_write` gives: these are rows in the shape a *previous*
    revision stored, and the adapters only know how to write the current one.
    """
    _write(
        url,
        [
            (
                "INSERT INTO tool_server (id, tenant_id, server_name, endpoint)"
                " VALUES (:sid, :tid, 'deepwiki', '{}'::jsonb)",
                {"sid": server, "tid": tenant},
            ),
            (
                "INSERT INTO registered_tool"
                " (tenant_id, name, server_id, remote_name, parameters, scope_bindings)"
                " VALUES (:tid, 'ask', :sid, 'ask', '{}'::jsonb,"
                ' \'[{"parameter": "q", "scope": "query"}]\'::jsonb)',
                {"tid": tenant, "sid": server},
            ),
            (
                "INSERT INTO session (id, tenant_id, definition_id,"
                " definition_revision, grant_tools, scope, budget_minor_units,"
                " budget_currency, retention_days)"
                " VALUES (:xid, :tid, :did, 'r1', '[\"ask\"]'::jsonb, '{}'::jsonb,"
                " 1000, 'USD', 30)",
                {"xid": session, "tid": tenant, "did": uuid4()},
            ),
        ],
    )


def test_the_rename_to_per_server_names_runs_against_a_registry_that_has_tools(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**0032 could not run against any database that had ever registered a tool.**

    Its backfill is an `UPDATE registered_tool`, and 0005 put a trigger on that table
    that refuses every update -- so the statement raises `restrict_violation` the moment
    one row matches. Against an empty table it matches nothing and the trigger never
    fires, which is why every migration case before this one passed: they all upgrade a
    fresh database. The first deploy to a cluster holding 208 registered tools failed
    here, ten minutes into a `kubectl wait`, with the schema left at 0031.

    Seeding before the upgrade is the whole point of the case. A migration that alters
    data is only exercised by data, and "upgrade head on an empty database" grades the
    DDL and nothing else.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    command.upgrade(_config(), "0031")
    tenant, server, session = uuid4(), uuid4(), uuid4()
    _seed_a_tenant_with_one_tool_and_one_grant(
        unmigrated_url, tenant=tenant, server=server, session=session
    )

    command.upgrade(_config(), "0032")

    assert (
        _read(
            unmigrated_url,
            "SELECT advertised_name FROM registered_tool"
            " WHERE tenant_id = :tid AND name = 'ask'",
            {"tid": tenant},
        )
        == "deepwiki__ask"
    )


def test_a_grant_written_before_the_rename_still_names_the_tool_it_was_written_for(
    unmigrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Grant holds the name the Gateway compares, and the rename moves that name.

    Every Grant stored before 0032 holds a bare tool name, because that is what a tool
    was called. After it, the Gateway checks membership against `advertised_name` --
    `deepwiki__ask` -- so a Grant still saying `ask` matches nothing and the Session
    silently loses every tool it was granted. Measured on the live database before the
    roll: **146 of 146** Sessions with a non-empty Grant belong to tenants that have
    registered tools, so every one of them was in that position.

    Rewriting the Grant is a rename and not a widening, and the migration is where that
    distinction can still be made safely: before 0032 a bare name was unique within a
    tenant -- it was half the primary key -- so each entry maps to exactly one tool.
    An entry matching no tool of that tenant is left exactly as it is: it granted
    nothing before and grants nothing after, and inventing a qualified name for it would
    be the one edit here that could widen a Grant.
    """
    monkeypatch.setenv(_URL_ENV, unmigrated_url)
    command.upgrade(_config(), "0031")
    tenant, server, session = uuid4(), uuid4(), uuid4()
    _seed_a_tenant_with_one_tool_and_one_grant(
        unmigrated_url, tenant=tenant, server=server, session=session
    )

    command.upgrade(_config(), "0032")

    carried = _read(
        unmigrated_url,
        "SELECT grant_tools FROM session WHERE id = :xid",
        {"xid": session},
    )
    assert carried == ["deepwiki__ask"], (
        f"the Grant now names {carried}, which the Tool Gateway matches nothing "
        "against. A Session that was granted a tool has silently lost it."
    )
