"""Alembic entry point: resolve a database URL, then run the revisions against it.

Runs over **asyncpg**, the driver the application already uses, rather than pulling in a
second synchronous one. Alembic's own DDL layer is synchronous, so the connection is
handed to it through `run_sync`. The alternative -- adding `psycopg2` for migrations
only -- would mean the schema is created by a driver no other code path exercises, and a
connection-level difference between the two would show up first in a deployment.

Offline mode is deliberately unsupported. A generated SQL script is applied by something
outside this repository, and nothing here could then say whether the schema in the
database matches the revisions in the tree -- the one question a runner exists to
answer.

The URL comes from the environment and has no default. A default would be a live
connection string committed to the repository, and a wrong one would silently migrate
whichever database it happened to reach.
"""

from __future__ import annotations

import asyncio
import logging.config
import os

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

# Two names, checked in this order, because migrating and serving want different
# privileges: DDL for the runner, DML for the application. Naming them separately is
# what lets a deployment hand the runner a role that can ALTER TABLE without granting
# that to every request handler. But the application's variable is the one
# `environment.md` provisions and the one MAP-50's manifest sets, so requiring only the
# runner's name would mean a correctly deployed process fails on an unset variable
# nobody was told to set.
#
# The fallback cannot do the wrong thing quietly. A deployment that meant to supply a
# DDL credential and misspelled the variable falls back to the application's, which
# lacks DDL and so fails on the first statement with a permission error naming the table
# -- loudly, and before any revision has half-applied.
_URL_ENVS = ("MAP_DATABASE_URL", "DATABASE_URL")


def _url() -> str:
    """Return the database URL with the asyncpg driver named explicitly.

    Accepts a bare `postgresql://` and normalises it, so whichever variable supplied it
    reaches the same database as the application and a deployment cannot point them at
    two different ones by writing the DSN two ways.
    """
    raw = next((v for name in _URL_ENVS if (v := os.environ.get(name))), None)
    if not raw:
        raise RuntimeError(
            f"none of {', '.join(_URL_ENVS)} is set. There is no default on purpose: a "
            "default would be a connection string committed to the repository, and a "
            "wrong one would migrate whichever database it reached."
        )
    if raw.startswith("postgresql+asyncpg://"):
        return raw
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    raise RuntimeError(
        f"the database URL must be a postgresql:// or postgresql+asyncpg:// URL; got a "
        f"{raw.split('://', 1)[0]!r} scheme. Refused rather than guessed: the wrong "
        "driver here fails at DDL time, halfway through a migration."
    )


def _run(connection: Connection) -> None:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = create_async_engine(_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run)
            await connection.commit()
    finally:
        await engine.dispose()


def _configure_logging() -> None:
    """Load the logging configuration `alembic.ini` already carries.

    Without this call the configuration is inert: alembic's own `INFO` progress lines
    -- `Running upgrade <from> -> <to>` -- are emitted to a logger with no handler and
    go nowhere. That was measured against the real migration Job in `map-dev`, whose
    pod logs were **0 bytes** after applying eleven revisions. A run that applied
    eleven and a run that applied none printed exactly the same thing, which is the
    property that makes it dangerous: the only record of what a schema migration did to
    a production database was its exit code.

    `disable_existing_loggers=False` because the application configures its own loggers
    and a migration must not silence them on the way past. The file name can be absent
    when alembic is driven programmatically without an ini, so it is checked rather
    than assumed -- a runner that cannot find its logging config should still migrate.
    """
    ini = context.config.config_file_name
    if ini is not None:
        logging.config.fileConfig(ini, disable_existing_loggers=False)


_configure_logging()

if context.is_offline_mode():
    raise RuntimeError(
        "offline mode is not supported: a generated script is applied by something "
        "outside this repository, and nothing here could then tell whether the "
        "database matches the revisions in the tree."
    )
asyncio.run(_run_async())
