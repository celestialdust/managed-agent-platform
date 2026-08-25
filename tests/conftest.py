"""Tier-1 fixtures: a real PostgreSQL, started per test session, migrated once.

Tier 1 is the local TDD loop and it runs against a real database rather than a fake,
because every property this slice claims — a primary key that refuses a duplicate, a
check constraint that refuses sequence 0, an advisory lock that serializes two writers —
is a property of PostgreSQL and not of any code we could stand in for it.

The container image tracks the major version the platform runs on (`environment.md` pass
2 records RDS at PostgreSQL 17). Tier 2, the in-cluster run against that RDS instance,
is a separate thing entirely and is never what these fixtures give you.

The database is migrated once per session and shared. Tests keep out of each other's way
by each using a fresh Session id, which is the same isolation the production table has:
the sequence is per Session, so two Sessions cannot collide by construction.
"""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from testcontainers.community.postgres import PostgresContainer

_ROOT = Path(__file__).resolve().parents[1]
_URL_ENV = "MAP_DATABASE_URL"

# Testcontainers' reaper bind-mounts the Docker socket into a container of its own.
# Docker Desktop's default endpoint is a per-user path the VM refuses as a mount source,
# so the reaper fails to start and every container fixture errors before Postgres is
# reached. The canonical path is a symlink to it and does mount, which is what this
# override selects; on Linux it is already the endpoint, so the line changes nothing
# there.
os.environ.setdefault("TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE", "/var/run/docker.sock")

# The override above fixes the reaper's socket path but not its reliability: roughly two
# runs in ten still failed every container fixture with `mkdir
# /host_mnt/.../docker.sock:
# operation not supported` from the reaper itself. A test that fails one time in five is
# a
# bug regardless of which side it is on, and this one fails the whole tier at setup,
# which
# reads as a broken database rather than a broken reaper.
#
# The trade is explicit: the reaper exists to remove containers after an *abnormal*
# exit.
# Every fixture here uses `with PostgresContainer(...)`, so a normal exit -- pass, fail,
# or
# assertion error -- still stops its container. What is given up is cleanup after the
# process is killed outright, which leaves a container to remove by hand.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")


def _migrate(url: str) -> None:
    """Bring the database to head through the runner a deployment uses.

    This used to load `0001_event_log.py` by path and drive its `upgrade()` through a
    hand-built `MigrationContext`, to stay independent of a runner config file. That was
    defensible while no config file existed, and became two problems once one did: the
    deployed path stayed unexercised, and -- the sharper one -- a second revision would
    simply not be applied here, so a schema change would ship with its tests passing
    against the schema as it stood before it.
    """
    os.environ[_URL_ENV] = url
    command.upgrade(Config(str(_ROOT / "alembic.ini")), "head")


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as postgres:
        url = postgres.get_connection_url()
        _migrate(url)
        yield url


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    """A pool wide enough that a concurrency test measures the database, not the pool.

    With the default pool a sixteen-way concurrent append would spend most of its time
    queued for a connection, and a serialization bug could hide behind that queue.
    """
    created = create_async_engine(database_url, pool_size=24, max_overflow=8)
    try:
        yield created
    finally:
        await created.dispose()
