"""One replica at a time, for a sweep whose own store cannot say it.

The webhook delivery claim looks like the exclusion this is for and is not it.
`INSERT ... ON CONFLICT (webhook_id, session_id, state) DO UPDATE SET attempts =
attempts + 1 WHERE delivered_at_ms IS NULL AND attempts < :max RETURNING attempts`
grants the loser of a race a row as readily as the winner: the second caller conflicts,
takes the `DO UPDATE` arm, finds the callback still undelivered with attempts to spare,
increments and returns. So two replicas sweeping one window both claim, both post, and
the tenant's endpoint is called twice for one state change while two of its five
attempts are spent. The claim counts attempts; it does not pick a runner.

An advisory lock is what picks one. It is the mechanism `event_log_append.py` already
uses for the same shape of problem -- serialize the common path, keep the constraint
underneath for the truth -- and it costs no write and no table.

**Transaction-scoped rather than session-scoped**, which is the whole reason a replica
that dies mid-sweep cannot wedge the platform: PostgreSQL releases an xact lock when the
transaction ends, and a killed backend ends its transaction. A session-scoped lock plus
an explicit unlock would be correct on every path this file writes and would strand the
lock on the one path it cannot write, which is the process being killed.

**The transaction stays open for the whole tick**, so one pooled connection is idle in
transaction while the sweep runs -- one of the fifty `composition.py` sizes the pool
for, against a sweep that is one loop over a bounded window. That is the price of a
lock whose lifetime is a block rather than a pair of calls, and it is the right way
round: the alternative leaks the lock, and a leaked lease stops every replica from
sweeping.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

# `pg_try_advisory_xact_lock` and not `pg_advisory_xact_lock`: a replica that cannot
# have the lease must skip this tick, not queue behind the holder. Waiting would run
# the second pass over a window the winner had already delivered -- the duplicate the
# lease exists to prevent, arriving one sweep later.
#
# hashtextextended maps the sweep's name onto the bigint keyspace advisory locks use,
# the same way the append path maps a Session's uuid. A collision between two sweep
# names would make those two take turns instead of running together, which costs a tick
# of latency and never correctness -- nothing here depends on two sweeps overlapping.
_TAKE = sa.text(
    "SELECT pg_try_advisory_xact_lock(hashtextextended(cast(:name AS text), 0))"
).bindparams(sa.bindparam("name", type_=sa.Text()))


class PostgresSweepLease:
    """Hands out one lease per sweep name, for the length of the block that holds it."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @asynccontextmanager
    async def held(self, name: str) -> AsyncIterator[bool]:
        """True for the one caller holding this name's lease, False for the others.

        False rather than an exception, because losing is the ordinary outcome for the
        replica that did not win and happens once per tick for the life of the process.

        The lock is taken inside `engine.begin()` and released when that transaction
        commits on the way out of this block -- including when the body raised, since
        the transaction rolls back and the lock goes with it. There is no path out of
        here that keeps the lease.
        """
        async with self._engine.begin() as conn:
            yield bool((await conn.execute(_TAKE, {"name": name})).scalar_one())
