"""Two replicas over one delivery window, in front of a real PostgreSQL 17.

Tier 1, and it has to be: every claim here is a claim about what the database does under
two callers, and a fake standing in for it would be a fake of the answer.

**The question this file settles.** `deploy/k8s/control-plane.yaml` runs two replicas,
so whatever starts a sweep starts it twice. The webhook dispatcher's own comments read
as though the delivery claim settles that -- "another dispatcher holds this attempt",
"lost the race" -- and it does not. The claim is

    INSERT INTO webhook_delivery (...) VALUES (...)
    ON CONFLICT (webhook_id, session_id, state) DO UPDATE
      SET attempts = webhook_delivery.attempts + 1
      WHERE webhook_delivery.delivered_at_ms IS NULL
        AND webhook_delivery.attempts < :max_attempts
    RETURNING attempts

and the second caller takes the `DO UPDATE` arm, finds the callback undelivered with
attempts to spare, increments, and is handed a row. It counts attempts; it does not pick
a runner. So the delivery sweep takes a lease and the pod sweep does not, and the first
case below is the evidence for that split rather than a re-reading of the SQL.

**Why the existing race case does not already cover this.**
`test_two_dispatchers_claiming_one_state_change_produce_exactly_one_winner` in
`test_webhook_signed_no_credential.py` passes `max_attempts=1`, where the `DO UPDATE`'s
own `WHERE` refuses the second caller and there genuinely is one winner. The dispatcher
passes `MAX_ATTEMPTS`, which is five. That case is green, correct about the cap it
names, and blind to the cap production uses -- the same shape as a guard wired behind
the thing it alarms about.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.adapters.postgres.sweep_lease import PostgresSweepLease
from managed_agent.adapters.postgres.webhook_store import PostgresWebhookStore
from managed_agent.control.webhooks.dispatcher import MAX_ATTEMPTS
from managed_agent.control.webhooks.registry import CallbackUrl
from managed_agent.core.ids import Seq, TenantId, new_session_id
from managed_agent.core.session.session import SessionState

_A_SWEEP = "webhook-delivery"
_ANOTHER_SWEEP = "session-pods"


async def test_the_delivery_claim_grants_two_replicas_the_same_undelivered_callback(
    engine: AsyncEngine,
) -> None:
    """The reason the delivery sweep needs a lease, taken from the database itself.

    Two claims of one state change, at the cap the dispatcher actually passes, on their
    own connections. Both are granted -- one by inserting and one by incrementing -- and
    a granted claim is a dispatcher that goes on to fetch a secret and post. That is one
    callback delivered twice to a tenant's endpoint and two of its five attempts spent,
    with nothing in any log calling it a failure.

    Concurrent rather than sequential only because that is the production shape; the
    outcome does not depend on the interleaving, since the loser of the unique-index
    race is exactly the caller the `DO UPDATE` arm then admits.
    """
    store = PostgresWebhookStore(engine)
    hook = await store.register(
        TenantId(uuid4()),
        CallbackUrl("https://hooks.example.com/two-replicas"),
        frozenset({SessionState.STOPPED}),
        "signing-two-replicas",
    )
    session_id = new_session_id()

    granted = await asyncio.gather(
        *(
            store.claim(hook.id, session_id, SessionState.STOPPED, Seq(3), MAX_ATTEMPTS)
            for _ in range(2)
        )
    )

    assert sorted(g for g in granted if g is not None) == [1, 2], (
        f"the claim answered {granted}. If either of these were None the delivery "
        "sweep would not need a lease at all -- this is the case to re-read before "
        "removing one."
    )


async def test_the_lease_admits_exactly_one_of_two_replicas_at_once(
    engine: AsyncEngine,
) -> None:
    """And this is the fix: one runner per sweep name while a tick is in flight.

    The second attempt is made *while* the first is still inside its block, forced by
    the two events rather than by a sleep -- a sleep long enough to be reliable is long
    enough to be slow, and this way the overlap is a fact of the test instead of a race
    it usually wins.
    """
    lease = PostgresSweepLease(engine)
    first_is_in = asyncio.Event()
    second_has_tried = asyncio.Event()

    async def the_replica_that_gets_there_first() -> bool:
        async with lease.held(_A_SWEEP) as mine:
            first_is_in.set()
            await second_has_tried.wait()
            return mine

    async def the_replica_that_arrives_while_it_is_held() -> bool:
        await first_is_in.wait()
        async with lease.held(_A_SWEEP) as mine:
            second_has_tried.set()
            return mine

    async with asyncio.timeout(10):
        held = await asyncio.gather(
            the_replica_that_gets_there_first(),
            the_replica_that_arrives_while_it_is_held(),
        )

    # `list(...)` because `gather` hands back a list at runtime while mypy types this
    # pair as a tuple, and comparing the two shapes directly is False whatever they
    # hold -- which is a green-looking assertion that grades nothing.
    assert list(held) == [True, False], (
        f"two replicas were told {held} about one sweep; both being True is the "
        "duplicate callback this lease exists to prevent"
    )


async def test_the_lease_is_released_when_the_tick_ends(engine: AsyncEngine) -> None:
    """A lease kept past its tick stops every replica, not only the one holding it.

    Asserted by taking it again rather than by reading `pg_locks`: what the next tick
    needs is to be able to have it, which is the same thing said without depending on a
    catalog view.
    """
    lease = PostgresSweepLease(engine)
    for _ in range(3):
        async with lease.held(_A_SWEEP) as mine:
            assert mine is True


async def test_the_lease_is_released_when_the_tick_raises(engine: AsyncEngine) -> None:
    """The path no `finally` in this repository can cover, covered by the transaction.

    An exception inside the block rolls the transaction back, and an advisory lock taken
    with the transaction goes with it. This is why the lock is transaction-scoped rather
    than session-scoped with an unlock: the same property is what stops a replica that
    is killed mid-sweep from wedging every other replica.
    """
    lease = PostgresSweepLease(engine)
    with pytest.raises(RuntimeError):
        async with lease.held(_A_SWEEP) as mine:
            assert mine is True
            raise RuntimeError("the sweep raised while it held the lease")

    async with lease.held(_A_SWEEP) as after:
        assert after is True, "the lease survived the raise and no replica can sweep"


async def test_one_sweep_holding_its_lease_does_not_block_the_other_sweep(
    engine: AsyncEngine,
) -> None:
    """The lease is per name, so a slow delivery pass cannot stall pod reclamation.

    Worth its own case because the two names are hashed onto one integer keyspace: a
    lease that keyed on something constant, or a hash collision treated as correctness
    rather than as a shared turn, would serialize two passes that have nothing to do
    with each other.
    """
    lease = PostgresSweepLease(engine)
    async with (
        lease.held(_A_SWEEP) as delivering,
        lease.held(_ANOTHER_SWEEP) as reclaiming,
    ):
        assert (delivering, reclaiming) == (True, True)
