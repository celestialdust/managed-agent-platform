"""Webhook registrations, the once-per-state delivery claim, and the tail's watermark.

One class over three tables, because the dispatcher takes the registration query and the
claim as two ports and both are answered by the same connection pool over rows that must
agree with each other: `undelivered` joins the claim to the registration for the url and
the secret reference, and splitting the two would make that read a cross-adapter join
performed in Python.

The whole concurrency argument lives in `claim`. Two dispatchers reaching one state
change at the same instant is the case that decides whether "one callback" is true, and
it is decided by the primary key on `(webhook_id, session_id, state)` rather than by a
read followed by a write -- there is no point between the two where a second caller can
see the row absent and insert its own.

Bind parameter types are declared because these are textual statements and SQLAlchemy
has no column metadata to infer from: without one asyncpg receives a bare Python object.
`.columns(...)` on the reads is the same declaration in the other direction. Measured
against this dialect the result-type declarations change nothing today -- the asyncpg
driver already decodes what these columns hold -- but the repository requires them and a
driver without that decoding would make them load-bearing.

No method here logs, returns or accepts a signing secret. `secret_ref` is a name in the
credential vault; the value behind it never reaches this module.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.control.webhooks.registry import CallbackUrl, WebhookRecord
from managed_agent.core.ids import Seq, SessionId, TenantId
from managed_agent.core.session.session import SessionState

_RECORD_TYPES: dict[str, sa.types.TypeEngine[Any]] = {
    "id": sa.Uuid(),
    "tenant_id": sa.Uuid(),
    "url": sa.Text(),
    "states": sa.ARRAY(sa.Text()),
    "secret_ref": sa.Text(),
    "created_at_ms": sa.BigInteger(),
}

_REGISTER = sa.text(
    "INSERT INTO webhook (id, tenant_id, url, states, secret_ref)"
    " VALUES (:wid, :tid, :url, :states, :ref)"
    " RETURNING created_at_ms"
).bindparams(
    sa.bindparam("wid", type_=sa.Uuid()),
    sa.bindparam("tid", type_=sa.Uuid()),
    sa.bindparam("states", type_=sa.ARRAY(sa.Text())),
)

_SELECT_RECORD = (
    "SELECT id, tenant_id, url, states, secret_ref, created_at_ms FROM webhook"
)

_LIST_FOR_TENANT = (
    sa.text(f"{_SELECT_RECORD} WHERE tenant_id = :tid ORDER BY created_at_ms, id")
    .bindparams(sa.bindparam("tid", type_=sa.Uuid()))
    .columns(**_RECORD_TYPES)
)

# `:state = ANY(states)` rather than an overlap operator, because it is the form the GIN
# index on `states` plans directly and because the question really is about one state.
_WATCHING = (
    sa.text(f"{_SELECT_RECORD} WHERE tenant_id = :tid AND :state = ANY(states)")
    .bindparams(
        sa.bindparam("tid", type_=sa.Uuid()),
        sa.bindparam("state", type_=sa.Text()),
    )
    .columns(**_RECORD_TYPES)
)

_DELETE = sa.text(
    "DELETE FROM webhook WHERE id = :wid AND tenant_id = :tid"
).bindparams(
    sa.bindparam("wid", type_=sa.Uuid()),
    sa.bindparam("tid", type_=sa.Uuid()),
)

_SCANNED_THROUGH = sa.text("SELECT scanned_through_ms FROM webhook_scan WHERE id = 1")

# Guarded by `scanned_through_ms < :at` rather than written unconditionally, so a pass
# that somehow ran behind another cannot move the watermark backwards and make the
# window in between be read a second time -- or, worse, be read by a pass that has
# already advanced past it.
_ADVANCE_SCAN = sa.text(
    "UPDATE webhook_scan SET scanned_through_ms = :at"
    " WHERE id = 1 AND scanned_through_ms < :at"
)

# `RETURNING` yields a row only when this call inserted, or when the `DO UPDATE`'s own
# `WHERE` held -- so "first attempt", "a retry that is still owed" and "already
# delivered, or spent, or lost the race" are one round trip with no read-then-write
# between them.
_CLAIM = sa.text(
    "INSERT INTO webhook_delivery"
    " (webhook_id, session_id, state, seq, attempts)"
    " VALUES (:wid, :sid, :state, :seq, 1)"
    " ON CONFLICT (webhook_id, session_id, state) DO UPDATE"
    " SET attempts = webhook_delivery.attempts + 1"
    " WHERE webhook_delivery.delivered_at_ms IS NULL"
    " AND webhook_delivery.attempts < :max_attempts"
    " RETURNING attempts"
).bindparams(
    sa.bindparam("wid", type_=sa.Uuid()),
    sa.bindparam("sid", type_=sa.Uuid()),
    sa.bindparam("state", type_=sa.Text()),
)

_MARK_DELIVERED = sa.text(
    "UPDATE webhook_delivery"
    " SET delivered_at_ms = (extract(epoch from now()) * 1000)::bigint,"
    " last_response_code = :status"
    " WHERE webhook_id = :wid AND session_id = :sid AND state = :state"
).bindparams(
    sa.bindparam("wid", type_=sa.Uuid()),
    sa.bindparam("sid", type_=sa.Uuid()),
    sa.bindparam("state", type_=sa.Text()),
)

# Ordered by attempts so a row that has failed least is tried first: a receiver that is
# permanently gone otherwise crowds out one that was briefly down.
_UNDELIVERED = sa.text(
    "SELECT d.webhook_id, w.tenant_id, w.url, w.secret_ref,"
    " d.session_id, d.state, d.seq"
    " FROM webhook_delivery d JOIN webhook w ON w.id = d.webhook_id"
    " WHERE d.delivered_at_ms IS NULL AND d.attempts < :max_attempts"
    " ORDER BY d.attempts, d.webhook_id, d.session_id, d.state"
    " LIMIT :limit"
).columns(
    webhook_id=sa.Uuid(),
    tenant_id=sa.Uuid(),
    url=sa.Text(),
    secret_ref=sa.Text(),
    session_id=sa.Uuid(),
    state=sa.Text(),
    seq=sa.BigInteger(),
)


@dataclass(frozen=True, slots=True)
class PendingRow:
    """One claimed-but-undelivered callback, joined to the registration it belongs to.

    Frozen like every other row type here. It carries the url and the secret reference
    so a retry needs no second read of the registration, and it carries `seq` so the
    retry rebuilds the identical callback without going back to the Event Log, whose
    window the sweep has left behind.

    It carries the tenant for the same reason it carries the reference: the reference is
    not a vault key on its own, and the retry composes one from the two. Without the
    tenant on this row a retry would have only the tenant's own text to key on, which is
    the shape the cross-tenant read had.
    """

    webhook_id: UUID
    tenant_id: TenantId
    url: str
    secret_ref: str
    session_id: SessionId
    state: SessionState
    seq: Seq


class PostgresWebhookStore:
    """Registrations, the delivery claim and the watermark, over one engine."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def register(
        self,
        tenant_id: TenantId,
        url: CallbackUrl,
        states: frozenset[SessionState],
        secret_ref: str,
    ) -> WebhookRecord:
        """Write one registration and return it as stored.

        The id is minted here rather than accepted from the caller: a tenant choosing
        its own would be choosing a key in a table it shares with every other tenant.

        States are written sorted so two registrations of the same set store equal
        arrays -- a `frozenset`'s iteration order is not part of its value, and the
        stored array would otherwise differ run to run.
        """
        identifier = uuid4()
        async with self._engine.begin() as conn:
            created = (
                await conn.execute(
                    _REGISTER,
                    {
                        "wid": identifier,
                        "tid": tenant_id,
                        "url": str(url),
                        "states": sorted(state.value for state in states),
                        "ref": secret_ref,
                    },
                )
            ).scalar_one()
        return WebhookRecord(
            id=identifier,
            tenant_id=tenant_id,
            url=url,
            states=frozenset(states),
            secret_ref=secret_ref,
            created_at_ms=int(created),
        )

    async def list_for_tenant(self, tenant_id: TenantId) -> Sequence[WebhookRecord]:
        """This tenant's registrations, oldest first.

        The tenant is a term in the query rather than a filter applied afterwards, so
        another tenant's registration is absent rather than fetched and dropped.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(_LIST_FOR_TENANT, {"tid": tenant_id})
            return [_record(row) for row in result]

    async def delete(self, webhook_id: UUID, tenant_id: TenantId) -> bool:
        """Remove one registration, returning whether there was one to remove.

        False for an id that never existed and for one belonging to somebody else
        alike: the statement matches on both columns, so the two cases are the same
        zero-row result and no caller can tell them apart.

        The delivery rows go with it, by the foreign key's ON DELETE CASCADE. A
        registration that is gone cannot be retried, and rows keyed to it would keep the
        retry pass reading a destination nobody can reach.
        """
        async with self._engine.begin() as conn:
            result = await conn.execute(_DELETE, {"wid": webhook_id, "tid": tenant_id})
            return result.rowcount == 1

    async def watching(
        self, tenant_id: TenantId, state: SessionState
    ) -> Sequence[WebhookRecord]:
        """This tenant's registrations naming this state."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _WATCHING, {"tid": tenant_id, "state": state.value}
            )
            return [_record(row) for row in result]

    async def scanned_through_ms(self) -> int:
        """The instant the cross-Session tail has read the Event Log through."""
        async with self._engine.connect() as conn:
            return int((await conn.execute(_SCANNED_THROUGH)).scalar_one())

    async def advance_scan_to(self, at_ms: int) -> None:
        """Move the watermark forward, and never backward."""
        async with self._engine.begin() as conn:
            await conn.execute(_ADVANCE_SCAN, {"at": at_ms})

    async def claim(
        self,
        webhook_id: UUID,
        session_id: SessionId,
        state: SessionState,
        seq: Seq,
        max_attempts: int,
    ) -> int | None:
        """Take ownership of one attempt at this callback, or None when there is none.

        None covers three cases with one statement: the callback is already delivered,
        its attempts are spent, or another dispatcher inserted the claim first. They are
        one answer on purpose -- every one of them means "do not post", and telling them
        apart would need a second read that the racing dispatcher could invalidate
        between the two.

        The returned number is which attempt this is, counting from 1.
        """
        async with self._engine.begin() as conn:
            attempts = (
                await conn.execute(
                    _CLAIM,
                    {
                        "wid": webhook_id,
                        "sid": session_id,
                        "state": state.value,
                        "seq": seq,
                        "max_attempts": max_attempts,
                    },
                )
            ).scalar_one_or_none()
        return None if attempts is None else int(attempts)

    async def mark_delivered(
        self, webhook_id: UUID, session_id: SessionId, state: SessionState, status: int
    ) -> None:
        """Stamp this callback delivered, taking it out of the retry set."""
        async with self._engine.begin() as conn:
            await conn.execute(
                _MARK_DELIVERED,
                {
                    "wid": webhook_id,
                    "sid": session_id,
                    "state": state.value,
                    "status": status,
                },
            )

    async def undelivered(self, max_attempts: int, limit: int) -> Sequence[PendingRow]:
        """Claimed callbacks still owed a delivery, fewest attempts first."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _UNDELIVERED, {"max_attempts": max_attempts, "limit": limit}
            )
            return [
                PendingRow(
                    webhook_id=UUID(str(row.webhook_id)),
                    tenant_id=TenantId(row.tenant_id),
                    url=str(row.url),
                    secret_ref=str(row.secret_ref),
                    session_id=SessionId(row.session_id),
                    state=SessionState(row.state),
                    seq=Seq(int(row.seq)),
                )
                for row in result
            ]


def _record(row: Any) -> WebhookRecord:
    """One `webhook` row as the domain record.

    `CallbackUrl` is applied as a cast rather than re-parsed: the stored value passed
    `parse_callback_url` on the way in and the table's own check constraint refuses a
    non-https destination, so re-parsing here would be a second answer to a question
    already settled -- and one that could refuse a row the platform is holding.
    """
    return WebhookRecord(
        id=UUID(str(row.id)),
        tenant_id=TenantId(row.tenant_id),
        url=CallbackUrl(str(row.url)),
        states=frozenset(SessionState(state) for state in row.states),
        secret_ref=str(row.secret_ref),
        created_at_ms=int(row.created_at_ms),
    )
