"""How many Turns are waiting for a pod, asked of the log rather than of a process.

The count this replaces was per-process, and the error was not a rounding. Measured
against `map-dev` on 2026-08-24 with six Sessions placing at once behind two control-
plane replicas: one replica answered `turns_awaiting_placement: 6` and the other
answered `0`, for the same six placements. A client's connection is sticky, so an
operator's whole workload lands on one replica and an operator polling the other reads
an idle platform under full load.

**No new write pays for this.** Both events the query needs were already being appended:
`session.placing` per waiting Turn, and the Turn's own `turn.started` or `turn.failed`
when the wait ends. The terminal event is the sweep, so there is no crashed-process
residue to clean up -- only a time bound, because a process that died mid-placement
leaves a `session.placing` with no terminal event and nobody on the other end of it.

Lock-free, like the range reader beside it. It reads rows already committed and has
nothing to serialize against, so a replica can back it.
"""

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent.control.session.placement import PlacementBacklog
from managed_agent.core.vocabulary import placement, turn

_TERMINAL = (turn.TURN_STARTED, turn.TURN_FAILED)
"""The two events that end a placement wait, and there is no third.

`turn.completed` is not among them: a Turn cannot complete without having started, so a
completed Turn's `turn.started` has already cleared it, and naming both would be one
fact in two places. `session.archived` is not either -- archiving a Session does not
retract a Turn that was queued, and a wait the log never closed is exactly what the time
bound is for.
"""

_BACKLOG = (
    sa.text(
        """
    WITH waiting AS (
        SELECT session_id, seq, appended_at, payload ->> 'turn_id' AS turn_id
        FROM event_log
        WHERE type = :placing_type
          AND appended_at > now() - (:within_s * interval '1 second')
    )
    SELECT
        count(*) AS turns,
        count(DISTINCT session_id) AS sessions,
        min(appended_at) AS oldest
    FROM waiting
    WHERE NOT EXISTS (
        SELECT 1 FROM event_log later
        WHERE later.session_id = waiting.session_id
          AND later.seq > waiting.seq
          AND later.type = ANY(:terminal)
          AND later.payload ->> 'turn_id' = waiting.turn_id
    )
    """
    )
    .bindparams(sa.bindparam("terminal", type_=sa.ARRAY(sa.Text())))
    .columns(
        turns=sa.BigInteger(),
        sessions=sa.BigInteger(),
        oldest=sa.TIMESTAMP(timezone=True),
    )
)
"""Live placement waits, in one round trip.

Two index reads and no table scan, and the plan was read rather than assumed. At 24,000
events PostgreSQL chooses a **Nested Loop Anti Join** whose outer side is an `Index Scan
using event_log_placement_waits` -- migration 0025's partial index, which holds one row
per placement wait ever recorded rather than one per event -- and whose inner side is a
`Bitmap Index Scan on event_log_pkey` under `Index Cond: (session_id = ... AND seq >
...)`. So the candidate set comes from one index and each candidate's terminal-event
lookup from another.

At 2,000 events it picks a **Merge Anti Join** with a sequential scan on the inner side
instead, and that is the planner being right rather than a problem to fix: sorting a
tiny table beats a nested loop over it. The choice flips on its own as the log grows,
because the outer side stays a handful of live waits while the inner side does not. This
is recorded and not asserted -- the plan belongs to the planner, and a test pinning a
plan shape is already the one flaky test in this tree (`tests/adapters/test_event_log_ra
nge.py::test_the_floor_read_is_an_index_lookup_rather_than_a_scan`).

`NOT EXISTS` rather than a `LEFT JOIN ... IS NULL`: the join would materialise every
terminal event for each candidate Session before discarding all but the absence, and a
Session with many Turns is the common case rather than the exotic one.

Matched on `turn_id` and not on the Session, because two Turns of one Session can be
waiting at once -- admission refuses a Session that will not take a Turn, not one that
already has a Turn open. Keyed on the Session alone, the first Turn's `turn.started`
would clear the second Turn's wait and the depth would under-report exactly when it
mattered.

`seq > placing.seq` rather than a timestamp comparison. `appended_at` is a server clock
read and two rows can share a millisecond; `seq` is contiguous per Session by
construction (migration 0001), so "later in this Session" is exactly `seq >` and needs
no tie-break.

The window arrives as `:within_s * interval '1 second'` rather than as a concatenated
interval literal, so the caller's number stays a bound parameter and never reaches the
SQL text. `make_interval(secs => ...)` reads better and cannot be used: SQLAlchemy's
`text()` parses `:name` itself, and a named-argument arrow beside a bind parameter is
ambiguous to it.

The CTE is `waiting` and the bind parameter is `placing_type`, and the two names are
kept apart deliberately. Named alike, `text()` binds the CTE reference as a parameter
and PostgreSQL reports `syntax error at or near "placing"` -- an error that points at
the SQL and not at the collision.
"""


class PostgresPlacementBacklog:
    """Reads the standing placement backlog out of `event_log`.

    Holds the engine and nothing else. There is no state to keep: the answer is a
    property of the log at the moment it is asked, which is what makes it the same
    answer from every replica -- the whole reason this exists.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def placement_backlog(self, within_s: int, /) -> PlacementBacklog:
        """One reading of the backlog, bounded to waits younger than `within_s`.

        `oldest` comes back as a timezone-aware datetime and leaves as epoch
        milliseconds, because that is the unit every other clock read in this tree
        publishes and the surface that renders it should be the only module that knows a
        wire format.

        `count(*)` over an empty set is `0` and `min()` over one is NULL, so the empty
        case needs no special path -- but the pair is asserted rather than assumed,
        because a count above zero with no oldest instant would mean the two aggregates
        disagreed about which rows they saw, and publishing that would put an unreadable
        gauge in front of an operator.
        """
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    _BACKLOG,
                    {
                        "placing_type": placement.SESSION_PLACING,
                        "terminal": list(_TERMINAL),
                        "within_s": within_s,
                    },
                )
            ).one()
        turns = int(row.turns)
        oldest = row.oldest
        if turns == 0:
            return PlacementBacklog(
                turns_awaiting=0, sessions_placing=0, oldest_awaiting_at_ms=None
            )
        assert oldest is not None, (
            f"{turns} waits were counted and none carried an append time, so the two "
            "aggregates in one query saw different rows"
        )
        return PlacementBacklog(
            turns_awaiting=turns,
            sessions_placing=int(row.sessions),
            oldest_awaiting_at_ms=int(oldest.timestamp() * 1000),
        )
