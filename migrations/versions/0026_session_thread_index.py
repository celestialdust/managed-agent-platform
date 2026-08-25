"""Four partial indexes so reading a Session's threads is four seeks, not a scan.

A thread has no row of its own. `GET /v1/sessions/{id}/threads` derives every thread it
lists from `event_log`: where each one was announced, the last event that carried its
identifier, whether it was archived, and whether the Turn that opened it has ended. That
is one statement with a `DISTINCT ON`, two correlated aggregates and a nested anti join,
and on the table's own primary key alone every one of those reads the whole Session.

**Measured before any of these existed.** PostgreSQL 17, `event_log` at 121,908 rows and
34 MB, one long-lived Session holding 10,979 of them across 41 threads, first page of 25
(median of seven `EXPLAIN (ANALYZE, BUFFERS)` runs):

    listing a page   194.8 ms   393,266 buffers
    one named thread   9.9 ms    19,367 buffers

With all four, the same reads are 0.325 ms and 0.089 ms over 1,384 and 90 buffers -- 600
times less work for the listing. Every subplan was a `Bitmap Index Scan on
event_log_pkey` for the Session followed by a filter that threw away everything the
Session had done, once per thread on the page.

**Each index was measured by removing it, not by adding it.** From the four,
dropping one at a time (same seed, same page):

    without event_log_thread_activity   170.3 ms   listing   6.9 ms  one thread
    without event_log_turn_events         1.108 ms           0.428 ms
    without event_log_thread_archives     1.077 ms           0.062 ms
    without event_log_thread_starts       0.932 ms           0.068 ms

So the activity index is the one that turns the read from a scan into a seek, and the
other three each cost a further 3x when absent. None of them is idle: every one appears
in the plan, and each of the four subplans is served by a different one.

**A fifth was measured and rejected.** `(session_id, type, seq)` with no
predicate -- the obvious index for the announcements -- reads 0.281 ms against
`event_log_thread_starts`'s 0.325 ms, a difference inside the run-to-run spread,
and costs 7,912 kB against 96 kB. It also indexes every event in the table to
serve a query about the few that open a thread. Eighty times the size for no
measured gain is not a trade; it is left out.

**Why the predicates are shaped the way they are.** A partial index is only usable when
the planner can prove its predicate from the query's own clauses.
`(payload ->> 'thread_id') IS NOT NULL` is provable from `payload ->> 'thread_id' = <the
thread>`, because that operator is strict and cannot be true of a NULL -- so those two
predicates hold whatever the parameter values are. The two that name a type do not: the
query binds the type name rather than spelling it, and the proof works only because
PostgreSQL folds a bound parameter into a constant while it plans, which is what every
plan measured here shows it doing. If it ever stops -- a generic plan for a re-prepared
statement -- those two become unusable and the read falls back to the two rows above,
around 1 ms. That is 3x worse and still 200x better than no index, which is why the two
that carry the weight are deliberately the two whose predicates do not depend on it. The
generic case is reasoned rather than measured: `plan_cache_mode = force_generic_plan`
still produced folded constants in this harness, so it could not be provoked.

**What they cost the append path**, which is the hot one and has to be said out loud:
400 single-row appends through `PostgresEventLogAppend`, median of three runs, went from
1.975 ms to 2.117 ms per event -- about 7%, or 0.14 ms of index maintenance per event.

`event_log_thread_activity` is the expensive one on disk, 10 MB against the table's 34,
because almost every event carries a thread identifier and its third key column is the
append time, which is distinct per row and so defeats the btree's deduplication of
repeated keys. `event_log_turn_events` covers nearly as many rows and is 1,184 kB for
exactly the opposite reason: a Session has a handful of Turn identifiers, so the key
repeats and is compressed.

Revision ID: 0026
Revises: 0025
"""

from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None

_INDEXES: tuple[tuple[str, str], ...] = (
    (
        "event_log_thread_starts",
        "(session_id, seq) WHERE type = 'thread.started'",
    ),
    (
        "event_log_thread_archives",
        "(session_id, (payload ->> 'thread_id')) WHERE type = 'thread.archived'",
    ),
    (
        "event_log_thread_activity",
        "(session_id, (payload ->> 'thread_id'), appended_at)"
        " WHERE (payload ->> 'thread_id') IS NOT NULL",
    ),
    (
        "event_log_turn_events",
        "(session_id, (payload ->> 'turn_id'), type)"
        " WHERE (payload ->> 'turn_id') IS NOT NULL",
    ),
)
"""One entry per subplan of the threads read, name first, then key and predicate.

Held as data rather than written out twice, so `downgrade` drops exactly what `upgrade`
created. Four `CREATE INDEX` statements with four matching `DROP INDEX` statements
somewhere below them is where a rollback quietly leaves one behind.

The order is the order they are created in and reads from cheapest to most
expensive, which is also the order a reader should read them: the first two are seek
indexes over the two thread-family types, the third is the aggregate the listing
could not do without, and the last serves the Turn lookup that decides whether a
thread is still running.
"""


def upgrade() -> None:
    for name, definition in _INDEXES:
        op.execute(f"CREATE INDEX {name} ON event_log {definition};")


def downgrade() -> None:
    for name, _ in reversed(_INDEXES):
        op.execute(f"DROP INDEX IF EXISTS {name};")
