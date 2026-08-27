"""The process entrypoint: what `uvicorn` serves, and what closes the pool behind it.

Run it as a factory, because the app cannot be built without a database URL and a module
that opened a connection pool at import time would fail in every context that merely
imports it — a test collector, a linter, `--help`:

    uvicorn managed_agent.asgi:build_app --factory --host 0.0.0.0 --port 8080

`create_app` deliberately takes an already-wired Platform rather than building one,
which is what keeps `composition.build` the only place a concrete adapter is chosen. The
consequence is that something outside both has to call `build`, hand the result over,
and own the engine afterwards. This module is that something, and it is the whole of it.

**Why the shutdown hook is not optional.** `build` returns the engine alongside the
Platform precisely because the pool outlives any one request and has to be disposed by
whoever owns it; a port that closed its own engine would close it for every other port
sharing it. `_POOL_SIZE` is 50, so a process that exits without disposing leaves up to
fifty server-side connections to be reaped by PostgreSQL's own timeouts rather than
released — and with several replicas against one `max_connections`, a rolling restart is
exactly when that matters and exactly when the headroom is smallest.

The hook is installed by assigning `router.lifespan_context` rather than by passing
`lifespan=` to `FastAPI(...)`, because the app is constructed by `create_app`, which
several slices append routers to. Reaching into that call to thread a lifespan argument
through would make this module a co-writer of a file with many, for a hook that has
nothing to do with routing.

**Why the periodic sweeps start here too.** They are the platform's two invokers -- the
webhook deliveries and the Session-pod reclamation -- and the property they need is that
a control plane which is serving requests is a control plane which is sweeping. This
hook is the only place that holds both halves of that: it runs once per serving process,
and it has an exit to stop the tasks from. One lifespan and not two, because assigning
`router.lifespan_context` twice would silently keep only the second, and the one it
dropped would be the pool disposal or the sweeps depending on the order somebody wrote
them in. The sweeps are nested inside the disposal so the shutdown order is the only one
that works: every task is cancelled and awaited first, and only then is the pool the
tasks were querying through taken away.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from managed_agent.composition import (
    build,
    install_platform_logging,
    pod_runner_from_environment,
)
from managed_agent.control.api.app import create_app
from managed_agent.control.sweep_loop import sweep_interval_from_env, sweeping


def build_app() -> FastAPI:
    """Wire the platform from the environment and return the app that serves it.

    Reads `DATABASE_URL`; its absence raises rather than defaulting, so an unconfigured
    process fails to start instead of starting and reaching the wrong database. Returns
    a new app on every call — two calls mean two pools, which is why this is a factory
    uvicorn invokes once and not a module-level singleton anything can import twice.

    Whether this process places Session pods is the composition root's question, not
    this function's: the answer names a concrete cluster client and `composition.py` is
    the only module allowed to name one. It answers with a cluster when the environment
    names a pod manifest and with None when it does not, in which case the process
    refuses every Turn -- and neither case is a line here.

    Also reads `MAP_SWEEP_INTERVAL_S`, which has no default: a process started on a
    guessed cadence would look configured from every angle while the number deciding
    how long a tenant waits for a callback was nobody's decision. *Which* sweeps run is
    again the composition root's answer -- a control plane with no cluster wires one of
    the two -- and this function starts whatever it was handed, an empty tuple included.
    """
    install_platform_logging()
    platform, engine = build(pod_runner=pod_runner_from_environment())
    # Read after `build` and not before it, so a process missing its database URL still
    # fails on that -- the earlier read would raise first and an operator would fix the
    # wrong variable. Nothing is open yet to leak either way: the pool is lazy, so an
    # engine whose first checkout never happens holds no connection.
    every_s = sweep_interval_from_env()
    app = create_app(platform)

    @asynccontextmanager
    async def _sweep_while_serving_then_dispose_the_pool(
        _: FastAPI,
    ) -> AsyncIterator[None]:
        try:
            async with sweeping(platform.sweeps, every_s=every_s):
                try:
                    yield
                finally:
                    # Innermost, so the Turns stop before the sweeps that watch them
                    # and long before the pool both are writing through. A Turn
                    # cancelled here is left open on purpose and collected by
                    # `AbandonedTurnSweeper` from the replica still serving; closing it
                    # from inside a task being cancelled is not available, because
                    # every further await in that task is cancelled too.
                    await platform.background_turns.aclose()
        finally:
            await engine.dispose()

    app.router.lifespan_context = _sweep_while_serving_then_dispose_the_pool
    return app
