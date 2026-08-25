"""The tenant-facing REST surface's app factory.

It takes an already-wired Platform rather than constructing one, so a test drives
the real routes against in-memory ports and the composition root stays the only
place a concrete adapter is chosen.

Routers are attached here; each resource concern owns its own module, so two slices
adding two resources never edit one file.
"""

from fastapi import FastAPI

from managed_agent.composition import Platform
from managed_agent.control.api import refusals, schema
from managed_agent.control.api.request import beta
from managed_agent.control.api.routes import (
    agent_versions,
    agents_lifecycle,
    artifacts,
    audit,
    capacity,
    environments,
    events,
    files,
    health,
    registration_definitions,
    registration_servers,
    resources,
    session_list,
    session_threads,
    sessions,
    skills,
    skills_eval,
    stream,
    turns,
    vaults,
    webhooks,
)


def create_app(platform: Platform) -> FastAPI:
    app = FastAPI(title="Managed Agent Platform", version="v1")
    app.state.platform = platform
    # Before every router, and the order is load-bearing. This installs the
    # request-id middleware and the two exception handlers that envelope the
    # refusals this codebase does not author: a body FastAPI rejected before any
    # route was entered, and an exception that escaped one. A route added after
    # this line is covered by both. A router included before it would answer
    # those two cases in Starlette's shapes rather than ours -- the one
    # difference in this file that no route-level test would notice.
    #
    # `beta` is installed FIRST and `refusals` SECOND, which makes `beta` the
    # INNER of the two: Starlette runs the last-registered middleware outermost.
    # Measured, and load-bearing in one direction -- the inner middleware sees the
    # request id on its ContextVar, so a refusal from the beta check is attributed
    # to the request that caused it. Swapped, every one of them would come back as
    # `req_unattributed`, and nothing about the response shape would look wrong.
    beta.install_beta_header(app)
    refusals.install_request_envelope(app)
    # The published document, corrected after both. FastAPI attaches a 422 with its
    # own HTTPValidationError to every operation taking a body or a typed parameter,
    # and this app answers 400 with its own envelope instead -- no published code
    # carries 422 at all. A client is generated from the document rather than from the
    # code, so the uncorrected version is wrong about the commonest refusal on 28
    # operations, and no test that exercises the app can see it.
    schema.publish_the_schema_the_app_answers_with(app)
    # First, and it is the only router here that reads nothing off `platform`: a
    # probe answers about the process, not about what the process can reach.
    app.include_router(health.router, prefix="/v1")
    app.include_router(sessions.router, prefix="/v1")
    app.include_router(environments.router, prefix="/v1")
    app.include_router(registration_definitions.router, prefix="/v1")
    app.include_router(agent_versions.router, prefix="/v1")
    # After `agent_versions`, and the order is a deliberate no-op recorded so
    # nobody reorders it thinking it matters. This router's widest path is
    # `POST /agents/{agent_id}`, which cannot shadow
    # `/agents/{agent_id}/versions` -- Starlette matches on segment count first,
    # so a two-segment pattern is never tried against a three-segment path.
    app.include_router(agents_lifecycle.router, prefix="/v1")
    app.include_router(registration_servers.router, prefix="/v1")
    app.include_router(session_list.router, prefix="/v1")
    app.include_router(resources.router, prefix="/v1")
    app.include_router(artifacts.router, prefix="/v1")
    app.include_router(events.router, prefix="/v1")
    app.include_router(files.router, prefix="/v1")
    app.include_router(skills.router, prefix="/v1")
    app.include_router(skills_eval.router, prefix="/v1")
    app.include_router(stream.router, prefix="/v1")
    app.include_router(webhooks.router, prefix="/v1")
    app.include_router(turns.router, prefix="/v1")
    app.include_router(audit.router, prefix="/v1")
    # Beside `audit` because it shares that surface's two dependencies: a
    # reviewer principal read from a presented token, then authorized from the
    # claims alone. Both read across tenants, so both refuse a tenant principal
    # outright rather than scoping a query to it.
    app.include_router(capacity.router, prefix="/v1")
    app.include_router(session_threads.router, prefix="/v1")
    # Last, and independent of every router above it: nothing here reads a vault,
    # and this reads nothing any of them wrote. Its paths are all under `/vaults`,
    # which no other router claims a segment of.
    app.include_router(vaults.router, prefix="/v1")
    return app
