"""Whether this process is up, answered without reading anything it depends on.

A kubelet has no tenant and no credential, so this is the one route here that takes
neither. Every other router in this app resolves a tenant from a request header, and a
probe that had to present one would mean a long-lived credential in the cluster for the
sake of a health check.

**It reads no database, and that is the design rather than an economy.** A readiness
probe that queries Postgres removes every replica from service the moment the database
blips -- turning a hiccup into a total outage -- and the connection pool already
discards a dead connection and opens a fresh one by itself. So what a 200 here means is
narrow and worth stating: this process started, imported its dependencies and is
accepting connections. It does **not** mean the schema is present or the database is
reachable. Whatever proves that has to drive a route that reads.

One route serves both probes. Liveness and readiness differ in what the kubelet does
with the answer -- restart the container, or take it out of a Service -- not in what the
process can truthfully say about itself, and a second path answering the same question
is a second thing to keep in step with the manifest.
"""

from typing import Final

from fastapi import APIRouter

router = APIRouter()

HEALTH_ROUTE: Final[str] = "/healthz"
"""The path, exported so a manifest test can read it instead of repeating it.

Mounted under this app's `/v1` prefix, so the served path is `/v1/healthz` -- the same
spelling the Model Gateway serves, which keeps the platform's three services at two
probe paths rather than three.
"""


@router.get(HEALTH_ROUTE)
async def healthz() -> dict[str, str]:
    """Answer that this process is serving. Reads nothing."""
    return {"status": "ok"}
