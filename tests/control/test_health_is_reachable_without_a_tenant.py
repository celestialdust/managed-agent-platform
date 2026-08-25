"""The probe route: what a kubelet gets, and what it must not have to present.

Tier 1, no cluster and no database. The app is built over a Platform whose ports are
never touched, because the point of the route under test is that it touches none of
them -- so a Platform stubbed with objects that raise on any call is the strongest
available statement of that, and the case that drives the route would fail if the route
grew a read.
"""

from __future__ import annotations

from typing import Any

import httpx

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes.health import HEALTH_ROUTE


class _Explodes:
    """Any attribute access is a failure. Stands in for every port the app is given."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the probe reached a port: {name}")


def _app() -> Any:
    """The real app over ports that cannot be used.

    The dict is annotated `dict[str, Any]` rather than cast afterwards: a cast on the
    constructed Platform would leave every keyword argument graded against the port
    types it deliberately does not satisfy.
    """
    ports: dict[str, Any] = {
        field: _Explodes() for field in Platform.__dataclass_fields__
    }
    return create_app(Platform(**ports))


async def test_the_probe_answers_with_no_headers_at_all() -> None:
    """No tenant header, no credential, no query string. A kubelet has none of them."""
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://probe"
    ) as client:
        answered = await client.get(f"/v1{HEALTH_ROUTE}")
    assert answered.status_code == 200
    assert answered.json() == {"status": "ok"}


async def test_the_probe_touches_no_port_on_the_platform() -> None:
    """The positive control for the case above: every port raises on any access, so a
    route that read one would fail rather than pass quietly."""
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://probe"
    ) as client:
        answered = await client.get(f"/v1{HEALTH_ROUTE}")
    assert answered.status_code == 200


async def test_a_tenant_header_changes_nothing() -> None:
    """Sending one is allowed and ignored. If the route ever grew a tenant dependency,
    this and the no-headers case would disagree, and that is why both exist."""
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://probe"
    ) as client:
        with_header = await client.get(
            f"/v1{HEALTH_ROUTE}",
            headers={TENANT_HEADER: "00000000-0000-0000-0000-000000000000"},
        )
    assert with_header.status_code == 200


def test_the_probe_route_declares_no_dependency() -> None:
    """Read off the route table rather than inferred from a 200: a dependency that
    happened to accept an absent header would pass every case above."""
    from managed_agent.control.api.routes import health

    route = next(
        r for r in health.router.routes if getattr(r, "path", None) == HEALTH_ROUTE
    )
    assert not getattr(route, "dependencies", ()), route


async def test_the_app_serves_the_probe_path_the_manifest_names() -> None:
    """The prefix is load-bearing: create_app mounts every router under /v1, so the
    served path is /v1/healthz and a manifest probing /healthz would never become ready.

    Driven over the app rather than read off `app.routes`: this FastAPI version keeps an
    included router as one opaque entry whose own `path` is None, so a route table walk
    finds neither spelling and would pass on a mounted-nowhere router. Two requests
    settle it -- the prefixed path answers and the bare one does not exist.
    """
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://probe"
    ) as client:
        prefixed = await client.get(f"/v1{HEALTH_ROUTE}")
        bare = await client.get(HEALTH_ROUTE)
    assert prefixed.status_code == 200
    assert bare.status_code == 404, "the manifest must probe the prefixed path"
