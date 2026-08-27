"""The readiness port answers over plain HTTP, and says the same thing the Session port
would.

This exists because of a failure that would only appear in a cluster. Once a Session pod
holds a certificate its Session port requires one back, and a kubelet `httpGet` probe
presents none -- so the probe is refused at the handshake, the pod never becomes Ready,
and it never enters DNS. Every Turn would then be undeliverable against a pod reading
`2/2 Running` with a healthy shim, which is the exact shape this repository already paid
for once from a different cause.

The second listener is the fix, and the two things that can go wrong with it are both
graded here: that it answers at all without a certificate, and that it answers about the
same pod as the app holding the runtime connection. A readiness route with its own idea
of readiness would report ready before the connection existed, which is the thing the
route is there to prove.
"""

from __future__ import annotations

from ipaddress import ip_address
from typing import cast

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from managed_agent.session_shim import serve
from managed_agent.session_shim.serve import (
    PROBE_BIND_HOST,
    PROBE_PORT,
    READY_ROUTE,
    SHIM_PORT,
    ServedSession,
    build_probe_app,
)


@pytest.fixture(autouse=True)
def _a_pod_with_no_connection_open() -> None:
    """Reset the shared readiness holder, which is module state two apps read."""
    serve._READINESS.served = None


async def _probe() -> int:
    async with AsyncClient(
        transport=ASGITransport(app=build_probe_app()), base_url="http://probe"
    ) as client:
        return (await client.get(READY_ROUTE)).status_code


async def test_a_pod_whose_runtime_connection_is_not_open_is_not_ready() -> None:
    """503 until the lifespan has connected, which is what keeps the pod out of DNS."""
    assert await _probe() == status.HTTP_503_SERVICE_UNAVAILABLE


async def test_a_pod_whose_runtime_connection_is_open_is_ready() -> None:
    """204 once the Session app's lifespan has set the shared holder.

    Set through the same field the lifespan sets rather than through a second flag, so a
    pod cannot be ready on one listener and not on the other.
    """
    serve._READINESS.served = cast(ServedSession, object())

    assert await _probe() == status.HTTP_204_NO_CONTENT


def test_the_probe_listener_is_bound_where_a_kubelet_can_reach_it() -> None:
    """**A kubelet `httpGet` probe dials the pod IP**, so this port must answer there.

    This case asserted the opposite for one commit -- that the listener was on loopback,
    with a docstring saying the kubelet probes over the pod's own loopback and that
    binding wide would put an unauthenticated endpoint on every Session pod. The first
    half is false: `HTTPGetAction.host` defaults to the pod's IP address and the kubelet
    runs in the node's network namespace, so a loopback listener has no route from it at
    all. Every Session pod would have hung un-Ready and stayed out of the headless
    Service, which is the same outage the separate probe port exists to prevent, moved
    one layer down from the port to the bind address.

    The second half was a real concern with the wrong remedy. What keeps this port off
    the pod network is `session-pod`'s NetworkPolicy -- the control plane, on 8081, and
    nothing else -- while the node's probe traffic arrives because the CNI exempts the
    node from policy so that probes can work at all. A bind address decided neither.

    Graded as "not a loopback address" rather than against `0.0.0.0`, because the
    property is reachability from another network namespace and any address with that
    property is correct here. Asserting the literal is what let the broken value pass.
    """
    assert not ip_address(PROBE_BIND_HOST).is_loopback, (
        f"the probe listener is bound to {PROBE_BIND_HOST}, which a kubelet dialling "
        "the pod IP from the node's namespace can never reach"
    )
    # Compared through `int()` because mypy narrows both to their literal values and
    # calls the comparison non-overlapping -- which is true today and is exactly the
    # property this asserts, so the check has to survive being obviously true.
    assert int(PROBE_PORT) != int(SHIM_PORT)


def test_the_probe_app_carries_the_readiness_route_and_nothing_else() -> None:
    """One route, because everything else on this port would be unauthenticated.

    The Session port's routes are reached with a bearer token and, once this pod has a
    certificate, a client certificate as well. Nothing on this listener has either, so
    the surface has to stay exactly the route the kubelet needs.
    """
    paths = {
        path
        for path in (getattr(route, "path", "") for route in build_probe_app().routes)
        if path.startswith("/session")
    }

    assert paths == {READY_ROUTE}
