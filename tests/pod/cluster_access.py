"""Reaching the live `map-dev` cluster from a developer's machine.

Not a test. The plumbing every case under `tests/pod/` needs before it can assert
anything: run `kubectl` against one namespace, and hold open a local port that reaches
one in-cluster target.

Extracted because there were two copies and a third was about to be written, and the
copies had already drifted -- one waited for an HTTP answer on a route it named, the
other for a TCP accept, and only the second was right for a target whose readiness route
this module cannot know. Drift in a helper like this reads as the cluster misbehaving:
a forward that returns before it is listening surfaces three assertions later as a
connection refused against the wrong process.

A port-forward and not a Service address, everywhere. The reason used to be that
`control-plane.yaml` declared no Service at all, and it now declares one -- but a
`ClusterIP` is reachable only from inside the cluster, and these cases run on a
developer's machine. So the forward is what crosses that boundary rather than a choice
about addressing, and a case that wants a specific pod rather than an arbitrary replica
needs it regardless: `pod/<name>` is a target only a forward accepts.
"""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Final

import pytest

NAMESPACE: Final = "map-dev"
"""The one namespace every case here reads and writes.

A constant rather than a parameter because a case that could point at another namespace
would be a case whose failure does not tell you which cluster disagreed.
"""

_FORWARD_DEADLINE_S: Final = 30.0
"""How long a port-forward is given to accept a connection before the case fails.

Thirty seconds is generous for a process that only has to bind locally and open one
stream to the API server; it is short enough that a target which does not exist fails
the run rather than hanging it. A forward to a missing pod exits on its own and is
caught by the poll below rather than by this deadline.
"""


def kubectl(*argv: str, check: bool = True) -> str:
    """One `kubectl` call against `NAMESPACE`, returning its stdout.

    `check=False` is for the calls whose failure is expected and not interesting -- a
    delete of something already gone. Everything else fails the case with stderr
    attached, because a `kubectl` that returned non-zero and was read as empty output is
    indistinguishable from an object that is genuinely absent.
    """
    done = subprocess.run(
        ["kubectl", "-n", NAMESPACE, *argv],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if check and done.returncode != 0:
        pytest.fail(f"kubectl {' '.join(argv)} failed:\n{done.stderr}")
    return done.stdout


def a_free_port() -> int:
    """A local port nothing is listening on, chosen by the kernel.

    Bind zero and read back what was assigned. There is a race in principle -- the
    socket is closed before the forward binds it -- and it is preferable to a fixed
    port, which collides with whatever else a developer is forwarding and reports the
    collision as the cluster refusing a connection.
    """
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


@contextmanager
def forwarded(target: str, remote: int) -> Iterator[str]:
    """A local base URL reaching one in-cluster target, for as long as the block runs.

    `target` is whatever `kubectl port-forward` accepts -- `pod/<name>`,
    `deploy/<name>`, `svc/<name>` -- and is passed through rather than parsed, so a
    caller forwarding to a Deployment and a caller forwarding to one specific pod use
    one function. That distinction matters to a caller: a forward to `deploy/x` lands on
    an arbitrary replica, which is wrong for a case whose subject is a particular pod.

    Waits for a TCP accept and not for an HTTP answer. This module does not know the
    target's readiness route, and a helper that guessed one would report a healthy
    process as a failed forward. The cost is that a process listening but not yet
    serving is yielded as ready, which each caller's own first request surfaces.
    """
    port = a_free_port()
    process = subprocess.Popen(
        ["kubectl", "-n", NAMESPACE, "port-forward", target, f"{port}:{remote}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + _FORWARD_DEADLINE_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(
                    f"port-forward to {target} exited: {process.communicate()[1]}"
                )
            try:
                with closing(socket.create_connection(("127.0.0.1", port), timeout=1)):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail(f"port-forward to {target} never accepted a connection")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        process.wait(timeout=30)
