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

import base64
import socket
import ssl
import subprocess
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import cache
from typing import Final
from uuid import uuid4

import httpx
import pytest

from managed_agent.control.session.pod_tls import dial_context
from managed_agent.core.tls.session_certificate import InternalCa
from managed_agent.session_shim.pod_channel import shim_host

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


def submit_a_turn(
    caller: httpx.Client, session_id: object, prompt: str
) -> httpx.Response:
    """POST one Turn with the header it requires, refusing to return a refusal quietly.

    Both halves of this exist because of one incident, and neither is optional. The
    header is mandatory (`routes/turns.py`) and omitting it is answered `400` in
    milliseconds with the missing field named. The status check is the half that
    actually cost the time: a case that submits and goes straight to polling converts
    every instant, precise refusal this route can give into the same vague "no terminal
    event" timeout -- fifteen minutes, in the run that prompted this, pointing a reader
    at the pod, the placement and the deadline, which were all fine.

    Here rather than in each file because the four lines were hand-written in six of
    them and the seventh was about to be. A rule that lives in one file's comments is
    not a rule; it is that file's good luck.

    Returns the response, so a caller that wants the timing or the body still has it.
    """
    answered = caller.post(
        f"/v1/sessions/{session_id}/events",
        json={"prompt": prompt},
        headers={"Idempotency-Key": uuid4().hex},
    )
    assert answered.status_code == 202, (
        "the submission was refused, so no Turn ever ran and anything polling for one "
        f"would time out having learnt nothing. The route said: {answered.text}"
    )
    return answered


MODEL: Final = "gsds-claude-opus-4-6"
"""The model every bare live Session is opened on.

One name in one place because a rename that misses a call site fails at Session
creation, minutes into a run, as a `400` naming a field rather than a model.
"""

_A_REACHABLE_REVISION: Final = "0" * 39 + "a"
"""A well-formed 40-character commit for a repository under no eval gate.

The registration route refuses a revision its repository's CI gate rejected, and a
repository nobody has ever submitted a run for is not under the gate at all. So this
registers freely while still being the shape the field parses.
"""


def environment_payload(name: str, runtime_image: str) -> dict[str, object]:
    """The body `POST /v1/environments` takes for a bare environment."""
    return {"name": name, "runtime_image": runtime_image}


def definition_payload(name: str, instructions: str) -> dict[str, object]:
    """The body `POST /v1/agents` takes for a definition with nothing attached.

    No skills and no tool servers: a live case asserting something about the platform
    wants every attachment absent, so a reader has one fewer thing to rule out.
    """
    return {
        "name": name,
        "instructions": instructions,
        "model": MODEL,
        "skills_repository": "git@github.com:acme/skills.git",
        "skills_revision": _A_REACHABLE_REVISION,
        "skills": [],
        "tool_servers": [],
    }


def session_payload(definition_id: object, environment_id: object) -> dict[str, object]:
    """The body `POST /v1/sessions` takes, with a budget large enough not to be the
    thing that ends a live case.

    `retention_days` is 1 because these Sessions are debris: the run deletes the pod,
    and the shortest retention the field accepts is the least a test should ask the
    platform to keep.
    """
    return {
        "definition_id": str(definition_id),
        "environment_id": str(environment_id),
        "budget_minor_units": 500_000,
        "budget_currency": "USD",
        "retention_days": 1,
    }


def open_a_bare_session(
    caller: httpx.Client, name: str, runtime_image: str, instructions: str
) -> str:
    """Register an environment and a definition, open a Session on them, return its id.

    Three posts rather than one because that is what the API asks for, and each of the
    three is checked here: a `400` on the second surfaces as a named field now, instead
    of as an unexplained `KeyError` on `["id"]` one line later.

    The payloads come from the builders above rather than being written inline, because
    `tests/pod/test_the_create_payloads_still_parse.py` validates *those* against the
    request models the routes actually declare. A payload written inline here would be
    outside that guard and free to drift until a live run paid for it.
    """
    ids = []
    for route, body in (
        ("/v1/environments", environment_payload(name, runtime_image)),
        ("/v1/agents", definition_payload(name, instructions)),
    ):
        answered = caller.post(route, json=body)
        assert answered.status_code in (200, 201), (
            f"{route} refused the body this helper builds, so no Session exists and "
            f"nothing downstream can run. The route said: {answered.text}"
        )
        ids.append(answered.json()["id"])
    environment_id, definition_id = ids
    answered = caller.post(
        "/v1/sessions", json=session_payload(definition_id, environment_id)
    )
    assert answered.status_code in (200, 201), (
        "/v1/sessions refused the body this helper builds, so no Session exists. "
        f"The route said: {answered.text}"
    )
    created: str = answered.json()["id"]
    return created


@dataclass(frozen=True, slots=True)
class ShimDial:
    """What a port-forward to a Session pod's shim needs, in whichever mode it is in.

    Three fields rather than one context, because two of them are not TLS: the scheme
    the URL has to carry, and the SNI name the certificate is verified against. A caller
    holding only a context would still have to decide both, and deciding them beside a
    context that came from somewhere else is how the two disagree.
    """

    scheme: str
    verify: ssl.SSLContext | bool
    extensions: dict[str, str]

    def base(self, forwarded_base: str) -> str:
        """`forwarded()`'s URL with the scheme this dial actually speaks."""
        return forwarded_base.replace("http://", f"{self.scheme}://", 1)


@cache
def internal_ca_of_the_cluster() -> InternalCa | None:
    """The CA the deployed control plane signs with, or None if it runs without one.

    Read out of the Secret rather than minted here. A CA generated locally would produce
    a client certificate no pod trusts and a trust bundle that verifies no pod, so every
    dial would fail against a perfectly healthy platform -- and the failure would look
    exactly like the one this is meant to detect.

    Cached because both sides of a live case read it: the placement signs the pod's
    certificate with it, and the dial verifies against it. Two reads could straddle a
    rotation and hand the case a pod whose certificate its own trust bundle refuses --
    a failure indistinguishable from the one the case exists to catch.
    """
    encoded = kubectl(
        "get",
        "secret",
        "map-control-plane",
        "-o",
        "jsonpath={.data.internal-ca-cert}",
        check=False,
    ).strip()
    if not encoded:
        return None
    key = kubectl(
        "get", "secret", "map-control-plane", "-o", "jsonpath={.data.internal-ca-key}"
    ).strip()
    return InternalCa.from_pem(
        key_pem=base64.b64decode(key), certificate_pem=base64.b64decode(encoded)
    )


def shim_dial(pod_name: str) -> ShimDial:
    """How to reach `pod_name`'s shim through a forward, mTLS or plain.

    **The SNI name is the whole difficulty.** A forward lands on `127.0.0.1`, and the
    pod's certificate names it by the address the control plane dials in the cluster --
    so verifying against the URL's host refuses a correct certificate every time. The
    `sni_hostname` extension is what httpx passes to the TLS handshake as the server
    name, and hostname verification then runs against *that*. So this is not a weakened
    check: the certificate is still required to name the pod, and the name it must carry
    comes from the same `shim_host` expression that put it in the certificate.

    The plain-HTTP branch is not a fallback for a failure. A platform with no CA
    configured serves the Session port over HTTP by design, and a case that demanded TLS
    there would fail on a cluster that is behaving exactly as deployed.
    """
    ca = internal_ca_of_the_cluster()
    if ca is None:
        return ShimDial(scheme="http", verify=True, extensions={})
    return ShimDial(
        scheme="https",
        verify=dial_context(ca),
        extensions={"sni_hostname": shim_host(pod_name, NAMESPACE)},
    )
