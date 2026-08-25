"""What a Session's own commands can reach, measured inside the sandbox on real pods.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NOTHING RAN.

`Environment.allowed_domains` is the one field on a shape that WIDENS what an agent
can do, and whether it works at all is downstream of code this repository does not own:
codex compiles the managed keys into a proxy, starts it, and enforces the allowlist.
`tests/control/test_egress_is_bounded_or_absent.py` grades the document we emit. Nothing
in that file, or in any other file that runs offline, can tell you whether the runtime
reads it -- so this one asks the kernel.

**Two pods, one Environment apart.** One shape grants a domain, the other grants none,
and the pods are otherwise identical: same image, same instructions, same probe script
byte for byte. A single pod could only report what it reached, and "the connection
failed" is the same observation whether egress was bounded or the feature was inert. Two
pods make the comparison, and the comparison is the finding.

**Three destinations per pod, because a grant that leaks is worse than one that does not
work.** The granted name must connect. A name that was NOT granted must not -- otherwise
the allowlist is decoration. And this cluster's own control plane must not, because it
answers without authentication (`deploy/k8s/network-policies.yaml` says so in those
words) and a Session that could dial it could act as the platform.

No model is asked anything. The Turn exists only to make the control plane place a pod;
every measurement is a command run under `codex sandbox` through `kubectl exec`.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest
from cluster_access import NAMESPACE, forwarded, kubectl

from managed_agent.control.pod_config.compiler import PROFILE_NAME, WORKSPACE_ROOT
from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import SessionId

_CONTROL_PLANE: Final = "deploy/control-plane"
_CONTROL_PLANE_PORT: Final = 8080
_REPOSITORY: Final = "map/session-shim"
_REGION: Final = "us-east-1"
_MODEL: Final = "gsds-claude-opus-4-6"
TENANT_HEADER: Final = "X-Tenant-Id"

_GRANTED: Final = "pypi.org"
_NOT_GRANTED: Final = "example.com"
_CLUSTER: Final = f"control-plane.{NAMESPACE}.svc.cluster.local"

_TURN_DEADLINE_S: Final = 600
_SECRET_SUFFIXES: Final = ("compiled", "requirements", "shim-token")

requires_the_cluster = pytest.mark.skipif(
    __import__("os").environ.get("MAP_CLUSTER_TESTS") != "1",
    reason="MAP_CLUSTER_TESTS=1 places real pods and calls a real model",
)

_REACH = '''
import os, socket, sys, urllib.request

def say(label, call):
    """One labelled line per attempt, and never a raise: a refusal is a finding."""
    try:
        value = call()
    except BaseException as err:
        print(f"{label}=REFUSED {type(err).__name__}: {str(err)[:120]}")
    else:
        print(f"{label}=REACHED {value}")


def tcp(host, port):
    sock = socket.create_connection((host, port), 8)
    peer = sock.getpeername()
    sock.close()
    return peer[0]


def fetched(url):
    with urllib.request.urlopen(url, timeout=15) as answer:
        return f"status={answer.status} bytes={len(answer.read(64))}"


print("proxy-env=" + ",".join(
    f"{k}={v}" for k, v in sorted(os.environ.items())
    if "proxy" in k.lower() or "PROXY" in k
) or "proxy-env=none")
say("granted-tcp", lambda: tcp("GRANTED", 443))
say("granted-https", lambda: fetched("https://GRANTED/simple/"))
say("ungranted-tcp", lambda: tcp("UNGRANTED", 443))
say("ungranted-https", lambda: fetched("https://UNGRANTED/"))
say("cluster-tcp", lambda: tcp("CLUSTERHOST", 80))
say("cluster-http", lambda: fetched("http://CLUSTERHOST/v1/healthz"))
print("reach=complete")
'''


def _probe_script() -> str:
    body = (
        _REACH.replace("GRANTED", _GRANTED)
        .replace("UNGRANTED", _NOT_GRANTED)
        .replace("CLUSTERHOST", _CLUSTER)
    )
    return f"""
set -u
cat > /tmp/reach.py <<'REACH'
{body}
REACH
codex sandbox -P {PROFILE_NAME} --include-managed-config -C {WORKSPACE_ROOT} \\
  -- python3 /tmp/reach.py > /tmp/reach.out 2>&1
echo "reach-rc=$?"
cat /tmp/reach.out
echo probe=complete
"""


def _session_image() -> str:
    done = subprocess.run(
        (
            "aws",
            "ecr",
            "describe-images",
            "--repository-name",
            _REPOSITORY,
            "--region",
            _REGION,
            "--output",
            "json",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    details = json.loads(done.stdout)["imageDetails"]
    assert details, f"{_REPOSITORY} holds no images, so no Session pod can start"
    newest = max(details, key=lambda one: str(one["imagePushedAt"]))
    return (
        f"{newest['registryId']}.dkr.ecr.{_REGION}.amazonaws.com/"
        f"{_REPOSITORY}@{newest['imageDigest']}"
    )


def _client(base: str, tenant_id: str, timeout: int = 900) -> httpx.Client:
    return httpx.Client(
        base_url=base, timeout=timeout, headers={TENANT_HEADER: tenant_id}
    )


def _created(answered: httpx.Response) -> dict[str, Any]:
    assert answered.status_code == 201, answered.text
    body: dict[str, Any] = answered.json()
    return body


def _a_session(
    base: str, tenant_id: str, image: str, run: str, domains: tuple[str, ...]
) -> SessionId:
    with _client(base, tenant_id) as caller:
        environment = _created(
            caller.post(
                "/v1/environments",
                json={
                    "name": f"reach-{run}",
                    "runtime_image": image,
                    "allowed_domains": list(domains),
                },
            )
        )
        definition = _created(
            caller.post(
                "/v1/agents",
                json={
                    "name": f"reach-{run}",
                    "instructions": "You reply with one word.",
                    "model": _MODEL,
                    "skills_repository": "git@github.com:acme/skills.git",
                    "skills_revision": "0" * 39 + "a",
                    "skills": [],
                    "tool_servers": [],
                },
            )
        )
        session = _created(
            caller.post(
                "/v1/sessions",
                json={
                    "definition_id": definition["id"],
                    "environment_id": environment["id"],
                    "budget_minor_units": 500_000,
                    "budget_currency": "USD",
                    "retention_days": 1,
                },
            )
        )
    return SessionId(UUID(session["id"]))


def _place(base: str, tenant_id: str, session_id: SessionId) -> None:
    """One trivial Turn, whose only purpose is to make the control plane place a pod.

    The Turn is not the measurement and its answer is never read. A pod is what this
    file needs and the platform places one when a Turn arrives, so this is the cheapest
    prompt that gets one.
    """
    with _client(base, tenant_id) as caller:
        answered = caller.post(
            f"/v1/sessions/{session_id}/events",
            json={"prompt": "Reply with exactly the word READY and nothing else."},
            headers={"Idempotency-Key": uuid4().hex},
        )
    assert answered.status_code == 202, answered.text
    deadline = time.monotonic() + _TURN_DEADLINE_S
    while time.monotonic() < deadline:
        with _client(base, tenant_id, timeout=60) as caller:
            events = caller.get(f"/v1/sessions/{session_id}/events").json()["events"]
        if any(one["type"] in ("turn.completed", "turn.failed") for one in events):
            return
        time.sleep(3)
    pytest.fail(f"session {session_id} never reached a terminal event")


def _reach(session_id: SessionId) -> str:
    """The probe's transcript, run under the sandbox inside the runtime container."""
    return kubectl(
        "exec",
        pod_name_for(session_id),
        "-c",
        "agent-runtime",
        "--",
        "/bin/sh",
        "-c",
        _probe_script(),
        check=False,
    )


def _clean_up(session_id: SessionId) -> None:
    name = pod_name_for(session_id)
    kubectl("delete", "pod", name, "--ignore-not-found", "--wait=false", check=False)
    for suffix in _SECRET_SUFFIXES:
        kubectl(
            "delete", "secret", f"{name}-{suffix}", "--ignore-not-found", check=False
        )


@dataclass(frozen=True, slots=True)
class _Reached:
    granted: str
    ungranted: str


@pytest.fixture(scope="module")
def reached() -> Iterator[_Reached]:
    """Two pods, one Environment apart, and the transcript each produced.

    Both are torn down whatever happened, including a failure while placing the second:
    a run that died between them still created a Session the control plane placed a pod
    for, and three aborted runs once left forty-two squatting the namespace.
    """
    run = uuid4().hex[:8]
    tenant_id = str(uuid4())
    image = _session_image()
    with forwarded(_CONTROL_PLANE, _CONTROL_PLANE_PORT) as base:
        with_grant = _a_session(base, tenant_id, image, f"y{run}", (_GRANTED,))
        without = _a_session(base, tenant_id, image, f"n{run}", ())
        try:
            _place(base, tenant_id, with_grant)
            _place(base, tenant_id, without)
            yield _Reached(
                granted=_reach(with_grant),
                ungranted=_reach(without),
            )
        finally:
            _clean_up(with_grant)
            _clean_up(without)


@requires_the_cluster
def test_both_probes_ran_to_completion(reached: _Reached) -> None:
    """Asserted first: a probe that did not finish makes every line below absent, and
    an absent line reads exactly like a refusal."""
    assert "reach=complete" in reached.granted, reached.granted[-800:]
    assert "reach=complete" in reached.ungranted, reached.ungranted[-800:]


@requires_the_cluster
def test_the_transcripts_are_printed_for_the_record(reached: _Reached) -> None:
    """Not an assertion -- a place to read what the cluster actually said.

    Kept because this file's subject is code this repository does not own, and the first
    question about any failure below is what the runtime did, not what we expected.
    """
    print("--- with a granted domain ---")
    print(reached.granted)
    print("--- with no granted domain ---")
    print(reached.ungranted)


@requires_the_cluster
def test_the_only_route_out_of_the_sandbox_is_the_proxy(reached: _Reached) -> None:
    """The finding that makes every allowlist claim below airtight, and it was measured.

    **DNS does not resolve inside the sandbox at all.** Every raw `socket` attempt in
    both pods answered `gaierror: Name or service not known`, for the granted name as
    readily as the refused one. So there is no socket-level path around the allowlist to
    argue about: a command that wants the network has to speak to the loopback proxy the
    runtime injected, and the proxy is the thing holding the list.

    This is asserted rather than left as a happy accident because the alternative would
    change what the refusals below prove. If raw TCP worked, `ungranted-https` being 403
    would mean only that the proxy declines to tunnel -- a command could open its own
    socket and skip it. It cannot.

    It is also why a reader must not take `granted-tcp=REFUSED` for a failure: the
    granted destination is reachable, over HTTPS, through the proxy, which the case
    below asserts.
    """
    for transcript in (reached.granted, reached.ungranted):
        assert "granted-tcp=REFUSED gaierror" in transcript, transcript[-600:]
        assert "ungranted-tcp=REFUSED gaierror" in transcript, transcript[-600:]
        assert "cluster-tcp=REFUSED gaierror" in transcript, transcript[-600:]


@requires_the_cluster
def test_a_granted_domain_is_reachable(reached: _Reached) -> None:
    """The capability, end to end: an Environment named `pypi.org` and the agent's own
    command fetched from it.

    `CODEX_NETWORK_PROXY_ACTIVE=1` is asserted beside the fetch because the two answer
    different questions. The variable says the runtime read our managed keys and started
    its proxy -- the half this repository is responsible for. The 200 says the proxy
    then let this destination through.
    """
    assert "CODEX_NETWORK_PROXY_ACTIVE=1" in reached.granted, reached.granted[:400]
    assert "granted-https=REACHED status=200" in reached.granted, reached.granted[-800:]


@requires_the_cluster
def test_a_domain_that_was_not_granted_is_refused(reached: _Reached) -> None:
    """The allowlist is a list and not a switch.

    Without this the grant would be "network on", and the domains in the document would
    be decoration -- which is exactly the failure `test_egress_is_bounded_or_absent.py`
    refuses on paper. This is the same claim measured: same pod, same proxy, same run,
    one destination through and one refused at the tunnel.
    """
    assert "ungranted-https=REFUSED" in reached.granted, reached.granted[-800:]
    assert "403" in reached.granted, reached.granted[-800:]


@requires_the_cluster
def test_the_platforms_own_control_plane_is_refused(reached: _Reached) -> None:
    """The refusal that matters most, because of what is on the other side of it.

    The control plane answers without authentication -- `network-policies.yaml` says so
    in those words -- so a Session that could reach it could create Sessions,
    read another tenant's events, and place pods. It is refused here through the proxy
    even though the pod holds a granted domain, and the NetworkPolicy that would also
    forbid it is declared and not enforced by this cluster's CNI, so this is the guard
    that is actually running.
    """
    assert "cluster-http=REFUSED" in reached.granted, reached.granted[-800:]
    assert "cluster-http=REFUSED" in reached.ungranted, reached.ungranted[-800:]


@requires_the_cluster
def test_a_shape_that_granted_nothing_has_no_network_and_no_proxy(
    reached: _Reached,
) -> None:
    """The default, and the arm that makes every case above a comparison.

    Not merely "the fetch failed" -- the proxy was never started, which is what an
    Environment granting no domain is supposed to produce. A pod with the proxy running
    and an empty list would fail these fetches too and would be a different platform.
    """
    assert "proxy-env=\n" in reached.ungranted, reached.ungranted[:400]
    assert "CODEX_NETWORK_PROXY_ACTIVE" not in reached.ungranted
    assert "granted-https=REFUSED" in reached.ungranted, reached.ungranted[-800:]
