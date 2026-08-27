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

**No Turn, and no control plane, and that is what ADR-041 changed here.** This file used
to create two Sessions through the public API, submit one throwaway Turn each purely to
make the control plane place a pod, and then `kubectl exec` the probe into those pods
afterwards. A pod is now leased for exactly one Turn and destroyed when that Turn ends,
so by the time the Turn the probe was waiting on is over there is nothing left to exec
into -- the exec failed, its error was swallowed, and six cases read an empty transcript
as six refusals. The Turn was never the measurement; its own helper said so. So the pods
are built here directly from `deploy/k8s/session-pod.yaml`, which is a pod the control
plane never leased and therefore never releases, and every measurement is a command run
under `codex sandbox` exactly as before.

**What that costs, stated plainly.** Building the pods here means the documents they
mount are compiled in this process rather than fetched from a Session the API created,
so the hop from `POST /v1/environments` to the Environment record no longer runs inside
this file. That hop is graded offline -- `tests/control/test_environment_lifecycle.py`
and `test_environment_reference.py` carry `allowed_domains` through the route and the
store -- and the hop this file still owns is the one nothing else covers: a real
`Environment` through `compile_session_config` into a real pod, and out to the kernel.
What no single test walks any more is the whole chain in one piece. That is the price of
observing the property at all under the lease, and the alternative -- asking a model,
inside a Turn, to try the fetches and report what happened -- would rest a security
finding on an LLM's willingness to cooperate and report verbatim.

The two arms cannot both read the Secret names the manifest spells, since they differ
only in what those documents say, so each pod is pointed at a pair of its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final
from uuid import uuid4

import pytest
import yaml

# The pod wrappers, imported rather than copied, for the reason
# `test_a_confined_command_runs_in_a_session_pod.py` gives where it does the same: this
# would be the third copy. By bare module name because `tests/` carries no
# `__init__.py`, so pytest's own sys.path entry is what resolves it.
from test_the_pod_materialises_its_sandbox_targets import (
    _EXPIRY,
    _TOKEN_KEY,
    _image,
    _kubectl,
    _probe_pod,
    _secret,
    _transcript,
    drop_the_workspace_volume,
    make_the_workspace_volume,
)

from managed_agent.control.pod_config import compiler as config_compiler
from managed_agent.control.pod_config.compiler import PROFILE_NAME, WORKSPACE_ROOT
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.registration.definition import AgentDefinition, SkillsRevision
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord

_NAMESPACE: Final = "map-egress-reach"

_GRANTED: Final = "pypi.org"
_NOT_GRANTED: Final = "example.com"
_CLUSTER: Final = "control-plane.map-dev.svc.cluster.local"
"""The platform's own control plane, addressed across namespaces on purpose.

These pods live in a namespace of their own and the destination has to stay the real
one: the finding is that a confined command cannot reach the service that answers
without authentication, and a copy of that service would be a different claim.
"""

requires_the_cluster = pytest.mark.skipif(
    __import__("os").environ.get("MAP_CLUSTER_TESTS") != "1",
    reason=(
        "MAP_CLUSTER_TESTS=1 creates a namespace of its own and two pods in it, and "
        "reaches the public internet from inside them. SKIPPED MEANS NOTHING RAN."
    ),
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


def _compiled_for(
    image: str, domains: tuple[str, ...]
) -> config_compiler.CompiledConfig:
    """The documents the compiler emits for a shape granting exactly `domains`.

    Compiled rather than hand-written, and this is the seam the file is here to grade:
    the allowlist has to travel from an `Environment` through `compile_session_config`
    into the document the runtime reads, and a copy of that document written here would
    be free to name something the compiler never emits.

    A near-twin of `_compiled` in the module the wrappers come from, which differs in
    one argument: that one pins an empty allowlist, because its own subject is the
    sandbox's filesystem targets and egress would be noise in it. Parameterising it
    there would put this file's variable into a module that has no use for it, so the
    second copy is deliberate and it is the second rather than the third.

    The signing key and the expiry are the same test-only pair, reaching nothing outside
    the namespace this file deletes. No container here starts a Turn or dials the Tool
    Gateway -- the runtime's command is replaced by the probe -- so the token in the
    document is never presented to anything, and both arguments exist only because the
    compiler refuses to emit a document without them.
    """
    return config_compiler.compile_session_config(
        SessionRecord(
            id=new_session_id(),
            tenant_id=TenantId(uuid4()),
            definition_id=new_definition_id(),
            definition_revision="rev-1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=10_000,
            budget_currency="USD",
            retention_days=30,
        ),
        tool_gateway_url="https://tool-gateway.map.internal/mcp",
        model_gateway_url="http://model-gateway.map-dev.svc.cluster.local/v1",
        definition=AgentDefinition(
            name="egress-probe",
            instructions="Nothing here starts a Turn.",
            model="gpt-5-codex",
            skills_repository="git@github.com:acme/skills.git",
            skills_revision=SkillsRevision("0" * 39 + "a"),
        ),
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="egress-probe",
            runtime_image=image,
            denied_paths=(),
            allowed_domains=domains,
        ),
        session_token_key=_TOKEN_KEY,
        session_token_expiry_epoch_s=_EXPIRY,
    )


def _point_at_its_own_documents(pod: dict[str, Any], name: str) -> dict[str, str]:
    """Give this pod Secrets no other pod in the namespace shares, and name them back.

    The manifest spells one Secret name per document, and the two pods here differ ONLY
    in what those documents say -- so they cannot both read the names it spells. The
    alternative is a namespace per arm, which doubles the workspace volume and the
    teardown in order to separate two pods that otherwise want to be as alike as it is
    possible to make them.

    Returns the new name per volume so the caller creates exactly the Secrets the pod
    will mount, rather than a list of names written out twice and free to disagree.

    The assertion is the guard on the pruning `_probe_pod` does. It drops the shim, and
    with it every volume the shim was the only mounter of -- `shim-token` among them --
    so a pod arriving here with a third Secret volume is one whose containers changed
    shape, and it would be pointed at a Secret this function never created.
    """
    renamed = {}
    for volume in pod["spec"]["volumes"]:
        secret = volume.get("secret")
        if secret is None:
            continue
        renamed[volume["name"]] = f"{name}-{volume['name']}"
        secret["secretName"] = renamed[volume["name"]]
    assert set(renamed) == {"compiled", "requirements"}, renamed
    return renamed


def _labelled(transcript: str, label: str) -> str:
    """The one probe line carrying `label`, or the empty string if it never printed.

    Read by label rather than searched for as a substring of the whole transcript,
    because a transcript is built from container logs and status fields and is wider
    than the probe's own output. A bare `"403" in transcript` could be satisfied by
    something with no connection to the destination it is supposed to be about, and a
    case that can be satisfied by an unrelated line is a case that cannot fail honestly.
    """
    for line in transcript.splitlines():
        if line.startswith(f"{label}="):
            return line
    return ""


@dataclass(frozen=True, slots=True)
class _Reached:
    granted: str
    ungranted: str


@pytest.fixture(scope="module")
def reached() -> Iterator[_Reached]:
    """Two pods, one Environment apart, and the transcript each produced.

    Both are applied before either transcript is read, so the two arms run against the
    same cluster at the same moment: a comparison whose halves were measured ten minutes
    apart is a comparison a passing deploy could have changed underneath.

    The namespace is created and deleted here, which is what takes the pods with it --
    and the volume is dropped separately, because `Retain` leaves it behind otherwise.
    The create tolerates a namespace an interrupted run left behind rather than failing
    the tier on it; if it is genuinely unusable, applying the pods below says so.
    """
    image = _image()
    _kubectl("create", "namespace", _NAMESPACE, check=False)
    try:
        make_the_workspace_volume(_NAMESPACE)
        pods = {}
        for label, domains in (("granted", (_GRANTED,)), ("ungranted", ())):
            name = f"egress-{label}"
            pod = _probe_pod(
                name, image=image, namespace=_NAMESPACE, probe=_probe_script()
            )
            renamed = _point_at_its_own_documents(pod, name)
            documents = _compiled_for(image, domains)
            _secret(
                renamed["compiled"],
                {"config.toml": documents.config_toml},
                namespace=_NAMESPACE,
            )
            _secret(
                renamed["requirements"],
                {"requirements.toml": documents.requirements_toml},
                namespace=_NAMESPACE,
            )
            pods[label] = pod
        for pod in pods.values():
            _kubectl("apply", "-n", _NAMESPACE, "-f", "-", stdin=yaml.safe_dump(pod))
        yield _Reached(
            granted=_transcript("egress-granted", namespace=_NAMESPACE),
            ungranted=_transcript("egress-ungranted", namespace=_NAMESPACE),
        )
    finally:
        _kubectl("delete", "namespace", _NAMESPACE, "--ignore-not-found", check=False)
        drop_the_workspace_volume(_NAMESPACE)


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
        for label in ("granted-tcp", "ungranted-tcp", "cluster-tcp"):
            line = _labelled(transcript, label)
            assert line.startswith(f"{label}=REFUSED gaierror"), (
                f"{label!r} was {line!r}\n{transcript[-600:]}"
            )


@requires_the_cluster
def test_a_granted_domain_is_reachable(reached: _Reached) -> None:
    """The capability, end to end: an Environment named `pypi.org` and a command in the
    pod fetched from it.

    `CODEX_NETWORK_PROXY_ACTIVE=1` is asserted beside the fetch because the two answer
    different questions. The variable says the runtime read our managed keys and started
    its proxy -- the half this repository is responsible for. The 200 says the proxy
    then let this destination through.
    """
    assert "CODEX_NETWORK_PROXY_ACTIVE=1" in _labelled(reached.granted, "proxy-env"), (
        reached.granted[:800]
    )
    assert _labelled(reached.granted, "granted-https").startswith(
        "granted-https=REACHED status=200"
    ), reached.granted[-800:]


@requires_the_cluster
def test_a_domain_that_was_not_granted_is_refused(reached: _Reached) -> None:
    """The allowlist is a list and not a switch.

    Without this the grant would be "network on", and the domains in the document would
    be decoration -- which is exactly the failure `test_egress_is_bounded_or_absent.py`
    refuses on paper. This is the same claim measured: same pod, same proxy, same run,
    one destination through and one refused at the tunnel.

    The status is read off the refusal's own line rather than looked for anywhere in the
    transcript, so `403` cannot be supplied by something that is not this attempt.
    """
    refused = _labelled(reached.granted, "ungranted-https")
    assert refused.startswith("ungranted-https=REFUSED"), reached.granted[-800:]
    assert "403" in refused, refused


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
    for transcript in (reached.granted, reached.ungranted):
        line = _labelled(transcript, "cluster-http")
        assert line.startswith("cluster-http=REFUSED"), (
            f"cluster-http was {line!r}\n{transcript[-800:]}"
        )


@requires_the_cluster
def test_a_shape_that_granted_nothing_has_no_network_and_no_proxy(
    reached: _Reached,
) -> None:
    """The default, and the arm that makes every case above a comparison.

    Not merely "the fetch failed" -- the proxy was never started, which is what an
    Environment granting no domain is supposed to produce. A pod with the proxy running
    and an empty list would fail these fetches too and would be a different platform.

    The proxy line has to be PRESENT and empty. Asserting only that
    `CODEX_NETWORK_PROXY_ACTIVE` is absent from the transcript would be satisfied by a
    probe that never printed the line at all, which is the same evidence a pod that
    never started produces.
    """
    assert _labelled(reached.ungranted, "proxy-env") == "proxy-env=", reached.ungranted[
        :800
    ]
    assert "CODEX_NETWORK_PROXY_ACTIVE" not in reached.ungranted
    assert _labelled(reached.ungranted, "granted-https").startswith(
        "granted-https=REFUSED"
    ), reached.ungranted[-800:]
