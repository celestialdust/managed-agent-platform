"""What a pod built from session-pod.yaml gives the sandbox, measured in the cluster.

Skipped unless MAP_CLUSTER_TESTS=1. SKIPPED MEANS NO CONTAINER RAN -- every other test
over this manifest reads YAML, and YAML cannot say whether a mkdir succeeded or whether
bwrap accepted an argv.

Three pods, and two of them are mutants that exist to make the first one evidence. Two
of the findings are absences -- a panic that no longer appears, a bwrap refusal that no
longer appears -- and an absence is satisfied by a pod that never pulled its image. So
the same run drives a copy with the scratch volume removed, where the panic MUST appear,
and a copy with the two mkdir lines removed, where the refusal MUST appear. Two mutants
rather than one because each isolates one of the two changes; with a single mutant,
either change alone would look sufficient.

This file used to drive a fourth pod, mounting a requirements document patched to drop
the leaf control-socket deny rule, as the positive control for a refusal the shipped
document still provoked. The compiler emits that document now, so the patch has nothing
to patch: the first pod IS that configuration, and the socket-bound half below asserts
what the patched pod used to. The pod that proves the refusal is still reachable in this
harness moved with the rule, to
`tests/pod/test_a_confined_command_runs_in_a_session_pod.py`, which mounts the leaf rule
back and requires bubblewrap to refuse.

EVERY POD RUNS THE PROBE TWICE, and the two halves are not interchangeable. The first
half runs with the runtime's control socket absent; the second starts the shipped
`codex app-server` command, waits for it to bind that socket, and asks the same
questions again. Whether the socket exists decides what the two control-path deny rules
compile to, and it was measured here rather than assumed: with the socket absent the pod
as shipped RUNS a confined command, and once the socket is bound the same pod refuses
with `Can't mkdir parents for /run/codex/ctl/app-server-control.sock`. A real Session is
always in the second state -- agent-runtime's own startupProbe is `test -S` on exactly
that path -- so a file that measured only the first would report this slice as closing
more than it does. `_phases` is the split, and every case names the half it is about.

A transcript is built out of container logs and a handful of short status fields, and
never out of the pod spec. That is load-bearing rather than tidy: the init container's
own comments quote the bwrap refusal this file asserts the absence of, so a transcript
that carried the spec would make two of these findings unfalsifiable.
`test_no_transcript_carries_the_manifest_s_own_prose` is the guard on that.

What this file cannot say: that a Turn works. It drives `codex sandbox`, which compiles
the same argv through the same helper, in the same container, at the same uid, under the
same seccomp profile -- and is not the caller a Turn uses. It says nothing either about
the arg0 helper directory a Turn's patch tooling needs, which is a further
refusal behind these. And the app-server it starts in the second half is started
to bind a socket: it is never dialled, no thread is opened, and no model provider
is reached.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import subprocess
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import pytest
import yaml

from managed_agent.control.pod_config import compiler as config_compiler
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.registration.definition import AgentDefinition, SkillsRevision
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "deploy" / "k8s" / "session-pod.yaml"
_NAMESPACE = "map-67-targets"
_REPOSITORY = "map/session-shim"
_REGION = "us-east-1"
_GATE = "MAP_CLUSTER_TESTS"
_RUNTIME = "agent-runtime"
_INIT = "seed-runtime-home"
_DEADLINE_SECONDS = 420

requires_the_cluster = pytest.mark.skipif(
    os.environ.get(_GATE) != "1",
    reason=(
        f"the cluster proof is opt-in: set {_GATE}=1 to run it. It needs kubectl "
        "pointed at map-dev and the aws CLI, creates its own namespace and three pods "
        "in it, and takes three to five minutes. SKIPPED MEANS NO CONTAINER RAN."
    ),
)

PANIC = "failed to create synthetic bubblewrap mount registry"
COLLISION = "Can't create file at /session/workspace/.agents: Is a directory"
CONTROL_PATH = "Can't mkdir parents for /run/codex/ctl"
RAN = "CONFINED-COMMAND-RAN"
BOUND_RAN = "SOCKET-BOUND-COMMAND-RAN"
SEED = "SEEDED-METADATA-BODY"

# Splits a transcript into its two halves. Deliberately not a substring of either marker
# above, so a phase boundary cannot be mistaken for a command having run.
PHASE_BOUND = "phase=socket-bound"

# A fragment of the init container's own commentary about the collision. It is asserted
# ABSENT from every transcript, because that comment quotes COLLISION verbatim: a
# transcript builder that ever picked up the pod spec would satisfy
# `COLLISION in transcripts["no-targets"]` without any container having run.
_MANIFEST_PROSE = "the protected-metadata mask makes a directory"

# This test's own signing key and a far-future expiry. Neither reaches anything outside
# the namespace this test deletes: see _compiled.
_TOKEN_KEY = b"a signing key that is thirty-two"
_EXPIRY = 4102444800

# For the one case in this file that compiles a document and never creates a pod. The
# compiler refuses an Environment whose image is not digest-pinned, and that case has no
# use for a real digest -- resolving one would put an ECR call behind a string check.
_A_DIGEST_NO_POD_PULLS = "registry.invalid/session@sha256:" + "0" * 64

PROBE = r"""
set -u
mkdir -p /tmp/probe 2>/dev/null && echo tmp-writable=ok || echo tmp-writable=FAIL
for p in /session/workspace/.codex /session/workspace/.agents; do
  if [ -d "$p" ]; then echo "target $p=directory"
  elif [ -e "$p" ]; then echo "target $p=NOT-A-DIRECTORY"
  else echo "target $p=MISSING"; fi
done
# Seeded only where the pod already created the directory, so the read refusal below is
# a refusal over real bytes rather than over a missing file -- and so the mutant that
# creates neither directory is not quietly handed one here.
if [ -d /session/workspace/.agents ]; then
  printf 'SEEDED-METADATA-BODY\n' > /session/workspace/.agents/seeded
  echo seeded=ok
else
  echo seeded=skipped
fi
codex sandbox -P map-session --include-managed-config -C /session/workspace \
  -- /bin/echo CONFINED-COMMAND-RAN 2>&1
codex sandbox -P map-session --include-managed-config -C /session/workspace \
  -- /bin/sh -c 'ls -ld /session/workspace/.agents 2>&1;
    cat /session/workspace/.agents/seeded 2>&1;
    : > /session/workspace/.agents/x 2>&1' 2>&1
# The same question from outside the sandbox, and it never prints the seeded bytes: the
# marker has to be absent from the whole transcript for the refusal above to mean the
# confined process could not read it, rather than meaning this line printed it twice.
if [ -d /session/workspace/.agents ]; then
  test "$(cat /session/workspace/.agents/seeded 2>/dev/null)" = SEEDED-METADATA-BODY \
    && echo host-seed-intact=ok || echo host-seed-intact=FAIL
  test -e /session/workspace/.agents/x \
    && echo confined-write-landed=YES || echo confined-write-landed=no
fi
# Everything above ran with the control socket ABSENT, which is not the state a Session
# is ever in when it takes a Turn. The runtime binds that socket before the pod is ready
# -- agent-runtime's startupProbe is `test -S` on exactly this path -- and whether the
# socket exists changes which bwrap operations the two control-path deny rules compile
# to. So the same questions are asked again below with the socket bound, by the shipped
# command from the shipped manifest rather than by something shaped like it.
echo phase=socket-bound
codex app-server --listen unix:///run/codex/ctl/app-server-control.sock \
  > /var/lib/map/codex/app-server.log 2>&1 &
i=0
while [ "$i" -lt 60 ]; do
  if [ -S /run/codex/ctl/app-server-control.sock ]; then break; fi
  i=$((i + 1))
  sleep 1
done
if [ -S /run/codex/ctl/app-server-control.sock ]; then
  echo socket-bound=ok
else
  echo socket-bound=FAIL
  echo "--- app-server log ---"
  cat /var/lib/map/codex/app-server.log 2>&1
fi
codex sandbox -P map-session --include-managed-config -C /session/workspace \
  -- /bin/echo SOCKET-BOUND-COMMAND-RAN 2>&1
codex sandbox -P map-session --include-managed-config -C /session/workspace \
  -- /bin/sh -c 'ls -ld /session/workspace/.agents 2>&1;
    cat /session/workspace/.agents/seeded 2>&1;
    : > /session/workspace/.agents/y 2>&1;
    ls -ld /run/codex/ctl 2>&1;
    cat /run/codex/ctl/app-server-control.sock 2>&1' 2>&1
if [ -d /session/workspace/.agents ]; then
  test -e /session/workspace/.agents/y \
    && echo bound-confined-write-landed=YES || echo bound-confined-write-landed=no
fi
echo probe=complete
"""


def _kubectl(*argv: str, stdin: str | None = None, check: bool = True) -> str:
    done = subprocess.run(
        ["kubectl", *argv],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if check and done.returncode != 0:
        pytest.fail(f"kubectl {' '.join(argv)} failed:\n{done.stderr}")
    return done.stdout


def _aws(*argv: str) -> Any:
    done = subprocess.run(
        ["aws", *argv, "--output", "json"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if done.returncode != 0:
        pytest.fail(f"aws {' '.join(argv)} failed:\n{done.stderr}")
    parsed: Any = json.loads(done.stdout)
    return parsed


def _secret(name: str, data: dict[str, str], *, namespace: str = _NAMESPACE) -> None:
    """A Secret in the caller's namespace, under a name the manifest already spells.

    The namespace is a parameter with this module's own as its default, because a
    sibling pod module reuses these wrappers and must create and delete a namespace of
    its own -- two modules sharing one namespace would race on the Secrets they apply
    under the names the manifest spells.

    Built here rather than through `kubectl create secret --from-file`, which would want
    the compiled documents on disk; base64 in a manifest keeps them in memory and keeps
    this from leaving a file behind on a failure.
    """
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace},
        "data": {
            key: base64.b64encode(value.encode()).decode()
            for key, value in data.items()
        },
    }
    _kubectl("apply", "-n", namespace, "-f", "-", stdin=yaml.safe_dump(manifest))


CODEX_VERSION: Final = "0.149.0"
"""The runtime version every empirical claim in this file was measured against.

Load-bearing rather than decorative. `_image()` below resolves the **newest push** in
the Session repository, so a newer image built on a newer codex-cli is picked up with no
signal that anything changed -- and these measurements are cited as settled fact in
production docstrings a reader will not re-derive (`config_compiler.py:33`, `:355`,
`:498`, `:642` and `environment_registry.py:54`), among them `_denied_paths`' inversion
argument and the nested-deny fatality claim, both of which bound what a floor accepts
from a tenant.

The failure this pins is the quiet one. If a newer runtime behaves differently and a
measurement flips, a test fails and that is the file working. If the measurements still
pass while the *reasoning* recorded in the compiler is now about a runtime nobody runs,
nothing anywhere says so. `map/session-shim` being immutable per tag stops an overwrite
and does nothing about `max(imagePushedAt)` moving.
"""


def _image() -> str:
    """The newest digest in the Session repository, resolved the way the push script
    prints it. The manifest's own reference is sixty-four zeros -- a placeholder that
    resolves to no image anywhere, so an unsubstituted pod fails at the pull."""
    account = _aws("sts", "get-caller-identity")["Account"]
    images = _aws("ecr", "describe-images", "--repository-name", _REPOSITORY)[
        "imageDetails"
    ]
    newest = max(images, key=lambda detail: str(detail["imagePushedAt"]))
    registry = f"{account}.dkr.ecr.{_REGION}.amazonaws.com"
    return f"{registry}/{_REPOSITORY}@{newest['imageDigest']}"


def _record() -> SessionRecord:
    return SessionRecord(
        id=new_session_id(),
        tenant_id=TenantId(uuid4()),
        definition_id=new_definition_id(),
        definition_revision="rev-1",
        grant=frozenset(),
        scope=(),
        budget_minor_units=10_000,
        budget_currency="USD",
        retention_days=30,
    )


def _compiled(image: str) -> config_compiler.CompiledConfig:
    """The documents the compiler actually emits, for the image these pods will run.

    Compiled rather than hand-written: the whole question is whether the pod
    materialises what THIS profile's deny rules name, and a copy of the document
    would be free to name something else.

    The signing key and the expiry are this test's own and reach nothing outside the
    namespace it deletes. No container here starts a Turn or dials the Tool Gateway --
    the runtime's command is replaced by the probe -- so the token in the document is
    never presented to anything, and the two arguments exist only because the compiler
    refuses to emit a document without them.
    """
    return config_compiler.compile_session_config(
        _record(),
        tool_gateway_url="https://tool-gateway.map.internal/mcp",
        model_gateway_url="http://model-gateway.map-dev.svc.cluster.local/v1",
        definition=AgentDefinition(
            name="map-67-probe",
            instructions="Nothing here starts a Turn.",
            model="gpt-5-codex",
            skills_repository="git@github.com:acme/skills.git",
            skills_revision=SkillsRevision("0" * 39 + "a"),
        ),
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="map-67-probe",
            runtime_image=image,
            denied_paths=(),
        ),
        session_token_key=_TOKEN_KEY,
        session_token_expiry_epoch_s=_EXPIRY,
    )


def _probe_pod(
    name: str, *, image: str, namespace: str = _NAMESPACE, probe: str = PROBE
) -> dict[str, Any]:
    """The manifest with the shim dropped, the runtime's command replaced by the probe,
    and every volume nothing mounts any more removed.

    The shim goes because it dials a socket nothing binds for it; dropping the
    volumes it was the only mounter of is what keeps this from needing a Secret
    invented for a token no container reads. The startup probe goes with it,
    because the runtime's own command is replaced -- the probe binds that socket
    itself, in its second half, and reports when it has.
    """
    pod: dict[str, Any] = copy.deepcopy(yaml.safe_load(_MANIFEST.read_text()))
    pod["metadata"]["name"] = name
    pod["metadata"]["namespace"] = namespace
    pod["spec"].pop("subdomain", None)
    pod["spec"]["containers"] = [
        container
        for container in pod["spec"]["containers"]
        if container["name"] == _RUNTIME
    ]
    runtime = pod["spec"]["containers"][0]
    runtime.pop("startupProbe", None)
    runtime["command"] = ["/bin/sh", "-c", probe]
    for container in pod["spec"]["initContainers"] + pod["spec"]["containers"]:
        container["image"] = image
    mounted = {
        mount["name"]
        for container in pod["spec"]["initContainers"] + pod["spec"]["containers"]
        for mount in container.get("volumeMounts", [])
    }
    pod["spec"]["volumes"] = [
        volume for volume in pod["spec"]["volumes"] if volume["name"] in mounted
    ]
    assert any(volume["name"] == "scratch" for volume in pod["spec"]["volumes"]), (
        "session-pod.yaml declares no scratch volume the runtime container mounts, so "
        "the sandbox helper has nowhere to write its mount registry and no confined "
        "command runs in any Session"
    )
    return pod


def _without_the_scratch_mount(pod: dict[str, Any]) -> dict[str, Any]:
    mutant = copy.deepcopy(pod)
    mutant["metadata"]["name"] += "-no-scratch"
    removed = {
        mount["name"]
        for container in mutant["spec"]["containers"]
        for mount in container["volumeMounts"]
        if mount["mountPath"] == "/tmp"
    }
    assert removed, "nothing was removed, so this mutant proves nothing"
    for container in mutant["spec"]["containers"]:
        container["volumeMounts"] = [
            mount for mount in container["volumeMounts"] if mount["name"] not in removed
        ]
    mutant["spec"]["volumes"] = [
        volume for volume in mutant["spec"]["volumes"] if volume["name"] not in removed
    ]
    return mutant


def _without_the_dot_path_mkdir(pod: dict[str, Any]) -> dict[str, Any]:
    """Removes the two paths from the mkdir and the two `test -d` lines with them.

    Leaving the `test -d` in would fail the init container under `set -eu` and the pod
    would never reach the runtime -- which is a true outcome but the wrong control: what
    this mutant has to show is bwrap refusing an argv, not a pod refusing to start.

    Comment lines are kept even where they name the same paths. They execute
    nothing, and the alternative -- a filter that also strips prose -- would make
    the count below depend on how the manifest is worded rather than on what it
    does.
    """
    mutant = copy.deepcopy(pod)
    mutant["metadata"]["name"] += "-no-targets"
    script = mutant["spec"]["initContainers"][0]["args"][0]
    dot_paths = ("/session/workspace/.agents", "/session/workspace/.codex")
    kept = [
        line
        for line in script.splitlines()
        if line.strip().startswith("#") or not any(path in line for path in dot_paths)
    ]
    assert len(kept) == len(script.splitlines()) - 3, (
        "expected exactly three executable lines naming the two dot-paths -- one mkdir "
        f"and two `test -d` -- and removed {len(script.splitlines()) - len(kept)}"
    )
    mutant["spec"]["initContainers"][0]["args"][0] = "\n".join(kept) + "\n"
    return mutant


def _transcript(name: str, *, namespace: str = _NAMESPACE) -> str:
    """Poll the pod to a terminal phase, then read every container's log.

    Polls the phase rather than the log, because `kubectl logs` against a pod whose
    container has not started fails, and a failed kubectl here would be reported as a
    missing directory rather than as a pod that had not been scheduled.

    A failed read contributes its own stderr instead of raising: every finding in
    this file is read out of a transcript, so a pod that died in its init container
    has to say so in the transcript rather than abort the fixture for all four.

    Deliberately does NOT include the pod spec. The init container's comments quote
    the bwrap refusal this file asserts the absence of, so a spec in the transcript
    would make that absence unfalsifiable. The status fields below are the exit
    code and reason only.
    """
    deadline = time.monotonic() + _DEADLINE_SECONDS
    phase = "unknown"
    while time.monotonic() < deadline:
        phase = _kubectl(
            "get", "pod", name, "-n", namespace, "-o", "jsonpath={.status.phase}"
        ).strip()
        if phase in ("Succeeded", "Failed"):
            break
        time.sleep(3)
    parts = [f"pod={name} phase={phase}"]
    for container in (_INIT, _RUNTIME):
        done = subprocess.run(
            ["kubectl", "logs", name, "-n", namespace, "-c", container],
            capture_output=True,
            text=True,
            timeout=120,
        )
        parts.append(f"--- {container} ---")
        parts.append(done.stdout)
        if done.returncode != 0:
            parts.append(f"[logs {container} rc={done.returncode}] {done.stderr}")
    parts.append(
        _kubectl(
            "get",
            "pod",
            name,
            "-n",
            namespace,
            "-o",
            "jsonpath="
            "{range .status.initContainerStatuses[*]}"
            "init/{.name} exit={.state.terminated.exitCode} "
            "reason={.state.terminated.reason} waiting={.state.waiting.reason};"
            "{end}"
            "{range .status.containerStatuses[*]}"
            "{.name} exit={.state.terminated.exitCode} "
            "reason={.state.terminated.reason} waiting={.state.waiting.reason};"
            "{end}",
            check=False,
        )
    )
    return "\n".join(parts)


@pytest.fixture(scope="module")
def transcripts() -> Iterator[dict[str, str]]:
    """Three pods in a namespace this test creates and deletes, and their transcripts.

    The compiled documents are rendered here by calling the compiler, so what the
    pods read is the real thing and not a copy of it. They go into this namespace's
    own Secrets under the names the manifest already spells, so the manifest needs
    no edit to find them -- and nothing in map-dev is read, written or borrowed.
    """
    image = _image()
    compiled = _compiled(image)
    _kubectl("create", "namespace", _NAMESPACE)
    try:
        _secret("map-session-compiled-config", {"config.toml": compiled.config_toml})
        _secret(
            "map-session-requirements",
            {"requirements.toml": compiled.requirements_toml},
        )
        real = _probe_pod("targets", image=image)
        pods = {
            "real": real,
            "no-scratch": _without_the_scratch_mount(real),
            "no-targets": _without_the_dot_path_mkdir(real),
        }
        for pod in pods.values():
            _kubectl("apply", "-n", _NAMESPACE, "-f", "-", stdin=yaml.safe_dump(pod))
        yield {
            label: _transcript(pod["metadata"]["name"]) for label, pod in pods.items()
        }
    finally:
        _kubectl("delete", "namespace", _NAMESPACE, "--ignore-not-found", check=False)


def _phases(transcript: str) -> tuple[str, str]:
    """Split one pod's transcript into what happened with the control socket absent and
    what happened once the runtime had bound it.

    Every finding in this file belongs to one of those two states and not to the pod as
    a whole, because whether the socket exists decides what the two control-path deny
    rules compile to -- a missing leaf outside every writable root produces no bwrap
    operation, and an existing one needs a mkdir inside the read-only tmpfs its own
    parent's rule just created. A test asserting over a whole transcript would credit a
    finding from one state to the other.

    The marker has to appear exactly once. Zero means the probe was cut short before the
    second half, which would make every socket-bound assertion vacuous; more than once
    means the transcript picked up the script text rather than its output.
    """
    parts = transcript.split(PHASE_BOUND)
    assert len(parts) == 2, (
        f"expected exactly one {PHASE_BOUND!r} marker, found {len(parts) - 1}:\n"
        f"{transcript}"
    )
    return parts[0], parts[1]


@requires_the_cluster
def test_every_pod_ran_its_probe(transcripts: dict[str, str]) -> None:
    """Guard the guard. Every case below reads a transcript, and a pod that never
    started produces an empty one that satisfies both absences by default."""
    for label, transcript in transcripts.items():
        assert "probe=complete" in transcript, f"{label}:\n{transcript}"


@requires_the_cluster
def test_no_transcript_carries_the_manifest_s_own_prose(
    transcripts: dict[str, str],
) -> None:
    """The guard on the two absences below, and not a tidiness check.

    The init container's commentary quotes the bwrap refusal verbatim, so a transcript
    built from the pod spec -- or a probe run under `set -x` -- would satisfy
    `COLLISION in ...` and violate `COLLISION not in ...` for reasons that have
    nothing to do with bwrap. This fails first if that ever becomes true.
    """
    for label, transcript in transcripts.items():
        assert _MANIFEST_PROSE not in transcript, f"{label} carries the manifest text"


def test_the_manifest_really_carries_the_prose_the_guard_above_looks_for() -> None:
    """Not gated on the cluster, because it is the positive control for the guard and a
    guard whose subject is absent reports "clean" and "not looked at" identically.

    Both strings have to be in the manifest for
    `test_no_transcript_carries_the_manifest_s_own_prose` to be doing anything: the
    refusal string is what a spec-carrying transcript would smuggle in, and the prose
    marker is the cheap thing to look for that only the manifest says. If the manifest
    is reworded, this fails and names the marker to update rather than leaving two
    absence assertions quietly unfalsifiable.
    """
    manifest = _MANIFEST.read_text()
    assert COLLISION in manifest, (
        "the manifest no longer quotes the bwrap refusal, so a transcript built from "
        "the pod spec would no longer satisfy it and the guard has nothing to catch"
    )
    assert _MANIFEST_PROSE in manifest, (
        "the marker the guard looks for is not in the manifest any more; pick a "
        "fragment of the current wording that only the manifest says"
    )


@requires_the_cluster
def test_the_helper_has_somewhere_to_write_and_does_not_panic(
    transcripts: dict[str, str],
) -> None:
    assert "tmp-writable=ok" in transcripts["real"]
    assert PANIC not in transcripts["real"]


@requires_the_cluster
def test_removing_the_mount_brings_the_panic_back(
    transcripts: dict[str, str],
) -> None:
    """The control. Without it, "no panic appeared" is a claim a broken pod
    satisfies, and this is also the only case in the file that measures what the
    mount is FOR."""
    assert "tmp-writable=FAIL" in transcripts["no-scratch"]
    assert PANIC in transcripts["no-scratch"]


@requires_the_cluster
def test_both_workspace_deny_targets_are_directories_and_bwrap_accepts_them(
    transcripts: dict[str, str],
) -> None:
    assert "target /session/workspace/.codex=directory" in transcripts["real"]
    assert "target /session/workspace/.agents=directory" in transcripts["real"]
    assert COLLISION not in transcripts["real"]


@requires_the_cluster
def test_removing_the_mkdir_brings_the_collision_back(
    transcripts: dict[str, str],
) -> None:
    """The second control, and the one that separates the two changes: this pod has the
    mount, so a panic here would mean the mount is not what fixed the panic."""
    assert "target /session/workspace/.agents=MISSING" in transcripts["no-targets"]
    assert PANIC not in transcripts["no-targets"]
    assert COLLISION in transcripts["no-targets"]


@requires_the_cluster
def test_with_the_socket_absent_the_pod_as_shipped_runs_a_confined_command(
    transcripts: dict[str, str],
) -> None:
    """What this slice's two changes are worth on their own, measured in the only state
    where nothing else is in the way.

    This is not the state a Session takes a Turn in -- the runtime binds its control
    socket before the pod is ready -- so it is not the finding that closes the slice. It
    is the finding that says the two changes are sufficient for the two refusals they
    were made for, with no third refusal masking the result: no panic, no `.agents`
    collision, and a command whose output arrives.
    """
    absent, _ = _phases(transcripts["real"])
    assert RAN in absent
    assert PANIC not in absent
    assert COLLISION not in absent
    assert CONTROL_PATH not in absent, (
        "the control-path refusal fired with the socket absent, so the mechanism is "
        "not the one this file records and the socket-bound half below is measuring "
        "something else"
    )


@requires_the_cluster
def test_once_the_socket_is_bound_a_confined_command_still_runs(
    transcripts: dict[str, str],
) -> None:
    """The state a Session is actually in, and the one this pod used to fail in.

    While the compiled document named the control socket by name AS WELL as its
    directory, this half refused: `/run/codex` is masked as a read-only tmpfs and the
    leaf rule then needs a mkdir inside it, so bubblewrap built no sandbox and no
    confined command ran. The condition is the part worth carrying: with the socket
    ABSENT the same pod ran the command, so the refusal was never "the control rules
    are unexpressible" but "they are unexpressible once the leaf exists" -- and the
    runtime creates the leaf on every start, which is why it always fired for a real
    Session.

    The leaf rule is gone from the compiled document now, so this half runs the
    command. What makes that a finding about the pod rather than about the compiler is
    the rest of it: the socket really is bound, both of this file's own targets still
    hold, and the two dot-path masks still refuse a read of seeded bytes and a write.
    The mask's own mode is asserted beside them, because `d---------` over a directory
    whose contents are readable is a mask in name only.
    """
    _, bound = _phases(transcripts["real"])
    assert "socket-bound=ok" in bound, (
        "the runtime never bound its control socket, so this half measured the same "
        "state as the half above rather than the one a Session runs in"
    )
    assert BOUND_RAN in bound, (
        "no confined command ran with the socket bound, which is the state every "
        "Session takes a Turn in"
    )
    assert CONTROL_PATH not in bound
    assert PANIC not in bound and COLLISION not in bound, (
        "a refusal this slice was supposed to close came back once the socket existed"
    )
    assert "d---------" in bound
    assert "Permission denied" in bound
    assert SEED not in transcripts["real"], (
        "the confined process read the denied directory"
    )
    assert "bound-confined-write-landed=no" in bound


def test_the_shipped_document_no_longer_names_the_leaf_control_rule() -> None:
    """The one line that used to differ between this file's real pod and its fourth.

    Not gated on the cluster, on purpose: it is a fact about what the compiler emits,
    and a pod is a slow way to read a string. What the cluster is still needed for is
    the consequence, and that is the case above.

    Both lists, because the compiler writes the rule down twice -- a row in the profile
    table and an entry in the managed deny_read -- and every deny_read entry is pushed
    into the same policy the sandbox argv is compiled from. A document that dropped
    only the row would leave the argv unchanged and this half of the slice undone.
    """
    document = _compiled(_A_DIGEST_NO_POD_PULLS).requirements_toml
    assert f'"{config_compiler.CONTROL_SOCKET}" = "deny"' not in document
    assert f'"{config_compiler.CONTROL_SOCKET_DIR}" = "deny"' in document
    deny_read = tomllib.loads(document)["permissions"]["filesystem"]["deny_read"]
    assert config_compiler.CONTROL_SOCKET not in deny_read
    assert config_compiler.CONTROL_SOCKET_DIR in deny_read
