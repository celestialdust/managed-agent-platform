"""The pod manifest and the code that dials it, compared against each other.

Tier 1: this reads YAML and constants, and says nothing about whether a cluster would
schedule the pod. The port, the Service name and the subdomain are one contract written
in three places -- a manifest cannot import a Python constant, so this comparison is the
guard. It exists for a defect this repository has already met: a path derived twice
produced a runtime binding one, a sandbox denying another and a shim connecting to a
third, with every component reporting success on its own terms. The guard written then
covers `config_compiler` and the pod's socket paths; it does not cover the port, the
Service or the readiness route, and this does.

What it cannot say: no image in this tree builds `map-session`, so nothing here shows
that `uvicorn` or `codex` is present in the container these specs launch, and no test in
this file creates a pod.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from managed_agent.control.pod_config.compiler import (
    CODEX_HOME,
    CONTROL_SOCKET,
    WORKSPACE_ROOT,
)
from managed_agent.session_shim.serve import (
    PROBE_PORT,
    READY_ROUTE,
    RUNTIME_WAIT_ATTEMPTS,
    RUNTIME_WAIT_SECONDS,
    SHIM_PORT,
    SHIM_SERVICE,
    WORKSPACE_FILES,
    WORKSPACE_READ_ROOT,
)

_K8S = Path(__file__).resolve().parents[2] / "deploy" / "k8s"
_POD: dict[str, Any] = yaml.safe_load((_K8S / "session-pod.yaml").read_text())
_SERVICE: dict[str, Any] = yaml.safe_load(
    (_K8S / "session-shim-service.yaml").read_text()
)


def _names() -> list[str]:
    return [container["name"] for container in _POD["spec"]["containers"]]


def _container(name: str) -> dict[str, Any]:
    found = [c for c in _POD["spec"]["containers"] if c["name"] == name]
    assert found, f"no container named {name}; the pod has {_names()}"
    first: dict[str, Any] = found[0]
    return first


def test_both_manifests_parsed_and_have_something_to_examine() -> None:
    """The positive half. Every assertion below reads a discovered collection, and a
    manifest that failed to parse into one would satisfy several of them by being
    empty."""
    assert len(_names()) >= 2
    assert _SERVICE["kind"] == "Service"


def test_the_pod_runs_a_shim_beside_the_runtime() -> None:
    assert _names() == ["agent-runtime", "session-shim"]


def test_the_port_the_manifest_publishes_is_the_port_the_control_plane_dials() -> None:
    """Both ports, and both against the constants the code binds them from.

    The second one exists for the kubelet's readiness probe: once a pod holds a
    certificate the Session port requires one back, and a probe presents none. Declared
    in the manifest so the probe can name it rather than repeat a number. It is bound on
    every interface, because a probe dials the pod IP -- what keeps it off the pod
    network is the NetworkPolicy, which admits nothing on this port.
    """
    ports = _container("session-shim")["ports"]
    assert [port["containerPort"] for port in ports] == [SHIM_PORT, PROBE_PORT]


def test_the_service_name_the_pod_subdomain_and_the_constant_all_agree() -> None:
    assert _POD["spec"]["subdomain"] == SHIM_SERVICE
    assert _SERVICE["metadata"]["name"] == SHIM_SERVICE


def test_the_pod_declares_a_hostname_and_not_only_a_subdomain() -> None:
    """Half the contract is `spec.hostname`, and the subdomain alone buys nothing.

    A headless Service's per-pod A record is `<hostname>.<subdomain>.<ns>.svc...`, and
    the endpoints controller fills in an address's `hostname` only when
    `pod.spec.hostname` is set. Until 2026-08-23 this manifest set the subdomain and no
    hostname, so no Session pod had a record of its own and every Turn failed at the
    shim hop while the pod itself read 2/2 Running.

    The value here is the manifest's placeholder; `pod_runner.py` substitutes the pod's
    name into it, and `tests/adapters/test_pod_runner.py` grades that. What this asserts
    is that the field exists to be substituted -- deleting it would leave that
    substitution writing a key the manifest never declared, which no reader of this file
    would catch.
    """
    assert _POD["spec"]["hostname"] == _POD["metadata"]["name"]


def test_the_service_is_headless_so_no_turn_can_land_on_another_session() -> None:
    """A load-balanced Service over every Session pod would send one Session's Turn to
    another Session's shim.

    `clusterIP: None` is the literal string Kubernetes wants, and YAML parses it as the
    string `'None'` rather than as a null -- so the assertion is an equality against
    that string. Written as `is None` it fails on a correct manifest.
    """
    assert _SERVICE["spec"]["clusterIP"] == "None"
    assert _SERVICE["spec"]["publishNotReadyAddresses"] is False
    assert _SERVICE["spec"]["selector"] == _POD["metadata"]["labels"]


def test_the_service_forwards_to_the_port_the_shim_container_names() -> None:
    published = _SERVICE["spec"]["ports"][0]
    named = _container("session-shim")["ports"][0]["name"]
    assert published["targetPort"] == named
    assert published["port"] == SHIM_PORT


def test_the_token_is_mounted_into_the_shim_and_into_nothing_else() -> None:
    """A token the runtime container could read is a token the confined agent can read,
    and the whole of this token's blast-radius argument is that it cannot."""
    mounted = {
        container["name"]
        for container in _POD["spec"]["containers"] + _POD["spec"]["initContainers"]
        if any(m["name"] == "shim-token" for m in container.get("volumeMounts", ()))
    }
    assert mounted == {"session-shim"}


def test_the_token_mount_is_outside_the_runtime_s_configuration_tree() -> None:
    """A confined process reading its own configuration tree must not find it."""
    codex_home = [
        variable["value"]
        for variable in _container("agent-runtime")["env"]
        if variable["name"] == "CODEX_HOME"
    ]
    assert codex_home, "the runtime container names no CODEX_HOME to be outside of"
    token = [
        mount
        for mount in _container("session-shim")["volumeMounts"]
        if mount["name"] == "shim-token"
    ][0]
    assert not token["mountPath"].startswith(codex_home[0])
    assert token["readOnly"] is True


def test_the_readiness_probe_points_at_the_route_that_reports_readiness() -> None:
    """A probe aimed at the POST-only Turn route answers 405, which kubelet reads as
    not-ready forever -- the pod would never enter DNS and every Turn would be
    undeliverable while the shim worked."""
    probe = _container("session-shim")["readinessProbe"]["httpGet"]
    assert probe["path"] == READY_ROUTE
    published = {
        port["name"]: port["containerPort"]
        for port in _container("session-shim")["ports"]
    }
    # The probe port, not the Session port. Asserted through the published mapping
    # rather than by index, because "the first port" stopped being a fact about this
    # manifest the moment there were two.
    assert published[probe["port"]] == PROBE_PORT
    # No `host`, so the kubelet dials the pod IP. Setting one here would send the probe
    # to that address from the NODE's namespace -- `host: 127.0.0.1` would probe the
    # node itself -- and the shim would never see it. The default is the only value
    # that reaches this container, so its absence is the assertion.
    assert "host" not in probe, (
        f"the probe names a host ({probe.get('host')!r}); a kubelet resolves it in the "
        "node's network namespace, not this pod's"
    )


def test_the_shim_command_names_a_module_that_runs_when_it_is_run() -> None:
    """`python -m <module>` resolves this name at container start, so a typo in it is a
    container that fails to start against a manifest nothing else reads.

    It used to be `uvicorn <module>:<attribute> --factory`, and the module took over
    because whether this pod serves TLS depends on what was mounted into it -- a
    decision a manifest cannot express, since naming `--ssl-certfile` unconditionally
    fails every pod placed without CA material.

    What is checked is that the module imports and that `python -m` will find something
    to run in it, which is `__main__` at module scope. `serve` is looked up but never
    called: calling it binds a port.
    """
    command = _container("session-shim")["command"]
    assert command[:2] == ["python", "-m"], command
    module = importlib.import_module(command[2])
    assert callable(module.serve)
    assert importlib.util.find_spec(command[2]) is not None, (
        f"{command[2]} is not importable as a module"
    )


def test_the_runtime_container_still_launches_the_argv_config_compiler_emits() -> None:
    """The regression-surface guard: this slice adds a container and moves nothing.

    The socket path comes from `config_compiler.CONTROL_SOCKET` rather than from a
    second literal here, because the invariant is that the manifest launches the argv
    the compiler emits. Two literals can be changed together and wrongly while this
    test still passes, which is what it exists to prevent.
    """
    assert _container("agent-runtime")["command"] == [
        "codex",
        "app-server",
        "--listen",
        f"unix://{CONTROL_SOCKET}",
    ]


def test_the_shim_container_drops_what_every_container_in_this_pod_drops() -> None:
    """Container-level and not inherited from the pod's securityContext, so a container
    added without this block runs with the default bounding set while its siblings run
    with none."""
    security = _container("session-shim")["securityContext"]
    assert security["allowPrivilegeEscalation"] is False
    assert security["capabilities"]["drop"] == ["ALL"]
    assert security["readOnlyRootFilesystem"] is True


def test_the_shim_can_read_the_record_the_runtime_writes_and_cannot_write_it() -> None:
    """The control plane reads a Session's Rollout out through this container, so this
    container has to be able to see the tree the runtime writes it into.

    Without the mount the shim-side glob over `$CODEX_HOME/sessions/...` returns empty
    **with no exception**: every ship-out would upload nothing, recovery would find no
    rollout, and nothing anywhere would say so. That is the failure this asserts
    against, and it is why the path comes from the compiler's constant rather than a
    literal here -- the glob and the mount have to name the same directory.

    Read-only, because ship-out reads. A writable mount would give the pod's one
    outward-facing process write access to the record recovery depends on.
    """
    same_volume_as_the_runtime = "codex-home"
    runtime = [
        mount
        for mount in _container("agent-runtime")["volumeMounts"]
        if mount["name"] == same_volume_as_the_runtime
    ]
    assert runtime, "the runtime container writes its record somewhere else now"
    assert runtime[0]["mountPath"] == CODEX_HOME

    shim = [
        mount
        for mount in _container("session-shim")["volumeMounts"]
        if mount["name"] == same_volume_as_the_runtime
    ]
    assert shim, (
        "the shim container does not mount the runtime's home, so find_rollout globs a "
        "path that does not exist and every ship-out silently uploads nothing"
    )
    assert shim[0]["mountPath"] == CODEX_HOME
    assert shim[0]["readOnly"] is True


def test_the_shim_reaches_the_attachment_directory_and_nothing_else_beside_it() -> None:
    """The file route writes here, and `subPath` is what stops it writing anywhere else.

    Three things have to line up and each fails silently on its own. The mount has to
    exist, or the route answers 500 for every attached file. Its path has to be the
    constant the route writes to, or the file lands in the pod at a path the agent has
    no route to -- the bytes are on the disk and the agent reports the document missing.
    And it has to narrow to the attachment directory, because a mount that stopped at
    this Session's own root would let the pod's one outward-facing process write over
    anything the agent produced, from off the pod, through a route whose only check is a
    bearer token.

    Asserted as a SUFFIX rather than as the whole string, because the rest of the
    subPath is the Session's subtree of a volume shared by every Session on the cluster
    (ADR-035) and is filled in per Session by the pod runner. The property here is the
    last segment -- how far past its own root this container reaches -- and pinning the
    whole literal would make this test fail on a correct manifest the day the subtree
    layout changes. The subtree itself is graded by `tests/adapters/test_pod_runner.py`,
    which is where the substitution lives.

    The runtime container is asserted to mount the same volume at its own root, the one
    the compiler makes writable, because that is the half that makes the file reachable.
    Two containers, one volume, two mounts of different shapes -- and the file path only
    works out if both are as written.
    """
    volume = "workspace"
    # Selected by mountPath rather than by volume name, because this container mounts
    # this volume twice now: read-write and narrowed here, and read-only over the whole
    # root for the ship-out (the case below). Indexing the name-filtered list would
    # grade whichever of the two the manifest happens to list first.
    shim = [
        mount
        for mount in _container("session-shim")["volumeMounts"]
        if mount["name"] == volume and mount["mountPath"] == str(WORKSPACE_FILES)
    ]
    assert shim, (
        "the shim container does not mount the workspace at the attachment directory, "
        "so the file route writes into its own container filesystem and the agent "
        "never sees the document"
    )
    assert str(shim[0].get("subPath", "")).endswith(f"/{WORKSPACE_FILES.name}"), (
        "this container's attachment mount does not end at the attachment directory, "
        "so it reaches further up its Session's tree read-write and a caller holding "
        "the shim token could overwrite anything the agent wrote"
    )
    assert shim[0].get("readOnly") is not True, (
        "the attachment mount is read-only, so every attached file fails to be placed "
        "and every Session with one refuses to start"
    )

    runtime = [
        mount
        for mount in _container("agent-runtime")["volumeMounts"]
        if mount["name"] == volume
    ]
    assert runtime, "the runtime container no longer mounts the workspace"
    assert runtime[0]["mountPath"] == WORKSPACE_ROOT
    # Both containers carry a subPath now, because the volume is one claim shared by
    # every Session (ADR-035) and the subtree is what separates two tenants. So the
    # question this asks is no longer "does the runtime mount a subtree" -- it must --
    # but whether the two subtrees line up: the agent sees the Session's root, the shim
    # writes one directory inside it, and the file the shim places is therefore at
    # WORKSPACE_FILES for the agent. Compared against the shim's own mount rather than
    # against a literal, so the two cannot be edited apart.
    assert f"{runtime[0]['subPath']}/{WORKSPACE_FILES.name}" == shim[0]["subPath"], (
        "the runtime and the shim are not mounting one Session's tree one directory "
        f"apart, so a file the shim places is not at {WORKSPACE_FILES} for the agent"
    )


def test_the_shim_can_read_what_the_agent_produced_and_still_cannot_write_it() -> None:
    """The second workspace mount, and the whole of why it does not undo the first.

    Shipping a produced file out needs this container to SEE this Session's workspace
    root, which the attachment mount above does not reach. Without a second mount the
    listing route reads a directory that does not exist, answers `{"files": []}` with
    **no exception**, and every completed Turn ships nothing while nothing anywhere says
    so -- which is the failure this asserts against, and the same shape as the
    codex-home mount's above.

    "The workspace root" now means this SESSION's root and not the volume's. The volume
    is one claim shared by every Session on the cluster (ADR-035), so a mount here with
    no subPath at all would map every tenant's workspace into the one container in this
    pod reachable from outside it -- and `readOnly` would not save it, because this
    route's whole job is to read bytes out and hand them to whoever holds the token.
    That is why the widening is asserted as a RELATIONSHIP between the two mounts below
    rather than as the absence of a subPath: exactly one directory of narrowing
    separates them, which is what "reads the Session root, writes only the attachment
    directory" means once both live under a per-Session subtree.

    What makes the widening safe is that it is read-only and at its own path, and both
    halves are asserted. `readOnly: true` is the security control: the reason the write
    mount is narrowed is that a caller holding the shim token must not be able to write
    over anything the agent produced, and a second read-write mount of the Session's
    whole tree would hand that back while leaving the narrowed mount in place and
    passing the test above it.

    The distinct `mountPath` is the correctness control. Mounted at the workspace root
    this would be the *parent* of the read-write mount, so which of the two governed the
    attachment directory would rest on kubelet ordering nested mounts by path depth --
    true today, and a read-only attachment directory, meaning every Session's uploads
    failing to be placed, on the day it is not.
    """
    volume = "workspace"
    mounts = [
        mount
        for mount in _container("session-shim")["volumeMounts"]
        if mount["name"] == volume
    ]
    read_root = [m for m in mounts if m["mountPath"] == str(WORKSPACE_READ_ROOT)]
    assert read_root, (
        "the shim container does not mount the workspace root, so the outputs listing "
        "reads a directory that is not there, answers with no files and no error, and "
        "every completed Turn ships nothing the agent wrote"
    )
    assert read_root[0].get("readOnly") is True, (
        "the read of the agent's output is mounted read-write, which gives the pod's "
        "one outward-facing process write access to everything the agent produced -- "
        "the exact thing the subPath on the other mount exists to prevent"
    )
    writable = [m for m in mounts if m["mountPath"] == str(WORKSPACE_FILES)]
    assert writable, "the attachment mount is gone, so no attached file can be placed"

    assert read_root[0].get("subPath"), (
        "the mount that reads the agent's output carries no subPath, so it maps the "
        "whole shared volume -- every tenant's workspace -- into the one container in "
        "this pod that is reachable from off it"
    )
    assert (
        f"{read_root[0]['subPath']}/{WORKSPACE_FILES.name}" == writable[0]["subPath"]
    ), (
        "the read mount and the write mount are not one directory apart, so either the "
        "read is narrowed below this Session's root -- and a file written at that root "
        "is never shipped -- or the write reaches above the attachment directory"
    )
    assert (
        Path(str(read_root[0]["mountPath"]))
        not in Path(str(writable[0]["mountPath"])).parents
    ), (
        "the read-only mount is a parent of the read-write one, so which governs the "
        "attachment directory depends on the order kubelet applies nested mounts"
    )


def test_the_shim_waits_for_the_socket_as_long_as_the_manifest_budgets() -> None:
    """The shim's retry budget and the runtime's startupProbe wait for one event.

    That event is the control socket appearing, and this manifest is where the platform
    says how long it may take. A shim that gave up sooner would exit; uvicorn treats a
    raising lifespan as fatal and this pod's `restartPolicy` is `Never`, so that pod
    would never run a Turn again while the runtime came up perfectly beside it.
    """
    probe = _container("agent-runtime")["startupProbe"]
    budgeted = probe["periodSeconds"] * probe["failureThreshold"]
    waited = RUNTIME_WAIT_ATTEMPTS * RUNTIME_WAIT_SECONDS

    assert budgeted > 0, "the runtime's startupProbe budgets nothing to compare against"
    assert waited >= budgeted, (
        f"the shim waits {waited}s for the socket while this manifest budgets "
        f"{budgeted}s for it to appear, so the shim gives up on a runtime that is "
        "still inside its own start-up window"
    )


def test_the_runtime_log_filter_names_targets_and_sets_no_broad_global_level() -> None:
    """What `RUST_LOG` is set TO, because the risk is in the value and not the key.

    This variable exists so the runtime's own diagnosis of a skill it could not load
    reaches the container log: a Session had a skill delivered into `/etc/codex/skills`,
    readable there, and absent from the model's catalogue, and the record naming the
    reason was filtered out before anything could read it.

    The value is the part that can do harm. `tracing`'s filter syntax makes the first
    bare level the GLOBAL one, so `info` alone -- or `debug`, or `trace` -- turns on
    every span in the process, and this is the container the model runs in: prompts,
    answers and tool arguments would land in node logs, readable by anyone who can read
    logs on that node and retained on the node's clock rather than the tenant's. That is
    the disclosure ADR-013 exists to prevent, and it would arrive as an operator
    widening a filter to debug something unrelated.

    So two things are asserted. The global level is `error`, which keeps a genuine
    failure visible and nothing else; and at least one per-target directive is present,
    because a filter that is only `error` turns the diagnostic off and would leave this
    variable set to no purpose while reading as configured.
    """
    filters = [
        entry["value"]
        for entry in _container("agent-runtime")["env"]
        if entry["name"] == "RUST_LOG"
    ]
    assert len(filters) == 1, filters
    directives = [one.strip() for one in filters[0].split(",") if one.strip()]

    globals_ = [one for one in directives if "=" not in one]
    assert globals_ == ["error"], (
        f"the global level in RUST_LOG is {globals_}. Anything above `error` turns on "
        "every span in the container the model runs in, which puts prompt and answer "
        "text into node logs"
    )
    targeted = [one for one in directives if "=" in one]
    assert targeted, (
        "RUST_LOG carries no per-target directive, so it raises nothing above `error` "
        "and the skills diagnosis it exists for is still filtered out"
    )
