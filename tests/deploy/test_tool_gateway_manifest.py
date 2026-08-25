"""The Tool Gateway's manifest, graded on the things a stdio registration makes matter.

Tier 1 (local, no infrastructure). This container runs commands a tenant registered, so
its container-level hardening is not boilerplate copied from a template — it is the
blast radius of every stdio server anybody ever registers, and each assertion below is
about a way that radius could widen.

NOT PROVEN by anything here: nothing validates these documents against the real
Kubernetes schemas. `kubeconform` is not installed and is not an attested tool for this
repo, so a tactic naming it would be a check that silently never runs; a field name
misspelled in a way that still parses as YAML reaches `kubectl apply` uncaught.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast
from unittest import mock

import pytest
import yaml
from starlette.routing import Route

from managed_agent import composition
from managed_agent.adapters.s3.session_vfs import UnconfiguredSessionVfs
from managed_agent.core.ids import SessionId
from managed_agent.gateway.tool.mcp_proxy import ToolEventTypes
from managed_agent.gateway.tool.server import (
    MCP_PATH,
    GatewaySessions,
    SessionTokenMiddleware,
    create_gateway_app,
)

_MANIFEST = Path(__file__).resolve().parents[2] / "deploy" / "k8s" / "tool-gateway.yaml"


def _no_sessions() -> GatewaySessions:
    """A registry with nothing behind it; only the route table is being read."""
    return GatewaySessions(
        scopes=cast(Any, None),
        registry=cast(Any, None),
        broker=cast(Any, None),
        append=cast(Any, None),
        events=cast(Any, None),
        types_=ToolEventTypes(
            progress="p", elicitation_requested="q", elicitation_answered="r"
        ),
        evidence=cast(Any, None),
    )


def _documents() -> list[dict[str, Any]]:
    loaded = list(yaml.safe_load_all(_MANIFEST.read_text()))
    assert loaded, f"{_MANIFEST} parsed into no documents at all"
    return loaded


def _container() -> dict[str, Any]:
    deployment = next(d for d in _documents() if d["kind"] == "Deployment")
    containers = deployment["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1, "one container, or the assertions below grade one of N"
    only = containers[0]
    assert isinstance(only, dict)
    return only


def test_the_file_holds_exactly_a_deployment_and_a_service() -> None:
    assert [d["kind"] for d in _documents()] == ["Deployment", "Service"]


def test_the_pod_runs_as_a_non_root_user() -> None:
    deployment = next(d for d in _documents() if d["kind"] == "Deployment")
    security = deployment["spec"]["template"]["spec"]["securityContext"]

    assert security["runAsNonRoot"] is True
    assert security["runAsUser"] != 0


def test_the_container_cannot_escalate_write_its_root_or_keep_a_capability() -> None:
    """Every line here is about the command a tenant's registration names."""
    security = _container()["securityContext"]

    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"]["drop"] == ["ALL"]


def test_the_pod_takes_none_of_the_host() -> None:
    deployment = next(d for d in _documents() if d["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]

    assert "hostNetwork" not in pod
    assert "hostPID" not in pod
    assert "privileged" not in _container()["securityContext"]


ALLOWED_LITERAL_VARIABLES: Final = frozenset(
    {"MAP_OBJECT_BUCKET", "MAP_ROLLOUT_BUCKET"}
)
"""The environment variables in this manifest that may carry a `value:`.

Two, and both are names rather than secrets: the bucket captured Evidence is written to,
and the bucket a resuming Session's Rollout is read back from. What makes reaching
either possible is this pod's role, not these strings, and an operator has to be able to
read which bucket their Evidence went to, and which one a resume reads, out of the file
they applied.

An allowlist rather than a relaxed rule, so a further literal is a decision somebody
takes here with the case in front of them. `test_every_allowed_literal_is_still_in_the
_manifest` below is what stops this set outliving the entries it was written for.
"""


def test_no_secret_value_can_be_written_into_this_file() -> None:
    """Every environment entry is a reference; a literal `value` fails here.

    That is the check, not a style preference: a manifest is committed and read widely,
    and the difference between a name and a value is the whole of whether this file can
    leak a credential. The one exception is named above and nowhere else.
    """
    env = _container()["env"]

    assert env, "no environment entries found, so this assertion grades nothing"
    for entry in env:
        if entry["name"] in ALLOWED_LITERAL_VARIABLES:
            continue
        assert "value" not in entry, f"{entry['name']} carries a literal value"
        assert "secretKeyRef" in entry["valueFrom"], entry["name"]


def test_every_allowed_literal_is_still_in_the_manifest() -> None:
    """The exemption above cannot outlive the entry it was written for.

    Without this, removing `MAP_OBJECT_BUCKET` from the manifest would leave a standing
    permission to write a literal under that name, and the next person to add it back --
    as a credential, say -- would find the guard already stood down for them.
    """
    named = {entry["name"] for entry in _container()["env"]}
    assert named >= ALLOWED_LITERAL_VARIABLES, (
        f"{sorted(ALLOWED_LITERAL_VARIABLES - named)} is exempted from the literal "
        "check and appears in no environment entry"
    )


def test_both_probes_name_a_path_the_app_actually_serves_without_a_token() -> None:
    """A probe holds no Session token, so it can only reach an unguarded path.

    Read off the app rather than written down twice: a manifest naming a path the code
    stopped serving is a pod that never becomes ready, and a constant repeated in two
    files is free to disagree with itself.

    Filtered on the token middleware and not on the MCP path, because the app now
    serves more than one guarded route. Naming the one path to exclude would have made
    every route added later count as probe-reachable by default, which is the wrong way
    round for a set whose whole meaning is "reachable holding nothing".
    """
    served = {
        route.path
        for route in create_gateway_app(
            _no_sessions(), b"unused", UnconfiguredSessionVfs(), _NoRollouts()
        ).routes
        if isinstance(route, Route)
        and not isinstance(route.endpoint, SessionTokenMiddleware)
    }
    container = _container()

    assert served, "the app serves nothing outside the MCP path, so nothing can probe"
    for probe in ("readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"]["path"] in served
        assert container[probe]["httpGet"]["port"] == "mcp"


def test_the_service_targets_the_port_name_the_container_declares() -> None:
    service = next(d for d in _documents() if d["kind"] == "Service")
    declared = {port["name"] for port in _container()["ports"]}

    for port in service["spec"]["ports"]:
        assert port["targetPort"] in declared


def test_the_service_selects_the_pods_this_deployment_labels() -> None:
    deployment = next(d for d in _documents() if d["kind"] == "Deployment")
    service = next(d for d in _documents() if d["kind"] == "Service")

    assert service["spec"]["selector"] == deployment["spec"]["selector"]["matchLabels"]


def test_the_args_name_a_factory_that_exists_and_this_test_imports_it() -> None:
    """This case used to assert the string and decline to import the name, because the
    factory belonged to a later slice. It exists now, so the weaker assertion is gone: a
    manifest naming a callable nothing exports is a Deployment that CrashLoops with
    `Error loading ASGI app`, and a string comparison passes either way.
    """
    args = _container()["args"]
    named = "managed_agent.composition:tool_gateway_app"

    assert named in args
    assert "--factory" in args
    module, _, attribute = named.partition(":")
    imported = importlib.import_module(module)
    assert callable(getattr(imported, attribute))


def test_the_manifests_own_env_block_drives_the_factory() -> None:
    """Every variable the manifest declares, set from the manifest, and the factory
    built from exactly those. The mismatch this catches is the one the secretKeyRef
    case above cannot: MAP_SESSION_TOKEN_KEY was declared here and read by nothing
    anywhere in the tree, and a rule that only checks whether an entry is a reference
    is satisfied by a reference to a variable no process ever looks up.

    `clear=True` on every patch is what makes it "and no others": a variable the factory
    reads but the manifest does not declare is absent from this environment, so the pod
    could not have been given it either.
    """
    declared = {entry["name"] for entry in _container()["env"]}
    assert declared, "no environment entries found, so this assertion grades nothing"
    supplied = {
        name: "postgresql+asyncpg://u:p@localhost:5432/none"
        if "DATABASE" in name
        else "k"
        for name in declared
    }

    with mock.patch.dict(os.environ, supplied, clear=True):
        built = composition.tool_gateway_app()

    served = {route.path for route in built.routes if isinstance(route, Route)}
    assert MCP_PATH in served

    for name in declared:
        partial = {k: v for k, v in supplied.items() if k != name}
        with (
            mock.patch.dict(os.environ, partial, clear=True),
            pytest.raises(KeyError, match=name),
        ):
            composition.tool_gateway_app()


def test_the_image_is_the_digest_placeholder_and_not_a_tag() -> None:
    """A tag is a name for whatever was pushed last; a digest is a name for bytes.

    `PLACEHOLDER_ECR/map-tool-gateway:latest` was wrong three ways at once: the
    repository is `map/tool-gateway` with a slash, `:latest` is the one string
    core/registration/environment.py refuses, and an IMMUTABLE repository accepts that
    tag once.
    """
    image = _container()["image"]

    assert image == "map/tool-gateway@sha256:" + "0" * 64, image
    assert ":latest" not in image
    assert "PLACEHOLDER" not in image


def _ceiling_owner() -> ModuleType:
    """`tests/deploy/test_control_plane_manifest.py`, loaded as a module.

    By path rather than by `import`, matching how this repository loads the modules
    under `deploy/`: these two files sit in a directory with no `__init__.py`, so the
    name a plain import would need depends on how the runner assembled `sys.path`.
    """
    path = Path(__file__).resolve().parent / "test_control_plane_manifest.py"
    spec = importlib.util.spec_from_file_location("map_connection_ceiling_owner", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_this_files_surge_is_pinned_and_the_cross_manifest_sum_is_owned_elsewhere() -> (
    None
):
    """maxSurge here, and the whole sum in the one place that computes it.

    THIS CASE REPLACES A STALE DUPLICATE, and the duplicate is worth describing because
    the shape of its failure is the point. It asserted
    `(this_replicas + control_plane_replicas) * per_process <= 225`, and both halves
    had gone wrong. 225 came from the parameter group's formula at db.t4g.small;
    `test_control_plane_manifest.py` records that figure as measured 47 too high, and
    the instance answers `show max_connections` = 400 with three reserved for
    superusers. And the sum itself only ever counted two Deployments, so the migration
    Job's connection was outside it and a third database-using workload would have been
    too. The result was a check that read as a ceiling guard while its number was wrong
    in the permissive direction on one side and the strict direction on the other -- and
    it refused every control-plane replica count above 1, which is a real change it had
    no measurement behind.

    `tool-gateway.yaml:30-32` already said the sum was not maintained here. Its own test
    contradicted it, which is how this survived: a comment claiming a mechanism lives
    elsewhere is not evidence that it does.

    So this asserts the boundary rather than describing it. The owner is loaded and
    asked, and THIS manifest must appear in the set that owner discovers, with a peak at
    least this Deployment's replica count. A discovery that stopped finding this file --
    a renamed Secret key, a changed `kind`, a glob that stopped matching -- fails here
    instead of quietly leaving this workload out of the platform's connection budget.
    """
    from managed_agent import composition as wiring

    spec = next(d for d in _documents() if d["kind"] == "Deployment")["spec"]
    assert spec["strategy"]["rollingUpdate"]["maxSurge"] == 0

    owner = _ceiling_owner()
    workloads = owner._database_workloads()
    mine = f"{_MANIFEST.name}:tool-gateway"
    assert mine in workloads, (
        f"{owner.__file__} computes the platform's connection budget and did not find "
        f"{mine} in it, so this Deployment's {spec['replicas']} pools are outside the "
        f"only sum anybody adds up. Found: {sorted(workloads)}"
    )
    assert workloads[mine] >= int(spec["replicas"]), (
        f"{mine} contributes {workloads[mine]} peak pods against {spec['replicas']} "
        "replicas, so the sum understates this workload"
    )
    assert wiring._POOL_SIZE + wiring._MAX_OVERFLOW > 0, (
        "the per-process pool is what the owner multiplies those peak pods by; at zero "
        "its inequality holds for any replica count"
    )


def test_the_stdio_scratch_mount_exists_because_the_root_filesystem_is_read_only() -> (
    None
):
    """A spawned child needs somewhere to write, and the root is denied to it."""
    deployment = next(d for d in _documents() if d["kind"] == "Deployment")
    volumes = {v["name"]: v for v in deployment["spec"]["template"]["spec"]["volumes"]}
    mounts = {m["name"]: m for m in _container()["volumeMounts"]}

    assert mounts["scratch"]["mountPath"] == "/tmp"
    assert "emptyDir" in volumes["scratch"]


@pytest.mark.parametrize("kind", ["Deployment", "Service"])
def test_every_document_names_the_same_component(kind: str) -> None:
    document = next(d for d in _documents() if d["kind"] == kind)

    assert document["metadata"]["name"] == "tool-gateway"


class _NoRollouts:
    """A rollout store holding nothing, for cases that are not about resuming.

    Answers None rather than raising, because that is the honest answer for the
    Sessions these cases drive: none of them has completed a Turn, so none has a
    stored Rollout. A raising stand-in would make every case here assert something
    about a store it is not testing.
    """

    async def restore_for_resume(self, session_id: SessionId) -> None:
        return None
