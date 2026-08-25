"""What this adapter turns the repository's manifest into, and what it makes of a pod.

Three tiers, and what each can and cannot show.

**Tier 1 is not a mocked cluster.** No `CoreV1Api` is faked or patched: the manifest is
the file under `deploy/k8s/`, and the pods are real `kubernetes_asyncio` model objects
built with the fields the cluster was measured returning. The four status fields the
adapter branches on are the whole of `_phase_of` and `_why_it_will_not_start`, and this
is where they are graded.

**Tier 2 is the real client against a fake API server** (`fake_kubernetes_api`), and it
exists because tier 1 cannot reach `ensure` or `remove` at all -- the two methods that
create things and delete them. A real credential is loaded from a real kubeconfig, a
real HTTP request goes out, and what comes back is deserialized by the generated client;
only the cluster's behaviour is canned. What it grades is the orchestration: whether a
create happens, whether a squatted name is adopted, and what is left behind when a
placement fails. Asserted on the store's contents rather than on which calls were made,
so a cleanup that forgets one Secret fails here.

**Tier 3 is `map-dev`,** at the bottom, skipped unless `MAP_CLUSTER_TESTS=1`. It creates
a pod in EKS, so it is slow, needs credentials, and must not run because somebody typed
`pytest`. It is the only tier that can show the API server admits these bodies and a
kubelet reaches these states.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import kubernetes_asyncio.config.kube_config as kube_config
import pytest
import yaml
from fake_kubernetes_api import (
    FakeCluster,
    fake_kubernetes_api,
    rfc3339,
    running,
    stuck,
)
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import (
    V1ContainerState,
    V1ContainerStateTerminated,
    V1ContainerStateWaiting,
    V1ContainerStatus,
    V1Namespace,
    V1ObjectMeta,
    V1Pod,
    V1PodCondition,
    V1PodSpec,
    V1PodStatus,
)

import managed_agent.adapters.kubernetes.pod_runner as pod_runner_module
from managed_agent.adapters.kubernetes.pod_runner import (
    _ALREADY_EXISTS,
    _CONTAINER_MAY_RESOLVE,
    _POLL_SECONDS,
    _REASON_CAP,
    _SECRET_FILES,
    _SEED_ENV,
    _SHIM_CONTAINER,
    _SHIM_ENV,
    _SKILL_VOLUME,
    _WILL_NOT_START,
    KubernetesPodRunner,
    _claims_this_session,
    _core_api,
    _is_scheduled,
    _phase_of,
    _pod_for,
    _secret_name,
    _secret_volumes,
    _secrets_for,
    _skill_secret_key,
    _skill_volume_items,
    _why_a_container_has_not_been_created,
    _why_it_is_not_scheduled,
    _why_it_will_not_start,
)
from managed_agent.composition import pod_runner_from_environment
from managed_agent.control.pod_config.compiler import (
    CompiledConfig,
    compile_session_config,
)
from managed_agent.control.session.placement import (
    Placement,
    PodNotStarted,
    PodPhase,
    pod_name_for,
)
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.registration.definition import AgentDefinition, SkillsRevision
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.registration.skill import SkillFile, ValidatedSkill, skill_files
from managed_agent.core.session.session import SessionRecord
from managed_agent.session_shim.pod_channel import shim_token_for, shim_url_for

MANIFEST = Path(__file__).resolve().parents[2] / "deploy" / "k8s" / "session-pod.yaml"
A_KEY = b"a signing key for tests only"

# Where a Session pod reaches the Model Gateway. The `/v1` is load-bearing at both ends:
# the Agent Runtime POSTs `{base_url}/responses`, and the Gateway's router mounts
# `POST /v1/responses`.
#
# The name the Service will have, not a placeholder, because the live tier at the bottom
# mounts this into a real pod and the runtime resolves the provider against it. The
# Gateway is not deployed, so nothing here shows the address answers -- what it shows is
# that the configuration loads and the Session gets as far as the model call.
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"

# The Gateway's signing key and the token's deadline, which the compiler takes from
# its caller and never defaults. Literals, so no case here can expire mid-run.
SESSION_TOKEN_KEY = b"a signing key that is thirty-two"
SESSION_TOKEN_EXPIRY = 4102444800
_A_DIGEST = "sha256:" + "ab" * 32
_AN_IMAGE = f"registry.invalid/map/session-shim@{_A_DIGEST}"

# The definition a Session pins, for the one field the compiler reads off it: the model.
A_DEFINITION = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)


def _manifest() -> dict[str, Any]:
    """The manifest as the adapter will see it.

    `Any`-valued rather than `object`-valued so the nested reads below type-check at
    all: under `dict[str, object]`, `raw["spec"]["containers"]` is an index into
    `object`.
    """
    parsed = yaml.safe_load(MANIFEST.read_text())
    assert isinstance(parsed, dict)
    return parsed


def _compiled(image: str = _AN_IMAGE) -> CompiledConfig:
    return compile_session_config(
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
        tool_gateway_url="http://tool-gateway.invalid",
        model_gateway_url=MODEL_GATEWAY_URL,
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="t1",
            runtime_image=image,
            denied_paths=(),
        ),
        definition=A_DEFINITION,
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )


_TWO_SKILLS = skill_files(
    (
        ValidatedSkill(
            name="pdf",
            description="Build a PDF report.",
            text="---\nname: pdf\ndescription: Build a PDF report.\n---\nSteps.\n",
        ),
        ValidatedSkill(
            name="deep-wiki",
            description="Read a repository's docs.",
            text="---\nname: deep-wiki\ndescription: Read docs.\n---\nSteps.\n",
        ),
    )
)
"""Two, and one of them with a hyphen in its name.

One skill would not catch a delivery that projects whichever key it happened to build
last, and a name with a separator in it is what proves the Secret key is derived rather
than assumed to be a bare word -- a key is not allowed to hold a `/`, and a hyphen is
the character a reader is most likely to think needs escaping and does not.
"""


def _compiled_with_skills(
    files: tuple[SkillFile, ...] = _TWO_SKILLS,
) -> CompiledConfig:
    """The same Session, with skills attached.

    Built through the compiler rather than by replacing a field, so what these cases
    grade is the value the placement path really hands the runner.

    The file set is a parameter, defaulting to the two whole skills above, because the
    cases below have to hand this a set no `skill_files()` call would build -- a pair
    whose keys collide, a path holding a character a Secret key cannot -- and those are
    reached by constructing `SkillFile` directly rather than through a skill.
    """
    return compile_session_config(
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
        tool_gateway_url="http://tool-gateway.invalid",
        model_gateway_url=MODEL_GATEWAY_URL,
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="t1",
            runtime_image=_AN_IMAGE,
            denied_paths=(),
        ),
        definition=A_DEFINITION,
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
        skill_files=files,
    )


def _a_pod(
    *,
    phase: str = "Running",
    ready: tuple[bool, ...] = (True, True),
    statuses_reported: bool = True,
    deleting: bool = False,
    labels: dict[str, str] | None = None,
    waiting: str | None = None,
    waiting_detail: str = "m",
    init_waiting: str | None = None,
    terminated: tuple[str, int] | None = None,
    restart_policy: str = "Never",
    unschedulable: str | None = None,
    scheduling_reason: str = "Unschedulable",
    node_name: str | None = None,
) -> V1Pod:
    """One pod, in the shape the cluster was measured returning it.

    `statuses_reported=False` puts `container_statuses` at None rather than at an empty
    list, because None is what the cluster actually returned for a pod nothing could
    schedule.

    `init_waiting` is a separate parameter and not a fourth entry in `ready`, because
    the cluster keeps the init container's status in its own list -- which is exactly
    the distinction the failing case below exists for.

    The spec is always populated, because a pod the cluster hands back always has one
    and the restart policy is what decides whether a container that exited is coming
    back.

    `scheduling_reason` exists because `Unschedulable` and `SchedulingGated` are the two
    reasons a real cluster puts on a `PodScheduled=False` condition and the adapter
    treats them oppositely -- one is waited out, the other refused. A builder that could
    only produce one of them would let the other's branch go ungraded.

    `node_name` is the fact the scheduling wait reads, and it is None by default because
    an unscheduled pod is the state that matters here; a pod the cluster hands back
    after scheduling carries the name of the node it landed on.
    """
    names = ("agent-runtime", "session-shim")

    def _status(name: str, is_ready: bool, reason: str | None) -> V1ContainerStatus:
        ended = (
            V1ContainerStateTerminated(
                reason=terminated[0], exit_code=terminated[1], message=waiting_detail
            )
            if terminated is not None and name == names[-1]
            else None
        )
        return V1ContainerStatus(
            name=name,
            ready=is_ready,
            restart_count=0,
            image="i",
            image_id="",
            state=V1ContainerState(
                waiting=V1ContainerStateWaiting(reason=reason, message=waiting_detail)
                if reason
                else None,
                terminated=ended,
            ),
        )

    reported = [
        _status(name, is_ready, waiting)
        for name, is_ready in zip(names, ready, strict=False)
    ]
    return V1Pod(
        spec=V1PodSpec(
            containers=[], restart_policy=restart_policy, node_name=node_name
        ),
        metadata=V1ObjectMeta(
            name="map-session-x",
            labels=labels if labels is not None else {"map.role": "session-pod"},
            deletion_timestamp=datetime(2026, 8, 22, tzinfo=UTC) if deleting else None,
        ),
        status=V1PodStatus(
            phase=phase,
            container_statuses=reported if statuses_reported else None,
            init_container_statuses=[_status("seed-runtime-home", False, init_waiting)]
            if init_waiting
            else None,
            conditions=[
                V1PodCondition(
                    type="PodScheduled",
                    status="False",
                    reason=scheduling_reason,
                    message=unschedulable,
                )
            ]
            if unschedulable
            else None,
        ),
    )


def test_the_manifest_this_reads_is_the_one_the_cluster_gets() -> None:
    """The non-vacuous pair for every substitution case below.

    Without it, a `_pod_for` that returned an empty pod would satisfy each "and nothing
    else changed" assertion perfectly.
    """
    raw = _manifest()
    assert raw["kind"] == "Pod"
    names = [container["name"] for container in raw["spec"]["containers"]]
    assert "agent-runtime" in names
    assert "session-shim" in names
    assert _secret_volumes(raw) == ("compiled", "requirements", "shim-token")


def test_the_compiled_pod_carries_this_sessions_image_name_label_and_secrets() -> None:
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    pod = _pod_for(_manifest(), pod_name, compiled)
    spec: dict[str, Any] = pod["spec"]
    metadata: dict[str, Any] = pod["metadata"]
    assert metadata["name"] == pod_name
    assert metadata["labels"]["map.session-id"] == str(compiled.session_id)
    every_container = list(spec["initContainers"]) + list(spec["containers"])
    assert {container["image"] for container in every_container} == {_AN_IMAGE}
    assert {
        volume["secret"]["secretName"]
        for volume in spec["volumes"]
        if "secret" in volume
    } == {
        f"{pod_name}-compiled",
        f"{pod_name}-requirements",
        f"{pod_name}-shim-token",
    }
    shim = next(c for c in spec["containers"] if c["name"] == "session-shim")
    environment = {entry["name"]: entry["value"] for entry in shim["env"]}
    assert environment["MAP_SESSION_ID"] == str(compiled.session_id)


def test_the_address_the_control_plane_dials_is_the_record_this_pod_publishes() -> None:
    """The pod's own DNS name, computed here from `spec`, must be the one dialled.

    Kubernetes builds a headless Service's per-pod A record from `spec.hostname` and
    `spec.subdomain`. `metadata.name` reaches the container's `/etc/hostname` and
    nothing else, and the endpoints controller publishes an address's `hostname` only
    when `spec.hostname` is set. So a pod carrying the subdomain alone has no record of
    its own, which is what shipped until 2026-08-23: a Session pod ran 2/2 while a Turn
    against it failed as undeliverable, and from inside the control-plane pod the name
    below raised `gaierror` where the Service's bare name resolved to one arbitrary pod.

    Built from the two spec fields rather than asserted against `metadata.name`, because
    the equality that matters is between what the cluster will publish and what
    `shim_url_for` dials. Asserting `spec.hostname == pod_name` would pass while both
    sides drifted off the record together.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    spec: dict[str, Any] = _pod_for(_manifest(), pod_name, compiled)["spec"]
    namespace = "map-dev"
    record = f"{spec['hostname']}.{spec['subdomain']}.{namespace}.svc.cluster.local"
    assert record in shim_url_for(pod_name, namespace)


def test_the_shim_starts_its_thread_with_the_model_the_configuration_names() -> None:
    """All three of the shim's substituted variables carry a value, not a placeholder.

    Measured why this is asserted rather than assumed: with `MAP_MODEL` and
    `MAP_MODEL_PROVIDER` left at the manifest's empty strings, the shim's lifespan sent
    `thread/start` and the Agent Runtime answered `-32600`, so the container exited 3,
    the readiness probe never passed, and the pod never had both halves ready. An empty
    string here is a pod that starts and cannot serve, which is the failure a
    placeholder makes look like a slow start.
    """
    compiled = _compiled()
    pod = _pod_for(_manifest(), pod_name_for(compiled.session_id), compiled)
    spec: dict[str, Any] = pod["spec"]
    shim = next(c for c in spec["containers"] if c["name"] == "session-shim")
    environment = {entry["name"]: entry["value"] for entry in shim["env"]}
    assert environment["MAP_MODEL"] == compiled.model
    assert environment["MAP_MODEL_PROVIDER"] == compiled.model_provider
    assert all(
        environment[name]
        for name in ("MAP_SESSION_ID", "MAP_MODEL", "MAP_MODEL_PROVIDER")
    ), environment


def test_nothing_but_those_seven_fields_differs_from_the_manifest_on_disk() -> None:
    """Everything the manifest decides about confinement is copied through untouched.

    The security contexts, the probes, the node selector and the deny-rule ordering the
    init container enforces are the manifest's business. This blanks the substituted
    fields on both sides and asserts the rest is identical, so a rewrite that reached
    one field further fails here rather than surprising somebody in a pod.

    It was five fields until `spec.hostname` joined them, and this test is how that was
    noticed: the blanking below did not cover the new field, so it failed rather than
    letting a sixth substitution in silently.

    The seventh is the seed container's `MAP_RESUMING`, and it went in the other way --
    the blanking did NOT fail when that substitution arrived, because a Session that is
    not resuming is substituted with the same `false` the manifest already carries. So
    this drives a RESUMING Session, where the substituted value and the manifest's
    differ, and blanks both tables. A version of this reading `_SHIM_ENV` alone passes
    against a seed container filled with anything at all.
    """
    compiled = replace(_compiled(), resuming=True)
    pod_name = pod_name_for(compiled.session_id)
    produced = _pod_for(_manifest(), pod_name, compiled)
    substituted_names = set(_SHIM_ENV) | set(_SEED_ENV)

    def _blanked(metadata: dict[str, Any], spec: dict[str, Any]) -> str:
        metadata["name"] = "<name>"
        spec["hostname"] = "<name>"
        metadata["labels"].pop("map.session-id", None)
        for container in list(spec["initContainers"]) + list(spec["containers"]):
            container["image"] = "<image>"
            for entry in container.get("env", ()):
                if entry["name"] in substituted_names:
                    entry["value"] = "<substituted>"
        for volume in spec["volumes"]:
            if "secret" in volume:
                volume["secret"]["secretName"] = "<secret>"
        return yaml.safe_dump({"metadata": metadata, "spec": spec}, sort_keys=True)

    original = _manifest()
    assert _blanked(produced["metadata"], produced["spec"]) == _blanked(
        original["metadata"], original["spec"]
    )


def test_the_manifest_on_disk_is_not_mutated_by_compiling_a_pod_from_it() -> None:
    """Two Sessions compiled from one parsed manifest do not inherit each other.

    `_pod_for` deep-copies because the runner holds the parsed manifest for the life of
    the process. Without the copy the first Session's image and secret names would be
    written into the shared mapping and every later pod would carry them.
    """
    manifest = _manifest()
    first = _compiled("registry.invalid/first@" + _A_DIGEST)
    _pod_for(manifest, pod_name_for(first.session_id), first)
    second = _compiled("registry.invalid/second@" + _A_DIGEST)
    pod = _pod_for(manifest, pod_name_for(second.session_id), second)
    spec: dict[str, Any] = pod["spec"]
    assert manifest["metadata"]["name"] == "map-session"
    assert "map.session-id" not in manifest["metadata"]["labels"]
    assert {container["image"] for container in spec["containers"]} == {
        "registry.invalid/second@" + _A_DIGEST
    }


def test_a_manifest_without_a_shim_container_is_refused() -> None:
    manifest = _manifest()
    manifest["spec"]["containers"] = [
        container
        for container in manifest["spec"]["containers"]
        if container["name"] != "session-shim"
    ]
    compiled = _compiled()
    with pytest.raises(PodNotStarted, match="session-shim"):
        _pod_for(manifest, pod_name_for(compiled.session_id), compiled)


def test_a_manifest_that_declares_no_model_variable_is_refused() -> None:
    """A variable the manifest does not declare is a value the shim reads as unset.

    The substitution only rewrites entries that are already there, so dropping one from
    the manifest silently produces a pod whose shim starts, asks the Agent Runtime for a
    thread it will refuse, and exits -- indistinguishable from a slow start until the
    readiness budget runs out. Refused at compile time instead, naming the variable.
    """
    manifest = _manifest()
    shim = next(
        c for c in manifest["spec"]["containers"] if c["name"] == "session-shim"
    )
    shim["env"] = [entry for entry in shim["env"] if entry["name"] != "MAP_MODEL"]
    compiled = _compiled()
    with pytest.raises(PodNotStarted, match="MAP_MODEL"):
        _pod_for(manifest, pod_name_for(compiled.session_id), compiled)


def test_a_secret_volume_this_adapter_has_no_file_for_is_refused() -> None:
    manifest = _manifest()
    manifest["spec"]["volumes"].append(
        {"name": "invented", "secret": {"secretName": "whatever"}}
    )
    with pytest.raises(PodNotStarted, match="invented"):
        _secret_volumes(manifest)


def test_each_compiled_document_lands_under_the_filename_the_runtime_reads() -> None:
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    secrets = _secrets_for(
        pod_name, ("compiled", "requirements", "shim-token"), compiled, A_KEY
    )
    by_name = {secret.metadata.name: secret.string_data for secret in secrets}
    assert by_name[f"{pod_name}-compiled"] == {"config.toml": compiled.config_toml}
    assert by_name[f"{pod_name}-requirements"] == {
        "requirements.toml": compiled.requirements_toml
    }
    assert by_name[f"{pod_name}-shim-token"] == {
        "token": shim_token_for(compiled.session_id, A_KEY)
    }
    assert all(secret.data is None for secret in secrets)


def test_the_shim_token_is_the_one_the_control_plane_will_present() -> None:
    """Derived through the same function, so the two halves cannot disagree.

    A token written here by a second derivation would be a bearer check that fails for
    the only caller allowed through it, and the failure would look like a network fault.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    (secret,) = _secrets_for(pod_name, ("shim-token",), compiled, A_KEY)
    assert secret.string_data["token"] == shim_token_for(compiled.session_id, A_KEY)
    assert secret.string_data["token"] != shim_token_for(
        compiled.session_id, b"another key"
    )


def test_each_skill_rides_in_the_secret_that_already_mounts_the_config_root() -> None:
    """One key per skill, beside the file that volume was already carrying.

    The base file is asserted present in the same breath as the skills, because the
    Secret is what the `items` projection below draws from -- a Secret holding only
    skills would produce a mount with no `requirements.toml` in it and the runtime
    would start with no managed configuration at all.
    """
    compiled = _compiled_with_skills()
    pod_name = pod_name_for(compiled.session_id)
    (secret,) = _secrets_for(pod_name, (_SKILL_VOLUME,), compiled, A_KEY)
    assert secret.string_data == {
        "requirements.toml": compiled.requirements_toml,
        "skill.skills_pdf_SKILL.md": _TWO_SKILLS[1].text,
        "skill.skills_deep-wiki_SKILL.md": _TWO_SKILLS[0].text,
    }


def test_a_session_with_no_skills_carries_exactly_what_it_carried_before() -> None:
    """The one key, and nothing beside it.

    Asserted as an equality rather than as three `in` checks: a delivery that leaked an
    empty skill key, or a placeholder, would satisfy every containment assertion and
    would put a file the runtime tries to read into every Session that asked for none.
    """
    compiled = _compiled()
    assert compiled.skill_files == ()
    pod_name = pod_name_for(compiled.session_id)
    (secret,) = _secrets_for(pod_name, (_SKILL_VOLUME,), compiled, A_KEY)
    assert secret.string_data == {"requirements.toml": compiled.requirements_toml}


def test_a_skill_is_projected_to_the_path_the_runtime_discovers_it_under() -> None:
    """The whole delivery, asserted as an absolute path inside the pod.

    This is the case that grades the claim the other three only support. The Secret key
    and the projected relative path are both this adapter's inventions, and either one
    could be self-consistently wrong -- so the mount path is read off the manifest and
    joined to the projection, and the result is compared against the directory codex
    reads admin-scope skills from. A delivery that lands one segment away is a Session
    whose agent reports having no skills, with every unit assertion still green.
    """
    compiled = _compiled_with_skills()
    pod = _pod_for(_manifest(), pod_name_for(compiled.session_id), compiled)
    spec: dict[str, Any] = pod["spec"]
    volume = next(one for one in spec["volumes"] if one["name"] == _SKILL_VOLUME)
    mounts = [
        mount
        for container in spec["containers"]
        for mount in container["volumeMounts"]
        if mount["name"] == _SKILL_VOLUME
    ]
    assert len(mounts) == 1, "one container mounts the config root, or there are two"
    root = mounts[0]["mountPath"]
    assert mounts[0].get("readOnly") is True
    delivered = {f"{root}/{item['path']}" for item in volume["secret"]["items"]}
    assert delivered == {
        "/etc/codex/requirements.toml",
        "/etc/codex/skills/pdf/SKILL.md",
        "/etc/codex/skills/deep-wiki/SKILL.md",
    }


def test_every_key_the_secret_holds_is_a_key_the_projection_names() -> None:
    """The two halves are one delivery, so neither names what the other does not.

    A key with no item is a skill written into the cluster and mounted nowhere; an item
    with no key makes kubelet refuse the mount and the pod never starts. Both are
    invisible in a diff that touches only one of the two functions, which is why this
    compares them rather than checking either against a literal.
    """
    compiled = _compiled_with_skills()
    pod_name = pod_name_for(compiled.session_id)
    (secret,) = _secrets_for(pod_name, (_SKILL_VOLUME,), compiled, A_KEY)
    pod = _pod_for(_manifest(), pod_name, compiled)
    volume = next(one for one in pod["spec"]["volumes"] if one["name"] == _SKILL_VOLUME)
    assert {item["key"] for item in volume["secret"]["items"]} == set(
        secret.string_data
    )


def test_a_session_with_no_skills_leaves_the_manifests_own_projection_alone() -> None:
    """No `items` at all, rather than an empty list.

    `items` is exhaustive and an empty one projects nothing, so a Session with no skills
    that emitted `items: []` would mount an empty directory over the config root and
    every Session on the platform would lose `requirements.toml` at once. This is the
    assertion that keeps the no-skills path byte-identical to the manifest on disk.
    """
    compiled = _compiled()
    pod = _pod_for(_manifest(), pod_name_for(compiled.session_id), compiled)
    for volume in pod["spec"]["volumes"]:
        if "secret" in volume:
            assert "items" not in volume["secret"], volume["name"]


def test_skills_with_no_volume_to_ride_in_are_refused_rather_than_dropped() -> None:
    """A manifest that stopped declaring the volume refuses the pod.

    The alternative is what makes this worth a case: the Secret would still be created
    with every skill in it, the pod would start, and the agent would report having no
    skills. Nothing else in this module would notice, because every other assertion
    here is about a volume that is present.
    """
    manifest = _manifest()
    kept = [
        volume
        for volume in manifest["spec"]["volumes"]
        if volume["name"] != _SKILL_VOLUME
    ]
    manifest["spec"]["volumes"] = kept
    compiled = _compiled_with_skills()
    with pytest.raises(PodNotStarted, match=_SKILL_VOLUME):
        _pod_for(manifest, pod_name_for(compiled.session_id), compiled)


_A_SKILL_OF_SEVERAL_FILES = (
    SkillFile(
        relative_path="skills/pdf/SKILL.md",
        text="---\nname: pdf\ndescription: Build a PDF.\n---\nRead forms.md.\n",
    ),
    SkillFile(relative_path="skills/pdf/forms.md", text="How to fill a form.\n"),
    SkillFile(relative_path="skills/pdf/reference/tables.md", text="Columns.\n"),
)
"""One skill delivered as three files, the last of them two directories down.

The shape Anthropic's published `pdf` skill has: a `SKILL.md` that tells the model to
read a sibling beside it. The nested third file is what separates a delivery that
handles one extra path segment from one that handles any number -- a key built by
stripping a fixed `/SKILL.md` suffix passes the flat sibling and fails this one.
"""

_A_COLLIDING_PAIR = (
    SkillFile(relative_path="skills/pdf/ref/forms.md", text="Nested.\n"),
    SkillFile(relative_path="skills/pdf/ref_forms.md", text="Flat.\n"),
)
"""Two paths that flattening separators cannot keep apart.

Flattening is what makes a nested path legal as a key, and it is not injective: the
separator and the character replacing it both occur in paths a tenant may send. This is
the smallest pair that proves it, and the bundle parse accepts both members -- neither
is absolute, neither climbs, neither has an empty segment.
"""

_A_PATH_NO_KEY_CAN_HOLD = (
    SkillFile(relative_path="skills/pdf/forms (1).md", text="A second copy.\n"),
)
"""A path the bundle parse accepts and a Secret key cannot hold.

The upstream parse refuses an absolute path, a `..`, an empty or `.` segment and a
control character, and nothing else -- a space and a parenthesis both survive it, and a
browser's own name for a duplicated download has both. So the key's character class is
not guaranteed anywhere upstream, which is why the delivery enforces it itself.
"""

_LEGAL_SECRET_KEY = re.compile(r"^[-._a-zA-Z0-9]+$")
"""What the Kubernetes API admits as a Secret key.

Matched against every key a delivery produced rather than against a hand-listed set, so
a case grades the rule the API server actually applies instead of the keys whoever wrote
the case happened to think of.
"""


def test_a_secret_key_is_the_whole_path_with_its_separators_flattened() -> None:
    """The encoding itself, stated once so no case below has to restate it.

    Worth its own case because every other skill-delivery assertion rests on it: the
    key is derived from the whole relative path, so two files of one skill cannot share
    a key and no file of one skill can take the key of another skill's file.
    """
    assert _skill_secret_key("skills/pdf/SKILL.md") == "skill.skills_pdf_SKILL.md"
    assert (
        _skill_secret_key("skills/pdf/reference/tables.md")
        == "skill.skills_pdf_reference_tables.md"
    )


def test_every_file_of_a_multi_file_skill_is_projected_at_its_own_path() -> None:
    """The whole file set delivered at its own paths, beside the base file.

    A `SKILL.md` telling the model to read `forms.md` is a broken skill when only the
    `SKILL.md` arrives: the agent follows the instruction, finds no file, and reports
    the skill unusable -- while every assertion about the file that *was* delivered
    stays green. So this compares the full projected set, which is the only shape of
    assertion that fails when a sibling is silently dropped.
    """
    compiled = _compiled_with_skills(_A_SKILL_OF_SEVERAL_FILES)
    pod = _pod_for(_manifest(), pod_name_for(compiled.session_id), compiled)
    volume = next(one for one in pod["spec"]["volumes"] if one["name"] == _SKILL_VOLUME)
    assert {item["path"] for item in volume["secret"]["items"]} == {
        "requirements.toml",
        "skills/pdf/SKILL.md",
        "skills/pdf/forms.md",
        "skills/pdf/reference/tables.md",
    }


def test_a_multi_file_skill_keeps_the_base_file_in_the_secret() -> None:
    """`requirements.toml` still a key, asserted as an equality.

    `items` is exhaustive, so the base file has to be both a key in the Secret and an
    item in the projection or the runtime mounts a config root with no managed
    configuration in it. An equality and not a containment check: a delivery that
    grew a spare key would satisfy every `in`.
    """
    compiled = _compiled_with_skills(_A_SKILL_OF_SEVERAL_FILES)
    pod_name = pod_name_for(compiled.session_id)
    (secret,) = _secrets_for(pod_name, (_SKILL_VOLUME,), compiled, A_KEY)
    assert set(secret.string_data) == {
        "requirements.toml",
        _skill_secret_key("skills/pdf/SKILL.md"),
        _skill_secret_key("skills/pdf/forms.md"),
        _skill_secret_key("skills/pdf/reference/tables.md"),
    }


def test_no_path_a_skill_can_hold_produces_a_key_kubernetes_refuses() -> None:
    """Every key legal, graded against the API's own character class.

    A key holding a `/` does not degrade the delivery -- the pod create fails outright
    with an API error naming a key, and that message says neither which skill nor which
    file produced it. The nested path in the set above is the one that used to produce
    exactly that.
    """
    compiled = _compiled_with_skills(_A_SKILL_OF_SEVERAL_FILES)
    pod_name = pod_name_for(compiled.session_id)
    (secret,) = _secrets_for(pod_name, (_SKILL_VOLUME,), compiled, A_KEY)
    assert len(secret.string_data) == 4
    for key in secret.string_data:
        assert _LEGAL_SECRET_KEY.match(key), key


def test_two_paths_whose_keys_collide_are_refused_rather_than_merged() -> None:
    """One of the two would win the mapping and the other would vanish silently.

    Refused rather than encoded around: a digest suffix would keep both keys unique and
    make every key unreadable, and an operator reading keys off a Secret to find out why
    a skill did not appear is the whole reason they are readable. A tenant can rename
    one of two files; nobody can read a hash.

    The message is required to name both paths and the key they share, because a refusal
    saying only that something collided leaves the tenant to find the pair themselves.
    """
    compiled = _compiled_with_skills(_A_COLLIDING_PAIR)
    with pytest.raises(PodNotStarted) as refusal:
        _skill_volume_items(compiled)
    message = str(refusal.value)
    assert "skills/pdf/ref/forms.md" in message
    assert "skills/pdf/ref_forms.md" in message
    assert _skill_secret_key("skills/pdf/ref_forms.md") in message


def test_a_colliding_pair_is_refused_before_any_secret_is_written() -> None:
    """The refusal cannot live only where `items` is built, because Secrets go first.

    `_create` writes every Secret and only then creates the pod. A collision caught
    just in the projection would be caught after a Secret holding one of the two files
    -- the other silently overwritten -- already existed in the cluster, and an
    already-existing Secret is left alone on the next attempt, so the wrong content
    would outlive the failure that revealed it.
    """
    compiled = _compiled_with_skills(_A_COLLIDING_PAIR)
    with pytest.raises(PodNotStarted) as refusal:
        _secrets_for(
            pod_name_for(compiled.session_id), (_SKILL_VOLUME,), compiled, A_KEY
        )
    assert "skills/pdf/ref/forms.md" in str(refusal.value)


def test_a_path_no_secret_key_can_hold_is_refused_rather_than_written() -> None:
    """Refused here, because nothing upstream makes the character unrepresentable.

    The refusal names the path, so the tenant knows which file to rename. The
    alternative is a pod create failing with the API server's own complaint about an
    invalid key, which names neither the skill nor the file that produced it.
    """
    compiled = _compiled_with_skills(_A_PATH_NO_KEY_CAN_HOLD)
    with pytest.raises(PodNotStarted, match="forms"):
        _skill_volume_items(compiled)


def test_a_secret_is_named_after_the_pod_that_owns_it() -> None:
    assert _secret_name("map-session-x", "compiled") == "map-session-x-compiled"


def test_a_running_pod_with_every_container_ready_is_running() -> None:
    assert _phase_of(_a_pod()) is PodPhase.RUNNING


def test_a_running_pod_with_one_container_not_ready_is_still_starting() -> None:
    """A Session is both halves. The runtime alone takes no Turn and says nothing."""
    assert _phase_of(_a_pod(ready=(True, False))) is PodPhase.STARTING
    assert _phase_of(_a_pod(ready=(False, True))) is PodPhase.STARTING


def test_a_running_pod_that_has_reported_no_container_statuses_is_not_running() -> None:
    """The emptiness check, asserted rather than left to `all(())` being true.

    Every pod looks like this for its first moment and an unschedulable one looks like
    it for ever, so a vacuous `all()` here would dispatch a tenant's Turn into a pod
    with no containers.
    """
    assert _phase_of(_a_pod(statuses_reported=False)) is PodPhase.STARTING


def test_a_pod_being_deleted_is_gone_even_while_its_phase_reads_running() -> None:
    """Deletion is asynchronous: the phase and the ready flags survive the grace period.

    So a pod being torn down still reads Running, and a Turn dispatched into it dies
    with it. The deletion timestamp is the only thing that says so.
    """
    dying = _a_pod(deleting=True)
    assert dying.status.phase == "Running"
    assert all(status.ready for status in dying.status.container_statuses)
    assert _phase_of(dying) is PodPhase.GONE


def test_a_failed_a_succeeded_and_an_unknown_pod_are_all_gone() -> None:
    """Unknown is folded in with the two finished cases on purpose.

    It means the node stopped reporting, and a pod nobody can hear from is not one to
    keep waiting for.
    """
    for phase in ("Failed", "Succeeded", "Unknown"):
        assert _phase_of(_a_pod(phase=phase, ready=(False, False))) is PodPhase.GONE


def test_an_unschedulable_pod_is_waited_out_rather_than_refused() -> None:
    """Unschedulable is what a Cluster Autoscaler acts on, so it is not a refusal.

    This asserted the opposite until 2026-08-23 -- that `_why_it_will_not_start` names
    the scheduler's message -- and the assertion was the platform's whole concurrent-
    Session ceiling. Measured against `map-dev`: twenty-four Sessions submitted at once,
    and every pod past the node count was refused with `Unschedulable: 0/2 nodes are
    available: 2 Too many pods` in the same second it was created, on a cluster whose
    ASG goes to eight nodes. The autoscaler was never given the chance it exists for.

    Two halves, because dropping the refusal without keeping the message would trade one
    defect for another: an operator whose pods never schedule needs to be told it was
    capacity and not a bug, and the condition is overwritten the moment a pod finally
    lands.
    """
    unschedulable = _a_pod(
        phase="Pending",
        statuses_reported=False,
        unschedulable="0/2 nodes are available: 2 Insufficient cpu",
    )
    assert _why_it_will_not_start(unschedulable) is None
    waiting_on = _why_it_is_not_scheduled(unschedulable)
    assert waiting_on is not None
    assert "Unschedulable" in waiting_on
    assert "Insufficient cpu" in waiting_on


def test_a_pod_held_by_a_scheduling_gate_is_refused_and_not_waited_out() -> None:
    """The non-vacuous pair for the case above, and a distinction with a reason.

    A gate is removed by a controller, and this platform sets none -- so a gated Session
    pod means something outside the platform is holding it, and waiting seven minutes to
    say so helps nobody. Without this case, `_SCHEDULING_MAY_RESOLVE` could be widened
    to every `PodScheduled=False` reason and every assertion here would still pass.
    """
    gated = _a_pod(
        phase="Pending",
        statuses_reported=False,
        unschedulable="waiting for a gate nothing here sets",
        scheduling_reason="SchedulingGated",
    )
    refusal = _why_it_will_not_start(gated)
    assert refusal is not None
    assert "SchedulingGated" in refusal


def test_a_pod_is_scheduled_exactly_when_it_names_the_node_it_landed_on() -> None:
    """`spec.node_name` and not the condition, and the two disagree where it matters.

    A pod the API server has only just accepted has NO `PodScheduled` condition at
    all -- neither True nor False -- so a reader that required `status == "True"` would
    treat "the scheduler has not answered yet" and "the scheduler said no" as one state.
    They are not: the first is every pod's first moment, and starting a capacity clock
    over it would spend the scheduling bound on a pod about to be placed anyway.
    """
    assert not _is_scheduled(_a_pod(phase="Pending", statuses_reported=False))
    assert not _is_scheduled(
        _a_pod(phase="Pending", statuses_reported=False, unschedulable="no room")
    )
    assert _is_scheduled(_a_pod(node_name="ip-172-31-0-1.ec2.internal"))
    # A pod that landed on a node while a stale unschedulable condition is still on the
    # object. The condition is what the old reader believed; the node name is the fact.
    assert _is_scheduled(
        _a_pod(
            phase="Pending",
            statuses_reported=False,
            unschedulable="no room",
            node_name="ip-172-31-0-1.ec2.internal",
        )
    )


@pytest.mark.parametrize("reason", sorted(_WILL_NOT_START))
def test_every_reason_in_the_refusal_set_really_refuses(reason: str) -> None:
    """The set is graded entry by entry, and it was graded by nothing at all.

    Four of its five members were reachable by no assertion in this file -- only
    `ImagePullBackOff` was, through the `stuck` helper -- so the set could be edited in
    either direction and every test here would still pass. That is how
    `CreateContainerError` came to sit in it while making every autoscaled node
    unusable: removing it broke no test, and nor would removing the other four.

    Parametrized over the set itself rather than over a list written here. A copy would
    be free to fall behind, which is the failure this exists to prevent.
    """
    refusal = _why_it_will_not_start(
        _a_pod(phase="Pending", ready=(False, False), waiting=reason)
    )
    assert refusal is not None, f"{reason} is in the refusal set and does not refuse"
    assert reason in refusal
    assert "agent-runtime" in refusal, (
        "the refusal does not say which container, which is the half that tells an "
        f"operator where to look: {refusal}"
    )


@pytest.mark.parametrize("reason", sorted(_CONTAINER_MAY_RESOLVE))
def test_a_container_create_failure_is_waited_out_and_its_reason_kept(
    reason: str,
) -> None:
    """Not a refusal, and the reason survives for the timeout that may follow.

    `CreateContainerError` was a refusal until 2026-08-23, and on a cluster that can
    add nodes that made every new node unusable. Measured: the autoscaler added two
    nodes in 1m0s, Session pods scheduled onto them, and each was refused with `cannot
    load seccomp profile ... no such file or directory` -- the profile is written by a
    DaemonSet, and a DaemonSet pod and a Session pod become schedulable on a new node in
    the same instant.

    Both halves again. Dropping the refusal without keeping the message would leave a
    pod that genuinely never comes up timing out with nothing but "still STARTING", and
    the sentence naming the missing file only in a kubelet event.
    """
    pod = _a_pod(
        phase="Pending",
        ready=(False, False),
        waiting=reason,
        waiting_detail="cannot load seccomp profile: no such file or directory",
    )
    assert _why_it_will_not_start(pod) is None
    kept = _why_a_container_has_not_been_created(pod)
    assert kept is not None
    assert reason in kept
    assert "seccomp" in kept


def test_the_two_container_verdicts_never_claim_the_same_reason() -> None:
    """A reason cannot be both refused on sight and waited out.

    Written because the two sets are edited by hand, and an entry moved from one to the
    other by copying rather than cutting would leave the wait refusing what it also
    believes is transient -- and the refusal runs first, so the transient treatment
    would silently do nothing.
    """
    assert not (_WILL_NOT_START & _CONTAINER_MAY_RESOLVE)


def test_a_pod_backing_off_an_image_pull_names_the_container_and_the_reason() -> None:
    reason = _why_it_will_not_start(
        _a_pod(
            phase="Pending",
            ready=(False, False),
            waiting="ImagePullBackOff",
            waiting_detail='Back-off pulling image "registry.invalid/x"',
        )
    )
    assert reason is not None
    assert "agent-runtime" in reason
    assert "ImagePullBackOff" in reason
    assert "registry.invalid/x" in reason


def test_an_init_container_that_cannot_pull_its_image_is_a_refusal() -> None:
    """The list this is read from is the one the failure actually lands in.

    Measured against `map-dev`: given a digest that does not exist, this pod reported
    `ImagePullBackOff` on its *init* container while all three main containers sat at
    `PodInitializing` -- because every container here carries the same image, so the
    init container is always the first to try to pull it. A reader that looked only at
    `container_statuses` therefore saw nothing wrong, waited out the whole 180 s bound,
    and reported "still starting" with the registry's own reason discarded.
    """
    reason = _why_it_will_not_start(
        _a_pod(
            phase="Pending",
            ready=(False, False),
            waiting="PodInitializing",
            init_waiting="ImagePullBackOff",
            waiting_detail='Back-off pulling image "registry.invalid/x"',
        )
    )
    assert reason is not None
    assert "seed-runtime-home" in reason
    assert "ImagePullBackOff" in reason


def test_a_container_that_exited_and_will_not_be_restarted_is_a_refusal() -> None:
    """The case a pod that never leaves `Running` looks exactly like a slow start.

    Measured against `map-dev` with the real image: the runtime container reached ready
    and the shim container exited 3 on `thread/start`. Under `restartPolicy: Never`
    kubelet does not bring it back, so the pod sat at phase `Running` with one container
    terminated for as long as it was watched -- it never became `Failed`, which is what
    a reader of the phase alone would have been waiting for. Nothing was in a waiting
    state, so the whole 180 s bound was spent and the message said only "still
    starting", with the exit code discarded.
    """
    reason = _why_it_will_not_start(
        _a_pod(ready=(True, False), terminated=("Error", 3))
    )
    assert reason is not None
    assert "session-shim" in reason
    assert "3" in reason


def test_a_container_that_exited_where_it_will_be_restarted_is_not_a_refusal() -> None:
    """The restart policy is what makes the case above terminal, so it is read.

    Under a policy that restarts, a terminated container is a container between attempts
    -- and this adapter would be refusing a pod that was about to come up. The manifest
    pins `Never` today; reading the policy is what keeps this correct if it stops.
    """
    assert (
        _why_it_will_not_start(
            _a_pod(
                ready=(True, False),
                terminated=("Error", 3),
                restart_policy="OnFailure",
            )
        )
        is None
    )


def test_a_container_that_finished_cleanly_is_not_a_refusal() -> None:
    """Exit 0 is the init container completing, which is how every pod here starts."""
    assert _why_it_will_not_start(_a_pod(terminated=("Completed", 0))) is None


def test_a_pod_transiently_failing_one_pull_is_not_yet_a_refusal() -> None:
    """`ErrImagePull` is the transient state and settles into the terminal one.

    Measured: ContainerCreating at 0 s, ErrImagePull at 3 s, ImagePullBackOff at 15 s --
    all well inside the bound. Refusing on the first `ErrImagePull` would refuse a pod
    that a registry blip would have let through.
    """
    for transient in ("ContainerCreating", "ErrImagePull", "PodInitializing"):
        assert (
            _why_it_will_not_start(
                _a_pod(phase="Pending", ready=(False, False), waiting=transient)
            )
            is None
        )


def test_a_pod_that_is_merely_starting_has_no_reason() -> None:
    assert _why_it_will_not_start(_a_pod(phase="Pending", ready=(False, False))) is None
    assert _why_it_will_not_start(_a_pod()) is None


def test_a_reason_longer_than_the_cap_is_truncated_rather_than_logged_whole() -> None:
    """An image pull failure carries a whole containerd trace; a log line does not."""
    reason = _why_it_will_not_start(
        _a_pod(
            phase="Pending",
            ready=(False, False),
            waiting="ImagePullBackOff",
            waiting_detail="x" * 5_000,
        )
    )
    assert reason is not None
    assert reason.count("x") == _REASON_CAP
    scheduling = _why_it_is_not_scheduled(
        _a_pod(phase="Pending", statuses_reported=False, unschedulable="y" * 5_000)
    )
    assert scheduling is not None
    assert scheduling.count("y") == _REASON_CAP


def _a_runner(namespace: str = "map-test") -> KubernetesPodRunner:
    return KubernetesPodRunner.from_manifest_file(
        MANIFEST, namespace=namespace, token_key=A_KEY
    )


def test_the_manifest_is_read_at_construction_so_a_missing_one_fails_before_a_session(
    tmp_path: Path,
) -> None:
    """A manifest that is absent or is not a manifest fails before any placement.

    Read once at construction rather than per call, so a mistyped `MAP_POD_MANIFEST`
    stops the process starting instead of failing the first tenant's Session with a
    filesystem error three layers down.
    """
    assert _a_runner().manifest["kind"] == "Pod"
    with pytest.raises(FileNotFoundError):
        KubernetesPodRunner.from_manifest_file(
            tmp_path / "nothing.yaml", namespace="map-test", token_key=A_KEY
        )
    not_a_manifest = tmp_path / "list.yaml"
    not_a_manifest.write_text("- one\n- two\n")
    with pytest.raises(PodNotStarted, match="does not parse"):
        KubernetesPodRunner.from_manifest_file(
            not_a_manifest, namespace="map-test", token_key=A_KEY
        )


def test_a_runner_with_no_signing_key_is_refused_rather_than_signing_with_nothing() -> (
    None
):
    """An empty key derives a token every pod on the cluster can also derive.

    The shim's bearer check is the only thing between a Session's runtime and another
    Session's Turn route, and HMAC under an empty key is a public function of the
    Session id. Refused at construction, so the process does not start rather than
    serving with a check that passes for anyone.
    """
    with pytest.raises(PodNotStarted, match="signing key"):
        KubernetesPodRunner.from_manifest_file(
            MANIFEST, namespace="map-test", token_key=b""
        )


async def test_a_mismatched_pod_name_and_configuration_is_refused_before_anything() -> (
    None
):
    """The port hands the name and the configuration over separately, so they can part.

    Started from another Session's compiled documents, a pod would run under the wrong
    Permission Profile and the wrong Tool Gateway identity while `Placement.locate`
    still found it under this Session's name. Nothing downstream would notice, which is
    why the re-derivation is the first statement of `ensure` -- and why this case needs
    no cluster: nothing is created before it raises.
    """
    runner = _a_runner()
    compiled = _compiled()
    with pytest.raises(PodNotStarted, match=str(compiled.session_id)):
        await runner.ensure(pod_name_for(new_session_id()), compiled)


def test_a_pod_without_the_session_label_is_reported_absent() -> None:
    """Names collide: anything in the namespace can create `map-session-<uuid>`.

    Reporting such a pod RUNNING would have the control plane dispatch a tenant's Turn
    into a process it knows nothing about.
    """
    assert not _claims_this_session(_a_pod(labels={}), "map-session-x")
    assert not _claims_this_session(_a_pod(labels=None), "map-session-x")


def test_a_pod_labelled_with_a_different_session_is_reported_absent() -> None:
    mine = new_session_id()
    theirs = new_session_id()
    pod = _a_pod(labels={"map.session-id": str(theirs)})
    assert not _claims_this_session(pod, pod_name_for(mine))
    assert _claims_this_session(pod, pod_name_for(theirs))


def test_a_pod_labelled_with_something_that_is_not_a_session_id_is_absent() -> None:
    for claimed in ("", "not-a-uuid", "map-session-x", "0"):
        pod = _a_pod(labels={"map.session-id": claimed})
        assert not _claims_this_session(pod, "map-session-x")


def test_the_runner_is_built_from_the_environment_with_no_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composition root hands back the port, built from named variables only.

    Asserted through `pod_runner_from_environment` rather than by constructing the
    class, because what a deployed process gets is whatever that function returns -- and
    the served process already calls it.
    """
    monkeypatch.setenv("MAP_POD_MANIFEST", str(MANIFEST))
    monkeypatch.setenv("MAP_NAMESPACE", "map-somewhere")
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a-key")
    runner = pod_runner_from_environment()
    assert isinstance(runner, KubernetesPodRunner)
    assert runner.namespace == "map-somewhere"
    assert runner.token_key == b"a-key"
    assert runner.manifest["kind"] == "Pod"


def test_a_process_that_names_no_manifest_places_no_pods_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No manifest named means this process is not a placer, and it says so with None.

    `build` then wires `NoPodTransport` and every Turn is refused as undeliverable,
    which is the honest answer. Guessing a manifest path relative to the source tree
    would resolve in a checkout and not in an image.

    The namespace is cleared explicitly rather than left to whatever the environment
    holds, and that is load-bearing now: a process naming a namespace with no manifest
    refuses to start, because it has declared an intention it cannot act on. This case
    is about the process that declared nothing, so it has to say so.
    """
    monkeypatch.delenv("MAP_POD_MANIFEST", raising=False)
    monkeypatch.delenv("MAP_NAMESPACE", raising=False)
    assert pod_runner_from_environment() is None


def test_a_named_manifest_with_no_namespace_or_key_stops_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half-configured is refused outright, in both directions.

    A namespace defaulting to `default` would put a tenant's Session wherever the
    process happened to land, and the dispatch computes the shim's address from the same
    variable -- so a default here and a default there could disagree about which
    namespace a Session is in.
    """
    monkeypatch.setenv("MAP_POD_MANIFEST", str(MANIFEST))
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a-key")
    monkeypatch.delenv("MAP_NAMESPACE", raising=False)
    with pytest.raises(KeyError, match="MAP_NAMESPACE"):
        pod_runner_from_environment()
    monkeypatch.setenv("MAP_NAMESPACE", "map-somewhere")
    monkeypatch.delenv("MAP_SHIM_TOKEN_KEY", raising=False)
    with pytest.raises(KeyError, match="MAP_SHIM_TOKEN_KEY"):
        pod_runner_from_environment()


def test_the_signing_key_is_not_in_this_object_or_its_repr() -> None:
    """A rendered runner must not carry the key every shim bearer is derived from.

    This is a disclosure guard, not a tidiness one. pytest prints the local variables of
    a failing frame, and an error reporter that captures frame locals ships them off the
    machine -- so one `repr` of this object in a CI log hands over every Session token
    the deployment will ever mint. It was observed happening: the live-tier test below
    holds the runner in a local and fails today by design.

    Asserted two ways because they can fail apart. The key must be absent from the text,
    and no field may be `repr`-visible whose value is the key -- the second catches a
    field added later that carries it under another name.
    """
    key = b"a signing key this test alone knows"
    runner = KubernetesPodRunner.from_manifest_file(
        MANIFEST, namespace="map-test", token_key=key
    )
    rendered = repr(runner)
    assert key.decode() not in rendered
    assert repr(key) not in rendered
    assert key.hex() not in rendered
    shown = [f.name for f in fields(runner) if f.repr]
    assert [name for name in shown if getattr(runner, name) == key] == []
    # The key is still there to sign with -- excluded from the rendering, not dropped.
    assert runner.token_key == key


def test_the_namespace_is_a_field_and_never_a_method_parameter() -> None:
    """This adapter's whole blast radius: no caller can address another namespace.

    Asserted structurally because it is a property of the signatures rather than of any
    one call, and because the three methods are what a caller reaches.
    """
    for method in (
        KubernetesPodRunner.ensure,
        KubernetesPodRunner.phase_of,
        KubernetesPodRunner.remove,
    ):
        assert (
            "namespace"
            not in method.__code__.co_varnames[: method.__code__.co_argcount]
        ), method.__name__


# --- Tier 2: the real client, a fake API server. -----------------------------------
#
# What the four tests below are for is `ensure` and `remove`, which tier 1 cannot reach
# and which are where the consequences are: one creates a pod and three Secrets, the
# other deletes them. Every assertion is against what the fake cluster is left holding,
# never against which calls were made -- a cleanup that deletes the pod and forgets the
# Secret named `-shim-token` leaves a bearer behind, and only the store shows that.


@pytest.fixture
async def a_fake_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[FakeCluster]:
    """The real generated client, talking to a fake API server over loopback.

    Two things are redirected and neither is a stub of this repository's code. The
    in-cluster tell is cleared so `_core_api` takes its kubeconfig branch -- the branch
    a developer's machine takes -- and the loader's default kubeconfig path is pointed
    at a file under `tmp_path`. That constant is read out of the library's module
    globals on each call, which is why setting it works, and the client still does its
    own real loading, parsing, connecting and deserializing.
    """
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    kubeconfig = tmp_path / "kubeconfig"
    monkeypatch.setattr(kube_config, "KUBE_CONFIG_DEFAULT_LOCATION", str(kubeconfig))
    async with fake_kubernetes_api(FakeCluster(), kubeconfig) as serving:
        yield serving


def _a_squatter(pod_name: str, labels: dict[str, str]) -> dict[str, Any]:
    """A pod already parked at a Session's name, made by something that is not us.

    Stamped, because every pod a real API server holds is. Leaving the stamp off made a
    listing test below pass for the wrong reason: the pod was skipped for having no age,
    so removing the label check it was written to grade broke nothing.
    """
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": labels,
            "creationTimestamp": rfc3339(0),
        },
        "spec": {"restartPolicy": "Never", "containers": [{"name": "impostor"}]},
    }


async def test_a_placement_creates_the_pod_and_all_three_of_its_secrets(
    a_fake_cluster: FakeCluster,
) -> None:
    """The whole of the happy path, at the seam where things are created.

    The three Secret names are asserted rather than the count, because the count is
    satisfied by three copies of the wrong one. Their contents are not asserted here --
    tier 1 grades those -- and the token's value is never read out at all.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.status_of[pod_name] = running("agent-runtime", "session-shim")

    phase = await _a_runner().ensure(pod_name, compiled)

    assert phase is PodPhase.RUNNING
    assert list(a_fake_cluster.pods) == [pod_name]
    assert sorted(a_fake_cluster.secrets) == sorted(
        f"{pod_name}-{volume}" for volume in ("compiled", "requirements", "shim-token")
    )
    owners = [
        secret["metadata"]["ownerReferences"][0]["name"]
        for secret in a_fake_cluster.secrets.values()
    ]
    assert owners == [pod_name, pod_name, pod_name]


async def test_a_pod_at_this_name_that_is_not_this_sessions_is_refused_not_adopted(
    a_fake_cluster: FakeCluster,
) -> None:
    """`ensure` reads the Session label back, the same as `phase_of` always has.

    Without this the two disagreed about one pod: `ensure` found the squatter, skipped
    the create -- so no pod of ours and no shim token ever existed -- and answered
    RUNNING, while `phase_of` answered ABSENT. The caller recorded a placed Session that
    every later Turn refused as undeliverable.

    That nothing was created is the second half of the assertion and the more important
    half: a refusal that had already minted the Secrets would leave a bearer token for a
    Session that does not exist.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.pods[pod_name] = _a_squatter(pod_name, labels={})
    a_fake_cluster.status_of[pod_name] = running("impostor")

    with pytest.raises(PodNotStarted, match="map.session-id") as refused:
        await _a_runner().ensure(pod_name, compiled)

    assert pod_name in str(refused.value)
    assert a_fake_cluster.secrets == {}
    assert a_fake_cluster.pods[pod_name]["spec"]["containers"][0]["name"] == "impostor"
    assert await _a_runner().phase_of(pod_name) is PodPhase.ABSENT


async def test_a_pod_that_labels_itself_consistently_is_adopted_whoever_made_it(
    a_fake_cluster: FakeCluster,
) -> None:
    """The limit of the adoption check, pinned so nobody has to rediscover it.

    This asserts the *permissive* answer on purpose. `phase_of` reports RUNNING for a
    pod that is not this platform's, provided its `map.session-id` label agrees with
    its own name -- which whatever created it sets for free. Demonstrated against the
    real cluster with a `busybox` pod: RUNNING from `locate`, and `HttpPodDispatch`
    would then POST the tenant's prompt and a valid shim bearer to the address the
    headless Service resolves to that pod.

    Here so that the gap is a fact under a name rather than a sentence in a docstring
    somebody deletes. A change that closes it -- namespace RBAC over pod creation, or a
    pod identity the platform mints -- has to come here and say so; a change that only
    reshuffles the label check will leave this green and prove nothing.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.pods[pod_name] = _a_squatter(
        pod_name, labels={"map.session-id": str(compiled.session_id)}
    )
    a_fake_cluster.status_of[pod_name] = running("impostor")

    assert await _a_runner().phase_of(pod_name) is PodPhase.RUNNING
    assert a_fake_cluster.pods[pod_name]["spec"]["containers"][0]["name"] == "impostor"


async def test_a_pod_that_will_not_start_is_deleted_with_its_secrets_before_the_refusal(
    a_fake_cluster: FakeCluster,
) -> None:
    """A failed placement leaves nothing, and the shim's bearer is what it must not
    leave.

    Measured against the real cluster before this held: an unpullable digest left the
    pod `Pending` for ever and all three Secrets behind it, with nothing on any path to
    reap them -- the live test that looked like it covered this passed only because the
    test itself called `remove` in its own `finally`.

    Asserted with no `remove` anywhere in this test, deliberately. A cleanup the caller
    performs is not the cleanup under test.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.status_of[pod_name] = stuck("ImagePullBackOff")

    with pytest.raises(PodNotStarted, match="ImagePullBackOff"):
        await _a_runner().ensure(pod_name, compiled)

    assert a_fake_cluster.pods == {}
    assert a_fake_cluster.secrets == {}


async def test_a_pod_that_waits_for_a_node_and_gets_one_is_placed_not_refused(
    a_fake_cluster: FakeCluster,
) -> None:
    """The whole point of treating Unschedulable as transient, driven end to end.

    A pod that no node will take, which then gets one. That is the ordinary shape of a
    burst on an autoscaled cluster: the scheduler cannot place the pod, the Cluster
    Autoscaler sees exactly that and adds a node, and the pod lands. Before 2026-08-23
    this raised `PodNotStarted` on the first read and the Session failed while the
    capacity it needed was on its way.

    The node arrives from a task of its own rather than from a scripted read count,
    because the assertion is about the wait outliving the unschedulable state and a
    read counter would let the loop pass by reading twice for its own reasons. Both
    halves of the transition are applied: `spec.nodeName`, which is what `_is_scheduled`
    reads, and a Running status -- a pod that is scheduled and never ready is a
    different case, graded by the timeout above.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.status_of[pod_name] = {
        "phase": "Pending",
        "conditions": [
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "Unschedulable",
                "message": "0/2 nodes are available: 2 Too many pods",
            }
        ],
    }

    async def a_node_arrives() -> None:
        await asyncio.sleep(_POLL_SECONDS * 2.5)
        a_fake_cluster.pods[pod_name]["spec"]["nodeName"] = "ip-172-31-0-9.ec2.internal"
        a_fake_cluster.status_of[pod_name] = running("agent-runtime", "session-shim")

    async with asyncio.TaskGroup() as tasks:
        placing = tasks.create_task(_a_runner().ensure(pod_name, compiled))
        tasks.create_task(a_node_arrives())

    assert placing.result() is PodPhase.RUNNING
    assert pod_name in a_fake_cluster.pods, (
        "the pod was placed and then deleted, so the wait refused it after all"
    )


async def test_a_pod_no_node_ever_takes_is_refused_saying_it_was_never_scheduled(
    a_fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-vacuous pair: waiting for a node is bounded, and says so when it ends.

    Without this, `_SCHEDULING_MAY_RESOLVE` would be indistinguishable from never
    refusing an unschedulable pod at all, and the case above would pass either way.

    The bound is monkeypatched down rather than waited out. Seven real minutes in a unit
    test is not a test anybody runs, and the number itself is graded by nothing here --
    what is graded is that the loop ends, that it ends as a refusal, and that the
    refusal carries the scheduler's own message about what was short. That message is
    the thing an operator acts on and it is gone from the object the moment a pod
    schedules.
    """
    monkeypatch.setattr(pod_runner_module, "_SCHEDULING_TIMEOUT_SECONDS", 2.0)
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.status_of[pod_name] = {
        "phase": "Pending",
        "conditions": [
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "Unschedulable",
                "message": "0/2 nodes are available: 2 Too many pods",
            }
        ],
    }

    with pytest.raises(PodNotStarted, match="never scheduled") as refused:
        await _a_runner().ensure(pod_name, compiled)
    assert "Too many pods" in str(refused.value), str(refused.value)
    assert a_fake_cluster.pods == {}, "a pod nothing would take was left behind"
    assert a_fake_cluster.secrets == {}


async def test_a_pod_that_was_already_here_is_not_deleted_when_it_will_not_start(
    a_fake_cluster: FakeCluster,
) -> None:
    """The exact complement, and the reason the cleanup is scoped to this call's create.

    A pod that this call did not make belongs to a Session that may still be coming up.
    Deleting it because a second caller's `ensure` did not like what it saw would
    destroy a live Session to tidy up after a caller -- so a pod that was already here
    is refused and left exactly where it was.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.pods[pod_name] = _a_squatter(
        pod_name, labels={"map.session-id": str(compiled.session_id)}
    )
    a_fake_cluster.status_of[pod_name] = stuck("ImagePullBackOff")

    with pytest.raises(PodNotStarted, match="ImagePullBackOff"):
        await _a_runner().ensure(pod_name, compiled)

    assert list(a_fake_cluster.pods) == [pod_name]


async def test_remove_takes_the_pod_and_every_secret_and_is_safe_to_repeat(
    a_fake_cluster: FakeCluster,
) -> None:
    """Absent is success at every step, which is what `release` promises its caller.

    The second `remove` is the assertion: it runs against a cluster where all four
    objects are already gone, and every one of its six deletions gets a 404 back from a
    real client. If any of them raised, releasing a Session twice would fail.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.status_of[pod_name] = running("agent-runtime", "session-shim")
    runner = _a_runner()
    await runner.ensure(pod_name, compiled)

    await runner.remove(pod_name)
    assert a_fake_cluster.pods == {}
    assert a_fake_cluster.secrets == {}

    await runner.remove(pod_name)


async def test_an_existing_secret_is_left_alone_rather_than_replaced(
    a_fake_cluster: FakeCluster,
) -> None:
    """The residue of a run that died between minting the Secrets and creating the pod.

    The compiled configuration is immutable for a Session's whole life, so a Secret
    bearing this Session's name already holds this content and replacing it would buy
    nothing. What this grades is that the 409 is tolerated rather than raised -- the
    placement has to complete over it, because otherwise a Session is permanently
    unplaceable after one badly-timed kill.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.secrets[f"{pod_name}-compiled"] = {
        "metadata": {"name": f"{pod_name}-compiled"},
        "stringData": {"config.toml": "left over from the run before"},
    }
    a_fake_cluster.status_of[pod_name] = running("agent-runtime", "session-shim")

    assert await _a_runner().ensure(pod_name, compiled) is PodPhase.RUNNING
    assert (
        a_fake_cluster.secrets[f"{pod_name}-compiled"]["stringData"]["config.toml"]
        == "left over from the run before"
    )


# What the cases below are for is `placed_pods`, which is the one read in this adapter
# that is not addressed to a single object -- so what they grade is what it *narrows*.
# A lister that came back with everything in the namespace would hand a sweep pods that
# are not this platform's, and the sweep deletes.


def _now_ms() -> int:
    """This process's clock in epoch milliseconds, for bounding a pod's stamped age.

    A bound and never an equality: the stamp comes from the fake at the moment of
    creation and this is read afterwards, so any exact comparison would be a race on the
    second boundary.
    """
    return int(datetime.now(UTC).timestamp() * 1000)


async def test_a_placed_pod_is_listed_with_its_session_phase_and_age(
    a_fake_cluster: FakeCluster,
) -> None:
    """The three fields a sweep decides on, read back off a pod the adapter created.

    The Session is asserted rather than the pod's name, because that is what the row
    carries: the name is a pure function of the Session and the reverse is a parse that
    can fail, so a row handing back a name would push that parse onto a caller that
    deletes. The age is asserted as a bound rather than a value -- the fake stamps the
    pod at the moment of creation, so any exact number here would be a clock race.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.status_of[pod_name] = running("agent-runtime", "session-shim")
    await _a_runner().ensure(pod_name, compiled)

    listed = await _a_runner().placed_pods()

    assert [pod.session_id for pod in listed] == [compiled.session_id]
    assert listed[0].phase is PodPhase.RUNNING
    assert 0 < listed[0].created_at_ms <= _now_ms()


async def test_a_pod_carrying_no_session_label_is_not_listed(
    a_fake_cluster: FakeCluster,
) -> None:
    """The label selector is the narrowing, and this is what it keeps out.

    A pod parked in this namespace by a hand-applied manifest, or by another tool, has
    no `map.session-id` label -- and a sweep that saw it would be deciding whether to
    delete something it knows nothing about.
    """
    a_fake_cluster.pods["something-else-entirely"] = _a_squatter(
        "something-else-entirely", labels={}
    )
    a_fake_cluster.status_of["something-else-entirely"] = running("impostor")

    assert await _a_runner().placed_pods() == []


async def test_a_pod_whose_label_and_name_disagree_is_not_listed(
    a_fake_cluster: FakeCluster,
) -> None:
    """The same consistency check `phase_of` applies, in the same words.

    A pod labelled with one Session and named after another is describable as neither,
    and handing it to a sweep under either name would delete the wrong pod. It is
    labelled, so the selector lets it through -- which is exactly why the check has to
    happen after the list and not only in the query.
    """
    mine, other = new_session_id(), new_session_id()
    pod_name = pod_name_for(mine)
    a_fake_cluster.pods[pod_name] = _a_squatter(
        pod_name, labels={"map.session-id": str(other)}
    )
    a_fake_cluster.status_of[pod_name] = running("impostor")

    assert await _a_runner().placed_pods() == []
    assert await _a_runner().phase_of(pod_name) is PodPhase.ABSENT, (
        "the two entry points disagreed about one pod"
    )


async def test_a_pod_with_no_creation_timestamp_is_not_listed(
    a_fake_cluster: FakeCluster,
) -> None:
    """A pod with no age is a pod no sweep can judge, so it is left out entirely.

    Its age is the guard that lets a sweep delete without trusting any store, so
    inventing one here would be a deletion resting on a value this adapter made up.
    Omitting it keeps the pod alive, which is the safe direction. Seeded rather than
    created, because the adapter's own create path stamps one -- which is the point:
    this can only arrive from a pod something else made.
    """
    session_id = new_session_id()
    pod_name = pod_name_for(session_id)
    unstamped = _a_squatter(pod_name, labels={"map.session-id": str(session_id)})
    unstamped["metadata"].pop("creationTimestamp", None)
    a_fake_cluster.pods[pod_name] = unstamped
    a_fake_cluster.status_of[pod_name] = running("impostor")

    assert await _a_runner().placed_pods() == []
    assert await _a_runner().phase_of(pod_name) is PodPhase.RUNNING, (
        "the pod is still there and still describable one object at a time"
    )


async def test_an_older_pod_is_listed_with_the_age_the_cluster_stamped(
    a_fake_cluster: FakeCluster,
) -> None:
    """A pod's age comes from the API server's stamp and not from this process's clock.

    That is what makes it usable as a guard: a control plane whose clock has drifted, or
    one that restarted a second ago, reads the same age for the same pod.
    """
    session_id = new_session_id()
    pod_name = pod_name_for(session_id)
    an_hour = 3600
    aged = _a_squatter(pod_name, labels={"map.session-id": str(session_id)})
    aged["metadata"]["creationTimestamp"] = rfc3339(an_hour)
    a_fake_cluster.pods[pod_name] = aged
    a_fake_cluster.status_of[pod_name] = running("impostor")

    listed = await _a_runner().placed_pods()

    assert [pod.session_id for pod in listed] == [session_id]
    age_ms = _now_ms() - listed[0].created_at_ms
    assert an_hour * 1000 <= age_ms < (an_hour + 60) * 1000


async def test_an_empty_namespace_lists_nothing(a_fake_cluster: FakeCluster) -> None:
    """Guard the guard: three cases above assert an empty list, and this is what says
    that an empty list is not simply what this method always returns."""
    assert await _a_runner().placed_pods() == []
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.status_of[pod_name] = running("agent-runtime", "session-shim")
    await _a_runner().ensure(pod_name, compiled)

    assert len(await _a_runner().placed_pods()) == 1


async def test_a_removed_pod_stops_being_listed(a_fake_cluster: FakeCluster) -> None:
    """The handback and the listing are the same cluster, which is what a sweep needs.

    A sweep that listed from one view and deleted into another would find the same pod
    on every pass and delete it for ever. Asserted end to end over the real client
    rather than by trusting that both methods name the same namespace field.
    """
    compiled = _compiled()
    pod_name = pod_name_for(compiled.session_id)
    a_fake_cluster.status_of[pod_name] = running("agent-runtime", "session-shim")
    await _a_runner().ensure(pod_name, compiled)
    assert len(await _a_runner().placed_pods()) == 1

    await _a_runner().remove(pod_name)

    assert await _a_runner().placed_pods() == []
    assert a_fake_cluster.pods == {}
    assert a_fake_cluster.secrets == {}


CLUSTER = pytest.mark.skipif(
    os.environ.get("MAP_CLUSTER_TESTS") != "1",
    reason=(
        "creates a real pod in EKS: needs cluster credentials, a pushed Session image "
        "in MAP_SESSION_IMAGE, and up to four minutes. Set MAP_CLUSTER_TESTS=1 to run."
    ),
)


async def _make_the_namespace_and_its_service(namespace: str) -> None:
    """Create the namespace and the headless Service, both idempotently.

    Without the Service a Session pod has no DNS record at all, so the shim has no
    address even once it is ready. Both creates tolerate 409 so a namespace left by a
    previous run is reused rather than fought over.
    """
    async with _core_api() as core:
        try:
            await core.create_namespace(
                body=V1Namespace(metadata=V1ObjectMeta(name=namespace))
            )
        except ApiException as err:
            if err.status != _ALREADY_EXISTS:
                raise
        try:
            await core.create_namespaced_service(
                namespace=namespace,
                body=yaml.safe_load(
                    (MANIFEST.parent / "session-shim-service.yaml").read_text()
                ),
            )
        except ApiException as err:
            if err.status != _ALREADY_EXISTS:
                raise


async def _drop_the_namespace(namespace: str) -> None:
    """Delete the namespace, which takes the Service and anything else in it.

    Issued and not waited on: a namespace spends its last seconds in `Terminating` and
    nothing here needs to watch it finish. A failure is swallowed because this runs in a
    teardown, and a teardown that raises replaces a real test failure with a story about
    cleanup.
    """
    async with _core_api() as core:
        with contextlib.suppress(ApiException):
            await core.delete_namespace(name=namespace)


@pytest.fixture(scope="module")
def a_namespace_of_our_own() -> Iterator[str]:
    """The namespace this file's live tier runs in, created once and deleted at the end.

    Module-scoped, and the scope is the fix for a leak. Created and deleted per test,
    the second test finds the first one's namespace in `Terminating`, cannot re-create
    it, and the whole tier errors -- measured. Created and never deleted, which is what
    this was before, a namespace and a Service accrete for every distinct
    `MAP_NAMESPACE` a run has ever used; the fixture's own docstring called itself the
    thing that reaps what a run leaves behind, which was true of pods and false of the
    two objects it made itself.

    Synchronous, driving `asyncio.run` twice, because a module-scoped async fixture has
    to agree with pytest-asyncio about which event loop it belongs to and this has no
    reason to care: the API client is built inside each call and outlives nothing.
    """
    namespace = os.environ["MAP_NAMESPACE"]
    asyncio.run(_make_the_namespace_and_its_service(namespace))
    try:
        yield namespace
    finally:
        asyncio.run(_drop_the_namespace(namespace))


@pytest.fixture
async def a_swept_namespace(a_namespace_of_our_own: str) -> AsyncIterator[str]:
    """The run's namespace, emptied before this test starts.

    The sweep has no selector, which is why it is confined to a namespace holding only
    this file's pods: narrowed by `map.role=session-pod` it would delete every live
    Session the day this cluster has one. This is the mechanism that reaps what a killed
    run left behind -- the mechanism for the run being killed *now* is the deadline
    patched onto the pod once it is up.
    """
    async with _core_api() as core:
        await core.delete_collection_namespaced_pod(namespace=a_namespace_of_our_own)
    yield a_namespace_of_our_own


@CLUSTER
async def test_a_session_pod_reaches_running_in_the_real_cluster(
    a_swept_namespace: str,
) -> None:
    """A pod in EKS, from the compiled manifest, with no fake anywhere on the path.

    `Placement` is built over `composition.pod_runner_from_environment()`, which reads
    the namespace, the signing key and the manifest path out of the environment -- so
    what is exercised is the wiring a deployed process would use and not a runner
    assembled here.

    The deadline is patched onto the pod the moment it is up. It is the dead-man switch:
    if this process is killed from here on, the API server stops the pod's containers on
    its own. It is set here and never in the adapter, because a real Session runs for
    hours and a deadline in the manifest would kill it.
    """
    runner = pod_runner_from_environment()
    assert runner is not None
    placement = Placement(runner)
    compiled = _compiled(os.environ["MAP_SESSION_IMAGE"])
    pod_name = pod_name_for(compiled.session_id)
    try:
        binding = await placement.place(compiled)
        async with _core_api() as core:
            await core.patch_namespaced_pod(
                name=pod_name,
                namespace=a_swept_namespace,
                body={"spec": {"activeDeadlineSeconds": 600}},
            )
            pod = await core.read_namespaced_pod(
                name=pod_name, namespace=a_swept_namespace
            )
        assert binding.phase is PodPhase.RUNNING
        assert binding.pod_name == pod_name
        assert sorted(status.name for status in pod.status.container_statuses) == [
            "agent-runtime",
            "session-shim",
        ]
        assert all(status.ready for status in pod.status.container_statuses)
        located = await placement.locate(compiled.session_id)
        assert located.phase is PodPhase.RUNNING
        assert located.pod_name == pod_name
    finally:
        await placement.release(compiled.session_id)
    assert (await placement.locate(compiled.session_id)).phase is not PodPhase.RUNNING


@CLUSTER
async def test_release_takes_the_secrets_with_the_pod_in_the_real_cluster(
    a_swept_namespace: str,
) -> None:
    """The three Secrets exist while the Session does and not a moment longer.

    Asserted against the cluster rather than against the owner reference, because the
    owner reference only collects them once the pod object is gone -- after its grace
    period. What this shows is that `remove` does not wait for that: the token and the
    compiled documents stop existing when the Session is released.

    Driven through `_create` rather than through `ensure`, and that is a change forced
    by a fix: `ensure` now deletes what it created when the pod will not come up, so a
    failed `ensure` no longer leaves Secrets to read. `_create` is the half whose
    lifetime is under test here, and it costs a create and a delete rather than an
    image pull. The owner reference is read off the real objects, which is the only
    place it can be graded -- it is set by a patch the API server applies.

    Built from the concrete class rather than through the composition root, because
    `_create` is not on the `PodRunner` port. What the root hands over is graded by the
    other live test and by three tier-1 cases.
    """
    runner = _a_runner(os.environ["MAP_NAMESPACE"])
    compiled = _compiled(_AN_IMAGE)
    pod_name = pod_name_for(compiled.session_id)
    names = [f"{pod_name}-{volume}" for volume in ("compiled", "requirements")]
    names.append(f"{pod_name}-shim-token")
    try:
        async with _core_api() as core:
            await runner._create(core, pod_name, compiled)
            for name in names:
                secret = await core.read_namespaced_secret(
                    name=name, namespace=a_swept_namespace
                )
                assert secret.metadata.owner_references[0].name == pod_name
    finally:
        await runner.remove(pod_name)
    async with _core_api() as core:
        for name in names:
            with pytest.raises(ApiException) as gone:
                await core.read_namespaced_secret(
                    name=name, namespace=a_swept_namespace
                )
            assert gone.value.status == 404
    await runner.remove(pod_name)


@CLUSTER
async def test_a_pod_whose_image_does_not_exist_is_refused_in_the_real_cluster(
    a_swept_namespace: str,
) -> None:
    """The failure that will actually be hit, and the cluster's own words for it.

    The bad reference is derived from the good one rather than given its own variable,
    so the registry host and repository cannot be wrong in a way that makes this pass
    for the wrong reason -- a nonexistent registry fails differently from a nonexistent
    tag.

    Measured at fifteen seconds against minutes for the happy case, because the refusal
    happens as soon as the cluster says `ImagePullBackOff` rather than at the end of the
    wait.
    """
    good = os.environ["MAP_SESSION_IMAGE"]
    repository = good.rsplit("@", 1)[0] if "@" in good else good.rsplit(":", 1)[0]
    runner = pod_runner_from_environment()
    assert runner is not None
    compiled = _compiled(f"{repository}@sha256:{'cd' * 32}")
    pod_name = pod_name_for(compiled.session_id)
    names = [f"{pod_name}-{volume}" for volume in _SECRET_FILES]
    with pytest.raises(PodNotStarted, match="ImagePullBackOff") as refused:
        await runner.ensure(pod_name, compiled)
    assert repository.rsplit("/", 1)[-1] in str(refused.value)
    # No `remove` on this path, deliberately: what is under test is that the refusal
    # itself cleaned up. Before this held, the pod stayed `Pending` for ever and all
    # three Secrets stayed with it -- the shim's bearer for a Session that never started
    # among them -- and nothing anywhere reaped them.
    async with _core_api() as core:
        for name in names:
            with pytest.raises(ApiException) as gone:
                await core.read_namespaced_secret(
                    name=name, namespace=a_swept_namespace
                )
            assert gone.value.status == 404
    # GONE and not ABSENT, measured: deletion is asynchronous, so a read this soon after
    # a delete still finds the object with its deletion timestamp set. Both answers say
    # the same thing to a caller -- there is nothing here to dispatch into -- and which
    # one comes back depends on how far through its grace period the pod is.
    assert await runner.phase_of(pod_name) in (PodPhase.GONE, PodPhase.ABSENT)


# --------------------------------------------------------------------------------------
# The pod is told whether it is continuing a thread or opening one
# --------------------------------------------------------------------------------------


def test_a_first_placement_tells_the_seed_container_it_is_not_resuming() -> None:
    """The common path says so out loud rather than by omitting a variable.

    An absent variable and a false one are the same to a reader of the pod and not to
    the process reading it, which is why the manifest declares the entry and this fills
    it either way. The seed container refuses to start a pod told to resume with
    nothing stored to resume from, so the value it reads decides whether an empty
    bucket is a refusal or the ordinary first placement.
    """
    compiled = _compiled()
    pod = _pod_for(_manifest(), pod_name_for(compiled.session_id), compiled)
    assert _seed_env(pod)["MAP_RESUMING"] == "false"


def test_a_resuming_session_tells_the_seed_container_to_expect_a_rollout() -> None:
    compiled = replace(_compiled(), resuming=True)
    pod = _pod_for(_manifest(), pod_name_for(compiled.session_id), compiled)
    assert _seed_env(pod)["MAP_RESUMING"] == "true"


def test_the_resume_fact_reaches_the_seed_container_and_not_the_shim() -> None:
    """Which container is told is the whole of who acts on it.

    The shim decides between resuming and starting from whether a Rollout was seeded on
    disk, never from a variable -- so a `MAP_RESUMING` reaching the shim would be a
    second answer to a question the file already answers, free to disagree with it. The
    two failure modes that disagreement produces are a shim that resumes from nothing
    and a shim that starts fresh over a seeded record, and the second is the silent one.
    """
    compiled = replace(_compiled(), resuming=True)
    pod = _pod_for(_manifest(), pod_name_for(compiled.session_id), compiled)
    shim = [c for c in pod["spec"]["containers"] if c["name"] == _SHIM_CONTAINER][0]
    assert "MAP_RESUMING" not in {entry["name"] for entry in shim["env"]}
    assert set(_SEED_ENV) == {"MAP_RESUMING"}


def test_a_manifest_whose_seed_container_declares_no_resume_variable_is_refused() -> (
    None
):
    """A pod that started here would resume-as-fresh, silently, for every Session.

    The seed container reads this before it asks for anything. Unset, it cannot tell a
    Session that has completed Turns from one that has not, and the safe reading of an
    unset variable is the dangerous one: it would treat every resume as a first
    placement, open a new thread over a stored Rollout, and report success.
    """
    manifest = _manifest()
    compiled = _compiled()
    seed = [
        c for c in manifest["spec"]["initContainers"] if c["name"] == "seed-rollout"
    ][0]
    seed["env"] = [e for e in seed["env"] if e["name"] != "MAP_RESUMING"]
    with pytest.raises(PodNotStarted, match="MAP_RESUMING"):
        _pod_for(manifest, pod_name_for(compiled.session_id), compiled)


def test_a_manifest_with_no_seed_container_at_all_is_refused() -> None:
    """Removing the container is how the refusal above would be routed around.

    Without this, a manifest that simply dropped `seed-rollout` would compile cleanly
    and place a pod that opens a fresh thread for a Session holding a Rollout -- the
    exact outcome the variable above exists to prevent, reached by deleting its reader
    rather than its value.
    """
    manifest = _manifest()
    compiled = _compiled()
    manifest["spec"]["initContainers"] = [
        c for c in manifest["spec"]["initContainers"] if c["name"] != "seed-rollout"
    ]
    with pytest.raises(PodNotStarted, match="seed-rollout"):
        _pod_for(manifest, pod_name_for(compiled.session_id), compiled)


def _seed_env(pod: dict[str, Any]) -> dict[str, str]:
    seed = [c for c in pod["spec"]["initContainers"] if c["name"] == "seed-rollout"][0]
    return {entry["name"]: entry["value"] for entry in seed["env"]}
