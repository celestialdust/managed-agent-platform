"""What the Session pod can hold, graded on the two artifacts that decide it.

A pod holds what its manifest gives it and what its compiled configuration names. So the
questions "can a tool credential reach this pod" and "can anything durable live only
inside it" are answered by reading those two artifacts, and each assertion below fails
when one of them changes.

**This file does not grade the live-pod inspection scenarios.** Those inspect a pod that
has run a Turn against a credentialed server, and this tree cannot produce one:
`composition.build()` wires a transport with no pod runner behind it and refuses every
Turn, nothing in `src/` calls the compiler that renders a Session's documents, and
neither attested conformance server reads a credential header. The live inspection is
owed by a later slice. Saying so here is the point -- a green run of this file is not
evidence for those scenarios, and a docstring is where a reader finds that out.

Every absence asserted below has a presence beside it, because an absence is also
satisfied by an empty document: a manifest with no containers sources no variable from a
secret and mounts no durable volume. `test_the_manifest_is_the_real_pod` is that
presence, and each sweep carries its own planted control.

The sweeps name what the manifest **may** carry rather than what it may not, wherever
that is expressible. An earlier version asked whether an env entry had a `valueFrom`,
which is an allowlist of the one route it had thought of: `envFrom: [{secretRef: ...}]`
injects every key of a Secret as a variable and was invisible to it, as were
`automountServiceAccountToken` and `hostNetwork` -- three ways to a credential in this
pod that left all four assertions green. Naming the permitted shape instead means an
unrecognised route fails here and the failure says which key it did not expect.

Two names below belong to the manifest and are restated rather than imported, because
YAML has nothing to import: the three container names and the six volume names. Renaming
one makes this fail rather than quietly pass, which is the only direction a coupling
like this may fail in.
"""

import tomllib
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

import yaml

from managed_agent.control.pod_config.compiler import (
    GATEWAY_SERVER_ID,
    compile_session_config,
)
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.registration.definition import (
    AgentDefinition,
    SkillsRevision,
)
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER_NAME,
    verify_session_token,
)

MANIFEST: Final[Path] = (
    Path(__file__).resolve().parents[2] / "deploy" / "k8s" / "session-pod.yaml"
)
CONTAINERS: Final[tuple[str, ...]] = (
    "seed-runtime-home",
    "restore-working-lane",
    "seed-rollout",
    "agent-runtime",
    "session-shim",
)
DISCARDED_WITH_THE_POD: Final[frozenset[str]] = frozenset(
    {"codex-home", "control", "workspace", "scratch"}
)
PLATFORM_DOCUMENTS: Final[frozenset[str]] = frozenset(
    {"compiled", "requirements", "shim-token"}
)
GATEWAY_URL: Final[str] = "https://tool-gateway.map.internal/mcp"

# Where a Session pod reaches the Model Gateway. The `/v1` is load-bearing at both ends:
# the Agent Runtime POSTs `{base_url}/responses`, and the Gateway's router mounts
# `POST /v1/responses`.
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"

# The Gateway's signing key and the token's deadline are the caller's to supply, and
# this file is the caller. Literals rather than a clock, so nothing here can expire
# between two assertions.
SESSION_TOKEN_KEY: Final[bytes] = b"a signing key that is thirty-two"
SESSION_TOKEN_EXPIRY: Final[int] = 4102444800

_POD: Final[dict[str, Any]] = yaml.safe_load(MANIFEST.read_text())

# The definition a Session pins, for the one field the compiler reads off it: the model.
# The provider is not here because it is not the definition's to name -- every model
# call leaves a Session pod through the Model Gateway.
A_DEFINITION = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)


def _containers_of(pod: dict[str, Any]) -> list[dict[str, Any]]:
    """Both stages, because an init container is a place to put a variable too.

    Takes the pod rather than reading the module-level one, so the sweeps above can be
    driven over a synthetic manifest whose answer is known. A structural check nobody
    has watched fail is a check nobody has verified.
    """
    spec = pod["spec"]
    return [*spec.get("initContainers", ()), *spec["containers"]]


def _every_container() -> list[dict[str, Any]]:
    return _containers_of(_POD)


LITERAL_ENV_KEYS: Final[frozenset[str]] = frozenset({"name", "value"})
"""Everything an env entry may carry. A key outside this set is a value from somewhere.

Inverted rather than enumerated. The previous version asked `"valueFrom" in entry`,
which is an allowlist of the one source it had thought of -- `envFrom` was invisible to
it, and so would be whatever Kubernetes adds next. Naming what an entry may hold instead
means an unrecognised source fails here and the failure names the key.
"""

HOST_NAMESPACES: Final[tuple[str, ...]] = ("hostNetwork", "hostPID", "hostIPC")
"""Pod-spec fields that put this pod on the node's namespaces.

`hostNetwork` is the one that reaches a credential: it puts the pod on the node's
network, where the instance metadata service answers with the node role's credentials.
The other two are the same class of escape and cost nothing to name beside it.
"""

SERVICE_ACCOUNT_FIELDS: Final[tuple[str, ...]] = (
    "serviceAccountName",
    "serviceAccount",
)
"""Naming a service account here is how the pod gets one that can read something.

The default account is what `automountServiceAccountToken: false` refuses a token for.
Naming a different one and leaving the token off is harmless; naming one *and* letting
the token in is the escalation, and the two fields are checked together below.
"""


def _unreadable_env_keys(pod: dict[str, Any]) -> dict[str, list[str]]:
    """Every env entry carrying a key that is not a literal, by container.entry."""
    return {
        f"{container['name']}.{entry.get('name', '?')}": sorted(
            set(entry) - LITERAL_ENV_KEYS
        )
        for container in _containers_of(pod)
        for entry in container.get("env", ())
        if set(entry) - LITERAL_ENV_KEYS
    }


def _env_from(pod: dict[str, Any]) -> dict[str, Any]:
    """Every `envFrom` block declared, by container.

    Its own check because it is the cheapest route to a credential in a pod and the
    least visible: one `secretRef` injects **every** key of a Secret as an environment
    variable, under names that appear nowhere in this manifest.
    """
    return {
        container["name"]: container["envFrom"]
        for container in _containers_of(pod)
        if container.get("envFrom")
    }


def _host_namespaces_joined(pod: dict[str, Any]) -> list[str]:
    """Which node namespaces this pod spec asks to share."""
    return [field for field in HOST_NAMESPACES if pod["spec"].get(field)]


def _backings(volume: dict[str, Any]) -> set[str]:
    """Which volume source a declared volume names, `name` aside."""
    return {key for key in volume if key != "name"}


def _compiled() -> tuple[dict[str, Any], dict[str, Any]]:
    """The two documents one Session's pod is started with, parsed."""
    compiled = compile_session_config(
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
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        definition=A_DEFINITION,
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="fixture",
            runtime_image="registry.map.internal/session@sha256:" + "a" * 64,
            denied_paths=(),
        ),
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )
    return (
        tomllib.loads(compiled.config_toml),
        tomllib.loads(compiled.requirements_toml),
    )


def _keys_anywhere(node: object) -> set[str]:
    """Every key name at any depth.

    A credential-bearing key nests as easily as it sits at the top, so a root-level
    check would pass a document that carries one inside the server's own table.
    """
    if isinstance(node, dict):
        found = set(node)
        for value in node.values():
            found |= _keys_anywhere(value)
        return found
    if isinstance(node, list):
        nested: set[str] = set()
        for item in node:
            nested |= _keys_anywhere(item)
        return nested
    return set()


def test_the_manifest_is_the_real_pod() -> None:
    """The presence every absence below needs. An empty document satisfies them all,
    and would look exactly like a pod that holds nothing."""
    assert _POD["kind"] == "Pod"
    assert [c["name"] for c in _every_container()] == list(CONTAINERS)
    assert {v["name"] for v in _POD["spec"]["volumes"]} == (
        DISCARDED_WITH_THE_POD | PLATFORM_DOCUMENTS
    )


def test_every_variable_in_the_pod_is_a_literal_this_manifest_states() -> None:
    """A tool credential reaching this pod as a variable arrives one of two ways, and
    the sweep reads both: an entry sourcing a value, and an `envFrom` block sourcing a
    whole Secret at once.

    The entry check names what an entry may hold rather than what it may not. Asking
    `"valueFrom" in entry` passes every source nobody listed, and there is one --
    `envFrom` -- which injects every key of a Secret under names appearing nowhere here.
    """
    assert _unreadable_env_keys(_POD) == {}, "an env entry sources its value"
    assert _env_from(_POD) == {}, (
        "a container declares envFrom, which injects every key of whatever it names as "
        "an environment variable -- the cheapest route to a credential in this pod"
    )
    assert [
        entry["name"]
        for container in _every_container()
        for entry in container.get("env", ())
    ] == [
        # Whether this pod continues a thread or opens one, on the init container that
        # seeds a stored Rollout. Two spellings and no third, neither of them a secret,
        # and it is in this list rather than exempted from it for the reason PYTHONPATH
        # is below: the list IS the assertion.
        "MAP_RESUMING",
        "CODEX_HOME",
        # Where a package the agent installed at run time is imported from. A path and
        # not a credential, and it is in this list rather than exempted from it: the
        # list is the assertion. A check that skipped names it judged harmless would be
        # a check whose coverage shrank every time somebody added a variable.
        "PYTHONPATH",
        "RUST_LOG",
        "MAP_SESSION_ID",
        "MAP_MODEL",
        "MAP_MODEL_PROVIDER",
    ]
    # Spelled out rather than checked for shape, and that is what makes adding one a
    # decision. `RUST_LOG` was added on 2026-08-23 to make the runtime's own diagnosis
    # of a skill it could not load reach the container log; a Session had a skill
    # delivered, readable, and absent from its catalogue, with no line anywhere saying
    # why. It is a filter string and not a value the pod may read anything with -- but
    # a broad one on the runtime container would put prompt and answer text into node
    # logs, so what it is set TO is the part worth reading, and
    # `tests/deploy/test_session_pod_runs_a_shim.py` is where that is graded.


def test_the_env_sweep_reads_both_routes_and_not_only_the_one_it_was_written_for() -> (
    None
):
    """The planted controls. Each is a real Kubernetes spelling, and each must be
    reported by the sweep that claims to cover it."""
    sourced = {
        "spec": {
            "containers": [
                {
                    "name": "c",
                    "env": [
                        {"name": "X", "valueFrom": {"secretKeyRef": {"name": "s"}}}
                    ],
                }
            ]
        }
    }
    injected = {
        "spec": {
            "containers": [{"name": "c", "envFrom": [{"secretRef": {"name": "creds"}}]}]
        }
    }
    literal = {
        "spec": {"containers": [{"name": "c", "env": [{"name": "X", "value": "y"}]}]}
    }

    assert _unreadable_env_keys(sourced) == {"c.X": ["valueFrom"]}
    assert _env_from(injected) == {"c": [{"secretRef": {"name": "creds"}}]}
    assert _unreadable_env_keys(literal) == {} and _env_from(literal) == {}, (
        "a literal entry is reported, which is a check somebody deletes"
    )


def test_the_pod_is_refused_the_service_account_token() -> None:
    """The most load-bearing line in the manifest, and nothing read it.

    Kubernetes projects a ServiceAccount token into every container by default, and on
    this cluster that token is `AssumeRoleWithWebIdentity` material. With it the
    confined process reaches Secrets Manager itself and needs no help from the Gateway
    at all -- so `automountServiceAccountToken: false` is what the rest of this file is
    about, and flipping it or deleting it left every assertion here green.

    The key must be present and exactly `False`. Absent is not the same: the Kubernetes
    default is to mount, so a deleted line reads as a pod that holds no credential and
    is a pod that holds one.
    """
    spec = _POD["spec"]

    assert "automountServiceAccountToken" in spec, (
        "the line refusing the service-account token is gone; Kubernetes mounts one by "
        "default, so its absence is not neutrality"
    )
    assert spec["automountServiceAccountToken"] is False, (
        f"the token is mounted: automountServiceAccountToken is "
        f"{spec['automountServiceAccountToken']!r}"
    )
    named = [f for f in SERVICE_ACCOUNT_FIELDS if f in spec]
    assert named == [], (
        f"{named} names a service account for this pod. Harmless only while no "
        "token is mounted, and the two lines are one edit apart -- so neither "
        "stands alone."
    )


def test_the_pod_is_on_no_namespace_of_the_node_s() -> None:
    """`hostNetwork` puts this pod on the node's network, where the instance metadata
    service answers with the node role's credentials -- a credential reached without
    any Secret being mounted anywhere, and invisible to every other check here."""
    assert _host_namespaces_joined(_POD) == []
    assert _host_namespaces_joined(
        {"spec": {"hostNetwork": True, "containers": []}}
    ) == ["hostNetwork"], (
        "the sweep above cannot see a pod that joined the node's network"
    )


def test_the_secret_volumes_are_a_closed_set_of_platform_documents() -> None:
    """The other route a credential could take in. Each of these three is written by
    the control plane for this Session; a fourth would be something else.

    The `emptyDir` half is the durability question read off the same artifact: a volume
    backed by nothing outside the pod goes away with it, so nothing durable lives only
    inside -- and a claim on persistent storage would appear here as a third backing.
    """
    by_backing: dict[str, set[str]] = {}
    for volume in _POD["spec"]["volumes"]:
        for backing in _backings(volume):
            by_backing.setdefault(backing, set()).add(volume["name"])

    assert by_backing == {
        "emptyDir": set(DISCARDED_WITH_THE_POD),
        "secret": set(PLATFORM_DOCUMENTS),
    }


def test_the_compiled_documents_name_a_url_and_one_identity_and_no_credential() -> None:
    """The second artifact, and the distinction it turns on.

    Two different things could be called a credential here and only one of them is
    forbidden. The *tool credential* -- an upstream's secret, fetched under
    `map/tool-credential/<tenant>/<ref>` -- is attached on the outbound call at the
    Gateway and appears in neither document, which is what the rest of this file
    grades. The Session's *own identity assertion* to the Gateway must appear here,
    because the pod is the caller: the Gateway refuses every request that carries no
    `x-map-session`, so a document naming only a destination describes a pod that
    cannot use a tool at all. An earlier version of this case asserted that absence as
    a property; `plan/MAP-16.md:80-84` had already recorded it as a missing
    prerequisite, and this is the case reading the way that step file did.

    What makes the identity safe to hold is what is checked below: one header and no
    other, whose value is a signature over this Session and its tenant and nothing
    else. The key that made it is on the Tool Gateway's Deployment and on no Session
    pod -- `test_every_variable_in_the_pod_is_a_literal_this_manifest_states` above is
    what keeps it that way, and it fails on any `secretKeyRef` added to this manifest.
    """
    config, requirements = _compiled()

    assert _keys_anywhere(config["mcp_servers"]) == {
        GATEWAY_SERVER_ID,
        "url",
        "required",
        # A policy word, not a secret: it decides whether the runtime asks before
        # calling a tool, and its only legal values are four fixed strings. It is listed
        # here because this set is closed and adding a key must be a decision somebody
        # takes with this case in front of them -- which is the whole point of asserting
        # equality rather than membership.
        "default_tools_approval_mode",
        "http_headers",
        SESSION_TOKEN_HEADER_NAME,
    }
    header = config["mcp_servers"][GATEWAY_SERVER_ID]["http_headers"]
    assert set(header) == {SESSION_TOKEN_HEADER_NAME}
    session, tenant, expiry, signature = header[SESSION_TOKEN_HEADER_NAME].split(".")
    assert UUID(session) and UUID(tenant) and int(expiry) > 0
    assert len(signature) == 64
    assert _keys_anywhere(requirements["mcp_servers"]) == {
        GATEWAY_SERVER_ID,
        "identity",
        "url",
    }


def test_the_identity_in_the_document_is_a_signature_and_not_the_key_that_made_it() -> (
    None
):
    """The property that makes the header safe to hold, asserted rather than argued.

    The value verifies under the Gateway's key, which is what admits the pod. Nothing
    in either document, and nothing in the pod's environment, lets the pod produce a
    second one: the key is on the Tool Gateway's Deployment. So the manifest sweeps
    above and this case together say a pod holds an assertion about itself and no
    means of forging another.
    """
    config, _ = _compiled()
    token = config["mcp_servers"][GATEWAY_SERVER_ID]["http_headers"][
        SESSION_TOKEN_HEADER_NAME
    ]

    context = verify_session_token(token, SESSION_TOKEN_KEY, SESSION_TOKEN_EXPIRY - 1)
    assert str(context.session_id) == token.split(".")[0]
    assert str(context.tenant_id) == token.split(".")[1]
    assert SESSION_TOKEN_KEY not in config["mcp_servers"][GATEWAY_SERVER_ID].values()
    assert str(SESSION_TOKEN_KEY) not in str(config)
