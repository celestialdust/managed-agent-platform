"""Put one platform workload in the cluster, running the image built from this commit.

Run it against the cluster that runs the platform, from anywhere:

    uv run python deploy/platform.py control-plane

It is `deploy/bootstrap.py`'s sibling and stops where that one stops: bootstrap puts the
objects a Session pod needs in place and "deploys no workload". This deploys workloads,
and it creates no Secret either -- for the same reason, which is that the values are not
in this repository and must not be.

Nine things it does that `kubectl apply -f` cannot, and each is a way this job
goes wrong silently rather than a convenience:

* **A manifest can lose a variable and stay valid.** A workload whose environment is
  incomplete starts, passes its probes, and does less than it is for. That is not
  hypothetical: a control plane ran in `map-dev` accepting Sessions and placing no pod
  for any of them, because two variables were absent and the process cannot tell "not
  configured to place" from "not a placer". `undeclared_variables` is where that is a
  fact somebody wrote down, and it refuses before anything is applied.

* **A Deployment can mount a ConfigMap nobody created**, which leaves a pod at
  ContainerCreating -- a state a pod sits in rather than a crash. `companions` and
  `generated_config_maps` put the identities, the permissions and the generated
  documents in place first, in that order.

* **The manifests carry a placeholder image.** Sixty-four zeros is digest-shaped, so it
  parses everywhere and resolves nowhere; a bare apply starts nothing and says
  ImagePullBackOff. This substitutes the digest the registry holds for *this commit's*
  tag, so what runs is what is checked out -- and if nothing has pushed this commit it
  refuses and says which command to run.

* **`map-dev` holds no Secrets.** An absent `secretKeyRef` is
  `CreateContainerConfigError` -- a state a pod sits in, not a crash -- so this checks
  first and applies nothing when a Secret or a key is missing.

* **The database credential can be right and stop working.** The RDS master password
  rotates every seven days (measured: RotationEnabled true, AutomaticallyAfterDays 7),
  and the control-plane role cannot read the rotated value, so a DSN built from the
  master credential works and then fails authentication days after the deploy. This
  refuses that DSN unless `--allow-rotating-credential` says it was chosen.

* **Finishing is not the same as working.** `kubectl rollout status` exits 0 for a
  Deployment scaled to zero. `deployment_shortfall` is what refuses that.

* **A Service can apply cleanly and route to nothing.** A selector matching no pod, and
  a named `targetPort` no container declares, are both valid YAML the API server
  accepts; neither shows up in `kubectl rollout status` or in `deployment_shortfall`,
  because both of those are about the Deployment. What is left is an address that
  resolves and then connects to nothing. That has happened in this project:
  `session-shim-service.yaml` sat in the repository applied by nothing, and a Session
  pod at `2/2 Running` with a healthy shim failed every Turn as unreachable, with every
  probe in the chain reporting fine. `declared_services` reads the Service names out of
  the manifest and `unrouted_service` refuses when one of them has no ready endpoint.

* **A workload can name a vault entry that is not in the account.** Nothing fetches one
  until a request needs it, so the pod starts, the probes pass, the rollout completes --
  and every request that needs that entry fails at AWS, hours later, with an error that
  names a secret path rather than a manifest line. `declared_vault_entries` reads those
  names out of the manifest itself and refuses before applying anything.

* **A NetworkPolicy can be accepted, listed, and enforced by nothing.** This cluster's
  CNI ships the agent that would enforce one and runs it with
  `--enable-network-policy=false`, so `kubectl apply` succeeds, `kubectl get netpol`
  shows the object, and every packet it names goes through. That is worse than having no
  policy: it is a security control that exists as text, and nothing about the cluster
  says which it is. `inert_network_policies` is what refuses -- it costs nothing while
  the namespace holds no policies, and once it holds one it will not let a deploy
  through while the flag is off.

**On reading the Secret.** Checking that a key exists means reading it -- `kubectl get
secret -o jsonpath` returns the value -- so the DSN passes through this process either
way. Only its username is inspected, nothing derived from it is printed or logged, and
the refusal names the Secret, the key and the master username. That is the whole of what
this file does with a credential.

**On the one import from the application package.** `routing_table_from_json` is the
parser the Model Gateway itself uses. Reading `entries[].credential_name` here with
`json.loads` would be a second definition of what a routing document is, free to keep
passing after the first one changed -- and this check's whole value is that the names it
asks the account about are the names the service will ask the vault for. Nothing else
in this file knows the package exists.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import yaml

from managed_agent.gateway.model.router import routing_table_from_json

DIGEST_PLACEHOLDER: Final = "sha256:" + "0" * 64
"""The un-substituted image reference every platform manifest carries.

Digest-shaped so that nothing downstream needs a special case for it, and sixty-four
zeros so it resolves to no image anywhere -- an unsubstituted manifest fails at the pull
rather than starting something. The same value and the same argument as
`deploy/k8s/session-pod.yaml`'s.
"""

ACCOUNT_PLACEHOLDER: Final = "0" * 12
"""The un-substituted AWS account id every committed manifest carries.

Twelve digits, so an ARN built around it parses everywhere an ARN is parsed, and all
zeros so it names no account that exists -- the same argument as `DIGEST_PLACEHOLDER`
above, one field over. An IRSA annotation left at this value gets the pod a role that
cannot be assumed, and a bucket name left at it gets an S3 call that 404s, both of which
are loud. A *real* account id written into a committed manifest is the failure worth
preventing: it is silent, it works on the machine that wrote it, and it is published.

The value the deploy substitutes in comes from `aws sts get-caller-identity`, which is
where `deploy/docker/push-platform-image.sh` already gets the registry host and where
`deploy/terraform/account.tf` already gets every ARN. One authority, asked three times,
rather than one literal copied into ninety places.
"""

_K8S: Final = Path("deploy/k8s")
_IDENTITIES: Final = _K8S / "cluster-bootstrap.yaml"
_PUSH_SCRIPT: Final = Path("deploy/docker/push-platform-image.sh")

_ROLLOUT_TIMEOUT: Final = "300s"
_JOB_TIMEOUT: Final = "600s"

_ENDPOINTS_TIMEOUT_S: Final = 30.0
_ENDPOINTS_POLL_S: Final = 1.0
"""How long to keep asking whether a Service has an endpoint, and how often.

Not zero, and the reason is a race rather than slowness. The EndpointSlice controller
and the Deployment's status are written by two controllers reacting to the same pod
Ready transition, in no fixed order, so one read taken the instant `kubectl rollout
status` returns can see no slices and mean nothing by it. A refusal here stops a deploy,
so it has to be about the Service and not about which controller ran first.
"""

_SESSION_POD: Final = _K8S / "session-pod.yaml"
POD_MANIFEST_CONFIG_MAP: Final = "map-session-pod"
"""The ConfigMap the control plane mounts the Session-pod manifest from.

Generated from `deploy/k8s/session-pod.yaml` at apply time rather than checked in as a
second document, so there is one description of a Session pod and not two free to
diverge. `--from-file=<path>` keys the entry by the file's basename, which is what makes
`MAP_POD_MANIFEST` a path ending in that name.
"""

_CNI_NAMESPACE: Final = "kube-system"
_CNI_DAEMONSET: Final = "aws-node"
_NODEAGENT_CONTAINER: Final = "aws-eks-nodeagent"
_NETWORK_POLICY_FLAG: Final = "--enable-network-policy"
"""Where the switch that makes a NetworkPolicy real is.

The Amazon VPC CNI enforces policies from a second container in its own DaemonSet, and
that container takes the decision as a command-line argument. So the DaemonSet is the
only place that answers "is a policy in this cluster a filter" -- the NetworkPolicy
objects themselves look identical either way, and so does
`NETWORK_POLICY_ENFORCING_MODE`, which says what happens to a pod's traffic while its
policies are being programmed rather than whether they are programmed at all.
"""

_TRUE_SPELLINGS: Final = frozenset({"1", "t", "T", "TRUE", "true", "True"})
"""Every value the flag above can carry that means enforcement is on.

Six, not one, and the list is the Go standard library's rather than a guess: the agent
is a Go program, a boolean flag's value is read with `strconv.ParseBool`, and that
function takes exactly these for true and `0 f F FALSE false False` for false. Anything
else makes the program exit at start-up, which is a nodeagent in CrashLoopBackOff and
no enforcement either -- so a value outside this set is read as off rather than as a
surprise.

This project has paid twice for the other approach, matching one spelling of a flag and
believing the flag was covered; `docs/lessons.md` carries both entries, and
`tests/test_guards_read_values_not_spellings.py` is the guard that keeps a test from
repeating it. Reading the value rather than the argument is the same rule applied to
production code, where no guard is watching.
"""


@dataclass(frozen=True, slots=True)
class Workload:
    """One platform Deployment, and everything that must exist before it can start.

    `secrets` is every (Secret, key) pair the manifest reads through a `secretKeyRef`,
    written out rather than parsed back out of the manifest: a manifest that misspelled
    a key would otherwise be checked against its own mistake.

    `database_secret` is the one of those pairs whose value is a PostgreSQL DSN, named
    rather than taken as `secrets[0]`: the rotating-credential check below reads it, and
    a positional convention is exactly the implicit coupling that stops being true the
    first time somebody reorders a tuple.

    `schema_job` is the migration this workload's schema needs, or None. One schema
    means one runner, so only the workload that owns the schema names it; a second
    workload reading the same tables names none and is applied after this one.

    `vault_variables` and `routing_table_key` say where in the manifest this workload
    writes down the names of AWS Secrets Manager entries it will fetch at run time.
    They are *locations*, never the names themselves: a name written here would be a
    second copy of a value the manifest already carries, and the copy the account is
    asked about would be free to stop being the one the pod reads. `database_secret`
    is None for a workload that opens no database -- which is also what keeps it out of
    the connection-ceiling arithmetic the manifests do against one `max_connections`.
    """

    component: str
    manifest: Path
    repository: str
    secrets: tuple[tuple[str, str], ...]
    database_secret: tuple[str, str] | None = None
    schema_job: Path | None = None
    vault_variables: tuple[str, ...] = ()
    routing_table_key: str | None = None
    companions: tuple[Path, ...] = ()
    """Manifests applied before this workload's own, in order.

    A Deployment mounting a ConfigMap nothing created leaves a pod at
    ContainerCreating, which is a state a pod sits in rather than a crash -- the same
    failure `absent_secrets` exists to prevent one layer up.
    """

    generated_config_maps: tuple[tuple[str, Path], ...] = ()
    """(name, source file) for each ConfigMap generated from a file this repo holds.

    Generated at apply time rather than committed as a second YAML document, so the
    file an operator reads and the bytes a pod mounts are one thing. Applied before this
    workload's own manifest, because a Deployment mounting a ConfigMap nothing created
    sits at ContainerCreating.

    A field rather than a branch in `main`, so a second workload that mounts a generated
    document is an entry in the table below like everything else.
    """

    required_variables: tuple[str, ...] = ()
    """Environment variables this workload's manifest MUST declare.

    Written here rather than parsed back out of the manifest for the reason `secrets`
    is: a manifest that lost a variable would otherwise be checked against its own
    omission. The set for the control plane is the placer's, and it is what makes
    "this workload is supposed to place pods" a fact somebody wrote down. The process
    cannot know it about itself -- with no manifest named it is simply not a placer,
    which is a legitimate configuration for something else -- and that is how a
    control plane came to accept Sessions and place none with nothing noticing.
    """


CONTROL_PLANE: Final = Workload(
    component="control-plane",
    manifest=_K8S / "control-plane.yaml",
    repository="map/control-plane",
    secrets=(
        ("map-control-plane", "database-url"),
        ("map-control-plane", "shim-token-key"),
        # The Tool Gateway's Secret, read by this workload too and on purpose: this
        # process signs the Session token that one verifies, so they name one key.
        ("map-tool-gateway", "session-token-key"),
    ),
    database_secret=("map-control-plane", "database-url"),
    schema_job=_K8S / "schema-migration.yaml",
    companions=(
        _K8S / "cluster-bootstrap.yaml",
        # The headless Service every Session pod answers on, and it belongs to this
        # workload because this workload is the only thing that dials it: the control
        # plane addresses a shim at `<pod>.map-session.<ns>.svc.cluster.local`, which
        # resolves to nothing at all unless the Service exists.
        #
        # It was in the repository and applied by nothing until 2026-08-23, and the
        # failure was measured that day: a Session pod reached `2/2 Running` with its
        # shim answering `/session/ready` 204 to the kubelet, and the Turn still failed
        # `the shim for session ... could not be reached`. A pod that is genuinely up,
        # genuinely healthy, and genuinely unaddressable is the worst shape this could
        # have taken -- every probe says the pod is fine, because it is.
        _K8S / "session-shim-service.yaml",
        # What may reach what inside this namespace. A companion of this workload
        # because it is the first one applied, so the exceptions are in place before
        # any pod rolls -- and re-asserted every release rather than at bootstrap,
        # because the set names the pods of all three workloads and a policy that
        # stopped matching one of them would otherwise stay wrong until somebody
        # rebuilt the cluster.
        #
        # This file was applied by nothing until 2026-08-26, and that is the third time
        # this class has cost a deploy. The first two announced themselves -- a pod at
        # ContainerCreating, a Deployment at 0/2. This one could not: absent, the
        # namespace is default-allow, every manifest still reads as a boundary, and
        # nothing anywhere is unhealthy. `test_every_manifest_in_this_tree_is_applied_
        # by_something` is the guard, and it runs offline.
        _K8S / "network-policies.yaml",
    ),
    generated_config_maps=((POD_MANIFEST_CONFIG_MAP, _SESSION_POD),),
    required_variables=(
        "MAP_NAMESPACE",
        "MAP_POD_MANIFEST",
        "MAP_SHIM_TOKEN_KEY",
        "MAP_SESSION_TOKEN_KEY",
        "MAP_SESSION_TOKEN_LIFETIME_S",
        "MAP_TOOL_GATEWAY_URL",
        "MAP_MODEL_GATEWAY_URL",
        # Absent, the process builds no S3 client and POST /v1/files answers 500 for
        # the life of the Deployment -- which is what it did until the role was granted
        # s3:GetObject and s3:PutObject. Listed here so the applier refuses rather than
        # applying a control plane whose upload endpoint is dead.
        "MAP_OBJECT_BUCKET",
        # Absent, nothing answers an error at all: a completed Turn ships no Rollout and
        # every later resume refuses for want of one. That is the shape this list is for
        # -- a variable whose absence is invisible from outside the process, which is
        # exactly the kind an operator cannot be asked to remember.
        "MAP_ROLLOUT_BUCKET",
        # Unlike the two above, absence here is loud: `build_app` reads it with
        # `os.environ[...]` and the process fails to start. Listed anyway, because the
        # applier refusing before it rolls out is a message naming the variable, and a
        # CrashLoopBackOff is a message naming a KeyError in a traceback -- and the
        # second one arrives after the old ReplicaSet has already been scaled down.
        "MAP_SWEEP_INTERVAL_S",
    ),
)

TOOL_GATEWAY: Final = Workload(
    component="tool-gateway",
    manifest=_K8S / "tool-gateway.yaml",
    repository="map/tool-gateway",
    # Two Secrets, and they are not both this service's. The database URL is the control
    # plane's Secret because it is the same relational store -- one credential for one
    # database, rather than two things to rotate that can reach two databases. The
    # signing key is this service's alone: nothing else verifies a Session token.
    secrets=(
        ("map-control-plane", "database-url"),
        ("map-tool-gateway", "session-token-key"),
    ),
    database_secret=("map-control-plane", "database-url"),
    # None. One schema, one runner, and the control plane's Job is it. This service
    # reads tables that Job's revisions create, so it is applied after that workload --
    # an order a person keeps, not a dependency this file models, because modelling it
    # would mean this file deciding when somebody else's Deployment is healthy enough.
    schema_job=None,
    # Required, because absent it this service fails a tool call AFTER the upstream has
    # already answered -- the agent is told the tool broke, and answers from memory
    # instead. That is worse than a refused deploy in every way: it is silent, it looks
    # like a flaky upstream, and the Evidence a reviewer would need to see it is exactly
    # what did not get written.
    required_variables=("MAP_OBJECT_BUCKET",),
)

MODEL_GATEWAY: Final = Workload(
    component="model-gateway",
    manifest=_K8S / "model-gateway.yaml",
    repository="map/model-gateway",
    # One Kubernetes Secret, and it is not this service's own. Every *provider*
    # credential is still fetched from the vault per TTL under this pod's IRSA identity
    # and never written down -- that is the point of this service. The entry below is a
    # different kind of value: the symmetric key that signs a Session token, shared with
    # the control plane that mints one and the Tool Gateway that verifies one, and
    # already an environment variable in both. Reading it from a fourth place would not
    # make it less exposed, and the attempt to do so is what produced a verifier for a
    # token layout nothing minted (ADR-023).
    secrets=(("map-tool-gateway", "session-token-key"),),
    # None. `managed_agent.composition:model_gateway_app` calls
    # `create_model_gateway_app` directly and never `build`, so this process opens no
    # SQLAlchemy engine and its replica count costs the database nothing.
    database_secret=None,
    schema_job=None,
    routing_table_key="routing.json",
)

WORKLOADS: Final[tuple[Workload, ...]] = (CONTROL_PLANE, TOOL_GATEWAY, MODEL_GATEWAY)
"""Every workload this can apply. A new one is a new entry, never a branch below."""


def namespace(root: Path) -> str:
    """The namespace, read out of the file that decides it.

    Not a constant here. `deploy/k8s/cluster-bootstrap.yaml`'s header explains that the
    name was fixed by three IAM trust policies before any manifest named one, and
    `deploy/bootstrap.py` cites that file for its own constant. A third copy would be a
    third answer free to disagree with the other two.
    """
    documents = yaml.safe_load_all((root / _IDENTITIES).read_text())
    names = [d["metadata"]["name"] for d in documents if d and d["kind"] == "Namespace"]
    if len(names) != 1:
        raise RuntimeError(
            f"{_IDENTITIES} declares {len(names)} namespaces; expected 1"
        )
    return str(names[0])


def _kubectl(
    root: Path, *argv: str, stdin: str | None = None, in_namespace: str | None = None
) -> str:
    """One `kubectl` call, in the platform's namespace unless another is named.

    `in_namespace` exists for the one read that is not about this platform's own
    objects: the CNI's DaemonSet lives in `kube-system`, and the alternative to a
    parameter here is a second runner that knows how to call kubectl -- which is a
    second place to get the error handling wrong.
    """
    done = subprocess.run(
        ("kubectl", "-n", in_namespace or namespace(root), *argv),
        input=stdin,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(argv)} failed: {done.stderr.strip()}")
    return done.stdout


def _aws(*argv: str) -> str:
    done = subprocess.run(
        ("aws", *argv, "--output", "text"), capture_output=True, text=True
    )
    if done.returncode != 0:
        raise RuntimeError(f"aws {' '.join(argv)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def absent_secrets(root: Path, workload: Workload) -> tuple[str, ...]:
    """Every (Secret, key) the manifest needs that the cluster does not hold.

    Checked before anything is applied, because a partial apply is worse than none: a
    Deployment whose Secret is missing leaves a pod in CreateContainerConfigError, which
    reads like a slow start rather than a missing input, and the Job ahead of it would
    have run first.
    """
    missing = []
    for secret, key in workload.secrets:
        try:
            value = _kubectl(
                root, "get", "secret", secret, "-o", f"jsonpath={{.data.{key}}}"
            )
        except RuntimeError:
            missing.append(f"Secret {secret} does not exist (needs key {key})")
            continue
        if not value:
            missing.append(f"Secret {secret} exists but has no key {key}")
    return tuple(missing)


def _declared_environment(root: Path, manifest: Path) -> dict[str, str | None]:
    """Every container environment entry a manifest declares, name to literal value.

    `None` for an entry that arrives through a `valueFrom`, so a caller can tell a
    variable that is declared as a reference from one that is declared as a literal from
    one that is not declared at all -- three states, and the two checks below act on
    different pairs of them.
    """
    documents = [d for d in yaml.safe_load_all((root / manifest).read_text()) if d]
    return {
        entry["name"]: entry.get("value")
        for document in documents
        for container in document.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
        for entry in container.get("env", [])
    }


def undeclared_variables(root: Path, workload: Workload) -> tuple[str, ...]:
    """Every required variable this workload's manifest does not declare.

    Checked before anything is applied, because this is the check whose absence let a
    control plane run in map-dev accepting Sessions it could not place while every gate
    in this repository stayed green. The process cannot raise that alarm about itself:
    with no manifest named it is simply not a placer, which is a legitimate shape for
    some other deployment, so which workload was *supposed* to place pods is a fact only
    a file like this one holds.

    Declaration and not value: an entry behind a `secretKeyRef` counts, because whether
    the Secret holds the key is `absent_secrets`' question and answering it twice would
    be two answers free to disagree.
    """
    declared = _declared_environment(root, workload.manifest)
    return tuple(
        f"{workload.manifest} declares no {variable}"
        for variable in workload.required_variables
        if variable not in declared
    )


def declared_vault_entries(
    root: Path, workload: Workload
) -> tuple[tuple[str, str], ...]:
    """Every vault entry this workload's manifest names, and what in it named each.

    Read out of the manifest rather than listed on the `Workload`, which is the opposite
    choice from `secrets` above and for the opposite reason. A Kubernetes `secretKeyRef`
    is a small closed set an author types once, so a copy beside it catches a typo; a
    routing table is data that grows an entry whenever a model is added, and a list here
    would silently stop covering the new ones. Only the *locations* are declared, and
    each location is required to be there: a manifest that renamed the variable raises
    here instead of quietly checking nothing.

    The second element of each pair is the entry name; the first is human text saying
    where it came from, so the refusal below can say `env MAP_POD_TOKEN_KEY_NAME` or
    `model gpt-5-codex` rather than leaving somebody to grep for a secret path.
    """
    documents = [
        d for d in yaml.safe_load_all((root / workload.manifest).read_text()) if d
    ]
    environment = _declared_environment(root, workload.manifest)
    found: list[tuple[str, str]] = []
    for variable in workload.vault_variables:
        named = environment.get(variable)
        if not named:
            raise RuntimeError(
                f"{workload.manifest} declares no literal {variable}. That variable "
                "is where this reads a vault entry name, so a manifest that renamed "
                "it or moved it behind a valueFrom would be applied with nothing "
                "checking the entry it names."
            )
        found.append((f"env {variable}", named))
    if workload.routing_table_key is not None:
        tables = [
            document["data"][workload.routing_table_key]
            for document in documents
            if document["kind"] == "ConfigMap"
            and workload.routing_table_key in document.get("data", {})
        ]
        if len(tables) != 1:
            raise RuntimeError(
                f"{workload.manifest} holds {len(tables)} ConfigMap documents "
                f"carrying {workload.routing_table_key}; expected 1. The pod mounts "
                "one and this would be checking another."
            )
        # No "the table declares no model" branch, because there is no way to reach
        # one: the parser's own schema requires at least one entry, and a document
        # holding `"entries": []` is refused here with `List should have at least 1
        # item after validation, not 0` (measured). So a routing document this gets
        # past always names a credential, and this loop cannot run zero times.
        table = routing_table_from_json(tables[0].encode())
        for model in sorted(table.declared_models()):
            entry = table.entry_for(model)
            found.append(
                (f"model {model}, routed to {entry.base_url}", entry.credential_name)
            )
    return tuple(found)


def unreachable_vault_entries(
    declared: tuple[tuple[str, str], ...], present: frozenset[str]
) -> tuple[str, ...]:
    """One refusal line per declared entry the account does not hold.

    Takes the account's entry names as an argument rather than fetching them, for the
    reason `rotating_credential` takes a username: the interesting behaviour is the
    comparison, and a function that made the call could only be exercised against a real
    account. Order follows `declared`, so the reader sees the pod's own key before the
    models that need one.
    """
    return tuple(
        f"vault entry {name} does not exist in Secrets Manager, and {why} names it"
        for why, name in declared
        if name not in present
    )


def _vault_entries_in_the_account() -> frozenset[str]:
    """Every Secrets Manager entry name this principal can list.

    One `list-secrets` rather than a `describe-secret` per name, so an absent entry and
    a refused read are never confused: a `describe-secret` on a name outside the
    caller's policy answers AccessDenied, which reads exactly like "not there". Here a
    failure of the call raises and stops the deploy, and only a name missing from a
    listing that succeeded counts as absent.
    """
    listed = _aws("secretsmanager", "list-secrets", "--query", "SecretList[].Name")
    return frozenset(listed.split())


def _secret_value(root: Path, secret: str, key: str) -> str:
    encoded = _kubectl(root, "get", "secret", secret, "-o", f"jsonpath={{.data.{key}}}")
    return base64.b64decode(encoded).decode()


def rotating_credential(dsn: str, master_username: str) -> bool:
    """Whether this DSN connects as the user whose password AWS rotates.

    Takes the username and the DSN rather than reading either, so it is testable without
    a cluster and without a credential. Compares only `urlsplit(...).username`; nothing
    else in the DSN is looked at, and no part of it is returned.
    """
    return urlsplit(dsn).username == master_username


def substituted(text: str, repository: str, reference: str) -> str:
    """Replace the placeholder image reference with a real one.

    Raises when the placeholder is not there. That is the case worth a raise rather than
    a no-op: a manifest whose digest was hand-edited into it once would then be applied
    unchanged for ever, running whatever that commit built long after the tree moved on.
    """
    placeholder = f"{repository}@{DIGEST_PLACEHOLDER}"
    if placeholder not in text:
        raise RuntimeError(
            f"{repository} is not at its placeholder in this manifest. Every platform "
            f"manifest carries {placeholder} and this one does not, so the image it "
            "would run is whatever somebody wrote in by hand."
        )
    done = text.replace(placeholder, reference)
    if DIGEST_PLACEHOLDER in done:
        raise RuntimeError(
            "a placeholder digest survived substitution, so some container in this "
            "manifest names a repository this workload does not declare"
        )
    return done


def with_account(text: str, account_id: str) -> str:
    """Replace the placeholder account id in a manifest with the caller's own.

    Unlike `substituted()` this does not raise when the placeholder is absent. Most
    manifests here name no account at all -- only the IRSA annotations and the two
    bucket-name variables do -- and a manifest that legitimately has none must not be
    refused. What is refused is the opposite case, and it is checked by the caller: a
    twelve-digit literal that is neither the placeholder nor the account being deployed
    to would be somebody else's account, applied.
    """
    return text.replace(ACCOUNT_PLACEHOLDER, account_id)


def caller_account(root: Path) -> str:
    """The account the ambient credentials belong to, as AWS itself reports it.

    Asked rather than configured. A variable naming the account would be a second answer
    free to disagree with the credentials actually in the environment, and it would
    disagree in the direction nobody notices: a manifest applied to the account the
    credentials name, carrying ARNs for the account the variable named.
    """
    found = subprocess.run(
        ("aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"),
        cwd=root,
        capture_output=True,
        text=True,
    )
    if found.returncode != 0:
        raise RuntimeError(
            "cannot read the AWS account from the current credentials, so no manifest "
            f"can be given one: {found.stderr.strip()}"
        )
    account = found.stdout.strip()
    if not (len(account) == 12 and account.isdigit()):
        raise RuntimeError(
            f"aws sts get-caller-identity answered {account!r}, which is not an "
            "account id"
        )
    return account


FOUNDRY_RESOURCE_VAR: Final = "MAP_FOUNDRY_RESOURCE"
FOUNDRY_PLACEHOLDER: Final = "map-foundry.services.ai.azure.com"
"""The un-substituted Azure Foundry host the committed routing table carries.

A Foundry resource name identifies one company's Azure account, so the value in the
repository names nobody: this host returns NXDOMAIN, measured. That is the same property
`DIGEST_PLACEHOLDER` and `ACCOUNT_PLACEHOLDER` are chosen for -- a deploy that did not
configure the real resource fails at DNS, rather than sending a tenant's prompts to
whatever host happened to be committed.

Unlike those two it cannot be derived. An account id is a property of the credentials
already in the environment; a Foundry resource is a choice, so it is read from
`MAP_FOUNDRY_RESOURCE` and its absence is a refusal rather than a default.
"""


def with_foundry(text: str, resource: str | None) -> str:
    """Point the routing table at this deployment's Foundry resource.

    A manifest carrying no placeholder is returned untouched -- only the Model Gateway's
    routing table names a host, and the other two workloads must not be refused for
    lacking one. A manifest that *does* carry it and has no resource configured is
    refused here, before anything is applied, because the alternative is a Model Gateway
    that rolls out cleanly, reports itself healthy, and fails every Turn at DNS.
    """
    if FOUNDRY_PLACEHOLDER not in text:
        return text
    if not resource:
        raise RuntimeError(
            f"this manifest's routing table is still at {FOUNDRY_PLACEHOLDER}, which "
            f"resolves nowhere. Set {FOUNDRY_RESOURCE_VAR} to this deployment's Azure "
            "Foundry resource name -- the name alone, not the whole host -- and deploy "
            "again."
        )
    return text.replace(FOUNDRY_PLACEHOLDER, f"{resource}.services.ai.azure.com")


def deployment_shortfall(status: Mapping[str, object], desired: object) -> str | None:
    """Say why a finished rollout does not mean the workload is serving, or None.

    Zero desired is the case worth a function, for `deploy/bootstrap.py`'s reason in a
    different shape: `kubectl rollout status` prints "successfully rolled out" and
    exits 0 for a Deployment scaled to zero, so the exit code cannot tell a running
    workload from an absent one.

    ABOVE ONE REPLICA THIS SAYS MORE THAN IT USED TO, and the extra meaning is
    deliberate. At `desired == 1` the equality below has two outcomes and they are
    "nothing is serving" and "everything is": there is no third state. At `desired == 2`
    there is, and `1 of 2 replicas are available` is it -- the workload IS serving, and
    the redundancy the second replica was added for is absent. Refusing that is the
    intent: a deploy that ends one pod short has not delivered what the manifest asked
    for, and the one moment somebody is watching the output is the moment to say so. The
    message states the counts and does not claim nothing is serving, because at two
    replicas that would be false.

    Only equality, and not `available < desired`. After `kubectl rollout status`
    succeeds the Deployment's own replica count is the new ReplicaSet's, so a larger
    `availableReplicas` than `desired` is not a state this is read in; if one ever
    appears it is a surprise worth failing on rather than rounding down to healthy.
    """
    if not isinstance(desired, int) or desired == 0:
        return (
            "the Deployment asks for no replicas, so nothing is serving; a rollout "
            "over zero pods succeeds and means nothing"
        )
    available = status.get("availableReplicas", 0)
    if available != desired:
        return f"{available} of {desired} replicas are available"
    return None


def declared_services(root: Path, workload: Workload) -> tuple[str, ...]:
    """Every Service name this workload's manifest declares, in file order.

    Read out of the manifest rather than declared on the `Workload`, which is
    `declared_vault_entries`' choice and for its reason: a Service somebody adds to a
    manifest is a Service whose address has to reach a pod, and a list kept here would
    go on covering the old set while continuing to pass. File order, so a refusal reads
    in the order somebody scrolling the manifest would meet them.

    An empty tuple is a legitimate answer -- `tool-gateway.yaml` and
    `model-gateway.yaml` each declare one Service and `schema-migration.yaml` declares
    none -- so the caller must not read "no Services" as "nothing to check went wrong".
    """
    documents = [
        d for d in yaml.safe_load_all((root / workload.manifest).read_text()) if d
    ]
    return tuple(
        str(d["metadata"]["name"]) for d in documents if d.get("kind") == "Service"
    )


def unrouted_service(service: str, listing: Mapping[str, object]) -> str | None:
    """Say why a Service routes to nothing, or None when it routes to a ready pod.

    Takes the parsed `kubectl get endpointslice -l kubernetes.io/service-name=<service>`
    listing rather than fetching it, for `unreachable_vault_entries`' reason: the
    interesting behaviour is the reading, and a function that made the call could only
    be exercised against a cluster.

    AN EMPTY LISTING IS THE ANSWER THIS EXISTS FOR, and it is never None. `kubectl
    apply` accepts a Service whose selector matches no pod and a Service whose named
    `targetPort` no container declares; `rollout status` and `deployment_shortfall` stay
    green through both, because both of those read the Deployment. What is left is a
    name that resolves and connects to nothing.

    Three things a naive reader would fold into "the listing was empty", and where each
    one actually goes:

    * No slices, because the selector matches no ready pod. That arrives as
      `{"items": []}` and is what this reports.
    * No `items` key, or an `items` that is not a list. That is not a kubectl listing at
      all, so it RAISES rather than being read as empty -- a parser that shrugged at
      unexpected input would report a healthy Service as broken, or the reverse, and
      there would be nothing to tell which.
    * A refused read, or an EndpointSlice API the server does not serve. `kubectl` exits
      non-zero for both, `_kubectl` raises, and neither reaches this function.

    What this cannot separate: a listing empty because it was taken in the wrong
    namespace. `_kubectl` passes the namespace `namespace()` reads out of
    `cluster-bootstrap.yaml`, and the manifest tests pin every document's namespace to
    that same file, so the two cannot disagree -- but nothing inside this function knows
    that, and a caller reaching past `_kubectl` could.

    An ABSENT `ready` condition means ready, which is the EndpointSlice API's own rule,
    so it is read that way. Counting a missing field as not-ready would refuse every
    healthy deploy against a server that omits it.

    The port half is a second net and a weaker one: a slice carrying no numeric port
    routes to no port, but whether Kubernetes actually produces that shape for a
    `targetPort` naming nothing is NOT measured here. What closes that hole locally is
    the manifest test requiring the name to be one the container declares.
    """
    items = listing.get("items")
    if not isinstance(items, list):
        raise RuntimeError(
            f"the EndpointSlice listing for Service {service} carries no items list, "
            "so it is not a kubectl listing and must not be read as an empty one"
        )
    addresses = 0
    ports: list[int] = []
    for entry in items:
        for endpoint in entry.get("endpoints") or []:
            conditions = endpoint.get("conditions") or {}
            if conditions.get("ready", True) is not False:
                addresses += len(endpoint.get("addresses") or [])
        ports.extend(
            port["port"]
            for port in entry.get("ports") or []
            if isinstance(port.get("port"), int)
        )
    if addresses == 0:
        return (
            f"Service {service} has no ready endpoint across {len(items)} "
            "EndpointSlices, so the address it publishes resolves and then connects to "
            "nothing. Its selector matches no ready pod of this workload."
        )
    if not ports:
        return (
            f"Service {service} has {addresses} ready endpoint addresses and no "
            "resolved port, so its targetPort names no port this pod declares"
        )
    return None


def _routing_refusal(root: Path, service: str) -> str | None:
    """`unrouted_service` against the live cluster, retried until the slices settle.

    The retry is for the ordering race `_ENDPOINTS_TIMEOUT_S` describes and for nothing
    else: a refusal is returned the moment the deadline passes, so a Service that really
    routes nowhere costs one wait and not a hang.
    """
    deadline = time.monotonic() + _ENDPOINTS_TIMEOUT_S
    while True:
        refusal = unrouted_service(
            service,
            json.loads(
                _kubectl(
                    root,
                    "get",
                    "endpointslice",
                    "-l",
                    f"kubernetes.io/service-name={service}",
                    "-o",
                    "json",
                )
            ),
        )
        if refusal is None or time.monotonic() >= deadline:
            return refusal
        time.sleep(_ENDPOINTS_POLL_S)


def enforcing_network_policy(nodeagent_arguments: tuple[str, ...]) -> bool:
    """Whether the CNI's enforcing container is switched on, read from its flag's VALUE.

    The value and not the argument: see `_TRUE_SPELLINGS` for why one spelling is not
    the flag. Three shapes are handled, each for a reason rather than defensively.

    `--enable-network-policy` with no value at all is TRUE, because that is what a Go
    boolean flag means when it is named and not assigned. Reading it as off would be
    this function inventing a state the program does not have.

    The LAST occurrence wins, matching the parser: a flag given twice takes its second
    value, so a list ending in the off spelling is off however it began.

    Absent entirely is FALSE. That is the fail-safe direction and it is the one that
    matters: the caller uses this to decide whether a policy set is decoration, and a
    list this cannot find the flag in must not be reported as enforcing.
    """
    verdict = False
    for argument in nodeagent_arguments:
        name, separator, value = argument.partition("=")
        if name != _NETWORK_POLICY_FLAG:
            continue
        verdict = separator != "=" or value in _TRUE_SPELLINGS
    return verdict


def inert_network_policies(
    policies: tuple[str, ...], nodeagent_arguments: tuple[str, ...]
) -> str | None:
    """Say why this namespace's NetworkPolicies filter nothing, or None.

    Takes the policy names and the CNI's arguments rather than fetching either, for
    `unrouted_service`'s reason: the interesting behaviour is the comparison, and a
    function that made the calls could only be exercised against a real cluster.

    No policies is None and not a refusal. A namespace with no NetworkPolicy is
    unfiltered and says so -- there is no gap between what it claims and what it does,
    which is the whole of what this looks for.

    Whether the flag is on is `enforcing_network_policy`'s question, including what an
    absent flag means -- which is off, so an argument list this could not find the flag
    in produces a refusal rather than being passed over. That direction is the one that
    matters: a permissive reading turns "we could not tell" into "it is enforcing", the
    exact sentence this function exists to stop somebody writing.
    """
    if not policies:
        return None
    if enforcing_network_policy(nodeagent_arguments):
        return None
    return (
        f"{len(policies)} NetworkPolicy objects are in this namespace "
        f"({', '.join(policies)}) and the CNI is not enforcing any of them: container "
        f"{_NODEAGENT_CONTAINER} of {_CNI_NAMESPACE}/{_CNI_DAEMONSET} was run with "
        f"{nodeagent_arguments or 'no arguments this could read'}. Every one of those "
        "objects is accepted by the API server, listed by `kubectl get netpol`, and "
        "applied to no packet. Turn enforcement on -- `enableNetworkPolicy` on the "
        "vpc-cni add-on, declared in deploy/terraform/vpc_cni.tf -- or delete the "
        "policies; leaving both is a security boundary that exists only as text."
    )


def _network_policies(root: Path) -> tuple[str, ...]:
    """Every NetworkPolicy name this platform's namespace holds, sorted."""
    listed = json.loads(_kubectl(root, "get", "networkpolicy", "-o", "json"))
    return tuple(sorted(str(item["metadata"]["name"]) for item in listed["items"]))


def _nodeagent_arguments(root: Path) -> tuple[str, ...]:
    """The CNI enforcing container's command-line arguments, or none if it is absent.

    The container is found BY NAME and not by position. `aws-node` carries two
    containers and the enforcing one is second today, so an index works and stops
    working the day the order changes -- and it would stop working by reading the wrong
    container's arguments, which is a reading that looks like an answer.
    """
    described = json.loads(
        _kubectl(
            root,
            "get",
            "daemonset",
            _CNI_DAEMONSET,
            "-o",
            "json",
            in_namespace=_CNI_NAMESPACE,
        )
    )
    for container in described["spec"]["template"]["spec"]["containers"]:
        if container.get("name") == _NODEAGENT_CONTAINER:
            return tuple(str(argument) for argument in container.get("args") or ())
    return ()


def _reference(root: Path, repository: str) -> str:
    """The digest-pinned reference for the image built from HEAD.

    The tag comes from `push-platform-image.sh`'s own print-tag seam rather than being
    recomputed here, so there is one definition of what a tag looks like; the digest
    comes from the registry, because that is what a kubelet resolves.
    """
    tag = subprocess.run(
        ("sh", str(root / _PUSH_SCRIPT)),
        capture_output=True,
        text=True,
        env={"MAP_PRINT_TAG_ONLY": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        cwd=root,
    )
    if tag.returncode != 0:
        raise RuntimeError(f"could not derive the tag for HEAD: {tag.stderr.strip()}")
    # `describe-images` EXITS NON-ZERO for an absent tag rather than returning nothing,
    # so `_aws` raised and the message below -- written for exactly this case -- was
    # unreachable. A caller who had committed anything after their last push got a
    # traceback ending in `ImageNotFoundException` instead of the one line that says
    # what to do. Only that one failure is turned back into an empty digest; expired
    # credentials and an unreachable registry are different problems and still raise.
    try:
        digest = _aws(
            "ecr",
            "describe-images",
            "--repository-name",
            repository,
            "--image-ids",
            f"imageTag={tag.stdout.strip()}",
            "--query",
            "imageDetails[0].imageDigest",
        )
    except RuntimeError as failure:
        if "ImageNotFoundException" not in str(failure):
            raise
        digest = ""
    if not digest or digest == "None":
        raise RuntimeError(
            f"{repository} holds no image tagged {tag.stdout.strip()}. Nothing has "
            "pushed this commit; run deploy/docker/push-platform-image.sh -- and note "
            "that the tag names HEAD, so any commit made since the last push is enough "
            "to produce this, whatever it changed."
        )
    account = _aws("sts", "get-caller-identity", "--query", "Account")
    region = _aws("configure", "get", "region")
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{repository}@{digest}"


def main(argv: tuple[str, ...] | None = None) -> int:
    """Apply one workload, refusing before touching the cluster when an input is
    absent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=[w.component for w in WORKLOADS])
    parser.add_argument(
        "--allow-rotating-credential",
        action="store_true",
        help=(
            "apply even though the database DSN connects as the RDS master user, whose "
            "password AWS rotates every 7 days -- the workload will fail "
            "authentication within that window"
        ),
    )
    parser.add_argument(
        "--allow-absent-vault-entry",
        action="store_true",
        help=(
            "apply even though the manifest names AWS Secrets Manager entries the "
            "account does not hold -- the workload will start and serve its health "
            "path, and every request that needs one of those entries will fail at the "
            "vault"
        ),
    )
    options = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    workload = next(w for w in WORKLOADS if w.component == options.component)

    undeclared = undeclared_variables(root, workload)
    if undeclared:
        for line in undeclared:
            print(f"missing input: {line}", file=sys.stderr)
        print(
            "nothing was applied. A workload whose manifest omits one of these starts, "
            "passes its probes and does less than it is for -- which is how a control "
            "plane came to accept Sessions and place no pod for any of them, with "
            "every check green.",
            file=sys.stderr,
        )
        return 1

    absent = absent_secrets(root, workload)
    if absent:
        for line in absent:
            print(f"missing input: {line}", file=sys.stderr)
        print(
            "nothing was applied. These hold values that are not in this repository "
            "and must not be; create them from a file rather than from a command-line "
            "argument so the value does not land in shell history.",
            file=sys.stderr,
        )
        return 1

    # The DaemonSet is read only when there is a policy for it to be inert about, which
    # keeps a deploy from depending on kube-system read access it does not otherwise
    # need -- and keeps this from costing anything at all today, where the namespace
    # holds no NetworkPolicy.
    policies = _network_policies(root)
    inert = (
        inert_network_policies(policies, _nodeagent_arguments(root))
        if policies
        else None
    )
    if inert is not None:
        print(f"inert boundary: {inert}", file=sys.stderr)
        print(
            "nothing was applied. This is not about the workload being deployed -- it "
            "is about the cluster it would join, where a filter everybody can read is "
            "filtering nothing.",
            file=sys.stderr,
        )
        return 1

    declared = declared_vault_entries(root, workload)
    unreachable = (
        unreachable_vault_entries(declared, _vault_entries_in_the_account())
        if declared
        else ()
    )
    if unreachable:
        for line in unreachable:
            print(f"missing input: {line}", file=sys.stderr)
        if not options.allow_absent_vault_entry:
            print(
                "nothing was applied. Nothing fetches a vault entry until a request "
                "needs it, so applying past this produces a workload whose pods are "
                "Ready and whose every real request fails at AWS. Create the entries "
                "-- in deploy/terraform/ under ADR-021, not at a shell -- or pass "
                "--allow-absent-vault-entry to accept that deliberately.",
                file=sys.stderr,
            )
            return 1
        print(
            "WARNING: applying with the entries above absent. This workload will pass "
            "its probes and refuse every request that needs one of them.",
            file=sys.stderr,
        )

    if workload.database_secret is not None:
        secret, key = workload.database_secret
        master = _aws(
            "rds",
            "describe-db-instances",
            "--db-instance-identifier",
            "map-dev-db",
            "--query",
            "DBInstances[0].MasterUsername",
        )
        if rotating_credential(_secret_value(root, secret, key), master):
            if not options.allow_rotating_credential:
                print(
                    f"{secret}/{key} connects as {master}, the RDS master user. AWS "
                    "rotates that password every 7 days and this role cannot read "
                    "the rotated value (its policy is map/dev/platform/* and the "
                    "RDS-managed secret is not under it), so this workload would "
                    "serve and then fail authentication days after the deploy. "
                    "Connect as a dedicated application role, or pass "
                    "--allow-rotating-credential to accept the expiry deliberately.",
                    file=sys.stderr,
                )
                return 1
            print(
                f"WARNING: {secret}/{key} expires with the next rotation",
                file=sys.stderr,
            )

    reference = _reference(root, workload.repository)
    print(f"== image {reference}")
    account = caller_account(root)
    print(f"== account {account}")

    for companion in workload.companions:
        # No image to substitute -- a companion carries identities and permissions. It
        # does carry account ids, in the `eks.amazonaws.com/role-arn` annotation on
        # each ServiceAccount, so it goes through stdin rather than by path: left at the
        # placeholder those annotations name a role in account 000000000000 and every
        # pod using them fails to assume anything.
        print(f"== {companion.name}")
        _kubectl(
            root,
            "apply",
            "-f",
            "-",
            stdin=with_account((root / companion).read_text(), account),
        )

    if workload.schema_job is not None:
        print(f"== {workload.schema_job.name}, from scratch")
        # Deleted rather than applied over. A completed Job's pod template is immutable,
        # so `kubectl apply` on one whose image changed fails on a message about an
        # immutable field; and a Job left from the last deploy would satisfy a `wait`
        # without having run this deploy's revisions.
        _kubectl(root, "delete", "job", "map-schema-migration", "--ignore-not-found")
        _kubectl(
            root,
            "apply",
            "-f",
            "-",
            stdin=with_account(
                substituted(
                    (root / workload.schema_job).read_text(),
                    workload.repository,
                    reference,
                ),
                account,
            ),
        )
        _kubectl(
            root,
            "wait",
            "--for=condition=complete",
            f"--timeout={_JOB_TIMEOUT}",
            "job/map-schema-migration",
        )
        print(_kubectl(root, "logs", "job/map-schema-migration"))

    for name, source in workload.generated_config_maps:
        print(f"== configmap {name}, from {source.name}")
        # Unsubstituted, and that is not an oversight. `substituted()` rewrites the
        # digest placeholder to THIS commit's platform image; session-pod.yaml carries
        # the same placeholder and pod_runner._pod_for rewrites it per Session to the
        # registered Environment's digest. Substituting here would start every Session
        # pod on the control plane's own image -- which holds no `codex` -- and the
        # failure would arrive as a readiness timeout with nothing in it about an image.
        _kubectl(
            root,
            "apply",
            "-f",
            "-",
            stdin=_kubectl(
                root,
                "create",
                "configmap",
                name,
                f"--from-file={root / source}",
                "--dry-run=client",
                "-o",
                "yaml",
            ),
        )

    print(f"== {workload.manifest.name}")
    _kubectl(
        root,
        "apply",
        "-f",
        "-",
        stdin=with_foundry(
            with_account(
                substituted(
                    (root / workload.manifest).read_text(),
                    workload.repository,
                    reference,
                ),
                account,
            ),
            os.environ.get(FOUNDRY_RESOURCE_VAR),
        ),
    )
    _kubectl(
        root,
        "rollout",
        "status",
        f"deploy/{workload.component}",
        f"--timeout={_ROLLOUT_TIMEOUT}",
    )
    described = json.loads(
        _kubectl(root, "get", "deploy", workload.component, "-o", "json")
    )
    shortfall = deployment_shortfall(described["status"], described["spec"]["replicas"])
    if shortfall is not None:
        print(f"deploy incomplete: {shortfall}", file=sys.stderr)
        return 1

    # After the rollout and not before it. A Service acquires an endpoint when a pod
    # this apply created becomes Ready, so checking earlier would be measuring the
    # previous revision's pods -- or, on a first deploy, nothing at all.
    for service in declared_services(root, workload):
        refusal = _routing_refusal(root, service)
        if refusal is not None:
            print(f"deploy incomplete: {refusal}", file=sys.stderr)
            return 1
        print(f"== service {service} routes to a ready pod")

    print(f"{workload.component} is serving in namespace {namespace(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
