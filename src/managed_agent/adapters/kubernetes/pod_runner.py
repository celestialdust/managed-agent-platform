"""Creating, watching and destroying one Session's pod in a Kubernetes cluster.

This is the concrete cluster behind `PodRunner`. It compiles the pod manifest the
repository keeps under `deploy/k8s/` into one Session's pod: the image the compiled
configuration names, three per-Session Secrets holding the two compiled documents and
the shim's bearer token, and the Session's identifier in the one env var the shim reads.
It substitutes those and nothing else, so the manifest stays the single source for how a
Session pod is confined and this module has no opinion about it.

**A pod is reported running only when every container in it is ready.** The runtime and
the shim are two halves of one Session -- a runtime with no shim beside it takes no
Turn, and a shim whose runtime socket never appeared answers none -- so a phase that
meant "the pod exists" would hand the control plane an address that refuses everything.
The emptiness check beside the `all()` is load-bearing: `all(())` is true, and a pod
whose container statuses have not been reported yet would otherwise be reported ready.

The API client and the credential are built inside each public call rather than held.
That is what lets both environments take one code path -- the in-cluster loader is
synchronous, the kubeconfig loader is a coroutine because it runs a credential plugin as
a subprocess -- and it leaves this object with nothing to dispose, unlike a connection
pool. It also means a rotated service-account token and an expired `eks get-token`
credential are both picked up without a refresh hook. It costs one TLS handshake per
call, measured at 33 ms against 17 ms for a reused client, which sits inside a Turn
measured in seconds.

The frozen dataclass is shallow: `manifest` is a mutable mapping and freezing the fields
does not freeze it. Nothing here mutates it -- `_pod_for` deep-copies before it writes
-- and that is a discipline this docstring states because the type cannot.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
import time
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from uuid import UUID

import yaml
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import V1ObjectMeta, V1Pod, V1Secret
from kubernetes_asyncio.config.incluster_config import (
    SERVICE_CERT_FILENAME,
    SERVICE_TOKEN_FILENAME,
    InClusterConfigLoader,
)

from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import (
    NodeHeadroom,
    PlacedPod,
    PodNotStarted,
    PodPhase,
    pod_name_for,
)
from managed_agent.core.ids import SessionId
from managed_agent.core.registration.skill import SkillFile
from managed_agent.session_shim.pod_channel import shim_token_for

_NOT_FOUND: Final = 404
_ALREADY_EXISTS: Final = 409
_LOG = logging.getLogger(__name__)

_IN_CLUSTER_TELL: Final = "KUBERNETES_SERVICE_HOST"

_AUTOSCALER_DEPLOYMENT: Final = "cluster-autoscaler"
"""The Deployment whose arguments carry the node ceiling this platform publishes.

It lives in this platform's OWN namespace -- `deploy/k8s/cluster-autoscaler.yaml` puts
it there -- so the read is namespaced and needs no cross-namespace authority. Worth
stating because the obvious guess is `kube-system`, where the upstream example installs
it, and a read addressed there answers 404 rather than being refused: the failure would
read as "no autoscaler is deployed" instead of "we looked in the wrong place", and the
field would go quietly empty on a cluster that has one.
"""

_MAX_NODES_TOTAL: Final = re.compile(r"^--max-nodes-total=(\d+)$")
"""The flag that binds how far the autoscaler will grow the cluster unattended.

Anchored and whole rather than searched for as a substring, so a longer flag that merely
starts with this name cannot be read as this one. An argument written as two list
entries -- the flag, then the number -- does not match, and not matching is the right
answer for a spelling this cannot prove it understands.
"""
_SESSION_LABEL: Final = "map.session-id"
_SHIM_CONTAINER: Final = "session-shim"
# What the shim reads out of its environment before it opens the Session's thread, and
# where each value comes from. Keyed by the variable the manifest declares, so a
# variable the manifest stops declaring is simply not substituted rather than invented,
# and one it adds is refused below until this table knows a value for it.
_SHIM_ENV: Final[Mapping[str, Callable[[CompiledConfig], str]]] = {
    "MAP_SESSION_ID": lambda compiled: str(compiled.session_id),
    "MAP_MODEL": lambda compiled: compiled.model,
    "MAP_MODEL_PROVIDER": lambda compiled: compiled.model_provider,
}

_SEED_CONTAINER: Final = "seed-rollout"
# What the init container that seeds a restored Rollout reads, on the same terms as the
# shim's table above: keyed by the variable the manifest declares, and refused below
# when the manifest declares fewer than this knows values for.
#
# Lowercase spellings rather than `str(bool)`'s capitalised ones, because the reader is
# a separate process parsing a string and lowercase is what every other configuration
# format in this pod uses. One spelling, decided here and parsed there.
#
# This is the ONLY container told. The shim decides between resuming and starting from
# whether a Rollout is on disk, so a second copy of the fact in its environment would
# be free to disagree with the file -- and the disagreement that starts a fresh thread
# over a seeded Rollout is the silent one (ADR-004, ADR-031).
_SEED_ENV: Final[Mapping[str, Callable[[CompiledConfig], str]]] = {
    "MAP_RESUMING": lambda compiled: "true" if compiled.resuming else "false",
}

# The restart policy under which a container that exited is not coming back, and so is
# terminal. The manifest pins this; the reader below checks it rather than assuming it,
# because under a restarting policy the same status means "between attempts".
_NO_RESTARTS: Final = "Never"

# Which file each secret volume in the manifest carries, keyed by the volume's own name.
# Keyed that way so the volume list stays the manifest's business: a volume this table
# does not know about is refused rather than mounted empty, because a pod with an
# unpopulated secret volume never starts and the reason is three layers down.
#
# `config.toml` is the name the pod's init container copies out of the mount, and
# `requirements.toml` is the name the runtime reads its managed configuration under --
# it is also the key `config_compiler.check_floors` parses that document back by.
# `token` is the leaf of the path the shim reads its bearer from. None of the three is
# ours to choose.
_SECRET_FILES: Final[Mapping[str, str]] = {
    "compiled": "config.toml",
    "requirements": "requirements.toml",
    "shim-token": "token",
}

# 2 s x 30 for the runtime's startupProbe plus 2 s x 30 for the shim's readinessProbe is
# 120 s the manifest itself considers healthy, so a shorter bound would refuse a pod
# nothing is wrong with. A further 60 s is the image pull. Polled every second: finer
# than the 2 s probe period, so no readiness transition is seen more than a second late.
#
# The last 120 s is the `restore-working-lane` init container, and it is why this number
# moved from 180. That container fetches everything the Session's working lane holds
# before any regular container starts, and the ceiling it fetches under is 2048 objects
# -- serialized at ~20 ms a round trip that is ~41 s on its own, against a budget the
# pull and the two probes had already spent. It fetches concurrently for that reason,
# and this bound is the other half of the same arithmetic: either alone is a coin flip,
# and an init container that overruns gets its pod deleted by `ensure`'s own cleanup
# path -- which reads as a Session that would not start rather than as a restore that
# ran out of time. Raising it costs nothing on the common path, because a pod that is
# ready answers the poll a second later either way, and `_why_it_will_not_start` still
# refuses a doomed pod in seconds rather than at the end of this.
_READY_TIMEOUT_SECONDS: Final = 300.0

# How long a pod may sit unscheduled before the wait gives up on a node arriving.
#
# Its own number, because it is waiting on a different thing than the bound above: not
# an image pull on a node that exists, but a node that does not exist yet. Measured on
# this cluster, an autoscaled t3.medium takes minutes to launch, register with the API
# server and pull the Session image, where the ready wait above is bounded by a probe
# period the manifest itself chooses.
#
# The two are not added together. The clock is reset to the ready bound at the moment
# the pod is first seen scheduled, so a pod that waited four minutes for a node still
# gets its full ready window afterwards, and a doomed image on a node that was already
# there is still refused in seconds by `_why_it_will_not_start`.
_SCHEDULING_TIMEOUT_SECONDS: Final = 420.0

_POLL_SECONDS: Final = 1.0

# The cluster's failure messages are for a log line, not for a parser, and an image pull
# error carries a whole containerd trace.
_REASON_CAP: Final = 200


@contextmanager
def _absent_is_success() -> Iterator[None]:
    """Swallow a 404, because an object that is already gone is the outcome asked for.

    Written as a context manager rather than a helper taking the delete callable: the
    generated client's methods are not typed as a common callable, so passing them
    around would need a cast, and a cast on the deletion path is the wrong place to
    start lying to the type checker.
    """
    try:
        yield
    except ApiException as err:
        if err.status != _NOT_FOUND:
            raise


@asynccontextmanager
async def _core_api() -> AsyncIterator[client.CoreV1Api]:
    """A CoreV1Api authenticated for wherever this process is running.

    The service host variable is the in-cluster tell -- it is the same one the client's
    own in-cluster loader keys on -- and the two loaders differ in shape, not only in
    source: the in-cluster one reads two files synchronously, the kubeconfig one is a
    coroutine because an EKS kubeconfig authenticates by running `aws eks get-token`.
    The class is used in place of the module-level `load_incluster_config` because only
    the class carries annotations, and an unannotated call fails this repository's
    `mypy --strict` gate.
    """
    async with _api_client() as api:
        yield client.CoreV1Api(api)


@asynccontextmanager
async def _apps_api() -> AsyncIterator[client.AppsV1Api]:
    """An AppsV1Api authenticated the same way, for the one Deployment this reads.

    A second context manager rather than a parameter on the one above, because the two
    yield different classes and a factory returning a union would push an `isinstance`
    onto every caller. They share `_api_client`, so the authentication rule -- the part
    that could be got wrong -- is written once.

    Used for exactly one read: the cluster autoscaler's own Deployment, whose arguments
    carry the node ceiling this platform publishes. It creates and changes nothing.
    """
    async with _api_client() as api:
        yield client.AppsV1Api(api)


@asynccontextmanager
async def _api_client() -> AsyncIterator[client.ApiClient]:
    """An authenticated client for wherever this process is running.

    Extracted so the two typed API wrappers above share one statement of how to
    authenticate. That is a rule and not a coincidence: written twice, the in-cluster
    and kubeconfig branches could come to disagree, and the failure would be a process
    that reads pods in the cluster and nodes only on a laptop.
    """
    configuration = client.Configuration()
    if _IN_CLUSTER_TELL in os.environ:
        InClusterConfigLoader(
            token_filename=SERVICE_TOKEN_FILENAME,
            cert_filename=SERVICE_CERT_FILENAME,
        ).load_and_set(configuration)
    else:
        await config.load_kube_config(client_configuration=configuration)
    async with client.ApiClient(configuration=configuration) as api:
        yield api


def _secret_name(pod_name: str, volume: str) -> str:
    """The name of the Secret backing one of this pod's secret volumes.

    Derived from the pod's name, which is itself derived from the Session's id, so a
    Session's secrets are as recomputable as its pod is and no table holds them. The
    volume's own name is the suffix, which means a volume added to the manifest needs no
    second list here to be named.
    """
    return f"{pod_name}-{volume}"


def _secret_volumes(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Every secret volume the manifest declares, by volume name, in order.

    Refuses a secret volume this module has no file for. A volume left unpopulated makes
    kubelet retry the mount forever and the pod never starts; refusing at compile time
    names the manifest change that caused it instead.
    """
    volumes = tuple(
        volume["name"] for volume in manifest["spec"]["volumes"] if "secret" in volume
    )
    unknown = [name for name in volumes if name not in _SECRET_FILES]
    if unknown:
        raise PodNotStarted(
            f"the pod manifest declares secret volumes {unknown} that this adapter has "
            "no file for; a volume mounted from a Secret nobody creates keeps the pod "
            "out of Running for ever"
        )
    return volumes


def _pod_for(
    manifest: Mapping[str, Any], pod_name: str, compiled: CompiledConfig
) -> dict[str, Any]:
    """This Session's pod: the repository's manifest with six things substituted.

    The six are the pod's name, the Session label that `_claims_this_session` reads
    back for both `ensure` and `phase_of`, the image on every container, the Secret name
    behind every secret volume, the skill files this Session projects out of that
    volume, and the shim's environment entries that `_SHIM_ENV` names. Everything else
    -- the security contexts, the probes, the deny-rule ordering the init container
    enforces, the node selector -- is the manifest's and is copied through untouched.

    Substituted by container and volume *name* rather than by position, so a manifest
    whose containers are reordered still has the right one rewritten. A missing name is
    refused, because the alternative is a pod that starts without the half that answers
    Turns -- and so is a manifest that declares fewer environment entries than
    `_SHIM_ENV` has values for, because the shim reads all of them before it opens the
    Session's thread and one it cannot read leaves a pod that starts and cannot serve.

    Returns the manifest body as a mapping rather than a `V1Pod`, and that is the whole
    point: the generated client serializes a mapping through unchanged, so the file
    under `deploy/k8s/` stays the only description of a Session pod. Transcribing it
    into the generated models would put a second description in Python, free to drift
    from the one an operator reads and `kubectl apply`s. The read path is where the
    models earn their keep -- `_phase_of` and `_why_it_will_not_start` branch on typed
    status fields of what the cluster hands back.
    """
    body = copy.deepcopy(dict(manifest))
    spec: dict[str, Any] = body["spec"]
    body["metadata"]["name"] = pod_name
    # The per-pod DNS record is built from `spec.hostname`, not from `metadata.name`,
    # so the two are set from one value here. Without this the record does not exist
    # and every Turn fails at the shim hop -- deploy/k8s/session-pod.yaml says more.
    spec["hostname"] = pod_name
    body["metadata"]["labels"][_SESSION_LABEL] = str(compiled.session_id)
    for container in list(spec.get("initContainers", ())) + list(spec["containers"]):
        container["image"] = compiled.runtime_image
    skill_items = _skill_volume_items(compiled)
    projected = False
    for volume in spec["volumes"]:
        if "secret" in volume:
            volume["secret"]["secretName"] = _secret_name(pod_name, volume["name"])
            if skill_items and volume["name"] == _SKILL_VOLUME:
                volume["secret"]["items"] = skill_items
                projected = True
    if skill_items and not projected:
        raise PodNotStarted(
            f"the pod manifest declares no {_SKILL_VOLUME!r} secret volume, so this "
            f"Session's {len(compiled.skill_files)} skill file(s) would be written "
            "into a Secret nothing mounts and the runtime would discover no skill at "
            "all -- a Session that asked for one and silently has none"
        )
    _fill_env(
        spec["containers"],
        _SHIM_CONTAINER,
        _SHIM_ENV,
        compiled,
        absent="would run a runtime nothing can send a Turn to",
        unfilled="the shim would read them as unset and refuse to open the Session",
    )
    _fill_env(
        spec.get("initContainers", ()),
        _SEED_CONTAINER,
        _SEED_ENV,
        compiled,
        absent=(
            "would open a fresh thread for a Session that already holds a Rollout, "
            "replaying history its compaction checkpoints have folded"
        ),
        unfilled=(
            "the seed would read them as unset and take every resume for a first "
            "placement"
        ),
    )
    return body


def _fill_env(
    containers: Iterable[Mapping[str, Any]],
    name: str,
    table: Mapping[str, Callable[[CompiledConfig], str]],
    compiled: CompiledConfig,
    *,
    absent: str,
    unfilled: str,
) -> None:
    """Substitute one container's declared environment entries, or refuse.

    Two containers in this pod read values compiled per Session, and the rule they
    share is the one worth having in a single place: a manifest that declares fewer
    entries than this adapter has values for is refused rather than partly filled,
    because the process reading an unfilled entry cannot tell it from one nobody meant
    to set. Entries this table has no value for are left exactly as the manifest wrote
    them, so a variable the manifest adds for its own reasons is not invented here.

    Located by container *name*, so a reordered manifest still has the right one
    rewritten -- and an absent name is refused, because the substitution silently
    doing nothing is how a pod starts without the half that acts on it.

    The two consequence strings are the callers' because they are the only part that
    differs, and they are what a reader of the refusal actually needs: which pod would
    have started, and what it would have done wrong.
    """
    found = [c for c in containers if c["name"] == name]
    if not found:
        raise PodNotStarted(
            f"the pod manifest declares no {name!r} container, so this pod {absent}"
        )
    substituted = set()
    for entry in found[0]["env"]:
        value = table.get(entry["name"])
        if value is not None:
            entry["value"] = value(compiled)
            substituted.add(entry["name"])
    missing = sorted(set(table) - substituted)
    if missing:
        raise PodNotStarted(
            f"the pod manifest's {name!r} container declares no {missing} for this "
            f"adapter to substitute, so {unfilled}"
        )


# The volume a skill rides in on. Named once because two places have to agree: the
# Secret key written here and the `items` projection that maps it back to a path. A
# skill goes into the volume the runtime already reads its managed configuration
# from, because that is the directory codex discovers admin-scope skills under -- a
# second mount inside the same path would shadow the file the first one delivers.
_SKILL_VOLUME: Final = "requirements"
_SKILL_KEY_PREFIX: Final = "skill."
_SKILL_KEY_SEPARATOR: Final = "_"

# What the Kubernetes API admits as a Secret key, as the API server's own validator
# spells it. Held here because the delivery below has to answer the question the API
# server would answer at pod-create time, and answer it early: by then the Secret is
# already written and the error names a key rather than a file.
_LEGAL_SECRET_KEY: Final = re.compile(r"^[-._a-zA-Z0-9]+$")


def _skill_secret_key(relative_path: str) -> str:
    """The Secret key that carries one of a skill's files.

    A Kubernetes Secret key admits `[-._a-zA-Z0-9]` and no `/`, and a skill is delivered
    as a whole file set -- `skills/<name>/SKILL.md` and every sibling it tells the model
    to read, at any depth beneath it. So the key is the whole relative path with its
    separators flattened to `_`: total over every path a delivery can produce, and
    derived from the whole path rather than a fragment of it, so no file of one skill
    can take the key of another skill's file.

    The whole path and not the part after `skills/`, because a function that strips a
    prefix is only injective over paths that carry it -- and this one is handed whatever
    the delivery built. Flattening the whole thing keeps the argument short: two keys
    are equal only if the two paths differ nowhere except at a `/`-versus-`_` position.

    That remaining ambiguity is real -- `a/b` and `a_b` flatten alike -- and it is
    refused rather than encoded around, in `_skill_secret_entries` below where the whole
    file set is in view. A digest suffix would make every key unique and every key
    unreadable, and these keys are read by hand off a Secret by whoever is working out
    why a skill did not appear.

    The `skill.` prefix is what keeps a skill's key out of the namespace of the base
    file this volume already carries: every key here holds a `.` after a fixed word no
    `_SECRET_FILES` value begins with, so no path can flatten into `requirements.toml`.
    """
    return _SKILL_KEY_PREFIX + relative_path.replace("/", _SKILL_KEY_SEPARATOR)


def _skill_secret_entries(
    compiled: CompiledConfig,
) -> tuple[tuple[str, SkillFile], ...]:
    """Each of this Session's skill files paired with the key that carries it.

    One function rather than the same comprehension in two, because the pairing is one
    piece of knowledge: the Secret writes the key and the volume's `items` maps that key
    back to a path, and a Secret key with no item is a file written into the cluster and
    mounted nowhere while an item with no key makes kubelet refuse the mount for ever.
    Both consumers deriving from this is what makes the two halves impossible to skew.

    Both refusals live here for a reason the call order settles: `_create` writes every
    Secret and only then creates the pod. A check living only where `items` is built
    would fire after a Secret already existed in the cluster -- and an already-existing
    Secret is left alone on the next attempt, so wrong content would outlive the failure
    that revealed it. Raising from the pairing means whichever consumer runs first
    refuses, and neither can be the one that skipped the check.

    Two paths whose keys collide are refused rather than merged. A Secret holds one
    value per key, so delivering both would write one file over the other and hand the
    model a skill missing a file it was told to read -- with no error anywhere, because
    a dict that overwrites a key is not a failure. A tenant can rename one of two files.

    A path holding a character no key admits is refused too, and that check can fire:
    the upstream bundle parse rejects an absolute path, a `..`, an empty or `.` segment
    and a control character, and stops there. A space, a parenthesis, a `+` and a
    non-ASCII letter all survive it, so the character class is guaranteed nowhere above
    this line.
    """
    seen: dict[str, str] = {}
    entries: list[tuple[str, SkillFile]] = []
    for file in compiled.skill_files:
        key = _skill_secret_key(file.relative_path)
        if not _LEGAL_SECRET_KEY.match(key):
            raise PodNotStarted(
                f"{file.relative_path!r} cannot be delivered into a Session: a "
                "Kubernetes Secret key admits only [-._a-zA-Z0-9], and this path holds "
                "a character outside that set even after its separators are flattened, "
                f"so the key {key!r} is one the API server refuses. Rename the file "
                "using letters, digits, '-', '.' and '_' only"
            )
        collides_with = seen.get(key)
        if collides_with is not None:
            raise PodNotStarted(
                f"{collides_with!r} and {file.relative_path!r} both need the Secret "
                f"key {key!r}, and a Secret holds one value per key -- delivering both "
                "would write one of the two files over the other and the model would "
                "be told to read a file that never arrived. A key is the file's path "
                "with its '/' separators flattened to '_', so a path holding '_' where "
                "another holds '/' collides; rename one of the two files"
            )
        seen[key] = file.relative_path
        entries.append((key, file))
    return tuple(entries)


def _skill_volume_items(compiled: CompiledConfig) -> list[dict[str, str]]:
    """The `items` this Session's requirements volume projects, base file included.

    `items` is exhaustive: a volume that declares any of them projects those keys and
    no others, so `requirements.toml` has to be listed here or the runtime loses the
    file it reads on every start. That is the trap this function exists to close -- the
    mistake is invisible in a diff that only adds skills.

    Every skill file is projected at its own relative path, which is why one skill can
    be more than one file: a `SKILL.md` that tells the model to read a sibling needs the
    sibling mounted beside it, and a delivery carrying only the `SKILL.md` produces an
    agent that follows the instruction, finds nothing, and reports the skill unusable.

    Returned empty when the Session has no skills, so the manifest keeps its default
    projection and a Session without skills is byte-identical to one from before skills
    existed.
    """
    if not compiled.skill_files:
        return []
    return [
        {
            "key": _SECRET_FILES[_SKILL_VOLUME],
            "path": _SECRET_FILES[_SKILL_VOLUME],
        },
        *(
            {"key": key, "path": file.relative_path}
            for key, file in _skill_secret_entries(compiled)
        ),
    ]


def _secrets_for(
    pod_name: str, volumes: tuple[str, ...], compiled: CompiledConfig, key: bytes
) -> tuple[V1Secret, ...]:
    """One Secret per secret volume, holding the file that volume mounts.

    `stringData` rather than `data`, so the compiled documents are handed over as the
    text they are and no base64 step sits between what was compiled and what the runtime
    loads.

    The shim's token is derived here through the same function the control plane derives
    it with, so the two cannot disagree. It is written into its own Secret -- mounted,
    per the manifest, into the shim container alone -- and not beside the compiled
    documents, which the runtime container also mounts.
    """
    contents: dict[str, dict[str, str]] = {
        "compiled": {_SECRET_FILES["compiled"]: compiled.config_toml},
        # The one volume that carries more than its own file. A skill is delivered as a
        # Secret key rather than through a second volume because the runtime discovers
        # skills under the directory this volume already mounts -- a second mount inside
        # the same path would shadow the file the first one is there to deliver.
        "requirements": {
            _SECRET_FILES["requirements"]: compiled.requirements_toml,
            **{key: file.text for key, file in _skill_secret_entries(compiled)},
        },
        "shim-token": {
            _SECRET_FILES["shim-token"]: shim_token_for(compiled.session_id, key)
        },
    }
    return tuple(
        V1Secret(
            api_version="v1",
            kind="Secret",
            metadata=V1ObjectMeta(name=_secret_name(pod_name, volume)),
            type="Opaque",
            string_data=contents[volume],
        )
        for volume in volumes
    )


def _phase_of(pod: V1Pod) -> PodPhase:
    """Reduce a pod's status to the four cases placement acts on.

    A pod with a deletion timestamp is GONE even while its phase still reads Running.
    Deletion is asynchronous -- the object keeps its phase and its ready containers
    through the grace period -- so reading the phase alone would report a pod being torn
    down as dispatchable, and a Turn sent into it would die with it.

    RUNNING requires every container ready, not merely the pod's phase. The runtime and
    the shim are two halves of one Session and neither serves a Turn alone. The
    emptiness check beside the `all()` is why this reads the way it does: `all(())` is
    true, so a pod whose container statuses have not been reported yet -- which is every
    pod for its first moment, and every unschedulable pod for ever, where the list comes
    back as None -- would otherwise be reported ready.

    Unknown is folded into GONE rather than STARTING: it means the node stopped
    reporting, and a pod nobody can hear from is not one to wait for.
    """
    if pod.metadata.deletion_timestamp is not None:
        return PodPhase.GONE
    if pod.status.phase in ("Succeeded", "Failed", "Unknown"):
        return PodPhase.GONE
    statuses = pod.status.container_statuses or []
    if pod.status.phase == "Running" and statuses and all(c.ready for c in statuses):
        return PodPhase.RUNNING
    return PodPhase.STARTING


# Waiting reasons that will not resolve on their own. Measured on this cluster rather
# than recalled: a pod given a tag that does not exist went ContainerCreating ->
# ErrImagePull at 3 s -> ImagePullBackOff at 15 s. So ErrImagePull is the transient
# state and is absent from this set on purpose -- refusing on it would refuse a pod a
# registry blip would have let through. CrashLoopBackOff is absent because
# `restartPolicy: Never` cannot produce it; that pod reaches Failed instead, which
# `_phase_of` calls GONE.
_SCHEDULING_MAY_RESOLVE: Final = frozenset({"Unschedulable"})
"""`PodScheduled=False` reasons that are not this pod's last word.

Unschedulable used to be refused here, and that was the whole of the platform's
concurrent-Session ceiling. Measured on 2026-08-23 against `map-dev`: twenty-four
Sessions submitted at once, and the pods past the node count were refused with
`Unschedulable: 0/2 nodes are available: 2 Too many pods` in the same second they were
created. The cluster runs a Cluster Autoscaler whose ASG goes to eight nodes -- and
Unschedulable is precisely the signal it acts on, so refusing on sight meant the
platform could never use a node it was entitled to. Twelve concurrent Sessions
completed; twenty-four could not, on a cluster with four times that capacity available.

So it is treated as transient, and the bound is what decides: an autoscaled node has to
arrive within `_SCHEDULING_TIMEOUT_SECONDS` or the wait ends with the scheduler's own
message about what was short. A pod that is unschedulable for a permanent reason -- a
node selector nothing matches, a request larger than any node -- is not distinguishable
from this one by its reason or its message, so it costs that whole bound rather than
failing fast. That is the trade taken deliberately: the fast failure was costing real
capacity on every burst, and the slow one costs a wait on a misconfiguration somebody is
about to be told about anyway.

`SchedulingGated` is deliberately NOT here. A gated pod is waiting for a controller to
remove its scheduling gate, and nothing in this platform sets one -- so a gated Session
pod means something outside the platform is holding it, which is a refusal rather than
something to wait out.
"""

_WILL_NOT_START: Final = frozenset(
    {
        "ImagePullBackOff",
        "InvalidImageName",
        "CreateContainerConfigError",
        "RunContainerError",
    }
)

_CONTAINER_MAY_RESOLVE: Final = frozenset({"CreateContainerError"})
"""Container-create failures that a node still coming up produces and then stops.

`CreateContainerError` was in the set above, and on a cluster that can add nodes that
made every autoscaled node unusable. Measured on 2026-08-23: the Cluster Autoscaler
added two nodes in 1m0s, Session pods scheduled onto them, and every one was refused
with `cannot load seccomp profile "/var/lib/kubelet/seccomp/map/session-sandbox.json":
no such file or directory`. The profile is written by a DaemonSet, and a DaemonSet pod
and a Session pod become schedulable on a new node at the same instant -- so the Session
pod wins the race about half the time and the platform declares the node broken.

Treated the way `ErrImagePull` already is, and for the same reason written down there:
kubelet retries container creation every sync, so a file that arrives seconds later
brings the pod up, and refusing on sight refuses a pod that a moment would have fixed.

What it costs, stated rather than left to be discovered: a profile that never arrives --
a genuinely absent file, a DaemonSet that cannot run on that node -- now costs the full
ready bound instead of about fifteen seconds, and the message that named it arrives in
the timeout rather than immediately. The wait carries the last one it saw so the
diagnosis is not lost.

This is a race made survivable and not a race removed. The ordering fix is a node taint
the installer clears once the profile is on disk, which makes "this node can run a
Session" a fact rather than a probability; it needs a nodegroup taint, an identity for
the installer and node-patch permission, and none of that is here.
"""


def _is_scheduled(pod: V1Pod) -> bool:
    """Whether the scheduler has placed this pod on a node.

    Read from `spec.node_name` and not from the `PodScheduled` condition. That condition
    is absent for a pod's first moments -- neither True nor False -- so a reader
    requiring `status == "True"` would call a freshly created pod unscheduled, which is
    right, and would call a pod whose conditions are merely not written yet unscheduled
    too, which restarts the scheduling clock over nothing. The node name is set once, by
    the scheduler, and is the fact the rest of the wait actually needs.
    """
    return bool(pod.spec.node_name)


def _why_it_is_not_scheduled(pod: V1Pod) -> str | None:
    """The scheduler's own account of what this pod is short of, or None.

    Kept for the timeout message. The scheduler says things like "0/2 nodes are
    available: 2 Too many pods", which is the difference between an operator raising a
    node count and an operator hunting a bug -- and it is gone from the object by the
    time anybody reads a log, because a pod that finally schedules overwrites the
    condition.
    """
    for condition in pod.status.conditions or []:
        if condition.type == "PodScheduled" and condition.status == "False":
            return f"{condition.reason}: {(condition.message or '')[:_REASON_CAP]}"
    return None


def _why_a_container_has_not_been_created(pod: V1Pod) -> str | None:
    """A container-create failure that may still resolve, for the timeout message.

    Kept for the same reason `_why_it_is_not_scheduled` is: the wait no longer refuses
    on this, so without carrying it forward a pod that never came up would time out
    saying only "still STARTING" -- and the sentence an operator needs, naming the file
    the node was missing, would be in a kubelet event nobody is looking at.
    """
    reported = list(pod.status.init_container_statuses or []) + list(
        pod.status.container_statuses or []
    )
    for status in reported:
        waiting = status.state.waiting if status.state else None
        if waiting is not None and waiting.reason in _CONTAINER_MAY_RESOLVE:
            detail = (waiting.message or "")[:_REASON_CAP]
            return f"container {status.name} is {waiting.reason}: {detail}"
    return None


def _why_it_will_not_start(pod: V1Pod) -> str | None:
    """The cluster's reason this pod is never going to come up, or None.

    Checked so that a doomed pod is refused in seconds instead of at the end of the
    wait, and -- the part that matters more -- so the refusal carries what to fix. "The
    pod did not start" and "ImagePullBackOff: manifest for map/session-shim:x not found"
    send whoever is on call to two different places.

    Scheduling is read from the pod's conditions and not from a container status,
    because an unschedulable pod has no containers to have a status: measured, the
    condition said Unschedulable with a message naming the shortfall while
    `container_statuses` was None.

    **The init container's status is in a list of its own, and reading only the main
    list is how this misses the failure that actually happens.** Every container in a
    Session pod carries the same image, so the init container is always the first to try
    to pull it. Measured against a digest that does not exist: the init status went
    ErrImagePull at 6 s and ImagePullBackOff at 18 s while all three main statuses sat
    at `PodInitializing` -- so a reader of `container_statuses` alone saw nothing wrong,
    waited out the whole bound, and threw the registry's own account of it away.

    **A container that exited is the other case a phase read cannot see.** Under
    `restartPolicy: Never` kubelet does not bring it back, so the pod stays at phase
    `Running` with one container terminated for ever -- it never becomes `Failed`.
    Measured with the real image: the runtime reached ready and the shim exited 3, and
    the pod sat exactly like that. So the exit is read too, gated on the restart policy,
    because under a policy that restarts, a terminated container is a container between
    attempts and refusing it would refuse a pod that was about to come up.
    """
    for condition in pod.status.conditions or []:
        if condition.type != "PodScheduled" or condition.status != "False":
            continue
        if condition.reason in _SCHEDULING_MAY_RESOLVE:
            continue
        return f"{condition.reason}: {(condition.message or '')[:_REASON_CAP]}"
    reported = list(pod.status.init_container_statuses or []) + list(
        pod.status.container_statuses or []
    )
    for status in reported:
        waiting = status.state.waiting if status.state else None
        if waiting is not None and waiting.reason in _WILL_NOT_START:
            detail = (waiting.message or "")[:_REASON_CAP]
            return f"container {status.name} is {waiting.reason}: {detail}"
    if pod.spec.restart_policy != _NO_RESTARTS:
        return None
    for status in reported:
        ended = status.state.terminated if status.state else None
        if ended is not None and ended.exit_code != 0:
            detail = (ended.message or "")[:_REASON_CAP]
            return (
                f"container {status.name} exited with code {ended.exit_code} "
                f"({ended.reason}) and will not be restarted: {detail}"
            )
    return None


def _will_take_a_pod(node: Any) -> bool:
    """Whether the scheduler would place an ordinary pod on this node.

    Three signals, and any one of them disqualifies. They are written by different
    actors at different moments -- a cordon sets `spec.unschedulable`, the node
    controller sets the Ready condition, and the scheduler itself consults the taints --
    so a count resting on one alone would report a node as capacity that the scheduler
    skips.

    Disqualifying on any rather than requiring all is the direction that under-reports,
    and under-reporting is the safe error for a number somebody reads to decide whether
    the cluster is full: it can cause a look at a cluster that was fine, where over-
    reporting hides one that is not.

    A node missing its status or its conditions is not counted. An absent Ready
    condition is not a ready node, and treating unknown as available is how a number
    like this comes to disagree with the scheduler.
    """
    if getattr(node.spec, "unschedulable", None):
        return False
    for taint in getattr(node.spec, "taints", None) or ():
        if taint.key in _UNSCHEDULABLE_TAINTS:
            return False
    conditions = getattr(getattr(node, "status", None), "conditions", None) or ()
    return any(c.type == "Ready" and c.status == "True" for c in conditions)


def _declared_node_ceiling(deployment: Any) -> int | None:
    """The `--max-nodes-total` the autoscaler runs with, or None if it passes none.

    Searched across every container rather than assuming the first one is the
    autoscaler, and the first match wins. Two containers both passing the flag is not a
    shape this cluster has, and choosing one is a better answer than raising about a
    Deployment somebody else configured.

    Reads the arguments the process was started with, which is the whole point of
    reading the Deployment rather than a file: this is the number in force, not the
    number somebody committed.
    """
    containers = deployment.spec.template.spec.containers or ()
    for container in containers:
        for argument in container.args or ():
            found = _MAX_NODES_TOTAL.match(str(argument))
            if found is not None:
                return int(found.group(1))
    return None


_UNSCHEDULABLE_TAINTS: Final = frozenset(
    {"node.kubernetes.io/unschedulable", "node.kubernetes.io/not-ready"}
)
"""Taints that mean a node will not take an ordinary pod.

Two rather than every taint a cluster can carry, because a taint is not in general a
statement that a node is unusable -- a dedicated nodegroup taints itself so that only
pods tolerating it land there, and counting such a node out would under-report a cluster
deliberately partitioned. These two are set by the node lifecycle itself and mean the
node is not currently taking work.
"""


def _claims_this_session(pod: V1Pod, pod_name: str) -> bool:
    """Whether this pod's own label agrees that it is this Session's pod.

    A consistency check and **not an authentication**, and the difference decides what
    may be built on it. The label is set by whoever created the pod, and this reads it
    back off that same pod, so all it establishes is that the pod's self-description and
    its name agree. Anything that can create a pod in this namespace satisfies it for
    free by setting the label it is about to be checked on. Demonstrated: a `busybox`
    pod named `map-session-<uuid>` and labelled with that uuid is reported RUNNING here,
    and a dispatch would then POST a tenant's prompt and a valid shim bearer to it.

    What it does catch is the accident, which is the common case: a pod parked at a
    colliding name by something that is not this platform, a leftover from a manifest
    applied by hand, a Session's name reused by a different tool. Those carry no label
    or the wrong one and are reported absent rather than dispatched into.

    What actually keeps another tenant's process out of a Session's name is not here and
    cannot be: it is the Session namespace's RBAC -- who may create a pod in it at all
    -- or a pod identity this platform mints and the label cannot forge. Neither exists
    yet, and the gap is a live design question rather than something this function is
    one commit away from closing.

    The label is round-tripped back through `pod_name_for` rather than compared by
    stripping a prefix off the name: the prefix is private to the module that owns the
    naming rule, and a second copy of it here would be free to disagree with it.
    """
    claimed = (pod.metadata.labels or {}).get(_SESSION_LABEL)
    if claimed is None:
        return False
    try:
        return pod_name_for(SessionId(UUID(claimed))) == pod_name
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class KubernetesPodRunner:
    """One Session, one pod, in one namespace of one cluster.

    The namespace is a field and never a method parameter. That is the whole of this
    adapter's blast radius: no caller can address a pod outside the namespace this
    object was built for, and the pod name it is given goes into the API as a typed path
    parameter rather than into a selector string, so there is nothing to inject into
    either.

    `manifest` is the parsed contents of the repository's Session-pod manifest, read
    once by `from_manifest_file` so a missing or unparseable one fails before any
    Session rather than at the first placement. It is never mutated here.

    `token_key` is kept out of the `repr` and that is not tidiness. It derives every
    Session's shim bearer token, and the bearer is the shim route's only check, so one
    rendering of this object discloses every token the platform will ever mint. A
    dataclass `repr` is rendered in more places than a reader expects: pytest prints the
    local variables of a failing frame, and an error reporter that captures frame locals
    ships them off the machine. The live-tier test that starts a real pod holds this
    object in a local and fails today by design, so with the key in the `repr` a CI run
    of that tier would print the deployment's real signing key into a build log.
    Excluded at the field so no call site has to remember.
    """

    namespace: str
    token_key: bytes = field(repr=False)
    manifest: Mapping[str, Any]

    @classmethod
    def from_manifest_file(
        cls, path: Path, *, namespace: str, token_key: bytes
    ) -> KubernetesPodRunner:
        """Read the Session-pod manifest from disk and build a runner around it.

        `safe_load`, not `load`: this file is deployment input and a loader that can
        construct arbitrary Python objects has no business reading it.

        An empty signing key is refused rather than used. HMAC under an empty key is a
        public function of the Session id, so every pod on the cluster could derive
        every other Session's token, and the shim's bearer check -- its only check --
        would pass for all of them. Refused here so the process does not start, because
        a running process with that hole looks exactly like a working one.
        """
        if not token_key:
            raise PodNotStarted(
                "a Session pod needs a non-empty signing key: an empty one derives a "
                "shim token every pod can also derive"
            )
        parsed = yaml.safe_load(path.read_text())
        if not isinstance(parsed, dict):
            raise PodNotStarted(f"{path} does not parse as a Kubernetes manifest")
        return cls(namespace=namespace, token_key=token_key, manifest=parsed)

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        """Bring this Session's pod up if it is absent, and report where it got to.

        The name is re-derived from the configuration before anything is created. The
        port hands both over as separate arguments, so they can disagree, and a pod
        started from another Session's compiled documents would run under the wrong
        Permission Profile and the wrong Tool Gateway identity while `Placement.locate`
        still found it under this Session's name. Nothing downstream would notice, which
        is why this is the first statement here rather than a check somewhere later.

        A pod that already exists is not recreated, whatever state it is in. A pod that
        has finished -- failed or succeeded -- is reported GONE straight away instead of
        waited on: it is not absent, so this call has nothing to start, and reviving a
        Session is `release` followed by `place`.

        **A pod already at this name that does not claim this Session is refused, not
        adopted.** `phase_of` has always read the Session label back; this did not, and
        the two disagreeing is worse than either answer alone. Measured: with something
        else squatting the name, `ensure` found it, skipped the create -- so the
        Session's real pod was never made and its shim token never minted -- and
        returned RUNNING, while `locate` on the same Session answered ABSENT. The caller
        recorded a placed Session that every later Turn refuses as undeliverable, and
        the two diagnostics contradicted each other. The check is the same one, in the
        same words, so the two entry points cannot drift apart again.

        **A pod this call created and that then fails to come up is deleted before the
        refusal.** Without that, a failed placement left the pod `Pending` for ever and
        all three Secrets behind it -- including the shim bearer for a Session that
        never started -- with nothing on any path to reap them. Measured against the
        cluster with an unpullable digest: pod present, three Secrets present, and a
        retry refused the same way. Only a pod this call created is cleaned up: a pod
        that was already here and is merely slow belongs to a Session that may still be
        coming up, and deleting it on a timeout would destroy a live Session to tidy up
        after a caller.
        """
        expected = pod_name_for(compiled.session_id)
        if pod_name != expected:
            raise PodNotStarted(
                f"asked to start {pod_name} from the compiled configuration of session "
                f"{compiled.session_id}, whose only pod is {expected}"
            )
        async with _core_api() as core:
            existing = await self._read(core, pod_name)
            if existing is not None:
                if not _claims_this_session(existing, pod_name):
                    raise PodNotStarted(
                        f"a pod called {pod_name} is already in {self.namespace} and "
                        f"does not carry this Session's {_SESSION_LABEL} label, so it "
                        "is not this platform's to adopt or to replace"
                    )
                if _phase_of(existing) is PodPhase.GONE:
                    return PodPhase.GONE
                return await self._wait_for_both_halves(core, pod_name)
            await self._create(core, pod_name, compiled)
            try:
                return await self._wait_for_both_halves(core, pod_name)
            except BaseException:
                # Not suppressed if it fails in turn: the original refusal is this
                # frame's active exception, so a failure here is chained to it and
                # reaches the caller with both, rather than replacing the diagnosis
                # with a story about the cleanup.
                await self._delete_pod_and_secrets(core, pod_name)
                raise

    async def phase_of(self, pod_name: str) -> PodPhase:
        """Where this Session's pod is, asked of the cluster.

        A pod at the right name whose own label does not claim this Session is reported
        ABSENT rather than described, which catches the pod parked at a colliding name
        by something that is not this platform -- a hand-applied manifest, a name reused
        by another tool.

        **That is a consistency check and not a tenancy control**, and this docstring
        said otherwise until it was measured. It read: "anything in this namespace can
        create a pod called `map-session-<uuid>`, and reporting it RUNNING would have
        the control plane dispatch a tenant's Turn into a process it knows nothing
        about" -- naming exactly the threat the check does not stop, because whatever
        creates that pod sets the label it is about to be checked on. Demonstrated with
        a `busybox` pod: RUNNING from here, and a dispatch would then POST the tenant's
        prompt and a valid shim bearer into it. `_claims_this_session` says what does
        close it and why it is not here. Nothing downstream may read this answer as an
        authentication.
        """
        async with _core_api() as core:
            pod = await self._read(core, pod_name)
            if pod is None or not _claims_this_session(pod, pod_name):
                return PodPhase.ABSENT
            return _phase_of(pod)

    async def remove(self, pod_name: str) -> None:
        """Delete the pod and the three Secrets that exist only for it.

        Absent is success at every step, so a repeated stop and a stop of a Session
        whose pod already went away both succeed -- which is what `release` promises its
        caller.

        The secrets are deleted here as well as being owned by the pod. The owner
        reference alone would collect them, but only once the pod object is gone, which
        is after its grace period -- measured at 32 s on this cluster. Deleting them
        outright makes the token and the compiled documents stop existing when the
        Session does rather than half a minute later.
        """
        async with _core_api() as core:
            await self._delete_pod_and_secrets(core, pod_name)

    async def placed_pods(self) -> Sequence[PlacedPod]:
        """Every pod in this namespace that claims to be some Session's.

        A label selector rather than a name, which is the one read here that is not
        addressed to a single object -- so it is worth saying what bounds it. The
        selector is the constant `map.session-id` with nothing interpolated into it, and
        the namespace is this object's field, so the widest thing this can return is
        every labelled pod of the one namespace this adapter was built for.

        A pod is described only when `_claims_this_session` agrees that its label and
        its name name the same Session, which is the same consistency check `phase_of`
        applies, in the same words, for the same reason: a pod squatting the naming
        pattern without the matching label is not this platform's to describe and must
        not be handed to anything that deletes. That check is not an authentication and
        nothing built on this answer may read it as one -- whatever creates a pod in
        this namespace can satisfy it by setting the label -- but the caller of this is
        a sweep that hands pods back, and the failure mode of a forged label here is a
        pod deleted rather than a Turn dispatched into a stranger.

        **A pod the API server has not stamped with a creation timestamp is left out of
        the answer entirely.** Its age is the guard that lets a sweep delete without
        trusting any store, so a pod with no age is one no sweep can judge; omitting it
        keeps it alive, which is the safe direction, where inventing an age for it
        would be a deletion resting on a value this module made up. In practice the API
        server sets the field at admission and this branch is unreachable -- it is here
        because the generated model types it as optional and the honest answer to an
        absent one is not "assume it is old".
        """
        async with _core_api() as core:
            listed = await core.list_namespaced_pod(
                namespace=self.namespace, label_selector=_SESSION_LABEL
            )
        placed: list[PlacedPod] = []
        for pod in listed.items:
            name = pod.metadata.name
            label = (pod.metadata.labels or {}).get(_SESSION_LABEL)
            if label is None or not _claims_this_session(pod, name):
                continue
            stamped = pod.metadata.creation_timestamp
            if stamped is None:
                continue
            placed.append(
                PlacedPod(
                    session_id=SessionId(UUID(label)),
                    phase=_phase_of(pod),
                    created_at_ms=int(stamped.timestamp() * 1000),
                )
            )
        return placed

    async def node_headroom(self) -> NodeHeadroom:
        """Count the nodes that can take a pod, and read the ceiling that binds them.

        **Neither number is allowed to fail the caller.** This is read when something is
        already wrong, and an endpoint that returns nothing because an optional field
        was refused is an endpoint nobody can use during an incident. So each half is
        asked for separately and each degrades to `None` on its own -- which is also why
        they are two reads: they are granted by different authority, so on a cluster
        holding one binding and not the other, one number still arrives.

        A refusal is logged with the status the API server gave and the authority the
        read needs. That log line is the only place the difference between "the cluster
        refused" and "this build cannot ask" survives -- `None` reaches a reader
        identically for both -- and without it the next person re-derives which RBAC a
        missing field implies.

        Creates nothing, changes nothing, deletes nothing. Both calls are reads.
        """
        return NodeHeadroom(
            schedulable=await self._schedulable_nodes(),
            ceiling=await self._autoscaler_ceiling(),
        )

    async def _schedulable_nodes(self) -> int | None:
        """How many nodes would accept an ordinary pod now, or None if the read failed.

        **A cluster-scoped read, and the only one this platform makes.** `nodes` is not
        a namespaced resource, so no Role in any namespace can grant it and it needs a
        ClusterRole with `list` alone. Worth knowing before reading the refusal branch:
        a 403 here is the expected answer on a deployment whose ClusterRole was never
        applied, and says nothing about the health of the cluster.

        This is not "nodes with room on them". A node already at its pod limit is still
        counted, because the per-node ceiling here is an ENI address count this read
        cannot see -- and a number that quietly meant two different things would be
        worse than one that plainly means the simpler thing.
        """
        try:
            async with _core_api() as core:
                listed = await core.list_node()
        except ApiException as refused:
            _LOG.warning(
                "not counting schedulable nodes: the API server answered %s. `nodes` "
                "is cluster-scoped, so this read needs a ClusterRole granting `list` "
                "on nodes and no namespaced Role can grant it. Reporting unknown.",
                refused.status,
            )
            return None
        return sum(1 for node in listed.items if _will_take_a_pod(node))

    async def _autoscaler_ceiling(self) -> int | None:
        """The most nodes the autoscaler will add unattended, or None if unreadable.

        **Read from the running Deployment's own arguments, never from a manifest on
        disk.** The number a reader needs is the one the live process is using, and read
        this way it cannot drift from the flag because it *is* the flag. A file read of
        the autoscaler's manifest would agree in the test suite and answer nothing in
        the cluster, where that file is not on the container's filesystem -- the worse
        of the two failure modes, because it looks correct locally.

        Three distinguishable ways this comes back `None`, all logged: the read was
        refused, no such Deployment is there, or it is there and passes no such flag.
        The third is the interesting one -- an autoscaler running without the flag is
        bounded only by its nodegroup's own maximum, so the ceiling in force genuinely
        is not this number, and publishing the nodegroup's instead would publish a bound
        that does not bind.
        """
        try:
            async with _apps_api() as apps:
                deployment = await apps.read_namespaced_deployment(
                    name=_AUTOSCALER_DEPLOYMENT, namespace=self.namespace
                )
        except ApiException as unreadable:
            _LOG.warning(
                "not reporting a node ceiling: reading Deployment %s/%s answered %s. "
                "404 means no autoscaler is deployed there; 403 means this process "
                "needs `get` on that one Deployment by name. Reporting unknown.",
                self.namespace,
                _AUTOSCALER_DEPLOYMENT,
                unreadable.status,
            )
            return None
        ceiling = _declared_node_ceiling(deployment)
        if ceiling is None:
            _LOG.warning(
                "not reporting a node ceiling: Deployment %s/%s passes no "
                "--max-nodes-total, so nothing caps how far it will grow the cluster "
                "but the nodegroup's own maximum. Reporting unknown.",
                self.namespace,
                _AUTOSCALER_DEPLOYMENT,
            )
        return ceiling

    async def _delete_pod_and_secrets(
        self, core: client.CoreV1Api, pod_name: str
    ) -> None:
        """Delete the pod and its Secrets over a client somebody else opened.

        Split out from `remove` because `ensure` needs the same six deletions inside the
        client it already holds -- and because a second copy of them would be free to
        forget one, which for a Secret named `-shim-token` is a leaked bearer.

        Every deletion tolerates absent, so this is safe to call over a partly-created
        Session: the run that died between minting the Secrets and creating the pod
        leaves the Secrets and no pod, and this clears exactly that.
        """
        with _absent_is_success():
            await core.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
        for volume in _SECRET_FILES:
            with _absent_is_success():
                await core.delete_namespaced_secret(
                    name=_secret_name(pod_name, volume), namespace=self.namespace
                )

    async def _create(
        self, core: client.CoreV1Api, pod_name: str, compiled: CompiledConfig
    ) -> None:
        """Create the secrets, then the pod, then make the pod own the secrets.

        Secrets first: a pod whose secret volume has nothing behind it makes kubelet
        retry the mount indefinitely, so it would never reach Running and the reason
        would be a FailedMount event nobody is reading.

        An already-existing Secret is left alone rather than replaced. A compiled
        configuration is immutable for a Session's whole life, so a Secret bearing this
        Session's name already holds this content -- it is the residue of an attempt
        that died between creating the secrets and creating the pod. The one case that
        gets wrong is a signing-key rotation inside that window, which would leave the
        old token mounted; the Session is not serving in that window, and `remove`
        clears the secrets with the pod.

        The owner reference is set afterwards because the pod's uid does not exist
        before the pod does. Its job is not the ordinary teardown -- `remove` deletes
        these explicitly -- but every path `remove` is not on: an operator deleting the
        pod by hand, a namespace teardown, a test sweep. A patch that fails leaves the
        secrets to those explicit deletes, which is why both mechanisms are here.
        """
        volumes = _secret_volumes(self.manifest)
        for secret in _secrets_for(pod_name, volumes, compiled, self.token_key):
            try:
                await core.create_namespaced_secret(
                    namespace=self.namespace, body=secret
                )
            except ApiException as err:
                if err.status != _ALREADY_EXISTS:
                    raise
        # The stub types `body` as `V1Pod`, which is narrower than what the API takes:
        # the client serializes whatever it is handed, and the library's own
        # `utils.create_from_dict` passes a plain mapping here too. Ignored rather than
        # cast, because a cast would assert this *is* a `V1Pod` and it is not -- it is
        # the manifest, which is the thing that must reach the cluster unaltered.
        # Verified against the real API server with `dry_run="All"`.
        pod = await core.create_namespaced_pod(
            namespace=self.namespace,
            body=_pod_for(self.manifest, pod_name, compiled),  # type: ignore[arg-type]
        )
        owner = [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "name": pod_name,
                "uid": pod.metadata.uid,
                "controller": False,
                "blockOwnerDeletion": False,
            }
        ]
        for volume in volumes:
            with _absent_is_success():
                await core.patch_namespaced_secret(
                    name=_secret_name(pod_name, volume),
                    namespace=self.namespace,
                    body={"metadata": {"ownerReferences": owner}},
                )

    async def _wait_for_both_halves(
        self, core: client.CoreV1Api, pod_name: str
    ) -> PodPhase:
        """Poll until the pod is running, terminal, or out of time.

        Bounded on purpose, and the bound is the loop's own condition rather than a
        break somewhere inside it. An unbounded wait on a pod is a hang, and a hang
        reads to whoever is watching exactly like a slow image pull.

        A doomed pod is refused as soon as the cluster says why, so the common failure
        -- an image that is not there -- costs fifteen seconds rather than three
        minutes. The timeout carries the last phase seen, because "still starting after
        180 s" and "gone after 180 s" are different problems.
        """
        deadline = time.monotonic() + _SCHEDULING_TIMEOUT_SECONDS
        scheduled = False
        phase = PodPhase.ABSENT
        waited_for_a_node = ""
        not_created = ""
        while time.monotonic() < deadline:
            pod = await self._read(core, pod_name)
            if pod is None:
                raise PodNotStarted(
                    f"the pod {pod_name} was created and is no longer there"
                )
            if not scheduled and _is_scheduled(pod):
                # The clock restarts here rather than continuing, so the ready bound is
                # measured from the moment there was a node to be ready on. Without the
                # reset, a pod that waited five minutes for an autoscaled node would
                # arrive with its whole image-pull window already spent.
                scheduled = True
                deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
            elif not scheduled:
                waited_for_a_node = _why_it_is_not_scheduled(pod) or waited_for_a_node
            phase = _phase_of(pod)
            if phase in (PodPhase.RUNNING, PodPhase.GONE):
                return phase
            reason = _why_it_will_not_start(pod)
            if reason is not None:
                raise PodNotStarted(f"the pod {pod_name} will not start: {reason}")
            not_created = _why_a_container_has_not_been_created(pod) or not_created
            await asyncio.sleep(_POLL_SECONDS)
        if not scheduled:
            # A different sentence from the one below, because it sends the reader
            # somewhere else entirely: no node took this pod, which is capacity, where
            # "still STARTING" is a pod that has a node and cannot come up on it.
            raise PodNotStarted(
                f"the pod {pod_name} was never scheduled within "
                f"{_SCHEDULING_TIMEOUT_SECONDS:.0f}s"
                + (f": {waited_for_a_node}" if waited_for_a_node else "")
            )
        raise PodNotStarted(
            f"the pod {pod_name} was still {phase.value} after "
            f"{_READY_TIMEOUT_SECONDS:.0f}s"
            + (f": {not_created}" if not_created else "")
        )

    async def _read(self, core: client.CoreV1Api, pod_name: str) -> V1Pod | None:
        """The named pod, or None where the cluster says there is none.

        The name travels as the typed path parameter of a generated method, never
        interpolated into a label or field selector, so a pod name cannot be written to
        widen this read past the one object it names.
        """
        try:
            return await core.read_namespaced_pod(
                name=pod_name, namespace=self.namespace
            )
        except ApiException as err:
            if err.status == _NOT_FOUND:
                return None
            raise
