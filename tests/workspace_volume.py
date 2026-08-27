"""The Session workspace volume, retargeted at a namespace that is not `map-dev`.

Every tier that places a real `session-pod.yaml` outside the platform namespace needs
this. The manifest mounts a claim called `session-workspaces`, a claim is namespaced,
and a pod may only mount one in its own namespace -- so a tier that creates its own
namespace and applies the manifest unmodified gets a pod that never schedules. The
event says `persistentvolumeclaim "session-workspaces" not found`, which reads as the
scheduler refusing the pod rather than as an object nobody created, and the pod sits in
`Pending` until something times out.

**Why this is shared rather than written per tier.** Three rules have to hold together,
and each of them is a way to damage the real volume rather than merely fail:

- The PersistentVolume is **renamed** per namespace. It is cluster-scoped, so a fixed
  name collides with `map-session-workspaces` -- the volume every tenant's workspace is
  bound to -- and a teardown would then delete it.
- The claim **keeps** its name. `session-pod.yaml` names `session-workspaces`, and the
  whole point of these tiers is to place that manifest unmodified.
- `volumeName` is **repointed** at the renamed volume. The claim in the manifest names
  `map-session-workspaces` explicitly, so a copy that changed only the namespace would
  be a second claim on the real volume.

Written once because a tier that gets any of the three wrong does not fail loudly; it
either sits Pending or reaches for production's volume.

The documents are read out of `deploy/k8s/session-vfs.yaml` rather than written here.
The field that must be right is the `volumeHandle` -- its `s3files:` prefix is what
stops the driver mounting the file system as plain EFS and hanging the pod in
`ContainerCreating` with no error naming the cause -- and a second copy of it here is
one that keeps working after somebody repoints the real one, which is exactly when
these tiers would stop testing what production runs.

Applying is left to the caller. One tier drives the asyncio Kubernetes client, another
shells out to `kubectl`; both are correct in their own module and neither is worth
converting to the other's idiom for the sake of sharing three dictionary edits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

_MANIFEST: Final = (
    Path(__file__).resolve().parents[1] / "deploy" / "k8s" / "session-vfs.yaml"
)


def documents_for(namespace: str) -> tuple[Any, Any]:
    """The PersistentVolume and claim from `session-vfs.yaml`, aimed at `namespace`.

    Returns them in that order, unapplied. `claimRef` is dropped from the volume: the
    manifest's copy may carry a binding to the real claim, and a volume that arrives
    already spoken for never binds to the one made here.

    Typed `Any` rather than `dict[str, Any]`, which is what they are. One caller passes
    them straight to the Kubernetes client, whose generated stubs declare `body` as
    `V1PersistentVolume` -- a parsed manifest is a mapping and the client accepts one at
    runtime, so a precise type here buys a cast at that call site and nothing else.

    Raises if the namespace is the platform's own. Retargeting `map-dev` would rename
    the real volume out from under every tenant workspace and hand the caller a
    teardown that deletes it -- and a tier running in `map-dev` needs none of this,
    because bootstrap has already created the claim there.
    """
    if namespace == platform_namespace():
        raise AssertionError(
            f"{namespace!r} is the platform namespace, whose claim bootstrap already "
            "creates; retargeting it would rename the volume every tenant's workspace "
            "is bound to"
        )

    text = _MANIFEST.read_text()
    volume = next(
        one for one in yaml.safe_load_all(text) if one["kind"] == "PersistentVolume"
    )
    claim = next(
        one
        for one in yaml.safe_load_all(text)
        if one["kind"] == "PersistentVolumeClaim"
    )

    volume["metadata"]["name"] = f"{volume['metadata']['name']}-{namespace}"
    volume["spec"].pop("claimRef", None)
    claim["metadata"]["namespace"] = namespace
    claim["spec"]["volumeName"] = volume["metadata"]["name"]
    return volume, claim


def platform_namespace() -> str:
    """The namespace the manifest's own claim is written for.

    Read from the file rather than named here, so this stays true if the platform
    namespace is ever renamed -- the guard above is only worth having if it cannot
    drift away from the thing it protects.
    """
    claim = next(
        one
        for one in yaml.safe_load_all(_MANIFEST.read_text())
        if one["kind"] == "PersistentVolumeClaim"
    )
    return str(claim["metadata"]["namespace"])
