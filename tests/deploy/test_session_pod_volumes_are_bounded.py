"""Every scratch volume a Session pod gets is bounded, and the numbers fit a node.

A Session pod is one tenant's Session and nobody else's -- its workspace is an
`emptyDir`, so two Sessions of one tenant cannot see each other's files either. That
isolation is what the pod-per-Session design buys, and an unbounded `emptyDir` is the
one hole in it: the volume draws on the node's disk, so a Session writing without limit
does not exhaust its own quota, it exhausts the node and evicts whatever else was
scheduled there. Filesystem isolation is total and disk isolation is absent, which reads
as a stable platform until the first tenant writes a large file.

Graded here rather than in `deploy/platform.py` because it is a fact about the manifest
that is true before anything is applied, and because `sizeLimit` is exactly the kind of
key a later edit drops without noticing -- it changes no behaviour until a node fills.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_POD = _ROOT / "deploy" / "k8s" / "session-pod.yaml"

_NODE_EPHEMERAL_KIB: Final = 41865196
"""What a node in `map-dev` reports as `status.capacity.ephemeral-storage`.

Measured, not chosen. Pinned so the budget below is checked against the machine the
pods actually land on; a nodegroup moved to a smaller instance type fails this file
rather than failing under load.
"""

_PODS_PER_NODE: Final = 17
"""What a node reports as `status.capacity.pods` -- the ENI-derived cap on t3.medium."""


def _pod() -> dict[str, Any]:
    for document in yaml.safe_load_all(_POD.read_text()):
        if document and document.get("kind") == "Pod":
            spec: dict[str, Any] = document["spec"]
            return spec
    raise AssertionError(f"{_POD} declares no Pod")


def _scratch_volumes() -> list[tuple[str, dict[str, Any]]]:
    """Every `emptyDir` volume, by name. Secret volumes are not this file's business:
    their size is bounded by the Secret, which the API server caps at 1Mi."""
    return [
        (volume["name"], volume["emptyDir"])
        for volume in _pod()["volumes"]
        if "emptyDir" in volume
    ]


def _kib(size: str) -> int:
    """One Kubernetes quantity as KiB. Only the suffixes this manifest uses."""
    for suffix, factor in (("Gi", 1024 * 1024), ("Mi", 1024), ("Ki", 1)):
        if size.endswith(suffix):
            return int(float(size[: -len(suffix)]) * factor)
    raise AssertionError(f"unhandled quantity {size!r}")


def test_the_manifest_really_declares_scratch_volumes() -> None:
    """The guard on the guard: an empty collection makes every case below vacuous."""
    assert len(_scratch_volumes()) >= 4


@pytest.mark.parametrize("name", [name for name, _ in _scratch_volumes()])
def test_every_scratch_volume_declares_a_size_limit(name: str) -> None:
    """Parametrized per volume because each one is a separate decision to bound it.

    Written as one assertion over the list, a volume added later without a limit would
    be a case that never existed rather than a case that failed.
    """
    limits = dict(_scratch_volumes())
    assert limits[name].get("sizeLimit"), (
        f"volume {name} is an emptyDir with no sizeLimit, so it is bounded only by the "
        f"node's disk. A Session filling it evicts the pods sharing its node."
    )


def test_the_volumes_one_node_full_of_pods_could_claim_fit_on_that_node() -> None:
    """The sum, not each limit. A per-volume limit bounds one Session; only the sum
    bounds the node, and the node is what the neighbours share.

    Held to 60% deliberately. The remainder is not slack -- it is the container images,
    the writable layers, and the kubelet's own eviction threshold, none of which appear
    in this arithmetic and all of which come out of the same disk.
    """
    per_pod = sum(_kib(str(spec["sizeLimit"])) for _, spec in _scratch_volumes())
    worst_case = per_pod * _PODS_PER_NODE

    assert worst_case < _NODE_EPHEMERAL_KIB * 0.6, (
        f"{_PODS_PER_NODE} pods x {per_pod}KiB = {worst_case}KiB against a node's "
        f"{_NODE_EPHEMERAL_KIB}KiB. Lower a sizeLimit or raise the instance type; "
        f"do not raise this fraction without saying what pays for the images."
    )
