"""Every scratch volume a Session pod gets is bounded, and the numbers fit a node.

An unbounded `emptyDir` is the one hole in a pod-per-Session design: the volume draws on
the node's disk, so a Session writing without limit does not exhaust its own quota, it
exhausts the node and evicts whatever else was scheduled there. Filesystem isolation is
total and disk isolation is absent, which reads as a stable platform until the first
tenant writes a large file.

The workspace is deliberately NOT graded here any more, and its absence is the point
rather than a gap. It is a PersistentVolumeClaim now (ADR-035), so a Session's writes
land in a bucket over NFS and spend no node ephemeral storage at all -- there is
nothing for a `sizeLimit` to bound. What that leaves unbounded is a tenant's spend in
the bucket, which this file cannot see and must not pretend to: it is bounded where the
writes are billed. `tests/pod/test_pod_holds_no_credential.py` is what pins the
workspace to that backing, so a workspace that quietly became an `emptyDir` again fails
there rather than slipping silently past this file's `emptyDir`-only walk.

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

_NODE_EPHEMERAL_KIB: Final = 83000000
"""What a node in `map-dev` will report as `status.capacity.ephemeral-storage`.

Derived rather than measured, unlike every figure that stood here before it, because
the node it describes does not exist yet: the nodegroup now declares an 80 GiB root
volume and no node of that size has run. The 40 GiB node being replaced reported
41865196Ki, filesystem overhead already taken out, so twice that is the honest estimate
and this is twice that rounded down -- under-reading the disk keeps the budget below
conservative while the number is still a derivation. Replace it with a real reading
from `kubectl get node` once a replacement node has come up, and expect that reading to
be slightly larger than this rather than smaller.
"""

_PODS_PER_NODE: Final = 22
"""How many Session pods one node can hold, which is a memory figure and used to be a
slot count.

`status.capacity.pods` was the right multiplier while a Session pod declared no
requests: the scheduler packed slots until the ENI-derived cap ran out, 17 of them on a
`t3.medium`. A pod now declares 640Mi across its two containers, so what fills first is
memory -- an `m6i.xlarge` allocates about 14 GiB and reaches 22 Session pods, well
under its slot cap of 58, which is a ceiling nothing can climb to any more.

Its CPU requests bind tighter still, at 15 per node. The budget below is held against
22 anyway, because memory is the bound the machine imposes and CPU is one a manifest
can lower -- holding it against 15 would mean this file failing the day somebody trims
a request. ADR-039 and ADR-040.
"""


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


_TODAY_NODE_EPHEMERAL_KIB: Final = 36630100
"""What a node in `map-dev` reports TODAY as `status.allocatable.ephemeral-storage`.

Measured on the `t3.medium` nodes the cluster is actually running, not derived:
37,509,222,746 bytes, which is 34.9 GiB. Allocatable rather than capacity, and the two
answer different questions -- capacity is the partition, allocatable is what the
kubelet will let pods claim before its own `nodefs.available: 10%` eviction threshold
and its reservations come out. A budget held against capacity spends the threshold it
is trying to stay under.

It sits beside `_NODE_EPHEMERAL_KIB` rather than replacing it because the two describe
two machines. That one is the node ADR-040 brings and no node of that size has run;
this one is the node a Session is placed on this afternoon. A sizeLimit has to fit
both, and only this one can fail today.
"""

_TODAY_PODS_PER_NODE: Final = 17
"""How many pods one of today's nodes can hold, which is the ENI-derived slot cap.

`status.capacity.pods` on a `t3.medium`. Used rather than the memory-derived figure
beside it because this is the number that binds first on this machine and because it
is the larger of the two readings a pod count can take here -- a budget held against
the smaller one passes by assuming the node stays under-packed.
"""


def test_the_manifest_really_declares_scratch_volumes() -> None:
    """The guard on the guard: an empty collection makes every case below vacuous.

    Three, not the four it was: `workspace` left this set when it became a claim on
    persistent storage. Lowered rather than deleted -- the floor is what keeps a walk
    that has stopped matching the manifest from passing as a clean tree.
    """
    assert len(_scratch_volumes()) >= 3


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


def test_those_volumes_also_fit_the_node_a_session_is_placed_on_today() -> None:
    """The same sum against the machine that exists, which is the smaller of the two.

    The case above is held against an 80 GiB node at 22 pods, and the docstring on that
    figure says plainly that no node of that size has run: it is the nodegroup ADR-040
    declares and terraform has not applied, blocked on a nodegroup replacement nobody
    has approved. So that case cannot fail for a Session placed this afternoon, and a
    volume sized to pass it can still evict a pod on a `t3.medium` tonight.

    Held to the same 60%, and the remainder is the same remainder: the container image
    -- one copy per node, not seventeen -- the container logs the kubelet rotates at
    10Mi x 5 per container, and the node's own writes. None of the three is in this
    arithmetic and all three come out of the same disk.

    The failure this guards is not a slow Session. Enforcement here is kubelet's
    periodic `du` with no filesystem-quota feature gate, so a volume over its limit
    EVICTS the pod, and `restartPolicy: Never` ends the Session where it stood.
    """
    per_pod = sum(_kib(str(spec["sizeLimit"])) for _, spec in _scratch_volumes())
    worst_case = per_pod * _TODAY_PODS_PER_NODE

    assert worst_case < _TODAY_NODE_EPHEMERAL_KIB * 0.6, (
        f"{_TODAY_PODS_PER_NODE} pods x {per_pod}KiB = {worst_case}KiB against a "
        f"t3.medium's {_TODAY_NODE_EPHEMERAL_KIB}KiB allocatable. This is the node "
        f"running today; lower a sizeLimit rather than waiting on ADR-040."
    )
