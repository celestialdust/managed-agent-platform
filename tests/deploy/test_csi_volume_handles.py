"""Every `efs.csi.aws.com` volumeHandle in this repository names its file system type.

The EFS CSI driver serves two AWS services from one driver name, and it decides which
one from the volumeHandle rather than from a volumeAttribute or a StorageClass
parameter. The form is `{fsType}:{fileSystemId}:{mountPath}:{accessPointId}`, and a
first token that does not parse as a type falls back to `efs` -- a deliberate backward
compatibility path for handles written before the field existed
(kubernetes-sigs/aws-efs-csi-driver v3.4.1, `pkg/driver/node.go`, `parseVolumeId`).

That fallback is silent, and its failure is expensive. An S3 Files file system reached
through an `efs`-typed handle makes the driver build the EFS DNS name
`{fs_id}.efs.{region}.amazonaws.com` for a file system whose name is really
`{az_id}.{fs_id}.s3files.{region}.on.aws`. Nothing rejects the manifest, the PVC binds,
and the pod sits in `ContainerCreating` until somebody kills it -- with the reason five
layers down in the CSI node driver's log rather than anywhere `kubectl describe pod`
looks. It cost a working day to find once.

So the property asserted here is not which type any particular volume uses. It is that
every handle SAYS, so that the fallback can never be reached by accident. A file system
type this repository does not use yet fails here on purpose: adding one should be a
decision somebody writes down, not a default nobody chose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_MANIFESTS = Path(__file__).resolve().parents[2] / "deploy" / "k8s"

# The driver's own set, as of v3.4.1. Neither is a guess: `/sbin/mount.efs` and
# `/sbin/mount.s3files` are both present in the driver image, and each is a thin
# wrapper selecting its own `*-utils.conf`.
_KNOWN_TYPES = frozenset({"efs", "s3files"})

_HANDLE = re.compile(r"^(?P<fs_type>[a-z0-9]+):(?P<rest>fs-[0-9a-f]+:.*)$")


def _csi_volumes() -> list[tuple[Path, str, dict[str, object]]]:
    """Every PersistentVolume in `deploy/k8s/` whose source is a CSI driver.

    Read as YAML rather than grepped, so a handle that is commented out does not count
    and one written as a block scalar does.
    """
    found: list[tuple[Path, str, dict[str, object]]] = []
    for manifest in sorted(_MANIFESTS.glob("*.yaml")):
        for document in yaml.safe_load_all(manifest.read_text()):
            if not isinstance(document, dict):
                continue
            if document.get("kind") != "PersistentVolume":
                continue
            csi = document.get("spec", {}).get("csi")
            if isinstance(csi, dict):
                name = document.get("metadata", {}).get("name", "<unnamed>")
                found.append((manifest, str(name), csi))
    return found


def test_the_repository_declares_at_least_one_csi_volume() -> None:
    """The guard below is vacuous if nothing matches, so say so out loud.

    Every case here iterates a collection read off disk. If the workspace mount were
    renamed or moved out of `deploy/k8s/`, those cases would pass over an empty list
    and report green while asserting nothing at all.
    """
    assert _csi_volumes(), (
        f"no PersistentVolume with a csi source under {_MANIFESTS}; either the mount "
        "moved and this file must follow it, or the guard below is asserting nothing"
    )


@pytest.mark.parametrize(
    ("manifest", "name", "csi"),
    [pytest.param(m, n, c, id=n) for m, n, c in _csi_volumes()],
)
def test_an_efs_driver_handle_names_a_file_system_type(
    manifest: Path, name: str, csi: dict[str, object]
) -> None:
    if csi.get("driver") != "efs.csi.aws.com":
        served_by = csi.get("driver")
        pytest.skip(f"{name} is served by {served_by!r}, not the EFS CSI driver")

    handle = str(csi.get("volumeHandle", ""))
    matched = _HANDLE.match(handle)
    assert matched, (
        f"{manifest.name}: PersistentVolume {name} has volumeHandle {handle!r}, which "
        "names no file system type. The driver will fall back to `efs` in silence and "
        "the pod will hang in ContainerCreating. Write it as "
        "`{fsType}:{fileSystemId}:{mountPath}:{accessPointId}`."
    )
    fs_type = matched.group("fs_type")
    assert fs_type in _KNOWN_TYPES, (
        f"{manifest.name}: PersistentVolume {name} names file system type {fs_type!r}, "
        f"which this driver does not serve (it serves {sorted(_KNOWN_TYPES)}). A "
        "handle "
        "whose first token does not parse as a type is treated as `efs`, so this would "
        "mount the wrong service without any error."
    )
