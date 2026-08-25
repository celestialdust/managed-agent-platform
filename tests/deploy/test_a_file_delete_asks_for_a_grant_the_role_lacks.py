"""The uploaded-file delete path, graded against the grant it actually runs under.

Tier 1, local, no cluster. This reads two documents that nobody reads together: the
policy the control plane assumes, and the adapter method its delete path calls. Each is
unremarkable alone, which is why the gap between them survived -- the policy grants what
uploads need and the code deletes what a tenant asked to delete, and only the pair says
the delete cannot finish.

**These cases pin a live defect rather than assert a property that holds.** The control
plane calls `delete_object` and its role is not granted `s3:DeleteObject`, so
`DELETE /v1/files/{id}` commits its tombstone and then faults: the file reads as
deleted, the bytes stay, and the retry `UploadedFiles.delete` documents as the
converging path can never converge, because the erase fails the same way every time.
Adding the grant will break these cases, and that is the point -- whoever adds it should
delete this file in the same commit and say so, rather than find a stale test.

Verified against the live account rather than inferred from the file: for
`arn:aws:iam::062677866851:role/map-control-plane`, `iam simulate-principal-policy`
answers `implicitDeny` for `s3:DeleteObject` and `allowed` for `s3:PutObject` and
`s3:GetObject` on that bucket.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Final

from managed_agent.adapters.s3.uploaded_file import S3UploadedFiles

_ROOT: Final = Path(__file__).resolve().parents[2]
_IAM: Final = _ROOT / "deploy" / "iam"


def _s3_actions(role: str) -> set[str]:
    """Every S3 action the named role's policy document allows.

    Actions may be written as a bare string or a list, and both spellings appear in
    this directory, so both are flattened here rather than at each call site.
    """
    document = json.loads((_IAM / f"{role}.json").read_text())
    granted: set[str] = set()
    for statement in document["Statement"]:
        if statement.get("Effect") != "Allow":
            continue
        actions = statement["Action"]
        listed = [actions] if isinstance(actions, str) else actions
        granted.update(one for one in listed if one.startswith("s3:"))
    return granted


def test_the_control_plane_deletes_an_object_it_is_not_granted_to_delete() -> None:
    """The two halves of the defect, asserted together because neither is odd alone."""
    calls_delete = "delete_object" in inspect.getsource(S3UploadedFiles.erase)
    granted = _s3_actions("map-control-plane")

    assert calls_delete, (
        "S3UploadedFiles.erase no longer calls delete_object; if the delete path was "
        "reworked, this file is describing a defect that no longer exists"
    )
    assert "s3:DeleteObject" not in granted, (
        "map-control-plane now grants s3:DeleteObject, so the defect these cases pin "
        f"is fixed -- delete this file in that commit. Granted: {sorted(granted)}"
    )


def test_the_grant_is_held_by_the_process_that_never_deletes() -> None:
    """The same permission, on the same bucket, on the role with no caller for it.

    Recorded because it is what makes the fix cheap in one direction and tempting in
    the wrong one: the grant already exists on this bucket, so narrowing it costs
    nothing today, while widening the control plane's is the change that needs a
    decision rather than a copy.
    """
    assert "s3:DeleteObject" in _s3_actions("map-tool-gateway")

    gateway = _ROOT / "src" / "managed_agent" / "gateway"
    callers = [
        path.name
        for path in gateway.rglob("*.py")
        if "delete_object" in path.read_text()
    ]
    assert callers == [], (
        f"the gateway now deletes objects, so the grant has a caller: {callers}"
    )
