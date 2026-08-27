"""IAM grants every prefix this platform composes keys under, and grants no other.

The other half of `test_nothing_writes_the_mounted_workspace.py`. That file proves no
key builder reaches the mounted workspace prefix; this one proves AWS would refuse the
write even if one did, and -- the direction that actually breaks a deploy -- that AWS
does not refuse the writes the platform makes on purpose.

Both roles that can write the platform bucket held `Resource: arn:...:bucket/*` with no
Condition until this file existed. So the mount-safety property was held by code
discipline alone: S3 would have accepted the write that silently discards an agent's
live working tree, and the tool gateway's grant carried `s3:DeleteObject` bucket-wide,
which reaches the mount as well. Narrowing each `Resource` to the prefixes the builders
actually compose puts AWS behind the code's guarantee instead of downstream of it.

**Both directions are silent failures, which is why both are asserted here.**

- A prefix granted that no builder composes is reach nothing needs. `workspaces/` is
  the one that matters -- it is the mount -- but any of them widens what a compromised
  or merely buggy process can destroy.
- A prefix a builder composes and IAM does not grant is an AccessDenied at the first
  write, in production, from a policy that reads as perfectly reasonable. This is the
  direction a narrowing gets wrong, and it is the reason the grant is compared against
  **called** builders rather than against a list somebody keeps up to date.

The prefixes are derived from `tests/object_key_builders.py`, which calls each builder.
Nothing in this file names a prefix, so a builder whose prefix moves fails here rather
than passing against a stale copy.

Tier 1 (local, no cluster). What is graded is the committed policy document, which is
what `deploy/terraform/irsa.tf` applies -- it reads these files and substitutes the
account id, so the file and the deployed policy cannot disagree about anything except
that number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest
from object_key_builders import key_roots

_ROOT: Final = Path(__file__).resolve().parents[2]
_IAM: Final = _ROOT / "deploy" / "iam"

_ROLES: Final = ("map-control-plane", "map-tool-gateway")
"""The two roles granted object actions on the platform bucket.

Named rather than globbed over `deploy/iam/`: `map-model-gateway.json` is in that
directory and holds no S3 grant at all, and a glob would turn "this role writes no
objects" into a case that skips. A third role that starts writing objects is a
deliberate addition here, made by whoever grants it.
"""


def _statements(role: str) -> list[dict[str, Any]]:
    return list(json.loads((_IAM / f"{role}.json").read_text())["Statement"])


def _resources_of(statement: dict[str, Any]) -> list[str]:
    """A statement's resources, whether it spells them as one string or as a list.

    Both spellings are legal IAM and mean the same thing. Mirrors the same defence
    `test_control_plane_manifest.py` applies to `Action`, and for the same reason: a
    reader must not be able to change what these cases grade by rewriting a list of
    one as a bare string.
    """
    resource = statement["Resource"]
    return [resource] if isinstance(resource, str) else list(resource)


def _object_grant(role: str) -> list[str]:
    """The resources this role is granted OBJECT actions on, across every statement.

    Object grants are told from bucket grants by the ARN rather than by the Sid,
    because IAM enforces that difference: an object action needs `arn:...:bucket/*`
    and a bucket action needs `arn:...:bucket`, and each is inert on the other's ARN.
    Collected across statements rather than assuming one, so splitting the grant in
    two does not quietly halve what this file grades.
    """
    return [
        resource
        for statement in _statements(role)
        if any(action.startswith("s3:") for action in statement["Action"])
        for resource in _resources_of(statement)
        if resource.endswith("/*")
    ]


def _granted_prefixes(role: str) -> set[str]:
    """The key prefixes an object grant covers, as `evidence/`-shaped strings.

    Raises on a bucket-wide `/*`, rather than reporting it as the empty prefix. That
    ARN grants every key in the bucket including the mounted workspace, and a helper
    that returned `""` for it would let the comparison below read as a subset check
    that passes.
    """
    prefixes = set()
    for arn in _object_grant(role):
        _, _, key = arn.removesuffix("*").partition("/")
        assert key, (
            f"{role} is granted {arn}, which covers every key in the bucket -- "
            "including `workspaces/`, the Session workspace mount. S3 Files resolves "
            "a bucket-versus-file-system conflict in the bucket's favour, so a write "
            "there discards an agent's live working tree with no call failing."
        )
        prefixes.add(key)
    return prefixes


@pytest.mark.parametrize("role", _ROLES)
def test_the_role_has_an_object_grant_to_grade(role: str) -> None:
    """Guard the guard: the two cases below compare against this set.

    A role whose object grant went missing would satisfy "grants no prefix the code
    does not write" vacuously, which is how this file would pass while every upload
    answered AccessDenied.
    """
    assert _object_grant(role), (
        f"{role}.json grants no S3 object actions, so the cases below grade nothing"
    )


@pytest.mark.parametrize("role", _ROLES)
def test_every_prefix_the_code_writes_is_one_this_role_is_granted(role: str) -> None:
    """The direction that breaks a deploy, asserted per role.

    Both roles construct all four stores: `tool_gateway_app` calls `build()`, which
    wires `S3UploadedFiles`, `EvidenceStore`, `SessionVfsStore` and `S3RolloutStore`
    over the one bucket regardless of which process is asking. So the grant is the
    same on both, and a narrowing that reasoned about "what the tool gateway is for"
    rather than about what it constructs would deny it three of the four.
    """
    granted = _granted_prefixes(role)
    composed = set(key_roots())

    assert composed <= granted, (
        f"{role} composes keys under {sorted(composed - granted)} and is not granted "
        "them. Every write there is an AccessDenied at the first call, from a policy "
        "that reads as correct; the builders are in tests/object_key_builders.py"
    )


@pytest.mark.parametrize("role", _ROLES)
def test_this_role_is_granted_no_prefix_the_code_never_writes(role: str) -> None:
    """The direction the mount depends on, asserted per role.

    Equality with the composed set rather than an absence check on `workspaces/`
    specifically. Naming the one forbidden prefix would go on passing the day a fifth
    prefix is granted for a reason nobody records, and the point of a least-privilege
    grant is that reach nothing uses is reach nothing should have.

    A prefix genuinely wanted fails here and is meant to: it is added to this platform
    by adding the builder that composes it, at which point `key_roots()` reports it and
    this case passes without a line changing in this file.
    """
    granted = _granted_prefixes(role)
    composed = set(key_roots())

    assert granted <= composed, (
        f"{role} is granted {sorted(granted - composed)}, which no key builder in the "
        "tree composes. If that is `workspaces/`, it is the Session workspace mount "
        "and the grant makes an agent's working tree silently discardable"
    )
