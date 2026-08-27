"""Every function in this tree that composes an S3 object key, in one list.

Two test modules need to know where this platform's object keys land, and they need
the same answer. `test_nothing_writes_the_mounted_workspace.py` asks whether any of
them reaches the mounted workspace prefix, which would silently discard an agent's
working tree. `test_the_object_grant_matches_the_keys_the_code_writes.py` asks whether
IAM grants all of them and nothing more, which decides whether a writer gets an
AccessDenied in production. One question each, opposite directions, one list.

Kept here rather than copied into both because the two copies would diverge silently
and in the worse direction: a builder added to the mount-safety list and forgotten in
the grant list reads as fully guarded while its writes are refused by AWS. The list is
also the thing most likely to go stale, since a new builder is a new module and neither
test imports it.

Builders are **called**, not described. A prefix changed in the module is a prefix
changed here, with no line in this file to update -- which is the only version of this
list that cannot be wrong about the code it grades.

Nothing here guards against a builder that never made the list. That is a source scan,
and it lives in `test_nothing_writes_the_mounted_workspace.py` because the scan is for
a written prefix string rather than for a call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final
from uuid import uuid4

from managed_agent.control.files.rollout_sync import rollout_key
from managed_agent.control.files.store import FileId, upload_object_key
from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.vfs.evidence import digest_of, evidence_object_key
from managed_agent.core.vfs.session_vfs import LANES, lane_prefix

A_TENANT: Final = TenantId(uuid4())
A_SESSION: Final = SessionId(uuid4())

KEYS: Final[Mapping[str, Callable[[], str]]] = {
    "a VFS lane object": lambda: lane_prefix(A_TENANT, A_SESSION, LANES[0]) + "a.txt",
    "an evidence object": lambda: evidence_object_key(A_SESSION, digest_of(b"x")),
    "an uploaded file": lambda: upload_object_key(A_TENANT, FileId(uuid4())),
    "a Session's Rollout": lambda: rollout_key(A_SESSION),
}


def key_roots() -> frozenset[str]:
    """The first path segment of every key the builders compose, each with a `/`.

    A prefix rather than a whole key, because that is the unit IAM grants: a
    `Resource` ARN ending `/uploads/*` covers every key under it, and the tenant and
    identifier segments below the root are exactly what an ARN must not pin.

    Raises if any builder composes a key with no `/` in it. A bare leaf would yield
    the whole key as its own root, and the grant derived from it would name one
    object -- which parses, deploys, and denies every other write under that prefix.
    """
    roots = set()
    for name, build in KEYS.items():
        key = build()
        head, slash, _ = key.partition("/")
        if not slash or not head:
            raise AssertionError(
                f"{name} composes {key!r}, which is not a key under a prefix; "
                "an IAM Resource derived from it would grant that one object"
            )
        roots.add(f"{head}/")
    return frozenset(roots)
