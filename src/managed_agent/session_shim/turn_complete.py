"""What the pod must get right at a Turn boundary for the Turn to survive the pod.

Two pure functions and no I/O beyond a directory listing. Nothing here holds a
credential and nothing here reaches the network: the Session pod is given no cloud
identity, so the Rollout leaves by being *read* by the control plane over the channel
the control plane opened, not by being written to the object store from in here.

Both functions run in the `session-shim` container. `find_rollout` reads the tree the
Agent Runtime writes its record into, which that container mounts read-only.
"""

import hashlib
from pathlib import Path

from managed_agent.control.pod_config.compiler import CODEX_HOME
from managed_agent.core.ids import SessionId

RUNTIME_HOME = Path(CODEX_HOME)
"""The runtime's home as this container sees it, taken from the compiler's constant.

Not re-spelled and not read from the environment. `config_compiler.CODEX_HOME` is the
string the Permission Profile's deny rules are compiled against and the string the
runtime container's `CODEX_HOME` is set to; a third spelling here is a third thing that
can drift. `shim/serve.py` already imports two constants from that module, so the
direction of this import is the established one.
"""

_TAG_DIGEST_BYTES = 8


class RolloutNotFound(RuntimeError):
    """No Rollout file for this thread exists under the pod's runtime home."""


def find_rollout(codex_home: Path, thread_id: str) -> Path:
    """Locate the live Rollout for one runtime thread.

    The runtime reports no path, so the file is found the way the runtime's own fallback
    finds it: a scan of `sessions/<YYYY>/<MM>/<DD>/`, whose leaf name is
    `rollout-<timestamp>-<thread_id>.jsonl`. The trailing wildcard catches the reverted
    form, whose name carries a second id after the thread id
    (`rollout-<ts>-<thread_id>_<rollout_id>.jsonl`); when more than one matches, the
    newest by modification time is the one still being appended to.

    Only plain `.jsonl` is matched, and that is a statement rather than a gap: a rollout
    is compressed to `.jsonl.zst` by a background worker once it is seven days cold and
    no pod lives that long. A Session suspended for a week has no pod and no local file
    at all -- only the object its last completed Turn shipped out.

    `codex_home` is a parameter rather than a read of `RUNTIME_HOME`, so a test can
    build a tree in a tmp_path. The route in `shim/serve.py` passes `RUNTIME_HOME`.
    """
    matches = sorted(
        codex_home.glob(f"sessions/*/*/*/rollout-*-{thread_id}*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise RolloutNotFound(
            f"no rollout under {codex_home / 'sessions'} for thread {thread_id}"
        )
    return matches[-1]


def subagent_tag(session_id: SessionId, runtime_subagent_id: str) -> str:
    """The tenant-facing name for one subagent inside one Session.

    Derived, never counted. A counter handed out per pod restarts at zero when the
    Session comes back on a new pod, and the second subagent to be numbered 1 would be
    folded into the first by any consumer grouping events by tag. This is a pure
    function of two values that never change, so the tag a subagent had before a
    recovery is the tag it has after, and a subagent that did not exist before cannot
    collide with one that did.

    Keyed on the Session so the same runtime identifier in two Sessions yields two tags,
    and one-way so the identifier it is built from cannot be read back out of the tag
    (ADR-007 keeps Agent Runtime identifiers off the tenant surface). `SessionId` is a
    `NewType` over `UUID`, so `.bytes` is sixteen bytes -- inside blake2s's 32-byte key
    limit.
    """
    digest = hashlib.blake2s(
        runtime_subagent_id.encode("utf-8"),
        digest_size=_TAG_DIGEST_BYTES,
        key=session_id.bytes,
    ).hexdigest()
    return f"sub_{digest}"
