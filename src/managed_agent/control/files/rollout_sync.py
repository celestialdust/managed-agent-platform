"""Moving a Session's Rollout out of the pod that wrote it, and back into the next one.

The Rollout is the Agent Runtime's own resume state and cannot be rebuilt from anything
this platform records: the compaction checkpoints live inside it and never travel over
the runtime's protocol. It is preserved rather than reconstructed, and the unit of
preservation is one completed Turn (ADR-004).

Restoring is therefore not a plain read-back. A record that ends mid-Turn is replayed
including its incomplete tail, which would put a resumed Session at its last written
line. The cut below happens before the bytes ever reach a pod, so the recovery boundary
is a property of this platform rather than a hope about how the pod happened to die.

Nothing in this module imports a web framework or an object-store client. It sits on the
control-plane side because the Session pod is given no cloud identity and cannot write
the object store itself; the two collaborators it needs are Protocols declared below and
satisfied elsewhere, so a cycle back through the pod wire is not reachable from here.
"""

import json
from dataclasses import dataclass
from typing import Protocol

from managed_agent.core.ids import SessionId, TurnId

_SESSION_META = "session_meta"
_EVENT_MSG = "event_msg"

TURN_OPENED = frozenset({"turn_started"})
"""The event name that opens a Turn in the runtime's persisted record.

Exactly the name the research measured, and no synonym. `turn_started` /
`turn_complete` / `turn_aborted` are cited with file and line at
`research/plan-pass-rollout-and-resume.md` lines 6 and 98, read off codex-rs at the
pinned 0.149.0.

The set is deliberately not a superset, and the direction of the risk is why.
Matching a name that is not a Turn boundary cuts *after* an unfinished Turn and hands
a partial tail back, which is precisely the state ADR-004 exists to make unreachable.
Matching too few cuts too early: the Session replays from its header, which costs
context and money and loses no correctness claim this platform makes. Between a
fail-open and a fail-expensive, this takes the expensive one -- and
`RestoredRollout.completed_turns` is returned so a caller comparing it against its own
count of `turn.completed` in the Event Log sees the disagreement instead of inferring
it. Widening this set is a deliberate edit that needs a measurement against the binary
behind it.
"""

TURN_CLOSED_COMPLETE = frozenset({"turn_complete"})
"""The event name that closes a Turn *successfully*. The cut lands on this line."""

TURN_CLOSED_ABORTED = frozenset({"turn_aborted"})
"""A Turn that ended without completing. Never a cut point.

Separate from the two above rather than folded into either. An aborted Turn started and
did not complete, so it sits past the cut like an interrupted one -- but it is not a
torn tail and the record says so explicitly, which is worth keeping distinguishable when
a reader is asking why a Turn went missing.
"""

_TURN_OPENERS = TURN_OPENED | TURN_CLOSED_ABORTED


class MalformedRollout(ValueError):
    """The bytes do not open with a session_meta line, so they are not a Rollout."""


@dataclass(frozen=True, slots=True)
class RestoredRollout:
    """A Rollout cut back to its last completed Turn, and what the cut took off.

    `partial_turn_dropped` is true only when a Turn had started and not completed past
    the cut, not merely when bytes were dropped: token-count lines are written after a
    Turn completes and those are not a Turn in flight.

    **It is a fact about these bytes and not a signal that a Turn was lost.** Ship-out
    fires when a Turn completes, while the runtime is idle, so the record that reaches
    the store ends on its completion line. A Turn that starts after that and dies with
    the pod is not in these bytes at all, and a cut cannot report what was never handed
    to it -- so on the dominant recovery path this reads False, and a caller that hung a
    discard marker on it would append nothing exactly when one is owed. True here means
    the narrower thing it says: the bytes carried a Turn that had opened and not closed,
    which happens when the fetch overlapped a later append.

    `completed_turns` is the count the caller reconciles against its own log, and that
    reconciliation -- not the flag above -- is what names a Turn as lost. ADR-004 says
    the Event Log may be up to one Turn ahead of this record; a caller that reads a
    count of zero against a log holding completed Turns is looking at a cut that matched
    nothing, and that is the failure mode `TURN_CLOSED_COMPLETE` above is narrow enough
    to have.

    Nothing here names a Turn the way the Agent Runtime names one. A runtime identifier
    must not reach a tenant (ADR-007), and the surest way to keep one out of a marker
    payload is for it never to be in the value the marker is built from.
    """

    body: bytes
    completed_turns: int
    partial_turn_dropped: bool
    dropped_lines: int


def truncate_at_last_completed_turn(body: bytes) -> RestoredRollout:
    """Cut a Rollout back to the last line that completed a Turn.

    A record with no completed Turn keeps its `session_meta` header alone. A rollout
    with no lines is rejected, and so is one that does not open with that header, so the
    header is the smallest thing that can be handed back -- the runtime's own paginated
    read treats a first line that is not `session_meta` as a hard error.

    A line that will not decode is counted and skipped rather than fatal: the only way
    one exists is a tail torn by a write that never finished, and such a line sits past
    the cut by construction. Ship-out reads the file while the runtime may still be
    appending to it, so a torn last line is expected rather than exceptional. Both ways
    a torn line fails to decode are handled -- see `_decoded`, where only one of them is
    the obvious one.
    """
    lines = [line for line in body.split(b"\n") if line.strip()]
    if not lines:
        raise MalformedRollout("a Rollout with no lines cannot be resumed")
    if _line_kind(lines[0]) != _SESSION_META:
        raise MalformedRollout("a Rollout must open with its session_meta line")

    cut = 0
    completed = 0
    for index, line in enumerate(lines):
        if _event_name(line) in TURN_CLOSED_COMPLETE:
            cut = index
            completed += 1

    dropped = lines[cut + 1 :]
    return RestoredRollout(
        body=b"\n".join(lines[: cut + 1]) + b"\n",
        completed_turns=completed,
        partial_turn_dropped=any(
            _event_name(line) in _TURN_OPENERS for line in dropped
        ),
        dropped_lines=len(dropped),
    )


def _decoded(line: bytes) -> dict[str, object] | None:
    """One line read as a JSON object, or None when these bytes are not one.

    Two ways a torn line fails, and the second is the one that bites. `json.loads` on
    bytes decodes UTF-8 *before* it parses, so a line cut mid-character raises
    `UnicodeDecodeError` -- which shares no base with `JSONDecodeError` short of
    `ValueError` and so is not caught by naming the parse failure alone. The Agent
    Runtime's serializer emits non-ASCII raw rather than escaped, so an em-dash, a
    smart quote or a non-English word in model output is two to four bytes on disk, and
    a write that stopped halfway lands inside one. Letting that escape made the
    caller's "counted and skipped" false for precisely the records most likely to have
    a torn tail -- and since the stored object is replaced only by the next completed
    Turn, which needs a resume of its own, a Session that hit it never got out.
    """
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _line_kind(line: bytes) -> str | None:
    value = _decoded(line)
    kind = None if value is None else value.get("type")
    return kind if isinstance(kind, str) else None


def _event_name(line: bytes) -> str | None:
    """The inner event name of an `event_msg` line; None for every other line shape."""
    value = _decoded(line)
    if value is None or value.get("type") != _EVENT_MSG:
        return None
    payload = value.get("payload")
    if not isinstance(payload, dict):
        return None
    name = payload.get("type")
    return name if isinstance(name, str) else None


class RolloutObjectStore(Protocol):
    """The two blob operations a Rollout needs, and no others.

    Declared beside its one consumer rather than among the shared ports because a
    Rollout is replaced wholesale: it needs neither the hash-addressed write Evidence
    needs nor the version listing an Artifact needs, and a port offering those would
    invite a caller to give a Rollout a history it does not have.

    **There is no delete, and that is the type doing the work.** Stopping a Session
    leaves its record readable until retention expires, so the one path that may remove
    a Rollout is the retention sweep that owns expiry for everything a Session leaves
    behind. A second remover could take a Rollout away while the sweep still believed
    the Session was inside its window -- and because this Protocol is the only handle
    `RolloutSync` holds, a `RolloutSync` that tried to delete would fail to type-check
    rather than fail in production.
    """

    async def put(self, key: str, body: bytes) -> None: ...

    async def get(self, key: str) -> bytes | None:
        """The stored body, or None when no Turn of this Session has ever completed."""


class RolloutFetch(Protocol):
    """Getting one Session's live Rollout bytes out of the pod that holds them.

    An abstraction rather than the pod client itself, so this module imports nothing
    from the pod wire. The concrete implementation is `shim/pod_channel.py`'s
    `PodRolloutFetch`, which already computes the pod's address and the Session's token;
    `placement` is on that side of the seam and importing it here would close a cycle
    through `control/session/placement.py`.

    The direction is the whole point of the shape and not an accident of it: state
    leaves a pod because the control plane came and read it, never because the pod
    pushed. A Protocol shaped the other way -- a `report_rollout(session_id, body)` the
    pod called -- would need a control-plane address and a durable write credential
    inside the least-trusted process in the platform, and the per-Session token that
    would authorize it is a constant for the Session's life with no nonce and no
    expiry, so every passive observation of one would be a permanent write capability
    (ADR-022).
    """

    async def fetch_rollout(self, session_id: SessionId, /) -> bytes | None:
        """The Session pod's current Rollout, or None when it holds none yet."""


def rollout_key(session_id: SessionId) -> str:
    """One object per Session. It carries no Turn: each completed Turn replaces it.

    Both halves of this string are load-bearing, and nothing outside it does either job.
    The Session segment is the whole of the separation between one tenant's resume state
    and another's: without it every Session's completed Turn writes the same object, and
    the next resume hands a tenant whoever shipped last -- an entire agent conversation
    belonging to somebody else. The prefix is the whole of the separation from the other
    things a bucket here holds; the bucket comes from its own variable, and nothing
    stops an operator pointing that at the bucket uploaded files already live under.
    """
    return f"rollouts/{session_id}/rollout.jsonl"


class RolloutSync:
    """Ship-out at the end of a Turn, restore at a resume, and no third operation."""

    def __init__(self, store: RolloutObjectStore) -> None:
        self._store = store

    async def ship_out(self, session_id: SessionId, body: bytes) -> None:
        """Replace the Session's stored Rollout with what the pod holds now."""
        await self._store.put(rollout_key(session_id), body)

    async def restore_for_resume(self, session_id: SessionId) -> RestoredRollout | None:
        """What the next pod is given, or None when no Turn ever finished.

        **An object holding zero bytes reads as None, not as a malformed Rollout.** The
        port answers None for a key that was never written, but an object that exists
        and is empty is a different value with the same meaning, and `is None` alone
        would send those bytes into the cut -- which refuses a Rollout with no lines.
        That refusal is unrecoverable rather than merely loud: the stored object is
        replaced only by the next completed Turn, and a completed Turn needs a resume,
        so a Session that met it never started again. None says the true thing instead:
        no Turn of this Session is recoverable from what is stored.
        """
        stored = await self._store.get(rollout_key(session_id))
        if not stored:
            return None
        return truncate_at_last_completed_turn(stored)


class ShipOutAtTurnCompletion:
    """The `TurnCompleted` a Turn's durability hangs on, however that Turn ended.

    Satisfies `shim.turn_runner.TurnCompleted` structurally -- no import, because that
    Protocol lives on the pod-wire side and this module deliberately reaches none of it.
    `HttpPodDispatch` awaits this after a Turn's terminal event is appended, for a Turn
    that completed and -- wired as `on_terminal` as well -- for one that ended without
    completing. The second wiring is why the name says *completion* and the work says
    *the Rollout as it stands*: nothing here reads the outcome, because the bytes a pod
    holds are worth the same either way and the pod is about to be allowed to die.

    This class was for a while the completed-only seam, and its docstring argued that
    was the correct reading -- there being no completed Turn to make durable. That
    argument was about the wrong noun. What a failed Turn has is a *conversation*, which
    the runtime folded into compaction checkpoints no other record here reproduces, and
    losing it is not the tenant declining to save work; it is the platform discarding
    what the tenant already paid for.

    **What this does NOT yet buy, and a reader should not assume it does.** The bytes
    reach the store, and `restore_for_resume` cuts them at the last `turn_complete` on
    the way back out, so a failed Turn's lines are stored and then dropped before they
    reach the next pod. Widening that cut moves ADR-004's recovery boundary from the
    last completed Turn to the last closed one, which is a decision this class is not
    where to make. Until it is made, this seam is a prerequisite that is not yet a
    remedy.

    **A pod holding no Rollout ships nothing rather than an empty object.** Overwriting
    a good stored Rollout with zero bytes destroys the very thing recovery reads, so the
    test is on the bytes and not on the sentinel: a fetch that answers None and a fetch
    that answers `b""` are the same fact about the pod, and only one of them is `None`.

    Both arrive in practice. The route answers 204 for a pod that has written no file,
    which the fetch turns into None; it decides that on one stat, and a record emptied
    or replaced after that stat reaches the wire as an empty 200, which the fetch hands
    back as the bytes it was given. A guard spelled `is None` admits the second, and
    what follows is not a failure but silent unrecoverable loss -- the put succeeds, the
    stored record becomes zero bytes, every later resume reads a Rollout with no lines
    and refuses, and the object is replaced only by the next completed Turn, which needs
    a resume. The Session is wedged for good. This is the one place that decides whether
    a put happens, which is why the guard belongs here and not only at the route: the
    route can only report what one stat saw.

    **A failed ship-out raises, and the Turn's tenant-visible append has already
    happened.** ADR-004 puts that divergence in writing -- the Event Log may be up to
    one Turn ahead of the resume state, and resume reconciles them -- so the honest
    outcome is a dispatch that fails loudly over one that swallows the error and leaves
    a Turn reading as durable when its bytes are still only inside a pod about to be
    allowed to die.
    """

    def __init__(self, fetch: RolloutFetch, sync: RolloutSync) -> None:
        self._fetch = fetch
        self._sync = sync

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        body = await self._fetch.fetch_rollout(session_id)
        if not body:
            return
        await self._sync.ship_out(session_id, body)
