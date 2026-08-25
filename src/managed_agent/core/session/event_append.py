"""The one retry for a sequence another writer took first.

Three writers append to a Session's Event Log and none of them coordinates: the Turn
runner inside the pod, the Tool Gateway writing progress about a tool call that same
Turn is still appending to, and the control plane recording what a pod streamed out.
The store picks the sequence and raises `SequenceRace` when a concurrent writer took
the one this append wanted, so the only recovery is to ask again.

It lives here rather than beside any one caller because the ceiling is a single rule.
Two copies of the loop, each with its own attempt count, drift silently: both sides
keep passing their own tests while one gives up sooner than the other, and a lost race
surfaces as a failed tool call rather than as a number an append did not get.
"""

from __future__ import annotations

from typing import Final

from managed_agent.core.ids import Seq, SessionId
from managed_agent.core.ports import EventLogAppend, SequenceRace

APPEND_ATTEMPTS: Final[int] = 8
"""How many sequences an append may lose before the failure is real.

Losing one is ordinary; losing eight in a row means something other than concurrency,
so it is raised instead of retried forever.
"""


async def append_in_order(
    log: EventLogAppend,
    session_id: SessionId,
    type_: str,
    payload: dict[str, object],
) -> Seq:
    """Append one event, asking again for each sequence a concurrent writer took.

    Returns the sequence the store assigned. Raises `SequenceRace` once the attempts
    are spent — exhausting them is a real failure and is never swallowed, because a
    caller that treats a dropped append as success reports a Turn that has no record.
    """
    for _ in range(APPEND_ATTEMPTS):
        try:
            return await log.append(session_id, type_, payload)
        except SequenceRace:
            continue
    raise SequenceRace(
        f"{type_} for session {session_id} lost {APPEND_ATTEMPTS} sequence races"
    )
