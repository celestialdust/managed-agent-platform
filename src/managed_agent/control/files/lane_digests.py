"""What the Event Log says each path in one lane currently hashes to.

A Turn-completion seam needs this and cannot get it from the object store. A lane
listing carries no digest -- that is deliberate at the store, because filling the field
would make listing a lane cost a download of it -- so the answer is folded out of the
Session's own append-only log, where every write recorded the digest it wrote.

**Written as a walk over any lane, and one caller uses it.** `output_shipout.py` asks it
which produced files this Turn has not already delivered. It took a lane rather than
assuming `artifacts` when a second seam asked the same question about the working lane;
that seam is gone with the lane (ADR-035), and the parameter stays because the question
is about a lane either way and hard-coding one would be narrowing a walk to fit its
current caller. What must not be duplicated is the paging -- this repository has got
that wrong three times in other modules, each time by taking a page cap's default and
folding one page as though it were the whole log.

It fails safe in one direction on purpose. A path whose write has fallen below the
log's retained floor has no entry here, so its caller re-transfers it. That costs one
upload. The other direction -- reporting a path as unchanged because its history was
forgotten -- would silently stop storing a file, and nothing downstream could notice.
"""

from collections.abc import Sequence
from typing import Protocol

from managed_agent.core.ids import Seq, SessionId
from managed_agent.core.ports import EventRecord
from managed_agent.core.vfs.vfs_provenance import provenance

_UNBOUNDED_END = Seq(2**63 - 1)
"""The high end of a range meaning "to the end of the log", as the port spells it."""


class SessionLogReader(Protocol):
    """The one read this makes of the Event Log, and not the whole range port.

    `EventLogRange` also carries `follow` and `retained_floor`, and nothing here calls
    either. Narrowed so a double built to exercise a caller has to implement what that
    caller uses rather than what the port's other consumers use.

    Declared here rather than in either consumer so both name the same shape. A
    Protocol declared twice is two shapes that agree until somebody widens one.
    """

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]: ...


async def digests_in_lane(
    log: SessionLogReader, session_id: SessionId, lane_directory: str
) -> dict[str, str | None]:
    """Every path this Session has written into that lane, mapped to its digest.

    `.hex` and not the digest object. `EvidenceDigest` carries the algorithm and the
    covered length beside the hex, so stringifying it yields a repr that matches no
    pod's report -- and a comparison that never matches degrades to re-transferring
    everything every Turn. That failure is invisible in the bucket, which holds the
    right bytes either way; only a count of what moved shows it.

    Keyed by the lane-relative path alone, so a caller compares it directly against
    what a pod listed. The fold underneath is keyed by lane as well, which is why the
    lane is an argument rather than something the caller filters afterwards: filtering
    afterwards is a step a caller can forget, and forgetting it silently mixes two
    lanes' digests.
    """
    folded = provenance(await _whole_log(log, session_id))
    return {
        record.relative: record.digest.hex
        for record in folded.values()
        if record.lane == lane_directory
    }


async def _whole_log(log: SessionLogReader, session_id: SessionId) -> list[EventRecord]:
    """Every event of one Session, across as many reads as it takes.

    **A short page means "read again", never "the range is empty".** The port caps a
    read and promises nothing about how much of a range comes back, so folding whatever
    the first call returned reports the state as of that page -- which reads exactly
    like the state as of the log, and is wrong only for the Sessions old enough for it
    to matter.

    Reads from just past the highest sequence seen and stops on an empty page. That
    terminates for any page size, because each non-empty page raises the cursor above
    its own highest sequence and a log is finite. Stopping on empty rather than on a
    short page matters: a page shorter than the cap is what the last read looks like,
    and also what a sparse range looks like in the middle.
    """
    events: list[EventRecord] = []
    cursor = 0
    while True:
        page = await log.read(session_id, Seq(cursor + 1), _UNBOUNDED_END)
        if not page:
            return events
        events.extend(page)
        cursor = max(event.seq for event in page)


def digest_differs(reported: str | None, stored: str | None) -> bool:
    """Whether these bytes have to be transferred, given what the lane already holds.

    **A missing digest on either side means "transfer it", never "assume unchanged".**
    A pod that reports no digest is running an older shim image, and reading its
    silence as "nothing changed" would freeze the lane at whatever it held when that
    pod started -- a wrong answer that looks exactly like a working one. A path with no
    provenance is one the lane has never been told about, or one whose write has fallen
    below the retained floor. Both resolve to the expensive, correct branch.

    Two digests and not two files, so this module never names the type a produced file
    is carried in. That type lives in `output_shipout.py`, which imports this one;
    taking it as a parameter here would close the cycle.
    """
    if stored is None or reported is None:
        return True
    return reported != stored
