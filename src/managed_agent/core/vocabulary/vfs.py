"""The vfs family: an object was created, or one in the mutable lane was rewritten.

**These two events are the provenance record, and there is deliberately no file beside
them saying the same thing.** The obvious implementation of an artifact contract is a
`manifest.json` in the lane carrying the path, the digest, the sources and a delivery
status — and it is the wrong one here, because a mutable blob asserting `accepted: true`
is a second source of truth that can disagree with the log. Two records of one fact
means a reader has to know which one is lying. So the facts a manifest would hold are
appended here once, and anything that wants them folds the log
(`core/vfs/vfs_provenance.py` does exactly that).

Two types rather than one with a direction field. Creating an object and rewriting one
are different acts with different consequences — the first is the only write a sealed
lane has, the second is possible in one lane and nowhere else — and folding them would
make "was this artifact ever revised" a question about a field rather than about which
events are in the log.

**Nothing here carries a timestamp.** The Event Log's own row has a server-defaulted
`created_at` (migration 0001), so "when it was produced" is already recorded by the
store that orders these events. A payload copy would be a second spelling of a fact the
log already holds, free to disagree with it, and written by the one party whose clock
nobody checks — the caller's.
"""

from managed_agent.core.vocabulary import declare

FAMILY = "vfs"

OBJECT_PLACED = declare("vfs.object_placed", FAMILY)
"""A new object was created in a lane, at a path that held nothing.

Carries the lane, the lane-relative path, the digest and the byte length — and, when the
writer named them, the sources the object was derived from with the digest each had at
the time. A later reader rehashes a source and learns whether the derivation still
describes the bytes it was made from; that check is the whole reason the source digests
are recorded here rather than looked up when somebody asks.
"""

OBJECT_REPLACED = declare("vfs.object_replaced", FAMILY)
"""An object in the one rewritable lane was overwritten, and by what.

Only the mutable lane can produce this. Its existence in a Session's log is what makes a
recorded source digest checkable: an artifact whose source was rewritten after the
artifact was made has a log that says so, in order, rather than a source file that
quietly stopped matching the hash beside its name.
"""


WORKING_LANE_PARTIAL = declare("vfs.working_lane_partial", FAMILY)
"""The workspace was larger than one Turn will sync, so the lane holds part of it.

Not a failure and not a refusal. A Turn that produced good work in a large workspace
must not be failed for the size of that workspace, so the sync takes what fits, in a
deterministic order, and says here what it left. Carries how many paths and bytes were
taken and what the ceilings are.

It exists because the alternative is silence. A tenant resuming from a lane that holds
most of their tree, with nothing anywhere saying part of it was dropped, would read the
absence as the platform having lost work rather than as a limit they can act on.
"""
