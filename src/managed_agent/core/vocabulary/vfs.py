"""The vfs family: an object was created, or one was rewritten before that could not be.

**These two events are the provenance record, and there is deliberately no file beside
them saying the same thing.** The obvious implementation of an artifact contract is a
`manifest.json` in the lane carrying the path, the digest, the sources and a delivery
status — and it is the wrong one here, because a mutable blob asserting `accepted: true`
is a second source of truth that can disagree with the log. Two records of one fact
means a reader has to know which one is lying. So the facts a manifest would hold are
appended here once, and anything that wants them folds the log
(`core/vfs/vfs_provenance.py` does exactly that).

Two types rather than one with a direction field. Creating an object and rewriting one
are different acts with different consequences, and folding them would make "was this
artifact ever revised" a question about a field rather than about which events are in
the log. Nothing emits the second any more — see `OBJECT_REPLACED` — and it stays
declared here because Session logs written before that already carry rows of it.

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
"""An object in a rewritable lane was overwritten, and by what.

**Nothing writes this any more, and it is not deprecated — it is historical.** The lane
that could produce it was `working`, carrying the agent's tree between Turns; the
workspace became a mounted volume that needs no carrying and the lane went with it
(ADR-035), taking the only `replace` in the tree. Every lane is sealed now, so no call
that exists can append one of these.

It stays declared because Sessions that ran before that hold rows of this type, and
`core/vfs/vfs_provenance.py` folds them. Deleting the constant would not delete the
rows: the fold skips a type it does not recognise, so those Sessions would quietly start
reporting `revisions=0` over a log that says otherwise, and a provenance record that
under-reports a revision is the exact failure this family exists to make impossible.
Delete it when no retained log can still contain one, and not before.
"""
