"""The output family: the agent wrote a file, and here is the id that downloads it.

**Without this event a produced file is unreachable, and that is why it exists.**
`control/files/output_shipout.py` pulls what the agent wrote off the pod and places it
in the Session's `artifacts` lane before the pod is reaped, so the bytes survive -- and
tells nobody. There is no route that lists a lane, deliberately (see
`control/api/routes/artifacts.py`), so without this the path existed in one process, for
the length of one call, and nowhere a tenant could ask. A Turn that produced a document
ended with a completed Turn, bytes safely in S3, and no answer to "where is my
document".

Announced as an event rather than added to the resources listing, and the reason is the
Turn. A listing answers "which files does this Session hold" and loses which Turn
produced each one; the log answers both, in order, and a tenant already polls it to
learn the Turn finished. The path arrives in the same stream as `turn.completed`, after
it, which is also the true causal order: ship-out runs at completion.

**Distinct from `vfs.object_placed`, which the same write also appends.** That event
records that an object exists in a lane and is appended for every lane write there is --
including a working-tree sync, which is dozens of scratch files a Turn. This one says a
deliverable was produced. Two facts, so two events; folding them would make "what did
this Turn make for me" a question about which lane a path happens to sit in.

**The payload names the artifact and does not carry it.** `path` and `byte_length` --
the bytes are at `GET /v1/sessions/{id}/artifacts/{path}` and a copy here would be a
second answer about one object, unbounded, on the platform's clock. No digest either:
the sibling `vfs.object_placed` carries the digest taken over the octets as stored, and
a copy here is a second spelling of a fact free to disagree with it.
"""

from managed_agent.core.vocabulary import declare

FAMILY = "output"

OUTPUT_PRODUCED = declare("output.produced", FAMILY)
"""One file the agent wrote, now in the Session's artifacts lane at the path this names.

Emitted once per file, not once per Turn: a Turn that wrote three documents appends
three of these, so a tenant that wanted the second one is not left folding a list out of
a single event. A Turn that wrote nothing appends none, which is what makes the absence
of this event mean "no document" rather than "look somewhere else".

`path` is where the agent wrote the file, relative to the directory the platform told it
to put deliverables in, so `out/report/fig1.png` arrives as `report/fig1.png`. It may
carry separators -- a deliverable that is a tree keeps its shape -- and it is re-parsed
against the lane's own grammar before anything is stored under it.

**A path IS unique within a Session, and that is a change from what the file id gave.**
The lane is sealed, so a later Turn cannot write the same path with different bytes: the
store refuses it and the Turn is refused saying so. A Turn that re-offers the identical
bytes appends nothing at all, so a tenant polling this log is never told a document was
produced again on a Turn that did not touch it.
"""

OUTPUT_PARTIAL = declare("output.partial", FAMILY)
"""Not everything the agent has produced is in the artifacts lane yet, and here is why.

**Not a failure and not a refusal**, which is the whole reason it exists. A Turn that
did the work and wrote the documents must not be failed for how many there were --
that would make the platform's own bound read as the agent's error, and it would repeat
on every Turn afterwards, because nothing removes a file from a pod. So ship-out takes
what fits, in sorted order so two runs over one tree take the same prefix, and says here
what it left.

`paths_shipped` and `paths_left` count this Turn's NEW files only; a path already
delivered was never weighed. The two ceilings are carried so a tenant reading this knows
which one bound them without having to know the platform's constants.

**What was left behind is still in the pod's output directory and is offered again on
the next Turn**, where it costs nothing already delivered and is taken by the same
sorted prefix -- so a large batch drains over successive Turns rather than being lost.
Two things that does not cover, and both are the tenant's to act on: a Session whose
last Turn was this one never drains the tail, and a single file larger than the whole
byte budget is left behind every Turn forever, because no prefix holding it ever fits.
This event is the only place either becomes visible.

`tree_truncated` is the third and the worst, and it is a different fact from the other
two. It says the pod stopped walking the output directory at `tree_ceiling` entries, so
the counts above are of what it *listed* and there are further paths nobody here has
seen. Those do not drain: the walk is sorted, so a name beyond the cut is beyond it on
every Turn, and no amount of waiting brings it in. The move is to stop adding paths --
collect into fewer, larger files, or start a new Session -- and it is the tenant's move
because the platform will not walk further and holds no grant to delete out of a pod.

It is appended whenever that is true, including on a Turn where everything listed was
already delivered and `paths_left` is zero. That Turn is the steady state of a Session
past the bound, and it is the one that would otherwise look perfect.
"""
