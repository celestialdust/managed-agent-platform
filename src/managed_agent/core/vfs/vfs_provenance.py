"""Where a stored object came from, computed by reading the Event Log forward.

This is the projection that replaces a `manifest.json`. The facts such a file would hold
about one produced thing — its path, the digest taken over its bytes, what it was
derived from and the digest each source had at the time, and whether it has been
rewritten since — are all appended to the log by `core/vocabulary/vfs.py`'s two events,
and this folds them back into one view per object. Nothing writes those facts down a
second time, so there is no second copy free to disagree with the record, and no state a
writer can assert into existence that the log does not already support.

The fold is total over event types it does not know: an unrecognised type advances the
sequence and changes nothing. That is what lets a delivery or approval act become a
third event type later without this function being edited to stay correct — the same
property, for the same reason, as `core/session/projection.py`.

**`placed_at_seq` is optional and that is the retention story, not laxity.** A Session's
log has a retained floor, so a read can legitimately begin after the placement it
would otherwise have seen. Reporting the replacement's own sequence as the placement
would manufacture a provenance fact the log does not carry; `None` says the placement
is not in the range that was read, which differs from an object never placed at all.

Keyed by `(lane, relative)` rather than by a joined path string. A third spelling of a
path — after the bucket key and the tree the agent reads — is a third thing that has to
agree with the other two, and the pair needs no agreement to be correct.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from managed_agent.core.ids import Seq
from managed_agent.core.ports import EventRecord
from managed_agent.core.vfs.evidence import EvidenceDigest
from managed_agent.core.vfs.session_vfs import Lane, SourceRef
from managed_agent.core.vocabulary.vfs import OBJECT_PLACED, OBJECT_REPLACED


def written_payload(
    lane: Lane,
    relative: str,
    digest: EvidenceDigest,
    sources: Sequence[SourceRef] = (),
) -> dict[str, object]:
    """The payload a `vfs.object_placed` or `vfs.object_replaced` event carries.

    Written here, in the module that folds these events back, so the shape has one
    definition rather than one at each end. Two spellings of a payload -- one where it
    is built and one where it is read -- pass their own tests separately and disagree
    the first time a field is renamed on one side.

    `sources` is omitted from the payload entirely when empty rather than written as
    `[]`, so an event carries the field only when a writer had something to say. The
    fold treats absent and empty identically, so this costs a reader nothing and keeps
    the stored row honest about what was actually declared.
    """
    payload: dict[str, object] = {
        "lane": lane.directory,
        "relative": relative,
        "digest": digest.model_dump(),
    }
    if sources:
        payload["sources"] = [
            {"relative": source.relative, "digest": source.digest.model_dump()}
            for source in sources
        ]
    return payload


@dataclass(frozen=True, slots=True)
class ObjectProvenance:
    """One stored object's history, as the log records it."""

    lane: str
    relative: str
    digest: EvidenceDigest
    """The digest of the most recent write this fold saw, not of the first.

    An object's current bytes are what a consumer downloads, so the current digest is
    what a consumer can check. What the earlier bytes hashed to is still in the log, in
    the events this folded.
    """

    last_written_seq: Seq
    placed_at_seq: Seq | None
    revisions: int
    """How many rewrites the log records after the placement. Zero for a sealed lane
    always, because nothing can write one twice."""

    sources: tuple[SourceRef, ...]
    """What the most recent write named as its inputs, with each input's digest then.

    Empty means the writer named nothing, which is not the same claim as "derived from
    nothing" -- a write with no sources to declare and a write whose author did not
    declare them are indistinguishable here, and neither is asserted to be the other.
    """


def _digest_from(payload: Mapping[str, object], seq: Seq) -> EvidenceDigest:
    """Read the digest a vfs event carries, or say which event failed to carry one.

    Raises `ValueError` naming the sequence rather than skipping the event. These
    payloads are written by one adapter from already-typed values, so a payload that
    does not parse is a defect in a writer or a schema change -- and a provenance view
    that quietly omitted the object would report an artifact as never produced, which is
    the exact failure this record exists to make impossible.
    """
    raw = payload.get("digest")
    if not isinstance(raw, Mapping):
        raise ValueError(f"vfs event at seq {seq} carries no digest object")
    return EvidenceDigest.model_validate(dict(raw))


def _sources_from(payload: Mapping[str, object], seq: Seq) -> tuple[SourceRef, ...]:
    """Read the optional sources list. Absent is empty; present and malformed raises."""
    raw = payload.get("sources")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"vfs event at seq {seq} has a non-list sources field")
    parsed: list[SourceRef] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError(f"vfs event at seq {seq} has a non-object source entry")
        relative = entry.get("relative")
        if not isinstance(relative, str):
            raise ValueError(f"vfs event at seq {seq} has a source with no path")
        parsed.append(
            SourceRef(relative=relative, digest=_digest_from(entry, seq)),
        )
    return tuple(parsed)


def _identity(payload: Mapping[str, object], seq: Seq) -> tuple[str, str]:
    lane = payload.get("lane")
    relative = payload.get("relative")
    if not isinstance(lane, str) or not isinstance(relative, str):
        raise ValueError(f"vfs event at seq {seq} does not name a lane and a path")
    return lane, relative


def provenance(
    events: Iterable[EventRecord],
) -> Mapping[tuple[str, str], ObjectProvenance]:
    """Fold the log into one provenance record per object written in it.

    Returns an empty mapping for a log that wrote no objects, which is not an error: a
    Session that produced nothing durable has exactly this provenance, and inventing a
    row for it would be inventing a file.
    """
    found: dict[tuple[str, str], ObjectProvenance] = {}
    for event in events:
        if event.type not in (OBJECT_PLACED, OBJECT_REPLACED):
            continue
        key = _identity(event.payload, event.seq)
        digest = _digest_from(event.payload, event.seq)
        sources = _sources_from(event.payload, event.seq)
        standing = found.get(key)
        if event.type == OBJECT_PLACED:
            found[key] = ObjectProvenance(
                lane=key[0],
                relative=key[1],
                digest=digest,
                last_written_seq=event.seq,
                placed_at_seq=event.seq,
                revisions=0,
                sources=sources,
            )
            continue
        if standing is None:
            # A rewrite whose placement sits below the log's retained floor. The
            # rewrite is real and is recorded; the placement is reported absent rather
            # than backfilled from this event's own sequence.
            found[key] = ObjectProvenance(
                lane=key[0],
                relative=key[1],
                digest=digest,
                last_written_seq=event.seq,
                placed_at_seq=None,
                revisions=1,
                sources=sources,
            )
            continue
        found[key] = replace(
            standing,
            digest=digest,
            last_written_seq=event.seq,
            revisions=standing.revisions + 1,
            sources=sources,
        )
    return found
