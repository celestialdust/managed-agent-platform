"""Resource event types: what a Session holds after creation named something else.

Its own family rather than a member of `lifecycle`, because an attach moves no state.
Every type declared in that module has a row in the projection's table, so one sitting
there with no row reads as a transition somebody forgot to write. The table also folds
part of the turn family now, which does not change the reading -- an attach neither
opens a unit of work nor ends one, so it belongs to neither family.
"""

from pydantic import BaseModel, ConfigDict

from managed_agent.core.vocabulary import declare

FAMILY = "resource"

SESSION_FILE_ATTACHED = declare("session.file_attached", FAMILY)


class SessionFileAttached(BaseModel):
    """The payload of a `session.file_attached` event: one file id and nothing else.

    The filename, the media type, the length and the digest are all reachable from
    `uploaded_file` by this id, and an `UploadedFile` row is frozen and never rewritten
    -- so a copy here would be a second statement of a fact that cannot change, free to
    disagree with the first only by being written wrong. What the event has to say is
    *which file*, and when, which is the id and the event's own sequence.

    A consequence worth naming: the byte ledger that bounds a Session's attachments
    prices each held file with a store read rather than by summing a number out of the
    log. That is one indexed row per file and it is the cost of not keeping two copies
    of a length.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_id: str
