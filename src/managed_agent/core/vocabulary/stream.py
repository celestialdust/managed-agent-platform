"""The one event name the stream surface adds, and the payload it carries.

Every other name a caller sees on the stream is a log row's own type, carried through
with its sequence as the SSE event id. This family holds the single exception: a frame
the stream originates because it cannot continue, which has no row in the log and
therefore no sequence of its own.

It is a published event rather than an HTTP status because the condition it reports can
arrive after the response has already begun -- a retention sweep moving the retained
floor under a stream someone is holding open -- and by then a status line is no longer
available. One shape for one condition on this route is worth more than the status code,
and the code inside it is the same closed-set code a range read returns for the same
condition, so a caller branches once and covers both surfaces.

Keepalives are deliberately absent from this family. They go out as SSE comment lines,
carrying neither an event name nor an id, so an idle connection is held open without
widening the published set -- and a name added here is a version event for as long as
the API version lives (ADR-013).
"""

from pydantic import BaseModel, ConfigDict, Field

from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import Seq
from managed_agent.core.vocabulary import declare

FAMILY = "stream"

STREAM_ERROR = declare("stream.error", FAMILY)


class StreamError(BaseModel):
    """What a stream.error frame carries.

    `retained_floor` is here so a caller refused for reading below it can reconnect at a
    position that still exists rather than guess and be refused a second time. It is the
    lowest sequence readable at the moment of the refusal, so a caller resuming from it
    knows exactly what it gave up: everything under that number.

    `code` is typed as the closed error set rather than as a string, which is what stops
    this frame becoming a second, unversioned place to invent a refusal code: an
    unpublished one will not construct.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ErrorCode
    message: str = Field(min_length=1)
    retained_floor: Seq
