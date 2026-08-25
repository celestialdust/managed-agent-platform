"""The tool family's in-flight notices: a call is progressing, or is asking something.

Three types, and none of them is an outcome. A tool call that has not finished can say
two things to the Session that made it -- how far along it is, and that it needs an
answer before it can continue -- and both are readable as "still running" by a reader
that understands neither. The call's terminal event is `tool.call_failed`, in `tool.py`,
and there is deliberately no success counterpart there: a classified result already
writes a row from the Evidence family (ADR-019).

**Two modules, one family label, and that is the registry's design rather than a
compromise.** `core/vocabulary/__init__.py` discovers the modules in this package at
import and keys the published set on the type *name*; the family string beside it is a
reader's grouping. So `tool.py` and this file both declare `FAMILY = "tool"` and the
published names share one prefix, while the two slices that write them never write one
file -- which is the property the package's own docstring says the file-per-family shape
exists to give.

What makes the split safe rather than merely tidy: `declare` raises on a duplicate type
name, at import, in every process. If a second module ever declares one of the three
names below, the package fails to load loudly instead of two spellings of one event
quietly coexisting with one of them unreachable.

The elicitation pair is two types and not one with a direction field. A reader following
a Session waits on the request and records the answer; those are different moments with
different payloads, and folding them into one type would make "did anybody answer yet" a
question about a field rather than about which events are in the log.
"""

from managed_agent.core.vocabulary import declare

FAMILY = "tool"

TOOL_PROGRESS = declare("tool.progress", FAMILY)
"""A registered server reported progress on a call that is still running."""

TOOL_ELICITATION_REQUESTED = declare("tool.elicitation_requested", FAMILY)
"""A registered server asked the Session for input before it could continue."""

TOOL_ELICITATION_ANSWERED = declare("tool.elicitation_answered", FAMILY)
"""The answer to such a request reached the server."""
