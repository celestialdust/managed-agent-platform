"""The tool family's outcome: an agent called a registered tool, and how it went.

**The hole this fills was total.** Before this type existed the platform recorded that
a Turn started, what the agent said, and that the Turn ended -- and nothing at all about
what the agent DID. A tenant asking "which tools did my agent call" had one answer
available: read the prose and believe it. That is not an audit trail, and on a platform
whose whole proposition is running an agent on somebody's behalf against their own
credentials, the absence was the more serious half of what was missing.

Why one type for the outcome rather than a started/finished pair. The runtime reports a
call's completion with its terminal status already in the frame, so a `tool.called`
carrying `status` says everything a reader needs in one row; a pair would double the log
for a Turn that calls twenty tools and would leave the reader joining ids to learn what
the second row already says. `tool.progress` next door exists for the other question --
whether a call that has NOT finished is still moving -- and the two do not overlap.

**Three fields cross and the arguments do not.** The server, the tool and the status say
what happened; the arguments are the tenant's own data and are unbounded, and an Event
Log that carried them would hold a copy of every payload an agent ever sent to a third
party, retained on the platform's retention clock rather than the tenant's. A reader who
needs the arguments has the tool server's own logs. The duration crosses because it is
the one number that makes a slow Session diagnosable from the log alone.

**What this deliberately does NOT record: a shell command the agent ran.** The runtime
reports those as their own item kind and they are dropped, so an agent reading a file or
writing one leaves no row here. That is a real remaining gap rather than a decision that
commands do not matter -- it is a different payload with a different disclosure question
(a command line can carry a secret an argument list cannot), and folding it in as a
fourth field of this type would answer that question by accident.
"""

from managed_agent.core.vocabulary import declare

FAMILY = "tool"

TOOL_CALLED = declare("tool.called", FAMILY)
"""A registered tool was called through the Tool Gateway and reached a terminal state.

Its payload carries `server`, `tool`, `status` and `duration_ms`. `status` is the
runtime's own word for the outcome and is one of `completed`, `failed` or `inProgress`
-- the last of which should not appear on a completion frame and is passed through
rather than corrected, because a platform that rewrote it would hide a runtime whose
frames had changed shape under us.
"""
