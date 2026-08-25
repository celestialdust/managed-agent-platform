"""The tool family's third question: was the server there to be called at all.

`tool_call.py` records how a call went and `tool_in_flight.py` records that one is still
moving. Neither can describe a Turn in which the call was never made because the server
never came up -- and that is the Turn that needs describing most, because in the log it
is indistinguishable from a Turn whose agent simply chose not to use its tools.

**Measured on this platform rather than imagined.** A Session whose definition named a
tool server ran a Turn where the agent said, in its own words, that the tool was not in
its toolset, and then asked the tenant how it should proceed. The Turn reported
completion. The Tool Gateway logged no request from that pod at all. The Agent Runtime
knew the whole time -- it announces every server's startup outcome, failure text
included -- and nothing on this side was listening, so the platform's own record of a
Turn whose grant went unhonoured read exactly like the record of one that worked.

Only the outcomes that leave the Session without the tool cross. A `ready` counterpart
would write a row per server per placement to say that the expected thing happened, and
the evidence a server IS up is already in the log as the call that reached it.

The family label is `tool` and this is the third module carrying it, which the registry
supports by design: it keys the published set on the type name and treats the family as
a reader's grouping, so files may split by slice while the prefix stays one family.
"""

from managed_agent.core.vocabulary import declare

FAMILY = "tool"

TOOL_SERVER_UNAVAILABLE = declare("tool.server_unavailable", FAMILY)
"""A tool server this Session was given did not come up, and the runtime's reason.

The payload carries `server`, the name the Agent Runtime knows it by; `state`, the
runtime's own word for how startup ended, which is `failed` or `cancelled`; and
`error`, the runtime's own text, which is present when it had anything to say and
absent when it did not.

`state` and `error` are both carried because they answer different questions. The word
separates a server that could not be reached from one whose startup was abandoned while
the pod was going away, which are not the same incident; the text is the only place the
difference between a handshake that timed out and an address that refused survives, and
those are two different people's problems.

The text is passed through rather than rewritten. A platform that normalised it would
be inventing a vocabulary for failures it does not enumerate, and the first runtime
version to phrase something new would arrive here as a familiar-looking lie.

This does not interrupt an agent that has already begun. What it buys is that a grant
the platform failed to honour stops being invisible to everyone including the platform.
"""
