"""The one name a model is shown for a tool, built from the pair that identifies it.

A tool's identity in this platform is the pair `(server_name, tool_name)`. Two servers
may each offer `search`, and a person reading either registration sees `search` -- the
bare name the server itself reports. That pairing is the same shape the upstream
Managed Agents API uses, where an `mcp_tool_use` block carries `name` and `server_name`
as separate fields and a tool name alone is never required to be unique.

**This module exists because one layer cannot carry a pair.** The Agent Runtime is
configured with exactly one MCP server -- the Tool Gateway -- so every tool a tenant has
arrives at the model through a single namespace, as a single string. There is no second
field to put the server in. So the pair is joined here, on the way out to the runtime,
and the joined form is what the model calls and what the Gateway resolves.

**Why the join has to stay tenant-unique**, which is the constraint that shapes
everything else here: the runtime rewrites the names it is handed. It qualifies each as
`mcp__<server>__<tool>`, maps every character outside `[a-zA-Z0-9_]` to `_`, and appends
a SHA1-derived twelve-hex suffix when two names would sanitize to one. A Grant written
against a name that later acquires such a suffix resolves to nothing. Registered names
are shaped so that the sanitizer is the identity function over them, and tenant-unique
advertised names are what leave the collision suffix with nothing to disambiguate. Per
*server* uniqueness alone would not do that -- it is the joined name that must be
unique, and the store's index on `(tenant_id, advertised_name)` is what makes it so.

Nothing here parses an advertised name back into its halves, deliberately. `ServerName`
admits `_` and `-`, so the join is not injective by shape -- `ab_` with `c` and `ab`
with `_c` produce one string -- and a splitter would have to either forbid characters
tenants use or guess. The Gateway resolves a call by looking the advertised name up
whole, which needs no split and cannot guess wrong.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import Field

from managed_agent.core.registration.tool_names import (
    MAX_TOOL_NAME_BYTES,
    ServerName,
    ToolName,
)

SERVER_TOOL_SEPARATOR: Final[str] = "__"
"""What sits between the two halves.

A double underscore rather than a dot or a slash, because the advertised name passes
through the runtime's sanitizer and must come out unchanged: `.` and `-` are both
rewritten to `_` there, so a dotted name would arrive as an underscored one and a Grant
written against the dotted form would miss. `__` is already the separator the runtime
itself uses in `mcp__<server>__<tool>`, so the shape reads the same at both layers.
"""

_ADVERTISED_NAME_PATTERN: Final[str] = (
    rf"^[a-z][a-z0-9_-]{{0,{MAX_TOOL_NAME_BYTES - 1}}}$"
)

AdvertisedToolName = Annotated[str, Field(pattern=_ADVERTISED_NAME_PATTERN)]
"""The name the Tool Gateway advertises and the model calls.

Its character class is `ServerName`'s rather than `ToolName`'s, because the server half
is what widens it: a server may be named `deep-wiki`, and the join carries that `-`
through. The runtime maps `-` to `_` when it sanitizes, which is why the store's
uniqueness index is the guarantee rather than this pattern -- two servers named
`deep-wiki` and `deep_wiki` would advertise names that differ here and collide there.
Such a pair is refused at registration, where a person can still rename one.

Bounded by `MAX_TOOL_NAME_BYTES` and not by anything of its own: this string is what
`mcp__<gateway>__` is prepended to, so it consumes exactly the budget a bare tool name
used to.
"""


def advertised_name_for(server_name: ServerName, tool_name: ToolName) -> str:
    """The single name a model sees for this tool, from the pair that identifies it.

    Returns `str` rather than `AdvertisedToolName` because `Annotated` is erased at
    runtime -- the annotation would be a claim nothing checks. The value is parsed where
    it is stored and where it is read back, which is where a claim can actually bite.
    Callers that need the budget honoured must ask `pair_fits_the_budget` first; this
    joins whatever it is given.
    """
    return f"{server_name}{SERVER_TOOL_SEPARATOR}{tool_name}"


def pair_fits_the_budget(server_name: ServerName, tool_name: ToolName) -> bool:
    """Whether this pair's advertised name fits what the runtime will qualify.

    Checked on the sum rather than on each name alone, and that is the whole reason this
    is a function. Bounding the two patterns independently means picking numbers that
    add up in the *worst* case -- a 63-byte server name would leave 31 bytes for a tool
    name, and ordinary registrations would start failing for want of room nobody was
    using. Checking the pair costs a long server name nothing but shorter tool names,
    and refuses exactly the registrations that could never be advertised.

    Byte length rather than character count, to stay honest against the ceiling the
    runtime actually applies. Both patterns are ASCII, so today the two agree; a pattern
    widened later would find this already measuring the right thing.
    """
    return len(advertised_name_for(server_name, tool_name).encode()) <= (
        MAX_TOOL_NAME_BYTES
    )
