"""The committed form of a registered name, and the byte budget it is committed to.

Split out from `scope_binding`, which holds the registration shapes these names appear
in. The split is what lets `advertised_name` join a server name to a tool name without
importing a module that has to import it back: both depend on this, and this depends on
neither.

The forms are narrow on purpose, and the reason is the Agent Runtime rather than taste.
It qualifies an MCP tool as `mcp__<server>__<tool>`, rewrites every character outside
`[a-zA-Z0-9_]` to `_`, appends a SHA1-derived twelve-hex suffix when two tools would
sanitize to one name, and truncates the result to fit 128 bytes. A name a Grant was
written against would then be free to change under a runtime upgrade. Every pattern here
is chosen so that each of those transformations is the identity function over a name
that matches it -- which is why loosening one by a single character class is not a
convenience but a Grant that silently resolves to nothing.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import Field

MAX_QUALIFIED_TOOL_NAME_BYTES: Final[int] = 128
"""The Agent Runtime's own ceiling on the qualified name it shows a model."""

QUALIFICATION_RESERVE_BYTES: Final[int] = 32
"""Held back from that ceiling for the `mcp__<server>__` prefix it prepends."""

MAX_TOOL_NAME_BYTES: Final[int] = (
    MAX_QUALIFIED_TOOL_NAME_BYTES - QUALIFICATION_RESERVE_BYTES
)

_TOOL_NAME_PATTERN: Final[str] = rf"^[a-z][a-z0-9_]{{0,{MAX_TOOL_NAME_BYTES - 1}}}$"

ToolName = Annotated[str, Field(pattern=_TOOL_NAME_PATTERN)]
"""The name a Grant names and the Tool Gateway advertises.

Lowercase only, though the sanitizer would preserve uppercase: it keeps `Search` and
`search` apart as two names, so allowing both would let a Grant written against one
silently miss the other. The character class is ASCII, so the pattern's length limit and
the byte limit are the same number and neither needs encoding to check.
"""

ServerName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,62}$")]
"""The name an agent definition names a server by. Unique within one tenant."""


def qualification_fits(runtime_facing_server_name: str) -> bool:
    """Whether `mcp__<name>__` fits the reserve held back above.

    The Agent Runtime is configured with exactly one MCP server, the Tool Gateway, and
    whatever name the compiled configuration gives it is what consumes this reserve.
    Stating the arithmetic as a function means the compiler checks it against the same
    number the names were bounded by, rather than carrying a copy of that number.
    """
    qualified = f"mcp__{runtime_facing_server_name}__"
    return len(qualified.encode()) <= QUALIFICATION_RESERVE_BYTES
