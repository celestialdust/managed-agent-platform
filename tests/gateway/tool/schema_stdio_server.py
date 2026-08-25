"""A stdio MCP server whose one tool declares an output schema, run as a child process.

It exists because a declared output schema changes what the *caller's* MCP client does
with a result rather than what this Gateway does with it: the client caches the schema
from the listing and revalidates every non-error result against it, unasked. A fake
upstream cannot exercise that, and neither can the conformance server this one sits
beside, which declares no schema on any tool.

The report it returns is the value of the environment variable the registration named,
so a test chooses how large a result the Gateway has to classify by choosing what the
vault answers -- the same lever `tests/conformance/mcp/servers/stdio_server.py` offers,
for the same reason: nothing here takes a size argument, and adding one would be a
second way to say the same thing.

It speaks on stdin/stdout and logs nowhere, because anything written to stdout that is
not a protocol frame corrupts the stream.
"""

from __future__ import annotations

import os
import sys

import anyio
import mcp.types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

CREDENTIAL_ENV_VAR = "MAP_CONFORMANCE_TOKEN"
TOOL_NAME = "big_report"
OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"report": {"type": "string"}},
    "required": ["report"],
    "additionalProperties": False,
}


async def on_list_tools(
    context: ServerRequestContext[object],
    params: types.PaginatedRequestParams | None,
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name=TOOL_NAME,
                description="Return a report, in a shape this tool declares.",
                input_schema={"type": "object", "properties": {}},
                output_schema=OUTPUT_SCHEMA,
            )
        ]
    )


async def on_call_tool(
    context: ServerRequestContext[object],
    params: types.CallToolRequestParams,
) -> types.CallToolResult:
    """Answer with the report both ways, which is what a schema-declaring server does.

    The text block carries the report and the structured content carries it again. That
    duplication is the protocol's own recommendation for a tool with an output schema --
    a client reading only text still gets the answer -- and it is also what makes this
    server a fair test of the capture, because both halves are weighed.
    """
    report = os.environ.get(CREDENTIAL_ENV_VAR, "")
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=report)],
        structured_content={"report": report},
    )


async def main() -> None:
    server = Server(
        "map-schema-stdio",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    sys.exit(anyio.run(main))
