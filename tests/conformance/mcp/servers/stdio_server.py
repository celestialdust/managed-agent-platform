"""A minimal MCP server over stdio, run as a child process by the conformance tests.

Real in the only sense the Tool Gateway can tell the difference: a separate process
speaking the protocol down a pipe, spawned by the code under test rather than stubbed
inside it. Nothing here is a fake the proxy could see through — it is reached the same
way a tenant's own registered command is.

It exists because no third-party stdio server is attested for this repo: the only two
attested MCP servers are remote and speak Streamable HTTP, and nothing runs locally.
What that costs is stated where it is felt — this server is written to the same SDK the
Gateway is, so it cannot catch a real server's protocol quirks, only the Gateway's own
handling of them.

The tools are chosen so each exercises one path through the Gateway that a passive
server could not reach. `echo_credential` returns the value of the environment variable
the registration named, which is how a test proves a credential reached the child and
did not leak into the parent. `explode` raises, so the error map is exercised against a
real transport carrying a real upstream failure rather than an exception a test
constructed. `crawl` reports progress before it answers, and `sleep_forever` never
answers at all, so the read deadline is measured rather than reasoned about.
`ask_operator` asks the caller a question mid-call and reports what came back, which is
the only way to exercise elicitation in the direction it actually travels: server to
Gateway to Session, and the answer all the way back.

Run by path, optionally with one argument: a file to write this process's pid into
before serving. That is how a test proves the child was reaped rather than merely
signalled — the proxy hands back no handle on the process it spawned, so the process has
to say who it is. It speaks on stdin/stdout and logs nowhere, because anything written
to stdout that is not a protocol frame corrupts the stream.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import anyio
import mcp.types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

CREDENTIAL_ENV_VAR = "MAP_CONFORMANCE_TOKEN"
LEAKY_MESSAGE = "conformance failure from internal-host-7.corp"
RESOURCE_URI = "conformance://stdio/notes"
RESOURCE_TEMPLATE = "conformance://stdio/pages/{page}"
RESOURCE_TEXT = "a resource the stdio server serves"
ELICITATION_MESSAGE = "which account should this run against?"
ELICITATION_FIELD = "account"
ELICITATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {ELICITATION_FIELD: {"type": "string"}},
    "required": [ELICITATION_FIELD],
}

_NO_ARGUMENTS: dict[str, object] = {"type": "object", "properties": {}}

_ANY_ARGUMENTS: dict[str, object] = {"type": "object", "additionalProperties": True}
"""Accepts whatever it is sent, because what it is sent is the thing under test."""


async def on_list_tools(
    context: ServerRequestContext[object],
    params: types.PaginatedRequestParams | None,
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name="echo_credential",
                description="Return the credential this process was started with.",
                input_schema=_NO_ARGUMENTS,
            ),
            types.Tool(
                name="explode",
                description="Raise, so the error map sees a real upstream failure.",
                input_schema=_NO_ARGUMENTS,
            ),
            types.Tool(
                name="crawl",
                description="Report progress three times, then answer.",
                input_schema=_NO_ARGUMENTS,
            ),
            types.Tool(
                name="ask_operator",
                description="Ask the caller a question, then report the answer.",
                input_schema=_NO_ARGUMENTS,
            ),
            types.Tool(
                name="echo_arguments",
                description="Return the arguments this call actually carried.",
                input_schema=_ANY_ARGUMENTS,
            ),
            types.Tool(
                name="sleep_forever",
                description="Never answer, so a read deadline is what ends the call.",
                input_schema=_NO_ARGUMENTS,
            ),
        ]
    )


async def on_call_tool(
    context: ServerRequestContext[object],
    params: types.CallToolRequestParams,
) -> types.CallToolResult:
    """Answer one call, or fail in the specific way the caller asked for.

    Progress is sent through the request context rather than returned, because a
    progress notification is a separate message on the same connection and arrives
    while this call is still open — which is the property the Gateway's own progress
    path has to be exercised against. `report_progress` is a no-op when the caller
    attached no progress token, so `crawl` is safe to call without asking for progress.

    An elicitation is the same shape in the other direction and one step stronger: it is
    a *request* this process makes of its caller while still owing that caller an
    answer, so `ask_operator` only completes if the reply travelled back down the same
    open connection. It carries this call's request id so the caller can tell which of
    several in-flight calls is asking.
    """
    if params.name == "echo_arguments":
        # Sorted, so the assertion reads against a stable string rather than against
        # whatever order the JSON round trip happened to preserve. This is a real
        # server on the other end of a real pipe on purpose: the property under test
        # is what the *outbound call* carried, and a fake that reported its own
        # arguments back would be the test writing down what it expected to see.
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text", text=json.dumps(params.arguments or {}, sort_keys=True)
                )
            ]
        )
    if params.name == "explode":
        raise RuntimeError(LEAKY_MESSAGE)
    if params.name == "sleep_forever":
        await anyio.sleep(3600)
    if params.name == "ask_operator":
        answered = await context.session.elicit_form(
            ELICITATION_MESSAGE,
            ELICITATION_SCHEMA,
            related_request_id=context.request_id,
        )
        given = (answered.content or {}).get(ELICITATION_FIELD)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"{answered.action}:{given}")]
        )
    if params.name == "crawl":
        for step in (1.0, 2.0, 3.0):
            await context.session.report_progress(step, 3.0, f"step {step:.0f}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="crawled")]
        )
    return types.CallToolResult(
        content=[
            types.TextContent(type="text", text=os.environ.get(CREDENTIAL_ENV_VAR, ""))
        ]
    )


async def on_list_resources(
    context: ServerRequestContext[object],
    params: types.PaginatedRequestParams | None,
) -> types.ListResourcesResult:
    return types.ListResourcesResult(
        resources=[
            types.Resource(
                uri=RESOURCE_URI,
                name="notes",
                mime_type="text/plain",
            )
        ]
    )


async def on_list_resource_templates(
    context: ServerRequestContext[object],
    params: types.PaginatedRequestParams | None,
) -> types.ListResourceTemplatesResult:
    return types.ListResourceTemplatesResult(
        resource_templates=[
            types.ResourceTemplate(
                uri_template=RESOURCE_TEMPLATE,
                name="pages",
                mime_type="text/plain",
            )
        ]
    )


async def on_read_resource(
    context: ServerRequestContext[object],
    params: types.ReadResourceRequestParams,
) -> types.ReadResourceResult:
    return types.ReadResourceResult(
        contents=[
            types.TextResourceContents(
                uri=params.uri,
                mime_type="text/plain",
                text=RESOURCE_TEXT,
            )
        ]
    )


async def main() -> None:
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
    server = Server(
        "map-conformance-stdio",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_resources=on_list_resources,
        on_list_resource_templates=on_list_resource_templates,
        on_read_resource=on_read_resource,
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    sys.exit(anyio.run(main))
