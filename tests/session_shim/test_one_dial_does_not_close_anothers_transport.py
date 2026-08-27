"""Two Turns dialling at once must not share the pool one of them will close.

`httpx.AsyncClient.aclose` closes the transport it was handed, whoever built it. So a
dialler that builds one transport and hands it to every client has whichever Turn ends
first tear down the connections the others are still streaming through. The survivors
fail with `ReadError`, which `HttpPodDispatch` reports as `runtime_lost`: a Turn that
did its work, lost because another Turn finished.

It only became reachable when the internal CA was rolled. Until then `transport_for`
was handed no TLS context, returned `None`, and every client built its own transport --
there was nothing to share. A control plane holding a CA was the first to have one
object worth sharing and the first to share it wrongly.
"""

from __future__ import annotations

import ast
import asyncio
import ssl
from pathlib import Path

import httpx

from managed_agent.session_shim.pod_channel import transport_for

_POD_CHANNEL = (
    Path(__file__).parents[2]
    / "src"
    / "managed_agent"
    / "session_shim"
    / "pod_channel.py"
)


async def _hold_a_chunked_response_open(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Answer with one chunk, wait, then finish -- a Turn streaming its events."""
    await reader.readuntil(b"\r\n\r\n")
    writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
    writer.write(b"5\r\nfirst\r\n")
    await writer.drain()
    await asyncio.sleep(0.5)
    writer.write(b"4\r\nlast\r\n0\r\n\r\n")
    await writer.drain()
    writer.close()


async def _read_a_stream_while_another_client_closes(
    streaming: httpx.AsyncBaseTransport, finishing: httpx.AsyncBaseTransport, port: int
) -> list[str]:
    """What the streaming client sees when the other client closes mid-stream."""
    seen: list[str] = []

    async def stream() -> None:
        client = httpx.AsyncClient(transport=streaming)
        try:
            async with client.stream("GET", f"http://127.0.0.1:{port}/") as response:
                async for chunk in response.aiter_bytes():
                    seen.append(chunk.decode())
        except httpx.HTTPError as broke:
            seen.append(f"BROKE {type(broke).__name__}")

    reading = asyncio.create_task(stream())
    await asyncio.sleep(0.2)
    async with httpx.AsyncClient(transport=finishing):
        pass
    await asyncio.wait_for(reading, timeout=10.0)
    return seen


async def test_a_turn_that_ends_does_not_break_one_still_streaming() -> None:
    """A per-client transport survives another client closing; a shared one does not.

    Both halves are asserted, because only the shared half proves this case can tell
    the difference at all. A version that checked the good path alone would pass just
    as happily against a harness where nothing was ever closed.
    """
    server = await asyncio.start_server(_hold_a_chunked_response_open, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    try:
        tls = ssl.create_default_context()

        shared = transport_for(None, tls)
        assert shared is not None
        broke = await _read_a_stream_while_another_client_closes(shared, shared, port)
        assert broke == ["first", "BROKE ReadError"], broke

        streaming = transport_for(None, tls)
        finishing = transport_for(None, tls)
        assert streaming is not None and finishing is not None
        survived = await _read_a_stream_while_another_client_closes(
            streaming, finishing, port
        )
        assert survived == ["first", "last"], survived
    finally:
        server.close()
        await server.wait_closed()


def test_no_dialler_hands_one_stored_transport_to_more_than_one_client() -> None:
    """Every `AsyncClient` in `pod_channel` builds its own transport at the call site.

    The rule the case above proves, read off the syntax so it holds for the four
    diallers rather than for the one a behavioural case could reach. A stored
    `transport=self._...` is the shape of the defect: one object, built once per
    dialler, handed to every client that dialler ever opens.
    """
    for node in ast.walk(ast.parse(_POD_CHANNEL.read_text())):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if not (isinstance(called, ast.Attribute) and called.attr == "AsyncClient"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "transport":
                continue
            assert not isinstance(keyword.value, ast.Attribute), (
                "a client is being handed a transport stored on the dialler; "
                "whichever client closes first takes the others' connections with it"
            )
