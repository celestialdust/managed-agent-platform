"""A stand-in Agent Runtime on a unix socket, speaking the real frame codec.

Real in the half this slice has to prove and fake in the half it cannot reach here. The
transport is genuine: a WebSocket server bound to a unix socket, exchanging one JSON
message per text frame, which is what the runtime's own unix listener does and what a
client writing newline-delimited JSON would fail against. The answers are canned.

The `codex` binary is not installed in this environment, so nothing here can show that
the real Agent Runtime accepts these method names or parameter shapes. What it does show
is what the shim puts on the wire and what it does with what comes back — including the
set of requests it issues, which is measured from the frames this server received rather
than asserted about the code that sent them.

Not a `conftest.py`: mypy refuses a second module named `conftest` in a tree with no
`__init__.py`, and the repository has one at `tests/`. So this is a context manager a
test opens rather than a fixture it names.
"""

from __future__ import annotations

import asyncio
import http
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

from websockets.asyncio.server import ServerConnection, unix_serve
from websockets.http11 import Request, Response

_DEFAULT_RESULTS: Final[dict[str, dict[str, Any]]] = {
    "initialize": {"userAgent": "fake-agent-runtime/1"},
    "thread/start": {"thread": {"id": "th_fake_1"}},
    "thread/resume": {"thread": {"id": "th_fake_2"}},
    "thread/read": {"thread": {"id": "th_fake_1"}},
    "thread/goal/set": {},
    "turn/start": {"turn": {"id": "tn_fake_1"}},
    "turn/steer": {"turnId": "tn_fake_1"},
    "turn/interrupt": {},
}


class FakeAgentRuntime:
    """Records every frame the shim sends and answers with a canned result.

    `received` is the measurement the Repertoire guard rests on: the outbound request
    set is read off the frames that actually arrived, so a call the shim declines to
    make is absent from it rather than present with an error beside it.

    `errors` makes one method fail; `silent` makes one method never answer, which is how
    a call still in flight when the socket goes is set up.
    """

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.received: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = dict(_DEFAULT_RESULTS)
        self.errors: dict[str, dict[str, Any]] = {}
        self.silent: set[str] = set()
        self._attached = asyncio.Event()
        self._conn: ServerConnection | None = None

    @property
    def methods_received(self) -> list[str]:
        return [str(frame["method"]) for frame in self.received if "method" in frame]

    async def handle(self, conn: ServerConnection) -> None:
        self._conn = conn
        self._attached.set()
        async for frame in conn:
            message: dict[str, Any] = json.loads(frame)
            self.received.append(message)
            request_id = message.get("id")
            method = str(message.get("method", ""))
            if request_id is None or method in self.silent:
                continue
            if method in self.errors:
                answer: dict[str, Any] = {
                    "id": request_id,
                    "error": self.errors[method],
                }
            else:
                answer = {"id": request_id, "result": self.results.get(method, {})}
            await conn.send(json.dumps(answer))

    async def push(self, message: dict[str, Any]) -> None:
        """Send one server-to-client frame, once a client has attached."""
        await asyncio.wait_for(self._attached.wait(), timeout=5.0)
        assert self._conn is not None
        await self._conn.send(json.dumps(message))

    async def vanish(self) -> None:
        """Drop the socket with no closing handshake, the way a killed process does.

        `transport.abort()` and not `close()`: closing sends the WebSocket closing
        frames and the client reads that as an orderly end. A runtime container that
        exits leaves a socket that simply stops, which the client raises as
        `ConnectionClosedError: no close frame received or sent` -- and that is the
        state worth reproducing, because it is the one the shim's shutdown met.
        """
        await asyncio.wait_for(self._attached.wait(), timeout=5.0)
        assert self._conn is not None
        self._conn.transport.abort()

    async def wait_until(self, settled: Callable[[], bool], what: str) -> None:
        """Block until this server has seen what a case is about to assert on.

        Frames cross a socket and a scheduler, so "the shim sent it" and "the server
        recorded it" are two moments. Asserting at the first makes a test that passes
        or fails on machine speed, which is a defect in the test and reads as one in
        the shim.
        """
        async with asyncio.timeout(5.0):
            while not settled():
                await asyncio.sleep(0.005)
        assert settled(), what


_EXTENSIONS_HEADER: Final = "Sec-WebSocket-Extensions"


def _refuse_a_compression_offer(
    connection: ServerConnection, request: Request
) -> Response | None:
    """Refuse a handshake carrying **any** extension offer, as the real runtime does.

    This is the one place the fake is deliberately *less* permissive than `websockets`'
    own defaults, and it exists because being more permissive hid a defect that broke
    every Turn. `codex app-server 0.149.0` does not decline an extension the way the
    protocol allows -- it closes the connection with **zero bytes**, so the client
    raises `InvalidMessage: did not receive a valid HTTP response` and never learns why.

    Measured against the real binary inside the built Session image, one header at a
    time. The trigger is the **header**, not the extension named in it:

        plain                                             -> HTTP/1.1 101
        User-Agent: ...                                   -> HTTP/1.1 101
        X-Whatever: ...                                   -> HTTP/1.1 101
        Sec-WebSocket-Protocol: chat                      -> HTTP/1.1 101
        Sec-WebSocket-Extensions: permessage-deflate      -> b''
        Sec-WebSocket-Extensions: x-not-a-real-extension  -> b''

    A first version of this hook matched `permessage-deflate` specifically, which was
    the extension `websockets.connect` offers by default (`compression="deflate"`) and
    so caught the live defect -- but it left the fake still kinder than the runtime for
    every other extension. Matching the header closes that, and it is also the honest
    reading of the measurement: nothing observed suggests the runtime parses the value.

    `websockets.connect` offers `permessage-deflate` by default and a Python server
    negotiates it happily, so both ends of the fake agreed, fifty-nine shim tests
    passed, and the real runtime refused outright. Refusing here is what makes
    `compression=None` at the client a tested property rather than a comment.

    A 400 rather than a silent close: closing with no bytes would reproduce the runtime
    byte for byte, but what a reader of a red test needs to see is *which* header was
    rejected, and a status line carries that where an empty read does not. The property
    under test is that no extension offer is ever sent, and both refusals prove it.
    """
    if _EXTENSIONS_HEADER not in request.headers:
        return None
    return connection.respond(
        http.HTTPStatus.BAD_REQUEST,
        f"the Agent Runtime closes the connection on any {_EXTENSIONS_HEADER} offer; "
        "the client must disable compression and negotiate no extension\n",
    )


@asynccontextmanager
async def fake_agent_runtime() -> AsyncIterator[FakeAgentRuntime]:
    """A fake Agent Runtime listening on a short-pathed unix socket.

    `tempfile.mkdtemp` rather than pytest's `tmp_path`: a unix socket path is capped
    around 104 bytes by the kernel, and pytest's per-test directory names are long
    enough that a test with a descriptive name would fail to bind rather than fail an
    assertion.
    """
    directory = Path(tempfile.mkdtemp(prefix="map-"))
    fake = FakeAgentRuntime(directory / "ctl.sock")
    try:
        async with unix_serve(
            fake.handle,
            str(fake.socket_path),
            compression=None,
            process_request=_refuse_a_compression_offer,
        ):
            yield fake
    finally:
        shutil.rmtree(directory, ignore_errors=True)
