"""The Session-shim's connection to the Agent Runtime over the pod's control socket.

The published `openai-codex` client cannot serve here. It is a stdio client that starts
the Agent Runtime as a child process it owns and writes to that process's stdin; its
async class is the same client on a worker thread. ADR-001 chose a unix-socket
transport, which that package never opens, so the transport below is ours while the
protocol types are not invented — they are the Repertoire's.

The socket carries WebSocket frames, not raw JSON. The Agent Runtime completes a
WebSocket server handshake on accept and then exchanges one JSON message per text frame,
so a client that opened the path and wrote newline-delimited JSON would hang at the
handshake. The message envelope carries no `jsonrpc` member: a request is
`{id, method, params}` and a notification is the same without the id.

This transport publishes no readiness endpoint, so readiness is synthesized: a
connection is ready once the `initialize` exchange has returned, and reporting it ready
earlier would be reporting that a socket file exists.

Every outbound message goes through `_write`, which takes a RepertoireEntry and never a
method name. Naming a call the Repertoire does not declare is therefore not something
this module can express (ADR-002).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection, unix_connect
from websockets.exceptions import ConnectionClosed

from managed_agent.control.pod_config.compiler import CONTROL_SOCKET
from managed_agent.core.pod.repertoire import (
    REPERTOIRE,
    REQUIRES_EXPERIMENTAL_API,
    ClientInfo,
    InitializeCapabilities,
    InitializedNotification,
    InitializeRequest,
    InitializeResponse,
    RepertoireEntry,
    RepertoireMethod,
    ThreadGoalSetRequest,
    ThreadReadRequest,
    ThreadReadResponse,
    ThreadResumeRequest,
    ThreadResumeResponse,
    ThreadStartRequest,
    ThreadStartResponse,
    TurnInterruptRequest,
    TurnStartRequest,
    TurnStartResponse,
    TurnSteerRequest,
    TurnSteerResponse,
)

_WS_URI: Final = "ws://localhost/"
_MAX_MESSAGE_BYTES: Final = 128 << 20
_CONNECT_TIMEOUT_S: Final = 10.0
_METHOD_NOT_FOUND: Final = -32601

CONTROL_SOCKET_PATH: Final = Path(CONTROL_SOCKET)
"""Where the Agent Runtime listens inside a Session's pod.

Imported from the module that compiles the pod's launch argv rather than derived
again here, because the same string is also the target of the sandbox deny rule that
keeps the confined agent off this socket. Two modules deriving it independently is how
the deny rule ends up guarding a path nothing listens on, and neither side would fail:
the runtime would answer on one path while the profile denied another.
"""


class NotInRepertoire(Exception):
    """A call was attempted with an entry the Repertoire does not declare."""


class RuntimeConnectionClosed(Exception):
    """The control socket was not open, or closed under an in-flight call."""


class RuntimeCallFailed(Exception):
    """A Repertoire call the Agent Runtime refused or could not complete.

    `pod_local_detail` holds the Agent Runtime's own message and must never be
    serialized out of the pod: a runtime error name reaching a tenant is exactly what
    the platform forbids (ADR-007). `str()` renders only the method and the numeric
    code, so the safe rendering is the one a caller gets by accident.
    """

    def __init__(
        self, method: RepertoireMethod, code: int, pod_local_detail: str
    ) -> None:
        self.method = method
        self.code = code
        self.pod_local_detail = pod_local_detail
        super().__init__(f"{method} failed with code {code}")


class RuntimeConnection:
    """One shim-to-Agent-Runtime connection, restricted to the Repertoire."""

    def __init__(self, socket_path: Path = CONTROL_SOCKET_PATH) -> None:
        self._socket_path = socket_path
        self._stack = AsyncExitStack()
        self._conn: ClientConnection | None = None
        self._pending: dict[
            str, tuple[RepertoireMethod, asyncio.Future[dict[str, Any]]]
        ] = {}
        self._notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None

    async def connect(self) -> InitializeResponse:
        """Open the socket, complete the handshake, and only then report ready."""
        conn = await self._stack.enter_async_context(
            unix_connect(
                path=str(self._socket_path),
                uri=_WS_URI,
                max_size=_MAX_MESSAGE_BYTES,
                open_timeout=_CONNECT_TIMEOUT_S,
                # Not a default worth inheriting. `websockets.connect` uses
                # `compression="deflate"`, which puts
                # `Sec-WebSocket-Extensions: permessage-deflate` in the handshake, and
                # `codex app-server` does not decline it the way the protocol allows --
                # it closes the connection having written **zero bytes**. The client
                # then raises `InvalidMessage: did not receive a valid HTTP response`,
                # which names the symptom and not one thing about the cause. Measured
                # against the real binary, one header at a time: plain and `User-Agent`
                # both answer `HTTP/1.1 101`; the extension header answers `b''`.
                #
                # Omitting this is a total failure, not a degradation: the connection
                # never opens, so the shim's lifespan raises, uvicorn exits, the
                # readiness probe never passes, and every Turn in every Session fails.
                # `tests/session_shim/fake_agent_runtime.py` refuses the offer for that
                # reason -- with the extension allowed, twenty tests passed against a
                # fake more permissive than the runtime they stand in for.
                compression=None,
            )
        )
        self._conn = conn
        self._reader = asyncio.create_task(self._read_forever(conn))
        result = await self._request(
            REPERTOIRE[RepertoireMethod.INITIALIZE],
            InitializeRequest(
                client_info=ClientInfo(
                    name="managed-agent-session-shim",
                    title="Managed Agent Session-shim",
                    version="v1",
                ),
                capabilities=InitializeCapabilities(
                    experimental_api=REQUIRES_EXPERIMENTAL_API
                ),
            ),
        )
        await self._notify(
            REPERTOIRE[RepertoireMethod.INITIALIZED], InitializedNotification()
        )
        return InitializeResponse.model_validate(result)

    async def close(self) -> None:
        """Stop reading, close the socket, and fail anything still waiting.

        Idempotent: closing a connection that is already closed does nothing and is not
        an error, so a lifecycle transition that retries after a timeout takes the same
        path as the first attempt.
        """
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.cancel()
            with suppress(asyncio.CancelledError):
                await reader
        await self._stack.aclose()
        self._conn = None
        self._fail_pending("the connection was closed")

    async def notifications(self) -> AsyncGenerator[dict[str, Any], None]:
        """Every inbound notification, in arrival order, uninterpreted.

        Uninterpreted deliberately: mapping an Agent Runtime event onto this platform's
        published vocabulary is a separate responsibility in a separate module, and a
        connection that also mapped would change both when the transport moved and when
        the vocabulary did.

        Typed as a generator rather than as a plain iterator so a consumer can close it:
        an abandoned one stays parked on the queue until the interpreter tears down its
        async generators, which is well after the socket it was reading from went.
        """
        while True:
            yield await self._notifications.get()

    async def start_thread(self, request: ThreadStartRequest) -> str:
        """Create the Session's one root thread and return the runtime's thread id.

        The id is returned so later calls can carry it. It is pod-local and never leaves
        the pod: what a tenant holds is the platform's own Session identifier, and no
        response this platform serializes carries the value returned here (ADR-007).

        `request` is built from the compiled configuration for this Session, which is
        another component's output — this method neither reads nor validates that
        configuration, it sends what it was handed.
        """
        result = await self._request(REPERTOIRE[RepertoireMethod.THREAD_START], request)
        return ThreadStartResponse.model_validate(result).thread.id

    async def resume_thread(self, request: ThreadResumeRequest) -> str:
        """Re-attach to a thread from a restored Rollout file. Returns its id."""
        result = await self._request(
            REPERTOIRE[RepertoireMethod.THREAD_RESUME], request
        )
        return ThreadResumeResponse.model_validate(result).thread.id

    async def read_thread(self, request: ThreadReadRequest) -> ThreadReadResponse:
        """Read a thread back, optionally with its Turns, as the runtime holds it."""
        result = await self._request(REPERTOIRE[RepertoireMethod.THREAD_READ], request)
        return ThreadReadResponse.model_validate(result)

    async def set_goal(self, request: ThreadGoalSetRequest) -> None:
        """Set the thread's objective and token budget. Reported on, not enforced."""
        await self._request(REPERTOIRE[RepertoireMethod.THREAD_GOAL_SET], request)

    async def start_turn(self, request: TurnStartRequest) -> str:
        """Begin a Turn and return its id. Its events arrive over `notifications()`."""
        result = await self._request(REPERTOIRE[RepertoireMethod.TURN_START], request)
        return TurnStartResponse.model_validate(result).turn.id

    async def steer_turn(self, request: TurnSteerRequest) -> str:
        """Redirect the active Turn, returning the id the Agent Runtime steered.

        Fails when `expected_turn_id` is not the active Turn, which is the point of the
        field: the alternative is a steer written for one Turn landing on the next.
        """
        result = await self._request(REPERTOIRE[RepertoireMethod.TURN_STEER], request)
        return TurnSteerResponse.model_validate(result).turn_id

    async def interrupt_turn(self, request: TurnInterruptRequest) -> None:
        """Stop the named Turn. Its already-appended events stay in the Event Log."""
        await self._request(REPERTOIRE[RepertoireMethod.TURN_INTERRUPT], request)

    async def _write(
        self, entry: RepertoireEntry, params: BaseModel, request_id: str | None
    ) -> None:
        """Serialize one Repertoire call. The only place a method name reaches the wire.

        Membership is re-checked even though holding an entry already implies it,
        because an entry can be constructed by hand and the contract is that such an
        attempt is refused rather than passed through. The check runs before anything
        is written, so a refused call puts nothing at all on the socket — the request
        is absent, which is a stronger property than the Agent Runtime answering it
        with an error.
        """
        if REPERTOIRE.get(entry.method) is not entry:
            raise NotInRepertoire(f"{entry.method} is not the declared entry")
        if self._conn is None:
            raise RuntimeConnectionClosed(f"{entry.method} attempted before connect")
        message: dict[str, Any] = {"method": entry.method.value}
        if request_id is not None:
            message["id"] = request_id
        body = params.model_dump(by_alias=True, exclude_none=True, mode="json")
        if body:
            message["params"] = body
        await self._conn.send(json.dumps(message))

    async def _request(
        self, entry: RepertoireEntry, params: BaseModel
    ) -> dict[str, Any]:
        if entry.response_model is None:
            raise NotInRepertoire(f"{entry.method} is a notification, not a request")
        request_id = str(uuid4())
        waiter: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = (entry.method, waiter)
        try:
            await self._write(entry, params, request_id)
            return await waiter
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, entry: RepertoireEntry, params: BaseModel) -> None:
        if entry.response_model is not None:
            raise NotInRepertoire(f"{entry.method} is a request, not a notification")
        await self._write(entry, params, None)

    def _fail_pending(self, why: str) -> None:
        """Fail every call still waiting for an answer that is not coming.

        Called wherever the inbound path stops, because a waiter left alone when the
        reader ends never completes: the caller would hang for the life of the pod
        rather than see the socket go away. The dict is copied first — settling a
        future can run its awaiting task, which pops its own entry.
        """
        for _, waiter in list(self._pending.values()):
            if not waiter.done():
                waiter.set_exception(RuntimeConnectionClosed(why))

    async def _read_forever(self, conn: ClientConnection) -> None:
        """Route inbound frames: responses to their waiter, notifications to the queue.

        An inbound server-to-client request is answered with method-not-found rather
        than handled. Threads are started with approvals off, so the Agent Runtime
        raises none; anything else arriving is a capability nobody has decided to
        support, and answering it is how the inbound half stays as closed as the
        outbound half.

        The connection is passed in rather than read off the instance so this task
        cannot observe a half-built state: it is created after the socket is open and
        it holds the socket it was created for.

        **The socket ending is this task's ordinary terminal condition, not an error.**
        It used to escape, and nobody awaits this task except `close`, so the only
        place it could surface was there -- inside the shim's lifespan shutdown, which
        then failed with `Application shutdown failed. Exiting.` and tore the process
        down hard. That cut the response stream the control plane was still reading and
        reported a Turn that had done its work as `runtime_lost`. Callers waiting on a
        call are not left in the dark by swallowing it: the `finally` below is what
        tells them, and it says the same thing either way.
        """
        try:
            async for frame in conn:
                message: dict[str, Any] = json.loads(frame)
                request_id = message.get("id")
                if request_id is not None and "method" in message:
                    await conn.send(
                        json.dumps(
                            {
                                "id": request_id,
                                "error": {
                                    "code": _METHOD_NOT_FOUND,
                                    "message": "unsupported inbound request",
                                },
                            }
                        )
                    )
                    continue
                if request_id is None:
                    await self._notifications.put(message)
                    continue
                pending = self._pending.get(str(request_id))
                if pending is None or pending[1].done():
                    continue
                method, waiter = pending
                error = message.get("error")
                if error is None:
                    waiter.set_result(message.get("result") or {})
                else:
                    waiter.set_exception(
                        RuntimeCallFailed(
                            method, int(error.get("code", 0)), str(error.get("message"))
                        )
                    )
        except ConnectionClosed:
            return
        finally:
            self._fail_pending("the control socket closed under an in-flight call")
