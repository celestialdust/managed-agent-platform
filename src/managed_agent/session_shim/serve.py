"""The pod-side process: it accepts one Session's Turns and runs them next door.

This is the half of the platform that runs inside the Session pod. The Agent Runtime
listens on a unix socket in this same pod (ADR-001) and the control plane runs outside
it, so something in here has to be the thing a Turn is handed to. That is this app, and
until it existed every Turn was refused for want of a listener in the pod
(`control/session/turn_dispatch.py`).

**What crosses the pod boundary is mapped events, never runtime frames.** `run_turn` is
driven here rather than in the control plane for that reason: the runtime's
notifications carry `threadId`, `turnId` and `itemId` beside their payload, and the
mapping in `turn_runner._MAPPED` is what stops one riding along in a field nobody
thought about (ADR-013). Mapping in the control plane would mean shipping the raw frames
there first, which is the thing being prevented.

**What this app does not hold is a database credential.** `run_turn` asks for an
`EventLogAppend`; what it is given here writes to the response stream, and the control
plane performs the real append at the other end. The Event Log's table carries no
tenant, so a Postgres credential in this pod would read and write every tenant's log --
from a process that shares a pod with model-driven code.

**And what it may put on that stream is a closed set, for the same reason.** The control
plane appends what arrives here into that untenanted table, so any event type this app
can name is an event type a compromised pod can write. `ShimEventType` is a `Literal` of
the four types a Turn produces rather than a free string: `session.stopped` folds a
Session to STOPPED and would let a pod refuse all of its own later Turns, and
`turn.submitted` is how a replayed submission is recognised. Neither is something the
pod may say about itself.

One pod holds one Session. Every Turn this app accepts names a Session, and one that
names a different Session is refused exactly as an unauthenticated one is: the two
refusals are byte-identical, so a caller cannot use the answer to learn which Session a
pod is serving.

**Half of running this is built and half is not, and the difference is worth stating.**
The app, the route and the connection to the runtime are real, and are exercised in
process against a real unix-socket server. Nothing in this tree builds the image the
pod's containers run, and nothing implements the cluster client that would place the
pod -- so no test here creates a pod, and this process has never been started inside
one.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import stat
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO, Final, Literal, get_args
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, FastAPI, Header, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from managed_agent.control.files.output_shipout import OUTPUT_COUNT_LIMIT
from managed_agent.control.pod_config.compiler import PROFILE_NAME, WORKSPACE_ROOT
from managed_agent.core.errors import STATUS_FOR, ErrorCode, ErrorEnvelope
from managed_agent.core.ids import Seq, SessionId, TurnId
from managed_agent.core.pod.repertoire import (
    ThreadResumeRequest,
    ThreadStartRequest,
)
from managed_agent.core.pod.workspace_contract import (
    INPUT_DIR_NAME,
    OUTPUT_DIR_NAME,
    is_a_produced_path,
    is_a_synced_path,
)
from managed_agent.core.vfs.session_vfs import (
    MAX_RELATIVE_LEN,
)
from managed_agent.session_shim.client import CONTROL_SOCKET_PATH, RuntimeConnection
from managed_agent.session_shim.seed_rollout import find_seeded, thread_id_at
from managed_agent.session_shim.turn_complete import (
    RUNTIME_HOME,
    RolloutNotFound,
    find_rollout,
)
from managed_agent.session_shim.turn_runner import run_turn

SHIM_PORT: Final = 8081
"""The port the shim container publishes and the control plane dials.

Declared here and imported by `pod_channel.py` so the two halves cannot disagree.
`deploy/k8s/session-pod.yaml` cannot import it, so `tests/deploy/
test_session_pod_runs_a_shim.py` compares the manifest against this constant instead.
"""

SHIM_SERVICE: Final = "map-session"
"""The headless Service whose name is the pod's `subdomain`.

A bare pod has no DNS record. With `subdomain: map-session` set on the pod and a
headless Service of that name selecting it, `<pod-name>.map-session.<ns>.svc` resolves
to the pod -- which is what lets the address be computed from the pod name rather than
stored beside it.
"""

TURN_ROUTE: Final = "/session/turn"
READY_ROUTE: Final = "/session/ready"

ROLLOUT_ROUTE: Final = "/session/{session_id}/rollout"
"""Where the control plane reads this Session's Rollout from, at a completed Turn.

The Session id is in the path rather than taken from the served Session, so a refusal
can echo what the caller claimed. A route that answered with the served id in its
refusal body would tell an unauthenticated caller which Session this pod is running,
which is the fact the Turn route's two identical refusals exist to withhold.
"""

OUTPUTS_ROUTE: Final = "/session/{session_id}/outputs"
"""Where the control plane learns what this Session's agent has produced.

A listing rather than a bulk transfer, because the two bounds the control plane owes --
how many files it will take and how many bytes -- can only be applied before a transfer
starts. An archive of the whole directory would arrive as one body of unknown size and
the decision to refuse it would come after it had already been read.

No 204 here, unlike `ROLLOUT_ROUTE`. That route needs one because an empty body and a
body holding an empty file are the same bytes; a JSON listing carries its own emptiness,
so "the agent produced nothing" is `{"files": []}` and needs no status to disambiguate.
"""

OUTPUT_ROUTE: Final = "/session/{session_id}/outputs/{name:path}"
"""Where one file the agent produced is read from. The read counterpart of FILE_ROUTE.

The name is re-parsed here for the reason the write route re-parses one: it arrives over
the network, and a process that trusts a path from the network reads whatever the sender
names -- and this container's read mount covers the whole workspace, so a traversal here
has somewhere to arrive that the write route's does not.
"""


def output_path_for(session_id: SessionId, relative: str) -> str:
    """The read route's path for one produced file, with its path segment encoded.

    A function rather than `OUTPUT_ROUTE.format(...)` at each caller, because the
    template now carries FastAPI's `:path` converter and `str.format` reads `path` as a
    format spec and raises on it. One builder rather than a second literal spelled out
    beside the route, so the route and the URL that reaches it cannot drift apart.

    Encoded with `/` left safe, which is the whole difference from encoding a bare name:
    the separators ARE the path here, and percent-encoding them would address a single
    segment literally called `report/fig1.png`. Everything else is still escaped, so a
    name the agent chose that carries `?` or `#` cannot address a different route.
    """
    return OUTPUT_ROUTE.replace("{session_id}", str(session_id)).replace(
        "{name:path}", quote(relative, safe="/")
    )


WORKSPACE_ROUTE: Final = "/session/{session_id}/workspace"
"""Where the whole working tree is listed, for the sync into the `working` lane.

The outputs pair above answers "what did the agent produce"; this answers "what does the
agent's workspace hold". They are different questions with different rules -- `out/` is
excluded here and is the whole of what is included there -- so they are different routes
rather than one route with a flag, which would be one handler whose two halves share
nothing but a path.
"""

WORKSPACE_FILE_ROUTE: Final = "/session/{session_id}/workspace/{name:path}"
"""Where one file of the working tree is read from. The read counterpart above.

Read-only, like the outputs routes and unlike `FILE_ROUTE`. Restoring a working tree
into a pod would need a WRITE mount of the workspace root, and this container mounts the
workspace `subPath: files` on purpose -- see `WORKSPACE_FILES`. So the lane is written
from the pod and is not yet read back into one; `control/files/workspace_sync.py` says
what that costs and what would have to change.
"""


def workspace_path_for(session_id: SessionId, relative: str) -> str:
    """The read URL for one working file, built without `str.format`.

    `{name:path}` is a Starlette converter and `str.format` reads `path` as a format
    spec, so formatting this template raises `ValueError`. Encoded with `/` left safe,
    for the reason `output_path_for` gives: the separators are the path's structure.
    """
    return WORKSPACE_FILE_ROUTE.replace("{session_id}", str(session_id)).replace(
        "{name:path}", quote(relative, safe="/")
    )


FILE_ROUTE: Final = "/session/{session_id}/files/{name}"
"""Where the control plane places one of this Session's attached files.

A write, and the only one this pod accepts from outside. The pod holds no cloud
identity and cannot read the object store itself (ADR-004), so a file a tenant uploaded
reaches the workspace by being pushed down the hop that is already authenticated rather
than by the pod being given a credential to go and fetch it.

The Session id is in the path for the same reason `ROLLOUT_ROUTE` carries one: a refusal
can echo what the caller claimed instead of disclosing which Session this pod runs.
"""

WORKSPACE_FILES: Final = Path(WORKSPACE_ROOT) / INPUT_DIR_NAME
"""The one directory this process may write into, and it is enforced by the mount.

This container mounts the workspace volume with `subPath: files`, so the directory below
is the whole of what it can reach -- not a convention this code keeps, but the only path
that exists for it. That matters because this is the pod's outward-facing process: a
mount of the whole workspace would let whatever reaches this port write anything the
agent later reads, including over a file the agent itself produced.

The same absolute path is what the runtime container sees, because that container mounts
the volume whole at `/session/workspace`. One string for both ends;
`tests/deploy/test_session_pod_runs_a_shim.py` compares it against the manifest, which
cannot import it.
"""

WORKSPACE_READ_ROOT: Final = Path("/session/produced")
"""Where this container sees the whole workspace, read-only, to ship out what it holds.

A **second** mount of the workspace volume, beside the read-write `subPath: files` one
above, and the narrowing on that one is untouched. This container is the pod's only
outward-facing process and the reason the write mount is narrowed is that a caller
holding the shim token must not be able to write over what the agent produced. Reading
what the agent produced is the whole of this feature, so the read has to be widened and
the write does not -- and `readOnly: true` here is what keeps those two facts separate
in the manifest rather than in a convention this code keeps.

A different path rather than the workspace's own, unlike `WORKSPACE_FILES`. That
constant is one string for both containers because the shim writes a file the *agent*
must then open, so the two have to agree on where it is. Nothing the agent does depends
on this path: what leaves here is a bare leaf name and bytes, and no agent-visible path
is ever built from either. Mounting the volume a second time at `/session/workspace`
would nest inside the read-write mount and leave the result depending on the order
kubelet applies them.

`tests/deploy/test_session_pod_runs_a_shim.py` compares this against the manifest,
which cannot import it.
"""

SHIM_TOKEN_PATH: Final = Path("/etc/map/shim/token")
"""Where this Session's bearer token is mounted, for this container only.

A file rather than an environment variable: the runtime container in this pod must not
be able to read it, and a mount is per-container while `env` on the pod is not. It is
also kept out of `CODEX_HOME`, so a confined process reading its own configuration tree
does not find it.
"""

RUNTIME_WAIT_ATTEMPTS: Final = 30
RUNTIME_WAIT_SECONDS: Final = 2.0
"""How long this process waits for the runtime's socket, and why it is these numbers.

Thirty attempts two seconds apart is the same sixty seconds the runtime container's own
`startupProbe` budgets for that socket to appear (`deploy/k8s/session-pod.yaml`). The
two containers start together, so the manifest is already saying that the socket is not
instant; matching its budget rather than inventing a second one keeps one answer to
"how long is the runtime allowed to take".
"""

_NDJSON: Final = "application/x-ndjson"

_ROLLOUT_CHUNK_BYTES: Final = 64 * 1024
"""How much of a Rollout is held in memory at once while it is served.

`FileResponse`'s own chunk size, kept rather than chosen: the point of streaming this
file is that its size is unbounded, and the number that bound was already the one this
process was living with.
"""
_BEARER: Final = "Bearer "
_STREAM_BACKLOG: Final = 256

ShimEventType = Literal[
    "turn.started",
    "turn.message_delta",
    "turn.completed",
    "turn.failed",
    "tool.called",
    "thread.started",
    "tool.server_unavailable",
]
"""The only event types the pod may put on the wire, spelled out rather than imported.

`Literal` accepts literal strings only, so these cannot be the constants in
    `core/vocabulary/` even though they must equal them. Two tests are the link, and one
    of them was added because the other's promise turned out to be false.

    `tests/session_shim/test_shim_serves_a_turn.py` compares this set against the
    published `turn` family minus `turn.submitted`, plus the two members named from
    outside it. It used to claim that a mapping emitting a sixth type would therefore
    fail it rather than failing a Turn in production. It did not: `thread.started` is in
    the `thread` family, so it changed neither side of that comparison and the type
    shipped in an image that killed every Turn with a validation error from this model.
    `tests/session_shim/test_turn_runner.py` now also asserts that EVERY type `_MAPPED`
    can emit is in this set, derived from the map itself, which is the form no new
    family is invisible to.

`turn.submitted` is the one turn type excluded, because the control plane writes it when
it admits a Turn and a pod that could write one could plant an idempotency key.

`tool.called` is the one member from outside the turn family, and it is here rather than
excluded because the pod is the ONLY process that can see a tool call at all -- the
runtime reports one to the shim over a unix socket and nothing else in the platform is
told. Admitting it widens what a compromised pod may write, and the widening is bounded
by what the type carries: a server name, a tool name, a status and a duration, all four
already chosen by the tenant's own registration. A pod inventing a `tool.called` can
therefore claim a call that did not happen -- worth writing down, because it means this
event is evidence about what the runtime reported and not proof a third party was
reached. The Tool Gateway is where a claim like that could be checked against a request
that really crossed it, and it keeps no such record today.

`thread.started` is the second member from outside the turn family, admitted on the
    same terms and with the same bounded damage. The pod is again the only process that
    can see a subagent begin, and a compromised pod writing one can invent a thread that
    never ran -- the same class of claim as an invented `tool.called`, and for the same
    reason not a forgery of anything outside its own Session. What it cannot do is name
    another tenant's Session or plant an idempotency key, which are the two admissions
    this set exists to withhold.

`tool.server_unavailable` is the third outsider, and it is here because the pod is once
    more the only process that can see the thing: the Agent Runtime announces a tool
    server's startup outcome over the same unix socket, and until this type existed a
    Session whose granted server never came up ran a Turn that looked, in the log,
    exactly like one whose agent chose not to call anything.

    The damage a compromised pod can do with it runs the other way from the two above,
    and is worth stating because it is not obvious. An invented `tool.called` claims
    work that did not happen; an invented `tool.server_unavailable` claims an excuse --
    it can make a Turn that simply did nothing read as a Turn the platform failed. That
    is a claim about this Session's own placement, checkable against the Tool Gateway's
    view of whether anything ever connected, and it names no other tenant and plants no
    idempotency key. The alternative -- withholding it -- leaves the unhonoured grant
    invisible, which is the failure this type exists to end.
"""

SHIM_EVENT_TYPES: Final[frozenset[str]] = frozenset(get_args(ShimEventType))
"""The same four as a set, for a caller holding a type it has not parsed.

Derived from the annotation rather than restated beside it, so the wire type and the
control plane's own check cannot come to disagree about what is allowed.
"""


class RunTurn(BaseModel):
    """What the control plane sends. Unknown fields are refused, not ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: SessionId
    turn_id: TurnId
    prompt: str = Field(min_length=1)


class TurnEventLine(BaseModel):
    """One already-mapped platform event, on its way out of the pod.

    `type` is closed rather than a free string because the far end of this stream
    appends into the Event Log. This module's docstring says what an open one would
    let a pod write about its own Session.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["event"] = "event"
    type: ShimEventType
    payload: dict[str, object] = Field(default_factory=dict)


class TurnCompletedLine(BaseModel):
    """The terminal line, written only when the Turn actually completed.

    Carried as its own line rather than as a flag on the last event, because "the Turn
    completed" and "the stream stopped" are different facts and the control plane acts
    differently on each: the first ships the Rollout out, the second means the pod went
    quiet mid-Turn and the Turn is already recorded as failed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["completed"] = "completed"


TurnLine = Annotated[TurnEventLine | TurnCompletedLine, Field(discriminator="kind")]


class ProducedFileEntry(BaseModel):
    """One file the agent left at its working root, as the listing reports it.

    The length is reported so the far end can check the transfer that follows against
    it. A produced file is not append-only the way a Rollout is, so a body that stops
    short has no torn-tail reading: it is a partial document, and a partial document
    stored under the whole one's name is the failure worth spending a field to catch.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    byte_length: int = Field(ge=0)
    content_sha256: str | None = None
    """SHA-256 of the file's bytes, lowercase hex, as they were at listing time.

    **Optional on the wire, and that is a compatibility fact rather than a preference.**
    A Session's pod runs the shim image it was started with for the whole life of the
    Session, so a control plane deployed today reads listings from pods that started
    before this field existed. Required here, every Turn already in flight would fail
    at its completion the moment the control plane rolled -- which is the one failure a
    durability boundary must not have. So absence means "this pod does not report
    digests", and the far end verifies when it is present and skips when it is not.

    The digest is a check on the TRANSFER, not a lock on the file. It is computed in one
    pass at listing time and the body is fetched afterwards, so an agent still writing
    when the listing was taken produces a mismatch -- which is the honest answer, since
    what arrived is then not the document that was listed.
    """


class ProducedFiles(BaseModel):
    """Everything the agent produced, or the first `OUTPUT_COUNT_LIMIT` + 1 of it.

    `files` may hold one entry more than the limit, and that extra entry is the signal
    rather than a payload: it says the workspace holds more than the far end will take,
    without this process having listed a directory of unknown size to find out how many
    more. The far end refuses on it; nothing here decides what "too many" costs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: tuple[ProducedFileEntry, ...] = ()


class StreamingEventSink:
    """Hands each mapped event to the response stream instead of to a database.

    Satisfies `EventLogAppend` structurally because that is the port `run_turn` asks
    for. The `Seq` it returns is the count of events this Turn has streamed and **is
    not the platform's sequence** -- that is assigned by the real append in the control
    plane, and a pod that chose one would be writing the gap `core/ports.py` warns
    about. Nothing reads this value; `run_turn` discards every append's return. It is
    the count rather than a constant so that a reader who does look at it in a debugger
    sees a position in this stream and cannot mistake it for a log sequence that
    happens to be stuck.

    The queue is bounded by its owner. An unbounded one turns a control plane that
    stopped reading into a pod that grows until the node evicts it, and the pod is the
    process with no operator watching it.
    """

    def __init__(self, out: asyncio.Queue[TurnEventLine | TurnCompletedLine | None]):
        self._out = out
        self._streamed = 0

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        """Queue one mapped event as a line, refusing a type the pod may not write.

        Parsed rather than constructed: the port hands over a plain `str` and the
        line's own field is the closed set, so this is the boundary where one becomes
        the other. A type outside the set raises here, inside the pod, and the Turn
        ends with no completion line -- which the control plane reads as a Turn that
        produced no answer. That is the right end for it: the alternative is a line the
        far end has to decide about after it has already left the pod.
        """
        self._streamed += 1
        await self._out.put(
            TurnEventLine.model_validate(
                {"kind": "event", "type": type_, "payload": dict(payload)}
            )
        )
        return Seq(self._streamed)


class StreamedCompletion:
    """Writes the terminal line, and only when the Turn reached a completion.

    Satisfies `TurnCompleted`, which `run_turn` notifies once after a Turn's last event
    is recorded -- and deliberately does not notify for a Turn that went quiet. So the
    presence of this line in the stream *is* the distinction between a Turn that ended
    and a Turn that stopped, and the control plane needs no second signal for it.

    What acts on the completion is the control plane, not this: shipping the Rollout
    out of the pod has failure modes a stream writer should not own.
    """

    def __init__(self, out: asyncio.Queue[TurnEventLine | TurnCompletedLine | None]):
        self._out = out

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        await self._out.put(TurnCompletedLine())


@dataclass(frozen=True, slots=True)
class ServedSession:
    """The one Session this process serves, and what running its Turns needs.

    Process-wide rather than per-request, because one pod holds one Session. The thread
    id is minted at startup by `start_thread` and never leaves the pod (ADR-007): it is
    passed to `run_turn` and appears in nothing this app writes to the stream.
    """

    session_id: SessionId
    thread_id: str
    connection: RuntimeConnection
    token: str


router = APIRouter(tags=["shim"])


def _served_from_request(request: Request) -> ServedSession:
    """The `ServedSession` the entry point installed, typed once.

    `app.state` is untyped by construction, so the one read of it is funnelled through
    here rather than left as an `Any` that silently disables checking downstream. The
    `assert` narrows rather than validates: only this module's two factories write the
    attribute, so a wrong type here is a programming error in the entry point and not
    something a request can cause.
    """
    served = request.app.state.served
    assert isinstance(served, ServedSession)
    return served


def _presented_this_sessions_token(header: str | None, served: ServedSession) -> bool:
    """Whether the caller presented the token derived for this Session.

    `compare_digest` rather than `==`: the comparison is against a secret, and a
    short-circuiting equality tells a caller who can time the answer how much of the
    token it guessed right.

    Compared as **bytes**, which is not a detail. `compare_digest` on two `str` values
    raises `TypeError` the moment either holds a non-ASCII character, and a header
    value arrives here decoded latin-1 straight off the wire -- so one byte above 0x7F
    from an unauthenticated caller would leave this route with an unhandled exception
    and a bare 500 instead of the platform's refusal. Encoding first makes every
    presented value comparable, so every wrong one is refused the same way.
    """
    if header is None or not header.startswith(_BEARER):
        return False
    presented = header[len(_BEARER) :].encode("utf-8", "surrogateescape")
    return hmac.compare_digest(presented, served.token.encode())


def _refuse_without_saying_which(session_id: SessionId) -> JSONResponse:
    """The one refusal this route emits, for both of the two ways in.

    A caller with no token and a caller naming another Session get byte-identical
    answers, so neither learns which Session this pod is serving. The published
    `session.not_found` is reused rather than a code of its own: adding a member to the
    closed set is a version event under ADR-013.
    """
    envelope = ErrorEnvelope(
        code=ErrorCode.SESSION_NOT_FOUND,
        message="no such session is served by this pod",
        detail={"session_id": str(session_id)},
    )
    return JSONResponse(
        status_code=STATUS_FOR[ErrorCode.SESSION_NOT_FOUND],
        content=envelope.model_dump(mode="json"),
    )


@router.get(READY_ROUTE)
async def ready(request: Request) -> Response:
    """204 once this pod's Session is open, 503 until then.

    A route of its own rather than a probe pointed at the Turn route, because a GET on
    a POST-only route answers 405 and kubelet counts every 4xx as not-ready -- a pod
    probed that way never becomes ready, is never published by the headless Service,
    and every Turn to it fails as undeliverable while the shim works perfectly.

    What it reports is that the Session is *open*, not that the process is up. Be
    precise about the window, though: uvicorn completes lifespan startup before it
    listens, so while this shim is still waiting for the runtime's socket a probe gets a
    refused connection rather than this 503. Both read as not-ready, and the pod stays
    out of the headless Service either way. The 503 is what this answers if it is ever
    reached with no Session open -- a state a request cannot cause in the pod, and the
    right answer if some other entry point ever can.
    """
    served = getattr(request.app.state, "served", None)
    if not isinstance(served, ServedSession):
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(TURN_ROUTE, response_model=None)
async def run_a_turn(
    body: RunTurn,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> StreamingResponse | JSONResponse:
    """Run one Turn against the runtime next door, streaming what it produces.

    Both checks happen before the runtime is touched at all, so a refused request puts
    nothing on the control socket -- the Turn is absent rather than started and then
    abandoned, which is the stronger property.

    `response_model=None` is on the decorator because the return type is a union of two
    responses; without it FastAPI reads that annotation as a body schema and refuses
    the route at definition time, which makes this whole module unimportable.
    """
    served = _served_from_request(request)
    if not _presented_this_sessions_token(authorization, served):
        return _refuse_without_saying_which(body.session_id)
    if body.session_id != served.session_id:
        return _refuse_without_saying_which(body.session_id)
    return StreamingResponse(_stream_the_turn(body, served), media_type=_NDJSON)


@router.get(ROLLOUT_ROUTE)
async def current_rollout(
    session_id: SessionId,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """This Session's Rollout as the runtime has written it so far.

    A read. The control plane calls it after a Turn completes and puts what came back in
    the object store, because the pod has no cloud identity and cannot write there
    itself (ADR-004). The two output routes below make that trip for what the agent
    wrote.

    Both checks happen before the filesystem is touched, and both refuse identically, so
    a caller with no token and a caller naming another Session learn the same nothing.

    **204 for a pod that has written no Rollout yet, not 404 and not an empty body.** A
    Session's first Turn can complete before the runtime has flushed a file, and a
    control plane that read empty bytes as a Rollout would overwrite a good stored one
    with nothing. The distinction is carried in the status because the body cannot carry
    it: an empty 200 and a 200 holding an empty file are the same bytes.

    **A file that exists and holds nothing gets that same 204**, which is the sentence
    the paragraph above needs in order to be true. The runtime creates its record before
    it flushes the first line into it, so "no Rollout yet" is two states on disk and
    only one of them is the file being absent -- and `find_rollout` locates a path, not
    bytes, so `RolloutNotFound` cannot report the other. Serving 200 over zero bytes
    hands the control plane an empty body it cannot tell from a real Rollout, and the
    ship-out that follows replaces a good stored record with nothing; from there every
    resume reads a Rollout with no lines and refuses, and only a completed Turn replaces
    the object, which needs a resume. The control plane refuses to store empty bytes
    whatever the status says as well -- see `control/files/rollout_sync.py` -- because
    one stat cannot promise what the read will find.

    The file is streamed rather than read into memory. It grows with the length of the
    thread -- the one superlinear cost term this platform has, and it is not bounded --
    and this process shares a pod with model-driven code, so buffering it here is the
    read most worth not doing. A last line torn by a write in flight is expected and is
    handled at the far end, where the truncation drops it.

    **Streamed with no declared length, and bounded to the size read above.** A
    `FileResponse` would set `content-length` from its own stat and then read to
    whatever EOF it found, and this file is one the runtime may be appending to while
    the transfer runs -- the truncation at the far end exists because that is normal. A
    body that outgrows the header is not a torn tail the far end can drop: uvicorn
    raises `RuntimeError("Response content longer than Content-Length")` and drops the
    connection, the fetch sees `httpx.RemoteProtocolError`, and the completed Turn is
    recorded as failed with nothing shipped. Sending no length makes the mismatch
    unrepresentable, and stopping at `size` makes the body one prefix of one moment
    rather than a stripe across several.
    """
    served = _served_from_request(request)
    if not _presented_this_sessions_token(authorization, served):
        return _refuse_without_saying_which(session_id)
    if session_id != served.session_id:
        return _refuse_without_saying_which(session_id)
    try:
        path = find_rollout(RUNTIME_HOME, served.thread_id)
    except RolloutNotFound:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    size = path.stat().st_size
    if size == 0:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return StreamingResponse(_rollout_prefix(path, size), media_type=_NDJSON)


@router.get(OUTPUTS_ROUTE)
async def list_the_outputs(
    session_id: SessionId,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """What the agent produced, by lane-relative path, length and digest.

    The two authorisation checks are the rollout route's, in the same order and with the
    same two identical refusals, and both run before the filesystem is touched.

    **What counts as produced is "a regular file at or below `out/`".** The platform
    tells the agent where to put a deliverable (`workspace_contract`), so the directory
    it was told about is the answer, and the walk descends -- a report and its figures
    are one deliverable with a shape, and flattening it would be the platform deciding
    that a deliverable is always a single file. The workspace root is the fallback when
    `out/` holds nothing, and the fallback does NOT descend: without the convention
    there is no way to tell a project tree from a set of documents, and walking one is
    the unbounded transfer `OUTPUT_COUNT_LIMIT` exists to prevent.

    Two things are left out either way. Anything that is not a regular file -- a socket,
    a fifo, a dangling symlink -- has no bytes to ship. And a path `is_a_produced_path`
    rejects is left out because it could not be stored at the far end at all: a dotted
    segment at any depth is runtime scratch or an installed dependency tree rather than
    a document. That predicate is shared with the control plane rather than restated, so
    a path this offers and the far end refuses cannot exist.

    Sorted by path, so two listings of one unchanged workspace are the same listing and
    the entry past the limit is not an arbitrary one.
    """
    served = _served_from_request(request)
    if not _presented_this_sessions_token(authorization, served):
        return _refuse_without_saying_which(session_id)
    if session_id != served.session_id:
        return _refuse_without_saying_which(session_id)
    return JSONResponse(content=_produced_files().model_dump(mode="json"))


@router.get(WORKSPACE_ROUTE)
async def list_the_workspace(
    session_id: SessionId,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """The agent's whole working tree, by path, length and digest.

    The same two authorisation checks as every other route here, in the same order.

    **Everything the outputs listing excludes on purpose is included here, and that is
    the point of having two.** A working file IS the scratch -- a half-written script,
    a cache under `src/.cache/`, the project tree -- and the reason to keep it is that
    the agent will want it back. What is excluded is `NOT_SYNCED`: the tenant's own
    uploads, the deliverables that go to the other lane, and an installed dependency
    tree that is rebuildable by definition.

    A dotfile at the workspace *root* is excluded too, and by the lane grammar rather
    than by that list -- it composes to no key, so `.gitignore` is absent here. That is
    a known gap. `.codex/` is absent for the same mechanical reason and is not a loss:
    the runtime keeps its conversation and tool state under `CODEX_HOME`, on a volume
    outside the workspace, and ships it out by `ROLLOUT_ROUTE` rather than any lane.

    The digest on every entry is what makes the far end's diff possible: it uploads only
    the paths whose bytes differ from what the lane already holds, so a Session whose
    workspace did not change costs one listing and no transfer at all.
    """
    served = _served_from_request(request)
    if not _presented_this_sessions_token(authorization, served):
        return _refuse_without_saying_which(session_id)
    if session_id != served.session_id:
        return _refuse_without_saying_which(session_id)
    return JSONResponse(content=_workspace_files().model_dump(mode="json"))


@router.get(WORKSPACE_FILE_ROUTE)
async def read_a_workspace_file(
    session_id: SessionId,
    name: str,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """One working file's bytes, or 204 if nothing is there now.

    The outputs read route with the other predicate and no shipping prefix: the working
    lane is the workspace root, so the path is relative to the root itself.

    204 rather than 404 for a vanished path, matching the outputs route, and for the
    same reason: a file the listing named a moment ago and the agent has since removed
    is not an error, and the far end skips it rather than failing the Turn over it.
    """
    served = _served_from_request(request)
    if not _presented_this_sessions_token(authorization, served):
        return _refuse_without_saying_which(session_id)
    if session_id != served.session_id:
        return _refuse_without_saying_which(session_id)
    if not is_a_synced_path(name):
        return _refuse(
            ErrorCode.REQUEST_INVALID,
            "a working file's path must be relative and outside the reserved roots",
        )
    handle = _opened_without_following(name)
    if handle is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return StreamingResponse(
        _produced_prefix(handle, os.fstat(handle.fileno()).st_size),
        media_type="application/octet-stream",
    )


@router.get(OUTPUT_ROUTE)
async def read_an_output(
    session_id: SessionId,
    name: str,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """One produced file's bytes, streamed and bounded to the size read here.

    The authorisation checks and the name check are `place_a_file`'s, in that order, so
    nothing is opened for a caller that has not proved which Session it is talking to.

    **204 when the name no longer names a regular file**, which is the same answer the
    rollout route gives for a pod that has written nothing. A file can be listed and
    then unlinked -- an agent that tidies up after its own Turn does exactly that -- and
    that is not a failure to report: there is no document there to lose. A 404 would be
    the refusal this route reserves for a caller talking about the wrong Session.

    Streamed with no declared length and stopped at the size stat here, for the reason
    `current_rollout` explains: a body that outgrows a `content-length` is not a short
    read the far end can reason about, it is a dropped connection. What the far end does
    with a body shorter than the listing said is refuse it -- see
    `control/files/output_shipout.py` -- so a file being rewritten under this read costs
    the Turn rather than storing half a document.
    """
    served = _served_from_request(request)
    if not _presented_this_sessions_token(authorization, served):
        return _refuse_without_saying_which(session_id)
    if session_id != served.session_id:
        return _refuse_without_saying_which(session_id)
    if not is_a_produced_path(name):
        return _refuse(
            ErrorCode.REQUEST_INVALID,
            "a produced file's path must be relative, with no dotted segment",
        )
    handle = _opened_without_following(_shipping_prefix() + name)
    if handle is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return StreamingResponse(
        _produced_prefix(handle, os.fstat(handle.fileno()).st_size),
        media_type="application/octet-stream",
    )


@router.put(FILE_ROUTE)
async def place_a_file(
    session_id: SessionId,
    name: str,
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Write one of this Session's attached files into the workspace.

    The two authorisation checks are the rollout route's, in the same order and with the
    same two identical refusals, so a caller with no token and a caller naming another
    Session learn the same nothing. They run before the name is looked at and before the
    filesystem is touched.

    **The name is re-parsed here even though the control plane already parsed it.** It
    arrives over the network, and a process that trusts a path from the network writes
    wherever the sender says: `..%2f..%2f` in a path segment, or a name that is simply
    `..`, would leave this subtree. The mount is the outer guard -- this container can
    reach nothing but the directory below -- and this is the inner one, because a mount
    narrowed to a subtree still lets a traversal reach a *different file inside it*, and
    the file it would most usefully overwrite is another of this Session's.

    **Streamed to disk, never buffered.** The default upload cap is 100 MiB and this
    process shares a pod with model-driven code, so reading the body into memory is the
    read most worth not doing -- the same reasoning the rollout route streams out for.

    **Written to a temporary name and renamed into place.** A crashed or truncated
    transfer must not leave a partial file under the real name: the agent cannot tell a
    half-written document from a whole one, and would summarise the half. A rename
    within one directory is atomic, so the name either does not exist or holds it all.

    204, not 201: there is nothing to return and no location to name that the caller did
    not just choose.
    """
    served = _served_from_request(request)
    if not _presented_this_sessions_token(authorization, served):
        return _refuse_without_saying_which(session_id)
    if session_id != served.session_id:
        return _refuse_without_saying_which(session_id)
    if not _is_a_bare_leaf(name):
        return _refuse(
            ErrorCode.REQUEST_INVALID, "a file name must be a single path component"
        )
    WORKSPACE_FILES.mkdir(parents=True, exist_ok=True)
    final = WORKSPACE_FILES / name
    partial = WORKSPACE_FILES / f".{name}.partial"
    try:
        with partial.open("wb") as sink:
            async for chunk in request.stream():
                sink.write(chunk)
        partial.replace(final)
    except OSError as failed:
        partial.unlink(missing_ok=True)
        return _refuse(
            ErrorCode.INTERNAL, f"the file could not be written: {failed.strerror}"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _refuse(code: ErrorCode, message: str) -> JSONResponse:
    """One refusal in the published envelope, with no detail beyond the message.

    Deliberately carries no `detail`: this route's two refusals are about the caller's
    own request, and echoing the name or the errno back into a structured field would
    put a value the caller chose into a place a log aggregator indexes.
    """
    return JSONResponse(
        status_code=STATUS_FOR[code],
        content=ErrorEnvelope(code=code, message=message).model_dump(mode="json"),
    )


def _is_a_bare_leaf(name: str) -> bool:
    """Whether this name refers to a file in one directory and nowhere else.

    Written as an allowlist of what a name may be rather than a denylist of separators,
    because the set of ways to spell "parent directory" is longer than it looks and
    grows with whatever decodes the path next. `Path(name).name == name` rejects every
    name holding a separator, `""`, `"."` and `".."`; the leading-dot refusal keeps a
    caller from naming this route's own `.<name>.partial` scratch file, and keeps an
    attached file from being one the agent's tools skip as hidden.
    """
    return bool(name) and Path(name).name == name and not name.startswith(".")


def _shipping_prefix() -> str:
    """Which directory this Turn's deliverables are in: `out/` if it has any, else root.

    **Two roots because a convention the model ignores must not lose the work.** The
    platform tells the agent to put deliverables in `out/`
    (`core/pod/workspace_contract.py`), and an agent that followed that is the case
    worth optimising: ship-out returns what the tenant asked for and nothing else. But
    an agent that did not follow it -- an older prompt, a model that forgot, a Turn
    where the instruction lost a race with the tenant's own -- must still have its
    document collected. Preferring `out/` and falling back makes the untidy outcome
    "your file arrives with a scratch script beside it" instead of "your file is gone".

    "Has any" means at least one shippable regular file at any depth below it, not
    merely existing. An empty `out/` is what a model that created the directory and then
    wrote elsewhere leaves behind, and treating that as the answer would ship nothing
    from a Turn that produced something -- the exact failure the fallback exists to
    prevent.

    **Returns a prefix rather than a path, so every open starts at the read mount and
    walks down.** `out/` is a directory the *agent* creates, from the other container,
    so it can be a symlink -- and a root handed out as an absolute path is one some
    caller opens without checking what it resolved through. Returning the prefix keeps
    the one trusted starting point in `_opened_without_following`, which refuses to
    traverse a symlink at any segment including this one.

    **This decides for the listing AND the read, which is why it is a function and not a
    branch inside one of them.** A name listed out of `out/` and then opened at the root
    is a 204 for a file that exists, and the far end reads that as a vanished document.
    The two calls are still separate moments: an agent that creates `out/` between them
    moves the root under the read, which yields 204 and a refused Turn rather than the
    wrong bytes. That is the same outcome as any file unlinked mid-Turn, and
    `control/files/output_shipout.py` already refuses a short read rather than storing
    half a document.
    """
    found: list[ProducedFileEntry] = []
    try:
        _collect_produced(found, WORKSPACE_READ_ROOT / OUTPUT_DIR_NAME, "", True, 1)
    except OSError:
        return ""
    return f"{OUTPUT_DIR_NAME}/" if found else ""


def _not_a_dotted_directory(relative: str) -> bool:
    """Whether the produced walk should enter this directory.

    A dotted directory holds runtime state or an installed dependency tree, and
    `is_a_produced_path` refuses every path with a dotted segment anyway -- so entering
    one can only cost the walk. Read the segment rather than the whole path, because a
    dotted parent has already been refused and its children are reached only through it.
    """
    return not relative.rsplit("/", 1)[-1].startswith(".")


def _worth_entering_for_the_working_lane(relative: str) -> bool:
    """Whether the working walk should enter this directory.

    Only the three `NOT_SYNCED` roots are skipped, and only at the workspace root: a
    directory named `out` one level down is the agent's own. A dotted directory below
    the first segment is entered -- unlike the produced walk, which refuses one at any
    depth -- because the agent's tree is the thing being kept rather than filtered.

    A dotted directory AT the root is not entered, and that is `is_a_synced_path`
    deferring to the lane grammar rather than a choice made here. See its docstring:
    the cost is `.codex/`, and it is recorded there rather than restated.
    """
    return is_a_synced_path(relative)


_DIGEST_BLOCK: Final = 1 << 20
"""How much of a file is held in memory at once while hashing it.

A megabyte: large enough that the syscall count is irrelevant beside the read itself,
small enough that hashing a large file costs a bounded amount of this pod's memory.
"""

_MAX_PRODUCED_DEPTH: Final = 16
"""How far below the output directory a produced file may sit and still be shipped.

A deliverable is a document or a small tree of them -- a report with its figures, a site
with its assets. Sixteen is far past anything an agent writing for a tenant produces and
far short of what a recursive walk of a dependency tree would need, so the bound costs
no real output and stops a symlink-free but pathological tree from recursing this
process into a stack overflow. Depth and not just path length, because 512 characters of
path can still be 250 directories.
"""


def _collect_produced(
    found: list[ProducedFileEntry],
    directory: Path,
    prefix: str,
    descend: bool,
    depth: int,
    keep: Callable[[str], bool] = is_a_produced_path,
    enter: Callable[[str], bool] = _not_a_dotted_directory,
) -> None:
    """Every regular file at or below `directory` that `keep` accepts, bounded as we go.

    **Two predicates and not one, because "do not ship this file" and "do not walk
    this directory" are different claims.** `keep` decides a file, and is the rule that
    has to agree with the far end. `enter` decides whether a directory can hold anything
    `keep` would accept -- an optimisation and never a correctness boundary, so a wrong
    `enter` costs a walk and cannot admit a file `keep` refuses. That split is what lets
    the produced walk skip `.map/lib` without opening it while the working walk descends
    into `src/.cache/`, whose files it keeps. Neither walk enters a dotted directory at
    the workspace root: `_worth_entering_for_the_working_lane` IS `is_a_synced_path`,
    which parses through the lane grammar, and a leading dot composes to no key.

    **The scan stops at `OUTPUT_COUNT_LIMIT` + 1 entries rather than reading the whole
    tree and truncating.** Reading it all would make this process's memory a function of
    how many files the agent chose to create, which is the one thing about the workspace
    nothing bounds -- an `emptyDir` `sizeLimit` bounds bytes, not inodes. The cost is
    that a tree over the limit yields an arbitrary subset, and that costs nothing real:
    the far end refuses on the count and never looks at the names.

    **`descend` is false at the workspace root and true under `out/`, and that asymmetry
    is the whole reason this takes the flag.** Under `out/` the agent has said what it
    produced, so a tree there is a deliverable with its parts and walking it is what
    delivers `report/fig1.png`. At the root nothing has been said: descending there
    would walk `files/` and hand the tenant their own uploads back, walk the runtime's
    directories, and turn the fallback into the unbounded transfer `OUTPUT_COUNT_LIMIT`
    exists to prevent. The fallback stays exactly as flat as it has always been.

    `follow_symlinks=False` at every test is not tidiness. A symlink the agent placed
    here would be resolved inside *this* container, whose mounts include the Session's
    bearer token -- a file the agent cannot read and this process can. Excluded at the
    listing and refused again at the open, because a check and an open are two moments
    and the agent shares this filesystem.
    """
    with os.scandir(directory) as scan:
        for entry in scan:
            if len(found) > OUTPUT_COUNT_LIMIT:
                return
            relative = prefix + entry.name
            if len(relative) >= MAX_RELATIVE_LEN:
                continue
            if entry.is_dir(follow_symlinks=False):
                if descend and depth < _MAX_PRODUCED_DEPTH and enter(relative):
                    with suppress(OSError):
                        _collect_produced(
                            found,
                            Path(entry.path),
                            relative + "/",
                            True,
                            depth + 1,
                            keep,
                            enter,
                        )
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            if not keep(relative):
                continue
            found.append(
                ProducedFileEntry(
                    name=relative,
                    byte_length=entry.stat(follow_symlinks=False).st_size,
                    content_sha256=_digest_of(entry.path),
                )
            )


def _digest_of(path: str) -> str | None:
    """SHA-256 of one file, lowercase hex, or None if it could not be read.

    Chunked rather than read whole, because this runs over every file in the listing
    and a workspace may hold one larger than this process should hold in memory at once.

    `O_NOFOLLOW` for the same reason the read route uses it: a name that was a plain
    file when `scandir` reported it can be a symlink by the time this opens it, and the
    two are separate moments on a filesystem the agent shares. None on any error rather
    than raising -- a file that vanished mid-listing costs its digest, not the listing,
    and the far end treats a missing digest as "not reported" either way.

    The hex must equal `control/files/store.content_digest` over the same bytes. It is
    written out here rather than imported so the pod-side module does not depend on a
    control-plane one for a hash; `tests/session_shim/` holds the guard that the two
    agree, because two spellings of one rule are two rules that can drift.
    """
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        with os.fdopen(descriptor, "rb") as handle:
            for block in iter(lambda: handle.read(_DIGEST_BLOCK), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _workspace_files() -> ProducedFiles:
    """The agent's working tree, by workspace-relative path, length and digest.

    The same walk the produced listing uses, with the other pair of predicates and no
    shipping prefix: the working lane IS the workspace root, so a path here is relative
    to the root itself rather than to a directory below it.

    Bounded by the same `OUTPUT_COUNT_LIMIT` + 1 signal, so the far end learns "more
    than will be taken" without this process counting an unbounded tree. What the far
    end does with that differs -- a workspace over the limit is synced in part rather
    than refused -- and that is `control/files/workspace_sync.py`'s call, not this
    route's.
    """
    found: list[ProducedFileEntry] = []
    try:
        _collect_produced(
            found,
            WORKSPACE_READ_ROOT,
            "",
            True,
            1,
            is_a_synced_path,
            _worth_entering_for_the_working_lane,
        )
    except OSError:
        return ProducedFiles()
    return ProducedFiles(files=tuple(sorted(found, key=lambda e: e.name)))


def _produced_files() -> ProducedFiles:
    """What the agent produced, by lane-relative path and by length.

    Sorted by path, so two listings of one unchanged workspace are the same listing and
    the entry past the limit is not an arbitrary one. At or under the limit the whole
    tree was read, which is the case where stability is worth having; over it the far
    end refuses on the count and never looks at the names.

    A missing root reads as nothing produced. In the pod it always exists, having been
    made by the init container; in a test it need not.
    """
    prefix = _shipping_prefix()
    root = WORKSPACE_READ_ROOT / prefix if prefix else WORKSPACE_READ_ROOT
    found: list[ProducedFileEntry] = []
    try:
        _collect_produced(found, root, "", bool(prefix), 1)
    except OSError:
        return ProducedFiles()
    return ProducedFiles(files=tuple(sorted(found, key=lambda e: e.name)))


def _opened_without_following(relative: str) -> BinaryIO | None:
    """The file at that path below the read mount, or None when it is not a plain file.

    **Every segment is opened with `O_NOFOLLOW`, not just the last one.** While a
    produced name was one path component, an `O_NOFOLLOW` on the leaf was the whole
    guard; a nested path has intermediate directories, and a symlinked directory in the
    middle would be traversed by a single `os.open` of the joined path -- which resolves
    every component but the final one. So this descends with `dir_fd`, refusing a
    symlink at each step, and it starts at `WORKSPACE_READ_ROOT` rather than at whatever
    `_shipping_prefix` chose, so `out/` itself is one of the segments checked.

    That matters because the agent writes this tree from the other container. The file
    most worth naming through a symlink is this container's own `/etc/map/shim/token`:
    the agent cannot read it, this process can, and shipping it out would hand the
    tenant a credential to their own pod's one write route. `ELOOP` from the kernel
    closes that window in a way no check-then-open can, because a check and an open are
    two moments and the agent is writing between them.

    None rather than an exception for every way this can fail, because the caller does
    the same thing with all of them: a path that is a directory, one unlinked after it
    was listed, and one that became a symlink are all "there is no produced document
    here", which is not a failure to report to a control plane that listed it a moment
    ago.
    """
    segments = relative.split("/")
    try:
        directory = os.open(WORKSPACE_READ_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return None
    try:
        for segment in segments[:-1]:
            below = os.open(
                segment,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = below
        descriptor = os.open(
            segments[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory
        )
    except OSError:
        return None
    finally:
        os.close(directory)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            return None
    except OSError:
        os.close(descriptor)
        return None
    return os.fdopen(descriptor, "rb")


async def _produced_prefix(handle: BinaryIO, size: int) -> AsyncIterator[bytes]:
    """The open file's first `size` bytes, in chunks, closing it however this ends.

    Bounded by the stat the caller took, for the reason `_rollout_prefix` is: a body
    that outgrows a declared length drops the connection. Unlike a Rollout there is no
    torn-tail reading available here -- a produced file is not append-only, so a short
    body is a partial document rather than a record with an unfinished last line, and
    what the far end does about it is refuse the transfer against the length the listing
    reported.

    The handle arrives open because opening it is where the symlink refusal lives, and
    that decision belongs to the route rather than to a generator the route may never
    start. Closed in a `with` so a control plane that disconnects mid-transfer does not
    leave a descriptor held for the life of the pod.
    """
    remaining = size
    with handle:
        while remaining > 0:
            chunk = handle.read(min(_ROLLOUT_CHUNK_BYTES, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk


async def _rollout_prefix(path: Path, size: int) -> AsyncIterator[bytes]:
    """The file's first `size` bytes, in chunks, stopping early if it shrank.

    Bounded by the caller's stat rather than by EOF, so an append that lands while this
    is running is not half-served. Stopping short on a shrink is deliberate too: the
    record is append-only, so a file that lost bytes was rotated or truncated under us
    and the honest answer is the prefix that was still there, which the truncation at
    the far end reads as a torn tail.
    """
    remaining = size
    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(_ROLLOUT_CHUNK_BYTES, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk


async def _stream_the_turn(
    body: RunTurn, served: ServedSession
) -> AsyncIterator[bytes]:
    """Yield each mapped event as it is produced, then end.

    `run_turn` is driven on its own task rather than awaited first, because it writes
    into the queue this generator drains: awaiting it would collect the whole Turn
    before a byte reached the control plane, and a Turn's answer is worth having while
    it is still being produced.

    The task is cancelled in a `finally` so a control plane that disconnects mid-Turn
    does not leave a Turn running against the runtime with nobody reading it.
    """
    out: asyncio.Queue[TurnEventLine | TurnCompletedLine | None] = asyncio.Queue(
        maxsize=_STREAM_BACKLOG
    )
    running = asyncio.create_task(_drive_the_turn(body, served, out))
    try:
        while (line := await out.get()) is not None:
            yield line.model_dump_json().encode() + b"\n"
    finally:
        running.cancel()
        with suppress(asyncio.CancelledError):
            await running


async def _drive_the_turn(
    body: RunTurn,
    served: ServedSession,
    out: asyncio.Queue[TurnEventLine | TurnCompletedLine | None],
) -> None:
    """Run one Turn to its end, then close the stream exactly once.

    The sentinel goes in a `finally` so a `run_turn` that raises still ends the stream.
    The control plane then sees a stream that stopped without a completion line, which
    is the true reading: this Turn produced no answer.
    """
    try:
        await run_turn(
            body.session_id,
            body.turn_id,
            served.thread_id,
            body.prompt,
            served.connection,
            StreamingEventSink(out),
            StreamedCompletion(out),
        )
    finally:
        await out.put(None)


async def connect_when_the_runtime_is_listening(
    socket_path: Path = CONTROL_SOCKET_PATH,
    attempts: int = RUNTIME_WAIT_ATTEMPTS,
    pause_s: float = RUNTIME_WAIT_SECONDS,
) -> RuntimeConnection:
    """Open the control socket, waiting out the runtime's own start-up.

    Retried rather than attempted once, and the difference is the whole pod. Both
    containers start together and the runtime binds its socket some time after that, so
    a shim that dials immediately usually finds nothing there. A lifespan that raises
    is fatal to uvicorn and the pod's `restartPolicy` is `Never`, so losing that race
    once would mean the shim never runs again for the life of that Session -- with
    correct code, and nothing to see but a pod that stayed unready.

    Waiting is safe rather than merely convenient: the readiness route answers 503 for
    the whole of this window and the headless Service publishes only ready pods, so a
    Turn dispatched into it is refused rather than accepted by a process that cannot
    run it.

    A fresh `RuntimeConnection` per attempt, because a failed `connect()` leaves its
    exit stack part-entered and reusing it would dial into that. The last failure is
    raised once the budget is spent: a socket that has not appeared in a minute is not
    late, it is absent.
    """
    latest: OSError = FileNotFoundError(
        f"the runtime's control socket at {socket_path} was never dialled"
    )
    for _ in range(attempts):
        connection = RuntimeConnection(socket_path)
        try:
            await connection.connect()
        except OSError as not_listening_yet:
            latest = not_listening_yet
            await asyncio.sleep(pause_s)
            continue
        return connection
    raise latest


async def open_the_thread(connection: RuntimeConnection) -> str:
    """Continue this Session's conversation, or open its first one. Returns the id.

    **Which of the two happens is decided by one fact: whether a Rollout is on disk.**
    The `seed-rollout` init container puts one there when this Session has completed a
    Turn, and refuses to let the pod start when it should have and could not -- so by
    the time this runs, a file present means "continue this" and a file absent means
    "this Session has never run". Reading the environment for a second opinion would be
    a second answer to a settled question, and the way those two disagree is a shim
    that opens a fresh thread over a seeded record: the history the Rollout's compaction
    checkpoints folded gets replayed, the tenant is billed for it, and the Turn reports
    success (ADR-004).

    The thread id sent alongside the path comes out of the record's own `session_meta`,
    so the conversation and the identifier it is resumed under cannot be about different
    Sessions. The runtime's own precedence puts a non-empty path above the id, and the
    id is what it verifies the file against.

    Neither the model nor the working directory is re-sent on a resume, and that is the
    protocol rather than an omission: a resumed thread takes both from the record it is
    resuming, which is the same model this Session pinned at creation and the same
    workspace root every pod of it mounts.
    """
    seeded = find_seeded(RUNTIME_HOME)
    if seeded is None:
        return await connection.start_thread(
            ThreadStartRequest(
                cwd=WORKSPACE_ROOT,
                model=os.environ["MAP_MODEL"],
                model_provider=os.environ["MAP_MODEL_PROVIDER"],
                permissions=PROFILE_NAME,
            )
        )
    return await connection.resume_thread(
        ThreadResumeRequest(
            thread_id=thread_id_at(seeded),
            path=str(seeded),
            permissions=PROFILE_NAME,
        )
    )


def create_shim_app(served: ServedSession) -> FastAPI:
    """Build the app around an already-connected Session. Takes no environment.

    Separated from `build_shim_app` for the reason `control/api/app.py` is separated
    from `asgi.py`: a test drives the real route against a scripted runtime, and
    nothing about reading the environment is in the way.
    """
    app = FastAPI(title="Managed Agent Session-shim", version="v1")
    app.state.served = served
    app.include_router(router)
    return app


def build_shim_app() -> FastAPI:
    """The pod's entry point:

        uvicorn managed_agent.session_shim.serve:build_shim_app --factory --port 8081

    A factory, because the connection cannot be opened at import time and a module that
    tried would fail in every context that merely imports it.

    The connection is opened and the thread started in the lifespan rather than here,
    because both are async and both must have happened before the first Turn arrives.
    Until they have, the readiness route answers 503 and the pod stays out of DNS.

    The environment reads have no defaults. A shim that guessed its own Session id
    would serve Turns for a Session it is not running, and a missing token file is a
    shim that would have to choose between refusing everyone and admitting anyone.
    """
    app = FastAPI(title="Managed Agent Session-shim", version="v1")
    app.include_router(router)

    @asynccontextmanager
    async def _open_the_session(_: FastAPI) -> AsyncIterator[None]:
        connection = await connect_when_the_runtime_is_listening()
        thread_id = await open_the_thread(connection)
        app.state.served = ServedSession(
            session_id=SessionId(UUID(os.environ["MAP_SESSION_ID"])),
            thread_id=thread_id,
            connection=connection,
            token=SHIM_TOKEN_PATH.read_text().strip(),
        )
        try:
            yield
        finally:
            await connection.close()

    app.router.lifespan_context = _open_the_session
    return app
