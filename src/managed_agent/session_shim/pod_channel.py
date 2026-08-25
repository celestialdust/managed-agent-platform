"""Reaching the Session-shim in a placed pod, and recording what its Turn produced.

This is the control-plane half of the transport whose absence made every Turn
undeliverable. It locates the Session's pod through `placement`, hands the Turn to the
shim over the pod network, and appends the events the shim streams back.

**It parses no runtime frame.** What arrives here is already mapped to this platform's
published vocabulary by the shim, inside the pod, which is what keeps a runtime thread
id or turn id from crossing the boundary in a field nobody thought about (ADR-013). The
append happens here rather than in the pod because the Event Log's table carries no
tenant: a credential for it inside a Session pod would read and write every tenant's
log, from a process sharing a pod with model-driven code.

**And the pod is the untrusted end of this connection.** It supplies the events; the
Session they are appended under comes from the caller, so a compromised pod reaches no
other tenant -- but every event type it names is refused unless it is one of the four a
Turn produces, because the types it could otherwise name act on its own Session. A
`session.stopped` here folds that Session to STOPPED and it refuses all of its own later
Turns; an unpublished type is filtered out of the SSE surface on the way to the tenant,
so it would sit in the durable log with nothing reporting it.

Both the address and the token are computed from what `placement` already answers, never
stored. That is the same reason `placement.py` computes the pod's name instead of
recording it: a stored address is a second source, free to disagree with the cluster
about where a Session is, and a stored secret is a table to lose. A control-plane
replica that did not place the pod derives both by arithmetic.

**Nothing in this tree can place that pod yet.** `PodRunner` has no implementation
outside test fakes (MAP-55) and no Dockerfile builds the image its containers run
(MAP-56), so every test of this module dials a shim app in the same process over an ASGI
transport. What is proven here is the wire and the refusals; what is not is that a
Kubernetes pod ever answers on it.
"""

from __future__ import annotations

import hmac
import json
from hashlib import sha256
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from managed_agent.control.files.output_shipout import OUTPUT_COUNT_LIMIT, ProducedFile
from managed_agent.control.session.placement import Placement, PodPhase
from managed_agent.control.session.pods import SessionPods
from managed_agent.control.session.turn_dispatch import TurnUndeliverable
from managed_agent.core.ids import SessionId, TurnId
from managed_agent.core.ports import EventLogAppend
from managed_agent.core.session.event_append import append_in_order
from managed_agent.core.vocabulary import placement as placement_events
from managed_agent.core.vocabulary import turn
from managed_agent.session_shim.serve import (
    FILE_ROUTE,
    OUTPUTS_ROUTE,
    ROLLOUT_ROUTE,
    SHIM_EVENT_TYPES,
    SHIM_PORT,
    SHIM_SERVICE,
    TURN_ROUTE,
    WORKSPACE_ROUTE,
    ProducedFiles,
    TurnCompletedLine,
    TurnLine,
    output_path_for,
    workspace_path_for,
)
from managed_agent.session_shim.turn_runner import TurnCompleted

_LINE: Final[TypeAdapter[TurnLine]] = TypeAdapter(TurnLine)
MAX_LINES_PER_TURN: Final = 100_000
"""How many lines one Turn may stream before the control plane gives up on it.

Public because a test has to build a stream that crosses it, and a test that hard-coded
a second number would be measuring its own constant. Far above any real Turn: what this
catches is a pod with nothing left to say and no intention of stopping.
"""
_TERMINAL: Final = frozenset({turn.TURN_COMPLETED, turn.TURN_FAILED})
_ACCEPTED: Final = 200

_CONNECT_DEADLINE_S: Final = 10.0
_INTER_BYTE_DEADLINE_S: Final = 120.0
_SHIM_TIMEOUT: Final = httpx.Timeout(
    connect=_CONNECT_DEADLINE_S,
    read=_INTER_BYTE_DEADLINE_S,
    write=_CONNECT_DEADLINE_S,
    pool=_CONNECT_DEADLINE_S,
)
"""The deadlines one Turn's HTTP call is held to, and what each of them bounds.

`read` is the one that matters, and in httpx it is an **inter-byte** deadline rather
than a total one: it bounds the gap between two bytes arriving, so a Turn that
legitimately runs for an hour with steady output is untouched by it while a pod that
accepts a Turn and then goes silent fails in two minutes. Without it such a pod hangs
this dispatch for the life of the process -- and `control/api/routes/turns.py` awaits
the dispatch inside the tenant's own POST, so the tenant's request hangs with it and a
control-plane worker is held for as long as the pod stays wedged.

The line cap below is not a substitute and does not overlap: it bounds a pod that
streams too much, and this bounds a pod that streams nothing.

There is no total deadline, and that is a gap rather than a decision. The Turn Ceiling
this platform defines is denominated in tokens, so it is not a wall-clock bound and
cannot stand in for one; nothing in this tree says how long a Turn may take.
"""


ROLLOUT_FETCH_LIMIT_BYTES: Final = 256 * 1024 * 1024
"""How large a Rollout this control plane will pull into memory before giving up.

A guess, and named as one. A Rollout's size grows with a Session's age and its
distribution is undecided, so there is no measured number to put here. What the cap buys
is the failure mode: past it the ship-out fails loudly and closes the Turn, instead of a
control-plane worker being killed by the OOM reaper while serving one tenant's Session.
The pod's codex-home volume is an emptyDir with no sizeLimit, so nothing upstream bounds
the input.
"""

_NO_ROLLOUT_YET: Final = 204
_PLACED: Final = 204
_NOTHING_AT_THAT_NAME: Final = 204

_LISTING_LIMIT_BYTES: Final = (OUTPUT_COUNT_LIMIT + 1) * 1024
"""How large a listing of produced files this control plane will read.

Derived from the count limit rather than chosen, so it cannot come to disagree with it.
A kilobyte per entry is four times the longest name a filename may hold plus its length
and its punctuation, so every listing a real pod can send fits with room to spare -- and
a pod padding one with megabytes of whitespace is refused on the way in rather than
after it is in this worker's memory.
"""

_FILE_TRANSFER_DEADLINE_S: Final = 120.0
_FILE_TIMEOUT: Final = httpx.Timeout(
    connect=_CONNECT_DEADLINE_S,
    read=_FILE_TRANSFER_DEADLINE_S,
    write=_FILE_TRANSFER_DEADLINE_S,
    pool=_CONNECT_DEADLINE_S,
)
"""The deadlines a file placement is held to, and why `write` is not the connect one.

This is the only hop in this module that sends a large body, so it is the only one
where the write deadline is a transfer deadline rather than a handshake one. The rollout
read's `write` is `_CONNECT_DEADLINE_S` because its request body is empty; reusing that
here would abort a 100 MiB push after ten seconds of perfectly healthy transfer.

Inter-byte in both directions, not total: the default upload cap is 100 MiB and no
sensible total bound covers both a 2 KiB document and that, so what is bounded is a
stall. Two minutes of no progress is a pod that has stopped.
"""

_ROLLOUT_TRANSFER_DEADLINE_S: Final = 30.0
_ROLLOUT_TIMEOUT: Final = httpx.Timeout(
    connect=_CONNECT_DEADLINE_S,
    read=_ROLLOUT_TRANSFER_DEADLINE_S,
    write=_CONNECT_DEADLINE_S,
    pool=_CONNECT_DEADLINE_S,
)
"""The deadlines the Rollout read is held to, and why they are not the Turn's.

httpx's `read` is an inter-byte deadline in both cases, so this is not a total transfer
bound either -- what differs is what a legitimate gap looks like. A Turn's 120 s allows
for a model thinking between tokens; this is a file being served off a local disk, so a
thirty-second gap between two chunks is a pod that has stopped rather than one that is
slow. Reusing the Turn's would hold a control-plane worker four times as long for a pod
that is never going to answer.
"""


def _shim_host(pod_name: str, namespace: str) -> str:
    """The pod's DNS name. One construction, read by both routes below."""
    return f"{pod_name}.{SHIM_SERVICE}.{namespace}.svc.cluster.local"


def shim_url_for(pod_name: str, namespace: str) -> str:
    """The one address a Session's shim ever has.

    Resolvable because the pod sets `hostname` to its own name and `subdomain:
    map-session`, and a headless Service of that name selects it -- the three together
    are what give a bare pod a stable DNS record. The hostname is the half that is easy
    to leave out and impossible to notice: without it this name does not resolve at all
    and every Turn fails as undeliverable while the pod reads 2/2 Running, which is what
    shipped until 2026-08-23. `tests/deploy/test_session_pod_runs_a_shim.py` asserts the
    manifest carries all three.
    """
    return f"http://{_shim_host(pod_name, namespace)}:{SHIM_PORT}{TURN_ROUTE}"


def shim_rollout_url_for(session_id: SessionId, pod_name: str, namespace: str) -> str:
    """Where this Session's Rollout is read from, on the pod placement just answered."""
    route = ROLLOUT_ROUTE.format(session_id=session_id)
    return f"http://{_shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_file_url_for(
    session_id: SessionId, name: str, pod_name: str, namespace: str
) -> str:
    """Where one of this Session's attached files is placed, on the pod just answered.

    `name` is interpolated raw and that is safe for a reason the caller owns rather
    than this function: every name reaching this point came through
    `parse_upload_filename`, which admits no separator and no dot-dot. The shim
    re-parses it anyway -- a receiver that trusts a path from the network writes
    wherever the sender says, and "the sender is us" is not a property it can check.
    """
    route = FILE_ROUTE.format(session_id=session_id, name=name)
    return f"http://{_shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_outputs_url_for(session_id: SessionId, pod_name: str, namespace: str) -> str:
    """Where this Session's produced files are listed, on the pod just answered."""
    route = OUTPUTS_ROUTE.format(session_id=session_id)
    return f"http://{_shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_workspace_url_for(session_id: SessionId, pod_name: str, namespace: str) -> str:
    """Where this Session's whole working tree is listed, on the pod just answered."""
    route = WORKSPACE_ROUTE.format(session_id=session_id)
    return f"http://{_shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_workspace_file_url_for(
    session_id: SessionId, name: str, pod_name: str, namespace: str
) -> str:
    """Where one working file is read from, on the pod placement just answered.

    `workspace_path_for` rather than a `quote` here, for the reason the produced builder
    gives: the separators are the path's structure and encoding them would name a file
    with `%2F` in its name, which is a different file and does not exist.
    """
    route = workspace_path_for(session_id, name)
    return f"http://{_shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_output_url_for(
    session_id: SessionId, name: str, pod_name: str, namespace: str
) -> str:
    """Where one produced file is read from, on the pod placement just answered.

    `name` is a lane-relative path and may carry separators, so the encoding is the
    shim's `output_path_for` rather than a `quote` here: separators have to survive it
    or the URL addresses one segment literally named `report/fig1.png`, and everything
    else has to be escaped or a name carrying `?` addresses a different route. That is
    two rules pulling opposite ways, which is exactly the kind of pair worth having in
    one function beside the route it builds for.

    Encoded at all, which `shim_file_url_for` does not need to be: every name it
    interpolates came through `parse_upload_filename` in this process, while a path here
    came off the pod's own listing and is a value the untrusted end chose.
    """
    route = output_path_for(session_id, name)
    return f"http://{_shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_token_for(session_id: SessionId, key: bytes) -> str:
    """This Session's bearer token for its own shim, derived rather than stored.

    Scoped to one Session by construction: a pod that reads its own token holds nothing
    that reaches another Session's shim, which is what bounds the damage when the
    sandbox this platform relies on fails -- and today it is known non-functional. The
    derivation also means any control-plane replica can produce the token, including one
    that did not place the pod, and that a restart recovers it by arithmetic rather than
    by a lookup in a table of secrets.
    """
    return hmac.new(key, str(session_id).encode(), sha256).hexdigest()


def refuse_an_event_the_pod_may_not_write(type_: str, session_id: SessionId) -> None:
    """Raise unless `type_` is one of the four types a Turn produces.

    A second gate over the same set the wire model already enforces, and deliberately
    so: `TurnEventLine.type` stops the defect at the parse, and this stops a caller that
    reached an append without parsing one. They read one set -- `SHIM_EVENT_TYPES` is
    derived from that annotation -- so the two cannot come to disagree about what is
    allowed.

    Raising rather than returning a bool, because there is nothing else a caller could
    do with the answer: an event the pod may not write must not be appended and must not
    be skipped past either, and `TurnUndeliverable` is what closes the Turn in
    `control/api/routes/turns.py` instead of leaving it open on a stream nobody trusts.
    """
    if type_ not in SHIM_EVENT_TYPES:
        raise TurnUndeliverable(
            f"the shim for session {session_id} tried to write the event type "
            f"{type_!r}, which is not one a Turn produces"
        )


def _shape_of(raw: str) -> str:
    """A refused line described by its structure, carrying none of its content.

    The message this feeds used to name only the session and the Turn, which made the
    refusal impossible to act on: every cause -- an unknown `kind`, an event type
    outside the closed set, a field the models forbid, a line that is not JSON at all --
    produced one sentence, and finding out which meant reproducing the Turn with a
    debugger attached. That happened, and it cost most of an afternoon.

    What it must not do is put the line itself in a log. The payload carries the model's
    own words about a tenant's work, and this exception is logged by the control plane
    and returned to the caller. So only the *shape* crosses: the key names, and the two
    discriminator values that decide which model a line is validated against. Both are
    drawn from a closed vocabulary this module already publishes; neither can hold
    tenant text.

    A line that is not JSON at all has no keys to name, so its length is reported
    instead -- enough to tell a truncated line from a log line that reached the stream.
    """
    try:
        parsed = json.loads(raw)
    except ValueError:
        return f"not JSON, {len(raw)} characters"
    if not isinstance(parsed, dict):
        return f"JSON {type(parsed).__name__}, not an object"
    kind = parsed.get("kind")
    type_ = parsed.get("type")
    return f"keys {sorted(str(k) for k in parsed)}, kind={kind!r}, type={type_!r}"


def _one_line(raw: str, session_id: SessionId, turn_id: TurnId) -> TurnLine:
    """One line of the pod's stream, parsed into the shapes it is allowed to be.

    A line that is none of them ends the Turn rather than being logged and skipped. The
    pod is the untrusted end here, so "I could not read that" is not a hiccup to step
    over -- it means the rest of this stream cannot be trusted to be what it claims, and
    the honest answer is a Turn the control plane closes.
    """
    try:
        return _LINE.validate_json(raw)
    except ValidationError as unreadable:
        raise TurnUndeliverable(
            f"the shim for session {session_id} sent a line that is not a turn event, "
            f"for turn {turn_id}: {_shape_of(raw)}"
        ) from unreadable


class HttpPodDispatch:
    """Carries a Turn to the shim in the Session's pod and records what came back.

    The pod's phase is checked before it is dialled, so a Session whose pod is starting
    or gone is refused by the cluster's own answer rather than by a connection timing
    out -- a refusal delivered in milliseconds instead of at the end of a socket
    timeout.

    An **absent** pod is the one phase this does not simply refuse. A Session's pod is
    placed lazily, at the first Turn that finds none, and this is where that happens:
    `SessionPods.ensure_for` either brings one up or says why this Session may not have
    one, and the cluster is asked again afterwards. Starting and gone still fall through
    to the refusal, because neither is an absence -- placing over either would be a
    second pod for one Session.

    Returns once the Turn has finished and its events are appended, which is the
    contract `control/api/routes/turns.py` is written against: a dispatch that raises
    closes the Turn, and a Turn nobody closes is a submission nobody can explain. The
    shim answers its status line as soon as the Turn is accepted and holds the body open
    until the Turn ends, so acceptance and completion stay separately observable.

    A client is opened per Turn rather than held for the life of the process, because
    nothing here could close a long-lived one: `composition.build` hands back the engine
    for exactly that reason and the shutdown hook that disposes it belongs to another
    module. A Turn is seconds to minutes long, so one connect at the start of it costs
    nothing worth owning a resource for. The transport is injectable so a test can point
    this at a shim app in the same process.
    """

    def __init__(
        self,
        placement: Placement,
        pods: SessionPods,
        log: EventLogAppend,
        on_completed: TurnCompleted,
        namespace: str,
        token_key: bytes,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._placement = placement
        self._pods = pods
        self._log = log
        self._on_completed = on_completed
        self._namespace = namespace
        self._token_key = token_key
        self._transport = transport

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        binding = await self._placement.locate(session_id)
        waited_ms = 0
        if binding.phase is PodPhase.ABSENT:
            # Appended BEFORE the wait, not after it, and the order is the whole value
            # of the event. A tenant watching the stream needs to learn that its Turn is
            # queued *while* it is queued -- that is the moment "waiting for a node" and
            # "the model is thinking" are indistinguishable, and the moment the answer
            # is wanted. Appended after, it would arrive beside `turn.started` and tell
            # somebody what had already stopped being true.
            #
            # It is appended even when the placement then fails. That is correct rather
            # than untidy: the Turn did wait, and `control/api/routes/turns.py` closes
            # it with a `turn.failed` whose cause says why -- so the log reads as "this
            # Turn waited for a pod and did not get one", which is what happened.
            await append_in_order(
                self._log,
                session_id,
                placement_events.SESSION_PLACING,
                placement_events.SessionPlacing(turn_id=turn_id).model_dump(
                    mode="json"
                ),
            )
            # A Session with no pod gets one here or is told why it cannot have one;
            # STARTING and GONE fall through to the refusal below exactly as before,
            # because neither is an absence and placing over either would be a second
            # pod for one Session.
            #
            # Wrapped here rather than inside `ensure_for`, and in exactly one of the
            # two places: this is the only site that both measures the wait and holds
            # the log to record it, so counting it in both would report every queued
            # Turn twice on `GET /v1/capacity` -- an inflated depth being the one
            # failure that makes the number worth less than no number at all.
            async with self._placement.awaiting(session_id) as wait:
                await self._pods.ensure_for(session_id)
            waited_ms = wait.elapsed_ms
            # Located again rather than trusting what placement returned: `ensure` can
            # answer for a pod that finished while it waited, and the phase this method
            # acts on has to be the one the cluster reports now.
            binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            raise TurnUndeliverable(
                f"the pod for session {session_id} is {binding.phase.value}"
            )
        token = shim_token_for(session_id, self._token_key)
        try:
            async with (
                httpx.AsyncClient(
                    transport=self._transport, timeout=_SHIM_TIMEOUT
                ) as client,
                client.stream(
                    "POST",
                    shim_url_for(binding.pod_name, self._namespace),
                    json={
                        "session_id": str(session_id),
                        "turn_id": str(turn_id),
                        "prompt": prompt,
                    },
                    headers={"Authorization": f"Bearer {token}"},
                ) as response,
            ):
                if response.status_code != _ACCEPTED:
                    raise TurnUndeliverable(
                        f"the shim for session {session_id} answered "
                        f"{response.status_code}"
                    )
                await self._record(session_id, turn_id, response, waited_ms)
        except httpx.HTTPError as unreachable:
            raise TurnUndeliverable(
                f"the shim for session {session_id} could not be reached"
            ) from unreachable

    async def _record(
        self,
        session_id: SessionId,
        turn_id: TurnId,
        response: httpx.Response,
        waited_ms: int,
    ) -> None:
        """Append each streamed event in arrival order, then act on the completion.

        Order is the whole of it: the platform's sequence is assigned by the append, so
        the order these land in *is* the order a caller reads the Turn in, and a reader
        of the Event Log cannot tell a reordered stream from a wrong one. Each append is
        awaited before the next line is read, for that reason.

        Every line is refused before it is appended rather than after. What this method
        writes is what the pod said to write, into a table with no tenant column, so an
        event type outside the four a Turn produces is a pod acting on its own Session
        through the control plane's credential.

        Bounded twice, at two different failures. The line cap fails a pod that streams
        without stopping; the read deadline on the client fails one that stops streaming
        without ending. Neither covers the other's case, and the second is the likelier.

        **A completion seam that fails becomes `TurnUndeliverable`, whatever it
        raised.** `TurnDispatch` promises a caller of the port never sees a transport
        error, and `control/api/routes/turns.py` catches that one type -- so anything
        else escaping here reaches the tenant as a bare 500 carrying no code from the
        published set, with no `turn.failed` appended. The seam this method awaits
        reaches an object store, whose client raises its own errors sharing no base with
        `httpx.HTTPError`, so the outer handler in `dispatch` does not cover them.
        Failing loudly is the intended outcome (ADR-004 puts the
        Event-Log-ahead-of-the-resume-state divergence in writing); what is fixed here
        is only that it fails as the exception the port declares.

        The two shapes `dispatch` already translates accurately pass through
        untouched. The seam's first act is to read the Rollout back out of the pod,
        so a refused token or an unreachable pod surfaces here as `TurnUndeliverable`
        or as an `httpx` error -- and re-labelling either as "could not be stored"
        would name the object store for a failure that never reached it.

        The cause the Turn is then recorded under is `POD_UNREACHABLE`, which is the
        closest of the three `TurnFailureCause` values and not an exact one -- the pod
        was reachable and the store was not. A precise cause is an addition to that
        closed set in `core/vocabulary/turn.py`, which is a published-vocabulary change
        and another slice's file.
        """
        completed = False
        terminal = False
        seen = 0
        async for raw in response.aiter_lines():
            if not raw:
                continue
            seen += 1
            if seen > MAX_LINES_PER_TURN:
                raise TurnUndeliverable(
                    f"the shim for session {session_id} streamed more than "
                    f"{MAX_LINES_PER_TURN} lines for turn {turn_id}"
                )
            line = _one_line(raw, session_id, turn_id)
            if isinstance(line, TurnCompletedLine):
                completed = True
                continue
            refuse_an_event_the_pod_may_not_write(line.type, session_id)
            terminal = terminal or line.type in _TERMINAL
            # The placement wait is stamped on here rather than inside the pod, because
            # the pod cannot know it: the wait happened out here, before the pod was
            # reachable, and a number the pod reported would be a number it was told. It
            # goes on `turn.started` because that event is the instant the model was
            # given the work, so the two together bracket exactly what the tenant paid
            # for -- everything before it was queueing and everything after was
            # thinking.
            await append_in_order(
                self._log,
                session_id,
                line.type,
                placement_events.with_placement_wait(line.payload, waited_ms)
                if line.type == turn.TURN_STARTED
                else line.payload,
            )
        if not terminal:
            raise TurnUndeliverable(
                f"the shim for session {session_id} stopped mid-turn {turn_id}"
            )
        if completed:
            try:
                await self._on_completed.turn_completed(session_id, turn_id)
            except (TurnUndeliverable, httpx.HTTPError):
                raise
            except Exception as not_made_durable:
                raise TurnUndeliverable(
                    f"the rollout for session {session_id} could not be stored after "
                    f"turn {turn_id}"
                ) from not_made_durable


class PodFilePlacement:
    """Writes one of a Session's attached files into its pod.

    The write counterpart of `PodRolloutFetch`, and built the same way for the same
    reason: the address and the token are computed from what `placement` answers, so
    there is no stored address free to disagree with the cluster and no stored secret to
    lose.

    **A pod that is not running raises rather than answering quietly.** This is the
    opposite of the rollout read's choice and the asymmetry is the point. A missing pod
    at ship-out time means a Turn that already completed and bytes that no longer exist,
    so there is nothing a refusal would save. A missing pod here means the file never
    arrived, and a Session whose attached file is absent will run: the agent will report
    that it cannot find the document, the tenant will read that as the platform having
    lost their upload, and nothing will have said otherwise.

    The body is passed as bytes rather than streamed from the object store, and that
    is a deliberate ceiling rather than an oversight. It bounds this hop's memory at one
    file, which the caller has already checked against a budget; streaming through would
    remove the ceiling and put an unbounded transfer in a control-plane worker.
    """

    def __init__(
        self,
        placement: Placement,
        namespace: str,
        token_key: bytes,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._placement = placement
        self._namespace = namespace
        self._token_key = token_key
        self._transport = transport

    async def place_file(
        self, session_id: SessionId, name: str, body: bytes, /
    ) -> None:
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            raise TurnUndeliverable(
                f"session {session_id} has no running pod to place {name!r} into, so "
                "its attached file would be missing from a Session that starts anyway"
            )
        url = shim_file_url_for(session_id, name, binding.pod_name, self._namespace)
        token = shim_token_for(session_id, self._token_key)
        async with httpx.AsyncClient(
            transport=self._transport, timeout=_FILE_TIMEOUT
        ) as client:
            response = await client.put(
                url, content=body, headers={"Authorization": f"Bearer {token}"}
            )
        if response.status_code != _PLACED:
            raise TurnUndeliverable(
                f"the shim for session {session_id} answered "
                f"{response.status_code} placing {name!r}"
            )


class PodRolloutFetch:
    """Reads a Session's Rollout out of its pod. Satisfies `RolloutFetch`.

    The address and the token are computed from what `placement` answers, the same way
    `HttpPodDispatch` computes them: a stored address is a second source free to
    disagree with the cluster, and a stored secret is a table to lose.

    **A pod that is not running answers None rather than raising.** A Turn cannot have
    completed on a pod that is gone, so reaching here with an absent pod means the pod
    died between the Turn's last event and this read -- and there are no bytes to ship.
    Raising would fail a Turn that did complete and whose events are already in the log.

    Its own client and its own deadlines, not the Turn's. A Turn's read deadline is an
    inter-byte one sized for a model thinking; this is a file transfer that should
    either move or fail.
    """

    def __init__(
        self,
        placement: Placement,
        namespace: str,
        token_key: bytes,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._placement = placement
        self._namespace = namespace
        self._token_key = token_key
        self._transport = transport

    async def fetch_rollout(self, session_id: SessionId, /) -> bytes | None:
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            return None
        url = shim_rollout_url_for(session_id, binding.pod_name, self._namespace)
        token = shim_token_for(session_id, self._token_key)
        async with (
            httpx.AsyncClient(
                transport=self._transport, timeout=_ROLLOUT_TIMEOUT
            ) as client,
            client.stream(
                "GET", url, headers={"Authorization": f"Bearer {token}"}
            ) as response,
        ):
            if response.status_code == _NO_ROLLOUT_YET:
                return None
            if response.status_code != _ACCEPTED:
                raise TurnUndeliverable(
                    f"the shim for session {session_id} answered "
                    f"{response.status_code} for its rollout"
                )
            return await _read_capped(
                response, "rollout", session_id, ROLLOUT_FETCH_LIMIT_BYTES
            )


class PodOutputFetch:
    """Reads what a Session's agent wrote out of its pod. Satisfies `WorkspaceOutputs`.

    Built like `PodRolloutFetch` and for the same reasons -- address and token computed
    from what `placement` answers, its own client, its own deadlines -- and it reuses
    that class's transfer deadlines rather than declaring a third set: a produced file
    and a Rollout are both files served off the pod's local disk, so what a legitimate
    gap between two chunks looks like is the same question with the same answer.

    **The two methods differ on a pod that is not running, and the asymmetry is the
    point.** A listing that finds no pod answers empty, exactly as `PodRolloutFetch`
    does: a Turn cannot have completed on a pod that is gone, so an absent pod at this
    moment means it died after the Turn's last event and there are no bytes to ship --
    raising would fail a Turn that did complete. A *fetch* that finds no pod raises,
    because by then this Session's files have been listed and some may already be
    stored, and a ship-out that stops half way through a set is the one outcome worse
    than shipping nothing: the tenant holds three documents of five and nothing says so.
    """

    def __init__(
        self,
        placement: Placement,
        namespace: str,
        token_key: bytes,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._placement = placement
        self._namespace = namespace
        self._token_key = token_key
        self._transport = transport

    async def list_outputs(self, session_id: SessionId, /) -> tuple[ProducedFile, ...]:
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            return ()
        url = shim_outputs_url_for(session_id, binding.pod_name, self._namespace)
        async with (
            httpx.AsyncClient(
                transport=self._transport, timeout=_ROLLOUT_TIMEOUT
            ) as client,
            client.stream("GET", url, headers=self._bearer(session_id)) as response,
        ):
            if response.status_code != _ACCEPTED:
                raise TurnUndeliverable(
                    f"the shim for session {session_id} answered "
                    f"{response.status_code} listing what it produced"
                )
            body = await _read_capped(
                response, "output listing", session_id, _LISTING_LIMIT_BYTES
            )
        return _parsed_listing(body, session_id)

    async def fetch_output(
        self, session_id: SessionId, name: str, limit_bytes: int, /
    ) -> bytes | None:
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            raise TurnUndeliverable(
                f"the pod for session {session_id} became {binding.phase.value} part "
                f"way through shipping what it produced, with {name!r} still in it"
            )
        url = shim_output_url_for(session_id, name, binding.pod_name, self._namespace)
        async with (
            httpx.AsyncClient(
                transport=self._transport, timeout=_ROLLOUT_TIMEOUT
            ) as client,
            client.stream("GET", url, headers=self._bearer(session_id)) as response,
        ):
            if response.status_code == _NOTHING_AT_THAT_NAME:
                return None
            if response.status_code != _ACCEPTED:
                raise TurnUndeliverable(
                    f"the shim for session {session_id} answered "
                    f"{response.status_code} for the file {name!r} it produced"
                )
            return await _read_capped(
                response, f"file {name!r}", session_id, limit_bytes
            )

    async def list_workspace(
        self, session_id: SessionId, /
    ) -> tuple[ProducedFile, ...]:
        """The agent's whole working tree, or `()` for a pod that is not running.

        `()` and not a refusal for a pod past RUNNING, matching `list_outputs`: the
        working lane is a convenience the Session can be resumed from, and a pod that
        died before its tree could be read has already cost the Turn whatever the Turn
        cost. Failing the completion on top of that would turn a lost convenience into
        a lost Turn.
        """
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            return ()
        url = shim_workspace_url_for(session_id, binding.pod_name, self._namespace)
        async with (
            httpx.AsyncClient(
                transport=self._transport, timeout=_ROLLOUT_TIMEOUT
            ) as client,
            client.stream("GET", url, headers=self._bearer(session_id)) as response,
        ):
            if response.status_code != _ACCEPTED:
                raise TurnUndeliverable(
                    f"the shim for session {session_id} answered "
                    f"{response.status_code} listing its workspace"
                )
            body = await _read_capped(
                response, "workspace listing", session_id, _LISTING_LIMIT_BYTES
            )
        return _parsed_listing(body, session_id)

    async def fetch_workspace_file(
        self, session_id: SessionId, name: str, limit_bytes: int, /
    ) -> bytes | None:
        """One working file's bytes, or None if it is no longer there.

        A pod that stopped part way answers None rather than raising, which is the
        difference from `fetch_output`. A produced file is a deliverable the tenant was
        promised and losing one silently is the failure that path guards; a working file
        is state the platform keeps so a resume can have it, and a partial working lane
        is strictly better than a failed Turn.
        """
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            return None
        url = shim_workspace_file_url_for(
            session_id, name, binding.pod_name, self._namespace
        )
        async with (
            httpx.AsyncClient(
                transport=self._transport, timeout=_ROLLOUT_TIMEOUT
            ) as client,
            client.stream("GET", url, headers=self._bearer(session_id)) as response,
        ):
            if response.status_code == _NOTHING_AT_THAT_NAME:
                return None
            if response.status_code != _ACCEPTED:
                raise TurnUndeliverable(
                    f"the shim for session {session_id} answered "
                    f"{response.status_code} for the working file {name!r}"
                )
            return await _read_capped(
                response, f"working file {name!r}", session_id, limit_bytes
            )

    def _bearer(self, session_id: SessionId) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {shim_token_for(session_id, self._token_key)}"
        }


def _parsed_listing(body: bytes, session_id: SessionId) -> tuple[ProducedFile, ...]:
    """The pod's listing, or a Turn the control plane closes.

    A listing that will not parse ends the Turn rather than being read as "nothing was
    produced", which is the same call `_one_line` makes about an unreadable event line
    and for the same reason: the pod is the untrusted end here, so "I could not read
    that" does not mean "there was nothing there". Reading it as empty would silently
    drop every document a Turn produced, which is the failure this whole path exists to
    remove -- arriving through the one channel nobody would think to check.

    `ProducedFiles` forbids unknown fields, so a pod padding its listing with anything
    the platform does not read is refused here rather than ignored.
    """
    try:
        listing = ProducedFiles.model_validate_json(body)
    except ValidationError as unreadable:
        raise TurnUndeliverable(
            f"the shim for session {session_id} sent a listing of what it produced "
            "that is not one"
        ) from unreadable
    return tuple(
        ProducedFile(
            name=entry.name,
            byte_length=entry.byte_length,
            content_sha256=entry.content_sha256,
        )
        for entry in listing.files
    )


async def _read_capped(
    response: httpx.Response, what: str, session_id: SessionId, limit: int
) -> bytes:
    """The whole body, or a refusal once it passes the cap.

    Accumulated in chunks rather than by `aread()` so the cap is enforced on the way in.
    `aread()` would have the whole thing in memory before anything could check its size,
    which is the failure the cap exists to prevent.

    The cap is a parameter rather than one constant, because the three bodies read
    through here are bounded by three different facts: a Rollout by what a worker can
    hold, a listing by how many entries the protocol allows, and a produced file by the
    length its own listing reported. `what` names the body in the refusal so a reader of
    the message knows which of the three was too large.
    """
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > limit:
            raise TurnUndeliverable(
                f"the {what} for session {session_id} is larger than {limit} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)
