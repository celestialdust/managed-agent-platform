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

**The scheme is not decided here, and is not decided twice.** A pod either serves mutual
TLS or plain HTTP, and `scheme_for` reads that off the same `SSLContext` the dial
credentials come from -- one object, so a URL saying `https` and a transport carrying no
client certificate is a state that cannot be built. A pod that mounted a partial set of
TLS material refuses to start rather than let this end guess which scheme to speak at
it.

**What proves this wire, and at which tier.** The offline suite dials a shim app in the
same process over an ASGI transport. That grades the protocol, the ordering and every
refusal above, and it says nothing about whether a real pod answers -- so two tiers
cover the rest, and a green offline run is not evidence that either ran. The handshake
is graded against a real socket configured exactly as the pod's uvicorn is, because a
correct SAN, a correct CA and a correctly built context can all hold while the two ends
still fail to agree; those failures exist only when both halves run at once. Above that,
the live tier behind `MAP_CLUSTER_TESTS` places real pods through `KubernetesPodRunner`
and dials them over this module. `pytest -rs` prints the skips that say which did not
run.
"""

from __future__ import annotations

import hmac
import json
import logging
import ssl
from hashlib import sha256
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from managed_agent.control.files.output_shipout import (
    OUTPUT_TREE_LIMIT,
    OutputNotRevisable,
    ProducedFile,
)
from managed_agent.control.session.placement import Placement, PodPhase
from managed_agent.control.session.pods import SessionPods
from managed_agent.control.session.turn_dispatch import (
    TurnOutputNotRevisable,
    TurnUndeliverable,
)
from managed_agent.control.session.turn_execution import RUNTIME_SILENCE_DEADLINE_S
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
    ProducedFiles,
    TurnCompletedLine,
    TurnLine,
    output_path_for,
)
from managed_agent.session_shim.turn_runner import TurnCompleted

_LINE: Final[TypeAdapter[TurnLine]] = TypeAdapter(TurnLine)
MAX_LINES_PER_TURN: Final = 100_000
"""How many lines one Turn may stream before the control plane gives up on it.

Public because a test has to build a stream that crosses it, and a test that hard-coded
a second number would be measuring its own constant. Far above any real Turn: what this
catches is a pod with nothing left to say and no intention of stopping.
"""
_LOG = logging.getLogger(__name__)

_TERMINAL: Final = frozenset({turn.TURN_COMPLETED, turn.TURN_FAILED})
_ACCEPTED: Final = 200

_CONNECT_DEADLINE_S: Final = 10.0
_SHIM_TIMEOUT: Final = httpx.Timeout(
    connect=_CONNECT_DEADLINE_S,
    read=RUNTIME_SILENCE_DEADLINE_S,
    write=_CONNECT_DEADLINE_S,
    pool=_CONNECT_DEADLINE_S,
)
"""The deadlines one Turn's HTTP call is held to, and what each of them bounds.

`read` is httpx's **inter-byte** deadline: it bounds the gap between two bytes
arriving, not the call. It is set to the Turn's total deadline so that it can never be
the bound that fires -- `run_turn` wraps the whole dispatch in that same number, and it
starts counting earlier, at placement. One bound, deliberately.

**Silence on this wire is not evidence that anything is wrong**, and until 2026-08-26
this constant was 120 seconds because it was believed to be. The reasoning that stood
here claimed a clean dichotomy -- a Turn with steady output runs for an hour untouched,
a wedged pod dies in two minutes -- and it was false, because it enumerated only two of
the states a pod can be in. An agent **writing a file** is the third, and it is the
most ordinary thing an agent does: the runtime emits the whole write before anything
reaches this socket, so a large one produces exactly the signature of a dead pod. It
was measured killing tenant Turns at +125 seconds while the same payload, written in
small batches, walked straight through -- nothing about the work got smaller, only the
silence did. A healthy Turn was later observed silent for about seven minutes.

So a gap between bytes cannot be a failure signal here at any value. Raising 120 to a
bigger guess would only move the size of file that dies, and the largest thing an agent
may legitimately write is not a number this platform knows.

**What bounds a pod that is alive but wedged is not this call.** Until 2026-08-26 it
was: an outer `asyncio.timeout` in `run_turn` gave up on the whole dispatch after an
hour, and this `read` value was deliberately the same number so the two could not
disagree. Both are gone. An agent run has no natural length, that hour was killing
healthy long Turns, and what it actually came from was a transport's need for a finite
socket timeout -- which had quietly become the maximum length of an agent run.

What bounds a wedged pod now is `STUCK_IDLE_MS`: the abandoned-Turn sweep reads the
pod's own `turn.progress` report and closes the Turn ten minutes after the runtime
stops speaking to this shim. That is shorter than the hour it replaces and it rests on
evidence from inside the pod rather than on a clock.

**What still does not shorten it, so nobody looks there first:**

- The idle reaper does not, and cannot. Its third guard returns `A_TURN_IS_STILL_OPEN`
  for any Session whose Turn is open (`control/session/reaper.py:380` and `:397`), and
  a wedged Turn is an open Turn -- so `IDLE_GRACE_MS`'s fifteen minutes never start.
- `AbandonedTurnSweeper`'s pod-gone signal does not either. Its two-minute grace is for
  a pod that has *gone*; a wedged pod is present and running.
- Neither does kubelet. The Session pod declares a `startupProbe` and a
  `readinessProbe` and no `livenessProbe`, so a container that starts cleanly and then
  stops making progress is never restarted.

**And one case nothing bounds at all**, stated because a gap named is cheaper than a
gap discovered: a shim that dies with its pod still `Ready` sends no report, and the
sweep refuses to act on the absence of one -- deliberately, because pods on older
digests emit no progress at all and reading their silence would close every Turn they
run. That Turn stays open. It wants a signal that can tell reports *ceasing* from
reports *never starting*, which is buildable and is not built.

It is worth writing down that the reasoning this passage replaces was also confident,
was believed for months, and was wrong -- twice now, in opposite directions. So what is
claimed here is only what has been checked.

**`read` is set rather than left at `None` for a reason that changed.** It used to be
redundant behind `run_turn`'s wrapper and kept as a second line of defence. That
wrapper is gone, so this is now the only thing standing between a half-open socket and
a connection held for the life of the process. It is set long on purpose -- long enough
that no working pod can reach it -- and it bounds a socket, never a Turn. A later
reader tempted to shorten it toward "how long should an agent run" should read this
line as the reason not to: that is the exact coupling that was removed.

The line cap below is not a substitute and does not overlap: it bounds a pod that
streams too much, and this bounds a socket that has stopped delivering anything at all.
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

_LISTING_LIMIT_BYTES: Final = (OUTPUT_TREE_LIMIT + 1) * 1024
"""How large a listing of produced files this control plane will read.

Derived from the enumeration limit rather than chosen, so it cannot come to disagree
with it -- and from that one rather than from what a Turn transfers, because this bounds
the listing and the listing carries the whole tree.
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


def transport_for(
    transport: httpx.AsyncBaseTransport | None, tls: ssl.SSLContext | None
) -> httpx.AsyncBaseTransport | None:
    """The transport a dialler should use, given an injected one and this pod TLS.

    An injected transport wins outright: that is a test driving the routes in-process,
    and it is not speaking TLS to anything. Otherwise a context means a transport built
    to present it, and no context means `None` -- which `httpx.AsyncClient` reads as its
    own default, the plain-HTTP behaviour the platform had before certificates existed.

    One source for the credentials and the scheme, so two settings that had to be kept
    in step cannot disagree. But **called once per client, never once per dialler**, and
    that distinction is load-bearing: `httpx.AsyncClient.aclose` closes the transport it
    was handed, so a dialler that built one transport and gave it to every client had
    whichever Turn finished first tear down the pool the others were streaming through.
    The survivors failed with `ReadError`, which reaches the tenant as `runtime_lost` --
    a Turn that had done its work, lost because another Turn ended.

    It was invisible until the internal CA was rolled. Before it, `tls` was `None`, this
    returned `None`, and every client built its own transport; a control plane holding a
    CA was the first to have anything to share.
    """
    if transport is not None:
        return transport
    if tls is None:
        return None
    return httpx.AsyncHTTPTransport(verify=tls)


def scheme_for(tls: ssl.SSLContext | None) -> str:
    """`https` when this control plane holds pod TLS material, `http` when it does not.

    Derived rather than configured, because the two must not be able to disagree: a
    control plane that dialled `https://` against pods it never gave certificates to
    would fail every dispatch, and one that dialled `http://` against pods that require
    a client certificate would fail every dispatch the other way. The same object that
    decides the scheme is the one that carries the credentials, so there is nothing to
    keep in step (ADR-044).
    """
    return "https" if tls is not None else "http"


def shim_host(pod_name: str, namespace: str) -> str:
    """The pod's DNS name. One construction, read by every route below.

    Public because it has a second reader outside this module: the control plane signs
    each pod's TLS certificate for the name it is going to dial, and a certificate whose
    SAN and whose URL were built by two different string expressions is one that fails
    verification the first time either moves. Sharing the construction is what makes the
    two provably the same name rather than the same name today.
    """
    return f"{pod_name}.{SHIM_SERVICE}.{namespace}.svc.cluster.local"


def shim_url_for(
    pod_name: str, namespace: str, tls: ssl.SSLContext | None = None
) -> str:
    """The one address a Session's shim ever has.

    Resolvable because the pod sets `hostname` to its own name and `subdomain:
    map-session`, and a headless Service of that name selects it -- the three together
    are what give a bare pod a stable DNS record. The hostname is the half that is easy
    to leave out and impossible to notice: without it this name does not resolve at all
    and every Turn fails as undeliverable while the pod reads 2/2 Running, which is what
    shipped until 2026-08-23. `tests/deploy/test_session_pod_runs_a_shim.py` asserts the
    manifest carries all three.
    """
    return (
        f"{scheme_for(tls)}://{shim_host(pod_name, namespace)}:{SHIM_PORT}{TURN_ROUTE}"
    )


def shim_rollout_url_for(
    session_id: SessionId,
    pod_name: str,
    namespace: str,
    tls: ssl.SSLContext | None = None,
) -> str:
    """Where this Session's Rollout is read from, on the pod placement just answered."""
    route = ROLLOUT_ROUTE.format(session_id=session_id)
    return f"{scheme_for(tls)}://{shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_file_url_for(
    session_id: SessionId,
    name: str,
    pod_name: str,
    namespace: str,
    tls: ssl.SSLContext | None = None,
) -> str:
    """Where one of this Session's attached files is placed, on the pod just answered.

    `name` is interpolated raw and that is safe for a reason the caller owns rather
    than this function: every name reaching this point came through
    `parse_upload_filename`, which admits no separator and no dot-dot. The shim
    re-parses it anyway -- a receiver that trusts a path from the network writes
    wherever the sender says, and "the sender is us" is not a property it can check.
    """
    route = FILE_ROUTE.format(session_id=session_id, name=name)
    return f"{scheme_for(tls)}://{shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_outputs_url_for(
    session_id: SessionId,
    pod_name: str,
    namespace: str,
    tls: ssl.SSLContext | None = None,
) -> str:
    """Where this Session's produced files are listed, on the pod just answered."""
    route = OUTPUTS_ROUTE.format(session_id=session_id)
    return f"{scheme_for(tls)}://{shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


def shim_output_url_for(
    session_id: SessionId,
    name: str,
    pod_name: str,
    namespace: str,
    tls: ssl.SSLContext | None = None,
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
    return f"{scheme_for(tls)}://{shim_host(pod_name, namespace)}:{SHIM_PORT}{route}"


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

    **Two completion seams, told apart by how the Turn ended.** `on_completed` runs for
    a Turn that reached its completion line; `on_terminal` runs for one that ended any
    other way it is allowed to end. They are separate parameters rather than one seam
    with an outcome argument because the two answer different questions: what a
    *completed* Turn owes the tenant includes its declared outputs, and a Turn that
    failed declared none -- shipping them would publish a half-written tree under a
    completed Turn's name. So the caller decides which work belongs on which ending,
    here, once, instead of every seam re-deriving it from an outcome flag.

    `on_terminal` defaults to None, and None means the failed ending does no durability
    work at all -- which is what this platform did before the parameter existed.

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
        on_terminal: TurnCompleted | None = None,
        tls: ssl.SSLContext | None = None,
    ) -> None:
        self._placement = placement
        self._pods = pods
        self._log = log
        self._on_completed = on_completed
        self._on_terminal = on_terminal
        self._namespace = namespace
        self._token_key = token_key
        self._tls = tls
        self._injected = transport

    async def dispatch(
        self, session_id: SessionId, turn_id: TurnId, prompt: str
    ) -> None:
        """Carry one Turn into a pod, and give the pod back when that Turn ends.

        The lease is this method's own extent, which is what ADR-041 decided a pod's
        life should be: a Session between Turns owns no pod and occupies no slot.

        Released in a `finally` and not after the call, because the endings that leak
        are the failing ones. A Turn whose model failed, whose pod answered the wrong
        status, or whose shim could not be dialled at all still created a pod somebody
        pays for, and a release reached only by the happy path leaks at the platform's
        failure rate -- which is highest exactly when it is under the most load.

        Released even when nothing was placed. `PodRunner.remove` treats absent as
        success at every step, so the refusals that raise out of `_carry` before a pod
        exists cost one tolerated no-op rather than needing a flag to be read here.

        **This does not cover a control plane that dies mid-Turn.** No `finally` runs
        after that, and the pod outlives the process that leased it. The sweep in
        `control/session/reaper.py` is what collects it, which is why that sweep
        survives this decision even though `IDLE_GRACE_MS` does not.
        """
        try:
            await self._carry(session_id, turn_id, prompt)
        finally:
            await self._release_without_masking_the_turn(session_id, turn_id)

    async def _release_without_masking_the_turn(
        self, session_id: SessionId, turn_id: TurnId
    ) -> None:
        """Give the pod back, and never let that failing become the Turn's answer.

        Called from the `finally` above, where an exception raised here would
        **replace** whatever `_carry` was raising -- Python discards the in-flight
        exception when the `finally` block raises one of its own. So a cluster that
        refused a single DELETE would erase the Turn's real diagnosis: the route in
        `control/api/routes/turns.py` catches two types and a Kubernetes API error is
        neither, so the tenant would get a bare 500, no `turn.failed` would be
        appended, and the Turn would stay open forever -- which pins both
        `archive_session` and the sweep in `control/session/reaper.py`, neither of
        which will act on a Session whose Turn is open. One transient API error would
        cost the Session rather than the Turn.

        So a failed release is logged and dropped. What that costs is a pod left
        running with nobody to collect it, since the sweep that would is the one this
        Turn staying open would have blocked. That is a leaked node slot against a
        Session that keeps working and a Turn that keeps its true outcome, and the
        logged line names the Session so an operator can find the pod.
        """
        try:
            await self._placement.release(session_id)
        except Exception:
            _LOG.exception(
                "the pod for session %s was not released after turn %s and is still "
                "holding a node slot; the Turn's own outcome is unchanged",
                session_id,
                turn_id,
            )

    async def _carry(self, session_id: SessionId, turn_id: TurnId, prompt: str) -> None:
        """Place a pod if this Turn has none, then stream the Turn through it.

        Split from `dispatch` so the lease above reads as one thing rather than as a
        `finally` thirty lines below the `try` it belongs to. Everything here is what
        `dispatch` did before the lease existed.
        """
        binding = await self._placement.locate(session_id)
        waited_ms = 0
        # GONE is a Turn's cue to place, the same as ABSENT, and under this lease it is
        # the commoner of the two. A Session that has taken a Turn before finds the pod
        # that Turn released: deletion is asynchronous, so for its grace period the
        # object is still there, stamped, and `_phase_of` reads it as GONE. The other
        # way to find one is a control plane that died mid-Turn, leaving a pod that
        # finished on its own with no `finally` to collect it.
        #
        # Both used to fall through to the refusal below, and that was right when a
        # Session held one pod for its whole life -- a pod in either state then meant
        # something had gone wrong that a new pod would not fix. It is wrong now: the
        # commonest cause is this platform's own release one Turn ago, and refusing
        # failed every second Turn inside a grace window, which is the ordinary
        # interactive case rather than an edge. Measured on `map-dev` as a 502 whose log
        # line read "the pod for session ... is gone".
        #
        # STARTING still falls through. A pod on its way up belongs to a Turn that is
        # already placing it, and placing a second over it would be two pods for one
        # Session -- which is what the refusal is for and remains for.
        if binding.phase in (PodPhase.ABSENT, PodPhase.GONE):
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
            # `session.resumed` was appended here and is not any more. It meant "this
            # Session is coming back from a suspension", and under ADR-041 every Turn
            # takes this branch -- so the append would fire on every Turn, and the type
            # is webhook-eligible, which makes that a callback per Turn to every
            # endpoint a tenant registered for it, each announcing a cold start that is
            # now simply what a Turn is. `session.placing` above carries the signal a
            # tenant actually needs, and it is appended before the wait rather than
            # after it. The type itself still exists so stored rows keep projecting;
            # nothing writes one.
            #
            # Located again rather than trusting what placement returned: `ensure` can
            # answer for a pod that finished while it waited, and the phase this method
            # acts on has to be the one the cluster reports now.
            binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            # A pod was asked for and the cluster does not report one running: either
            # it is still starting, or it went away between the placement and this
            # line. Not `POD_UNREACHABLE`, which claims something is there and will not
            # answer -- here there is nothing to answer, and the remedy is to submit
            # again rather than to look for a network fault.
            raise TurnUndeliverable(
                f"the pod for session {session_id} is {binding.phase.value}",
                turn.TurnFailureCause.RUNTIME_DID_NOT_START,
            )
        token = shim_token_for(session_id, self._token_key)
        # Flipped once the shim has answered a status line, which is the only thing
        # that separates "nothing is listening" from "it was listening and the stream
        # broke". Both arrive here as `httpx.HTTPError` and they are different tenant
        # problems: the first is a pod that never took the Turn, the second is a Turn
        # that was running and was lost with the process carrying it.
        answered = False
        try:
            async with (
                httpx.AsyncClient(
                    transport=transport_for(self._injected, self._tls),
                    timeout=_SHIM_TIMEOUT,
                ) as client,
                client.stream(
                    "POST",
                    shim_url_for(binding.pod_name, self._namespace, self._tls),
                    json={
                        "session_id": str(session_id),
                        "turn_id": str(turn_id),
                        "prompt": prompt,
                    },
                    headers={"Authorization": f"Bearer {token}"},
                ) as response,
            ):
                if response.status_code != _ACCEPTED:
                    # Something is listening and it declined. Distinct from every other
                    # failure on this path by the one fact that there was an answer at
                    # all, which is what a tenant needs: "nothing is there" and "it said
                    # no" are fixed by different people, and only the first is worth
                    # submitting again.
                    raise TurnUndeliverable(
                        f"the shim for session {session_id} answered "
                        f"{response.status_code}",
                        turn.TurnFailureCause.RUNTIME_REFUSED_THE_TURN,
                    )
                answered = True
                await self._record(session_id, turn_id, response, waited_ms)
        except httpx.HTTPError as unreachable:
            raise TurnUndeliverable(
                f"the shim for session {session_id} "
                + (
                    "stopped answering after the Turn had started"
                    if answered
                    else "could not be reached"
                ),
                turn.TurnFailureCause.RUNTIME_LOST
                if answered
                # `RUNTIME_LOST` is reused rather than duplicated: its published meaning
                # is already "the work was lost with the process carrying it, nothing
                # was delivered, submit it again", which is exactly a stream that broke
                # mid-Turn. A new cause beside it would be a second name for one thing.
                else turn.TurnFailureCause.POD_UNREACHABLE,
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

        **One seam failure is translated rather than collapsed**, added 2026-08-25.
        An artifact the agent rewrote after delivering it is refused by the seal on the
        lane, and that is the tenant's own doing: the platform worked, the pod was
        reachable, and the fix is a different path. Collapsed into `TurnUndeliverable`
        it reached the tenant as 502 "the session could not be reached", whose two
        readings -- retry, and report a platform fault -- are both wrong, and it was
        recorded under `POD_UNREACHABLE`, which this docstring already said was
        inexact and owed a precise cause. It now raises `TurnOutputNotRevisable`, and
        the route answers 409 naming the path.

        Told apart by type and never by message. Matching on the text would make a
        tenant-visible status depend on how an operator worded a sentence written for
        stderr.

        Every other seam failure still becomes `TurnUndeliverable` under
        `POD_UNREACHABLE`, and that cause is still not exact for all of them -- a store
        that refused a write is not an unreachable pod. It stays because the caller's
        move is the same for all of them, and a cause per reason would publish the
        platform's own topology into a tenant-visible field (ADR-013).

        **The `on_terminal` seam is the one exception: it logs and does not raise.**
        Everything above is about a Turn whose tenant-visible outcome is still open, and
        raising is how this method reports that a Turn reading as durable is not. A Turn
        that already streamed `turn.failed` has no such claim left to protect -- its
        failure is appended, the tenant has been told, and the Turn is over. Raising
        there would hand `control/api/routes/turns.py` a `TurnUndeliverable` it answers
        by appending a *second* `turn.failed` for one Turn and telling the tenant 502
        "the session could not be reached" about a Turn that ran and failed for a reason
        the pod already named. Both of those are worse records than the one true thing
        left to say, which goes to the operator's log because there is nobody else it
        could be for.
        """
        completed = False
        terminal = False
        seen = 0
        # `turn.completed` is held back here and appended below, after the seam that
        # ships the Rollout out has returned. Appending it as it arrives -- which is
        # what this loop did until 2026-08-26 -- records the Session's durability
        # boundary (ADR-004) before the thing that makes it durable has happened, and
        # the log is append-only, so a seam that then failed left a claim nothing
        # could walk back. That claim is read: `control/session/pods.py` folds
        # `turn.completed` into `resuming`, and `session_shim/seed_rollout.py` refuses
        # to start a pod told it is resuming with no Rollout to resume from. So one
        # transient store failure permanently bricked the Session -- every later Turn
        # compiled `resuming=True`, found nothing, and refused at pod start.
        #
        # Under the old always-on pod this was nearly unreachable: a second Turn
        # reused the pod it found and never seeded again. ADR-041 made every Turn a
        # placement and so made every Turn run the seed, which is what turned a rare
        # resume-time fault into a permanent one.
        #
        # Holding it inverts the failure into the recoverable direction. If the seam
        # raises, no `turn.completed` is ever written, the route appends `turn.failed`,
        # and the next placement compiles `resuming=False` and opens a fresh thread --
        # a Turn replayed at the tenant's cost, which is the loss ADR-004 names and
        # accepts, against a Session that can never run again.
        held_completion: dict[str, object] | None = None
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
            if line.type == turn.TURN_COMPLETED:
                held_completion = line.payload
                continue
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
            except OutputNotRevisable as revised:
                # **This Turn keeps its completion marker, and it is the one seam
                # failure that does.** The composite runs the Rollout's ship-out first
                # and stops at the first seam that raises -- graded in
                # `tests/control/test_recovery_boundary.py` by the case asserting the
                # outputs seam did not run when the Rollout's refused -- so arriving
                # here proves the Rollout was stored. The durability boundary this
                # marker claims was therefore reached, `resuming` will find bytes to
                # resume from, and withholding it would fail a Turn that worked.
                #
                # What went wrong is the agent rewriting an artifact it had already
                # delivered, which the seal on the lane refuses. That is the tenant's
                # own doing rather than a platform failure, and the route answers 409
                # naming the path. So the Turn carries `turn.completed` and then
                # `turn.failed`, in that order -- both true, about different things.
                await self._record_the_completion(session_id, held_completion)
                raise TurnOutputNotRevisable(revised.path) from revised
            except (TurnUndeliverable, httpx.HTTPError):
                raise
            except Exception as not_made_durable:
                raise TurnUndeliverable(
                    f"the rollout for session {session_id} could not be stored after "
                    f"turn {turn_id}"
                ) from not_made_durable
        elif self._on_terminal is not None:
            try:
                await self._on_terminal.turn_completed(session_id, turn_id)
            except Exception:
                _LOG.exception(
                    "the rollout for session %s could not be stored after failed "
                    "turn %s; that Turn's conversation stays only in a pod about to "
                    "be allowed to die",
                    session_id,
                    turn_id,
                )
        if not completed and held_completion is not None:
            # A stream that wrote `turn.completed` as an event and never sent the
            # terminal line contradicts itself: `StreamedCompletion` in
            # `session_shim/serve.py` is notified after a Turn's last event is
            # recorded and only for a Turn that reached a completion, so in any
            # well-formed stream these two arrive together. Appending the event
            # without the line would record a durability boundary whose ship-out was
            # never run -- exactly the state the hold above exists to prevent -- so
            # this refuses instead, and the route closes the Turn as failed.
            raise TurnUndeliverable(
                f"the shim for session {session_id} wrote turn.completed for turn "
                f"{turn_id} without ending the stream as completed"
            )
        # Last, and only now that the seam above has returned. Every other event this
        # Turn produced is already appended, so appending here keeps the log's order --
        # `turn.completed` is the pod's final event and nothing follows it. The comment
        # at the top of the loop says why it waited.
        await self._record_the_completion(session_id, held_completion)

    async def _record_the_completion(
        self, session_id: SessionId, held: dict[str, object] | None
    ) -> None:
        """Append the completion event the stream loop held back, if there is one.

        A method rather than two copies of one `append`, because it is reached from two
        places that must not drift: the ordinary path below the seam, and the one seam
        failure that still leaves a Turn completed. `None` means the pod signalled a
        completion without ever writing the event, which nothing here invents.
        """
        if held is not None:
            await append_in_order(self._log, session_id, turn.TURN_COMPLETED, held)


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
        tls: ssl.SSLContext | None = None,
    ) -> None:
        self._placement = placement
        self._namespace = namespace
        self._token_key = token_key
        self._tls = tls
        self._injected = transport

    async def place_file(
        self, session_id: SessionId, name: str, body: bytes, /
    ) -> None:
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            raise TurnUndeliverable(
                f"session {session_id} has no running pod to place {name!r} into, so "
                "its attached file would be missing from a Session that starts anyway"
            )
        url = shim_file_url_for(
            session_id, name, binding.pod_name, self._namespace, self._tls
        )
        token = shim_token_for(session_id, self._token_key)
        async with httpx.AsyncClient(
            transport=transport_for(self._injected, self._tls), timeout=_FILE_TIMEOUT
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
        tls: ssl.SSLContext | None = None,
    ) -> None:
        self._placement = placement
        self._namespace = namespace
        self._token_key = token_key
        self._tls = tls
        self._injected = transport

    async def fetch_rollout(self, session_id: SessionId, /) -> bytes | None:
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            return None
        url = shim_rollout_url_for(
            session_id, binding.pod_name, self._namespace, self._tls
        )
        token = shim_token_for(session_id, self._token_key)
        async with (
            httpx.AsyncClient(
                transport=transport_for(self._injected, self._tls),
                timeout=_ROLLOUT_TIMEOUT,
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
        tls: ssl.SSLContext | None = None,
    ) -> None:
        self._placement = placement
        self._namespace = namespace
        self._token_key = token_key
        self._tls = tls
        self._injected = transport

    async def list_outputs(self, session_id: SessionId, /) -> tuple[ProducedFile, ...]:
        binding = await self._placement.locate(session_id)
        if binding.phase is not PodPhase.RUNNING:
            return ()
        url = shim_outputs_url_for(
            session_id, binding.pod_name, self._namespace, self._tls
        )
        async with (
            httpx.AsyncClient(
                transport=transport_for(self._injected, self._tls),
                timeout=_ROLLOUT_TIMEOUT,
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
        url = shim_output_url_for(
            session_id, name, binding.pod_name, self._namespace, self._tls
        )
        async with (
            httpx.AsyncClient(
                transport=transport_for(self._injected, self._tls),
                timeout=_ROLLOUT_TIMEOUT,
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
