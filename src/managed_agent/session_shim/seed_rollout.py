"""Put a Session's Rollout back under `CODEX_HOME` before the runtime starts.

Runs as the second half of the pod's one init container, once per pod placement: the
shell half ahead of it in the same `/bin/sh -c` body creates `CODEX_HOME` and copies
`config.toml` into it, and `exec` hands this process the container. So this runs after
that root exists and before any regular container. By the time the shim opens the
Session, a resuming Session's runtime home holds the record its previous pod was
writing, and a Session placed for the first time has none -- which is what the shim
reads to decide between continuing a thread and opening one.

The two halves were two init containers until they were merged to take one serial
container start off every placement. One container has one termination message, so each
half names itself on the stream that becomes it: every line this module writes is
prefixed `seed-rollout:` and every line the shell half writes is prefixed
`seed-runtime-home:`. The prefixes are what a pod status now uses to say which of the
two jobs refused, and `deploy/k8s/session-pod.yaml` is where that trade is recorded.

**The workspace is not this module's business.** A Session's working tree lives on a
mounted volume that outlives the pod, so there is nothing to put back and no boundary to
cross (ADR-035). This restores the one thing a mount cannot: the runtime's own
conversation record, which is a database rather than ordinary files and stays pod-local
for that reason. The merge did put the workspace within this process's reach -- the
shell half needs it and the mount is the container's, not a half's -- so "not its
business" is now a property of this code rather than of what it was given. Nothing here
opens a path under the workspace root, and nothing here should start.

**Why a fresh thread is the failure and not the fallback.** The Rollout carries the
runtime's compaction checkpoints, which have already folded the Session's history into
a form no other record here reproduces. A pod that starts a new thread for a Session
that has one replays what was folded, charges the tenant for the replay, and reports
success -- a wrong answer that costs money and announces nothing (ADR-004). So a pod
told it is resuming and handed nothing to resume from **refuses to start**. Refusing
loses a placement and says why in the pod's own status; continuing loses the property
the whole recovery boundary exists to hold.

**It asks the Tool Gateway, over the one arrow this pod already has**, rather than
reaching for storage itself: this pod's egress is kube-dns and the two gateways, this
VPC has no S3 endpoint, and the `x-map-session` token is already in the compiled
document the `compiled` volume carries. Nothing is minted here, no volume is added, no
arrow is opened. Which tenant and which Session is the token's to say.

**Where the file goes is a contract with two readers, and both are why it is not just
any path.** The runtime, handed this path, keeps appending to this same file for the
rest of the pod's life -- so ship-out at the next completed Turn has to find it, and
ship-out finds a Rollout by globbing `sessions/*/*/*/rollout-*-<thread_id>*.jsonl`
under the runtime's home. A file written anywhere else would resume correctly, run a
Turn, and then ship out nothing: the Session would come back a second time from the
Rollout it had before this pod ran, silently losing a Turn. So this writes the runtime's
own layout, and `seeded_path` is the single place that spelling lives --
`find_seeded` beside it is the read of the same layout, so the shim that resumes
from this file and the container that writes it cannot disagree about where it is.

**The thread id comes out of the bytes, never from anywhere else.** A Rollout's first
line is its `session_meta`, whose `id` is the thread the record belongs to -- the same
field the runtime itself verifies a resolved path against. Deriving it from the file
means a Rollout and the id it is resumed under cannot come from different Sessions.

Every line this writes goes to stderr and none of them carries the token -- including
the lines written on the way out of a failure, which is where a token most easily
escapes. The container declares `terminationMessagePolicy: FallbackToLogsOnError`, so
on a refusal these lines become the pod's termination message, read by everyone who can
read the pod rather than only by whoever runs `kubectl logs`.

Provenance for the shape of all of this: ADR-031, and ADR-030 for the transport it
inherits.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import httpx

from managed_agent.control.pod_config.compiler import CODEX_HOME, GATEWAY_SERVER_ID
from managed_agent.core.session.session_token import SESSION_TOKEN_HEADER_NAME

COMPILED_CONFIG: Final = Path("/etc/map/compiled/config.toml")
"""Where the `compiled` secret volume is mounted, plus the one file it carries.

Both halves are `deploy/k8s/session-pod.yaml`'s -- the mount path and the Secret key --
and the init container beside this one copies the same file out of the same place.
Spelled here rather than passed in: a wrong value is a container that exits saying "no
such file", which is the least dangerous way for this to be wrong.
"""

TERMINATION_LOG: Final = Path("/dev/termination-log")
"""Where kubelet reads a container's own last word from, into the pod's status.

Written on success only, so that what was seeded is legible to a reader of the pod
rather than only to a reader of the container log. On a refusal it is deliberately left
empty, because `terminationMessagePolicy: FallbackToLogsOnError` then promotes the log
instead -- and the log is where the reason is.
"""

_REQUEST_TIMEOUT_SECONDS: Final = 30.0
"""Per request, not for the seed as a whole.

The whole is bounded by the pod's readiness budget, which is the bound that actually
ends the attempt: an init container that overruns it gets the pod deleted by `ensure`'s
own cleanup. This one is here so that a single hung connection fails its own request
rather than hanging the placement until that outer bound.
"""

Emit = Callable[[str], None]
"""Where this process says what it did. One line, already free of the token."""


class RestoreRefused(Exception):
    """The Rollout could not be put back, so this pod must not start.

    One type for every reason -- an unreadable configuration, a body over the budget, a
    Rollout whose first line names no thread, a Gateway that will not answer -- because
    the caller does the same thing with all of them: print the reason and exit non-zero.
    The difference between them belongs in the message, which is what a reader actually
    gets.

    Refusing is the whole point rather than a failure mode. A pod told it is resuming
    and handed nothing to resume from must not quietly open a fresh thread; see this
    module's docstring for what that costs.
    """


@dataclass(frozen=True, slots=True)
class GatewayBinding:
    """Where to ask, and what to present. Both read out of one compiled document.

    Read from one `mcp_servers` entry rather than from two sources, so a URL and a token
    cannot come from different Sessions or different deployments.
    """

    base_url: str
    token: str


def read_binding(config_toml: str) -> GatewayBinding:
    """The Gateway's base URL and this Session's token, out of the compiled document.

    Read back out of the rendered TOML rather than from an environment variable, for the
    reason the pod holds no such variable: the token rides in this document so that the
    pod specification needs no new Secret and no field that anyone able to read the pod
    can read.

    The base is the server's URL with its path removed, because the compiled value
    points at the MCP endpoint (`.../mcp`) while the seed route is a sibling of it
    rather than a child. A scheme this cannot dial, or an empty host, is refused here
    rather than left to surface as an httpx error naming a URL and no cause.
    """
    document = tomllib.loads(config_toml)
    servers = document.get("mcp_servers")
    if not isinstance(servers, dict):
        raise RestoreRefused(
            "the compiled configuration declares no mcp_servers table, so it names "
            "neither the Tool Gateway nor this Session's token"
        )
    server = servers.get(GATEWAY_SERVER_ID)
    if not isinstance(server, dict):
        raise RestoreRefused(
            f"the compiled configuration declares no {GATEWAY_SERVER_ID!r} server, so "
            "there is nothing here to ask for this Session's Rollout"
        )
    url = server.get("url")
    if not isinstance(url, str):
        raise RestoreRefused(
            f"the {GATEWAY_SERVER_ID!r} server names no url to reach it at"
        )
    headers = server.get("http_headers")
    token = (
        headers.get(SESSION_TOKEN_HEADER_NAME) if isinstance(headers, dict) else None
    )
    if not isinstance(token, str) or not token:
        raise RestoreRefused(
            f"the {GATEWAY_SERVER_ID!r} server carries no {SESSION_TOKEN_HEADER_NAME} "
            "header, so nothing here can prove which Session is asking"
        )
    split = urlsplit(url)
    if split.scheme not in ("http", "https") or not split.netloc:
        raise RestoreRefused(
            f"the {GATEWAY_SERVER_ID!r} server's url is not one this can dial: {url!r}"
        )
    # Userinfo is refused rather than stripped, and the reason is downstream: the base
    # composed below is quoted into the refusal an unreachable Gateway raises, and that
    # refusal reaches the pod's own status. A `user:pass@` left in the authority would
    # be a credential printed where every reader of the pod can see it. Nothing this
    # platform writes carries one -- the value comes from MAP_TOOL_GATEWAY_URL, an
    # in-cluster Service address -- so a url that does carry one did not come from here,
    # and refusing is both safer and more honest than silently editing a document the
    # pod was started with.
    if split.username is not None or split.password is not None:
        raise RestoreRefused(
            f"the {GATEWAY_SERVER_ID!r} server's url embeds credentials in its "
            "authority, which this platform does not write and which this process "
            "would echo into the pod's status when naming what it could not reach"
        )
    return GatewayBinding(
        base_url=urlunsplit((split.scheme, split.netloc, "", "", "")),
        token=token,
    )


def client_for(binding: GatewayBinding) -> httpx.AsyncClient:
    """The client every request in this process goes through, and the three decisions
    that go into it: where it points, what it presents, and how long it waits.

    The token is a DEFAULT HEADER rather than something a call site composes. A header
    is not part of a URL, so no route string, no log line and no httpx exception repr
    can carry the value -- which matters here because the container's message policy
    promotes this process's output into the pod's status on a refusal.

    Its own function so the three decisions are readable off the returned object. A
    wrong header name or a base URL that kept the MCP path is a 401 or a 404 on every
    request from every pod, answered identically to a token nobody sent.
    """
    return httpx.AsyncClient(
        base_url=binding.base_url,
        headers={SESSION_TOKEN_HEADER_NAME: binding.token},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


SEED_ROUTE: Final = "/v1/session/rollout"
"""The Tool Gateway's seed route: the whole of this Session's Rollout, or 204.

It takes no tenant, no Session and no path -- all three are read off the token, which
is what keeps this process unable to ask for a conversation that is not its own.
"""

RESUMING_ENV: Final = "MAP_RESUMING"
"""Whether this pod continues a thread, substituted per Session onto this container.

Read here and nowhere else in the pod. The shim decides between resuming and starting
from whether a file was seeded, so this value has exactly one job: to say whether an
empty answer from the Gateway is the ordinary first placement or a Session whose record
has gone missing.
"""

_SEED_BUDGET_BYTES: Final = 64 * 1024 * 1024
"""The largest Rollout this will write, against the volume it writes into.

Priced against `codex-home` rather than against a measurement of a Rollout, because
what fails is the volume and not the read: that `emptyDir` is capped at 256Mi in
`deploy/k8s/session-pod.yaml` and holds the runtime's own growing sqlite state beside
this file -- measured there at about 5Mi after three Turns. A quarter of the volume
leaves the rest for the state and for the Turns this Session has yet to run, since the
runtime keeps APPENDING to the file this writes: the seeded length is a floor on what
it will occupy, never the total.

Refusing is the right answer at this size rather than writing and hoping. A pod that
overruns the volume is evicted mid-Turn, which loses the Turn AND leaves the eviction
looking like a node problem; a pod that never starts says which Session and which
number in its own status.

Checked against the declared answer before a byte is read, so a Rollout that will be
refused is not first downloaded.
"""

_META = "session_meta"

_THREAD_ID: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")
"""The shape a thread id must have before anything composes a path out of it.

This is a filename grammar and not a guess at the runtime's own: `seeded_path` puts this
value INTO a filename, and `pathlib` splits an embedded `/` into components without
normalising a `..` among them, so an id is a path fragment whether or not it was meant
as one. The alphabet is what every id the runtime writes already satisfies -- Codex
thread ids are UUIDs -- and `turn_complete.find_rollout` globs `rollout-*-{thread_id}*`
over the same names, which a separator breaks too.
"""


def is_resuming(environ: object) -> bool:
    """Whether this placement continues a thread, read from the pod's environment.

    Parsed strictly and with no default: the two spellings this accepts are the two
    `pod_runner._SEED_ENV` emits, and anything else -- including absence -- refuses.
    A tolerant reader is the wrong shape here, because every value it would have to
    guess at maps onto "not resuming", which is exactly the answer that turns a missing
    variable into a silently fresh thread for a Session that had a conversation.
    """
    if not isinstance(environ, dict):
        raise RestoreRefused("this process was given no environment to read")
    said = environ.get(RESUMING_ENV)
    if said == "true":
        return True
    if said == "false":
        return False
    raise RestoreRefused(
        f"{RESUMING_ENV} is {said!r}, which is neither 'true' nor 'false', so this pod "
        "cannot tell a Session continuing a conversation from one starting its first"
    )


def thread_id_in(body: bytes) -> str:
    """The thread this Rollout belongs to, off its own first line.

    A Rollout's first line is always its `session_meta`, and a record whose first line
    is something else is one the runtime's own reader treats as a hard error -- so a
    body that does not open with one is refused here, where the message can name the
    Session, rather than inside a runtime whose failure reaches nobody.

    The id is required to be a non-empty string of `_THREAD_ID`'s alphabet, and the
    second half of that is load-bearing rather than fussy. This value is not merely
    carried: `seeded_path` interpolates it into a filename, `pathlib` splits an embedded
    `/` into path components and normalises no `..` among them, and `write_seed` then
    creates the parents and writes. An id of `a/../../..` composes a write outside
    `CODEX_HOME` entirely -- reaching, since the two init containers were merged, the
    mounted workspace that outlives the pod. It arrives off the first line of a body the
    Gateway returned, so it is not a value this pod controls. Refusing the shape here is
    a parse and not a second opinion: every id the runtime writes already satisfies it,
    so a body that fails this is not a Rollout worth writing anywhere.
    """
    first = next((line for line in body.split(b"\n") if line.strip()), None)
    if first is None:
        raise RestoreRefused(
            "the stored Rollout holds no lines, so there is no conversation in it to "
            "continue and a pod started from it would resume from nothing"
        )
    try:
        parsed = json.loads(first)
    except (json.JSONDecodeError, UnicodeDecodeError) as unreadable:
        raise RestoreRefused(
            "the stored Rollout's first line does not parse as JSON, so this is not a "
            f"record the runtime can resume from ({type(unreadable).__name__})"
        ) from unreadable
    if not isinstance(parsed, dict) or parsed.get("type") != _META:
        raise RestoreRefused(
            f"the stored Rollout does not open with its {_META} line, which is the one "
            "shape the runtime's own reader refuses outright"
        )
    payload = parsed.get("payload")
    found = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(found, str) or not found:
        raise RestoreRefused(
            f"the stored Rollout's {_META} line names no thread id, so nothing here "
            "can say which conversation these bytes are"
        )
    if not _THREAD_ID.fullmatch(found):
        raise RestoreRefused(
            f"the stored Rollout's {_META} line names a thread id that is not a bare "
            "identifier, and this value becomes part of the path these bytes would be "
            "written to -- so it is refused here rather than resolved by the kernel"
        )
    return found


def seeded_path(root: Path, thread_id: str, at: datetime) -> Path:
    """Where the seeded Rollout lands: the runtime's own layout, spelled once.

    `sessions/<YYYY>/<MM>/<DD>/rollout-<YYYY-MM-DD>T<HH-MM-SS>-<thread_id>.jsonl`, which
    is both what the runtime's filename parser accepts and what the shim's ship-out glob
    finds. Neither is optional: a name the parser rejects fails the resume, and a name
    outside that glob resumes fine and then ships nothing at the next completed Turn --
    losing a Turn silently, which is the worse of the two.

    The timestamp is this container's clock rather than the record's own creation time.
    It is free to differ: nothing reads it back, the glob wildcards it, and taking it
    from the stored bytes would mean re-deriving a timestamp format from a field this
    process has no other reason to parse -- a way to produce an unparseable name out of
    a perfectly good Rollout.

    UTC explicitly. A naive local clock would put the directory on the node's timezone,
    so two pods of one Session could seed under two dates with nothing saying why.
    """
    stamp = at.strftime("%Y-%m-%dT%H-%M-%S")
    return (
        root
        / "sessions"
        / at.strftime("%Y")
        / at.strftime("%m")
        / at.strftime("%d")
        / f"rollout-{stamp}-{thread_id}.jsonl"
    )


def find_seeded(root: Path) -> Path | None:
    """The Rollout this pod was seeded with, or None when it was not seeded.

    The read half of `seeded_path`, in the same file so the two cannot drift: the shim
    decides between continuing a thread and opening one from what this answers, and a
    glob that stopped matching what the seed writes would put every resuming Session on
    a fresh thread -- silently, which is the whole hazard.

    **A file found here is unambiguous, and that rests on when it runs.** The runtime
    writes no Rollout until a thread is started, and nothing starts one before the shim
    does, so at shim start-up the only record that can exist under this root is the one
    the init container put there. Two would mean something else wrote one, which is a
    state this cannot resolve and does not guess at: which of two conversations a
    Session continues is not a coin to flip.

    Matched by the runtime's own filename shape rather than by any suffix, so a stray
    file under `sessions/` -- a lock, an editor's leavings -- is not mistaken for a
    conversation.
    """
    found = sorted(root.glob("sessions/*/*/*/rollout-*.jsonl"))
    if not found:
        return None
    if len(found) > 1:
        raise RestoreRefused(
            f"{len(found)} Rollouts are under {root / 'sessions'} before this Session "
            "has opened a thread, so which conversation to continue is ambiguous: "
            f"{[path.name for path in found]}"
        )
    return found[0]


def thread_id_at(path: Path) -> str:
    """The thread a seeded Rollout belongs to, off the first line of the file.

    The first line alone, because a Rollout is append-only and its `session_meta` is
    always the first record -- and because the file may be tens of megabytes, which is
    not a thing to read into memory to learn one identifier.
    """
    with path.open("rb") as record:
        first = record.readline()
    return thread_id_in(first)


async def fetch(client: httpx.AsyncClient) -> bytes | None:
    """The stored Rollout, `None` when the Session has none, or a refusal.

    Streamed rather than read whole, so the declared length can be refused BEFORE the
    body is downloaded: a Rollout over the budget should cost one round trip and not
    the whole transfer. A response that declares no length is refused for the same
    reason -- an undeclared body is one whose size is only known once it has already
    been read into this process.
    """
    try:
        async with client.stream("GET", SEED_ROUTE) as answer:
            if answer.status_code == httpx.codes.NO_CONTENT:
                return None
            if answer.status_code != httpx.codes.OK:
                raise RestoreRefused(
                    f"the Tool Gateway answered {answer.status_code} for this "
                    "Session's Rollout, so whether it has one is not known and a pod "
                    "started now would open a fresh thread over a record it should "
                    "have continued"
                )
            _refuse_an_oversized_body(answer.headers.get("content-length"))
            return await answer.aread()
    except httpx.HTTPError as unreachable:
        raise RestoreRefused(
            "the Tool Gateway did not answer for this Session's Rollout: "
            f"{type(unreachable).__name__}"
        ) from unreachable


def _refuse_an_oversized_body(declared: str | None) -> None:
    """Refuse a Rollout too large for the volume, before any of it is read."""
    if declared is None or not declared.isdigit():
        raise RestoreRefused(
            "the Tool Gateway declared no readable length for this Session's Rollout, "
            f"so it cannot be held against the {_SEED_BUDGET_BYTES}-byte budget "
            "without first reading a body of unknown size"
        )
    if int(declared) > _SEED_BUDGET_BYTES:
        raise RestoreRefused(
            f"this Session's Rollout is {declared} bytes, over the "
            f"{_SEED_BUDGET_BYTES} this pod's runtime home can hold beside the state "
            "the runtime writes beside it"
        )


def write_seed(root: Path, body: bytes, at: datetime, /, *, report: Emit) -> Path:
    """Write the Rollout where the runtime and the ship-out glob will both find it."""
    destination = seeded_path(root, thread_id_in(body), at)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    report(f"seeded {len(body)} byte(s) of this Session's Rollout into {destination}")
    return destination


async def seed(
    client: httpx.AsyncClient, root: Path, resuming: bool, /, *, report: Emit
) -> Path | None:
    """Put this Session's Rollout under `root`, or say why the pod must not start.

    Asks only when this is a resume, which is what keeps a first placement to zero
    round trips and keeps a stray record out of a runtime home that should be empty.

    The two ways this returns are both announced, because they are the two the shim
    then acts on and they are indistinguishable from outside the pod: a Session that
    opened a thread and a Session that continued one look identical in every status
    Kubernetes reports.
    """
    if not resuming:
        report(
            "this Session has completed no Turn, so nothing is seeded and the shim "
            "will open a new thread. This is a first placement."
        )
        return None
    body = await fetch(client)
    if body is None:
        raise RestoreRefused(
            "this Session has completed a Turn and the Tool Gateway holds no Rollout "
            "for it. Starting anyway would open a NEW thread over a conversation that "
            "already exists, replaying history its compaction checkpoints have folded "
            "and charging the tenant for the replay, so this pod does not start"
        )
    return write_seed(root, body, datetime.now(UTC), report=report)


async def run(
    config_toml: str, root: Path, resuming: bool, /, *, report: Emit
) -> Path | None:
    """Read this pod's own configuration and seed through the client it describes.

    The lane restore's client, unchanged and not re-decided: where to ask, what to
    present and how long to wait are one set of answers for both fetches out of this
    pod, and its 30 s is a bound on a stalled read rather than on a whole transfer --
    httpx applies it per chunk received, so a large body streams under it.
    """
    async with client_for(read_binding(config_toml)) as client:
        return await seed(client, root, resuming, report=report)


def _to_stderr(line: str) -> None:
    """One line, on the stream kubelet collects, flushed as it is written.

    stderr rather than a `logging` handler for the reason the lane restore gives: a
    level is one more thing that can be configured into silence, and this process's
    whole account of itself is these lines. Flushed because the process may be about to
    exit non-zero, and a buffered last word is no word at all.
    """
    print(f"seed-rollout: {line}", file=sys.stderr, flush=True)


def main() -> int:
    """Seed the Rollout, or refuse and take the pod down with the refusal.

    Returns the status rather than calling `sys.exit`, so the whole of it is reachable
    from a test.

    An unexpected exception is caught and restated by type and message for one reason:
    a traceback is not a sentence, and under `FallbackToLogsOnError` a traceback is what
    a reader of the pod's status would otherwise get. Restated, never swallowed -- the
    status is non-zero on every path but the last.
    """
    root = Path(CODEX_HOME)
    try:
        resuming = is_resuming(dict(os.environ))
        config_toml = COMPILED_CONFIG.read_text(encoding="utf-8")
    except RestoreRefused as unreadable:
        _to_stderr(f"refusing to start this pod: {unreadable}")
        return 1
    except OSError as unreadable:
        _to_stderr(
            f"the compiled configuration at {COMPILED_CONFIG} could not be read "
            f"({type(unreadable).__name__}), so this pod cannot ask for its Rollout"
        )
        return 1
    try:
        seeded = asyncio.run(run(config_toml, root, resuming, report=_to_stderr))
    except RestoreRefused as refused:
        _to_stderr(f"refusing to start this pod: {refused}")
        return 1
    except Exception as unexpected:  # noqa: BLE001 - see the docstring
        _to_stderr(
            f"refusing to start this pod: {type(unexpected).__name__}: {unexpected}"
        )
        return 1
    _record(seeded)
    return 0


def _record(seeded: Path | None) -> None:
    """Put what was seeded where a reader of the pod's status can see it.

    Best-effort on purpose, and for the reason the lane restore gives: the termination
    log is a file kubelet bind-mounts in, every container here runs on a read-only root,
    and failing a pod over a status line after the seed has already succeeded would be
    the wrong trade. A failure to write it is said out loud rather than swallowed, so
    the absence of the line is not read as a seed that did not run.
    """
    summary = (
        f"seeded this Session's Rollout into {seeded}"
        if seeded is not None
        else "first placement: no Rollout to seed"
    )
    try:
        TERMINATION_LOG.write_text(summary, encoding="utf-8")
    except OSError as unwritable:
        _to_stderr(
            f"{summary}, but {TERMINATION_LOG} could not be written "
            f"({type(unwritable).__name__}), so it is only in this log"
        )


if __name__ == "__main__":
    raise SystemExit(main())
