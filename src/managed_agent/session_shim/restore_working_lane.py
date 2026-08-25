"""Put a Session's `working` lane back under its workspace before the runtime starts.

Runs as the `restore-working-lane` init container, once per pod placement: after
`seed-runtime-home` has made the workspace root and the two dot-directories bwrap
refuses to build a sandbox without, and before any regular container. By the time the
agent runs, `/session/workspace` holds the tree the Session's previous pod was working
in, and a Session placed for the first time finds it empty.

**It asks the Tool Gateway, over the one arrow this pod already has.** A Session pod's
egress is kube-dns and the two gateways, and this VPC has no S3 endpoint, so a presigned
URL handed to the pod is a credential that does not work. The Gateway already holds the
bucket grant and already verifies the `x-map-session` token -- and that token is already
in this pod, in the compiled `config.toml` the `compiled` volume carries, beside the
Gateway's own URL. So nothing is minted here, no volume is added, and no arrow is
opened. Which tenant and which Session is the token's to say: this process names neither
and cannot ask for another Session's bytes.

**All or nothing.** Every path the listing names is fetched, at the length it was listed
at, or this exits non-zero and the pod never starts. A partial tree presents itself as a
complete one -- the agent reports a file missing and nothing anywhere says why. Bytes
already written on the way to a refusal are left where they are rather than swept,
because they do not survive: the workspace is an `emptyDir` that goes away with the pod
that never started.

**The ceilings are the sync's own, so a lane that synced in full restores in full.**
Imported from the module that writes the lane rather than restated, and checked against
the listing before a single object is fetched -- a restore that is going to refuse on
object 2049 should not first have spent the 2048 under it.

**The fetch is concurrent because the object count is what binds, not the bytes.** 2048
serialized GETs at ~20 ms is ~41 s, against a readiness budget already spent on the
image pull and two probes; the same 256 MiB in one stream is seconds.
`adapters/kubernetes/pod_runner._READY_TIMEOUT_SECONDS` carries the other half of that
arithmetic and says so.

Every line this writes goes to stderr, and none of them carries the token. That is not
tidiness: the container declares `terminationMessagePolicy: FallbackToLogsOnError`, so
on a refusal these lines become the pod's termination message, which is read by everyone
who can read the pod rather than only by whoever runs `kubectl logs`.

Provenance for the shape of all of this: ADR-030.
"""

from __future__ import annotations

import asyncio
import sys
import tomllib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import httpx

from managed_agent.control.files.workspace_sync import (
    WORKING_BUDGET_BYTES,
    WORKING_COUNT_LIMIT,
)
from managed_agent.control.pod_config.compiler import GATEWAY_SERVER_ID, WORKSPACE_ROOT
from managed_agent.core.session.session_token import SESSION_TOKEN_HEADER_NAME
from managed_agent.core.vfs.session_vfs import VfsPathInvalid, parse_relative_path

COMPILED_CONFIG: Final = Path("/etc/map/compiled/config.toml")
"""Where the `compiled` secret volume is mounted, plus the one file it carries.

Both halves are `deploy/k8s/session-pod.yaml`'s -- the mount path and the Secret key --
and the init container beside this one copies the same file out of the same place.
Spelled here rather than passed in: a wrong value is a container that exits saying "no
such file", which is the least dangerous way for this to be wrong.
"""

LANE_ROUTE: Final = "/v1/session/working-lane"
"""The Tool Gateway's two working-lane routes, listing and object, sharing a prefix.

`GET <prefix>` answers `{"objects": [{"path": ..., "size": ...}, ...]}` and
`GET <prefix>/<path>` answers the object body, or 404 when the lane does not hold it.
Neither takes a tenant or a Session: both read them off the token, which is what keeps
this process unable to ask for bytes that are not its own.
"""

TERMINATION_LOG: Final = Path("/dev/termination-log")
"""Where kubelet reads a container's own last word from, into the pod's status.

Written on success only, so that what was restored is legible to a reader of the pod
rather than only to a reader of the container log. On a refusal it is deliberately left
empty, because `terminationMessagePolicy: FallbackToLogsOnError` then promotes the log
instead -- and the log is where the reason is.
"""

_CONCURRENT_FETCHES: Final = 16
"""How many object GETs are in flight at once.

Sized against the ceiling rather than against a measurement: 2048 objects sixteen at a
time is 128 round trips, so a ~20 ms round trip costs ~2.6 s where the serial form costs
~41 s. Higher buys little, because past this the transfer is bandwidth-bound rather than
latency-bound, and it costs the Gateway a larger burst from every pod placed at once.
"""

_REQUEST_TIMEOUT_SECONDS: Final = 30.0
"""Per request, not for the restore as a whole.

The whole is bounded by the pod's readiness budget, which is the bound that actually
ends the attempt: an init container that overruns it gets the pod deleted by `ensure`'s
own cleanup. This one is here so that a single hung connection fails its own request
rather than holding one of the concurrent slots above until that outer bound.
"""

Emit = Callable[[str], None]
"""Where this process says what it did. One line, already free of the token."""


class RestoreRefused(Exception):
    """The lane could not be put back in full, so this pod must not start.

    One type for every reason -- an unreadable configuration, a listing over the
    ceiling, a path that is not lane-relative, an object the Gateway would not serve, a
    body shorter than its listing -- because the caller does the same thing with all of
    them: print the reason and exit non-zero. The difference between them belongs in the
    message, which is what a reader actually gets.
    """


@dataclass(frozen=True, slots=True)
class GatewayBinding:
    """Where to ask, and what to present. Both read out of one compiled document.

    Read from one `mcp_servers` entry rather than from two sources, so a URL and a token
    cannot come from different Sessions or different deployments.
    """

    base_url: str
    token: str


@dataclass(frozen=True, slots=True)
class LaneObject:
    """One entry of the listing: its path within the lane, and how long it should be.

    `byte_length` is what the fetched body is compared against. A body of another length
    is a refusal rather than a file, because a short one is the truncation this module
    exists to make impossible.
    """

    relative: str
    byte_length: int


@dataclass(frozen=True, slots=True)
class RestoreReport:
    """What was put back. There is no field for what was not: a shortfall raises."""

    objects_restored: int
    bytes_restored: int


def read_binding(config_toml: str) -> GatewayBinding:
    """The Gateway's base URL and this Session's token, out of the compiled document.

    Read back out of the rendered TOML rather than from an environment variable, for the
    reason the pod holds no such variable: the token rides in this document so that the
    pod specification needs no new Secret and no field that anyone able to read the pod
    can read.

    The base is the server's URL with its path removed, because the compiled value
    points at the MCP endpoint (`.../mcp`) while the lane routes are siblings of it
    rather than children. A scheme this cannot dial, or an empty host, is refused here
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
            "there is nothing here to ask for this Session's working lane"
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


def parse_listing(payload: object) -> tuple[LaneObject, ...]:
    """The listing, once every entry in it is one this may act on.

    Parses rather than validates: what comes back is a tuple whose paths are already
    known to compose to a location under the workspace root, and whose count and total
    length are already known to be under the ceilings. Nothing downstream re-checks any
    of that, so nothing downstream can forget to.

    The ceilings are checked HERE, before the first fetch, and that ordering is the
    all-or-nothing rule showing up early: a restore that will refuse on the object past
    the ceiling should not first have spent the whole budget beneath it.

    A path goes through the lane's own `parse_relative_path`, so what this accepts is
    exactly what the sync was able to write. That is stricter than "does not escape" --
    it also refuses a leading dot, which is why no entry here can collide with the
    `.codex` and `.agents` directories the init container before this one creates.
    """
    if not isinstance(payload, dict):
        raise RestoreRefused(
            f"the working-lane listing is not an object: {type(payload).__name__}"
        )
    entries = payload.get("objects")
    if not isinstance(entries, list):
        raise RestoreRefused(
            "the working-lane listing carries no 'objects' array, so what the lane "
            "holds cannot be told apart from what it does not"
        )
    if len(entries) > WORKING_COUNT_LIMIT:
        raise RestoreRefused(
            f"the working lane lists {len(entries)} objects, over the "
            f"{WORKING_COUNT_LIMIT} that one Turn's sync will ever write"
        )
    parsed: list[LaneObject] = []
    total = 0
    for entry in entries:
        one = _parse_entry(entry)
        total += one.byte_length
        if total > WORKING_BUDGET_BYTES:
            raise RestoreRefused(
                f"the working lane lists more than {WORKING_BUDGET_BYTES} bytes, "
                "which is what this pod's workspace can hold beside what the agent "
                "writes"
            )
        parsed.append(one)
    return tuple(parsed)


def _parse_entry(entry: object) -> LaneObject:
    """One listing entry. Every way it can be wrong refuses the whole restore.

    `bool` is excluded from the length explicitly, because it is an `int` in Python: a
    `size: true` would otherwise parse as one byte and turn a broken listing into a
    file.
    """
    if not isinstance(entry, dict):
        raise RestoreRefused(
            f"a working-lane listing entry is not an object: {entry!r}"
        )
    relative = entry.get("path")
    if not isinstance(relative, str):
        raise RestoreRefused(f"a working-lane listing entry names no path: {entry!r}")
    length = entry.get("size")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise RestoreRefused(
            f"the working-lane object {relative!r} is listed at no readable size"
        )
    try:
        parse_relative_path(relative)
    except VfsPathInvalid as bad:
        raise RestoreRefused(
            f"the working lane lists {relative!r}, which is not a lane-relative path "
            "and so names a location this will not write"
        ) from bad
    return LaneObject(relative=relative, byte_length=length)


async def restore(
    client: httpx.AsyncClient, root: Path, /, *, report: Emit
) -> RestoreReport:
    """List this Session's working lane and write every object of it under `root`.

    Takes the client rather than building one, so the caller owns the base URL, the
    token header and the timeout -- and so a test drives this over the same httpx stack
    the pod uses, against a server answering the real contract.

    The empty lane is announced rather than passed over in silence. It is the common
    case -- every Session's first placement -- and it is also what a listing route that
    has quietly stopped finding anything looks like, so a reader of the log needs the
    two to be one sentence apart rather than indistinguishable.
    """
    objects = parse_listing(await _listing(client))
    if not objects:
        report(
            "the working lane holds no objects: restored 0 object(s), 0 byte(s). "
            "This Session has completed no Turn yet, or its last one left nothing."
        )
        return RestoreReport(objects_restored=0, bytes_restored=0)
    await _fetch_all(client, objects, root)
    written = sum(one.byte_length for one in objects)
    report(
        f"restored {len(objects)} object(s), {written} byte(s) of the working lane "
        f"into {root}"
    )
    return RestoreReport(objects_restored=len(objects), bytes_restored=written)


async def _listing(client: httpx.AsyncClient) -> object:
    """The lane's listing as JSON, or a refusal naming the status that came back.

    The response body is deliberately NOT quoted into the refusal. It is the Gateway's
    to write, and this message reaches the pod's status under
    `FallbackToLogsOnError` -- a body echoed there is a body disclosed to every reader
    of the pod.

    The ADDRESS is quoted, and that is the opposite decision for the opposite reason.
    The likeliest way this refuses in the cluster is that the connection never opened:
    `deploy/k8s/network-policies.yaml` allows this pod egress to the gateways on TCP
    8080 while both Services publish 80, and that file says in as many words that
    whether the translation holds is a live measurement nothing in this tree can settle.
    An operator reading a pod that would not start needs the host and port that was
    dialed in front of them rather than one code read away. It is safe to print because
    `read_binding` refuses a url carrying userinfo, so the authority is a Service
    address and never a credential.
    """
    try:
        answer = await client.get(LANE_ROUTE)
    except httpx.HTTPError as unreachable:
        raise RestoreRefused(
            f"the Tool Gateway at {client.base_url} did not answer for this Session's "
            f"working lane: {type(unreachable).__name__}"
        ) from unreachable
    if answer.status_code != httpx.codes.OK:
        raise RestoreRefused(
            f"the Tool Gateway answered {answer.status_code} for this Session's "
            "working lane, so what the lane holds is not known and a pod started now "
            "would present an empty tree as a complete one"
        )
    try:
        return answer.json()
    except ValueError as unreadable:
        raise RestoreRefused(
            "the Tool Gateway's working-lane listing did not parse as JSON"
        ) from unreadable


async def _fetch_all(
    client: httpx.AsyncClient, objects: Sequence[LaneObject], root: Path
) -> None:
    """Fetch and write every object, several at a time, refusing on the first failure.

    A `TaskGroup` rather than `gather`, and the difference is the point: `gather`
    propagates the first exception while leaving its siblings running, so a refusal
    would return to a caller with fetches still writing into the tree behind it. A task
    group cancels them, which is what makes "this refused" and "nothing more is being
    written" the same moment.

    What it costs is the exception shape, and that cost is paid here rather than passed
    on. A task group raises an `ExceptionGroup`, whose own message is "unhandled errors
    in a TaskGroup (1 sub-exception)" -- which names no object, no status and no cause,
    and which is what a reader of the pod's status would otherwise get, because this
    process prints the message and not the traceback.
    """
    limit = asyncio.Semaphore(_CONCURRENT_FETCHES)

    async def one(obj: LaneObject) -> None:
        async with limit:
            _write(root, obj, await _fetch(client, obj))

    try:
        async with asyncio.TaskGroup() as group:
            for obj in objects:
                group.create_task(one(obj))
    except BaseExceptionGroup as failures:
        raise _refusal_in(failures) from failures


def _refusal_in(failures: BaseExceptionGroup[BaseException]) -> RestoreRefused:
    """The one sentence a caller gets out of a task group's several failures.

    The first refusal is the one reported, because the reasons are rarely independent:
    a Gateway that has gone away fails every object in flight, and a reader needs the
    cause once rather than sixteen times. The count of the rest is carried anyway --
    one object missing and two hundred missing are different problems, and the
    difference is invisible from a single message.

    A failure that is not a refusal has escaped `_fetch`'s own translation -- a write
    that hit a full volume is the realistic one -- and is restated by type rather than
    re-raised, so the caller still has exactly one exception type to catch.
    """
    flat = list(_flattened(failures))
    refusals = [one for one in flat if isinstance(one, RestoreRefused)]
    first: BaseException = refusals[0] if refusals else flat[0]
    said = (
        str(first)
        if isinstance(first, RestoreRefused)
        else f"fetching the working lane failed: {type(first).__name__}: {first}"
    )
    if len(flat) == 1:
        return RestoreRefused(said)
    return RestoreRefused(f"{said} ({len(flat) - 1} further object(s) also failed)")


def _flattened(failures: BaseExceptionGroup[BaseException]) -> Iterator[BaseException]:
    """Every leaf of a group, in order, however deeply the groups nest.

    Written recursively rather than reading `.exceptions` once, because a task group's
    members may themselves be groups and a one-level read would report the inner group
    as the failure -- which is the same unreadable message this exists to avoid.
    """
    for one in failures.exceptions:
        if isinstance(one, BaseExceptionGroup):
            yield from _flattened(one)
        else:
            yield one


async def _fetch(client: httpx.AsyncClient, obj: LaneObject) -> bytes:
    """One object's bytes, at the length the listing promised or not at all.

    The length is compared because a short body is exactly the failure this module
    exists to prevent, and it is the one failure that leaves behind a file looking like
    a whole one. Nothing downstream re-reads it: the write trusts this.
    """
    try:
        answer = await client.get(f"{LANE_ROUTE}/{obj.relative}")
    except httpx.HTTPError as unreachable:
        raise RestoreRefused(
            f"the Tool Gateway at {client.base_url} did not answer for working-lane "
            f"object {obj.relative!r}: {type(unreachable).__name__}"
        ) from unreachable
    if answer.status_code != httpx.codes.OK:
        raise RestoreRefused(
            f"the working lane lists {obj.relative!r} and the Tool Gateway answered "
            f"{answer.status_code} for it, so the tree cannot be restored in full"
        )
    body = answer.content
    if len(body) != obj.byte_length:
        raise RestoreRefused(
            f"the working-lane object {obj.relative!r} was listed at "
            f"{obj.byte_length} bytes and {len(body)} arrived, so writing it would "
            "leave a truncated file the agent cannot tell from a whole one"
        )
    return body


def _write(root: Path, obj: LaneObject, body: bytes) -> None:
    """One object onto the workspace, with the directories above it made as needed.

    No traversal check here, and that is deliberate rather than missing: `parse_listing`
    has already put every path through the lane's own grammar, which admits no `..`, no
    leading separator and no empty segment. A second check on this side would be a
    branch no input can reach, and an unreachable branch reads to the next author as the
    thing that makes this safe.
    """
    destination = root / obj.relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)


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


async def run(config_toml: str, root: Path, /, *, report: Emit) -> RestoreReport:
    """Read this pod's own configuration and restore through the client it describes."""
    async with client_for(read_binding(config_toml)) as client:
        return await restore(client, root, report=report)


def _to_stderr(line: str) -> None:
    """One line, on the stream kubelet collects, flushed as it is written.

    stderr rather than a `logging` handler, because a level is one more thing that can
    be configured into silence -- and this process's whole account of itself is these
    lines. `flush` because the process may be about to exit non-zero, and a buffered
    last word is no word at all.
    """
    print(f"restore-working-lane: {line}", file=sys.stderr, flush=True)


def main() -> int:
    """Restore the lane, or refuse and take the pod down with the refusal.

    Returns the exit status rather than calling `sys.exit`, so the whole of it is
    reachable from a test. Every refusal is one line on stderr and a non-zero status:
    the pod does not start, the Turn is refused, and the reason is in the pod's own
    status because of `terminationMessagePolicy: FallbackToLogsOnError`.

    An unexpected exception is caught for one reason -- a traceback is not a sentence,
    and under that policy a traceback is what a reader of the pod would get. It is
    re-stated by type and message and the status is still non-zero, so nothing is
    swallowed.
    """
    root = Path(WORKSPACE_ROOT)
    try:
        config_toml = COMPILED_CONFIG.read_text(encoding="utf-8")
    except OSError as unreadable:
        _to_stderr(
            f"the compiled configuration at {COMPILED_CONFIG} could not be read "
            f"({type(unreadable).__name__}), so this pod cannot ask for its lane"
        )
        return 1
    try:
        report = asyncio.run(run(config_toml, root, report=_to_stderr))
    except RestoreRefused as refused:
        _to_stderr(f"refusing to start this pod: {refused}")
        return 1
    except Exception as unexpected:  # noqa: BLE001 - see the docstring
        _to_stderr(
            f"refusing to start this pod: {type(unexpected).__name__}: {unexpected}"
        )
        return 1
    _record(report)
    return 0


def _record(report: RestoreReport) -> None:
    """Put the restored totals where a reader of the pod's status can see them.

    Best-effort on purpose. The termination log is a file kubelet bind-mounts in, and
    every container here runs on a read-only root -- if that ever stops being writable,
    the restore has already succeeded and failing the pod over a status line would be
    the wrong trade. A failure to write it is said out loud rather than swallowed, so
    the absence of the line is not read as a restore that did not run.
    """
    summary = (
        f"restored {report.objects_restored} object(s), "
        f"{report.bytes_restored} byte(s) of the working lane"
    )
    try:
        TERMINATION_LOG.write_text(summary, encoding="utf-8")
    except OSError as unwritable:
        _to_stderr(
            f"restored the lane, but {TERMINATION_LOG} could not be written "
            f"({type(unwritable).__name__}), so the totals are only in this log"
        )


if __name__ == "__main__":
    raise SystemExit(main())
