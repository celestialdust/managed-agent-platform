"""The init container that puts a Session's Rollout back under `CODEX_HOME`.

Tier 1 (local, no infrastructure). The fetch runs over the real httpx stack against a
Starlette app answering the contract the Tool Gateway serves, so what is graded is this
module's behaviour against a server rather than against a mock of one.

**The case this file exists for is the one that must NOT be quiet.** A Session that has
completed a Turn and has no stored Rollout takes the pod down, because the only other
option is a fresh thread over a folded conversation -- which succeeds, costs the tenant
a replay, and announces nothing. Every "refuses" case here is that hazard under a
different cause, and the parametrized statuses include 404 on purpose: a route that has
moved answers exactly like a Session with no Rollout.

The other property graded here is the seeded PATH, and it is not cosmetic. The runtime
keeps appending to whatever file it was resumed from, and ship-out finds a Session's
Rollout by globbing the runtime's own layout -- so a file written anywhere else resumes
correctly, runs a Turn, and ships nothing, losing that Turn silently at the next
recovery. The path is therefore compared against the real `find_rollout` rather than
against a string this file writes down.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from managed_agent.core.session.session_token import SESSION_TOKEN_HEADER_NAME
from managed_agent.session_shim import seed_rollout
from managed_agent.session_shim.restore_working_lane import Emit, RestoreRefused
from managed_agent.session_shim.seed_rollout import (
    _SEED_BUDGET_BYTES,
    RESUMING_ENV,
    SEED_ROUTE,
    find_seeded,
    is_resuming,
    main,
    run,
    seed,
    seeded_path,
    thread_id_at,
    thread_id_in,
    write_seed,
)
from managed_agent.session_shim.turn_complete import RolloutNotFound, find_rollout

THREAD: Final = "0199c2f7-0000-7000-8000-0000000000ab"
TOKEN: Final = "a-session-token-no-line-may-carry"

A_ROLLOUT: Final = (
    json.dumps({"type": "session_meta", "payload": {"id": THREAD}}).encode()
    + b"\n"
    + json.dumps({"type": "event_msg", "payload": {"type": "turn_complete"}}).encode()
    + b"\n"
)

AT: Final = datetime(2026, 8, 24, 11, 22, 33, tzinfo=UTC)

NOWHERE: Final = "http://127.0.0.1:1/mcp"
"""A URL that resolves and refuses, so an unreachable Gateway costs no DNS wait."""


async def _one_chunk(body: bytes) -> AsyncIterator[bytes]:
    """The body as a stream, so the answer carries no content-length."""
    yield body


def _said() -> tuple[list[str], Emit]:
    """A report sink and the list it fills, so what a call announced can be asserted."""
    lines: list[str] = []
    return lines, lines.append


def _compiled(url: str = NOWHERE, token: str = TOKEN) -> str:
    """The pod's compiled document, in the shape `read_binding` reads a binding out of.

    Carries the token in the header table it really rides in, so a refusal that
    interpolated the document it was handed would put the value in its own message.
    """
    return (
        '[mcp_servers."map-tool-gateway"]\n'
        f'url = "{url}"\n'
        f'http_headers = {{ "{SESSION_TOKEN_HEADER_NAME}" = "{token}" }}\n'
    )


# --------------------------------------------------------------------------------------
# A fake Gateway, answering the contract the real route serves
# --------------------------------------------------------------------------------------


class FakeGateway:
    """The seed route, with the answer under test and a record of what was presented."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = A_ROLLOUT,
        declared_length: str | None = None,
        streamed: bool = False,
    ) -> None:
        self.status = status
        self.body = body
        self.declared_length = declared_length
        self.streamed = streamed
        self.tokens: list[str | None] = []
        self.paths: list[str] = []

    def app(self) -> Starlette:
        async def serve(request: Request) -> Response:
            self.tokens.append(request.headers.get(SESSION_TOKEN_HEADER_NAME))
            self.paths.append(request.url.path)
            if self.status == 204:
                return Response(status_code=204)
            if self.streamed:
                # A chunked answer, which declares no length at all -- the state the
                # `None` branch of the ceiling exists for, and one no plain `Response`
                # can produce because Starlette always measures a body it holds whole.
                return StreamingResponse(_one_chunk(self.body))
            answer = Response(content=self.body, status_code=self.status)
            if self.declared_length is not None:
                answer.headers["content-length"] = self.declared_length
            return answer

        return Starlette(routes=[Route(SEED_ROUTE, serve, methods=["GET"])])

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app()),
            base_url="http://tool-gateway.invalid",
            headers={SESSION_TOKEN_HEADER_NAME: TOKEN},
        )


# --------------------------------------------------------------------------------------
# Whether this pod is resuming at all
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("said", "expected"), [("true", True), ("false", False)])
def test_the_two_spellings_the_adapter_emits_are_the_two_this_reads(
    said: str, expected: bool
) -> None:
    assert is_resuming({RESUMING_ENV: said}) is expected


@pytest.mark.parametrize("said", ["True", "FALSE", "1", "yes", "", "  true"])
def test_any_other_spelling_refuses_rather_than_guessing(said: str) -> None:
    """Every value a tolerant reader would have to guess at maps onto "not resuming",
    which is the dangerous reading: it turns a variable somebody got wrong into a
    silently fresh thread for a Session that had a conversation. `True` is in this list
    on purpose -- it is what `str(bool)` produces, so it is the misspelling most likely
    to arrive from the other side of this contract."""
    with pytest.raises(RestoreRefused, match=RESUMING_ENV):
        is_resuming({RESUMING_ENV: said})


def test_an_absent_variable_refuses_rather_than_assuming_a_first_placement() -> None:
    """Absence is what a manifest that dropped the entry produces, and the tempting
    default -- treating it as a first placement -- is exactly the wrong one."""
    with pytest.raises(RestoreRefused, match=RESUMING_ENV):
        is_resuming({})


# --------------------------------------------------------------------------------------
# The thread id comes out of the bytes
# --------------------------------------------------------------------------------------


def test_the_thread_is_read_from_the_records_own_first_line() -> None:
    assert thread_id_in(A_ROLLOUT) == THREAD


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"\n\n",
        b"not json\n",
        b'{"type":"event_msg","payload":{"type":"turn_complete"}}\n',
        b'{"type":"session_meta"}\n',
        b'{"type":"session_meta","payload":{}}\n',
        b'{"type":"session_meta","payload":{"id":""}}\n',
        b'{"type":"session_meta","payload":{"id":7}}\n',
    ],
)
def test_bytes_that_are_not_a_resumable_record_refuse_here(body: bytes) -> None:
    """Refused where the message can name the cause, rather than inside a runtime whose
    failure reaches nobody. A record that does not open with `session_meta` is the one
    shape the runtime's own reader treats as a hard error, so writing these to disk
    would trade a clear refusal for an obscure one."""
    with pytest.raises(RestoreRefused):
        thread_id_in(body)


def test_the_id_is_read_off_a_record_whose_tail_does_not_parse(tmp_path: Path) -> None:
    """A Rollout is append-only and may be read while a line is half-written.

    So the id is taken from the first line alone: an implementation that parsed the
    whole file to find the `session_meta` would refuse a record that is perfectly
    resumable, and would refuse it at the moment a pod is trying to come back.
    """
    record = tmp_path / "rollout.jsonl"
    record.write_bytes(A_ROLLOUT + b'{"type":"event_msg","payl')

    assert thread_id_at(record) == THREAD


# --------------------------------------------------------------------------------------
# Where the file lands -- checked against the reader that has to find it
# --------------------------------------------------------------------------------------


def test_the_seeded_file_is_where_ship_out_looks_for_it(tmp_path: Path) -> None:
    """The property, stated against the real finder rather than against a path string.

    `find_rollout` is what the shim's ship-out calls at every completed Turn. A seeded
    file it cannot locate resumes perfectly and then ships NOTHING -- the Session comes
    back a second time from the Rollout it had before this pod ran, losing a whole Turn
    with nothing anywhere saying so. Comparing against a literal here would prove two
    strings match; comparing against the finder proves the contract holds.
    """
    lines, say = _said()

    written = write_seed(tmp_path, A_ROLLOUT, AT, report=say)

    assert find_rollout(tmp_path, THREAD) == written
    assert written.read_bytes() == A_ROLLOUT
    assert any(str(written) in line for line in lines)


def test_the_name_carries_the_thread_and_the_layout_the_runtime_parses() -> None:
    """Both halves of the filename are load-bearing, to a different reader each.

    The three date directories and the `rollout-` prefix are what the runtime's own
    filename parser accepts; the trailing thread id is what the ship-out glob matches
    on. A name satisfying one and not the other fails in a different component from the
    one that wrote it.
    """
    path = seeded_path(Path("/var/lib/map/codex"), THREAD, AT)

    assert path.parts[-4:-1] == ("2026", "08", "24")
    assert path.name == f"rollout-2026-08-24T11-22-33-{THREAD}.jsonl"


def test_a_pod_that_was_not_seeded_finds_nothing(tmp_path: Path) -> None:
    """The common path: a first placement, whose runtime home is empty."""
    assert find_seeded(tmp_path) is None


def test_what_was_seeded_is_what_is_found(tmp_path: Path) -> None:
    """The two halves of one spelling, round-tripped, because the shim decides between
    continuing a thread and opening one from the read half alone."""
    lines, say = _said()
    written = write_seed(tmp_path, A_ROLLOUT, AT, report=say)

    assert find_seeded(tmp_path) == written


def test_two_records_before_a_thread_is_opened_refuse_rather_than_pick_one(
    tmp_path: Path,
) -> None:
    """Which of two conversations a Session continues is not a coin to flip.

    Unreachable in the pod as built -- the runtime writes nothing before the shim opens
    a thread -- and kept anyway because it is a property of a FILESYSTEM rather than of
    a constant: anything that ever wrote a second record here would otherwise have one
    of the two silently chosen by sort order.
    """
    lines, say = _said()
    write_seed(tmp_path, A_ROLLOUT, AT, report=say)
    other = A_ROLLOUT.replace(THREAD.encode(), b"a-second-thread")
    write_seed(tmp_path, other, datetime(2026, 8, 25, 1, 2, 3, tzinfo=UTC), report=say)

    with pytest.raises(RestoreRefused, match="ambiguous"):
        find_seeded(tmp_path)


def test_a_stray_file_under_the_sessions_tree_is_not_read_as_a_conversation(
    tmp_path: Path,
) -> None:
    """Matched by the runtime's filename shape, not by living in the right directory.

    A lock file or an editor's leavings beside the record would otherwise either be
    resumed from or make the seeded record look ambiguous -- and the second of those
    turns a stray byte into a pod that will not start.
    """
    lines, say = _said()
    written = write_seed(tmp_path, A_ROLLOUT, AT, report=say)
    (written.parent / ".rollout.jsonl.swp").write_bytes(b"x")
    (written.parent / "history.jsonl").write_bytes(b"x")

    assert find_seeded(tmp_path) == written


# --------------------------------------------------------------------------------------
# The seed itself
# --------------------------------------------------------------------------------------


async def test_a_first_placement_asks_for_nothing_and_seeds_nothing(
    tmp_path: Path,
) -> None:
    """Zero round trips on the common path, and an empty runtime home afterwards.

    The Gateway below WOULD serve a Rollout, which is what makes this an assertion
    about the caller rather than about an empty store: a version that asked anyway
    would write a record into a Session that has never run, and the shim would then
    continue a conversation that is not this Session's.
    """
    gateway = FakeGateway()
    lines, say = _said()

    async with gateway.client() as client:
        assert await seed(client, tmp_path, False, report=say) is None

    assert gateway.paths == []
    assert find_seeded(tmp_path) is None
    assert any("first placement" in line for line in lines)


async def test_a_resuming_placement_writes_what_the_gateway_serves(
    tmp_path: Path,
) -> None:
    """One GET, at the one path, presenting the one token, and the bytes land intact."""
    gateway = FakeGateway()
    lines, say = _said()

    async with gateway.client() as client:
        written = await seed(client, tmp_path, True, report=say)

    assert written is not None
    assert written.read_bytes() == A_ROLLOUT
    assert gateway.paths == [SEED_ROUTE]
    assert gateway.tokens == [TOKEN]


async def test_a_resuming_session_with_no_stored_rollout_takes_the_pod_down(
    tmp_path: Path,
) -> None:
    """The whole reason this container exists, and the one case that must not be quiet.

    Starting anyway is not a degraded mode: it opens a NEW thread over a conversation
    that already exists, replays history the runtime's compaction checkpoints have
    folded, charges the tenant for the replay and reports success. Nothing downstream
    can see that it happened. So this refuses, the pod never starts, and the placement
    fails with the reason in the pod's own status.
    """
    gateway = FakeGateway(status=204)
    lines, say = _said()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused, match="NEW thread"):
            await seed(client, tmp_path, True, report=say)

    assert find_seeded(tmp_path) is None


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 503])
async def test_a_gateway_that_will_not_answer_takes_the_pod_down(
    tmp_path: Path, status: int
) -> None:
    """An error is not an absence. 404 is in this list deliberately: a route that has
    moved answers exactly like a Session with no Rollout, and reading the two the same
    way would turn one deployment mistake into a fleet of silently fresh threads."""
    gateway = FakeGateway(status=status)
    lines, say = _said()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused, match=str(status)):
            await seed(client, tmp_path, True, report=say)

    assert find_seeded(tmp_path) is None


async def test_a_rollout_over_the_budget_is_refused_before_it_is_downloaded(
    tmp_path: Path,
) -> None:
    """Held against what the answer DECLARES, before any of the body is read.

    The Gateway below serves two short lines and claims a length over the budget, so an
    implementation that measured what it had already downloaded would accept this. That
    is the distinction being drawn: the point of the ceiling is to refuse without first
    pulling the bytes it is refusing.

    What fails at this size is the volume and not the read: `codex-home` is a bounded
    emptyDir shared with everything the runtime writes, and a pod that overruns it is
    evicted mid-Turn -- which loses the Turn and looks like a node problem. Refusing
    says which Session and which number instead.
    """
    gateway = FakeGateway(declared_length=str(_SEED_BUDGET_BYTES + 1))
    lines, say = _said()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused, match=str(_SEED_BUDGET_BYTES)):
            await seed(client, tmp_path, True, report=say)

    assert find_seeded(tmp_path) is None


@pytest.mark.parametrize(
    "gateway",
    [
        FakeGateway(streamed=True),
        FakeGateway(declared_length="not-a-number"),
    ],
    ids=["chunked", "unreadable"],
)
async def test_a_body_whose_length_is_not_declared_is_refused(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """A size known only once the body has been read is no ceiling at all.

    Both spellings of "no usable length", because they arrive by different routes: a
    chunked answer declares none, and a malformed header declares one that cannot be
    compared. Reading either as "small enough" would put the ceiling back where it
    started -- after the download.
    """
    lines, say = _said()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused, match="length"):
            await seed(client, tmp_path, True, report=say)


async def test_a_served_body_that_is_not_a_rollout_is_refused_before_it_is_written(
    tmp_path: Path,
) -> None:
    """Nothing lands unless it names the thread it belongs to.

    A file written first and validated later is one the shim would then find and resume
    from, which is the failure this ordering removes rather than reports.
    """
    gateway = FakeGateway(body=b"{}\n")
    lines, say = _said()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused):
            await seed(client, tmp_path, True, report=say)

    assert find_seeded(tmp_path) is None


async def test_a_gateway_that_is_not_there_takes_the_pod_down(tmp_path: Path) -> None:
    """Unreachable and empty are different facts, and only one of them is safe."""
    lines, say = _said()

    async with httpx.AsyncClient(base_url=NOWHERE) as client:
        with pytest.raises(RestoreRefused, match="did not answer"):
            await seed(client, tmp_path, True, report=say)


async def test_the_client_is_built_from_this_pods_own_compiled_document(
    tmp_path: Path,
) -> None:
    """`run` reads where to ask and what to present out of the mounted document.

    A document naming no Gateway refuses here rather than surfacing as an httpx error
    naming a URL and no cause -- and, worse, rather than as a request sent with no
    token, which the Gateway answers with the same fixed 401 as a forged one.
    """
    lines, say = _said()

    with pytest.raises(RestoreRefused):
        await run("", tmp_path, True, report=say)


# --------------------------------------------------------------------------------------
# Nothing this process says carries the token -- on the error paths especially
# --------------------------------------------------------------------------------------


async def test_no_error_path_prints_or_raises_this_session_s_token(
    tmp_path: Path,
) -> None:
    """Load-bearing rather than tidy, and enumerated over failures rather than success.

    The container declares `terminationMessagePolicy: FallbackToLogsOnError`, so every
    line an error path writes is promoted into the pod's status, which more people can
    read than can run `kubectl logs`. A happy-path-only version of this passes while the
    leak sits in an exception handler -- which is exactly where a value gets
    interpolated into a message by somebody explaining what went wrong. So each refusal
    below is a different handler, and the two `run` cases hold the token in hand.
    """
    lines, say = _said()
    refusals: list[str] = []

    async def refused(gateway: FakeGateway, at: Path) -> None:
        async with gateway.client() as client:
            with pytest.raises(RestoreRefused) as refusal:
                await seed(client, at, True, report=say)
        refusals.append(str(refusal.value))

    await refused(FakeGateway(status=204), tmp_path / "absent")
    await refused(FakeGateway(status=500), tmp_path / "erroring")
    await refused(FakeGateway(status=401), tmp_path / "unauthorised")
    await refused(FakeGateway(body=b"{}\n"), tmp_path / "unparseable")
    await refused(
        FakeGateway(declared_length=str(_SEED_BUDGET_BYTES + 1)), tmp_path / "oversized"
    )
    await refused(FakeGateway(declared_length="?"), tmp_path / "undeclared")

    for document in (_compiled(), _compiled(url="file:///x"), f'token = "{TOKEN}"\n'):
        with pytest.raises(RestoreRefused) as refusal:
            await run(document, tmp_path / "dialling", True, report=say)
        refusals.append(str(refusal.value))

    async with FakeGateway().client() as client:
        await seed(client, tmp_path / "seeded", True, report=say)

    assert len(refusals) == 9, refusals
    assert lines, "nothing was announced, so the emitted half grades nothing"
    for line in (*refusals, *lines):
        assert TOKEN not in line, line


def test_the_refusal_the_pod_exits_with_carries_no_token_either(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same guard one level up, over what `main` actually writes to stderr.

    The functions above are checked on what they raise; this is checked on what the
    process prints, because between the two sits a handler that restates an exception
    by type and message -- and that restatement is the line kubelet promotes.
    """
    document = tmp_path / "config.toml"
    document.write_text(_compiled(), encoding="utf-8")
    monkeypatch.setattr(seed_rollout, "COMPILED_CONFIG", document)
    monkeypatch.setattr(seed_rollout, "CODEX_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(RESUMING_ENV, "true")

    assert main() == 1

    printed = capsys.readouterr().err
    assert "refusing to start this pod" in printed
    assert TOKEN not in printed, printed


def test_a_first_placement_exits_zero_without_asking_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The common path all the way through the entry point, against a Gateway that is
    not there: a first placement must not need one, so an unreachable URL is the
    sharpest way to assert that it never dialled."""
    document = tmp_path / "config.toml"
    document.write_text(_compiled(), encoding="utf-8")
    monkeypatch.setattr(seed_rollout, "COMPILED_CONFIG", document)
    monkeypatch.setattr(seed_rollout, "CODEX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(seed_rollout, "TERMINATION_LOG", tmp_path / "termination")
    monkeypatch.setenv(RESUMING_ENV, "false")

    assert main() == 0
    assert "no Rollout to seed" in (tmp_path / "termination").read_text()


def test_an_unreadable_environment_takes_the_pod_down_before_anything_is_dialled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pod whose resume fact did not arrive cannot tell the two placements apart, and
    guessing is what this whole container exists to stop."""
    monkeypatch.setattr(seed_rollout, "COMPILED_CONFIG", tmp_path / "config.toml")
    monkeypatch.delenv(RESUMING_ENV, raising=False)

    assert main() == 1
    assert RESUMING_ENV in capsys.readouterr().err


def test_an_unreadable_compiled_document_names_the_path_it_could_not_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The mount is what failed here, and the message has to say so: without the path,
    a missing `compiled` volume and a malformed document read identically."""
    missing = tmp_path / "no" / "such" / "config.toml"
    monkeypatch.setattr(seed_rollout, "COMPILED_CONFIG", missing)
    monkeypatch.setenv(RESUMING_ENV, "true")

    assert main() == 1
    assert str(missing) in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# The negative control under the round trip above
# --------------------------------------------------------------------------------------


def test_a_pod_that_was_never_seeded_makes_find_rollout_say_so(tmp_path: Path) -> None:
    """`find_rollout` really does fail on an empty home, so its agreement with
    `seeded_path` above is a fact about that tree rather than about any tree."""
    with pytest.raises(RolloutNotFound):
        find_rollout(tmp_path, THREAD)
