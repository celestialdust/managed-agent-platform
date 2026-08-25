"""What the pod must get right at a Turn boundary, over a real directory tree.

Two halves. The pure functions are exercised against a `tmp_path` laid out the way the
Agent Runtime lays out its own record, so the glob that finds a Rollout is the shipped
glob rather than a description of one. The route is driven through `create_shim_app`
over an ASGI transport, so what answers is the real route with the real refusals.

**What is not here.** No Agent Runtime wrote any of these files: the names and the
directory shape come from `research/plan-pass-rollout-and-resume.md` (Q1), read off
codex-rs at the pinned version, and no test in this repository has run the binary. If
the runtime lays its record out some other way, `find_rollout` returns nothing and the
route answers 204 -- which is why 204 is distinguishable from an empty body at the far
end.

The `ServedSession` below carries a `RuntimeConnection` that is constructed and never
dialled. That is honest rather than lazy: the Rollout route touches the runtime not at
all, and a socket opened here would be a fixture proving something this route does not
do. The Turn route's own tests are the ones that dial a real socket.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.control.pod_config.compiler import CODEX_HOME
from managed_agent.core.ids import SessionId
from managed_agent.session_shim.client import RuntimeConnection
from managed_agent.session_shim.pod_channel import shim_token_for
from managed_agent.session_shim.serve import (
    ROLLOUT_ROUTE,
    ServedSession,
    create_shim_app,
)
from managed_agent.session_shim.turn_complete import (
    RUNTIME_HOME,
    RolloutNotFound,
    find_rollout,
    subagent_tag,
)

_THREAD = "0199c4de-6f2a-7b81-9c3d-4e5f60718293"
_ANOTHER_THREAD = "7c1d2e3f-4a5b-6c7d-8e9f-a0b1c2d3e4f5"
_KEY = b"a signing key for these cases only"
_A_DAY = "sessions/2026/08/22"


def _rollout_file(home: Path, name: str, body: bytes, mtime: float) -> Path:
    """One file in the runtime's own layout, with its modification time pinned.

    Pinned rather than written in order: `find_rollout` sorts on `st_mtime` and two
    files written microseconds apart can share a timestamp on a filesystem with coarse
    granularity, which would make the newest-wins case pass or fail by luck.
    """
    path = home / _A_DAY / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    os.utime(path, (mtime, mtime))
    return path


# ------------------------------------------------------------------------------------
# Finding the Rollout
# ------------------------------------------------------------------------------------


def test_the_newest_file_for_a_thread_is_the_one_still_being_appended_to(
    tmp_path: Path,
) -> None:
    older = _rollout_file(
        tmp_path, f"rollout-2026-08-22T09-00-00-{_THREAD}.jsonl", b"older", 1_000.0
    )
    newer = _rollout_file(
        tmp_path, f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl", b"newer", 2_000.0
    )

    assert find_rollout(tmp_path, _THREAD) == newer
    assert older.exists(), "the older file is left alone, not consumed"


def test_the_reverted_form_carrying_a_second_id_is_matched(tmp_path: Path) -> None:
    """`thread/revert` appends `_<rollout_id>` after the stable thread id, so a name
    that ends at the thread id is not the only shape a live Rollout has."""
    reverted = _rollout_file(
        tmp_path,
        f"rollout-2026-08-22T10-00-00-{_THREAD}_{uuid4()}.jsonl",
        b"reverted",
        1_000.0,
    )
    assert find_rollout(tmp_path, _THREAD) == reverted


def test_a_compressed_rollout_is_not_offered_as_the_live_one(tmp_path: Path) -> None:
    """A rollout is compressed to `.jsonl.zst` once it is seven days cold and no pod
    lives that long, so a `.zst` here is not the file the runtime is appending to."""
    plain = _rollout_file(
        tmp_path, f"rollout-2026-08-22T09-00-00-{_THREAD}.jsonl", b"plain", 1_000.0
    )
    _rollout_file(
        tmp_path,
        f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl.zst",
        b"compressed",
        9_000.0,
    )
    assert find_rollout(tmp_path, _THREAD) == plain


def test_another_threads_record_in_the_same_day_is_not_chosen(tmp_path: Path) -> None:
    mine = _rollout_file(
        tmp_path, f"rollout-2026-08-22T09-00-00-{_THREAD}.jsonl", b"mine", 1_000.0
    )
    _rollout_file(
        tmp_path,
        f"rollout-2026-08-22T10-00-00-{_ANOTHER_THREAD}.jsonl",
        b"not mine",
        9_000.0,
    )
    assert find_rollout(tmp_path, _THREAD) == mine


def test_a_thread_with_no_record_says_where_it_looked(tmp_path: Path) -> None:
    """The message names the directory, because the failure this has actually produced
    is a shim container that does not mount the tree at all -- and a bare "not found"
    would read as a Session that has written nothing."""
    with pytest.raises(RolloutNotFound) as refused:
        find_rollout(tmp_path, _THREAD)
    assert str(tmp_path / "sessions") in str(refused.value)


def test_a_record_outside_the_dated_layout_is_not_reached(tmp_path: Path) -> None:
    """The glob is three levels deep by year/month/day. A file directly under
    `sessions/` is not what the runtime writes, and matching it would mean the depth in
    the pattern was decoration."""
    (tmp_path / "sessions").mkdir()
    (
        tmp_path / "sessions" / f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl"
    ).write_bytes(b"misplaced")
    with pytest.raises(RolloutNotFound):
        find_rollout(tmp_path, _THREAD)


def test_the_runtime_home_is_the_compilers_constant_and_not_a_second_spelling() -> None:
    """`config_compiler.CODEX_HOME` is the string the Permission Profile's deny rules
    are compiled against and the string the runtime container's `CODEX_HOME` is set to.

    Both halves are needed. The equality catches a wrong path; the source check catches
    the *right* path written out again as a literal, which the equality cannot see and
    which is the thing that drifts -- a third spelling is a third thing to change.
    """
    assert Path(CODEX_HOME) == RUNTIME_HOME

    module = Path(__file__).resolve().parents[2] / (
        "src/managed_agent/session_shim/turn_complete.py"
    )
    assert CODEX_HOME not in module.read_text(), (
        f"turn_complete.py spells {CODEX_HOME!r} out instead of importing it"
    )


# ------------------------------------------------------------------------------------
# Naming subagents
# ------------------------------------------------------------------------------------


def test_a_subagent_tag_is_the_same_after_a_recovery_as_before_it() -> None:
    session_id = SessionId(uuid4())
    runtime_id = str(uuid4())
    tag = subagent_tag(session_id, runtime_id)

    assert subagent_tag(session_id, runtime_id) == tag
    assert subagent_tag(session_id, str(uuid4())) != tag
    assert subagent_tag(SessionId(uuid4()), runtime_id) != tag
    assert runtime_id not in tag


# ------------------------------------------------------------------------------------
# The route the control plane reads the Rollout from
# ------------------------------------------------------------------------------------


def _a_shim_over(
    monkeypatch: pytest.MonkeyPatch, home: Path, session_id: SessionId
) -> FastAPI:
    """The real app, serving one Session, reading its Rollout out of `home`.

    `RUNTIME_HOME` is redirected on the module the route reads it from rather than
    threaded through `ServedSession`. That dataclass is the pod's process-wide identity
    and every field of it is required, so a fourth field would be an edit to every site
    that builds one -- for a value that is a deployment constant everywhere but here.
    """
    monkeypatch.setattr("managed_agent.session_shim.serve.RUNTIME_HOME", home)
    return create_shim_app(
        ServedSession(
            session_id=session_id,
            thread_id=_THREAD,
            connection=RuntimeConnection(home / "never-dialled.sock"),
            token=shim_token_for(session_id, _KEY),
        )
    )


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pod.map-session"
    )


def _path_for(session_id: SessionId) -> str:
    return ROLLOUT_ROUTE.format(session_id=session_id)


async def test_the_control_plane_reads_back_exactly_what_the_runtime_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = SessionId(uuid4())
    body = b'{"type":"session_meta"}\n{"type":"event_msg"}\n'
    _rollout_file(
        tmp_path, f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl", body, 1_000.0
    )

    app = _a_shim_over(monkeypatch, tmp_path, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _path_for(session_id),
            headers={"Authorization": f"Bearer {shim_token_for(session_id, _KEY)}"},
        )

    assert answer.status_code == 200
    assert answer.content == body


async def test_a_pod_that_has_written_no_rollout_answers_204_and_not_a_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Session's first Turn can complete before the runtime has flushed a file, and a
    control plane that read empty bytes as a Rollout would overwrite a good stored one
    with nothing. An empty 200 and a 200 holding an empty file are the same bytes, so
    the distinction has to be in the status."""
    session_id = SessionId(uuid4())

    app = _a_shim_over(monkeypatch, tmp_path, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _path_for(session_id),
            headers={"Authorization": f"Bearer {shim_token_for(session_id, _KEY)}"},
        )

    assert answer.status_code == 204
    assert answer.content == b""


def _a_stat_that_saw_less(
    monkeypatch: pytest.MonkeyPatch, target: Path, saw: int
) -> None:
    """Make the route's stat of `target` report `saw` bytes, whatever the file holds.

    "The file grew after the stat" stated as the stat rather than as a write hook, which
    makes it deterministic. It cannot be driven by appending from the test while the
    response streams: `httpx.ASGITransport` drains the whole response generator before
    the client is given a byte -- measured, not assumed -- so an append timed off the
    client's own reads always lands after the read it was meant to race.

    The window it stands in for is the ordinary one. Ship-out fires as a Turn completes
    and the runtime writes its token-count lines right after, so a transfer racing an
    append is the normal case, which is why the control plane's cut expects a torn tail.
    """
    real_stat = Path.stat

    def stating(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        seen = real_stat(self, follow_symlinks=follow_symlinks)
        if self != target:
            return seen
        fields = list(seen)
        fields[6] = saw
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "stat", stating)


async def test_a_rollout_declares_no_length_its_body_could_outgrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared `content-length` is what turns a racing append into a lost Turn.

    A response that sized itself from its own stat and then read to whatever EOF it
    found would send more bytes than it promised, and that is not a torn tail the
    control plane can drop: uvicorn raises `RuntimeError("Response content longer than
    Content-Length")` and drops the connection, the fetch sees
    `httpx.RemoteProtocolError`, and a Turn that did complete is recorded as failed with
    nothing shipped. `httpx.ASGITransport` enforces no length at all, which is why the
    header is asserted here directly rather than inferred from a transfer that survived.
    """
    session_id = SessionId(uuid4())
    body = b'{"type":"session_meta"}\n{"type":"event_msg"}\n'
    _rollout_file(
        tmp_path, f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl", body, 1_000.0
    )

    app = _a_shim_over(monkeypatch, tmp_path, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _path_for(session_id),
            headers={"Authorization": f"Bearer {shim_token_for(session_id, _KEY)}"},
        )

    assert answer.status_code == 200
    assert "content-length" not in answer.headers
    assert answer.content == body


async def test_a_rollout_that_grew_after_the_stat_serves_what_the_stat_saw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One prefix of one moment, not a stripe across several.

    The bound is the size the route read, so the bytes the control plane gets are a
    prefix of the file as it stood at one instant. Reading to EOF instead would hand
    back a body assembled from two states of a file that was being appended to -- and
    the far end cuts at the last completed Turn, so bytes past the size that was
    measured buy nothing and make the response non-deterministic for no gain.

    The chunk size is lowered so the read spans several chunks; at the shipped 64 KiB
    these bytes are one chunk and a bound that did nothing would look identical.
    """
    session_id = SessionId(uuid4())
    at_stat_time = b'{"type":"session_meta"}\n{"type":"event_msg"}\n'
    late = b'{"type":"event_msg","late":true}\n'
    path = _rollout_file(
        tmp_path,
        f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl",
        at_stat_time + late,
        1_000.0,
    )
    monkeypatch.setattr("managed_agent.session_shim.serve._ROLLOUT_CHUNK_BYTES", 8)

    app = _a_shim_over(monkeypatch, tmp_path, session_id)
    _a_stat_that_saw_less(monkeypatch, path, len(at_stat_time))
    async with _client(app) as client:
        answer = await client.get(
            _path_for(session_id),
            headers={"Authorization": f"Bearer {shim_token_for(session_id, _KEY)}"},
        )

    assert answer.status_code == 200
    assert path.read_bytes() == at_stat_time + late, "the file really does hold more"
    assert answer.content == at_stat_time


async def test_a_rollout_file_that_exists_and_holds_nothing_answers_204_as_well(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case above has no file at all. This one has a file with nothing in it, and
    the two are the same fact about the pod: no Rollout has been written yet.

    The runtime creates the file before it flushes its first line, so between those two
    moments a `find_rollout` finds a path and the bytes behind it are zero. Answering
    200 there sends an empty body a control plane cannot tell from a real Rollout, and
    the ship-out that follows replaces a good stored record with nothing -- after which
    every resume of that Session reads a Rollout with no lines and refuses, and the
    stored object is only replaced by the next completed Turn, which needs a resume.

    The status is where the distinction has to live because the body cannot carry it:
    an empty 200 and a 200 holding an empty file are the same bytes.
    """
    session_id = SessionId(uuid4())
    _rollout_file(
        tmp_path, f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl", b"", 1_000.0
    )

    app = _a_shim_over(monkeypatch, tmp_path, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _path_for(session_id),
            headers={"Authorization": f"Bearer {shim_token_for(session_id, _KEY)}"},
        )

    assert answer.status_code == 204
    assert answer.content == b""


async def test_the_two_ways_of_being_refused_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-identical, which is what makes the refusal useless as an existence oracle.

    Both arms name **another** Session; the second presents this pod's own correct
    token, which is the input that separates the two checks. Presenting the other
    Session's token instead would let the token check refuse it on its own, both arms
    would agree for one reason, and deleting the Session comparison outright would leave
    this green -- the same trap the Turn route's own case records.

    The positive read runs last against the same app, so the negative assertions hold in
    a world where a 200 was reachable.
    """
    served_id = SessionId(uuid4())
    other_id = SessionId(uuid4())
    body = b'{"type":"session_meta"}\n'
    _rollout_file(
        tmp_path, f"rollout-2026-08-22T10-00-00-{_THREAD}.jsonl", body, 1_000.0
    )
    this_pods_token = {"Authorization": f"Bearer {shim_token_for(served_id, _KEY)}"}

    app = _a_shim_over(monkeypatch, tmp_path, served_id)
    async with _client(app) as client:
        untokened = await client.get(_path_for(other_id))
        with_this_pods_token = await client.get(
            _path_for(other_id), headers=this_pods_token
        )
        served = await client.get(_path_for(served_id), headers=this_pods_token)

    assert untokened.status_code == with_this_pods_token.status_code == 404
    assert untokened.content == with_this_pods_token.content
    assert untokened.json()["code"] == "session.not_found"
    assert untokened.json()["detail"] == {"session_id": str(other_id)}
    assert str(served_id) not in untokened.text, "the served Session is not named"
    assert body not in untokened.content
    assert served.status_code == 200, "a 200 was reachable on this app"
    assert served.content == body


async def test_a_refused_read_never_reaches_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both checks happen before the filesystem is touched. Pointed at a home that does
    not exist, a refusal must still be the platform's refusal rather than a 500 out of
    a glob -- and the served Session's own read below shows the tree is genuinely
    unreadable, so the first answer is not merely a coincidence."""
    served_id = SessionId(uuid4())
    absent = tmp_path / "no-such-home"

    app = _a_shim_over(monkeypatch, absent, served_id)
    async with _client(app) as client:
        refused = await client.get(_path_for(served_id))
        served = await client.get(
            _path_for(served_id),
            headers={"Authorization": f"Bearer {shim_token_for(served_id, _KEY)}"},
        )

    assert refused.status_code == 404
    assert served.status_code == 204


async def test_the_rollout_is_readable_and_not_writable_over_this_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GET and nothing else. A route that also accepted a write would be a way to
    put chosen bytes into a Session's resume state from inside the pod."""
    session_id = SessionId(uuid4())

    app = _a_shim_over(monkeypatch, tmp_path, session_id)
    async with _client(app) as client:
        posted = await client.post(
            _path_for(session_id),
            content=b"chosen bytes",
            headers={"Authorization": f"Bearer {shim_token_for(session_id, _KEY)}"},
        )

    assert posted.status_code == 405
