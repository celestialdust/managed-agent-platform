"""The one write this pod accepts from outside it: a file placed into the workspace.

Driven through `create_shim_app` against the real route, with the target directory
redirected onto a tmp_path. What is NOT redirected is the authorisation: the token is
derived by the same function the control plane derives it with, so a case that passes
here would pass against the real pod.

The four refusals are the point of this file rather than the happy path. A write route
on the pod's outward-facing process is the one surface where "it worked" is the least
interesting thing that can be said about it: the questions worth asking are what it
does with no token, with another Session's id, with a name that climbs out of the
directory, and with a transfer that dies half way.

`WORKSPACE_FILES` is patched on the module the route reads it from rather than threaded
through `ServedSession`, for the same reason `RUNTIME_HOME` is in
`tests/session_shim/test_turn_complete.py`: that dataclass is the pod's process-wide
identity and this is a deployment constant everywhere but here.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.core.ids import SessionId, new_session_id
from managed_agent.session_shim.client import RuntimeConnection
from managed_agent.session_shim.pod_channel import shim_token_for
from managed_agent.session_shim.serve import (
    FILE_ROUTE,
    WORKSPACE_FILES,
    ServedSession,
    create_shim_app,
)

_THREAD = "0199c4de-6f2a-7b81-9c3d-4e5f60718293"
_KEY = b"a signing key for these cases only"
_BODY = b"# Brief\n\nRead this.\n"


def _app(
    monkeypatch: pytest.MonkeyPatch, files: Path, session_id: SessionId
) -> FastAPI:
    monkeypatch.setattr("managed_agent.session_shim.serve.WORKSPACE_FILES", files)
    return create_shim_app(
        ServedSession(
            session_id=session_id,
            thread_id=_THREAD,
            connection=RuntimeConnection(files / "never-dialled.sock"),
            token=shim_token_for(session_id, _KEY),
        )
    )


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pod.map-session"
    )


def _path_for(session_id: SessionId, name: str) -> str:
    return FILE_ROUTE.format(session_id=session_id, name=name)


def _bearer(session_id: SessionId) -> dict[str, str]:
    return {"Authorization": f"Bearer {shim_token_for(session_id, _KEY)}"}


async def test_the_body_lands_under_the_name_the_caller_chose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte for byte, in a directory the route creates if it is not there.

    The directory is created rather than assumed because a `subPath` mount presents an
    empty tree on a pod's first start and nothing else in this process makes it. A route
    that assumed it would answer 500 on the first file of every Session.
    """
    session_id = new_session_id()
    app = _app(monkeypatch, tmp_path / "files", session_id)
    async with _client(app) as client:
        answer = await client.put(
            _path_for(session_id, "brief.md"),
            content=_BODY,
            headers=_bearer(session_id),
        )
    assert answer.status_code == 204
    assert (tmp_path / "files" / "brief.md").read_bytes() == _BODY


async def test_a_second_file_does_not_disturb_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two files, both present afterwards.

    Worth its own case because the route writes through a scratch name and renames: a
    scratch name that were not per-file, or a rename that moved the directory rather
    than the file, would pass the single-file case above and lose one of these.
    """
    session_id = new_session_id()
    app = _app(monkeypatch, tmp_path / "files", session_id)
    async with _client(app) as client:
        for name, body in (("brief.md", _BODY), ("data.csv", b"a,b\n1,2\n")):
            answer = await client.put(
                _path_for(session_id, name),
                content=body,
                headers=_bearer(session_id),
            )
            assert answer.status_code == 204
    assert sorted(p.name for p in (tmp_path / "files").iterdir()) == [
        "brief.md",
        "data.csv",
    ]


async def test_a_re_placed_file_leaves_what_is_already_there_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first body, not the second, not the two concatenated, and not a refusal.

    A placement has to be idempotent -- a repeated one must not leave the agent reading
    a document twice over and must not fail because the name is already there -- and
    both halves of that still hold. What changed on 2026-08-26 is which bytes win, and
    the second write is now the one that does nothing.

    Until then the rename overwrote, which was harmless while a placement happened once
    per Session. ADR-041 leases a pod for one Turn, so every Turn is a placement and
    every placement re-pushes the Session's whole attachment set; ADR-035 puts the
    workspace on a volume that outlives the pod. Together those made this route
    overwrite, at the start of every Turn, a file the agent had spent the previous Turn
    editing -- silently, with the Turn reporting success.

    Different bytes on the second call, because identical ones could not tell the two
    behaviours apart. Production never sends different bytes under one name: a
    re-delivery is the same stored object, and a genuinely new document under a name the
    workspace holds is refused upstream at `control/api/routes/resources.py`.
    """
    session_id = new_session_id()
    app = _app(monkeypatch, tmp_path / "files", session_id)
    async with _client(app) as client:
        for body in (b"first", b"second"):
            answer = await client.put(
                _path_for(session_id, "brief.md"),
                content=body,
                headers=_bearer(session_id),
            )
            assert answer.status_code == 204
    assert (tmp_path / "files" / "brief.md").read_bytes() == b"first"


async def test_what_the_agent_wrote_into_an_attached_file_survives_the_next_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect the case above is the mechanism of, stated as the agent would meet it.

    Turn one places the document and the agent edits it. Turn two is a fresh pod on the
    same durable workspace, and its placement re-pushes the same attachment. The agent's
    work has to still be there -- it is the Session's output, the tenant paid for it,
    and nothing in the platform records that it was ever taken away.
    """
    session_id = new_session_id()
    files = tmp_path / "files"
    app = _app(monkeypatch, files, session_id)
    async with _client(app) as client:
        await client.put(
            _path_for(session_id, "brief.md"),
            content=b"# Brief\n",
            headers=_bearer(session_id),
        )
        (files / "brief.md").write_bytes(
            b"# Brief\n\n## Findings\n\nThe agent's work.\n"
        )
        answer = await client.put(
            _path_for(session_id, "brief.md"),
            content=b"# Brief\n",
            headers=_bearer(session_id),
        )
    assert answer.status_code == 204
    assert b"The agent's work." in (files / "brief.md").read_bytes()


async def test_no_scratch_file_is_left_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory holds the file and nothing else.

    The agent lists this directory. A leftover `.brief.md.partial` beside `brief.md` is
    a second copy of the tenant's document in the agent's own workspace, which is both
    confusing to it and a thing nobody asked to be stored twice.
    """
    session_id = new_session_id()
    app = _app(monkeypatch, tmp_path / "files", session_id)
    async with _client(app) as client:
        await client.put(
            _path_for(session_id, "brief.md"),
            content=_BODY,
            headers=_bearer(session_id),
        )
    assert [p.name for p in (tmp_path / "files").iterdir()] == ["brief.md"]


async def test_a_call_with_no_token_writes_nothing_and_says_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """404 with the rollout route's own body, and an empty directory.

    Both halves matter. The status alone would pass for a route that refused after
    writing, and the directory alone would pass for a route that answered 200 and did
    nothing.
    """
    session_id = new_session_id()
    files = tmp_path / "files"
    app = _app(monkeypatch, files, session_id)
    async with _client(app) as client:
        answer = await client.put(_path_for(session_id, "brief.md"), content=_BODY)
    assert answer.status_code == 404
    assert not files.exists() or list(files.iterdir()) == []


async def test_a_caller_naming_another_session_learns_the_same_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-identical to the no-token refusal, so neither says which Session this is.

    Compared as whole bodies rather than as statuses: two refusals with one status and
    two different messages still disclose which of the two conditions was met, and which
    Session a pod serves is the fact both refusals exist to withhold.
    """
    served = new_session_id()
    other = new_session_id()
    files = tmp_path / "files"
    app = _app(monkeypatch, files, served)
    async with _client(app) as client:
        no_token = await client.put(_path_for(other, "brief.md"), content=_BODY)
        wrong_session = await client.put(
            _path_for(other, "brief.md"), content=_BODY, headers=_bearer(served)
        )
    assert no_token.status_code == wrong_session.status_code == 404
    assert no_token.json() == wrong_session.json()
    assert not files.exists() or list(files.iterdir()) == []


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.md",
        "..%2Fescaped.md",
        "sub/brief.md",
        "..",
        ".",
        ".hidden.md",
        ".brief.md.partial",
    ],
)
async def test_a_name_that_is_not_a_bare_leaf_is_refused(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every one of these is a name that would write somewhere it should not.

    Parametrized because each entry encodes its own decision and a test naming two of
    them grades two. The last two are not traversals: a leading dot would hide the
    tenant's own document from the tools that list this directory, and
    `.brief.md.partial` is this route's scratch name -- a caller able to choose it could
    make the next placement of `brief.md` rename their bytes into place.

    **Two layers refuse these, and which does which was measured rather than assumed.**
    Only `.hidden.md` and `.brief.md.partial` reach `_is_a_bare_leaf` at all, and they
    come back 422. The five holding a separator or a dot component never reach the
    handler: Starlette's router does not match a `/` inside one path segment, does not
    decode `%2F` into one, and normalises `.` and `..` out of the path -- so those are
    404s from routing. Deleting the guard leaves those five green, which is why this
    says so rather than implying the guard stops all seven. The guard is still the one
    that matters: it is the only thing refusing the two dot names, and the layer that
    would still hold if this route's parameter ever took a `:path` converter.

    The assertion is that NOTHING was written anywhere under the tmp_path, not merely
    that the response was a 4xx: a route that refused after creating the file would
    satisfy a status check and would have already put the bytes on disk.
    """
    session_id = new_session_id()
    files = tmp_path / "files"
    app = _app(monkeypatch, files, session_id)
    async with _client(app) as client:
        answer = await client.put(
            f"/session/{session_id}/files/{name}",
            content=_BODY,
            headers=_bearer(session_id),
        )
    assert answer.status_code in (400, 404, 405, 422), answer.status_code
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [], written


def test_the_directory_it_writes_into_is_under_the_workspace_the_agent_reads() -> None:
    """One path, derived, not two spellings that happen to agree today.

    The runtime container mounts the workspace whole and the agent's writable root is
    `WORKSPACE_ROOT`; this container mounts the same volume with `subPath: files`. If
    this directory were not under that root, the file would arrive in the pod and the
    agent would have no path to it -- which is a Session that says the document is
    missing while the bytes sit on the same disk.
    """
    from managed_agent.control.pod_config.compiler import WORKSPACE_ROOT

    assert WORKSPACE_FILES.parent == Path(WORKSPACE_ROOT)
    assert Path(WORKSPACE_ROOT) != WORKSPACE_FILES, (
        "the attachment directory is the workspace root itself, so a placement could "
        "overwrite anything the agent wrote"
    )
