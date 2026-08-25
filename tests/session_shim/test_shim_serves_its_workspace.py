"""The read that carries the agent's WORKING tree out of the pod.

The sibling of `test_shim_serves_its_outputs.py`, over the same read mount and the same
two authorisation checks, and it exists because the two listings answer opposite
questions. The outputs listing asks "what did the agent say it produced"; this one asks
"what would the agent miss if this pod vanished", and the honest answer to that includes
the runtime state and the half-finished script the other listing exists to leave behind.

**The inclusions are the point of this file rather than the refusals.** A case asserting
that `.codex/` is walked is the one that would fail if somebody later reused the
produced predicate here for tidiness, and reusing it would silently make a resumed
Session start with an amnesiac agent. What is excluded is only `NOT_SYNCED`, and there
is a case per member of it.

`WORKSPACE_READ_ROOT` is patched on the module the routes read it from, for the reason
the outputs file gives: it is a deployment constant everywhere but here.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.control.files.store import content_digest
from managed_agent.core.ids import SessionId, new_session_id
from managed_agent.core.pod.workspace_contract import (
    INPUT_DIR_NAME,
    NOT_SYNCED,
    OUTPUT_DIR_NAME,
)
from managed_agent.session_shim.client import RuntimeConnection
from managed_agent.session_shim.pod_channel import shim_token_for
from managed_agent.session_shim.serve import (
    WORKSPACE_ROUTE,
    ProducedFiles,
    ServedSession,
    create_shim_app,
    workspace_path_for,
)

_THREAD = "0199c4de-6f2a-7b81-9c3d-4e5f60718293"
_KEY = b"a signing key for these cases only"
_SCRIPT = b"import pandas\n\nprint('halfway through')\n"


def _app(monkeypatch: pytest.MonkeyPatch, root: Path, session_id: SessionId) -> FastAPI:
    monkeypatch.setattr("managed_agent.session_shim.serve.WORKSPACE_READ_ROOT", root)
    return create_shim_app(
        ServedSession(
            session_id=session_id,
            thread_id=_THREAD,
            connection=RuntimeConnection(root / "never-dialled.sock"),
            token=shim_token_for(session_id, _KEY),
        )
    )


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pod.map-session"
    )


def _bearer(session_id: SessionId) -> dict[str, str]:
    return {"Authorization": f"Bearer {shim_token_for(session_id, _KEY)}"}


def _listing_path(session_id: SessionId) -> str:
    return WORKSPACE_ROUTE.format(session_id=session_id)


def _write(root: Path, relative: str, body: bytes) -> None:
    """One file at a workspace-relative path, with its parents made as needed."""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)


async def _listing(
    monkeypatch: pytest.MonkeyPatch, root: Path, session_id: SessionId
) -> ProducedFiles:
    app = _app(monkeypatch, root, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )
    assert answer.status_code == 200, answer.content
    return ProducedFiles.model_validate(answer.json())


async def test_a_script_the_agent_left_at_the_root_is_listed_with_length_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path, byte length and content digest, because the far end needs all three.

    The digest is asserted against `content_digest` -- the function the control plane
    hashes an arriving body with -- rather than against a literal, because the whole
    value of carrying it is that the two agree. A digest computed here by a different
    rule would make every file look changed on every Turn, and nothing would fail: the
    sync would just re-upload the whole tree forever.
    """
    root = tmp_path / "workspace"
    _write(root, "analysis.py", _SCRIPT)
    listing = await _listing(monkeypatch, root, new_session_id())
    assert [(f.name, f.byte_length, f.content_sha256) for f in listing.files] == [
        ("analysis.py", len(_SCRIPT), content_digest(_SCRIPT))
    ]


async def test_a_file_nested_below_the_root_is_listed_at_its_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The working walk descends at the root, where the produced walk does not.

    That asymmetry is deliberate and it is what this case pins. The produced fallback
    stays flat because nothing has been said about the root; here everything has been
    said -- the agent's project tree IS the thing being kept -- so a source file two
    directories down has to come back at the path it sits at.
    """
    root = tmp_path / "workspace"
    _write(root, "src/pipeline/load.py", _SCRIPT)
    listing = await _listing(monkeypatch, root, new_session_id())
    assert [f.name for f in listing.files] == ["src/pipeline/load.py"]


async def test_a_dotted_directory_below_the_first_segment_is_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The produced walk refuses a dotted segment at any depth; this one does not.

    `src/.cache/build.json` is the agent's own tree and comes back. Reusing the produced
    predicate here for tidiness would compile, pass most of this file, and quietly drop
    whatever the agent kept under a dotted directory of its own making.
    """
    root = tmp_path / "workspace"
    _write(root, "src/.cache/build.json", b'{"built": true}\n')
    listing = await _listing(monkeypatch, root, new_session_id())
    assert [f.name for f in listing.files] == ["src/.cache/build.json"]


async def test_a_dotted_directory_at_the_root_is_left_behind_by_the_lane_grammar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.codex/` does not come back, and that is a limitation rather than a rule.

    `parse_relative_path` requires an alphanumeric first character so a path cannot
    address its lane's own prefix, so `.codex/history.jsonl` composes to no object key
    and there is nowhere to put it. The agent's own runtime state is the one thing a
    resumed Session would most want, and this pins the gap so that whoever builds the
    restore finds it asserted rather than discovering it from an amnesiac agent.

    Asserted alongside a file that IS kept, so a listing that broke outright could not
    pass this case by returning nothing.
    """
    root = tmp_path / "workspace"
    _write(root, ".codex/history.jsonl", b'{"role":"user"}\n')
    _write(root, "analysis.py", _SCRIPT)
    listing = await _listing(monkeypatch, root, new_session_id())
    assert [f.name for f in listing.files] == ["analysis.py"]


@pytest.mark.parametrize("reserved", list(NOT_SYNCED))
async def test_a_reserved_root_is_not_part_of_the_working_tree(
    reserved: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One case per member of `NOT_SYNCED`, driven off the constant itself.

    Parametrised over the tuple rather than over three literals so that a fourth member
    added later arrives with a case already written for it -- and so that a member
    quietly removed takes its case with it rather than leaving a passing test that
    guards nothing.
    """
    root = tmp_path / "workspace"
    _write(root, f"{reserved}/held-back.bin", b"not the working lane's business")
    _write(root, "kept.py", _SCRIPT)
    listing = await _listing(monkeypatch, root, new_session_id())
    assert [f.name for f in listing.files] == ["kept.py"]


async def test_a_directory_named_like_a_reserved_root_but_nested_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`NOT_SYNCED` names roots, and only roots.

    A project with its own `src/out/` build directory is the agent's own tree, and
    dropping it because of a name match one level down would lose work for a reason the
    agent could never guess. The exclusion is a prefix on the whole relative path, not
    a segment test anywhere in it.
    """
    root = tmp_path / "workspace"
    _write(root, f"src/{OUTPUT_DIR_NAME}/built.js", _SCRIPT)
    _write(root, f"nested/{INPUT_DIR_NAME}/notes.txt", b"the agent's own notes")
    listing = await _listing(monkeypatch, root, new_session_id())
    assert [f.name for f in listing.files] == [
        f"nested/{INPUT_DIR_NAME}/notes.txt",
        f"src/{OUTPUT_DIR_NAME}/built.js",
    ]


async def test_a_workspace_holding_nothing_lists_nothing_rather_than_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty and absent are both 200 with no files, as on the outputs listing.

    A Session whose Turn answered in text wrote nothing, and a first Turn may run before
    anything this process can see has created the root. Neither is a failure, and a sync
    that raised on either would fail exactly the Turns that did the least.
    """
    session_id = new_session_id()
    empty = tmp_path / "empty"
    empty.mkdir()
    for root in (empty, tmp_path / "never-made"):
        listing = await _listing(monkeypatch, root, session_id)
        assert listing.files == (), root


async def test_the_bytes_come_back_exactly_as_the_agent_wrote_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read route, at a nested path, with no shipping prefix in front of it.

    The working lane IS the workspace root, so the path the listing gave is the path the
    read takes. A prefix quietly added here -- the outputs route adds `out/` -- would
    read the wrong file or nothing at all.
    """
    root = tmp_path / "workspace"
    _write(root, "src/pipeline/load.py", _SCRIPT)
    session_id = new_session_id()
    app = _app(monkeypatch, root, session_id)
    async with _client(app) as client:
        answer = await client.get(
            workspace_path_for(session_id, "src/pipeline/load.py"),
            headers=_bearer(session_id),
        )
    assert answer.status_code == 200
    assert answer.content == _SCRIPT


async def test_a_path_that_was_listed_and_then_unlinked_answers_that_it_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """204, not 404 and not a 500.

    The agent keeps writing while the sync reads, so a path named by a listing a moment
    ago and removed since is ordinary rather than exceptional. The far end skips it; a
    failure here would let a `rm` in the agent's last second fail the whole Turn.
    """
    root = tmp_path / "workspace"
    root.mkdir()
    session_id = new_session_id()
    app = _app(monkeypatch, root, session_id)
    async with _client(app) as client:
        answer = await client.get(
            workspace_path_for(session_id, "never-written.py"),
            headers=_bearer(session_id),
        )
    assert answer.status_code == 204
    assert answer.content == b""


@pytest.mark.parametrize(
    "name",
    [
        "%2E%2E/escaped.py",
        "sub/%2E%2E/%2E%2E/escaped.py",
        f"{INPUT_DIR_NAME}/upload.csv",
        f"{OUTPUT_DIR_NAME}/report.md",
        ".map/lib/pandas/__init__.py",
    ],
)
async def test_a_path_outside_the_working_tree_reads_nothing(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read route refuses on the same rule the listing selects on.

    The traversal cases are percent-encoded because httpx resolves dot segments before
    it sends, per RFC 3986: spelled plainly they would be collapsed client-side and
    never reach the handler, and the case would pass without testing anything.

    The reserved-root cases matter separately from the listing's. A path the listing
    would never have offered must still be refused when asked for directly, because the
    caller is the control plane and the listing is not a capability.
    """
    root = tmp_path / "workspace"
    _write(root, f"{INPUT_DIR_NAME}/upload.csv", b"the tenant's own upload")
    _write(root, f"{OUTPUT_DIR_NAME}/report.md", b"the other lane's business")
    _write(root, ".map/lib/pandas/__init__.py", b"a rebuildable dependency")
    (tmp_path / "escaped.py").write_bytes(b"outside the read root entirely")
    session_id = new_session_id()
    app = _app(monkeypatch, root, session_id)
    async with _client(app) as client:
        answer = await client.get(
            f"/session/{session_id}/workspace/{name}", headers=_bearer(session_id)
        )
    assert answer.status_code == 400, answer.content
    assert b"escaped" not in answer.content
    assert b"upload" not in answer.content


async def test_a_symlink_the_agent_planted_is_neither_listed_nor_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The widened mount's own risk, asked again on the widest listing there is.

    This route walks more of the tree than any other, so it is the one most likely to
    meet a link the agent placed. A symlink resolves inside THIS container, whose mounts
    include the pod's bearer token -- readable here, not by the agent. Excluded at the
    walk and refused again at the open, because a check and an open are two moments.
    """
    root = tmp_path / "workspace"
    root.mkdir()
    secret = tmp_path / "token"
    secret.write_bytes(b"the pod's own bearer token")
    (root / "innocent.py").symlink_to(secret)
    (root / "src").mkdir()
    (root / "src" / "elsewhere").symlink_to(tmp_path)
    session_id = new_session_id()
    listing = await _listing(monkeypatch, root, session_id)
    assert listing.files == ()
    app = _app(monkeypatch, root, session_id)
    async with _client(app) as client:
        direct = await client.get(
            workspace_path_for(session_id, "innocent.py"),
            headers=_bearer(session_id),
        )
        through = await client.get(
            workspace_path_for(session_id, "src/elsewhere/token"),
            headers=_bearer(session_id),
        )
    assert direct.status_code == through.status_code == 204
    assert b"bearer token" not in direct.content + through.content


async def test_a_call_with_no_token_learns_neither_the_paths_nor_the_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """404 on both routes, with no path and no byte of any file in either body.

    Asserted on the content and not only the status, because this route hands back the
    agent's whole tree and a refusal that leaked a path list would be worse here than
    anywhere else on this process.
    """
    root = tmp_path / "workspace"
    _write(root, "analysis.py", _SCRIPT)
    session_id = new_session_id()
    app = _app(monkeypatch, root, session_id)
    async with _client(app) as client:
        listed = await client.get(_listing_path(session_id))
        served = await client.get(workspace_path_for(session_id, "analysis.py"))
    assert listed.status_code == served.status_code == 404
    assert b"analysis.py" not in listed.content
    assert _SCRIPT not in served.content


async def test_a_caller_naming_another_session_learns_the_same_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-identical to the no-token refusal, on both routes.

    Whole bodies rather than statuses: two refusals sharing a status but differing in
    message still tell the caller which condition it met, and which Session this pod is
    serving is exactly what both refusals exist to withhold.
    """
    served_session = new_session_id()
    other = new_session_id()
    root = tmp_path / "workspace"
    _write(root, "analysis.py", _SCRIPT)
    app = _app(monkeypatch, root, served_session)
    async with _client(app) as client:
        no_token_list = await client.get(_listing_path(other))
        wrong_list = await client.get(
            _listing_path(other), headers=_bearer(served_session)
        )
        no_token_get = await client.get(workspace_path_for(other, "analysis.py"))
        wrong_get = await client.get(
            workspace_path_for(other, "analysis.py"),
            headers=_bearer(served_session),
        )
    assert no_token_list.json() == wrong_list.json()
    assert no_token_get.json() == wrong_get.json()
    assert {r.status_code for r in (no_token_list, wrong_list)} == {404}
    assert {r.status_code for r in (no_token_get, wrong_get)} == {404}


async def test_a_listing_is_sorted_and_therefore_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory order is arbitrary; the diff downstream is not.

    The far end compares this listing against what the lane already holds. An order that
    varied between two Turns over an unchanged tree would still diff correctly -- but a
    listing truncated at the ceiling would take a different arbitrary subset each time,
    and a workspace over the limit would never converge.
    """
    root = tmp_path / "workspace"
    for relative in ("zeta.py", "alpha.py", "src/mid.py", "src/.cache/state.json"):
        _write(root, relative, _SCRIPT)
    listing = await _listing(monkeypatch, root, new_session_id())
    assert [f.name for f in listing.files] == [
        "alpha.py",
        "src/.cache/state.json",
        "src/mid.py",
        "zeta.py",
    ]
