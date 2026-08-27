"""The read that carries a file the agent WROTE out of the pod.

Driven through `create_shim_app` against the real routes, with the read root redirected
onto a tmp_path. What is not redirected is the authorisation: the token is derived by
the same function the control plane derives it with, so a case that passes here would
pass against a real pod.

The refusals are the point of this file rather than the happy path. This is a *read*
route on the pod's outward-facing process, over a mount that covers the whole workspace,
so the questions worth asking are what it does with no token, with another Session's id,
with a name that climbs out of the tree, and -- the one that only exists because the
mount was widened -- with a symlink the agent planted pointing at a file this container
can read and the agent cannot.

`WORKSPACE_READ_ROOT` is patched on the module the routes read it from rather than
threaded through `ServedSession`, the way `test_shim_places_a_file.py` patches
`WORKSPACE_FILES`: that dataclass is the pod's process-wide identity and this is a
deployment constant everywhere but here.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.control.files.output_shipout import (
    OUTPUT_COUNT_LIMIT,
    OUTPUT_TREE_LIMIT,
)
from managed_agent.core.ids import SessionId, new_session_id
from managed_agent.core.pod.workspace_contract import OUTPUT_DIR_NAME
from managed_agent.session_shim.client import RuntimeConnection
from managed_agent.session_shim.pod_channel import shim_token_for
from managed_agent.session_shim.serve import (
    OUTPUTS_ROUTE,
    WORKSPACE_FILES,
    WORKSPACE_READ_ROOT,
    ProducedFiles,
    ServedSession,
    create_shim_app,
    output_path_for,
)

_THREAD = "0199c4de-6f2a-7b81-9c3d-4e5f60718293"
_KEY = b"a signing key for these cases only"
_REPORT = b"# Report\n\nWhat the agent found.\n"


def _app(
    monkeypatch: pytest.MonkeyPatch, produced: Path, session_id: SessionId
) -> FastAPI:
    monkeypatch.setattr(
        "managed_agent.session_shim.serve.WORKSPACE_READ_ROOT", produced
    )
    return create_shim_app(
        ServedSession(
            session_id=session_id,
            thread_id=_THREAD,
            connection=RuntimeConnection(produced / "never-dialled.sock"),
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
    return OUTPUTS_ROUTE.format(session_id=session_id)


def _output_path(session_id: SessionId, name: str) -> str:
    """The read route's path, built the way the control plane builds it.

    `output_path_for` rather than a `.format` of the template, because the template now
    carries FastAPI's `:path` converter and `str.format` reads `path` as a format spec
    and raises. Using the shipping builder rather than a spelling of its own is also
    what makes these cases exercise the encoding the real caller applies.
    """
    return output_path_for(session_id, name)


async def test_a_file_the_agent_left_at_its_root_is_listed_with_its_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing is the whole of how the control plane learns a document exists.

    The length is asserted as well as the name because it is what the transfer is later
    checked against: a listing that reported a name with the wrong size would be refused
    at the far end for a file that is perfectly intact.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )
    assert answer.status_code == 200
    listing = ProducedFiles.model_validate(answer.json())
    assert [(f.name, f.byte_length) for f in listing.files] == [
        ("report.md", len(_REPORT))
    ]


async def test_a_workspace_holding_nothing_lists_nothing_rather_than_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty list, at 200, and the same for a root that does not exist yet.

    Both are real: most Turns answer in text and write no file, and a pod on its first
    Turn may not have had its root created by anything this process can see. Neither is
    an error, and a 404 or a 500 for either would fail every text-only Turn.
    """
    session_id = new_session_id()
    empty = tmp_path / "empty"
    empty.mkdir()
    absent = tmp_path / "never-made"
    for root in (empty, absent):
        app = _app(monkeypatch, root, session_id)
        async with _client(app) as client:
            answer = await client.get(
                _listing_path(session_id), headers=_bearer(session_id)
            )
        assert answer.status_code == 200, root
        assert answer.json() == {"files": []}, root


async def test_the_attachment_directory_is_not_listed_as_something_the_agent_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tenant's own upload must not come back as an output.

    The read mount covers the whole workspace, so `files/` is visible from here -- and
    shipping what is inside it would hand a tenant a second copy of their own document
    under a new identifier they did not ask for and cannot match to the one they hold.
    The directory itself is excluded because nothing that is not a regular file is
    listed; this asserts that the consequence holds rather than that the rule exists.
    """
    produced = tmp_path / "produced"
    (produced / "files").mkdir(parents=True)
    (produced / "files" / "brief.md").write_bytes(b"the tenant sent this")
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )
        reached = await client.get(
            _output_path(session_id, "files"), headers=_bearer(session_id)
        )
    listing = ProducedFiles.model_validate(answer.json())
    assert [f.name for f in listing.files] == ["report.md"]
    assert reached.status_code == 204


@pytest.mark.parametrize(
    "name",
    [
        ".hidden",
        ".codex",
        "with\\a\\backslash",
        'with"a"quote',
        "with\ta\ttab",
    ],
)
async def test_a_name_the_platform_could_not_record_is_not_listed(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each of these is a name that has no home at the far end, for its own reason.

    Parametrized because each entry encodes a separate decision and a test naming two of
    them grades two. A leading dot is runtime scratch rather than a document, and it is
    also what the write route refuses so this pod's traffic obeys one rule in both
    directions. The backslash, the quote and the tab are refused by
    `parse_upload_filename` -- the first two because the name is written verbatim into a
    Content-Disposition header, the third because a control character truncates it for
    whatever reads it next.

    The over-255-byte rule that predicate also enforces is *not* here, and that is a
    finding rather than a gap: 255 bytes is also the filesystem's own limit for one path
    component, so no such file can exist to be listed. The rule is unreachable from this
    direction and a case for it would be measuring `write_bytes` raising ENAMETOOLONG.

    The assertion is that a *real* file under this name is skipped, not that the name is
    unrepresentable: every one of these is a name Linux will happily create, which is
    why the filter has to be here rather than assumed.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / name).write_bytes(b"scratch")
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )
    listing = ProducedFiles.model_validate(answer.json())
    assert [f.name for f in listing.files] == ["report.md"]


async def test_a_symlink_the_agent_planted_is_neither_listed_nor_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal that only exists because the read mount was widened.

    The agent writes this same directory from the other container, and this process
    holds mounts the agent does not -- above all `/etc/map/shim/token`, this Session's
    bearer token, mounted into the shim container and no other precisely so a confined
    agent cannot read it. A symlink at the workspace root pointing at that file would,
    on a route that followed symlinks, be shipped to the object store and handed to the
    tenant as a document their agent produced: a credential to their own pod's one write
    route, leaked by the feature meant to deliver their report.

    Both halves are asserted. The listing must not offer it -- that is
    `follow_symlinks=False` -- and the fetch must refuse it even when a caller names it
    directly, because the listing and the fetch are two moments and the link can be
    planted between them.
    """
    outside = tmp_path / "not-the-workspace"
    outside.mkdir()
    (outside / "token").write_bytes(b"a-secret-this-route-must-never-serve")
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    (produced / "innocent.txt").symlink_to(outside / "token")
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        listed = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )
        served = await client.get(
            _output_path(session_id, "innocent.txt"), headers=_bearer(session_id)
        )
    listing = ProducedFiles.model_validate(listed.json())
    assert [f.name for f in listing.files] == ["report.md"]
    assert served.status_code == 204
    assert served.content == b""


def test_the_enumeration_bound_is_well_above_what_one_turn_transfers() -> None:
    """**Two bounds, and the gap between them is what keeps the transfer bound honest.**

    They were one constant, and that is exactly what made the transfer bound accumulate
    over a Session's life: the pod stopped scanning at what one Turn ships, so a Session
    already holding that many never got the files it added this Turn into a listing at
    all, and the control plane's already-delivered filter cannot weigh a path it was
    never shown.

    Asserted as a strict inequality with room, not as two literals. A later reader
    tuning either number is free to; a reader who tunes them back to equality has undone
    the fix, and this is the only thing in the tree that would notice.
    """
    assert OUTPUT_TREE_LIMIT > OUTPUT_COUNT_LIMIT * 2, (
        "the pod must enumerate far more than one Turn transfers, or the transfer "
        "bound is a bound on the Session's whole lifetime again"
    )


async def test_a_listing_stops_at_one_past_the_enumeration_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound, exceeded on purpose so the cap is measured and not described.

    One entry past the limit rather than exactly the limit, and that extra entry is the
    whole signal: the far end has to be able to tell "the tree holds exactly what I will
    enumerate" from "it holds more than that", and a listing truncated at the limit is
    the same bytes in both cases.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    for index in range(OUTPUT_TREE_LIMIT + 5):
        (produced / f"file-{index:04d}.md").write_bytes(b"x")
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )
    listing = ProducedFiles.model_validate(answer.json())
    assert len(listing.files) == OUTPUT_TREE_LIMIT + 1


async def test_a_tree_larger_than_one_turn_ships_is_listed_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case the old coupling could not express, at the size where it bit.

    A Session that has delivered more files than one Turn transfers is the ordinary
    shape of a long run, and the file it wrote this Turn has to be in the listing for
    the control plane to have any chance of shipping it. Under the old bound the
    listing stopped one entry past what a Turn ships, so it came back holding an
    arbitrary handful of a much larger tree and `late.md` was usually not among them.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    for index in range(OUTPUT_COUNT_LIMIT * 3):
        (produced / f"file-{index:03d}.md").write_bytes(b"x")
    (produced / "late.md").write_bytes(b"written this turn")
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )

    listing = ProducedFiles.model_validate(answer.json())

    assert len(listing.files) == OUTPUT_COUNT_LIMIT * 3 + 1
    assert "late.md" in {entry.name for entry in listing.files}


async def test_a_listing_inside_the_limit_is_sorted_and_therefore_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two reads of one unchanged workspace are the same listing.

    Created in an order that is not the sorted one, because `scandir` yields directory
    order and a test that made them alphabetically would pass without any sort at all.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    for name in ("zeta.md", "alpha.md", "middle.md"):
        (produced / name).write_bytes(b"x")
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )
    listing = ProducedFiles.model_validate(answer.json())
    assert [f.name for f in listing.files] == ["alpha.md", "middle.md", "zeta.md"]


async def test_the_bytes_come_back_exactly_as_the_agent_wrote_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte for byte, over a body large enough to cross the streaming chunk size.

    The route streams in 64 KiB chunks, so a body under that never exercises the loop
    and a truncation after the first chunk would pass a small-file test.
    """
    body = bytes(range(256)) * 700
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "big.bin").write_bytes(body)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _output_path(session_id, "big.bin"), headers=_bearer(session_id)
        )
    assert answer.status_code == 200
    assert answer.content == body
    assert "content-length" not in answer.headers


async def test_a_name_that_was_listed_and_then_unlinked_answers_that_it_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """204 rather than a refusal, because there is no document in that to lose.

    An agent that tidies up after its own Turn produces exactly this: the listing was
    true when it was taken and the file is not there now.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _output_path(session_id, "vanished.md"), headers=_bearer(session_id)
        )
    assert answer.status_code == 204


async def test_a_call_with_no_token_learns_neither_the_names_nor_the_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """404 on both routes, and nothing of the workspace in either body.

    The status alone would pass for a route that refused after reading, and this is a
    read route: what has to be true is that the answer carries no file name and no byte
    of any file.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        listed = await client.get(_listing_path(session_id))
        served = await client.get(_output_path(session_id, "report.md"))
    assert listed.status_code == served.status_code == 404
    assert b"report.md" not in listed.content
    assert _REPORT not in served.content


async def test_a_caller_naming_another_session_learns_the_same_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-identical to the no-token refusal, on both routes.

    Compared as whole bodies rather than as statuses: two refusals with one status and
    two different messages still disclose which of the two conditions was met, and which
    Session a pod is serving is the fact both refusals exist to withhold.
    """
    served_session = new_session_id()
    other = new_session_id()
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    app = _app(monkeypatch, produced, served_session)
    async with _client(app) as client:
        no_token_list = await client.get(_listing_path(other))
        wrong_list = await client.get(
            _listing_path(other), headers=_bearer(served_session)
        )
        no_token_get = await client.get(_output_path(other, "report.md"))
        wrong_get = await client.get(
            _output_path(other, "report.md"), headers=_bearer(served_session)
        )
    assert no_token_list.json() == wrong_list.json()
    assert no_token_get.json() == wrong_get.json()
    assert {r.status_code for r in (no_token_list, wrong_list)} == {404}
    assert {r.status_code for r in (no_token_get, wrong_get)} == {404}


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.md",
        "..%2Fescaped.md",
        "sub/report.md",
        "..",
        ".hidden.md",
    ],
)
async def test_a_name_that_is_not_a_bare_leaf_reads_nothing(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every one of these would read somewhere the listing never offered.

    Parametrized because each entry is its own decision. The traversals matter more here
    than on the write route: that one is narrowed by a `subPath` mount so a climb has
    nowhere to arrive, and this one reads through a mount of the whole workspace.

    Which layer refuses which was measured rather than assumed, exactly as on the write
    route: only `.hidden.md` reaches the handler's own check, and the four holding a
    separator or a dot-dot component never get there -- Starlette does not match a `/`
    inside one path segment, does not decode `%2F` into one, and normalises `..` out of
    the path. What this asserts is that no body carries a file that exists beside them.

    A bare `.` is not in this list because it is not refused at all; it normalises onto
    the listing route, and the case below records that.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    (produced / "escaped.md").write_bytes(b"should not be reachable by climbing")
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            f"/session/{session_id}/outputs/{name}", headers=_bearer(session_id)
        )
    assert answer.status_code in (204, 400, 404, 405, 422), answer.status_code
    assert _REPORT not in answer.content
    assert b"climbing" not in answer.content


async def test_a_bare_dot_reaches_the_listing_rather_than_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured, and recorded here because it is the one dot name nothing refuses.

    Starlette normalises `/outputs/.` to `/outputs/`, which matches the listing route --
    so this is not a traversal that got through, it is a caller reaching a route it was
    already entitled to call, with the same token and the same two checks ahead of it.
    Nothing about the workspace is disclosed that `GET /outputs` does not disclose.

    Asserted rather than left to be discovered because a reader of the refusal list
    above would otherwise reasonably assume every dot name is refused, and be wrong.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        dotted = await client.get(
            f"/session/{session_id}/outputs/.", headers=_bearer(session_id)
        )
        listed = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )
    assert dotted.status_code == listed.status_code == 200
    assert dotted.json() == listed.json()
    assert _REPORT not in dotted.content


def test_the_read_root_is_not_the_directory_this_process_can_write() -> None:
    """The two workspace mounts this container holds are two different paths.

    That is the whole of how the widened read stays a read: `WORKSPACE_FILES` is the
    read-write `subPath` mount and `WORKSPACE_READ_ROOT` is the read-only whole-volume
    one, and if they were the same path the manifest would be granting write over
    everything the agent produced through a route whose only check is a bearer token.
    """
    assert WORKSPACE_READ_ROOT != WORKSPACE_FILES
    assert WORKSPACE_FILES not in WORKSPACE_READ_ROOT.parents


def test_the_listing_the_route_emits_is_the_shape_the_control_plane_parses(
    tmp_path: Path,
) -> None:
    """The wire model round-trips, so the far end's parse is not a second grammar.

    `extra="forbid"` is asserted here rather than left implied: it is what makes a pod
    padding its listing with fields the platform does not read a refusal at the far end
    instead of something ignored.
    """
    one = ProducedFiles.model_validate(
        json.loads(ProducedFiles.model_dump_json(ProducedFiles()))
    )
    assert one.files == ()
    with pytest.raises(ValueError):
        ProducedFiles.model_validate({"files": [], "extra": 1})


# --------------------------------------------------------------------------------------
# out/ is where the deliverables are, and the root is what happens when it is not used
# --------------------------------------------------------------------------------------


async def test_only_what_is_in_the_output_directory_is_listed_when_it_holds_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The case this was added for, and it is measured rather than invented.**

    A live run asked an agent for a PDF. It wrote `make_pdf.py` at the workspace root,
    ran it, and the PDF landed there too -- so ship-out returned BOTH, and the tenant
    got a generator script nobody asked for, counted against their file and byte
    budgets. The platform now tells the agent where deliverables go
    (`core/pod/workspace_contract.py`), and this is the half that honours it.

    The scratch file is deliberately the same shape as the real one -- a regular file,
    at the root, with a name ship-out would happily record. Nothing about the file
    itself says which is the document; the directory the agent chose is the only
    signal, which is exactly why the directory is the rule.
    """
    produced = tmp_path / "produced"
    (produced / OUTPUT_DIR_NAME).mkdir(parents=True)
    (produced / "make_pdf.py").write_bytes(b"# scratch the agent wrote\n")
    (produced / OUTPUT_DIR_NAME / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )

    assert answer.status_code == 200
    listing = ProducedFiles.model_validate(answer.json())
    assert [f.name for f in listing.files] == ["report.md"]


async def test_the_bytes_of_a_deliverable_come_from_the_directory_it_was_listed_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing and the read must agree about which directory they mean.

    They are two calls, so they could disagree, and the way they would is the nastiest
    kind: a name listed out of `out/` and opened at the root answers 204, which the far
    end reads as a document that vanished mid-Turn. Worse, a root file of the SAME name
    would be served instead -- the tenant would receive real bytes from the wrong file.
    That is what this asserts against, which is why both directories hold `report.md`.
    """
    produced = tmp_path / "produced"
    (produced / OUTPUT_DIR_NAME).mkdir(parents=True)
    (produced / "report.md").write_bytes(b"the scratch copy, not the deliverable\n")
    (produced / OUTPUT_DIR_NAME / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _output_path(session_id, "report.md"), headers=_bearer(session_id)
        )

    assert answer.status_code == 200
    assert answer.content == _REPORT


async def test_a_workspace_with_no_output_directory_still_ships_what_it_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The fallback, and it is not caution.**

    An agent that ignored the convention -- an older prompt, a model that forgot, a Turn
    where the platform's instruction lost to the tenant's own -- has still done the
    work the tenant paid for. Losing it is the worse failure by a wide margin: untidy
    beats gone. So the previous rule stays as the answer when `out/` is not.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )

    assert answer.status_code == 200
    listing = ProducedFiles.model_validate(answer.json())
    assert [f.name for f in listing.files] == ["report.md"]


async def test_an_output_directory_holding_nothing_falls_back_rather_than_shipping_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty `out/` is what a model that made it and then wrote elsewhere leaves.

    Treating its existence as the answer would ship nothing from a Turn that produced
    something -- the exact failure the fallback exists to prevent, reached through the
    convention rather than despite it. So "holds a shippable file" is the test, not
    "exists".
    """
    produced = tmp_path / "produced"
    (produced / OUTPUT_DIR_NAME).mkdir(parents=True)
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )

    assert answer.status_code == 200
    listing = ProducedFiles.model_validate(answer.json())
    assert [f.name for f in listing.files] == ["report.md"]


async def test_an_output_directory_holding_only_unshippable_names_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dotfile in `out/` is not a deliverable, so `out/` is not the answer.

    Without this the directory could be "occupied" by a `.DS_Store` or an editor's swap
    file and the real document at the root would go unshipped -- a Turn losing its work
    to a file nobody wrote on purpose.
    """
    produced = tmp_path / "produced"
    (produced / OUTPUT_DIR_NAME).mkdir(parents=True)
    (produced / OUTPUT_DIR_NAME / ".hidden").write_bytes(b"not a document\n")
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )

    assert answer.status_code == 200
    listing = ProducedFiles.model_validate(answer.json())
    assert [f.name for f in listing.files] == ["report.md"]


async def test_a_file_named_out_rather_than_a_directory_does_not_break_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent can create a FILE called `out`, and a scandir on it raises.

    Uncaught that is a 500 on the route the control plane uses to find a Turn's
    output, so a Turn that produced a document would fail over the name of an
    unrelated file. That file is still a regular file at the root, so it ships too.
    """
    produced = tmp_path / "produced"
    produced.mkdir()
    (produced / OUTPUT_DIR_NAME).write_bytes(b"a file, not a directory\n")
    (produced / "report.md").write_bytes(_REPORT)
    session_id = new_session_id()
    app = _app(monkeypatch, produced, session_id)
    async with _client(app) as client:
        answer = await client.get(
            _listing_path(session_id), headers=_bearer(session_id)
        )

    assert answer.status_code == 200
    listing = ProducedFiles.model_validate(answer.json())
    assert sorted(f.name for f in listing.files) == [OUTPUT_DIR_NAME, "report.md"]
