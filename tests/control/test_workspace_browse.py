"""Browsing a Session's workspace on the mount: what is there, and one file's bytes.

**These cases are written against a real directory tree, not a double, and that is the
whole point of them.** The defences this route carries are file-system defences -- a
`..` segment, an absolute path, an empty segment, and a symlink pointing out of the
subtree -- and three of those four are only *lexical* until something resolves them
against a real root. A fake tree keyed by string would answer every one of these cases
correctly while the real `os` call underneath escaped, because a dict has no symlinks
and no `..` to follow. So `tmp_path` builds the tree, the route reads it with the same
calls it will use in the cluster, and the escape cases are real escapes.

The tree is composed as `<mount>/<tenant>/<session>/`, which is the layout the proven
mount has: the access point roots at the bucket's `workspaces/` prefix and a Session pod
is handed `<tenant>/<session>` as its `subPath`. So a file these cases seed sits where
an agent's own write would land, not at a path invented to suit the route.

**Each case mounts the router itself rather than building `create_app`.** That is the
division `test_every_router_is_mounted.py` states: a router's own tests isolate the
router, and whether `create_app` includes it is that file's question and not this one's.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.control.api import refusals
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.api.routes import workspace
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import SessionId, TenantId, new_session_id

_REPORT = b"# Report\n\nWhat the agent found so far.\n"
_SECRET = b"OPENAI_API_KEY=not-a-real-key\n"
_MODULE = b"def add(a, b):\n    return a + b\n"


def _an_app(root: Path | None) -> FastAPI:
    """The router alone, over a mount root the case chose.

    `install_request_envelope` is installed because one case below is refused by the
    router's tenancy dependency, which raises rather than returning -- without the
    handler that refusal would surface as an unhandled exception and the case would
    grade the wrong thing.
    """
    app = FastAPI()
    refusals.install_request_envelope(app)
    app.include_router(workspace.router, prefix="/v1")
    app.dependency_overrides[workspace.mounted_workspace_root] = lambda: root
    return app


def _workspace_of(root: Path, tenant_id: TenantId, session_id: SessionId) -> Path:
    """Where that Session's workspace sits under the mount, composed the platform's way.

    The two segments a Session pod is handed as its `subPath`, which is what makes a
    file these cases seed and a file the route reads the same file.
    """
    return root / str(tenant_id) / str(session_id)


def _a_workspace_holding(
    root: Path, tenant_id: TenantId, session_id: SessionId, files: dict[str, bytes]
) -> Path:
    """Write those relative paths under that Session's workspace, parents and all."""
    base = _workspace_of(root, tenant_id, session_id)
    for relative, body in files.items():
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _a_populated_workspace(
    root: Path, tenant_id: TenantId, session_id: SessionId
) -> Path:
    """The tree most cases below read: a deliverable, scratch, and the undeclared rest.

    `.env`, `.git/config` and `node_modules/` are here in the fixture rather than in one
    case of their own because they are what this route exists to reach -- ADR-038 says
    everything on disk is reachable -- and a fixture that held only well-named files
    would let a route that quietly filtered them pass every case in this file.
    """
    return _a_workspace_holding(
        root,
        tenant_id,
        session_id,
        {
            "report.md": _REPORT,
            ".env": _SECRET,
            ".git/config": b"[core]\n\trepositoryformatversion = 0\n",
            "node_modules/left-pad/index.js": b"module.exports = () => {}\n",
            "src/lib/util.py": _MODULE,
            "a..b.txt": b"a filename with two dots in it\n",
        },
    )


def _caller(app: FastAPI, tenant: TenantId | None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://s",
        headers={TENANT_HEADER: str(tenant)} if tenant is not None else {},
    )


def _telling(response: httpx.Response) -> tuple[object, object, object]:
    """The parts of a refusal that could distinguish one cause from another.

    The code, the message and any `reason`. Not `session_id`, which the caller supplied,
    and not `request_id`, which differs between any two requests.
    """
    error = dict(response.json()["error"])
    detail = error.get("detail") or {}
    return error["code"], error["message"], dict(detail).get("reason")


def _named(listing: httpx.Response) -> dict[str, str]:
    """Every entry in a listing, as name to kind."""
    return {e["name"]: e["kind"] for e in listing.json()["entries"]}


@pytest.fixture
def a_tenant_and_session() -> Iterator[tuple[TenantId, SessionId]]:
    yield TenantId(uuid4()), new_session_id()


async def test_a_listing_names_every_kind_of_thing_on_disk(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """The root of a workspace, with a file's size and a directory's absence of one.

    Sizes are asserted on the file and asserted absent on the directory, because a
    listing that reported a directory's `st_size` would be reporting the size of the
    directory entry itself -- a number with no meaning to a caller, which reads as the
    size of what is inside it.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        listing = await caller.get(f"/v1/sessions/{session_id}/workspace")

    assert listing.status_code == 200
    kinds = _named(listing)
    assert kinds["report.md"] == "file"
    assert kinds["src"] == "directory"
    by_name = {e["name"]: e for e in listing.json()["entries"]}
    assert by_name["report.md"]["byte_length"] == len(_REPORT)
    assert by_name["src"]["byte_length"] is None


async def test_a_listing_reaches_into_a_subdirectory(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """`path` names a directory below the root, and only that directory is listed.

    Both the intermediate and the leaf are asked for, because a route that ignored
    `path` and always listed the root would pass a single-directory case that happened
    to assert only on absence.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        one_down = await caller.get(
            f"/v1/sessions/{session_id}/workspace", params={"path": "src"}
        )
        two_down = await caller.get(
            f"/v1/sessions/{session_id}/workspace", params={"path": "src/lib"}
        )

    assert _named(one_down) == {"lib": "directory"}
    assert _named(two_down) == {"util.py": "file"}


async def test_everything_the_agent_left_behind_is_reachable(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """`.env`, `.git` and `node_modules` list and read. That is the route, not a leak.

    **This is the case `parse_relative_path` would fail.** That grammar requires a
    relative path's first character to be alphanumeric, so `.env` and `.git/config`
    compose to no lane key at all -- which is correct for a lane and wrong for a
    workspace, and is exactly why this route parses paths itself instead of borrowing.

    Listed *and* read, because a route could enumerate a dotfile and still refuse to
    serve it, and a caller who can see a file they cannot open is worse served than one
    who can see nothing.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        root = await caller.get(f"/v1/sessions/{session_id}/workspace")
        env = await caller.get(f"/v1/sessions/{session_id}/workspace/.env")
        git = await caller.get(f"/v1/sessions/{session_id}/workspace/.git/config")
        modules = await caller.get(
            f"/v1/sessions/{session_id}/workspace", params={"path": "node_modules"}
        )

    assert {".env", ".git", "node_modules"} <= set(_named(root))
    assert (env.status_code, env.content) == (200, _SECRET)
    assert git.status_code == 200
    assert _named(modules) == {"left-pad": "directory"}


async def test_a_filename_carrying_two_dots_is_not_mistaken_for_a_traversal(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """`a..b.txt` is a file, and `..` is a segment. This route tells them apart.

    `parse_relative_path` refuses `..` as a *substring*, and says why it can afford to:
    a lane path that spells `a..b` is one nothing has a reason to write. A workspace has
    no such licence -- ADR-038 puts every byte the agent left on disk behind this route,
    and a file the agent actually created must not be unreachable because of how it is
    spelled. So containment here is per-segment, and this is the case that proves the
    difference is real rather than incidental.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(f"/v1/sessions/{session_id}/workspace/a..b.txt")

    assert got.status_code == 200, _telling(got)
    assert b"two dots" in got.content


async def test_a_file_comes_back_byte_for_byte_and_cannot_render(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """The bytes unchanged, as an unsniffable attachment named by the last segment.

    The same claim `artifacts.py` makes and for the same reason, and it is stronger
    here: nothing in a workspace was ever declared as anything, so a file the agent
    happened to name `.html` is script that must not run on this platform's origin.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(f"/v1/sessions/{session_id}/workspace/src/lib/util.py")

    assert got.status_code == 200
    assert got.content == _MODULE
    assert got.headers["x-content-type-options"] == "nosniff"
    assert got.headers["content-type"] == "application/octet-stream"
    assert "attachment" in got.headers["content-disposition"]
    assert "util.py" in got.headers["content-disposition"]


async def test_another_tenant_naming_the_same_session_reaches_nothing(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """Cross-tenant isolation is the composed path's shape, so this grades the shape.

    The tenant segment comes from the REQUEST and the Session id from the PATH, so a
    caller who has learned another tenant's Session id addresses a directory under their
    own tenant segment -- which does not exist. There is no arrangement of inputs that
    reads across, because the tenant component is never taken from the caller's claim
    about which Session this is.

    Both operations are exercised. A route that composed the tenant into one path and
    not the other would pass whichever half this case omitted.
    """
    owner, stranger = TenantId(uuid4()), TenantId(uuid4())
    _, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, owner, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, stranger) as caller:
        listing = await caller.get(f"/v1/sessions/{session_id}/workspace")
        read = await caller.get(f"/v1/sessions/{session_id}/workspace/report.md")

    assert listing.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert read.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert _REPORT not in read.content


async def test_a_session_that_never_existed_answers_exactly_as_an_absent_path(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """Identical answers, deliberately: separating them makes this a Session-id probe.

    Compared body-to-body rather than status-to-status, so a later change that gave one
    of them a distinguishing `reason` fails here. `session_id` and `request_id` are
    dropped from the comparison -- the first is the caller's own input echoed back and
    the second differs between any two requests at all.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        absent_path = await caller.get(
            f"/v1/sessions/{session_id}/workspace/never-written.md"
        )
        absent_session = await caller.get(
            f"/v1/sessions/{new_session_id()}/workspace/report.md"
        )

    assert absent_path.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert absent_session.status_code == absent_path.status_code
    assert _telling(absent_session) == _telling(absent_path)


async def test_an_empty_directory_lists_empty_rather_than_refusing(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """A directory that exists and holds nothing is a 200 with no entries.

    The other half of the case above: "nothing is there" and "there is nothing here" are
    different facts, and collapsing them would tell a caller whose agent has not written
    yet that their Session does not exist.
    """
    tenant_id, session_id = a_tenant_and_session
    base = _a_populated_workspace(tmp_path, tenant_id, session_id)
    (base / "out").mkdir()
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        listing = await caller.get(
            f"/v1/sessions/{session_id}/workspace", params={"path": "out"}
        )

    assert listing.status_code == 200
    assert listing.json()["entries"] == []


@pytest.mark.parametrize(
    "path",
    [
        "%2E%2E%2F%2E%2E%2Fetc%2Fpasswd",
        "%2E%2E%2Fworking",
        "%2E%2E",
        "%2Fetc%2Fpasswd",
        "src%2F%2Futil.py",
        "src%2F.%2Futil.py",
        "src%2F%2E%2E%2F%2E%2E%2Freport.md",
    ],
)
async def test_a_read_path_that_could_leave_the_workspace_is_refused(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId], path: str
) -> None:
    """Every lexical escape, refused before anything touches the file system.

    **Percent-encoded, which is the realistic shape rather than an evasion.** An HTTP
    client resolves dot segments before it sends -- `httpx` does, per RFC 3986 -- so a
    literal `..` never survives to be dispatched at all; it is removed and the request
    addresses a shorter path, which is a different route. The encoded form is what
    arrives here, because Starlette percent-decodes a path parameter after routing, and
    it is also the only form an attacker could send.

    Each entry is its own case rather than one composite: `..` at the front, `..` in the
    middle, a bare `..`, an absolute path, an empty segment and a `.` segment fail for
    different reasons in a parser, and a single case would stop grading the rest the
    moment the first one was fixed.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(f"/v1/sessions/{session_id}/workspace/{path}")

    assert got.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID], (
        f"{path!r} came back {got.status_code}; a path this route cannot contain must "
        "be refused BY THIS ROUTE, and any other status means the request went "
        "somewhere else and this case is grading nothing"
    )
    assert got.json()["error"]["detail"]["reason"] == "workspace_path_invalid"
    assert b"root:" not in got.content


@pytest.mark.parametrize(
    "path",
    ["../..", "/etc", "src//lib", "src/./lib", "..", "with\x00nul", "x" * 300],
)
async def test_a_listing_path_that_could_leave_the_workspace_is_refused(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId], path: str
) -> None:
    """The same containment on the other operation, which takes its path differently.

    Written out rather than percent-encoded: this path arrives as a query parameter, and
    a client does not normalise dot segments in a query string the way it does in a
    path. So the literal form is what would actually reach this handler.

    **Both operations are parametrised because both compose a path**, and a defence
    written into one handler protects only that one. The over-long segment and the NUL
    are here rather than above for the same reason -- a NUL in a URL path is refused by
    the client before it is sent, so the query string is the only place this route can
    be asked to hold one.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(
            f"/v1/sessions/{session_id}/workspace", params={"path": path}
        )

    assert got.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID], (
        f"{path!r} came back {got.status_code}"
    )
    assert got.json()["error"]["detail"]["reason"] == "workspace_path_invalid"


async def test_a_symlink_pointing_out_of_the_mount_is_not_followed(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """The escape no lexical check can see, because the path that names it is clean.

    `getaway` is a well-formed single segment. Nothing about the request says anything
    is wrong; the escape is on disk, planted by whatever wrote the workspace. This is
    the case ADR-038 means when it says path safety here cannot be borrowed from the
    artifacts route -- an object key is a literal string and traverses nothing, while a
    file system resolves this one all the way out of the tree.

    Refused as *absent* rather than as invalid, and the body is compared against a
    genuinely absent path to hold that. Two reasons: the caller's path was not malformed
    so calling it malformed is a lie, and a distinguishable refusal would turn this
    route into a probe for what exists outside the subtree.
    """
    tenant_id, session_id = a_tenant_and_session
    base = _a_populated_workspace(tmp_path, tenant_id, session_id)
    outside = tmp_path / "outside-the-mount.txt"
    outside.write_bytes(b"bytes from beyond the mount root\n")
    (base / "getaway").symlink_to(outside)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        escape = await caller.get(f"/v1/sessions/{session_id}/workspace/getaway")
        absent = await caller.get(f"/v1/sessions/{session_id}/workspace/never-here")

    assert escape.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert b"beyond the mount" not in escape.content
    assert _telling(escape) == _telling(absent)


async def test_a_symlink_into_another_tenants_workspace_is_not_followed(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """The escape that stays inside the mount, which is the one that matters most.

    Every subtree on this file system is under one root, so a containment check written
    against the MOUNT root instead of against the SESSION's own subtree would resolve
    this symlink, find it comfortably inside the mount, and serve another tenant's file.
    The case above would pass against that bug; this one is what catches it.
    """
    tenant_id, session_id = a_tenant_and_session
    victim = TenantId(uuid4())
    _a_workspace_holding(tmp_path, victim, session_id, {"private.md": b"not yours\n"})
    base = _a_populated_workspace(tmp_path, tenant_id, session_id)
    (base / "theirs").symlink_to(
        _workspace_of(tmp_path, victim, session_id) / "private.md"
    )
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(f"/v1/sessions/{session_id}/workspace/theirs")

    assert got.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert b"not yours" not in got.content


async def test_a_symlink_inside_the_workspace_is_served(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """Containment refuses an escape, not a symlink. An agent's own link still reads.

    Without this the cheapest way to pass every case above is to refuse symlinks
    outright, which would put a file the agent deliberately linked out of reach and make
    this route stop being "everything on disk".
    """
    tenant_id, session_id = a_tenant_and_session
    base = _a_populated_workspace(tmp_path, tenant_id, session_id)
    (base / "latest.md").symlink_to(base / "report.md")
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        listing = await caller.get(f"/v1/sessions/{session_id}/workspace")
        got = await caller.get(f"/v1/sessions/{session_id}/workspace/latest.md")

    assert _named(listing)["latest.md"] == "symlink"
    assert (got.status_code, got.content) == (200, _REPORT)


async def test_a_listing_does_not_resolve_the_symlinks_it_reports(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """An escaping link is *named* by a listing and still not followed by it.

    Naming it is honest -- it is on disk, and ADR-038 puts what is on disk behind this
    route. Reporting it as `symlink` rather than as whatever it points at is what keeps
    the listing from becoming a second, quieter way to learn about the file it targets:
    a `kind` and a `byte_length` taken through the link would describe a file outside
    the subtree without ever serving it.
    """
    tenant_id, session_id = a_tenant_and_session
    base = _a_populated_workspace(tmp_path, tenant_id, session_id)
    outside = tmp_path / "outside-the-mount.txt"
    outside.write_bytes(b"x" * 4096)
    (base / "getaway").symlink_to(outside)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        listing = await caller.get(f"/v1/sessions/{session_id}/workspace")

    entries = {e["name"]: e for e in listing.json()["entries"]}
    assert entries["getaway"]["kind"] == "symlink"
    assert entries["getaway"]["byte_length"] != 4096


async def test_reading_a_directory_and_listing_a_file_are_told_apart(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """Each operation refuses the other's target, and says which mistake was made.

    A 404 for either would be wrong and actively misleading: the path exists, the caller
    asked the wrong door about it, and telling them nothing is there sends them looking
    for a file they can see in the listing they just read.
    """
    tenant_id, session_id = a_tenant_and_session
    _a_populated_workspace(tmp_path, tenant_id, session_id)
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        read_a_dir = await caller.get(f"/v1/sessions/{session_id}/workspace/src")
        list_a_file = await caller.get(
            f"/v1/sessions/{session_id}/workspace", params={"path": "report.md"}
        )

    assert read_a_dir.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert read_a_dir.json()["error"]["detail"]["reason"] == "workspace_path_not_a_file"
    assert list_a_file.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert (
        list_a_file.json()["error"]["detail"]["reason"]
        == "workspace_path_not_a_directory"
    )


async def test_a_deployment_with_no_mount_says_so_rather_than_saying_empty() -> None:
    """An unmounted workspace and an empty one are different facts.

    A 404 against a process with no mount would tell a tenant their agent's work is gone
    -- a lie they cannot detect and would act on by re-running a Turn that already
    succeeded. The reason travels in `detail` because the published code set is closed
    (ADR-013) and carries no member for a missing mount, which is the shape
    `artifacts.py` uses for its own unconfigured store.
    """
    tenant_id, session_id = TenantId(uuid4()), new_session_id()
    app = _an_app(None)

    async with _caller(app, tenant_id) as caller:
        listing = await caller.get(f"/v1/sessions/{session_id}/workspace")
        read = await caller.get(f"/v1/sessions/{session_id}/workspace/report.md")

    assert listing.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    assert listing.json()["error"]["detail"]["reason"] == "workspace_not_mounted"
    assert read.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    assert read.json()["error"]["detail"]["reason"] == "workspace_not_mounted"


async def test_a_caller_with_no_tenant_learns_nothing(tmp_path: Path) -> None:
    """The router-level gate, asserted because it is inherited rather than written here.

    The tenancy dependency is declared on the router so a route added to it later cannot
    forget one. That makes it exactly the kind of protection no route body mentions, so
    nothing else in this file would notice its removal.
    """
    app = _an_app(tmp_path)

    async with _caller(app, None) as caller:
        listing = await caller.get(f"/v1/sessions/{new_session_id()}/workspace")
        read = await caller.get(f"/v1/sessions/{new_session_id()}/workspace/report.md")

    assert listing.status_code == STATUS_FOR[ErrorCode.REQUEST_TENANT_MISSING]
    assert read.status_code == STATUS_FOR[ErrorCode.REQUEST_TENANT_MISSING]


def test_the_mount_root_is_read_from_the_environment_and_absent_by_default() -> None:
    """No mount configured means no mount, rather than a guessed path.

    A default -- `/mnt/session-vfs`, the current directory, anything -- is the failure
    that survives deployment: a control plane whose mount never came up would read some
    unrelated directory and report its contents as a tenant's workspace. Absent is the
    only safe answer, and the surface above turns it into a refusal that names it.
    """
    assert workspace.workspace_root_in({}) is None
    assert workspace.workspace_root_in(
        {workspace.WORKSPACE_MOUNT_ENV_VAR: "/mnt/session-vfs"}
    ) == Path("/mnt/session-vfs")


def test_a_relative_mount_root_is_refused_rather_than_resolved() -> None:
    """A mount is an absolute path or it is a misconfiguration.

    A relative root resolves against the serving process's working directory, which
    nothing in a deployment pins -- so the same configuration would read a different
    tree depending on where the process was started, and the containment check below it
    would be defending a subtree of somewhere nobody chose.

    Refused rather than quietly treated as no mount at all, which is the failure that
    would survive deployment: every request would then answer "your workspace is not
    mounted" while a mount was configured and merely misspelled, and the one person who
    could fix it would be told the opposite of what was wrong.
    """
    with pytest.raises(workspace.WorkspaceMountMisconfigured):
        workspace.workspace_root_in({workspace.WORKSPACE_MOUNT_ENV_VAR: "session-vfs"})


async def test_a_misconfigured_mount_refuses_without_publishing_the_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The misconfiguration reaches a caller as a refusal, and reaches nobody as detail.

    Driven through the real dependency rather than an override, because what is under
    test is exactly the part an override replaces: a value error raised inside a FastAPI
    dependency, which cannot return a response and would otherwise escape to be rendered
    by the framework in a body no handler here wrote.

    The offending value is asserted absent from the response. A deployment's own paths
    are not a tenant's business, and the closed code set exists so that a caller learns
    this platform is misconfigured without learning how.
    """
    monkeypatch.setenv(workspace.WORKSPACE_MOUNT_ENV_VAR, "relative-mount-path")
    app = FastAPI()
    refusals.install_request_envelope(app)
    app.include_router(workspace.router, prefix="/v1")
    tenant_id, session_id = TenantId(uuid4()), new_session_id()

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(f"/v1/sessions/{session_id}/workspace")

    assert got.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    assert got.json()["error"]["detail"]["reason"] == "workspace_mount_misconfigured"
    assert b"relative-mount-path" not in got.content


def test_a_workspace_path_is_parsed_into_segments_or_refused() -> None:
    """The containment parse alone, at the boundary the routes both call it from.

    Exercised directly as well as through the routes because it is the one function both
    operations depend on for containment: a case that reached it only through HTTP would
    stop grading it the day a handler grew a second path into the file system.
    """
    assert workspace.parse_workspace_path("") == ()
    assert workspace.parse_workspace_path("src/lib/util.py") == (
        "src",
        "lib",
        "util.py",
    )
    assert workspace.parse_workspace_path(".env") == (".env",)
    assert workspace.parse_workspace_path("a..b") == ("a..b",)
    for refused in ("..", "a/../b", "/abs", "a//b", "a/./b", "a/", "x" * 300, "a\x00b"):
        with pytest.raises(workspace.WorkspacePathInvalid):
            workspace.parse_workspace_path(refused)


async def test_a_file_the_process_cannot_open_refuses_rather_than_breaking(
    tmp_path: Path, a_tenant_and_session: tuple[TenantId, SessionId]
) -> None:
    """A workspace holds things that are not readable files, and this must not 500.

    A FIFO is the cheap reproduction of a real hazard: opening one blocks until a writer
    arrives, so a route that opened whatever the path named would hang a worker on a
    file an agent created for its own use. The rule is that this route serves regular
    files, and everything else is refused as not-a-file rather than attempted.
    """
    tenant_id, session_id = a_tenant_and_session
    base = _a_populated_workspace(tmp_path, tenant_id, session_id)
    os.mkfifo(base / "pipe")
    app = _an_app(tmp_path)

    async with _caller(app, tenant_id) as caller:
        listing = await caller.get(f"/v1/sessions/{session_id}/workspace")
        got = await caller.get(f"/v1/sessions/{session_id}/workspace/pipe")

    assert _named(listing)["pipe"] == "other"
    assert got.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert got.json()["error"]["detail"]["reason"] == "workspace_path_not_a_file"
