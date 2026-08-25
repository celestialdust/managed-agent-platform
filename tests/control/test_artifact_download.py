"""Downloading what an agent produced, at the path it produced it under.

**This route is the half of the artifacts change that a tenant actually touches.** The
ship-out cases in `test_output_shipout.py` prove a produced file reaches a Session's
`artifacts` lane; nothing there proves anybody can get it back. Before this route
existed, a produced file was minted as an upload and fetched from
`GET /v1/files/{id}/content` -- and that door cannot serve `report/fig1.png`, because an
upload is keyed by a filename and a filename holds no separator. So the nested case
below is not an extra: it is the case the whole change exists for.

The store under these cases is the real `SessionVfsStore` over a fake bucket, so the key
the route composes and the key ship-out writes are composed by the same code. A double
at the lane would let the two drift into agreeing about a method name while disagreeing
about a key, which is exactly the failure a tenant would experience as "my document is
not there".
"""

from dataclasses import replace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from test_file_upload_download import FakeStorage, a_platform, client_for
from test_output_shipout import FakeLaneBlobs, FakeLog

from managed_agent.control.api.app import create_app
from managed_agent.control.files.store import FileStore, UploadSizeLimit
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import SessionId, TenantId, new_session_id
from managed_agent.core.vfs.session_vfs import (
    ARTIFACTS,
    SealedFile,
    SessionFiles,
)

from managed_agent.adapters.s3.session_vfs import (  # isort: skip
    SessionVfsStore,
    UnconfiguredSessionVfs,
)

_REPORT = b"# Report\n\nWhat the agent found.\n"
_FIGURE = b"\x89PNG\r\n\x1a\n and then some bytes"


def _an_app(artifacts: SessionFiles) -> FastAPI:
    """The real app factory, with the artifact lane swapped into a real `Platform`.

    `dataclasses.replace` over the roster `test_file_upload_download` already builds,
    rather than a tenth stand-in written out here: `Platform` has no default for its
    first nine fields, and a second copy of that roster is a second thing to update the
    day a tenth port is added.

    `create_app` and not a hand-built FastAPI, for the reason that file gives: this is
    what makes these cases notice if the `include_router` line for this router is
    dropped, which every case below would otherwise survive by mounting it itself.
    """
    base = a_platform(FileStore(FakeStorage(), UploadSizeLimit(1024)))
    return create_app(replace(base, session_artifacts=artifacts))


async def _a_lane_holding(
    tenant_id: TenantId, session_id: SessionId, objects: dict[str, bytes]
) -> SessionVfsStore:
    """A lane store whose bucket already holds those paths, placed the real way.

    Placed through `place` rather than poked into the bucket dict, so the keys these
    cases read back are keys the shipping path actually writes. A test that seeded the
    dict itself would pass just as happily against a route composing a key nothing else
    in the platform composes.
    """
    store = SessionVfsStore(FakeLaneBlobs(), FakeLog())
    for relative, body in objects.items():
        await store.place(
            SealedFile(
                tenant_id=tenant_id,
                session_id=session_id,
                lane=ARTIFACTS,
                relative=relative,
            ),
            body,
        )
    return store


def _caller(app: FastAPI, tenant: UUID) -> httpx.AsyncClient:
    return client_for(app, tenant)


def _telling(response: httpx.Response) -> tuple[object, object, object]:
    """The parts of a refusal that could distinguish one cause from another.

    The code, the message and any `reason`. Not `session_id`, which the caller supplied,
    and not `request_id`, which differs between any two requests.
    """
    error = dict(response.json()["error"])
    detail = error.get("detail") or {}
    return error["code"], error["message"], dict(detail).get("reason")


async def test_an_artifact_comes_back_exactly_as_the_agent_wrote_it() -> None:
    """The plain case: one file, one path, the bytes unchanged.

    `content-disposition` and `nosniff` are asserted here rather than in a case of their
    own because they are the same claim: nothing declared a media type -- an agent
    writes bytes, not a `Content-Type` -- so a file the model happened to name `.html`
    must not render on this platform's origin.
    """
    tenant_id, session_id = TenantId(uuid4()), new_session_id()
    app = _an_app(await _a_lane_holding(tenant_id, session_id, {"report.md": _REPORT}))

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(f"/v1/sessions/{session_id}/artifacts/report.md")

    assert got.status_code == 200
    assert got.content == _REPORT
    assert got.headers["x-content-type-options"] == "nosniff"
    assert "attachment" in got.headers["content-disposition"]
    assert got.headers["content-type"] == "application/octet-stream"


async def test_a_deliverable_one_directory_down_is_reachable() -> None:
    """The acceptance case for the whole change.

    A flat file passing proves nothing this route added -- the old file-id door served
    those. What it could not serve is a path with a separator in it, so this is the case
    that says a report with its figures beside it survives the round trip whole.

    Both files are fetched, not just the nested one, because the failure this guards
    against is a route that matches the first segment and drops the rest: fetching only
    `report/fig1.png` would pass against a route that served `report/index.md` for every
    path under `report/`.
    """
    tenant_id, session_id = TenantId(uuid4()), new_session_id()
    app = _an_app(
        await _a_lane_holding(
            tenant_id,
            session_id,
            {"report/index.md": _REPORT, "report/fig1.png": _FIGURE},
        )
    )

    async with _caller(app, tenant_id) as caller:
        page = await caller.get(f"/v1/sessions/{session_id}/artifacts/report/index.md")
        figure = await caller.get(
            f"/v1/sessions/{session_id}/artifacts/report/fig1.png"
        )

    assert (page.status_code, page.content) == (200, _REPORT)
    assert (figure.status_code, figure.content) == (200, _FIGURE)
    assert "fig1.png" in figure.headers["content-disposition"]


async def test_another_tenant_naming_the_same_session_gets_nothing() -> None:
    """Cross-tenant isolation here is the key's shape, so this grades the shape.

    The route composes `sessions/<tenant>/<session>/artifacts/<path>` from the tenant on
    the REQUEST and the Session id in the PATH, so a caller who has somehow learned
    another tenant's Session id composes a key under their own tenant segment -- which
    holds nothing. There is no arrangement of inputs that reads across, because the
    tenant component is never taken from the caller's claim about which Session this is.

    Without this case a route that dropped the tenant from the key would pass every
    other case in this file, since every one of them uses the tenant that owns the
    Session.
    """
    owner, stranger = TenantId(uuid4()), TenantId(uuid4())
    session_id = new_session_id()
    app = _an_app(await _a_lane_holding(owner, session_id, {"report.md": _REPORT}))

    async with _caller(app, stranger) as caller:
        got = await caller.get(f"/v1/sessions/{session_id}/artifacts/report.md")

    assert got.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert _REPORT not in got.content


async def test_a_session_that_never_existed_answers_exactly_as_an_absent_path() -> None:
    """Identical answers, deliberately: separating them makes this a Session-id probe.

    Asserted by comparing the two refusals rather than by asserting each is a 404, so a
    later change adding a distinguishing `reason` or message to one of them fails here.
    That is the whole risk: the two bodies are easy to make differ by accident, and the
    difference is invisible unless something compares them.

    The comparison drops `session_id` and `request_id`. Neither can leak anything: the
    session id is the caller's own input echoed back, and the request id is minted per
    request and differs between any two calls at all. Comparing whole bodies would fail
    on those two every time and would have to be relaxed into meaninglessness.
    """
    tenant_id, session_id = TenantId(uuid4()), new_session_id()
    app = _an_app(await _a_lane_holding(tenant_id, session_id, {"report.md": _REPORT}))

    async with _caller(app, tenant_id) as caller:
        absent_path = await caller.get(
            f"/v1/sessions/{session_id}/artifacts/never-written.md"
        )
        absent_session = await caller.get(
            f"/v1/sessions/{new_session_id()}/artifacts/report.md"
        )

    assert absent_path.status_code == STATUS_FOR[ErrorCode.FILE_NOT_FOUND]
    assert absent_session.status_code == absent_path.status_code
    assert _telling(absent_session) == _telling(absent_path)


@pytest.mark.parametrize(
    "path",
    ["%2E%2E%2F%2E%2E%2Fetc%2Fpasswd", "%2E%2E", "a%2F%2Fb.md", "trailing%2F"],
)
async def test_a_path_that_composes_to_no_key_is_refused_as_malformed(
    path: str,
) -> None:
    """A malformed path and an absent one are different facts, and are told apart.

    `SealedFile` refuses each of these at construction, which is the check that matters
    -- but its refusal is a `ValueError` reaching a route, and an uncaught one is a 500
    telling the caller this platform broke on their request. Each entry is its own case:
    the traversals would address a key outside the Session's prefix, and the empty and
    trailing segments compose to a key naming a prefix rather than an object.

    **Percent-encoded, and that is the realistic shape rather than an evasion of the
    test.** An HTTP client resolves dot segments before it sends -- `httpx` does, per
    RFC 3986 -- so a literal `..` in a URL never survives to be dispatched anywhere; it
    is removed and the request addresses a shorter path, which is a different route
    entirely. What DOES arrive here is the encoded form, because Starlette
    percent-decodes a path parameter after routing. So this is the only way a `..` can
    reach this handler, and it is also the only way an attacker would send one.
    """
    tenant_id, session_id = TenantId(uuid4()), new_session_id()
    app = _an_app(await _a_lane_holding(tenant_id, session_id, {"report.md": _REPORT}))

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(f"/v1/sessions/{session_id}/artifacts/{path}")

    assert got.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID], (
        f"{path!r} came back {got.status_code}; a path this route cannot compose a key "
        "from must be refused BY THIS ROUTE, and any other status means the request "
        "went somewhere else and this case is grading nothing"
    )
    assert got.json()["error"]["detail"]["reason"] == "artifact_path_invalid"
    assert _REPORT not in got.content


async def test_a_deployment_with_no_bucket_says_so_rather_than_saying_empty() -> None:
    """An unwired store and a lane that holds nothing are different facts.

    A read that answered 404 against an unwired store would tell a tenant their document
    was never produced, which is a lie they cannot detect and would act on. The refusal
    names the reason in `detail`, because the published code set is closed and has no
    member for an unconfigured object store.
    """
    tenant_id = TenantId(uuid4())
    app = _an_app(UnconfiguredSessionVfs())

    async with _caller(app, tenant_id) as caller:
        got = await caller.get(f"/v1/sessions/{new_session_id()}/artifacts/report.md")

    assert got.status_code == STATUS_FOR[ErrorCode.INTERNAL]
    assert got.json()["error"]["detail"]["reason"] == "artifact_store_unconfigured"


async def test_a_caller_with_no_tenant_learns_nothing() -> None:
    """The router-level gate, asserted because it is inherited rather than written here.

    `artifacts.router` declares the tenancy dependency on the router so a route added to
    it later cannot forget one. That makes it exactly the kind of protection no route
    body mentions, so nothing in this file would notice its removal without this case.
    """
    app = _an_app(UnconfiguredSessionVfs())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://s"
    ) as caller:
        got = await caller.get(f"/v1/sessions/{new_session_id()}/artifacts/report.md")

    assert got.status_code == STATUS_FOR[ErrorCode.REQUEST_TENANT_MISSING]
