"""The working-lane read routes: a pod fetching back its own earlier workspace.

Driven through the real ASGI stack with the real `SessionVfsStore` between the routes
and a fake bucket, so the object key is composed by the shipping code on both sides --
the seeding write and the serving read. A fake at the lane instead of at the bucket
would let the two sides agree on a key neither S3 nor the Turn-boundary sync would ever
produce.

**The isolation case is the one to read first.** ADR-030's whole argument for exposing
this surface to a confined agent is that the token decides whose lane is read and
nothing in the request does. The case named
`test_a_second_sessions_token_reaches_nothing_of_the_firsts` seeds one Session's object
and asks for that exact path with another Session's token -- and asserts in the same
test that the owner does get the bytes, because a 404 for the stranger proves nothing
if the path was wrong for everybody.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Final, cast
from uuid import UUID, uuid4

import httpx2
import pytest
from fastapi import FastAPI

from managed_agent.adapters.s3.session_vfs import SessionVfsStore
from managed_agent.control.files.workspace_sync import WORKING_COUNT_LIMIT
from managed_agent.core.ids import Seq, SessionId, TenantId
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER,
    mint_session_token,
)
from managed_agent.core.vfs.session_vfs import (
    WORKING,
    LaneEntry,
    MutableFile,
    SessionFiles,
)
from managed_agent.gateway.tool.mcp_proxy import ToolEventTypes
from managed_agent.gateway.tool.server import GatewaySessions, create_gateway_app
from managed_agent.gateway.tool.working_lane import LANE_LISTING_PATH

KEY: Final[bytes] = b"a signing key that is thirty-two"

TENANT: Final[TenantId] = TenantId(UUID("11111111-1111-4111-8111-111111111111"))
OTHER_TENANT: Final[TenantId] = TenantId(UUID("22222222-2222-4222-8222-222222222222"))

ABSENT_BODY: Final[dict[str, str]] = {
    "error": "no object at that path in this session's working lane"
}
"""The refusal the working-lane route itself writes.

Asserted instead of the status alone, and the distinction is not fussiness: Starlette
answers a path it could not route with 404 too. A request that never reached the route
satisfies every assertion made about a 404 status, which is exactly how the traversal
case in this file spent one commit proving nothing.
"""


class FakeLaneBlobs:
    """One dict of objects, keyed exactly as the real bucket keys them.

    A fake at the *bucket*, so `SessionVfsStore` above it is the shipping code and the
    key it composes is the real one. `asked` is what makes the isolation case provable:
    it records every key the routes actually reached for, so a test can assert the
    stranger's request never even named the owner's object rather than only that the
    response was a refusal.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.asked: list[str] = []
        self.listed: list[str] = []

    async def put_new(self, key: str, body: bytes) -> None:
        raise AssertionError("the working lane creates through replace, not place")

    async def put(self, key: str, body: bytes) -> None:
        self.objects[key] = body

    async def get(self, key: str) -> bytes | None:
        self.asked.append(key)
        return self.objects.get(key)

    async def list_prefix(self, prefix: str) -> Sequence[LaneEntry]:
        self.listed.append(prefix)
        return [
            LaneEntry(relative=key[len(prefix) :], byte_length=len(body))
            for key, body in self.objects.items()
            if key.startswith(prefix)
        ]


class SilentLog:
    """An Event Log the seeding writes append to and nothing here reads back."""

    def __init__(self) -> None:
        self.appended: list[tuple[str, dict[str, object]]] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        self.appended.append((type_, payload))
        return Seq(len(self.appended))


def _no_upstreams() -> GatewaySessions:
    """A Session registry with nothing behind it; no case here reaches the MCP path."""
    return GatewaySessions(
        scopes=cast(Any, None),
        registry=cast(Any, None),
        broker=cast(Any, None),
        append=cast(Any, None),
        events=cast(Any, None),
        types_=ToolEventTypes(
            progress="p", elicitation_requested="q", elicitation_answered="r"
        ),
        evidence=cast(Any, None),
    )


def _app(files: SessionFiles) -> FastAPI:
    return create_gateway_app(_no_upstreams(), KEY, files, _NoRollouts())


class _NoRollouts:
    """A rollout store holding nothing, for cases that are not about resuming.

    Answers None rather than raising, because that is the honest answer for the
    Sessions these cases drive: none of them has completed a Turn, so none has a
    stored Rollout. A raising stand-in would make every case here assert something
    about a store it is not testing.
    """

    async def restore_for_resume(self, session_id: SessionId) -> None:
        return None


def _token(session_id: SessionId, tenant_id: TenantId = TENANT) -> str:
    return mint_session_token(
        session_id=session_id,
        tenant_id=tenant_id,
        expiry_epoch_s=int(time.time()) + 300,
        key=KEY,
    )


def _client(app: FastAPI, token: str | None) -> httpx2.AsyncClient:
    headers = {} if token is None else {SESSION_TOKEN_HEADER.decode(): token}
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app),
        base_url="http://gateway",
        headers=headers,
    )


async def _seed(
    store: SessionVfsStore,
    session_id: SessionId,
    relative: str,
    body: bytes,
    tenant_id: TenantId = TENANT,
) -> None:
    """Put one object in a Session's working lane the way the Turn sync does."""
    await store.replace(
        MutableFile(
            tenant_id=tenant_id,
            session_id=session_id,
            lane=WORKING,
            relative=relative,
        ),
        body,
    )


def _object_path(relative: str) -> str:
    return f"{LANE_LISTING_PATH}/{relative}"


async def test_the_listing_names_every_object_the_calling_sessions_lane_holds() -> None:
    """What a restore has to fetch, with the size it will have to account for."""
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    session = SessionId(uuid4())
    await _seed(store, session, "analysis.py", b"print('hello')\n")
    await _seed(store, session, "out/report/fig1.png", b"\x89PNG" + b"\x00" * 60)

    async with _client(_app(store), _token(session)) as http:
        response = await http.get(LANE_LISTING_PATH)

    assert response.status_code == 200
    # Sorted in the assertion rather than in the route: what order a lane lists in is
    # the store's, and pinning one here would assert a property of this fake's dict.
    assert sorted(response.json()["objects"], key=lambda o: str(o["path"])) == [
        {"path": "analysis.py", "size": 15},
        {"path": "out/report/fig1.png", "size": 64},
    ]


async def test_an_object_comes_back_byte_for_byte_at_its_lane_relative_path() -> None:
    """A workspace arrives with its shape intact, separators and all."""
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    session = SessionId(uuid4())
    body = b"\x00\x01\x02 not text \xff\xfe"
    await _seed(store, session, "src/deep/nested/main.py", body)

    async with _client(_app(store), _token(session)) as http:
        response = await http.get(_object_path("src/deep/nested/main.py"))

    assert response.status_code == 200
    assert response.content == body
    assert response.headers["content-type"] == "application/octet-stream"


async def test_a_second_sessions_token_reaches_nothing_of_the_firsts() -> None:
    """The token decides whose lane is read, and no part of the request does.

    Both halves matter. The stranger is refused, and the owner is served that same path
    in the same test -- a refusal on a path nobody could read would pass while proving
    nothing. The recorded key is asserted too, because a 404 could equally come from a
    route that read the owner's object and then declined to return it; what makes this
    an isolation property is that the composed key never named it.

    **What this test grades is the composed key, which is the proxy and not the
    property.** A cross-session read is not expressible through `SessionFiles`: its
    only read takes a file that already carries the caller's identity, so there is no
    mutation of this route that hands a stranger the owner's bytes. The impossibility
    is carried by the type rather than by anything asserted here, and the strongest
    honest claim this file can make is the one below -- that the key a stranger's
    request composed was built from the stranger's own token.

    A companion line asserting `str(owner) not in blobs.asked[0]` was removed rather
    than kept. The assertion below is exact equality over the whole key, so any change
    to it fails there first and the companion could never be the line that failed --
    implied by its neighbour, and a line that cannot fail is a comment wearing an
    assert.
    """
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    owner = SessionId(uuid4())
    stranger = SessionId(uuid4())
    await _seed(store, owner, "secrets/notes.md", b"the first session's work")
    app = _app(store)

    async with _client(app, _token(stranger)) as http:
        refused = await http.get(_object_path("secrets/notes.md"))
        empty = await http.get(LANE_LISTING_PATH)
    async with _client(app, _token(owner)) as http:
        served = await http.get(_object_path("secrets/notes.md"))

    assert (refused.status_code, refused.json()) == (404, ABSENT_BODY)
    assert empty.json() == {"objects": []}
    assert served.status_code == 200
    assert served.content == b"the first session's work"
    assert blobs.asked[0] == f"sessions/{TENANT}/{stranger}/working/secrets/notes.md"


async def test_a_token_for_another_tenant_reaches_nothing_of_this_ones() -> None:
    """The tenant comes from the token too, so the same Session id is not a way in."""
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    session = SessionId(uuid4())
    await _seed(store, session, "analysis.py", b"print('hello')\n")

    async with _client(_app(store), _token(session, OTHER_TENANT)) as http:
        refused = await http.get(_object_path("analysis.py"))
        listing = await http.get(LANE_LISTING_PATH)

    assert refused.status_code == 404
    assert listing.json() == {"objects": []}
    assert blobs.asked == [f"sessions/{OTHER_TENANT}/{session}/working/analysis.py"]


@pytest.mark.parametrize("path", [LANE_LISTING_PATH, "/v1/session/working-lane/a.py"])
async def test_an_unsigned_request_is_refused_before_the_lane_is_touched(
    path: str,
) -> None:
    """Both routes sit behind the same check the MCP path does, and it runs first."""
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())

    async with _client(_app(store), None) as http:
        response = await http.get(path)

    assert response.status_code == 401
    assert (blobs.asked, blobs.listed) == ([], [])


async def test_a_path_the_lane_cannot_hold_is_absent_not_parsed_loosely() -> None:
    """A path that composes to no key in this lane is answered as what it is: nothing.

    The parser is not widened to serve this route, so `..` is refused here exactly as it
    is refused at the write, and the refusal happens before the bucket is touched.

    **The traversal is sent percent-encoded, and that is the whole reason this case
    means anything.** httpx resolves dot-segments in a URL before it sends one, so a
    literal `../../etc/passwd` leaves this process as `/v1/etc/passwd`: it matches no
    route, Starlette answers its own 404, and an assertion on the status alone passes
    without the parser ever running. Measured, not supposed -- that is what the first
    version of this test did. Encoding the separators keeps the segments from being
    removed client-side; the ASGI layer then decodes them, so what the route receives is
    the traversal a hostile client meant to send.

    The double-encoded case is the other half: it arrives still carrying a literal `%`,
    which `_RELATIVE_PATTERN` refuses because `%` is not in its allowed character set.
    So one decode does not become two, and a second pass at the parser is not needed.
    """
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    session = SessionId(uuid4())
    await _seed(store, session, "analysis.py", b"print('hello')\n")

    async with _client(_app(store), _token(session)) as http:
        traversal = await http.get(_object_path("..%2F..%2Fetc/passwd"))
        dot_encoded = await http.get(_object_path("%2E%2E%2F%2E%2E%2Fetc/passwd"))
        double_encoded = await http.get(_object_path("%252E%252E%252Fetc/passwd"))
        empty_segment = await http.get(_object_path("out//fig1.png"))
        dotfile = await http.get(_object_path(".ssh/id_rsa"))

    refusals = (traversal, dot_encoded, double_encoded, empty_segment, dotfile)
    # The body and not only the status: a 404 carrying `Not Found` is Starlette saying
    # it could not route the request, which is a different fact and used to pass here.
    every_one_refused = [(404, ABSENT_BODY)] * len(refusals)
    assert [(r.status_code, r.json()) for r in refusals] == every_one_refused
    assert blobs.asked == []


async def test_an_encoded_separator_names_the_same_object_as_a_raw_one() -> None:
    """`src%2Fmain.py` and `src/main.py` are one path by the time the route sees it.

    Pinned rather than left to chance, because the decode happens above this route --
    the ASGI server unquotes the target before Starlette matches on it -- and the whole
    safety of the path handling rests on that ordering: decode first, then parse. If a
    server ever stopped decoding, an encoded separator would arrive as a literal `%`,
    which the parser refuses outright rather than treating as a separator. Either
    ordering is safe; what would not be safe is a third one where the parse ran first
    and the decode after it, and this case is what would notice.
    """
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    session = SessionId(uuid4())
    await _seed(store, session, "src/deep/main.py", b"the same bytes either way")

    async with _client(_app(store), _token(session)) as http:
        raw = await http.get(_object_path("src/deep/main.py"))
        encoded = await http.get(_object_path("src%2Fdeep%2Fmain.py"))

    assert (raw.status_code, encoded.status_code) == (200, 200)
    assert raw.content == encoded.content == b"the same bytes either way"
    assert blobs.asked == [
        f"sessions/{TENANT}/{session}/working/src/deep/main.py",
        f"sessions/{TENANT}/{session}/working/src/deep/main.py",
    ]


async def test_a_write_verb_is_refused_by_the_route_table_itself() -> None:
    """This surface reads a lane and has no door onto writing one.

    The lane is seeded first and its bytes asserted unchanged afterwards, because an
    empty bucket compared against an empty bucket is an assertion that cannot fail.
    Nothing the route does can write to this fake, so `objects == {}` held before the
    request was made and would have held whatever the route did with it -- which is a
    line that reads like a guarantee and carries none.
    """
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    session = SessionId(uuid4())
    await _seed(store, session, "analysis.py", b"print('hello')\n")

    async with _client(_app(store), _token(session)) as http:
        posted = await http.post(_object_path("analysis.py"), content=b"overwrite me")
        deleted = await http.delete(_object_path("analysis.py"))

    assert (posted.status_code, deleted.status_code) == (405, 405)
    assert blobs.objects == {
        f"sessions/{TENANT}/{session}/working/analysis.py": b"print('hello')\n"
    }


async def test_an_empty_lane_is_reported_at_warning_rather_than_in_silence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lane with nothing in it is an answer, and the deployment has to hear it.

    A restore that materializes an empty workspace and a Session that never synced one
    are indistinguishable downstream, and `{"objects": []}` is a perfectly well-formed
    response either way. The level is part of the assertion because this deployment's
    root logger sits at WARNING with no handler -- a line emitted at INFO exists in the
    source and nowhere else, and from a review it looks identical to one that works.
    """
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    session = SessionId(uuid4())

    with caplog.at_level(logging.DEBUG, logger="managed_agent.gateway.tool"):
        async with _client(_app(store), _token(session)) as http:
            response = await http.get(LANE_LISTING_PATH)

    assert response.json() == {"objects": []}
    assert [(r.levelname, str(session) in r.getMessage()) for r in caplog.records] == [
        ("WARNING", True)
    ]


async def test_a_lane_past_the_restore_ceiling_is_reported_whole() -> None:
    """The ceiling belongs to the client, and this route enforces no second one.

    One object more than `WORKING_COUNT_LIMIT` is listed in full. A route that clipped
    the listing here would leave the init container fetching a tree it believed was
    complete, which is the exact failure ADR-030 refuses a partial restore to avoid.
    """
    blobs = FakeLaneBlobs()
    store = SessionVfsStore(blobs, SilentLog())
    session = SessionId(uuid4())
    over = WORKING_COUNT_LIMIT + 1
    for index in range(over):
        await _seed(store, session, f"f{index}.txt", b"x")

    async with _client(_app(store), _token(session)) as http:
        response = await http.get(LANE_LISTING_PATH)

    assert len(response.json()["objects"]) == over
