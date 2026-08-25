"""The init container that puts a Session's `working` lane back into its pod.

Driven over a real httpx client against a real ASGI app, rather than over a stubbed
client object. The point is not fidelity for its own sake: the module's whole job is to
tell a 200 from a 404 from a 401, to read a body's length, and to compose a path onto a
route -- and a hand-written double agrees with whatever the code does on every one of
those. `tests/shim/fake_agent_runtime.py` cost this repository a whole slice by being
kinder than the thing it stood in for, so the fake below is written to REFUSE: a wrong
token is 401, an object the lane does not hold is 404, and a route nobody pinned is 404.

The contract it serves is the one ADR-030 pins and the Tool Gateway implements:

    GET /v1/session/working-lane        -> {"objects": [{"path": ..., "size": ...}]}
    GET /v1/session/working-lane/{path} -> the object body, or 404

Tenant and Session are the token's to say. Neither route takes them, so neither does
anything here.
"""

from __future__ import annotations

import asyncio
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from managed_agent.control.files.workspace_sync import (
    WORKING_BUDGET_BYTES,
    WORKING_COUNT_LIMIT,
)
from managed_agent.control.pod_config.compiler import compile_session_config
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.registration.definition import AgentDefinition, SkillsRevision
from managed_agent.core.registration.environment import Environment, new_environment_id
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.session.session_token import SESSION_TOKEN_HEADER_NAME
from managed_agent.session_shim import restore_working_lane
from managed_agent.session_shim.restore_working_lane import (
    _CONCURRENT_FETCHES,
    LANE_ROUTE,
    GatewayBinding,
    RestoreRefused,
    RestoreReport,
    client_for,
    main,
    parse_listing,
    read_binding,
    restore,
)

A_TOKEN: Final = "a-session.a-tenant.4102444800.deadbeef"
"""Stands in for a minted token. Never signed, because nothing here verifies one --
what this file cares about is that the value is presented and never printed."""

SESSION_TOKEN_KEY: Final[bytes] = b"a signing key that is thirty-two"
SESSION_TOKEN_EXPIRY: Final[int] = 4102444800

SETTLE_PASSES: Final = 200
"""Event-loop passes a held request waits out before it counts itself finished.

Passes rather than seconds, so this costs no wall clock and does not vary with the
machine. Comfortably more than the awaits one request crosses between the client and the
route -- the number only has to be large enough that every request a client is willing
to open at once has arrived before the first one leaves.
"""

A_DEFINITION: Final = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)


def compiled_config_toml(*, tool_gateway_url: str) -> str:
    """A real compiled `config.toml`, from the compiler that writes the real one.

    Hand-written TOML would prove nothing about the document this actually reads: the
    table name, the header key and the URL field are all the compiler's choices, and a
    fixture that restates them agrees with itself while the pod disagrees.
    """
    return compile_session_config(
        SessionRecord(
            id=new_session_id(),
            tenant_id=TenantId(uuid4()),
            definition_id=new_definition_id(),
            definition_revision="rev-1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=10_000,
            budget_currency="USD",
            retention_days=30,
        ),
        tool_gateway_url=tool_gateway_url,
        model_gateway_url="http://model-gateway.map-dev.svc.cluster.local/v1",
        definition=A_DEFINITION,
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="fixture",
            runtime_image="registry.map.internal/session@sha256:" + "a" * 64,
            denied_paths=(),
        ),
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    ).config_toml


def token_in(config_toml: str) -> str:
    """The token the compiler put in that document, read back as the pod reads it."""
    servers: Any = tomllib.loads(config_toml)["mcp_servers"]
    headers: Any = next(iter(servers.values()))["http_headers"]
    value: str = headers[SESSION_TOKEN_HEADER_NAME]
    return value


class FakeGateway:
    """The two working-lane routes, and a refusal for everything else.

    `objects` is what the lane holds; `listing` is what the listing route says it holds.
    They are separate on purpose -- a listing promising a length the object does not
    have is one of the failures this module exists to catch, and a fake that derives one
    from the other could not express it.

    `objects_requested` is the record the tests assert work was NOT done against. A
    claim that a ceiling was enforced "before anything was fetched" cannot be graded on
    the workspace, because a refusal leaves the same empty tree either way.

    `hold_until` makes the object route block until that many requests are inside it at
    once, which is the only way concurrency is observable from out here: a serial client
    never reaches the number, so it waits for a release that will not come.
    """

    def __init__(
        self,
        objects: Mapping[str, bytes],
        *,
        token: str = A_TOKEN,
        listing: object | None = None,
        hold_until: int | None = None,
    ) -> None:
        self._objects = dict(objects)
        self._token = token
        self._hold_until = hold_until
        self._released = asyncio.Event()
        self._in_flight = 0
        self.most_in_flight = 0
        self._listing = (
            listing
            if listing is not None
            else {
                "objects": [
                    {"path": path, "size": len(body)}
                    for path, body in self._objects.items()
                ]
            }
        )
        self.objects_requested: list[str] = []
        self.listings_requested = 0

    def client(self, *, token: str = A_TOKEN) -> httpx.AsyncClient:
        """An httpx client onto this app, headed the way the pod heads its own."""
        app = Starlette(
            routes=[
                Route(LANE_ROUTE, self._list),
                Route(f"{LANE_ROUTE}/{{path:path}}", self._object),
            ]
        )
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://tool-gateway.map.test",
            headers={SESSION_TOKEN_HEADER_NAME: token},
        )

    async def _list(self, request: Request) -> Response:
        refused = self._refuse_untokened(request)
        if refused is not None:
            return refused
        self.listings_requested += 1
        return JSONResponse(self._listing)

    async def _object(self, request: Request) -> Response:
        refused = self._refuse_untokened(request)
        if refused is not None:
            return refused
        path = request.path_params["path"]
        self.objects_requested.append(path)
        await self._hold()
        body = self._objects.get(path)
        if body is None:
            return Response(status_code=404)
        return Response(body, media_type="application/octet-stream")

    async def _hold(self) -> None:
        """Stay inside this handler until `hold_until` callers are here together, then
        stay a little longer so that everyone else who can get in has.

        No wall-clock sleep anywhere. A sleep long enough to overlap is a test that
        fails on a loaded machine and a sleep short enough to be quick is a test that
        passes on a serial client, which is two ways to write a flake. The gate waits
        on a condition only concurrency can satisfy; the settle loop below then yields a
        fixed number of event-loop passes, which is what makes "the most that were ever
        inside at once" a number rather than a race -- without it, a client with no
        bound at all measured the same 16 as one bounded to 16, because the extra
        requests arrived after the first had already left.
        """
        if self._hold_until is None:
            return
        self._in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self._in_flight)
        if self._in_flight >= self._hold_until:
            self._released.set()
        await self._released.wait()
        for _ in range(SETTLE_PASSES):
            await asyncio.sleep(0)
            self.most_in_flight = max(self.most_in_flight, self._in_flight)
        self._in_flight -= 1

    def _refuse_untokened(self, request: Request) -> Response | None:
        """The Gateway's own check, mirrored so this fake is not the kinder one."""
        if request.headers.get(SESSION_TOKEN_HEADER_NAME) != self._token:
            return Response(status_code=401)
        return None


class Emitted:
    """Every line the module said, kept so a test can read what a pod's log would."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        self.lines.append(line)

    def joined(self) -> str:
        return "\n".join(self.lines)


async def test_a_lane_with_nothing_in_it_restores_nothing_and_says_so(
    tmp_path: Path,
) -> None:
    """The common path: a Session's first placement. It must be fast, and it must not
    be silent -- an empty listing and a listing route that has quietly stopped finding
    anything look identical to everything downstream."""
    gateway = FakeGateway({})
    said = Emitted()

    async with gateway.client() as client:
        report = await restore(client, tmp_path, report=said)

    assert report.objects_restored == 0
    assert report.bytes_restored == 0
    assert list(tmp_path.iterdir()) == []
    assert gateway.objects_requested == []
    assert "holds no objects" in said.joined()
    assert "0 object(s), 0 byte(s)" in said.joined()


async def test_every_object_the_lane_lists_lands_under_the_workspace_root(
    tmp_path: Path,
) -> None:
    """Nested paths and all, because a working tree is a project rather than a folder,
    and the directories above a restored file are this container's to create."""
    lane = {
        "notes.md": b"# what I found\n",
        "src/analysis/run.py": b"print('hello')\n",
        "data/rows.csv": b"a,b\n1,2\n",
    }
    gateway = FakeGateway(lane)
    said = Emitted()

    async with gateway.client() as client:
        report = await restore(client, tmp_path, report=said)

    for path, body in lane.items():
        assert (tmp_path / path).read_bytes() == body
    assert report.objects_restored == 3
    assert report.bytes_restored == sum(len(body) for body in lane.values())
    assert sorted(gateway.objects_requested) == sorted(lane)
    assert "restored 3 object(s)" in said.joined()


async def test_a_nested_path_goes_on_the_wire_with_its_separators_literal(
    tmp_path: Path,
) -> None:
    """Asserted on the REQUEST TARGET, because every other assertion in this file reads
    the path after something has already decoded it.

    Measured, which is why this exists: percent-encoding the separators in `_fetch`
    (`quote(obj.relative, safe="")`, so `a/b` becomes `a%2Fb`) left all 44 other cases
    green. httpx's `ASGITransport` unquotes the target into `scope["path"]` before
    Starlette routes on it, so the fake receives `a/b` either way and the file lands
    either way -- the encoding is invisible everywhere downstream of the wire.

    That is not a defect in the module today and it would not break this deployment,
    where the pod dials the Gateway's Service directly and uvicorn decodes the same way.
    What it is, is a property two slices agreed on in review and neither suite could
    see. The lane grammar admits only `[A-Za-z0-9_./-]`, none of which needs encoding,
    so the honest expectation is that the target carries the path verbatim -- and an
    expectation nothing reads is the thing this repository keeps paying for.
    """
    sent: list[str] = []

    async def record(request: httpx.Request) -> None:
        sent.append(request.url.raw_path.decode("ascii"))

    gateway = FakeGateway({"src/analysis/run.py": b"print('hello')\n"})
    said = Emitted()

    async with gateway.client() as client:
        client.event_hooks["request"].append(record)
        await restore(client, tmp_path, report=said)

    assert f"{LANE_ROUTE}/src/analysis/run.py" in sent, sent
    assert not any("%2F" in target or "%2f" in target for target in sent), sent


OVERLAP: Final = 2
"""Two objects fetched at once: the smallest thing that is not a serial fetch.

A literal, and deliberately not `_CONCURRENT_FETCHES`. Written against that constant,
the test below moves with it -- dropping the module to one in flight also drops what the
fake waits for, the gate opens on the first request, and the whole thing stays green
while the restore has become the ~41 s serial fetch ADR-030 exists to avoid. Measured:
that mutation produced 36 passed. Whether the constant is BIG ENOUGH is a separate
question with a separate guard, in
`tests/deploy/test_the_pod_restores_its_working_lane`, where the ceiling is priced
against the readiness budget.
"""


async def test_the_fetches_really_do_overlap(tmp_path: Path) -> None:
    """The half of ADR-030's fix that lives in the code rather than in the readiness
    bound. 2048 objects fetched one at a time is ~41 s of a budget the image pull and
    two probes have already spent, so a serial restore is a pod deleted mid-start.

    The gate opens only once two requests are inside the object route together, so a
    serial client waits on a release that will never come and the bounded wait below
    ends the test in seconds rather than hanging it.
    """
    lane = {f"f{n}.txt": b"x" for n in range(OVERLAP * 4)}
    gateway = FakeGateway(lane, hold_until=OVERLAP)
    said = Emitted()

    async with gateway.client() as client:
        report = await asyncio.wait_for(restore(client, tmp_path, report=said), 5.0)

    assert gateway.most_in_flight >= OVERLAP
    assert report.objects_restored == len(lane)


async def test_no_more_than_the_declared_number_are_ever_in_flight(
    tmp_path: Path,
) -> None:
    """The other direction, and it needs its own case: overlapping at all and
    overlapping without bound are different behaviours, and the second is what puts a
    burst of 2048 simultaneous requests on a Gateway shared by every pod being placed.

    Relative to the constant on purpose -- what it grades is that the semaphore is
    honoured, whatever the number is -- with twice that many objects, so an unbounded
    fetch would be caught reaching the whole set.
    """
    lane = {f"f{n}.txt": b"x" for n in range(_CONCURRENT_FETCHES * 2)}
    gateway = FakeGateway(lane, hold_until=_CONCURRENT_FETCHES)
    said = Emitted()

    async with gateway.client() as client:
        report = await asyncio.wait_for(restore(client, tmp_path, report=said), 5.0)

    assert gateway.most_in_flight == _CONCURRENT_FETCHES
    assert report.objects_restored == len(lane)


async def test_an_object_the_gateway_will_not_serve_refuses_the_whole_restore(
    tmp_path: Path,
) -> None:
    """All or nothing. A pod started on a partial tree presents it as a complete one,
    and the agent then reports a file missing with nothing anywhere saying why."""
    gateway = FakeGateway(
        {"kept.txt": b"still here"},
        listing={
            "objects": [
                {"path": "kept.txt", "size": 10},
                {"path": "gone.txt", "size": 4},
            ]
        },
    )
    said = Emitted()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused) as refusal:
            await restore(client, tmp_path, report=said)

    assert "gone.txt" in str(refusal.value)
    assert "404" in str(refusal.value)


async def test_a_body_shorter_than_its_listing_refuses_rather_than_writing_it(
    tmp_path: Path,
) -> None:
    """The one failure that leaves a file looking like a whole one. Nothing downstream
    re-reads the length, so if this does not compare it, nothing ever does."""
    gateway = FakeGateway(
        {"report.md": b"half"},
        listing={"objects": [{"path": "report.md", "size": 400}]},
    )
    said = Emitted()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused) as refusal:
            await restore(client, tmp_path, report=said)

    assert "report.md" in str(refusal.value)
    assert "400" in str(refusal.value)


async def test_a_listing_over_the_object_ceiling_is_refused_before_any_fetch(
    tmp_path: Path,
) -> None:
    """Asserted on the fetches, not on the workspace. A refusal that spent the whole
    budget first and a refusal that spent nothing leave the same empty tree behind, so
    the tree cannot grade the claim this test is making."""
    over = WORKING_COUNT_LIMIT + 1
    gateway = FakeGateway(
        {},
        listing={"objects": [{"path": f"f{n}.txt", "size": 1} for n in range(over)]},
    )
    said = Emitted()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused) as refusal:
            await restore(client, tmp_path, report=said)

    assert str(over) in str(refusal.value)
    assert str(WORKING_COUNT_LIMIT) in str(refusal.value)
    assert gateway.objects_requested == []


async def test_a_listing_over_the_byte_ceiling_is_refused_before_any_fetch(
    tmp_path: Path,
) -> None:
    """The other ceiling, and the same reason for checking it here: the restore is a
    fourth claim on one 1Gi workspace `emptyDir`, and an `emptyDir` over its limit is
    enforced by evicting the pod rather than by refusing the write."""
    half = WORKING_BUDGET_BYTES // 2 + 1
    gateway = FakeGateway(
        {},
        listing={
            "objects": [
                {"path": "first.bin", "size": half},
                {"path": "second.bin", "size": half},
            ]
        },
    )
    said = Emitted()

    async with gateway.client() as client:
        with pytest.raises(RestoreRefused) as refusal:
            await restore(client, tmp_path, report=said)

    assert str(WORKING_BUDGET_BYTES) in str(refusal.value)
    assert gateway.objects_requested == []


@pytest.mark.parametrize(
    "escape",
    [
        "../outside",
        "/etc/passwd",
        "sub/../../outside",
        ".codex/config.toml",
        ".agents/agent.md",
        "trailing/",
        "double//slash",
        "",
    ],
)
def test_a_path_that_is_not_lane_relative_refuses_the_whole_listing(
    escape: str,
) -> None:
    """The lane's own grammar, not a second one written here. What the sync could write
    is exactly what this will restore -- which is also why a leading dot is refused, and
    why no restored path can collide with the `.codex` and `.agents` directories the
    init container before this one creates as sandbox targets."""
    with pytest.raises(RestoreRefused) as refusal:
        parse_listing({"objects": [{"path": escape, "size": 1}]})

    assert "lane-relative" in str(refusal.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"objects": {"a.txt": 1}},
        {"objects": ["a.txt"]},
        {"objects": [{"size": 1}]},
        {"objects": [{"path": "a.txt"}]},
        {"objects": [{"path": "a.txt", "size": "1"}]},
        {"objects": [{"path": "a.txt", "size": -1}]},
        {"objects": [{"path": "a.txt", "size": True}]},
    ],
)
def test_a_listing_this_cannot_read_refuses_rather_than_restoring_what_parsed(
    payload: object,
) -> None:
    """A listing half of which parses describes a tree this would restore half of, and
    half a tree is the outcome the all-or-nothing rule exists to prevent. `size: true`
    is in the list because `bool` is an `int` in Python: without an explicit refusal it
    parses as one byte."""
    with pytest.raises(RestoreRefused):
        parse_listing(payload)


def test_a_listing_at_the_ceiling_exactly_is_accepted() -> None:
    """The other side of the bound, so the refusal above is not passing because the
    parser refuses everything of that shape."""
    at = WORKING_COUNT_LIMIT
    parsed = parse_listing(
        {"objects": [{"path": f"f{n}.txt", "size": 0} for n in range(at)]}
    )

    assert len(parsed) == at


def test_the_binding_is_read_out_of_the_document_the_compiler_writes() -> None:
    """Both halves out of one `mcp_servers` entry, so a URL and a token cannot come
    from different Sessions. The base drops the MCP path because the lane routes are
    siblings of that endpoint rather than children of it."""
    url = "http://tool-gateway.map-dev.svc.cluster.local/mcp"
    document = compiled_config_toml(tool_gateway_url=url)

    binding = read_binding(document)

    assert binding == GatewayBinding(
        base_url="http://tool-gateway.map-dev.svc.cluster.local",
        token=token_in(document),
    )


@pytest.mark.parametrize(
    "authority",
    ["operator:hunter2@tool-gateway", "operator@tool-gateway", ":secret@tool-gateway"],
)
def test_a_gateway_url_that_embeds_credentials_is_refused(authority: str) -> None:
    """Refused rather than stripped, because of where the composed base ends up.

    An unreachable Gateway's refusal names the address it dialed, and that refusal is
    promoted into the pod's status by `terminationMessagePolicy: FallbackToLogsOnError`
    -- so a `user:pass@` surviving into the base would be a credential printed for every
    reader of the pod. Refusing here is what makes the address safe to print at all,
    which is why this test and the one naming the address are two halves of one
    property.
    """
    document = (
        f'[mcp_servers."map-tool-gateway"]\nurl = "http://{authority}/mcp"\n'
        'http_headers = { "x-map-session" = "a-token" }\n'
    )

    with pytest.raises(RestoreRefused) as refusal:
        read_binding(document)

    assert "credentials" in str(refusal.value)
    assert "hunter2" not in str(refusal.value)
    assert "secret" not in str(refusal.value)


async def test_a_gateway_that_never_answers_names_the_address_it_dialed(
    tmp_path: Path,
) -> None:
    """The likeliest live failure of this whole container is a connection that never
    opens: `deploy/k8s/network-policies.yaml` allows egress to the gateways on TCP 8080
    while both Services publish 80, and that file records that whether the translation
    holds is a live measurement nothing in the tree can settle. When it refuses, the
    host and port have to be in the message -- an operator reading a pod that would not
    start should not need a code read to learn what was dialed."""
    said = Emitted()
    binding = GatewayBinding(base_url="http://127.0.0.1:1", token=A_TOKEN)

    async with client_for(binding) as client:
        with pytest.raises(RestoreRefused) as refusal:
            await restore(client, tmp_path, report=said)

    assert "127.0.0.1:1" in str(refusal.value)
    assert A_TOKEN not in str(refusal.value)


async def test_the_client_presents_the_token_as_a_header_and_not_in_the_url() -> None:
    """The three decisions `client_for` makes, read back off the object the pod uses.

    Each of them fails the same way in the cluster and is invisible from inside this
    process: a header under the wrong name, a base that kept the `/mcp` path, or a
    token spliced into a URL all produce a 401 or a 404 on every request from every
    pod -- and the Gateway answers a token it cannot read with the same fixed 401 as a
    request that carried none. The last one is worse than a failure: a token in a URL
    is a token in every access log it passes through.

    The third is graded by the exact-equality check below rather than by a line of
    its own: a base URL pinned to a literal that holds no token cannot also hold
    one, so a separate "token not in the url" assertion could never fail while that
    one passed.
    """
    document = compiled_config_toml(
        tool_gateway_url="http://tool-gateway.map-dev.svc.cluster.local/mcp"
    )

    async with client_for(read_binding(document)) as client:
        assert client.headers[SESSION_TOKEN_HEADER_NAME] == token_in(document)
        assert str(client.base_url) == "http://tool-gateway.map-dev.svc.cluster.local"


def test_what_was_restored_is_written_where_a_reader_of_the_pod_can_see_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kubelet lifts this file into `terminated.message`, which is in the pod's own
    status rather than in a container log. The counts are there so the restore's claim
    on the workspace is legible beside the attachments' -- one 1Gi `emptyDir` carries
    both, and nothing today adds them up."""
    log = tmp_path / "termination-log"
    monkeypatch.setattr(restore_working_lane, "TERMINATION_LOG", log)

    restore_working_lane._record(RestoreReport(objects_restored=3, bytes_restored=4096))

    assert "3 object(s)" in log.read_text()
    assert "4096 byte(s)" in log.read_text()


def test_a_termination_log_that_cannot_be_written_says_so_and_does_not_fail_the_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The restore has already succeeded by this point, so failing the pod over a status
    line would be the wrong trade. Saying nothing would be the other wrong one: the
    absence of the line would read as a restore that never ran."""
    monkeypatch.setattr(
        restore_working_lane, "TERMINATION_LOG", tmp_path / "no" / "such" / "dir" / "f"
    )

    restore_working_lane._record(RestoreReport(objects_restored=1, bytes_restored=2))

    assert "could not be written" in capsys.readouterr().err


@pytest.mark.parametrize(
    "document",
    [
        "",
        'mcp_servers = "not a table"',
        '[mcp_servers.other]\nurl = "http://x/mcp"\n',
        '[mcp_servers."map-tool-gateway"]\nrequired = true\n',
        '[mcp_servers."map-tool-gateway"]\nurl = "http://x/mcp"\n',
        '[mcp_servers."map-tool-gateway"]\nurl = "file:///x"\n'
        'http_headers = { "x-map-session" = "t" }\n',
        '[mcp_servers."map-tool-gateway"]\nurl = "http:///mcp"\n'
        'http_headers = { "x-map-session" = "t" }\n',
    ],
)
def test_a_configuration_that_names_no_reachable_gateway_refuses(document: str) -> None:
    """Each of these would otherwise surface as an httpx error naming a URL and no
    cause, or -- worse -- as a request sent with no token, which the Gateway answers
    with the same fixed 401 as a token it could not read."""
    with pytest.raises(RestoreRefused):
        read_binding(document)


async def test_no_error_path_prints_or_raises_this_session_s_token(
    tmp_path: Path,
) -> None:
    """Load-bearing rather than tidy. The container declares
    `terminationMessagePolicy: FallbackToLogsOnError`, so every line an error path
    writes is promoted into the pod's status, where more readers see it than ever run
    `kubectl logs`. The happy path is checked too, but the error paths are the ones that
    get promoted -- so they are the ones enumerated here."""
    said = Emitted()
    refusals: list[str] = []

    unauthorised = FakeGateway({"a.txt": b"x"}, token="a-different-token")
    async with unauthorised.client(token=A_TOKEN) as client:
        with pytest.raises(RestoreRefused) as refusal:
            await restore(client, tmp_path, report=said)
    refusals.append(str(refusal.value))

    absent = FakeGateway({}, listing={"objects": [{"path": "a.txt", "size": 1}]})
    async with absent.client() as client:
        with pytest.raises(RestoreRefused) as refusal:
            await restore(client, tmp_path, report=said)
    refusals.append(str(refusal.value))

    over = FakeGateway(
        {},
        listing={
            "objects": [
                {"path": f"f{n}.txt", "size": 1} for n in range(WORKING_COUNT_LIMIT + 1)
            ]
        },
    )
    async with over.client() as client:
        with pytest.raises(RestoreRefused) as refusal:
            await restore(client, tmp_path, report=said)
    refusals.append(str(refusal.value))

    happy = FakeGateway({"a.txt": b"x"})
    async with happy.client() as client:
        await restore(client, tmp_path, report=said)

    everything = said.joined() + "\n" + "\n".join(refusals)
    assert A_TOKEN not in everything
    assert SESSION_TOKEN_HEADER_NAME not in everything


def test_a_pod_whose_compiled_configuration_is_unreadable_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole entry point, including its exit status, because that status is the
    entire mechanism: a non-zero init container means the pod never starts and the Turn
    is refused, and a zero one means the agent runs against whatever tree is there."""
    missing = tmp_path / "not-mounted" / "config.toml"
    monkeypatch.setattr(restore_working_lane, "COMPILED_CONFIG", missing)

    assert main() == 1
    assert str(missing) in capsys.readouterr().err


def test_a_gateway_that_cannot_be_reached_refuses_and_says_so_without_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Port 1 on the loopback is a connect refusal in microseconds and reaches no
    network, which is what makes it usable as "the Gateway is not there" -- the shape a
    pod placed during a Gateway rollout would actually meet."""
    document = compiled_config_toml(tool_gateway_url="http://127.0.0.1:1/mcp")
    config = tmp_path / "config.toml"
    config.write_text(document, encoding="utf-8")
    monkeypatch.setattr(restore_working_lane, "COMPILED_CONFIG", config)

    assert main() == 1
    said = capsys.readouterr().err
    assert "refusing to start this pod" in said
    assert token_in(document) not in said
