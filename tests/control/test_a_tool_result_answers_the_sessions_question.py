"""A question a tool put to the Session gets answered, over the surface a tenant holds.

Tier 1 (local, in-memory ports). The Tool Gateway already puts the question on the
Session's Event Log and follows that log for the answer; what had no producer was the
answer. So the case that matters here runs both halves at once -- the shipped
`EventLogSessionChannel.ask` waiting on one side, the real route appending on the other
-- because either half can be correct on its own while the pair never meets.

**The two halves are in different processes in a deployment, and that is the point of
the design rather than an inconvenience for this file.** The Tool Gateway holds no
inbound route of its own; the tenant keeps talking to the control plane it already talks
to, and the Event Log is the whole of the channel between them. A test that faked the
Gateway's side would grade the route against a shape nobody reads.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from mcp.types import ElicitRequestFormParams

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.control.session.turn_dispatch import NoPodTransport
from managed_agent.core.errors import ErrorCode
from managed_agent.core.ids import (
    DefinitionId,
    Seq,
    SessionId,
    TenantId,
    new_session_id,
)
from managed_agent.core.ports import SessionNotVisible
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.vocabulary import tool_in_flight
from managed_agent.gateway.tool.mcp_proxy import EventLogSessionChannel, ToolEventTypes

_TYPES = ToolEventTypes(
    progress="tool.progress",
    elicitation_requested=tool_in_flight.TOOL_ELICITATION_REQUESTED,
    elicitation_answered=tool_in_flight.TOOL_ELICITATION_ANSWERED,
)


@dataclass(frozen=True, slots=True)
class Record:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


class FakeLog:
    """One in-memory log behind both ports, whose `follow` waits rather than ending.

    A `follow` that stopped when it ran out of rows would end every wait in this file as
    a cancellation, so the case that proves an answer arrives would pass identically
    against a route that appended nothing. Waiting on an event is what makes the arrival
    the thing under test.
    """

    def __init__(self) -> None:
        self.records: list[Record] = []
        self._appended = asyncio.Event()

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        return self.add(session_id, type_, payload)

    def add(self, session_id: SessionId, type_: str, payload: dict[str, object]) -> Seq:
        """Append without going through the port, standing in for another writer.

        The question is written by the Tool Gateway, which is a different process from
        the one serving the route under test. Cases that only need the question there
        put it there this way.
        """
        seq = Seq(len(self.records) + 1)
        self.records.append(Record(session_id, seq, type_, dict(payload)))
        self._appended.set()
        return seq

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[Record]:
        span = [r for r in self.records if r.session_id == session_id]
        return [r for r in span if start <= r.seq <= end][:limit]

    async def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[Record]:
        seen = after
        while True:
            for record in [r for r in self.records if r.seq > seen]:
                seen = record.seq
                yield record
            self._appended.clear()
            await self._appended.wait()

    async def retained_floor(self, session_id: SessionId) -> Seq:
        return Seq(1)

    def types(self) -> list[str]:
        return [r.type for r in self.records]

    def of_type(self, type_: str) -> list[Record]:
        return [r for r in self.records if r.type == type_]


class OneOwnedSession:
    """A registry showing one Session to one tenant and refusing everything else.

    The ownership check is the only thing between a caller and another tenant's Session,
    and this route writes into the Event Log -- which carries no tenant of its own -- so
    the check is real here rather than waved through.
    """

    def __init__(self, owner: TenantId, session_id: SessionId) -> None:
        self._owner = owner
        self._session_id = session_id

    async def create(self, record: SessionRecord) -> None:
        raise AssertionError("this surface created a Session")

    async def fetch(self, session_id: SessionId, tenant_id: TenantId) -> SessionRecord:
        if session_id != self._session_id or tenant_id != self._owner:
            raise SessionNotVisible(str(session_id))
        return SessionRecord(
            id=session_id,
            tenant_id=tenant_id,
            definition_id=DefinitionId(UUID(int=1)),
            definition_revision="1",
            grant=frozenset(),
            scope=(),
            budget_minor_units=1_000,
            budget_currency="USD",
            retention_days=30,
        )

    async def page(
        self, tenant_id: TenantId, after: tuple[int, UUID] | None, limit: int
    ) -> Sequence[Any]:
        raise AssertionError("this surface paged the Session registry")


class RefusesContact:
    """Every port this route must not touch, as one refusing object."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the tool-result route reached a store: .{name}()")


def _client(log: FakeLog, owner: TenantId, session_id: SessionId) -> AsyncClient:
    """The shipped app factory over a real `Platform`, with fakes behind the ports.

    Built through `create_app` rather than by mounting one router by hand, so a route
    that works and is never included in the app fails here.
    """
    app = create_app(
        Platform(
            event_log_append=log,
            event_log_range=log,
            definition_registry=RefusesContact(),
            tool_registry=RefusesContact(),
            session_registry=OneOwnedSession(owner, session_id),
            webhooks=RefusesContact(),
            environment_store=RefusesContact(),
            turn_dispatch=NoPodTransport(),
            file_store=unconfigured_file_store(),
        )
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://control")


def _url(session_id: SessionId, call_id: str) -> str:
    return f"/v1/sessions/{session_id}/tool_calls/{call_id}/result"


def _a_schema_ask(elicitation_id: str) -> dict[str, object]:
    """The question shape the shipped MCP path writes: a message and a form schema."""
    return {
        "elicitation_id": elicitation_id,
        "message": "which environment should I deploy to?",
        "requested_schema": {
            "type": "object",
            "properties": {"environment": {"type": "string"}},
        },
    }


async def test_an_answer_posted_here_reaches_the_tool_that_asked() -> None:
    """The whole slice, with both halves running: the mechanism completes.

    The Gateway's side is the shipped `ask`, not a stand-in, so what is graded is that
    the route writes the event that loop actually matches on -- the right type, the
    right correlation id, and content keyed the way the MCP result has to be keyed. A
    route that appended something merely plausible would leave the tool call hanging
    until its deadline and report a cancel, which is what this file exists to catch.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    channel = EventLogSessionChannel(session_id, log, log, _TYPES)
    asking = asyncio.create_task(
        channel.ask(
            ElicitRequestFormParams(
                message="which environment should I deploy to?",
                requested_schema={
                    "type": "object",
                    "properties": {"environment": {"type": "string"}},
                },
            )
        )
    )
    while not log.of_type(_TYPES.elicitation_requested):
        await asyncio.sleep(0)
    asked = log.of_type(_TYPES.elicitation_requested)[0]
    call_id = str(asked.payload["elicitation_id"])

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, call_id),
            json={
                "content_items": [{"type": "input_text", "text": "staging"}],
                "success": True,
            },
            headers={TENANT_HEADER: str(owner)},
        )

    assert answered.status_code == 202, answered.text
    result = await asyncio.wait_for(asking, timeout=5)
    assert result.action == "accept"
    assert result.content == {"environment": "staging"}


async def test_an_answer_is_recorded_as_the_session_s_own_event() -> None:
    """One event, of the published type, carrying the id the question was minted with.

    Asserted on the log rather than only through the Gateway, because the Event Log is
    also what a tenant reads back and what a Turn resuming after the hold window would
    have to find the answer in.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_schema_ask("e1"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e1"),
            json={
                "content_items": [{"type": "input_text", "text": "staging"}],
                "success": True,
            },
            headers={TENANT_HEADER: str(owner)},
        )

    assert answered.status_code == 202, answered.text
    recorded = log.of_type(_TYPES.elicitation_answered)
    assert [r.payload for r in recorded] == [
        {
            "elicitation_id": "e1",
            "action": "accept",
            "content": {"environment": "staging"},
        }
    ]
    assert answered.json()["seq"] == recorded[0].seq


async def test_a_result_that_did_not_succeed_declines_and_carries_no_content() -> None:
    """The other arm of `success`, and the content items do not ride along with it.

    A decline is the MCP action meaning the caller would not answer, and the result
    shape it maps to carries nothing -- so text sent beside a failure is the caller's
    account of the failure rather than an answer, and writing it into the answer's
    content would hand the registered server a value nobody supplied as one.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_schema_ask("e2"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e2"),
            json={
                "content_items": [{"type": "input_text", "text": "nobody was around"}],
                "success": False,
            },
            headers={TENANT_HEADER: str(owner)},
        )

    assert answered.status_code == 202, answered.text
    assert [r.payload for r in log.of_type(_TYPES.elicitation_answered)] == [
        {"elicitation_id": "e2", "action": "decline", "content": {}}
    ]


async def test_an_id_naming_no_question_is_refused_and_writes_nothing() -> None:
    """An orphan answer in the log is worse than a refusal: nobody would ever read it.

    The Gateway matches an answer to the question it minted an id for, so an answer
    naming an id nothing asked under is followed by no one and expires as a cancel --
    silently, minutes later, on the tool call rather than on the request that got it
    wrong.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_schema_ask("e3"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "not-a-question"),
            json={
                "content_items": [{"type": "input_text", "text": "staging"}],
                "success": True,
            },
            headers={TENANT_HEADER: str(owner)},
        )

    assert answered.status_code == 400, answered.text
    assert answered.json()["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert log.of_type(_TYPES.elicitation_answered) == []


async def test_a_result_with_the_wrong_number_of_items_is_refused() -> None:
    """The answers are paired with the questions by position, so the counts must agree.

    Pairing a short list would leave a question silently unanswered and the registered
    server proceeding on a field nobody filled in, which is the one failure mode an
    elicitation exists to prevent. Refusing names both numbers, because the caller
    cannot see the question's shape from the refusal alone.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_schema_ask("e4"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e4"),
            json={"content_items": [], "success": True},
            headers={TENANT_HEADER: str(owner)},
        )

    assert answered.status_code == 400, answered.text
    assert answered.json()["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert log.of_type(_TYPES.elicitation_answered) == []


async def test_another_tenant_cannot_answer_this_session_s_question() -> None:
    """The Event Log carries no tenant, so the registry is the only check there is.

    Refused with the code an absent Session gets, so the refusal cannot be used to learn
    whether an id names somebody else's Session.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_schema_ask("e5"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e5"),
            json={
                "content_items": [{"type": "input_text", "text": "production"}],
                "success": True,
            },
            headers={TENANT_HEADER: str(TenantId(uuid4()))},
        )

    assert answered.status_code == 404, answered.text
    assert answered.json()["error"]["code"] == ErrorCode.SESSION_NOT_FOUND.value
    assert log.of_type(_TYPES.elicitation_answered) == []


@pytest.mark.parametrize(
    "body",
    [
        {"content_items": [{"type": "input_image", "image_url": "x"}], "success": True},
        {"content_items": [{"type": "input_text"}], "success": True},
        {"content_items": [{"type": "input_text", "text": "s"}]},
    ],
    ids=["an-image-item", "a-text-item-with-no-text", "no-success-field"],
)
async def test_a_body_this_route_does_not_serve_is_refused(
    body: dict[str, object],
) -> None:
    """Text answers only, and both fields required.

    An image or an audio item is a shape the generic tool-result envelope carries and
    this route has nothing to do with -- an elicitation answers a question with words --
    so it is refused at the boundary rather than dropped somewhere below it.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_schema_ask("e6"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e6"), json=body, headers={TENANT_HEADER: str(owner)}
        )

    assert answered.status_code == 400, answered.text
    assert log.of_type(_TYPES.elicitation_answered) == []


def _a_secret_ask(elicitation_id: str) -> dict[str, object]:
    """The envelope shape, with one of its questions asking for a credential.

    Stated as the generic tool-call envelope because that is the shape that can say a
    question is secret at all: a form schema has no such keyword, so a secret question
    can only arrive this way.
    """
    return {
        "elicitation_id": elicitation_id,
        "tool": "request_user_input",
        "arguments": {
            "questions": [
                {"id": "environment", "question": "which environment?"},
                {
                    "id": "deploy_token",
                    "question": "paste the deploy token",
                    "is_secret": True,
                },
            ]
        },
    }


async def test_a_question_asking_for_a_secret_is_refused_and_nothing_is_written() -> (
    None
):
    """The hard requirement, and the assertion that matters is the one about the log.

    A refusal that still appended would be worse than no refusal: the caller would be
    told the value was rejected while it sat in the Event Log on their retention clock,
    and in the Rollout the next Turn ships out of the pod. So the status is checked and
    then the log is searched for the value itself -- across every event, not only the
    answer -- because a secret written under some other type is the same secret.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_secret_ask("e7"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e7"),
            json={
                "content_items": [
                    {"type": "input_text", "text": "staging"},
                    {"type": "input_text", "text": "hunter2-the-real-token"},
                ],
                "success": True,
            },
            headers={TENANT_HEADER: str(owner)},
        )

    assert answered.status_code == 409, answered.text
    assert (
        answered.json()["error"]["code"] == ErrorCode.ELICITATION_SECRET_REFUSED.value
    )
    assert log.of_type(_TYPES.elicitation_answered) == []
    assert "hunter2" not in str([r.payload for r in log.records])


async def test_the_refusal_points_the_caller_at_the_surface_that_holds_a_secret() -> (
    None
):
    """A refusal with no next move in it is a dead end, and this one has an answer.

    Vaults is where a credential belongs and where a registered tool already reads one
    from, so the caller's move is to put the value there and answer with the reference
    -- which is a different thing from "your request was malformed" and is why this
    carries a code of its own.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_secret_ask("e8"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e8"),
            json={
                "content_items": [
                    {"type": "input_text", "text": "staging"},
                    {"type": "input_text", "text": "a token"},
                ],
                "success": True,
            },
            headers={TENANT_HEADER: str(owner)},
        )

    body = answered.json()["error"]
    assert "vault" in body["message"].lower()
    assert body["detail"]["questions"] == "deploy_token"


async def test_a_secret_question_is_refused_even_when_the_result_declines() -> None:
    """Neither arm of `success` gets past it, because the refusal is about the question.

    A decline carries no answer, so recording one would leak nothing -- and it is still
    refused, because accepting it would tell the caller this is a question they may
    answer here, which is the belief the whole rule exists to prevent.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(session_id, _TYPES.elicitation_requested, _a_secret_ask("e9"))

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e9"),
            json={"content_items": [], "success": False},
            headers={TENANT_HEADER: str(owner)},
        )

    assert answered.status_code == 409, answered.text
    assert log.of_type(_TYPES.elicitation_answered) == []


async def test_an_envelope_question_that_is_not_secret_is_answered_normally() -> None:
    """The other arm of the envelope shape, so the refusal is not the whole of it.

    Without this, a route that refused every envelope-shaped ask would pass the secret
    cases and nothing would notice that the shape had stopped working entirely.
    """
    owner, session_id = TenantId(uuid4()), new_session_id()
    log = FakeLog()
    log.add(
        session_id,
        _TYPES.elicitation_requested,
        {
            "elicitation_id": "e10",
            "tool": "request_user_input",
            "arguments": {"questions": [{"id": "environment"}]},
        },
    )

    async with _client(log, owner, session_id) as client:
        answered = await client.post(
            _url(session_id, "e10"),
            json={
                "content_items": [{"type": "input_text", "text": "staging"}],
                "success": True,
            },
            headers={TENANT_HEADER: str(owner)},
        )

    assert answered.status_code == 202, answered.text
    assert [r.payload for r in log.of_type(_TYPES.elicitation_answered)] == [
        {
            "elicitation_id": "e10",
            "action": "accept",
            "content": {"environment": "staging"},
        }
    ]
