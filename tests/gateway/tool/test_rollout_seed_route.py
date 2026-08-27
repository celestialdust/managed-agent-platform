"""The Gateway route a pod reads its Session's Rollout back from.

Tier 1 (local, no infrastructure). Driven through the real app, over the real token
middleware, against a store that answers the way the control plane's own does.

What every case here is really about is one property: **which Session's conversation
comes back is decided by the token and by nothing a caller can vary.** A Rollout is a
whole agent conversation, so serving somebody else's is the worst thing this surface
could do, and there is no path parameter, no query and no body for a test to try -- so
the cases that matter drive TWO tokens at one app and assert the answers differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, cast
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from managed_agent.core.ids import SessionId, TenantId
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER_NAME,
    mint_session_token,
)
from managed_agent.gateway.tool.rollout_seed import ROLLOUT_SEED_PATH
from managed_agent.gateway.tool.server import GatewaySessions, create_gateway_app

KEY: Final = b"a session token signing key, 32b"
TENANT: Final = TenantId(uuid4())
FOREVER: Final = 4_102_444_800

A_ROLLOUT: Final = (
    b'{"type":"session_meta","payload":{"id":"thread-one"}}\n'
    b'{"type":"event_msg","payload":{"type":"turn_complete"}}\n'
)
ANOTHER_ROLLOUT: Final = b'{"type":"session_meta","payload":{"id":"thread-two"}}\n'


@dataclass(frozen=True, slots=True)
class _Restored:
    """What the control plane's cut hands back, in the one field this route reads."""

    body: bytes


class _StoredRollouts:
    """The Sessions whose conversations this store holds, and no others.

    Keyed by Session exactly as the real key layout is, so a route composing its key
    from anything a request carries would come back with the wrong answer here rather
    than with the right one by luck.
    """

    def __init__(self, held: dict[SessionId, bytes]) -> None:
        self._held = held
        self.asked: list[SessionId] = []

    async def restore_for_resume(self, session_id: SessionId) -> _Restored | None:
        self.asked.append(session_id)
        body = self._held.get(session_id)
        return None if body is None else _Restored(body)


def _app(store: _StoredRollouts) -> FastAPI:
    return create_gateway_app(cast(GatewaySessions, _no_upstreams()), KEY, store)


class _NoUpstreams:
    """The MCP half of this app, which no case here reaches."""

    async def sweep(self) -> int:
        return 0

    async def aclose(self) -> None:
        return None


def _no_upstreams() -> Any:
    return _NoUpstreams()


def _token(session_id: SessionId, tenant_id: TenantId = TENANT) -> str:
    return mint_session_token(
        session_id=session_id,
        tenant_id=tenant_id,
        expiry_epoch_s=FOREVER,
        key=KEY,
    )


async def _get(app: FastAPI, token: str | None) -> httpx.Response:
    headers = {} if token is None else {SESSION_TOKEN_HEADER_NAME: token}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway.invalid"
    ) as client:
        return await client.get(ROLLOUT_SEED_PATH, headers=headers)


# --------------------------------------------------------------------------------------
# What comes back, and for whom
# --------------------------------------------------------------------------------------


async def test_a_session_that_has_run_gets_its_own_conversation_back() -> None:
    session_id = SessionId(uuid4())
    store = _StoredRollouts({session_id: A_ROLLOUT})

    answer = await _get(_app(store), _token(session_id))

    assert answer.status_code == 200
    assert answer.content == A_ROLLOUT
    assert store.asked == [session_id]


async def test_two_sessions_holding_two_tokens_get_two_different_answers() -> None:
    """The whole security argument for this surface, asserted rather than reasoned.

    Nothing in a request names a Session, so a route that composed its key wrongly
    would serve one conversation to every caller -- and a single-Session case would
    pass against exactly that bug. Two Sessions through ONE app is what separates
    "reads the token" from "returns whatever it has".
    """
    mine = SessionId(uuid4())
    yours = SessionId(uuid4())
    app = _app(_StoredRollouts({mine: A_ROLLOUT, yours: ANOTHER_ROLLOUT}))

    assert (await _get(app, _token(mine))).content == A_ROLLOUT
    assert (await _get(app, _token(yours))).content == ANOTHER_ROLLOUT


async def test_a_session_asking_while_another_holds_a_rollout_gets_204_not_theirs() -> (
    None
):
    """Absence is answered as absence, not as whatever the store happens to hold.

    The store below is NOT empty, which is the point: a route that ignored the caller
    would find something and serve it. What must come back is nothing, because nothing
    belongs to this Session.
    """
    somebody = SessionId(uuid4())
    asking = SessionId(uuid4())
    app = _app(_StoredRollouts({somebody: A_ROLLOUT}))

    answer = await _get(app, _token(asking))

    assert answer.status_code == 204
    assert answer.content == b""


async def test_a_first_placement_is_204_rather_than_404_or_an_empty_200() -> None:
    """Three statuses are possible here and two of them break a pod.

    404 would put every Session's ordinary first start-up on an error path. An empty
    200 is worse: the caller writes a zero-byte file, the runtime is asked to resume
    from a record with no lines, and that is the one shape its own reader treats as a
    hard error -- so a Session that has simply never run would fail to place.
    """
    answer = await _get(_app(_StoredRollouts({})), _token(SessionId(uuid4())))

    assert answer.status_code == 204
    assert answer.content == b""


# --------------------------------------------------------------------------------------
# What is behind the token check
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [None, "not-a-token", "a.b.c.d"],
)
async def test_nothing_is_served_without_a_token_this_gateway_minted(
    token: str | None,
) -> None:
    """The route is behind the same middleware the MCP surface is, not beside it."""
    session_id = SessionId(uuid4())
    store = _StoredRollouts({session_id: A_ROLLOUT})

    answer = await _get(_app(store), token)

    assert answer.status_code == 401
    assert store.asked == [], "the store was read before the token was checked"


async def test_a_token_signed_by_another_key_reads_nothing() -> None:
    """A token this Gateway did not mint is not a token, however well-formed it is."""
    session_id = SessionId(uuid4())
    store = _StoredRollouts({session_id: A_ROLLOUT})
    forged = mint_session_token(
        session_id=session_id,
        tenant_id=TENANT,
        expiry_epoch_s=FOREVER,
        key=b"a different key of thirty-two by",
    )

    assert (await _get(_app(store), forged)).status_code == 401
    assert store.asked == []


async def test_the_route_serves_no_write_door() -> None:
    """GET only. A write here would be a second way to replace a Session's resume state,
    behind a different check from the ship-out that owns it -- and the pod holding this
    token is the least-trusted process in the platform."""
    session_id = SessionId(uuid4())
    app = _app(_StoredRollouts({session_id: A_ROLLOUT}))
    headers = {SESSION_TOKEN_HEADER_NAME: _token(session_id)}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway.invalid"
    ) as client:
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            answer = await client.request(
                method, ROLLOUT_SEED_PATH, headers=headers, content=b"x"
            )
            assert answer.status_code == 405, method
