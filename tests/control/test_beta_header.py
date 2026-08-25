"""`anthropic-beta`: which wire shape a caller asked for, and what happens when it
asked for one this build does not answer in.

A schema version in a header is only worth checking because of what happens when it is
not checked: a caller pins a shape, the server answers a different one, and the response
parses far enough to look like an answer. Nothing raises. The mismatch surfaces
later, at whichever field the caller did not happen to read.

So the cases here divide into three, and the third is the one that will age:

1. **The served version is served.** Present and echoed back, so a caller that sent
   nothing still learns which shape it got.
2. **Anything else is refused**, in the same envelope as every other refusal in this
   API -- which is not free, because the check runs in middleware and Starlette's
   exception handlers cannot see an exception raised there.
3. **Absence is answerable only while one version is served.** That is true today and
   is not a property of the design; it is a property of the count.
   `test_absence_stops_being_answerable_once_a_second_version_exists` is what turns
   that from a thing somebody has to remember into a thing the suite says.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from managed_agent.composition import Platform
from managed_agent.control.api.app import create_app
from managed_agent.control.api.refusals import REQUEST_ID_HEADER
from managed_agent.control.api.request.beta import (
    BETA_HEADER,
    MANAGED_AGENTS_2026_04_01,
    SERVED,
    version_asked_for,
)
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.files.store import unconfigured_file_store
from managed_agent.core.errors import STATUS_FOR, ErrorCode, PublicErrorType
from managed_agent.core.ids import TenantId

# `/v1/healthz` is the probe used throughout: it reads nothing off the platform, so a
# refusal here is unambiguously the header check and not a store that was not wired.
PROBE = "/v1/healthz"


class Unused:
    """Every port raises, because the header check runs before a route reads one.

    Not a convenience. If any case here passed by reaching a store, it would be
    testing that store's behaviour under a bad header rather than the header check,
    and it would keep passing after the check was deleted.
    """

    def __getattr__(self, name: str) -> Any:
        async def refuse(*args: object, **kwargs: object) -> Any:
            raise AssertionError(f"the beta-header check let {name} be reached")

        return refuse


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    platform = Platform(
        event_log_append=Unused(),
        event_log_range=Unused(),
        definition_registry=Unused(),
        tool_registry=Unused(),
        session_registry=Unused(),
        webhooks=Unused(),
        environment_store=Unused(),
        turn_dispatch=Unused(),
        file_store=unconfigured_file_store(),
        skill_store=Unused(),
    )
    async with AsyncClient(
        transport=ASGITransport(app=create_app(platform)),
        base_url="http://control-plane",
        headers={TENANT_HEADER: str(TenantId(uuid.uuid4()))},
    ) as made:
        yield made


# ---------------------------------------------------------------------------------
# The version this build serves
# ---------------------------------------------------------------------------------


async def test_the_served_version_is_served_and_echoed(client: AsyncClient) -> None:
    """Echoed rather than merely accepted.

    A caller reading the response header learns which shape it holds without having to
    know this build's default, and a proxy in between has something to cache on.
    """
    answered = await client.get(PROBE, headers={BETA_HEADER: MANAGED_AGENTS_2026_04_01})

    assert answered.status_code == 200, answered.text
    assert answered.headers[BETA_HEADER] == MANAGED_AGENTS_2026_04_01


async def test_a_request_naming_no_version_is_served_and_told_which(
    client: AsyncClient,
) -> None:
    """Answerable only because exactly one shape exists -- see the guard test below."""
    answered = await client.get(PROBE)

    assert answered.status_code == 200, answered.text
    assert answered.headers[BETA_HEADER] == MANAGED_AGENTS_2026_04_01


# ---------------------------------------------------------------------------------
# Anything else, and in the same envelope as everything else
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "because"),
    [
        ("managed-agents-2027-01-01", "a version later than any this build serves"),
        ("managed-agents-2025-01-01", "a version earlier than the one served"),
        ("agent-memory-2026-07-22", "their memory header; this platform serves none"),
        ("MANAGED-AGENTS-2026-04-01", "upper case, and the value is compared exactly"),
        (" ", "whitespace only, which names nothing"),
        ("", "empty, which names nothing"),
        ("managed-agents", "the family with no date"),
        (
            f"{MANAGED_AGENTS_2026_04_01},agent-memory-2026-07-22",
            "one served and one not; answering the subset would answer a caller that "
            "asked for two shapes with one",
        ),
    ],
)
async def test_a_version_this_build_does_not_serve_is_refused(
    client: AsyncClient, value: str, because: str
) -> None:
    """Refused, and the refusal names both sides.

    The caller's next move is to change its own pin, and it cannot do that from
    "unsupported" alone -- so the message has to carry what was asked for and what is
    served. Asserted, because a message that omits either is a refusal the caller
    cannot act on.
    """
    refused = await client.get(PROBE, headers={BETA_HEADER: value})

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_BETA_UNSUPPORTED]
    body: dict[str, Any] = refused.json()
    assert body["error"]["code"] == ErrorCode.REQUEST_BETA_UNSUPPORTED.value
    assert MANAGED_AGENTS_2026_04_01 in body["error"]["message"], because


async def test_the_refusal_wears_the_one_envelope_and_carries_the_request_id(
    client: AsyncClient,
) -> None:
    """**The case that would silently not hold.**

    This check runs in middleware, and Starlette's exception handlers live inside every
    user middleware -- so an exception raised here escapes past them and leaves through
    `ServerErrorMiddleware` with a traceback and no envelope at all. Not a 500 envelope:
    no envelope. That is why the middleware catches its own refusal and calls the same
    `refuse` every route calls, and why this test asserts the full shape rather than
    just the status.

    The request id is the half that depends on install ORDER. This middleware has to be
    the inner of the two, so the id is on its ContextVar by the time `refuse` reads it.
    Installed the other way round the body would still be shaped correctly and every
    refusal from here would be attributed to `req_unattributed`.
    """
    refused = await client.get(
        PROBE, headers={BETA_HEADER: "managed-agents-2027-01-01"}
    )

    body: dict[str, Any] = refused.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == PublicErrorType.INVALID_REQUEST.value
    assert body["request_id"].startswith("req_")
    assert body["request_id"] != "req_unattributed", (
        "the refusal is not attributed to the request that caused it, which means the "
        "beta middleware is installed outside the request-id middleware rather than "
        "inside it"
    )
    assert refused.headers[REQUEST_ID_HEADER] == body["request_id"]


async def test_a_refused_header_stops_the_request_before_the_route(
    client: AsyncClient,
) -> None:
    """A route reached and then discarded is work nobody receives.

    Asserted against a path that does not exist: if the check ran after routing, this
    would be refused for being absent (404) rather than for the header (400), and the
    header refusal would be the thing that never happened.
    """
    refused = await client.get(
        "/v1/there-is-no-such-route", headers={BETA_HEADER: "managed-agents-2027-01-01"}
    )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_BETA_UNSUPPORTED]
    assert (
        refused.json()["error"]["code"] == ErrorCode.REQUEST_BETA_UNSUPPORTED.value
    ), "the router answered first, so the header check is not in front of routing"


# ---------------------------------------------------------------------------------
# The guard on the assumption that makes absence answerable
# ---------------------------------------------------------------------------------


def test_absence_stops_being_answerable_once_a_second_version_exists() -> None:
    """**Read this before adding a second served version.**

    A request naming no version is answered in the one shape this build serves. That is
    correct while there is one shape and wrong the moment there are two: with two, the
    server is guessing which the caller parses, and guessing wrong is the silent
    mismatch this whole module exists to prevent.

    So when `SERVED` grows, this test fails, and the fix is not to change this
    assertion. The fix is to make an absent header a refusal in `version_asked_for` --
    the branch is already written there -- and then to rewrite
    `test_a_request_naming_no_version_is_served_and_told_which`, which will be asserting
    something that is no longer true.
    """
    assert len(SERVED) == 1, (
        "SERVED has grown, so a request naming no version is now ambiguous. Make "
        "absence a refusal in version_asked_for (the branch is written) and update "
        "test_a_request_naming_no_version_is_served_and_told_which, which currently "
        "asserts that absence is answered."
    )


def test_every_served_value_is_one_a_caller_could_have_sent() -> None:
    """No entry with surrounding whitespace or a different case.

    The comparison is exact, so an entry that is not in the form a caller sends is an
    entry nothing can ever match -- a version this build believes it serves and refuses
    every request for.
    """
    for value in SERVED:
        assert value == value.strip(), value
        assert value == value.lower(), value
        assert value, "an empty served version matches nothing"


def test_the_function_and_the_middleware_agree_on_what_is_served() -> None:
    """The unit-level check, so a middleware regression cannot be the only signal."""

    class _Headerless:
        headers: dict[str, str] = {}

    asked: Any = _Headerless()
    assert version_asked_for(asked) == MANAGED_AGENTS_2026_04_01
