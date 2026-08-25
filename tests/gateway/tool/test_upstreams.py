"""One Session's upstream connections: opened once, owned by one task, closed for good.

Tier 1, but not fake: every connection here is a real child process speaking the real
protocol down a pipe. The cancel-scope property this module exists for cannot be
observed against an in-memory stand-in — it is a fact about anyio task groups and the
task that entered one, so the transport has to be a transport.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import traceback
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import NoReturn

import pytest
from mcp import ClientSession
from mcp.types import ElicitRequestParams, ElicitResult
from tool_gateway_harness import (
    CREDENTIAL_REF,
    TENANT,
    CountingVault,
    broker,
    stdio_endpoint,
)

from managed_agent.core.errors import ErrorCode
from managed_agent.core.registration.scope_binding import (
    ServerEndpoint,
    ServerName,
    StdioServer,
    StreamableHttpServer,
)
from managed_agent.gateway.tool import error_map
from managed_agent.gateway.tool.credential_broker import vault_name
from managed_agent.gateway.tool.mcp_proxy import (
    GATEWAY_ELICITATION_TIMEOUT_S,
    GATEWAY_STARTUP_TIMEOUT_S,
    GATEWAY_TOOL_TIMEOUT_S,
    RUNTIME_MCP_STARTUP_TIMEOUT_S,
    RUNTIME_MCP_TOOL_TIMEOUT_S,
    UPSTREAM_READ_TIMEOUT_S,
    SessionUpstreams,
    deadlines_nest,
)

_SERVER: ServerName = "conformance_stdio"


async def _decline(params: ElicitRequestParams) -> ElicitResult:
    return ElicitResult(action="decline")


@contextlib.asynccontextmanager
async def _running(vault: CountingVault) -> AsyncIterator[SessionUpstreams]:
    """A `SessionUpstreams` with its owning task scheduled, closed on the way out."""
    upstreams = SessionUpstreams(
        tenant_id=TENANT, broker=broker(vault), elicitation=_decline
    )
    owner = asyncio.create_task(upstreams.run())
    try:
        yield upstreams
    finally:
        await upstreams.aclose()
        await owner


def test_every_gateway_deadline_sits_strictly_inside_the_runtimes() -> None:
    """The ordering, stated as the arithmetic rather than as prose.

    `UPSTREAM_READ_TIMEOUT_S` is compared only against the tool deadline it is derived
    from. It is not inside the startup deadline and is not meant to be: those two bound
    different things — a connection coming up, versus a request on a connection already
    up — and neither encloses the other. An earlier draft of this test asserted it was
    below all three, which is arithmetically false twice at the values in force (54.0 is
    below neither 5.0 nor 50.0) and would have been "fixed" by changing a constant.
    """
    assert GATEWAY_STARTUP_TIMEOUT_S < RUNTIME_MCP_STARTUP_TIMEOUT_S
    assert GATEWAY_TOOL_TIMEOUT_S < RUNTIME_MCP_TOOL_TIMEOUT_S
    assert GATEWAY_ELICITATION_TIMEOUT_S < GATEWAY_TOOL_TIMEOUT_S
    assert UPSTREAM_READ_TIMEOUT_S < GATEWAY_TOOL_TIMEOUT_S


def test_the_nesting_predicate_answers_both_ways() -> None:
    """A reload cannot reach the import guard, so the predicate is what gets tested.

    `importlib.reload` re-executes the module body, which reassigns the headroom to its
    source value before the guard runs — so no monkeypatched constant can ever provoke
    the `RuntimeError`. The second case is the figures a headroom of 60 would produce.
    """
    assert deadlines_nest(5.0, 55.0, 50.0) is True
    assert deadlines_nest(-50.0, 0.0, -60.0) is False


async def test_the_credential_reaches_the_child_and_stays_out_of_this_process() -> None:
    vault = CountingVault()
    async with _running(vault) as upstreams:
        session = await upstreams.session_for(_SERVER, stdio_endpoint())
        result = await session.call_tool("echo_credential", {})

    assert [c.text for c in result.content if c.type == "text"] == [vault.value]
    assert vault.fetches == [vault_name(TENANT, CREDENTIAL_REF)]
    assert os.environ.get("MAP_CONFORMANCE_TOKEN") is None


async def test_a_second_call_for_one_server_reuses_the_child_it_already_spawned() -> (
    None
):
    vault = CountingVault()
    async with _running(vault) as upstreams:
        first = await upstreams.session_for(_SERVER, stdio_endpoint())
        second = await upstreams.session_for(_SERVER, stdio_endpoint())

        assert first is second
    assert len(vault.fetches) == 1


async def test_two_concurrent_first_calls_spawn_exactly_one_child() -> None:
    """The opening lock is the whole of this: without it both callers miss and spawn."""
    vault = CountingVault()
    async with _running(vault) as upstreams:
        both = await asyncio.gather(
            upstreams.session_for(_SERVER, stdio_endpoint()),
            upstreams.session_for(_SERVER, stdio_endpoint()),
        )

        assert both[0] is both[1]
    assert len(vault.fetches) == 1


async def test_three_calls_from_three_separate_tasks_all_succeed() -> None:
    """The regression test for the cancel scope, and it fails without the owning task.

    A connection opened inside one request's task is torn down when that task ends, so
    the second caller finds a dispatcher already closed. Each call here runs in a task
    of its own and none of them is the task that entered the exit stack.
    """

    async def one(upstreams: SessionUpstreams) -> str:
        session = await upstreams.session_for(_SERVER, stdio_endpoint())
        result = await session.call_tool("echo_credential", {})
        return "".join(c.text for c in result.content if c.type == "text")

    vault = CountingVault()
    async with _running(vault) as upstreams:
        answers = [await asyncio.create_task(one(upstreams)) for _ in range(3)]

    assert answers == [vault.value] * 3
    assert len(vault.fetches) == 1


async def test_a_command_that_is_not_on_path_fails_fast_as_an_unavailable_server() -> (
    None
):
    endpoint = StdioServer(
        transport="stdio",
        command="/opt/acme/secret-mcp-server",
        args=(),
        credential_ref="conformance/stdio-token",
        credential_env_var="MAP_CONFORMANCE_TOKEN",
    )

    async with _running(CountingVault()) as upstreams:
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(Exception) as caught:  # noqa: B017 - the shape is the point
            await upstreams.session_for(_SERVER, endpoint)
        elapsed = loop.time() - started

    assert elapsed < GATEWAY_STARTUP_TIMEOUT_S
    assert error_map.classify(caught.value) is ErrorCode.TOOL_UNAVAILABLE


async def test_closing_reaps_the_child_rather_than_only_signalling_it(
    tmp_path: Path,
) -> None:
    """`aclose` returns only once the owning task has left its exit stack.

    The child writes its own pid because the proxy hands back no handle on the process
    it spawned. Asserting the pid is gone is what separates "the stack unwound" from
    "the stack was asked to unwind".
    """
    pid_file = tmp_path / "child.pid"
    upstreams = SessionUpstreams(
        tenant_id=TENANT, broker=broker(CountingVault()), elicitation=_decline
    )
    owner = asyncio.create_task(upstreams.run())
    await upstreams.session_for(_SERVER, stdio_endpoint(str(pid_file)))
    child_pid = int(pid_file.read_text())
    assert _alive(child_pid), "the child was not running before the close"

    await upstreams.aclose()
    await owner

    assert owner.done()
    assert not _alive(child_pid)


@contextlib.asynccontextmanager
async def _refuses_to_leave() -> AsyncIterator[None]:
    """A transport that entered cleanly and will not come apart again.

    Reaching into `_open` to plant this is the only way to build the state: no
    registration can produce a transport that opens and then refuses to close, and that
    is precisely the state the two failures below live in.
    """
    yield
    raise RuntimeError("the transport refused to unwind")


async def test_a_stack_that_refuses_to_unwind_still_releases_the_close() -> None:
    """`aclose` is a wait on an event, and the event has to be set either way.

    Set it only on the success path and one bad unwind leaves every `aclose` for this
    Session waiting forever — and because the front door releases Sessions in a loop,
    one wedged Session would stall the sweep of every other. The exception still reaches
    whoever awaits the owning task; what is asserted here is that the waiter is released
    to see it.
    """
    upstreams = SessionUpstreams(
        tenant_id=TENANT, broker=broker(CountingVault()), elicitation=_decline
    )
    original = upstreams._open

    async def open_then_poison(
        stack: AsyncExitStack, endpoint: ServerEndpoint
    ) -> ClientSession:
        session = await original(stack, endpoint)
        # Entered last, so it unwinds first: the failure lands in the middle of the
        # Session-wide close rather than at the open.
        await stack.enter_async_context(_refuses_to_leave())
        return session

    upstreams._open = open_then_poison  # type: ignore[method-assign]
    owner = asyncio.create_task(upstreams.run())
    await upstreams.session_for(_SERVER, stdio_endpoint())

    async with asyncio.timeout(5.0):
        await upstreams.aclose()

    assert owner.done(), "aclose returned before the owner had left its stack"
    with pytest.raises(Exception) as caught:  # noqa: B017 - the shape is the point
        await owner
    # Nested groups, because the transport's own task group is between the failure and
    # this frame. What matters is that the cause survived the close rather than the
    # depth it survived at.
    assert "refused to unwind" in "".join(traceback.format_exception(caught.value))


async def test_one_servers_failed_open_leaves_another_servers_connection_alive() -> (
    None
):
    """A failing open is contained to that server, and the owning task survives it.

    Each transport runs on an anyio task group whose failure cancels the task that
    entered it, so a half-open transport left on a stack shared with every other server
    delivers that cancellation later and takes the whole Session down with it. Measured
    that way: the owning task died and the caller waiting on its answer waited forever.

    The failing server has to be a *Streamable HTTP* one, and that is the finding rather
    than a detail. A stdio command that exits during `initialize` fails without ever
    cancelling this task, so a version of this test built on one passes against the
    shared stack too — a guard that would have proved nothing. Probed: with the shared
    stack restored, a stdio-doomed variant stayed green and this one fails.

    Offline despite the URL: port 1 on the loopback interface refuses immediately, with
    no name to resolve and nothing leaving the machine. It has to be `https://` because
    that is what a registration will accept.

    Both halves are asserted together, because either alone is satisfiable by breaking
    the other: the good server answers *after* the bad one failed, and the failure is
    still reported rather than swallowed.
    """
    doomed = StreamableHttpServer(
        transport="streamable_http",
        url="https://127.0.0.1:1/mcp",
        credential_ref="conformance/stdio-token",
        credential_header="X-Map-Conformance",
    )
    vault = CountingVault()

    async with _running(vault) as upstreams:
        healthy = await upstreams.session_for("good", stdio_endpoint())

        with pytest.raises(Exception) as caught:  # noqa: B017 - the shape is the point
            await upstreams.session_for("doomed", doomed)

        after = await healthy.call_tool("echo_credential", {})

    assert error_map.classify(caught.value) is ErrorCode.TOOL_UNAVAILABLE
    assert [c.text for c in after.content if c.type == "text"] == [vault.value]


async def test_a_caller_waiting_on_an_open_is_released_when_the_owner_ends() -> None:
    """Nobody may be left waiting on a future only a task that has ended can settle.

    `session_for` hands its request to the owning task and waits. If that task ends for
    any reason — a cancelled release, an unwind that raised — the waiter is waiting on a
    reader of a queue nobody reads any more, and no deadline anywhere covers it.
    """
    upstreams = SessionUpstreams(
        tenant_id=TENANT, broker=broker(CountingVault()), elicitation=_decline
    )

    async def never_opens(stack: AsyncExitStack, endpoint: ServerEndpoint) -> NoReturn:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    upstreams._open = never_opens  # type: ignore[method-assign]
    owner = asyncio.create_task(upstreams.run())
    waiting = asyncio.create_task(upstreams.session_for(_SERVER, stdio_endpoint()))
    await asyncio.sleep(0.05)
    assert not waiting.done()

    owner.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await owner

    async with asyncio.timeout(2.0):
        with pytest.raises(ConnectionError):
            await waiting


async def test_asking_after_the_owner_has_stopped_is_refused_rather_than_queued() -> (
    None
):
    """The queue outlives its reader, so a late request has to be turned away."""
    upstreams = SessionUpstreams(
        tenant_id=TENANT, broker=broker(CountingVault()), elicitation=_decline
    )
    owner = asyncio.create_task(upstreams.run())
    await upstreams.aclose()
    await owner

    async with asyncio.timeout(2.0):
        with pytest.raises(ConnectionError):
            await upstreams.session_for(_SERVER, stdio_endpoint())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
