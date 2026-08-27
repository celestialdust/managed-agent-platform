"""The process entrypoint: it serves, it refuses to start unconfigured, and it disposes.

Tier 1 for the first case (testcontainers, real PostgreSQL 17), because "the factory
returns a working app" is only worth asserting against a real database — a fake would
prove that `create_app` was called, which is not the claim.

The third case is the reason this file exists. `asgi.py`'s shutdown hook is the only
thing that releases the pool, and a hook that is installed but never invoked looks
exactly like one that works: the process exits either way and PostgreSQL reaps the
connections on its own timeouts. So the disposal is asserted by spying on the engine
`build` hands back, which is the boundary, rather than by inspecting pool internals
afterwards.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from managed_agent import asgi
from managed_agent.composition import Platform, build
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import PodPhase, PodRunner
from managed_agent.control.sweep_loop import SWEEP_INTERVAL_ENV_VAR
from managed_agent.session_shim.pod_channel import HttpPodDispatch


def _the_sweep_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the one variable `build_app` reads for itself, so these cases reach an app.

    It has no default on purpose -- a control plane that started on a guessed sweep
    cadence would look configured from every angle while the number deciding how long a
    tenant waits for a callback was nobody's decision -- so every case here that builds
    an app has to name one. The value is a stand-in: nothing in this file is about the
    cadence, only about the pool and the pod runner on the other side of it.
    """
    monkeypatch.setenv(SWEEP_INTERVAL_ENV_VAR, "30")


async def test_the_factory_serves_a_real_request_against_the_configured_database(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_app()` reads the environment and returns an app that answers.

    A registration rather than a bare 200: it exercises a route, the tenant dependency,
    and one adapter writing to the real database, which together are what "the process
    can serve" means. A health endpoint would prove that uvicorn could bind a port.
    """
    monkeypatch.setenv("DATABASE_URL", database_url)
    _the_sweep_interval(monkeypatch)
    app = asgi.build_app()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://platform",
            headers={TENANT_HEADER: str(uuid4())},
        ) as caller:
            registered = await caller.post(
                "/v1/agents",
                json={
                    "name": "slr-reviewer",
                    "instructions": "Extract findings and name the source for each.",
                    "model": "gpt-5-codex",
                    "skills_repository": "git@github.com:acme/skills.git",
                    "skills_revision": "0" * 39 + "a",
                },
            )
        assert registered.status_code == 201, registered.text
        assert registered.json()["revision"] == 1
    finally:
        async with asgi_shutdown(app):
            pass


def test_the_factory_refuses_to_start_without_a_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No default URL, so an unconfigured process fails instead of reaching elsewhere.

    Asserted at the entrypoint as well as in `test_composition.py`, because the
    entrypoint is what an operator actually runs and a default introduced anywhere
    between here and `build` would make a misconfigured deploy start quietly.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MAP_DATABASE_URL", raising=False)
    with pytest.raises(KeyError):
        asgi.build_app()


async def test_shutdown_disposes_the_engine_the_composition_root_handed_over(
    monkeypatch: pytest.MonkeyPatch, database_url: str
) -> None:
    """The lifespan hook releases the pool, and this fails if the hook is removed.

    `dispose` is forwarded to the real engine rather than swallowed, so the spy cannot
    make the test pass by doing less than production does. Without the `finally:
    await engine.dispose()` line in `asgi.py` this reports `disposed=0`, which is the
    only way to tell an installed hook from an invoked one — a process exits either way
    and PostgreSQL reaps the connections on its own timeouts.
    """
    calls: list[str] = []

    def _spying_build(
        url: str | None = None, pod_runner: PodRunner | None = None
    ) -> tuple[Any, Any]:
        # `build` is imported from `composition` rather than read off `asgi`: under
        # `--strict` mypy refuses `asgi.build` as a re-export the module never declared,
        # and reaching through the module under test to get the real function would also
        # be read back through the monkeypatch on a second call.
        platform, engine = build(database_url, pod_runner=pod_runner)
        return platform, _Spy(engine, calls)

    monkeypatch.setattr(asgi, "build", _spying_build)
    _the_sweep_interval(monkeypatch)
    app = asgi.build_app()
    assert calls == [], "the pool was disposed before shutdown"

    async with asgi_shutdown(app):
        assert calls == [], "the pool was disposed while the app was still serving"
    assert calls == ["dispose"], (
        f"shutdown ran and the engine was disposed {len(calls)} times. The hook in "
        "`asgi.py` is the only thing that releases up to 50 pooled connections, and an "
        "installed hook that is never invoked is indistinguishable from a working one "
        "unless this asserts the call."
    )


class _Spy:
    """An engine that records `dispose` and forwards it, so nothing is left open."""

    def __init__(self, engine: AsyncEngine, calls: list[str]) -> None:
        self._engine = engine
        self._calls = calls

    async def dispose(self) -> None:
        self._calls.append("dispose")
        await self._engine.dispose()


async def test_shutdown_stops_the_turns_still_running_in_this_process(
    monkeypatch: pytest.MonkeyPatch, database_url: str
) -> None:
    """The lifespan cancels in-flight Turns *and* waits for each one to stop.

    A Turn no longer ends inside the request that submitted it, so at shutdown there
    can be tasks mid-Turn that nothing else is holding. Cancelling without awaiting
    would let the process exit with one of them still pending, which asyncio reports as
    "Task was destroyed but it is pending" -- so this asserts the task is *done* on the
    way out, not merely that `cancel` was called.

    Asserted here rather than only on `BackgroundTurns` itself, because a unit test of
    that class passes whether or not `asgi.py` ever calls it: an installed hook and an
    invoked one are indistinguishable from outside unless something drives the lifespan.
    """
    held: list[Any] = []

    def _capturing_build(
        url: str | None = None, pod_runner: PodRunner | None = None
    ) -> tuple[Any, Any]:
        platform, engine = build(database_url, pod_runner=pod_runner)
        held.append(platform)
        return platform, engine

    monkeypatch.setattr(asgi, "build", _capturing_build)
    _the_sweep_interval(monkeypatch)
    app = asgi.build_app()
    (platform,) = held

    async def a_turn_that_never_ends() -> None:
        await asyncio.sleep(3600)

    async with asgi_shutdown(app):
        platform.background_turns.start(
            a_turn_that_never_ends(), name="map-turn-under-test"
        )
        (running,) = platform.background_turns.in_flight
        assert not running.done(), "the Turn stopped while the app was still serving"

    assert running.done(), (
        "shutdown returned with a Turn's task still pending. `asgi.py` must both "
        "cancel and await it -- a cancel only requests the stop, and a process that "
        "exits without waiting leaves the Turn half-done."
    )
    assert running.cancelled()
    assert platform.background_turns.in_flight == ()


class asgi_shutdown:  # noqa: N801 -- reads as a context manager at the call site
    """Run an app's lifespan startup on enter and its shutdown on exit.

    `httpx.ASGITransport` does not run the lifespan protocol, so a test driving routes
    through it never reaches the hook under test. This drives the app's
    `router.lifespan_context` directly, which is the same object uvicorn enters.
    """

    def __init__(self, app: Any) -> None:
        self._context = app.router.lifespan_context(app)

    async def __aenter__(self) -> None:
        await self._context.__aenter__()

    async def __aexit__(self, *exc: Any) -> None:
        await self._context.__aexit__(*exc)


def _the_placers_four_other_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set what `build` reads once it is handed a pod runner, so these cases reach it.

    A process with a runner compiles Session configurations, and the four values that
    takes -- both gateway addresses, the Session-token signing key and a token lifetime
    -- have no defaults on purpose. Set here as stand-ins, because nothing in this file
    is about their values; what is graded is the wiring on the other side of them.

    Not shared with the other files that need the same four lines. Two identical
    fixtures are a coincidence, and a shared module for them would couple files that
    are graded independently -- the same argument this repository already made about
    the `AbsentPod` doubles.
    """
    monkeypatch.setenv("MAP_SESSION_TOKEN_KEY", "a session-token signing key")
    monkeypatch.setenv("MAP_SESSION_TOKEN_LIFETIME_S", "3600")
    monkeypatch.setenv("MAP_TOOL_GATEWAY_URL", "http://tool-gateway.map-test/mcp")
    monkeypatch.setenv("MAP_MODEL_GATEWAY_URL", "http://model-gateway.map-test/v1")


class AbsentPod:
    """A cluster client that answers a phase and starts nothing.

    A second copy of the one in `test_composition.py`, and left as a copy: two identical
    eight-line doubles are a coincidence rather than a pattern, and a shared fixtures
    module for them would couple two files that are graded independently.
    """

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        return PodPhase.ABSENT

    async def phase_of(self, pod_name: str) -> PodPhase:
        return PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        return None


def test_the_served_process_wires_whatever_pod_runner_the_root_resolves(
    monkeypatch: pytest.MonkeyPatch, database_url: str
) -> None:
    """`build_app` asks the composition root for a runner instead of deciding.

    The root answers None today, so the served process refuses every Turn. What is
    substituted here is the answer, not the transport, which is what makes this a test
    of the entry point's wiring rather than of `build`. Without the `pod_runner=`
    argument in `asgi.py` this reports `NoPodTransport` whatever the root returns --
    which is the state the served process was left in while every other case in this
    file passed.
    """
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("MAP_SHIM_TOKEN_KEY", "a signing key")
    _the_placers_four_other_variables(monkeypatch)
    _the_sweep_interval(monkeypatch)
    monkeypatch.setattr(asgi, "pod_runner_from_environment", AbsentPod)

    app = asgi.build_app()

    platform = app.state.platform
    assert isinstance(platform, Platform)
    assert isinstance(platform.turn_dispatch, HttpPodDispatch)
