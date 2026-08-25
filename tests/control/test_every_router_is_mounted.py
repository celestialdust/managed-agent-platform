"""A route module that declares a router but is never mounted answers nothing.

This exists because of a hole a whole wave of work fell into at once. Five route modules
were built in parallel, each with its own tests, and every one of those test files
builds its own `FastAPI` and calls `include_router` on the module under test. That is
the right way to test a router -- it isolates the routes from the rest of the app -- and
it means those tests stay green forever whether or not anybody ever mounts the module in
`create_app`. Forty-nine passing tests for one module proved its routes work and said
nothing about whether a caller can reach them.

So the failure this catches is not a broken route. It is a **correct route nobody can
call**, which is the one defect that looks identical to finished work from every angle
except this one: the module exists, its tests pass, its handlers are right, and every
request to it gets a 404 that reads like a wrong URL.

Read from the source rather than from a built app, and that is deliberate. Building the
app needs a `Platform`, which needs a database, a bucket and a cluster client -- so a
test that built one would be a tier-2 test with fixtures to maintain, for a question
that is answerable from twenty lines of Python. The structural read also cannot be
satisfied accidentally: there is no way to make this pass except by writing the
`include_router` call.

The pairing is by module name rather than by route path on purpose. Comparing paths
would need the app built, and it would also make this test fail for a *renamed* route,
which is not what it is asking. The question is only "is this module wired in".
"""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_API = _ROOT / "src" / "managed_agent" / "control" / "api"


def _modules_declaring_a_router() -> set[str]:
    """Every module under `control/api/` with a module-level `router = APIRouter(...)`.

    Module level specifically. A router built inside a function is a fixture or a test
    helper, not a surface this app is meant to serve, and counting one would make this
    test demand a mount that should not exist.
    """
    declaring: set[str] = set()
    for path in sorted(_API.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "router" in names:
                declaring.add(path.stem)
    assert declaring, (
        f"no module under {_API} declares a module-level router, so the comparison "
        "below holds between two empty sets; the package moved and this path did not"
    )
    return declaring


def _modules_mounted_by_create_app() -> set[str]:
    """Every module named in an `include_router(<module>.router, ...)` call.

    Scoped to `create_app` rather than to the whole file, because a mount written
    anywhere else does not happen: `create_app` is what `composition.build` calls, and a
    router included in some other function is exactly the kind of present-but-inert
    wiring this test exists to catch.
    """
    tree = ast.parse((_API / "app.py").read_text())
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_app"
    )
    mounted: set[str] = set()
    for node in ast.walk(factory):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if not isinstance(called, ast.Attribute) or called.attr != "include_router":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if (
            isinstance(first, ast.Attribute)
            and first.attr == "router"
            and isinstance(first.value, ast.Name)
        ):
            mounted.add(first.value.id)
    return mounted


def test_every_route_module_is_mounted_in_the_real_app() -> None:
    """Nothing declares a router and then goes unserved.

    The assertion is one-directional on purpose. A module mounted but declaring no
    router cannot happen -- the import would fail -- so the only reachable failure is a
    router nobody included, and naming just that direction keeps the message about the
    thing that went wrong.
    """
    declaring = _modules_declaring_a_router()
    mounted = _modules_mounted_by_create_app()
    unserved = sorted(declaring - mounted)
    assert not unserved, (
        "these modules declare a router that `create_app` never mounts, so every "
        f"request to their routes gets a 404: {unserved}. Their own tests pass because "
        "each builds its own app and includes its own router -- which is the correct "
        "way to test a router and is why nothing else in this suite can see this."
    )


def test_the_pairing_can_fail() -> None:
    """The two readers above disagree when they should.

    Without this, both functions returning the empty set would satisfy the test above
    forever, and a refactor that broke either reader would read as every router being
    mounted. Asserting a known-mounted module appears in both is what makes the
    comparison above evidence rather than a tautology over two empty sets.
    """
    declaring = _modules_declaring_a_router()
    mounted = _modules_mounted_by_create_app()
    assert "sessions" in declaring, declaring
    assert "sessions" in mounted, mounted
    assert "refusals" not in declaring, (
        "refusals.py installs middleware and declares no router; if it now declares "
        "one, this test's premise changed rather than its subject"
    )
