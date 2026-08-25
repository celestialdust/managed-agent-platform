"""Which registered shapes yield a sandbox that can actually be built, and which do not.

Tier 1: in-memory ports, the real routers, no cluster and no container. Everything here
runs in the default offline suite.

Three shapes a tenant could register until this slice landed make bubblewrap refuse to
build ANY sandbox, so every Session created against them fails before it runs a command.
The refusal is total and loud rather than a hole -- a tenant's self-inflicted denial of
service on itself -- which is why it is refused at registration, where the tenant who
wrote the shape is present, instead of inside a compilation whose failure surfaces to
whoever opened the Session.

THIS FILE DOES NOT MEASURE THAT FATALITY. Nothing here starts a pod. The two
measurements it rests on were taken in a real cluster against the real runtime:
`Can't create file at /session/workspace/.git: Is a directory` for a protected name, and
`Can't mkdir /session/workspace/a/b: Read-only file system` for a nested pair, both with
the confined command not running -- against a control in the same run where a single
non-nested deny at a path nothing created ran a confined command at exit 0 -- which is
what makes the nesting and not the missing target the cause. A nested pair is fatal
only once the descendant EXISTS; the runtime accepts one whose descendant is absent. The
registry refuses the class regardless, for the reason the compiler's floor does: this
runs where no pod exists, so whether the descendant will be there is not decidable here.

The accepted shapes are not decoration. Four of them -- `.agentsX`, `.gitignore`,
`my.git`, `agents` -- are refused by candidate forms of these clauses that were measured
before the clauses were written: a substring test refuses three of them and a
basename-prefix test refuses two. A refusal-only file is satisfied by a parse that
refuses everything, and these are how the two are told apart.

Both doors are exercised here because they close by two different mechanisms and one
refusal has only the first. The route refuses a shape with a `ValueError` rendered as
400. An `Environment` constructed directly, which every one of this repository's
non-registry construction sites does, meets the compiler's floor instead and raises
`FloorViolation` -- for a nested pair. For a protected name it meets nothing: such a
component adds no nested pair to the merged deny set, and a floor that refused one would
fail against the platform's own profile, which denies two protected names on purpose and
is measured working. That case is asserted here so that nobody deletes the registry
clause believing something downstream has it.
"""

import tomllib
from uuid import uuid4

import pytest

# The reference file's harness, imported rather than rebuilt. Every name here is
# module-level and public there, and this file is already in the same regression
# surface, so this is the second copy of the harness rather than the third. Imported by
# bare module name because `tests/` carries no `__init__.py`: pytest prepends this
# directory to `sys.path`, which is how `tests/pod/` already imports its own sibling.
from test_environment_reference import (
    A_DEFINITION,
    GATEWAY_URL,
    IMAGE,
    MODEL_GATEWAY_URL,
    SESSION_TOKEN_EXPIRY,
    SESSION_TOKEN_KEY,
    a_harness,
    a_record,
    build_app,
    caller,
)

from managed_agent.control.catalog.environments import (
    ALREADY_DENIED,
    RUNTIME_PROTECTED_NAMES,
)
from managed_agent.control.pod_config.compiler import (
    PROFILE_NAME,
    WORKSPACE_ROOT,
    CompiledConfig,
    FloorViolation,
    compile_session_config,
)
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import TenantId
from managed_agent.core.pod.permission_profile import nested_deny_pairs
from managed_agent.core.registration.environment import Environment, new_environment_id

# Every shape the route must still accept. Named once so the refusal cases, the accept
# cases and the compile-through sweep below all read one list -- a shape accepted in one
# and forgotten in another is how a rule that over-refuses at compile time survives.
_ACCEPTED_SHAPES: tuple[tuple[str, ...], ...] = (
    (),
    (f"{WORKSPACE_ROOT}/secrets",),
    (f"{WORKSPACE_ROOT}/secrets", f"{WORKSPACE_ROOT}/keys"),
    (f"{WORKSPACE_ROOT}/.agentsX",),
    (f"{WORKSPACE_ROOT}/.gitignore",),
    (f"{WORKSPACE_ROOT}/my.git",),
    (f"{WORKSPACE_ROOT}/agents",),
)


def _merged_deny_set(compiled: CompiledConfig) -> list[str]:
    """Every path the compiled document denies, from both lists that feed the argv.

    The profile table is what a reader of the document sees; `deny_read` is the
    non-weakenable copy. Both are pushed into the one policy the sandbox argv is
    compiled from, so a path in either reaches the sandbox and a check that read only
    one of them would have a half that never fires.
    """
    parsed = tomllib.loads(compiled.requirements_toml)
    table = parsed["permissions"][PROFILE_NAME]["filesystem"]
    deny_read = parsed["permissions"]["filesystem"]["deny_read"]
    return sorted({p for p, a in table.items() if a == "deny"} | set(deny_read))


def _compiled(denied_paths: tuple[str, ...]) -> CompiledConfig:
    """Compile a Session against a shape built WITHOUT the registry.

    Every place under `tests/` that constructs an `Environment` directly hands it to the
    compiler this way, and the one place in `src/` that builds one is the registry
    itself. So this is not a contrived path: it is the shape of every non-registry
    caller, and what refuses a bad shape there is the compiler's floor rather than any
    clause in this file's subject.
    """
    return compile_session_config(
        a_record(),
        tool_gateway_url=GATEWAY_URL,
        model_gateway_url=MODEL_GATEWAY_URL,
        environment=Environment(
            id=new_environment_id(),
            tenant_id=TenantId(uuid4()),
            name="analysis",
            runtime_image=IMAGE,
            denied_paths=denied_paths,
        ),
        definition=A_DEFINITION,
        session_token_key=SESSION_TOKEN_KEY,
        session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
    )


@pytest.mark.parametrize(
    ("denied_paths", "expected"),
    [
        ([f"{WORKSPACE_ROOT}/.git"], "keeps read-only in every workspace"),
        ([f"{WORKSPACE_ROOT}/sub/.codex"], "keeps read-only in every workspace"),
        ([f"{WORKSPACE_ROOT}/.git/config"], "keeps read-only in every workspace"),
        ([f"{WORKSPACE_ROOT}/.agents/x"], "one inside the"),
        ([f"{WORKSPACE_ROOT}/a", f"{WORKSPACE_ROOT}/a/b"], "one inside the"),
        ([f"{WORKSPACE_ROOT}/a/b", f"{WORKSPACE_ROOT}/a"], "one inside the"),
        ([f"{WORKSPACE_ROOT}/.codex"], "already denied"),
    ],
)
async def test_a_shape_the_sandbox_could_not_be_built_from_is_refused_and_says_why(
    denied_paths: list[str], expected: str
) -> None:
    """The sentence is asserted, not only the status.

    Three of these break two rules at once. `/session/workspace/.agents/x` is both
    inside a path every Session denies and named after one the runtime protects;
    `/session/workspace/.codex` is both already denied and a protected name. A tenant
    reads the first sentence, and which one that is has to be a decision somebody made
    rather than whichever clause happened to come first in the file.

    The reversed pair is here because the check must not care which of the two the
    tenant wrote first: `nested_deny_pairs` reports a pair, not a direction.
    """
    harness = a_harness()
    app = build_app(harness.platform)

    async with caller(app, TenantId(uuid4())) as client:
        refused = await client.post(
            "/v1/environments",
            json={
                "name": "analysis",
                "runtime_image": IMAGE,
                "denied_paths": denied_paths,
            },
        )

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID], refused.text
    assert refused.json()["error"]["code"] == ErrorCode.REQUEST_INVALID.value
    assert expected in refused.json()["error"]["message"], refused.json()["error"][
        "message"
    ]
    assert any(path in refused.json()["error"]["message"] for path in denied_paths), (
        f"the message names none of {denied_paths}: {refused.json()['message']}"
    )
    assert harness.store.inserted == [], "a refused registration reached the store"


@pytest.mark.parametrize("denied_paths", [list(shape) for shape in _ACCEPTED_SHAPES])
async def test_a_shape_the_sandbox_can_be_built_from_still_registers(
    denied_paths: list[str],
) -> None:
    """The positive half; four of these catch an over-broad clause.

    Measured before the clauses were written: a substring form of the protected-name
    rule refuses `.gitignore`, `my.git` and `.agentsX`; a basename-prefix form refuses
    `.gitignore` and `.agentsX`; and a prefix form of the nesting rule without its
    separator pairs `.agents` with `.agentsX`, and pairs this repository's own `p1` with
    `p10` -- 55 pairs over the `p0`..`p64` fixture where the real predicate finds none.
    `agents` and the sibling pair are here because neither is a dot-path at all, and a
    rule that ended up banning dot-paths would still pass a file that only tried
    dot-paths.

    The empty list is the guard on the union: the platform's own denies go into the same
    check as the tenant's, so a profile that nested two of its own would refuse every
    registration in the system. That is deliberate and loud, and this is where it would
    be heard.
    """
    harness = a_harness()
    app = build_app(harness.platform)

    async with caller(app, TenantId(uuid4())) as client:
        created = await client.post(
            "/v1/environments",
            json={
                "name": "analysis",
                "runtime_image": IMAGE,
                "denied_paths": denied_paths,
            },
        )
        assert created.status_code == 201, created.text
        read = await client.get(f"/v1/environments/{created.json()['id']}")

    assert read.status_code == 200, read.text
    assert read.json()["denied_paths"] == denied_paths
    assert [str(stored) for stored in harness.store.inserted] == [created.json()["id"]]


def test_the_protected_names_are_not_an_empty_set_and_cover_the_platforms_own() -> None:
    """Two ways the clauses this file grades could quietly stop meaning anything.

    An empty `RUNTIME_PROTECTED_NAMES` makes the protected clause unreachable and every
    case above passes by never entering the branch. And the clause order in
    `parse_environment` is only load-bearing while every path the platform denies under
    the writable root is itself a protected name -- the day one is not, the nesting
    clause's platform arm stops being covered by the protected clause and the ordering
    assertion above stops testing anything.
    """
    assert RUNTIME_PROTECTED_NAMES
    assert ALREADY_DENIED
    for path in ALREADY_DENIED:
        assert path.rsplit("/", 1)[-1] in RUNTIME_PROTECTED_NAMES, (
            f"{path} is denied by the platform and is not a protected name, so the "
            "clause order in parse_environment now hides a case instead of ordering "
            "two answers to it"
        )


def test_a_nested_pair_that_never_met_the_registry_is_refused_by_the_compiler() -> None:
    """The other door, closed by the compiler's floor rather than by the registry.

    Asserted here and not only where that floor lives because the two are one guarantee
    split across two files, and a reader of either alone would have to guess whether the
    other exists.
    """
    with pytest.raises(FloorViolation) as refused:
        _compiled((f"{WORKSPACE_ROOT}/a", f"{WORKSPACE_ROOT}/a/b"))

    assert f"{WORKSPACE_ROOT}/a/b" in str(refused.value)
    assert f"{WORKSPACE_ROOT}/a" in str(refused.value)


def test_a_protected_name_that_never_met_the_registry_is_refused_by_nothing() -> None:
    """An absence, asserted on purpose, and the reason the registry clause is not
    redundant.

    A protected component adds no NESTED pair to the merged deny set, so the floor that
    refuses nesting has nothing to see -- which is why this reads the merged set rather
    than asking whether compilation raised, since a future unrelated floor would turn
    that into a false alarm.

    A floor for protected names is not available either: the platform's own profile
    denies two of them at the workspace root on purpose, and those are pre-created as
    directories by the pod so they collide with nothing. Such a floor would fail against
    the shipped configuration on the day it was added.

    If this case ever starts failing because something downstream did close the gap,
    that is good news and this test is what should be rewritten -- not the clause.
    """
    denied = _merged_deny_set(_compiled((f"{WORKSPACE_ROOT}/.git",)))

    assert f"{WORKSPACE_ROOT}/.git" in denied
    assert nested_deny_pairs(denied) == ()


def test_every_shape_the_route_accepts_compiles_to_a_deny_set_with_no_nested_pair() -> (
    None
):
    """The registry's check and the compiler's floor are one predicate over one set, and
    this is where that correspondence is pinned rather than assumed.

    The registry runs it over the platform's denies under the writable root plus the
    tenant's entries; the floor runs it over the rendered document's two deny lists.
    They agree because every tenant path is strictly inside the writable root, so no
    tenant path can pair with a platform deny outside it. A platform deny added under
    another one breaks the correspondence, and this is what would say so.
    """
    for denied_paths in _ACCEPTED_SHAPES:
        assert nested_deny_pairs(_merged_deny_set(_compiled(denied_paths))) == (), (
            f"an accepted shape compiles to a deny set the sandbox cannot build: "
            f"{denied_paths}"
        )
