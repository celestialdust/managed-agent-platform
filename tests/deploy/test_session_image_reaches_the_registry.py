"""The Session image in the registry a node pulls from, and the tag that names it.

Two tiers, and neither of them does the thing that matters most.

The always-running tier interrogates `deploy/docker/push-session-image.sh`: that a shell
can parse it, that the tag it derives names the commit and the runtime version, and that
it refuses a tree whose build inputs are uncommitted. Both behavioural cases run the
script against a throwaway git repository built in `tmp_path` rather than against this
checkout, so neither of them turns red because a developer has an edit open in `src/`.
Nothing in this tier reaches a daemon, an account or a registry.

The gated tier reads the registry, and is skipped unless `MAP_ECR_TESTS=1` -- an
environment check rather than a registered pytest marker, because `pyproject.toml` has
one writer. A skip means the registry was not consulted at all, so a run that skipped
says nothing about whether `map/session-shim` holds anything.

What no tier does is push. A node pulls from the registry and never from a developer's
daemon, and the account is shared, so the push is one command a person runs:

    CODEX_VERSION=0.149.0 deploy/docker/push-session-image.sh

It prints one digest-pinned reference, which is then registered as a sandbox shape --
`POST /v1/environments` carrying it as `runtime_image`. That is the only route to
`CompiledConfig.runtime_image`, which is what a Session's pod is started from. The last
case below is what goes green once both have happened, and is red until then.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

import httpx
import pytest

from managed_agent.composition import build
from managed_agent.control.api.app import create_app
from managed_agent.control.api.request.tenancy import TENANT_HEADER
from managed_agent.control.catalog.environments import resolve_environment
from managed_agent.control.pod_config.compiler import compile_session_config
from managed_agent.core.ids import TenantId, new_definition_id, new_session_id
from managed_agent.core.registration.definition import (
    AgentDefinition,
    SkillsRevision,
)
from managed_agent.core.registration.environment import EnvironmentId
from managed_agent.core.session.session import SessionRecord

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "deploy" / "docker" / "push-session-image.sh"
_ECR_GATE = "MAP_ECR_TESTS"

REPOSITORY = "map/session-shim"
GATEWAY_URL = "http://tool-gateway.default.svc.cluster.local:8080"

# Where a Session pod reaches the Model Gateway. The `/v1` is load-bearing at both ends:
# the Agent Runtime POSTs `{base_url}/responses`, and the Gateway's router mounts
# `POST /v1/responses`.
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"

# The Gateway's signing key and the token's deadline, which the compiler takes from
# its caller and never defaults. Literals, so no case here can expire mid-run.
SESSION_TOKEN_KEY = b"a signing key that is thirty-two"
SESSION_TOKEN_EXPIRY = 4102444800

_TAG = re.compile(r"^git-[0-9a-f]{40}-codex-[0-9A-Za-z.+_-]+$")
"""The one spelling of the tag form, read by both tiers.

The script derives a tag and the registry hands tags back, and the two have to agree
about the shape or the second tier is grading a different rule from the first.
"""

_A_RUNTIME_VERSION = "0.149.0"

# Every spelling buildx reads as false; see docs/lessons.md for why matching the literal
# "false" is not enough. A value outside this set either enables provenance or fails to
# parse, and both end with ECR refusing the tag rather than with a clean error.
_BUILDX_FALSE = frozenset({"false", "False", "FALSE", "f", "F", "0"})

# The definition a Session pins, for the one field the compiler reads off it: the model.
# The provider is not here because it is not the definition's to name -- every model
# call leaves a Session pod through the Model Gateway.
A_DEFINITION = AgentDefinition(
    name="slr-reviewer",
    instructions="Extract findings and name the source for each.",
    model="gpt-5-codex",
    skills_repository="git@github.com:acme/skills.git",
    skills_revision=SkillsRevision("0" * 39 + "a"),
)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )


# The five paths the script's dirty check names, and one committed file for each. `src`
# is a directory pathspec, so it is represented by a file underneath it -- git tracks no
# empty directory, and a fixture with an empty `src/` would leave that clause unreached
# too.
_BUILD_INPUTS: Final[dict[str, str]] = {
    "pyproject.toml": 'version = "0.1.0"\n',
    "uv.lock": "version = 1\n",
    "src/placeholder.py": "x = 1\n",
    "deploy/docker/session.Dockerfile": "FROM scratch\n",
    "deploy/docker/session.Dockerfile.dockerignore": "*\n!src\n",
}


def _pathspecs_the_script_watches() -> set[str]:
    """The dirty check's pathspec list, read out of the script rather than repeated.

    Repeating it here would be a second copy free to disagree with the first, and the
    direction it would disagree in is the dangerous one: a sixth pathspec added to the
    script would be ungraded by a test that still lists five and still passes. Parsed
    from the `git ... status --porcelain --` invocation and its backslash continuations.
    """
    text = _SCRIPT.read_text()
    start = text.index("status --porcelain --")
    invocation = ""
    for line in text[start:].splitlines():
        invocation += " " + line.rstrip("\\").strip()
        if not line.rstrip().endswith("\\"):
            break
    # The invocation sits inside `$( ... )`, so the last word carries the closing
    # paren. Stripped rather than matched with a paren-aware parser: one character of
    # shell syntax does not justify one, and a pathspec ending in `)` would be a
    # filename this repository will never have.
    words = invocation.replace("status --porcelain --", "").split()
    return {word.rstrip(")") for word in words}


def _a_repository_holding_the_script(tmp_path: Path) -> Path:
    """A throwaway git repository with the script at the path it expects to sit at.

    The script finds the tree from its own location, so the copy has to live at
    `deploy/docker/` inside the repository it interrogates. Built here rather than run
    against this checkout because the rule under test is about uncommitted files, and a
    case that read this checkout's git status would pass or fail on whatever the person
    running it happened to have open.
    """
    (tmp_path / "deploy" / "docker").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    copied = tmp_path / "deploy" / "docker" / _SCRIPT.name
    copied.write_bytes(_SCRIPT.read_bytes())
    copied.chmod(0o755)
    # Every build input the script watches is committed, not only the two the first
    # version of this fixture happened to create. `git status --porcelain -- <path>` is
    # empty for a path that does not exist, so a fixture missing an input makes the
    # guard's clause for that input pass by never being reached -- and deleting two
    # pathspecs from the script left this tier green at `5 passed, 3 skipped`.
    for relative, body in _BUILD_INPUTS.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "probe@example.invalid")
    _git(tmp_path, "config", "user.name", "probe")
    _git(tmp_path, "add", *_BUILD_INPUTS, "deploy/docker/push-session-image.sh")
    _git(tmp_path, "commit", "--quiet", "-m", "the inputs a tag names")
    return tmp_path


def _print_the_tag(repository: Path) -> subprocess.CompletedProcess[str]:
    """Run the script's tag half: no daemon, no account, no push."""
    return subprocess.run(
        [str(repository / "deploy" / "docker" / _SCRIPT.name)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "CODEX_VERSION": _A_RUNTIME_VERSION,
            "MAP_PRINT_TAG_ONLY": "1",
        },
    )


def test_the_push_script_is_one_a_shell_can_parse() -> None:
    """`sh -n` is the only gate on this file that runs everywhere.

    `ruff` and `mypy` do not see a shell script, and `shellcheck` -- which it also
    passes -- is on no dependency list here, so it cannot be asserted on a machine that
    may not have it. Without this case a syntax error would first be reported by the
    person who ran the script, against a shared registry.
    """
    parsed = subprocess.run(
        ["sh", "-n", str(_SCRIPT)], capture_output=True, text=True, timeout=60
    )
    assert parsed.returncode == 0, parsed.stderr


def test_the_tag_names_the_commit_and_the_runtime_version_it_was_built_from(
    tmp_path: Path,
) -> None:
    """Nothing else goes into it, and both halves are load-bearing.

    The commit, because everything the build reads out of the tree is at one commit. The
    runtime version, because it is a build argument with no default: two images from one
    commit at two runtime versions are two byte sets, and against an IMMUTABLE
    repository they would contend for one tag and the second push would be refused.
    """
    repository = _a_repository_holding_the_script(tmp_path)
    head = _git(repository, "rev-parse", "HEAD").stdout.strip()

    printed = _print_the_tag(repository)

    assert printed.returncode == 0, printed.stderr
    tag = printed.stdout.strip()
    assert tag == f"git-{head}-codex-{_A_RUNTIME_VERSION}"
    assert _TAG.match(tag), "the registry tier below reads this same form"


@pytest.mark.parametrize("dirtied", sorted(_BUILD_INPUTS))
def test_the_script_refuses_a_tree_whose_build_inputs_are_uncommitted(
    tmp_path: Path, dirtied: str
) -> None:
    """A tag naming a commit that is not what was built is worse than no tag.

    Every build input, not one of them. The version this replaces dirtied `src` only,
    and the fixture created neither `uv.lock` nor either Dockerfile -- so four of the
    five clauses were satisfied by paths that did not exist, and deleting two pathspecs
    from the script left this tier green at `5 passed, 3 skipped`. The consequence of
    that gap is not a failing build: it is bytes pushed to an IMMUTABLE repository under
    a tag naming a commit that does not contain them, which cannot be undone.

    Asserted through the print-tag path on purpose: the refusal runs before that seam,
    so the seam cannot be used to obtain a tag for a tree not allowed to push one.
    """
    repository = _a_repository_holding_the_script(tmp_path)
    target = repository / dirtied
    target.write_text(target.read_text() + "# edited after the commit\n")

    refused = _print_the_tag(repository)

    assert refused.returncode == 1, (
        f"dirtying {dirtied} did not stop the script; stdout={refused.stdout!r}. That "
        "path is in the dirty check's pathspec list and is not being graded."
    )
    assert dirtied in refused.stderr, (
        f"the refusal does not name {dirtied}: {refused.stderr!r}. Whoever hits this "
        "needs to know which file to commit."
    )


def test_the_fixture_commits_every_path_the_dirty_check_watches() -> None:
    """The case above is only as complete as the set it is parametrised over.

    Read out of the script rather than repeated, so a sixth pathspec added there fails
    here instead of being silently ungraded -- which is the shape that let four of five
    clauses go unexercised for a slice and a half. See `docs/lessons.md`.
    """
    watched = _pathspecs_the_script_watches()
    covered = {
        path.split("/")[0] if path.startswith("src/") else path
        for path in _BUILD_INPUTS
    }
    assert watched == covered, (
        f"the script watches {sorted(watched)} and this file exercises "
        f"{sorted(covered)}. Add a committed file for each missing pathspec to "
        "_BUILD_INPUTS; a pathspec with no file in the fixture is a clause that "
        "cannot fail."
    )


def test_the_script_ignores_an_uncommitted_file_the_image_cannot_see(
    tmp_path: Path,
) -> None:
    """The paired case, and the reason the check names paths instead of the whole tree.

    The build context is an allowlist of `pyproject.toml`, `uv.lock` and `src/**`, so an
    uncommitted document elsewhere changes no byte of the image. Without this case the
    one above is satisfied by a script that refuses every tree, which would make the
    push unrunnable in a repository whose working notes are always in flight.
    """
    repository = _a_repository_holding_the_script(tmp_path)
    (repository / "docs").mkdir()
    (repository / "docs" / "notes.md").write_text("in flight\n")

    printed = _print_the_tag(repository)

    assert printed.returncode == 0, printed.stderr
    assert _TAG.match(printed.stdout.strip())


def test_the_script_puts_one_manifest_under_the_tag() -> None:
    """One tag, one manifest, one PUT. Both flags that hold that are asserted here.

    Two ways to break it, both measured against this registry on 2026-08-22 and both
    ending in a bare `400 Bad Request` after the image had already landed -- the failure
    that costs somebody a second push at a second commit for nothing.

    Dropping `--provenance=false` makes buildx export an attestation manifest and an
    index over it, and bind the tag to the index; ECR then refuses one of that index's
    two children under that tag. Using `--push` instead of `--load` plus one
    `docker push` makes Docker Desktop's `docker` builder driver push from buildkit and
    again from the daemon copy, which is two PUTs of a tag that allows one.

    A text assertion, and weak for it: it cannot tell that a push works, only that
    neither of those two shapes has come back. The behavioural check is the gated tier
    below -- and the failure this guards against is invisible there, because the image
    reaches the registry either way and only the exit status lies.
    """
    # Comment lines are dropped first, because the comment above the fix names the flag
    # it is there to keep out and a text search over the whole file would find it there.
    commands = [
        line
        for line in _SCRIPT.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not [line for line in commands if "--push" in line], (
        "`--push` on the buildx invocation makes the docker driver PUT the tag twice; "
        "build with `--load` and push with one `docker push`"
    )
    assert len([line for line in commands if line.startswith("docker push ")]) == 1, (
        commands
    )
    provenance = [
        found.group(1)
        for line in commands
        if (found := re.search(r"--provenance=(\S+)", line)) is not None
    ]
    assert provenance, (
        "no buildx line passes --provenance at all, so buildx binds the tag to an "
        "index over an attestation manifest and ECR refuses its children under it"
    )
    assert all(value in _BUILDX_FALSE for value in provenance), (
        f"--provenance={provenance} does not disable provenance. Matched as a value "
        "rather than as the literal 'false', because buildx reads six spellings of it."
    )


requires_the_registry = pytest.mark.skipif(
    os.environ.get(_ECR_GATE) != "1",
    reason=(
        f"reading the registry is opt-in: set {_ECR_GATE}=1 to run it. It needs the "
        "`aws` CLI and credentials for the account the Session pod's repository lives "
        "in. SKIPPED MEANS THE REGISTRY WAS NOT CONSULTED -- nothing here says whether "
        f"{REPOSITORY} holds an image."
    ),
)


def _aws(*arguments: str) -> Any:
    """One read-only ECR call, parsed. Never a call that writes."""
    answered = subprocess.run(
        ["aws", "ecr", *arguments, "--output", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert answered.returncode == 0, answered.stderr
    return json.loads(answered.stdout)


def _images() -> list[Any]:
    described = _aws("describe-images", "--repository-name", REPOSITORY)
    details = described["imageDetails"]
    assert isinstance(details, list)
    return details


def _newest_tagged_reference() -> str:
    """The digest-pinned reference for the most recently pushed tagged image.

    By digest rather than by tag, because a digest is what `Environment` accepts and a
    tag is what it refuses. The repository URI comes from the registry rather than being
    spelled here, so this file carries no account id and no region.
    """
    tagged = [detail for detail in _images() if detail.get("imageTags")]
    assert tagged, f"{REPOSITORY} holds no tagged image"
    newest = max(tagged, key=lambda detail: str(detail["imagePushedAt"]))
    described = _aws("describe-repositories", "--repository-names", REPOSITORY)
    return f"{described['repositories'][0]['repositoryUri']}@{newest['imageDigest']}"


@requires_the_registry
def test_the_registry_holds_a_session_image() -> None:
    """The floor under the two cases below, and the fact a real pod cannot do
    without: an empty repository is not a slow pull, it is no pull at all."""
    assert _images(), (
        f"{REPOSITORY} is empty. A node pulls from the registry and never from a "
        "developer's daemon, so no Session pod can start until somebody runs "
        "`CODEX_VERSION=<version> deploy/docker/push-session-image.sh`"
    )


@requires_the_registry
def test_every_tag_in_the_repository_names_a_commit_and_a_runtime_version() -> None:
    """Nothing enforces the tag form on the way in, so it is checked on the way out.

    A hand-rolled `latest` pushed once would sit in an IMMUTABLE repository forever,
    naming bytes nobody can trace to a commit, and the digest-only production path would
    never notice.
    """
    tags = [tag for detail in _images() for tag in detail.get("imageTags", [])]
    assert tags, f"{REPOSITORY} holds no tagged image; the push records no provenance"
    for tag in tags:
        assert _TAG.match(tag), (
            f"{tag!r} was not produced by deploy/docker/push-session-image.sh, so the "
            "bytes it names cannot be traced to a commit and a runtime version"
        )


@requires_the_registry
async def test_the_digest_the_registry_holds_is_the_image_a_pod_is_started_from(
    database_url: str,
) -> None:
    """The whole custody chain, from the registry to the value a pod is created with.

    A digest read out of ECR is registered as a sandbox shape through the tenant route,
    resolved back out of the real relational store, and compiled -- and the assertion is
    on `CompiledConfig.runtime_image`, which is the field `PodRunner.ensure` is handed.
    Registered through the route rather than inserted, because the route is where
    `parse_environment` runs and a row written past it is a row the platform would not
    accept today.
    """
    reference = _newest_tagged_reference()
    tenant = TenantId(uuid4())
    platform, engine = build(database_url)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(platform)),
            base_url="http://platform",
            headers={TENANT_HEADER: str(tenant)},
        ) as caller:
            registered = await caller.post(
                "/v1/environments",
                json={
                    "name": "the pushed session image",
                    "runtime_image": reference,
                    "denied_paths": [],
                },
            )
        assert registered.status_code == 201, registered.text
        environment = await resolve_environment(
            platform.environment_store,
            EnvironmentId(UUID(registered.json()["id"])),
            tenant,
        )
        compiled = compile_session_config(
            SessionRecord(
                id=new_session_id(),
                tenant_id=tenant,
                definition_id=new_definition_id(),
                definition_revision="1",
                grant=frozenset(),
                scope=(),
                budget_minor_units=10_000,
                budget_currency="USD",
                retention_days=30,
            ),
            tool_gateway_url=GATEWAY_URL,
            model_gateway_url=MODEL_GATEWAY_URL,
            definition=A_DEFINITION,
            environment=environment,
            session_token_key=SESSION_TOKEN_KEY,
            session_token_expiry_epoch_s=SESSION_TOKEN_EXPIRY,
        )
    finally:
        await engine.dispose()

    assert compiled.runtime_image == reference
    assert "@sha256:" in compiled.runtime_image
