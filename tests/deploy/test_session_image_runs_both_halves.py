"""The Session image: what its Dockerfile says, and what the built image does.

Two tiers, and the split is deliberate.

Tier A reads `deploy/docker/session.Dockerfile` and `deploy/k8s/session-pod.yaml` off
disk. It needs no daemon and no network and always runs. What it can prove is that the
Dockerfile still *says* the right thing -- that the runtime version has no default, that
the assertion chain sits after the layer that installs the venv and after the USER line,
that the guard names which `codex` resolved rather than only what it printed. Deleting
one of those is a failure here rather than a silent loss. It proves nothing about the
image.

Tier B builds the image and runs things inside it, which is the only thing that proves
the halves resolve. It carries `@pytest.mark.image`, so `addopts` deselects it from the
default run and `pytest -m image` selects it. READ THAT AS: A DEFAULT RUN OF THIS FILE
SAYS NOTHING ABOUT THE IMAGE. It needs a Docker daemon and reaches public.ecr.aws,
registry.npmjs.org and pypi.org, and costs about a minute on a cold cache, which is why
it sits outside the offline gate rather than inside it. Deselected rather than skipped,
which is this repository's decided posture for an opt-in suite: a skip reads as "this
ran and had nothing to say", and this did not run at all.

The ordering assertions are worth reading twice. A first draft of the Dockerfile checked
`command -v codex` inside the layer that installs it -- before the venv was on PATH,
where a binary shadowing it could not exist yet -- and above the USER line, as a uid
that can read and execute files the shipping uid cannot. Either placement is a green
check on a property that had not yet become falsifiable. So the property asserted here
is not that the checks are present but that they run last.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from managed_agent.core.ids import TenantId
from managed_agent.core.registration.environment import Environment, new_environment_id

_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _ROOT / "deploy" / "docker" / "session.Dockerfile"
_IGNOREFILE = _ROOT / "deploy" / "docker" / "session.Dockerfile.dockerignore"
_POD: dict[str, Any] = yaml.safe_load(
    (_ROOT / "deploy" / "k8s" / "session-pod.yaml").read_text()
)

PLACEHOLDER = "map-session@sha256:" + "0" * 64
"""The un-substituted image reference every container spec in the pod carries."""

BASE_IMAGE = "public.ecr.aws/amazonlinux/amazonlinux:2023"


def _instructions(text: str) -> tuple[tuple[str, str], ...]:
    """Split a Dockerfile into (INSTRUCTION, argument) pairs, continuations joined.

    Whole-line comments are dropped *before* the continuations are joined, and the order
    matters: a comment sitting inside a backslash-continued chain would otherwise be
    glued into that chain's text, and an assertion about what a RUN chain contains would
    then be satisfied by a comment that happens to quote it.
    """
    uncommented = [
        line for line in text.splitlines() if not line.strip().startswith("#")
    ]
    joined = re.sub(r"\\\s*\n", " ", "\n".join(uncommented))
    pairs: list[tuple[str, str]] = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        head, _, rest = stripped.partition(" ")
        pairs.append((head.upper(), rest.strip()))
    return tuple(pairs)


_INSTRUCTIONS = _instructions(_DOCKERFILE.read_text())
_RUNS = tuple(argument for head, argument in _INSTRUCTIONS if head == "RUN")


def _index_of_run(fragment: str) -> int:
    """Which RUN chain holds this fragment, asserting there is exactly one."""
    matches = [i for i, run in enumerate(_RUNS) if fragment in run]
    assert len(matches) == 1, (
        f"{fragment!r} appears in {len(matches)} of the {len(_RUNS)} RUN chains; "
        "the ordering assertions below need exactly one"
    )
    return matches[0]


def _first_run_below_user() -> int:
    """The index into the RUN chains of the first one that runs as the shipping uid.

    Every chain at or after it runs as the USER the image declares; every chain before
    it runs as root. A Dockerfile with no USER at all makes the question unanswerable
    rather than trivially true, so that case raises instead of returning a bound every
    index satisfies.
    """
    seen = 0
    for head, _ in _INSTRUCTIONS:
        if head == "USER":
            return seen
        if head == "RUN":
            seen += 1
    raise AssertionError("session.Dockerfile declares no USER instruction")


def test_the_dockerfile_parsed_into_instructions_to_examine() -> None:
    """The positive half. Every case below reads a discovered collection, and a parse
    that produced an empty one would satisfy several of them by having nothing to
    check."""
    assert _INSTRUCTIONS, "no instructions parsed out of session.Dockerfile"
    assert _INSTRUCTIONS[0][0] == "FROM"
    assert len(_RUNS) >= 4, f"only {len(_RUNS)} RUN chains parsed"


def test_the_base_image_is_the_distribution_the_sandbox_was_measured_on() -> None:
    """A different distribution's bwrap is a different measurement of the boundary
    invariants I5-I7 rest on, and nothing in the tree would report the change."""
    assert _INSTRUCTIONS[0][1] == BASE_IMAGE


def test_the_runtime_version_has_no_default_so_a_build_must_name_one() -> None:
    """A defaulted ARG is spelled `NAME=value`; a required one is the bare name."""
    declared = [argument for head, argument in _INSTRUCTIONS if head == "ARG"]
    assert "CODEX_VERSION" in declared, f"ARGs declared: {declared}"
    assert not any(item.startswith("CODEX_VERSION=") for item in declared), (
        "CODEX_VERSION has a default; a build could take whatever npm called latest "
        f"that morning. ARGs declared: {declared}"
    )


@pytest.mark.parametrize(
    "fragment",
    [
        "command -v unshare",
        "command -v bwrap",
        'codex --version | grep -qF "codex-cli ${CODEX_VERSION}"',
        "codex app-server --help",
        'python -c "import managed_agent.composition"',
        "command -v uvicorn",
        "from managed_agent.session_shim.serve import build_shim_app",
    ],
)
def test_every_half_is_asserted_at_build_time(fragment: str) -> None:
    """Each half fails the build rather than the pod.

    This is a text assertion and only claims what the Dockerfile says. That a missing
    half actually fails the build is Tier B's `test_the_image_builds_at_all`, which
    cannot pass while any of these clauses is false.
    """
    assert any(fragment in run for run in _RUNS), (
        f"no RUN chain asserts {fragment!r}; a build missing that half would succeed "
        "and the pod would fail instead"
    )


def test_the_halves_are_asserted_after_the_venv_and_as_the_user_that_ships() -> None:
    """Ordering, not presence, is the property, and there are two orderings.

    `command -v codex` inside the layer that installs codex runs before the venv exists,
    so it cannot see a binary shadowing it on PATH -- and the venv does carry a second,
    differently-versioned codex. Run above the USER line it answers as root, which can
    read and execute files uid 10001 cannot: a venv that landed mode 0700 would satisfy
    every clause there and fail every one of them in the pod. A check placed at either
    spot is green in a world where what it forbids is impossible.
    """
    after_the_venv = _index_of_run("uv sync")
    below_user = _first_run_below_user()
    # Every clause that reads a file the venv owns, not just the version one. Checking
    # one of them leaves the rest free to be hoisted above `USER` by anybody splitting
    # the chain to shorten a layer -- and above it they answer as root, which can read
    # and execute what uid 10001 cannot. That is the exact defect the docstring above
    # says this test exists to prevent, so covering one clause covered the wrong thing.
    for clause in (
        'python -c "import managed_agent.composition"',
        "codex --version | grep -qF",
        "command -v uvicorn",
        "from managed_agent.session_shim.serve import build_shim_app",
    ):
        where = _index_of_run(clause)
        assert where > after_the_venv, f"{clause!r} is asserted before the venv exists"
        assert where >= below_user, (
            f"{clause!r} is asserted above USER, so it answers as root -- a venv that "
            "landed mode 0700 satisfies it here and fails it in the pod"
        )


def test_the_runtime_install_runs_no_package_lifecycle_script() -> None:
    """`--ignore-scripts`, which is the one security clause here nothing else guards.

    Its neighbours are guarded off-file -- `--no-dev` and `--locked` at
    `tests/test_workspace.py` -- and every other load-bearing clause in the Dockerfile
    has a case in this file: the base image, the ARG's absent default, the seven
    assertion fragments, their ordering, the `case` arm, the context allowlist. This one
    had none, while its own comment argues at length that it matters.

    Without it, `npm install -g` executes `preinstall`/`postinstall` from the whole
    transitive tree, as root, with the network up, and bakes the result into the image.
    Somebody debugging an install drops the flag, every other case stays green, and no
    diff after that point shows anything. The failure has no symptom at all until the
    compromised package is the one that ships.
    """
    install = _RUNS[_index_of_run("npm install -g")]

    assert "--ignore-scripts" in install, (
        "the runtime install no longer passes --ignore-scripts, so a postinstall from "
        "any transitive dependency runs as root during the build and is baked into the "
        f"image that every Session pod pulls. The RUN chain is:\n{install}"
    )


def test_the_guard_names_which_codex_resolved_and_not_only_its_version() -> None:
    """`openai-codex` bundles its own codex inside the venv, at another version.

    A version check alone would miss it the moment that copy's version matched. The
    guard has to ask which binary answered, which is what the case arm over
    $VIRTUAL_ENV does.
    """
    guard = _RUNS[_index_of_run("codex --version | grep -qF")]
    assert 'case "$(command -v codex)"' in guard
    assert '"$VIRTUAL_ENV"/*)' in guard


def test_the_build_context_is_an_allowlist_naming_only_what_the_image_needs() -> None:
    """The repository root is gigabytes and the image needs five paths out of it.

    An allowlist excludes the next large directory somebody adds without anybody
    remembering to; a denylist does not.

    The set is pinned rather than merely checked for a leading `*`, so widening the
    context is a deliberate edit here as well as there. The two catalogue-bake entries
    name single files rather than `!tools/**` and `!deploy/**` for the same reason the
    allowlist exists at all -- a directory glob ships whatever anybody adds beside them
    next, which is the failure this file is guarding against one level up.
    """
    lines = [
        line.strip()
        for line in _IGNOREFILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines[0] == "*", f"first pattern is {lines[0]!r}, not a deny-all"
    assert set(lines[1:]) == {
        "!pyproject.toml",
        "!uv.lock",
        "!src/**",
        "!tools/bake_model_catalog.py",
        "!deploy/k8s/model-gateway.yaml",
    }
    globbed = [line for line in lines[1:] if line.endswith("/**")]
    assert globbed == ["!src/**"], (
        "a new directory glob joined the context allowlist: "
        f"{globbed}. Name the files the image needs instead"
    )


def _container_specs() -> tuple[dict[str, Any], ...]:
    return tuple(
        list(_POD["spec"]["initContainers"]) + list(_POD["spec"]["containers"])
    )


def _image_references() -> tuple[str, ...]:
    return tuple(str(spec["image"]) for spec in _container_specs())


def test_the_manifest_declares_the_containers_this_file_grades() -> None:
    """The floor under the set comparison below, which zero containers would satisfy.

    By NAME and in order rather than by count. A count answers the vacuity question and
    nothing else: it was `== 4`, and the fifth container arriving made it fail with
    "expected four, found 5" -- a message about arithmetic, from a check whose real
    subject is which containers exist. Bumping the number would have kept that true and
    would still not have said that a container had been REPLACED rather than added.
    """
    assert [str(spec["name"]) for spec in _container_specs()] == [
        "seed-runtime-home",
        "restore-working-lane",
        "seed-rollout",
        "agent-runtime",
        "session-shim",
    ]


def test_every_container_spec_names_the_same_reference() -> None:
    """One pod, one image. Asserted against the value, not just between the specs.

    Specs agreeing with each other is satisfied by all of them being wrong in the same
    way, and that failure is invisible because no expected value is ever named.
    """
    assert set(_image_references()) == {PLACEHOLDER}


def test_the_reference_is_one_core_environment_accepts() -> None:
    """The placeholder is checked by the parser that decides the real thing.

    A Session's pod is started from CompiledConfig.runtime_image, which comes from a
    registered Environment, whose runtime_image must be digest-pinned. Re-implementing
    that rule here would give it a second home free to disagree with the first.
    """
    Environment(
        id=new_environment_id(),
        tenant_id=TenantId(uuid4()),
        name="the pod manifest's un-substituted placeholder",
        runtime_image=_image_references()[0],
        denied_paths=(),
    )


def test_no_container_spec_names_a_mutable_tag() -> None:
    """`:latest` is unpushable more than once to an IMMUTABLE repository, and is the
    exact string Environment rejects."""
    for reference in _image_references():
        assert "@sha256:" in reference, f"{reference!r} is not digest-pinned"
        assert ":latest" not in reference


CODEX_VERSION = "0.149.0"
PLATFORM = "linux/amd64"
IMAGE_TAG = "map-session:pytest"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=1800)


@pytest.fixture(scope="module")
def image() -> str:
    """Build the Session image and yield its tag.

    --platform linux/amd64 because the nodegroup is t3.medium on
    AL2023_x86_64_STANDARD. On an arm64 developer machine this runs under emulation,
    which is slower but keeps the tested image byte-identical to the one a node would
    pull -- and an image built for the host's own architecture would fail on the node
    with an exec format error that says nothing about why.

    Module-scoped: the build is the expensive part and every case below reads the same
    image.
    """
    built = _run(
        [
            "docker",
            "build",
            "--platform",
            PLATFORM,
            "--build-arg",
            f"CODEX_VERSION={CODEX_VERSION}",
            "-f",
            str(_DOCKERFILE),
            "-t",
            IMAGE_TAG,
            str(_ROOT),
        ]
    )
    if built.returncode != 0:
        pytest.fail(
            f"docker build failed rc={built.returncode}\n{built.stderr[-4000:]}"
        )
    return IMAGE_TAG


def _in_image(tag: str, shell_command: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            PLATFORM,
            "--entrypoint",
            "/bin/sh",
            tag,
            "-c",
            shell_command,
        ]
    )


@pytest.mark.image
def test_the_image_builds_at_all(image: str) -> None:
    """The checkpoint. The build is the assertion, not a precondition of one.

    Its final RUN chains eight checks with &&, so a build that produced this tag is an
    image in which codex resolved at the pinned version from outside the venv, the
    app-server subcommand existed, managed_agent.composition imported, uvicorn resolved
    and the shim's factory was importable. The cases below exist to say which of those
    broke.
    """
    assert image == IMAGE_TAG


@pytest.mark.image
def test_the_codex_the_pod_runs_is_the_pinned_one_outside_the_venv(image: str) -> None:
    resolved = _in_image(image, "command -v codex")
    assert resolved.returncode == 0, resolved.stderr
    assert not resolved.stdout.strip().startswith("/opt/map/venv/"), (
        f"codex resolves to {resolved.stdout.strip()!r}, inside the venv -- the "
        "bundled 0.147.0 copy, not the pinned install"
    )
    version = _in_image(image, "codex --version")
    assert f"codex-cli {CODEX_VERSION}" in version.stdout, version.stdout


@pytest.mark.image
def test_the_second_bundled_codex_is_present_and_is_not_what_codex_resolves_to(
    image: str,
) -> None:
    """The positive half first: the second binary really is in there.

    `openai-codex` is declared in pyproject.toml, nothing in this repository imports it,
    and it drags in openai-codex-cli-bin -- a complete codex at a different version,
    built against musl, inside the venv. Asserting only that `codex` resolves elsewhere
    would pass just as well if the bundled copy had silently disappeared, which is the
    cheapest way for a negative assertion to be satisfied.
    """
    bundled = _in_image(
        image,
        'python -c "import codex_cli_bin, sys; '
        'sys.stdout.write(str(codex_cli_bin.bundled_codex_path()))"',
    )
    assert bundled.returncode == 0, (
        "the bundled codex is gone; this case no longer guards anything and the "
        f"dependency may have changed shape. stderr={bundled.stderr}"
    )
    resolved = _in_image(image, "command -v codex")
    assert bundled.stdout.strip() != resolved.stdout.strip()


@pytest.mark.image
def test_the_package_and_the_shim_entry_point_both_import(image: str) -> None:
    """Both halves of the pod, in the one image. The factory is imported, not called:
    it opens the runtime connection in its lifespan, so importing it needs no socket.

    `managed_agent.composition` rather than `managed_agent`, because the package's
    __init__.py is empty -- importing it loads no third-party module, so it cannot fail
    for a dependency the install was missing.
    """
    imported = _in_image(
        image,
        'python -c "import managed_agent.composition; '
        "from managed_agent.session_shim.serve import build_shim_app; "
        'assert callable(build_shim_app)" && command -v uvicorn',
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip().endswith("uvicorn")


@pytest.mark.image
def test_the_default_user_is_the_unprivileged_uid(image: str) -> None:
    """The pod sets runAsUser, and the image carries the same uid so that running it
    outside Kubernetes cannot do anything an unprivileged Session pod could not."""
    identity = _in_image(image, "id -u; id -g")
    assert identity.returncode == 0, identity.stderr
    assert identity.stdout.split() == ["10001", "10001"]


@pytest.mark.image
def test_a_build_that_names_no_runtime_version_is_refused() -> None:
    """Built to a throwaway tag, so a refused build cannot leave the tag the other
    cases read pointing at an image whose runtime version nothing recorded."""
    built = _run(
        [
            "docker",
            "build",
            "--platform",
            PLATFORM,
            "-f",
            str(_DOCKERFILE),
            "-t",
            "map-session:pytest-unpinned",
            str(_ROOT),
        ]
    )
    assert built.returncode != 0, (
        "the build succeeded with no CODEX_VERSION; the pod could run whatever npm "
        "called latest that morning and nothing in the tree would say which"
    )
    # Which failure, and not merely that there was one. This is the only case here that
    # takes no `image` fixture, so it is the one that stays green while everything else
    # goes red -- and a slow dnf mirror, a rate-limited `public.ecr.aws`, or a moved
    # base tag all fail the build early and satisfy a bare `returncode != 0`. It would
    # then report that the no-default guard works on a run that never reached it.
    assert 'test -n "$CODEX_VERSION"' in built.stderr, (
        "the build failed, but not at the CODEX_VERSION guard, so this case proves "
        f"nothing about it. stderr tail:\n{built.stderr[-1500:]}"
    )
