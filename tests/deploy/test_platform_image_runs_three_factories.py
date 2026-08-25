"""The platform image: what its Dockerfile says, and what the built image does.

Two tiers, and the split is deliberate.

Tier A reads `deploy/docker/platform.Dockerfile` and its dockerignore off disk. It needs
no daemon and no network and always runs. What it can prove is that the file still says
the right thing -- that the sandbox and the runtime are not installed, that the bundled
runtime binary is deleted in the layer that installed it rather than in a later one
where the deletion would remove nothing, that the assertion chain is last and sits
below the USER line. Deleting one of those is a failure here rather than a silent loss.

Tier B builds the image and runs things inside it, which is the only thing that proves
the three factories import. It carries `@pytest.mark.image`, so `addopts` deselects it
from the default run and `pytest -m image` selects it. READ THAT AS: A DEFAULT RUN OF
THIS FILE SAYS NOTHING ABOUT THE IMAGE. It needs a Docker daemon and reaches
public.ecr.aws and pypi.org. Deselected rather than skipped, which is this repository's
decided posture for an opt-in suite: a skip reads as "this ran and had nothing to say",
and this did not run.

NOT PROVEN by any tier here: that the image starts a working service. Every factory is
imported and none is called -- `build_app` opens a connection pool and
`create_gateway_app` builds an MCP session manager, so calling either needs a database
and a signing key. MAP-63 and MAP-64 are where a running process is graded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _ROOT / "deploy" / "docker" / "platform.Dockerfile"
_IGNOREFILE = _ROOT / "deploy" / "docker" / "platform.Dockerfile.dockerignore"

PLATFORM = "linux/amd64"
IMAGE_TAG = "map-platform:pytest"

# The uid every platform manifest runs its container as. Read from a manifest in
# test_control_plane_manifest.py; here it is the number the image has to carry a passwd
# entry for, and the two files are compared by MAP-63's suite rather than by this one.
SHIPPING_UID = "10002"

# The package whose payload the venv layer deletes, and the module whose importability
# proves the deletion stayed safe.
DELETED_PACKAGE = "codex_cli_bin"
WIDEST_IMPORT = "managed_agent.composition"


def _instructions(text: str) -> tuple[tuple[str, str], ...]:
    """Split a Dockerfile into (INSTRUCTION, argument) pairs, continuations joined.

    Whole-line comments are dropped before continuations are joined, so a comment
    inside a continued RUN does not swallow the next line of it.
    """
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    joined: list[str] = []
    for line in lines:
        if joined and joined[-1].endswith("\\"):
            joined[-1] = joined[-1][:-1].rstrip() + " " + line.strip()
        else:
            joined.append(line.rstrip())
    pairs: list[tuple[str, str]] = []
    for entry in joined:
        instruction, _, argument = entry.partition(" ")
        pairs.append((instruction.upper(), argument.strip()))
    return tuple(pairs)


def test_the_dockerfile_installs_no_sandbox_and_no_runtime() -> None:
    """The three absences this image is defined by.

    Each is asserted against the install line rather than against the whole file,
    because the file's comments name all three and a substring search over the text is
    satisfied by the comment that explains why they are not there.
    """
    installs = [
        argument
        for instruction, argument in _instructions(_DOCKERFILE.read_text())
        if instruction == "RUN" and "dnf -y install" in argument
    ]
    assert len(installs) == 1, "one dnf install layer, or this grades one of N"
    for absent in ("bubblewrap", "util-linux", "nodejs", "npm"):
        assert absent not in installs[0], (
            f"{absent} is installed; no service here uses it"
        )
    assert "npm install" not in _DOCKERFILE.read_text()


def test_the_assertion_chain_is_the_last_instruction_and_sits_below_the_user_line() -> (
    None
):
    """A check that runs as root, or before the venv is on PATH, cannot fail.

    root can read and execute files uid 10002 cannot, so a venv that landed mode 0700
    satisfies every clause as root and fails every one in the pod, with the build green.
    """
    pairs = _instructions(_DOCKERFILE.read_text())
    runs = [
        index for index, (instruction, _) in enumerate(pairs) if instruction == "RUN"
    ]
    users = [
        index for index, (instruction, _) in enumerate(pairs) if instruction == "USER"
    ]

    assert users, "no USER line; the image would run as root"
    assert runs[-1] > users[-1], "the assertion layer runs above USER, so as root"
    assert runs[-1] == len(pairs) - 1, "something runs after the assertions"

    chain = pairs[runs[-1]][1]
    for clause in (
        "command -v uvicorn",
        "command -v alembic",
        "alembic -c /opt/map/alembic.ini heads",
        "from managed_agent.asgi import build_app",
        "from managed_agent.gateway.tool.server import create_gateway_app",
        f"import {WIDEST_IMPORT}",
        "! command -v codex",
        "! command -v bwrap",
        "! command -v unshare",
    ):
        assert clause in chain, f"{clause!r} is not asserted at build time"


def test_the_bundled_runtime_binary_is_deleted_in_the_layer_that_installed_it() -> None:
    """A deletion in a later layer removes nothing from the image.

    This is the one line that is not session.Dockerfile's, and it is 47% of the image
    (measured: 260,057,681 bytes with it, 136,903,657 without). A well-meaning tidy-up
    that moved it into its own RUN would leave the size unchanged and the diff looking
    better.
    """
    syncing = [
        argument
        for instruction, argument in _instructions(_DOCKERFILE.read_text())
        if instruction == "RUN" and "uv sync" in argument
    ]
    assert len(syncing) == 1
    assert DELETED_PACKAGE in syncing[0], (
        f"{DELETED_PACKAGE} is not deleted in the layer that installed it"
    )


def test_the_migration_runner_and_its_revisions_are_both_copied_in() -> None:
    """alembic is in the wheel's dependencies; the revisions are not in the wheel.

    `pyproject.toml` packages `src/managed_agent` only, so an image built the Session
    image's way has the runner on PATH and nothing for it to run -- and `alembic upgrade
    head` then reports success having applied nothing.
    """
    pairs = _instructions(_DOCKERFILE.read_text())
    copied = " ".join(
        argument for instruction, argument in pairs if instruction == "COPY"
    )
    assert "alembic.ini" in copied
    assert "migrations" in copied

    workdirs = [argument for instruction, argument in pairs if instruction == "WORKDIR"]
    assert workdirs == ["/opt/map"], (
        "alembic resolves script_location against the working directory, measured -- "
        "from anywhere else it answers CommandError: Path doesn't exist: migrations"
    )


def test_the_context_admits_exactly_the_five_inputs_the_dockerfile_reads() -> None:
    """An allowlist that drifts from COPY lines fails the build or ships the repo."""
    admitted = {
        line[1:]
        for line in _IGNOREFILE.read_text().splitlines()
        if line.startswith("!")
    }
    assert admitted == {
        "pyproject.toml",
        "uv.lock",
        "src/**",
        "alembic.ini",
        "migrations/**",
    }
    assert "*" in _IGNOREFILE.read_text().splitlines(), "not an allowlist"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def image() -> str:
    """Build the image once for every case below.

    --platform linux/amd64 because the nodegroup is t3.medium on AL2023_x86_64_STANDARD.
    On an arm64 developer machine this runs under emulation, which is slower and keeps
    the tested image byte-identical to the one a node would pull; an image built for the
    host's architecture fails on the node with an exec format error that says nothing
    about why.
    """
    built = _run(
        [
            "docker",
            "build",
            "--platform",
            PLATFORM,
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

    Its final RUN chains nine clauses with &&, so an image carrying this tag is one in
    which uvicorn and alembic resolved, alembic read its revisions, all three imports
    succeeded, and codex, bwrap and unshare each resolved to nothing. The cases below
    exist to say which of those broke.
    """
    assert image == IMAGE_TAG


@pytest.mark.image
def test_every_factory_the_manifests_name_imports(image: str) -> None:
    """The two factories that exist on main, imported rather than called.

    MAP-64's suite adds `composition:tool_gateway_app` when that name exists; it is
    absent here on purpose, so this file does not go red for a slice it does not own.
    """
    imported = _in_image(
        image,
        'python -c "from managed_agent.asgi import build_app; '
        "from managed_agent.gateway.tool.server import create_gateway_app; "
        f"import {WIDEST_IMPORT}; "
        'assert callable(build_app) and callable(create_gateway_app)"',
    )
    assert imported.returncode == 0, imported.stderr


@pytest.mark.image
def test_the_deleted_runtime_binary_is_gone_and_the_package_still_imports(
    image: str,
) -> None:
    """Both halves, because either alone passes for the wrong reason.

    The absence alone would be satisfied by a lock refresh that dropped the dependency
    entirely; the import alone would be satisfied by an image that kept 123 MB it cannot
    execute.
    """
    payload = _in_image(
        image, f"ls /opt/map/venv/lib/python3.12/site-packages/{DELETED_PACKAGE}"
    )
    assert payload.returncode != 0, (
        f"{DELETED_PACKAGE} is still in the venv; the image carries 123,154,024 bytes "
        "of a musl-linked runtime no service here can execute"
    )
    still_imports = _in_image(image, f'python -c "import {WIDEST_IMPORT}"')
    assert still_imports.returncode == 0, (
        "deleting the bundled runtime broke the widest import in the package, which "
        f"means something now imports {DELETED_PACKAGE}: {still_imports.stderr}"
    )


@pytest.mark.image
def test_the_migration_runner_reads_its_revisions_inside_the_image(image: str) -> None:
    """One head, and it is the same head the repository's own chain reports.

    Read with `heads` rather than `current`, because `current` needs a database and this
    case is about whether the revisions reached the image at all.
    """
    inside = _in_image(image, "alembic -c /opt/map/alembic.ini heads")
    assert inside.returncode == 0, inside.stderr
    outside = _run(["uv", "run", "alembic", "-c", str(_ROOT / "alembic.ini"), "heads"])
    assert outside.returncode == 0, outside.stderr
    assert inside.stdout.strip() == outside.stdout.strip(), (
        f"the image is at {inside.stdout.strip()!r} and the tree at "
        f"{outside.stdout.strip()!r}; a Job run from this image would upgrade to the "
        "wrong schema"
    )
    assert "(head)" in inside.stdout


@pytest.mark.image
def test_the_default_user_is_the_uid_the_platform_manifests_run_as(image: str) -> None:
    """10002, and with a passwd entry, so HOME resolves.

    A numeric USER with no entry has no HOME, which is the failure the Session image's
    own passwd entry was written to avoid -- and it runs as 10001, so copying that
    image's user layer here would produce exactly it.
    """
    identity = _in_image(image, "id -u; id -g; echo HOME=$HOME; pwd")
    assert identity.returncode == 0, identity.stderr
    lines = identity.stdout.split()
    assert lines[0] == SHIPPING_UID and lines[1] == SHIPPING_UID
    assert "HOME=/home/map" in identity.stdout
    assert identity.stdout.strip().endswith("/opt/map")


@pytest.mark.image
def test_no_sandbox_and_no_runtime_resolve_inside_the_image(image: str) -> None:
    """Asserted at run time as well as at build time, because the build assertion is one
    layer's exit status and a later layer could reintroduce any of them."""
    for absent in ("codex", "bwrap", "unshare", "npm", "node"):
        found = _in_image(image, f"command -v {absent}")
        assert found.returncode != 0, f"{absent} resolves to {found.stdout.strip()}"
