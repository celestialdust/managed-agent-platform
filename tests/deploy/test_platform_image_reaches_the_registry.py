"""The platform image in the three registries the manifests pull from.

Two tiers, and neither of them pushes.

The always-running tier interrogates `deploy/docker/push-platform-image.sh`: that a
shell can parse it, that the tag it derives names the commit and nothing else, that it
refuses a tree whose build inputs are uncommitted, and that the build disables
provenance and pushes once per repository. Both behavioural cases run the script against
a throwaway git repository built in `tmp_path` rather than against this checkout, so
neither turns red because a developer has an edit open in `src/`.

The gated tier reads the registries, and is skipped unless `MAP_ECR_TESTS=1` -- an
environment check rather than a registered pytest marker, because `pyproject.toml` has
one writer. A skip means no registry was consulted, so a run that skipped says nothing
about whether anything holds an image.

The push is one command a person runs, because a node pulls from a registry and the
account is shared:

    deploy/docker/push-platform-image.sh

It prints three digest-pinned references. `deploy/platform.py` (MAP-63) is what resolves
one of them into a manifest.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "deploy" / "docker" / "push-platform-image.sh"
_ECR_GATE = "MAP_ECR_TESTS"

REPOSITORIES = ("map/control-plane", "map/tool-gateway", "map/model-gateway")

_TAG = re.compile(r"^git-[0-9a-f]{40}$")
"""The one spelling of the tag form, read by both tiers.

Deliberately disjoint from the Session image's `git-<40 hex>-codex-<version>`: this
image takes no build argument, so a version component here would be a constant or a
lie, and two grammars that cannot match each other's tags mean neither repository's
contents can be mistaken for the other's.
"""


def test_the_push_script_is_one_a_shell_can_parse() -> None:
    parsed = subprocess.run(["sh", "-n", str(_SCRIPT)], capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr


# Every spelling buildx reads as false, so a legal one cannot fail this and an illegal
# one cannot pass it. Matching the literal "false" alone was the shape that let five
# spellings disable a live protection elsewhere in this suite; see docs/lessons.md.
_BUILDX_FALSE = frozenset({"false", "False", "FALSE", "f", "F", "0"})


def _command_lines() -> str:
    """The script with its whole-line comments dropped.

    A "must contain" assertion over the whole file is satisfied by the comment that
    explains the flag, and that is not a hypothetical: deleting `--provenance=false`
    from the buildx invocation left this file's guard green, because the paragraph above
    the invocation names the flag it is there to keep. The Session image's suite records
    the same correction for the same reason.
    """
    return "\n".join(
        line
        for line in _SCRIPT.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_build_disables_provenance_and_pushes_once_per_repository() -> None:
    """Three findings from one measured failure, asserted so none is tidied away.

    By default buildx binds the tag to an index over an attestation manifest and ECR
    refuses one of its children under that tag -- after the bytes have landed. There is
    a 1430-byte orphan in the Session pod's repository from exactly that.

    The flags are read off the command lines and the forbidden spelling off the whole
    file: a comment may not stand in for an invocation, and a comment that reintroduced
    the spelling would still be a comment worth failing on.
    """
    commands = _command_lines()
    provenance = re.search(r"--provenance=(\S+)", commands)
    assert provenance is not None, "the buildx invocation does not pass --provenance"
    assert provenance.group(1) in _BUILDX_FALSE, (
        f"--provenance={provenance.group(1)} does not disable provenance. buildx binds "
        "the tag to an index over an attestation manifest, and ECR refuses one of its "
        "children under that tag -- after the bytes have landed."
    )
    assert "--load" in commands
    assert "--push" not in _SCRIPT.read_text()
    assert commands.count("docker buildx build") == 1, "one build, three tags"
    assert "docker push" in commands


def test_the_three_repositories_are_written_out_rather_than_discovered() -> None:
    """A registry listing also holds the Session pod's repository and the spike
    repository, and a script that pushed to whatever it found would push this image over
    the Session pod's."""
    commands = _command_lines()
    for repository in REPOSITORIES:
        assert repository in commands
    whole = _SCRIPT.read_text()
    assert "map/session-shim" not in whole
    assert "describe-repositories" not in whole


def _throwaway_repository(tmp_path: Path) -> Path:
    """A git repository holding only the paths the guard names, committed.

    Built here rather than reusing this checkout so the two behavioural cases below
    cannot turn red because somebody has an edit open.
    """
    for relative in ("src", "migrations", "deploy/docker"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "alembic.ini").write_text("[alembic]\n")
    (tmp_path / "src" / "x.py").write_text("")
    (tmp_path / "migrations" / "env.py").write_text("")
    (tmp_path / "deploy" / "docker" / "platform.Dockerfile").write_text(
        "FROM scratch\n"
    )
    (tmp_path / "deploy" / "docker" / "platform.Dockerfile.dockerignore").write_text(
        "*\n"
    )
    (tmp_path / "deploy" / "docker" / _SCRIPT.name).write_bytes(_SCRIPT.read_bytes())
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
    ):
        subprocess.run(argv, cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def _print_tag(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(root / "deploy" / "docker" / _SCRIPT.name)],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "MAP_PRINT_TAG_ONLY": "1"},
    )


def test_the_tag_names_the_commit_and_nothing_else(tmp_path: Path) -> None:
    root = _throwaway_repository(tmp_path)
    printed = _print_tag(root)
    assert printed.returncode == 0, printed.stderr
    tag = printed.stdout.strip()
    assert _TAG.match(tag), tag
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tag == f"git-{head.stdout.strip()}"


def test_the_tag_grammar_cannot_be_confused_with_the_session_images() -> None:
    """The negative half, and the whole reason the version component was dropped."""
    assert not _TAG.match("git-" + "a" * 40 + "-codex-0.149.0")


def test_the_script_refuses_a_tree_whose_build_inputs_are_uncommitted(
    tmp_path: Path,
) -> None:
    root = _throwaway_repository(tmp_path)
    (root / "src" / "leftover.py").write_text("# uncommitted\n")
    refused = _print_tag(root)
    assert refused.returncode == 1
    assert "uncommitted" in refused.stderr


def test_the_script_ignores_an_uncommitted_file_the_image_cannot_see(
    tmp_path: Path,
) -> None:
    """A guard that refuses on any dirty file is a guard that gets deleted: the working
    notes in this repository are always in flight, and none of them changes a byte of
    the image."""
    root = _throwaway_repository(tmp_path)
    (root / "notes.md").write_text("# not a build input\n")
    printed = _print_tag(root)
    assert printed.returncode == 0, printed.stderr
    assert _TAG.match(printed.stdout.strip())


requires_the_registry = pytest.mark.skipif(
    os.environ.get(_ECR_GATE) != "1",
    reason=(
        f"reading the registries is opt-in: set {_ECR_GATE}=1 to run it. It needs the "
        "`aws` CLI and credentials for the account these repositories live in. SKIPPED "
        "MEANS NO REGISTRY WAS CONSULTED -- nothing here says whether any of them "
        "holds an image."
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


def _tagged(repository: str) -> list[Any]:
    described = _aws("describe-images", "--repository-name", repository)
    return [d for d in described["imageDetails"] if d.get("imageTags")]


@requires_the_registry
@pytest.mark.parametrize("repository", REPOSITORIES)
def test_each_repository_holds_a_platform_image(repository: str) -> None:
    """The floor under everything else. An empty repository is not a slow pull; it is no
    pull at all, and the Deployment stays ImagePullBackOff."""
    assert _tagged(repository), (
        f"{repository} holds no tagged image. A node pulls from the registry and never "
        "from a developer's daemon, so nothing can start until somebody runs "
        "`deploy/docker/push-platform-image.sh`"
    )


@requires_the_registry
@pytest.mark.parametrize("repository", REPOSITORIES)
def test_every_tag_in_the_repository_names_a_commit(repository: str) -> None:
    """Nothing enforces the tag form on the way in, so it is checked on the way out.

    A hand-rolled `latest` pushed once would sit in an IMMUTABLE repository for ever,
    naming bytes nobody can trace to a commit, and the digest-only production path would
    never notice.
    """
    for detail in _tagged(repository):
        for tag in detail["imageTags"]:
            assert _TAG.match(tag), (
                f"{tag!r} in {repository} was not produced by "
                "deploy/docker/push-platform-image.sh, so the bytes it names cannot be "
                "traced to a commit"
            )


@requires_the_registry
def test_one_byte_set_carries_all_three_services() -> None:
    """The claim D2 rests on, checked rather than asserted in prose.

    A digest is content-addressed and carries no repository, so three equal digests mean
    one build reached all three. Three different digests mean three builds, and three
    services running code from three commits.
    """
    newest: dict[str, tuple[str, str]] = {}
    for repository in REPOSITORIES:
        tagged = _tagged(repository)
        assert tagged, f"{repository} is empty"
        latest = max(tagged, key=lambda detail: str(detail["imagePushedAt"]))
        newest[repository] = (latest["imageTags"][0], latest["imageDigest"])
    tags = {tag for tag, _ in newest.values()}
    digests = {digest for _, digest in newest.values()}
    assert len(tags) == 1, f"three repositories at three commits: {newest}"
    assert len(digests) == 1, f"one tag, three byte sets: {newest}"
