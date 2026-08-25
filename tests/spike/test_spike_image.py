"""Build-time guard on the two binaries the spike probe's verdict depends on.

The probe's first assertion shells out to `unshare`. The AL2023 base image does
not carry it, and a missing binary exits 127 — the same non-zero status a kernel
that refuses user namespaces produces. Read off the exit status alone the two are
one failure, and the wrong one of them says the node cannot host the sandbox at
all. So both binaries are proven resolvable here, when the image is built, rather
than inside the spike where the false negative would already be the answer.

The build is also proven to refuse a missing `CODEX_VERSION`. The spike's whole
output is a set of numbers read off that binary, and a build that quietly took
whatever npm called `latest` that morning would produce numbers nobody can
re-derive later.

Tier 1: these run against a locally built image on this machine's Docker daemon.
They say nothing about whether the kernel on a cluster node permits anything —
that is the pod run's job, and no assertion here substitutes for it. The image is
built for `linux/amd64` because the cluster's nodes are amd64; on an arm64 host it
runs under emulation, which is slow but keeps the tested image byte-identical to
the pushed one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SPIKE_DIR = Path(__file__).resolve().parents[2] / "deploy" / "spike"
IMAGE_TAG = "map-spike:pytest"
CODEX_VERSION = "0.149.0"
PLATFORM = "linux/amd64"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=1800)


@pytest.fixture(scope="module")
def image() -> str:
    """Build the probe image and yield its tag, failing the module if it will not build.

    Module-scoped because a `dnf install` plus an `npm install -g` under amd64
    emulation costs minutes, and every assertion below reads the same image.
    """
    built = _run(
        [
            "docker",
            "build",
            "--platform",
            PLATFORM,
            "--build-arg",
            f"CODEX_VERSION={CODEX_VERSION}",
            "-t",
            IMAGE_TAG,
            str(SPIKE_DIR),
        ]
    )
    if built.returncode != 0:
        pytest.fail(
            f"docker build failed rc={built.returncode}\n{built.stderr[-4000:]}"
        )
    return IMAGE_TAG


def _in_image(image_tag: str, shell_command: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            PLATFORM,
            "--entrypoint",
            "/bin/sh",
            image_tag,
            "-c",
            shell_command,
        ]
    )


def test_unshare_resolves_on_path(image: str) -> None:
    result = _in_image(image, "command -v unshare")
    assert result.returncode == 0, (
        "unshare does not resolve in the image; the probe would exit 127 and "
        f"report a kernel refusal that never happened. stderr={result.stderr}"
    )
    assert result.stdout.strip().endswith("unshare")


def test_bwrap_resolves_on_path(image: str) -> None:
    result = _in_image(image, "command -v bwrap")
    assert result.returncode == 0, (
        f"bwrap does not resolve in the image. stderr={result.stderr}"
    )
    assert result.stdout.strip().endswith("bwrap")


def test_codex_reports_the_pinned_version(image: str) -> None:
    result = _in_image(image, "codex --version")
    assert result.returncode == 0, f"codex --version failed: {result.stderr}"
    assert CODEX_VERSION in result.stdout, (
        f"image reports {result.stdout.strip()!r}, expected to contain {CODEX_VERSION}"
    )


def test_default_user_is_the_unprivileged_uid(image: str) -> None:
    """Root inside the image would report capabilities a Session pod never has.

    pod.yaml sets runAsUser, but the image carries the same uid so that running it
    outside Kubernetes cannot accidentally probe as root and record a permissive
    answer to the one question this spike exists to settle.
    """
    result = _in_image(image, "id -u")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "10001", (
        f"image default uid is {result.stdout.strip()!r}, not 10001"
    )


def test_build_without_codex_version_is_refused() -> None:
    """A build with no CODEX_VERSION must fail rather than install a floating latest.

    Built to a throwaway tag so a refused build cannot leave the tag the other
    assertions read pointing at an unpinned image.
    """
    built = _run(
        [
            "docker",
            "build",
            "--platform",
            PLATFORM,
            "-t",
            "map-spike:unpinned-should-not-exist",
            str(SPIKE_DIR),
        ]
    )
    assert built.returncode != 0, (
        "docker build succeeded with no CODEX_VERSION; the spike could measure "
        "whatever version was newest that morning and the record would not say which"
    )
