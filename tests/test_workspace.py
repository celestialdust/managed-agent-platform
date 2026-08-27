"""The workspace resolves, and the layer linter actually refuses a layering violation.

Tier 1 (local, no infrastructure). The layer rule — nothing under `core/` may import an
adapter or a third-party infrastructure client — is enforced by a ruff banned-api rule,
and a lint rule that is configured but not exercised is indistinguishable from one that
is misconfigured. These tests run ruff against a module written to break the rule and
against one allowed to bend it, so the ban is graded in both directions.
"""

import ast
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"

_VIOLATION = """\
from managed_agent.adapters.postgres.event_log_append import PostgresEventLogAppend

__all__ = ["PostgresEventLogAppend"]
"""


def _ruff(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ruff, and let a missing ruff be a failure rather than a skip.

    `shutil.which` plus `pytest.skip` used to guard this. That guard turns the layer
    linter green in the one environment where the linter is genuinely absent -- which is
    exactly the environment whose result cannot be trusted. `subprocess.run` raises
    FileNotFoundError on its own, loudly, which is the behaviour wanted. The dev extra
    declares ruff, so a machine without it has a broken checkout, not a special case.

    `--no-cache` is applied here rather than at the call sites, for the same reason the
    paragraph above gives. Ruff keys its cache on file metadata, so a checkout whose
    `.ruff_cache` was written under other conditions -- a worktree, a rebase, an
    interrupted run -- hands back a stale pass for a file that breaks a rule right now.
    That happened: this file's whole-tree check reported green on a tree where
    `ruff check --no-cache` found an unsorted import block, in a file byte-identical to
    the one a second checkout failed on. A guard that can return a false green is worse
    than no guard, because the green is what stops anybody looking. It goes on the
    helper so a call site added later cannot forget it, and after the subcommand
    because ruff rejects it before one.
    """
    subcommand, *rest = args
    return subprocess.run(
        ["ruff", subcommand, "--no-cache", *rest],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=False,
    )


def test_every_declared_runtime_dependency_resolved() -> None:
    """The environment holds what `pyproject.toml` asks for, so the lock is real."""
    declared = tomllib.loads(_PYPROJECT.read_text())["project"]["dependencies"]
    names = [
        requirement.split("[")[0].split(">")[0].split("=")[0].strip()
        for requirement in declared
    ]
    missing = []
    for name in names:
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(name)
    assert missing == [], f"declared but not installed: {missing}"


def test_the_lockfile_exists_and_pins_the_workspace() -> None:
    lock = _ROOT / "uv.lock"
    assert lock.exists(), "uv sync has not been run, so nothing is pinned"
    assert 'name = "managed-agent-platform"' in lock.read_text()


def test_ruff_flags_a_core_module_that_imports_an_adapter(tmp_path: Path) -> None:
    offender = tmp_path / "leaky_core_module.py"
    offender.write_text(_VIOLATION)

    result = _ruff("check", "--config", str(_PYPROJECT), str(offender))

    assert "TID251" in result.stdout, result.stdout or result.stderr


def test_ruff_permits_the_adapter_package_to_name_infrastructure() -> None:
    """The ban is scoped: an adapter is where a driver is supposed to appear."""
    result = _ruff(
        "check",
        "--select",
        "TID251",
        "src/managed_agent/adapters",
        "src/managed_agent/composition.py",
    )

    assert result.returncode == 0, result.stdout or result.stderr


def test_ruff_passes_over_the_whole_tree() -> None:
    """Every file ruff can see is clean, decided fresh -- see `_ruff` on the cache."""
    result = _ruff("check", ".")
    assert result.returncode == 0, result.stdout or result.stderr


def test_the_whole_tree_is_formatted_as_ruff_would_format_it() -> None:
    """`ruff format` is a separate command from `ruff check`, and nothing ran it.

    The lint gate beside this one passes over an unformatted file, because formatting is
    not a lint rule -- so three files were committed in a shape `ruff format` would
    rewrite, and the drift surfaced only when somebody happened to run the formatter
    against the whole tree instead of the paths they had edited. That is the failure
    this closes: the gate now grades the same tree the formatter does, rather than
    whichever paths a developer thought to name.

    `--check` rather than `--diff`, because the assertion needs the file names and not
    the rewrite; `result.stdout` carries them.
    """
    result = _ruff("format", "--check", ".")
    assert result.returncode == 0, result.stdout or result.stderr


def test_mypy_strict_passes_over_the_tree() -> None:
    """`mypy --strict` over src, tests and migrations.

    No arguments: the paths come from `[tool.mypy] files` in pyproject.toml, so this
    gate and a developer's bare `mypy --strict` grade the same tree and cannot drift
    apart. A missing mypy fails here rather than skipping, for the reason `_ruff` gives.
    """
    result = subprocess.run(
        ["mypy", "--strict"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stdout or result.stderr


_OPT_IN_PROBE = """\
import pytest


def test_a_plain_case_the_default_run_must_keep() -> None:
    assert True


@pytest.mark.network
def test_a_network_case_the_default_run_must_drop() -> None:
    assert True


@pytest.mark.image
def test_an_image_case_the_default_run_must_drop() -> None:
    assert True
"""


def _requirement_names(requirements: list[str]) -> set[str]:
    """The distribution names out of a PEP 508 list, extras and specifiers dropped.

    Lower-cased, so `types-PyYAML` and `types-pyyaml` compare equal -- the two
    spellings are the same distribution and both appear in the wild.
    `test_every_declared_runtime_dependency_resolved` above parses the same list
    inline and is deliberately left alone: two copies of a parse is a coincidence,
    and the third caller is the one that extracts it.
    """
    return {
        requirement.split("[")[0].split(">")[0].split("=")[0].strip().lower()
        for requirement in requirements
    }


def _project_table() -> dict[str, list[str]]:
    table = tomllib.loads(_PYPROJECT.read_text())["project"]
    return {
        "runtime": table["dependencies"],
        "dev": table["optional-dependencies"]["dev"],
    }


def _pytest(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a nested pytest against this repository's own config.

    `-c` rather than a generated ini: the whole point is to grade the `addopts` and
    `markers` this file lives beside, and a copy of them would pass while the real
    ones were wrong.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(_PYPROJECT), *args],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=False,
    )


def _collect(probe: Path, *args: str) -> set[str]:
    """The bare test names a nested collection kept. `-p no:cacheprovider` stops the
    inner run writing a cache directory into the outer run's tree."""
    result = _pytest(
        "--collect-only", "-q", "-p", "no:cacheprovider", *args, str(probe)
    )
    assert result.returncode == 0, result.stdout or result.stderr
    return {
        line.rsplit("::", 1)[1]
        for line in result.stdout.splitlines()
        if "::test_" in line
    }


def _opt_in_probe(tmp_path: Path) -> Path:
    probe = tmp_path / "test_opt_in_probe.py"
    probe.write_text(_OPT_IN_PROBE)
    return probe


def test_the_default_run_keeps_the_unmarked_case_and_drops_both_opt_in_marks(
    tmp_path: Path,
) -> None:
    """Set equality, not two `not in` checks.

    A probe module that failed to import collects nothing, and nothing satisfies every
    absence assertion for free. Naming the one case that must survive is what makes the
    two absences mean something.
    """
    assert _collect(_opt_in_probe(tmp_path)) == {
        "test_a_plain_case_the_default_run_must_keep"
    }


def test_each_opt_in_mark_is_still_selectable_on_its_own(tmp_path: Path) -> None:
    """The positive control, and it passes with `image` unregistered too.

    An unregistered mark is still selectable -- pytest only warns about it. So this case
    is not the guard on registration (the `--markers` case below is); it is what proves
    the probe module above is real and reachable, so that "deselected" is a decision
    rather than an absence.
    """
    probe = _opt_in_probe(tmp_path)
    assert _collect(probe, "-m", "image") == {
        "test_an_image_case_the_default_run_must_drop"
    }
    assert _collect(probe, "-m", "network") == {
        "test_a_network_case_the_default_run_must_drop"
    }


def test_both_opt_in_marks_are_registered_so_neither_is_an_unknown_mark() -> None:
    """An unregistered mark raises PytestUnknownMarkWarning and still runs.

    That is the failure worth guarding: a typo in a mark name silently puts a
    container build back into the default offline run rather than erroring.
    """
    listing = _pytest("--markers").stdout
    assert "@pytest.mark.network:" in listing, listing
    assert "@pytest.mark.image:" in listing, listing


def test_the_yaml_parser_and_the_kubernetes_client_are_runtime_dependencies() -> None:
    """Both are imported by `src/`, so a `--no-dev` install must carry them.

    The Session image installs with `uv sync --locked --no-dev`, so a dev-extra
    declaration of either one builds an image that fails at `import
    managed_agent.composition` rather than at build time.
    """
    table = _project_table()
    runtime = _requirement_names(table["runtime"])
    dev = _requirement_names(table["dev"])
    assert {"pyyaml", "kubernetes-asyncio"} <= runtime
    assert not {"pyyaml", "kubernetes-asyncio"} & dev


def test_the_yaml_stubs_stay_a_dev_dependency() -> None:
    """Stubs are read by mypy and imported by nothing, so they belong in dev only."""
    table = _project_table()
    assert "types-pyyaml" in _requirement_names(table["dev"])
    assert "types-pyyaml" not in _requirement_names(table["runtime"])


def test_the_lockfile_is_in_step_with_the_manifest() -> None:
    """`uv lock --check`, which is the same disagreement `uv sync --locked` refuses.

    The Session image's build runs `uv sync --locked`, so a manifest edited without a
    re-lock fails the image build rather than any test. `--offline` keeps this
    deterministic: the check reads the lock and resolves nothing from a registry.
    """
    result = subprocess.run(
        ["uv", "lock", "--check", "--offline"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stdout or result.stderr


_KEY_MATERIAL_NAMES = (
    "ca.key",
    "ca.crt",
    "tls.key",
    "server.pem",
    "client.p12",
    "bundle.pfx",
)


def test_no_private_key_can_be_committed_from_the_repository_root() -> None:
    """Every shape of key material this repo produces is ignored, and none is tracked.

    The internal CA's private key is minted by a command that writes it to the
    repository root, which puts it one `git add -A` away from history -- and a private
    key that reaches history is a key that must be rotated, not one that can be
    removed, because every clone already has it. The `.gitignore` entries are the
    guard; this is what stops them being deleted by somebody who cannot see what they
    were for.

    Asked of `git check-ignore` rather than by reading the file, so the assertion is
    about the decision git actually makes -- a later negation (`!ca.crt`) further down
    the file would pass a text search and fail here, which is the right way round.

    The second half is the other direction and is not redundant: an ignore rule that
    happened to cover a file already tracked would silently do nothing, because git
    ignores nothing it is already following.
    """
    for name in _KEY_MATERIAL_NAMES:
        decided = subprocess.run(
            ["git", "check-ignore", "-q", name],
            cwd=_ROOT,
            check=False,
        )
        assert decided.returncode == 0, f"{name} at the repository root is not ignored"

    tracked = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=True,
    ).stdout.splitlines()
    suffixes = {".key", ".pem", ".crt", ".p12", ".pfx"}
    carrying = [path for path in tracked if Path(path).suffix in suffixes]
    assert carrying == [], f"key material is tracked: {carrying}"


def test_every_live_case_that_places_a_pod_signs_it_with_the_clusters_ca() -> None:
    """No live case builds a pod runner without handing it an internal CA.

    `KubernetesPodRunner.from_manifest_file` defaults `internal_ca` to `None`, which is
    correct for the platform -- a deployment with no CA material places pods that serve
    plain HTTP. It is wrong for a case running against a cluster that *does* hold CA
    material: the pod comes up on plain HTTP while the dial, reading the deployed
    Secret, speaks TLS at it. The handshake dies as `record layer failure`, which names
    neither the placement nor the dial and points at neither file.

    Read out of the syntax rather than by running anything, because the failure only
    reproduces against a live cluster holding a CA -- which is exactly the environment
    an offline suite does not have.
    """
    live_tier = _ROOT / "tests" / "pod"
    placements = 0
    for module in sorted(live_tier.glob("*.py")):
        for node in ast.walk(ast.parse(module.read_text())):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if not isinstance(called, ast.Attribute):
                continue
            if called.attr != "from_manifest_file":
                continue
            placements += 1
            named = {keyword.arg for keyword in node.keywords}
            assert "internal_ca" in named, (
                f"{module.name} builds a pod runner without `internal_ca`, so the pod "
                "it places serves plain HTTP while `shim_dial` speaks TLS at it"
            )
    assert placements, "no live case places a pod any more; delete this guard"
