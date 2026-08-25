"""The workspace resolves, and the layer linter actually refuses a layering violation.

Tier 1 (local, no infrastructure). The layer rule — nothing under `core/` may import an
adapter or a third-party infrastructure client — is enforced by a ruff banned-api rule,
and a lint rule that is configured but not exercised is indistinguishable from one that
is misconfigured. These tests run ruff against a module written to break the rule and
against one allowed to bend it, so the ban is graded in both directions.
"""

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
    """
    return subprocess.run(
        ["ruff", *args],
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
    result = _ruff("check", ".")
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
