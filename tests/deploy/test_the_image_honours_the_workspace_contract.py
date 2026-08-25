"""Every place that has to honour the workspace contract, checked against it.

`core/pod/workspace_contract.py` tells the model four things: where its inputs are,
where to put deliverables, how to install a package, and that network is off unless
granted. Each clause is a promise, and each is kept somewhere else -- the shim's
ship-out scan, the manifest's mounts and PYTHONPATH, the image's `map-pip` wrapper.
Nothing connects them except this file.

**A clause nobody honours is worse than no clause.** The model would follow it: write
its document into a directory nothing collects, or install a package nothing can
import, and the Turn would end looking like the model failed. Nothing would log a
thing -- the contract would still be there in `requirements.toml`, readable, exactly
as wrong as before.

So these cases compare strings across files that cannot import each other. The
Dockerfile is read as text because it is not Python; the manifest as YAML for the same
reason. Both would be a second copy of a path this repository already knows, were the
comparison left out.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Final

import yaml

from managed_agent.control.pod_config.compiler import (
    WORKSPACE_ROOT,
    _render_requirements,
    session_profile,
)
from managed_agent.core.pod.workspace_contract import (
    INPUT_DIR_NAME,
    OUTPUT_DIR_NAME,
    PACKAGE_DIR,
    PIP_WRAPPER,
    workspace_contract,
)
from managed_agent.core.toml_text import toml_string
from managed_agent.session_shim.serve import WORKSPACE_FILES

_ROOT: Final = Path(__file__).resolve().parents[2]
_DOCKERFILE: Final = (_ROOT / "deploy" / "docker" / "session.Dockerfile").read_text()
_POD: Final[dict[str, Any]] = yaml.safe_load(
    (_ROOT / "deploy" / "k8s" / "session-pod.yaml").read_text()
)

_PACKAGE_PATH: Final = f"{WORKSPACE_ROOT}/{PACKAGE_DIR}"


def _runtime_env() -> dict[str, str]:
    found = [c for c in _POD["spec"]["containers"] if c["name"] == "agent-runtime"]
    assert found, "the pod has no agent-runtime container"
    return {entry["name"]: str(entry["value"]) for entry in found[0].get("env", [])}


# --------------------------------------------------------------------------------------
# The contract reaches the model at all
# --------------------------------------------------------------------------------------


def test_the_managed_document_does_not_carry_the_key_no_codex_has() -> None:
    """**The refusal, not an omission.**

    `additional_developer_instructions` is the runtime's administrator channel and is
    the right home for this contract in every way but one: it exists on codex's `main`
    branch and in no release. Measured over the shipped binaries with `strings` -- zero
    occurrences in 0.149.0, which this platform pins, and zero in 0.149.1, against 39
    and 43 for `developer_instructions`. A document carrying it parses, loads, and
    drops the text without a word, which is how it went unnoticed: the same file's
    permission profile and egress proxy were in force the whole time.

    So the contract rides `config.toml`'s `developer_instructions` instead --
    `tests/control/test_model_binding.py` is where its delivery is asserted -- and this
    case exists to keep the dead key OUT. Writing it "for a later codex" is the same
    mistake as the inert egress keys `_refuse_egress_the_proxy_does_not_bound` rejects:
    a key that is present and does nothing is what the next reader takes as evidence
    the contract is delivered.

    Delete this case when a released codex contains the key, and move the contract at
    the same time. Not before: the two halves of that change are what keep the text
    reaching the model.
    """
    text = _render_requirements(session_profile(), gateway_url="https://gw.example/mcp")

    assert "additional_developer_instructions" not in text
    assert "additional_developer_instructions" not in tomllib.loads(text)
    # Nor by any other name. The managed document is the administrator's, and prose
    # addressed to the model has no business in it while no key here delivers prose.
    assert workspace_contract() not in text


def test_the_contract_survives_being_written_as_toml() -> None:
    """It is a paragraph, and a raw newline inside a TOML basic string is a parse error.

    Round-tripped rather than pattern-matched: the quoting that emitted it escaped only
    backslash and quote, which was correct for the paths it had been written for and
    produced an unloadable document the first time it was handed prose. See
    `core/toml_text.py`, which both emitters now share for that reason.

    Asserted here against the quoter directly, since the field this text now travels in
    is rendered by `control/pod_config/model_binding.py`. What must hold is a property
    of the contract itself: it is multi-line prose, and it has to come back the way it
    went in.
    """
    text = workspace_contract()
    parsed = tomllib.loads("carried = " + toml_string(text))

    assert parsed["carried"] == text
    assert parsed["carried"].count("\n") > 5


def test_the_contract_names_no_tenant_and_no_turn() -> None:
    """It is compiled per Session, so it COULD carry something tenant-specific.

    The moment it does, the platform is writing prompt text on a tenant's behalf into a
    document the tenant cannot see or override. Asserted as an absence because that is
    the only way this stays true: the check has to fail when somebody interpolates.
    """
    text = workspace_contract()

    assert "{" not in text
    assert "%s" not in text
    assert not re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-", text), "an id leaked in"


# --------------------------------------------------------------------------------------
# The clause about inputs
# --------------------------------------------------------------------------------------


def test_the_contract_names_the_input_directory_that_is_actually_mounted() -> None:
    """The manifest mounts the workspace volume with `subPath` for the shim's write
    mount, so the directory that exists is the manifest's decision. A contract naming a
    different one sends the model to look somewhere empty."""
    mounts = [
        mount
        for container in _POD["spec"]["containers"]
        if container["name"] == "session-shim"
        for mount in container.get("volumeMounts", [])
        if mount.get("subPath") == INPUT_DIR_NAME
    ]

    assert mounts, f"no session-shim mount uses subPath {INPUT_DIR_NAME!r}"
    assert INPUT_DIR_NAME in workspace_contract()
    assert WORKSPACE_FILES.name == INPUT_DIR_NAME


# --------------------------------------------------------------------------------------
# The clause about outputs
# --------------------------------------------------------------------------------------


def test_the_contract_tells_the_model_where_deliverables_go() -> None:
    """Named in the text, because the shim prefers that directory and no other.

    This is the clause that changes what a tenant receives: before it, ship-out returned
    every regular file at the workspace root, so a Turn that wrote a generator script to
    render a PDF returned the script too -- measured on a live run.
    """
    text = workspace_contract()

    assert f"./{OUTPUT_DIR_NAME}/" in text
    assert "Only files there are" in text


# --------------------------------------------------------------------------------------
# The clause about installing a package
# --------------------------------------------------------------------------------------


def test_the_wrapper_the_contract_names_is_the_one_the_image_installs() -> None:
    """A command name in the contract that the image does not provide is `command not
    found` inside a Turn, on the one instruction the platform itself authored."""
    assert PIP_WRAPPER in workspace_contract()
    assert f"/usr/local/bin/{PIP_WRAPPER}" in _DOCKERFILE
    assert f"command -v {PIP_WRAPPER}" in _DOCKERFILE, (
        "the build never proves the wrapper is on PATH"
    )


def test_the_wrapper_installs_where_the_manifest_says_imports_come_from() -> None:
    """**The two halves of one promise, in two files that cannot import each other.**

    `map-pip` installs to `--target`; PYTHONPATH decides where an import looks. If those
    disagree, `map-pip requests` succeeds, prints nothing alarming, and the next line of
    Python cannot import it -- a Turn that installed a package and cannot use it, with
    every command reporting success.
    """
    assert _PACKAGE_PATH in _DOCKERFILE, (
        f"the wrapper does not install to {_PACKAGE_PATH}"
    )
    assert _runtime_env().get("PYTHONPATH") == _PACKAGE_PATH


def test_the_install_directory_can_never_be_shipped_to_a_tenant() -> None:
    """A dependency tree is not a document.

    `_is_a_bare_leaf` in `shim/serve.py` rejects a leading dot in both directions of
    this pod's file traffic, so a dotted directory is excluded from ship-out by a rule
    that already existed. Asserted here because the exclusion IS the reason for the
    dot: a later reader tidying `.map` to `map` starts returning site-packages.
    """
    assert PACKAGE_DIR.startswith("."), (
        "the package directory must be dotted or ship-out will return it"
    )


def test_the_image_installs_a_pip_for_the_wrapper_to_run() -> None:
    """The wrapper runs `python3 -m pip`, and `uv sync` prunes what the lock does not
    name -- so a pip seeded by `uv venv --seed` is installed and deleted in one layer.
    The build has to prove pip is there after the sync, not before it."""
    assert "python3 -m pip --version" in _DOCKERFILE, (
        "the build never proves the agent's interpreter has pip"
    )
    sync = _DOCKERFILE.index("uv sync")
    install = _DOCKERFILE.index("uv pip install")
    assert install > sync, "pip is installed before the sync that would prune it"
