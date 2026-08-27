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
from managed_agent.core.pod.permission_profile import is_strictly_under
from managed_agent.core.pod.workspace_contract import (
    INPUT_DIR_NAME,
    OUTPUT_DIR_NAME,
    PACKAGE_DIR,
    PIP_WRAPPER,
    SCRATCH_LIMIT_MEBIBYTES,
    SCRATCH_ROOT,
    workspace_contract,
)
from managed_agent.core.toml_text import toml_string
from managed_agent.core.vfs.session_vfs import ARTIFACTS, SealedLane
from managed_agent.session_shim.serve import WORKSPACE_FILES

_ROOT: Final = Path(__file__).resolve().parents[2]
_DOCKERFILE: Final = (_ROOT / "deploy" / "docker" / "session.Dockerfile").read_text()
_POD: Final[dict[str, Any]] = yaml.safe_load(
    (_ROOT / "deploy" / "k8s" / "session-pod.yaml").read_text()
)


def _runtime_env() -> dict[str, str]:
    found = [c for c in _POD["spec"]["containers"] if c["name"] == "agent-runtime"]
    assert found, "the pod has no agent-runtime container"
    return {entry["name"]: str(entry["value"]) for entry in found[0].get("env", [])}


def _image_env() -> dict[str, str]:
    """Every `ENV NAME=value` the image declares, as the pod's processes see it.

    Read as text rather than from a built image, for the reason the module docstring
    gives: the Dockerfile is not Python and cannot be imported. One name per line is
    the form this file writes them in, so a multi-variable `ENV` would be missed --
    which is why `test_every_cache_the_image_redirects_lands_on_pod_local_scratch`
    asserts the set it found is the whole set rather than merely non-empty.
    """
    found: dict[str, str] = {}
    for line in _DOCKERFILE.splitlines():
        match = re.fullmatch(r"ENV\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)", line.strip())
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _scratch_volume_name() -> str:
    """The volume behind the scratch mount, found through the mount rather than by name.

    Looked up this way so that renaming the volume and leaving the mount, or the
    reverse, fails against a mount nothing backs instead of passing against a volume
    nothing uses.
    """
    for container in _POD["spec"]["containers"]:
        for mount in container.get("volumeMounts", []):
            if mount["mountPath"] == SCRATCH_ROOT:
                return str(mount["name"])
    raise AssertionError(f"no container mounts {SCRATCH_ROOT}")


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
    different one sends the model to look somewhere empty.

    Matched on the LAST segment of the subPath rather than on the whole of it: the rest
    is this Session's subtree of a volume shared by every Session on the cluster
    (ADR-035), filled in per Session by the pod runner, and no concern of the contract
    the model reads. What the model is told is a directory name, and this is the mount
    that decides which name exists.
    """
    mounts = [
        mount
        for container in _POD["spec"]["containers"]
        if container["name"] == "session-shim"
        for mount in container.get("volumeMounts", [])
        if str(mount.get("subPath", "")).rsplit("/", 1)[-1] == INPUT_DIR_NAME
    ]

    assert mounts, f"no session-shim mount ends its subPath at {INPUT_DIR_NAME!r}"
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


def test_the_contract_tells_the_model_a_produced_path_is_written_once() -> None:
    """The clause and the thing that enforces it, compared across two modules.

    A tenant asking for four rounds of edits on one document is the ordinary case, and
    the agent's ordinary answer is to rewrite the same path. The `artifacts` lane is
    sealed, so the second write is refused -- and until this clause existed the agent
    had no way to know that before it happened. `SealedLane` asserted beside it because
    the clause is only true while the lane stays sealed: a later reader who made the
    lane mutable would leave the model told to invent version names for no reason.
    """
    text = workspace_contract()

    assert "Each path there is written once" in text
    assert "a new path" in text
    assert isinstance(ARTIFACTS, SealedLane), (
        "the contract promises a produced path ships once; nothing enforces it"
    )


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

    This is the case that catches a half-done move of the directory itself, which is
    why both halves are compared against the constant and not against each other. The
    directory moved off the workspace and onto pod-local scratch, and it had to move in
    three files at once -- this constant, the wrapper's `--target`, and the manifest's
    PYTHONPATH. Moving two of the three leaves a Session installing packages onto a
    network mount that nothing imports from, or importing from a directory nothing
    installs into; neither says a word at the time.
    """
    assert PACKAGE_DIR in _DOCKERFILE, f"the wrapper does not install to {PACKAGE_DIR}"
    assert _runtime_env().get("PYTHONPATH") == PACKAGE_DIR


def test_the_install_directory_is_outside_the_tree_ship_out_walks() -> None:
    """A dependency tree is not a document, and it is now out of reach rather than
    filtered.

    It used to sit at `<workspace>/.map/lib` and be excluded by the leading dot, which
    `_is_a_bare_leaf` in `shim/serve.py` rejects in both directions of this pod's file
    traffic. On scratch there is nothing to exclude: ship-out walks the workspace, and
    this path is not under it, so no filter has to hold for a site-packages tree to
    stay out of a tenant's deliverables.

    Asserted as the containment fact rather than as the old leading dot, because a
    later reader moving it back under the workspace would restore a filter dependency
    this no longer has -- and would put every run-time install back on the network
    mount, which is what ADR-037 moved it off.
    """
    assert is_strictly_under(PACKAGE_DIR, SCRATCH_ROOT)
    assert not is_strictly_under(PACKAGE_DIR, WORKSPACE_ROOT), (
        "the package directory is back inside the workspace, so every run-time install "
        "crosses the network again and ship-out has to filter it out by name"
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


# --------------------------------------------------------------------------------------
# The clause about where a large intermediate goes
# --------------------------------------------------------------------------------------


def test_every_cache_the_image_redirects_lands_on_pod_local_scratch() -> None:
    """The half of ADR-037 that needs no compliance from the agent, checked whole.

    A build tool picks its own output path and no instruction to the model reaches it,
    so these variables are the only lever over `cargo build`, `npm install` and the
    package caches. Each one pointing at scratch is what keeps those writes off the
    network mount; one of them left out is a gigabyte of build output crossing NFS on
    a Turn nobody will connect back to this file.

    Asserted as an exact set, not as a subset, for the failure in the other direction:
    a variable added here pointing at a path the sandbox cannot write turns a tool that
    worked into one that fails, which is worse than not redirecting it at all. So a new
    redirect has to be added to this list deliberately, and it has to name scratch.

    `TMPDIR` is deliberately absent and its absence is a separate case --
    `test_no_container_redirects_the_system_temporary_directory` in
    `tests/control/test_compiled_config_floors.py` is what refuses it, in the image as
    well as in the manifest.
    """
    redirected = {
        name: value
        for name, value in _image_env().items()
        if value.startswith("/session/")
    }

    assert set(redirected) == {
        "CARGO_TARGET_DIR",
        "npm_config_cache",
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
        "GOCACHE",
    }
    for name, value in redirected.items():
        assert is_strictly_under(value, SCRATCH_ROOT), f"{name} is not on scratch"


def test_the_scratch_those_defaults_name_is_mounted_where_the_agent_runs() -> None:
    """An environment default is a promise about a path, and this is the mount that
    makes the path exist.

    Without it every variable above names a directory on a read-only root: `npm
    install` fails at its cache before it fetches anything, and the failure names a
    path no skill's own text mentions. That is strictly worse than leaving the default
    alone, which is why the mount and the variables are graded together.

    On `agent-runtime` alone, and asserted as an exact set. It is the only container
    that runs a confined command, and an `emptyDir` carries no sticky bit -- a second
    mounting container is a second process able to unlink the first's files.
    """
    mounting = {
        container["name"]
        for container in _POD["spec"]["containers"] + _POD["spec"]["initContainers"]
        for mount in container.get("volumeMounts", [])
        if mount["mountPath"] == SCRATCH_ROOT
    }

    assert mounting == {"agent-runtime"}


def test_the_scratch_the_contract_names_is_one_the_profile_lets_the_agent_write() -> (
    None
):
    """**The clause and the kernel, compared.**

    This is the case the whole scratch change turns on. The permission profile extends
    `:read-only`, so a path is writable only where a rule says so -- and a contract
    sending the model to a directory no rule covers, with build tools already pointed
    there by the image, produces a Turn where every tool fails on a path the platform
    itself chose. The model would have no way to tell that from its own mistake.
    """
    writable = session_profile().writable()

    assert SCRATCH_ROOT in writable, (
        "the contract names scratch and the profile does not make it writable"
    )
    assert f"{SCRATCH_ROOT}/" in workspace_contract()


def test_the_contract_tells_the_model_the_bound_the_kubelet_will_enforce() -> None:
    """**The one clause here whose cost is the Session rather than the deliverable.**

    Every other promise in this contract degrades: an agent that ignores `out/` gets
    its file shipped with scratch beside it. This one does not. Enforcement is
    kubelet's periodic `du` with no filesystem-quota feature gate, so a write past the
    limit does not return ENOSPC -- the pod is EVICTED, and `restartPolicy: Never`
    means the Session ends there, mid-Turn, with whatever it had done.

    So the number the model is told has to be the number the manifest declares. Told a
    larger one it unpacks a dataset that kills its own Session; told none at all it has
    no basis to decide not to.
    """
    volume = next(
        volume
        for volume in _POD["spec"]["volumes"]
        if volume["name"] == _scratch_volume_name()
    )

    assert volume["emptyDir"]["sizeLimit"] == f"{SCRATCH_LIMIT_MEBIBYTES}Mi"
    assert f"{SCRATCH_LIMIT_MEBIBYTES} MB" in workspace_contract()
