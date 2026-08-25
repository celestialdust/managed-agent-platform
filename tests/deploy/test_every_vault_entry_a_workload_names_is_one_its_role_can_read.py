"""A workload's IAM policy must allow every Secrets Manager entry its manifest names.

The defect this exists for is written down in the production code that suffers it.
`src/managed_agent/gateway/model/router.py` says of its own configuration: *"on the
manifest as written that is every request, because `MAP_POD_TOKEN_KEY_NAME` sits outside
the one prefix this pod's IAM role can read"*. That is a comment describing a live
misconfiguration, and when it was written nothing failed -- not a unit test, not the
applier, not the deploy. The pods come up Ready, the rollout completes, and every
request that needs the entry dies at AWS with an error naming a secret path.

`deploy/platform.py` already refuses a workload whose declared entry is **absent
from the account**. That check and this one are different questions and neither
implies the other: an entry can exist and be unreadable by the role that needs it, which
is the state `map-model-gateway` was in for as long as
`map/dev/platform/pod-token-signing-key` has been named in its manifest.

Both sides are derived, never listed. The names come from the applier's own
`declared_vault_entries` -- the same function that gates the deploy -- so a manifest
that adds a model adds an entry here. The permissions come from parsing the policy JSON
the Terraform resource actually reads via `file()`, so a policy edited without a
matching Terraform change is still the document under test.

On the trailing `-*`: Secrets Manager appends six random characters to every ARN, so a
policy for one entry is written `...pod-token-signing-key-*` and a policy for a prefix
is written `...providers/*`. A match is therefore tried against the bare name **and**
against the name plus a synthetic suffix; checking only the bare name would report a
correct single-entry policy as a gap.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from fnmatch import fnmatch
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest

from managed_agent.control.webhooks import dispatcher
from managed_agent.gateway.tool import credential_broker

_ROOT: Final = Path(__file__).resolve().parents[2]
_IAM: Final = _ROOT / "deploy" / "iam"
_ARN_MARKER: Final = ":secret:"
_SYNTHETIC_SUFFIX: Final = "-AbCdEf"
"""Stands in for the six characters Secrets Manager appends to an ARN.

Six characters and mixed case, matching the shape of the real suffixes in
`deploy/terraform/secrets.tf` (`-qOB2YZ`, `-yNBhsj`). Its value never matters -- only
that it is a hyphen followed by something -- because it exists to make a
`...key-*` pattern reachable from a bare entry name.
"""

_READ_ACTIONS: Final = frozenset(
    {"secretsmanager:getsecretvalue", "secretsmanager:*", "*"}
)
"""Actions that let a principal read a secret's value.

Lower-cased at comparison because IAM action names are case-insensitive, and a policy
spelling one `secretsmanager:GetSecretValue` and another `SecretsManager:getsecretvalue`
grants the same thing.
"""


def _module(name: str, relative: str) -> ModuleType:
    """A module under `deploy/` loaded by path.

    `deploy/` is not a package and is not on `sys.path`; the wheel packages
    `src/managed_agent` only. Registered in `sys.modules` before execution because a
    dataclass defined in the module resolves `sys.modules[cls.__module__]` while it is
    being built, and a module absent from that mapping fails with an error naming
    `NoneType` rather than naming the import.
    """
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, relative
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _as_list(value: object) -> list[str]:
    """IAM writes a single-element field as a bare string and many as a list."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(one) for one in value]
    return []


def _readable_patterns(policy: dict[str, Any]) -> list[str]:
    """Every secret-name glob this policy allows a read of.

    Deny statements are not subtracted. A `Deny` would make this report a gap that is
    real, so ignoring them can only make the guard stricter than the account -- never
    laxer, which is the direction that would let a defect through. The policies here
    carry no `Deny`, and one appearing is caught by the assertion below that every
    statement was understood.
    """
    patterns: list[str] = []
    for statement in policy.get("Statement", []):
        if statement.get("Effect") != "Allow":
            continue
        actions = {one.lower() for one in _as_list(statement.get("Action"))}
        if not actions & _READ_ACTIONS:
            continue
        for resource in _as_list(statement.get("Resource")):
            _, _, tail = resource.partition(_ARN_MARKER)
            patterns.append(tail if tail else resource)
    return patterns


def _allows(patterns: list[str], entry: str) -> bool:
    return any(
        fnmatch(entry, pattern) or fnmatch(entry + _SYNTHETIC_SUFFIX, pattern)
        for pattern in patterns
    )


def _workloads() -> list[Any]:
    platform = _module("_deploy_platform_iam", "deploy/platform.py")
    return list(platform.WORKLOADS)


def _policy_for(component: str) -> tuple[Path, dict[str, Any]] | None:
    """The policy document a workload's role reads, or None if it has no role here.

    Named by convention -- `deploy/iam/map-<component>.json` -- and the convention is
    checked rather than assumed: `test_the_naming_convention_finds_a_policy` asserts at
    least one workload resolves, so a rename that makes every lookup miss fails loudly
    instead of turning this file into a pass over an empty set.
    """
    path = _IAM / f"map-{component}.json"
    if not path.exists():
        return None
    return path, json.loads(path.read_text())


def test_the_naming_convention_finds_a_policy() -> None:
    """At least one workload resolves to a policy file, or every check below is vacuous.

    This is the assertion that fails if `deploy/iam/` is reorganised. Without it, a
    directory rename turns the parametrised tests into a sweep over nothing and the file
    reports success while reading no policy at all.
    """
    resolved = [w.component for w in _workloads() if _policy_for(w.component)]
    assert resolved, (
        "no workload in platform.WORKLOADS resolves to a "
        "deploy/iam/map-<component>.json policy; looked for "
        f"{sorted(f'map-{w.component}.json' for w in _workloads())} "
        f"in {_IAM}. Either the naming convention changed or the directory moved, and "
        "until this is fixed the checks in this file read no policies."
    )


@pytest.mark.parametrize("component", [w.component for w in _workloads()])
def test_every_statement_in_the_policy_is_one_this_file_understands(
    component: str,
) -> None:
    """No `Deny`, no `NotAction`, no `NotResource`, no `Condition` on a read grant.

    Each of those changes what the policy means in a way the glob comparison below does
    not model, so meeting one means this file's answer is no longer about the real
    policy. Refused rather than approximated: an approximation here reads as a pass.
    """
    found = _policy_for(component)
    if found is None:
        pytest.skip(f"no deploy/iam/map-{component}.json")
    path, policy = found
    for statement in policy.get("Statement", []):
        sid = statement.get("Sid", "<unnamed>")
        assert statement.get("Effect") == "Allow", (
            f"{path.name} statement {sid} is not an Allow. This file models Allow "
            "statements only, so its answer would no longer describe this policy."
        )
        for unmodelled in ("NotAction", "NotResource", "Condition"):
            assert unmodelled not in statement, (
                f"{path.name} statement {sid} carries {unmodelled}, which changes what "
                "the policy grants in a way the glob comparison here does not model. "
                "Extend the comparison before adding it."
            )


@pytest.mark.parametrize("component", [w.component for w in _workloads()])
def test_every_declared_vault_entry_is_readable_by_the_role(component: str) -> None:
    """An entry the role cannot read fails every request, hours after a clean deploy.

    This is the check that was missing when `router.py` documented its own
    misconfiguration in a comment. The failure it prevents is not a crash and not a
    probe failure: the pods are Ready, `kubectl rollout status` exits 0, and the service
    returns a 502 for every model call because it cannot fetch the key it verifies
    tokens with.
    """
    found = _policy_for(component)
    if found is None:
        pytest.skip(f"no deploy/iam/map-{component}.json")
    path, policy = found
    platform = _module("_deploy_platform_iam", "deploy/platform.py")
    workload = next(w for w in _workloads() if w.component == component)
    declared = platform.declared_vault_entries(_ROOT, workload)
    if not declared:
        pytest.skip(f"{component} names no vault entry")
    patterns = _readable_patterns(policy)
    unreadable = [
        f"{name} (named by {why})"
        for why, name in declared
        if not _allows(patterns, name)
    ]
    assert not unreadable, (
        f"{path.name} does not allow secretsmanager:GetSecretValue on: "
        f"{unreadable}. Its read patterns are {patterns}. A workload whose manifest "
        "names an entry its role cannot read starts, passes every probe, completes its "
        "rollout, and fails every request that needs the entry."
    )


def test_the_comparison_can_report_a_gap() -> None:
    """The matcher says no to something, or its yes means nothing.

    Directly falsifies `_allows`: a policy scoped to one prefix must refuse an entry
    outside it. Without this, a bug that made `_allows` return True unconditionally
    would leave every assertion above passing and the guard measuring nothing -- which
    is the state `docs/lessons.md` records this repository reaching six times over.
    """
    patterns = _readable_patterns(
        {
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["secretsmanager:GetSecretValue"],
                    "Resource": (
                        "arn:aws:secretsmanager:us-east-1:1:secret:map/dev/providers/*"
                    ),
                }
            ]
        }
    )
    assert _allows(patterns, "map/dev/providers/anthropic")
    assert not _allows(patterns, "map/dev/platform/pod-token-signing-key")


def test_a_single_entry_policy_matches_the_bare_name() -> None:
    """The `-*` an ARN needs must not read as a gap against the name without a suffix.

    `declared_vault_entries` yields `map/dev/platform/pod-token-signing-key`; the policy
    must be written `...pod-token-signing-key-*` because Secrets Manager appends six
    characters to the ARN. A matcher comparing only the bare name calls a correct policy
    broken, which would push whoever hit it toward widening the policy to a prefix.
    """
    patterns = _readable_patterns(
        {
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["secretsmanager:GetSecretValue"],
                    "Resource": (
                        "arn:aws:secretsmanager:us-east-1:1:secret:"
                        "map/dev/platform/pod-token-signing-key-*"
                    ),
                }
            ]
        }
    )
    assert _allows(patterns, "map/dev/platform/pod-token-signing-key")
    assert not _allows(patterns, "map/dev/platform/db")


def test_a_statement_granting_no_read_action_contributes_nothing() -> None:
    """A write-only or describe-only statement is not a read grant.

    `DescribeSecret` tells a caller an entry exists and nothing about its value, so a
    policy holding only that would leave every fetch failing while looking permissive.
    """
    patterns = _readable_patterns(
        {
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["secretsmanager:DescribeSecret"],
                    "Resource": "arn:aws:secretsmanager:us-east-1:1:secret:map/dev/*",
                }
            ]
        }
    )
    assert patterns == []
    assert not _allows(patterns, "map/dev/providers/anthropic")


_COMPOSED_PREFIXES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "tool-gateway",
        "gateway.tool.credential_broker.VAULT_PREFIX",
        credential_broker.VAULT_PREFIX,
    ),
    (
        "control-plane",
        "control.webhooks.dispatcher.WEBHOOK_SECRET_PREFIX",
        dispatcher.WEBHOOK_SECRET_PREFIX,
    ),
)
"""Every prefix a workload's code builds a vault name under, and who builds it.

Imported rather than spelled, so a prefix renamed in the source is renamed here and
this file cannot go on asserting about a string production stopped using.

The checks above cannot see any of these, and that is structural rather than an
oversight: they derive their names from `declared_vault_entries`, which reads the
manifest, and a name with a tenant in the middle of it is composed per request from
the calling tenant's id. There is no manifest line for it to appear on. So the guard
that exists covers exactly the entries a deploy can enumerate and is blind to every
entry a request invents -- which is the whole tenant-scoped half of this platform.
"""


@pytest.mark.parametrize(
    ("component", "source", "prefix"),
    _COMPOSED_PREFIXES,
    ids=[one[1] for one in _COMPOSED_PREFIXES],
)
def test_every_prefix_the_code_composes_is_one_the_role_can_read(
    component: str, source: str, prefix: str
) -> None:
    """A prefix the code composes under must be a prefix the role can read.

    The failure is silent in the way that costs the most: nothing at deploy time
    consults these constants, so a policy naming a prefix no code composes and code
    composing a prefix no policy names both look correct in isolation. Measured on
    2026-08-25, `map-tool-gateway` was denied `GetSecretValue` on every name
    `vault_name` can build -- so no authenticated MCP registration could ever have
    worked in this account, while the public-server path went on passing because it
    reads no credential at all.

    A representative name rather than the bare prefix, because a bare prefix is not a
    name any fetch uses and a policy could match it while refusing everything real.
    The tenant is a syntactically valid uuid so the composed name has the shape a
    request produces.
    """
    found = _policy_for(component)
    assert found is not None, f"no deploy/iam/map-{component}.json for {source}"
    path, policy = found
    composed = f"{prefix}/40e9465c-0000-4000-8000-000000000000/a-ref"
    patterns = _readable_patterns(policy)
    assert _allows(patterns, composed), (
        f"{source} composes vault names under {prefix!r}, and {path.name} allows "
        f"secretsmanager:GetSecretValue on {patterns} -- none of which match "
        f"{composed!r}. Every fetch under this prefix fails at AWS with AccessDenied, "
        "hours after a deploy that reported success."
    )
