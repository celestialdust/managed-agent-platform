"""Nothing committed here names a real AWS account or a real Foundry resource.

Both used to. The account id sat in ninety-odd places across twenty-two files -- every
ARN in the Terraform configuration, every IRSA annotation, the bucket names, the
registry host -- and one Azure resource name sat in the Model Gateway's routing table.
Neither is a credential, and neither was a problem while the repository was private.
Published, they are a company's account and a company's endpoint, handed to anyone who
clones it.

They are gone by being *derived* rather than by being scrubbed, which is the part this
grades. The account comes from `aws sts get-caller-identity` -- in Terraform through
`data.aws_caller_identity` (`deploy/terraform/account.tf`), in the deploy scripts
through `caller_account()`, and in the two live-tier tests that need it through a call
of their own. What is committed is a twelve-zero placeholder that parses everywhere an
account id parses and names no account anywhere, so a manifest applied without
substitution fails loudly rather than reaching somebody else's account.

**Why a pattern and not a blocklist of the two old values.** A blocklist grades history.
The question here is about the next account id somebody pastes in while debugging, which
is not a value this file can know. So it matches the *positions* an account id occupies
-- the account field of an ARN, the registry host, the bucket names -- and requires each
one to hold the placeholder or an expression that resolves to it.

A bare twelve-digit run is deliberately not matched. Measured: it hits thirty-seven
files here, almost all of them UUID segments like `000000000001` in a fixture, and a
guard that cries wolf thirty-seven times is one somebody switches off.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parent.parent

ACCOUNT_PLACEHOLDER: Final = "0" * 12
"""The same twelve zeros `deploy/platform.py` and `deploy/terraform/account.tf` carry.

Written out a third time rather than imported, and that is the one duplication here
worth having: this file exists to grade what the committed tree contains, so importing
the constant from the code under test would let a change to that constant move the
goalposts and the tree it is checking at the same time.
"""

FOUNDRY_PLACEHOLDER: Final = "map-foundry"
"""The Foundry resource name the committed routing table carries. Resolves nowhere."""

_ALLOWED_ACCOUNTS: Final = frozenset(
    {
        ACCOUNT_PLACEHOLDER,
        # Any twelve digits, used by the bootstrap tests to prove the substitution
        # happened. Deliberately not the placeholder, so a case asserting the
        # placeholder cannot pass by coincidence -- see `_AN_ACCOUNT` in
        # `tests/deploy/test_cluster_bootstrap.py`.
        "210987654321",
    }
)

# The three places a twelve-digit account id actually appears, and one expression form.
# `\$\{[^}]*\}` admits `${local.account_id}` and `${account_id}` without admitting a
# literal, because a literal contains no braces.
_AN_ACCOUNT_FIELD: Final = r"(\$\{[^}]*\}|[0-9]{12})"

_IN_AN_ARN: Final = re.compile(
    r"arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:" + _AN_ACCOUNT_FIELD + r":"
)
_IN_A_REGISTRY_HOST: Final = re.compile(_AN_ACCOUNT_FIELD + r"\.dkr\.ecr\.")
_IN_A_BUCKET_NAME: Final = re.compile(r"map-(?:dev|prod)-(?:tfstate-)?([0-9]{12})-")

_A_FOUNDRY_HOST: Final = re.compile(r"([A-Za-z0-9_-]+)\.services\.ai\.azure\.com")

_THE_FALSIFICATION_FIXTURES: Final = "tests/test_the_tree_names_no_real_account.py"
"""This file, which is exempt from its own scan, and is the only file that is.

It has to hold the dirty shapes literally -- an ARN and a registry host and a bucket
name carrying an account id, a Foundry host carrying a resource name -- because
`test_the_patterns_catch_the_values_that_were_actually_committed` is what proves the two
guards above are checking a tree rather than passing over one. Assembling those strings
from parts at runtime would dodge the scan and would also hide, from the person reading
the fixture, exactly the shape it claims to catch.

The exemption is safe only because every value in it is invented. The account digits are
the first twelve of pi, and the Foundry resource is a name nobody registered. A real
value must never be pasted in here to make the fixture "realistic": this is the one file
where the guard cannot catch it.
"""

_TEXT_SUFFIXES: Final = frozenset(
    {
        ".py",
        ".tf",
        ".json",
        ".yaml",
        ".yml",
        ".sh",
        ".md",
        ".toml",
        ".hcl",
        ".example",
        ".cfg",
        ".ini",
        ".txt",
    }
)


def _committed_text_files() -> tuple[Path, ...]:
    """Every tracked file this can read as text, as git lists them.

    Driven off `git ls-files` rather than a directory walk, so a file that is present
    but untracked -- a developer's `terraform.tfvars`, a scratch note -- is out of
    scope. It is what git holds that gets published.

    One tracked file is dropped: this one, for the reason `_THE_FALSIFICATION_FIXTURES`
    gives.
    """
    listing = subprocess.run(
        ("git", "-C", str(_ROOT), "ls-files", "-z"),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return tuple(
        _ROOT / name
        for name in listing.split("\0")
        if name
        and Path(name).suffix in _TEXT_SUFFIXES
        and name != _THE_FALSIFICATION_FIXTURES
    )


def _accounts_named_in(text: str) -> set[str]:
    found = {match.group(1) for match in _IN_AN_ARN.finditer(text)}
    found |= {match.group(1) for match in _IN_A_REGISTRY_HOST.finditer(text)}
    found |= {match.group(1) for match in _IN_A_BUCKET_NAME.finditer(text)}
    return {
        account
        for account in found
        if not account.startswith("${") and account not in _ALLOWED_ACCOUNTS
    }


def test_no_committed_file_names_a_real_aws_account() -> None:
    offenders: list[str] = []
    for path in _committed_text_files():
        named = _accounts_named_in(path.read_text(encoding="utf-8", errors="replace"))
        if named:
            offenders.append(f"{path.relative_to(_ROOT)}: {sorted(named)}")

    assert not offenders, (
        "these committed files name a real AWS account, which publishes it to everyone "
        f"who clones this repository. Use {ACCOUNT_PLACEHOLDER} and let the deploy "
        "substitute the caller's own account -- deploy/terraform/account.tf and "
        "deploy/platform.py's caller_account() are the two ends of that:\n  "
        + "\n  ".join(offenders)
    )


def test_no_committed_file_names_a_real_foundry_resource() -> None:
    """The Model Gateway's upstream host names one company's Azure resource.

    Same argument as the account and a weaker guarantee, because there is nothing to
    derive it from: a Foundry resource is a choice rather than a property of the
    credentials in the environment. So it is configured at deploy time from
    `MAP_FOUNDRY_RESOURCE`, and what is committed is a host that returns NXDOMAIN.
    """
    offenders: list[str] = []
    for path in _committed_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        named = {
            match.group(1)
            for match in _A_FOUNDRY_HOST.finditer(text)
            if match.group(1) != FOUNDRY_PLACEHOLDER
        }
        if named:
            offenders.append(f"{path.relative_to(_ROOT)}: {sorted(named)}")

    assert not offenders, (
        "these committed files name a real Azure Foundry resource. Use "
        f"{FOUNDRY_PLACEHOLDER}, which resolves nowhere, and set MAP_FOUNDRY_RESOURCE "
        "at deploy time:\n  " + "\n  ".join(offenders)
    )


def test_the_patterns_catch_the_values_that_were_actually_committed() -> None:
    """The two guards above pass over a clean tree, so they are graded against a dirty
    one here -- otherwise they would go on passing with their regexes broken.

    The strings are the real *shapes* that were in this repository until they were
    taken out: an ARN, a registry host and a bucket name carrying an account id, and a
    Foundry host carrying a resource name. Every value in them is invented -- see
    `_THE_FALSIFICATION_FIXTURES` for why that is load-bearing rather than tidy.
    """
    dirty = (
        "arn:aws:secretsmanager:us-east-1:314159265358:secret:map/dev/platform/db",
        "314159265358.dkr.ecr.us-east-1.amazonaws.com/map/platform:1.0",
        "map-dev-314159265358-us-east-1-an",
        "map-dev-tfstate-314159265358-us-east-1",
    )
    for text in dirty:
        assert _accounts_named_in(text) == {"314159265358"}, (
            f"the account-id patterns do not catch {text!r}, so the guard above is "
            "passing over a clean tree rather than checking one"
        )

    clean = (
        f"arn:aws:iam::{ACCOUNT_PLACEHOLDER}:role/map-tool-gateway",
        "arn:aws:iam::${local.account_id}:policy/map-control-plane",
        f"map-dev-{ACCOUNT_PLACEHOLDER}-us-east-1-an",
    )
    for text in clean:
        assert not _accounts_named_in(text), f"{text!r} is the substituted form"

    a_named_resource = "acme-w4k2p9x1-eastus2.services.ai.azure.com"
    matched = _A_FOUNDRY_HOST.search(a_named_resource)
    assert matched is not None and matched.group(1) != FOUNDRY_PLACEHOLDER
    placeholder = _A_FOUNDRY_HOST.search(f"{FOUNDRY_PLACEHOLDER}.services.ai.azure.com")
    assert placeholder is not None and placeholder.group(1) == FOUNDRY_PLACEHOLDER
