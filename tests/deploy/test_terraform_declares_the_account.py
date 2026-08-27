"""The Terraform configuration adopts the account rather than proposing to build one.

Three tiers, and the split is about what each can prove.

Tier A parses the `.tf` files as text and needs no terraform binary, no AWS
credentials and no network. It grades the properties a reader cannot check by
eye: that every declared resource is paired with an `import` block (an unpaired
resource is a plan to CREATE a second copy of something that already exists --
measured: `Plan: 1 to add`), that the resources whose loss would end the
environment carry `prevent_destroy`, that every IAM role's policy sets are
declared *exclusively* rather than one attachment at a time, that no resource
type here can carry a secret VALUE, and that state is kept out of the bucket a
Session pod can read.

Tier B is the comparison itself and it cannot be faked: it runs the same
`tools/terraform_drift.py` the hand-run gate runs, against the real account, and
enumerates the account's IAM roles -- the one class of drift `terraform plan`
provably cannot see. It is behind `MAP_TERRAFORM_DRIFT=1` because a bare `pytest`
must not need AWS credentials, and its skip reason says the account was NOT
compared -- a skipped run of this file is not a passing one.

Tier C grades the gate tool's own contract with a fake `terraform` on `PATH`, so
it runs in the default offline suite. It exists because the whole suite was once
green while the gate could not run at all: `terraform_drift.py` exited 1 --
"nothing was compared" -- and no test noticed. A tool that reported that as
agreement would be worse than no gate, so the mapping from terraform's exit code
to the tool's is asserted rather than read.

What Tier A cannot say: nothing there reads the account, so a config that is
internally perfect and describes a different account passes every text test
below. That is Tier B's job and there is no substitute for it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_TERRAFORM = _ROOT / "deploy" / "terraform"
_K8S = _ROOT / "deploy" / "k8s"
_DRIFT_TOOL = _ROOT / "tools" / "terraform_drift.py"

_PLATFORM_BUCKET = "map-dev-000000000000-us-east-1-an"
"""The platform bucket as the committed tree spells it, at the account placeholder.

Both names compared against this carry the same placeholder, so the comparison is still
between two bucket names rather than between a placeholder and a real account -- which
would differ for free and prove nothing."""

# Losing any of these breaks the development environment, and the reasons differ
# enough to be worth separating. The RDS instance holds the Event Log and the
# state bucket holds the record of what everything else is: those two are
# irreplaceable outright. The rest are load-bearing by *identity* -- AWS will
# make you another default subnet, but not with the same id, and the cluster's
# `subnet_ids`, the NAT gateway and the DB subnet group all name ids. A
# destroy-and-recreate of any of them is an outage plus an id chase, which is why
# a plan proposing one should be a refusal (exit 1) rather than a diff.
#
# The two private subnets are here because the nodegroup and the DB subnet group
# sit in them and nothing recreates them at all -- they are hand-made, not
# defaults. An earlier version of this file guarded the two *default* subnets and
# not these, on the reasoning that losing a private subnet "costs a rebuild and
# an RDS move, not data". A rebuild plus an RDS move is precisely the outcome
# `prevent_destroy` exists to refuse.
#
# Each bucket's `versioning` and `public_access_block` are guarded and its
# `server_side_encryption_configuration` is not, and that asymmetry is deliberate:
# S3 applies SSE-S3 to every object whether or not a configuration resource
# exists, so destroying that resource changes nothing observable, while destroying
# a versioning or public-access-block resource silently removes a property
# ADR-021 requires of the state bucket and `environment.md` requires of the
# platform bucket.
_MUST_NOT_BE_DESTROYED = frozenset(
    {
        "aws_default_vpc.map",
        "aws_default_subnet.public_1a",
        "aws_default_subnet.public_1b",
        "aws_subnet.private_1a",
        "aws_subnet.private_1b",
        "aws_internet_gateway.default",
        "aws_route_table.main",
        "aws_eip.nat",
        "aws_nat_gateway.map",
        "aws_eks_cluster.map_dev",
        "aws_eks_node_group.map_dev_nodes_m6i",
        "aws_db_instance.map_dev",
        "aws_iam_openid_connect_provider.cluster",
        "aws_iam_policy.provisioning",
        "aws_secretsmanager_secret.platform_db",
        "aws_secretsmanager_secret.provider_anthropic",
        "aws_s3_bucket.platform",
        "aws_s3_bucket_versioning.platform",
        "aws_s3_bucket_public_access_block.platform",
        "aws_s3_bucket.tfstate",
        "aws_s3_bucket_versioning.tfstate",
        "aws_s3_bucket_public_access_block.tfstate",
    }
)

# Every one of these stores a secret VALUE, so importing one writes that value
# into Terraform state. `map/dev/platform/db` and `map/dev/providers/anthropic`
# are declared as secret CONTAINERS above; their versions are deliberately not.
_CARRIES_A_SECRET_VALUE = (
    "aws_secretsmanager_secret_version",
    "aws_ssm_parameter",
    "aws_secretsmanager_secret_rotation",
)

# Any argument whose NAME contains "password". Deliberately wider than the
# argument that spells it exactly: `aws_db_instance` in provider 6.61.0 also
# takes `password_wo` and `password_wo_version` (both appear in this
# configuration's own plan JSON), and `aws_rds_cluster` takes `master_password`.
# Measured against the previous form, which anchored on `^\s*password\s*=`:
# `password = "hunter2"` was caught while `password_wo = "hunter2"` and
# `master_password = "hunter2"` both passed. `.gitignore`'s own Terraform comment
# states the cost of missing one -- git does not forget, so deleting the file
# later leaves the secret in history.
#
# `manage_master_user_password` matches too, and that is not collateral damage:
# `data.tf` argues it must never be written here, because the provider does not
# refresh it from the API and so it can never detect the drift it appears to
# guard. No argument naming a password belongs in this directory at all.
#
# The name is matched at the start of a line, so prose mentioning a password in a
# `description` is not a hit -- `secrets.tf` has one.
_PASSWORD_ARGUMENT = re.compile(
    r"^\s*[A-Za-z0-9_]*password[A-Za-z0-9_]*\s*=", re.MULTILINE
)

_ROLE_ANNOTATION = "eks.amazonaws.com/role-arn"

_IMPORT_TO = re.compile(r"^\s*to\s*=\s*(\S+)\s*$", re.MULTILINE)
_IMPORT_ID = re.compile(r'^\s*id\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
_RESOURCE = re.compile(r'^resource\s+"([a-z0-9_]+)"\s+"([a-z0-9_]+)"\s*\{', re.M)
_RESOURCE_BLOCK = re.compile(
    r'^resource\s+"([a-z0-9_]+)"\s+"([a-z0-9_]+)"\s*\{\n(.*?)^\}',
    re.MULTILINE | re.DOTALL,
)
_ROLE_NAME_REF = re.compile(
    r"^\s*role_name\s*=\s*aws_iam_role\.([a-z0-9_]+)\.name", re.MULTILINE
)
# An IAM role's own `name`, which is what the account calls it. Indented, so a
# top-level argument of some other kind cannot be read as one, and literal, so a
# role whose name is computed fails loudly instead of vanishing from the compare.
_NAME_ARGUMENT = re.compile(r'^\s+name\s*=\s*"([^"]+)"', re.MULTILINE)

# The two resource types that declare a role's policy set authoritatively, as
# opposed to one member of it.
_EXCLUSIVE_SET_TYPES = (
    "aws_iam_role_policy_attachments_exclusive",
    "aws_iam_role_policies_exclusive",
)

# The `.tf` files whose resources this configuration CREATES rather than adopts,
# each with the reason it cannot be paired with an import block.
#
# Named one file at a time rather than reached by widening `_tf_files()`: a wider
# glob would silently stop grading the other nine files' imports, and the pairing
# is the parse contract the rest of this module rests on. The exemption is also
# two-sided -- `test_a_created_not_imported_file_carries_no_import_block` requires
# a file listed here to carry ZERO import blocks -- so this is an inverted rule
# rather than a hole, and an import block appearing in one of these files fails
# just as loudly as a missing one elsewhere.
#
# THIS EXEMPTION REACHES EXACTLY TWO CASES, and keeping it that narrow took work.
# It reached six, because four more cases walked `_pairs` to answer questions that
# were not about import blocks at all -- and a file with no import blocks yields no
# pairs, so it left `_declared_role_names`, the prevent_destroy sweep, the
# exclusive-policy-set case and the manifest cross-read without failing any of
# them. Measured while that was true: `pytest -k whole_policy_set` reported
# `3 passed` while this file declared three roles and not one exclusive set. Those
# four now walk `_resource_blocks`, which has no exemption, and
# `test_a_created_not_imported_files_roles_are_graded_like_any_other` is what
# keeps them there.
_CREATED_NOT_IMPORTED = frozenset(
    {
        # The Session VFS mount: an S3 Files file system, one mount target per node
        # subnet, an access point, a synchronisation configuration, the mount-target
        # security group and its NFS ingress rule, three IAM roles with their
        # policies, and the EFS CSI add-on. None of them exists in the account --
        # `aws eks list-addons --cluster-name map-dev` returns no add-ons and
        # `aws s3files list-file-systems` returns an empty list -- so an `import`
        # block for any of them fails the plan outright with `Cannot import
        # non-existent remote object` at exit 1, instead of planning the creations.
        # The absoluteness the rest of this module asserts is correct for a
        # directory where every resource already exists and wrong the moment one
        # does not.
        "session_vfs.tf",
        # The VPC CNI add-on, which carries the flag that decides whether a
        # NetworkPolicy in this cluster filters anything. The CNI itself is running,
        # and it is NOT an add-on: `aws eks list-addons --cluster-name map-dev`
        # returns `aws-efs-csi-driver` and nothing else, and the `aws-node`
        # DaemonSet's labels say `app.kubernetes.io/managed-by: Helm`, which is how
        # EKS installs the default CNI for a cluster created without the add-on. So
        # there is nothing to import -- there is a self-managed installation for the
        # add-on to take over, which is what `resolve_conflicts_on_create =
        # "OVERWRITE"` is there for -- and an `import` block would fail the plan the
        # same way `session_vfs.tf`'s would.
        "vpc_cni.tf",
    }
)


def _pairs(path: Path) -> list[tuple[str, str, str]]:
    """Split one `.tf` file into its (import target, import id, body) triples.

    The file shape this relies on is a requirement, not an observation: each
    resource is preceded by its own `import` block and nothing else sits between
    them. That shape is what makes a text parse exact -- splitting on `import {`
    yields one chunk per resource, so `prevent_destroy` found in a chunk belongs
    to that chunk's resource and cannot be attributed across a boundary. A file
    that breaks the shape fails `test_every_terraform_file_is_import_then_resource`
    rather than being silently mis-parsed.
    """
    chunks = re.split(r"^import \{", path.read_text(), flags=re.MULTILINE)[1:]
    triples: list[tuple[str, str, str]] = []
    for chunk in chunks:
        target = _IMPORT_TO.search(chunk)
        ident = _IMPORT_ID.search(chunk)
        assert target and ident, f"{path.name}: an import block names no to/id"
        triples.append((target.group(1), ident.group(1), chunk))
    return triples


def _tf_files() -> list[Path]:
    return sorted(p for p in _TERRAFORM.glob("*.tf") if p.name != "versions.tf")


def _resource_blocks(path: Path) -> list[tuple[str, str, str]]:
    """Every top-level `resource` block in one file, as (type, name, body).

    This is the parse to use for any question about a RESOURCE, as opposed to a
    question about its import block. `_pairs` answers the latter and is the wrong
    tool for the former, because a file with no import blocks yields no pairs: it
    drops out of the walk silently, and a test written over `_pairs` then reports
    that it found nothing wrong. Four cases below were written that way and
    `session_vfs.tf` -- the one file exempt from import pairing -- left all four
    without a word.

    The body ends at the first `}` in column 0, which is exact for this directory
    because every nested brace here is indented. The assertion below compares the
    (type, name) pairs this reads against `_RESOURCE`, which finds resource HEADERS
    by a different pattern, so a resource the body pattern skips outright -- or one
    it finds that the header pattern does not -- fails here rather than dropping out
    of the walk.

    What that assertion does NOT catch is a body cut short, and it is worth being
    exact about because the obvious reading of it is wrong. A stray `}` in column 0
    inside a body ends that body early and leaves the pair sequence untouched, so
    the counts still agree: measured, a `}` inserted inside
    `aws_iam_role.s3files_csi_node` returns its body as one line, without
    `assume_role_policy` in it and without raising. It is a header/body agreement
    check, not a truncation check.

    Nothing here is vacuous because of that, and the reason is a property of the
    callers rather than of this function: all four assert on what a body CONTAINS --
    a literal `name`, a `prevent_destroy = true`, a `role_name` reference -- so a
    truncated body fails them instead of passing. A case asserting something ABSENT
    from a body would be the vacuous one, and would need a truncation guard of its
    own before it could be trusted.
    """
    text = path.read_text()
    found = [
        (match.group(1), match.group(2), match.group(3))
        for match in _RESOURCE_BLOCK.finditer(text)
    ]
    assert [(kind, name) for kind, name, _ in found] == _RESOURCE.findall(text), (
        f"{path.name}: the resource bodies this reads do not line up with the "
        f"resource headers, so some resource is being read partly or not at all"
    )
    return found


def _declared_role_names() -> set[str]:
    """The IAM role NAMES this configuration declares, read off the `name` argument.

    Read off `name` rather than off the import id, which is what this did first and
    what made it blind to exactly the roles that most needed watching. The import id
    of an `aws_iam_role` IS its name, so the two agreed for as long as every role was
    adopted -- and then `session_vfs.tf` declared three roles that are created rather
    than adopted, so they carry no import block and can carry none. Measured at the
    time: this returned 5 names while the configuration declared 8, which made the
    account enumeration below fail permanently the moment those three were applied,
    and made the manifest cross-read unable to see a role bound under
    `deploy/k8s/`. `name` is the argument the account actually reads.
    """
    names: set[str] = set()
    for path in _tf_files():
        for kind, local_name, body in _resource_blocks(path):
            if kind == "aws_iam_role":
                found = _NAME_ARGUMENT.search(body)
                assert found, (
                    f"{path.name}: aws_iam_role.{local_name} declares no literal "
                    f"name, so nothing can compare it to the account"
                )
                names.add(found.group(1))
    return names


def _imported_role_names() -> set[str]:
    """The IAM role names this configuration says ALREADY EXIST, off the import ids.

    A strictly smaller set than `_declared_role_names()`, and the difference matters
    in one direction only: a role declared with an import block is a claim about the
    account, so it had better be there. A role declared WITHOUT one is a creation the
    plan has yet to make, and its absence from the account is what `terraform plan`
    reports as `1 to add` rather than drift.
    """
    names: set[str] = set()
    for path in _tf_files():
        for target, ident, _ in _pairs(path):
            if target.startswith("aws_iam_role."):
                names.add(ident)
    return names


def _annotated_role_names() -> set[str]:
    """Role names the `deploy/k8s/` manifests ask a pod to assume, by IRSA.

    Walked out of the parsed YAML rather than grepped, so a role arn under a
    differently-indented or differently-quoted annotation is still found.
    """
    names: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == _ROLE_ANNOTATION and isinstance(value, str):
                    names.add(value.rsplit("/", 1)[-1])
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for path in sorted(_K8S.glob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text()):
            walk(document)
    return names


def test_the_configuration_directory_exists_and_holds_more_than_a_backend() -> None:
    assert _tf_files(), f"no .tf files besides versions.tf under {_TERRAFORM}"


def test_every_terraform_file_is_import_then_resource() -> None:
    """Each resource has its own import block, immediately before it.

    This is the parse contract the rest of this file rests on, so it is asserted
    directly rather than assumed. `_CREATED_NOT_IMPORTED` is skipped here and
    graded by the inverted rule below instead.
    """
    for path in _tf_files():
        if path.name in _CREATED_NOT_IMPORTED:
            continue
        text = path.read_text()
        declared = [f"{t}.{n}" for t, n in _RESOURCE.findall(text)]
        imported = [target for target, _, _ in _pairs(path)]
        assert imported == declared, (
            f"{path.name}: import order {imported} does not match "
            f"resource order {declared}"
        )


def test_no_resource_is_declared_without_an_import_block() -> None:
    """A resource with no import block is a plan to create a duplicate.

    Measured against the real account: removing one `import` block turned the
    plan from `36 to import, 0 to add` into `35 to import, 1 to add`, and the
    apply would have failed on a repository that already exists.

    `_CREATED_NOT_IMPORTED` is skipped here for the opposite reason: those
    resources do not exist, so creating them is the intent and importing one
    would fail the plan. The inverted rule below is what grades them.
    """
    for path in _tf_files():
        if path.name in _CREATED_NOT_IMPORTED:
            continue
        declared = {f"{t}.{n}" for t, n in _RESOURCE.findall(path.read_text())}
        imported = {target for target, _, _ in _pairs(path)}
        assert declared - imported == set(), (
            f"{path.name}: declared but not imported: {sorted(declared - imported)}"
        )
        assert imported - declared == set(), (
            f"{path.name}: imported but not declared: {sorted(imported - declared)}"
        )


def test_a_created_not_imported_file_carries_no_import_block() -> None:
    """The other half of the exemption, which is what keeps it a rule.

    An exemption that only *stopped* grading a file would let an import block
    appear in it unnoticed -- and an import block naming a resource that does not
    exist fails the whole plan at exit 1 with `Cannot import non-existent remote
    object`, taking the drift comparison down with it. So a file listed as
    created-not-imported must carry no import block at all, and must still
    declare resources: an exempt file that declares none is a stale entry.
    """
    for name in sorted(_CREATED_NOT_IMPORTED):
        path = _TERRAFORM / name
        assert path.exists(), f"{name} is exempt from import pairing but is not here"
        text = path.read_text()
        assert _RESOURCE.findall(text), (
            f"{name} declares no resource, so its exemption is stale -- drop it "
            f"from _CREATED_NOT_IMPORTED"
        )
        assert _pairs(path) == [], (
            f"{name} carries an import block, but its resources do not exist in "
            f"the account: importing one fails the plan at exit 1. If one now "
            f"does exist, move it to a file that pairs it with its import."
        )


def test_a_created_not_imported_files_roles_are_graded_like_any_other() -> None:
    """The exemption is about import blocks and must reach nothing else.

    It reached four more cases when it was first written, all of them because they
    walked `_pairs`: a file with no import block yields no pairs, so it left each one
    without failing it. This is the sharpest of the four to assert directly, because
    its consequence was the worst -- a role invisible to `_declared_role_names()` is
    invisible to the account enumeration below AND to the manifest cross-read, so a
    role bound by a `deploy/k8s/` manifest could be declared here and still counted as
    undeclared. Measured while that was true: 5 names returned for 8 declared roles.

    THE ANTI-VACUITY FLOOR IS OVER THE SET AND NOT PER FILE, and the distinction is
    what keeps this case honest without deciding what an exempt file may hold. It used
    to require every exempt file to declare a role, which was the same assertion while
    `session_vfs.tf` was the only entry and became a different one the moment a second
    arrived: `vpc_cni.tf` declares one add-on and no IAM anything, and a per-file
    requirement would have been satisfiable only by moving an unrelated resource into
    it to feed a test. The property being checked is that the exemption does not hide
    a role from `_declared_role_names()`, and a file with no roles hides none. What
    would make the case vacuous is the exemption holding no roles AT ALL, and that is
    the assertion below the loop.
    """
    declared = _declared_role_names()
    graded = 0
    for file_name in sorted(_CREATED_NOT_IMPORTED):
        path = _TERRAFORM / file_name
        roles = [
            (name, body)
            for kind, name, body in _resource_blocks(path)
            if kind == "aws_iam_role"
        ]
        for local_name, body in roles:
            graded += 1
            found = _NAME_ARGUMENT.search(body)
            assert found and found.group(1) in declared, (
                f"{file_name}: aws_iam_role.{local_name} is invisible to "
                f"_declared_role_names(), so both the account enumeration and the "
                f"manifest cross-read silently skip it"
            )
    assert graded, (
        f"no file in {sorted(_CREATED_NOT_IMPORTED)} declares an IAM role, so this "
        "case iterated nothing and proves nothing about the exemption's reach; the "
        "reach has to be checked another way"
    )


def test_every_import_id_is_readable_once_the_account_is_substituted() -> None:
    """An import id may interpolate the account, and nothing else.

    A wrong id fails loudly at plan time (`Cannot import non-existent remote object`,
    exit 1 -- measured), but only if a human can read what it names. An id assembled
    from an arbitrary expression cannot be read off the page, and worse, one that
    referenced a *managed resource* would not be known at plan time at all.

    This asserted no interpolation whatsoever until the account stopped being written
    down. `local.account_id` is now the one permitted reference: it comes from
    `aws_caller_identity`, resolves at plan time -- measured, a scratch configuration
    importing an IAM policy by an id built from it planned `1 to import` -- and it is
    the same value in every id, so an id carrying it is still one fixed string a reader
    can check. Every other interpolation stays refused, which is the half of this that
    was doing the work.
    """
    permitted = "${local.account_id}"
    for path in _tf_files():
        for target, ident, _ in _pairs(path):
            assert ident.strip() == ident and ident, f"{path.name}: {target} blank id"
            rest = ident.replace(permitted, "")
            assert "${" not in rest, (
                f"{path.name}: {target} has a computed id. {permitted} is the only "
                f"interpolation an import id may carry; this one is {ident!r}"
            )


def test_the_irreplaceable_resources_refuse_to_be_destroyed() -> None:
    """`prevent_destroy` on each, which turns a planned destroy into exit 1.

    Measured: with it, a plan that would replace the nodegroup stops with
    `Error: Instance cannot be destroyed` and exits 1 while still printing the
    whole plan, so the diagnosis is not lost. Without it the same plan exits 2
    and an apply would have gone through.
    """
    guarded: set[str] = set()
    for path in _tf_files():
        for kind, name, body in _resource_blocks(path):
            if "prevent_destroy = true" in body:
                guarded.add(f"{kind}.{name}")
    assert guarded >= _MUST_NOT_BE_DESTROYED, (
        f"missing prevent_destroy: {sorted(_MUST_NOT_BE_DESTROYED - guarded)}"
    )


def test_every_role_declares_its_whole_policy_set_and_not_one_attachment() -> None:
    """A role's attached and inline policy sets are declared authoritatively.

    `aws_iam_role_policy_attachment` is per-attachment and non-authoritative:
    terraform refreshes only the attachments already in state, so it never
    enumerates what the role actually carries. Measured against the live account:
    `AmazonEC2ContainerRegistryReadOnly` is attached to `map-dev-eks-node`, and a
    config that declared the other two and not it planned
    `0 to add, 3 to change, 0 to destroy` -- no diff for the undeclared
    attachment at all. (The arn surfaces once in the plan text, inside the role's
    deprecated computed `managed_policy_arns`, which is not a change and is not
    rendered once the resources are in state.) So naming three attachments does
    NOT make a fourth show up, and the escalation is concrete: the EKS S3 CSI
    driver's documented install attaches `AmazonS3ReadOnlyAccess` to the node
    role, which is read access to every bucket in the account -- this
    configuration's own state bucket included -- from any Session pod.

    `aws_iam_role_policy_attachments_exclusive` declares the set instead, and it
    reads the account: with the same policy dropped from its list the plan
    rendered `- "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"` and
    the change count went 3 -> 4. `aws_iam_role_policies_exclusive` does the same
    for inline policies, which matter here because this identity holds
    `iam:PutRolePolicy`.
    """
    roles: set[str] = set()
    covered: dict[str, set[str]] = {kind: set() for kind in _EXCLUSIVE_SET_TYPES}
    for path in _tf_files():
        for kind, name, body in _resource_blocks(path):
            if kind == "aws_iam_role":
                roles.add(name)
            elif kind in covered:
                referenced = _ROLE_NAME_REF.search(body)
                assert referenced, (
                    f"{path.name}: {kind}.{name} does not name its role as "
                    f"aws_iam_role.<name>.name, so the link cannot be checked"
                )
                covered[kind].add(referenced.group(1))
    for kind, seen in covered.items():
        assert seen == roles, f"{kind} does not cover {sorted(roles - seen)}"


def test_every_role_a_manifest_binds_a_pod_to_is_declared_here() -> None:
    """An IRSA annotation names an AWS role, so ADR-021 puts it in this directory.

    `deploy/k8s/cluster-bootstrap.yaml` binds a service account to
    `role/map-tool-gateway` -- the identity in front of every tool call -- and
    that role's trust policy is what decides which service account may assume it.
    A role bound by a manifest and declared nowhere is the shape ADR-021 rejects
    in as many words: widening it, or adding a second subject to its trust
    policy, would have nothing comparing it.
    """
    declared = _declared_role_names()
    for name in sorted(_annotated_role_names()):
        assert name in declared, (
            f"a manifest under deploy/k8s/ binds a pod to role/{name}, which "
            f"this configuration does not declare; declared: {sorted(declared)}"
        )


def test_no_declared_resource_can_hold_a_secret_value() -> None:
    """The secret containers are declared; their versions are not.

    Terraform state records every attribute it reads, so importing a secret
    version writes the secret into the state file. The RDS master password is
    kept out by the same rule from the other direction: the instance is declared
    with no password argument of any spelling, because AWS manages and rotates it.
    """
    for path in _tf_files():
        text = path.read_text()
        for kind in _CARRIES_A_SECRET_VALUE:
            assert f'resource "{kind}"' not in text, f"{path.name} declares {kind}"
        found = _PASSWORD_ARGUMENT.search(text)
        assert not found, (
            f"{path.name} sets {found.group(0).strip() if found else ''} -- no "
            f"argument naming a password belongs here; the RDS master password "
            f"is AWS-managed and only its secret ARN is ever read"
        )


def test_the_password_guard_catches_every_spelling_of_the_argument() -> None:
    """The guard above, falsified in both directions.

    A guard only ever run against a clean tree is a guard whose failure
    behaviour nobody knows. Every name below was read off the provider's own
    schema: `password_wo` and `password_wo_version` are `aws_db_instance`
    arguments in 6.61.0 and appear in this configuration's plan JSON, and
    `master_password` is `aws_rds_cluster`'s.
    """
    for line in (
        '  password = "hunter2"',
        '  password_wo = "hunter2"',
        "  password_wo_version = 1",
        '  master_password = "hunter2"',
        '  master_password_wo = "hunter2"',
        "  manage_master_user_password = true",
        'password="hunter2"',
    ):
        assert _PASSWORD_ARGUMENT.search(line), f"missed: {line}"
    for line in (
        '  description = "No password here: it lives in the RDS-managed secret"',
        '  master_username = "mapadmin"',
        '  name = "map/dev/platform/db"',
    ):
        assert not _PASSWORD_ARGUMENT.search(line), f"false positive: {line}"


def test_state_is_not_kept_in_the_bucket_a_session_pod_can_read() -> None:
    """The Session VFS synchronises the platform bucket into the pod filesystem.

    State there would be readable, and writable, by whatever the agent runs.

    The bucket is not in `versions.tf` any more -- its name embeds the account id and a
    backend block takes no expressions, so it arrives through `-backend-config` and the
    committed file is a partial configuration. That moves this check rather than ending
    it: the name a reader is told to use lives in `backend.hcl.example`, so that is what
    is graded, plus the fact that `versions.tf` names no bucket at all. A bucket
    reappearing inline would put the account back in the repository, which is the other
    half of what this now guards.
    """
    backend = (_TERRAFORM / "versions.tf").read_text()
    assert 'backend "s3"' in backend, "no s3 backend declared"
    assert not re.search(r'bucket\s*=\s*"', backend), (
        "versions.tf names the state bucket inline again. The name carries the account "
        "id, so it belongs in the gitignored backend.hcl -- see the comment there."
    )

    example = (_TERRAFORM / "backend.hcl.example").read_text()
    named = re.search(r'bucket\s*=\s*"([^"]+)"', example)
    assert named, "backend.hcl.example names no bucket, so nobody can init from it"
    assert named.group(1) != _PLATFORM_BUCKET, (
        f"state is in {_PLATFORM_BUCKET}, which the Session VFS mounts"
    )


def test_locking_uses_the_state_bucket_and_not_a_table_nobody_can_read() -> None:
    """`use_lockfile`, because `dynamodb:ListTables` is denied to this identity.

    Measured 2026-08-22: `aws dynamodb list-tables` returns AccessDeniedException,
    so a lock table cannot be inspected, let alone created. `use_lockfile` needs
    Terraform 1.10+; 1.13.4 accepts it and rejects an unknown backend argument
    with "Unsupported argument", so the acceptance is a real positive.
    """
    backend = (_TERRAFORM / "versions.tf").read_text()
    assert "use_lockfile = true" in backend
    assert "dynamodb_table" not in backend


def test_the_provider_pins_a_region_and_names_no_local_profile() -> None:
    """Region literal, credentials from the environment.

    Every subnet and security-group id in this configuration is regional, so a
    region read from `AWS_REGION` could point the comparison at an account that
    holds none of them. `profile` is the opposite case: it names a developer's
    local credential file entry, which is not a property of the account.

    ASSERTED IN THE PROVIDER'S OWN BODY, because the claim is about the provider
    and the file is not the provider. This read
    `'region  = "us-east-1"' in versions or 'region = "us-east-1"' in versions`
    against the whole of `versions.tf`, which spells a region at TWO places: the
    backend's and the provider's. It looked per-site only by an accident of column
    padding -- `terraform fmt` aligns `=` to the longest key in a block, so the
    backend's `region` is padded to two spaces past `use_lockfile` and matches
    neither alternative today. Remove `use_lockfile` and `encrypt` becomes the
    longest key, the backend's spelling re-pads to exactly `region  = "us-east-1"`,
    and the backend then satisfies a claim about the provider with the provider's
    region deleted.

    Both sites are covered, one claim each: the provider's region in the provider's
    body, and every region SPELLED anywhere in the file, so the backend cannot
    drift to another region either.
    """
    versions = (_TERRAFORM / "versions.tf").read_text()
    providers = re.findall(
        r'^provider\s+"aws"\s*\{\n(.*?)^\}', versions, re.MULTILINE | re.DOTALL
    )
    assert len(providers) == 1, f"expected one aws provider, found {len(providers)}"
    assert re.search(r'^\s*region\s*=\s*"us-east-1"\s*$', providers[0], re.MULTILINE), (
        f"the aws provider block pins no region of its own: {providers[0]!r}"
    )
    spelled = re.findall(r'^\s*region\s*=\s*"([^"]+)"', versions, re.MULTILINE)
    assert spelled and set(spelled) == {"us-east-1"}, (
        f"versions.tf spells the regions {sorted(set(spelled))}, not just us-east-1"
    )
    assert not re.search(r"^\s*profile\s*=", versions, re.MULTILINE)


def test_terraform_state_and_provider_cache_are_ignored_and_the_lock_is_not() -> None:
    """State must never be committed; the provider lock must always be.

    Both directions, because an ignore rule wide enough to swallow
    `.terraform.lock.hcl` would silently unpin the provider version -- and two
    people planning against different provider versions can read the same account
    differently.
    """
    for path in (
        "deploy/terraform/.terraform/providers",
        "deploy/terraform/terraform.tfstate",
        "deploy/terraform/terraform.tfstate.backup",
        "deploy/terraform/.terraform.tfstate.lock.info",
        "deploy/terraform/tfplan",
    ):
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=_ROOT,
            check=False,
        )
        assert ignored.returncode == 0, f"{path} would be committed"

    lock = subprocess.run(
        ["git", "check-ignore", "-q", "deploy/terraform/.terraform.lock.hcl"],
        cwd=_ROOT,
        check=False,
    )
    assert lock.returncode != 0, "the provider lock is ignored and must be committed"


def test_the_nodegroup_is_declared_in_exactly_one_place() -> None:
    """One declaration of the nodegroup, and it is the compared one.

    `deploy/spike/nodegroup.yaml` was the other one. It asked for one m6i.xlarge
    against two t3.medium that run, said so in its own header, and stayed wrong for
    a month because nothing compared it to anything -- which is what ADR-021 was
    written about. Terraform declares the nodegroup now; a second declaration
    elsewhere under deploy/ would be uncompared again, and uncompared is the one a
    reader trusts.

    Three things this walk skips, all because a declaration is something a person
    wrote. `__pycache__` holds bytecode compiled from `deploy/bootstrap.py`, which
    is itself graded here, so the cache adds no coverage and is not decodable as
    text -- reading it raised UnicodeDecodeError from inside this assertion, which
    named a Terraform test for a Python bytecode file and told the reader nothing.
    `errors="ignore"` is the same defence generalised: the two tokens searched for
    are ASCII, so a byte this cannot decode is a byte that cannot be part of a
    match, and a future binary under `deploy/` should make this test report a
    finding rather than crash. And `.terraform/` is the downloaded provider --
    hundreds of megabytes of somebody else's fixtures, which `git check-ignore`
    already refuses to commit.
    """
    elsewhere = [
        p
        for p in (_ROOT / "deploy").rglob("*")
        if p.is_file()
        and _TERRAFORM not in p.parents
        and "__pycache__" not in p.parts
        and ".terraform" not in p.parts
        and re.search(r"instanceType|managedNodeGroups", p.read_text(errors="ignore"))
    ]
    assert elsewhere == [], f"a second nodegroup declaration: {elsewhere}"


# --- Tier C: the gate tool's own contract, with a fake terraform on PATH. -----

_FAKE_TERRAFORM = """\
#!/bin/sh
# A fake terraform. `fmt` returns MAP_FAKE_FMT_EXIT and `plan` returns
# MAP_FAKE_PLAN_EXIT, and both print something, so the tool's pass-through of
# terraform's output is exercised rather than assumed.
case "$1" in
  fmt)  echo "fake fmt";  exit "${MAP_FAKE_FMT_EXIT:-0}"  ;;
  plan) echo "fake plan"; exit "${MAP_FAKE_PLAN_EXIT:-0}" ;;
esac
echo "fake terraform: unexpected subcommand $1" >&2
exit 64
"""


def _run_gate(path: str, **extra: str) -> subprocess.CompletedProcess[str]:
    """Run `tools/terraform_drift.py` with `path` as its entire PATH."""
    return subprocess.run(
        [sys.executable, str(_DRIFT_TOOL)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": path, **extra},
    )


@pytest.fixture
def fake_terraform(tmp_path: Path) -> str:
    """A directory holding an executable fake `terraform`, for use as a PATH."""
    binary = tmp_path / "terraform"
    binary.write_text(_FAKE_TERRAFORM)
    binary.chmod(0o755)
    return str(tmp_path)


@pytest.mark.parametrize(("plan_exit", "expected"), [("0", 0), ("2", 2), ("1", 1)])
def test_the_gate_reports_terraforms_verdict_and_never_upgrades_it(
    fake_terraform: str, plan_exit: str, expected: int
) -> None:
    """0, 2 and 1 stay apart, because they mean three different things.

    1 is the one that matters and is why this test exists: it means no plan was
    produced, so NOTHING was compared. On 2026-08-22 the real gate returned
    exactly that -- `Error: Backend initialization required` against a state
    bucket that did not exist yet -- while `pytest` reported 11 passed and 1
    skipped. A tool that folded 1 into 0 would make that indistinguishable from
    agreement, and the suite would stay green for ever.
    """
    result = _run_gate(fake_terraform, MAP_FAKE_PLAN_EXIT=plan_exit)
    assert result.returncode == expected, result.stdout + result.stderr


def test_a_precondition_failure_is_its_own_code_and_not_a_verdict(
    fake_terraform: str,
) -> None:
    """3 when the comparison was never attempted: no binary, or unformatted HCL.

    Kept off 1 so that "the account could not be compared" can never be read as
    "a protected resource would have been destroyed", which is the other thing
    terraform's own exit 1 means.
    """
    absent = _run_gate(str(Path(fake_terraform) / "no-such-directory"))
    assert absent.returncode == 3, absent.stdout + absent.stderr
    assert "not on PATH" in absent.stderr

    unformatted = _run_gate(fake_terraform, MAP_FAKE_FMT_EXIT="3")
    assert unformatted.returncode == 3, unformatted.stdout + unformatted.stderr
    assert "not canonically formatted" in unformatted.stderr


def test_the_gate_asks_terraform_for_plain_output_it_cannot_hang_on() -> None:
    """`-no-color` on every terraform call, and `-input=false` on the plan.

    Without `-no-color` every `Plan:` line the gate prints carries ANSI escapes,
    so each consumer -- a human reading a log, a grep in a run record -- has to
    strip them. Without `-input=false` a future required variable turns the gate
    into a process waiting on a prompt nobody is at, which is worse than a
    failure because it never reports one.

    `-input=false` is asserted on `plan` only, and that is not laziness:
    `terraform fmt` rejects the flag outright -- measured, it prints its usage
    and does not format -- because it has nothing to prompt for.
    """
    source = _DRIFT_TOOL.read_text()
    calls = re.findall(r"_run\(\s*(\"[^)]*?)\)", source, re.DOTALL)
    assert calls, "no _run() calls found in the gate"
    for call in calls:
        flat = " ".join(call.split())
        assert '"-no-color"' in call, f"no -no-color in _run({flat})"
        if '"plan"' in call:
            assert '"-input=false"' in call, f"no -input=false in _run({flat})"


# --- Tier B: the account itself. ----------------------------------------------

_NEEDS_THE_ACCOUNT = pytest.mark.skipif(
    os.environ.get("MAP_TERRAFORM_DRIFT") != "1",
    reason=(
        "MAP_TERRAFORM_DRIFT is not 1, so the ACCOUNT WAS NOT COMPARED. This "
        "needs terraform on PATH, AWS credentials, and an initialised backend."
    ),
)


@pytest.mark.network
def test_the_configuration_is_one_terraform_accepts(tmp_path: Path) -> None:
    """`terraform validate`, the cheapest proof the configuration is usable.

    Marked `network` because a cold checkout downloads the provider to do it, and
    a bare `pytest` in this repo is offline by contract. `-backend=false` is what
    keeps it off AWS: no credentials, no state, no bucket -- so this fails on a
    typo'd argument or a dangling reference and on nothing else.

    It is here because the suite was once green while the configuration could not
    be planned at all. Tier A grades the `.tf` files as text, and text cannot tell
    you that `instance_typo` is not an argument.

    Run against a COPY, and that is the whole reason `deploy/iam/` is copied
    alongside: `terraform init` rewrites `.terraform/` in whatever directory it
    runs in, and `deploy/terraform/.terraform/` is where the S3 backend the drift
    gate uses records that it is initialised. A test that re-inits the real
    directory would leave the next gate run reporting `Backend initialization
    required` -- exit 1, nothing compared -- which is the exact failure this file
    exists to catch, caused by the test looking for it. The policies are copied
    because the configuration reaches them as `${path.module}/../iam/*.json`.
    """
    if shutil.which("terraform") is None:
        pytest.fail("terraform is not on PATH, so the configuration was NOT checked")
    config = tmp_path / "deploy" / "terraform"
    shutil.copytree(_TERRAFORM, config, ignore=shutil.ignore_patterns(".terraform"))
    shutil.copytree(_ROOT / "deploy" / "iam", tmp_path / "deploy" / "iam")
    env = {**os.environ, "TF_DATA_DIR": str(tmp_path / "tfdata")}
    init = subprocess.run(
        ["terraform", "init", "-backend=false", "-input=false", "-no-color"],
        cwd=config,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert init.returncode == 0, init.stdout + init.stderr
    validate = subprocess.run(
        ["terraform", "validate", "-no-color"],
        cwd=config,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr


@_NEEDS_THE_ACCOUNT
def test_the_declared_account_and_the_live_account_agree() -> None:
    """The gate, run through the same entry point a human runs.

    Calling the tool rather than terraform directly is deliberate: a second
    invocation here would be a second place that knows where the config lives and
    which exit codes mean what, free to drift from the one the gate uses.
    """
    result = subprocess.run(
        [sys.executable, str(_DRIFT_TOOL)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@_NEEDS_THE_ACCOUNT
def test_every_role_in_the_account_is_declared_in_this_configuration() -> None:
    """The one class of drift `terraform plan` provably cannot see.

    A plan refreshes the resources already in state; it never lists the account's
    roles. So a sixth hand-made IAM role appears in no plan at any exit code --
    which is how `map-control-plane`, `map-model-gateway` and `map-tool-gateway`
    came to be trusted by this cluster's OIDC provider, bound by
    `deploy/k8s/cluster-bootstrap.yaml`, and declared nowhere. Enumerating the
    account is the only thing that closes it, so this enumerates.

    Service-linked roles are excluded by path: AWS creates them on demand under
    `/aws-service-role/` and they cannot be managed from here.

    TWO ASSERTIONS RATHER THAN `live == declared`, and the split is not a softening --
    it is the difference between the drift this can see and the drift a plan already
    sees, asserted against the right set each way.

    `live - declared` is the whole reason this case exists and is unchanged: a role in
    the account that appears in no `.tf` file is invisible to every plan at every exit
    code, and nothing but an enumeration finds it.

    The other direction is compared against the IMPORTED names, not every declared
    name. `live == declared` was the same thing only while every declared role was an
    adopted one; `session_vfs.tf` declares three that are created, so equality would
    have been false before the apply and, read the other way round with role names
    taken off import ids, false for ever after it. What the reverse direction is
    actually for is "this file claims the account holds a role and it does not", and
    that claim is exactly what an import block makes. A role declared with no import
    block is a creation `terraform plan` reports as `1 to add` -- drift the gate
    already covers, at an exit code of its own.
    """
    listed = subprocess.run(
        ["aws", "iam", "list-roles", "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0, listed.stdout + listed.stderr
    live = {
        role["RoleName"]
        for role in json.loads(listed.stdout)["Roles"]
        if "/aws-service-role/" not in role["Path"]
    }
    declared = _declared_role_names()
    assert live - declared == set(), (
        f"in the account and declared in no .tf file: {sorted(live - declared)}; "
        f"a plan cannot see these at any exit code"
    )
    imported = _imported_role_names()
    assert imported - live == set(), (
        f"imported as though the account held them, and it does not: "
        f"{sorted(imported - live)}"
    )
