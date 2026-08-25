"""The Session VFS mount's Terraform, graded on the things a shared mount makes matter.

Three tiers, and the split is about what each can honestly prove.

Tier A (local, no network, no AWS) reads the configuration and asserts its shape. It
is the tier that runs in the default gate, and all it can see is what the file *says*.
It reads the file two ways on purpose -- as text, and as the IAM documents the text
renders to (`documents()` and `ROLES` below) -- because the two catch different things
and the second exists only because the first kept missing the surface next door.

Tier A' (opt-in, `MAP_TERRAFORM_RENDER=1`) renders a real `terraform plan` and grades
`ROLES` against the documents terraform itself computes. It is the guard on Tier A's
reader: without it, a reader that misparses an expression and a table written to match
what it read would agree with each other and with nothing else. It needs credentials,
network and the state bucket, so the default suite SKIPS it and its skip reason says so.

Tier B (opt-in, `MAP_PROVISION_SESSION_VFS=1`) mounts the real file system into a real
pod and writes a real file. It is the only tier that can tell a mount from a directory.

NOT PROVEN by anything here: Tier A does not run `terraform validate`, because validate
needs `terraform init` and init downloads a ~600 MB provider -- so a schema-level
mistake (a misspelled optional attribute, a wrong type) reaches `terraform plan`
uncaught by this file. `test_terraform_declares_the_account.py`'s
`test_the_configuration_is_one_terraform_accepts` is the case that runs validate over
the whole directory, and it is marked `network` for that reason.

Tier A also cannot see anything the configuration computes. Two consequences worth
naming, because they decide how the cases below are written:

  - `for_each` is read as text, so one-mount-target-per-subnet is proved as its two
    halves (the resource iterates the subnet list, and the subnet list is the
    nodegroup's) rather than as the conclusion.
  - `lifecycle.precondition` is read as text too. Tier A can assert the guard is a
    precondition and not a `check` block -- which is what makes a same-AZ pair exit 1
    rather than warn at exit 2 -- but only a plan against the real VPC can watch it
    fire, and that measurement lives in `docs/progress.md`, not here.

WHY SO MANY CASES PIN A `<resource>.<attribute>` REFERENCE rather than the value it
resolves to: every id this configuration needs is already an attribute of a resource
in the same root module, and a literal copy of one is a second source of truth that no
plan compares -- `terraform plan` diffs the configuration against the account, never
one literal against another. The first draft of `session_vfs.tf` copied six, and each
copy made a live failure reachable by a one-line edit that every test then passed:
repointing the bucket mounted Terraform's own state into every Session read-write,
and moving a subnet or the cluster's OIDC issuer in the file that owns it left the
mount targets and the CSI trust policies naming something that no longer existed.

AND WHY THAT IS ASSERTED PER SITE rather than per id: an id read at one place and an
id read at five are not the same problem, and the guard that closed the first left the
second open. "Something in this file reads `aws_s3_bucket.platform.arn`" was true with
four of its five sites repointed at the Terraform state bucket, because the fifth kept
saying it. Reading an owner is a property of a SITE; a file-wide check cannot express
it, and a check that cannot express its claim reads as a guard while being an
accident. So every site is counted, and every site is pinned by value in some case
below -- which for the two inline policies means the whole grant surface, actions and
Resource together, in `GRANTS`.

AND WHY THERE IS A SECOND READER BESIDE ALL OF THAT: because per-site was still a
claim about ONE surface. Four rounds of this file each closed the grant surface in
front of it with a check on one spelling -- a copied literal, then five sites of one
id, then the managed-policy attachments, then the trust statements -- and each time
the surface beside it was open with the whole suite green. `AdministratorAccess`
attached to the CSI controller, `Principal = { AWS = "*" }` added to a role holding
DeleteObject on the platform bucket, and the `cloudwatch:namespace` condition that is
the entire scope of a `Resource = "*"` statement deleted: each was one line, and each
passed 1365 cases. So `documents()` resolves what this file actually hands IAM and
`ROLES` names the whole of it -- trust statement lists, managed-ARN sets and inline
documents with their conditions -- compared with `==` both ways, so an unlisted role,
policy, statement or key is a diff instead of an addition nobody sees. The text cases
above are kept: they are cheap, they catch a semantically-equivalent rewrite the
document reader is right not to care about, and their `Effect`/argument-order checks
are load-bearing where the document comparison is order-blind.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "deploy" / "terraform" / "session_vfs.tf"
_SESSION_POD = _REPO / "deploy" / "k8s" / "session-pod.yaml"

NFS_PORT = "2049"
ACCOUNT = "062677866851"

# The service account each CSI role may be assumed by, and nothing else. Both roles
# carry `AmazonS3FilesClientFullAccess` -- ClientMount + ClientWrite +
# ClientRootAccess on `Resource: "*"` -- so the subject in the trust policy IS the
# access-control decision: widened to a wildcard, any service account in the cluster
# (a Session pod's included) could mount and write every S3 Files file system in the
# account as root.
CSI_SERVICE_ACCOUNTS = {
    "s3files_csi_controller": "system:serviceaccount:kube-system:efs-csi-controller-sa",
    "s3files_csi_node": "system:serviceaccount:kube-system:efs-csi-node-sa",
}

# Each id this configuration needs, as the pattern a copied literal would match, the
# resource attribute to read instead, and HOW MANY places read it. The failure message
# names the replacement because "do not write this here" without "write this instead"
# sends the reader back to the account for a value they should never have needed.
#
# The count is the part that makes the presence half per-SITE. At one site, "the owner
# appears somewhere" and "this site reads the owner" are the same sentence, so any edit
# to that site removes the only occurrence. At more than one they stop being the same
# sentence: the sites satisfy the presence half for each other, and every one of them
# has to be pinned by value in a case of its own or it is editable with the whole suite
# green. That is not hypothetical -- `aws_s3_bucket.platform.arn` reached five sites
# with one pinned, and four one-line edits pointed a grant at the Terraform state
# bucket, two of them at every bucket in the account. Raising a number here without
# adding the case that pins the new site re-opens exactly that.
COPIED_ID_PATTERNS = (
    (r"subnet-[0-9a-f]{8}", "aws_eks_node_group.map_dev_nodes.subnet_ids", 1),
    (
        r"sg-[0-9a-f]{8}",
        "aws_eks_cluster.map_dev.vpc_config[0].cluster_security_group_id",
        1,
    ),
    (r"vpc-[0-9a-f]{8}", "aws_default_vpc.map.id", 1),
    (r"oidc-provider/", "aws_iam_openid_connect_provider.cluster.arn", 2),
    (r"arn:aws:s3:::", "aws_s3_bucket.platform.arn", 5),
    (r'=\s*"map-dev"', "aws_eks_cluster.map_dev.name", 1),
)

# The complete grant surface of this file: every inline policy, and every `Allow`
# statement in it as the actions it permits and the Resource it permits them on.
# Written out by value because a grant is the one thing here that a plan renders as a
# clean create whichever way it reads -- `Resource = "*"` and
# `Resource = aws_s3_bucket.platform.arn` are the same shape of diff and a different
# blast radius.
#
# Enumerated in full, and compared as a whole, because the failure this closes was a
# grant no case named. A case that checks only the statements it lists cannot see a
# sixth statement somebody adds beside them, and the two `Resource = "*"` statements
# below are why that matters: the wildcard is legitimate there -- neither
# `elasticfilesystem:DescribeMountTargets`, `ec2:DescribeAvailabilityZones` nor
# `cloudwatch:PutMetricData` has a resource-level form -- so it cannot be banned
# outright, and an `s3:GetObject` added to that action list is account-wide bucket read
# reached without touching a `Resource` line at all. Pinning the actions of a wildcard
# statement is what keeps the wildcard attached to the actions that earned it.
#
# Adding a grant means adding it here, deliberately, in a diff that says so.
GRANTS: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "session_vfs_service_bucket": (
        (
            (
                "s3:GetBucketLocation",
                "s3:GetBucketVersioning",
                "s3:ListBucket",
                "s3:ListBucketVersions",
            ),
            "aws_s3_bucket.platform.arn",
        ),
        (
            (
                "s3:AbortMultipartUpload",
                "s3:DeleteObject*",
                "s3:GetObject*",
                "s3:List*",
                "s3:PutObject*",
            ),
            '"${aws_s3_bucket.platform.arn}/*"',
        ),
        (
            (
                "events:DeleteRule",
                "events:DisableRule",
                "events:EnableRule",
                "events:PutRule",
                "events:PutTargets",
                "events:RemoveTargets",
            ),
            '"arn:aws:events:*:*:rule/DO-NOT-DELETE-S3-Files*"',
        ),
        (
            ("events:DescribeRule", "events:ListTargetsByRule"),
            '"arn:aws:events:*:*:rule/*"',
        ),
        (
            ("events:ListRuleNamesByTarget", "events:ListRules"),
            '"*"',
        ),
    ),
    "s3files_csi_node_bucket_read": (
        (("s3:GetBucketLocation", "s3:ListBucket"), "aws_s3_bucket.platform.arn"),
        (
            ("s3:GetObject", "s3:GetObjectVersion"),
            '"${aws_s3_bucket.platform.arn}/*"',
        ),
    ),
    "s3files_csi_node_mount_helper": (
        (
            ("elasticfilesystem:DescribeMountTargets", "ec2:DescribeAvailabilityZones"),
            '"*"',
        ),
        (("cloudwatch:PutMetricData",), '"*"'),
    ),
}

_HEADER = re.compile(r'^\s*([a-z_]+)((?:\s+"[^"]*")*)\s*\{\s*$')
_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_LABEL = re.compile(r'"([^"]*)"')
# One `Allow` statement in a flattened policy body. The three alternatives for a value
# are a list, a quoted string (an interpolation included -- `"${x}/*"` has no inner
# quote) and a bare resource reference, which is the form this file's grants take and
# the form no quoted literal can be mistaken for.
_ALLOW = re.compile(
    r'Effect\s*=\s*"Allow"\s+'
    r'Action\s*=\s*(\[[^]]*\]|"[^"]*")\s+'
    r'Resource\s*=\s*(\[[^]]*\]|"[^"]*"|[A-Za-z_][A-Za-z0-9_.\[\]]*)'
)


class Block(NamedTuple):
    """One HCL block: its keyword, its quoted labels, and its verbatim body lines."""

    kind: str
    labels: tuple[str, ...]
    body: list[str]


def _uncommented(text: str) -> list[str]:
    """Every line, with whole-line comments blanked rather than dropped.

    Blanked rather than dropped so that a line number in a failure message still
    corresponds to the file. `#` inside a string literal is not handled; this file has
    none, and the reader is not a general HCL parser.
    """
    return [
        "" if line.lstrip().startswith(("#", "//")) else line
        for line in text.splitlines()
    ]


def blocks(text: str) -> list[Block]:
    """Every block at any nesting depth, closed by brace balance.

    A block is a line whose whole content is a bare keyword, optional quoted labels
    and an opening brace: `resource "x" "y" {`, `lifecycle {`, `posix_user {`. A line
    like `tags = {` or `assume_role_policy = jsonencode({` is an argument rather than a
    block and gets no `Block` of its own -- but its braces are still counted, because
    the enclosing block does not end until they are closed.

    That counting is the whole point and it was the bug. An earlier version closed a
    block on the first line that was exactly `}`, so a `tags` map ended its own
    resource: every argument written after one was invisible, `assigns` returned None
    for it, and any absence assertion over such a body was vacuous rather than false.
    `depends_on` on the file system sits after its `tags` and is exactly that case.

    A brace inside a string is the one thing this still cannot see. Every interpolation
    in this file (`"${local.oidc_issuer}:aud"`) is balanced, so the count survives; an
    unbalanced brace in a literal would not, and there is no such thing here.
    """
    lines = _uncommented(text)
    found: list[Block] = []
    opened: list[tuple[str, tuple[str, ...], int, int]] = []
    depth = 0
    for index, line in enumerate(lines):
        header = _HEADER.match(line)
        if header:
            labels = tuple(_LABEL.findall(header.group(2)))
            opened.append((header.group(1), labels, index, depth))
        depth += line.count("{") - line.count("}")
        while opened and depth <= opened[-1][3]:
            kind, labels, start, _ = opened.pop()
            found.append(Block(kind, labels, lines[start + 1 : index]))
    if opened:
        kind, _, start, _ = opened[-1]
        raise AssertionError(f"unclosed block {kind} at line {start + 1}")
    return found


def one(text: str, kind: str, *labels: str) -> Block:
    """The single block with this keyword and these labels.

    Exactness is the point rather than convenience: "one ingress rule" is itself an
    assertion, and a helper that returned the first of several would hide a second one.
    """
    hits = [b for b in blocks(text) if (b.kind, b.labels) == (kind, labels)]
    address = ".".join((kind, *labels))
    assert len(hits) == 1, f"expected exactly 1 {address}, found {len(hits)}"
    return hits[0]


def assigns(block: Block) -> dict[str, str]:
    """Top-level `key = value` pairs in a body, values exactly as written.

    Nested bodies are skipped by counting brackets, and a line whose value is a bare
    `{` or `[` is an opener rather than a value.
    """
    depth = 0
    out: dict[str, str] = {}
    for line in block.body:
        match = _ASSIGN.match(line)
        if depth == 0 and match and not _HEADER.match(line):
            value = match.group(2).rstrip(",")
            if value not in ("{", "["):
                out[match.group(1)] = value
        depth += line.count("{") + line.count("[") - line.count("}") - line.count("]")
    return out


def flat(block: Block) -> str:
    """A block's body as one line, runs of whitespace collapsed to one space.

    `terraform fmt` aligns `=` within a group of adjacent arguments, so the spacing
    around any one of them depends on the longest key beside it -- adding an argument
    silently re-pads its neighbours. Collapsing whitespace lets an assertion about a
    value survive that, which a raw `in` against the file does not.
    """
    return " ".join("\n".join(block.body).split())


def grants(block: Block) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Every `Allow` statement in an inline policy, as (actions, resource).

    Read off the flattened body, so `terraform fmt`'s re-padding of an `=` column
    cannot change the answer. The resource is returned exactly as written -- a bare
    `aws_s3_bucket.platform.arn`, an interpolated `"${...}/*"` and a literal `"*"` are
    three different strings here, which is the point: they are three different blast
    radii and a policy that means one must not compare equal to a policy that means
    another.

    A statement this pattern cannot read is the failure mode worth naming, because it
    would be silent: the statement would simply be absent from the result and whatever
    it granted would go ungraded. So the number of statements read is compared against
    the number of `Effect`s in the body, of which every statement has exactly one. A
    statement written `Resource` before `Action`, or one whose `Effect` is `Deny`,
    is missing from the result and fails that comparison rather than passing quietly.
    This file writes all six in Effect-Action-Resource order and `terraform fmt` does
    not reorder arguments, so the pattern is exact here and loud where it is not.
    """
    body = flat(block)
    found = tuple(
        (tuple(_LABEL.findall(match.group(1))), match.group(2))
        for match in _ALLOW.finditer(body)
    )
    assert len(found) == body.count("Effect"), (
        f"{'.'.join(block.labels)}: read {len(found)} statements from a policy with "
        f"{body.count('Effect')} of them, so at least one grant is not being graded"
    )
    return found


# --- The documents these roles actually get, and the whole of what they say. ---
#
# WHY A SECOND READER when `grants()` above already reads the inline policies.
# Because four rounds of this file each closed one grant surface with a check on
# one spelling, and each time the surface next door was open with the whole suite
# green. In order: a copied literal id; the same id at five sites, where a
# file-wide presence check could not tell five right references from four right
# ones and a wrong one; the managed-policy attachments, where
# `AdministratorAccess` on the CSI controller was a one-token edit no case saw;
# and the trust policies, graded by presence, where a second statement reading
# `Principal = { AWS = "*" }` made a role holding PutObject and DeleteObject on
# the platform bucket assumable by any principal in AWS. The shape is the defect:
# a substring check grades the spelling it names and nothing else, so "four
# surfaces found by hand" is not evidence that the fifth does not exist.
#
# So this reads the DOCUMENTS. It resolves the `jsonencode` expressions and the
# attachment lists into the structures IAM is handed, and `ROLES` names the whole
# of what each role gets: its trust document, its managed-policy ARN set, and
# every inline policy's document with its `Condition`. The comparison is `==` on
# the whole structure in both directions, so an unlisted role, policy, statement
# or KEY is a diff rather than an invisible addition.
#
# It resolves rather than plans, deliberately: a plan needs credentials, network
# and the state bucket, and a guard the default suite cannot run is not a guard.
# Everything in these documents is decidable from the file -- the values are
# literals and the ids are references. A reference this cannot resolve renders as
# the text `${...}` and is compared as that, so repointing one at a different
# resource is a diff without this ever learning an ARN.
# `test_terraform_renders_the_same_documents_the_reader_reads` checks this reader
# against terraform's own rendering; it is opt-in and the default suite skips it.

Rendered = Any

_UNRESOLVED = object()

# An HCL traversal -- `each.value`, `local.oidc_issuer`,
# `aws_eks_cluster.map_dev.vpc_config[0].cluster_security_group_id` -- lexed as
# one token, so the parser never reassembles a dotted path from pieces.
_TRAVERSAL = r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*|\[[0-9]+\])*"
_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"'
    rf"|{_TRAVERSAL}"
    r"|-?[0-9]+(?:\.[0-9]+)?"
    r"|[{}\[\](),=:]"
    r"|\S"  # so a character the reader does not expect stops it
)
_INTERPOLATION = re.compile(rf"\$\{{\s*({_TRAVERSAL})\s*\}}")

# `jsonencode` returns the STRUCTURE, not its JSON text: the structure is what IAM
# is handed and what a comparison can read key by key, and for the types here
# `json.loads(jsonencode(x)) == x`. Any other call fails loudly.
_FUNCTIONS = frozenset({"jsonencode", "toset", "tolist"})

# The resource types this reader folds into a role's documents. Every `aws_iam_*`
# resource in the file, and every resource whose type names a policy, has to be
# one of these -- which is what makes a FIFTH grant surface a failure here rather
# than a discovery two rounds from now.
_GRADED_IAM_TYPES = frozenset(
    {
        "aws_iam_role",
        "aws_iam_role_policy",
        "aws_iam_role_policy_attachment",
        "aws_iam_role_policies_exclusive",
        "aws_iam_role_policy_attachments_exclusive",
    }
)

_ROLE_REFERENCE = re.compile(r"^aws_iam_role\.([a-z0-9_]+)\.(?:id|name|arn)$")


def _balance(text: str) -> int:
    """How many brackets this text opens and does not close."""
    return sum(text.count(c) for c in "({[") - sum(text.count(c) for c in ")}]")


def arguments(block: Block) -> dict[str, str]:
    """Every top-level `key = <expression>` in a body, each expression WHOLE.

    Unlike `assigns`, a value spanning lines comes back entire, which is the point:
    `assume_role_policy = jsonencode({` is where a trust document starts and
    `assigns` returns that first line. Nested BLOCKS (`lifecycle {`) are skipped by
    brace balance; a nested MAP (`tags = {`) is an argument and comes back as one.

    A line this cannot read raises. Silence is the failure mode that matters: an
    argument dropped from the result is a grant every comparison below then passes
    over.
    """
    out: dict[str, str] = {}
    lines = list(block.body)
    index = 0
    while index < len(lines):
        line = lines[index]
        if _HEADER.match(line):
            depth = 0
            while index < len(lines):
                depth += _balance(lines[index])
                index += 1
                if depth <= 0:
                    break
            continue
        match = _ASSIGN.match(line)
        if match:
            text = match.group(2)
            depth = _balance(text)
            while depth > 0 and index + 1 < len(lines):
                index += 1
                text += "\n" + lines[index]
                depth += _balance(lines[index])
            assert depth == 0, f"unclosed expression for {match.group(1)}: {text!r}"
            out[match.group(1)] = text
            index += 1
            continue
        assert not line.strip(), (
            f"the reader cannot read {line.strip()!r}; an argument it skips is a "
            f"grant nothing below grades, so it refuses instead of skipping"
        )
        index += 1
    return out


class Configuration:
    """The file's resources, and their arguments both raw and resolved.

    Resolution is the whole reason this exists. What a grant reaches is decided by
    what its `Resource` and `Principal` RESOLVE to, and every guard here before
    compared the text a human typed -- which is why a second statement, a fourth
    managed policy and a deleted `Condition` were all invisible.

    A reference to a resource in this file resolves to that resource's own
    argument. Anything else -- `local.oidc_issuer`, `aws_s3_bucket.platform.arn`,
    a resource in another file -- resolves to the text `${...}` and is compared as
    that text: it says WHICH thing the grant points at, which is what lets the
    comparison run with no credentials.
    """

    def __init__(self, text: str) -> None:
        self._resources: dict[tuple[str, str], Block] = {}
        for block in blocks(text):
            if block.kind == "resource" and len(block.labels) == 2:
                key = (block.labels[0], block.labels[1])
                assert key not in self._resources, f"{key} is declared twice"
                self._resources[key] = block

    def types(self) -> frozenset[str]:
        return frozenset(kind for kind, _ in self._resources)

    def named(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(n for declared, n in self._resources if declared == kind))

    def raw(self, kind: str, name: str) -> dict[str, str]:
        block = self._resources.get((kind, name))
        assert block is not None, f"{kind}.{name} is not declared in session_vfs.tf"
        return arguments(block)

    def instances(self, kind: str, name: str) -> tuple[dict[str, Rendered], ...]:
        """One attribute mapping per instance the resource declares.

        A `for_each` resource yields one instance per element with `each.value`
        bound, which is how the controller's three managed-policy attachments come
        back as three ARNs instead of as the string `each.value`.
        """
        raw = self.raw(kind, name)
        each = raw.pop("for_each", None)
        if each is None:
            return ({key: self.value(text) for key, text in raw.items()},)
        elements = self.value(each)
        assert isinstance(elements, list), f"{kind}.{name}: for_each is not a list"
        return tuple(
            {
                key: self.value(text, {"each": {"value": item, "key": item}})
                for key, text in raw.items()
            }
            for item in elements
        )

    def only(self, kind: str, name: str) -> dict[str, Rendered]:
        found = self.instances(kind, name)
        assert len(found) == 1, f"{kind}.{name} declares {len(found)} instances, not 1"
        return found[0]

    def value(self, text: str, env: dict[str, Rendered] | None = None) -> Rendered:
        return _Expression(self, _TOKEN.findall(text), env or {}).parse()

    def reference(self, path: str, env: dict[str, Rendered]) -> Rendered:
        """What a traversal resolves to, or `_UNRESOLVED` if this file cannot say."""
        parts = path.split(".")
        if parts[0] in env:
            found: Rendered = env[parts[0]]
            for part in parts[1:]:
                if not isinstance(found, dict) or part not in found:
                    return _UNRESOLVED
                found = found[part]
            return found
        if len(parts) == 3 and (parts[0], parts[1]) in self._resources:
            return self.only(parts[0], parts[1]).get(parts[2], _UNRESOLVED)
        return _UNRESOLVED


class _Expression:
    """One HCL expression, read as the value terraform would compute for it."""

    def __init__(
        self, config: Configuration, tokens: list[str], env: dict[str, Rendered]
    ) -> None:
        self._config = config
        self._tokens = tokens
        self._env = env
        self._at = 0

    def parse(self) -> Rendered:
        value = self._value()
        assert self._at == len(self._tokens), (
            f"the reader stopped at {' '.join(self._tokens[self._at :])!r}; an "
            f"expression read in part is a grant graded in part"
        )
        return value

    def _peek(self, ahead: int = 0) -> str:
        at = self._at + ahead
        return self._tokens[at] if at < len(self._tokens) else ""

    def _take(self) -> str:
        token = self._peek()
        assert token, "the expression ended early"
        self._at += 1
        return token

    def _expect(self, token: str) -> None:
        got = self._take()
        assert got == token, f"expected {token!r}, read {got!r}"

    def _value(self) -> Rendered:
        token = self._peek()
        assert token, "the expression ended early"
        if token == "{":
            return self._object()
        if token == "[":
            return self._list()
        if token.startswith('"'):
            return self._string(self._take())
        if token in ("true", "false", "null"):
            self._take()
            return {"true": True, "false": False, "null": None}[token]
        if re.fullmatch(r"-?[0-9]+", token):
            return int(self._take())
        if re.fullmatch(r"-?[0-9]+\.[0-9]+", token):
            return float(self._take())
        if self._peek(1) == "(":
            return self._call()
        return self._traversal(self._take())

    def _call(self) -> Rendered:
        name = self._take()
        assert name in _FUNCTIONS, (
            f"the reader does not evaluate {name}(); it refuses rather than read "
            f"past a call whose value it cannot compute"
        )
        self._expect("(")
        inner = self._value()
        self._expect(")")
        if name == "jsonencode":
            return inner
        assert isinstance(inner, list), f"{name}() was not given a collection"
        return list(dict.fromkeys(inner))

    def _object(self) -> dict[str, Rendered]:
        self._expect("{")
        out: dict[str, Rendered] = {}
        while self._peek() != "}":
            token = self._take()
            key = self._string(token) if token.startswith('"') else token
            assert self._peek() in ("=", ":"), (
                f"expected an assignment after the key {key!r}, read {self._peek()!r}"
            )
            self._take()
            assert key not in out, f"the key {key!r} is written twice in one object"
            out[key] = self._value()
            if self._peek() == ",":
                self._take()
        self._expect("}")
        return out

    def _list(self) -> list[Rendered]:
        self._expect("[")
        if self._peek() == "for":
            return self._comprehension()
        out: list[Rendered] = []
        while self._peek() != "]":
            out.append(self._value())
            if self._peek() == ",":
                self._take()
        self._expect("]")
        return out

    def _comprehension(self) -> list[Rendered]:
        """`[for x in <resource> : <expr>]`, evaluated once per instance.

        The one comprehension this file writes, and the one that matters:
        `aws_iam_role_policy_attachments_exclusive.s3files_csi_controller` builds
        its `policy_arns` this way. Read as TEXT it says `attachment.policy_arn`
        whatever the attachment attaches -- which is how the resource whose job is
        to make the policy set explicit widened along with an escalation and
        reported nothing. Expanded, it says the ARNs.
        """
        self._expect("for")
        variable = self._take()
        self._expect("in")
        source = self._take()
        self._expect(":")
        depth = 0
        body: list[str] = []
        while True:
            token = self._take()
            if token in "([{":
                depth += 1
            elif token in ")]}":
                if depth == 0:
                    assert token == "]", f"a comprehension closed with {token!r}"
                    break
                depth -= 1
            body.append(token)
        kind, _, name = source.partition(".")
        assert name, f"a comprehension iterates {source!r}, which names no resource"
        return [
            _Expression(
                self._config, list(body), {**self._env, variable: attributes}
            ).parse()
            for attributes in self._config.instances(kind, name)
        ]

    def _string(self, token: str) -> str:
        """A string literal, with every reference this file can resolve resolved.

        An unresolvable `${...}` is kept verbatim, so
        `"${aws_s3_bucket.platform.arn}/*"` compares as that text -- which is what
        distinguishes the three blast radii a plan renders identically: the
        bucket, an object under it, and `"*"`.
        """
        raw = cast(str, json.loads(token))

        def resolved(match: re.Match[str]) -> str:
            value = self._config.reference(match.group(1), self._env)
            return match.group(0) if value is _UNRESOLVED else str(value)

        return _INTERPOLATION.sub(resolved, raw)

    def _traversal(self, token: str) -> Rendered:
        value = self._config.reference(token, self._env)
        return f"${{{token}}}" if value is _UNRESOLVED else value


class RoleDocuments(NamedTuple):
    """Everything one IAM role is granted, as IAM will be handed it."""

    trust: Rendered
    managed: frozenset[str]
    inline: dict[str, Rendered]


def _role_reference(address: str, raw: dict[str, str], attribute: str) -> str:
    """Which role in this file a policy or attachment is written onto.

    Taken from the REFERENCE rather than from a resolved role name, because the
    reference is the thing that has to be a role declared here: a `role` naming a
    role in another file, or naming one by literal string, puts a grant on an
    identity this file does not grade, and a resolved name cannot tell the two
    apart.
    """
    written = raw.get(attribute)
    assert written is not None, f"{address} has no {attribute}"
    match = _ROLE_REFERENCE.match(written.strip())
    assert match, (
        f"{address} names {written!r} as its {attribute} rather than a reference "
        f"to a role declared in this file, so the grant it carries reaches an "
        f"identity nothing here grades"
    )
    return match.group(1)


def documents(text: str) -> dict[str, RoleDocuments]:
    """Each role in the file, with every grant that reaches it folded in.

    Three routes reach a role and all three are followed: the trust document that
    decides who may become it, the managed policies attached to it, and the inline
    policies written into it. A policy or attachment whose `role` is not a
    reference to a role declared here fails rather than being ignored -- an
    attachment nothing collects is an attachment nothing grades.
    """
    config = Configuration(text)
    trusts = {
        name: config.only("aws_iam_role", name)["assume_role_policy"]
        for name in config.named("aws_iam_role")
    }
    managed: dict[str, set[str]] = {name: set() for name in trusts}
    inline: dict[str, dict[str, Rendered]] = {name: {} for name in trusts}
    for name in config.named("aws_iam_role_policy_attachment"):
        address = f"aws_iam_role_policy_attachment.{name}"
        role = _role_reference(
            address, config.raw("aws_iam_role_policy_attachment", name), "role"
        )
        assert role in managed, f"{address} is written onto undeclared role {role!r}"
        for attachment in config.instances("aws_iam_role_policy_attachment", name):
            managed[role].add(cast(str, attachment["policy_arn"]))
    for name in config.named("aws_iam_role_policy"):
        address = f"aws_iam_role_policy.{name}"
        role = _role_reference(address, config.raw("aws_iam_role_policy", name), "role")
        assert role in inline, f"{address} is written onto undeclared role {role!r}"
        policy = config.only("aws_iam_role_policy", name)
        written = cast(str, policy["name"])
        assert written not in inline[role], f"{role} has two policies named {written!r}"
        inline[role][written] = policy["policy"]
    return {
        name: RoleDocuments(trust, frozenset(managed[name]), inline[name])
        for name, trust in trusts.items()
    }


def declared_sets(text: str) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """Each role's `*_exclusive` declarations, resolved to the sets they name.

    Resolved rather than read, because both of this file's `policy_arns` lists
    derive from the attachment resources' own `policy_arn`: as text they say
    nothing about which policies the role holds, so the resource whose job is to
    make the set explicit widens with any escalation. Resolved and compared
    against the expectation rather than against the attachments, the derivation
    stops laundering.
    """
    config = Configuration(text)
    arns: dict[str, frozenset[str]] = {}
    names: dict[str, frozenset[str]] = {}
    for kind, key, into in (
        ("aws_iam_role_policy_attachments_exclusive", "policy_arns", arns),
        ("aws_iam_role_policies_exclusive", "policy_names", names),
    ):
        for name in config.named(kind):
            address = f"{kind}.{name}"
            role = _role_reference(address, config.raw(kind, name), "role_name")
            assert role not in into, f"{role} has two {kind} declarations"
            into[role] = frozenset(cast("list[str]", config.only(kind, name)[key]))
    assert set(arns) == set(names), (
        f"roles declaring only one of the two exclusive sets: "
        f"{sorted(set(arns) ^ set(names))}"
    )
    return {role: (arns[role], names[role]) for role in arns}


def _web_identity(subject: str) -> Rendered:
    """The trust document both CSI roles carry, differing only in the subject.

    One shape because it IS one shape: an edit giving either role a second
    statement, a different audience or an unconditioned principal renders a
    document this does not describe, and the comparison is `==` on the whole.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Federated": "${aws_iam_openid_connect_provider.cluster.arn}"
                },
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "${local.oidc_issuer}:aud": "sts.amazonaws.com",
                        "${local.oidc_issuer}:sub": subject,
                    }
                },
            }
        ],
    }


_BUCKET = "${aws_s3_bucket.platform.arn}"

# The whole of what this file grants, per role, as IAM will be handed it.
#
# Every field here is a claim that was not being made. `trust` is a STATEMENT
# LIST rather than a substring because a second statement reading
# `Principal = { AWS = "*" }` on `session_vfs_service` -- which holds PutObject,
# DeleteObject and DeleteObjectVersion on the platform bucket -- passed every case
# in this file. `managed` is an exact SET because
# `arn:aws:iam::aws:policy/AdministratorAccess` inserted into the controller's
# attachment list passed every case in this file, and so did `AmazonS3FullAccess`;
# the one assertion covering that route named a single forbidden spelling. Each
# inline document is compared WHOLE, `Condition` included, because deleting the
# `cloudwatch:namespace` condition -- the entire scope of the mount helper's
# `Resource = "*"` statement -- passed every case in this file too.
#
# Compared with `==` both ways: a role, policy, statement or key not written here
# fails, and one written here that the file does not render fails as well, because
# an expectation nothing reaches stops being true quietly.
ROLES: dict[str, RoleDocuments] = {
    "session_vfs_service": RoleDocuments(
        trust={
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "elasticfilesystem.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": ACCOUNT},
                        "ArnLike": {
                            "aws:SourceArn": (
                                f"arn:aws:s3files:us-east-1:{ACCOUNT}:file-system/*"
                            )
                        },
                    },
                }
            ],
        },
        managed=frozenset(),
        inline={
            "vfs-bucket-synchronisation": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetBucketLocation",
                            "s3:GetBucketVersioning",
                            "s3:ListBucket",
                            "s3:ListBucketVersions",
                        ],
                        "Resource": _BUCKET,
                        "Condition": {"StringEquals": {"aws:ResourceAccount": ACCOUNT}},
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:AbortMultipartUpload",
                            "s3:DeleteObject*",
                            "s3:GetObject*",
                            "s3:List*",
                            "s3:PutObject*",
                        ],
                        "Resource": f"{_BUCKET}/*",
                        "Condition": {"StringEquals": {"aws:ResourceAccount": ACCOUNT}},
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "events:DeleteRule",
                            "events:DisableRule",
                            "events:EnableRule",
                            "events:PutRule",
                            "events:PutTargets",
                            "events:RemoveTargets",
                        ],
                        "Resource": "arn:aws:events:*:*:rule/DO-NOT-DELETE-S3-Files*",
                        "Condition": {
                            "StringEquals": {
                                "events:ManagedBy": "elasticfilesystem.amazonaws.com"
                            }
                        },
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "events:DescribeRule",
                            "events:ListTargetsByRule",
                        ],
                        "Resource": "arn:aws:events:*:*:rule/*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "events:ListRuleNamesByTarget",
                            "events:ListRules",
                        ],
                        "Resource": "*",
                    },
                ],
            }
        },
    ),
    "s3files_csi_controller": RoleDocuments(
        trust=_web_identity(CSI_SERVICE_ACCOUNTS["s3files_csi_controller"]),
        managed=frozenset(
            {
                "arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy",
                "arn:aws:iam::aws:policy/service-role/AmazonS3FilesCSIDriverPolicy",
                "arn:aws:iam::aws:policy/AmazonS3FilesClientFullAccess",
            }
        ),
        inline={},
    ),
    "s3files_csi_node": RoleDocuments(
        trust=_web_identity(CSI_SERVICE_ACCOUNTS["s3files_csi_node"]),
        managed=frozenset({"arn:aws:iam::aws:policy/AmazonS3FilesClientFullAccess"}),
        inline={
            "vfs-bucket-read-only": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                        "Resource": _BUCKET,
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                        "Resource": f"{_BUCKET}/*",
                    },
                ],
            },
            "efs-utils-mount-helper": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "elasticfilesystem:DescribeMountTargets",
                            "ec2:DescribeAvailabilityZones",
                        ],
                        "Resource": "*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": "cloudwatch:PutMetricData",
                        "Resource": "*",
                        "Condition": {
                            "StringEquals": {
                                "cloudwatch:namespace": [
                                    "efs-utils/S3Files",
                                    "efs-utils/EFS",
                                ]
                            }
                        },
                    },
                ],
            },
        },
    ),
}


def code(text: str) -> str:
    """The configuration with comment lines blanked.

    Every absence assertion below runs against this rather than the raw file. The
    comments explaining why `AmazonS3ReadOnlyAccess` was replaced necessarily name it,
    so a raw-text `not in` check would force a choice between explaining a decision and
    testing it -- and the first draft of this file failed for exactly that reason.
    """
    return "\n".join(_uncommented(text))


# --- Tier A: the configuration as text. ---------------------------------------


@pytest.fixture(scope="module")
def config() -> str:
    return _CONFIG.read_text()


@pytest.fixture(scope="module")
def source(config: str) -> str:
    return code(config)


def test_the_reader_sees_an_argument_written_after_a_map() -> None:
    """The reader's own regression, because its failure mode was silence.

    `last` below sits after a `tags`-shaped map, which is where an earlier `blocks`
    ended the enclosing resource: `assigns` reported the argument missing and any
    absence assertion over that body was vacuous rather than false. A helper whose bug
    makes other cases meaningless, instead of red, earns a case of its own.
    """
    fixture = """
resource "aws_thing" "x" {
  first = "a"
  tags = {
    Name = "n"
  }
  last = "b"
}
"""
    assert assigns(one(fixture, "resource", "aws_thing", "x")) == {
        "first": '"a"',
        "last": '"b"',
    }


def test_no_id_is_copied_in_that_a_resource_here_already_holds(source: str) -> None:
    """Every id read off its owner, at every site that reads it.

    Both halves, because either alone is satisfiable by an accident. The absence half
    fails on a copied literal; the presence half fails on a copy replaced with a
    different literal, or with a variable whose default is one -- which is what a
    reader reaching for "just parameterise it" writes next.

    The presence half counts the sites rather than asking whether the owner appears,
    and that is the whole difference between a guard and the appearance of one. Asked
    as a membership it is one claim for the file; asked as a count it is one claim per
    site, because an id read at N places drops to N-1 the moment any one of them is
    repointed. Measured while it was a membership: `aws_s3_bucket.platform.arn` was
    read at five places, one of them pinned by value, and each of the other four could
    be repointed at the Terraform state bucket -- or at `"*"` -- with all 1363 cases
    green, because the four covered for each other.

    A count is still not a value: two edits that move an owner from a right site to a
    wrong one leave it at N. That is what the by-value cases are for, and every site
    counted here has one -- the grant sites in `test_the_synchronisation_role_...` and
    `test_the_node_role_holds_no_account_wide_s3_read`, the mount in
    `test_the_file_system_synchronises_the_platform_bucket`, the subnet and security
    group locals, both CSI trust policies, the addon's cluster name.

    Falsified: restoring any of the six copies this replaced -- e.g. the bucket as
    `arn:aws:s3:::map-dev-tfstate-062677866851-us-east-1`, which mounts Terraform
    state into every Session read-write -- fails here.
    """
    for pattern, owner, sites in COPIED_ID_PATTERNS:
        found = re.search(pattern, source)
        assert not found, (
            f"session_vfs.tf writes {found.group(0)!r} as a literal; read it off "
            f"{owner} instead, which is the resource that owns it"
        )
        seen = source.count(owner)
        assert seen == sites, (
            f"session_vfs.tf reads {owner} at {seen} places, not {sites}. One fewer "
            f"and a site that must read it does not; one more and the sites cover for "
            f"each other, so raise the number here only together with the case that "
            f"pins the new site by value"
        )


def test_the_file_declares_one_mount_target_per_node_subnet(config: str) -> None:
    """Structural, not computed: the reader cannot expand `for_each`.

    So both halves are pinned -- the resource iterates the subnet list, and that list
    is the nodegroup's, asserted in the next case. A resource with a hardcoded
    `subnet_id` fails here even if the list still holds two.
    """
    mount = assigns(one(config, "resource", "aws_s3files_mount_target", "session_vfs"))
    assert mount.get("for_each") == "toset(local.node_subnet_ids)"
    assert mount.get("subnet_id") == "each.value"


def test_the_node_subnets_are_the_nodegroups_own(config: str) -> None:
    """The mount targets follow the nodegroup, rather than agreeing with it by hand.

    This case used to compare two hand-typed subnet lists -- one in `session_vfs.tf`,
    one at the top of this file -- and so could not make the claim in its own name:
    neither copy was tied to the nodegroup, and re-CIDRing a private subnet in
    `network.tf` moved the nodegroup while leaving both copies green and the mount
    targets in an AZ the nodes had left.

    The variable's default has to be null rather than the reference, because Terraform
    requires a variable default to be a constant. So the reference lives in the local
    and the override is what the null admits -- and the override is the only reason the
    variable exists at all: a test needs it to plan a same-AZ pair.
    """
    assert re.search(
        r"node_subnet_ids\s*=\s*coalesce\(\s*var\.node_subnet_ids\s*,\s*"
        r"tolist\(aws_eks_node_group\.map_dev_nodes\.subnet_ids\)\s*\)",
        flat(one(config, "locals")),
    ), "local.node_subnet_ids is not the nodegroup's own subnets with an override"

    subnets = assigns(one(config, "variable", "node_subnet_ids"))
    assert subnets.get("default") == "null", (
        "a non-null default is a second copy of the nodegroup's subnets"
    )


def test_a_repeated_subnet_is_refused_before_anything_is_described(config: str) -> None:
    """The variable's own `validation`, which is the cheaper half of the AZ rule.

    S3 Files allows one mount target per file system per AZ, so the same subnet twice
    is an apply-time failure. Deleting this block is not a no-op that the precondition
    then catches: the precondition compares distinct AZs to the subnet COUNT, and two
    copies of one subnet collapse to one AZ and one `for_each` element, so the counts
    disagree with a message about Availability Zones rather than about a duplicate.
    Measured: with the validation, exit 1 and `node_subnet_ids must not repeat a
    subnet.`
    """
    variable = one(config, "variable", "node_subnet_ids")
    assert "validation {" in "\n".join(variable.body), (
        "the duplicate-subnet validation is gone"
    )
    check = one(config, "validation")
    assert "distinct(var.node_subnet_ids)" in flat(check)
    assert assigns(check).get("error_message") == (
        '"node_subnet_ids must not repeat a subnet."'
    )


def test_the_same_az_guard_is_a_precondition_and_not_a_check_block(
    config: str, source: str
) -> None:
    """The shape is the guarantee, so the shape is what is asserted.

    A failed `check` block is a WARNING: `terraform plan -detailed-exitcode` still
    exits 2, the same code the drift gate reads as "changes present", so the gate would
    go green over a configuration that cannot apply. A failed `precondition` exits 1.
    Both measured against the real VPC, and the difference is the whole reason this is
    a `lifecycle` block rather than the more obvious `check`.

    Deleting the precondition leaves a valid empty `lifecycle {}` and a plan that still
    exits 2, which is why the emptiness is asserted rather than only the file parsing.
    """
    mount = one(config, "resource", "aws_s3files_mount_target", "session_vfs")
    assert "precondition {" in "\n".join(mount.body), (
        "the mount target carries no precondition, so a same-AZ pair plans cleanly "
        "and fails at apply"
    )
    guard = flat(one(config, "precondition"))
    assert "s.availability_zone" in guard
    assert "== length(local.node_subnet_ids)" in guard
    assert "Availability Zone" in guard
    assert 'check "' not in source, (
        "a check block's failure is a warning at exit 2, which the drift gate reads "
        "as 'changes present' rather than as a refusal"
    )


def test_nfs_ingress_comes_from_the_node_security_group_and_nowhere_else(
    config: str, source: str
) -> None:
    """One rule, one port, one source, one target, and no CIDR anywhere.

    The absence half carries as much as the presence half. A second rule admitting a
    CIDR would leave this rule intact and every positive assertion green, so the count
    and the absence of the three non-security-group sources are what close it.

    The TARGET is asserted too, and it is not redundant. `security_group_id` and
    `referenced_security_group_id` are the two ends of one rule and swapping them is a
    one-line edit that a plan renders as a clean create: inbound 2049 lands on the
    cluster security group -- widening what every node ENI accepts -- while the
    mount-target group ends up with no ingress at all, so nothing can reach the mount.

    The absent EGRESS rule is asserted for the same reason the CIDRs are: the file
    states in a comment that it has none deliberately, and a decision this file only
    explains is a decision nothing holds it to. An egress rule referencing a security
    group rather than a CIDR would satisfy every other assertion here.
    """
    rules = [
        b
        for b in blocks(config)
        if b.kind == "resource" and b.labels[0] == "aws_vpc_security_group_ingress_rule"
    ]
    assert len(rules) == 1, f"expected one ingress rule, found {len(rules)}"
    rule = assigns(rules[0])
    assert rule.get("ip_protocol") == '"tcp"'
    assert (rule.get("from_port"), rule.get("to_port")) == (NFS_PORT, NFS_PORT)
    assert rule.get("security_group_id") == "aws_security_group.session_vfs_mount.id", (
        "the rule is written into some other group than the mount targets'"
    )
    assert rule.get("referenced_security_group_id") == "local.node_security_group_id"
    for source_kind in ("cidr_ipv4", "cidr_ipv6", "prefix_list_id"):
        assert source_kind not in source
    assert "aws_vpc_security_group_egress_rule" not in source, (
        "this file declares an egress rule; security groups are stateful, so the "
        "mount's replies need none, and one only widens what a compromised "
        "mount-target ENI could originate"
    )

    shared = assigns(one(config, "locals"))
    assert shared.get("node_security_group_id") == (
        "aws_eks_cluster.map_dev.vpc_config[0].cluster_security_group_id"
    ), "the node security group is not the cluster's own"


def test_the_mount_targets_carry_only_the_mount_security_group(config: str) -> None:
    mount = assigns(one(config, "resource", "aws_s3files_mount_target", "session_vfs"))
    assert mount.get("security_groups") == "[aws_security_group.session_vfs_mount.id]"


def test_the_file_system_synchronises_the_platform_bucket(config: str) -> None:
    """The bucket, by reference, and the ordering that lets the create succeed.

    Two things, because both were unguarded and the second is what makes the first
    apply. `bucket` as a literal made the whole mount retargetable by one edit -- at
    `map-dev-tfstate-062677866851-us-east-1` every Session would have mounted
    Terraform's own state, and the service role grants PutObject, DeleteObject and
    DeleteObjectVersion on it. The existing state guard in
    `test_terraform_declares_the_account.py` only points the other way (state must not
    live in the bucket a Session reads), so nothing covered this direction.

    `depends_on` is the ordering: CreateFileSystem puts a bucket policy on the bucket
    and validates that the role can reach it as it does so, so the role's inline policy
    has to exist first. Terraform infers no edge from `role_arn` alone -- that names
    the role, not its policy -- so deleting this line is a create-order race with no
    diff to show for it.
    """
    system = assigns(one(config, "resource", "aws_s3files_file_system", "session_vfs"))
    assert system.get("bucket") == "aws_s3_bucket.platform.arn"
    assert system.get("role_arn") == "aws_iam_role.session_vfs_service.arn"
    assert system.get("depends_on") == (
        "[aws_iam_role_policy.session_vfs_service_bucket]"
    )


def test_every_service_trust_in_this_file_is_pinned_to_one_caller(
    config: str,
) -> None:
    """Every SERVICE-principal trust statement carries both deputy pins, whatever
    it names.

    This case used to read `Principal = { Service = "s3files.amazonaws.com" }` out of
    the text and assert that exact string. It passed for five rounds and it was worth
    nothing: the spelling it pinned is one IAM REFUSES -- `CreateRole` fails with
    MalformedPolicyDocument -- so the assertion was green over a role that could not be
    created at all. A test that grades a spelling grades the author's guess, and the
    authority on a service principal is IAM, which is not consulted by reading a file.

    So this asserts the PROPERTY instead, over every role in the file rather than the
    one that happened to be wrong, and says nothing about which service is named. A
    trust statement whose principal is an AWS service is unbounded by construction --
    the principal is not an account, so `Allow ... sts:AssumeRole` on its own reads "any
    caller of that service, in any account, may become this role", and this role holds
    the object-delete grant on the platform bucket. Two pins bound it, and both are
    required here: `aws:SourceAccount` for the account, `aws:SourceArn` for the calling
    resource. The second is not redundant -- S3 Files assumes as the EFS principal,
    which EFS proper shares, so the account pin alone still admits every other
    elasticfilesystem-backed resource in this account.

    Stated as a property it also keeps holding for a role nobody has written yet. The
    whole-document comparison against `ROLES` sees any change to a trust policy, but it
    sees it as "the table disagrees", and the table is what an author updates to make
    the disagreement stop. This says the shape is refused however the table reads.

    The CSI roles are federated (`Federated` + `sts:AssumeRoleWithWebIdentity`), not
    service trusts, so they are outside this claim and graded by
    `test_each_csi_role_is_assumable_by_exactly_one_service_account` -- the filter below
    is what separates them, and it is asserted to have kept something so that a rename
    of the principal key cannot empty this case into a vacuous pass.
    """
    checked = []
    for name, role in sorted(documents(config).items()):
        for statement in role.trust["Statement"]:
            principal = statement.get("Principal", {})
            if "Service" not in principal:
                continue
            checked.append((name, principal["Service"]))
            condition = statement.get("Condition", {})
            keys = {key for operator in condition.values() for key in operator}
            assert "aws:SourceAccount" in keys, (
                f"{name} trusts service {principal['Service']!r} with no "
                f"aws:SourceAccount pin, so any account's use of that service may "
                f"assume it; conditions present: {sorted(keys)}"
            )
            assert "aws:SourceArn" in keys, (
                f"{name} trusts service {principal['Service']!r} with no aws:SourceArn "
                f"pin, so any resource in this account that calls as that service may "
                f"assume it -- and this principal is shared with EFS proper; "
                f"conditions present: {sorted(keys)}"
            )
            account = condition.get("StringEquals", {}).get("aws:SourceAccount")
            assert account == ACCOUNT, (
                f"{name} pins aws:SourceAccount to {account!r}, not this account"
            )
    assert checked, (
        "no service-principal trust statement was found in this file, so this case "
        "asserted nothing; if the sync role's trust moved or its Principal key was "
        "renamed, this filter has to move with it"
    )


def test_the_synchronisation_role_writes_only_the_platform_bucket(config: str) -> None:
    """The role that holds PutObject and DeleteObject, scoped by value.

    This is the most dangerous grant in the file: `PutObject`, `DeleteObject` and
    `DeleteObjectVersion`, held by a role an AWS service assumes. `Resource` is the
    only thing deciding which bucket they land on, and a plan cannot tell one bucket
    from another -- both render as a create.

    Three edits this closes, each one line, each previously green across the whole
    suite. The object statement's `Resource` at `"${aws_s3_bucket.tfstate.arn}/*"`
    gives this role delete on Terraform's own state, and at `"*"` delete on every
    object in the account; the bucket statement's at `aws_s3_bucket.tfstate.arn` gives
    it the listing that finds them. The presence half of the copied-id case could see
    none of the three, because the bucket is read at four other places in this file and
    they satisfied it on these ones' behalf.
    """
    policy = one(
        config, "resource", "aws_iam_role_policy", "session_vfs_service_bucket"
    )
    assert grants(policy) == GRANTS["session_vfs_service_bucket"]


def test_every_inline_policy_in_this_file_has_its_grants_pinned(config: str) -> None:
    """The two cases above are exhaustive only if nothing else grants anything.

    Each of them names its own policy, which is what lets each make a claim in its own
    name -- and it is also what would leave a fourth inline policy graded by neither.
    This is the case that closes that: `GRANTS` has to name every `aws_iam_role_policy`
    in the file, so a new one fails here until somebody writes down what it grants.

    A default that admits the unlisted is the shape of the defect being closed, so the
    comparison runs both ways. A stale entry for a policy that no longer exists fails
    too, because an entry nothing reads is an entry that stops being true quietly.

    The name is taken as the LAST label rather than the second so that a header with a
    label missing lands in the compare as its own type name and fails readably, rather
    than raising an IndexError out of a set comprehension.
    """
    written = {
        block.labels[-1]
        for block in blocks(config)
        if (block.kind, block.labels[:1]) == ("resource", ("aws_iam_role_policy",))
    }
    assert written == set(GRANTS), (
        f"inline policies with no pinned grants: {sorted(written - set(GRANTS))}; "
        f"pinned grants for no such policy: {sorted(set(GRANTS) - written)}"
    )


# One role, one for_each'd attachment, and both exclusive declarations, in the
# shapes `session_vfs.tf` writes them. The reader's own regression fixture: its
# failure mode is silence, so it needs a case that reads a document whose answer
# is written out here rather than derived from the same file being graded.
_FIXTURE = """
resource "aws_iam_role" "example" {
  name = "map-dev-example"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:kube-system:example"
        }
      }
    }]
  })

  tags = {
    Name = "map-dev-example"
  }
}

resource "aws_iam_role_policy_attachment" "example" {
  for_each = toset([
    "arn:aws:iam::aws:policy/service-role/First",
    "arn:aws:iam::aws:policy/Second",
  ])

  role       = aws_iam_role.example.name
  policy_arn = each.value
}

resource "aws_iam_role_policy_attachments_exclusive" "example" {
  role_name = aws_iam_role.example.name
  policy_arns = [
    for attachment in aws_iam_role_policy_attachment.example :
    attachment.policy_arn
  ]
}

resource "aws_iam_role_policies_exclusive" "example" {
  role_name    = aws_iam_role.example.name
  policy_names = []
}
"""


def test_the_renderer_reads_a_document_as_terraform_would_encode_it() -> None:
    """The reader graded against an answer written out by hand.

    Every case below compares the reader's output to `ROLES`, so a reader that
    misread a document consistently would agree with an expectation somebody wrote
    to match it. This is the one case whose answer does not come from the file
    being graded: the trust document nested three deep with an interpolated KEY,
    the `tags` map that must not end the resource early, and a `for_each` over
    `toset` that has to come back as two ARNs rather than as `each.value`.
    """
    assert documents(_FIXTURE) == {
        "example": RoleDocuments(
            trust={
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Federated": (
                                "${aws_iam_openid_connect_provider.cluster.arn}"
                            )
                        },
                        "Action": "sts:AssumeRoleWithWebIdentity",
                        "Condition": {
                            "StringEquals": {
                                "${local.oidc_issuer}:sub": (
                                    "system:serviceaccount:kube-system:example"
                                )
                            }
                        },
                    }
                ],
            },
            managed=frozenset(
                {
                    "arn:aws:iam::aws:policy/service-role/First",
                    "arn:aws:iam::aws:policy/Second",
                }
            ),
            inline={},
        )
    }


def test_the_renderer_expands_a_comprehension_into_the_arns_it_names() -> None:
    """`policy_arns` resolved to ARNs, which is what stops it laundering.

    Written as `[for attachment in ... : attachment.policy_arn]`, the declaration
    says the same words whatever the attachment attaches -- so the resource whose
    whole job is to make a role's policy set explicit widened along with an
    escalation and no case noticed. Expanded, it names the ARNs and a fourth one
    is a diff.
    """
    assert declared_sets(_FIXTURE) == {
        "example": (
            frozenset(
                {
                    "arn:aws:iam::aws:policy/service-role/First",
                    "arn:aws:iam::aws:policy/Second",
                }
            ),
            frozenset(),
        )
    }


def test_the_renderer_refuses_what_it_cannot_read_rather_than_dropping_it() -> None:
    """Refusal, because the alternative is a grant that goes ungraded in silence.

    A reader that skipped what it did not understand would leave every comparison
    below passing over the part it dropped -- which is the same defect as a
    substring check, arrived at from the other side. So an unknown function, a
    tail it did not consume, an expression that ends early and a line in a
    resource body it cannot parse are each an error naming the text.
    """
    config = Configuration(_FIXTURE)
    for expression, complaint in (
        ('base64encode("x")', "does not evaluate"),
        ('"one" "two"', "stopped at"),
        ("{ Effect = }", "ended early"),
    ):
        with pytest.raises(AssertionError, match=complaint):
            config.value(expression)
    with pytest.raises(AssertionError, match="cannot read"):
        arguments(Block("resource", ("aws_iam_role", "x"), ["  policy_arn |= 1"]))


def test_the_roles_this_file_declares_are_the_roles_graded_here(config: str) -> None:
    """Both directions, because either alone admits an ungraded role.

    A fourth role declared here with no entry in `ROLES` holds whatever its own
    documents say and nothing compares them; an entry for a role the file no
    longer declares is an expectation nothing reaches, which stops being true
    without failing.
    """
    rendered = documents(config)
    assert set(rendered) == set(ROLES), (
        f"roles with no documents pinned: {sorted(set(rendered) - set(ROLES))}; "
        f"pinned documents for no such role: {sorted(set(ROLES) - set(rendered))}"
    )


def test_each_roles_trust_policy_is_the_whole_statement_list_named_here(
    config: str,
) -> None:
    """WHO may become each role, as a statement LIST rather than a substring.

    The unit of this claim is the list. Every earlier version asked whether the
    right statement was present, and it always was -- so a SECOND statement was
    invisible: `Principal = { AWS = "*" }` with `sts:AssumeRole` added beside the
    service statement on `session_vfs_service` passed all 1365 cases, and that
    role holds PutObject, DeleteObject and DeleteObjectVersion on the platform
    bucket, so an unconditioned wildcard principal makes those reachable by any
    principal in AWS. Account-root trust added to `s3files_csi_node` passed too,
    and that role carries ClientMount + ClientWrite + ClientRootAccess on
    `Resource: "*"`, so every identity in the account -- a Session pod's included
    -- could mount and write every S3 Files file system as root.

    Compared whole and both ways, so a second statement, a widened principal, a
    dropped `Condition` and a changed audience are each a diff.
    """
    rendered = documents(config)
    assert {name: role.trust for name, role in rendered.items()} == {
        name: role.trust for name, role in ROLES.items()
    }, (
        "a trust document is not what ROLES says it is; the statement list decides "
        "who may become the role, so an extra statement here is an extra principal"
    )


def test_each_role_holds_exactly_the_managed_policies_named_here(config: str) -> None:
    """The managed-attachment route, as an exact SET.

    The route no case graded. `arn:aws:iam::aws:policy/AdministratorAccess` put
    into the CSI controller's attachment list passed all 1365 cases, as did
    `AmazonS3FullAccess` on the node role and a fourth entry inserted into the
    controller's `for_each`; the only assertion covering the route named one
    forbidden spelling, `AmazonS3ReadOnlyAccess`, which is a claim about that
    spelling and no other.

    A set rather than a list because `for_each` iteration order is the set's, not
    the file's, and a set is the honest unit: what matters is which policies the
    role holds, not the order terraform happens to attach them in.
    """
    rendered = documents(config)
    assert {name: role.managed for name, role in rendered.items()} == {
        name: role.managed for name, role in ROLES.items()
    }, (
        "a role is attached managed policies ROLES does not name; an AWS-managed "
        "policy is whatever AWS says it is, so the ARN set is the whole grant"
    )


def test_each_roles_inline_policies_are_the_whole_documents_named_here(
    config: str,
) -> None:
    """Every inline document whole -- statements, keys and `Condition`.

    `grants()` above reads the same policies as (actions, Resource) pairs and
    those pins stand; this adds what that shape cannot express. Deleting
    `Condition = { StringEquals = { "cloudwatch:namespace" = [...] } }` from the
    mount helper passed all 1365 cases, and that condition is the ENTIRE scope of
    a `Resource = "*"` statement: without it `cloudwatch:PutMetricData` widens
    from two namespaces to every namespace in the account. A policy whose safety
    rests on a condition is not graded by anything that reads only its actions and
    its resource.

    Keyed by the IAM policy NAME rather than the terraform local name, because the
    name is what `aws_iam_role_policies_exclusive` declares and what IAM stores.
    """
    rendered = documents(config)
    assert {name: role.inline for name, role in rendered.items()} == {
        name: role.inline for name, role in ROLES.items()
    }, (
        "an inline policy is not the document ROLES says it is; a statement, a key "
        "or a Condition that is not written down there is not graded anywhere"
    )


def test_each_roles_declared_policy_sets_are_the_sets_it_actually_holds(
    config: str,
) -> None:
    """The `*_exclusive` resources compared to the expectation, not to themselves.

    These two resources exist because terraform refreshes only the attachments
    already in state, so a policy attached by hand appears in no plan at any exit
    code. But both of this file's `policy_arns` lists are built from the attachment
    resources' own `policy_arn`, so the declaration widens automatically with any
    escalation -- the resource meant to make the set explicit laundered
    `AdministratorAccess` into its own `policy_arns` and reported nothing.

    Resolved and compared against `ROLES` rather than against the attachments,
    that stops: the declared set and the held set are now two claims about the
    same table, and a role with no exclusive declaration at all fails here too.
    """
    declared = declared_sets(config)
    assert declared == {
        name: (role.managed, frozenset(role.inline)) for name, role in ROLES.items()
    }, (
        "a role's declared policy set is not the set ROLES says it holds; an "
        "undeclared set leaves a hand-attached policy invisible to every plan"
    )


def test_no_grant_reaches_a_role_through_a_resource_type_nothing_here_grades(
    config: str,
) -> None:
    """The fifth surface, refused in advance rather than found in a fifth round.

    Four rounds found four grant surfaces one at a time, and each guard was
    written for the surface in front of it. The reader above follows three routes
    into a role -- trust, managed attachment, inline policy -- and a fourth
    resource type would reach a role past all three: `aws_iam_policy` plus an
    attachment by ARN, an `aws_iam_user_policy`, an `aws_s3_bucket_policy` naming
    a principal. Any of those is a grant, and none of them is read by anything
    above.

    So the type list is closed. A resource here whose type is an IAM type, or
    whose type names a policy at all, has to be one the reader folds into a role's
    documents -- and adding a new kind of grant means teaching the reader first.
    """
    ungraded = sorted(
        kind
        for kind in Configuration(config).types()
        if (kind.startswith("aws_iam_") or "policy" in kind)
        and kind not in _GRADED_IAM_TYPES
    )
    assert not ungraded, (
        f"session_vfs.tf declares {ungraded}, which the document reader does not "
        f"fold into any role -- so whatever they grant is graded by nothing. Teach "
        f"`documents()` to read them and add them to _GRADED_IAM_TYPES"
    )


def test_no_other_file_in_this_directory_writes_a_grant_onto_these_roles(
    config: str,
) -> None:
    """The document reader reads ONE file, so the other nine must not reach these roles.

    `documents()` folds every grant in `session_vfs.tf` into the role it lands on, and
    that file is the whole of its scope. An `aws_iam_role_policy_attachment` written
    into `iam.tf` naming `aws_iam_role.s3files_csi_controller` would be a fourth managed
    policy the reader never opens the file to see -- the same defect the managed-policy
    case above closes, reached from the one direction that case cannot look. Terraform's
    root module is one namespace, so nothing about the file boundary stops it.

    The guard is that nobody else names them, which is honest here because these three
    roles are created by this file, referenced by the add-on in this file, and used by
    nothing else in the directory. A future file that genuinely needs one of them fails
    here, and the fix is to widen the reader rather than to widen this list -- an
    attachment nothing folds into a role is a grant nothing grades.
    """
    declared = sorted(Configuration(config).named("aws_iam_role"))
    assert declared == sorted(ROLES), (
        f"the roles this guard covers are {sorted(ROLES)}, but the file declares "
        f"{declared}; a role missing from the sweep is a role other files may grant to"
    )
    for path in sorted(_CONFIG.parent.glob("*.tf")):
        if path == _CONFIG:
            continue
        elsewhere = code(path.read_text())
        for role in declared:
            assert f"aws_iam_role.{role}" not in elsewhere, (
                f"{path.name} references aws_iam_role.{role}, which session_vfs.tf "
                f"declares -- a grant written there reaches a role only this file's "
                f"document reader grades, so it would be graded by nothing"
            )


def test_each_csi_role_is_assumable_by_exactly_one_service_account(
    config: str, source: str
) -> None:
    """The subject, the audience, and the issuer -- all three, per role.

    The subject is the access-control decision: both roles carry
    `AmazonS3FilesClientFullAccess`, so `system:serviceaccount:*:*` would let anything
    in the cluster mount and write every S3 Files file system in the account as root.
    The audience keeps a token minted for some other relying party from working here.

    The issuer is `local.oidc_issuer`, which `irsa.tf` derives from the provider
    resource, and that reference is the point: written out as a literal, rolling the
    cluster's issuer id in `iam.tf` moved the three IRSA roles and left these two
    trusting an issuer that no longer exists -- `AssumeRoleWithWebIdentity` fails, the
    driver cannot mount, and no plan shows a thing.
    """
    for name, subject in sorted(CSI_SERVICE_ACCOUNTS.items()):
        role = flat(one(config, "resource", "aws_iam_role", name))
        assert (
            "Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn }"
        ) in role, f"{name} does not federate the cluster's own OIDC provider"
        assert f'"${{local.oidc_issuer}}:sub" = "{subject}"' in role
        assert '"${local.oidc_issuer}:aud" = "sts.amazonaws.com"' in role
    assert "system:serviceaccount:*" not in source
    assert ':sub" = "system:serviceaccount:kube-system:*' not in source


def test_no_data_is_imported_onto_the_fast_tier(config: str) -> None:
    """`size_less_than = 0` is the whole cost model, so it is asserted by value.

    Zero means metadata is imported and object data is not, which is what makes a
    recursive walk cheap and every read an S3 round trip. The default is 131072, and a
    silent drift to it changes the bill and the latency without changing behaviour.
    """
    rule = assigns(one(config, "import_data_rule"))
    assert rule.get("size_less_than") == "0"
    assert rule.get("trigger") == '"ON_DIRECTORY_FIRST_ACCESS"'


def test_the_fast_tier_ages_out_what_a_session_leaves_behind(config: str) -> None:
    """The other half of the cost model, and deleting it is not a no-op.

    With no expiration rule nothing ever leaves the fast tier, so the whole bucket
    accumulates there and is billed there -- silently, because behaviour does not
    change. One day is the service floor and a Session is hours; the bucket is the
    durable copy either way.
    """
    rule = assigns(one(config, "expiration_data_rule"))
    assert rule.get("days_after_last_access") == "1"


def test_the_access_point_forces_the_uid_the_session_pod_runs_as(config: str) -> None:
    user = assigns(one(config, "posix_user"))
    assert (user.get("uid"), user.get("gid")) == (
        "local.session_uid",
        "local.session_gid",
    )
    shared = assigns(one(config, "locals"))
    assert (shared.get("session_uid"), shared.get("session_gid")) == ("10001", "10001")


def test_the_access_point_serves_the_whole_bucket(config: str) -> None:
    """`root_directory` at `/`, because the pod's volumeMount supplies the subPath.

    `deploy/k8s/session-pod.yaml` mounts `subPath: artifacts/<id>`, which is resolved
    relative to the access point's root. Moving the root to a prefix would leave every
    mount succeeding against the wrong tree -- an agent writing to
    `<prefix>/artifacts/...` while everything reading the bucket looks under
    `artifacts/...`, with no error at any layer.
    """
    root = assigns(one(config, "root_directory"))
    assert root.get("path") == '"/"'


def test_the_mount_uid_matches_the_pods_runasuser_and_fsgroup(config: str) -> None:
    """Cross-read, because the two files are what must agree.

    The readiness check that runs before the Agent Runtime starts refuses a tree whose
    remote subtrees report a different owner than pod-local scratch. Scratch is an
    emptyDir owned by the pod's fsGroup, so the access point's forced identity and the
    pod's security context are one fact stored in two files -- and this is the only
    thing that notices when one of them moves.
    """
    pod = cast(dict[str, Any], yaml.safe_load(_SESSION_POD.read_text()))
    security = cast(dict[str, Any], pod["spec"]["securityContext"])
    shared = assigns(one(config, "locals"))
    assert shared.get("session_uid") == str(security["runAsUser"])
    assert shared.get("session_gid") == str(security["fsGroup"])


def test_the_node_role_holds_no_account_wide_s3_read(config: str, source: str) -> None:
    """`AmazonS3ReadOnlyAccess` is `s3:Get*` on `Resource: "*"`.

    That is read access to every bucket in the account, including the one holding
    Terraform state -- which ADR-021 put in its own bucket precisely so a Session could
    not reach it. The add-on catalogue recommends this policy for the node service
    account, so re-adding it is a plausible edit rather than a wild one, which is why
    there is a test rather than a comment.

    The managed policy is one of three ways in, and for a while this case saw only
    that one -- it asserted the managed policy absent and the inline replacement
    present, which is not the same claim as the one in its name. Both assertions stayed
    true while the inline replacement itself granted `s3:GetObject` on `"*"`, and while
    `"s3:GetObject"` was added to the mount helper's wildcard statement instead. So all
    three routes are graded here now: the managed attachment by name, and both inline
    policies by their whole grant surface, which is what makes the wildcard in the
    mount helper safe to have -- it is pinned to the two describes and the metric
    write, and an s3 action added beside them fails.
    """
    assert "AmazonS3ReadOnlyAccess" not in source
    assert "s3files_csi_node_bucket_read" in source
    for name in ("s3files_csi_node_bucket_read", "s3files_csi_node_mount_helper"):
        policy = one(config, "resource", "aws_iam_role_policy", name)
        assert grants(policy) == GRANTS[name], (
            f"{name} does not grant exactly what GRANTS says it does; a node-role "
            f"grant is read access reachable from every Session pod on the cluster"
        )


def test_the_node_role_holds_no_session_manager_grant(source: str) -> None:
    """`AmazonElasticFileSystemsUtils` carries `ssmmessages:*` and `ec2messages:*`.

    Those are the Session Manager data plane -- remote shell capability -- on
    `Resource: "*"`, in a policy attached for the sake of two describe calls. The inline
    replacement keeps the two describes and the metric namespace.
    """
    assert "AmazonElasticFileSystemsUtils" not in source
    for kept in (
        "elasticfilesystem:DescribeMountTargets",
        "ec2:DescribeAvailabilityZones",
    ):
        assert kept in source
    for dropped in ("ssmmessages:", "ec2messages:", "ssm:GetParameter"):
        assert dropped not in source


def test_the_csi_driver_policy_arn_carries_its_service_role_path(source: str) -> None:
    """The EKS user guide prints this policy name without its path, and that ARN does
    not exist -- `iam:GetPolicy` on the pathless form answers `NoSuchEntity`. Copying
    the doc string produces an apply-time failure that names a policy rather than a
    path, so the path is asserted where the mistake would be made.
    """
    assert "arn:aws:iam::aws:policy/service-role/AmazonS3FilesCSIDriverPolicy" in source
    assert "arn:aws:iam::aws:policy/AmazonS3FilesCSIDriverPolicy" not in source


def test_the_addon_binds_each_service_account_to_its_own_role(config: str) -> None:
    """The other half of the trust-policy claim, which nothing was making.

    A trust policy says which service account may assume a role;
    `configuration_values` says which role each service account is handed. Grading one
    without the other leaves the pair half-checked, and the ungraded half is the
    dangerous one: pointing the `node` section's annotation at
    `aws_iam_role.s3files_csi_controller.arn` is a one-line edit that gives every node
    pod the controller's `AmazonEFSCSIDriverPolicy` and `AmazonS3FilesCSIDriverPolicy`
    -- CreateAccessPoint and DeleteAccessPoint on `Resource: "*"` -- while the trust
    policies, the attachments and the inline documents all stay exactly what `ROLES`
    says they are, because none of them moved.

    Read out of the rendered `configuration_values` rather than off the text, so the
    pairing is asserted as the structure the add-on is handed instead of as two strings
    that happen to sit near each other. The section list is compared both ways: a
    section this does not name would carry an unexamined binding, and a name with no
    section would be an expectation nothing reaches. The two roles are exactly the two
    `CSI_SERVICE_ACCOUNTS` names their trust policies are graded under, which is what
    makes the two halves one claim.
    """
    values = Configuration(config).only("aws_eks_addon", "efs_csi")[
        "configuration_values"
    ]
    assert set(values) == {"controller", "node"}, (
        f"the add-on is configured for {sorted(values)}; a section nothing names "
        f"here carries a role binding nothing examines"
    )
    bound = {
        section: body["serviceAccount"]["annotations"]["eks.amazonaws.com/role-arn"]
        for section, body in values.items()
    }
    assert bound == {
        "controller": "${aws_iam_role.s3files_csi_controller.arn}",
        "node": "${aws_iam_role.s3files_csi_node.arn}",
    }, (
        f"the add-on hands its service accounts {bound}; each one must get its own "
        f"role, or a pod holds a role whose trust policy was graded for another"
    )
    assert set(bound.values()) == {
        f"${{aws_iam_role.{role}.arn}}" for role in CSI_SERVICE_ACCOUNTS
    }, "a role bound here is not one of the two whose trust policy is graded above"


def test_the_addon_version_is_pinned(config: str) -> None:
    addon = assigns(one(config, "resource", "aws_eks_addon", "efs_csi"))
    assert addon.get("addon_name") == '"aws-efs-csi-driver"'
    assert addon.get("addon_version") == '"v3.4.1-eksbuild.1"'
    assert addon.get("resolve_conflicts_on_create") == '"OVERWRITE"'
    assert addon.get("cluster_name") == "aws_eks_cluster.map_dev.name"


# --- Tier A': terraform's own rendering of the same documents, opt-in. --------

_RENDER_GATE = "MAP_TERRAFORM_RENDER"

requires_a_rendered_plan = pytest.mark.skipif(
    os.environ.get(_RENDER_GATE) != "1",
    reason=(
        f"THE TABLE WAS NOT CHECKED AGAINST TERRAFORM: set {_RENDER_GATE}=1 to "
        "render the plan. It needs terraform on PATH, AWS credentials and the "
        "state bucket, and a bare `pytest` in this repo needs none of those. Every "
        "Tier A case still ran and they are what grades the documents; this one "
        "grades whether ROLES describes what terraform actually renders."
    ),
)


def _agrees(mine: str, theirs: str, bound: dict[str, str]) -> bool:
    """Whether `theirs` is what `mine` renders to, binding each `${...}` in it.

    A `${...}` the reader could not resolve stands for one value terraform did
    resolve, so it is matched as a hole and the value it captures is remembered.
    Remembered rather than discarded because the consistency is the check: the same
    reference has to render as the same string everywhere it appears, and a table
    that says two grants name the same bucket must not agree with a plan where they
    name two.
    """
    pieces = _INTERPOLATION.split(mine)
    literals, symbols = pieces[0::2], pieces[1::2]
    if not symbols:
        return mine == theirs
    pattern = re.escape(literals[0]) + "".join(
        f"(.+){re.escape(literal)}" for literal in literals[1:]
    )
    found = re.fullmatch(pattern, theirs)
    if found is None:
        return False
    trial = dict(bound)
    for symbol, value in zip(symbols, found.groups(), strict=True):
        if trial.setdefault(symbol, value) != value:
            return False
    bound.update(trial)
    return True


def _same_document(
    mine: Rendered,
    theirs: Rendered,
    bound: dict[str, str],
    at: str,
    source: str = "terraform renders",
) -> None:
    """Assert two documents are the same one, `${...}` treated as a hole.

    `source` only names where `theirs` came from, so a failure says which authority
    disagreed with the table. It matters because the two callers consult different
    ones: the plan says what terraform WOULD apply, IAM says what it actually holds,
    and "terraform renders X" is an actively misleading way to report the second.

    Structural in both directions at every level: a dict whose key set differs in
    size, a list of a different length and a scalar that differs are each a
    failure, so a statement terraform renders and the table does not name is a
    diff rather than something the walk steps over. Keys are unified too, because
    `"${local.oidc_issuer}:aud"` is a KEY and terraform renders it with the issuer
    substituted.
    """
    if isinstance(mine, str):
        assert isinstance(theirs, str) and _agrees(mine, theirs, bound), (
            f"{at}: the table says {mine!r}; {source} {theirs!r}"
        )
    elif isinstance(mine, dict):
        assert isinstance(theirs, dict) and len(mine) == len(theirs), (
            f"{at}: the table names {sorted(mine)}; terraform renders {theirs!r}"
        )
        left = dict(theirs)
        for key in mine:
            paired = [other for other in left if _agrees(key, other, dict(bound))]
            assert len(paired) == 1, (
                f"{at}.{key} matches {paired} of terraform's keys {sorted(left)}"
            )
            _agrees(key, paired[0], bound)
            _same_document(mine[key], left.pop(paired[0]), bound, f"{at}.{key}", source)
    elif isinstance(mine, list):
        assert isinstance(theirs, list) and len(mine) == len(theirs), (
            f"{at}: the table names {len(mine)} entries; terraform renders {theirs!r}"
        )
        for index, (ours, other) in enumerate(zip(mine, theirs, strict=True)):
            _same_document(ours, other, bound, f"{at}[{index}]", source)
    else:
        assert mine == theirs, f"{at}: the table says {mine!r}; {source} {theirs!r}"


def _rendered_plan(tmp_path: Path) -> dict[str, Any]:
    """`terraform show -json` over a real plan of this directory.

    Against a COPY, for the reason `test_terraform_declares_the_account.py` gives at
    length: `terraform init` rewrites `.terraform/` in whatever directory it runs in,
    and `deploy/terraform/.terraform/` is where the drift gate records that it is
    initialised. `TF_DATA_DIR` under `tmp_path` keeps even the copy's init out of the
    repository, and `-lock=false` keeps a read-only plan from taking a lock somebody
    else's apply needs.

    Shared by every case that needs terraform's own rendering rather than this file's
    reading of the text, so the rendering happens one way and a second caller cannot
    drift into planning the directory differently.
    """
    if shutil.which("terraform") is None:
        pytest.fail("terraform is not on PATH, so the plan was NOT rendered")
    config = tmp_path / "deploy" / "terraform"
    shutil.copytree(_CONFIG.parent, config, ignore=shutil.ignore_patterns(".terraform"))
    shutil.copytree(_REPO / "deploy" / "iam", tmp_path / "deploy" / "iam")
    env = {**os.environ, "TF_DATA_DIR": str(tmp_path / "tfdata")}

    def run(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["terraform", *argv, "-no-color"],
            cwd=config,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=900,
        )

    started = run("init", "-input=false")
    assert started.returncode == 0, started.stdout + started.stderr
    # 0 is "no changes" and 2 is "changes present". Every object in this file is a
    # create until somebody applies, so 2 is the expected outcome and 1 is the failure.
    planned = run(
        "plan", "-out=tfplan", "-lock=false", "-input=false", "-detailed-exitcode"
    )
    assert planned.returncode in (0, 2), planned.stdout + planned.stderr
    shown = run("show", "-json", "tfplan")
    assert shown.returncode == 0, shown.stderr
    return cast(dict[str, Any], json.loads(shown.stdout))


def _planned_documents(
    tmp_path: Path, ours: frozenset[str]
) -> dict[str, RoleDocuments]:
    """The file's roles as `terraform show -json` renders them, documents and edges.

    Against a COPY, for the reason `test_terraform_declares_the_account.py` gives at
    length: `terraform init` rewrites `.terraform/` in whatever directory it runs in,
    and `deploy/terraform/.terraform/` is where the drift gate records that it is
    initialised. `TF_DATA_DIR` under `tmp_path` keeps even the copy's init out of the
    repository, and `-lock=false` keeps a read-only plan from taking a lock somebody
    else's apply needs.

    TWO SECTIONS OF THE PLAN, because neither carries both halves. The documents come
    from `resource_changes`, which is where the rendered JSON is. The role each policy
    is written onto comes from `configuration.root_module.resources`, because an inline
    policy's `role` is NOT in its `after` at all: `role = aws_iam_role.X.id` and `id` is
    computed, so the value is unknown before the apply and the key is absent. Reading
    the edge out of terraform's own `references` list also keeps it terraform's answer
    rather than a second parse of the same file by the module under test.

    `ours` scopes the result to the roles `session_vfs.tf` declares. The plan covers the
    whole root module, and `iam.tf` declares five more roles that are not this slice's
    to grade.
    """
    plan = _rendered_plan(tmp_path)

    written_onto: dict[str, str] = {}
    for resource in cast(
        "list[dict[str, Any]]", plan["configuration"]["root_module"]["resources"]
    ):
        expressions = cast(dict[str, Any], resource.get("expressions", {}))
        reference = expressions.get("role") or expressions.get("role_name") or {}
        match = _ROLE_REFERENCE.match((reference.get("references") or [""])[0])
        if match and match.group(1) in ours:
            written_onto[resource["address"]] = match.group(1)

    trusts: dict[str, Rendered] = {}
    managed: dict[str, set[str]] = {name: set() for name in ours}
    inline: dict[str, dict[str, Rendered]] = {name: {} for name in ours}
    for change in cast("list[dict[str, Any]]", plan["resource_changes"]):
        after = cast("dict[str, Any] | None", change["change"]["after"])
        if after is None:
            continue
        kind = change["type"]
        if kind == "aws_iam_role" and change["name"] in ours:
            assert "assume_role_policy" in after, (
                f"{change['address']}: terraform could not render the trust policy, "
                f"so it was NOT compared"
            )
            trusts[change["name"]] = json.loads(after["assume_role_policy"])
        # A for_each'd resource's address carries its key; the edge is per resource.
        role = written_onto.get(str(change["address"]).split("[")[0])
        if role is None:
            continue
        if kind == "aws_iam_role_policy_attachment":
            assert "policy_arn" in after, f"{change['address']}: ARN unknown"
            managed[role].add(after["policy_arn"])
        elif kind == "aws_iam_role_policy":
            assert "policy" in after and "name" in after, (
                f"{change['address']}: terraform could not render the document, so "
                f"it was NOT compared"
            )
            inline[role][after["name"]] = json.loads(after["policy"])
    return {
        name: RoleDocuments(trust, frozenset(managed[name]), inline[name])
        for name, trust in trusts.items()
    }


@requires_a_rendered_plan
def test_terraform_renders_the_same_documents_the_reader_reads(
    config: str, tmp_path: Path
) -> None:
    """`ROLES` graded against the plan, which is the only authority on what applies.

    Tier A proves the reader agrees with `ROLES`. That leaves one way for both to
    be wrong together: a reader that misreads an expression, and a table written to
    match what it read. This closes it from the other side -- the table against the
    documents terraform itself computes, with every reference resolved to a real
    ARN and every `${...}` in the table treated as a hole that has to be filled
    consistently.

    It also states the cost plainly. This needs credentials, network and the state
    bucket, so the default suite skips it and says so in capitals. The guard that
    runs everywhere is Tier A; this is the guard on the guard, and a round that
    skipped it has checked the reader against nothing.
    """
    ours = frozenset(Configuration(config).named("aws_iam_role"))
    rendered = _planned_documents(tmp_path, ours)
    assert set(rendered) == set(ROLES), (
        f"terraform plans roles ROLES does not name: "
        f"{sorted(set(rendered) - set(ROLES))}; and names roles it does not plan: "
        f"{sorted(set(ROLES) - set(rendered))}"
    )
    bound: dict[str, str] = {}
    for name in sorted(ROLES):
        expected, planned = ROLES[name], rendered[name]
        _same_document(expected.trust, planned.trust, bound, f"{name}.trust")
        assert planned.managed == expected.managed, (
            f"{name} is attached {sorted(planned.managed)}; ROLES names "
            f"{sorted(expected.managed)}"
        )
        assert set(planned.inline) == set(expected.inline), (
            f"{name} holds inline policies {sorted(planned.inline)}; ROLES names "
            f"{sorted(expected.inline)}"
        )
        for policy in sorted(expected.inline):
            _same_document(
                expected.inline[policy],
                planned.inline[policy],
                bound,
                f"{name}.{policy}",
            )


@requires_a_rendered_plan
def test_terraform_renders_one_import_rule_scoped_to_the_whole_bucket(
    tmp_path: Path,
) -> None:
    """The import rule's `prefix` is the whole-bucket form, on the RENDERED value.

    `prefix` is an S3 KEY prefix and `""` is the documented spelling for the entire
    bucket. This said `"/"` -- which is neither `""` nor a real key prefix, because no
    object in this bucket has a key beginning with a slash -- and the docs require
    exactly one import rule covering the root, so the configuration carried no root rule
    at all while looking like it did.

    Graded here rather than in Tier A, and on the rendered value rather than the text,
    because nothing cheaper can see it. The provider declares `prefix` as a bare
    required string with no validator, so `terraform plan` renders `"/"` as a clean
    create and only `PutSynchronizationConfiguration` ever objects -- at apply time,
    against a file system that has to exist first. Reading the plan is the earliest
    point the real value is observable.

    The count is asserted with the value. One rule whose prefix is wrong and a second
    that is right would satisfy "some rule covers the root" while leaving the wrong one
    shadowing part of the tree, and the docs cap the whole set at 10 with exactly one
    for the root.

    Note what is deliberately NOT asserted: the access point's `root_directory.path`,
    which is `"/"` and correct. It is a POSIX path in a different namespace, and a guard
    that pushed the two fields into agreement would break the mount to fix the spelling.
    """
    plan = _rendered_plan(tmp_path)
    rules = [
        rule
        for change in cast("list[dict[str, Any]]", plan.get("resource_changes", []))
        if change["type"] == "aws_s3files_synchronization_configuration"
        for rule in (change["change"]["after"] or {}).get("import_data_rule") or []
    ]
    assert len(rules) == 1, (
        f"terraform renders {len(rules)} import rules; S3 Files requires exactly one "
        f"for the root directory: {rules}"
    )
    assert rules[0]["prefix"] == "", (
        f"terraform renders the import rule's prefix as {rules[0]['prefix']!r}; the "
        f'whole-bucket form is "" and any other value leaves the root with no '
        f"rule, which PutSynchronizationConfiguration refuses only at apply time"
    )


# --- Tier A'': the documents IAM actually holds, opt-in. -----------------------
#
# WHY THIS TIER EXISTS AT ALL, when Tier A grades the text and Tier A' grades the plan:
# because neither of them can see the defect that stopped this slice's apply. The trust
# policy said `Service = "s3files.amazonaws.com"`, a spelling IAM does not know, and
# `CreateRole` refused it with MalformedPolicyDocument. The text reader saw a well
# formed
# trust document. The plan renderer saw a well-formed trust document and a clean create.
# Both were right about what the file says, and the file said something IAM will not
# accept -- so a green suite sat in front of an apply that could not run.
#
# The authority on whether a principal is real is IAM, and the only way to ask IAM is to
# look at what IAM ended up holding. That is the whole of this tier: for every role the
# file declares, read the role back out of the account and compare the document. A
# principal IAM refuses cannot produce a role, so the read fails and the case fails --
# naming the principal, because "the role is absent" is otherwise a confusing sign of
# a spelling mistake.
#
# It is opt-in for the same reason Tier A' is: credentials and network. A round that
# skipped it has NOT checked any spelling against IAM, and the skip reason says so.

_READBACK_GATES = ("MAP_CLUSTER_TESTS", "MAP_IAM_READBACK")
"""Either variable runs this tier, and `MAP_CLUSTER_TESTS` is the one that matters.

It was gated on `MAP_IAM_READBACK` alone, a name that appears nowhere else in the
repository -- not in a document, not in a runbook, not in any invocation anyone has
recorded. Every other live tier in this suite is gated on `MAP_CLUSTER_TESTS=1`, so a
run that deliberately turned the live tiers on still skipped the only case in the tree
that can adjudicate a service principal, and reported a clean pass. That is how a wrong
principal survived six rounds of verification: the check existed, and no invocation
reached it.

Both are honoured rather than one, because widening when to run a guard is the safe
direction and a private name that already works should not start failing silently.
"""

requires_the_live_roles = pytest.mark.skipif(
    not any(os.environ.get(gate) == "1" for gate in _READBACK_GATES),
    reason=(
        "NO PRINCIPAL WAS CHECKED AGAINST IAM: set "
        f"{'=1 or '.join(_READBACK_GATES)}=1 to read the roles back out of the "
        "account. It needs AWS credentials, and Tier A cannot substitute -- a trust "
        "document IAM refuses is well-formed text and a clean plan, so every offline "
        "case passes over a role that does not exist."
    ),
)


def _live_role(name: str) -> dict[str, Any] | None:
    """One role as IAM holds it, or None when IAM holds no such role.

    None rather than an exception for the absent case, because absent is a RESULT here
    and not an error: a trust policy naming a principal IAM does not know is refused at
    `CreateRole`, so the role never comes into being, and that is the outcome this tier
    is built to report. Any other failure -- no credentials, a throttle, a denial -- is
    re-raised as a test failure instead, so a broken connection cannot read as "the
    role is missing" and quietly become the finding.
    """
    listed = subprocess.run(
        ["aws", "iam", "get-role", "--role-name", name, "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode == 0:
        return cast(dict[str, Any], json.loads(listed.stdout)["Role"])
    if "NoSuchEntity" in listed.stderr:
        return None
    pytest.fail(
        f"could not read role {name!r} from IAM: {listed.stdout}{listed.stderr}"
    )


@requires_the_live_roles
def test_iam_holds_every_role_this_file_declares(config: str) -> None:
    """Each declared role exists in IAM, which is what proves its principal is real.

    This is the case the five failed rounds needed and did not have. `CreateRole` is
    the only thing that adjudicates a service principal, and it adjudicates by refusing
    -- so the observable consequence of a wrong spelling is not a bad document anywhere,
    it is the ABSENCE of the role. Nothing that reads the configuration can see an
    absence; only the account can.

    Enumerated over every role the file declares rather than over the one that was
    wrong, so the guard covers a role added later, and it reports the principal each
    trust names so that a failure reads as the spelling problem it is.
    """
    declared = Configuration(config)
    missing = []
    for name in sorted(declared.named("aws_iam_role")):
        role_name = cast(str, declared.only("aws_iam_role", name)["name"])
        if _live_role(role_name) is None:
            principals = [
                value
                for statement in documents(config)[name].trust["Statement"]
                for value in statement.get("Principal", {}).values()
            ]
            missing.append(f"{name} (as {role_name!r}, trusting {principals})")
    assert not missing, (
        "IAM holds no such role, so these were never created: "
        + "; ".join(missing)
        + ". A trust policy naming a principal IAM does not know is refused outright "
        "at CreateRole, which is exactly what this looks like -- check the principal "
        "against the service's own documentation rather than against the resource type"
    )


@requires_the_live_roles
def test_iam_holds_the_trust_documents_this_file_declares(config: str) -> None:
    """The live trust document equals the declared one, compared whole and both ways.

    Existence alone is not enough: a role created once with a good principal and since
    edited by hand -- in the console, by another configuration, by an unrelated apply --
    keeps existing while trusting somebody else. This role holds the object-delete grant
    on the platform bucket, so WHO may assume it is the whole of its blast radius, and
    the account is the only place that answers.

    Compared with `_same_document` so that a `${...}` in `ROLES` binds to whatever
    terraform resolved it to, consistently across the roles -- the same treatment Tier
    A' gives the plan, pointed at the account instead.
    """
    declared = Configuration(config)
    bound: dict[str, str] = {}
    for name in sorted(declared.named("aws_iam_role")):
        role_name = cast(str, declared.only("aws_iam_role", name)["name"])
        role = _live_role(role_name)
        assert role is not None, (
            f"{name} is declared here and IAM holds no role {role_name!r}; "
            f"test_iam_holds_every_role_this_file_declares says why that happens"
        )
        _same_document(
            ROLES[name].trust,
            role["AssumeRolePolicyDocument"],
            bound,
            f"{name}.trust (live in IAM)",
            source="IAM holds",
        )


@requires_the_live_roles
def test_the_file_system_is_available_and_not_merely_created(config: str) -> None:
    """The file system reports `available`, which is the only status that means working.

    `CreateFileSystem` returns HTTP 200 and a real file system id for a role S3 Files
    cannot assume, and for a caller that cannot finish provisioning it. The verdict is
    asynchronous and it lands in `GetFileSystem.statusMessage` -- so "terraform created
    it", "the API accepted it" and "the plan is empty" are all true of a file system
    that never works. Measured: a file system sat in `creating` for five minutes with
    `Access denied` in its status message and then went to `error`.

    So this grades the status and nothing else. It reads the id out of terraform state
    rather than by listing, because listing finds any file system in the region and the
    claim is about THIS one; and it SKIPS rather than passes when state holds no file
    system, because a case that silently passes on an absent resource is how a blocked
    object comes to look finished.
    """
    state = subprocess.run(
        ["terraform", "show", "-json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_CONFIG.parent,
    )
    assert state.returncode == 0, state.stdout + state.stderr
    values = json.loads(state.stdout).get("values", {}).get("root_module", {})
    identifiers = [
        resource["values"]["id"]
        for resource in values.get("resources", [])
        if resource.get("type") == "aws_s3files_file_system"
    ]
    if not identifiers:
        pytest.skip(
            "THE FILE SYSTEM WAS NOT GRADED: terraform state holds no "
            "aws_s3files_file_system, so it has not been created. This case is the "
            "only one that can tell a working file system from a created one, and it "
            "has nothing to read -- see docs/progress.md for why it is not applied."
        )
    for identifier in identifiers:
        described = subprocess.run(
            [
                "aws",
                "s3files",
                "get-file-system",
                "--file-system-id",
                identifier,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert described.returncode == 0, described.stdout + described.stderr
        live = json.loads(described.stdout)
        assert live["status"] == "available", (
            f"file system {identifier} reports status {live['status']!r}, not "
            f"'available': {live.get('statusMessage', '(no status message)')}. A "
            f"status other than available means the mount cannot serve, however "
            f"cleanly it was created"
        )


# --- Tier B: the real mount, opt-in. ------------------------------------------

_PROVISION_GATE = "MAP_PROVISION_SESSION_VFS"
VFS_BUCKET = "map-dev-062677866851-us-east-1-an"
PROBE_POD = "map61-vfs-probe"
PROBE_PVC = "map61-vfs-probe"
PROBE_PV = "map61-vfs-probe"

requires_a_provisioned_mount = pytest.mark.skipif(
    os.environ.get(_PROVISION_GATE) != "1",
    reason=(
        f"the real mount is opt-in: set {_PROVISION_GATE}=1 to run it. It needs an "
        "applied deploy/terraform/, a kubeconfig for map-dev, and credentials holding "
        "s3files:* -- none of which the planning identity has. SKIPPED MEANS NOTHING "
        "WAS MOUNTED: every case above asserts only what the configuration says, and "
        "an emptyDir at the right path satisfies all of them."
    ),
)

_PV_MANIFEST = """
apiVersion: v1
kind: PersistentVolume
metadata:
  name: map61-vfs-probe
spec:
  capacity: {{ storage: 1Gi }}
  volumeMode: Filesystem
  accessModes: [ReadWriteMany]
  persistentVolumeReclaimPolicy: Retain
  csi:
    driver: efs.csi.aws.com
    # Unverified: this is the EFS CSI access-point handle form. If S3 Files differs, the
    # pod stays ContainerCreating and `kubectl describe pod` names the volume.
    volumeHandle: {file_system_id}::{access_point_id}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: map61-vfs-probe
spec:
  accessModes: [ReadWriteMany]
  storageClassName: ""
  volumeName: map61-vfs-probe
  resources: {{ requests: {{ storage: 1Gi }} }}
---
apiVersion: v1
kind: Pod
metadata:
  name: {pod}
spec:
  restartPolicy: Never
  securityContext: {{ runAsUser: 10001, runAsGroup: 10001, fsGroup: 10001 }}
  containers:
  - name: probe
    image: public.ecr.aws/amazonlinux/amazonlinux:2023
    command: ["/bin/bash", "-c"]
    args:
    - |
      set -e
      echo probe > /session/workspace/artifacts/{key}
      python3 -c "import os; \
        m = os.stat('/session/workspace/artifacts'); \
        s = os.stat('/session/scratch'); \
        print(f'mount_dev={{m.st_dev}} scratch_dev={{s.st_dev}}'); \
        print(f'mount_owner={{m.st_uid}}:{{m.st_gid}}')"
    volumeMounts:
    - {{ name: vfs, mountPath: /session/workspace/artifacts, subPath: artifacts/map61 }}
    - {{ name: scratch, mountPath: /session/scratch }}
  volumes:
  - {{ name: vfs, persistentVolumeClaim: {{ claimName: map61-vfs-probe }} }}
  - {{ name: scratch, emptyDir: {{}} }}
"""


def _kubectl(*argv: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *argv], capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture
def probe_pod() -> Any:
    """Name the probe pod, and delete everything the probe creates however it ended.

    All three objects, in the order that lets each go: the pod releases the claim, the
    claim releases the volume, and the volume is last. The PV is the one that does not
    clean itself up -- `persistentVolumeReclaimPolicy: Retain` is deliberate (a probe
    must not be able to delete a real file system) and it means a PV left behind stays
    `Released` for ever, holding a name the next run needs and pointing at a file
    system nobody can see from `kubectl`.

    A teardown rather than `--rm`, because a pod that never reaches Running is not
    deleted by `--rm`, and a leaked pod holding an NFS mount is a worse outcome than a
    failed test. `--ignore-not-found` on each, so a run that failed before creating
    one of them still tears down the others.
    """
    yield PROBE_POD
    _kubectl("delete", "pod", PROBE_POD, "--ignore-not-found", "--wait=true")
    _kubectl("delete", "pvc", PROBE_PVC, "--ignore-not-found", "--wait=true")
    _kubectl("delete", "pv", PROBE_PV, "--ignore-not-found", "--wait=true")


def _terraform_output(name: str) -> str:
    done = subprocess.run(
        [
            "terraform",
            f"-chdir={_REPO / 'deploy' / 'terraform'}",
            "output",
            "-raw",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, f"terraform output {name} failed: {done.stderr}"
    return done.stdout.strip()


def _await_probe(deadline_seconds: int = 300) -> str:
    """Wait until the probe pod terminates, and say which way it went.

    Polled rather than `kubectl wait --for=jsonpath=...=Succeeded`, because that call
    waits out its whole timeout on a pod that has already reached `Failed` -- five
    minutes of nothing, and then a timeout message rather than the reason. `Failed` is
    the interesting outcome here (a mount that did not happen, a write that was
    refused), so it is reported as soon as it is known, with the pod's own logs and
    the last event kubectl recorded.
    """
    deadline = time.monotonic() + deadline_seconds
    phase = ""
    while time.monotonic() < deadline:
        phase = _kubectl(
            "get", "pod", PROBE_POD, "-o", "jsonpath={.status.phase}", timeout=60
        ).stdout.strip()
        if phase == "Succeeded":
            return phase
        if phase == "Failed":
            logs = _kubectl("logs", PROBE_POD, timeout=60).stdout
            described = _kubectl("describe", "pod", PROBE_POD, timeout=60).stdout
            pytest.fail(f"the probe pod Failed.\nlogs:\n{logs}\ndescribe:\n{described}")
        time.sleep(5)
    described = _kubectl("describe", "pod", PROBE_POD, timeout=60).stdout
    pytest.fail(
        f"the probe pod was {phase or 'not found'} after {deadline_seconds}s, never "
        f"Succeeded:\n{described}"
    )


def _probe_logs(key: str) -> str:
    """Apply the probe manifests, wait for the pod to finish, and return its output."""
    manifest = _PV_MANIFEST.format(
        file_system_id=_terraform_output("session_vfs_file_system_id"),
        access_point_id=_terraform_output("session_vfs_access_point_id"),
        pod=PROBE_POD,
        key=key,
    )
    applied = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert applied.returncode == 0, applied.stderr
    _await_probe()
    return _kubectl("logs", PROBE_POD).stdout


@requires_a_provisioned_mount
def test_the_file_system_mounts_and_a_write_reaches_the_bucket(probe_pod: str) -> None:
    """The write is half of it; the object appearing in the bucket is the other half.

    Polled rather than asserted once: S3 Files waits for 60 seconds of write inactivity
    before exporting a change back to the bucket, so a single immediate head-object
    fails intermittently and looks like a permissions problem.
    """
    key = f"probe-{os.getpid()}.txt"
    assert "mount_dev=" in _probe_logs(key)

    deadline = time.monotonic() + 240
    last = ""
    while time.monotonic() < deadline:
        found = subprocess.run(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                VFS_BUCKET,
                "--key",
                f"artifacts/map61/{key}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if found.returncode == 0:
            return
        last = found.stderr
        time.sleep(15)
    pytest.fail(f"artifacts/map61/{key} never reached {VFS_BUCKET}: {last}")


@requires_a_provisioned_mount
def test_the_mount_is_not_the_pods_own_disk(probe_pod: str) -> None:
    """The assertion the write cannot make.

    A write into an emptyDir at the same path succeeds identically, so the case above
    passes against a mount that never happened -- which is exactly the hole
    `managed_agent.session_shim.vfs_mount.verify_ready` compares st_dev to close. This
    is the same comparison made from outside the pod, plus the owner the access point
    forces.
    """
    logs = _probe_logs(f"probe-dev-{os.getpid()}.txt")
    devices = dict(
        pair.split("=", 1)
        for pair in logs.split()
        if pair.startswith(("mount_dev", "scratch_dev"))
    )
    assert devices["mount_dev"] != devices["scratch_dev"], (
        f"the mount shares a device with pod-local scratch: {logs}"
    )
    assert "mount_owner=10001:10001" in logs
