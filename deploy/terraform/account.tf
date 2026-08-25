# Which account this configuration is talking about, asked rather than written down.
#
# Every ARN below used to carry a twelve-digit literal, in ninety-odd places across nine
# files. That is a second copy of a fact the credentials already carry, and it fails in
# the two directions a duplicated fact always does: applied with credentials for another
# account it plans against resources that do not exist there, and published in a public
# repository it hands a reader the account it was developed in.
#
# `aws_caller_identity` is the same authority `deploy/docker/push-platform-image.sh`
# already derives the registry host from, which is why this is the answer rather than a
# variable in a `.tfvars` -- a variable would be a third place to keep the account
# correct, and one that can disagree with the credentials in the environment while
# looking perfectly well-formed.
#
# It resolves at plan time, so it reaches the `import` blocks too. Measured before this
# was written, because the ordering is not obvious and a data source that resolved too
# late would fail every import at once: a scratch configuration importing an IAM policy
# by an id built from this local planned `1 to import` against the live account.
#
# The one thing it cannot reach is the S3 backend in `versions.tf`. A backend block is
# read before any provider is configured, so it takes no expressions at all -- see the
# partial-configuration comment there.

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id

  # What the committed IAM policy documents under `deploy/iam/` carry in place of an
  # account, substituted into them by `replace()` in irsa.tf. Twelve zeros, and the same
  # twelve zeros `deploy/k8s/`'s manifests carry and `deploy/platform.py` substitutes at
  # apply time -- ONE spelling across all three, because several tests cross-check a
  # bucket name in a manifest against the bucket name in the policy that grants it, and
  # two spellings of "no account yet" would make those comparisons fail while both sides
  # were correct.
  account_placeholder = "000000000000"
}
