# The provider, the state, and the one lock this configuration can actually take.
#
# `region` is a literal rather than $AWS_REGION. Every subnet, security group and
# KMS key id below is regional, so a region read from the environment could point
# the comparison at an account holding none of them and report a clean plan by
# describing nothing. `profile` is deliberately absent for the opposite reason: it
# names an entry in a developer's local credential file, which is not a property of
# the account. Credentials come from the environment (`AWS_PROFILE`), as
# `environment.md`'s AWS_PROFILE row says.
#
# The state bucket is NOT the platform bucket `map-dev-<account>-<region>-an`. That is
# the one the Session VFS synchronises into a Session pod's filesystem, so state kept
# there would be readable -- and writable -- by whatever the agent runs (ADR-021).
#
# `bucket` is absent, which makes this a PARTIAL configuration: the name embeds the
# account id, and a backend block is the one place in Terraform that can take no
# expression at all. It is read before any provider is configured, so
# `local.account_id` -- which every other file here uses -- does not exist yet. There is
# no version of this that both names the bucket and keeps the account out of the
# repository, so the name arrives at init time instead:
#
#     terraform init -backend-config=backend.hcl
#
# `backend.hcl` is gitignored; `backend.hcl.example` beside it carries the shape. Running
# a bare `terraform init` against this asks for the bucket interactively rather than
# guessing one, which is the failure mode to want -- a wrong guess would silently start a
# second state file for an environment that already has one.
#
# `use_lockfile` rather than a DynamoDB table: S3-native locking, holding the lock
# as an object beside the state. Two readings decided it. `aws dynamodb list-tables`
# returns AccessDeniedException for this identity, so a lock table cannot be read,
# let alone created; and Terraform 1.13.4 accepts `use_lockfile` (added in 1.10)
# while rejecting an unknown backend argument outright with "Unsupported argument",
# so the acceptance is a positive result and not silence.
terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  backend "s3" {
    key          = "map-dev/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = "us-east-1"
}
