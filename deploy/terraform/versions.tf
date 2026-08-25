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
# The state bucket is NOT map-dev-062677866851-us-east-1-an. That bucket is the one
# the Session VFS synchronises into a Session pod's filesystem, so state kept there
# would be readable -- and writable -- by whatever the agent runs (ADR-021).
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
    bucket       = "map-dev-tfstate-062677866851-us-east-1"
    key          = "map-dev/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = "us-east-1"
}
