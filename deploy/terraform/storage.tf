# The Evidence and Artifact bucket, and the bucket the Session VFS synchronises
# into a Session pod. Which is why Terraform's own state is NOT in here: see
# versions.tf and ADR-021.
#
# Four resources, because the AWS provider splits a bucket's settings out of the
# bucket. Importing only `aws_s3_bucket` would declare the bucket's existence and
# leave versioning, public access and encryption undeclared -- three settings whose
# silent change is exactly the class of event this directory exists to catch.
import {
  to = aws_s3_bucket.platform
  id = "map-dev-${local.account_id}-us-east-1-an"
}
resource "aws_s3_bucket" "platform" {
  bucket = "map-dev-${local.account_id}-us-east-1-an"
  lifecycle {
    prevent_destroy = true
  }
}

# environment.md: versioning is what lets a hand-back after a Takeover be
# reconciled against what the agent last wrote. Turning it off would not fail
# anything until the first reconciliation, and then silently.
#
# prevent_destroy for that exact reason, and on the public-access block for the
# obvious one. A bucket's settings are separate resources from the bucket, so
# guarding `aws_s3_bucket.platform` alone protects the container and not the two
# properties anything actually depends on. The encryption resource is not guarded:
# S3 applies SSE-S3 whether or not it exists, so destroying it changes nothing.
import {
  to = aws_s3_bucket_versioning.platform
  id = "map-dev-${local.account_id}-us-east-1-an"
}
resource "aws_s3_bucket_versioning" "platform" {
  bucket = aws_s3_bucket.platform.id
  versioning_configuration {
    status = "Enabled"
  }
  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = aws_s3_bucket_public_access_block.platform
  id = "map-dev-${local.account_id}-us-east-1-an"
}
resource "aws_s3_bucket_public_access_block" "platform" {
  bucket                  = aws_s3_bucket.platform.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = aws_s3_bucket_server_side_encryption_configuration.platform
  id = "map-dev-${local.account_id}-us-east-1-an"
}
resource "aws_s3_bucket_server_side_encryption_configuration" "platform" {
  bucket = aws_s3_bucket.platform.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}
