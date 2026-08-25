# The bucket holding this configuration's own state, imported into that state.
#
# Self-referential on purpose. The alternative is that the one bucket whose loss
# destroys the record of everything else keeps its versioning and its public-access
# block in nobody's memory -- which is the situation ADR-021 was written about,
# reproduced in the most expensive possible place. prevent_destroy contains the
# obvious hazard: a plan that would destroy this exits 1 with "Instance cannot be
# destroyed" and refuses, rather than deleting the state mid-run.
#
# It cannot be created from here -- Step 1 creates it with four `aws s3api`
# commands, and Terraform enforces the ordering itself: with this file present and
# the bucket absent, the plan ends `Error: Cannot import non-existent remote object
# ... aws_s3_bucket.tfstate`, exit 1.
import {
  to = aws_s3_bucket.tfstate
  id = "map-dev-tfstate-062677866851-us-east-1"
}
resource "aws_s3_bucket" "tfstate" {
  bucket = "map-dev-tfstate-062677866851-us-east-1"
  lifecycle {
    prevent_destroy = true
  }
}

# ADR-021 requires versioning on this bucket, and S3-native locking wants it too:
# `use_lockfile` writes the lock as an object beside the state.
#
# prevent_destroy, because a bucket's settings are separate resources from the
# bucket: guarding `aws_s3_bucket.tfstate` alone leaves a plan free to destroy
# the versioning that makes a corrupted state recoverable and the public-access
# block that keeps the account's whole resource inventory off the internet, while
# the bucket itself sits there looking protected. The encryption resource below is
# deliberately NOT guarded -- S3 applies SSE-S3 to every object whether or not
# that resource exists, so destroying it changes nothing about the data.
import {
  to = aws_s3_bucket_versioning.tfstate
  id = "map-dev-tfstate-062677866851-us-east-1"
}
resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = aws_s3_bucket_public_access_block.tfstate
  id = "map-dev-tfstate-062677866851-us-east-1"
}
resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = aws_s3_bucket_server_side_encryption_configuration.tfstate
  id = "map-dev-tfstate-062677866851-us-east-1"
}
resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}
