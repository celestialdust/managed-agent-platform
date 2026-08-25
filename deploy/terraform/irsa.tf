# The four roles a workload in this cluster assumes, and the policies they carry.
#
# Separate from iam.tf because these are a different kind of thing. The roles
# there are assumed by AWS services -- EKS assumes the cluster role, EC2 assumes
# the node role -- and exist so the cluster can run at all. These are assumed by
# pods, through the cluster's OIDC provider, and each one is the whole reason a
# process holds no long-lived credential: `deploy/k8s/cluster-bootstrap.yaml`
# annotates a service account with the role arn and the pod gets a token.
#
# Three of them were hand-made on 2026-08-22 and declared nowhere, which is the
# exact shape ADR-021 rejects. What made that dangerous rather than untidy is the
# trust policy: the condition below is what decides WHICH service account may
# become this role, so a second subject added to `map-tool-gateway` -- the identity
# in front of every tool call -- would hand every tool credential to whatever runs
# under it, and nothing in this repository would have been comparing.
#
# Three of the four roles get the same five resources: the role, its
# customer-managed policy, the attachment, and the two exclusive sets that make the
# attachment set and the inline set authoritative. See iam.tf on why the exclusive
# pair is not optional. The fourth carries an inline policy instead, because
# `iam:CreatePolicy` is denied to the identity that applies this; see its own
# section.

locals {
  # The issuer as IAM condition keys spell it: the provider's URL with the scheme
  # removed. Derived from the provider resource rather than written out again, so
  # the cluster's issuer id stays in one place (iam.tf) and a cluster rebuild that
  # changes it cannot leave three stale copies here.
  oidc_issuer = replace(
    aws_iam_openid_connect_provider.cluster.url, "https://", ""
  )
}

# --- map-control-plane: the database credential, and uploaded file objects. ---
# The S3 grant is objects-only on the artifacts bucket and carries no Delete: an
# uploaded file is referenced by a Session's mount long after the upload call
# returned, so the control plane must never be able to remove one. Until this
# statement existed the manifest left MAP_OBJECT_BUCKET unset on purpose -- rather
# than turn every upload into an AccessDenied -- and `POST /v1/files` answered 500
# for the life of every Deployment. The variable is set now, and the applier lists
# it as required, so removing this statement breaks the endpoint rather than
# quietly reverting it.

import {
  to = aws_iam_role.control_plane
  id = "map-control-plane"
}
resource "aws_iam_role" "control_plane" {
  name = "map-control-plane"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:map-dev:control-plane"
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

import {
  to = aws_iam_policy.control_plane
  id = "arn:aws:iam::${local.account_id}:policy/map-control-plane"
}
resource "aws_iam_policy" "control_plane" {
  name = "map-control-plane"
  policy = replace(
    file("${path.module}/../iam/map-control-plane.json"),
    local.account_placeholder,
    local.account_id,
  )
}

import {
  to = aws_iam_role_policy_attachment.control_plane
  id = "map-control-plane/arn:aws:iam::${local.account_id}:policy/map-control-plane"
}
resource "aws_iam_role_policy_attachment" "control_plane" {
  role       = aws_iam_role.control_plane.name
  policy_arn = aws_iam_policy.control_plane.arn
}

import {
  to = aws_iam_role_policy_attachments_exclusive.control_plane
  id = "map-control-plane"
}
resource "aws_iam_role_policy_attachments_exclusive" "control_plane" {
  role_name   = aws_iam_role.control_plane.name
  policy_arns = [aws_iam_role_policy_attachment.control_plane.policy_arn]
}

import {
  to = aws_iam_role_policies_exclusive.control_plane
  id = "map-control-plane"
}
resource "aws_iam_role_policies_exclusive" "control_plane" {
  role_name    = aws_iam_role.control_plane.name
  policy_names = []
}

# --- map-model-gateway: reads the provider credentials and nothing else. ------

import {
  to = aws_iam_role.model_gateway
  id = "map-model-gateway"
}
resource "aws_iam_role" "model_gateway" {
  name = "map-model-gateway"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:map-dev:model-gateway"
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

import {
  to = aws_iam_policy.model_gateway
  id = "arn:aws:iam::${local.account_id}:policy/map-model-gateway"
}
resource "aws_iam_policy" "model_gateway" {
  name = "map-model-gateway"
  policy = replace(
    file("${path.module}/../iam/map-model-gateway.json"),
    local.account_placeholder,
    local.account_id,
  )
}

import {
  to = aws_iam_role_policy_attachment.model_gateway
  id = "map-model-gateway/arn:aws:iam::${local.account_id}:policy/map-model-gateway"
}
resource "aws_iam_role_policy_attachment" "model_gateway" {
  role       = aws_iam_role.model_gateway.name
  policy_arn = aws_iam_policy.model_gateway.arn
}

import {
  to = aws_iam_role_policy_attachments_exclusive.model_gateway
  id = "map-model-gateway"
}
resource "aws_iam_role_policy_attachments_exclusive" "model_gateway" {
  role_name   = aws_iam_role.model_gateway.name
  policy_arns = [aws_iam_role_policy_attachment.model_gateway.policy_arn]
}

import {
  to = aws_iam_role_policies_exclusive.model_gateway
  id = "map-model-gateway"
}
resource "aws_iam_role_policies_exclusive" "model_gateway" {
  role_name    = aws_iam_role.model_gateway.name
  policy_names = []
}

# --- map-tool-gateway: the tool credentials, and the evidence bucket. ---------

import {
  to = aws_iam_role.tool_gateway
  id = "map-tool-gateway"
}
resource "aws_iam_role" "tool_gateway" {
  name = "map-tool-gateway"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:map-dev:tool-gateway"
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

import {
  to = aws_iam_policy.tool_gateway
  id = "arn:aws:iam::${local.account_id}:policy/map-tool-gateway"
}
resource "aws_iam_policy" "tool_gateway" {
  name = "map-tool-gateway"
  policy = replace(
    file("${path.module}/../iam/map-tool-gateway.json"),
    local.account_placeholder,
    local.account_id,
  )
}

import {
  to = aws_iam_role_policy_attachment.tool_gateway
  id = "map-tool-gateway/arn:aws:iam::${local.account_id}:policy/map-tool-gateway"
}
resource "aws_iam_role_policy_attachment" "tool_gateway" {
  role       = aws_iam_role.tool_gateway.name
  policy_arn = aws_iam_policy.tool_gateway.arn
}

import {
  to = aws_iam_role_policy_attachments_exclusive.tool_gateway
  id = "map-tool-gateway"
}
resource "aws_iam_role_policy_attachments_exclusive" "tool_gateway" {
  role_name   = aws_iam_role.tool_gateway.name
  policy_arns = [aws_iam_role_policy_attachment.tool_gateway.policy_arn]
}

import {
  to = aws_iam_role_policies_exclusive.tool_gateway
  id = "map-tool-gateway"
}
resource "aws_iam_role_policies_exclusive" "tool_gateway" {
  role_name    = aws_iam_role.tool_gateway.name
  policy_names = []
}

# --- map-cluster-autoscaler: changes the nodegroup's size, and nothing else. ---
#
# FOUR resources, not the five above, and the difference is a permission rather
# than a preference: `iam:CreatePolicy` is implicitDeny to the identity that
# applies this directory, so this role cannot have a customer-managed policy of
# its own. The three above only work because their policies were made by hand
# before this file existed and are IMPORTED; a fourth would be a create, and the
# create is denied. So the policy is INLINE, via iam:PutRolePolicy, which is
# allowed. Turning it back into an aws_iam_policy will fail at apply.
#
# The exclusive pair stays, inverted: policy_arns = [] says this role carries no
# managed attachment, and policy_names names the one inline policy it does carry.
# Dropping either would make the set non-authoritative, which is the whole
# argument at the top of this file.
#
# Like everything else here it is IMPORTED rather than created, and that is what
# decides the order of the apply: the role and its inline policy are made with
# two `aws iam` calls first, and only then can this be planned. With the role
# absent the plan ends `Error: Cannot import non-existent remote object`, exit 1
# -- measured, and the same ordering state.tf records for the state bucket.
# Three tests depend on the import block: two require one on every resource, and
# the role-name comparison in tests/deploy/test_terraform_declares_the_account.py
# reads a role's NAME out of its import id, so a role without one is invisible to
# both the manifest cross-read and the account enumeration.
#
# This is the one identity in the cluster that can terminate an EC2 instance.
# Read the condition below on its own, not as part of a diff: one subject, both
# keys pinned. The header of this file says why -- a second subject here hands
# that power to whatever runs under it, with nothing comparing.

import {
  to = aws_iam_role.cluster_autoscaler
  id = "map-cluster-autoscaler"
}
resource "aws_iam_role" "cluster_autoscaler" {
  name = "map-cluster-autoscaler"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:map-dev:cluster-autoscaler"
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

import {
  to = aws_iam_role_policy.cluster_autoscaler
  id = "map-cluster-autoscaler:map-cluster-autoscaler"
}
resource "aws_iam_role_policy" "cluster_autoscaler" {
  name = "map-cluster-autoscaler"
  role = aws_iam_role.cluster_autoscaler.name
  policy = replace(
    file("${path.module}/../iam/map-cluster-autoscaler.json"),
    local.account_placeholder,
    local.account_id,
  )
}

import {
  to = aws_iam_role_policy_attachments_exclusive.cluster_autoscaler
  id = "map-cluster-autoscaler"
}
resource "aws_iam_role_policy_attachments_exclusive" "cluster_autoscaler" {
  role_name   = aws_iam_role.cluster_autoscaler.name
  policy_arns = []
}

import {
  to = aws_iam_role_policies_exclusive.cluster_autoscaler
  id = "map-cluster-autoscaler"
}
resource "aws_iam_role_policies_exclusive" "cluster_autoscaler" {
  role_name    = aws_iam_role.cluster_autoscaler.name
  policy_names = [aws_iam_role_policy.cluster_autoscaler.name]
}
