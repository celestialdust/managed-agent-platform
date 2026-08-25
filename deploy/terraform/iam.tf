import {
  to = aws_iam_role.eks_cluster
  id = "map-dev-eks-cluster"
}
resource "aws_iam_role" "eks_cluster" {
  name = "map-dev-eks-cluster"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

import {
  to = aws_iam_role_policy_attachment.eks_cluster_policy
  id = "map-dev-eks-cluster/arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}
resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# The authoritative pair for this role: the attached set and the inline set.
# Between them they say "these and no others", which is the claim the individual
# attachment above cannot make. `policy_names = []` is not a placeholder -- it is
# the assertion that this role carries no inline policy, and this identity holds
# iam:PutRolePolicy, so an inline policy is a thing somebody here can add.
import {
  to = aws_iam_role_policy_attachments_exclusive.eks_cluster
  id = "map-dev-eks-cluster"
}
resource "aws_iam_role_policy_attachments_exclusive" "eks_cluster" {
  role_name   = aws_iam_role.eks_cluster.name
  policy_arns = [aws_iam_role_policy_attachment.eks_cluster_policy.policy_arn]
}

import {
  to = aws_iam_role_policies_exclusive.eks_cluster
  id = "map-dev-eks-cluster"
}
resource "aws_iam_role_policies_exclusive" "eks_cluster" {
  role_name    = aws_iam_role.eks_cluster.name
  policy_names = []
}

import {
  to = aws_iam_role.eks_node
  id = "map-dev-eks-node"
}
resource "aws_iam_role" "eks_node" {
  name = "map-dev-eks-node"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# The three attachments this role carries, each its own resource.
#
# What that does NOT do, stated because an earlier version of this comment claimed
# it did: naming three attachments does not make a FOURTH show up as a diff.
# `aws_iam_role_policy_attachment` is non-authoritative and per-attachment --
# terraform refreshes the attachments that are in state and never asks the role
# what it actually carries -- so an attachment made outside this repository is
# invisible to the plan at every exit code. Measured: with
# `AmazonEC2ContainerRegistryReadOnly` attached in the account and its declaration
# deleted, the plan still reported `0 to add, 3 to change, 0 to destroy` and
# proposed no diff for it. The arn does appear once in the plan TEXT, inside the
# role's deprecated computed `managed_policy_arns` list, which an import plan
# renders in full -- so it is visible to nobody who reads an exit code, and it is
# not rendered at all once the resources are in state.
#
# `aws_iam_role_policy_attachments_exclusive` below is what closes it, and the
# hazard is why the closing is worth two extra resources: the EKS S3 CSI driver's
# documented install attaches AmazonS3ReadOnlyAccess to this role, which is read
# access to every bucket in the account -- this configuration's own state bucket
# included -- from any node, and so from any Session pod on one.
import {
  to = aws_iam_role_policy_attachment.node_worker
  id = "map-dev-eks-node/arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}
resource "aws_iam_role_policy_attachment" "node_worker" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

import {
  to = aws_iam_role_policy_attachment.node_cni
  id = "map-dev-eks-node/arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}
resource "aws_iam_role_policy_attachment" "node_cni" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

import {
  to = aws_iam_role_policy_attachment.node_ecr_read
  id = "map-dev-eks-node/arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}
resource "aws_iam_role_policy_attachment" "node_ecr_read" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# The set, declared as a set. A fourth attachment made at a shell -- the S3 CSI
# driver install above being the one with a name -- now renders inside this
# resource as `- "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"`, a MINUS
# because terraform's proposal is to detach what the declared set does not name,
# and the plan is non-empty so the gate exits 2. That is the guarantee the three
# resources above were wrongly credited with. Measured in that exact shape:
# dropping one arn from the list below rendered
# `- "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"` and took the
# plan's change count from 3 to 4.
#
# The arns are read off the attachment resources rather than repeated, so the two
# declarations of one fact cannot drift apart: adding an attachment without adding
# it here is a plan to detach it again, which is loud.
import {
  to = aws_iam_role_policy_attachments_exclusive.eks_node
  id = "map-dev-eks-node"
}
resource "aws_iam_role_policy_attachments_exclusive" "eks_node" {
  role_name = aws_iam_role.eks_node.name
  policy_arns = [
    aws_iam_role_policy_attachment.node_worker.policy_arn,
    aws_iam_role_policy_attachment.node_cni.policy_arn,
    aws_iam_role_policy_attachment.node_ecr_read.policy_arn,
  ]
}

import {
  to = aws_iam_role_policies_exclusive.eks_node
  id = "map-dev-eks-node"
}
resource "aws_iam_role_policies_exclusive" "eks_node" {
  role_name    = aws_iam_role.eks_node.name
  policy_names = []
}

# IRSA. The whole pod-holds-no-credential invariant rests on this provider
# existing, and `thumbprint_list` is a fact with an expiry date -- when the
# issuer's CA rolls, this value stops matching and the gate is where that shows up
# rather than a Session failing to assume a role.
import {
  to = aws_iam_openid_connect_provider.cluster
  id = "arn:aws:iam::062677866851:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/F672DDC4DEB48BB78E47544C237E5B77"
}
resource "aws_iam_openid_connect_provider" "cluster" {
  url             = "https://oidc.eks.us-east-1.amazonaws.com/id/F672DDC4DEB48BB78E47544C237E5B77"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["2ad974a775f73cbdbbd8f5ac3a49255fa8fb1f8c"]
  lifecycle {
    prevent_destroy = true
  }
}

# The policy this whole environment is provisioned through. deploy/iam/ held a copy
# of it and nothing compared the two; the account's copy is at version 5 and the
# repository recorded none of those edits. `file()` rather than a heredoc so there
# is one JSON, readable by `aws accessanalyzer validate-policy` and by this.
#
# Note what this can and cannot do. It DETECTS a change: an edit on either side
# makes the gate red. It cannot apply one -- the attached policy grants
# iam:GetPolicy and iam:GetPolicyVersion but not iam:CreatePolicyVersion, so
# reconciling in the account's direction needs the owner. That asymmetry is the
# correct one for a policy that governs the identity Terraform itself runs as.
import {
  to = aws_iam_policy.provisioning
  id = "arn:aws:iam::062677866851:policy/map-dev-provisioning"
}
resource "aws_iam_policy" "provisioning" {
  name   = "map-dev-provisioning"
  policy = file("${path.module}/../iam/map-dev-provisioning.json")
  lifecycle {
    prevent_destroy = true
  }
}
