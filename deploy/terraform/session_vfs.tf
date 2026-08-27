# The Session VFS: one S3 Files file system over the platform bucket, reachable from both
# node subnets, mounted by the EFS CSI driver.
#
# Why NFS and not FUSE: every FUSE client needs /dev/fuse handed into the container plus a
# userspace daemon alive for the mount's whole life, and ADR-006 binds the mount to exist at
# sandbox construction -- before the runtime's first instruction. An NFS-protocol mount is
# established by the kubelet, so the ordering is the container runtime's rather than a claim.
#
# One file system, not two: the artifacts subtree and the evidence subtree are the same
# device on purpose. The readiness check that runs before the Agent Runtime starts compares
# st_dev to tell a real mount from a directory somebody created, and it expects those two to
# match each other and to differ from pod-local scratch. Two file systems would be two
# superblocks and the check would read them as two mounts.
#
# EVERY RESOURCE HERE IS CREATED, NOT IMPORTED, and that is why this file carries no
# `import` blocks while the other nine in this directory carry one per resource.
#
# As of 2026-08-23, **19 of these 24 objects now exist** -- the two CSI roles with all their
# policies and exclusive sets, the mount security group and its ingress rule, the EFS CSI
# add-on, and the service role with its policy. The header used to say "nothing below
# exists in the account yet"; that stopped being true the moment the first apply ran, and a
# file that describes an empty account while nineteen of its resources are live is a file
# that will be believed by the next person to read it.
#
# The remaining five -- the file system, both mount targets, the access point and the
# synchronisation configuration -- CANNOT be created by this configuration, and the reason
# is not in this file. S3 Files builds its `DO-NOT-DELETE-S3-Files*` EventBridge rules AS
# THE CALLER, and `map-dev-provisioning` grants `s3files:*` with no `events:*` at all. A
# file system created with a fully correct role and trust policy sits in `creating` for
# about five and a half minutes with `statusMessage` = "Access denied: User:
# .../user/map-dev-agent is not authorized to perform: events:ListRules" and then goes to
# `error`. The denial names the USER, not the role -- the role holds every documented
# EventBridge permission and it changes nothing. Someone with IAM admin has to add the
# `events:*` actions to `map-dev-provisioning`; `iam:PutUserPolicy` and
# `iam:AttachUserPolicy` are explicit denies for the agent identity.
#
# EVERY ID BELOW IS READ OFF THE RESOURCE THAT OWNS IT, never written out again. irsa.tf
# states the rule for the cluster's issuer id; it holds for all of them, and the first draft
# of this file broke it six times -- the bucket arn, the cluster name, the vpc id, the node
# subnet ids, the node security group and the OIDC provider arn were all literal defaults
# beside the resources that already held them. A literal costs more here than untidiness,
# because nothing in a `terraform plan` compares two copies of a value:
#
#   - `bucket` as a literal made the whole file retargetable by one edit. Pointed at
#     map-dev-tfstate-..., every Session would have mounted Terraform's own state, and the
#     service role below grants PutObject/DeleteObject/DeleteObjectVersion on it.
#   - the subnet ids, the security group and the issuer id as literals made this file the
#     stale copy: re-CIDR a subnet in network.tf, or roll the cluster's issuer in iam.tf, and
#     the nodegroup follows while the mount targets and the CSI trust policies stay behind --
#     nodes in an AZ with no mount target, roles trusting a dead issuer, no diff anywhere.

# Null -- the only value a real plan uses -- means the nodegroup's own subnets, read off the
# nodegroup in `locals` below. The override exists so a test can plan a deliberately same-AZ
# pair and prove the precondition refuses it; there is no other reason to pass it.
variable "node_subnet_ids" {
  description = "Override the subnets one mount target is created in. Null means the nodegroup's own."
  type        = list(string)
  default     = null

  validation {
    # S3 Files allows one mount target per file system per Availability Zone, so two
    # subnets in one AZ is not a redundancy win -- it is an apply-time failure. This half
    # (the same subnet twice) is decidable without describing anything, so it is caught at
    # variable-parse time where the message can name the variable. The other half (two
    # distinct subnets that share an AZ) needs the subnets described and so waits for the
    # mount target's precondition.
    condition = (
      var.node_subnet_ids == null ||
      length(var.node_subnet_ids) == length(distinct(var.node_subnet_ids))
    )
    error_message = "node_subnet_ids must not repeat a subnet."
  }
}

locals {
  # The Session pod runs as this uid/gid. The access point forces every file operation to
  # them, which is what makes the mount's st_uid/st_gid match pod-local scratch's -- the
  # readiness check compares owner across the subtrees and refuses a tree that disagrees.
  session_uid = 10001
  session_gid = 10001

  # The nodegroup's own subnets, unless a caller overrode them. This cannot be the
  # variable's `default`: Terraform requires a default to be a constant, so a reference has
  # to live in a local and the default has to be null.
  node_subnet_ids = coalesce(
    var.node_subnet_ids, tolist(aws_eks_node_group.map_dev_nodes_m6i.subnet_ids)
  )

  # EKS attaches the cluster security group to every managed-nodegroup ENI, so this is what
  # "from the nodes" means for an ingress rule.
  node_security_group_id = aws_eks_cluster.map_dev.vpc_config[0].cluster_security_group_id
}

data "aws_subnet" "session_vfs_node" {
  for_each = toset(local.node_subnet_ids)
  id       = each.value
}

# ---------------------------------------------------------------------------
# Network reachability: NFS from the nodes, and from nothing else.
# ---------------------------------------------------------------------------

resource "aws_security_group" "session_vfs_mount" {
  name        = "map-dev-session-vfs-mount"
  description = "Mount-target ENIs for the Session VFS file system"
  vpc_id      = aws_default_vpc.map.id

  tags = {
    Name = "map-dev-session-vfs-mount"
  }
}

# No egress rule, deliberately. Security groups are stateful, so replies on a connection
# admitted by the rule below need no matching egress; an allow-all egress would only widen
# what a compromised mount-target ENI could originate.
#
# The two ids are the two ENDS of one rule and they are not interchangeable:
# `security_group_id` is the group the rule is written into, `referenced_security_group_id`
# is who may come in. Swapping them puts inbound 2049 on the cluster security group --
# widening what every node ENI accepts -- and leaves the mount-target group with no ingress
# at all, so nothing can reach the mount. A plan renders either arrangement as a clean
# create, which is why both ends are asserted rather than just the source.
resource "aws_vpc_security_group_ingress_rule" "session_vfs_nfs_from_nodes" {
  security_group_id            = aws_security_group.session_vfs_mount.id
  description                  = "NFSv4.2 from the nodegroup only"
  ip_protocol                  = "tcp"
  from_port                    = 2049
  to_port                      = 2049
  referenced_security_group_id = local.node_security_group_id
}

# ---------------------------------------------------------------------------
# The file system, and the role S3 Files itself uses to reach the bucket.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "session_vfs_service" {
  name        = "map-dev-session-vfs-service"
  description = "Assumed by S3 Files to import from and export to the VFS bucket"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      # S3 Files is EFS underneath, and it assumes roles as the EFS principal. The
      # `s3files` spelling belongs to the ARN NAMESPACE only -- it appears in
      # aws:SourceArn below and nowhere else -- so guessing the principal from the
      # resource type gives `s3files.amazonaws.com`, which IAM refuses outright with
      # MalformedPolicyDocument and no role is created at all.
      #
      # Measured, not guessed, and the measurement is a CONTROL PAIR because the two
      # failures look nothing alike and only one of them is about the principal: with
      # this principal a file system gets PAST the assume-role check and stops later,
      # on the caller's own missing events:ListRules; with `s3.amazonaws.com` -- and
      # with every other spelling IAM accepts -- it stops AT the assume check saying
      # "S3 Files does not have permissions to assume the provided role". The assume
      # check runs first, so reaching a later error is what proves this string right.
      Principal = { Service = "elasticfilesystem.amazonaws.com" }
      Action    = "sts:AssumeRole"
      # Both conditions are the confused-deputy guard and neither is decoration. The
      # principal is an AWS SERVICE, so with no condition the trust policy says "any
      # file system in any account may assume this role" -- and a role that can write
      # this bucket is exactly what a stranger's file system would want.
      # aws:SourceAccount pins the calling account; aws:SourceArn additionally pins the
      # caller to a FILE SYSTEM, which matters precisely because the principal is
      # shared -- EFS proper assumes as this same string, so without it any other
      # elasticfilesystem-backed resource in this account fits the policy too.
      # The account is a literal because this whole directory is single-account by
      # construction (every import id names this account), not because it could be
      # derived and was not.
      Condition = {
        StringEquals = { "aws:SourceAccount" = local.account_id }
        ArnLike      = { "aws:SourceArn" = "arn:aws:s3files:us-east-1:${local.account_id}:file-system/*" }
      }
    }]
  })
}

resource "aws_iam_role_policy" "session_vfs_service_bucket" {
  name = "vfs-bucket-synchronisation"
  role = aws_iam_role.session_vfs_service.id

  # The wildcards are the documented form and they are not laziness: S3 Files requires
  # bucket versioning and drives it with the version-specific twins of each call, so
  # `s3:GetObject*` stands for GetObject AND GetObjectVersion AND GetObjectVersionTagging
  # -- the tagging one having no non-versioned spelling to enumerate. The previous
  # enumeration here named five actions and silently lacked object tagging and
  # AbortMultipartUpload, which is a half-working outcome rather than an error: the file
  # system comes up and individual syncs fail later.
  #
  # aws:ResourceAccount narrows what the wildcards widen. The actions are broad, so the
  # condition is what keeps this role from being usable against a bucket in any other
  # account should its Resource ever be loosened.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:GetBucketVersioning",
          "s3:ListBucket",
          "s3:ListBucketVersions",
        ]
        Resource  = aws_s3_bucket.platform.arn
        Condition = { StringEquals = { "aws:ResourceAccount" = local.account_id } }
      },
      {
        Effect = "Allow"
        Action = [
          "s3:AbortMultipartUpload",
          "s3:DeleteObject*",
          "s3:GetObject*",
          "s3:List*",
          "s3:PutObject*",
        ]
        Resource  = "${aws_s3_bucket.platform.arn}/*"
        Condition = { StringEquals = { "aws:ResourceAccount" = local.account_id } }
      },
      # S3 Files watches the bucket through EventBridge rules that IT creates, under a
      # reserved `DO-NOT-DELETE-S3-Files*` name, and this role is what creates them. With
      # these two statements absent the file system still comes up and bucket-side changes
      # simply never reach it -- no error anywhere, which is why they are written here
      # rather than discovered later. events:ManagedBy scopes the mutating half to rules
      # the service itself owns, so this cannot touch a rule anything else in the account
      # created. The read half is unconditioned because that is the documented policy; if
      # it proves insufficient the symptom is a sync that never starts, not a wrong grant.
      {
        Effect = "Allow"
        Action = [
          "events:DeleteRule",
          "events:DisableRule",
          "events:EnableRule",
          "events:PutRule",
          "events:PutTargets",
          "events:RemoveTargets",
        ]
        Resource  = "arn:aws:events:*:*:rule/DO-NOT-DELETE-S3-Files*"
        Condition = { StringEquals = { "events:ManagedBy" = "elasticfilesystem.amazonaws.com" } }
      },
      {
        Effect = "Allow"
        Action = [
          "events:DescribeRule",
          "events:ListTargetsByRule",
        ]
        Resource = "arn:aws:events:*:*:rule/*"
      },
      # Split off the rule-scoped read above, and NOT a copy of the documented policy:
      # these two are list-level operations that authorise against no resource, so the
      # docs' own `rule/*` ARN cannot grant them. Measured with
      # `iam simulate-principal-policy` against this very role: DescribeRule and
      # ListTargetsByRule come back `allowed` on a rule ARN, ListRules and
      # ListRuleNamesByTarget come back `implicitDeny` on the same ARN and need "*".
      # (ListRuleNamesByTarget is scoped by its TARGET, not by a rule, which is why a
      # rule ARN does not reach it either.) Both are read-only and neither has a
      # resource-level form to narrow to, which is the same reason the mount-helper
      # policy below carries its own `Resource = "*"` statements.
      {
        Effect = "Allow"
        Action = [
          "events:ListRuleNamesByTarget",
          "events:ListRules",
        ]
        Resource = "*"
      },
      # NOT here, deliberately: the docs' `UseKmsKeyWithS3Files` statement. It is needed
      # only for an SSE-KMS bucket, and this bucket is SSE-S3 (AES256), so a kms:Decrypt
      # grant would be a permission with nothing to decrypt. It becomes required the day
      # the bucket moves to a CMK, together with the file system's `kms_key_id`.
    ]
  })
}

resource "aws_iam_role_policy_attachments_exclusive" "session_vfs_service" {
  role_name   = aws_iam_role.session_vfs_service.name
  policy_arns = []
}

resource "aws_iam_role_policies_exclusive" "session_vfs_service" {
  role_name    = aws_iam_role.session_vfs_service.name
  policy_names = [aws_iam_role_policy.session_vfs_service_bucket.name]
}

resource "aws_s3files_file_system" "session_vfs" {
  bucket   = aws_s3_bucket.platform.arn
  role_arn = aws_iam_role.session_vfs_service.arn

  tags = {
    Name = "map-dev-session-vfs"
  }

  # CreateFileSystem validates that it can actually reach the bucket AS the role, so the
  # role's inline policy has to exist first; without this terraform is free to create it
  # afterwards and the file system fails on a role that cannot yet read anything. The
  # provider allows for IAM propagation on top of ordering, which is the other half of the
  # same race. (An earlier note here claimed CreateFileSystem WRITES a bucket policy onto
  # the bucket -- it does not, and the docs say the opposite: make sure the bucket policy
  # does not DENY the service. The ordering is still required, for the reason above.)
  depends_on = [aws_iam_role_policy.session_vfs_service_bucket]
}

resource "aws_s3files_mount_target" "session_vfs" {
  for_each = toset(local.node_subnet_ids)

  file_system_id  = aws_s3files_file_system.session_vfs.id
  subnet_id       = each.value
  security_groups = [aws_security_group.session_vfs_mount.id]

  lifecycle {
    # A precondition rather than a check block: a failed check block is a warning and
    # `terraform plan -detailed-exitcode` still reports only "changes present", so the
    # drift gate would go green over a config that cannot apply.
    precondition {
      condition = length(distinct([
        for s in data.aws_subnet.session_vfs_node : s.availability_zone
      ])) == length(local.node_subnet_ids)
      error_message = "Two node subnets share an Availability Zone; S3 Files allows one mount target per AZ."
    }
  }
}

resource "aws_s3files_access_point" "session_vfs" {
  file_system_id = aws_s3files_file_system.session_vfs.id

  posix_user {
    uid = local.session_uid
    gid = local.session_gid
  }

  # The whole bucket, not a subtree. The pod's volumeMount takes the subPath, so a
  # root_directory anywhere but "/" would move every Session's workspace to a prefix the
  # manifests do not name -- the mount would succeed and the tree would be the wrong one.
  root_directory {
    path = "/"
  }

  tags = {
    Name = "map-dev-session-vfs"
  }
}

resource "aws_s3files_synchronization_configuration" "session_vfs" {
  file_system_id = aws_s3files_file_system.session_vfs.id

  # size_less_than = 0 is the docs' own named configuration for an agent that browses a
  # large tree and reads a few files from it: no object data is imported, metadata still is,
  # so a recursive walk is served at low latency from the fast tier while every read is an
  # S3 round trip. The cost is stated rather than hidden -- reads pay S3 latency, so a
  # workload that re-reads the same small file in a loop pays the round trip every time.
  #
  # `prefix` is an S3 KEY prefix, and "" is the documented spelling for the whole bucket --
  # NOT "/", which this said until a file system read its own scope back as `prefix: ""`.
  # A key prefix must end in "/" unless it is the entire-bucket "", and exactly one rule
  # must cover the root, so "/" satisfies neither form: it describes keys beginning with a
  # literal slash, of which this bucket has none, leaving the configuration with no root
  # rule at all. Nothing catches it early -- the provider declares a bare required string
  # with no validator, so `terraform plan` renders it as a clean create and only
  # PutSynchronizationConfiguration ever objects.
  #
  # The "/" on the access point's root_directory below is a POSIX path and IS correct.
  # Two fields spelled the same, two namespaces; do not make them agree.
  import_data_rule {
    prefix         = ""
    size_less_than = 0
    trigger        = "ON_DIRECTORY_FIRST_ACCESS"
  }

  # Nothing the agent writes needs to stay resident past the Session that wrote it; the
  # bucket is the durable copy and a Session is hours, not days. One day is the floor.
  # Deleting this rule is not a no-op: nothing then ages out of the fast tier, so the whole
  # bucket accumulates there and is billed there.
  expiration_data_rule {
    days_after_last_access = 1
  }
}

# ---------------------------------------------------------------------------
# The CSI driver, and its two service-account roles.
# ---------------------------------------------------------------------------

# `local.oidc_issuer` is irsa.tf's, derived from the provider resource. The `:sub` condition
# is the whole access-control decision on these two roles: both carry
# AmazonS3FilesClientFullAccess, which is ClientMount + ClientWrite + ClientRootAccess on
# Resource "*", so widening the subject to a wildcard would let ANY service account in the
# cluster -- a Session pod's included -- mount and write every S3 Files file system in the
# account as root.
resource "aws_iam_role" "s3files_csi_controller" {
  name = "map-dev-s3files-csi-controller"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          "${local.oidc_issuer}:sub" = "system:serviceaccount:kube-system:efs-csi-controller-sa"
        }
      }
    }]
  })
}

# The three the add-on catalogue names for efs-csi-controller-sa, verbatim. Note the
# service-role/ path on two of them: the EKS user guide prints the S3 Files one as
# "AmazonS3FilesCSIDriverPolicy" with no path, and that ARN does not exist.
resource "aws_iam_role_policy_attachment" "s3files_csi_controller" {
  for_each = toset([
    "arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy",
    "arn:aws:iam::aws:policy/service-role/AmazonS3FilesCSIDriverPolicy",
    "arn:aws:iam::aws:policy/AmazonS3FilesClientFullAccess",
  ])

  role       = aws_iam_role.s3files_csi_controller.name
  policy_arn = each.value
}

# Naming three attachments does not make a fourth one visible -- terraform refreshes only
# the attachments already in state, so a policy attached by hand or by the add-on's own
# documented install appears in no plan at any exit code. iam.tf says the same at length.
# The documented install for THIS driver attaches AmazonS3ReadOnlyAccess, which is s3:Get*
# on Resource "*" and includes the Terraform state bucket, so the set is declared.
resource "aws_iam_role_policy_attachments_exclusive" "s3files_csi_controller" {
  role_name = aws_iam_role.s3files_csi_controller.name
  policy_arns = [
    for attachment in aws_iam_role_policy_attachment.s3files_csi_controller :
    attachment.policy_arn
  ]
}

resource "aws_iam_role_policies_exclusive" "s3files_csi_controller" {
  role_name    = aws_iam_role.s3files_csi_controller.name
  policy_names = []
}

resource "aws_iam_role" "s3files_csi_node" {
  name = "map-dev-s3files-csi-node"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.cluster.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
          "${local.oidc_issuer}:sub" = "system:serviceaccount:kube-system:efs-csi-node-sa"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3files_csi_node_client" {
  role       = aws_iam_role.s3files_csi_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FilesClientFullAccess"
}

# Replaces AmazonS3ReadOnlyAccess, which the add-on catalogue recommends here and which is
# s3:Get*/List*/Describe* on Resource "*" -- read access to every bucket in the account,
# including the one holding Terraform state. Scoped to the one bucket the mount reads by
# reference rather than by tag: a tag condition grants access to any future bucket somebody
# tags, which is permissive by default, and a bucket reference is not.
resource "aws_iam_role_policy" "s3files_csi_node_bucket_read" {
  name = "vfs-bucket-read-only"
  role = aws_iam_role.s3files_csi_node.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetBucketLocation", "s3:ListBucket"]
        Resource = aws_s3_bucket.platform.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:GetObjectVersion"]
        Resource = "${aws_s3_bucket.platform.arn}/*"
      },
    ]
  })
}

# Replaces AmazonElasticFileSystemsUtils, which is an EC2 instance-management policy: three
# of its statements are ssm:*, ssmmessages:* and ec2messages:* on Resource "*", which is the
# Session Manager data plane -- remote shell capability a mount helper has no use for. Only
# the mount-time reads and the metric namespace it actually writes are kept.
resource "aws_iam_role_policy" "s3files_csi_node_mount_helper" {
  name = "efs-utils-mount-helper"
  role = aws_iam_role.s3files_csi_node.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "elasticfilesystem:DescribeMountTargets",
          "ec2:DescribeAvailabilityZones",
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "cloudwatch:PutMetricData"
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = ["efs-utils/S3Files", "efs-utils/EFS"]
          }
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachments_exclusive" "s3files_csi_node" {
  role_name   = aws_iam_role.s3files_csi_node.name
  policy_arns = [aws_iam_role_policy_attachment.s3files_csi_node_client.policy_arn]
}

# This identity holds iam:PutRolePolicy, so the inline set is as reachable as the attached
# one and is declared the same way.
resource "aws_iam_role_policies_exclusive" "s3files_csi_node" {
  role_name = aws_iam_role.s3files_csi_node.name
  policy_names = [
    aws_iam_role_policy.s3files_csi_node_bucket_read.name,
    aws_iam_role_policy.s3files_csi_node_mount_helper.name,
  ]
}

resource "aws_eks_addon" "efs_csi" {
  cluster_name  = aws_eks_cluster.map_dev.name
  addon_name    = "aws-efs-csi-driver"
  addon_version = "v3.4.1-eksbuild.1"

  # A bare CSIDriver object named efs.csi.aws.com was hand-applied to this cluster with no
  # driver behind it. Without OVERWRITE the add-on install fails on that object rather than
  # adopting it, and the failure names a resource conflict rather than its cause.
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  configuration_values = jsonencode({
    controller = {
      serviceAccount = {
        annotations = {
          "eks.amazonaws.com/role-arn" = aws_iam_role.s3files_csi_controller.arn
        }
      }
    }
    node = {
      serviceAccount = {
        annotations = {
          "eks.amazonaws.com/role-arn" = aws_iam_role.s3files_csi_node.arn
        }
      }
    }
  })
}

resource "aws_s3files_access_point" "session_workspaces" {
  file_system_id = aws_s3files_file_system.session_vfs.id

  posix_user {
    uid = local.session_uid
    gid = local.session_gid
  }

  # Rooted at a prefix THIS RESOURCE CREATES, and that is the whole reason it exists
  # separately from the access point above.
  #
  # `creation_permissions` is applied only when S3 Files has to create `path`. Point an
  # access point at a prefix the bucket already holds and the field is silently ignored:
  # the directory keeps the ownership it already had, which for every prefix synthesised
  # from existing S3 objects is uid 0 / gid 0, mode 0755. The mount then succeeds and is
  # unwritable, because `posix_user` squashes every operation through it to 10001.
  #
  # That is not a theoretical failure. The kubelet creates a volumeMount's `subPath`
  # directory at mount time, through this access point, as 10001 -- so a root-owned root
  # directory fails the pod at `CreateContainerConfigError` before its first container
  # starts, with `failed to create subPath directory`. `workspaces` does not exist in the
  # bucket, so S3 Files creates it here, owned by the identity that has to write into it,
  # and every per-Session directory below it inherits that ownership by being created by
  # a process the access point has already squashed to 10001.
  #
  # It is also why the workspace does not live under `sessions/`, where ADR-036's
  # per-Session shape would otherwise have put it: that prefix exists and is root-owned.
  # The split it forces turns out to be the one the design wanted anyway -- `sessions/`
  # holds sealed artifacts written through the S3 API, `workspaces/` holds live mutable
  # state written through NFS by the pod. Different writers, different lifecycles.
  root_directory {
    path = "/workspaces"

    creation_permissions {
      owner_uid   = local.session_uid
      owner_gid   = local.session_gid
      permissions = "0755"
    }
  }

  tags = {
    Name = "map-dev-session-workspaces"
  }
}

output "session_vfs_file_system_id" {
  description = "What the PersistentVolume's volumeHandle names."
  value       = aws_s3files_file_system.session_vfs.id
}

output "session_vfs_access_point_id" {
  description = "The access point that forces uid/gid 10001 on every file operation."
  value       = aws_s3files_access_point.session_vfs.id
}

output "session_workspaces_access_point_id" {
  description = "The access point a Session pod's PersistentVolume mounts; its root is `workspaces/`."
  value       = aws_s3files_access_point.session_workspaces.id
}
