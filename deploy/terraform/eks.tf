# The cluster and its one nodegroup, as they exist. Both are adopted, not built.
#
# `bootstrap_cluster_creator_admin_permissions` is here because leaving it out
# plans to DESTROY AND RECREATE the cluster. Measured: an import whose
# `access_config` names only `authentication_mode` renders
# `- bootstrap_cluster_creator_admin_permissions = true -> null # forces
# replacement` and `Plan: 1 to add, 1 to destroy`. The argument only ever applies
# at creation and cannot be changed afterwards, so it looks omissible and is not.
# It is the single most dangerous line in this directory.
import {
  to = aws_eks_cluster.map_dev
  id = "map-dev"
}
resource "aws_eks_cluster" "map_dev" {
  name     = "map-dev"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.31"

  access_config {
    authentication_mode                         = "CONFIG_MAP"
    bootstrap_cluster_creator_admin_permissions = true
  }

  # Four subnets, and the two public ones are not an accident: the control plane
  # places an ENI in each, and the cluster was created with all four. Declaring
  # only the two private ones would plan a vpc_config change.
  vpc_config {
    subnet_ids = [
      aws_subnet.private_1a.id,
      aws_subnet.private_1b.id,
      aws_default_subnet.public_1a.id,
      aws_default_subnet.public_1b.id,
    ]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }

  upgrade_policy {
    support_type = "EXTENDED"
  }

  tags = { Name = "map-dev" }

  # The attachment is a separate resource, so without this the destroy order can
  # strip the role's policy before the cluster is gone and leave it undeletable.
  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policy]

  lifecycle {
    prevent_destroy = true
  }
}

# The cluster's only nodegroup. A `t3.medium` group called `map-dev-nodes` ran it until
# 2026-08-26, when this one was created ALONGSIDE it, both its nodes were cordoned and
# drained onto this one, and it was removed in a separate apply. Burst credits are why
# it went: a `t3.medium` that exhausts them is throttled with nothing in the cluster
# reporting it, which reads to a tenant as a slow model and to an operator as nothing.
#
# WHY THAT MIGRATION WAS ADDITIVE, and it is written here because the next machine
# change faces the same choice. Editing `instance_types` or `disk_size` in place does
# not edit a nodegroup: both force `must be replaced`, measured from a drift check, and
# a replacement cycles every node in the cluster at once. Session pods are bare --
# `pod_channel.py` places them with no `ownerReferences`, so a Turn's pod belongs to its
# Turn and to nothing else -- which means nothing in Kubernetes reschedules one that a
# node takes down. Every Turn in flight dies. At the time `turn.failed` was appended
# only on the in-process request path, so each of those Sessions was left with an open
# Turn refusing its own next Turn and refusing its archive: the replacement would not
# have cost a restart, it would have cost a Session per Turn in flight, permanently.
#
# `AbandonedTurnSweeper` now closes exactly that state, so the permanent half of that
# cost is gone. The rest is not -- a replacement still kills every Turn in flight, and
# adding a group and draining the old one still costs only a window of double capacity.
# Do it the same way next time.
#
# Why this machine. A Session pod declares 200m/512Mi for the runtime and 50m/128Mi
# for the shim, so a node gives up 250m and 640Mi per Turn -- and against that, memory
# binds long before the ENI-derived pod slot cap does. Measured on the two live nodes:
# 17 slots each, of which 9-10 are already held by four DaemonSet pods per node
# (`aws-node`, `kube-proxy`, `efs-csi-node`, the seccomp installer) plus the platform's
# own ReplicaSets. A `t3.medium` allocates ~3.28 GiB, which is five or six Session
# pods; an `m6i.xlarge` allocates ~14 GiB, which is about twenty-two. The slots were
# never the ceiling. The memory underneath them was, and it was being spent by pods
# that did not declare it.
#
# And `t3` is burstable, which is the part that had to change rather than merely grow:
# its two vCPU are earned at a 20% baseline and spent from a credit balance, so a node
# full of agents running builds drains that balance and is then throttled to baseline
# with nothing in the cluster reporting it. It reads to a tenant as a slow model and to
# an operator as nothing at all. `m6i` has no credit balance to exhaust, so contention
# degrades gradually instead of off a step nobody measures. ADR-040.
#
# `disk_size` is 80 GiB against the 40 the older group carries, and the number is what
# a full node could claim rather than what one pod uses. Every Session pod's scratch
# volumes are bounded in `session-pod.yaml` at 1297Mi in the worst case, and twenty-two
# of those is 27.9 GiB, against the ~21.5 GiB seventeen could claim on a 40 GiB node.
# The rest is not slack: it is images, and this cluster shows why the margin is wanted
# -- the two live nodes each hold ~14 GB of cached images against ~40 GB of ephemeral
# storage. `DiskPressure` is False today, and kubelet image GC evicting the Session
# image is what would silently turn every warm placement back into a 552 MB pull.
#
# The import block below, for a group Terraform itself created. Every resource in this
# file is import-then-resource and two tests grade that -- a resource with no import
# block is a plan to create a duplicate, and removing one turned a real plan from
# `36 to import, 0 to add` into `35 to import, 1 to add`. This group was created by an
# apply rather than adopted, so it was the one resource here with nothing in front of
# it. An import block naming a resource already in state is a no-op to Terraform, and
# it is what makes the next person's `terraform init` on a fresh state find this group
# instead of proposing a second one beside it.
import {
  to = aws_eks_node_group.map_dev_nodes_m6i
  id = "map-dev:map-dev-nodes-m6i"
}
resource "aws_eks_node_group" "map_dev_nodes_m6i" {
  cluster_name    = aws_eks_cluster.map_dev.name
  node_group_name = "map-dev-nodes-m6i"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = [aws_subnet.private_1a.id, aws_subnet.private_1b.id]
  version         = "1.31"

  ami_type       = "AL2023_x86_64_STANDARD"
  capacity_type  = "ON_DEMAND"
  disk_size      = 80
  instance_types = ["m6i.xlarge"]

  # Not decoration: five slices select Session pods onto `map.role: session-pod`, so a
  # node without this label takes no Session pod at all. Declared on the group rather
  # than applied with `kubectl label nodes`, because a node replacement wipes the latter
  # and the symptom is Session pods Pending for ever with nothing on the pod saying why.
  labels = { "map.role" = "session-pod" }

  # No `tags` for autoscaler discovery, and that is not an omission. EKS tags a managed
  # nodegroup's ASG itself -- verified on the group this one replaced, whose declared
  # tags were `{"Name": "map-dev-nodes"}` alone while its ASG carried both
  # `k8s.io/cluster-autoscaler/enabled=true` and `k8s.io/cluster-autoscaler/map-dev=owned`,
  # which is what `--node-group-auto-discovery` matches on.

  # `max_size` here is not the operative ceiling: `cluster-autoscaler.yaml` passes
  # `--max-nodes-total=8`, which is cluster-wide and binds first. Raising this without
  # raising that buys nothing. It is 8 rather than equal to `min_size` because a group
  # whose ceiling equals its floor cannot be grown at all, by an autoscaler or by hand.
  #
  # The EC2 on-demand vCPU quota is the thing that would bind, and this account's IAM
  # cannot read it: `servicequotas:GetServiceQuota` is denied to `map-dev-agent`. 8 x 4
  # vCPU = 32 sits above what the two-group transition asked for and is the number to
  # check first if a scale-out ever fails to launch -- a quota that binds fails loudly
  # rather than degrading anything quietly, and this comment is why.
  scaling_config {
    min_size     = 2
    max_size     = 8
    desired_size = 2
  }

  # `desired_size` is the autoscaler's at runtime, not this file's, and `ignore_changes`
  # is what says so. Cluster Autoscaler scales by calling SetDesiredCapacity, which moves
  # the number this resource also declares -- so without this line the next apply, even a
  # routine one for an unrelated resource, plans the count back to 2 and drains whatever
  # had been added, with the Sessions on it. That plan IS the outage. Measured on the
  # group this one replaced: with the line, `No changes`; without it, `~ desired_size =
  # 3 -> 2` and `Plan: 0 to add, 1 to change, 0 to destroy`.
  #
  # One attribute wide, deliberately. `all`, or a second entry, would stop the drift gate
  # reporting an instance type, a label or a subnet somebody changed by hand, which is
  # the drift ADR-021 exists to catch. `prevent_destroy` earns its place separately: it
  # turns an accidental ForceNew into a plan error instead of a replacement, and a
  # replacement here cycles every node at once.
  lifecycle {
    prevent_destroy = true
    ignore_changes  = [scaling_config[0].desired_size]
  }
}
