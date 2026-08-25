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

# `instance_types` is two `t3.medium`, which is what runs. `deploy/spike/nodegroup.yaml`
# asked for one `m6i.xlarge` and never matched the account; this slice deletes that
# file rather than encoding its wish here. Declaring m6i.xlarge was measured and it
# plans `must be replaced` -- a nodegroup replacement, from a drift check. What is
# still undecided is whether the machine should change; that is a capacity question
# with a cost, and ADR-021 leaves it open. Reality is declared so that the NEXT
# change to it is visible; see MAP-60's blockers.
import {
  to = aws_eks_node_group.map_dev_nodes
  id = "map-dev:map-dev-nodes"
}
resource "aws_eks_node_group" "map_dev_nodes" {
  cluster_name    = aws_eks_cluster.map_dev.name
  node_group_name = "map-dev-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = [aws_subnet.private_1a.id, aws_subnet.private_1b.id]
  version         = "1.31"

  ami_type       = "AL2023_x86_64_STANDARD"
  capacity_type  = "ON_DEMAND"
  disk_size      = 40
  instance_types = ["t3.medium"]

  # Declared, and this is an adoption rather than a wish: describe-nodegroup
  # reports {"map.role": "session-pod"} since 2026-08-22, when it was set with
  # update-nodegroup-config after this file was written. Before that the label
  # existed only on the two Node objects, applied with `kubectl label nodes`,
  # which a node replacement wipes -- and five slices select on it, so a
  # replacement node meant Session pods Pending for ever with nothing on the
  # pod to say why. EKS applies a label change in place: the update was one
  # ConfigUpdate and both instance ids survived it.
  labels = { "map.role" = "session-pod" }

  # `max_size` is the ceiling the group may be grown to, not a number of instances
  # that run: `desired_size` is what runs. This said "nothing in this cluster
  # changes it today -- there is no Cluster Autoscaler and no Karpenter", and that
  # is no longer true. A Cluster Autoscaler runs in `map-dev` and acts on this
  # group: measured 2026-08-23, twenty-four Sessions submitted at once took it from
  # two nodes to four, and its own log reports `Scale up in group
  # eks-map-dev-nodes-... finished successfully in 1m0.2s`.
  #
  # So the ceiling here is no longer free, and it is also no longer the operative
  # one: `cluster-autoscaler.yaml` passes `--max-nodes-total=4`, which binds first.
  # Raising this without raising that buys nothing.
  #
  # It was raised because while it equalled `min_size` the group could not be grown
  # at all, by an autoscaler or by hand, and 2 x t3.medium is 34 pod slots of which
  # 8 are held by daemons -- about 22 concurrent Session pods, with the 23rd Pending
  # for ever because no node could arrive. Measured on four nodes: 24 concurrent
  # Sessions all completed.
  #
  # 8 rather than a larger number for two reasons, one measured and one not. The
  # two private subnets hold 4079 and 4078 free addresses, so the VPC CNI is
  # nowhere near binding and did not choose this. The EC2 on-demand vCPU quota
  # would, and this account's IAM cannot read it -- `servicequotas:GetServiceQuota`
  # is denied to `map-dev-agent` -- so 8 x 2 vCPU = 16 sits under the smallest
  # quota any current account is likely to carry rather than being argued from a
  # reading. If a scale-out ever fails to launch, the quota is the first thing to
  # check and this comment is why.
  scaling_config {
    min_size     = 2
    max_size     = 8
    desired_size = 2
  }

  update_config {
    max_unavailable = 1
  }

  tags = { Name = "map-dev-nodes" }

  # `desired_size` is the autoscaler's, not this file's, and `ignore_changes` is
  # what says so. Cluster Autoscaler scales by calling SetDesiredCapacity on the
  # group, which moves the number this resource also declares -- so without this
  # line the next apply, including a routine one for an unrelated resource, plans
  # the count back to 2 and drains whatever had been added, with the Sessions on
  # it. Measured through tools/terraform_drift.py against the real account while a
  # third node the autoscaler had just added was running: with this line the gate
  # answers `No changes` and exits 0; with the line removed and nothing else
  # changed it renders `~ desired_size = 3 -> 2`, `Plan: 0 to add, 1 to change,
  # 0 to destroy`, and exits 2. That plan IS the outage -- 3 -> 2 is the node
  # going away -- so the line is not tidiness, it is the thing standing in front
  # of it.
  #
  # One attribute wide, deliberately. `all`, or a second entry, would stop the
  # gate reporting an instance type, a label or a subnet somebody changed by
  # hand, and that is the drift ADR-021 exists to catch. min_size and max_size
  # stay compared: the ceiling is still declared here, and still argued above.
  #
  # prevent_destroy stays for its own reason -- it turns an accidental ForceNew
  # into a plan error instead of a replacement, and its silence in every plan
  # rendered against this change is a second confirmation nothing is replaced.
  lifecycle {
    prevent_destroy = true
    ignore_changes  = [scaling_config[0].desired_size]
  }
}
