# The pod network, and the one setting that decides whether a NetworkPolicy in this
# cluster is a filter or a comment.
#
# WHY THIS FILE EXISTS AT ALL. `deploy/k8s/network-policies.yaml` is a complete
# default-deny policy set for `map-dev`. The API server accepts every object in it today,
# `kubectl get netpol` lists them, and not one packet is examined -- because the agent
# that would enforce them is deployed and switched off. Read off the live cluster: the
# `aws-eks-nodeagent` container of `kube-system/aws-node` runs with
# `--enable-network-policy=false`. `enableNetworkPolicy` below is what turns that into
# `true`. Nothing else in this repository can: the flag is an argument on a DaemonSet the
# cluster installs for itself, and editing that DaemonSet by hand is a change the next
# add-on reconciliation reverts.
#
# CREATED, NOT ADOPTED, and this is the one resource in this directory where that is true
# of a thing that is already running. The CNI is here -- two `aws-node` pods, image
# `amazon-k8s-cni:v1.22.4-eksbuild.3` -- but it is not an add-on: `aws eks list-addons
# --cluster-name map-dev` returns `aws-efs-csi-driver` and nothing else, and the
# DaemonSet's own labels say `app.kubernetes.io/managed-by: Helm`, which is how EKS
# installs the default CNI for a cluster created without the add-on. So there is no
# add-on to import; there is a self-managed installation to take over. An `import` block
# here would fail the whole plan with `Cannot import non-existent remote object`, which is
# why `vpc_cni.tf` is named in `_CREATED_NOT_IMPORTED` in
# `tests/deploy/test_terraform_declares_the_account.py`.
#
# WHAT APPLYING THIS DOES TO A LIVE CLUSTER. Creating the add-on hands the existing
# DaemonSet to EKS and rewrites it with the flag flipped, which rolls every `aws-node` pod
# -- one per node. While a node's `aws-node` is restarting, the CNI daemon on that node
# cannot service an IPAM request, so a pod that is being created there waits; already
# running pods keep their addresses and their connections, because the datapath is routes
# and eBPF programs in the kernel rather than anything inside that pod. Sessions in flight
# survive it; a Session being placed in that window is slow to start and, if the wait
# outlasts the placer's budget, fails to start. That makes this a change to apply when the
# platform is quiet, not one to slip in beside a deploy.
#
# It is also the moment the policy set starts filtering. The order that does not take the
# platform down is: this, first and alone, verified with the DaemonSet's args; then the
# policy set, whose own header says why its blanket deny is the last document in it.
#
# THE FASTEST WAY BACK. If the policies turn out to be wrong and the platform breaks,
# delete the policy set -- `kubectl delete -f deploy/k8s/network-policies.yaml` -- and
# every pod is unfiltered again within the eBPF agent's reconcile, with no rollout and no
# add-on change. A pod that no policy selects is unrestricted, so removing the policies is
# the whole of the undo; turning enforcement back off is a second `aws-node` roll and is
# not the quick lever.

resource "aws_eks_addon" "vpc_cni" {
  cluster_name = aws_eks_cluster.map_dev.name
  addon_name   = "vpc-cni"

  # Exactly the version already running, read from the image tag on the live DaemonSet,
  # and also the default EKS offers for Kubernetes 1.31 (`aws eks describe-addon-versions
  # --addon-name vpc-cni --kubernetes-version 1.31`). Pinned to what runs so that this
  # apply changes the flag and not the CNI: an upgrade to v1.23.0 is a separate decision
  # with its own blast radius, and bundling it here would mean a network outage and a
  # version bump arriving as one event with one rollback.
  addon_version = "v1.22.4-eksbuild.3"

  # OVERWRITE on create is what makes adoption possible rather than a conflict: every
  # object the add-on owns -- the DaemonSet, its ConfigMap, the ServiceAccount, the CRDs --
  # already exists under Helm's management, and without this the create fails naming a
  # resource conflict instead of taking the installation over. On update for the same
  # reason: a `configuration_values` change has to win over the field values the running
  # DaemonSet holds, or the flag is set in the add-on and not in the cluster.
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  # `preserve` decides what a DESTROY does, and it is the difference between removing a
  # declaration and taking the cluster's networking away. Without it, destroying this
  # resource deletes the CNI DaemonSet: no node can then assign an address to a new pod,
  # and nothing schedulable starts anywhere. With it, the Kubernetes objects stay and only
  # EKS's management of them ends -- which is the state this cluster is in today and is a
  # state it can live in.
  preserve = true

  # A string, not a boolean, and the API is what says so: the add-on's configuration schema
  # types `enableNetworkPolicy` as `{"type": "string", "format": "boolean"}` (read from
  # `aws eks describe-addon-configuration --addon-name vpc-cni --addon-version
  # v1.22.4-eksbuild.3`). `jsonencode({ enableNetworkPolicy = true })` renders an unquoted
  # `true` and is rejected against that schema.
  #
  # One key, deliberately. Everything the add-on is not told is rendered from the chart's
  # defaults, and today's DaemonSet is that default set -- so naming a second field here
  # would be this file taking ownership of a value nobody decided. The corollary is worth
  # acting on rather than trusting: diff the DaemonSet after the first apply. Any field
  # the add-on renders differently from what runs today is a change this configuration did
  # not ask for and cannot see.
  configuration_values = jsonencode({
    enableNetworkPolicy = "true"
  })

  # A planned destroy of this becomes exit 1 with the plan still printed, rather than an
  # apply that takes the pod network with it. `preserve` above already keeps the Kubernetes
  # objects in that case; this is the second layer, and it is here because the two protect
  # against different mistakes -- `preserve` against the destroy succeeding harmlessly,
  # this against the destroy being planned at all.
  lifecycle {
    prevent_destroy = true
  }
}
