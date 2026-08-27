"""GET /v1/capacity -- how much work is waiting for a pod, and how much room is left.

**Why a number and not a log line.** When this platform is under load the symptom is a
Turn that takes a long time, and from outside, *waiting for a node* and *the model is
thinking* are indistinguishable. Nothing else in this API separates them. That has
already cost a wrong conclusion once: three aborted runs left forty-two pods squatting
in the namespace, and the next run's capacity refusal was read as the cluster's ceiling
rather than as litter from our own earlier failures. One queue-depth number settles
which it is immediately, which is why this exists before the load test rather than after
it.

Authorized as the audit surface is -- a platform reviewer, never a tenant header -- and
mounted on its own router so the pair of dependencies sits on the router rather than on
the route. The numbers describe the platform and not one tenant's work, and a tenant
that could read them would learn the fleet's shape: how many other Sessions hold pods,
how close the cluster is to refusing, and whether now is a good moment to submit. None
of that is theirs to know, and none of it is answerable per-tenant without inventing a
tenant to scope it by.

Nothing here writes, and the router declares nothing but GET -- the same property
`audit.py` states for itself, and true here for the same reason: reading how full the
cluster is must not open a path to changing it.

**Two of the six fields have no counterpart in the API this platform mirrors, and they
are the reason for building ours rather than copying theirs.** A hosted API does not own
your cluster; we own this one. Publishing the autoscaler's ceiling beside the
schedulable node count turns a known mismatch into a number a reader can see, instead of
an inference drawn from a refusal. See `nodes_schedulable` and `node_ceiling` below for
what this deployment can and cannot actually answer.
"""

from datetime import UTC, datetime
from typing import Final, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from managed_agent.control.api.request.dependencies import platform_from_request
from managed_agent.control.api.request.reviewer_auth import establish_reviewer_principal
from managed_agent.control.api.routes.audit import platform_reviewer_of
from managed_agent.control.session.placement import CapacityReport, PlacementStats

router = APIRouter(
    tags=["capacity"],
    # The same two, in the same order, as the audit router: the first establishes a
    # reviewer principal from a presented token and establishes nothing otherwise, the
    # second authorizes from the claims on the request alone. Imported from those two
    # modules rather than restated, so this surface cannot come to accept a principal
    # the audit surface refuses -- a second copy of the rule is a second contract, and
    # the first thing it would disagree about is who may read across tenants.
    dependencies=[
        Depends(establish_reviewer_principal),
        Depends(platform_reviewer_of),
    ],
)

PLACEMENT_STATS: Final = "placement_stats"
"""The `type` discriminator on the body, so a consumer branches on a field it reads
rather than on which URL it called."""


class PlacementStatsView(BaseModel):
    """The capacity answer as it goes on the wire.

    A view distinct from `PlacementStats` rather than the dataclass serialized directly,
    because the two differ in the one place that matters to a caller: the domain type
    carries an epoch-millisecond integer and this carries an instant. Rendering is this
    surface's business, and keeping the conversion here means only one module has to
    know what the wire format is.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["placement_stats"] = PLACEMENT_STATS

    turns_awaiting_placement: int
    """How many Turns are between finding no pod and having one.

    Work waiting, and the number the forty-two-squatting-pods incident needed. Counts
    Turns rather than Sessions because two Turns of one Session can wait at once:
    admission refuses a Session that will not take a Turn, not one that already has a
    Turn open, so both reach the dispatch and both find no pod.

    **This replica's share, not the fleet's.** What is waiting right now is held by the
    coroutines that are blocked, so it is per-process by construction, and the serving
    Deployment runs two replicas. A reader comparing this against the pod count is
    comparing one process's queue against the whole cluster's capacity; the queue is the
    half that is short.
    """

    sessions_placing: int
    """How many distinct Sessions those waiting Turns belong to.

    Claimed and unfinished. It differs from the count above exactly when a Session has
    more than one Turn queued behind one pod, and the gap between them is the signal:
    equal numbers mean the wait is spread across Sessions and is a capacity story, while
    a much smaller number here means a few Sessions are each holding several Turns and
    the story is about those Sessions.

    Per-replica for the same reason as the field above.
    """

    oldest_awaiting_placement_at: datetime | None
    """When the Turn that has waited longest and is still waiting began waiting.

    The unluckiest request, as an instant rather than a duration, so it does not go
    stale between being read and being looked at -- a duration is only true at the
    moment it was measured, and an instant stays true.

    `None` means nothing is waiting. Not "we did not measure": an empty queue has no
    oldest member, and a zero or an epoch here would read as a Turn that has been
    waiting since 1970.
    """

    session_pods_running: int
    """How many Turns are being served right now.

    **This number changed meaning without changing name or arithmetic.** It still
    counts Session pods in the running phase, and it is still read the same way. What
    moved is what a running pod is: while a Session held one for its whole life, a
    running pod was a Session that *could* take a Turn, so this read as headroom. A pod
    now exists only for the length of the Turn it carries, so every one of them is
    already carrying one and this reads as utilisation -- the concurrency in flight,
    against the fleet's ceiling rather than under it.

    The name is kept because it is published: a tenant reading this field would not be
    helped by having it disappear, and what it counts is unchanged. Read it beside
    `node_ceiling` and `nodes_schedulable`, which are the supply this demand is measured
    against.

    Still the running phase only, and now for a sharper reason than before. A pod
    that is starting belongs to a Turn that is queued rather than served, and that Turn
    is already counted by `turns_awaiting_placement` -- so counting its pod here would
    count one Turn twice, in two fields an operator reads against each other.

    Cluster-wide rather than per-replica, unlike the two queue numbers, because it is
    read from the cluster: whichever replica answers gives the same count.

    Provenance: ADR-041.
    """

    nodes_schedulable: int | None
    """How many nodes are currently able to take a pod.

    `None` when this deployment's cluster client cannot answer, which is the case today:
    the read is `list_node` on the Kubernetes API and the adapter this platform wires
    does not make it. Null rather than a stand-in, because every plausible stand-in --
    the pod count, the autoscaler's minimum, a constant -- is a different number wearing
    this one's name.
    """

    node_ceiling: int | None
    """The most nodes the autoscaler will add unattended.

    **The field the mirrored API has no room for, and the one worth the most here.** The
    cluster autoscaler runs with `--max-nodes-total=4` while the nodegroup declares
    `max_size = 8`, so the smaller binds and a reader who sees only a refusal cannot
    tell which of the two stopped them. Published beside `nodes_schedulable`, the
    mismatch is visible instead of inferred.

    **Where the number comes from, and why it is `None` today.** Its single declaration
    is the autoscaler's own manifest, `deploy/k8s/cluster-autoscaler.yaml`, which is
    applied to the cluster and is not on the serving container's filesystem -- so a
    file read here would work in the test suite and answer nothing in the cluster, which
    is the worse of the two failure modes because it looks correct locally. The only
    runtime-trustworthy source is the argument list of the running autoscaler
    Deployment, read through the cluster API, and that read belongs beside the other
    cluster reads in the Kubernetes adapter, which does not have it yet.

    So this is null until that adapter answers, deliberately and not by oversight. A
    field that silently drifts from the flag it claims to publish is worse than a field
    that says it does not know, because a drifted ceiling reads as measured.
    """


def _rendered(stats: PlacementStats) -> PlacementStatsView:
    """Turn the domain numbers into the wire shape, converting the one instant.

    Epoch milliseconds become an aware UTC instant. Aware rather than naive because a
    naive datetime serializes with no offset and a consumer then has to guess the zone,
    and UTC rather than local because the answer must not depend on which node answered.
    """
    oldest = stats.oldest_awaiting_placement_at_ms
    return PlacementStatsView(
        turns_awaiting_placement=stats.turns_awaiting_placement,
        sessions_placing=stats.sessions_placing,
        oldest_awaiting_placement_at=(
            None if oldest is None else datetime.fromtimestamp(oldest / 1000, tz=UTC)
        ),
        session_pods_running=stats.session_pods_running,
        nodes_schedulable=stats.nodes_schedulable,
        node_ceiling=stats.node_ceiling,
    )


_NO_CLUSTER: Final = PlacementStats(
    turns_awaiting_placement=0,
    sessions_placing=0,
    oldest_awaiting_placement_at_ms=None,
    session_pods_running=0,
    nodes_schedulable=None,
    node_ceiling=None,
)
"""The answer from a process that places no Session pods.

Zeros for the queue and the pods, nulls for the nodes, and every one of those is true
rather than a default. `composition.build` wires the real pod release inside the same
branch that builds the only thing in this tree that creates a Session pod, so a process
without one is a process in which no Session pod can exist and none can be waiting. The
node fields are null because such a process has no cluster client to ask.
"""


@router.get("/capacity", response_model=PlacementStatsView)
async def read_capacity(request: Request) -> PlacementStatsView:
    """The placement pipeline's six numbers, for a reader who holds no tenant.

    Never refuses on the state of the platform, and that is a decision rather than an
    absence of error handling. This surface is read when something is already wrong, so
    every path through it has to produce numbers: a refusal here would take away the one
    instrument at the moment it is wanted, and "which of these is unknown" is already
    expressible in the body as a null.

    The narrowing rather than a `Platform` field is explained on `CapacityReport`: the
    assembled process already holds the placement object, typed narrowly so no
    end-of-life path can reach `place` through it, and this widens that view by one
    read-only method.
    """
    release = platform_from_request(request).session_pod_release
    if not isinstance(release, CapacityReport):
        return _rendered(_NO_CLUSTER)
    return _rendered(await release.capacity())
