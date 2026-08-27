"""The single place a concrete adapter is constructed.

Every other module names a port. This module is the only one allowed to know that the
relational port is Postgres and the object port is S3, which is what makes the AWS-to-
internal move a swap rather than a rewrite — and it is the reason the ruff banned-api
rule in pyproject.toml exempts exactly this file.

Adapters are added here as later slices introduce them; nothing in this file is read by
the modules it wires, so growing it couples nothing.
"""

import logging
import os
import sys
import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

import aioboto3  # type: ignore[import-untyped]
import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from managed_agent.adapters.kubernetes.pod_runner import KubernetesPodRunner
from managed_agent.adapters.postgres.definition_registry import (
    PostgresDefinitionRegistry,
)
from managed_agent.adapters.postgres.environment_store import PostgresEnvironmentStore
from managed_agent.adapters.postgres.event_log_append import PostgresEventLogAppend
from managed_agent.adapters.postgres.event_log_range import PostgresEventLogRange
from managed_agent.adapters.postgres.placement_backlog import (
    PostgresPlacementBacklog,
)
from managed_agent.adapters.postgres.session_registry import PostgresSessionRegistry
from managed_agent.adapters.postgres.session_thread_index import (
    PostgresSessionThreadIndex,
)
from managed_agent.adapters.postgres.skill_inventory import PostgresSkillInventory
from managed_agent.adapters.postgres.skill_registry import PostgresSkillRegistry
from managed_agent.adapters.postgres.sweep_lease import PostgresSweepLease
from managed_agent.adapters.postgres.tool_registry import PostgresToolRegistry
from managed_agent.adapters.postgres.vault_store import PostgresVaultStore
from managed_agent.adapters.postgres.webhook_store import PostgresWebhookStore
from managed_agent.adapters.s3.evidence_store import (
    EvidenceStore,
    S3EvidenceBlobs,
    UnconfiguredEvidence,
)
from managed_agent.adapters.s3.rollout_store import S3RolloutStore
from managed_agent.adapters.s3.session_vfs import (
    S3LaneBlobs,
    SessionVfsStore,
    UnconfiguredSessionVfs,
)
from managed_agent.adapters.s3.uploaded_file import S3UploadedFiles
from managed_agent.adapters.secrets.vault import (
    SecretsClient,
    SecretsManagerVault,
    SecretsManagerVaultWriter,
    SecretsWriteClient,
)
from managed_agent.control.catalog.environments import EnvironmentStore
from managed_agent.control.files.attachments import (
    AttachedFiles,
    SessionAttachments,
    UnconfiguredAttachments,
)
from managed_agent.control.files.output_shipout import (
    EachAtTurnCompletion,
    ShipOutOutputsAtTurnCompletion,
)
from managed_agent.control.files.rollout_sync import (
    RolloutSync,
    ShipOutAtTurnCompletion,
)
from managed_agent.control.files.store import (
    UPLOAD_BUCKET_ENV_VAR,
    FileStore,
    NoUploadBucket,
    UploadedFileStorage,
    upload_limit_from_env,
)
from managed_agent.control.reviewers.token import (
    HmacReviewerTokens,
    NoReviewerKey,
    ReviewerAuthenticator,
)
from managed_agent.control.session.abandoned_turns import AbandonedTurnSweeper
from managed_agent.control.session.lifecycle import NoSessionPods, SessionPodRelease
from managed_agent.control.session.placement import PlacedPods, Placement, PodRunner
from managed_agent.control.session.pod_tls import dial_context
from managed_agent.control.session.pods import FirstTurnPlacement
from managed_agent.control.session.reaper import SessionPodReaper
from managed_agent.control.session.threads import (
    NoSessionThreads,
    SessionThreadIndex,
)
from managed_agent.control.session.turn_dispatch import NoPodTransport, TurnDispatch
from managed_agent.control.session.turn_execution import BackgroundTurns
from managed_agent.control.skills.inventory import (
    NoSkillInventory,
    SkillInventory,
)
from managed_agent.control.skills.registry import SkillStore, UnconfiguredSkills
from managed_agent.control.sweep_loop import Sweep
from managed_agent.control.webhooks.dispatcher import WebhookDispatcher
from managed_agent.control.webhooks.registry import WebhookStore
from managed_agent.core.ids import SessionId, TurnId
from managed_agent.core.ports import (
    CredentialVaultWriter,
    DefinitionRegistry,
    EventLogAppend,
    EventLogRange,
    SessionRegistry,
    ToolRegistry,
    VaultCatalogue,
)
from managed_agent.core.tls.session_certificate import InternalCa
from managed_agent.core.vault_catalogue import (
    UnconfiguredCredentialWriter,
    UnconfiguredVaultCatalogue,
)
from managed_agent.core.vfs.evidence import EvidenceRecorder, threshold_from_env
from managed_agent.core.vfs.session_vfs import SessionFiles
from managed_agent.core.vocabulary import tool_in_flight
from managed_agent.gateway.model.anthropic_messages import AnthropicMessagesHandler
from managed_agent.gateway.model.credential_broker import ProviderCredentialBroker
from managed_agent.gateway.model.passthrough import ResponsesPassthrough
from managed_agent.gateway.model.router import (
    ModelGateway,
    SessionTokenVerifier,
    UpstreamWire,
    WireHandler,
    create_model_gateway_app,
    routing_table_from_json,
)
from managed_agent.gateway.tool.credential_broker import ToolCredentialBroker
from managed_agent.gateway.tool.evidence_capture import EvidenceCapture
from managed_agent.gateway.tool.mcp_proxy import ToolEventTypes
from managed_agent.gateway.tool.server import GatewaySessions, create_gateway_app
from managed_agent.session_shim.pod_channel import (
    HttpPodDispatch,
    PodFilePlacement,
    PodOutputFetch,
    PodRolloutFetch,
)
from managed_agent.session_shim.turn_runner import TurnCompleted

# `pool_size` has to be at least the peak concurrency, and `max_overflow` does not
# substitute for it. That is the whole of this setting, and it is not what the keyword
# names suggest: a connection handed out *above* `pool_size` is created for that one
# checkout and closed on return, so every request served by overflow pays a fresh TCP
# connect and PostgreSQL authentication, and pays it again next time.
#
# Measured here, 50 concurrent appends against a real PostgreSQL 17 on 11 cores, median
# of five rounds after a warm-up round, each configuration run in both orders to keep
# the cold-cache advantage off whichever one went first:
#
#   size=5   overflow=10   ceiling 15    142.5 ms
#   size=32  overflow=8    ceiling 40    114.0 ms
#   size=40  overflow=10   ceiling 50    132.0 ms   <- ceiling reaches 50
#   size=50  overflow=0    ceiling 50     20.5 ms   <- pool reaches 50
#   size=64  overflow=16   ceiling 80     20.9 ms
#
# The two ceiling-50 rows are the ones that matter: identical ceilings, 6x apart. And 64
# buys nothing over 50, so this is not "bigger is faster" -- it is a cliff at the point
# where the pool itself covers the concurrency, and flat on either side of it.
#
# 50 is ADR-029's M1 capacity plan: N = 50 concurrent live Sessions, each
# appending to its Event Log on every Turn. The overflow is small and is there to absorb
# a burst above the plan, slowly, rather than to raise the plan.
#
# These are per-process, so replica count multiplies them against one `max_connections`.
# That is the number to check before scaling out rather than after.
#
# 50 is ADR-029's M1 plan for concurrent live Sessions, and it is
# `pool_size` rather than `pool_size + max_overflow` because a connection handed out
# above the pool is created for that one checkout and closed on return.
# tests/test_composition.py records the measurement: 50 concurrent appends took
# 20.5 ms at size=50 and 132 ms at size=40. Overflow does not stand in for pool.
#
# On 2026-08-23 these did not fit the database, and the discovery is worth keeping. The
# ceiling had been computed from the RDS parameter group's formula -- "roughly 225 at
# 2 GiB" -- and never measured. Measured from inside a control-plane pod, db.t4g.small
# gave `show max_connections` = 181 with three reserved for superusers, so 178; the
# platform's own demand was 180 and it had passed a guard comparing against 225.
#
# The instance was resized to db.t4g.medium rather than the pool cut to 30, because
# cutting it would have moved the capacity plan by 40% and paid the 6x cliff above to
# save an instance size. Measured again after the resize: 400, so 397 for map_app.
_POOL_SIZE = 50
_MAX_OVERFLOW = 10

# The namespace a Session's pod is dialled in when nothing says otherwise. A default is
# safe here in a way `DATABASE_URL`'s would not be: a wrong namespace resolves to no pod
# and the Turn is refused as undeliverable, where a wrong database URL would be written
# to.
_DEFAULT_NAMESPACE = "default"

# Whether this process places Session pods at all. A path and not a flag, because the
# thing the placer cannot do without is the manifest, and a flag would let a process
# declare itself a placer and then fail at the first Session.
_POD_MANIFEST_ENV = "MAP_POD_MANIFEST"

_NAMESPACE_ENV: Final = "MAP_NAMESPACE"
"""The namespace a Session's pod is placed into, and the only variable the placer reads
that a deployed control plane already sets.

Named as a constant because two things read it and one of them reads it to decide
whether this process *meant* to be a placer -- see `pod_runner_from_environment`.
"""

_ANTHROPIC_MAX_TOKENS_ENV: Final = "MAP_ANTHROPIC_MAX_TOKENS"
"""The output ceiling every Anthropic Messages request carries.

The Messages API rejects a request with no `max_tokens`, and the Responses body the
Agent Runtime sends has no field that means the same thing, so this number is the
platform's own choice rather than a translation of anything. An operator's, because it
trades a truncated answer against a runaway one and neither cost is the translator's to
weigh -- a value compiled in would need a rebuild to change.

The Model Gateway is the only process that reads it, and it reads it whether or not any
routed model is on this wire: a handler is registered per shape, once, at start-up.
"""

_SESSION_TOKEN_KEY_ENV: Final = "MAP_SESSION_TOKEN_KEY"
"""The key a Session's compiled configuration is signed with, on the SIGNING side.

The same variable name the Tool Gateway reads on the verifying side, and it must
resolve to the same bytes: that Gateway's only check on an incoming tool call is that
this signature verifies, so two different values make every Session's tool calls a 401
whose cause nothing names. Both manifests therefore name one Secret and one key, and a
test scans every manifest rather than trusting two.

A DIFFERENT key from MAP_SHIM_TOKEN_KEY, which signs the control-plane-to-shim bearer:
one key for two hops in opposite directions would make either side's compromise the
other's.
"""

_SESSION_TOKEN_LIFETIME_ENV: Final = "MAP_SESSION_TOKEN_LIFETIME_S"
"""How long a Session's token stays valid, in seconds, with no default.

A lifetime and not an expiry because the compiler wants an absolute epoch second and
only a running process knows what `now` is. No default because this is a security
parameter, and one hidden in code is one no operator can see -- and because the token
cannot be refreshed: the document is copied into the pod at start and read once, so
this is a ceiling on how long the Session may use enterprise tools rather than a
window that rolls forward. When it passes the Session stops taking Turns altogether,
because the Gateway is a `required` server and its refusal fails the thread.
"""

_TOOL_GATEWAY_URL_ENV: Final = "MAP_TOOL_GATEWAY_URL"
_MODEL_GATEWAY_URL_ENV: Final = "MAP_MODEL_GATEWAY_URL"
"""Where the two gateways answer, for the document a Session's pod is started from.

Deployment configuration and not constants, because one cluster's Services are not
another's -- config_compiler makes the same point about the model provider's address
arriving beside the Tool Gateway's rather than being written down here. Required
rather than defaulted for a reason measured on real codex-cli 0.149.0 (MAP-55 round 2,
variant E): a provider table carrying a name and no base_url LOADS, reports the
provider by name, and then sends model calls to the runtime's own default endpoint --
a request leaving a Session pod for an address no Egress Policy allowed, with no error
anywhere.
"""
_BUCKET_ENV: Final = "MAP_ROLLOUT_BUCKET"
"""Where a Session's Rollout is kept, named rather than defaulted.

`MAP_`-prefixed for the reason every variable here is: a generic name is one a base
image or a sidecar could set for its own reasons, which this platform would then
silently adopt as the place every tenant's resume state goes.
"""

_WEBHOOK_SWEEP: Final = "webhook-delivery"
_SESSION_POD_SWEEP: Final = "session-pods"
_ABANDONED_TURN_SWEEP: Final = "abandoned-turns"
"""What each periodic pass is called, in a task name and in every log line about it.

Constants because the names are also the advisory-lock keys the lease is taken under, so
a second spelling of one would be a second lock -- and two replicas each holding "their"
lock is exactly the state the lease exists to make impossible.
"""


@dataclass(frozen=True)
class Platform:
    """Every port the process needs, already bound to an adapter.

    Later slices add fields. Nothing that reads a field of this knows which adapter is
    behind it, so growing this class couples nothing to the growth.
    """

    event_log_append: EventLogAppend
    event_log_range: EventLogRange
    definition_registry: DefinitionRegistry
    tool_registry: ToolRegistry
    session_registry: SessionRegistry
    webhooks: WebhookStore
    environment_store: EnvironmentStore
    turn_dispatch: TurnDispatch
    file_store: FileStore
    evidence_capture: EvidenceCapture = field(
        default_factory=lambda: EvidenceCapture(
            UnconfiguredEvidence(), threshold_from_env()
        )
    )
    """Where a large tool result becomes hashed Evidence before a model reads it.

    The one field here with a default, and the default refuses every call. Required
    would have been the honest signature and it is not available: `Platform` is
    constructed in two dozen places, so a required field breaks all of them at once --
    including branches other slices are writing right now, where the field does not
    exist yet, with `mypy --strict` reporting it only at merge. `tools/plan_waves.py`
    names that shape and this slice shares a wave with three others that declare this
    file.

    So the default carries the safety instead of the type: a `Platform` built without
    one holds a capture point that fails any tool call it is asked to classify, rather
    than one that quietly lets the bytes through.
    """

    session_attachments: SessionAttachments = field(
        default_factory=UnconfiguredAttachments
    )
    """How a file reaches a running Session's pod, for the route that attaches one late.

    The same object the placement path holds, not a second one built beside it: two
    `AttachedFiles` over one file store would be two byte budgets, and the budget is a
    property of one pod's disk.

    Defaulted for the reason the fields above are, and the default refuses. A deployment
    with no pod runner has nowhere to put a file, so the honest answer there is a
    refusal naming that -- not a 201 for bytes that went nowhere, which is the exact
    failure the whole delivery path exists to prevent.
    """

    vault_catalogue: VaultCatalogue = field(default_factory=UnconfiguredVaultCatalogue)
    """Where a tenant's vaults and credential names live. No value passes through it.

    Defaulted and refusing for the reason the fields above are. Refusing matters more
    here than on most of them: the failure this prevents is a 201 for a credential that
    was recorded nowhere, and every symptom of that appears later at somebody else's
    MCP server, as an authentication error about a service this platform does not own.
    """

    credential_writer: CredentialVaultWriter = field(
        default_factory=UnconfiguredCredentialWriter
    )
    """Where a submitted credential's value goes, and the only thing putting one there.

    A writer with no `fetch`, deliberately. The control plane accepts a tenant's
    credential and may overwrite it; reading one back is the Tool Gateway's job under a
    different IAM role, and the split is a type here as well as a policy there so that
    code which would read a tenant's tool credential does not compile.
    """

    session_artifacts: SessionFiles = field(default_factory=UnconfiguredSessionVfs)
    """A Session's durable lanes, for the route that serves one artifact back.

    The same object the ship-out seam places into, not a second one built beside it. Two
    stores over one bucket agree today and stop agreeing the day either grows a cache or
    a limit -- and here the disagreement would be a tenant told their artifact does not
    exist while the Turn that produced it says it does.

    Defaulted for the reason the fields above are, and the default refuses every call
    rather than answering empty. "There is no bucket" and "the lane holds nothing" are
    different facts, and a read that answered None against an unwired store would tell a
    tenant their document was never produced.
    """

    skill_inventory: SkillInventory = field(default_factory=NoSkillInventory)
    """Where the skill surface reads what a tenant holds, across both write doors.

    A refusing default rather than an empty-answering one, which is the opposite of
    `session_threads` below and the difference is worth stating. Empty is a true answer
    for a Session that never delegated. Empty is never a true answer here: a tenant who
    uploaded a skill and got a 201 holds one, so a process wired without a store would
    report an empty collection to somebody it can see is wrong about. A tenant told "you
    have no skills" believes it; a tenant handed an error asks.
    """

    session_threads: SessionThreadIndex = field(default_factory=NoSessionThreads)
    """Where the thread surface reads a Session's threads and their timestamps.

    A refusing-by-emptiness default rather than `None`, so no route tests the field
    before using it. Empty is the honest answer for a process wired without the index:
    it knows of no threads, which is also true of every Session whose events predate
    attribution, so a caller already handles it.
    """

    skill_store: SkillStore = field(default_factory=UnconfiguredSkills)
    """Where an uploaded skill is held, and what a definition's skills resolve out of.

    Defaulted for the reason the two fields below are -- `Platform` is built in two
    dozen places, several of them on branches being written right now -- and the default
    refuses every call rather than answering "no skills".

    That direction is the whole point. A store that answered emptily would make a
    platform assembled without one indistinguishable from a tenant who attached none:
    every definition would resolve, every Session would start, and every agent would
    quietly have none of the skills its definition names. That silence is the defect
    this field exists to end, so the unconfigured case is loud instead.
    """

    reviewer_authenticator: ReviewerAuthenticator = field(default_factory=NoReviewerKey)
    """What turns a presented token into the platform reviewer allowed to read any
    tenant's Event Log.

    Defaulted for the reason above -- `Platform` is built in two dozen places -- and the
    default refuses every token, so a process wired without one serves every other route
    and answers the audit surface exactly as it answers a stranger.

    **The default has to be a refusing object and not an empty key.** An empty key is a
    perfectly valid HMAC key, so a field typed `bytes` and defaulted to `b""` would
    verify tokens anybody can mint while looking like a closed door -- and it would look
    closed in review too, which is why this is written down rather than tidied into the
    shorter form.
    """

    session_pod_release: SessionPodRelease = field(default_factory=NoSessionPods)
    """What gives a Session's pod back when that Session stops needing it.

    Defaulted for the reason the two fields above are -- `Platform` is built in two
    dozen places -- and this is the one defaulted field whose default does nothing
    rather than refusing. That is not a weaker choice here, it is the accurate one: the
    only thing that creates a Session pod is `FirstTurnPlacement`, built below inside
    the same `pod_runner is not None` branch that wires the real release, so a process
    holding the default is a process in which no Session pod can exist. A refusing
    default would make archiving fail in every process that has nothing to hand back.

    Narrow on purpose. `Placement` satisfies it by having `release`, and this field's
    type names only that method, so the archive path cannot reach `place` -- which would
    make every end-of-life route a place a Session could be revived by accident.
    """

    background_turns: BackgroundTurns = field(default_factory=BackgroundTurns)
    """The tasks carrying Turns that have already been answered with a 202.

    Wiring only, and it holds no collaborators: what a task does arrives as a coroutine
    the route builds, so this field needs nothing from `build` and can be defaulted
    without the two dozen callers that construct a `Platform` having to say anything.

    Defaulted to a real one rather than to a refusing double, which is the opposite of
    the stores above and the difference is the point. A process with no store genuinely
    cannot serve that surface; a process always can run a Turn it admitted, and a
    refusing default here would turn every submission into a 500. What the default does
    risk is a process whose lifespan never calls `aclose`, so `asgi.py` is graded on
    doing it.
    """

    sweeps: tuple[Sweep, ...] = ()
    """The periodic passes the serving process is supposed to be running.

    Wiring only. What starts them is the app's lifespan in `asgi.py`, and holding them
    here rather than starting them at construction is what keeps `build` free of running
    tasks -- a `Platform` is built by two dozen callers, most of them tests, and a
    constructor that spawned a loop would spawn one in each of them.

    Defaulted to empty for the reason the four fields above are defaulted: `Platform` is
    constructed in two dozen places and a required field breaks all of them at once,
    including branches other slices are writing right now. The default is honest here in
    a way it is not for a store -- a process assembled without sweeps genuinely has none
    to run, and it serves every route exactly as before -- so what an empty tuple risks
    is a control plane that silently stops sweeping. `tests/control/` grades the wired
    tuple against the two sweeps this platform has for that reason.
    """


def _a_positive_lifetime(raw: str) -> int:
    """A Session-token lifetime in seconds, refusing every value that is not one.

    Zero and negative are refused here rather than left to arithmetic downstream,
    because what they produce is a token that is already expired when it is minted --
    and no floor over a compiled configuration reads the expiry at all. That document
    compiles, the pod starts, and every tool call it makes is answered with the Tool
    Gateway's fixed 401, which is the same answer an absent token gets. Nothing
    downstream can name the cause, so the value is refused at the one point where the
    variable it came from can be named.

    Raises `ValueError` with the variable in the message, which reaches an operator the
    same way a missing Secret key does: the process does not start.
    """
    try:
        seconds = int(raw)
    except ValueError as unparseable:
        raise ValueError(
            f"{_SESSION_TOKEN_LIFETIME_ENV} is {raw!r}, which is not a whole number of "
            "seconds"
        ) from unparseable
    if seconds <= 0:
        raise ValueError(
            f"{_SESSION_TOKEN_LIFETIME_ENV} is {seconds}, so every Session's token "
            "would be expired at the moment it is minted and that Session's pod would "
            "answer 401 on every tool call for its whole life"
        )
    return seconds


_LOG_LEVEL_ENV_VAR: Final = "MAP_LOG_LEVEL"
_HANDLER_NAME: Final = "map-platform"


def install_platform_logging() -> None:
    """Give this package's logger a handler, because `uvicorn` gives it none.

    `uvicorn` configures logging from its own `LOGGING_CONFIG`, which names `uvicorn`,
    `uvicorn.error` and `uvicorn.access` and declares no `root` entry. A `dictConfig`
    carrying no `root` key leaves the root logger as Python built it -- level WARNING,
    holding no handler -- so a `managed_agent` record at INFO propagated there, found
    nothing to write it, and was dropped by `logging.lastResort`, whose own level is
    WARNING. The effect was that under `uvicorn` this package could not emit an INFO
    line at all, and a guard written at INFO was indistinguishable from outside the
    process from a guard that was never called.

    Attached to `managed_agent` rather than to root so `uvicorn`'s own three loggers
    stay exactly as `uvicorn` configured them, and `propagate` is turned off so a
    deployment that later does configure root does not write every line twice.

    Called once per process entrypoint rather than at import, because importing a module
    should not reconfigure logging for whoever imported it -- a test collector and a
    linter both import this file. It is idempotent because a factory may be called more
    than once in a process.

    An unreadable `MAP_LOG_LEVEL` falls back to INFO and says so rather than refusing
    to start. Fail-fast is the usual answer and is wrong here: it would trade a
    platform that will not boot against catching a typo seconds earlier.
    """
    logger = logging.getLogger("managed_agent")
    if any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    requested = os.environ.get(_LOG_LEVEL_ENV_VAR)
    resolved = logging.getLevelNamesMapping().get((requested or "INFO").upper())
    logger.setLevel(logging.INFO if resolved is None else resolved)
    if requested is not None and resolved is None:
        logger.warning(
            "%s is %r, which names no logging level, so this process logs at INFO",
            _LOG_LEVEL_ENV_VAR,
            requested,
        )


class _RolloutNotYetShipped:
    """The `TurnCompleted` wired when no Rollout bucket is configured: it does nothing.

    A completed Turn is the platform's durability boundary (ADR-004), and what happens
    at it -- shipping the Session's Rollout out of the pod before the pod goes -- is
    built now, but only reachable with a bucket to put the bytes in. Named rather than
    written as an inline lambda so that the gap is greppable and so a reader of the
    wiring can see that a Turn's completion is observed here and then deliberately
    dropped, instead of wondering whether it was forgotten.
    """

    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        return None


def build(
    database_url: str | None = None,
    pod_runner: PodRunner | None = None,
    rollout_bucket: str | None = None,
) -> tuple[Platform, AsyncEngine]:
    """Wire the ports and hand back the engine that owns their connections.

    The engine comes back rather than staying private because its pool outlives any one
    request and something has to dispose of it at shutdown; a port that closed its own
    engine would close it for every other port sharing it.

    With no argument the URL is read from `DATABASE_URL` and its absence raises. That is
    deliberate: a default would let an unconfigured process start and reach the wrong
    database, which is worse than not starting.

    Without a `PodRunner` the Turn transport is `NoPodTransport` and every Turn is
    refused. That is not a placeholder standing in for something nearly finished: a
    caller with no runner has no cluster to locate a Session's pod in, and a transport
    wired over a placement that cannot answer would fail at the first Turn rather than
    at start-up. With one, the real transport is on the wired path and `NoPodTransport`
    is nowhere on it. Whether the served process has one is
    `pod_runner_from_environment`'s answer, below.

    `MAP_SHIM_TOKEN_KEY` has no default and its absence raises, for the same reason
    `DATABASE_URL`'s does -- and it is read only on the path that needs it, so a process
    with no pod runner is not asked for a key it will never sign with. A process that
    started with an empty signing key would derive a token every pod could also derive,
    and the shim route's only check would then pass for anyone on the cluster network.

    The four values a Session's configuration is compiled from -- the two gateway
    addresses, the Session-token signing key and how long a token stays valid -- are
    read on that same path and none has a default either. They are read here rather
    than at the Turn so that a placer missing one fails to start instead of accepting
    Sessions it will refuse a pod to; each constant above says why its own value cannot
    be guessed.

    Without a Rollout bucket a completed Turn ships nothing and `_RolloutNotYetShipped`
    stays wired. That is the same shape as the pod runner above and for the same reason:
    a bucket name guessed from a default would write every tenant's resume state
    somewhere nobody chose, and a process that cannot ship should say so rather than
    look configured. The name is read from `MAP_ROLLOUT_BUCKET` when the argument is
    absent -- with `.get` and not `os.environ[...]`, so a control plane configured for
    Turns but not yet for recovery still starts.
    """
    url = database_url or os.environ["DATABASE_URL"]
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        pool_size=_POOL_SIZE,
        max_overflow=_MAX_OVERFLOW,
    )
    log = PostgresEventLogAppend(engine)
    # `.get` and not `os.environ[...]`: an absent bucket makes the one file surface
    # refuse at its first request, naming the variable, rather than making a resource
    # only that surface uses a start-up condition for the whole control plane.
    bucket = os.environ.get(UPLOAD_BUCKET_ENV_VAR)
    files: UploadedFileStorage = NoUploadBucket()
    # Evidence and uploaded files share the one bucket and are separated by prefix, so
    # they read the one variable. The fallback here refuses rather than degrading: an
    # upload surface that answers a refusal is a feature nobody can use, while tool
    # output that reached the model with no Evidence behind it is an audit record that
    # is simply gone.
    evidence: EvidenceRecorder = UnconfiguredEvidence()
    # The third thing in that bucket, under a third prefix: a Session's durable lanes.
    # Built here beside the other two rather than at the seam that ships a Turn's
    # artifacts, because the route that serves an artifact back reads the same lanes --
    # and a store constructed twice is two objects over one bucket that agree until
    # either grows a cache or a limit.
    #
    # The unconfigured fallback refuses every call rather than answering empty, which is
    # the choice `UnconfiguredEvidence` makes and for a sharper version of the same
    # reason: a listing that reported no objects against an unwired store would read as
    # a Session that produced nothing, and an agent told its artifact was stored with
    # nothing behind the call has been told a lie it cannot detect.
    artifacts: SessionFiles = UnconfiguredSessionVfs()
    if bucket is not None:
        files = S3UploadedFiles(aioboto3.Session(), bucket, engine)
        evidence = EvidenceStore(S3EvidenceBlobs(aioboto3.Session(), bucket), engine)
        artifacts = SessionVfsStore(S3LaneBlobs(aioboto3.Session(), bucket), log)
    events = PostgresEventLogRange(engine)
    sessions = PostgresSessionRegistry(engine)
    environments = PostgresEnvironmentStore(engine)
    definitions = PostgresDefinitionRegistry(engine)
    # One store, read by two callers: the API route that registers a skill and the
    # placement path that resolves a definition's skills into files. Two constructions
    # would be two objects over one table today and nothing would notice -- and the day
    # one of them is given a cache or a different schema, a Session would resolve
    # against a store the upload never reached.
    skills = PostgresSkillRegistry(engine)
    # One store, and the same reasoning as the skill store above: the route a tenant
    # uploads through and the placement path that reads those bytes back into a pod have
    # to be the same object over the same bucket. Two constructions would agree today
    # and would be two answers the day either grows a cache or a limit.
    uploads = FileStore(files, upload_limit_from_env())
    # One store with three readers -- the route that registers a webhook, the
    # registrations the sweep matches a state change against, and the ledger it claims
    # an attempt in -- so it is a name here rather than an argument written inline
    # below. Two constructions would be two objects over one table today and nothing
    # would notice, which is the same trap the one skill store above avoids.
    webhooks = PostgresWebhookStore(engine)
    clock = _SystemClock()
    # Not closed anywhere, and that is the same trade `model_gateway_app` takes with its
    # own client: it lives exactly as long as the process, and the alternative is
    # threading an `aclose` through a lifespan that owns the sweeps rather than the
    # sockets. httpx's default timeouts apply -- five seconds to connect and five to
    # read -- which is the bound that keeps one hanging receiver from stalling a pass.
    dispatcher = WebhookDispatcher(
        webhooks, webhooks, events, _secrets_manager_vault(), httpx.AsyncClient()
    )

    async def _deliver_the_callbacks_that_are_due() -> object:
        """One delivery pass, at the instant it is asked for.

        `sweep_once` takes the instant rather than reading a clock, so something has to
        read one and this is that something. Named rather than written as a lambda in
        the wiring below, so a reader can see which clock a callback's `delivered_at_ms`
        and its signed timestamp header come from.
        """
        return await dispatcher.sweep_once(clock.now_epoch_ms())

    sweeps: list[Sweep] = [
        Sweep(
            name=_WEBHOOK_SWEEP,
            run=_deliver_the_callbacks_that_are_due,
            # A lease, because the delivery claim does not exclude a second runner. Its
            # `ON CONFLICT ... DO UPDATE ... WHERE delivered_at_ms IS NULL AND attempts
            # < :max RETURNING attempts` hands the loser of a race a row exactly as
            # readily as the winner -- it counts attempts, it does not pick a runner --
            # so two replicas over one window post the same callback to the tenant twice
            # and spend two of its five attempts. `sweep_lease.py` carries the argument
            # and `tests/control/` proves both halves of it against a real database.
            lease=PostgresSweepLease(engine),
        )
    ]
    dispatch: TurnDispatch = NoPodTransport()
    # Bound beside the dispatch for the same reason the release below is: a process
    # that places no pod has nowhere to write a file, and the refusing default says
    # exactly that rather than answering 201 for bytes that went nowhere.
    session_attachments: SessionAttachments = UnconfiguredAttachments()
    # Bound beside the dispatch and from the same branch, because the two are the same
    # fact about this process: it either places Session pods or it does not, and a
    # process that never places one has none to hand back.
    release: SessionPodRelease = NoSessionPods()
    if pod_runner is not None:
        # The backlog reader is the log, not this process. `PlacementWaits` inside
        # `Placement` counts only the waits this replica is holding open, and the
        # serving Deployment runs two: measured against `map-dev` on 2026-08-24, six
        # Sessions placing at once produced `turns_awaiting_placement: 6` from one
        # replica and `0` from the other. A client's connection is sticky, so an
        # operator's whole workload lands on one replica and an operator polling the
        # other reads an idle platform.
        #
        # Wired here because this is the only place that holds both the engine and the
        # placement object, and it costs no new write: the two events the query reads --
        # `session.placing` and the Turn's own `turn.started` or `turn.failed` -- were
        # already being appended.
        placement = Placement(pod_runner, backlog=PostgresPlacementBacklog(engine))
        release = placement
        namespace = os.environ.get(_NAMESPACE_ENV, _DEFAULT_NAMESPACE)
        token_key = os.environ["MAP_SHIM_TOKEN_KEY"].encode()
        # The same CA the pod runner signs pods with, on the dialling side. Read once
        # here rather than per dialler so the scheme every one of them builds and the
        # credentials every one of them presents come from a single decision -- two
        # diallers that disagreed about the scheme would fail half the routes and leave
        # the other half working, which is the hardest shape of this to diagnose.
        #
        # `None` when no CA is configured, which is what keeps a deployment with no
        # material on plain HTTP exactly as before (ADR-044).
        internal_ca = internal_ca_from_environment()
        pod_dial = None if internal_ca is None else dial_context(internal_ca)
        # `.encode()` and `int()` here rather than inside `FirstTurnPlacement`: parse at
        # the boundary and let the type carry the proof inward. An unparseable lifetime
        # raises `ValueError` at start-up naming the variable, which reaches an operator
        # the same way a missing key does.
        #
        # The five adapters are named as arguments rather than reached through the
        # `Platform` being built: a collaborator that reached into the object it is a
        # field of would make construction order load-bearing.
        # Built once and named, because two callers hold it: the placement path
        # below and `Platform.session_attachments` for the route that attaches a
        # file to a Session already running. Two instances over one file store
        # would be two byte budgets for one pod's disk.
        attachments = AttachedFiles(
            uploads,
            # `tls=pod_dial` like every other hop to a pod. Left off, this one dialled
            # `http://` at a listener holding a certificate, which drops the connection
            # without a TLS alert -- so a Session with an attached file failed
            # `pod_unreachable` at placement and the error named neither TLS nor files.
            PodFilePlacement(placement, namespace, token_key, tls=pod_dial),
        )
        session_attachments = attachments
        pods = FirstTurnPlacement(
            placement=placement,
            sessions=sessions,
            environments=environments,
            definitions=definitions,
            events=events,
            skills=skills,
            # The file store built above, and the write half of the shim hop. Both are
            # named here rather than reached out of the `Platform` for the same reason
            # every other adapter is: a collaborator that read a field of the object it
            # is a field of would make construction order load-bearing.
            attachments=attachments,
            clock=_SystemClock(),
            session_token_key=os.environ[_SESSION_TOKEN_KEY_ENV].encode(),
            session_token_lifetime_s=_a_positive_lifetime(
                os.environ[_SESSION_TOKEN_LIFETIME_ENV]
            ),
            tool_gateway_url=os.environ[_TOOL_GATEWAY_URL_ENV],
            model_gateway_url=os.environ[_MODEL_GATEWAY_URL_ENV],
        )
        # Stripped and tested for truth, not for `is not None`: a variable set to the
        # empty string or to whitespace is an operator who meant to name a bucket and
        # did not, and admitting it builds a real store addressing a blank bucket. Every
        # ship-out then fails at the AWS call, after the Turn is already appended, where
        # the unset case fails visibly at the seam instead.
        rollouts = (rollout_bucket or os.environ.get(_BUCKET_ENV) or "").strip()
        # Held under its own name, and that is the whole point of the extra binding.
        # `on_completed` below becomes a COMPOSITE of this seam and the output
        # ship-out, and the two owe a Turn different things depending on how it ended.
        # The Rollout is the Session's ability to run again, which a Turn owes whether
        # it succeeded or failed; the produced files are a claim about output, which a
        # failed Turn has not made. A caller holding only the composite cannot ask for
        # one without the other, so it either loses a failed Turn's conversation or
        # appends `output.produced` for a Turn that produced nothing.
        rollout_seam: TurnCompleted = _RolloutNotYetShipped()
        if rollouts:
            rollout_seam = ShipOutAtTurnCompletion(
                PodRolloutFetch(placement, namespace, token_key, tls=pod_dial),
                RolloutSync(S3RolloutStore(aioboto3.Session(), rollouts)),
            )
        on_completed: TurnCompleted = rollout_seam
        # The second seam a completed Turn owes: the files the agent wrote. Conditional
        # on the object bucket, because `uploads` is the refusing `NoUploadBucket` store
        # without one -- wired unconditionally, every Turn that produced a file would be
        # recorded as failed on a deployment that has no object store at all.
        #
        # Composed rather than replacing, and second rather than first: the Rollout
        # is the Session's resume state, and a seam that raises stops the ones behind
        # it, so a failure to ship one document must not also cost the Session its
        # ability to run again.
        if bucket is not None:
            # Two seams, not three. A third used to copy the agent's working tree
            # out at every completed Turn. The workspace is a mounted volume now, so
            # there is nothing to copy -- but the seam had to go rather than merely
            # having nothing to do, because it wrote the same bytes the mount holds
            # to the same file system through a different door. S3 Files settles a
            # bucket-versus-file-system conflict in the bucket's favour and moves the
            # file system's copy to `.s3files-lost+found-<fs-id>`, which sits above
            # the access point root and is therefore invisible to the pod and to the
            # browse route both. Left in, it would silently discard the agent's live
            # workspace. ADR-035.
            outputs = PodOutputFetch(placement, namespace, token_key, tls=pod_dial)
            on_completed = EachAtTurnCompletion(
                on_completed,
                ShipOutOutputsAtTurnCompletion(
                    outputs, artifacts, sessions, log, events
                ),
            )
        dispatch = HttpPodDispatch(
            placement=placement,
            pods=pods,
            log=log,
            on_completed=on_completed,
            # The same object as `on_completed`'s first member, deliberately. A Turn
            # owes its Rollout however it ended -- that is the Session's ability to
            # run again -- but only a completed Turn owes the files it produced, and
            # a failed one declared none. Handing the composite to both would publish
            # a half-written tree under a name that reads as delivered.
            on_terminal=rollout_seam,
            namespace=namespace,
            token_key=token_key,
            tls=pod_dial,
        )
        # The pod sweep, from the same branch as the release above and for the same
        # reason: a process that places no pod has none to reclaim, and a sweep that
        # required a cluster would make an optional capability a start-up condition for
        # the whole control plane.
        #
        # `placed_pods` is narrowed rather than required of the `pod_runner` parameter,
        # for the reason `PlacedPods` gives: the real cluster client has all four
        # methods and a `PodRunner` double has three, so requiring the fourth here would
        # make every double in this suite a type error and prove nothing about what a
        # deployment wires. What the narrowing skips is therefore a test double; that
        # the client this root actually resolves is on the other side of it is asserted
        # in `tests/control/` against the class rather than trusted here.
        if isinstance(pod_runner, PlacedPods):
            sweeps.append(
                Sweep(
                    name=_SESSION_POD_SWEEP,
                    run=SessionPodReaper(
                        pods=pod_runner,
                        release=placement,
                        log=log,
                        events=events,
                        # The same reader for both, and the two are one adapter's
                        # capability rather than one port: the fold reads one Session's
                        # log and the recency scan reads every Session's boundary
                        # events, and `session_reaper.py` declares them separately so a
                        # double implements one method instead of a whole log.
                        activity=events,
                        clock=clock,
                    ).sweep,
                    # No lease, and this is a decision rather than an omission. Every
                    # verdict is a function of state both replicas read; a handback
                    # treats an absent pod as success ("Absent is success at every
                    # step" -- `pod_runner.remove`); and the one branch that writes goes
                    # through `_end_and_release`, which documents the redundant ending
                    # event as reachable, bounded and accepted platform-wide. So the
                    # worst a second replica costs is one extra `session.suspended` on a
                    # Session that was being suspended anyway, with no call leaving the
                    # cluster -- where the webhook sweep's duplicate is an HTTP request
                    # a tenant sees.
                    lease=None,
                )
            )
            # From the same branch as the pod sweep, and it has to be: the pod-gone
            # signal is a question only a cluster can answer, so a control plane that
            # places no pods has no way to ask it.
            sweeps.append(
                Sweep(
                    name=_ABANDONED_TURN_SWEEP,
                    run=AbandonedTurnSweeper(
                        pods=pod_runner,
                        scan=events,
                        events=events,
                        log=log,
                        clock=clock,
                    ).sweep,
                    # No lease, on the same reasoning as the pod sweep above and one
                    # more of its own. Every input is state both replicas read, and the
                    # one branch that writes goes through `lifecycle.close_abandoned_
                    # turn`, which folds the Turn first and appends nothing to a Turn
                    # already closed -- so the worst a second replica costs is one extra
                    # `turn.failed` on a Turn that was ending anyway, which changes no
                    # fold because `open_turn` matches terminal events by `turn_id`.
                    #
                    # A lease would also be actively wrong here rather than merely
                    # wasteful: the pod-gone grace is counted in each sweeper's own
                    # memory, so a leased sweep would hand the count to whichever
                    # replica won the tick and restart it every time the winner changed.
                    lease=None,
                )
            )
    # The same variable the placer's path reads above, read again here because the audit
    # surface needs it whether or not this process places pods -- and with `.get` rather
    # than `os.environ[...]` for the reason the upload bucket uses it: configuration one
    # surface needs should make that surface refuse, not stop the whole control plane
    # from starting.
    #
    # Empty is treated as absent, and that is not defensive tidiness. An empty string is
    # a usable HMAC key, so `MAP_SHIM_TOKEN_KEY=""` would build a verifier that accepts
    # tokens anyone can mint, and it would look configured from every angle.
    reviewer_key = os.environ.get("MAP_SHIM_TOKEN_KEY")
    reviewers: ReviewerAuthenticator = NoReviewerKey()
    if reviewer_key:
        reviewers = HmacReviewerTokens(key=reviewer_key.encode(), clock=_SystemClock())
    return (
        Platform(
            event_log_append=log,
            event_log_range=events,
            definition_registry=definitions,
            tool_registry=PostgresToolRegistry(engine),
            session_registry=sessions,
            webhooks=webhooks,
            environment_store=environments,
            turn_dispatch=dispatch,
            sweeps=tuple(sweeps),
            # The hoisted store, NOT a second `FileStore(files, ...)` built here. The
            # comment at its construction says why: the upload route and the placement
            # path that reads those bytes back into a pod must be one object over one
            # bucket, and an inline construction here is how they quietly become two.
            file_store=uploads,
            # The one reading of the one threshold. Both capture points are built from
            # this; either one reading the variable for itself would make Evidence
            # coverage a function of which tool ran.
            evidence_capture=EvidenceCapture(evidence, threshold_from_env()),
            skill_store=skills,
            reviewer_authenticator=reviewers,
            session_pod_release=release,
            # The same object the placement path holds. Two `AttachedFiles` over
            # one file store would be two byte budgets for one pod's disk.
            session_attachments=session_attachments,
            session_artifacts=artifacts,
            # Unconditional, unlike the pod-shaped fields above: reading a Session's
            # threads needs the database and nothing else, so there is no branch this
            # could be absent on. Left unwired it would not fail -- the stand-in
            # answers an empty page, which is also the honest answer for a Session that
            # never delegated, so the whole surface would look like it worked.
            session_threads=PostgresSessionThreadIndex(engine),
            # Unconditional for the same reason: the inventory is two tables and a
            # join, so there is no deployment shape this could legitimately be absent
            # on. Unlike the thread index, leaving this unwired fails loudly rather
            # than quietly -- the stand-in refuses instead of answering empty, which
            # is deliberate and is argued where the field is declared.
            skill_inventory=PostgresSkillInventory(engine),
            # Unconditional for the reason the thread index is: vault and credential
            # rows are two tables and a foreign key, so there is no deployment shape
            # this could legitimately be absent on. It refuses rather than answering
            # empty, and that is the sharper choice here than anywhere else on this
            # surface -- a credential registration that reported 201 with nothing
            # recorded produces its first symptom inside somebody else's MCP server,
            # a process and a network hop away from anyone who could read the cause.
            vault_catalogue=PostgresVaultStore(engine),
            # The writing half only, and a *separate* object from the reader the Tool
            # Gateway holds. They are two processes with two IAM roles: this one may
            # create and overwrite an entry under `map/tool-credential/` and may not
            # read one back, and the port it is typed at has no `fetch` to call. A
            # single object carrying both halves would make that separation a matter
            # of which methods this process happens to invoke.
            credential_writer=_secrets_manager_vault_writer(),
        ),
        engine,
    )


def pod_runner_from_environment() -> PodRunner | None:
    """The cluster client this process places Session pods in, or None if it has none.

    Returned as the port rather than as the concrete class, so the only line in this
    codebase that names a Kubernetes client is the import above, and a caller cannot
    come to depend on the adapter it happened to be given.

    `MAP_POD_MANIFEST` is what decides whether this process places pods at all. Named,
    and this is a placer; absent, and it is not: `build` wires `NoPodTransport` and
    every Turn is refused as undeliverable, which is the honest answer for a process
    with no cluster to place into. The variable names a file rather than being read out
    of the package because `deploy/` is not in the wheel --
    `[tool.hatch.build.targets.wheel]` packages `src/managed_agent` only -- so a
    manifest resolved by import would be present in a checkout and absent in the image.
    Whoever deploys this mounts it and says where.

    Once it is a placer, the other two variables are required and neither has a default.
    A namespace defaulting to `default` would put a tenant's Session in whatever
    namespace the process happened to land in -- and the Turn transport computes the
    shim's address from the same variable, so two defaults could disagree about where a
    Session is. An empty signing key derives a token every pod could also derive, which
    makes the shim's only check pass for anyone; the runner refuses that at
    construction.

    It is a function, and it is here rather than in the process entry point, because the
    answer names a concrete cluster client and this module is the only one allowed to
    name one. Importing an adapter is legal in this module and nowhere else: the ruff
    banned-api rule forbids `managed_agent.adapters` everywhere and exempts this file by
    name, which is the machine-checked form of the sentence at the top of this module.

    Two absences used to be silent together, and closing that is what the guard below
    is for. An unset manifest makes this return None, so the signing key is never read
    and the KeyError that was meant to be loud never fires: the alarm was wired behind
    the thing it was alarming about. Nothing that reads only absent variables can fire
    on that state, so this fires on a value that is PRESENT. `MAP_NAMESPACE` has
    exactly two readers and both are the placer's, so a process naming one with no
    manifest has declared an intention it cannot act on.

    What this does NOT catch: a process deployed with neither variable, which is a
    legitimate non-placer and indistinguishable from in here. Which workload was
    supposed to be a placer is a fact only the deployment holds, and it is checked
    there -- `deploy/platform.py` refuses to apply a workload whose manifest omits a
    variable this path requires.
    """
    manifest = os.environ.get(_POD_MANIFEST_ENV)
    if manifest is None:
        if _NAMESPACE_ENV in os.environ:
            raise RuntimeError(
                f"{_NAMESPACE_ENV} is set to {os.environ[_NAMESPACE_ENV]!r} but "
                f"{_POD_MANIFEST_ENV} is not, so this process names a namespace to "
                "place Session pods in and has no manifest to place them from. It "
                "would accept Sessions, answer 'running', and place nothing. Mount "
                f"the Session-pod manifest and name it in {_POD_MANIFEST_ENV}, or "
                f"unset {_NAMESPACE_ENV} if this process is deliberately not a placer."
            )
        return None
    return KubernetesPodRunner.from_manifest_file(
        Path(manifest),
        namespace=os.environ[_NAMESPACE_ENV],
        token_key=os.environ["MAP_SHIM_TOKEN_KEY"].encode(),
        internal_ca=internal_ca_from_environment(),
    )


_CA_CERTIFICATE_ENV: Final = "MAP_INTERNAL_CA_CERT"
_CA_KEY_ENV: Final = "MAP_INTERNAL_CA_KEY"


def internal_ca_from_environment() -> InternalCa | None:
    """The CA this process signs Session-pod certificates with, where it has one.

    Both variables absent is not an error. It is the state every deployment is in until
    an operator creates the CA material, and the platform in that state behaves exactly
    as it did before this existed -- a pod gets its bearer token and no certificate,
    and the hops stay plain HTTP inside the namespace.

    **Exactly one present is refused.** That is not a partial configuration to work
    around, it is somebody halfway through creating the material, and both ways it can
    happen are silent: a certificate with no key signs nothing, and a key with no
    certificate signs chains no pod can verify. Refusing here means the process does not
    start, which is a deploy that fails visibly rather than a fleet of pods that come up
    and cannot be dialled.
    """
    certificate_pem = os.environ.get(_CA_CERTIFICATE_ENV)
    key_pem = os.environ.get(_CA_KEY_ENV)
    if certificate_pem is None and key_pem is None:
        return None
    if certificate_pem is None or key_pem is None:
        present, missing = (
            (_CA_KEY_ENV, _CA_CERTIFICATE_ENV)
            if certificate_pem is None
            else (_CA_CERTIFICATE_ENV, _CA_KEY_ENV)
        )
        raise RuntimeError(
            f"{present} is set but {missing} is not, so this process holds half of an "
            "internal CA. Either half alone produces Session pods that cannot be "
            f"dialled. Set both, or unset {present} to keep the namespace on plain "
            "HTTP."
        )
    return InternalCa.from_pem(key_pem.encode(), certificate_pem.encode())


class _SystemClock:
    """Wall-clock milliseconds since the epoch.

    Concrete here for the reason every other concrete thing is: a module that read the
    clock itself would be a module whose expiry behaviour no test can move time for.
    """

    def now_epoch_ms(self) -> int:
        return time.time_ns() // 1_000_000


def _secrets_manager_vault() -> SecretsManagerVault:
    """The credential-vault port over AWS Secrets Manager.

    A client is entered per fetch, which is what the adapter asks for: whoever asked
    holds what it read for minutes, so this runs a handful of times per entry per
    process.

    No `botocore.config.Config` is attached, so the connect and read deadlines on a
    vault read are botocore's own 60 s. The adapter's own docstring names that bound as
    owed, and `S3UploadedFiles` above is constructed the same way -- so this is a gap
    this file already has rather than one introduced here.
    """
    session = aioboto3.Session()

    def client() -> AbstractAsyncContextManager[SecretsClient]:
        return cast(
            AbstractAsyncContextManager[SecretsClient],
            session.client("secretsmanager"),
        )

    return SecretsManagerVault(client)


def _secrets_manager_vault_writer() -> SecretsManagerVaultWriter:
    """The credential-vault *writing* port over AWS Secrets Manager.

    A second object beside `_secrets_manager_vault` rather than one with both halves,
    because the halves live in different processes under different IAM roles: the
    control plane creates and overwrites entries under `map/tool-credential/` and is
    granted no `GetSecretValue` on that prefix at all, while the Tool Gateway reads
    them and writes nothing. Typing the control plane at a port with no `fetch` makes
    "no route here can return a credential value" a fact about the available calls
    rather than a promise about which ones get made.

    A client per call, and no `botocore.config.Config`, for the reasons the reader's
    docstring above gives -- the same gap, not a new one.
    """
    session = aioboto3.Session()

    def client() -> AbstractAsyncContextManager[SecretsWriteClient]:
        return cast(
            AbstractAsyncContextManager[SecretsWriteClient],
            session.client("secretsmanager"),
        )

    return SecretsManagerVaultWriter(client)


def model_gateway_app() -> FastAPI:
    """The Model Gateway process, wired from its environment.

    Zero-argument because `uvicorn --factory` calls it with none, and every value it
    needs is read here rather than passed in: the routing table off the path its
    ConfigMap is mounted at, the vault entry names, and the two windows.

    Nothing has a default and every absence raises, for the reason `DATABASE_URL`'s
    does -- a process that started with an unconfigured routing path would serve 404 for
    every model, and one that started with an unconfigured signing-key name would fetch
    the empty entry and refuse every token, both of which look like a code defect from
    the outside.

    One `httpx.AsyncClient` for the process. Its read timeout is the only value moved
    off httpx's defaults, and it belongs inside the Agent Runtime's own stream-idle
    timeout so that this service gives up first and the runtime sees a failed request
    rather than a stall. The client is not closed anywhere: it lives exactly as long as
    the process, and a lifespan hook that closed it would be closing it at exit.

    Two shapes are registered, and the third -- Chat Completions -- is not. A model
    declared on it is refused loudly rather than sent a Responses body to an endpoint
    that does not accept one, which is the intended behaviour until the slice owning
    that wire registers a handler.

    `max_tokens` comes from the environment because the Messages API requires a cap on
    every request and the Agent Runtime has no field to carry one, so somebody has to
    choose it out loud. Read here rather than defaulted inside the handler: a default
    would be an output ceiling chosen by whoever wrote the translator, changeable only
    by a rebuild, and the two other numbers this factory reads are already tunable in
    the manifest for exactly that reason. Absent, this factory fails at start-up.
    """
    install_platform_logging()
    vault = _secrets_manager_vault()
    clock = _SystemClock()
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            5.0, read=float(os.environ["MAP_UPSTREAM_READ_TIMEOUT_S"])
        )
    )
    broker = ProviderCredentialBroker(
        vault, clock, ttl_ms=int(os.environ["MAP_CREDENTIAL_TTL_MS"])
    )
    handlers: dict[UpstreamWire, WireHandler] = {
        UpstreamWire.RESPONSES: ResponsesPassthrough(client, broker),
        UpstreamWire.ANTHROPIC_MESSAGES: AnthropicMessagesHandler(
            client, broker, max_tokens=int(os.environ[_ANTHROPIC_MAX_TOKENS_ENV])
        ),
    }
    return create_model_gateway_app(
        ModelGateway(
            table=routing_table_from_json(
                Path(os.environ["MAP_ROUTING_TABLE_PATH"]).read_bytes()
            ),
            handlers=handlers,
            tokens=SessionTokenVerifier(
                key=os.environ[_SESSION_TOKEN_KEY_ENV].encode(), clock=clock
            ),
        )
    )


def tool_gateway_app() -> FastAPI:
    """The Tool Gateway process, wired from the environment the pod was started with.

    Zero-argument because `uvicorn --factory` calls it with none. This function's name
    is part of a manifest -- `deploy/k8s/tool-gateway.yaml` names it as the string
    `managed_agent.composition:tool_gateway_app` -- so renaming it stops a Deployment
    rather than breaking an import, and a test drives the manifest's own args through it
    for that reason.

    The engine comes back from `build` and is dropped here on purpose.
    `create_gateway_app` installs its own lifespan -- it runs the MCP session manager,
    an idle sweeper and `sessions.aclose()` -- and reaching into that to thread a pool
    disposal through would make this function a co-writer of a lifespan that has nothing
    to do with connections. The pool therefore outlives every request and is released
    when the process exits, which is one connection set reaped by PostgreSQL's own
    timeouts on a rolling restart rather than released. `asgi.py` does better for the
    control plane and this service has no equivalent.

    `MAP_SESSION_TOKEN_KEY` has no default and its absence raises, for the reason
    `build`'s docstring gives about the other signing key: a process that started with
    an empty key would verify a token every pod could also mint, and the MCP route's
    only check would then pass for anyone on the cluster network. It is a **different**
    key from `MAP_SHIM_TOKEN_KEY` -- that one signs the control-plane-to-shim bearer,
    this one signs the pod-to-Gateway Session token -- and one key for two hops in
    opposite directions would make either side's compromise the other's.

    `platform.tool_registry` is passed where a `ToolRegistryReader` is wanted. The port
    it is typed as declares `lookup` and `list_for_tenant` with the same signatures and
    one method more, so this is structural narrowing and needs no cast; the Gateway
    asking for the two it uses rather than for the whole port is what keeps `register`
    out of a service that must never write a registration.

    The three event-type names come from `core/vocabulary/tool_in_flight`, which is the
    only legal source: `ToolEventTypes`'s own docstring refuses a second copy of them,
    because a service holding its own strings can emit a name the published set does not
    carry, which is the one thing a closed set exists to prevent (ADR-013).
    """
    # Read eagerly and discarded: this process refuses to start without it. `build`
    # resolves the same variable with `.get` and falls back to a recorder that refuses
    # per call, and that trade was decided when the control plane's upload surface was
    # its only reader -- where a refusal is a feature nobody can use, which is legible.
    # Here it is not. Measured on a live Turn with this variable absent: the upstream
    # tool ran, answered, and the capture then failed, so the model was told the tool
    # was broken and answered from memory instead. Nothing errored, the Turn completed,
    # and the Evidence a reviewer would need in order to notice is exactly what was not
    # written. A pod that will not start is the cheaper of those two failures by a wide
    # margin, and it is the one an operator sees immediately.
    install_platform_logging()
    os.environ[UPLOAD_BUCKET_ENV_VAR]
    platform, _engine = build()
    sessions = GatewaySessions(
        registry=platform.tool_registry,
        broker=ToolCredentialBroker(_secrets_manager_vault()),
        append=platform.event_log_append,
        events=platform.event_log_range,
        types_=ToolEventTypes(
            progress=tool_in_flight.TOOL_PROGRESS,
            elicitation_requested=tool_in_flight.TOOL_ELICITATION_REQUESTED,
            elicitation_answered=tool_in_flight.TOOL_ELICITATION_ANSWERED,
        ),
        # The one capture point `build` made, not a second one built here. The threshold
        # is one number read once, and a capture point constructed at this seam would be
        # a second reading of it -- which would make Evidence coverage a function of
        # which process served the call.
        evidence=platform.evidence_capture,
        # The same registry the control plane writes a Session through, narrowed by
        # the port this service is typed at: `SessionScopeReader` declares `fetch`
        # alone, so the Gateway can read a Session's Scope and has no way to create
        # one or page a tenant's list. Same structural narrowing as `tool_registry`
        # above, and it needs no cast for the same reason.
        #
        # Wired unconditionally. A Gateway built without this could not narrow a call
        # to the Session's Scope, and the two ways to behave then are to refuse every
        # tool call or to make every one of them at the full breadth of the tenant's
        # data -- the second is the failure ADR-003 exists to prevent, and the first
        # is a platform that does nothing. Neither is worth a default.
        scopes=platform.session_registry,
    )
    return create_gateway_app(
        sessions,
        os.environ[_SESSION_TOKEN_KEY_ENV].encode(),
        # Where a resuming pod reads its Session's conversation back from.
        #
        # `os.environ[...]` and NOT `.get` with a default, and a reader tempted to
        # soften that should read this first. A Gateway with no rollout bucket answers
        # "no Rollout" to every pod seeding one, every resuming Session then opens a
        # FRESH thread over a record it should have continued, and the platform replays
        # history the Rollout's compaction checkpoints have already folded -- billing
        # the tenant for the replay and reporting success while doing it (ADR-004).
        # That failure is invisible from every angle except the bill. A process that
        # refuses to start is the loud version of the same misconfiguration.
        #
        # Constructed inline rather than hoisted onto `Platform`, which is the other
        # question this raises: `build` reads the rollout bucket only inside its
        # `if pod_runner is not None:` placer branch, and this process places no pods,
        # so a Platform field would mean restructuring that branch. The usual warning
        # about two stores over one bucket is about two inside ONE process; the other
        # `RolloutSync` lives in the control plane, in a different pod.
        RolloutSync(S3RolloutStore(aioboto3.Session(), os.environ[_BUCKET_ENV])),
    )
