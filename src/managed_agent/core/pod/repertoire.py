"""The closed list of Agent Runtime requests the Session-shim is able to issue.

A request absent from this module is not implemented rather than denied. There is no
method-by-string entry point anywhere in the shim, so a caller cannot name a method
this file does not declare — a stronger property than a refusal, because it does not
depend on a list being kept current against the Agent Runtime's releases (ADR-002).

The parameter models here are this platform's own rather than the published SDK's
generated ones. The SDK ships the stable protocol export, whose thread-start
parameters carry no `permissions` field at all, and a named Permission Profile is the
only way this platform bounds a thread. A model that cannot express the protocol's
`sandbox` field also cannot break the Agent Runtime's rule that `sandbox` and
`permissions` are mutually exclusive: the illegal combination has nowhere to live.

`experimental_fields` names, per entry, the fields this shim sends that sit behind the
Agent Runtime's `capabilities.experimentalApi` gate. Those fields carry no compatibility
guarantee, so the set is the re-check list for a binary upgrade rather than a surprise
found in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _Outbound(BaseModel):
    """Base for everything the shim sends: frozen, closed, camelCase on the wire.

    `protected_namespaces` is emptied because two real protocol fields are named `model`
    and `modelProvider`, and Pydantic reserves the `model_` prefix by default.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        alias_generator=to_camel,
        protected_namespaces=(),
    )


class _Inbound(BaseModel):
    """Base for everything the shim parses back.

    Extra fields are ignored rather than forbidden: the Agent Runtime adds response
    fields between versions, and a strict model would turn an additive change into an
    outage.
    """

    model_config = ConfigDict(
        frozen=True, extra="ignore", populate_by_name=True, alias_generator=to_camel
    )


class TextInput(_Outbound):
    """One text item of a Turn's input. `type` is the input union's tag on the wire."""

    type: Literal["text"] = "text"
    text: str


class ClientInfo(_Outbound):
    name: str
    title: str
    version: str


class InitializeCapabilities(_Outbound):
    experimental_api: bool


class InitializeRequest(_Outbound):
    client_info: ClientInfo
    capabilities: InitializeCapabilities


class InitializedNotification(_Outbound):
    """`initialized` carries no parameters. Sent once, after `initialize` returns."""


class ThreadStartRequest(_Outbound):
    """Create the one root thread a Session wraps.

    There is deliberately no `sandbox` field. The Agent Runtime accepts `sandbox` or
    `permissions` and rejects the pair, and this platform bounds a thread with a named
    Permission Profile only — so the field that would express the forbidden mechanism
    does not exist to be set (ADR-005).

    `approval_policy` is fixed to "never" because nobody is present inside a pod to
    approve anything. It is also why the connection's inbound path handles responses
    and notifications only: with approvals off, the Agent Runtime raises no approval
    request.
    """

    cwd: str
    model: str
    model_provider: str
    permissions: str
    approval_policy: Literal["never"] = "never"
    base_instructions: str | None = None
    developer_instructions: str | None = None


class ThreadResumeRequest(_Outbound):
    """Re-attach to a thread from a restored Rollout file.

    Both `path` and `thread_id` are sent, and the file wins: the Agent Runtime's own
    precedence puts a non-empty path above a thread id. Which file is handed over is the
    recovery boundary's question and belongs to the slice that truncates the Rollout.
    """

    thread_id: str
    path: str
    permissions: str


class ThreadReadRequest(_Outbound):
    thread_id: str
    include_turns: bool


class ThreadGoalSetRequest(_Outbound):
    """Set the thread's objective and its token budget.

    Token-denominated, and the Agent Runtime reports against it without enforcing it.
    It is not the Session's Budget, which is money and is enforced between Turns.
    """

    thread_id: str
    objective: str
    status: Literal[
        "active", "paused", "blocked", "usageLimited", "budgetLimited", "complete"
    ]
    token_budget: int | None = None


class TurnStartRequest(_Outbound):
    thread_id: str
    input: tuple[TextInput, ...]


class TurnSteerRequest(_Outbound):
    """Redirect the active Turn.

    `expected_turn_id` is a required precondition on the wire: the call fails when it
    does not match the currently active Turn, which is what stops a steer racing a
    completion from landing on the Turn after the one it was written for.
    """

    thread_id: str
    expected_turn_id: str
    input: tuple[TextInput, ...]


class TurnInterruptRequest(_Outbound):
    thread_id: str
    turn_id: str


class InitializeResponse(_Inbound):
    """The handshake's answer. `user_agent` is the only version the protocol reports."""

    user_agent: str | None = None


class ThreadRef(_Inbound):
    id: str


class TurnRef(_Inbound):
    id: str


class ThreadStartResponse(_Inbound):
    thread: ThreadRef


class ThreadResumeResponse(_Inbound):
    thread: ThreadRef


class ThreadReadResponse(_Inbound):
    thread: ThreadRef


class TurnStartResponse(_Inbound):
    turn: TurnRef


class TurnSteerResponse(_Inbound):
    turn_id: str


class EmptyResponse(_Inbound):
    """A response the shim reads nothing out of. Its fields stay the Agent Runtime's."""


class RepertoireMethod(StrEnum):
    """Every Agent Runtime method the Session-shim may name — the whole set, here."""

    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    THREAD_START = "thread/start"
    THREAD_RESUME = "thread/resume"
    THREAD_READ = "thread/read"
    THREAD_GOAL_SET = "thread/goal/set"
    TURN_START = "turn/start"
    TURN_STEER = "turn/steer"
    TURN_INTERRUPT = "turn/interrupt"


@dataclass(frozen=True, slots=True)
class RepertoireEntry:
    """One declared call: its method, what it sends, and what it expects back.

    A `response_model` of None means the call is a notification. Encoding it that way
    rather than with a separate flag makes "a notification with a response model" and "a
    request without one" unrepresentable instead of merely wrong.
    """

    method: RepertoireMethod
    params_model: type[BaseModel]
    response_model: type[BaseModel] | None
    experimental_fields: frozenset[str]


_ENTRIES: Final[tuple[RepertoireEntry, ...]] = (
    RepertoireEntry(
        method=RepertoireMethod.INITIALIZE,
        params_model=InitializeRequest,
        response_model=InitializeResponse,
        experimental_fields=frozenset(),
    ),
    RepertoireEntry(
        method=RepertoireMethod.INITIALIZED,
        params_model=InitializedNotification,
        response_model=None,
        experimental_fields=frozenset(),
    ),
    RepertoireEntry(
        method=RepertoireMethod.THREAD_START,
        params_model=ThreadStartRequest,
        response_model=ThreadStartResponse,
        experimental_fields=frozenset({"permissions"}),
    ),
    RepertoireEntry(
        method=RepertoireMethod.THREAD_RESUME,
        params_model=ThreadResumeRequest,
        response_model=ThreadResumeResponse,
        experimental_fields=frozenset({"path", "permissions"}),
    ),
    RepertoireEntry(
        method=RepertoireMethod.THREAD_READ,
        params_model=ThreadReadRequest,
        response_model=ThreadReadResponse,
        experimental_fields=frozenset(),
    ),
    RepertoireEntry(
        method=RepertoireMethod.THREAD_GOAL_SET,
        params_model=ThreadGoalSetRequest,
        response_model=EmptyResponse,
        experimental_fields=frozenset(),
    ),
    RepertoireEntry(
        method=RepertoireMethod.TURN_START,
        params_model=TurnStartRequest,
        response_model=TurnStartResponse,
        experimental_fields=frozenset(),
    ),
    RepertoireEntry(
        method=RepertoireMethod.TURN_STEER,
        params_model=TurnSteerRequest,
        response_model=TurnSteerResponse,
        experimental_fields=frozenset(),
    ),
    RepertoireEntry(
        method=RepertoireMethod.TURN_INTERRUPT,
        params_model=TurnInterruptRequest,
        response_model=EmptyResponse,
        experimental_fields=frozenset(),
    ),
)

REPERTOIRE: Final[MappingProxyType[RepertoireMethod, RepertoireEntry]] = (
    MappingProxyType({entry.method: entry for entry in _ENTRIES})
)
"""The closed mapping. Read-only, so no import can widen it after this module loads."""

if set(REPERTOIRE) != set(RepertoireMethod):
    raise RuntimeError(
        "every RepertoireMethod needs exactly one entry in the Repertoire"
    )

REQUIRES_EXPERIMENTAL_API: Final[bool] = any(e.experimental_fields for e in _ENTRIES)
"""Whether the handshake must opt in. Derived, so removing the last gated field turns it
off on its own rather than leaving the opt-in behind as a habit."""
