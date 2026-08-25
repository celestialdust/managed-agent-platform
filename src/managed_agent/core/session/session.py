"""A Session's creation facts, the states it can be in, and the shape of a change ask.

Everything here is fixed when the Session is created. There is deliberately no
current state field: state is computed from the Event Log on every read (see
projection.py), so a stored copy would be a second source free to disagree with
the log. The pod a Session is bound to is likewise absent — that binding is
mutable, and it lives with whatever owns placement rather than in a record whose
whole point is immutability.

SessionState is derived, never stored, which is why it lives beside the record
instead of inside it.

`UpdateSession` is here for the same reason `CreateSession` is — it is the parsed
form of one request body — and it names two fields that this platform refuses to
change. That is not a contradiction of the immutability above but a statement of
it: the refusal has to name a field to explain itself, so the field has to be
parseable before it can be refused.
"""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.registration.environment import EnvironmentId


class SessionState(StrEnum):
    """The states a fold over the Event Log can arrive at. Never persisted."""

    RUNNING = "running"
    SUSPENDED = "suspended"
    TAKEN_OVER = "taken_over"
    STOPPED = "stopped"

    def accepts_a_turn(self) -> bool:
        """Whether a Turn may start now.

        The state answers this rather than each caller deciding for itself, so
        the projection that computed the state and the component about to start a
        Turn cannot reach different verdicts about the same Session.
        """
        return self is SessionState.RUNNING


MAX_FILES_PER_SESSION = 16
"""How many uploaded files one Session may attach.

Not a capacity figure -- the pod's workspace bound is in bytes and is checked where the
rows are read. This is the coarse guard in front of it: without a count limit, a create
call naming ten thousand ids would have the control plane read ten thousand rows before
it could say no.

Sixteen because that is also `MAX_SKILLS_PER_AGENT`, and the two travel to the same pod
by different routes. Keeping one number for "how many small things may a Session carry"
means a reader has one figure to hold rather than two that happen to differ.
"""


class CreateSession(BaseModel):
    """What a tenant sends. Parsed once, at the boundary, into a typed value.

    Unknown fields are refused rather than ignored: a caller that misspells a budget
    field would otherwise believe it set one while the platform ran with the default.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    definition_id: DefinitionId
    definition_version: StrictInt | None = Field(default=None, ge=1)
    """Which revision to run: an exact one, or `None` for whichever is current.

    `None` is not a default standing in for revision 1. The two behave differently the
    moment a second revision exists, which is the only time either matters: `None`
    follows the edits, an integer does not.

    `StrictInt` rather than `int`, and the difference is load-bearing rather than
    fussy. `bool` is a subclass of `int`, so a plain `int` annotation accepts `true`
    and stores `1` -- measured on pydantic 2.13.4, `ge=1` does not refuse it either.
    A caller who sent `true` would then be pinned to revision 1 having written no pin
    at all, and would keep running revision 1 through every later edit without one
    error anywhere. `StrictInt` refuses `true`, `"2"` and `1.0`; `ge=1` refuses 0 and
    below; and this model's `extra="forbid"` refuses a misspelled `versoin` rather
    than reading it as unpinned. Every refusal therefore happens at the boundary, as
    a 400 naming the field.
    """

    environment_id: EnvironmentId
    """The registered sandbox shape this Session runs in. Required, and not defaulted.

    A default would be a second, unnamed shape: the whole point of naming one is that
    twenty Sessions on one shape are one registered thing rather than twenty
    compilations nothing says are the same. There is no field here for an image, a
    denied path or a permission profile, so a create call cannot restate the shape --
    `extra="forbid"` refuses one that tries.
    """

    file_ids: tuple[UUID, ...] = ()
    """Files already uploaded that this Session's agent should be able to read.

    Named at creation and never afterwards, which is what makes them the Session's:
    a Session whose file set could change mid-run is one whose earlier Turns and later
    Turns saw different worlds, with nothing in the record saying when it moved.

    `UUID` and not `FileId`, for the same reason `SkillAttachment.skill_id` is a plain
    `UUID`: the id type is named in the module that owns the store, and `core` does not
    import `control`. The conversion happens at the boundary, once.

    Bounded at parse time by count and at placement by total bytes, and the two bound
    different things. This one keeps a create call from asking the control plane to
    fetch an unbounded number of objects before it can refuse; the byte budget is what
    the pod's workspace can actually hold, and it needs the rows read to be checked.
    """

    grant: frozenset[str] = frozenset()
    scope: dict[str, str] = Field(default_factory=dict)
    budget_minor_units: int = Field(gt=0)
    budget_currency: str = Field(min_length=3, max_length=3)
    retention_days: int = Field(gt=0, le=3650)

    @model_validator(mode="after")
    def _refuse_more_files_than_a_session_carries(self) -> "CreateSession":
        """Refuse an over-long list, and refuse a repeated id.

        A repeat is refused rather than de-duplicated because the two readings differ in
        what the caller believes: someone who listed one id twice either meant two files
        and named the wrong one, or is looking at a list they have lost track of. Either
        way the honest answer names the id rather than quietly attaching it once.
        """
        if len(self.file_ids) > MAX_FILES_PER_SESSION:
            raise ValueError(
                f"a Session may attach at most {MAX_FILES_PER_SESSION} files and "
                f"this names {len(self.file_ids)}; they are written into the pod's "
                "workspace, which is what the limit pays for"
            )
        if len(set(self.file_ids)) != len(self.file_ids):
            raise ValueError("file_ids names the same file more than once")
        return self


class UpdateSession(BaseModel):
    """What a tenant may send to `POST /v1/sessions/{id}`. Every field is refused.

    The two fields are the ones the surface this platform mirrors lets a caller change
    mid-Session, translated into this platform's own vocabulary: a revised tool and MCP
    server list is a revised **Grant**, and a revised spend ceiling is
    `budget_minor_units`. Both are declared so that naming one is answered with a coded
    refusal that says *why* it cannot change, rather than the generic 400 that only
    says the field is unknown. The route that refuses them is the only reader of this
    type; nothing here writes anywhere.

    An empty body is the one accepted request, and it changes nothing by construction —
    a caller gets the Session's current state back, which is what "nothing about a
    Session is revisable" looks like as a request that succeeded.

    Anything else is refused by `extra="forbid"` as a 400 naming the field, and that is
    the right answer for the rest of the mirrored surface: a title, a metadata map and a
    vault list are fields no store in this platform has, so there is no concept here to
    refuse and nothing truthful to say beyond "no such field".

    Neither field carries a bound, and the absence is deliberate rather than an
    oversight. `CreateSession.budget_minor_units` is bounded because the number is
    stored and later read; this one is never stored, so a bound would be a check on a
    value nothing consumes — and it would answer a request with a 400 about the value
    when the honest answer is a refusal about the field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grant: frozenset[str] | None = None
    budget_minor_units: int | None = None


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """The creation facts. No field of this is ever rewritten.

    `scope` is a tuple of pairs rather than a mapping because the record is hashable and
    a mapping is not — and because a Scope that could be mutated in place would be a
    creation fact that changed after creation.

    The revision is stored, not looked up: an agent definition that changed under a
    running Session would silently change what that Session is, so what it resolved to
    at creation is pinned here.
    """

    id: SessionId
    tenant_id: TenantId
    definition_id: DefinitionId
    definition_revision: str
    grant: frozenset[str]
    scope: tuple[tuple[str, str], ...]
    budget_minor_units: int
    budget_currency: str
    retention_days: int
