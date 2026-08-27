"""The closed set of error codes this platform publishes, and their envelope.

Closed means a caller can branch on the code and be exhaustive: nothing outside this
enum ever reaches a caller, and in particular no Agent Runtime error name, internal
code or stack detail does.

A refusal is a value, not an exception, and `core` therefore names no web framework. The
surface returns one the way it returns any other answer — asking for a span that expired
is a question with an answer, not a crash — which also keeps the envelope usable by the
cross-tenant audit read, which is authorized differently and shares this set.

Two closed sets exist in this codebase and they answer different questions. This one
names why a call at the surface was refused. The marker cause set names why work inside
a Session was discarded or failed, and it lives with the markers that carry it. A
refusal is something the caller did; a marker cause is something that happened to the
agent.

The code is the contract and the message is not: consumers will branch on `code`, so
it is committed, while `message` is free text for whoever is reading a log and may be
reworded at any time. Anything a consumer needs to act on goes in `detail` under a
name, never in the sentence (ADR-013).
"""

from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(StrEnum):
    """Every refusal a caller can receive. Exhaustive for this API version."""

    SESSION_NOT_FOUND = "session.not_found"
    SESSION_NOT_ACCEPTING_TURNS = "session.not_accepting_turns"
    EVENT_RANGE_EXPIRED = "event_log.range_expired"
    DEFINITION_NOT_FOUND = "definition.not_found"
    DEFINITION_INVALID = "definition.invalid"
    DEFINITION_SKILLS_REVISION_UNREACHABLE = "definition.skills_revision_unreachable"
    DEFINITION_VERSION_ARCHIVED = "definition.version_archived"
    DEFINITION_SKILLS_REVISION_NOT_ACCEPTED = "definition.skills_revision_not_accepted"
    SKILL_EVAL_REGRESSED = "skill_eval.regressed"
    ENVIRONMENT_NOT_FOUND = "environment.not_found"
    TOOL_SERVER_INVALID = "tool.server_invalid"
    TOOL_NAME_CONFLICT = "tool.name_conflict"
    TOOL_NOT_GRANTED = "tool.not_granted"
    TOOL_OUT_OF_SCOPE = "tool.out_of_scope"
    TOOL_UNAVAILABLE = "tool.unavailable"
    TOOL_TIMED_OUT = "tool.timed_out"
    TURN_UNDELIVERABLE = "turn.undeliverable"
    BUDGET_EXHAUSTED = "budget.exhausted"
    TAKEOVER_HELD = "takeover.held"
    TAKEOVER_ALREADY_HELD = "takeover.already_held"
    REQUEST_INVALID = "request.invalid"
    # Added 2026-08-24. Each of these already reached callers, as an `HTTPException`
    # whose `detail` happened to carry a `code` key -- so this enum's own claim, that
    # "nothing outside this enum ever reaches a caller", was false for eight codes and
    # nothing could have noticed: an exhaustive `match` over the enum compiled fine
    # while the wire carried strings the enum had never heard of. They are here now,
    # and the raisers go through `refuse()`, which is what makes the claim checkable.
    REQUEST_TENANT_MISSING = "request.tenant_missing"
    REQUEST_TENANT_MALFORMED = "request.tenant_malformed"
    REQUEST_ROUTE_NOT_FOUND = "request.route_not_found"
    REQUEST_METHOD_NOT_ALLOWED = "request.method_not_allowed"
    AUDIT_PRINCIPAL_UNRESOLVED = "auth.audit_principal_unresolved"
    PAGINATION_CURSOR_INVALID = "pagination.cursor_invalid"
    # Added 2026-08-24 with the beta-header check. A caller naming a schema version this
    # build does not serve is refused rather than served the one version there is: the
    # header's whole function is to let a caller pin a wire shape, and answering a
    # different shape than the one asked for is the failure the header exists to
    # prevent.
    REQUEST_BETA_UNSUPPORTED = "request.beta_version_unsupported"
    # Added 2026-08-24. This one was held out of the enum deliberately and the reason
    # expired: `sessions.py` recorded that adding a member is a version event belonging
    # to whoever owns this file, and today that is the same change that added nine
    # others. It was the fourteenth hand-built envelope and the ninth code escaping a
    # set this module claimed was closed -- and it was the hardest to find, because its
    # own docstring explained why it was absent.
    SESSION_TURN_IN_FLIGHT = "session.turn_in_flight"
    # Two members and not one with the kind in `detail`, because the caller's next move
    # differs: a taken SERVER name means this registration was already made, and a taken
    # TOOL name means two servers offer one name to the same Grant. The formatted string
    # these replace -- `tool_registration.{kind}_name_already_registered` -- could not
    # live in a closed enum at all.
    TOOL_SERVER_NAME_TAKEN = "tool_registration.server_name_already_registered"
    TOOL_NAME_TAKEN = "tool_registration.tool_name_already_registered"
    # Added 2026-08-24 for wave 1, all eleven at once and by whoever holds the branch
    # rather than by the four agents that need them. A member added here is a change to
    # a set every route reads, and four writers adding to one enum is a conflict that
    # resolves cleanly in git and badly in meaning -- two of them would pick the same
    # spelling for different refusals, and nothing downstream could tell.
    # Retirement is terminal, so a reference to a retired thing is not a not-found: the
    # caller's id is right and the thing is deliberately unusable. 409 rather than 404,
    # because 404 invites the caller to check their id and they would find nothing wrong
    # with it.
    AGENT_ARCHIVED = "agent.archived"
    ENVIRONMENT_ARCHIVED = "environment.archived"
    # Optimistic concurrency on `POST /v1/agents/{id}`. Their reference: "the request
    # fails if it does not match the server's current version; omit to apply the update
    # unconditionally". It compares versions rather than content, so a caller resending
    # the values already stored is still refused when their version is stale -- the
    # refusal is about what they had read, not about what they are asking for.
    AGENT_VERSION_CONFLICT = "agent.version_conflict"
    # A delete refused because something still points at the thing. Separate codes per
    # resource rather than one `resource.in_use`, for the reason the two TOOL_*_TAKEN
    # members above give: the caller's next move differs. An Environment in use is freed
    # by ending the Sessions that name it; a file in use is freed the same way but the
    # file may also be an *output* of the Session holding it, which is not something the
    # caller can end early without losing it.
    ENVIRONMENT_IN_USE = "environment.in_use"
    FILE_IN_USE = "file.in_use"
    # Added 2026-08-24 with `POST /v1/sessions/{id}/resources`. A separate member rather
    # than `FILE_IN_USE`, which reads the other way round: that one refuses a *delete*
    # because a Session holds the file, and this refuses an *attach* because the Session
    # already holds a file of that name. Not `REQUEST_INVALID` either -- the request is
    # well-formed and would be accepted against a different Session, which is what 409
    # says and 400 does not. What the caller does next is rename nothing and attach
    # nothing: the pod's receiver renames atomically into one flat directory, so
    # honouring this would replace the earlier file with the later one and no record
    # anywhere would name the moment.
    RESOURCE_FILENAME_ATTACHED = "resource.filename_already_attached"
    # 404s for the three resources that had no way to be read back, so had no way to be
    # missing. `FILE_NOT_FOUND` replaces the `file_not_found` *reason string* in
    # `sessions.py`, which is a different thing in a different field and stays where it
    # is: that one says why a Session creation was refused, this one answers a read.
    FILE_NOT_FOUND = "file.not_found"
    SKILL_NOT_FOUND = "skill.not_found"
    SKILL_VERSION_NOT_FOUND = "skill_version.not_found"
    # 410 and not 404, for the reason `EVENT_RANGE_EXPIRED` gives: these existed, and
    # the difference between "gone" and "never here" is the whole reason the code
    # exists. It matters more here than there -- a tenant who deleted a file to honour a
    # deletion request needs the platform to say the deletion happened, and a 404 would
    # leave them unable to distinguish it from an id they mistyped.
    FILE_DELETED = "file.deleted"
    SKILL_DELETED = "skill.deleted"
    SKILL_VERSION_RETIRED = "skill_version.retired"
    # The two the thread surface needs. `THREAD_NOT_FOUND` is a 404 and deliberately
    # the same refusal a Session this caller cannot see gets, for the same reason: a
    # thread that belongs to another tenant's Session and one that never existed must
    # be indistinguishable, or the refusal itself answers a question about somebody
    # else's data. `THREAD_RUNNING` is a 409 and not a 400 -- the request is
    # well-formed and would be accepted a moment later, which is what 409 says.
    THREAD_NOT_FOUND = "thread.not_found"
    THREAD_RUNNING = "thread.running"
    # A repository-submitted skill has an id now, so it can be named by a request that
    # wants to delete it or add a version to it -- and neither is a thing that skill can
    # do. Its body is fixed by the commit a definition pins, so the commit *is* its
    # version, and removing the row would strand an already-registered definition while
    # the commit it names still exists. 409 and not 400: the request is well-formed and
    # names a skill that really is there, which is what distinguishes this from a
    # mistyped id, and 409 is what the surface already answers for a request refused by
    # the state of the thing it names rather than by its own shape.
    SKILL_OWNED_BY_COMMIT = "skill.owned_by_commit"
    # Added 2026-08-25. A Turn whose agent wrote a produced path it had already
    # delivered, with different bytes. Not `TURN_UNDELIVERABLE`, which is a 502 and
    # which this wore until then: 502 tells a caller the platform failed and the two
    # moves it invites -- retry, and report it -- are both wrong. Retrying re-runs an
    # agent that writes the same path again, and there is nothing to report. 409 and
    # the same reading `RESOURCE_FILENAME_ATTACHED` carries: the request was
    # well-formed and the state of the thing it names refused it. The refusal's
    # `detail` carries the path, because a Session that produced several files leaves
    # the caller no way to tell which one collided -- and knowing which is the whole of
    # the next move, which is to write it under a different name.
    OUTPUT_NOT_REVISABLE = "output.not_revisable"
    # Added 2026-08-25 with the tool-result route. A question a tool call put to the
    # Session, marked by whoever put it as one whose answer is a credential. Refused
    # rather than recorded: the answer to a tool-call question comes to rest in the
    # Event Log, on the tenant's retention clock, and in the Rollout copied out of the
    # pod -- two stores built to be read back and neither built to hold a secret. 409
    # and the reading `OUTPUT_NOT_REVISABLE` carries: the request is well-formed, and
    # what refuses it is the state of the thing it names. Its own member rather than
    # `REQUEST_INVALID` because the next move is specific and is not "fix your request"
    # -- the value goes into a Vault, which a tool already reads its credentials from.
    ELICITATION_SECRET_REFUSED = "elicitation.secret_refused"
    WEBHOOK_NOT_FOUND = "webhook.not_found"
    VAULT_NOT_FOUND = "vault.not_found"
    CREDENTIAL_NOT_FOUND = "credential.not_found"
    # A vault name and a credential name are both components of the ref a Tool
    # registration names, so a second row of either name would make one ref address two
    # credentials -- resolved inside the Tool Gateway, where nothing can ask which was
    # meant. 409 rather than a silent second row, and named per kind for the reason
    # `_NAME_TAKEN` gives on the Tool surface: the two mean different things to fix.
    VAULT_NAME_TAKEN = "vault.name_already_registered"
    CREDENTIAL_NAME_TAKEN = "credential.name_already_registered"
    # Archived is read-only and there is no unarchive, so a write against one is
    # refused rather than applied to a row a tenant believes is retired. Distinct from
    # NOT_FOUND because the row is right there and the tenant can list it.
    VAULT_ARCHIVED = "vault.archived"
    VAULT_FULL = "vault.full"
    OVERLOADED = "platform.overloaded"
    INTERNAL = "platform.internal"


def _status(code: ErrorCode) -> int:
    """The HTTP status one code is returned with.

    A match with an `assert_never` tail rather than a dict literal, because mypy
    --strict then fails the build when a member is added with no arm — a code that
    reached a caller with a default status would be a silent second contract.

    410 for an expired range and not 404: those events existed, and the difference
    between "gone" and "never here" is the whole reason the code exists.
    """
    match code:
        case (
            ErrorCode.SESSION_NOT_FOUND
            | ErrorCode.DEFINITION_NOT_FOUND
            | ErrorCode.WEBHOOK_NOT_FOUND
            | ErrorCode.ENVIRONMENT_NOT_FOUND
            | ErrorCode.FILE_NOT_FOUND
            | ErrorCode.SKILL_NOT_FOUND
            | ErrorCode.SKILL_VERSION_NOT_FOUND
            | ErrorCode.THREAD_NOT_FOUND
            | ErrorCode.VAULT_NOT_FOUND
            | ErrorCode.CREDENTIAL_NOT_FOUND
            | ErrorCode.REQUEST_ROUTE_NOT_FOUND
        ):
            return 404
        case (
            ErrorCode.EVENT_RANGE_EXPIRED
            | ErrorCode.FILE_DELETED
            | ErrorCode.SKILL_DELETED
            | ErrorCode.SKILL_VERSION_RETIRED
        ):
            return 410
        case ErrorCode.REQUEST_METHOD_NOT_ALLOWED:
            # 405, which the published status table has no row for. Kept because it is
            # the status the router already answers with and it is more informative than
            # collapsing it into 400: a caller who used the wrong verb on a real path
            # learns something a generic bad-request does not tell them.
            return 405
        case ErrorCode.AUDIT_PRINCIPAL_UNRESOLVED:
            return 401
        case (
            ErrorCode.DEFINITION_INVALID
            | ErrorCode.DEFINITION_SKILLS_REVISION_UNREACHABLE
            | ErrorCode.DEFINITION_SKILLS_REVISION_NOT_ACCEPTED
            | ErrorCode.SKILL_EVAL_REGRESSED
            | ErrorCode.TOOL_SERVER_INVALID
            | ErrorCode.REQUEST_INVALID
            | ErrorCode.REQUEST_TENANT_MISSING
            | ErrorCode.REQUEST_TENANT_MALFORMED
            | ErrorCode.PAGINATION_CURSOR_INVALID
            | ErrorCode.REQUEST_BETA_UNSUPPORTED
        ):
            # 400 and not 422, changed 2026-08-24. 422 is the more precise reading of
            # a well-formed body a handler will not act on, and it is not the status
            # this API's consumers are written against: the Managed Agents surface this
            # one is modelled on answers 400 for every malformed or unactionable
            # request, and an SDK built for that treats an unexpected 422 as a
            # transport fault rather than a refusal it can read a code out of.
            return 400
        case (
            ErrorCode.SESSION_NOT_ACCEPTING_TURNS
            | ErrorCode.SESSION_TURN_IN_FLIGHT
            | ErrorCode.TOOL_NAME_CONFLICT
            | ErrorCode.TAKEOVER_HELD
            | ErrorCode.TAKEOVER_ALREADY_HELD
            | ErrorCode.DEFINITION_VERSION_ARCHIVED
            | ErrorCode.TOOL_SERVER_NAME_TAKEN
            | ErrorCode.TOOL_NAME_TAKEN
            | ErrorCode.AGENT_ARCHIVED
            | ErrorCode.ENVIRONMENT_ARCHIVED
            | ErrorCode.VAULT_ARCHIVED
            | ErrorCode.VAULT_FULL
            | ErrorCode.VAULT_NAME_TAKEN
            | ErrorCode.CREDENTIAL_NAME_TAKEN
            | ErrorCode.AGENT_VERSION_CONFLICT
            | ErrorCode.ENVIRONMENT_IN_USE
            | ErrorCode.FILE_IN_USE
            | ErrorCode.RESOURCE_FILENAME_ATTACHED
            | ErrorCode.THREAD_RUNNING
            | ErrorCode.SKILL_OWNED_BY_COMMIT
            | ErrorCode.OUTPUT_NOT_REVISABLE
            | ErrorCode.ELICITATION_SECRET_REFUSED
        ):
            return 409
        case ErrorCode.TOOL_NOT_GRANTED | ErrorCode.TOOL_OUT_OF_SCOPE:
            return 403
        case ErrorCode.BUDGET_EXHAUSTED:
            return 402
        case ErrorCode.OVERLOADED:
            # 529 and not 429, changed 2026-08-24. The two mean different things to a
            # caller deciding whether to retry: 429 says this caller asked too often
            # and a quota governs when to come back, and 529 says the service itself
            # has no capacity right now and the caller did nothing wrong. This platform
            # publishes no per-caller quota yet, so every 429 it could emit would be
            # the second thing wearing the first thing's status -- and it holds 429
            # free for the rate limiter that does not exist yet.
            return 529
        case ErrorCode.TOOL_UNAVAILABLE | ErrorCode.TURN_UNDELIVERABLE:
            return 502
        case ErrorCode.TOOL_TIMED_OUT:
            return 504
        case ErrorCode.INTERNAL:
            return 500
        case _ as unreachable:
            assert_never(unreachable)


STATUS_FOR: Final[MappingProxyType[ErrorCode, int]] = MappingProxyType(
    {code: _status(code) for code in ErrorCode}
)
"""Every code's status, built once at import so no caller can reach an unmapped one."""


class ErrorEnvelope(BaseModel):
    """The body of every refusal, whatever refused it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ErrorCode
    message: str = Field(min_length=1)
    detail: dict[str, str | int] = Field(default_factory=dict)


class PublicErrorType(StrEnum):
    """The coarse error classes a consumer's SDK branches on.

    Eight values, and they are not ours: they are the classes the Managed Agents surface
    this API is modelled on publishes, so a client generated against those docs can
    classify a refusal from here without knowing this platform's own vocabulary.

    They sit *beside* `ErrorCode` rather than replacing it. `ErrorCode` has 24 members
    and answers "what exactly was wrong"; this has 8 and answers "what kind of thing was
    wrong". Collapsing the first into the second would delete every distinction a caller
    can act on -- `session.not_accepting_turns` and `tool.name_conflict` are both
    `invalid_request_error` and want opposite responses -- and publishing only the first
    means a client that has never heard of this platform cannot tell a retryable failure
    from a permanent one. The envelope carries both for that reason.
    """

    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found_error"
    REQUEST_TOO_LARGE = "request_too_large"
    RATE_LIMIT = "rate_limit_error"
    API = "api_error"
    OVERLOADED = "overloaded_error"


def _public_type(code: ErrorCode) -> PublicErrorType:
    """The coarse class one code belongs to.

    A match with an `assert_never` tail for the same reason `_status` is one: a code
    added with no arm fails the build rather than reaching a caller under a default
    class, and a default here would be `invalid_request_error` -- which would tell a
    client to stop retrying a fault that was ours and temporary.

    `api_error` covers the codes whose HTTP status is 502, 504 or 500. Those are
    failures inside a Turn rather than anything the caller did, and their status already
    says so; the class exists to tell a client that a retry is reasonable.
    """
    match code:
        case (
            ErrorCode.SESSION_NOT_FOUND
            | ErrorCode.DEFINITION_NOT_FOUND
            | ErrorCode.WEBHOOK_NOT_FOUND
            | ErrorCode.ENVIRONMENT_NOT_FOUND
            | ErrorCode.FILE_NOT_FOUND
            | ErrorCode.SKILL_NOT_FOUND
            | ErrorCode.SKILL_VERSION_NOT_FOUND
            | ErrorCode.THREAD_NOT_FOUND
            | ErrorCode.VAULT_NOT_FOUND
            | ErrorCode.CREDENTIAL_NOT_FOUND
            | ErrorCode.REQUEST_ROUTE_NOT_FOUND
        ):
            return PublicErrorType.NOT_FOUND
        case ErrorCode.TOOL_NOT_GRANTED | ErrorCode.TOOL_OUT_OF_SCOPE:
            return PublicErrorType.PERMISSION
        case ErrorCode.AUDIT_PRINCIPAL_UNRESOLVED:
            return PublicErrorType.AUTHENTICATION
        case ErrorCode.OVERLOADED:
            return PublicErrorType.OVERLOADED
        case (
            ErrorCode.TOOL_UNAVAILABLE
            | ErrorCode.TOOL_TIMED_OUT
            | ErrorCode.TURN_UNDELIVERABLE
            | ErrorCode.INTERNAL
        ):
            return PublicErrorType.API
        case (
            ErrorCode.SESSION_NOT_ACCEPTING_TURNS
            | ErrorCode.EVENT_RANGE_EXPIRED
            | ErrorCode.DEFINITION_INVALID
            | ErrorCode.DEFINITION_SKILLS_REVISION_UNREACHABLE
            | ErrorCode.DEFINITION_SKILLS_REVISION_NOT_ACCEPTED
            | ErrorCode.DEFINITION_VERSION_ARCHIVED
            | ErrorCode.SKILL_EVAL_REGRESSED
            | ErrorCode.TOOL_SERVER_INVALID
            | ErrorCode.SESSION_TURN_IN_FLIGHT
            | ErrorCode.TOOL_NAME_CONFLICT
            | ErrorCode.BUDGET_EXHAUSTED
            | ErrorCode.TAKEOVER_HELD
            | ErrorCode.TAKEOVER_ALREADY_HELD
            | ErrorCode.REQUEST_INVALID
            | ErrorCode.REQUEST_TENANT_MISSING
            | ErrorCode.REQUEST_TENANT_MALFORMED
            | ErrorCode.REQUEST_METHOD_NOT_ALLOWED
            | ErrorCode.PAGINATION_CURSOR_INVALID
            | ErrorCode.REQUEST_BETA_UNSUPPORTED
            | ErrorCode.TOOL_SERVER_NAME_TAKEN
            | ErrorCode.TOOL_NAME_TAKEN
            | ErrorCode.AGENT_ARCHIVED
            | ErrorCode.ENVIRONMENT_ARCHIVED
            | ErrorCode.VAULT_ARCHIVED
            | ErrorCode.VAULT_FULL
            | ErrorCode.VAULT_NAME_TAKEN
            | ErrorCode.CREDENTIAL_NAME_TAKEN
            | ErrorCode.AGENT_VERSION_CONFLICT
            | ErrorCode.ENVIRONMENT_IN_USE
            | ErrorCode.FILE_IN_USE
            | ErrorCode.RESOURCE_FILENAME_ATTACHED
            | ErrorCode.FILE_DELETED
            | ErrorCode.SKILL_DELETED
            | ErrorCode.SKILL_VERSION_RETIRED
            | ErrorCode.THREAD_RUNNING
            | ErrorCode.SKILL_OWNED_BY_COMMIT
            | ErrorCode.OUTPUT_NOT_REVISABLE
            | ErrorCode.ELICITATION_SECRET_REFUSED
        ):
            return PublicErrorType.INVALID_REQUEST
        case _ as unreachable:
            assert_never(unreachable)


PUBLIC_TYPE_FOR: Final[MappingProxyType[ErrorCode, PublicErrorType]] = MappingProxyType(
    {code: _public_type(code) for code in ErrorCode}
)
"""Every code's coarse class, built once so no caller reaches an unmapped one."""


class PublicError(BaseModel):
    """The inner object of a refusal: the class, the sentence, and our own code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: PublicErrorType
    message: str = Field(min_length=1)
    code: ErrorCode
    detail: dict[str, str | int] = Field(default_factory=dict)


class PublicErrorEnvelope(BaseModel):
    """The body of every refusal this API returns, in the shape a consumer expects.

    `{"type": "error", "error": {...}, "request_id": ...}`. The outer `type` is the
    constant string `error` and carries no information: it exists because the surface
    this one is modelled on emits it, and a client that switches on the top-level
    `type` of every response body would otherwise fall through on ours.

    `request_id` is here and is not optional. It is the only field in this envelope
    about *this call* rather than about a class of failure, and it is what makes a
    report actionable -- "your API returned 400" is unanswerable, and "request
    req_0a1b... returned 400" is a log lookup. It is generated per request by
    middleware rather than per refusal here, because the same id has to appear on the
    success path too or a caller cannot correlate the two.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["error"] = "error"
    error: PublicError
    request_id: str = Field(min_length=1)


def public_envelope(
    code: ErrorCode,
    message: str,
    request_id: str,
    detail: dict[str, str | int] | None = None,
) -> PublicErrorEnvelope:
    """One refusal rendered for a consumer, with the class derived, never passed in.

    The class is looked up from the code rather than accepted as an argument, so no
    call site can publish a code and a class that disagree -- a `not_found_error`
    carrying `session.not_accepting_turns` would be a body no client could act on
    coherently, and it is the mistake a two-argument signature invites on every call.
    """
    return PublicErrorEnvelope(
        error=PublicError(
            type=PUBLIC_TYPE_FOR[code],
            message=message,
            code=code,
            detail=dict(detail or {}),
        ),
        request_id=request_id,
    )
