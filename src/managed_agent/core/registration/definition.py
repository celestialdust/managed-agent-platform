"""What a declarative agent definition may say, and how a revision is pinned.

Format and revision-pin live together because both answer one question -- what a
definition resolves to at the moment a Session is created. Splitting them would put the
resolution in two files that have to agree.

`model` is deliberately unvalidated against any catalog. A model name is a routing key,
not a contract: a capability mismatch surfaces as a failed Turn naming the missing
feature, and a catalog checked here would go stale in a way indistinguishable from the
failure it exists to prevent (ADR-010).

`skills_revision` is a full 40-character git object id, never a branch or a tag. A
moving reference would let a definition change under a running Session, which is the one
thing the version pin exists to stop.

The revision is checked in two places on purpose, and they are not two copies of the
rule: both read `SKILLS_REVISION_PATTERN`, so they cannot diverge. The annotation is
what makes the constraint part of the *type* -- it travels into the OpenAPI schema and
into any other model that declares a field of `SkillsRevision`. The validator is what
makes the refusal *explain itself*: a bare pattern mismatch tells the reader their
string is the wrong shape, which they can already see, and leaves them looking for the
switch that turns the check off. It runs in `before` mode because pydantic's own
annotated constraints are part of the core schema and reject the value before any
`after` validator is reached -- an `after` validator here would never run for a single
value it was written to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from managed_agent.core.registration.skill import MAX_SKILLS_PER_AGENT, SkillAttachment

SKILLS_REVISION_PATTERN = r"^[0-9a-f]{40}$"
"""A full, lowercase, 40-character git object id and nothing else.

Anchored at both ends, so a sha with a branch name appended is not a match. Lowercase
only, because that is the one form `git rev-parse` prints -- accepting both would make
the same commit two distinct pinned strings, and the revision is compared as text.
"""

_SHA1 = re.compile(SKILLS_REVISION_PATTERN)

SkillsRevision = Annotated[str, Field(pattern=SKILLS_REVISION_PATTERN)]
"""A pinned skills commit. Carries its constraint wherever a pydantic model declares it.

Written at a call site this is a plain `str` and asserts nothing -- `Annotated` is
erased at runtime, so `SkillsRevision(x)` is `str(x)`. Anything refusing a bad revision
outside a model boundary matches `_SHA1` explicitly.
"""

DEFINITION_NOT_FOUND = "definition.not_found"
"""The refusal code for a definition that does not resolve for the asking tenant.

A literal here rather than a member of the platform's closed `ErrorCode` set, because
that set does not exist yet -- it arrives with the slice that builds the error
envelope, which runs after this one. It is a string in one place so that folding it
into the closed set later is one edit, and so a reader grepping for the code finds a
definition rather than a scattering of quoted literals in route handlers.

Both "no such definition" and "another tenant's definition" answer with this one code.
Distinguishing them would tell anyone holding an id whether it exists.
"""

_MOVING_REFERENCE = (
    "skills_revision must be a full 40-character lowercase commit id, not a branch or "
    "a tag: a moving reference would let a definition change under a running Session, "
    "which is what pinning a revision exists to prevent"
)


@dataclass(frozen=True, slots=True)
class VersionFact:
    """One registered revision of one agent definition, and whether it is retired.

    Registration and retirement are two different tables, and this is the pair read
    together so nothing can observe them out of step: a revision that exists and a
    revision that may still be started are two questions with one answer here.

    Retirement is a property of a revision rather than of the whole definition. An
    agent is retired by retiring each of its revisions, which is why there is no
    definition-level flag anywhere.
    """

    revision: int
    archived: bool


class MultiAgentPosture(BaseModel):
    """Whether this agent may delegate, and how deep the tree of delegates may go.

    Depth 1 is a single agent that delegates to nothing, which is why it is the default
    and the floor rather than 0: there is always one agent. The ceiling is a platform
    bound, not a tenant preference -- each level multiplies the pods and the spend a
    single Session can reach, so an unbounded depth is an unbounded bill.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    max_depth: int = Field(default=1, ge=1, le=4)


class AgentDefinition(BaseModel):
    """A definition as a tenant submits it. Parsed, not validated.

    Frozen and `extra="forbid"` for the same reason, one turned outward and one inward.
    Forbidding unknown fields means a misspelled key is refused rather than dropped: a
    tenant who writes `multiagnet` would otherwise be told nothing while the platform
    ran the default. Freezing means the value cannot drift after it is parsed, which is
    what lets a stored revision number keep describing what is actually running.

    `tool_servers` is a set because the order a tenant lists servers in carries no
    meaning, and duplicates in the submission are the same grant twice. `skills` is a
    set for exactly the same two reasons.

    A definition names skills twice over and the two are not redundant. `skills`
    attaches one that was uploaded here and is addressed by an id. `skills_repository`
    and `skills_revision` name a checkout whose whole skill directory is attached, and
    which skills that is depends on what was submitted for that pair rather than on
    anything in this document. Both resolve at the moment a Session is created, and a
    definition may use either, both, or -- with nothing submitted against its pinned
    revision -- neither.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    instructions: str = Field(min_length=1, max_length=64_000)
    model: str = Field(min_length=1, max_length=128)
    skills_repository: str = Field(min_length=1)
    skills_revision: SkillsRevision
    skills: frozenset[SkillAttachment] = Field(
        default=frozenset(), max_length=MAX_SKILLS_PER_AGENT
    )
    tool_servers: frozenset[str] = frozenset()
    multiagent: MultiAgentPosture = MultiAgentPosture()

    @field_validator("skills_revision", mode="before")
    @classmethod
    def _reject_moving_reference(cls, value: object) -> object:
        """Refuse anything but a full commit id, and say why rather than how.

        Non-string input is passed through untouched for the core schema to refuse with
        its own type error -- this validator knows about references, not about types,
        and a message about branches in answer to `None` would only mislead.
        """
        if isinstance(value, str) and not _SHA1.match(value):
            raise ValueError(_MOVING_REFERENCE)
        return value

    @field_validator("name", "instructions", "model", mode="before")
    @classmethod
    def _refuse_whitespace_padded_strings(cls, value: object) -> object:
        """Refuse strings with leading or trailing whitespace rather than strip them.

        A padded value stored as-is would fail downstream when rendered into config.toml
        and compared against routing tables -- the failure surfaces far from its cause,
        as a Session that will not compile. Refusing at the boundary surfaces the error
        immediately and ensures what is stored is exactly what was sent.

        Non-string input is passed through for the core schema to refuse with its own
        type error.
        """
        if isinstance(value, str) and value != value.strip():
            raise ValueError(
                "field must not have leading or trailing whitespace: a padded value "
                "stored as-is would compile into a malformed config and break Session "
                "creation far from where it was registered"
            )
        return value
