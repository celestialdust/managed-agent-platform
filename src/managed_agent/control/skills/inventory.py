"""Every skill a tenant holds, across both doors that register one.

This platform has two write doors for skills and only one of them minted an id. An
upload gets a `skill` row with a primary key, and everything a tenant can do afterwards
is addressed by it. A repository submission writes `skill_repository_file` rows keyed by
`(tenant, repository, revision, name)` and had no id at all -- so a team whose CI
submitted its skills could register them and then had **no way to enumerate, read or
retire any of them.** The listing served the uploaded half and said so in its own
docstring; the read and the delete took a uuid those skills did not have.

Migration `0028` closed the id half by assigning every repository skill a `uuid5` over
the four columns that key it, and `core.skill.repository_skill_id` is the one
function that computes it. This module is the read half: one port that answers across
both origins, so a caller pages one collection and asks one question.

**A discriminated pair rather than one row with nullable columns.** The objection on
record against paging the two together was that it would need a row with a nullable id
and a nullable `(repository, revision)` pair -- two shapes in one payload. That
objection was right about the shape and is answered by using two types instead of one:
an uploaded row cannot carry a revision and a repository row cannot lack one, so neither
state a reader has to check for is representable. The route matches on the type it got.

Held apart from `skill_registry.py` rather than folded into `SkillStore`. That module
answers "what does this definition resolve to", which is a question about delivery and
is asked on the placement path. This answers "what does this tenant hold", which is a
question about inventory and is asked by a listing. Two reasons to change, so two
modules -- and the split is what let this be built while `skill_registry.py` was being
changed for something else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from managed_agent.core.ids import SkillId, TenantId


class SkillOrigin(StrEnum):
    """Which door a skill arrived by.

    Published on every listing row and every read, because the origin decides which
    operations apply: a repository skill cannot be deleted and cannot take a version,
    since its body is fixed by the commit a definition pins. Left for a caller to infer
    from the presence of a `repository` field, that fact would be carried by the absence
    of something rather than by a value, and a client written against one row shape
    would read a missing key as a missing skill.

    Deliberately not Anthropic's `source`, which is a different axis with different
    values: `source` says whether a skill is ours or from their pre-built catalogue, and
    every skill this platform holds is `custom` under it. Two fields, two meanings; the
    one time they were spelled with one word, a reader had to guess which was meant.
    """

    UPLOAD = "upload"
    REPOSITORY = "repository"


@dataclass(frozen=True, slots=True)
class UploadedSkillRow:
    """One skill that arrived through `POST /v1/skills`, as a listing shows it.

    No `body`. It is the one column that can be 32 KiB, no listing shows it, and a page
    carrying bodies is a page nobody can walk -- `name` and `description` were read out
    of the document at upload and stored as columns precisely so this read opens none.
    """

    skill_id: SkillId
    name: str
    description: str
    display_name: str | None

    @property
    def origin(self) -> SkillOrigin:
        """`UPLOAD`, from the type rather than from a stored column.

        A field would be a second place for the answer to live and a chance for a row to
        claim an origin its type contradicts.
        """
        return SkillOrigin.UPLOAD


@dataclass(frozen=True, slots=True)
class RepositorySkillRow:
    """One skill that arrived through `POST /v1/skills/repository`, as a listing shows
    it.

    Carries the checkout it came from, because that pair is how a definition names it
    and is the only thing that distinguishes two submissions of the same skill at
    different commits. Both are non-optional: a repository skill without a revision is
    not a state this platform can hold, and making it unrepresentable is cheaper than
    checking for it at every read.

    No `display_name`. The label is a field of the upload door's request, and a checkout
    submits a directory rather than a labelled skill -- so this is not a null waiting to
    be filled in, it is a field that does not apply.
    """

    skill_id: SkillId
    name: str
    description: str
    repository: str
    revision: str

    @property
    def origin(self) -> SkillOrigin:
        """`REPOSITORY`, from the type rather than from a stored column."""
        return SkillOrigin.REPOSITORY


SkillRow = UploadedSkillRow | RepositorySkillRow
"""One row of the combined inventory, of whichever origin.

A union rather than a base class with optional fields, so a caller that forgets one arm
fails to type-check rather than reading a `None` that means "not applicable" in one case
and "not set" in the other.
"""


@dataclass(frozen=True, slots=True)
class RepositorySkillHeld:
    """One repository skill read back in full, body included.

    The read surface's answer for a single skill, which is the one place a body is worth
    fetching: the caller asked for exactly this skill and there is one of it.
    """

    skill_id: SkillId
    name: str
    description: str
    body: str
    repository: str
    revision: str


class SkillInventoryUnavailable(Exception):
    """This deployment cannot answer inventory questions, so none is answered falsely.

    Raised by the refusing stand-in below and never by a real implementation. A caller
    catching it is a caller that would otherwise have shown a tenant an empty collection
    that the tenant cannot distinguish from holding nothing.
    """


@runtime_checkable
class SkillInventory(Protocol):
    """Read across both doors, and assign the ids the repository door needs.

    Three methods rather than one, because they are asked at three different moments: a
    submission assigns, a read resolves one id, and a listing walks. Folding the walk
    and the single read together would mean paging a collection to answer a lookup by
    primary key, which is the cost the read exists to avoid."""

    async def assign_repository_ids(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        names: Sequence[str],
    ) -> tuple[tuple[str, SkillId], ...]:
        """`(name, id)` for every skill of this checkout, minting the missing ones.

        Idempotent per `(tenant, repository, revision, name)`: a resubmission returns
        the ids already assigned and writes nothing, which is the property the file
        table already has and the reason the id is a `uuid5` of the key rather than a
        fresh `uuid4`. Every name asked for comes back, not only the newly assigned -- a
        caller that submitted a checkout wants the whole checkout's ids, and cannot tell
        a retried submission from a first one by the shape of the answer."""
        ...

    async def repository_skill_at(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> RepositorySkillHeld | None:
        """One repository skill by the id `0028` assigned it, or None for no such row.

        None rather than a refusal, so the caller can try the uploaded half before
        deciding a skill does not exist. An id that names neither is one 404, because a
        caller holding an id has no way to know which door minted it.
        """
        ...

    async def page(
        self,
        tenant_id: TenantId,
        after: tuple[str, SkillId] | None,
        limit: int,
    ) -> tuple[SkillRow, ...]:
        """One page across both origins, ordered by name then id.

        The keyset is `(name, id)` over the union rather than a page from each side
        merged by the caller. Merging two pages cannot be made total without
        over-fetching both, and the failure mode of getting it wrong is a listing that
        silently drops one origin's tail -- a caller walks to what looks like the end
        and has seen part of what it holds, with nothing saying so.

        `limit` is the page size the caller wants. Whether to ask for one row more, to
        learn if another page exists, is the caller's decision and not this port's: the
        probe row is a fact about how a cursor surface is built, and a store that added
        one silently would hand back a page longer than was asked for.
        """
        ...


class NoSkillInventory:
    """Refuses every question. What a `Platform` built without a database holds.

    A refusing default rather than `None`, so no caller tests the field before using it
    and none can forget to. Every method raises, including the two reads -- and that is
    the decision rather than an oversight. An empty page from here would be
    indistinguishable from a tenant holding no skills, and a `None` from the single read
    indistinguishable from an id that names nothing, so a deployment wired without a
    store would answer "you have no skills" to a tenant who has some. A refusal is the
    only answer that is not a lie.
    """

    async def assign_repository_ids(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        names: Sequence[str],
    ) -> tuple[tuple[str, SkillId], ...]:
        raise SkillInventoryUnavailable(
            f"this deployment holds no skill inventory, so the {len(names)} skill(s) "
            f"of {repository} at {revision} cannot be assigned ids"
        )

    async def repository_skill_at(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> RepositorySkillHeld | None:
        raise SkillInventoryUnavailable(
            f"this deployment holds no skill inventory, so whether skill {skill_id} "
            "exists cannot be answered"
        )

    async def page(
        self,
        tenant_id: TenantId,
        after: tuple[str, SkillId] | None,
        limit: int,
    ) -> tuple[SkillRow, ...]:
        raise SkillInventoryUnavailable(
            "this deployment holds no skill inventory, so a listing would report an "
            "empty collection rather than an unanswerable question"
        )
