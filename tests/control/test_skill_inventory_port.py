"""The inventory port's row types, and the stand-in that answers nothing.

No database. Everything here is a property of a frozen record or of a class that
refuses, and a real store would only make the assertions slower without making them
stronger.

The id these rows are addressed by is `core.skill.repository_skill_id`, and it is tested
in `tests/core/test_skill.py` beside its own definition -- including the check that
reads migration `0028`'s source to prove the two copies of its namespace and its fold
have not drifted. This file deliberately does not repeat those. A second copy of an
assertion is a second thing to update when the first one changes, and the copy that does
not get updated is the one that quietly stops meaning anything.

What the port's behaviour against a real table looks like is
`tests/adapters/test_skill_inventory.py`: the keyset staying total across a page
boundary that falls between the two origins, the tenant term making another tenant's
skill absent rather than filtered, and the assignment refusing when a stored id is not
the one this code computes.
"""

from __future__ import annotations

import uuid

import pytest

from managed_agent.control.skills.inventory import (
    NoSkillInventory,
    RepositorySkillRow,
    SkillInventory,
    SkillInventoryUnavailable,
    SkillOrigin,
    UploadedSkillRow,
)
from managed_agent.core.ids import SkillId, TenantId

_REPOSITORY = "git@github.com:acme/skills.git"
_SHA = "0" * 39 + "a"


def test_a_rows_origin_comes_from_its_type_rather_than_a_stored_field() -> None:
    """Each row reports its own origin, and neither can be built claiming the other.

    A stored `origin` column would be a second place for the answer to live, free to
    disagree with the shape of the row it sits on -- and the shape is what a caller
    branches on, since a repository skill cannot be deleted and cannot take a version.
    Deriving it from the type makes "a repository row claiming to be an upload" not a
    state that can be constructed.

    The fields are asserted too, because the split is the other half of the argument:
    the uploaded row carries a label and no checkout, the repository row carries a
    checkout and no label, and neither holds a field belonging to the other as a null.
    """
    uploaded = UploadedSkillRow(
        skill_id=SkillId(uuid.uuid4()),
        name="pdf",
        description="Build a PDF.",
        display_name="PDF tools",
    )
    from_repository = RepositorySkillRow(
        skill_id=SkillId(uuid.uuid4()),
        name="pdf",
        description="Build a PDF.",
        repository=_REPOSITORY,
        revision=_SHA,
    )

    assert uploaded.origin is SkillOrigin.UPLOAD
    assert from_repository.origin is SkillOrigin.REPOSITORY
    assert not hasattr(from_repository, "display_name")
    assert not hasattr(uploaded, "repository")


def test_the_stand_in_satisfies_the_port_it_stands_in_for() -> None:
    """So a method added to the port cannot be left off the stand-in.

    A stand-in missing a method is a `Platform` that raises `AttributeError` out of a
    route rather than the refusal the caller catches, and that surfaces as an
    unexplained 500 instead of as the wiring mistake it is.
    """
    assert isinstance(NoSkillInventory(), SkillInventory)


@pytest.mark.parametrize("asked", ["assign", "read", "page"])
async def test_the_stand_in_refuses_every_question_including_the_reads(
    asked: str,
) -> None:
    """All three refuse, and the two reads refusing is a decision rather than an
    oversight.

    An empty page from a deployment with no store is indistinguishable from a tenant who
    holds no skills, and a `None` from the single read is indistinguishable from an id
    that names nothing. Either would have a tenant told "you have no skills" when they
    have some -- which is worse than an error, because it is an answer and they would
    believe it. This is deliberately the opposite of `NoSessionThreads`, which answers
    empty: for a Session that never delegated, empty is true.
    """
    stand_in = NoSkillInventory()
    tenant = TenantId(uuid.uuid4())

    with pytest.raises(SkillInventoryUnavailable):
        if asked == "assign":
            await stand_in.assign_repository_ids(tenant, _REPOSITORY, _SHA, ["pdf"])
        elif asked == "read":
            await stand_in.repository_skill_at(tenant, SkillId(uuid.uuid4()))
        else:
            await stand_in.page(tenant, None, 25)
