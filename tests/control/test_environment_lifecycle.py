"""Listing, editing, retiring and deleting a registered shape, over the real routes.

Tier 1: in-memory ports, the real routers, no cluster and no container.

The harness is imported from `test_environment_reference` rather than rebuilt -- same
`Platform`, same in-memory store, same tenant header -- so the four operations here are
exercised against the same store the create and the read are, and a change to how a row
is keyed surfaces in one place.

Four claims, one per operation, and each is asserted against what the store holds
afterwards rather than against a status code alone:

- **A list is one row per id**, latest revision each, this tenant's only, and a walk
  over it covers every shape exactly once even while somebody is editing.
- **An edit appends.** The response carries the new revision, a refused edit leaves the
  id meaning what it meant, and a retired shape takes no edit at all.
- **A retirement is terminal and idempotent.** The second call answers with the FIRST
  retirement's timestamp, because a retried call must not claim the shape stopped being
  referenceable later than it did.
- **A delete is real, and refused while anything holds the shape.** The refusal carries
  the count; a second delete is a 404 and not a second 200, because the row is gone and
  a 200 would claim this call removed something.
"""

from dataclasses import replace
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

# The reference file's harness, imported rather than rebuilt, for the reason
# `test_environment_shapes_the_sandbox_can_build.py` gives: every name is module-level
# and public there, this file is in the same regression surface, and `tests/` carries no
# `__init__.py`, so pytest's own sys.path entry is what makes the bare name work.
from test_environment_reference import (
    IMAGE,
    OTHER_IMAGE,
    SECRETS,
    FakeEnvironmentStore,
    a_harness,
    an_environment,
    build_app,
    caller,
)

from managed_agent.control.catalog.environments import parse_environment
from managed_agent.core.errors import STATUS_FOR, ErrorCode
from managed_agent.core.ids import TenantId
from managed_agent.core.registration.environment import (
    Environment,
    EnvironmentId,
    new_environment_id,
)


def _masked(response: httpx.Response) -> str:
    """One response's body with its per-request id masked out.

    The id is minted per request, before any handler decides anything, so it reports
    nothing about either response and has to come out before two refusals can be
    compared byte for byte. Left in, every pair differs on it and the comparison loses
    the power to detect what it exists to detect.
    """
    body: dict[str, object] = response.json()
    return response.text.replace(str(body["request_id"]), "<request>")


def _an_edit(
    environment_id: EnvironmentId,
    tenant: TenantId,
    *,
    name: str = "analysis-revised",
    image: str = OTHER_IMAGE,
    domains: tuple[str, ...] = (),
) -> Environment:
    """The same id, a different shape, parsed the way the route parses one."""
    return parse_environment(
        environment_id=environment_id,
        tenant_id=tenant,
        name=name,
        runtime_image=image,
        denied_paths=(SECRETS,),
        allowed_domains=domains,
    )


def _a_body(**overrides: Any) -> dict[str, Any]:
    """A registration body, so a case names only the field it is about."""
    return {
        "name": "analysis",
        "runtime_image": IMAGE,
        "denied_paths": [SECRETS],
        "allowed_domains": [],
        **overrides,
    }


async def _registered(
    store: FakeEnvironmentStore, tenant: TenantId, how_many: int
) -> list[Environment]:
    """`how_many` shapes for one tenant, oldest first.

    Oldest first, so a case can say "the newest" and "the oldest" without counting
    backwards: the store's position counter increments per insert, and the list order is
    the insert order.
    """
    made = []
    for _ in range(how_many):
        environment = an_environment(tenant)
        await store.insert(environment)
        made.append(environment)
    return made


# --------------------------------------------------------------------------------------
# GET /v1/environments
# --------------------------------------------------------------------------------------


async def test_a_list_answers_this_tenants_shapes_newest_first() -> None:
    """`data` and `next_page` are Anthropic's names for these two fields, and a walk
    that is over says so with a null rather than with a token leading nowhere."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    made = await _registered(store, tenant, 3)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        listed = await client.get("/v1/environments")

    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert [row["id"] for row in body["data"]] == [
        str(one.id) for one in reversed(made)
    ]
    assert body["next_page"] is None
    assert {row["revision"] for row in body["data"]} == {1}
    assert {row["archived_at"] for row in body["data"]} == {None}


async def test_a_list_answers_one_row_per_id_at_its_latest_revision() -> None:
    """An edit changes what a row says and never how many rows there are.

    A list of shapes has as many entries as there are names a Session can be started
    under, and editing one does not create a second name. A store that paged revisions
    would show an Environment once per edit, so a tenant with one heavily-edited shape
    would read a list mostly about itself.
    """
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    await store.insert_revision(_an_edit(only.id, tenant))
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        listed = await client.get("/v1/environments")

    (row,) = listed.json()["data"]
    assert row["id"] == str(only.id)
    assert row["revision"] == 2
    assert row["runtime_image"] == OTHER_IMAGE


async def test_a_list_does_not_carry_another_tenants_shapes() -> None:
    """The tenant is a term in the store's own query, so a cross-tenant row is absent
    from the answer rather than fetched here and dropped."""
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    store = FakeEnvironmentStore()
    (yours,) = await _registered(store, mine, 1)
    (hidden,) = await _registered(store, theirs, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, mine) as client:
        listed = await client.get("/v1/environments")

    assert [row["id"] for row in listed.json()["data"]] == [str(yours.id)]
    assert str(hidden.id) not in listed.text


async def test_a_walk_covers_every_shape_exactly_once() -> None:
    """One page at a time to the end, and the union is the whole collection.

    Asserted as a sequence and not as a set, so a boundary that repeated a row fails
    here: a keyset walk whose second half is missing repeats the row two Environments
    share a position with, and a set comparison would hide exactly that.
    """
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    made = await _registered(store, tenant, 5)
    app = build_app(a_harness(store).platform)

    walked: list[str] = []
    async with caller(app, tenant) as client:
        page: str | None = None
        for _ in range(len(made) + 1):
            query = f"?limit=2{'' if page is None else f'&page={page}'}"
            answered = await client.get(f"/v1/environments{query}")
            assert answered.status_code == 200, answered.text
            body = answered.json()
            walked.extend(row["id"] for row in body["data"])
            page = body["next_page"]
            if page is None:
                break

    assert page is None, "the walk did not end"
    assert walked == [str(one.id) for one in reversed(made)]


async def test_an_edit_does_not_move_a_shape_within_a_walk() -> None:
    """The page is ordered by when an id was FIRST registered, not by its latest row.

    This is what makes a walk stable across a concurrent edit. Ordered by the shown
    revision's own timestamp, editing the oldest Environment mid-walk would lift it
    above a page the caller already holds -- so that id would arrive a second time and
    whatever it displaced would never arrive at all.
    """
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    oldest, middle, newest = await _registered(store, tenant, 3)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        first = await client.get("/v1/environments?limit=1")
        assert [row["id"] for row in first.json()["data"]] == [str(newest.id)]
        # The edit lands between the two reads, on the shape furthest from the cursor.
        await store.insert_revision(_an_edit(oldest.id, tenant))
        rest = await client.get(
            f"/v1/environments?limit=10&page={first.json()['next_page']}"
        )

    assert [row["id"] for row in rest.json()["data"]] == [
        str(middle.id),
        str(oldest.id),
    ]
    assert rest.json()["data"][-1]["revision"] == 2


async def test_a_retired_shape_is_absent_from_a_list_unless_asked_for() -> None:
    """The default direction is the useful one: an archived shape starts no Session, so
    a list carrying it by default would put entries in front of a caller that every
    create would refuse."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    live, retired = await _registered(store, tenant, 2)
    await store.archive(retired.id, tenant)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        default = await client.get("/v1/environments")
        asked = await client.get("/v1/environments?include_archived=true")

    assert [row["id"] for row in default.json()["data"]] == [str(live.id)]
    shown = {row["id"]: row for row in asked.json()["data"]}
    assert set(shown) == {str(live.id), str(retired.id)}
    assert shown[str(retired.id)]["archived_at"] is not None
    assert shown[str(live.id)]["archived_at"] is None


async def test_a_cursor_this_surface_did_not_issue_is_refused() -> None:
    """Refused rather than treated as the start of the collection. Starting over on a
    bad cursor hands back the newest page again, which reads as the walk having looped
    rather than failed."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        refused = await client.get("/v1/environments?page=not-a-cursor")

    assert refused.status_code == STATUS_FOR[ErrorCode.PAGINATION_CURSOR_INVALID]
    assert refused.json()["error"]["code"] == ErrorCode.PAGINATION_CURSOR_INVALID.value


async def test_a_page_larger_than_the_surface_publishes_is_refused_by_field() -> None:
    """A 400 naming the field rather than the adapter's 500, so a caller learns the
    bound from the answer instead of from an outage."""
    tenant = TenantId(uuid4())
    app = build_app(a_harness(FakeEnvironmentStore()).platform)

    async with caller(app, tenant) as client:
        refused = await client.get("/v1/environments?limit=1000")

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert "limit" in refused.json()["error"]["detail"]["fields"]


# --------------------------------------------------------------------------------------
# POST /v1/environments/{id}
# --------------------------------------------------------------------------------------


async def test_an_edit_answers_with_the_next_revision_and_the_new_shape() -> None:
    """The response is the Environment as it now stands, which is what makes the edit
    observable without a second call -- including the number, so a caller can tell their
    revision from one somebody else wrote."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        edited = await client.post(
            f"/v1/environments/{only.id}",
            json=_a_body(
                name="analysis-revised",
                runtime_image=OTHER_IMAGE,
                allowed_domains=["api.example.com"],
            ),
        )
        read = await client.get(f"/v1/environments/{only.id}")

    assert edited.status_code == 200, edited.text
    assert edited.json() == {
        "id": str(only.id),
        "name": "analysis-revised",
        "runtime_image": OTHER_IMAGE,
        "denied_paths": [SECRETS],
        "allowed_domains": ["api.example.com"],
        "revision": 2,
        "archived_at": None,
    }
    assert read.json() == edited.json(), (
        "the edit's own answer and the next read of that id disagree, so one of the "
        "two is describing an Environment that does not exist"
    )


async def test_an_edit_of_an_id_nobody_registered_writes_nothing() -> None:
    """Existence is settled before the write, because the write numbers itself from
    whatever rows carry this id -- so without the read an unknown id would be written as
    a brand new Environment at revision 1 by a call that reads as an edit."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    app = build_app(a_harness(store).platform)
    unknown = new_environment_id()

    async with caller(app, tenant) as client:
        refused = await client.post(f"/v1/environments/{unknown}", json=_a_body())
        read = await client.get(f"/v1/environments/{unknown}")

    assert refused.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]
    assert refused.json()["error"]["code"] == ErrorCode.ENVIRONMENT_NOT_FOUND.value
    assert read.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND], (
        "the refused edit registered the id it refused"
    )
    assert store.inserted == []


async def test_an_edit_of_another_tenants_shape_is_the_identical_refusal() -> None:
    """Byte for byte against an id nobody registered, or the refusal tells anybody
    holding an id whether it names somebody else's shape."""
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    store = FakeEnvironmentStore()
    (hidden,) = await _registered(store, theirs, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, mine) as client:
        refused = await client.post(f"/v1/environments/{hidden.id}", json=_a_body())
        absent = await client.post(
            f"/v1/environments/{new_environment_id()}", json=_a_body()
        )

    assert (refused.status_code, _masked(refused)) == (
        absent.status_code,
        _masked(absent),
    )


async def test_an_edit_to_a_shape_the_rules_refuse_leaves_the_id_alone() -> None:
    """The new shape is parsed before anything is written, so a refusal leaves no
    revision behind and the id goes on meaning what it meant.

    The path here is one every Session already denies, which the registry refuses
    because the Permission Profile refuses two rules over one path -- so a revision
    carrying it would be stored and then fail every Session created against it.
    """
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        refused = await client.post(
            f"/v1/environments/{only.id}",
            json=_a_body(denied_paths=["/etc/codex"]),
        )
        read = await client.get(f"/v1/environments/{only.id}")

    assert refused.status_code == STATUS_FOR[ErrorCode.REQUEST_INVALID]
    assert read.json()["revision"] == 1
    assert read.json()["denied_paths"] == [SECRETS]


async def test_a_retired_shape_takes_no_edit() -> None:
    """Retirement is terminal, and an edit landing on one would produce a revision no
    Session can ever be started in -- accepted, stored, and unusable."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    await store.archive(only.id, tenant)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        refused = await client.post(f"/v1/environments/{only.id}", json=_a_body())
        read = await client.get(f"/v1/environments/{only.id}")

    assert refused.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_ARCHIVED]
    assert refused.json()["error"]["code"] == ErrorCode.ENVIRONMENT_ARCHIVED.value
    assert read.json()["revision"] == 1


# --------------------------------------------------------------------------------------
# POST /v1/environments/{id}/archive
# --------------------------------------------------------------------------------------


async def test_archiving_answers_with_the_whole_shape_and_when_it_was_retired() -> None:
    """The full Environment and not an acknowledgement, which is what their reference
    types this response as -- and the timestamp is the part that says the call took
    effect."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        retired = await client.post(f"/v1/environments/{only.id}/archive")

    assert retired.status_code == 200, retired.text
    body = retired.json()
    assert body["id"] == str(only.id)
    assert body["name"] == only.name
    assert body["runtime_image"] == only.runtime_image
    assert body["revision"] == 1
    assert body["archived_at"] is not None, (
        "their sample response shows archived_at null on an archive, which is a "
        "schema-sampler artefact; archiving sets the timestamp"
    )


async def test_archiving_twice_answers_with_the_first_retirement() -> None:
    """A retried call must not claim the shape stopped being referenceable later than it
    did, and a refusal would push every client into a read-then-write race to avoid a
    second call it cannot tell it already made."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        first = await client.post(f"/v1/environments/{only.id}/archive")
        again = await client.post(f"/v1/environments/{only.id}/archive")

    assert first.status_code == again.status_code == 200
    assert first.json() == again.json()


async def test_archiving_an_id_nobody_registered_is_the_published_not_found() -> None:
    tenant = TenantId(uuid4())
    app = build_app(a_harness(FakeEnvironmentStore()).platform)

    async with caller(app, tenant) as client:
        refused = await client.post(f"/v1/environments/{new_environment_id()}/archive")

    assert refused.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]
    assert refused.json()["error"]["code"] == ErrorCode.ENVIRONMENT_NOT_FOUND.value


async def test_archiving_another_tenants_shape_retires_nothing() -> None:
    """The refusal is the identical one, and the shape is still live for its owner --
    the second half being the one a status code alone cannot show."""
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    store = FakeEnvironmentStore()
    (hidden,) = await _registered(store, theirs, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, mine) as client:
        refused = await client.post(f"/v1/environments/{hidden.id}/archive")
    async with caller(app, theirs) as owner:
        read = await owner.get(f"/v1/environments/{hidden.id}")

    assert refused.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]
    assert read.json()["archived_at"] is None


async def test_a_retired_shape_still_reads_back() -> None:
    """Retirement is a fact on the resource rather than a deletion, which is what lets
    the tenant who archived one by mistake see what they had."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        retired = await client.post(f"/v1/environments/{only.id}/archive")
        read = await client.get(f"/v1/environments/{only.id}")

    assert read.status_code == 200
    assert read.json() == retired.json()


# --------------------------------------------------------------------------------------
# DELETE /v1/environments/{id}
# --------------------------------------------------------------------------------------


async def test_a_delete_answers_with_a_body_and_not_with_a_204() -> None:
    """Two local prose guides say this route returns 204. Both are wrong: the reference
    types the response `BetaEnvironmentDeleteResponse` and shows an id and a type, and a
    body is not a 204."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        deleted = await client.delete(f"/v1/environments/{only.id}")

    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"id": str(only.id), "type": "environment_deleted"}


async def test_a_deleted_shape_reads_back_as_absent() -> None:
    """A real delete and not a tombstone: what the caller asked for was that the id stop
    resolving, and it does."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        await client.delete(f"/v1/environments/{only.id}")
        read = await client.get(f"/v1/environments/{only.id}")
        listed = await client.get("/v1/environments?include_archived=true")

    assert read.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]
    assert listed.json()["data"] == []


async def test_deleting_twice_is_a_not_found_and_not_a_second_success() -> None:
    """Where a delete and an archive part company. An archive answers about a fact that
    is still there to be read; a delete removed the row, so a repeat has nothing to
    report and a 200 would claim this call deleted something it did not."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        first = await client.delete(f"/v1/environments/{only.id}")
        again = await client.delete(f"/v1/environments/{only.id}")

    assert first.status_code == 200
    assert again.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]
    assert again.json()["error"]["code"] == ErrorCode.ENVIRONMENT_NOT_FOUND.value


async def test_a_shape_a_session_still_holds_refuses_a_delete_and_survives() -> None:
    """The guard that makes a hard delete safe, and the count that makes the refusal
    actionable.

    A count and not a list of ids: the list is unbounded, and a refusal whose size
    depends on how much work a tenant has is one that stops being readable exactly when
    it matters most.
    """
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    store.sessions_holding[only.id] = 3
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        refused = await client.delete(f"/v1/environments/{only.id}")
        read = await client.get(f"/v1/environments/{only.id}")

    assert refused.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_IN_USE]
    assert refused.json()["error"]["code"] == ErrorCode.ENVIRONMENT_IN_USE.value
    assert refused.json()["error"]["detail"]["sessions"] == 3
    assert read.status_code == 200, "a refused delete removed the shape anyway"


async def test_deleting_another_tenants_shape_removes_nothing() -> None:
    mine, theirs = TenantId(uuid4()), TenantId(uuid4())
    store = FakeEnvironmentStore()
    (hidden,) = await _registered(store, theirs, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, mine) as client:
        refused = await client.delete(f"/v1/environments/{hidden.id}")
    async with caller(app, theirs) as owner:
        read = await owner.get(f"/v1/environments/{hidden.id}")

    assert refused.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]
    assert read.status_code == 200, "one tenant deleted another tenant's shape"


async def test_a_retired_shape_may_still_be_deleted() -> None:
    """Retirement stops new Sessions; it is not a reason to keep rows nobody can
    reach."""
    tenant = TenantId(uuid4())
    store = FakeEnvironmentStore()
    (only,) = await _registered(store, tenant, 1)
    app = build_app(a_harness(store).platform)

    async with caller(app, tenant) as client:
        await client.post(f"/v1/environments/{only.id}/archive")
        deleted = await client.delete(f"/v1/environments/{only.id}")
        read = await client.get(f"/v1/environments/{only.id}")

    assert deleted.status_code == 200, deleted.text
    assert read.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]


# --------------------------------------------------------------------------------------
# The store these four routes need, and what happens without it
# --------------------------------------------------------------------------------------


class OnlyReadsAndWrites:
    """The two-method port a Session's create and a pod's placement use, and no more.

    Exactly what `Platform.environment_store` is typed as, which is why the routes above
    have to ask the wired object whether it can do the rest.
    """

    async def insert(self, environment: Environment, /) -> None:
        raise AssertionError("this store was never meant to be written to")

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> None:
        return None


def _narrowed_app() -> FastAPI:
    return build_app(
        replace(a_harness().platform, environment_store=OnlyReadsAndWrites())
    )


@pytest.mark.parametrize(
    ("verb", "path"),
    [
        ("GET", "/v1/environments"),
        ("POST", "/v1/environments/{id}"),
        ("POST", "/v1/environments/{id}/archive"),
        ("DELETE", "/v1/environments/{id}"),
    ],
)
async def test_a_store_that_cannot_do_the_lifecycle_refuses_as_a_platform_fault(
    verb: str, path: str
) -> None:
    """Wired with a reader, the four lifecycle routes refuse and say it is ours.

    A composition root that wired something narrower than the whole store is a fault
    here and not something a request can cause, so this is `platform.internal` with a
    request id to quote rather than a refusal that blames the caller. What the case
    actually pins is that the widening is *asked for*: without the check these routes
    would reach a method the object does not have and answer with an AttributeError
    wearing a 500, naming a method rather than the wiring.

    All four, because the check is per-route: one route acquiring the store some other
    way is exactly the shape this would otherwise miss.
    """
    app = _narrowed_app()
    concrete = path.replace("{id}", str(new_environment_id()))

    async with caller(app, TenantId(uuid4())) as client:
        answered = await client.request(verb, concrete, json=_a_body())

    assert answered.status_code == STATUS_FOR[ErrorCode.INTERNAL], answered.text
    assert answered.json()["error"]["code"] == ErrorCode.INTERNAL.value
    assert answered.json()["request_id"]


async def test_the_read_and_the_create_need_no_more_than_the_narrow_store() -> None:
    """The other half of the claim above, and without it that one is satisfied by a
    surface where every route refuses.

    A create and a read go through `insert` and `fetch` alone, which is what makes the
    narrow port the right type for `Platform` -- and what would make a sixth route
    quietly widening it a change somebody has to notice.
    """
    app = _narrowed_app()

    async with caller(app, TenantId(uuid4())) as client:
        read = await client.get(f"/v1/environments/{new_environment_id()}")

    assert read.status_code == STATUS_FOR[ErrorCode.ENVIRONMENT_NOT_FOUND]
