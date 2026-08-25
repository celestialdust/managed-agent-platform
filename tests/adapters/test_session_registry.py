"""Writing a Session's creation facts, reading them back, and walking a tenant's list.

Tier 1 (testcontainers, real PostgreSQL 17). Against the real database and not a fake,
because two of the three properties here belong to PostgreSQL rather than to our code:
whether a tenant term in a WHERE clause really excludes another tenant's rows, and
whether a row-constructor comparison against `(created_at_ms, id)` really walks the
index without repeating or skipping a row at a page boundary.

Creation times are written explicitly wherever ordering is under test. Two consecutive
`create()` calls can land in the same millisecond on a fast machine and can land in
different ones on a slow one, so a walk that depended on the clock would grade a
different property on each run -- and the interesting case, two Sessions sharing a
millisecond, would be the one that never ran.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.elements import TextClause

from managed_agent.adapters.postgres.session_registry import (
    _FIRST_PAGE,
    _PAGE_AFTER,
    _PAGE_ENDING_AT,
    PostgresSessionRegistry,
    SessionListing,
)
from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.ports import (
    SessionListing as SessionListingPort,
)
from managed_agent.core.ports import (
    SessionNotVisible,
    SessionRegistry,
    SessionsWalkedBackward,
)
from managed_agent.core.session.session import SessionRecord

# Deliberately not the values any other fixture in this suite uses. Every one of these
# would be indistinguishable from a hardcoded default if it were 1, 30, 500 or "USD" --
# a `fetch` that returned a constant record would pass a round-trip assertion built out
# of the defaults.
_REVISION = "4"
_BUDGET = 731
_RETENTION = 17
_CURRENCY = "EUR"

_INSERT_AT = sa.text(
    "INSERT INTO session"
    " (id, tenant_id, definition_id, definition_revision, grant_tools, scope,"
    "  budget_minor_units, budget_currency, retention_days, created_at_ms)"
    " VALUES (:id, :tenant, :definition, :revision, :grant, :scope,"
    "  :budget, :currency, :retention, :created_at_ms)"
).bindparams(
    sa.bindparam("id", type_=sa.Uuid()),
    sa.bindparam("tenant", type_=sa.Uuid()),
    sa.bindparam("definition", type_=sa.Uuid()),
    sa.bindparam("grant", type_=sa.JSON()),
    sa.bindparam("scope", type_=sa.JSON()),
)


def _record(
    tenant_id: TenantId,
    session_id: SessionId | None = None,
    grant: frozenset[str] = frozenset({"web.fetch", "fs.read"}),
    scope: tuple[tuple[str, str], ...] = (
        ("repository", "acme/widgets"),
        ("workspace", "research"),
    ),
) -> SessionRecord:
    return SessionRecord(
        id=session_id or SessionId(uuid.uuid4()),
        tenant_id=tenant_id,
        definition_id=DefinitionId(uuid.uuid4()),
        definition_revision=_REVISION,
        grant=grant,
        scope=scope,
        budget_minor_units=_BUDGET,
        budget_currency=_CURRENCY,
        retention_days=_RETENTION,
    )


async def _at(
    engine: AsyncEngine, tenant_id: TenantId, created_at_ms: int
) -> SessionId:
    """Write one row with its creation time chosen, and return its id.

    Not through `create()`, because the column takes its value from a server default
    there and the table refuses an UPDATE, so there is no way to move it afterwards --
    which is correct for the table and inconvenient for exactly these tests.
    """
    session_id = SessionId(uuid.uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            _INSERT_AT,
            {
                "id": session_id,
                "tenant": tenant_id,
                "definition": uuid.uuid4(),
                "revision": _REVISION,
                "grant": [],
                "scope": {},
                "budget": _BUDGET,
                "currency": _CURRENCY,
                "retention": _RETENTION,
                "created_at_ms": created_at_ms,
            },
        )
    return session_id


async def _pages(
    registry: PostgresSessionRegistry,
    tenant_id: TenantId,
    page_size: int,
    expected_rows: int,
) -> list[list[SessionListing]]:
    """Walk a tenant's list a page at a time, exactly as a caller would, and stop.

    Bounded, and the bound is the interesting part. A keyset comparison written `<=`
    instead of `<` hands back the boundary row again on every page, so the walk never
    reaches an empty one -- and an unbounded loop turns that defect into a test that
    hangs rather than a test that fails. Measured: the `<=` mutation ran past ten
    minutes here before this bound existed. A hang reports nothing, blocks the suite,
    and looks like a slow database; a refusal names the row that came back twice.
    """
    pages: list[list[SessionListing]] = []
    seen: set[SessionListing] = set()
    after: tuple[int, uuid.UUID] | None = None
    while len(pages) <= expected_rows + 1:
        page: Sequence[SessionListing] = list(
            await registry.page(tenant_id, after, page_size)
        )
        if not page:
            return pages
        repeated = seen & set(page)
        assert not repeated, (
            f"page {len(pages) + 1} returned {len(repeated)} row(s) an earlier page "
            f"already returned: {sorted(str(row.id) for row in repeated)}. The keyset "
            "boundary is inclusive somewhere, so the walk never ends."
        )
        seen.update(page)
        pages.append(list(page))
        after = (page[-1].created_at_ms, page[-1].id)
    raise AssertionError(
        f"the walk read {len(pages)} pages of at most {page_size} for "
        f"{expected_rows} rows without reaching an empty one"
    )


async def _walk(
    registry: PostgresSessionRegistry,
    tenant_id: TenantId,
    page_size: int,
    expected_rows: int,
) -> list[SessionListing]:
    """Every row a tenant has, flattened out of the pages it took to read them."""
    return [
        row
        for page in await _pages(registry, tenant_id, page_size, expected_rows)
        for row in page
    ]


async def test_create_then_fetch_round_trips_an_equal_record(
    engine: AsyncEngine,
) -> None:
    """The whole record, compared as one value rather than field by field.

    Comparing the dataclass is what catches a Grant that came back as a list, a Scope
    that came back as a dict, or a revision that came back as an integer -- each of
    which a handful of scalar assertions would let through, and each of which breaks a
    consumer somewhere else.
    """
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    written = _record(tenant_id)

    await registry.create(written)

    assert await registry.fetch(written.id, tenant_id) == written


async def test_an_empty_grant_and_an_empty_scope_round_trip_as_empty(
    engine: AsyncEngine,
) -> None:
    """Empty is a value here, not an absence.

    A Session that may call no tool and reaches no scoped resource is a legitimate thing
    to create, and it must not come back looking like a Session whose Grant was never
    decided -- the store has no way to express the second, and this is what keeps the
    first from being confused for it.
    """
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    written = _record(tenant_id, grant=frozenset(), scope=())

    await registry.create(written)
    read = await registry.fetch(written.id, tenant_id)

    assert read == written
    assert read.grant == frozenset()
    assert read.scope == ()


async def test_a_scope_round_trips_in_a_stable_order(engine: AsyncEngine) -> None:
    """Two reads of one row are equal values, whatever order the mapping came back in.

    `SessionRecord` is frozen and compared by value, and a mapping's iteration order is
    not part of the row it was stored as -- so a Scope read back in the store's order
    would make one Session compare unequal to itself across two reads.
    """
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    written = _record(tenant_id, scope=(("alpha", "1"), ("beta", "2"), ("gamma", "3")))

    await registry.create(written)

    first = await registry.fetch(written.id, tenant_id)
    second = await registry.fetch(written.id, tenant_id)
    assert first == second
    assert first.scope == (("alpha", "1"), ("beta", "2"), ("gamma", "3"))


async def test_another_tenants_session_is_refused_rather_than_returned(
    engine: AsyncEngine,
) -> None:
    """The tenant is a term in the query, so the row is never fetched to be dropped."""
    registry = PostgresSessionRegistry(engine)
    owner, stranger = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())
    written = _record(owner)
    await registry.create(written)

    with pytest.raises(SessionNotVisible):
        await registry.fetch(written.id, stranger)

    assert await registry.fetch(written.id, owner) == written


async def test_an_id_nobody_created_is_refused_the_same_way(
    engine: AsyncEngine,
) -> None:
    """One refusal for "no such Session" and for "not yours".

    Two distinguishable answers would turn a read into an existence oracle: a caller
    holding an id could learn from the shape of the refusal whether it names another
    tenant's Session. Asserted as the same exception type here; the route asserts the
    same response body in `tests/control/test_sessions_tenant_scope.py`.
    """
    registry = PostgresSessionRegistry(engine)
    owner, stranger = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())
    written = _record(owner)
    await registry.create(written)

    with pytest.raises(SessionNotVisible) as absent:
        await registry.fetch(SessionId(uuid.uuid4()), stranger)
    with pytest.raises(SessionNotVisible) as hidden:
        await registry.fetch(written.id, stranger)

    assert type(absent.value) is type(hidden.value)


async def test_a_page_carries_only_the_calling_tenants_sessions(
    engine: AsyncEngine,
) -> None:
    """Another tenant's Sessions are absent, not present-and-redacted.

    Both tenants hold Sessions and both interleave in creation order, so a page that
    filtered after the read -- or forgot to -- would show the difference here. That is
    MAP-A11 in the words it is written in.
    """
    registry = PostgresSessionRegistry(engine)
    ours, theirs = TenantId(uuid.uuid4()), TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    mine = [await _at(engine, ours, base + step) for step in (0, 2, 4)]
    yours = [await _at(engine, theirs, base + step) for step in (1, 3, 5)]

    listed = [row.id for row in await _walk(registry, ours, 2, expected_rows=3)]

    assert sorted(listed) == sorted(mine)
    assert not set(listed) & set(yours), (
        "another tenant's Session appeared in this tenant's list"
    )


async def test_seven_rows_walked_three_at_a_time_yield_each_once_newest_first(
    engine: AsyncEngine,
) -> None:
    """The page boundary neither repeats a row nor skips one, and the order holds.

    Seven and three so the last page is short rather than exact: a walk that stopped on
    a short page would still be right here, and a walk that mishandled the boundary
    between full pages would not.
    """
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    written = [await _at(engine, tenant_id, base + step) for step in range(7)]

    pages = await _pages(registry, tenant_id, 3, expected_rows=7)

    assert [len(page) for page in pages] == [3, 3, 1]
    walked = [row.id for row in [row for page in pages for row in page]]
    assert walked == list(reversed(written)), (
        "the walk did not return every Session exactly once, newest first"
    )
    assert [row.created_at_ms for page in pages for row in page] == sorted(
        (base + step for step in range(7)), reverse=True
    )


async def test_two_sessions_sharing_a_millisecond_cross_a_page_boundary_intact(
    engine: AsyncEngine,
) -> None:
    """The id in the key is what makes this work, and the boundary is placed on purpose.

    Four Sessions, the middle two created in the same millisecond, walked two at a time
    so the page boundary falls **between** the two equal timestamps. Ordering on
    `created_at_ms` alone leaves their relative order undefined, so the next page would
    either hand back the one already seen or skip past both.
    """
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    oldest = await _at(engine, tenant_id, base)
    tied_low = await _at(engine, tenant_id, base + 1)
    tied_high = await _at(engine, tenant_id, base + 1)
    newest = await _at(engine, tenant_id, base + 2)

    walked = [row.id for row in await _walk(registry, tenant_id, 2, 4)]

    assert len(walked) == len(set(walked)) == 4, f"walk returned {walked}"
    assert walked[0] == newest
    assert walked[3] == oldest
    assert set(walked[1:3]) == {tied_low, tied_high}


async def test_a_session_created_during_a_walk_neither_appears_nor_displaces_a_row(
    engine: AsyncEngine,
) -> None:
    """Keyset paging, not offset paging, and this is the difference between them.

    The new Session is the newest, so under offset paging it would shift every remaining
    row one place further down and the walk would hand back a duplicate and drop the
    oldest. Under a keyset the walk is already below it and cannot see it at all.
    """
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    written = [await _at(engine, tenant_id, base + step) for step in range(5)]

    first = list(await registry.page(tenant_id, None, 2))
    interloper = await _at(engine, tenant_id, base + 99)

    rest: list[SessionListing] = []
    after = (first[-1].created_at_ms, first[-1].id)
    for _ in range(6):
        page = list(await registry.page(tenant_id, after, 2))
        if not page:
            break
        rest.extend(page)
        after = (page[-1].created_at_ms, page[-1].id)
    else:
        raise AssertionError("the walk never reached an empty page")

    walked = [row.id for row in first + rest]
    assert walked == list(reversed(written)), (
        f"the walk returned {walked}, not the five Sessions that existed when it began"
    )
    assert interloper not in walked


async def test_a_tenant_with_no_sessions_gets_an_empty_page(
    engine: AsyncEngine,
) -> None:
    """Empty rather than a refusal: having no Sessions is a normal thing to have."""
    registry = PostgresSessionRegistry(engine)

    assert await registry.page(TenantId(uuid.uuid4()), None, 10) == []


@pytest.mark.parametrize("limit", [0, -1, 501, 10_000])
async def test_a_limit_outside_the_window_is_refused_rather_than_clamped(
    engine: AsyncEngine, limit: int
) -> None:
    """Refused because a clamped page is a short page and a short page means the end.

    Silently reducing an over-large limit would hand back fewer rows than asked for, and
    this port says a short page means there is nothing below it -- so the reduction
    would read as the end of the walk and the caller would stop, having seen part of its
    Sessions with nothing to tell it so.
    """
    registry = PostgresSessionRegistry(engine)

    with pytest.raises(ValueError, match="outside 1..500"):
        await registry.page(TenantId(uuid.uuid4()), None, limit)


async def test_a_page_returns_the_facts_a_caller_lists_by(engine: AsyncEngine) -> None:
    """A listing row carries the id, its definition, that definition's pinned revision
    and the creation time -- and nothing about state, because state is a fold over the
    Session's own log and a page of twenty-five rows would be twenty-five folds."""
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    written = _record(tenant_id)
    await registry.create(written)

    (listed,) = await registry.page(tenant_id, None, 10)

    assert listed.id == written.id
    assert listed.definition_id == written.definition_id
    assert listed.definition_revision == _REVISION
    assert listed.created_at_ms > 10**12
    assert not hasattr(listed, "state")


@pytest.mark.parametrize(
    ("name", "statement"),
    [("_FIRST_PAGE", _FIRST_PAGE), ("_PAGE_AFTER", _PAGE_AFTER)],
)
def test_a_page_orders_by_every_component_of_the_cursor(
    name: str, statement: TextClause
) -> None:
    """The ORDER BY names both halves of the keyset, in the keyset's order, descending.

    Structural because the property cannot be observed here. Dropping `id DESC` leaves
    every behavioural test in this file passing: PostgreSQL happens to satisfy the
    remaining ORDER BY from `session_by_tenant_creation`, which carries the id anyway,
    so the rows come back in the right order by accident of the plan. Measured -- the
    mutation survived the whole file. A different plan (a sequential scan on a small
    table, a parallel scan, a later index) is free to order two rows sharing a
    millisecond either way, and then the keyset walk repeats one and skips the other.

    `test_two_sessions_sharing_a_millisecond_cross_a_page_boundary_intact` reads like
    the behavioural half of this, and it is not -- measured, it is **nondeterministic**
    against this mutation rather than merely insensitive to it. Run alone against a
    four-row table it failed 2 of 4 times; run inside this file, where nineteen other
    tests have loaded `session` and changed the plan, it passed 5 of 5. So in the
    configuration the suite actually runs it never catches the mutation, and in
    isolation it is a coin flip. That is worse than a gap: a guard that fires sometimes
    invites the reading that the property is covered.

    Nor can it be fixed from the outside, which is why this assertion is structural
    rather than a better walk. The mutation is invisible *because* the planner satisfies
    the remaining ORDER BY from an index that happens to carry the id -- so whether a
    test can see it is a property of the plan, and the plan is chosen by row counts the
    test does not control. The only place the property is stably observable is the SQL.

    So this assertion is the guard for the ordering. The behavioural test is the guard
    for the page-boundary arithmetic, a different property that it does measure well.
    """
    clause = str(statement).split("ORDER BY", 1)
    assert len(clause) == 2, f"{name} has no ORDER BY at all"
    ordering = re.findall(r"(\w+)\s+(ASC|DESC)", clause[1].split("LIMIT")[0])

    assert ordering == [("created_at_ms", "DESC"), ("id", "DESC")], (
        f"{name} orders by {ordering}, not by the whole cursor key. The cursor carries "
        "(created_at_ms, id) and the walk compares against both, so an ORDER BY naming "
        "fewer columns leaves rows that tie on the ones it does name in whatever order "
        "the plan produced -- and the page boundary then repeats a row or skips one."
    )


async def test_the_stored_grant_is_sorted_rather_than_in_set_order(
    engine: AsyncEngine,
) -> None:
    """The stored jsonb array is sorted, so one Grant has exactly one stored form.

    `fetch` reads the Grant back into a `frozenset`, which erases order -- so every
    round-trip assertion in this file passes whether the array was sorted on the way in
    or not. This reads the raw column instead, because the order *is* part of a jsonb
    array's value: two Sessions carrying the same Grant should store the same document,
    and anything comparing stored rows -- a diff, a checksum, an audit export -- reports
    a change nobody made when they do not.

    The summary line used to say "one record written twice produces byte-identical
    documents", and that is not what this proves. Both records below share one
    `frozenset` object, so two writes of it are byte-identical whether the code sorts
    or not -- the claim was true and vacuous. What is proved is *sortedness*, which is
    the stronger property and the one that generalises to two Grants built separately.

    Written as a claim in `create`'s docstring before it was written as a test, and
    surviving a mutation from `sorted(...)` to `list(...)` is what showed the claim was
    unguarded.
    """
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    # Names chosen so set iteration is unlikely to land in sorted order. "Unlikely" is
    # the honest word: the earlier comment here said "under any plausible set
    # iteration", which was measurably too strong. PYTHONHASHSEED is pinned nowhere, so
    # this is a per-run draw, and across 3000 seeds the mutation survived 18 times
    # (0.60%). A one-in-167 false negative in a guard is not worth keeping when the fix
    # is to assert the draw instead of hoping for it.
    grant = frozenset({"web.fetch", "fs.read", "a.tool", "zz.last", "mid.dle"})
    assert list(grant) != sorted(grant), (
        "this fixture's set iteration came out sorted under this run's PYTHONHASHSEED, "
        "so list(grant) and sorted(grant) are the same sequence and the assertion "
        "below cannot tell them apart -- it would pass while `create` did no sorting "
        "at all. That is a defect in this test, not in the product. Fix the fixture "
        "(more names, or different ones) rather than deleting this check, which is "
        "what turns a silent false negative into a loud one."
    )
    first = _record(tenant_id, grant=grant)
    second = _record(tenant_id, grant=grant)

    await registry.create(first)
    await registry.create(second)

    async with engine.connect() as conn:
        stored = (
            (
                await conn.execute(
                    sa.text(
                        "SELECT grant_tools FROM session WHERE id = ANY(:ids)"
                    ).bindparams(sa.bindparam("ids", type_=sa.ARRAY(sa.Uuid()))),
                    {"ids": [first.id, second.id]},
                )
            )
            .scalars()
            .all()
        )

    assert list(stored) == [sorted(grant), sorted(grant)], (
        f"the stored Grants are {stored}; a jsonb array keeps its order, so an "
        "unsorted write makes one record store two different documents"
    )


def test_the_adapter_satisfies_the_port_it_is_wired_behind(engine: AsyncEngine) -> None:
    """The composition root binds this class to `SessionRegistry`; check it can be.

    `runtime_checkable` sees names and not signatures, so this catches the adapter that
    never grew a method rather than the one that grew it wrong -- the tests around it
    cover the second. Worth having anyway: the field is typed as the port, and a missing
    method would otherwise surface as an `AttributeError` on the first request rather
    than here.
    """
    assert isinstance(PostgresSessionRegistry(engine), SessionRegistry)


def test_a_listing_row_satisfies_the_shape_the_port_promises() -> None:
    """The row type the adapter returns is the one a caller was told to expect."""
    row = SessionListing(
        id=SessionId(uuid.uuid4()),
        definition_id=DefinitionId(uuid.uuid4()),
        definition_revision=_REVISION,
        created_at_ms=1_700_000_000_000,
    )

    assert isinstance(row, SessionListingPort)


# --- walking back to a page already read -----------------------------------------


def test_the_adapter_can_also_be_the_narrower_backward_port(
    engine: AsyncEngine,
) -> None:
    """The composition root will bind this to the narrower port; check it can be.

    Separate from the assertion above it because the two ports say different things.
    `SessionRegistry` is what every deployment must satisfy; this one is the capability
    a deployment may or may not have, and the route asks with `isinstance` before it
    offers a caller any way to use it. If this ever came back false the wire would
    quietly stop carrying `prev_page`, with every other test here still passing."""
    assert isinstance(PostgresSessionRegistry(engine), SessionsWalkedBackward)


def test_the_backward_page_orders_by_every_component_of_the_cursor() -> None:
    """Both halves of the keyset, ascending, for `_PAGE_AFTER`'s reason and one more.

    Ascending because the `LIMIT` has to cut at the end *away* from the key -- a DESC
    sort with this WHERE returns the newest rows in the tenant's whole list on every
    backward page.

    Structural for the reason the forward version is: dropping `id ASC` leaves the
    behavioural tests below passing, because the plan satisfies the rest of the ordering
    from an index that carries the id anyway. There is a second reason here. The row
    this cut discards is the one furthest from the key, so an unbroken tie loses
    whichever of two rows sharing a millisecond the plan happened to put last -- and the
    page then differs from the forward page it exists to reproduce, in a way no
    assertion about the key's own end can see."""
    clause = str(_PAGE_ENDING_AT).split("ORDER BY", 1)
    assert len(clause) == 2, "_PAGE_ENDING_AT has no ORDER BY at all"
    ordering = re.findall(r"(\w+)\s+(ASC|DESC)", clause[1].split("LIMIT")[0])

    assert ordering == [("created_at_ms", "ASC"), ("id", "ASC")], (
        f"_PAGE_ENDING_AT orders by {ordering}. The limit cuts the walk at the far end "
        "from the key, so a tie the ORDER BY does not break lets the plan decide which "
        "of two rows sharing a millisecond falls off the page -- and the backward page "
        "then differs from the forward page it is supposed to reproduce."
    )


async def test_the_page_a_forward_cursor_closed_comes_back_whole(
    engine: AsyncEngine,
) -> None:
    """The contract in one sentence: the row that closed a page names that page.

    Six rows at three so both pages are full, then each page is walked back to from its
    own last row. This is what makes the comparison inclusive rather than a matter of
    taste -- exclusive, each of these returns the page minus its final row plus one from
    further up, which is still three rows and still looks like a page.

    Against real PostgreSQL because `>=` on a row constructor is the half that is not
    ours. An in-memory fake sorts a list in Python and agrees with any comparison
    written the same way twice."""
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    for step in range(6):
        await _at(engine, tenant_id, base + step)

    pages = await _pages(registry, tenant_id, 3, expected_rows=6)

    assert [len(page) for page in pages] == [3, 3]
    for forward in pages:
        closed = forward[-1]
        walked = await registry.page_ending_at(
            tenant_id, (closed.created_at_ms, closed.id), 3
        )
        assert list(reversed(walked)) == forward, (
            "walking back from the row that closed a page did not reproduce that page"
        )


async def test_the_row_past_a_backward_page_is_the_next_one_newer(
    engine: AsyncEngine,
) -> None:
    """One row more than a page, asked for at the store, is the look-ahead.

    The route reads a backward page one row long to learn whether a further page exists
    that way, and this is that read. Four asked for from the oldest row of six returns
    four -- the three of the page and the one past it -- which is the arithmetic the
    route's `walked[limit:]` slice depends on."""
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    written = [await _at(engine, tenant_id, base + step) for step in range(6)]

    walked = await registry.page_ending_at(tenant_id, (base, written[0]), 4)

    assert [row.id for row in walked] == written[:4]
    assert walked[3].id == written[3], (
        "the fourth row is the look-ahead past a page of three, not a page row"
    )


async def test_a_backward_page_from_the_newest_row_is_short(
    engine: AsyncEngine,
) -> None:
    """A page ending at the newest row has nothing above it, and comes back short.

    A short page is how this read says there is nothing further back, the same way a
    short forward page says the walk is over. Three asked for, one returned, and the
    route turns that into a null `prev_page`."""
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    written = [await _at(engine, tenant_id, base + step) for step in range(5)]

    walked = await registry.page_ending_at(tenant_id, (base + 4, written[4]), 3)

    assert [row.id for row in walked] == [written[4]]


async def test_a_key_on_no_row_still_names_the_place_it_would_have_been(
    engine: AsyncEngine,
) -> None:
    """A key naming no row is a position, not a lookup, and does not fail.

    Rows ten milliseconds apart and a key landing between two of them. This matters
    because a cursor outlives the row it was minted from: retention deletes rows, and a
    caller can hold a token across that. The read walks from where the key would sit,
    which is the only answer that keeps a page boundary meaningful once the boundary row
    is gone."""
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    written = [await _at(engine, tenant_id, base + 10 * step) for step in range(4)]

    walked = await registry.page_ending_at(tenant_id, (base + 15, uuid.uuid4()), 10)

    assert [row.id for row in walked] == written[2:]


async def test_two_sessions_sharing_a_millisecond_come_back_the_same_way_both_ways(
    engine: AsyncEngine,
) -> None:
    """Two Sessions in one millisecond, with the page boundary between them.

    The forward walk already proves the boundary holds going down. This proves the
    backward read reproduces the same split -- which is a different property, because
    the tie is broken at the other end of the page here: the row that falls off a
    backward page is the one furthest from the key, so a comparison or an ordering that
    ignored the id could keep the tied pair together and drop the row the forward page
    kept."""
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    await _at(engine, tenant_id, base)
    await _at(engine, tenant_id, base + 1)
    await _at(engine, tenant_id, base + 1)
    await _at(engine, tenant_id, base + 2)

    pages = await _pages(registry, tenant_id, 2, expected_rows=4)
    closed = pages[1][-1]
    back = await registry.page_ending_at(
        tenant_id, (closed.created_at_ms, closed.id), 2
    )

    assert list(reversed(back)) == pages[1]


async def test_another_tenants_rows_are_absent_from_a_backward_page(
    engine: AsyncEngine,
) -> None:
    """The tenant is a term in this query too, and not a filter applied after it.

    Written with keyword arguments on purpose. The signature is `(tenant_id, oldest,
    limit)` and `oldest` is a tuple, so a transposition would be a type error -- but the
    tenant going missing from the WHERE clause would not be, and this is the read that
    would then return two other tenants' Sessions."""
    registry = PostgresSessionRegistry(engine)
    mine = TenantId(uuid.uuid4())
    theirs = TenantId(uuid.uuid4())
    base = 1_700_000_000_000
    oldest = await _at(engine, mine, base)
    await _at(engine, theirs, base + 1)
    await _at(engine, theirs, base + 2)

    walked = await registry.page_ending_at(
        tenant_id=mine, oldest=(base, oldest), limit=5
    )

    assert [row.id for row in walked] == [oldest]


@pytest.mark.parametrize("limit", [0, -1, 501, 10_000])
async def test_a_backward_limit_outside_the_window_is_refused_not_clamped(
    engine: AsyncEngine, limit: int
) -> None:
    """The same refusal window as `page`, and refused rather than reduced.

    A clamped page is a short page, and a short page is how both of these reads say
    there is nothing further. Reducing the limit would make "you asked for too many"
    indistinguishable from "there are no more", and a caller reading the second would
    stop walking with rows still unread."""
    registry = PostgresSessionRegistry(engine)
    tenant_id = TenantId(uuid.uuid4())

    with pytest.raises(ValueError, match="outside 1..500"):
        await registry.page_ending_at(
            tenant_id, (1_700_000_000_000, uuid.uuid4()), limit
        )
