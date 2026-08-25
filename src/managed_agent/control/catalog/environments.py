"""Registering a sandbox shape, and resolving one a Session named.

One parse, used twice. A shape is parsed on the way in from a tenant and parsed again
on the way out of the store, through the same function -- so a row written before a rule
existed fails to resolve rather than compiling into a pod under rules the platform no
longer permits. That direction is deliberate: the cost is a Session that will not start,
and the alternative cost is a Session that starts in a shape nobody would grant today.

`UnknownEnvironment` covers "no such id" and "not yours" with one answer. Two
distinguishable refusals would turn a create call into an existence oracle over other
tenants' ids.

Nothing here reaches a database. The port below is what a store must do, declared
beside its one consumer; the Postgres adapter satisfies it structurally and imports
nothing from here.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, runtime_checkable
from uuid import UUID

from managed_agent.control.pod_config.compiler import WORKSPACE_ROOT, session_profile
from managed_agent.core.ids import TenantId
from managed_agent.core.pod.permission_profile import nested_deny_pairs
from managed_agent.core.registration.environment import Environment, EnvironmentId

ENVIRONMENT_COLUMNS: Final = (
    "id",
    "tenant_id",
    "name",
    "runtime_image",
    "denied_paths",
)
"""The keys 0014's columns oblige a stored row to carry, spelled once.

A test compares this against that migration's `sa.Column` names, because a rename on
one side is a KeyError on the other.

A read carries three more that this tuple deliberately omits, because none of the three
is a column 0014 declared and the comparison is against that file. `allowed_domains`
arrived in 0020 and `revision` in 0022; `archived_at` is not a column of this table at
all -- it comes from the LEFT JOIN onto `environment_archive`, and it is null for every
Environment nobody has retired. All three are pinned against a real database in
`tests/adapters/test_environment_revisions_schema.py` instead of here.
"""

_WRITABLE_PREFIX: Final = f"{WORKSPACE_ROOT}/"

ALREADY_DENIED: Final = tuple(
    path for path in session_profile().denied() if path.startswith(_WRITABLE_PREFIX)
)
"""The paths under the writable root that every Session already denies.

Derived from the profile rather than restated, because a second copy is free to fall
behind the compiler that owns the first: a deny rule added there and not here would
make a shape naming that path register cleanly and then fail *every* Session created
against it -- the Permission Profile refuses two rules over one path -- and the tenant
would meet that fault far from the mistake that caused it.
"""

RUNTIME_PROTECTED_NAMES: Final = (".git", ".agents", ".codex")
"""Path components the agent runtime keeps read-only inside every writable root.

These three at codex-cli 0.149.0, read off the runtime's own
`protocol/src/permissions.rs` -- not derived, because the set lives in a compiled
binary this platform runs rather than in anything here. This tuple is NOT asserted to
be every name a later runtime protects. A version that added a fourth would let a
shape naming it register and then fail every Session created against it, which is
loud and one-sided -- it never makes a path reachable that was not -- but it is real,
so re-read that constant when the pinned runtime version moves.

Denying one of these as well is fatal rather than redundant, and the mechanism is why
the check is by component rather than by whole path. A deny path that does not exist
under a writable root is masked with a *file* bound over its first missing component;
a protected metadata name that does not exist is masked with an empty read-only
*directory*. Name a path that is both and the sandbox is handed two contradictory
operations against one target, and bubblewrap refuses to build any sandbox at all
rather than choosing -- so no command in that Session runs. Measured at the workspace
root: `bwrap: Can't create file at /session/workspace/.git: Is a directory`, with the
confined command not running. Below the root -- `<workspace>/sub/.codex` -- the same
reading holds and no pod has confirmed it; refusing there is the fail-safe direction
and costs a tenant nothing, because the runtime already keeps those subtrees
read-only whatever this platform denies.

Compared as whole components and never as a substring or a prefix. `.gitignore`,
`.gitmodules`, `my.git` and `.agentsX` are ordinary names a substring test refuses and
a tenant is entitled to deny.
"""


class UnknownEnvironment(Exception):
    """No environment with that id is visible to that tenant."""


@dataclass(frozen=True, slots=True)
class ResolvedEnvironment:
    """A shape read back, the revision it was read at, and when it was retired.

    The revision travels with the shape because either half alone is an unusable
    answer: a shape with no number cannot be pinned by whatever stores it, and a number
    with no shape cannot be run. An id can now stand for several revisions, so "the
    Environment behind this id" is only a complete answer once it says which one.

    `archived_at` travels with both because whether a shape may still be *used* is not a
    property of the shape. A retired Environment parses exactly as it did the day it was
    registered -- nothing about its image or its denied paths changes -- so a caller
    that read only the shape would start a Session in an Environment somebody
    retired, and the read would look entirely successful.

    None means not retired. There is no third state and no unarchive: the timestamp is
    written once and never cleared, which is what makes `archived` below a fact rather
    than a snapshot.
    """

    environment: Environment
    revision: int
    archived_at: datetime | None

    @property
    def archived(self) -> bool:
        """Whether this Environment has been retired, which is terminal."""
        return self.archived_at is not None


@dataclass(frozen=True, slots=True)
class ListedEnvironment:
    """One row of a tenant's list, carrying the position a later page resumes after.

    `created_at_ms` is when the *id* was first registered and not when the revision in
    this row was written, and that is what keeps a walk stable across an edit. An edit
    appends a revision; a sort key that moved with it would lift the Environment above a
    page the caller already holds, so that id would arrive a second time and whatever it
    displaced would never arrive at all.
    """

    resolved: ResolvedEnvironment
    created_at_ms: int


class EnvironmentStore(Protocol):
    """The store, as much of it as this module needs."""

    async def insert(self, environment: Environment, /) -> None:
        """Write one shape. The id is new, so a collision is a store error, not an
        update."""

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        """One row keyed by `ENVIRONMENT_COLUMNS`, or None when that tenant has no such
        id.

        Returns the row rather than an `Environment` so that parsing happens in exactly
        one place, above, instead of once per adapter. The tenant is a predicate the
        store applies, not a field the caller compares afterwards: another tenant's
        shape must be absent from the answer rather than fetched and then dropped.
        """


class EnvironmentRevisions(EnvironmentStore, Protocol):
    """The store as the placement path needs it: a read at a revision the caller names.

    A sibling of `EnvironmentLifecycle` below rather than a base of it, and the reason
    is least privilege rather than tidiness. The path that builds a pod has to read a
    pinned revision and has no business being typed as able to delete an Environment;
    the four routes have to do all four of those and never read a pinned revision. Two
    ports, each exactly as wide as its own consumer, over one store that satisfies both.

    Not `runtime_checkable`, because nothing narrows to it at run time -- its consumer
    receives the store as a constructor argument, so the widening is settled by the
    annotation where the object is passed in and mypy checks it there.
    """

    async def fetch_revision(
        self,
        environment_id: EnvironmentId,
        tenant_id: TenantId,
        revision: int,
        /,
    ) -> Mapping[str, object] | None:
        """One named revision of that tenant's shape, or None when there is no such row.

        None for a missing id, another tenant's id and a revision that does not exist:
        one answer, for the reason `fetch` gives, and a caller able to tell the third
        apart from the first could count another tenant's edits.

        A retired Environment is returned like any other. Whether a shape may still be
        used is the caller's question and depends on what the caller is doing: starting
        a new Session is refused, continuing one already created is not.
        """


@runtime_checkable
class EnvironmentLifecycle(EnvironmentStore, Protocol):
    """The rest of the store: what listing, editing, retiring and deleting need.

    Declared beside `EnvironmentStore` rather than folded into it, and the reason is a
    measurement rather than a preference. `Platform.environment_store` is typed at that
    Protocol, and five test modules hand it a stand-in carrying only `insert` and
    `fetch`; a member added there turns each of those into a `mypy --strict` failure at
    the composition, in files that have no business knowing an Environment can be
    retired. So the port a Session's create and a pod's placement need stays exactly as
    wide as those two need, and the four lifecycle routes ask for this wider one --
    which is also the truthful description of the dependency, since reading a shape
    is all either of those paths does.

    `runtime_checkable` so the widening can be asserted where the binding is made, for
    the reason `core/ports.py` gives for its own ports. It compares method names and not
    signatures, which is enough for what it is asked here: whether the object the
    composition root wired is the whole store or a reader of it.
    """

    async def insert_revision(self, environment: Environment, /) -> int:
        """Append the next revision of an id that already exists, and say which.

        The number is computed inside the statement that writes it, so there is no
        window between reading it and using it, and two concurrent edits of one id
        resolve as a primary-key conflict rather than as one silently overwriting the
        other. Raises `sqlalchemy.exc.IntegrityError` for that loser.
        """

    async def archive(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> datetime | None:
        """Retire this Environment and return when it was retired, or None if it is
        not this tenant's.

        Idempotent, and the timestamp is what makes that observable: a second call
        returns the moment of the FIRST one. A fresh timestamp would claim the
        Environment stopped being referenceable at the moment of a retry, which is a
        false fact about when new Sessions started being refused.
        """

    async def delete(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> bool:
        """Remove every revision of this id, and say whether anything was removed.

        False means the id was already gone or was never this tenant's, which are one
        answer for the reason `fetch` gives. A real delete rather than a tombstone: the
        caller has already established that no Session references it, so there is no
        history a removal could make unreadable.
        """

    async def sessions_referencing(self, environment_id: EnvironmentId, /) -> int:
        """How many Sessions that have not stopped were created in this Environment.

        The count is what a delete is refused on. It is not scoped by tenant, and that
        is the fail-safe direction rather than an oversight: an id is only resolvable by
        the tenant who owns it, so every Session naming one is theirs, and a tenant
        predicate could only ever make this number *smaller* than the set of Sessions a
        delete would strand.
        """

    async def page(
        self,
        tenant_id: TenantId,
        after: tuple[int, UUID] | None,
        limit: int,
        include_archived: bool,
        /,
    ) -> Sequence[Mapping[str, object]]:
        """One page of this tenant's Environments, newest first, latest revision each.

        One row per id and never one per revision: a list of shapes a tenant may name
        has as many entries as there are names.

        `after` is the `(created_at_ms, id)` of the last row the caller holds, as one
        value rather than two nullable arguments -- half a keyset is not a position.
        Rows, not parsed shapes, for the reason `fetch` gives.
        """


def parse_environment(
    *,
    environment_id: EnvironmentId,
    tenant_id: TenantId,
    name: str,
    runtime_image: str,
    denied_paths: tuple[str, ...],
    allowed_domains: tuple[str, ...] = (),
) -> Environment:
    """Build a shape, or raise `ValueError` saying which part is not acceptable.

    The `Environment`'s own rules run first, so a path is normalised, absolute,
    glob-free and unrepeated before it is compared against anything.

    Then four rules, in this order, because a path can break more than one and the
    first sentence is the one a tenant reads. A path must be strictly inside the one
    writable root -- everything outside it is either already denied for every Session
    or is a platform path, and an allowlist refuses both without anyone remembering to
    extend a list. It must not restate a path the platform already denies, which the
    Permission Profile would refuse as two rules over one path. No two denied paths,
    counting the platform's own, may lie one inside the other. And no denied path may
    name a component the runtime protects.

    The last two are two mechanisms with one outcome: bubblewrap builds no sandbox at
    all, so a shape that broke either would register cleanly and then fail every
    Session created against it -- the tenant meeting the fault far from the mistake
    that caused it. Nested denies fail because unreadable roots are applied
    parent-first and the descendant's own operation is then attempted inside a
    filesystem already frozen; a protected name fails because the runtime's own
    directory mask and this platform's file mask are two contradictory operations
    against one target.

    The order of the last two matters and is not cosmetic. Every path the platform
    denies under the writable root is itself a protected name, so checking protected
    names first would leave the nesting rule unable to fire on any value this platform
    actually ships, and its only test would be one that substituted a fake profile.
    """
    environment = Environment(
        id=environment_id,
        tenant_id=tenant_id,
        name=name,
        runtime_image=runtime_image,
        denied_paths=denied_paths,
        allowed_domains=allowed_domains,
    )
    # `allowed_domains` gets no check here and that is not an omission. Every rule it
    # has is a spelling rule and belongs to `Environment`, which has already run: unlike
    # a denied path, a domain has no relationship to this platform's own configuration
    # for a second pass to compare it against. There is no writable root for it to be
    # inside, no platform entry it could restate, and no nesting between two names that
    # bubblewrap would refuse. What could go wrong with a domain goes wrong at the
    # sandbox's proxy, which resolves it and blocks private destinations by address.
    for path in environment.denied_paths:
        if not path.startswith(_WRITABLE_PREFIX):
            raise ValueError(
                f"{path!r} is outside {WORKSPACE_ROOT}; an environment narrows what "
                "the agent may write and reaches nothing else"
            )
        if path in ALREADY_DENIED:
            raise ValueError(f"{path!r} is already denied for every Session")
    # One call over the union rather than two directional checks of "is this tenant
    # path inside something". The pair form asks only whether a tenant path is inside
    # another and never whether a platform path is inside a tenant one, and that second
    # direction goes live the moment the platform denies anything below the writable
    # root's first level -- a tenant denying `<root>/vault` beside a platform
    # `<root>/vault/keys` is the same fatal pair with the sides swapped. There is no
    # direction to omit here, and no filter on which side of a pair the platform owns:
    # a profile that nested two of its own would refuse every registration in the
    # system, which is deliberate and loud, because such a profile starts no Session
    # anyway and the compiler's floor refuses to emit it.
    nested = nested_deny_pairs(ALREADY_DENIED + environment.denied_paths)
    if nested:
        ancestor, descendant = nested[0]
        raise ValueError(
            f"{descendant!r} is inside {ancestor!r}: two denied paths one inside the "
            f"other, and bubblewrap refuses to build any sandbox for the pair once the "
            f"descendant exists, so no Session in this shape would run a command. "
            f"Every Session already denies {list(ALREADY_DENIED)}"
        )
    # A second loop, after the nesting call rather than merged into the first: the
    # docstring's ordering argument is what keeps the nesting rule's platform arm
    # reachable, and merging the two would answer about a protected name for every path
    # nested under a platform deny. Split on the raw separator because a component is
    # what the runtime protects; `Environment` has already refused `//`, `.` and `..`,
    # and none of those spellings could hide a protected name from this split anyway.
    for path in environment.denied_paths:
        protected = [
            part for part in path.split("/") if part in RUNTIME_PROTECTED_NAMES
        ]
        if protected:
            raise ValueError(
                f"{path!r} names {protected[0]!r}, which the agent runtime keeps "
                f"read-only in every workspace and this pod does not create: denying "
                f"it as well makes the sandbox refuse to build, so no Session in this "
                f"shape would run a command"
            )
    return environment


def _resolved_from(row: Mapping[str, object]) -> ResolvedEnvironment:
    """Parse one stored row into a shape, the revision it is, and its retirement.

    The parse the read path owes, in one function, so the four routes that read this
    table and the placement path that reads it for a pod all get the same answer.

    Raises `ValueError` on a row the rules no longer accept, which is a fault and not a
    refusal: it means a stored shape and today's rules disagree, and handing it back
    would put a pod in a shape the platform would not register now.

    Three keys are read with `.get` and the three reasons differ, so none of them is
    "be lenient".

    `allowed_domains` is absent only from a store that omitted the column from its
    SELECT, which would be a fault -- but the value that fault must not read as is "no
    restriction on egress", and an absent value and an empty list mean the same thing
    here. The same thing is the safe one: no network.

    `revision` is absent only from a store that keeps one revision per id, which every
    in-memory stand-in in this repository is, and 1 is the only number such a store
    could mean. It is also the column's own server default, so the absent value and the
    schema agree rather than merely coinciding.

    `archived_at` is absent from a store that does not join the archive table, and that
    default is the one that is NOT fail-safe: it reads as "not retired", so a read that
    lost the join would let a retired Environment start Sessions. Nothing here can make
    that safe, so it is pinned where it can be -- against a real database, in
    `tests/adapters/test_environment_revisions_schema.py`, which asserts the join is
    there and that the timestamp arrives.
    """
    environment_id = EnvironmentId(UUID(str(row["id"])))
    stored_paths = row["denied_paths"]
    if not isinstance(stored_paths, list):
        raise ValueError(f"denied_paths for {environment_id} is not a list")
    stored_domains = row.get("allowed_domains", [])
    if not isinstance(stored_domains, list):
        raise ValueError(f"allowed_domains for {environment_id} is not a list")
    revision = row.get("revision", 1)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError(
            f"revision for {environment_id} is not a revision: {revision!r}"
        )
    archived_at = row.get("archived_at")
    if archived_at is not None and not isinstance(archived_at, datetime):
        raise ValueError(
            f"archived_at for {environment_id} is not a time: {archived_at!r}"
        )
    return ResolvedEnvironment(
        environment=parse_environment(
            environment_id=environment_id,
            tenant_id=TenantId(UUID(str(row["tenant_id"]))),
            name=str(row["name"]),
            runtime_image=str(row["runtime_image"]),
            denied_paths=tuple(str(path) for path in stored_paths),
            allowed_domains=tuple(str(domain) for domain in stored_domains),
        ),
        revision=revision,
        archived_at=archived_at,
    )


async def resolve_environment_revision(
    store: EnvironmentStore,
    environment_id: EnvironmentId,
    tenant_id: TenantId,
) -> ResolvedEnvironment:
    """The LATEST revision behind an id, or `UnknownEnvironment`.

    Latest, because an id stands for a sequence of revisions rather than for a row: an
    edit appends, so "the row with this id" stopped being a question with one answer.
    Which revision that was is returned rather than discarded, so a caller that has to
    keep running in one shape can write the number down and stop asking.

    A retired Environment resolves normally and says it is retired. Refusing here
    instead would make reading one back impossible, and reading one back is exactly what
    a tenant does after archiving it by mistake; what may not happen is a *new* Session
    or an edit, and both of those are decided by the routes that do them.
    """
    row = await store.fetch(environment_id, tenant_id)
    if row is None:
        raise UnknownEnvironment(str(environment_id))
    return _resolved_from(row)


async def resolve_environment(
    store: EnvironmentStore,
    environment_id: EnvironmentId,
    tenant_id: TenantId,
) -> Environment:
    """The shape behind an id, or `UnknownEnvironment`.

    The narrow answer, for callers that want a shape to compile and have no revision to
    pin -- so it drops the number rather than making every one of them unpack a value
    they would ignore.

    Raises `ValueError` on a row that no longer parses, which is a fault rather than a
    refusal: it means a stored shape and the rules disagree, and returning it would put
    a pod in a shape the platform would not register today.
    """
    resolved = await resolve_environment_revision(store, environment_id, tenant_id)
    return resolved.environment


async def resolve_environment_at(
    store: EnvironmentRevisions,
    environment_id: EnvironmentId,
    tenant_id: TenantId,
    revision: int,
) -> Environment:
    """The shape a Session pinned, whatever the newest revision has since become.

    The counterpart to `resolve_environment` and the reason the revision is written into
    a Session's creation event at all. Resolving the newest revision when a pod is built
    would make an edit retroactive: the Session would run in a sandbox its creator never
    agreed to, and nothing in the log would say when the reach changed. Resolving the
    pinned number makes the edit reach the next Session and no earlier one.

    Raises `UnknownEnvironment` when that id and revision name no row this tenant owns.
    In practice that means the Environment was deleted -- which the delete guard refuses
    while any Session that has not stopped holds it, so reaching this is a race with a
    delete rather than an ordinary outcome.

    A retired Environment resolves here and is deliberately not refused. Archiving
    refuses a *new* Session; a Session created before the retirement keeps running, or
    archiving would be a way to stop live work rather than a way to stop new work.
    """
    row = await store.fetch_revision(environment_id, tenant_id, revision)
    if row is None:
        raise UnknownEnvironment(f"{environment_id} has no revision {revision}")
    return _resolved_from(row).environment


async def list_environments(
    store: EnvironmentLifecycle,
    tenant_id: TenantId,
    after: tuple[int, UUID] | None,
    limit: int,
    include_archived: bool,
) -> tuple[ListedEnvironment, ...]:
    """One page of a tenant's Environments, each parsed the way a single read parses.

    Parsed rather than passed through, so a page cannot hand back a shape that
    `GET /v1/environments/{id}` would refuse to return -- two answers about one
    Environment, disagreeing, from the same store.

    That is also what makes the failure mode worth stating: one unparseable row fails
    the whole page, not just its own entry. Skipping it would publish a list that is
    quietly short, and a caller walking pages to enumerate what it owns would never
    learn the id it never saw.
    """
    rows = await store.page(tenant_id, after, limit, include_archived)
    return tuple(
        ListedEnvironment(
            resolved=_resolved_from(row),
            created_at_ms=int(str(row["created_at_ms"])),
        )
        for row in rows
    )
