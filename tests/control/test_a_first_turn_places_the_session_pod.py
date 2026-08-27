"""What a Turn does about a Session with no pod, and what it refuses to do about one
that has already completed a Turn.

Tier 1: in-memory ports, a fake `PodRunner` that records what it was asked to create,
no cluster and no container. Everything here runs in the default offline suite.

THIS FILE CANNOT SAY A POD APPEARED. A fake runner reports whatever it is told to, and
a fake kinder than production turns a total failure into a green suite. What it grades
is the branch and the ADR-004 rule. That a real pod appears in a real namespace is the
live tier's job (`tests/pod/test_the_control_plane_places_a_session_pod.py`) and the two
are not substitutes.

The fake is deliberately UNkind in the one way that matters here: `ensure` answers
STARTING, which is what a real Session pod does today, because the shim's readiness
waits on a thread the runtime cannot open until three other slices land. So the happy
case in this file also ends in a refusal -- and the assertion is that the pod was
CREATED and that the refusal names STARTING rather than ABSENT. A fake answering
RUNNING would make this file pass in a world the cluster is not in.
"""

from __future__ import annotations

import tomllib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any
from uuid import UUID, uuid4

import pytest

from managed_agent.control.catalog.environments import parse_environment
from managed_agent.control.files.attachments import AttachedFiles
from managed_agent.control.files.store import (
    FileId,
    UploadedFile,
    UploadedFileNotFound,
    content_digest,
    new_file_id,
    parse_upload_filename,
)
from managed_agent.control.pod_config.compiler import CompiledConfig
from managed_agent.control.session.placement import Placement, PodPhase, pod_name_for
from managed_agent.control.session.pods import FirstTurnPlacement
from managed_agent.control.session.turn_dispatch import TurnUndeliverable
from managed_agent.control.skills.registry import (
    SkillHeld,
    SkillListing,
    SkillRecord,
    SkillsUnresolvable,
    SkillVersionBundle,
    SkillVersionFile,
    SkillVersionRecord,
)
from managed_agent.core.ids import (
    DefinitionId,
    Seq,
    SessionId,
    SkillId,
    TenantId,
    TurnId,
    new_definition_id,
    new_session_id,
    new_turn_id,
)
from managed_agent.core.ports import (
    EventRecord,
    Resolution,
    SessionNotVisible,
)
from managed_agent.core.registration.definition import AgentDefinition, VersionFact
from managed_agent.core.registration.environment import (
    Environment,
    EnvironmentId,
    new_environment_id,
)
from managed_agent.core.registration.skill import ValidatedSkill
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.session.session_token import (
    InvalidSessionToken,
    verify_session_token,
)
from managed_agent.core.vocabulary import lifecycle, placement, resource, turn
from managed_agent.session_shim.pod_channel import HttpPodDispatch

IMAGE = "registry.map.internal/session@sha256:" + "a" * 64
OTHER_IMAGE = "registry.map.internal/session@sha256:" + "b" * 64
TOOL_GATEWAY_URL = "http://tool-gateway.map-dev.svc.cluster.local/mcp"
MODEL_GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"
SESSION_TOKEN_KEY = b"a signing key that is thirty-two"
LIFETIME_S = 86_400
NOW_MS = 1_700_000_000_000
_SKILLS_SHA = "0" * 39 + "a"


# --------------------------------------------------------------------------------------
# The doubles. Each one refuses every method this file does not exercise, so a case that
# started reaching for a collaborator it was not written against says so instead of
# silently getting a default.
# --------------------------------------------------------------------------------------


class RecordingCluster:
    """A `PodRunner` that records every create and answers STARTING, never RUNNING.

    STARTING because that is what a real Session pod does today: `_phase_of` requires
    every container ready and the shim's readiness probe waits on a thread the runtime
    cannot open yet. A fake that answered RUNNING would make the happy case here assert
    a Turn that succeeds, in a cluster where no Turn succeeds -- and it would keep
    passing after the real pod stopped reaching RUNNING for some new reason.
    """

    def __init__(self) -> None:
        self.created: list[tuple[str, CompiledConfig]] = []

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        self.created.append((pod_name, compiled))
        return PodPhase.STARTING

    async def phase_of(self, pod_name: str) -> PodPhase:
        """What the cluster says now: STARTING once created, ABSENT before.

        Answered off `created` rather than from a fixed value, so the second `locate`
        the transport does after a placement sees the consequence of that placement --
        which is the whole reason the transport reads the phase twice.
        """
        if any(name == pod_name for name, _ in self.created):
            return PodPhase.STARTING
        return PodPhase.ABSENT

    async def remove(self, pod_name: str) -> None:
        """A no-op, where this used to refuse.

        Under ADR-041 a pod is leased for one Turn, so every dispatch releases one and
        the refusal written here asserted the opposite of the contract. Nothing is
        recorded because no case in this file grades which pod went -- the lease itself
        is graded in `tests/control/test_a_pod_is_leased_for_one_turn.py`, against a
        cluster whose phase actually reflects the removal.
        """


class FixedPhase:
    """A cluster stuck in one phase, for the rows where placement must not be reached.

    Separate from `RecordingCluster` because those rows assert an absence -- that
    nothing was created -- and an absence asserted against a runner that could have
    created something is the only version of that assertion worth having.
    """

    def __init__(self, phase: PodPhase) -> None:
        self._phase = phase
        self.created: list[tuple[str, CompiledConfig]] = []

    async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
        self.created.append((pod_name, compiled))
        return self._phase

    async def phase_of(self, pod_name: str) -> PodPhase:
        return self._phase

    async def remove(self, pod_name: str) -> None:
        """A no-op, where this used to refuse.

        Under ADR-041 a pod is leased for one Turn, so every dispatch releases one and
        the refusal written here asserted the opposite of the contract. Nothing is
        recorded because no case in this file grades which pod went -- the lease itself
        is graded in `tests/control/test_a_pod_is_leased_for_one_turn.py`, against a
        cluster whose phase actually reflects the removal.
        """


@dataclass(frozen=True, slots=True)
class _Event:
    session_id: SessionId
    seq: Seq
    type: str
    payload: dict[str, object]


@dataclass
class FakeLog:
    """A Session's events, paged the way the port promises they arrive.

    Pages at two records rather than handing the whole log back in one answer, because
    the port caps a read and a caller that stopped after one page would read a Session's
    first two events as its whole history -- which for the ADR-004 question is the
    difference between "no Turn has completed" and "I did not look".
    """

    events: list[_Event] = field(default_factory=list)
    reads: int = 0

    def append(self, session_id: SessionId, type_: str, **payload: object) -> None:
        seq = Seq(len([e for e in self.events if e.session_id == session_id]) + 1)
        self.events.append(_Event(session_id, seq, type_, payload))

    async def read(
        self, session_id: SessionId, start: Seq, end: Seq, limit: int = 500
    ) -> Sequence[EventRecord]:
        self.reads += 1
        mine = [e for e in self.events if e.session_id == session_id]
        return [e for e in mine if start <= e.seq <= end][:2]

    def follow(self, session_id: SessionId, after: Seq) -> AsyncIterator[EventRecord]:
        raise AssertionError("a test in this file followed the log")

    async def retained_floor(self, session_id: SessionId) -> Seq:
        raise AssertionError("a test in this file asked for a retained floor")


@dataclass
class FakeSessions:
    """Creation facts by Session id, with no tenant in the key.

    The narrow read the placement path uses. What the real adapter does with a tenant
    column is `tests/adapters/`'s; what this holds is the one property the placement
    path depends on -- that the record it gets back carries the owning tenant, which is
    what every resolution below is then scoped by.
    """

    records: dict[SessionId, SessionRecord] = field(default_factory=dict)

    async def read(self, session_id: SessionId) -> SessionRecord:
        try:
            return self.records[session_id]
        except KeyError as absent:
            raise SessionNotVisible(str(session_id)) from absent


@dataclass
class FakeEnvironments:
    """Revisions keyed the way migration 0022 keys them: `(id, tenant, revision)`.

    Keyed on three columns and not two, because an id stopped standing for a row when an
    edit began appending. A stand-in keyed on two could hold only one shape per id, and
    a test asking for the shape a Session pinned would get whatever shape was there --
    which is indistinguishable from the placement path reading the newest revision, the
    exact defect the pinned read exists to prevent.
    """

    rows: dict[tuple[UUID, UUID, int], dict[str, object]] = field(default_factory=dict)

    def add(self, environment: Environment) -> int:
        """Append the next revision of this id, and say which number it got.

        Computed from what is already held rather than passed in, so a test cannot
        accidentally write revision 5 with no 2, 3 or 4 -- a gap the real table's
        `max(revision) + 1` cannot produce.
        """
        held = [
            revision
            for (one, owner, revision) in self.rows
            if one == environment.id and owner == environment.tenant_id
        ]
        revision = max(held, default=0) + 1
        self.rows[(environment.id, environment.tenant_id, revision)] = {
            "id": str(environment.id),
            "tenant_id": str(environment.tenant_id),
            "name": environment.name,
            "runtime_image": environment.runtime_image,
            "denied_paths": list(environment.denied_paths),
            "revision": revision,
        }
        return revision

    async def insert(self, environment: Environment, /) -> None:
        self.add(environment)

    async def fetch(
        self, environment_id: EnvironmentId, tenant_id: TenantId, /
    ) -> Mapping[str, object] | None:
        """The newest revision behind this id, which is what `fetch` means now."""
        held = [
            revision
            for (one, owner, revision) in self.rows
            if one == environment_id and owner == tenant_id
        ]
        if not held:
            return None
        return self.rows[(environment_id, tenant_id, max(held))]

    async def fetch_revision(
        self, environment_id: EnvironmentId, tenant_id: TenantId, revision: int, /
    ) -> Mapping[str, object] | None:
        """The one revision named, or None -- including for a number never written.

        None rather than the newest for an unknown number, and that is the half of this
        fake that does the work. A stand-in answering the newest row whatever was asked
        would let the placement path read any revision and still pass every test here.
        """
        return self.rows.get((environment_id, tenant_id, revision))


@dataclass(frozen=True, slots=True)
class _Resolved:
    definition: AgentDefinition
    revision: int


class AlwaysResolves:
    """A definition registry that resolves any id to revision 1."""

    async def register(
        self,
        definition_id: DefinitionId,
        tenant_id: TenantId,
        definition: AgentDefinition,
    ) -> int:
        return 1

    async def resolve(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Resolution:
        return _Resolved(_a_definition(), 1)

    async def list_versions(
        self, definition_id: DefinitionId, tenant_id: TenantId
    ) -> Sequence[VersionFact]:
        return (VersionFact(revision=1, archived=False),)

    async def read_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> AgentDefinition | None:
        return _a_definition()

    async def archive_version(
        self, definition_id: DefinitionId, tenant_id: TenantId, revision: int
    ) -> bool:
        raise AssertionError("a test in this file retired a definition version")


class NoSkillsHeld:
    """A skill store holding nothing, satisfying the port structurally.

    The definitions in this file attach no skill, so `read_attached` never asks this
    anything -- only the repository route is consulted, and it answers empty. Every
    other method raises rather than returning, because a test in this file that started
    uploading a skill should say so loudly instead of resolving against a store that
    silently forgot it. Skill resolution is graded beside the delivery it feeds; what
    this file is about is which pod phase a Turn does what with.
    """

    async def read_repository_skills(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> tuple[ValidatedSkill, ...]:
        return ()

    async def page_uploaded_skills(
        self, tenant_id: TenantId, after: tuple[str, SkillId] | None, limit: int
    ) -> tuple[SkillListing, ...]:
        raise AssertionError("a test in this file listed a tenant's skills")

    async def read_skills(
        self, tenant_id: TenantId, skill_ids: Sequence[SkillId]
    ) -> tuple[SkillRecord, ...]:
        raise AssertionError("a definition in this file attached a skill")

    async def add_skill(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        skill: ValidatedSkill,
        *,
        display_name: str | None,
    ) -> None:
        raise AssertionError("a test in this file uploaded a skill")

    async def set_repository_skills(
        self,
        tenant_id: TenantId,
        repository: str,
        revision: str,
        skills: Sequence[ValidatedSkill],
    ) -> int:
        raise AssertionError("a test in this file submitted a skill repository")

    async def read_skill(
        self, tenant_id: TenantId, skill_id: SkillId
    ) -> SkillHeld | None:
        raise AssertionError("a test in this file read one skill back")

    async def delete_skill(self, tenant_id: TenantId, skill_id: SkillId) -> None:
        raise AssertionError("a test in this file deleted a skill")

    async def add_skill_version(
        self,
        tenant_id: TenantId,
        skill_id: SkillId,
        version: int,
        skill: ValidatedSkill,
        directory: str,
        files: Sequence[SkillVersionFile],
    ) -> None:
        raise AssertionError("a test in this file wrote a skill version")

    async def page_skill_versions(
        self, tenant_id: TenantId, skill_id: SkillId, after: int | None, limit: int
    ) -> tuple[SkillVersionRecord, ...]:
        raise AssertionError("a test in this file paged a skill's versions")

    async def read_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionRecord | None:
        raise AssertionError("a test in this file read one skill version")

    async def read_skill_version_bundle(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> SkillVersionBundle | None:
        raise AssertionError("a test in this file downloaded a skill version")

    async def retire_skill_version(
        self, tenant_id: TenantId, skill_id: SkillId, version: int
    ) -> None:
        raise AssertionError("a test in this file retired a skill version")


class HoldsOneRepositorySkill(NoSkillsHeld):
    """The repository route with one skill in it.

    Subclasses the empty one so the three refusing methods keep refusing: what this
    changes is the single answer the placement path reads, and nothing else.
    """

    async def read_repository_skills(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> tuple[ValidatedSkill, ...]:
        return (A_SKILL,)


class SkillsThatWillNotResolve(NoSkillsHeld):
    """A store that refuses to resolve, in the type the resolution policy raises."""

    async def read_repository_skills(
        self, tenant_id: TenantId, repository: str, revision: str
    ) -> tuple[ValidatedSkill, ...]:
        raise SkillsUnresolvable("two skills resolve to one name", resolved_skills=2)


class HoldsTheseFiles:
    """An uploaded-file reader over a dict, satisfying the port structurally.

    Only `fetch` exists, because only `fetch` is in the port. What a real store adds --
    the size limit, the hash check on the way back, the tenant term in the WHERE clause
    -- is graded against PostgreSQL in `tests/adapters/`; what stands in here is the
    holding, so the placement policy above it can be exercised without a bucket.
    """

    def __init__(self, held: dict[FileId, tuple[str, bytes]]) -> None:
        self.held = held

    async def fetch(
        self, *, tenant_id: TenantId, file_id: FileId
    ) -> tuple[UploadedFile, bytes]:
        if file_id not in self.held:
            raise UploadedFileNotFound(str(file_id))
        name, body = self.held[file_id]
        return (
            UploadedFile(
                id=file_id,
                tenant_id=tenant_id,
                filename=parse_upload_filename(name),
                media_type="text/markdown",
                byte_length=len(body),
                content_sha256=content_digest(body),
            ),
            body,
        )


class RecordingFilePlacement:
    """A `FilePlacement` that records what it was asked to write into the pod."""

    def __init__(self) -> None:
        self.placed: list[tuple[SessionId, str, bytes]] = []

    async def place_file(
        self, session_id: SessionId, name: str, body: bytes, /
    ) -> None:
        self.placed.append((session_id, name, body))


class FrozenClock:
    def now_epoch_ms(self) -> int:
        return NOW_MS


class UnusedAppend:
    """The event log a dispatch appends to. One type is expected; the rest are not.

    This began as a fake that refused every append, on the premise that no case here
    reaches a pod dialled and RUNNING so no Turn event can be produced. That premise
    still holds for Turn events and stopped holding for the two events a placement
    itself writes -- `session.placing` before the wait, which is the whole point of it,
    and `session.resumed` after it, which says a Session that had no pod now has one.
    Both are reached by every case here that places a pod, and refusing either turns
    passing tests red without anything being wrong.

    Recorded rather than ignored, so a case that wants to assert either event can.
    Refusal is kept for every other type: a case that somehow got past the phase check
    should still fail on that rather than on an assertion about a fake HTTP exchange
    nobody wrote. What the two events say is graded in
    `test_placement_is_visible_to_the_tenant.py`, where the log is real; this only has
    to stop being in the way.
    """

    _PLACEMENT_EVENTS = frozenset(
        {placement.SESSION_PLACING, lifecycle.SESSION_RESUMED}
    )

    def __init__(self) -> None:
        self.placements: list[tuple[SessionId, str, dict[str, object]]] = []

    async def append(
        self, session_id: SessionId, type_: str, payload: dict[str, object]
    ) -> Seq:
        if type_ in self._PLACEMENT_EVENTS:
            self.placements.append((session_id, type_, payload))
            return Seq(len(self.placements))
        raise AssertionError(f"a test in this file appended {type_}")


class UnusedCompletion:
    async def turn_completed(self, session_id: SessionId, turn_id: TurnId) -> None:
        raise AssertionError("a test in this file completed a Turn")


A_SKILL = ValidatedSkill(
    name="pdf",
    description="Build a PDF report.",
    text="---\nname: pdf\ndescription: Build a PDF report.\n---\nSteps.\n",
)


def _a_definition() -> AgentDefinition:
    return AgentDefinition.model_validate(
        {
            "name": "placement-fixture",
            "instructions": "irrelevant to these tests",
            "model": "gpt-5-codex",
            "skills_repository": "git@github.com:acme/skills.git",
            "skills_revision": _SKILLS_SHA,
        }
    )


def _an_environment(tenant_id: TenantId, *, image: str = IMAGE) -> Environment:
    return parse_environment(
        environment_id=new_environment_id(),
        tenant_id=tenant_id,
        name="analysis",
        runtime_image=image,
        denied_paths=(),
    )


@dataclass(frozen=True, slots=True)
class Harness:
    """One Session, its collaborators, and the transport a Turn goes through."""

    session_id: SessionId
    environment: Environment
    environments: FakeEnvironments
    cluster: Any
    log: FakeLog
    dispatch: HttpPodDispatch

    async def take_a_turn(self) -> None:
        await self.dispatch.dispatch(self.session_id, new_turn_id(), "summarise it")


def _harness(
    *,
    cluster: Any,
    completed_a_turn: bool = False,
    image: str = IMAGE,
    tenant_id: TenantId | None = None,
    skills: Any = None,
    attachments: AttachedFiles | None = None,
    file_ids: tuple[FileId, ...] = (),
    attached_later: tuple[FileId, ...] = (),
    attached_after_a_turn: tuple[FileId, ...] = (),
    attach_before_creation: tuple[FileId, ...] = (),
    environment_revision: int | None = None,
) -> Harness:
    """Wire one Session end to end: a registry row, a creation event, a transport.

    Built through `HttpPodDispatch` rather than by calling `FirstTurnPlacement`
    directly, because what a reader needs proved is that a Turn places a pod -- not
    that one method calls another.
    """
    tenant = tenant_id if tenant_id is not None else TenantId(uuid4())
    session_id = new_session_id()
    environment = _an_environment(tenant, image=image)
    environments = FakeEnvironments()
    environments.add(environment)
    sessions = FakeSessions()
    sessions.records[session_id] = SessionRecord(
        id=session_id,
        tenant_id=tenant,
        definition_id=new_definition_id(),
        definition_revision="1",
        grant=frozenset(),
        scope=(),
        budget_minor_units=10_000,
        budget_currency="USD",
        retention_days=30,
    )
    # `environment_revision` is written only when a test names one, and the omission is
    # the point. A Session created before Environments were revisioned has no such key,
    # and the placement path reads that absence as revision 1 -- which migration 0022
    # made true by giving every row that already existed exactly that number. Leaving it
    # out by default means every other test in this file exercises the pre-revision log
    # a real deployment still holds.
    log = FakeLog()
    # Appended BEFORE the creation event, which is a shape no route can produce: the
    # attach route folds the log and refuses a Session with no state. It is here so the
    # guard that ignores such an event has an arm that can fail -- without it, deleting
    # the guard changes no test.
    for early in attach_before_creation:
        log.append(session_id, resource.SESSION_FILE_ATTACHED, file_id=str(early))
    pinned: dict[str, object] = (
        {}
        if environment_revision is None
        else {"environment_revision": environment_revision}
    )
    log.append(
        session_id,
        lifecycle.SESSION_CREATED,
        environment_id=str(environment.id),
        definition_revision=1,
        file_ids=[str(one) for one in file_ids],
        **pinned,
    )
    # After the creation event and before any Turn, which is one of the two places a
    # real attach lands: `POST /v1/sessions/{id}/resources` accepts one whenever the
    # Session would still take a Turn, and no pod exists to push into, so the event sits
    # here waiting for the placement this harness drives.
    for late in attached_later:
        log.append(session_id, resource.SESSION_FILE_ATTACHED, file_id=str(late))
    if completed_a_turn:
        log.append(session_id, turn.TURN_SUBMITTED, turn_id=str(new_turn_id()))
        log.append(session_id, turn.TURN_COMPLETED, turn_id=str(new_turn_id()))
    # The other place, and it needs its own parameter because position in the log is the
    # whole difference: this is an attach to a Session that has already run. The route
    # used to push those bytes itself, on the reading that a Session past its first Turn
    # had a pod standing; under ADR-041 it has none, so this event is delivered the same
    # way every other one is -- by the next Turn's placement, which is what the fold
    # below has to pick up.
    for after in attached_after_a_turn:
        log.append(session_id, resource.SESSION_FILE_ATTACHED, file_id=str(after))
    placement = Placement(cluster)
    return Harness(
        session_id=session_id,
        environment=environment,
        environments=environments,
        cluster=cluster,
        log=log,
        dispatch=HttpPodDispatch(
            placement=placement,
            pods=FirstTurnPlacement(
                placement=placement,
                sessions=sessions,
                environments=environments,
                definitions=AlwaysResolves(),
                events=log,
                skills=skills if skills is not None else NoSkillsHeld(),
                attachments=(
                    attachments
                    if attachments is not None
                    else AttachedFiles(HoldsTheseFiles({}), RecordingFilePlacement())
                ),
                clock=FrozenClock(),
                session_token_key=SESSION_TOKEN_KEY,
                session_token_lifetime_s=LIFETIME_S,
                tool_gateway_url=TOOL_GATEWAY_URL,
                model_gateway_url=MODEL_GATEWAY_URL,
            ),
            log=UnusedAppend(),
            on_completed=UnusedCompletion(),
            namespace="map-test",
            token_key=b"a shim signing key",
        ),
    )


# --------------------------------------------------------------------------------------
# The decision table D2 writes down, as five rows nobody can add a phase to without
# also saying what happens on it.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "completed_a_turn", "creates_a_pod", "refusal"),
    [
        (PodPhase.ABSENT, False, True, "starting"),
        (PodPhase.ABSENT, True, True, "starting"),
        (PodPhase.STARTING, False, False, "starting"),
        (PodPhase.GONE, False, True, "gone"),
        (PodPhase.RUNNING, False, False, None),
    ],
)
async def test_what_a_turn_does_about_each_phase_the_cluster_can_report(
    phase: PodPhase, completed_a_turn: bool, creates_a_pod: bool, refusal: str | None
) -> None:
    """One row per phase, and the two ABSENT rows say the same thing on purpose.

    They did not always. `(ABSENT, completed_a_turn=True)` used to assert that NOTHING
    was placed, because a Session with history could not be given a pod at all. It is
    placed now, and what makes it safe is not a different decision here but a different
    compiled value -- which the case below asserts, because it is the part an absence
    could never have shown.

    The RUNNING row expects no refusal from the phase check and does not reach a real
    pod either: it fails at the HTTP call, which is what a transport with no shim to
    dial does. What it proves here is that placement was not attempted for a Session
    that already has a pod.

    **The GONE row places now, and did not before.** GONE covers a pod whose object is
    still addressable but which nothing may be dispatched into -- one stamped for
    deletion, or one that reached Succeeded or Failed. When a Session held a pod for its
    whole life, finding either meant something had gone wrong that a new pod would not
    fix, so the Turn refused without placing. Under ADR-041's per-Turn lease the
    commonest way to find one is that the previous Turn released it a moment ago and the
    grace period has not run out, so refusing failed every second Turn inside that
    window. The refusal is still expected here because `FixedPhase` never leaves the
    phase it was built with -- a cluster where the replacement does not come up either,
    which is a real ending and the right one to refuse on.

    STARTING keeps its row unchanged, and the difference is worth saying: a pod on its
    way up belongs to a Turn that is already placing it, and a second placement over it
    would be two pods for one Session.
    """
    cluster = RecordingCluster() if phase is PodPhase.ABSENT else FixedPhase(phase)
    harness = _harness(cluster=cluster, completed_a_turn=completed_a_turn)

    with pytest.raises((TurnUndeliverable, Exception)) as raised:
        await harness.take_a_turn()

    if refusal is not None:
        assert isinstance(raised.value, TurnUndeliverable)
        assert refusal in str(raised.value), raised.value
    created = [name for name, _ in cluster.created]
    if creates_a_pod:
        assert created == [pod_name_for(harness.session_id)], created
    else:
        assert created == [], created


async def test_a_session_that_completed_a_turn_is_placed_to_continue_its_thread() -> (
    None
):
    """The ADR-004 rule, on its own, because it is the one that is silent when broken.

    A pod that opens a NEW thread for a Session with history starts empty while the
    Rollout holds a conversation whose compaction checkpoints have already folded it.
    The tenant pays for the replay and the platform reports success while doing it -- so
    the failure mode of getting this wrong is a Turn that *works*, which no assertion
    about a status code would catch. `resuming` is the one value that separates the two
    outcomes anywhere in this process, so it is asserted directly rather than through
    something it causes.

    Driven beside a Session that has NOT completed a Turn, from the identical harness,
    because "the flag is true" is satisfied by a placement path that hard-codes it true
    -- which would refuse every first placement in the fleet.
    """
    resuming = _harness(cluster=RecordingCluster(), completed_a_turn=True)
    fresh = _harness(cluster=RecordingCluster(), completed_a_turn=False)

    for harness in (resuming, fresh):
        with pytest.raises(TurnUndeliverable):
            await harness.take_a_turn()

    assert [c.resuming for _, c in resuming.cluster.created] == [True]
    assert [c.resuming for _, c in fresh.cluster.created] == [False]


async def test_a_resuming_placement_differs_in_that_one_value_and_nothing_else() -> (
    None
):
    """The second pod for a Session is the same pod, not a similar one.

    Everything a Session's shape is built from is read back out of `session.created`
    rather than resolved afresh at placement, so a Session that ran a month ago comes
    back on the Environment revision and the definition it pinned. If that ever stopped
    being true, a resume would silently move a live Session into another sandbox -- and
    the symptom would be a Turn behaving differently with nothing in the log saying when
    the world changed.

    Compared field by field with `resuming` excluded, so a value that starts differing
    fails here by name instead of being absorbed into a shape check nobody reads. The
    Session token is excluded too and only that: it carries an expiry minted from the
    clock, so two placements of one Session differ there by construction.
    """
    resuming = _harness(cluster=RecordingCluster(), completed_a_turn=True)
    fresh = _harness(cluster=RecordingCluster(), completed_a_turn=False)

    for harness in (resuming, fresh):
        with pytest.raises(TurnUndeliverable):
            await harness.take_a_turn()

    (_, on_resume) = resuming.cluster.created[0]
    (_, on_first) = fresh.cluster.created[0]
    moving = {"resuming", "session_id", "tenant_id", "config_toml"}
    differing = {
        field.name
        for field in fields(CompiledConfig)
        if field.name not in moving
        and getattr(on_resume, field.name) != getattr(on_first, field.name)
    }
    assert differing == set(), differing
    assert on_resume.resuming is not on_first.resuming


async def test_the_pod_is_compiled_against_the_environment_the_session_named() -> None:
    """Two Sessions on two environments compile to two different runtime images.

    The id is read out of the `session.created` payload, because a Session's registry
    row does not carry one. A reader who assumes it does will look at `SessionRecord` --
    nine fields, and no environment among them -- and this is the case that fails if
    somebody later "simplifies" the log read away, or hard-codes one shape into the
    placement path.
    """
    one = _harness(cluster=RecordingCluster(), image=IMAGE)
    other = _harness(cluster=RecordingCluster(), image=OTHER_IMAGE)

    for harness in (one, other):
        with pytest.raises(TurnUndeliverable):
            await harness.take_a_turn()

    assert [c.runtime_image for _, c in one.cluster.created] == [IMAGE]
    assert [c.runtime_image for _, c in other.cluster.created] == [OTHER_IMAGE]


def _the_same_environment_with(environment: Environment, image: str) -> Environment:
    """The same id and tenant, a different runtime image: what an edit produces."""
    return parse_environment(
        environment_id=environment.id,
        tenant_id=environment.tenant_id,
        name=environment.name,
        runtime_image=image,
        denied_paths=environment.denied_paths,
    )


async def test_an_edit_after_creation_does_not_reach_a_session_that_pinned_it() -> None:
    """An edit made after creation does not reach a Session that pinned the old shape.

    The whole reason the revision is written into `session.created`. Resolving the
    newest revision when the pod is built would make every edit retroactive: this
    Session's creator agreed to one sandbox and the pod would come up in another, mid-
    run, with nothing anywhere recording when the reach changed. The Environment table's
    append-only trigger was installed to make that impossible to do to the *record*;
    this is what makes it impossible to do to the *pod*.

    The runtime image is the observable because it is the one field of a compiled config
    that a reader can trace back to a single revision by eye. The property is not about
    images.
    """
    harness = _harness(cluster=RecordingCluster(), image=IMAGE, environment_revision=1)
    appended = harness.environments.add(
        _the_same_environment_with(harness.environment, OTHER_IMAGE)
    )
    assert appended == 2, "the edit has to be a second revision for this to prove much"

    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert [c.runtime_image for _, c in harness.cluster.created] == [IMAGE]


async def test_a_session_pinned_at_the_newer_revision_gets_the_newer_shape() -> None:
    """A Session pinned at the newer revision gets the newer shape.

    The arms-disagree half of the test above, and without it that one passes over a
    placement path that always read revision 1 -- or one that read the row a dict
    happened to yield first. What is being asserted across the pair is that the pod
    follows the *pinned number*, in both directions, and not that it follows the oldest.
    """
    harness = _harness(cluster=RecordingCluster(), image=IMAGE, environment_revision=2)
    harness.environments.add(
        _the_same_environment_with(harness.environment, OTHER_IMAGE)
    )

    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert [c.runtime_image for _, c in harness.cluster.created] == [OTHER_IMAGE]


async def test_a_session_from_before_revisions_existed_keeps_its_shape() -> None:
    """A Session created before revisions existed keeps its shape across a later edit.

    The case a real deployment has and a fresh one does not: every `session.created`
    event written before migration 0022 names no revision at all. Reading that absence
    as revision 1 is not a fallback -- 0022 gave every row that existed at that moment
    exactly that number, so revision 1 *is* the shape those Sessions named.

    Which makes this the one that would break silently. A default of "newest" would pass
    every other test in this file, because nothing else here appends a second revision
    -- and it would place every pre-0022 Session into whatever shape its Environment has
    drifted to since.
    """
    harness = _harness(cluster=RecordingCluster(), image=IMAGE)
    assert "environment_revision" not in harness.log.events[0].payload
    harness.environments.add(
        _the_same_environment_with(harness.environment, OTHER_IMAGE)
    )

    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert [c.runtime_image for _, c in harness.cluster.created] == [IMAGE]


async def test_the_pod_carries_the_skills_the_store_resolved_for_it() -> None:
    """The store is asked, and its answer is what the pod is compiled with.

    The whole delivery chain is longer than this -- the Secret key, the volume
    projection, the mount path -- and every link past here is graded in
    `tests/adapters/test_pod_runner.py`. This is the link nothing else can see: a
    placement path that never asked the store would compile a Session whose definition
    attaches skills into a pod with none, and every assertion downstream of it would
    still pass, on an empty tuple.
    """
    without = _harness(cluster=RecordingCluster())
    with_one = _harness(cluster=RecordingCluster(), skills=HoldsOneRepositorySkill())

    for harness in (without, with_one):
        with pytest.raises(TurnUndeliverable):
            await harness.take_a_turn()

    assert [c.skill_files for _, c in without.cluster.created] == [()]
    (compiled,) = [c for _, c in with_one.cluster.created]
    assert [(f.relative_path, f.text) for f in compiled.skill_files] == [
        ("skills/pdf/SKILL.md", A_SKILL.text)
    ]


async def test_a_skill_that_will_not_resolve_refuses_the_turn_it_cannot_serve() -> None:
    """In the one type the transport carries, and before any pod is created.

    Not a bare 500: `ensure_for` names the refusal types it converts, and a resolution
    failure escaping outside that set would reach a tenant as an unexplained error with
    no Turn closed behind it. The absence of a create is the other half -- resolving is
    ordered ahead of compiling so that a Session which cannot be given its skills is
    never given a pod either, and a pod started and then found wanting is a Session
    running with skills silently missing.
    """
    harness = _harness(cluster=RecordingCluster(), skills=SkillsThatWillNotResolve())
    with pytest.raises(TurnUndeliverable, match="SkillsUnresolvable"):
        await harness.take_a_turn()
    assert harness.cluster.created == []


async def test_the_files_a_session_attached_are_written_into_its_pod() -> None:
    """Both of them, under the names they were uploaded with, in the order named.

    Order is asserted rather than compared as a set because it is the one thing that
    makes two placements of the same Session produce the same workspace -- and because
    a delivery that happened to reverse them would satisfy every containment check.
    """
    one, other = new_file_id(), new_file_id()
    placement = RecordingFilePlacement()
    harness = _harness(
        cluster=RecordingCluster(),
        attachments=AttachedFiles(
            HoldsTheseFiles(
                {one: ("brief.md", b"# Brief\n"), other: ("data.csv", b"a,b\n1,2\n")}
            ),
            placement,
        ),
        file_ids=(one, other),
    )
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert [(name, body) for _, name, body in placement.placed] == [
        ("brief.md", b"# Brief\n"),
        ("data.csv", b"a,b\n1,2\n"),
    ]
    assert {session for session, _, _ in placement.placed} == {harness.session_id}


async def test_a_session_attaching_nothing_has_nothing_written_into_it() -> None:
    """No push at all, rather than a push of an empty set.

    The pod's workspace has to be byte-identical to what it was before attachments
    existed for every Session that names none, which is nearly all of them. A delivery
    that created an empty `files/` directory or touched a placeholder would put a path
    into every Session on the platform that nothing asked for.
    """
    placement = RecordingFilePlacement()
    harness = _harness(
        cluster=RecordingCluster(),
        attachments=AttachedFiles(HoldsTheseFiles({}), placement),
    )
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()
    assert placement.placed == []


async def test_a_file_that_cannot_be_read_refuses_the_turn_and_places_nothing() -> None:
    """No partial delivery, and the refusal arrives in the type the transport carries.

    The two assertions are one claim in two halves. A Session holding a prefix of its
    attachments starts, reads the first file, and reports the second as missing -- which
    a tenant reads as the platform losing an upload. So the refusal has to happen before
    anything is written, and `placed == []` is the only thing that says it did.
    """
    present, absent = new_file_id(), new_file_id()
    placement = RecordingFilePlacement()
    harness = _harness(
        cluster=RecordingCluster(),
        attachments=AttachedFiles(
            HoldsTheseFiles({present: ("brief.md", b"# Brief\n")}), placement
        ),
        file_ids=(present, absent),
    )
    with pytest.raises(TurnUndeliverable, match="FilesNotPlaceable"):
        await harness.take_a_turn()
    assert placement.placed == []


async def test_the_files_are_written_only_after_the_pod_exists() -> None:
    """Ordering, asserted against the cluster rather than read off the source.

    A push issued before the pod is placed has no shim to reach and fails every time;
    one issued before the pod is READY reaches a container that may not be serving yet.
    `place` returns when every container is ready, so what this checks is that the push
    is downstream of it -- and the only evidence outside the code is that the cluster
    recorded its create first.
    """
    attached = new_file_id()
    order: list[str] = []

    class NotesTheOrder(RecordingFilePlacement):
        async def place_file(
            self, session_id: SessionId, name: str, body: bytes, /
        ) -> None:
            order.append(f"placed {name}")
            await super().place_file(session_id, name, body)

    class NotesTheCreate(RecordingCluster):
        async def ensure(self, pod_name: str, compiled: CompiledConfig) -> PodPhase:
            order.append("pod created")
            return await super().ensure(pod_name, compiled)

    harness = _harness(
        cluster=NotesTheCreate(),
        attachments=AttachedFiles(
            HoldsTheseFiles({attached: ("brief.md", b"# Brief\n")}), NotesTheOrder()
        ),
        file_ids=(attached,),
    )
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()
    assert order == ["pod created", "placed brief.md"]


async def test_a_file_attached_after_creation_is_written_into_the_pod() -> None:
    """A late attach reaches the workspace, and this is the half the route cannot show.

    `POST /v1/sessions/{id}/resources` appends and pushes nothing when the Session has
    no pod, on the grounds that this placement will deliver the file. That is a claim
    about this code, not about the route, and every test of the route passes whether or
    not it holds -- the route answers 201 either way and the list route reads the same
    event it appended. Here the event is in the log and the assertion is that bytes came
    out.

    Ordered created-first-then-attached, and both are asserted, because the workspace is
    one flat directory and the order files are written in is what makes two placements
    of one Session produce the same one.
    """
    created_with, attached = new_file_id(), new_file_id()
    placement = RecordingFilePlacement()
    harness = _harness(
        cluster=RecordingCluster(),
        attachments=AttachedFiles(
            HoldsTheseFiles(
                {
                    created_with: ("brief.md", b"# Brief\n"),
                    attached: ("appendix.md", b"# Appendix\n"),
                }
            ),
            placement,
        ),
        file_ids=(created_with,),
        attached_later=(attached,),
    )
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert [(name, body) for _, name, body in placement.placed] == [
        ("brief.md", b"# Brief\n"),
        ("appendix.md", b"# Appendix\n"),
    ]


async def test_two_attaches_reach_the_pod_in_the_order_they_were_appended() -> None:
    """Log order, not id order and not name order.

    The two are appended so that neither their names nor the ids they were minted with
    put them in the order the log does -- a fold that sorted the set would come out
    different. Also the case where a Session was created holding nothing and every file
    it has arrived late, which is the shape a caller that uploads after creating gets.
    """
    second, first = new_file_id(), new_file_id()
    placement = RecordingFilePlacement()
    harness = _harness(
        cluster=RecordingCluster(),
        attachments=AttachedFiles(
            HoldsTheseFiles(
                {
                    first: ("zeta.md", b"z\n"),
                    second: ("alpha.md", b"a\n"),
                }
            ),
            placement,
        ),
        attached_later=(first, second),
    )
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert [name for _, name, _ in placement.placed] == ["zeta.md", "alpha.md"]


async def test_an_attach_ahead_of_the_creation_event_is_ignored() -> None:
    """An attach the creation event has not reached yet is discarded by it.

    Unreachable through the routes -- the attach route folds the log and refuses a
    Session with no state -- and pinned anyway, because what discards it is not a guard.
    The creation branch ASSIGNS the attached set rather than extending it, so anything
    read before it is overwritten. A guard was written for this and deleted: with the
    guard removed this test still passed, which is what proved the guard was doing
    nothing.

    So what this actually protects is the assignment. Change `attached =
    _file_ids_in(...)` to extend instead of assign -- which is the natural edit once a
    second place adds to the same variable -- and a Session would place a file no route
    ever attached to it.

    The Session still places its own file, which is the second half. A fold that gave up
    on a stray event would also satisfy "places nothing for the stray".
    """
    stray, real = new_file_id(), new_file_id()
    placement = RecordingFilePlacement()
    harness = _harness(
        cluster=RecordingCluster(),
        attachments=AttachedFiles(
            HoldsTheseFiles({real: ("brief.md", b"# Brief\n")}),
            placement,
        ),
        file_ids=(real,),
        attach_before_creation=(stray,),
    )
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert [name for _, name, _ in placement.placed] == ["brief.md"]


async def test_the_compiled_document_carries_what_the_deployment_gave_it() -> None:
    """The four deployment values reach the pod's document rather than a default.

    Asserted on the rendered `config.toml` and not on the arguments, because what a
    Session runs against is the document -- a value accepted at the seam and dropped
    before rendering is exactly the failure a constructor-argument assertion cannot see.
    The expiry is asserted as an arithmetic result of the clock and the lifetime, so a
    lifetime silently ignored fails here.
    """
    harness = _harness(cluster=RecordingCluster())
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    (_, compiled) = harness.cluster.created[0]
    document = tomllib.loads(compiled.config_toml)
    rendered = compiled.config_toml
    assert TOOL_GATEWAY_URL in rendered, rendered
    assert MODEL_GATEWAY_URL in rendered, rendered

    token = _session_token_in(document)
    assert token, f"no x-map-session header in the compiled document: {document}"
    now = NOW_MS // 1000
    assert _accepted_at(token, now), "the token is already expired when it is minted"
    assert _accepted_at(token, now + LIFETIME_S - 1), (
        "the token expires before the lifetime it was given, so a Session would lose "
        "its tools early with a 401 naming nothing"
    )
    assert not _accepted_at(token, now + LIFETIME_S), (
        "the token outlives the lifetime it was given, so the ceiling is not a ceiling"
    )


async def test_one_pass_over_the_log_answers_both_questions_it_is_read_for() -> None:
    """The environment id and "has a Turn completed" come from one read, not two.

    Two reads could see two different logs -- a Turn completing between them is enough
    -- and the pair of answers is what the resume branch turns on. Counted rather than
    asserted structurally because the property is about how many times the port was
    asked, and the port pages, so this counts *walks* by comparing against the pages one
    walk of this log takes.
    """
    harness = _harness(cluster=RecordingCluster(), completed_a_turn=True)
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    # Three events, two per page, so one walk is two full pages plus the empty page
    # that ends it. A second walk would double this.
    assert harness.log.reads == 3, harness.log.reads


async def test_a_session_whose_log_names_no_environment_is_refused_not_guessed() -> (
    None
):
    """No creation event means no shape, and no shape means no pod.

    The alternative is inventing one, which would start a Session in a shape its tenant
    did not name and would start it successfully. Refused as `TurnUndeliverable` like
    every other reason, because the transport's caller catches one type.
    """
    harness = _harness(cluster=RecordingCluster())
    harness.log.events.clear()

    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert harness.cluster.created == []


def _session_token_in(document: dict[str, Any]) -> str:
    """The `x-map-session` header value the compiled document carries, or ''.

    Walks the parsed document rather than matching the rendered text, so a change in how
    the header table is spelled does not turn this into a test of a formatter.
    """
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "x-map-session" and isinstance(value, str):
                    found.append(value)
                else:
                    walk(value)

    walk(document)
    return found[0] if found else ""


def _accepted_at(token: str, second: int) -> bool:
    """Whether the Tool Gateway's own reader would accept this token at that second.

    Asked of `verify_session_token` -- the function the Gateway calls -- rather than of
    a string split, because what a Session needs is not that the third field holds a
    particular integer but that the process which decides says yes. The key is the one
    the token was signed with, so what varies between the two calls below is the clock
    and nothing else.
    """
    try:
        verify_session_token(token, SESSION_TOKEN_KEY, second)
    except InvalidSessionToken:
        return False
    return True


async def test_a_file_attached_after_a_turn_is_carried_by_the_next_placement() -> None:
    """The guarantee that replaced the attach route's own push, graded on its own.

    `POST /v1/sessions/{id}/resources` used to PUT the bytes into a running pod whenever
    the Session had completed a Turn, because a completed Turn meant a pod was standing
    and meant placement would not run again. ADR-041 falsified both halves at once: the
    pod goes when its Turn does, and the next Turn places another. The route now appends
    and nothing else, which is only safe if the fold that builds a placement's file set
    reaches an attach event sitting *after* the Turn events -- so that is what this
    asserts, with the event in exactly that position.

    Both files, and in log order. The file the Session was created with is not
    re-derived from anywhere else, so a fold that restarted at the attach would deliver
    the appendix into a workspace with no brief in it, and the agent would read a
    document referring to one that is not there.
    """
    created_with, attached = new_file_id(), new_file_id()
    placement = RecordingFilePlacement()
    harness = _harness(
        cluster=RecordingCluster(),
        attachments=AttachedFiles(
            HoldsTheseFiles(
                {
                    created_with: ("brief.md", b"# Brief\n"),
                    attached: ("appendix.md", b"# Appendix\n"),
                }
            ),
            placement,
        ),
        file_ids=(created_with,),
        completed_a_turn=True,
        attached_after_a_turn=(attached,),
    )
    with pytest.raises(TurnUndeliverable):
        await harness.take_a_turn()

    assert [(name, body) for _, name, body in placement.placed] == [
        ("brief.md", b"# Brief\n"),
        ("appendix.md", b"# Appendix\n"),
    ], (
        "a file attached after this Session's last Turn did not reach the pod the next "
        "Turn placed, so the attach route accepted a document nothing will ever deliver"
    )
