"""Bringing up the pod a Session's next Turn needs, whether or not it is its first.

A Session's pod is placed lazily, at the first Turn that finds none, rather than when
the Session is created. `PodRunner.ensure` returns only when every container is ready
or 180 s have gone, and a create call that blocks for three minutes is a different
product from the one this API offers; a Turn already returns when the Turn finishes, so
the wait goes where the caller is already waiting.

**The two cases differ in one compiled value and in nothing else.** A Session that has
completed a Turn must not be given a thread that starts empty: the Rollout holds a
history whose compaction checkpoints have already folded it, so a fresh thread replays
what was folded, the tenant pays for the replay, and the platform reports success while
doing it. So the placement compiles `resuming=True`, which reaches the pod as an
environment entry, and the init container that seeds the stored Rollout refuses to let
the pod start if it was told to expect one and the Gateway holds none. The refusal
lives there rather than here because that is where the answer is known -- asking the
object store from this process would put the same question in two places, free to
disagree, and the disagreement that starts a fresh thread is the silent one.

Everything else about the two placements is identical by construction: the environment
revision and the definition are read back out of `session.created` rather than resolved
afresh, so the second pod for a Session is the same pod and not a similar one.

The environment a Session was created against is read back out of its `session.created`
event and not out of its registry row, because the row does not carry one -- the id is
written into that payload and nowhere else. The same read answers whether any Turn has
completed, so one pass over the log settles both questions rather than two passes that
could see different logs.

Provenance for the recovery boundary this rests on: docs/adr/ADR-004.
"""

from __future__ import annotations

import tomllib
from typing import Protocol
from uuid import UUID

from managed_agent.control.catalog.definitions import (
    AgentReference,
    AgentVersionArchived,
    UnknownAgentVersion,
    resolve_reference,
)
from managed_agent.control.catalog.environments import (
    EnvironmentRevisions,
    UnknownEnvironment,
    resolve_environment_at,
)
from managed_agent.control.files.attachments import (
    AttachedFiles,
    FilesNotPlaceable,
)
from managed_agent.control.files.store import FileId
from managed_agent.control.pod_config.compiler import (
    CompiledConfig,
    FloorViolation,
    compile_session_config,
)
from managed_agent.control.session.placement import Placement, PodNotStarted
from managed_agent.control.session.turn_dispatch import TurnUndeliverable
from managed_agent.control.skills.registry import (
    SkillStore,
    SkillsUnresolvable,
    resolve,
)
from managed_agent.core.ids import Seq, SessionId
from managed_agent.core.ports import (
    Clock,
    DefinitionRegistry,
    EventLogRange,
    EventRecord,
)
from managed_agent.core.registration.environment import EnvironmentId
from managed_agent.core.session.session import SessionRecord
from managed_agent.core.session.session_token import (
    SESSION_TOKEN_HEADER_NAME,
    InvalidSessionToken,
    verify_session_token,
)
from managed_agent.core.vocabulary import lifecycle, resource, turn

_UNBOUNDED_END: Seq = 2**62
"""Higher than any sequence a Session will reach, so a read is bounded only by the cap.

The same value and the same reason as the one `control/api/routes/sessions.py` folds a
Session with. Copied rather than shared: two constants naming "no upper bound" is a
coincidence of arithmetic, not a rule two modules could disagree about.
"""


class SessionPods(Protocol):
    """Whatever brings a Session's pod into existence when a Turn needs one."""

    async def ensure_for(self, session_id: SessionId) -> None:
        """Return once this Session has a pod, or raise saying why it will not have one.

        Raises `TurnUndeliverable` for every reason, because the caller is a transport
        whose contract is that a caller of the port never sees anything else, and
        `control/api/routes/turns.py` catches that one type. Anything else escaping here
        reaches the tenant as a bare 500 carrying no code from the published set, with
        no `turn.failed` appended -- the trap `shim/pod_channel.py` already spells out
        for its own completion seam.

        Returning is not a promise that the pod is running. `PodRunner.ensure` answers
        once every container is ready or the wait runs out, and the caller re-reads the
        cluster's own phase afterwards rather than trusting an answer from a moment ago.
        """
        ...


class SessionRecordByIdAlone(Protocol):
    """A Session's creation facts, keyed by the Session and not by a tenant.

    Declared here rather than on `core.ports.SessionRegistry`, and taking no tenant,
    which is worth being exact about because every other read of that store takes one.
    The tenant is what a *caller-supplied* id is filtered by, so that another tenant's
    Session is absent from an answer rather than fetched and then dropped. Nothing
    supplies an id here: the only caller is the Turn transport, reached from a route
    that has already fetched this Session for the authenticated tenant, and the record
    this returns is never handed back out -- its own `tenant_id` is what every
    resolution below is then scoped by. A tenant argument here would have to be
    invented by the caller, and an invented tenant is a filter that always agrees with
    itself.

    Narrow on purpose. A store satisfies this by having the one method, so no in-memory
    double of the whole registry has to grow one, and no other caller acquires a
    tenant-free read by importing the wide port.
    """

    async def read(self, session_id: SessionId) -> SessionRecord:
        """The creation facts of that Session, whoever owns it.

        Raises `SessionNotVisible` when no such Session exists -- the same refusal the
        tenant-scoped read gives, because the absence is the same absence and a second
        exception type would be a second thing for a caller to handle identically.
        """
        ...


class FirstTurnPlacement:
    """Compiles a Session's configuration and places its pod, once, at its first Turn.

    Holds the four values a compilation needs that only a deployment knows -- both
    gateway addresses, the Session-token signing key and how long a token stays valid --
    because they are read once at the composition root and a collaborator that read them
    itself would make every Session's configuration a function of process environment at
    the moment of the Turn rather than at the moment of the deploy.

    The five ports arrive as constructor arguments rather than being reached through the
    `Platform` this is wired into: a collaborator that read fields of the object it is a
    field of would make construction order load-bearing.
    """

    def __init__(
        self,
        *,
        placement: Placement,
        sessions: SessionRecordByIdAlone,
        environments: EnvironmentRevisions,
        definitions: DefinitionRegistry,
        events: EventLogRange,
        clock: Clock,
        skills: SkillStore,
        attachments: AttachedFiles,
        session_token_key: bytes,
        session_token_lifetime_s: int,
        tool_gateway_url: str,
        model_gateway_url: str,
    ) -> None:
        self._placement = placement
        self._sessions = sessions
        self._environments = environments
        self._definitions = definitions
        self._events = events
        self._skills = skills
        self._attachments = attachments
        self._clock = clock
        self._session_token_key = session_token_key
        self._session_token_lifetime_s = session_token_lifetime_s
        self._tool_gateway_url = tool_gateway_url
        self._model_gateway_url = model_gateway_url

    async def ensure_for(self, session_id: SessionId) -> None:
        """Place this Session's pod, or refuse in the one type the transport promises.

        Ordered so that no side effect precedes a refusal: everything a compilation
        can refuse is resolved before a Secret or a pod is created.

        A resuming Session refuses LATER than that, and it is the one refusal this
        ordering cannot pull forward. Whether a stored Rollout exists is the object
        store's answer, this process does not hold that store, and asking it from here
        would put the question in two places free to disagree. So the pod is created,
        its seeding init container asks, and a Session told to resume with nothing to
        resume from never reaches a running container -- surfacing here as
        `PodNotStarted` carrying that container's own last words.

        **This method is not what counts the wait, and deliberately is not.** Its own
        extent is the right window -- it is entered only by a Turn that found no pod and
        returns only once that Turn has one or cannot have one -- but the caller,
        `shim/pod_channel.HttpPodDispatch.dispatch`, wraps this call in
        `Placement.awaiting` instead, because that is the one place that both measures
        the wait and holds the Event Log needed to tell the tenant about it. Counting in
        both would report every queued Turn twice on `GET /v1/capacity`, and an inflated
        queue depth is the one failure that makes that number worth less than no number.
        """
        try:
            await self._place(session_id)
        except PodNotStarted as refused:
            # Told apart from the six below, because the tenant's next move is the
            # opposite one. Nothing about this Session is wrong: a pod was asked for and
            # did not come up, which is transient far more often than not, and
            # resubmitting is the remedy. The six below are configuration this Session
            # names, where resubmitting changes nothing at all.
            raise TurnUndeliverable(
                f"session {session_id} was given a pod that did not start: {refused}",
                turn.TurnFailureCause.RUNTIME_DID_NOT_START,
            ) from refused
        except (
            UnknownEnvironment,
            UnknownAgentVersion,
            AgentVersionArchived,
            FloorViolation,
            SkillsUnresolvable,
            FilesNotPlaceable,
        ) as refused:
            # The refusal's own words are carried, not only its type. This text never
            # reaches a tenant -- `control/session/turn_execution.py` logs it to stderr
            # and appends the published cause -- so what it costs is nothing and what it
            # buys is the difference between "this Session's Rollout could not be
            # seeded", "that environment is not registered" and "the image will not
            # pull", which are three different people's problems.
            #
            # All six share one published cause because they share one remedy: the
            # tenant fixes what the Session names, and until they do a resubmission
            # fails identically. Splitting them further would publish six codes a
            # consumer branches on the same way, and each addition is a version event
            # under ADR-013.
            raise TurnUndeliverable(
                f"session {session_id} has no pod and could not be given one "
                f"({type(refused).__name__}): {refused}",
                turn.TurnFailureCause.SESSION_NOT_PLACEABLE,
            ) from refused

    async def _place(self, session_id: SessionId) -> None:
        record = await self._sessions.read(session_id)
        (
            environment_id,
            environment_revision,
            file_ids,
            has_completed_a_turn,
        ) = await self._creation_facts(session_id)
        # At the revision this Session was created with, not the newest one. An
        # Environment can be edited now, and resolving the newest here would make every
        # edit retroactive: a Session created against one sandbox would find itself in
        # another, mid-run, with nothing in its log saying when the reach changed. The
        # number was written into `session.created` for exactly this read.
        #
        # A retired Environment still resolves and still places. Archiving refuses a
        # *new* Session; a Session created before the retirement keeps running, or
        # archiving would be a way to kill live work rather than a way to stop new work.
        environment = await resolve_environment_at(
            self._environments,
            environment_id,
            record.tenant_id,
            environment_revision,
        )
        definition = await resolve_reference(
            self._definitions,
            record.tenant_id,
            AgentReference(record.definition_id, int(record.definition_revision)),
        )
        # Resolved before anything is compiled, for the reason `ensure_for` states: a
        # definition naming a skill that cannot be delivered refuses the placement here,
        # where the refusal is one type the transport already carries. Resolved rather
        # than defaulted -- passing no skills to a Session whose definition attaches one
        # is the failure this whole path exists to prevent, and it is silent.
        skill_files = await resolve(
            self._skills, record.tenant_id, definition.definition
        )
        now_epoch_s = self._clock.now_epoch_ms() // 1000
        compiled = compile_session_config(
            record,
            tool_gateway_url=self._tool_gateway_url,
            model_gateway_url=self._model_gateway_url,
            environment=environment,
            definition=definition.definition,
            session_token_key=self._session_token_key,
            session_token_expiry_epoch_s=now_epoch_s + self._session_token_lifetime_s,
            skill_files=skill_files,
            # The one thing a resuming placement compiles differently, and it changes
            # neither document: it reaches the pod as an environment entry telling the
            # init container that seeds a Rollout whether to expect one. Everything
            # else about a Session's shape is identical across its placements by
            # construction -- the environment revision and the definition are read back
            # out of the creation event, not resolved afresh -- which is what makes a
            # second pod for one Session the same pod rather than a similar one.
            resuming=has_completed_a_turn,
        )
        _refuse_a_token_born_expired(compiled, self._session_token_key, now_epoch_s)
        await self._placement.place(compiled)
        # After the pod is ready and before the Turn is dispatched, which is the only
        # window where both halves are true: the shim is answering, and the agent has
        # not been given an instruction yet. `place` returns when every container is
        # ready, so this is not a race against the runtime coming up -- and a Session
        # whose file did not arrive raises here, before the Turn, rather than running
        # with the document missing.
        await self._attachments.place_for(session_id, record.tenant_id, file_ids)

    async def _creation_facts(
        self, session_id: SessionId
    ) -> tuple[EnvironmentId, int, tuple[FileId, ...], bool]:
        """The Session's pinned shape, its attached files, and whether a Turn ran.

        One pass over the log for all three. The environment id and the file ids live
        only in the `session.created` payload -- `SessionRecord` has neither field and
        `session` has neither column -- and "has a completed Turn" is the ADR-004
        question that decides whether a fresh pod may be placed at all. Two reads could
        see two different logs, and the answers together are what the branch above turns
        on.

        The file ids are in the event rather than in a column for the reason the
        environment id is: they are a creation fact, the log is append-only, and a
        column would be a second place they could be changed after creation. A Session
        whose attachment set moved mid-run is one whose earlier and later Turns saw
        different worlds with nothing in the record saying when.

        Reads the whole log, which is what `GET /v1/sessions/{id}` already pays and is
        the part of that route its own docstring names as not scaling. It happens only
        when a Turn finds no pod, which is once per Session in the ordinary case.

        Raises `UnknownEnvironment` when the log carries no creation event or its
        payload holds no id that parses as one. Both mean the same thing to a caller --
        there is no shape to start this Session's pod from -- and inventing one here
        would compile a Session into whatever shape happened to be reachable.
        """
        created: EnvironmentId | None = None
        revision = _FIRST_REVISION
        attached: tuple[FileId, ...] = ()
        completed = False
        for event in await self._whole_log(session_id):
            if event.type == lifecycle.SESSION_CREATED and created is None:
                created = _environment_id_in(event.payload)
                revision = _environment_revision_in(event.payload)
                attached = _file_ids_in(event.payload)
            elif event.type == resource.SESSION_FILE_ATTACHED:
                # Appended by `POST /v1/sessions/{id}/resources`, which accepts an
                # attach only while the Session would still take a Turn. Read here
                # because a Session attached to before its first Turn has no pod yet:
                # the route appended and pushed nothing, and this is the placement that
                # was always going to deliver its files.
                #
                # Deliberately NOT guarded on having met the creation event first. A
                # guard was written here and removed rather than left as reassurance:
                # the creation branch above ASSIGNS `attached` rather than extending
                # it, so an attach read ahead of creation is discarded by the creation
                # event itself. Measured -- with the guard removed every test passed,
                # including the one written to catch its removal, which is the shape of
                # a check that reads as protection and protects nothing.
                attached = (*attached, _attached_file_id(event.payload))
            elif event.type == turn.TURN_COMPLETED:
                completed = True
        if created is None:
            raise UnknownEnvironment(
                f"session {session_id} has no readable session.created environment id, "
                "so there is no registered shape to start its pod from"
            )
        return created, revision, attached, completed

    async def _whole_log(self, session_id: SessionId) -> list[EventRecord]:
        """Every event of one Session, across as many reads as it takes.

        The port caps a read and promises nothing about how much of a range comes back,
        so a short page means "read again" and never "the range is empty". Reads from
        just past the highest sequence seen so far and stops on an empty page, which
        terminates for any page size because each non-empty page raises the cursor above
        its own highest sequence and a log is finite.

        A second implementation of the same walk lives in
        `control/api/routes/sessions.py`. Left as two: what they share is arithmetic the
        port's own contract dictates, not a rule this platform could change its mind
        about, and the fold there returns a projected state while this returns records.
        """
        events: list[EventRecord] = []
        cursor = 0
        while True:
            page = await self._events.read(session_id, Seq(cursor + 1), _UNBOUNDED_END)
            if not page:
                return events
            events.extend(page)
            cursor = max(event.seq for event in page)


def _refuse_a_token_born_expired(
    compiled: CompiledConfig, key: bytes, now_epoch_s: int
) -> None:
    """Refuse a compiled document whose Session token is already dead.

    No floor reads the expiry. `check_floors` grades the token's *shape* and the two
    identifiers inside it, and both hold for an expiry of `0` -- so a lifetime of zero,
    a clock read in the wrong unit, or an arithmetic slip compiles cleanly, places a
    pod, and that pod answers 401 on every tool call for its whole life. The Gateway
    consults no relational store and no clock but this token's own, and its refusal is
    a fixed 401 indistinguishable from every other failure, so nothing downstream can
    report the cause.

    Checked by handing the token to `verify_session_token` -- the function the Tool
    Gateway itself calls -- rather than by comparing the third field to an integer. What
    matters is not that a number looks right but that the process which decides says
    yes, and a shape check has already been shown not to be that: `_TOKEN_SHAPE` accepts
    `0`.

    **What this cannot see, said plainly so the next reader does not stop looking.** It
    compares the token against *this* process's clock, which is the same clock that
    minted it, so a control plane whose clock disagrees with the Gateway's passes here
    and 401s in the cluster. Clock agreement between two pods is the node's business and
    nothing in this repository asserts it. What this closes is every way the expiry goes
    wrong inside one process.
    """
    header = _session_token_in(compiled.config_toml)
    if header is None:
        raise FloorViolation(
            "the compiled document carries no session token to check an expiry on"
        )
    try:
        verify_session_token(header, key, now_epoch_s)
    except InvalidSessionToken as dead:
        raise FloorViolation(
            "the session token this document was compiled with is already expired or "
            "unreadable at the moment it was minted, so the pod would answer 401 on "
            "every tool call for its whole life with nothing naming the cause"
        ) from dead


def _session_token_in(config_toml: str) -> str | None:
    """The `x-map-session` header value out of a compiled `config.toml`, or None.

    Read back out of the rendered document rather than off the value that was passed to
    the renderer, because what a pod presents is the document -- a token accepted at the
    seam and dropped before rendering is exactly the failure a check on the argument
    cannot see.
    """
    document = tomllib.loads(config_toml)
    servers = document.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    for server in servers.values():
        if not isinstance(server, dict):
            continue
        headers = server.get("http_headers")
        if isinstance(headers, dict):
            value = headers.get(SESSION_TOKEN_HEADER_NAME)
            if isinstance(value, str):
                return value
    return None


def _file_ids_in(payload: dict[str, object]) -> tuple[FileId, ...]:
    """The file ids a `session.created` payload names, in the order it names them.

    **An unreadable entry refuses the whole list rather than being skipped**, which is
    the opposite of returning what parsed. A Session that attached three files and got
    two would start, and the agent would report one document missing -- which reads to a
    tenant as the platform losing an upload. A list this cannot parse is a payload this
    code did not write, so the honest answer is none of it, and the placement then fails
    on the id that does not resolve.

    Absent is not the same as unreadable and is not an error: every Session created
    before this field existed has no key here, and every Session that attaches nothing
    has an empty list. Both mean no files.

    Order is preserved because the caller places them in it, and a stable order is what
    makes two placements of one Session produce the same workspace.
    """
    raw = payload.get("file_ids")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        return ()
    parsed: list[FileId] = []
    for entry in raw:
        if not isinstance(entry, str):
            return ()
        try:
            parsed.append(FileId(UUID(entry)))
        except ValueError:
            return ()
    return tuple(parsed)


def _environment_id_in(payload: dict[str, object]) -> EnvironmentId | None:
    """The environment id a `session.created` payload names, or None if it names none.

    Parsed rather than cast. The payload comes back out of a JSONB column as whatever
    was written, and a value that is not a uuid string would otherwise reach
    `resolve_environment` and be compared against a column of uuids -- which is a driver
    error at the query rather than a refusal naming the Session.
    """
    raw = payload.get("environment_id")
    if not isinstance(raw, str):
        return None
    try:
        return EnvironmentId(UUID(raw))
    except ValueError:
        return None


_FIRST_REVISION = 1
"""The revision every Environment had before an edit could append a second one."""


def _attached_file_id(payload: dict[str, object]) -> FileId:
    """The file id out of one `session.file_attached` payload.

    Parsed through the payload model the appender used rather than picked out by hand,
    so the two cannot drift: the route builds that model and this reads it, and a field
    renamed on one side fails on the other instead of silently placing no file.

    Raises rather than skipping an unreadable payload. Every other reason this fold
    gives up refuses the placement, and for the same reason -- a Session whose attached
    file did not arrive will run, report that it cannot find the document, and reach a
    tenant as the platform having lost their upload.
    """
    attached = resource.SessionFileAttached.model_validate(payload)
    return FileId(UUID(attached.file_id))


def _environment_revision_in(payload: dict[str, object]) -> int:
    """The revision a `session.created` payload pinned, or the first one.

    Defaults to 1 rather than refusing, and the default is a fact rather than a guess: a
    Session created before Environments were revisioned has no such key, and migration
    0022 gave every row that existed at that moment `revision = 1`. So the shape those
    Sessions named is revision 1 exactly, and reading it is not a fallback -- it is the
    right answer for the whole of the log written before the column existed.

    A present-but-unusable value is also the default, for the same reason
    `_environment_id_in` parses rather than casts: the payload comes out of a JSONB
    column as whatever was written, and a float or a string reaching a query against an
    integer column is a driver error rather than a refusal naming the Session. `bool` is
    excluded explicitly because it is an `int` in Python, and `True` would otherwise pin
    revision 1 while looking like a deliberate number.
    """
    raw = payload.get("environment_revision")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < _FIRST_REVISION:
        return _FIRST_REVISION
    return raw
