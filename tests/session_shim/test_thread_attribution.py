"""Which agent said a thing, on the events a tenant reads.

ADR-007 accepted that subagent activity reaches the tenant "tagged with the thread that
produced it and that thread's parent". ADR-007 (MAP-A10), signed, requires that no
Agent Runtime thread identifier reaches the caller and that every identifier the caller
sees is one the platform issued. Publishing the runtime's string satisfies the first and
breaks the second; publishing nothing satisfies the second and breaks the first.

So the platform issues its own, derived from the runtime's by `uuid5` over
`session:runtime` in a fixed namespace, and this file grades the properties that make
attribution work: distinct threads get distinct ids, one thread keeps one id, two
Sessions never share one, and the runtime's own value appears nowhere.

The harness is imported from `test_turn_runner.py` rather than rebuilt. A second set of
frame builders is a second opinion about what the runtime sends, and the point of the
nested-object case below is that getting that shape wrong is invisible.
"""

from typing import Any, Final
from uuid import UUID, uuid5

from test_turn_runner import (
    _CHILD,
    _THREAD,
    RecordingLog,
    _completed,
    _delta,
    _run,
    _started,
)

from managed_agent.core.ids import SessionId
from managed_agent.core.vocabulary import PUBLISHED, thread, turn
from managed_agent.session_shim.turn_runner import _THREAD_NAMESPACE

_UNRELATED: Final = "thread-nobody-asked-about"


def _issued_for(session_id: SessionId, runtime_id: str) -> str:
    """The platform id for one runtime thread, re-derived from the contract.

    `uuid5` over the published namespace and the documented `session:runtime` form,
    written out here rather than obtained by calling the module's own function. Calling
    it would make every assertion below hold for an implementation that returned the
    runtime string unchanged, which is the one failure ADR-007 (MAP-A10) names.
    """
    return str(uuid5(_THREAD_NAMESPACE, f"{session_id}:{runtime_id}"))


def _thread_started(
    thread_id: str | None = _CHILD,
    parent: str | None = _THREAD,
    *,
    nested: bool = True,
) -> dict[str, Any]:
    """A `thread/started` notification, in either of the two shapes that exist.

    `nested=True` is the shape the runtime's own documentation gives -- `{"method":
    "thread/started", "params": {"thread": {"id": ...}}}` -- and it is the default
    because it is the one a real frame uses. `nested=False` writes the identifiers
    beside the params instead, which is how requests in this protocol address a thread.

    The beside-it `threadId` names the PARENT in the nested shape, deliberately. That is
    the arrangement a reader would expect on a frame announcing a child, and it is what
    lets the nested-versus-flat case fail: with both names holding one value, reading
    the wrong one would be invisible.

    `None` for either identifier omits the key rather than writing an empty string. An
    absent field and an empty one reach the code under test as different things, and
    only one of them is what a runtime sends.
    """
    params: dict[str, Any] = {"threadId": _THREAD}
    if nested:
        inner: dict[str, Any] = {}
        if thread_id is not None:
            inner["id"] = thread_id
        if parent is not None:
            inner["parentThreadId"] = parent
        params["thread"] = inner
    else:
        if thread_id is not None:
            params["threadId"] = thread_id
        if parent is not None:
            params["parentThreadId"] = parent
    return {"method": "thread/started", "params": params}


def _payloads(log: RecordingLog, type_: str) -> list[dict[str, object]]:
    return [one.payload for one in log.written if one.type == type_]


async def test_a_subagent_beginning_is_published_at_all() -> None:
    """A subagent beginning becomes an event at all. Before this it was dropped.

    `shim/turn_runner.py`'s map had four entries and its docstring states the rule: a
    notification with no entry is dropped. `thread/started` had none, so nothing marked
    the moment a subagent began, and a thread listing assembled from deltas alone could
    name a thread without saying when it started or whose child it was.

    The whole type sequence is asserted rather than membership, so an implementation
    that published the new event by dropping an old one fails here.
    """
    log, _, _, _, _ = await _run(
        [_started(), _thread_started(), _delta("delegating"), _completed()]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [
        turn.TURN_STARTED,
        thread.THREAD_STARTED,
        turn.TURN_MESSAGE_DELTA,
        turn.TURN_COMPLETED,
    ]


async def test_thread_started_is_a_type_this_platform_publishes() -> None:
    """The type is in the sealed vocabulary, under its own family.

    A type the shim appends and the vocabulary does not declare is a type a tenant can
    receive and no consumer can validate. The family is asserted too -- `thread` and not
    `turn` -- because several threads live inside one Turn, and filing it under `turn`
    would suggest an ordering between threads that does not exist.
    """
    assert thread.THREAD_STARTED in PUBLISHED
    assert PUBLISHED[thread.THREAD_STARTED] == thread.FAMILY


async def test_the_subagent_event_names_the_child_and_its_parent() -> None:
    """The event names the child and the parent, and carries nothing else.

    Exact equality rather than two key lookups, so a payload that also carried the
    runtime's frame wholesale fails here instead of passing two assertions about the two
    keys somebody thought to check. `turn_id` is read back out of the payload rather
    than transcribed, because it is the platform's own value and this case is not about
    it.
    """
    log, _, session_id, _, _ = await _run([_started(), _thread_started(), _completed()])

    assert isinstance(log, RecordingLog)
    started = _payloads(log, thread.THREAD_STARTED)
    assert started == [
        {
            "turn_id": started[0]["turn_id"],
            "thread_id": _issued_for(session_id, _CHILD),
            "parent_thread_id": _issued_for(session_id, _THREAD),
        }
    ]


async def test_the_child_id_is_read_from_the_nested_object_not_from_beside_it() -> None:
    """The child's id comes from the nested object, not from beside it.

    This is the trap, and it is the one that module's failure reader documents for
    `turn`: the runtime tags these externally, so a read of `params.threadId` on a
    `thread/started` frame finds the parent or finds nothing. Either way every published
    `thread.started` names the wrong thread, and no test putting one value in both
    places could tell.

    The second assertion gives the first its teeth. Asserting only that the id equals
    the child's would also pass if the frame had put the child's id in both fields;
    asserting it differs from the parent's says the read went to the right one.
    """
    log, _, session_id, _, _ = await _run([_started(), _thread_started(), _completed()])

    assert isinstance(log, RecordingLog)
    started = _payloads(log, thread.THREAD_STARTED)[0]
    assert started["thread_id"] == _issued_for(session_id, _CHILD)
    assert started["thread_id"] != _issued_for(session_id, _THREAD)


async def test_the_top_level_shape_is_accepted_too() -> None:
    """A frame carrying the identifiers beside the params still publishes.

    Both shapes exist in this protocol -- notifications carry a `thread` object,
    requests address a thread by a top-level `threadId` -- and this platform reads the
    notification shape first and falls back. Without the fallback a runtime version
    using the other arrangement would publish nothing at all, silently, which is the
    failure this whole file exists to remove.
    """
    log, _, session_id, _, _ = await _run(
        [_started(), _thread_started(nested=False), _completed()]
    )

    assert isinstance(log, RecordingLog)
    assert _payloads(log, thread.THREAD_STARTED)[0]["thread_id"] == _issued_for(
        session_id, _CHILD
    )


async def test_a_beginning_that_names_no_thread_is_not_published() -> None:
    """A beginning naming no thread publishes nothing, rather than an empty id.

    An event saying "a thread started, id unknown" is a node in the tree that can never
    be matched to the events it exists to group, and an empty id compares equal to
    another empty id -- so two such events would merge into a thread that never existed.
    The map's documented way for a mapped method to carry no event is an extractor
    returning None, and this is that path.

    The whole sequence is asserted, so an implementation that swallowed the surrounding
    Turn's events along with this one fails here too.
    """
    log, _, _, _, _ = await _run(
        [_started(), _thread_started(thread_id=None, parent=None), _completed()]
    )

    assert isinstance(log, RecordingLog)
    assert log.types() == [turn.TURN_STARTED, turn.TURN_COMPLETED]


async def test_a_thread_with_no_known_parent_is_still_published() -> None:
    """A thread whose parent the frame withheld is still published, parent null.

    The opposite call from the case above, and the difference is which fact is missing.
    Without an id there is nothing to attribute anything to; without a parent there is
    still a thread that began, and dropping the event would lose the only record of it
    in order to avoid recording a null.
    """
    log, _, session_id, _, _ = await _run(
        [_started(), _thread_started(parent=None), _completed()]
    )

    assert isinstance(log, RecordingLog)
    started = _payloads(log, thread.THREAD_STARTED)[0]
    assert started["thread_id"] == _issued_for(session_id, _CHILD)
    assert started["parent_thread_id"] is None


async def test_every_event_of_the_turn_says_which_thread_produced_it() -> None:
    """Every mapped event of the Turn says which thread produced it.

    Attribution on the beginning alone would leave the text unattributed, which is the
    shipped behaviour this replaces: subagent output arrived under the same method the
    root agent uses, so a tenant saw one undifferentiated voice. The `assert mapped`
    line is the vacuity control -- `all()` over an empty list is true, and this case's
    whole subject is a list of events.
    """
    log, _, session_id, _, _ = await _run(
        [_started(), _delta("one"), _delta("two"), _completed()]
    )

    assert isinstance(log, RecordingLog)
    mapped = [
        one
        for one in log.written
        if one.type in (turn.TURN_STARTED, turn.TURN_MESSAGE_DELTA)
    ]
    assert mapped, "no mapped event was appended, so this asserts nothing"
    assert all(
        one.payload.get("thread_id") == _issued_for(session_id, _THREAD)
        for one in mapped
    ), [one.payload for one in mapped]


async def test_one_runtime_thread_keeps_one_platform_identity_across_events() -> None:
    """One runtime thread is one platform thread across every event it produced.

    Grouping is the entire mechanism: a thread is addressed by collecting the events
    attributed to it, so an id minted per event would produce as many threads as there
    are events -- and every other assertion in this file would still pass.
    """
    log, _, _, _, _ = await _run(
        [_started(), _delta("one"), _delta("two"), _completed()]
    )

    assert isinstance(log, RecordingLog)
    seen = {
        one.payload["thread_id"] for one in log.written if "thread_id" in one.payload
    }
    assert len(seen) == 1, seen


async def test_two_sessions_never_share_a_platform_thread_identity() -> None:
    """The same runtime thread under two Sessions is two platform threads.

    The runtime mints thread ids per process and this platform runs one process per
    Session, so the same string genuinely does arrive under different Sessions.
    Unscoped, two tenants' threads would merge under one identifier. `_run` gives each
    call a fresh Session, which is what makes the two ids comparable at all.
    """
    first, _, one_session, _, _ = await _run([_started(), _completed()])
    second, _, other_session, _, _ = await _run([_started(), _completed()])

    assert isinstance(first, RecordingLog) and isinstance(second, RecordingLog)
    assert one_session != other_session
    assert (
        first.written[0].payload["thread_id"] != second.written[0].payload["thread_id"]
    )


async def test_two_runtime_threads_do_not_collapse_into_one() -> None:
    """Two runtime threads do not collapse into one platform thread.

    The child and its parent differ on a real event, and two arbitrary runtime ids
    differ under one Session. A hash mapping everything to one value would satisfy the
    stability case above and destroy attribution completely, so stability is half the
    property and this is the other half.
    """
    log, _, session_id, _, _ = await _run([_started(), _thread_started(), _completed()])

    assert isinstance(log, RecordingLog)
    started = _payloads(log, thread.THREAD_STARTED)[0]
    assert started["thread_id"] != started["parent_thread_id"]
    assert _issued_for(session_id, _CHILD) != _issued_for(session_id, _UNRELATED)


async def test_no_runtime_thread_identifier_reaches_the_tenant() -> None:
    """No Agent Runtime thread identifier appears in anything published.

    ADR-007 (MAP-A10), restated for the field this slice adds. The signed scenario
    names the runtime thread identifier specifically and requires that every identifier
    the caller sees is one the platform issued. `test_turn_runner.py` already asserts
    this over the frames it scripts; this adds the frame carrying a SECOND runtime
    thread id, which is the one an implementation would be most tempted to pass through.

    The two `in repr(...)` lines are the vacuity control and they are not decoration:
    they say the frames under test actually contain the strings being searched for.
    Without them a builder that stopped putting a runtime id in the frame would turn
    this case green while proving nothing.
    """
    log, _, _, _, _ = await _run(
        [_started(), _thread_started(), _delta("delegating"), _completed()]
    )

    assert isinstance(log, RecordingLog)
    written = repr(log.written)
    assert _THREAD in repr(_thread_started()), "the frames carry no runtime id to leak"
    assert _CHILD in repr(_thread_started()), "the child id is not in the frame either"
    assert _THREAD not in written
    assert _CHILD not in written


async def test_what_is_published_parses_as_the_identifier_it_claims_to_be() -> None:
    """What is published is a uuid5, which is what platform-issued means here.

    Version 5 specifically, because that is the derivation the design rests on: a uuid4
    would be a fresh random value per call and fail stability, and a v3 would mean the
    namespace hashing changed underneath. This is the cheapest statement that the value
    is ours rather than something copied through.
    """
    log, _, _, _, _ = await _run([_started(), _thread_started(), _completed()])

    assert isinstance(log, RecordingLog)
    started = _payloads(log, thread.THREAD_STARTED)[0]
    for key in ("thread_id", "parent_thread_id"):
        assert UUID(str(started[key])).version == 5


# ------------------------------------------------------------------------------------
# The Turn's own ending, which the central attribution does not reach
# ------------------------------------------------------------------------------------
#
# `turn.completed` and `turn.failed` are built by the runner's own code rather than
# translated from a mapped frame, so the attribution every mapped event goes through
# never runs on them. Left alone they are the only events of a Turn that do not say
# which agent produced them -- and they are the two a tenant is most likely to read.


async def test_the_turns_completion_says_which_thread_it_ran_on() -> None:
    """The Turn's answer names the root thread, in the platform's own identifier.

    Exact equality rather than a key lookup, for the same reason the subagent case above
    uses it: the guard is against a payload growing a field nobody decided to publish,
    and a lookup cannot see one. `turn_id` is transcribed here because this event is the
    one place both identifiers meet, and reading it back off the payload would leave the
    case asserting that a value equals itself.
    """
    log, _, session_id, turn_id, _ = await _run(
        [_started(), _delta("the "), _delta("answer"), _completed()]
    )

    assert isinstance(log, RecordingLog)
    assert _payloads(log, turn.TURN_COMPLETED) == [
        {
            "turn_id": str(turn_id),
            "thread_id": _issued_for(session_id, _THREAD),
            "text": "the answer",
        }
    ]


async def test_a_turn_the_runtime_reported_as_failed_says_which_thread_it_ran_on() -> (
    None
):
    """A failure names its thread too, and this is the harder half to remember.

    Three separate appends end a Turn -- an answer, a reported failure, and a stream
    that stopped -- and each is written out on its own branch. Attributing the happy
    one and forgetting the others is the shape the mistake takes, so each branch gets a
    case rather than one case standing for all three.
    """
    log, _, session_id, turn_id, _ = await _run(
        [_started(), _delta("half an "), _completed(status="failed")]
    )

    assert isinstance(log, RecordingLog)
    assert _payloads(log, turn.TURN_FAILED) == [
        {
            "turn_id": str(turn_id),
            "thread_id": _issued_for(session_id, _THREAD),
            "cause": "runtime_reported_failure",
        }
    ]


async def test_a_turn_whose_runtime_went_quiet_says_which_thread_it_ran_on() -> None:
    """The third branch: no completion arrived at all, and the Turn is still a thread's.

    This one is reached by the stream ending rather than by a frame, so there is no
    notification to read a thread off even in principle -- the id comes from the thread
    the Turn was started on, which is the only thing known about a runtime that has gone
    quiet. A reader diagnosing a lost Turn is asking which agent was mid-sentence, and
    this is the event that answers.
    """
    log, _, session_id, turn_id, _ = await _run([_started(), _delta("half an ")])

    assert isinstance(log, RecordingLog)
    assert _payloads(log, turn.TURN_FAILED) == [
        {
            "turn_id": str(turn_id),
            "thread_id": _issued_for(session_id, _THREAD),
            "cause": "runtime_lost",
        }
    ]


async def test_the_ending_shares_the_threads_identity_with_the_rest_of_the_turn() -> (
    None
):
    """One thread, one id -- including on the event the runner writes for itself.

    Equality with the mapped events is the property, not merely being present. The
    ending is built on a different code path from every other event, so it is exactly
    where a second way of deriving the id would creep in; two derivations that disagree
    would split one thread into two in any listing assembled by grouping on this field,
    and every per-event assertion in this file would still pass.

    The last two lines are the MAP-A10 half. `_THREAD` is the runtime's own string, and
    the `in repr(_completed())` line is the vacuity control -- without it, a frame
    builder that stopped carrying a runtime thread id would turn this green while
    proving nothing.
    """
    log, _, session_id, _, _ = await _run(
        [_started(), _thread_started(), _delta("the answer"), _completed()]
    )

    assert isinstance(log, RecordingLog)
    ending = _payloads(log, turn.TURN_COMPLETED)[0]
    beginning = _payloads(log, turn.TURN_STARTED)[0]
    assert ending["thread_id"] == beginning["thread_id"]
    assert ending["thread_id"] == _issued_for(session_id, _THREAD)
    assert _THREAD in repr(_completed()), "the frame carries no runtime id to leak"
    assert _THREAD not in repr(log.written)
