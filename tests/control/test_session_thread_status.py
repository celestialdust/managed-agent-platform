"""The rule that turns one thread's log facts into the state a tenant reads.

Every case here is the fold and nothing else -- no store, no route, no Session. That is
the point of `status_of` taking a record and a boolean: the rule is the part that can be
wrong in a way no integration test would name, because a wrong state still serialises.
"""

import pytest

from managed_agent.control.session.threads import (
    DEFAULT_PAGE_SIZE,
    NoSessionThreads,
    SessionThreadIndex,
    ThreadActivity,
    ThreadStatus,
    status_of,
)
from managed_agent.core.ids import Seq, SessionId, new_session_id


def _activity(
    *, archived_at_ms: int | None = None, turn_ended: bool = False
) -> ThreadActivity:
    return ThreadActivity(
        thread_id="t-1",
        parent_thread_id=None,
        was_announced=True,
        started_seq=Seq(4),
        created_at_ms=1_700_000_000_000,
        updated_at_ms=1_700_000_001_000,
        archived_at_ms=archived_at_ms,
        turn_ended=turn_ended,
    )


def test_a_thread_whose_turn_is_still_open_is_running() -> None:
    assert (
        status_of(_activity(turn_ended=False), session_open=True)
        is ThreadStatus.RUNNING
    )


def test_a_thread_whose_turn_has_closed_is_idle() -> None:
    assert status_of(_activity(turn_ended=True), session_open=True) is ThreadStatus.IDLE


def test_an_archived_thread_is_terminated_even_with_its_turn_still_open() -> None:
    """The ordering that matters, and the one a positional rule would get wrong.

    A Session can be resumed and take a further Turn after a thread was archived, so
    "archived" and "its Turn is open" are simultaneously true in the log. Reporting
    running there would tell a consumer to keep waiting for events that will not come.
    """
    archived = _activity(archived_at_ms=1_700_000_002_000, turn_ended=False)
    assert status_of(archived, session_open=True) is ThreadStatus.TERMINATED


def test_a_closed_session_terminates_a_thread_that_was_never_archived() -> None:
    assert (
        status_of(_activity(turn_ended=False), session_open=False)
        is ThreadStatus.TERMINATED
    )


def test_there_is_no_rescheduling_state_to_report() -> None:
    """A member this platform can never emit would sit unreachable in a client's switch.

    Asserted against the whole enum rather than by naming the absent value, so a future
    member added without a way to reach it fails here instead of shipping.
    """
    reachable = {"running", "idle", "terminated"}
    assert {member.value for member in ThreadStatus} == reachable


async def test_a_platform_with_no_thread_index_knows_of_no_threads() -> None:
    """Empty rather than raising: a Session predating attribution genuinely has none."""
    stand_in = NoSessionThreads()
    session_id = new_session_id()
    assert await stand_in.threads_of(session_id) == ()
    assert await stand_in.thread_at(session_id, "t-1") is None


def test_the_refusing_stand_in_satisfies_the_port_it_stands_in_for() -> None:
    assert isinstance(NoSessionThreads(), SessionThreadIndex)


def test_something_missing_a_method_does_not_satisfy_the_port() -> None:
    """The control that stops the check above from passing on anything at all."""

    class OnlyReadsOne:
        async def thread_at(
            self, session_id: SessionId, thread_id: str
        ) -> ThreadActivity | None:
            return None

    assert not isinstance(OnlyReadsOne(), SessionThreadIndex)


def test_a_thread_record_cannot_be_rewritten_after_it_is_read() -> None:
    """Frozen because these are log facts: an event already written cannot change."""
    with pytest.raises(AttributeError):
        _activity().thread_id = "t-2"  # type: ignore[misc]


def test_the_default_page_holds_the_runtime_s_whole_live_thread_ceiling() -> None:
    """25 is the runtime's limit on live threads, so a fully-live Session fits one page.

    Named rather than left implicit because the number's only justification is that
    ceiling -- a default below it would hand back a prefix of a Session's live threads
    to every caller that never pages.
    """
    assert DEFAULT_PAGE_SIZE == 25
