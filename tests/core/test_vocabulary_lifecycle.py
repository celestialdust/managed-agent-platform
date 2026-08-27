"""The projection's transition table and the published vocabulary do not drift apart.

Tier 1 (local, no infrastructure). Two directions, and both matter. A table entry
whose event type nothing declares is a transition that can never fire — nothing
may emit an unpublished type, so the row is dead and the state it leads to is
unreachable. A declared lifecycle type that no table entry reads is an event a
reader can meet that moves the state nowhere. The two files are edited by
different slices, which is exactly why the agreement between them needs a test
rather than a convention.

**Two of the four types are appended by nothing, and they are still declared here on
purpose.** `session.suspended` and `session.resumed` lost their producers when a pod
became a thing leased for one Turn, so no new row of either will ever be written — but
the rows a tenant's Event Log already holds have to keep folding and keep replaying, and
the read surfaces drop an undeclared type in silence rather than refusing it. Retiring
the declaration would therefore punch holes in a stored history instead of ending a
story, which is why the case below asserts they are still published. What did end is the
subscription, and `test_webhook_eligibility.py` grades that half.

The stop reasons below are graded the same way and for the same reason. Each member
encodes a decision — which ending event carries it — and that decision is made in a
`Literal` on a payload model, so a member added to the enum without being admitted by
any model would be a reason nothing can ever append. One assertion per member,
parametrized over the enum, because a single assertion over the whole set is
satisfied by whichever member happens to be checked first.
"""

from typing import get_args

import pytest

from managed_agent.core import vocabulary
from managed_agent.core.session.projection import _TRANSITIONS
from managed_agent.core.vocabulary import lifecycle
from managed_agent.core.vocabulary.lifecycle import StopReason


def _session_transitions() -> set[str]:
    return {name for name in _TRANSITIONS if name.startswith("session.")}


def test_every_session_transition_the_projection_reads_is_published() -> None:
    """A row here with no declared type is a state the fold can never arrive at."""
    undeclared = {
        name for name in _session_transitions() if not vocabulary.is_published(name)
    }
    assert undeclared == set(), (
        f"transition table reads unpublished types: {sorted(undeclared)}"
    )


def test_every_lifecycle_type_is_read_by_the_transition_table() -> None:
    """A published lifecycle event with no row at all is a transition nobody wired.

    Membership and not the value behind it, because a row may deliberately carry `None`
    -- "read, and moves nothing" -- which is what `session.resumed` carries: every row
    of it a stored log holds was appended inside an open Turn, so any state it named
    would overrule the Turn events around it. What this refuses is the type nobody
    considered, and the two cases are told apart by whether the key is there.
    """
    declared = {
        value
        for name, value in vars(lifecycle).items()
        if name.isupper() and name != "FAMILY" and isinstance(value, str)
    }
    assert declared, "the lifecycle family declares no event types"
    assert declared <= set(_TRANSITIONS), (
        f"unread lifecycle types: {sorted(declared - set(_TRANSITIONS))}"
    )


def test_the_lifecycle_family_is_registered_under_its_own_name() -> None:
    assert vocabulary.PUBLISHED[lifecycle.SESSION_CREATED] == lifecycle.FAMILY


@pytest.mark.parametrize(
    "retired", [lifecycle.SESSION_SUSPENDED, lifecycle.SESSION_RESUMED]
)
def test_a_type_whose_producer_is_gone_is_still_published(retired: str) -> None:
    """The declaration outlives the producer, because a stored row still has to arrive.

    Both surfaces a tenant reads history through -- the SSE tail and the replay it
    serves a resume from -- emit a row only when its type is published, and a row whose
    type is not is dropped with no error, no gap marker and no line anywhere saying a
    sequence went missing. So undeclaring a type does not retire it: it deletes every
    row of it that a tenant already has, from a history that is supposed to be
    append-only, at whatever moment the deploy lands.

    The producers are gone and this is what is left of them. Asserted per type rather
    than over the pair, so removing one declaration fails naming which one.
    """
    assert vocabulary.is_published(retired)


def _reasons_of(model: type[lifecycle.SessionStopped]) -> set[StopReason]:
    """The stop reasons one payload model's `Literal` admits.

    Read off the annotation rather than restated here, so this file holds no second copy
    of the partition. A test that listed the members itself would agree with itself
    after somebody widened a `Literal` and forgot the other one.
    """
    return set(get_args(model.model_fields["stop_reason"].annotation))


@pytest.mark.parametrize("reason", list(StopReason), ids=lambda r: r.value)
def test_every_stop_reason_is_carried_by_exactly_one_ending_event(
    reason: StopReason,
) -> None:
    """A reason no ending event admits is a reason nothing can append.

    A member sitting in the enum that no payload model will accept is a value a consumer
    can read in the docs, branch on, and never see. The opposite direction -- two events
    admitting one reason, so that it could mean "parked" or "finished" depending on
    which carried it -- is what the count below is written for rather than a truthiness
    check, and it is unreachable while one ending event is left. It stops being
    unreachable the day a second one lands, which is the day it is needed.
    """
    carried_by = [
        event
        for event, model in ((lifecycle.SESSION_STOPPED, lifecycle.SessionStopped),)
        if reason in _reasons_of(model)
    ]
    assert len(carried_by) == 1, (
        f"{reason.value} is carried by {carried_by or 'no ending event'}; every stop "
        "reason belongs to exactly one event that ends a Session's work"
    )


def test_the_ending_payload_refuses_the_reason_a_reclaimed_pod_used_to_carry() -> None:
    """The narrowing is enforced at run time and not only by the type checker.

    `mypy --strict` already refuses the wrong pairing at every call site in `src/`, and
    that is the guard that matters. This one exists because the payload is also parsed
    back out of a JSONB column by anything reading a log, where no annotation is present
    -- so the model has to refuse on its own.

    Graded against `idle_timeout` rather than an invented spelling, because that is not
    a hypothetical value: it is what every stored `session.suspended` payload carries,
    it is the reason a pod reclaimed on a clock used to be given, and it is the first
    thing somebody re-widening this model would reach for. A `session.stopped` carrying
    it would say a Session ended when a pod merely went.
    """
    with pytest.raises(ValueError):
        lifecycle.SessionStopped(stop_reason="idle_timeout")  # type: ignore[arg-type]
