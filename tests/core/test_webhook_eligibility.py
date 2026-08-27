"""Which event types a tenant may point a webhook at.

Tier 1 (local, no infrastructure).

A webhook subscribes to event types rather than to Session states, so the registry has
to answer which types are eligible -- and the answer has to be narrower than "every
published type". The tail that finds deliverable events scans a time window and filters
by type, so a set that admitted `turn.message_delta` would put a per-token event through
a delivery ledger.

Eligibility is therefore a property recorded at the declaration, the way the family is,
and read back off the registry. The alternative -- a tuple listed in the dispatcher --
is a second place that has to be edited when a family gains a type, and the one that
gets forgotten. That failure already happened here once in the other direction: a
transition table naming two event types nothing could publish (`docs/lessons.md`).

Opt-in rather than opt-out, because the cost of the two mistakes is not symmetric.
Forgetting to mark a type eligible means a tenant cannot subscribe to it and says so;
forgetting to mark one ineligible means a per-token event reaches a delivery ledger and
a tenant's endpoint.
"""

from managed_agent.core import vocabulary
from managed_agent.core.vocabulary import lifecycle, turn


def test_eligibility_is_opt_in_so_a_new_type_cannot_reach_a_webhook_by_default() -> (
    None
):
    """The fail-safe direction, asserted against a real unmarked type.

    `turn.message_delta` is the type this rule exists for: it is published, a tenant can
    read it off the stream, and it arrives once per token. If eligibility ever defaults
    to true, this is the case that says so before a tenant's endpoint finds out.
    """
    assert vocabulary.is_published(turn.TURN_MESSAGE_DELTA)
    assert turn.TURN_MESSAGE_DELTA not in vocabulary.WEBHOOK_ELIGIBLE


def test_the_lifecycle_types_something_still_appends_are_eligible() -> None:
    """Naming them here rather than comparing against the family, so the equality is
    not restated as itself: a type dropped from the family would leave both sides equal
    and this assertion failing, which is the direction that matters.
    """
    assert {
        lifecycle.SESSION_CREATED,
        lifecycle.SESSION_STOPPED,
    } <= vocabulary.WEBHOOK_ELIGIBLE


def test_a_type_no_producer_writes_is_not_something_a_tenant_can_subscribe_to() -> None:
    """Eligibility ended with the producer; the declaration did not.

    `session.suspended` and `session.resumed` are still published, because the rows a
    tenant already has must keep replaying off a surface that drops an undeclared type
    in silence. Nothing appends either one any more, and the two facts have to be able
    to disagree -- which is the whole reason eligibility is recorded separately from
    publication rather than derived from it.

    Left eligible, a registration for either would be a callback a tenant configured, an
    endpoint they kept a certificate alive for, and a delivery that is never coming. The
    register route reads this set, so the refusal now happens at the registration
    instead of in a silence a tenant has to notice on their own.
    """
    assert lifecycle.SESSION_SUSPENDED not in vocabulary.WEBHOOK_ELIGIBLE
    assert lifecycle.SESSION_RESUMED not in vocabulary.WEBHOOK_ELIGIBLE
    assert vocabulary.is_published(lifecycle.SESSION_SUSPENDED)
    assert vocabulary.is_published(lifecycle.SESSION_RESUMED)


def test_an_eligible_type_is_always_a_published_one() -> None:
    """Eligibility cannot widen the published set.

    `declare` is the only way into either registry, so this holds by construction today.
    It is asserted because the construction is the thing under test: a second entry
    point that recorded eligibility without publishing would make a type deliverable to
    a webhook and invisible on the stream that authorizes reading it.
    """
    unpublished = {
        type_
        for type_ in vocabulary.WEBHOOK_ELIGIBLE
        if not vocabulary.is_published(type_)
    }

    assert unpublished == set()


def test_the_eligible_set_cannot_be_widened_after_discovery() -> None:
    """It is a frozenset, so a caller holding it cannot add to what it may deliver."""
    assert isinstance(vocabulary.WEBHOOK_ELIGIBLE, frozenset)
