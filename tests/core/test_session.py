"""A Session's creation facts are parsed at the boundary and never rewritten after.

Tier 1 (local, no infrastructure). Three properties are graded. First, the
create request is parsed rather than validated: a bad budget or a bad currency is
refused at the type, so nothing downstream has to re-check it. Second, the record
is frozen in fact and not only by convention — every field is probed, because a
`frozen=True` that a later field addition escapes would be a silent hole. Third,
the record carries no state and no pod field: state is folded from the log on
every read, so a stored copy would be a second source free to disagree with it,
and the pod binding is mutable and belongs wherever placement lives.
"""

import dataclasses
from uuid import uuid4

import pytest
from pydantic import ValidationError

from managed_agent.core.ids import DefinitionId, SessionId, TenantId
from managed_agent.core.session.session import (
    CreateSession,
    SessionRecord,
    SessionState,
)


def _valid_create_payload() -> dict[str, object]:
    return {
        "definition_id": str(uuid4()),
        # Required: a Session runs in a registered sandbox shape and there is no
        # default one, so a body omitting this is refused at the boundary.
        "environment_id": str(uuid4()),
        "grant": ["fs.read"],
        "scope": {"repo": "acme/widgets"},
        "budget_minor_units": 500,
        "budget_currency": "USD",
        "retention_days": 30,
    }


def _record() -> SessionRecord:
    return SessionRecord(
        id=SessionId(uuid4()),
        tenant_id=TenantId(uuid4()),
        definition_id=DefinitionId(uuid4()),
        definition_revision="rev-1",
        grant=frozenset({"fs.read"}),
        scope=(("repo", "acme/widgets"),),
        budget_minor_units=500,
        budget_currency="USD",
        retention_days=30,
    )


def test_a_well_formed_create_request_parses() -> None:
    """The baseline the refusals below are measured against.

    Without this the refusal tests could all be passing for the wrong reason — a model
    that rejects everything refuses an unknown field too.
    """
    parsed = CreateSession.model_validate(_valid_create_payload())
    assert parsed.grant == frozenset({"fs.read"})
    assert parsed.budget_minor_units == 500


def test_create_refuses_a_field_it_does_not_publish() -> None:
    """An unknown field is a caller misunderstanding, and absorbing it hides the bug.

    Ignoring it would let a caller believe it set something — a budget under a misspelt
    name, say — while the platform ran with the default.
    """
    payload = _valid_create_payload() | {"pod": "pod-7"}
    with pytest.raises(ValidationError) as refusal:
        CreateSession.model_validate(payload)
    assert "pod" in str(refusal.value)


def test_create_refuses_a_budget_of_zero() -> None:
    payload = _valid_create_payload() | {"budget_minor_units": 0}
    with pytest.raises(ValidationError):
        CreateSession.model_validate(payload)


def test_create_refuses_a_currency_that_is_not_three_letters() -> None:
    for currency in ("US", "USDD"):
        payload = _valid_create_payload() | {"budget_currency": currency}
        with pytest.raises(ValidationError):
            CreateSession.model_validate(payload)


def test_create_refuses_a_retention_period_of_zero_days() -> None:
    payload = _valid_create_payload() | {"retention_days": 0}
    with pytest.raises(ValidationError):
        CreateSession.model_validate(payload)


def test_every_field_of_the_record_refuses_assignment() -> None:
    """Probed field by field, so a field added later cannot escape the freeze."""
    record = _record()
    names = [field.name for field in dataclasses.fields(record)]
    assert names, "read no fields at all — the record is probably not a dataclass"
    for name in names:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, name, getattr(record, name))


def test_the_record_holds_no_state_and_no_pod() -> None:
    """State is folded from the log and the pod binding is mutable; neither is a
    fact fixed at creation, so a field for either would be a second source."""
    names = {field.name for field in dataclasses.fields(SessionRecord)}
    assert "state" not in names
    assert "pod" not in names
    assert not any("pod" in name for name in names)


def test_only_a_running_session_accepts_a_turn() -> None:
    """One answer to "may this Session take a Turn now", so callers cannot differ."""
    accepting = {state for state in SessionState if state.accepts_a_turn()}
    assert accepting == {SessionState.RUNNING}
