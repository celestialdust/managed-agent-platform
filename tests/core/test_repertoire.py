"""The Repertoire is closed, complete, and says what it sends on the wire.

Every case here is about the contract rather than about a call: that the declared set
and the mapping cannot drift apart, that the mapping cannot be widened after import,
and that the parameter models spell the protocol's own field names. The wire spellings
matter because nothing in this repository can catch a wrong one — the Agent Runtime
would answer a `model_provider` key with an error the shim reports as the platform's.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from managed_agent.core.pod.repertoire import (
    REPERTOIRE,
    REQUIRES_EXPERIMENTAL_API,
    RepertoireMethod,
    ThreadStartRequest,
    TurnSteerRequest,
)


def test_every_declared_method_has_exactly_one_entry_and_no_entry_is_undeclared() -> (
    None
):
    assert set(REPERTOIRE) == set(RepertoireMethod)
    assert len(REPERTOIRE) == len(RepertoireMethod)
    assert all(method is entry.method for method, entry in REPERTOIRE.items())


def test_the_repertoire_cannot_be_widened_after_import() -> None:
    with pytest.raises(TypeError):
        REPERTOIRE[RepertoireMethod.INITIALIZE] = REPERTOIRE[  # type: ignore[index]
            RepertoireMethod.TURN_START
        ]


def test_thread_start_cannot_express_a_sandbox() -> None:
    """The forbidden mechanism has no field, so the illegal pair cannot be built."""
    assert "sandbox" not in ThreadStartRequest.model_fields
    with pytest.raises(ValidationError):
        ThreadStartRequest(
            cwd="/session/workspace",
            model="m",
            model_provider="p",
            permissions="map-session",
            sandbox="workspace-write",  # type: ignore[call-arg]
        )


def test_thread_start_serializes_the_protocol_spellings() -> None:
    body = ThreadStartRequest(
        cwd="/session/workspace",
        model="gpt-5",
        model_provider="map-model-gateway",
        permissions="map-session",
        base_instructions="be brief",
    ).model_dump(by_alias=True, exclude_none=True, mode="json")
    assert body == {
        "cwd": "/session/workspace",
        "model": "gpt-5",
        "modelProvider": "map-model-gateway",
        "permissions": "map-session",
        "approvalPolicy": "never",
        "baseInstructions": "be brief",
    }


def test_the_approval_policy_cannot_be_set_to_anything_else() -> None:
    """Nobody is inside a pod to approve, so the only legal value is the fixed one."""
    with pytest.raises(ValidationError):
        ThreadStartRequest(
            cwd="/session/workspace",
            model="m",
            model_provider="p",
            permissions="map-session",
            approval_policy="on-request",  # type: ignore[arg-type]
        )


def test_an_outbound_model_is_frozen() -> None:
    request = ThreadStartRequest(
        cwd="/w", model="m", model_provider="p", permissions="map-session"
    )
    with pytest.raises(ValidationError):
        request.model = "another"


def test_a_steer_cannot_be_sent_without_the_turn_it_was_written_for() -> None:
    with pytest.raises(ValidationError):
        TurnSteerRequest(thread_id="th_1", input=())  # type: ignore[call-arg]
    body = TurnSteerRequest(
        thread_id="th_1", expected_turn_id="tn_1", input=()
    ).model_dump(by_alias=True, mode="json")
    assert body["expectedTurnId"] == "tn_1"


def test_every_experimental_field_is_one_its_own_params_model_declares() -> None:
    """Otherwise the upgrade re-check list names fields nothing sends."""
    named = {
        entry.method: entry.experimental_fields
        for entry in REPERTOIRE.values()
        if entry.experimental_fields
    }
    assert named, "no entry declares an experimental field; the list would be vacuous"
    for method, fields in named.items():
        declared = set(REPERTOIRE[method].params_model.model_fields)
        assert fields <= declared, f"{method} names {fields - declared}"


def test_the_handshake_opts_in_exactly_while_a_gated_field_is_sent() -> None:
    gated = any(entry.experimental_fields for entry in REPERTOIRE.values())
    assert REQUIRES_EXPERIMENTAL_API is gated
    assert (
        "permissions" in REPERTOIRE[RepertoireMethod.THREAD_START].experimental_fields
    )


def test_the_one_notification_is_the_only_entry_without_a_response_model() -> None:
    without = {
        entry.method for entry in REPERTOIRE.values() if entry.response_model is None
    }
    assert without == {RepertoireMethod.INITIALIZED}
