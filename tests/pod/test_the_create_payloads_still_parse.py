"""The bodies the live tier posts still satisfy the models the routes declare.

Offline, and deliberately so. Every case under `tests/pod/` needs a Session before it
can assert anything, and getting one takes three `POST`s whose bodies were, until now,
written out by hand in each file. When a route's model gains a required field, those
bodies stop parsing -- and the way that failure is discovered is what this case exists
to change.

Discovered live, it costs a port-forward, an ECR lookup and a `400` several minutes in,
naming three fields at once with no indication of which file is stale or how many
others are. Discovered here it costs milliseconds and names the builder. The bodies and
the models are two descriptions of one contract, kept in separate trees that no
compiler relates, so the only thing that can relate them is a case that parses one with
the other.

This does not reach the cluster and must never be marked as though it does: a guard
that only runs under `MAP_CLUSTER_TESTS=1` would be absent from exactly the run that is
supposed to catch the drift before anyone pays for it.
"""

from __future__ import annotations

from uuid import uuid4

from cluster_access import (
    definition_payload,
    environment_payload,
    session_payload,
)

from managed_agent.core.registration.definition import AgentDefinition
from managed_agent.core.registration.environment import CreateEnvironment
from managed_agent.core.session.session import CreateSession


def test_the_environment_body_the_live_tier_posts_still_parses() -> None:
    parsed = CreateEnvironment.model_validate(
        environment_payload(
            "wedge-abc123", "registry.example/session-shim@sha256:" + "0" * 64
        )
    )
    assert parsed.name == "wedge-abc123"


def test_the_definition_body_the_live_tier_posts_still_parses() -> None:
    parsed = AgentDefinition.model_validate(
        definition_payload("wedge-abc123", "Run what you are asked.")
    )
    assert parsed.skills == frozenset(), (
        "the live tier asks for a definition with nothing attached, so that a case "
        "failing there is not first suspected of a skill problem"
    )


def test_the_session_body_the_live_tier_posts_still_parses() -> None:
    parsed = CreateSession.model_validate(session_payload(str(uuid4()), str(uuid4())))
    assert parsed.definition_version is None, (
        "the live tier pins no revision, so it follows whichever one a fresh "
        "registration just wrote rather than assuming that is revision 1"
    )
