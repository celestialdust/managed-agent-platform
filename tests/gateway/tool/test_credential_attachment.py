"""A credential that will not render itself, and the two attachments it can become.

Tier 1, no infrastructure. Every assertion here is about the two properties the type is
for: a value cannot reach a log line by being formatted into one, and a credential bound
for a spawned child's environment cannot be written into a request header by an edit
that type-checks.

The redaction is checked through all four routes a value reaches text by -- `repr`,
`str`, `format` and an f-string -- because a value that hides from one and not the
others is worse than one that hides from none: it teaches a reader a rule that does not
hold.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle
from collections.abc import Callable, Mapping

import pytest
from mcp.client.stdio import get_default_environment

from managed_agent.core.errors import ErrorCode
from managed_agent.gateway.tool.credential_broker import (
    CredentialUnavailable,
    HttpAttachment,
    Secret,
    StdioAttachment,
)
from managed_agent.gateway.tool.error_map import classify

_VALUE = "s3cr3t-attachment-value"


@dataclasses.dataclass(frozen=True, slots=True)
class _HolderWithoutASecret:
    """The same shape as an attachment, minus the one field under test.

    The negative control for the walks below: without it, a `TypeError` from `asdict`
    would be evidence that `asdict` fails on this shape, not that it fails on a Secret.
    """

    credential_ref: str
    env_var: str


def test_a_secret_renders_as_a_redaction_through_every_route_to_text() -> None:
    secret = Secret(_VALUE)

    assert repr(secret) == "Secret(<redacted>)"
    assert str(secret) == "<redacted>"
    assert format(secret) == "<redacted>"
    assert f"{secret}" == "<redacted>"
    assert f"{secret!r}" == "Secret(<redacted>)"
    for rendered in (repr(secret), str(secret), format(secret), f"{secret}"):
        assert _VALUE not in rendered
        assert _VALUE[:8] not in rendered


def test_the_value_comes_back_only_from_reveal() -> None:
    assert Secret(_VALUE).reveal() == _VALUE


def test_a_secret_cannot_be_reassigned() -> None:
    secret = Secret(_VALUE)

    with pytest.raises(AttributeError):
        secret._value = "something else"
    with pytest.raises(AttributeError):
        del secret._value
    assert secret.reveal() == _VALUE


def test_no_generic_walk_of_a_holder_reaches_the_value() -> None:
    """`__repr__` and `__str__` cover formatting and nothing else, and formatting is not
    the only way a value reaches a log line. A structured logger handed an attachment
    calls `asdict` on it; a cache or a queue calls `pickle`. Each walked straight
    through a `Secret` that was a dataclass, and each produced the raw string.

    So the containment is a property of the type and not of a rule somebody follows:
    these three raise rather than answering, and the value is reachable only through
    `reveal`.
    """
    attachment = StdioAttachment("vendor/token", "MAP_TOKEN", Secret(_VALUE))

    for walk in (
        lambda: dataclasses.asdict(attachment),
        lambda: dataclasses.astuple(attachment),
        lambda: pickle.dumps(attachment),
        lambda: pickle.dumps(Secret(_VALUE)),
        lambda: copy.deepcopy(Secret(_VALUE)),
    ):
        with pytest.raises(TypeError) as raised:
            walk()
        assert _VALUE not in str(raised.value)

    plain = _HolderWithoutASecret("vendor/token", "MAP_TOKEN")
    assert dataclasses.asdict(plain) == {
        "credential_ref": "vendor/token",
        "env_var": "MAP_TOKEN",
    }, "the walks fail for some reason other than the Secret, so this proves nothing"


def test_two_secrets_over_one_value_are_not_equal_but_are_hashable() -> None:
    """Value equality would make this an oracle: a holder could recover the value by
    comparing against guesses. `eq=False` leaves `object.__hash__` in place, so a
    Secret stays usable as a key by identity, which discloses nothing about the
    value -- and a test asserting a TypeError here would be measurably wrong."""
    one, other = Secret(_VALUE), Secret(_VALUE)

    assert one != other
    assert one == one
    assert isinstance(hash(one), int)
    assert hash(one) != hash(other)


def test_a_stdio_attachment_writes_a_new_environment_and_leaves_the_base_alone() -> (
    None
):
    base = {"PATH": "/usr/bin", "HOME": "/root"}
    attached = StdioAttachment("vendor/token", "MAP_TOKEN", Secret(_VALUE))

    child = attached.into_env(base)

    assert child == {"PATH": "/usr/bin", "HOME": "/root", "MAP_TOKEN": _VALUE}
    assert base == {"PATH": "/usr/bin", "HOME": "/root"}


def test_a_stdio_attachment_refuses_to_overwrite_a_variable_already_set() -> None:
    attached = StdioAttachment("vendor/token", "MAP_TOKEN", Secret(_VALUE))

    with pytest.raises(CredentialUnavailable, match="MAP_TOKEN"):
        attached.into_env({"MAP_TOKEN": "whatever was there"})


def test_a_registration_naming_path_is_refused_before_the_child_is_spawned() -> None:
    """The case a registration can actually reach. `credential_env_var`'s pattern is
    `^[A-Z][A-Z0-9_]{0,63}$` and every name in the SDK's default environment matches
    it, so without this refusal a registration naming PATH would replace the child's
    search path with a credential and the child would fail for an unlike reason."""
    default = get_default_environment()
    assert "PATH" in default, "the SDK's default environment no longer carries PATH"

    with pytest.raises(CredentialUnavailable, match="PATH"):
        StdioAttachment("vendor/token", "PATH", Secret(_VALUE)).into_env(default)


def test_an_http_attachment_sends_the_vault_value_with_no_scheme_prepended() -> None:
    """`Authorization` wants a scheme and `X-Api-Key` does not, and guessing would
    produce a header that is wrong in a way only the far end can see. So the vault
    holds the header value exactly as it goes on the wire."""
    attached = HttpAttachment("vendor/token", "Authorization", Secret("Bearer abc123"))

    assert attached.into_headers({}) == {"Authorization": "Bearer abc123"}
    assert HttpAttachment("vendor/token", "X-Api-Key", Secret(_VALUE)).into_headers(
        {}
    ) == {"X-Api-Key": _VALUE}


def test_an_http_attachment_leaves_the_base_headers_alone() -> None:
    base = {"Accept": "application/json"}

    sent = HttpAttachment("vendor/token", "X-Api-Key", Secret(_VALUE)).into_headers(
        base
    )

    assert sent == {"Accept": "application/json", "X-Api-Key": _VALUE}
    assert base == {"Accept": "application/json"}


def test_an_http_attachment_refuses_to_overwrite_a_header_already_set() -> None:
    attached = HttpAttachment("vendor/token", "Authorization", Secret(_VALUE))

    with pytest.raises(CredentialUnavailable, match="Authorization"):
        attached.into_headers({"Authorization": "something else"})


def test_a_registration_the_platform_refuses_is_not_reported_as_a_platform_fault() -> (
    None
):
    """The refusal above is correct and its classification was not. A registration
    naming PATH is the tenant's own to fix, and it reached a Session as
    `platform.internal` -- so a page went to the platform for a line only the tenant
    can change, and every one of that tenant's tool calls carried it.

    `classify` is called here rather than asserted about: the arm this needs already
    exists for the other credential failure, and what has to be true is that this
    refusal reaches it.
    """
    default = get_default_environment()
    assert "PATH" in default, "the SDK's default environment no longer carries PATH"

    attachments: tuple[Callable[[], Mapping[str, str]], ...] = (
        lambda: StdioAttachment("token", "PATH", Secret(_VALUE)).into_env(default),
        lambda: HttpAttachment("token", "Authorization", Secret(_VALUE)).into_headers(
            {"Authorization": "already here"}
        ),
    )
    for attach in attachments:
        with pytest.raises(CredentialUnavailable) as raised:
            attach()
        assert raised.value.reason == "unattachable"
        assert classify(raised.value) is ErrorCode.TOOL_UNAVAILABLE, (
            "an attachment point the tenant registered over is reported as a fault of "
            "the platform, which pages the wrong people for a registration only the "
            "tenant can fix"
        )


def test_a_refused_attachment_names_the_point_and_never_the_value() -> None:
    """The message reaches a service log. It carries the tenant's own reference and the
    name of the attachment point, and nothing that was read out of the vault."""
    with pytest.raises(CredentialUnavailable) as raised:
        StdioAttachment("vendor/token", "PATH", Secret(_VALUE)).into_env({"PATH": "/x"})

    rendered = str(raised.value)
    assert _VALUE not in rendered and _VALUE[:8] not in rendered
    assert "vendor/token" in rendered


def test_neither_attachment_carries_the_other_transport_s_method() -> None:
    """The whole reason these are two types rather than one with a mode: handing a
    stdio attachment to the HTTP branch does not type-check rather than half working,
    and this is the runtime half of that claim."""
    assert not hasattr(
        StdioAttachment("vendor/token", "MAP_TOKEN", Secret(_VALUE)), "into_headers"
    )
    assert not hasattr(
        HttpAttachment("vendor/token", "Authorization", Secret(_VALUE)), "into_env"
    )
