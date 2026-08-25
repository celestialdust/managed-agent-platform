"""The one field on an Environment that widens, and the floor that keeps it honest.

`allowed_domains` turns the sandbox's egress on. Before it existed, an agent's own
commands reached nothing at all and nothing in the platform said so -- a skill telling
the agent to install a package could not be followed and the failure looked like the
model's. So this is a real capability, and every case here exists because the safe state
and the dangerous state are one keyword apart.

**The dangerous state is not "no rules". It is rules that are not rules.** A profile
with `network.enabled = true` grants outbound access. The domain allowlist beside it
bounds that access only while the managed proxy is running, and
`experimental_network.enabled = true` is what runs it -- the list alone does not. A
document naming three domains, network on, proxy off, is a Session with the whole
internet, and it is the configuration an auditor is most likely to wave through, because
the domains are right there in front of them. `test_a_granted_network_with_no_proxy_is
_refused` is the case that matters most in this file.

The spelling rules are the other half. A name the proxy compares differently from how a
caller reads it is a grant reaching somewhere nobody asked for, so this parses rather
than validates: what comes out of `Environment` is a name the proxy reads the way it was
written, and the constructor is the only way to get one.
"""

from __future__ import annotations

import tomllib
from typing import Any
from uuid import uuid4

import pytest

from managed_agent.control.pod_config.compiler import (
    PROFILE_NAME,
    FloorViolation,
    _refuse_egress_the_proxy_does_not_bound,
    _render_requirements,
    session_profile,
)
from managed_agent.core.ids import TenantId
from managed_agent.core.registration.environment import (
    MAX_ALLOWED_DOMAINS,
    Environment,
    new_environment_id,
)

IMAGE = "registry.example/agent@sha256:" + "a" * 64
GATEWAY = "https://tool-gateway.example/mcp"


def _a_shape(*domains: str) -> Environment:
    return Environment(
        id=new_environment_id(),
        tenant_id=TenantId(uuid4()),
        name="analysis",
        runtime_image=IMAGE,
        denied_paths=(),
        allowed_domains=domains,
    )


def _document(*domains: str) -> dict[str, Any]:
    text = _render_requirements(
        session_profile(), gateway_url=GATEWAY, allowed_domains=domains
    )
    parsed: dict[str, Any] = tomllib.loads(text)
    return parsed


# --------------------------------------------------------------------------------------
# What a shape accepts
# --------------------------------------------------------------------------------------


def test_a_shape_with_no_domains_is_the_default_and_grants_nothing() -> None:
    """Saying nothing is saying no network, and that is what makes the default safe.

    Asserted rather than assumed because the alternative reading -- "unset, therefore
    unrestricted" -- is the one a field like this attracts, and it is the reading that
    would give every Session ever created the run of the internet.
    """
    assert _a_shape().allowed_domains == ()


@pytest.mark.parametrize(
    "domain",
    [
        "pypi.org",
        "files.pythonhosted.org",
        "*.pythonhosted.org",
        "api.github.com",
        "a-b.example.co.uk",
        "x1.example.com",
    ],
)
def test_a_name_the_proxy_reads_as_written_is_accepted(domain: str) -> None:
    assert _a_shape(domain).allowed_domains == (domain,)


# --------------------------------------------------------------------------------------
# What a shape refuses, and why each one is a grant and not a typo
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "because"),
    [
        ("", "an empty entry"),
        ("PyPI.org", "upper case, which the proxy lower-cases and would not match"),
        ("https://pypi.org", "a scheme, so the whole entry is not a host name"),
        ("pypi.org/simple", "a path, which the proxy does not compare"),
        ("pypi.org:443", "a port, which is not part of a host name here"),
        ("user@pypi.org", "userinfo"),
        ("pypi.org.", "a trailing dot the proxy's own names do not carry"),
        ("pypi org", "whitespace"),
        ("*", "the whole internet written as one entry"),
        ("a.*.com", "a wildcard the proxy does not expand, so it matches nothing"),
        ("localhost", "one label, so what it matches depends on the pod's resolver"),
        ("example", "one label, same reason"),
        ("10.0.0.1", "an address rather than a name"),
        ("control-plane.map-dev.svc", "this cluster's own network"),
        ("tool-gateway.map-dev.svc.cluster.local", "this cluster's own network"),
        ("something.internal", "this cluster's own network"),
        ("a..b.com", "an empty label"),
        ("-lead.example.com", "a label that starts with a hyphen"),
        ("trail-.example.com", "a label that ends with a hyphen"),
        ("under_score.example.com", "a character a domain label cannot carry"),
    ],
)
def test_a_name_that_would_grant_something_else_is_refused(
    domain: str, because: str
) -> None:
    """Every row is a grant rather than a typo, which is why each is refused and not
    normalised. Normalising `PyPI.org` would be helpful and would also mean this type
    decides what a caller meant; refusing it makes the caller decide."""
    with pytest.raises(ValueError):
        _a_shape(domain)


def test_the_cluster_suffix_refusal_is_not_the_guard_and_says_so() -> None:
    """A reader must not take this list for the protection.

    The protection is the proxy, which blocks private destinations by RESOLVED address
    -- so a public name pointing at a private IP is blocked whatever it is called, which
    is what keeps the node's instance metadata service out of reach. That is why this
    case asserts the message and not just the refusal: the refusal is here so a caller
    who reached for a cluster name learns something, and a future reader who deletes
    this list must know they are not deleting the guard.
    """
    with pytest.raises(ValueError, match="this cluster's own network"):
        _a_shape("control-plane.map-dev.svc.cluster.local")


def test_the_same_domain_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="allowed twice"):
        _a_shape("pypi.org", "pypi.org")


def test_more_domains_than_the_bound_is_refused() -> None:
    """Each entry is a rule the proxy evaluates per connection, so the list is bounded.

    At the bound rather than past it as well, because a cap tested only by exceeding it
    can be off by one in the permissive direction and nothing would notice.
    """
    at_the_bound = tuple(
        f"host{index}.example.com" for index in range(MAX_ALLOWED_DOMAINS)
    )
    assert len(_a_shape(*at_the_bound).allowed_domains) == MAX_ALLOWED_DOMAINS
    with pytest.raises(ValueError, match="at most"):
        _a_shape(*at_the_bound, "one-too-many.example.com")


# --------------------------------------------------------------------------------------
# What the compiled document says
# --------------------------------------------------------------------------------------


def test_no_domain_emits_no_network_keys_at_all() -> None:
    """Absent, not present-and-off. There is no spelling of off that would be safer than
    silence, and a key written as `false` is one edit away from `true` in a document a
    reader skims."""
    document = _document()
    assert "experimental_network" not in document
    assert "network" not in document["permissions"][PROFILE_NAME]


def test_a_granted_domain_emits_the_grant_the_proxy_and_the_ceiling() -> None:
    """Three keys and the profile's own, because any subset is worse than none.

    `enabled` runs the proxy. `allowed_domains` is what it bounds egress to.
    `managed_allowed_domains_only` makes that list a ceiling rather than a floor
    a config layer inside the pod can raise. And the profile's own `network.enabled`
    is what grants the agent's commands outbound access at all -- the managed keys
    cannot grant it, so an administrator can bound egress without turning it on.
    """
    document = _document("pypi.org", "files.pythonhosted.org")
    managed = document["experimental_network"]
    assert managed["enabled"] is True
    assert managed["managed_allowed_domains_only"] is True
    assert managed["allowed_domains"] == ["pypi.org", "files.pythonhosted.org"]
    assert document["permissions"][PROFILE_NAME]["network"]["enabled"] is True


def test_the_egress_keys_land_at_the_top_level_and_not_inside_a_table() -> None:
    """Dotted keys emitted before the first table header, which is why they parse as top
    level. A reader moving them below a header would put the platform's egress ceiling
    inside `[allowed_permission_profiles]`, where the runtime never looks -- and the
    document would still parse."""
    text = _render_requirements(
        session_profile(), gateway_url=GATEWAY, allowed_domains=("pypi.org",)
    )
    header = text.index("[allowed_permission_profiles]")
    assert text.index("experimental_network.enabled") < header


def test_the_filesystem_rules_survive_the_network_table_being_added() -> None:
    """The network table follows the filesystem table, so a rule accidentally written
    after the new header would become a network key and stop denying anything."""
    rules = _document("pypi.org")["permissions"][PROFILE_NAME]["filesystem"]
    assert rules["/session/workspace"] == "write"
    assert rules["/run/codex"] == "deny"
    assert rules["/var/lib/map/codex"] == "deny"


# --------------------------------------------------------------------------------------
# The floor refuses
# --------------------------------------------------------------------------------------


def test_the_floor_admits_both_shapes_the_compiler_emits() -> None:
    """The presence that makes the refusals below worth having."""
    _refuse_egress_the_proxy_does_not_bound(_document())
    _refuse_egress_the_proxy_does_not_bound(_document("pypi.org"))


def test_a_granted_network_with_no_proxy_is_refused() -> None:
    """**The most important case in this file.**

    A profile granting network access with the managed proxy absent is a Session with
    the whole internet. The allowlist restricts nothing without the proxy running, so
    this document reads to a human exactly like bounded egress and behaves like none.
    Written as raw dictionaries because no path through the compiler can produce it --
    which is the reason it has to be written by hand rather than left uncovered.
    """
    document = _document("pypi.org")
    del document["experimental_network"]
    with pytest.raises(FloorViolation, match="whole internet"):
        _refuse_egress_the_proxy_does_not_bound(document)


def test_an_allowlist_with_the_proxy_switched_off_is_refused() -> None:
    """The same hole one keyword smaller: the domains are named, the proxy configured,
    and `enabled` is false -- so the proxy never starts and the list bounds nothing."""
    document = _document("pypi.org")
    document["experimental_network"]["enabled"] = False
    with pytest.raises(FloorViolation, match="whole internet"):
        _refuse_egress_the_proxy_does_not_bound(document)


def test_a_grant_with_no_domains_is_refused() -> None:
    """Unbounded egress written as if it were bounded. The proxy is running and has
    nothing to compare against, which is not the same as denying everything."""
    document = _document("pypi.org")
    document["experimental_network"]["allowed_domains"] = []
    with pytest.raises(FloorViolation, match="unbounded egress"):
        _refuse_egress_the_proxy_does_not_bound(document)


def test_a_list_a_pod_layer_could_add_to_is_refused() -> None:
    """Without the managed-only flag the compiled list is a floor rather than a ceiling:
    a config layer inside the pod appends its own domains and the platform's list stops
    being the answer to where this agent can reach."""
    document = _document("pypi.org")
    del document["experimental_network"]["managed_allowed_domains_only"]
    with pytest.raises(FloorViolation, match="managed-only"):
        _refuse_egress_the_proxy_does_not_bound(document)


def test_a_proxy_configured_with_no_grant_is_refused() -> None:
    """Not dangerous, and refused anyway.

    A document with the proxy active and the profile's network absent grants nothing, so
    nobody is at risk from it. It is refused because nothing in this tree emits it, and
    admitting one direction of the pair coming apart is how the other direction -- the
    dangerous one -- eventually gets emitted without anything noticing.
    """
    document = _document()
    document["experimental_network"] = {
        "enabled": True,
        "managed_allowed_domains_only": True,
        "allowed_domains": ["pypi.org"],
    }
    with pytest.raises(FloorViolation, match="nothing in this tree emits"):
        _refuse_egress_the_proxy_does_not_bound(document)


def test_inert_egress_keys_beside_no_grant_are_refused() -> None:
    """Keys left behind by an edit that removed the grant and not the block. They do
    nothing today, and they are exactly what a later reader would take as evidence that
    egress is bounded on a Session that has none."""
    document = _document()
    document["experimental_network"] = {"allowed_domains": ["pypi.org"]}
    with pytest.raises(FloorViolation, match="present and inert"):
        _refuse_egress_the_proxy_does_not_bound(document)
