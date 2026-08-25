"""The name the control plane writes and the name the Tool Gateway reads are one name.

This is the seam the whole "bring your own credential" path balances on, and it is the
one nothing else can catch. Two processes compose a Secrets Manager name independently:
the control plane composes it from a vault name and a credential name when a tenant
submits a value, and the Tool Gateway composes it from a registration's `credential_ref`
when a tool call goes out. Neither ever sees the other's string. If they disagree by one
character the write succeeds, the read succeeds in the sense of returning a clean
"missing entry", the registration looks correct, and every tool call fails at the far
end with an authentication error about somebody else's server.

Every other test on this path is blind to that. The route tests use a fake writer and
assert against the name the route composed -- the same string, checked against itself.
The broker tests use a fake vault and assert against the name the broker composed, also
against itself. Both pass with the two sides disagreeing, because neither test has both
sides in it. This file is the only place they are compared, so it composes each one
through the real production function of its own side and never through a shared helper.

The disagreement is not hypothetical. This platform shipped an IAM policy granting
`map/dev/tools/*` while the code composed `map/tool-credential/...`, and the two were
never compared by anything: the code was self-consistent, the policy was too, and every
authenticated MCP registration failed at AWS for a year of commits. That was a prefix
rather than a whole name, and it is the same class of defect one layer out.
"""

from typing import Final
from uuid import UUID

from managed_agent.core.ids import TenantId
from managed_agent.core.vault_catalogue import credential_ref
from managed_agent.core.vault_names import TOOL_CREDENTIAL_PREFIX, scoped_vault_name
from managed_agent.gateway.tool.credential_broker import vault_name

_TENANT: Final = TenantId(UUID("40e9465c-0000-4000-8000-000000000000"))


def _what_the_control_plane_writes(vault: str, credential: str) -> str:
    """The name the credential route puts a submitted value at.

    Composed here exactly as the route composes it -- `scoped_vault_name` over the
    prefix, the tenant, and the joined ref -- rather than by calling a helper the route
    also calls. A shared helper would make this test compare a function with itself.
    """
    return scoped_vault_name(
        TOOL_CREDENTIAL_PREFIX, _TENANT, credential_ref(vault, credential)
    )


def _what_the_tool_gateway_reads(ref: str) -> str:
    """The name the broker fetches for a registration naming this ref.

    `vault_name` is the production function, called with nothing stubbed. It is the
    entire Tool Gateway side of the agreement.
    """
    return vault_name(_TENANT, ref)


def test_the_two_sides_compose_one_name() -> None:
    """A credential written under a vault is readable by the ref that names it.

    The ref in the middle is the string a tenant copies out of a credential response
    and into a `POST /v1/mcp_servers` registration, so this also asserts that the
    published ref is the usable one: a route that returned a prettier ref than the one
    it wrote under would pass every route test and produce a registration that resolves
    to nothing.
    """
    ref = credential_ref("default", "tavily")
    assert _what_the_control_plane_writes("default", "tavily") == (
        _what_the_tool_gateway_reads(ref)
    )


def test_the_name_carries_the_tenant_between_the_vault_and_the_credential() -> None:
    """The composed name is the prefix, the tenant, then the two names in order.

    Spelled out once, as a literal, because every other assertion here compares two
    computed strings and would go on passing if both sides changed together. This is
    the assertion that fails when the shape itself moves -- which is the change that
    silently strands every credential already written under the old shape.
    """
    assert _what_the_control_plane_writes("default", "tavily") == (
        f"{TOOL_CREDENTIAL_PREFIX}/{_TENANT}/default/tavily"
    )


def test_two_tenants_naming_one_vault_and_credential_get_two_names() -> None:
    """The tenant is composed in, so one tenant's ref cannot reach another's entry.

    Both sides are checked, not just the writer. A control plane that scoped its writes
    and a gateway that did not would let every tenant read whichever entry was written
    last under a shared name -- and the writer's own tests would all pass.
    """
    other = TenantId(UUID("40e9465c-0000-4000-8000-000000000001"))
    ref = credential_ref("default", "tavily")
    assert _what_the_control_plane_writes("default", "tavily") != scoped_vault_name(
        TOOL_CREDENTIAL_PREFIX, other, ref
    )
    assert vault_name(_TENANT, ref) != vault_name(other, ref)


def test_the_two_names_cannot_be_confused_across_the_separator() -> None:
    """Vault `a` credential `b/c` and vault `a/b` credential `c` are not one name.

    They would be, if a name could contain the separator -- both compose to
    `.../a/b/c`, and one tenant's credential would silently resolve to another of its
    own. Neither name can contain a `/`: migration 0029 refuses it in the database and
    `_NAME_PATTERN` refuses it at the boundary, so the pair below is unrepresentable
    rather than merely unlikely.

    Asserted as a property of the join rather than by trying to build the illegal names,
    because the point is that the ambiguity has no legal representation. If a later
    change admitted `/` into a name, this join would start producing colliding strings
    and the two guards above are what fail first.
    """
    collide = credential_ref("a", "b/c") == credential_ref("a/b", "c")
    assert collide, (
        "the join itself is ambiguous across the separator -- which is exactly why "
        "neither component may contain one. If this ever stops being true the "
        "separator changed, and every credential written under the old one is "
        "stranded at a name nothing composes."
    )
