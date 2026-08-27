"""How a (server, tool) pair becomes the name the model is shown, and what bounds it.

Tier 1, no infrastructure. This is the seam that lets two servers each offer a tool
called `search` while the name the Agent Runtime receives stays unique per tenant --
which is not a nicety but the precondition the whole naming scheme rests on. The runtime
appends a SHA1-derived suffix when two names it is handed would sanitize to one, and a
Grant written against the original then resolves to nothing. Tenant-unique advertised
names are what leave that suffix with nothing to disambiguate, and `scope_binding`'s
module docstring records it as the reason registered names are shaped the way they are.

So the property under test is not "the join produces a nice string". It is: **every
advertised name fits the byte budget the runtime will qualify it into, and the pair is
what is checked against that budget rather than either name alone.** Uniqueness is the
store's, for the reason one case below states in full.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from managed_agent.core.registration.advertised_name import (
    SERVER_TOOL_SEPARATOR,
    AdvertisedToolName,
    advertised_name_for,
    pair_fits_the_budget,
)
from managed_agent.core.registration.scope_binding import (
    MAX_TOOL_NAME_BYTES,
    ServerName,
    ToolName,
)


class _Advertised(BaseModel):
    """The annotated type is erased at a call site; a field is where it bites."""

    name: AdvertisedToolName


def test_two_servers_may_each_offer_the_same_tool_name() -> None:
    """The whole point of the change: the pair disambiguates, the bare name need not."""
    assert advertised_name_for(
        ServerName("github"), ToolName("search")
    ) != advertised_name_for(ServerName("slack"), ToolName("search"))


def test_the_advertised_name_carries_both_halves_in_order() -> None:
    """Stated as an equality rather than a property, because this string is a contract.

    It is what a tenant reads in a session event, what a Grant's refusal names, and what
    an operator greps for. A join nobody can predict from the pair would make every one
    of those a lookup.
    """
    assert (
        advertised_name_for(ServerName("github"), ToolName("search"))
        == f"github{SERVER_TOOL_SEPARATOR}search"
    )


def test_the_join_is_not_relied_on_to_be_injective_by_shape() -> None:
    """Where the tenant-unique guarantee actually lives, said out loud.

    `ServerName` admits `_` and `-`, so distinct pairs can produce one advertised name
    -- `ab_` + `c` and `ab` + `_c` are the shape of it. Nothing here prevents that, and
    a pattern rule that did would have to forbid characters tenants legitimately use.
    What prevents two tools *advertising* one name is the unique index the store keeps
    on `(tenant_id, advertised_name)`: the second registration is refused, by the same
    mechanism and with the same message as any other taken name.

    Written as a test so the collision is a recorded fact rather than a latent surprise.
    If this ever becomes an assertion that the join IS injective, the index is what has
    to have been made redundant first.
    """
    assert advertised_name_for(ServerName("ab_"), ToolName("c")) == advertised_name_for(
        ServerName("ab"), ToolName("_c")
    )


def test_a_pair_that_would_not_fit_the_runtime_budget_is_refused() -> None:
    """The budget is on the *pair*, not on either name alone, and that is deliberate.

    Bounding the two patterns independently would mean choosing numbers that add up in
    the worst case -- and the worst case is a 63-byte server name, which would leave a
    tool name so short that ordinary registrations start failing. Checking the sum
    instead costs a long server name nothing except shorter tool names, and refuses
    exactly the pairs that cannot be advertised.
    """
    server = ServerName("s" * 60)
    fits = ToolName("t" * (MAX_TOOL_NAME_BYTES - 60 - len(SERVER_TOOL_SEPARATOR)))
    assert pair_fits_the_budget(server, fits)

    one_too_many = ToolName(fits + "t")
    assert not pair_fits_the_budget(server, one_too_many)


def test_every_advertised_name_the_budget_admits_is_a_valid_advertised_name() -> None:
    """The type and the predicate agree, so neither can drift into permitting the other.

    A `pair_fits_the_budget` that said yes to a name `AdvertisedToolName` rejects would
    put a registration in the store that the Gateway could never advertise.
    """
    server = ServerName("s" * 60)
    tool = ToolName("t" * (MAX_TOOL_NAME_BYTES - 60 - len(SERVER_TOOL_SEPARATOR)))
    assert pair_fits_the_budget(server, tool)

    parsed = _Advertised(name=advertised_name_for(server, tool))
    assert len(parsed.name.encode()) == MAX_TOOL_NAME_BYTES


def test_an_advertised_name_past_the_budget_is_refused_by_the_type() -> None:
    with pytest.raises(ValidationError):
        _Advertised(name="a" * (MAX_TOOL_NAME_BYTES + 1))
