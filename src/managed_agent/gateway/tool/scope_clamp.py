"""Narrow one tool call to the Session's Scope, or refuse the call that cannot be.

A tool registration declares, per tool, which Session Scope dimension binds into which
of the tool's arguments, and a tool that declares none is refused at registration. This
is where that declaration becomes a boundary rather than a promise: on every call, the
Scope's value for a bound dimension is written into the named argument, replacing
whatever the model supplied.

**Replacing, not checking.** A disagreement between the model's argument and the Scope
is not an error, because answering it with a refusal would hand the model an oracle: it
could call with a guess, read whether the call was refused, and walk its way to the
Scope's value. Overwriting closes that, and it costs nothing the tenant wanted -- a
model that named a different repository was going to be refused either way, and now it
is simply answered about the repository the Session is for.

**A Scope that is silent on a bound dimension refuses the whole call**, including the
bindings it could have narrowed. Applying the ones it can and sending the rest wide is
the dangerous half-measure: the call looks narrowed at the point it goes out and the
one dimension nobody supplied is running across everything the credential can reach.

Nothing here reaches a network, a clock or a store, so what a call is narrowed to is a
function of the registration and the Session's creation facts alone -- both of which
are fixed before the Session's first Turn.

The clamp is the enforcement half of ADR-003, whose Status records that it was deferred
and that `tool.out_of_scope` was published with nothing raising it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from managed_agent.core.registration.scope_binding import RegisteredTool


@dataclass(frozen=True, slots=True)
class OutOfScope:
    """The call could not be narrowed, and the dimension that had no value.

    Carries names only. `dimension` is a term the tenant wrote into their own
    registration and `tool_name` is the name the caller just used, so neither tells a
    reader of the refusal anything it did not already have -- and the Scope's values
    are deliberately not here, for the reason the module docstring gives about oracles.
    """

    tool_name: str
    dimension: str


def narrow(
    tool: RegisteredTool,
    scope: Mapping[str, str],
    arguments: Mapping[str, object],
) -> dict[str, object] | OutOfScope:
    """The arguments the outbound call should carry, or why it must not be made.

    Returns a new mapping; `arguments` is left as the caller passed it, so the record
    of what the model asked for stays readable beside what the platform sent.

    Every binding is resolved before any is applied, so a tool whose Scope covers some
    dimensions and not others refuses rather than going out half-narrowed.
    """
    bound: dict[str, object] = {}
    for binding in tool.scope_bindings:
        value = scope.get(binding.dimension)
        if value is None:
            return OutOfScope(tool_name=tool.name, dimension=binding.dimension)
        bound[binding.argument] = value
    return {**arguments, **bound}
